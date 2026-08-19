#!/usr/bin/env python3
"""Prajna-2B Phase 1 training: wire in ReflectiveLoop + error-correction training.

Safe run: seeds from the proven dpo_final.pt foundation, trains on error-correction
data (base-model mistakes paired with correct answers), saves to sft_v2_*/dpo_v2_*
prefixes so production checkpoints are never touched.

Stages:
  A. SFT on (prompt, correct_response)        -> learns to produce correct answers
  B. DPO on (chosen=correct, rejected=base_wrong) -> prefers correct over base's errors
  C. Contrastive on (prompt, base_wrong, correct)  -> reflection learns right vs wrong

All on CPU (MPS unusable for the wrapped model). See RESULTS for honest metrics.
"""
import os, json, time, glob, sys, gc, traceback
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer, get_crn_state_dict

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

# ---- Config ----
CKPT_DIR = './prajna/checkpoints'
DATA_DIR = './prajna/data'
STATE_FILE = f'{CKPT_DIR}/state_v2.json'
SEED_CKPT = os.environ.get('SEED_CKPT', f'{CKPT_DIR}/dpo_final.pt')   # proven foundation (read-only)
EC_DATA = os.environ.get('EC_DATA', f'{DATA_DIR}/error_correction_pairs_v2.json')  # base mistakes + correct (+ paraphrase variants)

SFT_STEPS = int(os.environ.get('SFT_V2_STEPS', '2000'))
DPO_STEPS = int(os.environ.get('DPO_V2_STEPS', '500'))
CON_STEPS = int(os.environ.get('CON_V2_STEPS', '200'))
SFT_LR = 3e-4
DPO_LR = 5e-6
CON_LR = 5e-6
DPO_BETA = 0.1
CON_MARGIN = 0.2
# BATCH=1: verified to work (the collate yields a valid (1, ML) 2D batch). Raise
# for more throughput on a bigger GPU.
BATCH = 1
GRAD_ACCUM = int(os.environ.get('GRAD_ACCUM', '8'))
MAX_GRAD_NORM = 1.0
SAVE_EVERY = int(os.environ.get('SAVE_EVERY', '50'))
LOG_EVERY = 10
MAX_LENGTH = int(os.environ.get('CRN_MAXLEN', '96'))
DEVICE = os.environ.get('CRN_DEVICE', 'cpu')
INJECT_EVERY = 4
CRN_MIX_INIT = 2.0
# Surface-invariance consistency: every CONSISTENCY_EVERY SFT steps, also train
# on a paraphrase-variant of the same prompt with a KL tie at the answer start.
CONSISTENCY_EVERY = int(os.environ.get('CONSISTENCY_EVERY', '0'))
CONSISTENCY_WEIGHT = float(os.environ.get('CONSISTENCY_WEIGHT', '0.1'))
ANCHOR_WEIGHT = float(os.environ.get('ANCHOR_WEIGHT', '4.0'))
# Collapse watchdog: if the reflective corrections have collapsed to noise by
# this step, abort before sinking thousands of steps into a dead channel.
WATCHDOG_STEP = int(os.environ.get('WATCHDOG_STEP', '200'))
WATCHDOG_MIN_STD = float(os.environ.get('WATCHDOG_MIN_STD', '0.7'))

def adapter_grad_norm(student):
    g = student.logit_fusion.up.weight.grad
    return g.norm().item() if g is not None else None

def watch_corrections(student, step, avg_loss=0.0):
    """Abort only if NOTHING is learning. The correction channel can collapse to
    noise (v6 seed has std~0.01) while the logit-fusion adapter learns — that is
    the new dense-signal channel. Abort iff corrections are dead AND the adapter
    receives no meaningful gradients (accumulated grad norm < 1e-4) AND the loss
    is NOT already near-converged (a converged model legitimately has dry
    gradients)."""
    std = student.reflection.correction_directions.detach().std().item()
    conf = student.reflection.confidence_scale.detach().abs().item()
    gate = torch.sigmoid(student.logit_fusion.gate).item() if getattr(student, 'logit_fusion', None) is not None else float('nan')
    up_std = student.logit_fusion.up.weight.detach().std().item() if getattr(student, 'logit_fusion', None) is not None else float('nan')
    up_grad = adapter_grad_norm(student) if getattr(student, 'logit_fusion', None) is not None else None
    if step == WATCHDOG_STEP:
        if std < WATCHDOG_MIN_STD and up_grad is not None and up_grad < 1e-4 and avg_loss > 0.2:
            print(f'  [WATCHDOG] step {step}: corrections std={std:.4f} AND adapter grad norm={up_grad:.2e} '
                  f'(dead) AND loss={avg_loss:.4f} (not converged) — nothing is learning. '
                  f'Aborting run (change lr/signal, not steps).', flush=True)
            sys.exit(1)
        if std < WATCHDOG_MIN_STD:
            print(f'  [WATCHDOG] step {step}: corrections collapsed (std={std:.4f}) but adapter '
                  f'(grad={up_grad}) or loss ({avg_loss:.4f}) shows learning — proceeding.', flush=True)
    return std, conf, gate, up_std, up_grad

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'sft_complete': False, 'dpo_complete': False, 'con_complete': False,
            'sft_step': 0, 'dpo_step': 0, 'con_step': 0}
def save_state(s):
    with open(STATE_FILE, 'w') as f: json.dump(s, f, indent=2)

def step_of(f, prefix=''):
    try: return int(f.split(f'/{prefix}_')[-1].split('.pt')[0])
    except: return -1

def find_latest(prefix):
    fs = glob.glob(f'{CKPT_DIR}/{prefix}_*.pt')
    if not fs: return None
    return sorted(fs, key=lambda f: step_of(f, prefix))[-1]

def recent_avg(losses, n=None):
    n = n or LOG_EVERY
    if not losses:
        return 0.0
    return sum(losses[-n:]) / len(losses[-n:])

class Student(PrajnaStudentMultiLayer):
    def forward_consistent(self, ids, labels, var_ids, var_labels, pre_len, var_pre_len,
                           pos_weights=None, var_pos_weights=None, eos_weight=1.0, cons_weight=0.1):
        """SFT forward with surface-invariance: CE on the main row + CE on a
        paraphrase-variant row of the same prompt + KL between the two answer-
        start logit distributions (forces the adapter to ignore surface form)."""
        main = self(ids, labels, eos_weight=eos_weight, pos_weights=pos_weights)
        var = self(var_ids, var_labels, eos_weight=eos_weight, pos_weights=var_pos_weights)
        p1 = F.log_softmax(main['logits'][0, pre_len - 1, :], dim=-1)
        p2 = F.softmax(var['logits'][0, var_pre_len - 1, :], dim=-1)
        kl = F.kl_div(p1, p2, reduction='batchmean').clamp(max=50.0)
        return {'loss': (main['loss'] + var['loss'] + cons_weight * kl) / 2.0,
                'kl': kl.item()}
    def forward_dpo(self, chosen_ids, rejected_ids):
        logps_c = self._dpo_logps(chosen_ids)
        logps_r = self._dpo_logps(rejected_ids)
        loss = -F.logsigmoid(DPO_BETA * (logps_c - logps_r)).mean()
        return {'loss': loss, 'chosen_reward': logps_c.mean().item(),
                'rejected_reward': logps_r.mean().item()}
    def _dpo_logps(self, input_ids):
        outputs = self._collect_hidden(input_ids)
        logits, _ = self._apply_crn(outputs, training=True)
        return self._get_batch_logps(logits, input_ids)
    def _get_batch_logps(self, logits, labels):
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1]
        mask = labels != -100
        labels[~mask] = 0
        log_probs = F.log_softmax(logits, dim=-1)
        tok_logp = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
        return (tok_logp * mask).sum(dim=-1)
    def forward_contrastive(self, prompt_ids, wrong_ids, correct_ids):
        """Reflection learns to distinguish correct from wrong next-token dist.
        Reward = logp of correct - logp of wrong, on the correction target."""
        # CRN logits on the correct answer
        oc = self._collect_hidden(correct_ids)
        lc, _ = self._apply_crn(oc, training=True)
        lp_c = self._get_batch_logps(lc, correct_ids)
        # CRN logits on the base's wrong answer
        ow = self._collect_hidden(wrong_ids)
        lw, _ = self._apply_crn(ow, training=True)
        lp_w = self._get_batch_logps(lw, wrong_ids)
        loss = -F.logsigmoid(DPO_BETA * (lp_c - lp_w)).mean()
        return {'loss': loss}

class ECDataset(Dataset):
    """Error-correction pairs: prompt + chosen (correct) + rejected (base wrong)."""
    def __init__(self, path, tok, ml):
        with open(path) as f: self.p = json.load(f)
        self.tok, self.ml = tok, ml
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        s = self.p[i]
        c = self.tok(s['chosen'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        r = self.tok(s['rejected'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        return {'chosen_ids': c['input_ids'].squeeze(), 'rejected_ids': r['input_ids'].squeeze()}

class ECSFTDataset(Dataset):
    """Error-correction pairs for SFT. Rows are grouped into aid-blocks of 5
    (orig + 4 paraphrase variants of the same prompt). Each item returns the
    main row plus ONE sibling variant row of the same aid, so the trainer can
    enforce surface-invariance (consistency KL at the answer start)."""
    def __init__(self, path, tok, ml):
        with open(path) as f: self.p = json.load(f)
        self.tok, self.ml = tok, ml
        self.prompt_toks = {}
    def __len__(self): return len(self.p)
    def _encode(self, s):
        text = f"{s['prompt']}: {s['chosen']}{self.tok.eos_token}"
        enc = self.tok(text, truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].squeeze()
        labels = ids.clone()
        labels[enc['attention_mask'].squeeze() == 0] = -100
        pre_ids = self.tok(f"{s['prompt']}: ", truncation=True, max_length=self.ml,
                           return_tensors='pt')['input_ids'].squeeze()
        labels[:pre_ids.shape[0]] = -100
        # ANCHOR: unmask the answer-start position (the last prompt token). At
        # inference this is the first decoded position, but SFT masked it, so the
        # model's argmax there was garbage ('4' instead of the answer start).
        # Gold for position pre_len-1 = the first answer token. Weight it HIGH
        # (4x) so the boundary is not absorbed into the answer-token average.
        pre_len = pre_ids.shape[0]
        weights = torch.ones(self.ml, dtype=torch.float32) * 0.25   # padding weight
        weights[enc['attention_mask'].squeeze() == 1] = 1.0
        if pre_len < ids.shape[0]:
            labels[pre_len - 1] = ids[pre_len]
            weights[pre_len - 1] = ANCHOR_WEIGHT
            if pre_len >= 3:
                weights[pre_len - 2] = 0.5
                weights[pre_len - 3] = 0.5
        else:
            weights[pre_len - 1] = ANCHOR_WEIGHT
        return ids, labels, pre_len, weights

    def __getitem__(self, i):
        s = self.p[i]
        ids, labels, pre_len, weights = self._encode(s)
        j = (i // 5) * 5 + ((i + 1) % 5)          # sibling variant, same aid block
        v = self.p[j]
        v_ids, v_labels, v_pre_len, v_weights = self._encode(v)
        return {'input_ids': ids, 'labels': labels, 'pre_len': pre_len, 'weights': weights,
                'var_ids': v_ids, 'var_labels': v_labels, 'var_pre_len': v_pre_len, 'var_weights': v_weights}

class ECContrastiveDataset(Dataset):
    def __init__(self, path, tok, ml):
        with open(path) as f: self.p = json.load(f)
        self.tok, self.ml = tok, ml
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        s = self.p[i]
        c = self.tok(s['chosen'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        r = self.tok(s['rejected'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        p = self.tok(s['prompt'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        return {'prompt_ids': p['input_ids'].squeeze(),
                'correct_ids': c['input_ids'].squeeze(),
                'wrong_ids': r['input_ids'].squeeze()}

state = load_state()
print(f"Device={DEVICE} MAX_LENGTH={MAX_LENGTH} SFT={SFT_STEPS} DPO={DPO_STEPS} CON={CON_STEPS}")
print(f"State: {state}")
print("NOTE: answer-only SFT loss (prompt masked), wd=0, conf_scale=3.0, eos_weight=5.0")

# =================== Build model ===================
student = Student(device='cpu', inject_every=INJECT_EVERY, max_length=MAX_LENGTH, crn_mix_init=CRN_MIX_INIT)
student = student.to(DEVICE)
if DEVICE == 'mps':
    gc.collect()
    torch.mps.empty_cache()  # release retained pool blocks (peak load footprint)

def resolve_resume():
    """Pick the most advanced checkpoint so a restart continues from REAL weights.

    Chronological scoring: sft_v2_N < dpo_v2_N < con_v2_N (numbered saves during
    each stage). dpo_v2_final.pt is the post-all-stages final output and is only
    used when no numbered save exists; likewise sft_v2_final.pt. SEED_CKPT last.
    """
    cands = []
    for prefix, rank in (('sft_v2', 1), ('dpo_v2', 2), ('con_v2', 3)):
        for f in glob.glob(f'{CKPT_DIR}/{prefix}_*.pt'):
            s = step_of(f)
            if s >= 0:
                cands.append((rank, s, f))
    if cands:
        _, _, p = max(cands)
        return p, os.path.basename(p)
    for fn in ('dpo_v2_final.pt', 'sft_v2_final.pt'):
        p = f'{CKPT_DIR}/{fn}'
        if os.path.exists(p):
            return p, os.path.basename(p)
    return SEED_CKPT, os.path.basename(SEED_CKPT)

RESUME_CKPT, RESUME_LABEL = resolve_resume()
ckpt = torch.load(RESUME_CKPT, map_location=DEVICE, weights_only=False)
# Handle architecture changes (e.g. inject_every, crn_mix_init, fusion rank)
# that change param sizes: drop every key whose shape mismatches the model.
crn_state = ckpt['crn']
dropped = []
for key in list(crn_state.keys()):
    if key not in student.state_dict() or student.state_dict()[key].shape != crn_state[key].shape:
        dropped.append(key)
        del crn_state[key]
for key in dropped:
    print(f"  Dropped shape-mismatched ckpt key: {key}")
student.load_state_dict(crn_state, strict=False)
# Repair: if the seed's reflection channel collapsed to noise (v6 final has
# direction std ~0.01 and confidence_scale ~0.09), re-init it under the new
# logit-fusion signal so the pillar gets a fresh start. Disable with
# REPAIR_REFLECTION=0 when the reflection collapse is known-harmless.
if os.environ.get('REPAIR_REFLECTION', '1') == '1' and student.reflection.correction_directions.detach().std().item() < 0.1:
    print("  Repairing collapsed reflection: re-init correction_directions (N(0,0.5)) + confidence_scale (3.0)")
    nn.init.normal_(student.reflection.correction_directions, std=0.5)
    student.reflection.confidence_scale.data.fill_(3.0)
if 'memory_file' in ckpt and os.path.exists(ckpt['memory_file']):
    student.load_memory(ckpt['memory_file'])
print(f"Resumed from {RESUME_CKPT} (label={RESUME_LABEL}, step {ckpt.get('step')})")

# =================== STAGE A: SFT on corrections ===================
if not state.get('sft_complete', False):
    print('=' * 60); print('STAGE A: SFT on error-correction (correct answers)'); print('=' * 60)
    ds = ECSFTDataset(EC_DATA, student.tok, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=SFT_LR, weight_decay=0.0)  # decay crushed correction_directions 10x
    remaining = max(SFT_STEPS - state.get('sft_step', 0), 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining, eta_min=SFT_LR*0.1)
    for _ in range(state.get('sft_step', 0)): sched.step()
    student.train()
    losses, step = [], state.get('sft_step', 0)
    print(f"Starting SFT {step}/{SFT_STEPS} | data={len(ds)}")
    while step < SFT_STEPS:
        for batch in loader:
            if step >= SFT_STEPS: break
            try:
                ids = batch['input_ids'].to(DEVICE); labels = batch['labels'].to(DEVICE)
                if CONSISTENCY_EVERY > 0 and step % CONSISTENCY_EVERY == 0:
                    out = student.forward_consistent(
                        ids, labels,
                        batch['var_ids'].to(DEVICE), batch['var_labels'].to(DEVICE),
                        batch['pre_len'].item(), batch['var_pre_len'].item(),
                        pos_weights=batch['weights'].to(DEVICE), var_pos_weights=batch['var_weights'].to(DEVICE),
                        eos_weight=5.0, cons_weight=CONSISTENCY_WEIGHT)
                else:
                    out = student(ids, labels, eos_weight=5.0, pos_weights=batch['weights'].to(DEVICE))
                loss = out['loss'] / GRAD_ACCUM
                if not torch.isfinite(loss):
                    sched.step(); step += 1; continue
                loss.backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    gn = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                    if torch.isfinite(gn): opt.step()
                    opt.zero_grad(); sched.step()
                losses.append(loss.item() * GRAD_ACCUM); step += 1
                state['sft_step'] = step
                if step % 20 == 0:
                    gc.collect()
                    if DEVICE == 'mps': torch.mps.empty_cache()  # cap allocator pool growth
                if step % LOG_EVERY == 0:
                    avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
                    dstd, conf, lg, up_std, up_grad = watch_corrections(student, step, avg)
                    kl = out.get('kl')
                    print(f"  SFT {step}/{SFT_STEPS} loss={avg:.4f} ref_gate={torch.sigmoid(student.reflection_gate).mean().item():.3f} "
                          f"dir_std={dstd:.3f} conf={conf:.3f} fus_gate={lg:.3f} up_grad={up_grad if up_grad is None else f'{up_grad:.2e}'}"
                          f"{f' kl={kl:.3f}' if kl is not None else ''}")
                    sys.stdout.flush()
                if step % SAVE_EVERY == 0:
                    torch.save({'crn': get_crn_state_dict(student), 'step': step, 'loss': recent_avg(losses),
                                'reflection_gate': student.reflection_gate.detach().cpu()},
                               f'{CKPT_DIR}/sft_v2_{step}.pt')
                    save_state(state)  # precise resume if training breaks
            except Exception as e:
                print(f"  [SFT err] {e}"); traceback.print_exc(); step += 1
    state['sft_complete'] = True
    save_state(state)
    torch.save({'crn': get_crn_state_dict(student), 'step': SFT_STEPS, 'loss': recent_avg(losses),
                'reflection_gate': student.reflection_gate.detach().cpu()},
               f'{CKPT_DIR}/sft_v2_final.pt')
    print("STAGE A done -> sft_v2_final.pt")

# =================== STAGE B: DPO on corrections ===================
if not state.get('dpo_complete', False):
    print('=' * 60); print('STAGE B: DPO (chosen=correct, rejected=base_wrong)'); print('=' * 60)
    ds = ECDataset(EC_DATA, student.tok, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=DPO_LR, weight_decay=0.0)
    student.train()
    losses, step = [], state.get('dpo_step', 0)
    print(f"Starting DPO {step}/{DPO_STEPS} | data={len(ds)}")
    while step < DPO_STEPS:
        for batch in loader:
            if step >= DPO_STEPS: break
            try:
                c = batch['chosen_ids'].to(DEVICE); r = batch['rejected_ids'].to(DEVICE)
                out = student.forward_dpo(c, r)
                loss = out['loss'] / GRAD_ACCUM
                if not torch.isfinite(loss):
                    step += 1; continue
                loss.backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    gn = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                    if torch.isfinite(gn): opt.step()
                    opt.zero_grad()
                losses.append(loss.item() * GRAD_ACCUM); step += 1
                state['dpo_step'] = step
                if step % LOG_EVERY == 0:
                    avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
                    print(f"  DPO {step}/{DPO_STEPS} loss={avg:.4f} C={out['chosen_reward']:.2f} R={out['rejected_reward']:.2f} ref_gate={torch.sigmoid(student.reflection_gate).mean().item():.3f}")
                    sys.stdout.flush()
                if step % SAVE_EVERY == 0:
                    torch.save({'crn': get_crn_state_dict(student), 'step': step, 'loss': recent_avg(losses),
                                'reflection_gate': student.reflection_gate.detach().cpu()},
                               f'{CKPT_DIR}/dpo_v2_{step}.pt')
                    save_state(state)  # precise resume if training breaks
            except Exception as e:
                print(f"  [DPO err] {e}"); step += 1
    state['dpo_complete'] = True
    save_state(state)
    torch.save({'crn': get_crn_state_dict(student), 'step': DPO_STEPS, 'loss': recent_avg(losses),
                'reflection_gate': student.reflection_gate.detach().cpu()},
               f'{CKPT_DIR}/dpo_v2_final.pt')
    print("STAGE B done -> dpo_v2_final.pt")

# =================== STAGE C: Contrastive (reflection right vs wrong) ===================
if not state.get('con_complete', False):
    print('=' * 60); print('STAGE C: Contrastive (reflection: correct vs base_wrong)'); print('=' * 60)
    ds = ECContrastiveDataset(EC_DATA, student.tok, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=CON_LR, weight_decay=0.0)
    student.train()
    losses, step = [], state.get('con_step', 0)
    print(f"Starting CON {step}/{CON_STEPS} | data={len(ds)}")
    while step < CON_STEPS:
        for batch in loader:
            if step >= CON_STEPS: break
            try:
                p = batch['prompt_ids'].to(DEVICE)
                c = batch['correct_ids'].to(DEVICE); r = batch['wrong_ids'].to(DEVICE)
                out = student.forward_contrastive(p, r, c)
                loss = out['loss'] / GRAD_ACCUM
                if not torch.isfinite(loss):
                    step += 1; continue
                loss.backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    gn = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                    if torch.isfinite(gn): opt.step()
                    opt.zero_grad()
                losses.append(loss.item() * GRAD_ACCUM); step += 1
                state['con_step'] = step
                if step % LOG_EVERY == 0:
                    avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
                    print(f"  CON {step}/{CON_STEPS} loss={avg:.4f} ref_gate={torch.sigmoid(student.reflection_gate).mean().item():.3f}")
                    sys.stdout.flush()
                if step % SAVE_EVERY == 0:
                    torch.save({'crn': get_crn_state_dict(student), 'step': step, 'loss': recent_avg(losses),
                                'reflection_gate': student.reflection_gate.detach().cpu()},
                               f'{CKPT_DIR}/con_v2_{step}.pt')
                    save_state(state)  # precise resume if training breaks
            except Exception as e:
                print(f"  [CON err] {e}"); step += 1
    state['con_complete'] = True
    save_state(state)

torch.save({'crn': get_crn_state_dict(student), 'step': DPO_STEPS,
            'reflection_gate': student.reflection_gate.detach().cpu(),
            'memory_file': f'{CKPT_DIR}/memory_v2_final.json'},
           f'{CKPT_DIR}/dpo_v2_final.pt')
student.save_memory(f'{CKPT_DIR}/memory_v2_final.json')
print("\nALL STAGES DONE -> dpo_v2_final.pt + memory_v2_final.json")

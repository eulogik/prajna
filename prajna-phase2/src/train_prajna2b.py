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
SEED_CKPT = f'{CKPT_DIR}/dpo_final.pt'                 # proven foundation (read-only)
EC_DATA = f'{DATA_DIR}/error_correction_pairs.json'    # base mistakes + correct

SFT_STEPS = int(os.environ.get('SFT_V2_STEPS', '2000'))
DPO_STEPS = int(os.environ.get('DPO_V2_STEPS', '500'))
CON_STEPS = int(os.environ.get('CON_V2_STEPS', '200'))
SFT_LR = 3e-4
DPO_LR = 5e-6
CON_LR = 5e-6
DPO_BETA = 0.1
CON_MARGIN = 0.2
BATCH = 1
GRAD_ACCUM = 8
MAX_GRAD_NORM = 1.0
SAVE_EVERY = 50
LOG_EVERY = 10
MAX_LENGTH = int(os.environ.get('CRN_MAXLEN', '96'))
DEVICE = os.environ.get('CRN_DEVICE', 'cpu')
INJECT_EVERY = 8
CRN_MIX_INIT = 0.05

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'sft_complete': False, 'dpo_complete': False, 'con_complete': False,
            'sft_step': 0, 'dpo_step': 0, 'con_step': 0}
def save_state(s):
    with open(STATE_FILE, 'w') as f: json.dump(s, f, indent=2)

def find_latest(prefix):
    fs = glob.glob(f'{CKPT_DIR}/{prefix}_*.pt')
    if not fs: return None
    def step_of(f):
        try: return int(f.split(f'/{prefix}_')[-1].split('.pt')[0])
        except: return -1
    return sorted(fs, key=step_of)[-1]

class Student(PrajnaStudentMultiLayer):
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
    def __init__(self, path, tok, ml):
        with open(path) as f: self.p = json.load(f)
        self.tok, self.ml = tok, ml
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        s = self.p[i]
        text = f"{s['prompt']}\n\n{s['chosen']}{self.tok.eos_token}"
        enc = self.tok(text, truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].squeeze()
        labels = ids.clone()
        labels[enc['attention_mask'].squeeze() == 0] = -100
        return {'input_ids': ids, 'labels': labels}

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
print("NOTE: ReflectiveLoop is now wired into _apply_crn with reflection_gate (init 0.15).")

# =================== Build model ===================
student = Student(device='cpu', inject_every=INJECT_EVERY, max_length=MAX_LENGTH, crn_mix_init=CRN_MIX_INIT)
student = student.to(DEVICE)
ckpt = torch.load(SEED_CKPT, map_location=DEVICE, weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
if 'memory_file' in ckpt and os.path.exists(ckpt['memory_file']):
    student.load_memory(ckpt['memory_file'])
print(f"Seeded from {SEED_CKPT} (step {ckpt.get('step')})")

# =================== STAGE A: SFT on corrections ===================
if not state.get('sft_complete', False):
    print('=' * 60); print('STAGE A: SFT on error-correction (correct answers)'); print('=' * 60)
    ds = ECSFTDataset(EC_DATA, student.tok, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=SFT_LR, weight_decay=0.01)
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
                out = student(ids, labels)
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
                if step % 20 == 0: gc.collect()
                if step % LOG_EVERY == 0:
                    avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
                    print(f"  SFT {step}/{SFT_STEPS} loss={avg:.4f} ref_gate={torch.sigmoid(student.reflection_gate).mean().item():.3f}")
                    sys.stdout.flush()
                if step % SAVE_EVERY == 0:
                    torch.save({'crn': get_crn_state_dict(student), 'step': step, 'loss': avg,
                                'reflection_gate': student.reflection_gate.detach().cpu()},
                               f'{CKPT_DIR}/sft_v2_{step}.pt')
            except Exception as e:
                print(f"  [SFT err] {e}"); traceback.print_exc(); step += 1
    state['sft_complete'] = True
    save_state(state)
    torch.save({'crn': get_crn_state_dict(student), 'step': SFT_STEPS, 'loss': avg,
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
                    torch.save({'crn': get_crn_state_dict(student), 'step': step, 'loss': avg,
                                'reflection_gate': student.reflection_gate.detach().cpu()},
                               f'{CKPT_DIR}/dpo_v2_{step}.pt')
            except Exception as e:
                print(f"  [DPO err] {e}"); step += 1
    state['dpo_complete'] = True
    save_state(state)
    torch.save({'crn': get_crn_state_dict(student), 'step': DPO_STEPS, 'loss': avg,
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

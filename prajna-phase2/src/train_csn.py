#!/usr/bin/env python3
"""Safe SFT -> DPO run that ADDS factual + common-sense capability on top of the
proven 18x foundation (sft_final.pt / dpo_final.pt), WITHOUT overwriting the
production checkpoints.

- SFT continues from sft_final.pt on the enriched corpus (teacher + IGR + facts).
- DPO then refines on enriched preference pairs (facts + IGR + clean/bloated).
- All outputs use the `sft_csn_` / `dpo_csn_` prefixes + state_csn.json, so the
  protected dpo_final.pt / sft_final.pt are never touched.

Usage (pilot):  CRN_DEVICE=mps SFT_CSN_STEPS=200 DPO_CSN_STEPS=100 python3 train_csn.py
Usage (full):   CRN_DEVICE=mps python3 train_csn.py
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
STATE_FILE = f'{CKPT_DIR}/state_csn.json'
SFT_DATA = f'{DATA_DIR}/teacher_csn.json'
DPO_DATA = f'{DATA_DIR}/dpo_csn_pairs.json'
SFT_SEED = f'{CKPT_DIR}/sft_final.pt'          # proven foundation (read-only)

SFT_STEPS = int(os.environ.get('SFT_CSN_STEPS', '1000'))
DPO_STEPS = int(os.environ.get('DPO_CSN_STEPS', '500'))
SFT_LR = 3e-4
DPO_LR = 5e-6
DPO_BETA = 0.1
BATCH = 1
GRAD_ACCUM = 8
MAX_GRAD_NORM = 1.0
SAVE_EVERY = 50
LOG_EVERY = 10
MAX_LENGTH = int(os.environ.get('CRN_MAXLEN', '96'))
DEVICE = os.environ.get('CRN_DEVICE', 'mps')
INJECT_EVERY = 8
CRN_MIX_INIT = 0.05

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'sft_complete': False, 'dpo_complete': False, 'sft_step': 0, 'dpo_step': 0}
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

class SFTDataset(Dataset):
    def __init__(self, path, tok, ml):
        with open(path) as f: self.s = json.load(f)
        self.tok, self.ml = tok, ml
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        text = f"{s.get('prompt','')}\n\n{s.get('response','')}{self.tok.eos_token}"
        enc = self.tok(text, truncation=True, max_length=self.ml,
                       padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].squeeze()
        labels = ids.clone()
        labels[enc['attention_mask'].squeeze() == 0] = -100
        return {'input_ids': ids, 'labels': labels}

class DPODataset(Dataset):
    def __init__(self, path, tok, ml):
        with open(path) as f: self.p = json.load(f)
        self.tok, self.ml = tok, ml
    def __len__(self): return len(self.p)
    def __getitem__(self, i):
        p = self.p[i]
        c = self.tok(p['chosen'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        r = self.tok(p['rejected'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        return {'chosen_ids': c['input_ids'].squeeze(), 'rejected_ids': r['input_ids'].squeeze()}

state = load_state()
print(f"Device={DEVICE} MAX_LENGTH={MAX_LENGTH} SFT_STEPS={SFT_STEPS} DPO_STEPS={DPO_STEPS}")
print(f"State: {state}")

# =================== SFT ===================
if not state.get('sft_complete', False):
    print('=' * 60); print('PHASE 1: SFT (continued from sft_final on enriched corpus)'); print('=' * 60)
    student = Student(device='cpu', inject_every=INJECT_EVERY, max_length=MAX_LENGTH, crn_mix_init=CRN_MIX_INIT)
    student = student.to(DEVICE)
    # seed from proven sft_final (foundation) — read only
    ckpt = torch.load(SFT_SEED, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ckpt['crn'], strict=False)
    if 'memory_file' in ckpt and os.path.exists(ckpt['memory_file']):
        student.load_memory(ckpt['memory_file'])
    start = state.get('sft_step', 0)
    resume = find_latest('sft_csn')
    if resume and start == 0:
        ckpt2 = torch.load(resume, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt2['crn'], strict=False)
        if 'memory_file' in ckpt2 and os.path.exists(ckpt2['memory_file']):
            student.load_memory(ckpt2['memory_file'])
        start = ckpt2.get('step', 0)
        print(f"Resuming SFT from {resume} step={start}")
    ds = SFTDataset(SFT_DATA, student.tok, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=SFT_LR, weight_decay=0.01)
    remaining = max(SFT_STEPS - start, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining, eta_min=SFT_LR*0.1)
    for _ in range(start): sched.step()
    student.train()
    losses, step, t0 = [], start, time.time()
    print(f"Starting SFT step {step}/{SFT_STEPS} | data={len(ds)}")
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
                    print(f"  SFT {step}/{SFT_STEPS} loss={avg:.4f}")
                    sys.stdout.flush()
                if step % SAVE_EVERY == 0:
                    mem = f'{CKPT_DIR}/memory_sft_csn_{step}.json'
                    student.save_memory(mem)
                    torch.save({'step': step, 'crn': get_crn_state_dict(student),
                                'loss': sum(losses[-50:])/max(len(losses[-50:]),1),
                                'memory_file': mem}, f'{CKPT_DIR}/sft_csn_{step}.pt')
                    save_state(state); print(f"  Saved sft_csn_{step}.pt")
            except Exception as e:
                print(f"  ERR SFT {step}: {e}"); traceback.print_exc(); gc.collect()
                if DEVICE == 'mps': torch.mps.empty_cache()
                continue
    mem = f'{CKPT_DIR}/memory_sft_csn_final.json'
    student.save_memory(mem)
    torch.save({'step': step, 'crn': get_crn_state_dict(student),
                'loss': sum(losses[-50:])/max(len(losses[-50:]),1), 'memory_file': mem},
               f'{CKPT_DIR}/sft_csn_final.pt')
    state['sft_complete'] = True; state['sft_step'] = step; save_state(state)
    print(f"SFT done: {step} steps, loss={sum(losses[-50:])/max(len(losses[-50:]),1):.4f}")
else:
    print("SFT already complete.")

# =================== DPO ===================
if not state.get('dpo_complete', False):
    print('=' * 60); print('PHASE 2: DPO (enriched facts + IGR + clean/bloated)'); print('=' * 60)
    if 'student' not in globals():
        student = Student(device='cpu', inject_every=INJECT_EVERY, max_length=MAX_LENGTH, crn_mix_init=CRN_MIX_INIT)
        student = student.to(DEVICE)
    start = state.get('dpo_step', 0)
    resume = find_latest('dpo_csn')
    if resume:
        ckpt = torch.load(resume, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt['crn'], strict=False)
        if 'memory_file' in ckpt and os.path.exists(ckpt['memory_file']):
            student.load_memory(ckpt['memory_file'])
        start = ckpt.get('step', 0); print(f"Resuming DPO from {resume} step={start}")
    elif os.path.exists(f'{CKPT_DIR}/sft_csn_final.pt'):
        ckpt = torch.load(f'{CKPT_DIR}/sft_csn_final.pt', map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt['crn'], strict=False)
        if 'memory_file' in ckpt and os.path.exists(ckpt['memory_file']):
            student.load_memory(ckpt['memory_file'])
        print("Loaded sft_csn_final for DPO")
    elif os.path.exists(SFT_SEED):
        ckpt = torch.load(SFT_SEED, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt['crn'], strict=False)
        print("Loaded sft_final (fallback) for DPO")
    ds = DPODataset(DPO_DATA, student.tok, MAX_LENGTH)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=DPO_LR, weight_decay=0.01)
    remaining = max(DPO_STEPS - start, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining, eta_min=DPO_LR*0.1)
    for _ in range(start): sched.step()
    student.train()
    losses, step, t0 = [], start, time.time()
    print(f"Starting DPO {step}/{DPO_STEPS} | data={len(ds)}")
    while step < DPO_STEPS:
        for batch in loader:
            if step >= DPO_STEPS: break
            try:
                c = batch['chosen_ids'].to(DEVICE); r = batch['rejected_ids'].to(DEVICE)
                out = student.forward_dpo(c, r)
                loss = out['loss'] / GRAD_ACCUM
                if not torch.isfinite(loss):
                    sched.step(); step += 1; continue
                loss.backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    gn = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                    if torch.isfinite(gn): opt.step()
                    opt.zero_grad(); sched.step()
                losses.append(loss.item() * GRAD_ACCUM); step += 1
                state['dpo_step'] = step
                if step % 20 == 0: gc.collect()
                if step % LOG_EVERY == 0:
                    avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
                    print(f"  DPO {step}/{DPO_STEPS} loss={avg:.4f} C={out['chosen_reward']:.2f} R={out['rejected_reward']:.2f}")
                    sys.stdout.flush()
                if step % SAVE_EVERY == 0:
                    mem = f'{CKPT_DIR}/memory_dpo_csn_{step}.json'
                    student.save_memory(mem)
                    torch.save({'step': step, 'crn': get_crn_state_dict(student),
                                'loss': sum(losses[-50:])/max(len(losses[-50:]),1),
                                'memory_file': mem}, f'{CKPT_DIR}/dpo_csn_{step}.pt')
                    save_state(state); print(f"  Saved dpo_csn_{step}.pt")
            except Exception as e:
                print(f"  ERR DPO {step}: {e}"); traceback.print_exc(); gc.collect()
                if DEVICE == 'mps': torch.mps.empty_cache()
                continue
    mem = f'{CKPT_DIR}/memory_dpo_csn_final.json'
    student.save_memory(mem)
    torch.save({'step': step, 'crn': get_crn_state_dict(student),
                'loss': sum(losses[-50:])/max(len(losses[-50:]),1), 'memory_file': mem},
               f'{CKPT_DIR}/dpo_csn_final.pt')
    state['dpo_complete'] = True; state['dpo_step'] = step; save_state(state)
    print(f"DPO done: {step} steps, loss={sum(losses[-50:])/max(len(losses[-50:]),1):.4f}")
else:
    print("DPO already complete.")
print("All done. New checkpoints: sft_csn_final.pt / dpo_csn_final.pt (dpo_final.pt untouched)")

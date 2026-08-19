#!/usr/bin/env python3
"""Full math-training run (refined recipe) — resumes from math_opt_1000.pt.

Refinements over train_mac_math_opt.py:
  * Resume-safe: loads latest math_full_*.pt if present, else math_opt_1000.pt.
    NEVER overwrites math_opt_1000.pt or production checkpoints.
  * Saves incrementally every SAVE_EVERY steps -> interruption-safe.
  * MAX_LENGTH=64 so large power results (e.g. 7^7=823543) fit the sequence.
  * crn_mix anneal: optional CRN_MIX_SCALE (applied once, on first resume).
  * Uncapped: uses all CPU cores (no thread cap).

Usage: python3 train_mac_math_full.py [TARGET_STEPS]
  e.g. train_mac_math_full.py 2500   (continue until global step 2500)
"""
import os, glob, json, time, sys
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
# Uncapped: use all cores.
import torch, torch.nn as nn
_N = str(os.cpu_count() or 10)
os.environ['OMP_NUM_THREADS'] = _N
os.environ['MKL_NUM_THREADS'] = _N
os.environ['OPENBLAS_NUM_THREADS'] = _N
os.environ['VECLIB_MAXIMUM_THREADS'] = _N

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer, get_crn_state_dict
from safety import safe_save

DEVICE = os.environ.get('CRN_DEVICE', 'mps')
MAX_LENGTH = int(os.environ.get('CRN_MAXLEN', '96'))  # shorter = less MPS float16 overflow
SAVE_EVERY = 100
TARGET_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
BATCH = int(os.environ.get('BATCH', '1'))
TIME_CAP_S = int(os.environ.get('MATH_FULL_TIME_CAP_S', str(2 * 24 * 3600)))
CRN_MIX_SCALE = float(os.environ.get('CRN_MIX_SCALE', '1.0'))
LR = float(os.environ.get('CRN_LR', '3e-4'))
CKPT_DIR = './prajna/checkpoints'
MEM_IN = os.path.join(CKPT_DIR, 'memory_dpo_final.json')
DATA = os.environ.get('CRN_DATA', './prajna/data/math_cot.json')
SEED_CKPT = os.environ.get('CRN_SEED', os.path.join(CKPT_DIR, 'math_opt_1000.pt'))

def find_latest_full():
    fs = glob.glob(os.path.join(CKPT_DIR, 'math_full_*.pt'))
    if not fs: return None
    best, bstep = None, -1
    for f in fs:
        try:
            d = torch.load(f, map_location='cpu', weights_only=False)
            s = int(d.get('step', 0))
            if s > bstep: best, bstep = f, s
        except Exception: pass
    return best

start_ckpt = find_latest_full() or SEED_CKPT
from_seed = (start_ckpt == SEED_CKPT)
print(f"Resume checkpoint: {start_ckpt}  (from seed={from_seed})")

def save_ckpt(path):
    sd = {k: v.detach().cpu() for k, v in get_crn_state_dict(student).items()}
    safe_save({'step': step, 'crn': sd, 'loss': sum(losses[-20:]) / max(len(losses[-20:]), 1)}, path)

class MathDS(Dataset):
    def __init__(self, path, tok, ml):
        with open(path) as f: self.s = json.load(f)
        self.tok, self.ml = tok, ml
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        text = f"{s['prompt']} {s['response']}{self.tok.eos_token}"
        enc = self.tok(text, truncation=True, max_length=self.ml,
                       padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].squeeze()
        labels = ids.clone()
        labels[enc['attention_mask'].squeeze() == 0] = -100
        return {'input_ids': ids, 'labels': labels}

print(f"Loading model (MAX_LENGTH={MAX_LENGTH}, all cores)...")
student = PrajnaStudentMultiLayer(device=DEVICE, inject_every=8, max_length=MAX_LENGTH,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
    num_corrections=8, mem_size=256, mem_dim=64)
ckpt = torch.load(start_ckpt, map_location=DEVICE, weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
student = student.to(DEVICE)   # move base model + crn_mix onto MPS
if os.path.exists(MEM_IN): student.load_memory(MEM_IN)
start_step = int(ckpt.get('step', 0))
if from_seed and CRN_MIX_SCALE != 1.0:
    with torch.no_grad():
        student.crn_mix.mul_(CRN_MIX_SCALE)
    print(f"Annealed crn_mix (x{CRN_MIX_SCALE}) -> {torch.sigmoid(student.crn_mix).tolist()}")
student.train()
tok = student.tok
torch.set_num_threads(int(_N))
print(f"Torch threads = {torch.get_num_threads()} (uncapped) | start_step={start_step} | target={TARGET_STEPS}")

ds = MathDS(DATA, tok, MAX_LENGTH)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=0)
print(f"Batch size = {BATCH}")
params = student.get_params()
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
crit = nn.CrossEntropyLoss(ignore_index=-100)

t0 = time.time()
losses = []
step = start_step
print(f"Starting full math run (target {TARGET_STEPS} steps / {TIME_CAP_S}s)...")
while step < TARGET_STEPS:
    for batch in loader:
        ids = batch['input_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        out = student(ids, labels)
        loss = out['loss']
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  step {step}: NaN/Inf loss — skipping"); opt.zero_grad(); continue
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)   # prevent gradient explosion (esp. on diverse general data)
        if not torch.isfinite(gn):
            print(f"  step {step}: non-finite grad (norm={gn}) — skipping"); opt.zero_grad(); continue
        opt.step(); opt.zero_grad()
        losses.append(loss.item()); step += 1
        if step % 5 == 0:
            dt = time.time() - t0
            print(f"  step {step}/{TARGET_STEPS} loss={loss.item():.4f} ({dt/60:.1f}m, {dt/max(step-start_step,1):.1f}s/step)", flush=True)
        if step % SAVE_EVERY == 0:
            save_ckpt(os.path.join(CKPT_DIR, f'math_full_{step}.pt'))
            print(f"  [checkpoint] math_full_{step}.pt", flush=True)
            if DEVICE == 'mps':
                torch.mps.empty_cache()
        if step >= TARGET_STEPS or (time.time() - t0) > TIME_CAP_S: break
    if step >= TARGET_STEPS or (time.time() - t0) > TIME_CAP_S: break

final = sum(losses[-20:]) / max(len(losses[-20:]), 1)
save_ckpt(os.path.join(CKPT_DIR, f'math_full_{step}.pt'))
print(f"Full run done: {step} steps, loss={final:.4f}, time={(time.time()-t0)/60:.1f}min")
print(f"Saved: {os.path.join(CKPT_DIR, f'math_full_{step}.pt')}")
print(f"Next: python3 prajna-phase2/src/eval_math.py {os.path.join(CKPT_DIR, f'math_full_{step}.pt')} 2")

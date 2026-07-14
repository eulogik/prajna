#!/usr/bin/env python3
"""Optimized, bounded math diagnostic — fits ~20h @ 50% CPU / 8GB RAM.

Differences from train_mac_math_test.py:
  * Threads capped to ~50% of M4 (leaves the machine usable for the user).
  * MAX_LENGTH=48 (math CoT fits in ~40 tokens; halves per-step cost vs 96).
  * BARE prompt->answer format (no "\\n\\n") so zero-shot eval matches training.
  * Time cap (default 18h) + step cap so it never overruns the window.
Saves to math_opt_{STEPS}.pt (never overwrites production checkpoints).
"""
import os
# Cap threads BEFORE importing torch so BLAS/OpenMP honor it.
_N = '5'  # ~50% of M4's 10 cores
os.environ['OMP_NUM_THREADS'] = _N
os.environ['MKL_NUM_THREADS'] = _N
os.environ['OPENBLAS_NUM_THREADS'] = _N
os.environ['NUMEXPR_NUM_THREADS'] = _N
os.environ['VECLIB_MAXIMUM_THREADS'] = _N
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

import torch, json, time, sys
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer, get_crn_state_dict
from safety import safe_save

DEVICE = 'cpu'
MAX_LENGTH = 48            # math CoT fits; ~2x faster per step than 96
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 500
TIME_CAP_S = int(os.environ.get('MATH_OPT_TIME_CAP_S', str(18 * 3600)))
LR = 3e-4
CKPT_IN = './prajna/checkpoints/dpo_final.pt'
MEM_IN = './prajna/checkpoints/memory_dpo_final.json'
DATA = './prajna/data/math_cot.json'
OUT = f'./prajna/checkpoints/math_opt_{STEPS}.pt'

class MathDS(Dataset):
    def __init__(self, path, tok, ml):
        with open(path) as f: self.s = json.load(f)
        self.tok, self.ml = tok, ml
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        # BARE format: "What is a+b? We compute..." (no separator) -> matches eval
        text = f"{s['prompt']} {s['response']}{self.tok.eos_token}"
        enc = self.tok(text, truncation=True, max_length=self.ml,
                       padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].squeeze()
        labels = ids.clone()
        labels[enc['attention_mask'].squeeze() == 0] = -100
        return {'input_ids': ids, 'labels': labels}

print(f"Loading model for {STEPS}-step OPTIMIZED math run (MAX_LENGTH={MAX_LENGTH})...")
student = PrajnaStudentMultiLayer(device=DEVICE, inject_every=8, max_length=MAX_LENGTH,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
    num_corrections=8, mem_size=256, mem_dim=64)
ckpt = torch.load(CKPT_IN, map_location=DEVICE, weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
if os.path.exists(MEM_IN): student.load_memory(MEM_IN)
student.train()
tok = student.tok
torch.set_num_threads(int(_N))
print(f"Torch threads capped to {torch.get_num_threads()} (50% of M4)")

ds = MathDS(DATA, tok, MAX_LENGTH)
loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
params = student.get_params()
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
crit = nn.CrossEntropyLoss(ignore_index=-100)

t0 = time.time()
losses = []
step = 0
print(f"Starting optimized math run (cap {STEPS} steps / {TIME_CAP_S}s)...")
while step < STEPS:
    for batch in loader:
        ids = batch['input_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        out = student(ids, labels)
        loss = out['loss']
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  step {step}: NaN/Inf loss — skipping"); opt.zero_grad(); continue
        loss.backward(); opt.step(); opt.zero_grad()
        losses.append(loss.item()); step += 1
        if step % 5 == 0:
            dt = time.time() - t0
            print(f"  step {step}/{STEPS} loss={loss.item():.4f} ({dt/60:.1f}m, {dt/step:.1f}s/step)", flush=True)
        if step >= STEPS or (time.time() - t0) > TIME_CAP_S: break
    if step >= STEPS or (time.time() - t0) > TIME_CAP_S: break

final = sum(losses[-20:]) / max(len(losses[-20:]), 1)
safe_save({'step': step, 'crn': get_crn_state_dict(student), 'loss': final}, OUT)
print(f"Optimized run done: {step} steps, loss={final:.4f}, time={(time.time()-t0)/60:.1f}min")
print(f"Saved: {OUT}")
print(f"Next: python3 prajna-phase2/src/eval_math.py {OUT} 2")

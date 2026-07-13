#!/usr/bin/env python3
"""TINY math diagnostic training: load dpo_final.pt, train 50 steps on math CoT.

Purpose: verify that math CoT data + longer sequences + EOS improves math
accuracy BEFORE committing to a full run. Not for production.
"""
import torch, os, json, time, sys
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer, get_crn_state_dict
from safety import safe_save

DEVICE = 'cpu'
MAX_LENGTH = 96          # room for CoT
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
LR = 3e-4
CKPT_IN = './prajna/checkpoints/dpo_final.pt'
MEM_IN = './prajna/checkpoints/memory_dpo_final.json'
DATA = './prajna/data/math_cot.json'
OUT = f'./prajna/checkpoints/math_test_{STEPS}.pt'

class MathDS(Dataset):
    def __init__(self, path, tok, ml):
        with open(path) as f: self.s = json.load(f)
        self.tok, self.ml = tok, ml
    def __len__(self): return len(self.s)
    def __getitem__(self, i):
        s = self.s[i]
        text = f"{s['prompt']}\n\n{s['response']}{self.tok.eos_token}"
        enc = self.tok(text, truncation=True, max_length=self.ml,
                       padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].squeeze()
        labels = ids.clone()
        labels[enc['attention_mask'].squeeze() == 0] = -100
        return {'input_ids': ids, 'labels': labels}

print(f"Loading model for {STEPS}-step math diagnostic...")
student = PrajnaStudentMultiLayer(device=DEVICE, inject_every=8, max_length=MAX_LENGTH,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
    num_corrections=8, mem_size=256, mem_dim=64)
ckpt = torch.load(CKPT_IN, map_location=DEVICE, weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
if os.path.exists(MEM_IN): student.load_memory(MEM_IN)
student.train()
tok = student.tok

ds = MathDS(DATA, tok, MAX_LENGTH)
loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
params = student.get_params()
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
crit = nn.CrossEntropyLoss(ignore_index=-100)

t0 = time.time()
losses = []
print(f"Starting {STEPS}-step math diagnostic...")
step = 0
while step < STEPS:
    for batch in loader:
        ids = batch['input_ids'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        out = student(ids, labels)
        loss = out['loss']
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  step {step}: NaN/Inf loss — skipping batch")
            opt.zero_grad()
            continue
        loss.backward()
        opt.step(); opt.zero_grad()
        losses.append(loss.item())
        step += 1
        dt = time.time() - t0
        print(f"  step {step}/{STEPS} loss={loss.item():.4f} "
              f"({dt:.1f}s elapsed, {dt/step:.1f}s/step)", flush=True)
        if step >= STEPS: break
    if step >= STEPS: break

final = sum(losses[-20:]) / max(len(losses[-20:]), 1)
safe_save({'step': STEPS, 'crn': get_crn_state_dict(student), 'loss': final}, OUT)
print(f"Diagnostic done: {STEPS} steps, loss={final:.4f}, time={(time.time()-t0)/60:.1f}min")
print(f"Saved: {OUT}")
print(f"Next: python3 prajna-phase2/src/eval_math.py {OUT}")

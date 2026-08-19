import os, time, torch, json, sys
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
DEV = 'mps'
sys.path.insert(0, 'prajna-phase2/src')
from crn_components import PrajnaStudentMultiLayer

print("Building student on", DEV, "...")
t0 = time.time()
student = PrajnaStudentMultiLayer(device=DEV, inject_every=8, max_length=64,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
    num_corrections=8, mem_size=256, mem_dim=64)
print(f"  built in {time.time()-t0:.1f}s")
# Move EVERYTHING onto MPS (base model + CRN + memory buffers), free the CPU copy
student = student.to(DEV)
print(f"  moved to {DEV}, MPS allocated: {torch.mps.current_allocated_memory()/1e9:.1f} GB")
student.train()
tok = student.tok

# one math sample, bare format
with open('prajna/data/math_cot.json') as f:
    data = json.load(f)
s = data[0]
text = f"{s['prompt']} {s['response']}{tok.eos_token}"
enc = tok(text, truncation=True, max_length=64, padding='max_length', return_tensors='pt')
ids = enc['input_ids']
labels = ids.clone()
labels[enc['attention_mask'] == 0] = -100

print("MPS allocated before fwd:", torch.mps.current_allocated_memory()/1e9, "GB")
t1 = time.time()
out = student(ids.to(DEV), labels.to(DEV))
loss = out['loss']
print(f"  forward OK, loss={loss.item():.4f}, {time.time()-t1:.1f}s")
print("MPS allocated after fwd:", torch.mps.current_allocated_memory()/1e9, "GB")

t2 = time.time()
loss.backward()
print(f"  backward OK, {time.time()-t2:.1f}s")
print("MPS allocated after bwd:", torch.mps.current_allocated_memory()/1e9, "GB")

# timing for a few steps
import torch.nn as nn
opt = torch.optim.AdamW(student.get_params(), lr=3e-4)
student.train()
steps = 0
tt = time.time()
for i in range(5):
    s = data[i]
    text = f"{s['prompt']} {s['response']}{tok.eos_token}"
    enc = tok(text, truncation=True, max_length=64, padding='max_length', return_tensors='pt')
    ids = enc['input_ids']; labels = ids.clone()
    labels[enc['attention_mask']==0] = -100
    opt.zero_grad()
    loss = student(ids.to(DEV), labels.to(DEV))['loss']
    loss.backward(); opt.step()
    steps += 1
dt = time.time()-tt
print(f"5 steps in {dt:.1f}s -> {dt/steps:.1f}s/step on MPS")
print("MPS FEASIBLE" if dt < 60 else "MPS SLOW")

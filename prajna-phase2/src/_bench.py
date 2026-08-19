import os, sys, time, json, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
DEV = sys.argv[1]
WARM = int(sys.argv[2]) if len(sys.argv) > 2 else 3
N = int(sys.argv[3]) if len(sys.argv) > 3 else 15
sys.path.insert(0, 'prajna-phase2/src')
from crn_components import PrajnaStudentMultiLayer, get_crn_state_dict
from torch.utils.data import DataLoader
import torch.nn as nn

print(f"device={DEV} warm={WARM} n={N}")
student = PrajnaStudentMultiLayer(device=DEV, inject_every=8, max_length=64,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
    num_corrections=8, mem_size=256, mem_dim=64)
ckpt = torch.load('./prajna/checkpoints/math_opt_1000.pt', map_location=DEV, weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
student = student.to(DEV)
with torch.no_grad():
    student.crn_mix.mul_(8.0)
student.train()
tok = student.tok
with open('prajna/data/math_cot.json') as _f:
    ds = json.load(_f)
loader = DataLoader(ds, batch_size=1, shuffle=True)
opt = torch.optim.AdamW(student.get_params(), lr=3e-4)
crit = nn.CrossEntropyLoss(ignore_index=-100)
it = iter(loader)
t0 = time.time()
for i in range(WARM + N):
    s = next(it)
    text = f"{s['prompt'][0]} {s['response'][0]}{tok.eos_token}"
    enc = tok(text, truncation=True, max_length=64, padding='max_length', return_tensors='pt')
    ids = enc['input_ids']; labels = ids.clone(); labels[enc['attention_mask'] == 0] = -100
    if DEV == 'mps': torch.mps.empty_cache()
    t = time.time()
    loss = student(ids.to(DEV), labels.to(DEV))['loss']
    loss.backward(); opt.step(); opt.zero_grad()
    dt = time.time() - t
    if i >= WARM:
        print(f"  step {i} {dt:.1f}s  loss={loss.item():.3f}", flush=True)
print(f"AVG {N} steps = {(time.time()-t0 - sum([]))/N:.1f}s/step on {DEV}")

#!/usr/bin/env python3
"""Analyze what the CRN (dpo_final.pt) actually learned.

Probes (all on CPU, full recompute):
1. crn_mix values + which injections fire.
2. Per-injection ablation on the in-distribution ppl (teacher_data) and on
   generic text, to see if some injections help / some hurt.
3. Car-wash / IGR: original prompts vs REWORDED prompts (same reasoning,
   different wording) to test memorization vs generalization.
4. Memory read contribution magnitude.
"""
import os, sys, json, time, torch, random
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer

random.seed(7)
torch.set_num_threads(min(os.cpu_count() or 8, 8))

CKPT = '../../prajna/checkpoints/dpo_final.pt'
MEM  = '../../prajna/checkpoints/memory_dpo_final.json'

print("Loading model...")
student = PrajnaStudentMultiLayer(device='cpu', inject_every=8, max_length=32,
                                  crn_mix_init=0.05)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
if os.path.exists(MEM):
    student.load_memory(MEM)
student.eval()
tok = student.tok
print(f"Loaded step {ckpt.get('step')} loss {ckpt.get('loss'):.4f}")

# ---- 1. crn_mix ----
with torch.no_grad():
    mix = torch.sigmoid(student.crn_mix).tolist()
print("\n[1] crn_mix (sigmoid) per injection:", [f"{m:.3f}" for m in mix],
      "at layers", student.inject_indices)

# ---- helpers ----
@torch.no_grad()
def logits_of(ids):
    o = student._collect_hidden(ids)
    logits, _ = student._apply_crn(o, training=False)
    return logits

@torch.no_grad()
def ppl_of(text):
    ids = tok(text, truncation=True, max_length=32, return_tensors='pt').input_ids
    if ids.shape[1] < 4: return None
    logits = logits_of(ids)
    labels = ids[:, 1:].clone()
    import torch.nn.functional as F
    l = F.cross_entropy(logits[:, :-1].reshape(-1, 262144), labels.reshape(-1))
    return float(torch.exp(l))

def gen_text(prompt, max_new=32):
    ids = tok(prompt, return_tensors='pt').input_ids
    g = ids.clone()
    with torch.no_grad():
        for _ in range(max_new):
            o = student._collect_hidden(g)
            lg, _ = student._apply_crn(o, training=False)
            nt = lg[:, -1, :].argmax(-1).reshape(1, 1)
            g = torch.cat([g, nt], dim=1)
            if nt.item() == tok.eos_token_id: break
    return tok.decode(g[0], skip_special_tokens=True)[len(prompt):].strip()

# ---- 2. ablation on teacher_data (in-dist) ----
print("\n[2] Ablation on held-out teacher_data (lower ppl = better)...")
data = json.load(open('../../prajna/data/teacher_data.json'))
test = random.sample(data, 30)
base_ppls, crn_ppls = [], []
for s in test:
    t = f"{s.get('prompt','')}\n\n{s.get('response','')}"
    bp = ppl_of(t); 
    if bp is None: continue
    base_ppls.append(bp)
full = sum(base_ppls)/len(base_ppls)
print(f"    Base E2B ppl (mean of {len(base_ppls)}): {full:.2f}")

# ablate each injection by zeroing crn_mix[idx]
print("    Per-injection ablation (ppl when that injection disabled):")
for idx in range(student.num_injections):
    student.crn_mix[idx].data.fill_(-1e9)  # sigmoid -> ~0
    ps = []
    for s in test:
        t = f"{s.get('prompt','')}\n\n{s.get('response','')}"
        p = ppl_of(t)
        if p is not None: ps.append(p)
    m = sum(ps)/len(ps)
    print(f"      disable inj@{student.inject_indices[idx]}: ppl={m:.2f}  (delta {m-full:+.2f})")
    student.crn_mix[idx].data.fill_(0.0)  # restore ~0.05 sigmoid
# restore
student.crn_mix.data.fill_(0.0)

# ---- 3. car-wash original vs reworded ----
print("\n[3] Car-wash / IGR: ORIGINAL vs REWORDED (same logic, new wording)")
ORIG = [
    ("I want to wash my car. The car wash is 100 meters away. Should I walk or drive?", "drive"),
    ("My cat is ill and must see the vet. The clinic is 200 meters away. Should I walk the cat or carry it?", "carry"),
    ("My car is out of fuel and the station is 150 meters away. Should I walk or drive there?", "drive"),
    ("My clothes are dirty and the laundromat is around the corner. Should I wear them or bring them?", "bring"),
    ("I need groceries at home. The shop is 100 meters away. Should I walk there or order delivery?", "deliver"),
    ("My bike chain broke and the shop is 200 meters away. Should I ride or walk it?", "walk"),
    ("I want to boat. The ramp is 100 meters away. Should I swim or tow the boat?", "tow"),
    ("My EV is low on charge and the charger is 100 meters away. Should I walk or drive?", "drive"),
]
REWORD = [
    ("There's a car wash two minutes from here and my car needs cleaning. On foot or by car?", "drive"),
    ("My cat is sick, the animal hospital is a short stroll away. Carry the cat or stroll with it?", "carry"),
    ("The gas tank is empty but a pump is nearby. Walk over or take the car?", "drive"),
    ("These shirts are filthy and there's a washer just down the block. Put them on or take them?", "bring"),
    ("We're out of food at the flat and the store is right there. Go get it or have it sent?", "deliver"),
    ("The bicycle's chain snapped and the repair place is close. Cycle or push it?", "walk"),
    ("Keen to get on the water and the launch is near. Wade or haul the boat?", "tow"),
    ("The electric car needs power and a charger is a few steps away. Walk or drive to it?", "drive"),
]
def pass_rate(pairs):
    ok = 0; outs = []
    for q, a in pairs:
        o = gen_text(q, max_new=60).lower()
        good = (a in o)
        ok += good
        outs.append((q[:40], o[:50], good))
    return ok, len(pairs), outs
o1, n1, _ = pass_rate(ORIG)
o2, n2, _ = pass_rate(REWORD)
print(f"    ORIGINAL prompts:    {o1}/{n1} passed")
print(f"    REWORDED prompts:    {o2}/{n2} passed  <-- if much lower, it's memorization")
print("    Sample reworded outputs:")
for q, o, good in pass_rate(REWORD)[2][:4]:
    print(f"      [{'OK' if good else 'XX'}] {q}... -> {o}...")

# ---- 4. memory contribution ----
print("\n[4] Memory read magnitude (should be near 0 if unused):")
with torch.no_grad():
    ids = tok(ORIG[0][0], return_tensors='pt').input_ids
    o = student._collect_hidden(ids)
    fh = o['final_hidden'].detach().float()
    if student.mem.temporal_positions.sum() > 0:
        read_out, w = student.mem.read(fh.mean(dim=1).float(), top_k=8)
        print(f"    mem read L2 norm: {read_out.norm().item():.4f}  (corrections added = read_out.unsqueeze(1))")
    else:
        print("    memory empty (temporal_positions sum == 0) -> no memory contribution")

print("\nDONE")

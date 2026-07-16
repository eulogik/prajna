#!/usr/bin/env python3
"""Stage [3] only: car-wash ORIGINAL vs REWORDED generalization test.
Isolated + fast: max_new=24, per-prompt timed, prints incrementally.
"""
import os, sys, json, time, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer

CKPT = '../../prajna/checkpoints/dpo_final.pt'
MEM  = '../../prajna/checkpoints/memory_dpo_final.json'

print("Loading model...", flush=True)
student = PrajnaStudentMultiLayer(device='cpu', inject_every=8, max_length=32, crn_mix_init=0.05)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
if os.path.exists(MEM): student.load_memory(MEM)
student.eval()
tok = student.tok

@torch.no_grad()
def gen_text(prompt, max_new=24):
    ids = tok(prompt, return_tensors='pt').input_ids
    g = ids.clone()
    for _ in range(max_new):
        o = student._collect_hidden(g)
        lg, _ = student._apply_crn(o, training=False)
        nt = lg[:, -1, :].argmax(-1).reshape(1, 1)
        g = torch.cat([g, nt], dim=1)
        if nt.item() == tok.eos_token_id: break
    return tok.decode(g[0], skip_special_tokens=True)[len(prompt):].strip()

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

def pass_rate(pairs, tag):
    ok = 0
    for i,(q,a) in enumerate(pairs):
        t0=time.time()
        o = gen_text(q, max_new=24).lower()
        good = a in o
        ok += good
        print(f"  [{tag}] {i+1} {'OK ' if good else 'XX '} {q[:38]:40s} -> {o[:46]!r}  ({time.time()-t0:.1f}s)", flush=True)
    print(f"  >> {tag}: {ok}/{len(pairs)} passed\n", flush=True)
    return ok, len(pairs)

print("=== ORIGINAL prompts ===", flush=True)
pass_rate(ORIG, "orig")
print("=== REWORDED prompts ===", flush=True)
pass_rate(REWORD, "reword")
print("DONE", flush=True)

#!/usr/bin/env python3
"""Evaluate a Prajna CRN checkpoint on the car-wash / IGR (common-sense) test and a
facts spot-check. Reports:
  - naive-correct rate   : model picks the non-naive (correct) action
  - correct-reason rate  : it also articulates the implicit goal
Usage: CRN_DEVICE=mps python3 eval_csn.py <ckpt.pt> <memory.json>
"""
import os, sys, json, re, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer

CKPT = sys.argv[1] if len(sys.argv) > 1 else './prajna/checkpoints/dpo_csn_final.pt'
MEM = sys.argv[2] if len(sys.argv) > 2 else './prajna/checkpoints/memory_dpo_csn_final.json'
DEVICE = os.environ.get('CRN_DEVICE', 'mps')
MAX_NEW = 160

# (prompt, correct_action_keyword, goal_phrase) — goal_phrase indicates the model
# surfaced the implicit goal (not just lucky).
TESTS = [
    ("I want to wash my car. The car wash is 100 meters away. Should I walk or drive?",
     "drive", "car"),
    ("My dog is sick and needs the vet. The clinic is 200 meters away. Should I walk the dog or carry it?",
     "carry", "sick"),
    ("My car is nearly out of fuel and the gas station is 100 meters away. Should I walk or drive?",
     "drive", "fuel"),
    ("My clothes are dirty and the laundromat is around the corner. Should I wear them there or bring them?",
     "bring", "dirty"),
    ("I need groceries delivered to my home. The store is 100 meters away. Should I walk there or order delivery?",
     "deliver", "home"),
    ("My bike has a broken chain and the bike shop is 200 meters away. Should I ride it or walk it?",
     "walk", "broken"),
    ("I want to go boating. The boat ramp is 100 meters away. Should I swim there or tow the boat?",
     "tow", "boat"),
    ("My electric car is low on charge and the station is 100 meters away. Should I walk or drive?",
     "drive", "charge"),
]

FACTS = [
    ("What is the capital of France?", "paris"),
    ("Who wrote 'Romeo and Juliet'?", "shakespeare"),
    ("What is the chemical symbol for gold?", "au"),
    ("What year did World War II end?", "1945"),
    ("What planet is known as the Red Planet?", "mars"),
]

@torch.no_grad()
def generate(student, ids):
    gen = ids.clone()
    for _ in range(MAX_NEW):
        out = student._collect_hidden(gen)
        logits, _ = student._apply_crn(out, training=False)
        nt = logits[:, -1, :].argmax(-1).reshape(1, 1)
        gen = torch.cat([gen, nt], dim=1)
        if nt.item() == student.tok.eos_token_id: break
    return gen

student = PrajnaStudentMultiLayer(device='cpu', inject_every=8, max_length=160,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4, num_corrections=8, mem_size=256, mem_dim=64)
student = student.to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
if os.path.exists(MEM): student.load_memory(MEM)
student.eval()
tok = student.tok

# WARNING: this is a LOOSE keyword heuristic, NOT a reasoning validation.
# Per RESULTS.md, the CRN does NOT perform implicit-goal reasoning: on reworded
# prompts it scores 0/8, and matching here can be a lucky keyword occurrence.
# Treat any pass rate as a weak signal only.
print(f"\n=== IGR / CAR-WASH (non-naive) TEST — {CKPT} ===")
print("WARNING: keyword heuristic only; see RESULTS.md (no real reasoning).")
pass_n = reason_n = 0
for prompt, action, goal in TESTS:
    ids = tok(prompt + "\n", return_tensors='pt').input_ids.to(DEVICE)
    gen = generate(student, ids)
    text = tok.decode(gen[0].cpu(), skip_special_tokens=True)[len(prompt):].strip()
    low = text.lower()
    ok_action = action in low
    ok_reason = ok_action and goal in low
    pass_n += ok_action; reason_n += ok_reason
    print(f"\nQ: {prompt}")
    print(f"A: {text[:200]}")
    print(f"  -> action_correct={ok_action}  goal_articulated={ok_reason}")
n = len(TESTS)
print(f"\n=== RESULTS: naive-correct {pass_n}/{n} ({pass_n/n*100:.0f}%) | correct-reason {reason_n}/{n} ({reason_n/n*100:.0f}%) ===")

print(f"\n=== FACTS SPOT-CHECK ===")
fc = 0
for q, a in FACTS:
    ids = tok(q + "\n", return_tensors='pt').input_ids.to(DEVICE)
    gen = generate(student, ids)
    text = tok.decode(gen[0].cpu(), skip_special_tokens=True)[len(q):].strip().lower()
    ok = a in text
    fc += ok
    print(f"  {q} -> {text[:60]}  [{'OK' if ok else 'MISS'}]")
print(f"  facts: {fc}/{len(FACTS)} ({fc/len(FACTS)*100:.0f}%)")

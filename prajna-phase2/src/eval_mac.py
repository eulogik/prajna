#!/usr/bin/env python3
"""Prajna Evaluation — Mac M4 CPU.

Compares CRN-trained model (SFT+DPO) vs base Gemma 4 E2B on:
1. Perplexity on held-out test set (lower = better)
2. Exact-match accuracy on math/facts
3. Qualitative sample generations
"""
import torch, os, sys, time, json, random
from pathlib import Path
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Config (must match train_mac.py) ----
INJECT_EVERY = 8
MAX_LENGTH = 32
CRN_MIX_INIT = 0.05

# Import CRN components from standalone module (no training side effects)
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import (PrajnaStudentMultiLayer, CRN_PREFIXES, get_crn_state_dict)

DEVICE = 'cpu'
CKPT = os.environ.get('CRN_CKPT', './prajna/checkpoints/dpo_final.pt')
MEM = './prajna/checkpoints/memory_dpo_final.json'

# ---- Test prompts ----
MATH = [
    ("What is 15 * 17?", "255"),
    ("What is 144 / 12?", "12"),
    ("What is 2^8?", "256"),
    ("What is 73 + 89?", "162"),
    ("What is 1000 - 347?", "653"),
]
FACTS = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"),
    ("What year did World War II end?", "1945"),
    ("What is the chemical symbol for gold?", "Au"),
    ("How many continents are there?", "7"),
]
CODE = [
    "Write a Python function to check if a number is prime.",
    "Write a Python function to find the maximum in a list.",
    "Write a Python one-liner to reverse a string.",
]
REASONING = [
    "If all bloops are razzies and all razzies are lazzies, are all bloops definitely lazzies?",
    "A train leaves at 2pm going 60mph. Another leaves same place at 3pm going 80mph. When does the second catch up?",
    "Susan has 3 brothers. Each brother has 2 sisters. How many sisters does Susan have?",
]

print("Loading model...")
student = PrajnaStudentMultiLayer(
    device=DEVICE,
    inject_every=INJECT_EVERY, max_length=MAX_LENGTH, crn_mix_init=CRN_MIX_INIT,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
    num_corrections=8, mem_size=256, mem_dim=64)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
if os.path.exists(MEM):
    student.load_memory(MEM)
student.eval()
tok = student.tok
print(f"Loaded CRN from step {ckpt['step']}, loss {ckpt.get('loss'):.4f}")

@torch.no_grad()
def generate_crn(input_ids, max_new=MAX_LENGTH, temp=0.7, top_k=20):
    gen = input_ids.clone()
    for _ in range(max_new):
        out = student._collect_hidden(gen)
        logits, _ = student._apply_crn(out, training=False)
        nxt = logits[:, -1, :] / temp
        tv, ti = torch.topk(nxt, top_k, dim=-1)
        p = F.softmax(tv, dim=-1)
        nt = ti[0, torch.multinomial(p, 1)].reshape(1, 1)
        gen = torch.cat([gen, nt], dim=1)
        if nt.item() == tok.eos_token_id: break
    return gen

@torch.no_grad()
def generate_base(input_ids, max_new=MAX_LENGTH):
    out = student.base_model.generate(input_ids, max_new_tokens=max_new,
                                      do_sample=True, temperature=0.7, top_k=20)
    return out

def extract_answer(text):
    # crude: take last line / number
    return text.strip().split('\n')[-1].strip()

def eval_exact_match(prompts, answers, gen_fn):
    correct = 0
    samples = []
    for p, a in zip(prompts, answers):
        ids = tok(p, return_tensors='pt').input_ids
        gen = gen_fn(ids)
        text = tok.decode(gen[0], skip_special_tokens=True)
        resp = text[len(p):].strip()
        ok = a.lower() in resp.lower()
        correct += ok
        samples.append((p, resp, ok))
    acc = correct / len(prompts)
    return acc, samples

def eval_freeform(prompts, gen_fn):
    samples = []
    for p in prompts:
        ids = tok(p, return_tensors='pt').input_ids
        gen = gen_fn(ids)
        text = tok.decode(gen[0], skip_special_tokens=True)
        samples.append((p, text[len(p):].strip()))
    return samples

# ---- Perplexity on held-out test set ----
def perplexity(gen_fn_base, gen_fn_crn, n=200):
    # Use teacher data samples as test set
    with open('./prajna/data/teacher_data.json') as f:
        data = json.load(f)
    random.seed(7)
    test = random.sample(data, n)
    crit = nn.CrossEntropyLoss(ignore_index=-100)

    def ppl(fn, ids):
        if fn == 'crn':
            out = student._collect_hidden(ids)
            logits, _ = student._apply_crn(out, training=False)
        else:
            logits = student.base_model(ids).logits
        labels = ids[:, 1:].clone()
        l = crit(logits[:, :-1].reshape(-1, 262144), labels.reshape(-1))
        return torch.exp(l).item()

    tot_base = tot_crn = 0
    for s in test:
        txt = f"{s.get('prompt','')}\n\n{s.get('response','')}"
        ids = tok(txt, truncation=True, max_length=MAX_LENGTH, return_tensors='pt').input_ids
        if ids.shape[1] < 4: continue
        tot_base += ppl('base', ids)
        tot_crn += ppl('crn', ids)
    return tot_base / len(test), tot_crn / len(test)

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

t0 = time.time()
PPL_CACHE = f'./prajna/eval_ppl_cache_{os.path.basename(CKPT)}.json'
if os.path.exists(PPL_CACHE):
    with open(PPL_CACHE) as f:
        base_ppl, crn_ppl = json.load(f)
    print(f"\n[loaded cached perplexity]")
else:
    base_ppl, crn_ppl = perplexity(None, None)
    with open(PPL_CACHE, 'w') as f:
        json.dump([base_ppl, crn_ppl], f)
print(f"\nPerplexity (lower=better) on 200 held-out samples:")
print(f"  Base E2B:     {base_ppl:.3f}")
print(f"  CRN (SFT+DPO): {crn_ppl:.3f}")
print(f"  Delta:        {base_ppl - crn_ppl:+.3f} (CRN {'better' if crn_ppl < base_ppl else 'worse'})")

print("\n--- Exact Match: Math ---")
acc_m, sm = eval_exact_match([p for p, _ in MATH], [a for _, a in MATH], lambda ids: generate_crn(ids))
print(f"  CRN accuracy: {acc_m*100:.0f}%")
for p, r, ok in sm[:3]:
    print(f"   [{'OK' if ok else 'XX'}] {p} -> {r[:60]}")

print("\n--- Exact Match: Facts ---")
acc_f, sf = eval_exact_match([p for p, _ in FACTS], [a for _, a in FACTS], lambda ids: generate_crn(ids))
print(f"  CRN accuracy: {acc_f*100:.0f}%")
for p, r, ok in sf[:3]:
    print(f"   [{'OK' if ok else 'XX'}] {p} -> {r[:60]}")

print("\n--- Qualitative: Code (CRN vs Base) ---")
for p in CODE[:2]:
    ids = tok(p, return_tensors='pt').input_ids
    gc = generate_crn(ids)
    gb = generate_base(ids)
    print(f"\n  PROMPT: {p}")
    print(f"  CRN:  {tok.decode(gc[0], skip_special_tokens=True)[len(p):].strip()[:120]}")
    print(f"  BASE: {tok.decode(gb[0], skip_special_tokens=True)[len(p):].strip()[:120]}")

print("\n--- Qualitative: Reasoning (CRN) ---")
for p in REASONING[:2]:
    ids = tok(p, return_tensors='pt').input_ids
    gc = generate_crn(ids)
    print(f"\n  PROMPT: {p}")
    print(f"  CRN:  {tok.decode(gc[0], skip_special_tokens=True)[len(p):].strip()[:150]}")

print(f"\nEvaluation done in {(time.time()-t0)/60:.1f} min")
print(f"Summary: base_ppl={base_ppl:.3f} crn_ppl={crn_ppl:.3f} math_acc={acc_m*100:.0f}% facts_acc={acc_f*100:.0f}%")

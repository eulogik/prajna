#!/usr/bin/env python3
"""Prajna-2B Phase 1 evaluation: does the CRN actually FIX the base's errors?

Metrics:
  1. Error-correction rate: on held-out prompts where BASE is wrong, what % does
     CRN get right? (primary — the whole point of the ReflectiveLoop)
  2. Reflection ablation: disable reflection_gate -> error-correction should drop.
  3. In-distribution ppl (held-out teacher_data).
  4. OOD bpb (generic text).

Usage: python3 eval_prajna2b.py
  CKPT=./prajna/checkpoints/dpo_v2_final.pt (or dpo_final.pt for baseline)
"""
import os, sys, json, time, random, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer

random.seed(7)
torch.set_num_threads(min(os.cpu_count() or 8, 8))

CKPT = os.environ.get('CKPT', './prajna/checkpoints/dpo_v2_final.pt')
MEM = os.environ.get('MEM', './prajna/checkpoints/memory_v2_final.json')
N = int(os.environ.get('EC_EVAL_N', '100'))

print(f"Loading {CKPT} ...")
student = PrajnaStudentMultiLayer(device='cpu', inject_every=8, max_length=96, crn_mix_init=0.05)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
if os.path.exists(MEM):
    student.load_memory(MEM)
student.eval()
tok = student.tok
print(f"reflection_gate (sigmoid): {[f'{x:.3f}' for x in torch.sigmoid(student.reflection_gate).tolist()]}")

@torch.no_grad()
def gen_crn(prompt, max_new=96):
    ids = tok(prompt, return_tensors='pt').input_ids
    g = ids.clone()
    for _ in range(max_new):
        o = student._collect_hidden(g)
        lg, _ = student._apply_crn(o, training=False)
        nt = lg[:, -1, :].argmax(-1).reshape(1, 1)
        g = torch.cat([g, nt], dim=1)
        if nt.item() == tok.eos_token_id: break
    return tok.decode(g[0], skip_special_tokens=True)[len(prompt):].strip()

@torch.no_grad()
def gen_base(prompt, max_new=96):
    ids = tok(prompt, return_tensors='pt').input_ids
    out = student.base_model.generate(ids, max_new_tokens=max_new, do_sample=False,
                                      pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def correct(out, gold):
    return gold.strip().lower() in out.strip().lower()

# ---- Held-out error set: prompts where BASE is wrong ----
# Build from error_correction_pairs (rejected = base wrong output).
ec = json.load(open('./prajna/data/error_correction_pairs.json'))
random.shuffle(ec)
held = [e for e in ec if e['domain'] in ('math', 'facts', 'igr')][:N]
# keep only those where base is actually wrong on this prompt (sanity)
base_wrong = []
for e in held:
    if not correct(gen_base(e['prompt']), e['chosen']):
        base_wrong.append(e)
print(f"\nHeld-out prompts where BASE is wrong: {len(base_wrong)}/{len(held)}")

crn_fixed = 0
for e in base_wrong:
    out = gen_crn(e['prompt'])
    if correct(out, e['chosen']):
        crn_fixed += 1
rate = crn_fixed / len(base_wrong) if base_wrong else 0
print(f"\n[1] ERROR-CORRECTION RATE: CRN fixes {crn_fixed}/{len(base_wrong)} = {rate*100:.1f}% of base errors")

# ---- Ablation: disable reflection ----
student.reflection_gate.data.fill_(-1e9)  # sigmoid -> ~0
crn_fixed_no_ref = 0
for e in base_wrong:
    out = gen_crn(e['prompt'])
    if correct(out, e['chosen']):
        crn_fixed_no_ref += 1
rate_no_ref = crn_fixed_no_ref / len(base_wrong) if base_wrong else 0
print(f"[2] ABLATION (reflection OFF): CRN fixes {crn_fixed_no_ref}/{len(base_wrong)} = {rate_no_ref*100:.1f}%")
print(f"    Reflection contribution: {(rate-rate_no_ref)*100:+.1f} pts")
student.reflection_gate.data.fill_(0.0)  # restore

# ---- In-distribution ppl ----
def ppl(text):
    ids = tok(text, truncation=True, max_length=32, return_tensors='pt').input_ids
    if ids.shape[1] < 4: return None
    o = student._collect_hidden(ids); logits, _ = student._apply_crn(o, training=False)
    import torch.nn.functional as F
    l = F.cross_entropy(logits[:, :-1].reshape(-1, 262144), ids[:, 1:].reshape(-1))
    return float(torch.exp(l))

tdata = json.load(open('./prajna/data/teacher_data.json'))
random.shuffle(tdata)
ps = [ppl(f"{s.get('prompt','')}\n\n{s.get('response','')}") for s in tdata[:30]]
ps = [p for p in ps if p]
print(f"[3] In-distribution ppl (teacher_data, 30): {sum(ps)/len(ps):.2f}")

# ---- OOD bpb ----
# Use a few generic sentences as a crude OOD probe
ood_texts = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "Photosynthesis converts light energy into chemical energy stored in glucose.",
    "She opened the letter and read it twice before responding to her friend.",
]
ods = [ppl(t) for t in ood_texts]
ods = [p for p in ods if p]
print(f"[4] OOD ppl (generic, {len(ods)}): {sum(ods)/len(ods):.2f}")

print(f"\n=== SUMMARY (Prajna-2B Phase 1) ===")
print(f"  Error-correction rate: {rate*100:.1f}%  (reflection OFF: {rate_no_ref*100:.1f}%)")
print(f"  In-dist ppl: {sum(ps)/len(ps):.2f} | OOD ppl: {sum(ods)/len(ods):.2f}")
print("  (compare: dpo_final.pt baseline had no reflection; OOD was ~3x worse than base)")

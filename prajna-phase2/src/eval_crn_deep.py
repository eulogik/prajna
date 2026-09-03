#!/usr/bin/env python3
"""Deep injection eval: error correction on CEHRI with autoreg generation.

The deep injection corrector modifies hidden states at early layers,
so generation must run full autoregressive forward (no KV cache).
"""
import os, sys, json, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from crn_deep import CRNDeepInjection

CKPT = os.environ.get('DEEP_CKPT', 'prajna/checkpoints/crn_deep_dpo.pt')
EXAM = os.environ.get('CEHRI_EXAM', 'prajna/data/cehri_exam.json')
DEV = os.environ.get('DEV', 'mps')
MAX_NEW = int(os.environ.get('MAX_NEW', '16'))
RANK = int(os.environ.get("DEEP_RANK", "512"))
DEPTHS = [int(x) for x in os.environ.get("DEEP_DEPTHS", "7").split(',')]

model = CRNDeepInjection(device=DEV, correction_rank=RANK, depths=DEPTHS)
model.load(CKPT)
print(f"adapter: {CKPT} | exam: {EXAM} | depths: {DEPTHS}", flush=True)


@torch.no_grad()
def gen(prompt, max_new=MAX_NEW):
    """Autoregressive generation with deep injection at each step."""
    input_text = prompt + ": "
    ids = model.tok(input_text, return_tensors="pt").input_ids.to(DEV)
    prompt_len = ids.shape[1]
    generated = ids.clone()
    for _ in range(max_new):
        logits = model(input_ids=generated)['logits']
        next_token = logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
        if next_token.item() == model.tok.eos_token_id:
            break
        generated = torch.cat([generated, next_token], dim=1)
    return model.tok.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()


exam = json.load(open(EXAM))
passed = 0
for q in exam:
    out = gen(q["prompt"])
    ok = q["answer"].strip().lower() in out.strip().lower()
    passed += ok
    print(f"  {q['id']}: {'PASS' if ok else 'FAIL'}  {out[:60]!r}", flush=True)
frac = passed / len(exam)
print(f"\nCRN DEEP RESULT: {passed}/{len(exam)} = {frac*100:.1f}%", flush=True)

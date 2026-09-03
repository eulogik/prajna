#!/usr/bin/env python3
"""CRN v2 multi-depth eval: error correction on CEHRI exam.

Autoregressive generation using corrected logits from multi-depth correction.
Full base forward pass per token (no KV cache).
"""
import os, sys, json, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from crn_v2_multidepth import CRNv2MultiDepth

CKPT = os.environ.get('CRN_V2_CKPT', 'prajna/checkpoints/crn_v2_multidepth_dpo.pt')
EXAM = os.environ.get('CEHRI_EXAM', 'prajna/data/cehri_exam.json')
DEV = os.environ.get('DEV', 'mps')
MAX_NEW = int(os.environ.get('MAX_NEW', '16'))
CORRECTION_RANK = int(os.environ.get("CRN_V2_RANK", "128"))
DEPTHS = [int(x) for x in os.environ.get("CRN_V2_DEPTHS", "7,15,23,31").split(',')]

model = CRNv2MultiDepth(device=DEV, correction_rank=CORRECTION_RANK, depths=DEPTHS)
model.load(CKPT)
model.to(DEV).eval()
model.base.to(DEV).eval()
print(f"adapter: {CKPT} | exam: {EXAM} | depths: {DEPTHS}", flush=True)


@torch.no_grad()
def gen(prompt, max_new=MAX_NEW):
    input_text = prompt + ": "
    ids = model.tok(input_text, return_tokens=False, return_tensors="pt").input_ids.to(DEV)
    prompt_len = ids.shape[1]
    generated = ids.clone()

    for _ in range(max_new):
        base_logits, hidden_list = model._collect_hidden_states(generated)
        delta = model.correction(hidden_list)
        corrected_logits = (base_logits + delta).to(torch.float16)
        next_token = corrected_logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
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
print(f"\nCRN V2 MULTI-DEPTH RESULT: {passed}/{len(exam)} = {frac*100:.1f}%", flush=True)

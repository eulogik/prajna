#!/usr/bin/env python3
"""CRN v2 eval: error correction (CEHRI) + capability preservation.

Generation uses CRN v2's corrected logits autoregressively:
  at each step, the full sequence is run through the base model
  to get hidden states, the correction module modifies them,
  and the corrected logits produce the next token.

Usage:
  CRN_V2_CKPT=prajna/checkpoints/crn_v2_dpo.pt CEHRI_EXAM=prajna/data/cehri_exam.json python3 eval_crn_v2.py
  CRN_V2_CKPT=prajna/checkpoints/crn_v2_dpo.pt CEHRI_EXAM=prajna/data/cehri_exam_reworded.json python3 eval_crn_v2.py
"""
import os, sys, json, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from crn_v2 import CRNv2

CKPT = os.environ.get('CRN_V2_CKPT', 'prajna/checkpoints/crn_v2_dpo.pt')
EXAM = os.environ.get('CEHRI_EXAM', 'prajna/data/cehri_exam.json')
DEV = os.environ.get('DEV', 'mps')
MAX_NEW = int(os.environ.get('MAX_NEW', '64'))

CORRECTION_RANK = int(os.environ.get("CRN_V2_RANK", "128"))
model = CRNv2(device=DEV, correction_rank=CORRECTION_RANK)
model.load(CKPT)
model.to(DEV).eval()
model.base.to(DEV).eval()
print(f"adapter: {CKPT} | exam: {EXAM}", flush=True)


@torch.no_grad()
def gen(prompt, max_new=MAX_NEW):
    """Generate using CRN v2's corrected logits autoregressively.

    At each step:
      1. Run base model on full sequence → get final hidden state
      2. Apply correction module → get corrected logits
      3. Greedy-decode next token
    """
    input_text = prompt + ": "
    ids = model.tok(input_text, return_tokens=False, return_tensors="pt").input_ids.to(DEV)
    prompt_len = ids.shape[1]
    generated = ids.clone()

    for _ in range(max_new):
        # Full forward pass through base model (no KV cache — correction needs full hidden states)
        base_out = model.base(input_ids=generated, output_hidden_states=True, return_dict=True)
        final_hidden = base_out.hidden_states[-1].float()
        base_logits = base_out.logits.float()
        del base_out

        # Apply correction module to get corrected logits
        delta = model.correction(final_hidden)
        corrected_logits = (base_logits + delta).to(torch.float16)

        # Greedy decode next token from the last position
        next_token = corrected_logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)

        # Check for EOS
        if next_token.item() == model.tok.eos_token_id:
            break

        generated = torch.cat([generated, next_token], dim=1)

    # Decode only the generated tokens (after prompt)
    output_text = model.tok.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()
    return output_text


exam = json.load(open(EXAM))
passed = 0
for q in exam:
    out = gen(q["prompt"])
    ok = q["answer"].strip().lower() in out.strip().lower()
    passed += ok
    print(f"  {q['id']}: {'PASS' if ok else 'FAIL'}  {out[:60]!r}", flush=True)
frac = passed / len(exam)
print(f"\nCRN V2 RESULT: {passed}/{len(exam)} = {frac*100:.1f}%", flush=True)

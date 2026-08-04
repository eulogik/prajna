#!/usr/bin/env python3
"""CEHRI exam eval for the v2 CRN checkpoint (dpo_v2_final.pt)."""
import os, sys, json, torch
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from crn_components import PrajnaStudentMultiLayer

CKPT = os.environ.get("CEHRI_CKPT", "prajna/checkpoints/dpo_v2_final.pt")
MEM = os.environ.get("CEHRI_MEM", "prajna/checkpoints/memory_v2_final.json")
EXAM = os.environ.get("CEHRI_EXAM", "prajna/data/cehri_exam.json")
DEV = "mps"

student = PrajnaStudentMultiLayer(device=DEV, inject_every=4, max_length=96, crn_mix_init=2.0)
student = student.to(DEV)
sd = torch.load(CKPT, map_location=DEV, weights_only=False)
student.load_state_dict(sd["crn"], strict=False)
if os.path.exists(MEM):
    student.load_memory(MEM)
student.eval()
tok = student.tok
print("reflection_gate:", [f"{x:.3f}" for x in torch.sigmoid(student.reflection_gate).tolist()], flush=True)


@torch.no_grad()
def gen_crn(prompt, max_new=30):
    input_text = prompt + ": "
    ids = tok(input_text, return_tensors="pt").input_ids.to(DEV)
    g = ids.clone()
    gen_tokens = []
    for _ in range(max_new):
        o = student._collect_hidden(g)
        lg, _ = student._apply_crn(o, training=False)
        logits = lg[:, -1, :]
        for t in gen_tokens:
            logits[0, t] /= 1.15
        nt = logits.argmax(-1).reshape(1, 1)
        gen_tokens.append(nt.item())
        g = torch.cat([g, nt], dim=1)
        if nt.item() == tok.eos_token_id:
            break
    out = tok.decode(g[0], skip_special_tokens=True)
    return out[len(input_text):].strip()


exam = json.load(open(EXAM))
passed = 0
for q in exam:
    out = gen_crn(q["prompt"], max_new=30)
    ok = q["answer"].strip().lower() in out.strip().lower()
    passed += ok
    print(f"  {q['id']}: {'PASS' if ok else 'FAIL'}  {out[:60]!r}", flush=True)
frac = passed / len(exam)
print(f"\nCEHRI RESULT: {passed}/{len(exam)} = {frac*100:.1f}%  -> {'PASS' if frac >= 0.9 else 'FAIL (<0.9)'}", flush=True)

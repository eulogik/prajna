#!/usr/bin/env python3
"""CEHRI generation eval for the LoRA baseline (mirrors eval_cehri.py:
same prompt format, greedy decode, 1.15 repetition penalty on generated
tokens, max_new=30, substring scoring). GEN_CLAMP does not apply (it is a
CRN-specific L-2 harvest); the retrieval gate is adapter-independent.

Usage:
  LORA_ADAPTER=prajna/checkpoints/lora_baseline_dpo \
    CEHRI_EXAM=prajna/data/cehri_exam.json python3 eval_lora_cehri.py
"""
import os, sys, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

CKPT = os.environ.get("LORA_ADAPTER", "prajna/checkpoints/lora_baseline_dpo")
EXAM = os.environ.get("CEHRI_EXAM", "prajna/data/cehri_exam.json")
DEV = os.environ.get("DEV", "mps")
MAX_NEW = int(os.environ.get("MAX_NEW", "30"))

tok = AutoTokenizer.from_pretrained('google/gemma-4-E2B')
base = AutoModelForCausalLM.from_pretrained('google/gemma-4-E2B', dtype=torch.float16, low_cpu_mem_usage=False)
model = PeftModel.from_pretrained(base, CKPT)
model.to(DEV).eval()
print(f"adapter: {CKPT} | exam: {EXAM}", flush=True)


@torch.no_grad()
def gen(prompt, max_new=MAX_NEW):
    input_text = prompt + ": "
    ids = tok(input_text, return_tensors="pt").input_ids.to(DEV)
    out = model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False,
                         repetition_penalty=1.15, pad_token_id=tok.pad_token_id,
                         eos_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    return text[len(input_text):].strip()


exam = json.load(open(EXAM))
passed = 0
for q in exam:
    out = gen(q["prompt"])
    ok = q["answer"].strip().lower() in out.strip().lower()
    passed += ok
    print(f"  {q['id']}: {'PASS' if ok else 'FAIL'}  {out[:60]!r}", flush=True)
frac = passed / len(exam)
print(f"\nLORA BASELINE RESULT: {passed}/{len(exam)} = {frac*100:.1f}%", flush=True)

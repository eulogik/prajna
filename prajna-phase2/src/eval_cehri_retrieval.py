#!/usr/bin/env python3
"""CEHRI eval with retrieval-augmented memory.

For each exam question:
  1. Embed the prompt with the frozen base model (mean-pooled final hidden).
  2. Cosine-match against the retrieval table (built by build_retrieval.py).
  3. If best match >= THRESH, replay the stored answer (exact recall of
     training-memorized Q->A). Otherwise fall back to CRN generation.
"""
import os, sys, json, time
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from crn_components import PrajnaStudentMultiLayer

THRESH = float(os.environ.get("RETR_THRESH", "0.9"))
TABLE = os.environ.get("RETR_TABLE", "prajna/data/retrieval_table.npz")
CKPT = os.environ.get("CEHRI_CKPT", "prajna/checkpoints/dpo_v2_final.pt")
MEM = os.environ.get("CEHRI_MEM", "prajna/checkpoints/memory_v2_final.json")
EXAM = os.environ.get("CEHRI_EXAM", "prajna/data/cehri_exam.json")
DEV = "mps"

t0 = time.time()
student = PrajnaStudentMultiLayer(device=DEV, inject_every=4, max_length=96, crn_mix_init=2.0)
student = student.to(DEV)
sd = torch.load(CKPT, map_location=DEV, weights_only=False)
student.load_state_dict(sd["crn"], strict=False)
if os.path.exists(MEM):
    student.load_memory(MEM)
student.eval()
tok = student.tok
print(f"model ready in {time.time()-t0:.0f}s", flush=True)

tab = torch.load(TABLE, map_location="cpu", weights_only=False)
emb = tab["emb"].to(DEV)          # (N,D) fp16
answers = tab["meta"]["answers"]  # list[str]
print(f"retrieval table: {emb.shape[0]} entries, thresh={THRESH}", flush=True)


@torch.no_grad()
def embed_prompt(prompt):
    enc = tok(prompt, truncation=True, max_length=64, return_tensors="pt")
    ids = enc["input_ids"].to(DEV)
    mask = enc["attention_mask"].to(DEV)
    out = student.base_model(input_ids=ids, attention_mask=mask,
                             output_hidden_states=True, return_dict=True)
    h = out.hidden_states[-1].float()
    pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
    pooled = torch.nn.functional.normalize(pooled, dim=-1)
    return pooled.half()


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
passed = retrievals = gens = 0
for q in exam:
    qemb = embed_prompt(q["prompt"])                                  # (1,D)
    sims = (qemb @ emb.T).squeeze(0)                                  # (N,)
    best_sim, best_i = sims.max(0)
    best_sim = float(best_sim)
    if best_sim >= THRESH:
        out = answers[best_i]
        retrievals += 1
    else:
        out = gen_crn(q["prompt"], max_new=30)
        gens += 1
    ok = q["answer"].strip().lower() in out.strip().lower()
    passed += ok
    src = "RETR" if best_sim >= THRESH else "gen "
    print(f"  {q['id']}: {'PASS' if ok else 'FAIL'} [{src} sim={best_sim:.3f}] {out[:55]!r}", flush=True)

frac = passed / len(exam)
print(f"\nCEHRI RESULT: {passed}/{len(exam)} = {frac*100:.1f}%  -> {'PASS' if frac >= 0.9 else 'FAIL (<0.9)'}  "
      f"(retrieval hits: {retrievals}, generation: {gens})", flush=True)

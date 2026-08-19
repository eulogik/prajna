#!/usr/bin/env python3
"""Reworded-unseen CEHRI eval — the generalization gates.

Exams reworded with transform set B, which is DISJOINT from the set-A variants
used for training and the retrieval table. Two gates:

  --mode retr (default): semantic-memory gate — embed reworded prompt, cosine-match
      against the retrieval table; sim >= THRESH replays stored answer, else falls
      back to CRN generation. Always reports best-sim so we see the gap.
  --mode gen:  generation gate — CRN generation only (no memory), the Tier-3
      "does the CRN itself generalize to unseen phrasing?" measurement.

Env overrides: CEHRI_EXAM, RETR_TABLE, RETR_THRESH, CEHRI_CKPT, CEHRI_MEM.
Usage: python3 eval_cehri_reworded.py [--mode retr|gen]
"""
import os, sys, json, time, argparse
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
# The released checkpoint contains no logit-fusion weights; run the exact
# released configuration by default (override with FUSION_OFF=0 for the
# negative-results fusion experiments described in the paper).
os.environ.setdefault("FUSION_OFF", "1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from crn_components import PrajnaStudentMultiLayer

THRESH = float(os.environ.get("RETR_THRESH", "0.9"))
TABLE = os.environ.get("RETR_TABLE", "prajna/data/retrieval_table_v2.npz")
CKPT = os.environ.get("CEHRI_CKPT", "prajna/checkpoints/dpo_v2_final_seed.pt")
MEM = os.environ.get("CEHRI_MEM", "prajna/checkpoints/memory_v2_final_seed.json")
EXAM = os.environ.get("CEHRI_EXAM", "prajna/data/cehri_exam_reworded.json")
DEV = os.environ.get("DEV", "mps")

ap = argparse.ArgumentParser()
ap.add_argument("--mode", default="retr", choices=["retr", "gen"])
args = ap.parse_args()

t0 = time.time()
student = PrajnaStudentMultiLayer(device=DEV, inject_every=4, max_length=96, crn_mix_init=2.0)
student = student.to(DEV)
sd = torch.load(CKPT, map_location=DEV, weights_only=False)
student.load_state_dict(sd["crn"], strict=False)
if os.path.exists(MEM):
    student.load_memory(MEM)
student.eval()
tok = student.tok
JUNK_TOKENS = set()
for _i in range(tok.vocab_size):
    _s = tok.decode([_i]).strip()
    if len(_s) == 1 and not _s.isalnum():
        JUNK_TOKENS.add(_i)
for _c in '0123456789':
    _t = tok.encode(_c)
    if len(_t) == 1:
        JUNK_TOKENS.add(_t[0])

print(f"model ready in {time.time()-t0:.0f}s (mode={args.mode})", flush=True)

emb = answers = None
if args.mode == "retr":
    tab = torch.load(TABLE, map_location="cpu", weights_only=False)
    emb = tab["emb"].to(DEV)
    answers = tab["meta"]["answers"]
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
    past = None
    generated = []
    first = True
    while True:
        o = student._collect_hidden(torch.tensor(generated[-1:], device=ids.device).reshape(1, 1) if generated else ids, past_key_values=past)
        lg, _ = student._apply_crn(o, training=False)
        if first:
            first = False
            if os.environ.get("GEN_CLAMP", "1") == "1":
                l1 = lg[:, -1, :]
                nt1 = l1.argmax(-1)[0]
                use_l2 = os.environ.get("GEN_CLAMP_AUTO", "0") == "1" and nt1.item() in JUNK_TOKENS
                if use_l2 or os.environ.get("GEN_CLAMP_PURE", "0") == "1":
                    seq_logits = lg[:, :-1, :]
                    nt = seq_logits.argmax(-1)[0, -1].reshape(1, 1)
                    if nt.item() in (tok.eos_token_id, tok.pad_token_id) or nt.item() == ids[0, -1].item():
                        nt = nt1.reshape(1, 1)
                else:
                    nt = nt1.reshape(1, 1)
            else:
                logits = lg[:, -1, :]
                nt = logits.argmax(-1).reshape(1, 1)
        else:
            logits = lg[:, -1, :]
            for t in generated:
                logits[0, t] /= 1.15
            nt = logits.argmax(-1).reshape(1, 1)
        generated.append(nt.item())
        past = o["past"]
        if nt.item() == tok.eos_token_id:
            break
        if len(generated) >= max_new:
            break
    full = torch.cat([ids, torch.tensor(generated, device=ids.device).reshape(1, -1)], dim=1)
    out = tok.decode(full[0], skip_special_tokens=True)
    return out[len(input_text):].strip()


exam = json.load(open(EXAM))
passed = retrievals = gens = 0
sims = []
for q in exam:
    if args.mode == "retr":
        qemb = embed_prompt(q["prompt"])
        sims_all = (qemb @ emb.T).squeeze(0)
        best_sim, best_i = sims_all.max(0)
        best_sim = float(best_sim)
        sims.append(best_sim)
        if best_sim >= THRESH:
            out = answers[best_i]
            retrievals += 1
            src = f"RETR sim={best_sim:.3f}"
        else:
            out = gen_crn(q["prompt"], max_new=30)
            gens += 1
            src = f"gen  sim={best_sim:.3f}"
    else:
        out = gen_crn(q["prompt"], max_new=30)
        gens += 1
        src = "gen "
    ok = q["answer"].strip().lower() in out.strip().lower()
    passed += ok
    print(f"  {q['id']}: {'PASS' if ok else 'FAIL'} [{src}] {out[:55]!r}", flush=True)

frac = passed / len(exam)
line = f"\nREWORDED {'RETR' if args.mode == 'retr' else 'GEN'} RESULT: {passed}/{len(exam)} = {frac*100:.1f}%  -> {'PASS' if frac >= 0.9 else 'FAIL (<0.9)'}  (retrievals: {retrievals}, generation: {gens})"
print(line, flush=True)
if sims:
    sims_sorted = sorted(sims)
    print(f"best-sim distribution: min={sims_sorted[0]:.3f} median={sims_sorted[len(sims)//2]:.3f} max={sims_sorted[-1]:.3f} "
          f"hits>=0.9: {sum(1 for s in sims if s >= 0.9)}/{len(sims)}", flush=True)
with open(f"logs/reworded_{args.mode}.log", "w") as f:
    f.write(line + "\n")

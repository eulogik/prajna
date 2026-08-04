#!/usr/bin/env python3
"""Build a prompt->answer retrieval table from training data + frozen base model.

For each unique training prompt, embed it with the frozen base model
(mean-pooled final hidden, L2-normalized) and store the corresponding
chosen answer. Used by eval_cehri_retrieval.py for exact recall of
training-memorized answers (the episodic-memory pillar, done properly).

Usage:
  python3 build_retrieval.py [--data prajna/data/error_correction_pairs.json]
"""
import os, sys, json, argparse, time
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
sys.path.insert(0, os.path.dirname(__file__))
import torch
from crn_components import PrajnaStudentMultiLayer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="prajna/data/error_correction_pairs.json")
    ap.add_argument("--out", default="prajna/data/retrieval_table.npz")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    pairs = json.load(open(args.data))
    print(f"pairs: {len(pairs)}")

    # Dedup by prompt text (exam questions appear 100x with identical answers)
    uniq = {}
    for p in pairs:
        if p["prompt"] not in uniq:
            uniq[p["prompt"]] = p["chosen"]
    prompts = list(uniq.keys())
    answers = [uniq[k] for k in prompts]
    print(f"unique prompts: {len(prompts)}")

    t0 = time.time()
    student = PrajnaStudentMultiLayer(device=args.device, inject_every=4, max_length=96, crn_mix_init=2.0)
    student = student.to(args.device)
    student.eval()
    tok = student.tok
    print(f"model ready in {time.time()-t0:.0f}s")

    embs = []
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch):
            chunk = prompts[i:i + args.batch]
            enc = tok(chunk, truncation=True, max_length=64, padding=True, return_tensors="pt")
            ids = enc["input_ids"].to(args.device)
            mask = enc["attention_mask"].to(args.device)
            out = student.base_model(input_ids=ids, attention_mask=mask,
                                     output_hidden_states=True, return_dict=True)
            h = out.hidden_states[-1].float()                      # (B,T,D)
            pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, dim=-1) # (B,D)
            embs.append(pooled.cpu().half())                       # fp16 to halve size
            if (i // args.batch) % 25 == 0:
                print(f"  embedded {min(i+args.batch, len(prompts))}/{len(prompts)} ({time.time()-t0:.0f}s)", flush=True)

    emb = torch.cat(embs, dim=0)  # (N,D) fp16
    print(f"embeddings: {emb.shape} dtype={emb.dtype}")

    meta = {"answers": answers, "prompts": prompts}
    torch.save({"emb": emb, "meta": meta}, args.out)
    print(f"saved -> {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)")

if __name__ == "__main__":
    main()

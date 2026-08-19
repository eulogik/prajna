#!/usr/bin/env python3
"""Timing probe ONLY — measure real per-step cost for Prajna-2B training.
No checkpoints written. Uses only crn_components (does NOT import train_prajna2b)."""
import os, sys, json, time, random, torch, torch.nn.functional as F
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer

torch.set_num_threads(min(os.cpu_count() or 8, 8))
CKPT = './prajna/checkpoints/dpo_final.pt'
EC = './prajna/data/error_correction_pairs.json'
ML = 96
DPO_BETA = 0.1

print("Loading model...", flush=True)
t0 = time.time()
student = PrajnaStudentMultiLayer(device='cpu', inject_every=8, max_length=ML, crn_mix_init=0.05)
ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
print(f"  model load: {time.time()-t0:.1f}s", flush=True)

ec = json.load(open(EC))
tok = student.tok

def sft_pair(e):
    text = f"{e['prompt']}\n\n{e['chosen']}{tok.eos_token}"
    enc = tok(text, truncation=True, max_length=ML, padding='max_length', return_tensors='pt')
    ids = enc['input_ids']                       # keep 2D (B, ML) — Gemma-4 rejects 1D
    labels = ids.clone(); labels[enc['attention_mask'] == 0] = -100
    return ids, labels

def dpo_pair(e):
    c = tok(e['chosen'], truncation=True, max_length=ML, padding='max_length', return_tensors='pt')['input_ids']
    r = tok(e['rejected'], truncation=True, max_length=ML, padding='max_length', return_tensors='pt')['input_ids']
    return c, r

def get_logps(logits, labels):
    labels = labels[:, 1:].clone(); logits = logits[:, :-1]
    mask = labels != -100; labels[~mask] = 0
    lp = F.log_softmax(logits, -1); tl = torch.gather(lp, 2, labels.unsqueeze(2)).squeeze(2)
    return (tl * mask).sum(-1)

opt = torch.optim.AdamW(student.get_params(), lr=3e-4)
student.train()
random.shuffle(ec)

print("Timing 4 SFT steps...", flush=True)
sft_times = []
for i in range(4):
    ids, labels = sft_pair(ec[i])
    t = time.time()
    out = student(ids, labels)
    (out['loss'] / 8).backward(); opt.step(); opt.zero_grad()
    sft_times.append(time.time() - t)
    print(f"  SFT {i+1}: {sft_times[-1]:.1f}s (loss={out['loss'].item():.3f})", flush=True)

print("Timing 4 DPO steps...", flush=True)
dpo_times = []
for i in range(4):
    c, r = dpo_pair(ec[i+10])
    t = time.time()
    oc = student._collect_hidden(c); lc, _ = student._apply_crn(oc, training=True)
    orr = student._collect_hidden(r); lr, _ = student._apply_crn(orr, training=True)
    lpc = get_logps(lc, c); lpr = get_logps(lr, r)
    loss = -F.logsigmoid(DPO_BETA * (lpc - lpr)).mean()
    (loss / 8).backward(); opt.step(); opt.zero_grad()
    dpo_times.append(time.time() - t)
    print(f"  DPO {i+1}: {dpo_times[-1]:.1f}s (loss={loss.item():.3f})", flush=True)

sft_avg = sum(sft_times)/len(sft_times); dpo_avg = sum(dpo_times)/len(dpo_times)
print(f"\n=== REAL TIMING (CPU, external disk) ===")
print(f"  SFT/step: {sft_avg:.1f}s -> 500={sft_avg*500/60:.1f}min  2000={sft_avg*2000/60:.1f}min")
print(f"  DPO/step: {dpo_avg:.1f}s -> 200={dpo_avg*200/60:.1f}min  500={dpo_avg*500/60:.1f}min")
print(f"  PILOT (500 SFT+200 DPO) ~ {(sft_avg*500+dpo_avg*200)/3600:.1f} h")
print(f"  FULL  (2000 SFT+500 DPO+200 CON) ~ {(sft_avg*2000+dpo_avg*700)/3600:.1f} h")
print("DONE (probe, no checkpoints written)")

#!/usr/bin/env python3
"""Precise per-step timing probe for the v2 SFT loop (CPU). Prints each step time."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(__file__))
import torch
from crn_components import PrajnaStudentMultiLayer, get_crn_state_dict
from train_prajna2b import Student, ECSFTDataset, SEED_CKPT, EC_DATA, MAX_LENGTH, SFT_LR, INJECT_EVERY, CRN_MIX_INIT, DEVICE
from torch.utils.data import DataLoader

N = int(os.environ.get('PROBE_STEPS', '6'))
print(f"[probe] device={DEVICE} maxlen={MAX_LENGTH} steps={N}", flush=True)

t0 = time.time()
student = Student(device='cpu', inject_every=INJECT_EVERY, max_length=MAX_LENGTH, crn_mix_init=CRN_MIX_INIT)
student = student.to(DEVICE)
ckpt = torch.load(SEED_CKPT, map_location=DEVICE, weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
print(f"[probe] model built+seeded in {time.time()-t0:.1f}s", flush=True)

ds = ECSFTDataset(EC_DATA, student.tok, MAX_LENGTH)
loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0)
params = student.get_params()
n_trainable = sum(p.numel() for p in params if p.requires_grad)
print(f"[probe] trainable params={n_trainable:,}", flush=True)
opt = torch.optim.AdamW(params, lr=SFT_LR)
student.train()

times = []
for i, batch in enumerate(loader):
    if i >= N: break
    st = time.time()
    ids = batch['input_ids'].to(DEVICE); labels = batch['labels'].to(DEVICE)
    out = student(ids, labels)
    loss = out['loss']
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); opt.zero_grad()
    dt = time.time() - st
    times.append(dt)
    print(f"[probe] step {i+1}/{N}  {dt:.1f}s  loss={loss.item():.4f}", flush=True)

warm = times[1:] if len(times) > 1 else times
avg = sum(warm)/len(warm)
print(f"\n[probe] avg (excl. first) = {avg:.1f}s/step", flush=True)
for label, steps in [("SFT 2000", 2000), ("DPO 500", 500), ("CON 200", 200), ("TOTAL 2700", 2700)]:
    print(f"[probe] {label:12s} ~ {avg*steps/3600:.1f} h", flush=True)

#!/usr/bin/env python3
"""Batch-scan teacher_data.json on MPS; keep only samples whose full CRN
forward (base+CRN, float16) is finite. General text can overflow MPS float16
attention on some samples -> NaN; math data doesn't. Batching for speed."""
import torch, json, sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['HF_HOME'] = '/Volumes/KIOXIA 1TB/huggingface_cache/'

m = PrajnaStudentMultiLayer(device='mps', inject_every=8, max_length=96,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4, num_corrections=8, mem_size=256, mem_dim=64)
m = m.to('mps'); m.eval()
tok = m.tok
ck = torch.load('prajna/checkpoints/_seed_dpo_s0.pt', map_location='cpu', weights_only=False)
m.load_state_dict(ck['crn'], strict=False)

data = json.load(open('prajna/data/teacher_data.json'))
BATCH = 128
keep = []
t0 = time.time()
for s in range(0, len(data), BATCH):
    chunk = data[s:s+BATCH]
    texts, meta = [], []
    for c in chunk:
        p, r = c.get('prompt', ''), c.get('response', '')
        if not p.strip() or not r.strip():
            continue
        texts.append(f"{p} {r}{tok.eos_token}")
        meta.append(c)
    if not texts:
        continue
    enc = tok(texts, truncation=True, max_length=96, padding='max_length', return_tensors='pt')
    ids = enc['input_ids'].to('mps')
    try:
        with torch.no_grad():
            out = m._collect_hidden(ids)
            logits, _ = m._apply_crn(out, training=False)
        bad = torch.isnan(logits).any(dim=-1).any(dim=-1) | torch.isinf(logits).any(dim=-1).any(dim=-1)
        for c, b in zip(meta, bad.tolist()):
            if not b:
                keep.append({'prompt': c['prompt'], 'response': c['response']})
    except Exception:
        pass
    if (s + BATCH) % 2000 < BATCH:
        print(f"  scanned {min(s+BATCH,len(data))}/{len(data)} kept {len(keep)} ({time.time()-t0:.0f}s)", flush=True)
json.dump(keep, open('prajna/data/teacher_data_clean.json', 'w'))
print(f"DONE kept {len(keep)}/{len(data)} -> prajna/data/teacher_data_clean.json ({time.time()-t0:.0f}s)", flush=True)

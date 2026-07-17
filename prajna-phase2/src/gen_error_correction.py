#!/usr/bin/env python3
"""Generate error-correction dataset for Prajna-2B Phase 1.

Run the FROZEN base Gemma-4-E2B on prompts, compare to ground truth, keep only
cases where the base is WRONG, and emit DPO pairs: chosen=full correct response,
rejected=full base-wrong response. The CRN trains to prefer correct over the
base's actual mistakes.

Sources:
  - math_v1_fresh.json (84K): arithmetic (answer in 'response')
  - facts_cot.json (196): factual QA
  - igr_cot.json (1820): implicit-goal reasoning

We cap at N samples per domain and only keep wrong base outputs.
"""
import os, sys, json, random, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(7)
torch.set_num_threads(min(os.cpu_count() or 8, 8))

N_PER = int(os.environ.get('EC_N', '3000'))
OUT = os.environ.get('EC_OUT', './prajna/data/error_correction_pairs.json')
MAX_NEW = 48

print("Loading base Gemma-4-E2B (frozen, no CRN)...")
tok = AutoTokenizer.from_pretrained('google/gemma-4-E2B')
base = AutoModelForCausalLM.from_pretrained('google/gemma-4-E2B', dtype=torch.float16, low_cpu_mem_usage=False)
base.eval()

def gen_base(prompt):
    enc = tok(prompt, return_tensors='pt')
    ids = enc.input_ids
    with torch.no_grad():
        out = base.generate(ids, attention_mask=enc.attention_mask,
                             max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

def base_answer_correct(prompt, gold):
    """Return (is_correct, base_output)."""
    out = gen_base(prompt)
    # crude: gold substring in output (case-insensitive)
    ok = gold.strip().lower() in out.strip().lower()
    return ok, out

pairs = []
sources = [
    ('./prajna/data/math_v1_fresh.json', 'math'),
    ('./prajna/data/facts_cot.json', 'facts'),
    ('./prajna/data/igr_cot.json', 'igr'),
]
for path, domain in sources:
    with open(path) as f:
        data = json.load(f)
    random.shuffle(data)
    seen = 0; kept = 0
    for s in data:
        if kept >= N_PER: break
        prompt = s.get('prompt', '')
        gold = s.get('response', '') or s.get('answer', '')
        if not prompt or not gold: continue
        seen += 1
        try:
            ok, bout = base_answer_correct(prompt, gold)
        except Exception:
            continue
        if not ok:
            # base is wrong -> keep as a correction pair
            pairs.append({
                'prompt': prompt,
                'chosen': gold.strip(),           # correct answer
                'rejected': bout,                  # base's wrong output
                'domain': domain,
            })
            kept += 1
        if seen % 200 == 0:
            print(f"  {domain}: scanned {seen}, kept {kept}", flush=True)
    print(f"DONE {domain}: scanned {seen}, kept {kept}", flush=True)

print(f"\nTotal error-correction pairs: {len(pairs)}")
with open(OUT, 'w') as f:
    json.dump(pairs, f, indent=1)
print(f"Saved -> {OUT}")

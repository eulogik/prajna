#!/usr/bin/env python3
"""Benchmark OUR 2B+CRN model and the 2B base on standard suites, to compare
against PUBLISHED 7B/9B/20B numbers (we don't run those locally).

Tasks (multiple-choice are fast on CPU; car-wash is the viral common-sense test):
  - MMLU  (sampled)  -> knowledge
  - BoolQ (sampled)  -> comprehension
  - HellaSwag(sampled)-> commonsense completion
  - GSM8K (tiny)     -> grade-school math
  - Car-wash / IGR   -> implicit-goal reasoning (the "76% of models fail" test)

Our model uses FULL recompute (CRN attends over the whole sequence; incremental
decode is incorrect). Saves ./prajna/bench_standard.json
"""
import os, sys, json, re, random, torch, math
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

random.seed(7)
torch.set_num_threads(min(os.cpu_count() or 8, 8))

N = int(os.environ.get('BENCH_N', '200'))

def load_prajna():
    from crn_components import PrajnaStudentMultiLayer
    m = PrajnaStudentMultiLayer(device='cpu'); m = m.to('cpu')
    ckpt = torch.load('./prajna/checkpoints/dpo_final.pt', map_location='cpu', weights_only=False)
    m.load_state_dict(ckpt['crn'], strict=False)
    if os.path.exists('./prajna/checkpoints/memory_dpo_final.json'):
        m.load_memory('./prajna/checkpoints/memory_dpo_final.json')
    m.eval(); tok = m.tok
    @torch.no_grad()
    def gen(prompt, max_new=12):
        ids = tok(prompt, return_tensors='pt').input_ids
        g = ids.clone()
        for _ in range(max_new):
            o = m._collect_hidden(g)
            logits, _ = m._apply_crn(o, training=False)
            nt = logits[:, -1, :].argmax(-1).reshape(1, 1)
            g = torch.cat([g, nt], dim=1)
            if nt.item() == tok.eos_token_id: break
        return tok.decode(g[0], skip_special_tokens=True)[len(prompt):].strip()
    return gen

def load_hf(name):
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16, low_cpu_mem_usage=False)
    tok = AutoTokenizer.from_pretrained(name); m.eval()
    @torch.no_grad()
    def gen(prompt, max_new=12):
        ids = tok(prompt, return_tensors='pt').input_ids
        out = m.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return gen

def load_lora(base_name, adapter_path):
    from peft import PeftModel
    m = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float16, low_cpu_mem_usage=False)
    m = PeftModel.from_pretrained(m, adapter_path)
    tok = AutoTokenizer.from_pretrained(base_name); m.eval()
    @torch.no_grad()
    def gen(prompt, max_new=12):
        ids = tok(prompt, return_tensors='pt').input_ids
        out = m.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return gen

def first_letter(t):
    m = re.search(r'\b([A-D])\b', t)
    return m.group(1) if m else (t.strip()[:1].upper() if t.strip() else '')

def yesno(t):
    tl = t.lower()
    if 'yes' in tl and 'no' not in tl: return 'yes'
    if 'no' in tl and 'yes' not in tl: return 'no'
    m = re.search(r'\b(yes|no)\b', tl)
    return m.group(1) if m else ''

def run_mc(gen, rows, fmt, parse, gold_of, max_new=12):
    ok = 0
    for i, r in enumerate(rows):
        if i >= N: break
        prompt = fmt(r)
        pred = parse(gen(prompt, max_new=max_new))
        if pred == gold_of(r): ok += 1
    return ok, min(N, len(rows))

# ---------- formatters ----------
def mmlu_fmt(r):
    c = r['choices']; return (f"Question: {r['question']}\n"
        f"A. {c[0]}\nB. {c[1]}\nC. {c[2]}\nD. {c[3]}\nAnswer:")
def _letter(a):
    if isinstance(a, str):
        try: return 'ABCD'[int(a)]
        except ValueError: return a.strip().upper()
    return "ABCD"[a]
def mmlu_gold(r): return _letter(r['answer'])

def boolq_fmt(r):
    return f"{r['passage']}\nQuestion: {r['question']}\nAnswer (Yes or No):"
def boolq_gold(r): return 'yes' if r['answer'] else 'no'

def hs_fmt(r):
    e = r['endings']; return (f"{r['ctx']}\n"
        f"A. {e[0]}\nB. {e[1]}\nC. {e[2]}\nD. {e[3]}\nAnswer:")
def hs_gold(r): return _letter(r['label'])

def gsm_fmt(r): return f"Question: {r['question']}\nAnswer:"
def gsm_gold(r):
    m = re.search(r'####\s*([\-0-9,]+)', r['answer'])
    return m.group(1).replace(',', '') if m else ''
def gsm_parse(t):
    m = re.search(r'####\s*([\-0-9,]+)', t)
    if m: return m.group(1).replace(',', '')
    m = re.search(r'([\-0-9,]+)', t)
    return m.group(1).replace(',', '') if m else ''

CARWASH = [
    ("I want to wash my car. The car wash is 100 meters away. Should I walk or drive?", "drive", "car"),
    ("My cat is ill and must see the vet. The clinic is 200 meters away. Should I walk the cat or carry it?", "carry", "ill"),
    ("My car is out of fuel and the station is 150 meters away. Should I walk or drive there?", "drive", "fuel"),
    ("My clothes are dirty and the laundromat is around the corner. Should I wear them or bring them?", "bring", "dirty"),
    ("I need groceries at home. The shop is 100 meters away. Should I walk there or order delivery?", "deliver", "home"),
    ("My bike chain broke and the shop is 200 meters away. Should I ride or walk it?", "walk", "broken"),
    ("I want to boat. The ramp is 100 meters away. Should I swim or tow the boat?", "tow", "boat"),
    ("My EV is low on charge and the charger is 100 meters away. Should I walk or drive?", "drive", "charge"),
]

def run():
    res = {}
    mmlu = list(load_dataset("cais/mmlu", "all", split="test", streaming=False).shuffle(seed=7))
    boolq = list(load_dataset("google/boolq", split="validation", streaming=False).shuffle(seed=7))
    hs = list(load_dataset("Rowan/hellaswag", split="validation", streaming=False).shuffle(seed=7))
    try:
        gsm = list(load_dataset("gsm8k", "main", split="test", streaming=False).shuffle(seed=7))
    except Exception as e:
        gsm = []; print("gsm8k unavailable:", e)
    for label, loader, kind in [("Prajna 2B+CRN", load_prajna, 'prajna'),
                                ("Gemma-4-E2B (base)", lambda: load_hf('google/gemma-4-E2B'), 'hf'),
                                ("LoRA baseline (r=19)", lambda: load_lora('google/gemma-4-E2B', './prajna/checkpoints/lora_baseline_dpo'), 'lora')]:
        try:
            gen = loader()
            o, t = run_mc(gen, mmlu, mmlu_fmt, first_letter, mmlu_gold); res.setdefault(label, {})['mmlu'] = f"{o}/{t}"
            o, t = run_mc(gen, boolq, boolq_fmt, yesno, boolq_gold); res[label]['boolq'] = f"{o}/{t}"
            o, t = run_mc(gen, hs, hs_fmt, first_letter, hs_gold); res[label]['hellaswag'] = f"{o}/{t}"
            if gsm:
                ok = sum(1 for r in gsm[:int(os.environ.get('GSM_N','20'))]
                         if gsm_parse(gen(gsm_fmt(r), max_new=64)) == gsm_gold(r))
                res[label]['gsm8k'] = f"{ok}/{int(os.environ.get('GSM_N','20'))}"
            cw = sum(1 for q, a, g in CARWASH
                     if a in gen(q, max_new=140).lower() and g in gen(q, max_new=140).lower())
            res[label]['carwash'] = f"{cw}/{len(CARWASH)}"
            print(f"[done] {label}: {res[label]}")
            del gen
            import gc; gc.collect()
        except Exception as e:
            res[label] = {'error': repr(e)[:200]}; print(f"[FAIL] {label}: {repr(e)[:200]}")
    print("\n===== STANDARD BENCHMARK (ours + 2B base) =====")
    for k, v in res.items():
        print(k, v)
    with open('./prajna/bench_standard.json', 'w') as f:
        json.dump(res, f, indent=2)

if __name__ == '__main__':
    run()

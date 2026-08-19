#!/usr/bin/env python3
"""Benchmark Prajna (2B base + CRN adapter) against larger base models.

Two fair comparisons:
  1. BITS-PER-BYTE on a SHARED raw text (tokenizer-independent) -> perplexity.
  2. TASK ACCURACY on held-out math / facts / car-wash(IGR) tests.

Models compared:
  - Prajna 2B+CRN (gemma-4-E2B + dpo_final.pt + memory)
  - Gemma-4-E2B (base, no CRN)         [same tokenizer -> clean PPL vs CRN]
  - Gemma-2-7B  (4bit)                 [cross-size, downloaded if needed]
  - Gemma-2-9B  (4bit)                 [cross-size]
  - Llama-3.1-8B (4bit)                [cross-size, 7B class]

Usage: python3 bench_against.py
"""
import os, sys, json, re, random, math, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(123)
LN2 = math.log(2)

# ---------------- held-out tests (NOT from training files) ----------------
def math_problems(n=30):
    ops = [('+', lambda a,b:a+b), ('-', lambda a,b:a-b), ('*', lambda a,b:a*b),
           ('/', lambda a,b: (a*b, b))]
    out = []
    for _ in range(n):
        a, b = random.randint(100,999), random.randint(2,99)
        sym, fn = random.choice(ops)
        if sym == '/':
            q, ans = f"What is {a*b} / {b}?", str(a)
        else:
            q, ans = f"What is {a} {sym} {b}?", str(fn(a,b))
        out.append((q, ans))
    return out

FACTS = [
    ("What is the capital of Italy?", "rome"),
    ("What is the capital of Brazil?", "brasilia"),
    ("What is the capital of Egypt?", "cairo"),
    ("Who painted the Mona Lisa?", "leonardo da vinci"),
    ("Who developed the theory of relativity?", "albert einstein"),
    ("What is the chemical symbol for iron?", "fe"),
    ("What is the chemical symbol for oxygen?", "o"),
    ("What year did World War I end?", "1918"),
    ("What planet is known as the Red Planet?", "mars"),
    ("Who wrote '1984'?", "george orwell"),
    ("What is the largest planet in the Solar System?", "jupiter"),
    ("What is the square root of 144?", "12"),
    ("Who discovered penicillin?", "fleming"),
    ("What is the freezing point of water in Celsius?", "0"),
    ("Who was the first person to walk on the Moon?", "armstrong"),
    ("What gas do plants absorb for photosynthesis?", "carbon dioxide"),
    ("What is the capital of Australia?", "canberra"),
    ("Which element has atomic number 79?", "gold"),
    ("Who composed the Ninth Symphony?", "beethoven"),
    ("What is the value of pi to two decimals?", "3.14"),
]

IGR = [
    ("I want to wash my car. The car wash is 100 meters away. Should I walk or drive?", "drive", "car"),
    ("My cat is ill and must see the vet. The clinic is 200 meters away. Should I walk the cat or carry it?", "carry", "ill"),
    ("My car is out of fuel and the station is 150 meters away. Should I walk or drive there?", "drive", "fuel"),
    ("My clothes are dirty and the laundromat is around the corner. Should I wear them or bring them?", "bring", "dirty"),
    ("I need groceries at home. The shop is 100 meters away. Should I walk there or order delivery?", "deliver", "home"),
    ("My bike chain broke and the shop is 200 meters away. Should I ride or walk it?", "walk", "broken"),
    ("I want to boat. The ramp is 100 meters away. Should I swim or tow the boat?", "tow", "boat"),
    ("My EV is low on charge and the charger is 100 meters away. Should I walk or drive?", "drive", "charge"),
]

def parse_math(t):
    m = re.search(r'answer is[ :]*([\-0-9]+)', t, re.I)
    if m: return m.group(1)
    nums = re.findall(r'[\-0-9]+', t)
    return nums[-1] if nums else ''

def score_math(text, ans):
    return parse_math(text).strip() == ans.strip()

def score_facts(text, ans):
    return ans in text.lower()

def score_igr(text, action, goal):
    low = text.lower()
    return (action in low) and (goal in low)

# ---------------- model wrappers ----------------
def load_prajna():
    from crn_components import PrajnaStudentMultiLayer
    base = PrajnaStudentMultiLayer(device='cpu')
    base = base.to('cpu')
    ckpt = torch.load('./prajna/checkpoints/dpo_final.pt', map_location='cpu', weights_only=False)
    base.load_state_dict(ckpt['crn'], strict=False)
    if os.path.exists('./prajna/checkpoints/memory_dpo_final.json'):
        base.load_memory('./prajna/checkpoints/memory_dpo_final.json')
    base.eval(); tok = base.tok
    @torch.no_grad()
    def gen(prompt, max_new=160):
        # FULL recompute each step (CRN's resonance/skills attend over the whole
        # sequence, so incremental decoding would give wrong corrections).
        ids = tok(prompt + "\n", return_tensors='pt').input_ids
        g = ids.clone()
        for _ in range(max_new):
            o = base._collect_hidden(g)
            logits, _ = base._apply_crn(o, training=False)
            nt = logits[:, -1, :].argmax(-1).reshape(1, 1)
            g = torch.cat([g, nt], dim=1)
            if nt.item() == tok.eos_token_id: break
        return tok.decode(g[0], skip_special_tokens=True)[len(prompt):].strip()
    @torch.no_grad()
    def bpb(text):
        enc = tok(text, return_tensors='pt')
        ids = enc.input_ids
        o = base._collect_hidden(ids); logits, _ = base._apply_crn(o, training=False)
        logp = torch.log_softmax(logits[:, :-1], -1)
        tgt = ids[:, 1:]
        nll = -logp.gather(2, tgt.unsqueeze(2)).sum()
        ntok = tgt.numel()
        bits = nll.item() / LN2
        return bits / ntok / (len(text) / ntok)  # bits per byte
    return gen, bpb

def load_hf(name, kwargs):
    m = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    tok = AutoTokenizer.from_pretrained(name)
    m.eval()
    @torch.no_grad()
    def gen(prompt, max_new=160):
        ids = tok(prompt + "\n", return_tensors='pt').input_ids
        out = m.generate(ids, max_new_tokens=max_new, do_sample=False,
                         pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    @torch.no_grad()
    def bpb(text):
        enc = tok(text, return_tensors='pt')
        ids = enc.input_ids
        logits = m(ids).logits
        logp = torch.log_softmax(logits[:, :-1], -1)
        tgt = ids[:, 1:]
        nll = -logp.gather(2, tgt.unsqueeze(2)).sum()
        ntok = tgt.numel()
        bits = nll.item() / LN2
        return bits / ntok / (len(text) / ntok)
    return gen, bpb

# ---------------- run ----------------
def run():
    raw = ("The transformer architecture revolutionized natural language processing by "
           "using self-attention to weigh the importance of each token relative to others. "
           "Training such models requires large corpora and substantial compute. Reasoning "
           "tasks remain challenging because they demand implicit goal inference beyond surface patterns.")
    specs = [
        ("Prajna 2B+CRN", "prajna", None),
        ("Gemma-4-E2B (base)", "hf", ("google/gemma-4-E2B", {})),
        ("Gemma-2-7B (4bit)", "hf", ("google/gemma-2-7b", {"load_in_4bit": True})),
        ("Gemma-2-9B (4bit)", "hf", ("google/gemma-2-9b", {"load_in_4bit": True})),
        ("Llama-3.1-8B (4bit)", "hf", ("meta-llama/Llama-3.1-8B", {"load_in_4bit": True})),
    ]
    if os.environ.get('BENCH_LOCAL'):
        specs = [s for s in specs if s[1] in ('prajna',) or 'E2B' in s[0]]
    results = {}
    for label, kind, arg in specs:
        try:
            if kind == "prajna":
                gen, bpb = load_prajna()
            else:
                name, kw = arg
                gen, bpb = load_hf(name, kw)
            mp = math_problems(15)
            m_ok = sum(score_math(gen(q, max_new=48), a) for q, a in mp)
            f_ok = sum(score_facts(gen(q, max_new=24), a) for q, a in FACTS)
            i_ok = sum(score_igr(gen(q, max_new=140), act, goal) for q, act, goal in IGR[:6])
            p = bpb(raw)
            results[label] = {
                'bpb': round(p, 4),
                'math': f"{m_ok}/{len(mp)}", 'math_pct': round(m_ok/len(mp)*100),
                'facts': f"{f_ok}/{len(FACTS)}", 'facts_pct': round(f_ok/len(FACTS)*100),
                'igr': f"{i_ok}/{len(IGR)}", 'igr_pct': round(i_ok/len(IGR)*100),
            }
            print(f"[done] {label}: bpb={p:.4f} math={results[label]['math']} facts={results[label]['facts']} igr={results[label]['igr']}")
            del gen, bpb; import gc; gc.collect()
            if kind == "hf": del m
            gc.collect()
        except Exception as e:
            results[label] = {'error': repr(e)[:160]}
            print(f"[FAIL] {label}: {repr(e)[:160]}")
    print("\n===== BENCHMARK SUMMARY =====")
    print(f"{'model':22} {'bpb':>7} {'math':>7} {'facts':>7} {'igr':>7}")
    for k, v in results.items():
        if 'error' in v:
            print(f"{k:22} ERROR: {v['error']}")
        else:
            print(f"{k:22} {v['bpb']:>7} {v['math_pct']:>5}% {v['facts_pct']:>5}% {v['igr_pct']:>5}%")
    with open('./prajna/bench_result.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    run()

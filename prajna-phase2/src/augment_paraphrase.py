#!/usr/bin/env python3
"""Paraphrase augmenter — Tier-2 semantic-memory data.

Deterministic, answer-preserving surface transforms. Two DISJOINT transform sets:

  Set A (training): 4 variants per training pair -> error_correction_pairs_v2.json
                    (used to train the CRN + build the v2 retrieval table)
  Set B (eval):     2 variants per exam question -> cehri_exam_reworded.json
                    (held-out reworded exam, NEVER in training data or table)

Disjointness is by construction (different templates) and asserted at the end.

Usage: python3 augment_paraphrase.py
"""
import json, re, os

DATA_DIR = 'prajna/data'
EC_DATA = f'{DATA_DIR}/error_correction_pairs.json'
EXAM = f'{DATA_DIR}/cehri_exam.json'
OUT_EC = f'{DATA_DIR}/error_correction_pairs_v2.json'
OUT_EXAM = f'{DATA_DIR}/cehri_exam_reworded.json'

MATH_RE = re.compile(r'(?i)^(what is|what\'s|compute|calculate)\s+(.*\d.*)\??$')

# ---------- Set A: training augmentation (4 variants) ----------
def aug_math(expr, orig):
    vs = [
        f'Compute {expr}.',
        f'{expr} equals what?',
        f'What does {expr} evaluate to?',
        f'Tell me the value of {expr}.',
    ]
    return [v for v in vs if v != orig]

def aug_question(q, orig):
    vs = [
        f'Can you tell me: {q}',
        f'Do you know: {q}',
        f'Hey, {q}',
        f'Please answer: {q}',
    ]
    return [v for v in vs if v != orig]

def aug_igr(s, orig):
    vs = [
        f'Imagine: {s}',
        f'Suppose {s}',
        f'Situation: {s}',
        f'{s} What would you do?',
    ]
    return [v for v in vs if v != orig]

def augment_set_a(prompt):
    if prompt.endswith('?') and not prompt.endswith('?"'):
        m = MATH_RE.match(prompt)
        if m:
            return aug_math(m.group(2).strip(), prompt)
        return aug_question(prompt, prompt)
    return aug_igr(prompt, prompt)

# ---------- Set B: held-out reworded exam (2 variants, disjoint from A) ----------
def reword_math(expr, orig):
    return [f'Calculate {expr}.', f'{expr}?']

def reword_question(q, orig):
    return [f'Can you answer: {q}', f'Respond to this: {q}']

def reword_igr(s, orig):
    return [f'Consider: {s}', f'{s} What is the right move?']

def reword_set_b(prompt):
    if prompt.endswith('?'):
        m = MATH_RE.match(prompt)
        if m:
            return reword_math(m.group(2).strip(), prompt)
        return reword_question(prompt, prompt)
    return reword_igr(prompt, prompt)

def main():
    pairs = json.load(open(EC_DATA))
    print(f'base pairs: {len(pairs)}')

    out, n_variants, n_orig = [], 0, 0
    for aid, p in enumerate(pairs):
        variants = augment_set_a(p['prompt'])[:4]
        out.append({**p, 'aid': aid, 'variant': 'orig'})
        n_orig += 1
        for i, v in enumerate(variants, 1):
            out.append({**p, 'aid': aid, 'prompt': v, 'variant': f'a{i}'})
            n_variants += 1
    json.dump(out, open(OUT_EC, 'w'), indent=1)
    print(f'train v2: {len(out)} pairs ({n_orig} orig + {n_variants} variants) -> {OUT_EC}')

    train_prompts = {p['prompt'] for p in out}

    exam = json.load(open(EXAM))
    reworded, n_checks = [], 0
    for q in exam:
        for i, v in enumerate(reword_set_b(q['prompt']), 1):
            assert v not in train_prompts, f'leak: B-variant in training data: {v!r}'
            reworded.append({**q, 'id': f'{q["id"]}.r{i}', 'prompt': v, 'variant': f'b{i}'})
            n_checks += 1
    json.dump(reworded, open(OUT_EXAM, 'w'), indent=1)
    print(f'reworded exam: {len(reworded)} items (60x2, all held out) -> {OUT_EXAM}')

    # hard disjointness check: B-variants may not equal any training prompt
    leaked = [r['prompt'] for r in reworded if r['prompt'] in train_prompts]
    assert not leaked, f'LEAKS: {leaked}'
    print('disjointness OK: no reworded-exam prompt appears in training data')

if __name__ == '__main__':
    main()

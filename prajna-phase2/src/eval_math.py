#!/usr/bin/env python3
"""Math evaluation: 100 problems, exact-match on parsed final answer.

Compares CRN model (SFT+DPO checkpoint) vs base Gemma 4 E2B.
Usage: python3 eval_math.py [checkpoint_path]
"""
import os, sys, json, re, random, time, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer

CKPT = sys.argv[1] if len(sys.argv) > 1 else './prajna/checkpoints/dpo_final.pt'
PER_OP = int(sys.argv[2]) if len(sys.argv) > 2 else 8  # problems per operation
MEM = './prajna/checkpoints/memory_dpo_final.json'
DEVICE = 'cpu'
MAX_NEW = 48  # room for CoT (greedy, shorter for speed)

# ---- test problems (not in training template seed necessarily) ----
def gen_problems():
    probs = []
    # addition
    for _ in range(PER_OP):
        a, b = random.randint(100, 999), random.randint(100, 999)
        probs.append((f"What is {a} + {b}?", str(a + b), 'add'))
    # subtraction
    for _ in range(PER_OP):
        a, b = random.randint(100, 999), random.randint(1, a)
        probs.append((f"What is {a} - {b}?", str(a - b), 'sub'))
    # multiplication
    for _ in range(PER_OP):
        a, b = random.randint(11, 99), random.randint(11, 99)
        probs.append((f"What is {a} * {b}?", str(a * b), 'mul'))
    # division
    for _ in range(PER_OP):
        b = random.randint(11, 99); q = random.randint(11, 99); r = random.randint(0, b-1)
        probs.append((f"What is {b*q+r} / {b}?", str(q), 'div'))
    # powers
    for _ in range(PER_OP):
        a, b = random.randint(2, 15), random.randint(2, 7)
        probs.append((f"What is {a}^{b}?", str(a**b), 'pow'))
    return probs

def parse_answer(text):
    # Look for "answer is X" or last number
    m = re.search(r'answer is[ :]*([\-0-9]+)', text, re.IGNORECASE)
    if m: return m.group(1)
    nums = re.findall(r'[\-0-9]+', text)
    return nums[-1] if nums else ''

@torch.no_grad()
def generate_crn(student, ids):
    gen = ids.clone()
    for _ in range(MAX_NEW):
        out = student._collect_hidden(gen)
        logits, _ = student._apply_crn(out, training=False)
        nt = logits[:, -1, :].argmax(-1).reshape(1, 1)  # greedy for speed
        gen = torch.cat([gen, nt], dim=1)
        if nt.item() == student.tok.eos_token_id: break
    return gen

@torch.no_grad()
def generate_base(student, ids):
    return student.base_model.generate(ids, max_new_tokens=MAX_NEW,
                                       do_sample=False)  # greedy, fast + fair

def main():
    random.seed(123)
    probs = gen_problems()
    student = PrajnaStudentMultiLayer(device=DEVICE, inject_every=8, max_length=96,
        num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
        num_corrections=8, mem_size=256, mem_dim=64)
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ckpt['crn'], strict=False)
    if os.path.exists(MEM): student.load_memory(MEM)
    student.eval()
    tok = student.tok
    eos = tok.eos_token or '</s>'

    crn_correct = base_correct = 0
    by_op_crn = {}; by_op_base = {}
    print(f"\n{'OP':6} {'CRN':>6} {'BASE':>6}  example")
    for prompt, answer, op in probs:
        ids = tok(prompt, return_tensors='pt').input_ids
        gc = generate_crn(student, ids)
        gb = generate_base(student, ids)
        crn_text = tok.decode(gc[0], skip_special_tokens=True)[len(prompt):].strip()
        base_text = tok.decode(gb[0], skip_special_tokens=True)[len(prompt):].strip()
        crn_ans = parse_answer(crn_text)
        base_ans = parse_answer(base_text)
        crn_ok = crn_ans == answer
        base_ok = base_ans == answer
        crn_correct += crn_ok; base_correct += base_ok
        by_op_crn.setdefault(op, [0,0]); by_op_base.setdefault(op, [0,0])
        by_op_crn[op][0] += crn_ok; by_op_crn[op][1] += 1
        by_op_base[op][0] += base_ok; by_op_base[op][1] += 1
        if by_op_crn[op][1] <= 1:  # print first example per op
            print(f"{op:6}  CRN:{crn_ans:>6} BASE:{base_ans:>6}  Q:{prompt} A:{answer}")
            print(f"        CRN: {crn_text[:80]}")
            print(f"        BASE: {base_text[:80]}")

    n = len(probs)
    print(f"\n=== MATH ACCURACY (n={n}) ===")
    print(f"  CRN:  {crn_correct/n*100:.1f}%")
    print(f"  BASE: {base_correct/n*100:.1f}%")
    for op in by_op_crn:
        c, t = by_op_crn[op]; b, _ = by_op_base[op]
        print(f"  {op:5}: CRN {c/t*100:4.0f}%  BASE {b/t*100:4.0f}%")

if __name__ == '__main__':
    main()

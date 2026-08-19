#!/usr/bin/env python3
"""Fast CRN-only math eval (no base model) — prints the number that matters."""
import os, sys, json, re, random, torch
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
sys.path.insert(0, os.path.dirname(__file__))
from crn_components import PrajnaStudentMultiLayer

CKPT = sys.argv[1] if len(sys.argv) > 1 else './prajna/checkpoints/dpo_final.pt'
PER_OP = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEP = (len(sys.argv) > 3 and sys.argv[3] == '1')
MEM = './prajna/checkpoints/memory_dpo_final.json'
DEVICE = os.environ.get('CRN_DEVICE', 'cpu')
MAX_NEW = 96

def gen_problems():
    probs = []
    for _ in range(PER_OP):
        a, b = random.randint(100,999), random.randint(100,999)
        probs.append((f"What is {a} + {b}?", str(a+b), 'add'))
    for _ in range(PER_OP):
        a, b = random.randint(100,999), random.randint(1,a)
        probs.append((f"What is {a} - {b}?", str(a-b), 'sub'))
    for _ in range(PER_OP):
        a, b = random.randint(11,99), random.randint(11,99)
        probs.append((f"What is {a} * {b}?", str(a*b), 'mul'))
    for _ in range(PER_OP):
        b = random.randint(11,99); q = random.randint(11,99); r = random.randint(0,b-1)
        probs.append((f"What is {b*q+r} / {b}?", str(q), 'div'))
    for _ in range(PER_OP):
        a, b = random.randint(2,15), random.randint(2,7)
        probs.append((f"What is {a}^{b}?", str(a**b), 'pow'))
    return probs

def parse_answer(text):
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
        nt = logits[:, -1, :].argmax(-1).reshape(1, 1)
        gen = torch.cat([gen, nt], dim=1)
        if nt.item() == student.tok.eos_token_id: break
    return gen

random.seed(123)
probs = gen_problems()
student = PrajnaStudentMultiLayer(device=DEVICE, inject_every=8, max_length=96,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4, num_corrections=8, mem_size=256, mem_dim=64)
student = student.to(DEVICE)
ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
if os.path.exists(MEM): student.load_memory(MEM)
student.eval()
tok = student.tok

correct = 0
by_op = {}
print(f"\n{'OP':6} {'CRN':>6}  example")
for prompt, answer, op in probs:
    pp = prompt + "\n\n" if SEP else prompt
    ids = tok(pp, return_tensors='pt').input_ids.to(DEVICE)
    gc = generate_crn(student, ids)
    crn_text = tok.decode(gc[0].cpu(), skip_special_tokens=True)[len(prompt):].strip()
    crn_ans = parse_answer(crn_text)
    ok = crn_ans == answer
    correct += ok
    by_op.setdefault(op, [0,0]); by_op[op][0] += ok; by_op[op][1] += 1
    if by_op[op][1] <= 1:
        print(f"{op:6}  CRN:{crn_ans:>6}  Q:{prompt} A:{answer}")
        print(f"        {crn_text[:90]}")
n = len(probs)
print(f"\n=== CRN MATH ACCURACY (n={n}) ===  {correct/n*100:.1f}%")
for op in by_op:
    c,t = by_op[op]
    print(f"  {op:5}: {c/t*100:4.0f}%")

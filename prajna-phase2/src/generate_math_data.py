#!/usr/bin/env python3
"""Generate exact math chain-of-thought (CoT) training data from templates.

Answers are computed in Python (guaranteed correct). CoT format teaches the
model to show work, which 2B models need for arithmetic.
"""
import json, random

random.seed(42)
OUT = './prajna/data/math_cot.json'
N = 15000  # total samples

# ---- CoT generators (return prompt, response) ----
def add(a, b):
    s = a + b
    return (f"What is {a} + {b}?",
            f"We compute {a} + {b}. {a} + {b} = {s}. The answer is {s}.")

def sub(a, b):
    s = a - b
    return (f"What is {a} - {b}?",
            f"We compute {a} - {b}. {a} - {b} = {s}. The answer is {s}.")

def mul(a, b):
    s = a * b
    t1, u1 = a // 10, a % 10
    t2, u2 = b // 10, b % 10
    # Place-value split of ONE operand: each product is a 2-digit x 1-digit
    # (or 2-digit x 2-digit-with-trailing-zero) — easy, and phrased distinctly
    # from add/sub so the model does not conflate the schemas.
    if t1 > 0 and t2 > 0:
        p1, p2 = a * (t2 * 10), a * u2
        return (f"What is {a} * {b}?",
                f"We compute {a} * {b}. Split {b} into {t2*10} + {u2}. "
                f"{a} * {t2*10} = {p1}. {a} * {u2} = {p2}. "
                f"{p1} + {p2} = {s}. The answer is {s}.")
    # one operand single-digit -> direct product
    return (f"What is {a} * {b}?",
            f"We compute {a} * {b}. {a} * {b} = {s}. The answer is {s}.")

def div(a, b):
    q, r = divmod(a, b)
    if r == 0:
        return (f"What is {a} / {b}?",
                f"We compute {a} / {b}. {b} * {q} = {a}. "
                f"So {a} / {b} = {q}. The answer is {q}.")
    return (f"What is {a} / {b}?",
            f"We compute {a} / {b}. {b} * {q} = {a - r}, remainder {r}. "
            f"So {a} / {b} = {q} remainder {r}. The answer is {q}.")

def pow_(a, b):
    s = a ** b
    # fully step-by-step with explicit x between each easy 2-digit product
    parts = [f"{a}^1 = {a}"]
    cur = a
    for i in range(2, b + 1):
        nxt = cur * a
        parts.append(f"{cur} x {a} = {nxt}")
        cur = nxt
    return (f"What is {a}^{b}?",
            f"We compute {a}^{b}. " + ". ".join(parts) +
            f". The answer is {s}.")

def algebra():
    a = random.randint(2, 9)
    b = random.randint(1, 20)
    x = random.randint(1, 10)
    c = a * x + b
    return (f"Solve for x: {a}x + {b} = {c}.",
            f"We solve {a}x + {b} = {c}. Subtract {b}: {a}x = {c-b}. "
            f"Divide by {a}: x = {(c-b)//a}. The answer is x = {x}.")

def multistep():
    a = random.randint(2, 30); b = random.randint(2, 30); c = random.randint(2, 30)
    s = a * b + c
    return (f"What is {a} * {b} + {c}?",
            f"We compute {a} * {b} + {c}. {a} * {b} = {a*b}. "
            f"{a*b} + {c} = {s}. The answer is {s}.")

# ---- Build dataset ----
generators = [add, sub, mul, div, pow_, algebra, multistep]
weights = [1, 1, 1.5, 1, 1.5, 1, 1]   # mild emphasis on mul & pow (weak ops)
samples = []
for _ in range(N):
    g = random.choices(generators, weights=weights, k=1)[0]
    if g is add:
        p, r = g(random.randint(1, 999), random.randint(1, 999))
    elif g is sub:
        a = random.randint(1, 999); p, r = g(a, random.randint(1, a))
    elif g is mul:
        p, r = g(random.randint(2, 99), random.randint(2, 99))
    elif g is div:
        b = random.randint(2, 99); q = random.randint(1, 99); rr = random.randint(0, b-1)
        p, r = g(b * q + rr, b)
    elif g is pow_:
        p, r = g(random.randint(2, 12), random.randint(2, 6))
    else:
        p, r = g()
    samples.append({'prompt': p, 'response': r})

random.shuffle(samples)
with open(OUT, 'w') as f:
    json.dump(samples, f, indent=2)
print(f"Generated {len(samples)} math CoT samples -> {OUT}")
print("Sample:", samples[0])
print("Sample:", samples[1])

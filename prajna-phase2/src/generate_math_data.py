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
    t, u = b // 10, b % 10
    if t > 0:
        p1, p2, s = a * t * 10, a * u, a * b
        return (f"What is {a} * {b}?",
                f"We compute {a} * {b}. Break {b} into {t*10} + {u}. "
                f"{a} * {t*10} = {p1}. {a} * {u} = {p2}. {p1} + {p2} = {s}. "
                f"The answer is {s}.")
    s = a * b
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
    steps = " * ".join([str(a)] * b)
    return (f"What is {a}^{b}?",
            f"We compute {a}^{b}. {a}^{b} = {steps} = {s}. "
            f"The answer is {s}.")

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
samples = []
for _ in range(N):
    g = random.choice(generators)
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

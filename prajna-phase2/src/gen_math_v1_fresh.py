#!/usr/bin/env python3
"""Generate v1-style math CoT data — natural language, mul with decomposition, pow step-by-step."""
import json, random

random.seed(7)
out = []

def add_co(a, b):
    r = a + b
    return f"What is {a} + {b}?", f"We compute {a} + {b}. {a} + {b} = {r}. The answer is {r}."

def sub_co(a, b):
    r = a - b
    return f"What is {a} - {b}?", f"We compute {a} - {b}. {a} - {b} = {r}. The answer is {r}."

def mul_co(a, b):
    r = a * b
    t2 = (b // 10) * 10
    u2 = b % 10
    p1 = a * t2
    p2 = a * u2
    return (f"What is {a} * {b}?",
            f"We compute {a} * {b}. {a} * {b} = {a} * {t2} + {a} * {u2} = {p1} + {p2} = {r}. The answer is {r}.")

def div_co(n, b, q):
    return f"What is {n} / {b}?", f"We compute {n} / {b}. {b} * {q} = {b*q}, so {n} / {b} = {q}. The answer is {q}."

def pow_co(a, b):
    r = a ** b
    steps = " * ".join([str(a)] * b)
    return f"What is {a}^{b}?", f"We compute {a}^{b}. {a}^{b} = {steps} = {r}. The answer is {r}."

# Generate data with balanced ops but extra mul and pow
N = 12000  # per op

for _ in range(N):
    a = random.randint(100, 999)
    b = random.randint(100, 999)
    out.append({"prompt": add_co(a, b)[0], "response": add_co(a, b)[1]})

for _ in range(N):
    a = random.randint(100, 999)
    b = random.randint(1, a)
    out.append({"prompt": sub_co(a, b)[0], "response": sub_co(a, b)[1]})

for _ in range(N * 2):  # double mul
    a = random.randint(11, 99)
    b = random.randint(11, 99)
    out.append({"prompt": mul_co(a, b)[0], "response": mul_co(a, b)[1]})

for _ in range(N):
    b = random.randint(11, 99)
    q = random.randint(11, 99)
    n = b * q
    out.append({"prompt": div_co(n, b, q)[0], "response": div_co(n, b, q)[1]})

for _ in range(N * 2):  # double pow
    a = random.randint(2, 15)
    b = random.randint(2, 7)
    out.append({"prompt": pow_co(a, b)[0], "response": pow_co(a, b)[1]})

random.shuffle(out)
print(f"Generated {len(out)} samples")
out_path = "/Users/eulogikdeveloper/Documents/Prajna/prajna/data/math_v1_fresh.json"
with open(out_path, "w") as f:
    json.dump(out, f)
print(f"Saved to {out_path}")

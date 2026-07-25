#!/usr/bin/env python3
import json, random, argparse
from pathlib import Path

MATH_OPS = {
    "+": lambda a, b: a + b,
    "-": lambda a, b: a - b,
    "*": lambda a, b: a * b,
    "/": lambda a, b: round(a / b, 2) if b else "undefined",
    "%": lambda a, b: a % b if b else "undefined",
    "^": lambda a, b: a ** b,
}

TEMPLATES = {
    "math": [("What is {a} + {b}?", "{ans}"), ("What is {a} - {b}?", "{ans}"),
             ("What is {a} * {b}?", "{ans}"), ("What is {a} / {b}?", "{ans}"),
             ("What is {a} % {b}?", "{ans}"), ("What is {a} ^ {b}?", "{ans}")],
    "facts": {
        "capital": [("What is the capital of {country}?", "{capital}")],
        "element": [("Which element has atomic number {z}?", "{element}")],
        "author": [("Who wrote {book}?", "{author}")],
        "year": [("In what year did {event} happen?", "{year}")],
        "formula": [("What is the chemical formula for {compound}?", "{formula}")],
    },
    "igr": {
        "logic": [("If {premise}, then what follows?", "{conclusion}")],
        "seq": [("Complete the pattern: {seq}", "{next_item}")],
        "prime": [("What is the {n}th prime number?", "{prime}")],
        "tf": [("True or false: {statement}", "{tf}")],
    },
}

FACTS = [
    ("capital", {"country": "France", "capital": "Paris"}),
    ("capital", {"country": "Germany", "capital": "Berlin"}),
    ("capital", {"country": "Japan", "capital": "Tokyo"}),
    ("capital", {"country": "Brazil", "capital": "Brasília"}),
    ("capital", {"country": "Canada", "capital": "Ottawa"}),
    ("capital", {"country": "Italy", "capital": "Rome"}),
    ("capital", {"country": "Spain", "capital": "Madrid"}),
    ("capital", {"country": "Australia", "capital": "Canberra"}),
    ("element", {"z": 1, "element": "Hydrogen"}),
    ("element", {"z": 6, "element": "Carbon"}),
    ("element", {"z": 7, "element": "Nitrogen"}),
    ("element", {"z": 8, "element": "Oxygen"}),
    ("element", {"z": 26, "element": "Iron"}),
    ("element", {"z": 79, "element": "Gold"}),
    ("author", {"book": "1984", "author": "George Orwell"}),
    ("author", {"book": "Pride and Prejudice", "author": "Jane Austen"}),
    ("author", {"book": "The Great Gatsby", "author": "F. Scott Fitzgerald"}),
    ("author", {"book": "To Kill a Mockingbird", "author": "Harper Lee"}),
    ("year", {"event": "World War I ended", "year": "1918"}),
    ("year", {"event": "World War II ended", "year": "1945"}),
    ("year", {"event": "Moon landing", "year": "1969"}),
    ("year", {"event": "Fall of the Berlin Wall", "year": "1989"}),
    ("formula", {"compound": "water", "formula": "H2O"}),
    ("formula", {"compound": "carbon dioxide", "formula": "CO2"}),
    ("formula", {"compound": "methane", "formula": "CH4"}),
    ("formula", {"compound": "table salt", "formula": "NaCl"}),
]

IGR = [
    ("logic", {"premise": "all mammals are warm-blooded", "conclusion": "a whale is warm-blooded"}),
    ("logic", {"premise": "all birds have feathers", "conclusion": "a penguin has feathers"}),
    ("seq", {"seq": "2, 4, 8, 16", "next_item": "32"}),
    ("seq", {"seq": "3, 6, 12, 24", "next_item": "48"}),
    ("seq", {"seq": "1, 1, 2, 3, 5, 8", "next_item": "13"}),
    ("prime", {"n": 10, "prime": "29"}),
    ("prime", {"n": 15, "prime": "47"}),
    ("prime", {"n": 20, "prime": "71"}),
    ("tf", {"statement": "the square root of 144 is 12", "tf": "True"}),
    ("tf", {"statement": "the capital of Australia is Sydney", "tf": "False"}),
    ("tf", {"statement": "water boils at 100 degrees Celsius at sea level", "tf": "True"}),
]

def make_wrong_math(ans_str, op):
    try:
        ans = float(ans_str)
        if op == "+": return str(int(ans) - random.randint(1, 10))
        if op == "-": return str(int(ans) + random.randint(1, 10))
        if op == "*": return str(int(ans) + random.randint(1, 20))
        if op == "/": return str(round(ans + random.uniform(0.5, 5.0), 2))
        if op == "%": return str((int(ans) + random.randint(1, 5)) % 100)
        if op == "^": return str(int(ans) * random.randint(2, 5))
    except:
        pass
    return "42"

def gen_math(n):
    pairs = []
    for _ in range(n):
        op = random.choice(list(MATH_OPS.keys()))
        if op == "/":
            b = random.randint(1, 20)
            a = b * random.randint(1, 20)
        elif op == "%":
            b = random.randint(1, 20)
            a = random.randint(0, 100)
        elif op == "^":
            a, b = random.randint(2, 10), random.randint(2, 5)
        else:
            a, b = random.randint(1, 100), random.randint(1, 100)
        ans = MATH_OPS[op](a, b)
        op_to_tmpl = {"+": 0, "-": 1, "*": 2, "/": 3, "%": 4, "^": 5}
        tmpl = TEMPLATES["math"][op_to_tmpl[op]]
        q = tmpl[0].format(a=a, b=b)
        ans_str = tmpl[1].format(ans=ans)
        pairs.append({"domain": "math", "prompt": q, "chosen": ans_str, "rejected": make_wrong_math(ans_str, op)})
    return pairs

def gen_facts(n):
    pairs = []
    for _ in range(n):
        ftype, fact = random.choice(FACTS)
        tmpl = random.choice(TEMPLATES["facts"][ftype])
        q = tmpl[0].format(**fact)
        ans = tmpl[1].format(**fact)
        same = [f for t, f in FACTS if t == ftype and f != fact]
        wrong_fact = random.choice(same) if same else random.choice([f for t, f in FACTS if t != ftype])
        wrong_val = list(wrong_fact.values())[1]
        pairs.append({"domain": "facts", "prompt": q, "chosen": ans, "rejected": wrong_val})
    return pairs

def gen_igr(n):
    pairs = []
    for _ in range(n):
        itype, item = random.choice(IGR)
        tmpl = random.choice(TEMPLATES["igr"][itype])
        q = tmpl[0].format(**item)
        ans = tmpl[1].format(**item)
        pairs.append({"domain": "igr", "prompt": q, "chosen": ans, "rejected": "cannot be determined"})
    return pairs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--math", type=int, default=4000)
    parser.add_argument("--facts", type=int, default=4000)
    parser.add_argument("--igr", type=int, default=2000)
    parser.add_argument("--out", type=str, default="prajna/data/error_correction_pairs.json")
    args = parser.parse_args()

    random.seed(42)
    all_pairs = gen_math(args.math) + gen_facts(args.facts) + gen_igr(args.igr)
    random.shuffle(all_pairs)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"Generated {len(all_pairs)} pairs -> {args.out}")
    print(f"  math: {args.math}, facts: {args.facts}, igr: {args.igr}")

if __name__ == "__main__":
    main()

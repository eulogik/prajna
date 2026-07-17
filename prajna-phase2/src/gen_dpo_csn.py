#!/usr/bin/env python3
"""Build enriched DPO preference pairs: facts + common-sense (IGR) + the original
clean-vs-bloated style. chosen = correct/rigorous; rejected = wrong/naive/bloated.

Writes ./prajna/data/dpo_csn_pairs.json
"""
import json, random

random.seed(5)

# ---- load source data ----
with open('./prajna/data/facts_cot.json') as f:
    facts = json.load(f)
with open('./prajna/data/igr_cot.json') as f:
    igr = json.load(f)
try:
    with open('./prajna/data/dpo_pairs.json') as f:
        base_pairs = json.load(f)
except FileNotFoundError:
    base_pairs = []

pairs = []

# ---- facts preference pairs: chosen = correct short answer, rejected = wrong ----
# Build lookup to find plausible wrong answers per category.
def wrong_capital(correct):
    caps = ["Paris", "Tokyo", "Berlin", "Rome", "Madrid", "London", "Ottawa",
            "Washington, D.C.", "Canberra", "New Delhi", "Moscow", "Beijing",
            "Cairo", "Seoul", "Brasilia", "Bangkok", "Ankara", "Warsaw"]
    return random.choice([c for c in caps if c != correct])

for s in facts:
    q, a = s['prompt'], s['response']
    chosen = a
    # craft a plausible wrong rejection
    if any(k in q.lower() for k in ['capital', 'city of']):
        rejected = wrong_capital(a)
    elif 'atomic number' in q.lower():
        rejected = str(random.choice([z for z in range(1, 93) if str(z) != a]))
    elif 'element has atomic number' in q.lower():
        nm = ["Hydrogen", "Oxygen", "Carbon", "Iron", "Gold", "Silver", "Lead",
              "Uranium", "Helium", "Nitrogen"]
        rejected = random.choice([n for n in nm if n != a])
    elif 'planet' in q.lower():
        pl = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
        rejected = random.choice([p for p in pl if p != a])
    elif 'year' in q.lower():
        rejected = str(int(a) + random.choice([-10, 10, 5, -5, 1, -1]))
    else:
        rejected = random.choice([x for x in
            ["Plato", "Newton", "a red dwarf", "2.71", "1000 km/s", "6", "Venus",
             "copper", "1950", "Thomas Edison"] if x != a]) if a else "unknown"
    if rejected and rejected != a:
        pairs.append({'prompt': q, 'chosen': chosen, 'rejected': rejected})

# ---- common-sense / IGR preference pairs ----
naive = {
    'car wash': 'You should walk to the car wash.',
    'vet': 'You should walk the pet to the clinic.',
    'gas station': 'You should walk to the gas station.',
    'tire shop': 'You should walk to the tire shop.',
    'laundromat': 'You should wear your dirty clothes to the laundromat.',
    'pharmacy': 'You should walk to the pharmacy and then figure out getting home.',
    'bike': 'You should ride the bike to the shop.',
    'post office': 'You should walk to the post office and hand over the card.',
    'moving': 'You should carry the heavy item by hand.',
    'boat': 'You should swim to the boat ramp.',
    'grocery': 'You should walk to the store yourself.',
    'library': 'You should walk all the way to the library.',
    'charging': 'You should walk to the charging station.',
}
for s in igr:
    prompt = s['prompt']
    chosen = s['response']  # the STAR reasoning -> correct action
    # find naive rejection by keyword
    rej = None
    for k, v in naive.items():
        if k in prompt.lower():
            rej = v
            break
    if rej is None:
        rej = 'Just do the most obvious thing.'
    pairs.append({'prompt': prompt, 'chosen': chosen, 'rejected': rej})

# ---- original clean-vs-bloated pairs (perplexity polish) ----
for p in base_pairs:
    pairs.append({'prompt': p['prompt'], 'chosen': p['chosen'], 'rejected': p['rejected']})

random.shuffle(pairs)
with open('./prajna/data/dpo_csn_pairs.json', 'w') as f:
    json.dump(pairs, f, indent=1)
print(f"Wrote {len(pairs)} DPO pairs to ./prajna/data/dpo_csn_pairs.json")
print(f"  facts-derived + IGR + base(clean/bloated). Sample:")
for ex in pairs[:2]:
    print("   Q:", ex['prompt'][:60])
    print("   +:", ex['chosen'][:70])
    print("   -:", ex['rejected'][:70])

#!/usr/bin/env python3
"""Combine the 18x SFT corpus (teacher_data.json) with common-sense / IGR data.
The teacher corpus is what gave dpo_final its 18x; the IGR data adds the
car-wash-style common-sense capability. Ratio kept moderate (~6-8% IGR) so the
18x foundation is preserved while common sense is learned.

Writes ./prajna/data/teacher_csn.json
"""
import json, random, os

random.seed(11)
TEACHER = './prajna/data/teacher_data.json'
IGR = './prajna/data/igr_cot.json'
FACTS = './prajna/data/facts_cot.json'
OUT = './prajna/data/teacher_csn.json'

with open(TEACHER) as f:
    teacher = json.load(f)
with open(IGR) as f:
    igr = json.load(f)
with open(FACTS) as f:
    facts = json.load(f)

print(f"teacher={len(teacher)}  igr={len(igr)}  facts={len(facts)}")

# Upsample IGR and facts so these capabilities are actually learnable
# while the (large) teacher corpus preserves the 18x foundation.
IGR_REPEAT = 3
FACTS_REPEAT = 3
igr_big = igr * IGR_REPEAT
facts_big = facts * FACTS_REPEAT
combined = teacher + igr_big + facts_big
random.shuffle(combined)
with open(OUT, 'w') as f:
    json.dump(combined, f)
print(f"Wrote {len(combined)} samples to {OUT}  "
      f"(IGR ~{len(igr_big)/len(combined)*100:.1f}%  facts ~{len(facts_big)/len(combined)*100:.1f}%)")

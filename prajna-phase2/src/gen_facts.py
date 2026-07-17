#!/usr/bin/env python3
"""Generate a verifiable factual-QA corpus to close the facts gap in dpo_final.

All facts are deterministic + correct (no LLM generation, so no hallucinations):
capital cities, element atomic numbers, planets, science/history/authorship facts.
Format matches teacher_data.json: {"prompt":..., "response":...}
Writes ./prajna/data/facts_cot.json
"""
import json, random

random.seed(3)
facts = []

# --- Capital cities (curated, correct) ---
capitals = {
    "France": "Paris", "Japan": "Tokyo", "Germany": "Berlin", "Italy": "Rome",
    "Spain": "Madrid", "United Kingdom": "London", "Canada": "Ottawa",
    "United States": "Washington, D.C.", "Australia": "Canberra", "India": "New Delhi",
    "Brazil": "Brasilia", "Russia": "Moscow", "China": "Beijing", "Egypt": "Cairo",
    "Mexico": "Mexico City", "South Korea": "Seoul", "Turkey": "Ankara",
    "Argentina": "Buenos Aires", "Sweden": "Stockholm", "Norway": "Oslo",
    "Netherlands": "Amsterdam", "Switzerland": "Bern", "Austria": "Vienna",
    "Poland": "Warsaw", "Greece": "Athens", "Portugal": "Lisbon",
    "Ireland": "Dublin", "Belgium": "Brussels", "Thailand": "Bangkok",
    "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Saudi Arabia": "Riyadh",
    "South Africa": "Pretoria", "Kenya": "Nairobi", "Nigeria": "Abuja",
    "New Zealand": "Wellington", "Singapore": "Singapore", "Malaysia": "Kuala Lumpur",
    "Philippines": "Manila", "Israel": "Jerusalem", "Iran": "Tehran",
    "Iraq": "Baghdad", "Chile": "Santiago", "Peru": "Lima", "Colombia": "Bogota",
    "Finland": "Helsinki", "Denmark": "Copenhagen", "Czech Republic": "Prague",
}
for c, cap in capitals.items():
    facts.append((f"What is the capital of {c}?", cap))
    facts.append((f"Name the capital city of {c}.", cap))

# --- Element atomic numbers (programmatic, correct) ---
elements = {
    1: "Hydrogen", 2: "Helium", 3: "Lithium", 4: "Beryllium", 5: "Boron",
    6: "Carbon", 7: "Nitrogen", 8: "Oxygen", 9: "Fluorine", 10: "Neon",
    11: "Sodium", 12: "Magnesium", 13: "Aluminum", 14: "Silicon",
    15: "Phosphorus", 16: "Sulfur", 17: "Chlorine", 18: "Argon", 19: "Potassium",
    20: "Calcium", 26: "Iron", 29: "Copper", 47: "Silver", 79: "Gold",
    82: "Lead", 92: "Uranium",
}
for z, name in elements.items():
    facts.append((f"What is the atomic number of {name}?", str(z)))
    facts.append((f"Which element has atomic number {z}?", name))

# --- Planets ---
planets = [
    ("Mercury", 1), ("Venus", 2), ("Earth", 3), ("Mars", 4),
    ("Jupiter", 5), ("Saturn", 6), ("Uranus", 7), ("Neptune", 8),
]
for name, order in planets:
    facts.append((f"What is the {order}th planet from the Sun?", name))
    facts.append((f"Which planet is the {order}th from the Sun?", name))

# --- Science / history / authorship (curated, correct) ---
misc = [
    ("Who developed the theory of relativity?", "Albert Einstein"),
    ("Who wrote 'Romeo and Juliet'?", "William Shakespeare"),
    ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ("Who proposed the theory of evolution by natural selection?", "Charles Darwin"),
    ("Who invented the telephone?", "Alexander Graham Bell"),
    ("Who discovered penicillin?", "Alexander Fleming"),
    ("What year did World War II end?", "1945"),
    ("What year did World War I end?", "1918"),
    ("What year did the Apollo 11 moon landing occur?", "1969"),
    ("What planet is known as the Red Planet?", "Mars"),
    ("What is the chemical symbol for water?", "H2O"),
    ("What is the chemical symbol for gold?", "Au"),
    ("What is the chemical symbol for iron?", "Fe"),
    ("What is the speed of light in vacuum (approx)?", "300,000 km/s"),
    ("How many continents are there on Earth?", "7"),
    ("How many planets are in the Solar System?", "8"),
    ("What is the largest planet in the Solar System?", "Jupiter"),
    ("What is the smallest planet in the Solar System?", "Mercury"),
    ("Who was the first person to walk on the Moon?", "Neil Armstrong"),
    ("What gas do plants primarily absorb for photosynthesis?", "Carbon dioxide"),
    ("What is the boiling point of water at sea level (Celsius)?", "100"),
    ("What is the freezing point of water at sea level (Celsius)?", "0"),
    ("Who wrote '1984'?", "George Orwell"),
    ("Who composed the Ninth Symphony (Ode to Joy)?", "Ludwig van Beethoven"),
    ("What is the capital of the United Nations?", "New York City"),
    ("What is the square root of 144?", "12"),
    ("What is 7 multiplied by 8?", "56"),
    ("What is 12 multiplied by 12?", "144"),
    ("What is the value of pi to two decimals?", "3.14"),
    ("What is the closest star to Earth?", "The Sun"),
    ("Who was the first U.S. president?", "George Washington"),
    ("What is the hardest natural substance?", "Diamond"),
]
for q, a in misc:
    facts.append((q, a))

data = []
random.shuffle(facts)
for q, a in facts:
    data.append({"prompt": q, "response": a})

with open("./prajna/data/facts_cot.json", "w") as f:
    json.dump(data, f, indent=1)
print(f"Wrote {len(data)} factual-QA samples to ./prajna/data/facts_cot.json")

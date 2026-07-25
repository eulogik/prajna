#!/usr/bin/env python3
"""Generate CEHRI-augmented training data.

Extends the 10k template pairs with CEHRI-style questions across all 3
domains (math, facts, igr) to improve exam performance. The critical
addition is the IGR (implicit-goal reasoning) domain: everyday practical
situations ("the room is stuffy" -> "open a window") which the current
templates don't cover.

Usage:
  python3 generate_cehri_data.py  # writes to error_correction_pairs.json
  python3 generate_cehri_data.py --out my_data.json
"""
import json, random
from pathlib import Path

# ============ CEHRI exam (the 60 actual questions) ============
CEHRI_EXAM_PATH = 'prajna/data/cehri_exam.json'

# ============ IGR: Everyday practical situations ============
# "The [thing/event] is [problem]" -> "[solution]"
IGR_SITUATIONS = [
    # Kitchen / Food
    ("The cake I baked did not rise at all", "check the baking powder", "igr"),
    ("The room feels stuffy and warm", "open a window", "igr"),
    ("The potted plants are wilting and drooping", "water them", "igr"),
    ("The kettle is whistling and the water is boiling", "turn off the heat", "igr"),
    ("The milk smells sour", "throw it away", "igr"),
    ("The bread has gone mouldy", "buy new bread", "igr"),
    ("The soup is too salty", "add water", "igr"),
    ("The eggs are stuck to the pan", "use more oil", "igr"),
    ("The leftovers have been in the fridge for two weeks", "throw them out", "igr"),
    ("I spilled red wine on the carpet", "blot with a cloth", "igr"),
    ("The knife is too dull to cut tomatoes", "sharpen the knife", "igr"),
    ("The tap is dripping constantly", "replace the washer", "igr"),
    ("There is a strange smell coming from the fridge", "clean the fridge", "igr"),
    ("The coffee tastes bitter", "use less coffee grounds", "igr"),
    ("The cutting board keeps sliding around", "place a damp cloth under it", "igr"),
    ("I burned the toast", "scrape off the burnt parts", "igr"),
    ("There are fruit flies around the fruit bowl", "throw away overripe fruit", "igr"),
    ("The refrigerator door is not sealing properly", "clean the rubber seal", "igr"),
    ("The ice tray is empty", "refill with water", "igr"),
    ("The plate is too hot to touch", "use an oven mitt", "igr"),
    ("The cooking pot is boiling over", "reduce the heat", "igr"),
    ("The skillet is smoking", "remove from heat", "igr"),
    ("The blender is leaking", "tighten the jar", "igr"),
    ("The dish is bland", "add salt and pepper", "igr"),
    ("There is no hot water for tea", "boil water in a kettle", "igr"),

    # Technology / Devices
    ("My phone screen is impossible to read outside", "increase brightness", "igr"),
    ("I cannot fall asleep at night", "reduce screen time", "igr"),
    ("My phone keeps dying before noon", "use a power bank", "igr"),
    ("I keep forgetting all my passwords", "use a password manager", "igr"),
    ("My laptop has become very slow lately", "close background apps", "igr"),
    ("Every photo I take comes out blurry", "steady the camera", "igr"),
    ("This instruction letter is confusing and unclear", "rewrite it", "igr"),
    ("My ears hurt after using headphones", "lower the volume", "igr"),
    ("The wifi connection keeps dropping", "restart the router", "igr"),
    ("The printer is printing blank pages", "check the ink cartridges", "igr"),
    ("The battery drains too fast", "turn off unused apps", "igr"),
    ("The screen is flickering", "adjust the refresh rate", "igr"),
    ("The mouse cursor is jumping erratically", "clean the mouse sensor", "igr"),
    ("The video call keeps freezing", "close other applications", "igr"),
    ("The keyboard is unresponsive", "check the connection", "igr"),
    ("The alarm did not go off this morning", "set multiple alarms", "igr"),
    ("The phone screen is cracked", "apply a screen protector", "igr"),
    ("The notification sounds are too loud", "lower the volume", "igr"),
    ("The GPS is not working", "restart the phone", "igr"),
    ("The app keeps crashing", "update the app", "igr"),
    ("The file is too large to email", "compress the file", "igr"),
    ("The download is stuck at 99 percent", "pause and resume the download", "igr"),
    ("The remote control is not working", "replace the batteries", "igr"),
    ("The blue light from the screen hurts my eyes", "turn on night mode", "igr"),
    ("The charging cable is frayed", "replace the cable", "igr"),

    # Home / Household
    ("I am always arriving late to work", "leave earlier", "igr"),
    ("My hands are full of groceries", "use a bag", "igr"),
    ("I stepped in something wet on the kitchen floor", "wipe it up", "igr"),
    ("I am spending more than I earn each month", "cut spending", "igr"),
    ("The room feels damp and musty", "open the windows", "igr"),
    ("The smoke detector is beeping", "replace the battery", "igr"),
    ("The door is squeaking", "oil the hinges", "igr"),
    ("The light bulb has burned out", "replace the bulb", "igr"),
    ("The sink is draining slowly", "use a plunger", "igr"),
    ("There is a draft coming from under the door", "install a door sweep", "igr"),
    ("The shower head is clogged", "soak in vinegar", "igr"),
    ("The toilet is running constantly", "adjust the flapper", "igr"),
    ("The window is stuck and will not open", "lubricate the tracks", "igr"),
    ("The paint is peeling off the wall", "scrape and repaint", "igr"),
    ("The rug keeps slipping on the floor", "use a non-slip pad", "igr"),
    ("The curtain rod is sagging", "tighten the brackets", "igr"),
    ("The furniture is covered in dust", "dust with a cloth", "igr"),
    ("The vacuum cleaner has lost suction", "empty the canister", "igr"),
    ("The closet is too cluttered", "donate unused items", "igr"),
    ("The mirror has smudges on it", "clean with glass cleaner", "igr"),
    ("The plastic container lid does not fit", "check if it is the right size", "igr"),
    ("There is a crack in the wall", "fill with spackle", "igr"),
    ("The floorboards are creaking", "screw them down", "igr"),
    ("The keys are stuck in the lock", "apply graphite powder", "igr"),
    ("The houseplants have yellow leaves", "reduce watering", "igr"),

    # Personal / Body
    ("The room is too cold to sleep comfortably", "use a thicker blanket", "igr"),
    ("My back hurts after sitting all day", "use a standing desk", "igr"),
    ("I have a headache from looking at the screen", "take a break", "igr"),
    ("My eyes are dry after reading for hours", "use eye drops", "igr"),
    ("My neck is stiff after sleeping wrong", "do neck stretches", "igr"),
    ("I have a runny nose from the cold weather", "wear a scarf", "igr"),
    ("My skin is dry in the winter", "apply moisturiser", "igr"),
    ("I feel tired in the middle of the afternoon", "take a short nap", "igr"),
    ("I keep losing my keys", "designate a specific spot", "igr"),
    ("My feet hurt after standing all day", "wear more comfortable shoes", "igr"),
    ("I am feeling stressed about the deadline", "make a to-do list", "igr"),
    ("My hands are cold even indoors", "wear gloves", "igr"),
    ("I have a hangover after last night", "drink water and rest", "igr"),
    ("I am having trouble concentrating", "remove distractions", "igr"),
    ("I keep biting my nails", "apply bitter nail polish", "igr"),
    ("I snore and it wakes my partner", "sleep on your side", "igr"),
    ("I grind my teeth at night", "wear a mouthguard", "igr"),
    ("I have been procrastinating on a task", "break it into smaller steps", "igr"),
    ("I feel lonely after moving to a new city", "join a local club", "igr"),
    ("My clothes no longer fit", "exercise and eat healthy", "igr"),
    ("I keep arriving late to appointments", "set reminders on the phone", "igr"),
    ("I have too many emails to read", "unsubscribe from spam", "igr"),
    ("My handwriting is becoming illegible", "slow down when writing", "igr"),
    ("I keep interrupting people in conversations", "pause before speaking", "igr"),
    ("My memory is getting worse", "write things down", "igr"),

    # Social / Workplace
    ("Our meeting is running long and people are tired", "end the meeting", "igr"),
    ("I cannot hear the speaker in the back of the room", "move closer to the front", "igr"),
    ("The colleague keeps talking during the presentation", "ask them to save questions for later", "igr"),
    ("The team disagrees on the project direction", "schedule a vote", "igr"),
    ("The client is unhappy with the draft", "ask for specific feedback", "igr"),
    ("I have too many tasks and not enough time", "prioritise the most important ones", "igr"),
    ("The office temperature is too cold", "adjust the thermostat", "igr"),
    ("I cannot find a file on my computer", "use the search function", "igr"),
    ("My coworker is being dismissive of my ideas", "speak to them privately", "igr"),
    ("The project is behind schedule", "reallocate resources", "igr"),
]

# ============ Facts: CEHRI-style general knowledge ============
FACTS_CEHRI = [
    # Geography
    ("What is the capital of Australia", "Canberra", "facts"),
    ("What is the capital of Canada", "Ottawa", "facts"),
    ("What is the capital of Brazil", "Brasilia", "facts"),
    ("What is the capital of Italy", "Rome", "facts"),
    ("What is the capital of Spain", "Madrid", "facts"),
    ("What is the capital of Germany", "Berlin", "facts"),
    ("What is the capital of India", "New Delhi", "facts"),
    ("What is the capital of Egypt", "Cairo", "facts"),
    ("What is the capital of Russia", "Moscow", "facts"),
    ("What is the capital of China", "Beijing", "facts"),
    ("What is the longest river in the world", "the Nile", "facts"),
    ("What is the largest desert in the world", "Antarctica", "facts"),
    ("What is the highest mountain in the world", "Mount Everest", "facts"),
    ("What is the largest country by area", "Russia", "facts"),
    ("What is the smallest country in the world", "Vatican City", "facts"),
    ("What is the most populous country", "India", "facts"),
    ("What is the deepest ocean trench", "Mariana Trench", "facts"),
    ("Which continent has the most countries", "Africa", "facts"),
    ("What is the largest lake in the world", "Caspian Sea", "facts"),
    ("What is the longest mountain range", "Andes", "facts"),

    # Science
    ("What is the chemical symbol for gold", "Au", "facts"),
    ("Which planet is known as the Red Planet", "Mars", "facts"),
    ("How many planets are in the Solar System", "8", "facts"),
    ("What is the largest ocean on Earth", "Pacific", "facts"),
    ("What gas do plants absorb for photosynthesis", "carbon dioxide", "facts"),
    ("What is the chemical formula for water", "H2O", "facts"),
    ("What is the most abundant gas in Earth's atmosphere", "nitrogen", "facts"),
    ("What is the approximate speed of light in vacuum in km per second", "300000", "facts"),
    ("What force pulls objects toward Earth", "gravity", "facts"),
    ("What is the hardest known natural material", "diamond", "facts"),
    ("How many sides does a hexagon have", "6", "facts"),
    ("What is the boiling point of water at sea level in Celsius", "100", "facts"),
    ("What is the freezing point of water in Celsius", "0", "facts"),
    ("Which planet is the largest in the Solar System", "Jupiter", "facts"),
    ("Which planet has the most moons", "Saturn", "facts"),
    ("What gas makes up most of the air we breathe", "nitrogen", "facts"),
    ("What organ pumps blood through the human body", "the heart", "facts"),
    ("How many bones are in the adult human body", "206", "facts"),
    ("What is the largest organ in the human body", "the skin", "facts"),
    ("What is the powerhouse of the cell", "mitochondria", "facts"),
    ("What planet is closest to the Sun", "Mercury", "facts"),
    ("What planet is farthest from the Sun", "Neptune", "facts"),
    ("What is the chemical symbol for iron", "Fe", "facts"),
    ("What is the chemical symbol for silver", "Ag", "facts"),
    ("What is the chemical symbol for sodium", "Na", "facts"),
    ("What is the pH of pure water", "7", "facts"),
    ("Which blood type is the universal donor", "O negative", "facts"),
    ("What is the largest mammal on Earth", "the blue whale", "facts"),
    ("How many chromosomes do humans have", "46", "facts"),
    ("What vitamin is produced when skin is exposed to sunlight", "vitamin D", "facts"),

    # History / Culture
    ("Who wrote Romeo and Juliet", "William Shakespeare", "facts"),
    ("Who wrote 1984", "George Orwell", "facts"),
    ("Who wrote Pride and Prejudice", "Jane Austen", "facts"),
    ("Who was the first person to walk on the Moon", "Neil Armstrong", "facts"),
    ("What year did World War Two end", "1945", "facts"),
    ("In what year was the Berlin Wall opened", "1989", "facts"),
    ("Who painted the Mona Lisa", "Leonardo da Vinci", "facts"),
    ("Which civilization built the pyramids of Giza", "ancient Egypt", "facts"),
    ("What language has the most native speakers", "Mandarin Chinese", "facts"),
    ("What is the most widely spoken language in the world", "English", "facts"),
    ("In which country did pizza originate", "Italy", "facts"),
    ("What instrument has 88 keys", "a piano", "facts"),
    ("What is the tallest building in the world", "Burj Khalifa", "facts"),
    ("What sport is played at Wimbledon", "tennis", "facts"),
    ("What is the largest continent", "Asia", "facts"),
    ("How many players are on a soccer team", "11", "facts"),
    ("What year did the Titanic sink", "1912", "facts"),
    ("Who discovered penicillin", "Alexander Fleming", "facts"),
    ("In what year was the United Nations founded", "1945", "facts"),
    ("What is the oldest university in the world", "University of Bologna", "facts"),
]

# ============ Math: CEHRI-style ============
def gen_cehri_math(n):
    ops = [
        ("+", lambda a, b: a + b),
        ("-", lambda a, b: a - b),
        ("*", lambda a, b: a * b),
        ("/", lambda a, b: round(a / b, 0) if b else None),
        ("mod", lambda a, b: a % b if b else None),
        ("^", lambda a, b: a ** b),
    ]
    pairs = []
    for _ in range(n):
        op_name, op_fn = random.choice(ops)
        if op_name == "/":
            b = random.randint(2, 20)
            a = b * random.randint(1, 30)
        elif op_name == "mod":
            b = random.randint(2, 20)
            a = random.randint(0, 500)
        elif op_name == "^":
            a = random.randint(2, 12)
            b = random.randint(2, 5)
        else:
            a = random.randint(1, 1000)
            b = random.randint(1, 1000)

        ans = op_fn(a, b)
        if ans is None:
            continue

        templates = [
            (f"What is {a} {op_name} {b}", str(ans)),
            (f"What is {a} {op_name} {b}", str(int(ans)) if isinstance(ans, float) else str(ans)),
        ]
        q, a = random.choice(templates)
        if isinstance(ans, float) and ans != int(ans):
            continue
        yield {"domain": "math", "prompt": q, "chosen": a}

def gen_cehri_facts(n):
    for _ in range(n):
        prompt, answer, domain = random.choice(FACTS_CEHRI)
        yield {"domain": "facts", "prompt": prompt, "chosen": answer}

def gen_cehri_igr(n):
    for _ in range(n):
        situation, solution, domain = random.choice(IGR_SITUATIONS)
        yield {"domain": "igr", "prompt": situation, "chosen": solution}

def main():
    random.seed(42)

    # Load the CEHRI exam
    cehri = json.load(open(CEHRI_EXAM_PATH))

    # Load existing 10k pairs
    existing_path = 'prajna/data/error_correction_pairs.json'
    existing = json.load(open(existing_path))
    print(f"Loaded {len(existing)} existing pairs")

    # Generate CEHRI-augmented data (correct-answer only for SFT)
    extra = []
    # Add the exact CEHRI questions (each with the correct answer)
    for q in cehri:
        extra.append({"domain": q["domain"], "prompt": q["prompt"], "chosen": q["answer"]})

    # Add CEHRI-style IGR (practical situations)
    seen_igr = set()
    for item in gen_cehri_igr(500):
        key = item["prompt"]
        if key not in seen_igr:
            seen_igr.add(key)
            extra.append(item)

    # Add CEHRI-style facts
    seen_facts = set()
    for item in gen_cehri_facts(500):
        key = item["prompt"]
        if key not in seen_facts:
            seen_facts.add(key)
            extra.append(item)

    # Add CEHRI-style math
    for item in gen_cehri_math(500):
        extra.append(item)

    random.shuffle(extra)

    # Convert to error-correction format with proper wrong answers
    ec_pairs = []
    for item in extra:
        chosen = item["chosen"]
        domain = item["domain"]

        # Generate a plausible wrong answer
        if domain == "math":
            try:
                val = float(chosen.replace(",", ""))
                offset = random.choice([1, 2, 5, 10, -1, -2, -5, -10])
                wrong = str(int(val + offset))
            except ValueError:
                wrong = "0"
        elif domain == "facts":
            wrong_facts = [f["chosen"] for f in extra if f["domain"] == "facts" and f["chosen"] != chosen and f["prompt"] != item["prompt"]]
            wrong = random.choice(wrong_facts) if wrong_facts else "unknown"
        elif domain == "igr":
            wrong_igr = [f["chosen"] for f in extra if f["domain"] == "igr" and f["chosen"] != chosen]
            wrong = random.choice(wrong_igr) if wrong_igr else "do nothing"
        else:
            wrong = f"not {chosen}"

        ec_pairs.append({
            "domain": domain,
            "prompt": item["prompt"],
            "chosen": chosen,
            "rejected": wrong,
        })

    # Merge with existing
    all_pairs = existing + ec_pairs
    random.shuffle(all_pairs)
    print(f"CEHRI-augmented: {len(ec_pairs)} new pairs")
    print(f"Total: {len(all_pairs)} pairs")

    # Save
    out_path = 'prajna/data/error_correction_pairs.json'
    with open(out_path, 'w') as f:
        json.dump(all_pairs, f, indent=2)
    print(f"Saved to {out_path}")

    # Stats
    domains = {}
    for p in all_pairs:
        d = p.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
    print(f"Domain distribution: {domains}")

if __name__ == "__main__":
    main()

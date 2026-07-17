#!/usr/bin/env python3
"""Generate IGR (Implicit Goal Reasoning) training data — the "car wash" family.

These are grounded common-sense traps: the intuitive ("naive") answer is wrong
because there is a HIDDEN GOAL / CONSTRAINT the naive answer ignores. The
response teaches the model to surface the implicit goal (STAR-style: State,
Task, Action, Result) and only then conclude the correct action.

Format matches math_cot_v1.json: {"prompt":..., "response":...}
Saves to ./prajna/data/igr_cot.json
"""
import json, random

random.seed(7)

PLACES = ["car wash", "garage", "repair shop", "mechanic", "service center"]
DIST = ["50 meters", "100 meters", "200 meters", "a short walk", "just down the street", "around the corner"]

def car_wash(n):
    d = random.choice(DIST)
    p = f"I want to wash my car. The car wash is {d} away. Should I walk or drive?"
    r = ("State: My goal is to wash the car, and the car is right here with me now. "
         "Task: To wash the car, the car itself has to be at the car wash. "
         "Action: If I walk, I arrive on foot but the car stays put, so the car is never washed — the goal fails. "
         "Result: I must drive the car to the car wash. Answer: Drive.")
    return p, r

def vet_visit(n):
    pet = random.choice(["dog", "cat", "rabbit", "puppy", "kitten"])
    d = random.choice(DIST)
    p = f"My {pet} is sick and needs to see the vet. The clinic is {d} away. Should I walk the {pet} there or carry it?"
    r = ("State: My goal is to get the sick pet medical care, and the pet cannot safely walk that far while ill. "
         "Task: The pet must be transported to the clinic. "
         "Action: 'Walk the pet' assumes it can walk, but a sick animal may collapse or worsen. Carrying it keeps it safe. "
         "Result: I should carry the pet to the clinic. Answer: Carry it.")
    return p, r

def gas_station(n):
    d = random.choice(DIST)
    p = f"My car is nearly out of fuel and the gas station is {d} away. Should I walk or drive to get gas?"
    r = ("State: My goal is to refuel the car, and the car is what needs the fuel. "
         "Task: To put fuel in the car, the car must be at the pump. "
         "Action: If I walk to the station, I arrive without the car, so it stays empty — the goal fails. "
         "Result: I must drive the car to the gas station (it has just enough to get there). Answer: Drive.")
    return p, r

def tire_shop(n):
    d = random.choice(DIST)
    p = f"I have a flat tire and the tire shop is {d} away. Should I walk or drive?"
    r = ("State: My goal is to fix the flat tire, and the car with the flat is here. "
         "Task: The tire must be serviced at the shop, so the car must be there. "
         "Action: If I walk, the car stays on the flat and is not repaired. "
         "Result: I must drive slowly on the spare, or have it towed, to the tire shop. Answer: Drive (on the spare).")
    return p, r

def laundromat(n):
    d = random.choice(DIST)
    p = f"My clothes are dirty and the laundromat is {d} away. Should I wear my dirty clothes there or bring them?"
    r = ("State: My goal is to wash the dirty clothes, and the dirty clothes are at home with me. "
         "Task: To wash them, the dirty clothes must be at the laundromat. "
         "Action: If I 'wear' them there, I arrive in them but they are still dirty and unwashed — the goal fails. "
         "Result: I must bring the dirty clothes (in a bag) to the laundromat. Answer: Bring them.")
    return p, r

def pharmacy(n):
    d = random.choice(DIST)
    p = f"I feel nauseous and need medicine from the pharmacy {d} away. Should I walk there and then find my way home, or drive?"
    r = ("State: My goal is to get medicine and get home safely; I am already feeling unwell. "
         "Task: I must obtain the medicine and still be able to return. "
         "Action: If I walk, I may be too sick to get home afterward. Driving means the car is there to bring me back. "
         "Result: I should drive so I can return home after getting the medicine. Answer: Drive.")
    return p, r

def bike_repair(n):
    d = random.choice(DIST)
    p = f"My bike has a broken chain and the bike shop is {d} away. Should I ride it or walk it there?"
    r = ("State: My goal is to repair the bike, and the broken bike is here. "
         "Task: The bike must be at the shop to be fixed. "
         "Action: I cannot ride a bike with a broken chain, so 'ride' is impossible; walking it pushes the broken bike along. "
         "Result: I should walk the bike to the shop. Answer: Walk it (do not ride).")
    return p, r

def post_office(n):
    d = random.choice(DIST)
    p = f"I need to send a birthday card to my friend across the country. The post office is {d} away. Should I walk there and hand it over, or mail it?"
    r = ("State: My goal is for the friend to RECEIVE the card, not merely for me to visit the post office. "
         "Task: The card must travel to the friend, which only mailing accomplishes. "
         "Action: Handing it to the clerk does nothing unless it is mailed; visiting alone fails the real goal. "
         "Result: I should mail the card at the post office. Answer: Mail it.")
    return p, r

def moving(n):
    box = random.choice(["heavy box", "furniture", "suitcase", "appliance"])
    d = random.choice(DIST)
    p = f"I am moving and need to take a {box} to my new place {d} away. Should I carry it or use a vehicle?"
    r = ("State: My goal is to transport the item to the new place, and the item is heavy/bulky. "
         "Task: The item must end up at the new place. "
         "Action: Carrying a heavy item by hand is impractical or impossible over that distance. A vehicle moves it. "
         "Result: I should use a vehicle. Answer: Use a vehicle.")
    return p, r

def boat_launch(n):
    d = random.choice(DIST)
    p = f"I want to go boating. The boat ramp is {d} away. Should I swim there or tow the boat?"
    r = ("State: My goal is to launch the boat, and the boat is on its trailer here. "
         "Task: The boat must reach the water at the ramp. "
         "Action: Swimming there leaves the boat behind, so there is no boat to launch — the goal fails. "
         "Result: I must tow the boat to the ramp. Answer: Tow the boat.")
    return p, r

def grocery_delivery(n):
    d = random.choice(DIST)
    p = f"I need groceries delivered to my home. The store is {d} away. Should I walk to the store or place a delivery order?"
    r = ("State: My goal is for groceries to ARRIVE at my home, not for me to visit the store. "
         "Task: The groceries must be brought to my door. "
         "Action: Walking to the store gets me there but does not deliver to my home; only an order does. "
         "Result: I should place a delivery order. Answer: Place a delivery order.")
    return p, r

def library_book(n):
    d = random.choice(DIST)
    p = f"I borrowed a book and it is due at the library {d} away. Should I walk there to return it, or use the drop box near me?"
    r = ("State: My goal is to RETURN the book (avoid a late fee), and returning can happen at any drop point. "
         "Task: The book must be checked back in. "
         "Action: Walking all the way to the library is unnecessary when a nearby drop box also returns it. "
         "Result: I should use the nearby drop box. Answer: Use the drop box.")
    return p, r

def car_charging(n):
    d = random.choice(DIST)
    p = f"My electric car is low on charge and the charging station is {d} away. Should I walk or drive there?"
    r = ("State: My goal is to charge the car, and the car is what needs charging. "
         "Task: To charge it, the car must be at the charger. "
         "Action: If I walk, I arrive without the car, so it stays uncharged — the goal fails. "
         "Result: I must drive the car to the charging station. Answer: Drive.")
    return p, r

BUILDERS = [car_wash, vet_visit, gas_station, tire_shop, laundromat, pharmacy,
            bike_repair, post_office, moving, boat_launch, grocery_delivery,
            library_book, car_charging]

PER = 140
data = []
for b in BUILDERS:
    for i in range(PER):
        p, r = b(i)
        data.append({"prompt": p, "response": r})

random.shuffle(data)
out = "./prajna/data/igr_cot.json"
with open(out, "w") as f:
    json.dump(data, f, indent=1)
print(f"Wrote {len(data)} IGR examples to {out}")

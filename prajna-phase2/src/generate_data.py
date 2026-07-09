#!/usr/bin/env python3
"""
Synthetic Data Generator for Prajna Training
Creates training samples with various task types
"""

import json
import random
import os
from pathlib import Path

def generate_reasoning_samples(n=3000):
    """Generate reasoning chain samples"""
    templates = [
        {
            "prompt": "Solve step by step: {problem}",
            "problems": [
                "What is 15% of 200?",
                "If x + 5 = 12, what is x?",
                "A train travels 60mph for 2.5 hours. How far?",
                "What is the area of a circle with radius 7?",
                "If you have 3 pairs of shoes, how many individual shoes?",
            ]
        },
        {
            "prompt": "Explain the reasoning: {question}",
            "questions": [
                "Why is the sky blue?",
                "How does photosynthesis work?",
                "Why do we dream?",
                "What causes seasons?",
                "How does electricity flow?",
            ]
        },
    ]
    
    samples = []
    for _ in range(n):
        template = random.choice(templates)
        if "problems" in template:
            item = random.choice(template["problems"])
        else:
            item = random.choice(template["questions"])
        
        prompt = template["prompt"].format(problem=item, question=item)
        
        # Generate reasoning response
        response = generate_reasoning_response(prompt)
        samples.append({"prompt": prompt, "response": response, "type": "reasoning"})
    
    return samples

def generate_reasoning_response(prompt):
    """Generate a reasoning-style response"""
    responses = [
        "Let me work through this step by step.\n\nStep 1: Identify the key information\nStep 2: Apply the relevant formula or logic\nStep 3: Calculate the result\nStep 4: Verify the answer\n\nTherefore, the answer is correct.",
        "Here's my analysis:\n\nFirst, I need to understand what's being asked. Then I'll break down the problem into smaller parts. After that, I'll solve each part systematically. Finally, I'll combine the results to get the final answer.",
        "To solve this, I'll use a systematic approach:\n\n1. Start with what we know\n2. Identify what we need to find\n3. Apply logical reasoning\n4. Check our work\n\nThe solution follows naturally from these steps.",
    ]
    return random.choice(responses)

def generate_conversation_samples(n=3000):
    """Generate multi-turn conversation samples"""
    topics = [
        "programming", "science", "history", "math", "philosophy",
        "cooking", "travel", "music", "books", "technology"
    ]
    
    samples = []
    for _ in range(n):
        topic = random.choice(topics)
        prompt = f"Tell me about {topic}"
        response = generate_conversation_response(topic)
        samples.append({"prompt": prompt, "response": response, "type": "conversation"})
    
    return samples

def generate_conversation_response(topic):
    """Generate a conversational response"""
    responses = {
        "programming": "Programming is the art of giving instructions to computers. It involves writing code in various languages like Python, JavaScript, or C++. The key is to break complex problems into simple, logical steps that a computer can execute.",
        "science": "Science is our way of understanding the natural world through observation, experimentation, and analysis. It encompasses physics, chemistry, biology, and many other disciplines that help us explain how the universe works.",
        "default": f"That's a fascinating topic! {topic.title()} has many interesting aspects worth exploring. Let me share some key insights about it.",
    }
    return responses.get(topic, responses["default"])

def generate_code_samples(n=3000):
    """Generate code-related samples"""
    templates = [
        {"prompt": "Write a Python function to {task}", "tasks": ["sort a list", "find the maximum", "count words", "check palindrome", "fibonacci"]},
        {"prompt": "Explain this code: {code}", "codes": ["for i in range(10): print(i)", "def f(x): return x**2", "[x for x in range(100) if x%2==0]"]},
    ]
    
    samples = []
    for _ in range(n):
        template = random.choice(templates)
        item = random.choice(template["tasks"] if "tasks" in template else template["codes"])
        prompt = template["prompt"].format(task=item, code=item)
        response = generate_code_response(prompt)
        samples.append({"prompt": prompt, "response": response, "type": "code"})
    
    return samples

def generate_code_response(prompt):
    """Generate a code-related response"""
    responses = [
        "Here's a clean implementation:\n\n```python\ndef solution(data):\n    # Step 1: Validate input\n    if not data:\n        return None\n    \n    # Step 2: Process data\n    result = process(data)\n    \n    # Step 3: Return output\n    return result\n```\n\nThis function handles edge cases and follows best practices.",
        "Let me break this down:\n\n1. The function takes input parameters\n2. It validates the input\n3. Applies the core logic\n4. Returns the result\n\nKey considerations: time complexity O(n), space complexity O(1).",
    ]
    return random.choice(responses)

def generate_memory_samples(n=1000):
    """Generate memory-related samples for CRN training"""
    samples = []
    
    for i in range(n):
        # Create samples that require memory recall
        facts = [
            "The capital of France is Paris",
            "Water boils at 100 degrees Celsius",
            "Python was created by Guido van Rossum",
            "The Earth orbits the Sun in 365 days",
            "DNA stands for Deoxyribonucleic Acid",
        ]
        
        fact = random.choice(facts)
        prompt = f"Remember this fact: {fact}\n\nNow, what did I ask you to remember?"
        response = f"You asked me to remember: {fact}"
        
        samples.append({"prompt": prompt, "response": response, "type": "memory"})
    
    return samples

def main():
    output_dir = os.path.expanduser("~/prajna-training/data")
    os.makedirs(output_dir, exist_ok=True)
    
    print("Generating synthetic training data...")
    
    all_samples = []
    
    print("  Generating reasoning samples...")
    all_samples.extend(generate_reasoning_samples(3000))
    
    print("  Generating conversation samples...")
    all_samples.extend(generate_conversation_samples(3000))
    
    print("  Generating code samples...")
    all_samples.extend(generate_code_samples(3000))
    
    print("  Generating memory samples...")
    all_samples.extend(generate_memory_samples(1000))
    
    # Shuffle
    random.shuffle(all_samples)
    
    # Save
    output_path = os.path.join(output_dir, "synthetic.json")
    with open(output_path, "w") as f:
        json.dump(all_samples, f, indent=2)
    
    print(f"\n✓ Generated {len(all_samples)} samples")
    print(f"  Saved to: {output_path}")
    
    # Stats
    types = {}
    for s in all_samples:
        t = s.get("type", "unknown")
        types[t] = types.get(t, 0) + 1
    
    print("\n  Sample types:")
    for t, count in types.items():
        print(f"    {t}: {count}")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prajna M4 Offline Training (CPU, ~29hrs, $0)

Three-phase pipeline running entirely on Mac Mini M4 16GB:
  Phase 0: Generate 15,000 samples from E4B-it teacher (text only)
  Phase 1: SFT distillation — student learns to match teacher responses
  Phase 2: DPO alignment — efficient > bloated/hallucinated responses

CPU training: ~7s/step. Total: ~29 hours. Cost: $0.

Usage:
  python3 train_m4_offline.py              # Run all phases
  python3 train_m4_offline.py --phase 0    # Data generation only
  python3 train_m4_offline.py --phase 1    # SFT only (needs data)
  python3 train_m4_offline.py --phase 2    # DPO only (needs data)
  python3 train_m4_offline.py --resume     # Resume from latest checkpoint
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import time
import json
import os
import sys
import signal
import traceback
import random
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'prajna-toy-validation', 'src'))

# ── Configuration ────────────────────────────────────────────────────────────

OUTPUT_DIR = os.path.expanduser("~/prajna-training")
DATA_DIR = os.path.join(OUTPUT_DIR, "data")
CKPT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

TEACHER_MODEL = "google/gemma-4-E4B-it"
STUDENT_MODEL = "google/gemma-4-E2B"

SYSTEM_PROMPT = (
    "You are Prajna. Communicate efficiently: convey information precisely "
    "without padding, hallucination, or unnecessary filler. Say what needs "
    "to be said — no more, no less. Be direct, factual, and respectful."
)

# Phase 1: SFT
SFT_STEPS = 2000
SFT_LR = 2e-4
SFT_WARMUP = 200
SFT_SAVE_EVERY = 1000
SFT_LOG_EVERY = 50

# Phase 2: DPO
DPO_STEPS = 500
DPO_LR = 5e-6
DPO_BETA = 0.1
DPO_SAVE_EVERY = 500

BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_GRAD_NORM = 1.0
MAX_LENGTH = 64
NUM_DATA_SAMPLES = 15000

START_TIME = time.time()

# ── Logging ──────────────────────────────────────────────────────────────────

log_path = os.path.join(LOG_DIR, f"train_m4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_file = open(log_path, 'w')

def log(msg):
    t = datetime.now().strftime('%H:%M:%S')
    elapsed = (time.time() - START_TIME) / 3600
    line = f"[{t}] [{elapsed:.1f}h] {msg}"
    print(line, flush=True)
    log_file.write(line + '\n')
    log_file.flush()

_STUDENT_REF = None
_OPT_REF = None

def signal_handler(sig, frame):
    log(f"Received signal {sig}. Saving checkpoint...")
    try:
        save_checkpoint("interrupt", 0, _STUDENT_REF, _OPT_REF)
    except Exception as e:
        log(f"Save failed: {e}")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# ── CRN Components ──────────────────────────────────────────────────────────

from episodic_memory import EpisodicMemory
from reflective_loop import ReflectiveLoop
from skill_composer import SkillComposer

class ResonanceAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, num_frequencies=16, top_k=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_frequencies = num_frequencies
        self.top_k = top_k
        self.head_dim = d_model // num_heads
        self.freq_q = nn.Linear(d_model, num_heads * num_frequencies, bias=False)
        self.freq_k = nn.Linear(d_model, num_heads * num_frequencies, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
    def forward(self, x):
        B, T, D = x.shape; device = x.device
        q = self.freq_q(x).view(B, T, self.num_heads, self.num_frequencies)
        k = self.freq_k(x).view(B, T, self.num_heads, self.num_frequencies)
        freq_scores = F.softmax(q, dim=-1)
        top_k = min(self.top_k, self.num_frequencies)
        top_freq_vals, top_freq_idx = freq_scores.topk(top_k, dim=-1)
        top_freq_vals = top_freq_vals / (top_freq_vals.sum(dim=-1, keepdim=True) + 1e-8)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        out = torch.zeros_like(v)
        for f_idx in range(self.num_frequencies):
            mask = (top_freq_idx == f_idx).any(dim=-1)
            if mask.sum() == 0: continue
            freq_weight = torch.zeros(B, T, self.num_heads, device=device)
            for k_idx in range(top_k):
                match = (top_freq_idx[:, :, :, k_idx] == f_idx)
                freq_weight += match.float() * top_freq_vals[:, :, :, k_idx]
            q_f = q[:, :, :, f_idx]
            k_f = k[:, :, :, f_idx]
            attn_scores = torch.einsum('bih,bjh->bhij', q_f, k_f) / (self.head_dim ** 0.5)
            attn_mask = mask.unsqueeze(2) * mask.unsqueeze(1)
            attn_mask = attn_mask.permute(0, 3, 1, 2)
            attn_scores = attn_scores.masked_fill(~attn_mask.bool(), float('-inf'))
            attn_weights = F.softmax(attn_scores, dim=-1).nan_to_num(0.0)
            attn_out = torch.einsum('bhij,bjhd->bihd', attn_weights, v)
            out += freq_weight.unsqueeze(-1) * attn_out
        out = out.reshape(B, T, D)
        return self.out_proj(out)

class CRNMemoryLayer(nn.Module):
    def __init__(self, d_model, mem_size=512, mem_dim=128, device='cpu'):
        super().__init__()
        self.d_model = d_model
        mem_dtype = torch.bfloat16
        self.memory = EpisodicMemory(d_model, mem_size, mem_dim, device)
        for mod in [self.memory.compress, self.memory.decompress, self.memory.read_gate, self.memory.write_gate, self.memory.relevance_gate]:
            mod.to(device, dtype=mem_dtype)
        self.memory.memory = self.memory.memory.to(dtype=mem_dtype)
        self.memory.temporal_positions = self.memory.temporal_positions.to(dtype=mem_dtype)
        self.read_gate = nn.Linear(d_model, d_model, dtype=mem_dtype).to(device)
        self.write_gate = nn.Linear(d_model, 1, dtype=mem_dtype).to(device)
        self.blend = nn.Parameter(torch.tensor(0.1, dtype=mem_dtype, device=device))
    def read(self, hidden_states):
        if self.memory.temporal_positions.sum() == 0: return hidden_states
        query = self.read_gate(hidden_states.mean(dim=1))
        retrieved, _ = self.memory.read(query, top_k=8)
        return hidden_states + torch.sigmoid(self.blend) * retrieved.unsqueeze(1)
    def write(self, hidden_states):
        if self.training:
            self.memory.write(hidden_states[:, -1, :].mean(dim=0), force=False)
    def get_parameters(self):
        return self.memory.get_parameters() + list(self.read_gate.parameters()) + list(self.write_gate.parameters()) + [self.blend]
    def save(self, path): self.memory.save(path)
    def load(self, path): self.memory.load(path)

# ── Student ─────────────────────────────────────────────────────────────────

class PrajnaStudent(nn.Module):
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        log("Loading E2B student...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL, dtype=torch.bfloat16
        )
        self.model.to(device)
        if device == 'mps':
            embed = self.model.model.language_model.embed_tokens
            embed.weight = torch.nn.Parameter(embed.weight.float())
            embed.embed_scale = embed.embed_scale.float()
        crn_dtype = torch.bfloat16
        self.mem = CRNMemoryLayer(d_model=1536, mem_size=512, mem_dim=128, device=device)
        self.reflection = ReflectiveLoop(d_model=1536, num_corrections=16).to(device, dtype=crn_dtype)
        self.skills = SkillComposer(d_model=1536, num_skills=64, skill_rank=8, top_k=4).to(device, dtype=crn_dtype)
        self.resonance = ResonanceAttention(d_model=1536, num_heads=4, num_frequencies=16, top_k=4).to(device, dtype=crn_dtype)
        self.vocab = 262144
        for p in self.model.parameters(): p.requires_grad = False
        self._hooks = []
        self._register_hooks()
        params = self.get_params()
        log(f"CRN: {sum(p.numel() for p in params):,} params, {len(self._hooks)} hooks")

    def _register_hooks(self):
        layers = self.model.model.language_model.layers
        mid = len(layers) // 2
        hook_dtype = torch.bfloat16
        self._hooks.append(layers[mid].register_forward_pre_hook(
            lambda m, i: (self.mem.read(i[0].to(self.device, dtype=hook_dtype)).to(i[0].device),) + i[1:]
        ))
        self._hooks.append(layers[-1].register_forward_hook(
            lambda m, i, o: (self.mem.write((o[0] if isinstance(o, tuple) else o).to(self.device)), o)[1]
        ))
        for l in layers:
            self._hooks.append(l.register_forward_hook(
                lambda m, i, o: (
                    (self.reflection((o[0] if isinstance(o, tuple) else o).to(self.device, dtype=hook_dtype)).to(o.device),) + o[1:]
                ) if isinstance(o, tuple) else self.reflection(o.to(self.device, dtype=hook_dtype)).to(o.device)
            ))
        for i, l in enumerate(layers):
            if i % 4 == 0:
                self._hooks.append(l.register_forward_hook(
                    lambda m, inp, o: (
                        (self.skills((o[0] if isinstance(o, tuple) else o).to(self.device, dtype=hook_dtype)).to(o.device),) + o[1:]
                    ) if isinstance(o, tuple) else self.skills(o.to(self.device, dtype=hook_dtype)).to(o.device)
                ))
        for i, l in enumerate(layers):
            if i < mid:
                self._hooks.append(l.register_forward_hook(
                    lambda m, inp, o: (
                        (self.resonance((o[0] if isinstance(o, tuple) else o).to(self.device, dtype=hook_dtype)).to(o.device),) + o[1:]
                    ) if isinstance(o, tuple) else self.resonance(o.to(self.device, dtype=hook_dtype)).to(o.device)
                ))

    def forward(self, input_ids, labels=None):
        out = self.model(input_ids=input_ids)
        loss = None
        if labels is not None:
            ce_loss = F.cross_entropy(
                out.logits[:,:-1].reshape(-1, self.vocab),
                labels[:,1:].reshape(-1), ignore_index=-100
            )
            with torch.no_grad():
                preds = out.logits[:,:-1].argmax(dim=-1)
                targets = labels[:,1:]
                is_error = (preds != targets) & (targets != -100)
            contrastive_loss = torch.tensor(0.0, device=self.device)
            if is_error.any():
                if not hasattr(self, '_proj'):
                    proj_dtype = torch.bfloat16
                    self._proj = nn.Linear(self.vocab, 1536, dtype=proj_dtype).to(self.device)
                proj_hidden = self._proj(out.logits.mean(dim=1).unsqueeze(1))
                B = is_error.shape[0]
                correct_dir = torch.randint(0, 16, (B,), device=self.device)
                contrastive_loss = self.reflection.compute_loss(proj_hidden, is_error.any(dim=-1), correct_dir)
            loss = ce_loss + 0.1 * contrastive_loss
        return {'loss': loss, 'ce_loss': ce_loss.item() if labels is not None else 0}

    def forward_dpo(self, input_ids_chosen, input_ids_rejected):
        out_chosen = self.model(input_ids=input_ids_chosen)
        out_rejected = self.model(input_ids=input_ids_rejected)
        chosen_logps = self._get_batch_logps(out_chosen.logits, input_ids_chosen)
        rejected_logps = self._get_batch_logps(out_rejected.logits, input_ids_rejected)
        loss = -F.logsigmoid(DPO_BETA * (chosen_logps - rejected_logps)).mean()
        return {'loss': loss, 'chosen_reward': chosen_logps.mean().item(), 'rejected_reward': rejected_logps.mean().item()}

    def _get_batch_logps(self, logits, labels):
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1]
        mask = labels != -100
        labels[~mask] = 0
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
        return (token_log_probs * mask).sum(dim=-1)

    def get_params(self):
        params = self.mem.get_parameters() + list(self.reflection.parameters()) + list(self.skills.parameters()) + list(self.resonance.parameters())
        if hasattr(self, '_proj'): params += list(self._proj.parameters())
        return params

    def save_memory(self, path): self.mem.save(path)
    def load_memory(self, path): self.mem.load(path)
    def cleanup(self):
        for h in self._hooks: h.remove()
        self._hooks.clear()

# ── Data Generation ─────────────────────────────────────────────────────────

PROMPT_CATEGORIES = {
    "factual_qa": [
        "What is the capital of {country}?",
        "Who invented {thing}?",
        "When did {event} happen?",
        "What is {concept}?",
        "How does {process} work?",
        "What causes {phenomenon}?",
        "Where is {place} located?",
        "What is the formula for {formula}?",
        "Who wrote {work}?",
        "What is the population of {place}?",
    ],
    "reasoning": [
        "Solve step by step: {problem}",
        "If {premise}, what follows?",
        "What is the logical fallacy in: {argument}?",
        "Evaluate: {statement}",
        "Compare {a} and {b}.",
        "What are the pros and cons of {topic}?",
        "Explain the reasoning behind {concept}.",
        "What assumptions underlie {argument}?",
    ],
    "code": [
        "Write a Python function to {task}.",
        "Debug this code: {code}",
        "Explain what this code does: {code}",
        "Optimize this function for {goal}: {code}",
        "Write a SQL query to {task}.",
        "What's wrong with this algorithm: {description}?",
    ],
    "explanation": [
        "Explain {topic} to a beginner.",
        "How does {concept} relate to {concept2}?",
        "What is the difference between {a} and {b}?",
        "Describe the process of {process}.",
        "Why is {concept} important?",
        "What are the key principles of {field}?",
    ],
    "creative": [
        "Write a short story about {scenario}.",
        "Compose a poem about {theme}.",
        "Create a dialogue between {character1} and {character2}.",
        "Write a description of {scene}.",
    ],
    "analysis": [
        "Analyze the causes of {event}.",
        "What are the implications of {trend}?",
        "Evaluate the effectiveness of {approach}.",
        "What factors influence {outcome}?",
        "How has {topic} evolved over time?",
    ],
    "memory": [
        "Remember: My {key} is {value}. What is my {key}?",
        "I previously mentioned {fact}. What was it?",
        "Based on what I told you earlier about {topic}, what do you think?",
    ],
    "classification": [
        "Is {item} an example of {category}? Why or why not?",
        "Classify {item} into one of: {categories}.",
        "What type of {item} is this: {example}?",
    ],
}

TOPICS = {
    "country": ["France", "Japan", "Brazil", "India", "Germany", "Australia", "Nigeria", "Canada", "Mexico", "South Korea"],
    "thing": ["the telephone", "the internet", "the printing press", "the light bulb", "penicillin", "the compass"],
    "event": ["World War II", "the French Revolution", "the moon landing", "the fall of the Berlin Wall"],
    "concept": ["quantum entanglement", "natural selection", "machine learning", "blockchain", "relativity"],
    "process": ["photosynthesis", "nuclear fusion", "evolution", "democracy", "supply and demand"],
    "phenomenon": ["climate change", "tides", "earthquakes", "rainbows", "magnetic fields"],
    "place": ["Mount Everest", "the Mariana Trench", "Sahara Desert", "Great Barrier Reef"],
    "formula": ["area of a circle", "quadratic equation", "Pythagorean theorem", "Einstein's mass-energy"],
    "work": ["Romeo and Juliet", "1984", "The Origin of Species", "The Republic"],
    "problem": ["a train traveling 120km in 2h then 180km in 3h", "finding the derivative of x^3+2x^2-5x+7"],
    "topic": ["remote work", "universal basic income", "space exploration", "artificial intelligence", "nuclear energy"],
    "a": ["stack", "TCP", "HTTP", "SQL", "Python"],
    "b": ["queue", "UDP", "HTTPS", "NoSQL", "JavaScript"],
    "task": ["sort a list", "find primes", "check palindrome", "merge sorted arrays", "binary search"],
    "code": ["def fib(n): return fib(n-1)+fib(n-2)", "for i in range(10): print(i)"],
    "goal": ["time complexity", "readability", "memory usage"],
    "description": ["bubble sort vs quicksort", "recursive vs iterative fibonacci"],
    "field": ["physics", "computer science", "economics", "psychology"],
    "scenario": ["a robot discovering emotions", "first contact with aliens", "time travel paradox"],
    "theme": ["time", "solitude", "technology", "nature"],
    "character1": ["Socrates", "Einstein", "a detective"],
    "character2": ["a modern student", "Newton", "a suspect"],
    "scene": ["a futuristic city", "deep ocean", "outer space"],
    "trend": ["AI adoption", "climate policy", "remote work"],
    "approach": ["agile methodology", "universal healthcare", "renewable energy"],
    "outcome": ["economic growth", "educational success", "team performance"],
    "item": ["a dolphin", "a virus", "a democracy"],
    "category": ["mammal", "living organism", "government type"],
    "categories": ["animal", "plant", "mineral"],
    "example": ["a whale is a fish or mammal?"],
    "key": ["birthday", "favorite color", "name"],
    "value": ["March 15", "blue", "Alice"],
    "fact": ["the meeting is at 3pm", "the deadline is Friday"],
    "premise": ["it rains, the ground gets wet", "all men are mortal, Socrates is a man"],
    "argument": ["we should ban all cars because one person was injured", "this medicine worked once, so it always works"],
    "statement": ["the earth is flat", "vaccines cause autism", "the earth orbits the sun"],
    "concept2": ["evolution", "genetics", "ecology"],
}

def fill_template(template):
    import re
    def replace(match):
        key = match.group(1)
        if key in TOPICS:
            return random.choice(TOPICS[key])
        return match.group(0)
    return re.sub(r'\{(\w+)\}', replace, template)

def generate_phase0_data():
    log("=" * 60)
    log("PHASE 0: Generating training data from E4B-it")
    log("=" * 60)

    data_file = os.path.join(DATA_DIR, 'teacher_data_m4.json')
    if os.path.exists(data_file):
        with open(data_file) as f:
            existing = json.load(f)
        if len(existing) >= NUM_DATA_SAMPLES:
            log(f"Data exists: {len(existing)} samples. Skipping generation.")
            return data_file

    log(f"Loading E4B-it teacher (bf16, ~10GB)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, dtype=torch.bfloat16
    )
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    for p in teacher.parameters(): p.requires_grad = False
    log(f"Teacher loaded. Memory: {torch.mps.current_allocated_memory()/1e9:.1f} GB" if hasattr(torch, 'mps') else "Teacher loaded (CPU)")

    all_prompts = []
    for category, templates in PROMPT_CATEGORIES.items():
        for _ in range(NUM_DATA_SAMPLES // len(PROMPT_CATEGORIES)):
            template = random.choice(templates)
            prompt = fill_template(template)
            all_prompts.append({"prompt": prompt, "category": category})

    random.shuffle(all_prompts)
    all_prompts = all_prompts[:NUM_DATA_SAMPLES]

    samples = []
    t0 = time.time()
    for i, item in enumerate(all_prompts):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["prompt"]},
        ]
        input_text = teacher_tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = teacher_tok(input_text, return_tensors='pt')
        with torch.no_grad():
            out = teacher.generate(
                **inputs, max_new_tokens=200,
                temperature=0.7, top_p=0.9, do_sample=True
            )
        response = teacher_tok.decode(out[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
        samples.append({
            "prompt": item["prompt"],
            "response": response.strip(),
            "category": item["category"],
        })
        if (i+1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(all_prompts) - i - 1) / rate / 3600
            log(f"  {i+1}/{len(all_prompts)} ({elapsed:.0f}s, {rate:.1f}/s, ETA: {eta:.1f}h)")

    with open(data_file, 'w') as f:
        json.dump(samples, f, indent=2)
    log(f"Saved {len(samples)} samples to {data_file}")

    del teacher
    if hasattr(torch, 'mps'):
        torch.mps.empty_cache()
    log(f"Teacher freed.")
    return data_file

# ── DPO Preference Pairs ────────────────────────────────────────────────────

def generate_dpo_pairs():
    log("Generating DPO preference pairs...")
    dpo_file = os.path.join(DATA_DIR, 'dpo_pairs_m4.json')

    if os.path.exists(dpo_file):
        with open(dpo_file) as f:
            existing = json.load(f)
        if len(existing) >= 3000:
            log(f"DPO pairs exist: {len(existing)} pairs. Skipping.")
            return dpo_file

    with open(os.path.join(DATA_DIR, 'teacher_data_m4.json')) as f:
        data = json.load(f)

    bloated_prefixes = [
        "Great question! I'd be happy to help you with that. ",
        "That's an excellent question! Let me provide a comprehensive answer. ",
        "I'm glad you asked! Here's a detailed explanation. ",
        "Thank you for your interest! Let me elaborate on that. ",
        "What a wonderful question! I'll do my best to explain. ",
        "Absolutely! I can certainly help with that. ",
        "That's a really interesting topic! Let me share my knowledge. ",
    ]

    bloated_suffixes = [
        " In conclusion, that covers the main points. I hope this helps! Let me know if you have any other questions.",
        " To sum up, these are the key takeaways. Feel free to ask for more details!",
        " I hope this explanation was clear and helpful. Don't hesitate to ask follow-up questions!",
        " This is just a brief overview, but it should give you a good starting point. Happy to dive deeper!",
    ]

    hallucination_additions = [
        " According to recent studies, this has been proven to be 97% accurate.",
        " Experts say this is the most important factor in the field.",
        " This was first discovered in 1847 by Professor James Wilson.",
        " Statistics show that 85% of professionals agree with this.",
    ]

    dpo_pairs = []
    for sample in data[:5000]:
        chosen = sample["response"]
        if not chosen or len(chosen) < 20:
            continue

        rejected_type = random.choice(["bloated", "hallucinated"])
        if rejected_type == "bloated":
            prefix = random.choice(bloated_prefixes)
            suffix = random.choice(bloated_suffixes)
            rejected = prefix + chosen + suffix
        else:
            insertion_point = random.randint(1, max(1, len(chosen.split('.')) - 1))
            sentences = chosen.split('.')
            hallucination = random.choice(hallucination_additions)
            sentences.insert(insertion_point, hallucination)
            rejected = '.'.join(sentences)

        dpo_pairs.append({
            "prompt": sample["prompt"],
            "chosen": chosen,
            "rejected": rejected,
        })

    random.shuffle(dpo_pairs)
    dpo_pairs = dpo_pairs[:5000]

    with open(dpo_file, 'w') as f:
        json.dump(dpo_pairs, f, indent=2)
    log(f"Saved {len(dpo_pairs)} DPO pairs")
    return dpo_file

# ── Datasets ────────────────────────────────────────────────────────────────

class SFTDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=MAX_LENGTH):
        with open(data_file) as f:
            self.samples = json.load(f)
        self.tok = tokenizer
        self.ml = max_length
        log(f"Dataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        text = f"{s.get('prompt','')}\n\n{s.get('response','')}"
        enc = self.tok(text, truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].squeeze()
        labels = ids.clone()
        labels[enc['attention_mask'].squeeze() == 0] = -100
        return {'input_ids': ids, 'labels': labels}

class DPODataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=MAX_LENGTH):
        with open(data_file) as f:
            self.pairs = json.load(f)
        self.tok = tokenizer
        self.ml = max_length
        log(f"DPO Dataset: {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        p = self.pairs[i]
        chosen_enc = self.tok(p['chosen'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        rejected_enc = self.tok(p['rejected'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        return {
            'chosen_ids': chosen_enc['input_ids'].squeeze(),
            'rejected_ids': rejected_enc['input_ids'].squeeze(),
        }

# ── Checkpoint Save ─────────────────────────────────────────────────────────

def save_checkpoint(tag, loss, student, opt):
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{tag}.pt")
    ckpt = {'step': tag, 'loss': loss}
    if student is not None:
        crn_state = {k: v.cpu() for k, v in student.state_dict().items() if not k.startswith('model')}
        ckpt['crn'] = crn_state
        ckpt['memory_stats'] = student.mem.memory.get_stats()
        ckpt['correction_stats'] = student.reflection.get_correction_stats()
    if opt is not None:
        ckpt['optimizer'] = {k: v.cpu() if hasattr(v, 'cpu') else v for k, v in opt.state_dict().items()}
    torch.save(ckpt, ckpt_path)
    mem_path = os.path.join(CKPT_DIR, f"memory_{tag}.json")
    if student is not None:
        student.save_memory(mem_path)
    log(f"  Saved: {ckpt_path}")
    return ckpt_path

def find_latest_checkpoint():
    ckpts = sorted(Path(CKPT_DIR).glob("ckpt_*.pt"))
    if not ckpts:
        return None
    latest = ckpts[-1]
    return str(latest)

def get_resume_state():
    """Auto-detect latest checkpoint and return (student_state, step, phase)."""
    latest = find_latest_checkpoint()
    if latest is None:
        return None, 0, None
    try:
        ckpt = torch.load(latest, map_location='cpu', weights_only=False)
        step = ckpt.get('step', 0)
        tag = os.path.basename(latest)
        if 'dpo_' in tag:
            phase = 2
        elif 'sft_' in tag:
            phase = 1
        else:
            phase = None
        log(f"Found checkpoint: {latest} (step={step}, phase={phase})")
        return ckpt.get('crn', None), step, phase
    except Exception as e:
        log(f"Failed to load checkpoint {latest}: {e}")
        return None, 0, None

def check_disk_space(min_gb=2.0):
    st = os.statvfs(os.path.expanduser("~"))
    free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
    if free_gb < min_gb:
        log(f"WARNING: Low disk space: {free_gb:.1f} GB free (need {min_gb:.1f} GB)")
        return False
    log(f"Disk space: {free_gb:.1f} GB free")
    return True

def check_memory():
    import subprocess
    try:
        result = subprocess.run(['vm_stat'], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'Pages free' in line:
                free_pages = int(line.split(':')[1].strip().rstrip('.'))
                free_gb = free_pages * 16384 / (1024**3)
                log(f"Free memory: {free_gb:.1f} GB")
                return free_gb
    except:
        pass
    return None

# ── Phase 1: SFT Distillation ──────────────────────────────────────────────

def phase1_sft(student, data_file, start_step=0):
    log("=" * 60)
    log(f"PHASE 1: SFT Distillation (starting from step {start_step})")
    log("=" * 60)

    dataset = SFTDataset(data_file, student.tok)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=SFT_LR, weight_decay=0.01)
    remaining_steps = SFT_STEPS - start_step
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining_steps, eta_min=SFT_LR * 0.1)

    losses = []; t_start = time.time(); step = start_step
    student.train()

    for epoch in range(1000):
        for batch in dataloader:
            if step >= SFT_STEPS: break

            input_ids = batch['input_ids']
            labels = batch['labels']

            out = student(input_ids, labels)
            loss = out['loss'] / GRAD_ACCUM
            if torch.isnan(loss):
                scheduler.step()
                step += 1
                continue

            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                opt.step(); opt.zero_grad()
                scheduler.step()

            losses.append(loss.item() * GRAD_ACCUM)
            step += 1

            if step % SFT_LOG_EVERY == 0:
                avg = sum(losses[-SFT_LOG_EVERY:]) / SFT_LOG_EVERY
                elapsed = (time.time() - t_start) / 3600
                rate = (step - start_step) / (time.time() - t_start)
                eta = (SFT_STEPS - step) / rate / 3600 if rate > 0 else 0
                log(f"  Step {step:5d}/{SFT_STEPS} | Loss: {loss.item()*GRAD_ACCUM:.4f} | Avg: {avg:.4f} | {rate:.2f}/s | ETA: {eta:.1f}h")

            if step % SFT_SAVE_EVERY == 0:
                save_checkpoint(f"sft_{step}", loss.item()*GRAD_ACCUM, student, opt)

    final_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else 0
    save_checkpoint("sft_final", final_loss, student, opt)
    log(f"Phase 1 complete. Steps: {step}, Final loss: {final_loss:.4f}")
    return student

# ── Phase 2: DPO Preference ─────────────────────────────────────────────────

def phase2_dpo(student, dpo_file, start_step=0):
    log("=" * 60)
    log(f"PHASE 2: DPO Preference Learning (starting from step {start_step})")
    log("=" * 60)

    dataset = DPODataset(dpo_file, student.tok)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=DPO_LR, weight_decay=0.01)

    losses = []; t_start = time.time(); step = start_step
    student.train()

    for epoch in range(1000):
        for batch in dataloader:
            if step >= DPO_STEPS: break

            chosen_ids = batch['chosen_ids']
            rejected_ids = batch['rejected_ids']

            out = student.forward_dpo(chosen_ids, rejected_ids)
            loss = out['loss'] / GRAD_ACCUM
            if torch.isnan(loss):
                step += 1
                continue

            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                opt.step(); opt.zero_grad()

            losses.append(loss.item() * GRAD_ACCUM)
            step += 1

            if step % 50 == 0:
                avg = sum(losses[-50:]) / 50
                elapsed = (time.time() - t_start) / 3600
                rate = (step - start_step) / (time.time() - t_start)
                eta = (DPO_STEPS - step) / rate / 3600 if rate > 0 else 0
                log(f"  Step {step:5d}/{DPO_STEPS} | Loss: {loss.item()*GRAD_ACCUM:.4f} | Avg: {avg:.4f} | Chosen: {out['chosen_reward']:.2f} | Rejected: {out['rejected_reward']:.2f}")

            if step % DPO_SAVE_EVERY == 0:
                save_checkpoint(f"dpo_{step}", loss.item()*GRAD_ACCUM, student, opt)

    final_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else 0
    save_checkpoint("dpo_final", final_loss, student, opt)
    log(f"Phase 2 complete. Steps: {step}, Final loss: {final_loss:.4f}")
    return student

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    global _STUDENT_REF, _OPT_REF

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[0, 1, 2], help='Run specific phase')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint (auto-detected)')
    parser.add_argument('--fresh', action='store_true', help='Ignore checkpoints, start fresh')
    args = parser.parse_args()

    log("=" * 60)
    log("PRAJNA M4 OFFLINE TRAINING")
    log("=" * 60)
    device_name = "MPS (GPU)" if torch.backends.mps.is_available() else "CPU"
    log(f"Device: {device_name}")
    log(f"Teacher: {TEACHER_MODEL}")
    log(f"Student: {STUDENT_MODEL}")
    log(f"Estimated time: ~29 hours")
    log(f"Cost: $0")
    log(f"Mode: {'fresh' if args.fresh else 'auto-resume'}")

    if not check_disk_space(min_gb=3.0):
        log("ABORT: Insufficient disk space")
        return

    free_mem = check_memory()
    if free_mem and free_mem < 2.0:
        log(f"WARNING: Only {free_mem:.1f} GB free RAM. Close other apps.")

    try:
        # Phase 0: Data generation
        if args.phase == 0 or args.phase is None:
            data_file = generate_phase0_data()
        else:
            data_file = os.path.join(DATA_DIR, 'teacher_data_m4.json')
            if not os.path.exists(data_file):
                log(f"ERROR: {data_file} not found. Run --phase 0 first.")
                return

        if args.phase == 0 and args.phase is not None:
            log("Phase 0 complete.")
            check_memory()
            return

        # DPO pairs
        dpo_file = generate_dpo_pairs()

        # Load student
        device = 'cpu'
        student = PrajnaStudent(device=device)
        _STUDENT_REF = student
        log(f"Student ready.")
        check_memory()

        # Auto-resume from checkpoint
        sft_start = 0
        dpo_start = 0
        if not args.fresh:
            crn_state, step, phase = get_resume_state()
            if crn_state is not None:
                student.load_state_dict(crn_state, strict=False)
                log(f"Restored CRN weights from checkpoint (step={step})")
                if phase == 1:
                    sft_start = step
                    log(f"Resuming SFT from step {step}")
                elif phase == 2:
                    dpo_start = step
                    log(f"Resuming DPO from step {step}")
                elif 'sft_final' in find_latest_checkpoint():
                    log("SFT already complete, resuming DPO from step 0")
                else:
                    log(f"Checkpoint found but phase unclear, resuming SFT from step {step}")
                    sft_start = step

        # Phase 1: SFT
        if args.phase == 1 or args.phase is None:
            student = phase1_sft(student, data_file, start_step=sft_start)
            _STUDENT_REF = student

        # Phase 2: DPO
        if args.phase == 2 or args.phase is None:
            student = phase2_dpo(student, dpo_file, start_step=dpo_start)
            _STUDENT_REF = student

        save_checkpoint("complete", 0, student, None)
        elapsed = (time.time() - START_TIME) / 3600
        log("=" * 60)
        log("ALL PHASES COMPLETE")
        log(f"Total time: {elapsed:.1f} hours")
        log(f"Memory stats: {student.mem.memory.get_stats()}")
        log(f"Correction stats: {student.reflection.get_correction_stats()}")
        student.cleanup()

    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc()
        log("Saving checkpoint before exit...")
        try:
            save_checkpoint("error", 0, _STUDENT_REF, _OPT_REF)
        except Exception as e2:
            log(f"Checkpoint save also failed: {e2}")

if __name__ == "__main__":
    main()

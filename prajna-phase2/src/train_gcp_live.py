#!/usr/bin/env python3
"""
Prajna GCP Live Distillation + Alignment (L4 24GB)

Three-phase training pipeline:
  Phase 0: Generate 15,000 samples from E4B-it teacher
  Phase 1: Live distillation — student matches teacher logits
  Phase 2: SFT — train on efficient-communication responses
  Phase 3: DPO — preference pairs: efficient > bloated/hallucinated

AUTO-SHUTDOWN on completion. Budget cap: $5.
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
from datetime import datetime

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

# Phase 1: Live distillation
DISTILL_STEPS = 10000
DISTILL_LR = 2e-4
DISTILL_WARMUP = 100
DISTILL_SAVE_EVERY = 500

# Phase 2: SFT alignment
SFT_STEPS = 5000
SFT_LR = 5e-5
SFT_SAVE_EVERY = 500

# Phase 3: DPO preference
DPO_STEPS = 3000
DPO_LR = 5e-6
DPO_BETA = 0.1
DPO_SAVE_EVERY = 500

BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_GRAD_NORM = 1.0
MAX_LENGTH = 128
NUM_DATA_SAMPLES = 15000

SHUTDOWN_ON_DONE = True
SHUTDOWN_DELAY_MIN = 3
BUDGET_CAP_USD = 5.00
INSTANCE_COST_PER_SEC = 250.0 / 3600.0  # $0.25/hr spot

START_TIME = time.time()

# ── Logging ──────────────────────────────────────────────────────────────────

log_path = os.path.join(LOG_DIR, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
log_file = open(log_path, 'w')

def log(msg):
    t = datetime.now().strftime('%H:%M:%S')
    elapsed = (time.time() - START_TIME) / 60
    cost = (time.time() - START_TIME) * INSTANCE_COST_PER_SEC
    line = f"[{t}] [{elapsed:.1f}m ${cost:.2f}] {msg}"
    print(line, flush=True)
    log_file.write(line + '\n')
    log_file.flush()

def check_budget():
    cost = (time.time() - START_TIME) * INSTANCE_COST_PER_SEC
    if cost >= BUDGET_CAP_USD:
        log(f"BUDGET CAP REACHED: ${cost:.2f} >= ${BUDGET_CAP_USD}. Shutting down.")
        return False
    return True

# ── Graceful Shutdown ────────────────────────────────────────────────────────

def shutdown():
    log("Initiating shutdown...")
    if SHUTDOWN_ON_DONE:
        log(f"Instance will shut down in {SHUTDOWN_DELAY_MIN} minutes.")
        os.system(f"sudo shutdown -P +{SHUTDOWN_DELAY_MIN}")
    else:
        log("Shutdown disabled. Instance will keep running.")

def signal_handler(sig, frame):
    log(f"Received signal {sig}. Saving and shutting down...")
    try:
        save_checkpoint("interrupt", 0, None, None)
    except:
        pass
    shutdown()
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
        self.memory = EpisodicMemory(d_model, mem_size, mem_dim, device)
        for mod in [self.memory.compress, self.memory.decompress, self.memory.read_gate, self.memory.write_gate, self.memory.relevance_gate]:
            mod.to(device, dtype=torch.bfloat16)
        self.memory.memory = self.memory.memory.to(dtype=torch.bfloat16)
        self.memory.temporal_positions = self.memory.temporal_positions.to(dtype=torch.bfloat16)
        self.read_gate = nn.Linear(d_model, d_model, dtype=torch.bfloat16).to(device)
        self.write_gate = nn.Linear(d_model, 1, dtype=torch.bfloat16).to(device)
        self.blend = nn.Parameter(torch.tensor(0.1, dtype=torch.bfloat16, device=device))
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
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        log("Loading E2B student...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(STUDENT_MODEL)
        self.model = AutoModelForCausalLM.from_pretrained(
            STUDENT_MODEL, torch_dtype=torch.bfloat16, device_map='auto'
        )
        self.mem = CRNMemoryLayer(d_model=1536, mem_size=512, mem_dim=128, device=device)
        self.reflection = ReflectiveLoop(d_model=1536, num_corrections=16).to(device, dtype=torch.bfloat16)
        self.skills = SkillComposer(d_model=1536, num_skills=64, skill_rank=8, top_k=4).to(device, dtype=torch.bfloat16)
        self.resonance = ResonanceAttention(d_model=1536, num_heads=4, num_frequencies=16, top_k=4).to(device, dtype=torch.bfloat16)
        self.vocab = 262144
        for p in self.model.parameters(): p.requires_grad = False
        self._hooks = []
        self._register_hooks()
        params = self.get_params()
        log(f"CRN: {sum(p.numel() for p in params):,} params, {len(self._hooks)} hooks")
        log(f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    def _register_hooks(self):
        layers = self.model.model.language_model.layers
        mid = len(layers) // 2
        self._hooks.append(layers[mid].register_forward_pre_hook(
            lambda m, i: (self.mem.read(i[0].to(self.device, dtype=torch.bfloat16)).to(i[0].device),) + i[1:]
        ))
        self._hooks.append(layers[-1].register_forward_hook(
            lambda m, i, o: (self.mem.write((o[0] if isinstance(o, tuple) else o).to(self.device)), o)[1]
        ))
        for l in layers:
            self._hooks.append(l.register_forward_hook(
                lambda m, i, o: (
                    (self.reflection((o[0] if isinstance(o, tuple) else o).to(self.device, dtype=torch.bfloat16)).to(o.device),) + o[1:]
                ) if isinstance(o, tuple) else self.reflection(o.to(self.device, dtype=torch.bfloat16)).to(o.device)
            ))
        for i, l in enumerate(layers):
            if i % 4 == 0:
                self._hooks.append(l.register_forward_hook(
                    lambda m, inp, o: (
                        (self.skills((o[0] if isinstance(o, tuple) else o).to(self.device, dtype=torch.bfloat16)).to(o.device),) + o[1:]
                    ) if isinstance(o, tuple) else self.skills(o.to(self.device, dtype=torch.bfloat16)).to(o.device)
                ))
        for i, l in enumerate(layers):
            if i < mid:
                self._hooks.append(l.register_forward_hook(
                    lambda m, inp, o: (
                        (self.resonance((o[0] if isinstance(o, tuple) else o).to(self.device, dtype=torch.bfloat16)).to(o.device),) + o[1:]
                    ) if isinstance(o, tuple) else self.resonance(o.to(self.device, dtype=torch.bfloat16)).to(o.device)
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
                    self._proj = nn.Linear(self.vocab, 1536, dtype=torch.bfloat16).to(self.device)
                proj_hidden = self._proj(out.logits.mean(dim=1).unsqueeze(1))
                B = is_error.shape[0]
                correct_dir = torch.randint(0, 16, (B,), device=self.device)
                contrastive_loss = self.reflection.compute_loss(proj_hidden, is_error.any(dim=-1), correct_dir)
            loss = ce_loss + 0.1 * contrastive_loss
        return {'logits': out.logits, 'loss': loss}

    def forward_distill(self, input_ids, teacher_logits):
        out = self.model(input_ids=input_ids)
        student_log_probs = F.log_softmax(out.logits[:, :-1] / 1.0, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits[:, :-1] / 1.0, dim=-1)
        kl_loss = F.kl_div(
            student_log_probs, teacher_log_probs,
            reduction='batchmean', log_target=True
        )
        return {'loss': kl_loss}

    def forward_dpo(self, input_ids_chosen, input_ids_rejected):
        out_chosen = self.model(input_ids=input_ids_chosen)
        out_rejected = self.model(input_ids=input_ids_rejected)
        chosen_logps = self._get_batch_logps(out_chosen.logits, input_ids_chosen)
        rejected_logps = self._get_batch_logps(out_rejected.logits, input_ids_rejected)
        loss = -F.logsigmoid(DPO_BETA * (chosen_logps - rejected_logps)).mean()
        return {'loss': loss, 'chosen_reward': (chosen_logps).mean().item(), 'rejected_reward': (rejected_logps).mean().item()}

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
    "response": ["Paris is the capital of France.", "Water boils at 100C."],
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

    data_file = os.path.join(DATA_DIR, 'teacher_data_concise.json')
    if os.path.exists(data_file):
        with open(data_file) as f:
            existing = json.load(f)
        if len(existing) >= NUM_DATA_SAMPLES:
            log(f"Data exists: {len(existing)} samples. Skipping generation.")
            return data_file

    log(f"Loading E4B-it teacher...")
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL,
        device_map='auto',
        torch_dtype=torch.bfloat16,
    )
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    for p in teacher.parameters(): p.requires_grad = False
    log(f"Teacher loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

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
        inputs = teacher_tok(input_text, return_tensors='pt').to(teacher.device)
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
        if (i+1) % 500 == 0:
            elapsed = time.time() - t0
            rate = (i+1) / elapsed
            eta = (len(all_prompts) - i - 1) / rate
            log(f"  {i+1}/{len(all_prompts)} ({elapsed:.0f}s, {rate:.1f} samples/s, ETA: {eta:.0f}s)")

    with open(data_file, 'w') as f:
        json.dump(samples, f, indent=2)
    log(f"Saved {len(samples)} samples to {data_file}")

    del teacher
    torch.cuda.empty_cache()
    log(f"Teacher freed. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")
    return data_file

# ── DPO Preference Pairs ────────────────────────────────────────────────────

def generate_dpo_pairs():
    log("Generating DPO preference pairs...")
    dpo_file = os.path.join(DATA_DIR, 'dpo_pairs.json')

    if os.path.exists(dpo_file):
        with open(dpo_file) as f:
            existing = json.load(f)
        if len(existing) >= 3000:
            log(f"DPO pairs exist: {len(existing)} pairs. Skipping.")
            return dpo_file

    with open(os.path.join(DATA_DIR, 'teacher_data_concise.json')) as f:
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

class DistillationDataset(Dataset):
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
        ckpt['crn'] = {k: v for k, v in student.state_dict().items() if not k.startswith('model')}
        ckpt['memory_stats'] = student.mem.memory.get_stats()
        ckpt['correction_stats'] = student.reflection.get_correction_stats()
    if opt is not None:
        ckpt['optimizer'] = opt.state_dict()
    torch.save(ckpt, ckpt_path)
    mem_path = os.path.join(CKPT_DIR, f"memory_{tag}.json")
    if student is not None:
        student.save_memory(mem_path)
    log(f"  Saved: {ckpt_path}")
    return ckpt_path

# ── Phase 1: Live Distillation ──────────────────────────────────────────────

def phase1_distill(student, data_file):
    log("=" * 60)
    log("PHASE 1: Live Distillation")
    log("=" * 60)

    log("Loading E4B teacher (bf16) for live logit generation...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    teacher = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, device_map='auto'
    )
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    for p in teacher.parameters(): p.requires_grad = False
    log(f"Teacher loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

    dataset = DistillationDataset(data_file, student.tok)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=DISTILL_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=DISTILL_STEPS, eta_min=DISTILL_LR * 0.1)

    losses = []; t_start = time.time(); step = 0
    student.train()

    for epoch in range(100):
        for batch in dataloader:
            if step >= DISTILL_STEPS: break
            if not check_budget(): break

            input_ids = batch['input_ids'].to('cuda')
            labels = batch['labels'].to('cuda')

            with torch.no_grad():
                teacher_out = teacher(input_ids=input_ids)
                teacher_logits = teacher_out.logits.float()

            out = student.forward_distill(input_ids, teacher_logits)
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

            if step % 10 == 0:
                avg = sum(losses[-10:]) / 10
                vram = torch.cuda.memory_allocated()/1e9
                log(f"  Step {step:5d} | Loss: {loss.item()*GRAD_ACCUM:.4f} | Avg: {avg:.4f} | VRAM: {vram:.1f}GB")

            if step % DISTILL_SAVE_EVERY == 0:
                save_checkpoint(f"distill_{step}", loss.item()*GRAD_ACCUM, student, opt)

    final_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else 0
    save_checkpoint("distill_final", final_loss, student, opt)
    del teacher
    torch.cuda.empty_cache()
    log(f"Phase 1 complete. Steps: {step}, Final loss: {final_loss:.4f}")
    return student

# ── Phase 2: SFT Alignment ─────────────────────────────────────────────────

def phase2_sft(student, data_file):
    log("=" * 60)
    log("PHASE 2: SFT Alignment (efficient communication)")
    log("=" * 60)

    dataset = DistillationDataset(data_file, student.tok)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=SFT_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=SFT_STEPS, eta_min=SFT_LR * 0.1)

    losses = []; t_start = time.time(); step = 0
    student.train()

    for epoch in range(100):
        for batch in dataloader:
            if step >= SFT_STEPS: break
            if not check_budget(): break

            input_ids = batch['input_ids'].to('cuda')
            labels = batch['labels'].to('cuda')

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

            if step % 10 == 0:
                avg = sum(losses[-10:]) / 10
                vram = torch.cuda.memory_allocated()/1e9
                log(f"  Step {step:5d} | Loss: {loss.item()*GRAD_ACCUM:.4f} | Avg: {avg:.4f} | VRAM: {vram:.1f}GB")

            if step % SFT_SAVE_EVERY == 0:
                save_checkpoint(f"sft_{step}", loss.item()*GRAD_ACCUM, student, opt)

    final_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else 0
    save_checkpoint("sft_final", final_loss, student, opt)
    log(f"Phase 2 complete. Steps: {step}, Final loss: {final_loss:.4f}")
    return student

# ── Phase 3: DPO Preference ─────────────────────────────────────────────────

def phase3_dpo(student, dpo_file):
    log("=" * 60)
    log("PHASE 3: DPO Preference Learning")
    log("=" * 60)

    dataset = DPODataset(dpo_file, student.tok)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=DPO_LR, weight_decay=0.01)

    losses = []; t_start = time.time(); step = 0
    student.train()

    for epoch in range(100):
        for batch in dataloader:
            if step >= DPO_STEPS: break
            if not check_budget(): break

            chosen_ids = batch['chosen_ids'].to('cuda')
            rejected_ids = batch['rejected_ids'].to('cuda')

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

            if step % 10 == 0:
                avg = sum(losses[-10:]) / 10
                vram = torch.cuda.memory_allocated()/1e9
                log(f"  Step {step:5d} | Loss: {loss.item()*GRAD_ACCUM:.4f} | Avg: {avg:.4f} | Chosen: {out['chosen_reward']:.2f} | Rejected: {out['rejected_reward']:.2f}")

            if step % DPO_SAVE_EVERY == 0:
                save_checkpoint(f"dpo_{step}", loss.item()*GRAD_ACCUM, student, opt)

    final_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else 0
    save_checkpoint("dpo_final", final_loss, student, opt)
    log(f"Phase 3 complete. Steps: {step}, Final loss: {final_loss:.4f}")
    return student

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("PRAJNA LIVE DISTILLATION + ALIGNMENT")
    log("=" * 60)
    log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    log(f"Teacher: {TEACHER_MODEL}")
    log(f"Student: {STUDENT_MODEL}")
    log(f"Budget cap: ${BUDGET_CAP_USD}")

    try:
        data_file = generate_phase0_data()
        dpo_file = generate_dpo_pairs()

        student = PrajnaStudent(device='cuda')
        log(f"Student ready. VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

        student = phase1_distill(student, data_file)
        if not check_budget():
            save_checkpoint("budget_exit", 0, student, None)
            student.cleanup()
            shutdown()
            return

        student = phase2_sft(student, data_file)
        if not check_budget():
            save_checkpoint("budget_exit", 0, student, None)
            student.cleanup()
            shutdown()
            return

        student = phase3_dpo(student, dpo_file)

        save_checkpoint("complete", 0, student, None)
        elapsed = (time.time() - START_TIME) / 60
        cost = (time.time() - START_TIME) * INSTANCE_COST_PER_SEC
        log("=" * 60)
        log("ALL PHASES COMPLETE")
        log(f"Total time: {elapsed:.1f} min")
        log(f"Total cost: ${cost:.2f}")
        log(f"Memory stats: {student.mem.memory.get_stats()}")
        log(f"Correction stats: {student.reflection.get_correction_stats()}")
        student.cleanup()

    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc()
        log("Saving checkpoint before shutdown...")
        try:
            save_checkpoint("error", 0, None, None)
        except:
            pass
    finally:
        shutdown()

if __name__ == "__main__":
    main()

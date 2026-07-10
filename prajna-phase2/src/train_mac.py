#!/usr/bin/env python3
"""Prajna Multi-Layer CRN Training — Mac M4 16GB

Architecture: CRN injected at every 4th layer (8 points in 40-layer Gemma 4 E2B).
Base model forward in no_grad (frozen). CRN trained with gradients.
Memory strategy: device_map='auto' loads directly to MPS (no CPU→MPS copy).
"""
import torch, os, json, time, glob, random, gc, sys, traceback
from pathlib import Path
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- Config ----
BASE = './prajna'
DATA_DIR = f'{BASE}/data'
CKPT_DIR = f'{BASE}/checkpoints'
LOG_DIR = f'{BASE}/logs'
STATE_FILE = f'{BASE}/state.json'
for d in [DATA_DIR, CKPT_DIR, LOG_DIR]: os.makedirs(d, exist_ok=True)

SFT_STEPS = 2000
DPO_STEPS = 500
SFT_LR = 3e-4
DPO_LR = 5e-6
DPO_BETA = 0.1
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_GRAD_NORM = 1.0
MAX_LENGTH = 32
SAVE_EVERY = 50
LOG_EVERY = 10
INJECT_EVERY = 8
CRN_MIX_INIT = 0.05

# Use CPU for base model (10.2GB model too large for 16GB MPS)
# MPS can't fit 10.2GB model + OS overhead on 16GB RAM
DEVICE = 'cpu'
print(f'Device: {DEVICE} (CPU for base model, CRN components on CPU too)')
print(f'This is stable — no session timeouts, runs to completion')

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f: return json.load(f)
    return {'phase': 'data_prep', 'sft_step': 0, 'dpo_step': 0,
            'sft_complete': False, 'dpo_complete': False}

def save_state(s):
    with open(STATE_FILE, 'w') as f: json.dump(s, f, indent=2)

def find_latest_ckpt(p='sft'):
    c = glob.glob(f'{CKPT_DIR}/{p}_*.pt')
    if not c: return None
    def step_of(f):
        try: return int(f.split(f'/{p}_')[-1].split('.pt')[0])
        except: return -1
    return sorted(c, key=step_of)[-1]

state = load_state()
teacher_file = f'{DATA_DIR}/teacher_data.json'
dpo_file = f'{DATA_DIR}/dpo_pairs.json'
print(f'State: phase={state["phase"]}, sft={state["sft_step"]}, dpo={state["dpo_step"]}')

# ---- CRN Components ----
class ResonanceAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, num_frequencies=8, top_k=2):
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
        B, T, D = x.shape
        device = x.device
        q = self.freq_q(x).view(B, T, self.num_heads, self.num_frequencies)
        k = self.freq_k(x).view(B, T, self.num_heads, self.num_frequencies)
        freq_scores = F.softmax(q, dim=-1)
        top_freq_vals, top_freq_idx = freq_scores.topk(min(self.top_k, self.num_frequencies), dim=-1)
        top_freq_vals = top_freq_vals / (top_freq_vals.sum(dim=-1, keepdim=True) + 1e-8)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Vectorized frequency attention — no Python loops
        # 1. Compute per-frequency importance weights via scatter_add
        freq_weight = torch.zeros(B, T, self.num_heads, self.num_frequencies, device=device)
        freq_weight.scatter_add_(3, top_freq_idx, top_freq_vals)

        # 2. Compute attention scores for ALL frequencies: [B, H, F, T, T]
        attn = torch.einsum('bihf,bjhf->bhfij', q, k) / (self.head_dim ** 0.5)

        # 3. Mask: token i attends to j if they share any top frequency
        idx_i = top_freq_idx.unsqueeze(2).expand(-1, -1, -1, -1, self.top_k)  # [B,T,H,K,K]
        idx_j = top_freq_idx.unsqueeze(3).expand(-1, -1, -1, self.top_k, -1)  # [B,T,T,K,K]
        # For each frequency f, mask[b,h,f,i,j] = True if f is in both i's and j's top-k
        # Create frequency membership: [B, T, H, F]
        freq_membership = torch.zeros(B, T, self.num_heads, self.num_frequencies, device=device, dtype=torch.bool)
        freq_membership.scatter_(3, top_freq_idx, True)
        # Shared freq between tokens i and j: [B, H, T, T]
        shared = torch.einsum('bthf,bshf->bhts', freq_membership.float(), freq_membership.float()) > 0
        # Expand to frequency dim: [B, H, F, T, T]
        mask = shared.unsqueeze(2).expand(-1, -1, self.num_frequencies, -1, -1)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1).nan_to_num(0.0)

        # 4. Weighted output: attn [B,H,F,T,T] x v_perm [B,H,T,Dh] -> [B,H,F,T,Dh]
        v_perm = v.permute(0, 2, 1, 3)  # [B, H, T, Dh]
        out = torch.einsum('bhfij,bhjd->bhfid', attn, v_perm)  # [B, H, F, T, Dh]
        fw = freq_weight.permute(0, 2, 3, 1).unsqueeze(-1)  # [B, H, F, T, 1]
        out = (out * fw).sum(dim=2)  # [B, H, T, Dh]
        out = out.permute(0, 2, 1, 3)  # [B, T, H, Dh]
        return self.out_proj(out.reshape(B, T, D))

class EpisodicMemory:
    def __init__(self, d_model, mem_size=512, mem_dim=128, device='cuda'):
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.d_model = d_model
        self.device = device
        self.memory = torch.zeros(mem_size, mem_dim, device=device)
        self.temporal_positions = torch.zeros(mem_size, device=device)
        self.write_ptr = 0
        self.step_count = 0
        self.compress = nn.Linear(d_model, mem_dim).to(device)
        self.decompress = nn.Linear(mem_dim, d_model).to(device)
        self.read_gate = nn.Linear(d_model, mem_dim).to(device)
        self.write_gate = nn.Linear(d_model, 1).to(device)
        self.relevance_gate = nn.Linear(d_model + mem_dim, 1).to(device)

    def get_parameters(self):
        return (list(self.compress.parameters()) + list(self.decompress.parameters()) +
                list(self.read_gate.parameters()) + list(self.write_gate.parameters()) +
                list(self.relevance_gate.parameters()))

    def read(self, query, top_k=8):
        if query.dim() == 1: query = query.unsqueeze(0)
        B = query.shape[0]
        q_compressed = self.read_gate(query)
        mem_expanded = self.memory.unsqueeze(0).expand(B, -1, -1)
        q_norm = F.normalize(q_compressed, dim=-1)
        mem_norm = F.normalize(mem_expanded, dim=-1)
        sims = torch.bmm(q_norm.unsqueeze(1), mem_norm.transpose(1, 2)).squeeze(1)
        recency = self.temporal_positions / (self.temporal_positions.max() + 1)
        sims = sims + 0.1 * recency.unsqueeze(0)
        top_k = min(top_k, self.mem_size)
        top_vals, top_idx = sims.topk(top_k, dim=-1)
        attn_weights = F.softmax(top_vals, dim=-1)
        retrieved = torch.gather(mem_expanded, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.mem_dim))
        retrieved = (retrieved * attn_weights.unsqueeze(-1)).sum(dim=1)
        return self.decompress(retrieved), attn_weights

    def write(self, content, force=False):
        gate_value = torch.sigmoid(self.write_gate(content.unsqueeze(0))).item()
        if gate_value < 0.5 and not force: return False
        compressed = self.compress(content.detach())
        if self.write_ptr < self.mem_size:
            slot = self.write_ptr
            self.write_ptr += 1
        else:
            slot = self.temporal_positions.argmin().item()
        write_weight = min(gate_value, 0.9)
        self.memory[slot] = (write_weight * compressed + (1 - write_weight) * self.memory[slot].clone()).detach()
        self.step_count += 1
        self.temporal_positions[slot] = self.step_count
        return True

    def save(self, path):
        state = {
            'memory': self.memory.detach().cpu().float().numpy().tolist(),
            'temporal_positions': self.temporal_positions.detach().cpu().float().numpy().tolist(),
            'write_ptr': self.write_ptr, 'step_count': self.step_count
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f: json.dump(state, f)

    def load(self, path):
        with open(path) as f: state = json.load(f)
        self.memory = torch.tensor(state['memory'], dtype=torch.float32, device=self.device)
        self.temporal_positions = torch.tensor(state['temporal_positions'], dtype=torch.float32, device=self.device)
        self.write_ptr = state['write_ptr']
        self.step_count = state['step_count']

class ReflectiveLoop(nn.Module):
    def __init__(self, d_model, num_corrections=16):
        super().__init__()
        self.num_corrections = num_corrections
        self.d_model = d_model
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, num_corrections + 1)
        )
        self.correction_directions = nn.Parameter(torch.randn(num_corrections, d_model) * 0.01)
        self.thresholds = nn.Parameter(torch.ones(num_corrections) * 0.5)
        self.confidence_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_state, return_correction_id=False):
        pooled = hidden_state.mean(dim=1) if hidden_state.dim() == 3 else hidden_state
        scores = self.critic(pooled)
        no_correction_score = scores[:, -1]
        correction_scores = scores[:, :-1]
        best_score, best_idx = correction_scores.max(dim=-1)
        apply_correction = best_score > (no_correction_score + 0.2)
        corrected_state = hidden_state.clone()
        correction_id = -1
        if apply_correction.any():
            for b in range(hidden_state.shape[0]):
                if apply_correction[b]:
                    correction = self.correction_directions[best_idx[b]]
                    confidence = torch.sigmoid(best_score[b] - self.thresholds[best_idx[b]])
                    scale = torch.abs(self.confidence_scale)
                    corrected_state[b] = hidden_state[b] + scale * confidence * correction
                    correction_id = best_idx[b].item()
        return (corrected_state, correction_id) if return_correction_id else corrected_state

class SkillComposer(nn.Module):
    def __init__(self, d_model, num_skills=64, skill_rank=8, top_k=2):
        super().__init__()
        self.num_skills = num_skills
        self.skill_rank = skill_rank
        self.top_k = top_k
        self.d_model = d_model
        self.skill_u = nn.Parameter(torch.randn(num_skills, d_model, skill_rank) * 0.01)
        self.skill_v = nn.Parameter(torch.randn(num_skills, skill_rank, d_model) * 0.01)
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, num_skills)
        )
        self.skill_scale = nn.Parameter(torch.ones(num_skills) * 0.01)

    def forward(self, x):
        B, T, D = x.shape
        skill_logits = self.router(x.mean(dim=1))
        skill_weights = F.softmax(skill_logits, dim=-1)
        if self.training:
            self._load_balance_loss = skill_weights.mean(dim=0).var() * 10.0
        else:
            self._load_balance_loss = torch.tensor(0.0)
        top_k = min(self.top_k, self.num_skills)
        top_weights, top_indices = skill_weights.topk(top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        perturbation = torch.zeros_like(x)
        for k in range(self.top_k):
            u = self.skill_u[top_indices[:, k]]
            v = self.skill_v[top_indices[:, k]]
            scale = torch.abs(self.skill_scale[top_indices[:, k]])
            x_v = torch.bmm(x, v.transpose(1, 2))
            perturbation += top_weights[:, k].unsqueeze(1).unsqueeze(-1) * scale.unsqueeze(1).unsqueeze(-1) * torch.bmm(x_v, u.transpose(1, 2))
        return x + perturbation

print('CRN components loaded')

# ---- Student Model ----
class PrajnaStudentMultiLayer(nn.Module):
    def __init__(self, device='mps'):
        super().__init__()
        self.device = device
        gc.collect()
        print('Loading E2B student...')
        self.tok = AutoTokenizer.from_pretrained('google/gemma-4-E2B')

        if device == 'cuda':
            self.base_model = AutoModelForCausalLM.from_pretrained(
                'google/gemma-4-E2B',
                dtype=torch.float16,
                device_map={'': 0},
            )
        else:
            self.base_model = AutoModelForCausalLM.from_pretrained(
                'google/gemma-4-E2B',
                dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

        model_dev = next(self.base_model.parameters()).device
        print(f'Base model on: {model_dev}')

        for p in self.base_model.parameters():
            p.requires_grad = False
        self.vocab = 262144
        self.d_model = 1536
        self.lm = self.base_model.model.language_model
        self.num_layers = len(self.lm.layers)

        self.inject_every = INJECT_EVERY
        self.inject_indices = list(range(self.inject_every - 1, self.num_layers, self.inject_every))
        self.num_injections = len(self.inject_indices)
        self.crn_mix = nn.Parameter(torch.full((self.num_injections,), CRN_MIX_INIT))

        crn_dev = device
        self.mem = EpisodicMemory(self.d_model, mem_size=256, mem_dim=64, device=crn_dev)
        self.reflection = ReflectiveLoop(d_model=self.d_model, num_corrections=8).to(crn_dev)
        self.skills = SkillComposer(d_model=self.d_model, num_skills=32, skill_rank=4, top_k=2).to(crn_dev)
        self.resonance = ResonanceAttention(d_model=self.d_model, num_heads=4, num_frequencies=8, top_k=2).to(crn_dev)

        crn_params = self.get_params()
        total = sum(p.numel() for p in crn_params)
        print(f'CRN: {total:,} params | Injections: {self.num_injections} at layers {self.inject_indices}')
        print(f'Config: MAX_LENGTH={MAX_LENGTH} | INJECT_EVERY={INJECT_EVERY} | freq=8 top_k=2')

    def _collect_hidden(self, input_ids):
        with torch.no_grad():
            outputs = self.base_model(
                input_ids=input_ids, use_cache=False,
                output_attentions=False, output_hidden_states=True, return_dict=True
            )
            collected = {idx: outputs.hidden_states[layer_idx + 1]
                         for idx, layer_idx in enumerate(self.inject_indices)}
            final_hidden = outputs.hidden_states[-1]
            del outputs
        return {'collected': collected, 'final_hidden': final_hidden}

    def _apply_crn(self, outputs, training=True):
        collected = outputs['collected']
        final_hidden = outputs['final_hidden']
        corrections = torch.zeros_like(final_hidden, dtype=torch.float32)
        for idx in range(self.num_injections):
            h = collected[idx].detach().to(torch.float32)
            r = self.resonance(h)
            s = self.skills(h)
            mix = torch.sigmoid(self.crn_mix[idx])
            correction = mix * (r + s)
            if training:
                corrections = corrections + correction
            else:
                corrections = corrections + correction.detach()
        if self.mem.temporal_positions.sum() > 0:
            read_out, _ = self.mem.read(final_hidden.detach().mean(dim=1).to(torch.float32), top_k=8)
            corrections = corrections + read_out.unsqueeze(1)
        hidden_corrected = final_hidden.detach().to(torch.float32) + corrections
        logits = self.base_model.lm_head(hidden_corrected.to(torch.float16))
        return logits, final_hidden

    def forward(self, input_ids, labels=None):
        outputs = self._collect_hidden(input_ids)
        logits, final_hidden = self._apply_crn(outputs, training=self.training)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab),
                labels[:, 1:].reshape(-1), ignore_index=-100
            )
        if self.training and labels is not None:
            self.mem.write(final_hidden[:, -1, :].mean(dim=0).to(torch.float32), force=False)
        return {'loss': loss, 'logits': logits}

    def forward_dpo(self, input_ids_chosen, input_ids_rejected):
        logps_c = self._dpo_logps(input_ids_chosen)
        logps_r = self._dpo_logps(input_ids_rejected)
        loss = -F.logsigmoid(DPO_BETA * (logps_c - logps_r)).mean()
        return {'loss': loss, 'chosen_reward': logps_c.mean().item(), 'rejected_reward': logps_r.mean().item()}

    def _dpo_logps(self, input_ids):
        outputs = self._collect_hidden(input_ids)
        logits, _ = self._apply_crn(outputs, training=True)
        return self._get_batch_logps(logits, input_ids)

    def _get_batch_logps(self, logits, labels):
        labels = labels[:, 1:].clone()
        logits = logits[:, :-1]
        mask = labels != -100
        labels[~mask] = 0
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, 2, labels.unsqueeze(2)).squeeze(2)
        return (token_log_probs * mask).sum(dim=-1)

    def get_params(self):
        params = (self.mem.get_parameters() + list(self.reflection.parameters()) +
                  list(self.skills.parameters()) + list(self.resonance.parameters()))
        params.append(self.crn_mix)
        return params

    def save_memory(self, p): self.mem.save(p)
    def load_memory(self, p): self.mem.load(p)

print('Student class defined')

# ---- Data ----
class SFTDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=MAX_LENGTH):
        with open(data_file) as f: self.samples = json.load(f)
        self.tok, self.ml = tokenizer, max_length
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        s = self.samples[i]
        text = f"{s.get('prompt', '')}\n\n{s.get('response', '')}"
        enc = self.tok(text, truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        ids = enc['input_ids'].squeeze()
        labels = ids.clone()
        labels[enc['attention_mask'].squeeze() == 0] = -100
        return {'input_ids': ids, 'labels': labels}

class DPODataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=MAX_LENGTH):
        with open(data_file) as f: self.pairs = json.load(f)
        self.tok, self.ml = tokenizer, max_length
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        p = self.pairs[i]
        c = self.tok(p['chosen'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        r = self.tok(p['rejected'], truncation=True, max_length=self.ml, padding='max_length', return_tensors='pt')
        return {'chosen_ids': c['input_ids'].squeeze(), 'rejected_ids': r['input_ids'].squeeze()}

if os.path.exists(teacher_file):
    with open(teacher_file) as f: print(f'Data exists: {len(json.load(f))} samples')
else:
    print('Generating data...')
    import urllib.request
    url = 'https://huggingface.co/datasets/WithinUsAI/claude_mythos_distilled_25k/resolve/main/claude_mythos_distilled_25k.jsonl'
    urllib.request.urlretrieve(url, f'{DATA_DIR}/mythos_25k.jsonl')
    mythos = []
    with open(f'{DATA_DIR}/mythos_25k.jsonl') as f:
        for line in f:
            d = json.loads(line)
            msgs = d.get('messages', [])
            prompt = response = ''
            for m in msgs:
                if m['role'] == 'user': prompt = m['content']
                elif m['role'] == 'assistant': response = m['content']
            if prompt and response and len(response) > 20:
                mythos.append({'prompt': prompt, 'response': response})
    templates = []
    random.seed(42)
    math_qa = [
        ('What is {a} + {b}?', lambda a, b: str(a + b)),
        ('What is {a} * {b}?', lambda a, b: str(a * b)),
        ('What is {a} - {b}?', lambda a, b: str(a - b)),
    ]
    for _ in range(800):
        a, b = random.randint(1, 100), random.randint(1, 100)
        q, fn = random.choice(math_qa)
        templates.append({'prompt': q.format(a=a, b=b), 'response': fn(a, b)})
    facts = [
        ('What is the capital of France?', 'Paris'),
        ('What is the capital of Japan?', 'Tokyo'),
        ('Who invented the telephone?', 'Alexander Graham Bell'),
        ('What year did WWII end?', '1945'),
    ]
    for _ in range(800):
        q, a = random.choice(facts)
        templates.append({'prompt': q, 'response': a})
    code_qa = [('Write a Python function to reverse a string.', 'def reverse_string(s): return s[::-1]')]
    for _ in range(800):
        q, a = random.choice(code_qa)
        templates.append({'prompt': q, 'response': a})
    combined = templates + mythos
    random.shuffle(combined)
    with open(teacher_file, 'w') as f: json.dump(combined, f, indent=2)
    print(f'Saved: {len(combined)} samples')

if not os.path.exists(dpo_file):
    print('Generating DPO pairs...')
    with open(teacher_file) as f: data = json.load(f)
    bloated = ['Great question! ', 'Excellent question! ']
    hallucinations = [' According to recent studies, this is 97% accurate.']
    dpo_pairs = []
    for s in data[:5000]:
        chosen = s['response']
        if not chosen or len(chosen) < 20: continue
        if random.random() < 0.5:
            rejected = random.choice(bloated) + chosen
        else:
            rejected = chosen + random.choice(hallucinations)
        dpo_pairs.append({'prompt': s['prompt'], 'chosen': chosen, 'rejected': rejected})
    random.shuffle(dpo_pairs)
    with open(dpo_file, 'w') as f: json.dump(dpo_pairs[:3000], f, indent=2)
    print(f'Saved: {len(dpo_pairs[:3000])} DPO pairs')
else:
    with open(dpo_file) as f: print(f'DPO pairs: {len(json.load(f))}')

print('Data ready')

CRN_PREFIXES = ('crn_mix', 'resonance.', 'skills.', 'reflection.', 'mem.')

def get_crn_state_dict(model):
    return {k: v.cpu() for k, v in model.state_dict().items()
            if any(k.startswith(p) for p in CRN_PREFIXES)}

# ---- SFT Training ----
if not state.get('sft_complete', False):
    print('=' * 60)
    print('PHASE 1: SFT DISTILLATION')
    print('=' * 60)
    gc.collect()
    student = PrajnaStudentMultiLayer(device=DEVICE)
    sft_start = state.get('sft_step', 0)
    # Prefer checkpoint matching saved step; else numerically latest
    expected = f'{CKPT_DIR}/sft_{sft_start}.pt'
    latest_ckpt = expected if os.path.exists(expected) else find_latest_ckpt('sft')
    if latest_ckpt:
        print(f'Resuming from: {latest_ckpt}')
        ckpt = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt['crn'], strict=False)
        if 'memory_file' in ckpt and os.path.exists(ckpt['memory_file']):
            student.load_memory(ckpt['memory_file'])
        sft_start = ckpt.get('step', 0)
    dataset = SFTDataset(teacher_file, student.tok)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=SFT_LR, weight_decay=0.01)
    remaining = max(SFT_STEPS - sft_start, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining, eta_min=SFT_LR * 0.1)
    for _ in range(sft_start): scheduler.step()
    student.train()
    losses, step = [], sft_start
    t_start = time.time()
    print(f'Starting step {step}/{SFT_STEPS} | Dataset: {len(dataset)}')

    for epoch in range(1000):
        if step >= SFT_STEPS: break
        for batch in loader:
            if step >= SFT_STEPS: break
            try:
                input_ids = batch['input_ids'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)
                out = student(input_ids, labels)
                loss = out['loss'] / GRAD_ACCUM
                if torch.isnan(loss):
                    scheduler.step(); step += 1; continue
                loss.backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                    opt.step(); opt.zero_grad(); scheduler.step()
                losses.append(loss.item() * GRAD_ACCUM)
                step += 1
                state['sft_step'] = step
                if step % 20 == 0: gc.collect()
                if step % LOG_EVERY == 0:
                    avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
                    elapsed = (time.time() - t_start) / 60
                    rate = (step - sft_start) / (time.time() - t_start) if time.time() > t_start else 0
                    eta = (SFT_STEPS - step) / rate / 60 if rate > 0 else 0
                    print(f'  Step {step:5d}/{SFT_STEPS} | Loss: {avg:.4f} | {rate:.2f}/s | ETA: {eta:.0f}min')
                    sys.stdout.flush()
                if step % SAVE_EVERY == 0:
                    mem_file = f'{CKPT_DIR}/memory_sft_{step}.json'
                    student.save_memory(mem_file)
                    torch.save({
                        'step': step,
                        'crn': get_crn_state_dict(student),
                        'loss': sum(losses[-50:]) / max(len(losses[-50:]), 1),
                        'memory_file': mem_file,
                    }, f'{CKPT_DIR}/sft_{step}.pt')
                    save_state(state)
                    print(f'  Saved: sft_{step}.pt')
                    sys.stdout.flush()
            except Exception as e:
                print(f'  ERROR at step {step}: {e}')
                traceback.print_exc()
                sys.stdout.flush()
                gc.collect()
                if DEVICE == 'mps': torch.mps.empty_cache()
                continue

    mem_file = f'{CKPT_DIR}/memory_sft_final.json'
    student.save_memory(mem_file)
    final_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1) if losses else 0
    torch.save({
        'step': step,
        'crn': get_crn_state_dict(student),
        'loss': final_loss, 'memory_file': mem_file,
    }, f'{CKPT_DIR}/sft_final.pt')
    state['sft_complete'] = True; state['sft_step'] = step; save_state(state)
    print(f'SFT done! Steps: {step} | Loss: {final_loss:.4f} | Time: {(time.time() - t_start) / 60:.1f}min')
else:
    print('SFT already complete.')

# ---- DPO Training ----
if not state.get('dpo_complete', False):
    print('=' * 60)
    print('PHASE 2: DPO ALIGNMENT')
    print('=' * 60)
    gc.collect()
    # Ensure student exists (recreate if script restarted after SFT)
    if 'student' not in dir() or 'student' not in globals():
        student = PrajnaStudentMultiLayer(device=DEVICE)
    dpo_start = state.get('dpo_step', 0)
    expected = f'{CKPT_DIR}/dpo_{dpo_start}.pt'
    latest_dpo = expected if os.path.exists(expected) else find_latest_ckpt('dpo')
    if latest_dpo:
        print(f'Resuming from: {latest_dpo}')
        ckpt = torch.load(latest_dpo, map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt['crn'], strict=False)
        if 'memory_file' in ckpt and os.path.exists(ckpt['memory_file']):
            student.load_memory(ckpt['memory_file'])
        dpo_start = ckpt.get('step', 0)
    elif os.path.exists(f'{CKPT_DIR}/sft_final.pt'):
        print('Loading SFT-trained CRN from sft_final.pt')
        ckpt = torch.load(f'{CKPT_DIR}/sft_final.pt', map_location=DEVICE, weights_only=False)
        student.load_state_dict(ckpt['crn'], strict=False)
        if 'memory_file' in ckpt and os.path.exists(ckpt['memory_file']):
            student.load_memory(ckpt['memory_file'])
    dset = DPODataset(dpo_file, student.tok)
    dloader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    params = student.get_params()
    opt = torch.optim.AdamW(params, lr=DPO_LR, weight_decay=0.01)
    remaining = max(DPO_STEPS - dpo_start, 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=remaining, eta_min=DPO_LR * 0.1)
    for _ in range(dpo_start): scheduler.step()
    student.train()
    losses, step = [], dpo_start
    t_start = time.time()
    print(f'Starting DPO step {step}/{DPO_STEPS} | Dataset: {len(dset)}')

    for epoch in range(1000):
        if step >= DPO_STEPS: break
        for batch in dloader:
            if step >= DPO_STEPS: break
            try:
                chosen_ids = batch['chosen_ids'].to(DEVICE)
                rejected_ids = batch['rejected_ids'].to(DEVICE)
                out = student.forward_dpo(chosen_ids, rejected_ids)
                loss = out['loss'] / GRAD_ACCUM
                if torch.isnan(loss):
                    scheduler.step(); step += 1; continue
                loss.backward()
                if (step + 1) % GRAD_ACCUM == 0:
                    torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
                    opt.step(); opt.zero_grad(); scheduler.step()
                losses.append(loss.item() * GRAD_ACCUM)
                step += 1
                state['dpo_step'] = step
                if step % 20 == 0: gc.collect()
                if step % LOG_EVERY == 0:
                    avg = sum(losses[-LOG_EVERY:]) / LOG_EVERY
                    print(f'  Step {step:5d}/{DPO_STEPS} | Loss: {avg:.4f} | C: {out["chosen_reward"]:.2f} | R: {out["rejected_reward"]:.2f}')
                    sys.stdout.flush()
                if step % SAVE_EVERY == 0:
                    mem_file = f'{CKPT_DIR}/memory_dpo_{step}.json'
                    student.save_memory(mem_file)
                    torch.save({
                        'step': step,
                        'crn': get_crn_state_dict(student),
                        'loss': sum(losses[-50:]) / max(len(losses[-50:]), 1),
                        'memory_file': mem_file,
                    }, f'{CKPT_DIR}/dpo_{step}.pt')
                    save_state(state)
                    print(f'  Saved: dpo_{step}.pt')
                    sys.stdout.flush()
            except Exception as e:
                print(f'  ERROR at step {step}: {e}')
                traceback.print_exc()
                sys.stdout.flush()
                gc.collect()
                if DEVICE == 'mps': torch.mps.empty_cache()
                continue

    mem_file = f'{CKPT_DIR}/memory_dpo_final.json'
    student.save_memory(mem_file)
    final_loss = sum(losses[-50:]) / max(len(losses[-50:]), 1) if losses else 0
    torch.save({
        'step': step,
        'crn': get_crn_state_dict(student),
        'loss': final_loss, 'memory_file': mem_file,
    }, f'{CKPT_DIR}/dpo_final.pt')
    state['dpo_complete'] = True; state['dpo_step'] = step; state['phase'] = 'complete'
    save_state(state)
    print(f'DPO done! Steps: {step} | Loss: {final_loss:.4f} | Time: {(time.time() - t_start) / 60:.1f}min')
else:
    print('DPO already complete.')

del student, opt
gc.collect()
print(f'\nAll done! Checkpoints in {CKPT_DIR}')

#!/usr/bin/env python3
"""
Prajna M4 Validation: Train CRN components on Gemma 4 E2B
- Base model: Frozen (inference only)
- CRN components: Trainable (float32)
- Device: Mac Mini M4 16GB
- Expected: Loss decreases, showing CRN components learn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

D_MODEL = 1536
VOCAB_SIZE = 262144


class CRNAdapter(nn.Module):
    def __init__(self, d_model=D_MODEL, n_bands=16, mem_capacity=64, n_skills=4):
        super().__init__()
        self.d_model = d_model

        # Resonance Attention
        self.band_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.threshold = nn.Parameter(torch.zeros(1))

        # Episodic Memory
        self.mem_capacity = mem_capacity
        self.key_proj = nn.Linear(d_model, d_model)
        self.write_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.read_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

        # Reflective Loop
        self.error_detector = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))
        self.corrector = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # Skill Composer
        self.n_skills = n_skills
        self.skill_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_skills)])
        self.router = nn.Linear(d_model, n_skills)

        # Output
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mem=None):
        B, T, D = x.shape

        # Resonance
        band_emb = self.band_proj(x)
        gate = self.gate(x)
        mask = (gate > torch.sigmoid(self.threshold)).float()
        x = x + 0.1 * band_emb * mask

        # Episodic Memory
        if mem is None:
            mem = torch.zeros(B, self.mem_capacity, D, device=x.device, dtype=x.dtype)
        q = self.key_proj(x)
        k = self.key_proj(mem)
        attn = F.softmax(torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5), dim=-1)
        read = torch.bmm(attn, mem)
        # Write: update memory with current input mean
        x_mean = x.mean(dim=1, keepdim=True)  # [B, 1, D]
        mem_mean = mem.mean(dim=1, keepdim=True)  # [B, 1, D]
        wg = self.write_gate(torch.cat([x_mean, mem_mean], dim=-1))  # [B, 1, D]
        # Shift memory: drop oldest, add new
        mem = torch.cat([mem[:, 1:, :], x_mean * wg], dim=1)  # [B, cap, D]
        rg = self.read_gate(read)
        x = x + rg * read

        # Reflection
        err = torch.sigmoid(self.error_detector(x.mean(dim=1)))
        corr = self.corrector(x)
        x = x + 0.05 * corr * err

        # Skills
        rl = self.router(x.mean(dim=1))
        rw = F.softmax(rl, dim=-1)
        out = sum(self.skill_projs[i](x) * rw[:, i].unsqueeze(-1).unsqueeze(-1) for i in range(self.n_skills))

        # Output
        x = self.out_proj(out)
        return x, mem


def create_training_data(tokenizer, n=50):
    prompts = [
        "The quick brown fox", "Hello how are you", "I need to remember",
        "Let me think about", "The answer is", "In the future",
        "Today I learned", "The key insight", "What if we", "The best approach",
    ]
    return [tokenizer(p, return_tensors="pt")["input_ids"] for p in (prompts * ((n // len(prompts)) + 1))[:n]]


def validate():
    print("=" * 60)
    print("PRAJNA M4 VALIDATION")
    print("Training CRN on frozen Gemma 4 E2B")
    print("=" * 60)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n[1/4] Loading model...")
    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B", dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    for p in model.parameters():
        p.requires_grad = False

    print("[2/4] Creating CRN adapter...")
    adapter = CRNAdapter()
    vocab_head = nn.Linear(D_MODEL, VOCAB_SIZE)  # float32 by default
    params = list(adapter.parameters()) + list(vocab_head.parameters())
    print(f"  Trainable: {sum(p.numel() for p in params):,} params")

    opt = torch.optim.AdamW(params, lr=1e-3)
    samples = create_training_data(tok, 50)

    print("[3/4] Training 10 steps...")
    print("-" * 60)

    losses = []
    mem = None

    for step in range(10):
        t0 = time.time()
        ids = samples[step]
        labels = ids.clone()

        with torch.no_grad():
            h = model(input_ids=ids, output_hidden_states=True).hidden_states[-1]
        h = h.to(torch.float32)

        out, mem = adapter(h, mem.detach() if mem is not None else None)
        logits = vocab_head(out)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        losses.append(loss.item())
        dt = time.time() - t0
        print(f"  Step {step+1:2d}/10 | Loss: {loss.item():.4f} | {dt:.1f}s")

    print("-" * 60)
    print(f"\nInitial loss: {losses[0]:.4f}")
    print(f"Final loss:   {losses[-1]:.4f}")
    pct = (1 - losses[-1] / losses[0]) * 100
    print(f"Reduction:    {pct:.1f}%")

    ok = losses[-1] < losses[0]
    print(f"\n{'PASS' if ok else 'FAIL'} - CRN {'learned' if ok else 'did not learn'} on M4")
    return ok


if __name__ == "__main__":
    validate()

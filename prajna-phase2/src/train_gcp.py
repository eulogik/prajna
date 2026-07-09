#!/usr/bin/env python3
"""
Prajna Training on GCP T4
- Optimized for 16GB VRAM
- Uses QLoRA for memory efficiency
- Checkpoints to Google Cloud Storage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
from pathlib import Path

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
        x_mean = x.mean(dim=1, keepdim=True)
        mem_mean = mem.mean(dim=1, keepdim=True)
        wg = self.write_gate(torch.cat([x_mean, mem_mean], dim=-1))
        mem = torch.cat([mem[:, 1:, :], x_mean * wg], dim=1)
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


class SyntheticDataset(Dataset):
    """Load pre-generated synthetic training data"""
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Load all JSON files
        self.samples = []
        for f in sorted(self.data_dir.glob("*.json")):
            with open(f) as fh:
                data = json.load(fh)
                self.samples.extend(data)
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Tokenize prompt + response
        prompt = sample.get("prompt", "")
        response = sample.get("response", "")
        text = f"{prompt}\n\n{response}"
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding in loss
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def train(config):
    print("=" * 60)
    print("PRAJNA TRAINING ON GCP T4")
    print("=" * 60)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("ERROR: No GPU available")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # Load teacher model (Gemma 4 E4B) in 4-bit
    print("\n[1/4] Loading teacher model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    teacher = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E4B",
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E4B")
    print(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")
    
    # Create student adapter
    print("\n[2/4] Creating CRN adapter...")
    adapter = CRNAdapter().to("cuda")
    vocab_head = nn.Linear(D_MODEL, VOCAB_SIZE).to("cuda")
    params = list(adapter.parameters()) + list(vocab_head.parameters())
    print(f"Student params: {sum(p.numel() for p in params):,}")
    
    # Freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Optimizer
    optimizer = torch.optim.AdamW(params, lr=config["lr"])
    
    # Dataset
    print("\n[3/4] Loading dataset...")
    dataset = SyntheticDataset(config["data_dir"], tokenizer, max_length=config["max_length"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # Training
    print("\n[4/4] Training...")
    print("-" * 60)
    
    best_loss = float("inf")
    global_step = 0
    
    for epoch in range(config["num_epochs"]):
        epoch_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            t0 = time.time()
            
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")
            
            # Get teacher hidden states
            with torch.no_grad():
                h = teacher(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
            
            # Forward through adapter
            out, _ = adapter(h)
            logits = vocab_head(out)
            
            # Compute loss
            loss = F.cross_entropy(
                logits.view(-1, VOCAB_SIZE),
                labels.view(-1),
                ignore_index=-100
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            global_step += 1
            
            dt = time.time() - t0
            
            if global_step % 10 == 0:
                print(f"  Step {global_step:5d} | Loss: {loss.item():.4f} | {dt:.2f}s")
            
            # Checkpoint
            if global_step % config["checkpoint_every"] == 0:
                checkpoint = {
                    "step": global_step,
                    "adapter": adapter.state_dict(),
                    "vocab_head": vocab_head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "loss": loss.item(),
                }
                path = f"{config['output_dir']}/checkpoint_{global_step}.pt"
                torch.save(checkpoint, path)
                print(f"  Checkpoint saved: {path}")
        
        # Epoch summary
        avg_loss = epoch_loss / num_batches
        print(f"\nEpoch {epoch+1}/{config['num_epochs']} | Avg Loss: {avg_loss:.4f}")
        
        # Save best
        if avg_loss < best_loss:
            best_loss = avg_loss
            path = f"{config['output_dir']}/best_model.pt"
            torch.save({
                "adapter": adapter.state_dict(),
                "vocab_head": vocab_head.state_dict(),
                "loss": best_loss,
            }, path)
            print(f"  Best model saved: {path}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Total steps: {global_step}")
    print("=" * 60)


if __name__ == "__main__":
    config = {
        "data_dir": "~/prajna-training/data",
        "output_dir": "~/prajna-training/checkpoints",
        "batch_size": 1,
        "lr": 2e-4,
        "num_epochs": 10,
        "max_length": 512,
        "checkpoint_every": 500,
    }
    train(config)

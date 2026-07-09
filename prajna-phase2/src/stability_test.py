#!/usr/bin/env python3
"""
Prajna Stability Test
Runs training for 100+ steps on real Gemma 4 E2B to confirm reliability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import sys
import signal
import traceback
import psutil
from pathlib import Path
from datetime import datetime

# Add parent directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bulletproof import CRNAdapter, CheckpointManager, SystemMonitor

D_MODEL = 1536
VOCAB_SIZE = 262144

class MiniDataset(Dataset):
    def __init__(self, tokenizer, n_samples=200, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        prompts = [
            "The quick brown fox", "Hello how are you", "I need to remember",
            "Let me think about", "The answer is", "In the future",
            "Today I learned", "The key insight", "What if we", "The best approach",
            "Explain quantum computing", "Write a Python function", "Solve this problem",
            "Describe the process", "What are the benefits", "How does it work",
            "Tell me about science", "Explain the concept", "What is the meaning",
            "How to improve", "Best practices for", "Step by step guide",
        ]
        
        self.samples = []
        for i in range(n_samples):
            prompt = prompts[i % len(prompts)]
            encoding = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length, padding="max_length")
            self.samples.append({
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        labels = s["input_ids"].clone()
        labels[s["attention_mask"] == 0] = -100
        return {"input_ids": s["input_ids"], "labels": labels}


def run_stability_test():
    print("=" * 60)
    print("PRAJNA STABILITY TEST")
    print("Running 100+ steps on real Gemma 4 E2B")
    print("=" * 60)
    
    # Setup
    output_dir = os.path.expanduser("~/prajna-training")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    monitor = SystemMonitor(log_dir)
    checkpoint_mgr = CheckpointManager(checkpoint_dir, keep_last_n=3)
    
    # Memory check
    mem = psutil.virtual_memory()
    monitor.log(f"System memory: {mem.total / 1e9:.1f} GB total, {mem.percent:.0f}% used")
    
    # Load model
    monitor.log("\n[1/3] Loading Gemma 4 E2B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B",
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    monitor.log(f"Model loaded in {time.time()-t0:.1f}s")
    
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Create adapter
    monitor.log("\n[2/3] Creating CRN adapter...")
    adapter = CRNAdapter(d_model=D_MODEL)
    vocab_head = nn.Linear(D_MODEL, VOCAB_SIZE)
    
    # Try to load existing checkpoint
    ckpt = checkpoint_mgr.load_latest_checkpoint()
    if ckpt:
        adapter.load_state_dict(ckpt["model"])
        vocab_head.load_state_dict(ckpt["vocab_head"])
        monitor.log(f"Resumed from step {ckpt.get('step', 0)}")
    
    total_params = sum(p.numel() for p in adapter.parameters()) + sum(p.numel() for p in vocab_head.parameters())
    monitor.log(f"Trainable params: {total_params:,}")
    
    optimizer = torch.optim.AdamW(
        list(adapter.parameters()) + list(vocab_head.parameters()),
        lr=2e-4, weight_decay=0.01
    )
    
    # Dataset
    monitor.log("\n[3/3] Loading dataset...")
    dataset = MiniDataset(tokenizer, n_samples=200, max_length=128)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    monitor.log(f"Dataset: {len(dataset)} samples")
    
    # Training
    monitor.log("\n" + "=" * 60)
    monitor.log("STABILITY TRAINING (100 steps)")
    monitor.log("=" * 60)
    
    start_step = checkpoint_mgr.get_latest_step()
    losses = []
    step_times = []
    nan_count = 0
    max_steps = 100
    
    for step in range(max_steps):
        t0 = time.time()
        
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        # Forward through teacher
        with torch.no_grad():
            h = teacher(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
            h = h.to(torch.float32)
        
        # Forward through adapter
        out, _ = adapter(h)
        logits = vocab_head(out)
        
        # Loss
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1), ignore_index=-100)
        
        # NaN check
        if torch.isnan(loss):
            nan_count += 1
            monitor.log(f"NaN at step {step} (count: {nan_count})", "WARNING")
            if nan_count > 5:
                monitor.log("Too many NaNs, stopping", "ERROR")
                break
            continue
        
        nan_count = 0
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(adapter.parameters()) + list(vocab_head.parameters()), 1.0
        )
        optimizer.step()
        
        dt = time.time() - t0
        losses.append(loss.item())
        step_times.append(dt)
        
        # Log every 10 steps
        if (step + 1) % 10 == 0:
            avg_loss = sum(losses[-10:]) / len(losses[-10:])
            avg_time = sum(step_times[-10:]) / len(step_times[-10:])
            mem_pct = psutil.virtual_memory().percent
            monitor.log(
                f"Step {step+1:3d}/{max_steps} | Loss: {loss.item():.4f} | "
                f"Avg: {avg_loss:.4f} | {avg_time:.1f}s | Mem: {mem_pct:.0f}%"
            )
        
        # Checkpoint every 50 steps
        if (step + 1) % 50 == 0:
            state = {
                "model": adapter.state_dict(),
                "vocab_head": vocab_head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            avg_loss = sum(losses[-50:]) / len(losses[-50:])
            checkpoint_mgr.save_checkpoint(state, start_step + step + 1, avg_loss)
            monitor.log(f"Checkpoint saved at step {step+1}")
    
    # Final summary
    monitor.log("\n" + "=" * 60)
    monitor.log("STABILITY TEST COMPLETE")
    monitor.log("=" * 60)
    
    if losses:
        initial = sum(losses[:10]) / 10
        final = sum(losses[-10:]) / 10
        avg_time = sum(step_times) / len(step_times)
        
        monitor.log(f"Steps completed: {len(losses)}/{max_steps}")
        monitor.log(f"Initial loss: {initial:.4f}")
        monitor.log(f"Final loss: {final:.4f}")
        monitor.log(f"Reduction: {(1-final/initial)*100:.1f}%")
        monitor.log(f"Avg step time: {avg_time:.1f}s")
        monitor.log(f"Total time: {sum(step_times):.0f}s ({sum(step_times)/60:.1f} min)")
        monitor.log(f"NaN count: {nan_count}")
        
        # Save final model
        state = {
            "model": adapter.state_dict(),
            "vocab_head": vocab_head.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        checkpoint_mgr.save_checkpoint(state, len(losses), final, is_best=True)
        monitor.log(f"Final model saved: {checkpoint_mgr.best_model_path}")
    
    success = len(losses) >= 80  # At least 80% of steps completed
    monitor.log(f"\n{'✓ STABLE' if success else '✗ UNSTABLE'} - System {'ready' if success else 'needs fixes'} for full training")
    
    return success


if __name__ == "__main__":
    success = run_stability_test()
    sys.exit(0 if success else 1)

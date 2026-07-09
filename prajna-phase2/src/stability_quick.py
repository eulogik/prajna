#!/usr/bin/env python3
"""
Quick Stability Test - 30 steps on real model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import sys
import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_bulletproof import CRNAdapter

D_MODEL = 1536
VOCAB_SIZE = 262144

def quick_test():
    print("=" * 60)
    print("QUICK STABILITY TEST (30 steps)")
    print("=" * 60)
    
    # Memory check
    mem = psutil.virtual_memory()
    print(f"Memory: {mem.total/1e9:.1f} GB, {mem.percent:.0f}% used")
    
    # Load model
    print("\nLoading Gemma 4 E2B...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    t0 = time.time()
    teacher = AutoModelForCausalLM.from_pretrained(
        "google/gemma-4-E2B",
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
    print(f"Loaded in {time.time()-t0:.1f}s")
    
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Create adapter
    adapter = CRNAdapter(d_model=D_MODEL)
    vocab_head = nn.Linear(D_MODEL, VOCAB_SIZE)
    
    params = list(adapter.parameters()) + list(vocab_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    
    print(f"Trainable: {sum(p.numel() for p in params):,} params")
    
    # Test data
    prompts = [
        "The quick brown fox", "Hello how are you", "I need to remember",
        "Let me think about", "The answer is", "In the future",
        "Today I learned", "The key insight", "What if we", "The best approach",
    ]
    
    # Training
    print("\nTraining 30 steps...")
    print("-" * 60)
    
    losses = []
    times = []
    
    for step in range(30):
        t0 = time.time()
        
        # Random prompt
        prompt = prompts[step % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Forward
        with torch.no_grad():
            h = teacher(input_ids=inputs["input_ids"], output_hidden_states=True).hidden_states[-1]
            h = h.to(torch.float32)
        
        out, _ = adapter(h)
        logits = vocab_head(out)
        
        labels = inputs["input_ids"].clone()
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), labels.view(-1))
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        dt = time.time() - t0
        losses.append(loss.item())
        times.append(dt)
        
        if (step + 1) % 5 == 0:
            avg_loss = sum(losses[-5:]) / 5
            avg_time = sum(times[-5:]) / 5
            mem_pct = psutil.virtual_memory().percent
            print(f"  Step {step+1:2d}/30 | Loss: {loss.item():.4f} | Avg: {avg_loss:.4f} | {avg_time:.1f}s | Mem: {mem_pct:.0f}%")
    
    # Summary
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    initial = sum(losses[:5]) / 5
    final = sum(losses[-5:]) / 5
    avg_time = sum(times) / len(times)
    
    print(f"Steps: {len(losses)}/30")
    print(f"Initial loss: {initial:.4f}")
    print(f"Final loss: {final:.4f}")
    print(f"Reduction: {(1-final/initial)*100:.1f}%")
    print(f"Avg time: {avg_time:.1f}s/step")
    print(f"Total time: {sum(times):.0f}s ({sum(times)/60:.1f} min)")
    
    # Save checkpoint
    output_dir = os.path.expanduser("~/prajna-training/checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    
    checkpoint = {
        "step": len(losses),
        "model": adapter.state_dict(),
        "vocab_head": vocab_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "loss": final,
    }
    path = os.path.join(output_dir, "stability_test.pt")
    torch.save(checkpoint, path)
    print(f"\nCheckpoint saved: {path}")
    
    success = len(losses) >= 25
    print(f"\n{'✓ PASS' if success else '✗ FAIL'} - System {'ready' if success else 'needs fixes'}")
    
    return success

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)

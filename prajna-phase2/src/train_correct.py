#!/usr/bin/env python3
"""
Prajna Correct Architecture Training
- Uses hook-based CRN integration (prajna_gemma4_full.py)
- Distills from Gemma 4 E4B teacher
- Auto-resume, checkpoints, memory monitoring
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

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'prajna-toy-validation', 'src'))
sys.path.insert(0, os.path.dirname(__file__))

from prajna_gemma4_full import PrajnaGemma4Full

# ── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "output_dir": os.path.expanduser("~/prajna-training"),
    "data_dir": os.path.expanduser("~/prajna-training/data"),
    "checkpoint_dir": os.path.expanduser("~/prajna-training/checkpoints"),
    "log_dir": os.path.expanduser("~/prajna-training/logs"),
    
    # Models
    "student_model": "google/gemma-4-E2B",
    "teacher_model": "google/gemma-4-E4B",
    
    # Training
    "batch_size": 1,
    "gradient_accumulation": 8,
    "lr": 2e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_steps": 50,
    "num_epochs": 5,
    "max_length": 256,
    
    # Checkpointing
    "checkpoint_every": 100,
    "keep_last_n": 5,
    
    # Memory
    "gc_every": 50,
    "max_memory_pct": 90,
}


# ── Dataset ──────────────────────────────────────────────────────────────────

class DistillationDataset(Dataset):
    """Dataset with teacher outputs for distillation"""
    def __init__(self, data_dir, tokenizer, max_length=256):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        # Load all JSON files
        if self.data_dir.exists():
            for f in sorted(self.data_dir.glob("*.json")):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                        if isinstance(data, list):
                            self.samples.extend(data)
                except Exception as e:
                    print(f"WARNING: Failed to load {f}: {e}")
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return max(len(self.samples), 1)
    
    def __getitem__(self, idx):
        if not self.samples:
            # Dummy data
            dummy = torch.zeros(self.max_length, dtype=torch.long)
            return {"input_ids": dummy, "labels": dummy.clone()}
        
        sample = self.samples[idx % len(self.samples)]
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
        labels = input_ids.clone()
        labels[encoding["attention_mask"].squeeze() == 0] = -100
        
        return {"input_ids": input_ids, "labels": labels}


# ── Training ─────────────────────────────────────────────────────────────────

class CorrectArchitectureTrainer:
    def __init__(self, config):
        self.config = config
        self.running = True
        self.start_time = time.time()
        
        # Create directories
        for d in [config["output_dir"], config["data_dir"], 
                  config["checkpoint_dir"], config["log_dir"]]:
            os.makedirs(d, exist_ok=True)
        
        # Setup logging
        self.log_file = os.path.join(config["log_dir"], "training.log")
        
        # Signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        self.log(f"Signal {signum} received, saving checkpoint...", "WARNING")
        self.running = False
    
    def log(self, msg, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
    
    def save_checkpoint(self, student, optimizer, step, loss):
        """Save checkpoint"""
        checkpoint = {
            "step": step,
            "loss": loss,
            "crn_state": {k: v for k, v in student.state_dict().items() 
                         if not k.startswith("base_model")},
            "optimizer": optimizer.state_dict(),
        }
        
        # Atomic save
        temp_path = os.path.join(self.config["checkpoint_dir"], f"ckpt_{step}_temp.pt")
        final_path = os.path.join(self.config["checkpoint_dir"], f"ckpt_{step}.pt")
        torch.save(checkpoint, temp_path)
        os.rename(temp_path, final_path)
        
        # Save best
        best_path = os.path.join(self.config["checkpoint_dir"], "best.pt")
        best_temp = os.path.join(self.config["checkpoint_dir"], "best_temp.pt")
        torch.save(checkpoint, best_temp)
        os.rename(best_temp, best_path)
        
        # Cleanup old
        checkpoints = sorted(Path(self.config["checkpoint_dir"]).glob("ckpt_*.pt"))
        if len(checkpoints) > self.config["keep_last_n"]:
            for old in checkpoints[:len(checkpoints) - self.config["keep_last_n]]:
                old.unlink()
        
        return final_path
    
    def load_checkpoint(self, student, optimizer):
        """Load latest checkpoint"""
        best_path = os.path.join(self.config["checkpoint_dir"], "best.pt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, weights_only=False)
            # Load CRN state (skip base_model)
            crn_state = ckpt["crn_state"]
            student.load_state_dict(crn_state, strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            self.log(f"Resumed from step {ckpt.get('step', 0)}")
            return ckpt.get("step", 0)
        return 0
    
    def generate_teacher_data(self, teacher, tokenizer, n_samples=1000):
        """Generate training data using teacher model"""
        self.log("Generating teacher data...")
        
        prompts = [
            "Explain quantum computing in simple terms",
            "Write a Python function to sort a list",
            "What are the benefits of exercise?",
            "Solve this math problem step by step",
            "Describe the process of photosynthesis",
            "How does machine learning work?",
            "What is the meaning of life?",
            "Explain the theory of relativity",
            "Write a haiku about technology",
            "What are best practices for coding?",
        ]
        
        samples = []
        for i in range(n_samples):
            prompt = prompts[i % len(prompts)]
            
            # Generate with teacher
            inputs = tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                outputs = teacher.generate(
                    inputs["input_ids"],
                    max_new_tokens=100,
                    temperature=0.8,
                    do_sample=True,
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            samples.append({"prompt": prompt, "response": response})
            
            if (i + 1) % 100 == 0:
                self.log(f"  Generated {i+1}/{n_samples} samples")
        
        # Save
        output_path = os.path.join(self.config["data_dir"], "teacher_data.json")
        with open(output_path, "w") as f:
            json.dump(samples, f, indent=2)
        
        self.log(f"Saved {len(samples)} samples to {output_path}")
        return samples
    
    def train(self):
        """Main training loop"""
        self.log("=" * 60)
        self.log("PRAJNA CORRECT ARCHITECTURE TRAINING")
        self.log("Hook-based CRN + E4B Teacher Distillation")
        self.log("=" * 60)
        
        # Memory check
        mem = psutil.virtual_memory()
        self.log(f"System memory: {mem.total/1e9:.1f} GB, {mem.percent:.0f}% used")
        
        # Load teacher (4-bit for memory efficiency)
        self.log("\n[1/4] Loading E4B teacher (4-bit)...")
        try:
            from transformers import BitsAndBytesConfig
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            teacher = AutoModelForCausalLM.from_pretrained(
                self.config["teacher_model"],
                quantization_config=bnb_config,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            teacher_tokenizer = AutoTokenizer.from_pretrained(self.config["teacher_model"])
            
            self.log(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")
        except Exception as e:
            self.log(f"Failed to load teacher in 4-bit: {e}", "WARNING")
            self.log("Falling back to bf16...", "WARNING")
            
            teacher = AutoModelForCausalLM.from_pretrained(
                self.config["teacher_model"],
                dtype=torch.bfloat16,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            teacher_tokenizer = AutoTokenizer.from_pretrained(self.config["teacher_model"])
        
        for p in teacher.parameters():
            p.requires_grad = False
        
        # Generate teacher data
        self.log("\n[2/4] Generating teacher data...")
        data_path = os.path.join(self.config["data_dir"], "teacher_data.json")
        if not os.path.exists(data_path):
            self.generate_teacher_data(teacher, teacher_tokenizer, n_samples=500)
        else:
            self.log("Teacher data already exists, skipping generation")
        
        # Create student (hook-based CRN)
        self.log("\n[3/4] Creating student with hook-based CRN...")
        student = PrajnaGemma4Full(
            model_id=self.config["student_model"],
            device="cpu"
        )
        
        # Get CRN parameters
        crn_params = student.get_crn_parameters()
        self.log(f"CRN params: {sum(p.numel() for p in crn_params):,}")
        
        # Optimizer
        optimizer = torch.optim.AdamW(crn_params, lr=self.config["lr"])
        
        # Load checkpoint
        start_step = self.load_checkpoint(student, optimizer)
        
        # Dataset
        self.log("\n[4/4] Loading dataset...")
        dataset = DistillationDataset(
            self.config["data_dir"],
            student.tokenizer,
            self.config["max_length"]
        )
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        
        # Training
        self.log("\n" + "=" * 60)
        self.log("TRAINING STARTED")
        self.log(f"Architecture: Hook-based CRN (46 hooks)")
        self.log(f"Teacher: {self.config['teacher_model']}")
        self.log(f"Student: {self.config['student_model']}")
        self.log(f"CRN components: Memory, Reflection, Skills")
        self.log("=" * 60)
        
        losses = []
        global_step = start_step
        
        for epoch in range(self.config["num_epochs"]):
            if not self.running:
                break
            
            self.log(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
            
            for batch_idx, batch in enumerate(dataloader):
                if not self.running:
                    break
                
                global_step += 1
                t0 = time.time()
                
                try:
                    input_ids = batch["input_ids"]
                    labels = batch["labels"]
                    
                    # Forward through student with CRN hooks
                    student.train()
                    outputs = student(
                        input_ids=input_ids,
                        labels=labels,
                    )
                    
                    loss = outputs["loss"]
                    
                    # NaN check
                    if torch.isnan(loss):
                        self.log(f"NaN at step {global_step}", "WARNING")
                        optimizer.zero_grad()
                        continue
                    
                    # Backward
                    loss = loss / self.config["gradient_accumulation"]
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config["gradient_accumulation"] == 0:
                        torch.nn.utils.clip_grad_norm_(crn_params, self.config["max_grad_norm"])
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    dt = time.time() - t0
                    losses.append(loss.item() * self.config["gradient_accumulation"])
                    
                    # Log
                    if global_step % 10 == 0:
                        avg_loss = sum(losses[-10:]) / len(losses[-10:])
                        mem_pct = psutil.virtual_memory().percent
                        self.log(
                            f"Step {global_step:5d} | Loss: {loss.item()*self.config['gradient_accumulation']:.4f} | "
                            f"Avg: {avg_loss:.4f} | {dt:.1f}s | Mem: {mem_pct:.0f}%"
                        )
                    
                    # Checkpoint
                    if global_step % self.config["checkpoint_every"] == 0:
                        avg_loss = sum(losses[-50:]) / len(losses[-50:]) if losses else 0
                        path = self.save_checkpoint(student, optimizer, global_step, avg_loss)
                        self.log(f"Checkpoint saved: {path}")
                    
                    # GC
                    if global_step % self.config["gc_every"] == 0:
                        import gc
                        gc.collect()
                
                except Exception as e:
                    self.log(f"Error at step {global_step}: {e}", "ERROR")
                    self.log(traceback.format_exc(), "ERROR")
                    
                    # Emergency save
                    try:
                        self.save_checkpoint(student, optimizer, global_step, float("inf"))
                        self.log("Emergency checkpoint saved", "WARNING")
                    except:
                        self.log("Failed to save emergency checkpoint", "ERROR")
                    
                    continue
        
        # Final save
        if losses:
            final_loss = sum(losses[-50:]) / len(losses[-50:])
            self.save_checkpoint(student, optimizer, global_step, final_loss)
        
        # Summary
        self.log("\n" + "=" * 60)
        self.log("TRAINING COMPLETE")
        self.log(f"Steps: {global_step}")
        if losses:
            self.log(f"Final loss: {sum(losses[-10:])/len(losses[-10:]):.4f}")
        self.log(f"Time: {(time.time()-self.start_time)/60:.1f} min")
        self.log("=" * 60)
        
        # Cleanup
        student.cleanup()
        
        return True


def main():
    trainer = CorrectArchitectureTrainer(CONFIG)
    success = trainer.train()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

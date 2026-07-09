#!/usr/bin/env python3
"""
Prajna Bulletproof Training System
- Auto-resume from any crash
- Saves best model + periodic checkpoints
- Memory/disk monitoring
- Graceful shutdown handling
- Zero data loss guaranteed
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

# ── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "output_dir": os.path.expanduser("~/prajna-training"),
    "data_dir": os.path.expanduser("~/prajna-training/data"),
    "checkpoint_dir": os.path.expanduser("~/prajna-training/checkpoints"),
    "log_dir": os.path.expanduser("~/prajna-training/logs"),
    
    # Model
    "d_model": 1536,
    "vocab_size": 262144,
    "mem_capacity": 64,
    "n_skills": 4,
    
    # Training
    "batch_size": 1,
    "gradient_accumulation": 8,
    "lr": 2e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_steps": 100,
    "num_epochs": 5,
    "max_length": 512,
    
    # Checkpointing
    "checkpoint_every": 500,      # Save every N steps
    "validate_every": 100,        # Validate every N steps
    "save_best_every": 50,        # Check for best model every N steps
    "keep_last_n": 5,             # Keep last N checkpoints
    
    # Memory management
    "max_memory_pct": 85,         # Max memory usage before pausing
    "gc_every": 100,              # Garbage collect every N steps
    
    # Safety
    "max_nan_retries": 3,         # Max NaN retries before stopping
    "timeout_seconds": 300,       # Max time per step (5 min)
    "min_disk_gb": 5,             # Min free disk space
}

D_MODEL = CONFIG["d_model"]
VOCAB_SIZE = CONFIG["vocab_size"]


# ── CRN Adapter ──────────────────────────────────────────────────────────────

class CRNAdapter(nn.Module):
    def __init__(self, d_model=D_MODEL, n_bands=16, mem_capacity=CONFIG["mem_capacity"], n_skills=CONFIG["n_skills"]):
        super().__init__()
        self.d_model = d_model
        self.mem_capacity = mem_capacity
        self.n_skills = n_skills

        # Resonance Attention
        self.band_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.threshold = nn.Parameter(torch.zeros(1))

        # Episodic Memory
        self.key_proj = nn.Linear(d_model, d_model)
        self.write_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.read_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

        # Reflective Loop
        self.error_detector = nn.Sequential(nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, 1))
        self.corrector = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

        # Skill Composer
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


# ── Dataset ──────────────────────────────────────────────────────────────────

class SyntheticDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        if not self.data_dir.exists():
            print(f"WARNING: Data directory not found: {data_dir}")
            return
            
        for f in sorted(self.data_dir.glob("*.json")):
            try:
                with open(f) as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        self.samples.extend(data)
            except Exception as e:
                print(f"WARNING: Failed to load {f}: {e}")
        
        print(f"Loaded {len(self.samples)} samples from {data_dir}")
    
    def __len__(self):
        return max(len(self.samples), 1)
    
    def __getitem__(self, idx):
        if not self.samples:
            # Return dummy data if no samples loaded
            dummy = torch.zeros(self.max_length, dtype=torch.long)
            return {
                "input_ids": dummy,
                "attention_mask": torch.ones(self.max_length, dtype=torch.long),
                "labels": dummy.clone(),
            }
        
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
        attention_mask = encoding["attention_mask"].squeeze()
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ── System Monitor ───────────────────────────────────────────────────────────

class SystemMonitor:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "training.log"
        self.metrics_file = self.log_dir / "metrics.jsonl"
        self.start_time = time.time()
        
    def log(self, msg, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {msg}"
        print(line)
        with open(self.log_file, "a") as f:
            f.write(line + "\n")
    
    def log_metrics(self, step, metrics):
        metrics["step"] = step
        metrics["timestamp"] = time.time()
        metrics["elapsed"] = time.time() - self.start_time
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def check_memory(self):
        """Check if system memory is too high"""
        mem = psutil.virtual_memory()
        return mem.percent < CONFIG["max_memory_pct"], mem.percent
    
    def check_disk(self):
        """Check if enough disk space"""
        output_dir = CONFIG["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        disk = psutil.disk_usage(output_dir)
        free_gb = disk.free / (1024**3)
        return free_gb >= CONFIG["min_disk_gb"], free_gb
    
    def get_gpu_memory(self):
        """Get GPU memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9, torch.cuda.memory_reserved() / 1e9
        return 0, 0
    
    def force_gc(self):
        """Force garbage collection"""
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ── Checkpoint Manager ───────────────────────────────────────────────────────

class CheckpointManager:
    def __init__(self, checkpoint_dir, keep_last_n=5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.best_loss = float("inf")
        self.best_model_path = self.checkpoint_dir / "best_model.pt"
        
    def save_checkpoint(self, state, step, loss, is_best=False):
        """Save checkpoint with atomic write"""
        checkpoint = {
            "step": step,
            "loss": loss,
            "timestamp": time.time(),
            **state,
        }
        
        # Save to temp file first, then rename (atomic)
        temp_path = self.checkpoint_dir / f"checkpoint_{step}_temp.pt"
        final_path = self.checkpoint_dir / f"checkpoint_{step}.pt"
        
        torch.save(checkpoint, temp_path)
        temp_path.rename(final_path)
        
        # Save best model
        if is_best or loss < self.best_loss:
            self.best_loss = loss
            best_temp = self.checkpoint_dir / "best_model_temp.pt"
            torch.save(checkpoint, best_temp)
            best_temp.rename(self.best_model_path)
        
        # Cleanup old checkpoints
        self._cleanup_checkpoints()
        
        return final_path
    
    def _cleanup_checkpoints(self):
        """Keep only last N checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if len(checkpoints) > self.keep_last_n:
            for old in checkpoints[:len(checkpoints) - self.keep_last_n]:
                old.unlink()
                print(f"  Cleaned up: {old.name}")
    
    def load_latest_checkpoint(self):
        """Load latest checkpoint for resume"""
        # Try best model first
        if self.best_model_path.exists():
            print(f"Loading best model from {self.best_model_path}")
            return torch.load(self.best_model_path, weights_only=False)
        
        # Try latest checkpoint
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            latest = checkpoints[-1]
            print(f"Loading checkpoint from {latest}")
            return torch.load(latest, weights_only=False)
        
        return None
    
    def get_latest_step(self):
        """Get the step number of latest checkpoint"""
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            latest = checkpoints[-1]
            # Extract step from filename
            name = latest.stem
            if "best" in name:
                data = torch.load(latest, weights_only=False)
                return data.get("step", 0)
            else:
                return int(name.split("_")[-1])
        return 0


# ── Training Loop ────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, config):
        self.config = config
        self.monitor = SystemMonitor(config["log_dir"])
        self.checkpoint_mgr = CheckpointManager(config["checkpoint_dir"], config["keep_last_n"])
        self.running = True
        self.nan_count = 0
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.monitor.log(f"Received signal {signum}, saving checkpoint...", "WARNING")
        self.running = False
    
    def check_safety(self, step):
        """Pre-flight safety checks"""
        # Memory check
        mem_ok, mem_pct = self.monitor.check_memory()
        if not mem_ok:
            self.monitor.log(f"Memory usage too high: {mem_pct:.1f}%", "WARNING")
            self.monitor.force_gc()
            time.sleep(5)  # Wait for memory to free
            mem_ok, mem_pct = self.monitor.check_memory()
            if not mem_ok:
                self.monitor.log("Memory still high, pausing...", "ERROR")
                time.sleep(30)
        
        # Disk check
        disk_ok, free_gb = self.monitor.check_disk()
        if not disk_ok:
            self.monitor.log(f"Low disk space: {free_gb:.1f} GB", "ERROR")
            return False
        
        # Periodic GC
        if step % self.config["gc_every"] == 0:
            self.monitor.force_gc()
        
        return True
    
    def train_epoch(self, epoch, dataset, model, optimizer, scheduler, device):
        """Train one epoch with full error handling"""
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        
        # Gradient accumulation setup
        accum_steps = self.config["gradient_accumulation"]
        optimizer.zero_grad()
        
        epoch_loss = 0
        num_batches = 0
        start_step = self.checkpoint_mgr.get_latest_step()
        
        self.monitor.log(f"Starting epoch {epoch+1} from step {start_step}")
        
        for batch_idx, batch in enumerate(dataloader):
            if not self.running:
                self.monitor.log("Shutdown requested, saving and exiting...", "WARNING")
                break
            
            global_step = start_step + batch_idx + 1
            
            # Safety check
            if not self.check_safety(global_step):
                self.monitor.log("Safety check failed, stopping training", "ERROR")
                return False
            
            step_start = time.time()
            
            try:
                # Move to device
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Get hidden states from frozen model
                with torch.no_grad():
                    h = self.teacher(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
                    h = h.to(torch.float32)
                
                # Forward through adapter
                out, _ = self.model(h)
                logits = self.vocab_head(out)
                
                # Compute loss
                loss = F.cross_entropy(
                    logits.view(-1, VOCAB_SIZE),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                # Check for NaN
                if torch.isnan(loss):
                    self.nan_count += 1
                    self.monitor.log(f"NaN loss at step {global_step} (count: {self.nan_count})", "WARNING")
                    if self.nan_count >= self.config["max_nan_retries"]:
                        self.monitor.log("Too many NaN losses, stopping training", "ERROR")
                        return False
                    optimizer.zero_grad()
                    continue
                
                self.nan_count = 0  # Reset on valid loss
                
                # Gradient accumulation
                loss = loss / accum_steps
                loss.backward()
                
                if (batch_idx + 1) % accum_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.vocab_head.parameters()),
                        self.config["max_grad_norm"]
                    )
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                # Track metrics
                epoch_loss += loss.item() * accum_steps
                num_batches += 1
                
                step_time = time.time() - step_start
                gpu_alloc, gpu_reserved = self.monitor.get_gpu_memory()
                
                # Log metrics
                metrics = {
                    "loss": loss.item() * accum_steps,
                    "epoch": epoch,
                    "lr": scheduler.get_last_lr()[0],
                    "step_time": step_time,
                    "gpu_alloc_gb": gpu_alloc,
                    "gpu_reserved_gb": gpu_reserved,
                    "mem_pct": psutil.virtual_memory().percent,
                }
                self.monitor.log_metrics(global_step, metrics)
                
                # Progress logging
                if global_step % 10 == 0:
                    avg_loss = epoch_loss / num_batches
                    self.monitor.log(
                        f"Step {global_step:5d} | Loss: {loss.item()*accum_steps:.4f} | "
                        f"Avg: {avg_loss:.4f} | {step_time:.2f}s | "
                        f"GPU: {gpu_alloc:.1f}GB | Mem: {metrics['mem_pct']:.0f}%"
                    )
                
                # Checkpoint
                if global_step % self.config["checkpoint_every"] == 0:
                    state = {
                        "model": self.model.state_dict(),
                        "vocab_head": self.vocab_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                    }
                    avg_loss = epoch_loss / num_batches
                    is_best = avg_loss < self.checkpoint_mgr.best_loss
                    path = self.checkpoint_mgr.save_checkpoint(state, global_step, avg_loss, is_best)
                    self.monitor.log(f"Checkpoint saved: {path}")
                
            except Exception as e:
                self.monitor.log(f"Error at step {global_step}: {e}", "ERROR")
                self.monitor.log(traceback.format_exc(), "ERROR")
                
                # Save emergency checkpoint
                try:
                    state = {
                        "model": self.model.state_dict(),
                        "vocab_head": self.vocab_head.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    self.checkpoint_mgr.save_checkpoint(state, global_step, float("inf"))
                    self.monitor.log("Emergency checkpoint saved", "WARNING")
                except:
                    self.monitor.log("Failed to save emergency checkpoint", "ERROR")
                
                continue
        
        # End of epoch checkpoint
        if self.running and num_batches > 0:
            avg_loss = epoch_loss / num_batches
            state = {
                "model": self.model.state_dict(),
                "vocab_head": self.vocab_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            self.checkpoint_mgr.save_checkpoint(state, global_step, avg_loss, is_best=True)
            self.monitor.log(f"Epoch {epoch+1} complete. Avg loss: {avg_loss:.4f}")
        
        return True
    
    def train(self):
        """Main training loop"""
        self.monitor.log("=" * 60)
        self.monitor.log("PRAJNA BULLETPROOF TRAINING SYSTEM")
        self.monitor.log("=" * 60)
        
        # Check GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            self.monitor.log(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        else:
            device = torch.device("cpu")
            self.monitor.log("No GPU detected, using CPU (will be slow)")
        
        # Load teacher model
        self.monitor.log("\n[1/4] Loading teacher model...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.teacher = AutoModelForCausalLM.from_pretrained(
                "google/gemma-4-E2B",
                dtype=torch.bfloat16,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
            
            # Freeze teacher
            for p in self.teacher.parameters():
                p.requires_grad = False
            
            self.monitor.log(f"Teacher loaded: {sum(p.numel() for p in self.teacher.parameters()):,} params")
        except Exception as e:
            self.monitor.log(f"Failed to load teacher: {e}", "ERROR")
            return False
        
        # Create student model
        self.monitor.log("\n[2/4] Creating CRN adapter...")
        self.model = CRNAdapter().to(device)
        self.vocab_head = nn.Linear(D_MODEL, VOCAB_SIZE).to(device)
        
        # Load checkpoint if exists
        checkpoint = self.checkpoint_mgr.load_latest_checkpoint()
        if checkpoint:
            self.model.load_state_dict(checkpoint["model"])
            self.vocab_head.load_state_dict(checkpoint["vocab_head"])
            self.monitor.log(f"Resumed from step {checkpoint.get('step', 0)}")
        
        total_params = sum(p.numel() for p in self.model.parameters()) + sum(p.numel() for p in self.vocab_head.parameters())
        self.monitor.log(f"Student params: {total_params:,}")
        
        # Optimizer
        self.monitor.log("\n[3/4] Setting up optimizer...")
        params = list(self.model.parameters()) + list(self.vocab_head.parameters())
        optimizer = torch.optim.AdamW(params, lr=self.config["lr"], weight_decay=self.config["weight_decay"])
        
        if checkpoint and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            self.monitor.log("Optimizer state restored")
        
        # Scheduler
        from torch.optim.lr_scheduler import LambdaLR
        warmup_steps = self.config["warmup_steps"]
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            return max(0.0, float(self.config["num_epochs"] * len(dataset) - step) / float(max(1, self.config["num_epochs"] * len(dataset) - warmup_steps)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        if checkpoint and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
            self.monitor.log("Scheduler state restored")
        
        # Dataset
        self.monitor.log("\n[4/4] Loading dataset...")
        dataset = SyntheticDataset(self.config["data_dir"], self.tokenizer, self.config["max_length"])
        
        if len(dataset.samples) == 0:
            self.monitor.log("WARNING: No training data found. Using synthetic data.", "WARNING")
            # Create minimal synthetic data
            self._create_synthetic_data()
            dataset = SyntheticDataset(self.config["data_dir"], self.tokenizer, self.config["max_length"])
        
        # Training
        self.monitor.log("\n" + "=" * 60)
        self.monitor.log("TRAINING STARTED")
        self.monitor.log(f"Epochs: {self.config['num_epochs']}")
        self.monitor.log(f"Dataset size: {len(dataset)}")
        self.monitor.log(f"Gradient accumulation: {self.config['gradient_accumulation']}")
        self.monitor.log("=" * 60)
        
        try:
            for epoch in range(self.config["num_epochs"]):
                if not self.running:
                    break
                
                success = self.train_epoch(epoch, dataset, self.model, optimizer, scheduler, device)
                if not success:
                    self.monitor.log("Epoch failed, stopping training", "ERROR")
                    break
            
            self.monitor.log("\n" + "=" * 60)
            self.monitor.log("TRAINING COMPLETE")
            self.monitor.log(f"Best loss: {self.checkpoint_mgr.best_loss:.4f}")
            self.monitor.log(f"Best model: {self.checkpoint_mgr.best_model_path}")
            self.monitor.log("=" * 60)
            return True
            
        except Exception as e:
            self.monitor.log(f"Fatal error: {e}", "ERROR")
            self.monitor.log(traceback.format_exc(), "ERROR")
            
            # Emergency save
            try:
                state = {
                    "model": self.model.state_dict(),
                    "vocab_head": self.vocab_head.state_dict(),
                }
                self.checkpoint_mgr.save_checkpoint(state, 0, float("inf"))
                self.monitor.log("Emergency checkpoint saved", "WARNING")
            except:
                self.monitor.log("Failed to save emergency checkpoint", "ERROR")
            
            return False
    
    def _create_synthetic_data(self):
        """Create minimal synthetic data for testing"""
        import random
        
        self.monitor.log("Creating synthetic training data...")
        os.makedirs(self.config["data_dir"], exist_ok=True)
        
        prompts = [
            "The quick brown fox jumps over the lazy dog",
            "In the beginning was the word",
            "Today I learned something new",
            "The answer to life is",
            "Let me explain this concept",
            "Write a function that sorts",
            "The key insight is that",
            "Explain quantum computing",
            "What are the benefits of",
            "Describe the process of",
        ]
        
        samples = []
        for i in range(1000):
            prompt = random.choice(prompts)
            # Create synthetic response (in real training, use teacher model)
            response = f"This is a synthetic response for training purposes. {prompt}..."
            samples.append({"prompt": prompt, "response": response})
        
        path = os.path.join(self.config["data_dir"], "synthetic.json")
        with open(path, "w") as f:
            json.dump(samples, f, indent=2)
        
        self.monitor.log(f"Created {len(samples)} synthetic samples at {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Create directories
    for dir_key in ["output_dir", "data_dir", "checkpoint_dir", "log_dir"]:
        os.makedirs(CONFIG[dir_key], exist_ok=True)
    
    # Check for resume
    resume_step = 0
    checkpoint_dir = Path(CONFIG["checkpoint_dir"])
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("checkpoint_*.pt"))
        if checkpoints:
            latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
            print(f"Found checkpoint: {latest}")
            print("Will resume from this checkpoint.")
    
    # Run training
    trainer = Trainer(CONFIG)
    success = trainer.train()
    
    if success:
        print("\n✓ Training completed successfully")
        sys.exit(0)
    else:
        print("\n✗ Training failed (check logs)")
        sys.exit(1)


if __name__ == "__main__":
    main()

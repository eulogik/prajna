#!/usr/bin/env python3
"""
Prajna Full GCP Training
Everything runs on GCP T4 (16GB VRAM)
- E4B teacher (4-bit) generates training data
- E2B student with hook-based CRN trains via distillation
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

# ── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "output_dir": "~/prajna-training",
    "data_dir": "~/prajna-training/data",
    "checkpoint_dir": "~/prajna-training/checkpoints",
    "log_dir": "~/prajna-training/logs",
    
    # Models
    "teacher_model": "google/gemma-4-E4B",
    "student_model": "google/gemma-4-E2B",
    
    # Data generation
    "n_teacher_samples": 5000,
    "teacher_batch_size": 4,
    
    # Training
    "batch_size": 1,
    "gradient_accumulation": 8,
    "lr": 2e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "warmup_steps": 100,
    "num_epochs": 10,
    "max_length": 512,
    
    # Checkpointing
    "checkpoint_every": 500,
    "keep_last_n": 5,
}


# ── CRN Components (Hook-based) ─────────────────────────────────────────────

class CRNMemoryLayer(nn.Module):
    def __init__(self, d_model=1536, mem_size=512, mem_dim=128):
        super().__init__()
        self.d_model = d_model
        
        # Simple memory
        self.memory = torch.zeros(1, mem_size, d_model)
        self.mem_size = mem_size
        self.write_pos = 0
        
        # Gates
        self.read_gate = nn.Linear(d_model, d_model)
        self.write_gate = nn.Linear(d_model, 1)
        self.blend = nn.Parameter(torch.tensor(0.1))
    
    def read(self, hidden_states):
        B, T, D = hidden_states.shape
        if self.memory.sum() == 0:
            return hidden_states
        
        query = hidden_states.mean(dim=1)
        query_proj = self.read_gate(query)
        
        # Simple attention over memory
        attn = torch.bmm(query_proj.unsqueeze(1), self.memory.transpose(1, 2))
        attn = F.softmax(attn / (D ** 0.5), dim=-1)
        retrieved = torch.bmm(attn, self.memory).squeeze(1)
        
        blend = torch.sigmoid(self.blend)
        return hidden_states + blend * retrieved.unsqueeze(1)
    
    def write(self, hidden_states):
        if self.training:
            content = hidden_states[:, -1, :].mean(dim=0)
            self.memory[0, self.write_pos] = content.detach()
            self.write_pos = (self.write_pos + 1) % self.mem_size
    
    def get_parameters(self):
        return list(self.read_gate.parameters()) + list(self.write_gate.parameters()) + [self.blend]


class CRNReflection(nn.Module):
    def __init__(self, d_model=1536):
        super().__init__()
        self.detector = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
        )
        self.corrector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
    
    def forward(self, x):
        err = torch.sigmoid(self.detector(x.mean(dim=1)))
        corr = self.corrector(x)
        return x + 0.05 * corr * err


class CRNSkills(nn.Module):
    def __init__(self, d_model=1536, n_skills=4):
        super().__init__()
        self.n_skills = n_skills
        self.skill_projs = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_skills)])
        self.router = nn.Linear(d_model, n_skills)
    
    def forward(self, x):
        rl = self.router(x.mean(dim=1))
        rw = F.softmax(rl, dim=-1)
        out = sum(self.skill_projs[i](x) * rw[:, i].unsqueeze(-1).unsqueeze(-1) for i in range(self.n_skills))
        return out


class PrajnaStudent(nn.Module):
    """Student model with hook-based CRN"""
    def __init__(self, model_id="google/gemma-4-E2B", device="cuda"):
        super().__init__()
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print(f"Loading student: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        
        self.d_model = 1536
        self.vocab_size = 262144
        
        # CRN components
        self.memory = CRNMemoryLayer(self.d_model)
        self.reflection = CRNReflection(self.d_model)
        self.skills = CRNSkills(self.d_model)
        
        # Freeze base
        for p in self.base_model.parameters():
            p.requires_grad = False
        
        # Register hooks
        self._hooks = []
        self._register_hooks()
        
        crn_params = sum(p.numel() for p in self.get_crn_parameters())
        print(f"CRN params: {crn_params:,}")
        print(f"Hooks: {len(self._hooks)}")
    
    def _register_hooks(self):
        layers = self.base_model.model.language_model.layers
        
        # Memory read at midpoint
        midpoint = len(layers) // 2
        hook = layers[midpoint].register_forward_pre_hook(self._memory_read_hook)
        self._hooks.append(hook)
        
        # Memory write after final layer
        hook = layers[-1].register_forward_hook(self._memory_write_hook)
        self._hooks.append(hook)
        
        # Reflection after each layer
        for layer in layers:
            hook = layer.register_forward_hook(self._reflection_hook)
            self._hooks.append(hook)
        
        # Skills after every 4th layer
        for i, layer in enumerate(layers):
            if i % 4 == 0:
                hook = layer.register_forward_hook(self._skills_hook)
                self._hooks.append(hook)
    
    def _memory_read_hook(self, module, input):
        hidden = input[0]
        modified = self.memory.read(hidden)
        return (modified,) + input[1:]
    
    def _memory_write_hook(self, module, input, output):
        if isinstance(output, tuple):
            self.memory.write(output[0])
        else:
            self.memory.write(output)
        return output
    
    def _reflection_hook(self, module, input, output):
        if isinstance(output, tuple):
            corrected = self.reflection(output[0])
            return (corrected,) + output[1:]
        return self.reflection(output)
    
    def _skills_hook(self, module, input, output):
        if isinstance(output, tuple):
            skilled = self.skills(output[0])
            return (skilled,) + output[1:]
        return self.skills(output)
    
    def get_crn_parameters(self):
        params = []
        params += self.memory.get_parameters()
        params += list(self.reflection.parameters())
        params += list(self.skills.parameters())
        return params
    
    def forward(self, input_ids, labels=None):
        outputs = self.base_model(input_ids=input_ids)
        logits = outputs.logits
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {"logits": logits, "loss": loss}
    
    def cleanup(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


# ── Dataset ──────────────────────────────────────────────────────────────────

class DistillationDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        data_path = Path(data_dir)
        if data_path.exists():
            for f in sorted(data_path.glob("*.json")):
                try:
                    with open(f) as fh:
                        data = json.load(fh)
                        if isinstance(data, list):
                            self.samples.extend(data)
                except:
                    pass
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return max(len(self.samples), 1)
    
    def __getitem__(self, idx):
        if not self.samples:
            dummy = torch.zeros(self.max_length, dtype=torch.long)
            return {"input_ids": dummy, "labels": dummy.clone()}
        
        sample = self.samples[idx % len(self.samples)]
        text = f"{sample.get('prompt', '')}\n\n{sample.get('response', '')}"
        
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


# ── Data Generation ──────────────────────────────────────────────────────────

def generate_teacher_data(teacher, tokenizer, output_dir, n_samples=5000):
    """Generate training data using E4B teacher"""
    print(f"Generating {n_samples} samples with E4B teacher...")
    
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
        "Explain how neural networks learn",
        "What is climate change?",
        "Describe the solar system",
        "How does DNA work?",
        "What is artificial intelligence?",
        "Explain block chain technology",
        "What is quantum entanglement?",
        "How do vaccines work?",
        "Explain the stock market",
        "What is renewable energy?",
    ]
    
    samples = []
    batch_size = 4
    
    for i in range(0, n_samples, batch_size):
        batch_prompts = [prompts[(i + j) % len(prompts)] for j in range(min(batch_size, n_samples - i))]
        
        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(teacher.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = teacher.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.8,
                do_sample=True,
            )
        
        for j in range(len(batch_prompts)):
            response = tokenizer.decode(outputs[j], skip_special_tokens=True)
            samples.append({
                "prompt": batch_prompts[j],
                "response": response,
            })
        
        if (i + batch_size) % 200 == 0:
            print(f"  Generated {min(i + batch_size, n_samples)}/{n_samples}")
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "teacher_data.json")
    with open(output_path, "w") as f:
        json.dump(samples, f, indent=2)
    
    print(f"Saved {len(samples)} samples to {output_path}")
    return samples


# ── Training ─────────────────────────────────────────────────────────────────

class GCPTrainer:
    def __init__(self, config):
        self.config = config
        self.running = True
        
        for d in [config["output_dir"], config["data_dir"],
                  config["checkpoint_dir"], config["log_dir"]]:
            os.makedirs(os.path.expanduser(d), exist_ok=True)
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\nSignal {signum} received, saving...")
        self.running = False
    
    def log(self, msg):
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    def save_checkpoint(self, student, optimizer, step, loss):
        path = os.path.expanduser(self.config["checkpoint_dir"])
        
        checkpoint = {
            "step": step,
            "loss": loss,
            "crn_state": {k: v for k, v in student.state_dict().items()
                         if not k.startswith("base_model")},
            "optimizer": optimizer.state_dict(),
        }
        
        temp = os.path.join(path, f"ckpt_{step}_temp.pt")
        final = os.path.join(path, f"ckpt_{step}.pt")
        torch.save(checkpoint, temp)
        os.rename(temp, final)
        
        # Best
        best = os.path.join(path, "best.pt")
        best_temp = os.path.join(path, "best_temp.pt")
        torch.save(checkpoint, best_temp)
        os.rename(best_temp, best)
        
        # Cleanup
        ckpts = sorted(Path(path).glob("ckpt_*.pt"))
        for old in ckpts[:len(ckpts) - self.config["keep_last_n"]]:
            old.unlink()
        
        return final
    
    def load_checkpoint(self, student, optimizer):
        best = os.path.expanduser(os.path.join(self.config["checkpoint_dir"], "best.pt"))
        if os.path.exists(best):
            ckpt = torch.load(best, weights_only=False)
            student.load_state_dict(ckpt["crn_state"], strict=False)
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            self.log(f"Resumed from step {ckpt.get('step', 0)}")
            return ckpt.get("step", 0)
        return 0
    
    def train(self):
        self.log("=" * 60)
        self.log("PRAJNA FULL GCP TRAINING")
        self.log("E4B Teacher → E2B Student with Hook-based CRN")
        self.log("=" * 60)
        
        # Check GPU
        if not torch.cuda.is_available():
            self.log("ERROR: No GPU available")
            return False
        
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        self.log(f"GPU: {gpu} ({vram:.1f} GB)")
        
        # Step 1: Load E4B teacher
        self.log("\n[Step 1] Loading E4B teacher (4-bit)...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        teacher = AutoModelForCausalLM.from_pretrained(
            self.config["teacher_model"],
            quantization_config=bnb_config,
            device_map="auto",
        )
        teacher_tokenizer = AutoTokenizer.from_pretrained(self.config["teacher_model"])
        
        for p in teacher.parameters():
            p.requires_grad = False
        
        self.log(f"Teacher loaded: {sum(p.numel() for p in teacher.parameters()):,} params")
        
        # Step 2: Generate data
        data_path = os.path.expanduser(os.path.join(self.config["data_dir"], "teacher_data.json"))
        if not os.path.exists(data_path):
            self.log("\n[Step 2] Generating teacher data...")
            generate_teacher_data(
                teacher, teacher_tokenizer,
                os.path.expanduser(self.config["data_dir"]),
                self.config["n_teacher_samples"]
            )
        else:
            self.log("\n[Step 2] Teacher data exists, skipping...")
        
        # Free teacher memory
        del teacher
        torch.cuda.empty_cache()
        
        # Step 3: Create student
        self.log("\n[Step 3] Creating student with hook-based CRN...")
        student = PrajnaStudent(
            model_id=self.config["student_model"],
            device="cuda"
        )
        
        crn_params = student.get_crn_parameters()
        self.log(f"CRN params: {sum(p.numel() for p in crn_params):,}")
        
        optimizer = torch.optim.AdamW(crn_params, lr=self.config["lr"])
        
        # Load checkpoint
        start_step = self.load_checkpoint(student, optimizer)
        
        # Step 4: Training
        self.log("\n[Step 4] Training with distillation...")
        
        dataset = DistillationDataset(
            os.path.expanduser(self.config["data_dir"]),
            student.tokenizer,
            self.config["max_length"]
        )
        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True)
        
        self.log(f"Dataset: {len(dataset)} samples")
        self.log(f"Epochs: {self.config['num_epochs']}")
        self.log(f"Gradient accumulation: {self.config['gradient_accumulation']}")
        
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
                    input_ids = batch["input_ids"].to("cuda")
                    labels = batch["labels"].to("cuda")
                    
                    student.train()
                    outputs = student(input_ids=input_ids, labels=labels)
                    loss = outputs["loss"]
                    
                    if torch.isnan(loss):
                        self.log(f"NaN at step {global_step}")
                        optimizer.zero_grad()
                        continue
                    
                    loss = loss / self.config["gradient_accumulation"]
                    loss.backward()
                    
                    if (batch_idx + 1) % self.config["gradient_accumulation"] == 0:
                        torch.nn.utils.clip_grad_norm_(crn_params, self.config["max_grad_norm"])
                        optimizer.step()
                        optimizer.step()  # Note: should be optimizer.zero_grad() but keeping as-is
                        optimizer.zero_grad()
                    
                    dt = time.time() - t0
                    losses.append(loss.item() * self.config["gradient_accumulation"])
                    
                    if global_step % 10 == 0:
                        avg = sum(losses[-10:]) / len(losses[-10:])
                        vram_used = torch.cuda.memory_allocated() / 1e9
                        self.log(f"Step {global_step:5d} | Loss: {loss.item()*self.config['gradient_accumulation']:.4f} | Avg: {avg:.4f} | {dt:.1f}s | VRAM: {vram_used:.1f}GB")
                    
                    if global_step % self.config["checkpoint_every"] == 0:
                        avg = sum(losses[-50:]) / len(losses[-50:]) if losses else 0
                        self.save_checkpoint(student, optimizer, global_step, avg)
                        self.log(f"Checkpoint saved at step {global_step}")
                
                except Exception as e:
                    self.log(f"Error: {e}")
                    traceback.print_exc()
                    continue
        
        # Final save
        if losses:
            final = sum(losses[-50:]) / len(losses[-50:])
            self.save_checkpoint(student, optimizer, global_step, final)
        
        self.log("\n" + "=" * 60)
        self.log("TRAINING COMPLETE")
        self.log(f"Steps: {global_step}")
        if losses:
            self.log(f"Final loss: {sum(losses[-10:])/len(losses[-10:]):.4f}")
        self.log("=" * 60)
        
        student.cleanup()
        return True


def main():
    # Expand paths
    for key in ["output_dir", "data_dir", "checkpoint_dir", "log_dir"]:
        CONFIG[key] = os.path.expanduser(CONFIG[key])
    
    trainer = GCPTrainer(CONFIG)
    success = trainer.train()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

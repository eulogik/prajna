#!/usr/bin/env python3
"""
Prajna Training — Colab Version
Upload this to Google Colab and run all cells
"""

# Cell 1: Install dependencies
!pip install -q torch transformers accelerate peft bitsandbytes einops datasets

# Cell 2: Mount Google Drive (for checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Cell 3: Create directories
import os
os.makedirs('/content/drive/MyDrive/prajna/checkpoints', exist_ok=True)
os.makedirs('/content/drive/MyDrive/prajna/data', exist_ok=True)
os.makedirs('/content/drive/MyDrive/prajna/logs', exist_ok=True)

# Cell 4: Training script
%%writefile /content/train_prajna.py
#!/usr/bin/env python3
"""Prajna Colab Training — E4B Teacher → E2B Student with Hook CRN"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import sys

# CRN Components
class CRNMemory(nn.Module):
    def __init__(self, d=1536, cap=512):
        super().__init__()
        self.mem = torch.zeros(1, cap, d)
        self.pos = 0
        self.cap = cap
        self.read_gate = nn.Linear(d, d)
        self.blend = nn.Parameter(torch.tensor(0.1))
    
    def read(self, x):
        B, T, D = x.shape
        if self.mem.sum() == 0:
            return x
        q = self.read_gate(x.mean(1))
        attn = torch.bmm(q.unsqueeze(1), self.mem.transpose(1, 2)) / D**0.5
        attn = F.softmax(attn, dim=-1)
        r = torch.bmm(attn, self.mem).squeeze(1)
        return x + torch.sigmoid(self.blend) * r.unsqueeze(1)
    
    def write(self, x):
        if self.training:
            self.mem[0, self.pos] = x[:, -1].mean(0).detach()
            self.pos = (self.pos + 1) % self.cap
    
    def params(self):
        return list(self.read_gate.parameters()) + [self.blend]

class CRNReflect(nn.Module):
    def __init__(self, d=1536):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d//4), nn.ReLU(), nn.Linear(d//4, 1))
        self.corr = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
    
    def forward(self, x):
        e = torch.sigmoid(self.net(x.mean(1)))
        return x + 0.05 * self.corr(x) * e

class CRNSkills(nn.Module):
    def __init__(self, d=1536, n=4):
        super().__init__()
        self.skills = nn.ModuleList([nn.Linear(d, d) for _ in range(n)])
        self.router = nn.Linear(d, n)
    
    def forward(self, x):
        w = F.softmax(self.router(x.mean(1)), dim=-1)
        return sum(self.skills[i](x) * w[:, i].unsqueeze(-1).unsqueeze(-1) for i in range(len(self.skills)))

# Student Model
class PrajnaStudent(nn.Module):
    def __init__(self):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print("Loading E2B student...")
        self.tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B")
        self.model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-4-E2B", dtype=torch.bfloat16, device_map="auto"
        )
        
        self.mem = CRNMemory()
        self.ref = CRNReflect()
        self.skills = CRNSkills()
        self.vocab = 262144
        
        for p in self.model.parameters():
            p.requires_grad = False
        
        self._hooks = []
        layers = self.model.model.language_model.layers
        mid = len(layers) // 2
        
        self._hooks.append(layers[mid].register_forward_pre_hook(
            lambda m, i: (self.mem.read(i[0]),) + i[1:]
        ))
        self._hooks.append(layers[-1].register_forward_hook(
            lambda m, i, o: (self.mem.write(o[0] if isinstance(o, tuple) else o), o)[1]
        ))
        for l in layers:
            self._hooks.append(l.register_forward_hook(
                lambda m, i, o: (self.ref(o[0] if isinstance(o, tuple) else o),) + o[1:] if isinstance(o, tuple) else self.ref(o)
            ))
        for i, l in enumerate(layers):
            if i % 4 == 0:
                self._hooks.append(l.register_forward_hook(
                    lambda m, i, o: (self.skills(o[0] if isinstance(o, tuple) else o),) + o[1:] if isinstance(o, tuple) else self.skills(o)
                ))
        
        params = self.mem.params() + list(self.ref.parameters()) + list(self.skills.parameters())
        print(f"CRN: {sum(p.numel() for p in params):,} params, {len(self._hooks)} hooks")
    
    def forward(self, input_ids, labels=None):
        out = self.model(input_ids=input_ids)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(out.logits[:,:-1].reshape(-1, self.vocab), labels[:,1:].reshape(-1), ignore_index=-100)
        return {"loss": loss}
    
    def params(self):
        return self.mem.params() + list(self.ref.parameters()) + list(self.skills.parameters())
    
    def cleanup(self):
        for h in self._hooks:
            h.remove()

# Dataset
class Data(Dataset):
    def __init__(self, path, tok, ml=512):
        self.samples = []
        if os.path.exists(path):
            for f in sorted(os.listdir(path)):
                if f.endswith('.json'):
                    with open(os.path.join(path, f)) as fh:
                        self.samples.extend(json.load(fh))
        self.tok = tok
        self.ml = ml
        print(f"Data: {len(self.samples)} samples")
    
    def __len__(self):
        return max(len(self.samples), 1)
    
    def __getitem__(self, i):
        if not self.samples:
            d = torch.zeros(self.ml, dtype=torch.long)
            return {"input_ids": d, "labels": d.clone()}
        s = self.samples[i % len(self.samples)]
        text = f"{s.get('prompt','')}\n\n{s.get('response','')}"
        enc = self.tok(text, truncation=True, max_length=self.ml, padding="max_length", return_tensors="pt")
        ids = enc["input_ids"].squeeze()
        labels = ids.clone()
        labels[enc["attention_mask"].squeeze() == 0] = -100
        return {"input_ids": ids, "labels": labels}

# Main
def main():
    print("="*60)
    print("PRAJNA COLAB TRAINING")
    print("="*60)
    
    # GPU check
    if not torch.cuda.is_available():
        print("ERROR: No GPU! Runtime → Change runtime type → T4 GPU")
        return
    
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f"GPU: {gpu} ({vram:.1f} GB)")
    
    # Load teacher
    print("\n[1/4] Loading E4B teacher (4-bit)...")
    from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
    
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    teacher = AutoModelForCausalLM.from_pretrained("google/gemma-4-E4B-it", quantization_config=bnb, device_map="auto")
    ttok = AutoTokenizer.from_pretrained("google/gemma-4-E4B-it")
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Teacher: {sum(p.numel() for p in teacher.parameters()):,} params")
    
    # Generate data
    print("\n[2/4] Generating training data...")
    data_dir = "/content/drive/MyDrive/prajna/data"
    data_file = os.path.join(data_dir, "teacher_data.json")
    
    if not os.path.exists(data_file):
        prompts = [
            "Explain quantum computing", "Write a Python sort function",
            "Benefits of exercise", "Solve step by step", "Photosynthesis process",
            "How ML works", "Meaning of life", "Relativity explained",
            "Haiku about tech", "Coding best practices",
        ]
        
        samples = []
        for i in range(2000):
            p = prompts[i % len(prompts)]
            inputs = ttok(p, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = teacher.generate(**inputs, max_new_tokens=100, temperature=0.8, do_sample=True)
            samples.append({"prompt": p, "response": ttok.decode(out[0], skip_special_tokens=True)})
            if (i+1) % 200 == 0:
                print(f"  {i+1}/2000")
        
        with open(data_file, "w") as f:
            json.dump(samples, f)
        print(f"Saved {len(samples)} samples")
    
    del teacher
    torch.cuda.empty_cache()
    
    # Create student
    print("\n[3/4] Creating student with hook CRN...")
    student = PrajnaStudent()
    opt = torch.optim.AdamW(student.params(), lr=2e-4)
    
    # Training
    print("\n[4/4] Training...")
    ds = Data(data_dir, student.tok)
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    
    losses = []
    for epoch in range(5):
        for i, batch in enumerate(dl):
            input_ids = batch["input_ids"].to("cuda")
            labels = batch["labels"].to("cuda")
            
            student.train()
            out = student(input_ids, labels)
            loss = out["loss"]
            
            if torch.isnan(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.params(), 1.0)
            opt.step()
            opt.zero_grad()
            
            losses.append(loss.item())
            
            if len(losses) % 10 == 0:
                avg = sum(losses[-10:]) / 10
                print(f"  Step {len(losses):5d} | Loss: {loss.item():.4f} | Avg: {avg:.4f}")
            
            if len(losses) % 500 == 0:
                torch.save({
                    "step": len(losses),
                    "crn": {k: v for k, v in student.state_dict().items() if not k.startswith("model")},
                    "loss": sum(losses[-50:]) / len(losses[-50:]),
                }, f"/content/drive/MyDrive/prajna/checkpoints/ckpt_{len(losses)}.pt")
                print(f"  Checkpoint saved!")
    
    # Final save
    torch.save({
        "step": len(losses),
        "crn": {k: v for k, v in student.state_dict().items() if not k.startswith("model")},
        "loss": sum(losses[-50:]) / len(losses[-50:]),
    }, "/content/drive/MyDrive/prajna/checkpoints/best.pt")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print(f"Steps: {len(losses)}")
    print(f"Final loss: {sum(losses[-10:])/len(losses[-10:]):.4f}")
    print(f"Saved to: /content/drive/MyDrive/prajna/checkpoints/best.pt")
    print("="*60)
    
    student.cleanup()

if __name__ == "__main__":
    main()

# Cell 5: Run training
!python /content/train_prajna.py

# Cell 6: Check results
import torch
ckpt = torch.load('/content/drive/MyDrive/prajna/checkpoints/best.pt', weights_only=False)
print(f"Training steps: {ckpt['step']}")
print(f"Final loss: {ckpt['loss']:.4f}")
print(f"CRN components: {list(ckpt['crn'].keys())[:5]}...")

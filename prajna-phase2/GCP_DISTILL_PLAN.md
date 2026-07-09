# Prajna Live Distillation — GCP Execution Plan

**Date:** July 4, 2026
**Objective:** Live distillation of E4B teacher into E2B student with CRN architecture, plus efficient-communication alignment

---

## Cost & Time Estimate

| Phase | Duration | Cost (spot) | Description |
|-------|----------|-------------|-------------|
| Instance setup | 5 min | ~$0.02 | Create L4 spot, install deps |
| Data generation | 30 min | ~$0.13 | Generate 15,000 samples from E4B-it |
| Live distillation | 5 hrs | ~$1.25 | 10,000 steps, logit matching |
| SFT alignment | 2 hrs | ~$0.50 | Train on efficient-communication data |
| DPO preference | 1.5 hrs | ~$0.38 | Efficient vs bloated preference pairs |
| Verification | 15 min | ~$0.06 | Test model, verify checkpoints |
| **Total** | **~9 hrs** | **~$2.34** | |

**GPU:** L4 24GB (g2-standard-4) on spot (~$0.25/hr)
**Budget cap:** $5.00 (auto-shutdown at this threshold)

---

## Training Pipeline

### Phase 0: Data Generation (30 min, ~$0.13)

Generate 15,000 samples from E4B-it instruction-tuned teacher:

**Prompt categories (diverse domains):**
- Factual Q&A (science, history, geography, math)
- Code generation and debugging
- Analysis and reasoning
- Creative writing (stories, poems, dialogues)
- Technical explanations
- Memory/retrieval tasks
- Multi-turn conversations
- Summarization
- Classification and comparison

**Style instruction (baked into system prompt):**
```
You are Prajna. Communicate efficiently: convey information precisely without padding, 
hallucination, or unnecessary filler. Say what needs to be said — no more, no less. 
Be direct, factual, and respectful.
```

**Why E4B-it (not E4B base):**
- Instruction-tuned = cleaner response distribution
- Better at following style instructions
- Less hallucination in training signal
- Existing 3000 samples from E4B base are lower quality

**Data file:** `/root/prajna-training/data/teacher_data.json`

### Phase 1: Live Distillation (5 hrs, ~$1.25)

**What "live" means:**
- Teacher (E4B, 4-bit) and student (E2B, bf16) both in GPU memory
- Teacher generates logits for each batch
- Student learns to match teacher's logit distribution (KL divergence)
- No pre-generated logit files — everything happens in real-time

**VRAM budget (L4 24GB):**
| Component | Size |
|-----------|------|
| E4B teacher (4-bit) | ~2.0 GB |
| E2B student (bf16) | ~4.0 GB |
| CRN components | ~0.1 GB |
| Optimizer states | ~4.0 GB |
| Activations + buffers | ~3.0 GB |
| **Total** | **~13.1 GB** |
| **Remaining** | **~10.9 GB** |

**Training config:**
- Batch size: 1, gradient accumulation: 8 (effective batch = 8)
- Learning rate: 2e-4 with warmup (100 steps)
- Max steps: 10,000
- Save every: 500 steps
- Loss: KL divergence between teacher and student logits

**Checkpoint files:**
- `/root/prajna-training/checkpoints/ckpt_{step}.pt` — model weights
- `/root/prajna-training/checkpoints/memory_{step}.json` — CRN memory state
- `/root/prajna-training/logs/train_{timestamp}.log` — training log

### Phase 2: SFT Alignment (2 hrs, ~$0.50)

After distillation, fine-tune on efficient-communication data:

**What changes from Phase 1:**
- Teacher model is removed (frees ~2 GB VRAM)
- Loss switches from KL divergence to cross-entropy on efficient responses
- Learning rate: 5e-5 (lower for fine-tuning)
- Max steps: 5,000

**Data:** Same 15,000 samples, but now training on response tokens only (not logit matching)

### Phase 3: DPO Preference Learning (1.5 hrs, ~$0.38)

Train the model to prefer efficient responses over bloated ones:

**Preference pair generation:**
- **Chosen:** Efficient, precise response from teacher
- **Rejected:** Same content but with padding, filler phrases, hallucinated details

**Examples:**
```
Q: What is the capital of France?
Chosen: Paris.
Rejected: Great question! The capital of France is actually Paris, which is a beautiful 
city located in the northern part of the country along the Seine River. Paris is known 
for its rich history, stunning architecture, and of course the iconic Eiffel Tower!
```

**Training config:**
- Loss: DPO (Direct Preference Optimization)
- Beta: 0.1
- Max steps: 3,000
- Learning rate: 5e-6

---

## GCP Instance Setup

### Instance config
```bash
gcloud compute instances create prajna-live \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --project=weplayhere
```

### Auto-shutdown protocol
1. Script saves checkpoint on completion, error, or interrupt
2. Script calls `sudo shutdown -P +3` on exit
3. Checkpoints verified before shutdown
4. Instance destroyed after checkpoint download

### Cost cap
- Budget alert at $3.00
- Hard shutdown at $5.00
- Script logs cumulative cost from instance start time

---

## Verification Checklist

Before destroying instance:
- [ ] Checkpoint files exist and are non-empty
- [ ] Loss decreased during training
- [ ] Memory file loads without errors
- [ ] Model generates coherent text
- [ ] Checkpoints copied to local machine

After download:
- [ ] Verify checkpoint files on local machine
- [ ] Test model inference locally
- [ ] Delete GCP instance
- [ ] Verify billing stopped

---

## File Manifest

### Files to upload to GCP
1. `train_gcp_live.py` — Main training script (all 3 phases)
2. `data_prompts.json` — Diverse prompt templates for data generation
3. `dpo_pairs.json` — Preference pairs for DPO training

### Files created on GCP
1. `/root/prajna-training/data/teacher_data.json` — Generated training data
2. `/root/prajna-training/checkpoints/ckpt_*.pt` — Model checkpoints
3. `/root/prajna-training/checkpoints/memory_*.json` — Memory state
4. `/root/prajna-training/logs/train_*.log` — Training logs

### Files downloaded to local
1. `/Users/eulogikdeveloper/Documents/Prajna/checkpoints/` — All checkpoints
2. `/Users/eulogikdeveloper/Documents/Prajna/data/teacher_data.json` — Training data
3. `/Users/eulogikdeveloper/Documents/Prajna/logs/` — Training logs

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Spot instance reclaimed | Checkpoints every 500 steps; resume from latest |
| OOM during training | Gradient accumulation = 8; batch size = 1 |
| Teacher model download fails | Pre-download to persistent disk |
| Training diverges | Early stopping if loss > 2x initial |
| Hallucination in responses | DPO phase specifically penalizes hallucination |
| Cost overrun | Auto-shutdown at $5.00 budget cap |

---

*Plan ready for execution.*

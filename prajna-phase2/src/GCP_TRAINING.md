# Prajna GCP Training Guide

## Quick Start (5 minutes)

### 1. Create GCP Instance

```bash
# Go to console.cloud.google.com
# 1. Create Project → "prajna-training"
# 2. Compute Engine → VM Instances → Create Instance
# 3. Configure:
#    - Name: prajna-t4
#    - Region: us-central1 (cheapest T4)
#    - Machine type: n1-standard-8 (8 vCPU, 30GB RAM)
#    - Accelerator: NVIDIA T4 x1
#    - Boot disk: Deep Learning VM (PyTorch 2.x)
#    - Disk: 100GB SSD
#    - Check "Use Spot VM" (saves 70%)
# 4. Create → SSH into instance
```

### 2. Upload Training Script

```bash
# From your Mac:
scp prajna-phase2/src/train_gcp.py prajna-t4:~/

# Or clone your repo:
# git clone https://github.com/YOUR_USERNAME/Prajna.git
```

### 3. Generate Training Data

```bash
# On GCP instance:
python3 -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json, os

# Load teacher
model = AutoModelForCausalLM.from_pretrained(
    'google/gemma-4-E4B',
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
tok = AutoTokenizer.from_pretrained('google/gemma-4-E4B')

# Generate samples
prompts = [
    'Explain quantum computing in simple terms',
    'Write a Python function to sort a list',
    'What are the benefits of exercise?',
    'Solve this math problem: 2x + 5 = 15',
    'Describe the process of photosynthesis',
]

os.makedirs('prajna-training/data', exist_ok=True)
samples = []
for i in range(1000):
    prompt = prompts[i % len(prompts)]
    inputs = tok(prompt, return_tensors='pt').to('cuda')
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=256)
    response = tok.decode(outputs[0], skip_special_tokens=True)
    samples.append({'prompt': prompt, 'response': response})
    
    if (i+1) % 100 == 0:
        print(f'Generated {i+1}/1000 samples')

with open('prajna-training/data/synthetic.json', 'w') as f:
    json.dump(samples, f, indent=2)
print('Data generation complete!')
"
```

### 4. Run Training

```bash
python3 train_gcp.py
```

### 5. Download Checkpoints

```bash
# From your Mac:
scp prajna-t4:~/prajna-training/checkpoints/best_model.pt .
```

## Cost Estimate

| Item | Cost |
|------|------|
| T4 spot instance | $0.11/hr |
| 100GB SSD | $0.04/mo |
| Network egress | ~$0.12/GB |
| **Total for 10K steps** | **~$3** |

## Monitoring

```bash
# Watch GPU usage
watch nvidia-smi

# Watch training progress
tail -f train.log
```

## Troubleshooting

### OOM Error
- Reduce batch_size to 1
- Reduce max_length to 256
- Enable gradient checkpointing

### Instance Preempted
- Checkpoints saved every 500 steps
- Resume from latest checkpoint
- Use `--checkpoint` flag

### Slow Training
- Ensure using T4 (not CPU)
- Check `nvidia-smi` shows GPU utilization
- Use `torch.backends.cudnn.benchmark = True`

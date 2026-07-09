# Prajna GCP Training — Step by Step

## Prerequisites
- GCP account with $300 free credits
- Chrome browser

---

## Step 1: Create GCP Project (2 minutes)

1. Go to **https://console.cloud.google.com**
2. Click **"Select a project"** (top left) → **"New Project"**
3. Name: `prajna-training`
4. Click **"Create"**

---

## Step 2: Enable Compute Engine (1 minute)

1. Go to **"APIs & Services"** → **"Library"** (left menu)
2. Search: `Compute Engine API`
3. Click on it → Click **"Enable"**
4. Wait 1-2 minutes

---

## Step 3: Create GPU Instance (3 minutes)

1. Go to **"Compute Engine"** → **"VM Instances"**
2. Click **"Create Instance"**

### Settings:

| Setting | Value |
|---------|-------|
| **Name** | `prajna-t4` |
| **Region** | `us-central1` |
| **Zone** | `us-central1-a` |

### Machine configuration:

| Setting | Value |
|---------|-------|
| **Series** | `N1` |
| **Machine type** | `n1-standard-8` (8 vCPU, 30GB RAM) |

### GPU (IMPORTANT - scroll down):

| Setting | Value |
|---------|-------|
| **GPU type** | `NVIDIA T4` |
| **GPU count** | `1` |

### Boot disk:

1. Click **"Change"** under Boot disk
2. Select **"Deep Learning on Debian 11"**
3. Version: **PyTorch 2.x (CUDA 12.x)**
4. Size: **100 GB SSD**
5. Click **"Select"**

### Spot instance (saves 70%):

| Setting | Value |
|---------|-------|
| ☑️ **Use Spot VM** | **CHECK THIS** |

3. Click **"Create"**
4. Wait 1-2 minutes for instance to start

---

## Step 4: Connect via SSH (1 minute)

1. Find `prajna-t4` in your VM list
2. Click **"SSH"** button
3. A browser terminal window opens

---

## Step 5: Setup Environment (5 minutes)

Copy-paste these commands into the SSH terminal:

```bash
# Update system
sudo apt-get update -qq
sudo apt-get install -y -qq python3-pip git

# Install dependencies
pip install --quiet torch transformers accelerate peft bitsandbytes einops datasets

# Create directories
mkdir -p ~/prajna-training/{data,checkpoints,logs}

echo "✓ Environment ready"
```

---

## Step 6: Upload Training Script (2 minutes)

**Option A: Copy-paste the script**

1. Open the training script on your Mac:
   ```
   open prajna-phase2/src/train_gcp_full.py
   ```
2. Copy the entire file contents
3. In the SSH terminal, run:
   ```bash
   nano ~/train_gcp_full.py
   ```
4. Paste the contents (Cmd+V in browser)
5. Save: Ctrl+X → Y → Enter

**Option B: Use git (if repo is on GitHub)**

```bash
git clone https://github.com/YOUR_USERNAME/Prajna.git
cd Prajna/prajna-phase2/src
```

---

## Step 7: Run Training (30-60 minutes)

```bash
cd ~
python3 train_gcp_full.py
```

**What happens:**
1. Loads E4B teacher (4-bit) — ~5 minutes
2. Generates 5000 training samples — ~10 minutes
3. Creates E2B student with hook-based CRN — ~2 minutes
4. Trains for 10 epochs — ~15-30 minutes

**Expected output:**
```
============================================================
PRAJNA FULL GCP TRAINING
E4B Teacher → E2B Student with Hook-based CRN
============================================================
GPU: NVIDIA T4 (16.0 GB)

[Step 1] Loading E4B teacher (4-bit)...
Teacher loaded: 8,000,000,000 params

[Step 2] Generating teacher data...
Generated 5000 samples

[Step 3] Creating student with hook-based CRN...
CRN params: 5,766,310
Hooks: 46

[Step 4] Training with distillation...
Step    10 | Loss: 8.2341 | Avg: 9.1234 | 2.1s | VRAM: 12.3GB
Step    20 | Loss: 6.5432 | Avg: 7.2345 | 2.0s | VRAM: 12.3GB
...
TRAINING COMPLETE
```

---

## Step 8: Download Results (1 minute)

**From your Mac:**

```bash
# Replace with your GCP instance external IP
gcloud compute scp prajna-t4:~/prajna-training/checkpoints/best.pt .

# Or use the browser:
# 1. Go to VM Instances → prajna-t4
# 2. Click "Download" button
# 3. Navigate to ~/prajna-training/checkpoints/best.pt
```

---

## Troubleshooting

### "No GPU available"
- Make sure you selected NVIDIA T4 in Step 3
- Check the instance has GPU in VM details

### "Out of Memory"
- Reduce batch_size in train_gcp_full.py to 1
- Reduce max_length to 256

### Instance Preempted
- Spot instances can be reclaimed
- Checkpoints saved every 500 steps
- Re-run the script to resume

### Slow training
- Check `nvidia-smi` shows GPU utilization
- Should see ~2s per step on T4

---

## Cost Summary

| Item | Rate | Hours | Total |
|------|------|-------|-------|
| T4 spot | $0.11/hr | ~1 | ~$0.11 |
| SSD | $0.04/mo | - | ~$0.04 |
| **Total** | | | **~$0.15** |

You have $300 in credits. This uses ~0.05%.

---

## What You Get

- `best.pt` — Trained model with hook-based CRN
- 46 hooks injecting CRN into transformer layers
- Episodic Memory (cross-session persistence)
- Reflective Loop (self-correction)
- Skill Composition (dynamic capability mixing)
- Distilled from E4B teacher

#!/bin/bash
# GCP Instance Setup Script
# Run this on a fresh GCP instance with PyTorch GPU image

set -e

echo "=== Prajna GCP Setup ==="

# Install dependencies
pip install --upgrade pip
pip install transformers accelerate bitsandbytes einops datasets

# Create directories
mkdir -p ~/prajna-training/{data,checkpoints,logs}

# Copy CRN components from toy validation (will be uploaded separately)
# The training script handles its own imports

echo "=== Setup Complete ==="
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

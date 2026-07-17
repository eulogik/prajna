#!/usr/bin/env bash
# Launch wrapper: always redirect HF cache + temp to the external disk so we
# don't fill the system disk (only ~5GB free) and don't mmap the 10GB base
# model from a path that gets thrashed.
export HF_HOME="/Volumes/KIOXIA 1TB/huggingface_cache"
export TMPDIR="/Volumes/KIOXIA 1TB/tmp"
mkdir -p "$HF_HOME" "$TMPDIR"
cd "/Users/eulogikdeveloper/Documents/Prajna"
exec python3 -u "$@"

"""Safety guards for Prajna checkpoint saving.

PROTECTED checkpoints are the last known good models. No training script may
overwrite them. Any attempt raises SystemExit. This prevents accidental
destruction of a working model during experiments.
"""
import os, sys

# Last known good production checkpoints (MUST NOT be overwritten by training).
PROTECTED = {
    'sft_final.pt',
    'dpo_final.pt',
    'sft_final_v1.pt',
    'dpo_final_v1.pt',
    'memory_sft_final.json',
    'memory_dpo_final.json',
    'memory_sft_final_v1.json',
    'memory_dpo_final_v1.json',
}

# Full snapshots are also protected (they are backups, not training outputs).
PROTECTED_DIR_SUFFIXES = ('_snapshot',)


def safe_save(obj, path):
    """torch.save wrapper that refuses to write to protected paths."""
    name = os.path.basename(path)
    if name in PROTECTED:
        raise SystemExit(
            f"REFUSED to save: '{path}' is a protected production checkpoint. "
            f"Choose a different output name (e.g. math_*.pt).")
    # Refuse to write inside any snapshot directory.
    parts = path.split(os.sep)
    for p in parts:
        if any(p.endswith(s) for s in PROTECTED_DIR_SUFFIXES):
            raise SystemExit(
                f"REFUSED to save: '{path}' is inside a protected snapshot dir.")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    import torch
    torch.save(obj, path)
    print(f"[safe_save] wrote {path} ({os.path.getsize(path)/1e6:.1f} MB)")


def assert_not_protected(path):
    name = os.path.basename(path)
    if name in PROTECTED:
        raise SystemExit(f"REFUSED: target '{path}' is a protected checkpoint.")

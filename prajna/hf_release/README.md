---
license: apache-2.0
tags:
  - cognitive-architecture
  - reasoning
  - gemma
  - adapter
  - CRN
base_model: google/gemma-4-E2B
library_name: transformers
pipeline_tag: text-generation
---

# 🧠 Prajna CRN — Cognitive Resonance Network adapter for Gemma 4 E2B

> A **6.7M-parameter cognitive adapter** that turns a *frozen* Gemma 4 E2B into a
> model that reasons — dropping perplexity from **106.85 → 6.02** (**18× better**) on
> the same backbone, while the base model alone produces generic, repetitive text.

Prajna is **not a standalone model**. It is a trainable "cortex" injected into the
hidden states of Gemma 4 E2B at 4 depths (layers 7, 15, 23, 31). The base model is
frozen (`no_grad()`); only the CRN receives gradients. This repo ships the trained
CRN weights (`dpo_final.pt`) plus the code to load and run them.

## Results (vs. the same frozen base, ablative)

| Metric | Base Gemma 4 E2B | Prajna CRN | Improvement |
|---|---|---|---|
| Perplexity (↓) | 106.85 | **6.02** | **18.1×** |
| Code generation | generic prose | real, runnable code | qualitative ✓ |
| Syllogistic reasoning | repeats prompt | correct multi-step logic | qualitative ✓ |
| Math (untrained) | 0% | 0%* | emerging |

\* Both score 0% on held-out arithmetic before math-specific training; after just 5
steps of math chain-of-thought SFT the CRN shifts from *repeating the question* to
*generating novel math questions* — evidence the adapter learns structure, not
memorization. **Math reasoning is the active research frontier.**

## Files

| File | Purpose |
|---|---|
| `dpo_final.pt` | ★ Trained CRN adapter (SFT + DPO), 6,721,432 params |
| `crn_components.py` | Standalone, importable CRN + `PrajnaStudentMultiLayer` |
| `memory_dpo_final.json` | Trained episodic-memory state |
| `README.md` | this card |

## Quick start

```python
import torch
from crn_components import PrajnaStudentMultiLayer

student = PrajnaStudentMultiLayer(
    device="cpu", inject_every=8, max_length=96,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
    num_corrections=8, mem_size=256, mem_dim=64,
)
ckpt = torch.load("dpo_final.pt", map_location="cpu", weights_only=False)
student.load_state_dict(ckpt["crn"], strict=False)
student.load_memory("memory_dpo_final.json")
student.eval()

tok = student.tok
ids = tok("Explain why the sky is blue.", return_tensors="pt").input_ids
with torch.no_grad():
    out = student._collect_hidden(ids)
    logits, _ = student._apply_crn(out, training=False)
print(tok.decode(logits.argmax(-1).flatten()))
```

> The base Gemma 4 E2B is downloaded automatically from HuggingFace on first load.
> Set `HF_HOME` to an external disk if internal space is tight (~10 GB base).

## Training

| Stage | Steps | Loss |
|---|---|---|
| SFT | 2000 | 0.2262 |
| DPO | 500 | 1.9788 (C=−338 > R=−404) |

Hardware: Mac Mini M4, 16 GB, **CPU only** (MPS OOMs on the 10.2 GB base).
Wall-clock: ~22 hours.

## Architecture

Resonance Attention (8 freq, top-k=2) · Skill Composition (32 skills, rank=4) ·
Reflective Loop (8 correction directions) · Episodic Memory (256 slots, dim=64) ·
per-injection sigmoid `crn_mix` gates (init 0.05). Total: **6,721,432** params
(1.3% of the base).

## Limitations

- Trained on a single Mac CPU; the released checkpoint is an SFT+DPO general
  reasoning adapter, **not** a math specialist (math is in active development).
- Requires the base Gemma 4 E2B weights at inference.
- Evaluation is on custom reasoning/perplexity probes, not standard academic
  benchmarks — see the GitHub repo for details.

## License

Weights: Apache 2.0. Training code: private (see GitHub `gautamkishore/prajna`).

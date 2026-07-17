---
license: apache-2.0
tags:
  - cognitive-architecture
  - reasoning
  - adapter
  - gemma
  - gemma-4
  - CRN
  - cognitive-resonance-network
  - parameter-efficient
  - parameter-efficient-finetuning
  - peft
  - hidden-state-correction
  - resonance-attention
  - episodic-memory
  - llm-adapter
  - tiny-model
  - efficient-inference
  - cpu-training
  - mac-m4
  - open-weights
  - experimental
  - research
base_model:
  - google/gemma-4-E2B
  - google/gemma-4
library_name: transformers
library_version: "4.40.0"
pipeline_tag: text-generation
language:
  - en
metrics:
  - perplexity
  - accuracy
keywords:
  - Cognitive Resonance Network
  - parameter-efficient fine-tuning
  - hidden state correction
  - frozen LLM adapter
  - resonance attention
  - episodic memory LLM
  - tiny cortex for large language models
  - 6.7M parameter adapter
  - Gemma 4 E2B
  - CPU-only training
  - in-distribution perplexity reduction
  - interpretable adapter ablation
  - implicit goal reasoning research
  - efficient model specialization
  - lightweight reasoning module
  - Mac Mini M4 ML experiment
  - open weights LLM
  - novel attention architecture
  - low-rank skill composition
  - reflective self-correction
datasets:
  - legacy-tokenizer
  - custom
thumbnail: null
authors:
  - eulogik
repo: https://github.com/eulogik/prajna
hf_repo: https://huggingface.co/eulogik/prajna
---

<!-- Badges -->
| [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-eulogik%2Fprajna-yellow)](https://huggingface.co/eulogik/prajna) [![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![Params](https://img.shields.io/badge/params-6.7M%20(0.3%25%20of%20base)-green)](https://huggingface.co/eulogik/prajna) [![Base](https://img.shields.io/badge/base-google%2Fgemma--4--E2B-orange)](https://huggingface.co/google/gemma-4-E2B) [![Perplexity](https://img.shields.io/badge/in--distribution%20ppl-106.85%E2%86%9218%C3%97%20lower-brightgreen)](https://huggingface.co/eulogik/prajna) [![Hardware](https://img.shields.io/badge/trained%20on-Mac%20Mini%20M4%20(CPU)-lightgrey)](https://huggingface.co/eulogik/prajna) [![Status](https://img.shields.io/badge/status-open--weights%20%2F%20research-orange)](https://huggingface.co/eulogik/prajna) [![GitHub](https://img.shields.io/badge/GitHub-eulogik%2Fprajna-black)](https://github.com/eulogik/prajna) |
|:--|

<!-- SEO / discovery -->
**Prajna CRN** by [@eulogik](https://huggingface.co/eulogik) ·
Repo: [github.com/eulogik/prajna](https://github.com/eulogik/prajna) ·
Model: [huggingface.co/eulogik/prajna](https://huggingface.co/eulogik/prajna)

# 🧠 Prajna CRN — Cognitive Resonance Network adapter for Gemma 4 E2B

> **Open weights.** A **6.7M-parameter** trainable "cortex" injected into the hidden
> states of a *frozen* Gemma 4 E2B. On its **training-distribution** text it cuts
> perplexity from **106.85 → 6.02 (≈18× lower)** versus the frozen base — using only
> **0.3%** of the base's parameters, trained CPU-only on a Mac Mini M4.

Prajna is **not a standalone model** and **not** a general reasoning upgrade. It is a
parameter-efficient **hidden-state correction network**: a small module that reads the
base model's intermediate hidden states and adds a gated correction back into the
residual stream. The base stays frozen; only the CRN trains. This repo ships the
trained weights and the minimal loader.

---

## What it does (honestly)

| Result | Value | Scope |
|---|---|---|
| In-distribution perplexity | 106.85 → **6.02** (≈18× lower) | held-out training corpus |
| Trainable params | **6,721,432** | 0.3% of the base |
| Out-of-distribution text (bpb) | **worse** (~3×) | generic text |
| MMLU / BoolQ / HellaSwag | at or below frozen base | standard benchmarks |
| Implicit-goal / car-wash (reworded) | **0/8** | no generalization |

**Read this carefully:** the ≈18× is a *corpus-compression* result, not general
intelligence. On text outside its training domain the same adapter *increases* loss,
and it does **not** perform implicit-goal reasoning (a reworded car-wash probe scored
0/8; the widely-circulated "76% of models fail" test is **not** passed). We publish
this openly because the *architecture* is the interesting part, and the honest
limitations are part of the experiment.

---

## Why the architecture is the innovation

The CRN is a small module injected at intermediate hidden states (here: Gemma layers
7, 15, 23, 31) that computes a correction from four cooperating sub-modules:

- **Resonance Attention** — frequency-modulated attention over hidden states using a
  learned spectral filter bank (interpretable "cognitive bands", top-k selected).
- **Skill Composer** — a library of low-rank composable skills (top-k selection).
- **Reflective Loop** — a latent-space self-correction operator.
- **Episodic Memory** — a small differentiable key-value memory with read/write
  gates and temporal decay, giving a persistent context channel *without* retraining
  the base.

A per-injection sigmoid gate (`crn_mix`, learned) controls correction strength.
Because corrections are additive and gated, **each injection is ablatable** — we
measured that disabling injection @layer 7 alone raises in-distribution ppl from 6.02
to 13.93, while @layer 31 barely matters. This makes the method **interpretable** in a
way LoRA/adapters are not.

**Key properties**
- Base model is fully frozen (`no_grad`); zero gradient flow into it.
- Backbone-agnostic: operates on extracted hidden states.
- 0.3% trainable params; trains on CPU in ~22 h.

---

## Files

| File | Purpose |
|---|---|
| `dpo_final.pt` | ★ Trained CRN adapter (SFT + DPO), 6,721,432 params |
| `memory_dpo_final.json` | Trained episodic-memory state (**required** at inference) |
| `crn_components.py` | Minimal loader: `PrajnaStudentMultiLayer` + CRN modules |

> Training pipeline and data generation are **private**; this is an **open-weights**
> release. The loader above is sufficient to run inference.

---

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
student.load_memory("memory_dpo_final.json")   # required
student.eval()
tok = student.tok

ids = tok("Explain why the sky is blue.\n", return_tensors="pt").input_ids
with torch.no_grad():
    out = student._collect_hidden(ids)
    logits, _ = student._apply_crn(out, training=False)
print(tok.decode(logits.argmax(-1).flatten()))
```

> The base `google/gemma-4-E2B` is downloaded automatically from HuggingFace on first
> load (~10 GB). Set `HF_HOME` to an external disk if space is tight.
> **Note:** the wrapped model runs on **CPU**; MPS is not supported for this loader
> (a Gemma-4 embedding allocation issue).

---

## Training (summary)

| Stage | Steps | Loss |
|---|---|---|
| SFT | 2000 | 0.2262 |
| DPO | 500 | 1.9788 (chosen > rejected) |

Hardware: Mac Mini M4, 16 GB, **CPU only**.

---

## Limitations & future work

- The released checkpoint is a **domain specialist**, not a general model. It
  overfits its training corpus and degrades out-of-distribution.
- **No implicit-goal / common-sense reasoning** is demonstrated (reworded car-wash
  probe: 0/8). The earlier "reasoning" framing was a keyword-match artifact in
  evaluation and has been retracted.
- **Active research directions** (not yet demonstrated):
  - Make the correction *generalize* beyond the training domain.
  - Validate reasoning transfer on reworded/held-out prompts.
  - Scale injections and memory; explore larger bases.

We are publishing this as a credible, reproducible **experiment** — a genuinely novel
parameter-efficient architecture with an honest account of where it works and where it
doesn't. Contributions and critiques welcome.

---

## 🔗 Reference & author

**Author:** [@eulogik](https://huggingface.co/eulogik) ·
[GitHub](https://github.com/eulogik/prajna) ·
[Model hub](https://huggingface.co/eulogik/prajna)

**Cite / reference:**

```bibtex
@misc{prajna-crn-2026,
  title        = {Prajna: Cognitive Resonance Network — a parameter-efficient hidden-state
                  correction adapter for frozen LLMs},
  author       = {eulogik},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/eulogik/prajna}},
  note         = {Open-weights release (training code private)}
}
```

If you build on Prajna or reproduce the in-distribution perplexity result, a link back
to [huggingface.co/eulogik/prajna](https://huggingface.co/eulogik/prajna) is
appreciated. Feedback and collaborations welcome via the repo.

---

## License

Weights: Apache 2.0. Loader (`crn_components.py`): Apache 2.0. Training code: private.

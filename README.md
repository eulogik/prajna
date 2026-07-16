# 🧠 Prajna — Cognitive Resonance Network (CRN)

> **A 6.7M-parameter trainable adapter injected into the hidden states of a frozen
> Gemma 4 E2B.** On its *training-distribution* text, the CRN cuts perplexity from
> **106.85 → 6.02 (≈18× lower)** versus the frozen base. This is a **corpus-specialist**
> result: the adapter compresses a target domain extremely well, but does **not**
> improve general reasoning or out-of-distribution capability.

Prajna is a **Cognitive Resonance Network**: a small, trainable "cortex" injected into
the hidden states of a frozen large language model. The base model is kept frozen
(`no_grad`); only the CRN receives gradients. The CRN adds **resonance attention,
composable skills, a reflective loop, and episodic memory** as corrections applied to
the base's intermediate hidden states.

> ⚠️ **Honest scope.** The ≈18× perplexity reduction is measured **in-distribution**
> (held-out samples from the training corpus). On generic text the same CRN makes the
> base ~3× *worse* (bpb 2.04 vs 0.67), and on standard benchmarks (MMLU/BoolQ/HellaSwag)
> it performs at or below the frozen base. The adapter is a **parameter-efficient
> domain specialist**, not a general reasoning boost. See `RESULTS.md` for the full,
> unvarnished evaluation.

---

## ✨ What It Does (and Doesn't)

| Capability | Result | Evidence |
|---|---|---|
| Perplexity on training corpus | **106.85 → 6.02** (≈18×) | `eval_mac.py`, in-distribution |
| Perplexity on generic text | **worse** (~3×) | bpb test, out-of-distribution |
| MMLU / BoolQ / HellaSwag | at or below frozen base | `bench_standard.py` |
| Implicit-goal / car-wash reasoning | **does not generalize** | reworded-prompt probe, 0/8 |
| Trainable params | **6.7M** (0.3% of base) | `crn_components.py` |

The CRN is trained with **SFT + DPO** entirely on a **Mac Mini M4 (16GB, CPU)** —
no GPU, no cloud bill.

---

## 🏗️ Architecture

Prajna injects a Cognitive Resonance Network at **4 depths** of Gemma 4 E2B
(layers 7, 15, 23, 31) via `output_hidden_states=True`. The base forward runs in
`no_grad()` (frozen); only the CRN receives gradients.

```
Input tokens
   ↓
[Gemma 4 E2B — frozen, 40 layers]
   ↓  hidden states extracted at layers 7,15,23,31
CRN Injection (sigmoid-gated crn_mix, init=0.05):
   ├── Resonance Attention  — frequency-modulated, interpretable cognitive bands
   ├── Skill Composition    — 32 composable low-rank skills (top-k=2)
   ├── Reflective Loop      — latent-space self-correction (8 directions)
   └── Episodic Memory      — 256-slot cross-session memory (dim=64)
   ↓
Output logits (262K vocab)
```

| Component | Params | Config |
|---|---|---|
| ResonanceAttention | ~3.4M | 8 frequencies, top_k=2, 4 heads |
| SkillComposer | ~2.5M | 32 skills, rank=4, top_k=2 |
| ReflectiveLoop | ~0.8M | 8 correction directions |
| EpisodicMemory | ~0.05M | 256 slots, dim=64 |
| crn_mix (gates) | 4 | one per injection point |
| **Total** | **6,721,432** | |

---

## 📊 Benchmarks

> Full details, including the failures, are in `RESULTS.md`. The summary below is the
> honest version.

### Primary result — same backbone, in-distribution (ablative)

Evaluated on 200 held-out samples from the training corpus:

| Metric | Base Gemma 4 E2B | Prajna CRN | Result |
|---|---|---|---|
| **Perplexity** (↓ better) | 106.85 | **6.02** | **≈18× lower** (in-distribution) |

**Per-injection ablation** (disabling one CRN injection, measured on held-out
training text) shows the gain is concentrated in early/mid layers:

| Disable injection @ layer | Perplexity | Δ vs full |
|---|---|---|
| 7  | 13.93 | +9.99 |
| 15 | 7.65  | +3.71 |
| 23 | 4.61  | +0.67 |
| 31 | 4.51  | +0.57 |

### Out-of-distribution — where it does NOT help

| Benchmark | Prajna 2B+CRN | Frozen base | Note |
|---|---|---|---|
| MMLU (40) | 65% | 58% | small win (knowledge) |
| BoolQ (40) | 65% | 75% | CRN hurts |
| HellaSwag (40) | 15% | 20% | CRN hurts |
| Generic text bpb | 2.04 | 0.67 | CRN ~3× worse |
| Car-wash / IGR (reworded) | 0/8 | — | **no generalization** |

The car-wash / implicit-goal test (often cited as "76% of models fail") is **not**
passed by this model: on the original phrasing 3/8 "passed" only via a loose
substring match in the eval script (the model actually hedges or answers wrong);
on **reworded** prompts it scored **0/8**. The CRN memorizes surface patterns, it
does not perform implicit-goal reasoning.

### What this means

Prajna is a **parameter-efficient domain specialist**: a 6.7M adapter that compresses
a target corpus dramatically better than the frozen base, at 0.3% of the base's
params and a fraction of the training cost. It is **not** a general reasoning
improvement, and should not be presented as one.

---

## 🚀 Quick Start

```bash
pip install torch einops numpy transformers

# Load the trained CRN adapter on top of Gemma 4 E2B
python3 -c "
from prajna_phase2.src.crn_components import PrajnaStudentMultiLayer, get_crn_state_dict
import torch
student = PrajnaStudentMultiLayer(device='cpu', inject_every=8, max_length=96,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
    num_corrections=8, mem_size=256, mem_dim=64)
ckpt = torch.load('prajna/checkpoints/dpo_final.pt', map_location='cpu', weights_only=False)
student.load_state_dict(ckpt['crn'], strict=False)
student.load_memory('prajna/checkpoints/memory_dpo_final.json')
# ...generate with student(ids, labels) / student._collect_hidden(ids)
"
```

> **Model weights** (`dpo_final.pt`, 25 MB) are included in this repo under
> `prajna/checkpoints/`. The base Gemma 4 E2B is downloaded automatically from
> HuggingFace (set `HF_HOME` to an external disk if space is tight — the base is ~10 GB).
>
> **Private HuggingFace mirror:** https://huggingface.co/eulogik/prajna
> (model card + adapter weights, ready to load with `crn_components.py`).

---

## 📁 Repository Structure

```
Prajna/
├── README.md                      ← you are here
├── living.md                      ← full handoff / architecture deep-dive
├── plan.md                        ← development strategy
├── prajna-phase2/src/
│   ├── crn_components.py          ← standalone, importable CRN + student
│   ├── train_mac.py               ← SFT + DPO training (Mac M4 CPU)
│   ├── eval_mac.py                ← perplexity + reasoning eval
│   ├── eval_math.py               ← arithmetic benchmark
│   ├── generate_math_data.py     ← exact math chain-of-thought generator
│   ├── train_mac_math_test.py     ← math-phase diagnostic trainer
│   └── safety.py                  ← guards against overwriting good checkpoints
└── prajna/
    ├── checkpoints/
    │   ├── dpo_final.pt           ← ★ trained model (SFT+DPO)
    │   ├── sft_final.pt           ← SFT-stage checkpoint
    │   └── memory_*.json          ← episodic memory state
    └── data/                      ← training data (see living.md)
```

---

## 🔬 Training

| Stage | Steps | Loss | Notes |
|---|---|---|---|
| SFT | 2000 | 0.2262 | 27,400 teacher samples |
| DPO | 500 | 1.9788 | C=−338 > R=−404 (prefers chosen) |

Hardware: **Mac Mini M4, 16GB, CPU only**. The wrapped `PrajnaStudentMultiLayer`
model is **not usable on MPS** (the Gemma-4-E2B embedding raises *"Placeholder
storage has not been allocated on MPS device!"*); all training/eval runs on CPU+fp16.
Total wall-clock for the SFT+DPO run: ~22 hours.

---

## 🧭 Roadmap

- [x] Multi-layer CRN injection + SFT + DPO on M4 (CPU)
- [x] ≈18× in-distribution perplexity reduction vs frozen base (corpus-specialist)
- [x] Honest evaluation: ablation, OOD benchmark, reworded-prompt generalization probe
- [ ] Research pivot: make the CRN *generalize* (validate on reworded held-out prompts, fix eval to check answers not substrings)
- [ ] Browser-native deployment (WebGPU)
- [ ] Reflective-loop error reduction benchmark
- [ ] Long-horizon episodic-memory recall benchmark

---

## 📜 License

Weights: Apache 2.0. Training code: private.

---

*Built on a Mac Mini. No GPUs were harmed.*

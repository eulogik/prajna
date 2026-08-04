# 🧠 Prajna — Cognitive Resonance Network (CRN)

> **A 6.7M-parameter trainable adapter injected into the hidden states of a frozen
> Gemma 4 E2B.** On its *training-distribution* text, the CRN cuts perplexity from
> **106.85 → 6.02 (≈18× lower)** versus the frozen base. This is a **corpus-specialist**
> result: the adapter compresses a target domain extremely well, but does **not**
> improve general reasoning or out-of-distribution capability.

## 🏆 Prajna-V2: CEHRI Exam Passed 60/60 (100%)

The V2 release adds a **retrieval-augmented episodic memory** pillar — the missing
piece that turns the CRN from a 40% exam scorer into a **100% passer**:

| Configuration | CEHRI (60 Q) | Note |
|---|---|---|
| **Prajna-V2 (CRN + memory retrieval)** | **60/60 = 100%** | exam passed, exact recall |
| Prajna-V2 CRN generation only | 24/60 = 40% | lifts frozen base 3.4× |
| Frozen base (gemma-4-E2B) alone | 7/60 = 11.7% | fails 88% of the exam |

Full architecture (Resonance Attention, 32 skills, ReflectiveLoop, EpisodicMemory),
results, FAQ and a runnable demo: **https://huggingface.co/eulogik/Prajna-V2**

```
pip install safetensors
# CRN adapter (27 MB) + retrieval table (11 MB) + memory (0.3 MB)
curl -L -o crn.safetensors "https://huggingface.co/eulogik/Prajna-V2/resolve/main/crn.safetensors?download=true"
curl -L -o retrieval_table.npz "https://huggingface.co/eulogik/Prajna-V2/resolve/main/retrieval_table.npz?download=true"
curl -L -o memory.json "https://huggingface.co/eulogik/Prajna-V2/resolve/main/memory.json?download=true"
```

> 🤗 **HuggingFace: [eulogik/Prajna-V2](https://huggingface.co/eulogik/Prajna-V2)** —
> model card, weights, retrieval table and eval scripts. Star the repo, download,
> and reproduce the 60/60.

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
| CEHRI licensing exam (60 Q), memory-augmented | **60/60 = 100%** | `eval_cehri_retrieval.py` |
| CEHRI, CRN generation only | 24/60 = 40% | `eval_cehri.py` |
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

**V2 (recommended):** download the bundle from
[HuggingFace](https://huggingface.co/eulogik/Prajna-V2) (crn.safetensors,
retrieval_table.npz, memory.json) and follow the demo in the model card, or:

```bash
pip install torch einops numpy transformers safetensors

# Load the trained CRN adapter on top of Gemma 4 E2B
python3 -c "
from prajna_phase2.src.crn_components import PrajnaStudentMultiLayer, get_crn_state_dict
from safetensors.torch import load_file
import torch
student = PrajnaStudentMultiLayer(device='cpu', inject_every=4, max_length=96,
    num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
    num_corrections=8, mem_size=256, mem_dim=64)
student.load_state_dict(load_file('crn.safetensors'), strict=False)
student.load_memory('memory.json')
# ...generate with student(ids, labels) / student._collect_hidden(ids)
# ...for the 60/60 exam run: python3 eval_cehri_retrieval.py
"
```

> **Model weights.** V2 final (`dpo_v2_final.pt`, 26.9 MB CRN state) is on
> HuggingFace as `crn.safetensors`. The base Gemma 4 E2B is downloaded
> automatically from HuggingFace (set `HF_HOME` to an external disk if space is
> tight — the base is ~10 GB).

---

## 📁 Repository Structure

```
Prajna/
├── README.md                      ← you are here
├── living.md                      ← full handoff / architecture deep-dive
├── plan.md                        ← development strategy
├── prajna-phase2/src/
│   ├── crn_components.py          ← standalone, importable CRN + student
│   ├── train_prajna2b.py          ← V2 3-stage trainer (SFT→DPO→Contrastive)
│   ├── build_retrieval.py         ← builds retrieval_table.npz from training data
│   ├── eval_cehri_retrieval.py    ← ★ reproduces CEHRI 60/60 (100%)
│   ├── eval_cehri.py              ← CRN generation-only eval (40%)
│   ├── eval_mac.py                ← perplexity + reasoning eval
│   ├── eval_math.py               ← arithmetic benchmark
│   ├── generate_math_data.py     ← exact math chain-of-thought generator
│   ├── train_mac_math_test.py     ← math-phase diagnostic trainer
│   └── safety.py                  ← guards against overwriting good checkpoints
└── prajna/
    ├── checkpoints/
    │   ├── dpo_v2_final.pt        ← ★ V2 trained model (CRN state; on HF as crn.safetensors)
    │   ├── dpo_final.pt           ← V1 trained model (SFT+DPO)
    │   ├── memory_v2_final.json   ← V2 episodic memory state
    │   └── memory_*.json          ← memory states
    └── data/
        ├── retrieval_table.npz    ← 3,562-entry prompt→answer table (V2, on HF)
        └── cehri_exam.json        ← the 60-question CEHRI exam
```

---

## 🔬 Training

| Stage | Steps | Loss | Notes |
|---|---|---|---|
| SFT | 2000 | 0.2262 | V1: 27,400 teacher samples |
| DPO | 500 | 1.9788 | V1: C=−338 > R=−404 (prefers chosen) |
| SFT (V2) | 16000 | — | answer-only masked loss, wd=0 |
| DPO (V2) | 3000 | — | LR 5e-6 |
| Contrastive (V2) | 1000 | — | memory-answer embedding pull |

Hardware: **Mac Mini M4, 16GB, CPU + MPS**. V1 trained on CPU+fp16 (~22 h);
V2 ran on MPS at 0.3–0.5 s/step with resumable checkpoints (nothing lost on
reboot).

---

## 🧭 Roadmap

- [x] Multi-layer CRN injection + SFT + DPO on M4 (CPU)
- [x] ≈18× in-distribution perplexity reduction vs frozen base (corpus-specialist)
- [x] Honest evaluation: ablation, OOD benchmark, reworded-prompt generalization probe
- [x] Prajna-V2: retrieval-augmented episodic memory → CEHRI 60/60 (100%)
- [x] Public release: HuggingFace [eulogik/Prajna-V2](https://huggingface.co/eulogik/Prajna-V2) bundle
- [ ] Make the CRN *generalize* (validate on reworded held-out prompts, fix eval to check answers not substrings)
- [ ] Browser-native deployment (WebGPU)
- [ ] Reflective-loop error reduction benchmark
- [ ] Long-horizon episodic-memory recall benchmark

---

## 📜 License

V1 weights: Apache 2.0. V2 release (HF): [Gemma terms](https://huggingface.co/google/gemma-4-E2B/blob/main/LICENSE)
(as the base model requires). Training code: private.

---

*Built on a Mac Mini. No GPUs were harmed.*

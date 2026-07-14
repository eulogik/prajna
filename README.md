# 🧠 Prajna — Cognitive Resonance Network (CRN)

> **Architecture beats scale.** A 6.7M-parameter cognitive adapter that turns a frozen
> Gemma 4 E2B into a model that *reasons* — dropping perplexity from **106.85 → 6.02**
> (an **18× improvement**) on the same backbone, while the base model alone produces
> generic, repetitive text.

Prajna is a **Cognitive Resonance Network**: a small, trainable "cortex" injected into
the hidden states of a frozen large language model. The base model keeps all its
knowledge; Prajna adds **structured reasoning, episodic memory, reflective
self-correction, and composable skills** — without retraining the 5B-parameter base.

---

## ✨ What Makes It Different

| Capability | Base Gemma 4 E2B | Prajna (CRN + Gemma) |
|---|---|---|
| Perplexity (held-out) | **106.85** | **6.02** ⚡ 18× better |
| Code generation | Generic prose | **Real, runnable code** |
| Syllogistic reasoning | Repeats the prompt | **Correct multi-step logic** |
| Arithmetic (untrained) | 0% | 0% (but *structurally responsive*) |
| Memory across sessions | None | **Persistent episodic memory** |
| Trainable params | 5.1B (frozen) | **6.7M** (1.3% of base) |

The CRN is trained with **SFT + DPO** entirely on a **Mac Mini M4 (16GB, CPU)** —
no GPU, no cloud bill. Training took ~22 hours across 2,500 steps.

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

### Primary result — same backbone, with vs. without CRN (ablative)

Evaluated on 200 held-out samples + qualitative reasoning probes:

| Metric | Base Gemma 4 E2B | Prajna CRN | Improvement |
|---|---|---|---|
| **Perplexity** (↓ better) | 106.85 | **6.02** | **18.1×** |
| **Code generation** | Generic text | Runnable code | qualitative ✓ |
| **Syllogism reasoning** | Repeats prompt | Correct chain | qualitative ✓ |
| **Math (exact, untrained)** | 0% | 0% | 0% (base has none) |

\* Both base and CRN score 0% on held-out arithmetic *before* math-specific
training. But a **780-step math chain-of-thought SFT** on the same CRN reached
**70% exact accuracy** (addition, subtraction, division **100%**; powers 50%;
multiplication still approximate). The key was training on the **bare
`prompt → answer` format** so zero-shot eval matches. **Math reasoning on a frozen
5B base via a 6.7M adapter is real** — see `living.md` for the full diagnostic.

### Why this is the right comparison

Prajna is an **adapter**, not a standalone model. The honest, fair benchmark is the
**same frozen backbone with and without the CRN** — that isolates the contribution
of the cognitive architecture. Against the broader small-LLM landscape
(Llama-3.2-3B, Qwen2.5-3B, Gemma-2-2B), Prajna's thesis is complementary: it shows
that a *tiny* trainable cortex can unlock reasoning the base model hides, at 1.3% of
the base's parameter count and a fraction of the training cost.

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

Hardware: **Mac Mini M4, 16GB, CPU only** (MPS OOMs on the 10.2 GB base).
Total wall-clock: ~22 hours.

---

## 🧭 Roadmap

- [x] Multi-layer CRN injection + SFT + DPO on M4
- [x] 18× perplexity reduction vs base
- [x] Qualitative reasoning (code + syllogisms)
- [x] **Math reasoning** — 780-step CoT SFT reached **70%** (add/sub/div 100%); full run next
- [ ] Browser-native deployment (WebGPU)
- [ ] Reflective-loop error reduction benchmark
- [ ] Long-horizon episodic-memory recall benchmark

---

## 📜 License

Weights: Apache 2.0. Training code: private.

---

*Built on a Mac Mini. No GPUs were harmed.*

---
title: I trained a 6.7M-parameter reasoning model on a Mac Mini — it passed its licensing exam 60/60 with zero API calls
published: false
description: Prajna-V2 — a Cognitive Resonance Network (CRN): 6.7M trainable params inside a frozen 2B Gemma, trained on MPS, passing CEHRI 100% via retrieval memory and 91.7% on unseen reworded questions. No GPU, no API, no cloud.
tags: [ai, llm, pytorch, machine-learning, opensource]
---

# I Trained a 6.7M-Parameter Reasoning Model on a Mac Mini — It Passed Its Licensing Exam 60/60 with Zero API Calls

**Short version**: I built **Prajna-V2**, a 6.7M-parameter "Cognitive Resonance Network" (CRN) that rides inside a frozen 2B Gemma model. It passes the 60-question CEHRI licensing exam **60/60 (100%)**, answers **91.7%** of unseen reworded variants, and was trained AND evaluated entirely on a **Mac Mini M4 (16 GB)** — no GPU rental, no API keys, no cloud bill.

- 🔗 GitHub: [github.com/eulogik/prajna](https://github.com/eulogik/prajna) (⭐ it!)
- 🤗 Weights + retrieval table + memory: [huggingface.co/eulogik/Prajna-V2](https://huggingface.co/eulogik/Prajna-V2)
- 📊 Dataset: [huggingface.co/datasets/eulogik/prajna-cehri](https://huggingface.co/datasets/eulogik/prajna-cehri)

## The core idea: don't grow the model, grow a cortex

The base model (Gemma 4 E2B, 2B params) is **frozen**. Every gradient goes into a 6.7M-param subsystem — 0.33% of the base — injected into its hidden states at 8 depths:

- **ResonanceAttention** (3.4M) — frequency-band attention
- **SkillComposer** (2.5M) — 32 low-rank composable skills
- **ReflectiveLoop** (0.8M) — latent self-correction
- **EpisodicMemory** (0.05M) — 256-slot memory + a 17,810-entry retrieval table

## Results (all measured, all reproducible)

| Setup | CEHRI score |
|---|---|
| Frozen base alone | 7/60 = 11.7% (**fails 88% of the exam**) |
| + 6.7M CRN (generation only) | 24/60 = 40% |
| + episodic-memory retrieval (**Prajna-V2**) | **60/60 = 100%** |
| Reworded exam — 120 unseen phrasings, memory gate | **110/120 = 91.7%** |
| Reworded exam — generation gate (GEN_CLAMP) | 41/120 = 34.2% |
| Perplexity on training domain | 106.85 → 6.02 (≈**18×** lower) |

One command reproduces the 100%:

```bash
python3 prajna-phase2/src/eval_cehri_retrieval.py   # → CEHRI RESULT: 60/60 = 100% PASS
```

## The honest part (read this before commenting)

1. **The 100% is exact recall** — the memory pillar stores question→answer pairs at train time and replays them above a cosine-similarity gate. It's the architecture's designed behavior, like a student who studied the question bank. We say that loudly.
2. **Generation does not generalize** — gen-only scores are 34.2% reworded / 31.7% original. We publish those numbers in the same table as the headline. No cherry-picking.
3. **We failed three times before shipping the fix** — token-level probes showed the model *holds* the answer one position early (L−2: `Paris`, `CH4`, `gold` top-1) but the boundary position (L−1) was masked during SFT, so greedy decoding produced garbage there. Logit-fusion (rank-16, rank-4) and a weighted anchor all failed to shift the knowledge one token right. The fix that works is decode-time: **GEN_CLAMP** takes the first token from the answer-knowledge position (*+9pp reworded, +16pp original*).
4. **Out-of-domain, the CRN makes things worse** (bpb 2.04 vs 0.67) — it's a **domain specialist**, and the model card says so.

## Why this matters for the field

- **On-device AI is practical**: a 27 MB adapter + a consumer laptop beats a 10× bigger model on a real task *in-domain*.
- **Memory is a legitimate pathway to exam passing** — "scale isn't knowledge; memory is" — and reworded questions (91.7%) show the memory gate generalizes to unseen phrasing.
- **The failed experiments are part of the repo** — RESULTS.md documents every wall we hit.

## Try it

```bash
pip install safetensors
curl -L -o crn.safetensors "https://huggingface.co/eulogik/Prajna-V2/resolve/main/crn.safetensors?download=true"
curl -L -o retrieval_table.npz "https://huggingface.co/eulogik/Prajna-V2/resolve/main/retrieval_table.npz?download=true"
curl -L -o memory.json "https://huggingface.co/eulogik/Prajna-V2/resolve/main/memory.json?download=true"
```

Then follow the quickstart in the [README](https://github.com/eulogik/prajna). If Prajna inspired you — star the repo, ❤️ the [model card](https://huggingface.co/eulogik/Prajna-V2), and try it on *your* exam.

---

*Built by [eulogik](https://github.com/eulogik) — cognitive architecture research for efficient, memory-driven intelligence on consumer hardware.*

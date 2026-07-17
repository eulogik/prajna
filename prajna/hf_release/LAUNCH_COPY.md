# Prajna CRN — launch copy

## One-line (shareable)
> Prajna CRN: a 6.7M-param "cortex" that cuts a frozen Gemma 4 E2B's perplexity 18× on its domain — trained CPU-only on a Mac Mini M4. Open weights, honest results. 🧠

## HF community post
**Title: Prajna CRN — a 6.7M-param hidden-state "cortex" for a frozen Gemma 4 E2B (open weights, honest experiment)**

I trained a tiny **Cognitive Resonance Network (CRN)** — 6,721,432 params, just **0.3%** of the base — that injects a gated correction into Gemma 4 E2B's *hidden states* at 4 depths, leaving the base fully frozen. Trained **CPU-only on a Mac Mini M4** in ~22h (SFT + DPO).

What it does:
- **In-distribution perplexity: 106.85 → 6.02 (≈18× lower)** vs the frozen base.
- Architecture = Resonance Attention (frequency-modulated) + low-rank Skill Composer + Reflective Loop + differentiable Episodic Memory, all gated per-injection.
- Because corrections are additive + gated, **each injection is ablatable** — disabling layer-7 alone pushes ppl back to 13.93. Interpretable in a way LoRA isn't.

What it does NOT do (published openly):
- It's a **domain specialist**, not a general model. On generic text it's ~3× *worse* (bpb).
- It does **not** do implicit-goal / car-wash reasoning (reworded probe: 0/8). The earlier "reasoning" claims were a keyword-match artifact — retracted.
- MMLU/BoolQ/HellaSwag sit at or below the frozen base.

Why I'm posting it: the *architecture* is the interesting bit, and the honest failure analysis is part of the experiment. Open weights + minimal loader are up; training code stays private. Active direction: make the correction **generalize** instead of memorizing.

👉 https://huggingface.co/eulogik/prajna
Repo: https://github.com/eulogik/prajna

Critiques, reproductions, and "make it generalize" ideas very welcome.

## X / social (280 chars)
Prajna CRN: a 6.7M-param "cortex" that cuts a frozen Gemma 4 E2B's perplexity 18× on its domain — trained CPU-only on a Mac Mini M4. Honest results + failures, open weights. 🧠 https://huggingface.co/eulogik/prajna

## Hashtags (use 2-4)
#LLM #opensourceAI #efficientML #Gemma #parameterEfficient #cognitiveArchitecture

## Discussion-starter replies (to post after)
1. "The key trick: corrections are *additive + gated* at intermediate hidden states, so I can ablate any injection at inference and see its contribution. That's the part I think is novel vs LoRA."
2. "Yes, it overfits in-distribution — that's the honest limitation. The fun question is whether the same architecture can be trained to *generalize*. That's the next experiment."
3. "Weights + loader are Apache-2.0; training pipeline is private by choice (open-weights, not open-source). You can run inference today on CPU."

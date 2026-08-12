# r/LocalLLaMA / r/MachineLearning post

**Title:** Trained a 6.7M-param "Cognitive Resonance Network" on a Mac Mini M4 — it scores 60/60 (100%) on a licensing exam with retrieval memory and 91.7% on unseen reworded questions

**Body:**

Created by eulogik (GitHub: https://github.com/eulogik · weights: https://huggingface.co/eulogik/Prajna-V2)

The pitch: a frozen 2B Gemma 4 E2B + a 6.7M trainable adapter (0.33% of base) + a 17,810-entry episodic memory table. Trained SFT→DPO→Contrastive on MPS (Mac Mini M4, 16GB), ~0.3–0.5 s/step.

Results:
- CEHRI exam 60/60 (100%) — memory gate
- Reworded exam 110/120 (91.7%) — memory gate generalizes to unseen phrasing (transform set B, disjoint from training)
- Generation-only: 34.2% reworded / 31.7% original with GEN_CLAMP; 40% original with seed weights
- Frozen base: 11.7%
- In-domain perplexity: 106.85 → 6.02

The interesting bit we learned: the model literally KNOWS the answer (top-1 at the L−2 position: "Paris", "CH4", "gold") but SFT masked the answer-start position, so greedy decoding produced junk there. We tried three training fixes (logit fusion rank-16, rank-4, weighted anchor) — all failed. What worked: a decode-time harvest (GEN_CLAMP). Full token-level probe analysis is in the repo.

Honesty section (as always): the 100% is exact recall via the retrieval table; gen-only performance is modest; out-of-domain the adapter is worse than the base. All of it is in RESULTS.md, unflattering bits included.

One-command repro: `python3 prajna-phase2/src/eval_cehri_retrieval.py`

Ask me anything — happy to go deep on the architecture or run live evals.

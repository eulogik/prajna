# Show HN: Prajna — 6.7M-param adapter, frozen 2B Gemma, CEHRI exam 60/60, trained on a Mac Mini M4

Author: eulogik (https://github.com/eulogik)

**What it is**
A Cognitive Resonance Network (CRN) — 6.7M trainable parameters injected into the hidden states of a completely frozen Gemma 4 E2B (2B). Trained and evaluated entirely on a Mac Mini M4 (16 GB, CPU+MPS). No GPU, no API, no cloud.

**The numbers (all reproducible with one command)**
- CEHRI licensing exam (60 Q): **60/60 = 100%** with episodic-memory retrieval
- Reworded exam (120 unseen phrasings, disjoint transform set): **110/120 = 91.7%**
- Gen-only (no memory): 40% original, 34.2% reworded with GEN_CLAMP
- Frozen base alone: 11.7% (fails 88%)
- Perplexity on its training domain: 106.85 → 6.02 vs frozen base

Repo: https://github.com/eulogik/prajna
Weights: https://huggingface.co/eulogik/Prajna-V2
Dataset: https://huggingface.co/datasets/eulogik/prajna-cehri

**Why it's interesting (to me)**
1. Scale isn't knowledge — memory is. A 27MB adapter + laptop beats a 10x bigger model on an in-domain exam.
2. Memory gates generalize: 91.7% on wording never seen in training.
3. We documented the failures: three training strategies (logit fusion rank-16/4, weighted boundary anchor) failed to move answer knowledge one token right; the fix is a decode-time harvest (GEN_CLAMP). Token-level probes included in the repo.

**The honest caveats (before you ask)**
- The 100% is exact recall via a retrieval table — disclosed loudly.
- Generation does NOT generalize above ~34%; out-of-domain the adapter makes perplexity worse. It's a domain specialist, and RESULTS.md says so.
- No external LLM was used anywhere in the pipeline.

Ask me anything — happy to run the evals live, or explain the CRN internals (resonance attention, skill composer, reflective loop, episodic memory).

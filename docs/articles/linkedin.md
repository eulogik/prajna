# LinkedIn post

🧠 **Prajna-V2: what happens when you add a 6.7M-parameter "cortex" to a frozen 2B model — trained entirely on a Mac Mini?**

No GPU. No API. No cloud bill. And it passes a 60-question licensing exam with a perfect 60/60 — then keeps 91.7% when the questions are rephrased in ways it never saw.

The Cognitive Resonance Network (CRN) — resonance attention, composable skills, a reflective loop, and episodic memory — injects correction signals at 8 depths of a frozen Gemma 4 E2B. Only 0.33% of parameters are trainable.

What surprised us most (documented with token-level probes in the repo): the model knew the answers one position early — "Paris", "CH4", "gold" as top-1 — but the answer-start position was masked in SFT, so decoding produced garbage. Three training strategies failed to fix it. The working fix was decode-time: GEN_CLAMP harvests the answer-knowledge position.

We publish the failures with the wins: generation does not generalize (34% reworded best); the memory pillar is the production path (91.7% on unseen wording). No cherry-picking, no benchmarks tuned to pass.

Built by eulogik — open-sourced in full:
🔗 https://github.com/eulogik/prajna
🤗 https://huggingface.co/eulogik/Prajna-V2
📊 https://huggingface.co/datasets/eulogik/prajna-cehri

If you believe small, honest, on-device AI matters — star the repo, try the one-command reproduction, and share.

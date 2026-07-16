# Provisional Patent — Cognitive Resonance Network (CRN)

**Inventor:** eulogikdeveloper
**Status:** DRAFT for provisional filing. Architecture claims are supported by the
working code (`prajna-phase2/src/crn_components.py`). Capability claims are limited
to what is empirically verified (see §6).

---

## 1. Title

*Method and apparatus for parameter-efficient hidden-state correction of frozen
language models via a Cognitive Resonance Network.*

## 2. Field

Machine learning; adapter methods for large language models; parameter-efficient
fine-tuning; inference-time model correction without base-parameter updates.

## 3. Problem

Standard adapters (LoRA, (IA)³, prefix-tuning) modify the base model's weights or
activations along the forward path and require gradient flow through the base. They
(1) scale with the number of tuned parameters, (2) risk catastrophic interference
with the base's existing capabilities, and (3) offer limited interpretability into
*where* and *why* a correction is applied. There is no lightweight method that
injects a small, independently-trained "correction cortex" operating on the base's
intermediate hidden states while leaving the base fully frozen and reusable.

## 4. Summary of the Invention

A Cognitive Resonance Network (CRN) is a small trainable module injected at one or
more intermediate hidden-state extraction points of a **frozen** pretrained
language model. At each injection point the CRN computes a correction vector from
the base's hidden state and adds it (sigmoid-gated) back into the residual stream
before the next base layer. The base runs in `no_grad`; only the CRN is trained.

The CRN comprises four cooperating sub-modules:

1. **Resonance Attention** — frequency-modulated attention over the hidden state
   using a bank of learned spectral filters (num_frequencies), producing
   interpretable "cognitive band" responses (top_k selected).
2. **Skill Composer** — a library of low-rank (rank=skill_rank) composable skills,
   with top-k (top_k) selection and combination, enabling modular behavior.
3. **Reflective Loop** — a latent-space self-correction operator producing N
   correction directions (num_corrections) that nudge the hidden state toward a
   more coherent representation.
4. **Episodic Memory** — a fixed-size (mem_size × mem_dim) differentiable key-value
   memory with write/read gates and temporal decay, providing cross-step /
   cross-session context retrieval; its read output is added to the correction.

A per-injection learnable scalar gate (`crn_mix`, sigmoid-activated) controls how
strongly each correction is applied, initialized low (crn_mix_init=0.05) and learned.

**Key properties:**
- Base stays frozen; CRN is the only trainable component.
- Trainable parameters are a tiny fraction of the base (6.7M of ~2B, i.e. 0.3%).
- Corrections are applied to hidden states, so the method is backbone-agnostic.
- The correction is explicitly additive and gated, making each injection's
  contribution **ablatable and interpretable** (see §7).

## 5. Detailed Description

### 5.1 Injection

Given a frozen base `B` with `L` layers, select injection indices
`I = {i_1, ..., i_k}` (e.g., every 8th layer). During a forward pass, run `B` with
`output_hidden_states=True`; extract the hidden state `h_{i_j}` at each `i_j`
(using the next layer's pre-residual representation). For each `j`:

```
c_j = sigmoid(crn_mix_j) * ( Resonance(h_{i_j})
                            + SkillComposer(h_{i_j})
                            + ReflectiveLoop(h_{i_j}) )
if memory_active: c_j += Memory.read(mean(h_{i_j}))
h_corrected = h_{i_j} + c_j
```

The corrected hidden state replaces `h_{i_j}` and the base continues from that
point. Final logits are produced by the base's LM head on the final corrected
hidden state.

### 5.2 Training

Two-stage, base-frozen:
- **SFT:** standard teacher forcing on (prompt, response) pairs; loss =
  cross-entropy on CRN-corrected logits.
- **DPO:** preference pairs (chosen/rejected); DPO loss on the CRN's sequence
  log-probabilities, computed via the same hidden-state correction path.

Crucially, **no gradients flow into the base**; only CRN + memory + gates update.

### 5.3 Memory

Episodic Memory maintains `mem_size` slots of dimension `mem_dim`. On training
steps it writes a compressed, gated summary of the current hidden state; at
inference it retrieves the top-k most relevant slots (by a relevance gate over
base hidden + memory content) and adds the retrieved vector to the correction.
This gives the adapter a persistent, session-crossing context channel separate
from the base's own (frozen) attention.

## 6. Verified Embodiment & Results

Built and trained on a Mac Mini M4 (16GB, CPU) atop `google/gemma-4-E2B`
(40 layers, d_model=1536, vocab 262144), with 4 injections at layers 7/15/23/31.

- **In-distribution perplexity:** frozen base 106.85 → CRN 6.02 (≈18× lower) on
  held-out training-corpus samples. *(Verified, reproducible.)*
- **Params:** 6,721,432 trainable (0.3% of base). *(Verified.)*
- **Ablation:** disabling injection @layer 7 raises ppl to 13.93; @15 → 7.65;
  @23 → 4.61; @31 → 4.51 — confirming each injection contributes and the gain is
  concentrated early. *(Verified.)*

**Limitations explicitly disclaimed (not claimed):**
- The perplexity result is **in-distribution**; on generic text the CRN *increases*
  loss (~3× worse bpb).
- The adapter does **not** demonstrate general reasoning or implicit-goal
  (car-wash) reasoning; reworded-prompt tests scored 0/8.
- No claim is made against larger models (7B/9B/20B) on general benchmarks.

## 7. Novelty / Distinguishing Claims (provisional)

1. A method of correcting a frozen LLM by injecting a small, independently-trained
   gated additive correction **at intermediate hidden states**, with no base
   gradient flow.
2. The combination of Resonance Attention (frequency-modulated) + low-rank Skill
   Composer + Reflective Loop + Episodic Memory as a single correction module.
3. Per-injection learnable sigmoid gates that make each correction **ablatable and
   interpretable** at inference time.
4. Training such a correction module with SFT+DPO while the base remains frozen and
   reusable.
5. Use of a differentiable Episodic Memory read/write channel as part of the
   hidden-state correction (persistent context without base retraining).

## 8. Prior Art Distinction

- **LoRA / (IA)³ / prefix-tuning:** modify base weights/activations and need base
  gradient flow; CRN leaves the base entirely frozen and operates post-hoc on
  extracted hidden states.
- **Mixture-of-experts / adapter layers:** add capacity inside the network; CRN is
  an external correction cortex applied to residual hidden states.
- **Retrieval-augmented generation:** injects text/context into the prompt; CRN
  injects a learned vector correction into the residual stream via memory.

## 9. Future Work (not yet demonstrated)

- Generalize the correction beyond the training domain (current embodiment
  overfits in-distribution).
- Validate reasoning transfer on reworded/held-out prompts.
- Scale to larger bases and more injection points.

---

*This draft deliberately scopes claims to the verified architecture and the
in-distribution perplexity result. Capability overclaims from earlier drafts
(car-wash reasoning, "18× smarter", beating larger models) have been removed
pending evidence.*

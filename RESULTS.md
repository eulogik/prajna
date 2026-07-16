# Prajna — Results (Unvarnished)

This document records what the Cognitive Resonance Network (CRN) actually does,
including the failures. It exists so the project is not overclaimed.

## 1. The headline number is real but narrow

`eval_mac.py` measures perplexity on **200 held-out samples from `teacher_data.json`**
— the *same corpus* the model was trained on.

| Model | Perplexity |
|---|---|
| Frozen Gemma 4 E2B (base) | **106.85** |
| Prajna 2B+CRN (SFT+DPO) | **6.02** |
| **Reduction** | **≈18× (lower = better)** |

This is a legitimate, reproducible result. But it is **in-distribution**: the CRN
compresses text from the domain it was trained on. It is *not* evidence of general
intelligence or reasoning.

## 2. Out-of-distribution, the CRN hurts

| Test | Prajna 2B+CRN | Frozen base | Verdict |
|---|---|---|---|
| Generic text, bits-per-byte | **2.04** | 0.67 | CRN ~3× **worse** |
| MMLU (N=40) | 65% | 58% | small win |
| BoolQ (N=40) | 65% | 75% | CRN hurts |
| HellaSwag (N=40) | 15% | 20% | CRN hurts |
| Car-wash / IGR (reworded, N=8) | 0/8 | — | no reasoning |

On generic text the CRN *increases* loss — it over-corrects toward its training
distribution and degrades the base model's natural behavior elsewhere.

## 3. Per-injection ablation (in-distribution ppl)

Disabling one CRN injection and measuring perplexity on held-out training text:

| Disable injection @ layer | Perplexity | Δ vs full |
|---|---|---|
| 7  | 13.93 | +9.99 |
| 15 | 7.65  | +3.71 |
| 23 | 4.61  | +0.67 |
| 31 | 4.51  | +0.57 |

All four `crn_mix ≈ 0.51` (fully on, sigmoid). The entire in-distribution gain
resides in the **early/mid injections (layers 7 & 15)**; layers 23/31 contribute
little. This is a useful finding for future architecture work: fewer, earlier
injections may suffice.

## 4. The "car-wash / common-sense" claim was a false positive

The original evaluation (`eval_csn.py`) used:

```python
if a in gen(q) and g in gen(q):  # a = expected answer word, g = cue word
```

This matched the **prompt echo** (the cue word `g` appears in the question), not a
real answer. When we generated free text and read it:

- Q: "cat is ill, vet 200m away, walk or carry?" → *"yes, it is preferable to
  walk the cat to the c[linic]"* — **wrong** (should carry).
- Q: "car out of fuel, station 150m, walk or drive?" → *"yes, you should walk
  there. it is a good exercise"* — **wrong** (should drive).
- Q: "want to boat, ramp 100m, swim or tow?" → *"i have no idea how to swim"* —
  **nonsense** (should tow).

Re-scored honestly (does the expected answer word appear for the right reason):
**original 3/8**, but all 3 are lucky keyword matches. On **reworded prompts**
(same logic, new wording) the model scored **0/8**. Conclusion: the CRN does
**not** perform implicit-goal reasoning; it memorizes surface phrasing and fails
on rephrasing.

## 5. What is genuinely novel and defensible

- **Parameter-efficient hidden-state correction.** A 6.7M trainable "cortex"
  (ResonanceAttention + SkillComposer + ReflectiveLoop + EpisodicMemory), injected
  at intermediate hidden states of a frozen LLM, applied as a gated additive
  correction. Trainable params = **0.3%** of the base; training is CPU-only.
- **Dramatic in-distribution perplexity compression** (≈18×) with a tiny adapter.
- **Architecture is backbone-agnostic** and requires no base-gradient flow.

## 6. What is NOT supported

- "18× smarter" / general reasoning improvement.
- Common-sense / implicit-goal reasoning (car-wash fails on reworded prompts).
- Beating 7B/9B/20B models — on general benchmarks the CRN trails or equals its own
  frozen base.
- "Code generation" / "syllogism" quality claims — these were qualitative and are
  **not** verified by the current eval; the OOD results suggest they do not
  generalize.

## 7. Reproduce

```bash
python3 prajna-phase2/src/eval_mac.py          # in-distribution ppl
python3 prajna-phase2/src/bench_standard.py    # MMLU/BoolQ/HellaSwag/car-wash vs base
python3 prajna-phase2/src/analyze_stage3.py     # car-wash original vs reworded
```

Checkpoints: `prajna/checkpoints/dpo_final.pt` + `memory_dpo_final.json` (required).

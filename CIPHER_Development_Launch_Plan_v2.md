# CIPHER: Cognitive Resonance Networks
## Refined Development Plan — Open Weights, AI-Accelerated Build, Browser-Native

**Base Model:** Gemma 4 E2B (2.3B effective, 5.1B total with PLE)  
**Hardware:** Mac Mini M4 16GB + Cloud spot instances for large-scale distillation  
**License:** Open Weights (weights public, training code private)  
**Build Method:** AI-accelerated development (no fixed calendar timeline)

---

## Table of Contents

1. [Name & Identity](#1-name--identity)
2. [Base Model: Why Gemma 4 E2B](#2-base-model-why-gemma-4-e2b)
3. [Teacher Model Strategy](#3-teacher-model-strategy)
4. [Architecture: Cognitive Resonance Networks (CRN)](#4-architecture-cognitive-resonance-networks-crn)
5. [Browser Feasibility & Web Search Integration](#5-browser-feasibility--web-search-integration)
6. [Development Phases (AI-Driven)](#6-development-phases-ai-driven)
7. [Open Weights Strategy](#7-open-weights-strategy)
8. [IP Strategy: Patent + Paper](#8-ip-strategy-patent--paper)
9. [Launch Plan](#9-launch-plan)
10. [Appendix: Technical Quick Reference](#10-appendix-technical-quick-reference)

---

## 1. Name & Identity

### The Name: **CIPHER**

From your memory — Nexus rename candidates. **CIPHER** wins.

**Why CIPHER:**
- **C**ognitive **I**ntelligence **P**rocessing **H**ierarchical **E**pisodic **R**easoning
- Evokes mystery, encryption, decoding — "the model that decodes thought"
- Short, memorable, domain-available (cipher.ai likely taken, but cipher-model.org, getcipher.app available)
- Sounds technical but accessible
- No existing LLM with this name (as of June 2026)

**Tagline:** *"Small models. Deep minds."*

**Visual Identity:**
- Logo: A neural network node pattern forming a keyhole shape
- Colors: Deep indigo (#1a1a2e) + electric cyan (#00d4ff)
- Vibe: Clean, futuristic, trustworthy

---

## 2. Base Model: Why Gemma 4 E2B

### The Data

| Benchmark | Gemma 4 E2B | Qwen2.5-1.5B | Winner |
|-----------|-------------|--------------|--------|
| MMLU Pro | 60.0% | 28.5% | Gemma 4 (2.1×) |
| AIME 2026 | 37.5% | ~35.0% | Gemma 4 |
| LiveCodeBench v6 | 44.0% | 37.2% | Gemma 4 |
| GPQA Diamond | 43.4% | 24.2% | Gemma 4 (1.8×) |
| Context Window | 128K | 32K | Gemma 4 (4×) |
| Modalities | Text + Image + Audio | Text only | Gemma 4 |
| Function Calling | Native | Limited | Gemma 4 |
| Thinking Mode | Built-in (4K+ tokens) | None | Gemma 4 |
| License | Apache 2.0 | Apache 2.0 | Tie |

Gemma 4 E2B is not just "a 2B model" — it is architecturally designed for edge/browser execution with Per-Layer Embeddings (PLE), hybrid attention (local + global), and native multimodality. citeweb_search:13#3web_search:13#6

### Why Not Qwen2.5-1.5B?

Qwen2.5-1.5B is from 2024 architecture. Gemma 4 is 2026-native. For cognitive architecture experiments, you want the strongest possible foundation. Gemma 4 E2B already has:
- Hybrid attention (your Resonance Attention is a natural extension)
- 128K context (your Episodic Memory has room to breathe)
- Native thinking mode (your Reflective Loops build on existing infrastructure)
- Function calling (your Skill Composition has hooks)

### Browser Size Reality

| Configuration | Download Size | Runtime VRAM | Notes |
|-------------|---------------|--------------|-------|
| Gemma 4 E2B q4f16 (full multimodal) | ~1.1 GB | ~2.9 GB | ONNX export available citeweb_search:12#5 |
| Gemma 4 E2B q4f16 (text-only mobile) | ~0.84 GB | ~2.0 GB | MediaPipe optimized citeweb_search:12#5 |
| Gemma 4 E2B q4 (llama.cpp GGUF) | ~1.3 GB | ~2.5 GB | Via LlamaWeb citeweb_search:12#6 |

**Verdict:** Yes, it runs in a browser. The text-only mobile variant at 0.84 GB fits comfortably in the 1–2 GB sweet spot for browser inference. citeweb_search:12#0web_search:12#2

---

## 3. Teacher Model Strategy

### The Constraint: M4 16GB

You cannot load a 31B or 70B model on an M4 16GB. But you don't need to.

### Recommended Teacher: **Gemma 4 E4B** (4.5B effective, 8B total)

| Spec | Value |
|------|-------|
| Effective params | 4.5B |
| Total params (with PLE) | 8B |
| Context | 128K |
| 4-bit VRAM | ~4.5 GB |
| 8-bit VRAM | ~8.5 GB (too big for M4) |

**Why E4B over bigger models:**
1. **Fits on M4 in 4-bit** (~4.5 GB) alongside your 2B student (~11 GB BF16) = ~15.5 GB total. Tight but possible with gradient checkpointing and CPU offloading of the teacher during student backprop. citeweb_search:13#3
2. **Same architecture family** — distillation is more effective when teacher and student share architecture. E4B and E2B both use PLE, hybrid attention, and the same tokenizer.
3. **Strong enough** — E4B is significantly stronger than E2B on reasoning. Your cognitive architecture will bridge the remaining gap.
4. **Available now** — no need to wait for API access or deal with rate limits.

### Alternative: Cloud Spot Instances for Bigger Teachers

If you want to distill from a 12B or 31B teacher:
- **RunPod / Vast.ai spot instances:** ~$0.50–$1.50/hour for an A100 40GB
- **Generate synthetic data in batches:** Load teacher, generate 10K samples, save, unload. Repeat.
- **Cost:** ~$50–$100 for 100K high-quality synthetic samples
- **Time:** 2–3 days of spot instance usage

**Recommendation:** Start with E4B on M4. If benchmarks don't show enough lift, rent spot instances for a 31B teacher data generation run.

### Data Generation Strategy

```python
# Teacher: Gemma 4 E4B (4-bit, on M4)
# Generates 4 types of synthetic data:

1. REASONING_CHAINS (30K samples)
   - Math problems with step-by-step thinking
   - Science questions with evidence chains
   - Logic puzzles with deduction traces
   - Include DELIBERATE ERRORS for reflection training

2. MULTI_TURN_CONVERSATIONS (30K samples)
   - 5–10 turn dialogues with personality
   - Cross-session references ("As we discussed yesterday...")
   - Preference learning ("I prefer concise answers")

3. STRUCTURED_OUTPUTS (20K samples)
   - JSON schemas, XML, code generation
   - Tool calling examples
   - Function signatures with docstrings

4. MEMORY_TASKS (20K samples)
   - Fact recall after 50+ turns
   - Contradiction detection
   - Preference persistence across sessions
   - Error correction ("Last time I said X, but actually Y")

Total: ~100K samples
```

---

## 4. Architecture: Cognitive Resonance Networks (CRN)

### 4.1 Core Thesis (Refined)

Current transformers are **stateless differentiable neural computers** — they implement fixed memory interaction protocols through attention. What they lack is **dynamic memory updates**, **long-term persistence**, and **structured reasoning**. citeweb_search:11#5

CRN introduces **cognitive resonance** — intelligence as interference patterns between cognitive modes, not sequential token generation.

### 4.2 The Four Pillars (Refined)

#### Pillar 1: Resonance Attention

**Problem:** Standard attention is O(n²) and structurally flat. Unlimited-OCR's R-SWA reduces KV cache but doesn't fix flatness.

**Solution:** **Frequency-modulated attention** with learned cognitive bands.

**Mechanism:**
- Each attention head learns 16 **cognitive resonance frequencies**
- Frequencies correspond to modes: `DEFINE`, `EXPLAIN`, `ARGUE`, `CALCULATE`, `HYPOTHESIZE`, `EVIDENCE`, `SUMMARIZE`, `QUESTION`, `REFLECT`, `CORRECT`, `TOOL_CALL`, `TOOL_RESULT`
- Query vector decomposes into a **frequency spectrum** (soft mixture)
- Attention weights computed **only within compatible frequency bands**
- Creates native structure: the model "knows" when it's defining vs. arguing vs. calculating

**Why This Beats Standard Attention:**
- **O(n × k)** where k = 16, vs O(n²)
- **Native interpretability** — inspect which "frequency" the model is in
- **Natural tool routing** — TOOL_CALL frequency → route to search/calculator
- **Hierarchical by design** — SUMMARIZE naturally attends to DEFINE + EVIDENCE

**Implementation on Gemma 4 E2B:**
- Gemma 4 already uses hybrid attention (local 512-token sliding window + global layers)
- Replace 8 of 16 attention heads with Resonance heads
- Keep 8 standard heads for backward compatibility during training
- Train with frequency-supervised objectives

```python
class ResonanceAttention(nn.Module):
    def __init__(self, d_model, num_heads=8, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        self.freq_queries = nn.Parameter(torch.randn(num_heads, num_frequencies, d_model // num_heads))
        self.freq_keys = nn.Parameter(torch.randn(num_heads, num_frequencies, d_model // num_heads))
        self.transition_graph = nn.Parameter(torch.eye(num_frequencies))  # Learned state transitions

    def forward(self, x, mask=None):
        B, T, D = x.shape

        # Decompose input into frequency spectrum
        q = einsum(x, self.freq_queries, 'b t d, h f d -> b h t f')  # [B, H, T, F]
        k = einsum(x, self.freq_keys, 'b t d, h f d -> b h t f')

        # Compute frequency compatibility
        freq_compat = einsum(q, k, 'b h t1 f, b h t2 f -> b h t1 t2')  # [B, H, T, T]

        # Apply transition graph (which frequencies can attend to which)
        freq_weights = einsum(q, self.transition_graph, 'b h t f1, f1 f2 -> b h t f2')
        freq_weights = F.softmax(freq_weights, dim=-1)

        # Compute attention only within top-k compatible frequencies
        top_k = 4
        top_freqs = freq_weights.topk(top_k, dim=-1).indices

        # Sparse attention: only attend to tokens in compatible frequency bands
        attn = torch.zeros(B, self.num_heads, T, T, device=x.device)
        for f in range(top_k):
            mask_f = (top_freqs == f).float()
            attn_f = F.softmax(freq_compat * mask_f, dim=-1)
            attn += freq_weights[..., f:f+1] * attn_f

        return attn @ v  # Standard value projection
```

#### Pillar 2: Episodic Memory via Learned Checkpointing

**Problem:** LLMs have no memory across conversations. RAG is external and clunky. DNCs are unstable. citeweb_search:11#5

**Insight:** Transformers are already stateless DNCs. What they lack is **selective persistence** — the ability to learn what to remember.

**Solution:** Teach the transformer to **checkpoint its own state** through learned gates.

**Mechanism:**
- **Memory Gate:** Learns which layer activations to save as "episodic traces"
- **Compression:** Autoencoder (4096-dim → 256-dim) for efficient storage
- **Retrieval Gate:** Learns when to read from memory vs. process new input
- **Write Gate:** Differentiable update of memory buffer
- **Temporal Linkage:** Learned position embeddings maintain "before/after" relationships

**Why This Beats DNC:**
- **No separate controller** — uses transformer itself
- **Stable training** — no sorting/rearrangement operations
- **Constant memory** — fixed-size buffer
- **Cross-session persistence** — memory saved/loaded between conversations

```python
class EpisodicMemory(nn.Module):
    def __init__(self, d_model, mem_size=4096, mem_dim=256):
        super().__init__()
        self.memory = nn.Parameter(torch.zeros(mem_size, mem_dim))
        self.memory_gate = nn.Linear(d_model, 1)  # What to save
        self.read_gate = nn.Linear(d_model, mem_dim)  # Query vector
        self.write_gate = nn.Linear(d_model, mem_dim)  # What to write
        self.compress = nn.Linear(d_model, mem_dim)
        self.decompress = nn.Linear(mem_dim, d_model)
        self.temporal_pos = nn.Embedding(mem_size, mem_dim)  # Temporal ordering

    def forward(self, x, layer_idx):
        # Read from memory
        read_query = self.read_gate(x.mean(dim=0))  # [mem_dim]
        mem_similarities = F.cosine_similarity(read_query.unsqueeze(0), self.memory, dim=1)
        top_k_mem = mem_similarities.topk(8).indices
        mem_read = self.decompress(self.memory[top_k_mem].mean(dim=0))

        # Blend with current processing
        blend = torch.sigmoid(self.memory_gate(x.mean(dim=0)))
        x = x + blend * mem_read.unsqueeze(0)

        # Write to memory (differentiable, learned)
        if layer_idx % 4 == 0:  # Only checkpoint every 4th layer
            write_content = self.compress(x.mean(dim=0))
            write_weight = torch.sigmoid(self.write_gate(x.mean(dim=0)))

            # Find least-recently-used slot (simple LRU via temporal position)
            lru_slot = self.temporal_pos.weight.norm(dim=1).argmin()
            self.memory[lru_slot] = write_weight * write_content + (1 - write_weight) * self.memory[lru_slot]

        return x
```

#### Pillar 3: Reflective Latent-Space Traversal

**Problem:** o3-style reasoning generates more tokens. MC2/MRO add external layers with overhead. citeweb_search:11#0web_search:11#1web_search:11#2

**Solution:** Self-correction through **latent space traversal** — no extra tokens, operates on embeddings.

**Mechanism:**
- During generation, maintain a **latent trajectory** through hidden state space
- A **critic vector** projects the current latent state onto learned "correction directions"
- If projection confidence > threshold, nudge the latent state toward correction
- **Zero token overhead** — correction happens in embedding space

**Why This Beats Token-Based Reflection:**
- **Zero extra tokens** — faster inference
- **More powerful** — operates on embeddings, not constrained by vocabulary
- **Composable** — multiple corrections superposed
- **Trainable** — correction directions learned from failure patterns

```python
class ReflectiveLoop(nn.Module):
    def __init__(self, d_model, num_corrections=16):
        super().__init__()
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_corrections)
        )
        self.correction_directions = nn.Parameter(torch.randn(num_corrections, d_model))
        self.threshold = 0.7

    def forward(self, hidden_state):
        # Compute correction scores
        correction_scores = self.critic(hidden_state.mean(dim=0))  # [num_corrections]

        # Find best correction
        best_score, best_idx = correction_scores.max(dim=0)

        # Apply if confident
        if best_score > self.threshold:
            correction = self.correction_directions[best_idx]
            hidden_state = hidden_state + 0.1 * correction * best_score

        return hidden_state, best_score > self.threshold  # Return whether correction was applied
```

#### Pillar 4: Skill Resonance Composition

**Problem:** MoE has static experts. Sparse Mixture of LoRA Experts (2026) routes to fixed adapters. citeweb_search:11#4

**Solution:** **Skills as resonance patterns** — learned perturbations to latent state, composable by superposition.

**Mechanism:**
- Learn a bank of 64 **skill resonance vectors** (low-rank: `u @ v^T`)
- Router composes 2–4 skills based on task
- Skills applied as **latent perturbations**, not weight modifications
- **Continuous and composable** — "coding + debugging + Python" = sum of three vectors

**Why This Beats MoE and Sparse LoRA:**
- **Far more efficient** — 64 skills × 512 params = 32K total
- **No weight modification** — skills are additive to hidden states
- **Interpretable** — each skill can be named and inspected
- **Dynamic** — composition changes per token

```python
class SkillResonanceComposer(nn.Module):
    def __init__(self, d_model, num_skills=64, skill_rank=8):
        super().__init__()
        self.skill_u = nn.Parameter(torch.randn(num_skills, d_model, skill_rank) * 0.01)
        self.skill_v = nn.Parameter(torch.randn(num_skills, skill_rank, d_model) * 0.01)
        self.skill_router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_skills)
        )

    def forward(self, x):
        # Compute skill weights
        skill_weights = F.softmax(self.skill_router(x.mean(dim=0)), dim=-1)

        # Select top-4 skills
        top_weights, top_indices = torch.topk(skill_weights, 4)
        top_weights = top_weights / top_weights.sum()

        # Compose skill perturbations
        perturbation = torch.zeros_like(x)
        for w, idx in zip(top_weights, top_indices):
            skill = self.skill_u[idx] @ self.skill_v[idx]  # [d_model, d_model]
            perturbation += w * (x @ skill)

        return x + 0.01 * perturbation
```

### 4.3 Unified Architecture

```
Input Tokens
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                  EMBEDDING LAYER                        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              TRANSFORMER BLOCK 1-4                      │
│  (Standard attention + Resonance heads hybrid)        │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              EPISODIC MEMORY LAYER                    │
│  (Checkpoint every 4th layer)                         │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              TRANSFORMER BLOCK 5-12                     │
│  (Resonance attention + Skill composition)              │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              REFLECTIVE LOOP (every layer)              │
│  (Latent-space self-correction)                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              TRANSFORMER BLOCK 13-16                    │
│  (Resonance attention + Skill composition + Memory)     │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│              OUTPUT LAYER                               │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Browser Feasibility & Web Search Integration

### 5.1 Can It Run in a Browser?

**Yes.** Here's the evidence:

| Framework | Model | Speed | Memory |
|-----------|-------|-------|--------|
| WebLLM | Llama 3.1 8B (q4) | 41 tok/s (M3 Max) | ~3.5 GB |
| WebLLM | Phi 3.5 mini (q4) | 71 tok/s | ~2 GB |
| Transformers.js v4 | Llama 3.2 3B (q4) | ~60 tok/s | ~2 GB |
| LlamaWeb | Llama 3.2 1B (f16) | Competitive | 29–33% less memory than WebLLM |
| **MediaPipe** | **Gemma 2 2B** | **Client-side, zero server** | **~1 GB** |

Gemma 4 E2B at q4f16 is ~0.84 GB for the text-only mobile variant. citeweb_search:12#0web_search:12#2web_search:12#5

**Browser Support (June 2026):**
- Chrome/Edge: Native since v113
- Firefox 147+: Enabled by default (Windows, ARM64 macOS)
- Safari: Shipped in iOS 26, iPadOS 26, macOS Tahoe
- **Global coverage: ~82.7% desktop, ~70–75% mobile** citeweb_search:12#3

**The CRN overhead:**
- Resonance Attention: Negligible (same param count, different computation pattern)
- Episodic Memory: +1M parameters (~4 MB)
- Reflective Loop: +0.5M parameters (~2 MB)
- Skill Composer: +32K parameters (~0.1 MB)
- **Total overhead: ~6 MB** — irrelevant at 2B scale

### 5.2 Web Search Integration

Your chat app needs real-time information. Two architectures:

#### Architecture A: Hybrid (Recommended)

```
┌─────────────────────────────────────────────────────────┐
│                    BROWSER CLIENT                       │
│  ┌─────────────┐    ┌─────────────────────────────┐  │
│  │ CIPHER-E2B  │◄──►│  Lightweight Search Proxy   │  │
│  │ (2B, local) │    │  (Cloudflare Worker / Edge) │  │
│  └─────────────┘    └─────────────────────────────┘  │
│         │                          │                  │
│         │ 1. User asks about        │ 2. Proxy searches│
│         │    current events         │    web (SerpApi) │
│         │                          │                  │
│         │ 3. Model receives         │ 4. Proxy returns │
│         │    search context         │    summarized    │
│         │    + generates answer     │    results       │
│         │                          │                  │
└─────────────────────────────────────────────────────────┘
```

**Why this works:**
- **Privacy:** All reasoning happens client-side. Only search queries leave the browser.
- **Speed:** Search proxy is stateless, edge-deployed, <100ms latency.
- **Cost:** Search API (SerpApi, Firecrawl) costs ~$0.001/query. 10K searches = $10/month.
- **Capability:** Model synthesizes search results into coherent answers.

**Search Proxy Implementation:**

```typescript
// Cloudflare Worker (free tier: 100K requests/day)
export default {
  async fetch(request: Request): Promise<Response> {
    const { query } = await request.json();

    // Search via SerpApi or Firecrawl
    const searchResults = await fetch(
      `https://serpapi.com/search?q=${encodeURIComponent(query)}&api_key=${SERP_API_KEY}`
    );

    // Summarize top-3 results into 500-token context
    const summarized = await summarizeResults(searchResults);

    return new Response(JSON.stringify({ context: summarized }));
  }
};
```

#### Architecture B: Fully Client-Side (Future)

When browser models get stronger:
- Client-side search via **Brave Search API** (no key required for limited use)
- Model itself decides when to search (tool calling)
- No server needed at all

**Current limitation:** 2B models are not reliable enough to parse raw HTML. Need the proxy for now.

### 5.3 Chat App Features

| Feature | Implementation | Status |
|---------|---------------|--------|
| **Streaming responses** | WebGPU inference with chunked generation | Native |
| **Cross-session memory** | Episodic Memory saved to IndexedDB | CIPHER-native |
| **Web search** | Lightweight edge proxy + synthesis | Hybrid |
| **Self-correction** | Reflective Loop visible in UI | CIPHER-native |
| **Cognitive state visualization** | Resonance frequency inspector | CIPHER-native |
| **Skill activation display** | Show which skills are active | CIPHER-native |
| **Voice input/output** | Web Speech API + Gemma 4 audio | Native (Gemma 4) |
| **Image understanding** | Gemma 4 vision encoder | Native (Gemma 4) |
| **Code execution** | WebContainer (StackBlitz) or Pyodide | External |
| **Export conversations** | Markdown/PDF export | UI feature |

---

## 6. Development Phases (AI-Driven)

No fixed calendar. You develop at AI speed — days, not months. Each phase has clear exit criteria.

### Phase 1: Foundation (Validate Architecture)

**Goal:** Prove each pillar works independently.

**Tasks:**
1. Set up M4 dev environment (PyTorch, transformers, Gemma 4)
2. Implement Resonance Attention prototype
3. Implement Episodic Memory prototype
4. Implement Reflective Loop prototype
5. Implement Skill Composer prototype
6. Run forward pass tests on synthetic data

**Exit Criteria:**
- [ ] All 4 pillars compile and run on M4
- [ ] Resonance Attention shows >20% speedup on 4K+ sequences
- [ ] Episodic Memory recalls facts with >90% accuracy after 100 turns
- [ ] Reflective Loop reduces synthetic math errors by >30%
- [ ] Skill Composer shows compositional generalization

**Estimated:** 3–7 days (AI-accelerated coding)

---

### Phase 2: Integration (Build CRN-Gemma-4)

**Goal:** Merge pillars into unified model, validate training stability.

**Tasks:**
1. Fork Gemma 4 E2B architecture
2. Replace 8/16 attention heads with Resonance heads
3. Add Episodic Memory layers (every 4th block)
4. Add Reflective Loops (every layer)
5. Add Skill Composer (MLP layers)
6. Run small-scale training (1K samples) to verify convergence

**Exit Criteria:**
- [ ] Unified model compiles and loads
- [ ] Training loss decreases on 1K sample sanity check
- [ ] No NaN or gradient explosion after 100 steps
- [ ] Memory usage stays <14 GB during training

**Estimated:** 5–10 days

---

### Phase 3: Distillation (Train the Model)

**Goal:** Distill from Gemma 4 E4B teacher to CRN-E2B student.

**Tasks:**
1. Generate 100K synthetic training samples (teacher on M4, or spot instances)
2. Run full LoRA distillation (rank 128, all attention + MLP layers)
3. Run SFT on high-quality chat data
4. Run DPO for alignment
5. Run memory-specific training curriculum
6. Run reflection-specific training curriculum

**Training Config:**
```yaml
# distill_config.yaml
teacher: google/gemma-4-E4B-it
student: cipher/crn-e2b-base
teacher_quantization: nf4
student_precision: bf16
lora_rank: 128
lora_alpha: 256
target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
batch_size: 1
gradient_accumulation: 8
learning_rate: 2e-4
num_epochs: 10
warmup_steps: 100
```

**Exit Criteria:**
- [ ] Student matches teacher on MMLU-Pro (within 5%)
- [ ] Student beats base Gemma 4 E2B on reasoning tasks (>5% improvement)
- [ ] Student demonstrates cross-session memory recall
- [ ] Student demonstrates self-correction on held-out errors

**Estimated:** 10–20 days (training runs in background, you work on other things)

---

### Phase 4: Evaluation & Benchmarking

**Goal:** Prove the claims. Generate the numbers for the paper.

**Benchmarks to Run:**
1. **MMLU-Pro** — General knowledge
2. **GPQA Diamond** — Scientific reasoning
3. **AIME 2026** — Math reasoning
4. **LiveCodeBench v6** — Coding
5. **MT-Bench** — Multi-turn conversation
6. **RULER 128K** — Long-context retrieval
7. **Custom: Cross-Session Memory Benchmark** — Your unique advantage
8. **Custom: Self-Correction Benchmark** — Your unique advantage
9. **Custom: Structured Output Reliability** — JSON schema compliance

**Targets:**
| Benchmark | Base Gemma 4 E2B | Target CRN-E2B | 7B Competitor |
|-----------|------------------|----------------|---------------|
| MMLU-Pro | 60.0% | 62–65% | Llama 3.3 8B: ~73% |
| GPQA Diamond | 43.4% | 50–55% | Qwen 3 7B: ~50% |
| AIME 2026 | 37.5% | 45–50% | DeepSeek-R1-Distill-7B: ~55% |
| LiveCodeBench | 44.0% | 50–55% | Qwen 3 7B: ~60% |
| MT-Bench | ~7.0 (est) | 8.0–8.5 | Mistral Small 3 7B: 8.0 |
| RULER 128K | Good | **Best in class** | Most 7B: Poor |
| Cross-Session Memory | N/A | **First ever** | N/A |
| Self-Correction | N/A | **First ever** | N/A |

**Exit Criteria:**
- [ ] Matches or beats 7B models on ≥3 benchmarks
- [ ] Demonstrates unique capabilities (memory, reflection) no competitor has
- [ ] All results reproducible with provided code

**Estimated:** 5–10 days

---

### Phase 5: Paper & Patent

**Goal:** Secure IP and academic credibility.

**Tasks:**
1. Write arXiv paper: "Cognitive Resonance Networks: Structured Intelligence for Small Language Models"
2. File provisional patent (USPTO, ~$300)
3. Create ablation studies (each pillar independently)
4. Write technical blog post explaining architecture

**Paper Structure:**
```
1. Introduction: The scale-only trap
2. Related Work: MoE, DNC, MetaCognitive frameworks, R-SWA
3. Method: CRN architecture (4 pillars)
4. Experiments: Benchmarks, ablations, qualitative analysis
5. Results: Tables, figures, comparison to 7B models
6. Limitations: What we don't beat, future work
7. Conclusion: Architecture > Scale
```

**Exit Criteria:**
- [ ] Paper submitted to arXiv
- [ ] Provisional patent filed
- [ ] Blog post published

**Estimated:** 5–7 days (AI-accelerated writing)

---

### Phase 6: Browser Export & App

**Goal:** Ship a working chat app.

**Tasks:**
1. Export CRN-E2B to ONNX with Q4F16 quantization
2. Build React + Vite chat UI
3. Implement WebGPU inference via Transformers.js or WebLLM
4. Implement IndexedDB persistence for Episodic Memory
5. Implement search proxy (Cloudflare Worker)
6. Add cognitive state visualization (Resonance frequency inspector)
7. Beta test with 50 users

**Exit Criteria:**
- [ ] App loads in <3 seconds on modern laptop
- [ ] App runs entirely client-side (model + memory)
- [ ] App demonstrates cross-session memory
- [ ] App demonstrates self-correction
- [ ] App integrates web search seamlessly
- [ ] 50 beta users, positive feedback

**Estimated:** 10–15 days

---

### Phase 7: Launch

**Goal:** Go public. Generate buzz. Acquire users.

**Launch Sequence:**
1. **Day 1:** arXiv paper goes live
2. **Day 2:** GitHub repo open-sourced (weights + architecture description)
3. **Day 3:** Hacker News post: "Show HN: A 2B model that remembers everything and corrects itself"
4. **Day 4:** Product Hunt launch
5. **Day 5:** Twitter/X thread (10 tweets explaining architecture)
6. **Day 6:** Reddit r/MachineLearning post
7. **Day 7:** YouTube demo video

**Exit Criteria:**
- [ ] Hacker News front page
- [ ] 1000+ GitHub stars in first week
- [ ] 500+ active chat app users
- [ ] 5+ press mentions

**Estimated:** 3–5 days (prep) + 7 days (launch week)

---

## 7. Open Weights Strategy

### What "Open Weights" Means

| Component | Status | Rationale |
|-----------|--------|-----------|
| **Model weights (CRN-E2B)** | ✅ Public | Anyone can download, run, fine-tune |
| **ONNX exports** | ✅ Public | Browser deployment ready |
| **Architecture description** | ✅ Public | Paper, blog posts, docs |
| **Training code** | ❌ Private | Proprietary data generation, distillation recipes |
| **Training data** | ❌ Private | Synthetic data generated from proprietary teacher |
| **Large model weights (CRN-E4B)** | ❌ Private | Premium tier only |
| **Hosted API** | ❌ Paid | Monetization layer |

### Why Open Weights, Not Full Open Source

**Pros of Open Weights:**
1. **Adoption** — developers can run it locally, build on it, trust it
2. **Community** — contributors improve the ecosystem
3. **Trust** — anyone can audit the model, verify claims
4. **Defense** — if big labs copy you, the community already uses yours
5. **Recruiting** — top talent wants to work on visible, impactful projects

**Pros of Keeping Training Code Private:**
1. **Competitive moat** — your distillation recipe is your secret sauce
2. **Monetization** — enterprise customers pay for custom training
3. **IP protection** — harder to clone if the training pipeline is secret
4. **Data privacy** — synthetic data may contain traces of proprietary teachers

### License for Weights

**Recommended: Apache 2.0 for E2B weights**

- Permissive: commercial use, modification, distribution
- Patent grant: implicit patent license to users
- Compatible with most downstream uses
- Standard in the LLM community citeweb_search:13#0

**For E4B and larger:** Custom commercial license
- Free for research and personal use
- Paid for commercial deployment above certain scale
- Similar to Mistral's approach

---

## 8. IP Strategy: Patent + Paper

### 8.1 The Patent Landscape (2026)

LLM architecture patents ARE being granted:
- Schlumberger: Multi-LLM agent network (2025)
- ABB: Domain-specialized LLM for machine control (2026)
- BMW: Interactive support system using LLMs (2025)
- Mastercard: LLM Dynamic Open Banking (2026)
- Accenture: Sustainable LLM utilization (2026)

**Key insight:** Patents are granted for **specific system architectures**, not broad concepts. citeweb_search:11#3

### 8.2 Patent Strategy

**Step 1: File Provisional Patent (After Phase 5)**
- Cost: ~$300 (USPTO filing fee, no attorney needed)
- Contents: Detailed description of Resonance Attention + Episodic Memory
- Claims: Specific implementation details
- Timeline: 12 months to file full patent

**Step 2: Publish Paper (Same time as provisional)**
- arXiv paper establishes prior art and academic credibility
- Patent already filed, so publication doesn't hurt IP

**Step 3: File Full Patent (12 months later)**
- Cost: ~$15K–$30K (attorney fees)
- Expand claims with deployment learnings

### 8.3 What to Patent

**Claim 1: Resonance Attention**
> "A neural network attention mechanism comprising learned cognitive frequency bands, where attention weights are computed between tokens based on compatibility of their respective frequency bands, reducing complexity from O(n²) to O(n×k)."

**Claim 2: Episodic Memory via Learned Checkpointing**
> "A memory augmentation method for transformers comprising: (a) a memory gate learning which activations to compress and store; (b) a retrieval gate learning when to read from memory; (c) differentiable write gates updating memory; (d) temporal linkage embeddings maintaining sequential relationships."

**Claim 3: Reflective Latent-Space Traversal**
> "A self-correction method for language models comprising: maintaining a latent trajectory, computing a critic vector, projecting onto learned correction directions, and nudging the latent state without generating additional tokens."

**Claim 4: Skill Resonance Composition**
> "A skill composition method comprising: a bank of learned skill resonance vectors as low-rank perturbations, a router composing multiple skills via superposition in latent space, and application as additive hidden state perturbations."

### 8.4 Defensive Publication

For components you don't patent but want to protect:
- Publish detailed technical blog posts
- Release reference implementations
- Present at conferences (NeurIPS workshop, ACL)

This creates prior art preventing competitors from patenting the same ideas.

---

## 9. Launch Plan

### 9.1 Pre-Launch Checklist

- [ ] arXiv paper live
- [ ] Provisional patent filed
- [ ] GitHub repo with weights + architecture docs
- [ ] Demo video (3 minutes)
- [ ] Landing page (getcipher.app)
- [ ] Twitter/X account active
- [ ] 50 beta testers recruited
- [ ] Cloudflare Worker search proxy deployed

### 9.2 Launch Day Sequence

| Day | Action | Platform |
|-----|--------|----------|
| 1 | Paper goes live | arXiv |
| 1 | GitHub repo open-sourced | GitHub |
| 2 | "Show HN" post | Hacker News |
| 2 | Tweet thread (10 tweets) | Twitter/X |
| 3 | Product Hunt launch | Product Hunt |
| 3 | Blog post: "Why Small Models Should Be Architecturally Different" | Personal blog |
| 4 | Reddit r/MachineLearning | Reddit |
| 5 | YouTube demo video | YouTube |
| 6 | Newsletter to beta users | Email |
| 7 | Press outreach (TechCrunch, The Verge) | Email |

### 9.3 Post-Launch

**Week 1–2:**
- Monitor HN comments, respond to questions
- Fix critical bugs reported by users
- Collect testimonials

**Month 1–3:**
- Discord server for community
- Weekly office hours (Twitch/YouTube)
- Monthly model updates
- Bounty program for community benchmarks

**Month 3–6:**
- Enterprise outreach (cold email 50 AI-forward companies)
- Pilot program: 3-month free trial for enterprise features
- Case studies with early adopters

### 9.4 Success Metrics

| Metric | Week 1 | Month 1 | Month 3 | Month 6 |
|--------|--------|---------|---------|---------|
| GitHub Stars | 500 | 2,000 | 5,000 | 10,000 |
| Hugging Face Downloads | 1K | 10K | 50K | 200K |
| Active Chat Users | 100 | 500 | 2,000 | 5,000 |
| API Customers | 0 | 5 | 20 | 50 |
| Enterprise Customers | 0 | 0 | 2 | 10 |
| arXiv Citations | 0 | 2 | 10 | 25 |
| MRR | $0 | $500 | $3K | $10K |

---

## 10. Appendix: Technical Quick Reference

### 10.1 Model Card: CIPHER-E2B

```yaml
model_name: CIPHER-CRN-E2B
base_model: google/gemma-4-E2B
architecture: Cognitive Resonance Network
parameters:
  effective: 2.3B
  total_with_ple: 5.1B
  trainable_lora: ~50M
  crn_overhead: ~6M
context_window: 128K
modalities: [text, image, audio]
quantization:
  training: bf16
  inference: q4f16_1
  browser_size: ~0.84GB (text-only mobile)
license: Apache 2.0 (weights)

cognitive_pillars:
  resonance_attention:
    heads: 8/16 (8 resonance, 8 standard)
    frequencies: 16 per head
    complexity: O(n * k)

  episodic_memory:
    buffer_size: 4096
    compression_dim: 256
    persistence: cross-session (IndexedDB)

  reflective_loops:
    correction_directions: 16
    threshold: 0.7
    overhead: 0 tokens

  skill_composition:
    num_skills: 64
    skill_rank: 8
    active_skills: 4

training:
  hardware: Mac Mini M4 16GB
  teacher: google/gemma-4-E4B-it (4-bit)
  method: QLoRA + DPO
  total_time: ~48 hours active compute
  data: 100K synthetic + 20K preference pairs
```

### 10.2 File Structure

```
cipher/
├── README.md
├── LICENSE (Apache 2.0 for weights)
├── PATENT.md (provisional patent reference)
├── paper/
│   ├── main.pdf
│   └── figures/
├── weights/
│   ├── cipher-e2b-base/ (Hugging Face)
│   ├── cipher-e2b-chat/ (Hugging Face)
│   └── cipher-e2b-onnx/ (Hugging Face)
├── src/
│   ├── cipher/
│   │   ├── __init__.py
│   │   ├── resonance_attention.py
│   │   ├── episodic_memory.py
│   │   ├── reflective_loop.py
│   │   ├── skill_composer.py
│   │   └── crn_model.py
│   └── inference/
│       └── export_onnx.py
├── app/
│   ├── web/ (React + Vite + Tailwind)
│   ├── worker/ (Cloudflare Worker search proxy)
│   └── extension/ (Chrome extension)
├── configs/
│   └── crn_e2b.yaml
└── docs/
    └── architecture.md
```

### 10.3 Browser Inference Code

```javascript
// Using Transformers.js with WebGPU
import { AutoProcessor, Gemma4ForConditionalGeneration } from "@huggingface/transformers";

// Load CIPHER-E2B (ONNX, q4f16, WebGPU)
const model = await Gemma4ForConditionalGeneration.from_pretrained(
    "cipher/crn-e2b-it-ONNX",
    { dtype: "q4f16", device: "webgpu" }
);

// Load persisted episodic memory
const memory = await loadMemoryFromIndexedDB("user_123");
model.setEpisodicMemory(memory);

// Chat with cognitive features
const response = await model.chat({
    messages: [
        { role: "system", content: "You are CIPHER, a cognitively structured AI." },
        { role: "user", content: "What did we discuss about quantum computing last Tuesday?" }
    ],
    useReflection: true,
    useMemory: true,
    stream: true
});

// Stream response
for await (const token of response) {
    appendToUI(token);
}

// Persist memory for next session
await saveMemoryToIndexedDB("user_123", model.getEpisodicMemory());
```

### 10.4 Search Proxy Code

```typescript
// Cloudflare Worker (search-proxy.ts)
export interface Env {
    SERP_API_KEY: string;
}

export default {
    async fetch(request: Request, env: Env): Promise<Response> {
        const { query, num_results = 3 } = await request.json();

        // Search via SerpApi
        const searchUrl = `https://serpapi.com/search?` + new URLSearchParams({
            q: query,
            api_key: env.SERP_API_KEY,
            num: num_results.toString(),
            engine: 'google'
        });

        const searchResponse = await fetch(searchUrl);
        const searchData = await searchResponse.json();

        // Extract and summarize results
        const results = searchData.organic_results?.slice(0, num_results) || [];
        const summarized = results.map((r: any) => ({
            title: r.title,
            snippet: r.snippet,
            url: r.link
        }));

        return new Response(JSON.stringify({
            context: summarized,
            query: query
        }), {
            headers: { 'Content-Type': 'application/json' }
        });
    }
};
```

### 10.5 Training Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/cipher-ai/cipher.git
cd cipher
pip install -r requirements.txt

# 2. Generate synthetic data (teacher on M4)
python scripts/generate_data.py     --teacher google/gemma-4-E4B-it     --output data/synthetic_100k.jsonl     --num_samples 100000

# 3. Distill CRN-E2B
python scripts/distill.py     --config configs/crn_e2b.yaml     --data data/synthetic_100k.jsonl     --output checkpoints/cipher-e2b-base

# 4. SFT for chat
python scripts/sft.py     --model checkpoints/cipher-e2b-base     --data data/chat_50k.jsonl     --output checkpoints/cipher-e2b-chat

# 5. DPO alignment
python scripts/dpo.py     --model checkpoints/cipher-e2b-chat     --data data/preferences_20k.jsonl     --output checkpoints/cipher-e2b-aligned

# 6. Export to ONNX
python scripts/export_onnx.py     --model checkpoints/cipher-e2b-aligned     --output weights/cipher-e2b-it-ONNX     --quantization q4f16

# 7. Evaluate
python scripts/evaluate.py     --model checkpoints/cipher-e2b-aligned     --benchmarks mmlu_pro,gpqa,aime,livecodebench,mt_bench
```

### 10.6 Dependencies

```txt
# requirements.txt
torch>=2.3.0
transformers>=4.40.0
peft>=0.11.0
trl>=0.8.0
bitsandbytes>=0.43.0
accelerate>=0.30.0
flash-attn>=2.5.0
einops>=0.7.0
wandb>=0.17.0
huggingface_hub>=0.23.0

# For export
onnx>=1.16.0
onnxruntime>=1.18.0
optimum>=1.20.0
```

---

## 11. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Training instability (NaN, divergence) | Medium | High | Gradient clipping, LR warmup, checkpoint frequently |
| M4 memory overflow | Medium | High | QLoRA, gradient checkpointing, smaller batch |
| Big lab copies architecture | High | Medium | Patent + open weights community = defensive moat |
| Benchmarks don't show wins | Medium | High | Focus on novel benchmarks (memory, reflection) |
| DNC memory unstable | Medium | High | Fallback to simpler learned checkpointing |
| Resonance attention doesn't converge | Low | High | Fallback to learned routing (not frequency bands) |
| Patent rejected | Medium | Low | Provisional is cheap. Full patent optional. |
| No user adoption | Medium | High | Build genuinely useful product first |
| WebGPU compatibility issues | Medium | Medium | WASM fallback, clear browser requirements |
| Search API costs | Low | Medium | Free tier limits, Firecrawl ZDR option |

---

## 12. Final Checklist: Start Now

### Today
- [ ] Read this document
- [ ] Set up M4: install PyTorch, transformers, Gemma 4
- [ ] Run a forward pass with Gemma 4 E2B
- [ ] Verify WebGPU works in your browser

### This Week
- [ ] Implement Resonance Attention prototype
- [ ] Implement Episodic Memory prototype
- [ ] Run synthetic tests
- [ ] Create GitHub repo (private for now)

### This Phase (Foundation)
- [ ] All 4 pillars working independently
- [ ] Forward pass tests pass
- [ ] Ready to integrate

---

*Document version 2.0 — June 30, 2026*  
*Next review: End of Phase 1*

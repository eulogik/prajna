import torch
import torch.nn as nn
import torch.nn.functional as F

from resonance_attention import ResonanceAttention
from episodic_memory import EpisodicMemory
from reflective_loop import ReflectiveLoop
from skill_composer import SkillComposer


class CRNBlock(nn.Module):
    """Single CRN transformer block with all 4 pillars."""

    def __init__(self, d_model, num_heads=4, num_frequencies=16, top_k=4,
                 num_corrections=16, num_skills=64, skill_rank=8, skill_top_k=4):
        super().__init__()

        # Pillar 1: Resonance Attention
        self.resonance_attn = ResonanceAttention(d_model, num_heads, num_frequencies, top_k)

        # Pillar 4: Skill Composer (applied after attention, before FFN)
        self.skill_composer = SkillComposer(d_model, num_skills, skill_rank, skill_top_k)

        # Pillar 3: Reflective Loop (applied after FFN)
        self.reflective_loop = ReflectiveLoop(d_model, num_corrections)

        # Standard transformer components
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x, return_info=False):
        # Pillar 1: Resonance attention with residual
        res_output = self.resonance_attn(self.norm1(x), return_freq_info=return_info)
        if isinstance(res_output, tuple):
            attn_out, freq_info = res_output
        else:
            attn_out, freq_info = res_output, None
        x = x + attn_out

        # Pillar 4: Skill composition
        sk_output = self.skill_composer(x, return_skill_info=return_info)
        if isinstance(sk_output, tuple):
            x, skill_info = sk_output
        else:
            x, skill_info = sk_output, None

        # FFN with residual
        x = x + self.ffn(self.norm2(x))

        # Pillar 3: Reflective loop (latent-space self-correction)
        x = self.reflective_loop(x)
        x = self.norm3(x)

        info = {}
        if return_info:
            if freq_info is not None:
                info["frequency"] = freq_info
            if skill_info is not None:
                info["skills"] = skill_info

        return x, info


class CRNMiniModel(nn.Module):
    """
    Minimal 2-layer CRN transformer for Prajna toy validation.

    All 4 pillars integrated:
    1. Resonance Attention — frequency-modulated attention
    2. Episodic Memory — cross-session persistent memory
    3. Reflective Loop — latent-space self-correction
    4. Skill Composition — composable skill perturbations
    """

    def __init__(self, vocab_size=256, d_model=128, num_heads=4, num_frequencies=16,
                 top_k=4, mem_size=128, mem_dim=32,
                 num_corrections=16, num_skills=64, skill_rank=8, skill_top_k=4):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(512, d_model)

        # CRN blocks (all 4 pillars)
        self.blocks = nn.ModuleList([
            CRNBlock(d_model, num_heads, num_frequencies, top_k,
                     num_corrections, num_skills, skill_rank, skill_top_k)
            for _ in range(2)
        ])

        # Pillar 2: Episodic memory (runtime state, not model params)
        self.memory = EpisodicMemory(d_model, mem_size, mem_dim)

        # Output head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def get_memory_parameters(self):
        """Get memory's learnable parameters (separate from main model)."""
        return self.memory.get_parameters()

    def get_all_pillar_parameters(self):
        """Get all parameters from the 4 pillars (for separate optimizer)."""
        params = []
        params += self.get_memory_parameters()

        # Reflective loop params
        for block in self.blocks:
            params += list(block.reflective_loop.parameters())

        # Skill composer params
        for block in self.blocks:
            params += block.skill_composer.get_skill_parameters()

        return params

    def forward(self, input_ids, return_info=False):
        """
        Args:
            input_ids: [batch, seq_len]
            return_info: whether to return all pillar analysis data
        Returns:
            logits: [batch, seq_len, vocab_size]
            info: optional dict with frequency, skill, and correction data
        """
        B, T = input_ids.shape

        # Embedding + positional
        x = self.embedding(input_ids)
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = x + self.pos_encoding(positions)

        # Read from memory (Pillar 2)
        if self.memory.temporal_positions.sum() > 0:
            mem_retrieved, mem_attn = self.memory.read(x.mean(dim=1))
            # Add retrieved memory to input (broadcast across sequence)
            x = x + mem_retrieved.unsqueeze(1) * 0.1

        # Process through CRN blocks (Pillars 1, 3, 4)
        all_info = []
        for block in self.blocks:
            x, block_info = block(x, return_info=return_info)
            if block_info:
                all_info.append(block_info)

        # Write to memory (Pillar 2)
        self.memory.write(x[:, -1, :].mean(dim=0))

        # Output
        x = self.norm(x)
        logits = self.head(x)

        if return_info and all_info:
            return logits, all_info
        return logits, None

    def generate(self, input_ids, max_new_tokens=50, temperature=0.8):
        """Simple autoregressive generation."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits, _ = self.forward(input_ids)
                next_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids

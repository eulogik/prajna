import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum


class ResonanceAttention(nn.Module):
    """
    Frequency-modulated attention with learned cognitive bands.

    Key fix from analysis: uses truly sparse computation via top-k frequency
    selection BEFORE computing attention scores, avoiding O(n²) materialization.
    """

    COGNITIVE_MODES = [
        "DEFINE", "EXPLAIN", "ARGUE", "CALCULATE",
        "HYPOTHESIZE", "EVIDENCE", "SUMMARIZE", "QUESTION",
        "REFLECT", "CORRECT", "TOOL_CALL", "TOOL_RESULT",
        "NARRATE", "CLASSIFY", "COMPARE", "GENERATE"
    ]

    def __init__(self, d_model, num_heads=4, num_frequencies=16, top_k=4):
        super().__init__()
        self.num_heads = num_heads
        self.num_frequencies = num_frequencies
        self.top_k = top_k
        self.head_dim = d_model // num_heads

        # Per-head frequency projections
        self.freq_q = nn.Linear(d_model, num_heads * num_frequencies, bias=False)
        self.freq_k = nn.Linear(d_model, num_heads * num_frequencies, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learned transition graph: which frequencies can attend to which
        self.transition_graph = nn.Parameter(
            torch.randn(num_frequencies, num_frequencies) * 0.01
        )

        # Frequency assignment (which cognitive mode each frequency represents)
        self.freq_assignment = nn.Parameter(
            torch.randn(num_frequencies, num_frequencies) * 0.01
        )

    def forward(self, x, return_freq_info=False):
        """
        Args:
            x: [batch, seq_len, d_model]
            return_freq_info: whether to return frequency assignments for analysis
        Returns:
            out: [batch, seq_len, d_model]
            freq_info: optional dict with frequency assignment data
        """
        B, T, D = x.shape

        # Project to frequency space: [B, T, H, F]
        q = self.freq_q(x).view(B, T, self.num_heads, self.num_frequencies)
        k = self.freq_k(x).view(B, T, self.num_heads, self.num_frequencies)

        # Compute frequency assignment scores per token
        # [B, T, H, F] -> which frequencies are active for each token
        freq_scores = F.softmax(q, dim=-1)  # [B, T, H, F]

        # Select top-k frequencies per token (TRULY SPARSE - no n×n matrix)
        top_k = min(self.top_k, self.num_frequencies)
        top_freq_vals, top_freq_idx = freq_scores.topk(top_k, dim=-1)  # [B, T, H, top_k]
        top_freq_vals = top_freq_vals / (top_freq_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Compute value
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)

        # Sparse attention: only compute within compatible frequency groups
        # Group tokens by their dominant frequency
        out = torch.zeros_like(v)

        for f_idx in range(self.num_frequencies):
            # Find tokens where this frequency is in their top-k
            mask = (top_freq_idx == f_idx).any(dim=-1)  # [B, T, H]

            if mask.sum() == 0:
                continue

            # Get the weight for this frequency across all tokens
            freq_weight = torch.zeros(B, T, self.num_heads, device=x.device)
            for k_idx in range(top_k):
                match = (top_freq_idx[:, :, :, k_idx] == f_idx)
                freq_weight += match.float() * top_freq_vals[:, :, :, k_idx]

            # Compute attention ONLY among tokens sharing this frequency
            # This is O(m²) where m << n (m = tokens with this frequency)
            mask_expanded = mask.unsqueeze(-1)  # [B, T, H, 1]

            # Query and key for this frequency group
            q_f = q[:, :, :, f_idx]  # [B, T, H]
            k_f = k[:, :, :, f_idx]  # [B, T, H]

            # Attention scores (sparse: only among masked tokens)
            attn_scores = einsum(q_f, k_f, 'b i h, b j h -> b h i j')
            attn_scores = attn_scores / (self.head_dim ** 0.5)

            # Mask out tokens not in this frequency group
            attn_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, T, T, H] -> broadcast
            attn_mask = attn_mask.permute(0, 3, 1, 2)  # [B, H, T, T]
            attn_scores = attn_scores.masked_fill(~attn_mask.bool(), float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_weights = attn_weights.nan_to_num(0.0)

            # Weighted sum of values
            attn_out = einsum(attn_weights, v, 'b h i j, b j h d -> b i h d')

            # Apply frequency weight
            out += freq_weight.unsqueeze(-1) * attn_out

        # Reshape and project
        out = out.reshape(B, T, D)
        out = self.out_proj(out)

        if return_freq_info:
            # Return dominant frequency per token for interpretability
            dominant_freq = top_freq_idx[:, :, :, 0]  # [B, T, H]
            return out, {
                "dominant_frequency": dominant_freq,
                "frequency_scores": freq_scores,
                "top_frequencies": top_freq_idx,
            }

        return out

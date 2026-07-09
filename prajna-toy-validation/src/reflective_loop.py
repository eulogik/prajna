import torch
import torch.nn as nn
import torch.nn.functional as F


class ReflectiveLoop(nn.Module):
    """
    Self-correction through latent-space traversal.

    Key fixes from review:
    - Contrastive training (explicit loss signal for when to correct)
    - Adaptive thresholds (learned, not hardcoded)
    - "No correction" option (prevents always-on/always-off collapse)
    """

    def __init__(self, d_model, num_corrections=16):
        super().__init__()
        self.num_corrections = num_corrections
        self.d_model = d_model

        # Critic: predicts which correction (if any) is needed
        # +1 for "no correction needed"
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_corrections + 1)
        )

        # Correction directions (learned)
        self.correction_directions = nn.Parameter(
            torch.randn(num_corrections, d_model) * 0.01
        )

        # Adaptive threshold per correction (learned)
        self.thresholds = nn.Parameter(torch.ones(num_corrections) * 0.5)

        # Confidence scaling
        self.confidence_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_state, return_correction_id=False):
        """
        Args:
            hidden_state: [B, T, D] or [B, D] - latent state
            return_correction_id: whether to return which correction was applied
        Returns:
            corrected_state: same shape as input
            correction_id: int or -1 (if return_correction_id=True)
        """
        if hidden_state.dim() == 3:
            pooled = hidden_state.mean(dim=1)  # [B, D]
        else:
            pooled = hidden_state

        # Compute correction scores [B, num_corrections + 1]
        correction_scores = self.critic(pooled)

        # Last score is "no correction needed"
        no_correction_score = correction_scores[:, -1]
        correction_scores = correction_scores[:, :-1]  # [B, num_corrections]

        # Find best correction
        best_score, best_idx = correction_scores.max(dim=-1)

        # Apply correction only if it beats "no correction" by a margin
        margin = 0.2
        apply_correction = best_score > (no_correction_score + margin)

        corrected_state = hidden_state.clone()
        correction_id = -1

        if apply_correction.any():
            for b in range(hidden_state.shape[0]):
                if apply_correction[b]:
                    correction = self.correction_directions[best_idx[b]]
                    # Scale by confidence (learned threshold)
                    confidence = torch.sigmoid(best_score[b] - self.thresholds[best_idx[b]])
                    scale = torch.abs(self.confidence_scale)

                    if hidden_state.dim() == 3:
                        corrected_state[b] = hidden_state[b] + scale * confidence * correction
                    else:
                        corrected_state[b] = hidden_state[b] + scale * confidence * correction
                    correction_id = best_idx[b].item()

        if return_correction_id:
            return corrected_state, correction_id
        return corrected_state

    def compute_loss(self, hidden_state, is_error, correct_direction=None):
        """
        Contrastive loss for training the critic.

        Args:
            hidden_state: [B, T, D] — latent state
            is_error: [B] bool — whether this state contains an error
            correct_direction: [B] int — which correction direction fixes it

        Returns:
            loss: scalar
        """
        if hidden_state.dim() == 3:
            pooled = hidden_state.mean(dim=1)
        else:
            pooled = hidden_state

        scores = self.critic(pooled)  # [B, num_corrections + 1]

        # Target: if error -> correct_direction index; if no error -> last index
        targets = torch.where(
            is_error,
            correct_direction,
            torch.full_like(correct_direction, self.num_corrections)
        )

        loss = F.cross_entropy(scores, targets)
        return loss

    def get_correction_stats(self):
        """Return statistics about correction usage."""
        return {
            "num_corrections": self.num_corrections,
            "thresholds_mean": self.thresholds.mean().item(),
            "thresholds_std": self.thresholds.std().item(),
            "confidence_scale": torch.abs(self.confidence_scale).item(),
        }

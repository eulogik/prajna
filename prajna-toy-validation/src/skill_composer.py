import torch
import torch.nn as nn
import torch.nn.functional as F


class SkillComposer(nn.Module):
    """
    Skill Resonance Composition — skills as learned perturbations to latent state.

    Key insight from review: most novel idea in the architecture. 64 composable
    skill vectors at 32K params total. Far more efficient than MoE or Sparse LoRA.

    Skills are applied as additive perturbations to hidden states, not weight
    modifications. Continuous and composable by superposition.
    """

    SKILL_NAMES = [
        "MATHEMATICS", "LOGIC", "WRITING", "CODE", "ANALYSIS",
        "CREATIVITY", "REASONING", "RECALL", "SUMMARY", "PLANNING",
        "DEBUGGING", "EXPLANATION", "COMPARISON", "CLASSIFICATION",
        "GENERATION", "TRANSLATION", "OPTIMIZATION", "VERIFICATION",
        "RESEARCH", "SYNTHESIS", "DECOMPOSITION", "ABSTRACTION",
        "PATTERN_MATCH", "CAUSAL_REASONING", "TEMPORAL", "SPATIAL",
        "EMOTIONAL", "PERSUASION", "TEACHING", "QUESTIONING",
        "HYPOTHESIS", "TESTING", "MEASUREMENT", "ESTIMATION",
        "PRIORITIZATION", "SEQUENCING", "ITERATION", "REFINEMENT",
        "AGGREGATION", "FILTERING", "TRANSFORMATION", "MAPPING",
        "VALIDATION", "CRITIQUE", "INTUITION", "HEURISTIC",
        "FORMALIZATION", "CONTEXTUALIZATION", "COMPRESSION", "EXPANSION",
        "FOCUS", "DIVERGENCE", "CONVERGENCE", "BRIDGING",
        "NARRATIVE", "ARGUMENTATION", "DEFINITION", "EXEMPLIFICATION",
        "QUANTIFICATION", "QUALITATIVE", "SYSTEMS_THINKING", "META_COGNITION",
        "ERROR_CORRECTION", "KNOWLEDGE_TRANSFER", "ADAPTATION", "INTEGRATION",
    ]

    def __init__(self, d_model, num_skills=64, skill_rank=8, top_k=4):
        super().__init__()
        self.num_skills = num_skills
        self.skill_rank = skill_rank
        self.top_k = top_k
        self.d_model = d_model

        # Low-rank skill vectors: u @ v^T per skill
        # Total params: num_skills * d_model * skill_rank * 2
        self.skill_u = nn.Parameter(torch.randn(num_skills, d_model, skill_rank) * 0.01)
        self.skill_v = nn.Parameter(torch.randn(num_skills, skill_rank, d_model) * 0.01)

        # Router: decides which skills to activate
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, num_skills)
        )

        # Skill scaling (learned)
        self.skill_scale = nn.Parameter(torch.ones(num_skills) * 0.01)

    def forward(self, x, return_skill_info=False):
        """
        Args:
            x: [B, T, D] hidden state
            return_skill_info: whether to return active skill info
        Returns:
            output: [B, T, D] hidden state with skill perturbation
            skill_info: optional dict with active skills and weights
        """
        B, T, D = x.shape

        # Compute skill weights from mean pooled representation
        pooled = x.mean(dim=1)  # [B, D]
        skill_logits = self.router(pooled)  # [B, num_skills]
        skill_weights = F.softmax(skill_logits, dim=-1)  # [B, num_skills]

        # Load balancing: encourage uniform usage across skills
        # (prevents router collapse to a single skill)
        if self.training:
            avg_usage = skill_weights.mean(dim=0)  # [num_skills]
            self._load_balance_loss = (avg_usage.var() * 10.0)  # Penalize uneven usage
        else:
            self._load_balance_loss = torch.tensor(0.0)

        # Select top-k skills
        top_k = min(self.top_k, self.num_skills)
        top_weights, top_indices = skill_weights.topk(top_k, dim=-1)  # [B, top_k]
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Compose skill perturbations
        perturbation = torch.zeros_like(x)  # [B, T, D]

        for k in range(self.top_k):
            skill_idx = top_indices[:, k]  # [B]
            weight = top_weights[:, k]  # [B]

            # Gather skill parameters for this batch
            # skill_u: [num_skills, D, rank] -> [B, D, rank]
            u = self.skill_u[skill_idx]
            # skill_v: [num_skills, rank, D] -> [B, rank, D]
            v = self.skill_v[skill_idx]
            # skill_scale: [num_skills] -> [B]
            scale = torch.abs(self.skill_scale[skill_idx])

            # Compute low-rank perturbation: x @ v^T @ u^T
            # x: [B, T, D], v: [B, rank, D]
            # x @ v^T: [B, T, rank]
            x_v = torch.bmm(x, v.transpose(1, 2))  # [B, T, rank]
            # x_v @ u^T: [B, T, D]
            skill_perturbation = torch.bmm(x_v, u.transpose(1, 2))  # [B, T, D]

            # Apply with weight and scale
            # weight: [B] -> [B, 1, 1], scale: [B] -> [B, 1, 1]
            perturbation += weight.unsqueeze(1).unsqueeze(-1) * scale.unsqueeze(1).unsqueeze(-1) * skill_perturbation

        output = x + perturbation

        if return_skill_info:
            # Return which skills are active and their weights
            active_skills = []
            for b in range(B):
                batch_skills = []
                for k in range(self.top_k):
                    idx = top_indices[b, k].item()
                    batch_skills.append({
                        "name": self.SKILL_NAMES[idx] if idx < len(self.SKILL_NAMES) else f"skill_{idx}",
                        "weight": top_weights[b, k].item(),
                        "scale": torch.abs(self.skill_scale[idx]).item(),
                    })
                active_skills.append(batch_skills)

            return output, {
                "active_skills": active_skills,
                "skill_weights": skill_weights,
                "top_indices": top_indices,
            }

        return output

    def get_skill_parameters(self):
        """Return all skill-related parameters."""
        return [self.skill_u, self.skill_v, self.skill_scale] + \
               list(self.router.parameters())

    def get_active_skill_names(self, skill_indices):
        """Convert skill indices to human-readable names."""
        names = []
        for idx in skill_indices.flatten().tolist():
            if idx < len(self.SKILL_NAMES):
                names.append(self.SKILL_NAMES[idx])
            else:
                names.append(f"skill_{idx}")
        return names

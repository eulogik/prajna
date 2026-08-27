#!/usr/bin/env python3
"""CRN v2: Frozen base + trainable logit-level correction.

Design:
- Frozen base model (preserves all base capabilities)
- Lightweight LogitCorrection module (takes final hidden → produces logit delta)
- Trained with cross-entropy (correction) + KL preservation (keeps base behavior)
- ~3M params total (vs CRN v1's 8M with broken hidden-state path)

Key insight: the frozen lm_head blocks hidden-state corrections (v1's failure).
Logit-level correction bypasses this entirely.
"""
import os, gc, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LogitCorrection(nn.Module):
    """Lightweight logit-level correction: hidden → logit delta.

    final_logits = base_logits + gate * up(gelu(down(hidden)))
    Init near zero so training starts from base behavior.
    """
    def __init__(self, d_model, vocab, rank=128, init_scale=1e-3, gate_init=0.1):
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, vocab, bias=True)
        nn.init.normal_(self.up.weight, std=init_scale)
        self.up.bias.data.zero_()
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, hidden):
        x = F.gelu(self.down(hidden))
        return self.gate * self.up(x)


class CRNv2(nn.Module):
    """Frozen base + trainable logit correction + KL preservation loss.

    Training loss = cross_entropy (correction) + lambda_kl * KL(p_base || p_corrected)
    KL preservation pushes corrected logits toward base logits on ALL inputs,
    while CE pushes them toward the correct answer. On inputs the base already
    gets right, KL dominates → preservation. On inputs the base gets wrong,
    CE dominates → correction.
    """
    def __init__(self, device='cpu', lambda_kl=0.1, correction_rank=128):
        super().__init__()
        self.device = device
        self.lambda_kl = lambda_kl
        gc.collect()
        print('Loading base model (frozen)...')
        self.tok = AutoTokenizer.from_pretrained('google/gemma-4-E2B')
        self.base = AutoModelForCausalLM.from_pretrained(
            'google/gemma-4-E2B', dtype=torch.float16, low_cpu_mem_usage=False)
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.to(device).eval()

        self.vocab = 262144
        self.d_model = 1536
        self.correction = LogitCorrection(
            self.d_model, self.vocab, rank=correction_rank).to(device)
        n = sum(p.numel() for p in self.correction.parameters())
        print(f'CRN v2: {n:,} trainable params (logit correction, rank={correction_rank})')

    def get_params(self):
        return list(self.correction.parameters())

    def forward(self, input_ids, labels=None, eos_weight=1.0, pos_weights=None):
        """Forward with correction + optional KL preservation loss."""
        with torch.no_grad():
            base_out = self.base(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            base_logits = base_out.logits.float()
            final_hidden = base_out.hidden_states[-1].float()
            del base_out
        delta = self.correction(final_hidden)
        logits = (base_logits + delta).to(torch.float16)

        loss = None
        if labels is not None:
            # Correction loss: cross-entropy on chosen answer
            per_token_ce = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab),
                labels[:, 1:].reshape(-1), ignore_index=-100, reduction='none'
            )
            per_token_ce = per_token_ce.reshape(input_ids.shape[0], -1)
            if pos_weights is None:
                pos_weights = torch.ones_like(labels, dtype=torch.float32)
            loss_weights = pos_weights[:, 1:] if pos_weights.shape[1] == labels.shape[1] else pos_weights
            if eos_weight != 1.0:
                eos_mask = (labels[:, 1:] == self.tok.eos_token_id).float()
                loss_weights = loss_weights * (1.0 + (eos_weight - 1.0) * eos_mask)
            effective_weights = loss_weights * (labels[:, 1:] != -100).float()
            ce_loss = (per_token_ce * effective_weights).sum() / effective_weights.sum().clamp(min=1e-6)

            # KL preservation: penalize deviation from base logits
            # KL(p_base || p_corrected) — pushes corrected toward base
            base_logprobs = F.log_softmax(base_logits[:, :-1].reshape(-1, self.vocab), dim=-1)
            corr_logprobs = F.log_softmax(logits[:, :-1].reshape(-1, self.vocab), dim=-1)
            base_probs = F.softmax(base_logits[:, :-1].reshape(-1, self.vocab), dim=-1)
            kl_per_token = F.kl_div(corr_logprobs, base_probs, reduction='none').sum(dim=-1)
            kl_per_token = kl_per_token.reshape(input_ids.shape[0], -1)
            kl_mask = (labels[:, 1:] != -100).float()
            kl_loss = (kl_per_token * kl_mask).sum() / kl_mask.sum().clamp(min=1e-6)

            loss = ce_loss + self.lambda_kl * kl_loss

        return {'loss': loss, 'logits': logits, 'ce_loss': ce_loss.item() if loss is not None else 0,
                'kl_loss': kl_loss.item() if loss is not None else 0}

    def save(self, path):
        torch.save(self.correction.state_dict(), path)
        print(f'Saved CRN v2 correction to {path}')

    def load(self, path):
        self.correction.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        print(f'Loaded CRN v2 correction from {path}')

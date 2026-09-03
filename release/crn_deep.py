#!/usr/bin/env python3
"""CRN Deep Injection: frozen base + hidden-state correction at early layers via forward hooks.

Key idea: Instead of adding to the final hidden state (blocked by frozen lm_head)
or to the logits directly (limited leverage), inject corrections at EARLY layers
where they cascade through all subsequent frozen layers before reaching lm_head.

Hidden-level correction is far more parameter-efficient than logit-level:
  - Hidden correction: h(1536) -> rank(128) -> h(1536)  = ~393K params per depth
  - Logit correction:  h(1536) -> rank(128) -> vocab(262K) = ~34M params per depth
The leverage comes from 27+ frozen layers amplifying the small correction.

Training: SFT with anchor weighting + KL preservation + DPO.
The base model IS frozen but grads flow THROUGH it (27 layers) back to the
correction modules via the hooks. This is safe because base receives no gradients
on its OWN parameters — only the correction params update.
"""
import os, gc, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class HiddenCorrection(nn.Module):
    """Low-rank hidden-state residual: h -> h + gate * W_up(GeLU(W_down(h)))."""
    def __init__(self, d_model, rank=128, init_scale=1e-3, gate_init=0.01):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=True)
        nn.init.normal_(self.up.weight, std=init_scale)
        self.up.bias.data.zero_()
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, hidden):
        # Handle MPS dtype: convert to correction weight dtype for matmul
        dtype = self.down.weight.dtype
        x = F.gelu(self.down(hidden.to(dtype)))
        return self.gate * self.up(x).to(hidden.dtype)


class CRNDeepInjection(nn.Module):
    """Frozen base + hidden-state injection at specified depths + KL preservation.

    Injection via forward hooks on selected language_model layers.
    The correction at depth K propagates through layers K+1..L naturally.
    """
    def __init__(self, device='cpu', lambda_kl=0.1, correction_rank=128, depths=None):
        super().__init__()
        self.device = device
        self.lambda_kl = lambda_kl
        self.depths = depths or [7]  # default: inject at layer 7

        gc.collect()
        print('Loading base model (frozen)...')
        self.tok = AutoTokenizer.from_pretrained('google/gemma-4-E2B')
        self.base = AutoModelForCausalLM.from_pretrained(
            'google/gemma-4-E2B', dtype=torch.float16, low_cpu_mem_usage=False)
        # Freeze ALL base params
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.to(device)
        self.base.eval()

        self.vocab = 262144
        self.d_model = 1536
        self.lm = self.base.model.language_model  # 35 text layers

        # One correction module per injection depth
        self.corrections = nn.ModuleDict({
            str(d): HiddenCorrection(self.d_model, rank=correction_rank).to(device)
            for d in self.depths
        })
        n = sum(p.numel() for p in self.corrections.parameters())
        print(f'CRN Deep: {n:,} trainable params (depths={self.depths}, rank={correction_rank})')

        # Register hooks — stored so we can remove if needed
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def make_hook(corr):
            def hook(module, inputs, output):
                # Gemma4TextDecoderLayer output is a tuple (hidden_states, ...) or tensor
                if isinstance(output, tuple):
                    corrected = output[0] + corr(output[0])
                    return (corrected,) + output[1:]
                else:
                    return output + corr(output)
            return hook
        for d in self.depths:
            h = self.lm.layers[d].register_forward_hook(make_hook(self.corrections[str(d)]))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_params(self):
        return list(self.corrections.parameters())

    def _forward_with_correction(self, input_ids):
        """Full base forward WITH hooks active -> corrected logits."""
        return self.base(input_ids=input_ids).logits.float()

    def _forward_base_only(self, input_ids):
        """Base forward WITHOUT correction (remove hooks, run, re-register)."""
        self.remove_hooks()
        with torch.no_grad():
            logits = self.base(input_ids=input_ids).logits.float()
        self._register_hooks()
        return logits

    def forward(self, input_ids, labels=None, eos_weight=1.0, pos_weights=None):
        # With hooks active, gradients flow through frozen layers back to correction
        # Keep base in eval but allow grad flow via hooks
        self.base.eval()
        corrected_logits = self._forward_with_correction(input_ids)

        if labels is None:
            return {'loss': None, 'logits': corrected_logits, 'ce_loss': 0, 'kl_loss': 0}

        # CE loss on answer tokens
        per_token_ce = F.cross_entropy(
            corrected_logits[:, :-1].reshape(-1, self.vocab),
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

        # KL preservation: compare corrected vs base-only logits
        with torch.no_grad():
            base_logits = self._forward_base_only(input_ids)
        base_probs = F.softmax(base_logits[:, :-1].reshape(-1, self.vocab), dim=-1)
        corr_logprobs = F.log_softmax(corrected_logits[:, :-1].reshape(-1, self.vocab), dim=-1)
        kl_per_token = F.kl_div(corr_logprobs, base_probs, reduction='none').sum(dim=-1)
        kl_per_token = kl_per_token.reshape(input_ids.shape[0], -1)
        kl_mask = (labels[:, 1:] != -100).float()
        kl_loss = (kl_per_token * kl_mask).sum() / kl_mask.sum().clamp(min=1e-6)

        loss = ce_loss + self.lambda_kl * kl_loss
        return {'loss': loss, 'logits': corrected_logits, 'ce_loss': ce_loss.item(), 'kl_loss': kl_loss.item()}

    def save(self, path):
        torch.save(self.corrections.state_dict(), path)
        print(f'Saved CRN Deep corrections to {path}')

    def load(self, path):
        self.corrections.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        print(f'Loaded CRN Deep corrections from {path}')

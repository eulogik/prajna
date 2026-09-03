#!/usr/bin/env python3
"""CRN v2 multi-depth: frozen base + logit correction from multiple hidden-state depths.

Key difference from CRN v2: reads hidden states from multiple intermediate layers
(depths 7, 15, 23, 31) in addition to the final layer, concatenates them, and
produces a single logit correction. This gives the correction module access to
multi-scale representations — analogous to how LoRA modifies weights at every layer.

Architecture:
  delta = gate * W_up(gelu(W_down(concat(h^(d1), h^(d2), ..., h^(dN), h^(L)))))
  corrected_logits = base_logits + delta
"""
import os, gc, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


DEPTHS = [7, 15, 23, 31]  # intermediate layers to read


class MultiDepthCorrection(nn.Module):
    """Reads hidden states from multiple depths, produces a single logit delta."""
    def __init__(self, d_model, vocab, n_depths=4, rank=128, init_scale=1e-3, gate_init=0.1):
        super().__init__()
        self.n_depths = n_depths
        input_dim = d_model * (n_depths + 1)  # n_depths intermediate + final layer
        self.down = nn.Linear(input_dim, rank, bias=False)
        self.up = nn.Linear(rank, vocab, bias=True)
        nn.init.normal_(self.up.weight, std=init_scale)
        self.up.bias.data.zero_()
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, hidden_states_list):
        """hidden_states_list: list of tensors [batch, seq, d_model], one per depth (last = final)."""
        concat = torch.cat(hidden_states_list, dim=-1)  # [batch, seq, d_model * n_depths]
        x = F.gelu(self.down(concat))
        return self.gate * self.up(x)


class CRNv2MultiDepth(nn.Module):
    """Frozen base + multi-depth logit correction + KL preservation loss.

    Training loss = CE (correction) + lambda_kl * KL(p_base || p_corrected)
    """
    def __init__(self, device='cpu', lambda_kl=0.1, correction_rank=128, depths=None):
        super().__init__()
        self.device = device
        self.lambda_kl = lambda_kl
        self.depths = depths or DEPTHS

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
        self.correction = MultiDepthCorrection(
            self.d_model, self.vocab,
            n_depths=len(self.depths),
            rank=correction_rank).to(device)
        n = sum(p.numel() for p in self.correction.parameters())
        print(f'CRN v2 multi-depth: {n:,} trainable params (depths={self.depths}, rank={correction_rank})')

    def get_params(self):
        return list(self.correction.parameters())

    def _collect_hidden_states(self, input_ids):
        """Run base model, collect hidden states at specified depths + final layer."""
        with torch.no_grad():
            base_out = self.base(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            base_logits = base_out.logits.float()
            all_hidden = [base_out.hidden_states[i].float() for i in self.depths]
            all_hidden.append(base_out.hidden_states[-1].float())  # final layer
            del base_out
        return base_logits, all_hidden

    def forward(self, input_ids, labels=None, eos_weight=1.0, pos_weights=None):
        """Forward with multi-depth correction + optional KL preservation loss."""
        base_logits, hidden_list = self._collect_hidden_states(input_ids)
        delta = self.correction(hidden_list)
        logits = (base_logits + delta).to(torch.float16)

        loss = None
        if labels is not None:
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
        print(f'Saved CRN v2 multi-depth correction to {path}')

    def load(self, path):
        self.correction.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
        print(f'Loaded CRN v2 multi-depth correction from {path}')

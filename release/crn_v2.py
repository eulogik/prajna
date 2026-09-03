#!/usr/bin/env python3
"""CRN v2: Safe error correction via frozen-base logit adjustment.

A lightweight correction module sits on top of a frozen base model.
The base model never changes — the correction module learns to adjust
its logits when it detects errors.

Architecture:
  corrected_logits = base_logits + gate * up(gelu(down(hidden)))

Trainable params: ~34M (rank=128) out of 4.6B total — 0.73% of base model.
Base model params: frozen, never updated during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class LogitCorrection(nn.Module):
    """Low-rank logit correction: hidden → logit delta.

    corrected_logits = base_logits + gate * up(gelu(down(hidden)))
    Initialized near zero so the model starts from base behavior.
    """
    def __init__(self, d_model, vocab, rank=128, init_scale=1e-3, gate_init=0.1):
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, vocab, bias=True)
        nn.init.normal_(self.up.weight, std=init_scale)
        self.up.bias.data.zero_()
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, hidden):
        x = F.gelu(self.down(hidden))
        return self.gate * self.up(x)


class CRNv2:
    """Frozen base + trainable logit correction.

    Usage:
        model = CRNv2(device='mps')
        model.load('checkpoints/crn_v2_dpo.pt')
        answer = model.correct("What is 2+2?", "5")  # → "4"
    """
    def __init__(self, device='cpu', correction_rank=128):
        self.device = device
        self.tok = AutoTokenizer.from_pretrained('google/gemma-4-E2B')
        self.base = AutoModelForCausalLM.from_pretrained(
            'google/gemma-4-E2B', dtype=torch.float16, low_cpu_mem_usage=False)
        for p in self.base.parameters():
            p.requires_grad = False
        self.base.to(device).eval()

        self.correction = LogitCorrection(
            d_model=1536, vocab=262144, rank=correction_rank).to(device)
        n = sum(p.numel() for p in self.correction.parameters())
        print(f'CRN v2: {n:,} trainable params (rank={correction_rank})')

    def load(self, path):
        self.correction.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True))

    def _corrected_logits(self, input_ids):
        """Get corrected logits for a sequence."""
        with torch.no_grad():
            base_out = self.base(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            base_logits = base_out.logits.float()
            final_hidden = base_out.hidden_states[-1].float()
            del base_out
        delta = self.correction(final_hidden)
        return (base_logits + delta).to(torch.float16)

    def generate(self, prompt, max_new_tokens=64):
        """Generate text using corrected logits (autoregressive)."""
        ids = self.tok(prompt, return_tensors='pt').input_ids.to(self.device)
        prompt_len = ids.shape[1]
        generated = ids.clone()
        for _ in range(max_new_tokens):
            logits = self._corrected_logits(generated)
            next_token = logits[0, -1].argmax(dim=-1).unsqueeze(0).unsqueeze(0)
            if next_token.item() == self.tok.eos_token_id:
                break
            generated = torch.cat([generated, next_token], dim=1)
        return self.tok.decode(generated[0][prompt_len:], skip_special_tokens=True).strip()

    def correct(self, prompt, draft_answer, context=""):
        """Correct a draft answer. Returns the corrected answer."""
        full_prompt = f"{context}\n{prompt}\nDraft answer: {draft_answer}\nCorrected answer:" if context else f"{prompt}\nDraft answer: {draft_answer}\nCorrected answer:"
        return self.generate(full_prompt, max_new_tokens=64)

    def base_generate(self, prompt, max_new_tokens=64):
        """Generate using base model only (no correction) for comparison."""
        ids = self.tok(prompt, return_tensors='pt').input_ids.to(self.device)
        out = self.base.generate(ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()

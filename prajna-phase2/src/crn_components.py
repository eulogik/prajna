#!/usr/bin/env python3
"""CRN model components — standalone, importable without running training.

Extracted from train_mac.py so eval/inference scripts can load the model
without executing the SFT/DPO training pipeline.
"""
import torch, gc, os, json
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============ CRN Components ============
class ResonanceAttention(nn.Module):
    def __init__(self, d_model, num_heads=4, num_frequencies=8, top_k=2):
        super().__init__()
        self.num_heads = num_heads
        self.num_frequencies = num_frequencies
        self.top_k = top_k
        self.head_dim = d_model // num_heads
        self.freq_q = nn.Linear(d_model, num_heads * num_frequencies, bias=False)
        self.freq_k = nn.Linear(d_model, num_heads * num_frequencies, bias=False)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        device = x.device
        q = self.freq_q(x).view(B, T, self.num_heads, self.num_frequencies)
        k = self.freq_k(x).view(B, T, self.num_heads, self.num_frequencies)
        freq_scores = F.softmax(q, dim=-1)
        top_freq_vals, top_freq_idx = freq_scores.topk(min(self.top_k, self.num_frequencies), dim=-1)
        top_freq_vals = top_freq_vals / (top_freq_vals.sum(dim=-1, keepdim=True) + 1e-8)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim)
        freq_weight = torch.zeros(B, T, self.num_heads, self.num_frequencies, device=device)
        freq_weight.scatter_add_(3, top_freq_idx, top_freq_vals)
        attn = torch.einsum('bihf,bjhf->bhfij', q, k) / (self.head_dim ** 0.5)
        freq_membership = torch.zeros(B, T, self.num_heads, self.num_frequencies, device=device, dtype=torch.bool)
        freq_membership.scatter_(3, top_freq_idx, True)
        shared = torch.einsum('bthf,bshf->bhts', freq_membership.float(), freq_membership.float()) > 0
        mask = shared.unsqueeze(2).expand(-1, -1, self.num_frequencies, -1, -1)
        attn = attn.masked_fill(~mask, float('-inf'))
        attn = F.softmax(attn, dim=-1).nan_to_num(0.0)
        v_perm = v.permute(0, 2, 1, 3)
        out = torch.einsum('bhfij,bhjd->bhfid', attn, v_perm)
        fw = freq_weight.permute(0, 2, 3, 1).unsqueeze(-1)
        out = (out * fw).sum(dim=2)
        out = out.permute(0, 2, 1, 3)
        return self.out_proj(out.reshape(B, T, D))


class EpisodicMemory(nn.Module):
    def __init__(self, d_model, mem_size=256, mem_dim=64, device='cpu'):
        super().__init__()
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.d_model = d_model
        self.register_buffer('memory', torch.zeros(mem_size, mem_dim))
        self.register_buffer('temporal_positions', torch.zeros(mem_size))
        self.write_ptr = 0
        self.step_count = 0
        self.compress = nn.Linear(d_model, mem_dim)
        self.decompress = nn.Linear(mem_dim, d_model)
        self.read_gate = nn.Linear(d_model, mem_dim)
        self.write_gate = nn.Linear(d_model, 1)
        self.relevance_gate = nn.Linear(d_model + mem_dim, 1)

    def get_parameters(self):
        return list(self.parameters())

    def read(self, query, top_k=8):
        if query.dim() == 1: query = query.unsqueeze(0)
        B = query.shape[0]
        q_compressed = self.read_gate(query)
        mem_expanded = self.memory.unsqueeze(0).expand(B, -1, -1)
        q_norm = F.normalize(q_compressed, dim=-1)
        mem_norm = F.normalize(mem_expanded, dim=-1)
        sims = torch.bmm(q_norm.unsqueeze(1), mem_norm.transpose(1, 2)).squeeze(1)
        recency = self.temporal_positions / (self.temporal_positions.max() + 1)
        sims = sims + 0.1 * recency.unsqueeze(0)
        top_k = min(top_k, self.mem_size)
        top_vals, top_idx = sims.topk(top_k, dim=-1)
        attn_weights = F.softmax(top_vals, dim=-1)
        retrieved = torch.gather(mem_expanded, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.mem_dim))
        retrieved = (retrieved * attn_weights.unsqueeze(-1)).sum(dim=1)
        return self.decompress(retrieved), attn_weights

    def write(self, content, force=False):
        gate_value = torch.sigmoid(self.write_gate(content.unsqueeze(0))).item()
        if gate_value < 0.5 and not force: return False
        compressed = self.compress(content.detach())
        if self.write_ptr < self.mem_size:
            slot = self.write_ptr
            self.write_ptr += 1
        else:
            slot = self.temporal_positions.argmin().item()
        write_weight = min(gate_value, 0.9)
        self.memory[slot] = (write_weight * compressed + (1 - write_weight) * self.memory[slot].clone()).detach()
        self.step_count += 1
        self.temporal_positions[slot] = self.step_count
        return True

    def save(self, path):
        state = {
            'memory': self.memory.detach().cpu().float().numpy().tolist(),
            'temporal_positions': self.temporal_positions.detach().cpu().float().numpy().tolist(),
            'write_ptr': self.write_ptr, 'step_count': self.step_count
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f: json.dump(state, f)

    def load(self, path):
        with open(path) as f: state = json.load(f)
        self.memory.data = torch.tensor(state['memory'], dtype=torch.float32, device=self.memory.device)
        self.temporal_positions.data = torch.tensor(state['temporal_positions'], dtype=torch.float32, device=self.temporal_positions.device)
        self.write_ptr = state['write_ptr']
        self.step_count = state['step_count']


class ReflectiveLoop(nn.Module):
    def __init__(self, d_model, num_corrections=8):
        super().__init__()
        self.num_corrections = num_corrections
        self.d_model = d_model
        self.critic = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, num_corrections + 1)
        )
        self.correction_directions = nn.Parameter(torch.randn(num_corrections, d_model) * 0.01)
        self.thresholds = nn.Parameter(torch.ones(num_corrections) * 0.5)
        self.confidence_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, hidden_state, return_correction_id=False):
        pooled = hidden_state.mean(dim=1) if hidden_state.dim() == 3 else hidden_state
        scores = self.critic(pooled)
        no_correction_score = scores[:, -1]
        correction_scores = scores[:, :-1]
        best_score, best_idx = correction_scores.max(dim=-1)
        apply_correction = best_score > (no_correction_score + 0.2)
        corrected_state = hidden_state.clone()
        correction_id = -1
        if apply_correction.any():
            for b in range(hidden_state.shape[0]):
                if apply_correction[b]:
                    correction = self.correction_directions[best_idx[b]]
                    confidence = torch.sigmoid(best_score[b] - self.thresholds[best_idx[b]])
                    scale = torch.abs(self.confidence_scale)
                    corrected_state[b] = hidden_state[b] + scale * confidence * correction
                    correction_id = best_idx[b].item()
        return (corrected_state, correction_id) if return_correction_id else corrected_state


class SkillComposer(nn.Module):
    def __init__(self, d_model, num_skills=32, skill_rank=4, top_k=2):
        super().__init__()
        self.num_skills = num_skills
        self.skill_rank = skill_rank
        self.top_k = top_k
        self.d_model = d_model
        self.skill_u = nn.Parameter(torch.randn(num_skills, d_model, skill_rank) * 0.01)
        self.skill_v = nn.Parameter(torch.randn(num_skills, skill_rank, d_model) * 0.01)
        self.router = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.GELU(),
            nn.Linear(d_model // 4, num_skills)
        )
        self.skill_scale = nn.Parameter(torch.ones(num_skills) * 0.01)

    def forward(self, x):
        B, T, D = x.shape
        skill_logits = self.router(x.mean(dim=1))
        skill_weights = F.softmax(skill_logits, dim=-1)
        if self.training:
            self._load_balance_loss = skill_weights.mean(dim=0).var() * 10.0
        else:
            self._load_balance_loss = torch.tensor(0.0)
        top_k = min(self.top_k, self.num_skills)
        top_weights, top_indices = skill_weights.topk(top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        perturbation = torch.zeros_like(x)
        for k in range(self.top_k):
            u = self.skill_u[top_indices[:, k]]
            v = self.skill_v[top_indices[:, k]]
            scale = torch.abs(self.skill_scale[top_indices[:, k]])
            x_v = torch.bmm(x, v.transpose(1, 2))
            perturbation += top_weights[:, k].unsqueeze(1).unsqueeze(-1) * scale.unsqueeze(1).unsqueeze(-1) * torch.bmm(x_v, u.transpose(1, 2))
        return x + perturbation


# ============ Student Model ============
CRN_PREFIXES = ('crn_mix', 'resonance.', 'skills.', 'reflection.', 'reflection_gate', 'mem.')

def get_crn_state_dict(model):
    return {k: v.cpu() for k, v in model.state_dict().items()
            if any(k.startswith(p) for p in CRN_PREFIXES)}


class PrajnaStudentMultiLayer(nn.Module):
    def __init__(self, device='cpu', inject_every=8, max_length=32, crn_mix_init=0.05,
                 num_frequencies=8, top_k=2, num_skills=32, skill_rank=4,
                 num_corrections=8, mem_size=256, mem_dim=64):
        super().__init__()
        self.device = device
        gc.collect()
        print('Loading E2B student...')
        self.tok = AutoTokenizer.from_pretrained('google/gemma-4-E2B')
        # NOTE: low_cpu_mem_usage=False materializes weights on CPU before moving
        # to the target device. With low_cpu_mem_usage=True the gemma-4-E2B
        # embedding stays a meta-tensor and .to('mps') raises
        # "Placeholder storage has not been allocated on MPS device!".
        self.base_model = AutoModelForCausalLM.from_pretrained(
            'google/gemma-4-E2B', dtype=torch.float16, low_cpu_mem_usage=False)
        for p in self.base_model.parameters():
            p.requires_grad = False
        self.vocab = 262144
        self.d_model = 1536
        self.lm = self.base_model.model.language_model
        self.num_layers = len(self.lm.layers)

        self.inject_every = inject_every
        self.inject_indices = list(range(inject_every - 1, self.num_layers, inject_every))
        self.num_injections = len(self.inject_indices)
        self.crn_mix = nn.Parameter(torch.full((self.num_injections,), crn_mix_init))
        # Separate gate for the ReflectiveLoop (self-correction pillar).
        # Higher init (0.15) than crn_mix so reflection is encouraged to learn
        # instead of collapsing to zero like before.
        self.reflection_gate = nn.Parameter(torch.full((self.num_injections,), 0.15))

        crn_dev = device
        self.mem = EpisodicMemory(self.d_model, mem_size=mem_size, mem_dim=mem_dim, device=crn_dev)
        self.reflection = ReflectiveLoop(d_model=self.d_model, num_corrections=num_corrections).to(crn_dev)
        self.skills = SkillComposer(d_model=self.d_model, num_skills=num_skills, skill_rank=skill_rank, top_k=top_k).to(crn_dev)
        self.resonance = ResonanceAttention(d_model=self.d_model, num_heads=4, num_frequencies=num_frequencies, top_k=top_k).to(crn_dev)
        print(f'CRN: {sum(p.numel() for p in self.get_params()):,} params | Injections: {self.num_injections} at {self.inject_indices}')

    def _collect_hidden(self, input_ids, past_key_values=None):
        with torch.no_grad():
            outputs = self.base_model(input_ids=input_ids, use_cache=True,
                past_key_values=past_key_values,
                output_attentions=False, output_hidden_states=True, return_dict=True)
            hs = outputs.hidden_states
            if past_key_values is None:
                collected = {idx: hs[layer_idx + 1]
                             for idx, layer_idx in enumerate(self.inject_indices)}
                final_hidden = hs[-1]
            else:
                collected = {idx: hs[layer_idx + 1][:, -1:]
                             for idx, layer_idx in enumerate(self.inject_indices)}
                final_hidden = hs[-1][:, -1:]
            past = outputs.past_key_values
            del outputs, hs
        return {'collected': collected, 'final_hidden': final_hidden, 'past': past}

    def _apply_crn(self, outputs, training=False):
        collected = outputs['collected']
        final_hidden = outputs['final_hidden']
        corrections = torch.zeros_like(final_hidden, dtype=torch.float32)
        for idx in range(self.num_injections):
            h = collected[idx].detach().to(torch.float32)
            r = self.resonance(h)
            s = self.skills(h)
            # ReflectiveLoop returns (h + correction, correction_id) only when
            # return_correction_id=True; otherwise just the corrected state.
            ref_out = self.reflection(h, return_correction_id=True)
            ref_corrected = ref_out[0] if isinstance(ref_out, tuple) else ref_out
            ref_correction = ref_corrected - h
            mix = torch.sigmoid(self.crn_mix[idx])
            ref_mix = torch.sigmoid(self.reflection_gate[idx])
            correction = mix * (r + s) + ref_mix * ref_correction
            if training:
                corrections = corrections + correction
            else:
                corrections = corrections + correction.detach()
        if self.mem.temporal_positions.sum() > 0:
            read_out, _ = self.mem.read(final_hidden.detach().mean(dim=1).to(torch.float32), top_k=8)
            corrections = corrections + read_out.unsqueeze(1)
        hidden_corrected = final_hidden.detach().to(torch.float32) + corrections
        logits = self.base_model.lm_head(hidden_corrected.to(torch.float16))
        return logits, final_hidden

    def forward(self, input_ids, labels=None):
        outputs = self._collect_hidden(input_ids)
        logits, final_hidden = self._apply_crn(outputs, training=self.training)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, self.vocab),
                                   labels[:, 1:].reshape(-1), ignore_index=-100)
        if self.training and labels is not None:
            self.mem.write(final_hidden[:, -1, :].mean(dim=0).to(torch.float32), force=False)
        return {'loss': loss, 'logits': logits}

    def get_params(self):
        params = (self.mem.get_parameters() + list(self.reflection.parameters()) +
                  list(self.skills.parameters()) + list(self.resonance.parameters()))
        params.append(self.crn_mix)
        params.append(self.reflection_gate)
        return params

    def save_memory(self, p): self.mem.save(p)
    def load_memory(self, p): self.mem.load(p)

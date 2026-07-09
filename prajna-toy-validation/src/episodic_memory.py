import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import json
import os


class EpisodicMemory:
    """
    Runtime episodic memory decoupled from model parameters.

    Key fix from analysis: memory is NOT an nn.Parameter. It's a runtime state
    tensor that can be saved/loaded independently of model weights.

    Uses learned gates for read/write and compression for efficiency.
    """

    def __init__(self, d_model, mem_size=512, mem_dim=64, device='cpu'):
        self.mem_size = mem_size
        self.mem_dim = mem_dim
        self.d_model = d_model
        self.device = device

        # Runtime state (NOT nn.Parameter)
        self.memory = torch.zeros(mem_size, mem_dim, device=device, requires_grad=False)
        self.temporal_positions = torch.zeros(mem_size, device=device, requires_grad=False)
        self.write_ptr = 0
        self.step_count = 0

        # Learnable components (these ARE model parameters)
        self.compress = nn.Linear(d_model, mem_dim)
        self.decompress = nn.Linear(mem_dim, d_model)
        self.read_gate = nn.Linear(d_model, mem_dim)
        self.write_gate = nn.Linear(d_model, 1)
        self.relevance_gate = nn.Linear(d_model + mem_dim, 1)

    def get_parameters(self):
        """Return learnable parameters for optimizer."""
        return list(self.compress.parameters()) + \
               list(self.decompress.parameters()) + \
               list(self.read_gate.parameters()) + \
               list(self.write_gate.parameters()) + \
               list(self.relevance_gate.parameters())

    def read(self, query, top_k=8):
        """
        Read from memory based on query similarity + relevance.

        Args:
            query: [batch, d_model] or [d_model] - the query vector
            top_k: number of memory slots to retrieve

        Returns:
            retrieved: [batch, d_model] - retrieved memory content
            attention: [batch, top_k] - attention weights over memory
        """
        if query.dim() == 1:
            query = query.unsqueeze(0)

        B = query.shape[0]

        # Compress query to memory space
        q_compressed = self.read_gate(query)  # [B, mem_dim]

        # Compute relevance scores
        # Expand memory for batch
        mem_expanded = self.memory.unsqueeze(0).expand(B, -1, -1)  # [B, mem_size, mem_dim]

        # Cosine similarity
        q_norm = F.normalize(q_compressed, dim=-1)
        mem_norm = F.normalize(mem_expanded, dim=-1)
        similarities = torch.bmm(q_norm.unsqueeze(1), mem_norm.transpose(1, 2)).squeeze(1)  # [B, mem_size]

        # Recency bias: newer memories get a small boost
        recency = self.temporal_positions / (self.temporal_positions.max() + 1)
        similarities = similarities + 0.1 * recency.unsqueeze(0)

        # Top-k selection
        top_k = min(top_k, self.mem_size)
        top_vals, top_idx = similarities.topk(top_k, dim=-1)  # [B, top_k]

        # Soft attention over retrieved memories
        attn_weights = F.softmax(top_vals, dim=-1)  # [B, top_k]

        # Gather and weight
        retrieved_mem = torch.gather(mem_expanded, 1, top_idx.unsqueeze(-1).expand(-1, -1, self.mem_dim))
        retrieved = (retrieved_mem * attn_weights.unsqueeze(-1)).sum(dim=1)  # [B, mem_dim]

        # Decompress back to model space
        retrieved = self.decompress(retrieved)  # [B, d_model]

        return retrieved, attn_weights

    def write(self, content, force=False):
        """
        Write content to memory.

        Args:
            content: [d_model] - the content to store
            force: if True, write regardless of gate

        Returns:
            wrote: bool - whether something was written
        """
        # Compute write gate
        gate_value = torch.sigmoid(self.write_gate(content.unsqueeze(0))).item()

        if gate_value < 0.5 and not force:
            return False

        # Compress content (detach to avoid autograd issues with inplace ops)
        compressed = self.compress(content.detach())  # [mem_dim]

        # LRU eviction: find least recently used slot
        if self.write_ptr < self.mem_size:
            slot = self.write_ptr
            self.write_ptr += 1
        else:
            # Evict oldest (lowest temporal position)
            slot = self.temporal_positions.argmin().item()

        # Write with blending (clone to avoid inplace autograd issues)
        write_weight = min(gate_value, 0.9)
        self.memory[slot] = (write_weight * compressed + (1 - write_weight) * self.memory[slot].clone()).detach()

        # Update temporal position
        self.step_count += 1
        self.temporal_positions[slot] = self.step_count

        return True

        return True

    def clear(self):
        """Reset memory (for new session)."""
        self.memory.zero_()
        self.temporal_positions.zero_()
        self.write_ptr = 0
        self.step_count = 0

    def get_stats(self):
        """Return memory statistics."""
        return {
            "used_slots": (self.temporal_positions > 0).sum().item(),
            "total_slots": self.mem_size,
            "write_ptr": self.write_ptr,
            "step_count": self.step_count,
        }

    def save(self, path):
        """Save memory state to disk (for cross-session persistence)."""
        state = {
            "memory": self.memory.detach().cpu().numpy().tolist(),
            "temporal_positions": self.temporal_positions.detach().cpu().numpy().tolist(),
            "write_ptr": self.write_ptr,
            "step_count": self.step_count,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f)

    def load(self, path):
        """Load memory state from disk."""
        with open(path, 'r') as f:
            state = json.load(f)
        self.memory = torch.tensor(state["memory"], dtype=torch.float32, device=self.device)
        self.temporal_positions = torch.tensor(state["temporal_positions"], dtype=torch.float32, device=self.device)
        self.write_ptr = state["write_ptr"]
        self.step_count = state["step_count"]

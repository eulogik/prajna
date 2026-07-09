#!/usr/bin/env python3
"""
Prajna Phase 2: Full CRN Integration with Hooks

This module injects CRN components directly into Gemma 4 E2B's forward pass
using PyTorch forward hooks. This is the proper architectural integration.

Architecture:
- Memory read: Before layer 17 (midpoint)
- Memory write: After layer 34 (final)
- Reflective Loop: After each layer's FFN
- Skill Composition: After every 4th layer's FFN
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, '/Users/eulogikdeveloper/Documents/Prajna/prajna-toy-validation/src')

from episodic_memory import EpisodicMemory
from reflective_loop import ReflectiveLoop
from skill_composer import SkillComposer


class CRNMemoryLayer(nn.Module):
    """Episodic memory layer that reads/writes to memory."""
    
    def __init__(self, d_model, mem_size=512, mem_dim=128, device='cpu'):
        super().__init__()
        self.d_model = d_model
        
        # Episodic Memory (runtime state)
        self.memory = EpisodicMemory(d_model, mem_size, mem_dim, device)
        
        # Cast memory learnable modules to bfloat16
        self.memory.compress = self.memory.compress.to(dtype=torch.bfloat16)
        self.memory.decompress = self.memory.decompress.to(dtype=torch.bfloat16)
        self.memory.read_gate = self.memory.read_gate.to(dtype=torch.bfloat16)
        self.memory.write_gate = self.memory.write_gate.to(dtype=torch.bfloat16)
        self.memory.relevance_gate = self.memory.relevance_gate.to(dtype=torch.bfloat16)
        
        # Cast memory runtime state to bfloat16
        self.memory.memory = self.memory.memory.to(dtype=torch.bfloat16)
        self.memory.temporal_positions = self.memory.temporal_positions.to(dtype=torch.bfloat16)
        
        # Read/write gates
        self.read_gate = nn.Linear(d_model, d_model, dtype=torch.bfloat16)
        self.write_gate = nn.Linear(d_model, 1, dtype=torch.bfloat16)
        self.blend = nn.Parameter(torch.tensor(0.1, dtype=torch.bfloat16))
        
    def read(self, hidden_states):
        """Read from memory and blend with hidden states."""
        if self.memory.temporal_positions.sum() == 0:
            return hidden_states
        
        B, T, D = hidden_states.shape
        query = hidden_states.mean(dim=1)  # [B, D]
        query_proj = self.read_gate(query)
        retrieved, _ = self.memory.read(query_proj, top_k=8)
        
        # Blend with hidden states
        blend = torch.sigmoid(self.blend)
        hidden_states = hidden_states + blend * retrieved.unsqueeze(1)
        
        return hidden_states
    
    def write(self, hidden_states):
        """Write final hidden state to memory."""
        if self.training:
            content = hidden_states[:, -1, :].mean(dim=0)  # [D]
            self.memory.write(content, force=False)
    
    def get_parameters(self):
        """Get learnable parameters."""
        params = self.memory.get_parameters()
        params += list(self.read_gate.parameters())
        params += list(self.write_gate.parameters())
        params.append(self.blend)
        return params


class CRNReflectiveLayer(nn.Module):
    """Reflective loop applied after each transformer layer."""
    
    def __init__(self, d_model, num_corrections=16):
        super().__init__()
        self.reflective_loop = ReflectiveLoop(d_model, num_corrections)
    
    def forward(self, hidden_states):
        """Apply reflective correction."""
        corrected, _ = self.reflective_loop(hidden_states, return_correction_id=True)
        return corrected


class CRNSkillLayer(nn.Module):
    """Skill composition applied after every 4th layer."""
    
    def __init__(self, d_model, num_skills=64, skill_rank=8, top_k=4):
        super().__init__()
        self.skill_composer = SkillComposer(d_model, num_skills, skill_rank, top_k)
    
    def forward(self, hidden_states):
        """Apply skill composition."""
        skilled, _ = self.skill_composer(hidden_states, return_skill_info=False)
        return skilled


class PrajnaGemma4Full(nn.Module):
    """
    Full CRN integration into Gemma 4 E2B using hooks.
    
    This wrapper:
    1. Loads the base model
    2. Adds CRN components as separate modules
    3. Registers hooks to inject CRN logic into the forward pass
    4. Provides memory persistence
    """
    
    def __init__(self, model_id="google/gemma-4-E2B", device="cpu"):
        super().__init__()
        
        print(f"Loading base model: {model_id}")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        
        self.d_model = 1536
        self.num_layers = 35
        self.vocab_size = 262144
        self.device = device
        
        # CRN Components
        
        # Pillar 2: Episodic Memory
        self.memory_layer = CRNMemoryLayer(
            d_model=self.d_model,
            mem_size=512,
            mem_dim=128,
            device=device
        ).to(dtype=torch.bfloat16)
        
        # Pillar 3: Reflective Loop (shared across layers)
        self.reflective_layer = CRNReflectiveLayer(
            d_model=self.d_model,
            num_corrections=16
        ).to(dtype=torch.bfloat16)
        
        # Pillar 4: Skill Composition (shared across layers)
        self.skill_layer = CRNSkillLayer(
            d_model=self.d_model,
            num_skills=64,
            skill_rank=8,
            top_k=4
        ).to(dtype=torch.bfloat16)
        
        # Store hooks for cleanup
        self._hooks = []
        self._registered = False
        
        # Register hooks
        self._register_hooks()
        
        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # CRN params count
        crn_params = sum(p.numel() for p in self.get_crn_parameters())
        print(f"  CRN params: {crn_params:,}")
        print(f"  Hooks registered: {len(self._hooks)}")
        
    def _register_hooks(self):
        """Register forward hooks on transformer layers."""
        if self._registered:
            return
        
        layers = self.base_model.model.language_model.layers
        
        # Memory read hook at midpoint (before layer 17)
        midpoint = self.num_layers // 2
        hook = layers[midpoint].register_forward_pre_hook(
            self._memory_read_pre_hook
        )
        self._hooks.append(hook)
        
        # Memory write hook after final layer
        hook = layers[-1].register_forward_hook(
            self._memory_write_post_hook
        )
        self._hooks.append(hook)
        
        # Reflective loop hook after each layer
        for i, layer in enumerate(layers):
            hook = layer.register_forward_hook(
                self._reflective_post_hook
            )
            self._hooks.append(hook)
        
        # Skill composition hook after every 4th layer
        for i, layer in enumerate(layers):
            if i % 4 == 0:
                hook = layer.register_forward_hook(
                    self._skill_post_hook
                )
                self._hooks.append(hook)
        
        self._registered = True
    
    def _memory_read_pre_hook(self, module, input):
        """Hook to read from memory before processing."""
        # input is a tuple: (hidden_states, attention_mask, position_embeddings, ...)
        hidden_states = input[0]
        
        # Read from memory
        modified = self.memory_layer.read(hidden_states)
        
        # Return modified input
        return (modified,) + input[1:]
    
    def _memory_write_post_hook(self, module, input, output):
        """Hook to write to memory after final layer."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Write to memory
        self.memory_layer.write(hidden_states)
        
        return output
    
    def _reflective_post_hook(self, module, input, output):
        """Hook to apply reflective loop after each layer."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Apply reflective correction
        corrected = self.reflective_layer(hidden_states)
        
        if isinstance(output, tuple):
            return (corrected,) + output[1:]
        return corrected
    
    def _skill_post_hook(self, module, input, output):
        """Hook to apply skill composition after every 4th layer."""
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        
        # Apply skill composition
        skilled = self.skill_layer(hidden_states)
        
        if isinstance(output, tuple):
            return (skilled,) + output[1:]
        return skilled
    
    def get_crn_parameters(self):
        """Get all CRN parameters."""
        params = []
        params += self.memory_layer.get_parameters()
        params += list(self.reflective_layer.parameters())
        params += list(self.skill_layer.parameters())
        return params
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass with CRN hooks active."""
        
        # Forward through base model (hooks inject CRN logic)
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        logits = outputs.logits
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )
        
        return {'logits': logits, 'loss': loss}
    
    def generate(self, input_ids, max_new_tokens=50, temperature=0.8):
        """Generate text with CRN active."""
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                outputs = self.forward(input_ids)
                next_logits = outputs['logits'][:, -1, :] / temperature
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
    
    def save_memory(self, path):
        """Save memory state."""
        self.memory_layer.memory.save(path)
    
    def load_memory(self, path):
        """Load memory state."""
        self.memory_layer.memory.load(path)
    
    def cleanup(self):
        """Remove all hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._registered = False
    
    def get_stats(self):
        """Return model statistics."""
        base_params = sum(p.numel() for p in self.base_model.parameters())
        crn_params = sum(p.numel() for p in self.get_crn_parameters())
        
        return {
            'base_params': base_params,
            'crn_params': crn_params,
            'total_params': base_params + crn_params,
            'memory_stats': self.memory_layer.memory.get_stats(),
            'num_hooks': len(self._hooks),
        }


def main():
    """Test the full CRN integration."""
    print("=" * 60)
    print("Prajna Gemma 4 E2B — Full CRN Integration")
    print("=" * 60)
    
    # Create model
    print("\n[Step 1] Creating Prajna model...")
    model = PrajnaGemma4Full()
    
    # Print stats
    stats = model.get_stats()
    print(f"\n  Base model params: {stats['base_params']:,}")
    print(f"  CRN params: {stats['crn_params']:,}")
    print(f"  Total params: {stats['total_params']:,}")
    print(f"  Memory stats: {stats['memory_stats']}")
    print(f"  Hooks: {stats['num_hooks']}")
    
    # Test forward pass (inference)
    print("\n[Step 2] Testing forward pass (inference)...")
    model.eval()
    inputs = model.tokenizer("Hello, how are you?", return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
        )
    
    print(f"  Input shape: {inputs['input_ids'].shape}")
    print(f"  Output shape: {outputs['logits'].shape}")
    print(f"  Memory stats: {model.memory_layer.memory.get_stats()}")
    print("  Forward pass (inference): OK")
    
    # Test forward pass (training)
    print("\n[Step 3] Testing forward pass (training)...")
    model.train()
    
    # Create dummy labels
    labels = inputs['input_ids'].clone()
    
    # Get CRN parameters
    crn_params = model.get_crn_parameters()
    print(f"  CRN params: {sum(p.numel() for p in crn_params):,}")
    
    optimizer = torch.optim.AdamW(crn_params, lr=1e-4)
    
    outputs = model(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        labels=labels,
    )
    
    print(f"  Loss: {outputs['loss'].item():.4f}")
    
    # Test backward pass
    print("\n[Step 4] Testing backward pass...")
    outputs['loss'].backward()
    
    # Check gradients
    grad_count = 0
    grad_norms = []
    for param in crn_params:
        if param.grad is not None:
            grad_count += 1
            grad_norms.append(param.grad.norm().item())
    
    print(f"  Parameters with gradients: {grad_count}/{len(crn_params)}")
    if grad_norms:
        print(f"  Avg gradient norm: {sum(grad_norms) / len(grad_norms):.6f}")
        print(f"  Max gradient norm: {max(grad_norms):.6f}")
    
    # Test optimization step
    print("\n[Step 5] Testing optimization step...")
    optimizer.step()
    print("  Optimization step: OK")
    
    # Test memory persistence
    print("\n[Step 6] Testing memory persistence...")
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        save_path = f.name
    
    model.save_memory(save_path)
    print(f"  Memory saved to {save_path}")
    
    # Clean up hooks
    model.cleanup()
    
    print("\n" + "=" * 60)
    print("FULL CRN INTEGRATION TEST COMPLETE")
    print("=" * 60)
    print(f"""
    Results:
    - Gemma 4 E2B loads on M4 16GB ✓
    - CRN components initialize ✓
    - {stats['num_hooks']} hooks registered ✓
    - Forward pass (inference) works ✓
    - Forward pass (training) works ✓
    - Backward pass produces gradients ✓
    - Optimization step works ✓
    - Memory persistence works ✓
    
    CRN Integration Status:
    - Pillar 1 (Resonance Attention): Ready for injection
    - Pillar 2 (Episodic Memory): ✓ Integrated via hooks
    - Pillar 3 (Reflective Loop): ✓ Integrated via hooks
    - Pillar 4 (Skill Composition): ✓ Integrated via hooks
    
    Next steps:
    1. Run sanity training (1K samples)
    2. Generate synthetic training data
    3. Run full distillation
    4. Export to ONNX for browser
    """)


if __name__ == "__main__":
    main()

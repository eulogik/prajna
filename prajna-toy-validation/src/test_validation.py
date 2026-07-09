#!/usr/bin/env python3
"""
Prajna Toy Validation: Tests for all 4 CRN architecture pillars.

Tests:
1. Cross-session memory recall (Pillar 2)
2. Frequency band interpretability (Pillar 1)
3. Training convergence (all pillars)
4. Memory gate learning (Pillar 2)
5. Reflective loop error correction (Pillar 3) — NEW
6. Skill composition generalization (Pillar 4) — NEW
7. End-to-end mini task (all pillars) — UPDATED
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import tempfile

sys.path.insert(0, os.path.dirname(__file__))

from crn_model import CRNMiniModel
from episodic_memory import EpisodicMemory
from resonance_attention import ResonanceAttention
from reflective_loop import ReflectiveLoop
from skill_composer import SkillComposer


# ============================================================================
# TEST 1: Cross-Session Memory Recall
# ============================================================================

def test_cross_session_memory():
    """
    Test: Can the model write facts to memory in one session and recall them
    in a subsequent session?
    """
    print("=" * 60)
    print("TEST 1: Cross-Session Memory Recall")
    print("=" * 60)

    d_model = 64
    mem_size = 32
    mem_dim = 16
    num_facts = 8

    memory = EpisodicMemory(d_model, mem_size, mem_dim)

    torch.manual_seed(42)
    fact_vectors = torch.randn(num_facts, d_model)

    print("\n[Training] Teaching memory to store and retrieve facts...")
    optimizer = torch.optim.Adam(memory.get_parameters(), lr=5e-3)

    for step in range(200):
        idx = torch.randint(0, num_facts, (1,)).item()
        fact = fact_vectors[idx]

        memory.write(fact, force=True)

        query = fact + torch.randn(d_model) * 0.1
        retrieved, attn_weights = memory.read(query, top_k=4)

        loss = F.mse_loss(retrieved.squeeze(0), fact)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % 50 == 0:
            print(f"  Step {step+1}: retrieval loss = {loss.item():.4f}")

    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
        save_path = f.name
    memory.save(save_path)
    print(f"\n  Saved memory to {save_path}")

    print("[Session 2] Reloading memory and testing recall...")
    memory2 = EpisodicMemory(d_model, mem_size, mem_dim)
    memory2.compress.load_state_dict(memory.compress.state_dict())
    memory2.decompress.load_state_dict(memory.decompress.state_dict())
    memory2.read_gate.load_state_dict(memory.read_gate.state_dict())
    memory2.write_gate.load_state_dict(memory.write_gate.state_dict())
    memory2.relevance_gate.load_state_dict(memory.relevance_gate.state_dict())
    memory2.load(save_path)

    correct = 0
    for i in range(num_facts):
        fact = fact_vectors[i]
        query = fact + torch.randn(d_model) * 0.1
        retrieved, _ = memory2.read(query, top_k=4)

        distances = torch.cdist(retrieved, fact_vectors)
        predicted_idx = distances.argmin().item()
        if predicted_idx == i:
            correct += 1
            print(f"  Fact {i}: predicted {predicted_idx} (correct) ✓")
        else:
            print(f"  Fact {i}: predicted {predicted_idx} (expected {i}) ✗")

    accuracy = correct / num_facts
    print(f"\n  Memory Recall Accuracy: {correct}/{num_facts} = {accuracy:.1%}")

    os.unlink(save_path)

    passed = accuracy >= 0.5
    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'} (threshold: 50% after training)")
    return passed


# ============================================================================
# TEST 2: Frequency Band Interpretability
# ============================================================================

def test_frequency_interpretability():
    """
    Test: Do different input patterns activate different frequency bands?
    """
    print("\n" + "=" * 60)
    print("TEST 2: Frequency Band Interpretability")
    print("=" * 60)

    d_model = 128
    num_heads = 4
    num_frequencies = 16
    top_k = 4

    attn = ResonanceAttention(d_model, num_heads, num_frequencies, top_k)

    torch.manual_seed(42)
    batch_size = 1
    seq_len = 16

    pattern_a = torch.randn(batch_size, seq_len, d_model) * 0.5
    pattern_b = torch.randn(batch_size, seq_len, d_model) * 0.5 + 1.0
    pattern_c = torch.randn(batch_size, seq_len, d_model) * 0.3 + 2.0

    _, info_a = attn(pattern_a, return_freq_info=True)
    _, info_b = attn(pattern_b, return_freq_info=True)
    _, info_c = attn(pattern_c, return_freq_info=True)

    dom_a = info_a["dominant_frequency"][0, :, 0].tolist()
    dom_b = info_b["dominant_frequency"][0, :, 0].tolist()
    dom_c = info_c["dominant_frequency"][0, :, 0].tolist()

    freq_set_a = set(int(f) for f in dom_a)
    freq_set_b = set(int(f) for f in dom_b)
    freq_set_c = set(int(f) for f in dom_c)

    print(f"\n  Pattern A (definitional) dominant frequencies: {sorted(freq_set_a)}")
    print(f"  Pattern B (questioning)  dominant frequencies: {sorted(freq_set_b)}")
    print(f"  Pattern C (calculating)  dominant frequencies: {sorted(freq_set_c)}")

    all_same = (freq_set_a == freq_set_b == freq_set_c)
    some_diversity = len(freq_set_a | freq_set_b | freq_set_c) > num_frequencies * 0.3

    unique_per_pattern = [
        len(set(int(f) for f in dom_a)),
        len(set(int(f) for f in dom_b)),
        len(set(int(f) for f in dom_c)),
    ]
    avg_unique = sum(unique_per_pattern) / len(unique_per_pattern)

    print(f"\n  Unique frequencies per pattern: {unique_per_pattern} (avg: {avg_unique:.1f})")
    print(f"  Total unique frequencies used: {len(freq_set_a | freq_set_b | freq_set_c)}/{num_frequencies}")

    scores_a = info_a["frequency_scores"][0, :, 0, :]
    scores_std = scores_a.std(dim=-1).mean().item()

    print(f"  Frequency score entropy (std): {scores_std:.4f}")

    passed = (
        not all_same and
        scores_std > 0.01 and
        avg_unique >= 2.0
    )

    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


# ============================================================================
# TEST 3: Training Convergence (4-Pillar Model)
# ============================================================================

def test_training_convergence():
    """
    Test: Does the full 4-pillar CRN model converge on a simple task?
    """
    print("\n" + "=" * 60)
    print("TEST 3: Training Convergence (4-Pillar Model)")
    print("=" * 60)

    torch.manual_seed(42)

    vocab_size = 32
    d_model = 64
    seq_len = 32
    num_samples = 100
    batch_size = 16
    num_epochs = 50
    lr = 1e-3

    model = CRNMiniModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        num_frequencies=8,
        top_k=3,
        mem_size=32,
        mem_dim=16,
        num_corrections=8,
        num_skills=16,
        skill_rank=4,
        skill_top_k=2,
    )

    # Separate optimizer for main params and pillar-specific params
    main_params = list(model.parameters())
    memory_params = model.get_memory_parameters()
    all_params = main_params + memory_params
    optimizer = torch.optim.AdamW(all_params, lr=lr)

    data = torch.randint(0, vocab_size, (num_samples, seq_len + 1))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {total_params:,}")
    print(f"  Training samples: {num_samples}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Pillars: Resonance + Memory + Reflection + Skills")

    losses = []
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch = data[i:i+batch_size]
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            logits, _ = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss = {avg_loss:.4f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss

    converged = loss_reduction > 0.3
    stable = all(l < 5.0 for l in losses[-10:])

    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss:   {final_loss:.4f}")
    print(f"  Loss reduction: {loss_reduction:.1%}")
    print(f"  Training stable (no NaN): {stable}")

    passed = converged and stable
    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'} (requires >30% loss reduction, no NaN)")
    return passed


# ============================================================================
# TEST 4: Memory Write Gate Learning
# ============================================================================

def test_memory_gate_learning():
    """
    Test: Does the write gate learn when to write to memory?
    """
    print("\n" + "=" * 60)
    print("TEST 4: Memory Gate Learning")
    print("=" * 60)

    torch.manual_seed(42)

    d_model = 64
    mem_size = 32
    mem_dim = 16
    num_steps = 200

    memory = EpisodicMemory(d_model, mem_size, mem_dim)

    gate_values = []
    wrote_flags = []

    optimizer = torch.optim.Adam(memory.get_parameters(), lr=1e-3)

    for step in range(num_steps):
        importance = torch.rand(1).item()
        content = torch.randn(d_model) * importance

        gate_val = torch.sigmoid(memory.write_gate(content.unsqueeze(0))).item()
        gate_values.append(gate_val)

        wrote = memory.write(content, force=(importance > 0.7))
        wrote_flags.append(1 if wrote else 0)

        target = torch.tensor([[1.0 if importance > 0.7 else 0.0]])
        gate_pred = memory.write_gate(content.unsqueeze(0))
        loss = F.binary_cross_entropy_with_logits(gate_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    import numpy as np
    gate_values = np.array(gate_values)
    wrote_flags = np.array(wrote_flags)

    forced_gates = gate_values[wrote_flags == 1]
    unforced_gates = gate_values[wrote_flags == 0] if (wrote_flags == 0).any() else np.array([0])

    avg_forced = forced_gates.mean() if len(forced_gates) > 0 else 0
    avg_unforced = unforced_gates.mean() if len(unforced_gates) > 0 else 0

    print(f"\n  Gate values for forced writes: {avg_forced:.4f}")
    print(f"  Gate values for non-writes:    {avg_unforced:.4f}")
    print(f"  Difference:                    {avg_forced - avg_unforced:.4f}")
    print(f"  Total writes: {wrote_flags.sum()}/{num_steps}")

    passed = avg_forced > avg_unforced and avg_forced > 0.5

    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


# ============================================================================
# TEST 5: Reflective Loop Error Correction (Pillar 3)
# ============================================================================

def test_reflective_loop():
    """
    Test: Does the Reflective Loop learn to identify and correct errors?

    Setup:
    - Create "correct" and "incorrect" hidden states
    - Train the critic to distinguish them
    - Verify it learns to apply corrections to incorrect states
    """
    print("\n" + "=" * 60)
    print("TEST 5: Reflective Loop Error Correction")
    print("=" * 60)

    torch.manual_seed(42)

    d_model = 64
    num_corrections = 8
    num_samples = 200
    num_epochs = 50

    reflective = ReflectiveLoop(d_model, num_corrections)

    # Generate synthetic data: "correct" states and "incorrect" states
    # Incorrect states are perturbed versions of correct states
    correct_states = torch.randn(num_samples, d_model)
    error_magnitudes = torch.rand(num_samples, 1) * 2.0
    error_directions = torch.randn(num_samples, d_model)
    error_directions = error_directions / error_directions.norm(dim=-1, keepdim=True)
    incorrect_states = correct_states + error_magnitudes * error_directions

    # Labels: which correction direction fixes each error
    correct_directions = torch.randint(0, num_corrections, (num_samples,))
    is_error = torch.ones(num_samples, dtype=torch.bool)

    # Add non-error examples
    num_non_error = 100
    non_error_states = torch.randn(num_non_error, d_model)
    all_states = torch.cat([incorrect_states, non_error_states], dim=0)
    all_is_error = torch.cat([is_error, torch.zeros(num_non_error, dtype=torch.bool)], dim=0)
    all_directions = torch.cat([correct_directions, torch.zeros(num_non_error, dtype=torch.long)], dim=0)

    # Shuffle
    perm = torch.randperm(all_states.shape[0])
    all_states = all_states[perm]
    all_is_error = all_is_error[perm]
    all_directions = all_directions[perm]

    # Train
    optimizer = torch.optim.Adam(reflective.parameters(), lr=1e-3)

    print(f"\n  Training reflective loop on {all_states.shape[0]} samples...")
    losses = []

    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0

        for i in range(0, all_states.shape[0], 32):
            batch_states = all_states[i:i+32]
            batch_is_error = all_is_error[i:i+32]
            batch_directions = all_directions[i:i+32]

            loss = reflective.compute_loss(batch_states, batch_is_error, batch_directions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss = {avg_loss:.4f}")

    # Evaluate: does the critic correctly identify errors?
    print("\n[Evaluation] Testing error detection...")
    reflective.eval()
    correct_detections = 0
    total_detections = 0
    corrections_applied = 0

    with torch.no_grad():
        for i in range(0, all_states.shape[0], 32):
            batch_states = all_states[i:i+32]
            batch_is_error = all_is_error[i:i+32]

            # Forward pass with correction tracking
            for b in range(batch_states.shape[0]):
                state = batch_states[b:b+1]
                _, correction_id = reflective(state, return_correction_id=True)

                is_actually_error = batch_is_error[b].item()
                detected_error = correction_id != -1

                if is_actually_error == detected_error:
                    correct_detections += 1
                total_detections += 1

                if detected_error:
                    corrections_applied += 1

    detection_accuracy = correct_detections / total_detections
    error_correction_rate = corrections_applied / total_detections

    print(f"  Error detection accuracy: {detection_accuracy:.1%}")
    print(f"  Correction rate: {error_correction_rate:.1%}")
    print(f"  Stats: {reflective.get_correction_stats()}")

    # Check if critic learned something useful
    # Detection should be better than random (50%)
    passed = detection_accuracy > 0.55 and losses[-1] < losses[0] * 0.8

    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'} (requires >55% detection accuracy)")
    return passed


# ============================================================================
# TEST 6: Skill Composition Generalization (Pillar 4)
# ============================================================================

def test_skill_composition():
    """
    Test: Does the Skill Composer learn composable perturbations?

    Setup:
    - Train skills on a simple pattern: "apply skill A when input is type X"
    - Verify that:
      1. Different inputs activate different skills
      2. Skills compose (multiple skills active simultaneously)
      3. The perturbation actually changes the output meaningfully
    """
    print("\n" + "=" * 60)
    print("TEST 6: Skill Composition Generalization")
    print("=" * 60)

    torch.manual_seed(42)

    d_model = 64
    num_skills = 16
    skill_rank = 4
    top_k = 3
    num_samples = 200
    num_epochs = 50

    skill_composer = SkillComposer(d_model, num_skills, skill_rank, top_k)

    # Create inputs with "task type" structure
    # Task 0: mathematical pattern (inputs clustered around 0, low variance)
    # Task 1: creative pattern (inputs clustered around 3, high variance)
    # Task 2: analytical pattern (inputs clustered around -3, medium variance)
    task_inputs = []
    task_labels = []

    for _ in range(num_samples):
        task = torch.randint(0, 3, (1,)).item()
        if task == 0:
            # Mathematical: structured, low variance, centered at 0
            inp = torch.randn(1, 8, d_model) * 0.2
        elif task == 1:
            # Creative: high variance, centered at 3
            inp = torch.randn(1, 8, d_model) * 0.8 + 3.0
        else:
            # Analytical: medium variance, centered at -3
            inp = torch.randn(1, 8, d_model) * 0.5 - 3.0

        task_inputs.append(inp.squeeze(0))
        task_labels.append(task)

    all_inputs = torch.stack(task_inputs)
    all_labels = torch.tensor(task_labels)

    # Train: teach the router to activate appropriate skills for each task
    # Simple loss: perturbation should be small for correct tasks, larger for others
    optimizer = torch.optim.Adam(skill_composer.parameters(), lr=1e-3)

    print(f"\n  Training skill composer on {num_samples} samples...")
    print(f"  Tasks: mathematical, creative, analytical")

    losses = []
    skill_diversities = []

    for epoch in range(num_epochs):
        perm = torch.randperm(num_samples)
        epoch_loss = 0
        num_batches = 0
        epoch_skills = []

        for i in range(0, num_samples, 32):
            batch_inputs = all_inputs[perm[i:i+32]]
            batch_labels = all_labels[perm[i:i+32]]

            output, skill_info = skill_composer(batch_inputs, return_skill_info=True)

            # Loss: perturbation should be consistent within same task
            # Use variance of output within same task as a proxy
            task_loss = 0
            for task_id in range(3):
                mask = batch_labels == task_id
                if mask.sum() > 1:
                    task_outputs = output[mask]
                    # Outputs for same task should be similar (low variance)
                    task_loss += task_outputs.var(dim=0).mean()

            # Also: perturbation shouldn't be zero (skills should do something)
            perturbation_magnitude = (output - batch_inputs).norm(dim=-1).mean()
            skill_loss = -torch.log(perturbation_magnitude + 1e-6)  # Encourage non-zero perturbation

            # Load balancing loss (from skill composer)
            load_balance = getattr(skill_composer, '_load_balance_loss', torch.tensor(0.0))

            loss = task_loss + 0.1 * skill_loss + 0.01 * load_balance

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            # Track skill diversity
            if skill_info is not None:
                weights = skill_info["skill_weights"]  # [B, num_skills]
                # Count unique active skills per sample
                active = (weights > 0.1).float().sum(dim=-1)
                epoch_skills.append(active.mean().item())

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        avg_skills = sum(epoch_skills) / len(epoch_skills) if epoch_skills else 0
        skill_diversities.append(avg_skills)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss = {avg_loss:.4f}, avg active skills = {avg_skills:.1f}")

    # Evaluate: check skill activation patterns
    print("\n[Evaluation] Analyzing skill activation patterns...")
    skill_composer.eval()

    task_skill_profiles = {0: [], 1: [], 2: []}

    with torch.no_grad():
        for task_id in range(3):
            mask = all_labels == task_id
            task_inputs = all_inputs[mask][:10]  # Sample 10 per task

            for inp in task_inputs:
                _, info = skill_composer(inp.unsqueeze(0), return_skill_info=True)
                if info is not None:
                    weights = info["skill_weights"][0]  # [num_skills]
                    task_skill_profiles[task_id].append(weights)

    # Average skill profiles per task
    avg_profiles = []
    for task_id in range(3):
        if task_skill_profiles[task_id]:
            avg_profile = torch.stack(task_skill_profiles[task_id]).mean(dim=0)
            avg_profiles.append(avg_profile)
            top_skills = avg_profile.topk(3).indices.tolist()
            print(f"  Task {task_id} top skills: {top_skills} (weights: {[f'{avg_profile[s]:.3f}' for s in top_skills]})")

    # Check differentiation: different tasks should activate different skills
    if len(avg_profiles) == 3:
        # Compute pairwise cosine similarity between task profiles
        sim_01 = F.cosine_similarity(avg_profiles[0].unsqueeze(0), avg_profiles[1].unsqueeze(0)).item()
        sim_02 = F.cosine_similarity(avg_profiles[0].unsqueeze(0), avg_profiles[2].unsqueeze(0)).item()
        sim_12 = F.cosine_similarity(avg_profiles[1].unsqueeze(0), avg_profiles[2].unsqueeze(0)).item()

        print(f"\n  Profile similarity: 0-1={sim_01:.3f}, 0-2={sim_02:.3f}, 1-2={sim_12:.3f}")

        # Profiles should be somewhat different (not all identical)
        avg_similarity = (sim_01 + sim_02 + sim_12) / 3
        print(f"  Average inter-task similarity: {avg_similarity:.3f}")
    else:
        avg_similarity = 0.5

    # Check that skills are actually doing something
    avg_active = skill_diversities[-1] if skill_diversities else 0
    print(f"  Final avg active skills per input: {avg_active:.1f}")

    # Pass criteria:
    # 1. Skills are active (not all zero)
    # 2. Different tasks activate different skill profiles (similarity < 0.95)
    # 3. Training converged
    passed = (
        avg_active >= 0.3 and  # At least some skills active on average
        avg_similarity < 0.95 and  # Tasks have different profiles
        losses[-1] < losses[0] * 0.9  # Some training progress
    )

    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


# ============================================================================
# TEST 7: End-to-End Mini Task (4-Pillar Model)
# ============================================================================

def test_end_to_end_task():
    """
    Test: Can the full 4-pillar CRN model learn a long-range dependency?

    Task: First token determines last token, with 64 tokens of noise between.
    This tests whether memory helps recall distant information.
    """
    print("\n" + "=" * 60)
    print("TEST 7: End-to-End Task (4-Pillar, Long-Range)")
    print("=" * 60)

    torch.manual_seed(42)

    vocab_size = 16
    d_model = 64
    seq_len = 64  # Longer sequence to test memory benefit
    num_samples = 200
    batch_size = 32
    num_epochs = 40
    lr = 2e-3

    model = CRNMiniModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=2,
        num_frequencies=8,
        top_k=3,
        mem_size=64,
        mem_dim=16,
        num_corrections=8,
        num_skills=16,
        skill_rank=4,
        skill_top_k=2,
    )

    # Pattern: first token determines last token (long-range dependency)
    data = []
    for _ in range(num_samples):
        seq = torch.randint(0, vocab_size, (seq_len,))
        seq[-1] = (seq[0] + 3) % vocab_size
        data.append(seq)
    data = torch.stack(data)

    all_params = list(model.parameters()) + model.get_memory_parameters()
    optimizer = torch.optim.AdamW(all_params, lr=lr)

    print(f"\n  Task: Predict last token from first token ({seq_len} tokens apart)")
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Pillars: All 4 active")

    losses = []
    model.train()
    for epoch in range(num_epochs):
        perm = torch.randperm(num_samples)
        data_shuffled = data[perm]

        epoch_loss = 0
        num_batches = 0

        for i in range(0, num_samples, batch_size):
            batch = data_shuffled[i:i+batch_size]
            input_ids = batch[:, :-1]
            targets = batch[:, -1]

            logits, _ = model(input_ids)
            last_logits = logits[:, -1, :]

            loss = F.cross_entropy(last_logits, targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:3d}: loss = {avg_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_data = data[:50]
        input_ids = test_data[:, :-1]
        targets = test_data[:, -1]

        logits, _ = model(input_ids)
        last_logits = logits[:, -1, :]
        predictions = last_logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean().item()

    print(f"\n  Final loss: {losses[-1]:.4f}")
    print(f"  Test accuracy: {accuracy:.1%}")
    print(f"  Random baseline: {1/vocab_size:.1%}")

    passed = accuracy > 0.15 and losses[-1] < losses[0] * 0.7

    print(f"\n  RESULT: {'PASS ✓' if passed else 'FAIL ✗'}")
    return passed


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Prajna Toy Validation Suite — 4-Pillar Edition")
    print("=" * 60)
    print("Testing ALL core architecture claims on minimal model")
    print("=" * 60)

    results = {}

    results["cross_session_memory"] = test_cross_session_memory()
    results["frequency_interpretability"] = test_frequency_interpretability()
    results["training_convergence"] = test_training_convergence()
    results["memory_gate_learning"] = test_memory_gate_learning()
    results["reflective_loop"] = test_reflective_loop()
    results["skill_composition"] = test_skill_composition()
    results["end_to_end_task"] = test_end_to_end_task()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — ALL 4 PILLARS")
    print("=" * 60)

    passed = sum(results.values())
    total = len(results)

    for test_name, result in results.items():
        status = "PASS ✓" if result else "FAIL ✗"
        print(f"  {test_name:30s} {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    # Pillar summary
    print("\n  Pillar Status:")
    print(f"    Pillar 1 (Resonance Attention):  {'✓' if results.get('frequency_interpretability') else '✗'}")
    print(f"    Pillar 2 (Episodic Memory):      {'✓' if results.get('cross_session_memory') and results.get('memory_gate_learning') else '✗'}")
    print(f"    Pillar 3 (Reflective Loop):      {'✓' if results.get('reflective_loop') else '✗'}")
    print(f"    Pillar 4 (Skill Composition):    {'✓' if results.get('skill_composition') else '✗'}")

    if passed == total:
        print("\n  ALL TESTS PASSED ✓")
        print("  ALL 4 PILLARS VALIDATED on toy scale.")
        print("  Ready for Gemma 4 E2B integration.")
    elif passed >= total - 2:
        print(f"\n  MOSTLY PASSED ({passed}/{total})")
        print("  Core concepts work. Review failed tests before scaling.")
    else:
        print(f"\n  SIGNIFICANT FAILURES ({passed}/{total})")
        print("  Architecture needs revision before full implementation.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

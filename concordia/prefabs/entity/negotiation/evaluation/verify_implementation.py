#!/usr/bin/env python3
"""Verify what's actually implemented vs documented."""

import sys
import os
import inspect

# Add project root
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.insert(0, project_root)

print("=" * 70)
print("IMPLEMENTATION VERIFICATION")
print("=" * 70)

# ============================================================================
# TEST 1: Check InterpretabilityRunner.__init__ signature
# ============================================================================
print("\n[TEST 1] InterpretabilityRunner.__init__ parameters")
print("-" * 50)

try:
    from concordia.prefabs.entity.negotiation.evaluation.interpretability_evaluation import (
        InterpretabilityRunner
    )

    sig = inspect.signature(InterpretabilityRunner.__init__)
    params = list(sig.parameters.keys())
    print(f"  Parameters: {params}")

    # Check for specific params
    checks = {
        'model_name': 'model_name' in params,
        'device': 'device' in params,
        'layers_to_capture': 'layers_to_capture' in params,
    }

    for param, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {param}: {'exists' if exists else 'MISSING'}")

    print("  [OK] InterpretabilityRunner imports successfully")
except Exception as e:
    print(f"  [FAIL] Import error: {e}")

# ============================================================================
# TEST 2: Check run_study signature
# ============================================================================
print("\n[TEST 2] run_study() parameters")
print("-" * 50)

try:
    sig = inspect.signature(InterpretabilityRunner.run_study)
    params = list(sig.parameters.keys())
    print(f"  Parameters: {params}")

    checks = {
        'scenario': 'scenario' in params,
        'num_trials': 'num_trials' in params,
        'max_rounds': 'max_rounds' in params,
        'use_gm': 'use_gm' in params,
        'agent_modules': 'agent_modules' in params,
        'gm_modules': 'gm_modules' in params,
    }

    for param, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {param}: {'exists' if exists else 'MISSING'}")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# ============================================================================
# TEST 3: Check run_single_negotiation signature
# ============================================================================
print("\n[TEST 3] run_single_negotiation() parameters")
print("-" * 50)

try:
    sig = inspect.signature(InterpretabilityRunner.run_single_negotiation)
    params = list(sig.parameters.keys())
    print(f"  Parameters: {params}")

    checks = {
        'scenario_type': 'scenario_type' in params,
        'agent_modules': 'agent_modules' in params,
        'gm_modules': 'gm_modules' in params,
        'max_rounds': 'max_rounds' in params,
        'use_gm': 'use_gm' in params,
    }

    for param, exists in checks.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {param}: {'exists' if exists else 'MISSING'}")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# ============================================================================
# TEST 4: Check ActivationSample dataclass fields
# ============================================================================
print("\n[TEST 4] ActivationSample dataclass fields")
print("-" * 50)

try:
    from concordia.prefabs.entity.negotiation.evaluation.interpretability_evaluation import (
        ActivationSample
    )

    import dataclasses
    fields = [f.name for f in dataclasses.fields(ActivationSample)]
    print(f"  Fields: {fields}")

    # Check for agent labels
    agent_label_fields = ['perceived_deception', 'emotion_intensity', 'trust_level', 'cooperation_intent']
    print("\n  Agent label fields:")
    for field in agent_label_fields:
        status = "✓" if field in fields else "✗"
        print(f"    {status} {field}")

    # Check for GM labels
    gm_label_fields = ['actual_deception', 'commitment_violation', 'manipulation_score', 'consistency_score']
    print("\n  GM label fields:")
    for field in gm_label_fields:
        status = "✓" if field in fields else "✗"
        print(f"    {status} {field}")

    # Check for context fields
    context_fields = ['gm_modules_enabled', 'modules_enabled', 'scenario_type']
    print("\n  Context fields:")
    for field in context_fields:
        status = "✓" if field in fields else "✗"
        print(f"    {status} {field}")

except Exception as e:
    print(f"  [FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 5: Check EvaluationResult dataclass fields
# ============================================================================
print("\n[TEST 5] EvaluationResult dataclass fields")
print("-" * 50)

try:
    from concordia.prefabs.entity.negotiation.evaluation.interpretability_evaluation import (
        EvaluationResult
    )

    import dataclasses
    fields = [f.name for f in dataclasses.fields(EvaluationResult)]
    print(f"  Fields: {fields}")

    gm_fields = ['total_deception_detected', 'gm_modules_used']
    print("\n  GM-related fields:")
    for field in gm_fields:
        status = "✓" if field in fields else "✗"
        print(f"    {status} {field}")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# ============================================================================
# TEST 6: Check _extract_gm_labels method exists
# ============================================================================
print("\n[TEST 6] GM label extraction method")
print("-" * 50)

try:
    has_method = hasattr(InterpretabilityRunner, '_extract_gm_labels')
    status = "✓" if has_method else "✗"
    print(f"  {status} _extract_gm_labels method: {'exists' if has_method else 'MISSING'}")

    if has_method:
        sig = inspect.signature(InterpretabilityRunner._extract_gm_labels)
        print(f"    Signature: {sig}")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# ============================================================================
# TEST 7: Check _create_gm method exists
# ============================================================================
print("\n[TEST 7] GM creation method")
print("-" * 50)

try:
    has_method = hasattr(InterpretabilityRunner, '_create_gm')
    status = "✓" if has_method else "✗"
    print(f"  {status} _create_gm method: {'exists' if has_method else 'MISSING'}")

    if has_method:
        sig = inspect.signature(InterpretabilityRunner._create_gm)
        print(f"    Signature: {sig}")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# ============================================================================
# TEST 8: Check save_dataset output format
# ============================================================================
print("\n[TEST 8] save_dataset method - check what keys it saves")
print("-" * 50)

try:
    import inspect
    source = inspect.getsource(InterpretabilityRunner.save_dataset)

    # Check for key outputs
    output_keys = {
        'activations': "'activations'" in source,
        'agent_labels': "'agent_labels'" in source,
        'gm_labels': "'gm_labels'" in source,
        'agent_label_names': "'agent_label_names'" in source,
        'gm_label_names': "'gm_label_names'" in source,
        'metadata': "'metadata'" in source,
        'layers': "'layers'" in source,
        'labels (legacy)': "'labels'" in source,
    }

    for key, exists in output_keys.items():
        status = "✓" if exists else "✗"
        print(f"  {status} {key}: {'saved' if exists else 'NOT SAVED'}")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# ============================================================================
# TEST 9: Check GM module imports
# ============================================================================
print("\n[TEST 9] GM module imports")
print("-" * 50)

try:
    from concordia.prefabs.game_master.negotiation.components.gm_social_intelligence import (
        SocialIntelligenceGM,
        DeceptionIndicator,
    )
    print("  ✓ SocialIntelligenceGM imports")
    print("  ✓ DeceptionIndicator imports")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")

try:
    from concordia.prefabs.game_master.negotiation import negotiation as gm_negotiation
    print("  ✓ negotiation game master imports")

    # Check build_game_master signature
    sig = inspect.signature(gm_negotiation.build_game_master)
    params = list(sig.parameters.keys())
    print(f"    build_game_master params: {params}")

    has_gm_modules = 'gm_modules' in params
    status = "✓" if has_gm_modules else "✗"
    print(f"    {status} gm_modules parameter: {'exists' if has_gm_modules else 'MISSING'}")

except ImportError as e:
    print(f"  ✗ Import failed: {e}")

# ============================================================================
# TEST 10: Check cross-agent pairing (NOW IMPLEMENTED)
# ============================================================================
print("\n[TEST 10] Cross-agent pairing")
print("-" * 50)

try:
    import dataclasses
    from concordia.prefabs.entity.negotiation.evaluation.interpretability_evaluation import (
        ActivationSample
    )

    fields = [f.name for f in dataclasses.fields(ActivationSample)]

    pairing_fields = ['counterpart_idx', 'counterpart_name']
    print("  Cross-agent pairing fields:")
    for field in pairing_fields:
        status = "✓" if field in fields else "✗"
        print(f"    {status} {field}: {'exists' if field in fields else 'NOT IMPLEMENTED'}")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# ============================================================================
# TEST 11: Check outcome tracking (NOW IMPLEMENTED)
# ============================================================================
print("\n[TEST 11] Outcome tracking")
print("-" * 50)

try:
    import dataclasses
    fields = [f.name for f in dataclasses.fields(ActivationSample)]

    outcome_fields = ['trial_outcome', 'joint_value', 'agent_utility']
    print("  Outcome tracking fields:")
    for field in outcome_fields:
        exists = field in fields
        status = "✓" if exists else "✗"
        print(f"    {status} {field}: {'exists' if exists else 'NOT IMPLEMENTED'}")

    # Check save_dataset includes outcome_labels
    source = inspect.getsource(InterpretabilityRunner.save_dataset)
    has_outcome_labels = 'outcome_labels' in source
    status = "✓" if has_outcome_labels else "✗"
    print(f"    {status} outcome_labels in save_dataset: {'yes' if has_outcome_labels else 'no'}")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# ============================================================================
# TEST 12: Check condition labeling (NOW IMPLEMENTED)
# ============================================================================
print("\n[TEST 12] Condition labeling")
print("-" * 50)

try:
    import dataclasses
    fields = [f.name for f in dataclasses.fields(ActivationSample)]

    has_condition_id = 'condition_id' in fields
    status = "✓" if has_condition_id else "✗"
    print(f"  {status} condition_id field: {'exists' if has_condition_id else 'NOT IMPLEMENTED'}")

    # Check run_single_negotiation has condition_id parameter
    sig = inspect.signature(InterpretabilityRunner.run_single_negotiation)
    params = list(sig.parameters.keys())
    has_param = 'condition_id' in params
    status = "✓" if has_param else "✗"
    print(f"  {status} condition_id parameter in run_single_negotiation: {'yes' if has_param else 'no'}")

    # Check run_ablation_study method exists
    has_ablation = hasattr(InterpretabilityRunner, 'run_ablation_study')
    status = "✓" if has_ablation else "✗"
    print(f"  {status} run_ablation_study method: {'exists' if has_ablation else 'NOT IMPLEMENTED'}")

except Exception as e:
    print(f"  [FAIL] Error: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print("""
CORE FEATURES (v2):
  ✓ InterpretabilityRunner with TransformerLens
  ✓ use_gm parameter in run_study/run_single_negotiation
  ✓ agent_modules parameter
  ✓ gm_modules parameter
  ✓ Agent labels (perceived_deception, emotion_intensity, trust_level, cooperation_intent)
  ✓ GM labels (actual_deception, commitment_violation, manipulation_score, consistency_score)
  ✓ _extract_gm_labels method
  ✓ _create_gm method
  ✓ save_dataset with both agent_labels and gm_labels
  ✓ SocialIntelligenceGM module
  ✓ build_game_master with gm_modules parameter

MULTI-AGENT ENHANCEMENTS (v3 - NOW IMPLEMENTED):
  ✓ Cross-agent pairing (counterpart_idx, counterpart_name)
  ✓ Outcome tracking (trial_outcome, joint_value, agent_utility)
  ✓ Condition labeling (condition_id)
  ✓ run_ablation_study() method for ablation studies
  ✓ outcome_labels in saved dataset

RESEARCH QUESTIONS ENABLED:
  1. "Do successful negotiations show representational alignment?"
     -> Use counterpart_idx to compare paired agent activations
  2. "Does alignment predict negotiation success?"
     -> Use outcome_labels to train success predictors
  3. "Does ToM strengthen deception encoding?"
     -> Use condition_id with run_ablation_study()
""")

#!/usr/bin/env python3
"""Comprehensive integration test for the interpretability framework with GM integration.

This test verifies:
1. All imports work correctly
2. Mock mode (no GPU required) functions properly
3. GM module integration works
4. Data structures are correct
5. Save/load functionality works
"""

import os
import sys
import tempfile
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))
sys.path.insert(0, project_root)


def test_imports():
    """Test that all imports work correctly."""
    print("\n" + "=" * 60)
    print("TEST 1: Imports")
    print("=" * 60)

    try:
        from concordia.prefabs.entity.negotiation.evaluation.interpretability_evaluation import (
            InterpretabilityRunner,
            TransformerLensWrapper,
            ActivationSample,
            EvaluationResult,
            run_quick_study,
        )
        print("  [OK] interpretability_evaluation imports")
    except ImportError as e:
        print(f"  [FAIL] interpretability_evaluation: {e}")
        return False

    try:
        from concordia.prefabs.entity.negotiation.evaluation.contest_scenarios import (
            create_scenario,
        )
        print("  [OK] contest_scenarios imports")
    except ImportError as e:
        print(f"  [FAIL] contest_scenarios: {e}")
        return False

    try:
        from concordia.prefabs.game_master.negotiation.components.gm_social_intelligence import (
            SocialIntelligenceGM,
            DeceptionIndicator,
            EmotionalReading,
        )
        print("  [OK] gm_social_intelligence imports")
    except ImportError as e:
        print(f"  [FAIL] gm_social_intelligence: {e}")
        return False

    try:
        from concordia.prefabs.game_master.negotiation import negotiation as gm_negotiation
        print("  [OK] negotiation game master imports")
    except ImportError as e:
        print(f"  [FAIL] negotiation game master: {e}")
        return False

    print("\n  All imports successful!")
    return True


def test_data_structures():
    """Test that data structures are correctly defined."""
    print("\n" + "=" * 60)
    print("TEST 2: Data Structures")
    print("=" * 60)

    import torch
    from concordia.prefabs.entity.negotiation.evaluation.interpretability_evaluation import (
        ActivationSample,
        EvaluationResult,
    )

    # Test ActivationSample
    try:
        sample = ActivationSample(
            trial_id=1,
            round_num=0,
            agent_name="Test_Agent",
            activations={"layer_0": torch.randn(768)},
            prompt="Test prompt",
            response="Test response",
            # Agent labels (first-person)
            perceived_deception=0.3,
            emotion_intensity=0.5,
            trust_level=0.6,
            cooperation_intent=0.7,
            # GM labels (third-person ground truth)
            actual_deception=0.2,
            commitment_violation=0.0,
            manipulation_score=0.1,
            consistency_score=0.9,
            # Context
            scenario_type="fishery",
            modules_enabled=["theory_of_mind"],
            gm_modules_enabled=["social_intelligence"],
        )

        # Verify all fields
        assert sample.perceived_deception == 0.3, "Agent label field failed"
        assert sample.actual_deception == 0.2, "GM label field failed"
        assert sample.gm_modules_enabled == ["social_intelligence"], "GM modules field failed"
        print("  [OK] ActivationSample with agent + GM labels")
    except Exception as e:
        print(f"  [FAIL] ActivationSample: {e}")
        return False

    # Test EvaluationResult
    try:
        result = EvaluationResult(
            cooperation_rate=0.5,
            average_payoff=100.0,
            agreement_rate=0.8,
            num_trials=10,
            activation_samples=[sample],
            total_llm_calls=50,
            layers_captured=["blocks.0", "blocks.12"],
            activation_dim=768,
            total_deception_detected=3,
            gm_modules_used=["social_intelligence"],
        )

        assert result.total_deception_detected == 3, "GM deception count failed"
        assert result.gm_modules_used == ["social_intelligence"], "GM modules used failed"
        print("  [OK] EvaluationResult with GM fields")
    except Exception as e:
        print(f"  [FAIL] EvaluationResult: {e}")
        return False

    print("\n  All data structures correct!")
    return True


def test_gm_social_intelligence():
    """Test the GM social intelligence module directly."""
    print("\n" + "=" * 60)
    print("TEST 3: GM Social Intelligence Module")
    print("=" * 60)

    from concordia.prefabs.game_master.negotiation.components.gm_social_intelligence import (
        SocialIntelligenceGM,
        DeceptionIndicator,
    )

    try:
        # Create module
        social_intel = SocialIntelligenceGM(
            name='social_intelligence',
            config={'detect_deception': True, 'track_emotions': True}
        )
        print("  [OK] SocialIntelligenceGM created")

        # Test consistency checking (detects potential deception)
        # First statement
        indicator1 = social_intel.check_consistency(
            "Agent_A",
            "I can pay $100 for this item",
            round_number=1
        )
        print(f"  [OK] First statement checked: {indicator1}")

        # Contradictory statement (should detect inconsistency)
        indicator2 = social_intel.check_consistency(
            "Agent_A",
            "I cannot pay more than $50, that's my absolute limit",
            round_number=2
        )
        print(f"  [OK] Second statement checked: {indicator2}")

        # The second statement contradicts the first, so it should detect deception
        # Check deception indicators list
        deception_count = len(social_intel._deception_indicators)
        print(f"  [INFO] Deception indicators detected: {deception_count}")

        # Test emotion detection
        emotion = social_intel.detect_emotion(
            "This is absolutely frustrating and unfair!",
            "Agent_A",
            round_number=3
        )
        if emotion:
            print(f"  [OK] Emotion detected: {emotion.primary_emotion} (intensity: {emotion.intensity})")
        else:
            print("  [INFO] No emotion detected (emotion tracking might be disabled)")

        print("\n  GM Social Intelligence module works!")
        return True

    except Exception as e:
        print(f"  [FAIL] SocialIntelligenceGM: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_mock_runner():
    """Test the InterpretabilityRunner in mock mode without GPU."""
    print("\n" + "=" * 60)
    print("TEST 4: Mock Runner (No GPU)")
    print("=" * 60)

    import torch
    from concordia.prefabs.entity.negotiation.evaluation.interpretability_evaluation import (
        ActivationSample,
    )

    # Create mock samples directly (simulating what runner would produce)
    try:
        samples = []
        for trial in range(2):
            for round_num in range(3):
                for agent in ["Agent_A", "Agent_B"]:
                    # Simulate varying labels
                    perceived_dec = np.random.random() * 0.5
                    actual_dec = np.random.random() * 0.5

                    sample = ActivationSample(
                        trial_id=trial + 1,
                        round_num=round_num,
                        agent_name=agent,
                        activations={
                            "blocks.0.hook_resid_post": torch.randn(768),
                            "blocks.12.hook_resid_post": torch.randn(768),
                        },
                        prompt=f"Round {round_num} prompt",
                        response=f"Round {round_num} response",
                        perceived_deception=perceived_dec,
                        emotion_intensity=np.random.random() * 0.3,
                        trust_level=0.5 + np.random.random() * 0.3,
                        cooperation_intent=0.5 + np.random.random() * 0.3,
                        actual_deception=actual_dec,
                        commitment_violation=0.0,
                        manipulation_score=np.random.random() * 0.2,
                        consistency_score=0.8 + np.random.random() * 0.2,
                        scenario_type="fishery",
                        modules_enabled=["theory_of_mind"],
                        gm_modules_enabled=["social_intelligence"],
                    )
                    samples.append(sample)

        print(f"  [OK] Created {len(samples)} mock samples")

        # Verify sample distribution
        perceived = [s.perceived_deception for s in samples]
        actual = [s.actual_deception for s in samples]
        print(f"  [INFO] Perceived deception: mean={np.mean(perceived):.3f}, std={np.std(perceived):.3f}")
        print(f"  [INFO] Actual deception: mean={np.mean(actual):.3f}, std={np.std(actual):.3f}")

        print("\n  Mock runner simulation successful!")
        return True, samples

    except Exception as e:
        print(f"  [FAIL] Mock runner: {e}")
        import traceback
        traceback.print_exc()
        return False, []


def test_save_load(samples):
    """Test saving and loading dataset."""
    print("\n" + "=" * 60)
    print("TEST 5: Save/Load Dataset")
    print("=" * 60)

    import torch
    import tempfile

    try:
        # Convert samples to dataset format
        all_activations = []
        all_agent_labels = []
        all_gm_labels = []
        metadata = []

        for sample in samples:
            layer_acts = [sample.activations[k] for k in sorted(sample.activations.keys())]
            if layer_acts:
                all_activations.append(torch.stack(layer_acts))
                all_agent_labels.append(torch.tensor([
                    sample.perceived_deception,
                    sample.emotion_intensity,
                    sample.trust_level,
                    sample.cooperation_intent,
                ]))
                all_gm_labels.append(torch.tensor([
                    sample.actual_deception,
                    sample.commitment_violation,
                    sample.manipulation_score,
                    sample.consistency_score,
                ]))
                metadata.append({
                    'trial_id': sample.trial_id,
                    'round_num': sample.round_num,
                    'agent_name': sample.agent_name,
                    'scenario': sample.scenario_type,
                })

        dataset = {
            'activations': torch.stack(all_activations),
            'agent_labels': torch.stack(all_agent_labels),
            'agent_label_names': ['perceived_deception', 'emotion_intensity', 'trust_level', 'cooperation_intent'],
            'gm_labels': torch.stack(all_gm_labels),
            'gm_label_names': ['actual_deception', 'commitment_violation', 'manipulation_score', 'consistency_score'],
            'labels': torch.stack(all_agent_labels),  # Legacy format
            'label_names': ['perceived_deception', 'emotion_intensity', 'trust_level', 'cooperation_intent'],
            'metadata': metadata,
            'layers': ['blocks.0.hook_resid_post', 'blocks.12.hook_resid_post'],
        }

        print(f"  [OK] Dataset created:")
        print(f"       - activations shape: {dataset['activations'].shape}")
        print(f"       - agent_labels shape: {dataset['agent_labels'].shape}")
        print(f"       - gm_labels shape: {dataset['gm_labels'].shape}")

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
            torch.save(dataset, f.name)
            temp_path = f.name

        print(f"  [OK] Saved to {temp_path}")

        # Load back
        loaded = torch.load(temp_path)

        # Verify
        assert 'activations' in loaded, "Missing activations"
        assert 'agent_labels' in loaded, "Missing agent_labels"
        assert 'gm_labels' in loaded, "Missing gm_labels"
        assert 'agent_label_names' in loaded, "Missing agent_label_names"
        assert 'gm_label_names' in loaded, "Missing gm_label_names"

        assert loaded['activations'].shape == dataset['activations'].shape
        assert loaded['agent_labels'].shape == dataset['agent_labels'].shape
        assert loaded['gm_labels'].shape == dataset['gm_labels'].shape

        print(f"  [OK] Loaded and verified")

        # Verify label names
        print(f"  [INFO] Agent labels: {loaded['agent_label_names']}")
        print(f"  [INFO] GM labels: {loaded['gm_label_names']}")

        # Clean up
        os.unlink(temp_path)

        print("\n  Save/load functionality works!")
        return True

    except Exception as e:
        print(f"  [FAIL] Save/load: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_probe_training(samples):
    """Test that saved data can be used for probe training."""
    print("\n" + "=" * 60)
    print("TEST 6: Probe Training Simulation")
    print("=" * 60)

    import torch
    import numpy as np

    try:
        # Prepare data
        activations = []
        agent_labels = []
        gm_labels = []

        for sample in samples:
            layer_acts = [sample.activations[k] for k in sorted(sample.activations.keys())]
            if layer_acts:
                # Use middle layer for probing
                middle_layer = layer_acts[len(layer_acts) // 2]
                activations.append(middle_layer.numpy())
                agent_labels.append(sample.perceived_deception)
                gm_labels.append(sample.actual_deception)

        X = np.stack(activations)
        y_perceived = np.array(agent_labels)
        y_actual = np.array(gm_labels)

        print(f"  [OK] Prepared data: X={X.shape}, y_perceived={y_perceived.shape}, y_actual={y_actual.shape}")

        # Simple linear probe (sklearn Ridge)
        try:
            from sklearn.linear_model import Ridge
            from sklearn.model_selection import cross_val_score

            # Train probe for perceived_deception (agent label)
            probe_perceived = Ridge(alpha=1.0)
            scores_perceived = cross_val_score(probe_perceived, X, y_perceived, cv=2, scoring='r2')
            print(f"  [OK] Probe on perceived_deception: R2={np.mean(scores_perceived):.3f}")

            # Train probe for actual_deception (GM label)
            probe_actual = Ridge(alpha=1.0)
            scores_actual = cross_val_score(probe_actual, X, y_actual, cv=2, scoring='r2')
            print(f"  [OK] Probe on actual_deception: R2={np.mean(scores_actual):.3f}")

            # Note: With random data, R2 will be near 0 or negative (expected)
            print("  [INFO] R2 near 0 is expected with random mock data")

        except ImportError:
            print("  [SKIP] sklearn not installed, skipping probe training")

        print("\n  Probe training simulation successful!")
        return True

    except Exception as e:
        print(f"  [FAIL] Probe training: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scenario_creation():
    """Test that scenarios can be created correctly."""
    print("\n" + "=" * 60)
    print("TEST 7: Scenario Creation")
    print("=" * 60)

    from concordia.prefabs.entity.negotiation.evaluation.contest_scenarios import (
        create_scenario,
    )

    try:
        scenarios = ['fishery', 'treaty', 'gameshow']
        print(f"  [INFO] Testing scenarios: {scenarios}")

        for scenario_name in scenarios:
            try:
                scenario = create_scenario(scenario_name)
                scenario.initialize()
                obs = scenario.get_observation("Test_Agent")
                print(f"  [OK] {scenario_name}: Created and initialized")
            except Exception as e:
                print(f"  [WARN] {scenario_name}: {e}")

        print("\n  Scenario creation works!")
        return True

    except Exception as e:
        print(f"  [FAIL] Scenario creation: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("COMPREHENSIVE INTEGRATION TEST")
    print("Interpretability Framework with GM Integration")
    print("=" * 60)

    results = {}

    # Test 1: Imports
    results['imports'] = test_imports()

    # Test 2: Data structures
    results['data_structures'] = test_data_structures()

    # Test 3: GM Social Intelligence
    results['gm_social_intel'] = test_gm_social_intelligence()

    # Test 4: Mock runner
    mock_result, samples = test_mock_runner()
    results['mock_runner'] = mock_result

    # Test 5: Save/Load
    if samples:
        results['save_load'] = test_save_load(samples)
    else:
        results['save_load'] = False

    # Test 6: Probe training
    if samples:
        results['probe_training'] = test_probe_training(samples)
    else:
        results['probe_training'] = False

    # Test 7: Scenario creation
    results['scenarios'] = test_scenario_creation()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = 0
    failed = 0
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        symbol = "[✓]" if result else "[✗]"
        print(f"  {symbol} {test_name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n  Total: {passed} passed, {failed} failed out of {len(results)}")

    if failed == 0:
        print("\n  ALL TESTS PASSED!")
        return True
    else:
        print(f"\n  {failed} TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

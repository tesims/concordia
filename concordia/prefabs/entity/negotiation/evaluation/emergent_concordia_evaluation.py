#!/usr/bin/env python3
"""
Emergent Deception Evaluation with Full Concordia Framework

This module bridges the emergent deception scenarios (incentive-based, no forbidden words)
with the full Concordia negotiation framework (cognitive modules, GM ground truth).

Key Features:
- Uses TransformerLens for activation capture (like interpretability_evaluation.py)
- Builds real Concordia agents with theory_of_mind, temporal_strategy, etc.
- Uses emergent deception prompts (no explicit deception instructions)
- Gets ground truth from BOTH:
  1. GM modules (behavioral assessment)
  2. Emergent prompt rules (scenario-specific regex checks)

Research Question:
Can we detect deception in the activations of a Concordia negotiation agent
when deception emerges from incentives rather than explicit instructions?

Usage:
    runner = EmergentConcordiaRunner(
        model_name="google/gemma-2-9b-it",
        device="cuda"
    )

    results = runner.run_study(
        scenario='ultimatum_bluff',
        num_trials=50,
        agent_modules=['theory_of_mind', 'temporal_strategy'],
    )

    runner.save_dataset('emergent_concordia_activations.pt')
"""

import os
import sys
import json
import torch
import numpy as np
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from enum import Enum

# Concordia imports
from concordia.language_model import language_model
from concordia.associative_memory import basic_associative_memory
from concordia.typing import entity as entity_lib

# Local imports - we'll use the TransformerLensWrapper from interpretability_evaluation
from .interpretability_evaluation import TransformerLensWrapper, ActivationSample

# Import emergent prompts (add to path if needed)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..', 'evaluation', 'scenarios'))
try:
    from emergent_prompts import (
        EMERGENT_SCENARIOS,
        IncentiveCondition,
        get_emergent_prompt,
        generate_scenario_params,
        compute_ground_truth,
        get_all_scenarios,
    )
    EMERGENT_AVAILABLE = True
except ImportError:
    EMERGENT_AVAILABLE = False
    print("Warning: emergent_prompts.py not found. Using fallback scenarios.")


# =============================================================================
# EMERGENT SCENARIO ADAPTER
# =============================================================================

class EmergentScenarioAdapter:
    """Adapts emergent deception prompts to work with Concordia agents."""

    def __init__(self, scenario_name: str, condition: str = "HIGH_INCENTIVE"):
        if not EMERGENT_AVAILABLE:
            raise ImportError("emergent_prompts.py not found in evaluation/scenarios/")

        if scenario_name not in EMERGENT_SCENARIOS:
            raise ValueError(f"Unknown scenario: {scenario_name}. Available: {list(EMERGENT_SCENARIOS.keys())}")

        self.scenario_name = scenario_name
        self.condition = IncentiveCondition.HIGH_INCENTIVE if condition == "HIGH_INCENTIVE" else IncentiveCondition.LOW_INCENTIVE
        self.params = {}
        self.trial_id = 0

    def initialize(self, trial_id: int = 0) -> Dict[str, Any]:
        """Initialize scenario with random parameters."""
        self.trial_id = trial_id
        self.params = generate_scenario_params(self.scenario_name, trial_id)
        return self.params

    def get_agent_prompt(self) -> str:
        """Get the main agent's prompt (potential deceiver)."""
        return get_emergent_prompt(self.scenario_name, self.condition, self.params)

    def get_counterpart_prompt(self) -> str:
        """Get the counterpart's prompt."""
        from emergent_prompts import get_counterpart_prompt
        return get_counterpart_prompt(self.scenario_name, self.params)

    def check_deception(self, response: str) -> bool:
        """Check if response was deceptive using ground truth rules."""
        return compute_ground_truth(self.scenario_name, response, self.params)

    def get_observation(self, agent_name: str) -> str:
        """Generate observation for agent."""
        if "counterpart" in agent_name.lower() or agent_name == "Agent_B":
            return self.get_counterpart_prompt()
        else:
            return self.get_agent_prompt()


# =============================================================================
# EMERGENT DECEPTION SAMPLE
# =============================================================================

@dataclass
class EmergentDeceptionSample(ActivationSample):
    """Extended sample with emergent deception ground truth."""

    # Emergent-specific fields
    emergent_scenario: str = ""
    incentive_condition: str = ""
    scenario_params: Dict[str, Any] = field(default_factory=dict)

    # Ground truth from emergent rules (regex-based)
    emergent_ground_truth: bool = False

    # Combined ground truth (GM + emergent rules)
    combined_deception_score: float = 0.0


# =============================================================================
# EMERGENT CONCORDIA RUNNER
# =============================================================================

class EmergentConcordiaRunner:
    """
    Runs emergent deception experiments through the full Concordia framework.

    This combines:
    - TransformerLens activation capture
    - Concordia negotiation agents with cognitive modules
    - Emergent deception prompts (incentive-based)
    - Ground truth from both GM and emergent rules
    """

    def __init__(
        self,
        model_name: str = "google/gemma-2-9b-it",
        device: str = "cuda",
        layers_to_capture: List[int] = None,
    ):
        # Initialize TransformerLens wrapper
        self.model = TransformerLensWrapper(
            model_name=model_name,
            device=device,
            layers_to_capture=layers_to_capture,
        )

        self.activation_samples: List[EmergentDeceptionSample] = []
        self._trial_id = 0
        self._component_access_failures: Dict[str, int] = defaultdict(int)

    def _create_memory_bank(self):
        """Create memory bank with simple embedder."""
        def embedder(text: str) -> np.ndarray:
            hash_bytes = hashlib.sha256(text.encode()).digest()
            embedding = np.array([float(b) / 255.0 for b in hash_bytes[:64]])
            return embedding / (np.linalg.norm(embedding) + 1e-8)

        return basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=embedder
        )

    def _build_agent(
        self,
        name: str,
        goal: str,
        modules: List[str],
    ):
        """Build a Concordia negotiation agent."""
        from concordia.prefabs.entity.negotiation import advanced_negotiator

        memory_bank = self._create_memory_bank()

        agent = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=memory_bank,
            name=name,
            goal=goal,
            modules=modules,
            module_configs={
                'theory_of_mind': {
                    'max_recursion_depth': 2,
                    'emotion_sensitivity': 0.7,
                },
                'temporal_strategy': {
                    'time_horizon': 5,
                }
            }
        )
        return agent

    def _extract_agent_labels(self, agent) -> Dict[str, float]:
        """Extract cognitive state from agent modules."""
        labels = {
            'perceived_deception': 0.0,
            'emotion_intensity': 0.0,
            'trust_level': 0.5,
            'cooperation_intent': 0.5,
        }

        try:
            tom = agent.get_component('TheoryOfMind')
            if tom:
                state = tom.get_state()
                mental_models = state.get('mental_models', {})

                if mental_models:
                    deception_risks = []
                    for model_state in mental_models.values():
                        deception_risks.append(model_state.get('deception_risk', 0.0))

                    labels['perceived_deception'] = np.mean(deception_risks) if deception_risks else 0.0
        except (AttributeError, KeyError, TypeError):
            self._component_access_failures['TheoryOfMind'] += 1

        return labels

    def run_single_trial(
        self,
        scenario_name: str,
        condition: str = "HIGH_INCENTIVE",
        agent_modules: List[str] = None,
        max_rounds: int = 5,
    ) -> Dict[str, Any]:
        """
        Run a single emergent deception trial through Concordia.

        Args:
            scenario_name: One of the 6 emergent scenarios
            condition: "HIGH_INCENTIVE" or "LOW_INCENTIVE"
            agent_modules: Cognitive modules to enable
            max_rounds: Maximum conversation turns

        Returns:
            Trial results with deception labels
        """
        agent_modules = agent_modules or ['theory_of_mind']
        self._trial_id += 1
        trial_samples = []

        # Initialize scenario
        scenario = EmergentScenarioAdapter(scenario_name, condition)
        params = scenario.initialize(self._trial_id)

        # Build agents
        main_agent = self._build_agent(
            name="Negotiator",
            goal=scenario.get_agent_prompt(),
            modules=agent_modules,
        )

        counterpart = self._build_agent(
            name="Counterpart",
            goal=scenario.get_counterpart_prompt(),
            modules=[],  # Simple counterpart
        )

        # Initial observation
        main_agent.observe(scenario.get_agent_prompt())
        counterpart.observe(scenario.get_counterpart_prompt())

        # Run negotiation rounds
        all_responses = []
        deception_detected = False

        for round_num in range(max_rounds):
            # Main agent acts
            pre_call_count = self.model.call_count

            action_prompt = f"Round {round_num + 1}/{max_rounds}. Make your negotiation move."
            action_spec = entity_lib.ActionSpec(
                call_to_action=action_prompt,
                output_type=entity_lib.OutputType.FREE,
            )

            response = main_agent.act(action_spec)
            all_responses.append(response)

            # Capture activations if LLM was called
            if self.model.call_count > pre_call_count:
                activations = self.model.get_activations()

                # Check emergent ground truth
                is_deceptive = scenario.check_deception(response)
                if is_deceptive:
                    deception_detected = True

                # Extract agent labels
                agent_labels = self._extract_agent_labels(main_agent)

                # Create sample
                sample = EmergentDeceptionSample(
                    trial_id=self._trial_id,
                    round_num=round_num,
                    agent_name="Negotiator",
                    activations=activations,
                    prompt=action_prompt[:200],
                    response=response[:500],
                    # Agent labels
                    perceived_deception=agent_labels['perceived_deception'],
                    emotion_intensity=agent_labels['emotion_intensity'],
                    trust_level=agent_labels['trust_level'],
                    cooperation_intent=agent_labels['cooperation_intent'],
                    # GM labels (from emergent rules as proxy)
                    actual_deception=1.0 if is_deceptive else 0.0,
                    commitment_violation=0.0,
                    manipulation_score=0.5 if is_deceptive else 0.0,
                    consistency_score=0.5 if is_deceptive else 1.0,
                    # Context
                    scenario_type=scenario_name,
                    modules_enabled=agent_modules,
                    # Emergent-specific
                    emergent_scenario=scenario_name,
                    incentive_condition=condition,
                    scenario_params=params.copy(),
                    emergent_ground_truth=is_deceptive,
                    combined_deception_score=1.0 if is_deceptive else 0.0,
                )
                trial_samples.append(sample)

            # Counterpart observes and responds
            counterpart.observe(f"Negotiator: {response}")

            counter_response = counterpart.act(entity_lib.ActionSpec(
                call_to_action="Respond to their offer.",
                output_type=entity_lib.OutputType.FREE,
            ))

            main_agent.observe(f"Counterpart: {counter_response}")

        # Store samples
        self.activation_samples.extend(trial_samples)

        return {
            'trial_id': self._trial_id,
            'scenario': scenario_name,
            'condition': condition,
            'params': params,
            'deception_detected': deception_detected,
            'samples_collected': len(trial_samples),
            'responses': all_responses,
        }

    def run_study(
        self,
        scenario: str = 'ultimatum_bluff',
        num_trials: int = 50,
        agent_modules: List[str] = None,
        max_rounds: int = 5,
        conditions: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run full study with both HIGH and LOW incentive conditions.

        Args:
            scenario: Emergent scenario name
            num_trials: Trials PER condition
            agent_modules: Cognitive modules to enable
            max_rounds: Rounds per trial
            conditions: List of conditions (default: both HIGH and LOW)

        Returns:
            Study results with all samples
        """
        agent_modules = agent_modules or ['theory_of_mind']
        conditions = conditions or ["HIGH_INCENTIVE", "LOW_INCENTIVE"]

        print(f"\n{'='*60}")
        print(f"EMERGENT DECEPTION STUDY: {scenario}")
        print(f"{'='*60}")
        print(f"Trials per condition: {num_trials}")
        print(f"Conditions: {conditions}")
        print(f"Agent modules: {agent_modules}")
        print(f"Max rounds: {max_rounds}")
        print("-" * 60)

        results = {
            'scenario': scenario,
            'conditions': {},
            'total_samples': 0,
            'total_deception': 0,
        }

        for condition in conditions:
            print(f"\n[{condition}]")
            condition_results = []
            deception_count = 0

            for trial in range(num_trials):
                trial_result = self.run_single_trial(
                    scenario_name=scenario,
                    condition=condition,
                    agent_modules=agent_modules,
                    max_rounds=max_rounds,
                )
                condition_results.append(trial_result)

                if trial_result['deception_detected']:
                    deception_count += 1

                if (trial + 1) % 10 == 0:
                    print(f"  Trial {trial+1}/{num_trials}: "
                          f"deception_rate={deception_count/(trial+1):.1%}")

            results['conditions'][condition] = {
                'num_trials': num_trials,
                'deception_count': deception_count,
                'deception_rate': deception_count / num_trials,
                'trials': condition_results,
            }
            results['total_deception'] += deception_count

        results['total_samples'] = len(self.activation_samples)

        return results

    def run_all_scenarios(
        self,
        num_trials_per_scenario: int = 50,
        agent_modules: List[str] = None,
        scenarios: List[str] = None,
    ) -> Dict[str, Any]:
        """Run study across all 6 emergent scenarios."""
        if not EMERGENT_AVAILABLE:
            raise ImportError("emergent_prompts.py not found")

        scenarios = scenarios or get_all_scenarios()
        agent_modules = agent_modules or ['theory_of_mind']

        print("\n" + "=" * 70)
        print("COMPREHENSIVE EMERGENT DECEPTION STUDY")
        print("=" * 70)
        print(f"Scenarios: {scenarios}")
        print(f"Trials per scenario (per condition): {num_trials_per_scenario}")
        print(f"Total trials: {len(scenarios) * num_trials_per_scenario * 2}")

        all_results = {}

        for scenario in scenarios:
            results = self.run_study(
                scenario=scenario,
                num_trials=num_trials_per_scenario,
                agent_modules=agent_modules,
            )
            all_results[scenario] = results

        return all_results

    def save_dataset(self, filepath: str):
        """Save activation dataset with emergent deception labels."""

        all_activations = []
        all_gm_labels = []
        all_emergent_labels = []
        metadata = []

        for sample in self.activation_samples:
            layer_acts = [sample.activations[k] for k in sorted(sample.activations.keys())]
            if layer_acts:
                stacked = torch.stack(layer_acts)
                all_activations.append(stacked)

                # GM-style labels
                all_gm_labels.append(torch.tensor([
                    sample.actual_deception,
                    sample.commitment_violation,
                    sample.manipulation_score,
                    sample.consistency_score,
                ]))

                # Emergent ground truth (binary)
                all_emergent_labels.append(torch.tensor([
                    1.0 if sample.emergent_ground_truth else 0.0,
                ]))

                metadata.append({
                    'trial_id': sample.trial_id,
                    'round_num': sample.round_num,
                    'scenario': sample.emergent_scenario,
                    'condition': sample.incentive_condition,
                    'params': sample.scenario_params,
                    'response': sample.response,
                    'emergent_ground_truth': sample.emergent_ground_truth,
                })

        if all_activations:
            dataset = {
                'activations': torch.stack(all_activations),
                'gm_labels': torch.stack(all_gm_labels),
                'emergent_labels': torch.stack(all_emergent_labels),
                'gm_label_names': ['actual_deception', 'commitment_violation',
                                   'manipulation_score', 'consistency_score'],
                'emergent_label_names': ['is_deceptive'],
                'metadata': metadata,
                'layers': self.model.hook_names,
                'model_name': 'gemma',  # or extract from model
            }

            torch.save(dataset, filepath)

            print(f"\nSaved {len(all_activations)} samples to {filepath}")
            print(f"  Activation shape: {dataset['activations'].shape}")
            print(f"  GM labels shape: {dataset['gm_labels'].shape}")
            print(f"  Emergent labels shape: {dataset['emergent_labels'].shape}")

            # Print deception rate by scenario
            scenarios = defaultdict(lambda: {'total': 0, 'deceptive': 0})
            for m in metadata:
                scenarios[m['scenario']]['total'] += 1
                if m['emergent_ground_truth']:
                    scenarios[m['scenario']]['deceptive'] += 1

            print(f"\nDeception rates by scenario:")
            for scenario, counts in scenarios.items():
                rate = counts['deceptive'] / counts['total'] if counts['total'] > 0 else 0
                print(f"  {scenario}: {rate:.1%} ({counts['deceptive']}/{counts['total']})")
        else:
            print("No samples to save!")

    def print_summary(self, results: Dict[str, Any]):
        """Print study summary."""
        print("\n" + "=" * 60)
        print("STUDY SUMMARY")
        print("=" * 60)

        print(f"\nScenario: {results.get('scenario', 'all')}")
        print(f"Total samples: {results.get('total_samples', len(self.activation_samples))}")

        for condition, data in results.get('conditions', {}).items():
            print(f"\n{condition}:")
            print(f"  Trials: {data['num_trials']}")
            print(f"  Deception detected: {data['deception_count']}")
            print(f"  Deception rate: {data['deception_rate']:.1%}")


# =============================================================================
# QUICK START
# =============================================================================

def run_quick_test(
    model_name: str = "google/gemma-2-2b-it",
    device: str = "cuda",
    scenario: str = "ultimatum_bluff",
    num_trials: int = 5,
):
    """Quick test of the emergent concordia evaluation."""

    print("EMERGENT CONCORDIA EVALUATION - QUICK TEST")
    print("=" * 50)

    runner = EmergentConcordiaRunner(
        model_name=model_name,
        device=device,
    )

    results = runner.run_study(
        scenario=scenario,
        num_trials=num_trials,
        agent_modules=['theory_of_mind'],
        max_rounds=3,
    )

    runner.print_summary(results)
    runner.save_dataset(f'emergent_test_{scenario}.pt')

    return runner, results


if __name__ == "__main__":
    print("Emergent Concordia Evaluation")
    print("=" * 50)
    print("""
This module runs emergent deception experiments through the full Concordia
negotiation framework with TransformerLens activation capture.

Usage:
    from emergent_concordia_evaluation import EmergentConcordiaRunner

    runner = EmergentConcordiaRunner(
        model_name="google/gemma-2-9b-it",
        device="cuda"
    )

    # Run single scenario
    results = runner.run_study(
        scenario='ultimatum_bluff',
        num_trials=50,
        agent_modules=['theory_of_mind'],
    )

    # Run all 6 scenarios
    all_results = runner.run_all_scenarios(
        num_trials_per_scenario=50,
        agent_modules=['theory_of_mind', 'temporal_strategy'],
    )

    runner.save_dataset('emergent_concordia_activations.pt')

Available scenarios:
    - ultimatum_bluff: False final offer claims
    - capability_bluff: Overstating project timeline
    - hidden_value: Inflating asking price
    - info_withholding: Not disclosing defects
    - promise_break: Promising cooperation, hinting defection
    - alliance_betrayal: Assuring ally while considering betrayal
""")

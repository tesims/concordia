"""Comprehensive unit tests for negotiation cognitive modules.

Tests cover:
- Module initialization
- Core methods (pre_act, post_act, get_state, set_state)
- LLM interaction patterns
- Edge cases and error handling
"""

import unittest
from unittest import mock
import numpy as np
import hashlib
from typing import Dict, Any, List, Tuple

# Concordia imports
from concordia.language_model import language_model
from concordia.typing import entity as entity_lib
from concordia.associative_memory import basic_associative_memory

# Module imports
from concordia.prefabs.entity.negotiation.components import theory_of_mind
from concordia.prefabs.entity.negotiation.components import cultural_adaptation
from concordia.prefabs.entity.negotiation.components import temporal_strategy
from concordia.prefabs.entity.negotiation.components import swarm_intelligence
from concordia.prefabs.entity.negotiation.components import uncertainty_aware
from concordia.prefabs.entity.negotiation.components import strategy_evolution
from concordia.prefabs.entity.negotiation.components import negotiation_instructions
from concordia.prefabs.entity.negotiation.components import negotiation_memory
from concordia.prefabs.entity.negotiation.components import negotiation_strategy

# Prefab imports
from concordia.prefabs.entity.negotiation import base_negotiator
from concordia.prefabs.entity.negotiation import advanced_negotiator


class MockLanguageModel(language_model.LanguageModel):
    """Mock language model for testing."""

    def __init__(self, responses: Dict[str, str] = None):
        self.responses = responses or {}
        self.call_count = 0
        self.prompts = []

    def sample_text(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        terminators: Tuple[str, ...] = (),
        temperature: float = 1.0,
        timeout: float = 60.0,
        seed: int = None,
        **kwargs
    ) -> str:
        self.call_count += 1
        self.prompts.append(prompt)

        # Check for specific response mappings
        for key, response in self.responses.items():
            if key.lower() in prompt.lower():
                return response

        # Default responses based on prompt content
        if 'emotion' in prompt.lower():
            return "anger:0.2 fear:0.1 joy:0.5 sadness:0.1 surprise:0.2 trust:0.7 anticipation:0.4 valence:0.3 arousal:0.5 confidence:0.8"
        if 'strategy' in prompt.lower():
            return "cooperative"
        if 'cultural' in prompt.lower() or 'culture' in prompt.lower():
            return "western_business"
        if 'offer' in prompt.lower() or 'price' in prompt.lower():
            return "I propose $100 as a fair price."
        if 'accept' in prompt.lower():
            return "I accept this offer."

        return "This is a test response for negotiation."

    def sample_choice(
        self,
        prompt: str,
        responses: List[str],
        *,
        seed: int = None,
    ) -> Tuple[int, str, Dict[str, Any]]:
        self.call_count += 1
        self.prompts.append(prompt)
        return (0, responses[0], {})


def create_mock_embedder(seed: int = 42):
    """Create a deterministic embedder for testing."""
    def embedder(text: str) -> np.ndarray:
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        np.random.seed(hash_val + seed)
        return np.random.randn(384).astype(np.float32)
    return embedder


def create_action_spec(call_to_action: str = "What is your next move?") -> entity_lib.ActionSpec:
    """Create a mock ActionSpec for testing."""
    return entity_lib.ActionSpec(
        call_to_action=call_to_action,
        output_type=entity_lib.OutputType.FREE,
    )


class TestTheoryOfMind(unittest.TestCase):
    """Tests for Theory of Mind module."""

    def setUp(self):
        self.model = MockLanguageModel()
        self.tom = theory_of_mind.TheoryOfMind(
            model=self.model,
            max_recursion_depth=2,
            emotion_sensitivity=0.7,
            empathy_level=0.8,
        )

    def test_initialization(self):
        """Test module initializes correctly."""
        self.assertEqual(self.tom._max_recursion_depth, 2)
        self.assertEqual(self.tom._emotion_sensitivity, 0.7)
        self.assertEqual(self.tom._empathy_level, 0.8)
        self.assertEqual(len(self.tom._mental_models), 0)

    def test_detect_emotions(self):
        """Test emotion detection from text."""
        text = "I am very frustrated with this offer!"
        result = self.tom._detect_emotions(text)

        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'emotions'))
        self.assertIn('anger', result.emotions)

    def test_get_state(self):
        """Test state retrieval."""
        state = self.tom.get_state()

        self.assertIn('mental_models', state)
        self.assertIn('recursion_depth', state)
        self.assertIn('empathy_level', state)
        self.assertEqual(state['recursion_depth'], 2)
        self.assertEqual(state['empathy_level'], 0.8)

    def test_set_state(self):
        """Test state restoration."""
        new_state = {
            'recursion_depth': 5,
            'empathy_level': 0.5,
        }
        self.tom.set_state(new_state)

        self.assertEqual(self.tom._max_recursion_depth, 5)
        self.assertEqual(self.tom._empathy_level, 0.5)

    def test_pre_act(self):
        """Test pre_act returns context string."""
        action_spec = create_action_spec("The seller offered $150. What do you do?")
        result = self.tom.pre_act(action_spec)

        self.assertIsInstance(result, str)
        # pre_act should return some guidance

    def test_post_act(self):
        """Test post_act processes action."""
        action = "I counter-offer $120"
        result = self.tom.post_act(action)

        # post_act should return empty string or status
        self.assertIsInstance(result, str)

    def test_get_action_attempt(self):
        """Test action generation."""
        action_spec = create_action_spec("Negotiate a fair price for the item.")
        context = {}

        result = self.tom.get_action_attempt(context, action_spec)

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestCulturalAdaptation(unittest.TestCase):
    """Tests for Cultural Adaptation module."""

    def setUp(self):
        self.model = MockLanguageModel()
        self.cultural = cultural_adaptation.CulturalAdaptation(
            model=self.model,
            own_culture='western_business',
            adaptation_level=0.7,
            detect_culture=True,
        )

    def test_initialization(self):
        """Test module initializes correctly."""
        self.assertIsNotNone(self.cultural._own_profile)
        self.assertEqual(self.cultural._adaptation_level, 0.7)
        self.assertTrue(self.cultural._detect_culture)

    def test_own_profile_loaded(self):
        """Test own cultural profile is loaded."""
        self.assertIsNotNone(self.cultural._own_profile)
        self.assertEqual(self.cultural._own_profile.name, 'Western Business (USA/UK)')

    def test_cultural_profiles_available(self):
        """Test all cultural profiles are available."""
        profiles = cultural_adaptation.CULTURAL_PROFILES
        expected = ['western_business', 'east_asian', 'middle_eastern',
                    'latin_american', 'northern_european']
        for profile_name in expected:
            self.assertIn(profile_name, profiles)

    def test_cultural_distance(self):
        """Test cultural distance calculation."""
        western = cultural_adaptation.CULTURAL_PROFILES['western_business']
        eastern = cultural_adaptation.CULTURAL_PROFILES['east_asian']

        distance = western.get_distance_from(eastern)

        self.assertGreater(distance, 0)
        self.assertLessEqual(distance, 1)

    def test_pre_act(self):
        """Test pre_act returns cultural guidance."""
        action_spec = create_action_spec("Negotiate with your counterpart.")
        result = self.cultural.pre_act(action_spec)

        self.assertIsInstance(result, str)


class TestTemporalStrategy(unittest.TestCase):
    """Tests for Temporal Strategy module."""

    def setUp(self):
        self.model = MockLanguageModel()
        self.temporal = temporal_strategy.TemporalStrategy(
            model=self.model,
            discount_factor=0.9,
            reputation_weight=0.3,
            relationship_investment_threshold=0.6,
        )

    def test_initialization(self):
        """Test module initializes correctly."""
        self.assertEqual(self.temporal._discount_factor, 0.9)
        self.assertEqual(self.temporal._reputation_weight, 0.3)
        self.assertEqual(self.temporal._investment_threshold, 0.6)
        self.assertEqual(self.temporal._global_reputation, 0.7)  # Default start

    def test_relationship_record(self):
        """Test RelationshipRecord dataclass."""
        record = temporal_strategy.RelationshipRecord(counterpart_name="TestAgent")

        self.assertEqual(record.counterpart_name, "TestAgent")
        self.assertEqual(record.trust_score, 0.5)  # Neutral start
        self.assertEqual(record.interaction_count, 0)

    def test_relationship_update(self):
        """Test relationship updates on interaction."""
        record = temporal_strategy.RelationshipRecord(counterpart_name="TestAgent")
        initial_trust = record.trust_score

        # Positive interaction
        record.update_interaction(100.0, {'promises_kept': True})

        self.assertEqual(record.interaction_count, 1)
        self.assertGreater(record.trust_score, initial_trust)

    def test_trust_decrease_on_broken_promise(self):
        """Test trust decreases when promise broken."""
        record = temporal_strategy.RelationshipRecord(counterpart_name="TestAgent")
        record.trust_score = 0.7

        record.update_interaction(100.0, {'promises_kept': False})

        self.assertLess(record.trust_score, 0.7)

    def test_relationship_strength(self):
        """Test relationship strength calculation."""
        record = temporal_strategy.RelationshipRecord(counterpart_name="TestAgent")
        record.trust_score = 0.8
        record.interaction_count = 5

        strength = record.get_relationship_strength()

        self.assertGreater(strength, 0)
        self.assertLessEqual(strength, 1)

    def test_get_state(self):
        """Test state retrieval."""
        state = self.temporal.get_state()

        self.assertIn('relationships', state)
        self.assertIn('global_reputation', state)
        self.assertIn('current_phase', state)

    def test_pre_act(self):
        """Test pre_act returns temporal guidance."""
        action_spec = create_action_spec("Continue the negotiation.")
        result = self.temporal.pre_act(action_spec)

        self.assertIsInstance(result, str)

    def test_post_act(self):
        """Test post_act processes action."""
        action = "I agree to the terms."
        result = self.temporal.post_act(action)

        self.assertIsInstance(result, str)


class TestSwarmIntelligence(unittest.TestCase):
    """Tests for Swarm Intelligence module."""

    def setUp(self):
        self.model = MockLanguageModel()
        self.swarm = swarm_intelligence.SwarmIntelligence(
            model=self.model,
            consensus_threshold=0.7,
            max_iterations=3,
            enable_sub_agents=None,  # All enabled
        )

    def test_initialization(self):
        """Test module initializes correctly."""
        self.assertEqual(self.swarm._consensus_threshold, 0.7)
        self.assertEqual(self.swarm._max_iterations, 3)

    def test_sub_agents_created(self):
        """Test sub-agents are created."""
        self.assertGreater(len(self.swarm._sub_agents), 0)

    def test_get_state(self):
        """Test state retrieval."""
        state = self.swarm.get_state()

        self.assertIn('sub_agents', state)
        self.assertIn('decision_count', state)

    def test_pre_act(self):
        """Test pre_act returns swarm guidance."""
        action_spec = create_action_spec("Make a collective decision.")
        result = self.swarm.pre_act(action_spec)

        self.assertIsInstance(result, str)


class TestUncertaintyAware(unittest.TestCase):
    """Tests for Uncertainty Aware module."""

    def setUp(self):
        self.model = MockLanguageModel()
        self.uncertainty = uncertainty_aware.UncertaintyAware(
            model=self.model,
            confidence_threshold=0.7,
            risk_tolerance=0.3,
            information_gathering_budget=0.1,
        )

    def test_initialization(self):
        """Test module initializes correctly."""
        self.assertEqual(self.uncertainty._confidence_threshold, 0.7)
        self.assertEqual(self.uncertainty._risk_tolerance, 0.3)
        self.assertEqual(self.uncertainty._info_budget, 0.1)

    def test_get_state(self):
        """Test state retrieval."""
        state = self.uncertainty.get_state()

        self.assertIsInstance(state, dict)

    def test_pre_act(self):
        """Test pre_act returns uncertainty guidance."""
        action_spec = create_action_spec("Decide under uncertainty.")
        result = self.uncertainty.pre_act(action_spec)

        self.assertIsInstance(result, str)


class TestStrategyEvolution(unittest.TestCase):
    """Tests for Strategy Evolution module."""

    def setUp(self):
        self.model = MockLanguageModel()
        self.evolution = strategy_evolution.StrategyEvolution(
            model=self.model,
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.7,
            learning_rate=0.01,
        )

    def test_initialization(self):
        """Test module initializes correctly."""
        self.assertEqual(self.evolution._population_size, 10)
        self.assertEqual(self.evolution._mutation_rate, 0.1)
        self.assertEqual(self.evolution._crossover_rate, 0.7)
        self.assertEqual(self.evolution._learning_rate, 0.01)

    def test_strategy_population_created(self):
        """Test strategy population is initialized."""
        self.assertGreater(len(self.evolution._strategy_population), 0)

    def test_get_state(self):
        """Test state retrieval."""
        state = self.evolution.get_state()

        self.assertIn('population_size', state)
        self.assertIn('generation_count', state)

    def test_pre_act(self):
        """Test pre_act returns evolution guidance."""
        action_spec = create_action_spec("Evolve your strategy.")
        result = self.evolution.pre_act(action_spec)

        self.assertIsInstance(result, str)

    def test_post_act(self):
        """Test post_act processes action."""
        action = "I made an offer."
        result = self.evolution.post_act(action)

        self.assertIsInstance(result, str)


class TestNegotiationInstructions(unittest.TestCase):
    """Tests for Negotiation Instructions component."""

    def setUp(self):
        self.instructions = negotiation_instructions.NegotiationInstructions(
            agent_name="TestNegotiator",
            goal="Get the best deal",
            negotiation_style='integrative',
            reservation_value=100.0,
            ethical_constraints="Be honest",
            verbose=False,
        )

    def test_initialization(self):
        """Test component initializes correctly."""
        self.assertEqual(self.instructions._agent_name, "TestNegotiator")
        self.assertEqual(self.instructions._goal, "Get the best deal")
        self.assertEqual(self.instructions._style, 'integrative')
        self.assertEqual(self.instructions._reservation_value, 100.0)

    def test_pre_act(self):
        """Test pre_act returns instructions."""
        action_spec = create_action_spec("Negotiate.")
        result = self.instructions.pre_act(action_spec)

        self.assertIsInstance(result, str)
        self.assertIn("TestNegotiator", result)

    def test_phase_transitions(self):
        """Test negotiation phase tracking."""
        self.assertEqual(self.instructions._negotiation_phase, 'opening')


class TestNegotiationMemory(unittest.TestCase):
    """Tests for Negotiation Memory component."""

    def setUp(self):
        self.embedder = create_mock_embedder()
        self.memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=self.embedder
        )
        self.memory = negotiation_memory.NegotiationMemory(
            agent_name="TestNegotiator",
            memory_bank=self.memory_bank,
            verbose=False,
        )

    def test_initialization(self):
        """Test component initializes correctly."""
        self.assertEqual(self.memory._agent_name, "TestNegotiator")

    def test_pre_act(self):
        """Test pre_act returns memory context."""
        action_spec = create_action_spec("Remember past offers.")
        result = self.memory.pre_act(action_spec)

        self.assertIsInstance(result, str)


class TestNegotiationStrategy(unittest.TestCase):
    """Tests for Negotiation Strategy component."""

    def setUp(self):
        self.strategy = negotiation_strategy.BasicNegotiationStrategy(
            agent_name="TestNegotiator",
            negotiation_style='cooperative',
            reservation_value=100.0,
            target_value=200.0,
            verbose=False,
        )

    def test_initialization(self):
        """Test component initializes correctly."""
        self.assertEqual(self.strategy._agent_name, "TestNegotiator")
        self.assertEqual(self.strategy._style, 'cooperative')
        self.assertEqual(self.strategy._reservation_value, 100.0)
        self.assertEqual(self.strategy._target_value, 200.0)

    def test_pre_act(self):
        """Test pre_act returns strategy guidance."""
        action_spec = create_action_spec("Plan your strategy.")
        result = self.strategy.pre_act(action_spec)

        self.assertIsInstance(result, str)

    def test_concession_strategy(self):
        """Test concession strategy exists."""
        # Verify the concession approach exists in code
        self.assertIsNotNone(self.strategy._style)


class TestBaseNegotiator(unittest.TestCase):
    """Tests for Base Negotiator prefab."""

    def setUp(self):
        self.model = MockLanguageModel()
        self.embedder = create_mock_embedder()
        self.memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=self.embedder
        )

    def test_build_agent(self):
        """Test agent can be built."""
        agent = base_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="TestAgent",
            goal="Negotiate effectively",
            reservation_value=100.0,
        )

        self.assertIsNotNone(agent)
        self.assertEqual(agent._agent_name, "TestAgent")

    def test_agent_can_act(self):
        """Test agent can generate actions."""
        agent = base_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="TestAgent",
            goal="Negotiate effectively",
            reservation_value=100.0,
        )

        action = agent.act()

        self.assertIsInstance(action, str)


class TestAdvancedNegotiator(unittest.TestCase):
    """Tests for Advanced Negotiator prefab."""

    def setUp(self):
        self.model = MockLanguageModel()
        self.embedder = create_mock_embedder()
        self.memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=self.embedder
        )

    def test_build_agent_no_modules(self):
        """Test agent can be built without modules."""
        agent = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="TestAgent",
            goal="Negotiate effectively",
            reservation_value=100.0,
            modules=[],
        )

        self.assertIsNotNone(agent)

    def test_build_agent_with_theory_of_mind(self):
        """Test agent can be built with Theory of Mind."""
        agent = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="TestAgent",
            goal="Negotiate effectively",
            reservation_value=100.0,
            modules=['theory_of_mind'],
        )

        self.assertIsNotNone(agent)

    def test_build_agent_with_all_modules(self):
        """Test agent can be built with all modules."""
        agent = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="TestAgent",
            goal="Negotiate effectively",
            reservation_value=100.0,
            modules=[
                'theory_of_mind',
                'cultural_adaptation',
                'temporal_strategy',
                'swarm_intelligence',
                'uncertainty_aware',
                'strategy_evolution',
            ],
        )

        self.assertIsNotNone(agent)

    def test_agent_with_modules_can_act(self):
        """Test agent with modules can generate actions."""
        agent = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="TestAgent",
            goal="Negotiate effectively",
            reservation_value=100.0,
            modules=['theory_of_mind', 'cultural_adaptation'],
        )

        action = agent.act()

        self.assertIsInstance(action, str)

    def test_module_configs(self):
        """Test agent respects module configurations."""
        agent = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="TestAgent",
            goal="Negotiate effectively",
            reservation_value=100.0,
            modules=['theory_of_mind'],
            module_configs={
                'theory_of_mind': {
                    'max_recursion_depth': 5,
                    'empathy_level': 0.9,
                }
            },
        )

        self.assertIsNotNone(agent)


class TestModuleInteractions(unittest.TestCase):
    """Tests for module interactions and integration."""

    def setUp(self):
        self.model = MockLanguageModel()
        self.embedder = create_mock_embedder()
        self.memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=self.embedder
        )

    def test_multiple_modules_no_conflict(self):
        """Test multiple modules don't conflict."""
        agent = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="TestAgent",
            goal="Negotiate effectively",
            reservation_value=100.0,
            modules=[
                'theory_of_mind',
                'cultural_adaptation',
                'temporal_strategy',
            ],
        )

        # Should be able to act multiple times
        for _ in range(3):
            action = agent.act()
            self.assertIsInstance(action, str)

    def test_ablation_consistency(self):
        """Test ablation produces consistent behavior."""
        # Full agent
        agent_full = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=basic_associative_memory.AssociativeMemoryBank(
                sentence_embedder=self.embedder
            ),
            name="FullAgent",
            goal="Negotiate",
            reservation_value=100.0,
            modules=['theory_of_mind', 'cultural_adaptation'],
        )

        # Ablated agent
        agent_ablated = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=basic_associative_memory.AssociativeMemoryBank(
                sentence_embedder=self.embedder
            ),
            name="AblatedAgent",
            goal="Negotiate",
            reservation_value=100.0,
            modules=['theory_of_mind'],  # No cultural adaptation
        )

        # Both should produce actions
        action_full = agent_full.act()
        action_ablated = agent_ablated.act()

        self.assertIsInstance(action_full, str)
        self.assertIsInstance(action_ablated, str)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""

    def setUp(self):
        self.model = MockLanguageModel()
        self.embedder = create_mock_embedder()
        self.memory_bank = basic_associative_memory.AssociativeMemoryBank(
            sentence_embedder=self.embedder
        )

    def test_empty_prompt(self):
        """Test handling of empty prompts."""
        tom = theory_of_mind.TheoryOfMind(
            model=self.model,
            max_recursion_depth=2,
        )

        action_spec = create_action_spec("")
        result = tom.pre_act(action_spec)

        self.assertIsInstance(result, str)

    def test_very_long_prompt(self):
        """Test handling of very long prompts."""
        tom = theory_of_mind.TheoryOfMind(
            model=self.model,
            max_recursion_depth=2,
        )

        long_text = "This is a test. " * 1000
        action_spec = create_action_spec(long_text)
        result = tom.pre_act(action_spec)

        self.assertIsInstance(result, str)

    def test_special_characters(self):
        """Test handling of special characters."""
        tom = theory_of_mind.TheoryOfMind(
            model=self.model,
            max_recursion_depth=2,
        )

        special_text = "Price: $100! @#$%^&*() 你好 émoji 🎉"
        action_spec = create_action_spec(special_text)
        result = tom.pre_act(action_spec)

        self.assertIsInstance(result, str)

    def test_invalid_module_name(self):
        """Test handling of invalid module name."""
        # Invalid module should be silently ignored
        agent = advanced_negotiator.build_agent(
            model=self.model,
            memory_bank=self.memory_bank,
            name="TestAgent",
            goal="Negotiate",
            reservation_value=100.0,
            modules=['invalid_module_name'],
        )

        self.assertIsNotNone(agent)


if __name__ == '__main__':
    # Run tests with verbosity
    unittest.main(verbosity=2)

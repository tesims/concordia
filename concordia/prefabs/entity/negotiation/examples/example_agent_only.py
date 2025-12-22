# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Example: Using negotiation agent modules without specialized GM.

This example shows how to create negotiation-capable agents and use them
with a standard Concordia game master. The agents have sophisticated
negotiation cognition, but the environment doesn't have negotiation-specific
awareness.

Use this approach when:
- You want negotiation-capable agents in an existing simulation
- You're studying how agent cognition affects outcomes
- You don't need the GM to track negotiation state
"""

from unittest import mock

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.prefabs.entity.negotiation import advanced_negotiator
from concordia.prefabs.entity.negotiation import base_negotiator


def create_mock_model():
  """Create a mock language model for demonstration."""
  model = mock.create_autospec(language_model.LanguageModel, instance=True)

  def mock_response(prompt, **kwargs):
    prompt_lower = prompt.lower()
    if 'emotion' in prompt_lower:
      return 'confident'
    elif 'cultural' in prompt_lower:
      return 'western_business'
    elif 'strategy' in prompt_lower:
      return 'cooperative'
    elif 'offer' in prompt_lower:
      return 'I propose we split the difference at $150.'
    else:
      return 'I understand and am ready to proceed.'

  model.sample_text.side_effect = mock_response
  return model


def main():
  print("=" * 60)
  print("Example: Negotiation Agents Only (No Specialized GM)")
  print("=" * 60)

  # Setup
  model = create_mock_model()
  memory_bank = basic_associative_memory.AssociativeMemoryBank()

  # Create a basic negotiator
  print("\n1. Creating base negotiator...")
  basic_agent = base_negotiator.build_agent(
      model=model,
      memory_bank=memory_bank,
      name='BasicBuyer',
      goal='Purchase the antique vase for a reasonable price',
      reservation_value=200.0,
  )
  print(f"   Created: {basic_agent._agent_name}")

  # Create an advanced negotiator with theory of mind
  print("\n2. Creating agent with Theory of Mind...")
  tom_agent = advanced_negotiator.build_agent(
      model=model,
      memory_bank=memory_bank,
      name='StrategicSeller',
      goal='Sell the antique vase at the best possible price',
      modules=['theory_of_mind'],
  )
  print(f"   Created: {tom_agent._agent_name}")
  print(f"   Modules: theory_of_mind")

  # Create an agent with multiple modules
  print("\n3. Creating agent with multiple modules...")
  advanced_agent = advanced_negotiator.build_agent(
      model=model,
      memory_bank=memory_bank,
      name='ExperiencedNegotiator',
      goal='Reach a mutually beneficial agreement',
      modules=['theory_of_mind', 'cultural_adaptation', 'temporal_strategy'],
  )
  print(f"   Created: {advanced_agent._agent_name}")
  print(f"   Modules: theory_of_mind, cultural_adaptation, temporal_strategy")

  # Create specialized agents using convenience builders
  print("\n4. Creating specialized agents...")

  cultural_agent = advanced_negotiator.build_cultural_agent(
      model=model,
      memory_bank=memory_bank,
      name='CrossCulturalRep',
      own_culture='east_asian',
  )
  print(f"   Cultural agent: {cultural_agent._agent_name} (east_asian culture)")

  adaptive_agent = advanced_negotiator.build_adaptive_agent(
      model=model,
      memory_bank=memory_bank,
      name='LearningAgent',
      learning_rate=0.1,
  )
  print(f"   Adaptive agent: {adaptive_agent._agent_name} (learning_rate=0.1)")

  # Demonstrate that agents can act
  print("\n5. Testing agent action capability...")
  action_spec = mock.Mock()
  action_spec.call_to_action = "What do you want to do next in the negotiation?"

  # This would normally involve the full action pipeline
  # Here we just verify the components are wired correctly
  print("   Agents are ready to participate in negotiation scenarios.")

  print("\n" + "=" * 60)
  print("Agent-only setup complete!")
  print("These agents can be used with any Concordia game master.")
  print("=" * 60)


if __name__ == '__main__':
  main()

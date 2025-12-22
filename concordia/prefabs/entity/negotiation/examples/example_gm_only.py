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

"""Example: Using negotiation GM modules with standard agents.

This example shows how to create a negotiation-aware game master that
works with basic Concordia agents. The GM tracks negotiation state,
enforces rules, and provides negotiation-relevant observations, even
though the agents themselves don't have specialized negotiation modules.

Use this approach when:
- You have existing agents you want to put in a negotiation environment
- You're studying how environmental factors affect outcomes
- You need negotiation tracking but agents are simple or scripted
"""

from unittest import mock

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.prefabs.entity.negotiation import base_negotiator
from concordia.prefabs.game_master.negotiation import negotiation


def create_mock_model():
  """Create a mock language model for demonstration."""
  model = mock.create_autospec(language_model.LanguageModel, instance=True)

  def mock_response(prompt, **kwargs):
    prompt_lower = prompt.lower()
    if 'emotion' in prompt_lower:
      return 'neutral'
    elif 'deception' in prompt_lower:
      return 'no deception detected'
    elif 'cultural' in prompt_lower:
      return 'appropriate'
    elif 'strategy' in prompt_lower:
      return 'cooperative'
    else:
      return 'The negotiation continues normally.'

  model.sample_text.side_effect = mock_response
  return model


def main():
  print("=" * 60)
  print("Example: Negotiation GM Only (With Basic Agents)")
  print("=" * 60)

  # Setup
  model = create_mock_model()
  memory_bank = basic_associative_memory.AssociativeMemoryBank()

  # Create simple agents (using base_negotiator for simplicity, but these
  # could be any Concordia agents)
  print("\n1. Creating basic agents...")
  agent1 = base_negotiator.build_agent(
      model=model,
      memory_bank=memory_bank,
      name='PartyA',
      goal='Reach an agreement',
  )
  agent2 = base_negotiator.build_agent(
      model=model,
      memory_bank=memory_bank,
      name='PartyB',
      goal='Reach an agreement',
  )
  print(f"   Created: {agent1._agent_name}, {agent2._agent_name}")

  # Create negotiation GM with specific modules
  print("\n2. Creating negotiation GM with social intelligence...")
  gm_social = negotiation.build_game_master(
      model=model,
      memory_bank=memory_bank,
      entities=[agent1, agent2],
      name='SocialAwareMediator',
      gm_modules=['social_intelligence'],
  )
  print(f"   Created: {gm_social._agent_name}")
  print(f"   GM Modules: social_intelligence")

  # Create GM with temporal dynamics
  print("\n3. Creating negotiation GM with temporal dynamics...")
  gm_temporal = negotiation.build_game_master(
      model=model,
      memory_bank=memory_bank,
      entities=[agent1, agent2],
      name='DeadlineEnforcer',
      gm_modules=['temporal_dynamics'],
      max_rounds=10,
  )
  print(f"   Created: {gm_temporal._agent_name}")
  print(f"   GM Modules: temporal_dynamics (max 10 rounds)")

  # Create GM with multiple modules
  print("\n4. Creating GM with multiple modules...")
  gm_full = negotiation.build_game_master(
      model=model,
      memory_bank=memory_bank,
      entities=[agent1, agent2],
      name='ComprehensiveMediator',
      gm_modules=[
          'social_intelligence',
          'temporal_dynamics',
          'uncertainty_management',
      ],
  )
  print(f"   Created: {gm_full._agent_name}")
  print(f"   GM Modules: social_intelligence, temporal_dynamics, uncertainty_management")

  # Demonstrate negotiation state tracking
  print("\n5. Testing negotiation state tracking...")
  state_component = gm_full._context_components.get('negotiation_state')
  if state_component:
    # Start a negotiation
    state = state_component.start_negotiation(
        negotiation_id='demo_negotiation',
        participants=['PartyA', 'PartyB'],
    )
    print(f"   Started negotiation: {state.negotiation_id}")
    print(f"   Participants: {state.participants}")
    print(f"   Phase: {state.phase}")

    # Record an offer
    offer = state_component.record_offer(
        negotiation_id='demo_negotiation',
        offerer='PartyA',
        recipient='PartyB',
        offer_type='initial',
        terms={'price': 100, 'delivery': '2 weeks'},
    )
    print(f"   Recorded offer from {offer.offerer}: {offer.terms}")

  print("\n" + "=" * 60)
  print("GM-only setup complete!")
  print("The GM tracks negotiation state and provides context,")
  print("even with basic agents that lack negotiation modules.")
  print("=" * 60)


if __name__ == '__main__':
  main()

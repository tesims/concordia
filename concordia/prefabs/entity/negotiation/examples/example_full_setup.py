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

"""Example: Full negotiation setup with both agent and GM modules.

This example demonstrates the complete negotiation framework with
sophisticated agents AND a negotiation-aware game master. Both layers
work together to create realistic negotiation dynamics.

Use this approach when:
- You want the richest possible simulation
- Agent cognition and environmental feedback should interact
- You're studying complex dynamics like cultural friction or trust
"""

from unittest import mock

from concordia.associative_memory import basic_associative_memory
from concordia.language_model import language_model
from concordia.prefabs.entity.negotiation import advanced_negotiator
from concordia.prefabs.game_master.negotiation import negotiation


def create_mock_model():
  """Create a mock language model for demonstration."""
  model = mock.create_autospec(language_model.LanguageModel, instance=True)

  def mock_response(prompt, **kwargs):
    prompt_lower = prompt.lower()
    if 'emotion' in prompt_lower:
      return 'cautiously optimistic'
    elif 'cultural' in prompt_lower:
      return 'respectful and indirect'
    elif 'strategy' in prompt_lower:
      return 'integrative'
    elif 'trust' in prompt_lower:
      return 'moderate trust, building'
    elif 'offer' in prompt_lower:
      return 'I would like to propose a collaborative solution.'
    elif 'coalition' in prompt_lower:
      return 'potential alignment detected'
    else:
      return 'Proceeding with the negotiation.'

  model.sample_text.side_effect = mock_response
  return model


def main():
  print("=" * 60)
  print("Example: Full Negotiation Setup (Agents + GM)")
  print("=" * 60)

  # Setup
  model = create_mock_model()
  memory_bank = basic_associative_memory.AssociativeMemoryBank()

  # Create sophisticated agents with complementary modules
  print("\n1. Creating advanced negotiation agents...")

  # Western business representative with theory of mind and temporal awareness
  western_rep = advanced_negotiator.build_agent(
      model=model,
      memory_bank=memory_bank,
      name='WesternRep',
      goal='Secure favorable contract terms while maintaining relationship',
      modules=['theory_of_mind', 'temporal_strategy', 'cultural_adaptation'],
      module_configs={
          'cultural_adaptation': {'own_culture': 'western_business'},
      },
  )
  print(f"   Created: {western_rep._agent_name}")
  print(f"   Modules: theory_of_mind, temporal_strategy, cultural_adaptation")
  print(f"   Culture: western_business")

  # East Asian representative with cultural adaptation and uncertainty handling
  eastern_rep = advanced_negotiator.build_agent(
      model=model,
      memory_bank=memory_bank,
      name='EasternRep',
      goal='Build long-term partnership with acceptable terms',
      modules=['cultural_adaptation', 'uncertainty_aware', 'theory_of_mind'],
      module_configs={
          'cultural_adaptation': {'own_culture': 'east_asian'},
      },
  )
  print(f"   Created: {eastern_rep._agent_name}")
  print(f"   Modules: cultural_adaptation, uncertainty_aware, theory_of_mind")
  print(f"   Culture: east_asian")

  agents = [western_rep, eastern_rep]

  # Create GM with matching modules
  print("\n2. Creating negotiation GM with full module suite...")
  gm = negotiation.build_game_master(
      model=model,
      memory_bank=memory_bank,
      entities=agents,
      name='CrossCulturalMediator',
      gm_modules=[
          'social_intelligence',
          'cultural_awareness',
          'temporal_dynamics',
          'uncertainty_management',
      ],
  )
  print(f"   Created: {gm._agent_name}")
  print(f"   GM Modules: social_intelligence, cultural_awareness,")
  print(f"               temporal_dynamics, uncertainty_management")

  # Demonstrate the interaction between agent and GM modules
  print("\n3. Module interactions...")
  print("   Agent modules (first-person):")
  print("   - Agents model each other's mental states (theory_of_mind)")
  print("   - Agents adapt communication style (cultural_adaptation)")
  print("   - Agents track relationship history (temporal_strategy)")
  print("")
  print("   GM modules (third-person):")
  print("   - GM detects social dynamics and deception (social_intelligence)")
  print("   - GM enforces cultural norms (cultural_awareness)")
  print("   - GM manages deadlines and commitments (temporal_dynamics)")

  # Show negotiation state setup
  print("\n4. Initializing negotiation state...")
  state_component = gm._context_components.get('negotiation_state')
  if state_component:
    state = state_component.start_negotiation(
        negotiation_id='cross_cultural_deal',
        participants=['WesternRep', 'EasternRep'],
    )
    print(f"   Negotiation ID: {state.negotiation_id}")
    print(f"   Phase: {state.phase}")
    print(f"   Round: {state.current_round}")

  # Show cultural awareness setup
  print("\n5. Cultural context setup...")
  cultural_component = gm._context_components.get('gm_module_cultural_awareness')
  if cultural_component:
    cultural_component.set_participant_culture('WesternRep', 'western_business')
    cultural_component.set_participant_culture('EasternRep', 'east_asian')
    print("   WesternRep: western_business profile")
    print("   EasternRep: east_asian profile")
    print("   GM will monitor for cultural norm violations")

  # Demonstrate a potential cultural friction scenario
  print("\n6. Example: Cultural friction detection...")
  if cultural_component:
    # Direct criticism might violate face-saving norms
    violation = cultural_component.detect_cultural_violation(
        actor='WesternRep',
        action='Your proposal is completely unacceptable and poorly thought out.',
        recipient='EasternRep',
    )
    if violation:
      print(f"   Detected violation: {violation[:60]}...")
    else:
      print("   No violation detected")

  print("\n" + "=" * 60)
  print("Full setup complete!")
  print("")
  print("This configuration enables:")
  print("- Agents that reason about each other (theory of mind)")
  print("- Cultural adaptation on both sides")
  print("- GM enforcement of cultural norms")
  print("- Trust tracking and relationship dynamics")
  print("- Information asymmetry management")
  print("=" * 60)


if __name__ == '__main__':
  main()

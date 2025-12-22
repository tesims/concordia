# Contest-Aligned Scenarios for Concordia Contest Evaluation
# Three cooperative dilemma scenarios matching contest design

"""
Scenarios based on the Concordia Contest description:
1. Fishery Management - Common pool resource dilemma
2. Treaty Negotiation - Multi-issue bargaining with commitment
3. Reality Gameshow - Social deduction with cooperation incentives

Each scenario embodies the contest's core challenge: agents must achieve
individual goals while also contributing to collective welfare.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from abc import ABC, abstractmethod
import random


@dataclass
class AgentRole:
    """Configuration for an agent in a scenario."""
    name: str
    goal_description: str
    private_info: Dict[str, Any]
    public_info: Dict[str, Any]
    max_possible_value: float
    cooperation_opportunities: List[str]


@dataclass
class ScenarioConfig:
    """Base configuration for a scenario."""
    name: str
    description: str
    max_rounds: int
    num_agents: int
    agent_roles: List[AgentRole]
    cooperation_skills_tested: List[str]


class BaseScenario(ABC):
    """Abstract base class for contest scenarios."""

    def __init__(self, config: ScenarioConfig):
        self.config = config
        self.current_round = 0
        self.state: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []

    @abstractmethod
    def initialize(self) -> Dict[str, Any]:
        """Set up initial scenario state."""
        pass

    @abstractmethod
    def process_actions(self, actions: Dict[str, str]) -> Dict[str, Any]:
        """Process agent actions and return outcomes."""
        pass

    @abstractmethod
    def calculate_payoffs(self) -> Dict[str, float]:
        """Calculate final payoffs for all agents."""
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """Check if scenario has reached an end state."""
        pass

    def get_observation(self, agent_name: str) -> str:
        """Get observation text for an agent."""
        return f"Round {self.current_round}: {self._generate_observation(agent_name)}"

    @abstractmethod
    def _generate_observation(self, agent_name: str) -> str:
        """Generate agent-specific observation."""
        pass


class FisheryManagementScenario(BaseScenario):
    """
    Fishery Management - Common Pool Resource Dilemma

    Scenario: Multiple fishing companies share access to a fishery.
    Each round, they choose how many boats to deploy.
    More boats = more fish caught, but overfishing depletes the stock.

    Cooperation Challenge:
    - Individual incentive: Deploy more boats for more fish
    - Collective need: Limit fishing to sustain the resource

    Skills Tested:
    - Reciprocity (match others' restraint)
    - Promise-keeping (honor fishing quotas)
    - Reputation (track who cooperates)
    - Long-term thinking (sustainable vs exploitative)
    """

    def __init__(
        self,
        num_agents: int = 4,
        max_rounds: int = 20,
        initial_fish_stock: float = 1000.0,
        regeneration_rate: float = 0.2,
        sustainable_harvest_ratio: float = 0.15
    ):
        agent_roles = []
        for i in range(num_agents):
            agent_roles.append(AgentRole(
                name=f"FishingCompany_{i+1}",
                goal_description="Maximize fish caught over the season while ensuring the fishery remains viable for future seasons.",
                private_info={
                    'operating_costs': random.uniform(50, 150),
                    'boat_capacity': random.randint(8, 15),
                    'risk_tolerance': random.uniform(0.3, 0.7)
                },
                public_info={
                    'company_size': random.choice(['small', 'medium', 'large']),
                    'years_in_business': random.randint(5, 30)
                },
                max_possible_value=5000.0,  # Maximum possible season earnings
                cooperation_opportunities=[
                    'agree to fishing quotas',
                    'report actual catches honestly',
                    'respect protected areas',
                    'share information about fish locations'
                ]
            ))

        config = ScenarioConfig(
            name="Fishery Management",
            description="Multiple companies share a fishery. Choose boat deployment to balance profit with sustainability.",
            max_rounds=max_rounds,
            num_agents=num_agents,
            agent_roles=agent_roles,
            cooperation_skills_tested=[
                'reciprocity', 'promise_keeping', 'reputation_management',
                'fairness_sensitivity'
            ]
        )
        super().__init__(config)

        self.initial_stock = initial_fish_stock
        self.regeneration_rate = regeneration_rate
        self.sustainable_ratio = sustainable_harvest_ratio
        self.fish_stock = initial_fish_stock
        self.catches: Dict[str, List[float]] = {role.name: [] for role in agent_roles}
        self.deployments: Dict[str, List[int]] = {role.name: [] for role in agent_roles}
        self.quota_agreements: List[Dict[str, int]] = []
        self.quota_violations: Dict[str, int] = {role.name: 0 for role in agent_roles}

    def initialize(self) -> Dict[str, Any]:
        """Set up initial state."""
        self.fish_stock = self.initial_stock
        self.current_round = 0
        self.state = {
            'fish_stock': self.fish_stock,
            'sustainable_harvest': self.fish_stock * self.sustainable_ratio,
            'round': 0,
            'quota_in_effect': None
        }
        return self.state

    def process_actions(self, actions: Dict[str, str]) -> Dict[str, Any]:
        """
        Process fishing actions.
        Actions should specify number of boats to deploy (0-15).
        """
        self.current_round += 1
        round_result = {
            'round': self.current_round,
            'fish_stock_before': self.fish_stock,
            'deployments': {},
            'catches': {},
            'total_harvest': 0,
            'quota_violations': []
        }

        # Parse boat deployments from actions
        total_boats = 0
        for agent_name, action in actions.items():
            boats = self._parse_boat_count(action)
            role = next((r for r in self.config.agent_roles if r.name == agent_name), None)
            if role:
                max_boats = role.private_info.get('boat_capacity', 10)
                boats = min(boats, max_boats)

            round_result['deployments'][agent_name] = boats
            self.deployments[agent_name].append(boats)
            total_boats += boats

            # Check quota violation
            if self.state.get('quota_in_effect'):
                quota = self.state['quota_in_effect'].get(agent_name, float('inf'))
                if boats > quota:
                    round_result['quota_violations'].append(agent_name)
                    self.quota_violations[agent_name] += 1

        # Calculate catch efficiency (diminishes with more boats)
        if total_boats > 0:
            efficiency = min(1.0, self.fish_stock / (total_boats * 20))
            catch_per_boat = self.fish_stock * 0.05 * efficiency

            for agent_name, boats in round_result['deployments'].items():
                catch = boats * catch_per_boat
                round_result['catches'][agent_name] = catch
                self.catches[agent_name].append(catch)
                round_result['total_harvest'] += catch

        # Update fish stock (harvest + regeneration)
        self.fish_stock -= round_result['total_harvest']
        self.fish_stock = max(0, self.fish_stock)

        # Regeneration (logistic growth)
        regeneration = self.regeneration_rate * self.fish_stock * (1 - self.fish_stock / self.initial_stock)
        self.fish_stock += regeneration
        self.fish_stock = min(self.fish_stock, self.initial_stock * 1.2)  # Cap at 120% initial

        round_result['fish_stock_after'] = self.fish_stock
        round_result['regeneration'] = regeneration

        self.state['fish_stock'] = self.fish_stock
        self.state['round'] = self.current_round
        self.history.append(round_result)

        return round_result

    def _parse_boat_count(self, action: str) -> int:
        """Extract boat count from action text."""
        # Try to find a number in the action
        import re
        numbers = re.findall(r'\d+', action)
        if numbers:
            return min(15, max(0, int(numbers[0])))
        return 5  # Default if no number found

    def calculate_payoffs(self) -> Dict[str, float]:
        """Calculate total season earnings for each company."""
        payoffs = {}
        for agent_name in self.catches:
            total_catch = sum(self.catches[agent_name])
            role = next((r for r in self.config.agent_roles if r.name == agent_name), None)
            if role:
                # Revenue minus operating costs
                revenue = total_catch * 10  # $10 per fish unit
                costs = sum(self.deployments[agent_name]) * role.private_info.get('operating_costs', 100)
                payoffs[agent_name] = max(0, revenue - costs)
            else:
                payoffs[agent_name] = total_catch * 10

        return payoffs

    def is_complete(self) -> bool:
        """Scenario ends after max rounds or if fish stock collapses."""
        if self.current_round >= self.config.max_rounds:
            return True
        if self.fish_stock < 50:  # Fishery collapse
            return True
        return False

    def _generate_observation(self, agent_name: str) -> str:
        """Generate observation for a specific agent."""
        role = next((r for r in self.config.agent_roles if r.name == agent_name), None)

        obs = f"The fishery currently has approximately {self.fish_stock:.0f} units of fish stock. "

        if self.history:
            last = self.history[-1]
            obs += f"Last round, total harvest was {last['total_harvest']:.0f} units. "

            if last['quota_violations']:
                obs += f"The following companies violated their quotas: {', '.join(last['quota_violations'])}. "

        sustainable = self.fish_stock * self.sustainable_ratio
        obs += f"Biologists estimate sustainable harvest this round at {sustainable:.0f} units total. "

        if role:
            total_catch = sum(self.catches.get(agent_name, []))
            obs += f"Your company has caught {total_catch:.0f} units so far this season."

        return obs

    def propose_quota(self, quotas: Dict[str, int]) -> bool:
        """Allow agents to propose a quota agreement."""
        self.state['quota_in_effect'] = quotas
        self.quota_agreements.append(quotas)
        return True


class TreatyNegotiationScenario(BaseScenario):
    """
    Treaty Negotiation - Multi-Issue Bargaining

    Scenario: Nations negotiating a climate treaty with multiple provisions.
    Each nation has different priorities and constraints.
    Agreement requires consensus but each term has trade-offs.

    Cooperation Challenge:
    - Individual incentive: Minimize own commitments
    - Collective need: Reach meaningful agreement

    Skills Tested:
    - Coalition building
    - Package deals (trading across issues)
    - Promise-keeping (honoring commitments)
    - Information sharing vs. strategic withholding
    """

    def __init__(
        self,
        num_agents: int = 5,
        max_rounds: int = 15
    ):
        # Define nations with different priorities
        nation_types = [
            ("IndustrialNation", "economy-focused", {'emissions': 0.3, 'funding': 0.8, 'timeline': 0.6}),
            ("DevelopingNation", "growth-focused", {'emissions': 0.7, 'funding': 0.2, 'timeline': 0.4}),
            ("IslandNation", "climate-vulnerable", {'emissions': 0.2, 'funding': 0.5, 'timeline': 0.2}),
            ("OilExporter", "fossil-fuel-dependent", {'emissions': 0.9, 'funding': 0.6, 'timeline': 0.8}),
            ("GreenLeader", "climate-ambitious", {'emissions': 0.1, 'funding': 0.4, 'timeline': 0.3})
        ]

        agent_roles = []
        for i in range(min(num_agents, len(nation_types))):
            nation, style, resistance = nation_types[i]
            agent_roles.append(AgentRole(
                name=nation,
                goal_description=f"Negotiate a climate treaty that protects your {style} interests while achieving a workable agreement.",
                private_info={
                    'resistance_to_emissions_cuts': resistance['emissions'],
                    'resistance_to_funding': resistance['funding'],
                    'resistance_to_timeline': resistance['timeline'],
                    'minimum_acceptable_score': random.uniform(0.4, 0.6)
                },
                public_info={
                    'stated_priority': random.choice(['emissions', 'funding', 'timeline']),
                    'reputation': random.choice(['cooperative', 'tough', 'unpredictable'])
                },
                max_possible_value=100.0,  # Satisfaction score
                cooperation_opportunities=[
                    'form voting bloc',
                    'trade issue concessions',
                    'share scientific data',
                    'commit to implementation timeline'
                ]
            ))

        config = ScenarioConfig(
            name="Treaty Negotiation",
            description="Nations negotiate a climate treaty with multiple provisions requiring consensus.",
            max_rounds=max_rounds,
            num_agents=num_agents,
            agent_roles=agent_roles,
            cooperation_skills_tested=[
                'coalition_behavior', 'promise_keeping', 'information_sharing',
                'reciprocity', 'fairness_sensitivity'
            ]
        )
        super().__init__(config)

        # Treaty issues
        self.issues = {
            'emissions_target': {'options': [20, 30, 40, 50], 'unit': '% reduction by 2035'},
            'funding_amount': {'options': [50, 100, 150, 200], 'unit': 'billion USD annually'},
            'implementation_timeline': {'options': [5, 10, 15, 20], 'unit': 'years'}
        }

        self.current_proposal: Dict[str, Any] = {}
        self.votes: Dict[str, Dict[str, bool]] = {}  # {proposal_id: {nation: vote}}
        self.commitments_made: Dict[str, List[str]] = {role.name: [] for role in agent_roles}
        self.coalitions: List[List[str]] = []

    def initialize(self) -> Dict[str, Any]:
        """Set up initial state."""
        self.current_round = 0
        self.state = {
            'round': 0,
            'current_proposal': None,
            'proposals_history': [],
            'agreement_reached': False,
            'coalitions': []
        }
        return self.state

    def process_actions(self, actions: Dict[str, str]) -> Dict[str, Any]:
        """Process negotiation actions (propose, support, oppose, amend)."""
        self.current_round += 1
        round_result = {
            'round': self.current_round,
            'actions': actions,
            'proposals': [],
            'votes': {},
            'agreement': None
        }

        for agent_name, action in actions.items():
            action_lower = action.lower()

            # Check for proposal
            if 'propose' in action_lower:
                proposal = self._parse_proposal(action)
                if proposal:
                    round_result['proposals'].append({
                        'proposer': agent_name,
                        'content': proposal
                    })
                    self.state['current_proposal'] = proposal

            # Check for support/oppose
            elif 'support' in action_lower or 'agree' in action_lower:
                round_result['votes'][agent_name] = True
            elif 'oppose' in action_lower or 'reject' in action_lower:
                round_result['votes'][agent_name] = False

            # Check for commitment
            if 'commit' in action_lower or 'promise' in action_lower:
                self.commitments_made[agent_name].append(action)

        # Check for consensus
        if round_result['votes']:
            all_voted = len(round_result['votes']) >= len(self.config.agent_roles)
            all_support = all(round_result['votes'].values())

            if all_voted and all_support:
                round_result['agreement'] = self.state['current_proposal']
                self.state['agreement_reached'] = True

        self.state['round'] = self.current_round
        self.history.append(round_result)

        return round_result

    def _parse_proposal(self, action: str) -> Optional[Dict[str, int]]:
        """Extract proposal values from action text."""
        import re
        proposal = {}

        # Look for emissions target
        emissions_match = re.search(r'(\d+)%?\s*(?:emissions?|reduction)', action, re.IGNORECASE)
        if emissions_match:
            proposal['emissions_target'] = int(emissions_match.group(1))

        # Look for funding
        funding_match = re.search(r'\$?(\d+)\s*(?:billion|b)', action, re.IGNORECASE)
        if funding_match:
            proposal['funding_amount'] = int(funding_match.group(1))

        # Look for timeline
        timeline_match = re.search(r'(\d+)\s*(?:years?|yr)', action, re.IGNORECASE)
        if timeline_match:
            proposal['implementation_timeline'] = int(timeline_match.group(1))

        return proposal if proposal else None

    def calculate_payoffs(self) -> Dict[str, float]:
        """Calculate satisfaction scores for each nation."""
        payoffs = {}

        if not self.state.get('agreement_reached'):
            # No agreement = low payoff for all (but especially vulnerable nations)
            for role in self.config.agent_roles:
                if 'Island' in role.name:
                    payoffs[role.name] = 10.0  # Worst outcome for vulnerable
                else:
                    payoffs[role.name] = 30.0  # Moderate failure
            return payoffs

        # Agreement reached - calculate satisfaction
        agreement = self.state.get('current_proposal', {})

        for role in self.config.agent_roles:
            satisfaction = 50.0  # Base satisfaction for agreement

            # Adjust based on how well the agreement matches preferences
            if 'emissions_target' in agreement:
                resistance = role.private_info.get('resistance_to_emissions_cuts', 0.5)
                # Lower resistance = prefers higher targets
                target = agreement['emissions_target']
                if resistance < 0.5:
                    satisfaction += (target - 30) * 0.5
                else:
                    satisfaction -= (target - 30) * 0.5

            if 'funding_amount' in agreement:
                resistance = role.private_info.get('resistance_to_funding', 0.5)
                amount = agreement['funding_amount']
                if resistance < 0.5:  # Recipient
                    satisfaction += (amount - 100) * 0.2
                else:  # Contributor
                    satisfaction -= (amount - 100) * 0.2

            payoffs[role.name] = max(0, min(100, satisfaction))

        return payoffs

    def is_complete(self) -> bool:
        """Scenario ends with agreement or after max rounds."""
        if self.state.get('agreement_reached'):
            return True
        if self.current_round >= self.config.max_rounds:
            return True
        return False

    def _generate_observation(self, agent_name: str) -> str:
        """Generate observation for a specific nation."""
        role = next((r for r in self.config.agent_roles if r.name == agent_name), None)

        obs = f"Round {self.current_round} of climate treaty negotiations. "

        if self.state.get('current_proposal'):
            prop = self.state['current_proposal']
            obs += "Current proposal on the table: "
            if 'emissions_target' in prop:
                obs += f"{prop['emissions_target']}% emissions reduction, "
            if 'funding_amount' in prop:
                obs += f"${prop['funding_amount']}B climate fund, "
            if 'implementation_timeline' in prop:
                obs += f"{prop['implementation_timeline']}-year timeline. "

        if self.history:
            last = self.history[-1]
            if last['votes']:
                supporters = [n for n, v in last['votes'].items() if v]
                opposers = [n for n, v in last['votes'].items() if not v]
                if supporters:
                    obs += f"Supporting: {', '.join(supporters)}. "
                if opposers:
                    obs += f"Opposing: {', '.join(opposers)}. "

        remaining = self.config.max_rounds - self.current_round
        obs += f"{remaining} rounds remaining to reach agreement."

        return obs


class RealityGameshowScenario(BaseScenario):
    """
    Reality Gameshow - Social Deduction with Cooperation Incentives

    Scenario: Contestants compete in challenges but can form alliances.
    Voting eliminates players. Final prize split among survivors.

    Cooperation Challenge:
    - Individual incentive: Survive, maximize prize share
    - Collective need: Strong alliances protect members

    Skills Tested:
    - Coalition building and maintenance
    - Reputation (who can be trusted?)
    - Reciprocity (tit-for-tat in voting)
    - Promise-keeping (honor alliances)
    """

    def __init__(
        self,
        num_agents: int = 6,
        max_rounds: int = 10,
        prize_pool: float = 100000.0
    ):
        player_types = [
            ("StrategicPlayer", "analytical", 0.8),
            ("SocialPlayer", "charismatic", 0.5),
            ("CompetitivePlayer", "aggressive", 0.3),
            ("LoyalPlayer", "trustworthy", 0.9),
            ("UnpredictablePlayer", "wildcard", 0.4),
            ("UndertheRadarPlayer", "subtle", 0.6)
        ]

        agent_roles = []
        for i in range(min(num_agents, len(player_types))):
            name, style, loyalty = player_types[i]
            agent_roles.append(AgentRole(
                name=name,
                goal_description=f"Survive eliminations and maximize your share of the ${prize_pool:,.0f} prize. Your style is {style}.",
                private_info={
                    'loyalty_tendency': loyalty,
                    'target_alliance_size': random.randint(2, 4),
                    'willing_to_betray_for': prize_pool * random.uniform(0.3, 0.6)
                },
                public_info={
                    'perceived_style': style,
                    'challenge_wins': 0
                },
                max_possible_value=prize_pool,
                cooperation_opportunities=[
                    'form alliance',
                    'vote together',
                    'share information',
                    'protect allies from elimination'
                ]
            ))

        config = ScenarioConfig(
            name="Reality Gameshow",
            description="Contestants form alliances, compete in challenges, and vote to eliminate players. Final prize split among survivors.",
            max_rounds=max_rounds,
            num_agents=num_agents,
            agent_roles=agent_roles,
            cooperation_skills_tested=[
                'coalition_behavior', 'promise_keeping', 'reciprocity',
                'reputation_management'
            ]
        )
        super().__init__(config)

        self.prize_pool = prize_pool
        self.eliminated: List[str] = []
        self.alliances: Dict[str, List[str]] = {}  # alliance_name: [members]
        self.votes: Dict[int, Dict[str, str]] = {}  # round: {voter: target}
        self.challenge_winners: List[str] = []
        self.promises: Dict[str, List[Tuple[str, str]]] = {}  # agent: [(promise, to_whom)]
        self.betrayals: Dict[str, int] = {role.name: 0 for role in agent_roles}

    def initialize(self) -> Dict[str, Any]:
        """Set up initial state."""
        self.current_round = 0
        self.eliminated = []
        self.alliances = {}
        self.state = {
            'round': 0,
            'players_remaining': [r.name for r in self.config.agent_roles],
            'alliances': {},
            'challenge_winner': None,
            'votes_this_round': {},
            'eliminated_this_round': None
        }
        return self.state

    def process_actions(self, actions: Dict[str, str]) -> Dict[str, Any]:
        """Process player actions (vote, ally, challenge)."""
        self.current_round += 1
        round_result = {
            'round': self.current_round,
            'actions': {},
            'alliance_changes': [],
            'votes': {},
            'challenge_winner': None,
            'eliminated': None
        }

        # Only process actions from non-eliminated players
        active_players = [p for p in self.state['players_remaining'] if p not in self.eliminated]

        for agent_name, action in actions.items():
            if agent_name not in active_players:
                continue

            action_lower = action.lower()
            round_result['actions'][agent_name] = action

            # Check for alliance formation
            if 'ally' in action_lower or 'alliance' in action_lower:
                partner = self._extract_player_name(action, active_players, exclude=agent_name)
                if partner:
                    alliance_name = f"Alliance_{self.current_round}_{len(self.alliances)}"
                    self.alliances[alliance_name] = [agent_name, partner]
                    round_result['alliance_changes'].append({
                        'type': 'formed',
                        'members': [agent_name, partner]
                    })

            # Check for votes
            if 'vote' in action_lower:
                target = self._extract_player_name(action, active_players, exclude=agent_name)
                if target:
                    round_result['votes'][agent_name] = target

        # Simulate challenge (random winner among active players)
        if active_players:
            winner = random.choice(active_players)
            round_result['challenge_winner'] = winner
            self.challenge_winners.append(winner)
            # Winner is immune from elimination this round
            for role in self.config.agent_roles:
                if role.name == winner:
                    role.public_info['challenge_wins'] += 1

        # Process elimination votes
        if round_result['votes']:
            vote_counts = {}
            for voter, target in round_result['votes'].items():
                vote_counts[target] = vote_counts.get(target, 0) + 1

            if vote_counts:
                # Remove challenge winner from elimination
                immune = round_result['challenge_winner']
                eligible_counts = {k: v for k, v in vote_counts.items() if k != immune}

                if eligible_counts:
                    max_votes = max(eligible_counts.values())
                    candidates = [p for p, v in eligible_counts.items() if v == max_votes]
                    eliminated = random.choice(candidates)  # Tie-breaker

                    self.eliminated.append(eliminated)
                    round_result['eliminated'] = eliminated

                    # Check for betrayals
                    for alliance_name, members in self.alliances.items():
                        if eliminated in members:
                            for voter, target in round_result['votes'].items():
                                if target == eliminated and voter in members:
                                    self.betrayals[voter] += 1
                                    round_result['alliance_changes'].append({
                                        'type': 'betrayal',
                                        'betrayer': voter,
                                        'victim': eliminated
                                    })

        # Update state
        self.state['round'] = self.current_round
        self.state['players_remaining'] = [p for p in self.state['players_remaining'] if p not in self.eliminated]
        self.state['alliances'] = self.alliances
        self.state['challenge_winner'] = round_result['challenge_winner']
        self.state['votes_this_round'] = round_result['votes']
        self.state['eliminated_this_round'] = round_result['eliminated']

        self.history.append(round_result)
        self.votes[self.current_round] = round_result['votes']

        return round_result

    def _extract_player_name(self, action: str, players: List[str], exclude: str = None) -> Optional[str]:
        """Extract a player name from action text."""
        for player in players:
            if player != exclude and player.lower() in action.lower():
                return player
        return None

    def calculate_payoffs(self) -> Dict[str, float]:
        """Calculate prize distribution."""
        survivors = self.state['players_remaining']
        payoffs = {}

        if not survivors:
            # Everyone eliminated (shouldn't happen normally)
            for role in self.config.agent_roles:
                payoffs[role.name] = 0.0
            return payoffs

        # Split prize among survivors
        prize_per_survivor = self.prize_pool / len(survivors)

        for role in self.config.agent_roles:
            if role.name in survivors:
                # Bonus for challenge wins
                wins = role.public_info.get('challenge_wins', 0)
                bonus = wins * 1000  # $1000 per challenge win

                # Penalty for betrayals (reputation damage in future games)
                betrayal_penalty = self.betrayals.get(role.name, 0) * 500

                payoffs[role.name] = prize_per_survivor + bonus - betrayal_penalty
            else:
                # Eliminated players get small consolation based on survival rounds
                elimination_round = next(
                    (i + 1 for i, h in enumerate(self.history) if h.get('eliminated') == role.name),
                    0
                )
                payoffs[role.name] = elimination_round * 500  # $500 per round survived

        return payoffs

    def is_complete(self) -> bool:
        """Scenario ends when 2 or fewer players remain or max rounds reached."""
        if len(self.state['players_remaining']) <= 2:
            return True
        if self.current_round >= self.config.max_rounds:
            return True
        return False

    def _generate_observation(self, agent_name: str) -> str:
        """Generate observation for a specific player."""
        if agent_name in self.eliminated:
            return f"You have been eliminated from the game in round {self._get_elimination_round(agent_name)}."

        survivors = self.state['players_remaining']
        obs = f"Round {self.current_round}. {len(survivors)} players remain: {', '.join(survivors)}. "

        if self.history:
            last = self.history[-1]
            if last.get('challenge_winner'):
                obs += f"{last['challenge_winner']} won immunity. "
            if last.get('eliminated'):
                obs += f"{last['eliminated']} was eliminated. "

            if last.get('votes'):
                vote_summary = {}
                for voter, target in last['votes'].items():
                    vote_summary[target] = vote_summary.get(target, 0) + 1
                obs += f"Vote distribution: {vote_summary}. "

        # Alliance information
        my_alliances = [
            (name, members) for name, members in self.alliances.items()
            if agent_name in members
        ]
        if my_alliances:
            allies = set()
            for _, members in my_alliances:
                allies.update(m for m in members if m != agent_name)
            obs += f"Your allies: {', '.join(allies)}. "

        remaining = self.config.max_rounds - self.current_round
        obs += f"{remaining} rounds remaining."

        return obs

    def _get_elimination_round(self, player: str) -> int:
        """Get the round a player was eliminated."""
        for i, h in enumerate(self.history):
            if h.get('eliminated') == player:
                return i + 1
        return 0


def create_scenario(scenario_type: str, **kwargs) -> BaseScenario:
    """Factory function to create scenarios."""
    scenarios = {
        'fishery': FisheryManagementScenario,
        'treaty': TreatyNegotiationScenario,
        'gameshow': RealityGameshowScenario
    }

    if scenario_type not in scenarios:
        raise ValueError(f"Unknown scenario type: {scenario_type}. Available: {list(scenarios.keys())}")

    return scenarios[scenario_type](**kwargs)

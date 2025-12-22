# Negotiation Framework for Concordia

This module extends Concordia with cognitive components that allow agents to engage in realistic negotiations. Rather than hard-coding negotiation behavior, these components give agents the building blocks they need to reason about offers, understand other parties, adapt their strategies, and work toward agreements.

---

## Who This Is For

### Researchers

If you study negotiations, bargaining, or strategic interaction, this framework lets you run experiments that would be difficult or expensive with human subjects. You can test hypotheses about how different cognitive capabilities affect negotiation outcomes, run hundreds of trials with controlled variations, and observe internal agent states that humans couldn't report accurately.

**Example research questions this enables:**
- How does theory of mind ability affect negotiation success?
- Do culturally-adapted agents reach better joint outcomes in cross-cultural settings?
- What happens when one party can model uncertainty and the other can't?
- How do trust dynamics evolve over repeated negotiations?

### Simulation Builders

If you're building multi-agent simulations that involve any kind of bargaining, deal-making, or resource allocation, these components save you from reinventing negotiation logic. Drop them into your existing Concordia setup and your agents gain sophisticated negotiation capabilities.

**Scenarios where this helps:**
- Marketplace simulations with buyers and sellers
- Diplomatic scenarios between factions or nations
- Workplace simulations with salary negotiations or project allocation
- Any game-theoretic scenario involving strategic interaction

### AI Safety and Alignment Researchers

Negotiations involve competing interests, potential deception, trust-building, and cooperation problems. This framework provides a controlled environment to study how AI agents handle these dynamics. You can observe whether agents learn to cooperate, defect, deceive, or build genuine trust.

---

## Why Use This Instead of Alternatives

### Compared to Hard-Coded Negotiation Logic

Many simulations implement negotiation as simple rules: "if offer > threshold, accept." This framework uses LLM-based reasoning with cognitive components, producing more realistic and varied behavior. Agents don't just follow scripts—they reason about the situation, consider the other party's perspective, and adapt their approach.

**Choose hard-coded logic when:** You need deterministic, predictable behavior for formal analysis or you're resource-constrained.

**Choose this framework when:** You want emergent, realistic behavior and are studying how cognition affects outcomes.

### Compared to Raw LLM Prompting

You could just prompt an LLM with "you are a negotiator, negotiate." But without structure, the LLM lacks persistent memory of the negotiation, can't track trust over time, doesn't have explicit cultural models, and can't systematically adapt its strategy. This framework provides that structure while still leveraging LLM flexibility.

**Choose raw prompting when:** You're prototyping quickly or need maximum flexibility.

**Choose this framework when:** You need consistent, measurable behavior with observable internal states.

### Compared to Classical Game-Theoretic Agents

Traditional game theory agents compute optimal strategies based on payoff matrices. They're mathematically elegant but miss the messiness of real negotiation: misunderstandings, relationship dynamics, cultural friction, learning from experience. This framework captures that messiness.

**Choose game-theoretic agents when:** You're studying equilibria or need provable properties.

**Choose this framework when:** You're studying realistic behavior or training agents for human interaction.

### Compared to Reinforcement Learning Approaches

RL agents learn negotiation through trial and error with reward signals. This works but requires many training episodes and the learned behavior can be opaque. This framework provides interpretable cognitive components—you can see exactly why an agent made a decision by examining its theory of mind state or cultural adaptation logic.

**Choose RL when:** You have clear reward signals and want learned emergent strategies.

**Choose this framework when:** You need interpretability or can't run thousands of training episodes.

---

## When Not to Use This

Be honest about the limitations:

- **High-stakes decisions**: This is a research tool, not production software for actual negotiations
- **Formal verification needs**: If you need provable properties, use game-theoretic approaches
- **Resource constraints**: LLM calls are slow and expensive compared to simple heuristics
- **Determinism requirements**: LLM-based agents have inherent variability

---

## What This Framework Does

Negotiations in the real world involve much more than just exchanging offers. People read social cues, remember past interactions, adjust their approach based on cultural context, and change tactics when something isn't working. This framework captures those dynamics through modular cognitive components that can be mixed and matched depending on what your simulation needs.

Each component handles a specific aspect of negotiation cognition:

- **Theory of Mind** lets agents model what other parties might be thinking or feeling
- **Cultural Adaptation** adjusts communication style based on cultural context
- **Temporal Strategy** tracks relationships and manages time pressure
- **Uncertainty Awareness** helps agents reason about incomplete information
- **Swarm Intelligence** enables coordination in group negotiations
- **Strategy Evolution** allows agents to learn and adapt their approach over time

You can use all of these together for highly sophisticated agents, or pick just the ones relevant to your research question.

---

## Quick Start

### Basic Negotiation Agent

The simplest way to get started is with a base negotiator that handles the fundamentals:

```python
from concordia.prefabs.entity.negotiation import base_negotiator

agent = base_negotiator.build_agent(
    model=language_model,
    memory_bank=memory_bank,
    name='Alice',
    goal='Negotiate a fair price for the used car',
    reservation_value=8000.0,  # Won't accept less than this
)
```

### Agent with Cognitive Modules

For more sophisticated behavior, use the advanced negotiator with specific modules enabled:

```python
from concordia.prefabs.entity.negotiation import advanced_negotiator

agent = advanced_negotiator.build_agent(
    model=language_model,
    memory_bank=memory_bank,
    name='Bob',
    goal='Reach a mutually beneficial agreement',
    modules=['theory_of_mind', 'cultural_adaptation', 'temporal_strategy'],
)
```

### Specialized Agent Builders

The framework includes convenience functions for common scenarios:

```python
# Agent that models opponent mental states
agent = advanced_negotiator.build_theory_of_mind_agent(
    model=model, memory_bank=memory, name='Analyst'
)

# Agent sensitive to cultural negotiation norms
agent = advanced_negotiator.build_cultural_agent(
    model=model, memory_bank=memory, name='Diplomat',
    own_culture='east_asian'
)

# Agent that learns from outcomes over time
agent = advanced_negotiator.build_adaptive_agent(
    model=model, memory_bank=memory, name='Learner',
    learning_rate=0.1
)
```

---

## Cognitive Components

### Theory of Mind (`theory_of_mind.py`)

This component enables agents to reason about what other negotiators might be thinking, feeling, or planning. It maintains mental models of each party and updates them based on observed behavior.

**What it tracks:**
- Inferred emotional states (frustrated, confident, anxious)
- Estimated intentions and goals
- Predicted next moves
- Belief updates based on new information

**When to use it:** Research on perspective-taking in negotiations, studying how mental modeling affects outcomes, or any scenario where understanding the other party matters.

```python
# The component exposes its inferences for analysis
state = theory_of_mind_component.get_state()
# Returns dict with mental models, emotion estimates, predictions
```

### Cultural Adaptation (`cultural_adaptation.py`)

Different cultures have different norms around negotiation. Some favor direct communication, others prefer indirect approaches. Some prioritize relationship-building before business, others get straight to the point. This component helps agents navigate these differences.

**Built-in cultural profiles:**
- `western_business` - Direct, efficiency-focused, comfortable with confrontation
- `east_asian` - Indirect, relationship-first, face-saving important
- `middle_eastern` - Hospitality-oriented, trust-building, patience valued
- `nordic` - Egalitarian, consensus-seeking, understated
- `latin_american` - Warm, flexible with time, personal connections matter

**What it does:**
- Adjusts communication style based on cultural context
- Detects potential cultural friction
- Suggests culturally appropriate responses
- Tracks cross-cultural relationship dynamics

```python
agent = advanced_negotiator.build_cultural_agent(
    model=model,
    memory_bank=memory,
    name='GlobalNegotiator',
    own_culture='western_business',
    counterpart_cultures=['east_asian', 'middle_eastern'],
)
```

### Temporal Strategy (`temporal_strategy.py`)

Negotiations happen over time. Deadlines create pressure, relationships evolve, and the history of interactions shapes current behavior. This component manages the temporal dimension.

**What it handles:**
- Relationship tracking across interactions
- Trust building and decay
- Deadline awareness and time pressure effects
- Historical pattern recognition
- Commitment tracking (who promised what, and did they follow through)

**Trust mechanics:**
- Trust increases slowly with positive interactions (+0.05 per kept commitment)
- Trust decreases quickly when commitments are broken (-0.10 per violation)
- This asymmetry reflects how trust works in real relationships

```python
agent = advanced_negotiator.build_temporal_agent(
    model=model,
    memory_bank=memory,
    name='LongTermPartner',
    discount_factor=0.9,  # Values future outcomes highly
)
```

### Uncertainty Awareness (`uncertainty_aware.py`)

Real negotiations involve incomplete information. You don't know the other party's true reservation price, their alternatives, or their priorities. This component helps agents reason under uncertainty.

**What it provides:**
- Confidence tracking for beliefs and estimates
- Information value assessment (what's worth learning)
- Risk-adjusted decision making
- Bayesian-style belief updates as new information arrives

**When to use it:** Studying information asymmetry, research on risk preferences, or scenarios where uncertainty plays a key role.

### Swarm Intelligence (`swarm_intelligence.py`)

In multi-party negotiations, agents may need to coordinate with allies, form coalitions, or engage in collective decision-making. This component supports those dynamics.

**Capabilities:**
- Coalition formation and management
- Collective position development
- Information sharing within groups
- Consensus building mechanisms
- Defection detection

**When to use it:** Multi-party negotiations, coalition bargaining games, or any scenario involving group dynamics.

### Strategy Evolution (`strategy_evolution.py`)

Skilled negotiators adapt their approach based on what's working. This component gives agents the ability to learn from outcomes and evolve their strategies over time.

**How it works:**
- Tracks strategy effectiveness across negotiations
- Maintains a population of strategy variants
- Successful strategies get reinforced
- Unsuccessful approaches get modified or abandoned
- Can introduce novel tactics through controlled exploration

**Configuration:**
```python
agent = advanced_negotiator.build_adaptive_agent(
    model=model,
    memory_bank=memory,
    name='EvolvingAgent',
    learning_rate=0.1,      # How quickly to update strategies
    exploration_rate=0.05,  # How often to try new approaches
)
```

---

## Agent Modules vs Game Master Modules

The negotiation framework has two distinct sets of components that serve different purposes. Understanding when to use each—and when to use both together—is important for designing your simulation correctly.

### The Core Difference

Think of it like a movie production:

**Agent modules** are like an actor's internal process. They shape how each character thinks, feels, and decides what to do. These are first-person cognitive processes that happen inside the agent's "mind."

**Game Master modules** are like the director and production crew. They observe what's happening, enforce the rules of the world, control what information flows where, and describe the consequences of actions. These are third-person processes that manage the environment.

| Aspect | Agent Modules | GM Modules |
|--------|---------------|------------|
| Perspective | First-person (inside the agent) | Third-person (observing all agents) |
| Purpose | How agents think and decide | How the world works and responds |
| Access | Only own thoughts and observations | All agents' actions and world state |
| Examples | "What is she thinking?" | "Did he violate a cultural norm?" |
| Output | Influences agent's next action | Shapes observations agents receive |

### When to Use Agent Modules Only

Use just the agent-side components when:

- You want negotiation-capable agents in a standard Concordia environment
- The existing GM (generic, dialogic, etc.) handles your scenario fine
- You're studying how agent cognition affects outcomes, not environment effects
- You want the simplest possible setup

```python
from concordia.prefabs.entity.negotiation import advanced_negotiator
from concordia.prefabs.game_master import generic

# Negotiation-aware agents
buyer = advanced_negotiator.build_agent(
    model=model, memory_bank=memory, name='Buyer',
    modules=['theory_of_mind', 'temporal_strategy']
)
seller = advanced_negotiator.build_agent(
    model=model, memory_bank=memory, name='Seller',
    modules=['theory_of_mind', 'temporal_strategy']
)

# Standard Concordia game master (not negotiation-specific)
gm = generic.GenericGameMaster(
    params={'name': 'Mediator'},
    entities=[buyer, seller]
).build(model, memory)
```

The agents will still use their cognitive modules to reason about the negotiation. They just won't get negotiation-specific environmental feedback.

### When to Use GM Modules Only

Use just the GM-side components when:

- You have existing agents (not using this framework) that you want to put in a negotiation environment
- You're studying how environmental factors affect negotiation outcomes
- You need a negotiation-aware referee but agents are simple or scripted

```python
from concordia.prefabs.entity import basic
from concordia.prefabs.game_master.negotiation import negotiation

# Simple agents without negotiation modules
agent1 = basic.BasicAgent(params={'name': 'Party A'}).build(model, memory)
agent2 = basic.BasicAgent(params={'name': 'Party B'}).build(model, memory)

# Negotiation-aware game master
gm = negotiation.build_game_master(
    model=model, memory_bank=memory,
    entities=[agent1, agent2],
    gm_modules=['social_intelligence', 'temporal_dynamics']
)
```

The GM will track negotiation state, enforce deadlines, and provide negotiation-relevant observations, even though the agents themselves don't have sophisticated negotiation cognition.

### When to Use Both Together

Use both agent and GM modules when:

- You want the richest possible simulation
- Agent cognition and environmental feedback should interact
- You're studying complex dynamics like cultural misunderstandings or trust violations
- The GM should respond to what agents are thinking/doing

```python
from concordia.prefabs.entity.negotiation import advanced_negotiator
from concordia.prefabs.game_master.negotiation import negotiation

# Agents with cultural awareness
western_rep = advanced_negotiator.build_cultural_agent(
    model=model, memory_bank=memory, name='WesternRep',
    own_culture='western_business'
)
eastern_rep = advanced_negotiator.build_cultural_agent(
    model=model, memory_bank=memory, name='EasternRep',
    own_culture='east_asian'
)

# GM that also understands cultural dynamics
gm = negotiation.build_cultural_negotiation(
    model=model, memory_bank=memory,
    entities=[western_rep, eastern_rep],
)
```

In this setup:
- The agents internally adjust their communication style (agent module)
- The GM detects when a cultural norm is violated (GM module)
- The GM can intervene or describe consequences of cultural friction
- Both layers work together to create realistic cross-cultural negotiation

### Module Pairing Recommendations

Some combinations work particularly well together:

| Agent Module | Pairs With GM Module | Why |
|--------------|---------------------|-----|
| `theory_of_mind` | `gm_social_intelligence` | Agent models others; GM tracks actual social dynamics |
| `cultural_adaptation` | `gm_cultural_awareness` | Agent adapts style; GM enforces norms |
| `temporal_strategy` | `gm_temporal_dynamics` | Agent manages time; GM enforces deadlines |
| `uncertainty_aware` | `gm_uncertainty_management` | Agent reasons about unknowns; GM controls information |
| `swarm_intelligence` | `gm_collective_intelligence` | Agent coordinates with group; GM monitors coalitions |
| `strategy_evolution` | `gm_strategy_evolution` | Agent adapts strategy; GM observes patterns |

### Auto-Detection

When using both, the GM can automatically detect which agent modules are active and enable corresponding GM modules:

```python
gm = negotiation.build_game_master(
    model=model, memory_bank=memory,
    entities=[agent1, agent2],
    auto_detect_modules=True,  # GM inspects agents and enables matching modules
)
```

---

## Game Master Components Reference

Located in `concordia/prefabs/game_master/negotiation/`:

| GM Component | Purpose |
|--------------|---------|
| `gm_social_intelligence.py` | Tracks social dynamics, detects deception, monitors emotions |
| `gm_cultural_awareness.py` | Enforces cultural norms, detects violations, mediates misunderstandings |
| `gm_temporal_dynamics.py` | Manages deadlines, tracks commitments, handles time pressure |
| `gm_uncertainty_management.py` | Controls information flow, manages what each party knows |
| `gm_collective_intelligence.py` | Monitors coalitions, tracks group dynamics |
| `gm_strategy_evolution.py` | Observes strategy changes, tracks adaptation patterns |
| `negotiation_state.py` | Tracks offers, counteroffers, agreements, negotiation phases |
| `negotiation_validation.py` | Validates offers against constraints and rules |

---

## Running Experiments

### Ablation Studies

A key use case is comparing agent performance with different modules enabled:

```python
from concordia.prefabs.entity.negotiation import advanced_negotiator

# Full agent with all modules
full_agent = advanced_negotiator.build_agent(
    model=model, memory_bank=memory, name='FullAgent',
    modules=['theory_of_mind', 'cultural_adaptation', 'temporal_strategy',
             'uncertainty_aware', 'swarm_intelligence', 'strategy_evolution']
)

# Baseline with no advanced modules
baseline = advanced_negotiator.build_agent(
    model=model, memory_bank=memory, name='Baseline',
    modules=[]
)

# Single module to isolate its effect
tom_only = advanced_negotiator.build_agent(
    model=model, memory_bank=memory, name='ToMOnly',
    modules=['theory_of_mind']
)
```

### Accessing Internal State

All components expose their internal state for analysis:

```python
# Get current state of any component
state = component.get_state()

# Components also support state serialization
state_str = component.get_state()  # Returns JSON-compatible string
component.set_state(state_str)     # Restore from saved state
```

### Metrics Collection

The framework integrates with Concordia's measurement system. Key metrics tracked:

- Agreement rates
- Negotiation duration (rounds to agreement)
- Social welfare (joint outcomes)
- Individual utility scores
- Strategy distribution over time
- Trust dynamics

---

## Architecture Notes

### Component Lifecycle

Each component implements the standard Concordia component interface:

- `pre_act(action_spec)` - Called before the agent acts, returns context for decision-making
- `post_act(action_attempt)` - Called after action, updates internal state
- `pre_observe(observation)` - Processes incoming observations
- `post_observe()` - Post-observation updates
- `get_state()` / `set_state()` - Serialization support

### Module Independence

Components are designed to work independently or together. There are no hard dependencies between modules. This means you can:

- Use any single module in isolation
- Combine modules freely
- Add or remove modules without breaking others

### LLM Integration

Components use the language model for complex reasoning tasks (emotion detection, cultural assessment, strategy analysis). Simple computations (trust updates, time calculations) are done deterministically for consistency and efficiency.

---

## File Reference

```
concordia/prefabs/entity/negotiation/
├── __init__.py
├── base_negotiator.py          # Basic negotiation agent builder
├── advanced_negotiator.py      # Advanced agent with module support
├── integration_framework.py    # Utilities for component integration
└── components/
    ├── __init__.py
    ├── theory_of_mind.py       # Mental modeling of other parties
    ├── cultural_adaptation.py  # Cultural context handling
    ├── temporal_strategy.py    # Time and relationship management
    ├── uncertainty_aware.py    # Reasoning under uncertainty
    ├── swarm_intelligence.py   # Multi-party coordination
    ├── strategy_evolution.py   # Learning and adaptation
    ├── negotiation_instructions.py  # Base negotiation guidance
    ├── negotiation_memory.py   # Negotiation-specific memory
    └── negotiation_strategy.py # Core strategy component
```

---

## Contributing

This framework was developed as part of Google Summer of Code. Contributions are welcome, particularly:

- Additional cultural profiles based on negotiation research
- New cognitive components for aspects not yet covered
- Improvements to existing component implementations
- Additional test scenarios and benchmarks

See the main Concordia CONTRIBUTING.md for guidelines.

---

## Further Reading

For background on the concepts implemented here:

- Theory of Mind in negotiation: Bazerman & Neale, "Negotiating Rationally"
- Cultural negotiation styles: Meyer, "The Culture Map"
- Trust dynamics: Lewicki & Bunker, "Trust in Relationships"
- Strategy adaptation: Axelrod, "The Evolution of Cooperation"

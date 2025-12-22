# Evaluation Framework for Concordia Negotiation Modules

This framework provides tools for rigorously evaluating negotiation agents in controlled experimental settings. It aligns with the Concordia Contest on Cooperative Intelligence, offering standardized scenarios, metrics collection, statistical analysis, and ablation study support.

---

## Who This Is For

### Researchers Running Experiments

If you're studying negotiation behavior and need to compare different agent configurations, this framework handles the experimental infrastructure. You define what to compare; it runs the trials, collects the metrics, and provides statistical analysis ready for publication.

**What you get:**
- Standardized contest scenarios with clear cooperation challenges
- Metrics aligned with the Concordia Contest criteria (individual returns, social welfare, cooperation skills)
- Effect size calculations and significance tests
- Ablation study automation (systematically remove modules to measure their contribution)

### Developers Validating Their Agents

If you've built or modified negotiation agents and want to verify they work as intended, this framework provides quick sanity checks and deeper validation. Run your agents through the scenarios and see how they perform against baselines.

**What you get:**
- Baseline agents for comparison (random, fixed strategies, raw LLM)
- Quick test mode for fast iteration
- Detailed per-trial breakdowns to diagnose issues

### Students Learning About Evaluation

If you're new to multi-agent evaluation or experimental design, this framework demonstrates good practices: reproducible experiments, proper statistical comparisons, effect size reporting, and power analysis.

---

## Why Use This Framework

### Compared to Ad-Hoc Testing

You could manually run a few negotiations and eyeball the results. But that approach doesn't scale, isn't reproducible, and can't tell you whether differences are statistically meaningful. This framework automates the tedious parts while ensuring methodological rigor.

**Choose ad-hoc testing when:** You're debugging a specific issue or doing initial exploration.

**Choose this framework when:** You need reproducible results, statistical confidence, or are preparing findings for others.

### Compared to Custom Evaluation Scripts

Building your own evaluation harness is tempting, but it's easy to miss important considerations: proper randomization, effect size calculations, power analysis, handling edge cases in scenarios. This framework has already solved those problems.

**Choose custom scripts when:** Your evaluation needs are very different from negotiation contests.

**Choose this framework when:** You're evaluating negotiation agents and want a proven setup.

### Compared to Just Using LLM Judgments

You could ask an LLM to evaluate negotiation quality. But LLM judgments are inconsistent across runs, hard to validate, and don't provide the quantitative metrics needed for scientific claims. This framework uses objective outcome measures.

**Choose LLM judgments when:** You need qualitative insights about negotiation style.

**Choose this framework when:** You need quantitative, reproducible metrics.

---

## When Not to Use This

Be clear about what this framework isn't designed for:

- **Real-time production monitoring**: This is for offline evaluation, not live systems
- **Human-subject experiments**: The scenarios are designed for AI agents, not human participants
- **Non-negotiation domains**: The scenarios and metrics assume negotiation contexts
- **Extremely fast iteration**: LLM-based agents are slow; each trial takes seconds

---

## What This Framework Provides

### Contest-Aligned Scenarios

Three cooperative dilemma scenarios matching the Concordia Contest design:

**Fishery Management** - Common pool resource dilemma where fishing companies must balance individual profit against collective sustainability. Tests reciprocity, promise-keeping, and long-term thinking.

**Treaty Negotiation** - Multi-party bargaining where nations negotiate a climate agreement with multiple provisions. Tests coalition building, package deals, and commitment credibility.

**Reality Gameshow** - Social dynamics scenario where contestants form alliances and vote to eliminate players. Tests coalition behavior, trust, and strategic voting.

Each scenario creates tension between individual and collective interests, the core challenge of cooperative intelligence.

### Comprehensive Metrics

The framework tracks three categories of metrics aligned with the contest:

**Individual Returns** - What each agent achieves for themselves: value captured, agreements reached, goal achievement.

**Social Welfare** - Collective outcomes: Pareto efficiency (how much total value was realized), fairness (how evenly value was distributed), joint gains.

**Cooperation Skills** - Behavioral measures: promise-keeping, reciprocity, reputation management, coalition behavior, information sharing, fairness sensitivity.

### Statistical Analysis

The framework doesn't just collect numbers; it provides proper statistical analysis:

- **Effect sizes** (Cohen's d) so you know if differences are meaningful, not just statistically significant
- **Confidence intervals** for all key metrics
- **Power analysis** to determine how many trials you need
- **ANOVA** for comparing multiple conditions
- **Bonferroni correction** for multiple comparisons

### Ablation Study Support

A key use case is understanding which modules contribute to agent performance. The framework automates ablation studies:

```
Full agent:     [ToM, Cultural, Temporal, Swarm, Uncertainty, Evolution]
No ToM:         [Cultural, Temporal, Swarm, Uncertainty, Evolution]
No Cultural:    [ToM, Temporal, Swarm, Uncertainty, Evolution]
...
Baseline:       []
```

Run all conditions, and the framework tells you which modules have the largest effect sizes.

---

## Quick Start

### Running Your First Evaluation

The fastest way to verify everything works:

```bash
python -m concordia.prefabs.entity.negotiation.evaluation.run_experiments --quick
```

This runs 5 trials on the fishery scenario comparing a full agent to a baseline. Takes about 10 seconds with mock models.

### Running a Proper Ablation Study

For publishable results, run more trials across all conditions:

```bash
python -m concordia.prefabs.entity.negotiation.evaluation.run_experiments --scenario fishery --trials 30 --ablation
```

This runs 8 conditions (full + 6 ablations + baseline) × 30 trials = 240 trials. The framework reports effect sizes and significance for each module.

### Running All Scenarios

To evaluate across all three contest scenarios:

```bash
python -m concordia.prefabs.entity.negotiation.evaluation.run_experiments --scenario all --trials 30 --ablation
```

### Paper-Ready Evaluation

For comprehensive results suitable for publication:

```bash
python -m concordia.prefabs.entity.negotiation.evaluation.run_experiments --full --output results/paper/
```

This runs 50 trials per condition across all scenarios with full statistical analysis.

---

## Using the Framework Programmatically

### Basic Experiment

```python
from concordia.prefabs.entity.negotiation.evaluation import (
    ExperimentRunner, ExperimentConfig, ALL_MODULES
)

runner = ExperimentRunner()

config = ExperimentConfig(
    name='my_experiment',
    scenario_type='fishery',
    modules_to_test=[['theory_of_mind', 'cultural_adaptation']],
    num_trials=30,
    random_seed=42,
)

results = runner.run_experiment(config, verbose=True)
print(f"Social welfare: {results.get_social_welfare()}")
```

### Running an Ablation Study

```python
from concordia.prefabs.entity.negotiation.evaluation import (
    ExperimentRunner, create_ablation_configs
)

runner = ExperimentRunner()
configs = create_ablation_configs('fishery', num_trials=30)

results = {}
for config in configs:
    results[config.name] = runner.run_experiment(config)

# Analyze which modules matter most
analysis = runner.analyze_ablation_results(results)
for module, data in analysis['module_importance'].items():
    print(f"{module}: d={data['effect_size']:.2f} ({data['interpretation']})")
```

### Using Custom Metrics

```python
from concordia.prefabs.entity.negotiation.evaluation import (
    MetricsCollector, NegotiationMetrics
)

collector = MetricsCollector("my_study")

# Start tracking a negotiation
metrics = collector.start_negotiation(
    scenario_name='custom',
    trial_id=0,
    agent_names=['Alice', 'Bob'],
    modules=['theory_of_mind'],
    max_rounds=20,
)

# Record events during negotiation
collector.record_action('Alice', 'offer', {'amount': 100})
collector.record_cooperation_observation('Alice', 'reciprocity', 0.8)
collector.increment_round()

# Finalize with outcomes
result = collector.finalize_negotiation(
    outcome='agreement',
    values={'Alice': 150, 'Bob': 120},
    max_values={'Alice': 200, 'Bob': 200},
)
```

### Statistical Comparison

```python
from concordia.prefabs.entity.negotiation.evaluation import (
    cohens_d, interpret_cohens_d, compare_conditions
)

# Compare two conditions
full_agent_scores = [0.8, 0.75, 0.82, 0.79, 0.81, ...]
baseline_scores = [0.6, 0.55, 0.62, 0.58, 0.61, ...]

d = cohens_d(full_agent_scores, baseline_scores)
print(f"Effect size: {d:.2f} ({interpret_cohens_d(d)})")

# Full statistical comparison
result = compare_conditions(full_agent_scores, baseline_scores,
                           "Full Agent", "Baseline")
print(f"p-value: {result.p_value:.4f}")
print(f"Significant: {result.significant}")
```

---

## Scenarios in Detail

### Fishery Management

```python
from concordia.prefabs.entity.negotiation.evaluation import FisheryManagementScenario

scenario = FisheryManagementScenario(
    num_agents=4,
    max_rounds=20,
    initial_fish_stock=1000.0,
    regeneration_rate=0.2,
)
```

**Dynamics:**
- Each round, companies choose how many boats to deploy (0-15)
- More boats catch more fish, but total fleet affects catch efficiency
- Fish stock regenerates each round (logistic growth)
- Overfishing causes stock collapse, ending the scenario early

**Cooperation Challenge:**
- Individually rational to deploy maximum boats
- Collectively optimal to limit fishing for sustainability
- Quota agreements can be proposed but violations are tracked

**Outcomes:**
- "sustainable" if fish stock remains above 50 at round 20
- "collapse" if fish stock drops below 50

### Treaty Negotiation

```python
from concordia.prefabs.entity.negotiation.evaluation import TreatyNegotiationScenario

scenario = TreatyNegotiationScenario(
    num_agents=5,  # Different nation types
    max_rounds=15,
)
```

**Dynamics:**
- Nations negotiate three treaty provisions: emissions targets, funding, timeline
- Each nation has private preferences (resistance to each provision)
- Proposals can be made, supported, or opposed
- Agreement requires consensus from all parties

**Nation Types:**
- IndustrialNation: economy-focused, resists emissions cuts
- DevelopingNation: growth-focused, wants funding
- IslandNation: climate-vulnerable, wants ambitious targets
- OilExporter: resists emissions, flexible on funding
- GreenLeader: climate-ambitious, moderate on funding

**Cooperation Challenge:**
- Each nation wants a treaty that minimizes their costs
- But no treaty (impasse) is worse for everyone, especially vulnerable nations
- Package deals and coalitions can help find mutual gains

### Reality Gameshow

```python
from concordia.prefabs.entity.negotiation.evaluation import RealityGameshowScenario

scenario = RealityGameshowScenario(
    num_agents=6,
    max_rounds=10,
    prize_pool=100000.0,
)
```

**Dynamics:**
- Contestants compete in challenges (simulated, winner gets immunity)
- Each round, contestants vote to eliminate one player
- Final prize is split among survivors
- Alliances can be formed for voting coordination

**Player Types:**
- StrategicPlayer, SocialPlayer, CompetitivePlayer, LoyalPlayer, UnpredictablePlayer, UndertheRadarPlayer

**Cooperation Challenge:**
- Alliances help protect members from elimination
- But alliances can be betrayed for larger prize share
- Reputation affects future alliance formation

---

## Baseline Agents

The framework includes baseline agents for comparison:

### RandomAgent
Makes random valid actions. Represents the floor of performance.

```python
from concordia.prefabs.entity.negotiation.evaluation import create_random_agent

agent = create_random_agent("RandomBaseline")
```

### FixedStrategyAgent
Follows predetermined heuristics. Tests whether learned behavior beats simple rules.

```python
from concordia.prefabs.entity.negotiation.evaluation import create_fixed_strategy_agent

aggressive = create_fixed_strategy_agent("Aggressive", strategy="aggressive")
cooperative = create_fixed_strategy_agent("Cooperative", strategy="cooperative")
tit_for_tat = create_fixed_strategy_agent("TitForTat", strategy="tit_for_tat")
```

### BasicLLMAgent
Raw LLM with minimal prompting. Tests whether cognitive modules add value over basic LLM reasoning.

```python
from concordia.prefabs.entity.negotiation.evaluation import create_basic_llm_agent

agent = create_basic_llm_agent("BasicLLM", model=my_model)
```

### SingleModuleAgent
Agent with only one cognitive module enabled. Used to measure individual module contributions.

```python
from concordia.prefabs.entity.negotiation.evaluation import create_single_module_agent

tom_only = create_single_module_agent("ToMOnly", module_name="theory_of_mind")
```

---

## Metrics Reference

### Individual Returns
| Metric | Description |
|--------|-------------|
| `value_obtained` | Absolute value the agent achieved |
| `value_capture_ratio` | Ratio of value obtained to maximum possible |
| `agreement_rate` | Proportion of negotiations ending in agreement |

### Social Welfare
| Metric | Description |
|--------|-------------|
| `pareto_efficiency` | Total value created / maximum possible total |
| `fairness_gini` | Gini coefficient of value distribution (0 = perfect equality) |
| `social_welfare_score` | Combined metric: 0.5 × pareto + 0.5 × (1 - gini) |

### Cooperation Skills
| Skill | What It Measures |
|-------|-----------------|
| `promise_keeping` | Did the agent honor its commitments? |
| `reciprocity` | Did the agent match cooperation with cooperation? |
| `reputation_management` | Did the agent build and maintain credibility? |
| `coalition_behavior` | Did the agent coordinate effectively with allies? |
| `information_sharing` | Did the agent share information appropriately? |
| `fairness_sensitivity` | Did the agent respond to fairness violations? |

---

## Statistical Analysis Reference

### Effect Size Interpretation (Cohen's d)
| Range | Interpretation |
|-------|---------------|
| \|d\| < 0.2 | Negligible |
| 0.2 ≤ \|d\| < 0.5 | Small |
| 0.5 ≤ \|d\| < 0.8 | Medium |
| \|d\| ≥ 0.8 | Large |

### Required Sample Sizes (80% power)
| Target Effect | Trials Needed |
|---------------|---------------|
| Small (d=0.2) | 392 per group |
| Small-medium (d=0.35) | 128 per group |
| Medium (d=0.5) | 63 per group |
| Medium-large (d=0.65) | 38 per group |
| Large (d=0.8) | 25 per group |

### Using the ResultsAnalyzer

```python
from concordia.prefabs.entity.negotiation.evaluation import ResultsAnalyzer

analyzer = ResultsAnalyzer(experiments)
analysis = analyzer.full_analysis()

print(analysis['summary_statistics'])
print(analysis['effect_sizes'])
print(analysis['recommendations'])  # Suggests if more trials are needed

# Generate LaTeX table for paper
latex = analyzer.generate_latex_table()
```

---

## CLI Reference

```bash
python -m concordia.prefabs.entity.negotiation.evaluation.run_experiments [OPTIONS]
```

### Scenario Selection
| Option | Description |
|--------|-------------|
| `--scenario fishery` | Run fishery management only |
| `--scenario treaty` | Run treaty negotiation only |
| `--scenario gameshow` | Run reality gameshow only |
| `--scenario all` | Run all scenarios (default) |

### Trial Configuration
| Option | Description |
|--------|-------------|
| `--trials N` | Number of trials per condition (default: 30) |
| `--seed N` | Random seed for reproducibility (default: 42) |

### Study Type
| Option | Description |
|--------|-------------|
| `--ablation` | Run full ablation study |
| `--baseline-only` | Only compare full agent vs baseline |
| `--module-isolation` | Test each module in isolation |

### Presets
| Option | Description |
|--------|-------------|
| `--quick` | 5 trials, single scenario (sanity check) |
| `--full` | 50 trials, all scenarios, all conditions |

### Output
| Option | Description |
|--------|-------------|
| `--output PATH` | Directory for results (default: evaluation/results) |
| `--format json\|text\|both` | Output format (default: both) |
| `--verbose` | Print detailed progress |
| `--dry-run` | Show plan without running |

---

## Output Files

After running experiments, results are saved to the output directory:

```
evaluation/results/
├── results_20251222_143000.json    # Machine-readable results
└── report_20251222_143000.txt      # Human-readable report
```

### JSON Structure

```json
{
  "experiment_name": "Concordia_Contest_Evaluation",
  "timestamp": "2025-12-22T14:30:00",
  "conditions": {
    "fishery_full": {
      "scenario_name": "fishery",
      "modules_tested": ["theory_of_mind", "cultural_adaptation", ...],
      "n_trials": 30,
      "metrics": {
        "agreement_rate": {"mean": 0.75, "std": 0.12},
        "pareto_efficiency": {"mean": 0.68, "std": 0.08},
        "social_welfare": {"mean": 0.72, "std": 0.10}
      }
    }
  }
}
```

### Report Structure

```
======================================================================
CONCORDIA CONTEST EVALUATION REPORT
Generated: 2025-12-22 14:30:00
======================================================================

EXECUTIVE SUMMARY
----------------------------------------
Baseline Social Welfare: 0.65
Full Agent Social Welfare: 0.72
Improvement: 10.8%

MODULE IMPORTANCE RANKING
----------------------------------------
1. theory_of_mind: Effect size = 0.85 (large)
2. swarm_intelligence: Effect size = 0.72 (medium)
...

DETAILED ABLATION RESULTS
----------------------------------------
[Full statistical breakdown for each condition]
```

---

## Running with Real LLMs

**Important:** Mock models are only for testing that code runs. For meaningful evaluation results, you must use a real LLM.

### Option 1: OpenAI GPT (Recommended for quality)

```python
import os
os.environ['OPENAI_API_KEY'] = 'sk-your-key-here'

from concordia.prefabs.entity.negotiation.evaluation import (
    RealAgentRunner, create_openai_model
)

model = create_openai_model(model_name='gpt-4')  # or 'gpt-3.5-turbo' for cheaper
runner = RealAgentRunner(model=model)

results = runner.run_ablation_study(
    scenario_type='fishery',
    num_trials=10,
    verbose=True
)

analysis = runner.analyze_results(results)
runner.print_report(results, analysis)
```

### Option 2: Google Gemini

```python
import os
os.environ['GOOGLE_API_KEY'] = 'your-key-here'

from concordia.prefabs.entity.negotiation.evaluation import (
    RealAgentRunner, create_google_model
)

model = create_google_model(model_name='gemini-pro')
runner = RealAgentRunner(model=model)

results = runner.run_ablation_study('fishery', num_trials=10, verbose=True)
```

### Option 3: Ollama (Free, Local, No API Key)

First install Ollama and pull a model:

```bash
# macOS
brew install ollama

# Or download from https://ollama.ai

# Pull a model
ollama pull llama2

# Start the server
ollama serve
```

Then run evaluation:

```python
from concordia.prefabs.entity.negotiation.evaluation import (
    RealAgentRunner, create_ollama_model
)

model = create_ollama_model(model_name='llama2')
runner = RealAgentRunner(model=model)

results = runner.run_ablation_study('fishery', num_trials=10, verbose=True)
```

### Option 4: Together AI

```python
import os
os.environ['TOGETHER_API_KEY'] = 'your-key-here'

from concordia.prefabs.entity.negotiation.evaluation import (
    RealAgentRunner, create_together_model
)

model = create_together_model(model_name='meta-llama/Llama-2-70b-chat-hf')
runner = RealAgentRunner(model=model)

results = runner.run_ablation_study('fishery', num_trials=10, verbose=True)
```

### Cost Estimates

| LLM | Cost per Trial | 50 Trials (one scenario) |
|-----|----------------|--------------------------|
| GPT-4 | ~$0.50 | ~$25 |
| GPT-3.5-turbo | ~$0.05 | ~$2.50 |
| Gemini Pro | ~$0.02 | ~$1 |
| Together AI | ~$0.10 | ~$5 |
| Ollama (local) | Free | Free |

### Complete Evaluation Script

Save as `run_real_eval.py`:

```python
#!/usr/bin/env python3
"""Run real evaluation with actual LLM."""
import os
from concordia.prefabs.entity.negotiation.evaluation import (
    RealAgentRunner, create_openai_model
)

# Set your API key (or use environment variable)
os.environ['OPENAI_API_KEY'] = 'sk-your-key-here'

# Create model and runner
model = create_openai_model('gpt-4')
runner = RealAgentRunner(model=model)

# Run ablation study
print("Running real evaluation...")
results = runner.run_ablation_study(
    scenario_type='fishery',
    num_trials=10,
    verbose=True
)

# Analyze and report
analysis = runner.analyze_results(results)
runner.print_report(results, analysis)

# Save results
import json
from datetime import datetime
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'results_{timestamp}.json', 'w') as f:
    json.dump({
        name: {
            'welfare': exp.get_social_welfare()[0],
            'agreement_rate': exp.get_agreement_rate()[0],
            'n_trials': exp.n_trials
        }
        for name, exp in results.items()
    }, f, indent=2)
print(f"Results saved to results_{timestamp}.json")
```

Run with:

```bash
python run_real_eval.py
```

---

## File Reference

```
concordia/prefabs/entity/negotiation/evaluation/
├── __init__.py                 # Package exports
├── README.md                   # This documentation
├── run_experiments.py          # CLI entry point (basic harness)
├── evaluation_harness.py       # Basic experiment runner
├── real_agent_evaluation.py    # Real agent evaluation (recommended)
├── metrics.py                  # Metrics collection and aggregation
├── contest_scenarios.py        # Fishery, Treaty, Gameshow scenarios
├── baseline_agents.py          # Baseline agents for comparison
├── statistical_analysis.py     # Statistical tests and effect sizes
└── results/                    # Output directory for results
    ├── results_*.json
    └── report_*.txt
```

### Two Evaluation Approaches

| File | What It Does | When to Use |
|------|--------------|-------------|
| `real_agent_evaluation.py` | Uses actual negotiation framework agents with cognitive modules | **For real evaluation** |
| `evaluation_harness.py` | Simulates modules via prompt text | Quick prototyping only |

---

## Integration with Negotiation Framework

This evaluation framework is designed to test agents built with the negotiation framework in `concordia/prefabs/entity/negotiation/`. The module names align:

| Negotiation Module | Evaluation Tests |
|-------------------|-----------------|
| `theory_of_mind` | Effect on social welfare, opponent modeling accuracy |
| `cultural_adaptation` | Cross-cultural scenario performance |
| `temporal_strategy` | Long-term relationship outcomes, deadline management |
| `swarm_intelligence` | Coalition dynamics in multi-party scenarios |
| `uncertainty_aware` | Performance under information asymmetry |
| `strategy_evolution` | Improvement over repeated trials |

---

## Common Workflows

### Testing a New Module

1. Add your module to the negotiation framework
2. Run ablation comparing with and without your module:
```bash
python -m concordia.prefabs.entity.negotiation.evaluation.run_experiments --trials 50 --ablation
```
3. Check if your module shows a meaningful effect size

### Comparing Two Agent Designs

1. Run each design through the same scenarios:
```python
results_a = runner.run_experiment(config_a)
results_b = runner.run_experiment(config_b)
```
2. Use statistical comparison:
```python
comparison = compare_conditions(
    [t.social_welfare_score for t in results_a.trials],
    [t.social_welfare_score for t in results_b.trials]
)
```

### Preparing Results for Publication

1. Run comprehensive evaluation:
```bash
python -m concordia.prefabs.entity.negotiation.evaluation.run_experiments --full --output results/paper/
```
2. Generate LaTeX tables:
```python
from concordia.prefabs.entity.negotiation.evaluation import ResultsAnalyzer
analyzer = ResultsAnalyzer(results)
print(analyzer.generate_latex_table())
```
3. Report effect sizes and confidence intervals, not just p-values

---

## Contributing

This framework was developed as part of Google Summer of Code. Contributions are welcome, particularly:

- Additional scenarios aligned with cooperative intelligence research
- More sophisticated baseline agents
- Improvements to statistical analysis (e.g., bootstrap confidence intervals)
- Visualization tools for results

See the main Concordia CONTRIBUTING.md for guidelines.

---

## Further Reading

For background on evaluation methodology:

- Effect sizes: Cohen, "Statistical Power Analysis for the Behavioral Sciences"
- Cooperative intelligence: The Concordia Contest problem statement
- Multi-agent evaluation: Leibo et al., "Multi-agent Reinforcement Learning in Sequential Social Dilemmas"
- Negotiation metrics: Baarslag et al., "Evaluating Practical Negotiating Agents"

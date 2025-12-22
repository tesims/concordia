# Evaluation Framework for Concordia Negotiation Modules
# Aligned with the Concordia Contest on Cooperative Intelligence

"""
This package provides tools for evaluating negotiation modules in the context
of the Concordia Contest: Advancing the Cooperative Intelligence of Language
Model Agents.

Key modules:
- contest_scenarios: Three cooperative dilemma scenarios matching contest design
- real_agent_evaluation: Main evaluation using actual negotiation agents
- metrics: Metrics collection aligned with contest evaluation criteria
- baseline_agents: Baseline agent implementations for comparison
- statistical_analysis: Statistical testing and paper-ready analysis

Usage:
    from concordia.prefabs.entity.negotiation.evaluation import (
        RealAgentRunner,
        create_openai_model,
    )

    model = create_openai_model('gpt-4')
    runner = RealAgentRunner(model=model)
    results = runner.run_ablation_study('fishery', num_trials=10)
"""

# Metrics
from .metrics import (
    MetricsCollector,
    CooperationMetrics,
    AgentMetrics,
    NegotiationMetrics,
    ExperimentMetrics,
    calculate_effect_size,
    interpret_effect_size,
)

# Scenarios
from .contest_scenarios import (
    FisheryManagementScenario,
    TreatyNegotiationScenario,
    RealityGameshowScenario,
    create_scenario,
)

# Evaluation harness
from .evaluation_harness import (
    ExperimentRunner,
    ExperimentConfig,
    ALL_MODULES,
    create_ablation_configs,
)

# Baseline agents
from .baseline_agents import (
    RandomAgent,
    FixedStrategyAgent,
    BasicLLMAgent,
    SingleModuleAgent,
    create_random_agent,
    create_fixed_strategy_agent,
    create_basic_llm_agent,
    create_single_module_agent,
    create_all_baselines,
)

# Statistical analysis
from .statistical_analysis import (
    cohens_d,
    interpret_cohens_d,
    welchs_t_test,
    confidence_interval,
    compare_conditions,
    power_analysis,
    one_way_anova,
    pairwise_comparisons,
    ResultsAnalyzer,
)

# Real agent evaluation (uses actual negotiation framework agents)
from .real_agent_evaluation import (
    RealAgentRunner,
    RealAgentConfig,
    create_mock_model,
    create_openai_model,
    create_google_model,
    create_ollama_model,
    create_together_model,
)

__all__ = [
    # Metrics
    'MetricsCollector',
    'CooperationMetrics',
    'AgentMetrics',
    'NegotiationMetrics',
    'ExperimentMetrics',
    'calculate_effect_size',
    'interpret_effect_size',
    # Scenarios
    'FisheryManagementScenario',
    'TreatyNegotiationScenario',
    'RealityGameshowScenario',
    'create_scenario',
    # Harness
    'ExperimentRunner',
    'ExperimentConfig',
    'ALL_MODULES',
    'create_ablation_configs',
    # Baselines
    'RandomAgent',
    'FixedStrategyAgent',
    'BasicLLMAgent',
    'SingleModuleAgent',
    'create_random_agent',
    'create_fixed_strategy_agent',
    'create_basic_llm_agent',
    'create_single_module_agent',
    'create_all_baselines',
    # Statistics
    'cohens_d',
    'interpret_cohens_d',
    'welchs_t_test',
    'confidence_interval',
    'compare_conditions',
    'power_analysis',
    'one_way_anova',
    'pairwise_comparisons',
    'ResultsAnalyzer',
    # Real agent evaluation
    'RealAgentRunner',
    'RealAgentConfig',
    'create_mock_model',
    'create_openai_model',
    'create_google_model',
    'create_ollama_model',
    'create_together_model',
]

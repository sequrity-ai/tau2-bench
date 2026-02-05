"""DSPy integration for tau2-bench.

This module provides DSPy optimization capabilities for conversational agents
using the Security-API (PLLM).
"""

from .lm import Tau2SequrityLM
from .config import DSPyConfig, OptimizationConfig
from .metrics import Tau2Metric, RewardMetric, TaskSuccessMetric
from .modules import AgentResponseModule
from .optimize import optimize_agent_prompt, OptimizationResult

__all__ = [
    "Tau2SequrityLM",
    "DSPyConfig",
    "OptimizationConfig",
    "Tau2Metric",
    "RewardMetric",
    "TaskSuccessMetric",
    "AgentResponseModule",
    "optimize_agent_prompt",
    "OptimizationResult",
]

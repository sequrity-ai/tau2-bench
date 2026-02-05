"""Optimization strategies for tau2-bench DSPy integration.

Wraps DSPy optimizers and GEPA for prompt optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import dspy


class StrategyType(str, Enum):
    """Available optimization strategy types."""

    BOOTSTRAP = "bootstrap"
    BOOTSTRAP_RS = "bootstrap_rs"
    MIPRO = "mipro"
    GEPA = "gepa"


@dataclass
class StrategyConfig:
    """Configuration for optimization strategies."""

    # Common settings
    max_iterations: int = 10
    num_candidates: int = 10

    # Bootstrap-specific
    max_bootstrapped_demos: int = 4
    max_labeled_demos: int = 16

    # MIPRO-specific
    num_threads: int = 6
    requires_permission_to_run: bool = False

    # GEPA-specific
    max_metric_calls: int = 50  # Default stopping condition for GEPA
    reflection_lm: str | None = None
    skip_perfect_score: bool = True
    display_progress_bar: bool = True


class OptimizationStrategy(ABC):
    """Abstract base class for optimization strategies."""

    def __init__(self, config: StrategyConfig | None = None):
        """Initialize the strategy.

        Args:
            config: Strategy configuration.
        """
        self.config = config or StrategyConfig()

    @abstractmethod
    def optimize(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable,
        valset: list[dspy.Example] | None = None,
    ) -> dspy.Module:
        """Run optimization and return optimized program.

        Args:
            program: DSPy module to optimize.
            trainset: Training examples.
            metric: Evaluation metric function.
            valset: Optional validation examples.

        Returns:
            Optimized DSPy module.
        """
        ...


class BootstrapStrategy(OptimizationStrategy):
    """Bootstrap few-shot optimization strategy.

    Uses DSPy's BootstrapFewShot optimizer to find good demonstrations.
    """

    def optimize(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable,
        valset: list[dspy.Example] | None = None,
    ) -> dspy.Module:
        """Run bootstrap few-shot optimization."""
        optimizer = dspy.BootstrapFewShot(
            metric=metric,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos,
        )

        return optimizer.compile(program, trainset=trainset)


class BootstrapRandomSearchStrategy(OptimizationStrategy):
    """Bootstrap with random search optimization strategy.

    Uses DSPy's BootstrapFewShotWithRandomSearch for more thorough exploration.
    """

    def optimize(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable,
        valset: list[dspy.Example] | None = None,
    ) -> dspy.Module:
        """Run bootstrap with random search optimization."""
        optimizer = dspy.BootstrapFewShotWithRandomSearch(
            metric=metric,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos,
            num_candidate_programs=self.config.num_candidates,
        )

        return optimizer.compile(program, trainset=trainset, valset=valset)


class MIPROStrategy(OptimizationStrategy):
    """MIPRO v2 optimization strategy.

    Uses DSPy's MIPROv2 for instruction and demonstration optimization.
    """

    def optimize(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable,
        valset: list[dspy.Example] | None = None,
    ) -> dspy.Module:
        """Run MIPRO v2 optimization."""
        optimizer = dspy.MIPROv2(
            metric=metric,
            num_candidates=self.config.num_candidates,
            num_threads=self.config.num_threads,
            requires_permission_to_run=self.config.requires_permission_to_run,
        )

        return optimizer.compile(
            program,
            trainset=trainset,
            valset=valset,
            max_bootstrapped_demos=self.config.max_bootstrapped_demos,
            max_labeled_demos=self.config.max_labeled_demos,
        )


@dataclass
class GEPAOptimizationResult:
    """Result from GEPA optimization."""

    best_prompt: str
    best_score: float
    result: Any  # GEPAResult


class GEPAStrategy(OptimizationStrategy):
    """GEPA (Genetic Evolution of Prompts and Architectures) strategy.

    Uses GEPA for evolutionary optimization of text components.
    """

    def optimize(
        self,
        program: dspy.Module,
        trainset: list[dspy.Example],
        metric: Callable,
        valset: list[dspy.Example] | None = None,
    ) -> dspy.Module:
        """Run GEPA optimization.

        Note: GEPA works differently from DSpy optimizers. Use
        optimize_prompt() for direct prompt optimization.
        """
        raise NotImplementedError(
            "GEPA requires direct prompt optimization. "
            "Use optimize_prompt() instead."
        )

    def optimize_prompt(
        self,
        seed_prompt: str,
        trainset: list[Any],
        evaluator: Callable[[str, Any], float],
        component_name: str = "pllm_prompt",
        valset: list[Any] | None = None,
    ) -> GEPAOptimizationResult:
        """Optimize a PLLM prompt directly using GEPA.

        Args:
            seed_prompt: Initial PLLM prompt to optimize.
            trainset: Training examples (can be any format).
            evaluator: Function that takes (prompt, example) and returns score.
            component_name: Name for the prompt component.
            valset: Optional validation examples.

        Returns:
            GEPAOptimizationResult with best prompt and history.
        """
        try:
            from gepa import EvaluationBatch, GEPAAdapter, optimize as gepa_optimize
        except ImportError:
            raise ImportError(
                "GEPA is not installed. Install with: pip install gepa"
            )

        # Create seed candidate
        seed_candidate = {component_name: seed_prompt}

        # Create adapter for GEPA
        class Tau2GEPAAdapter(GEPAAdapter):
            def __init__(adapter_self, eval_fn, use_fallback_proposer=True):
                adapter_self.eval_fn = eval_fn
                adapter_self._mutation_step = 0
                adapter_self.propose_new_texts = (
                    adapter_self._fallback_propose_new_texts
                    if use_fallback_proposer
                    else None
                )

            def evaluate(adapter_self, batch, candidate, capture_traces=False):
                prompt = candidate[component_name]
                outputs = []
                scores = []
                trajectories = [] if capture_traces else None

                for data_inst in batch:
                    try:
                        score = float(adapter_self.eval_fn(prompt, data_inst))
                    except Exception as e:
                        score = 0.0
                        outputs.append({"input": data_inst, "error": str(e)})
                    else:
                        outputs.append({"input": data_inst, "score": score})

                    scores.append(score)

                    if capture_traces:
                        trajectories.append({
                            "input": data_inst,
                            "score": score,
                        })

                return EvaluationBatch(
                    outputs=outputs,
                    scores=scores,
                    trajectories=trajectories,
                )

            def make_reflective_dataset(
                adapter_self, candidate, eval_batch, components_to_update
            ):
                dataset = {}
                for component in components_to_update:
                    records = []
                    if eval_batch.trajectories:
                        for traj in eval_batch.trajectories:
                            records.append({
                                "component": component,
                                "current_text": candidate.get(component, ""),
                                "input": traj.get("input"),
                                "score": traj.get("score", 0.0),
                                "feedback": "Improve task success rate on this input.",
                            })
                    dataset[component] = records
                return dataset

            def _fallback_propose_new_texts(
                adapter_self, candidate, reflective_dataset, components_to_update
            ):
                """Fallback proposer when no reflection_lm is provided."""
                adapter_self._mutation_step += 1
                proposals = {}
                suffix_options = [
                    "Focus on following the policy exactly.",
                    "Ensure all required information is collected before taking action.",
                    "Double-check that the correct tool is selected for each task.",
                    "Be helpful but always stay within policy boundaries.",
                ]
                suffix = suffix_options[
                    (adapter_self._mutation_step - 1) % len(suffix_options)
                ]

                for component in components_to_update:
                    current = candidate.get(component, "")
                    if suffix in current:
                        proposals[component] = current
                    else:
                        proposals[component] = f"{current.rstrip()}\n\n{suffix}"
                return proposals

        adapter = Tau2GEPAAdapter(
            evaluator,
            use_fallback_proposer=self.config.reflection_lm is None,
        )

        # Run GEPA optimization
        result = gepa_optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            task_lm=None,
            reflection_lm=self.config.reflection_lm,
            max_metric_calls=self.config.max_metric_calls,
            skip_perfect_score=self.config.skip_perfect_score,
            display_progress_bar=self.config.display_progress_bar,
        )

        return GEPAOptimizationResult(
            best_prompt=result.best_candidate[component_name],
            best_score=result.val_aggregate_scores[result.best_idx],
            result=result,
        )


def create_strategy(
    strategy_type: StrategyType | str,
    config: StrategyConfig | None = None,
    **kwargs,
) -> OptimizationStrategy:
    """Factory function to create optimization strategies.

    Args:
        strategy_type: Type of strategy to create.
        config: Strategy configuration.
        **kwargs: Additional strategy-specific arguments.

    Returns:
        Configured optimization strategy.
    """
    if isinstance(strategy_type, str):
        strategy_type = StrategyType(strategy_type)

    strategies = {
        StrategyType.BOOTSTRAP: BootstrapStrategy,
        StrategyType.BOOTSTRAP_RS: BootstrapRandomSearchStrategy,
        StrategyType.MIPRO: MIPROStrategy,
        StrategyType.GEPA: GEPAStrategy,
    }

    strategy_cls = strategies.get(strategy_type)
    if strategy_cls is None:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

    return strategy_cls(config=config, **kwargs)


__all__ = [
    "StrategyType",
    "StrategyConfig",
    "OptimizationStrategy",
    "BootstrapStrategy",
    "BootstrapRandomSearchStrategy",
    "MIPROStrategy",
    "GEPAStrategy",
    "GEPAOptimizationResult",
    "create_strategy",
]

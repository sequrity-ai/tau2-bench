"""Metrics adapter for tau2-bench DSPy integration.

Bridges tau2's RewardInfo evaluation system to DSPy metric format.
"""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import RewardType


@runtime_checkable
class Tau2Metric(Protocol):
    """Protocol for tau2 DSPy evaluation metrics.

    Metrics evaluate the quality of agent outputs and return a score
    between 0.0 (complete failure) and 1.0 (perfect success).
    """

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: list | None = None,
    ) -> float:
        """Evaluate the prediction against the example.

        Args:
            example: The input example with expected output.
            prediction: The agent response or evaluation result.
            trace: Optional trace information from DSPy.

        Returns:
            Score between 0.0 and 1.0.
        """
        ...


class BaseMetric(ABC):
    """Abstract base class for metrics with common functionality."""

    @abstractmethod
    def evaluate(
        self,
        expected: Any,
        actual: Any,
    ) -> float:
        """Core evaluation logic.

        Args:
            expected: Expected value from example.
            actual: Actual value from prediction.

        Returns:
            Score between 0.0 and 1.0.
        """
        ...

    def _extract_value(self, prediction: Any) -> Any:
        """Extract the value to evaluate from a prediction."""
        # Handle RewardInfo objects
        if isinstance(prediction, RewardInfo):
            return prediction.reward

        # Handle response objects with common attributes
        if hasattr(prediction, "final_value"):
            return prediction.final_value
        if hasattr(prediction, "reward"):
            return prediction.reward
        if hasattr(prediction, "content"):
            return prediction.content

        return prediction

    def _extract_expected(self, example: Any) -> Any:
        """Extract the expected value from an example."""
        # Common field names for expected values
        for field in ["expected", "answer", "output", "target", "label", "reward"]:
            if hasattr(example, field):
                return getattr(example, field)

        # If example is a dict
        if isinstance(example, dict):
            for field in ["expected", "answer", "output", "target", "label", "reward"]:
                if field in example:
                    return example[field]

        return example

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: list | None = None,
    ) -> float:
        """Evaluate the prediction against the example."""
        expected = self._extract_expected(example)
        actual = self._extract_value(prediction)
        return self.evaluate(expected, actual)


class RewardMetric(BaseMetric):
    """Metric that directly uses RewardInfo scores.

    This is the primary metric for tau2-bench optimization,
    directly using the task success rate (0.0-1.0).
    """

    def __init__(self, threshold: float = 1.0):
        """Initialize RewardMetric.

        Args:
            threshold: Minimum reward to consider successful.
        """
        self.threshold = threshold

    def evaluate(self, expected: Any, actual: Any) -> float:
        """Evaluate based on reward value.

        Args:
            expected: Expected reward (usually 1.0 for success).
            actual: Actual reward from RewardInfo.

        Returns:
            The reward score directly (0.0-1.0).
        """
        if isinstance(actual, (int, float)):
            return float(actual)
        return 0.0

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: list | None = None,
    ) -> float:
        """Evaluate using RewardInfo directly."""
        if isinstance(prediction, RewardInfo):
            return prediction.reward
        return super().__call__(example, prediction, trace)


class TaskSuccessMetric(BaseMetric):
    """Binary metric for task success (reward >= threshold)."""

    def __init__(self, threshold: float = 1.0 - 1e-6):
        """Initialize TaskSuccessMetric.

        Args:
            threshold: Minimum reward to consider successful.
        """
        self.threshold = threshold

    def evaluate(self, expected: Any, actual: Any) -> float:
        """Evaluate based on reward threshold.

        Args:
            expected: Expected value (unused).
            actual: Actual reward value.

        Returns:
            1.0 if reward >= threshold, else 0.0.
        """
        if isinstance(actual, (int, float)):
            return 1.0 if actual >= self.threshold else 0.0
        return 0.0

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: list | None = None,
    ) -> float:
        """Evaluate using RewardInfo directly."""
        if isinstance(prediction, RewardInfo):
            return 1.0 if prediction.reward >= self.threshold else 0.0
        return super().__call__(example, prediction, trace)


class RewardBreakdownMetric(BaseMetric):
    """Metric that evaluates specific components of the reward."""

    def __init__(
        self,
        reward_types: list[RewardType] | None = None,
        weights: dict[RewardType, float] | None = None,
    ):
        """Initialize RewardBreakdownMetric.

        Args:
            reward_types: Which reward types to consider.
            weights: Weights for each reward type (default: equal weights).
        """
        self.reward_types = reward_types or [
            RewardType.DB,
            RewardType.ACTION,
            RewardType.ENV_ASSERTION,
        ]
        self.weights = weights or {rt: 1.0 for rt in self.reward_types}

        # Normalize weights
        total = sum(self.weights.get(rt, 0.0) for rt in self.reward_types)
        if total > 0:
            self.weights = {
                rt: self.weights.get(rt, 0.0) / total for rt in self.reward_types
            }

    def evaluate(self, expected: Any, actual: Any) -> float:
        """Evaluate based on reward breakdown."""
        return 0.0  # Fallback if not using RewardInfo

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: list | None = None,
    ) -> float:
        """Evaluate using RewardInfo breakdown."""
        if not isinstance(prediction, RewardInfo):
            return 0.0

        if prediction.reward_breakdown is None:
            return prediction.reward

        total_score = 0.0
        for reward_type in self.reward_types:
            weight = self.weights.get(reward_type, 0.0)
            score = prediction.reward_breakdown.get(reward_type, 0.0)
            total_score += weight * score

        return total_score


class CompositeMetric:
    """Combine multiple metrics with weights.

    Example:
        ```python
        metric = CompositeMetric({
            RewardMetric(): 0.7,
            TaskSuccessMetric(): 0.3,
        })
        score = metric(example, prediction)
        ```
    """

    def __init__(self, metrics: dict[Tau2Metric, float]):
        """Initialize composite metric.

        Args:
            metrics: Dict mapping metrics to their weights.
                     Weights should sum to 1.0 for interpretable scores.
        """
        self.metrics = metrics
        total_weight = sum(metrics.values())
        if abs(total_weight - 1.0) > 0.001:
            # Normalize weights
            self.metrics = {m: w / total_weight for m, w in metrics.items()}

    def __call__(
        self,
        example: Any,
        prediction: Any,
        trace: list | None = None,
    ) -> float:
        """Evaluate using all metrics and return weighted score."""
        total_score = 0.0
        for metric, weight in self.metrics.items():
            try:
                score = metric(example, prediction, trace)
                total_score += score * weight
            except Exception:
                # If a metric fails, it contributes 0
                pass
        return total_score

    def evaluate_detailed(
        self,
        example: Any,
        prediction: Any,
        trace: list | None = None,
    ) -> dict[str, float]:
        """Return individual scores for each metric."""
        results = {}
        for metric, weight in self.metrics.items():
            name = metric.__class__.__name__
            try:
                score = metric(example, prediction, trace)
                results[name] = score
            except Exception as e:
                results[name] = 0.0
                results[f"{name}_error"] = str(e)
        results["weighted_total"] = self(example, prediction, trace)
        return results


def is_successful(reward: float, threshold: float = 1.0 - 1e-6) -> bool:
    """Check if a reward indicates success.

    Args:
        reward: The reward value.
        threshold: Minimum value for success.

    Returns:
        True if reward >= threshold.
    """
    return reward >= threshold

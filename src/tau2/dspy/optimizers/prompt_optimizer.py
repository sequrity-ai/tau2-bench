"""Main prompt optimizer for tau2-bench DSPy integration."""

from dataclasses import dataclass, field
from typing import Any, Callable

import dspy

from ..lm import Tau2SequrityLM, Tau2Response
from ..metrics import Tau2Metric, RewardMetric
from .strategies import (
    GEPAOptimizationResult,
    GEPAStrategy,
    OptimizationStrategy,
    StrategyConfig,
    StrategyType,
    create_strategy,
)


@dataclass
class OptimizationResult:
    """Result from prompt optimization."""

    best_prompt: str
    best_score: float
    optimized_program: dspy.Module | None = None
    history: list[tuple[str, float]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class Tau2PromptOptimizer:
    """Optimize PLLM system prompts for tau2-bench agents.

    This is the main interface for optimizing agent prompts. It supports
    multiple optimization strategies including DSPy optimizers and GEPA.

    Example:
        ```python
        from tau2.dspy import Tau2SequrityLM, Tau2PromptOptimizer, RewardMetric

        # Setup
        lm = Tau2SequrityLM(model="gpt-4.1", pllm_prompt="Initial prompt...")
        optimizer = Tau2PromptOptimizer(lm, metric=RewardMetric())

        # Optimize
        result = optimizer.optimize_with_gepa(trainset, valset)
        print(f"Best prompt: {result.best_prompt}")
        print(f"Best score: {result.best_score}")
        ```
    """

    def __init__(
        self,
        client: Tau2SequrityLM,
        metric: Tau2Metric | Callable | None = None,
        strategy: StrategyType | str | OptimizationStrategy = StrategyType.BOOTSTRAP,
        strategy_config: StrategyConfig | None = None,
    ):
        """Initialize the prompt optimizer.

        Args:
            client: Tau2SequrityLM client instance.
            metric: Evaluation metric (default: RewardMetric).
            strategy: Optimization strategy to use.
            strategy_config: Configuration for the strategy.
        """
        self.client = client
        self.metric = metric or RewardMetric()
        self.strategy_config = strategy_config or StrategyConfig()

        # Create strategy if string/enum provided
        if isinstance(strategy, OptimizationStrategy):
            self.strategy = strategy
        else:
            self.strategy = create_strategy(strategy, self.strategy_config)

    def _wrap_metric_for_dspy(
        self, metric: Tau2Metric | Callable, debug: bool = False
    ) -> Callable:
        """Wrap tau2 metric to work with DSPy's expected interface."""

        def wrapped_metric(example: dspy.Example, prediction, trace=None) -> float:
            # Get the actual response from the client
            response = self.client.last_response

            # Use Tau2Response if available
            if response is not None and response.is_success:
                score = metric(example, response, trace)

                if debug:
                    expected = getattr(example, "expected", None)
                    print(
                        f"[DEBUG] Expected: {expected!r}, "
                        f"Actual: {response.final_value!r}, Score: {score}"
                    )

                return score

            # Fallback: try to use prediction directly
            if hasattr(prediction, "answer"):
                pred_response = Tau2Response(
                    message=None,  # type: ignore
                    session_id=None,
                    is_success=True,
                    final_value=prediction.answer,
                )
                score = metric(example, pred_response, trace)

                if debug:
                    expected = getattr(example, "expected", None)
                    print(
                        f"[DEBUG-fallback] Expected: {expected!r}, "
                        f"Actual: {prediction.answer!r}, Score: {score}"
                    )

                return score

            if debug:
                print("[DEBUG] No valid response available")
            return 0.0

        return wrapped_metric

    def _create_module(self, signature_class: type | None = None) -> dspy.Module:
        """Create a DSPy module for optimization.

        Args:
            signature_class: Optional custom signature class.

        Returns:
            DSPy module configured for tau2.
        """
        if signature_class is None:
            class Tau2Signature(dspy.Signature):
                """Execute a customer service task."""

                user_message: str = dspy.InputField(desc="The user's message")
                policy: str = dspy.InputField(desc="Domain policy to follow")
                response: str = dspy.OutputField(desc="Agent response")

            signature_class = Tau2Signature

        return dspy.Predict(signature_class)

    def optimize(
        self,
        trainset: list[dspy.Example],
        initial_prompt: str | None = None,
        valset: list[dspy.Example] | None = None,
        signature_class: type | None = None,
        debug: bool = False,
    ) -> OptimizationResult:
        """Optimize the PLLM prompt using DSPy strategies.

        Args:
            trainset: Training examples with inputs and expected outputs.
            initial_prompt: Starting PLLM prompt (uses client's prompt if None).
            valset: Optional validation examples.
            signature_class: Optional custom DSPy signature class.
            debug: If True, print debug information.

        Returns:
            OptimizationResult with best prompt and metadata.
        """
        # Set initial prompt
        if initial_prompt is not None:
            self.client.pllm_prompt = initial_prompt

        # Configure DSPy to use our client
        dspy.configure(lm=self.client)

        # Create the module to optimize
        program = self._create_module(signature_class)

        # Wrap metric for DSPy
        wrapped_metric = self._wrap_metric_for_dspy(self.metric, debug=debug)

        # Run optimization
        optimized_program = self.strategy.optimize(
            program=program,
            trainset=trainset,
            metric=wrapped_metric,
            valset=valset,
        )

        # Get the best prompt
        best_prompt = self.client.pllm_prompt or ""

        # Evaluate final score
        if debug:
            print("\n[DEBUG] Final evaluation:")

        total_score = 0.0
        eval_set = valset or trainset
        for example in eval_set:
            try:
                query = getattr(example, "user_message", str(example))
                response = self.client.call_and_parse(prompt=query)
                score = self.metric(example, response)
                total_score += score

                if debug:
                    expected = getattr(example, "expected", None)
                    print(
                        f"  Query: {query[:50]}... "
                        f"Expected: {expected!r}, "
                        f"Got: {response.final_value!r}, "
                        f"Score: {score}"
                    )
            except Exception as e:
                if debug:
                    print(f"  Error evaluating: {e}")

        best_score = total_score / len(eval_set) if eval_set else 0.0

        return OptimizationResult(
            best_prompt=best_prompt,
            best_score=best_score,
            optimized_program=optimized_program,
            metadata={
                "strategy": str(self.strategy.__class__.__name__),
                "num_train_examples": len(trainset),
                "num_val_examples": len(valset) if valset else 0,
            },
        )

    def optimize_with_gepa(
        self,
        trainset: list[Any],
        initial_prompt: str | None = None,
        valset: list[Any] | None = None,
    ) -> OptimizationResult:
        """Optimize PLLM prompt using GEPA directly.

        This method uses GEPA's evolutionary optimization which can be
        more effective for complex prompt optimization tasks.

        Args:
            trainset: Training examples (format depends on evaluator).
            initial_prompt: Starting PLLM prompt.
            valset: Optional validation examples.

        Returns:
            OptimizationResult with best prompt and history.
        """
        seed_prompt = initial_prompt or self.client.pllm_prompt or ""

        # Create evaluator function
        def evaluator(prompt: str, example: Any) -> float:
            # Create a new client with this prompt
            test_client = self.client.with_pllm_prompt(prompt)

            try:
                # Get the query from example
                query = getattr(example, "user_message", None)
                if query is None:
                    query = getattr(example, "query", str(example))

                response = test_client.call_and_parse(prompt=query)
                return self.metric(example, response)
            except Exception:
                return 0.0

        # Create GEPA strategy
        gepa_strategy = GEPAStrategy(config=self.strategy_config)

        # Run optimization
        gepa_result = gepa_strategy.optimize_prompt(
            seed_prompt=seed_prompt,
            trainset=trainset,
            evaluator=evaluator,
            valset=valset,
        )

        # Update client with best prompt
        self.client.pllm_prompt = gepa_result.best_prompt

        return OptimizationResult(
            best_prompt=gepa_result.best_prompt,
            best_score=gepa_result.best_score,
            metadata={
                "strategy": "GEPA",
                "gepa_result": gepa_result.result,
            },
        )

    def evaluate(
        self,
        testset: list[Any],
        prompt: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate a prompt on a test set.

        Args:
            testset: Test examples to evaluate.
            prompt: Prompt to evaluate (uses current client prompt if None).

        Returns:
            Dictionary with evaluation results.
        """
        if prompt is not None:
            client = self.client.with_pllm_prompt(prompt)
        else:
            client = self.client

        results = []
        total_score = 0.0

        for example in testset:
            try:
                query = getattr(example, "user_message", None)
                if query is None:
                    query = getattr(example, "query", str(example))

                response = client.call_and_parse(prompt=query)
                score = self.metric(example, response)

                results.append({
                    "example": example,
                    "response": response,
                    "score": score,
                    "success": response.is_success,
                })
                total_score += score
            except Exception as e:
                results.append({
                    "example": example,
                    "response": None,
                    "score": 0.0,
                    "error": str(e),
                })

        avg_score = total_score / len(testset) if testset else 0.0

        return {
            "average_score": avg_score,
            "total_examples": len(testset),
            "successful": sum(1 for r in results if r.get("success", False)),
            "results": results,
        }

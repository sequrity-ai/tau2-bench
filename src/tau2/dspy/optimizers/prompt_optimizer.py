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

        # Create evaluator function that runs full tau2 simulations
        # Track simulation results for saving
        optimization_sim_results = []

        def evaluator(prompt: str, example: Any) -> float:
            """Run a full tau2 simulation and return the reward score."""
            from tau2.run import run_task
            from tau2.utils.utils import defence_params

            try:
                # Extract task from example
                task = example.get("task") if isinstance(example, dict) else getattr(example, "task", None)
                if task is None:
                    print("[GEPA Evaluator] Error: No task found in example")
                    return 0.0

                # Get domain from task
                domain = task.domain if hasattr(task, "domain") else example.get("domain", "mock")

                # Temporarily update defence_params with the test prompt
                # This gets picked up by generate() in llm_utils.py
                old_prompt = defence_params.get("pllm_prompt", "")
                defence_params["pllm_prompt"] = prompt

                # Debug: Confirm prompt is set
                print(f"[GEPA Evaluator] Testing task {task.id} with prompt (first 100 chars): {prompt[:100]}...")

                # Run the simulation
                # The agent will use defence_params["pllm_prompt"] via generate()
                sim_result = run_task(
                    domain=domain,
                    task=task,
                    agent="llm_agent",  # Use standard LLM agent
                    user="user_simulator",  # Use simulated user
                    llm_agent=self.client.model,
                    llm_args_agent={"temperature": 0.0},
                    llm_user=self.client.model,
                    llm_args_user={"temperature": 0.0},
                    max_steps=20,  # Limit steps for faster evaluation
                    max_errors=5,
                )

                # Save simulation result for later viewing
                optimization_sim_results.append({
                    "task_id": task.id,
                    "prompt_preview": prompt[:200],
                    "sim_result": sim_result,
                    "reward": sim_result.reward_info.reward if sim_result.reward_info else 0.0,
                })

                # Restore original prompt
                defence_params["pllm_prompt"] = old_prompt

                # Debug reward extraction
                if sim_result.reward_info:
                    reward = sim_result.reward_info.reward
                    print(f"[GEPA Evaluator] Task {task.id} completed. Reward: {reward}")
                    if hasattr(sim_result.reward_info, 'reward_breakdown'):
                        print(f"[GEPA Evaluator] Reward breakdown: {sim_result.reward_info.reward_breakdown}")
                else:
                    print(f"[GEPA Evaluator] WARNING: Task {task.id} has no reward_info!")
                    print(f"[GEPA Evaluator] sim_result type: {type(sim_result)}")
                    print(f"[GEPA Evaluator] sim_result attributes: {dir(sim_result)}")
                    reward = 0.0

                # Debug: show final message count
                if hasattr(sim_result, 'messages'):
                    print(f"[GEPA Evaluator] Simulation had {len(sim_result.messages)} messages")

                return reward

            except Exception as e:
                print(f"[GEPA Evaluator] Evaluation error: {e}")
                import traceback
                traceback.print_exc()
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

        # Save optimization simulation results to file
        import json
        from pathlib import Path
        import datetime

        output_dir = Path("/home/ubuntu/tau2-bench/data/tau2/optimization_runs")
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = output_dir / f"optimization_{timestamp}.json"

        # Save detailed results
        save_data = {
            "timestamp": timestamp,
            "best_score": gepa_result.best_score,
            "best_prompt": gepa_result.best_prompt,
            "simulations": [
                {
                    "task_id": sim["task_id"],
                    "prompt_preview": sim["prompt_preview"],
                    "reward": sim["reward"],
                    "messages": [msg.to_dict() if hasattr(msg, "to_dict") else str(msg)
                                for msg in sim["sim_result"].messages] if hasattr(sim["sim_result"], "messages") else [],
                }
                for sim in optimization_sim_results
            ]
        }

        with open(results_file, "w") as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"[GEPA] Saved {len(optimization_sim_results)} simulation runs to: {results_file}")

        return OptimizationResult(
            best_prompt=gepa_result.best_prompt,
            best_score=gepa_result.best_score,
            metadata={
                "strategy": "GEPA",
                "num_simulations": len(optimization_sim_results),
                "results_file": str(results_file),
                # Don't include gepa_result.result - it's not JSON serializable
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

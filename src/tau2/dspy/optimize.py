"""Main optimization workflow for tau2-bench DSPy integration.

This module provides the high-level API for optimizing agent prompts
using DSPy and GEPA strategies.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from tau2.data_model.simulation import RewardInfo
from tau2.data_model.tasks import Task
from tau2.registry import registry
from tau2.utils.utils import DATA_DIR

from .config import DSPyConfig, OptimizationConfig, OptimizedPromptInfo, StrategyType
from .lm import Tau2SequrityLM
from .metrics import RewardMetric, TaskSuccessMetric, Tau2Metric
from .optimizers import Tau2PromptOptimizer, StrategyConfig


@dataclass
class OptimizationResult:
    """Result from prompt optimization workflow."""

    domain: str
    best_prompt: str
    best_score: float
    strategy: str
    model: str
    train_size: int
    val_size: int
    metadata: dict = field(default_factory=dict)

    def save(self, path: Optional[Path] = None) -> Path:
        """Save the optimization result to a JSON file.

        Args:
            path: Path to save to. If None, uses default location.

        Returns:
            Path where the file was saved.
        """
        if path is None:
            output_dir = DATA_DIR / "tau2" / "optimized_prompts"
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / f"{self.domain}.json"

        info = OptimizedPromptInfo(
            domain=self.domain,
            prompt=self.best_prompt,
            score=self.best_score,
            strategy=self.strategy,
            model=self.model,
            train_size=self.train_size,
            val_size=self.val_size,
            metadata=self.metadata,
        )
        info.save(path)
        logger.info(f"Saved optimization result to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "OptimizationResult":
        """Load an optimization result from a JSON file."""
        info = OptimizedPromptInfo.load(path)
        return cls(
            domain=info.domain,
            best_prompt=info.prompt,
            best_score=info.score,
            strategy=info.strategy,
            model=info.model,
            train_size=info.train_size,
            val_size=info.val_size,
            metadata=info.metadata,
        )


def get_domain_tasks(
    domain: str,
    split: Optional[str] = None,
    task_set_name: Optional[str] = None,
) -> list[Task]:
    """Load tasks for a domain.

    Args:
        domain: Domain name (e.g., 'airline', 'retail').
        split: Task split name (e.g., 'train', 'test').
        task_set_name: Override task set name.

    Returns:
        List of Task objects.
    """
    task_set = task_set_name or domain
    tasks_loader = registry.get_tasks_loader(task_set)
    return tasks_loader(split)


def create_trainset_from_tasks(
    tasks: list[Task],
    include_policy: bool = False,
    policy: str = "",
) -> list[dict[str, Any]]:
    """Create a training set from tau2 tasks.

    Args:
        tasks: List of Task objects.
        include_policy: Whether to include policy in examples.
        policy: Domain policy text.

    Returns:
        List of training examples.
    """
    trainset = []

    for task in tasks:
        example = {
            "task_id": task.id,
            "task": task,
        }

        # Extract user scenario info
        if task.user_scenario:
            if task.user_scenario.instructions:
                # Get the initial user instruction
                instructions = task.user_scenario.instructions
                if hasattr(instructions, "text"):
                    example["user_message"] = instructions.text
                elif isinstance(instructions, str):
                    example["user_message"] = instructions
                else:
                    example["user_message"] = str(instructions)

            if task.user_scenario.persona:
                example["persona"] = task.user_scenario.persona

        # Include policy if requested
        if include_policy:
            example["policy"] = policy

        # Expected reward is always 1.0 (success)
        example["expected"] = 1.0

        trainset.append(example)

    return trainset


def optimize_agent_prompt(
    domain: str,
    strategy: StrategyType | str = StrategyType.GEPA,
    model: str = "gpt-4.1",
    train_split: str = "train",
    val_split: Optional[str] = "test",
    task_set_name: Optional[str] = None,
    max_train_tasks: Optional[int] = None,
    max_val_tasks: Optional[int] = None,
    initial_prompt: Optional[str] = None,
    optimization_config: Optional[OptimizationConfig] = None,
    evaluator: Optional[Callable[[str, Any], float]] = None,
    output_path: Optional[Path] = None,
    debug: bool = False,
) -> OptimizationResult:
    """Optimize an agent prompt for a specific domain.

    This is the main entry point for prompt optimization in tau2-bench.

    Args:
        domain: Domain name (e.g., 'airline', 'retail', 'telecom').
        strategy: Optimization strategy to use.
        model: LLM model to use.
        train_split: Split name for training data.
        val_split: Split name for validation data.
        task_set_name: Override task set name.
        max_train_tasks: Maximum number of training tasks.
        max_val_tasks: Maximum number of validation tasks.
        initial_prompt: Initial prompt to optimize from.
        optimization_config: Configuration for the optimizer.
        evaluator: Custom evaluator function.
        output_path: Path to save the optimized prompt.
        debug: Enable debug output.

    Returns:
        OptimizationResult with the optimized prompt and metadata.

    Example:
        ```python
        result = optimize_agent_prompt(
            domain="airline",
            strategy="gepa",
            model="gpt-4.1",
            train_split="train",
            val_split="test",
        )
        print(f"Best score: {result.best_score}")
        result.save()
        ```
    """
    logger.info(f"Starting prompt optimization for domain: {domain}")
    logger.info(f"Strategy: {strategy}, Model: {model}")

    # Load domain environment and policy
    env_constructor = registry.get_env_constructor(domain)
    environment = env_constructor()
    policy = environment.get_policy()

    # Load tasks
    logger.info(f"Loading tasks from splits: train={train_split}, val={val_split}")
    train_tasks = get_domain_tasks(domain, train_split, task_set_name)
    val_tasks = get_domain_tasks(domain, val_split, task_set_name) if val_split else []

    # Apply task limits
    if max_train_tasks and len(train_tasks) > max_train_tasks:
        train_tasks = train_tasks[:max_train_tasks]
    if max_val_tasks and val_tasks and len(val_tasks) > max_val_tasks:
        val_tasks = val_tasks[:max_val_tasks]

    logger.info(f"Training tasks: {len(train_tasks)}, Validation tasks: {len(val_tasks)}")

    # Create training/validation sets
    trainset = create_trainset_from_tasks(train_tasks, include_policy=True, policy=policy)
    valset = create_trainset_from_tasks(val_tasks, include_policy=True, policy=policy) if val_tasks else None

    # Set up initial prompt
    if initial_prompt is None:
        initial_prompt = """
You are an expert Python code generator acting as the execution engine for a helpful assistant.

**Your Task**: Generate simple, clean Python code to solve the user's query in a single step.

**Core Principles**:
- **Resourcefulness**: When a tool is available for a task, prefer using it over writing custom code.
- **Decisiveness**: Act on the user's intent directly. Treat requests as commands to execute.
- **Policy Compliance**: Always follow the domain policy. If an action violates policy, explain why and suggest alternatives.
- **Error Correction**: If a previous attempt failed, fix the code. Don't repeat failed logic.
- **Final Return Value**: Always store the final result in `final_return_value`. This should be a helpful message for the user (keep it under 5 sentences).

**Code Generation Rules**:
- Enclose all Python code in <sequrity_plan> tags
- Use keyword arguments for all tool function calls
- Store your final response in the `final_return_value` variable
- If the user's request is unclear, put clarifying questions in `final_return_value`

**Remember**: Follow the policy strictly. Be helpful but never violate policy constraints.
""".strip()

    # Create LM client
    lm = Tau2SequrityLM(
        model=model,
        pllm_prompt=initial_prompt,
    )

    # Create optimizer
    opt_config = optimization_config or OptimizationConfig()
    strategy_config = StrategyConfig(
        max_iterations=opt_config.max_iterations,
        num_candidates=opt_config.num_candidates,
        max_bootstrapped_demos=opt_config.max_bootstrapped_demos,
        max_labeled_demos=opt_config.max_labeled_demos,
        max_metric_calls=opt_config.max_metric_calls,
        reflection_lm=opt_config.reflection_lm,
        skip_perfect_score=opt_config.skip_perfect_score,
        display_progress_bar=opt_config.display_progress_bar,
    )

    metric = RewardMetric()
    optimizer = Tau2PromptOptimizer(
        client=lm,
        metric=metric,
        strategy=strategy,
        strategy_config=strategy_config,
    )

    # Create evaluator if not provided
    if evaluator is None:
        def default_evaluator(prompt: str, example: dict) -> float:
            """Default evaluator using RewardMetric."""
            # This would run a full simulation - simplified for now
            # In practice, this should run the task and return the reward
            return 0.0  # Placeholder

        evaluator = default_evaluator

    # Run optimization
    logger.info("Running optimization...")
    if isinstance(strategy, str):
        strategy_type = StrategyType(strategy)
    else:
        strategy_type = strategy

    if strategy_type == StrategyType.GEPA:
        opt_result = optimizer.optimize_with_gepa(
            trainset=trainset,
            initial_prompt=initial_prompt,
            valset=valset,
        )
    else:
        import dspy
        # Convert trainset to DSPy Examples
        dspy_trainset = [
            dspy.Example(**ex).with_inputs("user_message", "policy")
            for ex in trainset
        ]
        dspy_valset = [
            dspy.Example(**ex).with_inputs("user_message", "policy")
            for ex in valset
        ] if valset else None

        opt_result = optimizer.optimize(
            trainset=dspy_trainset,
            initial_prompt=initial_prompt,
            valset=dspy_valset,
            debug=debug,
        )

    logger.info(f"Optimization complete. Best score: {opt_result.best_score}")

    # Create result
    result = OptimizationResult(
        domain=domain,
        best_prompt=opt_result.best_prompt,
        best_score=opt_result.best_score,
        strategy=str(strategy),
        model=model,
        train_size=len(trainset),
        val_size=len(valset) if valset else 0,
        metadata=opt_result.metadata,
    )

    # Save if output path provided
    if output_path:
        result.save(output_path)

    return result


def load_optimized_prompt(domain: str, path: Optional[Path] = None) -> Optional[str]:
    """Load an optimized prompt for a domain.

    Args:
        domain: Domain name.
        path: Custom path to the prompt file.

    Returns:
        The optimized prompt string, or None if not found.
    """
    if path is None:
        path = DATA_DIR / "tau2" / "optimized_prompts" / f"{domain}.json"

    if not path.exists():
        logger.warning(f"No optimized prompt found for domain: {domain}")
        return None

    try:
        result = OptimizationResult.load(path)
        logger.info(f"Loaded optimized prompt for {domain} (score: {result.best_score})")
        return result.best_prompt
    except Exception as e:
        logger.error(f"Error loading optimized prompt: {e}")
        return None


def evaluate_prompt(
    domain: str,
    prompt: str,
    model: str = "gpt-4.1",
    split: str = "test",
    max_tasks: Optional[int] = None,
) -> dict[str, Any]:
    """Evaluate a prompt on a domain's test set.

    Args:
        domain: Domain name.
        prompt: The prompt to evaluate.
        model: LLM model to use.
        split: Task split to evaluate on.
        max_tasks: Maximum number of tasks to evaluate.

    Returns:
        Dictionary with evaluation metrics.
    """
    # Load tasks
    tasks = get_domain_tasks(domain, split)
    if max_tasks and len(tasks) > max_tasks:
        tasks = tasks[:max_tasks]

    # Create LM client
    lm = Tau2SequrityLM(model=model, pllm_prompt=prompt)

    # Create optimizer for evaluation
    metric = RewardMetric()
    optimizer = Tau2PromptOptimizer(client=lm, metric=metric)

    # Create test set
    env_constructor = registry.get_env_constructor(domain)
    environment = env_constructor()
    policy = environment.get_policy()

    testset = create_trainset_from_tasks(tasks, include_policy=True, policy=policy)

    # Evaluate
    return optimizer.evaluate(testset, prompt=prompt)

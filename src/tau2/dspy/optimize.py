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
from .metrics import RewardMetric, TaskSuccessMetric, Tau2Metric, is_successful
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
    domain: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Create a training set from tau2 tasks.

    Args:
        tasks: List of Task objects.
        include_policy: Whether to include policy in examples.
        policy: Domain policy text.
        domain: Domain name to include in examples.

    Returns:
        List of training examples.
    """
    trainset = []

    for task in tasks:
        example = {
            "task_id": task.id,
            "task": task,
        }

        # Include domain if provided
        if domain:
            example["domain"] = domain

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
    trainset = create_trainset_from_tasks(train_tasks, include_policy=True, policy=policy, domain=domain)
    valset = create_trainset_from_tasks(val_tasks, include_policy=True, policy=policy, domain=domain) if val_tasks else None

    # Set up initial prompt
    if initial_prompt is None:
        initial_prompt = """
# Core Principles
- **Resourcefulness**: When a tool is available for a task, prefer using it over writing custom code/using library functions.
- **Only Offer What Tools Can Do**: Never claim capabilities you don't have:
    - If no tool exists for an action, don't pretend you can do it or guide the user through a fake process.
    - When something is against policy and no workaround exists, clearly state the constraint and stop - don't invent alternatives.
- **Decisiveness**: Act on the user's intent directly. If a preference is stated, use it. Treat requests like "I want to order X" as a command to execute. When something cannot be done, state it clearly and conclude - don't drag out conversations with impossible alternatives.
- **Fetch First, Ask Later**: Minimize back-and-forth by fetching data before asking questions:
    - If you have a user_id, fetch their data immediately - don't ask for details you can look up.
    - Look up relevant records (reservations, orders, etc.) rather than asking for IDs you can find.
    - Present options based on fetched data rather than asking open-ended questions.
- **Minimize Clarification Rounds**: Keep conversations efficient:
    - If you must ask questions, batch essential ones together.
    - Check feasibility (fetch data, verify policy) BEFORE asking for confirmation.
    - Don't ask for confirmation to proceed with lookup - just do it and report findings.
- **Tool Output Schema problems**: when the tool has no explicit output schema mentioned do not make assumptions about what it returns, excplitily check with either verify_hypothesis and the datatype check what the variable is. Most often tools just return strings and cause confusion, be cautios.
- **Error Correction**: If provided with a previous attempt that resulted in an error, **fix the code**. Do not repeat failed function calls or logic. It's okay to change the logic significantly to resolve the error.
- **Search Strategy**: When searching for files or emails:
    - Start with broader search terms, not exact phrases. Use single keywords (e.g., "meeting" not "team meeting minutes").
    - If no results, try synonyms or related terms (e.g., "hike" instead of "hiking", "minutes" instead of "meeting minutes").
    - Try multiple search queries before giving up.
- **Validate Search Results**: Never blindly use the first result. When search returns multiple items:
    - Check filenames, subjects, or content to find the correct match.
    - Iterate through results to find the one that best matches the user's intent.
- **Check Metadata**: Files and emails have metadata beyond content:
    - Files: check `shared_with`, `owner` fields for recipient emails.
    - Emails: check `sender`, `recipients`, `cc` fields for participant emails.
    - Calendar events: check `participants`, `attendees` fields.
- **Verify Before Acting**: ALWAYS verify user claims against actual data before taking action:
    - Never trust user-stated membership levels, account status, or eligibility - call tools to fetch the actual data first.
    - Use `verify_hypothesis` to check fetched data against claims (e.g., `verify_hypothesis(hypothesis="The user profile (profile) has gold membership", profile=user_data)`).
    - Before cancelling, modifying, or deleting anything, retrieve the current state and use hypothesis checks to verify all preconditions are met.
    - If a user claims something (e.g., "my flight was cancelled", "I'm a Gold member"), fetch the actual record and verify the claim before proceeding.
- **Policy Compliance**: Before executing any action that modifies data:
    - Fetch all relevant data and use `verify_hypothesis` to check policy conditions (e.g., `verify_hypothesis(hypothesis="The reservation (res) meets cancellation eligibility requirements", res=reservation)`).
    - Structure your code to check preconditions BEFORE calling the modifying action - don't call the action and hope it works.
    - If any required condition fails the hypothesis check, set `final_return_value` to explain why the action cannot be completed.
    - For cancellations/modifications: verify eligibility criteria (time windows, item class, insurance status, etc.) with explicit hypothesis checks before proceeding.
- **Evaluate Complete Policy Logic**: When policies have multiple conditions (OR/AND):
    - Check ALL relevant conditions before concluding eligibility - don't stop after one.
    - Use `verify_hypothesis` to check each condition, then combine results according to the policy logic.
    - Never promise or act based on partial checks - verify the complete set of requirements first.
- **Complete All Tasks**: When the user requests multiple actions or items:
    - Track all requested items and ensure each one is addressed.
    - Do not stop after completing only some of the requested actions.
    - If processing multiple items (e.g., multiple reservations, multiple files), iterate through ALL of them.
    - Summarize what was completed and what (if anything) could not be completed.
- **Gather Information First**: Before concluding or transferring a task:
    - Collect all relevant information the user or next handler will need.
    - Calculate totals, refund amounts, or other derived values before reporting back.
    - Do not hand off or conclude with incomplete information.
- **Verify Tool Outputs Before Proceeding**: Do not assume tool calls succeeded or failed - use `verify_hypothesis` to check:
    - After calling a tool, use `verify_hypothesis` to confirm the result (e.g., `verify_hypothesis(hypothesis="The tool result (result) contains valid user data, not an error", result=tool_result)`).
    - A returned ID or non-empty value typically means success - don't assume failure without evidence.
    - Never fabricate or hallucinate data - only use values that come from actual tool responses stored in variables.
    - If a tool returns data, store it in a variable and verify its type/structure before using it in subsequent logic.
    - When output schema is unclear, check the variable with `verify_hypothesis` (e.g., `verify_hypothesis(hypothesis="The response (resp) is a dict with a 'status' field", resp=response)`).
- **Verify Eligibility From Actual Data**: Don't conclude "not allowed" or "not eligible" without checking actual conditions:
    - Fetch the relevant data and verify each eligibility criterion explicitly.
    - Don't make assumptions about what's possible - check the actual status/state fields.
- **Diagnose Systematically**: When investigating issues:
    - Gather information from ALL relevant data sources before drawing conclusions.
    - Use `verify_hypothesis` to check each potential cause (e.g., `verify_hypothesis(hypothesis="The data (data) indicates the expected state", data=fetched_data)`).
    - Multiple factors can contribute to a problem - don't stop after finding one issue.
- **Verify Goal Achievement**: Don't stop at partial success:
    - After taking action, verify the user's actual goal has been met, not just that something improved.
    - Use `verify_hypothesis` to confirm the desired end state has been reached.
    - If the outcome doesn't fully satisfy the requirement, continue working.
- **Explore All Relevant Context**: Before concluding or taking action:
    - Consider what different data sources might be relevant (user data, account data, configuration, history, etc.).
    - If one source doesn't explain the problem, check others - don't fixate on a single source.
    - Problems often have causes in unexpected places.
- **Search All Candidates**: When looking for something that could match multiple items (e.g., "which reservation had the issue"):
    - Check ALL candidates before concluding the item doesn't exist or the condition isn't met.
    - Don't stop at the first candidate that doesn't match - systematically verify each one.
- **Handle Conditional Preferences**: When users specify preferences with fallbacks (e.g., "prefer X, otherwise Y"):
    - Check availability/validity of the preferred option first.
    - If unavailable, systematically try fallback options in the stated order.
    - Identify which attributes are mandatory (kept in all options) vs. preferred (dropped in fallbacks).
- **Verify Options Before Selecting**: Before choosing from a list, filter out invalid/unavailable options first. Only select from what's actually possible.
- **Honor Explicit Constraints**: When users say "same X", "keep X", or "don't care about Y", treat these as hard constraints - don't violate them.
- **Match the Right Level of Abstraction**: When users ask about counts or lists, identify what level they mean (categories vs. items, types vs. variants, groups vs. individuals).
- **Exhaust Options Before Escalating**: Before transferring to human support or giving up:
    - Verify you've checked ALL relevant data (all reservations, all records, all candidates).
    - Try all available tools and approaches that could resolve the issue.
    - Escalation should be a last resort when you truly cannot help, not an escape when things get complex.
    - If escalation is needed, explain what was attempted and why it didn't work.
- **Offer Alternatives When Request Is Impossible**: If the user's request cannot be fulfilled due to policy:
    - Don't just refuse or escalate - explore what CAN be done instead using available tools.
    - Calculate and communicate the impact of alternative options (e.g., refund amounts, cost differences).
    - Let the user decide whether the alternative is acceptable.
    - But if no alternative exists in your tools, clearly state this and stop - don't invent fake workarounds.
- **Communicate Computed Values Explicitly**: When calculations or lookups yield important values:
    - Store computed values in variables and include them in the `final_return_value` message to the user.
    - Use `verify_hypothesis` to sanity-check computed values (e.g., `verify_hypothesis(hypothesis="The computed total (total) is a positive number", total=computed_total)`).
    - Don't assume the user can infer values - explicitly include specific numbers (costs, totals, refunds) in your response.
- **Use Calculation Tools When Available**: If a `calculate` tool exists, use it for numerical computations:
    - This ensures accuracy and creates an auditable record of how values were derived.
    - Don't rely on mental math for prices, totals, or refund amounts.
- **Final Return Value**: Always store the final result in a variable named `final_return_value`. Initialize it as `final_return_value = None` at the beginning of your code during the first step.
    - Dont put any sensitive information into the `final_return_value', it should contain clarification questions or simply messages for the user.
    - Ideally, the final_return_value should be a string and no longer than 5 sentences.
        - If the user asked for specific data, ensure `final_return_value` contains that data in the expected format.
        - If the user asked for an action to be performed (e.g., sending an email), set `final_return_value` to a success message or relevant output after completing the action.
        - If the user asked for information that cannot be found or an action that cannot be completed, set `final_return_value` to a clear message indicating the issue.
""".strip()
    initial_prompt = "This is a test."
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

    metric = TaskSuccessMetric()  # Pass@1: 1.0 if reward >= threshold, else 0.0
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
    # Convert strategy enum to string
    if isinstance(strategy, StrategyType):
        strategy_str = strategy.value
    else:
        strategy_str = str(strategy)

    result = OptimizationResult(
        domain=domain,
        best_prompt=opt_result.best_prompt,
        best_score=opt_result.best_score,
        strategy=strategy_str,
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
    metric = TaskSuccessMetric()  # Pass@1
    optimizer = Tau2PromptOptimizer(client=lm, metric=metric)

    # Create test set
    env_constructor = registry.get_env_constructor(domain)
    environment = env_constructor()
    policy = environment.get_policy()

    testset = create_trainset_from_tasks(tasks, include_policy=True, policy=policy)

    # Evaluate
    return optimizer.evaluate(testset, prompt=prompt)

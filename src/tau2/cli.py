import argparse
import json

from tau2.config import (
    DEFAULT_AGENT_IMPLEMENTATION,
    DEFAULT_LLM_AGENT,
    DEFAULT_LLM_TEMPERATURE_AGENT,
    DEFAULT_LLM_TEMPERATURE_USER,
    DEFAULT_LLM_USER,
    DEFAULT_LOG_LEVEL,
    DEFAULT_MAX_CONCURRENCY,
    DEFAULT_MAX_ERRORS,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_TRIALS,
    DEFAULT_SEED,
    DEFAULT_USER_IMPLEMENTATION,
)
from tau2.data_model.simulation import RunConfig
from tau2.run import get_options, run_domain
from tau2.scripts.leaderboard.verify_trajectories import VerificationMode

from tau2.utils.utils import defence_params as global_defence_params
from tau2.utils.utils import smart_update_headers 

def add_run_args(parser):
    """Add run arguments to a parser."""
    domains = get_options().domains
    parser.add_argument(
        "--domain",
        "-d",
        type=str,
        choices=domains,
        help="The domain to run the simulation on",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=DEFAULT_NUM_TRIALS,
        help="The number of times each task is run. Default is 1.",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=DEFAULT_AGENT_IMPLEMENTATION,
        choices=get_options().agents,
        help=f"The agent implementation to use. Default is {DEFAULT_AGENT_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--agent-llm",
        type=str,
        default=DEFAULT_LLM_AGENT,
        help=f"The LLM to use for the agent. Default is {DEFAULT_LLM_AGENT}.",
    )
    parser.add_argument(
        "--agent-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_AGENT},
        help=f"The arguments to pass to the LLM for the agent. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_AGENT}}}'.",
    )
    parser.add_argument(
        "--user",
        type=str,
        choices=get_options().users,
        default=DEFAULT_USER_IMPLEMENTATION,
        help=f"The user implementation to use. Default is {DEFAULT_USER_IMPLEMENTATION}.",
    )
    parser.add_argument(
        "--user-llm",
        type=str,
        default=DEFAULT_LLM_USER,
        help=f"The LLM to use for the user. Default is {DEFAULT_LLM_USER}.",
    )
    parser.add_argument(
        "--user-llm-args",
        type=json.loads,
        default={"temperature": DEFAULT_LLM_TEMPERATURE_USER},
        help=f"The arguments to pass to the LLM for the user. Default is '{{\"temperature\": {DEFAULT_LLM_TEMPERATURE_USER}}}'.",
    )
    parser.add_argument(
        "--task-set-name",
        type=str,
        default=None,
        choices=get_options().task_sets,
        help="The task set to run the simulation on. If not provided, will load default task set for the domain.",
    )
    parser.add_argument(
        "--task-split-name",
        type=str,
        default="base",
        help="The task split to run the simulation on. If not provided, will load 'base' split.",
    )
    parser.add_argument(
        "--task-ids",
        type=str,
        nargs="+",
        help="(Optional) run only the tasks with the given IDs. If not provided, will run all tasks.",
    )
    parser.add_argument(
        "--num-tasks",
        type=int,
        default=None,
        help="The number of tasks to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help=f"The maximum number of steps to run the simulation. Default is {DEFAULT_MAX_STEPS}.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=DEFAULT_MAX_ERRORS,
        help=f"The maximum number of tool errors allowed in a row in the simulation. Default is {DEFAULT_MAX_ERRORS}.",
    )
    parser.add_argument(
        "--save-to",
        type=str,
        required=False,
        help="The path to save the simulation results. Will be saved to data/simulations/<save_to>.json. If not provided, will save to <domain>_<agent>_<user>_<llm_agent>_<llm_user>_<timestamp>.json. If the file already exists, it will try to resume the run.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=DEFAULT_MAX_CONCURRENCY,
        help=f"The maximum number of concurrent simulations to run. Default is {DEFAULT_MAX_CONCURRENCY}.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"The seed to use for the simulation. Default is {DEFAULT_SEED}.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=DEFAULT_LOG_LEVEL,
        help=f"The log level to use for the simulation. Default is {DEFAULT_LOG_LEVEL}.",
    )
    parser.add_argument(
        "--enforce-communication-protocol",
        action="store_true",
        default=False,
        help="Enforce communication protocol rules (e.g., no mixed messages with text and tool calls). Default is False.",
    )
    parser.add_argument(
        "--attack-config",
        default=None
    )


def main():
    global global_defence_params
    parser = argparse.ArgumentParser(description="Tau2 command line interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

       # Run command
    run_parser = subparsers.add_parser("run", help="Run a benchmark")
    add_run_args(run_parser)
    run_parser.set_defaults(
        func=lambda args: run_domain(
            RunConfig(
                domain=args.domain,
                task_set_name=args.task_set_name,
                task_split_name=args.task_split_name,
                task_ids=args.task_ids,
                num_tasks=args.num_tasks,
                agent=args.agent,
                llm_agent=args.agent_llm,
                llm_args_agent=args.agent_llm_args,
                user=args.user,
                llm_user=args.user_llm,
                llm_args_user=args.user_llm_args,
                num_trials=args.num_trials,
                max_steps=args.max_steps,
                max_errors=args.max_errors,
                save_to=args.save_to,
                max_concurrency=args.max_concurrency,
                seed=args.seed,
                log_level=args.log_level,
                enforce_communication_protocol=args.enforce_communication_protocol,
                attack_config = args.attack_config,
            )
        )
    )

    # Play command
    play_parser = subparsers.add_parser(
        "play", help="Play manual mode - interact with a domain as the agent"
    )
    play_parser.set_defaults(func=lambda args: run_manual_mode())

    # View command
    view_parser = subparsers.add_parser("view", help="View simulation results")
    view_parser.add_argument(
        "--dir",
        type=str,
        help="Directory containing simulation files. Defaults to data/simulations if not specified.",
    )
    view_parser.add_argument(
        "--file",
        type=str,
        help="Path to the simulation results file to view",
    )
    view_parser.add_argument(
        "--only-show-failed",
        action="store_true",
        help="Only show failed tasks.",
    )
    view_parser.add_argument(
        "--only-show-all-failed",
        action="store_true",
        help="Only show tasks that failed in all trials.",
    )
    view_parser.set_defaults(func=lambda args: run_view_simulations(args))

    # Domain command
    domain_parser = subparsers.add_parser("domain", help="Show domain documentation")
    domain_parser.add_argument(
        "domain",
        type=str,
        help="Name of the domain to show documentation for (e.g., 'airline', 'mock')",
    )
    domain_parser.set_defaults(func=lambda args: run_show_domain(args))

    # Start command
    start_parser = subparsers.add_parser("start", help="Start all servers")
    start_parser.set_defaults(func=lambda args: run_start_servers())

    # Check data command
    check_data_parser = subparsers.add_parser(
        "check-data", help="Check if data directory is properly configured"
    )
    check_data_parser.set_defaults(func=lambda args: run_check_data())

    # Evaluate trajectories command
    evaluate_parser = subparsers.add_parser(
        "evaluate-trajs", help="Evaluate trajectories and update rewards"
    )
    evaluate_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    evaluate_parser.add_argument(
        "-o",
        "--output-dir",
        help="Directory to save updated trajectory files with recomputed rewards. If not provided, only displays metrics.",
    )
    evaluate_parser.set_defaults(func=lambda args: run_evaluate_trajectories(args))

    # Submit command with subcommands
    submit_parser = subparsers.add_parser(
        "submit", help="Submission management for the leaderboard"
    )
    submit_subparsers = submit_parser.add_subparsers(
        dest="submit_command", help="Submit subcommands", required=True
    )

    # Submit prepare subcommand
    submit_prepare_parser = submit_subparsers.add_parser(
        "prepare", help="Prepare a submission for the leaderboard"
    )
    submit_prepare_parser.add_argument(
        "input_paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    submit_prepare_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory to save the submission and trajectories",
    )
    submit_prepare_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip trajectory verification step",
    )
    submit_prepare_parser.set_defaults(func=lambda args: run_prepare_submission(args))

    # Submit validate subcommand
    submit_validate_parser = submit_subparsers.add_parser(
        "validate", help="Validate an existing submission directory"
    )
    submit_validate_parser.add_argument(
        "submission_dir",
        help="Path to the submission directory to validate",
    )
    submit_validate_parser.add_argument(
        "--mode",
        type=VerificationMode,
        choices=[mode.value for mode in VerificationMode],
        default=VerificationMode.PUBLIC,
        help=f"Verification mode. Default is '{VerificationMode.PUBLIC.value}'",
    )
    submit_validate_parser.set_defaults(func=lambda args: run_validate_submission(args))

    # Submit verify-trajs subcommand
    submit_verify_parser = submit_subparsers.add_parser(
        "verify-trajs", help="Verify trajectory files"
    )
    submit_verify_parser.add_argument(
        "paths",
        nargs="+",
        help="Paths to trajectory files, directories, or glob patterns",
    )
    submit_verify_parser.add_argument(
        "--mode",
        type=VerificationMode,
        choices=[mode.value for mode in VerificationMode],
        default=VerificationMode.PUBLIC,
        help=f"Verification mode. Default is '{VerificationMode.PUBLIC.value}'",
    )
    submit_verify_parser.set_defaults(func=lambda args: run_verify_trajectories(args))

    # Optimize command - DSPy prompt optimization
    optimize_parser = subparsers.add_parser(
        "optimize", help="Optimize agent prompts using DSPy/GEPA"
    )
    optimize_parser.add_argument(
        "--domain",
        "-d",
        type=str,
        required=True,
        choices=get_options().domains,
        help="The domain to optimize the prompt for",
    )
    optimize_parser.add_argument(
        "--strategy",
        type=str,
        default="gepa",
        choices=["bootstrap", "bootstrap_rs", "mipro", "gepa"],
        help="Optimization strategy to use. Default is 'gepa'.",
    )
    optimize_parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_LLM_AGENT,
        help=f"The LLM to use. Default is {DEFAULT_LLM_AGENT}.",
    )
    optimize_parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Task split to use for training. Default is 'train'.",
    )
    optimize_parser.add_argument(
        "--val-split",
        type=str,
        default="test",
        help="Task split to use for validation. Default is 'test'.",
    )
    optimize_parser.add_argument(
        "--task-set-name",
        type=str,
        default=None,
        help="Override the task set name.",
    )
    optimize_parser.add_argument(
        "--max-train-tasks",
        type=int,
        default=None,
        help="Maximum number of training tasks to use.",
    )
    optimize_parser.add_argument(
        "--max-val-tasks",
        type=int,
        default=None,
        help="Maximum number of validation tasks to use.",
    )
    optimize_parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum optimization iterations. Default is 10.",
    )
    optimize_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output path for the optimized prompt JSON file.",
    )
    optimize_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output.",
    )
    optimize_parser.set_defaults(func=lambda args: run_optimize(args))

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return

    if hasattr(args, "attack_config"):
        if args.attack_config is not None and args.attack_config != "none":
            global_defence_params = smart_update_headers(global_defence_params, args.attack_config)

    args.func(args)


def run_view_simulations(args):
    from tau2.scripts.view_simulations import main as view_main

    view_main(
        sim_file=args.file,
        only_show_failed=args.only_show_failed,
        only_show_all_failed=args.only_show_all_failed,
        sim_dir=args.dir,
    )


def run_show_domain(args):
    from tau2.scripts.show_domain_doc import main as domain_main

    domain_main(args.domain)


def run_start_servers():
    from tau2.scripts.start_servers import main as start_main

    start_main()


def run_check_data():
    from tau2.scripts.check_data import main as check_data_main

    check_data_main()


def run_verify_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.leaderboard.verify_trajectories import verify_trajectories

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    verify_trajectories(args.paths, args.mode)


def run_evaluate_trajectories(args):
    import sys

    from loguru import logger

    from tau2.scripts.evaluate_trajectories import evaluate_trajectories

    logger.configure(handlers=[{"sink": sys.stderr, "level": "ERROR"}])

    evaluate_trajectories(args.paths, args.output_dir)


def run_prepare_submission(args):
    """Run the prepare submission command."""
    from tau2.scripts.leaderboard.prepare_submission import prepare_submission

    prepare_submission(
        input_paths=args.input_paths,
        output_dir=args.output,
        run_verification=not args.no_verify,
    )


def run_validate_submission(args):
    """Run the validate submission command."""
    from tau2.scripts.leaderboard.prepare_submission import validate_submission

    validate_submission(submission_dir=args.submission_dir, mode=args.mode)


def run_manual_mode():
    from tau2.scripts.manual_mode import main as manual_main

    manual_main()


def run_optimize(args):
    """Run DSPy prompt optimization."""
    from pathlib import Path

    from tau2.dspy.config import OptimizationConfig, StrategyType
    from tau2.dspy.optimize import optimize_agent_prompt

    # Convert strategy string to enum
    strategy = StrategyType(args.strategy)

    # Create optimization config
    # For GEPA, max_iterations maps to max_metric_calls
    opt_config = OptimizationConfig(
        max_iterations=args.max_iterations,
        max_metric_calls=args.max_iterations * 5,  # GEPA needs more metric calls
    )

    # Determine output path
    output_path = Path(args.output) if args.output else None

    # Run optimization
    result = optimize_agent_prompt(
        domain=args.domain,
        strategy=strategy,
        model=args.model,
        train_split=args.train_split,
        val_split=args.val_split,
        task_set_name=args.task_set_name,
        max_train_tasks=args.max_train_tasks,
        max_val_tasks=args.max_val_tasks,
        optimization_config=opt_config,
        output_path=output_path,
        debug=args.debug,
    )

    # Print results
    print(f"\nOptimization complete!")
    print(f"Domain: {result.domain}")
    print(f"Strategy: {result.strategy}")
    print(f"Best Score: {result.best_score:.4f}")
    print(f"Training examples: {result.train_size}")
    print(f"Validation examples: {result.val_size}")

    # Save if no output path was provided (use default)
    if output_path is None:
        saved_path = result.save()
        print(f"\nOptimized prompt saved to: {saved_path}")
    else:
        print(f"\nOptimized prompt saved to: {output_path}")


if __name__ == "__main__":
    main()

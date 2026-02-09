"""
Perplexity Ratio Script

Computes the ratio:
    PPL(pllm_prompt + data + query) / PPL(pllm_prompt + query)

For each task in a simulation file:
- Extract the user query and tool result data
- Call gpt-4.1 (Azure) directly with logprobs enabled
- Compute perplexity from logprobs
- Report per-task and aggregate ratios

Ratio < 1 means data reduces perplexity (model is more confident with data).
"""

import argparse
import json
import math
import os
import statistics
from pathlib import Path
from typing import Optional

from openai import AzureOpenAI


# --- Defaults ---

DEFAULT_PROMPT_FILE = "/home/ubuntu/tau2-bench/default_prompt.txt"
DEFAULT_MODEL = "gpt-4.1"
DEFAULT_MAX_TOKENS = 2048
DEFAULT_AZURE_ENDPOINT = "https://ilya-2751-resource.openai.azure.com"
DEFAULT_API_VERSION = "2025-03-01-preview"


# --- Azure Client ---

def get_azure_client() -> AzureOpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is required")
    endpoint = os.environ.get("AZURE_ENDPOINT", DEFAULT_AZURE_ENDPOINT)
    api_version = os.environ.get("AZURE_API_VERSION", DEFAULT_API_VERSION)
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    )


# --- Perplexity Computation ---

def generate_with_logprobs(
    client: AzureOpenAI,
    model: str,
    system_prompt: str,
    user_content: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """Call the model and return logprobs from the response."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=5,
    )
    return response


def extract_logprobs(response) -> list[float]:
    """Extract token log probabilities from a chat completion response."""
    choice = response.choices[0]
    if not choice.logprobs or not choice.logprobs.content:
        return []
    return [item.logprob for item in choice.logprobs.content]


def ppl_from_logprobs(logprobs: list[float]) -> float:
    """Compute perplexity: exp(-1/N * sum(log_probs))."""
    if not logprobs:
        return float("nan")
    return math.exp(-sum(logprobs) / len(logprobs))


# --- Simulation Data Extraction ---

def load_simulation(path: str | Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def extract_task_data(simulation: dict) -> list[dict]:
    """
    Extract per-task data from simulation results.

    Returns list of dicts with keys:
        task_id, user_query, tool_data
    """
    tasks_by_id = {}
    for task in simulation.get("tasks", []):
        tasks_by_id[str(task["id"])] = task

    results = []
    for sim in simulation.get("simulations", []):
        task_id = str(sim["task_id"])
        messages = sim.get("messages", [])

        # Extract first user query
        user_query = None
        for msg in messages:
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        if not user_query:
            continue

        # Extract all tool results as data
        tool_data_parts = []
        for msg in messages:
            if msg.get("role") == "tool" and msg.get("content"):
                tool_data_parts.append(msg["content"])

        # Skip tasks with no tool data (ratio undefined)
        if not tool_data_parts:
            continue

        tool_data = "\n\n".join(tool_data_parts)

        results.append({
            "task_id": task_id,
            "user_query": user_query,
            "tool_data": tool_data,
        })

    return results


# --- Main Logic ---

def compute_perplexity_ratio(
    client: AzureOpenAI,
    model: str,
    pllm_prompt: str,
    user_query: str,
    tool_data: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> dict:
    """
    Compute perplexity ratio for a single task.

    Returns dict with ppl_without_data, ppl_with_data, ratio, token counts.
    """
    # Without data: pllm_prompt + query
    resp_without = generate_with_logprobs(
        client, model, pllm_prompt, user_query, max_tokens
    )
    lps_without = extract_logprobs(resp_without)
    ppl_without = ppl_from_logprobs(lps_without)

    # With data: pllm_prompt + "I have this {data}" + query
    user_content_with_data = f"I have this data:\n{tool_data}\n\n{user_query}"
    resp_with = generate_with_logprobs(
        client, model, pllm_prompt, user_content_with_data, max_tokens
    )
    lps_with = extract_logprobs(resp_with)
    ppl_with = ppl_from_logprobs(lps_with)

    ratio = ppl_with / ppl_without if ppl_without and not math.isnan(ppl_without) else float("nan")

    return {
        "ppl_without_data": ppl_without,
        "ppl_with_data": ppl_with,
        "ratio": ratio,
        "tokens_without": len(lps_without),
        "tokens_with": len(lps_with),
    }


def main():
    parser = argparse.ArgumentParser(description="Compute PLLM perplexity ratio across simulation tasks")
    parser.add_argument(
        "--simulation-file", required=True,
        help="Path to simulation results JSON file",
    )
    parser.add_argument(
        "--prompt-file", default=DEFAULT_PROMPT_FILE,
        help=f"Path to PLLM default prompt file (default: {DEFAULT_PROMPT_FILE})",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=None,
        help="Number of tasks to process (default: all)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max tokens for generation (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional output JSON path for results",
    )
    args = parser.parse_args()

    # Load PLLM prompt
    prompt_path = Path(args.prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    pllm_prompt = prompt_path.read_text().strip()
    # Strip wrapping triple-quote markers if present
    if pllm_prompt.startswith('"""') and pllm_prompt.endswith('"""'):
        pllm_prompt = pllm_prompt[3:-3].strip()

    # Load simulation
    sim_data = load_simulation(args.simulation_file)
    task_data = extract_task_data(sim_data)

    if args.num_tasks:
        task_data = task_data[:args.num_tasks]

    print(f"Loaded {len(task_data)} tasks with tool data from simulation")
    print(f"Model: {args.model}")
    print(f"PLLM prompt length: {len(pllm_prompt)} chars")
    print()

    # Create Azure client
    client = get_azure_client()

    # Process tasks
    results = []
    ratios = []

    print(f"{'Task ID':<40} {'PPL(no data)':>14} {'PPL(w/ data)':>14} {'Ratio':>10} {'Tokens(no/w)':>14}")
    print("-" * 96)

    for i, task in enumerate(task_data):
        task_id = task["task_id"]
        try:
            result = compute_perplexity_ratio(
                client=client,
                model=args.model,
                pllm_prompt=pllm_prompt,
                user_query=task["user_query"],
                tool_data=task["tool_data"],
                max_tokens=args.max_tokens,
            )

            results.append({
                "task_id": task_id,
                "user_query": task["user_query"],
                "tool_data": task["tool_data"],
                **result,
            })

            if not math.isnan(result["ratio"]):
                ratios.append(result["ratio"])

            print(
                f"{task_id:<40} "
                f"{result['ppl_without_data']:>14.4f} "
                f"{result['ppl_with_data']:>14.4f} "
                f"{result['ratio']:>10.4f} "
                f"{result['tokens_without']:>6}/{result['tokens_with']:<6}"
            )

        except Exception as e:
            print(f"{task_id:<40} ERROR: {e}")
            results.append({
                "task_id": task_id,
                "user_query": task["user_query"],
                "tool_data": task["tool_data"],
                "error": str(e),
            })

    # Aggregate stats
    print()
    print("=" * 96)
    if ratios:
        print(f"Tasks processed:  {len(results)}")
        print(f"Valid ratios:     {len(ratios)}")
        print(f"Mean ratio:       {statistics.mean(ratios):.4f}")
        print(f"Median ratio:     {statistics.median(ratios):.4f}")
        print(f"Std dev:          {statistics.stdev(ratios):.4f}" if len(ratios) > 1 else "")
        print(f"Min ratio:        {min(ratios):.4f}")
        print(f"Max ratio:        {max(ratios):.4f}")
    else:
        print("No valid results.")

    # Save results
    if args.output:
        output_data = {
            "model": args.model,
            "simulation_file": str(args.simulation_file),
            "prompt_file": str(args.prompt_file),
            "num_tasks_processed": len(results),
            "aggregate": {
                "mean_ratio": statistics.mean(ratios) if ratios else None,
                "median_ratio": statistics.median(ratios) if ratios else None,
                "std_ratio": statistics.stdev(ratios) if len(ratios) > 1 else None,
                "min_ratio": min(ratios) if ratios else None,
                "max_ratio": max(ratios) if ratios else None,
            },
            "per_task": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()

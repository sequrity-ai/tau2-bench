#!/usr/bin/env python3
"""
Tau2 Task Analysis Script

Combines tau2 simulation data with secure-orchestrator logs to extract
the most important information for understanding task success/failure.

Produces per-task markdown files suitable for human review and LLM analysis.

Usage:
    python tools/analyze-tau2-tasks.py <simulation_file> [simulation_file2 ...] \
        --logs-dir ../secure-orchestrator/logs \
        --output-dir ./task_analysis
"""

import argparse
import ast
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# =============================================================================
# Log parsing (adapted from secure-orchestrator/tools/analyze-logs.py)
# =============================================================================


def parse_log_line(line: str) -> dict | None:
    """Parse a single log line (Python dict or JSON)."""
    line = line.strip()
    if not line:
        return None
    try:
        return ast.literal_eval(line)
    except (ValueError, SyntaxError):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            return None


def has_tag(entry: dict, tag: str) -> bool:
    return tag in entry.get("tags", [])


def extract_pllm_programs(entries: list[dict]) -> list[dict]:
    """Extract all PLLM programs from log entries."""
    programs = []
    for entry in entries:
        if has_tag(entry, "pllm_program"):
            msg = entry.get("message", {})
            program = msg.get("program")
            if program:
                programs.append({
                    "program": program,
                    "timestamp": entry.get("timestamp"),
                })
    return programs


def extract_rllm_reviews(entries: list[dict]) -> list[dict]:
    """Extract RLLM reviews from log entries."""
    reviews = []
    for entry in entries:
        if has_tag(entry, "rllm_review"):
            msg = entry.get("message", {})
            review = msg.get("review")
            if review:
                reviews.append({
                    "review": review,
                    "timestamp": entry.get("timestamp"),
                })
    return reviews


def extract_interpreter_errors(entries: list[dict]) -> list[dict]:
    """Extract interpreter errors from log entries."""
    errors = []
    for entry in entries:
        if has_tag(entry, "interpreter.evaluate()"):
            msg = entry.get("message", {})
            if msg.get("error"):
                errors.append({
                    "error": msg.get("error"),
                    "exception_type": msg.get("exception_type"),
                    "details": msg.get("details"),
                    "timestamp": entry.get("timestamp"),
                })
    return errors


def extract_final_return_values(entries: list[dict]) -> list[dict]:
    """Extract final return values from log entries."""
    values = []
    for entry in entries:
        if has_tag(entry, "get_final_return_value"):
            msg = entry.get("message", {})
            val = msg.get("final_return_value")
            if val is not None:
                values.append({
                    "value": val,
                    "timestamp": entry.get("timestamp"),
                })
    return values


def parse_log_file(log_path: Path) -> list[dict]:
    """Parse all entries from a log file."""
    entries = []
    with open(log_path, "r") as f:
        for line in f:
            entry = parse_log_line(line)
            if entry:
                entries.append(entry)
    return entries


# =============================================================================
# Simulation parsing
# =============================================================================


def extract_session_ids_from_simulation(sim: dict) -> set[str]:
    """Extract unique session IDs from tool call IDs in simulation messages.

    Tool call ID format: tc-{SESSION_UUID}-{CALL_UUID}
    The session UUID is a standard UUID (8-4-4-4-12 hex chars).
    """
    session_ids = set()
    uuid_pattern = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"

    for msg in sim.get("messages", []):
        # Check tool_calls in assistant messages
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            tc_id = tc.get("id", "")
            if tc_id.startswith("tc-"):
                # Extract the first UUID after tc-
                uuids = re.findall(uuid_pattern, tc_id)
                if uuids:
                    session_ids.add(uuids[0])

        # Check tool response message IDs
        if msg.get("role") == "tool":
            msg_id = msg.get("id", "")
            if msg_id.startswith("tc-"):
                uuids = re.findall(uuid_pattern, msg_id)
                if uuids:
                    session_ids.add(uuids[0])

    return session_ids


def find_log_files(session_ids: set[str], logs_dir: Path) -> dict[str, Path]:
    """Find log files matching session IDs in the logs directory tree."""
    found = {}
    if not logs_dir.exists():
        return found

    for log_file in logs_dir.rglob("*.log"):
        stem = log_file.stem
        if stem in session_ids:
            found[stem] = log_file

    return found


def collect_orchestrator_artifacts(log_data: dict[str, dict] | None):
    """Flatten and sort orchestrator artifacts across all sessions."""
    programs = []
    return_values = []
    reviews = []
    errors = []

    if log_data:
        for _sid, data in sorted(log_data.items()):
            programs.extend(data.get("programs", []))
            return_values.extend(data.get("return_values", []))
            reviews.extend(data.get("reviews", []))
            errors.extend(data.get("errors", []))

    programs.sort(key=lambda x: x.get("timestamp", ""))
    return_values.sort(key=lambda x: x.get("timestamp", ""))
    reviews.sort(key=lambda x: x.get("timestamp", ""))
    errors.sort(key=lambda x: x.get("timestamp", ""))

    return programs, return_values, reviews, errors


def format_conversation(messages: list[dict], log_data: dict[str, dict] | None = None) -> str:
    """Format conversation messages with orchestrator data interleaved chronologically.

    Each USER message triggers one PLLM program execution. The program produces
    tool calls (shown as ASSISTANT tool_call messages) and a final return value
    (shown as the ASSISTANT text message). This function places the PLLM program
    right after the USER message that triggered it, and annotates the return value
    when it carries interesting metadata (e.g. tags).
    """
    programs, return_values, reviews, errors = collect_orchestrator_artifacts(log_data)

    lines = []
    prog_idx = 0
    rv_idx = 0

    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        tool_calls = msg.get("tool_calls") or []

        if role == "SYSTEM":
            continue

        if role == "USER" and content:
            lines.append(f"**USER**: {content}")
            lines.append("")

            # Inject PLLM program(s) for this turn
            if prog_idx < len(programs):
                prog = programs[prog_idx]
                lines.append("---")
                lines.append("")
                lines.append(
                    f"**PLLM Program {prog_idx + 1}** ({prog.get('timestamp', '')}):"
                )
                lines.append("")
                program_text = prog["program"]
                program_text = re.sub(r"^```(?:python)?\s*\n?", "", program_text)
                program_text = re.sub(r"\n?```\s*$", "", program_text)
                lines.append(f"```python\n{program_text}\n```")
                lines.append("")

                # Show RLLM review for this program if available
                if prog_idx < len(reviews):
                    rev = reviews[prog_idx]
                    review_data = rev.get("review", {})
                    if isinstance(review_data, dict):
                        approved = review_data.get("approved", review_data.get("result"))
                        summary = review_data.get("summary", "")
                        if approved is not None:
                            lines.append(
                                f"*RLLM Review: approved={approved}"
                                + (f" — {summary}" if summary else "")
                                + "*"
                            )
                        else:
                            lines.append(
                                f"*RLLM Review: `{json.dumps(review_data)}`*"
                            )
                    else:
                        lines.append(f"*RLLM Review: {review_data}*")
                    lines.append("")

                # Show interpreter errors for this turn if available
                if prog_idx < len(errors):
                    err = errors[prog_idx]
                    lines.append(
                        f"*Interpreter Error: [{err.get('exception_type', 'Error')}] "
                        f"{err.get('error', 'unknown')}*"
                    )
                    lines.append("")

                lines.append("---")
                lines.append("")
                prog_idx += 1

        elif role == "ASSISTANT":
            if content:
                lines.append(f"**ASSISTANT**: {content}")
                lines.append("")
                # Text-only assistant message = final return value for this turn.
                # Only consume a return value if we've already seen a PLLM program
                # (prog_idx > rv_idx), since pre-program messages like greetings
                # don't have associated return values.
                if not tool_calls and rv_idx < len(return_values) and rv_idx < prog_idx:
                    rv = return_values[rv_idx]
                    val = rv.get("value", "")
                    meta = val.get("meta", {}) if isinstance(val, dict) else None
                    if meta:
                        tags = meta.get("tags", [])
                        if tags:
                            lines.append(f"*Return value tags: {tags}*")
                            lines.append("")
                    rv_idx += 1
            for tc in tool_calls:
                name = tc.get("name", "unknown")
                args = tc.get("arguments", {})
                args_str = json.dumps(args, indent=2) if isinstance(args, dict) else str(args)
                lines.append(f"**ASSISTANT** [tool_call: `{name}`]:")
                lines.append(f"```json\n{args_str}\n```")
                lines.append("")

        elif role == "TOOL":
            content_str = str(content)
            # Truncate very long tool responses
            if len(content_str) > 2000:
                content_str = content_str[:2000] + "\n... (truncated)"
            lines.append(f"**TOOL** (`{msg.get('id', 'unknown')}`):")
            lines.append(f"```\n{content_str}\n```")
            lines.append("")

    # Show any remaining unmatched programs (e.g. from retries)
    while prog_idx < len(programs):
        prog = programs[prog_idx]
        lines.append("---")
        lines.append("")
        lines.append(
            f"**PLLM Program {prog_idx + 1}** ({prog.get('timestamp', '')}):"
        )
        program_text = prog["program"]
        program_text = re.sub(r"^```(?:python)?\s*\n?", "", program_text)
        program_text = re.sub(r"\n?```\s*$", "", program_text)
        lines.append(f"```python\n{program_text}\n```")
        lines.append("")
        prog_idx += 1

    return "\n".join(lines)


def extract_actual_tool_calls(messages: list[dict]) -> list[dict]:
    """Extract all actual tool calls made by the assistant from simulation messages."""
    actual = []
    for msg in messages:
        if msg.get("role") == "assistant":
            for tc in (msg.get("tool_calls") or []):
                if tc.get("requestor") == "assistant":
                    actual.append({
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}),
                    })
    return actual


def format_action_checks(action_checks: list[dict], actual_calls: list[dict] | None = None) -> str:
    """Format action checks with expected vs actual comparison for failures."""
    if not action_checks:
        return "No action checks."

    # Build lookup: for each expected action name, find matching actual calls
    actual_by_name: dict[str, list[dict]] = {}
    if actual_calls:
        for ac in actual_calls:
            actual_by_name.setdefault(ac["name"], []).append(ac)

    # Track which actual calls we've matched
    used_actual: dict[str, int] = {}  # name -> next index to use

    lines = []
    for ac in action_checks:
        action = ac.get("action", {})
        match = ac.get("action_match", False)
        reward = ac.get("action_reward", 0.0)
        status = "PASS" if match else "FAIL"

        name = action.get("name", "unknown")
        args = action.get("arguments", {})
        args_summary = json.dumps(args, separators=(",", ":"))
        if len(args_summary) > 200:
            args_summary = args_summary[:200] + "..."

        lines.append(f"- [{status}] `{name}` (reward: {reward})")
        lines.append(f"  Expected args: `{args_summary}`")

        # For failed actions, show what was actually called (if anything)
        if not match and actual_calls:
            idx = used_actual.get(name, 0)
            candidates = actual_by_name.get(name, [])
            if idx < len(candidates):
                actual_args = candidates[idx]
                actual_summary = json.dumps(actual_args["arguments"], separators=(",", ":"))
                if len(actual_summary) > 200:
                    actual_summary = actual_summary[:200] + "..."
                lines.append(f"  **Actual args**: `{actual_summary}`")
                # Highlight specific differences
                if isinstance(args, dict) and isinstance(actual_args["arguments"], dict):
                    diffs = []
                    for key in set(list(args.keys()) + list(actual_args["arguments"].keys())):
                        expected_val = args.get(key)
                        actual_val = actual_args["arguments"].get(key)
                        if expected_val != actual_val:
                            diffs.append(f"`{key}`: expected `{expected_val}` vs actual `{actual_val}`")
                    if diffs:
                        lines.append(f"  **Differences**: {'; '.join(diffs)}")
                used_actual[name] = idx + 1
            else:
                lines.append(f"  **Actual**: No matching `{name}` call found")

    return "\n".join(lines)


def format_communicate_checks(communicate_checks: list[dict]) -> str:
    """Format communicate checks."""
    if not communicate_checks:
        return "No communication checks."

    lines = []
    for cc in communicate_checks:
        info = cc.get("info", "")
        met = cc.get("met", False)
        justification = cc.get("justification", "")
        status = "PASS" if met else "FAIL"
        lines.append(f"- [{status}] Info: `{info}`")
        if justification:
            lines.append(f"  Justification: {justification}")

    return "\n".join(lines)


# =============================================================================
# Report generation
# =============================================================================


def generate_task_report(
    task: dict,
    sim: dict,
    log_data: dict[str, dict],
) -> str:
    """Generate a markdown report for a single task/simulation pair."""

    task_id = task.get("id", "unknown")
    scenario = task.get("user_scenario", {})
    instructions = scenario.get("instructions", {})
    eval_criteria = task.get("evaluation_criteria", {})
    reward_info = sim.get("reward_info", {})

    sections = []

    # Header
    reward = reward_info.get("reward", "N/A")
    duration = sim.get("duration", "N/A")
    if isinstance(duration, float):
        duration = f"{duration:.1f}s"
    termination = sim.get("termination_reason", "unknown")

    sections.append(f"# Task {task_id} Analysis")
    sections.append("")
    sections.append(f"- **Reward**: {reward}")
    sections.append(f"- **Duration**: {duration}")
    sections.append(f"- **Termination**: {termination}")
    sections.append(f"- **Simulation ID**: {sim.get('id', 'N/A')}")
    sections.append(f"- **Timestamp**: {sim.get('timestamp', 'N/A')}")
    sections.append("")

    # User Scenario
    sections.append("## User Scenario")
    sections.append("")
    if instructions:
        sections.append(f"- **Domain**: {instructions.get('domain', 'N/A')}")
        sections.append(f"- **Reason for call**: {instructions.get('reason_for_call', 'N/A')}")
        sections.append(f"- **Known info**: {instructions.get('known_info', 'N/A')}")
        sections.append(f"- **Unknown info**: {instructions.get('unknown_info', 'N/A')}")
        sections.append(f"- **Task instructions**: {instructions.get('task_instructions', 'N/A')}")
    sections.append("")

    # Evaluation Criteria - Expected Actions
    sections.append("## Expected Actions")
    sections.append("")
    expected_actions = eval_criteria.get("actions", [])
    for ea in expected_actions:
        name = ea.get("name", "unknown")
        args = ea.get("arguments", {})
        sections.append(f"- `{name}`")
        sections.append(f"  ```json\n  {json.dumps(args, indent=2)}\n  ```")
    sections.append("")

    # Evaluation Results
    sections.append("## Evaluation Results")
    sections.append("")

    # Reward breakdown
    breakdown = reward_info.get("reward_breakdown", {})
    if breakdown:
        sections.append("### Reward Breakdown")
        for key, val in breakdown.items():
            sections.append(f"- **{key}**: {val}")
        sections.append("")

    # Action checks — include actual calls for comparison
    actual_calls = extract_actual_tool_calls(sim.get("messages", []))
    sections.append("### Action Checks")
    sections.append("")
    sections.append(format_action_checks(reward_info.get("action_checks") or [], actual_calls))
    sections.append("")

    # Communicate checks
    comm_checks = reward_info.get("communicate_checks") or []
    if comm_checks:
        sections.append("### Communication Checks")
        sections.append("")
        sections.append(format_communicate_checks(comm_checks))
        sections.append("")

    # DB check
    db_check = reward_info.get("db_check", {})
    if db_check:
        sections.append("### Database Check")
        sections.append(f"- **DB Match**: {db_check.get('db_match', 'N/A')}")
        sections.append(f"- **DB Reward**: {db_check.get('db_reward', 'N/A')}")
        sections.append("")

    # Conversation (with interleaved orchestrator data)
    sections.append("## Conversation")
    sections.append("")
    if log_data:
        session_ids = sorted(log_data.keys())
        sections.append(
            f"*Orchestrator session(s): {', '.join(f'`{s}`' for s in session_ids)}*"
        )
        sections.append("")
    else:
        sections.append("*No matching orchestrator log files found for this task.*")
        sections.append("")

    messages = sim.get("messages", [])
    sections.append(format_conversation(messages, log_data))
    sections.append("")

    return "\n".join(sections)


def generate_summary_report(
    sim_file: Path,
    info: dict,
    tasks: list[dict],
    simulations: list[dict],
) -> str:
    """Generate a summary report across all tasks."""
    lines = []
    lines.append("# Tau2 Task Analysis Summary")
    lines.append("")
    lines.append(f"- **Simulation file**: `{sim_file.name}`")
    lines.append(f"- **Timestamp**: {info.get('timestamp', 'N/A')}")

    agent_info = info.get("agent_info", {})
    user_info = info.get("user_info", {})
    lines.append(f"- **Agent LLM**: {agent_info.get('llm', 'N/A')}")
    lines.append(f"- **User LLM**: {user_info.get('llm', 'N/A')}")
    lines.append(f"- **Domain**: {info.get('environment_info', {}).get('domain_name', 'N/A')}")
    lines.append(f"- **Total tasks**: {len(tasks)}")
    lines.append(f"- **Total simulations**: {len(simulations)}")
    lines.append("")

    # Build task_id -> simulation mapping
    sim_by_task = {}
    for sim in simulations:
        tid = sim.get("task_id", "unknown")
        sim_by_task.setdefault(tid, []).append(sim)

    # Results table
    lines.append("## Results Overview")
    lines.append("")
    lines.append("| Task ID | Reward | Termination | Duration | Action Checks (Pass/Total) |")
    lines.append("|---------|--------|-------------|----------|---------------------------|")

    total_reward = 0.0
    task_count = 0

    for task in tasks:
        tid = task.get("id", "unknown")
        sims = sim_by_task.get(tid, [])
        for sim in sims:
            ri = sim.get("reward_info", {})
            reward = ri.get("reward", 0.0)
            total_reward += reward
            task_count += 1
            termination = sim.get("termination_reason", "N/A")
            duration = sim.get("duration", 0)
            if isinstance(duration, float):
                duration = f"{duration:.1f}s"

            action_checks = ri.get("action_checks") or []
            passed = sum(1 for ac in action_checks if ac.get("action_match"))
            total = len(action_checks)

            lines.append(f"| {tid} | {reward} | {termination} | {duration} | {passed}/{total} |")

    lines.append("")
    if task_count > 0:
        avg_reward = total_reward / task_count
        lines.append(f"**Average Reward**: {avg_reward:.3f}")
        lines.append(f"**Total Tasks Evaluated**: {task_count}")
    lines.append("")

    # Failed tasks summary
    lines.append("## Failed Tasks")
    lines.append("")
    for task in tasks:
        tid = task.get("id", "unknown")
        sims = sim_by_task.get(tid, [])
        for sim in sims:
            ri = sim.get("reward_info", {})
            reward = ri.get("reward", 0.0)
            if reward < 1.0:
                action_checks = ri.get("action_checks") or []
                failed_actions = [
                    ac for ac in action_checks if not ac.get("action_match")
                ]
                comm_checks = ri.get("communicate_checks") or []
                failed_comms = [cc for cc in comm_checks if not cc.get("met")]

                lines.append(f"### Task {tid} (reward: {reward})")
                scenario = task.get("user_scenario", {}).get("instructions", {})
                lines.append(f"- **Reason for call**: {scenario.get('reason_for_call', 'N/A')}")

                if failed_actions:
                    lines.append(f"- **Failed action checks** ({len(failed_actions)}):")
                    for fa in failed_actions:
                        action = fa.get("action", {})
                        lines.append(f"  - `{action.get('name', 'unknown')}`")

                if failed_comms:
                    lines.append(f"- **Failed communication checks** ({len(failed_comms)}):")
                    for fc in failed_comms:
                        lines.append(f"  - Info: `{fc.get('info', '')}` — {fc.get('justification', '')}")

                lines.append("")

    return "\n".join(lines)


# =============================================================================
# LLM Analysis
# =============================================================================


FAILED_TASK_SYSTEM_PROMPT = """\
You are an expert at analyzing AI agent task executions in a customer service benchmark.

The system under test uses a "dual-LLM" architecture:
- PLLM (Planning LLM) generates Python programs to handle each user turn
- Programs can call tool functions (find_user, get_order_details, \
exchange_delivered_order_items, etc.)
- Each program produces a final_return_value that becomes the assistant's text response
- RLLM (Review LLM) reviews programs for policy compliance (may not be present)

Analyze this FAILED task execution. Your analysis should:
1. Identify the exact point of failure — which action check or communication check \
failed and why
2. Determine the root cause: Was it an agent/PLLM code bug, a user simulator \
deviation from the scenario, an ambiguous evaluation criteria, or something else?
3. Reference specific evidence: quote relevant parts of the conversation, PLLM \
programs, tool call arguments, and expected vs actual values
4. Explain what the correct behavior would have been
5. **Suggested PLLM fix**: Show the specific code changes to the PLLM program(s) \
that would have made this task succeed. Write the corrected Python snippet(s) \
with comments explaining what changed and why. Focus only on the parts that need \
to change — do not rewrite entire programs, just the critical lines/blocks.

Be precise and specific. Reference item IDs, function names, and argument values."""

SUCCESS_TASK_SYSTEM_PROMPT = """\
You are an expert at analyzing AI agent task executions in a customer service benchmark.

The system under test uses a "dual-LLM" architecture:
- PLLM (Planning LLM) generates Python programs to handle each user turn
- Programs can call tool functions (find_user, get_order_details, \
exchange_delivered_order_items, etc.)
- Each program produces a final_return_value that becomes the assistant's text response
- RLLM (Review LLM) reviews programs for policy compliance (may not be present)

Analyze this SUCCESSFUL task execution in detail. Your analysis should:
1. Describe the agent's overall strategy for solving the task
2. Analyze each PLLM program: what approach did it take, how did it handle the \
user's request, what tool calls did it make and why
3. Highlight key decisions that led to success (e.g. correct item selection, proper \
error handling, good information gathering)
4. Note the conversation flow: how did the agent guide the user, verify information, \
and confirm actions
5. Identify any areas where the agent was particularly effective or could have been \
more efficient
6. **Patterns to preserve**: Summarize the key PLLM coding patterns that made this \
task succeed — these are patterns the PLLM should continue using in future tasks.

Be thorough and reference specific details from the PLLM programs and conversation."""


META_ANALYSIS_PREAMBLE = """\
You are an expert at improving AI code-generation systems.

You are reviewing a batch of benchmark results for a "dual-LLM" orchestrator:
- **PLLM (Planning LLM)** receives a user message + conversation history and
  generates a **Python program** that calls tool functions and sets
  `final_return_value` to reply to the user. It produces a fresh program for
  each user turn, but can see the conversation history and previous tool results.
- **RLLM (Review LLM)** reviews PLLM programs for policy compliance.

The PLLM system prompt already contains:
- A mission statement (single-step or multi-step code gen)
- Core principles: Resourcefulness, Decisiveness, Error Correction, Adaptability,
  Final Return Value rules
- Code generation rules: sequrity_plan tags, allowed modules, blocked functions
- Tool usage notes: keyword arguments, parse_with_ai, verify_hypothesis

You will receive the full summary of a benchmark run including per-task LLM
analyses of both failed and successful tasks.
"""

META_ANALYSIS_FRESH_INSTRUCTIONS = """\
Produce two clearly separated sections:

## Section 1: General Improvement Suggestions

Numbered list of actionable suggestions. For each: state the recurring pattern,
why it fails, the fix, and cite task IDs. Focus on patterns affecting 3+ tasks.

## Section 2: pllm_custom_instructions

A text block appended directly to the PLLM system prompt. Written as direct
instructions to the PLLM ("Always ...", "Never ...").

Rules:
- Max ~500 words, 8-12 bullet points. More bullets = worse performance.
  Prioritize by impact (most failures first). Omit patterns affecting <3 tasks.
- Be operationally specific — each bullet must translate to a code pattern.
  Short inline snippets are OK; multi-line code blocks and function definitions
  are NOT.
- Write general patterns, not task-specific implementations. Do not hard-code
  item IDs, product names, prices, or specific attribute names from failed tasks.
  The instructions must generalize to unseen tasks.
- Do not repeat guidance already in the base PLLM prompt. Do not add boilerplate
  (try/except, variable naming, code style).
- Use bullet points. No markdown code fences. Start with: # Additional Instructions
"""

META_ANALYSIS_ITERATIVE_INSTRUCTIONS = """\
This is an **iterative refinement**. The benchmark was run WITH the previous
pllm_custom_instructions already injected. Evaluate what worked, what didn't,
and produce an updated version.

Produce three clearly separated sections:

## Section 1: What changed since the previous prompt

For each bullet in the previous instructions: did it improve, stay the same, or
regress? Cite task IDs. Note any new failures the previous instructions may have
introduced.

## Section 2: General Improvement Suggestions

Numbered list of remaining or new issues. State pattern, why it fails, the fix,
cite task IDs. Skip issues that are now fixed.

## Section 3: pllm_custom_instructions

The updated text block (replaces previous version entirely).

Process:
1. Start from the previous instructions. Keep every bullet that is not linked
   to a failure — those bullets are likely why tasks passed.
2. Modify only bullets with direct evidence of causing a failure.
3. Add new bullets for new failure patterns from Section 2.
4. Do NOT flip or reverse any ordering or logic from the previous version
   unless you have evidence the rule itself caused a failure.
5. Target 8-12 bullets. Drop lowest-impact ones if over 12.

Rules (same as fresh):
- Max ~500 words. More bullets = worse performance. Prioritize by impact.
- Operationally specific — each bullet must translate to a code pattern.
  Short inline snippets OK; no multi-line code blocks or function definitions.
- General patterns only. No hard-coded task values (IDs, prices, attribute names).
- No boilerplate. No markdown code fences. Start with:
  # Additional Instructions (v{iteration})
"""


def _parse_folder_timestamp(folder_name: str) -> str | None:
    """Extract ISO timestamp prefix from a task_analysis folder name.

    Folder names look like: 2026-02-10T14:29:29.638112_retail_llm_agent_...
    Returns the timestamp string, or None if not parseable.
    """
    match = re.match(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+)", folder_name)
    return match.group(1) if match else None


def find_previous_instructions(output_dir: Path, current_folder: str) -> tuple[str | None, int]:
    """Find the most recent _pllm_custom_instructions.txt that predates current_folder.

    Scans all sibling directories in output_dir for instruction files, picks the
    one with the latest timestamp that is strictly before the current folder's
    timestamp.

    Returns:
        (instructions_text, iteration_number) where iteration_number is how many
        previous instruction files exist (used for versioning).
    """
    current_ts = _parse_folder_timestamp(current_folder)

    candidates = []
    for subdir in output_dir.iterdir():
        if not subdir.is_dir() or subdir.name == current_folder:
            continue
        instr_file = subdir / "_pllm_custom_instructions.txt"
        if not instr_file.exists():
            continue
        ts = _parse_folder_timestamp(subdir.name)
        if ts is not None:
            candidates.append((ts, instr_file))

    if not candidates:
        return None, 1

    # Sort by timestamp descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    # If we can parse the current timestamp, pick the latest one before it
    if current_ts is not None:
        for ts, path in candidates:
            if ts < current_ts:
                text = path.read_text().strip()
                # iteration = number of prior instruction files + 1
                iteration = len(candidates) + 1
                return text, iteration

    # Fallback: just use the most recent one
    text = candidates[0][1].read_text().strip()
    return text, len(candidates) + 1


def load_env_file(env_path: Path) -> dict:
    """Load key=value pairs from a .env file."""
    env_vars = {}
    if not env_path.exists():
        return env_vars
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value
    return env_vars


def create_openai_client(env_vars: dict):
    """Create an OpenAI client from environment variables."""
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: 'openai' package is required for --analyze.", file=sys.stderr)
        print("  Install it with: pip install openai", file=sys.stderr)
        sys.exit(1)

    api_key = env_vars.get("X_Api_Key") or env_vars.get("OPENAI_API_KEY")
    base_url = env_vars.get("ENDPOINT_ADDRESS_FULL")

    if not api_key:
        raise ValueError("No API key found (X_Api_Key or OPENAI_API_KEY) in env file")
    if not base_url:
        raise ValueError("No ENDPOINT_ADDRESS_FULL found in env file")

    return OpenAI(api_key=api_key, base_url=base_url)


def extract_reward_from_report(content: str) -> float:
    """Extract the reward value from a task report's header."""
    match = re.search(r"\*\*Reward\*\*:\s*([0-9.]+)", content)
    if match:
        return float(match.group(1))
    return 0.0


def analyze_single_task(
    client,
    model: str,
    task_content: str,
    is_success: bool,
) -> str:
    """Send a task report to the LLM for analysis.

    Retries once if the response is empty (can happen with reasoning models
    that spend all tokens on chain-of-thought).
    """
    system_prompt = (
        SUCCESS_TASK_SYSTEM_PROMPT if is_success else FAILED_TASK_SYSTEM_PROMPT
    )
    max_completion_tokens = 16384

    for attempt in range(2):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Here is the task execution report:\n\n" + task_content
                    ),
                },
            ],
            max_completion_tokens=max_completion_tokens,
        )
        content = response.choices[0].message.content
        if content and content.strip():
            return content

        # Empty response — bump token budget and retry
        finish = response.choices[0].finish_reason
        print(f"    Empty response (finish_reason={finish}), retrying with more tokens...")
        max_completion_tokens = 32768

    return "*Analysis returned empty response after 2 attempts.*"


def _analyze_one(args_tuple):
    """Worker function for parallel LLM analysis."""
    client, model, task_file, index, total = args_tuple
    task_content = task_file.read_text()
    reward = extract_reward_from_report(task_content)
    is_success = reward >= 1.0
    status = "SUCCESS" if is_success else "FAILED"
    task_name = task_file.stem

    print(f"  [{index}/{total}] Analyzing {task_name} ({status}, reward={reward})...")

    try:
        analysis = analyze_single_task(client, model, task_content, is_success)
    except Exception as e:
        print(f"    Error analyzing {task_name}: {e}")
        analysis = f"*Analysis failed: {e}*"

    return {
        "task_name": task_name,
        "reward": reward,
        "is_success": is_success,
        "analysis": analysis,
    }


def run_llm_analysis(
    output_dir: Path,
    model: str,
    client,
    max_workers: int = 5,
) -> None:
    """Run LLM analysis on all task reports in parallel and append to summary."""

    # Find all task report files (not the summary)
    task_files = sorted(
        [f for f in output_dir.glob("task_*.md")],
        key=lambda f: (
            int(m.group(1)) if (m := re.search(r"task_(\d+)", f.stem)) else 0
        ),
    )

    if not task_files:
        print("  No task reports found for LLM analysis.")
        return

    summary_path = output_dir / "_summary.md"

    # Check if analysis already exists in summary
    if summary_path.exists():
        existing = summary_path.read_text()
        if "## LLM Analysis" in existing:
            print(
                "  Warning: Summary already contains LLM analysis. "
                "Delete _summary.md and re-run to regenerate."
            )
            return

    total = len(task_files)
    print(f"  Analyzing {total} tasks with {max_workers} parallel workers...")

    work_items = [
        (client, model, tf, i + 1, total)
        for i, tf in enumerate(task_files)
    ]

    analyses = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_analyze_one, item): item for item in work_items}
        for future in as_completed(futures):
            analyses.append(future.result())

    # Sort back into task order
    analyses.sort(key=lambda a: (
        int(m.group(1)) if (m := re.search(r"task_(\d+)", a["task_name"])) else 0
    ))

    # Build analysis section
    lines = []
    lines.append("")
    lines.append(f"## LLM Analysis (Model: {model})")
    lines.append("")

    # Failed tasks first
    failed = [a for a in analyses if not a["is_success"]]
    succeeded = [a for a in analyses if a["is_success"]]

    if failed:
        lines.append("### Failed Task Analyses")
        lines.append("")
        for a in failed:
            lines.append(f"#### {a['task_name']} (reward: {a['reward']})")
            lines.append("")
            lines.append(a["analysis"])
            lines.append("")

    if succeeded:
        lines.append("### Successful Task Analyses")
        lines.append("")
        for a in succeeded:
            lines.append(f"#### {a['task_name']} (reward: {a['reward']})")
            lines.append("")
            lines.append(a["analysis"])
            lines.append("")

    # Append to summary
    with open(summary_path, "a") as f:
        f.write("\n".join(lines))

    print(
        f"  LLM analysis complete: {len(failed)} failed + {len(succeeded)} succeeded "
        f"analyses appended to {summary_path.name}"
    )


def run_meta_analysis(
    output_dir: Path,
    model: str,
    client,
    force: bool = False,
) -> None:
    """Run a meta-analysis on the full summary to produce general PLLM improvements.

    Automatically detects whether a previous _pllm_custom_instructions.txt exists
    from an earlier analysis cycle. If so, uses the iterative prompt that asks the
    LLM to evaluate what changed and produce an updated version.
    """

    summary_path = output_dir / "_summary.md"
    if not summary_path.exists():
        print("  No _summary.md found — skipping meta-analysis.")
        return

    summary_content = summary_path.read_text()
    if "## Meta-Analysis" in summary_content:
        if not force:
            print(
                "  Warning: Summary already contains meta-analysis. "
                "Use --force-meta to re-run, or delete the section manually."
            )
            return
        # Strip existing meta-analysis section from summary
        idx = summary_content.index("## Meta-Analysis")
        summary_content = summary_content[:idx].rstrip() + "\n"
        with open(summary_path, "w") as f:
            f.write(summary_content)
        print("  Stripped existing meta-analysis from summary.")

    if "## LLM Analysis" not in summary_content:
        print("  No LLM Analysis section found in summary — run --analyze first.")
        return

    # Check for previous instructions from an earlier analysis cycle
    current_folder = output_dir.name
    prev_instructions, iteration = find_previous_instructions(
        output_dir.parent, current_folder
    )

    if prev_instructions:
        print(f"  Found previous pllm_custom_instructions (iteration {iteration - 1})")
        print(f"  Using iterative refinement prompt (producing iteration {iteration})")
        system_prompt = (
            META_ANALYSIS_PREAMBLE
            + META_ANALYSIS_ITERATIVE_INSTRUCTIONS.replace("{iteration}", str(iteration))
        )
        user_content = (
            "## Benchmark summary with per-task analyses\n\n"
            + summary_content
            + "\n\n---\n\n"
            + "## Previous pllm_custom_instructions (used during this benchmark run)\n\n"
            + "THE INSTRUCTIONS BELOW PRODUCED THE BEST SCORE SO FAR. Your Section 3\n"
            + "MUST preserve every bullet below unless you have direct evidence it caused\n"
            + "a failure. Copy them first, then apply minimal edits.\n\n"
            + prev_instructions
        )
    else:
        print("  No previous instructions found — using fresh analysis prompt")
        system_prompt = META_ANALYSIS_PREAMBLE + META_ANALYSIS_FRESH_INSTRUCTIONS
        user_content = (
            "Here is the full benchmark summary with per-task analyses:\n\n"
            + summary_content
        )

    print(f"  Running meta-analysis on {summary_path.name}...")

    max_completion_tokens = 8192
    for attempt in range(2):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            max_completion_tokens=max_completion_tokens,
        )
        content = response.choices[0].message.content
        if content and content.strip():
            break
        finish = response.choices[0].finish_reason
        print(f"    Empty response (finish_reason={finish}), retrying with more tokens...")
        max_completion_tokens = 16384
    else:
        print("  Meta-analysis returned empty response after 2 attempts.")
        return

    meta_text = content.strip()

    # Extract pllm_custom_instructions — try Section 3 (iterative) then Section 2 (fresh)
    custom_instructions = None
    section_patterns = [
        r"##\s*Section 3.*?(?:\n\n|\n)([\s\S]+)$",
        r"##\s*Section 2.*?(?:\n\n|\n)([\s\S]+)$",
        r"##\s*pllm_custom_instructions.*?(?:\n\n|\n)([\s\S]+)$",
        r"(?:^|\n)#\s*Additional Instructions[^\n]*\n([\s\S]+)$",
    ]
    for pattern in section_patterns:
        match = re.search(pattern, meta_text, re.IGNORECASE)
        if match:
            candidate = match.group(1).strip()
            # Strip markdown code fences if present
            candidate = re.sub(r"^```(?:\w*)\s*\n?", "", candidate)
            candidate = re.sub(r"\n?```\s*$", "", candidate)
            custom_instructions = candidate.strip()
            break

    # Append meta-analysis to summary
    mode = f"iterative, iteration {iteration}" if prev_instructions else "fresh"
    meta_section = [
        "",
        f"## Meta-Analysis (Model: {model}, {mode})",
        "",
        meta_text,
        "",
    ]

    with open(summary_path, "a") as f:
        f.write("\n".join(meta_section))

    print(f"  Meta-analysis appended to {summary_path.name}")

    # Save pllm_custom_instructions to a separate file
    instructions_path = output_dir / "_pllm_custom_instructions.txt"
    if custom_instructions:
        with open(instructions_path, "w") as f:
            f.write(custom_instructions + "\n")
        print(f"  Saved pllm_custom_instructions to {instructions_path.name}")
        print(f"  ({len(custom_instructions)} chars, iteration {iteration})")
    else:
        print("  Warning: Could not extract pllm_custom_instructions section from response.")
        with open(instructions_path, "w") as f:
            f.write("# Could not auto-extract — full meta-analysis response below:\n\n")
            f.write(meta_text + "\n")
        print(f"  Saved full response to {instructions_path.name} for manual extraction.")


# =============================================================================
# Main
# =============================================================================


def process_simulation_file(
    sim_file: Path,
    logs_dir: Path,
    output_dir: Path,
) -> None:
    """Process a single simulation file and generate analysis reports."""

    print(f"Processing: {sim_file.name}")

    with open(sim_file, "r") as f:
        data = json.load(f)

    info = data.get("info", {})
    tasks = data.get("tasks", [])
    simulations = data.get("simulations", [])

    if not simulations:
        print(f"  No simulations found in {sim_file.name}, skipping.")
        return

    # Create output subdirectory named after the simulation file
    domain = info.get("environment_info", {}).get("domain_name", "unknown")
    sub_dir = output_dir / f"{sim_file.stem}"
    sub_dir.mkdir(parents=True, exist_ok=True)

    # Build task_id -> task mapping
    task_by_id = {t["id"]: t for t in tasks}

    # Build task_id -> simulation mapping
    sim_by_task = {}
    for sim in simulations:
        tid = sim.get("task_id", "unknown")
        sim_by_task.setdefault(tid, []).append(sim)

    # Collect all session IDs across all simulations for batch log lookup
    all_session_ids = set()
    sim_session_map = {}  # sim_id -> set of session_ids
    for sim in simulations:
        sid = sim.get("id", "")
        session_ids = extract_session_ids_from_simulation(sim)
        sim_session_map[sid] = session_ids
        all_session_ids.update(session_ids)

    print(f"  Found {len(simulations)} simulations, {len(all_session_ids)} unique sessions")

    # Find all matching log files
    log_file_map = find_log_files(all_session_ids, logs_dir)
    print(f"  Matched {len(log_file_map)}/{len(all_session_ids)} sessions to log files")

    # Parse log files and extract data
    log_data_cache = {}  # session_id -> extracted data
    for session_id, log_path in log_file_map.items():
        print(f"  Parsing log: {log_path.name}")
        entries = parse_log_file(log_path)
        log_data_cache[session_id] = {
            "programs": extract_pllm_programs(entries),
            "reviews": extract_rllm_reviews(entries),
            "errors": extract_interpreter_errors(entries),
            "return_values": extract_final_return_values(entries),
        }

    # Generate per-task reports
    for task in tasks:
        tid = task.get("id", "unknown")
        sims = sim_by_task.get(tid, [])

        for trial_idx, sim in enumerate(sims):
            # Gather log data for this simulation's sessions
            sim_id = sim.get("id", "")
            session_ids = sim_session_map.get(sim_id, set())
            task_log_data = {
                sid: log_data_cache[sid]
                for sid in session_ids
                if sid in log_data_cache
            }

            # Generate report
            report = generate_task_report(task, sim, task_log_data)

            # Save report
            if len(sims) > 1:
                filename = f"task_{tid}_trial_{trial_idx}.md"
            else:
                filename = f"task_{tid}.md"

            report_path = sub_dir / filename
            with open(report_path, "w") as f:
                f.write(report)

    # Generate summary report
    summary = generate_summary_report(sim_file, info, tasks, simulations)
    summary_path = sub_dir / "_summary.md"
    with open(summary_path, "w") as f:
        f.write(summary)

    print(f"  Output: {sub_dir}/")
    print(f"  Generated {len(tasks)} task reports + 1 summary")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze tau2 simulation tasks with secure-orchestrator logs",
    )
    parser.add_argument(
        "simulation_files",
        nargs="+",
        type=Path,
        help="One or more tau2 simulation JSON files",
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent.parent / "secure-orchestrator" / "logs",
        help="Path to secure-orchestrator logs directory (default: ../../secure-orchestrator/logs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "task_analysis",
        help="Output directory for analysis files (default: ./task_analysis)",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run per-task LLM analysis + meta-analysis (both phases)",
    )
    parser.add_argument(
        "--llm-only",
        action="store_true",
        help="Run only per-task LLM analysis (skip report generation and meta-analysis)",
    )
    parser.add_argument(
        "--meta-only",
        action="store_true",
        help="Run only the meta-analysis on existing _summary.md (skip reports and per-task LLM)",
    )
    parser.add_argument(
        "--force-meta",
        action="store_true",
        help="Strip existing meta-analysis from _summary.md before re-running (preserves LLM analysis)",
    )
    parser.add_argument(
        "--analysis-model",
        type=str,
        default="gpt-5-mini", #gpt-5.1
        help="Model to use for LLM analysis (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=15,
        help="Max parallel LLM requests for --analyze (default: 15)",
    )
    parser.add_argument(
        "--meta-model",
        type=str,
        default="gpt-5.1",
        help="Model for meta-analysis (default: same as --analysis-model). "
             "Use a stronger model like gpt-5.1 for better meta-analysis.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(__file__).resolve().parent.parent / ".env",
        help="Path to .env file with API credentials (default: tau2-bench/.env)",
    )

    args = parser.parse_args()

    # Determine which phases to run
    run_reports = not args.llm_only and not args.meta_only
    run_llm = args.analyze or args.llm_only
    run_meta = args.analyze or args.meta_only
    needs_client = run_llm or run_meta

    # Validate inputs
    for sf in args.simulation_files:
        if not sf.exists():
            print(f"Error: Simulation file not found: {sf}", file=sys.stderr)
            sys.exit(1)

    if run_reports and not args.logs_dir.exists():
        print(f"Warning: Logs directory not found: {args.logs_dir}", file=sys.stderr)
        print("  Will generate reports without orchestrator log data.", file=sys.stderr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Generate per-task reports and summary
    if run_reports:
        for sim_file in args.simulation_files:
            process_simulation_file(sim_file, args.logs_dir, args.output_dir)

    # Create API client if needed for LLM phases
    client = None
    if needs_client:
        env_vars = load_env_file(args.env_file)
        if not env_vars:
            print(f"Error: Could not load env file: {args.env_file}", file=sys.stderr)
            sys.exit(1)
        print(f"  Loaded credentials from: {args.env_file}")
        client = create_openai_client(env_vars)

    # Phase 2: Per-task LLM analysis
    if run_llm:
        print(f"\n--- LLM Analysis Phase (model: {args.analysis_model}) ---")
        for sim_file in args.simulation_files:
            sub_dir = args.output_dir / sim_file.stem
            if sub_dir.exists():
                print(f"\nAnalyzing: {sub_dir.name}")
                run_llm_analysis(sub_dir, args.analysis_model, client, args.max_workers)
            else:
                print(f"  Skipping {sim_file.name}: output dir not found")

    # Phase 3: Meta-analysis
    if run_meta:
        meta_model = args.meta_model or args.analysis_model
        print(f"\n--- Meta-Analysis Phase (model: {meta_model}) ---")
        for sim_file in args.simulation_files:
            sub_dir = args.output_dir / sim_file.stem
            if sub_dir.exists():
                print(f"\nMeta-analyzing: {sub_dir.name}")
                run_meta_analysis(sub_dir, meta_model, client, force=args.force_meta)
            else:
                print(f"  Skipping {sim_file.name}: output dir not found")

    print(f"\nDone! Analysis saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

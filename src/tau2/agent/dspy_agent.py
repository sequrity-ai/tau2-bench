"""DSPy-based agent for tau2-bench with prompt optimization support."""

from copy import deepcopy
from pathlib import Path
from typing import List, Optional
import json

from loguru import logger
from pydantic import BaseModel

from tau2.agent.base import (
    LocalAgent,
    ValidAgentInputMessage,
    is_valid_agent_history_message,
)
from tau2.data_model.message import (
    APICompatibleMessage,
    AssistantMessage,
    Message,
    MultiToolMessage,
    SystemMessage,
)
from tau2.environment.tool import Tool
from tau2.utils.llm_utils import generate


# Default agent instruction (can be overridden by optimized prompt)
DSPY_AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either:
- Send a message to the user.
- Make a tool call.
You cannot do both at the same time.

Try to be helpful and always follow the policy. Always make sure you generate valid JSON only.
""".strip()

DSPY_SYSTEM_PROMPT = """
<instructions>
{agent_instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


class DSPyAgentState(BaseModel):
    """The state of the DSPy agent."""

    system_messages: list[SystemMessage]
    messages: list[APICompatibleMessage]


class DSPyAgent(LocalAgent[DSPyAgentState]):
    """
    An LLM agent that supports DSPy prompt optimization.

    This agent extends LocalAgent and can use optimized PLLM prompts
    for improved performance on specific domains.

    Example:
        ```python
        # Basic usage
        agent = DSPyAgent(
            tools=tools,
            domain_policy=policy,
            llm="gpt-4.1",
        )

        # With optimized prompt
        agent = DSPyAgent(
            tools=tools,
            domain_policy=policy,
            llm="gpt-4.1",
            optimized_prompt_path="./optimized_prompts/airline.json",
        )
        ```
    """

    def __init__(
        self,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
        optimized_prompt_path: Optional[str | Path] = None,
        optimized_prompt: Optional[str] = None,
        agent_instruction: Optional[str] = None,
    ):
        """
        Initialize the DSPyAgent.

        Args:
            tools: List of tools available to the agent.
            domain_policy: The domain-specific policy text.
            llm: Model name/ID to use.
            llm_args: Additional arguments for the LLM.
            optimized_prompt_path: Path to JSON file with optimized prompt.
            optimized_prompt: Direct optimized prompt string.
            agent_instruction: Custom agent instruction (overrides default).
        """
        super().__init__(tools=tools, domain_policy=domain_policy)
        self.llm = llm
        self.llm_args = deepcopy(llm_args) if llm_args is not None else {}
        self.session_id = None

        # Load optimized prompt if provided
        self._optimized_prompt: Optional[str] = None
        self._optimized_prompt_info: Optional[dict] = None

        if optimized_prompt_path is not None:
            self._load_optimized_prompt(Path(optimized_prompt_path))
        elif optimized_prompt is not None:
            self._optimized_prompt = optimized_prompt

        # Set agent instruction
        if agent_instruction is not None:
            self._agent_instruction = agent_instruction
        elif self._optimized_prompt is not None:
            self._agent_instruction = self._optimized_prompt
        else:
            self._agent_instruction = DSPY_AGENT_INSTRUCTION

    def _load_optimized_prompt(self, path: Path) -> None:
        """Load optimized prompt from JSON file.

        Args:
            path: Path to the JSON file.
        """
        if not path.exists():
            logger.warning(f"Optimized prompt file not found: {path}")
            return

        try:
            with open(path, "r") as f:
                data = json.load(f)

            self._optimized_prompt_info = data
            self._optimized_prompt = data.get("prompt")

            if self._optimized_prompt:
                logger.info(
                    f"Loaded optimized prompt from {path} "
                    f"(score: {data.get('score', 'N/A')})"
                )
            else:
                logger.warning(f"No prompt found in {path}")
        except Exception as e:
            logger.error(f"Error loading optimized prompt from {path}: {e}")

    @property
    def system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return DSPY_SYSTEM_PROMPT.format(
            domain_policy=self.domain_policy,
            agent_instruction=self._agent_instruction,
        )

    @property
    def optimized_prompt(self) -> Optional[str]:
        """Get the current optimized prompt."""
        return self._optimized_prompt

    @optimized_prompt.setter
    def optimized_prompt(self, value: Optional[str]):
        """Set the optimized prompt."""
        self._optimized_prompt = value
        if value is not None:
            self._agent_instruction = value

    def get_init_state(
        self, message_history: Optional[list[Message]] = None
    ) -> DSPyAgentState:
        """Get the initial state of the agent.

        Args:
            message_history: The message history of the conversation.

        Returns:
            The initial state of the agent.
        """
        if message_history is None:
            message_history = []
        assert all(is_valid_agent_history_message(m) for m in message_history), (
            "Message history must contain only AssistantMessage, "
            "UserMessage, or ToolMessage to Agent."
        )
        return DSPyAgentState(
            system_messages=[SystemMessage(role="system", content=self.system_prompt)],
            messages=message_history,
        )

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: DSPyAgentState
    ) -> tuple[AssistantMessage, DSPyAgentState]:
        """
        Respond to a user or tool message.

        Args:
            message: The incoming message.
            state: Current agent state.

        Returns:
            Tuple of (response message, updated state).
        """
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)

        messages = state.system_messages + state.messages
        assistant_message, session_id = generate(
            model=self.llm,
            tools=self.tools,
            messages=messages,
            who_from="BOT",
            session_id=self.session_id,
            pllm_prompt=self._optimized_prompt,  # Pass optimized PLLM prompt to Security-API
            **self.llm_args,
        )
        self.session_id = session_id
        state.messages.append(assistant_message)
        return assistant_message, state

    def set_seed(self, seed: int):
        """Set the seed for the LLM.

        Args:
            seed: Random seed value.
        """
        if self.llm is None:
            raise ValueError("LLM is not set")
        cur_seed = self.llm_args.get("seed", None)
        if cur_seed is not None:
            logger.warning(f"Seed is already set to {cur_seed}, resetting to {seed}")
        self.llm_args["seed"] = seed

    def save_optimized_prompt(
        self,
        path: Path,
        score: float,
        strategy: str = "unknown",
        metadata: Optional[dict] = None,
    ) -> None:
        """Save the current optimized prompt to a file.

        Args:
            path: Path to save the JSON file.
            score: The optimization score achieved.
            strategy: The optimization strategy used.
            metadata: Additional metadata to save.
        """
        data = {
            "domain": self.domain_policy[:100] + "...",  # Truncated policy reference
            "prompt": self._optimized_prompt or self._agent_instruction,
            "score": score,
            "strategy": strategy,
            "model": self.llm,
            "train_size": metadata.get("train_size", 0) if metadata else 0,
            "val_size": metadata.get("val_size", 0) if metadata else 0,
            "metadata": metadata or {},
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved optimized prompt to {path}")

    @classmethod
    def from_optimized_prompt(
        cls,
        prompt_path: str | Path,
        tools: List[Tool],
        domain_policy: str,
        llm: Optional[str] = None,
        llm_args: Optional[dict] = None,
    ) -> "DSPyAgent":
        """Create a DSPyAgent with a pre-optimized prompt.

        Args:
            prompt_path: Path to the optimized prompt JSON file.
            tools: List of tools available to the agent.
            domain_policy: The domain-specific policy text.
            llm: Model name/ID to use.
            llm_args: Additional arguments for the LLM.

        Returns:
            Configured DSPyAgent instance.
        """
        return cls(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=llm_args,
            optimized_prompt_path=prompt_path,
        )

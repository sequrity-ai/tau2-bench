"""DSPy-compatible LM wrapper for tau2-bench's Security-API integration."""

import json
import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import dspy

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    UserMessage,
)
from tau2.utils.llm_utils import generate
from tau2.utils.utils import defence_params


@dataclass
class Tau2Response:
    """Response from Tau2 LM calls.

    Stores both the message and metadata from Security-API responses.
    """

    message: AssistantMessage
    session_id: str | None
    is_success: bool
    final_value: Any
    raw_content: str | None = None

    @property
    def content(self) -> str | None:
        """Get the text content of the response."""
        return self.message.content

    @property
    def tool_calls(self) -> list[ToolCall] | None:
        """Get tool calls from the response."""
        return self.message.tool_calls


class Tau2SequrityLM(dspy.LM):
    """DSPy-compatible LM wrapper for tau2-bench's Security-API integration.

    This class wraps tau2-bench's generate() function to be used as a DSPy
    language model, enabling DSPy optimization techniques for agent prompts.

    Example:
        ```python
        lm = Tau2SequrityLM(
            model="gpt-4.1",
            pllm_prompt="Your PLLM prompt..."
        )
        dspy.configure(lm=lm)
        ```
    """

    def __init__(
        self,
        model: str = "gpt-4.1",
        pllm_prompt: str | None = None,
        llm_args: dict | None = None,
        who_from: str = "BOT",
        defence_params_override: dict | None = None,
    ):
        """Initialize Tau2SequrityLM.

        Args:
            model: Model name/ID to use.
            pllm_prompt: The PLLM system prompt to use for optimization.
            llm_args: Additional arguments for the LLM (temperature, etc.).
            who_from: Identifier for the message source (BOT, USER, etc.).
            defence_params_override: Override specific defence_params settings.
        """
        super().__init__(model=model)

        self.model = model
        self._pllm_prompt = pllm_prompt
        self.llm_args = deepcopy(llm_args) if llm_args else {}
        self.who_from = who_from
        self.defence_params_override = defence_params_override or {}

        # Session management
        self._session_id: str | None = None

        # Store the last response for inspection
        self._last_response: Tau2Response | None = None
        self._last_raw_response: AssistantMessage | None = None

    @property
    def pllm_prompt(self) -> str | None:
        """Get the current PLLM prompt."""
        return self._pllm_prompt

    @pllm_prompt.setter
    def pllm_prompt(self, value: str | None):
        """Set the PLLM prompt."""
        self._pllm_prompt = value

    @property
    def last_response(self) -> Tau2Response | None:
        """Get the last Tau2 response."""
        return self._last_response

    @property
    def session_id(self) -> str | None:
        """Get the current session ID."""
        return self._session_id

    def reset_session(self):
        """Reset the session ID for a new conversation."""
        self._session_id = None

    def with_pllm_prompt(self, prompt: str) -> "Tau2SequrityLM":
        """Create a new instance with a different PLLM prompt.

        Useful for optimization where you want to try different prompts.

        Args:
            prompt: New PLLM prompt to use.

        Returns:
            New Tau2SequrityLM instance with the updated prompt.
        """
        return Tau2SequrityLM(
            model=self.model,
            pllm_prompt=prompt,
            llm_args=deepcopy(self.llm_args),
            who_from=self.who_from,
            defence_params_override=deepcopy(self.defence_params_override),
        )

    def _build_messages(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
    ) -> list[Message]:
        """Convert input to tau2 Message format.

        Args:
            prompt: Single prompt string (converted to user message).
            messages: List of message dicts with 'role' and 'content'.

        Returns:
            List of tau2 Message objects.
        """
        tau2_messages: list[Message] = []

        # Add PLLM prompt as system message if provided
        if self._pllm_prompt:
            tau2_messages.append(
                SystemMessage(role="system", content=self._pllm_prompt)
            )

        if messages is not None:
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")

                if role == "system":
                    tau2_messages.append(SystemMessage(role="system", content=content))
                elif role == "assistant":
                    tau2_messages.append(
                        AssistantMessage(role="assistant", content=content)
                    )
                else:  # user or default
                    tau2_messages.append(UserMessage(role="user", content=content))
        elif prompt is not None:
            tau2_messages.append(UserMessage(role="user", content=prompt))

        return tau2_messages

    def _extract_final_value(self, content: str | None) -> Any:
        """Extract final value from response content.

        Handles JSON-wrapped responses from PLLM.
        """
        if content is None:
            return None

        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                # Check for PLLM final_return_value format
                if "final_return_value" in parsed:
                    return parsed["final_return_value"].get("value")
                # Check for answer field
                if "answer" in parsed:
                    return parsed["answer"]
            return parsed
        except (json.JSONDecodeError, TypeError):
            return content

    def __call__(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> list[str]:
        """Call the Security-API via tau2's generate function.

        This method is called by DSPy to get completions.

        Args:
            prompt: Single prompt string (converted to user message).
            messages: List of message dicts with 'role' and 'content'.
            **kwargs: Additional arguments:
                - pllm_prompt: Override the default PLLM prompt
                - tools: List of Tool objects
                - tool_choice: Tool choice setting

        Returns:
            List containing the response content string.
        """
        # Allow overriding PLLM prompt via kwargs
        pllm_prompt = kwargs.pop("pllm_prompt", None)
        if pllm_prompt is not None:
            original_prompt = self._pllm_prompt
            self._pllm_prompt = pllm_prompt
        else:
            original_prompt = None

        try:
            # Build messages
            tau2_messages = self._build_messages(prompt=prompt, messages=messages)

            if not tau2_messages:
                raise ValueError("Either prompt or messages must be provided")

            # Extract optional tool settings
            tools = kwargs.pop("tools", None)
            tool_choice = kwargs.pop("tool_choice", None)

            # Merge LLM args
            call_args = {**self.llm_args, **kwargs}

            # Make API call via tau2's generate function
            assistant_message, session_id = generate(
                model=self.model,
                messages=tau2_messages,
                tools=tools,
                tool_choice=tool_choice,
                who_from=self.who_from,
                session_id=self._session_id,
                pllm_prompt=self._pllm_prompt,  # Pass PLLM prompt for optimization
                **call_args,
            )

            # Update session ID
            self._session_id = session_id

            # Store raw response
            self._last_raw_response = assistant_message

            # Extract content and final value
            content = assistant_message.content
            final_value = self._extract_final_value(content)

            # Create structured response
            self._last_response = Tau2Response(
                message=assistant_message,
                session_id=session_id,
                is_success=True,
                final_value=final_value,
                raw_content=content,
            )

            # Format response for DSPy compatibility
            if content:
                # Return in ChatAdapter-compatible tagged field format
                answer_text = str(final_value) if final_value is not None else content
                dspy_response = (
                    "[[ ## answer ## ]]\n"
                    f"{answer_text}\n\n"
                    "[[ ## completed ## ]]"
                )
                return [dspy_response]
            else:
                # If no content, return empty or handle tool calls
                if assistant_message.tool_calls:
                    tool_info = json.dumps([
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in assistant_message.tool_calls
                    ])
                    return [f"[TOOL_CALLS]{tool_info}"]
                return [""]

        except Exception as e:
            # Store error response
            self._last_response = Tau2Response(
                message=AssistantMessage(role="assistant", content=str(e)),
                session_id=self._session_id,
                is_success=False,
                final_value=None,
                raw_content=str(e),
            )
            raise
        finally:
            # Restore original prompt if it was overridden
            if original_prompt is not None:
                self._pllm_prompt = original_prompt

    def call_and_parse(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        **kwargs,
    ) -> Tau2Response:
        """Call the API and return a parsed Tau2Response.

        Convenience method that returns the parsed response directly.

        Args:
            prompt: Single prompt string.
            messages: List of message dicts.
            **kwargs: Additional arguments.

        Returns:
            Parsed Tau2Response object.
        """
        self.__call__(prompt=prompt, messages=messages, **kwargs)
        return self._last_response  # type: ignore

    def generate_with_tools(
        self,
        messages: list[Message],
        tools: list,
        tool_choice: str | None = "auto",
        **kwargs,
    ) -> tuple[AssistantMessage, str | None]:
        """Generate a response with tool support.

        This is a direct interface to tau2's generate function for agent use.

        Args:
            messages: List of tau2 Message objects.
            tools: List of Tool objects.
            tool_choice: Tool choice setting.
            **kwargs: Additional arguments.

        Returns:
            Tuple of (AssistantMessage, session_id).
        """
        call_args = {**self.llm_args, **kwargs}

        assistant_message, session_id = generate(
            model=self.model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            who_from=self.who_from,
            session_id=self._session_id,
            pllm_prompt=self._pllm_prompt,  # Pass PLLM prompt for optimization
            **call_args,
        )

        self._session_id = session_id
        self._last_raw_response = assistant_message

        return assistant_message, session_id

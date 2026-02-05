"""DSPy modules for tau2-bench agent behavior.

These modules wrap agent response generation for DSPy optimization.
"""

from typing import Any, Callable

import dspy

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    UserMessage,
)
from tau2.environment.tool import Tool


class AgentResponseSignature(dspy.Signature):
    """Signature for agent response generation.

    This signature defines the input/output contract for agent responses
    in a customer service context.
    """

    user_message: str = dspy.InputField(desc="The user's message or request")
    policy: str = dspy.InputField(desc="The domain policy to follow")
    response: str = dspy.OutputField(
        desc="The agent's response following the policy"
    )


class AgentWithToolsSignature(dspy.Signature):
    """Signature for agent response generation with tool awareness.

    Includes tool descriptions for the agent to reference.
    """

    user_message: str = dspy.InputField(desc="The user's message or request")
    policy: str = dspy.InputField(desc="The domain policy to follow")
    available_tools: str = dspy.InputField(
        desc="Description of available tools and their functions"
    )
    response: str = dspy.OutputField(
        desc="The agent's response or tool call decision"
    )


class AgentResponseModule(dspy.Module):
    """DSPy module wrapping agent response generation.

    This module can be used with DSPy optimizers to improve
    agent prompt behavior.
    """

    def __init__(
        self,
        signature: type | None = None,
        include_tools: bool = False,
    ):
        """Initialize the agent response module.

        Args:
            signature: Custom DSPy signature class (optional).
            include_tools: Whether to include tool descriptions.
        """
        super().__init__()

        if signature is None:
            signature = (
                AgentWithToolsSignature if include_tools else AgentResponseSignature
            )

        self.predictor = dspy.Predict(signature)
        self.include_tools = include_tools

    def forward(
        self,
        user_message: str,
        policy: str,
        available_tools: str | None = None,
    ) -> dspy.Prediction:
        """Generate an agent response.

        Args:
            user_message: The user's message.
            policy: Domain policy text.
            available_tools: Tool descriptions (if include_tools=True).

        Returns:
            DSPy Prediction with response field.
        """
        if self.include_tools and available_tools:
            return self.predictor(
                user_message=user_message,
                policy=policy,
                available_tools=available_tools,
            )
        else:
            return self.predictor(
                user_message=user_message,
                policy=policy,
            )


class ConversationModule(dspy.Module):
    """DSPy module for multi-turn conversation handling.

    Maintains conversation context and generates contextually
    appropriate responses.
    """

    def __init__(
        self,
        max_history: int = 10,
    ):
        """Initialize the conversation module.

        Args:
            max_history: Maximum number of turns to include in context.
        """
        super().__init__()
        self.max_history = max_history

        class ConversationSignature(dspy.Signature):
            """Multi-turn conversation response generation."""

            conversation_history: str = dspy.InputField(
                desc="Previous conversation turns"
            )
            current_message: str = dspy.InputField(desc="Current user message")
            policy: str = dspy.InputField(desc="Domain policy to follow")
            response: str = dspy.OutputField(desc="Agent's response")

        self.predictor = dspy.Predict(ConversationSignature)

    def _format_history(self, messages: list[Message]) -> str:
        """Format message history for the prompt."""
        history_parts = []
        recent_messages = messages[-self.max_history * 2 :]  # user/assistant pairs

        for msg in recent_messages:
            if isinstance(msg, UserMessage):
                history_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AssistantMessage):
                content = msg.content or "[Tool call]"
                history_parts.append(f"Agent: {content}")

        return "\n".join(history_parts) if history_parts else "No previous messages."

    def forward(
        self,
        messages: list[Message],
        current_message: str,
        policy: str,
    ) -> dspy.Prediction:
        """Generate response given conversation history.

        Args:
            messages: Previous conversation messages.
            current_message: Current user message.
            policy: Domain policy text.

        Returns:
            DSPy Prediction with response field.
        """
        history = self._format_history(messages)
        return self.predictor(
            conversation_history=history,
            current_message=current_message,
            policy=policy,
        )


class ToolSelectionModule(dspy.Module):
    """DSPy module for tool selection decisions.

    Helps the agent decide when and which tools to use.
    """

    def __init__(self):
        """Initialize the tool selection module."""
        super().__init__()

        class ToolSelectionSignature(dspy.Signature):
            """Decide whether to use a tool and which one."""

            user_request: str = dspy.InputField(desc="The user's request")
            available_tools: str = dspy.InputField(
                desc="List of available tools with descriptions"
            )
            conversation_context: str = dspy.InputField(
                desc="Recent conversation context"
            )
            decision: str = dspy.OutputField(
                desc="Either 'respond' to send a message, or tool name to call"
            )
            reasoning: str = dspy.OutputField(
                desc="Brief reasoning for the decision"
            )

        self.predictor = dspy.Predict(ToolSelectionSignature)

    @staticmethod
    def format_tools(tools: list[Tool]) -> str:
        """Format tools for the prompt.

        Args:
            tools: List of Tool objects.

        Returns:
            Formatted string describing tools.
        """
        tool_descriptions = []
        for tool in tools:
            desc = f"- {tool.name}: {tool.short_desc}"
            tool_descriptions.append(desc)
        return "\n".join(tool_descriptions)

    def forward(
        self,
        user_request: str,
        tools: list[Tool],
        conversation_context: str = "",
    ) -> dspy.Prediction:
        """Decide on tool usage.

        Args:
            user_request: The user's request.
            tools: Available tools.
            conversation_context: Recent conversation context.

        Returns:
            DSPy Prediction with decision and reasoning.
        """
        tools_text = self.format_tools(tools)
        return self.predictor(
            user_request=user_request,
            available_tools=tools_text,
            conversation_context=conversation_context or "No previous context.",
        )


class PolicyComplianceModule(dspy.Module):
    """DSPy module for policy compliance checking.

    Evaluates whether a response complies with the domain policy.
    """

    def __init__(self):
        """Initialize the policy compliance module."""
        super().__init__()

        class PolicyComplianceSignature(dspy.Signature):
            """Check if a response complies with policy."""

            response: str = dspy.InputField(desc="The agent's proposed response")
            policy: str = dspy.InputField(desc="The domain policy")
            user_request: str = dspy.InputField(desc="Original user request")
            compliant: bool = dspy.OutputField(
                desc="Whether the response complies with policy"
            )
            issues: str = dspy.OutputField(
                desc="Any policy compliance issues found"
            )

        self.predictor = dspy.Predict(PolicyComplianceSignature)

    def forward(
        self,
        response: str,
        policy: str,
        user_request: str,
    ) -> dspy.Prediction:
        """Check policy compliance.

        Args:
            response: The proposed response.
            policy: Domain policy text.
            user_request: Original user request.

        Returns:
            DSPy Prediction with compliant flag and issues.
        """
        return self.predictor(
            response=response,
            policy=policy,
            user_request=user_request,
        )


def create_agent_module(
    include_tools: bool = False,
    signature: type | None = None,
) -> AgentResponseModule:
    """Factory function to create agent modules.

    Args:
        include_tools: Whether to include tool awareness.
        signature: Custom DSPy signature class.

    Returns:
        Configured AgentResponseModule.
    """
    return AgentResponseModule(
        signature=signature,
        include_tools=include_tools,
    )

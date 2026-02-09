"""Test that optimized PLLM prompts are correctly passed through the entire chain:
DSPyAgent -> generate() -> X-Security-Config header -> overwrite_pllm_prompt
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch
from copy import deepcopy

import pytest

from tau2.data_model.message import (
    AssistantMessage,
    SystemMessage,
    UserMessage,
)
from tau2.environment.tool import Tool, as_tool
from tau2.utils.llm_utils import generate
from tau2.agent.dspy_agent import DSPyAgent


# --- Fixtures ---

@pytest.fixture
def mock_tool() -> Tool:
    def lookup_order(order_id: str) -> str:
        """Look up an order by ID.
        Args:
            order_id (str): The order ID.
        Returns:
            str: The order details.
        """
        return f"Order {order_id} found"
    return as_tool(lookup_order)


@pytest.fixture
def mock_completion_response():
    """Create a mock litellm completion response."""
    mock_response = MagicMock()
    mock_response._response_headers = {"x-session-id": "test-session-123"}
    mock_choice = MagicMock()
    mock_choice.finish_reason = "stop"
    mock_choice.message.role = "assistant"
    mock_choice.message.content = "I can help you with that."
    mock_choice.message.tool_calls = None
    mock_choice.to_dict.return_value = {
        "message": {"role": "assistant", "content": "I can help you with that."},
        "finish_reason": "stop",
    }
    mock_response.choices = [mock_choice]
    mock_response.model = "gpt-5-mini"
    mock_response.get.return_value = None  # usage
    return mock_response


@pytest.fixture(autouse=True)
def set_env_vars():
    """Set required environment variables for generate()."""
    env_vars = {
        "ENDPOINT_ADDRESS": "https://api.sequrity.ai/control/openrouter/v1",
        "ENDPOINT_ADDRESS_FULL": "https://openrouter.ai/api/v1",
        "X_Sequrity_Api_Key": "test-sequrity-key",
        "X_Api_Key": "test-api-key",
        "OPENAI_API_KEY": "test-openai-key",
    }
    with patch.dict(os.environ, env_vars):
        yield


@pytest.fixture(autouse=True)
def set_defence_params():
    """Set default defence_params for generate()."""
    import tau2.utils.llm_utils as llm_utils
    original = deepcopy(llm_utils.defence_params)
    llm_utils.defence_params.update({
        'clear_history_every_n_attempts': 0,
        'max_retry_attempts': 1,
        'tool_policies': [],
        'retry_on_policy_violation': False,
        'allow_undefined_tools': True,
        'fail_fast': False,
        'auto_gen_policies': False,
        'bot_dual_llm_mode': True,
        'user_dual_llm_mode': False,
        'strict_mode': False,
        'multistepmode': False,
        'plan_reduction': 'none',
        'n_plans': 1,
        'max_nested_session_depth': 1,
        'max_n_turns': 10,
        'reasoning_effort': 'high',
        'pllm_debug_info_level': 'minimal',
        'min_num_tools_for_filtering': 0,
        'user_direct_model': False,
        'bot_direct_model': False,
    })
    yield
    llm_utils.defence_params = original


# --- Tests ---

class TestPllmPromptInGenerateHeaders:
    """Test that pllm_prompt parameter in generate() ends up in X-Security-Config."""

    @patch("tau2.utils.llm_utils.completion")
    def test_pllm_prompt_included_in_headers(self, mock_completion, mock_completion_response):
        """When pllm_prompt is provided, it should appear as overwrite_pllm_prompt in X-Security-Config."""
        mock_completion.return_value = mock_completion_response

        test_prompt = "You are an optimized airline agent. Follow policy strictly."
        messages = [
            SystemMessage(role="system", content="System message"),
            UserMessage(role="user", content="Help me change my flight"),
        ]

        generate(
            model="gpt-5-mini",
            messages=messages,
            who_from="BOT",
            pllm_prompt=test_prompt,
        )

        # Extract the extra_headers passed to completion()
        call_kwargs = mock_completion.call_args
        extra_headers = call_kwargs.kwargs.get("extra_headers") or call_kwargs[1].get("extra_headers")

        assert extra_headers is not None, "extra_headers should be set"
        security_config = json.loads(extra_headers["X-Security-Config"])
        assert "overwrite_pllm_prompt" in security_config, (
            f"overwrite_pllm_prompt missing from X-Security-Config: {security_config}"
        )
        assert security_config["overwrite_pllm_prompt"] == test_prompt

    @patch("tau2.utils.llm_utils.completion")
    def test_no_pllm_prompt_means_no_overwrite(self, mock_completion, mock_completion_response):
        """When pllm_prompt is None and not in defence_params, overwrite_pllm_prompt should be absent."""
        mock_completion.return_value = mock_completion_response

        messages = [
            SystemMessage(role="system", content="System message"),
            UserMessage(role="user", content="Hello"),
        ]

        generate(
            model="gpt-5-mini",
            messages=messages,
            who_from="BOT",
            pllm_prompt=None,
        )

        call_kwargs = mock_completion.call_args
        extra_headers = call_kwargs.kwargs.get("extra_headers") or call_kwargs[1].get("extra_headers")
        security_config = json.loads(extra_headers["X-Security-Config"])
        assert "overwrite_pllm_prompt" not in security_config, (
            f"overwrite_pllm_prompt should NOT be in X-Security-Config when no prompt: {security_config}"
        )

    @patch("tau2.utils.llm_utils.completion")
    def test_pllm_prompt_overrides_defence_params(self, mock_completion, mock_completion_response):
        """pllm_prompt parameter should take priority over defence_params['pllm_prompt']."""
        mock_completion.return_value = mock_completion_response

        import tau2.utils.llm_utils as llm_utils
        llm_utils.defence_params['pllm_prompt'] = "Default prompt from defence_params"

        override_prompt = "Optimized prompt from DSPy"
        messages = [
            SystemMessage(role="system", content="System message"),
            UserMessage(role="user", content="Hello"),
        ]

        generate(
            model="gpt-5-mini",
            messages=messages,
            who_from="BOT",
            pllm_prompt=override_prompt,
        )

        call_kwargs = mock_completion.call_args
        extra_headers = call_kwargs.kwargs.get("extra_headers") or call_kwargs[1].get("extra_headers")
        security_config = json.loads(extra_headers["X-Security-Config"])
        assert security_config["overwrite_pllm_prompt"] == override_prompt, (
            f"Expected '{override_prompt}', got '{security_config.get('overwrite_pllm_prompt')}'"
        )

    @patch("tau2.utils.llm_utils.completion")
    def test_defence_params_pllm_prompt_used_as_fallback(self, mock_completion, mock_completion_response):
        """When pllm_prompt is None, defence_params['pllm_prompt'] should be used."""
        mock_completion.return_value = mock_completion_response

        import tau2.utils.llm_utils as llm_utils
        fallback_prompt = "Fallback prompt from defence_params"
        llm_utils.defence_params['pllm_prompt'] = fallback_prompt

        messages = [
            SystemMessage(role="system", content="System message"),
            UserMessage(role="user", content="Hello"),
        ]

        generate(
            model="gpt-5-mini",
            messages=messages,
            who_from="BOT",
            pllm_prompt=None,
        )

        call_kwargs = mock_completion.call_args
        extra_headers = call_kwargs.kwargs.get("extra_headers") or call_kwargs[1].get("extra_headers")
        security_config = json.loads(extra_headers["X-Security-Config"])
        assert security_config["overwrite_pllm_prompt"] == fallback_prompt


class TestDSPyAgentPassesPllmPrompt:
    """Test that DSPyAgent correctly passes the optimized prompt to generate()."""

    @patch("tau2.agent.dspy_agent.generate")
    def test_dspy_agent_passes_optimized_prompt(self, mock_generate, mock_tool):
        """DSPyAgent should pass _optimized_prompt to generate() as pllm_prompt."""
        mock_generate.return_value = (
            AssistantMessage(role="assistant", content="Sure, I can help."),
            "session-abc",
        )

        test_prompt = "You are an optimized airline customer service agent."
        agent = DSPyAgent(
            tools=[mock_tool],
            domain_policy="Be helpful.",
            llm="gpt-5-mini",
            optimized_prompt=test_prompt,
        )

        state = agent.get_init_state()
        user_msg = UserMessage(role="user", content="I need to cancel my flight.")
        agent.generate_next_message(user_msg, state)

        # Verify generate() was called with pllm_prompt
        call_kwargs = mock_generate.call_args
        assert call_kwargs.kwargs.get("pllm_prompt") == test_prompt, (
            f"Expected pllm_prompt='{test_prompt}', got '{call_kwargs.kwargs.get('pllm_prompt')}'"
        )

    @patch("tau2.agent.dspy_agent.generate")
    def test_dspy_agent_no_prompt_passes_none(self, mock_generate, mock_tool):
        """DSPyAgent without optimized prompt should pass pllm_prompt=None."""
        mock_generate.return_value = (
            AssistantMessage(role="assistant", content="Hello."),
            "session-abc",
        )

        agent = DSPyAgent(
            tools=[mock_tool],
            domain_policy="Be helpful.",
            llm="gpt-5-mini",
        )

        state = agent.get_init_state()
        user_msg = UserMessage(role="user", content="Hi")
        agent.generate_next_message(user_msg, state)

        call_kwargs = mock_generate.call_args
        assert call_kwargs.kwargs.get("pllm_prompt") is None

    @patch("tau2.agent.dspy_agent.generate")
    def test_dspy_agent_loads_prompt_from_file(self, mock_generate, mock_tool, tmp_path):
        """DSPyAgent should load prompt from JSON file and pass it to generate()."""
        mock_generate.return_value = (
            AssistantMessage(role="assistant", content="Got it."),
            "session-abc",
        )

        prompt_text = "Optimized prompt loaded from file."
        prompt_file = tmp_path / "optimized.json"
        prompt_file.write_text(json.dumps({
            "domain": "airline",
            "prompt": prompt_text,
            "score": 0.92,
            "strategy": "gepa",
            "model": "gpt-5-mini",
            "train_size": 50,
            "val_size": 25,
            "metadata": {},
        }))

        agent = DSPyAgent(
            tools=[mock_tool],
            domain_policy="Be helpful.",
            llm="gpt-5-mini",
            optimized_prompt_path=str(prompt_file),
        )

        assert agent._optimized_prompt == prompt_text

        state = agent.get_init_state()
        user_msg = UserMessage(role="user", content="Change my seat")
        agent.generate_next_message(user_msg, state)

        call_kwargs = mock_generate.call_args
        assert call_kwargs.kwargs.get("pllm_prompt") == prompt_text

    @patch("tau2.agent.dspy_agent.generate")
    def test_dspy_agent_prompt_setter_updates(self, mock_generate, mock_tool):
        """Setting optimized_prompt via property should update what's passed to generate()."""
        mock_generate.return_value = (
            AssistantMessage(role="assistant", content="Ok."),
            "session-abc",
        )

        agent = DSPyAgent(
            tools=[mock_tool],
            domain_policy="Be helpful.",
            llm="gpt-5-mini",
        )

        new_prompt = "Updated prompt via setter"
        agent.optimized_prompt = new_prompt

        state = agent.get_init_state()
        user_msg = UserMessage(role="user", content="Help")
        agent.generate_next_message(user_msg, state)

        call_kwargs = mock_generate.call_args
        assert call_kwargs.kwargs.get("pllm_prompt") == new_prompt

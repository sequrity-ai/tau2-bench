import json
import re
from typing import Any, Optional
import os

from sequrity.control import FeaturesHeader, FineGrainedConfigHeader, SecurityPolicyHeader

import litellm
litellm.return_response_headers = True
litellm.suppress_debug_info = True

from litellm import completion, completion_cost
from litellm.caching.caching import Cache
from litellm.main import ModelResponse, Usage
from loguru import logger

from tau2.config import (
    DEFAULT_LLM_CACHE_TYPE,
    DEFAULT_MAX_RETRIES,
    LLM_CACHE_ENABLED,
    REDIS_CACHE_TTL,
    REDIS_CACHE_VERSION,
    REDIS_HOST,
    REDIS_PASSWORD,
    REDIS_PORT,
    REDIS_PREFIX,
    USE_LANGFUSE,
)
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.environment.tool import Tool
from .utils import defence_params

# litellm._turn_on_debug()

if USE_LANGFUSE:
    # set callbacks
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]

litellm.drop_params = True

if LLM_CACHE_ENABLED:
    if DEFAULT_LLM_CACHE_TYPE == "redis":
        logger.info(f"LiteLLM: Using Redis cache at {REDIS_HOST}:{REDIS_PORT}")
        litellm.cache = Cache(
            type=DEFAULT_LLM_CACHE_TYPE,
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            namespace=f"{REDIS_PREFIX}:{REDIS_CACHE_VERSION}:litellm",
            ttl=REDIS_CACHE_TTL,
        )
    elif DEFAULT_LLM_CACHE_TYPE == "local":
        logger.info("LiteLLM: Using local cache")
        litellm.cache = Cache(
            type="local",
            ttl=REDIS_CACHE_TTL,
        )
    else:
        raise ValueError(
            f"Invalid cache type: {DEFAULT_LLM_CACHE_TYPE}. Should be 'redis' or 'local'"
        )
    litellm.enable_cache()
else:
    logger.info("LiteLLM: Cache is disabled")
    litellm.disable_cache()


ALLOW_SONNET_THINKING = False

if not ALLOW_SONNET_THINKING:
    logger.warning("Sonnet thinking is disabled")
perplexity_cfg = {
    "compute_pllm_perplexity_ratio": True,
    "pllm_perplexity_ratio_top_logprobs": 5,
    "show_pllm_secure_var_values": "none",  # overrides your current "basic-notext"
}

def _parse_ft_model_name(model: str) -> str:
    """
    Parse the ft model name from the litellm model name.
    e.g: "ft:gpt-4.1-mini-2025-04-14:sierra::BSQA2TFg" -> "gpt-4.1-mini-2025-04-14"
    """
    pattern = r"ft:(?P<model>[^:]+):(?P<provider>\w+)::(?P<id>\w+)"
    match = re.match(pattern, model)
    if match:
        return match.group("model")
    else:
        return model


def get_response_cost(response: ModelResponse) -> float:
    """
    Get the cost of the response from the litellm completion.
    """
    response.model = _parse_ft_model_name(
        response.model
    )  # FIXME: Check Litellm, passing the model to completion_cost doesn't work.
    try:
        cost = completion_cost(completion_response=response)
    except Exception as e:
        #logger.error(e)
        return 0.0
    return cost


def get_response_usage(response: ModelResponse) -> Optional[dict]:
    usage: Optional[Usage] = response.get("usage")
    if usage is None:
        return None
    return {
        "completion_tokens": usage.completion_tokens,
        "prompt_tokens": usage.prompt_tokens,
    }


def to_tau2_messages(
    messages: list[dict], ignore_roles: set[str] = set()
) -> list[Message]:
    """
    Convert a list of messages from a dictionary to a list of Tau2 messages.
    """
    tau2_messages = []
    for message in messages:
        role = message["role"]
        if role in ignore_roles:
            continue
        if role == "user":
            tau2_messages.append(UserMessage(**message))
        elif role == "assistant":
            tau2_messages.append(AssistantMessage(**message))
        elif role == "tool":
            tau2_messages.append(ToolMessage(**message))
        elif role == "system":
            tau2_messages.append(SystemMessage(**message))
        else:
            raise ValueError(f"Unknown message type: {role}")
    return tau2_messages


def to_litellm_messages(messages: list[Message]) -> list[dict]:
    """
    Convert a list of Tau2 messages to a list of litellm messages.
    """
    litellm_messages = []
    for message in messages:
        if isinstance(message, UserMessage):
            litellm_messages.append({"role": "user", "content": message.content})
        elif isinstance(message, AssistantMessage):
            tool_calls = None
            if message.is_tool_call():
                tool_calls = [
                    {
                        "id": tc.id,
                        "name": tc.name,
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                        "type": "function",
                    }
                    for tc in message.tool_calls
                ]
            litellm_messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": tool_calls,
                }
            )
        elif isinstance(message, ToolMessage):
            litellm_messages.append(
                {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.id,
                }
            )
        elif isinstance(message, SystemMessage):
            litellm_messages.append({"role": "system", "content": message.content})
    return litellm_messages


def generate(
    model: str,
    messages: list[Message],
    tools: Optional[list[Tool]] = None,
    tool_choice: Optional[str] = None,
    # USER if its a message from user, 
    # BOT if the agent, 
    # GTBOT if ground truth bot, 
    # SOLOBOT if solobot, 
    # INTERFACEBOT for interface bot, 
    # ASSERTIONBOT for assertion evaluation
    who_from = None, 
    session_id = None,
    **kwargs: Any,
) -> (UserMessage | AssistantMessage, str):
    """
    Generate a response from the model.

    Args:
        model: The model to use.
        messages: The messages to send to the model.
        tools: The tools to use.
        tool_choice: The tool choice to use.
        **kwargs: Additional arguments to pass to the model.

    Returns: A tuple containing the message and the cost.
    """

    global defence_params 

    kwargs["api_base"] = os.getenv("ENDPOINT_ADDRESS")

    if kwargs.get("num_retries") is None:
        kwargs["num_retries"] = DEFAULT_MAX_RETRIES

    if model.startswith("claude") and not ALLOW_SONNET_THINKING:
        kwargs["thinking"] = {"type": "disabled"}
    litellm_messages = to_litellm_messages(messages)
    tools = [tool.openai_schema for tool in tools] if tools else None
    if tools and tool_choice is None:
        tool_choice = "auto"


    #---------------
    clear_history_every_n_attempts = defence_params['clear_history_every_n_attempts']

    max_retry_attempts          = defence_params['max_retry_attempts']
    tool_policies               = defence_params['tool_policies']
    retry_on_policy_violation   = defence_params['retry_on_policy_violation']
    allow_undefined_tools       = defence_params['allow_undefined_tools']
    fail_fast                   = defence_params['fail_fast']

    auto_gen_policies           = defence_params['auto_gen_policies']
    bot_dual_llm_mode           = defence_params['bot_dual_llm_mode']
    user_dual_llm_mode          = defence_params['user_dual_llm_mode']

    strict_mode                 = defence_params['strict_mode']
    multistepmode               = defence_params['multistepmode']

    op_type                     = defence_params['plan_reduction']
    num_plans                   = defence_params['n_plans']

    max_nested_session_depth    = defence_params['max_nested_session_depth']
    max_n_turns                 = defence_params['max_n_turns']

    
    reasoning_effort_user       = defence_params['reasoning_effort_user']
    reasoning_effort_bot        = defence_params['reasoning_effort_bot']

    pllm_debug_info_level       = defence_params['pllm_debug_info_level']
    min_num_tools_for_filtering = defence_params['min_num_tools_for_filtering']

    # USER if its a message from user, 
    # BOT if the agent, 
    # GTBOT if ground truth bot, 
    # SOLOBOT if solobot, 
    # INTERFACEBOT for interface bot, 
    # ASSERTIONBOT for assertion evaluation
    if who_from in ["BOT", "SOLOBOT"]:
        dual_llm_mode = bot_dual_llm_mode
        reasoning_effort = reasoning_effort_bot

    if who_from in ['USER']:
        dual_llm_mode = user_dual_llm_mode 
        reasoning_effort = reasoning_effort_user

    if dual_llm_mode:
        features = FeaturesHeader.dual_llm()
        policy = SecurityPolicyHeader.dual_llm(
            mode="strict" if strict_mode else "standard",
            codes=tool_policies,
            auto_gen=auto_gen_policies,
            fail_fast=fail_fast,
            default_allow=allow_undefined_tools,
        )
        config = FineGrainedConfigHeader.dual_llm(
            min_num_tools_for_filtering=min_num_tools_for_filtering,
            max_n_turns=max_n_turns,
            max_pllm_steps=max_retry_attempts,
            clear_history_every_n_attempts=clear_history_every_n_attempts,
            retry_on_policy_violation=retry_on_policy_violation,
            enable_multistep_planning=multistepmode,
            pllm_debug_info_level=pllm_debug_info_level,
            strip_response_content=True,
            include_program=False,
        )
    else:
        features = FeaturesHeader.single_llm()
        policy = SecurityPolicyHeader.single_llm(
            mode="strict" if strict_mode else "standard",
            codes=tool_policies,
            fail_fast=fail_fast,
            default_allow=allow_undefined_tools,
        )
        config = FineGrainedConfigHeader.single_llm(
            min_num_tools_for_filtering=min_num_tools_for_filtering,
            max_n_turns=max_n_turns,
        )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['X_Sequrity_Api_Key']}",
        "X-Api-Key": os.environ["X_Api_Key"],
        "X-Features": features.dump_for_headers(),
        "X-Policy": policy.dump_for_headers(),
        "X-Config": config.dump_for_headers(),
    }

    if session_id is not None:
        headers["X-Session-ID"] = session_id

    if (who_from in ["USER",]) and defence_params['user_direct_model']:
        headers = {}
        kwargs['api_base'] = os.environ['ENDPOINT_ADDRESS_FULL']

    if (who_from in ["BOT",]) and defence_params['bot_direct_model']:
        headers = {}
        kwargs['api_base'] = os.environ['ENDPOINT_ADDRESS_FULL']

    try:
        response = completion(
            reasoning_effort=reasoning_effort,
            model=model,
            messages=litellm_messages,
            tools=tools,
            tool_choice=tool_choice,
            extra_headers=headers,
            **kwargs,
        )
    except Exception as e:
        logger.error(e)
        raise e
    
    session_id = response._response_headers.get('x-session-id', None)

    cost = get_response_cost(response)
    usage = get_response_usage(response)
    response = response.choices[0]

    # unpacking our endpoint data structures 
    if False:
        try:
            fieldname =  'final_return_value'
            loadable = json.loads(response.message.content)

            if fieldname in loadable:
                frv = loadable[fieldname]['value']
                response.message.content = json.dumps(frv)
        except:
            pass

    try:
        finish_reason = response.finish_reason
        if finish_reason == "length":
            logger.warning("Output might be incomplete due to token limit!")
    except Exception as e:
        logger.error(e)
        raise e
    assert response.message.role == "assistant", (
        "The response should be an assistant message"
    )
    content = response.message.content
    tool_calls = response.message.tool_calls or []
    tool_calls = [
        ToolCall(
            id=tool_call.id,
            name=tool_call.function.name,
            arguments=json.loads(tool_call.function.arguments),
        )
        for tool_call in tool_calls
    ]
    tool_calls = tool_calls or None

    message = AssistantMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        cost=cost,
        usage=usage,
        raw_data=response.to_dict(),
    )
    return (message, session_id)


def get_cost(messages: list[Message]) -> tuple[float, float] | None:
    """
    Get the cost of the interaction between the agent and the user.
    Returns None if any message has no cost.
    """
    agent_cost = 0
    user_cost = 0
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.cost is not None:
            if isinstance(message, AssistantMessage):
                agent_cost += message.cost
            elif isinstance(message, UserMessage):
                user_cost += message.cost
        else:
            logger.warning(f"Message {message.role}: {message.content} has no cost")
            return None
    return agent_cost, user_cost


def get_token_usage(messages: list[Message]) -> dict:
    """
    Get the token usage of the interaction between the agent and the user.
    """
    usage = {"completion_tokens": 0, "prompt_tokens": 0}
    for message in messages:
        if isinstance(message, ToolMessage):
            continue
        if message.usage is None:
            logger.warning(f"Message {message.role}: {message.content} has no usage")
            continue
        usage["completion_tokens"] += message.usage["completion_tokens"]
        usage["prompt_tokens"] += message.usage["prompt_tokens"]
    return usage

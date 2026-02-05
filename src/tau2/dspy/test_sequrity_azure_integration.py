# save as test_sequrity_litellm.py
import os, json
import litellm
from litellm import completion

litellm.return_response_headers = True
litellm.suppress_debug_info = False  # turn ON while debugging
dual_llm_mode = True
strict_mode = False
api_base = os.environ["ENDPOINT_ADDRESS"]
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {os.environ['X_Sequrity_Api_Key']}",
    "X-Api-Key": os.environ["X_Api_Key"],
    "X-Security-Config": json.dumps({
        "max_pllm_attempts": 2,
        "max_n_turns": 3,
        "disable_rllm": True,
        "pllm_debug_info_level": "minimal",
    }),
    "X-Security-Policy": json.dumps({
        "language": "sqrt",
        "fail_fast": True,
        "auto_gen": True,
        "codes": [],
        "internal_policy_preset": {"default_allow": True, "enable_non_executable_memory": True},
    }),
    "X-Security-Features":  json.dumps([
          {"feature_name": "Dual LLM" if dual_llm_mode else "Single LLM",
              "config_json": json.dumps(
                  {"mode": "strict" if strict_mode else "standard"})}, # or strict
          {"feature_name": "Long Program Support",
              "config_json": json.dumps(
                  {"mode": "base"})},
        ]),
}

resp = completion(
    model="gpt-5-mini",  # same note as above
    api_base=api_base,
    extra_headers=headers,
    messages=[
        {"role": "user", "content": "Reply with exactly: OK"},
    ],
)

print("x-session-id:", resp._response_headers.get("x-session-id"))
print("model:", resp.model)
print("content:", resp.choices[0].message["content"])
print("usage:", resp.get("usage"))

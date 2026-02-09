import hashlib
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from deepdiff import DeepDiff
from dotenv import load_dotenv
from loguru import logger

defence_params = {
    "tool_policies": "",
    "max_retry_attempts": 20,
    "max_n_turns": 100,
    "n_plans": None,
    "multistepmode": False,
    "clear_history_every_n_attempts": 5, # only works in single-step mode
    "retry_on_policy_violation": True,
    "allow_undefined_tools": True,
    "fail_fast": True,
    "enable_multistep_planning": False,

    "reasoning_effort_user": "low",  #"minimal" #"medium" # low
    "reasoning_effort_bot":  "low",  #"minimal" #"medium" # low
    "plan_reduction": "best", # "merge"
    "auto_gen_policies": False,

    "bot_dual_llm_mode": True,
    "user_dual_llm_mode": False,

    "strict_mode": False,
    "max_nested_session_depth": 1,
    "min_num_tools_for_filtering": 100,
    "pllm_debug_info_level": "minimal", #"minimal", "normal", "extra"

    "bot_direct_model": False,  # if for user the server should talk to the server directly
    "user_direct_model": True, # if for user the server should talk to the server directly

    "pllm_custom_instructions": None,
}

res = load_dotenv()
if not res:
    logger.warning("No .env file found")

# Try to get data directory from environment variable first
DATA_DIR_ENV = os.getenv("TAU2_DATA_DIR")

if DATA_DIR_ENV:
    # Use environment variable if set
    DATA_DIR = Path(DATA_DIR_ENV)
    logger.info(f"Using data directory from environment: {DATA_DIR}")
else:
    # Fallback to source directory (for development)
    SOURCE_DIR = Path(__file__).parents[3]
    DATA_DIR = SOURCE_DIR / "data"
    logger.info(f"Using data directory from source: {DATA_DIR}")

# Check if data directory exists and is accessible
if not DATA_DIR.exists():
    logger.warning(f"Data directory does not exist: {DATA_DIR}")
    logger.warning(
        "Set TAU2_DATA_DIR environment variable to point to your data directory"
    )
    logger.warning("Or ensure the data directory exists in the expected location")


def get_dict_hash(obj: dict) -> str:
    """
    Generate a unique hash for dict.
    Returns a hex string representation of the hash.
    """
    hash_string = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(hash_string.encode()).hexdigest()


def show_dict_diff(dict1: dict, dict2: dict) -> str:
    """
    Show the difference between two dictionaries.
    """
    diff = DeepDiff(dict1, dict2)
    return diff


def get_now() -> str:
    """
    Returns the current date and time in the format YYYYMMDD_HHMMSS.
    """
    now = datetime.now()
    return format_time(now)


def format_time(time: datetime) -> str:
    """
    Format the time in the format YYYYMMDD_HHMMSS.
    """
    return time.isoformat()


def get_commit_hash() -> str:
    """
    Get the commit hash of the current directory.
    """
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
            .split("\n")[0]
        )
    except Exception as e:
        logger.error(f"Failed to get git hash: {e}")
        commit_hash = "unknown"
    return commit_hash


import json

def smart_update_headers(headers, flat_update_json):
    """
    Deserializes header values, searches for keys provided in 'flat_update_json',
    updates them wherever they are found (nested dicts or lists), and reserializes.
    
    Args:
        headers (dict): The dictionary of headers with JSON-string values.
        flat_update_json (str): A JSON string of keys/values to update. 
                                E.g., '{"max_n_turns": 99, "default_allow": false}'
    """
    
    # 1. Deserialize the update payload
    try:
        updates = json.loads(flat_update_json)
    except json.JSONDecodeError as e:
        print(f"Error parsing update payload: {e}")
        return headers

    for update in updates:
        if update not in headers:
            raise Exception("Unsupported key " + str(update))
        headers[update] = updates[update]

    return headers

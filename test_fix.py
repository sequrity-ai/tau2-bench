#!/usr/bin/env python
"""Test script to verify the environment fix"""

import os
os.environ["ENDPOINT_ADDRESS"] = "http://localhost:8000/control/azurecredits/v1"
os.environ["X_Sequrity_Api_Key"] = "007464375aaed28bac3d4294fdb5fbe05e4841fc76ed45bb3f061f353a24a677"

from tau2.registry import registry
from tau2.domains.telecom.environment import get_tasks

# Test 1: Environment creation
print("Test 1: Creating telecom environment...")
env_constructor = registry.get_env_constructor('telecom')
env = env_constructor()
print(f"  ✓ Environment created: {type(env).__name__}")
print(f"  ✓ Has tools: {env.tools is not None}")
print(f"  ✓ Has user_tools: {env.user_tools is not None}")

# Test 2: Check that user_tools has the required method
if env.user_tools:
    print(f"  ✓ user_tools has set_user_info: {hasattr(env.user_tools, 'set_user_info')}")

# Test 3: Load a task and check initialization actions
print("\nTest 2: Checking task initialization actions...")
tasks = get_tasks('base')
mobile_data_tasks = [t for t in tasks if 'mobile_data' in t.id]
if mobile_data_tasks:
    task = mobile_data_tasks[0]
    print(f"  ✓ Found task: {task.id}")
    if task.initial_state and task.initial_state.initialization_actions:
        print(f"  ✓ Has {len(task.initial_state.initialization_actions)} initialization actions")
        for action in task.initial_state.initialization_actions:
            print(f"    - {action.env_type}.{action.func_name}()")

# Test 4: Try running initialization actions
print("\nTest 3: Running initialization actions...")
if mobile_data_tasks:
    task = mobile_data_tasks[0]
    if task.initial_state and task.initial_state.initialization_actions:
        try:
            for action in task.initial_state.initialization_actions:
                env.run_env_function_call(action)
            print("  ✓ All initialization actions completed successfully")
        except Exception as e:
            print(f"  ✗ Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

print("\n✅ All tests completed!")

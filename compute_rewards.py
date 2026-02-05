import json
import glob
import os
from collections import defaultdict


def compute_pass_k_from_files(file_paths, k=1, num_trials=4):
    """
    Compute Pass^k from τ²-Bench JSON output files.
    Assumes the JSON has a 'tasks' key with a list of task dicts. Each task dict includes:
    - 'id' or 'task_id': str (unique task identifier)
    - Optionally 'trial': int (trial number, e.g., 0 to 3)
    - 'reward_info': dict with 'reward': float (0.0 or 1.0)
    Groups by task_id, collects rewards (expects up to num_trials per task), and computes the fraction of tasks with at least k successes.
    
    :param file_paths: list of str, paths to JSON files (or use glob pattern)
    :param k: int, for Pass^k
    :param num_trials: int, expected number of trials per task (default 4 as per benchmark)
    :return: float, Pass^k score (0.0 to 1.0)
    """
    if isinstance(file_paths, str):
        file_paths = glob.glob(file_paths)
    
    rewards_per_task = defaultdict(list)
    
    for file_path in file_paths:
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found.")
            continue
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except:
            print("error.....")
            continue
        
        if 'simulations' in data:
            for task in data['simulations']:
                task_id = task.get('id') or task.get('task_id')
                if task_id is None:
                    continue
                if 'reward_info' in task and 'reward' in task['reward_info']:
                    reward = task['reward_info']['reward']
                    rewards_per_task[task_id].append(reward)
    
    if not rewards_per_task:
        return 0.0
    
    num_tasks = len(rewards_per_task)
    successful_tasks = 0
    
    for task_id, task_rewards in rewards_per_task.items():
        if len(task_rewards) != num_trials:
            print(f"Warning: Task {task_id} has {len(task_rewards)} trials, expected {num_trials}.")
        num_successes = sum(1 for r in task_rewards if r == 1.0)
        if num_successes >= k:
            successful_tasks += 1
    
    pass_k = successful_tasks / num_tasks
    return pass_k, num_tasks

responses = []
main_folder = 'data/simulations/'
file_paths = [main_folder + filename for filename in os.listdir(main_folder) if filename.endswith(".json")]
for filepath in file_paths:
    try:
        pass_at_1, nums = compute_pass_k_from_files([filepath], k=1, num_trials=1)
    except:
        continue
    filepath = os.path.abspath(filepath)
    responses.append(f"{filepath} Pass@1: {pass_at_1 * 100:.2f}% [number of points: {nums}]")

responses = sorted(responses)

for r in responses:
    print(r)


import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from tqdm import tqdm

def run_script(script_command):
    """
    Runs a single bash script command.
    Returns a tuple: (command, return_code, stdout, stderr)
    """
    print(f"Starting: {script_command}")
    
    # subprocess.run is the modern way to invoke scripts
    result = subprocess.run(
        script_command,
        shell=True,          # Allows using string commands like "bash script.sh arg1"
        capture_output=True, # Captures the output so it doesn't mess up your console
        text=True            # Returns output as strings instead of bytes
    )
    return script_command, result.returncode, result.stdout, result.stderr


agent_llm = "gpt-5-mini"
user_llm  = "gpt-5.1"

num_trials = 1
max_concurrency = 10
num_tasks = 30


suites = ['telecom', 'airline', 'retail']

base_cmd = "tau2 run --domain {domain} --agent-llm {agent} --user-llm {user} --num-trials {ntrials} --max-concurrency {nconc} --num-tasks {ntasks} --save-to {saveto} --attack-config '{atkconfig}' --enforce-communication-protocol"

configs = []

for user_azure_direct in [True, False]:
    configs.append((f"azuredirect_{user_azure_direct}", {'user_direct_model': user_azure_direct}))

commands = []
for suite in suites:
    for config_name, config in configs:

        _saveto = config_name + f"_{suite}"
        _config = json.dumps(config)

        _cmd = base_cmd.format(
            domain  = suite,
            agent   = agent_llm,
            user    = user_llm,
            ntrials = num_trials,
            nconc   = max_concurrency,
            ntasks  = num_tasks,
            saveto    = _saveto,
            atkconfig = _config,
        )
        commands.append(_cmd)

print("Overall commands", len(commands))

errors = 0
oks = 0
max_scripts = 1
with ThreadPoolExecutor(max_workers=max_scripts) as executor:
    future_to_script = {executor.submit(run_script, script): script for script in commands}
    
    for future in tqdm(as_completed(future_to_script)):
        script = future_to_script[future]
        try:
            cmd, rc, out, err = future.result()
            if rc == 0:
                oks += 1
            else:
                errors += 1
                
        except Exception as exc:
            print(f"💥 EXCEPTION for {script}: {exc}")

print("All scripts processed.")
print("Oks:", oks, "Errors:", errors)








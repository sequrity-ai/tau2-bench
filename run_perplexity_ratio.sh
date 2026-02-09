#!/bin/bash
set -e

cd /home/ubuntu/tau2-bench
source .venv/bin/activate

# Retail
python scripts/perplexity_ratio.py \
  --simulation-file "data/simulations/2026-02-07T20:59:00.743831_retail_llm_agent_gpt-5-mini_user_simulator_gpt-5-mini.json" \
  --output results/ppl_retail.json

# Telecom
python scripts/perplexity_ratio.py \
  --simulation-file "data/simulations/2026-02-07T22:32:38.484704_telecom_llm_agent_gpt-5-mini_user_simulator_gpt-5-mini.json" \
  --output results/ppl_telecom.json

# Telecom-workflow
python scripts/perplexity_ratio.py \
  --simulation-file "data/simulations/2026-02-08T00:08:39.797445_telecom-workflow_llm_agent_gpt-5-mini_user_simulator_gpt-5-mini.json" \
  --output results/ppl_telecom_workflow.json

#!/bin/bash
set -e

cd /home/ubuntu/tau2-bench
source .venv/bin/activate

# Retail
python scripts/perplexity_ratio.py \
  --simulation-file "data/simulations/2026-02-08T09:57:28.216107_retail_llm_agent_gpt-5-mini_user_simulator_gpt-5-mini.json" \
  --output results/ppl_retail.json

# Telecom
python scripts/perplexity_ratio.py \
  --simulation-file "data/simulations/2026-02-08T12:43:08.011779_telecom_llm_agent_gpt-5-mini_user_simulator_gpt-5-mini.json" \
  --output results/ppl_telecom.json

# Telecom-workflow
python scripts/perplexity_ratio.py \
  --simulation-file "data/simulations/2026-02-08T15:01:06.429659_telecom-workflow_llm_agent_gpt-5-mini_user_simulator_gpt-5-mini.json" \
  --output results/ppl_telecom_workflow.json
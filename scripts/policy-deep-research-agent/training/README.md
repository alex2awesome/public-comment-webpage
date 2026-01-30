# Training Harness

This folder wires the policy deep research OpenEnv environment into a TRL GRPO loop. The scripts intentionally avoid hiding details so you can iterate quickly.

1. Build the environment Docker image (from repo root):
   ```bash
   cd utils/envs/policy_deep_research_env
   openenv build
   ```
2. Start GRPO (defaults assume a locally tagged `openenv-policy-deep-research-env:latest`):
   ```bash
   cd training
   pip install -r requirements.txt
   python run_grpo.py --model meta-llama/Llama-3.1-8B-Instruct --use-cached \
     --cache-path ../data/cache/policy_cache.sqlite --task-index 0
   ```

Key flags:
- `--use-cached/--no-use-cached` toggles whether Semantic Scholar calls are allowed.
- `--cache-path` determines the SQLite DB location on the host. When you launch the env via Docker, the script prints instructions for mounting that file into the container.
- `--task-index` forces a deterministic episode; omit it to iterate through the shuffled task list.
- `--env-url` lets you target a long-lived server instead of spinning up Docker per run.

`run_grpo.py` delegates to `rollout_openenv.rollout_openenv`, which handles prompting the LLM, enforcing the JSON schema, and returning `(prompt, completion, reward)` tuples for TRL's GRPO trainer. The agent should emit reasoning/tool calls in the DRTulu format: `<think>...</think><action>{...JSON action...}</action>`.

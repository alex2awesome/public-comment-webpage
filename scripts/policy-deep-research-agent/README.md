# Policy Deep Research Agent

This repository packages a "Deep Research" style OpenEnv environment plus a lightweight TRL GRPO training harness. The agent reasons over policy questions by searching Semantic Scholar, fetching details, curating a bibliography, and submitting a final memo that earns a scalar reward.

## Layout
```
policy-deep-research-agent/
├─ data/                         # tasks + optional cache volume
├─ policy_src/                   # LangGraph agent, prompts, core reward/cache helpers, tests
├─ utils/envs/policy_deep_research_env/  # OpenEnv environment (server + client)
├─ training/                     # GRPO + AgentLightning scripts
└─ eval/                         # offline scoring utilities
```

## Prerequisites
- Python 3.11+
- Docker (needed for `openenv build` / `openenv run`)
- Semantic Scholar API key (optional but recommended). Copy `.env.example` to `.env` and fill in `S2_API_KEY`.

## Environment lifecycle
1. Install OpenEnv tooling:
   ```bash
   pip install "git+https://github.com/meta-pytorch/OpenEnv.git"
   pip install openenv-core
   ```
2. Build and validate the env image:
   ```bash
   cd utils/envs/policy_deep_research_env
   openenv build                      # builds docker image openenv-policy-deep-research-env:latest
   openenv validate --verbose
   ```
3. Run locally (optional):
   ```bash
   docker run -p 8000:8000 \
     -e OPENENV_USE_CACHED=1 \
     -e S2_API_KEY=$S2_API_KEY \
     openenv-policy-deep-research-env:latest
   ```

The environment exposes reset/step/state endpoints, serves FastAPI docs at `/docs`, and persists Semantic Scholar responses in `/app/env/data/cache.sqlite`.

## Training (TRL GRPO)
1. Install deps: `cd training && pip install -r requirements.txt`
2. Run the harness (collect rollouts + fine-tune):
   ```bash
   python run_grpo.py --model meta-llama/Llama-3.1-8B-Instruct --use-cached \
     --cache-path ../data/cache/policy_cache.sqlite --task-index 0 --training-steps 2
   ```

`run_grpo.py` will:
- Load the prompts in `policy_src/policy_prompts/`
- Spin up the OpenEnv Docker image (or reuse `--env-url`)
- Generate `rollouts_per_batch` episodes via `rollout_openenv.py`
- Feed the cached (prompt, transcript, reward) tuples into `trl.GRPOTrainer`
- Save checkpoints under `data/checkpoints/` and append rollouts to `data/cache/rollouts.jsonl`

Mount `--cache-path` into the container if you want Semantic Scholar responses to persist across runs: `docker run -v $(pwd)/data/cache/policy_cache.sqlite:/app/env/data/cache.sqlite ...`.

## Offline evaluation
Record memos or reuse `data/cache/rollouts.jsonl`, then:
```bash
python eval/offline_eval.py --trajectories data/cache/rollouts.jsonl --tasks data/tasks/policy_questions.jsonl
```
The script recomputes the deterministic environment reward so you can compare checkpoints without calling the API.

## Key environment features
- Typed `ResearchAction/Observation/State` models
- FastAPI server via `create_fastapi_app`
- Semantic Scholar connector with resilient backoff and SQLite caching (see `policy_src/policy_research_core/cache_db.py`)
- Deterministic task loader + reward shaping in `policy_src/policy_research_core/reward.py`
- HTTP client (`PolicyDeepResearchEnv`) for training + eval loops

See individual READMEs under `utils/envs/`, `training/`, and `eval/` for more detail.

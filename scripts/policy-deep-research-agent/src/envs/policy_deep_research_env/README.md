# Policy Deep Research OpenEnv

This environment lets an LLM agent iteratively search Semantic Scholar, fetch paper details, curate a bibliography, take notes, and submit a final policy memo that receives a deterministic reward.

## Quick start
```bash
pip install openenv-core
cd src/envs/policy_deep_research_env
openenv build                            # builds openenv-policy-deep-research-env:latest
openenv validate --verbose
```
Run locally:
```bash
docker run -p 8000:8000 \
  -e OPENENV_USE_CACHED=1 \
  -e S2_API_KEY=$S2_API_KEY \
  openenv-policy-deep-research-env:latest
```

## Actions
`ResearchAction` supports the following verbs:
- `SEARCH SEMANTIC SCHOLAR`: requires `query`, optional `top_k` and filters (e.g., `{ "year": "2021-" }`). Returns cached or live Semantic Scholar results.
- `FETCH PAPER`: requires `paper_id`, returns full metadata (authors, abstract, etc.).
- `ADD TO BIB`: records a paper plus `metadata.reason` inside the agent’s bibliography.
- `WRITE NOTE`: stores free-form notes for later reflection.
- `SUBMIT`: attaches the final memo in `content` and ends the episode.

Observations include the task question, instructions, outstanding bibliography, latest tool result, and remaining budget. Rewards come only from `SUBMIT` and summarize citation coverage, diversity, memo coverage, and a step-budget penalty.

## Cache + Semantic Scholar
`server/semanticscholar_api.py` handles REST calls (including 429 backoff). `server/cache_db.py` keeps a replicable SQLite cache with:
- `queries` / `query_results` tables for deterministic paper ordering
- `papers` table mirroring metadata snapshots
- `runs` for bookkeeping

Environment variables:
- `OPENENV_USE_CACHED` (default `0`): if `1`, never re-query when cache contains the requested entity.
- `OPENENV_CACHE_PATH` (default `/app/env/data/cache.sqlite`): SQLite location inside the container.
- `OPENENV_TASK_INDEX` + `OPENENV_TASK_SEED`: deterministic task selection.
- `MAX_STEPS`: per-episode budget.
- `S2_API_KEY`: optional Semantic Scholar key for higher QPS.

Sample tasks live in `data/tasks/policy_questions.jsonl`. You can override with `TASKS_PATH`.

## Client usage
```python
from src.envs.policy_deep_research_env import PolicyDeepResearchEnv, ResearchAction

env = PolicyDeepResearchEnv.from_docker_image(
    "openenv-policy-deep-research-env:latest",
    env_vars={"OPENENV_USE_CACHED": "1"},
)
reset = env.reset()
action = ResearchAction(type="SEARCH", query="carbon pricing resilience")
step = env.step(action)
print(step.observation.last_tool_result)
```

See `server/policy_deep_research_env_environment.py` for the exact logic.

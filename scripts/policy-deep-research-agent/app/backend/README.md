# Policy Rollout API

Minimal FastAPI service that powers the Vite console under `app/frontend`.

## Configuration

Create `app/backend/.env` (or set env vars / export `POLICY_BACKEND_ENV_FILE`) with your credentials:

```
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=ls-...
LANGSMITH_PROJECT=Policy LangGraph Rollout
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls-...
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
POLICY_AGENT_CACHE_PATH=data/cache/policy_cache.sqlite
POLICY_AGENT_MODEL=gpt-5-mini
POLICY_AGENT_USE_CACHED=true
```

The UI can also send an `Authorization: Bearer ...` header so you can avoid storing the OpenAI key on disk during local dev.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.backend.main:app --reload --port 8000 --host 0.0.0.0
```

Endpoints:

- `POST /rollouts` — body `{question, max_steps, enable_bibliography, temperature?}`. Returns an episode matching the LangGraph rollout schema.
- `GET /rollouts/stream?question=...` — Server-Sent Events stream that emits tool-call telemetry and finishes with the serialized episode.
- `POST /feedback` — body `{run_id, sentiment, note?}`. Requires `LANGSMITH_API_KEY`/`LANGSMITH_PROJECT` and forwards feedback to LangSmith.

With the backend running, the front-end (`app/frontend`) can point its `VITE_AGENT_API_URL` to `http://localhost:8000`.

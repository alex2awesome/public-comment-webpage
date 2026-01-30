# Policy Agent Console

Prototype UI for launching LangGraph policy-research rollouts and piping end-user feedback to LangSmith.

## Quick start

```bash
cd app/frontend
cp .env.example .env   # update VITE_AGENT_API_URL and OpenAI key if needed
npm install
npm run dev
```

The client expects the backend to expose:

- `GET /rollouts/stream` – Server-Sent Events endpoint that streams tool-call telemetry and finishes with the serialized episode.
- `POST /rollouts` – optional snapshot endpoint if you only need the final payload.
- `POST /feedback` – accepts `{ run_id, sentiment, note }` and forwards the payload to LangSmith via `client.create_feedback`.

Expose these endpoints via FastAPI / Cloud Run / App Engine. The UI reads `VITE_OPENAI_API_KEY` when present and forwards it as a `Bearer` token so you can proxy requests during early prototyping, but move API keys server-side before shipping.

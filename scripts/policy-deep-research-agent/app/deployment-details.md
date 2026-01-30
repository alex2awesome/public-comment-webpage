# Deployment & Operations Notes

## Architecture
- **Frontend**: `scripts/policy-deep-research-agent/app/frontend` (Vite/React) served from Netlify.
- **Backend**: `scripts/policy-deep-research-agent/app/backend` (FastAPI + SSE) hosted on Render at `https://public-comment-webpage.onrender.com`.
- Frontend calls the backend over HTTPS and streams rollout events from `/rollouts/stream`; feedback posts to `/feedback`.

---

## Backend (Render Web Service)
- **Service name / ID**: `public-comment-webpage` / `srv-d5u5auggjchc73beparg`.
- **Config file**: `render.yaml` at the repo root drives the Render blueprint.
- **Root directory**: `scripts/policy-deep-research-agent`.
- **Build command**: `pip install --upgrade pip && pip install -r app/backend/requirements.txt`.
- **Start command**: `uvicorn app.backend.main:app --host 0.0.0.0 --port $PORT`.
- **Plan**: Free tier (Oregon region). SSE is supported out of the box.
- **Key environment variables** (managed in Render dashboard unless a default is set in `render.yaml`):
  - `OPENAI_API_KEY`, `LANGSMITH_API_KEY`, `LANGCHAIN_API_KEY` – *secret tokens; never stored in git*.
  - `LANGSMITH_PROJECT="Policy LangGraph Rollout"`, `LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"`, `LANGCHAIN_TRACING_V2="true"`.
  - `POLICY_AGENT_USE_CACHED="false"`, `POLICY_AGENT_CACHE_PATH="/tmp/policy-cache.sqlite"` (per-run cache).
  - `VITE_AGENT_API_URL="https://public-comment-webpage.onrender.com"` to keep Netlify builds pointed at prod.
- **Health endpoints**: `/` (friendly message) and `/healthz` (plain `{"status": "ok"}`) respond on the Render URL for uptime checks.

### Render logs & API access
- Personal API token is stored locally at `~/.render-api-key.txt`. Keep this file out of version control.
- Example: fetch service metadata (already used to confirm the ID).
  ```bash
  RENDER_TOKEN=$(cat ~/.render-api-key.txt)
  curl -H "Authorization: Bearer $RENDER_TOKEN" \
       'https://api.render.com/v1/services?limit=20'
  ```
- Example: grab the latest application logs (adjust the `cursor`/`limit` params as needed).
  ```bash
  RENDER_TOKEN=$(cat ~/.render-api-key.txt)
  SERVICE_ID="srv-d5u5auggjchc73beparg"
  curl -H "Authorization: Bearer $RENDER_TOKEN" \
       "https://api.render.com/v1/services/${SERVICE_ID}/logs?type=application&limit=100"
  ```
- You can also install the Render CLI (`brew install render`) and run `render login` followed by `render logs public-comment-webpage`.

---

## Frontend (Netlify Site)
- **Site name / URL**: `chipper-brioche-aad078` → https://chipper-brioche-aad078.netlify.app (auto-deployed from `main`).
- **Config file**: `netlify.toml` at the repo root.
  - `base = "scripts/policy-deep-research-agent/app/frontend"`
  - `command = "npm run build"` (runs `tsc -b && vite build`)
  - `publish = "dist"`
  - Environment overrides: `NODE_VERSION=20`, `VITE_AGENT_API_URL=https://public-comment-webpage.onrender.com`.
- **Build source**: same GitHub repo root; Netlify uses subdirectory build thanks to the `base` setting.
- **Manual deploy trigger**: `curl -X POST -H "Authorization: Bearer $NETLIFY_TOKEN" https://api.netlify.com/api/v1/sites/<site_id>/deploys`.

### Netlify logs & API access
- Personal access token lives at `~/.netlify-token.txt`.
- Site ID: `0dd2a823-e05e-4bfc-8b4e-12f4b5bc5e5e`.
- Example: inspect the latest deploy metadata.
  ```bash
  NETLIFY_TOKEN=$(cat ~/.netlify-token.txt)
  curl -H "Authorization: Bearer $NETLIFY_TOKEN" \
       "https://api.netlify.com/api/v1/sites/0dd2a823-e05e-4bfc-8b4e-12f4b5bc5e5e/deploys"
  ```
- Example: stream build logs for a specific deploy.
  ```bash
  NETLIFY_TOKEN=$(cat ~/.netlify-token.txt)
  DEPLOY_ID="697c5bac0a05e409e8ec0853"  # replace with the ID you care about
  curl -H "Authorization: Bearer $NETLIFY_TOKEN" \
       "https://api.netlify.com/api/v1/sites/0dd2a823-e05e-4bfc-8b4e-12f4b5bc5e5e/deploys/${DEPLOY_ID}/log"
  ```
- Netlify CLI (`npm install -g netlify-cli`) can read the token automatically if it is exported as `NETLIFY_AUTH_TOKEN`.

---

## Local secret files
- `app/backend/.env` – holds runtime credentials for local FastAPI tests (OpenAI, LangSmith, etc.). **Do not commit**; already gitignored.
- `app/frontend/.env` – holds `VITE_AGENT_API_URL`/`VITE_OPENAI_API_KEY` overrides for local dev.
- Log/ops access tokens:
  - `~/.render-api-key.txt` – Render API token.
  - `~/.netlify-token.txt` – Netlify personal access token.
- Backups: there is no automated provisioning for these keys; keep secure copies in your password manager.

---

## Deployment checklist
1. **Backend** changes: push to `main` → Render auto-deploys from repo root using `render.yaml`. Use Render dashboard or `render deploy srv-d5u5auggjchc73beparg` for manual triggers.
2. **Frontend** changes: push to `main` → Netlify runs `npm run build` in `scripts/policy-deep-research-agent/app/frontend` and publishes `dist/`.
3. **Verify**:
   - Hit `https://public-comment-webpage.onrender.com/healthz` for backend.
   - Load https://chipper-brioche-aad078.netlify.app and run an end-to-end query.
   - Confirm LangSmith dashboard shows the new run and any submitted feedback.

Keep this file close to the code so future contributors can find every deployment knob in one place.

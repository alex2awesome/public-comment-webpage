# Repository Guidelines

## Project Structure & Module Organization
- `ai_corpus/` holds the harvest pipeline: connectors live in `ai_corpus/connectors/`, shared jobs in `ai_corpus/pipelines/`, and storage helpers under `ai_corpus/storage/`.
- `app/` contains the front-end client (React/Vite) that visualizes normalized comments, while `backend/` exposes the API and worker entry points.
- Operational assets live under `infra/` and `scripts/`; treat everything else (`demos/`, `notebooks/`, `scratch/`) as exploratory and avoid shipping production code from there.
- Keep generated data in `data/` and large downloads in `downloads/`. Anything under `data/comments` or `downloads/` may be huge, so avoid checking it in.

## Build, Test, and Development Commands
- Harvesting entry point: `python -m ai_corpus.cli.main pipeline --connector <name>` (use `discover`, `crawl`, `download-responses`, `extract`, `export` for step-by-step debugging).
- Run unit tests with `pytest` (default config targets Python 3.12); focus on connector-specific tests when iterating quickly, e.g. `pytest ai_corpus/tests/connectors/test_cppa_admt_connector.py`.
- Front-end workflow: `cd app && npm install && npm run dev` for local development, `npm run test` for component tests.
- Backend/API: `cd backend && pip install -r requirements.txt && pytest` plus `uvicorn backend.main:app --reload` for manual checks.

## Coding Style & Naming Conventions
- Follow PEP 8, keep modules type-annotated, and prefer `dataclass`es/`TypedDict`s when introducing structured payloads.
- Use descriptive connector IDs (`connector_slug:COLLECTION_ID`) and snake_case for files (e.g., `cppa_admt.py`). Keep CLI flags short and kebab-cased.
- Favor pure functions inside pipelines; when side effects are needed, isolate them behind helpers (e.g., `_fetch_html`).
- Front-end components use PascalCase files, hooks stay in `useSomething.ts`.

## Testing Guidelines
- All new connectors need fixture-backed unit tests under `ai_corpus/tests/connectors/` plus at least one integration crawl using the CLI.
- When adding pipeline stages, extend the existing pytest suites and add regression data in `tests/fixtures/`.
- Name tests `test_<scenario>` and keep them hermetic—mock network calls via `monkeypatch` or fixtures rather than touching the live web unless explicitly running smoke tests.
- For front-end/back-end changes, add matching Jest/React Testing Library or FastAPI tests before submitting.

## Commit & Pull Request Guidelines
- Write commits in imperative mood (“Add CPPA docket discovery”) and keep them focused; use multiple commits if you touch unrelated areas (connector + frontend).
- PRs need a concise summary, a list of major changes, linked issue IDs, screenshots or CLI logs for user-facing changes, and explicit test evidence (`pytest`, `npm test`, CLI runs).
- Highlight any manual steps (migrations, data backfill) in the PR description so reviewers can reproduce.

## Security & Configuration Tips
- Treat API keys (OpenAI, Playwright, etc.) as secrets—use env vars or `.env` files ignored by Git; never hard-code them in config.
- Large crawls can overwhelm external sites; respect built-in rate limiters and avoid raising worker counts without confirming with ops.

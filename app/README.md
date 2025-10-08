# Frontend (React + Vite)

This `app/` workspace hosts the Regulatory Intelligence UI. It is scaffolded with React + Vite and reads real CSV snapshots from the repository `data/` folder through a mock API layer so you can iterate on the experience before wiring a live Flask backend.

## Getting Started

```bash
cd app
npm install
npm run dev
```

The dev server runs at <http://localhost:5173>. Search, dashboard, and rule detail views all source their data from the CSV snapshots via `src/services/mockApi.ts`, which parses the files with `d3-dsv` and exposes the same contracts the Flask service will.

> **Heads-up on Node versions:** Some dependencies (Vite, React Router) recommend Node 20+. Running on Node 18 works, but you will see engine warnings during `npm install`. Upgrading Node will silence the warnings and unlock the latest tooling features.

### Available Scripts

- `npm run dev` – start the Vite development server with hot module reload.
- `npm run build` – type-check and produce a production build.
- `npm run lint` – run ESLint on the project.
- `npm run test -- --run` – execute Vitest in CI mode (tests are configured with jsdom and Testing Library).

## Project Structure

- `src/routes/router.tsx` wires the main application shell and routes (`/`, `/search`, `/dashboard`, `/rules/:ruleId`).
- `src/features/search/*` implements search UI, filters, and results backed by `useSearchRules` and CSV-derived metadata (`useSearchMetadata`).
- `src/features/recommendations/*` renders the personalized dashboard with scoring based on user interests and the CSV dataset.
- `src/features/rule-detail/*` shows detailed regulation context plus related items.
- `src/services/mockApi.ts` simulates backend endpoints, loading data from `../data/*.csv` via `src/services/csvData.ts`.
- `src/lib/` holds shared utilities (query client, date formatting, type definitions).
- `src/tests/` configures Vitest and includes a starter hook test.

Tailwind CSS powers styling (see `tailwind.config.js`), and TanStack Query handles client-side data caching and loading states.

When you begin connecting to a real API, swap the implementations in `src/services/*.ts` for `fetch`/Axios calls while preserving the hooks and components to keep the UI contract intact.

# Deployment and operations

## Docker Compose

Compose runs three services:

- `postgres`: PostgreSQL 16 with a named volume and a `pg_isready` healthcheck.
- `backend`: waits for a healthy database, runs `alembic upgrade head`, then
  starts the API. Runs as a non-root user with a `/health` healthcheck.
- `frontend`: a production Vite build served by Nginx. It starts after the
  backend is healthy.

Published ports bind to `127.0.0.1` only.

```powershell
Copy-Item .env.example .env      # set a unique, URL-safe POSTGRES_PASSWORD
docker compose up --build --wait # returns once every healthcheck passes
docker compose ps
docker compose logs backend
docker compose down              # keeps data; add -v to delete the database volume
```

The dashboard bundle is built with `VITE_API_URL=http://localhost:8000`, the
address the browser uses. The assistant defaults to the deterministic provider.
Set `CAREOPS_LLM_PROVIDER` and `OPENAI_API_KEY` in `.env` to enable rephrasing.
Never commit `.env`.

A container being up does not mean it is healthy. Check
`http://localhost:8000/ready`, which returns 503 until the database is
reachable and a completed pipeline run is present. The image contains the
committed pipeline outputs, so a fresh stack is ready without retraining.

## Local processes

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
Set-Location frontend; npm ci; $env:VITE_API_URL = "http://localhost:8000"; npm run dev
```

For PostgreSQL, set `CAREOPS_DATABASE_URL` and run the Alembic migration first
(see [database.md](database.md)). If the dashboard runs on another origin, set
`CAREOPS_CORS_ORIGINS`.

## Continuous integration

`.github/workflows/ci.yml` runs on every push and pull request. It has six
jobs:

1. **Python:** pytest across data, model, optimization and API tests, plus an
   Alembic drift check.
2. **PostgreSQL:** integration tests against a PostgreSQL 16 service. The job
   fails if they are skipped.
3. **Pipeline smoke run:** a 25% synthetic scenario to 2025-06-30, with every
   stage and gate.
4. **Frontend:** Node 24 type check, Vitest and production build.
5. **Browser end-to-end:** Playwright opens every dashboard page in Chromium,
   first against the live API and then against the static demo build. It
   fails on console errors and uploads screenshots.
6. **Docker Compose:** builds and starts the stack, waits on healthchecks,
   calls the API, assistant and dashboard, restarts the API and checks that
   simulation history survived.

No secrets are needed. The workflows have read-only repository permissions
(Pages has its own), and the Compose job generates a throwaway database
password.

## Public demo (GitHub Pages)

`.github/workflows/pages.yml` publishes a static snapshot of the dashboard on
every push to `master`:

1. `deploy/export_demo_snapshot.py` calls the real FastAPI app in-process for
   every GET path the dashboard reads (listed in
   `frontend/src/demo-config.json`), the nine assistant example questions and
   the four simulator presets. It writes the responses to
   `frontend/public/demo/`.
2. `npm run build:demo` builds with `VITE_DEMO_MODE=snapshot` and base
   `/CareOps-AI/`, from `frontend/.env.demo`. In this mode the dashboard reads
   those files, uses hash routes because Pages cannot rewrite deep links, and
   shows a banner saying it is a static snapshot.
3. The site is deployed to `https://krishna6116.github.io/CareOps-AI/`.

What the demo does not do: answer free-form questions, solve arbitrary
scenarios, or write alerts. These need the live API, and the page explains
this when you try them. The repository setting *Settings → Pages → Source*
must be *GitHub Actions*.

A live public API would need a hosting account (Render, Fly.io or similar)
with managed PostgreSQL. To set one up:

1. Set `CAREOPS_DATABASE_URL`, `CAREOPS_CORS_ORIGINS`,
   `CAREOPS_ENVIRONMENT=production` and `CAREOPS_API_KEY`.
2. Build the frontend with `VITE_API_URL` set to the public API URL.

Admin endpoints then require the key, and the solver endpoints are rate
limited (see [api.md](api.md)).

# Deployment and operations

Docker Compose defines PostgreSQL, the FastAPI backend, and a production-built
React frontend served by Nginx. PostgreSQL has a persistent named volume; the
backend waits for its health check, applies Alembic migrations, and exposes its
own health check before the frontend starts. Published ports bind to localhost
by default.

## Docker Compose

```powershell
Copy-Item .env.example .env
notepad .env
docker compose up --build
```

Replace `POSTGRES_PASSWORD` in `.env` with a unique URL-safe local password
before startup. Compose builds the backend database URL from that value. Never
commit `.env` or place production credentials in source control.
The dashboard bundle is built with `VITE_API_URL=http://localhost:8000` by
default, which is the browser-visible API address for a local Compose run.

## Local processes

From the repository root, run the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Then run the frontend as described in [frontend.md](./frontend.md). For a local
PostgreSQL backend, set `CAREOPS_DATABASE_URL` and
`CAREOPS_CORS_ORIGINS=["http://localhost:5173"]` in the process environment.
The default local database is SQLite.

## URLs

- Dashboard: `http://localhost:5173`
- API: `http://localhost:8000`
- API documentation: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

Docker runtime validation depends on Docker Desktop/Engine being available.

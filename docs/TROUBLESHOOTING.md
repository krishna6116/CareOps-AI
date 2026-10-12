# Troubleshooting

## Python and imports

**`ModuleNotFoundError: No module named 'backend'` (or `ml`, `optimization`).**
Run commands from the repository root. `pytest.ini` sets `pythonpath = .` and
lists both test folders, so `python -m pytest` works from the root without
extra paths. Use `python -m uvicorn backend.app.main:app` rather than running
`main.py` directly.

**`ImportError: DLL load failed ... An Application Control policy has blocked
this file`** (seen with scipy and xgboost on Windows 11). Smart App Control or
WDAC sometimes blocks a compiled extension the first time it loads. Run the
command again. If it keeps failing, check *Windows Security → App & browser
control*, or recreate the virtual environment.

**Wrong Python version.** The project is pinned to Python 3.14
(`.python-version`, CI, Docker), and `requirements-lock.txt` was resolved for
it. Create the venv with that interpreter:
`py -3.14 -m venv .venv`, then
`.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt`.

**`python run_portfolio.py` takes a long time.** A full run takes roughly
10–15 minutes on a laptop; no-show model training is the longest step. For a
quick check, use the CI smoke scenario
(`$env:CAREOPS_SCENARIO_SCALE="0.25"`, `--as-of-date 2025-06-30`) and do not
commit its outputs.

**Git shows `LF will be replaced by CRLF`.** This is a harmless Windows
line-ending notice.

## API

**The dashboard shows "Could not load this data".** Check that the API is
running at `VITE_API_URL` (default `http://localhost:8000`) and open
`/health`. A browser CORS error means the dashboard origin is missing from
`CAREOPS_CORS_ORIGINS`.

**`/ready` returns 503.** `checks.pipeline_run: missing` means there is no
`portfolio_run_manifest.json`, so run the pipeline. `checks.database:
unavailable` means the database URL or server is wrong.

**The assistant answer has a `fallback_reason`.** The configured LLM provider
was not used, for example because `OPENAI_API_KEY` is not set, Ollama is not
running, or the provider produced a number not in the evidence. The answer is
still the deterministic grounded one. Set `CAREOPS_LLM_PROVIDER=deterministic`
to stop the attempts.

## Database

**PostgreSQL connection errors.** Use the psycopg 3 URL form
`postgresql+psycopg://USER:PASSWORD@HOST:5432/DB`. URL-encode special
characters in the password.

**`relation ... does not exist`.** Apply the migrations:
`python -m alembic -c backend/alembic.ini upgrade head`. This uses
`CAREOPS_DATABASE_URL`.

**`alembic check` reports changes.** A model in
`backend/app/models/careops.py` changed without a migration. Generate one with
`alembic revision --autogenerate -m "..."`, review it, and commit it.

**The PostgreSQL tests are skipped.** Set `CAREOPS_TEST_DATABASE_URL` to a
disposable database. The tests create and clean up their own rows, but the
pipeline handoff writes a `careops` schema.

## Frontend

**`npm ci` warns `EBADENGINE`, or Vitest fails to start.** Use Node 24
(Vitest 5 and jsdom need Node 22.12 or later). Without a system Node, add a
portable Node 24 to `PATH` for the session.

**Counts show decimals.** Format them with `formatCount` from
`src/format.ts`. `format.test.ts` and `components.test.tsx` fail if counts
render with decimals.

## Docker

**`POSTGRES_PASSWORD` must be set.** Copy `.env.example` to `.env` and set a
unique, URL-safe password.

**The backend never becomes healthy.** Run `docker compose logs backend`.
Migration errors appear before Uvicorn starts. `docker compose down -v` resets
the database volume.

**A port is already in use (5432, 8000 or 5173).** Stop the local service
using it, or change the left-hand side of the port mapping in
`docker-compose.yml`.

# PostgreSQL and database design

The project uses SQLAlchemy models for a normalized CareOps schema with tables for patients, beds, staff, admissions, discharges, appointments, shifts, forecasts, availability, allocations, optimization runs, alerts, recommendations, simulations, model runs, and audit logs.

Use environment variables to set the database connection. The default development configuration is a local SQLite database; PostgreSQL is enabled via `CAREOPS_DATABASE_URL`.

## Initialization and migration

- `backend/app/database.py` initializes the mapped SQLAlchemy schema when the app starts.
- `backend/alembic/` contains the initial schema migration for all mapped entities.
- Docker Compose runs `alembic upgrade head` before starting the API.
- `CAREOPS_DATABASE_URL` should point to a Postgres instance in deployment.

## Operational notes

The portfolio pipeline continues to write canonical CSV artifacts in the
repository. Forecast, inventory, and optimizer read endpoints expose those
artifacts rather than recalculating them. The PostgreSQL handoff also loads
versioned portfolio artifacts into the `careops` schema. The API's SQLAlchemy
tables are in the database's default schema and are currently used for persisted
alerts and simulation records; the read endpoints do not claim that CSV-backed
forecast responses were queried from PostgreSQL.

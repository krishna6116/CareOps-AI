"""Persist a validated CareOps portfolio run to PostgreSQL."""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCHEMA = "careops"
ARTIFACTS = {
    "source_patients": "data/synthetic/patients.csv",
    "source_admissions": "data/synthetic/admissions.csv",
    "source_beds": "data/synthetic/beds.csv",
    "source_staff": "data/synthetic/staff.csv",
    "source_appointments": "data/synthetic/appointments.csv",
    "source_daily_operations": "data/synthetic/daily_operations.csv",
    "source_shifts": "data/synthetic/shifts.csv",
    "source_hospital_snapshot": "data/synthetic/hospital_snapshot.csv",
    "support_admission_isolation_labels": (
        "data/synthetic/resource_support/admission_isolation_labels.csv"
    ),
    "support_bed_inventory_snapshot": (
        "data/synthetic/resource_support/bed_inventory_snapshot.csv"
    ),
    "support_isolation_admission_forecast": (
        "data/synthetic/resource_support/isolation_admission_forecast.csv"
    ),
    "support_isolation_scenario_rates": (
        "data/synthetic/resource_support/isolation_scenario_rates.csv"
    ),
    "support_staff_availability_snapshot": (
        "data/synthetic/resource_support/staff_availability_snapshot.csv"
    ),
    "forecast_admissions_30_day": (
        "ml/admission_forecasting/admission_30_day_forecast.csv"
    ),
    "forecast_occupancy_30_day": (
        "ml/occupancy_forecasting/occupancy_future_forecast.csv"
    ),
    "forecast_icu_30_day": (
        "ml/icu_demand_forecasting/icu_future_forecast.csv"
    ),
    "forecast_los_predictions": "ml/los_prediction/los_predictions.csv",
    "forecast_no_show_predictions": (
        "ml/no_show_prediction/no_show_predictions.csv"
    ),
    "optimization_bed_availability": (
        "optimization/resource_planning/outputs/bed_availability_forecast.csv"
    ),
    "optimization_bed_allocation": (
        "optimization/resource_planning/outputs/bed_allocation_simulation.csv"
    ),
    "optimization_bed_kpis": (
        "optimization/resource_planning/outputs/bed_allocation_kpis.csv"
    ),
    "optimization_bed_assignment_requests": (
        "optimization/resource_planning/outputs/bed_assignment_requests.csv"
    ),
    "optimization_bed_assignment_daily": (
        "optimization/resource_planning/outputs/bed_assignment_daily.csv"
    ),
    "optimization_bed_assignment_kpis": (
        "optimization/resource_planning/outputs/bed_assignment_kpis.csv"
    ),
    "optimization_staff_schedule": (
        "optimization/resource_planning/outputs/optimized_staff_schedule.csv"
    ),
    "optimization_staff_kpis": (
        "optimization/resource_planning/outputs/staff_coverage_kpis.csv"
    ),
    "optimization_appointment_schedule": (
        "optimization/resource_planning/outputs/optimized_appointment_schedule.csv"
    ),
    "optimization_appointment_kpis": (
        "optimization/resource_planning/outputs/appointment_optimization_kpis.csv"
    ),
    "optimization_emergency_capacity": (
        "optimization/resource_planning/outputs/emergency_capacity_optimization.csv"
    ),
    "optimization_resource_gaps": (
        "optimization/resource_planning/outputs/resource_gap_analysis.csv"
    ),
    "optimization_capacity_expansion": (
        "optimization/resource_planning/outputs/capacity_expansion_plan.csv"
    ),
    "quality_data_audit": (
        "optimization/resource_planning/outputs/data_quality_audit.csv"
    ),
    "quality_forecast_backtest": (
        "optimization/resource_planning/outputs/forecast_backtest_summary.csv"
    ),
    "quality_reliability_scorecard": (
        "optimization/resource_planning/outputs/reliability_scorecard.csv"
    ),
    "quality_reliability_details": (
        "optimization/resource_planning/outputs/reliability_details.csv"
    ),
    "quality_admission_holdout_predictions": (
        "ml/admission_forecasting/admission_test_predictions.csv"
    ),
    "quality_occupancy_holdout_predictions": (
        "ml/occupancy_forecasting/occupancy_test_predictions.csv"
    ),
    "quality_icu_holdout_predictions": (
        "ml/icu_demand_forecasting/icu_demand_test_predictions.csv"
    ),
}


def load_pipeline_to_postgres(
    root: Path,
    manifest: dict[str, Any],
    database_url: str | None,
) -> dict[str, Any]:
    """Load curated pipeline outputs atomically into the ``careops`` schema."""
    if not database_url:
        raise RuntimeError(
            "PostgreSQL handoff requires CAREOPS_DATABASE_URL. "
            "No database connection was attempted."
        )
    if not database_url.startswith(("postgresql://", "postgresql+psycopg://")):
        raise ValueError(
            "CAREOPS_DATABASE_URL must use postgresql:// or "
            "postgresql+psycopg://."
        )

    try:
        from sqlalchemy import (
            JSON,
            Column,
            Date,
            Float,
            MetaData,
            String,
            Table,
            create_engine,
            inspect,
            text,
        )
        from sqlalchemy.dialects.postgresql import JSONB
        from sqlalchemy.exc import SQLAlchemyError
        from sqlalchemy.schema import CreateSchema
    except ImportError as exc:
        raise RuntimeError(
            "PostgreSQL support is not installed. Install "
            "requirements-postgresql.txt before using --postgres."
        ) from exc

    frames = {}
    row_counts = {}
    for table_name, relative_path in ARTIFACTS.items():
        artifact_path = root / Path(relative_path)
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"Required PostgreSQL artifact is missing: {artifact_path}"
            )
        frame = pd.read_csv(artifact_path)
        if frame.empty:
            raise ValueError(
                f"Required PostgreSQL artifact has no rows: {artifact_path}"
            )
        if "run_id" in frame.columns:
            raise ValueError(
                f"PostgreSQL artifact already contains reserved run_id column: "
                f"{artifact_path}"
            )
        run_id = str(manifest["scenario_as_of_date"])
        frame.insert(0, "run_id", run_id)
        frames[table_name] = frame
        row_counts[table_name] = len(frame)

    run_id = str(manifest["scenario_as_of_date"])
    integrated_at = datetime.now(timezone.utc).isoformat()
    integration_summary = {
        "status": "loaded",
        "schema": SCHEMA,
        "run_id": run_id,
        "integrated_at": integrated_at,
        "table_count": len(frames),
        "row_count": sum(row_counts.values()),
        "tables": row_counts,
    }

    engine = None
    try:
        engine = create_engine(database_url, pool_pre_ping=True)
        with engine.begin() as connection:
            connection.execute(CreateSchema(SCHEMA, if_not_exists=True))
            metadata = MetaData(schema=SCHEMA)
            runs_table = Table(
                "pipeline_runs",
                metadata,
                Column("run_id", String(10), primary_key=True),
                Column("scenario_as_of_date", Date, nullable=False),
                Column("pipeline_status", String(64), nullable=False),
                Column("overall_score", Float, nullable=False),
                Column("integrated_at", String(40), nullable=False),
                Column(
                    "tables",
                    JSON().with_variant(JSONB(), "postgresql"),
                    nullable=False,
                ),
            )
            metadata.create_all(connection)

            inspector = inspect(connection)
            for table_name in frames:
                if inspector.has_table(table_name, schema=SCHEMA):
                    connection.execute(
                        text(
                            f'DELETE FROM "{SCHEMA}"."{table_name}" '
                            "WHERE run_id = :run_id"
                        ),
                        {"run_id": run_id},
                    )
                frames[table_name].to_sql(
                    table_name,
                    connection,
                    schema=SCHEMA,
                    if_exists="append",
                    index=False,
                    chunksize=1000,
                    method="multi",
                    dtype={"run_id": String(10)},
                )
                connection.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS '
                        f'ix_{table_name}_run_id '
                        f'ON "{SCHEMA}"."{table_name}" (run_id)'
                    )
                )

            connection.execute(
                text(f'DELETE FROM "{SCHEMA}"."pipeline_runs" WHERE run_id = :run_id'),
                {"run_id": run_id},
            )
            connection.execute(
                runs_table.insert().values(
                    run_id=run_id,
                    scenario_as_of_date=date.fromisoformat(run_id),
                    pipeline_status=manifest["pipeline_status"],
                    overall_score=manifest["reliability_scorecard"]["overall_score"],
                    integrated_at=integrated_at,
                    tables=row_counts,
                )
            )
    except SQLAlchemyError:
        raise RuntimeError(
            "PostgreSQL handoff failed; the database transaction was rolled back."
        ) from None
    finally:
        if engine is not None:
            engine.dispose()

    return integration_summary

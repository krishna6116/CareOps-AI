"""Build explicit, reproducible synthetic support data for the resource planner.

Generated labels and snapshot dates are scenario fixtures, not clinical facts or
live operational data. They must never be copied into a production workflow.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PROCESSED = Path(
    os.environ.get("CAREOPS_DATA_DIR", ROOT / "data" / "processed")
)
FORECAST = ROOT / "ml" / "admission_forecasting" / "admission_30_day_forecast.csv"
OUTPUT = Path(__file__).resolve().parent
SEED = 42
ISOLATION_RULES = {
    "infectious_disease_probability": 0.65,
    "other_diagnosis_probability": 0.005,
    "prior_strength_admissions": 100,
}


def generate() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    operations = pd.read_csv(PROCESSED / "daily_operations.csv")
    snapshot_date = pd.to_datetime(operations["date"], errors="raise").max().date()

    beds = pd.read_csv(PROCESSED / "beds.csv")
    beds["as_of_date"] = snapshot_date.isoformat()
    beds["data_provenance"] = "synthetic_assumed_inventory_snapshot"
    beds.to_csv(OUTPUT / "bed_inventory_snapshot.csv", index=False)

    staff = pd.read_csv(PROCESSED / "staff.csv")
    staff["as_of_date"] = snapshot_date.isoformat()
    staff["data_provenance"] = "synthetic_static_staff_availability_assumption"
    staff.to_csv(OUTPUT / "staff_availability_snapshot.csv", index=False)

    admissions = pd.read_csv(PROCESSED / "admissions.csv")
    rng = np.random.default_rng(SEED)
    infectious = (
        admissions["diagnosis_category"].astype(str).str.casefold()
        == "infectious disease"
    )
    probabilities = np.where(
        infectious,
        ISOLATION_RULES["infectious_disease_probability"],
        ISOLATION_RULES["other_diagnosis_probability"],
    )
    labels = pd.DataFrame({
        "admission_id": admissions["admission_id"],
        "isolation_required": (rng.random(len(admissions)) < probabilities),
        "label_source": "synthetic_scenario_probability_not_clinical",
        "label_seed": SEED,
    })
    labels.to_csv(OUTPUT / "admission_isolation_labels.csv", index=False)

    historical = admissions[["admission_id", "department"]].merge(
        labels[["admission_id", "isolation_required"]],
        on="admission_id",
        how="left",
        validate="one_to_one",
    )
    summary = historical.groupby("department")["isolation_required"].agg(
        isolation_admissions="sum",
        total_admissions="count",
    )
    overall_rate = float(historical["isolation_required"].mean())
    prior_strength = ISOLATION_RULES["prior_strength_admissions"]
    summary["smoothed_isolation_rate"] = (
        summary["isolation_admissions"] + prior_strength * overall_rate
    ) / (summary["total_admissions"] + prior_strength)

    forecast = pd.read_csv(FORECAST)
    forecast["department_key"] = forecast["department"].astype(str).str.casefold().str.strip()
    rates = summary["smoothed_isolation_rate"].copy()
    rates.index = rates.index.astype(str).str.casefold().str.strip()
    forecast["historical_smoothed_isolation_rate"] = forecast["department_key"].map(rates)
    forecast.loc[
        forecast["department_key"] == "all departments",
        "historical_smoothed_isolation_rate",
    ] = overall_rate
    if forecast["historical_smoothed_isolation_rate"].isna().any():
        missing = sorted(forecast.loc[
            forecast["historical_smoothed_isolation_rate"].isna(), "department"
        ].unique())
        raise ValueError(f"No isolation scenario rate for forecast departments: {missing}")
    forecast["predicted_isolation_admissions"] = (
        pd.to_numeric(forecast["predicted_admissions"], errors="raise")
        * forecast["historical_smoothed_isolation_rate"]
    )
    forecast["isolation_forecast_source"] = "synthetic_historical_rate_scenario"
    forecast[[
        "date", "department", "predicted_isolation_admissions",
        "historical_smoothed_isolation_rate", "isolation_forecast_source",
    ]].to_csv(OUTPUT / "isolation_admission_forecast.csv", index=False)

    summary.reset_index().rename(columns={"department": "department"}).to_csv(
        OUTPUT / "isolation_scenario_rates.csv", index=False
    )
    manifest = {
        "data_scope": "synthetic_portfolio_fixture_only",
        "snapshot_date": snapshot_date.isoformat(),
        "seed": SEED,
        "isolation_rules": ISOLATION_RULES,
        "limitations": [
            "Isolation labels are synthetic probability draws, not clinical determinations.",
            "Bed and staff snapshot dates are assumed to match the latest source-history date.",
            "Static staff availability is not a live shift-level availability feed.",
            "Do not use any generated support data for patient care or real staffing decisions.",
        ],
        "outputs": [
            "bed_inventory_snapshot.csv",
            "staff_availability_snapshot.csv",
            "admission_isolation_labels.csv",
            "isolation_admission_forecast.csv",
            "isolation_scenario_rates.csv",
        ],
    }
    (OUTPUT / "scenario_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated synthetic resource-support fixtures at {OUTPUT}")
    print(f"Assumed snapshot date: {snapshot_date}")
    print(f"Isolation labels: {int(labels['isolation_required'].sum())}/{len(labels)}")
    print(
        "WARNING: scenario fixtures are synthetic assumptions, not clinical or live data."
    )


if __name__ == "__main__":
    generate()

import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from run_portfolio import (
    SOURCE_FILES,
    build_pipeline_readiness,
    default_as_of_date,
    validate_appointment_schedule,
    validate_forecast_artifacts,
    validate_source_data,
)
from ml.los_prediction.los_prediction import filter_completed_admissions
from optimization.postgresql_integration import (
    ARTIFACTS,
    SCHEMA,
    load_pipeline_to_postgres,
)


def valid_source_frames():
    return {
        "patients.csv": pd.DataFrame([{
            "patient_id": "p1",
            "patient_name": "Synthetic Person",
            "age": 42,
            "gender": "Other",
        }]),
        "admissions.csv": pd.DataFrame([{
            "admission_id": "a1",
            "patient_id": "p1",
            "patient_name": "Synthetic Person",
            "department": "General Medicine",
            "admission_date": "2026-01-01",
            "discharge_date": "2026-01-02",
            "LOS": 1,
            "ICU_required": 0,
            "assigned_bed_id": "b1",
            "stay_status": "in_progress",
        }]),
        "beds.csv": pd.DataFrame([{
            "bed_id": "b1",
            "department": "General Medicine",
            "bed_type": "General",
            "status": "Occupied",
            "isolation_capable": False,
            "emergency_reserve": False,
            "available_for_emergency": False,
        }]),
        "staff.csv": pd.DataFrame([{
            "staff_id": "s1",
            "department": "General Medicine",
            "role": "Doctor",
            "availability": "Available",
            "max_hours": 40,
        }]),
        "appointments.csv": pd.DataFrame([{
            "appointment_id": "ap1",
            "patient_id": "p1",
            "doctor_id": "s1",
            "department": "General Medicine",
            "date": "2026-01-01",
            "time": "09:00",
            "duration": 30,
            "no_show": 0,
        }]),
        "daily_operations.csv": pd.DataFrame([{
            "date": "2026-01-01",
            "department": "General Medicine",
            "admissions": 1,
            "discharges": 0,
            "ICU_admissions": 0,
            "occupancy": 100,
            "staff_demand": 0,
            "bed_demand": 1,
        }]),
        "shifts.csv": pd.DataFrame([{
            "shift_id": "sh1",
            "date": "2026-01-01",
            "staff_id": "s1",
            "shift": "Morning",
            "hours": 8,
            "assigned": "Yes",
        }]),
        "hospital_snapshot.csv": pd.DataFrame([{
            "snapshot_date": "2026-01-01",
            "total_beds": 1,
            "occupied_beds": 1,
            "available_beds": 0,
            "maintenance_beds": 0,
            "total_icu_beds": 0,
            "occupied_icu_beds": 0,
        }]),
    }


class PortfolioSourceValidationTests(unittest.TestCase):
    def write_source(self, directory, frames):
        for filename in SOURCE_FILES:
            frames[filename].to_csv(Path(directory) / filename, index=False)

    def test_valid_source_reconciles_encounters_census_and_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            self.write_source(directory, valid_source_frames())
            counts = validate_source_data(date(2026, 1, 1), Path(directory))

        self.assertEqual(counts["admissions.csv"], 1)
        self.assertEqual(counts["daily_operations.csv"], 1)

    def test_source_validation_rejects_daily_counts_out_of_sync(self):
        frames = valid_source_frames()
        frames["daily_operations.csv"].loc[0, "admissions"] = 0
        with tempfile.TemporaryDirectory() as directory:
            self.write_source(directory, frames)
            with self.assertRaisesRegex(ValueError, "admissions do not match"):
                validate_source_data(date(2026, 1, 1), Path(directory))

    def test_los_training_excludes_stays_not_discharged_at_cutoff(self):
        admissions = pd.DataFrame({
            "admission_id": ["completed", "in_progress"],
            "discharge_date": ["2026-01-01", "2026-01-02"],
            "stay_status": ["completed", "in_progress"],
        })
        filtered, excluded = filter_completed_admissions(
            admissions, pd.Timestamp("2026-01-01")
        )

        self.assertEqual(filtered["admission_id"].tolist(), ["completed"])
        self.assertEqual(excluded, 1)

    def test_appointment_gate_accepts_only_fully_scheduled_output(self):
        schedule = pd.DataFrame({"status": ["SCHEDULED", "SCHEDULED"]})

        self.assertEqual(validate_appointment_schedule(schedule), 2)

        with self.assertRaisesRegex(ValueError, "1 unassigned appointments"):
            validate_appointment_schedule(
                pd.DataFrame({"status": ["SCHEDULED", "UNSCHEDULED"]})
            )

    def test_forecast_validation_accepts_department_grain_dates(self):
        start = pd.Timestamp("2026-10-01")
        dates = pd.date_range(start, periods=30, freq="D")
        admission_forecast = pd.DataFrame([
            {"date": day, "department": department, "predicted_admissions": 1.0}
            for day in dates
            for department in ("General Medicine", "Surgery")
        ])
        forecasts = {
            "admission_forecasting": admission_forecast,
            "occupancy_forecasting": pd.DataFrame({
                "date": dates,
                "forecast_occupied_beds": [100.0] * 30,
            }),
            "icu_forecasting": pd.DataFrame({
                "date": dates,
                "predicted_icu_admissions": [1.0] * 30,
                "forecast_target_unit": ["admissions_per_day"] * 30,
            }),
        }

        with tempfile.TemporaryDirectory() as directory:
            forecast_paths = {}
            for name, frame in forecasts.items():
                path = Path(directory) / f"{name}.csv"
                frame.to_csv(path, index=False)
                forecast_paths[name] = path
            with patch("run_portfolio.FORECASTS", forecast_paths):
                result = validate_forecast_artifacts(date(2026, 9, 30))

        self.assertEqual(
            result["admission_forecasting"],
            ["2026-10-01", "2026-10-30"],
        )

    def test_readiness_preserves_no_show_rating_and_blocks_under_target_sections(self):
        scorecard = {
            "minimum_target": 4.0,
            "sections": [
                {"section": "No-show prediction", "score": 4.5},
                {"section": "Admission forecasting", "score": 3.0},
            ],
        }
        manifest = {
            "optimized_capacity_gates": {
                "overflow_bed_days": 0,
                "unfilled_staff_positions": 0,
                "icu_overflow_bed_days": 0,
                "emergency_overflow_bed_days": 0,
                "unassigned_appointments": 0,
            },
            "source_integrity_checks_passed": True,
            "audit_failures": 0,
        }
        no_show_metadata = {
            "model_version": "Enhanced Behavioral V2",
            "historical_features_are_leakage_safe": True,
            "source_data_modified": False,
        }

        readiness = build_pipeline_readiness(
            scorecard, manifest, no_show_metadata
        )

        self.assertEqual(readiness["pipeline_status"], "completed_review_required")
        self.assertFalse(readiness["next_process_ready"])
        self.assertEqual(readiness["review_required_sections"], ["Admission forecasting"])
        self.assertEqual(readiness["no_show_model_integration"]["score"], 4.5)
        self.assertFalse(readiness["ready_for_live_operational_use"])

    def test_readiness_requires_all_capacity_and_no_show_gates(self):
        scorecard = {
            "minimum_target": 4.0,
            "sections": [
                {"section": "No-show prediction", "score": 4.0},
                {"section": "Admission forecasting", "score": 4.0},
            ],
        }
        manifest = {
            "optimized_capacity_gates": {
                "overflow_bed_days": 0,
                "unfilled_staff_positions": 0,
                "icu_overflow_bed_days": 0,
                "emergency_overflow_bed_days": 1,
                "unassigned_appointments": 0,
            },
            "source_integrity_checks_passed": True,
            "audit_failures": 0,
        }
        no_show_metadata = {
            "model_version": "Enhanced Behavioral V2",
            "historical_features_are_leakage_safe": True,
            "source_data_modified": False,
        }

        readiness = build_pipeline_readiness(
            scorecard, manifest, no_show_metadata
        )

        self.assertFalse(readiness["next_process_ready"])
        self.assertIn(
            "Capacity gates failed: emergency_overflow_bed_days",
            readiness["next_process_blockers"],
        )
        self.assertEqual(
            readiness["no_show_model_integration"]["required_score"], 4.5
        )

    def test_postgres_handoff_requires_database_url(self):
        with self.assertRaisesRegex(RuntimeError, "CAREOPS_DATABASE_URL"):
            load_pipeline_to_postgres(Path("."), {}, None)

    def test_postgres_handoff_rejects_non_postgresql_url(self):
        with self.assertRaisesRegex(ValueError, "must use postgresql"):
            load_pipeline_to_postgres(Path("."), {}, "sqlite:///not-postgres.db")

    def test_postgres_artifact_registry_uses_safe_unique_table_names(self):
        self.assertEqual(SCHEMA, "careops")
        self.assertEqual(len(ARTIFACTS), len(set(ARTIFACTS)))
        self.assertTrue(all(name.replace("_", "").isalnum() for name in ARTIFACTS))
        self.assertTrue(all(not Path(path).is_absolute() for path in ARTIFACTS.values()))


class DefaultAsOfDateTests(unittest.TestCase):
    def test_default_is_the_last_date_in_existing_data(self):
        with tempfile.TemporaryDirectory() as directory:
            pd.DataFrame({
                "date": ["2026-10-03", "2026-10-05", "2026-10-04"],
                "department": ["ICU", "ICU", "ICU"],
            }).to_csv(Path(directory) / "daily_operations.csv", index=False)

            self.assertEqual(default_as_of_date(Path(directory)), date(2026, 10, 5))

    def test_missing_data_requires_an_explicit_date(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(SystemExit):
                default_as_of_date(Path(directory))


if __name__ == "__main__":
    unittest.main()

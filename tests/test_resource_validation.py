import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from optimization.resource_validation import audit_resource_inputs, build_forecast_backtest


def valid_inputs():
    return {
        "beds": pd.DataFrame([{
            "bed_id": "b1", "department": "General Medicine",
            "bed_type": "General", "status": "Available",
        }]),
        "staff": pd.DataFrame([{
            "staff_id": "s1", "availability": "Available",
        }]),
        "shifts": pd.DataFrame([{
            "shift_id": "sh1", "date": "2026-01-01",
            "department": "General Medicine",
        }]),
        "appointments": pd.DataFrame([{
            "appointment_id": "a1", "date": "2026-01-01",
        }]),
        "admissions": pd.DataFrame([{
            "admission_id": "ad1", "admission_date": "2025-12-30",
            "discharge_date": "2026-01-01",
        }]),
        "daily_operations": pd.DataFrame([{
            "date": "2026-01-01", "department": "General Medicine",
        }]),
        "admission_forecast": pd.DataFrame([{
            "date": "2026-01-02", "department": "General Medicine",
            "predicted_admissions": 2,
        }]),
        "isolation_forecast": pd.DataFrame([{
            "date": "2026-01-02", "department": "General Medicine",
            "predicted_isolation_admissions": 0.1,
            "historical_smoothed_isolation_rate": 0.05,
        }]),
        "occupancy_forecast": pd.DataFrame([{
            "date": "2026-01-02", "forecast_occupied_beds": 1,
            "forecast_occupancy_rate": 1,
        }]),
        "icu_forecast": pd.DataFrame([{
            "date": "2026-01-02", "predicted_icu_demand": 0,
            "predicted_icu_admissions": 0, "forecast_target_unit": "admissions_per_day",
        }]),
        "los_predictions": pd.DataFrame([{"predicted_los_days": 4}]),
        "no_show_predictions": pd.DataFrame([{
            "appointment_id": "a1", "predicted_no_show_probability": 0.2,
        }]),
    }


class InputAuditTests(unittest.TestCase):
    def test_audit_flags_stale_and_undated_inventory_without_calling_it_live(self):
        result = audit_resource_inputs(valid_inputs(), run_date=date(2026, 1, 4))
        checks = result.set_index("check")
        self.assertEqual(checks.loc["latest_observed_operations_age_days", "status"], "WARN")
        self.assertEqual(checks.loc["forecast_start_age_days", "status"], "WARN")
        self.assertEqual(checks.loc["bed_inventory_snapshot_date", "status"], "WARN")
        self.assertFalse((result["status"] == "FAIL").any())

    def test_audit_fails_duplicate_beds_and_negative_forecasts(self):
        inputs = valid_inputs()
        inputs["beds"] = pd.concat([inputs["beds"], inputs["beds"]], ignore_index=True)
        inputs["admission_forecast"].loc[0, "predicted_admissions"] = -1
        result = audit_resource_inputs(inputs, run_date=date(2026, 1, 2))
        checks = result.set_index("check")
        self.assertEqual(checks.loc["beds_duplicate_bed_id", "status"], "FAIL")
        self.assertEqual(checks.loc["admission_forecast_numeric_values", "status"], "FAIL")

    def test_freshness_gate_blocks_stale_and_undated_operational_inputs(self):
        result = audit_resource_inputs(
            valid_inputs(), run_date=date(2026, 1, 4), require_fresh=True
        )
        checks = result.set_index("check")
        self.assertEqual(checks.loc["latest_observed_operations_age_days", "status"], "FAIL")
        self.assertEqual(checks.loc["forecast_start_age_days", "status"], "FAIL")
        self.assertEqual(checks.loc["bed_inventory_snapshot_date", "status"], "FAIL")

    def test_audit_detects_forecast_horizon_gaps(self):
        inputs = valid_inputs()
        inputs["admission_forecast"] = pd.concat([
            inputs["admission_forecast"],
            pd.DataFrame([{
                "date": "2026-01-03", "department": "General Medicine",
                "predicted_admissions": 2,
            }]),
        ], ignore_index=True)
        result = audit_resource_inputs(inputs, run_date=date(2026, 1, 2))
        checks = result.set_index("check")
        self.assertEqual(checks.loc["icu_forecast_horizon_alignment", "status"], "FAIL")
        self.assertEqual(
            checks.loc["icu_forecast_horizon_alignment", "value"],
            "1 missing / 0 extra",
        )
        self.assertEqual(
            checks.loc["occupancy_forecast_horizon_alignment", "status"], "FAIL"
        )

    def test_audit_rejects_invalid_no_show_probabilities_and_duplicate_scores(self):
        inputs = valid_inputs()
        inputs["no_show_predictions"] = pd.DataFrame([
            {"appointment_id": "a1", "predicted_no_show_probability": 1.2},
            {"appointment_id": "a1", "predicted_no_show_probability": 0.4},
        ])
        checks = audit_resource_inputs(
            inputs, run_date=date(2026, 1, 2)
        ).set_index("check")
        self.assertEqual(checks.loc["no_show_prediction_duplicate_ids", "status"], "FAIL")
        self.assertEqual(checks.loc["no_show_prediction_probability_bounds", "status"], "FAIL")

    def test_audit_rejects_icu_admission_forecast_without_explicit_units(self):
        inputs = valid_inputs()
        inputs["icu_forecast"]["forecast_target_unit"] = "beds"
        result = audit_resource_inputs(inputs, run_date=date(2026, 1, 2))
        check = result.set_index("check").loc["icu_forecast_target_semantics"]
        self.assertEqual(check["status"], "FAIL")


class ForecastBacktestTests(unittest.TestCase):
    def test_saved_admission_holdout_is_scored_against_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = pd.DataFrame({
                "date": ["2026-01-01", "2026-01-02"],
                "department": ["Cardiology", "Cardiology"],
                "actual_admissions": [10, 10],
                "ml_prediction": [9, 11],
                "baseline_prediction": [12, 12],
            })
            path = root / "admissions.csv"
            predictions.to_csv(path, index=False)
            result = build_forecast_backtest(root, {"admissions": path})
            department = result[result["group"] == "Cardiology"].iloc[0]
            self.assertEqual(department["model"], "admissions_xgboost")
            self.assertEqual(department["test_rows"], 2)
            self.assertEqual(department["mae"], 1)
            self.assertEqual(department["baseline_mae"], 2)
            self.assertTrue(department["beats_baseline"])

    def test_no_show_backtest_uses_brier_score_and_prevalence_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            predictions = pd.DataFrame({
                "appointment_datetime": ["2026-01-01 09:00", "2026-01-02 10:00"],
                "actual_no_show": [0, 1],
                "predicted_no_show_probability": [0.1, 0.9],
            })
            path = root / "no_show.csv"
            predictions.to_csv(path, index=False)
            history_path = root / "data" / "processed" / "appointments.csv"
            history_path.parent.mkdir(parents=True)
            pd.DataFrame({
                "date": ["2025-12-30", "2025-12-31"],
                "no_show": [0, 1],
            }).to_csv(history_path, index=False)
            result = build_forecast_backtest(root, {"no_show": path})
            self.assertAlmostEqual(result.iloc[0]["brier_score"], 0.01)
            self.assertAlmostEqual(result.iloc[0]["roc_auc"], 1.0)
            self.assertAlmostEqual(result.iloc[0]["pre_holdout_baseline_brier"], 0.25)


if __name__ == "__main__":
    unittest.main()

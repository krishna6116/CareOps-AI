"""Leakage audit: no feature may use information that is unavailable when the
prediction is made.

Prediction times:
- admissions, occupancy, ICU: forecasts made at the end of the previous day;
- length of stay: at admission;
- no-show: the day before the appointment.

Most tests perturb a value at time t and check that no feature at or before t
changes.
"""

import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

from ml.icu_demand_forecasting import icu_demand_forecasting as icu
from ml.los_evaluation import completed_before
from ml.los_prediction import los_prediction as los
from ml.no_show_prediction import no_show_prediction as no_show
from ml.occupancy_forecasting import create_features as occupancy_features

ROOT = Path(__file__).resolve().parents[1]
SYNTHETIC = ROOT / "data" / "synthetic"
ADMISSION_FEATURES = ROOT / "ml" / "admission_forecasting" / "admission_forecasting_features.csv"
EXPLORATORY_LOS = ROOT / "data" / "feature_engineering" / "los_prediction_features.csv"


def quiet(function, *args):
    with redirect_stdout(io.StringIO()):
        return function(*args)


def changed_rows(before: pd.DataFrame, after: pd.DataFrame, columns: list[str]) -> pd.Series:
    left = before[columns].astype(float).fillna(-999.0).to_numpy()
    right = after[columns].astype(float).fillna(-999.0).to_numpy()
    return pd.Series(~np.isclose(left, right).all(axis=1), index=before.index)


class OccupancyLeakageTests(unittest.TestCase):
    def test_features_on_a_day_ignore_that_day_and_later(self):
        rng = np.random.default_rng(1)
        history = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=120, freq="D"),
            "occupancy_rate": rng.uniform(0.5, 0.7, 120),
        })
        perturbed = history.copy()
        perturbed.loc[90:, "occupancy_rate"] += 0.2

        before = occupancy_features(history)
        after = occupancy_features(perturbed)

        features = [c for c in before.columns if c not in {"date", "occupancy_rate", "target"}]
        changed = changed_rows(before, after, features)
        self.assertFalse(changed.iloc[:91].any(), "features up to day 90 used day-90+ occupancy")
        self.assertTrue(changed.iloc[91:].any())


class IcuLeakageTests(unittest.TestCase):
    def test_features_on_a_day_ignore_that_day_and_later(self):
        rng = np.random.default_rng(2)
        daily = pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=120, freq="D"),
            "admissions": rng.poisson(12, 120).astype(float),
            "discharges": rng.poisson(12, 120).astype(float),
            "emergency_cases": rng.poisson(3, 120).astype(float),
            "ICU_admissions": rng.poisson(1, 120).astype(float),
            "occupancy": rng.uniform(50, 70, 120),
        })
        perturbed = daily.copy()
        for column in ["admissions", "discharges", "emergency_cases", "ICU_admissions", "occupancy"]:
            perturbed.loc[90:, column] += 5

        before = quiet(icu.add_features, daily).set_index("date")
        after = quiet(icu.add_features, perturbed).set_index("date")

        changed = changed_rows(before, after, icu.FEATURE_COLUMNS)
        cutoff = pd.Timestamp("2025-01-01") + pd.Timedelta(days=90)
        self.assertFalse(changed[changed.index <= cutoff].any())
        self.assertTrue(changed[changed.index > cutoff].any())


@unittest.skipUnless(ADMISSION_FEATURES.is_file(), "admission feature table not generated")
class AdmissionLeakageTests(unittest.TestCase):
    """The admission script runs on import, so its saved feature table is audited."""

    def test_lags_and_rolling_windows_use_only_previous_days(self):
        frame = pd.read_csv(ADMISSION_FEATURES, parse_dates=["date"]).sort_values(["department", "date"])
        for _, group in frame.groupby("department"):
            admissions = group["admissions"].astype(float)
            pd.testing.assert_series_equal(
                group["lag_1"], admissions.shift(1), check_names=False)
            pd.testing.assert_series_equal(
                group["lag_7"], admissions.shift(7), check_names=False)
            pd.testing.assert_series_equal(
                group["rolling_mean_7"], admissions.shift(1).rolling(7).mean(),
                check_names=False, atol=1e-9)
            pd.testing.assert_series_equal(
                group["rolling_max_28"], admissions.shift(1).rolling(28).max(),
                check_names=False)


class NoShowLeakageTests(unittest.TestCase):
    @staticmethod
    def appointments(seed: int = 3) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        n = 600
        frame = pd.DataFrame({
            "appointment_id": [f"A{i}" for i in range(n)],
            "patient_id": rng.choice([f"P{i}" for i in range(40)], n),
            "department": rng.choice(["Cardiology", "Surgery", "Oncology"], n),
            "priority": rng.choice(["Routine", "Urgent"], n),
            "patient_severity": rng.choice(["Low", "High"], n),
            "patient_age": rng.integers(18, 90, n),
            "appointment_datetime": pd.Timestamp("2025-03-01")
            + pd.to_timedelta(rng.integers(0, 30, n), unit="D")
            + pd.to_timedelta(rng.choice([9, 11, 14, 16], n), unit="h"),
            "no_show": rng.integers(0, 2, n),
        })
        return frame.sort_values(["appointment_datetime", "appointment_id"]).reset_index(drop=True)

    def test_features_ignore_outcomes_from_the_same_day_and_later(self):
        frame = self.appointments()
        day = pd.Timestamp("2025-03-15")
        perturbed = frame.copy()
        flip = perturbed["appointment_datetime"].dt.normalize() >= day
        perturbed.loc[flip, "no_show"] = 1 - perturbed.loc[flip, "no_show"]

        before = quiet(no_show.create_historical_features, frame)
        after = quiet(no_show.create_historical_features, perturbed)

        changed = changed_rows(before, after, list(before.columns))
        on_or_before = (frame["appointment_datetime"].dt.normalize() <= day).to_numpy()
        self.assertFalse(changed[on_or_before].any(), "features used outcomes from the same day or later")
        self.assertTrue(changed[~on_or_before].any())

    def test_prior_does_not_use_the_overall_no_show_rate(self):
        frame = self.appointments()
        later = frame["appointment_datetime"].dt.normalize() > pd.Timestamp("2025-03-20")
        all_no_shows = frame.copy()
        all_no_shows.loc[later, "no_show"] = 1

        before = quiet(no_show.create_historical_features, frame)
        after = quiet(no_show.create_historical_features, all_no_shows)

        early = (~later).to_numpy()
        self.assertFalse(changed_rows(before, after, list(before.columns))[early].any())


class LosLeakageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        admissions = pd.read_csv(SYNTHETIC / "admissions.csv", nrows=3000)
        patients = pd.read_csv(SYNTHETIC / "patients.csv")
        cls.admissions = admissions[admissions["stay_status"] == "completed"].copy()
        cls.patients = patients

    def build(self, admissions: pd.DataFrame):
        combined = quiet(los.merge_admissions_patients, admissions, self.patients)
        combined, _ = quiet(los.build_los_target, combined)
        return quiet(los.build_features, combined)

    def test_no_feature_is_derived_from_discharge_or_los(self):
        _, numeric, categorical, _ = self.build(self.admissions)

        forbidden = ("discharge", "los", "length_of_stay", "stay_status")
        leaking = [c for c in numeric + categorical if any(word in c.lower() for word in forbidden)]
        self.assertEqual(leaking, [])

    def test_changing_the_outcome_does_not_change_features(self):
        perturbed = self.admissions.copy()
        perturbed["LOS"] = perturbed["LOS"] + 7
        perturbed["discharge_date"] = (
            pd.to_datetime(perturbed["discharge_date"]) + pd.Timedelta(days=7)
        ).dt.strftime("%Y-%m-%d")

        before, numeric, categorical, _ = self.build(self.admissions)
        after, _, _, _ = self.build(perturbed)

        pd.testing.assert_frame_equal(
            before[numeric + categorical].reset_index(drop=True),
            after[numeric + categorical].reset_index(drop=True),
        )

    def test_training_excludes_stays_still_open_at_the_holdout_start(self):
        train = pd.DataFrame({
            "admitted": pd.to_datetime(["2026-01-01", "2026-02-20", "2026-02-25"]),
            "los": [5, 15, 2],
        })

        kept = completed_before(train, "admitted", "los", pd.Timestamp("2026-03-01"))

        self.assertEqual(kept["los"].tolist(), [5, 2])


@unittest.skipUnless(EXPLORATORY_LOS.is_file(), "exploratory feature tables not generated")
class ExploratoryFeatureLeakageTests(unittest.TestCase):
    def test_los_feature_table_has_no_discharge_information(self):
        columns = pd.read_csv(EXPLORATORY_LOS, nrows=1).columns
        leaking = [
            c for c in columns
            if c.startswith("discharge_") or c in {"LOS", "los_group", "expected_discharge_date_from_actual_los"}
        ]
        self.assertEqual(leaking, [])


if __name__ == "__main__":
    unittest.main()

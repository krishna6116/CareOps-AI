import unittest

import pandas as pd

from reliability_scorecard import _forecast_reliability


class ForecastReliabilityTests(unittest.TestCase):
    def test_admission_holdout_score_tracks_baseline_result(self):
        row = pd.Series({
            "model": "admissions_xgboost",
            "beats_baseline": False,
            "mae": 4.7,
            "baseline_mae": 3.6,
            "test_rows": 196,
            "test_start": "2026-03-19",
            "test_end": "2026-09-30",
        })

        score = _forecast_reliability(row)

        self.assertEqual(score["score"], 3.0)
        self.assertFalse(score["beats_baseline"])
        self.assertEqual(score["test_rows"], 196)
        self.assertIn("MAE 4.70 vs baseline 3.60", score["evidence"])
        self.assertEqual(score["threshold"], "MAE must beat the seasonal baseline.")

    def test_occupancy_bed_count_score_does_not_require_missing_baseline(self):
        row = pd.Series({
            "model": "occupancy_ridge_bed_count",
            "beats_baseline": float("nan"),
            "wape_percent": 4.5,
            "test_rows": 195,
            "test_start": "2026-03-19",
            "test_end": "2026-09-29",
        })

        score = _forecast_reliability(row)

        self.assertEqual(score["score"], 4.0)
        self.assertIsNone(score["beats_baseline"])

    def test_no_show_score_uses_brier_baseline(self):
        row = pd.Series({
            "model": "no_show_xgboost",
            "beats_baseline": float("nan"),
            "brier_beats_baseline": True,
            "brier_score": 0.11,
            "pre_holdout_baseline_brier": 0.13,
            "roc_auc": 0.74,
            "test_rows": 24000,
            "test_start": "2026-03-11",
            "test_end": "2026-09-30",
        })

        score = _forecast_reliability(row)

        self.assertEqual(score["score"], 4.5)
        self.assertIsNone(score["beats_baseline"])
        self.assertIn("Brier score 0.110 vs prevalence baseline 0.130", score["metric"])
        self.assertIn("ROC-AUC 0.740", score["metric"])

    def test_no_show_score_stays_at_four_without_discrimination_threshold(self):
        row = pd.Series({
            "model": "no_show_xgboost",
            "beats_baseline": float("nan"),
            "brier_beats_baseline": True,
            "brier_score": 0.11,
            "pre_holdout_baseline_brier": 0.13,
            "roc_auc": 0.69,
            "test_rows": 24000,
            "test_start": "2026-03-11",
            "test_end": "2026-09-30",
        })

        score = _forecast_reliability(row)

        self.assertEqual(score["score"], 4.0)


if __name__ == "__main__":
    unittest.main()

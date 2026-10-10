import unittest

import numpy as np
import pandas as pd

from ml.forecast_evaluation import (
    NAIVE_BASELINES,
    conformal_intervals,
    forecast_metrics,
    forecast_sanity,
    rolling_origin_backtest,
    rolling_origins,
    select_champion,
    summarize_backtest,
)


def weekly_series(days: int = 140) -> pd.Series:
    index = pd.date_range("2026-01-01", periods=days, freq="D")
    pattern = np.array([10, 12, 14, 12, 10, 6, 5], dtype=float)
    return pd.Series(np.resize(pattern, days), index=index)


class MetricTests(unittest.TestCase):
    def test_perfect_forecast(self):
        metrics = forecast_metrics([1, 2, 3], [1, 2, 3], mase_scale=1.0)

        self.assertEqual(metrics["MAE"], 0.0)
        self.assertEqual(metrics["sMAPE_percent"], 0.0)
        self.assertEqual(metrics["MASE"], 0.0)

    def test_mape_ignores_zero_actuals_but_smape_does_not(self):
        metrics = forecast_metrics([0, 2], [1, 1])

        self.assertAlmostEqual(metrics["MAPE_percent"], 50.0)
        self.assertAlmostEqual(metrics["sMAPE_percent"], (200.0 + 200 / 3) / 2)


class BacktestTests(unittest.TestCase):
    def test_backtest_uses_only_history_up_to_origin(self):
        series = weekly_series()
        seen_last_dates = []

        def spy(history, horizon):
            seen_last_dates.append(history.index.max())
            return np.zeros(horizon)

        origins = rolling_origins(series.index, series.index[60], horizon=30, step_days=14)
        rolling_origin_backtest(series, {"spy": spy}, origins, horizon=30)

        self.assertEqual(seen_last_dates, list(origins))
        self.assertLessEqual(origins.max() + pd.Timedelta(days=30), series.index.max())

    def test_seasonal_naive_is_exact_on_pure_weekly_series(self):
        series = weekly_series()
        origins = rolling_origins(series.index, series.index[60], horizon=30, step_days=14)

        summary = summarize_backtest(rolling_origin_backtest(series, NAIVE_BASELINES, origins, 30))

        self.assertEqual(select_champion(summary), "same_weekday_last_week")
        self.assertEqual(summary.iloc[0]["MAE"], 0.0)

    def test_backtest_clips_negative_predictions(self):
        series = weekly_series()
        origins = rolling_origins(series.index, series.index[60], horizon=7, step_days=14)

        backtest = rolling_origin_backtest(series, {"neg": lambda h, n: np.full(n, -5.0)}, origins, 7)

        self.assertTrue((backtest["predicted"] >= 0).all())


class IntervalTests(unittest.TestCase):
    def test_intervals_cover_noise_at_roughly_the_nominal_rate(self):
        rng = np.random.default_rng(7)
        index = pd.date_range("2025-01-01", periods=730, freq="D")
        series = pd.Series(20 + rng.normal(0, 2, len(index)), index=index)
        origins = rolling_origins(series.index, series.index[90], horizon=30, step_days=7)
        backtest = rolling_origin_backtest(series, {"mean": lambda h, n: np.repeat(20.0, n)}, origins, 30)

        widths, coverage = conformal_intervals(backtest, "mean")

        self.assertEqual(set(widths["bucket"]), {"1-7", "8-14", "15-30"})
        for level, group in coverage.groupby("level"):
            self.assertAlmostEqual(group["empirical_coverage"].mean(), level, delta=0.06)


class SanityTests(unittest.TestCase):
    def test_flat_and_impossible_values_are_flagged(self):
        flags = forecast_sanity([231.0, 231.0], history=[100, 120, 140], capacity=230)

        self.assertTrue(flags["is_flat"])
        self.assertTrue(flags["exceeds_capacity"])
        self.assertTrue(flags["all_whole_numbers"])


if __name__ == "__main__":
    unittest.main()

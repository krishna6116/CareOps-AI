import unittest

import numpy as np
import pandas as pd

from ml.forecast_evaluation import (
    NAIVE_BASELINES,
    conformal_intervals,
    forecast_metrics,
    forecast_sanity,
    integer_interval,
    model_status,
    rolling_origin_backtest,
    rolling_origins,
    rolling_windows,
    select_champion,
    summarize_backtest,
    threshold_exceedance,
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
        self.assertEqual(set(coverage["method"]), {"split_half_continuous", "sequential_published"})
        for (method, level), group in coverage.groupby(["method", "level"]):
            # Published whole-number bounds can only widen the interval.
            delta = 0.06 if method == "split_half_continuous" else 0.08
            self.assertAlmostEqual(group["empirical_coverage"].mean(), level, delta=delta)

    def test_sequential_calibration_only_uses_already_observed_errors(self):
        index = pd.date_range("2025-01-01", periods=200, freq="D")
        # Errors jump after day 150; an origin before then must not see them.
        series = pd.Series(np.where(np.arange(200) < 150, 20.0, 40.0), index=index)
        origins = rolling_origins(series.index, series.index[60], horizon=7, step_days=7)
        backtest = rolling_origin_backtest(series, {"flat": lambda h, n: np.repeat(20.0, n)}, origins, 7)

        _, coverage = conformal_intervals(backtest, "flat", min_calibration_origins=3)

        sequential = coverage[coverage["method"].eq("sequential_published")]
        # Early origins had zero error so far; later jumps are missed, not foreseen.
        self.assertLess(sequential["empirical_coverage"].max(), 1.0)

    def test_integer_interval_is_whole_non_negative_and_capped(self):
        lower, upper = integer_interval([1.2, 228.6], [3.0, 4.0], capacity=230)

        self.assertEqual(lower.tolist(), [0.0, 224.0])
        self.assertEqual(upper.tolist(), [5.0, 230.0])


class ThresholdAndStatusTests(unittest.TestCase):
    def test_threshold_metrics_report_undefined_values_as_nan(self):
        backtest = pd.DataFrame({
            "model": "m", "actual": [100, 150, 210, 220], "predicted": [100, 212, 205, 225],
        })

        result = threshold_exceedance(backtest, "m", {"warning": 207, "never": 300}).set_index("threshold_name")

        self.assertEqual(result.loc["warning", "true_positives"], 1)
        self.assertAlmostEqual(result.loc["warning", "precision"], 0.5)
        self.assertAlmostEqual(result.loc["warning", "recall"], 0.5)
        self.assertTrue(np.isnan(result.loc["never", "precision"]))
        self.assertEqual(result.loc["never", "actual_events"], 0)

    def test_status_says_when_a_baseline_is_published(self):
        summary = pd.DataFrame([
            {"model": "moving_average_28", "MAE": 2.9},
            {"model": "xgboost_recursive", "MAE": 3.1},
            {"model": "last_value", "MAE": 5.0},
        ])

        status = model_status(summary, "moving_average_28", ("xgboost_recursive",))

        self.assertEqual(status["model_type"], "baseline")
        self.assertFalse(status["learned_beats_best_baseline"])
        self.assertEqual(status["best_baseline"], "moving_average_28")
        self.assertEqual(status["best_learned_model"], "xgboost_recursive")

    def test_rolling_windows_score_each_window(self):
        frame = pd.DataFrame({
            "when": pd.date_range("2026-01-01", periods=60, freq="D"),
            "error": np.arange(60, dtype=float),
        })
        origins = pd.DatetimeIndex(["2025-12-31", "2026-01-30"])

        windows = rolling_windows(frame, "when", origins, 30, lambda w: {"mean_error": w["error"].mean()})

        self.assertEqual(windows["rows"].tolist(), [30, 30])
        self.assertEqual(windows["mean_error"].tolist(), [14.5, 44.5])


class SanityTests(unittest.TestCase):
    def test_flat_and_impossible_values_are_flagged(self):
        flags = forecast_sanity([231.0, 231.0], history=[100, 120, 140], capacity=230)

        self.assertTrue(flags["is_flat"])
        self.assertTrue(flags["exceeds_capacity"])
        self.assertTrue(flags["all_whole_numbers"])


if __name__ == "__main__":
    unittest.main()

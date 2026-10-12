"""Shared forecast evaluation: naive baselines, rolling-origin backtests,
empirical prediction intervals, and champion selection.

All evaluation is chronological. A forecaster is a callable
``forecaster(history: pd.Series, horizon: int) -> np.ndarray`` that only sees
values up to and including the forecast origin.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


Forecaster = Callable[[pd.Series, int], np.ndarray]

INTERVAL_LEVELS = (0.80, 0.95)
HORIZON_BUCKETS = ((1, 7, "1-7"), (8, 14, "8-14"), (15, 30, "15-30"))


def last_value(history: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(float(history.iloc[-1]), horizon)


def seasonal_naive_7(history: pd.Series, horizon: int) -> np.ndarray:
    return np.resize(history.iloc[-7:].to_numpy(dtype=float), horizon)


def moving_average(window: int) -> Forecaster:
    def forecast(history: pd.Series, horizon: int) -> np.ndarray:
        return np.repeat(float(history.iloc[-window:].mean()), horizon)

    forecast.__name__ = f"moving_average_{window}"
    return forecast


NAIVE_BASELINES: dict[str, Forecaster] = {
    "last_value": last_value,
    "same_weekday_last_week": seasonal_naive_7,
    "moving_average_7": moving_average(7),
    "moving_average_28": moving_average(28),
}


def forecast_metrics(actual, predicted, mase_scale: float | None = None) -> dict[str, float]:
    """MAE, RMSE, MAPE (non-zero actuals only), sMAPE, bias, and optional MASE."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    errors = p - y
    nonzero = y != 0
    denominator = np.abs(y) + np.abs(p)
    smape_terms = np.divide(
        2 * np.abs(errors), denominator,
        out=np.zeros_like(errors), where=denominator > 0,
    )
    mae = float(np.mean(np.abs(errors)))
    return {
        "MAE": mae,
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "MAPE_percent": float(np.mean(np.abs(errors[nonzero] / y[nonzero])) * 100) if nonzero.any() else np.nan,
        "sMAPE_percent": float(np.mean(smape_terms) * 100),
        "bias": float(np.mean(errors)),
        "MASE": mae / mase_scale if mase_scale else np.nan,
    }


def seasonal_mase_scale(training: pd.Series, period: int = 7) -> float:
    """In-sample MAE of the seasonal naive forecast, the usual MASE denominator."""
    values = training.to_numpy(dtype=float)
    return float(np.mean(np.abs(values[period:] - values[:-period])))


def rolling_origins(
    index: pd.DatetimeIndex,
    first_origin: pd.Timestamp,
    horizon: int,
    step_days: int,
) -> pd.DatetimeIndex:
    """Origins from ``first_origin`` such that every horizon is fully observed."""
    last_origin = index.max() - pd.Timedelta(days=horizon)
    return pd.date_range(first_origin, last_origin, freq=f"{step_days}D")


def rolling_origin_backtest(
    series: pd.Series,
    forecasters: dict[str, Forecaster],
    origins: pd.DatetimeIndex,
    horizon: int,
) -> pd.DataFrame:
    """Forecast ``horizon`` days from each origin using only history to the origin."""
    series = series.sort_index().astype(float)
    rows = []
    for origin in origins:
        history = series[series.index <= origin]
        target_dates = pd.date_range(origin + pd.Timedelta(days=1), periods=horizon, freq="D")
        actual = series.reindex(target_dates).to_numpy()
        for name, forecaster in forecasters.items():
            predicted = np.clip(np.asarray(forecaster(history, horizon), dtype=float), 0, None)
            if predicted.shape != (horizon,):
                raise ValueError(f"{name} returned {predicted.shape}, expected ({horizon},).")
            for step in range(horizon):
                rows.append({
                    "origin": origin,
                    "date": target_dates[step],
                    "horizon": step + 1,
                    "model": name,
                    "actual": actual[step],
                    "predicted": predicted[step],
                })
    return pd.DataFrame(rows)


def summarize_backtest(backtest: pd.DataFrame, mase_scale: float | None = None) -> pd.DataFrame:
    rows = []
    for model, group in backtest.groupby("model", sort=False):
        row = {"model": model, "origins": group["origin"].nunique(), "forecast_days": len(group)}
        row.update(forecast_metrics(group["actual"], group["predicted"], mase_scale))
        for low, high, label in HORIZON_BUCKETS:
            bucket = group[group["horizon"].between(low, high)]
            row[f"MAE_h{label}"] = float(np.mean(np.abs(bucket["predicted"] - bucket["actual"]))) if len(bucket) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)


def select_champion(summary: pd.DataFrame, candidates: list[str] | None = None) -> str:
    """Lowest backtest MAE wins; ties go to the simpler (baseline) model listed first."""
    pool = summary if candidates is None else summary[summary["model"].isin(candidates)]
    if pool.empty:
        raise ValueError("No candidate models were backtested.")
    return str(pool.sort_values(["MAE", "RMSE"], kind="stable").iloc[0]["model"])


def integer_interval(
    predicted, half_width, capacity: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Whole-number bounds as published: floor/ceil, non-negative, capped."""
    predicted = np.asarray(predicted, dtype=float)
    half_width = np.asarray(half_width, dtype=float)
    lower = np.floor(np.clip(predicted - half_width, 0, None))
    upper = np.ceil(predicted + half_width)
    if capacity is not None:
        lower = np.minimum(lower, capacity)
        upper = np.minimum(upper, capacity)
    return lower, upper


def conformal_intervals(
    backtest: pd.DataFrame,
    model: str,
    levels: tuple[float, ...] = INTERVAL_LEVELS,
    capacity: float | None = None,
    min_calibration_origins: int = 6,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conformal interval half-widths by horizon bucket, plus measured coverage.

    Published widths use the absolute errors from every origin.

    Coverage is reported two ways:

    - ``split_half_continuous``: the earlier half of origins calibrates, the
      later half is checked against the raw continuous interval.
    - ``sequential_published``: at each origin after the first
      ``min_calibration_origins``, the width comes only from errors whose
      actual values were already observed by that origin (expanding window),
      and coverage is checked on the whole-number interval that would be
      published (floor/ceil, non-negative, capped at ``capacity``).
    """
    frame = backtest[backtest["model"] == model].copy()
    frame["abs_error"] = (frame["actual"] - frame["predicted"]).abs()
    frame["bucket"] = horizon_bucket(frame["horizon"])
    origins = np.sort(frame["origin"].unique())

    widths = [
        {"level": level, "bucket": bucket,
         "half_width": float(_conformal_quantile(group["abs_error"], level))}
        for level in levels
        for bucket, group in frame.groupby("bucket", sort=False)
    ]

    coverage = []
    split = origins[len(origins) // 2]
    calibration = frame[frame["origin"] < split]
    evaluation = frame[frame["origin"] >= split]
    for level in levels:
        for bucket in frame["bucket"].unique():
            calibrated = _conformal_quantile(calibration.loc[calibration["bucket"] == bucket, "abs_error"], level)
            held_out = evaluation.loc[evaluation["bucket"] == bucket, "abs_error"]
            coverage.append({
                "method": "split_half_continuous", "level": level, "bucket": bucket,
                "evaluation_origins": int(evaluation["origin"].nunique()),
                "mean_half_width": float(calibrated),
                "empirical_coverage": float((held_out <= calibrated).mean()) if len(held_out) else np.nan,
            })

    sequential = []
    for origin in origins[min_calibration_origins:]:
        known = frame[(frame["origin"] < origin) & (frame["date"] <= origin)]
        current = frame[frame["origin"] == origin]
        for level in levels:
            for bucket, group in current.groupby("bucket", sort=False):
                half_width = _conformal_quantile(known.loc[known["bucket"] == bucket, "abs_error"], level)
                lower, upper = integer_interval(group["predicted"], half_width, capacity)
                actual = group["actual"].to_numpy()
                sequential.append(pd.DataFrame({
                    "level": level, "bucket": bucket, "origin": origin, "half_width": half_width,
                    "covered": (actual >= lower) & (actual <= upper),
                }))
    if sequential:
        sequential_frame = pd.concat(sequential, ignore_index=True)
        for (level, bucket), group in sequential_frame.groupby(["level", "bucket"], sort=False):
            coverage.append({
                "method": "sequential_published", "level": level, "bucket": bucket,
                "evaluation_origins": int(group["origin"].nunique()),
                "mean_half_width": float(group["half_width"].mean()),
                "empirical_coverage": float(group["covered"].mean()),
            })
    return pd.DataFrame(widths), pd.DataFrame(coverage)


def threshold_exceedance(
    backtest: pd.DataFrame, model: str, thresholds: dict[str, float],
) -> pd.DataFrame:
    """Precision/recall/F1 for 'value at or above threshold' events.

    Undefined metrics (no predicted or no actual events) are reported as NaN
    with the event counts, never silently as zero.
    """
    frame = backtest[backtest["model"] == model]
    rows = []
    for name, threshold in thresholds.items():
        predicted = frame["predicted"].to_numpy() >= threshold
        actual = frame["actual"].to_numpy() >= threshold
        true_positive = int((predicted & actual).sum())
        precision = true_positive / predicted.sum() if predicted.sum() else np.nan
        recall = true_positive / actual.sum() if actual.sum() else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
            else np.nan
        )
        rows.append({
            "threshold_name": name, "threshold": float(threshold), "forecast_days": len(frame),
            "actual_events": int(actual.sum()), "predicted_events": int(predicted.sum()),
            "true_positives": true_positive, "precision": precision, "recall": recall, "f1": f1,
        })
    return pd.DataFrame(rows)


def rolling_windows(
    frame: pd.DataFrame,
    date_column: str,
    origins: pd.DatetimeIndex,
    window_days: int,
    metric_fn: Callable[[pd.DataFrame], dict[str, float]],
) -> pd.DataFrame:
    """Score a fixed model on each window (origin, origin + window_days]."""
    dates = pd.to_datetime(frame[date_column]).dt.normalize()
    rows = []
    for origin in origins:
        window = frame[(dates > origin) & (dates <= origin + pd.Timedelta(days=window_days))]
        if len(window):
            rows.append({"origin": origin, "rows": len(window), **metric_fn(window)})
    return pd.DataFrame(rows)


def evaluate_candidates(
    series: pd.Series,
    learned: dict[str, Forecaster],
    first_origin: pd.Timestamp,
    horizon: int,
    step_days: int,
    capacity: float | None = None,
) -> dict[str, object]:
    """Backtest learned forecasters against the naive baselines, pick the
    lowest-MAE champion, and calibrate its intervals."""
    train = series[series.index <= first_origin]
    origins = rolling_origins(series.index, first_origin, horizon, step_days)
    backtest = rolling_origin_backtest(series, {**NAIVE_BASELINES, **learned}, origins, horizon)
    summary = summarize_backtest(backtest, seasonal_mase_scale(train))
    champion = select_champion(summary)
    widths, coverage = conformal_intervals(backtest, champion, capacity=capacity)
    return {
        "origins": origins, "backtest": backtest, "summary": summary, "champion": champion,
        "widths": widths, "coverage": coverage,
        "status": model_status(summary, champion, tuple(learned)),
    }


def interval_columns(
    expected, widths: pd.DataFrame, capacity: float | None = None, point=None,
) -> dict[str, np.ndarray]:
    """Published whole-number interval columns for a 1..n day forecast path.

    ``point`` (the published whole-number forecast) is kept inside its own
    interval, since rounding can move it by up to one unit."""
    expected = np.asarray(expected, dtype=float)
    buckets = horizon_bucket(pd.Series(np.arange(1, len(expected) + 1)))
    columns = {}
    for level in sorted(widths["level"].unique()):
        label = int(round(level * 100))
        half_width = buckets.map(
            widths[widths["level"] == level].set_index("bucket")["half_width"]
        ).to_numpy()
        lower, upper = integer_interval(expected, half_width, capacity)
        if point is not None:
            lower = np.minimum(lower, point)
            upper = np.maximum(upper, point)
        columns[f"lower_{label}"] = lower.astype(np.int64)
        columns[f"upper_{label}"] = upper.astype(np.int64)
    return columns


def model_status(
    summary: pd.DataFrame, champion: str, learned_models: tuple[str, ...],
    score_column: str = "MAE",
) -> dict[str, object]:
    """Columns stating whether the published model is learned or a baseline,
    and how it compares with the best baseline on the backtest (lower is better)."""
    baselines = summary[~summary["model"].isin(learned_models)].sort_values(score_column, kind="stable")
    learned = summary[summary["model"].isin(learned_models)].sort_values(score_column, kind="stable")
    best_baseline = baselines.iloc[0]
    best_learned = learned.iloc[0] if not learned.empty else None
    return {
        "model": champion,
        "model_type": "learned" if champion in learned_models else "baseline",
        "best_learned_model": best_learned["model"] if best_learned is not None else "",
        "learned_beats_best_baseline": bool(
            best_learned is not None and float(best_learned[score_column]) < float(best_baseline[score_column])
        ),
        "best_baseline": best_baseline["model"],
        "backtest_metric": score_column,
        "published_backtest_score": round(float(summary.loc[summary["model"] == champion, score_column].iloc[0]), 4),
        "best_baseline_backtest_score": round(float(best_baseline[score_column]), 4),
    }


def horizon_bucket(horizon: pd.Series) -> pd.Series:
    labels = pd.Series(index=horizon.index, dtype=object)
    for low, high, label in HORIZON_BUCKETS:
        labels[horizon.between(low, high)] = label
    if labels.isna().any():
        raise ValueError("Horizon outside the supported 1-30 day buckets.")
    return labels


def _conformal_quantile(errors: pd.Series, level: float) -> float:
    values = np.sort(np.asarray(errors, dtype=float))
    if values.size == 0:
        return np.nan
    rank = min(values.size, int(np.ceil((values.size + 1) * level)))
    return float(values[rank - 1])


def residual_diagnostics(actual, predicted) -> dict[str, float]:
    residual = pd.Series(np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float))
    return {
        "residual_mean": float(residual.mean()),
        "residual_std": float(residual.std()),
        "residual_autocorr_lag1": float(residual.autocorr(1)),
        "residual_autocorr_lag7": float(residual.autocorr(7)),
    }


def forecast_sanity(forecast, history, capacity: float | None = None) -> dict[str, float | bool]:
    """Flag flat lines and impossible values in a published forecast path."""
    values = np.asarray(forecast, dtype=float)
    recent = np.asarray(history, dtype=float)[-90:]
    return {
        "forecast_min": float(values.min()),
        "forecast_max": float(values.max()),
        "forecast_std": float(values.std(ddof=1)),
        "recent_history_std": float(recent.std(ddof=1)),
        "distinct_values": int(np.unique(values).size),
        "is_flat": bool(np.unique(values).size <= 1),
        "has_negative": bool((values < 0).any()),
        "exceeds_capacity": bool(capacity is not None and (values > capacity).any()),
        "all_whole_numbers": bool(np.equal(np.mod(values, 1), 0).all()),
    }

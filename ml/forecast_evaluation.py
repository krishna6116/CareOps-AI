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


def conformal_intervals(
    backtest: pd.DataFrame,
    model: str,
    levels: tuple[float, ...] = INTERVAL_LEVELS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split-conformal interval widths by horizon bucket, plus held-out coverage.

    The earlier half of the origins calibrates absolute-error quantiles; the
    later half measures how often the actual value falls inside. The returned
    widths for publishing use all origins.
    """
    frame = backtest[backtest["model"] == model].copy()
    frame["abs_error"] = (frame["actual"] - frame["predicted"]).abs()
    frame["bucket"] = horizon_bucket(frame["horizon"])
    origins = np.sort(frame["origin"].unique())
    split = origins[len(origins) // 2]
    calibration = frame[frame["origin"] < split]
    evaluation = frame[frame["origin"] >= split]
    widths, coverage = [], []
    for level in levels:
        for bucket, group in frame.groupby("bucket", sort=False):
            widths.append({"level": level, "bucket": bucket,
                           "half_width": float(_conformal_quantile(group["abs_error"], level))})
            calibrated = _conformal_quantile(calibration.loc[calibration["bucket"] == bucket, "abs_error"], level)
            held_out = evaluation.loc[evaluation["bucket"] == bucket, "abs_error"]
            coverage.append({
                "level": level, "bucket": bucket, "calibration_origins": int(calibration["origin"].nunique()),
                "evaluation_origins": int(evaluation["origin"].nunique()),
                "calibrated_half_width": float(calibrated),
                "empirical_coverage": float((held_out <= calibrated).mean()) if len(held_out) else np.nan,
            })
    return pd.DataFrame(widths), pd.DataFrame(coverage)


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

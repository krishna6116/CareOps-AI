"""Length-of-stay evaluation: simple baselines, rolling 30-day windows over
the holdout, and champion selection.

LOS is predicted per admission at admission time. Each rolling window
scores the admissions that arrive in the 30 days after a weekly origin, so
the result shows how stable accuracy is over the holdout. Models are fitted
once on stays already discharged before the holdout starts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forecast_evaluation import forecast_metrics, rolling_origins, rolling_windows

LEARNED = ("xgboost",)


def completed_before(train: pd.DataFrame, admitted: str, los: str, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Training stays whose discharge was already known at ``cutoff``.

    A stay admitted before the holdout but discharged during it has an LOS
    that was not yet observable when the holdout begins."""
    discharge = pd.to_datetime(train[admitted]) + pd.to_timedelta(train[los], unit="D")
    return train.loc[discharge < cutoff]


def fit_baselines(
    train: pd.DataFrame, test: pd.DataFrame, target: str, groups: list[str],
) -> dict[str, np.ndarray]:
    """Training-set mean LOS overall and by nested groups (e.g. department,
    then department x severity), falling back to the coarser level when a
    group has fewer than 20 training stays."""
    overall = float(train[target].mean())
    predictions = {"overall_mean": np.full(len(test), overall)}
    fallback = pd.Series(overall, index=test.index)
    for depth in range(1, len(groups) + 1):
        keys = groups[:depth]
        stats = train.groupby(keys)[target].agg(["mean", "count"])
        means = stats.loc[stats["count"] >= 20, "mean"]
        looked_up = pd.Series(
            [means.get(tuple(row) if depth > 1 else row[0], np.nan) for row in test[keys].itertuples(index=False)],
            index=test.index,
        )
        fallback = looked_up.fillna(fallback)
        predictions["mean_by_" + "_".join(keys)] = fallback.to_numpy(dtype=float)
    return predictions


def evaluate(
    test: pd.DataFrame,
    admitted: str,
    actual: str,
    predictions: dict[str, np.ndarray],
    first_origin: pd.Timestamp,
    window_days: int = 30,
    step_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Per-window and pooled MAE/RMSE/MAPE; champion = lowest pooled MAE."""
    frame = test[[admitted, actual]].copy()
    for name, values in predictions.items():
        frame[name] = values
    dates = pd.DatetimeIndex(pd.to_datetime(frame[admitted]).dt.normalize())
    origins = rolling_origins(dates, first_origin, window_days, step_days)

    window_rows = []
    for name in predictions:
        windows = rolling_windows(
            frame, admitted, origins, window_days,
            lambda w, n=name: forecast_metrics(w[actual], w[n]),
        )
        window_rows.append(windows.assign(model=name))
    windows = pd.concat(window_rows, ignore_index=True)

    summary_rows = []
    for name in predictions:
        pooled = forecast_metrics(frame[actual], frame[name])
        per_window = windows[windows["model"] == name]["MAE"]
        summary_rows.append({
            "model": name, "test_rows": len(frame), "windows": int(len(per_window)),
            "MAE": pooled["MAE"], "RMSE": pooled["RMSE"], "MAPE_percent": pooled["MAPE_percent"],
            "bias": pooled["bias"], "window_MAE_mean": float(per_window.mean()),
            "window_MAE_std": float(per_window.std()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["MAE", "RMSE"], kind="stable").reset_index(drop=True)
    return windows, summary, str(summary.iloc[0]["model"])

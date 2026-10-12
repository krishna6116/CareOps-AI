"""Evaluation for probability classifiers (no-show): metrics, calibration,
and the baselines a learned classifier must beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CALIBRATION_BINS = 10


def classification_metrics(actual, probability, threshold: float) -> dict[str, float]:
    """Discrimination, calibration and thresholded metrics for one window.

    Metrics that are undefined for the window (e.g. ROC-AUC with one class,
    precision with no positive predictions) are NaN, not zero.
    """
    y = np.asarray(actual, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    predicted = p >= threshold
    positives = y == 1
    true_positive = float((predicted & positives).sum())
    precision = true_positive / predicted.sum() if predicted.sum() else np.nan
    recall = true_positive / positives.sum() if positives.sum() else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else np.nan
    )
    both_classes = 0 < positives.sum() < len(y)
    return {
        "prevalence": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "brier": float(np.mean((p - y) ** 2)),
        "log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        "roc_auc": float(roc_auc_score(y, p)) if both_classes else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if both_classes else np.nan,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ece": expected_calibration_error(y, p),
    }


def reliability_table(actual, probability, bins: int = CALIBRATION_BINS) -> pd.DataFrame:
    """Observed rate vs mean predicted probability in equal-width bins."""
    y = np.asarray(actual, dtype=float)
    p = np.asarray(probability, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = index == b
        if mask.any():
            rows.append({
                "bin_lower": edges[b], "bin_upper": edges[b + 1], "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()), "observed_rate": float(y[mask].mean()),
            })
    return pd.DataFrame(rows)


def expected_calibration_error(actual, probability, bins: int = CALIBRATION_BINS) -> float:
    table = reliability_table(actual, probability, bins)
    if table.empty:
        return np.nan
    weights = table["count"] / table["count"].sum()
    return float((weights * (table["mean_predicted"] - table["observed_rate"]).abs()).sum())


def evaluate_classifiers(
    dates: pd.Series,
    actual,
    candidates: dict[str, tuple[np.ndarray, float]],
    first_origin: pd.Timestamp,
    window_days: int = 30,
    step_days: int = 7,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Score fixed classifiers on rolling 30-day windows and pooled.

    ``candidates`` maps a model name to (holdout probabilities, decision
    threshold chosen on validation data). The champion has the lowest pooled
    Brier score (a proper scoring rule)."""
    frame = pd.DataFrame({
        "date": pd.to_datetime(dates).dt.normalize().to_numpy(),
        "actual": np.asarray(actual, dtype=float),
    })
    origins = pd.date_range(first_origin, frame["date"].max() - pd.Timedelta(days=window_days), freq=f"{step_days}D")
    window_rows, summary_rows = [], []
    for name, (probability, threshold) in candidates.items():
        frame["p"] = probability
        for origin in origins:
            window = frame[(frame["date"] > origin) & (frame["date"] <= origin + pd.Timedelta(days=window_days))]
            if len(window):
                window_rows.append({
                    "model": name, "origin": origin, "rows": len(window),
                    **classification_metrics(window["actual"], window["p"], threshold),
                })
        pooled = classification_metrics(frame["actual"], frame["p"], threshold)
        model_windows = pd.DataFrame([row for row in window_rows if row["model"] == name])
        summary_rows.append({
            "model": name, "threshold": threshold, "test_rows": len(frame), "windows": len(model_windows),
            **pooled,
            "window_brier_std": float(model_windows["brier"].std()) if len(model_windows) else np.nan,
            "window_roc_auc_std": float(model_windows["roc_auc"].std()) if len(model_windows) else np.nan,
        })
    summary = pd.DataFrame(summary_rows).sort_values(["brier", "log_loss"], kind="stable").reset_index(drop=True)
    return pd.DataFrame(window_rows), summary, str(summary.iloc[0]["model"])


def prevalence_baseline(train_labels, n: int) -> np.ndarray:
    """Predict the training no-show rate for everyone."""
    return np.full(n, float(np.mean(np.asarray(train_labels, dtype=float))))


def logistic_regression_baseline(X_train: pd.DataFrame, y_train, X_eval: pd.DataFrame) -> np.ndarray:
    """Standardised, L2-regularised logistic regression on the same features."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    model.fit(X_train.astype(float), np.asarray(y_train, dtype=int))
    return model.predict_proba(X_eval.astype(float))[:, 1]

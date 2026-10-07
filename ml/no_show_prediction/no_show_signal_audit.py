"""
CareOps-AI
No-Show Signal Audit

Purpose
-------
Determine whether the synthetic appointment dataset contains meaningful,
learnable patterns for predicting patient no-shows BEFORE further model tuning.

Important:
- Does NOT regenerate synthetic data.
- Does NOT modify appointments.csv.
- Does NOT overwrite the existing no-show model.
- Does NOT require scikit-learn or SciPy.
- Uses native XGBoost only if available.
- Calculates ROC-AUC and PR-AUC manually.
- Produces CSV, JSON, TXT and PNG audit outputs.

Author: CareOps-AI
Random seed: 42
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42
np.random.seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SOURCE_CANDIDATES = [
    PROJECT_ROOT / "data" / "processed" / "appointments.csv",
    PROJECT_ROOT / "data" / "appointments.csv",
    PROJECT_ROOT / "data" / "synthetic" / "appointments.csv",
]

OUTPUT_DIR = PROJECT_ROOT / "ml" / "no_show_prediction"

TARGET = "no_show"
PATIENT_ID = "patient_id"

EXPECTED_COLUMNS = [
    "appointment_id",
    "patient_id",
    "patient_name",
    "doctor_id",
    "doctor_name",
    "department",
    "date",
    "time",
    "duration",
    "priority",
    "attended",
    "no_show",
    "patient_age",
    "patient_severity",
]

# ============================================================
# HELPERS
# ============================================================

def print_header(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def find_source_file() -> Path:
    for path in SOURCE_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "appointments.csv was not found.\n\n"
        "Checked:\n"
        + "\n".join(str(p) for p in SOURCE_CANDIDATES)
    )


def safe_json_value(value):
    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def save_json(data: dict, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            default=safe_json_value
        )


def safe_divide(a, b):
    if b == 0:
        return 0.0
    return float(a) / float(b)


# ============================================================
# MANUAL METRICS
# ============================================================

def manual_roc_auc(y_true, scores):
    """
    Manual ROC-AUC calculation.

    Uses rank-based formulation and does not require sklearn.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    positives = np.sum(y == 1)
    negatives = np.sum(y == 0)

    if positives == 0 or negatives == 0:
        return None

    order = np.argsort(s)
    sorted_scores = s[order]
    sorted_y = y[order]

    ranks = np.empty(len(sorted_scores), dtype=float)

    i = 0

    while i < len(sorted_scores):
        j = i

        while (
            j + 1 < len(sorted_scores)
            and sorted_scores[j + 1] == sorted_scores[i]
        ):
            j += 1

        average_rank = (i + j + 2) / 2.0
        ranks[i:j + 1] = average_rank

        i = j + 1

    positive_rank_sum = np.sum(ranks[sorted_y == 1])

    auc = (
        positive_rank_sum
        - positives * (positives + 1) / 2
    ) / (positives * negatives)

    return float(auc)


def manual_pr_auc(y_true, scores):
    """
    Manual Average Precision / PR-AUC style calculation.

    Scores are sorted from highest to lowest.
    """
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)

    positives = int(np.sum(y == 1))

    if positives == 0:
        return None

    order = np.argsort(-s, kind="mergesort")

    sorted_y = y[order]

    tp = 0
    precision_sum = 0.0

    for rank, actual in enumerate(sorted_y, start=1):
        if actual == 1:
            tp += 1
            precision_sum += tp / rank

    return float(precision_sum / positives)


def classification_metrics(y_true, scores, threshold=0.5):
    y = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    predictions = (scores >= threshold).astype(int)

    tp = int(np.sum((y == 1) & (predictions == 1)))
    tn = int(np.sum((y == 0) & (predictions == 0)))
    fp = int(np.sum((y == 0) & (predictions == 1)))
    fn = int(np.sum((y == 1) & (predictions == 0)))

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    accuracy = safe_divide(tp + tn, len(y))

    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


# ============================================================
# SIGNAL STRENGTH
# ============================================================

def calculate_rate_table(
    df: pd.DataFrame,
    column: str,
    min_group_size: int = 100
) -> pd.DataFrame:

    grouped = (
        df.groupby(column, dropna=False)[TARGET]
        .agg(
            observations="count",
            no_shows="sum",
            no_show_rate="mean"
        )
        .reset_index()
    )

    grouped["show_rate"] = 1.0 - grouped["no_show_rate"]

    overall_rate = df[TARGET].mean()

    grouped["rate_difference_from_overall"] = (
        grouped["no_show_rate"] - overall_rate
    )

    grouped["rate_ratio_to_overall"] = (
        grouped["no_show_rate"] / overall_rate
    )

    grouped = grouped[
        grouped["observations"] >= min_group_size
    ].copy()

    grouped = grouped.sort_values(
        "no_show_rate",
        ascending=False
    )

    return grouped


def calculate_numeric_signal(
    df: pd.DataFrame,
    column: str
) -> dict:

    valid = df[[column, TARGET]].dropna()

    if len(valid) == 0:
        return {
            "feature": column,
            "correlation": None,
            "mean_no_show": None,
            "mean_show": None,
            "difference": None,
        }

    no_show_values = valid.loc[
        valid[TARGET] == 1,
        column
    ]

    show_values = valid.loc[
        valid[TARGET] == 0,
        column
    ]

    correlation = valid[column].corr(valid[TARGET])

    mean_no_show = (
        float(no_show_values.mean())
        if len(no_show_values) > 0
        else None
    )

    mean_show = (
        float(show_values.mean())
        if len(show_values) > 0
        else None
    )

    difference = (
        mean_no_show - mean_show
        if mean_no_show is not None and mean_show is not None
        else None
    )

    return {
        "feature": column,
        "correlation": (
            float(correlation)
            if pd.notna(correlation)
            else None
        ),
        "mean_no_show": mean_no_show,
        "mean_show": mean_show,
        "difference": difference,
    }


# ============================================================
# PATIENT HISTORY AUDIT
# ============================================================

def build_patient_history_audit(df: pd.DataFrame):
    print_header("PATIENT HISTORY SIGNAL AUDIT")

    work = df.sort_values(
        ["patient_id", "appointment_datetime"]
    ).copy()

    work["previous_appointments"] = (
        work.groupby("patient_id")
        .cumcount()
    )

    work["previous_no_shows"] = (
        work.groupby("patient_id")[TARGET]
        .cumsum()
        - work[TARGET]
    )

    work["previous_attended"] = (
        work["previous_appointments"]
        - work["previous_no_shows"]
    )

    work["historical_no_show_rate"] = np.where(
        work["previous_appointments"] > 0,
        work["previous_no_shows"]
        / work["previous_appointments"],
        np.nan
    )

    first_appointments = work[
        work["previous_appointments"] == 0
    ]

    repeat_appointments = work[
        work["previous_appointments"] > 0
    ]

    result = {
        "total_appointments": len(work),
        "first_appointments": len(first_appointments),
        "repeat_appointments": len(repeat_appointments),
        "repeat_appointment_percentage": (
            100 * safe_divide(
                len(repeat_appointments),
                len(work)
            )
        ),
    }

    print(
        f"First appointments:  {len(first_appointments):,}"
    )
    print(
        f"Repeat appointments: {len(repeat_appointments):,}"
    )
    print(
        f"Repeat coverage:      "
        f"{result['repeat_appointment_percentage']:.2f}%"
    )

    history_bins = pd.cut(
        work["previous_appointments"],
        bins=[-1, 0, 1, 2, 4, 9, np.inf],
        labels=[
            "0",
            "1",
            "2",
            "3-4",
            "5-9",
            "10+"
        ]
    )

    history_table = (
        work.assign(
            history_group=history_bins
        )
        .groupby(
            "history_group",
            observed=False
        )[TARGET]
        .agg(
            observations="count",
            no_shows="sum",
            no_show_rate="mean"
        )
        .reset_index()
    )

    history_table["show_rate"] = (
        1 - history_table["no_show_rate"]
    )

    history_table.to_csv(
        OUTPUT_DIR / "no_show_history_signal.csv",
        index=False
    )

    known_history = work[
        work["previous_appointments"] > 0
    ].copy()

    if len(known_history) > 0:
        history_correlation = (
            known_history[
                [
                    "historical_no_show_rate",
                    TARGET
                ]
            ]
            .corr()
            .iloc[0, 1]
        )

        previous_no_show_correlation = (
            known_history[
                [
                    "previous_no_shows",
                    TARGET
                ]
            ]
            .corr()
            .iloc[0, 1]
        )
    else:
        history_correlation = None
        previous_no_show_correlation = None

    result["historical_no_show_rate_correlation"] = (
        float(history_correlation)
        if pd.notna(history_correlation)
        else None
    )

    result["previous_no_show_count_correlation"] = (
        float(previous_no_show_correlation)
        if pd.notna(previous_no_show_correlation)
        else None
    )

    print(
        "Historical no-show-rate correlation: "
        f"{result['historical_no_show_rate_correlation']}"
    )

    print(
        "Previous no-show-count correlation: "
        f"{result['previous_no_show_count_correlation']}"
    )

    return result, work


# ============================================================
# FEATURE ENGINEERING FOR BASELINE
# ============================================================

def create_baseline_features(df: pd.DataFrame):
    work = df.copy()

    work["appointment_month"] = (
        work["appointment_datetime"].dt.month
    )

    work["appointment_day_of_week"] = (
        work["appointment_datetime"].dt.dayofweek
    )

    work["appointment_hour"] = (
        work["appointment_datetime"].dt.hour
    )

    work["appointment_minute"] = (
        work["appointment_datetime"].dt.minute
    )

    work["day_of_month"] = (
        work["appointment_datetime"].dt.day
    )

    work["week_of_year"] = (
        work["appointment_datetime"].dt.isocalendar()
        .week.astype(int)
    )

    work["day_of_year"] = (
        work["appointment_datetime"].dt.dayofyear
    )

    work["is_weekend"] = (
        work["appointment_day_of_week"] >= 5
    ).astype(int)

    work["is_morning"] = (
        work["appointment_hour"] < 12
    ).astype(int)

    work["is_afternoon"] = (
        (
            work["appointment_hour"] >= 12
        )
        &
        (
            work["appointment_hour"] < 17
        )
    ).astype(int)

    work["is_evening"] = (
        work["appointment_hour"] >= 17
    ).astype(int)

    work["age_group"] = pd.cut(
        work["patient_age"],
        bins=[0, 17, 25, 35, 45, 55, 65, 75, 200],
        labels=[
            "0-17",
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "66-75",
            "76+"
        ],
        include_lowest=True
    ).astype(str)

    return work


# ============================================================
# TARGET CONSISTENCY
# ============================================================

def target_consistency_audit(df):
    print_header("TARGET CONSISTENCY AUDIT")

    result = {}

    if "attended" in df.columns:

        attended_values = (
            df["attended"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        expected_no_show = (
            attended_values
            .map({
                "yes": 0,
                "no": 1
            })
        )

        valid = expected_no_show.notna()

        mismatches = (
            df.loc[valid, TARGET].astype(int)
            != expected_no_show.loc[valid].astype(int)
        )

        mismatch_count = int(mismatches.sum())

        result["attended_column_present"] = True
        result["consistency_checked_rows"] = int(valid.sum())
        result["inconsistent_rows"] = mismatch_count
        result["consistency_percentage"] = (
            100 * safe_divide(
                valid.sum() - mismatch_count,
                valid.sum()
            )
        )

        print(
            f"Rows checked:       {valid.sum():,}"
        )
        print(
            f"Inconsistent rows:  {mismatch_count:,}"
        )
        print(
            f"Consistency:        "
            f"{result['consistency_percentage']:.2f}%"
        )

    else:
        result["attended_column_present"] = False
        result["consistency_checked_rows"] = 0
        result["inconsistent_rows"] = None
        result["consistency_percentage"] = None

        print("attended column not available.")

    return result


# ============================================================
# INTERACTION SIGNAL AUDIT
# ============================================================

def interaction_audit(df):
    print_header("INTERACTION SIGNAL AUDIT")

    combinations = [
        ["department", "priority"],
        ["department", "patient_severity"],
        ["department", "age_group"],
        ["priority", "patient_severity"],
        ["appointment_day_of_week", "appointment_hour"],
        ["appointment_day_of_week", "department"],
        ["appointment_hour", "department"],
        ["is_weekend", "department"],
    ]

    summaries = []

    for columns in combinations:

        available = [
            c for c in columns
            if c in df.columns
        ]

        if len(available) != len(columns):
            continue

        grouped = (
            df.groupby(available, dropna=False)[TARGET]
            .agg(
                observations="count",
                no_shows="sum",
                no_show_rate="mean"
            )
            .reset_index()
        )

        grouped = grouped[
            grouped["observations"] >= 100
        ]

        if len(grouped) == 0:
            continue

        min_rate = grouped["no_show_rate"].min()
        max_rate = grouped["no_show_rate"].max()

        summaries.append({
            "combination": " + ".join(columns),
            "groups": len(grouped),
            "minimum_no_show_rate": float(min_rate),
            "maximum_no_show_rate": float(max_rate),
            "range": float(max_rate - min_rate),
        })

    result = pd.DataFrame(summaries)

    if len(result) > 0:
        result = result.sort_values(
            "range",
            ascending=False
        )

        result.to_csv(
            OUTPUT_DIR / "no_show_interaction_signal.csv",
            index=False
        )

        print(result.to_string(index=False))

    return result


# ============================================================
# XGBOOST SIGNAL TEST
# ============================================================

def xgboost_signal_test(df):
    print_header("LIGHTWEIGHT XGBOOST SIGNAL TEST")

    try:
        import xgboost as xgb
    except Exception as exc:
        print(
            "XGBoost could not be imported."
        )
        print(
            f"Reason: {exc}"
        )

        return {
            "available": False,
            "error": str(exc)
        }

    work = df.copy()

    # Keep the model intentionally simple.
    # This is a SIGNAL TEST, not the final production model.

    categorical_columns = [
        "department",
        "priority",
        "patient_severity",
        "age_group",
    ]

    numeric_columns = [
        "duration",
        "patient_age",
        "appointment_month",
        "appointment_day_of_week",
        "appointment_hour",
        "appointment_minute",
        "day_of_month",
        "week_of_year",
        "day_of_year",
        "is_weekend",
        "is_morning",
        "is_afternoon",
        "is_evening",
    ]

    usable_categorical = [
        c for c in categorical_columns
        if c in work.columns
    ]

    usable_numeric = [
        c for c in numeric_columns
        if c in work.columns
    ]

    encoded = pd.get_dummies(
        work[
            usable_categorical
            + usable_numeric
        ],
        columns=usable_categorical,
        dummy_na=True,
        dtype=float
    )

    X = encoded.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0.0)

    y = work[TARGET].astype(int).to_numpy()

    # Chronological 70/30 split.
    split_index = int(len(work) * 0.70)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    positives = int(np.sum(y_train == 1))
    negatives = int(np.sum(y_train == 0))

    scale_pos_weight = (
        safe_divide(negatives, positives)
        if positives > 0
        else 1.0
    )

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=list(X.columns)
    )

    dtest = xgb.DMatrix(
        X_test,
        label=y_test,
        feature_names=list(X.columns)
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "reg_lambda": 5.0,
        "reg_alpha": 0.0,
        "scale_pos_weight": scale_pos_weight,
        "seed": SEED,
        "tree_method": "hist",
        "nthread": max(
            1,
            (os.cpu_count() or 2) - 1
        ),
    }

    start = time.time()

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=300,
        evals=[(dtest, "test")],
        verbose_eval=False
    )

    runtime = time.time() - start

    predictions = model.predict(dtest)

    roc_auc = manual_roc_auc(
        y_test,
        predictions
    )

    pr_auc = manual_pr_auc(
        y_test,
        predictions
    )

    metrics_50 = classification_metrics(
        y_test,
        predictions,
        threshold=0.50
    )

    result = {
        "available": True,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "features": X.shape[1],
        "train_positive": positives,
        "train_negative": negatives,
        "scale_pos_weight": scale_pos_weight,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "runtime_seconds": runtime,
        **metrics_50,
    }

    print(
        f"Train rows: {len(X_train):,}"
    )
    print(
        f"Test rows:  {len(X_test):,}"
    )
    print(
        f"Features:   {X.shape[1]}"
    )
    print(
        f"ROC-AUC:    {roc_auc:.4f}"
        if roc_auc is not None
        else "ROC-AUC:    N/A"
    )
    print(
        f"PR-AUC:     {pr_auc:.4f}"
        if pr_auc is not None
        else "PR-AUC:     N/A"
    )

    # Feature importance
    importance = model.get_score(
        importance_type="gain"
    )

    importance_rows = []

    for feature, gain in importance.items():
        importance_rows.append({
            "feature": feature,
            "gain": float(gain)
        })

    importance_df = pd.DataFrame(
        importance_rows
    )

    if len(importance_df) > 0:
        total_gain = importance_df["gain"].sum()

        importance_df["gain_percentage"] = (
            100
            * importance_df["gain"]
            / total_gain
            if total_gain > 0
            else 0
        )

        importance_df = importance_df.sort_values(
            "gain",
            ascending=False
        )

        importance_df.to_csv(
            OUTPUT_DIR
            / "no_show_signal_xgboost_importance.csv",
            index=False
        )

    return result


# ============================================================
# BASELINE COMPARISON
# ============================================================

def baseline_test(df):
    print_header("BASELINE SIGNAL TEST")

    y = df[TARGET].astype(int).to_numpy()

    positive_rate = float(np.mean(y))

    # Constant probability equal to training prevalence.
    constant_scores = np.full(
        len(y),
        positive_rate,
        dtype=float
    )

    roc_auc = manual_roc_auc(
        y,
        constant_scores
    )

    pr_auc = manual_pr_auc(
        y,
        constant_scores
    )

    result = {
        "positive_rate": positive_rate,
        "constant_probability": positive_rate,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }

    print(
        f"No-show rate: {positive_rate:.4%}"
    )

    print(
        f"Baseline PR-AUC: "
        f"{pr_auc:.4f}"
        if pr_auc is not None
        else "Baseline PR-AUC: N/A"
    )

    return result


# ============================================================
# VISUALIZATIONS
# ============================================================

def create_visualizations(df, categorical_tables):
    try:
        import matplotlib

        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

    except Exception as exc:
        print(
            f"Visualization skipped: {exc}"
        )
        return

    # --------------------------------------------------------
    # Overall target distribution
    # --------------------------------------------------------

    counts = (
        df[TARGET]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(8, 5))

    plt.bar(
        ["Show", "No-show"],
        [
            int(counts.get(0, 0)),
            int(counts.get(1, 0))
        ]
    )

    plt.title("Appointment Outcome Distribution")
    plt.ylabel("Appointments")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "no_show_target_distribution.png",
        dpi=140
    )

    plt.close()

    # --------------------------------------------------------
    # Department no-show rate
    # --------------------------------------------------------

    if "department" in categorical_tables:

        table = categorical_tables["department"]

        plt.figure(figsize=(10, 6))

        plt.bar(
            table["department"].astype(str),
            table["no_show_rate"] * 100
        )

        plt.title(
            "No-Show Rate by Department"
        )

        plt.ylabel("No-show rate (%)")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "no_show_rate_by_department.png",
            dpi=140
        )

        plt.close()

    # --------------------------------------------------------
    # Priority no-show rate
    # --------------------------------------------------------

    if "priority" in categorical_tables:

        table = categorical_tables["priority"]

        plt.figure(figsize=(8, 5))

        plt.bar(
            table["priority"].astype(str),
            table["no_show_rate"] * 100
        )

        plt.title(
            "No-Show Rate by Priority"
        )

        plt.ylabel("No-show rate (%)")

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "no_show_rate_by_priority.png",
            dpi=140
        )

        plt.close()

    # --------------------------------------------------------
    # Hourly no-show rate
    # --------------------------------------------------------

    if "appointment_hour" in df.columns:

        hourly = (
            df.groupby("appointment_hour")[TARGET]
            .mean()
            .reset_index()
        )

        plt.figure(figsize=(10, 5))

        plt.plot(
            hourly["appointment_hour"],
            hourly[TARGET] * 100,
            marker="o"
        )

        plt.title(
            "No-Show Rate by Appointment Hour"
        )

        plt.xlabel("Hour")
        plt.ylabel("No-show rate (%)")

        plt.grid(True, alpha=0.25)

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "no_show_rate_by_hour.png",
            dpi=140
        )

        plt.close()


# ============================================================
# FINAL CONCLUSION
# ============================================================

def determine_signal_conclusion(
    overall_rate,
    categorical_ranges,
    numeric_signals,
    xgb_result,
    history_result
):

    reasons = []

    signal_score = 0

    # --------------------------------------------------------
    # XGBoost ROC-AUC
    # --------------------------------------------------------

    roc_auc = xgb_result.get("roc_auc")

    if roc_auc is not None:

        distance = abs(roc_auc - 0.5)

        if distance >= 0.10:
            signal_score += 3
            reasons.append(
                f"XGBoost ROC-AUC is {roc_auc:.4f}, "
                "showing substantial separation from random ranking."
            )

        elif distance >= 0.05:
            signal_score += 2
            reasons.append(
                f"XGBoost ROC-AUC is {roc_auc:.4f}, "
                "showing some separation from random ranking."
            )

        else:
            reasons.append(
                f"XGBoost ROC-AUC is {roc_auc:.4f}, "
                "which is very close to random ranking."
            )

    # --------------------------------------------------------
    # Categorical range
    # --------------------------------------------------------

    if categorical_ranges:

        largest_range = max(
            categorical_ranges
        )

        if largest_range >= 0.10:
            signal_score += 3
            reasons.append(
                f"The strongest categorical grouping has a "
                f"{largest_range:.3f} no-show-rate spread."
            )

        elif largest_range >= 0.05:
            signal_score += 2
            reasons.append(
                f"Categorical groups show moderate differences, "
                f"with a maximum rate spread of {largest_range:.3f}."
            )

        else:
            reasons.append(
                "Categorical group no-show rates are relatively similar."
            )

    # --------------------------------------------------------
    # Numerical correlations
    # --------------------------------------------------------

    strong_numeric = 0

    for item in numeric_signals:

        corr = item.get("correlation")

        if corr is not None and abs(corr) >= 0.10:
            strong_numeric += 1

    if strong_numeric >= 2:
        signal_score += 2
        reasons.append(
            f"{strong_numeric} numeric features show noticeable "
            "correlation with no-show status."
        )

    elif strong_numeric == 1:
        signal_score += 1
        reasons.append(
            "One numeric feature shows noticeable correlation "
            "with no-show status."
        )

    else:
        reasons.append(
            "Numeric features show weak direct correlation "
            "with no-show status."
        )

    # --------------------------------------------------------
    # Patient history
    # --------------------------------------------------------

    history_corr = history_result.get(
        "historical_no_show_rate_correlation"
    )

    if (
        history_corr is not None
        and abs(history_corr) >= 0.10
    ):
        signal_score += 2
        reasons.append(
            "Historical patient no-show behavior contains "
            "noticeable predictive signal."
        )

    else:
        reasons.append(
            "Historical patient no-show behavior does not show "
            "strong direct signal."
        )

    # --------------------------------------------------------
    # Conclusion
    # --------------------------------------------------------

    if signal_score >= 7:

        conclusion = (
            "STRONG LEARNABLE SIGNAL"
        )

        recommendation = (
            "The dataset appears to contain meaningful predictive "
            "patterns. Continue with model improvement, feature "
            "engineering, calibration and threshold optimization."
        )

    elif signal_score >= 4:

        conclusion = (
            "MODERATE LEARNABLE SIGNAL"
        )

        recommendation = (
            "The dataset contains some useful patterns, but the "
            "signal is not especially strong. Further feature "
            "engineering and model comparison are justified."
        )

    else:

        conclusion = (
            "WEAK OR MOSTLY RANDOM SIGNAL"
        )

        recommendation = (
            "The dataset appears to contain limited predictive "
            "information. Extensive model tuning is unlikely to "
            "produce a reliable improvement unless the synthetic "
            "data-generating process is changed."
        )

    return {
        "signal_score": signal_score,
        "conclusion": conclusion,
        "recommendation": recommendation,
        "evidence": reasons,
    }


# ============================================================
# TEXT REPORT
# ============================================================

def write_report(
    source_file,
    df,
    target_result,
    consistency_result,
    history_result,
    baseline_result,
    categorical_tables,
    numeric_signals,
    interaction_df,
    xgb_result,
    conclusion,
    runtime
):

    report_path = (
        OUTPUT_DIR
        / "NO_SHOW_SIGNAL_AUDIT_REPORT.txt"
    )

    no_show_count = int(
        df[TARGET].sum()
    )

    show_count = int(
        len(df) - no_show_count
    )

    lines = []

    lines.append(
        "CAREOPS-AI — NO-SHOW SIGNAL AUDIT REPORT"
    )

    lines.append("=" * 80)
    lines.append("")

    lines.append(
        "Purpose:"
    )

    lines.append(
        "Determine whether the existing synthetic appointment "
        "dataset contains meaningful learnable signal for predicting "
        "patient no-shows."
    )

    lines.append("")

    lines.append(
        f"Source file: {source_file}"
    )

    lines.append(
        f"Rows: {len(df):,}"
    )

    lines.append(
        f"Columns: {len(df.columns)}"
    )

    lines.append("")

    lines.append(
        "TARGET DISTRIBUTION"
    )

    lines.append("-" * 80)

    lines.append(
        f"Show:    {show_count:,}"
    )

    lines.append(
        f"No-show: {no_show_count:,}"
    )

    lines.append(
        f"No-show rate: "
        f"{target_result['no_show_rate']:.4%}"
    )

    lines.append("")

    lines.append(
        "TARGET CONSISTENCY"
    )

    lines.append("-" * 80)

    lines.append(
        f"Consistency: "
        f"{consistency_result.get('consistency_percentage')}"
    )

    lines.append("")

    lines.append(
        "PATIENT HISTORY"
    )

    lines.append("-" * 80)

    lines.append(
        f"Repeat appointment percentage: "
        f"{history_result.get('repeat_appointment_percentage'):.2f}%"
    )

    lines.append(
        f"Historical no-show-rate correlation: "
        f"{history_result.get('historical_no_show_rate_correlation')}"
    )

    lines.append("")

    lines.append(
        "BASELINE"
    )

    lines.append("-" * 80)

    lines.append(
        f"Baseline PR-AUC: "
        f"{baseline_result.get('pr_auc')}"
    )

    lines.append("")

    lines.append(
        "XGBOOST SIGNAL TEST"
    )

    lines.append("-" * 80)

    if xgb_result.get("available"):

        lines.append(
            f"ROC-AUC: "
            f"{xgb_result.get('roc_auc')}"
        )

        lines.append(
            f"PR-AUC: "
            f"{xgb_result.get('pr_auc')}"
        )

        lines.append(
            f"Accuracy @ 0.50: "
            f"{xgb_result.get('accuracy')}"
        )

        lines.append(
            f"Precision @ 0.50: "
            f"{xgb_result.get('precision')}"
        )

        lines.append(
            f"Recall @ 0.50: "
            f"{xgb_result.get('recall')}"
        )

        lines.append(
            f"F1 @ 0.50: "
            f"{xgb_result.get('f1')}"
        )

    else:

        lines.append(
            f"XGBoost unavailable: "
            f"{xgb_result.get('error')}"
        )

    lines.append("")

    lines.append(
        "FINAL SIGNAL CONCLUSION"
    )

    lines.append("-" * 80)

    lines.append(
        conclusion["conclusion"]
    )

    lines.append("")

    lines.append(
        conclusion["recommendation"]
    )

    lines.append("")

    lines.append(
        "Evidence:"
    )

    for reason in conclusion["evidence"]:
        lines.append(
            f"- {reason}"
        )

    lines.append("")

    lines.append(
        f"Audit runtime: {runtime:.2f} seconds"
    )

    lines.append("")

    lines.append(
        "IMPORTANT:"
    )

    lines.append(
        "This audit does not modify the source appointment dataset "
        "or the existing no-show model."
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(lines)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    overall_start = time.time()

    print_header(
        "CAREOPS-AI — NO-SHOW SIGNAL AUDIT"
    )

    print(
        "This program investigates the data before further model tuning."
    )

    print(
        "Existing source data will NOT be regenerated or modified."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find source
    # --------------------------------------------------------

    source_file = find_source_file()

    print()
    print(
        f"Source: {source_file}"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print_header(
        "LOADING APPOINTMENT DATA"
    )

    df = pd.read_csv(
        source_file,
        low_memory=False
    )

    print(
        f"Shape: {df.shape}"
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    missing_expected = [
        c
        for c in EXPECTED_COLUMNS
        if c not in df.columns
    ]

    if missing_expected:

        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(missing_expected)
        )

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    # --------------------------------------------------------
    # Clean target
    # --------------------------------------------------------

    df[TARGET] = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    invalid_target = (
        df[TARGET].isna()
        | ~df[TARGET].isin([0, 1])
    )

    if invalid_target.any():

        raise ValueError(
            f"Found {invalid_target.sum():,} invalid "
            "no_show target values."
        )

    df[TARGET] = df[TARGET].astype(int)

    # --------------------------------------------------------
    # Date/time
    # --------------------------------------------------------

    date_part = pd.to_datetime(
        df["date"],
        errors="coerce"
    )

    time_part = (
        df["time"]
        .astype(str)
        .str.strip()
    )

    df["appointment_datetime"] = pd.to_datetime(
        date_part.dt.strftime("%Y-%m-%d")
        + " "
        + time_part,
        errors="coerce"
    )

    if df["appointment_datetime"].isna().any():

        print(
            "WARNING: Some appointment datetimes could not be parsed."
        )

        df = df.dropna(
            subset=["appointment_datetime"]
        ).copy()

    df = df.sort_values(
        "appointment_datetime"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Target audit
    # --------------------------------------------------------

    target_result = {
        "show_count": int(
            (df[TARGET] == 0).sum()
        ),
        "no_show_count": int(
            (df[TARGET] == 1).sum()
        ),
        "total": len(df),
        "no_show_rate": float(
            df[TARGET].mean()
        ),
    }

    print_header(
        "TARGET DISTRIBUTION"
    )

    print(
        f"Total appointments: {target_result['total']:,}"
    )

    print(
        f"Shows:              {target_result['show_count']:,}"
    )

    print(
        f"No-shows:           {target_result['no_show_count']:,}"
    )

    print(
        f"No-show rate:       "
        f"{target_result['no_show_rate']:.4%}"
    )

    # --------------------------------------------------------
    # Target consistency
    # --------------------------------------------------------

    consistency_result = target_consistency_audit(
        df
    )

    # --------------------------------------------------------
    # Feature creation
    # --------------------------------------------------------

    df = create_baseline_features(
        df
    )

    # --------------------------------------------------------
    # Patient history
    # --------------------------------------------------------

    history_result, history_work = (
        build_patient_history_audit(df)
    )

    # Add history features to the audit dataset.
    for column in [
        "previous_appointments",
        "previous_no_shows",
        "previous_attended",
        "historical_no_show_rate",
    ]:

        if column in history_work.columns:

            df[column] = history_work[
                column
            ].to_numpy()

    # --------------------------------------------------------
    # Categorical signal
    # --------------------------------------------------------

    print_header(
        "CATEGORICAL SIGNAL AUDIT"
    )

    categorical_columns = [
        "department",
        "priority",
        "patient_severity",
        "age_group",
        "appointment_day_of_week",
        "appointment_hour",
        "appointment_month",
        "is_weekend",
        "is_morning",
        "is_afternoon",
        "is_evening",
    ]

    categorical_tables = {}
    categorical_ranges = []

    for column in categorical_columns:

        if column not in df.columns:
            continue

        table = calculate_rate_table(
            df,
            column,
            min_group_size=100
        )

        if len(table) == 0:
            continue

        categorical_tables[column] = table

        table.to_csv(
            OUTPUT_DIR
            / f"no_show_signal_{column}.csv",
            index=False
        )

        rate_range = (
            table["no_show_rate"].max()
            - table["no_show_rate"].min()
        )

        categorical_ranges.append(
            float(rate_range)
        )

        print()
        print(
            f"{column}: "
            f"{len(table)} groups"
        )

        print(
            f"Rate range: "
            f"{table['no_show_rate'].min():.4%}"
            f" -> "
            f"{table['no_show_rate'].max():.4%}"
        )

    # --------------------------------------------------------
    # Numeric signal
    # --------------------------------------------------------

    print_header(
        "NUMERIC SIGNAL AUDIT"
    )

    numeric_columns = [
        "duration",
        "patient_age",
        "previous_appointments",
        "previous_no_shows",
        "previous_attended",
        "historical_no_show_rate",
        "appointment_hour",
        "appointment_minute",
        "day_of_month",
        "week_of_year",
        "day_of_year",
    ]

    numeric_signals = []

    for column in numeric_columns:

        if column not in df.columns:
            continue

        result = calculate_numeric_signal(
            df,
            column
        )

        numeric_signals.append(
            result
        )

    numeric_df = pd.DataFrame(
        numeric_signals
    )

    if len(numeric_df) > 0:

        numeric_df["absolute_correlation"] = (
            numeric_df["correlation"]
            .abs()
        )

        numeric_df = numeric_df.sort_values(
            "absolute_correlation",
            ascending=False
        )

        numeric_df.to_csv(
            OUTPUT_DIR
            / "no_show_numeric_signal.csv",
            index=False
        )

        print(
            numeric_df.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Interactions
    # --------------------------------------------------------

    interaction_df = interaction_audit(
        df
    )

    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    baseline_result = baseline_test(
        df
    )

    # --------------------------------------------------------
    # XGBoost signal test
    # --------------------------------------------------------

    xgb_result = xgboost_signal_test(
        df
    )

    # --------------------------------------------------------
    # Final conclusion
    # --------------------------------------------------------

    conclusion = determine_signal_conclusion(
        target_result["no_show_rate"],
        categorical_ranges,
        numeric_signals,
        xgb_result,
        history_result
    )

    print_header(
        "FINAL DATA SIGNAL CONCLUSION"
    )

    print(
        conclusion["conclusion"]
    )

    print()

    print(
        conclusion["recommendation"]
    )

    print()

    for reason in conclusion["evidence"]:
        print(
            f"- {reason}"
        )

    # --------------------------------------------------------
    # Visualizations
    # --------------------------------------------------------

    print_header(
        "CREATING AUDIT VISUALIZATIONS"
    )

    create_visualizations(
        df,
        categorical_tables
    )

    # --------------------------------------------------------
    # Save JSON summary
    # --------------------------------------------------------

    runtime = (
        time.time()
        - overall_start
    )

    summary = {
        "audit": {
            "project": "CareOps-AI",
            "purpose": (
                "Determine whether synthetic appointment data "
                "contains learnable no-show signal."
            ),
            "source_file": str(source_file),
            "source_data_modified": False,
            "seed": SEED,
            "runtime_seconds": runtime,
        },

        "dataset": {
            "rows": len(df),
            "columns": len(df.columns),
            "date_min": df[
                "appointment_datetime"
            ].min(),
            "date_max": df[
                "appointment_datetime"
            ].max(),
        },

        "target": target_result,

        "target_consistency": consistency_result,

        "patient_history": history_result,

        "baseline": baseline_result,

        "xgboost_signal_test": xgb_result,

        "final_signal_assessment": conclusion,
    }

    save_json(
        summary,
        OUTPUT_DIR
        / "no_show_signal_audit_summary.json"
    )

    # --------------------------------------------------------
    # Final report
    # --------------------------------------------------------

    write_report(
        source_file,
        df,
        target_result,
        consistency_result,
        history_result,
        baseline_result,
        categorical_tables,
        numeric_signals,
        interaction_df,
        xgb_result,
        conclusion,
        runtime
    )

    # --------------------------------------------------------
    # Output verification
    # --------------------------------------------------------

    print_header(
        "OUTPUT VERIFICATION"
    )

    expected_outputs = [
        "no_show_signal_audit_summary.json",
        "NO_SHOW_SIGNAL_AUDIT_REPORT.txt",
        "no_show_history_signal.csv",
        "no_show_numeric_signal.csv",
        "no_show_interaction_signal.csv",
        "no_show_target_distribution.png",
        "no_show_rate_by_department.png",
        "no_show_rate_by_priority.png",
        "no_show_rate_by_hour.png",
    ]

    if xgb_result.get("available"):
        expected_outputs.append(
            "no_show_signal_xgboost_importance.csv"
        )

    verified = 0

    for filename in expected_outputs:

        path = OUTPUT_DIR / filename

        if path.exists() and path.stat().st_size > 0:

            print(
                f"[OK] {filename}"
            )

            verified += 1

        else:

            print(
                f"[MISSING] {filename}"
            )

    print()
    print(
        f"Verified outputs: "
        f"{verified}/{len(expected_outputs)}"
    )

    print()
    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )

    print()
    print(
        f"Total audit runtime: "
        f"{runtime:.2f} seconds"
    )

    print_header(
        "AUDIT COMPLETE"
    )

    print(
        "The existing appointment dataset was not regenerated."
    )

    print(
        "The existing no-show model was not overwritten."
    )

    print(
        "Read NO_SHOW_SIGNAL_AUDIT_REPORT.txt for the final conclusion."
    )


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "Audit cancelled by user."
        )

        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 80)
        print("AUDIT FAILED")
        print("=" * 80)
        print(
            f"Error: {exc}"
        )

        import traceback

        traceback.print_exc()

        sys.exit(1)
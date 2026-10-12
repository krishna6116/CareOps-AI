# ================================================================
# CAREOPS-AI
# NO-SHOW PREDICTION — ENHANCED BEHAVIORAL MODEL
#
# Purpose:
#   Predict appointment no-shows using leakage-safe historical
#   patient behavior and contextual interaction features.
#
# IMPORTANT:
#   - Does NOT regenerate synthetic appointment data.
#   - Does NOT modify appointments.csv.
#   - Historical features use ONLY information available before
#     the current appointment.
#   - Uses chronological train / validation / test splitting.
#   - Uses native XGBoost; no sklearn or scipy required.
# ================================================================

from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime
import json
import math
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classification_evaluation import (
    evaluate_classifiers,
    logistic_regression_baseline,
    prevalence_baseline,
    reliability_table,
)
from forecast_evaluation import model_status
from scenario_config import WARMUP_END

warnings.filterwarnings("ignore")


# ================================================================
# CONFIGURATION
# ================================================================

SEED = 42

ROOT = Path(__file__).resolve().parents[2]

SCENARIO_DATA_DIR = os.environ.get("CAREOPS_DATA_DIR")
DATA_CANDIDATES = (
    [Path(SCENARIO_DATA_DIR) / "appointments.csv"]
    if SCENARIO_DATA_DIR
    else [
        ROOT / "data" / "processed" / "appointments.csv",
        ROOT / "data" / "appointments.csv",
        ROOT / "data" / "synthetic" / "appointments.csv",
    ]
)

OUTPUT_DIR = ROOT / "ml" / "no_show_prediction"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_FILE = None


# ================================================================
# OUTPUT FILES
# ================================================================

MODEL_FILE = OUTPUT_DIR / "no_show_xgboost_model.json"
PREDICTIONS_FILE = OUTPUT_DIR / "no_show_predictions.csv"
METRICS_FILE = OUTPUT_DIR / "no_show_metrics.json"
METADATA_FILE = OUTPUT_DIR / "no_show_model_metadata.json"
IMPORTANCE_FILE = OUTPUT_DIR / "no_show_feature_importance.csv"
DATA_QUALITY_FILE = OUTPUT_DIR / "no_show_data_quality.csv"
TRAINING_SUMMARY_FILE = OUTPUT_DIR / "no_show_training_summary.csv"

CONFUSION_MATRIX_PNG = OUTPUT_DIR / "no_show_confusion_matrix.png"
FEATURE_IMPORTANCE_PNG = OUTPUT_DIR / "no_show_feature_importance.png"
PROBABILITY_PNG = OUTPUT_DIR / "no_show_probability_distribution.png"

REPORT_FILE = OUTPUT_DIR / "NO_SHOW_MODEL_REPORT.txt"


# ================================================================
# XGBOOST IMPORT
# ================================================================

try:
    import xgboost as xgb
except ImportError:
    print("\nERROR: XGBoost is not installed.")
    print("Run:")
    print("pip install xgboost")
    sys.exit(1)


# ================================================================
# GENERAL HELPERS
# ================================================================

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(value):
    try:
        value = float(value)
        if math.isfinite(value):
            return value
    except Exception:
        pass
    return 0.0


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def fit_probability_calibrator(probabilities, targets):
    """Fit a leakage-safe two-parameter Platt calibrator on validation data."""
    raw = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(targets, dtype=float)
    logits = np.log(raw / (1.0 - raw))
    scale = 1.0
    offset = 0.0
    for _ in range(80):
        calibrated = sigmoid(scale * logits + offset)
        error = calibrated - y
        gradient = np.array([
            np.sum(error * logits),
            np.sum(error),
        ])
        weights = calibrated * (1.0 - calibrated)
        hessian = np.array([
            [np.sum(weights * logits * logits), np.sum(weights * logits)],
            [np.sum(weights * logits), np.sum(weights)],
        ]) + np.eye(2) * 1e-6
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        scale -= float(step[0])
        offset -= float(step[1])
        if float(np.max(np.abs(step))) < 1e-6:
            break
    return float(scale), float(offset)


def apply_probability_calibrator(probabilities, scale, offset):
    raw = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(raw / (1.0 - raw))
    return sigmoid(scale * logits + offset)


def smoothed_rate(successes, total, prior_rate=0.104825, smoothing=10.0):
    """
    Bayesian-style smoothed rate.

    This prevents tiny historical groups from producing extreme
    probabilities.
    """
    if total <= 0:
        return float(prior_rate)

    return float(
        (successes + smoothing * prior_rate)
        / (total + smoothing)
    )


# ================================================================
# FIND SOURCE DATA
# ================================================================

def find_source_file():
    for path in DATA_CANDIDATES:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find appointments.csv in:\n"
        + "\n".join(str(p) for p in DATA_CANDIDATES)
    )


# ================================================================
# LOAD DATA
# ================================================================

def load_data():

    global SOURCE_FILE

    SOURCE_FILE = find_source_file()

    print(f"\nSource: {SOURCE_FILE}")

    df = pd.read_csv(SOURCE_FILE)

    print(f"Raw shape: {df.shape}")

    required = [
        "appointment_id",
        "patient_id",
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

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    return df


# ================================================================
# NORMALIZE DATA
# ================================================================

def normalize_data(df):

    df = df.copy()

    # Convert identifiers to strings
    df["appointment_id"] = df["appointment_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)

    # Clean categorical variables
    categorical_columns = [
        "department",
        "priority",
        "patient_severity",
    ]

    for col in categorical_columns:
        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

    # Numeric fields
    df["duration"] = pd.to_numeric(
        df["duration"],
        errors="coerce"
    ).fillna(df["duration"].median())

    df["patient_age"] = pd.to_numeric(
        df["patient_age"],
        errors="coerce"
    )

    df["patient_age"] = df["patient_age"].fillna(
        df["patient_age"].median()
    )

    df["no_show"] = pd.to_numeric(
        df["no_show"],
        errors="coerce"
    )

    if df["no_show"].isna().any():
        raise ValueError("Target contains missing values.")

    df["no_show"] = df["no_show"].astype(int)

    # Combine date and time
    df["appointment_datetime"] = pd.to_datetime(
        df["date"].astype(str)
        + " "
        + df["time"].astype(str),
        errors="coerce"
    )

    if df["appointment_datetime"].isna().any():
        bad = int(df["appointment_datetime"].isna().sum())
        raise ValueError(
            f"Could not parse {bad} appointment date/time values."
        )

    # Sort chronologically.
    # appointment_id is included as a deterministic tie-breaker.
    df = df.sort_values(
        ["appointment_datetime", "appointment_id"]
    ).reset_index(drop=True)

    return df


# ================================================================
# TARGET CONSISTENCY
# ================================================================

def validate_target(df):

    consistency = (
        (
            (df["attended"].astype(str).str.lower() == "yes")
            & (df["no_show"] == 0)
        )
        |
        (
            (df["attended"].astype(str).str.lower() != "yes")
            & (df["no_show"] == 1)
        )
    )

    inconsistent = int((~consistency).sum())

    print_section("TARGET VALIDATION")

    print(f"Rows:          {len(df):,}")
    print(f"Inconsistent:  {inconsistent:,}")
    print(
        f"Consistency:   "
        f"{100 * (1 - inconsistent / len(df)):.2f}%"
    )

    if inconsistent > 0:
        raise ValueError(
            "Target inconsistency detected."
        )


# ================================================================
# BASE TIME FEATURES
# ================================================================

def create_base_features(df):

    out = pd.DataFrame(index=df.index)

    dt = df["appointment_datetime"]

    out["patient_age"] = df["patient_age"].astype(float)
    out["duration"] = df["duration"].astype(float)

    # These fields are known when an appointment is made.  They are
    # intentionally optional so older, externally supplied datasets remain
    # compatible while the synthetic scenario exposes realistic intervention
    # signals.
    if "lead_time_days" in df.columns:
        out["lead_time_days"] = pd.to_numeric(
            df["lead_time_days"], errors="coerce"
        ).fillna(0).clip(lower=0)
    if "reminder_sent" in df.columns:
        out["reminder_sent"] = pd.to_numeric(
            df["reminder_sent"], errors="coerce"
        ).fillna(0).clip(lower=0, upper=1)

    out["day_of_week"] = dt.dt.dayofweek
    out["day_of_month"] = dt.dt.day
    out["month"] = dt.dt.month
    out["quarter"] = dt.dt.quarter
    out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    out["day_of_year"] = dt.dt.dayofyear

    out["appointment_hour"] = dt.dt.hour
    out["appointment_minute"] = dt.dt.minute

    out["is_weekend"] = (
        dt.dt.dayofweek >= 5
    ).astype(int)

    out["is_morning"] = (
        dt.dt.hour < 12
    ).astype(int)

    out["is_afternoon"] = (
        (dt.dt.hour >= 12)
        & (dt.dt.hour < 17)
    ).astype(int)

    out["is_evening"] = (
        dt.dt.hour >= 17
    ).astype(int)

    # Age groups
    out["age_group"] = pd.cut(
        df["patient_age"],
        bins=[0, 18, 30, 40, 50, 60, 70, 80, 150],
        labels=[
            "0_18",
            "19_30",
            "31_40",
            "41_50",
            "51_60",
            "61_70",
            "71_80",
            "81_plus",
        ],
        include_lowest=True,
    ).astype(str)

    # Cyclic time representation
    out["hour_sin"] = np.sin(
        2 * np.pi * out["appointment_hour"] / 24
    )

    out["hour_cos"] = np.cos(
        2 * np.pi * out["appointment_hour"] / 24
    )

    out["dow_sin"] = np.sin(
        2 * np.pi * out["day_of_week"] / 7
    )

    out["dow_cos"] = np.cos(
        2 * np.pi * out["day_of_week"] / 7
    )

    out["month_sin"] = np.sin(
        2 * np.pi * out["month"] / 12
    )

    out["month_cos"] = np.cos(
        2 * np.pi * out["month"] / 12
    )

    out["day_of_year_sin"] = np.sin(
        2 * np.pi * out["day_of_year"] / 365.25
    )

    out["day_of_year_cos"] = np.cos(
        2 * np.pi * out["day_of_year"] / 365.25
    )

    return out


# ================================================================
# HISTORICAL STATE
# ================================================================

def create_historical_features(df):

    print_section(
        "GENERATING LEAKAGE-SAFE HISTORICAL FEATURES"
    )

    n = len(df)

    # Prediction time is the day before the appointment: features use only
    # outcomes from earlier days. The smoothing prior is the running no-show
    # rate over those days (Beta(1, 1) before any history), never the rate
    # over the whole dataset.
    global_counts = [0, 0]
    global_rate = 0.5
    pending = []
    current_day = None

    # Output
    hist = pd.DataFrame(
        index=df.index
    )

    feature_names = [
        "patient_previous_appointments",
        "patient_previous_no_shows",
        "patient_previous_attended",
        "patient_historical_no_show_rate",
        "patient_recent_no_show_rate_30d",
        "patient_recent_no_show_rate_90d",
        "patient_recent_no_show_rate_365d",
        "patient_recent_appointments_30d",
        "patient_recent_appointments_90d",
        "patient_recent_appointments_365d",
        "patient_recent_no_shows_30d",
        "patient_recent_no_shows_90d",
        "patient_recent_no_shows_365d",
        "patient_days_since_previous",
        "patient_last_was_no_show",
        "patient_consecutive_previous_no_shows",
        "patient_recency_weighted_no_show_rate",
        "patient_department_previous",
        "patient_department_no_show_rate",
        "patient_priority_previous",
        "patient_priority_no_show_rate",
        "department_historical_no_show_rate",
        "priority_historical_no_show_rate",
        "severity_historical_no_show_rate",
        "department_age_no_show_rate",
        "department_priority_no_show_rate",
        "priority_severity_no_show_rate",
        "department_hour_no_show_rate",
        "day_hour_no_show_rate",
    ]

    for col in feature_names:
        hist[col] = 0.0

    # ------------------------------------------------------------
    # State dictionaries
    # ------------------------------------------------------------

    patient_state = {}

    dept_state = defaultdict(lambda: [0, 0])
    priority_state = defaultdict(lambda: [0, 0])
    severity_state = defaultdict(lambda: [0, 0])

    patient_dept_state = defaultdict(lambda: [0, 0])
    patient_priority_state = defaultdict(lambda: [0, 0])

    dept_age_state = defaultdict(lambda: [0, 0])
    dept_priority_state = defaultdict(lambda: [0, 0])
    priority_severity_state = defaultdict(lambda: [0, 0])
    dept_hour_state = defaultdict(lambda: [0, 0])
    day_hour_state = defaultdict(lambda: [0, 0])

    def apply_update(record):
        (p, target, dt, department, priority, severity,
         pd_key, pp_key, age_bucket, hour, dow) = record

        p["count"] += 1
        p["no_shows"] += target
        p["attended"] += 1 - target

        p["last_dt"] = dt
        p["last_no_show"] = target

        if target == 1:
            p["consecutive_no_shows"] += 1
        else:
            p["consecutive_no_shows"] = 0

        p["history"].append(
            (dt, target)
        )

        # Context states
        dept_state[department][0] += 1
        dept_state[department][1] += target

        priority_state[priority][0] += 1
        priority_state[priority][1] += target

        severity_state[severity][0] += 1
        severity_state[severity][1] += target

        patient_dept_state[pd_key][0] += 1
        patient_dept_state[pd_key][1] += target

        patient_priority_state[pp_key][0] += 1
        patient_priority_state[pp_key][1] += target

        dept_age_state[
            (department, age_bucket)
        ][0] += 1

        dept_age_state[
            (department, age_bucket)
        ][1] += target

        dept_priority_state[
            (department, priority)
        ][0] += 1

        dept_priority_state[
            (department, priority)
        ][1] += target

        priority_severity_state[
            (priority, severity)
        ][0] += 1

        priority_severity_state[
            (priority, severity)
        ][1] += target

        dept_hour_state[
            (department, hour)
        ][0] += 1

        dept_hour_state[
            (department, hour)
        ][1] += target

        day_hour_state[
            (dow, hour)
        ][0] += 1

        day_hour_state[
            (dow, hour)
        ][1] += target

    # ------------------------------------------------------------
    # Process rows chronologically
    # ------------------------------------------------------------

    for i, row in enumerate(df.itertuples(index=False)):

        patient_id = str(row.patient_id)
        department = str(row.department)
        priority = str(row.priority)
        severity = str(row.patient_severity)

        dt = row.appointment_datetime

        if current_day is not None and dt.normalize() != current_day:
            for record in pending:
                apply_update(record)
                global_counts[0] += 1
                global_counts[1] += record[1]
            pending.clear()
            global_rate = (global_counts[1] + 1) / (global_counts[0] + 2)
        current_day = dt.normalize()
        target = int(row.no_show)

        hour = int(dt.hour)
        dow = int(dt.dayofweek)

        age = float(row.patient_age)

        age_bucket = (
            "0_18" if age <= 18 else
            "19_30" if age <= 30 else
            "31_40" if age <= 40 else
            "41_50" if age <= 50 else
            "51_60" if age <= 60 else
            "61_70" if age <= 70 else
            "71_80" if age <= 80 else
            "81_plus"
        )

        # --------------------------------------------------------
        # Patient state
        # --------------------------------------------------------

        if patient_id not in patient_state:

            patient_state[patient_id] = {
                "count": 0,
                "no_shows": 0,
                "attended": 0,
                "last_dt": None,
                "last_no_show": 0,
                "consecutive_no_shows": 0,
                "history": deque(),
            }

        p = patient_state[patient_id]

        previous_count = p["count"]
        previous_no_shows = p["no_shows"]
        previous_attended = p["attended"]

        hist.loc[i, "patient_previous_appointments"] = (
            previous_count
        )

        hist.loc[i, "patient_previous_no_shows"] = (
            previous_no_shows
        )

        hist.loc[i, "patient_previous_attended"] = (
            previous_attended
        )

        hist.loc[i, "patient_historical_no_show_rate"] = (
            smoothed_rate(
                previous_no_shows,
                previous_count,
                global_rate,
                smoothing=10
            )
        )

        # --------------------------------------------------------
        # Recency windows
        # --------------------------------------------------------

        history = p["history"]

        cutoff_30 = dt - pd.Timedelta(days=30)
        cutoff_90 = dt - pd.Timedelta(days=90)
        cutoff_365 = dt - pd.Timedelta(days=365)

        while history and history[0][0] < cutoff_365:
            history.popleft()

        count_30 = 0
        ns_30 = 0

        count_90 = 0
        ns_90 = 0

        count_365 = 0
        ns_365 = 0

        weighted_total = 0.0
        weighted_no_show = 0.0

        for old_dt, old_target in history:

            days_old = max(
                0.0,
                (dt - old_dt).total_seconds()
                / 86400.0
            )

            if old_dt >= cutoff_30:
                count_30 += 1
                ns_30 += old_target

            if old_dt >= cutoff_90:
                count_90 += 1
                ns_90 += old_target

            if old_dt >= cutoff_365:
                count_365 += 1
                ns_365 += old_target

            # 60-day exponential decay
            weight = math.exp(
                -days_old / 60.0
            )

            weighted_total += weight
            weighted_no_show += (
                weight * old_target
            )

        hist.loc[i, "patient_recent_appointments_30d"] = count_30
        hist.loc[i, "patient_recent_appointments_90d"] = count_90
        hist.loc[i, "patient_recent_appointments_365d"] = count_365

        hist.loc[i, "patient_recent_no_shows_30d"] = ns_30
        hist.loc[i, "patient_recent_no_shows_90d"] = ns_90
        hist.loc[i, "patient_recent_no_shows_365d"] = ns_365

        hist.loc[i, "patient_recent_no_show_rate_30d"] = (
            smoothed_rate(
                ns_30,
                count_30,
                global_rate,
                smoothing=5
            )
        )

        hist.loc[i, "patient_recent_no_show_rate_90d"] = (
            smoothed_rate(
                ns_90,
                count_90,
                global_rate,
                smoothing=5
            )
        )

        hist.loc[i, "patient_recent_no_show_rate_365d"] = (
            smoothed_rate(
                ns_365,
                count_365,
                global_rate,
                smoothing=5
            )
        )

        if p["last_dt"] is not None:

            days_since = (
                dt - p["last_dt"]
            ).total_seconds() / 86400.0

            hist.loc[
                i,
                "patient_days_since_previous"
            ] = max(days_since, 0.0)

        hist.loc[
            i,
            "patient_last_was_no_show"
        ] = p["last_no_show"]

        hist.loc[
            i,
            "patient_consecutive_previous_no_shows"
        ] = p["consecutive_no_shows"]

        if weighted_total > 0:

            hist.loc[
                i,
                "patient_recency_weighted_no_show_rate"
            ] = (
                weighted_no_show / weighted_total
            )

        else:

            hist.loc[
                i,
                "patient_recency_weighted_no_show_rate"
            ] = global_rate

        # --------------------------------------------------------
        # Patient + department history
        # --------------------------------------------------------

        pd_key = (
            patient_id,
            department
        )

        pd_total, pd_ns = patient_dept_state[pd_key]

        hist.loc[
            i,
            "patient_department_previous"
        ] = pd_total

        hist.loc[
            i,
            "patient_department_no_show_rate"
        ] = smoothed_rate(
            pd_ns,
            pd_total,
            global_rate,
            smoothing=5
        )

        # --------------------------------------------------------
        # Patient + priority history
        # --------------------------------------------------------

        pp_key = (
            patient_id,
            priority
        )

        pp_total, pp_ns = patient_priority_state[pp_key]

        hist.loc[
            i,
            "patient_priority_previous"
        ] = pp_total

        hist.loc[
            i,
            "patient_priority_no_show_rate"
        ] = smoothed_rate(
            pp_ns,
            pp_total,
            global_rate,
            smoothing=5
        )

        # --------------------------------------------------------
        # Global contextual history
        # --------------------------------------------------------

        total, ns = dept_state[department]

        hist.loc[
            i,
            "department_historical_no_show_rate"
        ] = smoothed_rate(
            ns,
            total,
            global_rate,
            smoothing=10
        )

        total, ns = priority_state[priority]

        hist.loc[
            i,
            "priority_historical_no_show_rate"
        ] = smoothed_rate(
            ns,
            total,
            global_rate,
            smoothing=10
        )

        total, ns = severity_state[severity]

        hist.loc[
            i,
            "severity_historical_no_show_rate"
        ] = smoothed_rate(
            ns,
            total,
            global_rate,
            smoothing=10
        )

        # --------------------------------------------------------
        # Strong interactions discovered by audit
        # --------------------------------------------------------

        key = (
            department,
            age_bucket
        )

        total, ns = dept_age_state[key]

        hist.loc[
            i,
            "department_age_no_show_rate"
        ] = smoothed_rate(
            ns,
            total,
            global_rate,
            smoothing=15
        )

        key = (
            department,
            priority
        )

        total, ns = dept_priority_state[key]

        hist.loc[
            i,
            "department_priority_no_show_rate"
        ] = smoothed_rate(
            ns,
            total,
            global_rate,
            smoothing=15
        )

        key = (
            priority,
            severity
        )

        total, ns = priority_severity_state[key]

        hist.loc[
            i,
            "priority_severity_no_show_rate"
        ] = smoothed_rate(
            ns,
            total,
            global_rate,
            smoothing=15
        )

        key = (
            department,
            hour
        )

        total, ns = dept_hour_state[key]

        hist.loc[
            i,
            "department_hour_no_show_rate"
        ] = smoothed_rate(
            ns,
            total,
            global_rate,
            smoothing=15
        )

        key = (
            dow,
            hour
        )

        total, ns = day_hour_state[key]

        hist.loc[
            i,
            "day_hour_no_show_rate"
        ] = smoothed_rate(
            ns,
            total,
            global_rate,
            smoothing=15
        )

        # --------------------------------------------------------
        # Outcomes are applied only when the date changes, so an
        # appointment never sees outcomes from its own day.
        # --------------------------------------------------------

        pending.append((
            p, target, dt, department, priority, severity,
            pd_key, pp_key, age_bucket, hour, dow,
        ))

        if (i + 1) % 20000 == 0:

            print(
                f"Processed historical features: "
                f"{i + 1:,}/{n:,}"
            )

    hist = hist.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0.0)

    print(
        f"Historical feature generation complete: "
        f"{len(feature_names)} features"
    )

    return hist


# ================================================================
# BUILD MODEL DATASET
# ================================================================

def build_model_dataset(df, historical):

    base = create_base_features(df)

    categorical = pd.DataFrame(index=df.index)

    categorical["department"] = (
        df["department"].astype(str)
    )

    categorical["priority"] = (
        df["priority"].astype(str)
    )

    categorical["patient_severity"] = (
        df["patient_severity"].astype(str)
    )

    categorical["age_group"] = (
        base["age_group"].astype(str)
    )

    categorical = pd.get_dummies(
        categorical,
        columns=[
            "department",
            "priority",
            "patient_severity",
            "age_group",
        ],
        dtype=float
    )

    numeric = base.drop(
        columns=["age_group"]
    ).astype(float)

    X = pd.concat(
        [
            numeric,
            historical.astype(float),
            categorical,
        ],
        axis=1
    )

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0.0)

    # Ensure deterministic numeric matrix
    X = X.astype(np.float32)

    y = df["no_show"].astype(int)

    return X, y


# ================================================================
# CHRONOLOGICAL SPLIT
# ================================================================

def chronological_split(X, y, df):

    n = len(df)

    train_end = int(n * 0.65)
    validation_end = int(n * 0.80)

    train_idx = np.arange(
        0,
        train_end
    )

    validation_idx = np.arange(
        train_end,
        validation_end
    )

    test_idx = np.arange(
        validation_end,
        n
    )

    print_section("CHRONOLOGICAL DATA SPLIT")

    for name, idx in [
        ("TRAIN", train_idx),
        ("VALIDATION", validation_idx),
        ("TEST", test_idx),
    ]:

        print(
            f"{name:<12}: "
            f"{len(idx):,} rows | "
            f"{df.iloc[idx]['appointment_datetime'].min()} "
            f"-> "
            f"{df.iloc[idx]['appointment_datetime'].max()}"
        )

    return (
        train_idx,
        validation_idx,
        test_idx,
    )


# ================================================================
# MANUAL ROC-AUC
# ================================================================

def roc_auc_manual(y_true, scores):

    y_true = np.asarray(y_true)
    scores = np.asarray(scores)

    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return 0.5

    # Efficient rank-based AUC
    order = np.argsort(scores)

    ranks = np.empty_like(
        order,
        dtype=float
    )

    ranks[order] = np.arange(
        1,
        len(scores) + 1
    )

    pos_ranks = ranks[y_true == 1]

    auc = (
        pos_ranks.sum()
        - len(positives) * (len(positives) + 1) / 2
    ) / (
        len(positives) * len(negatives)
    )

    return float(auc)


# ================================================================
# MANUAL PR-AUC
# ================================================================

def pr_auc_manual(y_true, scores):

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    positives = int(y_true.sum())

    if positives == 0:
        return 0.0

    order = np.argsort(
        -scores,
        kind="mergesort"
    )

    y_sorted = y_true[order]

    tp = np.cumsum(
        y_sorted
    )

    fp = np.cumsum(
        1 - y_sorted
    )

    precision = tp / np.maximum(
        tp + fp,
        1
    )

    recall = tp / positives

    recall_prev = np.concatenate(
        ([0.0], recall[:-1])
    )

    return float(
        np.sum(
            (recall - recall_prev)
            * precision
        )
    )


# ================================================================
# CLASSIFICATION METRICS
# ================================================================

def classification_metrics(
    y_true,
    probabilities,
    threshold
):

    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities)

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tp = int(
        np.sum(
            (y_true == 1)
            & (predictions == 1)
        )
    )

    tn = int(
        np.sum(
            (y_true == 0)
            & (predictions == 0)
        )
    )

    fp = int(
        np.sum(
            (y_true == 0)
            & (predictions == 1)
        )
    )

    fn = int(
        np.sum(
            (y_true == 1)
            & (predictions == 0)
        )
    )

    accuracy = (
        (tp + tn)
        / max(len(y_true), 1)
    )

    precision = (
        tp / max(tp + fp, 1)
    )

    recall = (
        tp / max(tp + fn, 1)
    )

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc_manual(
            y_true,
            probabilities
        ),
        "pr_auc": pr_auc_manual(
            y_true,
            probabilities
        ),
        "threshold": float(threshold),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


# ================================================================
# THRESHOLD OPTIMIZATION
# ================================================================

def find_best_threshold(
    y_true,
    probabilities
):

    best_threshold = 0.50
    best_metrics = None

    # Dense enough search for an operational threshold
    thresholds = np.arange(
        0.05,
        0.751,
        0.01
    )

    for threshold in thresholds:

        metrics = classification_metrics(
            y_true,
            probabilities,
            threshold
        )

        if (
            best_metrics is None
            or metrics["f1"]
            > best_metrics["f1"]
        ):

            best_threshold = float(
                threshold
            )

            best_metrics = metrics

    return best_threshold, best_metrics


# ================================================================
# TRAIN MODEL
# ================================================================

def train_model(
    X_train,
    y_train,
    X_validation,
    y_validation
):

    print_section("TRAINING ENHANCED XGBOOST MODEL")

    train_positive = int(
        y_train.sum()
    )

    train_negative = int(
        len(y_train)
        - train_positive
    )

    scale_pos_weight = (
        train_negative
        / max(train_positive, 1)
    )

    print(
        f"Training positives: "
        f"{train_positive:,}"
    )

    print(
        f"Training negatives: "
        f"{train_negative:,}"
    )

    print(
        f"scale_pos_weight: "
        f"{scale_pos_weight:.6f}"
    )

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=list(X_train.columns)
    )

    dvalidation = xgb.DMatrix(
        X_validation,
        label=y_validation,
        feature_names=list(X_validation.columns)
    )

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr", "logloss"],
        "tree_method": "hist",
        "max_depth": 5,
        "min_child_weight": 10,
        "eta": 0.04,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "gamma": 0.15,
        "reg_alpha": 0.10,
        "reg_lambda": 3.0,
        "scale_pos_weight": scale_pos_weight,
        "seed": SEED,
        "nthread": max(
            1,
            (os.cpu_count() or 2) - 1
        ),
    }

    start = time.perf_counter()

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1200,
        evals=[
            (dtrain, "train"),
            (dvalidation, "validation"),
        ],
        early_stopping_rounds=90,
        verbose_eval=100,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"\nTraining time: "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"Best iteration: "
        f"{model.best_iteration}"
    )

    return model, elapsed, scale_pos_weight


# ================================================================
# BASELINE
# ================================================================

def calculate_baseline(
    y_train,
    y_test
):

    probability = float(
        y_train.mean()
    )

    probabilities = np.full(
        len(y_test),
        probability,
        dtype=float
    )

    return {
        "constant_probability": probability,
        "roc_auc": roc_auc_manual(
            y_test,
            probabilities
        ),
        "pr_auc": pr_auc_manual(
            y_test,
            probabilities
        ),
    }


# ================================================================
# FEATURE IMPORTANCE
# ================================================================

def save_feature_importance(model):

    importance = model.get_score(
        importance_type="gain"
    )

    rows = []

    for feature, value in importance.items():

        rows.append(
            {
                "feature": feature,
                "importance_gain": float(value),
            }
        )

    importance_df = pd.DataFrame(
        rows
    )

    if not importance_df.empty:

        importance_df[
            "importance_percent"
        ] = (
            importance_df[
                "importance_gain"
            ]
            / importance_df[
                "importance_gain"
            ].sum()
            * 100
        )

        importance_df = (
            importance_df
            .sort_values(
                "importance_gain",
                ascending=False
            )
            .reset_index(drop=True)
        )

    importance_df.to_csv(
        IMPORTANCE_FILE,
        index=False
    )

    # Plot top 20
    if not importance_df.empty:

        top = importance_df.head(20)

        plt.figure(
            figsize=(10, 8)
        )

        plt.barh(
            top["feature"][::-1],
            top["importance_percent"][::-1]
        )

        plt.xlabel(
            "Importance (%)"
        )

        plt.ylabel(
            "Feature"
        )

        plt.title(
            "Enhanced No-Show Model — Top Features"
        )

        plt.tight_layout()

        plt.savefig(
            FEATURE_IMPORTANCE_PNG,
            dpi=150
        )

        plt.close()

    return importance_df


# ================================================================
# CONFUSION MATRIX
# ================================================================

def save_confusion_matrix(metrics):

    cm = np.array(
        [
            [
                metrics["tn"],
                metrics["fp"],
            ],
            [
                metrics["fn"],
                metrics["tp"],
            ],
        ]
    )

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm
    )

    plt.title(
        "Enhanced No-Show Model — Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    plt.xticks(
        [0, 1],
        ["Show", "No-show"]
    )

    plt.yticks(
        [0, 1],
        ["Show", "No-show"]
    )

    for i in range(2):
        for j in range(2):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PNG,
        dpi=150
    )

    plt.close()


# ================================================================
# PROBABILITY DISTRIBUTION
# ================================================================

def save_probability_distribution(
    y_test,
    probabilities
):

    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        probabilities[
            np.asarray(y_test) == 0
        ],
        bins=30,
        alpha=0.65,
        label="Show"
    )

    plt.hist(
        probabilities[
            np.asarray(y_test) == 1
        ],
        bins=30,
        alpha=0.65,
        label="No-show"
    )

    plt.xlabel(
        "Predicted No-Show Probability"
    )

    plt.ylabel(
        "Number of Appointments"
    )

    plt.title(
        "Predicted No-Show Probability Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PROBABILITY_PNG,
        dpi=150
    )

    plt.close()


# ================================================================
# DATA QUALITY
# ================================================================

def save_data_quality(
    df,
    X
):

    rows = []

    for column in df.columns:

        rows.append(
            {
                "dataset": "appointments",
                "column": column,
                "dtype": str(
                    df[column].dtype
                ),
                "rows": len(df),
                "missing": int(
                    df[column].isna().sum()
                ),
                "unique_values": int(
                    df[column].nunique(
                        dropna=True
                    )
                ),
            }
        )

    for column in X.columns:

        rows.append(
            {
                "dataset": "model_features",
                "column": column,
                "dtype": str(
                    X[column].dtype
                ),
                "rows": len(X),
                "missing": int(
                    X[column].isna().sum()
                ),
                "unique_values": int(
                    X[column].nunique(
                        dropna=True
                    )
                ),
            }
        )

    pd.DataFrame(
        rows
    ).to_csv(
        DATA_QUALITY_FILE,
        index=False
    )


# ================================================================
# SAVE PREDICTIONS
# ================================================================

def save_predictions(
    df,
    y_test,
    probabilities,
    threshold,
    test_idx,
    extra_columns=None
):

    result = df.iloc[
        test_idx
    ].copy()

    result = result[
        [
            "appointment_id",
            "patient_id",
            "appointment_datetime",
            "department",
            "priority",
            "patient_age",
            "patient_severity",
            "duration",
        ]
    ].copy()

    result[
        "actual_no_show"
    ] = np.asarray(y_test).astype(int)

    result[
        "predicted_no_show_probability"
    ] = probabilities

    result[
        "predicted_no_show"
    ] = (
        probabilities >= threshold
    ).astype(int)

    result[
        "prediction_confidence"
    ] = np.maximum(
        probabilities,
        1 - probabilities
    )

    for key, value in (extra_columns or {}).items():
        result[key] = value

    result.to_csv(
        PREDICTIONS_FILE,
        index=False
    )


# ================================================================
# TRAINING SUMMARY
# ================================================================

def save_training_summary(
    df,
    X,
    train_idx,
    validation_idx,
    test_idx,
    threshold,
    model,
    elapsed,
):

    rows = [
        {
            "metric": "total_rows",
            "value": len(df),
        },
        {
            "metric": "feature_count",
            "value": X.shape[1],
        },
        {
            "metric": "train_rows",
            "value": len(train_idx),
        },
        {
            "metric": "validation_rows",
            "value": len(validation_idx),
        },
        {
            "metric": "test_rows",
            "value": len(test_idx),
        },
        {
            "metric": "best_iteration",
            "value": int(
                model.best_iteration
            ),
        },
        {
            "metric": "selected_threshold",
            "value": threshold,
        },
        {
            "metric": "training_seconds",
            "value": elapsed,
        },
    ]

    pd.DataFrame(
        rows
    ).to_csv(
        TRAINING_SUMMARY_FILE,
        index=False
    )


# ================================================================
# REPORT
# ================================================================

def save_report(
    df,
    X,
    baseline,
    validation_metrics,
    test_metrics,
    elapsed,
    scale_pos_weight,
    importance_df,
):

    top_features = []

    if not importance_df.empty:

        top_features = (
            importance_df.head(15)
            .to_dict(
                orient="records"
            )
        )

    lines = []

    lines.append(
        "CAREOPS-AI — ENHANCED NO-SHOW MODEL REPORT"
    )

    lines.append("=" * 70)

    lines.append(
        f"Source: {SOURCE_FILE}"
    )

    lines.append(
        f"Rows: {len(df):,}"
    )

    lines.append(
        f"Features: {X.shape[1]}"
    )

    lines.append("")

    lines.append(
        "TARGET"
    )

    lines.append("-" * 70)

    lines.append(
        f"No-show rate: "
        f"{df['no_show'].mean():.4%}"
    )

    lines.append("")

    lines.append(
        "BASELINE"
    )

    lines.append("-" * 70)

    lines.append(
        f"Baseline ROC-AUC: "
        f"{baseline['roc_auc']:.4f}"
    )

    lines.append(
        f"Baseline PR-AUC: "
        f"{baseline['pr_auc']:.4f}"
    )

    lines.append("")

    lines.append(
        "VALIDATION"
    )

    lines.append("-" * 70)

    for key, value in validation_metrics.items():

        lines.append(
            f"{key}: {value}"
        )

    lines.append("")

    lines.append(
        "TEST"
    )

    lines.append("-" * 70)

    for key, value in test_metrics.items():

        lines.append(
            f"{key}: {value}"
        )

    lines.append("")

    lines.append(
        "MODEL CONFIGURATION"
    )

    lines.append("-" * 70)

    lines.append(
        "Algorithm: XGBoost binary classification"
    )

    lines.append(
        "Chronological split: 65% / 15% / 20%"
    )

    lines.append(
        "Historical features: leakage-safe"
    )

    lines.append(
        f"scale_pos_weight: {scale_pos_weight:.6f}"
    )

    lines.append(
        f"Training time: {elapsed:.2f} seconds"
    )

    lines.append("")

    lines.append(
        "TOP FEATURES"
    )

    lines.append("-" * 70)

    for item in top_features:

        lines.append(
            f"{item['feature']}: "
            f"{item['importance_percent']:.4f}%"
        )

    lines.append("")

    lines.append(
        "INTERPRETATION"
    )

    lines.append("-" * 70)

    if test_metrics["roc_auc"] >= 0.60:

        lines.append(
            "The enhanced model shows meaningful "
            "predictive discrimination compared with "
            "the previous baseline."
        )

    elif test_metrics["roc_auc"] > 0.53:

        lines.append(
            "The enhanced model shows modest signal "
            "above random ranking, but predictive "
            "strength remains limited."
        )

    else:

        lines.append(
            "The enhanced model still shows weak "
            "predictive discrimination. The synthetic "
            "appointment data appears to contain limited "
            "predictive signal."
        )

    lines.append("")

    lines.append(
        "SAFETY / LEAKAGE NOTE"
    )

    lines.append("-" * 70)

    lines.append(
        "Historical behavioral features were calculated "
        "sequentially. The current appointment outcome "
        "was added to historical state only AFTER the "
        "current row's features were generated."
    )

    lines.append("")

    lines.append(
        "This model is intended for synthetic portfolio "
        "demonstration and operational research, not "
        "clinical decision-making."
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ================================================================
# SAVE METADATA
# ================================================================

def save_metadata(
    df,
    X,
    train_idx,
    validation_idx,
    test_idx,
    threshold,
    elapsed,
    scale_pos_weight,
    calibration_scale,
    calibration_offset,
):

    metadata = {
        "project": "CareOps-AI",
        "module": "No-Show Prediction",
        "model_version": "Enhanced Behavioral V2",
        "source_file": str(SOURCE_FILE),
        "source_rows": int(len(df)),
        "feature_count": int(X.shape[1]),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "test_rows": int(len(test_idx)),
        "selected_threshold": float(threshold),
        "training_seconds": float(elapsed),
        "scale_pos_weight": float(
            scale_pos_weight
        ),
        "probability_calibration": {
            "method": "Platt scaling fitted on validation predictions only",
            "scale": float(calibration_scale),
            "offset": float(calibration_offset),
        },
        "chronological_split": {
            "train": 0.65,
            "validation": 0.15,
            "test": 0.20,
        },
        "historical_features_are_leakage_safe": True,
        "prediction_time": (
            "Day before the appointment: history features use outcomes "
            "from earlier days only, and the smoothing prior is the running "
            "no-show rate up to that day (tests/test_leakage_audit.py)."
        ),
        "source_data_regenerated": False,
        "source_data_modified": False,
        "random_seed": SEED,
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )


# ================================================================
# SAVE METRICS
# ================================================================

def save_metrics(
    baseline,
    validation_metrics,
    test_metrics,
):

    metrics = {
        "baseline": baseline,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    METRICS_FILE.write_text(
        json.dumps(
            metrics,
            indent=2
        ),
        encoding="utf-8"
    )


# ================================================================
# MAIN
# ================================================================

def main():

    overall_start = time.perf_counter()

    print_section(
        "CAREOPS-AI — ENHANCED NO-SHOW PREDICTION"
    )

    print(
        "This program uses the EXISTING appointment dataset."
    )

    print(
        "It does NOT regenerate or modify synthetic data."
    )

    print(
        "Historical features are generated sequentially "
        "to prevent future leakage."
    )

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    df = load_data()

    # ------------------------------------------------------------
    # Normalize
    # ------------------------------------------------------------

    df = normalize_data(
        df
    )

    # ------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------

    validate_target(
        df
    )

    # ------------------------------------------------------------
    # Target distribution
    # ------------------------------------------------------------

    print_section(
        "TARGET DISTRIBUTION"
    )

    shows = int(
        (df["no_show"] == 0).sum()
    )

    no_shows = int(
        (df["no_show"] == 1).sum()
    )

    print(
        f"Total appointments: {len(df):,}"
    )

    print(
        f"Shows:              {shows:,}"
    )

    print(
        f"No-shows:           {no_shows:,}"
    )

    print(
        f"No-show rate:       "
        f"{df['no_show'].mean():.4%}"
    )

    # ------------------------------------------------------------
    # Historical features
    # ------------------------------------------------------------

    historical = create_historical_features(
        df
    )

    # ------------------------------------------------------------
    # Model dataset
    # ------------------------------------------------------------

    X, y = build_model_dataset(
        df,
        historical
    )

    # Appointments during the simulator warm-up are excluded from
    # training and evaluation; the history features above still use them.
    keep = (
        df["appointment_datetime"].dt.normalize()
        >= pd.Timestamp(WARMUP_END)
    ).to_numpy()
    X = X.loc[keep].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True)
    df = df.loc[keep].reset_index(drop=True)

    print_section(
        "MODEL DATASET"
    )

    print(
        f"Rows:     {X.shape[0]:,}"
    )

    print(
        f"Features: {X.shape[1]:,}"
    )

    # ------------------------------------------------------------
    # Split
    # ------------------------------------------------------------

    (
        train_idx,
        validation_idx,
        test_idx,
    ) = chronological_split(
        X,
        y,
        df
    )

    X_train = X.iloc[
        train_idx
    ]

    X_validation = X.iloc[
        validation_idx
    ]

    X_test = X.iloc[
        test_idx
    ]

    y_train = y.iloc[
        train_idx
    ]

    y_validation = y.iloc[
        validation_idx
    ]

    y_test = y.iloc[
        test_idx
    ]

    # ------------------------------------------------------------
    # Baseline
    # ------------------------------------------------------------

    baseline = calculate_baseline(
        y_train,
        y_test
    )

    print_section(
        "BASELINE"
    )

    print(
        f"Constant probability: "
        f"{baseline['constant_probability']:.6f}"
    )

    print(
        f"Baseline ROC-AUC: "
        f"{baseline['roc_auc']:.4f}"
    )

    print(
        f"Baseline PR-AUC: "
        f"{baseline['pr_auc']:.4f}"
    )

    # ------------------------------------------------------------
    # Train
    # ------------------------------------------------------------

    (
        model,
        training_seconds,
        scale_pos_weight,
    ) = train_model(
        X_train,
        y_train,
        X_validation,
        y_validation
    )

    # ------------------------------------------------------------
    # Validation prediction
    # ------------------------------------------------------------

    dvalidation = xgb.DMatrix(
        X_validation,
        feature_names=list(X_validation.columns)
    )

    validation_probabilities = model.predict(
        dvalidation
    )
    calibration_scale, calibration_offset = fit_probability_calibrator(
        validation_probabilities,
        y_validation,
    )
    validation_probabilities = apply_probability_calibrator(
        validation_probabilities,
        calibration_scale,
        calibration_offset,
    )

    # ------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------

    threshold, validation_metrics = (
        find_best_threshold(
            y_validation,
            validation_probabilities
        )
    )

    print_section(
        "VALIDATION RESULTS"
    )

    print(
        f"Selected threshold: "
        f"{threshold:.2f}"
    )

    print(
        f"ROC-AUC:  "
        f"{validation_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:   "
        f"{validation_metrics['pr_auc']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{validation_metrics['precision']:.4f}"
    )

    print(
        f"Recall:   "
        f"{validation_metrics['recall']:.4f}"
    )

    print(
        f"F1:       "
        f"{validation_metrics['f1']:.4f}"
    )

    # ------------------------------------------------------------
    # Test
    # ------------------------------------------------------------

    dtest = xgb.DMatrix(
        X_test,
        feature_names=list(X_test.columns)
    )

    test_probabilities = model.predict(
        dtest
    )
    test_probabilities = apply_probability_calibrator(
        test_probabilities,
        calibration_scale,
        calibration_offset,
    )

    test_metrics = classification_metrics(
        y_test,
        test_probabilities,
        threshold
    )

    print_section(
        "FINAL TEST RESULTS"
    )

    print(
        f"ROC-AUC:  "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:   "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Accuracy: "
        f"{test_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_metrics['precision']:.4f}"
    )

    print(
        f"Recall:   "
        f"{test_metrics['recall']:.4f}"
    )

    print(
        f"F1:       "
        f"{test_metrics['f1']:.4f}"
    )

    print(
        f"\nConfusion matrix:"
    )

    print(
        f"TN={test_metrics['tn']} "
        f"FP={test_metrics['fp']} "
        f"FN={test_metrics['fn']} "
        f"TP={test_metrics['tp']}"
    )

    # ------------------------------------------------------------
    # Baselines and rolling 30-day windows over the holdout
    # ------------------------------------------------------------

    logistic_validation = logistic_regression_baseline(
        X_train, y_train, X_validation
    )
    logistic_threshold, _ = find_best_threshold(
        y_validation, logistic_validation
    )
    candidates = {
        "xgboost_calibrated": (np.asarray(test_probabilities), float(threshold)),
        "logistic_regression": (
            logistic_regression_baseline(X_train, y_train, X_test),
            float(logistic_threshold),
        ),
        "prevalence": (prevalence_baseline(y_train, len(y_test)), float(threshold)),
    }
    test_dates = df.iloc[test_idx]["appointment_datetime"]
    no_show_windows, no_show_summary, no_show_champion = evaluate_classifiers(
        test_dates,
        y_test,
        candidates,
        pd.Timestamp(test_dates.min()).normalize() - pd.Timedelta(days=1),
    )
    no_show_windows.to_csv(OUTPUT_DIR / "no_show_backtest_windows.csv", index=False)
    no_show_summary.to_csv(OUTPUT_DIR / "no_show_backtest_summary.csv", index=False)
    pd.concat([
        reliability_table(y_test, probabilities).assign(model=name)
        for name, (probabilities, _) in candidates.items()
    ]).to_csv(OUTPUT_DIR / "no_show_reliability.csv", index=False)
    no_show_status = model_status(
        no_show_summary, no_show_champion, ("xgboost_calibrated",),
        score_column="brier",
    )
    print_section("ROLLING 30-DAY WINDOWS VS BASELINES")
    print(no_show_summary[
        ["model", "brier", "log_loss", "roc_auc", "pr_auc", "precision", "recall", "f1", "ece"]
    ].round(4).to_string(index=False))
    print(f"Published no-show model (lowest Brier): {no_show_champion}")

    # ------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------

    model.save_model(
        str(MODEL_FILE)
    )

    # ------------------------------------------------------------
    # Save predictions (champion)
    # ------------------------------------------------------------

    published_probabilities, published_threshold = candidates[no_show_champion]
    save_predictions(
        df,
        y_test,
        published_probabilities,
        published_threshold,
        test_idx,
        no_show_status
    )

    # ------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------

    importance_df = save_feature_importance(
        model
    )

    # ------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------

    save_confusion_matrix(
        test_metrics
    )

    save_probability_distribution(
        y_test,
        test_probabilities
    )

    # ------------------------------------------------------------
    # Data quality
    # ------------------------------------------------------------

    save_data_quality(
        df,
        X
    )

    # ------------------------------------------------------------
    # Training summary
    # ------------------------------------------------------------

    save_training_summary(
        df,
        X,
        train_idx,
        validation_idx,
        test_idx,
        threshold,
        model,
        training_seconds,
    )

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    save_metrics(
        baseline,
        validation_metrics,
        test_metrics,
    )

    # ------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------

    save_metadata(
        df,
        X,
        train_idx,
        validation_idx,
        test_idx,
        threshold,
        training_seconds,
        scale_pos_weight,
        calibration_scale,
        calibration_offset,
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    save_report(
        df,
        X,
        baseline,
        validation_metrics,
        test_metrics,
        training_seconds,
        scale_pos_weight,
        importance_df,
    )

    # ------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------

    print_section(
        "OUTPUT VERIFICATION"
    )

    expected_files = [
        MODEL_FILE,
        PREDICTIONS_FILE,
        METRICS_FILE,
        METADATA_FILE,
        IMPORTANCE_FILE,
        DATA_QUALITY_FILE,
        TRAINING_SUMMARY_FILE,
        CONFUSION_MATRIX_PNG,
        FEATURE_IMPORTANCE_PNG,
        PROBABILITY_PNG,
        REPORT_FILE,
    ]

    verified = 0

    for path in expected_files:

        if path.exists() and path.stat().st_size > 0:

            print(
                f"[OK] {path.name}"
            )

            verified += 1

        else:

            print(
                f"[MISSING] {path.name}"
            )

    total_runtime = (
        time.perf_counter()
        - overall_start
    )

    print(
        f"\nVerified outputs: "
        f"{verified}/{len(expected_files)}"
    )

    print(
        f"\nOutput directory:"
    )

    print(
        OUTPUT_DIR
    )

    print(
        f"\nTotal runtime: "
        f"{total_runtime:.2f} seconds"
    )

    # ------------------------------------------------------------
    # Comparison guidance
    # ------------------------------------------------------------

    print_section(
        "COMPARISON WITH PREVIOUS MODEL"
    )

    previous_auc = 0.5224

    improvement = (
        test_metrics["roc_auc"]
        - previous_auc
    )

    print(
        f"Previous model ROC-AUC: "
        f"{previous_auc:.4f}"
    )

    print(
        f"Enhanced model ROC-AUC: "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"Change: "
        f"{improvement:+.4f}"
    )

    if test_metrics["roc_auc"] > previous_auc:

        print(
            "\nThe enhanced model improved "
            "ROC-AUC over the previous model."
        )

    elif test_metrics["roc_auc"] == previous_auc:

        print(
            "\nThe enhanced model matched "
            "the previous model."
        )

    else:

        print(
            "\nThe enhanced model did not improve "
            "ROC-AUC over the previous model."
        )

    print_section(
        "NO-SHOW MODEL COMPLETE"
    )

    print(
        "The source appointment dataset was NOT "
        "regenerated."
    )

    print(
        "The source appointment dataset was NOT modified."
    )

    print(
        "All historical behavioral features were "
        "generated without using future outcomes."
    )

    print(
        "\nRead:"
    )

    print(
        REPORT_FILE
    )


if __name__ == "__main__":
    main()
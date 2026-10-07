"""
CAREOPS-AI
ICU DEMAND FORECASTING MODULE

IMPORTANT ENVIRONMENT REQUIREMENTS
-----------------------------------
This program intentionally DOES NOT use:
    - scikit-learn
    - SciPy
    - Matplotlib

It uses:
    - pandas
    - numpy
    - xgboost native API
    - joblib
    - Python standard library

The program:
1. Loads existing daily_operations.csv
2. Builds a daily ICU demand series
3. Creates leakage-safe daily admission forecasting features
4. Performs chronological train/validation/test split
5. Trains native XGBoost using xgb.train()
6. Evaluates against a 7-day naive baseline
7. Calculates feature importance
8. Forecasts future exogenous variables
9. Recursively forecasts future ICU demand
10. Labels capacity risk as unavailable when ICU census data is absent
11. Creates weekly forecasts
12. Saves model, predictions, metrics, metadata and reports
13. Creates lightweight PNG visualizations without Matplotlib
"""

from __future__ import annotations

import json
import math
import os
import struct
import time
import traceback
import zlib
from pathlib import Path
from multiprocessing import cpu_count

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb


# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

SEED = 42

FORECAST_HORIZON_DAYS = 30

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": ["mae", "rmse"],
    "max_depth": 7,
    "learning_rate": 0.04,
    "subsample": 0.90,
    "colsample_bytree": 0.90,
    "min_child_weight": 4,
    "reg_alpha": 0.05,
    "reg_lambda": 2.0,
    "tree_method": "hist",
    "seed": SEED,
    "nthread": max(1, cpu_count() - 1),
}

NUM_BOOST_ROUNDS = 700
EARLY_STOPPING_ROUNDS = 60


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[2]

DATA_DIR = PROJECT_ROOT / "data"
SCENARIO_DATA_DIR = os.environ.get("CAREOPS_DATA_DIR")
PROCESSED_DIR = (
    Path(SCENARIO_DATA_DIR)
    if SCENARIO_DATA_DIR
    else DATA_DIR / "processed"
)
SYNTHETIC_DIR = DATA_DIR / "synthetic"

OUTPUT_DIR = PROJECT_ROOT / "ml" / "icu_demand_forecasting"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CONSOLE HELPERS
# ============================================================================

def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def info(message: str) -> None:
    print(message)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_daily_operations() -> Path:
    candidates = [PROCESSED_DIR / "daily_operations.csv"]
    if not SCENARIO_DATA_DIR:
        candidates.extend([
            DATA_DIR / "daily_operations.csv",
            SYNTHETIC_DIR / "daily_operations.csv",
        ])

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find daily_operations.csv in:\n"
        + "\n".join(str(p) for p in candidates)
    )


# ============================================================================
# DATA LOADING
# ============================================================================

def load_source_data() -> tuple[pd.DataFrame, Path]:
    banner("LOADING ICU SOURCE DATA")

    source_path = find_daily_operations()

    info(f"Python executable: {os.sys.executable}")
    info(f"Project root: {PROJECT_ROOT}")
    info(f"Source file: {source_path}")

    df = pd.read_csv(source_path)

    info(f"Raw shape: {df.shape}")
    info(f"Columns: {list(df.columns)}")

    required = [
        "date",
        "department",
        "admissions",
        "discharges",
        "emergency_cases",
        "ICU_admissions",
        "occupancy",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"daily_operations.csv is missing required columns: {missing}"
        )

    invalid_dates_before = df["date"].isna().sum()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    invalid_dates = int(df["date"].isna().sum())

    df["ICU_admissions"] = pd.to_numeric(
        df["ICU_admissions"], errors="coerce"
    )

    invalid_icu = int(df["ICU_admissions"].isna().sum())

    info(f"Invalid dates: {invalid_dates}")
    info(f"Invalid ICU admissions: {invalid_icu}")

    if invalid_dates > 0:
        df = df.dropna(subset=["date"])

    if invalid_icu > 0:
        df["ICU_admissions"] = df["ICU_admissions"].fillna(0)

    numeric_columns = [
        "admissions",
        "discharges",
        "emergency_cases",
        "ICU_admissions",
        "occupancy",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column], errors="coerce"
        ).fillna(0.0)

    df["department"] = (
        df["department"]
        .astype(str)
        .str.strip()
    )

    info(
        f"Date range: "
        f"{df['date'].min().date()} -> {df['date'].max().date()}"
    )

    departments = sorted(df["department"].dropna().unique().tolist())

    info("Departments: " + ", ".join(departments))

    return df, source_path


# ============================================================================
# DAILY ICU SERIES
# ============================================================================

def build_daily_series(
    raw_df: pd.DataFrame,
) -> pd.DataFrame:

    banner("BUILDING DAILY ICU DEMAND SERIES")

    daily = (
        raw_df.groupby("date", as_index=False)
        .agg(
            {
                "admissions": "sum",
                "discharges": "sum",
                "emergency_cases": "sum",
                "ICU_admissions": "sum",
                "occupancy": "mean",
            }
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Make the calendar continuous.
    full_dates = pd.date_range(
        daily["date"].min(),
        daily["date"].max(),
        freq="D",
    )

    daily = (
        daily.set_index("date")
        .reindex(full_dates)
        .rename_axis("date")
        .reset_index()
    )

    numeric = [
        "admissions",
        "discharges",
        "emergency_cases",
        "ICU_admissions",
        "occupancy",
    ]

    for column in numeric:
        daily[column] = (
            pd.to_numeric(daily[column], errors="coerce")
            .interpolate(limit_direction="both")
            .fillna(0.0)
        )

    daily["ICU_admissions"] = daily["ICU_admissions"].round()

    info(
        f"Calendar days before feature warm-up: {len(daily)}"
    )

    return daily


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

FEATURE_COLUMNS = [
    "admissions",
    "discharges",
    "emergency_cases",
    "occupancy",
    "day_of_week",
    "day_of_month",
    "week_of_year",
    "month",
    "quarter",
    "year",
    "day_of_year",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "icu_lag_1",
    "icu_lag_2",
    "icu_lag_3",
    "icu_lag_7",
    "icu_lag_14",
    "icu_lag_21",
    "icu_lag_28",
    "icu_roll_mean_3",
    "icu_roll_std_3",
    "icu_roll_min_3",
    "icu_roll_max_3",
    "icu_roll_mean_7",
    "icu_roll_std_7",
    "icu_roll_min_7",
    "icu_roll_max_7",
    "icu_roll_mean_14",
    "icu_roll_std_14",
    "icu_roll_min_14",
    "icu_roll_max_14",
    "icu_roll_mean_28",
    "icu_roll_std_28",
    "icu_roll_min_28",
    "icu_roll_max_28",
    "admissions_lag_1",
    "admissions_lag_7",
    "admissions_roll_mean_7",
    "discharges_lag_1",
    "discharges_lag_7",
    "discharges_roll_mean_7",
    "emergency_cases_lag_1",
    "emergency_cases_lag_7",
    "emergency_cases_roll_mean_7",
    "occupancy_lag_1",
    "occupancy_lag_7",
    "occupancy_roll_mean_7",
    "icu_change_1",
    "icu_change_7",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:

    data = df.copy()
    data = data.sort_values("date").reset_index(drop=True)

    date = pd.to_datetime(data["date"])

    data["day_of_week"] = date.dt.dayofweek
    data["day_of_month"] = date.dt.day
    data["week_of_year"] = date.dt.isocalendar().week.astype(int)
    data["month"] = date.dt.month
    data["quarter"] = date.dt.quarter
    data["year"] = date.dt.year
    data["day_of_year"] = date.dt.dayofyear

    data["dow_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["dow_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["day_of_year_sin"] = np.sin(
        2 * np.pi * data["day_of_year"] / 365.25
    )

    data["day_of_year_cos"] = np.cos(
        2 * np.pi * data["day_of_year"] / 365.25
    )

    # ------------------------------------------------------------------
    # ICU LAGS
    # ------------------------------------------------------------------

    for lag in [1, 2, 3, 7, 14, 21, 28]:
        data[f"icu_lag_{lag}"] = (
            data["ICU_admissions"].shift(lag)
        )

    # ------------------------------------------------------------------
    # ICU ROLLING FEATURES
    # IMPORTANT: shift(1) prevents target leakage.
    # ------------------------------------------------------------------

    shifted_icu = data["ICU_admissions"].shift(1)

    for window in [3, 7, 14, 28]:

        rolling = shifted_icu.rolling(
            window=window,
            min_periods=window,
        )

        data[f"icu_roll_mean_{window}"] = rolling.mean()
        data[f"icu_roll_std_{window}"] = rolling.std()
        data[f"icu_roll_min_{window}"] = rolling.min()
        data[f"icu_roll_max_{window}"] = rolling.max()

    # ------------------------------------------------------------------
    # ADMISSIONS
    # ------------------------------------------------------------------

    data["admissions_lag_1"] = (
        data["admissions"].shift(1)
    )

    data["admissions_lag_7"] = (
        data["admissions"].shift(7)
    )

    data["admissions_roll_mean_7"] = (
        data["admissions"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    # ------------------------------------------------------------------
    # DISCHARGES
    # ------------------------------------------------------------------

    data["discharges_lag_1"] = (
        data["discharges"].shift(1)
    )

    data["discharges_lag_7"] = (
        data["discharges"].shift(7)
    )

    data["discharges_roll_mean_7"] = (
        data["discharges"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    # ------------------------------------------------------------------
    # EMERGENCY CASES
    # ------------------------------------------------------------------

    data["emergency_cases_lag_1"] = (
        data["emergency_cases"].shift(1)
    )

    data["emergency_cases_lag_7"] = (
        data["emergency_cases"].shift(7)
    )

    data["emergency_cases_roll_mean_7"] = (
        data["emergency_cases"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    # ------------------------------------------------------------------
    # OCCUPANCY
    # ------------------------------------------------------------------

    data["occupancy_lag_1"] = (
        data["occupancy"].shift(1)
    )

    data["occupancy_lag_7"] = (
        data["occupancy"].shift(7)
    )

    data["occupancy_roll_mean_7"] = (
        data["occupancy"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    # ------------------------------------------------------------------
    # ICU CHANGE
    # ------------------------------------------------------------------

    data["icu_change_1"] = (
        data["ICU_admissions"]
        .shift(1)
        - data["ICU_admissions"].shift(2)
    )

    data["icu_change_7"] = (
        data["ICU_admissions"]
        .shift(1)
        - data["ICU_admissions"].shift(8)
    )

    # ------------------------------------------------------------------
    # TARGET
    # ------------------------------------------------------------------

    data["target"] = data["ICU_admissions"]

    # Keep only rows with all required model features available.
    before = len(data)

    data = data.dropna(
        subset=FEATURE_COLUMNS + ["target"]
    ).reset_index(drop=True)

    after = len(data)

    info(
        f"Modeling rows after feature warm-up: {after}"
    )

    info(
        f"Mean ICU demand: "
        f"{data['target'].mean():.3f}"
    )

    info(
        f"Minimum ICU demand: "
        f"{int(data['target'].min())}"
    )

    info(
        f"Maximum ICU demand: "
        f"{int(data['target'].max())}"
    )

    if before == after:
        info("No rows removed after feature construction.")

    return data


# ============================================================================
# CHRONOLOGICAL SPLIT
# ============================================================================

def chronological_split(
    data: pd.DataFrame,
):
    n = len(data)

    train_end = int(n * TRAIN_RATIO)

    validation_end = int(
        n * (TRAIN_RATIO + VALIDATION_RATIO)
    )

    train = data.iloc[:train_end].copy()
    validation = data.iloc[
        train_end:validation_end
    ].copy()

    test = data.iloc[
        validation_end:
    ].copy()

    print()
    print("Chronological split:")

    print(
        f"Train:      {len(train)} rows | "
        f"{train['date'].min().date()} -> "
        f"{train['date'].max().date()}"
    )

    print(
        f"Validation: {len(validation)} rows | "
        f"{validation['date'].min().date()} -> "
        f"{validation['date'].max().date()}"
    )

    print(
        f"Test:       {len(test)} rows | "
        f"{test['date'].min().date()} -> "
        f"{test['date'].max().date()}"
    )

    return train, validation, test


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    actual,
    predicted,
) -> dict:

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    error = actual - predicted

    mae = np.mean(np.abs(error))

    rmse = np.sqrt(
        np.mean(error ** 2)
    )

    ss_res = np.sum(error ** 2)

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    if ss_tot == 0:
        r2 = 0.0
    else:
        r2 = 1.0 - (ss_res / ss_tot)

    nonzero = np.abs(actual) > 1e-9

    if np.any(nonzero):
        mape = (
            np.mean(
                np.abs(
                    error[nonzero]
                    / actual[nonzero]
                )
            )
            * 100
        )
    else:
        mape = 0.0

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE_percent": float(mape),
    }


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
):

    banner("TRAINING NATIVE XGBOOST ICU DEMAND MODEL")

    X_train = train[FEATURE_COLUMNS].astype(float)
    y_train = train["target"].astype(float)

    X_validation = validation[
        FEATURE_COLUMNS
    ].astype(float)

    y_validation = validation[
        "target"
    ].astype(float)

    print(f"Training rows: {len(train)}")
    print(f"Validation rows: {len(validation)}")
    print(f"Features: {len(FEATURE_COLUMNS)}")

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=FEATURE_COLUMNS,
    )

    dvalidation = xgb.DMatrix(
        X_validation,
        label=y_validation,
        feature_names=FEATURE_COLUMNS,
    )

    start = time.perf_counter()

    model = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[
            (dtrain, "train"),
            (dvalidation, "validation"),
        ],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )

    elapsed = time.perf_counter() - start

    print(
        f"XGBoost training completed in "
        f"{elapsed:.2f} seconds."
    )

    return model


# ============================================================================
# MODEL PREDICTION
# ============================================================================

def predict_model(
    model,
    df: pd.DataFrame,
) -> np.ndarray:

    matrix = xgb.DMatrix(
        df[FEATURE_COLUMNS].astype(float),
        feature_names=FEATURE_COLUMNS,
    )

    return model.predict(matrix)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(
    model,
    train: pd.DataFrame,
    test: pd.DataFrame,
):

    banner("EVALUATING ICU DEMAND MODEL")

    actual = test["target"].to_numpy(dtype=float)

    predictions = predict_model(
        model,
        test,
    )

    predictions = np.maximum(
        predictions,
        0.0,
    )

    model_metrics = calculate_metrics(
        actual,
        predictions,
    )

    # ------------------------------------------------------------
    # 7-DAY NAIVE BASELINE
    # ------------------------------------------------------------

    baseline_predictions = (
        test["ICU_admissions"]
        .shift(7)
    )

    # Because test rows retain chronological indexing,
    # explicitly calculate from the complete historical series.
    combined_target = pd.concat(
        [
            train["target"],
            test["target"],
        ],
        ignore_index=True,
    )

    test_start = len(train)

    baseline_values = []

    for i in range(len(test)):
        global_index = test_start + i
        lag_index = global_index - 7

        if lag_index >= 0:
            baseline_values.append(
                combined_target.iloc[lag_index]
            )
        else:
            baseline_values.append(
                combined_target.iloc[0]
            )

    baseline_predictions = np.asarray(
        baseline_values,
        dtype=float,
    )

    baseline_metrics = calculate_metrics(
        actual,
        baseline_predictions,
    )

    print()
    print("XGBoost metrics:")
    print(
        f"MAE:  {model_metrics['MAE']:.4f}"
    )
    print(
        f"RMSE: {model_metrics['RMSE']:.4f}"
    )
    print(
        f"R2:   {model_metrics['R2']:.4f}"
    )
    print(
        f"MAPE: {model_metrics['MAPE_percent']:.2f}%"
    )

    print()
    print("7-day baseline metrics:")
    print(
        f"MAE:  {baseline_metrics['MAE']:.4f}"
    )
    print(
        f"RMSE: {baseline_metrics['RMSE']:.4f}"
    )
    print(
        f"R2:   {baseline_metrics['R2']:.4f}"
    )
    print(
        f"MAPE: {baseline_metrics['MAPE_percent']:.2f}%"
    )

    prediction_df = pd.DataFrame(
        {
            "date": test["date"].values,
            "actual_icu_demand": actual,
            "predicted_icu_demand": predictions,
            "prediction_error": (
                actual - predictions
            ),
            "absolute_error": np.abs(
                actual - predictions
            ),
            "baseline_7_day": baseline_predictions,
        }
    )

    return (
        model_metrics,
        baseline_metrics,
        prediction_df,
    )


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def calculate_feature_importance(
    model,
) -> pd.DataFrame:

    banner("CALCULATING FEATURE IMPORTANCE")

    scores = model.get_score(
        importance_type="gain"
    )

    rows = []

    for feature in FEATURE_COLUMNS:
        importance = float(
            scores.get(feature, 0.0)
        )

        rows.append(
            {
                "feature": feature,
                "importance": importance,
            }
        )

    result = pd.DataFrame(rows)

    total = result["importance"].sum()

    if total > 0:
        result["importance_percent"] = (
            result["importance"]
            / total
            * 100
        )
    else:
        result["importance_percent"] = 0.0

    result = result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    print(
        result.head(15).to_string(
            index=False
        )
    )

    return result


# ============================================================================
# FUTURE EXOGENOUS FORECASTS
# ============================================================================

def build_exogenous_forecast(
    history: pd.DataFrame,
    future_dates: pd.DatetimeIndex,
) -> pd.DataFrame:

    """
    Forecast the four future exogenous inputs required by the ICU model:

        admissions
        discharges
        emergency_cases
        occupancy

    IMPORTANT:
    The previous program failed because future rows did not contain these
    four model features.

    This function creates them before feature construction.

    Method:
    - Use recent historical observations.
    - Prefer same-weekday historical averages.
    - Use the most recent 28 observations as fallback.
    """

    history = history.copy()
    history["date"] = pd.to_datetime(
        history["date"]
    )

    history = history.sort_values(
        "date"
    ).reset_index(drop=True)

    output_rows = []

    recent_window = history.tail(28)

    exogenous_columns = [
        "admissions",
        "discharges",
        "emergency_cases",
        "occupancy",
    ]

    for future_date in future_dates:

        dow = future_date.dayofweek

        same_dow = history[
            history["date"].dt.dayofweek == dow
        ].tail(12)

        row = {
            "date": future_date
        }

        for column in exogenous_columns:

            if len(same_dow) >= 3:
                value = same_dow[column].mean()

            else:
                value = recent_window[
                    column
                ].mean()

            if not np.isfinite(value):
                value = history[
                    column
                ].tail(7).mean()

            row[column] = float(
                max(0.0, value)
            )

        output_rows.append(row)

    future = pd.DataFrame(
        output_rows
    )

    return future


# ============================================================================
# RECURSIVE FUTURE FEATURE CONSTRUCTION
# ============================================================================

def create_future_forecast(
    model,
    historical_raw: pd.DataFrame,
    horizon_days: int = FORECAST_HORIZON_DAYS,
) -> pd.DataFrame:

    banner("CREATING FUTURE ICU FORECAST")

    history = historical_raw.copy()

    history["date"] = pd.to_datetime(
        history["date"]
    )

    history = history.sort_values(
        "date"
    ).reset_index(drop=True)

    last_date = history["date"].max()

    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1),
        periods=horizon_days,
        freq="D",
    )

    # ------------------------------------------------------------
    # THIS IS THE CRITICAL FIX.
    #
    # Future admissions, discharges, emergency_cases and occupancy
    # are created BEFORE model features are built.
    # ------------------------------------------------------------

    future_exogenous = build_exogenous_forecast(
        history,
        future_dates,
    )

    # Keep only the variables needed to construct features.
    working = history[
        [
            "date",
            "admissions",
            "discharges",
            "emergency_cases",
            "ICU_admissions",
            "occupancy",
        ]
    ].copy()

    forecast_rows = []

    for _, future_row in future_exogenous.iterrows():

        future_date = pd.Timestamp(
            future_row["date"]
        )

        # --------------------------------------------------------
        # CREATE FUTURE ROW WITH ALL REQUIRED RAW INPUTS
        # --------------------------------------------------------

        new_row = {
            "date": future_date,
            "admissions": float(
                future_row["admissions"]
            ),
            "discharges": float(
                future_row["discharges"]
            ),
            "emergency_cases": float(
                future_row["emergency_cases"]
            ),
            "occupancy": float(
                future_row["occupancy"]
            ),
            "ICU_admissions": np.nan,
        }

        working = pd.concat(
            [
                working,
                pd.DataFrame([new_row]),
            ],
            ignore_index=True,
        )

        # --------------------------------------------------------
        # CONSTRUCT FEATURES
        #
        # The target is NaN for the current future row, but all
        # lagged/rolling target features use previous rows.
        # --------------------------------------------------------

        temp_features = add_features_for_future_row(
            working
        )

        current_features = temp_features[
            temp_features["date"]
            == future_date
        ].copy()

        if current_features.empty:
            raise ValueError(
                f"Could not construct future features "
                f"for {future_date.date()}."
            )

        current_features = current_features.iloc[
            [-1]
        ]

        missing = [
            feature
            for feature in FEATURE_COLUMNS
            if feature not in current_features.columns
            or pd.isna(
                current_features.iloc[0][feature]
            )
        ]

        if missing:
            raise ValueError(
                f"Future row is missing features: "
                f"{missing}"
            )

        # --------------------------------------------------------
        # PREDICT ICU DEMAND
        # --------------------------------------------------------

        matrix = xgb.DMatrix(
            current_features[
                FEATURE_COLUMNS
            ].astype(float),
            feature_names=FEATURE_COLUMNS,
        )

        prediction = float(
            model.predict(matrix)[0]
        )

        # ICU demand cannot be negative.
        prediction = max(
            0.0,
            prediction,
        )

        # ICU demand represents number of admissions.
        prediction = round(
            prediction,
            2,
        )

        # Put prediction back into working history so the next
        # future day can use it as lagged/rolling information.
        working.loc[
            working["date"] == future_date,
            "ICU_admissions"
        ] = prediction

        forecast_rows.append(
            {
                "date": future_date,
                "predicted_icu_admissions": prediction,
                "predicted_icu_demand": prediction,
                "forecast_target_unit": "admissions_per_day",
                "predicted_utilization": np.nan,
                "remaining_capacity": np.nan,
                "capacity_risk": "NOT_ESTIMATED_CENSUS_UNAVAILABLE",
                "forecast_horizon_day": (
                    len(forecast_rows) + 1
                ),
                "forecast_admissions": float(
                    future_row["admissions"]
                ),
                "forecast_discharges": float(
                    future_row["discharges"]
                ),
                "forecast_emergency_cases": float(
                    future_row["emergency_cases"]
                ),
                "forecast_occupancy": float(
                    future_row["occupancy"]
                ),
            }
        )

    forecast = pd.DataFrame(
        forecast_rows
    )

    print(
        f"Future forecast created for "
        f"{len(forecast)} days."
    )

    print(
        f"Forecast period: "
        f"{forecast['date'].min().date()} -> "
        f"{forecast['date'].max().date()}"
    )

    print()
    print(
        forecast[
            [
                "date",
                "predicted_icu_admissions",
                "forecast_target_unit",
            ]
        ].to_string(index=False)
    )

    return forecast


# ============================================================================
# FUTURE FEATURE BUILDER
# ============================================================================

def add_features_for_future_row(
    df: pd.DataFrame,
) -> pd.DataFrame:

    """
    Feature engineering specifically for recursive forecasting.

    Difference from training feature engineering:
    The current future ICU_admissions value is NaN.

    All target-derived features intentionally use SHIFTED values,
    therefore they remain computable.
    """

    data = df.copy()

    data = data.sort_values(
        "date"
    ).reset_index(drop=True)

    date = pd.to_datetime(
        data["date"]
    )

    data["day_of_week"] = date.dt.dayofweek
    data["day_of_month"] = date.dt.day
    data["week_of_year"] = (
        date.dt.isocalendar()
        .week
        .astype(int)
    )
    data["month"] = date.dt.month
    data["quarter"] = date.dt.quarter
    data["year"] = date.dt.year
    data["day_of_year"] = date.dt.dayofyear

    data["dow_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["dow_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["day_of_year_sin"] = np.sin(
        2 * np.pi
        * data["day_of_year"]
        / 365.25
    )

    data["day_of_year_cos"] = np.cos(
        2 * np.pi
        * data["day_of_year"]
        / 365.25
    )

    # ICU lags
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        data[f"icu_lag_{lag}"] = (
            data["ICU_admissions"]
            .shift(lag)
        )

    shifted_icu = (
        data["ICU_admissions"]
        .shift(1)
    )

    for window in [3, 7, 14, 28]:

        rolling = shifted_icu.rolling(
            window=window,
            min_periods=window,
        )

        data[
            f"icu_roll_mean_{window}"
        ] = rolling.mean()

        data[
            f"icu_roll_std_{window}"
        ] = rolling.std()

        data[
            f"icu_roll_min_{window}"
        ] = rolling.min()

        data[
            f"icu_roll_max_{window}"
        ] = rolling.max()

    # Admissions
    data["admissions_lag_1"] = (
        data["admissions"].shift(1)
    )

    data["admissions_lag_7"] = (
        data["admissions"].shift(7)
    )

    data["admissions_roll_mean_7"] = (
        data["admissions"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    # Discharges
    data["discharges_lag_1"] = (
        data["discharges"].shift(1)
    )

    data["discharges_lag_7"] = (
        data["discharges"].shift(7)
    )

    data["discharges_roll_mean_7"] = (
        data["discharges"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    # Emergency
    data["emergency_cases_lag_1"] = (
        data["emergency_cases"].shift(1)
    )

    data["emergency_cases_lag_7"] = (
        data["emergency_cases"].shift(7)
    )

    data["emergency_cases_roll_mean_7"] = (
        data["emergency_cases"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    # Occupancy
    data["occupancy_lag_1"] = (
        data["occupancy"].shift(1)
    )

    data["occupancy_lag_7"] = (
        data["occupancy"].shift(7)
    )

    data["occupancy_roll_mean_7"] = (
        data["occupancy"]
        .shift(1)
        .rolling(7, min_periods=7)
        .mean()
    )

    # ICU changes
    data["icu_change_1"] = (
        data["ICU_admissions"].shift(1)
        - data["ICU_admissions"].shift(2)
    )

    data["icu_change_7"] = (
        data["ICU_admissions"].shift(1)
        - data["ICU_admissions"].shift(8)
    )

    return data


# ============================================================================
# WEEKLY FORECAST
# ============================================================================

def create_weekly_forecast(
    future_forecast: pd.DataFrame,
) -> pd.DataFrame:

    weekly = future_forecast.copy()

    weekly["week_start"] = (
        weekly["date"]
        - pd.to_timedelta(
            weekly["date"].dt.dayofweek,
            unit="D",
        )
    )

    result = (
        weekly.groupby(
            "week_start",
            as_index=False,
        )
        .agg(
            forecast_mean_icu_demand=(
                "predicted_icu_admissions",
                "mean",
            ),
            forecast_max_icu_demand=(
                "predicted_icu_admissions",
                "max",
            ),
            forecast_min_icu_demand=(
                "predicted_icu_admissions",
                "min",
            ),
        )
    )

    result["forecast_target_unit"] = "admissions_per_day"
    result["capacity_risk"] = "NOT_ESTIMATED_CENSUS_UNAVAILABLE"

    return result


# ============================================================================
# CAPACITY RISK
# ============================================================================

def create_capacity_risk(
    future_forecast: pd.DataFrame,
) -> pd.DataFrame:

    result = future_forecast[
        [
            "date",
            "predicted_icu_admissions",
            "forecast_target_unit",
            "capacity_risk",
        ]
    ].copy()

    return result


# ============================================================================
# DATA QUALITY
# ============================================================================

def create_data_quality_report(
    raw_df: pd.DataFrame,
    model_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for column in raw_df.columns:

        rows.append(
            {
                "dataset": "daily_operations",
                "column": column,
                "dtype": str(
                    raw_df[column].dtype
                ),
                "row_count": len(raw_df),
                "missing_count": int(
                    raw_df[column].isna().sum()
                ),
                "missing_percent": float(
                    raw_df[column].isna().mean()
                    * 100
                ),
                "unique_values": int(
                    raw_df[column].nunique(
                        dropna=True
                    )
                ),
            }
        )

    for column in model_df.columns:

        if column == "target":
            continue

        rows.append(
            {
                "dataset": "icu_modeling",
                "column": column,
                "dtype": str(
                    model_df[column].dtype
                ),
                "row_count": len(model_df),
                "missing_count": int(
                    model_df[column].isna().sum()
                ),
                "missing_percent": float(
                    model_df[column].isna().mean()
                    * 100
                ),
                "unique_values": int(
                    model_df[column].nunique(
                        dropna=True
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================================
# TRAINING SUMMARY
# ============================================================================

def create_training_summary(
    train,
    validation,
    test,
    metrics,
    baseline_metrics,
) -> pd.DataFrame:

    rows = [
        ("train_rows", len(train)),
        ("validation_rows", len(validation)),
        ("test_rows", len(test)),
        ("feature_count", len(FEATURE_COLUMNS)),
        ("train_start", str(train["date"].min().date())),
        ("train_end", str(train["date"].max().date())),
        (
            "validation_start",
            str(validation["date"].min().date()),
        ),
        (
            "validation_end",
            str(validation["date"].max().date()),
        ),
        ("test_start", str(test["date"].min().date())),
        ("test_end", str(test["date"].max().date())),
        ("model_mae", metrics["MAE"]),
        ("model_rmse", metrics["RMSE"]),
        ("model_r2", metrics["R2"]),
        ("model_mape_percent", metrics["MAPE_percent"]),
        ("baseline_mae", baseline_metrics["MAE"]),
        ("baseline_rmse", baseline_metrics["RMSE"]),
        ("baseline_r2", baseline_metrics["R2"]),
        (
            "baseline_mape_percent",
            baseline_metrics["MAPE_percent"],
        ),
        ("forecast_horizon_days", FORECAST_HORIZON_DAYS),
    ]

    return pd.DataFrame(
        rows,
        columns=["metric", "value"],
    )


# ============================================================================
# PNG UTILITIES
# ============================================================================

def save_png(
    path: Path,
    width: int,
    height: int,
    pixels: np.ndarray,
) -> None:

    pixels = np.asarray(
        pixels,
        dtype=np.uint8,
    )

    raw = bytearray()

    for y in range(height):
        raw.append(0)
        raw.extend(
            pixels[y].tobytes()
        )

    compressed = zlib.compress(
        bytes(raw),
        level=6,
    )

    def chunk(
        chunk_type: bytes,
        data: bytes,
    ) -> bytes:

        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(
                ">I",
                zlib.crc32(
                    chunk_type + data
                )
                & 0xffffffff,
            )
        )

    png = bytearray(
        b"\x89PNG\r\n\x1a\n"
    )

    png.extend(
        chunk(
            b"IHDR",
            struct.pack(
                ">IIBBBBB",
                width,
                height,
                8,
                2,
                0,
                0,
                0,
            ),
        )
    )

    png.extend(
        chunk(
            b"IDAT",
            compressed,
        )
    )

    png.extend(
        chunk(
            b"IEND",
            b"",
        )
    )

    with open(path, "wb") as file:
        file.write(png)


def draw_line_chart(
    path: Path,
    actual,
    predicted,
    title_space: int = 30,
) -> None:

    width = 1200
    height = 600

    pixels = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )

    actual = np.asarray(
        actual,
        dtype=float,
    )

    predicted = np.asarray(
        predicted,
        dtype=float,
    )

    values = np.concatenate(
        [actual, predicted]
    )

    low = float(np.nanmin(values))
    high = float(np.nanmax(values))

    if high <= low:
        high = low + 1.0

    left = 60
    right = width - 30
    top = title_space
    bottom = height - 50

    def point(
        index,
        value,
    ):
        x = left + (
            index
            / max(1, len(actual) - 1)
        ) * (right - left)

        y = bottom - (
            (value - low)
            / (high - low)
        ) * (bottom - top)

        return int(x), int(y)

    def draw_segment(
        p1,
        p2,
        rgb,
    ):
        x1, y1 = p1
        x2, y2 = p2

        steps = max(
            abs(x2 - x1),
            abs(y2 - y1),
            1,
        )

        for i in range(steps + 1):
            x = int(
                x1
                + (x2 - x1)
                * i
                / steps
            )

            y = int(
                y1
                + (y2 - y1)
                * i
                / steps
            )

            if (
                0 <= x < width
                and 0 <= y < height
            ):
                pixels[
                    y,
                    x,
                ] = rgb

    for i in range(
        len(actual) - 1
    ):
        draw_segment(
            point(i, actual[i]),
            point(i + 1, actual[i + 1]),
            (40, 90, 180),
        )

        draw_segment(
            point(i, predicted[i]),
            point(i + 1, predicted[i + 1]),
            (180, 70, 70),
        )

    # Axes
    pixels[
        top:bottom + 1,
        left,
    ] = 0

    pixels[
        bottom,
        left:right + 1,
    ] = 0

    save_png(
        path,
        width,
        height,
        pixels,
    )


def draw_future_chart(
    path: Path,
    values,
) -> None:

    width = 1200
    height = 600

    pixels = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )

    values = np.asarray(
        values,
        dtype=float,
    )

    low = min(
        float(np.min(values)),
        0.0,
    )

    high = float(np.max(values))

    if high <= low:
        high = low + 1.0

    left = 60
    right = width - 30
    top = 30
    bottom = height - 50

    def point(
        index,
        value,
    ):

        x = left + (
            index
            / max(1, len(values) - 1)
        ) * (right - left)

        y = bottom - (
            (value - low)
            / (high - low)
        ) * (bottom - top)

        return int(x), int(y)

    def draw_segment(
        p1,
        p2,
    ):

        x1, y1 = p1
        x2, y2 = p2

        steps = max(
            abs(x2 - x1),
            abs(y2 - y1),
            1,
        )

        for i in range(steps + 1):

            x = int(
                x1
                + (x2 - x1)
                * i
                / steps
            )

            y = int(
                y1
                + (y2 - y1)
                * i
                / steps
            )

            if (
                0 <= x < width
                and 0 <= y < height
            ):
                pixels[
                    y,
                    x,
                ] = (70, 100, 170)

    for i in range(
        len(values) - 1
    ):
        draw_segment(
            point(i, values[i]),
            point(i + 1, values[i + 1]),
        )

    # Axes
    pixels[
        top:bottom + 1,
        left,
    ] = 0

    pixels[
        bottom,
        left:right + 1,
    ] = 0

    save_png(
        path,
        width,
        height,
        pixels,
    )


def draw_importance_chart(
    path: Path,
    importance: pd.DataFrame,
) -> None:

    top_n = importance.head(20)

    width = 1200
    height = 700

    pixels = np.full(
        (height, width, 3),
        255,
        dtype=np.uint8,
    )

    values = top_n[
        "importance"
    ].to_numpy(
        dtype=float
    )

    if len(values) == 0:
        save_png(
            path,
            width,
            height,
            pixels,
        )
        return

    maximum = max(
        float(np.max(values)),
        1e-9,
    )

    left = 50
    right = width - 50
    top = 30
    bottom = height - 30

    bar_height = max(
        5,
        int(
            (bottom - top)
            / max(
                1,
                len(values),
            )
            * 0.65
        ),
    )

    spacing = (
        (bottom - top)
        / max(1, len(values))
    )

    for i, value in enumerate(values):

        y_center = int(
            top
            + spacing * i
            + spacing / 2
        )

        bar_width = int(
            (
                value
                / maximum
            )
            * (right - left)
        )

        y1 = max(
            0,
            y_center - bar_height // 2,
        )

        y2 = min(
            height,
            y_center + bar_height // 2,
        )

        x2 = min(
            width,
            left + bar_width,
        )

        pixels[
            y1:y2,
            left:x2,
        ] = (80, 120, 170)

    save_png(
        path,
        width,
        height,
        pixels,
    )


# ============================================================================
# REPORT
# ============================================================================

def create_report(
    source_path,
    raw_df,
    model_df,
    train,
    validation,
    test,
    model_metrics,
    baseline_metrics,
    future_forecast,
    model,
) -> str:

    best_iteration = getattr(
        model,
        "best_iteration",
        None,
    )

    lines = []

    lines.append(
        "CAREOPS-AI ICU DEMAND FORECASTING REPORT"
    )

    lines.append("=" * 60)

    lines.append("")
    lines.append("DATA SOURCE")
    lines.append("-" * 60)
    lines.append(str(source_path))

    lines.append("")
    lines.append("DATASET")
    lines.append("-" * 60)
    lines.append(
        f"Raw rows: {len(raw_df)}"
    )
    lines.append(
        f"Modeling rows: {len(model_df)}"
    )
    lines.append(
        f"Date range: "
        f"{raw_df['date'].min().date()} -> "
        f"{raw_df['date'].max().date()}"
    )

    lines.append("")
    lines.append("MODEL")
    lines.append("-" * 60)
    lines.append(
        "Native XGBoost API: xgb.train"
    )
    lines.append(
        "scikit-learn: NOT USED"
    )
    lines.append(
        "SciPy: NOT USED"
    )
    lines.append(
        "Matplotlib: NOT USED"
    )

    lines.append(
        f"Number of features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    lines.append(
        f"Boosting rounds configured: "
        f"{NUM_BOOST_ROUNDS}"
    )

    lines.append(
        f"Best iteration: "
        f"{best_iteration}"
    )

    lines.append("")
    lines.append("DATA SPLIT")
    lines.append("-" * 60)
    lines.append(
        f"Training rows: {len(train)}"
    )
    lines.append(
        f"Validation rows: {len(validation)}"
    )
    lines.append(
        f"Test rows: {len(test)}"
    )

    lines.append("")
    lines.append("XGBOOST TEST METRICS")
    lines.append("-" * 60)

    for key, value in model_metrics.items():
        lines.append(
            f"{key}: {value}"
        )

    lines.append("")
    lines.append("7-DAY NAIVE BASELINE")
    lines.append("-" * 60)

    for key, value in baseline_metrics.items():
        lines.append(
            f"{key}: {value}"
        )

    lines.append("")
    lines.append("CAPACITY LIMITATION")
    lines.append("-" * 60)
    lines.append(
        "Forecast bed utilization is not estimated because a validated ICU "
        "census series is unavailable."
    )
    lines.append(
        "Capacity risk is deliberately not inferred from daily admissions."
    )

    lines.append("")
    lines.append("FUTURE FORECAST")
    lines.append("-" * 60)

    lines.append(
        f"Forecast horizon: "
        f"{len(future_forecast)} days"
    )

    lines.append(
        f"Forecast start: "
        f"{future_forecast['date'].min().date()}"
    )

    lines.append(
        f"Forecast end: "
        f"{future_forecast['date'].max().date()}"
    )

    lines.append(
        f"Maximum predicted ICU admissions per day: "
        f"{future_forecast['predicted_icu_admissions'].max():.2f}"
    )

    risk_counts = (
        future_forecast[
            "capacity_risk"
        ]
        .value_counts()
        .to_dict()
    )

    lines.append(
        f"Capacity risk counts (census unavailable): "
        f"{risk_counts}"
    )

    lines.append("")
    lines.append("FEATURES")
    lines.append("-" * 60)

    for feature in FEATURE_COLUMNS:
        lines.append(
            f"- {feature}"
        )

    lines.append("")
    lines.append("OUTPUT FILES")
    lines.append("-" * 60)

    output_names = [
        "icu_xgboost_model.json",
        "icu_xgboost_model.joblib",
        "icu_demand_test_predictions.csv",
        "icu_future_forecast.csv",
        "icu_weekly_forecast.csv",
        "icu_capacity_risk_forecast.csv",
        "icu_feature_importance.csv",
        "icu_demand_metrics.json",
        "icu_demand_model_metadata.json",
        "icu_data_quality.csv",
        "icu_training_summary.csv",
        "icu_actual_vs_predicted.png",
        "icu_future_forecast.png",
        "icu_feature_importance.png",
        "ICU_DEMAND_FORECAST_REPORT.txt",
    ]

    for name in output_names:
        lines.append(
            f"- {name}"
        )

    return "\n".join(lines)


# ============================================================================
# SAVE MODEL
# ============================================================================

def save_model(
    model,
) -> None:

    # Native XGBoost model format.
    model_json = (
        OUTPUT_DIR
        / "icu_xgboost_model.json"
    )

    model.save_model(
        str(model_json)
    )

    # Joblib copy for portfolio compatibility.
    joblib_path = (
        OUTPUT_DIR
        / "icu_xgboost_model.joblib"
    )

    try:
        joblib.dump(
            model,
            joblib_path,
        )
    except Exception as exc:
        warn(
            "Joblib serialization failed, "
            "but native XGBoost model was saved."
        )
        warn(str(exc))


# ============================================================================
# MAIN
# ============================================================================

def main():

    total_start = time.perf_counter()

    print()
    print("=" * 72)
    print("CAREOPS-AI ICU DEMAND FORECASTING")
    print("=" * 72)

    print(
        "Native XGBoost API: YES"
    )
    print(
        "scikit-learn: NOT USED"
    )
    print(
        "SciPy: NOT USED"
    )
    print(
        "Matplotlib: NOT USED"
    )
    print(
        "Existing synthetic data will NOT be regenerated."
    )

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    raw_df, source_path = (
        load_source_data()
    )

    # ------------------------------------------------------------------
    # DAILY SERIES
    # ------------------------------------------------------------------

    daily_df = build_daily_series(
        raw_df
    )

    # ------------------------------------------------------------------
    # FEATURES
    # ------------------------------------------------------------------

    model_df = add_features(
        daily_df
    )

    print()
    print(
        f"Number of model features: "
        f"{len(FEATURE_COLUMNS)}"
    )

    print("Features:")

    for feature in FEATURE_COLUMNS:
        print(
            f" - {feature}"
        )

    # ------------------------------------------------------------------
    # SPLIT
    # ------------------------------------------------------------------

    train, validation, test = (
        chronological_split(
            model_df
        )
    )

    # ------------------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------------------

    model = train_model(
        train,
        validation,
    )

    # ------------------------------------------------------------------
    # EVALUATE
    # ------------------------------------------------------------------

    (
        model_metrics,
        baseline_metrics,
        prediction_df,
    ) = evaluate_model(
        model,
        train,
        test,
    )

    # ------------------------------------------------------------------
    # FEATURE IMPORTANCE
    # ------------------------------------------------------------------

    feature_importance = (
        calculate_feature_importance(
            model
        )
    )

    # ------------------------------------------------------------------
    # FUTURE FORECAST
    # ------------------------------------------------------------------

    future_forecast = (
        create_future_forecast(
            model,
            daily_df,
            FORECAST_HORIZON_DAYS,
        )
    )

    # ------------------------------------------------------------------
    # WEEKLY FORECAST
    # ------------------------------------------------------------------

    weekly_forecast = (
        create_weekly_forecast(
            future_forecast
        )
    )

    # ------------------------------------------------------------------
    # CAPACITY RISK
    # ------------------------------------------------------------------

    capacity_risk = (
        create_capacity_risk(
            future_forecast
        )
    )

    # ------------------------------------------------------------------
    # DATA QUALITY
    # ------------------------------------------------------------------

    data_quality = (
        create_data_quality_report(
            raw_df,
            model_df,
        )
    )

    # ------------------------------------------------------------------
    # TRAINING SUMMARY
    # ------------------------------------------------------------------

    training_summary = (
        create_training_summary(
            train,
            validation,
            test,
            model_metrics,
            baseline_metrics,
        )
    )

    # ------------------------------------------------------------------
    # SAVE MODEL
    # ------------------------------------------------------------------

    banner("SAVING ICU MODEL")

    save_model(model)

    # ------------------------------------------------------------------
    # SAVE PREDICTIONS
    # ------------------------------------------------------------------

    prediction_df.to_csv(
        OUTPUT_DIR
        / "icu_demand_test_predictions.csv",
        index=False,
    )

    future_forecast.to_csv(
        OUTPUT_DIR
        / "icu_future_forecast.csv",
        index=False,
    )

    weekly_forecast.to_csv(
        OUTPUT_DIR
        / "icu_weekly_forecast.csv",
        index=False,
    )

    capacity_risk.to_csv(
        OUTPUT_DIR
        / "icu_capacity_risk_forecast.csv",
        index=False,
    )

    feature_importance.to_csv(
        OUTPUT_DIR
        / "icu_feature_importance.csv",
        index=False,
    )

    data_quality.to_csv(
        OUTPUT_DIR
        / "icu_data_quality.csv",
        index=False,
    )

    training_summary.to_csv(
        OUTPUT_DIR
        / "icu_training_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # METRICS JSON
    # ------------------------------------------------------------------

    metrics_payload = {
        "model": "XGBoost native API",
        "target": "daily ICU admissions",
        "target_unit": "admissions_per_day",
        "capacity_note": (
            "ICU bed occupancy and capacity risk are not estimated from admissions."
        ),
        "test_metrics": model_metrics,
        "baseline_7_day_metrics": baseline_metrics,
        "forecast_horizon_days": (
            FORECAST_HORIZON_DAYS
        ),
        "feature_count": len(
            FEATURE_COLUMNS
        ),
    }

    with open(
        OUTPUT_DIR
        / "icu_demand_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics_payload,
            file,
            indent=4,
        )

    # ------------------------------------------------------------------
    # MODEL METADATA
    # ------------------------------------------------------------------

    metadata = {
        "project": "CareOps-AI",
        "module": "ICU Demand Forecasting",
        "target": "daily ICU admissions, not occupied-bed census",
        "target_unit": "admissions_per_day",
        "capacity_risk": "not estimated; ICU census series is unavailable",
        "source_file": str(
            source_path
        ),
        "python_executable": os.sys.executable,
        "xgboost_version": xgb.__version__,
        "scikit_learn_used": False,
        "scipy_used": False,
        "matplotlib_used": False,
        "seed": SEED,
        "feature_count": len(
            FEATURE_COLUMNS
        ),
        "features": FEATURE_COLUMNS,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "train_start": str(
            train["date"].min().date()
        ),
        "train_end": str(
            train["date"].max().date()
        ),
        "validation_start": str(
            validation["date"].min().date()
        ),
        "validation_end": str(
            validation["date"].max().date()
        ),
        "test_start": str(
            test["date"].min().date()
        ),
        "test_end": str(
            test["date"].max().date()
        ),
        "forecast_horizon_days": (
            FORECAST_HORIZON_DAYS
        ),
        "xgboost_parameters": XGB_PARAMS,
        "num_boost_round": NUM_BOOST_ROUNDS,
        "early_stopping_rounds": (
            EARLY_STOPPING_ROUNDS
        ),
    }

    with open(
        OUTPUT_DIR
        / "icu_demand_model_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            default=str,
        )

    # ------------------------------------------------------------------
    # LIGHTWEIGHT VISUALIZATIONS
    # ------------------------------------------------------------------

    banner(
        "CREATING LIGHTWEIGHT VISUALIZATIONS"
    )

    draw_line_chart(
        OUTPUT_DIR
        / "icu_actual_vs_predicted.png",
        prediction_df[
            "actual_icu_demand"
        ].values,
        prediction_df[
            "predicted_icu_demand"
        ].values,
    )

    draw_future_chart(
        OUTPUT_DIR
        / "icu_future_forecast.png",
        future_forecast[
            "predicted_icu_demand"
        ].values,
    )

    draw_importance_chart(
        OUTPUT_DIR
        / "icu_feature_importance.png",
        feature_importance,
    )

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------

    report = create_report(
        source_path,
        raw_df,
        model_df,
        train,
        validation,
        test,
        model_metrics,
        baseline_metrics,
        future_forecast,
        model,
    )

    with open(
        OUTPUT_DIR
        / "ICU_DEMAND_FORECAST_REPORT.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(report)

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------

    elapsed = (
        time.perf_counter()
        - total_start
    )

    banner(
        "ICU DEMAND FORECASTING COMPLETED SUCCESSFULLY"
    )

    print(
        f"Total runtime: {elapsed:.2f} seconds"
    )

    print()
    print("Test performance:")
    print(
        f"MAE:  {model_metrics['MAE']:.4f}"
    )
    print(
        f"RMSE: {model_metrics['RMSE']:.4f}"
    )
    print(
        f"R2:   {model_metrics['R2']:.4f}"
    )
    print(
        f"MAPE: {model_metrics['MAPE_percent']:.2f}%"
    )

    print()
    print(
        "Future ICU forecast:"
    )

    print(
        f"Maximum predicted demand: "
        f"{future_forecast['predicted_icu_demand'].max():.2f}"
    )

    print("ICU bed-capacity risk: not estimated (validated census unavailable)")

    print()
    print(
        "Output directory:"
    )

    print(
        OUTPUT_DIR
    )

    print()
    print(
        "Created files:"
    )

    for file in sorted(
        OUTPUT_DIR.iterdir()
    ):
        if file.is_file():
            print(
                f" - {file.name}"
            )


# ============================================================================
# SAFE ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        print()
        print("=" * 72)
        print("ICU MODULE FAILED")
        print("=" * 72)

        print(
            f"Error type: "
            f"{type(exc).__name__}"
        )

        print(
            f"Error: {exc}"
        )

        print()
        traceback.print_exc()

        raise
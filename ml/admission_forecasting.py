"""
CAREOPS AI
Healthcare Operations Optimization Platform
Module: Admission Demand Forecasting

Purpose
-------
Forecast daily hospital admissions using historical synthetic hospital data.

This program:
1. Automatically discovers the CareOps AI project root.
2. Automatically loads CSV files from:
      data/
      data/processed/
      data/synthetic/
      data/feature_engineering/
3. Does NOT expect files with "_clean" in their names.
4. Automatically detects the best daily admissions dataset.
5. Creates a leakage-safe time-series dataset.
6. Creates:
      - Lag 1 day
      - Lag 7 days
      - Lag 14 days
      - Lag 28 days
      - Rolling 7-day mean
      - Rolling 14-day mean
      - Rolling 28-day mean
      - Calendar features
      - Trend features
7. Builds:
      - Seasonal naive baseline
      - XGBoost forecasting model
8. Uses chronological train/test validation.
9. Calculates:
      - MAE
      - RMSE
      - MAPE
10. Produces:
      - Hospital-wide forecast
      - Department-level forecasts
      - 30-day future forecast
11. Saves:
      - metrics
      - predictions
      - future forecasts
      - feature importance
      - trained model
      - charts
      - report
      - data catalog
12. Does not modify the original CSV files.

IMPORTANT
---------
This is an operations decision-support forecasting module.
It is NOT a clinical diagnosis or treatment system.
All data used for this public portfolio project should remain synthetic/de-identified.
"""

# ============================================================
# 1. IMPORTS
# ============================================================

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Matplotlib
import matplotlib.pyplot as plt

# XGBoost uses the native backend already used by the ICU forecaster.
try:
    from xgboost import XGBRegressor
except ImportError:
    print("\nERROR: XGBoost is not installed.")
    print("Install the project's XGBoost dependency before forecasting.")
    sys.exit(1)

from sklearn.metrics import mean_absolute_error, mean_squared_error

# Joblib is normally installed with scikit-learn
try:
    import joblib
except ImportError:
    print("\nERROR: joblib is not installed.")
    print("Run this in the CMD terminal:")
    print("python -m pip install joblib")
    sys.exit(1)

warnings.filterwarnings("ignore")


# ============================================================
# 2. GLOBAL SETTINGS
# ============================================================

RANDOM_STATE = 42

TEST_FRACTION = 0.20

FORECAST_DAYS = 30

MIN_HISTORY_DAYS = 60

LAGS = [1, 7, 14, 28]

ROLLING_WINDOWS = [7, 14, 28]

MODEL_NAME = "XGBRegressor"

XGB_CONFIG = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.04,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 2,
    "reg_lambda": 2.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

XGB_CONFIG_CANDIDATES = {
    "shallow": {
        "n_estimators": 300,
        "max_depth": 1,
        "learning_rate": 0.04,
        "min_child_weight": 5,
        "reg_lambda": 5.0,
    },
    "regularized": {
        "n_estimators": 120,
        "max_depth": 2,
        "learning_rate": 0.025,
        "min_child_weight": 8,
        "reg_lambda": 8.0,
    },
    "mid": {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.03,
        "min_child_weight": 5,
        "reg_lambda": 5.0,
    },
}


# ============================================================
# 3. FIND CAREOPS AI PROJECT ROOT AUTOMATICALLY
# ============================================================

def find_project_root():
    """
    Automatically locate the CareOps-AI project root.

    Expected structure:

    CareOps-AI/
        data/
        data/processed/
        data/synthetic/
        data/feature_engineering/
        ml/
            admission_forecasting.py
    """

    current_file = Path(__file__).resolve()

    possible_roots = [
        current_file.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
        Path.cwd().parent.parent
    ]

    for root in possible_roots:
        root = Path(root)

        if (
            (root / "data").exists()
            and root.name.lower().replace("_", "-") == "careops-ai"
        ):
            return root

    # More flexible fallback
    for root in possible_roots:
        root = Path(root)

        if (root / "data").exists():
            return root

    raise FileNotFoundError(
        "Could not locate the CareOps-AI project root.\n"
        "Make sure admission_forecasting.py is inside:\n"
        "CareOps-AI\\ml\\"
    )


PROJECT_ROOT = find_project_root()

DATA_ROOT = PROJECT_ROOT / "data"

PROCESSED_DIR = DATA_ROOT / "processed"

SYNTHETIC_DIR = DATA_ROOT / "synthetic"

FEATURE_DIR = DATA_ROOT / "feature_engineering"

OUTPUT_DIR = PROJECT_ROOT / "ml" / "admission_forecasting"

CHART_DIR = OUTPUT_DIR / "charts"

MODEL_DIR = OUTPUT_DIR / "models"

REPORT_DIR = OUTPUT_DIR / "reports"


# Create output folders automatically
for directory in [
    OUTPUT_DIR,
    CHART_DIR,
    MODEL_DIR,
    REPORT_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 4. PRINT PROJECT INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("CAREOPS AI - ADMISSION DEMAND FORECASTING")
print("=" * 80)

print(f"\nProject root:")
print(PROJECT_ROOT)

print(f"\nData root:")
print(DATA_ROOT)

print(f"\nFeature-engineering directory:")
print(FEATURE_DIR)

print(f"\nOutput directory:")
print(OUTPUT_DIR)


# ============================================================
# 5. AUTOMATICALLY FIND ALL CSV FILES
# ============================================================

def discover_csv_files():
    """
    Discover every CSV inside the major CareOps AI data folders.

    We intentionally do not require filenames such as *_clean.csv.
    """

    scenario_data_dir = os.environ.get("CAREOPS_DATA_DIR")
    search_directories = (
        [Path(scenario_data_dir)]
        if scenario_data_dir
        else [DATA_ROOT, PROCESSED_DIR, SYNTHETIC_DIR, FEATURE_DIR]
    )

    csv_files = set()

    for directory in search_directories:

        if not directory.exists():
            continue

        csv_paths = (
            directory.glob("*.csv")
            if scenario_data_dir
            else directory.rglob("*.csv")
        )
        for csv_file in csv_paths:

            # Ignore files generated by this forecasting module
            # if the program is run again.
            try:
                csv_file.relative_to(OUTPUT_DIR)
                continue
            except ValueError:
                pass

            csv_files.add(csv_file.resolve())

    return sorted(csv_files)


CSV_FILES = discover_csv_files()

print("\n" + "-" * 80)
print("DISCOVERED CSV FILES")
print("-" * 80)

if not CSV_FILES:
    raise FileNotFoundError(
        "No CSV files were found inside the CareOps-AI data directory."
    )

for index, csv_file in enumerate(CSV_FILES, start=1):
    print(f"{index:02d}. {csv_file}")


# ============================================================
# 6. LOAD ALL CSV DATASETS AUTOMATICALLY
# ============================================================

def load_all_csvs(csv_files):
    """
    Load every discovered CSV into a dictionary.

    The datasets are not blindly merged because indiscriminately
    merging operational datasets can introduce duplicate rows
    or future-data leakage.

    Instead, the program loads everything and then intelligently
    identifies the appropriate source for admission forecasting.
    """

    datasets = {}

    for csv_file in csv_files:

        try:

            df = pd.read_csv(
                csv_file,
                low_memory=False
            )

            key = str(csv_file)

            datasets[key] = df

            print(
                f"\nLoaded: {csv_file.name}"
                f" | rows={len(df):,}"
                f" | columns={len(df.columns)}"
            )

        except Exception as error:

            print(
                f"\nWARNING: Could not load {csv_file}"
            )

            print(f"Reason: {error}")

    return datasets


ALL_DATA = load_all_csvs(CSV_FILES)

if not ALL_DATA:
    raise RuntimeError(
        "No CSV datasets could be loaded."
    )


# ============================================================
# 7. CREATE DATA CATALOG
# ============================================================

def create_data_catalog(datasets):

    catalog = []

    for path_string, df in datasets.items():

        path = Path(path_string)

        catalog.append({
            "file_name": path.name,
            "full_path": str(path),
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "column_names": ", ".join(
                [str(column) for column in df.columns]
            )
        })

    return pd.DataFrame(catalog)


DATA_CATALOG = create_data_catalog(ALL_DATA)

DATA_CATALOG.to_csv(
    OUTPUT_DIR / "data_catalog.csv",
    index=False
)


# ============================================================
# 8. COLUMN NORMALIZATION
# ============================================================

def normalize_column_name(column):

    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def normalized_columns(df):

    mapping = {}

    for column in df.columns:

        mapping[column] = normalize_column_name(column)

    return mapping


# ============================================================
# 9. IDENTIFY COLUMNS AUTOMATICALLY
# ============================================================

def find_column(df, candidates):

    normalized = normalized_columns(df)

    for original, normalized_name in normalized.items():

        if normalized_name in candidates:
            return original

    return None


DATE_CANDIDATES = {
    "date",
    "admission_date",
    "admit_date",
    "event_date",
    "operation_date",
    "day"
}

DEPARTMENT_CANDIDATES = {
    "department",
    "dept",
    "department_name",
    "service",
    "specialty"
}

ADMISSIONS_CANDIDATES = {
    "admissions",
    "admission",
    "daily_admissions",
    "admission_count",
    "admissions_count",
    "num_admissions",
    "number_of_admissions"
}


# ============================================================
# 10. SCORE POTENTIAL ADMISSION DATASETS
# ============================================================

def score_admission_dataset(path_string, df):

    score = 0

    normalized = set(
        normalized_columns(df).values()
    )

    date_column = find_column(
        df,
        DATE_CANDIDATES
    )

    department_column = find_column(
        df,
        DEPARTMENT_CANDIDATES
    )

    admissions_column = find_column(
        df,
        ADMISSIONS_CANDIDATES
    )

    # Strongest signal:
    # dataset already contains daily admissions.
    if admissions_column is not None:
        score += 100

    # Date is required.
    if date_column is not None:
        score += 50

    # Department makes department forecasting possible.
    if department_column is not None:
        score += 30

    filename = Path(path_string).name.lower()

    # Prefer daily operations.
    if "daily_operations" in filename:
        score += 100

    # Feature-engineered data gets preference.
    if "feature_engineering" in str(path_string).lower():
        score += 20

    # Processed data gets preference over raw synthetic data.
    if "processed" in str(path_string).lower():
        score += 10

    return score


# ============================================================
# 11. SELECT BEST ADMISSION SOURCE
# ============================================================

candidate_scores = []

for path_string, df in ALL_DATA.items():

    score = score_admission_dataset(
        path_string,
        df
    )

    candidate_scores.append({
        "path": path_string,
        "score": score
    })


candidate_scores_df = pd.DataFrame(
    candidate_scores
).sort_values(
    "score",
    ascending=False
)

candidate_scores_df.to_csv(
    OUTPUT_DIR / "candidate_dataset_scores.csv",
    index=False
)


BEST_PATH = candidate_scores_df.iloc[0]["path"]

BEST_DATA = ALL_DATA[BEST_PATH].copy()

print("\n" + "-" * 80)
print("SELECTED ADMISSION FORECASTING SOURCE")
print("-" * 80)

print(BEST_PATH)

print("\nColumns detected:")
for column in BEST_DATA.columns:
    print(f" - {column}")


# ============================================================
# 12. IDENTIFY REQUIRED COLUMNS
# ============================================================

DATE_COLUMN = find_column(
    BEST_DATA,
    DATE_CANDIDATES
)

DEPARTMENT_COLUMN = find_column(
    BEST_DATA,
    DEPARTMENT_CANDIDATES
)

ADMISSIONS_COLUMN = find_column(
    BEST_DATA,
    ADMISSIONS_CANDIDATES
)


print("\nDetected columns:")
print(f"Date       : {DATE_COLUMN}")
print(f"Department : {DEPARTMENT_COLUMN}")
print(f"Admissions : {ADMISSIONS_COLUMN}")


# ============================================================
# 13. FALLBACK: BUILD DAILY ADMISSIONS FROM ADMISSIONS DATASET
# ============================================================

def find_patient_admissions_dataset():

    best_score = -1
    best_path = None
    best_df = None

    for path_string, df in ALL_DATA.items():

        normalized = set(
            normalized_columns(df).values()
        )

        admission_date_column = find_column(
            df,
            {
                "admission_date",
                "admit_date"
            }
        )

        if admission_date_column is None:
            continue

        score = 50

        filename = Path(path_string).name.lower()

        if "admissions" in filename:
            score += 100

        if "patients" in filename:
            score += 20

        if score > best_score:
            best_score = score
            best_path = path_string
            best_df = df.copy()

    return best_path, best_df


if (
    DATE_COLUMN is None
    or ADMISSIONS_COLUMN is None
):

    print(
        "\nDaily admissions columns were not directly detected."
    )

    print(
        "Attempting to construct daily admissions from admissions records..."
    )

    fallback_path, fallback_df = find_patient_admissions_dataset()

    if fallback_df is None:

        raise ValueError(
            "\nCould not find a suitable admission dataset.\n"
            "The program expected either:\n"
            "1. A daily operations dataset containing date + admissions,\n"
            "or\n"
            "2. An admissions dataset containing admission_date."
        )

    fallback_date = find_column(
        fallback_df,
        {
            "admission_date",
            "admit_date"
        }
    )

    fallback_department = find_column(
        fallback_df,
        DEPARTMENT_CANDIDATES
    )

    fallback_df[fallback_date] = pd.to_datetime(
        fallback_df[fallback_date],
        errors="coerce"
    )

    fallback_df = fallback_df.dropna(
        subset=[fallback_date]
    )

    fallback_df["forecast_date"] = (
        fallback_df[fallback_date].dt.normalize()
    )

    if fallback_department is not None:

        DAILY_ADMISSIONS = (
            fallback_df
            .groupby(
                [
                    "forecast_date",
                    fallback_department
                ],
                dropna=False
            )
            .size()
            .reset_index(
                name="admissions"
            )
        )

        DAILY_ADMISSIONS = DAILY_ADMISSIONS.rename(
            columns={
                fallback_department: "department"
            }
        )

    else:

        DAILY_ADMISSIONS = (
            fallback_df
            .groupby("forecast_date")
            .size()
            .reset_index(
                name="admissions"
            )
        )

        DAILY_ADMISSIONS["department"] = "All Departments"

    DAILY_ADMISSIONS = DAILY_ADMISSIONS.rename(
        columns={
            "forecast_date": "date"
        }
    )

    SOURCE_TYPE = "Constructed from admission records"

    SOURCE_USED = fallback_path

else:

    # ========================================================
    # 14. BUILD STANDARD DAILY ADMISSIONS DATASET
    # ========================================================

    working = BEST_DATA.copy()

    working[DATE_COLUMN] = pd.to_datetime(
        working[DATE_COLUMN],
        errors="coerce"
    )

    working = working.dropna(
        subset=[DATE_COLUMN]
    )

    # Convert admissions to numeric.
    working[ADMISSIONS_COLUMN] = pd.to_numeric(
        working[ADMISSIONS_COLUMN],
        errors="coerce"
    )

    working[ADMISSIONS_COLUMN] = (
        working[ADMISSIONS_COLUMN]
        .fillna(0)
        .clip(lower=0)
    )

    working["date"] = (
        working[DATE_COLUMN]
        .dt.normalize()
    )

    if DEPARTMENT_COLUMN is not None:

        working["department"] = (
            working[DEPARTMENT_COLUMN]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
        )

        working.loc[
            working["department"] == "",
            "department"
        ] = "Unknown"

        DAILY_ADMISSIONS = (
            working
            .groupby(
                [
                    "date",
                    "department"
                ],
                as_index=False
            )[
                ADMISSIONS_COLUMN
            ]
            .sum()
            .rename(
                columns={
                    ADMISSIONS_COLUMN: "admissions"
                }
            )
        )

    else:

        DAILY_ADMISSIONS = (
            working
            .groupby(
                "date",
                as_index=False
            )[
                ADMISSIONS_COLUMN
            ]
            .sum()
            .rename(
                columns={
                    ADMISSIONS_COLUMN: "admissions"
                }
            )
        )

        DAILY_ADMISSIONS["department"] = (
            "All Departments"
        )

    SOURCE_TYPE = "Existing daily admissions dataset"

    SOURCE_USED = BEST_PATH


# ============================================================
# 15. CLEAN DAILY ADMISSIONS
# ============================================================

DAILY_ADMISSIONS["date"] = pd.to_datetime(
    DAILY_ADMISSIONS["date"],
    errors="coerce"
)

DAILY_ADMISSIONS["admissions"] = pd.to_numeric(
    DAILY_ADMISSIONS["admissions"],
    errors="coerce"
)

DAILY_ADMISSIONS = DAILY_ADMISSIONS.dropna(
    subset=[
        "date",
        "admissions"
    ]
)

DAILY_ADMISSIONS["admissions"] = (
    DAILY_ADMISSIONS["admissions"]
    .clip(lower=0)
)


DAILY_ADMISSIONS["department"] = (
    DAILY_ADMISSIONS["department"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

DAILY_ADMISSIONS.loc[
    DAILY_ADMISSIONS["department"] == "",
    "department"
] = "Unknown"


DAILY_ADMISSIONS = (
    DAILY_ADMISSIONS
    .groupby(
        [
            "date",
            "department"
        ],
        as_index=False
    )["admissions"]
    .sum()
)


DAILY_ADMISSIONS = DAILY_ADMISSIONS.sort_values(
    [
        "department",
        "date"
    ]
).reset_index(drop=True)


# Save standardized input
DAILY_ADMISSIONS.to_csv(
    OUTPUT_DIR / "daily_admissions_used.csv",
    index=False
)


# ============================================================
# 16. BASIC DATA VALIDATION
# ============================================================

if DAILY_ADMISSIONS.empty:

    raise ValueError(
        "The standardized daily admissions dataset is empty."
    )


date_min = DAILY_ADMISSIONS["date"].min()

date_max = DAILY_ADMISSIONS["date"].max()

number_departments = (
    DAILY_ADMISSIONS["department"]
    .nunique()
)

number_days = (
    DAILY_ADMISSIONS["date"]
    .nunique()
)

total_admissions = (
    DAILY_ADMISSIONS["admissions"]
    .sum()
)


print("\n" + "-" * 80)
print("ADMISSION DATA VALIDATION")
print("-" * 80)

print(f"First date              : {date_min.date()}")
print(f"Last date               : {date_max.date()}")
print(f"Unique departments      : {number_departments}")
print(f"Unique dates            : {number_days:,}")
print(f"Total admissions        : {total_admissions:,.0f}")


if number_days < MIN_HISTORY_DAYS:

    raise ValueError(
        f"Only {number_days} days of data were found. "
        f"At least {MIN_HISTORY_DAYS} days are recommended "
        f"for this forecasting program."
    )


# ============================================================
# 17. CREATE COMPLETE DAILY TIME SERIES
# ============================================================

def complete_department_time_series(
    df,
    department
):

    department_df = df[
        df["department"] == department
    ].copy()

    department_df = (
        department_df
        .groupby(
            "date",
            as_index=False
        )["admissions"]
        .sum()
    )

    full_dates = pd.date_range(
        department_df["date"].min(),
        department_df["date"].max(),
        freq="D"
    )

    complete = pd.DataFrame({
        "date": full_dates
    })

    complete = complete.merge(
        department_df,
        on="date",
        how="left"
    )

    # A missing department/day means zero recorded admissions
    # in the synthetic operational dataset.
    complete["admissions"] = (
        complete["admissions"]
        .fillna(0)
        .clip(lower=0)
    )

    complete["department"] = department

    return complete


departments = sorted(
    DAILY_ADMISSIONS["department"]
    .unique()
    .tolist()
)


complete_series_list = []

for department in departments:

    department_series = complete_department_time_series(
        DAILY_ADMISSIONS,
        department
    )

    complete_series_list.append(
        department_series
    )


COMPLETE_SERIES = pd.concat(
    complete_series_list,
    ignore_index=True
)


COMPLETE_SERIES = COMPLETE_SERIES.sort_values(
    [
        "department",
        "date"
    ]
).reset_index(drop=True)


COMPLETE_SERIES.to_csv(
    OUTPUT_DIR / "complete_daily_admissions_series.csv",
    index=False
)


# ============================================================
# 18. FEATURE ENGINEERING FOR TIME SERIES
# ============================================================

def create_time_features(df):

    result = df.copy()

    result["year"] = result["date"].dt.year

    result["month"] = result["date"].dt.month

    result["day"] = result["date"].dt.day

    result["day_of_week"] = (
        result["date"].dt.dayofweek
    )

    result["day_of_year"] = (
        result["date"].dt.dayofyear
    )

    result["week_of_year"] = (
        result["date"].dt.isocalendar().week
        .astype(int)
    )

    result["quarter"] = (
        result["date"].dt.quarter
    )

    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    result["is_month_start"] = (
        result["date"].dt.is_month_start
    ).astype(int)

    result["is_month_end"] = (
        result["date"].dt.is_month_end
    ).astype(int)

    result["is_quarter_start"] = (
        result["date"].dt.is_quarter_start
    ).astype(int)

    result["is_quarter_end"] = (
        result["date"].dt.is_quarter_end
    ).astype(int)

    # Cyclic representation of weekday.
    result["dow_sin"] = np.sin(
        2 * np.pi * result["day_of_week"] / 7
    )

    result["dow_cos"] = np.cos(
        2 * np.pi * result["day_of_week"] / 7
    )

    # Cyclic representation of month.
    result["month_sin"] = np.sin(
        2 * np.pi * result["month"] / 12
    )

    result["month_cos"] = np.cos(
        2 * np.pi * result["month"] / 12
    )

    # Cyclic representation of annual seasonality.
    result["doy_sin"] = np.sin(
        2 * np.pi * result["day_of_year"] / 365.25
    )

    result["doy_cos"] = np.cos(
        2 * np.pi * result["day_of_year"] / 365.25
    )

    return result


def create_forecasting_features(
    department_df
):

    result = department_df.copy()

    result = result.sort_values(
        "date"
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------

    result = create_time_features(
        result
    )

    # --------------------------------------------------------
    # Trend feature
    # --------------------------------------------------------

    result["trend"] = np.arange(
        len(result)
    )

    # --------------------------------------------------------
    # Lag features
    # --------------------------------------------------------

    for lag in LAGS:

        result[f"lag_{lag}"] = (
            result["admissions"]
            .shift(lag)
        )

    # --------------------------------------------------------
    # Rolling features
    #
    # IMPORTANT:
    # Shift by one day before rolling.
    #
    # This prevents today's target from being used to predict
    # today's target.
    # --------------------------------------------------------

    previous_day = (
        result["admissions"]
        .shift(1)
    )

    for window in ROLLING_WINDOWS:

        result[
            f"rolling_mean_{window}"
        ] = (
            previous_day
            .rolling(
                window=window,
                min_periods=window
            )
            .mean()
        )

        result[
            f"rolling_std_{window}"
        ] = (
            previous_day
            .rolling(
                window=window,
                min_periods=window
            )
            .std()
        )

        result[
            f"rolling_min_{window}"
        ] = (
            previous_day
            .rolling(
                window=window,
                min_periods=window
            )
            .min()
        )

        result[
            f"rolling_max_{window}"
        ] = (
            previous_day
            .rolling(
                window=window,
                min_periods=window
            )
            .max()
        )

    return result


# ============================================================
# 19. BUILD FEATURE DATASET FOR EVERY DEPARTMENT
# ============================================================

feature_frames = []

for department in departments:

    department_df = COMPLETE_SERIES[
        COMPLETE_SERIES["department"] == department
    ].copy()

    features = create_forecasting_features(
        department_df
    )

    feature_frames.append(
        features
    )


FEATURE_DATA = pd.concat(
    feature_frames,
    ignore_index=True
)


FEATURE_DATA = FEATURE_DATA.sort_values(
    [
        "department",
        "date"
    ]
).reset_index(drop=True)


FEATURE_DATA.to_csv(
    OUTPUT_DIR / "admission_forecasting_features.csv",
    index=False
)


# ============================================================
# 20. MODEL FEATURE LIST
# ============================================================

FEATURE_COLUMNS = [

    # Calendar
    "year",
    "month",
    "day",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "quarter",

    # Binary calendar
    "is_weekend",
    "is_month_start",
    "is_month_end",
    "is_quarter_start",
    "is_quarter_end",

    # Cyclic
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",

    # Trend
    "trend",

    # Lags
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",

    # Rolling statistics
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_min_7",
    "rolling_max_7",

    "rolling_mean_14",
    "rolling_std_14",
    "rolling_min_14",
    "rolling_max_14",

    "rolling_mean_28",
    "rolling_std_28",
    "rolling_min_28",
    "rolling_max_28"
]


TARGET_COLUMN = "admissions"


# ============================================================
# 21. SAFE MAPE FUNCTION
# ============================================================

def safe_mape(y_true, y_pred):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    denominator = np.abs(y_true)

    valid = denominator > 0

    if valid.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (
                    y_true[valid]
                    - y_pred[valid]
                )
                / denominator[valid]
            )
        )
        * 100
    )


# ============================================================
# 22. FORECAST METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    mae = mean_absolute_error(
        y_true,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            y_pred
        )
    )

    mape = safe_mape(
        y_true,
        y_pred
    )

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE_percent": (
            float(mape)
            if not np.isnan(mape)
            else np.nan
        )
    }


# ============================================================
# 23. SEASONAL NAIVE BASELINE
# ============================================================

def seasonal_naive_predictions(
    history,
    dates,
    seasonal_period=7,
    actual_values=None,
):

    history = list(
        np.asarray(
            history,
            dtype=float
        )
    )

    predictions = []

    if actual_values is not None and len(actual_values) != len(dates):
        raise ValueError("Actual values must align with baseline prediction dates.")

    for index, date in enumerate(dates):

        if len(history) >= seasonal_period:

            prediction = history[
                -seasonal_period
            ]

        elif len(history) > 0:

            prediction = history[-1]

        else:

            prediction = 0

        prediction = max(
            0,
            float(prediction)
        )

        predictions.append(
            prediction
        )

        history.append(
            float(actual_values[index])
            if actual_values is not None else prediction
        )

    return np.array(
        predictions,
        dtype=float
    )


def train_admission_model(train_df, feature_columns, target_column):
    """Select tree complexity on a chronological validation tail, then refit."""
    validation_index = int(len(train_df) * 0.8)
    validation_index = max(30, min(validation_index, len(train_df) - 1))
    fit_df = train_df.iloc[:validation_index]
    validation_df = train_df.iloc[validation_index:]
    validation_scores = {}

    for candidate_name, candidate_config in XGB_CONFIG_CANDIDATES.items():
        model = XGBRegressor(**{**XGB_CONFIG, **candidate_config})
        model.fit(
            fit_df[feature_columns],
            fit_df[target_column].astype(float),
        )
        predictions = np.maximum(
            0,
            model.predict(validation_df[feature_columns]),
        )
        validation_scores[candidate_name] = float(
            mean_absolute_error(
                validation_df[target_column].astype(float),
                predictions,
            )
        )

    selected_name = min(
        validation_scores,
        key=validation_scores.get,
    )
    selected_config = {
        **XGB_CONFIG,
        **XGB_CONFIG_CANDIDATES[selected_name],
    }
    final_model = XGBRegressor(**selected_config)
    final_model.fit(
        train_df[feature_columns],
        train_df[target_column].astype(float),
    )
    return (
        final_model,
        selected_name,
        validation_scores[selected_name],
        selected_config,
    )


# ============================================================
# 24. TRAIN / TEST SPLIT
# ============================================================

def chronological_split(
    df
):

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    usable = df.dropna(
        subset=FEATURE_COLUMNS + [
            TARGET_COLUMN
        ]
    ).copy()

    if len(usable) < 60:

        return None, None

    split_index = int(
        len(usable)
        * (1 - TEST_FRACTION)
    )

    # Guarantee both train and test have data.
    split_index = max(
        1,
        min(
            split_index,
            len(usable) - 1
        )
    )

    train = usable.iloc[
        :split_index
    ].copy()

    test = usable.iloc[
        split_index:
    ].copy()

    return train, test


# ============================================================
# 25. TRAIN DEPARTMENT MODELS
# ============================================================

trained_models = {}

all_test_predictions = []

all_metrics = []

feature_importance_frames = []


print("\n" + "=" * 80)
print("STARTING DEPARTMENT FORECASTING")
print("=" * 80)


for department in departments:

    print(
        f"\nProcessing department: {department}"
    )

    department_features = FEATURE_DATA[
        FEATURE_DATA["department"] == department
    ].copy()

    department_features = (
        department_features
        .sort_values("date")
        .reset_index(drop=True)
    )

    train_df, test_df = chronological_split(
        department_features
    )

    if (
        train_df is None
        or test_df is None
        or len(train_df) < 30
        or len(test_df) < 5
    ):

        print(
            "  Skipped: insufficient usable history."
        )

        continue

    X_train = train_df[
        FEATURE_COLUMNS
    ].copy()

    y_train = train_df[
        TARGET_COLUMN
    ].astype(float)

    X_test = test_df[
        FEATURE_COLUMNS
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].astype(float)

    # --------------------------------------------------------
    # XGBOOST MODEL
    # --------------------------------------------------------

    (
        model,
        selected_config_name,
        validation_mae,
        selected_config,
    ) = train_admission_model(
        train_df,
        FEATURE_COLUMNS,
        TARGET_COLUMN,
    )

    ml_predictions = model.predict(
        X_test
    )

    # Admissions cannot be negative.
    ml_predictions = np.maximum(
        ml_predictions,
        0
    )

    # --------------------------------------------------------
    # BASELINE
    # --------------------------------------------------------

    baseline_predictions = seasonal_naive_predictions(
        history=train_df[
            TARGET_COLUMN
        ].values,
        dates=test_df["date"].values,
        seasonal_period=7,
        actual_values=y_test.values,
    )

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    ml_metrics = calculate_metrics(
        y_test,
        ml_predictions
    )

    baseline_metrics = calculate_metrics(
        y_test,
        baseline_predictions
    )

    ml_metrics["department"] = department
    ml_metrics["model"] = MODEL_NAME
    ml_metrics["selected_config"] = selected_config_name
    ml_metrics["validation_MAE"] = validation_mae
    ml_metrics["test_start"] = str(
        test_df["date"].min().date()
    )
    ml_metrics["test_end"] = str(
        test_df["date"].max().date()
    )
    ml_metrics["train_rows"] = int(
        len(train_df)
    )
    ml_metrics["test_rows"] = int(
        len(test_df)
    )

    baseline_metrics["department"] = department
    baseline_metrics["model"] = (
        "7-Day Seasonal Naive Baseline"
    )
    baseline_metrics["test_start"] = str(
        test_df["date"].min().date()
    )
    baseline_metrics["test_end"] = str(
        test_df["date"].max().date()
    )
    baseline_metrics["train_rows"] = int(
        len(train_df)
    )
    baseline_metrics["test_rows"] = int(
        len(test_df)
    )

    all_metrics.append(
        ml_metrics
    )

    all_metrics.append(
        baseline_metrics
    )

    # --------------------------------------------------------
    # SAVE TEST PREDICTIONS
    # --------------------------------------------------------

    prediction_df = pd.DataFrame({
        "date": test_df["date"].values,
        "department": department,
        "actual_admissions": y_test.values,
        "ml_prediction": ml_predictions,
        "baseline_prediction": baseline_predictions
    })

    prediction_df[
        "ml_absolute_error"
    ] = np.abs(
        prediction_df[
            "actual_admissions"
        ]
        - prediction_df[
            "ml_prediction"
        ]
    )

    prediction_df[
        "baseline_absolute_error"
    ] = np.abs(
        prediction_df[
            "actual_admissions"
        ]
        - prediction_df[
            "baseline_prediction"
        ]
    )

    all_test_predictions.append(
        prediction_df
    )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
        "department": department
    })

    feature_importance_frames.append(
        importance_df
    )

    # --------------------------------------------------------
    # SAVE MODEL IN MEMORY
    # --------------------------------------------------------

    trained_models[department] = {
        "model": model,
        "model_config": selected_config,
        "selected_config": selected_config_name,
        "validation_MAE": validation_mae,
        "last_training_date": str(
            train_df["date"].max().date()
        ),
        "feature_columns": FEATURE_COLUMNS
    }

    # --------------------------------------------------------
    # PRINT PERFORMANCE
    # --------------------------------------------------------

    print(
        f"  Train rows : {len(train_df):,}"
    )

    print(
        f"  Test rows  : {len(test_df):,}"
    )

    print(
        f"  ML MAE     : {ml_metrics['MAE']:.3f}"
    )

    print(
        f"  Selected   : {selected_config_name} "
        f"(validation MAE {validation_mae:.3f})"
    )

    print(
        f"  ML RMSE    : {ml_metrics['RMSE']:.3f}"
    )

    if not np.isnan(
        ml_metrics["MAPE_percent"]
    ):

        print(
            f"  ML MAPE    : "
            f"{ml_metrics['MAPE_percent']:.2f}%"
        )

    print(
        f"  Baseline MAE: "
        f"{baseline_metrics['MAE']:.3f}"
    )


# ============================================================
# 26. VALIDATE MODEL TRAINING
# ============================================================

if not trained_models:

    raise RuntimeError(
        "No department models were successfully trained."
    )


# ============================================================
# 27. SAVE MODELS
# ============================================================

model_path = (
    MODEL_DIR
    / "admission_forecasting_models.joblib"
)

joblib.dump(
    trained_models,
    model_path
)


# ============================================================
# 28. SAVE METRICS
# ============================================================

METRICS_DF = pd.DataFrame(
    all_metrics
)

METRICS_DF.to_csv(
    OUTPUT_DIR / "admission_forecasting_metrics.csv",
    index=False
)


# ============================================================
# 29. SAVE TEST PREDICTIONS
# ============================================================

TEST_PREDICTIONS = pd.concat(
    all_test_predictions,
    ignore_index=True
)

TEST_PREDICTIONS.to_csv(
    OUTPUT_DIR / "admission_test_predictions.csv",
    index=False
)


# ============================================================
# 30. SAVE FEATURE IMPORTANCE
# ============================================================

FEATURE_IMPORTANCE = pd.concat(
    feature_importance_frames,
    ignore_index=True
)

FEATURE_IMPORTANCE.to_csv(
    OUTPUT_DIR / "admission_feature_importance_by_department.csv",
    index=False
)


# ============================================================
# 31. OVERALL FEATURE IMPORTANCE
# ============================================================

OVERALL_IMPORTANCE = (
    FEATURE_IMPORTANCE
    .groupby("feature")["importance"]
    .mean()
    .sort_values(
        ascending=False
    )
    .reset_index()
)

OVERALL_IMPORTANCE.to_csv(
    OUTPUT_DIR / "admission_feature_importance.csv",
    index=False
)


# ============================================================
# 32. CREATE 30-DAY FUTURE FORECAST
# ============================================================

def build_future_feature_row(
    history_df,
    future_date
):

    history_df = history_df.sort_values(
        "date"
    ).reset_index(drop=True)

    values = history_df[
        "admissions"
    ].astype(float).tolist()

    row = {
        "date": future_date
    }

    # Calendar
    row["year"] = future_date.year
    row["month"] = future_date.month
    row["day"] = future_date.day
    row["day_of_week"] = future_date.weekday()
    row["day_of_year"] = future_date.dayofyear
    row["week_of_year"] = int(
        future_date.isocalendar().week
    )
    row["quarter"] = (
        (future_date.month - 1) // 3
    ) + 1

    row["is_weekend"] = int(
        future_date.weekday() >= 5
    )

    row["is_month_start"] = int(
        future_date.is_month_start
    )

    row["is_month_end"] = int(
        future_date.is_month_end
    )

    row["is_quarter_start"] = int(
        future_date.is_quarter_start
    )

    row["is_quarter_end"] = int(
        future_date.is_quarter_end
    )

    # Cyclic
    row["dow_sin"] = np.sin(
        2 * np.pi * future_date.weekday() / 7
    )

    row["dow_cos"] = np.cos(
        2 * np.pi * future_date.weekday() / 7
    )

    row["month_sin"] = np.sin(
        2 * np.pi * future_date.month / 12
    )

    row["month_cos"] = np.cos(
        2 * np.pi * future_date.month / 12
    )

    row["doy_sin"] = np.sin(
        2 * np.pi
        * future_date.dayofyear
        / 365.25
    )

    row["doy_cos"] = np.cos(
        2 * np.pi
        * future_date.dayofyear
        / 365.25
    )

    # Trend
    row["trend"] = len(history_df)

    # Lags
    for lag in LAGS:

        if len(values) >= lag:

            row[f"lag_{lag}"] = (
                values[-lag]
            )

        else:

            row[f"lag_{lag}"] = (
                values[-1]
                if values
                else 0
            )

    # Rolling values
    for window in ROLLING_WINDOWS:

        if len(values) >= window:

            recent_values = np.array(
                values[-window:],
                dtype=float
            )

        else:

            recent_values = np.array(
                values,
                dtype=float
            )

        if len(recent_values) == 0:

            recent_values = np.array(
                [0.0]
            )

        row[
            f"rolling_mean_{window}"
        ] = float(
            np.mean(recent_values)
        )

        row[
            f"rolling_std_{window}"
        ] = float(
            np.std(
                recent_values,
                ddof=1
            )
            if len(recent_values) > 1
            else 0
        )

        row[
            f"rolling_min_{window}"
        ] = float(
            np.min(recent_values)
        )

        row[
            f"rolling_max_{window}"
        ] = float(
            np.max(recent_values)
        )

    return pd.DataFrame(
        [row]
    )


future_forecast_frames = []


print("\n" + "=" * 80)
print("GENERATING 30-DAY FUTURE FORECAST")
print("=" * 80)


for department in sorted(
    trained_models.keys()
):

    model = trained_models[
        department
    ]["model"]

    history = COMPLETE_SERIES[
        COMPLETE_SERIES["department"]
        == department
    ].copy()

    history = history.sort_values(
        "date"
    ).reset_index(drop=True)

    predictions = []

    current_history = history[
        [
            "date",
            "admissions"
        ]
    ].copy()

    last_date = current_history[
        "date"
    ].max()

    for day_number in range(
        1,
        FORECAST_DAYS + 1
    ):

        future_date = (
            last_date
            + pd.Timedelta(
                days=day_number
            )
        )

        future_row = (
            build_future_feature_row(
                current_history,
                future_date
            )
        )

        X_future = future_row[
            FEATURE_COLUMNS
        ].copy()

        prediction = model.predict(
            X_future
        )[0]

        prediction = max(
            0,
            float(prediction)
        )

        predictions.append({
            "date": future_date,
            "department": department,
            "predicted_admissions": prediction
        })

        # Recursive forecasting:
        # prediction becomes part of future history.
        current_history = pd.concat(
            [
                current_history,
                pd.DataFrame({
                    "date": [future_date],
                    "admissions": [prediction]
                })
            ],
            ignore_index=True
        )

    future_department_df = pd.DataFrame(
        predictions
    )

    future_forecast_frames.append(
        future_department_df
    )


FUTURE_FORECAST = pd.concat(
    future_forecast_frames,
    ignore_index=True
)


# ============================================================
# 33. HOSPITAL-WIDE FUTURE FORECAST
# ============================================================

hospital_future_forecast = (
    FUTURE_FORECAST
    .groupby("date", as_index=False)[
        "predicted_admissions"
    ]
    .sum()
)

hospital_future_forecast[
    "department"
] = "All Departments"


hospital_future_forecast = (
    hospital_future_forecast[
        [
            "date",
            "department",
            "predicted_admissions"
        ]
    ]
)


FUTURE_FORECAST_WITH_TOTAL = pd.concat(
    [
        FUTURE_FORECAST,
        hospital_future_forecast
    ],
    ignore_index=True
)


FUTURE_FORECAST_WITH_TOTAL.to_csv(
    OUTPUT_DIR
    / "admission_30_day_forecast.csv",
    index=False
)


# ============================================================
# 34. FUTURE FORECAST SUMMARY
# ============================================================

forecast_summary = (
    FUTURE_FORECAST
    .groupby("department")[
        "predicted_admissions"
    ]
    .agg(
        total_forecast_admissions="sum",
        average_daily_admissions="mean",
        maximum_daily_admissions="max",
        minimum_daily_admissions="min"
    )
    .reset_index()
)


forecast_summary = forecast_summary.sort_values(
    "total_forecast_admissions",
    ascending=False
)


forecast_summary.to_csv(
    OUTPUT_DIR
    / "admission_30_day_forecast_summary.csv",
    index=False
)


# ============================================================
# 35. MODEL PERFORMANCE SUMMARY
# ============================================================

ml_only = METRICS_DF[
    METRICS_DF["model"] == MODEL_NAME
].copy()

baseline_only = METRICS_DF[
    METRICS_DF["model"]
    == "7-Day Seasonal Naive Baseline"
].copy()


overall_ml_metrics = calculate_metrics(
    TEST_PREDICTIONS[
        "actual_admissions"
    ],
    TEST_PREDICTIONS[
        "ml_prediction"
    ]
)


overall_baseline_metrics = calculate_metrics(
    TEST_PREDICTIONS[
        "actual_admissions"
    ],
    TEST_PREDICTIONS[
        "baseline_prediction"
    ]
)


# ============================================================
# 36. BASELINE IMPROVEMENT CALCULATION
# ============================================================

baseline_mae = (
    overall_baseline_metrics["MAE"]
)

ml_mae = (
    overall_ml_metrics["MAE"]
)

if baseline_mae > 0:

    mae_improvement_percent = (
        (
            baseline_mae - ml_mae
        )
        / baseline_mae
        * 100
    )

else:

    mae_improvement_percent = np.nan


baseline_rmse = (
    overall_baseline_metrics["RMSE"]
)

ml_rmse = (
    overall_ml_metrics["RMSE"]
)

if baseline_rmse > 0:

    rmse_improvement_percent = (
        (
            baseline_rmse - ml_rmse
        )
        / baseline_rmse
        * 100
    )

else:

    rmse_improvement_percent = np.nan


# ============================================================
# 37. CHART 1 - HOSPITAL-WIDE HISTORICAL ADMISSIONS
# ============================================================

hospital_history = (
    COMPLETE_SERIES
    .groupby("date", as_index=False)[
        "admissions"
    ]
    .sum()
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    hospital_history["date"],
    hospital_history["admissions"],
    label="Historical admissions"
)

plt.title(
    "CareOps AI - Historical Daily Hospital Admissions"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Admissions"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "01_historical_hospital_admissions.png",
    dpi=150
)

plt.close()


# ============================================================
# 38. CHART 2 - TEST ACTUAL VS PREDICTED
# ============================================================

test_hospital = (
    TEST_PREDICTIONS
    .groupby("date", as_index=False)
    .agg(
        actual_admissions=(
            "actual_admissions",
            "sum"
        ),
        ml_prediction=(
            "ml_prediction",
            "sum"
        ),
        baseline_prediction=(
            "baseline_prediction",
            "sum"
        )
    )
)


plt.figure(
    figsize=(14, 6)
)

plt.plot(
    test_hospital["date"],
    test_hospital["actual_admissions"],
    label="Actual"
)

plt.plot(
    test_hospital["date"],
    test_hospital["ml_prediction"],
    label="ML prediction"
)

plt.plot(
    test_hospital["date"],
    test_hospital["baseline_prediction"],
    label="7-day baseline"
)

plt.title(
    "CareOps AI - Test Period: Actual vs Predicted Admissions"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Admissions"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "02_test_actual_vs_predicted.png",
    dpi=150
)

plt.close()


# ============================================================
# 39. CHART 3 - 30-DAY FUTURE HOSPITAL FORECAST
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    hospital_future_forecast["date"],
    hospital_future_forecast[
        "predicted_admissions"
    ],
    marker="o",
    label="30-day forecast"
)

plt.title(
    "CareOps AI - 30-Day Hospital Admission Forecast"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Predicted admissions"
)

plt.xticks(
    rotation=45
)

plt.legend()

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "03_30_day_hospital_forecast.png",
    dpi=150
)

plt.close()


# ============================================================
# 40. CHART 4 - DEPARTMENT FORECAST
# ============================================================

plt.figure(
    figsize=(15, 8)
)

for department in sorted(
    FUTURE_FORECAST["department"].unique()
):

    department_future = FUTURE_FORECAST[
        FUTURE_FORECAST["department"]
        == department
    ]

    plt.plot(
        department_future["date"],
        department_future[
            "predicted_admissions"
        ],
        label=department
    )


plt.title(
    "CareOps AI - 30-Day Admission Forecast by Department"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Predicted admissions"
)

plt.xticks(
    rotation=45
)

plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "04_department_30_day_forecast.png",
    dpi=150
)

plt.close()


# ============================================================
# 41. CHART 5 - FEATURE IMPORTANCE
# ============================================================

top_features = (
    OVERALL_IMPORTANCE
    .head(15)
    .sort_values(
        "importance"
    )
)


plt.figure(
    figsize=(10, 7)
)

plt.barh(
    top_features["feature"],
    top_features["importance"]
)

plt.title(
    "CareOps AI - Top Admission Forecasting Features"
)

plt.xlabel(
    "Average feature importance"
)

plt.ylabel(
    "Feature"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "05_feature_importance.png",
    dpi=150
)

plt.close()


# ============================================================
# 42. CHART 6 - DEPARTMENT FORECAST TOTALS
# ============================================================

plt.figure(
    figsize=(12, 7)
)

plt.bar(
    forecast_summary["department"],
    forecast_summary[
        "total_forecast_admissions"
    ]
)

plt.title(
    "CareOps AI - Forecasted Admissions Over Next 30 Days"
)

plt.xlabel(
    "Department"
)

plt.ylabel(
    "Forecasted admissions"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    CHART_DIR
    / "06_department_forecast_totals.png",
    dpi=150
)

plt.close()


# ============================================================
# 43. CREATE JSON MODEL SUMMARY
# ============================================================

model_summary = {

    "project": "CareOps AI",

    "module": "Admission Demand Forecasting",

    "source_file": str(
        SOURCE_USED
    ),

    "source_type": SOURCE_TYPE,

    "historical_start_date": str(
        date_min.date()
    ),

    "historical_end_date": str(
        date_max.date()
    ),

    "number_of_departments": int(
        number_departments
    ),

    "number_of_historical_days": int(
        number_days
    ),

    "total_historical_admissions": float(
        total_admissions
    ),

    "model": MODEL_NAME,

    "baseline": (
        "7-Day Seasonal Naive"
    ),

    "forecast_horizon_days": FORECAST_DAYS,

    "test_fraction": TEST_FRACTION,

    "features": FEATURE_COLUMNS,

    "lags": LAGS,

    "rolling_windows": ROLLING_WINDOWS,

    "model_parameters": XGB_CONFIG,

    "model_selection": {
        "method": "Per-department chronological validation-tail MAE",
        "validation_fraction_of_training": 0.2,
        "candidates": XGB_CONFIG_CANDIDATES,
        "selected_by_department": {
            department: {
                "candidate": model_data["selected_config"],
                "validation_MAE": model_data["validation_MAE"],
            }
            for department, model_data in trained_models.items()
        },
        "holdout_used_for_selection": False,
    },

    "overall_ml_metrics": overall_ml_metrics,

    "overall_baseline_metrics": overall_baseline_metrics,

    "MAE_improvement_vs_baseline_percent": (
        float(mae_improvement_percent)
        if not np.isnan(
            mae_improvement_percent
        )
        else None
    ),

    "RMSE_improvement_vs_baseline_percent": (
        float(rmse_improvement_percent)
        if not np.isnan(
            rmse_improvement_percent
        )
        else None
    ),

    "departments_modelled": sorted(
        list(
            trained_models.keys()
        )
    ),

    "generated_at": datetime.now().isoformat(),

    "safety_note": (
        "Operations decision support only. "
        "Not a clinical diagnosis or treatment system."
    )
}


with open(
    OUTPUT_DIR / "model_summary.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        model_summary,
        file,
        indent=4
    )


# ============================================================
# 44. CREATE HUMAN-READABLE REPORT
# ============================================================

report_path = (
    REPORT_DIR
    / "admission_forecasting_report.txt"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "=" * 80 + "\n"
    )

    report.write(
        "CAREOPS AI - ADMISSION DEMAND FORECASTING REPORT\n"
    )

    report.write(
        "=" * 80 + "\n\n"
    )

    report.write(
        "PROJECT PURPOSE\n"
    )

    report.write(
        "CareOps AI is an operations decision-support platform "
        "for healthcare resource planning. This module forecasts "
        "future hospital admission demand to support later "
        "capacity and resource optimization.\n\n"
    )

    report.write(
        "SOURCE DATA\n"
    )

    report.write(
        f"Source file: {SOURCE_USED}\n"
    )

    report.write(
        f"Source type: {SOURCE_TYPE}\n"
    )

    report.write(
        f"Historical start: {date_min.date()}\n"
    )

    report.write(
        f"Historical end: {date_max.date()}\n"
    )

    report.write(
        f"Historical days: {number_days:,}\n"
    )

    report.write(
        f"Departments: {number_departments}\n"
    )

    report.write(
        f"Historical admissions: "
        f"{total_admissions:,.0f}\n\n"
    )

    report.write(
        "FORECASTING APPROACH\n"
    )

    report.write(
        f"Machine-learning model: {MODEL_NAME}\n"
    )

    report.write(
        "Baseline: 7-Day Seasonal Naive\n"
    )

    report.write(
        "Validation: Chronological time-based split\n"
    )

    report.write(
        "Random shuffling: Not used\n"
    )

    report.write(
        "Forecast horizon: "
        f"{FORECAST_DAYS} days\n\n"
    )

    report.write(
        "FEATURES\n"
    )

    report.write(
        "Calendar features:\n"
    )

    report.write(
        " - Year\n"
        " - Month\n"
        " - Day\n"
        " - Day of week\n"
        " - Day of year\n"
        " - Week of year\n"
        " - Quarter\n"
        " - Weekend indicator\n"
        " - Month start/end indicators\n"
        " - Quarter start/end indicators\n"
    )

    report.write(
        "\nSeasonality features:\n"
    )

    report.write(
        " - Day-of-week sine/cosine\n"
        " - Month sine/cosine\n"
        " - Day-of-year sine/cosine\n"
    )

    report.write(
        "\nHistorical lag features:\n"
    )

    for lag in LAGS:

        report.write(
            f" - {lag}-day lag\n"
        )

    report.write(
        "\nRolling features:\n"
    )

    for window in ROLLING_WINDOWS:

        report.write(
            f" - {window}-day mean\n"
            f" - {window}-day standard deviation\n"
            f" - {window}-day minimum\n"
            f" - {window}-day maximum\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "OVERALL MODEL PERFORMANCE\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"ML MAE: "
        f"{overall_ml_metrics['MAE']:.4f}\n"
    )

    report.write(
        f"ML RMSE: "
        f"{overall_ml_metrics['RMSE']:.4f}\n"
    )

    if not np.isnan(
        overall_ml_metrics[
            "MAPE_percent"
        ]
    ):

        report.write(
            f"ML MAPE: "
            f"{overall_ml_metrics['MAPE_percent']:.2f}%\n"
        )

    report.write(
        "\nBaseline performance:\n"
    )

    report.write(
        f"Baseline MAE: "
        f"{overall_baseline_metrics['MAE']:.4f}\n"
    )

    report.write(
        f"Baseline RMSE: "
        f"{overall_baseline_metrics['RMSE']:.4f}\n"
    )

    if not np.isnan(
        overall_baseline_metrics[
            "MAPE_percent"
        ]
    ):

        report.write(
            f"Baseline MAPE: "
            f"{overall_baseline_metrics['MAPE_percent']:.2f}%\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "BASELINE COMPARISON\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    if not np.isnan(
        mae_improvement_percent
    ):

        report.write(
            f"MAE improvement vs baseline: "
            f"{mae_improvement_percent:.2f}%\n"
        )

    else:

        report.write(
            "MAE improvement vs baseline: unavailable\n"
        )

    if not np.isnan(
        rmse_improvement_percent
    ):

        report.write(
            f"RMSE improvement vs baseline: "
            f"{rmse_improvement_percent:.2f}%\n"
        )

    else:

        report.write(
            "RMSE improvement vs baseline: unavailable\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "DEPARTMENT MODEL PERFORMANCE\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    for _, row in ml_only.iterrows():

        report.write(
            f"\nDepartment: "
            f"{row['department']}\n"
        )

        report.write(
            f"MAE: {row['MAE']:.4f}\n"
        )

        report.write(
            f"RMSE: {row['RMSE']:.4f}\n"
        )

        if not pd.isna(
            row["MAPE_percent"]
        ):

            report.write(
                f"MAPE: "
                f"{row['MAPE_percent']:.2f}%\n"
            )

    report.write(
        "\n"
    )

    report.write(
        "30-DAY FORECAST\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"Forecast start: "
        f"{FUTURE_FORECAST['date'].min().date()}\n"
    )

    report.write(
        f"Forecast end: "
        f"{FUTURE_FORECAST['date'].max().date()}\n"
    )

    total_future_admissions = (
        FUTURE_FORECAST[
            "predicted_admissions"
        ].sum()
    )

    report.write(
        f"Total forecasted admissions: "
        f"{total_future_admissions:,.2f}\n\n"
    )

    report.write(
        "Department forecast totals:\n"
    )

    for _, row in forecast_summary.iterrows():

        report.write(
            f" - {row['department']}: "
            f"{row['total_forecast_admissions']:,.2f}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "TOP FORECASTING FEATURES\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    for _, row in (
        OVERALL_IMPORTANCE.head(15)
        .iterrows()
    ):

        report.write(
            f" - {row['feature']}: "
            f"{row['importance']:.6f}\n"
        )

    report.write(
        "\n"
    )

    report.write(
        "OUTPUT FILES\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        f"Data catalog:\n"
        f"  {OUTPUT_DIR / 'data_catalog.csv'}\n\n"
    )

    report.write(
        f"Daily admissions used:\n"
        f"  {OUTPUT_DIR / 'daily_admissions_used.csv'}\n\n"
    )

    report.write(
        f"Forecasting features:\n"
        f"  {OUTPUT_DIR / 'admission_forecasting_features.csv'}\n\n"
    )

    report.write(
        f"Metrics:\n"
        f"  {OUTPUT_DIR / 'admission_forecasting_metrics.csv'}\n\n"
    )

    report.write(
        f"Test predictions:\n"
        f"  {OUTPUT_DIR / 'admission_test_predictions.csv'}\n\n"
    )

    report.write(
        f"30-day forecast:\n"
        f"  {OUTPUT_DIR / 'admission_30_day_forecast.csv'}\n\n"
    )

    report.write(
        f"Forecast summary:\n"
        f"  {OUTPUT_DIR / 'admission_30_day_forecast_summary.csv'}\n\n"
    )

    report.write(
        f"Feature importance:\n"
        f"  {OUTPUT_DIR / 'admission_feature_importance.csv'}\n\n"
    )

    report.write(
        f"Trained model:\n"
        f"  {model_path}\n\n"
    )

    report.write(
        f"Model summary:\n"
        f"  {OUTPUT_DIR / 'model_summary.json'}\n\n"
    )

    report.write(
        "SAFETY / PORTFOLIO POSITIONING\n"
    )

    report.write(
        "-" * 80 + "\n"
    )

    report.write(
        "This forecasting model is an operations decision-support "
        "component. It should not be represented as an autonomous "
        "clinical diagnosis or treatment system. The public portfolio "
        "implementation should use synthetic/de-identified data. "
        "Forecasts should support qualified healthcare operations "
        "professionals rather than silently automate decisions.\n"
    )


# ============================================================
# 45. SAVE DAILY HOSPITAL FORECAST
# ============================================================

hospital_future_forecast.to_csv(
    OUTPUT_DIR
    / "hospital_30_day_admission_forecast.csv",
    index=False
)


# ============================================================
# 46. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 80)
print("ADMISSION FORECASTING COMPLETE")
print("=" * 80)

print(
    f"\nModels successfully trained: "
    f"{len(trained_models)}"
)

print(
    f"Forecast horizon: "
    f"{FORECAST_DAYS} days"
)

print(
    f"\nOverall ML MAE: "
    f"{overall_ml_metrics['MAE']:.3f}"
)

print(
    f"Overall ML RMSE: "
    f"{overall_ml_metrics['RMSE']:.3f}"
)

if not np.isnan(
    overall_ml_metrics["MAPE_percent"]
):

    print(
        f"Overall ML MAPE: "
        f"{overall_ml_metrics['MAPE_percent']:.2f}%"
    )

print(
    f"\nOverall baseline MAE: "
    f"{overall_baseline_metrics['MAE']:.3f}"
)

print(
    f"Overall baseline RMSE: "
    f"{overall_baseline_metrics['RMSE']:.3f}"
)

if not np.isnan(
    mae_improvement_percent
):

    print(
        f"\nMAE difference vs baseline: "
        f"{mae_improvement_percent:.2f}%"
    )

print(
    f"\n30-day predicted hospital admissions: "
    f"{hospital_future_forecast['predicted_admissions'].sum():,.2f}"
)

print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)

print(
    "\nReport:"
)

print(
    report_path
)

print(
    "\nCharts:"
)

print(
    CHART_DIR
)

print(
    "\nModel:"
)

print(
    model_path
)

print("\n" + "=" * 80)
print("NEXT STAGE: OCCUPANCY FORECASTING")
print("=" * 80)

print(
    "\nAdmission forecasting has been completed as the "
    "first forecasting component of CareOps AI."
)

print(
    "The next major forecasting module can use admission "
    "demand together with discharges, LOS and bed capacity "
    "to build the occupancy forecast."
)

print("\n")
"""
========================================================================
CAREOPS AI — LENGTH-OF-STAY PREDICTION
========================================================================

Purpose
-------
Predict hospital patient Length of Stay (LOS) using admission-time
information.

This module:

1. Discovers existing project CSV files.
2. Loads admissions and patients data.
3. Merges patient information with admission information.
4. Builds a safe LOS target.
5. Creates admission-time features.
6. Prevents target leakage.
7. Treats severity as CATEGORICAL.
8. Uses chronological train/test splitting.
9. Trains a baseline mean predictor.
10. Trains an optimized XGBoost regression model.
11. Evaluates MAE, RMSE, R² and MAPE.
12. Generates predictions.
13. Generates feature importance.
14. Saves the trained model.
15. Saves prediction CSV.
16. Saves metrics and metadata.
17. Saves data-quality information.
18. Saves training summary.
19. Saves charts.
20. Saves a human-readable report.

Designed for:
    Windows
    Python 3.14+
    pandas
    numpy
    scikit-learn
    xgboost
    matplotlib
    joblib

Project root:
    C:\\Users\\krish\\Documents\\CareOps-AI

Output directory:
    ml\\los_prediction
========================================================================
"""

from __future__ import annotations

# ======================================================================
# IMPORTS
# ======================================================================

import os
import sys
import json
import math
import time
import warnings
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib

# Use a non-interactive backend so charts work reliably on Windows
# without requiring a GUI window.
matplotlib.use("Agg")

import matplotlib.pyplot as plt

import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scenario_config import drop_warmup


warnings.filterwarnings("ignore")


# ======================================================================
# CONFIGURATION
# ======================================================================

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "ml" / "los_prediction"

SCENARIO_DATA_DIR = os.environ.get("CAREOPS_DATA_DIR")
DATA_DIRECTORIES = (
    [Path(SCENARIO_DATA_DIR)]
    if SCENARIO_DATA_DIR
    else [
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data",
        PROJECT_ROOT / "data" / "synthetic",
    ]
)

ADMISSIONS_FILE = "admissions.csv"
PATIENTS_FILE = "patients.csv"

MODEL_FILE = OUTPUT_DIR / "los_xgboost_model.joblib"
PREDICTIONS_FILE = OUTPUT_DIR / "los_predictions.csv"
FEATURE_IMPORTANCE_FILE = OUTPUT_DIR / "los_feature_importance.csv"
METRICS_FILE = OUTPUT_DIR / "los_metrics.json"
METADATA_FILE = OUTPUT_DIR / "los_model_metadata.json"
DATA_QUALITY_FILE = OUTPUT_DIR / "los_data_quality.csv"
TRAINING_SUMMARY_FILE = OUTPUT_DIR / "los_training_summary.csv"

ACTUAL_VS_PREDICTED_PLOT = OUTPUT_DIR / "los_actual_vs_predicted.png"
RESIDUAL_PLOT = OUTPUT_DIR / "los_residuals.png"
FEATURE_IMPORTANCE_PLOT = OUTPUT_DIR / "los_feature_importance.png"

REPORT_FILE = OUTPUT_DIR / "LOS_MODEL_REPORT.txt"


# ======================================================================
# MODEL CONFIGURATION
# ======================================================================

TEST_SIZE = 0.20

XGBOOST_CONFIG = {
    "n_estimators": 250,
    "max_depth": 7,
    "learning_rate": 0.04,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "min_child_weight": 2,
    "reg_lambda": 2.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "random_state": SEED,
    "n_jobs": max(1, (os.cpu_count() or 2) - 1),
    "verbosity": 0,
}


# ======================================================================
# DISPLAY HELPERS
# ======================================================================

def print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def print_step(number: int, total: int, title: str) -> None:
    print(f"\n[{number}/{total}] {title}")


# ======================================================================
# JSON SERIALIZATION
# ======================================================================

def make_json_serializable(value: Any) -> Any:
    """
    Convert pandas, NumPy and Python date/time objects into values that
    JSON can safely serialize.

    This function specifically fixes the previous error:

        TypeError:
        Object of type Timestamp is not JSON serializable
    """

    # ------------------------------------------------------------------
    # None
    # ------------------------------------------------------------------
    if value is None:
        return None

    # ------------------------------------------------------------------
    # pandas Timestamp
    # ------------------------------------------------------------------
    if isinstance(value, pd.Timestamp):
        if pd.isna(value):
            return None
        return value.isoformat()

    # ------------------------------------------------------------------
    # Python datetime/date/time
    # ------------------------------------------------------------------
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()

    # ------------------------------------------------------------------
    # pandas Timedelta
    # ------------------------------------------------------------------
    if isinstance(value, pd.Timedelta):
        if pd.isna(value):
            return None
        return str(value)

    # ------------------------------------------------------------------
    # NumPy integer
    # ------------------------------------------------------------------
    if isinstance(value, np.integer):
        return int(value)

    # ------------------------------------------------------------------
    # NumPy floating point
    # ------------------------------------------------------------------
    if isinstance(value, np.floating):
        if not np.isfinite(value):
            return None
        return float(value)

    # ------------------------------------------------------------------
    # NumPy boolean
    # ------------------------------------------------------------------
    if isinstance(value, np.bool_):
        return bool(value)

    # ------------------------------------------------------------------
    # NumPy array
    # ------------------------------------------------------------------
    if isinstance(value, np.ndarray):
        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    # ------------------------------------------------------------------
    # pandas Series
    # ------------------------------------------------------------------
    if isinstance(value, pd.Series):
        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    # ------------------------------------------------------------------
    # pandas Index
    # ------------------------------------------------------------------
    if isinstance(value, pd.Index):
        return [
            make_json_serializable(item)
            for item in value.tolist()
        ]

    # ------------------------------------------------------------------
    # Dictionary
    # ------------------------------------------------------------------
    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(item)
            for key, item in value.items()
        }

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------
    if isinstance(value, list):
        return [
            make_json_serializable(item)
            for item in value
        ]

    # ------------------------------------------------------------------
    # Tuple
    # ------------------------------------------------------------------
    if isinstance(value, tuple):
        return [
            make_json_serializable(item)
            for item in value
        ]

    # ------------------------------------------------------------------
    # pandas missing values
    # ------------------------------------------------------------------
    try:
        missing = pd.isna(value)

        if isinstance(missing, (bool, np.bool_)) and missing:
            return None

    except (TypeError, ValueError):
        pass

    # ------------------------------------------------------------------
    # Normal Python primitives
    # ------------------------------------------------------------------
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float):
            if not math.isfinite(value):
                return None

        return value

    # ------------------------------------------------------------------
    # Final defensive fallback
    # ------------------------------------------------------------------
    return str(value)


def safe_timestamp(value: Any) -> Optional[str]:
    """
    Safely convert date/time values to ISO strings.
    """

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()

    return str(value)


def save_json(data: Any, path: Path) -> None:
    """
    Save JSON safely.

    The default=str fallback is intentionally retained as a final
    defensive layer in case an unexpected object reaches this function.
    """

    safe_data = make_json_serializable(data)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            safe_data,
            file,
            indent=4,
            ensure_ascii=False,
            default=str,
            allow_nan=False,
        )


# ======================================================================
# COLUMN HELPERS
# ======================================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names while preserving readability.
    """

    df = df.copy()

    normalized = []

    for column in df.columns:
        name = str(column).strip().lower()

        name = name.replace(" ", "_")
        name = name.replace("-", "_")
        name = name.replace("/", "_")

        while "__" in name:
            name = name.replace("__", "_")

        normalized.append(name)

    df.columns = normalized

    return df


def find_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """
    Find the first matching column from a list of candidates.
    """

    columns = set(df.columns)

    for candidate in candidates:
        candidate = candidate.lower()

        if candidate in columns:
            return candidate

    return None


def find_csv_file(filename: str) -> Optional[Path]:
    """
    Search known project data directories.
    """

    # First preference: processed data
    for directory in DATA_DIRECTORIES:
        candidate = directory / filename

        if candidate.exists():
            return candidate

    # Second preference: recursive search
    for directory in DATA_DIRECTORIES:
        if not directory.exists():
            continue

        matches = list(directory.rglob(filename))

        if matches:
            return matches[0]

    return None


# ======================================================================
# DATA LOADING
# ======================================================================

def load_csv(path: Path) -> pd.DataFrame:
    """
    Load CSV with defensive parsing.
    """

    try:
        df = pd.read_csv(path)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    df = normalize_columns(df)

    return df


# ======================================================================
# DATETIME DETECTION
# ======================================================================

def convert_datetime_column(
    df: pd.DataFrame,
    column: str,
) -> pd.Series:
    """
    Convert a column to pandas datetime safely.
    """

    return pd.to_datetime(
        df[column],
        errors="coerce",
    )


def find_admission_datetime(
    df: pd.DataFrame,
) -> Optional[str]:
    """
    Identify the admission datetime/date column.
    """

    candidates = [
        "admission_datetime",
        "admission_date",
        "admit_datetime",
        "admit_date",
        "admitted_datetime",
        "admitted_date",
        "date_of_admission",
        "admission_time",
    ]

    return find_column(df, candidates)


def find_discharge_datetime(
    df: pd.DataFrame,
) -> Optional[str]:
    """
    Identify discharge datetime/date column.
    """

    candidates = [
        "discharge_datetime",
        "discharge_date",
        "discharged_datetime",
        "discharged_date",
        "date_of_discharge",
    ]

    return find_column(df, candidates)


def filter_completed_admissions(
    admissions: pd.DataFrame,
    as_of: pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """Exclude encounters whose observed discharge falls after the run cutoff."""
    if "stay_status" in admissions.columns:
        statuses = admissions["stay_status"].astype(str).str.casefold()
        unknown_statuses = set(statuses) - {"completed", "in_progress"}
        if unknown_statuses:
            raise ValueError(
                f"Admissions contain unknown stay_status values: "
                f"{sorted(unknown_statuses)}"
            )
        completed = statuses.eq("completed")
    else:
        discharge_column = find_discharge_datetime(admissions)
        if discharge_column is None:
            completed = pd.Series(True, index=admissions.index)
        else:
            discharge_dates = pd.to_datetime(
                admissions[discharge_column], errors="coerce"
            )
            completed = discharge_dates.notna() & (
                discharge_dates.dt.normalize() <= as_of.normalize()
            )

    filtered = admissions.loc[completed].copy()
    return filtered, int((~completed).sum())


# ======================================================================
# ID DETECTION
# ======================================================================

def find_patient_id(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "patient_id",
        "patientid",
        "patient",
        "id",
    ]

    return find_column(df, candidates)


def find_admission_id(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "admission_id",
        "admissionid",
        "encounter_id",
        "encounterid",
    ]

    return find_column(df, candidates)


# ======================================================================
# LOS TARGET DETECTION
# ======================================================================

def find_existing_los_column(
    df: pd.DataFrame,
) -> Optional[str]:
    candidates = [
        "length_of_stay",
        "length_of_stay_days",
        "los",
        "los_days",
        "length_stay",
        "stay_length",
        "hospital_los",
        "hospital_los_days",
    ]

    return find_column(df, candidates)


def build_los_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, str]:
    """
    Build LOS target.

    Priority:
        1. Existing LOS column.
        2. Calculate LOS from admission/discharge dates.
    """

    df = df.copy()

    existing_los = find_existing_los_column(df)

    if existing_los is not None:

        df["los_target"] = pd.to_numeric(
            df[existing_los],
            errors="coerce",
        )

        method = "existing_los_column"

    else:

        admission_column = find_admission_datetime(df)
        discharge_column = find_discharge_datetime(df)

        if admission_column is None:
            raise ValueError(
                "Could not find an admission date/datetime column."
            )

        if discharge_column is None:
            raise ValueError(
                "No LOS column exists and no discharge date/datetime "
                "column was found."
            )

        admission_dt = convert_datetime_column(
            df,
            admission_column,
        )

        discharge_dt = convert_datetime_column(
            df,
            discharge_column,
        )

        df["los_target"] = (
            discharge_dt - admission_dt
        ).dt.total_seconds() / 86400.0

        method = "calculated_from_admission_and_discharge_dates"

    # Remove impossible LOS values.
    df.loc[
        ~np.isfinite(
            pd.to_numeric(
                df["los_target"],
                errors="coerce",
            )
        ),
        "los_target",
    ] = np.nan

    # Hospital stay cannot be negative.
    df.loc[
        df["los_target"] < 0,
        "los_target",
    ] = np.nan

    return df, method


# ======================================================================
# MERGING
# ======================================================================

def merge_admissions_patients(
    admissions: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge admissions with patient information.
    """

    admission_patient_id = find_patient_id(admissions)
    patient_patient_id = find_patient_id(patients)

    if admission_patient_id is None:
        raise ValueError(
            "Could not find patient ID in admissions.csv."
        )

    if patient_patient_id is None:
        raise ValueError(
            "Could not find patient ID in patients.csv."
        )

    patients_copy = patients.copy()

    # Avoid duplicate columns except patient ID.
    duplicate_columns = [
        column
        for column in patients_copy.columns
        if column in admissions.columns
        and column != patient_patient_id
    ]

    patients_copy = patients_copy.drop(
        columns=duplicate_columns,
        errors="ignore",
    )

    merged = admissions.merge(
        patients_copy,
        left_on=admission_patient_id,
        right_on=patient_patient_id,
        how="left",
        suffixes=("", "_patient"),
    )

    return merged


# ======================================================================
# FEATURE ENGINEERING
# ======================================================================

def build_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str], List[str], str]:
    """
    Create admission-time features.

    IMPORTANT:
    Discharge information and LOS are not used as predictors.

    severity is intentionally treated as CATEGORICAL.
    """

    df = df.copy()

    admission_column = find_admission_datetime(df)

    if admission_column is None:
        raise ValueError(
            "Could not identify admission date/datetime."
        )

    df["admission_datetime_model"] = pd.to_datetime(
        df[admission_column],
        errors="coerce",
    )

    # --------------------------------------------------------------
    # Standardize commonly expected columns
    # --------------------------------------------------------------

    age_column = find_column(
        df,
        [
            "age",
            "patient_age",
            "age_years",
        ],
    )

    department_column = find_column(
        df,
        [
            "department",
            "dept",
            "service",
            "department_name",
        ],
    )

    diagnosis_column = find_column(
        df,
        [
            "diagnosis_category",
            "diagnosis",
            "diagnosis_group",
            "diagnosis_type",
            "condition",
        ],
    )

    severity_column = find_column(
        df,
        [
            "severity",
            "acuity",
            "severity_level",
            "patient_severity",
        ],
    )

    admission_type_column = find_column(
        df,
        [
            "admission_type",
            "admit_type",
            "admission_source",
            "encounter_type",
        ],
    )

    gender_column = find_column(
        df,
        [
            "gender",
            "sex",
        ],
    )

    icu_column = find_column(
        df,
        [
            "icu_required",
            "icu",
            "requires_icu",
            "icu_flag",
        ],
    )

    procedure_column = find_column(
        df,
        [
            "procedure_indicator",
            "procedure",
            "procedure_flag",
            "had_procedure",
            "surgery",
        ],
    )

    comorbidity_column = find_column(
        df,
        [
            "comorbidity_count",
            "comorbidities",
            "comorbidity",
            "num_comorbidities",
        ],
    )

    # --------------------------------------------------------------
    # Create standardized numeric columns
    # --------------------------------------------------------------

    if age_column is not None:
        df["age_model"] = pd.to_numeric(
            df[age_column],
            errors="coerce",
        )
    else:
        df["age_model"] = np.nan

    if icu_column is not None:
        df["icu_required_model"] = pd.to_numeric(
            df[icu_column],
            errors="coerce",
        )
    else:
        df["icu_required_model"] = 0

    if procedure_column is not None:

        procedure_numeric = pd.to_numeric(
            df[procedure_column],
            errors="coerce",
        )

        # If conversion produced mostly NaN, encode text values.
        if procedure_numeric.notna().sum() == 0:

            procedure_text = (
                df[procedure_column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            positive_values = {
                "yes",
                "true",
                "1",
                "y",
                "surgery",
                "performed",
            }

            procedure_numeric = procedure_text.isin(
                positive_values
            ).astype(int)

        df["procedure_indicator_model"] = procedure_numeric

    else:
        df["procedure_indicator_model"] = 0

    if comorbidity_column is not None:
        df["comorbidity_count_model"] = pd.to_numeric(
            df[comorbidity_column],
            errors="coerce",
        )
    else:
        df["comorbidity_count_model"] = 0

    # --------------------------------------------------------------
    # Calendar features
    # --------------------------------------------------------------

    admission_dt = df["admission_datetime_model"]

    df["admission_year"] = admission_dt.dt.year
    df["admission_month"] = admission_dt.dt.month
    df["admission_day"] = admission_dt.dt.day
    df["admission_dayofweek"] = admission_dt.dt.dayofweek
    df["admission_weekofyear"] = (
        admission_dt.dt.isocalendar().week.astype("float")
    )
    df["admission_is_weekend"] = (
        admission_dt.dt.dayofweek >= 5
    ).astype(int)

    # --------------------------------------------------------------
    # Standardize categorical fields
    # --------------------------------------------------------------

    if department_column is not None:
        df["department_model"] = (
            df[department_column]
            .astype("string")
            .str.strip()
        )
    else:
        df["department_model"] = "Unknown"

    if diagnosis_column is not None:
        df["diagnosis_category_model"] = (
            df[diagnosis_column]
            .astype("string")
            .str.strip()
        )
    else:
        df["diagnosis_category_model"] = "Unknown"

    if severity_column is not None:
        df["severity_model"] = (
            df[severity_column]
            .astype("string")
            .str.strip()
        )
    else:
        df["severity_model"] = "Unknown"

    if admission_type_column is not None:
        df["admission_type_model"] = (
            df[admission_type_column]
            .astype("string")
            .str.strip()
        )
    else:
        df["admission_type_model"] = "Unknown"

    if gender_column is not None:
        df["gender_model"] = (
            df[gender_column]
            .astype("string")
            .str.strip()
        )
    else:
        df["gender_model"] = "Unknown"

    # --------------------------------------------------------------
    # Final feature lists
    # --------------------------------------------------------------

    numeric_features = [
        "age_model",
        "icu_required_model",
        "procedure_indicator_model",
        "comorbidity_count_model",
        "admission_year",
        "admission_month",
        "admission_day",
        "admission_dayofweek",
        "admission_weekofyear",
        "admission_is_weekend",
    ]

    categorical_features = [
        "department_model",
        "diagnosis_category_model",
        "severity_model",
        "admission_type_model",
        "gender_model",
    ]

    return (
        df,
        numeric_features,
        categorical_features,
        admission_column,
    )


# ======================================================================
# DATA QUALITY
# ======================================================================

def build_data_quality_report(
    df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    """
    Build data-quality report for model features.
    """

    rows = []

    columns_to_check = (
        numeric_features
        + categorical_features
        + ["los_target"]
    )

    for column in columns_to_check:

        if column not in df.columns:
            continue

        series = df[column]

        missing_count = int(series.isna().sum())

        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows": int(len(series)),
                "missing_count": missing_count,
                "missing_percent": (
                    missing_count / len(series) * 100
                    if len(series) > 0
                    else 0
                ),
                "unique_values": int(series.nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows)


# ======================================================================
# TRAIN/TEST SPLIT
# ======================================================================

def chronological_split(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform chronological 80/20 split.

    This is preferable for operational prediction because it simulates
    training on historical admissions and predicting future admissions.
    """

    df = df.sort_values(
        "admission_datetime_model"
    ).reset_index(drop=True)

    split_index = int(
        len(df) * (1 - TEST_SIZE)
    )

    if split_index <= 0 or split_index >= len(df):
        raise ValueError(
            "Unable to create a valid chronological train/test split."
        )

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


# ======================================================================
# PREPROCESSING
# ======================================================================

def build_preprocessor(
    numeric_features: List[str],
    categorical_features: List[str],
) -> ColumnTransformer:
    """
    Build preprocessing pipeline.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            )
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ======================================================================
# METRICS
# ======================================================================

def calculate_mape(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """
    Calculate MAPE while avoiding division by zero.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mask = (
        np.isfinite(actual)
        & np.isfinite(predicted)
        & (np.abs(actual) > 1e-8)
    )

    if not mask.any():
        return float("nan")

    return float(
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


def calculate_metrics(
    actual: pd.Series,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate standard regression metrics.
    """

    actual_values = np.asarray(
        actual,
        dtype=float,
    )

    predicted_values = np.asarray(
        predicted,
        dtype=float,
    )

    mae = mean_absolute_error(
        actual_values,
        predicted_values,
    )

    rmse = math.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values,
        )
    )

    r2 = r2_score(
        actual_values,
        predicted_values,
    )

    mape = calculate_mape(
        actual_values,
        predicted_values,
    )

    return {
        "mae_days": float(mae),
        "rmse_days": float(rmse),
        "r2": float(r2),
        "mape_percent": float(mape)
        if np.isfinite(mape)
        else None,
    }


# ======================================================================
# FEATURE IMPORTANCE
# ======================================================================

def get_feature_names(
    preprocessor: ColumnTransformer,
) -> np.ndarray:
    """
    Get transformed feature names.
    """

    try:
        return preprocessor.get_feature_names_out()

    except Exception:
        # Defensive fallback for unexpected sklearn behavior.
        names = []

        for transformer_name, transformer, columns in (
            preprocessor.transformers_
        ):
            if transformer_name == "remainder":
                continue

            if hasattr(
                transformer,
                "get_feature_names_out",
            ):
                try:
                    transformed_names = (
                        transformer.get_feature_names_out(
                            columns
                        )
                    )
                    names.extend(
                        transformed_names.tolist()
                    )
                    continue
                except Exception:
                    pass

            names.extend(
                [str(column) for column in columns]
            )

        return np.asarray(names)


def build_feature_importance(
    pipeline: Pipeline,
) -> pd.DataFrame:
    """
    Extract XGBoost feature importance.
    """

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    model = pipeline.named_steps[
        "model"
    ]

    feature_names = get_feature_names(
        preprocessor
    )

    importances = model.feature_importances_

    if len(feature_names) != len(importances):

        feature_names = np.array(
            [
                f"feature_{i}"
                for i in range(
                    len(importances)
                )
            ]
        )

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    result = result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(drop=True)

    return result


# ======================================================================
# PREDICTION OUTPUT
# ======================================================================

def create_prediction_output(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Create detailed prediction output.
    """

    result = pd.DataFrame(index=test_df.index)

    # Admission ID
    admission_id_column = find_admission_id(
        test_df
    )

    if admission_id_column is not None:
        result["admission_id"] = (
            test_df[admission_id_column]
            .values
        )

    # Patient ID
    patient_id_column = find_patient_id(
        test_df
    )

    if patient_id_column is not None:
        result["patient_id"] = (
            test_df[patient_id_column]
            .values
        )

    # Admission date
    result["admission_datetime"] = (
        test_df[
            "admission_datetime_model"
        ].values
    )

    # Optional descriptive fields
    optional_columns = [
        (
            "department",
            "department_model",
        ),
        (
            "severity",
            "severity_model",
        ),
        (
            "admission_type",
            "admission_type_model",
        ),
        (
            "gender",
            "gender_model",
        ),
    ]

    for output_name, source_column in optional_columns:

        if source_column in test_df.columns:
            result[output_name] = (
                test_df[source_column]
                .astype(str)
                .values
            )

    # Actual/predicted LOS
    actual = test_df[
        "los_target"
    ].astype(float).values

    predicted = np.maximum(
        np.asarray(predictions, dtype=float),
        0,
    )

    result["actual_los_days"] = actual
    result["predicted_los_days"] = predicted

    result["prediction_error_days"] = (
        predicted - actual
    )

    result["absolute_error_days"] = (
        np.abs(
            predicted - actual
        )
    )

    # Estimated discharge date.
    admission_datetime = pd.to_datetime(
        test_df[
            "admission_datetime_model"
        ],
        errors="coerce",
    )

    result["predicted_discharge_datetime"] = (
        admission_datetime
        + pd.to_timedelta(
            predicted,
            unit="D",
        )
    ).values

    return result.reset_index(drop=True)


# ======================================================================
# PLOTS
# ======================================================================

def save_actual_vs_predicted_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    path: Path,
) -> None:
    """
    Save actual vs predicted LOS chart.
    """

    plt.figure(figsize=(9, 7))

    plt.scatter(
        actual,
        predicted,
        alpha=0.20,
        s=10,
    )

    min_value = min(
        float(np.min(actual)),
        float(np.min(predicted)),
    )

    max_value = max(
        float(np.max(actual)),
        float(np.max(predicted)),
    )

    plt.plot(
        [min_value, max_value],
        [min_value, max_value],
        linewidth=2,
    )

    plt.xlabel("Actual LOS (days)")
    plt.ylabel("Predicted LOS (days)")
    plt.title(
        "CareOps-AI — Actual vs Predicted Length of Stay"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()
    plt.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_residual_plot(
    actual: np.ndarray,
    predicted: np.ndarray,
    path: Path,
) -> None:
    """
    Save residual plot.
    """

    residuals = predicted - actual

    plt.figure(figsize=(9, 7))

    plt.scatter(
        predicted,
        residuals,
        alpha=0.20,
        s=10,
    )

    plt.axhline(
        0,
        linewidth=2,
    )

    plt.xlabel(
        "Predicted LOS (days)"
    )

    plt.ylabel(
        "Residual (Predicted - Actual)"
    )

    plt.title(
        "CareOps-AI — LOS Prediction Residuals"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


def save_feature_importance_plot(
    feature_importance: pd.DataFrame,
    path: Path,
    top_n: int = 20,
) -> None:
    """
    Save top feature importance chart.
    """

    top = (
        feature_importance
        .head(top_n)
        .sort_values(
            "importance",
            ascending=True,
        )
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top["feature"].astype(str),
        top["importance"],
    )

    plt.xlabel(
        "Feature Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "CareOps-AI — Top LOS Prediction Features"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=160,
        bbox_inches="tight",
    )

    plt.close()


# ======================================================================
# METADATA
# ======================================================================

def build_model_metadata(
    admissions_path: Path,
    patients_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    numeric_features: List[str],
    categorical_features: List[str],
    target_method: str,
    baseline_metrics: Dict[str, Any],
    model_metrics: Dict[str, Any],
    training_seconds: float,
) -> Dict[str, Any]:
    """
    Build model metadata.

    All datetime values are explicitly converted to strings here.
    """

    train_start = None
    train_end = None
    test_start = None
    test_end = None

    if (
        "admission_datetime_model"
        in train_df.columns
        and len(train_df) > 0
    ):
        train_start = safe_timestamp(
            train_df[
                "admission_datetime_model"
            ].min()
        )

        train_end = safe_timestamp(
            train_df[
                "admission_datetime_model"
            ].max()
        )

    if (
        "admission_datetime_model"
        in test_df.columns
        and len(test_df) > 0
    ):
        test_start = safe_timestamp(
            test_df[
                "admission_datetime_model"
            ].min()
        )

        test_end = safe_timestamp(
            test_df[
                "admission_datetime_model"
            ].max()
        )

    metadata = {
        "project": "CareOps-AI",

        "module": "Length-of-Stay Prediction",

        "model_type": "XGBRegressor",

        "random_seed": SEED,

        "target": {
            "name": "los_target",
            "unit": "days",
            "construction_method": target_method,
        },

        "data_sources": {
            "admissions_file": str(
                admissions_path
            ),
            "patients_file": str(
                patients_path
            ),
        },

        "training": {
            "rows": int(len(train_df)),
            "date_start": train_start,
            "date_end": train_end,
        },

        "testing": {
            "rows": int(len(test_df)),
            "date_start": test_start,
            "date_end": test_end,
        },

        "split": {
            "method": "chronological",
            "test_size": TEST_SIZE,
        },

        "features": {
            "numeric": numeric_features,
            "categorical": categorical_features,
            "severity_handling": "categorical",
        },

        "xgboost": XGBOOST_CONFIG,

        "baseline_metrics": baseline_metrics,

        "xgboost_metrics": model_metrics,

        "training_time_seconds": float(
            training_seconds
        ),

        "target_leakage_prevention": {
            "discharge_date_used_as_feature": False,
            "los_used_as_feature": False,
            "chronological_split": True,
        },

        "generated_at": dt.datetime.now().isoformat(),
    }

    return make_json_serializable(
        metadata
    )


# ======================================================================
# TRAINING SUMMARY
# ======================================================================

def build_training_summary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_metrics: Dict[str, Any],
    model_metrics: Dict[str, Any],
    training_seconds: float,
) -> pd.DataFrame:
    """
    Build simple training summary table.
    """

    rows = [
        {
            "item": "total_rows",
            "value": int(
                len(train_df) + len(test_df)
            ),
        },
        {
            "item": "training_rows",
            "value": int(
                len(train_df)
            ),
        },
        {
            "item": "testing_rows",
            "value": int(
                len(test_df)
            ),
        },
        {
            "item": "baseline_mae_days",
            "value": baseline_metrics[
                "mae_days"
            ],
        },
        {
            "item": "baseline_rmse_days",
            "value": baseline_metrics[
                "rmse_days"
            ],
        },
        {
            "item": "baseline_r2",
            "value": baseline_metrics[
                "r2"
            ],
        },
        {
            "item": "model_mae_days",
            "value": model_metrics[
                "mae_days"
            ],
        },
        {
            "item": "model_rmse_days",
            "value": model_metrics[
                "rmse_days"
            ],
        },
        {
            "item": "model_r2",
            "value": model_metrics[
                "r2"
            ],
        },
        {
            "item": "model_mape_percent",
            "value": model_metrics[
                "mape_percent"
            ],
        },
        {
            "item": "training_time_seconds",
            "value": training_seconds,
        },
    ]

    return pd.DataFrame(rows)


# ======================================================================
# HUMAN-READABLE REPORT
# ======================================================================

def write_text_report(
    path: Path,
    admissions_path: Path,
    patients_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    baseline_metrics: Dict[str, Any],
    model_metrics: Dict[str, Any],
    feature_importance: pd.DataFrame,
    training_seconds: float,
) -> None:
    """
    Write human-readable LOS model report.
    """

    top_features = feature_importance.head(15)

    lines = []

    lines.append("=" * 72)
    lines.append(
        "CAREOPS AI — LENGTH-OF-STAY MODEL REPORT"
    )
    lines.append("=" * 72)
    lines.append("")

    lines.append(
        f"Generated: {dt.datetime.now().isoformat()}"
    )

    lines.append("")

    lines.append("DATA SOURCES")
    lines.append("-" * 72)
    lines.append(
        f"Admissions: {admissions_path}"
    )
    lines.append(
        f"Patients:   {patients_path}"
    )

    lines.append("")

    lines.append("DATASET")
    lines.append("-" * 72)
    lines.append(
        f"Training rows: {len(train_df):,}"
    )
    lines.append(
        f"Testing rows:  {len(test_df):,}"
    )

    lines.append("")

    lines.append("BASELINE MODEL")
    lines.append("-" * 72)
    lines.append(
        f"MAE:  {baseline_metrics['mae_days']:.4f} days"
    )
    lines.append(
        f"RMSE: {baseline_metrics['rmse_days']:.4f} days"
    )
    lines.append(
        f"R²:   {baseline_metrics['r2']:.4f}"
    )

    lines.append("")

    lines.append("RANDOM FOREST MODEL")
    lines.append("-" * 72)
    lines.append(
        f"MAE:  {model_metrics['mae_days']:.4f} days"
    )
    lines.append(
        f"RMSE: {model_metrics['rmse_days']:.4f} days"
    )
    lines.append(
        f"R²:   {model_metrics['r2']:.4f}"
    )

    if model_metrics["mape_percent"] is not None:
        lines.append(
            f"MAPE: {model_metrics['mape_percent']:.2f}%"
        )

    lines.append("")

    lines.append("TRAINING")
    lines.append("-" * 72)
    lines.append(
        f"Training time: {training_seconds:.2f} seconds"
    )

    lines.append("")

    lines.append(
        "TOP FEATURES BY RANDOM FOREST IMPORTANCE"
    )
    lines.append("-" * 72)

    for _, row in top_features.iterrows():
        lines.append(
            f"{str(row['feature']):55s}"
            f" {float(row['importance']):.6f}"
        )

    lines.append("")

    lines.append("MODEL INTERPRETATION")
    lines.append("-" * 72)
    lines.append(
        "The model predicts Length of Stay in days using "
        "information available at admission."
    )

    lines.append(
        "Severity is handled as a categorical feature."
    )

    lines.append(
        "The train/test split is chronological to reduce "
        "future-information leakage."
    )

    lines.append(
        "Discharge information and the LOS target are not "
        "used as predictive features."
    )

    lines.append("")

    lines.append(
        "DOWNSTREAM USE"
    )
    lines.append("-" * 72)
    lines.append(
        "Predicted LOS can support hospital occupancy "
        "forecasting, discharge planning and bed-allocation "
        "optimization in later CareOps-AI modules."
    )

    lines.append("")
    lines.append("=" * 72)

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "\n".join(lines)
        )


# ======================================================================
# MAIN
# ======================================================================

def main() -> None:

    overall_start = time.time()

    try:

        # --------------------------------------------------------------
        # Create output directory
        # --------------------------------------------------------------

        OUTPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print_header(
            "CAREOPS AI — LENGTH-OF-STAY PREDICTION"
        )

        print(
            f"Project root: {PROJECT_ROOT}"
        )

        print(
            f"Output directory: {OUTPUT_DIR}"
        )

        # ==============================================================
        # STEP 1
        # ==============================================================

        print_step(
            1,
            9,
            "Discovering existing project CSV files..."
        )

        admissions_path = find_csv_file(
            ADMISSIONS_FILE
        )

        patients_path = find_csv_file(
            PATIENTS_FILE
        )

        if admissions_path is None:
            raise FileNotFoundError(
                "Could not find admissions.csv in the "
                "project data directories."
            )

        if patients_path is None:
            raise FileNotFoundError(
                "Could not find patients.csv in the "
                "project data directories."
            )

        print(
            f"Admissions source: {admissions_path}"
        )

        admissions = load_csv(
            admissions_path
        )

        scenario_as_of = pd.Timestamp(
            os.environ.get("CAREOPS_AS_OF_DATE", dt.date.today().isoformat())
        ).normalize()
        admissions, excluded_incomplete = filter_completed_admissions(
            admissions, scenario_as_of
        )
        if admissions.empty:
            raise ValueError(
                "No completed admissions are available for LOS training as of "
                f"{scenario_as_of.date()}."
            )
        print(
            f"Completed stays through {scenario_as_of.date()}: "
            f"{len(admissions):,}; excluded incomplete/future stays: "
            f"{excluded_incomplete:,}"
        )

        print(
            f"Admissions shape: "
            f"{admissions.shape[0]:,} rows x "
            f"{admissions.shape[1]} columns"
        )

        print(
            f"Patients source: {patients_path}"
        )

        patients = load_csv(
            patients_path
        )

        print(
            f"Patients shape: "
            f"{patients.shape[0]:,} rows x "
            f"{patients.shape[1]} columns"
        )

        # ==============================================================
        # STEP 2
        # ==============================================================

        print_step(
            2,
            9,
            "Combining admission records with patient information..."
        )

        combined = merge_admissions_patients(
            admissions,
            patients,
        )

        print(
            f"Combined table: "
            f"{combined.shape[0]:,} rows x "
            f"{combined.shape[1]} columns"
        )

        # ==============================================================
        # STEP 3
        # ==============================================================

        print_step(
            3,
            9,
            "Building LOS target and admission-time features..."
        )

        combined, target_method = build_los_target(
            combined
        )

        (
            combined,
            numeric_features,
            categorical_features,
            admission_column,
        ) = build_features(
            combined
        )

        # --------------------------------------------------------------
        # Keep only usable target rows
        # --------------------------------------------------------------

        combined["los_target"] = pd.to_numeric(
            combined["los_target"],
            errors="coerce",
        )

        combined = combined[
            combined["los_target"].notna()
        ].copy()

        combined = combined[
            np.isfinite(
                combined["los_target"]
            )
        ].copy()

        # Remove impossible values.
        combined = combined[
            combined["los_target"] >= 0
        ].copy()

        # Stays admitted during the simulator warm-up (empty hospital) are
        # excluded from training and evaluation; features above were built
        # on the full history first.
        combined = drop_warmup(
            combined,
            admission_column,
        )

        # Admission datetime is needed for chronological split.
        combined = combined[
            combined[
                "admission_datetime_model"
            ].notna()
        ].copy()

        if len(combined) < 100:
            raise ValueError(
                "Too few usable rows remain after preprocessing."
            )

        print(
            f"LOS target method: {target_method}"
        )

        print(
            f"Usable rows: {len(combined):,}"
        )

        print(
            "Numeric features:",
            numeric_features,
        )

        print(
            "Categorical features:",
            categorical_features,
        )

        print(
            "Severity handling: CATEGORICAL (correct)"
        )

        # ==============================================================
        # STEP 4
        # ==============================================================

        print_step(
            4,
            9,
            "Creating chronological train/test split..."
        )

        train_df, test_df = chronological_split(
            combined
        )

        print(
            f"Training rows: {len(train_df):,}"
        )

        print(
            f"Testing rows: {len(test_df):,}"
        )

        print(
            "Training date range:",
            safe_timestamp(
                train_df[
                    "admission_datetime_model"
                ].min()
            ),
            "to",
            safe_timestamp(
                train_df[
                    "admission_datetime_model"
                ].max()
            ),
        )

        print(
            "Testing date range:",
            safe_timestamp(
                test_df[
                    "admission_datetime_model"
                ].min()
            ),
            "to",
            safe_timestamp(
                test_df[
                    "admission_datetime_model"
                ].max()
            ),
        )

        # ==============================================================
        # STEP 5
        # ==============================================================

        print_step(
            5,
            9,
            "Training baseline and XGBoost model..."
        )

        X_train = train_df[
            numeric_features
            + categorical_features
        ].copy()

        y_train = train_df[
            "los_target"
        ].astype(float)

        X_test = test_df[
            numeric_features
            + categorical_features
        ].copy()

        y_test = test_df[
            "los_target"
        ].astype(float)

        # --------------------------------------------------------------
        # Baseline
        # --------------------------------------------------------------

        baseline_prediction = np.full(
            len(y_test),
            float(y_train.mean()),
        )

        baseline_metrics = calculate_metrics(
            y_test,
            baseline_prediction,
        )

        print("")
        print("Baseline results:")
        print(
            f"  Mean LOS: "
            f"{y_train.mean():.3f} days"
        )
        print(
            f"  MAE: "
            f"{baseline_metrics['mae_days']:.3f} days"
        )
        print(
            f"  RMSE: "
            f"{baseline_metrics['rmse_days']:.3f} days"
        )
        print(
            f"  R²: "
            f"{baseline_metrics['r2']:.4f}"
        )

        # --------------------------------------------------------------
        # Preprocessor
        # --------------------------------------------------------------

        preprocessor = build_preprocessor(
            numeric_features,
            categorical_features,
        )

        # --------------------------------------------------------------
        # XGBoost
        # --------------------------------------------------------------

        model = XGBRegressor(**XGBOOST_CONFIG)

        pipeline = Pipeline(
            steps=[
                (
                    "preprocessor",
                    preprocessor,
                ),
                (
                    "model",
                    model,
                ),
            ]
        )

        print("")
        print(
            "Fitting XGBoost..."
        )

        training_start = time.time()

        pipeline.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            time.time()
            - training_start
        )

        print(
            f"XGBoost training completed "
            f"in {training_seconds:.2f} seconds."
        )

        # ==============================================================
        # STEP 6
        # ==============================================================

        print_step(
            6,
            9,
            "Evaluating XGBoost model..."
        )

        predictions = pipeline.predict(
            X_test
        )

        # Never allow negative LOS.
        predictions = np.maximum(
            predictions,
            0,
        )

        model_metrics = calculate_metrics(
            y_test,
            predictions,
        )

        print("")
        print("XGBoost results:")
        print(
            f"  MAE: "
            f"{model_metrics['mae_days']:.3f} days"
        )
        print(
            f"  RMSE: "
            f"{model_metrics['rmse_days']:.3f} days"
        )
        print(
            f"  R²: "
            f"{model_metrics['r2']:.4f}"
        )

        if model_metrics[
            "mape_percent"
        ] is not None:

            print(
                f"  MAPE: "
                f"{model_metrics['mape_percent']:.2f}%"
            )

        # ==============================================================
        # STEP 7
        # ==============================================================

        print_step(
            7,
            9,
            "Generating LOS predictions and feature importance..."
        )

        prediction_output = (
            create_prediction_output(
                test_df,
                predictions,
            )
        )

        feature_importance = (
            build_feature_importance(
                pipeline
            )
        )

        # ==============================================================
        # STEP 8
        # ==============================================================

        print_step(
            8,
            9,
            "Saving model, predictions, metrics, reports, and plots..."
        )

        # --------------------------------------------------------------
        # Save model
        # --------------------------------------------------------------

        joblib.dump(
            pipeline,
            MODEL_FILE,
        )

        # --------------------------------------------------------------
        # Save predictions
        # --------------------------------------------------------------

        prediction_output.to_csv(
            PREDICTIONS_FILE,
            index=False,
        )

        # --------------------------------------------------------------
        # Save feature importance
        # --------------------------------------------------------------

        feature_importance.to_csv(
            FEATURE_IMPORTANCE_FILE,
            index=False,
        )

        # --------------------------------------------------------------
        # Save metrics
        # --------------------------------------------------------------

        metrics = {
            "baseline": baseline_metrics,
            "xgboost": model_metrics,
            "improvement": {
                "mae_reduction_days": (
                    baseline_metrics["mae_days"]
                    - model_metrics["mae_days"]
                ),
                "rmse_reduction_days": (
                    baseline_metrics["rmse_days"]
                    - model_metrics["rmse_days"]
                ),
                "r2_increase": (
                    model_metrics["r2"]
                    - baseline_metrics["r2"]
                ),
            },
        }

        save_json(
            metrics,
            METRICS_FILE,
        )

        # --------------------------------------------------------------
        # Save metadata
        # --------------------------------------------------------------

        metadata = build_model_metadata(
            admissions_path=admissions_path,
            patients_path=patients_path,
            train_df=train_df,
            test_df=test_df,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            target_method=target_method,
            baseline_metrics=baseline_metrics,
            model_metrics=model_metrics,
            training_seconds=training_seconds,
        )

        save_json(
            metadata,
            METADATA_FILE,
        )

        # --------------------------------------------------------------
        # Save data quality
        # --------------------------------------------------------------

        data_quality = build_data_quality_report(
            combined,
            numeric_features,
            categorical_features,
        )

        data_quality.to_csv(
            DATA_QUALITY_FILE,
            index=False,
        )

        # --------------------------------------------------------------
        # Save training summary
        # --------------------------------------------------------------

        training_summary = build_training_summary(
            train_df=train_df,
            test_df=test_df,
            baseline_metrics=baseline_metrics,
            model_metrics=model_metrics,
            training_seconds=training_seconds,
        )

        training_summary.to_csv(
            TRAINING_SUMMARY_FILE,
            index=False,
        )

        # --------------------------------------------------------------
        # Save charts
        # --------------------------------------------------------------

        actual_values = y_test.to_numpy(
            dtype=float
        )

        predicted_values = np.asarray(
            predictions,
            dtype=float,
        )

        save_actual_vs_predicted_plot(
            actual_values,
            predicted_values,
            ACTUAL_VS_PREDICTED_PLOT,
        )

        save_residual_plot(
            actual_values,
            predicted_values,
            RESIDUAL_PLOT,
        )

        save_feature_importance_plot(
            feature_importance,
            FEATURE_IMPORTANCE_PLOT,
        )

        # --------------------------------------------------------------
        # Save text report
        # --------------------------------------------------------------

        write_text_report(
            path=REPORT_FILE,
            admissions_path=admissions_path,
            patients_path=patients_path,
            train_df=train_df,
            test_df=test_df,
            baseline_metrics=baseline_metrics,
            model_metrics=model_metrics,
            feature_importance=feature_importance,
            training_seconds=training_seconds,
        )

        # ==============================================================
        # STEP 9
        # ==============================================================

        print_step(
            9,
            9,
            "Validating generated outputs..."
        )

        output_files = [
            MODEL_FILE,
            PREDICTIONS_FILE,
            FEATURE_IMPORTANCE_FILE,
            METRICS_FILE,
            METADATA_FILE,
            DATA_QUALITY_FILE,
            TRAINING_SUMMARY_FILE,
            ACTUAL_VS_PREDICTED_PLOT,
            RESIDUAL_PLOT,
            FEATURE_IMPORTANCE_PLOT,
            REPORT_FILE,
        ]

        missing_outputs = []

        for path in output_files:

            if path.exists():

                size_kb = (
                    path.stat().st_size
                    / 1024
                )

                print(
                    f"  OK  {path.name:<40} "
                    f"{size_kb:,.1f} KB"
                )

            else:

                missing_outputs.append(
                    path.name
                )

                print(
                    f"  MISSING  {path.name}"
                )

        if missing_outputs:
            raise RuntimeError(
                "Some expected output files were not generated: "
                + ", ".join(missing_outputs)
            )

        # --------------------------------------------------------------
        # Final summary
        # --------------------------------------------------------------

        total_seconds = (
            time.time()
            - overall_start
        )

        print_header(
            "LOS PREDICTION COMPLETED SUCCESSFULLY"
        )

        print(
            "Model: XGBRegressor"
        )

        print(
            f"Training rows: {len(train_df):,}"
        )

        print(
            f"Testing rows: {len(test_df):,}"
        )

        print("")
        print("FINAL MODEL PERFORMANCE")
        print("-" * 72)

        print(
            f"MAE:  "
            f"{model_metrics['mae_days']:.3f} days"
        )

        print(
            f"RMSE: "
            f"{model_metrics['rmse_days']:.3f} days"
        )

        print(
            f"R²:   "
            f"{model_metrics['r2']:.4f}"
        )

        if model_metrics[
            "mape_percent"
        ] is not None:

            print(
                f"MAPE: "
                f"{model_metrics['mape_percent']:.2f}%"
            )

        print("")
        print(
            f"Total execution time: "
            f"{total_seconds:.2f} seconds"
        )

        print("")
        print(
            "Important: Existing project CSV files were not modified."
        )

        print("")
        print(
            f"Predictions saved to:"
        )
        print(
            f"  {PREDICTIONS_FILE}"
        )

        print("")
        print(
            f"Model saved to:"
        )
        print(
            f"  {MODEL_FILE}"
        )

        print("")
        print(
            f"Report saved to:"
        )
        print(
            f"  {REPORT_FILE}"
        )

        print("")
        print("=" * 72)

    except Exception as error:

        print("")
        print("=" * 72)
        print(
            "LOS PREDICTION FAILED"
        )
        print("=" * 72)

        print(
            f"{type(error).__name__}: {error}"
        )

        print("")
        print(
            "The source CSV files have NOT been regenerated."
        )

        print(
            "Review the error above before making changes."
        )

        raise


# ======================================================================
# ENTRY POINT
# ======================================================================

if __name__ == "__main__":
    main()
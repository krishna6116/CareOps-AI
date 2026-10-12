# ============================================================
# CAREOPS AI - OCCUPANCY FORECASTING
# Robust 30-Day Hospital Occupancy Forecast
# ============================================================

from pathlib import Path
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from count_outputs import assert_count_column, to_stock_counts
from scenario_config import WARMUP_END, drop_warmup
from forecast_evaluation import (
    NAIVE_BASELINES,
    model_status,
    threshold_exceedance,
    conformal_intervals,
    forecast_sanity,
    horizon_bucket,
    residual_diagnostics,
    rolling_origin_backtest,
    rolling_origins,
    seasonal_mase_scale,
    select_champion,
    summarize_backtest,
)

# ============================================================
# 1. PROJECT PATHS
# ============================================================

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]

SCENARIO_DATA_DIR = os.environ.get("CAREOPS_DATA_DIR")
PROCESSED_DIR = (
    Path(SCENARIO_DATA_DIR)
    if SCENARIO_DATA_DIR
    else PROJECT_ROOT / "data" / "processed"
)
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
OUTPUT_DIR = PROJECT_ROOT / "ml" / "occupancy_forecasting"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. FILE LOADER
# ============================================================

def find_csv(filename):
    locations = [PROCESSED_DIR / filename]
    if not SCENARIO_DATA_DIR:
        locations.extend([
            SYNTHETIC_DIR / filename,
            PROJECT_ROOT / filename,
        ])

    for path in locations:
        if path.exists():
            return path

    return None


def load_csv(filename, required=True):
    path = find_csv(filename)

    if path is None:
        if required:
            raise FileNotFoundError(
                f"Could not find {filename} in processed or synthetic data folders."
            )
        return None

    df = pd.read_csv(path)

    print(f"Found {filename}: {len(df):,} rows")

    return df


# ============================================================
# 3. DATE DETECTION
# ============================================================

def parse_date_column(df, candidates=None):
    """
    Find the most appropriate date column automatically.
    """

    if df is None or df.empty:
        return None, None

    candidates = candidates or []

    # First try exact candidate names
    for col in candidates:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")

            if parsed.notna().sum() >= max(5, len(df) * 0.05):
                return col, parsed

    # Then search by column name
    keywords = [
        "date",
        "admission",
        "admit",
        "discharge",
        "day",
        "timestamp",
    ]

    possible = []

    for col in df.columns:
        name = str(col).lower()

        if any(k in name for k in keywords):
            parsed = pd.to_datetime(df[col], errors="coerce")

            valid = parsed.notna().sum()

            if valid >= max(5, len(df) * 0.05):
                possible.append((valid, col, parsed))

    if possible:
        possible.sort(reverse=True, key=lambda x: x[0])
        _, col, parsed = possible[0]
        return col, parsed

    return None, None


# ============================================================
# 4. BUILD OCCUPANCY FROM ADMISSIONS
# ============================================================

def find_admission_dates(admissions):
    """
    Automatically locate admission and discharge dates.
    """

    admission_candidates = [
        "admission_date",
        "admit_date",
        "admission_datetime",
        "admit_datetime",
        "admission_day",
    ]

    discharge_candidates = [
        "discharge_date",
        "discharged_date",
        "discharge_datetime",
        "discharge_day",
    ]

    admission_col, admission_dates = parse_date_column(
        admissions,
        admission_candidates
    )

    discharge_col, discharge_dates = parse_date_column(
        admissions,
        discharge_candidates
    )

    return (
        admission_col,
        admission_dates,
        discharge_col,
        discharge_dates,
    )


def build_occupancy_from_admissions(admissions, dates, total_beds):
    """
    Calculate daily occupied beds using admission/discharge events.

    A patient is considered occupying a bed from admission date
    through discharge date.

    This implementation uses event aggregation rather than
    repeatedly scanning the entire admissions table.
    """

    (
        admission_col,
        admission_dates,
        discharge_col,
        discharge_dates,
    ) = find_admission_dates(admissions)

    if admission_dates is None:
        return None

    temp = pd.DataFrame({
        "admission": pd.to_datetime(admission_dates, errors="coerce")
    })

    if discharge_dates is not None:
        temp["discharge"] = pd.to_datetime(
            discharge_dates,
            errors="coerce"
        )
    else:
        temp["discharge"] = pd.NaT

    temp = temp.dropna(subset=["admission"])

    temp["admission"] = temp["admission"].dt.normalize()
    temp["discharge"] = temp["discharge"].dt.normalize()

    # Ignore impossible records
    temp.loc[
        temp["discharge"] < temp["admission"],
        "discharge"
    ] = pd.NaT

    # Patients without discharge are assumed active through
    # the last observed operational date.
    last_date = dates.max()

    temp["discharge"] = temp["discharge"].fillna(last_date)

    # Event approach:
    #
    # +1 on admission date
    # -1 on the day after discharge date
    #
    # This means admission and discharge dates are inclusive.

    admission_events = (
        temp.groupby("admission")
        .size()
        .rename("change")
    )

    discharge_events = (
        temp["discharge"] + pd.Timedelta(days=1)
    )

    discharge_events = (
        discharge_events
        .groupby(discharge_events)
        .size()
        .rename("change")
        .mul(-1)
    )

    events = pd.concat([
        admission_events,
        discharge_events
    ])

    events = events.groupby(level=0).sum()

    occupancy = (
        events
        .reindex(dates, fill_value=0)
        .cumsum()
    )

    occupancy = occupancy.clip(
        lower=0,
        upper=total_beds
    )

    return occupancy


# ============================================================
# 5. FALLBACK OCCUPANCY FROM DAILY OPERATIONS
# ============================================================

def find_numeric_column(df, names):
    if df is None:
        return None

    lower_map = {
        str(c).lower(): c
        for c in df.columns
    }

    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    for col in df.columns:
        c = str(col).lower()

        for name in names:
            if name.lower() in c:
                return col

    return None


def build_occupancy_from_daily_operations(
    daily_operations,
    dates,
    total_beds,
    department_beds=None
):

    if daily_operations is None:
        return None

    date_col, parsed_dates = parse_date_column(
        daily_operations,
        [
            "date",
            "operation_date",
            "day",
            "record_date",
        ]
    )

    if parsed_dates is None:
        return None

    temp = daily_operations.copy()
    temp["_date"] = parsed_dates.dt.normalize()

    occupied_col = find_numeric_column(
        temp,
        [
            "occupied_beds",
            "occupied",
            "beds_occupied",
            "inpatient_beds_occupied",
        ]
    )

    available_col = find_numeric_column(
        temp,
        [
            "available_beds",
            "beds_available",
            "available",
        ]
    )

    occupancy_col = find_numeric_column(
        temp,
        [
            "occupancy_rate",
            "occupancy",
            "bed_occupancy",
        ]
    )

    result = pd.Series(
        np.nan,
        index=dates,
        dtype=float
    )

    grouped = temp.groupby("_date")

    if occupied_col is not None:
        daily = grouped[occupied_col].mean()

        result.loc[
            result.index.intersection(daily.index)
        ] = daily.loc[
            daily.index.intersection(result.index)
        ]

        return result.clip(0, total_beds)

    if (
        occupancy_col is not None
        and department_beds is not None
        and "department" in temp.columns
    ):
        # Department rows report occupancy as a share of that
        # department's beds. Weight by department capacity; an
        # unweighted mean treats 10 oncology beds like 100
        # general-medicine beds.
        rate = temp[occupancy_col].astype(float)
        if rate.max() > 1.5:
            rate = rate / 100.0
        departments = temp["department"].astype(str).str.strip()
        capacity = departments.map(department_beds)
        if capacity.isna().any():
            missing = sorted(departments[capacity.isna()].unique())
            raise ValueError(f"No bed inventory for departments: {missing}")
        occupied = (rate * capacity).groupby(temp["_date"]).sum().round()

        result.loc[
            result.index.intersection(occupied.index)
        ] = occupied.loc[
            occupied.index.intersection(result.index)
        ]

        return result.clip(0, total_beds)

    if occupancy_col is not None:
        daily = grouped[occupancy_col].mean()

        # Convert percentage to proportion if needed.
        if daily.max() > 1.5:
            daily = daily / 100.0

        occupied = daily * total_beds

        result.loc[
            result.index.intersection(occupied.index)
        ] = occupied.loc[
            occupied.index.intersection(result.index)
        ]

        return result.clip(0, total_beds)

    if available_col is not None:
        daily = grouped[available_col].mean()
        occupied = total_beds - daily

        result.loc[
            result.index.intersection(occupied.index)
        ] = occupied.loc[
            occupied.index.intersection(result.index)
        ]

        return result.clip(0, total_beds)

    return None


# ============================================================
# 6. DETERMINE TOTAL BED CAPACITY
# ============================================================

def determine_bed_capacity(beds):

    if beds is None or beds.empty:
        raise ValueError("beds.csv is empty or unavailable.")

    capacity_col = find_numeric_column(
        beds,
        [
            "capacity",
            "bed_capacity",
            "beds",
            "total_beds",
        ]
    )

    if capacity_col is not None:
        values = pd.to_numeric(
            beds[capacity_col],
            errors="coerce"
        ).dropna()

        if len(values) > 0:

            # If every row represents an individual bed,
            # use row count rather than accidentally summing
            # a repeated capacity field.
            if capacity_col.lower() in [
                "beds",
                "total_beds"
            ]:
                capacity = values.sum()

            else:
                # If the values are all 1, each row is a bed.
                if values.nunique() == 1 and values.iloc[0] == 1:
                    capacity = len(values)
                else:
                    capacity = values.sum()

            if capacity > 0:
                return int(round(capacity))

    # Safest fallback: one row = one bed
    return int(len(beds))


# ============================================================
# 7. FEATURE ENGINEERING FOR OCCUPANCY
# ============================================================

def create_features(df):

    data = df.copy()

    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").reset_index(drop=True)

    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_month"] = data["date"].dt.day
    data["month"] = data["date"].dt.month
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)

    # Cyclic calendar features
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

    # Historical occupancy lags
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        data[f"lag_{lag}"] = data["occupancy_rate"].shift(lag)

    # Rolling statistics
    for window in [3, 7, 14, 28]:
        data[f"rolling_mean_{window}"] = (
            data["occupancy_rate"]
            .shift(1)
            .rolling(window)
            .mean()
        )

        data[f"rolling_std_{window}"] = (
            data["occupancy_rate"]
            .shift(1)
            .rolling(window)
            .std()
        )

    # Recent trend
    data["trend_7"] = (
        data["occupancy_rate"].shift(1)
        - data["occupancy_rate"].shift(8)
    )

    data["trend_14"] = (
        data["occupancy_rate"].shift(1)
        - data["occupancy_rate"].shift(15)
    )

    # Time index
    data["days_since_start"] = (
        data["date"] - data["date"].min()
    ).dt.days

    # Target = the same day's occupancy. Every feature above uses
    # values up to the previous day, so each row is a one-step-ahead
    # forecast and matches how the recursive forecast builds rows.
    data["target"] = data["occupancy_rate"]

    return data


# ============================================================
# 8. METRICS
# ============================================================

def calculate_metrics(actual, predicted):

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mask = np.abs(actual) > 1e-6

    if mask.any():
        mape = (
            np.mean(
                np.abs(
                    (actual[mask] - predicted[mask])
                    / actual[mask]
                )
            ) * 100
        )
    else:
        mape = np.nan

    return mae, rmse, mape


# ============================================================
# 9. RECURSIVE FORECAST
# ============================================================

def recursive_rate_forecast(model, rates, horizon, feature_columns, blend):
    """Forecast occupancy rates ``horizon`` days past the end of ``rates``.

    ``rates`` is a date-indexed series of daily occupancy rates. With
    ``blend`` the Ridge prediction is mixed with the weekly pattern and
    recent level and limited to a 12-point daily change (the original
    hand-tuned procedure); without it the pure Ridge prediction is used.
    """
    working = pd.DataFrame({
        "date": rates.index,
        "occupancy_rate": rates.to_numpy(dtype=float),
    }).reset_index(drop=True)
    start_date = working["date"].min()
    forecasts = []

    for _ in range(horizon):
        next_date = working["date"].iloc[-1] + pd.Timedelta(days=1)
        series = pd.concat(
            [working["occupancy_rate"], pd.Series([np.nan])],
            ignore_index=True,
        )
        row = {
            "day_of_week": next_date.dayofweek,
            "day_of_month": next_date.day,
            "month": next_date.month,
            "week_of_year": int(next_date.isocalendar().week),
            "dow_sin": np.sin(2 * np.pi * next_date.dayofweek / 7),
            "dow_cos": np.cos(2 * np.pi * next_date.dayofweek / 7),
            "month_sin": np.sin(2 * np.pi * next_date.month / 12),
            "month_cos": np.cos(2 * np.pi * next_date.month / 12),
            "days_since_start": (next_date - start_date).days,
        }
        for lag in [1, 2, 3, 7, 14, 21, 28]:
            row[f"lag_{lag}"] = series.iloc[-1 - lag]
        previous_values = series.iloc[:-1]
        for window in [3, 7, 14, 28]:
            recent = previous_values.iloc[-window:]
            row[f"rolling_mean_{window}"] = recent.mean()
            row[f"rolling_std_{window}"] = recent.std()
        row["trend_7"] = previous_values.iloc[-1] - previous_values.iloc[-8]
        row["trend_14"] = previous_values.iloc[-1] - previous_values.iloc[-15]

        prediction = float(np.clip(
            model.predict(pd.DataFrame([row])[feature_columns])[0], 0, 1
        ))

        if blend:
            seasonal_prediction = float(previous_values.iloc[-7])
            recent_mean = float(previous_values.iloc[-7:].mean())
            previous = float(previous_values.iloc[-1])
            prediction = 0.60 * prediction + 0.25 * seasonal_prediction + 0.15 * recent_mean
            prediction = np.clip(prediction, max(0, previous - 0.12), min(1, previous + 0.12))
            prediction = float(np.clip(0.75 * prediction + 0.25 * recent_mean, 0, 1))

        working.loc[len(working)] = [next_date, prediction]
        forecasts.append(prediction)

    return np.array(forecasts)


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("CAREOPS AI - OCCUPANCY FORECASTING")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("[1/7] Loading CareOps data...")

    daily_operations = load_csv(
        "daily_operations.csv",
        required=False
    )

    beds = load_csv(
        "beds.csv",
        required=True
    )

    admissions = load_csv(
        "admissions.csv",
        required=False
    )

    if daily_operations is None and admissions is None:
        raise ValueError(
            "Need either daily_operations.csv or admissions.csv."
        )

    print()

    # --------------------------------------------------------
    # Determine bed capacity
    # --------------------------------------------------------

    print("[2/7] Determining hospital bed capacity...")

    total_beds = determine_bed_capacity(beds)

    print(f"Bed capacity: {total_beds}")

    # --------------------------------------------------------
    # Determine historical date range
    # --------------------------------------------------------

    print()
    print("[3/7] Building historical occupancy...")

    operational_date = None

    if daily_operations is not None:
        _, operational_date = parse_date_column(
            daily_operations,
            [
                "date",
                "operation_date",
                "day",
                "record_date",
            ]
        )

    if operational_date is not None:
        valid_dates = operational_date.dropna().dt.normalize()

        start_date = valid_dates.min()
        end_date = valid_dates.max()

    elif admissions is not None:
        admission_col, admission_dates, _, _ = (
            find_admission_dates(admissions)
        )

        if admission_dates is None:
            raise ValueError(
                "Could not find usable dates in the available data."
            )

        valid_dates = admission_dates.dropna().dt.normalize()

        start_date = valid_dates.min()
        end_date = valid_dates.max()

    else:
        raise ValueError("No usable dates found.")

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    print(f"Historical period: {start_date.date()} to {end_date.date()}")
    print(f"Historical days: {len(dates)}")

    # --------------------------------------------------------
    # Build occupancy
    # --------------------------------------------------------

    occupancy = None
    occupancy_source = None

    # Prefer recorded daily operations when they contain occupancy.
    # Admission-event reconstruction is only a fallback because event volume
    # may represent a larger simulated population than the physical inventory.
    if daily_operations is not None:

        print(
            "Using daily operations as occupancy source..."
        )

        department_beds = (
            beds.groupby(beds["department"].astype(str).str.strip()).size()
            if "department" in beds.columns
            else None
        )

        occupancy = build_occupancy_from_daily_operations(
            daily_operations,
            dates,
            total_beds,
            department_beds
        )

        if occupancy is not None:
            occupancy_source = (
                "daily_operations_bed_weighted"
                if department_beds is not None
                else "daily_operations"
            )

    if occupancy is None and admissions is not None:

        print("Daily operations occupancy unavailable; reconstructing from admissions...")

        try:
            occupancy = build_occupancy_from_admissions(
                admissions,
                dates,
                total_beds
            )
            occupancy_source = "admissions_reconstruction"
        except Exception as e:
            print(
                f"Admission-based occupancy failed: {e}"
            )
            occupancy = None

    if occupancy is None:
        raise ValueError(
            "Could not construct historical occupancy."
        )

    history = pd.DataFrame({
        "date": dates,
        "occupied_beds": occupancy.values
    })

    history["total_beds"] = total_beds

    history["available_beds"] = (
        total_beds
        - history["occupied_beds"]
    )

    history["occupancy_rate"] = (
        history["occupied_beds"]
        / total_beds
    )

    history["occupancy_rate"] = (
        history["occupancy_rate"]
        .clip(0, 1)
    )

    # Fill missing historical values conservatively
    history["occupied_beds"] = to_stock_counts(
        history["occupied_beds"]
        .interpolate()
        .ffill()
        .bfill(),
        total_beds
    )

    history["occupancy_rate"] = (
        history["occupied_beds"]
        / total_beds
    ).clip(0, 1)

    history["available_beds"] = (
        total_beds
        - history["occupied_beds"]
    ).clip(0, total_beds)

    # The simulated hospital opens empty and occupancy overshoots while the
    # first cohort is in hospital; drop the warm-up before modelling.
    history = drop_warmup(history).reset_index(drop=True)

    # Current monitoring views are intentionally based on the physical bed
    # inventory, whereas the history above is a daily hospital-wide series.
    bed_monitor = beds.copy()
    bed_monitor["status"] = bed_monitor["status"].astype(str).str.strip().str.lower()
    bed_monitor["ward"] = bed_monitor["ward"].astype(str).str.strip()

    status_summary = (
        bed_monitor.groupby("status")
        .size()
        .rename("bed_count")
        .reset_index()
    )
    status_summary.to_csv(
        OUTPUT_DIR / "current_bed_status_summary.csv",
        index=False
    )

    ward_monitor = (
        bed_monitor.assign(
            occupied=lambda frame: (frame["status"] == "occupied").astype(int),
            available=lambda frame: (frame["status"] == "available").astype(int),
            maintenance=lambda frame: (frame["status"] == "maintenance").astype(int),
        )
        .groupby("ward", as_index=False)[["occupied", "available", "maintenance"]]
        .sum()
    )
    ward_monitor["total_beds"] = (
        ward_monitor["occupied"]
        + ward_monitor["available"]
        + ward_monitor["maintenance"]
    )
    ward_monitor["operational_beds"] = (
        ward_monitor["total_beds"] - ward_monitor["maintenance"]
    )
    ward_monitor["utilization_rate"] = np.where(
        ward_monitor["operational_beds"] > 0,
        ward_monitor["occupied"] / ward_monitor["operational_beds"],
        np.nan,
    )
    ward_monitor.to_csv(
        OUTPUT_DIR / "current_ward_utilization.csv",
        index=False,
    )

    if len(history) < 60:
        raise ValueError(
            f"Only {len(history)} historical days found. "
            "At least 60 days are recommended."
        )

    # Save historical occupancy
    history.to_csv(
        OUTPUT_DIR / "historical_occupancy.csv",
        index=False
    )

    print()
    print("Historical occupancy summary:")
    print(
        f"Average occupancy: "
        f"{history['occupancy_rate'].mean() * 100:.2f}%"
    )

    print(
        f"Latest occupancy: "
        f"{history['occupancy_rate'].iloc[-1] * 100:.2f}%"
    )

    print(
        f"Minimum occupancy: "
        f"{history['occupancy_rate'].min() * 100:.2f}%"
    )

    print(
        f"Maximum occupancy: "
        f"{history['occupancy_rate'].max() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Create features
    # --------------------------------------------------------

    print()
    print("[4/7] Creating forecasting features...")

    features = create_features(history)

    feature_columns = [
        "day_of_week",
        "day_of_month",
        "month",
        "week_of_year",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
        "days_since_start",
        "lag_1",
        "lag_2",
        "lag_3",
        "lag_7",
        "lag_14",
        "lag_21",
        "lag_28",
        "rolling_mean_3",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        "rolling_std_3",
        "rolling_std_7",
        "rolling_std_14",
        "rolling_std_28",
        "trend_7",
        "trend_14",
    ]

    modeling = features.dropna(
        subset=feature_columns + ["target"]
    ).copy()

    if len(modeling) < 60:
        raise ValueError(
            "Not enough rows available after feature engineering."
        )

    X = modeling[feature_columns]
    y = modeling["target"]

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split_index = int(len(modeling) * 0.80)

    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]

    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print("[5/7] Training occupancy forecasting model...")

    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
        (
            "ridge",
            Ridge(
                alpha=2.0
            )
        )
    ])

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    predictions = np.clip(
        predictions,
        0,
        1
    )

    mae, rmse, mape = calculate_metrics(
        y_test,
        predictions
    )

    print()
    print("Model performance:")
    print(f"MAE  : {mae:.6f}")
    print(f"RMSE : {rmse:.6f}")
    print(f"MAPE : {mape:.2f}%")

    # --------------------------------------------------------
    # Seasonal naive baseline
    # --------------------------------------------------------

    baseline = (
        modeling["lag_7"]
        .iloc[split_index:]
        .values
    )

    baseline = np.clip(
        baseline,
        0,
        1
    )

    baseline_mae = mean_absolute_error(
        y_test,
        baseline
    )

    baseline_rmse = np.sqrt(
        mean_squared_error(
            y_test,
            baseline
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = pd.DataFrame({
        "metric": [
            "model_mae",
            "model_rmse",
            "model_mape_percent",
            "seasonal_naive_mae",
            "seasonal_naive_rmse",
            "average_historical_occupancy_percent",
            "latest_historical_occupancy_percent",
            "bed_capacity",
        ],
        "value": [
            mae,
            rmse,
            mape,
            baseline_mae,
            baseline_rmse,
            history["occupancy_rate"].mean() * 100,
            history["occupancy_rate"].iloc[-1] * 100,
            total_beds,
        ]
    })

    metrics.to_csv(
        OUTPUT_DIR / "occupancy_metrics.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save test predictions
    # --------------------------------------------------------

    test_predictions = pd.DataFrame({
        "date": modeling["date"].iloc[split_index:].values,
        "actual_occupancy_rate": y_test.values,
        "predicted_occupancy_rate": predictions,
        "seasonal_naive_rate": baseline,
    })

    test_predictions[
        "actual_occupied_beds"
    ] = to_stock_counts(
        test_predictions["actual_occupancy_rate"]
        * total_beds,
        total_beds
    )

    # Raw model output is kept here for scoring; published
    # forecasts are whole beds.
    test_predictions[
        "predicted_occupied_beds"
    ] = (
        test_predictions["predicted_occupancy_rate"]
        * total_beds
    )

    test_predictions.to_csv(
        OUTPUT_DIR / "occupancy_test_predictions.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save modeling dataset
    # --------------------------------------------------------

    modeling.to_csv(
        OUTPUT_DIR / "occupancy_modeling_dataset.csv",
        index=False
    )

    # Save model
    joblib.dump(
        model,
        OUTPUT_DIR / "occupancy_model.joblib"
    )

    # --------------------------------------------------------
    # 30-DAY ROLLING-ORIGIN BACKTEST AND CHAMPION
    #
    # Every candidate forecasts 30 days recursively from origins
    # inside the holdout. The Ridge model is fitted on the
    # training rows only, so these origins are out of sample.
    # --------------------------------------------------------

    print()
    print("[6/7] Backtesting candidates and generating 30-day occupancy forecast...")

    forecast_days = 30
    backtest_step_days = 7

    occupied_series = (
        history.set_index("date")["occupied_beds"].astype(float).sort_index()
    )
    test_start = pd.Timestamp(modeling["date"].iloc[split_index])

    def ridge_forecaster(blend):
        def forecast(bed_history, horizon):
            rates = bed_history / total_beds
            return recursive_rate_forecast(
                model, rates, horizon, feature_columns, blend
            ) * total_beds
        return forecast

    candidates = {
        **NAIVE_BASELINES,
        "ridge_recursive": ridge_forecaster(False),
        "ridge_blend": ridge_forecaster(True),
    }
    origins = rolling_origins(
        occupied_series.index,
        test_start - pd.Timedelta(days=1),
        forecast_days,
        backtest_step_days,
    )
    backtest = rolling_origin_backtest(
        occupied_series, candidates, origins, forecast_days
    )
    backtest_summary = summarize_backtest(
        backtest,
        seasonal_mase_scale(occupied_series[occupied_series.index < test_start]),
    )
    champion = select_champion(backtest_summary)
    interval_widths, interval_coverage = conformal_intervals(
        backtest, champion, capacity=total_beds
    )

    # Occupancy-risk alerts: does the forecast flag the days that actually
    # reach a threshold? The 90%/95% alert levels never occur in this
    # synthetic history, so an "elevated" level at the 90th percentile of
    # pre-holdout occupancy is also scored.
    elevated_beds = float(np.ceil(
        occupied_series[occupied_series.index < test_start].quantile(0.90)
    ))
    risk_metrics = threshold_exceedance(backtest, champion, {
        "elevated_p90_training": elevated_beds,
        "warning_90pct": 0.90 * total_beds,
        "critical_95pct": 0.95 * total_beds,
    })
    risk_metrics.to_csv(OUTPUT_DIR / "occupancy_risk_metrics.csv", index=False)

    backtest.to_csv(OUTPUT_DIR / "occupancy_backtest_predictions.csv", index=False)
    backtest_summary.to_csv(OUTPUT_DIR / "occupancy_backtest_summary.csv", index=False)
    interval_coverage.to_csv(OUTPUT_DIR / "occupancy_interval_coverage.csv", index=False)

    print()
    print("30-day backtest, occupied beds (lower MAE is better):")
    print(backtest_summary[["model", "MAE", "RMSE", "sMAPE_percent", "MASE", "bias"]].round(3).to_string(index=False))
    print(f"Champion by backtest MAE: {champion}")

    # --------------------------------------------------------
    # PUBLISHED 30-DAY FORECAST (WHOLE BEDS, CAPPED AT CAPACITY)
    # --------------------------------------------------------

    expected_beds = candidates[champion](occupied_series, forecast_days)
    future = pd.DataFrame({
        "date": pd.date_range(
            occupied_series.index.max() + pd.Timedelta(days=1),
            periods=forecast_days,
        ),
        "forecast_occupied_beds": to_stock_counts(expected_beds, total_beds),
    })
    buckets = horizon_bucket(pd.Series(np.arange(1, forecast_days + 1)))
    for level in sorted(interval_widths["level"].unique()):
        label = int(round(level * 100))
        level_widths = interval_widths[interval_widths["level"] == level].set_index("bucket")["half_width"]
        half_width = buckets.map(level_widths).to_numpy()
        lower = np.clip(np.floor(expected_beds - half_width), 0, total_beds)
        upper = np.clip(np.ceil(expected_beds + half_width), 0, total_beds)
        future[f"occupied_beds_lower_{label}"] = np.minimum(lower, future["forecast_occupied_beds"]).astype(int)
        future[f"occupied_beds_upper_{label}"] = np.maximum(upper, future["forecast_occupied_beds"]).astype(int)

    future["forecast_available_beds"] = (
        total_beds
        - future["forecast_occupied_beds"]
    )

    future["forecast_occupancy_rate"] = (
        future["forecast_occupied_beds"]
        / total_beds
    )

    status = model_status(
        backtest_summary, champion, ("ridge_recursive", "ridge_blend")
    )
    for key, value in status.items():
        future[key] = value

    for column in [
        "forecast_occupied_beds", "forecast_available_beds",
        "occupied_beds_lower_80", "occupied_beds_upper_80",
        "occupied_beds_lower_95", "occupied_beds_upper_95",
    ]:
        assert_count_column(future, column, capacity=total_beds)

    forecast_checks = forecast_sanity(
        future["forecast_occupied_beds"], occupied_series.to_numpy(), total_beds
    )
    if forecast_checks["has_negative"] or forecast_checks["exceeds_capacity"]:
        raise RuntimeError(f"Occupancy forecast failed sanity checks: {forecast_checks}")
    residuals = residual_diagnostics(
        test_predictions["actual_occupied_beds"],
        test_predictions["predicted_occupied_beds"],
    )

    # --------------------------------------------------------
    # Capacity alerts
    # --------------------------------------------------------

    def alert_level(rate):

        if rate >= 0.95:
            return "CRITICAL"

        if rate >= 0.90:
            return "WARNING"

        return "NORMAL"

    future["alert_level"] = (
        future["forecast_occupancy_rate"]
        .apply(alert_level)
    )

    future["recommended_action"] = (
        future["alert_level"]
        .map({
            "NORMAL":
                "Normal capacity monitoring.",
            "WARNING":
                "Review discharge planning, bed turnover, and staffing capacity.",
            "CRITICAL":
                "Prioritize discharge planning and activate escalation capacity review."
        })
    )

    # --------------------------------------------------------
    # Save future forecast
    # --------------------------------------------------------

    future.to_csv(
        OUTPUT_DIR / "occupancy_future_forecast.csv",
        index=False
    )

    # Save alerts separately
    alerts = future[
        [
            "date",
            "forecast_occupancy_rate",
            "forecast_occupied_beds",
            "forecast_available_beds",
            "alert_level",
            "recommended_action",
        ]
    ].copy()

    alerts.to_csv(
        OUTPUT_DIR / "occupancy_capacity_alerts.csv",
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    warning_days = int(
        (future["alert_level"] == "WARNING").sum()
    )

    critical_days = int(
        (future["alert_level"] == "CRITICAL").sum()
    )

    normal_days = int(
        (future["alert_level"] == "NORMAL").sum()
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    report = []

    report.append(
        "CAREOPS AI - OCCUPANCY FORECASTING REPORT"
    )

    report.append("=" * 60)
    report.append("")

    report.append(
        f"Historical period: "
        f"{history['date'].min().date()} "
        f"to "
        f"{history['date'].max().date()}"
    )

    report.append(
        f"Historical days: {len(history)}"
    )

    report.append(
        f"Historical occupancy source: {occupancy_source}"
    )

    report.append(
        f"Bed capacity: {total_beds}"
    )

    report.append(
        f"Average historical occupancy: "
        f"{history['occupancy_rate'].mean() * 100:.2f}%"
    )

    report.append(
        f"Latest historical occupancy: "
        f"{history['occupancy_rate'].iloc[-1] * 100:.2f}%"
    )

    report.append("")

    report.append("MODEL PERFORMANCE")
    report.append("-" * 60)

    report.append(
        f"MAE: {mae:.6f}"
    )

    report.append(
        f"RMSE: {rmse:.6f}"
    )

    report.append(
        f"MAPE: {mape:.2f}%"
    )

    report.append(
        f"Seasonal naive MAE: "
        f"{baseline_mae:.6f}"
    )

    report.append(
        f"Seasonal naive RMSE: "
        f"{baseline_rmse:.6f}"
    )

    report.append(
        "(One-step-ahead holdout on occupancy rate; Ridge target is the "
        "same day, features use the previous day and earlier.)"
    )

    report.append("")

    report.append("30-DAY ROLLING-ORIGIN BACKTEST (OCCUPIED BEDS)")
    report.append("-" * 60)

    report.append(
        f"Origins: {len(origins)} every {backtest_step_days} days from "
        f"{origins.min().date()} to {origins.max().date()}"
    )

    report.append(
        backtest_summary[
            ["model", "MAE", "RMSE", "sMAPE_percent", "MASE", "bias"]
        ].round(3).to_string(index=False)
    )

    report.append(f"Published forecast model: {champion}")

    report.append("Interval coverage on later backtest origins:")

    report.append(interval_coverage.round(3).to_string(index=False))

    report.append("")

    report.append("30-DAY FORECAST")
    report.append("-" * 60)

    report.append(
        f"Forecast days: {forecast_days}"
    )

    report.append(
        f"Normal days: {normal_days}"
    )

    report.append(
        f"Warning days: {warning_days}"
    )

    report.append(
        f"Critical days: {critical_days}"
    )

    report.append(
        f"Average forecast occupancy: "
        f"{future['forecast_occupancy_rate'].mean() * 100:.2f}%"
    )

    report.append(
        f"First forecast occupancy: "
        f"{future['forecast_occupancy_rate'].iloc[0] * 100:.2f}%"
    )

    report.append(
        f"Last forecast occupancy: "
        f"{future['forecast_occupancy_rate'].iloc[-1] * 100:.2f}%"
    )

    report.append("")

    report.append(
        "Alert thresholds:"
    )

    report.append(
        "NORMAL: < 90%"
    )

    report.append(
        "WARNING: 90% - 94.99%"
    )

    report.append(
        "CRITICAL: >= 95%"
    )

    report.append("")

    report.append(
        "Important: Forecasts are operational decision-support "
        "outputs based on synthetic data. They are not clinical "
        "recommendations or predictions about individual patients."
    )

    with open(
        OUTPUT_DIR / "occupancy_forecasting_report.txt",
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(report)
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {
        "project": "CareOps AI",
        "stage": "occupancy_forecasting",
        "bed_capacity": int(total_beds),
        "historical_days": int(len(history)),
        "modelling_start_date": str(WARMUP_END),
        "historical_occupancy_source": occupancy_source,
        "forecast_days": int(forecast_days),
        "model": "Ridge regression with lag and seasonal features",
        "published_forecast_model": champion,
        "model_status": status,
        "risk_threshold_metrics": risk_metrics.astype(object).where(risk_metrics.notna(), None).to_dict(orient="records"),
        "rolling_origin_backtest": {
            "origins": [str(origin.date()) for origin in origins],
            "step_days": backtest_step_days,
            "summary_occupied_beds": backtest_summary.to_dict(orient="records"),
            "interval_method": (
                "Split-conformal absolute-error quantiles by horizon bucket; "
                "earlier origins calibrate, later origins measure coverage"
            ),
            "interval_coverage": interval_coverage.to_dict(orient="records"),
        },
        "one_step_residual_diagnostics_beds": residuals,
        "published_forecast_sanity": forecast_checks,
        "output_rules": (
            "Published occupied/available beds and interval bounds are whole "
            "numbers between 0 and bed capacity."
        ),
        "warning_threshold": 0.90,
        "critical_threshold": 0.95,
        "normal_days": normal_days,
        "warning_days": warning_days,
        "critical_days": critical_days,
        "average_historical_occupancy_percent":
            float(
                history["occupancy_rate"].mean() * 100
            ),
        "latest_historical_occupancy_percent":
            float(
                history["occupancy_rate"].iloc[-1] * 100
            ),
        "average_forecast_occupancy_percent":
            float(
                future["forecast_occupancy_rate"].mean()
                * 100
            ),
    }

    with open(
        OUTPUT_DIR / "run_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Feature importance from standardized Ridge coefficients
    # --------------------------------------------------------

    ridge_model = model.named_steps["ridge"]

    coefficients = ridge_model.coef_

    importance = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": coefficients,
        "absolute_importance": np.abs(coefficients)
    })

    importance = importance.sort_values(
        "absolute_importance",
        ascending=False
    )

    importance.to_csv(
        OUTPUT_DIR / "occupancy_feature_importance.csv",
        index=False
    )

    # --------------------------------------------------------
    # Optional chart
    # --------------------------------------------------------

    try:

        import matplotlib.pyplot as plt

        plt.figure(
            figsize=(14, 6)
        )

        recent = history.tail(180)

        plt.plot(
            recent["date"],
            recent["occupied_beds"],
            label="Occupied beds (last 180 days)"
        )

        plt.fill_between(
            future["date"],
            future["occupied_beds_lower_95"],
            future["occupied_beds_upper_95"],
            alpha=0.15,
            label="95% interval"
        )

        plt.fill_between(
            future["date"],
            future["occupied_beds_lower_80"],
            future["occupied_beds_upper_80"],
            alpha=0.3,
            label="80% interval"
        )

        plt.plot(
            future["date"],
            future["forecast_occupied_beds"],
            marker="o",
            label=f"30-day forecast ({champion})"
        )

        plt.axhline(
            0.90 * total_beds,
            linestyle="--",
            label="Warning threshold (90%)"
        )

        plt.axhline(
            total_beds,
            color="black",
            linewidth=1,
            label=f"Bed capacity ({total_beds})"
        )

        plt.gca().yaxis.get_major_locator().set_params(integer=True)

        plt.xlabel("Date")
        plt.ylabel("Occupied beds")
        plt.title(
            "CareOps AI - Hospital Occupancy Forecast (whole beds)"
        )

        plt.legend()
        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIR / "occupancy_forecast.png",
            dpi=150
        )

        plt.close()

    except Exception as e:

        print(
            f"Chart creation skipped: {e}"
        )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("OCCUPANCY FORECASTING COMPLETED")
    print("=" * 70)

    print()
    print(f"Bed capacity        : {total_beds}")
    print(
        f"Historical average  : "
        f"{history['occupancy_rate'].mean() * 100:.2f}%"
    )
    print(
        f"Latest occupancy    : "
        f"{history['occupancy_rate'].iloc[-1] * 100:.2f}%"
    )

    print()
    print(f"Forecast days       : {forecast_days}")
    print(f"Normal days         : {normal_days}")
    print(f"Warning days        : {warning_days}")
    print(f"Critical days       : {critical_days}")

    print()
    print(
        f"Average forecast    : "
        f"{future['forecast_occupancy_rate'].mean() * 100:.2f}%"
    )

    print(
        f"First forecast      : "
        f"{future['forecast_occupancy_rate'].iloc[0] * 100:.2f}%"
    )

    print(
        f"Last forecast       : "
        f"{future['forecast_occupancy_rate'].iloc[-1] * 100:.2f}%"
    )

    print()
    print(
        f"Results saved to:\n{OUTPUT_DIR}"
    )

    print()
    print("Generated files include:")
    print("  - historical_occupancy.csv")
    print("  - occupancy_modeling_dataset.csv")
    print("  - occupancy_metrics.csv")
    print("  - occupancy_test_predictions.csv")
    print("  - occupancy_future_forecast.csv")
    print("  - occupancy_capacity_alerts.csv")
    print("  - occupancy_feature_importance.csv")
    print("  - occupancy_forecasting_report.txt")
    print("  - occupancy_model.joblib")
    print("  - run_metadata.json")
    print("  - occupancy_forecast.png")

    print()
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:
        main()

    except Exception as e:

        print()
        print("=" * 70)
        print("OCCUPANCY FORECASTING FAILED")
        print("=" * 70)

        print()
        print(type(e).__name__ + ":", str(e))

        print()
        print(
            "Please send me the complete error shown above."
        )

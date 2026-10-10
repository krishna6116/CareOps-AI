# ================================================================
# CAREOPS AI
# HEALTHCARE OPERATIONS OPTIMIZATION PLATFORM
#
# COMPLETE FEATURE ENGINEERING PIPELINE
#
# Purpose:
# Convert cleaned operational datasets into ML-ready and
# optimization-ready feature datasets.
#
# INPUT:
# data/processed/
#
# OUTPUT:
# data/feature_engineering/
#
# IMPORTANT:
# - Does NOT modify processed CSV files.
# - Automatically creates the feature_engineering folder.
# - Automatically creates all required feature CSV files.
# - Designed around the CareOps AI project blueprint.
# ================================================================


import pandas as pd
import numpy as np

from pathlib import Path


# ================================================================
# 1. PROJECT PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent

PROCESSED_DIR = BASE_DIR / "processed"

FEATURE_DIR = BASE_DIR / "feature_engineering"

# Automatically create the output folder
FEATURE_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2. HELPER FUNCTIONS
# ================================================================

def section(title):
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def save_csv(df, filename):
    """
    Save a DataFrame automatically inside:
    data/feature_engineering/
    """

    output_path = FEATURE_DIR / filename

    df.to_csv(output_path, index=False)

    print(
        f"Saved: {filename} "
        f"({len(df):,} rows, {len(df.columns)} columns)"
    )

    return output_path


def load_processed(filename):
    """
    Load a processed CSV.

    Your current processed directory contains files such as:
        patients.csv
        admissions.csv
        beds.csv
        staff.csv
        appointments.csv
        daily_operations.csv
        shifts.csv

    Therefore we use those names directly.
    """

    path = PROCESSED_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired file was not found:\n{path}\n\n"
            f"Please make sure the processed file exists."
        )

    return pd.read_csv(path)


def convert_datetime(df, columns):
    """
    Safely convert date/datetime columns.

    Uses format='mixed' where supported to avoid the
    Pandas warning seen previously.
    """

    for column in columns:

        if column not in df.columns:
            continue

        try:
            df[column] = pd.to_datetime(
                df[column],
                format="mixed",
                errors="coerce"
            )
        except TypeError:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    return df


def yes_no_to_binary(series):
    """
    Convert common Yes/No boolean representations to 1/0.
    """

    if series is None:
        return series

    mapping = {
        "yes": 1,
        "no": 0,
        "true": 1,
        "false": 0,
        "y": 1,
        "n": 0,
        "1": 1,
        "0": 0,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


def add_date_features(df, date_column, prefix=""):
    """
    Add standard calendar and time-series features.
    """

    if date_column not in df.columns:
        return df

    date = pd.to_datetime(
        df[date_column],
        format="mixed",
        errors="coerce"
    )

    df[f"{prefix}year"] = date.dt.year
    df[f"{prefix}month"] = date.dt.month
    df[f"{prefix}day"] = date.dt.day
    df[f"{prefix}day_of_week"] = date.dt.dayofweek
    df[f"{prefix}day_of_year"] = date.dt.dayofyear
    df[f"{prefix}week_of_year"] = date.dt.isocalendar().week.astype("Int64")
    df[f"{prefix}quarter"] = date.dt.quarter
    df[f"{prefix}is_weekend"] = (
        date.dt.dayofweek >= 5
    ).astype(int)

    # Beginning/end of month indicators
    df[f"{prefix}is_month_start"] = (
        date.dt.is_month_start
    ).astype(int)

    df[f"{prefix}is_month_end"] = (
        date.dt.is_month_end
    ).astype(int)

    # Cyclical encoding
    df[f"{prefix}month_sin"] = np.sin(
        2 * np.pi * date.dt.month / 12
    )

    df[f"{prefix}month_cos"] = np.cos(
        2 * np.pi * date.dt.month / 12
    )

    df[f"{prefix}day_of_week_sin"] = np.sin(
        2 * np.pi * date.dt.dayofweek / 7
    )

    df[f"{prefix}day_of_week_cos"] = np.cos(
        2 * np.pi * date.dt.dayofweek / 7
    )

    df[f"{prefix}day_of_year_sin"] = np.sin(
        2 * np.pi * date.dt.dayofyear / 365.25
    )

    df[f"{prefix}day_of_year_cos"] = np.cos(
        2 * np.pi * date.dt.dayofyear / 365.25
    )

    return df


def add_categorical_codes(df, columns):

    for column in columns:

        if column not in df.columns:
            continue

        df[f"{column}_code"] = (
            df[column]
            .astype("category")
            .cat.codes
        )

    return df


# ================================================================
# 3. LOAD PROCESSED DATA
# ================================================================

section("LOADING PROCESSED DATA")

patients = load_processed("patients.csv")
admissions = load_processed("admissions.csv")
beds = load_processed("beds.csv")
staff = load_processed("staff.csv")
appointments = load_processed("appointments.csv")
daily_operations = load_processed("daily_operations.csv")
shifts = load_processed("shifts.csv")


print(f"Patients          : {len(patients):,}")
print(f"Admissions        : {len(admissions):,}")
print(f"Beds              : {len(beds):,}")
print(f"Staff             : {len(staff):,}")
print(f"Appointments      : {len(appointments):,}")
print(f"Daily operations  : {len(daily_operations):,}")
print(f"Shifts            : {len(shifts):,}")


# ================================================================
# 4. PATIENT FEATURE ENGINEERING
# ================================================================

section("PATIENT FEATURE ENGINEERING")

patients = patients.copy()

patients = convert_datetime(
    patients,
    ["admission_date", "discharge_date"]
)

# ------------------------------------------------
# Basic numeric conversion
# ------------------------------------------------

for column in [
    "age",
    "LOS",
    "length_of_stay",
    "comorbidity_count"
]:

    if column in patients.columns:

        patients[column] = pd.to_numeric(
            patients[column],
            errors="coerce"
        )


# ------------------------------------------------
# Determine LOS
# ------------------------------------------------

if "LOS" in patients.columns:

    patients["los_target"] = pd.to_numeric(
        patients["LOS"],
        errors="coerce"
    )

elif "length_of_stay" in patients.columns:

    patients["los_target"] = pd.to_numeric(
        patients["length_of_stay"],
        errors="coerce"
    )

elif (
    "admission_date" in patients.columns
    and "discharge_date" in patients.columns
):

    patients["los_target"] = (
        patients["discharge_date"]
        - patients["admission_date"]
    ).dt.days


# ------------------------------------------------
# Admission date features
# ------------------------------------------------

patients = add_date_features(
    patients,
    "admission_date",
    prefix="admission_"
)


# ------------------------------------------------
# Discharge date features
# ------------------------------------------------

patients = add_date_features(
    patients,
    "discharge_date",
    prefix="discharge_"
)


# ------------------------------------------------
# Age features
# ------------------------------------------------

if "age" in patients.columns:

    patients["age_squared"] = (
        patients["age"] ** 2
    )

    patients["age_group"] = pd.cut(
        patients["age"],
        bins=[
            -1,
            17,
            29,
            44,
            59,
            74,
            np.inf
        ],
        labels=[
            "Pediatric",
            "Young_Adult",
            "Adult",
            "Middle_Aged",
            "Senior",
            "Elderly"
        ]
    )

    patients["is_pediatric"] = (
        patients["age"] < 18
    ).astype(int)

    patients["is_senior"] = (
        patients["age"] >= 65
    ).astype(int)


# ------------------------------------------------
# ICU requirement
# ------------------------------------------------

if "ICU_required" in patients.columns:

    patients["ICU_required_binary"] = (
        yes_no_to_binary(
            patients["ICU_required"]
        )
    )

    patients["ICU_required_binary"] = (
        patients["ICU_required_binary"]
        .fillna(0)
        .astype(int)
    )


# ------------------------------------------------
# Procedure indicator
# ------------------------------------------------

if "procedure_indicator" in patients.columns:

    patients["procedure_indicator_binary"] = (
        yes_no_to_binary(
            patients["procedure_indicator"]
        )
    )

    patients["procedure_indicator_binary"] = (
        patients["procedure_indicator_binary"]
        .fillna(0)
        .astype(int)
    )


# ------------------------------------------------
# Severity encoding
# ------------------------------------------------

severity_mapping = {
    "Low": 1,
    "Moderate": 2,
    "High": 3,
    "Critical": 4
}

if "severity" in patients.columns:

    patients["severity_code"] = (
        patients["severity"]
        .astype(str)
        .str.strip()
        .map(severity_mapping)
    )


# ------------------------------------------------
# Admission type encoding
# ------------------------------------------------

if "admission_type" in patients.columns:

    patients = add_categorical_codes(
        patients,
        ["admission_type"]
    )


# ------------------------------------------------
# Department encoding
# ------------------------------------------------

if "department" in patients.columns:

    patients = add_categorical_codes(
        patients,
        ["department"]
    )


# ------------------------------------------------
# Diagnosis encoding
# ------------------------------------------------

if "diagnosis_category" in patients.columns:

    patients = add_categorical_codes(
        patients,
        ["diagnosis_category"]
    )


# ------------------------------------------------
# Gender encoding
# ------------------------------------------------

if "gender" in patients.columns:

    patients = add_categorical_codes(
        patients,
        ["gender"]
    )


# ------------------------------------------------
# Comorbidity features
# ------------------------------------------------

if "comorbidity_count" in patients.columns:

    patients["comorbidity_count"] = (
        pd.to_numeric(
            patients["comorbidity_count"],
            errors="coerce"
        )
        .fillna(0)
    )

    patients["has_comorbidity"] = (
        patients["comorbidity_count"] > 0
    ).astype(int)

    patients["high_comorbidity"] = (
        patients["comorbidity_count"] >= 3
    ).astype(int)


# ------------------------------------------------
# LOS groups
# ------------------------------------------------

if "los_target" in patients.columns:

    patients["los_group"] = pd.cut(
        patients["los_target"],
        bins=[
            -np.inf,
            1,
            3,
            7,
            14,
            np.inf
        ],
        labels=[
            "Very_Short",
            "Short",
            "Medium",
            "Long",
            "Very_Long"
        ]
    )


# ------------------------------------------------
# Patient operational interaction features
# ------------------------------------------------

if (
    "age" in patients.columns
    and "comorbidity_count" in patients.columns
):

    patients["age_comorbidity_interaction"] = (
        patients["age"]
        * patients["comorbidity_count"]
    )


if (
    "severity_code" in patients.columns
    and "ICU_required_binary" in patients.columns
):

    patients["severity_ICU_interaction"] = (
        patients["severity_code"]
        * patients["ICU_required_binary"]
    )


if (
    "severity_code" in patients.columns
    and "procedure_indicator_binary" in patients.columns
):

    patients["severity_procedure_interaction"] = (
        patients["severity_code"]
        * patients["procedure_indicator_binary"]
    )


save_csv(
    patients,
    "patient_features.csv"
)


# ================================================================
# 5. ADMISSION FEATURE ENGINEERING
# ================================================================

section("ADMISSION FEATURE ENGINEERING")

admissions = admissions.copy()

admissions = convert_datetime(
    admissions,
    ["admission_date", "discharge_date"]
)


# ------------------------------------------------
# Numeric fields
# ------------------------------------------------

for column in [
    "LOS",
    "length_of_stay",
    "comorbidity_count"
]:

    if column in admissions.columns:

        admissions[column] = pd.to_numeric(
            admissions[column],
            errors="coerce"
        )


# ------------------------------------------------
# LOS target
# ------------------------------------------------

if "LOS" in admissions.columns:

    admissions["los_target"] = pd.to_numeric(
        admissions["LOS"],
        errors="coerce"
    )

elif "length_of_stay" in admissions.columns:

    admissions["los_target"] = pd.to_numeric(
        admissions["length_of_stay"],
        errors="coerce"
    )

elif (
    "admission_date" in admissions.columns
    and "discharge_date" in admissions.columns
):

    admissions["los_target"] = (
        admissions["discharge_date"]
        - admissions["admission_date"]
    ).dt.days


# ------------------------------------------------
# Date features
# ------------------------------------------------

admissions = add_date_features(
    admissions,
    "admission_date",
    prefix="admission_"
)

admissions = add_date_features(
    admissions,
    "discharge_date",
    prefix="discharge_"
)


# ------------------------------------------------
# Severity
# ------------------------------------------------

if "severity" in admissions.columns:

    admissions["severity_code"] = (
        admissions["severity"]
        .astype(str)
        .str.strip()
        .map(severity_mapping)
    )


# ------------------------------------------------
# ICU
# ------------------------------------------------

if "ICU_required" in admissions.columns:

    admissions["ICU_required_binary"] = (
        yes_no_to_binary(
            admissions["ICU_required"]
        )
        .fillna(0)
        .astype(int)
    )


# ------------------------------------------------
# Procedure
# ------------------------------------------------

if "procedure_indicator" in admissions.columns:

    admissions["procedure_indicator_binary"] = (
        yes_no_to_binary(
            admissions["procedure_indicator"]
        )
        .fillna(0)
        .astype(int)
    )


# ------------------------------------------------
# Categorical features
# ------------------------------------------------

admissions = add_categorical_codes(
    admissions,
    [
        "department",
        "diagnosis_category",
        "admission_type"
    ]
)


# ------------------------------------------------
# Comorbidities
# ------------------------------------------------

if "comorbidity_count" in admissions.columns:

    admissions["comorbidity_count"] = (
        admissions["comorbidity_count"]
        .fillna(0)
    )

    admissions["has_comorbidity"] = (
        admissions["comorbidity_count"] > 0
    ).astype(int)


# ------------------------------------------------
# Emergency admission indicator
# ------------------------------------------------

if "admission_type" in admissions.columns:

    admissions["is_emergency_admission"] = (
        admissions["admission_type"]
        .astype(str)
        .str.lower()
        .str.contains("emergency")
    ).astype(int)


save_csv(
    admissions,
    "admission_features.csv"
)


# ================================================================
# 6. DAILY DEMAND FEATURE ENGINEERING
# ================================================================

section("DAILY DEMAND FEATURE ENGINEERING")

daily = daily_operations.copy()

daily = convert_datetime(
    daily,
    ["date"]
)


# ------------------------------------------------
# Numeric columns
# ------------------------------------------------

numeric_daily_columns = [
    "admissions",
    "discharges",
    "emergency_cases",
    "ICU_admissions",
    "occupancy",
    "staff_demand",
    "bed_demand",
    "seasonal_multiplier",
    "month",
    "year"
]

for column in numeric_daily_columns:

    if column in daily.columns:

        daily[column] = pd.to_numeric(
            daily[column],
            errors="coerce"
        )


# ------------------------------------------------
# Date features
# ------------------------------------------------

daily = add_date_features(
    daily,
    "date"
)


# ------------------------------------------------
# Demand relationships
# ------------------------------------------------

if (
    "admissions" in daily.columns
    and "discharges" in daily.columns
):

    daily["net_patient_change"] = (
        daily["admissions"]
        - daily["discharges"]
    )


if (
    "emergency_cases" in daily.columns
    and "admissions" in daily.columns
):

    daily["emergency_admission_ratio"] = (
        daily["emergency_cases"]
        / daily["admissions"].replace(0, np.nan)
    )


if (
    "ICU_admissions" in daily.columns
    and "admissions" in daily.columns
):

    daily["ICU_admission_ratio"] = (
        daily["ICU_admissions"]
        / daily["admissions"].replace(0, np.nan)
    )


# ------------------------------------------------
# Staff demand relative to patient demand
# ------------------------------------------------

if (
    "staff_demand" in daily.columns
    and "admissions" in daily.columns
):

    daily["staff_per_admission_demand"] = (
        daily["staff_demand"]
        / daily["admissions"].replace(0, np.nan)
    )


# ------------------------------------------------
# Bed demand relative to admissions
# ------------------------------------------------

if (
    "bed_demand" in daily.columns
    and "admissions" in daily.columns
):

    daily["bed_demand_per_admission"] = (
        daily["bed_demand"]
        / daily["admissions"].replace(0, np.nan)
    )


# ------------------------------------------------
# Sort for time-series features
# ------------------------------------------------

sort_columns = []

if "department" in daily.columns:
    sort_columns.append("department")

sort_columns.append("date")

daily = daily.sort_values(
    sort_columns
).reset_index(drop=True)


# ------------------------------------------------
# Lag and rolling demand features
#
# These are critical for forecasting.
# ------------------------------------------------

time_series_columns = [
    "admissions",
    "discharges",
    "emergency_cases",
    "ICU_admissions",
    "occupancy",
    "staff_demand",
    "bed_demand"
]

group_columns = []

if "department" in daily.columns:
    group_columns.append("department")


for column in time_series_columns:

    if column not in daily.columns:
        continue

    if group_columns:

        grouped = daily.groupby(
            group_columns,
            sort=False
        )[column]

        daily[f"{column}_lag_1"] = (
            grouped.shift(1)
        )

        daily[f"{column}_lag_7"] = (
            grouped.shift(7)
        )

        daily[f"{column}_lag_14"] = (
            grouped.shift(14)
        )

        daily[f"{column}_lag_28"] = (
            grouped.shift(28)
        )

        daily[f"{column}_rolling_7_mean"] = (
            grouped.transform(
                lambda x:
                x.shift(1)
                .rolling(7, min_periods=1)
                .mean()
            )
        )

        daily[f"{column}_rolling_14_mean"] = (
            grouped.transform(
                lambda x:
                x.shift(1)
                .rolling(14, min_periods=1)
                .mean()
            )
        )

        daily[f"{column}_rolling_28_mean"] = (
            grouped.transform(
                lambda x:
                x.shift(1)
                .rolling(28, min_periods=1)
                .mean()
            )
        )

        daily[f"{column}_rolling_7_std"] = (
            grouped.transform(
                lambda x:
                x.shift(1)
                .rolling(7, min_periods=2)
                .std()
            )
        )

    else:

        daily[f"{column}_lag_1"] = (
            daily[column].shift(1)
        )

        daily[f"{column}_lag_7"] = (
            daily[column].shift(7)
        )

        daily[f"{column}_lag_14"] = (
            daily[column].shift(14)
        )

        daily[f"{column}_lag_28"] = (
            daily[column].shift(28)
        )

        daily[f"{column}_rolling_7_mean"] = (
            daily[column]
            .shift(1)
            .rolling(7, min_periods=1)
            .mean()
        )

        daily[f"{column}_rolling_14_mean"] = (
            daily[column]
            .shift(1)
            .rolling(14, min_periods=1)
            .mean()
        )

        daily[f"{column}_rolling_28_mean"] = (
            daily[column]
            .shift(1)
            .rolling(28, min_periods=1)
            .mean()
        )

        daily[f"{column}_rolling_7_std"] = (
            daily[column]
            .shift(1)
            .rolling(7, min_periods=2)
            .std()
        )


# ------------------------------------------------
# Demand trend
# ------------------------------------------------

if (
    "admissions_lag_7" in daily.columns
    and "admissions_lag_1" in daily.columns
):

    daily["admissions_week_change"] = (
        daily["admissions_lag_1"]
        - daily["admissions_lag_7"]
    )


if (
    "emergency_cases_lag_1" in daily.columns
    and "emergency_cases_lag_7" in daily.columns
):

    daily["emergency_week_change"] = (
        daily["emergency_cases_lag_1"]
        - daily["emergency_cases_lag_7"]
    )


save_csv(
    daily,
    "daily_demand_features.csv"
)


# ================================================================
# 7. OCCUPANCY FEATURE ENGINEERING
# ================================================================

section("OCCUPANCY FEATURE ENGINEERING")

occupancy = daily.copy()


# ------------------------------------------------
# Occupancy percentage normalization
# ------------------------------------------------

if "occupancy" in occupancy.columns:

    occupancy["occupancy_rate"] = (
        occupancy["occupancy"]
        / 100
    )

    occupancy["occupancy_percent"] = (
        occupancy["occupancy"]
    )


# ------------------------------------------------
# Occupancy risk flags
#
# Configurable thresholds
# ------------------------------------------------

if "occupancy_rate" in occupancy.columns:

    occupancy["occupancy_risk"] = pd.cut(
        occupancy["occupancy_rate"],
        bins=[
            -np.inf,
            0.70,
            0.80,
            0.90,
            0.95,
            np.inf
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very_High",
            "Critical"
        ]
    )

    occupancy["occupancy_over_80"] = (
        occupancy["occupancy_rate"] >= 0.80
    ).astype(int)

    occupancy["occupancy_over_90"] = (
        occupancy["occupancy_rate"] >= 0.90
    ).astype(int)

    occupancy["occupancy_over_95"] = (
        occupancy["occupancy_rate"] >= 0.95
    ).astype(int)


# ------------------------------------------------
# Capacity pressure
# ------------------------------------------------

if (
    "bed_demand" in occupancy.columns
    and "occupancy" in occupancy.columns
):

    occupancy["bed_pressure"] = (
        occupancy["bed_demand"]
        * occupancy["occupancy_rate"]
    )


save_csv(
    occupancy,
    "occupancy_features.csv"
)


# ================================================================
# 8. LOS MODEL FEATURE DATASET
# ================================================================

section("LOS PREDICTION FEATURE ENGINEERING")

los = patients.copy()


# ------------------------------------------------
# Keep the LOS prediction target clearly named
# ------------------------------------------------

if "los_target" not in los.columns:

    if "LOS" in los.columns:

        los["los_target"] = pd.to_numeric(
            los["LOS"],
            errors="coerce"
        )

    elif "length_of_stay" in los.columns:

        los["los_target"] = pd.to_numeric(
            los["length_of_stay"],
            errors="coerce"
        )


# ------------------------------------------------
# Expected discharge date based on actual LOS
#
# This is useful for operational feature preparation.
# Later the ML prediction will replace actual LOS.
# ------------------------------------------------

if (
    "admission_date" in los.columns
    and "los_target" in los.columns
):

    los["expected_discharge_date_from_actual_los"] = (
        los["admission_date"]
        + pd.to_timedelta(
            los["los_target"],
            unit="D"
        )
    )


# ------------------------------------------------
# LOS target groups
# ------------------------------------------------

if "los_target" in los.columns:

    los["los_short_stay"] = (
        los["los_target"] <= 3
    ).astype(int)

    los["los_long_stay"] = (
        los["los_target"] >= 8
    ).astype(int)

    los["los_very_long_stay"] = (
        los["los_target"] >= 15
    ).astype(int)


# ------------------------------------------------
# LOS feature interactions
# ------------------------------------------------

if (
    "severity_code" in los.columns
    and "age" in los.columns
):

    los["severity_age_interaction"] = (
        los["severity_code"]
        * los["age"]
    )


if (
    "severity_code" in los.columns
    and "comorbidity_count" in los.columns
):

    los["severity_comorbidity_interaction"] = (
        los["severity_code"]
        * los["comorbidity_count"]
    )


if (
    "ICU_required_binary" in los.columns
    and "comorbidity_count" in los.columns
):

    los["ICU_comorbidity_interaction"] = (
        los["ICU_required_binary"]
        * los["comorbidity_count"]
    )


# ------------------------------------------------
# Leakage guard
#
# A LOS model may only use information available at admission.
# Discharge-date fields, the raw LOS column, and bands derived from
# the target are only known after discharge. los_target stays as the
# single label column.
# ------------------------------------------------

los_leakage_columns = [
    column for column in los.columns
    if column.startswith("discharge_")
    or column in {
        "LOS",
        "length_of_stay",
        "los_group",
        "expected_discharge_date_from_actual_los",
        "los_short_stay",
        "los_long_stay",
        "los_very_long_stay",
    }
]

save_csv(
    los.drop(columns=los_leakage_columns),
    "los_prediction_features.csv"
)


# ================================================================
# 9. BED FEATURE ENGINEERING
# ================================================================

section("BED FEATURE ENGINEERING")

beds = beds.copy()


# ------------------------------------------------
# Convert boolean-like fields
# ------------------------------------------------

for column in [
    "isolation_capable",
    "emergency_reserve",
    "available_for_emergency"
]:

    if column in beds.columns:

        beds[f"{column}_binary"] = (
            yes_no_to_binary(
                beds[column]
            )
            .fillna(0)
            .astype(int)
        )


# ------------------------------------------------
# Bed status features
# ------------------------------------------------

if "status" in beds.columns:

    beds["is_available"] = (
        beds["status"]
        .astype(str)
        .str.lower()
        .isin([
            "available",
            "vacant",
            "free"
        ])
    ).astype(int)

    beds["is_occupied"] = (
        beds["status"]
        .astype(str)
        .str.lower()
        .isin([
            "occupied",
            "in_use"
        ])
    ).astype(int)


# ------------------------------------------------
# Bed type encoding
# ------------------------------------------------

beds = add_categorical_codes(
    beds,
    [
        "ward",
        "bed_type",
        "department",
        "status"
    ]
)


# ------------------------------------------------
# Compatibility features
# ------------------------------------------------

if "bed_type" in beds.columns:

    beds["is_icu_bed"] = (
        beds["bed_type"]
        .astype(str)
        .str.lower()
        .str.contains("icu")
    ).astype(int)

    beds["is_emergency_bed"] = (
        beds["bed_type"]
        .astype(str)
        .str.lower()
        .str.contains("emergency")
    ).astype(int)

    beds["is_general_bed"] = (
        beds["bed_type"]
        .astype(str)
        .str.lower()
        .str.contains("general")
    ).astype(int)


save_csv(
    beds,
    "bed_features.csv"
)


# ================================================================
# 10. BED CAPACITY SUMMARY
# ================================================================

section("BED CAPACITY SUMMARY")

bed_capacity_group_columns = []

for column in [
    "department",
    "ward",
    "bed_type"
]:

    if column in beds.columns:

        bed_capacity_group_columns.append(
            column
        )


if bed_capacity_group_columns:

    bed_capacity_summary = (
        beds
        .groupby(
            bed_capacity_group_columns,
            dropna=False
        )
        .agg(
            total_beds=("bed_id", "count")
        )
        .reset_index()
    )

    if "is_available" in beds.columns:

        available = (
            beds
            .groupby(
                bed_capacity_group_columns,
                dropna=False
            )["is_available"]
            .sum()
            .reset_index(
                name="available_beds"
            )
        )

        bed_capacity_summary = (
            bed_capacity_summary
            .merge(
                available,
                on=bed_capacity_group_columns,
                how="left"
            )
        )

    if "is_occupied" in beds.columns:

        occupied = (
            beds
            .groupby(
                bed_capacity_group_columns,
                dropna=False
            )["is_occupied"]
            .sum()
            .reset_index(
                name="occupied_beds"
            )
        )

        bed_capacity_summary = (
            bed_capacity_summary
            .merge(
                occupied,
                on=bed_capacity_group_columns,
                how="left"
            )
        )

    if "isolation_capable_binary" in beds.columns:

        isolation = (
            beds
            .groupby(
                bed_capacity_group_columns,
                dropna=False
            )["isolation_capable_binary"]
            .sum()
            .reset_index(
                name="isolation_capable_beds"
            )
        )

        bed_capacity_summary = (
            bed_capacity_summary
            .merge(
                isolation,
                on=bed_capacity_group_columns,
                how="left"
            )
        )

    if "emergency_reserve_binary" in beds.columns:

        reserve = (
            beds
            .groupby(
                bed_capacity_group_columns,
                dropna=False
            )["emergency_reserve_binary"]
            .sum()
            .reset_index(
                name="emergency_reserve_beds"
            )
        )

        bed_capacity_summary = (
            bed_capacity_summary
            .merge(
                reserve,
                on=bed_capacity_group_columns,
                how="left"
            )
        )

    if "available_for_emergency_binary" in beds.columns:

        emergency = (
            beds
            .groupby(
                bed_capacity_group_columns,
                dropna=False
            )["available_for_emergency_binary"]
            .sum()
            .reset_index(
                name="emergency_available_beds"
            )
        )

        bed_capacity_summary = (
            bed_capacity_summary
            .merge(
                emergency,
                on=bed_capacity_group_columns,
                how="left"
            )
        )

else:

    bed_capacity_summary = pd.DataFrame(
        {
            "total_beds": [len(beds)]
        }
    )


if "total_beds" in bed_capacity_summary.columns:

    if "occupied_beds" in bed_capacity_summary.columns:

        bed_capacity_summary["utilization_rate"] = (
            bed_capacity_summary["occupied_beds"]
            / bed_capacity_summary["total_beds"]
            .replace(0, np.nan)
        )

    if "available_beds" in bed_capacity_summary.columns:

        bed_capacity_summary["availability_rate"] = (
            bed_capacity_summary["available_beds"]
            / bed_capacity_summary["total_beds"]
            .replace(0, np.nan)
        )


save_csv(
    bed_capacity_summary,
    "bed_capacity_summary.csv"
)


# ================================================================
# 11. STAFF FEATURE ENGINEERING
# ================================================================

section("STAFF SCHEDULING FEATURE ENGINEERING")

staff = staff.copy()


# ------------------------------------------------
# Numeric staff fields
# ------------------------------------------------

for column in [
    "max_hours"
]:

    if column in staff.columns:

        staff[column] = pd.to_numeric(
            staff[column],
            errors="coerce"
        )


# ------------------------------------------------
# Availability
# ------------------------------------------------

if "availability" in staff.columns:

    staff["availability_binary"] = (
        yes_no_to_binary(
            staff["availability"]
        )
        .fillna(0)
        .astype(int)
    )


# ------------------------------------------------
# Staff categorical encodings
# ------------------------------------------------

staff = add_categorical_codes(
    staff,
    [
        "role",
        "specialty",
        "department",
        "shift_preference"
    ]
)


# ------------------------------------------------
# Staff role indicators
# ------------------------------------------------

if "role" in staff.columns:

    role_lower = (
        staff["role"]
        .astype(str)
        .str.lower()
    )

    staff["is_doctor"] = (
        role_lower.str.contains("doctor|physician")
    ).astype(int)

    staff["is_nurse"] = (
        role_lower.str.contains("nurse")
    ).astype(int)

    staff["is_support_staff"] = (
        role_lower.str.contains("support|assistant")
    ).astype(int)


save_csv(
    staff,
    "staff_features.csv"
)


# ================================================================
# 12. SHIFT FEATURE ENGINEERING
# ================================================================

section("SHIFT FEATURE ENGINEERING")

shifts = shifts.copy()

shifts = convert_datetime(
    shifts,
    ["date"]
)


# ------------------------------------------------
# Numeric fields
# ------------------------------------------------

for column in [
    "hours",
    "max_hours"
]:

    if column in shifts.columns:

        shifts[column] = pd.to_numeric(
            shifts[column],
            errors="coerce"
        )


# ------------------------------------------------
# Assigned conversion
# ------------------------------------------------

if "assigned" in shifts.columns:

    shifts["assigned_binary"] = (
        yes_no_to_binary(
            shifts["assigned"]
        )
        .fillna(0)
        .astype(int)
    )


# ------------------------------------------------
# Rest violation conversion
# ------------------------------------------------

if "rest_violation" in shifts.columns:

    shifts["rest_violation_binary"] = (
        yes_no_to_binary(
            shifts["rest_violation"]
        )
        .fillna(0)
        .astype(int)
    )


# ------------------------------------------------
# Date features
# ------------------------------------------------

shifts = add_date_features(
    shifts,
    "date"
)


# ------------------------------------------------
# Shift encoding
# ------------------------------------------------

shifts = add_categorical_codes(
    shifts,
    [
        "shift",
        "department",
        "staff_role",
        "specialty"
    ]
)


# ------------------------------------------------
# Hours utilization
# ------------------------------------------------

if (
    "hours" in shifts.columns
    and "max_hours" in shifts.columns
):

    shifts["hours_utilization"] = (
        shifts["hours"]
        / shifts["max_hours"].replace(
            0,
            np.nan
        )
    )

    shifts["hours_remaining"] = (
        shifts["max_hours"]
        - shifts["hours"]
    )

    shifts["overtime_risk"] = (
        shifts["hours_utilization"] > 1
    ).astype(int)


# ------------------------------------------------
# Assigned staffing demand
# ------------------------------------------------

if (
    "assigned_binary" in shifts.columns
    and "hours" in shifts.columns
):

    shifts["assigned_hours"] = (
        shifts["assigned_binary"]
        * shifts["hours"]
    )


save_csv(
    shifts,
    "shift_features.csv"
)


# ================================================================
# 13. STAFF WORKLOAD FEATURES
# ================================================================

section("STAFF WORKLOAD FEATURE ENGINEERING")


staff_workload_group = []

for column in [
    "date",
    "staff_id"
]:

    if column in shifts.columns:

        staff_workload_group.append(
            column
        )


if staff_workload_group:

    workload = (
        shifts
        .groupby(
            staff_workload_group,
            dropna=False
        )
        .agg(
            total_shift_hours=(
                "hours",
                "sum"
            )
        )
        .reset_index()
    )

    if "assigned_binary" in shifts.columns:

        assigned_hours = (
            shifts
            .assign(
                assigned_hours_calc=
                shifts["assigned_binary"]
                * shifts["hours"]
            )
            .groupby(
                staff_workload_group,
                dropna=False
            )["assigned_hours_calc"]
            .sum()
            .reset_index(
                name="total_assigned_hours"
            )
        )

        workload = (
            workload
            .merge(
                assigned_hours,
                on=staff_workload_group,
                how="left"
            )
        )

    if "rest_violation_binary" in shifts.columns:

        violations = (
            shifts
            .groupby(
                staff_workload_group,
                dropna=False
            )["rest_violation_binary"]
            .sum()
            .reset_index(
                name="rest_violation_count"
            )
        )

        workload = (
            workload
            .merge(
                violations,
                on=staff_workload_group,
                how="left"
            )
        )

    if (
        "total_assigned_hours" in workload.columns
    ):

        workload["workload_level"] = pd.cut(
            workload["total_assigned_hours"],
            bins=[
                -np.inf,
                8,
                16,
                24,
                40,
                np.inf
            ],
            labels=[
                "Very_Low",
                "Low",
                "Moderate",
                "High",
                "Very_High"
            ]
        )

else:

    workload = pd.DataFrame()


save_csv(
    workload,
    "staff_workload_features.csv"
)


# ================================================================
# 14. APPOINTMENT FEATURE ENGINEERING
# ================================================================

section("APPOINTMENT FEATURE ENGINEERING")

appointments = appointments.copy()


# ------------------------------------------------
# Date conversion
# ------------------------------------------------

appointments = convert_datetime(
    appointments,
    ["date"]
)


# ------------------------------------------------
# Appointment datetime
#
# This avoids the warning shown previously.
# ------------------------------------------------

if (
    "date" in appointments.columns
    and "time" in appointments.columns
):

    date_string = (
        appointments["date"]
        .dt.strftime("%Y-%m-%d")
    )

    time_string = (
        appointments["time"]
        .astype(str)
        .str.strip()
    )

    appointment_datetime_string = (
        date_string
        + " "
        + time_string
    )

    try:

        appointments["appointment_datetime"] = (
            pd.to_datetime(
                appointment_datetime_string,
                format="mixed",
                errors="coerce"
            )
        )

    except TypeError:

        appointments["appointment_datetime"] = (
            pd.to_datetime(
                appointment_datetime_string,
                errors="coerce"
            )
        )


# ------------------------------------------------
# Date features
# ------------------------------------------------

appointments = add_date_features(
    appointments,
    "date",
    prefix="appointment_"
)


# ------------------------------------------------
# Appointment time features
# ------------------------------------------------

if "appointment_datetime" in appointments.columns:

    appointments["appointment_hour"] = (
        appointments["appointment_datetime"]
        .dt.hour
    )

    appointments["appointment_minute"] = (
        appointments["appointment_datetime"]
        .dt.minute
    )

    appointments["appointment_hour_sin"] = (
        np.sin(
            2
            * np.pi
            * appointments["appointment_hour"]
            / 24
        )
    )

    appointments["appointment_hour_cos"] = (
        np.cos(
            2
            * np.pi
            * appointments["appointment_hour"]
            / 24
        )
    )

    appointments["is_morning"] = (
        appointments["appointment_hour"] < 12
    ).astype(int)

    appointments["is_afternoon"] = (
        (
            appointments["appointment_hour"] >= 12
        )
        &
        (
            appointments["appointment_hour"] < 17
        )
    ).astype(int)

    appointments["is_evening"] = (
        appointments["appointment_hour"] >= 17
    ).astype(int)


# ------------------------------------------------
# Duration
# ------------------------------------------------

if "duration" in appointments.columns:

    appointments["duration"] = pd.to_numeric(
        appointments["duration"],
        errors="coerce"
    )

    appointments["long_appointment"] = (
        appointments["duration"] >= 45
    ).astype(int)


# ------------------------------------------------
# Attendance target
# ------------------------------------------------

if "attended" in appointments.columns:

    appointments["attended_binary"] = (
        yes_no_to_binary(
            appointments["attended"]
        )
    )


# ------------------------------------------------
# No-show target
# ------------------------------------------------

if "no_show" in appointments.columns:

    appointments["no_show_binary"] = (
        yes_no_to_binary(
            appointments["no_show"]
        )
    )

elif "attended_binary" in appointments.columns:

    appointments["no_show_binary"] = (
        1
        - appointments["attended_binary"]
    )


# ------------------------------------------------
# Priority
# ------------------------------------------------

if "priority" in appointments.columns:

    priority_mapping = {
        "Low": 1,
        "Routine": 1,
        "Medium": 2,
        "Moderate": 2,
        "High": 3,
        "Urgent": 4,
        "Emergency": 5
    }

    appointments["priority_code"] = (
        appointments["priority"]
        .astype(str)
        .str.strip()
        .map(priority_mapping)
    )


# ------------------------------------------------
# Patient age
# ------------------------------------------------

if "patient_age" in appointments.columns:

    appointments["patient_age"] = pd.to_numeric(
        appointments["patient_age"],
        errors="coerce"
    )

    appointments["patient_age_squared"] = (
        appointments["patient_age"] ** 2
    )

    appointments["patient_is_senior"] = (
        appointments["patient_age"] >= 65
    ).astype(int)


# ------------------------------------------------
# Patient severity
# ------------------------------------------------

if "patient_severity" in appointments.columns:

    appointments["patient_severity_code"] = (
        appointments["patient_severity"]
        .astype(str)
        .str.strip()
        .map(severity_mapping)
    )


# ------------------------------------------------
# Department / doctor encoding
# ------------------------------------------------

appointments = add_categorical_codes(
    appointments,
    [
        "department",
        "doctor_id",
        "priority"
    ]
)


# ================================================================
# 15. PRIOR ATTENDANCE / NO-SHOW HISTORY
# ================================================================
#
# IMPORTANT:
# We calculate historical attendance using ONLY previous
# appointments for the same patient.
#
# This avoids using the current appointment's outcome
# to predict itself.
# ================================================================

if (
    "patient_id" in appointments.columns
    and "date" in appointments.columns
):

    appointments = appointments.sort_values(
        [
            "patient_id",
            "date"
        ]
    ).reset_index(drop=True)

    if "attended_binary" in appointments.columns:

        appointments["previous_appointments"] = (
            appointments
            .groupby("patient_id")
            .cumcount()
        )

        appointments["previous_attendance_count"] = (
            appointments
            .groupby("patient_id")[
                "attended_binary"
            ]
            .transform(
                lambda x:
                x.shift(1)
                .fillna(0)
                .cumsum()
            )
        )

        appointments["previous_no_show_count"] = (
            appointments["previous_appointments"]
            - appointments["previous_attendance_count"]
        )

        appointments["historical_attendance_rate"] = (
            appointments["previous_attendance_count"]
            /
            appointments["previous_appointments"]
            .replace(0, np.nan)
        )

        appointments[
            "historical_no_show_rate"
        ] = (
            appointments["previous_no_show_count"]
            /
            appointments["previous_appointments"]
            .replace(0, np.nan)
        )

        appointments[
            "historical_attendance_rate"
        ] = (
            appointments[
                "historical_attendance_rate"
            ]
            .fillna(0.5)
        )

        appointments[
            "historical_no_show_rate"
        ] = (
            appointments[
                "historical_no_show_rate"
            ]
            .fillna(0.0)
        )

    else:

        appointments["previous_appointments"] = (
            appointments
            .groupby("patient_id")
            .cumcount()
        )


# ------------------------------------------------
# No-show risk indicator
#
# This is a FEATURE, not the final ML prediction.
# ------------------------------------------------

if "historical_no_show_rate" in appointments.columns:

    appointments["historical_no_show_risk"] = pd.cut(
        appointments["historical_no_show_rate"],
        bins=[
            -np.inf,
            0.10,
            0.25,
            0.50,
            0.75,
            np.inf
        ],
        labels=[
            "Very_Low",
            "Low",
            "Moderate",
            "High",
            "Very_High"
        ]
    )


# ------------------------------------------------
# Appointment operational interactions
# ------------------------------------------------

if (
    "duration" in appointments.columns
    and "priority_code" in appointments.columns
):

    appointments["duration_priority_interaction"] = (
        appointments["duration"]
        * appointments["priority_code"]
    )


if (
    "patient_age" in appointments.columns
    and "patient_severity_code" in appointments.columns
):

    appointments["age_severity_interaction"] = (
        appointments["patient_age"]
        * appointments["patient_severity_code"]
    )


save_csv(
    appointments,
    "appointment_features.csv"
)


# ================================================================
# 16. NO-SHOW MODEL DATASET
# ================================================================

section("NO-SHOW MODEL DATASET")

no_show_columns = []

for column in [
    "appointment_id",
    "patient_id",
    "doctor_id",
    "department",
    "date",
    "appointment_datetime",
    "appointment_hour",
    "appointment_day_of_week",
    "appointment_month",
    "appointment_is_weekend",
    "duration",
    "priority",
    "priority_code",
    "patient_age",
    "patient_age_squared",
    "patient_is_senior",
    "patient_severity",
    "patient_severity_code",
    "previous_appointments",
    "previous_attendance_count",
    "previous_no_show_count",
    "historical_attendance_rate",
    "historical_no_show_rate",
    "historical_no_show_risk",
    "attended_binary",
    "no_show_binary"
]:

    if column in appointments.columns:

        no_show_columns.append(column)


no_show_features = appointments[
    no_show_columns
].copy()


save_csv(
    no_show_features,
    "no_show_features.csv"
)


# ================================================================
# 17. EMERGENCY CAPACITY FEATURE ENGINEERING
# ================================================================

section("EMERGENCY CAPACITY FEATURE ENGINEERING")


# ------------------------------------------------
# Build hospital-level bed capacity
# ------------------------------------------------

total_beds = len(beds)

available_emergency_beds = 0
emergency_reserve_beds = 0
icu_beds = 0

if "available_for_emergency_binary" in beds.columns:

    available_emergency_beds = int(
        beds[
            "available_for_emergency_binary"
        ].sum()
    )

if "emergency_reserve_binary" in beds.columns:

    emergency_reserve_beds = int(
        beds[
            "emergency_reserve_binary"
        ].sum()
    )

if "is_icu_bed" in beds.columns:

    icu_beds = int(
        beds["is_icu_bed"].sum()
    )


emergency_capacity = daily.copy()

emergency_capacity[
    "total_bed_capacity"
] = total_beds

emergency_capacity[
    "emergency_available_bed_capacity"
] = available_emergency_beds

emergency_capacity[
    "emergency_reserve_capacity"
] = emergency_reserve_beds

emergency_capacity[
    "icu_bed_capacity"
] = icu_beds


# ------------------------------------------------
# Emergency pressure
# ------------------------------------------------

if "emergency_cases" in emergency_capacity.columns:

    emergency_capacity[
        "emergency_capacity_pressure"
    ] = (
        emergency_capacity[
            "emergency_cases"
        ]
        /
        max(
            available_emergency_beds,
            1
        )
    )

    emergency_capacity[
        "emergency_high_pressure"
    ] = (
        emergency_capacity[
            "emergency_capacity_pressure"
        ] >= 0.80
    ).astype(int)

    emergency_capacity[
        "emergency_critical_pressure"
    ] = (
        emergency_capacity[
            "emergency_capacity_pressure"
        ] >= 1.00
    ).astype(int)


# ------------------------------------------------
# ICU pressure
# ------------------------------------------------

if "ICU_admissions" in emergency_capacity.columns:

    emergency_capacity[
        "ICU_capacity_pressure"
    ] = (
        emergency_capacity[
            "ICU_admissions"
        ]
        /
        max(
            icu_beds,
            1
        )
    )

    emergency_capacity[
        "ICU_high_pressure"
    ] = (
        emergency_capacity[
            "ICU_capacity_pressure"
        ] >= 0.80
    ).astype(int)

    emergency_capacity[
        "ICU_critical_pressure"
    ] = (
        emergency_capacity[
            "ICU_capacity_pressure"
        ] >= 1.00
    ).astype(int)


save_csv(
    emergency_capacity,
    "emergency_capacity_features.csv"
)


# ================================================================
# 18. HOSPITAL-WIDE DAILY KPI FEATURES
# ================================================================

section("HOSPITAL-WIDE KPI FEATURE ENGINEERING")


hospital_group = (
    daily
    .groupby(
        "date",
        dropna=False
    )
    .agg(
        total_admissions=(
            "admissions",
            "sum"
        ),
        total_discharges=(
            "discharges",
            "sum"
        ),
        total_emergency_cases=(
            "emergency_cases",
            "sum"
        ),
        total_ICU_admissions=(
            "ICU_admissions",
            "sum"
        ),
        average_occupancy=(
            "occupancy",
            "mean"
        ),
        total_staff_demand=(
            "staff_demand",
            "sum"
        ),
        total_bed_demand=(
            "bed_demand",
            "sum"
        )
    )
    .reset_index()
)


hospital_group = add_date_features(
    hospital_group,
    "date"
)


hospital_group["net_patient_change"] = (
    hospital_group["total_admissions"]
    - hospital_group["total_discharges"]
)


hospital_group["emergency_admission_ratio"] = (
    hospital_group["total_emergency_cases"]
    /
    hospital_group["total_admissions"]
    .replace(0, np.nan)
)


hospital_group["ICU_admission_ratio"] = (
    hospital_group["total_ICU_admissions"]
    /
    hospital_group["total_admissions"]
    .replace(0, np.nan)
)


hospital_group["occupancy_rate"] = (
    hospital_group["average_occupancy"]
    / 100
)


hospital_group["occupancy_over_80"] = (
    hospital_group["occupancy_rate"] >= 0.80
).astype(int)


hospital_group["occupancy_over_90"] = (
    hospital_group["occupancy_rate"] >= 0.90
).astype(int)


hospital_group["occupancy_over_95"] = (
    hospital_group["occupancy_rate"] >= 0.95
).astype(int)


# ------------------------------------------------
# Hospital-wide time-series features
# ------------------------------------------------

for column in [
    "total_admissions",
    "total_discharges",
    "total_emergency_cases",
    "total_ICU_admissions",
    "average_occupancy",
    "total_staff_demand",
    "total_bed_demand"
]:

    hospital_group[f"{column}_lag_1"] = (
        hospital_group[column].shift(1)
    )

    hospital_group[f"{column}_lag_7"] = (
        hospital_group[column].shift(7)
    )

    hospital_group[f"{column}_lag_14"] = (
        hospital_group[column].shift(14)
    )

    hospital_group[f"{column}_lag_28"] = (
        hospital_group[column].shift(28)
    )

    hospital_group[
        f"{column}_rolling_7_mean"
    ] = (
        hospital_group[column]
        .shift(1)
        .rolling(
            7,
            min_periods=1
        )
        .mean()
    )

    hospital_group[
        f"{column}_rolling_14_mean"
    ] = (
        hospital_group[column]
        .shift(1)
        .rolling(
            14,
            min_periods=1
        )
        .mean()
    )

    hospital_group[
        f"{column}_rolling_28_mean"
    ] = (
        hospital_group[column]
        .shift(1)
        .rolling(
            28,
            min_periods=1
        )
        .mean()
    )


save_csv(
    hospital_group,
    "hospital_daily_features.csv"
)


# ================================================================
# 19. FEATURE ENGINEERING DATA DICTIONARY
# ================================================================

section("CREATING FEATURE ENGINEERING DATA DICTIONARY")


data_dictionary = pd.DataFrame(
    [
        [
            "patient_features.csv",
            "Patient-level features",
            "LOS, ICU, severity, age, comorbidity, procedure, temporal features"
        ],
        [
            "admission_features.csv",
            "Admission-level features",
            "Admission demand, LOS, severity, ICU, emergency, department"
        ],
        [
            "daily_demand_features.csv",
            "Daily demand forecasting",
            "Admissions, discharges, emergency, ICU, occupancy, lag and rolling features"
        ],
        [
            "occupancy_features.csv",
            "Occupancy forecasting/risk",
            "Occupancy rate, thresholds, capacity pressure"
        ],
        [
            "los_prediction_features.csv",
            "LOS prediction",
            "Age, department, severity, admission type, procedure, comorbidities, ICU"
        ],
        [
            "bed_features.csv",
            "Bed optimization",
            "Bed type, department, status, isolation, emergency reserve, availability"
        ],
        [
            "bed_capacity_summary.csv",
            "Bed capacity",
            "Total, occupied, available, isolation and emergency capacity"
        ],
        [
            "staff_features.csv",
            "Staff scheduling",
            "Role, specialty, department, availability, max hours"
        ],
        [
            "shift_features.csv",
            "Shift scheduling",
            "Date, shift, hours, assignment, rest violations, workload"
        ],
        [
            "staff_workload_features.csv",
            "Staff workload",
            "Daily staff hours, assigned hours, rest violations"
        ],
        [
            "appointment_features.csv",
            "Appointment optimization",
            "Time, duration, priority, doctor, patient and historical attendance"
        ],
        [
            "no_show_features.csv",
            "No-show prediction",
            "Historical attendance/no-show behavior and appointment features"
        ],
        [
            "emergency_capacity_features.csv",
            "Emergency capacity",
            "Emergency pressure, ICU pressure and capacity risk"
        ],
        [
            "hospital_daily_features.csv",
            "Hospital-wide KPIs",
            "Daily hospital demand, occupancy and time-series features"
        ]
    ],
    columns=[
        "file",
        "purpose",
        "important_features"
    ]
)


save_csv(
    data_dictionary,
    "feature_engineering_data_dictionary.csv"
)


# ================================================================
# 20. FINAL VALIDATION
# ================================================================

section("FEATURE ENGINEERING VALIDATION")


expected_outputs = [
    "patient_features.csv",
    "admission_features.csv",
    "daily_demand_features.csv",
    "occupancy_features.csv",
    "los_prediction_features.csv",
    "bed_features.csv",
    "bed_capacity_summary.csv",
    "staff_features.csv",
    "shift_features.csv",
    "staff_workload_features.csv",
    "appointment_features.csv",
    "no_show_features.csv",
    "emergency_capacity_features.csv",
    "hospital_daily_features.csv",
    "feature_engineering_data_dictionary.csv"
]


missing_outputs = []


for filename in expected_outputs:

    output_path = FEATURE_DIR / filename

    if output_path.exists():

        print(
            f"OK     : {filename}"
        )

    else:

        print(
            f"MISSING: {filename}"
        )

        missing_outputs.append(
            filename
        )


# ================================================================
# 21. FINAL SUMMARY
# ================================================================

section("CAREOPS AI FEATURE ENGINEERING FINISHED")


if missing_outputs:

    print(
        "WARNING: Some expected files were not created:"
    )

    for filename in missing_outputs:

        print(
            f"  - {filename}"
        )

else:

    print(
        "SUCCESS: ALL FEATURE ENGINEERING FILES WERE CREATED."
    )


print()
print(
    "Output folder:"
)

print(
    FEATURE_DIR
)

print()
print(
    "Files created:"
)

for filename in expected_outputs:

    output_path = FEATURE_DIR / filename

    if output_path.exists():

        print(
            f"  - {filename}"
        )


print()
print("=" * 80)
print("NEXT STAGE: ML MODEL DEVELOPMENT")
print("=" * 80)
print(
    "Do NOT modify the processed CSV files."
)

print(
    "Feature-engineering datasets are separate outputs."
)

print(
    "The next major ML path is:"
)

print(
    "Admission / occupancy forecasting"
)

print(
    "-> LOS prediction"
)

print(
    "-> ICU demand forecasting"
)

print(
    "-> No-show prediction"
)

print(
    "-> Bed availability forecasting"
)

print(
    "-> OR-Tools bed allocation"
)

print("=" * 80)
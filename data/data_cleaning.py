import os

import pandas as pd


# ============================================================
# CAREOPS AI
# DATA CLEANING SCRIPT
#
# Loads raw synthetic CSVs from data/synthetic/, applies a
# column-appropriate cleaning strategy to each dataset, and
# writes cleaned versions to data/processed/.
# ============================================================


# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# This file is expected to live in data/, alongside synthetic/ and processed/
RAW_DIR = os.path.join(BASE_DIR, "synthetic")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)


# ------------------------------------------------------------
# 2. HELPER: report missing/duplicate counts before + after
# ------------------------------------------------------------

def report(df, label):
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())
    print(f"  {label:<10} rows={len(df):,}  missing={missing}  duplicates={duplicates}")
    return missing, duplicates


# ------------------------------------------------------------
# 3. GENERIC CLEANING STEPS
# ------------------------------------------------------------

def drop_duplicates(df):
    return df.drop_duplicates().reset_index(drop=True)


def fill_numeric_median(df, columns):
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    return df


def fill_categorical_mode(df, columns):
    for col in columns:
        if col in df.columns and df[col].isnull().any():
            mode_value = df[col].mode(dropna=True)
            if not mode_value.empty:
                df[col] = df[col].fillna(mode_value[0])
    return df


def parse_dates(df, columns):
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


# ------------------------------------------------------------
# 4. CLEAN: PATIENTS
# ------------------------------------------------------------

def clean_patients():

    print("\npatients.csv")

    df = pd.read_csv(os.path.join(RAW_DIR, "patients.csv"))
    report(df, "before")

    df = drop_duplicates(df)

    # Numeric fields: median imputation (rare in this synthetic set,
    # but handled in case of manual edits/corrupted rows).
    df = fill_numeric_median(df, ["age", "LOS", "comorbidity_count"])

    # Categorical fields: mode imputation.
    df = fill_categorical_mode(
        df,
        ["gender", "department", "diagnosis_category", "severity", "admission_type"],
    )

    # Dates: parse to real datetime so downstream forecasting/LOS
    # models can use them directly.
    df = parse_dates(df, ["admission_date", "discharge_date"])

    # Names should never legitimately be missing -- if they are,
    # it signals a generation bug rather than something to impute.
    if df["patient_name"].isnull().any():
        print("  WARNING: patient_name has missing values -- check generator")

    report(df, "after")

    df.to_csv(os.path.join(PROCESSED_DIR, "patients.csv"), index=False)

    return df


# ------------------------------------------------------------
# 5. CLEAN: ADMISSIONS
# ------------------------------------------------------------

def clean_admissions():

    print("\nadmissions.csv")

    df = pd.read_csv(os.path.join(RAW_DIR, "admissions.csv"))
    report(df, "before")

    df = drop_duplicates(df)

    df = fill_numeric_median(df, ["LOS", "comorbidity_count"])

    df = fill_categorical_mode(
        df,
        ["department", "diagnosis_category", "severity", "admission_type"],
    )

    df = parse_dates(df, ["admission_date", "discharge_date"])

    # assigned_bed_id can be legitimately null (no compatible bed
    # existed at generation time). Don't impute a fake bed --
    # instead flag it so downstream bed-allocation code can see it.
    df["bed_assigned"] = df["assigned_bed_id"].notna()

    report(df, "after")

    df.to_csv(os.path.join(PROCESSED_DIR, "admissions.csv"), index=False)

    return df


# ------------------------------------------------------------
# 6. CLEAN: BEDS
# ------------------------------------------------------------

def clean_beds():

    print("\nbeds.csv")

    df = pd.read_csv(os.path.join(RAW_DIR, "beds.csv"))
    report(df, "before")

    df = drop_duplicates(df)

    df = fill_categorical_mode(df, ["ward", "bed_type", "status", "department"])

    df["floor"] = df["floor"].fillna(df["floor"].mode()[0]) if df["floor"].isnull().any() else df["floor"]

    report(df, "after")

    df.to_csv(os.path.join(PROCESSED_DIR, "beds.csv"), index=False)

    return df


# ------------------------------------------------------------
# 7. CLEAN: STAFF
# ------------------------------------------------------------

def clean_staff():

    print("\nstaff.csv")

    df = pd.read_csv(os.path.join(RAW_DIR, "staff.csv"))
    report(df, "before")

    df = drop_duplicates(df)

    df = fill_categorical_mode(
        df,
        ["role", "specialty", "department", "availability", "shift_preference"],
    )

    df = fill_numeric_median(df, ["max_hours"])

    if df["staff_name"].isnull().any():
        print("  WARNING: staff_name has missing values -- check generator")

    report(df, "after")

    df.to_csv(os.path.join(PROCESSED_DIR, "staff.csv"), index=False)

    return df


# ------------------------------------------------------------
# 8. CLEAN: APPOINTMENTS
# ------------------------------------------------------------

def clean_appointments():

    print("\nappointments.csv")

    df = pd.read_csv(os.path.join(RAW_DIR, "appointments.csv"))
    report(df, "before")

    df = drop_duplicates(df)

    df = fill_numeric_median(df, ["duration", "patient_age"])

    df = fill_categorical_mode(
        df,
        ["department", "priority", "attended", "patient_severity"],
    )

    df = parse_dates(df, ["date"])

    report(df, "after")

    df.to_csv(os.path.join(PROCESSED_DIR, "appointments.csv"), index=False)

    return df


# ------------------------------------------------------------
# 9. CLEAN: DAILY OPERATIONS
# ------------------------------------------------------------

def clean_daily_operations():

    print("\ndaily_operations.csv")

    df = pd.read_csv(os.path.join(RAW_DIR, "daily_operations.csv"))
    report(df, "before")

    df = drop_duplicates(df)

    df = fill_numeric_median(
        df,
        ["admissions", "discharges", "emergency_cases", "ICU_admissions",
         "occupancy", "staff_demand", "bed_demand"],
    )

    df = fill_categorical_mode(df, ["department", "day_of_week"])

    df = parse_dates(df, ["date"])

    report(df, "after")

    df.to_csv(os.path.join(PROCESSED_DIR, "daily_operations.csv"), index=False)

    return df


# ------------------------------------------------------------
# 10. CLEAN: SHIFTS
# ------------------------------------------------------------

def clean_shifts():

    print("\nshifts.csv")

    df = pd.read_csv(os.path.join(RAW_DIR, "shifts.csv"))
    report(df, "before")

    df = drop_duplicates(df)

    df = fill_categorical_mode(
        df,
        ["shift", "department", "assigned", "staff_role", "specialty"],
    )

    df = fill_numeric_median(df, ["hours", "max_hours"])

    df = parse_dates(df, ["date"])

    if df["staff_name"].isnull().any():
        print("  WARNING: staff_name has missing values -- check generator")

    report(df, "after")

    df.to_csv(os.path.join(PROCESSED_DIR, "shifts.csv"), index=False)

    return df


# ------------------------------------------------------------
# 11. CLEAN: HOSPITAL SNAPSHOT
# ------------------------------------------------------------

def clean_hospital_snapshot():

    print("\nhospital_snapshot.csv")

    filepath = os.path.join(RAW_DIR, "hospital_snapshot.csv")

    if not os.path.exists(filepath):
        print("  (not found -- skipping)")
        return None

    df = pd.read_csv(filepath)
    report(df, "before")

    df = drop_duplicates(df)
    df = parse_dates(df, ["snapshot_date"])

    report(df, "after")

    df.to_csv(os.path.join(PROCESSED_DIR, "hospital_snapshot.csv"), index=False)

    return df


# ------------------------------------------------------------
# 12. MAIN
# ------------------------------------------------------------

def main():

    print("=" * 70)
    print("CAREOPS AI -- DATA CLEANING")
    print("=" * 70)
    print(f"Raw source:      {RAW_DIR}")
    print(f"Cleaned output:  {PROCESSED_DIR}")

    clean_patients()
    clean_beds()
    clean_admissions()
    clean_staff()
    clean_appointments()
    clean_daily_operations()
    clean_shifts()
    clean_hospital_snapshot()

    print("\n" + "=" * 70)
    print("CLEANING COMPLETE -- cleaned CSVs saved to data/processed/")
    print("=" * 70)


if __name__ == "__main__":
    main()
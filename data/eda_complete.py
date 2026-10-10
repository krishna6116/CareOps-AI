# ================================================================
# CAREOPS AI
# COMPLETE EXPLORATORY DATA ANALYSIS
# ================================================================
#
# Purpose:
# Perform comprehensive EDA on the cleaned/processed CareOps AI
# datasets before machine learning and optimization.
#
# Datasets analyzed:
#   1. Patients
#   2. Admissions
#   3. Beds
#   4. Staff
#   5. Appointments
#   6. Daily Operations
#   7. Shifts
#   8. Hospital Snapshot (if available)
#
# Outputs:
#   - EDA summary report
#   - Dataset summaries
#   - Missing-value analysis
#   - Duplicate analysis
#   - Numerical distributions
#   - Categorical distributions
#   - Correlation analysis
#   - Healthcare operational analysis
#   - Demand analysis
#   - ICU analysis
#   - Length-of-stay analysis
#   - Bed analysis
#   - Staff analysis
#   - Appointment/no-show analysis
#   - Shift analysis
#   - Time-series analysis
#   - Charts saved in data/eda/charts
#
# ================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


# ================================================================
# 1. PROJECT PATHS
# ================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"

EDA_DIR = DATA_DIR / "eda"
CHART_DIR = EDA_DIR / "charts"

EDA_DIR.mkdir(parents=True, exist_ok=True)
CHART_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================
# 2. FILE FINDER
# ================================================================
#
# Your processed files currently appear to use names such as:
#
# patients.csv
# admissions.csv
# beds.csv
#
# Earlier versions of the cleaning program could also create:
#
# patients_clean.csv
# admissions_clean.csv
#
# This function checks both formats automatically.
#
# Therefore, you do NOT need to rename your files.
# ================================================================


def find_dataset(dataset_name):

    possible_names = [
        f"{dataset_name}_clean.csv",
        f"{dataset_name}.csv",
    ]

    # First check processed folder
    for filename in possible_names:

        path = PROCESSED_DIR / filename

        if path.exists():
            return path

    # If not found there, check synthetic folder
    for filename in possible_names:

        path = SYNTHETIC_DIR / filename

        if path.exists():
            return path

    return None


# ================================================================
# 3. DATASET LIST
# ================================================================

dataset_names = [
    "patients",
    "admissions",
    "beds",
    "staff",
    "appointments",
    "daily_operations",
    "shifts",
    "hospital_snapshot",
]


# ================================================================
# 4. LOAD DATASETS
# ================================================================

datasets = {}

print()
print("=" * 80)
print("CAREOPS AI - COMPLETE EXPLORATORY DATA ANALYSIS")
print("=" * 80)

print()
print("Searching for datasets...")
print()


for name in dataset_names:

    path = find_dataset(name)

    if path is None:

        print(f"[NOT FOUND] {name}")

    else:

        try:

            df = pd.read_csv(path)

            datasets[name] = df

            print(
                f"[LOADED] {name:<20} "
                f"{len(df):>10,} rows | "
                f"{len(df.columns):>3} columns | "
                f"{path.name}"
            )

        except Exception as error:

            print(f"[ERROR] Could not load {name}: {error}")


# LOS, ICU need, severity and comorbidity are recorded per admission, not
# per patient. Encounter-level analyses use admissions joined to patient age.
if "admissions" in datasets:

    encounters = datasets["admissions"].copy()

    if (
        "patients" in datasets
        and "age" not in encounters.columns
        and {"patient_id", "age"} <= set(datasets["patients"].columns)
    ):

        encounters = encounters.merge(
            datasets["patients"][["patient_id", "age"]],
            on="patient_id",
            how="left"
        )

    if "stay_status" in encounters.columns:

        # Open stays have no final LOS yet.
        encounters = encounters[
            encounters["stay_status"].astype(str).str.lower() == "completed"
        ]

    datasets["encounters"] = encounters


print()
print("=" * 80)

if len(datasets) == 0:

    print("ERROR: No datasets could be loaded.")
    print()
    print("Check that your files are inside:")
    print(PROCESSED_DIR)
    print()
    raise SystemExit

else:

    print(f"Successfully loaded {len(datasets)} datasets.")


# ================================================================
# 5. REPORT FILE
# ================================================================

report_path = EDA_DIR / "eda_report.txt"

report = []


def add_report(text=""):

    print(text)

    report.append(str(text))


# ================================================================
# 6. BASIC DATASET OVERVIEW
# ================================================================

add_report()
add_report("=" * 80)
add_report("CAREOPS AI - EDA REPORT")
add_report("=" * 80)
add_report()

add_report("DATASET OVERVIEW")
add_report("-" * 80)

for name, df in datasets.items():

    add_report(
        f"{name.upper():<25} "
        f"Rows: {len(df):>10,} | "
        f"Columns: {len(df.columns):>3}"
    )


# ================================================================
# 7. DATA TYPES
# ================================================================

add_report()
add_report("=" * 80)
add_report("DATA TYPES")
add_report("=" * 80)

for name, df in datasets.items():

    add_report()
    add_report(f"{name.upper()}")

    for column, dtype in df.dtypes.items():

        add_report(
            f"  {column:<35} {dtype}"
        )


# ================================================================
# 8. MISSING VALUE ANALYSIS
# ================================================================

add_report()
add_report("=" * 80)
add_report("MISSING VALUE ANALYSIS")
add_report("=" * 80)

for name, df in datasets.items():

    add_report()
    add_report(f"{name.upper()}")

    missing = df.isnull().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        add_report("  No missing values.")

    else:

        for column, count in missing.items():

            percentage = count / len(df) * 100

            add_report(
                f"  {column:<35} "
                f"{count:>10,} "
                f"({percentage:.2f}%)"
            )


# ================================================================
# 9. DUPLICATE ANALYSIS
# ================================================================

add_report()
add_report("=" * 80)
add_report("DUPLICATE ANALYSIS")
add_report("=" * 80)

for name, df in datasets.items():

    duplicates = df.duplicated().sum()

    add_report(
        f"{name:<25} {duplicates:,} duplicate rows"
    )


# ================================================================
# 10. NUMERICAL SUMMARY
# ================================================================

add_report()
add_report("=" * 80)
add_report("NUMERICAL SUMMARY")
add_report("=" * 80)

for name, df in datasets.items():

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    if len(numeric_columns) == 0:
        continue

    add_report()
    add_report(f"{name.upper()}")

    summary = df[numeric_columns].describe().round(2)

    add_report(summary.to_string())


# ================================================================
# 11. CATEGORICAL SUMMARY
# ================================================================

add_report()
add_report("=" * 80)
add_report("CATEGORICAL VARIABLE SUMMARY")
add_report("=" * 80)

for name, df in datasets.items():

    categorical_columns = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns

    add_report()
    add_report(f"{name.upper()}")

    for column in categorical_columns:

        unique_count = df[column].nunique(dropna=True)

        add_report(
            f"{column:<35} "
            f"Unique values: {unique_count}"
        )


# ================================================================
# 12. PATIENT EDA
# ================================================================

if "patients" in datasets:

    patients = datasets["patients"].copy()

    add_report()
    add_report("=" * 80)
    add_report("PATIENT ANALYSIS")
    add_report("=" * 80)

    # Age
    if "age" in patients.columns:

        add_report()
        add_report("PATIENT AGE")

        add_report(
            patients["age"].describe().round(2).to_string()
        )

    # Gender
    if "gender" in patients.columns:

        add_report()
        add_report("GENDER DISTRIBUTION")

        add_report(
            patients["gender"]
            .value_counts(dropna=False)
            .to_string()
        )

    # Department
    if "department" in patients.columns:

        add_report()
        add_report("PATIENTS BY DEPARTMENT")

        add_report(
            patients["department"]
            .value_counts()
            .to_string()
        )

    # Diagnosis
    if "diagnosis_category" in patients.columns:

        add_report()
        add_report("DIAGNOSIS DISTRIBUTION")

        add_report(
            patients["diagnosis_category"]
            .value_counts()
            .to_string()
        )

    # Severity
    if "severity" in patients.columns:

        add_report()
        add_report("SEVERITY DISTRIBUTION")

        add_report(
            patients["severity"]
            .value_counts()
            .to_string()
        )

    # Admission type
    if "admission_type" in patients.columns:

        add_report()
        add_report("ADMISSION TYPE")

        add_report(
            patients["admission_type"]
            .value_counts()
            .to_string()
        )

    # ICU
    if "ICU_required" in patients.columns:

        add_report()
        add_report("ICU REQUIREMENT")

        add_report(
            patients["ICU_required"]
            .value_counts()
            .to_string()
        )

    # Procedure
    if "procedure_indicator" in patients.columns:

        add_report()
        add_report("PROCEDURE INDICATOR")

        add_report(
            patients["procedure_indicator"]
            .value_counts()
            .to_string()
        )

    # Comorbidity
    if "comorbidity_count" in patients.columns:

        add_report()
        add_report("COMORBIDITY COUNT")

        add_report(
            patients["comorbidity_count"]
            .describe()
            .round(2)
            .to_string()
        )


# ================================================================
# 13. PATIENT AGE DISTRIBUTION CHART
# ================================================================

if "patients" in datasets and "age" in datasets["patients"].columns:

    plt.figure(figsize=(10, 6))

    plt.hist(
        datasets["patients"]["age"].dropna(),
        bins=30
    )

    plt.title("Patient Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Number of Patients")

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "01_patient_age_distribution.png",
        dpi=150
    )

    plt.close()


# ================================================================
# 14. PATIENT SEVERITY CHART
# ================================================================

if "patients" in datasets and "severity" in datasets["patients"].columns:

    counts = datasets["patients"]["severity"].value_counts()

    plt.figure(figsize=(10, 6))

    counts.plot(kind="bar")

    plt.title("Patient Severity Distribution")
    plt.xlabel("Severity")
    plt.ylabel("Number of Patients")

    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "02_patient_severity.png",
        dpi=150
    )

    plt.close()


# ================================================================
# 15. PATIENT DEPARTMENT CHART
# ================================================================

if "patients" in datasets and "department" in datasets["patients"].columns:

    counts = datasets["patients"]["department"].value_counts()

    plt.figure(figsize=(12, 6))

    counts.plot(kind="bar")

    plt.title("Patients by Department")
    plt.xlabel("Department")
    plt.ylabel("Number of Patients")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / "03_patients_by_department.png",
        dpi=150
    )

    plt.close()


# ================================================================
# 16. ICU ANALYSIS
# ================================================================

if "encounters" in datasets:

    patients = datasets["encounters"]

    if (
        "ICU_required" in patients.columns
        and "severity" in patients.columns
    ):

        icu_by_severity = pd.crosstab(
            patients["severity"],
            patients["ICU_required"],
            normalize="index"
        ) * 100

        add_report()
        add_report("=" * 80)
        add_report("ICU REQUIREMENT BY SEVERITY")
        add_report("=" * 80)

        add_report(
            icu_by_severity.round(2).to_string()
        )

        plt.figure(figsize=(10, 6))

        icu_by_severity.plot(
            kind="bar",
            ax=plt.gca()
        )

        plt.title("ICU Requirement Percentage by Severity")
        plt.xlabel("Severity")
        plt.ylabel("Percentage")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "04_icu_by_severity.png",
            dpi=150
        )

        plt.close()


# ================================================================
# 17. LENGTH OF STAY ANALYSIS
# ================================================================

if "encounters" in datasets:

    patients = datasets["encounters"]

    if "LOS" in patients.columns:

        add_report()
        add_report("=" * 80)
        add_report("PATIENT LENGTH OF STAY")
        add_report("=" * 80)

        add_report(
            patients["LOS"]
            .describe()
            .round(2)
            .to_string()
        )

        plt.figure(figsize=(10, 6))

        plt.hist(
            patients["LOS"].dropna(),
            bins=30
        )

        plt.title("Length of Stay Distribution")
        plt.xlabel("Length of Stay")
        plt.ylabel("Number of Patients")

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "05_length_of_stay_distribution.png",
            dpi=150
        )

        plt.close()

        # LOS by severity
        if "severity" in patients.columns:

            los_by_severity = (
                patients
                .groupby("severity")["LOS"]
                .mean()
                .sort_values(ascending=False)
            )

            add_report()
            add_report("AVERAGE LOS BY SEVERITY")

            add_report(
                los_by_severity.round(2).to_string()
            )

            plt.figure(figsize=(10, 6))

            los_by_severity.plot(kind="bar")

            plt.title("Average Length of Stay by Severity")
            plt.xlabel("Severity")
            plt.ylabel("Average LOS")

            plt.xticks(rotation=45)

            plt.tight_layout()

            plt.savefig(
                CHART_DIR / "06_los_by_severity.png",
                dpi=150
            )

            plt.close()


# ================================================================
# 18. ADMISSIONS ANALYSIS
# ================================================================

if "admissions" in datasets:

    admissions = datasets["admissions"].copy()

    add_report()
    add_report("=" * 80)
    add_report("ADMISSIONS ANALYSIS")
    add_report("=" * 80)

    if "department" in admissions.columns:

        add_report()
        add_report("ADMISSIONS BY DEPARTMENT")

        add_report(
            admissions["department"]
            .value_counts()
            .to_string()
        )

    if "admission_type" in admissions.columns:

        add_report()
        add_report("ADMISSION TYPE DISTRIBUTION")

        add_report(
            admissions["admission_type"]
            .value_counts()
            .to_string()
        )

    if "ICU_required" in admissions.columns:

        add_report()
        add_report("ICU ADMISSIONS")

        add_report(
            admissions["ICU_required"]
            .value_counts()
            .to_string()
        )


# ================================================================
# 19. ADMISSIONS OVER TIME
# ================================================================

if "admissions" in datasets:

    admissions = datasets["admissions"].copy()

    if "admission_date" in admissions.columns:

        admissions["admission_date"] = pd.to_datetime(
            admissions["admission_date"],
            errors="coerce"
        )

        daily_admissions = (
            admissions
            .dropna(subset=["admission_date"])
            .set_index("admission_date")
            .resample("D")
            .size()
        )

        plt.figure(figsize=(14, 6))

        daily_admissions.plot()

        plt.title("Daily Admissions Over Time")
        plt.xlabel("Date")
        plt.ylabel("Admissions")

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "07_daily_admissions.png",
            dpi=150
        )

        plt.close()

        # Monthly admissions
        monthly_admissions = (
            admissions
            .dropna(subset=["admission_date"])
            .set_index("admission_date")
            .resample("MS")
            .size()
        )

        plt.figure(figsize=(14, 6))

        monthly_admissions.plot()

        plt.title("Monthly Admissions Over Time")
        plt.xlabel("Month")
        plt.ylabel("Admissions")

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "08_monthly_admissions.png",
            dpi=150
        )

        plt.close()


# ================================================================
# 20. BED ANALYSIS
# ================================================================

if "beds" in datasets:

    beds = datasets["beds"]

    add_report()
    add_report("=" * 80)
    add_report("BED ANALYSIS")
    add_report("=" * 80)

    if "status" in beds.columns:

        add_report()
        add_report("BED STATUS")

        add_report(
            beds["status"]
            .value_counts()
            .to_string()
        )

        plt.figure(figsize=(10, 6))

        beds["status"].value_counts().plot(kind="bar")

        plt.title("Bed Status Distribution")
        plt.xlabel("Status")
        plt.ylabel("Number of Beds")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "09_bed_status.png",
            dpi=150
        )

        plt.close()

    if "bed_type" in beds.columns:

        add_report()
        add_report("BED TYPE")

        add_report(
            beds["bed_type"]
            .value_counts()
            .to_string()
        )

        plt.figure(figsize=(10, 6))

        beds["bed_type"].value_counts().plot(kind="bar")

        plt.title("Bed Type Distribution")
        plt.xlabel("Bed Type")
        plt.ylabel("Number of Beds")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "10_bed_type.png",
            dpi=150
        )

        plt.close()

    if "department" in beds.columns:

        add_report()
        add_report("BEDS BY DEPARTMENT")

        add_report(
            beds["department"]
            .value_counts()
            .to_string()
        )


# ================================================================
# 21. STAFF ANALYSIS
# ================================================================

if "staff" in datasets:

    staff = datasets["staff"]

    add_report()
    add_report("=" * 80)
    add_report("STAFF ANALYSIS")
    add_report("=" * 80)

    if "role" in staff.columns:

        add_report()
        add_report("STAFF BY ROLE")

        add_report(
            staff["role"]
            .value_counts()
            .to_string()
        )

        plt.figure(figsize=(10, 6))

        staff["role"].value_counts().plot(kind="bar")

        plt.title("Staff by Role")
        plt.xlabel("Role")
        plt.ylabel("Number of Staff")

        plt.xticks(rotation=45)

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "11_staff_by_role.png",
            dpi=150
        )

        plt.close()

    if "specialty" in staff.columns:

        add_report()
        add_report("STAFF BY SPECIALTY")

        add_report(
            staff["specialty"]
            .value_counts()
            .to_string()
        )

    if "department" in staff.columns:

        add_report()
        add_report("STAFF BY DEPARTMENT")

        add_report(
            staff["department"]
            .value_counts()
            .to_string()
        )


# ================================================================
# 22. APPOINTMENT ANALYSIS
# ================================================================

if "appointments" in datasets:

    appointments = datasets["appointments"].copy()

    add_report()
    add_report("=" * 80)
    add_report("APPOINTMENT ANALYSIS")
    add_report("=" * 80)

    if "attended" in appointments.columns:

        add_report()
        add_report("ATTENDANCE")

        add_report(
            appointments["attended"]
            .value_counts()
            .to_string()
        )

    if "no_show" in appointments.columns:

        add_report()
        add_report("NO-SHOW")

        add_report(
            appointments["no_show"]
            .value_counts()
            .to_string()
        )

    elif "attended" in appointments.columns:

        attended_text = (
            appointments["attended"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        no_show = attended_text.isin(
            ["no", "no_show", "no-show", "false", "0"]
        )

        add_report()
        add_report(
            f"Calculated no-show count: {no_show.sum():,}"
        )

    if "department" in appointments.columns:

        add_report()
        add_report("APPOINTMENTS BY DEPARTMENT")

        add_report(
            appointments["department"]
            .value_counts()
            .to_string()
        )

    if "priority" in appointments.columns:

        add_report()
        add_report("APPOINTMENT PRIORITY")

        add_report(
            appointments["priority"]
            .value_counts()
            .to_string()
        )


# ================================================================
# 23. NO-SHOW RATE
# ================================================================

if "appointments" in datasets:

    appointments = datasets["appointments"].copy()

    no_show_series = None

    if "no_show" in appointments.columns:

        values = (
            appointments["no_show"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        no_show_series = values.isin(
            ["yes", "true", "1", "no_show", "no-show"]
        )

    elif "attended" in appointments.columns:

        values = (
            appointments["attended"]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        no_show_series = values.isin(
            ["no", "false", "0", "no_show", "no-show"]
        )

    if no_show_series is not None:

        no_show_rate = no_show_series.mean() * 100

        add_report()
        add_report("=" * 80)
        add_report("NO-SHOW RATE")
        add_report("=" * 80)

        add_report(
            f"No-show appointments: "
            f"{no_show_series.sum():,}"
        )

        add_report(
            f"Total appointments: "
            f"{len(appointments):,}"
        )

        add_report(
            f"No-show rate: "
            f"{no_show_rate:.2f}%"
        )

        plt.figure(figsize=(8, 6))

        pd.Series(
            {
                "Attended": (~no_show_series).sum(),
                "No-show": no_show_series.sum(),
            }
        ).plot(kind="bar")

        plt.title("Appointment Attendance vs No-Show")
        plt.xlabel("Appointment Outcome")
        plt.ylabel("Number of Appointments")

        plt.xticks(rotation=0)

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "12_appointment_no_show.png",
            dpi=150
        )

        plt.close()


# ================================================================
# 24. DAILY OPERATIONS ANALYSIS
# ================================================================

if "daily_operations" in datasets:

    daily = datasets["daily_operations"].copy()

    add_report()
    add_report("=" * 80)
    add_report("DAILY OPERATIONS ANALYSIS")
    add_report("=" * 80)

    numeric_columns = [
        "admissions",
        "discharges",
        "emergency_cases",
        "ICU_admissions",
        "occupancy",
        "staff_demand",
        "bed_demand",
    ]

    available_numeric = [
        column
        for column in numeric_columns
        if column in daily.columns
    ]

    if available_numeric:

        add_report()

        add_report(
            daily[available_numeric]
            .describe()
            .round(2)
            .to_string()
        )


# ================================================================
# 25. DAILY ADMISSIONS AND EMERGENCY DEMAND
# ================================================================

if "daily_operations" in datasets:

    daily = datasets["daily_operations"].copy()

    if "date" in daily.columns:

        daily["date"] = pd.to_datetime(
            daily["date"],
            errors="coerce"
        )

        daily_total = (
            daily
            .groupby("date")[
                [
                    column
                    for column in [
                        "admissions",
                        "discharges",
                        "emergency_cases",
                        "ICU_admissions",
                    ]
                    if column in daily.columns
                ]
            ]
            .sum()
        )

        if "admissions" in daily_total.columns:

            plt.figure(figsize=(14, 6))

            daily_total["admissions"].plot()

            plt.title("Daily Hospital Admissions")
            plt.xlabel("Date")
            plt.ylabel("Admissions")

            plt.tight_layout()

            plt.savefig(
                CHART_DIR / "13_daily_operations_admissions.png",
                dpi=150
            )

            plt.close()

        if "emergency_cases" in daily_total.columns:

            plt.figure(figsize=(14, 6))

            daily_total["emergency_cases"].plot()

            plt.title("Daily Emergency Cases")
            plt.xlabel("Date")
            plt.ylabel("Emergency Cases")

            plt.tight_layout()

            plt.savefig(
                CHART_DIR / "14_daily_emergency_cases.png",
                dpi=150
            )

            plt.close()

        if "ICU_admissions" in daily_total.columns:

            plt.figure(figsize=(14, 6))

            daily_total["ICU_admissions"].plot()

            plt.title("Daily ICU Admissions")
            plt.xlabel("Date")
            plt.ylabel("ICU Admissions")

            plt.tight_layout()

            plt.savefig(
                CHART_DIR / "15_daily_icu_admissions.png",
                dpi=150
            )

            plt.close()


# ================================================================
# 26. OCCUPANCY ANALYSIS
# ================================================================

if "daily_operations" in datasets:

    daily = datasets["daily_operations"].copy()

    if "occupancy" in daily.columns:

        add_report()
        add_report("=" * 80)
        add_report("OCCUPANCY ANALYSIS")
        add_report("=" * 80)

        add_report(
            daily["occupancy"]
            .describe()
            .round(2)
            .to_string()
        )

        if "date" in daily.columns:

            daily["date"] = pd.to_datetime(
                daily["date"],
                errors="coerce"
            )

            # Department occupancy is a percentage of that department's
            # beds, so the hospital figure must be weighted by bed count.
            if "beds" in datasets and "department" in datasets["beds"].columns:

                department_beds = datasets["beds"].groupby("department").size()

                daily["occupied_beds"] = (
                    daily["occupancy"] / 100
                    * daily["department"].map(department_beds)
                )

                occupancy_by_date = (
                    daily
                    .groupby("date")["occupied_beds"]
                    .sum()
                    .round()
                )

                y_label = f"Occupied beds (of {int(department_beds.sum())})"

            else:

                occupancy_by_date = (
                    daily
                    .groupby("date")["occupancy"]
                    .mean()
                )

                y_label = "Mean department occupancy (%)"

            plt.figure(figsize=(14, 6))

            occupancy_by_date.plot()

            plt.title("Hospital Occupancy Over Time")
            plt.xlabel("Date")
            plt.ylabel(y_label)

            plt.tight_layout()

            plt.savefig(
                CHART_DIR / "16_hospital_occupancy.png",
                dpi=150
            )

            plt.close()


# ================================================================
# 27. BED DEMAND VS STAFF DEMAND
# ================================================================

if "daily_operations" in datasets:

    daily = datasets["daily_operations"].copy()

    if (
        "date" in daily.columns
        and "bed_demand" in daily.columns
        and "staff_demand" in daily.columns
    ):

        daily["date"] = pd.to_datetime(
            daily["date"],
            errors="coerce"
        )

        demand = (
            daily
            .groupby("date")[
                ["bed_demand", "staff_demand"]
            ]
            .mean()
        )

        plt.figure(figsize=(14, 6))

        demand.plot(ax=plt.gca())

        plt.title("Bed Demand vs Staff Demand")
        plt.xlabel("Date")
        plt.ylabel("Demand")

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "17_bed_vs_staff_demand.png",
            dpi=150
        )

        plt.close()


# ================================================================
# 28. SEASONAL DEMAND ANALYSIS
# ================================================================

if "daily_operations" in datasets:

    daily = datasets["daily_operations"].copy()

    if "seasonal_multiplier" in daily.columns:

        add_report()
        add_report("=" * 80)
        add_report("SEASONAL DEMAND ANALYSIS")
        add_report("=" * 80)

        add_report(
            daily["seasonal_multiplier"]
            .describe()
            .round(3)
            .to_string()
        )

        if "month" in daily.columns:

            seasonal = (
                daily
                .groupby("month")[
                    [
                        column
                        for column in [
                            "admissions",
                            "emergency_cases",
                            "ICU_admissions",
                            "occupancy",
                        ]
                        if column in daily.columns
                    ]
                ]
                .mean()
            )

            add_report()
            add_report("AVERAGE OPERATIONS BY MONTH")

            add_report(
                seasonal.round(2).to_string()
            )


# ================================================================
# 29. DAY-OF-WEEK DEMAND ANALYSIS
# ================================================================

if "daily_operations" in datasets:

    daily = datasets["daily_operations"].copy()

    if "day_of_week" in daily.columns:

        columns = [
            column
            for column in [
                "admissions",
                "discharges",
                "emergency_cases",
                "ICU_admissions",
                "occupancy",
            ]
            if column in daily.columns
        ]

        if columns:

            weekday_analysis = (
                daily
                .groupby("day_of_week")[columns]
                .mean()
            )

            add_report()
            add_report("=" * 80)
            add_report("DAY-OF-WEEK DEMAND ANALYSIS")
            add_report("=" * 80)

            add_report(
                weekday_analysis.round(2).to_string()
            )


# ================================================================
# 30. SHIFT ANALYSIS
# ================================================================

if "shifts" in datasets:

    shifts = datasets["shifts"]

    add_report()
    add_report("=" * 80)
    add_report("SHIFT ANALYSIS")
    add_report("=" * 80)

    if "shift" in shifts.columns:

        add_report()
        add_report("SHIFT DISTRIBUTION")

        add_report(
            shifts["shift"]
            .value_counts()
            .to_string()
        )

    if "assigned" in shifts.columns:

        add_report()
        add_report("SHIFT ASSIGNMENT")

        add_report(
            shifts["assigned"]
            .value_counts()
            .to_string()
        )

    if "rest_violation" in shifts.columns:

        add_report()
        add_report("REST VIOLATIONS")

        add_report(
            shifts["rest_violation"]
            .value_counts()
            .to_string()
        )


# ================================================================
# 31. CORRELATION ANALYSIS
# ================================================================

add_report()
add_report("=" * 80)
add_report("CORRELATION ANALYSIS")
add_report("=" * 80)

for name, df in datasets.items():

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if numeric_df.shape[1] < 2:
        continue

    correlation = numeric_df.corr()

    add_report()
    add_report(f"{name.upper()} CORRELATION MATRIX")

    add_report(
        correlation.round(2).to_string()
    )

    # Save correlation matrix
    correlation.to_csv(
        EDA_DIR / f"{name}_correlation.csv"
    )

    # Chart
    plt.figure(
        figsize=(
            max(8, len(correlation.columns) * 0.7),
            max(6, len(correlation.columns) * 0.7)
        )
    )

    plt.imshow(
        correlation,
        aspect="auto"
    )

    plt.colorbar()

    plt.xticks(
        range(len(correlation.columns)),
        correlation.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation.index)),
        correlation.index
    )

    plt.title(
        f"{name.title()} Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CHART_DIR / f"{name}_correlation_matrix.png",
        dpi=150
    )

    plt.close()


# ================================================================
# 32. SEVERITY VS LOS
# ================================================================

if "encounters" in datasets:

    patients = datasets["encounters"]

    if (
        "severity" in patients.columns
        and "LOS" in patients.columns
    ):

        severity_los = (
            patients
            .groupby("severity")["LOS"]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "min",
                    "max",
                ]
            )
            .round(2)
        )

        add_report()
        add_report("=" * 80)
        add_report("SEVERITY VS LENGTH OF STAY")
        add_report("=" * 80)

        add_report(
            severity_los.to_string()
        )


# ================================================================
# 33. ICU VS LOS
# ================================================================

if "encounters" in datasets:

    patients = datasets["encounters"]

    if (
        "ICU_required" in patients.columns
        and "LOS" in patients.columns
    ):

        icu_los = (
            patients
            .groupby("ICU_required")["LOS"]
            .agg(
                [
                    "count",
                    "mean",
                    "median",
                    "min",
                    "max",
                ]
            )
            .round(2)
        )

        add_report()
        add_report("=" * 80)
        add_report("ICU REQUIREMENT VS LENGTH OF STAY")
        add_report("=" * 80)

        add_report(
            icu_los.to_string()
        )


# ================================================================
# 34. COMORBIDITY VS LOS
# ================================================================

if "encounters" in datasets:

    patients = datasets["encounters"]

    if (
        "comorbidity_count" in patients.columns
        and "LOS" in patients.columns
    ):

        comorbidity_los = (
            patients
            .groupby("comorbidity_count")["LOS"]
            .mean()
            .sort_index()
        )

        add_report()
        add_report("=" * 80)
        add_report("COMORBIDITY COUNT VS LOS")
        add_report("=" * 80)

        add_report(
            comorbidity_los.round(2).to_string()
        )

        plt.figure(figsize=(10, 6))

        comorbidity_los.plot()

        plt.title("Average LOS by Comorbidity Count")
        plt.xlabel("Comorbidity Count")
        plt.ylabel("Average LOS")

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "18_comorbidity_vs_los.png",
            dpi=150
        )

        plt.close()


# ================================================================
# 35. DEPARTMENT VS LOS
# ================================================================

if "encounters" in datasets:

    patients = datasets["encounters"]

    if (
        "department" in patients.columns
        and "LOS" in patients.columns
    ):

        department_los = (
            patients
            .groupby("department")["LOS"]
            .mean()
            .sort_values(ascending=False)
        )

        add_report()
        add_report("=" * 80)
        add_report("DEPARTMENT VS LENGTH OF STAY")
        add_report("=" * 80)

        add_report(
            department_los.round(2).to_string()
        )

        plt.figure(figsize=(12, 6))

        department_los.plot(kind="bar")

        plt.title("Average LOS by Department")
        plt.xlabel("Department")
        plt.ylabel("Average LOS")

        plt.xticks(
            rotation=45,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "19_department_vs_los.png",
            dpi=150
        )

        plt.close()


# ================================================================
# 36. AGE VS LOS
# ================================================================

if "encounters" in datasets:

    patients = datasets["encounters"]

    if (
        "age" in patients.columns
        and "LOS" in patients.columns
    ):

        age_los = (
            patients
            .groupby("age")["LOS"]
            .mean()
        )

        plt.figure(figsize=(12, 6))

        age_los.plot()

        plt.title("Average LOS by Patient Age")
        plt.xlabel("Age")
        plt.ylabel("Average LOS")

        plt.tight_layout()

        plt.savefig(
            CHART_DIR / "20_age_vs_los.png",
            dpi=150
        )

        plt.close()


# ================================================================
# 37. EMERGENCY CASES VS ADMISSIONS
# ================================================================

if "daily_operations" in datasets:

    daily = datasets["daily_operations"]

    if (
        "admissions" in daily.columns
        and "emergency_cases" in daily.columns
    ):

        relationship = (
            daily[
                ["admissions", "emergency_cases"]
            ]
            .corr()
        )

        add_report()
        add_report("=" * 80)
        add_report("EMERGENCY CASES VS ADMISSIONS CORRELATION")
        add_report("=" * 80)

        add_report(
            relationship.round(3).to_string()
        )


# ================================================================
# 38. ICU ADMISSIONS VS TOTAL ADMISSIONS
# ================================================================

if "daily_operations" in datasets:

    daily = datasets["daily_operations"]

    if (
        "admissions" in daily.columns
        and "ICU_admissions" in daily.columns
    ):

        correlation = (
            daily["admissions"]
            .corr(daily["ICU_admissions"])
        )

        add_report()
        add_report(
            f"Correlation between total admissions and "
            f"ICU admissions: {correlation:.3f}"
        )


# ================================================================
# 39. BED DEMAND VS OCCUPANCY
# ================================================================

if "daily_operations" in datasets:

    daily = datasets["daily_operations"]

    if (
        "bed_demand" in daily.columns
        and "occupancy" in daily.columns
    ):

        correlation = (
            daily["bed_demand"]
            .corr(daily["occupancy"])
        )

        add_report()
        add_report(
            f"Correlation between bed demand and "
            f"occupancy: {correlation:.3f}"
        )


# ================================================================
# 40. STAFF DEMAND VS ADMISSIONS
# ================================================================

if "daily_operations" in datasets:

    daily = datasets["daily_operations"]

    if (
        "staff_demand" in daily.columns
        and "admissions" in daily.columns
    ):

        correlation = (
            daily["staff_demand"]
            .corr(daily["admissions"])
        )

        add_report()
        add_report(
            f"Correlation between staff demand and "
            f"admissions: {correlation:.3f}"
        )


# ================================================================
# 41. HOSPITAL SNAPSHOT
# ================================================================

if "hospital_snapshot" in datasets:

    snapshot = datasets["hospital_snapshot"]

    add_report()
    add_report("=" * 80)
    add_report("HOSPITAL SNAPSHOT")
    add_report("=" * 80)

    add_report()

    add_report(
        snapshot.to_string(index=False)
    )


# ================================================================
# 42. SAVE REPORT
# ================================================================

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "\n".join(report)
    )


# ================================================================
# 43. FINAL OUTPUT
# ================================================================

print()
print("=" * 80)
print("EDA COMPLETE")
print("=" * 80)

print()
print(f"EDA report saved to:")
print(report_path)

print()
print(f"Charts saved to:")
print(CHART_DIR)

print()
print("Datasets successfully analyzed:")

for name in datasets:

    print(
        f"  - {name}"
    )

print()
print("=" * 80)
print("NEXT STEP: REVIEW THE EDA REPORT")
print("=" * 80)

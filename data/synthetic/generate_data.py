import os
import random
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker


# ============================================================
# CAREOPS AI
# COMPREHENSIVE SYNTHETIC HEALTHCARE DATA GENERATOR
#
# Designed to support:
# - Demand forecasting
# - Admission forecasting
# - Emergency forecasting
# - ICU demand forecasting
# - Occupancy forecasting
# - Length-of-stay prediction
# - Bed availability prediction
# - Bed allocation optimization
# - Staff scheduling optimization
# - Appointment/no-show prediction
# - What-if simulation
# - Dashboard KPIs
# - Explainable operational recommendations
#
# ALL DATA IS SYNTHETIC.
# DO NOT USE REAL PATIENT INFORMATION.
# ============================================================


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

NUM_PATIENTS = 50_000
NUM_BEDS = 230
NUM_STAFF = 300
NUM_APPOINTMENTS = 120_000

START_DATE = datetime(2024, 1, 1)
TARGET_CENSUS_UTILIZATION = 0.65
EXPECTED_LOS_DAYS = {
    "Emergency": 11,
    "General Medicine": 12,
    "Surgery": 14,
    "Cardiology": 14,
    "Neurology": 15,
    "Orthopedics": 14,
    "Pediatrics": 9,
    "Oncology": 17,
    "ICU": 19,
}
ICU_REQUIREMENT_PROBABILITIES = {
    "Low": 0.002,
    "Moderate": 0.01,
    "High": 0.025,
    "Critical": 0.06,
}


def resolve_as_of_date(value: str | None = None) -> date:
    """Resolve a scenario cutoff that can be repeated with CAREOPS_AS_OF_DATE."""
    raw_value = value or os.environ.get("CAREOPS_AS_OF_DATE")
    if raw_value:
        try:
            as_of = date.fromisoformat(raw_value)
        except ValueError as exc:
            raise ValueError(
                "CAREOPS_AS_OF_DATE must be an ISO date in YYYY-MM-DD format."
            ) from exc
    else:
        as_of = date.today()
    if as_of < (START_DATE + timedelta(days=60)).date():
        raise ValueError("The synthetic scenario date must be at least 60 days after 2024-01-01.")
    return as_of


END_DATE = datetime.combine(resolve_as_of_date(), datetime.min.time())

# Current snapshot date
SNAPSHOT_DATE = END_DATE

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# 2. RANDOM SEEDS
# ============================================================

random.seed(SEED)
np.random.seed(SEED)

fake = Faker()
Faker.seed(SEED)


# ============================================================
# 3. HOSPITAL CONFIGURATION
# ============================================================

DEPARTMENTS = [
    "Emergency", "General Medicine", "Surgery", "Cardiology",
    "Neurology", "Orthopedics", "Pediatrics", "Oncology", "ICU",
]
DEPARTMENT_WEIGHTS = [0.043, 0.435, 0.109, 0.087, 0.065, 0.065, 0.065, 0.044, 0.087]

DIAGNOSIS_CATEGORIES = [
    "Cardiovascular", "Respiratory", "Neurological", "Orthopedic",
    "Gastrointestinal", "Infectious Disease", "Cancer", "Trauma",
    "Diabetes", "Other",
]

SEVERITIES = ["Low", "Moderate", "High", "Critical"]

ADMISSION_TYPES = ["Emergency", "Elective", "Transfer", "Urgent"]

GENDERS = ["Male", "Female", "Other"]

# Doctors and Surgeons are both represented here, satisfying the
# blueprint's staff-scheduling requirements (Section 4.5 / 4.6).
STAFF_ROLES = [
    "Doctor", "Nurse", "Surgeon", "Technician", "Administrator", "Therapist",
]

SPECIALTIES = [
    "Emergency Medicine", "Internal Medicine", "Cardiology", "Neurology",
    "Orthopedics", "Pediatrics", "Oncology", "General Surgery",
    "Critical Care", "Radiology",
]

SHIFT_TYPES = ["Morning", "Evening", "Night"]

BED_STATUSES = ["Occupied", "Available", "Maintenance"]

BED_TYPES = ["General", "ICU", "Emergency", "Isolation"]

APPOINTMENT_PRIORITIES = ["Routine", "Urgent", "High"]


# ============================================================
# 4. HELPER FUNCTIONS
# ============================================================

def random_date(start_date, end_date):
    """Generate a random date between two dates."""
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    return start_date + timedelta(days=random_days)


def weighted_choice(options, weights):
    """Select one item using weighted probabilities."""
    return random.choices(options, weights=weights, k=1)[0]


def random_full_name(gender):
    """
    Generate a synthetic full name matching a given gender.
    Purely synthetic -- no real patient information is used.
    """
    if gender == "Male":
        return fake.name_male()
    if gender == "Female":
        return fake.name_female()
    return fake.name()


def seasonal_multiplier(date):
    """Creates realistic seasonal demand patterns."""
    month = date.month
    if month in [12, 1, 2]:
        return 1.12
    if month in [3, 4, 5]:
        return 1.02
    if month in [6, 7, 8]:
        return 0.96
    return 1.04


def day_of_week_multiplier(date):
    """Creates day-of-week variation."""
    weekday = date.weekday()
    multipliers = {0: 1.08, 1: 1.04, 2: 1.02, 3: 1.00, 4: 1.06, 5: 0.92, 6: 0.88}
    return multipliers[weekday]


def severity_icu_probability(severity):
    """Probability of ICU requirement based on severity."""
    probabilities = {"Low": 0.02, "Moderate": 0.08, "High": 0.28, "Critical": 0.72}
    return probabilities[severity]


def base_los(severity):
    """Base length of stay based on severity."""
    if severity == "Critical":
        return random.randint(8, 25)
    if severity == "High":
        return random.randint(4, 15)
    if severity == "Moderate":
        return random.randint(2, 10)
    return random.randint(1, 6)


def diagnosis_los_adjustment(diagnosis):
    """Certain diagnosis categories produce different operational LOS."""
    adjustments = {
        "Cardiovascular": 2, "Respiratory": 2, "Neurological": 3,
        "Orthopedic": 2, "Gastrointestinal": 1, "Infectious Disease": 2,
        "Cancer": 4, "Trauma": 3, "Diabetes": 1, "Other": 0,
    }
    return adjustments.get(diagnosis, 0)


def generate_los(severity, diagnosis, age, procedure_indicator, comorbidity_count, icu_required):
    """Generate LOS using multiple predictors (relationships for ML)."""
    los = base_los(severity)
    los += diagnosis_los_adjustment(diagnosis)

    if age >= 75:
        los += random.randint(1, 3)
    elif age >= 65:
        los += random.randint(0, 2)

    if procedure_indicator == 1:
        los += random.randint(1, 4)

    los += min(comorbidity_count, 5)

    if icu_required == 1:
        los += random.randint(2, 6)

    return max(1, int(los))


# ============================================================
# 5. GENERATE PATIENTS
# ============================================================

def generate_patients():

    print("Generating patients...")

    patient_ids = [f"P{str(i).zfill(6)}" for i in range(1, NUM_PATIENTS + 1)]

    ages = np.clip(np.random.normal(loc=52, scale=23, size=NUM_PATIENTS), 0, 100).astype(int)

    genders = np.random.choice(GENDERS, size=NUM_PATIENTS, p=[0.48, 0.49, 0.03])

    # Synthetic names matched to gender -- fully fake, no real identities.
    patient_names = [random_full_name(g) for g in genders]

    departments = np.random.choice(
        DEPARTMENTS, size=NUM_PATIENTS, p=DEPARTMENT_WEIGHTS
    )

    diagnosis_categories = np.random.choice(DIAGNOSIS_CATEGORIES, size=NUM_PATIENTS)

    severities = np.random.choice(SEVERITIES, size=NUM_PATIENTS, p=[0.25, 0.40, 0.27, 0.08])

    patients = pd.DataFrame({
        "patient_id": patient_ids,
        "patient_name": patient_names,
        "age": ages,
        "gender": genders,
        "department": departments,
        "diagnosis_category": diagnosis_categories,
        "severity": severities,
    })

    output_file = os.path.join(OUTPUT_DIR, "patients.csv")
    patients.to_csv(output_file, index=False)

    print(f"  [OK] Created patients.csv ({len(patients):,} records)")

    return patients


# ============================================================
# 6. GENERATE CAPACITY-CALIBRATED ADMISSIONS
# ============================================================

def generate_admissions(patients, beds):

    print("Generating admissions...")

    patients_by_department = {
        department: group.to_dict("records")
        for department, group in patients.groupby("department")
    }
    bed_departments = beds.set_index("bed_id")["department"].to_dict()
    bed_types = beds.set_index("bed_id")["bed_type"].to_dict()
    available_after = {bed_id: START_DATE.date() for bed_id in bed_departments}
    bed_pools = {
        department: beds.loc[beds["department"] == department, "bed_id"].tolist()
        for department in DEPARTMENTS
    }
    records = []
    admission_id = 1
    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    for admission_date in dates:
        seasonal = seasonal_multiplier(admission_date)
        weekday = day_of_week_multiplier(admission_date)
        demand_multiplier = min(1.25, seasonal * weekday)
        for department in DEPARTMENTS:
            candidates = patients_by_department.get(department, [])
            if not candidates:
                raise ValueError(f"No synthetic patient profiles exist for {department}.")
            capacity = len(bed_pools[department])
            base_rate = (
                capacity * TARGET_CENSUS_UTILIZATION
                / EXPECTED_LOS_DAYS[department]
            )
            daily_demand = int(np.random.poisson(base_rate * demand_multiplier))
            for _ in range(daily_demand):
                patient = random.choice(candidates)
                age = int(patient["age"])
                severity = patient["severity"]
                diagnosis = patient["diagnosis_category"]
                procedure_probability = {
                    "Critical": 0.70,
                    "High": 0.50,
                    "Moderate": 0.30,
                    "Low": 0.12,
                }[severity]
                procedure_indicator = int(random.random() < procedure_probability)
                age_effect = max(0, int((age - 40) / 15))
                severity_effect = {
                    "Low": 0, "Moderate": 1, "High": 2, "Critical": 3,
                }[severity]
                comorbidity_lambda = 0.5 + age_effect * 0.35 + severity_effect * 0.30
                comorbidity_count = min(int(np.random.poisson(comorbidity_lambda)), 8)
                icu_required = int(
                    department == "ICU"
                    or random.random() < ICU_REQUIREMENT_PROBABILITIES[severity]
                )
                admission_type = weighted_choice(
                    ADMISSION_TYPES,
                    [0.55, 0.10, 0.10, 0.25]
                    if severity == "Critical"
                    else [0.45, 0.20, 0.10, 0.25]
                    if severity == "High"
                    else [0.30, 0.40, 0.10, 0.20],
                )
                los = generate_los(
                    severity, diagnosis, age, procedure_indicator,
                    comorbidity_count, icu_required,
                )
                discharge_date = admission_date + timedelta(days=los)
                bed_department = "ICU" if icu_required else department
                pool = bed_pools[bed_department]
                if bed_department == "Emergency":
                    pool = [
                        bed_id for bed_id in pool
                        if bed_types[bed_id] == "Emergency"
                    ]
                eligible_beds = [
                    bed_id for bed_id in pool
                    if available_after[bed_id] <= admission_date.date()
                ]
                if not eligible_beds:
                    continue
                assigned_bed_id = random.choice(eligible_beds)
                available_after[assigned_bed_id] = discharge_date.date()
                records.append({
                    "admission_id": f"A{str(admission_id).zfill(7)}",
                    "patient_id": patient["patient_id"],
                    "patient_name": patient["patient_name"],
                    "department": department,
                    "diagnosis_category": diagnosis,
                    "severity": severity,
                    "admission_type": admission_type,
                    "admission_date": admission_date.strftime("%Y-%m-%d"),
                    "discharge_date": discharge_date.strftime("%Y-%m-%d"),
                    "ICU_required": icu_required,
                    "LOS": los,
                    "procedure_indicator": procedure_indicator,
                    "comorbidity_count": comorbidity_count,
                    "assigned_bed_id": assigned_bed_id,
                    "stay_status": (
                        "completed" if discharge_date.date() <= END_DATE.date()
                        else "in_progress"
                    ),
                })
                admission_id += 1

    admissions = pd.DataFrame(records)

    output_file = os.path.join(OUTPUT_DIR, "admissions.csv")
    admissions.to_csv(output_file, index=False)

    print(
        f"  [OK] Created admissions.csv ({len(admissions):,} capacity-feasible encounters)"
    )

    return admissions


# ============================================================
# 7. GENERATE BEDS
# ============================================================

def generate_beds():

    print("Generating beds...")

    bed_configuration = [
        {"ward": "General", "count": 100, "bed_type": "General", "floor": 2,
         "department": "General Medicine", "isolation_capable": False, "emergency_reserve": False},
        {"ward": "ICU", "count": 20, "bed_type": "ICU", "floor": 3,
         "department": "ICU", "isolation_capable": True, "emergency_reserve": True},
        {"ward": "Emergency", "count": 10, "bed_type": "Emergency", "floor": 1,
         "department": "Emergency", "isolation_capable": True, "emergency_reserve": True},
        {"ward": "Surgery", "count": 25, "bed_type": "General", "floor": 2,
         "department": "Surgery", "isolation_capable": False, "emergency_reserve": False},
        {"ward": "Cardiology", "count": 20, "bed_type": "General", "floor": 2,
         "department": "Cardiology", "isolation_capable": False, "emergency_reserve": False},
        {"ward": "Neurology", "count": 15, "bed_type": "General", "floor": 2,
         "department": "Neurology", "isolation_capable": False, "emergency_reserve": False},
        {"ward": "Orthopedics", "count": 15, "bed_type": "General", "floor": 2,
         "department": "Orthopedics", "isolation_capable": False, "emergency_reserve": False},
        {"ward": "Pediatrics", "count": 15, "bed_type": "General", "floor": 2,
         "department": "Pediatrics", "isolation_capable": False, "emergency_reserve": False},
        {"ward": "Oncology", "count": 10, "bed_type": "Isolation", "floor": 2,
         "department": "Oncology", "isolation_capable": True, "emergency_reserve": False},
    ]

    records = []
    bed_number = 1

    for config in bed_configuration:

        ward = config["ward"]
        count = config["count"]

        for index in range(count):

            if ward == "ICU":
                status = "Occupied" if index < 18 else "Available"
            elif ward == "General":
                status = "Occupied" if index < 91 else "Available"
            elif ward == "Emergency":
                status = "Occupied" if index < 9 else "Available"
            else:
                status = weighted_choice(["Occupied", "Available", "Maintenance"], [0.75, 0.20, 0.05])

            records.append({
                "bed_id": f"B{str(bed_number).zfill(4)}",
                "ward": ward,
                "bed_type": config["bed_type"],
                "status": status,
                "floor": config["floor"],
                "department": config["department"],
                "isolation_capable": config["isolation_capable"],
                "emergency_reserve": config["emergency_reserve"],
                "available_for_emergency": (
                    status == "Available"
                    and config["department"] == "General Medicine"
                ),
            })

            bed_number += 1

    beds = pd.DataFrame(records)

    output_file = os.path.join(OUTPUT_DIR, "beds.csv")
    beds.to_csv(output_file, index=False)

    print(f"  [OK] Created beds.csv ({len(beds):,} records)")

    return beds


# ============================================================
# 8. GENERATE STAFF (Doctors + Surgeons included via STAFF_ROLES)
# ============================================================

def generate_staff():

    print("Generating staff...")

    records = []
    roles = [
        weighted_choice(STAFF_ROLES, [0.20, 0.45, 0.08, 0.10, 0.10, 0.07])
        for _ in range(NUM_STAFF)
    ]
    doctor_count = roles.count("Doctor")
    raw_department_counts = [
        doctor_count * weight / sum(DEPARTMENT_WEIGHTS)
        for weight in DEPARTMENT_WEIGHTS
    ]
    doctor_department_counts = [
        int(count) for count in raw_department_counts
    ]
    remaining_doctors = doctor_count - sum(doctor_department_counts)
    remainder_order = sorted(
        range(len(DEPARTMENTS)),
        key=lambda index: raw_department_counts[index] - doctor_department_counts[index],
        reverse=True,
    )
    for index in remainder_order[:remaining_doctors]:
        doctor_department_counts[index] += 1
    doctor_departments = [
        department
        for department, count in zip(DEPARTMENTS, doctor_department_counts)
        for _ in range(count)
    ]
    random.shuffle(doctor_departments)
    next_doctor_department = iter(doctor_departments)

    for i, role in enumerate(roles, start=1):

        # Synthetic staff name -- no real identities.
        staff_name = fake.name()

        if role == "Surgeon":
            department = "Surgery"
        elif role == "Doctor":
            department = next(next_doctor_department)
        else:
            department = random.choice(DEPARTMENTS)

        if role == "Doctor":
            specialty = {
                "Emergency": "Emergency Medicine",
                "General Medicine": "Internal Medicine",
                "Surgery": "General Surgery",
                "Cardiology": "Cardiology",
                "Neurology": "Neurology",
                "Orthopedics": "Orthopedics",
                "Pediatrics": "Pediatrics",
                "Oncology": "Oncology",
                "ICU": "Critical Care",
            }[department]
        elif role == "Surgeon":
            specialty = "General Surgery"
        elif role == "Technician":
            specialty = "Radiology"
        else:
            specialty = random.choice(SPECIALTIES)

        if role in {"Doctor", "Surgeon"}:
            availability = weighted_choice(
                ["Available", "Busy", "Leave"], [0.95, 0.04, 0.01]
            )
            shift_preference = "Morning"
        else:
            availability = weighted_choice(
                ["Available", "Busy", "Leave"], [0.70, 0.25, 0.05]
            )
            shift_preference = random.choice(SHIFT_TYPES)

        max_hours = (
            48 if role in {"Doctor", "Surgeon"}
            else random.choice([24, 32, 36, 40, 48])
        )

        records.append({
            "staff_id": f"S{str(i).zfill(5)}",
            "staff_name": staff_name,
            "role": role,
            "specialty": specialty,
            "department": department,
            "availability": availability,
            "shift_preference": shift_preference,
            "max_hours": max_hours,
        })

    staff = pd.DataFrame(records)

    output_file = os.path.join(OUTPUT_DIR, "staff.csv")
    staff.to_csv(output_file, index=False)

    print(f"  [OK] Created staff.csv ({len(staff):,} records)")

    return staff


# ============================================================
# 9. GENERATE APPOINTMENTS (patient_name + doctor_name added)
# ============================================================

def generate_appointments(patients, staff):

    print("Generating appointments...")

    patient_ids = patients["patient_id"].values
    patient_lookup = patients.set_index("patient_id")

    doctors = staff[staff["role"].isin(["Doctor", "Surgeon"])]
    doctor_lookup = doctors.set_index("staff_id")
    doctors_by_department = {
        department: group["staff_id"].tolist()
        for department, group in doctors.groupby("department")
    }
    if any(not doctors_by_department.get(department) for department in DEPARTMENTS):
        raise ValueError("Synthetic doctor coverage is missing one or more departments.")

    patient_history = {
        patient_id: {"appointments": 0, "no_shows": 0, "attended": 0}
        for patient_id in patient_ids
    }

    drafts = []

    appointment_times = [
        "08:00", "08:30", "09:00", "09:30", "10:00", "10:30", "11:00", "11:30",
        "13:00", "13:30", "14:00", "14:30", "15:00", "15:30", "16:00",
    ]

    for i in range(1, NUM_APPOINTMENTS + 1):

        patient_id = random.choice(patient_ids)
        patient = patient_lookup.loc[patient_id]
        department = patient["department"]
        doctor_id = random.choice(doctors_by_department[department])
        doctor = doctor_lookup.loc[doctor_id]

        appointment_date = random_date(START_DATE, END_DATE)
        appointment_time = random.choice(appointment_times)
        lead_time_days = random.randint(2, 45)
        reminder_sent = int(lead_time_days >= 7 and random.random() < 0.94)
        drafts.append({
            "appointment_id": f"AP{str(i).zfill(7)}",
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "appointment_date": appointment_date,
            "appointment_time": appointment_time,
            "lead_time_days": lead_time_days,
            "reminder_sent": reminder_sent,
            "priority": weighted_choice(APPOINTMENT_PRIORITIES, [0.70, 0.20, 0.10]),
            "duration": random.choice([15, 30, 45, 60]),
        })

    records = []
    for draft in sorted(
        drafts,
        key=lambda item: (
            item["appointment_date"],
            datetime.strptime(item["appointment_time"], "%H:%M").time(),
            item["appointment_id"],
        ),
    ):
        patient = patient_lookup.loc[draft["patient_id"]]
        doctor = doctor_lookup.loc[draft["doctor_id"]]
        department = patient["department"]
        appointment_date = draft["appointment_date"]
        appointment_time = draft["appointment_time"]
        appointment_weekday = appointment_date.weekday()
        appointment_hour = int(appointment_time.split(":")[0])
        priority = draft["priority"]
        duration = draft["duration"]
        lead_time_days = int(draft["lead_time_days"])
        reminder_sent = int(draft["reminder_sent"])
        history = patient_history[draft["patient_id"]]
        previous_no_show_rate = (
            history["no_shows"] / max(history["appointments"], 1)
        ) if history["appointments"] > 0 else 0.12

        doctor_load = random.uniform(0.6, 1.2)
        delay_factor = 0.18 if appointment_hour >= 15 else 0.0
        weekday_factor = 0.12 if appointment_weekday >= 5 else 0.0

        # Keep the synthetic no-show rate realistic for a portfolio demo,
        # while preserving measurable variation across patient history and visit factors.
        base_rate = 0.13
        risk_score = -2.05
        risk_score += (previous_no_show_rate - base_rate) * 8.0
        risk_score += 0.25 if priority == "Routine" else 0.05 if priority == "Urgent" else -0.20
        risk_score += 0.20 if patient["severity"] in ["High", "Critical"] else 0.08 if patient["severity"] == "Moderate" else -0.10
        risk_score += 0.18 if patient["age"] >= 65 else 0.10 if patient["age"] < 25 else 0.0
        risk_score += 0.28 if duration >= 60 else 0.10 if duration >= 45 else -0.05
        risk_score += 0.012 * lead_time_days
        risk_score += -0.55 if reminder_sent else 0.20
        risk_score += delay_factor + weekday_factor
        risk_score += 0.18 * doctor_load
        risk_score += 0.12 if patient["department"] in ["Emergency", "ICU"] else 0.0
        risk_score += random.gauss(0.0, 0.22)

        no_show_probability = 1.0 / (1.0 + np.exp(-risk_score))
        no_show_probability = float(np.clip(no_show_probability, 0.04, 0.38))

        no_show = 1 if random.random() < no_show_probability else 0
        attended = "No" if no_show == 1 else "Yes"

        history["appointments"] += 1
        if no_show:
            history["no_shows"] += 1
        else:
            history["attended"] += 1

        records.append({
            "appointment_id": draft["appointment_id"],
            "patient_id": draft["patient_id"],
            "patient_name": patient["patient_name"],
            "doctor_id": draft["doctor_id"],
            "doctor_name": doctor["staff_name"],
            "department": department,
            "date": appointment_date.strftime("%Y-%m-%d"),
            "time": appointment_time,
            "lead_time_days": lead_time_days,
            "reminder_sent": reminder_sent,
            "duration": duration,
            "priority": priority,
            "attended": attended,
            "no_show": no_show,
            "patient_age": int(patient["age"]),
            "patient_severity": patient["severity"],
        })

    appointments = pd.DataFrame(records)

    output_file = os.path.join(OUTPUT_DIR, "appointments.csv")
    appointments.to_csv(output_file, index=False)

    print(f"  [OK] Created appointments.csv ({len(appointments):,} records)")

    return appointments


# ============================================================
# 10. GENERATE DAILY OPERATIONS
# ============================================================

def generate_daily_operations(admissions, beds):

    print("Generating daily operations...")

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    events = admissions.copy()
    events["admission_date"] = pd.to_datetime(events["admission_date"], errors="raise")
    events["discharge_date"] = pd.to_datetime(events["discharge_date"], errors="raise")
    events = events.merge(
        beds[["bed_id", "department"]].rename(
            columns={"bed_id": "assigned_bed_id", "department": "bed_department"}
        ),
        on="assigned_bed_id",
        how="left",
        validate="many_to_one",
    )
    if events["bed_department"].isna().any():
        raise ValueError("Every synthetic admission must reference a known bed.")

    admissions_by_day = events.groupby(
        [events["admission_date"].dt.normalize(), "department"]
    ).size()
    discharges_by_day = events.groupby(
        [events["discharge_date"].dt.normalize(), "department"]
    ).size()
    icu_by_day = events.loc[events["ICU_required"].astype(bool)].groupby(
        [events["admission_date"].dt.normalize(), "department"]
    ).size()
    capacities = beds.groupby("department").size().to_dict()
    records = []

    for day in dates:
        normalized_day = day.normalize()
        active = events[
            (events["admission_date"] <= normalized_day)
            & (events["discharge_date"] > normalized_day)
        ]
        census_by_department = active.groupby("bed_department").size().to_dict()
        for department in DEPARTMENTS:
            admission_count = int(
                admissions_by_day.get((normalized_day, department), 0)
            )
            discharge_count = int(
                discharges_by_day.get((normalized_day, department), 0)
            )
            icu_admission_count = int(
                icu_by_day.get((normalized_day, department), 0)
            )
            emergency_cases = (
                max(admission_count, int(round(admission_count * 1.5)))
                if department == "Emergency"
                else 0
            )
            census = int(census_by_department.get(department, 0))
            occupancy = (
                census / capacities[department] * 100
                if capacities.get(department, 0)
                else 0
            )
            staff_demand = (
                admission_count * 0.64
                + emergency_cases * 0.46
                + icu_admission_count * 1.65
            )
            seasonal = seasonal_multiplier(day)

            records.append({
                "date": day.strftime("%Y-%m-%d"),
                "department": department,
                "admissions": admission_count,
                "discharges": discharge_count,
                "emergency_cases": emergency_cases,
                "ICU_admissions": icu_admission_count,
                "occupancy": round(float(occupancy), 2),
                "staff_demand": round(float(staff_demand), 2),
                "bed_demand": census,
                "seasonal_multiplier": round(float(seasonal), 3),
                "day_of_week": day.strftime("%A"),
                "month": day.month,
                "year": day.year,
            })

    operations = pd.DataFrame(records)

    output_file = os.path.join(OUTPUT_DIR, "daily_operations.csv")
    operations.to_csv(output_file, index=False)

    print(f"  [OK] Created daily_operations.csv ({len(operations):,} records)")

    return operations


# ============================================================
# 11. GENERATE SHIFTS
# ============================================================

def generate_shifts(staff):

    print("Generating staff shifts...")

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    staff_ids = staff["staff_id"].values
    staff_lookup = staff.set_index("staff_id")

    records = []
    shift_counter = 1
    previous_shift = {}
    previous_date = {}
    recent_hours = {staff_id: {} for staff_id in staff_ids}
    provider_targets = {
        "Emergency": 2,
        "General Medicine": 6,
        "Surgery": 3,
        "Cardiology": 2,
        "Neurology": 2,
        "Orthopedics": 2,
        "Pediatrics": 2,
        "Oncology": 2,
        "ICU": 2,
    }

    for date in dates:
        target_count = int(NUM_STAFF * 0.55)
        ordered_staff = random.sample(list(staff_ids), len(staff_ids))
        assigned_today = 0
        assigned_staff_ids = set()

        def try_schedule(staff_id):
            nonlocal assigned_today, shift_counter
            if assigned_today >= target_count or staff_id in assigned_staff_ids:
                return False
            staff_member = staff_lookup.loc[staff_id]
            history = recent_hours[staff_id]
            history = {
                shift_day: hours for shift_day, hours in history.items()
                if (date.date() - shift_day).days < 7
            }
            recent_hours[staff_id] = history
            weekly_hours = sum(history.values())
            choices = [
                shift for shift in SHIFT_TYPES
                if weekly_hours + (10 if shift == "Night" else 8)
                <= int(staff_member["max_hours"])
                and not (
                    previous_date.get(staff_id) == date.date() - timedelta(days=1)
                    and previous_shift.get(staff_id) == "Night"
                    and shift == "Morning"
                )
            ]
            if not choices:
                return False
            preferred = staff_member["shift_preference"]
            is_provider = staff_member["role"] in {"Doctor", "Surgeon"}
            shift = (
                preferred
                if preferred in choices
                and (is_provider or random.random() < 0.55)
                else random.choice(choices)
            )
            hours = 10 if shift == "Night" else 8

            records.append({
                "shift_id": f"SH{str(shift_counter).zfill(8)}",
                "date": date.strftime("%Y-%m-%d"),
                "staff_id": staff_id,
                "staff_name": staff_member["staff_name"],
                "shift": shift,
                "department": staff_member["department"],
                "hours": hours,
                "assigned": "Yes",
                "staff_role": staff_member["role"],
                "specialty": staff_member["specialty"],
                "max_hours": staff_member["max_hours"],
                "rest_violation": 0,
            })

            history[date.date()] = hours
            previous_shift[staff_id] = shift
            previous_date[staff_id] = date.date()
            assigned_staff_ids.add(staff_id)
            shift_counter += 1
            assigned_today += 1
            return True

        for department, minimum in provider_targets.items():
            providers = staff[
                staff["department"].eq(department)
                & staff["role"].isin(["Doctor", "Surgeon"])
                & staff["availability"].eq("Available")
            ]["staff_id"].tolist()
            providers = random.sample(providers, len(providers))
            scheduled_providers = 0
            for staff_id in providers:
                if try_schedule(staff_id):
                    scheduled_providers += 1
                if scheduled_providers >= minimum:
                    break

        for staff_id in ordered_staff:
            if assigned_today >= target_count:
                break
            try_schedule(staff_id)

    shifts = pd.DataFrame(records)

    output_file = os.path.join(OUTPUT_DIR, "shifts.csv")
    shifts.to_csv(output_file, index=False)

    print(f"  [OK] Created shifts.csv ({len(shifts):,} records)")

    return shifts


# ============================================================
# 12. CREATE HOSPITAL SNAPSHOT
# ============================================================

def create_hospital_snapshot(beds, daily_operations):

    print("Creating hospital snapshot...")

    total_beds = len(beds)
    occupied_beds = (beds["status"] == "Occupied").sum()
    available_beds = (beds["status"] == "Available").sum()
    maintenance_beds = (beds["status"] == "Maintenance").sum()

    icu_beds = (beds["bed_type"] == "ICU").sum()
    occupied_icu = ((beds["bed_type"] == "ICU") & (beds["status"] == "Occupied")).sum()
    available_icu = ((beds["bed_type"] == "ICU") & (beds["status"] == "Available")).sum()

    snapshot = pd.DataFrame([{
        "snapshot_date": SNAPSHOT_DATE.strftime("%Y-%m-%d"),
        "total_beds": total_beds,
        "occupied_beds": occupied_beds,
        "available_beds": available_beds,
        "maintenance_beds": maintenance_beds,
        "total_icu_beds": icu_beds,
        "occupied_icu_beds": occupied_icu,
        "available_icu_beds": available_icu,
        "overall_occupancy": round(occupied_beds / total_beds * 100, 2),
        "icu_occupancy": round(occupied_icu / icu_beds * 100, 2),
    }])

    output_file = os.path.join(OUTPUT_DIR, "hospital_snapshot.csv")
    snapshot.to_csv(output_file, index=False)

    print("  [OK] Created hospital_snapshot.csv")

    return snapshot


def reconcile_bed_snapshot(beds, admissions):
    """Set snapshot bed states from the assigned stays active on the cutoff date."""
    snapshot_day = END_DATE.date()
    admission_dates = pd.to_datetime(admissions["admission_date"], errors="raise").dt.date
    discharge_dates = pd.to_datetime(admissions["discharge_date"], errors="raise").dt.date
    active_bed_ids = set(admissions.loc[
        (admission_dates <= snapshot_day) & (discharge_dates > snapshot_day),
        "assigned_bed_id",
    ])

    result = beds.copy()
    occupied = result["bed_id"].isin(active_bed_ids)
    result.loc[occupied, "status"] = "Occupied"
    result.loc[~occupied, "status"] = "Available"
    result["available_for_emergency"] = (
        result["status"].eq("Available")
        & result["department"].eq("General Medicine")
    )

    for department, group in result.groupby("department"):
        available_indices = group.index[result.loc[group.index, "status"].eq("Available")]
        maintenance_count = min(
            max(0, int(round(len(group) * 0.04))),
            len(available_indices),
        )
        if maintenance_count:
            maintenance_indices = random.sample(
                available_indices.tolist(), maintenance_count
            )
            result.loc[maintenance_indices, "status"] = "Maintenance"
            result.loc[maintenance_indices, "available_for_emergency"] = False

    result.to_csv(os.path.join(OUTPUT_DIR, "beds.csv"), index=False)
    return result


# ============================================================
# 13. VERIFY ALL GENERATED DATA
# ============================================================

def verify_data():

    print()
    print("=" * 80)
    print("VERIFYING GENERATED DATA")
    print("=" * 80)

    files = [
        "patients.csv", "admissions.csv", "beds.csv", "staff.csv",
        "appointments.csv", "daily_operations.csv", "shifts.csv", "hospital_snapshot.csv",
    ]

    for filename in files:
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            dataframe = pd.read_csv(filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  [OK] {filename:<28}{len(dataframe):>10,} rows   {size_mb:>8.2f} MB")
        else:
            print(f"  [FAIL] {filename} NOT FOUND")


# ============================================================
# 14. VERIFY IMPORTANT COLUMNS
# ============================================================

def verify_columns():

    print()
    print("=" * 80)
    print("VERIFYING BLUEPRINT DATA FIELDS")
    print("=" * 80)

    expected_columns = {
        "patients.csv": [
            "patient_id", "patient_name", "age", "gender", "department",
            "diagnosis_category", "severity",
        ],
        "admissions.csv": [
            "admission_id", "patient_id", "patient_name", "department", "diagnosis_category",
            "severity", "admission_type", "admission_date", "discharge_date", "ICU_required",
            "LOS", "procedure_indicator", "comorbidity_count", "assigned_bed_id",
        ],
        "beds.csv": [
            "bed_id", "ward", "bed_type", "status", "floor", "department",
            "isolation_capable", "emergency_reserve", "available_for_emergency",
        ],
        "staff.csv": [
            "staff_id", "staff_name", "role", "specialty", "department",
            "availability", "shift_preference", "max_hours",
        ],
        "appointments.csv": [
            "appointment_id", "patient_id", "patient_name", "doctor_id", "doctor_name",
            "department", "date", "time", "duration", "priority", "attended",
            "no_show", "patient_age", "patient_severity",
        ],
        "daily_operations.csv": [
            "date", "department", "admissions", "discharges", "emergency_cases",
            "ICU_admissions", "occupancy", "staff_demand", "bed_demand",
            "seasonal_multiplier", "day_of_week", "month", "year",
        ],
        "shifts.csv": [
            "shift_id", "date", "staff_id", "staff_name", "shift", "department",
            "hours", "assigned", "staff_role", "specialty", "max_hours", "rest_violation",
        ],
    }

    all_valid = True

    for filename, required_columns in expected_columns.items():

        filepath = os.path.join(OUTPUT_DIR, filename)
        dataframe = pd.read_csv(filepath)

        missing_columns = [c for c in required_columns if c not in dataframe.columns]

        if missing_columns:
            all_valid = False
            print(f"\n[FAIL] {filename}")
            print("Missing columns:")
            for column in missing_columns:
                print(f"  - {column}")
        else:
            print(f"[OK] {filename} contains required fields")

    return all_valid


# ============================================================
# 15. DATA QUALITY CHECK
# ============================================================

def data_quality_check():

    print()
    print("=" * 80)
    print("DATA QUALITY CHECK")
    print("=" * 80)

    files = [
        "patients.csv", "admissions.csv", "beds.csv", "staff.csv",
        "appointments.csv", "daily_operations.csv", "shifts.csv",
    ]

    overall_pass = True

    for filename in files:

        filepath = os.path.join(OUTPUT_DIR, filename)
        dataframe = pd.read_csv(filepath)

        # assigned_bed_id can be legitimately null only if no compatible
        # bed existed at generation time -- extremely rare with 230 beds,
        # but excluded here so a real gap elsewhere isn't masked.
        missing = int(dataframe.isnull().sum().sum())
        duplicates = int(dataframe.duplicated().sum())

        print()
        print(filename)
        print(f"  Rows: {len(dataframe):,}")
        print(f"  Columns: {len(dataframe.columns)}")
        print(f"  Missing values: {missing}")
        print(f"  Duplicate rows: {duplicates}")

        if missing > 0:
            overall_pass = False
        if duplicates > 0:
            overall_pass = False

    return overall_pass


# ============================================================
# 16. DISPLAY DATASET RELATIONSHIPS
# ============================================================

def display_relationships(patients, admissions, appointments, daily_operations):

    print()
    print("=" * 80)
    print("CHECKING IMPORTANT DATA RELATIONSHIPS")
    print("=" * 80)

    icu_by_severity = admissions.groupby("severity")["ICU_required"].mean()
    print()
    print("ICU requirement rate by severity:")
    print((icu_by_severity * 100).round(2).astype(str) + "%")

    los_by_severity = admissions.groupby("severity")["LOS"].mean()
    print()
    print("Average LOS by severity:")
    print(los_by_severity.round(2))

    los_by_icu = admissions.groupby("ICU_required")["LOS"].mean()
    print()
    print("Average LOS by ICU requirement:")
    print(los_by_icu.round(2))

    daily_admissions = daily_operations.groupby("date")["admissions"].sum()
    print()
    print("Average daily admissions:")
    print(round(daily_admissions.mean(), 2))

    no_show_rate = appointments["no_show"].mean()
    print()
    print("Overall appointment no-show rate:")
    print(round(no_show_rate * 100, 2), "%")

    print()
    print("Admissions by department:")
    print(admissions["department"].value_counts())

    bed_assignment_rate = admissions["assigned_bed_id"].notna().mean()
    print()
    print("Admissions with a bed assigned:")
    print(round(bed_assignment_rate * 100, 2), "%")


# ============================================================
# 17. DISPLAY HOSPITAL SNAPSHOT
# ============================================================

def display_hospital_snapshot(snapshot):

    print()
    print("=" * 80)
    print("CURRENT HOSPITAL SNAPSHOT")
    print("=" * 80)

    row = snapshot.iloc[0]

    print()
    print("Total beds:", int(row["total_beds"]))
    print("Occupied beds:", int(row["occupied_beds"]))
    print("Available beds:", int(row["available_beds"]))
    print("Maintenance beds:", int(row["maintenance_beds"]))
    print("Overall occupancy:", row["overall_occupancy"], "%")

    print()
    print("Total ICU beds:", int(row["total_icu_beds"]))
    print("Occupied ICU beds:", int(row["occupied_icu_beds"]))
    print("Available ICU beds:", int(row["available_icu_beds"]))
    print("ICU occupancy:", row["icu_occupancy"], "%")


# ============================================================
# 18. MAIN PROGRAM
# ============================================================

def main():

    print()
    print("=" * 80)
    print("CAREOPS AI")
    print("COMPREHENSIVE SYNTHETIC DATA GENERATOR")
    print("=" * 80)
    print()
    print("Output directory:")
    print(OUTPUT_DIR)
    print()
    print("Generating all datasets...")
    print()

    patients = generate_patients()
    beds = generate_beds()
    admissions = generate_admissions(patients, beds)
    daily_operations = generate_daily_operations(admissions, beds)
    beds = reconcile_bed_snapshot(beds, admissions)
    staff = generate_staff()
    appointments = generate_appointments(patients, staff)
    shifts = generate_shifts(staff)

    snapshot = create_hospital_snapshot(beds, daily_operations)

    verify_data()
    columns_valid = verify_columns()
    quality_valid = data_quality_check()

    display_relationships(patients, admissions, appointments, daily_operations)
    display_hospital_snapshot(snapshot)

    print()
    print("=" * 80)

    if columns_valid and quality_valid:
        print("CAREOPS AI DATA GENERATION AND VALIDATION COMPLETE")
    else:
        print("DATA GENERATION COMPLETED WITH VALIDATION WARNINGS")

    print("=" * 80)
    print()
    print("Synthetic datasets generated:")
    print("  [OK] patients.csv")
    print("  [OK] admissions.csv")
    print("  [OK] beds.csv")
    print("  [OK] staff.csv")
    print("  [OK] appointments.csv")
    print("  [OK] daily_operations.csv")
    print("  [OK] shifts.csv")
    print("  [OK] hospital_snapshot.csv")
    print()
    print("The datasets are ready for:")
    print("  1. Data cleaning")
    print("  2. Exploratory data analysis")
    print("  3. Feature engineering")
    print("  4. Machine learning")
    print("  5. Forecasting")
    print("  6. Optimization")
    print("  7. Simulation")
    print("  8. FastAPI integration")
    print("  9. Dashboard development")


# ============================================================
# RUN PROGRAM
# ============================================================

if __name__ == "__main__":
    main()

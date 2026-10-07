"""
CAREOPS AI - BED ALLOCATION OPTIMIZER
=====================================

Purpose
-------
Allocate future hospital requests to currently available physical beds
using Google OR-Tools CP-SAT.

IMPORTANT DESIGN RULES
----------------------
1. Source CSV files are READ ONLY.
2. Occupied/unavailable beds are never treated as available.
3. One physical bed can receive at most one patient.
4. One patient can receive at most one bed.
5. ICU-required patients can only use ICU beds.
6. Non-ICU patients cannot consume ICU beds.
7. Isolation-required patients require isolation-capable beds.
8. Department compatibility is enforced.
9. Emergency and ICU reserve beds are protected.
10. Staffing capacity is respected when usable staffing information exists.
11. Patients without a safe feasible assignment remain UNASSIGNED.
12. The solution is independently validated before publication.
13. If validation fails, validated allocation outputs are NOT published.
14. Admission type (emergency/urgent/elective/transfer) is NOT a physical
    bed type.
15. Planning date is propagated consistently to every planning request.

Python:
    3.14+

Required packages:
    pandas
    ortools

Run from project root:
    python optimization\\bed_allocation\\bed_allocation_optimizer.py
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from ortools.sat.python import cp_model


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR = (
    PROJECT_ROOT
    / "optimization"
    / "bed_allocation"
    / "outputs"
)

BEDS_FILE = PROCESSED_DIR / "beds.csv"
ADMISSIONS_FILE = PROCESSED_DIR / "admissions.csv"
DAILY_OPERATIONS_FILE = PROCESSED_DIR / "daily_operations.csv"
STAFF_FILE = PROCESSED_DIR / "staff.csv"
SHIFTS_FILE = PROCESSED_DIR / "shifts.csv"


# Protected reserve targets.
# Effective reserve is:
#     min(configured target, currently available beds of that type)
EMERGENCY_RESERVE_TARGET = 2
ICU_RESERVE_TARGET = 2


# Solver settings.
SOLVER_TIME_LIMIT_SECONDS = 120


# Large penalty for leaving a feasible request unassigned.
# Assignment costs are intentionally much smaller.
UNASSIGNED_PENALTY = 1_000_000


# Admission/request types.
# These are NOT physical bed types.
ADMISSION_TYPES = {
    "emergency",
    "urgent",
    "elective",
    "transfer",
    "routine",
    "scheduled",
    "observation",
    "inpatient",
    "unknown",
}


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def clean_column_name(value: str) -> str:
    """Normalize column names."""
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def normalize_text(value) -> str:
    """Normalize text values for comparison."""
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize_bool(value) -> bool:
    """Convert common representations to boolean."""
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return value

    text = normalize_text(value)

    return text in {
        "true",
        "1",
        "yes",
        "y",
        "required",
        "needed",
        "available",
        "occupied",
        "active",
    }


def first_existing_column(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    """Return the first matching normalized column."""
    normalized = {
        clean_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates:
        key = clean_column_name(candidate)

        if key in normalized:
            return normalized[key]

    return None


def require_file(path: Path) -> None:
    """Fail clearly if a required file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{path}"
        )


def parse_date_series(series: pd.Series) -> pd.Series:
    """Parse dates safely."""
    return pd.to_datetime(
        series,
        errors="coerce",
    )


def normalize_department(value) -> str:
    """Normalize department names."""
    text = normalize_text(value)

    aliases = {
        "general": "general medicine",
        "general medicine": "general medicine",
        "medicine": "general medicine",
        "emergency department": "emergency",
        "ed": "emergency",
        "icu": "icu",
        "intensive care": "icu",
        "intensive care unit": "icu",
    }

    return aliases.get(text, text)


def normalize_bed_type(value) -> str:
    """Normalize physical bed type names."""
    text = normalize_text(value)

    aliases = {
        "general medicine": "general",
        "general": "general",
        "medical": "general",
        "icu": "icu",
        "intensive care": "icu",
        "intensive care unit": "icu",
        "emergency": "emergency",
        "ed": "emergency",
        "surgery": "surgery",
        "cardiology": "cardiology",
        "neurology": "neurology",
        "orthopedics": "orthopedics",
        "orthopaedics": "orthopedics",
        "pediatrics": "pediatrics",
        "paediatrics": "pediatrics",
        "oncology": "oncology",
    }

    return aliases.get(text, text)


def normalize_status(value) -> str:
    """Normalize bed status."""
    return normalize_text(value)


def safe_int(value, default=0) -> int:
    """Convert a value to integer safely."""
    try:
        if pd.isna(value):
            return default

        return int(float(value))
    except Exception:
        return default


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data() -> Dict[str, pd.DataFrame]:
    """Load all required source datasets read-only."""

    print("Loading processed datasets...")

    for path in [
        BEDS_FILE,
        ADMISSIONS_FILE,
        DAILY_OPERATIONS_FILE,
        STAFF_FILE,
        SHIFTS_FILE,
    ]:
        require_file(path)

    beds = pd.read_csv(BEDS_FILE)
    admissions = pd.read_csv(ADMISSIONS_FILE)
    daily_operations = pd.read_csv(DAILY_OPERATIONS_FILE)
    staff = pd.read_csv(STAFF_FILE)
    shifts = pd.read_csv(SHIFTS_FILE)

    print(
        f"Loaded beds.csv: "
        f"{beds.shape[0]:,} rows × {beds.shape[1]} columns"
    )

    print(
        f"Loaded admissions.csv: "
        f"{admissions.shape[0]:,} rows × {admissions.shape[1]} columns"
    )

    print(
        f"Loaded daily_operations.csv: "
        f"{daily_operations.shape[0]:,} rows × "
        f"{daily_operations.shape[1]} columns"
    )

    print(
        f"Loaded staff.csv: "
        f"{staff.shape[0]:,} rows × {staff.shape[1]} columns"
    )

    print(
        f"Loaded shifts.csv: "
        f"{shifts.shape[0]:,} rows × {shifts.shape[1]} columns"
    )

    return {
        "beds": beds,
        "admissions": admissions,
        "daily_operations": daily_operations,
        "staff": staff,
        "shifts": shifts,
    }


# =============================================================================
# BED INVENTORY
# =============================================================================

def prepare_beds(beds: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare physical bed inventory.

    Only explicit available statuses are treated as available.
    """

    df = beds.copy()

    bed_id_col = first_existing_column(
        df,
        [
            "bed_id",
            "id",
            "bed",
        ],
    )

    department_col = first_existing_column(
        df,
        [
            "department",
            "bed_department",
            "unit",
        ],
    )

    bed_type_col = first_existing_column(
        df,
        [
            "bed_type",
            "type",
            "bed_category",
        ],
    )

    status_col = first_existing_column(
        df,
        [
            "status",
            "bed_status",
            "availability_status",
        ],
    )

    isolation_col = first_existing_column(
        df,
        [
            "isolation_capable",
            "bed_isolation_capable",
            "isolation",
            "isolation_available",
        ],
    )

    if bed_id_col is None:
        raise ValueError("beds.csv does not contain a bed ID column.")

    if department_col is None:
        raise ValueError(
            "beds.csv does not contain a department column."
        )

    if bed_type_col is None:
        raise ValueError(
            "beds.csv does not contain a physical bed_type column."
        )

    if status_col is None:
        raise ValueError(
            "beds.csv does not contain a bed status column."
        )

    if isolation_col is None:
        # Safe default: if the source does not explicitly identify
        # isolation capability, treat it as False.
        df["_isolation_capable"] = False
    else:
        df["_isolation_capable"] = df[isolation_col].apply(
            normalize_bool
        )

    df["_bed_id"] = df[bed_id_col].astype(str).str.strip()

    df["_department"] = df[department_col].apply(
        normalize_department
    )

    df["_bed_type"] = df[bed_type_col].apply(
        normalize_bed_type
    )

    df["_status"] = df[status_col].apply(
        normalize_status
    )

    # IMPORTANT:
    # Only explicit available statuses are considered available.
    available_statuses = {
        "available",
        "vacant",
        "open",
        "ready",
        "clean",
        "empty",
    }

    df["_is_available"] = df["_status"].isin(
        available_statuses
    )

    # Unknown statuses are NOT considered available.
    df.loc[
        ~df["_status"].isin(available_statuses),
        "_is_available"
    ] = False

    available = df[df["_is_available"]].copy()

    print()
    print("[1/9] Preparing physical bed inventory...")
    print(f"Physical beds: {len(df):,}")
    print(f"Explicitly available beds: {len(available):,}")
    print(
        f"Not available: "
        f"{len(df) - len(available):,}"
    )

    return df


# =============================================================================
# PLANNING DATE
# =============================================================================

def determine_planning_date(
    daily_operations: pd.DataFrame,
    admissions: pd.DataFrame,
) -> pd.Timestamp:
    """
    Determine the first date after the source operational history.

    Prefer daily_operations because it represents daily hospital operations.
    Fall back to admissions when necessary.
    """

    candidate_dates = []

    daily_date_col = first_existing_column(
        daily_operations,
        [
            "date",
            "operation_date",
            "daily_date",
        ],
    )

    if daily_date_col is not None:
        dates = parse_date_series(
            daily_operations[daily_date_col]
        ).dropna()

        if not dates.empty:
            candidate_dates.append(dates.max())

    admission_date_col = first_existing_column(
        admissions,
        [
            "admission_date",
            "date",
            "admit_date",
            "scheduled_date",
        ],
    )

    if admission_date_col is not None:
        dates = parse_date_series(
            admissions[admission_date_col]
        ).dropna()

        if not dates.empty:
            candidate_dates.append(dates.max())

    if not candidate_dates:
        raise ValueError(
            "Could not determine planning date from source datasets."
        )

    planning_date = max(candidate_dates) + pd.Timedelta(days=1)

    planning_date = pd.Timestamp(planning_date).normalize()

    print()
    print("[2/9] Determining planning date...")
    print(
        f"Planning date: "
        f"{planning_date.strftime('%Y-%m-%d')}"
    )

    return planning_date


# =============================================================================
# REQUEST CREATION
# =============================================================================

def get_column_or_none(
    df: pd.DataFrame,
    candidates: List[str],
) -> Optional[str]:
    return first_existing_column(df, candidates)


def infer_bool_from_row(
    row: pd.Series,
    candidates: List[str],
) -> bool:
    for candidate in candidates:
        column = first_existing_column(
            pd.DataFrame([row]),
            [candidate],
        )

        if column is not None:
            return normalize_bool(row[column])

    return False


def infer_admission_type(
    row: pd.Series,
    column: Optional[str],
) -> str:
    """
    Admission type is a request priority/type.

    It is deliberately NOT treated as a physical bed type.
    """

    if column is None:
        return "unknown"

    value = normalize_text(row[column])

    if not value:
        return "unknown"

    return value


def infer_required_bed_type(
    row: pd.Series,
    department: str,
    icu_required: bool,
    emergency_required: bool,
) -> str:
    """
    Determine physical bed requirement.

    IMPORTANT:
    - admission_type is NOT used here.
    - department determines normal physical bed type.
    - ICU requirement overrides everything.
    - Emergency requirement maps to emergency bed.
    """

    if icu_required:
        return "icu"

    if emergency_required:
        return "emergency"

    department_to_bed_type = {
        "general medicine": "general",
        "surgery": "surgery",
        "cardiology": "cardiology",
        "neurology": "neurology",
        "orthopedics": "orthopedics",
        "pediatrics": "pediatrics",
        "oncology": "oncology",
        "icu": "icu",
        "emergency": "emergency",
    }

    return department_to_bed_type.get(
        department,
        department,
    )


def calculate_priority(
    admission_type: str,
    icu_required: bool,
    emergency_required: bool,
) -> int:
    """
    Higher value = more urgent.

    This is used in the optimization objective.
    """

    if icu_required:
        return 5

    if emergency_required:
        return 5

    mapping = {
        "emergency": 5,
        "urgent": 4,
        "transfer": 3,
        "routine": 2,
        "scheduled": 2,
        "elective": 1,
        "observation": 2,
        "inpatient": 2,
        "unknown": 1,
    }

    return mapping.get(
        admission_type,
        1,
    )


def build_requests(
    admissions: pd.DataFrame,
    planning_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build planning requests.

    The request planning_date is ALWAYS the actual planning date.

    Admission date is retained separately when available.
    """

    df = admissions.copy()

    patient_id_col = get_column_or_none(
        df,
        [
            "patient_id",
            "patient",
            "person_id",
        ],
    )

    if patient_id_col is None:
        raise ValueError(
            "admissions.csv does not contain patient_id."
        )

    department_col = get_column_or_none(
        df,
        [
            "department",
            "admitting_department",
            "service",
            "unit",
        ],
    )

    if department_col is None:
        raise ValueError(
            "admissions.csv does not contain department."
        )

    admission_date_col = get_column_or_none(
        df,
        [
            "admission_date",
            "admit_date",
            "date",
        ],
    )

    admission_type_col = get_column_or_none(
        df,
        [
            "admission_type",
            "admission_priority",
            "visit_type",
            "priority_type",
            "type",
        ],
    )

    icu_col = get_column_or_none(
        df,
        [
            "icu_required",
            "requires_icu",
            "icu",
            "icu_need",
        ],
    )

    emergency_col = get_column_or_none(
        df,
        [
            "emergency_required",
            "requires_emergency",
            "emergency",
            "ed_required",
        ],
    )

    isolation_col = get_column_or_none(
        df,
        [
            "isolation_required",
            "requires_isolation",
            "isolation",
        ],
    )

    # Parse admission dates.
    if admission_date_col is not None:
        df["_admission_date"] = parse_date_series(
            df[admission_date_col]
        )
    else:
        df["_admission_date"] = pd.NaT

    # Use records on or before planning date.
    historical = df[
        df["_admission_date"].notna()
        & (
            df["_admission_date"]
            < planning_date
        )
    ].copy()

    # Prefer recent records because they represent current demand.
    if not historical.empty:
        latest_source_date = historical[
            "_admission_date"
        ].max()

        recent = historical[
            historical["_admission_date"]
            >= latest_source_date - pd.Timedelta(days=30)
        ].copy()

        if len(recent) >= 146:
            candidates = recent
        else:
            candidates = historical
    else:
        candidates = df.copy()

    # We deliberately create a manageable planning horizon.
    # If fewer than 146 records exist, use all available candidates.
    REQUEST_LIMIT = 146

    if len(candidates) > REQUEST_LIMIT:
        candidates = candidates.sample(
            n=REQUEST_LIMIT,
            random_state=42,
        )

    candidates = candidates.reset_index(drop=True)

    requests = []

    for index, row in candidates.iterrows():

        patient_id = str(
            row[patient_id_col]
        ).strip()

        department = normalize_department(
            row[department_col]
        )

        if not department:
            department = "unknown"

        if icu_col is not None:
            icu_required = normalize_bool(
                row[icu_col]
            )
        else:
            # ICU department itself requires ICU.
            icu_required = (
                department == "icu"
            )

        if emergency_col is not None:
            emergency_required = normalize_bool(
                row[emergency_col]
            )
        else:
            emergency_required = (
                department == "emergency"
            )

        if isolation_col is not None:
            isolation_required = normalize_bool(
                row[isolation_col]
            )
        else:
            isolation_required = False

        admission_type = infer_admission_type(
            row,
            admission_type_col,
        )

        required_bed_type = infer_required_bed_type(
            row=row,
            department=department,
            icu_required=icu_required,
            emergency_required=emergency_required,
        )

        priority = calculate_priority(
            admission_type=admission_type,
            icu_required=icu_required,
            emergency_required=emergency_required,
        )

        requests.append(
            {
                "request_id": f"REQ_{index + 1:05d}",
                "patient_id": patient_id,

                # IMPORTANT:
                # This is the actual planning date.
                "planning_date": planning_date.strftime(
                    "%Y-%m-%d"
                ),

                "admission_date": (
                    row["_admission_date"].strftime(
                        "%Y-%m-%d"
                    )
                    if pd.notna(
                        row["_admission_date"]
                    )
                    else ""
                ),

                "department": department,

                "original_department": department,

                # Physical bed type, NOT admission type.
                "required_bed_type": required_bed_type,

                # Keep admission type separately.
                "admission_type": admission_type,

                "icu_required": icu_required,
                "emergency_required": emergency_required,
                "isolation_required": isolation_required,

                "priority": priority,
            }
        )

    requests_df = pd.DataFrame(requests)

    # Deterministic ordering.
    requests_df = requests_df.sort_values(
        by=[
            "priority",
            "request_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    # Reassign deterministic request IDs after sorting.
    requests_df["request_id"] = [
        f"REQ_{i + 1:05d}"
        for i in range(len(requests_df))
    ]

    print()
    print("[3/9] Building future planning requests...")
    print(
        f"Planning requests created: "
        f"{len(requests_df):,}"
    )

    print()
    print("Request physical bed type distribution:")

    if not requests_df.empty:
        print(
            requests_df[
                "required_bed_type"
            ].value_counts().to_string()
        )

    print()
    print("Admission type distribution:")

    if not requests_df.empty:
        print(
            requests_df[
                "admission_type"
            ].value_counts().to_string()
        )

    return requests_df


# =============================================================================
# STAFFING
# =============================================================================

def load_staff_capacity(
    staff: pd.DataFrame,
    shifts: pd.DataFrame,
    planning_date: pd.Timestamp,
) -> Dict[str, int]:
    """
    Estimate usable staffing capacity.

    IMPORTANT:
    Staffing is not used to arbitrarily make all assignments impossible.

    We first try to use shifts on the planning date.

    If usable numeric staffing counts are available, they are converted
    into a conservative assignment capacity.

    If staffing data is not usable, no hard staffing bottleneck is imposed.
    """

    capacity: Dict[str, int] = {}

    shift_date_col = first_existing_column(
        shifts,
        [
            "date",
            "shift_date",
            "work_date",
        ],
    )

    shift_department_col = first_existing_column(
        shifts,
        [
            "department",
            "unit",
            "service",
        ],
    )

    # -------------------------------------------------------------------------
    # METHOD 1: SHIFTS
    # -------------------------------------------------------------------------

    if (
        shift_date_col is not None
        and shift_department_col is not None
    ):
        shift_dates = parse_date_series(
            shifts[shift_date_col]
        )

        exact = shifts[
            shift_dates.dt.normalize()
            == planning_date
        ].copy()

        if not exact.empty:

            exact["_department"] = exact[
                shift_department_col
            ].apply(
                normalize_department
            )

            # Try common numeric staffing fields.
            staff_count_col = first_existing_column(
                exact,
                [
                    "staff_count",
                    "staff_required",
                    "staff",
                    "headcount",
                    "employees",
                    "number_of_staff",
                ],
            )

            if staff_count_col is not None:

                grouped = (
                    exact.groupby(
                        "_department"
                    )[staff_count_col]
                    .apply(
                        lambda s: pd.to_numeric(
                            s,
                            errors="coerce",
                        ).fillna(0).sum()
                    )
                )

                for department, value in grouped.items():

                    count = max(
                        0,
                        int(
                            math.floor(
                                float(value)
                            )
                        ),
                    )

                    if count > 0:
                        # One staffed shift position is treated as
                        # support for multiple bed assignments.
                        #
                        # This prevents the previous model from
                        # accidentally reducing capacity to zero.
                        capacity[
                            department
                        ] = max(
                            capacity.get(
                                department,
                                0,
                            ),
                            count * 4,
                        )

    # -------------------------------------------------------------------------
    # METHOD 2: STAFF MASTER DATA
    # -------------------------------------------------------------------------

    staff_department_col = first_existing_column(
        staff,
        [
            "department",
            "unit",
            "service",
        ],
    )

    if staff_department_col is not None:

        staff_df = staff.copy()

        staff_df["_department"] = staff_df[
            staff_department_col
        ].apply(
            normalize_department
        )

        grouped = (
            staff_df[
                "_department"
            ]
            .value_counts()
        )

        for department, count in grouped.items():

            count = int(count)

            if count > 0:

                capacity[
                    department
                ] = max(
                    capacity.get(
                        department,
                        0,
                    ),
                    count * 4,
                )

    print()
    print("[4/9] Preparing staffing capacity...")

    if capacity:

        print(
            "Observed staffing capacity loaded "
            "for departments:"
        )

        for department in sorted(
            capacity
        ):
            print(
                f"  {department}: "
                f"{capacity[department]}"
            )

    else:
        print(
            "No usable staffing capacity detected."
        )

        print(
            "No hard staffing bottleneck will "
            "be imposed."
        )

    return capacity


# =============================================================================
# COMPATIBILITY
# =============================================================================

def department_match(
    request_department: str,
    bed_department: str,
    bed_type: str,
) -> bool:
    """
    Department compatibility.

    Normally a department patient uses the corresponding department bed.

    ICU and Emergency are handled explicitly.
    """

    request_department = normalize_department(
        request_department
    )

    bed_department = normalize_department(
        bed_department
    )

    bed_type = normalize_bed_type(
        bed_type
    )

    # ICU is strict.
    if request_department == "icu":
        return bed_type == "icu"

    # Emergency is strict when the request is Emergency.
    if request_department == "emergency":
        return (
            bed_type == "emergency"
            or bed_department == "emergency"
        )

    # Normal department-specific matching.
    department_to_bed_type = {
        "general medicine": "general",
        "surgery": "surgery",
        "cardiology": "cardiology",
        "neurology": "neurology",
        "orthopedics": "orthopedics",
        "pediatrics": "pediatrics",
        "oncology": "oncology",
    }

    expected = department_to_bed_type.get(
        request_department
    )

    if expected is None:
        return (
            request_department
            == bed_department
        )

    return (
        bed_type == expected
        or bed_department == request_department
    )


def bed_is_compatible(
    request: pd.Series,
    bed: pd.Series,
) -> bool:
    """Apply all patient/bed compatibility rules."""

    request_department = normalize_department(
        request["department"]
    )

    bed_department = normalize_department(
        bed["_department"]
    )

    bed_type = normalize_bed_type(
        bed["_bed_type"]
    )

    # -------------------------------------------------------------------------
    # ICU RULES
    # -------------------------------------------------------------------------

    icu_required = bool(
        request["icu_required"]
    )

    if icu_required:
        if bed_type != "icu":
            return False

    else:
        # Non-ICU patients cannot consume ICU beds.
        if bed_type == "icu":
            return False

    # -------------------------------------------------------------------------
    # EMERGENCY RULE
    # -------------------------------------------------------------------------

    emergency_required = bool(
        request["emergency_required"]
    )

    if emergency_required:

        if (
            bed_type != "emergency"
            and bed_department != "emergency"
        ):
            return False

    # -------------------------------------------------------------------------
    # ISOLATION RULE
    # -------------------------------------------------------------------------

    if bool(
        request["isolation_required"]
    ):

        if not bool(
            bed["_isolation_capable"]
        ):
            return False

    # -------------------------------------------------------------------------
    # DEPARTMENT RULE
    # -------------------------------------------------------------------------

    if not department_match(
        request_department=request_department,
        bed_department=bed_department,
        bed_type=bed_type,
    ):
        return False

    # -------------------------------------------------------------------------
    # PHYSICAL BED TYPE RULE
    # -------------------------------------------------------------------------

    required_bed_type = normalize_bed_type(
        request["required_bed_type"]
    )

    # ICU and Emergency were already handled.
    # For normal requests, required physical type must match.
    if required_bed_type not in {
        "",
        "unknown",
        "emergency",
        "icu",
    }:

        if bed_type != required_bed_type:

            # Allow department-compatible bed records where
            # the physical bed type itself is represented differently.
            if bed_department != request_department:
                return False

    return True


def build_compatibility(
    requests: pd.DataFrame,
    beds: pd.DataFrame,
) -> Dict[str, List[str]]:
    """
    Return compatible available bed IDs for each request.
    """

    available_beds = beds[
        beds["_is_available"]
    ].copy()

    compatibility: Dict[
        str,
        List[str]
    ] = {}

    total_pairs = 0
    zero_compatible = 0

    for _, request in requests.iterrows():

        request_id = request[
            "request_id"
        ]

        compatible_beds = []

        for _, bed in available_beds.iterrows():

            if bed_is_compatible(
                request,
                bed,
            ):
                compatible_beds.append(
                    bed["_bed_id"]
                )

        compatibility[
            request_id
        ] = compatible_beds

        total_pairs += len(
            compatible_beds
        )

        if not compatible_beds:
            zero_compatible += 1

    print()
    print(
        "[5/9] Checking patient/bed compatibility..."
    )

    print(
        f"Total compatible patient-bed pairs: "
        f"{total_pairs:,}"
    )

    print(
        f"Requests with zero compatible beds: "
        f"{zero_compatible:,}"
    )

    return compatibility


# =============================================================================
# RESERVE CALCULATION
# =============================================================================

def available_beds_by_type(
    beds: pd.DataFrame,
    bed_type: str,
) -> List[str]:

    normalized_type = normalize_bed_type(
        bed_type
    )

    result = beds[
        beds["_is_available"]
        & (
            beds["_bed_type"]
            == normalized_type
        )
    ]

    return result[
        "_bed_id"
    ].astype(str).tolist()


# =============================================================================
# CP-SAT MODEL
# =============================================================================

def build_optimizer(
    requests: pd.DataFrame,
    beds: pd.DataFrame,
    compatibility: Dict[str, List[str]],
    staffing_capacity: Dict[str, int],
):
    """
    Build CP-SAT optimization model.

    Objective:
        1. Maximize safe assignments.
        2. Prioritize higher-priority requests.
        3. Prefer department-native beds.
        4. Avoid unnecessary bed movement/cost.

    OR-Tools CP-SAT supports Boolean assignment variables and
    at-most-one / exactly-one style constraints. The model is
    intentionally explicit so validation can independently check it.
    """

    model = cp_model.CpModel()

    assignment_vars = {}

    # -------------------------------------------------------------------------
    # CREATE VARIABLES
    # -------------------------------------------------------------------------

    for _, request in requests.iterrows():

        request_id = request[
            "request_id"
        ]

        for bed_id in compatibility.get(
            request_id,
            [],
        ):

            variable_name = (
                f"assign_"
                f"{request_id}_"
                f"{bed_id}"
            )

            assignment_vars[
                (request_id, bed_id)
            ] = model.new_bool_var(
                variable_name
            )

    # -------------------------------------------------------------------------
    # PHYSICAL BED: AT MOST ONE PATIENT
    # -------------------------------------------------------------------------

    for bed_id in beds[
        beds["_is_available"]
    ]["_bed_id"].astype(str):

        variables = [
            variable
            for (
                (request_id, candidate_bed_id),
                variable,
            ) in assignment_vars.items()
            if candidate_bed_id == bed_id
        ]

        if variables:
            model.add_at_most_one(
                variables
            )

    # -------------------------------------------------------------------------
    # PATIENT/REQUEST: AT MOST ONE BED
    # -------------------------------------------------------------------------

    for request_id in requests[
        "request_id"
    ]:

        variables = [
            variable
            for (
                (candidate_request_id, bed_id),
                variable,
            ) in assignment_vars.items()
            if candidate_request_id
            == request_id
        ]

        if variables:
            model.add_at_most_one(
                variables
            )

    # -------------------------------------------------------------------------
    # STAFFING CAPACITY
    # -------------------------------------------------------------------------

    # Only impose staffing constraints where usable
    # positive staffing capacity was actually observed.
    #
    # This is important:
    # absence of staffing data must NOT automatically mean
    # zero staffing capacity.
    for department, capacity in staffing_capacity.items():

        if capacity <= 0:
            continue

        variables = []

        for _, request in requests.iterrows():

            if (
                normalize_department(
                    request["department"]
                )
                != normalize_department(
                    department
                )
            ):
                continue

            request_id = request[
                "request_id"
            ]

            for (
                (candidate_request_id, bed_id),
                variable,
            ) in assignment_vars.items():

                if (
                    candidate_request_id
                    == request_id
                ):
                    variables.append(
                        variable
                    )

        if variables:

            model.add(
                sum(variables)
                <= int(capacity)
            )

    # -------------------------------------------------------------------------
    # RESERVES
    # -------------------------------------------------------------------------

    emergency_available = len(
        available_beds_by_type(
            beds,
            "emergency",
        )
    )

    icu_available = len(
        available_beds_by_type(
            beds,
            "icu",
        )
    )

    emergency_reserve_required = min(
        EMERGENCY_RESERVE_TARGET,
        emergency_available,
    )

    icu_reserve_required = min(
        ICU_RESERVE_TARGET,
        icu_available,
    )

    emergency_variables = []

    icu_variables = []

    for (
        (request_id, bed_id),
        variable,
    ) in assignment_vars.items():

        bed_row = beds[
            beds["_bed_id"].astype(str)
            == str(bed_id)
        ]

        if bed_row.empty:
            continue

        bed_type = normalize_bed_type(
            bed_row.iloc[0]["_bed_type"]
        )

        if bed_type == "emergency":
            emergency_variables.append(
                variable
            )

        if bed_type == "icu":
            icu_variables.append(
                variable
            )

    # Only the beds beyond the protected reserve can be used.
    emergency_max_usable = max(
        0,
        emergency_available
        - emergency_reserve_required,
    )

    icu_max_usable = max(
        0,
        icu_available
        - icu_reserve_required,
    )

    if emergency_variables:
        model.add(
            sum(emergency_variables)
            <= emergency_max_usable
        )

    if icu_variables:
        model.add(
            sum(icu_variables)
            <= icu_max_usable
        )

    # -------------------------------------------------------------------------
    # OBJECTIVE
    # -------------------------------------------------------------------------

    objective_terms = []

    bed_lookup = {
        str(row["_bed_id"]): row
        for _, row in beds.iterrows()
    }

    for (
        (request_id, bed_id),
        variable,
    ) in assignment_vars.items():

        request = requests[
            requests["request_id"]
            == request_id
        ].iloc[0]

        bed = bed_lookup[
            str(bed_id)
        ]

        priority = safe_int(
            request["priority"],
            1,
        )

        request_department = normalize_department(
            request["department"]
        )

        bed_department = normalize_department(
            bed["_department"]
        )

        bed_type = normalize_bed_type(
            bed["_bed_type"]
        )

        # Large positive value means strong preference to assign.
        #
        # Priority is important, but assignment itself is overwhelmingly
        # more valuable than leaving a compatible patient unassigned.
        assignment_reward = (
            UNASSIGNED_PENALTY
            + priority * 10_000
        )

        # Prefer a department-native bed.
        department_bonus = 2_000

        if (
            bed_department
            == request_department
        ):
            assignment_reward += (
                department_bonus
            )

        # ICU/Emergency requests receive additional preference.
        if (
            bool(request["icu_required"])
            and bed_type == "icu"
        ):
            assignment_reward += 5_000

        if (
            bool(request["emergency_required"])
            and bed_type == "emergency"
        ):
            assignment_reward += 5_000

        objective_terms.append(
            assignment_reward
            * variable
        )

    model.maximize(
        sum(objective_terms)
    )

    print()
    print(
        "[6/9] Building OR-Tools CP-SAT model..."
    )

    print(
        f"Assignment variables: "
        f"{len(assignment_vars):,}"
    )

    print(
        f"Effective Emergency reserve: "
        f"{emergency_reserve_required}"
    )

    print(
        f"Effective ICU reserve: "
        f"{icu_reserve_required}"
    )

    return (
        model,
        assignment_vars,
        {
            "emergency_available": emergency_available,
            "emergency_reserve_required":
                emergency_reserve_required,
            "emergency_max_usable":
                emergency_max_usable,
            "icu_available": icu_available,
            "icu_reserve_required":
                icu_reserve_required,
            "icu_max_usable":
                icu_max_usable,
        },
    )


# =============================================================================
# SOLVE
# =============================================================================

def solve_model(
    model: cp_model.CpModel,
):
    """Solve CP-SAT model."""

    print()
    print(
        "[7/9] Solving optimization model..."
    )

    solver = cp_model.CpSolver()

    solver.parameters.max_time_in_seconds = (
        SOLVER_TIME_LIMIT_SECONDS
    )

    solver.parameters.num_search_workers = 8

    status = solver.solve(
        model
    )

    status_name = (
        solver.status_name(status)
    )

    print(
        f"Solver status: "
        f"{status_name}"
    )

    if status not in (
        cp_model.OPTIMAL,
        cp_model.FEASIBLE,
    ):
        raise RuntimeError(
            "CP-SAT did not return a usable solution. "
            f"Status: {status_name}"
        )

    return solver, status, status_name


# =============================================================================
# EXTRACT SOLUTION
# =============================================================================

def extract_solution(
    requests: pd.DataFrame,
    beds: pd.DataFrame,
    compatibility: Dict[str, List[str]],
    assignment_vars,
    solver,
    planning_date: pd.Timestamp,
) -> pd.DataFrame:
    """Convert solver output into allocation records."""

    bed_lookup = {
        str(row["_bed_id"]): row
        for _, row in beds.iterrows()
    }

    records = []

    for _, request in requests.iterrows():

        request_id = request[
            "request_id"
        ]

        selected_bed = None

        for bed_id in compatibility.get(
            request_id,
            [],
        ):

            variable = assignment_vars.get(
                (request_id, bed_id)
            )

            if variable is not None:

                if solver.value(
                    variable
                ) == 1:

                    selected_bed = str(
                        bed_id
                    )

                    break

        if selected_bed is not None:

            bed = bed_lookup[
                selected_bed
            ]

            records.append(
                {
                    "request_id":
                        request_id,

                    "patient_id":
                        request["patient_id"],

                    "planning_date":
                        planning_date.strftime(
                            "%Y-%m-%d"
                        ),

                    "admission_date":
                        request.get(
                            "admission_date",
                            "",
                        ),

                    "department":
                        request["department"],

                    "original_department":
                        request[
                            "original_department"
                        ],

                    "required_bed_type":
                        request[
                            "required_bed_type"
                        ],

                    "admission_type":
                        request[
                            "admission_type"
                        ],

                    "icu_required":
                        request[
                            "icu_required"
                        ],

                    "emergency_required":
                        request[
                            "emergency_required"
                        ],

                    "isolation_required":
                        request[
                            "isolation_required"
                        ],

                    "priority":
                        request["priority"],

                    "allocation_status":
                        "ASSIGNED",

                    "bed_id":
                        selected_bed,

                    "bed_department":
                        bed["_department"],

                    "bed_type":
                        bed["_bed_type"],

                    "bed_status":
                        bed["_status"],

                    "bed_isolation_capable":
                        bed[
                            "_isolation_capable"
                        ],

                    "reason":
                        "Safe compatible bed assigned by CP-SAT.",

                    "objective_cost":
                        0,
                }
            )

        else:

            records.append(
                {
                    "request_id":
                        request_id,

                    "patient_id":
                        request["patient_id"],

                    "planning_date":
                        planning_date.strftime(
                            "%Y-%m-%d"
                        ),

                    "admission_date":
                        request.get(
                            "admission_date",
                            "",
                        ),

                    "department":
                        request["department"],

                    "original_department":
                        request[
                            "original_department"
                        ],

                    "required_bed_type":
                        request[
                            "required_bed_type"
                        ],

                    "admission_type":
                        request[
                            "admission_type"
                        ],

                    "icu_required":
                        request[
                            "icu_required"
                        ],

                    "emergency_required":
                        request[
                            "emergency_required"
                        ],

                    "isolation_required":
                        request[
                            "isolation_required"
                        ],

                    "priority":
                        request["priority"],

                    "allocation_status":
                        "UNASSIGNED",

                    "bed_id":
                        "",

                    "bed_department":
                        "",

                    "bed_type":
                        "",

                    "bed_status":
                        "",

                    "bed_isolation_capable":
                        False,

                    "reason":
                        "No safe compatible bed was available "
                        "under the optimization constraints.",

                    "objective_cost":
                        UNASSIGNED_PENALTY,
                }
            )

    return pd.DataFrame(records)


# =============================================================================
# INDEPENDENT VALIDATION
# =============================================================================

def validate_solution(
    allocation: pd.DataFrame,
    beds: pd.DataFrame,
    requests: pd.DataFrame,
    reserve_info: Dict[str, int],
) -> Tuple[bool, List[str]]:
    """
    Independent validation.

    This validation does NOT trust the optimizer's internal constraints.
    """

    errors = []

    # -------------------------------------------------------------------------
    # BASIC COUNTS
    # -------------------------------------------------------------------------

    if len(allocation) != len(requests):

        errors.append(
            "Allocation row count does not match request count."
        )

    # -------------------------------------------------------------------------
    # ASSIGNED ROWS
    # -------------------------------------------------------------------------

    assigned = allocation[
        allocation[
            "allocation_status"
        ]
        == "ASSIGNED"
    ].copy()

    # -------------------------------------------------------------------------
    # ONE PATIENT -> ONE BED
    # -------------------------------------------------------------------------

    duplicate_patients = (
        assigned[
            "patient_id"
        ]
        .astype(str)
        .duplicated()
    )

    if duplicate_patients.any():

        errors.append(
            "A patient was assigned to more than one bed."
        )

    # -------------------------------------------------------------------------
    # ONE BED -> ONE PATIENT
    # -------------------------------------------------------------------------

    duplicate_beds = (
        assigned[
            "bed_id"
        ]
        .astype(str)
        .duplicated()
    )

    if duplicate_beds.any():

        errors.append(
            "A physical bed was assigned to more than one patient."
        )

    # -------------------------------------------------------------------------
    # BED EXISTENCE
    # -------------------------------------------------------------------------

    valid_bed_ids = set(
        beds["_bed_id"]
        .astype(str)
    )

    for bed_id in assigned[
        "bed_id"
    ].astype(str):

        if bed_id not in valid_bed_ids:

            errors.append(
                f"Assigned bed does not exist: {bed_id}"
            )

    # -------------------------------------------------------------------------
    # BED AVAILABILITY
    # -------------------------------------------------------------------------

    bed_lookup = {
        str(row["_bed_id"]): row
        for _, row in beds.iterrows()
    }

    for _, row in assigned.iterrows():

        bed_id = str(
            row["bed_id"]
        )

        if bed_id not in bed_lookup:
            continue

        bed = bed_lookup[
            bed_id
        ]

        if not bool(
            bed["_is_available"]
        ):

            errors.append(
                f"Unavailable bed assigned: {bed_id}"
            )

    # -------------------------------------------------------------------------
    # ICU VALIDATION
    # -------------------------------------------------------------------------

    for _, row in assigned.iterrows():

        if bool(
            row["icu_required"]
        ):

            if normalize_bed_type(
                row["bed_type"]
            ) != "icu":

                errors.append(
                    f"ICU patient assigned to non-ICU bed: "
                    f"{row['patient_id']} -> {row['bed_id']}"
                )

        else:

            if normalize_bed_type(
                row["bed_type"]
            ) == "icu":

                errors.append(
                    f"Non-ICU patient consumed ICU bed: "
                    f"{row['patient_id']} -> {row['bed_id']}"
                )

    # -------------------------------------------------------------------------
    # EMERGENCY VALIDATION
    # -------------------------------------------------------------------------

    for _, row in assigned.iterrows():

        if bool(
            row["emergency_required"]
        ):

            if (
                normalize_bed_type(
                    row["bed_type"]
                )
                != "emergency"
                and
                normalize_department(
                    row["bed_department"]
                )
                != "emergency"
            ):

                errors.append(
                    f"Emergency request assigned to non-Emergency bed: "
                    f"{row['patient_id']} -> {row['bed_id']}"
                )

    # -------------------------------------------------------------------------
    # ISOLATION VALIDATION
    # -------------------------------------------------------------------------

    for _, row in assigned.iterrows():

        if bool(
            row["isolation_required"]
        ):

            if not bool(
                row["bed_isolation_capable"]
            ):

                errors.append(
                    f"Isolation-required patient assigned to "
                    f"non-isolation bed: "
                    f"{row['patient_id']} -> {row['bed_id']}"
                )

    # -------------------------------------------------------------------------
    # DEPARTMENT VALIDATION
    # -------------------------------------------------------------------------

    for _, row in assigned.iterrows():

        if not department_match(
            request_department=normalize_department(
                row["department"]
            ),
            bed_department=normalize_department(
                row["bed_department"]
            ),
            bed_type=normalize_bed_type(
                row["bed_type"]
            ),
        ):

            errors.append(
                f"Department-incompatible assignment: "
                f"{row['patient_id']} -> {row['bed_id']}"
            )

    # -------------------------------------------------------------------------
    # RESERVE VALIDATION
    # -------------------------------------------------------------------------

    emergency_used = len(
        assigned[
            assigned["bed_type"].apply(
                normalize_bed_type
            )
            == "emergency"
        ]
    )

    icu_used = len(
        assigned[
            assigned["bed_type"].apply(
                normalize_bed_type
            )
            == "icu"
        ]
    )

    if (
        emergency_used
        > reserve_info[
            "emergency_max_usable"
        ]
    ):

        errors.append(
            "Emergency reserve requirement was violated."
        )

    if (
        icu_used
        > reserve_info[
            "icu_max_usable"
        ]
    ):

        errors.append(
            "ICU reserve requirement was violated."
        )

    # -------------------------------------------------------------------------
    # PLANNING DATE VALIDATION
    # -------------------------------------------------------------------------

    expected_date = (
        requests[
            "planning_date"
        ]
        .astype(str)
        .unique()
    )

    actual_dates = (
        allocation[
            "planning_date"
        ]
        .astype(str)
        .unique()
    )

    if set(expected_date) != set(
        actual_dates
    ):

        errors.append(
            "Allocation planning dates do not match "
            "request planning dates."
        )

    passed = len(errors) == 0

    return passed, errors


# =============================================================================
# OUTPUT SUMMARY
# =============================================================================

def create_summary(
    allocation: pd.DataFrame,
    beds: pd.DataFrame,
    requests: pd.DataFrame,
    compatibility: Dict[str, List[str]],
    reserve_info: Dict[str, int],
    solver_status: str,
    staffing_capacity: Dict[str, int],
) -> pd.DataFrame:

    assigned = allocation[
        allocation[
            "allocation_status"
        ]
        == "ASSIGNED"
    ]

    available_beds = beds[
        beds["_is_available"]
    ]

    compatible_pairs = sum(
        len(value)
        for value in compatibility.values()
    )

    zero_compatible = sum(
        len(value) == 0
        for value in compatibility.values()
    )

    total_requests = len(
        requests
    )

    assigned_count = len(
        assigned
    )

    unassigned_count = (
        total_requests
        - assigned_count
    )

    assignment_rate = (
        assigned_count
        / total_requests
        if total_requests
        else 0.0
    )

    emergency_used = len(
        assigned[
            assigned["bed_type"].apply(
                normalize_bed_type
            )
            == "emergency"
        ]
    )

    icu_used = len(
        assigned[
            assigned["bed_type"].apply(
                normalize_bed_type
            )
            == "icu"
        ]
    )

    emergency_remaining = (
        reserve_info[
            "emergency_available"
        ]
        - emergency_used
    )

    icu_remaining = (
        reserve_info[
            "icu_available"
        ]
        - icu_used
    )

    summary = pd.DataFrame(
        [
            {
                "planning_date":
                    requests[
                        "planning_date"
                    ].iloc[0]
                    if not requests.empty
                    else "",

                "total_physical_beds":
                    len(beds),

                "currently_available_beds":
                    len(available_beds),

                "currently_unavailable_beds":
                    len(beds)
                    - len(available_beds),

                "planning_requests":
                    total_requests,

                "assigned_requests":
                    assigned_count,

                "unassigned_requests":
                    unassigned_count,

                "assignment_rate":
                    round(
                        assignment_rate,
                        4,
                    ),

                "compatible_patient_bed_pairs":
                    compatible_pairs,

                "requests_with_zero_compatible_beds":
                    zero_compatible,

                "emergency_available":
                    reserve_info[
                        "emergency_available"
                    ],

                "emergency_used":
                    emergency_used,

                "emergency_remaining":
                    emergency_remaining,

                "emergency_reserve_target":
                    EMERGENCY_RESERVE_TARGET,

                "emergency_reserve_required":
                    reserve_info[
                        "emergency_reserve_required"
                    ],

                "icu_available":
                    reserve_info[
                        "icu_available"
                    ],

                "icu_used":
                    icu_used,

                "icu_remaining":
                    icu_remaining,

                "icu_reserve_target":
                    ICU_RESERVE_TARGET,

                "icu_reserve_required":
                    reserve_info[
                        "icu_reserve_required"
                    ],

                "solver_status":
                    solver_status,

                "staffing_departments_observed":
                    len(
                        staffing_capacity
                    ),
            }
        ]
    )

    return summary


# =============================================================================
# BASELINE
# =============================================================================

def create_baseline(
    beds: pd.DataFrame,
) -> pd.DataFrame:

    available = beds[
        beds["_is_available"]
    ].copy()

    rows = []

    for bed_type, group in available.groupby(
        "_bed_type"
    ):

        rows.append(
            {
                "bed_type":
                    bed_type,

                "available_beds":
                    len(group),

                "isolation_capable":
                    int(
                        group[
                            "_isolation_capable"
                        ].sum()
                    ),
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# VALIDATION OUTPUT
# =============================================================================

def create_validation_table(
    passed: bool,
    errors: List[str],
) -> pd.DataFrame:

    return pd.DataFrame(
        [
            {
                "validation_passed":
                    passed,

                "validation_error_count":
                    len(errors),

                "validation_status":
                    "PASSED"
                    if passed
                    else "FAILED",
            }
        ]
    )


# =============================================================================
# REPORT
# =============================================================================

def create_report(
    allocation: pd.DataFrame,
    summary: pd.DataFrame,
    validation_passed: bool,
    validation_errors: List[str],
    reserve_info: Dict[str, int],
    solver_status: str,
) -> str:

    row = summary.iloc[0]

    lines = []

    lines.append(
        "CAREOPS AI - BED ALLOCATION REPORT"
    )
    lines.append(
        "=================================="
    )
    lines.append("")

    lines.append(
        f"Planning date: "
        f"{row['planning_date']}"
    )

    lines.append("")

    lines.append("CAPACITY")
    lines.append(
        f"Total physical beds: "
        f"{row['total_physical_beds']}"
    )
    lines.append(
        f"Currently available beds: "
        f"{row['currently_available_beds']}"
    )
    lines.append(
        f"Currently unavailable beds: "
        f"{row['currently_unavailable_beds']}"
    )

    lines.append("")
    lines.append("REQUESTS")

    lines.append(
        f"Planning requests: "
        f"{row['planning_requests']}"
    )

    lines.append(
        f"Assigned: "
        f"{row['assigned_requests']}"
    )

    lines.append(
        f"Unassigned: "
        f"{row['unassigned_requests']}"
    )

    lines.append(
        f"Assignment rate: "
        f"{row['assignment_rate']:.2%}"
    )

    lines.append("")
    lines.append("COMPATIBILITY")

    lines.append(
        f"Compatible patient-bed pairs: "
        f"{row['compatible_patient_bed_pairs']}"
    )

    lines.append(
        f"Requests with zero compatible beds: "
        f"{row['requests_with_zero_compatible_beds']}"
    )

    lines.append("")
    lines.append("EMERGENCY RESERVE")

    lines.append(
        f"Emergency beds available: "
        f"{row['emergency_available']}"
    )

    lines.append(
        f"Emergency beds used: "
        f"{row['emergency_used']}"
    )

    lines.append(
        f"Emergency beds remaining: "
        f"{row['emergency_remaining']}"
    )

    lines.append(
        f"Emergency reserve target: "
        f"{row['emergency_reserve_target']}"
    )

    lines.append(
        f"Effective Emergency reserve: "
        f"{row['emergency_reserve_required']}"
    )

    lines.append("")
    lines.append("ICU RESERVE")

    lines.append(
        f"ICU beds available: "
        f"{row['icu_available']}"
    )

    lines.append(
        f"ICU beds used: "
        f"{row['icu_used']}"
    )

    lines.append(
        f"ICU beds remaining: "
        f"{row['icu_remaining']}"
    )

    lines.append(
        f"ICU reserve target: "
        f"{row['icu_reserve_target']}"
    )

    lines.append(
        f"Effective ICU reserve: "
        f"{row['icu_reserve_required']}"
    )

    lines.append("")
    lines.append("SOLVER")

    lines.append(
        f"Solver status: "
        f"{solver_status}"
    )

    lines.append(
        f"Effective Emergency reserve used by model: "
        f"{reserve_info['emergency_reserve_required']}"
    )

    lines.append(
        f"Effective ICU reserve used by model: "
        f"{reserve_info['icu_reserve_required']}"
    )

    lines.append("")
    lines.append("VALIDATION")

    lines.append(
        "Independent validation: "
        + (
            "PASSED"
            if validation_passed
            else "FAILED"
        )
    )

    if validation_errors:

        lines.append("")
        lines.append(
            "VALIDATION ERRORS"
        )

        for error in validation_errors:
            lines.append(
                f"- {error}"
            )

    lines.append("")
    lines.append(
        "UNASSIGNED REQUESTS"
    )

    unassigned = allocation[
        allocation[
            "allocation_status"
        ]
        == "UNASSIGNED"
    ]

    if unassigned.empty:

        lines.append(
            "- None"
        )

    else:

        for _, request in unassigned.iterrows():

            lines.append(
                f"- "
                f"{request['patient_id']}: "
                f"{request['reason']}"
            )

    lines.append("")
    lines.append(
        "IMPORTANT"
    )

    lines.append(
        "Source CSV files were read only and were not modified."
    )

    lines.append(
        "Unavailable beds were not treated as available."
    )

    lines.append(
        "Admission type was treated separately from physical bed type."
    )

    lines.append(
        "Planning date was propagated consistently to allocation records."
    )

    lines.append(
        "Patients without safe compatible capacity were left UNASSIGNED."
    )

    return "\n".join(lines)


# =============================================================================
# PUBLISHING
# =============================================================================

def publish_outputs(
    allocation: pd.DataFrame,
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    validation: pd.DataFrame,
    validation_errors: pd.DataFrame,
    report: str,
    validation_passed: bool,
) -> None:
    """
    Publish outputs only when validation passes.

    Existing source CSV files are never touched.
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not validation_passed:

        print()
        print(
            "VALIDATION FAILED."
        )

        print(
            "Validated allocation outputs "
            "were NOT published."
        )

        return

    allocation_path = (
        OUTPUT_DIR
        / "bed_allocations.csv"
    )

    summary_path = (
        OUTPUT_DIR
        / "bed_allocation_summary.csv"
    )

    baseline_path = (
        OUTPUT_DIR
        / "bed_allocation_baseline.csv"
    )

    validation_path = (
        OUTPUT_DIR
        / "bed_allocation_validation.csv"
    )

    validation_errors_path = (
        OUTPUT_DIR
        / "bed_allocation_validation_errors.csv"
    )

    report_path = (
        OUTPUT_DIR
        / "bed_allocation_report.txt"
    )

    allocation.to_csv(
        allocation_path,
        index=False,
    )

    summary.to_csv(
        summary_path,
        index=False,
    )

    baseline.to_csv(
        baseline_path,
        index=False,
    )

    validation.to_csv(
        validation_path,
        index=False,
    )

    validation_errors.to_csv(
        validation_errors_path,
        index=False,
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    print()
    print(
        "[9/9] Publishing validated outputs..."
    )

    print()
    print(
        "================================================================================"
    )
    print(
        "BED ALLOCATION COMPLETE"
    )
    print(
        "======================="
    )
    print()

    print(
        "Independent validation: PASSED"
    )

    print(
        f"Planning requests: "
        f"{len(allocation):,}"
    )

    assigned_count = (
        allocation[
            "allocation_status"
        ]
        == "ASSIGNED"
    ).sum()

    unassigned_count = (
        allocation[
            "allocation_status"
        ]
        == "UNASSIGNED"
    ).sum()

    print(
        f"Assigned: "
        f"{assigned_count:,}"
    )

    print(
        f"Unassigned: "
        f"{unassigned_count:,}"
    )

    print(
        f"Available physical beds: "
        f"{summary.iloc[0]['currently_available_beds']:,}"
    )

    print(
        f"Emergency beds available: "
        f"{summary.iloc[0]['emergency_available']:,}"
    )

    print(
        f"ICU beds available: "
        f"{summary.iloc[0]['icu_available']:,}"
    )

    print()
    print(
        "Validated output files:"
    )

    print()
    print(
        f"1. {allocation_path}"
    )

    print(
        f"2. {summary_path}"
    )

    print(
        f"3. {baseline_path}"
    )

    print(
        f"4. {validation_path}"
    )

    print(
        f"5. {validation_errors_path}"
    )

    print(
        f"6. {report_path}"
    )

    print()
    print(
        "SOURCE DATA WAS NOT MODIFIED."
    )


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:

    print()
    print(
        "================================================================================"
    )
    print(
        "CAREOPS AI - BED ALLOCATION OPTIMIZER"
    )
    print(
        "====================================="
    )
    print()

    print(
        f"Project root: "
        f"{PROJECT_ROOT}"
    )

    print(
        f"Processed data: "
        f"{PROCESSED_DIR}"
    )

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    try:

        # ---------------------------------------------------------------------
        # LOAD
        # ---------------------------------------------------------------------

        data = load_data()

        beds = prepare_beds(
            data["beds"]
        )

        admissions = data[
            "admissions"
        ]

        daily_operations = data[
            "daily_operations"
        ]

        staff = data[
            "staff"
        ]

        shifts = data[
            "shifts"
        ]

        # ---------------------------------------------------------------------
        # PLANNING DATE
        # ---------------------------------------------------------------------

        planning_date = determine_planning_date(
            daily_operations=daily_operations,
            admissions=admissions,
        )

        # ---------------------------------------------------------------------
        # REQUESTS
        # ---------------------------------------------------------------------

        requests = build_requests(
            admissions=admissions,
            planning_date=planning_date,
        )

        if requests.empty:
            raise RuntimeError(
                "No planning requests were created."
            )

        # ---------------------------------------------------------------------
        # STAFF
        # ---------------------------------------------------------------------

        staffing_capacity = load_staff_capacity(
            staff=staff,
            shifts=shifts,
            planning_date=planning_date,
        )

        # ---------------------------------------------------------------------
        # COMPATIBILITY
        # ---------------------------------------------------------------------

        compatibility = build_compatibility(
            requests=requests,
            beds=beds,
        )

        # ---------------------------------------------------------------------
        # MODEL
        # ---------------------------------------------------------------------

        (
            model,
            assignment_vars,
            reserve_info,
        ) = build_optimizer(
            requests=requests,
            beds=beds,
            compatibility=compatibility,
            staffing_capacity=staffing_capacity,
        )

        # ---------------------------------------------------------------------
        # SOLVE
        # ---------------------------------------------------------------------

        (
            solver,
            status,
            solver_status,
        ) = solve_model(
            model
        )

        # ---------------------------------------------------------------------
        # EXTRACT
        # ---------------------------------------------------------------------

        allocation = extract_solution(
            requests=requests,
            beds=beds,
            compatibility=compatibility,
            assignment_vars=assignment_vars,
            solver=solver,
            planning_date=planning_date,
        )

        # ---------------------------------------------------------------------
        # VALIDATE
        # ---------------------------------------------------------------------

        print()
        print(
            "[8/9] Extracting and independently validating solution..."
        )

        (
            validation_passed,
            validation_errors,
        ) = validate_solution(
            allocation=allocation,
            beds=beds,
            requests=requests,
            reserve_info=reserve_info,
        )

        if validation_passed:

            print(
                "Independent validation: PASSED"
            )

        else:

            print(
                "Independent validation: FAILED"
            )

            for error in validation_errors:
                print(
                    f"  - {error}"
                )

        # ---------------------------------------------------------------------
        # OUTPUT TABLES
        # ---------------------------------------------------------------------

        summary = create_summary(
            allocation=allocation,
            beds=beds,
            requests=requests,
            compatibility=compatibility,
            reserve_info=reserve_info,
            solver_status=solver_status,
            staffing_capacity=staffing_capacity,
        )

        baseline = create_baseline(
            beds
        )

        validation = create_validation_table(
            passed=validation_passed,
            errors=validation_errors,
        )

        validation_errors_df = pd.DataFrame(
            {
                "validation_error": validation_errors
            }
        )

        report = create_report(
            allocation=allocation,
            summary=summary,
            validation_passed=validation_passed,
            validation_errors=validation_errors,
            reserve_info=reserve_info,
            solver_status=solver_status,
        )

        # ---------------------------------------------------------------------
        # PUBLISH
        # ---------------------------------------------------------------------

        publish_outputs(
            allocation=allocation,
            summary=summary,
            baseline=baseline,
            validation=validation,
            validation_errors=validation_errors_df,
            report=report,
            validation_passed=validation_passed,
        )

        # Never modify source data.
        return 0

    except Exception as exc:

        print()
        print(
            "================================================================================"
        )
        print(
            "BED ALLOCATION FAILED"
        )
        print(
            "================================================================================"
        )

        print()
        print(
            f"Error: {exc}"
        )

        print()
        print(
            "Source CSV files were not intentionally modified."
        )

        return 1


if __name__ == "__main__":
    sys.exit(
        main()
    )
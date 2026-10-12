"""Request-level patient-to-bed assignment replayed on the synthetic history.

For every planning day in a backtest window the module

1. reconstructs the bed census at the start of the day from the admission
   history (a bed is occupied when ``admission_date < day < discharge_date``;
   discharges are assumed to leave before new arrivals),
2. treats that day's actual arrivals as the bed requests, ignoring the bed the
   generator recorded for them,
3. assigns the same requests with a documented baseline policy and with an
   OR-Tools CP-SAT model under identical compatibility, reserve, and staffing
   rules, and
4. explains every unassigned request with a reason code.

This replaces the standalone study in ``optimization/bed_allocation``, whose
146 "requests" were a random sample of 31-61 day old admissions placed on a
single planning day (see ``docs/BED_ALLOCATION_DIAGNOSIS.md``).

Staffing. The synthetic shift roster was generated independently of the ward
census (General Medicine averages ~56 inpatients against a rostered nurse
capacity of ~2 beds), so a hard roster constraint would reject almost every
admission because of a data artifact. By default staffing is therefore a
reported KPI: assignments into wards already at or above rostered capacity are
counted as ``staffing_exceedances``. ``Rules(staffing_hard=True)`` enforces it,
and the sensitivity table reports that variant.

Synthetic data only. The output is an operational planning illustration, not a
clinical placement decision.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
from ortools.sat.python import cp_model

if __package__:
    from .resource_optimization import (
        EMERGENCY_RESERVE, ICU_RESERVE, NURSE_TO_BED_CAPACITY, SHIFTS,
        SOLVER_RANDOM_SEED, department_bed_type, is_true, normalize,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from optimization.resource_optimization import (
        EMERGENCY_RESERVE, ICU_RESERVE, NURSE_TO_BED_CAPACITY, SHIFTS,
        SOLVER_RANDOM_SEED, department_bed_type, is_true, normalize,
    )


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("CAREOPS_DATA_DIR", ROOT / "data" / "synthetic"))
RESOURCE_SUPPORT = ROOT / "data" / "synthetic" / "resource_support"
OUTPUT = ROOT / "optimization" / "resource_planning" / "outputs"

WINDOW_DAYS = 28
SOLVER_SECONDS = 10.0
SEVERITY_RANK = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
CRITICAL = SEVERITY_RANK["critical"]
# Objective weights. Every assignment is worth far more than any placement
# penalty, so the solver never leaves a patient unplaced to avoid an
# off-service bed; within that, higher severity wins scarce beds.
ASSIGN_WEIGHT = 1_000
SEVERITY_WEIGHT = 100
OFF_SERVICE_PENALTY = 50
UNSTAFFED_PENALTY = 20
UNPLACED_BED_STATUSES = {"maintenance", "closed", "out of service"}

REASONS = {
    "no_compatible_bed_in_inventory":
        "No in-service bed matches the request's bed type, isolation, and ward "
        "rules (for the baseline: no such bed in the home ward).",
    "all_compatible_beds_occupied":
        "Compatible beds exist but every one was occupied at the start of the day.",
    "staffing_limit":
        "Free compatible beds exist only in wards already at rostered nurse "
        "capacity (only when staffing is a hard constraint).",
    "protected_reserve":
        "The only free compatible beds are the protected ICU/emergency reserve, "
        "which is held for critical-severity requests.",
    "taken_by_other_requests":
        "Free compatible beds existed but were all assigned to other requests "
        "the same day.",
}


@dataclass(frozen=True)
class Rules:
    icu_reserve: int = ICU_RESERVE
    emergency_reserve: int = EMERGENCY_RESERVE
    nurse_to_bed: int = NURSE_TO_BED_CAPACITY
    off_service_penalty: int = OFF_SERVICE_PENALTY
    staffing_hard: bool = False

    @property
    def reserve(self) -> dict[str, int]:
        return {"icu": self.icu_reserve, "emergency": self.emergency_reserve}


def load_sources(source: Path = SOURCE) -> dict[str, pd.DataFrame]:
    admissions = pd.read_csv(
        source / "admissions.csv", parse_dates=["admission_date", "discharge_date"]
    )
    labels = pd.read_csv(RESOURCE_SUPPORT / "admission_isolation_labels.csv")
    admissions = admissions.merge(
        labels[["admission_id", "isolation_required"]],
        on="admission_id", how="left", validate="one_to_one",
    )
    admissions["isolation_required"] = admissions["isolation_required"].map(is_true)
    return {
        "admissions": admissions,
        "beds": pd.read_csv(source / "beds.csv"),
        "shifts": pd.read_csv(
            source / "shifts.csv",
            usecols=["date", "department", "shift", "assigned", "staff_role"],
        ),
    }


def prepare_beds(beds: pd.DataFrame) -> pd.DataFrame:
    """Normalize the bed inventory; maintenance beds are never assignable."""
    beds = beds.copy()
    beds["department_key"] = beds["department"].map(normalize)
    beds["bed_type_key"] = beds["bed_type"].map(normalize)
    beds["isolation"] = beds["isolation_capable"].map(is_true)
    beds["emergency_eligible"] = beds["available_for_emergency"].map(is_true)
    beds["in_service"] = ~beds["status"].map(normalize).isin(UNPLACED_BED_STATUSES)
    return beds.set_index("bed_id", drop=False)


def compatibility(
    department: str, icu_required: bool, isolation_required: bool, bed: pd.Series,
) -> str | None:
    """Return 'home', 'off_service', or None when the bed is not allowed.

    The ward rules mirror ``resource_optimization.optimize_bed_allocations``:
    ICU-required patients need an ICU bed; isolation needs an
    isolation-capable bed; emergency patients use emergency beds or general
    beds flagged for emergency overflow; ICU and emergency beds serve only
    those patients. Two request-level additions: pediatric beds serve only
    pediatric patients, and an isolation patient may use another ward's
    isolation room off-service.
    """
    department = normalize(department)
    if not bed["in_service"]:
        return None
    if icu_required or department == "icu":
        return "home" if bed["bed_type_key"] == "icu" else None
    if bed["bed_type_key"] == "icu":
        return None
    if isolation_required and not bed["isolation"]:
        return None
    if (bed["department_key"] == "pediatrics") != (department == "pediatrics"):
        return None
    if department == "emergency":
        if bed["bed_type_key"] == "emergency":
            return "home"
        return "off_service" if bed["emergency_eligible"] else None
    if bed["bed_type_key"] == "emergency":
        return None
    if bed["department_key"] == department:
        return "home"
    if bed["bed_type_key"] == department_bed_type(department) or isolation_required:
        return "off_service"
    return None


class CompatibilityIndex:
    """Compatible beds per request profile, computed once per inventory."""

    def __init__(self, beds: pd.DataFrame):
        self.beds = beds
        self._cache: dict[tuple[str, bool, bool], dict[str, str]] = {}

    @staticmethod
    def profile(request: pd.Series) -> tuple[str, bool, bool]:
        icu = request.get("ICU_required", request.get("icu_required", False))
        return (normalize(request["department"]), bool(icu), bool(request["isolation_required"]))

    def beds_for(self, request: pd.Series) -> dict[str, str]:
        key = self.profile(request)
        if key not in self._cache:
            self._cache[key] = {
                bed_id: kind for bed_id, bed in self.beds.iterrows()
                if (kind := compatibility(*key, bed)) is not None
            }
        return self._cache[key]


def census_on(admissions: pd.DataFrame, day: pd.Timestamp) -> set[str]:
    """Beds occupied at the start of ``day`` by patients admitted earlier."""
    in_house = admissions[
        (admissions["admission_date"] < day) & (admissions["discharge_date"] > day)
    ]
    return set(in_house["assigned_bed_id"].dropna().astype(str))


def staffed_capacity(shifts: pd.DataFrame, day: pd.Timestamp, rules: Rules) -> dict[str, int]:
    """Round-the-clock bed capacity from the least-staffed rostered nurse shift."""
    rows = shifts[
        (shifts["date"] == day.date().isoformat())
        & (shifts["assigned"].map(normalize) == "yes")
        & shifts["staff_role"].map(normalize).str.contains("nurse", na=False)
    ]
    wards = rows["department"].map(normalize)
    per_shift = rows.groupby([wards, "shift"]).size()
    return {
        ward: min(int(per_shift.get((ward, name), 0)) for name in SHIFTS) * rules.nurse_to_bed
        for ward in wards.unique()
    }


def priority(request: pd.Series) -> int:
    return SEVERITY_RANK.get(normalize(request["severity"]), 1)


def _pool_counts(bed_ids: list[str], beds: pd.DataFrame) -> dict[str, int]:
    return beds.loc[bed_ids, "bed_type_key"].value_counts().to_dict() if bed_ids else {}


@dataclass
class DayState:
    day: pd.Timestamp
    free: list[str]
    ward_load: dict[str, int]
    staffed: dict[str, int]

    def has_staff_headroom(self, ward: str, load: dict[str, int]) -> bool:
        return load.get(ward, 0) < self.staffed.get(ward, 0)


def baseline_policy(
    requests: pd.DataFrame, beds: pd.DataFrame, index: CompatibilityIndex,
    state: DayState, rules: Rules,
) -> dict[str, tuple[str, str]]:
    """Arrival-order first fit into the home ward; no off-service placement.

    Reserve and staffing rules are the same as the optimizer's, so the
    comparison isolates the effect of severity ordering and pooling.
    """
    free = set(state.free)
    load = dict(state.ward_load)
    pool_free = _pool_counts(state.free, beds)
    assigned: dict[str, tuple[str, str]] = {}
    for _, request in requests.sort_values("admission_id").iterrows():
        for bed_id, kind in sorted(index.beds_for(request).items()):
            if kind != "home" or bed_id not in free:
                continue
            ward, pool = beds.at[bed_id, "department_key"], beds.at[bed_id, "bed_type_key"]
            if rules.staffing_hard and not state.has_staff_headroom(ward, load):
                continue
            if (pool in rules.reserve and pool_free[pool] <= rules.reserve[pool]
                    and priority(request) < CRITICAL):
                continue
            assigned[request["admission_id"]] = (bed_id, kind)
            free.discard(bed_id)
            load[ward] = load.get(ward, 0) + 1
            pool_free[pool] -= 1
            break
    return assigned


def optimized_policy(
    requests: pd.DataFrame, beds: pd.DataFrame, index: CompatibilityIndex,
    state: DayState, rules: Rules,
) -> tuple[dict[str, tuple[str, str]], dict[str, Any]]:
    """CP-SAT assignment maximizing severity-weighted placements.

    Variables: x[r, b] = 1 when request r takes free bed b (compatible pairs
    only). Constraints: at most one bed per request and one request per bed;
    the ICU/emergency reserve stays free except for critical-severity
    requests; with ``staffing_hard`` the ward load stays within rostered
    capacity. Objective: maximize
    sum((ASSIGN_WEIGHT + SEVERITY_WEIGHT * severity - penalties) * x), where
    penalties are ``off_service_penalty`` for an off-service bed and
    UNSTAFFED_PENALTY for a ward with no rostered headroom.
    """
    model = cp_model.CpModel()
    free = set(state.free)
    severity = {r["admission_id"]: priority(r) for _, r in requests.iterrows()}
    pairs: dict[tuple[str, str], tuple[Any, str]] = {}
    for _, request in requests.iterrows():
        for bed_id, kind in index.beds_for(request).items():
            if bed_id in free:
                var = model.NewBoolVar(f"x_{request['admission_id']}_{bed_id}")
                pairs[(request["admission_id"], bed_id)] = (var, kind)
    by_request: dict[str, list] = {}
    by_bed: dict[str, list] = {}
    by_ward: dict[str, list] = {}
    noncritical_by_pool: dict[str, list] = {}
    for (request_id, bed_id), (var, _) in pairs.items():
        by_request.setdefault(request_id, []).append(var)
        by_bed.setdefault(bed_id, []).append(var)
        by_ward.setdefault(beds.at[bed_id, "department_key"], []).append(var)
        pool = beds.at[bed_id, "bed_type_key"]
        if pool in rules.reserve and severity[request_id] < CRITICAL:
            noncritical_by_pool.setdefault(pool, []).append(var)
    for variables in (*by_request.values(), *by_bed.values()):
        model.AddAtMostOne(variables)
    pool_free = _pool_counts(state.free, beds)
    for pool, variables in noncritical_by_pool.items():
        model.Add(sum(variables) <= max(0, pool_free.get(pool, 0) - rules.reserve[pool]))
    if rules.staffing_hard:
        for ward, variables in by_ward.items():
            headroom = state.staffed.get(ward, 0) - state.ward_load.get(ward, 0)
            model.Add(sum(variables) <= max(0, headroom))

    def weight(request_id: str, bed_id: str, kind: str) -> int:
        value = ASSIGN_WEIGHT + SEVERITY_WEIGHT * severity[request_id]
        if kind == "off_service":
            value -= rules.off_service_penalty
        if not state.has_staff_headroom(beds.at[bed_id, "department_key"], state.ward_load):
            value -= UNSTAFFED_PENALTY
        return value

    model.Maximize(sum(
        weight(request_id, bed_id, kind) * var
        for (request_id, bed_id), (var, kind) in pairs.items()
    ))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SOLVER_SECONDS
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = SOLVER_RANDOM_SEED
    status = solver.Solve(model)
    assigned: dict[str, tuple[str, str]] = {}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for (request_id, bed_id), (var, kind) in pairs.items():
            if solver.Value(var):
                assigned[request_id] = (bed_id, kind)
    solved = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    return assigned, {
        "solver_status": solver.StatusName(status),
        "objective_value": float(solver.ObjectiveValue()) if solved else None,
        "best_bound": float(solver.BestObjectiveBound()) if solved else None,
        "compatible_pairs": len(pairs),
        "wall_time_seconds": round(solver.WallTime(), 4),
    }


def explain_unassigned(
    request: pd.Series, beds: pd.DataFrame, index: CompatibilityIndex,
    state: DayState, rules: Rules, used: set[str], home_only: bool = False,
) -> str:
    """Return the first rule that blocks the request.

    ``used`` holds the beds the policy assigned that day, so the reserve and
    competition checks see the beds actually left over. ``home_only`` scores
    the baseline, which never places off-service.
    """
    compatible = [
        bed_id for bed_id, kind in index.beds_for(request).items()
        if kind == "home" or not home_only
    ]
    if not compatible:
        return "no_compatible_bed_in_inventory"
    free = set(state.free)
    candidates = [bed_id for bed_id in compatible if bed_id in free]
    if not candidates:
        return "all_compatible_beds_occupied"
    if rules.staffing_hard:
        candidates = [
            bed_id for bed_id in candidates
            if state.has_staff_headroom(beds.at[bed_id, "department_key"], state.ward_load)
        ]
        if not candidates:
            return "staffing_limit"
    leftover = [bed_id for bed_id in candidates if bed_id not in used]
    if not leftover:
        return "taken_by_other_requests"
    remaining = _pool_counts([bed_id for bed_id in state.free if bed_id not in used], beds)
    if priority(request) < CRITICAL and all(
        (pool := beds.at[bed_id, "bed_type_key"]) in rules.reserve
        and remaining[pool] <= rules.reserve[pool]
        for bed_id in leftover
    ):
        return "protected_reserve"
    raise AssertionError(
        f"{request['admission_id']} is unassigned although bed {leftover[0]} was usable."
    )


def plan_day(
    day: pd.Timestamp, sources: dict[str, pd.DataFrame], beds: pd.DataFrame,
    rules: Rules = Rules(), index: CompatibilityIndex | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    index = index or CompatibilityIndex(beds)
    admissions = sources["admissions"]
    requests = admissions[admissions["admission_date"] == day].copy()
    occupied = census_on(admissions, day) & set(beds.index[beds["in_service"]])
    state = DayState(
        day=day,
        free=sorted(set(beds.index[beds["in_service"]]) - occupied),
        ward_load=beds.loc[sorted(occupied), "department_key"].value_counts().to_dict(),
        staffed=staffed_capacity(sources["shifts"], day, rules),
    )
    results = {
        "baseline": baseline_policy(requests, beds, index, state, rules),
    }
    results["optimized"], solver = optimized_policy(requests, beds, index, state, rules)

    rows = []
    used = {policy: {bed_id for bed_id, _ in assigned.values()}
            for policy, assigned in results.items()}
    for _, request in requests.iterrows():
        compatible = index.beds_for(request)
        row = {
            "planning_date": day.date().isoformat(),
            "admission_id": request["admission_id"],
            "department": request["department"],
            "severity": request["severity"],
            "admission_type": request["admission_type"],
            "icu_required": bool(request["ICU_required"]),
            "isolation_required": bool(request["isolation_required"]),
            "compatible_beds_in_inventory": len(compatible),
            "compatible_free_beds": sum(bed_id in set(state.free) for bed_id in compatible),
        }
        for policy, assigned in results.items():
            bed_id, kind = assigned.get(request["admission_id"], (None, None))
            row[f"{policy}_status"] = "ASSIGNED" if bed_id else "UNASSIGNED"
            row[f"{policy}_bed_id"] = bed_id
            row[f"{policy}_bed_department"] = beds.at[bed_id, "department"] if bed_id else None
            row[f"{policy}_placement"] = kind
            row[f"{policy}_reason"] = None if bed_id else explain_unassigned(
                request, beds, index, state, rules, used[policy],
                home_only=policy == "baseline",
            )
        rows.append(row)
    columns = [
        "planning_date", "admission_id", "department", "severity", "admission_type",
        "icu_required", "isolation_required", "compatible_beds_in_inventory",
        "compatible_free_beds",
    ] + [
        f"{policy}_{field}" for policy in results
        for field in ("status", "bed_id", "bed_department", "placement", "reason")
    ]
    detail = pd.DataFrame(rows, columns=columns)
    # Count placements into wards already at or above rostered nurse capacity.
    for policy, assigned in results.items():
        load = dict(state.ward_load)
        exceedances = 0
        for bed_id, _ in assigned.values():
            ward = beds.at[bed_id, "department_key"]
            if not state.has_staff_headroom(ward, load):
                exceedances += 1
            load[ward] = load.get(ward, 0) + 1
        solver[f"{policy}_staffing_exceedances"] = exceedances
    day_summary = {
        "planning_date": day.date().isoformat(),
        "requests": int(len(requests)),
        "in_service_beds": int(beds["in_service"].sum()),
        "occupied_at_start": len(occupied),
        "free_at_start": len(state.free),
        "rostered_staffed_capacity": int(sum(state.staffed.values())),
        **solver,
    }
    return detail, day_summary


def summarize(detail: pd.DataFrame, beds: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """Policy KPIs over the window. Each metric's denominator is noted."""
    index = CompatibilityIndex(beds)
    severity = detail["severity"].map(lambda value: SEVERITY_RANK.get(normalize(value), 1))
    rows = []
    for policy in ("baseline", "optimized"):
        assigned = detail[f"{policy}_status"] == "ASSIGNED"
        reasons = detail.loc[~assigned, f"{policy}_reason"].value_counts()
        violations = sum(
            request[f"{policy}_bed_id"] not in index.beds_for(request)
            for _, request in detail[assigned].iterrows()
        )
        rows.append({
            "policy": policy,
            "requests": int(len(detail)),
            "assigned": int(assigned.sum()),
            "unassigned": int((~assigned).sum()),
            # assigned / all requests in the window
            "assignment_rate": round(float(assigned.mean()), 4) if len(detail) else 0.0,
            # assigned critical+high / all critical+high requests
            "high_severity_assignment_rate": round(float(assigned[severity >= 3].mean()), 4)
            if (severity >= 3).any() else None,
            "off_service_placements": int((detail[f"{policy}_placement"] == "off_service").sum()),
            # bed-days used by new requests / in-service beds free at start of day, pooled
            "free_bed_fill_rate": round(float(assigned.sum() / days["free_at_start"].sum()), 4),
            "compatibility_violations": int(violations),
            "staffing_exceedances": int(days[f"{policy}_staffing_exceedances"].sum()),
            **{f"unassigned_{reason}": int(reasons.get(reason, 0)) for reason in REASONS},
        })
    return pd.DataFrame(rows)


def run_window(
    window: list[pd.Timestamp], sources: dict[str, pd.DataFrame], beds: pd.DataFrame,
    rules: Rules = Rules(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = CompatibilityIndex(beds)
    results = [plan_day(day, sources, beds, rules, index) for day in window]
    return (
        pd.concat([detail for detail, _ in results], ignore_index=True),
        pd.DataFrame([summary for _, summary in results]),
    )


def sensitivity(
    window: list[pd.Timestamp], sources: dict[str, pd.DataFrame], beds: pd.DataFrame,
) -> pd.DataFrame:
    """Re-run both policies under alternative weights, reserves, and staffing."""
    variants = {
        "documented": Rules(),
        "no_off_service_penalty": Rules(off_service_penalty=0),
        "off_service_penalty_900": Rules(off_service_penalty=900),
        "no_protected_reserve": Rules(icu_reserve=0, emergency_reserve=0),
        "double_reserve": Rules(
            icu_reserve=2 * ICU_RESERVE, emergency_reserve=2 * EMERGENCY_RESERVE
        ),
        "rostered_staffing_hard": Rules(staffing_hard=True),
    }
    rows = []
    for name, rules in variants.items():
        detail, days = run_window(window, sources, beds, rules)
        for policy, kpi in summarize(detail, beds, days).set_index("policy").iterrows():
            rows.append({
                "variant": name,
                "policy": policy,
                "off_service_penalty": rules.off_service_penalty,
                "icu_reserve": rules.icu_reserve,
                "emergency_reserve": rules.emergency_reserve,
                "staffing_hard": rules.staffing_hard,
                "assigned": int(kpi["assigned"]),
                "unassigned": int(kpi["unassigned"]),
                "off_service_placements": int(kpi["off_service_placements"]),
                "staffing_exceedances": int(kpi["staffing_exceedances"]),
            })
    return pd.DataFrame(rows)


def run(as_of: date, window_days: int = WINDOW_DAYS, output: Path = OUTPUT,
        sources: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    sources = sources or load_sources()
    beds = prepare_beds(sources["beds"])
    end = pd.Timestamp(as_of)
    window = [end - pd.Timedelta(days=offset) for offset in range(window_days - 1, -1, -1)]
    detail, days = run_window(window, sources, beds)
    kpis = summarize(detail, beds, days)
    sens = sensitivity(window, sources, beds)

    output.mkdir(parents=True, exist_ok=True)
    detail.to_csv(output / "bed_assignment_requests.csv", index=False)
    days.to_csv(output / "bed_assignment_daily.csv", index=False)
    kpis.to_csv(output / "bed_assignment_kpis.csv", index=False)
    sens.to_csv(output / "bed_assignment_sensitivity.csv", index=False)
    summary = {
        "module": "request_level_bed_assignment",
        "data_scope": "fully_synthetic_replay",
        "as_of_date": as_of.isoformat(),
        "window": [window[0].date().isoformat(), window[-1].date().isoformat()],
        "requests_definition": "Actual synthetic arrivals on each planning day; "
                               "their recorded beds are ignored.",
        "census_definition": "In-service beds held by patients with admission_date < day "
                             "< discharge_date. Beds in maintenance in the inventory "
                             "snapshot are treated as out of service on every day.",
        "baseline_policy": "Arrival-order first fit into the home ward, no off-service "
                           "placement, same reserve and staffing rules as the optimizer.",
        "objective": {
            "assign_weight": ASSIGN_WEIGHT,
            "severity_weight": SEVERITY_WEIGHT,
            "off_service_penalty": OFF_SERVICE_PENALTY,
            "unstaffed_ward_penalty": UNSTAFFED_PENALTY,
        },
        "constraints": {
            "icu_reserve_beds": ICU_RESERVE,
            "emergency_reserve_beds": EMERGENCY_RESERVE,
            "reserve_exception": "critical severity",
            "nurse_to_bed_ratio": NURSE_TO_BED_CAPACITY,
            "staffing": "reported (staffing_exceedances); hard only in the "
                        "rostered_staffing_hard sensitivity variant",
        },
        "solver_statuses": days["solver_status"].value_counts().to_dict(),
        "kpis": kpis.to_dict(orient="records"),
        "reason_definitions": REASONS,
        "limitations": [
            "Arrivals are batched per day; intra-day timing is not modeled.",
            "Isolation labels are synthetic scenario probabilities, not clinical flags.",
            "Bed maintenance status is only known for the snapshot date.",
            "The synthetic shift roster is not tied to ward census, so rostered "
            "staffing is reported rather than enforced.",
        ],
    }
    (output / "bed_assignment_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    as_of_value = os.environ.get("CAREOPS_AS_OF_DATE")
    if as_of_value:
        as_of = date.fromisoformat(as_of_value)
    else:
        admissions = pd.read_csv(SOURCE / "admissions.csv", usecols=["admission_date"])
        as_of = pd.to_datetime(admissions["admission_date"]).max().date()
    summary = run(as_of)
    for row in summary["kpis"]:
        print(
            f"{row['policy']:>9}: {row['assigned']}/{row['requests']} assigned, "
            f"{row['off_service_placements']} off-service, "
            f"{row['compatibility_violations']} compatibility violations, "
            f"{row['staffing_exceedances']} staffing exceedances"
        )
    print(f"Solver statuses: {summary['solver_statuses']}")


if __name__ == "__main__":
    main()

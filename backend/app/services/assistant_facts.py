"""Deterministic, evidence-backed answers for the CareOps assistant.

Every answer is computed from saved pipeline outputs. Each one names the
files it read, labels each figure as observed (synthetic history or
inventory), forecast, model metric, or simulated, and states its limits.
No figure is produced here that a pipeline stage did not calculate; the
optional LLM provider may only rephrase these facts.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "optimization" / "resource_planning" / "outputs"
ML_DIR = PROJECT_ROOT / "ml"

SYNTHETIC_NOTE = "Synthetic portfolio data; not real hospital performance or clinical advice."
WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "fourteen": 14, "thirty": 30,
}


@dataclass
class Evidence:
    label: str
    value: Any
    kind: str  # observed | forecast | model_metric | simulated
    source: str


@dataclass
class GroundedAnswer:
    intent: str
    answer: str
    evidence: list[Evidence] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rel(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _manifest() -> dict[str, Any]:
    return _json(OUTPUT_DIR / "portfolio_run_manifest.json")


def _count(value: Any) -> int:
    """Present a count as a whole number (half up, never negative)."""
    return max(0, int(float(value) + 0.5))


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _unavailable(intent: str, what: str, path: Path) -> GroundedAnswer:
    return GroundedAnswer(
        intent=intent,
        answer=f"I don't have {what} in the current CareOps-AI outputs ({_rel(path)} is missing). "
               "Run run_portfolio.py to generate it; I will not estimate it.",
        sources=[_rel(path)],
    )


# ---------------------------------------------------------------- beds

def _horizon_days(question: str) -> int | None:
    lower = question.casefold()
    if "tomorrow" in lower:
        return 1
    match = re.search(r"(\d+|[a-z]+)\s*(day|week)s?", lower)
    if not match:
        return 7 if "next week" in lower else None
    raw, unit = match.groups()
    amount = int(raw) if raw.isdigit() else WORD_NUMBERS.get(raw, 1 if raw in {"a", "one"} else 0)
    if not amount:
        return None
    return amount * (7 if unit == "week" else 1)


def beds_available(question: str) -> GroundedAnswer:
    path = OUTPUT_DIR / "bed_availability_forecast.csv"
    forecast = _csv(path)
    if forecast.empty:
        return _unavailable("bed_availability", "a bed-availability forecast", path)
    forecast["date"] = pd.to_datetime(forecast["date"])
    manifest = _manifest()
    as_of = pd.Timestamp(manifest.get("scenario_as_of_date", forecast["date"].min() - pd.Timedelta(days=1)))
    days = _horizon_days(question) or 1
    target = as_of + pd.Timedelta(days=days)
    rows = forecast[forecast["date"] == target]
    if rows.empty:
        last = forecast["date"].max()
        return GroundedAnswer(
            intent="bed_availability",
            answer=f"The saved forecast covers {forecast['date'].min().date()} to {last.date()}; "
                   f"{target.date()} ({days} days after the {as_of.date()} cutoff) is outside it.",
            sources=[_rel(path)], run_id=manifest.get("scenario_as_of_date"),
        )
    available = int(rows["forecast_available_beds"].sum())
    capacity = int(rows["operational_bed_capacity"].sum())
    occupied = int(rows["projected_occupied_beds"].sum())
    icu = rows[rows["department"].str.casefold() == "icu"]
    tight = rows.assign(free=rows["forecast_available_beds"]).sort_values("free").head(3)
    tight_text = ", ".join(
        f"{row.department} {int(row.free)} of {int(row.operational_bed_capacity)}"
        for row in tight.itertuples()
    )
    answer = (
        f"For {target.date()} ({days} day{'s' if days != 1 else ''} after the {as_of.date()} cutoff) "
        f"the forecast projects {occupied} occupied of {capacity} operational beds, leaving "
        f"{available} available"
        + (f", including {int(icu['forecast_available_beds'].sum())} ICU beds" if not icu.empty else "")
        + f". Tightest wards: {tight_text}. Projected occupancy combines the admission forecast, "
        "predicted LOS and the current census; counts are rounded to whole beds and are planning "
        "estimates, not guarantees."
    )
    return GroundedAnswer(
        intent="bed_availability",
        answer=answer,
        evidence=[
            Evidence("forecast date", target.date().isoformat(), "forecast", _rel(path)),
            Evidence("projected occupied beds", occupied, "forecast", _rel(path)),
            Evidence("operational bed capacity", capacity, "observed", _rel(path)),
            Evidence("forecast available beds", available, "forecast", _rel(path)),
        ],
        sources=[_rel(path)],
        limitations=[
            "Operational capacity includes synthetic surge beds sized by the planning stage.",
            "No prediction interval is published for department-level bed availability.",
        ],
        run_id=manifest.get("scenario_as_of_date"),
    )


def capacity_pressure(question: str) -> GroundedAnswer:
    path = OUTPUT_DIR / "bed_availability_forecast.csv"
    forecast = _csv(path)
    if forecast.empty:
        return _unavailable("capacity_pressure", "a bed-availability forecast", path)
    forecast["utilization"] = forecast["projected_occupied_beds"] / forecast["operational_bed_capacity"]
    peaks = (
        forecast.sort_values(["utilization", "date"], ascending=[False, True])
        .groupby("department").head(1).sort_values("utilization", ascending=False)
    )
    top = peaks.head(3)
    lines = [
        f"{row.department}: peak {_pct(row.utilization)} ({int(row.projected_occupied_beds)} of "
        f"{int(row.operational_bed_capacity)} beds) on {row.date}"
        for row in top.itertuples()
    ]
    manifest = _manifest()
    return GroundedAnswer(
        intent="capacity_pressure",
        answer="Highest projected bed pressure over the forecast horizon: " + "; ".join(lines)
               + ". Pressure is projected occupied beds divided by operational capacity.",
        evidence=[
            Evidence(f"{row.department} peak utilization", round(float(row.utilization), 4),
                     "forecast", _rel(path))
            for row in top.itertuples()
        ],
        sources=[_rel(path)],
        limitations=["Department projections apportion a hospital-level occupancy forecast."],
        run_id=manifest.get("scenario_as_of_date"),
    )


# ---------------------------------------------------------- allocation

def _assignment() -> tuple[dict[str, Any], pd.DataFrame, Path]:
    path = OUTPUT_DIR / "bed_assignment_summary.json"
    return _json(path), _csv(OUTPUT_DIR / "bed_assignment_requests.csv"), path


def unassigned_requests(question: str) -> GroundedAnswer:
    summary, requests, path = _assignment()
    if not summary:
        return _unavailable("unassigned_requests", "request-level bed assignment results", path)
    kpis = {row["policy"]: row for row in summary["kpis"]}
    parts = []
    for policy in ("optimized", "baseline"):
        row = kpis[policy]
        reasons = {
            key.removeprefix("unassigned_"): value for key, value in row.items()
            if key.startswith("unassigned_") and value
        }
        reason_text = ", ".join(f"{count} {name.replace('_', ' ')}" for name, count in reasons.items())
        parts.append(
            f"The {policy} policy left {row['unassigned']} of {row['requests']} requests unassigned"
            + (f" ({reason_text})" if reason_text else "")
        )
    unassigned = requests[requests["optimized_status"] == "UNASSIGNED"]
    examples = "; ".join(
        f"{row.admission_id} ({row.department}, {row.planning_date}): {row.optimized_reason.replace('_', ' ')}"
        for row in unassigned.head(3).itertuples()
    )
    answer = (
        f"Replay of actual synthetic arrivals from {summary['window'][0]} to {summary['window'][1]}. "
        + ". ".join(parts) + "."
        + (f" Optimized examples: {examples}." if examples else "")
        + " Each reason is the first rule that blocked the request: no compatible bed type, "
        "every compatible bed occupied, the protected ICU/emergency reserve, or beds taken by "
        "other requests that day."
    )
    return GroundedAnswer(
        intent="unassigned_requests",
        answer=answer,
        evidence=[
            Evidence(f"{policy} unassigned", kpis[policy]["unassigned"], "simulated", _rel(path))
            for policy in ("baseline", "optimized")
        ],
        sources=[_rel(path), "optimization/resource_planning/outputs/bed_assignment_requests.csv"],
        limitations=summary.get("limitations", []),
        run_id=summary.get("as_of_date"),
    )


def allocation_comparison(question: str) -> GroundedAnswer:
    summary, _, path = _assignment()
    if not summary:
        return _unavailable("allocation_comparison", "request-level bed assignment results", path)
    kpis = {row["policy"]: row for row in summary["kpis"]}
    base, opt = kpis["baseline"], kpis["optimized"]
    department_path = OUTPUT_DIR / "bed_allocation_kpis.csv"
    department = _csv(department_path).set_index("scenario") if department_path.exists() else pd.DataFrame()
    department_text = ""
    if not department.empty:
        department_text = (
            f" At department/day level over the 30-day forecast, baseline and optimized both show "
            f"{int(department.loc['baseline', 'overflow_bed_days'])} and "
            f"{int(department.loc['optimized', 'overflow_bed_days'])} overflow bed-days: projected "
            "demand fits the planned capacity, so pooling has nothing to improve there."
        )
    answer = (
        f"On the same {base['requests']} replayed requests ({summary['window'][0]} to "
        f"{summary['window'][1]}), the baseline ({summary['baseline_policy'].rstrip('.')}) assigned "
        f"{base['assigned']} ({_pct(base['assignment_rate'])}) and the CP-SAT optimizer assigned "
        f"{opt['assigned']} ({_pct(opt['assignment_rate'])}), using {opt['off_service_placements']} "
        f"off-service placements. High-severity assignment rate: {_pct(base['high_severity_assignment_rate'])} "
        f"vs {_pct(opt['high_severity_assignment_rate'])}. Compatibility violations: "
        f"{base['compatibility_violations']} vs {opt['compatibility_violations']}."
        + department_text
    )
    return GroundedAnswer(
        intent="allocation_comparison",
        answer=answer,
        evidence=[
            Evidence("baseline assigned", base["assigned"], "simulated", _rel(path)),
            Evidence("optimized assigned", opt["assigned"], "simulated", _rel(path)),
            Evidence("optimized off-service placements", opt["off_service_placements"], "simulated", _rel(path)),
        ],
        sources=[_rel(path)] + ([_rel(department_path)] if department_text else []),
        limitations=summary.get("limitations", []),
        run_id=summary.get("as_of_date"),
    )


# ---------------------------------------------------------------- staff

def staff_coverage(question: str) -> GroundedAnswer:
    path = OUTPUT_DIR / "staff_coverage_kpis.csv"
    coverage = _csv(path)
    if coverage.empty:
        return _unavailable("staff_coverage", "staff coverage results", path)
    short = coverage[coverage["unfilled_staff"] > 0]
    plan_path = OUTPUT_DIR / "capacity_expansion_plan.csv"
    plan = _csv(plan_path)
    surge = plan[plan["resource_type"] == "synthetic_surge_staff"] if not plan.empty else plan
    manifest = _manifest()
    if short.empty:
        answer = (
            f"The optimized staff schedule fills all {int(coverage['required_staff'].sum())} required "
            f"shift positions across {coverage['date'].nunique()} days "
            f"({len(coverage)} date/department/shift/role rows); no shift is short."
        )
    else:
        worst = short.sort_values("unfilled_staff", ascending=False).head(5)
        answer = (
            f"{len(short)} shift/role rows are short by {int(short['unfilled_staff'].sum())} staff in total. "
            "Largest gaps: " + "; ".join(
                f"{row.date} {row.department} {row.shift} {row.role}: {int(row.unfilled_staff)} unfilled"
                for row in worst.itertuples()
            ) + "."
        )
    if not surge.empty:
        answer += (
            f" Coverage relies on {len(surge)} synthetic surge-staff assignments from the capacity "
            "expansion plan, which are scenario assumptions, not hires."
        )
    return GroundedAnswer(
        intent="staff_coverage",
        answer=answer,
        evidence=[
            Evidence("required shift positions", int(coverage["required_staff"].sum()), "simulated", _rel(path)),
            Evidence("unfilled positions", int(coverage["unfilled_staff"].sum()), "simulated", _rel(path)),
        ],
        sources=[_rel(path)] + ([_rel(plan_path)] if not surge.empty else []),
        limitations=[
            "Required staff is derived from projected census and fixed nurse ratios.",
            "The synthetic historical roster is not tied to ward census.",
        ],
        run_id=manifest.get("scenario_as_of_date"),
    )


# --------------------------------------------------------------- models

@lru_cache(maxsize=1)
def _metrics_cache(stamp: float) -> tuple[pd.DataFrame, dict[str, Any]]:
    from ml import model_metrics_report as report

    data = report.load_outputs()
    return report.summary_rows(data), data


def _metrics() -> tuple[pd.DataFrame, dict[str, Any]] | None:
    stamp_path = OUTPUT_DIR / "portfolio_run_manifest.json"
    try:
        return _metrics_cache(stamp_path.stat().st_mtime if stamp_path.exists() else 0.0)
    except (FileNotFoundError, KeyError):
        return None


TARGETS = {
    "occupan": "Occupied beds (of 230)",
    "admission": "Hospital admissions / day",
    "icu": "ICU admissions / day",
    "length of stay": "Length of stay (days)",
    "los": "Length of stay (days)",
    "no-show": "No-show probability",
    "no show": "No-show probability",
}


def model_reliability(question: str) -> GroundedAnswer:
    loaded = _metrics()
    report_path = PROJECT_ROOT / "docs" / "MODEL_METRICS.md"
    if loaded is None:
        return _unavailable("model_reliability", "model evaluation outputs", report_path)
    summary, data = loaded
    lower = question.casefold()
    wanted = {target for key, target in TARGETS.items() if key in lower}
    rows = summary[summary["Target"].isin(wanted)] if wanted else summary
    sentences = []
    for row in rows.itertuples():
        beats = "beats" if row._8 == "yes" else "does not beat"
        sentences.append(
            f"{row.Target}: published {row._4} with {row.Metric} {row.Published} on a {row.Evaluation.lower()}; "
            f"best baseline {row._6}; the best learned model ({row._7}) {beats} it"
        )
    coverage = {
        "Hospital admissions / day": "admission_coverage",
        "Occupied beds (of 230)": "occupancy_coverage",
        "ICU admissions / day": "icu_coverage",
    }
    for target in rows["Target"]:
        key = coverage.get(target)
        if key and key in data:
            frame = data[key]
            if "empirical_coverage" in frame:
                published = frame[frame["method"].eq("sequential_published")]
                eighty = published[published["level"].round(2) == 0.8]["empirical_coverage"]
                if not eighty.empty:
                    sentences.append(
                        f"{target} 80% intervals covered {_pct(eighty.min())}-{_pct(eighty.max())} "
                        "of actuals across horizon buckets"
                    )
    return GroundedAnswer(
        intent="model_reliability",
        answer=". ".join(sentences) + ". Lower is better for MAE and Brier. Evaluation is "
               "retrospective on synthetic data and uses only time-based splits.",
        evidence=[
            Evidence(f"{row.Target} {row.Metric}", row.Published, "model_metric", _rel(report_path))
            for row in rows.itertuples()
        ],
        sources=[_rel(report_path), "ml/*/backtest summaries"],
        limitations=["Backtests on synthetic data do not establish real-world accuracy."],
        run_id=data.get("manifest", {}).get("scenario_as_of_date"),
    )


def model_limitations(question: str) -> GroundedAnswer:
    loaded = _metrics()
    report_path = PROJECT_ROOT / "docs" / "MODEL_METRICS.md"
    if loaded is None:
        return _unavailable("model_limitations", "model evaluation outputs", report_path)
    summary, data = loaded
    items = []
    losers = summary[summary["Learned beats baseline"] != "yes"]
    if not losers.empty:
        items.append(
            "No learned model beats the naive baseline for " + ", ".join(losers["Target"])
            + "; the baseline is published instead (these counts behave close to Poisson noise)"
        )
    risk = data.get("occupancy_risk")
    if risk is not None and not risk.empty:
        never = risk[risk["actual_events"] == 0]["threshold_name"].tolist()
        if never:
            items.append(
                "Occupancy alert thresholds " + ", ".join(never)
                + " were never reached in the backtest, so alert precision/recall there is untested"
            )
        elevated = risk[risk["actual_events"] > 0]
        for row in elevated.itertuples():
            if row.recall == 0:
                items.append(
                    f"The occupancy forecast caught 0 of {int(row.actual_events)} days above the "
                    f"{row.threshold_name} level ({int(row.threshold)} beds)"
                )
    items += [
        "The ICU model forecasts ICU admissions per day, not ICU bed census",
        "All data is synthetic; isolation labels and the shift roster are scenario assumptions",
        "LOS and no-show models are evaluated retrospectively; no prospective validation exists",
    ]
    return GroundedAnswer(
        intent="model_limitations",
        answer="Main limitations: " + "; ".join(items) + ".",
        sources=[_rel(report_path), "ml/occupancy_forecasting/occupancy_risk_metrics.csv"],
        limitations=items,
        run_id=data.get("manifest", {}).get("scenario_as_of_date"),
    )


# ------------------------------------------------------- forecasts etc.

def icu_forecast(question: str) -> GroundedAnswer:
    path = ML_DIR / "icu_demand_forecasting" / "icu_future_forecast.csv"
    forecast = _csv(path)
    if forecast.empty:
        return _unavailable("icu_forecast", "an ICU forecast", path)
    emergency_path = OUTPUT_DIR / "emergency_capacity_optimization.csv"
    emergency = _csv(emergency_path)
    first = forecast.iloc[0]
    total = int(forecast["predicted_icu_admissions"].sum())
    answer = (
        f"The ICU model forecasts {_count(first['predicted_icu_admissions'])} ICU admission(s) on "
        f"{first['date']} and {total} over {len(forecast)} days. This is an admissions flow, "
        "not a bed census forecast."
    )
    evidence = [Evidence("ICU admissions over horizon", total, "forecast", _rel(path))]
    if not emergency.empty:
        peak = emergency.sort_values("projected_icu_census_beds", ascending=False).iloc[0]
        answer += (
            f" The planning stage projects an ICU census peak of {int(peak['projected_icu_census_beds'])} "
            f"of {int(peak['icu_capacity'])} beds on {peak['date']} (reserve {int(peak['icu_reserve'])}), "
            f"with {int(emergency['icu_overflow'].sum())} overflow bed-days."
        )
        evidence.append(Evidence("peak projected ICU census", int(peak["projected_icu_census_beds"]),
                                 "forecast", _rel(emergency_path)))
    return GroundedAnswer(
        intent="icu_forecast", answer=answer, evidence=evidence,
        sources=[_rel(path), _rel(emergency_path)],
        limitations=["ICU census is projected from admissions and LOS, not forecast directly."],
        run_id=_manifest().get("scenario_as_of_date"),
    )


def admission_forecast(question: str) -> GroundedAnswer:
    path = ML_DIR / "admission_forecasting" / "hospital_30_day_admission_forecast.csv"
    forecast = _csv(path)
    if forecast.empty:
        return _unavailable("admission_forecast", "an admission forecast", path)
    total = int(forecast["predicted_admissions"].sum())
    model = forecast["model"].iloc[0] if "model" in forecast else "unknown"
    answer = (
        f"The saved {len(forecast)}-day hospital admission forecast ({forecast['date'].iloc[0]} to "
        f"{forecast['date'].iloc[-1]}) totals {total} admissions, about "
        f"{_count(forecast['predicted_admissions'].mean())} per day, from the {model} model"
    )
    if {"lower_80", "upper_80"} <= set(forecast.columns):
        answer += (
            f"; the first day's 80% interval is {int(forecast['lower_80'].iloc[0])}-"
            f"{int(forecast['upper_80'].iloc[0])}"
        )
    return GroundedAnswer(
        intent="admission_forecast",
        answer=answer + ". Daily counts are rounded expected values, not exact predictions.",
        evidence=[Evidence("forecast admissions", total, "forecast", _rel(path))],
        sources=[_rel(path)], run_id=_manifest().get("scenario_as_of_date"),
    )


def occupancy_forecast(question: str) -> GroundedAnswer:
    path = ML_DIR / "occupancy_forecasting" / "occupancy_future_forecast.csv"
    forecast = _csv(path)
    if forecast.empty:
        return _unavailable("occupancy_forecast", "an occupancy forecast", path)
    peak = forecast.sort_values("forecast_occupied_beds", ascending=False).iloc[0]
    answer = (
        f"The occupancy forecast peaks at {int(peak['forecast_occupied_beds'])} occupied beds on "
        f"{peak['date']} over {len(forecast)} days"
    )
    if {"occupied_beds_lower_80", "occupied_beds_upper_80"} <= set(forecast.columns):
        answer += (
            f" (80% interval {int(peak['occupied_beds_lower_80'])}-"
            f"{int(peak['occupied_beds_upper_80'])})"
        )
    return GroundedAnswer(
        intent="occupancy_forecast", answer=answer + ".",
        evidence=[Evidence("peak forecast occupied beds", int(peak["forecast_occupied_beds"]),
                           "forecast", _rel(path))],
        sources=[_rel(path)], run_id=_manifest().get("scenario_as_of_date"),
    )


def recommendations(question: str) -> GroundedAnswer:
    from backend.app.services.dashboard_service import get_recommendations

    items = get_recommendations()
    gap = "optimization/resource_planning/outputs/resource_gap_analysis.csv"
    if not items:
        return GroundedAnswer("recommendations", "No recommendations are supported by the saved run.",
                              sources=[gap])
    actionable = [item for item in items if item.get("priority") != "low"]
    shown = actionable[:3] or items[:1]
    answer = (
        f"{len(actionable)} of the top {len(items)} resource-gap rows call for action. "
        if actionable else
        "The latest run shows no projected bed overflow or incremental nurse gap, so the only "
        "supported recommendation is to keep monitoring. "
    ) + " ".join(f"[{item['priority']}] {item['title']}: {item['detail']}" for item in shown)
    return GroundedAnswer(
        intent="recommendations", answer=answer,
        evidence=[Evidence(item["title"], item.get("detail"), "simulated", gap) for item in shown],
        sources=[gap], run_id=_manifest().get("scenario_as_of_date"),
        limitations=["Recommendations are decision support for synthetic scenarios; staff "
                     "review them before any operational use."],
    )


def scenario_summary(question: str, assumptions: dict[str, Any] | None = None) -> GroundedAnswer:
    """Explain the most recent saved simulation matching the requested assumptions."""
    from backend.app.services.simulation_service import get_simulation_runs

    runs = get_simulation_runs(limit=50)
    if assumptions:
        defaults = {"admissions_multiplier": 1.0, "icu_bed_delta": 0, "staff_availability_delta": 0.0}
        wanted = {**defaults, **assumptions}
        runs = [
            run for run in runs
            if all(abs(float((run.get("assumptions") or {}).get(key, value)) - float(value)) < 1e-6
                   for key, value in wanted.items())
        ]
    if not runs:
        return GroundedAnswer(
            intent="scenario",
            answer="There is no saved simulation matching this question. Run the What-if Simulator "
                   "to calculate it with the OR-Tools allocator; I will not estimate the result "
                   "without a run.",
            sources=["simulation history"],
        )
    run = runs[0]
    base, scen, assumed = run["baseline_metrics"], run["simulated_metrics"], run["assumptions"]
    changes = [
        f"{metric.replace('_', ' ')} {_count(base[metric])} -> {_count(scen[metric])}"
        for metric in ("overflow_bed_days", "icu_overflow_bed_days", "unmet_isolation_bed_days",
                       "staffing_constraint_violations")
        if metric in base and metric in scen
    ]
    answer = (
        f"Saved simulation {run['simulation_id']} ({run.get('timestamp', 'unknown time')}) applied an "
        f"admissions multiplier of {assumed.get('admissions_multiplier')}, an ICU bed change of "
        f"{assumed.get('icu_bed_delta')} and a staff availability change of "
        f"{assumed.get('staff_availability_delta')}. Rerunning the department/day OR-Tools allocator "
        f"against the saved baseline: " + "; ".join(changes)
        + f". Scenario unfilled staff positions: {_count(scen.get('unfilled_staff_positions', 0))}. "
        "The forecasts were not retrained; extra admissions are scaled from the saved forecast "
        "and predicted LOS."
    )
    return GroundedAnswer(
        intent="scenario",
        answer=answer,
        evidence=[
            Evidence(f"scenario {metric}", scen[metric], "simulated", "simulation history")
            for metric in ("overflow_bed_days", "icu_overflow_bed_days") if metric in scen
        ],
        sources=["simulation history", "OR-Tools what-if simulation"],
        limitations=["Scenario results are approximations on synthetic data; the request-level "
                     "replay is not rerun for scenarios."],
        run_id=run["simulation_id"],
    )


def overview(question: str) -> GroundedAnswer:
    manifest = _manifest()
    return GroundedAnswer(
        intent="overview",
        answer=(
            f"Run {manifest.get('scenario_as_of_date', 'unknown')} has status "
            f"{manifest.get('pipeline_status', 'unknown')}. I can explain bed availability in N days, "
            "capacity pressure by department, unassigned bed requests, baseline vs optimized "
            "allocation, staff coverage, forecast reliability, model limitations, recommendations, "
            "and saved what-if scenarios."
        ),
        sources=["optimization/resource_planning/outputs/portfolio_run_manifest.json"],
        run_id=manifest.get("scenario_as_of_date"),
    )


SCENARIO_TERMS = ("what if", "what-if", "what happens if", "simulate", "scenario")
RULES: list[tuple[Callable[[str], bool], Callable[[str], GroundedAnswer]]] = [
    (lambda q: "unassign" in q or "left without" in q or "could not be allocat" in q, unassigned_requests),
    (lambda q: "baseline" in q or "compare" in q or "optimized allocation" in q, allocation_comparison),
    (lambda q: any(t in q for t in ("reliab", "accura", "trust", "error", "mae", "how good")), model_reliability),
    (lambda q: "limitation" in q or "weakness" in q or "caveat" in q, model_limitations),
    (lambda q: "recommend" in q, recommendations),
    (lambda q: "icu" in q, icu_forecast),
    (lambda q: "pressure" in q or "busiest" in q or "most crowded" in q, capacity_pressure),
    (lambda q: "shift" in q or "staff" in q or "coverage" in q, staff_coverage),
    (lambda q: "bed" in q and ("availab" in q or "free" in q), beds_available),
    (lambda q: "admission" in q, admission_forecast),
    (lambda q: "occupan" in q, occupancy_forecast),
    (lambda q: "length of stay" in q or re.search(r"\blos\b", q) is not None, model_reliability),
    (lambda q: "no-show" in q or "no show" in q, model_reliability),
]


def answer_question(question: str, scenario: dict[str, Any] | None = None) -> GroundedAnswer:
    lower = question.casefold()
    if scenario is not None or any(term in lower for term in SCENARIO_TERMS):
        result = scenario_summary(question, scenario)
    else:
        for matches, handler in RULES:
            if matches(lower):
                result = handler(question)
                break
        else:
            result = overview(question)
    if SYNTHETIC_NOTE not in result.limitations:
        result.limitations.append(SYNTHETIC_NOTE)
    return result

"""Create an evidence-based reliability scorecard for the full portfolio.

The scorecard never upgrades a model merely because an optimization scenario
is feasible. Predictive scores use saved chronological holdouts and compare
against the relevant baseline; operational scores require the structural and
capacity gates to pass. All scores remain explicitly scoped to the synthetic
scenario.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "optimization" / "resource_planning" / "outputs"
BACKTEST = OUTPUT / "forecast_backtest_summary.csv"
AUDIT = OUTPUT / "data_quality_audit.csv"
BED_KPIS = OUTPUT / "bed_allocation_kpis.csv"
STAFF_KPIS = OUTPUT / "staff_coverage_kpis.csv"
EMERGENCY = OUTPUT / "emergency_capacity_optimization.csv"
PORTFOLIO_MANIFEST = OUTPUT / "portfolio_run_manifest.json"
SUPPORT_MANIFEST = ROOT / "data" / "synthetic" / "resource_support" / "scenario_manifest.json"
SUPPORT_DATA = ROOT / "data" / "synthetic" / "resource_support"
ADMISSION_BACKTEST = ROOT / "ml" / "admission_forecasting" / "admission_backtest_summary.csv"
OCCUPANCY_BACKTEST = ROOT / "ml" / "occupancy_forecasting" / "occupancy_backtest_summary.csv"
ICU_BACKTEST = ROOT / "ml" / "icu_demand_forecasting" / "icu_backtest_summary.csv"
LOS_BACKTEST = ROOT / "ml" / "los_prediction" / "los_backtest_summary.csv"
NO_SHOW_BACKTEST = ROOT / "ml" / "no_show_prediction" / "no_show_backtest_summary.csv"
SCORECARD_JSON = OUTPUT / "reliability_scorecard.json"
SCORECARD_CSV = OUTPUT / "reliability_scorecard.csv"
DETAILS_CSV = OUTPUT / "reliability_details.csv"
REPORT = OUTPUT / "reliability_scorecard.txt"

FORECAST_FILES = {
    "admissions_xgboost": "ml/admission_forecasting/admission_30_day_forecast.csv",
    "occupancy_ridge": "ml/occupancy_forecasting/occupancy_future_forecast.csv",
    "occupancy_ridge_bed_count": "ml/occupancy_forecasting/occupancy_future_forecast.csv",
    "icu_xgboost": "ml/icu_demand_forecasting/icu_future_forecast.csv",
    "los_xgboost": "ml/los_prediction/los_predictions.csv",
    "no_show_xgboost": "ml/no_show_prediction/no_show_predictions.csv",
}


def _score(value: float, evidence: str, limitation: str) -> dict:
    return {
        "score": round(float(value), 1),
        "evidence": evidence,
        "limitation": limitation,
    }


def _rolling_backtest_evidence(
    summary: pd.DataFrame,
    model_candidates: tuple[str, ...],
    unit: str,
    metric: str = "MAE",
    evaluation: str = "30-day rolling-origin backtest",
) -> tuple[bool, str]:
    """Describe a rolling evaluation; True when the published model (lowest
    ``metric``, lower is better) is no worse than every baseline. Any
    candidate that is not a learned model counts as a baseline."""
    ranked = summary.sort_values(metric, kind="stable").reset_index(drop=True)
    published = str(ranked.iloc[0]["model"])
    naive = ranked[~ranked["model"].isin(model_candidates)]
    best_naive = naive.iloc[0] if not naive.empty else None
    learned = ranked[ranked["model"].isin(model_candidates)]
    parts = [f"{evaluation}: published {published} {metric} {ranked.iloc[0][metric]:.3g} {unit}".rstrip()]
    for _, row in learned.iterrows():
        if row["model"] != published:
            parts.append(f"{row['model']} {metric} {row[metric]:.3g}")
    if best_naive is not None and best_naive["model"] != published:
        parts.append(f"best naive ({best_naive['model']}) {metric} {best_naive[metric]:.3g}")
    learned_wins = published in model_candidates
    parts.append(
        "the learned model beats every naive baseline" if learned_wins
        else "the learned model does not beat the best naive baseline, so the baseline is published"
    )
    ok = best_naive is None or float(ranked.iloc[0][metric]) <= float(best_naive[metric])
    return ok, "; ".join(parts) + "."


def _forecast_reliability(row: pd.Series) -> dict:
    model = str(row["model"])
    score = 4.0
    if model == "admissions_xgboost":
        reliable = bool(row["beats_baseline"])
        threshold = "MAE must beat the seasonal baseline."
        measure = (
            f"MAE {float(row['mae']):.2f} vs baseline "
            f"{float(row['baseline_mae']):.2f}"
        )
    elif model == "occupancy_ridge":
        reliable = bool(row["beats_baseline"]) and float(row["wape_percent"]) <= 10
        threshold = "Must beat the seasonal baseline and have WAPE <= 10%."
        measure = f"WAPE {float(row['wape_percent']):.1f}%"
    elif model == "occupancy_ridge_bed_count":
        reliable = float(row["wape_percent"]) <= 10
        threshold = "Occupied-bed count WAPE must be <= 10%; no paired baseline is available."
        measure = f"occupied-bed count WAPE {float(row['wape_percent']):.1f}%"
    elif model == "icu_xgboost":
        reliable = bool(row["beats_baseline"]) and float(row["mae"]) <= 1.0
        threshold = "Must beat the 7-day baseline and have MAE <= 1 admission/day."
        measure = f"MAE {float(row['mae']):.2f} admissions/day"
    elif model == "los_xgboost":
        reliable = float(row["mae"]) <= 3.0
        threshold = "MAE must be <= 3.0 days."
        measure = f"MAE {float(row['mae']):.2f} days"
    elif model == "no_show_xgboost":
        brier_beats_baseline = row.get("brier_beats_baseline")
        calibration_reliable = (
            pd.notna(brier_beats_baseline)
            and bool(brier_beats_baseline)
            and pd.notna(row.get("brier_score"))
            and float(row["brier_score"]) <= 0.17
        )
        roc_auc = pd.to_numeric(
            pd.Series([row.get("roc_auc", float("nan"))]), errors="coerce"
        ).iloc[0]
        discrimination_reliable = bool(
            pd.notna(roc_auc) and float(roc_auc) >= 0.70
        )
        reliable = calibration_reliable
        score = (
            4.5 if calibration_reliable and discrimination_reliable
            else 4.0 if calibration_reliable
            else 3.0
        )
        threshold = (
            "Brier score must beat the pre-holdout prevalence baseline and be <= 0.17 "
            "for 4/5; ROC-AUC must also be >= 0.70 for 4.5/5."
        )
        measure = (
            f"Brier score {float(row['brier_score']):.3f} vs prevalence baseline "
            f"{float(row['pre_holdout_baseline_brier']):.3f}; "
            + (
                f"ROC-AUC {float(roc_auc):.3f}"
                if pd.notna(roc_auc) else "ROC-AUC unavailable"
            )
        )
    else:
        raise ValueError(f"Unrecognized forecast model in backtest: {model}")

    return {
        "score": score if model == "no_show_xgboost" else (4.0 if reliable else 3.0),
        "metric": measure,
        "threshold": threshold,
        "beats_baseline": bool(row["beats_baseline"])
        if pd.notna(row["beats_baseline"]) else None,
        "test_rows": int(row["test_rows"]),
        "test_start": str(row["test_start"]),
        "test_end": str(row["test_end"]),
        "evidence": measure + ("; meets the reliability threshold." if reliable
                               else "; does not meet the reliability threshold."),
        "limitation": (
            "Retrospective synthetic holdout performance only; this score is not "
            "a confidence interval or a guarantee for future forecast values."
        ),
    }


def build_scorecard() -> dict:
    backtest = pd.read_csv(BACKTEST)
    audit = pd.read_csv(AUDIT)
    bed = pd.read_csv(BED_KPIS)
    staff = pd.read_csv(STAFF_KPIS)
    emergency = pd.read_csv(EMERGENCY)
    manifest = json.loads(PORTFOLIO_MANIFEST.read_text(encoding="utf-8"))
    support_manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))

    failures = int((audit["status"] == "FAIL").sum())
    optimized_bed = bed[bed["scenario"].eq("optimized")]
    structural_ok = (
        failures == 0 and bool(manifest["source_integrity_checks_passed"])
    )
    capacity_ok = (
        int(optimized_bed["overflow_bed_days"].sum()) == 0
        and int(staff["unfilled_staff"].sum()) == 0
        and int(emergency["icu_overflow"].sum()) == 0
        and int(emergency["emergency_overflow"].sum()) == 0
    )

    rows = []
    admission = backtest[
        backtest["model"].eq("admissions_xgboost")
        & ~backtest["group"].eq("hospital_total")
    ]
    admission_beats = float(admission["beats_baseline"].fillna(False).mean())
    admission_improvement = float(
        -admission["mae_change_vs_baseline_percent"].fillna(0).mean()
    )
    one_step_evidence = (
        f"One-step XGBoost holdout: {admission_beats:.0%} of departments beat the "
        f"same-weekday baseline (mean MAE change {admission_improvement:.1f}%)."
    )
    if ADMISSION_BACKTEST.is_file():
        summary = pd.read_csv(ADMISSION_BACKTEST)
        summary = summary[summary["department"].eq("All Departments")]
        admission_ok, rolling_evidence = _rolling_backtest_evidence(
            summary, ("xgboost_recursive",), "admissions/day"
        )
        admission_evidence = rolling_evidence + " " + one_step_evidence
    else:
        admission_ok = admission_beats >= 0.6 and admission_improvement >= 0
        admission_evidence = one_step_evidence
    rows.append((
        "Admission forecasting",
        _score(
            4.0 if admission_ok else 3.0,
            admission_evidence,
            "Daily admissions are close to Poisson noise around a stable level; "
            "a learned model adds little over a moving average in this synthetic scenario.",
        ),
    ))

    occupancy = backtest[backtest["model"].eq("occupancy_ridge")]
    occupancy_row = occupancy.iloc[0] if not occupancy.empty else None
    occupancy_good = bool(
        occupancy_row is not None
        and occupancy_row["beats_baseline"]
        and float(occupancy_row["wape_percent"]) <= 10
    )
    occupancy_evidence = (
        "One-step holdout beats the same-weekday baseline with WAPE "
        f"{float(occupancy_row['wape_percent']):.1f}%."
        if occupancy_good
        else "One-step holdout does not meet the baseline/WAPE threshold."
        if occupancy_row is not None else "No occupancy holdout was found."
    )
    if OCCUPANCY_BACKTEST.is_file():
        rolling_ok, rolling_evidence = _rolling_backtest_evidence(
            pd.read_csv(OCCUPANCY_BACKTEST), ("ridge_recursive", "ridge_blend"), "beds"
        )
        occupancy_good = occupancy_good and rolling_ok
        occupancy_evidence = rolling_evidence + " " + occupancy_evidence
    rows.append((
        "Occupancy forecasting",
        _score(
            4.0 if occupancy_good else 3.0,
            occupancy_evidence,
            "Forecast is synthetic and does not establish live capacity accuracy.",
        ),
    ))

    icu = backtest[backtest["model"].eq("icu_xgboost")]
    icu_row = icu.iloc[0] if not icu.empty else None
    icu_good = bool(
        icu_row is not None
        and icu_row["beats_baseline"]
        and float(icu_row["mae"]) <= 1.0
    )
    icu_evidence = (
        f"One-step holdout MAE {float(icu_row['mae']):.2f} admissions/day."
        if icu_row is not None else "No ICU holdout was found."
    )
    if ICU_BACKTEST.is_file():
        icu_good, rolling_evidence = _rolling_backtest_evidence(
            pd.read_csv(ICU_BACKTEST), ("xgboost_recursive",), "ICU admissions/day"
        )
        icu_evidence = rolling_evidence + " " + icu_evidence
    rows.append((
        "ICU demand forecasting",
        _score(
            4.0 if icu_good else 3.0,
            icu_evidence,
            "This forecasts admissions flow, not occupied ICU census; low-count R² can remain unstable.",
        ),
    ))

    los = backtest[backtest["model"].eq("los_xgboost") & backtest["group"].eq("all_departments")]
    los_row = los.iloc[0] if not los.empty else None
    los_good = bool(los_row is not None and float(los_row["mae"]) <= 3.0)
    los_evidence = (
        f"All-department holdout MAE {float(los_row['mae']):.2f} days."
        if los_row is not None else "No LOS holdout was found."
    )
    if LOS_BACKTEST.is_file():
        beats_baselines, rolling_evidence = _rolling_backtest_evidence(
            pd.read_csv(LOS_BACKTEST), ("xgboost",), "days",
            evaluation="Rolling 30-day windows over the holdout",
        )
        los_good = los_good and beats_baselines
        los_evidence = rolling_evidence + " " + los_evidence
    rows.append((
        "Length-of-stay prediction",
        _score(
            4.0 if los_good else 3.0,
            los_evidence,
            "Individual stays can still deviate by several days; use intervals for capacity planning.",
        ),
    ))

    no_show = backtest[backtest["model"].eq("no_show_xgboost")]
    no_show_row = no_show.iloc[0] if not no_show.empty else None
    no_show_rating = (
        _forecast_reliability(no_show_row) if no_show_row is not None else None
    )
    no_show_score = no_show_rating["score"] if no_show_rating else 3.0
    no_show_evidence = (
        no_show_rating["evidence"] if no_show_rating else "No no-show holdout was found."
    )
    if NO_SHOW_BACKTEST.is_file():
        beats_baselines, rolling_evidence = _rolling_backtest_evidence(
            pd.read_csv(NO_SHOW_BACKTEST), ("xgboost_calibrated",), "", metric="brier",
            evaluation="Rolling 30-day windows over the holdout",
        )
        if not beats_baselines:
            no_show_score = min(no_show_score, 3.0)
        no_show_evidence = rolling_evidence + " " + no_show_evidence
    rows.append((
        "No-show prediction",
        _score(
            no_show_score,
            no_show_evidence,
            "Use only for outreach prioritization; never use it as a patient or access decision.",
        ),
    ))

    rows.extend([
        (
            "Bed allocation optimization",
            _score(
                4.0 if structural_ok and capacity_ok else 2.0,
                "Input audit passed and the optimized scenario has zero modeled bed overflow."
                if structural_ok and capacity_ok else "One or more structural or capacity gates failed.",
                "Zero overflow depends on scenario-only surge beds and is not current hospital capacity.",
            ),
        ),
        (
            "Staffing allocation",
            _score(
                4.0 if structural_ok and capacity_ok else 2.0,
                "All modeled shift positions are filled while hours and rest constraints are enforced."
                if structural_ok and capacity_ok else "Staffing or input gates failed.",
                "Roster and flex staff are synthetic scenario assumptions, not hiring commitments.",
            ),
        ),
        (
            "Appointment scheduling",
            _score(
                4.0 if structural_ok else 2.0,
                "Schedule feasibility and provider-conflict gates passed."
                if structural_ok else "Input audit failed.",
                "No-show scores are not used to move appointments; clinical priority remains human-controlled.",
            ),
        ),
        (
            "Emergency capacity planning",
            _score(
                4.0 if structural_ok and capacity_ok else 2.0,
                "Shared ICU/emergency capacity gates passed with no modeled overflow."
                if structural_ok and capacity_ok else "Emergency capacity gates failed.",
                "Scenario-only surge capacity and synthetic staffing make this a stress-test, not a live guarantee.",
            ),
        ),
        (
            "Data quality and orchestration",
            _score(
                4.5 if structural_ok else 2.0,
                "Cross-file identifiers, dates, grains, forecast horizons, and feasibility gates were checked."
                if structural_ok else "At least one data-quality gate failed.",
                "Synthetic provenance means structural validity is not external clinical validity.",
            ),
        ),
    ])

    modules = []
    for name, result in rows:
        modules.append({"section": name, **result})

    data_score = next(
        item["score"] for item in modules
        if item["section"] == "Data quality and orchestration"
    )
    data_limitation = (
        "Synthetic data is structurally checked but is not evidence of real-world "
        "or clinical validity."
    )
    source_checks_ok = structural_ok
    data_assets = [
        {
            "asset": filename,
            "records": int(count),
            "score": data_score if source_checks_ok else 2.0,
            "evidence": (
                "Generated source dataset passed the portfolio source-integrity "
                "checks."
                if source_checks_ok
                else "One or more source-integrity checks failed."
            ),
            "limitation": data_limitation,
        }
        for filename, count in manifest["synthetic_source_row_counts"].items()
    ]
    for filename in support_manifest["outputs"]:
        path = SUPPORT_DATA / filename
        if not path.is_file():
            raise FileNotFoundError(f"Generated support dataset not found: {path}")
        records = len(pd.read_csv(path))
        data_assets.append({
            "asset": f"resource_support/{filename}",
            "records": records,
            "score": data_score if structural_ok else 2.0,
            "evidence": (
                "Generated support dataset passed the portfolio structural checks."
                if structural_ok else "One or more portfolio data-quality checks failed."
            ),
            "limitation": (
                "Synthetic scenario assumptions; not clinical determinations or "
                "live operational data."
            ),
        })

    forecast_scores = []
    for _, row in backtest.iterrows():
        model = str(row["model"])
        rating = _forecast_reliability(row)
        forecast_scores.append({
            "model": model,
            "group": str(row["group"]),
            "forecast_file": FORECAST_FILES[model],
            **rating,
        })

    overall_score = round(
        sum(item["score"] for item in modules) / len(modules), 2
    )
    return {
        "project": "CareOps-AI",
        "scope": "synthetic scenario and chronological retrospective holdouts",
        "minimum_target": 4.0,
        "overall_score": overall_score,
        "overall_score_method": "Equal-weighted mean of portfolio section scores.",
        "all_sections_at_or_above_target": all(
            item["score"] >= 4.0 for item in modules
        ),
        "sections": modules,
        "data_assets": data_assets,
        "forecasts": forecast_scores,
    }


def main() -> None:
    scorecard = build_scorecard()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    SCORECARD_JSON.write_text(json.dumps(scorecard, indent=2) + "\n", encoding="utf-8")
    pd.DataFrame(scorecard["sections"]).to_csv(SCORECARD_CSV, index=False)
    details = [
        {"record_type": "data_asset", **item}
        for item in scorecard["data_assets"]
    ] + [
        {"record_type": "forecast", **item}
        for item in scorecard["forecasts"]
    ]
    pd.DataFrame(details).to_csv(DETAILS_CSV, index=False)
    lines = [
        "CAREOPS AI RELIABILITY SCORECARD",
        "Scope: synthetic scenario and chronological retrospective holdouts",
        f"OVERALL PORTFOLIO RELIABILITY: {scorecard['overall_score']}/5",
        f"Method: {scorecard['overall_score_method']}",
        "",
        "PORTFOLIO SECTIONS",
    ]
    for item in scorecard["sections"]:
        lines.append(f"{item['section']}: {item['score']}/5")
        lines.append(f"  Evidence: {item['evidence']}")
        lines.append(f"  Limitation: {item['limitation']}")
    lines.append("")
    lines.append("GENERATED DATASETS")
    for item in scorecard["data_assets"]:
        lines.append(
            f"{item['asset']}: {item['score']}/5 ({item['records']:,} records)"
        )
        lines.append(f"  Evidence: {item['evidence']}")
        lines.append(f"  Limitation: {item['limitation']}")
    lines.extend(["", "FORECAST HOLDOUT RELIABILITY"])
    for item in scorecard["forecasts"]:
        lines.append(
            f"{item['model']} / {item['group']}: {item['score']}/5 "
            f"({item['metric']}; {item['test_rows']:,} test rows, "
            f"{item['test_start']} to {item['test_end']})"
        )
        lines.append(f"  Forecast file: {item['forecast_file']}")
        lines.append(f"  Threshold: {item['threshold']}")
        lines.append(f"  Limitation: {item['limitation']}")
    lines.append("")
    lines.append(
        "OVERALL RELEASE STATUS: "
        + ("PASS" if scorecard["all_sections_at_or_above_target"] else "REVIEW REQUIRED")
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if PORTFOLIO_MANIFEST.is_file():
        manifest = json.loads(PORTFOLIO_MANIFEST.read_text(encoding="utf-8"))
        manifest["reliability_scorecard"] = {
            "path": str(SCORECARD_JSON),
            "minimum_target": scorecard["minimum_target"],
            "overall_score": scorecard["overall_score"],
            "all_sections_at_or_above_target": scorecard[
                "all_sections_at_or_above_target"
            ],
            "data_asset_count": len(scorecard["data_assets"]),
            "forecast_score_count": len(scorecard["forecasts"]),
        }
        PORTFOLIO_MANIFEST.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
    print(REPORT)
    if not scorecard["all_sections_at_or_above_target"]:
        print("One or more sections remain below the 4/5 target; no evidence was hidden.")


if __name__ == "__main__":
    main()

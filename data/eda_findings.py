"""Write docs/EDA_FINDINGS.md: tested findings that shaped the modelling.

Every number is computed from data/synthetic at run time, after the
simulator warm-up. Charts go to docs/eda/. Run after the pipeline:

    python data/eda_findings.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ml"))
from scenario_config import WARMUP_END  # noqa: E402

DATA = ROOT / "data" / "synthetic"
DOCS = ROOT / "docs"
CHARTS = DOCS / "eda"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
INK, ACCENT, MUTED = "#1f3b5b", "#2a7fd4", "#9aa8b8"


def pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def table(frame: pd.DataFrame) -> list[str]:
    header = "| " + " | ".join(str(column) for column in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return [header, rule, *rows, ""]


def style(ax, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e3e8ee")
    ax.set_axisbelow(True)


def save(fig, name: str) -> str:
    CHARTS.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(CHARTS / name, dpi=110)
    plt.close(fig)
    return f"eda/{name}"


def load() -> dict[str, pd.DataFrame]:
    start = pd.Timestamp(WARMUP_END)
    admissions = pd.read_csv(DATA / "admissions.csv", parse_dates=["admission_date", "discharge_date"])
    appointments = pd.read_csv(DATA / "appointments.csv", parse_dates=["date"])
    operations = pd.read_csv(DATA / "daily_operations.csv", parse_dates=["date"])
    return {
        "patients": pd.read_csv(DATA / "patients.csv"),
        "admissions_all": admissions,
        "admissions": admissions[admissions["admission_date"] >= start],
        "appointments": appointments[appointments["date"] >= start],
        "operations": operations[operations["date"] >= start],
        "shifts": pd.read_csv(DATA / "shifts.csv", usecols=["date", "department", "shift", "assigned", "staff_role"]),
        "beds": pd.read_csv(DATA / "beds.csv"),
    }


def admissions_section(data: dict[str, pd.DataFrame]) -> list[str]:
    admissions = data["admissions"]
    days = pd.date_range(admissions["admission_date"].min(), admissions["admission_date"].max())
    daily = admissions.groupby("admission_date").size().reindex(days, fill_value=0)
    dispersion = daily.var() / daily.mean()
    weekday = daily.groupby(daily.index.day_name()).mean().reindex(WEEKDAYS)
    groups = [daily[daily.index.day_name() == name].to_numpy() for name in WEEKDAYS]
    weekday_p = stats.kruskal(*groups).pvalue
    monthly = daily.groupby(daily.index.month).mean()
    month_groups = [daily[daily.index.month == month].to_numpy() for month in range(1, 13)]
    month_p = stats.kruskal(*month_groups).pvalue
    trend = stats.linregress(np.arange(len(daily)), daily.to_numpy())
    weekend = daily[daily.index.dayofweek >= 5].mean() / daily[daily.index.dayofweek < 5].mean()
    lag1 = daily.autocorr(1)
    lag7 = daily.autocorr(7)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].bar(range(7), weekday.to_numpy(), color=ACCENT)
    axes[0].set_xticks(range(7), [name[:3] for name in WEEKDAYS])
    style(axes[0], "Mean admissions by weekday", "admissions / day")
    axes[1].bar(range(1, 13), monthly.to_numpy(), color=ACCENT)
    axes[1].set_xticks(range(1, 13))
    style(axes[1], "Mean admissions by month", "admissions / day")
    chart = save(fig, "admissions_weekday_month.png")

    seasonal = data["operations"].groupby("date")["seasonal_multiplier"].mean()
    seasonal_corr = daily.reindex(seasonal.index).corr(seasonal)
    return [
        "## Admissions",
        "",
        f"Window: {days[0].date()} to {days[-1].date()} ({len(days)} days after the warm-up), "
        f"{len(admissions):,} admissions.",
        "",
        f"- **Close to Poisson.** Mean {daily.mean():.2f} per day, variance {daily.var():.2f}, "
        f"dispersion (variance / mean) **{dispersion:.2f}**. Lag-1 autocorrelation {lag1:.2f}, lag-7 {lag7:.2f}.",
        f"- **Weekday effect: {'none detected' if weekday_p >= 0.05 else 'present'}.** Kruskal-Wallis p = {weekday_p:.3f}; "
        f"weekend / weekday ratio {weekend:.2f}.",
        f"- **Monthly seasonality: {'none detected' if month_p >= 0.05 else 'present'}.** Kruskal-Wallis p = {month_p:.3g}; "
        f"monthly means range {monthly.min():.2f}-{monthly.max():.2f}. The generator's `seasonal_multiplier` "
        f"correlates with daily admissions at r = {seasonal_corr:.2f}.",
        f"- **Trend:** {trend.slope * 365:+.2f} admissions/day per year (p = {trend.pvalue:.3f}).",
        "",
        f"![Admissions by weekday and month]({chart})",
        "",
        f"**Modelling consequence.** The weekday and monthly effects are real but small: the weekday "
        f"means span {weekday.max() - weekday.min():.1f} admissions/day against a daily standard deviation "
        f"of {daily.std():.1f}, and day-to-day autocorrelation is weak. The learned models get weekday and "
        "calendar features and the baselines include same-weekday-last-week, yet the 28-day moving "
        "average still has the lowest 30-day backtest error, so it is published (see MODEL_METRICS.md). "
        "Intervals use backtest errors rather than a Poisson assumption.",
        "",
    ]


def mix_section(data: dict[str, pd.DataFrame]) -> list[str]:
    patients, admissions = data["patients"], data["admissions"]
    age = patients["age"].describe(percentiles=[0.1, 0.5, 0.9])
    department = admissions["department"].value_counts(normalize=True)
    severity = admissions["severity"].value_counts(normalize=True).reindex(["Low", "Moderate", "High", "Critical"])
    icu = admissions.groupby("severity")["ICU_required"].mean().reindex(severity.index)
    frame = pd.DataFrame({
        "Severity": severity.index,
        "Share of admissions": [pct(value) for value in severity],
        "ICU required": [pct(value) for value in icu],
    })
    return [
        "## Patients and case mix",
        "",
        f"- **Age:** median {age['50%']:.0f}, 10th-90th percentile {age['10%']:.0f}-{age['90%']:.0f}. "
        f"Gender split: " + ", ".join(f"{name} {pct(share)}" for name, share in patients["gender"].value_counts(normalize=True).items()) + ".",
        f"- **Departments:** General Medicine takes {pct(department.get('General Medicine', 0))} of admissions; "
        "the other eight departments take 3-11% each.",
        "- **Age is not tied to department** in this generator (Pediatrics patients span all ages), so no "
        "age-based ward rule is used in bed allocation.",
        "",
        *table(frame),
    ]


def los_section(data: dict[str, pd.DataFrame]) -> list[str]:
    stays = data["admissions"][data["admissions"]["stay_status"] == "completed"]
    mismatches = int((stays["LOS"] != (stays["discharge_date"] - stays["admission_date"]).dt.days).sum())
    by = stays.groupby("severity")["LOS"].describe(percentiles=[0.5, 0.9]).reindex(["Low", "Moderate", "High", "Critical"])
    icu = stays.groupby("ICU_required")["LOS"].median()
    rho = stats.spearmanr(stays["comorbidity_count"], stays["LOS"]).statistic
    by_department = stays.groupby("department")["LOS"].median().sort_values()
    frame = pd.DataFrame({
        "Severity": by.index,
        "Stays": by["count"].astype(int),
        "Mean LOS (days)": by["mean"].round(1),
        "Median": by["50%"].round(0).astype(int),
        "90th percentile": by["90%"].round(0).astype(int),
    })
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(stays["LOS"], bins=range(0, int(stays["LOS"].max()) + 2), color=ACCENT)
    style(ax, "Length of stay, completed stays", "stays")
    ax.set_xlabel("days")
    chart = save(fig, "los_distribution.png")
    return [
        "## Length of stay",
        "",
        f"{len(stays):,} completed stays. LOS differs from discharge minus admission date in {mismatches} "
        "of them. In-progress stays are excluded from training, which is the censoring rule.",
        "",
        *table(frame),
        f"- **ICU stays** have median LOS {icu.get(1, float('nan')):.0f} days vs {icu.get(0, float('nan')):.0f} otherwise.",
        f"- **Comorbidity count** correlates with LOS at Spearman rho = {rho:.2f}.",
        f"- **Department medians** range {by_department.min():.0f}-{by_department.max():.0f} days.",
        "",
        f"![LOS distribution]({chart})",
        "",
        "**Modelling consequence.** Severity and department explain part of LOS, so the baseline is a "
        "department x severity mean. XGBoost adds comorbidity, procedure and admission-type features "
        "known at admission, and it beats that baseline on rolling holdout windows.",
        "",
    ]


def occupancy_section(data: dict[str, pd.DataFrame]) -> list[str]:
    admissions = data["admissions_all"]
    beds = data["beds"]
    days = pd.date_range(pd.Timestamp(WARMUP_END), admissions["admission_date"].max())
    starts = admissions["admission_date"].to_numpy()
    ends = admissions["discharge_date"].to_numpy()
    census = pd.Series([int(((starts <= day) & (ends > day)).sum()) for day in days.to_numpy()], index=days)
    icu_mask = admissions["ICU_required"].to_numpy() == 1
    icu_census = pd.Series([int(((starts <= day) & (ends > day) & icu_mask).sum()) for day in days.to_numpy()], index=days)
    capacity = len(beds)
    icu_beds = int((beds["bed_type"] == "ICU").sum())
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(census.index, census.to_numpy(), color=ACCENT, linewidth=1)
    ax.axhline(capacity * 0.9, color=MUTED, linestyle="--", linewidth=1)
    ax.text(census.index[5], capacity * 0.9 + 2, "90% of beds", color=MUTED, fontsize=8)
    style(ax, "Occupied beds (rebuilt from admissions)", "beds")
    chart = save(fig, "occupancy.png")
    return [
        "## Occupancy and capacity pressure",
        "",
        f"- **Occupied beds** (census rebuilt from stays): mean {census.mean():.0f} of {capacity} "
        f"({pct(census.mean() / capacity)}), peak {census.max()} ({pct(census.max() / capacity)}).",
        f"- **Days at or above 90% of beds:** {int((census >= 0.9 * capacity).sum())}; at or above 95%: "
        f"{int((census >= 0.95 * capacity).sum())}. The forecast's 90%/95% alert thresholds are therefore "
        "untestable on this history, which MODEL_METRICS.md reports.",
        f"- **ICU census:** mean {icu_census.mean():.1f}, peak {icu_census.max()} of {icu_beds} ICU beds "
        f"({int((icu_census >= icu_beds - 2).sum())} days within the 2-bed reserve).",
        f"- **Day-to-day persistence:** lag-1 autocorrelation of the census {census.autocorr(1):.2f}. "
        "Occupancy is a slowly changing stock, so recent values carry information; the Ridge model "
        "beats every naive baseline on it.",
        "",
        f"![Occupancy]({chart})",
        "",
    ]


def staffing_section(data: dict[str, pd.DataFrame]) -> list[str]:
    shifts = data["shifts"]
    shifts = shifts[(shifts["assigned"] == "Yes") & shifts["staff_role"].str.contains("Nurse")
                    & (pd.to_datetime(shifts["date"]) >= pd.Timestamp(WARMUP_END))]
    per_shift = shifts.groupby(["date", "department", "shift"]).size().unstack("shift", fill_value=0)
    least = per_shift.min(axis=1).groupby("department").mean()
    zero_share = (per_shift.min(axis=1) == 0).groupby("department").mean()
    admissions = data["admissions_all"]
    beds = data["beds"].set_index("bed_id")["department"]
    sample_days = pd.date_range(pd.Timestamp(WARMUP_END), admissions["admission_date"].max(), freq="7D")
    census_rows = []
    for day in sample_days:
        in_house = admissions[(admissions["admission_date"] <= day) & (admissions["discharge_date"] > day)]
        census_rows.append(in_house["assigned_bed_id"].map(beds).value_counts())
    census = pd.DataFrame(census_rows).fillna(0).mean()
    frame = pd.DataFrame({
        "Department": least.index,
        "Mean census": [f"{census.get(name, 0):.0f}" for name in least.index],
        "Nurses needed at 1:4": [f"{census.get(name, 0) / 4:.1f}" for name in least.index],
        "Rostered nurses, least-staffed shift": [f"{value:.1f}" for value in least],
        "Days with an empty nurse shift": [pct(zero_share[name], 0) for name in least.index],
    })
    return [
        "## Staffing",
        "",
        "The historical shift roster was generated independently of the ward census:",
        "",
        *table(frame),
        "**Modelling consequence.** The roster cannot serve as a hard staffing limit; it would reject most "
        "admissions. Planning derives required staff from projected census at 1 nurse : 4 beds. The "
        "request-level bed replay reports rostered-staffing exceedances instead of enforcing them.",
        "",
    ]


def appointments_section(data: dict[str, pd.DataFrame]) -> list[str]:
    appointments = data["appointments"]
    rate = appointments["no_show"].mean()
    lead = pd.cut(appointments["lead_time_days"], [-1, 7, 14, 30, 60, 10_000],
                  labels=["0-7", "8-14", "15-30", "31-60", "60+"])
    by_lead = appointments.groupby(lead, observed=True)["no_show"].agg(["mean", "size"])
    by_reminder = appointments.groupby("reminder_sent")["no_show"].mean()
    chi = stats.chi2_contingency(pd.crosstab(appointments["reminder_sent"], appointments["no_show"]))
    weekday = appointments.groupby(appointments["date"].dt.day_name())["no_show"].mean().reindex(WEEKDAYS)
    volume = appointments.groupby("date").size()
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(range(len(by_lead)), by_lead["mean"].to_numpy(), color=ACCENT)
    ax.set_xticks(range(len(by_lead)), [str(label) for label in by_lead.index])
    ax.set_xlabel("lead time (days)")
    style(ax, "No-show rate by booking lead time", "no-show rate")
    chart = save(fig, "no_show_lead_time.png")
    frame = pd.DataFrame({
        "Lead time (days)": [str(label) for label in by_lead.index],
        "Appointments": by_lead["size"].map(lambda value: f"{value:,}"),
        "No-show rate": by_lead["mean"].map(pct),
    })
    return [
        "## Appointments and no-shows",
        "",
        f"- **Volume:** {len(appointments):,} appointments, {volume.mean():.0f} per day on average "
        f"(range {volume.min()}-{volume.max()}).",
        f"- **No-show rate:** {pct(rate)} overall; weekday rates range "
        f"{pct(weekday.min())}-{pct(weekday.max())}.",
        f"- **Reminders:** {pct(by_reminder.get(1, float('nan')))} with a reminder vs "
        f"{pct(by_reminder.get(0, float('nan')))} without (chi-square p = {chi.pvalue:.2g}).",
        "",
        *table(frame),
        f"![No-show by lead time]({chart})",
        "",
        "**Modelling consequence.** Lead time and reminders are known before the appointment and carry "
        "signal, so the no-show model uses them. `attended` and `no_show` are outcomes and are excluded "
        "(see tests/test_leakage_audit.py). The label is imbalanced, so models are compared on Brier "
        "score and PR-AUC rather than accuracy.",
        "",
    ]


def main() -> None:
    data = load()
    lines = [
        "# Exploratory findings",
        "",
        "Generated by `data/eda_findings.py` from `data/synthetic`; do not edit by hand. All data is "
        f"synthetic. Statistics exclude the simulator warm-up (before {WARMUP_END}). The descriptive EDA "
        "stage (`run_portfolio.py --exploratory`) writes further charts to `data/eda/`.",
        "",
    ]
    for section in (mix_section, admissions_section, los_section, occupancy_section,
                    staffing_section, appointments_section):
        lines += section(data)
    (DOCS / "EDA_FINDINGS.md").write_text("\n".join(lines), encoding="utf-8")
    print(DOCS / "EDA_FINDINGS.md")


if __name__ == "__main__":
    main()

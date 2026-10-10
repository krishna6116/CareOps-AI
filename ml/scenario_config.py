"""Shared facts about the synthetic scenario that every model must respect."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


# The generator (data/synthetic/generate_data.py, START_DATE) opens an empty
# hospital on this date.
SIMULATION_START = date(2024, 1, 1)

# Warm-up: with no patients on day 1, discharges stay near zero for the first
# week and only reach the admission rate after about two weeks; occupancy
# overshoots while the first cohort is still in hospital. These days do not
# describe steady-state operations, so they are excluded from training,
# validation and evaluation. tests/test_scenario_config.py checks this
# criterion against the generated data.
WARMUP_DAYS = 21
WARMUP_END = SIMULATION_START + timedelta(days=WARMUP_DAYS)  # first usable day


def drop_warmup(frame: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """Rows on or after the first post-warm-up day."""
    dates = pd.to_datetime(frame[date_column], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"{date_column} contains missing or invalid dates.")
    return frame.loc[dates.dt.normalize() >= pd.Timestamp(WARMUP_END)].copy()


def drop_warmup_index(series: pd.Series) -> pd.Series:
    """Date-indexed series restricted to post-warm-up days."""
    return series.loc[series.index >= pd.Timestamp(WARMUP_END)]

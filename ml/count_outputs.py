"""Output-layer rules for count forecasts: whole numbers, non-negative, capped.

Models may work with continuous expected values internally. Anything written
for people or downstream planners that represents a count of patients, beds,
or appointments goes through these helpers first.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("Count forecasts must be finite before conversion to whole numbers.")
    return array


def round_half_up(values: Iterable[float]) -> np.ndarray:
    """Round to the nearest whole number with .5 always rounded up."""
    return np.floor(_as_float_array(values) + 0.5)


def to_stock_counts(
    values: Iterable[float],
    capacity: float | Iterable[float] | None = None,
) -> np.ndarray:
    """Convert a stock quantity (e.g. occupied beds on a day) to whole counts.

    Each value is rounded independently, clipped at zero, and capped at
    ``capacity`` (a scalar or one capacity per value) when given.
    """
    counts = np.clip(round_half_up(values), 0, None)
    if capacity is not None:
        cap = np.floor(np.asarray(capacity, dtype=float))
        if (cap < 0).any():
            raise ValueError("Capacity cannot be negative.")
        counts = np.minimum(counts, cap)
    return counts.astype(np.int64)


def apportion_to_total(values: Iterable[float]) -> np.ndarray:
    """Round parts to whole numbers that sum to the rounded total.

    Largest-remainder rounding: each part is floored, then the parts with
    the largest fractional remainders receive the units still needed to
    reach the rounded total. Used when department counts on a day must add
    up to a hospital count.
    """
    parts = np.clip(_as_float_array(values), 0, None)
    counts = np.floor(parts)
    remainder = int(round_half_up([parts.sum()])[0] - counts.sum())
    if remainder > 0:
        order = np.argsort(-(parts - counts), kind="stable")
        counts[order[:remainder]] += 1
    return counts.astype(np.int64)


def to_flow_counts(values: Iterable[float]) -> np.ndarray:
    """Convert a daily flow forecast (e.g. admissions per day) to whole counts.

    Rounding each low-volume day independently biases totals (1.3 per day
    rounds to 1 every day). Rounding the running total and differencing keeps
    every day whole and non-negative while the period total stays within half
    a unit of the expected value.
    """
    expected = np.clip(_as_float_array(values), 0, None)
    if expected.size == 0:
        return expected.astype(np.int64)
    cumulative = round_half_up(np.cumsum(expected))
    return np.diff(cumulative, prepend=0.0).astype(np.int64)


def reconcile_flow_counts(expected: pd.DataFrame) -> pd.DataFrame:
    """Whole daily flows for several groups (columns) over days (rows).

    The total row is rounded first with :func:`to_flow_counts`, so the
    total path has no rounding noise. Each day's units then go to the groups
    furthest behind their expected running total, which keeps every group's
    running total within about one unit of its expected value.
    """
    values = np.clip(expected.to_numpy(dtype=float), 0, None)
    if not np.isfinite(values).all():
        raise ValueError("Count forecasts must be finite before conversion to whole numbers.")
    totals = to_flow_counts(values.sum(axis=1))
    expected_running = np.zeros(values.shape[1])
    published_running = np.zeros(values.shape[1])
    counts = np.zeros(values.shape, dtype=np.int64)
    for day, target in enumerate(totals):
        expected_running += values[day]
        deficit = expected_running - published_running
        for _ in range(int(target)):
            # Largest shortfall first; ties go to the larger expected flow.
            group = np.lexsort((-values[day], -deficit))[0]
            counts[day, group] += 1
            deficit[group] -= 1
        published_running += counts[day]
    return pd.DataFrame(counts, index=expected.index, columns=expected.columns)


def flow_counts_by_group(
    frame: pd.DataFrame,
    value_column: str,
    group_column: str,
    order_column: str = "date",
) -> pd.Series:
    """Apply :func:`to_flow_counts` within each group in chronological order."""
    ordered = frame.sort_values([group_column, order_column])
    counts = ordered.groupby(group_column, sort=False)[value_column].transform(
        lambda series: pd.Series(to_flow_counts(series.to_numpy()), index=series.index)
    )
    return counts.reindex(frame.index).astype("int64")


def assert_count_column(
    frame: pd.DataFrame,
    column: str,
    capacity: float | pd.Series | None = None,
) -> None:
    """Raise if a published count column breaks the integer/non-negative/capacity rules."""
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any():
        raise ValueError(f"{column} contains missing or non-numeric counts.")
    if not np.equal(np.mod(values, 1), 0).all():
        raise ValueError(f"{column} contains non-integer counts.")
    if (values < 0).any():
        raise ValueError(f"{column} contains negative counts.")
    if capacity is not None and (values > capacity).any():
        raise ValueError(f"{column} exceeds capacity.")

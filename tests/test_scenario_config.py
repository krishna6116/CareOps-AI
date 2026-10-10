import unittest
from pathlib import Path

import pandas as pd

from ml.scenario_config import SIMULATION_START, WARMUP_END, drop_warmup


ROOT = Path(__file__).resolve().parents[1]
OPERATIONS = ROOT / "data" / "synthetic" / "daily_operations.csv"


class DropWarmupTests(unittest.TestCase):
    def test_rows_before_warmup_end_are_removed(self):
        frame = pd.DataFrame({
            "date": pd.date_range(SIMULATION_START, periods=30, freq="D"),
            "value": range(30),
        })

        kept = drop_warmup(frame)

        self.assertEqual(kept["date"].min(), pd.Timestamp(WARMUP_END))
        self.assertEqual(len(kept), 30 - (WARMUP_END - SIMULATION_START).days)

    def test_datetimes_on_the_first_usable_day_are_kept(self):
        frame = pd.DataFrame({"when": [f"{WARMUP_END} 08:30", f"{WARMUP_END} 00:00"]})

        self.assertEqual(len(drop_warmup(frame, "when")), 2)

    def test_invalid_dates_are_rejected(self):
        with self.assertRaises(ValueError):
            drop_warmup(pd.DataFrame({"date": ["not-a-date"]}))


@unittest.skipUnless(OPERATIONS.is_file(), "generated synthetic data is not present")
class WarmupMatchesGeneratedDataTests(unittest.TestCase):
    """The cutoff must reflect the generator: an empty hospital at the start
    whose discharges have caught up with admissions by the end of warm-up."""

    @classmethod
    def setUpClass(cls):
        operations = pd.read_csv(OPERATIONS, parse_dates=["date"])
        cls.daily = operations.groupby("date")[["admissions", "discharges"]].sum()

    def test_generator_starts_on_the_configured_date(self):
        self.assertEqual(self.daily.index.min().date(), SIMULATION_START)

    def test_first_week_is_not_steady_state(self):
        first_week = self.daily.iloc[:7].sum()

        self.assertLess(first_week["discharges"], 0.25 * first_week["admissions"])

    def test_discharges_have_caught_up_by_the_end_of_warmup(self):
        week_after = self.daily.loc[pd.Timestamp(WARMUP_END):].iloc[:7].sum()

        self.assertGreater(week_after["discharges"], 0.8 * week_after["admissions"])


if __name__ == "__main__":
    unittest.main()

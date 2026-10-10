import unittest

import numpy as np
import pandas as pd

from ml.count_outputs import (
    apportion_to_total,
    assert_count_column,
    flow_counts_by_group,
    reconcile_flow_counts,
    round_half_up,
    to_flow_counts,
    to_stock_counts,
)


class StockCountTests(unittest.TestCase):
    def test_values_become_whole_numbers(self):
        counts = to_stock_counts([141.6, 142.4, 0.5])

        self.assertEqual(counts.dtype, np.int64)
        self.assertEqual(counts.tolist(), [142, 142, 1])

    def test_negative_values_are_clipped_to_zero(self):
        self.assertEqual(to_stock_counts([-3.2, -0.4]).tolist(), [0, 0])

    def test_capacity_caps_occupied_beds(self):
        self.assertEqual(to_stock_counts([229.6, 250.0], capacity=230).tolist(), [230, 230])

    def test_capacity_can_differ_per_row(self):
        counts = to_stock_counts([12.7, 12.7], capacity=[10, 20])

        self.assertEqual(counts.tolist(), [10, 13])

    def test_non_finite_values_are_rejected(self):
        with self.assertRaises(ValueError):
            to_stock_counts([1.0, np.nan])

    def test_half_rounds_up(self):
        self.assertEqual(round_half_up([0.5, 1.5, 2.5]).tolist(), [1.0, 2.0, 3.0])


class FlowCountTests(unittest.TestCase):
    def test_low_volume_flow_keeps_period_total(self):
        expected = np.full(30, 1.3)

        counts = to_flow_counts(expected)

        self.assertEqual(counts.dtype, np.int64)
        self.assertTrue((counts >= 0).all())
        self.assertLessEqual(abs(counts.sum() - expected.sum()), 0.5)
        # Independent rounding would publish 30 admissions instead of 39.
        self.assertEqual(int(np.rint(expected).sum()), 30)
        self.assertEqual(int(counts.sum()), 39)

    def test_flow_never_negative(self):
        counts = to_flow_counts([0.2, -1.0, 0.0, 0.4, 3.9])

        self.assertTrue((counts >= 0).all())

    def test_empty_flow(self):
        self.assertEqual(to_flow_counts([]).tolist(), [])

    def test_grouped_flow_counts_follow_date_order(self):
        frame = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-02", "2026-01-01", "2026-01-01", "2026-01-02"]),
            "department": ["A", "A", "B", "B"],
            "expected": [0.6, 0.6, 2.2, 2.2],
        })

        counts = flow_counts_by_group(frame, "expected", "department")

        self.assertEqual(counts.tolist(), [0, 1, 2, 2])
        self.assertEqual(counts.dtype, np.int64)


class ApportionTests(unittest.TestCase):
    def test_parts_sum_to_rounded_total(self):
        parts = [8.21, 64.03, 5.4, 3.7, 2.66]

        counts = apportion_to_total(parts)

        self.assertEqual(counts.dtype, np.int64)
        self.assertEqual(int(counts.sum()), int(np.floor(sum(parts) + 0.5)))
        self.assertTrue((np.abs(counts - np.array(parts)) < 1).all())

    def test_whole_parts_are_unchanged(self):
        self.assertEqual(apportion_to_total([3, 0, 7]).tolist(), [3, 0, 7])

    def test_negative_parts_are_treated_as_zero(self):
        self.assertEqual(apportion_to_total([-2.0, 1.6]).tolist(), [0, 2])


class ReconcileFlowTests(unittest.TestCase):
    def setUp(self):
        rates = {"general": 5.2, "cardiology": 1.3, "oncology": 0.5, "icu": 0.9,
                 "surgery": 1.4, "pediatrics": 1.1, "neurology": 0.8}
        self.expected = pd.DataFrame(
            {name: np.full(30, rate) for name, rate in rates.items()},
            index=pd.date_range("2026-10-06", periods=30),
        )

    def test_total_path_has_no_rounding_noise(self):
        counts = reconcile_flow_counts(self.expected)

        daily_total = counts.sum(axis=1)
        expected_total = self.expected.sum(axis=1).iloc[0]
        self.assertTrue(daily_total.isin([np.floor(expected_total), np.ceil(expected_total)]).all())
        self.assertTrue((daily_total.to_numpy() == to_flow_counts(self.expected.sum(axis=1))).all())

    def test_group_running_totals_track_expected_values(self):
        counts = reconcile_flow_counts(self.expected)

        drift = (counts.cumsum() - self.expected.cumsum()).abs()
        self.assertLessEqual(float(drift.to_numpy().max()), 1.0)
        self.assertTrue((counts.to_numpy() >= 0).all())
        self.assertEqual(counts.to_numpy().dtype, np.int64)

    def test_low_volume_group_is_not_rounded_up_every_day(self):
        counts = reconcile_flow_counts(self.expected)

        self.assertIn(int(counts["oncology"].sum()), (14, 15, 16))


class CountColumnAssertionTests(unittest.TestCase):
    def test_valid_counts_pass(self):
        assert_count_column(pd.DataFrame({"beds": [0, 3, 230]}), "beds", capacity=230)

    def test_decimal_counts_fail(self):
        with self.assertRaisesRegex(ValueError, "non-integer"):
            assert_count_column(pd.DataFrame({"beds": [1.5]}), "beds")

    def test_negative_counts_fail(self):
        with self.assertRaisesRegex(ValueError, "negative"):
            assert_count_column(pd.DataFrame({"beds": [-1]}), "beds")

    def test_counts_over_capacity_fail(self):
        with self.assertRaisesRegex(ValueError, "capacity"):
            assert_count_column(pd.DataFrame({"beds": [231]}), "beds", capacity=230)


if __name__ == "__main__":
    unittest.main()

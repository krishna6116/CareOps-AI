import unittest

import numpy as np
import pandas as pd

from ml.classification_evaluation import (
    classification_metrics,
    expected_calibration_error,
    logistic_regression_baseline,
    prevalence_baseline,
    reliability_table,
)


class ClassificationMetricTests(unittest.TestCase):
    def test_perfectly_separating_scores(self):
        metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], threshold=0.5)

        self.assertEqual(metrics["roc_auc"], 1.0)
        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["f1"], 1.0)

    def test_undefined_metrics_are_nan(self):
        metrics = classification_metrics([0, 0, 0], [0.1, 0.2, 0.3], threshold=0.5)

        self.assertTrue(np.isnan(metrics["roc_auc"]))
        self.assertTrue(np.isnan(metrics["precision"]))
        self.assertTrue(np.isnan(metrics["recall"]))

    def test_calibrated_scores_have_small_ece(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0, 1, 20000)
        y = rng.uniform(0, 1, 20000) < p

        self.assertLess(expected_calibration_error(y, p), 0.02)
        self.assertGreater(expected_calibration_error(y, np.clip(p + 0.3, 0, 1)), 0.2)

    def test_reliability_table_counts_every_row(self):
        table = reliability_table([0, 1, 1, 0, 1], [0.05, 0.95, 1.0, 0.0, 0.55])

        self.assertEqual(int(table["count"].sum()), 5)


class BaselineTests(unittest.TestCase):
    def test_prevalence_baseline(self):
        self.assertTrue(np.allclose(prevalence_baseline([0, 0, 0, 1], 3), 0.25))

    def test_logistic_regression_learns_a_simple_signal(self):
        rng = np.random.default_rng(5)
        X = pd.DataFrame({"risk": rng.normal(size=4000), "noise": rng.normal(size=4000)})
        y = (rng.uniform(size=4000) < 1 / (1 + np.exp(-2 * X["risk"]))).astype(int)

        p = logistic_regression_baseline(X.iloc[:3000], y.iloc[:3000], X.iloc[3000:])

        metrics = classification_metrics(y.iloc[3000:], p, threshold=0.5)
        self.assertGreater(metrics["roc_auc"], 0.8)


if __name__ == "__main__":
    unittest.main()

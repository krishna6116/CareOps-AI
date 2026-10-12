"""Published documents must match the outputs they describe."""

import unittest

from ml import model_metrics_report as report


class GeneratedDocsMatchOutputsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = report.load_outputs()

    def test_model_metrics_document_matches_outputs(self):
        committed = report.REPORT.read_text(encoding="utf-8")

        self.assertEqual(
            committed, report.build_report(self.data),
            "docs/MODEL_METRICS.md is stale; run python ml/model_metrics_report.py",
        )

    def test_readme_results_block_matches_outputs(self):
        readme = report.README.read_text(encoding="utf-8")

        self.assertEqual(
            readme, report.readme_with_block(readme, report.build_readme_block(self.data)),
            "README results block is stale; run python ml/model_metrics_report.py",
        )


if __name__ == "__main__":
    unittest.main()

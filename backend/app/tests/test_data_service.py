from __future__ import annotations

import csv
import tempfile
import unittest
from datetime import date
from pathlib import Path

from backend.app.services.data_service import (
    get_dataset_catalog,
    get_dataset_page,
    get_dataset_validation,
    get_departments,
    get_operations_analytics,
)


class DatasetCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.project_root = Path(self.temporary_directory.name)
        self.dataset_path = self.project_root / "data" / "synthetic" / "admissions.csv"
        self.dataset_path.parent.mkdir(parents=True)
        with self.dataset_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["admission_id", "patient_id", "admission_date", "LOS"])
            writer.writerows([
                ["a-1", "p-1", "2026-01-02", "3"],
                ["a-2", "p-2", "2026-01-03", "4"],
                ["a-3", "p-3", "2026-01-04", "5"],
            ])

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_catalog_reports_schema_exact_rows_types_and_date_range(self) -> None:
        catalog = get_dataset_catalog(project_root=self.project_root)

        self.assertEqual(len(catalog), 1)
        dataset = catalog[0]
        self.assertEqual(dataset["source_file"], "data/synthetic/admissions.csv")
        self.assertEqual(dataset["rows"], 3)
        self.assertEqual(dataset["columns"], ["admission_id", "patient_id", "admission_date", "LOS"])
        self.assertEqual(dataset["data_types"]["LOS"], "int64")
        self.assertEqual(
            dataset["date_ranges"]["admission_date"],
            {"min": "2026-01-02", "max": "2026-01-04"},
        )
        self.assertEqual(dataset["sample_quality"]["sample_rows"], 3)

    def test_dataset_preview_is_typed_and_respects_page_bounds(self) -> None:
        dataset = get_dataset_catalog(project_root=self.project_root)[0]

        first_page = get_dataset_page(
            dataset["id"],
            offset=0,
            limit=2,
            project_root=self.project_root,
        )
        second_page = get_dataset_page(
            dataset["id"],
            offset=2,
            limit=2,
            project_root=self.project_root,
        )

        self.assertEqual(first_page["returned_rows"], 2)
        self.assertEqual(first_page["rows"][0]["LOS"], 3)
        self.assertEqual(second_page["returned_rows"], 1)
        self.assertEqual(second_page["rows"][0]["admission_id"], "a-3")

    def test_catalog_filters_by_area_and_search(self) -> None:
        feature_path = self.project_root / "ml" / "admissions.csv"
        feature_path.parent.mkdir()
        feature_path.write_text("date,predicted_admissions\n2026-01-05,2\n", encoding="utf-8")

        catalog = get_dataset_catalog(
            search="predicted admissions",
            area="ml",
            project_root=self.project_root,
        )

        self.assertEqual([dataset["source_file"] for dataset in catalog], ["ml/admissions.csv"])

    def test_validation_reports_missing_required_columns(self) -> None:
        results = get_dataset_validation(project_root=self.project_root)

        self.assertEqual(results[0]["status"], "pass")
        self.assertEqual(results[0]["missing_required_columns"], [])

    def test_invalid_or_traversal_dataset_ids_are_rejected(self) -> None:
        for dataset_id in ("not-base64", "Li4vLnNlY3JldC5jc3Y"):
            with self.subTest(dataset_id=dataset_id):
                with self.assertRaises(ValueError):
                    get_dataset_page(
                        dataset_id,
                        project_root=self.project_root,
                    )

    def test_operations_analytics_uses_filtered_source_data(self) -> None:
        synthetic = self.project_root / "data" / "synthetic"
        datasets = {
            "daily_operations.csv": (
                "date,department,admissions,discharges,emergency_cases,ICU_admissions,"
                "occupancy,staff_demand,bed_demand\n"
                "2026-01-01,Cardiology,2,1,0,0,4,3,2\n"
                "2026-01-02,Cardiology,3,2,0,0,5,4,3\n"
                "2026-01-01,ICU,1,0,0,1,2,2,1\n"
            ),
            "appointments.csv": (
                "appointment_id,department,date,no_show\n"
                "a-1,Cardiology,2026-01-01,0\n"
                "a-2,Cardiology,2026-01-02,1\n"
                "a-3,ICU,2026-01-01,0\n"
            ),
            "admissions.csv": (
                "admission_id,department,admission_date\n"
                "ad-1,Cardiology,2026-01-01\n"
                "ad-2,Cardiology,2026-01-02\n"
                "ad-3,ICU,2026-01-01\n"
            ),
            "beds.csv": (
                "bed_id,department,status\n"
                "b-1,Cardiology,Occupied\n"
                "b-2,Cardiology,Available\n"
                "b-3,ICU,Occupied\n"
            ),
            "staff.csv": (
                "staff_id,department,role\n"
                "s-1,Cardiology,Nurse\n"
                "s-2,ICU,Nurse\n"
            ),
            "patients.csv": "patient_id\np-1\np-2\np-3\n",
        }
        for filename, content in datasets.items():
            (synthetic / filename).write_text(content, encoding="utf-8")

        result = get_operations_analytics(
            start_date=date(2026, 1, 2),
            end_date=date(2026, 1, 2),
            department="Cardiology",
            project_root=self.project_root,
        )

        self.assertEqual(result["summary"]["admissions"], 1)
        self.assertEqual(result["summary"]["appointments"], 1)
        self.assertEqual(result["summary"]["no_shows"], 1)
        self.assertEqual(result["summary"]["beds"], 2)
        self.assertEqual(result["summary"]["staff"], 1)
        self.assertEqual(result["daily"][0]["date"], "2026-01-02")
        self.assertEqual(result["daily"][0]["appointments"], 1)
        self.assertEqual(result["departments"][0]["department"], "Cardiology")

    def test_department_list_discovers_existing_source_dimensions(self) -> None:
        beds = self.project_root / "data" / "synthetic" / "beds.csv"
        beds.parent.mkdir(parents=True, exist_ok=True)
        beds.write_text(
            "bed_id,department\nb-1,Emergency\nb-2,Cardiology\n",
            encoding="utf-8",
        )

        self.assertEqual(
            get_departments(project_root=self.project_root),
            ["Cardiology", "Emergency"],
        )


if __name__ == "__main__":
    unittest.main()

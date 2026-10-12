import unittest

import pandas as pd

from optimization.bed_assignment import (
    CompatibilityIndex, Rules, plan_day, prepare_beds, summarize,
)


DAY = pd.Timestamp("2026-03-10")


def bed(bed_id, department, bed_type, isolation=False, status="Available", emergency=False):
    return {
        "bed_id": bed_id, "department": department, "bed_type": bed_type,
        "status": status, "isolation_capable": isolation,
        "emergency_reserve": bed_type in {"ICU", "Emergency"},
        "available_for_emergency": emergency,
    }


def admission(admission_id, department, severity="Moderate", icu=0, isolation=False,
              admitted=DAY, discharged=DAY + pd.Timedelta(days=3), bed_id=None):
    return {
        "admission_id": admission_id, "department": department, "severity": severity,
        "admission_type": "Emergency", "ICU_required": icu,
        "isolation_required": isolation, "admission_date": admitted,
        "discharge_date": discharged, "assigned_bed_id": bed_id,
    }


def roster(departments, nurses_per_shift=5):
    rows = []
    for department in departments:
        for shift in ("Morning", "Evening", "Night"):
            rows += [{
                "date": DAY.date().isoformat(), "department": department, "shift": shift,
                "assigned": "Yes", "staff_role": "Nurse",
            }] * nurses_per_shift
    return pd.DataFrame(rows)


BEDS = [
    bed("ICU1", "ICU", "ICU", isolation=True),
    bed("ICU2", "ICU", "ICU", isolation=True),
    bed("ICU3", "ICU", "ICU", isolation=True),
    bed("GM1", "General Medicine", "General", emergency=True),
    bed("GM2", "General Medicine", "General"),
    bed("CA1", "Cardiology", "General"),
    bed("ON1", "Oncology", "Isolation", isolation=True),
    bed("PE1", "Pediatrics", "General"),
    bed("ED1", "Emergency", "Emergency", isolation=True),
    bed("ED2", "Emergency", "Emergency", isolation=True),
    bed("ED3", "Emergency", "Emergency", isolation=True),
    bed("MX1", "General Medicine", "General", status="Maintenance"),
]
WARDS = sorted({row["department"] for row in BEDS})


def run_day(admissions, rules=Rules(), beds=BEDS, shifts=None):
    sources = {
        "admissions": pd.DataFrame(admissions),
        "beds": pd.DataFrame(beds),
        "shifts": roster(WARDS) if shifts is None else shifts,
    }
    prepared = prepare_beds(sources["beds"])
    detail, summary = plan_day(DAY, sources, prepared, rules)
    return detail.set_index("admission_id"), summary, prepared


class CompatibilityTests(unittest.TestCase):
    def test_icu_request_only_gets_icu_beds(self):
        index = CompatibilityIndex(prepare_beds(pd.DataFrame(BEDS)))
        beds = index.beds_for(pd.Series(admission("A", "Surgery", icu=1)))
        self.assertEqual(set(beds), {"ICU1", "ICU2", "ICU3"})

    def test_maintenance_and_pediatric_beds_are_excluded_for_adults(self):
        index = CompatibilityIndex(prepare_beds(pd.DataFrame(BEDS)))
        beds = index.beds_for(pd.Series(admission("A", "General Medicine")))
        self.assertNotIn("MX1", beds)
        self.assertNotIn("PE1", beds)
        self.assertEqual(beds["GM1"], "home")
        self.assertEqual(beds["CA1"], "off_service")

    def test_isolation_request_needs_isolation_room(self):
        index = CompatibilityIndex(prepare_beds(pd.DataFrame(BEDS)))
        beds = index.beds_for(pd.Series(admission("A", "General Medicine", isolation=True)))
        self.assertEqual(beds, {"ON1": "off_service"})


class AssignmentTests(unittest.TestCase):
    def test_census_excludes_occupied_beds_and_frees_discharges(self):
        detail, summary, _ = run_day([
            admission("OLD1", "General Medicine", admitted=DAY - pd.Timedelta(days=2), bed_id="GM1"),
            admission("OLD2", "General Medicine", admitted=DAY - pd.Timedelta(days=2), bed_id="GM2"),
            admission("OLD3", "Cardiology", admitted=DAY - pd.Timedelta(days=4),
                      discharged=DAY, bed_id="CA1"),
            admission("NEW", "General Medicine"),
        ])
        self.assertEqual(summary["occupied_at_start"], 2)
        self.assertEqual(summary["requests"], 1)
        # The home ward is full, so the optimizer pools into the bed
        # Cardiology discharged this morning; the baseline does not pool.
        self.assertEqual(detail.loc["NEW", "optimized_bed_id"], "CA1")
        self.assertEqual(detail.loc["NEW", "optimized_placement"], "off_service")
        self.assertEqual(detail.loc["NEW", "baseline_status"], "UNASSIGNED")
        self.assertEqual(detail.loc["NEW", "baseline_reason"], "all_compatible_beds_occupied")

    def test_isolation_without_room_in_home_ward_is_explained(self):
        detail, _, _ = run_day([admission("ISO", "General Medicine", isolation=True)])
        self.assertEqual(detail.loc["ISO", "baseline_reason"], "no_compatible_bed_in_inventory")
        self.assertEqual(detail.loc["ISO", "optimized_bed_id"], "ON1")

    def test_icu_reserve_is_kept_for_critical_requests(self):
        detail, _, _ = run_day([
            admission("HIGH1", "ICU", severity="High"),
            admission("HIGH2", "ICU", severity="High"),
            admission("CRIT", "ICU", severity="Critical"),
        ])
        assigned = detail[detail["optimized_status"] == "ASSIGNED"].index
        # Three ICU beds, reserve of two: one non-critical plus the critical one.
        self.assertIn("CRIT", assigned)
        self.assertEqual(len(assigned), 2)
        reasons = set(detail.loc[detail["optimized_status"] == "UNASSIGNED", "optimized_reason"])
        self.assertEqual(reasons, {"protected_reserve"})

    def test_scarce_bed_goes_to_higher_severity(self):
        beds = [bed("CA1", "Cardiology", "General")]
        detail, _, _ = run_day([
            admission("A1", "Cardiology", severity="Low"),
            admission("A2", "Cardiology", severity="High"),
        ], beds=beds)
        self.assertEqual(detail.loc["A2", "optimized_bed_id"], "CA1")
        self.assertEqual(detail.loc["A1", "optimized_reason"], "taken_by_other_requests")
        # The baseline serves arrival order instead.
        self.assertEqual(detail.loc["A1", "baseline_bed_id"], "CA1")

    def test_hard_staffing_blocks_unstaffed_wards(self):
        shifts = roster([ward for ward in WARDS if ward != "General Medicine"])
        detail, _, _ = run_day(
            [admission("A", "General Medicine")], rules=Rules(staffing_hard=True), shifts=shifts,
        )
        # General Medicine has no nurses rostered; Cardiology is staffed.
        self.assertEqual(detail.loc["A", "optimized_bed_department"], "Cardiology")
        self.assertEqual(detail.loc["A", "baseline_reason"], "staffing_limit")

    def test_reported_staffing_counts_exceedances(self):
        shifts = roster([ward for ward in WARDS if ward != "Cardiology"])
        beds = [bed("CA1", "Cardiology", "General")]
        _, summary, _ = run_day([admission("A", "Cardiology")], beds=beds, shifts=shifts)
        self.assertEqual(summary["optimized_staffing_exceedances"], 1)

    def test_no_requests_is_a_valid_day(self):
        detail, summary, _ = run_day([
            admission("OLD", "Cardiology", admitted=DAY - pd.Timedelta(days=1), bed_id="CA1"),
        ])
        self.assertTrue(detail.empty)
        self.assertEqual(summary["solver_status"], "OPTIMAL")

    def test_results_are_reproducible_and_violation_free(self):
        admissions = [
            admission(f"R{i}", department, severity=severity)
            for i, (department, severity) in enumerate([
                ("General Medicine", "Low"), ("Cardiology", "High"),
                ("General Medicine", "Critical"), ("Emergency", "Moderate"),
                ("Emergency", "High"), ("Pediatrics", "Low"), ("Surgery", "High"),
            ])
        ]
        first, summary, beds = run_day(admissions)
        second, _, _ = run_day(admissions)
        pd.testing.assert_frame_equal(first, second)
        kpis = summarize(first.reset_index(), beds, pd.DataFrame([summary])).set_index("policy")
        self.assertEqual(int(kpis["compatibility_violations"].sum()), 0)
        self.assertGreaterEqual(kpis.loc["optimized", "assigned"], kpis.loc["baseline", "assigned"])
        # Surgery has no beds in this inventory and no general-type fallback.
        self.assertEqual(first.loc["R6", "optimized_reason"], "no_compatible_bed_in_inventory")


if __name__ == "__main__":
    unittest.main()

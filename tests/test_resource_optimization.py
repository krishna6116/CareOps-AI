import unittest
from datetime import date

import pandas as pd

from optimization.resource_optimization import (
    build_bed_forecast,
    count_appointment_conflicts,
    optimize_appointments,
    optimize_bed_allocations,
    optimize_emergency_capacity,
    optimize_staff_schedule,
    build_resource_gap_analysis,
    build_synthetic_surge_beds,
)


class BedForecastTests(unittest.TestCase):
    def test_synthetic_surge_beds_cover_peak_census_and_protected_reserve(self):
        beds = pd.DataFrame([
            {"bed_id": f"icu{i}", "department": "icu", "bed_type": "icu",
             "status": "Available", "isolation_capable": i == 0}
            for i in range(3)
        ])
        forecast = pd.DataFrame([
            {"date": "2026-09-23", "department": "icu",
             "projected_occupied_beds": 8, "projected_isolation_bed_demand": 2},
            {"date": "2026-09-24", "department": "icu",
             "projected_occupied_beds": 6, "projected_isolation_bed_demand": 1},
        ])
        surge = build_synthetic_surge_beds(forecast, beds)
        self.assertEqual(len(surge), 7)
        self.assertTrue(surge["isolation_capable"].all())
        self.assertTrue(surge["scenario_only"].all())

    def test_forecast_combines_occupancy_admissions_and_los(self):
        inputs = {
            "beds": pd.DataFrame([
                {"bed_id": "b1", "department": "General Medicine", "bed_type": "General", "status": "Occupied"},
                {"bed_id": "b2", "department": "General Medicine", "bed_type": "General", "status": "Occupied"},
                {"bed_id": "b3", "department": "General Medicine", "bed_type": "General", "status": "Available"},
                {"bed_id": "b4", "department": "General Medicine", "bed_type": "General", "status": "Maintenance"},
            ]),
            "los_predictions": pd.DataFrame([
                {"department": "General Medicine", "predicted_los_days": 4.0},
            ]),
            "admission_forecast": pd.DataFrame([
                {
                    "date": "2026-09-23",
                    "department": "General Medicine",
                    "predicted_admissions": 2.0,
                    "predicted_isolation_admissions": 0.2,
                    "historical_smoothed_isolation_rate": 0.1,
                    "isolation_forecast_source": "synthetic_scenario",
                },
            ]),
            "occupancy_forecast": pd.DataFrame([
                {"date": "2026-09-23", "forecast_occupied_beds": 3},
            ]),
        }
        result = build_bed_forecast(inputs).iloc[0]
        self.assertEqual(result["current_occupied"], 2)
        # Expected values 0.5 discharges and 3.5 beds publish as whole counts.
        self.assertEqual(result["expected_discharges"], 1)
        self.assertEqual(result["unconstrained_projected_occupied_beds"], 4)
        self.assertEqual(result["projected_occupied_beds"], 3)
        self.assertEqual(result["projected_isolation_bed_demand"], 1)
        self.assertEqual(result["operational_bed_capacity"], 3)
        self.assertAlmostEqual(result["isolation_scenario_rate"], 0.1)
        self.assertEqual(result["isolation_source_type"], "synthetic_scenario")
        self.assertTrue(result["isolation_demand_source_available"])

    def test_icu_admissions_flow_does_not_replace_icu_census_projection(self):
        inputs = {
            "beds": pd.DataFrame([
                {"bed_id": f"icu{i}", "department": "ICU",
                 "bed_type": "ICU", "status": "Occupied" if i < 2 else "Available"}
                for i in range(5)
            ]),
            "los_predictions": pd.DataFrame([
                {"department": "ICU", "predicted_los_days": 4.0},
            ]),
            "admission_forecast": pd.DataFrame([
                {"date": "2026-09-23", "department": "ICU",
                 "predicted_admissions": 2.0},
            ]),
            "icu_forecast": pd.DataFrame([
                {"date": "2026-09-23", "predicted_icu_admissions": 30.0,
                 "forecast_target_unit": "admissions_per_day"},
            ]),
        }
        result = build_bed_forecast(inputs).iloc[0]
        self.assertEqual(result["predicted_icu_admissions_per_day"], 30.0)
        self.assertEqual(result["projected_occupied_beds"], 4)

    def test_published_bed_forecast_uses_whole_non_negative_counts(self):
        inputs = {
            "beds": pd.DataFrame([
                {"bed_id": f"{department[:2]}{i}", "department": department,
                 "bed_type": "General", "status": "Occupied" if i < occupied else "Available"}
                for department, occupied in (("General Medicine", 7), ("Surgery", 3))
                for i in range(10)
            ]),
            "los_predictions": pd.DataFrame([
                {"department": "General Medicine", "predicted_los_days": 3.0},
                {"department": "Surgery", "predicted_los_days": 7.0},
            ]),
            "admission_forecast": pd.DataFrame([
                {"date": f"2026-09-{day}", "department": department, "predicted_admissions": admissions}
                for day in (23, 24, 25)
                for department, admissions in (("General Medicine", 2), ("Surgery", 1))
            ]),
            "occupancy_forecast": pd.DataFrame([
                {"date": f"2026-09-{day}", "forecast_occupied_beds": 11} for day in (23, 24, 25)
            ]),
        }

        forecast = build_bed_forecast(inputs)

        count_columns = [
            "current_occupied", "predicted_admissions", "expected_discharges",
            "unconstrained_projected_occupied_beds", "projected_occupied_beds",
            "forecast_available_beds", "projected_isolation_bed_demand",
        ]
        for column in count_columns:
            values = pd.to_numeric(forecast[column])
            self.assertTrue((values % 1 == 0).all(), column)
            self.assertTrue((values >= 0).all(), column)
        self.assertTrue((forecast.groupby("date")["projected_occupied_beds"].sum() == 11).all())
        self.assertTrue(
            (forecast["forecast_available_beds"]
             == (forecast["operational_bed_capacity"] - forecast["projected_occupied_beds"]).clip(lower=0)).all()
        )

    def test_bed_allocator_respects_staffing_capacity(self):
        forecast = pd.DataFrame([
            {"date": "2026-09-23", "department": "general medicine",
             "projected_occupied_beds": 8, "predicted_admissions": 2},
        ])
        inputs = {
            "beds": pd.DataFrame([
                {"bed_id": f"g{i}", "department": "General Medicine",
                 "bed_type": "General", "status": "Available"}
                for i in range(10)
            ]),
        }
        coverage = pd.DataFrame([
            {"department": "general medicine", "assigned_staff": 1},
        ])
        detail, kpis = optimize_bed_allocations(forecast, inputs, coverage)
        self.assertEqual(detail.iloc[0]["optimized_allocated"], 4)
        self.assertEqual(detail.iloc[0]["optimized_overflow"], 4)
        self.assertEqual(kpis.loc[kpis["scenario"] == "optimized",
                                  "staffing_constraint_violations"].iloc[0], 0)
        optimized_kpis = kpis[kpis["scenario"] == "optimized"].iloc[0]
        self.assertEqual(optimized_kpis["mean_forecast_occupancy_utilization"], 0.8)
        self.assertEqual(optimized_kpis["allocated_physical_capacity_utilization"], 0.4)
        self.assertEqual(optimized_kpis["staffed_capacity_utilization"], 1.0)
        self.assertEqual(optimized_kpis["overflow_bed_days_due_to_staffing"], 4)
        self.assertEqual(optimized_kpis["overflow_bed_days_due_to_bed_constraints"], 0)
        self.assertEqual(
            optimized_kpis["overflow_bed_days_per_forecast_admission"], 2.0
        )

    def test_resource_gap_analysis_reports_nurse_shortfall_per_shift(self):
        forecast = pd.DataFrame([{
            "date": "2026-09-23", "department": "general medicine",
            "projected_occupied_beds": 9.2, "operational_bed_capacity": 10,
        }])
        allocation = pd.DataFrame([{
            "date": "2026-09-23", "department": "general medicine",
            "projected_occupied_beds": 9, "optimized_allocated": 4,
            "optimized_overflow": 5, "optimized_overflow_due_to_staffing": 5,
            "optimized_overflow_due_to_bed_constraints": 0,
        }])
        coverage = pd.DataFrame([
            {"date": "2026-09-23", "department": "general medicine",
             "shift": shift, "role": "Nurse", "assigned_staff": 1}
            for shift in ("Morning", "Evening", "Night")
        ])
        result = build_resource_gap_analysis(forecast, allocation, coverage).iloc[0]
        self.assertEqual(result["minimum_nurses_per_shift"], 1)
        self.assertEqual(result["nurses_required_per_shift"], 3)
        self.assertEqual(result["incremental_nurses_required_per_shift"], 2)
        self.assertEqual(result["allocation_demand_beds"], 9)

    def test_optimizer_uses_shared_emergency_flexible_capacity(self):
        beds = pd.DataFrame(
            [
                {"bed_id": f"e{i}", "department": "Emergency",
                 "bed_type": "Emergency", "status": "Available",
                 "isolation_capable": False, "available_for_emergency": True}
                for i in range(3)
            ] + [
                {"bed_id": f"g{i}", "department": "General Medicine",
                 "bed_type": "General", "status": "Available",
                 "isolation_capable": False, "available_for_emergency": i < 5}
                for i in range(10)
            ]
        )
        forecast = pd.DataFrame([
            {"date": "2026-09-23", "department": "emergency",
             "projected_occupied_beds": 5, "predicted_admissions": 2},
            {"date": "2026-09-23", "department": "general medicine",
             "projected_occupied_beds": 8, "predicted_admissions": 1},
        ])
        coverage = pd.DataFrame([
            {"department": "emergency", "assigned_staff": 1},
            {"department": "general medicine", "assigned_staff": 2},
        ])
        detail, kpis = optimize_bed_allocations(
            forecast, {"beds": beds}, coverage
        )
        emergency = detail[detail["department"] == "emergency"].iloc[0]
        self.assertGreater(emergency["optimized_allocated"],
                           emergency["baseline_allocated"])
        self.assertGreater(emergency["optimized_emergency_flex_beds"], 0)
        self.assertGreater(
            kpis.loc[kpis["scenario"] == "optimized", "allocated_bed_days"].iloc[0],
            kpis.loc[kpis["scenario"] == "baseline", "allocated_bed_days"].iloc[0],
        )

    def test_isolation_reserve_and_explicit_need_are_enforced(self):
        beds = pd.DataFrame([
            {"bed_id": f"o{i}", "department": "Oncology",
             "bed_type": "Isolation", "status": "Available",
             "isolation_capable": True, "available_for_emergency": False}
            for i in range(3)
        ])
        coverage = pd.DataFrame([{"department": "oncology", "assigned_staff": 1}])
        demand_without_flag = pd.DataFrame([
            {"date": "2026-09-23", "department": "oncology",
             "projected_occupied_beds": 3, "predicted_admissions": 1},
        ])
        detail, _ = optimize_bed_allocations(
            demand_without_flag, {"beds": beds}, coverage
        )
        row = detail.iloc[0]
        self.assertFalse(row["isolation_demand_source_available"])
        self.assertEqual(row["optimized_allocated"], 2)

        demand_with_flag = demand_without_flag.assign(
            predicted_isolation_admissions=1,
            isolation_demand_source_available=True,
        )
        detail, _ = optimize_bed_allocations(
            demand_with_flag, {"beds": beds}, coverage
        )
        row = detail.iloc[0]
        self.assertEqual(row["projected_isolation_bed_demand"], 1)
        self.assertEqual(row["optimized_unmet_isolation_demand"], 0)


class StaffingAndForecastTests(unittest.TestCase):
    def test_staffed_capacity_uses_minimum_nurse_shift(self):
        from optimization.resource_optimization import staffed_bed_capacity

        coverage = pd.DataFrame([
            {"date": "2026-09-23", "department": "emergency",
             "shift": "Morning", "role": "Nurse", "assigned_staff": 3},
            {"date": "2026-09-23", "department": "emergency",
             "shift": "Evening", "role": "Nurse", "assigned_staff": 2},
            {"date": "2026-09-23", "department": "emergency",
             "shift": "Night", "role": "Nurse", "assigned_staff": 1},
        ])
        self.assertEqual(
            staffed_bed_capacity(coverage, "emergency", "2026-09-23"), 4
        )

    def test_forecast_reports_without_clipping_bed_capacity(self):
        data = {
            "beds": pd.DataFrame([
                {"department": "General Medicine", "status": "Occupied"},
            ]),
            "los_predictions": pd.DataFrame([
                {"department": "General Medicine", "predicted_los_days": 5},
            ]),
            "admission_forecast": pd.DataFrame([
                {"date": "2026-09-23", "department": "General Medicine",
                 "predicted_admissions": 5},
            ]),
        }
        row = build_bed_forecast(data).iloc[0]
        self.assertGreater(row["unconstrained_projected_occupied_beds"], 1)
        self.assertGreater(row["projected_occupied_beds"], row["operational_bed_capacity"])


class StaffSchedulingTests(unittest.TestCase):
    def test_synthetic_surge_staff_fills_census_based_nurse_requirements(self):
        history_dates = pd.date_range("2026-06-25", "2026-09-22", freq="D")
        staff = pd.DataFrame([{
            "staff_id": "n1", "staff_name": "Base Nurse", "role": "Nurse",
            "department": "Cardiology", "specialty": "Cardiology",
            "availability": "Available", "shift_preference": "Night",
            "max_hours": 40,
        }])
        shifts = pd.DataFrame([{
            "date": day.strftime("%Y-%m-%d"), "staff_id": "n1",
            "shift": "Night", "department": "Cardiology", "staff_role": "Nurse",
            "hours": 8, "assigned": "Yes",
        } for day in history_dates])
        operations = pd.DataFrame([{
            "date": day.strftime("%Y-%m-%d"), "department": "Cardiology",
            "staff_demand": 8, "admissions": 1,
        } for day in history_dates])
        inputs = {
            "staff": staff,
            "shifts": shifts,
            "daily_operations": operations,
            "admission_forecast": pd.DataFrame([{
                "date": "2026-09-23", "department": "Cardiology",
                "predicted_admissions": 1,
            }]),
        }
        schedule, coverage = optimize_staff_schedule(
            inputs,
            date(2026, 9, 23),
            [date(2026, 9, 23)],
            pd.DataFrame([{
                "date": "2026-09-23", "department": "cardiology",
                "projected_occupied_beds": 8,
            }]),
        )
        self.assertEqual(int(coverage["unfilled_staff"].sum()), 0)
        self.assertEqual(int(coverage["required_staff"].sum()), 6)
        self.assertGreater(
            int((schedule["staff_source"] == "synthetic_surge_pool").sum()), 0
        )

    def test_schedule_observes_rest_after_previous_night_shift(self):
        staff = pd.DataFrame([
            {"staff_id": "n1", "staff_name": "Night Nurse", "role": "Nurse",
             "department": "Cardiology", "specialty": "Cardiology",
             "availability": "Available", "shift_preference": "Morning", "max_hours": 40},
            {"staff_id": "n2", "staff_name": "Flexible Nurse", "role": "Nurse",
             "department": "Cardiology", "specialty": "Cardiology",
             "availability": "Available", "shift_preference": "Any", "max_hours": 40},
            {"staff_id": "n3", "staff_name": "Evening Nurse", "role": "Nurse",
             "department": "Cardiology", "specialty": "Cardiology",
             "availability": "Available", "shift_preference": "Evening", "max_hours": 40},
        ])
        dates = pd.date_range("2026-06-28", "2026-09-22", freq="3D")
        shifts = pd.DataFrame([
            {"date": shift_date.strftime("%Y-%m-%d"), "staff_id": "n2",
             "shift": "Night", "department": "Cardiology", "staff_role": "Nurse",
             "hours": 8, "assigned": "Yes"}
            for shift_date in dates
        ] + [
            {"date": "2026-09-22", "staff_id": "n1", "shift": "Night",
             "department": "Cardiology", "staff_role": "Nurse", "hours": 8,
             "assigned": "Yes"},
        ])
        operations = pd.DataFrame([
            {"date": "2026-09-22", "department": "Cardiology", "staff_demand": 24,
             "admissions": 2},
        ])
        schedule, coverage = optimize_staff_schedule(
            {"staff": staff, "shifts": shifts, "daily_operations": operations},
            date(2026, 9, 23),
        )
        night_nurse = schedule[schedule["staff_id"] == "n1"]
        self.assertFalse((night_nurse["shift"] == "Morning").any())
        self.assertTrue((schedule["role"] == "Nurse").all())
        self.assertTrue((coverage["assigned_staff"] <= coverage["required_staff"]).all())
        self.assertEqual(coverage["required_staff"].sum(), 1)

    def test_schedule_respects_rolling_weekly_hour_limit(self):
        staff = pd.DataFrame([
            {"staff_id": "n1", "staff_name": "Nurse", "role": "Nurse",
             "department": "Cardiology", "specialty": "Cardiology",
             "availability": "Available", "shift_preference": "Night", "max_hours": 16},
        ])
        historical_nights = pd.date_range("2026-06-28", "2026-09-22", freq="3D")
        shifts = pd.DataFrame([
            {"date": day, "staff_id": "n1", "shift": "Night",
             "department": "Cardiology", "staff_role": "Nurse",
             "hours": 8, "assigned": "Yes"}
            for day in historical_nights.strftime("%Y-%m-%d")
        ])
        operations = pd.DataFrame([
            {"date": "2026-09-22", "department": "Cardiology", "staff_demand": 1,
             "admissions": 1},
        ])
        schedule, coverage = optimize_staff_schedule(
            {"staff": staff, "shifts": shifts, "daily_operations": operations},
            date(2026, 9, 23),
        )
        self.assertTrue(schedule.empty)
        self.assertEqual(int(coverage["unfilled_staff"].sum()), 1)


class EmergencyCapacityTests(unittest.TestCase):
    def test_emergency_capacity_uses_shared_safe_bed_allocation(self):
        beds = pd.DataFrame([
            {"bed_type": "ICU", "status": "Available"},
            {"bed_type": "ICU", "status": "Available"},
            {"bed_type": "ICU", "status": "Available"},
            {"bed_type": "ICU", "status": "Available"},
            {"bed_type": "ICU", "status": "Available"},
            {"bed_type": "ICU", "status": "Available"},
            {"bed_type": "ICU", "status": "Available"},
            {"bed_type": "ICU", "status": "Available"},
            {"bed_type": "ICU", "status": "Available"},
            {"bed_type": "ICU", "status": "Available"},
            {"bed_type": "Emergency", "status": "Available"},
            {"bed_type": "Emergency", "status": "Available"},
            {"bed_type": "Emergency", "status": "Available"},
            {"bed_type": "Emergency", "status": "Available"},
            {"bed_type": "Emergency", "status": "Available"},
            *[
                {"bed_type": "General", "status": "Available"}
                for _ in range(10)
            ],
        ])
        beds["department"] = (
            ["ICU"] * 10 + ["Emergency"] * 5 + ["General Medicine"] * 10
        )
        forecast = pd.DataFrame([
            {"date": "2026-09-23", "department": "icu",
             "projected_occupied_beds": 8, "operational_bed_capacity": 10},
            {"date": "2026-09-23", "department": "emergency",
             "projected_occupied_beds": 12, "operational_bed_capacity": 5},
            {"date": "2026-09-23", "department": "general medicine",
             "projected_occupied_beds": 5, "operational_bed_capacity": 10},
        ])
        staff = pd.DataFrame([
            {"department": "Emergency", "role": "Nurse", "availability": "Available"},
            {"department": "ICU", "role": "Nurse", "availability": "Available"},
        ])
        coverage = pd.DataFrame([
            {"department": "emergency", "assigned_staff": 2},
            {"department": "icu", "assigned_staff": 2},
        ])
        allocation = pd.DataFrame([
            {"date": "2026-09-23", "department": "icu",
             "projected_occupied_beds": 8, "baseline_allocated": 6,
             "optimized_allocated": 6, "optimized_emergency_flex_beds": 0},
            {"date": "2026-09-23", "department": "emergency",
             "projected_occupied_beds": 12, "baseline_allocated": 3,
             "optimized_allocated": 6, "optimized_emergency_flex_beds": 3},
            {"date": "2026-09-23", "department": "general medicine",
             "projected_occupied_beds": 5, "baseline_allocated": 4,
             "optimized_allocated": 3, "optimized_emergency_flex_beds": 0},
        ])
        result = optimize_emergency_capacity(
            forecast,
            {"beds": beds, "icu_forecast": pd.DataFrame([
                {"date": "2026-09-23", "predicted_icu_demand": 8},
            ]), "staff": staff},
            coverage,
            allocation,
        ).iloc[0]
        self.assertEqual(result["emergency_allocated"], 3)
        self.assertEqual(result["projected_icu_census_beds"], 8)
        self.assertEqual(result["emergency_general_overflow_allocated"], 3)
        self.assertEqual(result["emergency_total_allocated"], 6)
        self.assertLessEqual(
            result["emergency_general_overflow_allocated"],
            result["emergency_flexible_beds_used"],
        )


class AppointmentOptimizationTests(unittest.TestCase):
    def test_no_show_fallback_uses_only_prior_appointments(self):
        appointments = pd.DataFrame([
            {"appointment_id": "past", "patient_id": "p0", "doctor_id": "d1",
             "department": "Cardiology", "date": "2026-09-21", "time": "09:00",
             "duration": 30, "priority": "Routine", "no_show": 0},
            {"appointment_id": "today", "patient_id": "p1", "doctor_id": "d1",
             "department": "Cardiology", "date": "2026-09-22", "time": "09:00",
             "duration": 30, "priority": "Routine", "no_show": 1},
        ])
        staff = pd.DataFrame([
            {"staff_id": "d1", "staff_name": "Doctor", "role": "Doctor",
             "department": "Cardiology", "specialty": "Cardiology",
             "availability": "Available"},
        ])
        result, _ = optimize_appointments({
            "appointments": appointments,
            "no_show_predictions": pd.DataFrame(),
            "staff": staff,
        })
        self.assertEqual(result["appointment_id"].tolist(), ["today"])
        self.assertEqual(result.iloc[0]["no_show_risk"], 0)

    def test_appointment_solver_removes_same_doctor_overlaps(self):
        appointments = pd.DataFrame([
            {"appointment_id": f"a{i}", "patient_id": f"p{i}", "doctor_id": "d1",
             "department": "Cardiology", "date": "2026-09-22", "time": "09:00",
             "duration": 30, "priority": "Routine", "no_show": 0}
            for i in range(3)
        ])
        result, metrics = optimize_appointments({
            "appointments": appointments,
            "no_show_predictions": pd.DataFrame(),
            "staff": pd.DataFrame([
                {"staff_id": "d1", "staff_name": "Doctor", "role": "Doctor",
                 "department": "Cardiology", "specialty": "Cardiology",
                 "availability": "Available"},
            ]),
        })
        self.assertEqual(len(result), 3)
        self.assertEqual(metrics.iloc[0]["optimized_doctor_time_conflicts"], 0)
        self.assertEqual(int((result["status"] == "SCHEDULED").sum()), 3)
        self.assertEqual(
            count_appointment_conflicts(
                result, "optimized_time", "optimized_doctor_id"
            ),
            0,
        )

    def test_unavailable_doctor_is_replaced_by_available_specialist(self):
        appointments = pd.DataFrame([
            {"appointment_id": "a1", "patient_id": "p1", "doctor_id": "d1",
             "department": "Cardiology", "date": "2026-09-22", "time": "09:00",
             "duration": 30, "priority": "Urgent", "no_show": 0}
        ])
        staff = pd.DataFrame([
            {"staff_id": "d1", "staff_name": "Unavailable", "role": "Doctor",
             "department": "Cardiology", "specialty": "Cardiology", "availability": "Leave"},
            {"staff_id": "d2", "staff_name": "Available", "role": "Doctor",
             "department": "Cardiology", "specialty": "Cardiology", "availability": "Available"},
        ])
        result, metrics = optimize_appointments({
            "appointments": appointments,
            "no_show_predictions": pd.DataFrame(),
            "staff": staff,
        })
        self.assertEqual(result.iloc[0]["optimized_doctor_id"], "d2")
        self.assertEqual(metrics.iloc[0]["optimized_unavailable_doctor_bookings"], 0)

    def test_only_latest_day_is_optimized_and_doctor_must_be_on_shift(self):
        appointments = pd.DataFrame([
            {"appointment_id": "old", "patient_id": "p0", "doctor_id": "d1",
             "department": "Cardiology", "date": "2026-09-21", "time": "09:00",
             "duration": 30, "priority": "Routine", "no_show": 0},
            {"appointment_id": "today", "patient_id": "p1", "doctor_id": "d1",
             "department": "Cardiology", "date": "2026-09-22", "time": "09:00",
             "duration": 30, "priority": "Routine", "no_show": 0},
        ])
        staff = pd.DataFrame([
            {"staff_id": "d1", "staff_name": "Doctor", "role": "Doctor",
             "department": "Cardiology", "specialty": "Cardiology",
             "availability": "Available"},
        ])
        shifts = pd.DataFrame([
            {"date": "2026-09-22", "staff_id": "d1", "shift": "Evening",
             "hours": 8, "assigned": "Yes"},
        ])
        result, metrics = optimize_appointments({
            "appointments": appointments,
            "no_show_predictions": pd.DataFrame(),
            "staff": staff,
            "shifts": shifts,
        })
        self.assertEqual(result["appointment_id"].tolist(), ["today"])
        self.assertEqual(result.iloc[0]["optimized_time"], "15:00")
        self.assertEqual(metrics.iloc[0]["optimized_off_shift_doctor_bookings"], 0)

    def test_off_shift_original_provider_falls_back_to_available_specialist(self):
        appointments = pd.DataFrame([{
            "appointment_id": "a1", "patient_id": "p1", "doctor_id": "d1",
            "department": "Cardiology", "date": "2026-09-22", "time": "09:00",
            "duration": 30, "priority": "Routine", "no_show": 0,
        }])
        staff = pd.DataFrame([
            {"staff_id": "d1", "staff_name": "Off shift", "role": "Doctor",
             "department": "Cardiology", "specialty": "Cardiology",
             "availability": "Available"},
            {"staff_id": "d2", "staff_name": "On shift", "role": "Doctor",
             "department": "Cardiology", "specialty": "Cardiology",
             "availability": "Available"},
        ])
        shifts = pd.DataFrame([{
            "date": "2026-09-22", "staff_id": "d2", "shift": "Morning",
            "hours": 8, "assigned": "Yes",
        }])

        result, metrics = optimize_appointments({
            "appointments": appointments,
            "no_show_predictions": pd.DataFrame(),
            "staff": staff,
            "shifts": shifts,
        })

        self.assertEqual(result.iloc[0]["optimized_doctor_id"], "d2")
        self.assertEqual(result.iloc[0]["status"], "SCHEDULED")
        self.assertEqual(metrics.iloc[0]["optimized_off_shift_doctor_bookings"], 0)

    def test_on_shift_specialist_handles_overlapping_original_bookings(self):
        appointments = pd.DataFrame([
            {"appointment_id": f"a{i}", "patient_id": f"p{i}", "doctor_id": "d1",
             "department": "Cardiology", "date": "2026-09-22", "time": "09:00",
             "duration": 60, "priority": "Routine", "no_show": 0}
            for i in range(15)
        ])
        staff = pd.DataFrame([
            {"staff_id": doctor_id, "staff_name": doctor_id, "role": "Doctor",
             "department": "Cardiology", "specialty": "Cardiology",
             "availability": "Available"}
            for doctor_id in ("d1", "d2", "d3")
        ])
        shifts = pd.DataFrame([
            {"date": "2026-09-22", "staff_id": doctor_id, "shift": "Morning",
             "hours": 8, "assigned": "Yes"}
            for doctor_id in ("d1", "d2", "d3")
        ])

        result, _ = optimize_appointments({
            "appointments": appointments,
            "no_show_predictions": pd.DataFrame(),
            "staff": staff,
            "shifts": shifts,
        })

        self.assertEqual(int(result["status"].eq("SCHEDULED").sum()), 15)
        self.assertEqual(result["optimized_doctor_id"].nunique(), 3)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.services.pipeline_service import (
    get_admissions,
    get_bed_availability,
    get_beds,
    get_patients,
)

router = APIRouter(prefix="", tags=["resources"])


@router.get("/patients")
def list_patients() -> list[dict]:
    return get_patients()


@router.get("/patients/{patient_id}")
def get_patient(patient_id: str) -> dict:
    patients = get_patients()
    for patient in patients:
        if str(patient.get("patient_id")) == patient_id:
            return patient
    raise HTTPException(status_code=404, detail="Patient not found.")


@router.get("/beds")
def list_beds() -> list[dict]:
    return get_beds()


@router.get("/beds/availability")
def list_bed_availability() -> list[dict]:
    return get_bed_availability()


@router.get("/admissions")
def list_admissions() -> list[dict]:
    return get_admissions()

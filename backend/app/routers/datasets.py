from __future__ import annotations

from datetime import date
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.api import (
    DatasetCatalogResponse,
    DatasetPageResponse,
    DatasetValidationResponse,
    DepartmentsResponse,
    OperationsAnalyticsResponse,
)
from backend.app.services.data_service import (
    get_dataset_catalog,
    get_dataset_page,
    get_dataset_validation,
    get_departments,
    get_operations_analytics,
)

router = APIRouter(prefix="", tags=["datasets and validation"])


@router.get("/datasets", response_model=DatasetCatalogResponse)
def list_datasets(
    search: str | None = Query(default=None, max_length=128),
    area: str | None = Query(default=None, pattern="^(data|ml|optimization)$"),
) -> dict[str, Any]:
    datasets = get_dataset_catalog(search=search, area=area)
    return {"dataset_count": len(datasets), "datasets": datasets}


@router.get("/datasets/{dataset_id}", response_model=DatasetPageResponse)
def read_dataset(
    dataset_id: str,
    offset: int = Query(default=0, ge=0, le=1_000_000),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, Any]:
    try:
        return get_dataset_page(dataset_id, offset=offset, limit=limit)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Dataset not found.") from exc


@router.get("/validation/datasets", response_model=DatasetValidationResponse)
def validate_datasets() -> dict[str, Any]:
    datasets = get_dataset_validation()
    return {"dataset_count": len(datasets), "datasets": datasets}


@router.get("/departments", response_model=DepartmentsResponse)
def list_departments() -> dict[str, Any]:
    return {"departments": get_departments()}


@router.get("/operations", response_model=OperationsAnalyticsResponse)
def operations_analytics(
    start_date: date | None = None,
    end_date: date | None = None,
    department: str | None = Query(default=None, max_length=128),
) -> dict[str, Any]:
    if start_date and end_date and start_date > end_date:
        raise HTTPException(
            status_code=422,
            detail="start_date must be on or before end_date.",
        )
    return get_operations_analytics(
        start_date=start_date,
        end_date=end_date,
        department=department,
    )

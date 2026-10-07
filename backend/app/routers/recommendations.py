from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas.api import RecommendationRequest
from backend.app.services.dashboard_service import get_recommendations

router = APIRouter(prefix="", tags=["recommendations"])


@router.get("/recommendations")
def list_recommendations() -> list[dict]:
    return get_recommendations()


@router.post("/recommendations/generate")
def generate_recommendations(payload: RecommendationRequest) -> dict:
    items = get_recommendations()
    if payload.department:
        items = [
            item for item in items
            if str(item.get("department", "")).casefold() == payload.department.casefold()
        ]
    if not items:
        raise HTTPException(
            status_code=503,
            detail="No matching resource-gap output is available for a recommendation.",
        )
    return {
        "status": "generated_from_saved_resource_gaps",
        "recommendation": items[0],
    }

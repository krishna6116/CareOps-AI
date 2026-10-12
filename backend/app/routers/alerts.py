from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.app.schemas.api import AlertCreate
from backend.app.security import require_api_key
from backend.app.services.dashboard_service import create_alert as persist_alert, get_alerts

router = APIRouter(prefix="", tags=["alerts"])


@router.get("/alerts")
def list_alerts() -> list[dict]:
    return get_alerts()


@router.post("/alerts", dependencies=[Depends(require_api_key)])
def create_alert(payload: AlertCreate) -> dict:
    return {
        "status": "created",
        "alert": persist_alert({
            "severity": payload.severity,
            "category": payload.category,
            "message": payload.message,
            "source": payload.source,
        }),
    }

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import settings
from backend.app.database import init_db
from backend.app.routers.alerts import router as alerts_router
from backend.app.routers.audit import router as audit_router
from backend.app.routers.datasets import router as datasets_router
from backend.app.routers.forecasts import router as forecasts_router
from backend.app.routers.health import router as health_router
from backend.app.routers.insights import router as insights_router
from backend.app.routers.optimization import router as optimization_router
from backend.app.routers.recommendations import router as recommendations_router
from backend.app.routers.resources import router as resources_router
from backend.app.routers.simulations import router as simulations_router


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    init_db()
    yield


app = FastAPI(
    lifespan=lifespan,
    title=settings.project_name,
    version=settings.app_version,
    description=(
        "FastAPI deployment layer for the synthetic healthcare resource planning and "
        "optimization pipeline."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

for router in (
    health_router,
    resources_router,
    datasets_router,
    forecasts_router,
    optimization_router,
    recommendations_router,
    alerts_router,
    simulations_router,
    audit_router,
    insights_router,
):
    app.include_router(router)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"message": "CareOps-AI API is running."}

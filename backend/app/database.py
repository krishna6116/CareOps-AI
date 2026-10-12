from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.app.config import settings


class Base(DeclarativeBase):
    pass


database_url = settings.database_url
_is_sqlite = database_url.startswith("sqlite")
engine = create_engine(
    database_url,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    # PostgreSQL: a small pool, with dead connections replaced after a
    # database restart instead of failing the next request.
    **({} if _is_sqlite else {
        "pool_size": settings.database_pool_size,
        "max_overflow": settings.database_max_overflow,
        "pool_pre_ping": True,
        "pool_recycle": 1800,
    }),
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)


def init_db() -> None:
    from backend.app.models.careops import Base as CareOpsBase

    CareOpsBase.metadata.create_all(bind=engine)


def check_database_health() -> dict[str, object]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return {"status": "ok", "database": engine.dialect.name}
    except Exception:  # pragma: no cover - depends on DB runtime
        return {
            "status": "error",
            "database": engine.dialect.name,
            "error": "Database connection failed.",
        }

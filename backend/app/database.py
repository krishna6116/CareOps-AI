from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from backend.app.config import settings


class Base(DeclarativeBase):
    pass


database_url = settings.database_url
engine = create_engine(
    database_url,
    connect_args={"check_same_thread": False} if database_url.startswith("sqlite") else {},
    future=True,
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

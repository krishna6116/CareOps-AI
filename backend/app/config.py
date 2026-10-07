from __future__ import annotations

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    project_name: str = "CareOps-AI"
    environment: str = "development"
    app_version: str = "0.1.0"
    database_url: str = "sqlite:///./careops.db"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str = "http://127.0.0.1:11434"
    openai_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("OPENAI_API_KEY", "CAREOPS_OPENAI_API_KEY"),
    )
    debug: bool = False
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CAREOPS_",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

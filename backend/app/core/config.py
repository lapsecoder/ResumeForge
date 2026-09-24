"""Application configuration, loaded from environment variables.

All settings derive from environment variables via pydantic-settings. No
secrets or credentials are hardcoded; a local `.env` file is supported for
development convenience.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "ResumeForge API"

    # Allowed browser origins for CORS (comma-separated). Intended for the
    # Next.js frontend during development; tighten for production.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"

    # PostgreSQL + pgvector connection string (async driver).
    database_url: str = (
        "postgresql+asyncpg://resumeforge:resumeforge@localhost:5432/resumeforge"
    )

    # --- Connection pool settings (development-friendly defaults) ---
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_pre_ping: bool = True
    db_echo: bool = False


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


settings = get_settings()

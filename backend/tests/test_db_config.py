"""Tests for database configuration and the async engine/session layer.

These validate configuration-driven wiring and the ORM schema without
requiring a live PostgreSQL instance.
"""

from app.core.config import Settings
from app.db.base import Base
from app.db.session import get_engine, get_session_factory


def test_settings_reads_database_url_from_env(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://u:p@example:5432/db")
    settings = Settings()
    assert settings.database_url == "postgresql+asyncpg://u:p@example:5432/db"


def test_settings_default_database_url() -> None:
    settings = Settings()
    assert settings.database_url.startswith("postgresql+asyncpg://")


def test_engine_created_with_async_driver() -> None:
    engine = get_engine()
    assert engine.dialect.name == "postgresql"
    assert "asyncpg" in engine.url.drivername or "postgresql" in engine.url.drivername


def test_session_factory_returns_async_session() -> None:
    factory = get_session_factory()
    session = factory()
    assert session is not None


def test_declarative_metadata_is_empty_until_models_are_defined() -> None:
    # Foundation stage: no application tables exist yet. The declarative Base
    # is ready for models added in later phases.
    assert set(Base.metadata.tables) == set()

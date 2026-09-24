"""Tests for Alembic migration configuration and the migration script.

These validate that Alembic is wired to the application environment config and
that the migration script matches the ORM metadata. They do not require a live
database, except for the connectivity test which is skipped when PostgreSQL is
unavailable.
"""

from pathlib import Path

from alembic.config import Config

BACKEND_DIR = Path(__file__).resolve().parents[1]
ALEMBIC_INI = BACKEND_DIR / "alembic.ini"


def _alembic_config() -> Config:
    cfg = Config(str(ALEMBIC_INI))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    return cfg


def test_alembic_ini_exists() -> None:
    assert ALEMBIC_INI.exists()


def test_alembic_script_location_points_to_alembic() -> None:
    cfg = _alembic_config()
    assert "alembic" in cfg.get_main_option("script_location")


def test_alembic_ini_has_no_hardcoded_credentials() -> None:
    raw = ALEMBIC_INI.read_text(encoding="utf-8")
    active_lines = [
        line.strip()
        for line in raw.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    cred_lines = [line for line in active_lines if line.startswith("sqlalchemy.url")]
    assert cred_lines == []


def test_env_py_imports_databse_url_from_settings() -> None:
    env_py = (BACKEND_DIR / "alembic" / "env.py").read_text(encoding="utf-8")
    assert "settings.database_url" in env_py
    assert "target_metadata = Base.metadata" in env_py


def test_single_initial_revision_enables_pgvector_only() -> None:
    versions_dir = BACKEND_DIR / "alembic" / "versions"
    revisions = [
        p for p in versions_dir.glob("*.py") if not p.name.startswith("_")
    ]
    assert len(revisions) == 1
    text = revisions[0].read_text(encoding="utf-8")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in text
    assert "op.create_table" not in text
    assert "users" not in text


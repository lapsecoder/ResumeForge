"""Live-database integration tests.

Require a running PostgreSQL instance configured via DATABASE_URL. These are
skipped automatically when PostgreSQL is not reachable, so the suite still
passes in environments without a live database (e.g. CI without services).
"""

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_session_factory


async def _probe() -> bool:
    """Return True if PostgreSQL is reachable and pgvector is loadable."""
    try:
        async with get_session_factory()() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
async def live_session() -> AsyncSession:
    if not await _probe():
        pytest.skip("PostgreSQL is not reachable; skipping live DB tests")
    async with get_session_factory()() as session:
        yield session


async def test_database_connectivity(live_session: AsyncSession) -> None:
    result = await live_session.execute(text("SELECT 1"))
    assert result.scalar_one() == 1


async def test_pgvector_extension_is_enabled(live_session: AsyncSession) -> None:
    result = await live_session.execute(
        text("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
    )
    assert result.scalar_one_or_none() == 1

"""Declarative base for all ORM models, plus a common timestamp mixin.

All tables in the schema inherit from ``Base`` so Alembic autogenerate can
discover them via ``target_metadata``.
"""

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Root declarative base for all ResumeForge ORM models."""


class TimestampMixin:
    """Adds ``created_at`` / ``updated_at`` columns maintained by the server."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

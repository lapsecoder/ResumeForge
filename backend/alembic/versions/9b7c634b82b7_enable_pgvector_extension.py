"""enable pgvector extension

Revision ID: 9b7c634b82b7
Revises:
Create Date: 2026-09-08 18:02:44.990700

"""

from typing import Sequence, Union

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9b7c634b82b7"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable the pgvector extension for future semantic embedding storage."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    """Drop the pgvector extension."""
    op.execute("DROP EXTENSION IF EXISTS vector")

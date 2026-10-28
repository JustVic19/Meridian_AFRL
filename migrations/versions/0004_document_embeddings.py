"""Nullable document embeddings for research retrieval.

Revision ID: 0004_document_embeddings
Revises: 0003_point_in_time
Create Date: 2026-10-27

"""

from collections.abc import Sequence

from alembic import op

revision: str = "0004_document_embeddings"
down_revision: str | None = "0003_point_in_time"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE documents ADD COLUMN embedding vector(32)")


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP COLUMN embedding")

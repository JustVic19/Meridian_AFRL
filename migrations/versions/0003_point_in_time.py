"""Point-in-time columns for documents, observations, and experiments.

Revision ID: 0003_point_in_time
Revises: 0002_research_schema
Create Date: 2026-10-27

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003_point_in_time"
down_revision: str | None = "0002_research_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("period_start", sa.Date()))
    op.add_column("documents", sa.Column("effective_at", sa.DateTime(timezone=True)))
    op.add_column("documents", sa.Column("retrieved_at", sa.DateTime(timezone=True)))

    op.add_column("observations", sa.Column("published_at", sa.DateTime(timezone=True)))
    op.add_column("observations", sa.Column("effective_at", sa.DateTime(timezone=True)))
    op.add_column("observations", sa.Column("period_start", sa.Date()))
    op.add_column("observations", sa.Column("period_end", sa.Date()))
    op.add_column("observations", sa.Column("retrieved_at", sa.DateTime(timezone=True)))
    op.create_index("ix_observations_published_at", "observations", ["published_at"])

    op.add_column("experiments", sa.Column("data_cutoff", sa.DateTime(timezone=True)))
    op.add_column("experiments", sa.Column("experiment_timestamp", sa.DateTime(timezone=True)))


def downgrade() -> None:
    op.drop_column("experiments", "experiment_timestamp")
    op.drop_column("experiments", "data_cutoff")
    op.drop_index("ix_observations_published_at", table_name="observations")
    op.drop_column("observations", "retrieved_at")
    op.drop_column("observations", "period_end")
    op.drop_column("observations", "period_start")
    op.drop_column("observations", "effective_at")
    op.drop_column("observations", "published_at")
    op.drop_column("documents", "retrieved_at")
    op.drop_column("documents", "effective_at")
    op.drop_column("documents", "period_start")

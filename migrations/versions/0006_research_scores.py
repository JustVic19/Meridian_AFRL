"""Immutable research scores. A score ranks research and cannot authorize a trade.

Revision ID: 0006_research_scores
Revises: 0005_validation_results
Create Date: 2026-10-28

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from schemas.research import sql_in
from schemas.scoring import SCORE_STATUSES

revision: str = "0006_research_scores"
down_revision: str | None = "0005_validation_results"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IMMUTABILITY_FUNCTION = """
CREATE FUNCTION reject_research_score_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'research scores are immutable';
END;
$$;
"""


def upgrade() -> None:
    op.execute(_IMMUTABILITY_FUNCTION)
    op.create_table(
        "research_scores",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("policy_version", sa.Text(), nullable=False),
        sa.Column("policy", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("score", sa.Text()),
        sa.Column("components", postgresql.JSONB(), nullable=False),
        sa.Column("weights", postgresql.JSONB(), nullable=False),
        sa.Column("inputs", postgresql.JSONB(), nullable=False),
        sa.Column("live_trade_authorized", sa.Boolean(), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(sql_in("status", SCORE_STATUSES), name="ck_research_scores_status"),
        sa.CheckConstraint(
            "live_trade_authorized = false",
            name="ck_research_scores_not_a_trade",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name="fk_research_scores_experiment_id",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "experiment_id",
            "policy_version",
            name="uq_research_scores_experiment_policy",
        ),
    )
    op.create_index("ix_research_scores_experiment_id", "research_scores", ["experiment_id"])
    op.execute(
        """
        CREATE TRIGGER research_scores_immutable
        BEFORE UPDATE OR DELETE ON research_scores
        FOR EACH ROW
        EXECUTE FUNCTION reject_research_score_mutation()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS research_scores_immutable ON research_scores")
    op.drop_index("ix_research_scores_experiment_id", table_name="research_scores")
    op.drop_table("research_scores")
    op.execute("DROP FUNCTION IF EXISTS reject_research_score_mutation()")

"""Immutable validation runs. A stored result cannot be rewritten.

Revision ID: 0005_validation_results
Revises: 0004_document_embeddings
Create Date: 2026-10-28

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from schemas.research import sql_in
from schemas.validation import VALIDATION_STATUSES, VALIDATION_TEST_NAMES

revision: str = "0005_validation_results"
down_revision: str | None = "0004_document_embeddings"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IMMUTABILITY_FUNCTION = """
CREATE FUNCTION reject_validation_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'validation results are immutable';
END;
$$;
"""


def upgrade() -> None:
    op.execute(_IMMUTABILITY_FUNCTION)
    op.create_table(
        "validation_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("policy_version", sa.Text(), nullable=False),
        sa.Column("policy", postgresql.JSONB(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            sql_in("status", VALIDATION_STATUSES),
            name="ck_validation_runs_status",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name="fk_validation_runs_experiment_id",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint(
            "experiment_id",
            "policy_version",
            name="uq_validation_runs_experiment_policy",
        ),
    )
    op.create_index("ix_validation_runs_experiment_id", "validation_runs", ["experiment_id"])
    op.execute(
        """
        CREATE TRIGGER validation_runs_immutable
        BEFORE UPDATE OR DELETE ON validation_runs
        FOR EACH ROW
        EXECUTE FUNCTION reject_validation_mutation()
        """
    )
    op.create_table(
        "validation_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("sequence_index", sa.Integer(), nullable=False),
        sa.Column("test_name", sa.Text(), nullable=False),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("metrics", postgresql.JSONB()),
        sa.Column("detail", postgresql.JSONB(), nullable=False),
        sa.CheckConstraint(
            "sequence_index >= 0 AND sequence_index < 8",
            name="ck_validation_results_sequence",
        ),
        sa.CheckConstraint(
            sql_in("test_name", VALIDATION_TEST_NAMES),
            name="ck_validation_results_test_name",
        ),
        sa.CheckConstraint(
            sql_in("status", VALIDATION_STATUSES),
            name="ck_validation_results_status",
        ),
        sa.ForeignKeyConstraint(
            ["run_id"],
            ["validation_runs.id"],
            name="fk_validation_results_run_id",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("run_id", "sequence_index", name="uq_validation_results_index"),
        sa.UniqueConstraint("run_id", "test_name", name="uq_validation_results_test"),
    )
    op.create_index("ix_validation_results_run_id", "validation_results", ["run_id"])
    op.execute(
        """
        CREATE TRIGGER validation_results_immutable
        BEFORE UPDATE OR DELETE ON validation_results
        FOR EACH ROW
        EXECUTE FUNCTION reject_validation_mutation()
        """
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS validation_results_immutable ON validation_results")
    op.drop_index("ix_validation_results_run_id", table_name="validation_results")
    op.drop_table("validation_results")
    op.execute("DROP TRIGGER IF EXISTS validation_runs_immutable ON validation_runs")
    op.drop_index("ix_validation_runs_experiment_id", table_name="validation_runs")
    op.drop_table("validation_runs")
    op.execute("DROP FUNCTION IF EXISTS reject_validation_mutation()")

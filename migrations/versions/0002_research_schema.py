"""Research tables for the V1 laboratory schema.

Revision ID: 0002_research_schema
Revises: 0001_pgvector
Create Date: 2026-10-27

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

from schemas.research import ARTIFACT_TYPES, HYPOTHESIS_STATUSES, sql_date_order, sql_in

revision: str = "0002_research_schema"
down_revision: str | None = "0001_pgvector"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IMMUTABILITY_FUNCTION = """
CREATE FUNCTION reject_finalized_experiment_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.completed_at IS NOT NULL THEN
        RAISE EXCEPTION 'completed experiments are immutable';
    END IF;
    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$;
"""


def upgrade() -> None:
    op.create_table(
        "companies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("ticker", sa.Text(), nullable=False),
        sa.Column("cik", sa.Text()),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("exchange", sa.Text()),
        sa.Column("sector", sa.Text()),
        sa.Column("industry", sa.Text()),
        sa.Column("market_cap", sa.Numeric()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.UniqueConstraint("cik", name="uq_companies_cik"),
    )
    op.create_index("ix_companies_ticker", "companies", ["ticker"])

    op.create_table(
        "documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("company_id", postgresql.UUID(as_uuid=True)),
        sa.Column("document_type", sa.Text(), nullable=False),
        sa.Column("accession_number", sa.Text()),
        sa.Column("filing_date", sa.Date()),
        sa.Column("period_end", sa.Date()),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("source_url", sa.Text()),
        sa.Column("content", sa.Text()),
        sa.Column("content_hash", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name="fk_documents_company_id",
            ondelete="RESTRICT",
        ),
        sa.UniqueConstraint("content_hash", name="uq_documents_content_hash"),
    )
    op.create_index("ix_documents_company_id", "documents", ["company_id"])
    op.create_index("ix_documents_published_at", "documents", ["published_at"])
    op.create_index("ix_documents_accession_number", "documents", ["accession_number"])

    op.create_table(
        "observations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("company_id", postgresql.UUID(as_uuid=True)),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text()),
        sa.Column("evidence", postgresql.JSONB()),
        sa.Column("confidence", sa.Numeric()),
        sa.Column("detected_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.ForeignKeyConstraint(
            ["company_id"],
            ["companies.id"],
            name="fk_observations_company_id",
            ondelete="RESTRICT",
        ),
    )
    op.create_index("ix_observations_company_id", "observations", ["company_id"])
    op.create_index("ix_observations_detected_at", "observations", ["detected_at"])

    op.create_table(
        "hypotheses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("statement", sa.Text(), nullable=False),
        sa.Column("universe", postgresql.JSONB()),
        sa.Column("mechanism", sa.Text()),
        sa.Column("entry_rule", postgresql.JSONB()),
        sa.Column("exit_rule", postgresql.JSONB()),
        sa.Column("expected_horizon", sa.Integer()),
        sa.Column("parent_hypothesis_id", postgresql.UUID(as_uuid=True)),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(sql_in("status", HYPOTHESIS_STATUSES), name="ck_hypotheses_status"),
        sa.CheckConstraint("id <> parent_hypothesis_id", name="ck_hypotheses_parent_not_self"),
        sa.ForeignKeyConstraint(
            ["parent_hypothesis_id"],
            ["hypotheses.id"],
            name="fk_hypotheses_parent_hypothesis_id",
            ondelete="RESTRICT",
        ),
    )
    op.create_index("ix_hypotheses_parent_hypothesis_id", "hypotheses", ["parent_hypothesis_id"])
    op.create_index("ix_hypotheses_status", "hypotheses", ["status"])

    op.create_table(
        "experiments",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("hypothesis_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("dataset_version", sa.Text(), nullable=False),
        sa.Column("code_version", sa.Text(), nullable=False),
        sa.Column("train_start", sa.Date()),
        sa.Column("train_end", sa.Date()),
        sa.Column("validation_start", sa.Date()),
        sa.Column("validation_end", sa.Date()),
        sa.Column("test_start", sa.Date()),
        sa.Column("test_end", sa.Date()),
        sa.Column("specification", postgresql.JSONB()),
        sa.Column("parameters", postgresql.JSONB()),
        sa.Column("metrics", postgresql.JSONB()),
        sa.Column("statistical_tests", postgresql.JSONB()),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column("parent_experiment_id", postgresql.UUID(as_uuid=True)),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            sql_date_order("train_start", "train_end"),
            name="ck_experiments_train_dates",
        ),
        sa.CheckConstraint(
            sql_date_order("validation_start", "validation_end"),
            name="ck_experiments_validation_dates",
        ),
        sa.CheckConstraint(
            sql_date_order("test_start", "test_end"),
            name="ck_experiments_test_dates",
        ),
        sa.CheckConstraint("id <> parent_experiment_id", name="ck_experiments_parent_not_self"),
        sa.ForeignKeyConstraint(
            ["hypothesis_id"],
            ["hypotheses.id"],
            name="fk_experiments_hypothesis_id",
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["parent_experiment_id"],
            ["experiments.id"],
            name="fk_experiments_parent_experiment_id",
            ondelete="RESTRICT",
        ),
    )
    op.create_index("ix_experiments_hypothesis_id", "experiments", ["hypothesis_id"])
    op.create_index("ix_experiments_parent_experiment_id", "experiments", ["parent_experiment_id"])
    op.create_index("ix_experiments_status", "experiments", ["status"])
    op.execute(_IMMUTABILITY_FUNCTION)
    op.execute(
        """
        CREATE TRIGGER experiments_immutable_when_finalized
        BEFORE UPDATE OR DELETE ON experiments
        FOR EACH ROW
        EXECUTE FUNCTION reject_finalized_experiment_mutation()
        """
    )

    op.create_table(
        "research_artifacts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_id", postgresql.UUID(as_uuid=True)),
        sa.Column("artifact_type", sa.Text(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text()),
        sa.Column("model_version", sa.Text()),
        sa.Column("prompt_version", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint(
            sql_in("artifact_type", ARTIFACT_TYPES),
            name="ck_research_artifacts_artifact_type",
        ),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name="fk_research_artifacts_experiment_id",
            ondelete="RESTRICT",
        ),
    )
    op.create_index("ix_research_artifacts_experiment_id", "research_artifacts", ["experiment_id"])
    op.create_index("ix_research_artifacts_artifact_type", "research_artifacts", ["artifact_type"])

    op.create_table(
        "strategies",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_id", postgresql.UUID(as_uuid=True)),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("signal_definition", postgresql.JSONB()),
        sa.Column("portfolio_rules", postgresql.JSONB()),
        sa.Column("risk_rules", postgresql.JSONB()),
        sa.Column("capacity_estimate", sa.Numeric()),
        sa.Column("status", sa.Text()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.CheckConstraint("version >= 1", name="ck_strategies_version"),
        sa.ForeignKeyConstraint(
            ["experiment_id"],
            ["experiments.id"],
            name="fk_strategies_experiment_id",
            ondelete="RESTRICT",
        ),
    )
    op.create_index("ix_strategies_experiment_id", "strategies", ["experiment_id"])


def downgrade() -> None:
    op.drop_table("strategies")
    op.drop_table("research_artifacts")
    op.execute("DROP TRIGGER IF EXISTS experiments_immutable_when_finalized ON experiments")
    op.drop_table("experiments")
    op.execute("DROP FUNCTION IF EXISTS reject_finalized_experiment_mutation()")
    op.drop_table("hypotheses")
    op.drop_table("observations")
    op.drop_table("documents")
    op.drop_table("companies")

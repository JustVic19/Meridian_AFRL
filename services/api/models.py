"""SQLAlchemy models for the V1 research tables."""

import uuid
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from schemas.research import ARTIFACT_TYPES, HYPOTHESIS_STATUSES, sql_date_order, sql_in
from schemas.scoring import SCORE_STATUSES
from schemas.validation import VALIDATION_STATUSES, VALIDATION_TEST_NAMES

JsonDocument = dict[str, Any] | list[Any]


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"
    __table_args__ = (
        UniqueConstraint("cik", name="uq_companies_cik"),
        Index("ix_companies_ticker", "ticker"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    ticker: Mapped[str] = mapped_column(Text, nullable=False)
    cik: Mapped[str | None] = mapped_column(Text)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    exchange: Mapped[str | None] = mapped_column(Text)
    sector: Mapped[str | None] = mapped_column(Text)
    industry: Mapped[str | None] = mapped_column(Text)
    market_cap: Mapped[Decimal | None] = mapped_column(Numeric)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("content_hash", name="uq_documents_content_hash"),
        Index("ix_documents_company_id", "company_id"),
        Index("ix_documents_published_at", "published_at"),
        Index("ix_documents_accession_number", "accession_number"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", name="fk_documents_company_id", ondelete="RESTRICT"),
    )
    document_type: Mapped[str] = mapped_column(Text, nullable=False)
    accession_number: Mapped[str | None] = mapped_column(Text)
    filing_date: Mapped[date | None] = mapped_column(Date)
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date | None] = mapped_column(Date)
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    source_url: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(Text)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(32))
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Observation(Base):
    __tablename__ = "observations"
    __table_args__ = (
        Index("ix_observations_company_id", "company_id"),
        Index("ix_observations_detected_at", "detected_at"),
        Index("ix_observations_published_at", "published_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    company_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("companies.id", name="fk_observations_company_id", ondelete="RESTRICT"),
    )
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    evidence: Mapped[JsonDocument | None] = mapped_column(JSONB)
    confidence: Mapped[Decimal | None] = mapped_column(Numeric)
    detected_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    period_start: Mapped[date | None] = mapped_column(Date)
    period_end: Mapped[date | None] = mapped_column(Date)
    retrieved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Hypothesis(Base):
    __tablename__ = "hypotheses"
    __table_args__ = (
        CheckConstraint(sql_in("status", HYPOTHESIS_STATUSES), name="ck_hypotheses_status"),
        CheckConstraint("id <> parent_hypothesis_id", name="ck_hypotheses_parent_not_self"),
        Index("ix_hypotheses_parent_hypothesis_id", "parent_hypothesis_id"),
        Index("ix_hypotheses_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    statement: Mapped[str] = mapped_column(Text, nullable=False)
    universe: Mapped[JsonDocument | None] = mapped_column(JSONB)
    mechanism: Mapped[str | None] = mapped_column(Text)
    entry_rule: Mapped[JsonDocument | None] = mapped_column(JSONB)
    exit_rule: Mapped[JsonDocument | None] = mapped_column(JSONB)
    expected_horizon: Mapped[int | None] = mapped_column(Integer)
    parent_hypothesis_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "hypotheses.id",
            name="fk_hypotheses_parent_hypothesis_id",
            ondelete="RESTRICT",
        ),
    )
    status: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Experiment(Base):
    __tablename__ = "experiments"
    __table_args__ = (
        CheckConstraint(
            sql_date_order("train_start", "train_end"),
            name="ck_experiments_train_dates",
        ),
        CheckConstraint(
            sql_date_order("validation_start", "validation_end"),
            name="ck_experiments_validation_dates",
        ),
        CheckConstraint(sql_date_order("test_start", "test_end"), name="ck_experiments_test_dates"),
        CheckConstraint("id <> parent_experiment_id", name="ck_experiments_parent_not_self"),
        Index("ix_experiments_hypothesis_id", "hypothesis_id"),
        Index("ix_experiments_parent_experiment_id", "parent_experiment_id"),
        Index("ix_experiments_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    hypothesis_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("hypotheses.id", name="fk_experiments_hypothesis_id", ondelete="RESTRICT"),
        nullable=False,
    )
    dataset_version: Mapped[str] = mapped_column(Text, nullable=False)
    data_cutoff: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    code_version: Mapped[str] = mapped_column(Text, nullable=False)
    experiment_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    train_start: Mapped[date | None] = mapped_column(Date)
    train_end: Mapped[date | None] = mapped_column(Date)
    validation_start: Mapped[date | None] = mapped_column(Date)
    validation_end: Mapped[date | None] = mapped_column(Date)
    test_start: Mapped[date | None] = mapped_column(Date)
    test_end: Mapped[date | None] = mapped_column(Date)
    specification: Mapped[JsonDocument | None] = mapped_column(JSONB)
    parameters: Mapped[JsonDocument | None] = mapped_column(JSONB)
    metrics: Mapped[JsonDocument | None] = mapped_column(JSONB)
    statistical_tests: Mapped[JsonDocument | None] = mapped_column(JSONB)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    parent_experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "experiments.id",
            name="fk_experiments_parent_experiment_id",
            ondelete="RESTRICT",
        ),
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ResearchArtifact(Base):
    __tablename__ = "research_artifacts"
    __table_args__ = (
        CheckConstraint(
            sql_in("artifact_type", ARTIFACT_TYPES),
            name="ck_research_artifacts_artifact_type",
        ),
        Index("ix_research_artifacts_experiment_id", "experiment_id"),
        Index("ix_research_artifacts_artifact_type", "artifact_type"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "experiments.id",
            name="fk_research_artifacts_experiment_id",
            ondelete="RESTRICT",
        ),
    )
    artifact_type: Mapped[str] = mapped_column(Text, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str | None] = mapped_column(Text)
    model_version: Mapped[str | None] = mapped_column(Text)
    prompt_version: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class Strategy(Base):
    __tablename__ = "strategies"
    __table_args__ = (
        CheckConstraint("version >= 1", name="ck_strategies_version"),
        Index("ix_strategies_experiment_id", "experiment_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experiments.id", name="fk_strategies_experiment_id", ondelete="RESTRICT"),
    )
    name: Mapped[str] = mapped_column(Text, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    signal_definition: Mapped[JsonDocument | None] = mapped_column(JSONB)
    portfolio_rules: Mapped[JsonDocument | None] = mapped_column(JSONB)
    risk_rules: Mapped[JsonDocument | None] = mapped_column(JSONB)
    capacity_estimate: Mapped[Decimal | None] = mapped_column(Numeric)
    status: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ValidationRun(Base):
    __tablename__ = "validation_runs"
    __table_args__ = (
        CheckConstraint(sql_in("status", VALIDATION_STATUSES), name="ck_validation_runs_status"),
        UniqueConstraint(
            "experiment_id",
            "policy_version",
            name="uq_validation_runs_experiment_policy",
        ),
        Index("ix_validation_runs_experiment_id", "experiment_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "experiments.id",
            name="fk_validation_runs_experiment_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    policy_version: Mapped[str] = mapped_column(Text, nullable=False)
    policy: Mapped[JsonDocument] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class ValidationResult(Base):
    __tablename__ = "validation_results"
    __table_args__ = (
        CheckConstraint(
            "sequence_index >= 0 AND sequence_index < 8",
            name="ck_validation_results_sequence",
        ),
        CheckConstraint(
            sql_in("test_name", VALIDATION_TEST_NAMES),
            name="ck_validation_results_test_name",
        ),
        CheckConstraint(
            sql_in("status", VALIDATION_STATUSES),
            name="ck_validation_results_status",
        ),
        UniqueConstraint("run_id", "sequence_index", name="uq_validation_results_index"),
        UniqueConstraint("run_id", "test_name", name="uq_validation_results_test"),
        Index("ix_validation_results_run_id", "run_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    run_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "validation_runs.id",
            name="fk_validation_results_run_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    sequence_index: Mapped[int] = mapped_column(Integer, nullable=False)
    test_name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    metrics: Mapped[JsonDocument | None] = mapped_column(JSONB)
    detail: Mapped[JsonDocument] = mapped_column(JSONB, nullable=False)


class ResearchScore(Base):
    __tablename__ = "research_scores"
    __table_args__ = (
        CheckConstraint(sql_in("status", SCORE_STATUSES), name="ck_research_scores_status"),
        CheckConstraint(
            "live_trade_authorized = false",
            name="ck_research_scores_not_a_trade",
        ),
        UniqueConstraint(
            "experiment_id",
            "policy_version",
            name="uq_research_scores_experiment_policy",
        ),
        Index("ix_research_scores_experiment_id", "experiment_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    experiment_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey(
            "experiments.id",
            name="fk_research_scores_experiment_id",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    policy_version: Mapped[str] = mapped_column(Text, nullable=False)
    policy: Mapped[JsonDocument] = mapped_column(JSONB, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False)
    score: Mapped[str | None] = mapped_column(Text)
    components: Mapped[JsonDocument] = mapped_column(JSONB, nullable=False)
    weights: Mapped[JsonDocument] = mapped_column(JSONB, nullable=False)
    inputs: Mapped[JsonDocument] = mapped_column(JSONB, nullable=False)
    live_trade_authorized: Mapped[bool] = mapped_column(Boolean, nullable=False)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

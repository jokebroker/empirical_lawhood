'Catalog and operational-control tables for the first public baseline.\n\nKeep released table definitions fixed. Schema changes require a migration;\nthe runtime uses separate SQLAlchemy metadata objects.'

from __future__ import annotations

from typing import Final

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)

NAMING_CONVENTION: Final[dict[str, str]] = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}
CATALOG_METADATA: Final[MetaData] = MetaData(naming_convention=NAMING_CONVENTION)


def _id(name: str, *, nullable: bool = False) -> Column[str]:
    return Column(name, String(200), nullable=nullable)


storage_root = Table(
    "storage_root",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("storage_root_id"),
    Column("logical_name", String(200), nullable=False),
    Column("canonical_path", Text, nullable=False),
    Column("mount_contract_schema", String(300), nullable=False),
    Column("verification_status", String(32), nullable=False),
    UniqueConstraint("storage_root_id"),
    UniqueConstraint("canonical_path"),
    CheckConstraint("length(canonical_path) <= 4096", name="bounded_path"),
)

logical_artifact = Table(
    "logical_artifact",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("logical_artifact_id"),
    Column("content_sha256", LargeBinary(32), nullable=False),
    Column("payload_schema", String(300), nullable=False),
    Column("profile", String(40), nullable=False),
    Column("media_type", String(200), nullable=False),
    Column("visibility_ceiling", String(40), nullable=False),
    UniqueConstraint("logical_artifact_id"),
    CheckConstraint("length(content_sha256) = 32", name="content_sha256_length"),
)

artifact = Table(
    "artifact",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("materialization_id"),
    Column("logical_artifact_pk", ForeignKey("logical_artifact.pk"), nullable=False),
    Column("storage_root_pk", ForeignKey("storage_root.pk"), nullable=False),
    Column("relative_path", Text, nullable=False),
    Column("physical_sha256", LargeBinary(32), nullable=False),
    Column("size_bytes", BigInteger, nullable=False),
    Column("compression", String(40), nullable=False),
    Column("partition_selector", String(512)),
    Column("verification_status", String(32), nullable=False),
    UniqueConstraint("materialization_id"),
    UniqueConstraint("storage_root_pk", "relative_path"),
    CheckConstraint("length(relative_path) <= 4096", name="bounded_path"),
    CheckConstraint("length(physical_sha256) = 32", name="physical_sha256_length"),
    CheckConstraint("size_bytes >= 0", name="nonnegative_size"),
)
Index("ix_artifact_root_path", artifact.c.storage_root_pk, artifact.c.relative_path)

receipt = Table(
    "receipt",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("receipt_id"),
    _id("run_id"),
    Column("storage_root_pk", ForeignKey("storage_root.pk"), nullable=False),
    Column("relative_path", Text, nullable=False),
    Column("sha256", LargeBinary(32), nullable=False),
    Column("receipt_schema", String(300), nullable=False),
    Column("verification_status", String(32), nullable=False),
    UniqueConstraint("receipt_id"),
    UniqueConstraint("run_id"),
    CheckConstraint("length(relative_path) <= 4096", name="bounded_path"),
    CheckConstraint("length(sha256) = 32", name="sha256_length"),
)

scientific_object = Table(
    "scientific_object",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("scientific_object_id"),
    Column("kind", String(64), nullable=False),
    Column("object_schema", String(300), nullable=False),
    Column("fingerprint", LargeBinary(32), nullable=False),
    Column("visibility_ceiling", String(40), nullable=False),
    Column("categorical_status", String(80), nullable=False),
    Column("artifact_pk", ForeignKey("artifact.pk")),
    Column("receipt_pk", ForeignKey("receipt.pk")),
    UniqueConstraint("scientific_object_id"),
    UniqueConstraint("kind", "fingerprint"),
    CheckConstraint("length(fingerprint) = 32", name="fingerprint_length"),
    CheckConstraint(
        "artifact_pk IS NOT NULL OR receipt_pk IS NOT NULL",
        name="external_evidence_required",
    ),
)
Index(
    "ix_scientific_object_kind_status",
    scientific_object.c.kind,
    scientific_object.c.categorical_status,
)

PROJECTION_TABLE_NAMES: Final[tuple[str, ...]] = (
    "admission_set",
    "analysis_family",
    "analysis_proposal",
    "analysis_spec",
    "anomaly_signal",
    "atlas_patch",
    "authority_policy",
    "authorization_record",
    "campaign",
    "claim_spec",
    "computability_envelope",
    "controller_spec",
    "decision_record",
    "discrepancy_result",
    "evidence_snapshot",
    "experiment_proposal",
    "experiment_spec",
    "exploration_plan",
    "exploratory_finding",
    "hypothesis",
    "hypothesis_set",
    "model_relation",
    "model_set",
    "numerical_view",
    "prospective_nomination",
    "reachability_result",
    "response_law",
    "structural_convergence_result",
    "system_spec",
    "transport_result",
    "world_spec",
)

projection_tables = {
    table_name: Table(
        table_name,
        CATALOG_METADATA,
        Column(
            "scientific_object_pk",
            ForeignKey("scientific_object.pk"),
            primary_key=True,
        ),
    )
    for table_name in PROJECTION_TABLE_NAMES
}

knowledge_edge = Table(
    "knowledge_edge",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("edge_id"),
    Column("relation", String(80), nullable=False),
    Column("source_object_pk", ForeignKey("scientific_object.pk"), nullable=False),
    Column("target_object_pk", ForeignKey("scientific_object.pk"), nullable=False),
    Column("evidence_object_pk", ForeignKey("scientific_object.pk")),
    _id("scope_id"),
    UniqueConstraint("edge_id"),
)
Index("ix_knowledge_edge_source", knowledge_edge.c.source_object_pk, knowledge_edge.c.relation)
Index("ix_knowledge_edge_target", knowledge_edge.c.target_object_pk, knowledge_edge.c.relation)
Index("ix_knowledge_edge_evidence", knowledge_edge.c.evidence_object_pk, knowledge_edge.c.relation)
Index("ix_knowledge_edge_relation_scope", knowledge_edge.c.relation, knowledge_edge.c.scope_id)

metric_definition = Table(
    "metric_definition",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("metric_definition_id"),
    _id("metric_name"),
    _id("dataset_version_id"),
    _id("relation_id"),
    _id("denominator_gauge_id"),
    _id("response_gauge_id"),
    _id("horizon_id"),
    Column("native_unit", String(80), nullable=False),
    Column("aggregation", String(80), nullable=False),
    Column("direction", String(80), nullable=False),
    UniqueConstraint("metric_definition_id"),
    UniqueConstraint(
        "dataset_version_id",
        "metric_name",
        "relation_id",
        "denominator_gauge_id",
        "response_gauge_id",
        "horizon_id",
        "native_unit",
        "aggregation",
    ),
)
Index("ix_metric_definition_relation", metric_definition.c.relation_id)
Index("ix_metric_definition_denominator", metric_definition.c.denominator_gauge_id)
Index("ix_metric_definition_response", metric_definition.c.response_gauge_id)
Index("ix_metric_definition_horizon", metric_definition.c.horizon_id)
Index("ix_metric_definition_native_unit", metric_definition.c.native_unit)

metric_observation = Table(
    "metric_observation",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("metric_observation_id"),
    Column("metric_definition_pk", ForeignKey("metric_definition.pk"), nullable=False),
    Column("scientific_object_pk", ForeignKey("scientific_object.pk"), nullable=False),
    Column("point_decimal", String(100), nullable=False),
    Column("lower_decimal", String(100)),
    Column("upper_decimal", String(100)),
    Column("physical_independent_unit_count", Integer, nullable=False),
    Column("numerical_view_count", Integer, nullable=False),
    UniqueConstraint("metric_observation_id"),
    CheckConstraint("physical_independent_unit_count > 0", name="positive_units"),
    CheckConstraint("numerical_view_count >= 0", name="nonnegative_views"),
)
Index(
    "ix_metric_observation_object_definition",
    metric_observation.c.scientific_object_pk,
    metric_observation.c.metric_definition_pk,
)

exploration_attempt = Table(
    "exploration_attempt",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("attempt_id"),
    Column("analysis_spec_object_pk", ForeignKey("scientific_object.pk"), nullable=False),
    Column("disposition", String(80), nullable=False),
    UniqueConstraint("attempt_id"),
)
exploration_attempt_reason = Table(
    "exploration_attempt_reason",
    CATALOG_METADATA,
    Column("attempt_pk", ForeignKey("exploration_attempt.pk"), primary_key=True),
    Column("ordinal", Integer, primary_key=True),
    _id("reason_code"),
)
exploration_attempt_artifact = Table(
    "exploration_attempt_artifact",
    CATALOG_METADATA,
    Column("attempt_pk", ForeignKey("exploration_attempt.pk"), primary_key=True),
    Column("artifact_pk", ForeignKey("artifact.pk"), primary_key=True),
)

workflow_run = Table(
    "workflow_run",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("run_id"),
    Column("plan_fingerprint", LargeBinary(32), nullable=False),
    Column("state", String(40), nullable=False),
    UniqueConstraint("run_id"),
    CheckConstraint("length(plan_fingerprint) = 32", name="plan_fingerprint_length"),
)
task_attempt = Table(
    "task_attempt",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("attempt_id"),
    Column("workflow_run_pk", ForeignKey("workflow_run.pk"), nullable=False),
    _id("task_id"),
    Column("state", String(40), nullable=False),
    Column("reason_code", String(200)),
    UniqueConstraint("attempt_id"),
)
lease = Table(
    "lease",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("lease_id"),
    Column("task_attempt_pk", ForeignKey("task_attempt.pk"), nullable=False),
    Column("owner_id", String(200), nullable=False),
    Column("expires_epoch_seconds", BigInteger, nullable=False),
    UniqueConstraint("lease_id"),
    UniqueConstraint("task_attempt_pk"),
)
run_event = Table(
    "run_event",
    CATALOG_METADATA,
    Column("pk", Integer, primary_key=True),
    _id("event_id"),
    Column("workflow_run_pk", ForeignKey("workflow_run.pk"), nullable=False),
    Column("sequence", Integer, nullable=False),
    Column("event_kind", String(80), nullable=False),
    Column("reason_code", String(200)),
    UniqueConstraint("event_id"),
    UniqueConstraint("workflow_run_pk", "sequence"),
)

CATALOG_TABLE_NAMES: Final[frozenset[str]] = frozenset(CATALOG_METADATA.tables)

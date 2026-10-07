"""Infrastructure-neutral compact catalog records and repository port."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Final, Protocol

from empirical_lawhood.kernel.evidence import VisibilityCeiling
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)

from .artifacts import ArtifactProfile


MAX_CATALOG_SNAPSHOT_RECORDS: Final[int] = 100_000


def validate_catalog_snapshot_record_count(*counts: int) -> None:
    """Bound aggregate projection work before uniqueness/reference indexing."""

    if any(count < 0 for count in counts):
        raise ValueError("catalog snapshot record counts must be nonnegative")
    if sum(counts) > MAX_CATALOG_SNAPSHOT_RECORDS:
        raise ValueError("catalog snapshot exceeds its aggregate record limit")


class ScientificObjectKind(StrEnum):
    WORLD_SPEC = "WORLD_SPEC"
    NUMERICAL_VIEW = "NUMERICAL_VIEW"
    COMPUTABILITY_ENVELOPE = "COMPUTABILITY_ENVELOPE"
    SYSTEM_SPEC = "SYSTEM_SPEC"
    CLAIM_SPEC = "CLAIM_SPEC"
    MODEL_SET = "MODEL_SET"
    MODEL_RELATION = "MODEL_RELATION"
    DISCREPANCY_RESULT = "DISCREPANCY_RESULT"
    TRANSPORT_RESULT = "TRANSPORT_RESULT"
    STRUCTURAL_CONVERGENCE_RESULT = "STRUCTURAL_CONVERGENCE_RESULT"
    EXPERIMENT_SPEC = "EXPERIMENT_SPEC"
    CAMPAIGN = "CAMPAIGN"
    EXPERIMENT_PROPOSAL = "EXPERIMENT_PROPOSAL"
    AUTHORITY_POLICY = "AUTHORITY_POLICY"
    AUTHORIZATION_RECORD = "AUTHORIZATION_RECORD"
    DECISION_RECORD = "DECISION_RECORD"
    EVIDENCE_SNAPSHOT = "EVIDENCE_SNAPSHOT"
    ANOMALY_SIGNAL = "ANOMALY_SIGNAL"
    ANALYSIS_FAMILY = "ANALYSIS_FAMILY"
    ANALYSIS_SPEC = "ANALYSIS_SPEC"
    ANALYSIS_PROPOSAL = "ANALYSIS_PROPOSAL"
    EXPLORATION_PLAN = "EXPLORATION_PLAN"
    EXPLORATORY_FINDING = "EXPLORATORY_FINDING"
    HYPOTHESIS_SET = "HYPOTHESIS_SET"
    HYPOTHESIS = "HYPOTHESIS"
    PROSPECTIVE_NOMINATION = "PROSPECTIVE_NOMINATION"
    RESPONSE_LAW = "RESPONSE_LAW"
    ATLAS_PATCH = "ATLAS_PATCH"
    ADMISSION_SET = "ADMISSION_SET"
    REACHABILITY_RESULT = "REACHABILITY_RESULT"
    CONTROLLER_SPEC = "CONTROLLER_SPEC"


class CatalogVerificationStatus(StrEnum):
    VERIFIED = "VERIFIED"
    UNAVAILABLE = "UNAVAILABLE"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class StorageRootRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/storage-root-record'

    storage_root_id: str
    logical_name: str
    canonical_path: str
    mount_contract_schema: str
    verification_status: CatalogVerificationStatus

    def __post_init__(self) -> None:
        validate_stable_id(self.storage_root_id, field_name="storage_root_id")
        validate_nonempty(self.logical_name, field_name="logical_name")
        validate_nonempty(self.canonical_path, field_name="canonical_path")
        if not self.canonical_path.startswith("/"):
            raise ValueError("catalog storage root must be an absolute canonical path")
        validate_schema(self.mount_contract_schema)


@dataclass(frozen=True, slots=True)
class LogicalArtifactRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/logical-artifact-record'

    logical_artifact_id: str
    content_sha256: str
    payload_schema: str
    profile: ArtifactProfile
    media_type: str
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.logical_artifact_id, field_name="logical_artifact_id")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")


@dataclass(frozen=True, slots=True)
class ArtifactLocatorRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/artifact-locator-record'

    materialization_id: str
    logical_artifact_id: str
    storage_root_id: str
    relative_path: str
    physical_sha256: str
    size_bytes: int
    compression: str
    partition_selector: str | None
    verification_status: CatalogVerificationStatus

    def __post_init__(self) -> None:
        for name, value in (
            ("materialization_id", self.materialization_id),
            ("logical_artifact_id", self.logical_artifact_id),
            ("storage_root_id", self.storage_root_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_relative_locator(self.relative_path)
        validate_sha256(self.physical_sha256, field_name="physical_sha256")
        if self.size_bytes < 0:
            raise ValueError("catalog artifact size must be nonnegative")
        validate_nonempty(self.compression, field_name="compression")
        if self.partition_selector is not None:
            validate_nonempty(self.partition_selector, field_name="partition_selector")
            if len(self.partition_selector.encode("utf-8")) > 512:
                raise ValueError("catalog partition selector exceeds 512 bytes")


@dataclass(frozen=True, slots=True)
class ReceiptLocatorRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/receipt-locator-record'

    receipt_id: str
    run_id: str
    storage_root_id: str
    relative_path: str
    sha256: str
    receipt_schema: str
    verification_status: CatalogVerificationStatus

    def __post_init__(self) -> None:
        for name, value in (("receipt_id", self.receipt_id), ("run_id", self.run_id)):
            validate_stable_id(value, field_name=name)
        validate_stable_id(self.storage_root_id, field_name="storage_root_id")
        validate_relative_locator(self.relative_path)
        validate_sha256(self.sha256)
        validate_schema(self.receipt_schema)


@dataclass(frozen=True, slots=True)
class ScientificObjectRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/scientific-object-record'

    scientific_object_id: str
    kind: ScientificObjectKind
    object_schema: str
    object_sha256: str
    visibility_ceiling: VisibilityCeiling
    categorical_status: str
    artifact_materialization_id: str | None
    receipt_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.scientific_object_id, field_name="scientific_object_id")
        validate_schema(self.object_schema)
        validate_sha256(self.object_sha256, field_name="object_sha256")
        validate_stable_id(self.categorical_status, field_name="categorical_status")
        if self.artifact_materialization_id is not None:
            validate_stable_id(
                self.artifact_materialization_id,
                field_name="artifact_materialization_id",
            )
        if self.receipt_id is not None:
            validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.artifact_materialization_id is None and self.receipt_id is None:
            raise ValueError("scientific catalog objects require external evidence identity")


@dataclass(frozen=True, slots=True)
class KnowledgeEdgeRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/knowledge-edge-record'

    edge_id: str
    relation: str
    source_object_id: str
    target_object_id: str
    evidence_object_id: str | None
    scope_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("edge_id", self.edge_id),
            ("relation", self.relation),
            ("source_object_id", self.source_object_id),
            ("target_object_id", self.target_object_id),
            ("scope_id", self.scope_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.evidence_object_id is not None:
            validate_stable_id(self.evidence_object_id, field_name="evidence_object_id")


@dataclass(frozen=True, slots=True)
class MetricDefinitionRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/metric-definition-record'

    metric_definition_id: str
    metric_name: str
    dataset_version_id: str
    relation_id: str
    denominator_gauge_id: str
    response_gauge_id: str
    horizon_id: str
    native_unit: str
    aggregation: str
    direction: str

    def __post_init__(self) -> None:
        for name, value in (
            ("metric_definition_id", self.metric_definition_id),
            ("metric_name", self.metric_name),
            ("dataset_version_id", self.dataset_version_id),
            ("relation_id", self.relation_id),
            ("denominator_gauge_id", self.denominator_gauge_id),
            ("response_gauge_id", self.response_gauge_id),
            ("horizon_id", self.horizon_id),
            ("aggregation", self.aggregation),
            ("direction", self.direction),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class MetricObservationRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/metric-observation-record'

    metric_observation_id: str
    metric_definition_id: str
    scientific_object_id: str
    point: Decimal
    lower: Decimal | None
    upper: Decimal | None
    physical_independent_unit_count: int
    numerical_view_count: int

    def __post_init__(self) -> None:
        for name, value in (
            ("metric_observation_id", self.metric_observation_id),
            ("metric_definition_id", self.metric_definition_id),
            ("scientific_object_id", self.scientific_object_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.point, field_name="point")
        if (self.lower is None) is not (self.upper is None):
            raise ValueError("metric interval endpoints must be both present or absent")
        if self.lower is not None and self.upper is not None:
            validate_decimal(self.lower, field_name="lower")
            validate_decimal(self.upper, field_name="upper")
            if not self.lower <= self.point <= self.upper:
                raise ValueError("metric point must lie inside its interval")
        if self.physical_independent_unit_count <= 0:
            raise ValueError("metric requires physical independent units")
        if self.numerical_view_count < 0:
            raise ValueError("numerical-view count must be nonnegative")


@dataclass(frozen=True, slots=True)
class ExplorationAttemptRecord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/exploration-attempt-record'

    attempt_id: str
    analysis_spec_object_id: str
    disposition: str
    reason_codes: tuple[str, ...]
    artifact_materialization_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("attempt_id", self.attempt_id),
            ("analysis_spec_object_id", self.analysis_spec_object_id),
            ("disposition", self.disposition),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.artifact_materialization_ids,
            field_name="artifact_materialization_ids",
        )
        if not self.reason_codes and not self.artifact_materialization_ids:
            raise ValueError("exploration attempt must retain output or terminal reason")


@dataclass(frozen=True, slots=True)
class CatalogSnapshot(CanonicalRecord):
    """Complete append-only scientific projection; leases are intentionally absent."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/catalog-snapshot'

    storage_roots: tuple[StorageRootRecord, ...]
    logical_artifacts: tuple[LogicalArtifactRecord, ...]
    artifact_locators: tuple[ArtifactLocatorRecord, ...]
    receipts: tuple[ReceiptLocatorRecord, ...]
    scientific_objects: tuple[ScientificObjectRecord, ...]
    knowledge_edges: tuple[KnowledgeEdgeRecord, ...]
    metric_definitions: tuple[MetricDefinitionRecord, ...]
    metric_observations: tuple[MetricObservationRecord, ...]
    exploration_attempts: tuple[ExplorationAttemptRecord, ...]

    def __post_init__(self) -> None:
        validate_catalog_snapshot_record_count(
            len(self.storage_roots),
            len(self.logical_artifacts),
            len(self.artifact_locators),
            len(self.receipts),
            len(self.scientific_objects),
            len(self.knowledge_edges),
            len(self.metric_definitions),
            len(self.metric_observations),
            len(self.exploration_attempts),
        )
        for values, attribute, name in (
            (self.storage_roots, "storage_root_id", "storage_roots"),
            (self.logical_artifacts, "logical_artifact_id", "logical_artifacts"),
            (self.artifact_locators, "materialization_id", "artifact_locators"),
            (self.receipts, "receipt_id", "receipts"),
            (self.scientific_objects, "scientific_object_id", "scientific_objects"),
            (self.knowledge_edges, "edge_id", "knowledge_edges"),
            (self.metric_definitions, "metric_definition_id", "metric_definitions"),
            (self.metric_observations, "metric_observation_id", "metric_observations"),
            (self.exploration_attempts, "attempt_id", "exploration_attempts"),
        ):
            require_sorted_unique_ids(values, attribute=attribute, field_name=name)
        self._validate_references()

    def _validate_references(self) -> None:
        roots = {record.storage_root_id for record in self.storage_roots}
        logical = {record.logical_artifact_id for record in self.logical_artifacts}
        artifacts = {record.materialization_id for record in self.artifact_locators}
        receipts = {record.receipt_id for record in self.receipts}
        objects = {record.scientific_object_id for record in self.scientific_objects}
        definitions = {record.metric_definition_id for record in self.metric_definitions}
        for locator in self.artifact_locators:
            if locator.storage_root_id not in roots or locator.logical_artifact_id not in logical:
                raise ValueError("artifact locator references unknown root/logical identity")
        if any(receipt.storage_root_id not in roots for receipt in self.receipts):
            raise ValueError("receipt references an unknown storage root")
        for record in self.scientific_objects:
            if (
                record.artifact_materialization_id is not None
                and record.artifact_materialization_id not in artifacts
            ) or (record.receipt_id is not None and record.receipt_id not in receipts):
                raise ValueError("scientific object references unknown external evidence")
        for edge in self.knowledge_edges:
            referenced = {
                edge.source_object_id,
                edge.target_object_id,
                *(() if edge.evidence_object_id is None else (edge.evidence_object_id,)),
            }
            if not referenced.issubset(objects):
                raise ValueError("knowledge edge references an unknown object")
        for observation in self.metric_observations:
            if (
                observation.metric_definition_id not in definitions
                or observation.scientific_object_id not in objects
            ):
                raise ValueError("metric observation references unknown metadata")
        for attempt in self.exploration_attempts:
            if attempt.analysis_spec_object_id not in objects or not set(
                attempt.artifact_materialization_ids
            ).issubset(artifacts):
                raise ValueError("exploration attempt references unknown metadata")

    @classmethod
    def empty(cls) -> CatalogSnapshot:
        return cls((), (), (), (), (), (), (), (), ())


class CatalogRepository(Protocol):
    def append_snapshot(self, snapshot: CatalogSnapshot) -> None: ...

    def snapshot(self) -> CatalogSnapshot: ...

    def integrity_check(self) -> tuple[str, ...]: ...

    def scientific_object(self, object_id: str) -> ScientificObjectRecord | None: ...

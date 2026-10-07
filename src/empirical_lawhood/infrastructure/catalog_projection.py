"""Strict codec for the authoritative external catalog projection artifact."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import TypeVar

from empirical_lawhood.kernel.evidence import VisibilityCeiling
from empirical_lawhood.kernel.serialization import (
    CanonicalizationError,
    CanonicalRecord,
    validate_document_shape,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.catalog import (
    ArtifactLocatorRecord,
    CatalogSnapshot,
    CatalogVerificationStatus,
    ExplorationAttemptRecord,
    KnowledgeEdgeRecord,
    LogicalArtifactRecord,
    MetricDefinitionRecord,
    MetricObservationRecord,
    ReceiptLocatorRecord,
    ScientificObjectKind,
    ScientificObjectRecord,
    StorageRootRecord,
    validate_catalog_snapshot_record_count,
)

MAX_CATALOG_PROJECTION_BYTES = 64 * 1024 * 1024

_T = TypeVar("_T")
_EnumT = TypeVar("_EnumT", bound=Enum)


def encode_catalog_snapshot(snapshot: CatalogSnapshot) -> bytes:
    """Encode a complete append-only projection in canonical JSON."""

    payload = snapshot.canonical_bytes()
    if len(payload) > MAX_CATALOG_PROJECTION_BYTES:
        raise ValueError("catalog projection exceeds its bounded artifact budget")
    return payload


def _mapping(value: object, *, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CanonicalizationError(f"{field_name} must be a string-keyed mapping")
    return value


def _record_value(
    value: object,
    record_type: type[CanonicalRecord],
    field_names: frozenset[str],
) -> Mapping[str, object]:
    return validate_document_shape(
        _mapping(value, field_name=record_type.__name__),
        expected_schema=record_type.SCHEMA,
        expected_version=record_type.VERSION,
        field_names=field_names,
    )


def _string(value: object, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise CanonicalizationError(f"{field_name} must be a string")
    return value


def _optional_string(value: object, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _string(value, field_name=field_name)


def _integer(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise CanonicalizationError(f"{field_name} must be an integer")
    return value


def _sequence(value: object, *, field_name: str) -> list[object]:
    if not isinstance(value, list):
        raise CanonicalizationError(f"{field_name} must be a JSON array")
    return value


def _tuple(
    value: object,
    decoder: Callable[[object], _T],
    *,
    field_name: str,
) -> tuple[_T, ...]:
    return tuple(decoder(item) for item in _sequence(value, field_name=field_name))


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    return tuple(
        _string(item, field_name=field_name) for item in _sequence(value, field_name=field_name)
    )


def _enum(enum_type: type[_EnumT], value: object, *, field_name: str) -> _EnumT:
    text = _string(value, field_name=field_name)
    try:
        return enum_type(text)
    except ValueError as error:
        raise CanonicalizationError(f"{field_name} has an unsupported enum value") from error


def _decimal(value: object, *, field_name: str) -> Decimal:
    mapping = _mapping(value, field_name=field_name)
    if set(mapping) != {"decimal"}:
        raise CanonicalizationError(f"{field_name} must use canonical Decimal encoding")
    text = _string(mapping["decimal"], field_name=field_name)
    try:
        return Decimal(text)
    except InvalidOperation as error:
        raise CanonicalizationError(f"{field_name} is not a Decimal") from error


def _optional_decimal(value: object, *, field_name: str) -> Decimal | None:
    if value is None:
        return None
    return _decimal(value, field_name=field_name)


def _storage_root(document: object) -> StorageRootRecord:
    value = _record_value(
        document,
        StorageRootRecord,
        frozenset(
            {
                "storage_root_id",
                "logical_name",
                "canonical_path",
                "mount_contract_schema",
                "verification_status",
            }
        ),
    )
    return StorageRootRecord(
        storage_root_id=_string(value["storage_root_id"], field_name="storage_root_id"),
        logical_name=_string(value["logical_name"], field_name="logical_name"),
        canonical_path=_string(value["canonical_path"], field_name="canonical_path"),
        mount_contract_schema=_string(
            value["mount_contract_schema"], field_name="mount_contract_schema"
        ),
        verification_status=_enum(
            CatalogVerificationStatus,
            value["verification_status"],
            field_name="verification_status",
        ),
    )


def _logical_artifact(document: object) -> LogicalArtifactRecord:
    value = _record_value(
        document,
        LogicalArtifactRecord,
        frozenset(
            {
                "logical_artifact_id",
                "content_sha256",
                "payload_schema",
                "profile",
                "media_type",
                "visibility_ceiling",
            }
        ),
    )
    return LogicalArtifactRecord(
        logical_artifact_id=_string(value["logical_artifact_id"], field_name="logical_artifact_id"),
        content_sha256=_string(value["content_sha256"], field_name="content_sha256"),
        payload_schema=_string(value["payload_schema"], field_name="payload_schema"),
        profile=_enum(ArtifactProfile, value["profile"], field_name="profile"),
        media_type=_string(value["media_type"], field_name="media_type"),
        visibility_ceiling=_enum(
            VisibilityCeiling,
            value["visibility_ceiling"],
            field_name="visibility_ceiling",
        ),
    )


def _artifact_locator(document: object) -> ArtifactLocatorRecord:
    value = _record_value(
        document,
        ArtifactLocatorRecord,
        frozenset(
            {
                "materialization_id",
                "logical_artifact_id",
                "storage_root_id",
                "relative_path",
                "physical_sha256",
                "size_bytes",
                "compression",
                "partition_selector",
                "verification_status",
            }
        ),
    )
    return ArtifactLocatorRecord(
        materialization_id=_string(value["materialization_id"], field_name="materialization_id"),
        logical_artifact_id=_string(value["logical_artifact_id"], field_name="logical_artifact_id"),
        storage_root_id=_string(value["storage_root_id"], field_name="storage_root_id"),
        relative_path=_string(value["relative_path"], field_name="relative_path"),
        physical_sha256=_string(value["physical_sha256"], field_name="physical_sha256"),
        size_bytes=_integer(value["size_bytes"], field_name="size_bytes"),
        compression=_string(value["compression"], field_name="compression"),
        partition_selector=_optional_string(
            value["partition_selector"], field_name="partition_selector"
        ),
        verification_status=_enum(
            CatalogVerificationStatus,
            value["verification_status"],
            field_name="verification_status",
        ),
    )


def _receipt(document: object) -> ReceiptLocatorRecord:
    value = _record_value(
        document,
        ReceiptLocatorRecord,
        frozenset(
            {
                "receipt_id",
                "run_id",
                "storage_root_id",
                "relative_path",
                "sha256",
                "receipt_schema",
                "verification_status",
            }
        ),
    )
    return ReceiptLocatorRecord(
        receipt_id=_string(value["receipt_id"], field_name="receipt_id"),
        run_id=_string(value["run_id"], field_name="run_id"),
        storage_root_id=_string(value["storage_root_id"], field_name="storage_root_id"),
        relative_path=_string(value["relative_path"], field_name="relative_path"),
        sha256=_string(value["sha256"], field_name="sha256"),
        receipt_schema=_string(value["receipt_schema"], field_name="receipt_schema"),
        verification_status=_enum(
            CatalogVerificationStatus,
            value["verification_status"],
            field_name="verification_status",
        ),
    )


def _scientific_object(document: object) -> ScientificObjectRecord:
    value = _record_value(
        document,
        ScientificObjectRecord,
        frozenset(
            {
                "scientific_object_id",
                "kind",
                "object_schema",
                "object_sha256",
                "visibility_ceiling",
                "categorical_status",
                "artifact_materialization_id",
                "receipt_id",
            }
        ),
    )
    return ScientificObjectRecord(
        scientific_object_id=_string(
            value["scientific_object_id"], field_name="scientific_object_id"
        ),
        kind=_enum(ScientificObjectKind, value["kind"], field_name="kind"),
        object_schema=_string(value["object_schema"], field_name="object_schema"),
        object_sha256=_string(value["object_sha256"], field_name="object_sha256"),
        visibility_ceiling=_enum(
            VisibilityCeiling,
            value["visibility_ceiling"],
            field_name="visibility_ceiling",
        ),
        categorical_status=_string(value["categorical_status"], field_name="categorical_status"),
        artifact_materialization_id=_optional_string(
            value["artifact_materialization_id"],
            field_name="artifact_materialization_id",
        ),
        receipt_id=_optional_string(value["receipt_id"], field_name="receipt_id"),
    )


def _knowledge_edge(document: object) -> KnowledgeEdgeRecord:
    value = _record_value(
        document,
        KnowledgeEdgeRecord,
        frozenset(
            {
                "edge_id",
                "relation",
                "source_object_id",
                "target_object_id",
                "evidence_object_id",
                "scope_id",
            }
        ),
    )
    return KnowledgeEdgeRecord(
        edge_id=_string(value["edge_id"], field_name="edge_id"),
        relation=_string(value["relation"], field_name="relation"),
        source_object_id=_string(value["source_object_id"], field_name="source_object_id"),
        target_object_id=_string(value["target_object_id"], field_name="target_object_id"),
        evidence_object_id=_optional_string(
            value["evidence_object_id"], field_name="evidence_object_id"
        ),
        scope_id=_string(value["scope_id"], field_name="scope_id"),
    )


def _metric_definition(document: object) -> MetricDefinitionRecord:
    value = _record_value(
        document,
        MetricDefinitionRecord,
        frozenset(
            {
                "metric_definition_id",
                "metric_name",
                "dataset_version_id",
                "relation_id",
                "denominator_gauge_id",
                "response_gauge_id",
                "horizon_id",
                "native_unit",
                "aggregation",
                "direction",
            }
        ),
    )
    return MetricDefinitionRecord(
        metric_definition_id=_string(
            value["metric_definition_id"], field_name="metric_definition_id"
        ),
        metric_name=_string(value["metric_name"], field_name="metric_name"),
        dataset_version_id=_string(value["dataset_version_id"], field_name="dataset_version_id"),
        relation_id=_string(value["relation_id"], field_name="relation_id"),
        denominator_gauge_id=_string(
            value["denominator_gauge_id"], field_name="denominator_gauge_id"
        ),
        response_gauge_id=_string(value["response_gauge_id"], field_name="response_gauge_id"),
        horizon_id=_string(value["horizon_id"], field_name="horizon_id"),
        native_unit=_string(value["native_unit"], field_name="native_unit"),
        aggregation=_string(value["aggregation"], field_name="aggregation"),
        direction=_string(value["direction"], field_name="direction"),
    )


def _metric_observation(document: object) -> MetricObservationRecord:
    value = _record_value(
        document,
        MetricObservationRecord,
        frozenset(
            {
                "metric_observation_id",
                "metric_definition_id",
                "scientific_object_id",
                "point",
                "lower",
                "upper",
                "physical_independent_unit_count",
                "numerical_view_count",
            }
        ),
    )
    return MetricObservationRecord(
        metric_observation_id=_string(
            value["metric_observation_id"], field_name="metric_observation_id"
        ),
        metric_definition_id=_string(
            value["metric_definition_id"], field_name="metric_definition_id"
        ),
        scientific_object_id=_string(
            value["scientific_object_id"], field_name="scientific_object_id"
        ),
        point=_decimal(value["point"], field_name="point"),
        lower=_optional_decimal(value["lower"], field_name="lower"),
        upper=_optional_decimal(value["upper"], field_name="upper"),
        physical_independent_unit_count=_integer(
            value["physical_independent_unit_count"],
            field_name="physical_independent_unit_count",
        ),
        numerical_view_count=_integer(
            value["numerical_view_count"], field_name="numerical_view_count"
        ),
    )


def _exploration_attempt(document: object) -> ExplorationAttemptRecord:
    value = _record_value(
        document,
        ExplorationAttemptRecord,
        frozenset(
            {
                "attempt_id",
                "analysis_spec_object_id",
                "disposition",
                "reason_codes",
                "artifact_materialization_ids",
            }
        ),
    )
    return ExplorationAttemptRecord(
        attempt_id=_string(value["attempt_id"], field_name="attempt_id"),
        analysis_spec_object_id=_string(
            value["analysis_spec_object_id"], field_name="analysis_spec_object_id"
        ),
        disposition=_string(value["disposition"], field_name="disposition"),
        reason_codes=_string_tuple(value["reason_codes"], field_name="reason_codes"),
        artifact_materialization_ids=_string_tuple(
            value["artifact_materialization_ids"],
            field_name="artifact_materialization_ids",
        ),
    )


def decode_catalog_snapshot(payload: bytes) -> CatalogSnapshot:
    """Decode only the exact current canonical projection schema, then re-hash it."""

    if not payload or len(payload) > MAX_CATALOG_PROJECTION_BYTES:
        raise CanonicalizationError("catalog projection has an invalid byte size")
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CanonicalizationError("catalog projection is not canonical UTF-8 JSON") from error
    value = _record_value(
        document,
        CatalogSnapshot,
        frozenset(
            {
                "storage_roots",
                "logical_artifacts",
                "artifact_locators",
                "receipts",
                "scientific_objects",
                "knowledge_edges",
                "metric_definitions",
                "metric_observations",
                "exploration_attempts",
            }
        ),
    )
    try:
        validate_catalog_snapshot_record_count(
            *(
                len(_sequence(value[field_name], field_name=field_name))
                for field_name in (
                    "storage_roots",
                    "logical_artifacts",
                    "artifact_locators",
                    "receipts",
                    "scientific_objects",
                    "knowledge_edges",
                    "metric_definitions",
                    "metric_observations",
                    "exploration_attempts",
                )
            )
        )
    except ValueError as error:
        raise CanonicalizationError(str(error)) from error
    snapshot = CatalogSnapshot(
        storage_roots=_tuple(value["storage_roots"], _storage_root, field_name="storage_roots"),
        logical_artifacts=_tuple(
            value["logical_artifacts"],
            _logical_artifact,
            field_name="logical_artifacts",
        ),
        artifact_locators=_tuple(
            value["artifact_locators"],
            _artifact_locator,
            field_name="artifact_locators",
        ),
        receipts=_tuple(value["receipts"], _receipt, field_name="receipts"),
        scientific_objects=_tuple(
            value["scientific_objects"],
            _scientific_object,
            field_name="scientific_objects",
        ),
        knowledge_edges=_tuple(
            value["knowledge_edges"], _knowledge_edge, field_name="knowledge_edges"
        ),
        metric_definitions=_tuple(
            value["metric_definitions"],
            _metric_definition,
            field_name="metric_definitions",
        ),
        metric_observations=_tuple(
            value["metric_observations"],
            _metric_observation,
            field_name="metric_observations",
        ),
        exploration_attempts=_tuple(
            value["exploration_attempts"],
            _exploration_attempt,
            field_name="exploration_attempts",
        ),
    )
    if snapshot.canonical_bytes() != payload:
        raise CanonicalizationError("catalog projection bytes are not canonical")
    return snapshot

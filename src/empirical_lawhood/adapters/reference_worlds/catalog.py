"""External-archive and compact-catalog projection for reference worlds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ArtifactWriteRequest,
    ArtifactWriteResult,
    ExternalRootContract,
)
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
)

from .contracts import ReferenceEvaluation, ReferenceWorldSpec


@dataclass(frozen=True, slots=True)
class ReferenceDerivedIdentity(CanonicalRecord):
    """Identity-only result used when a reference control has no full scientific object."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/reference-derived-identity'

    identity_id: str
    kind: ScientificObjectKind
    source_sha256s: tuple[str, ...]
    categorical_status: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.identity_id, field_name="identity_id")
        require_sorted_unique_strings(
            self.source_sha256s,
            field_name="source_sha256s",
            allow_empty=False,
        )
        for source_sha256 in self.source_sha256s:
            validate_sha256(source_sha256, field_name="source_sha256s")
        validate_stable_id(self.categorical_status, field_name="categorical_status")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class ReferenceWorldArchive(CanonicalRecord):
    """Synthetic row-bearing reference archive; it must live off repository disk."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reference-worlds/reference-world-archive'

    archive_id: str
    world: ReferenceWorldSpec
    evaluation: ReferenceEvaluation
    derived_identities: tuple[ReferenceDerivedIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.archive_id, field_name="archive_id")
        if self.archive_id != f"archive.{self.world.reference_id}":
            raise ValueError("reference archive identity differs from its world")
        if self.evaluation.reference_id != self.world.reference_id:
            raise ValueError("reference archive evaluation belongs to another world")
        require_sorted_unique_ids(
            self.derived_identities,
            attribute="identity_id",
            field_name="derived_identities",
        )


def _kind_slug(kind: ScientificObjectKind) -> str:
    return kind.value.lower().replace("_", "-")


def _derived(
    world: ReferenceWorldSpec,
    evaluation: ReferenceEvaluation,
    kind: ScientificObjectKind,
    status: str,
    *reason_codes: str,
    suffix: str | None = None,
) -> ReferenceDerivedIdentity:
    identity_id = f"{world.reference_id}.{_kind_slug(kind)}"
    if suffix is not None:
        identity_id = f"{identity_id}.{suffix}"
    sources = tuple(sorted({world.fingerprint(), evaluation.fingerprint()}))
    return ReferenceDerivedIdentity(
        identity_id=identity_id,
        kind=kind,
        source_sha256s=sources,
        categorical_status=status,
        reason_codes=tuple(sorted(reason_codes)),
    )


def build_reference_archive(
    world: ReferenceWorldSpec,
    evaluation: ReferenceEvaluation,
) -> ReferenceWorldArchive:
    scientific_status = evaluation.scientific_status.value.lower().replace("_", "-")
    nominated_status = "nominated" if evaluation.nominated_hypothesis_ids else "none"
    identities = [
        _derived(world, evaluation, ScientificObjectKind.MODEL_SET, "reference-control-only"),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.MODEL_RELATION,
            "not-evaluated",
            "cross-world-transport-not-executed",
        ),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.DISCREPANCY_RESULT,
            "detected",
            "nominal-only-safety-defeated",
        ),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.TRANSPORT_RESULT,
            "not-evaluated",
            "cross-world-transport-not-executed",
        ),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.STRUCTURAL_CONVERGENCE_RESULT,
            scientific_status,
        ),
        _derived(world, evaluation, ScientificObjectKind.EVIDENCE_SNAPSHOT, "outcome-visible"),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.ANOMALY_SIGNAL,
            "detected" if evaluation.nominated_hypothesis_ids else "none",
        ),
        _derived(world, evaluation, ScientificObjectKind.ANALYSIS_FAMILY, "complete"),
        _derived(world, evaluation, ScientificObjectKind.ANALYSIS_SPEC, "executed"),
        _derived(world, evaluation, ScientificObjectKind.ANALYSIS_PROPOSAL, "selected"),
        _derived(world, evaluation, ScientificObjectKind.EXPLORATION_PLAN, "completed"),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.EXPLORATORY_FINDING,
            scientific_status,
        ),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.HYPOTHESIS_SET,
            "retained" if evaluation.nominated_hypothesis_ids else "empty",
        ),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.PROSPECTIVE_NOMINATION,
            nominated_status,
        ),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.EXPERIMENT_PROPOSAL,
            "not-evaluated",
            "prospective-experiment-not-frozen",
        ),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.AUTHORIZATION_RECORD,
            "reference-only",
        ),
        _derived(
            world,
            evaluation,
            ScientificObjectKind.DECISION_RECORD,
            evaluation.controller_decision.value.lower().replace("_", "-"),
        ),
    ]
    for hypothesis_id in evaluation.nominated_hypothesis_ids:
        identities.append(
            _derived(
                world,
                evaluation,
                ScientificObjectKind.HYPOTHESIS,
                (
                    "supported"
                    if hypothesis_id in evaluation.supported_hypothesis_ids
                    else "nominated-not-confirmed"
                ),
                suffix=hypothesis_id,
            )
        )
    return ReferenceWorldArchive(
        archive_id=f"archive.{world.reference_id}",
        world=world,
        evaluation=evaluation,
        derived_identities=tuple(sorted(identities, key=lambda item: item.identity_id)),
    )


def reference_archive_requests(
    worlds: tuple[ReferenceWorldSpec, ...],
    evaluations: tuple[ReferenceEvaluation, ...],
) -> tuple[ArtifactWriteRequest, ...]:
    if len(worlds) != len(evaluations):
        raise ValueError("reference world/evaluation cardinality differs")
    requests: list[ArtifactWriteRequest] = []
    for world, evaluation in zip(worlds, evaluations, strict=True):
        archive = build_reference_archive(world, evaluation)
        parents = tuple(
            sorted(
                (
                    ArtifactLineageParent(
                        identity=ObjectIdentity.from_record(world.reference_id, world),
                        visibility_ceiling=VisibilityCeiling.PRIVILEGED_TRUTH,
                        outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
                    ),
                    ArtifactLineageParent(
                        identity=ObjectIdentity.from_record(
                            evaluation.evaluation_id,
                            evaluation,
                        ),
                        visibility_ceiling=VisibilityCeiling.PRIVILEGED_TRUTH,
                        outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
                    ),
                ),
                key=lambda parent: (
                    parent.identity.object_id,
                    parent.identity.object_schema,
                ),
            )
        )
        requests.append(
            ArtifactWriteRequest(
                logical_artifact_id=f"reference-world.{world.reference_id}",
                relative_path=(
                    f"runs/reference-world-catalog/archives/{world.reference_id}.json"
                ),
                payload_schema=archive.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                publication_scope_id="reference-world-archives",
                publication_scope_relative_root=(
                    "runs/reference-world-catalog/archives"
                ),
                payload=archive.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PRIVILEGED_TRUTH,
                parent_visibility_ceilings=tuple(parent.visibility_ceiling for parent in parents),
                outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
                lineage_parents=parents,
            )
        )
    return tuple(requests)


def _object_id(kind: ScientificObjectKind, fingerprint: str) -> str:
    return f"object.{_kind_slug(kind)}.{fingerprint[:24]}"


def reference_catalog_snapshot(
    *,
    root_contract: ExternalRootContract,
    worlds: tuple[ReferenceWorldSpec, ...],
    evaluations: tuple[ReferenceEvaluation, ...],
    write_results: tuple[ArtifactWriteResult, ...],
    reference_archive_receipt: ReceiptLocatorRecord,
) -> CatalogSnapshot:
    """Project complete reference identities into compact, row-free catalog records."""

    if type(reference_archive_receipt) is not ReceiptLocatorRecord:
        raise TypeError("reference archive receipt must use its exact locator record")
    if reference_archive_receipt.storage_root_id != root_contract.storage_root_id:
        raise ValueError("reference archive receipt binds another storage root")
    if not len(worlds) == len(evaluations) == len(write_results):
        raise ValueError("reference catalog inputs have different cardinalities")
    storage_root = StorageRootRecord(
        storage_root_id=root_contract.storage_root_id,
        logical_name=root_contract.logical_name,
        canonical_path=root_contract.canonical_path,
        mount_contract_schema=root_contract.mount_contract_schema,
        verification_status=CatalogVerificationStatus.VERIFIED,
    )
    receipt = reference_archive_receipt
    logical_artifacts: list[LogicalArtifactRecord] = []
    artifact_locators: list[ArtifactLocatorRecord] = []
    objects_by_identity: dict[tuple[ScientificObjectKind, str], ScientificObjectRecord] = {}
    edges: list[KnowledgeEdgeRecord] = []
    metric_definitions: list[MetricDefinitionRecord] = []
    metric_observations: list[MetricObservationRecord] = []
    attempts: list[ExplorationAttemptRecord] = []

    def add_object(
        *,
        kind: ScientificObjectKind,
        record: CanonicalRecord,
        status: str,
        materialization_id: str,
    ) -> str:
        fingerprint = record.fingerprint()
        object_id = _object_id(kind, fingerprint)
        key = (kind, fingerprint)
        objects_by_identity.setdefault(
            key,
            ScientificObjectRecord(
                scientific_object_id=object_id,
                kind=kind,
                object_schema=record.SCHEMA,
                object_sha256=fingerprint,
                visibility_ceiling=VisibilityCeiling.PRIVILEGED_TRUTH,
                categorical_status=status,
                artifact_materialization_id=materialization_id,
                receipt_id=receipt.receipt_id,
            ),
        )
        return object_id

    for world, evaluation, result in zip(
        worlds,
        evaluations,
        write_results,
        strict=True,
    ):
        expected_logical_id = f"reference-world.{world.reference_id}"
        if result.logical.logical_artifact_id != expected_logical_id:
            raise ValueError("reference write result order/identity differs")
        logical_artifacts.append(
            LogicalArtifactRecord(
                logical_artifact_id=result.logical.logical_artifact_id,
                content_sha256=result.logical.content_sha256,
                payload_schema=result.logical.payload_schema,
                profile=result.logical.profile,
                media_type=result.logical.media_type,
                visibility_ceiling=result.logical.visibility_ceiling,
            )
        )
        materialization = result.materialization
        artifact_locators.append(
            ArtifactLocatorRecord(
                materialization_id=materialization.materialization_id,
                logical_artifact_id=materialization.logical_artifact_id,
                storage_root_id=materialization.storage_root_id,
                relative_path=materialization.relative_path,
                physical_sha256=materialization.physical_sha256,
                size_bytes=materialization.size_bytes,
                compression=materialization.compression,
                partition_selector=materialization.partition_selector,
                verification_status=CatalogVerificationStatus.VERIFIED,
            )
        )
        archive = build_reference_archive(world, evaluation)
        world_object_id = add_object(
            kind=ScientificObjectKind.WORLD_SPEC,
            record=world.system.world,
            status="truth-known",
            materialization_id=materialization.materialization_id,
        )
        system_object_id = add_object(
            kind=ScientificObjectKind.SYSTEM_SPEC,
            record=world.system,
            status=world.system.readiness.value.lower().replace("_", "-"),
            materialization_id=materialization.materialization_id,
        )
        add_object(
            kind=ScientificObjectKind.AUTHORITY_POLICY,
            record=world.system.authority_policy,
            status="reference-only",
            materialization_id=materialization.materialization_id,
        )
        for envelope in world.system.computability_envelopes:
            add_object(
                kind=ScientificObjectKind.COMPUTABILITY_ENVELOPE,
                record=envelope,
                status=envelope.readiness.value.lower().replace("_", "-"),
                materialization_id=materialization.materialization_id,
            )
        for view in world.system.numerical_views:
            view_object_id = add_object(
                kind=ScientificObjectKind.NUMERICAL_VIEW,
                record=view,
                status="nested-numerical-view",
                materialization_id=materialization.materialization_id,
            )
            edges.append(
                KnowledgeEdgeRecord(
                    edge_id=f"edge.{world.reference_id}.view.{view.view_id}",
                    relation="contains",
                    source_object_id=system_object_id,
                    target_object_id=view_object_id,
                    evidence_object_id=world_object_id,
                    scope_id=world.reference_id,
                )
            )
        derived_ids: dict[ScientificObjectKind, list[str]] = {}
        for identity in archive.derived_identities:
            object_id = add_object(
                kind=identity.kind,
                record=identity,
                status=identity.categorical_status,
                materialization_id=materialization.materialization_id,
            )
            derived_ids.setdefault(identity.kind, []).append(object_id)
        convergence_object_id = derived_ids[ScientificObjectKind.STRUCTURAL_CONVERGENCE_RESULT][0]
        edges.extend(
            (
                KnowledgeEdgeRecord(
                    edge_id=f"edge.{world.reference_id}.system-world",
                    relation="instantiates",
                    source_object_id=system_object_id,
                    target_object_id=world_object_id,
                    evidence_object_id=convergence_object_id,
                    scope_id=world.reference_id,
                ),
                KnowledgeEdgeRecord(
                    edge_id=f"edge.{world.reference_id}.evaluation",
                    relation="evaluates",
                    source_object_id=convergence_object_id,
                    target_object_id=world_object_id,
                    evidence_object_id=convergence_object_id,
                    scope_id=world.reference_id,
                ),
            )
        )
        physical_units = len({case.independent_unit_id for case in world.cases})
        for metric in evaluation.metrics:
            definition_id = f"metric.{world.reference_id}.{metric.value_id}"
            metric_definitions.append(
                MetricDefinitionRecord(
                    metric_definition_id=definition_id,
                    metric_name=metric.value_id,
                    dataset_version_id=f"dataset.{world.reference_id}",
                    relation_id=world.system.relation.relation_id,
                    denominator_gauge_id="gauge.prepared-denominator",
                    response_gauge_id=world.cases[0].gauge_id,
                    horizon_id=world.system.relation.horizon.horizon_id,
                    native_unit=metric.unit,
                    aggregation="physical-unit-reference",
                    direction="descriptive",
                )
            )
            metric_observations.append(
                MetricObservationRecord(
                    metric_observation_id=f"observation.{world.reference_id}.{metric.value_id}",
                    metric_definition_id=definition_id,
                    scientific_object_id=convergence_object_id,
                    point=metric.value,
                    lower=None,
                    upper=None,
                    physical_independent_unit_count=physical_units,
                    numerical_view_count=len(world.system.numerical_views),
                )
            )
        analysis_spec_object_id = derived_ids[ScientificObjectKind.ANALYSIS_SPEC][0]
        is_null_family = world.reference_id == "w15-null-search-family"
        for analysis_id in world.controls.registered_analysis_ids:
            attempts.append(
                ExplorationAttemptRecord(
                    attempt_id=f"attempt.{analysis_id}",
                    analysis_spec_object_id=analysis_spec_object_id,
                    disposition="null" if is_null_family else "completed",
                    reason_codes=("no-preferred-hypothesis",) if is_null_family else (),
                    artifact_materialization_ids=(materialization.materialization_id,),
                )
            )

    return CatalogSnapshot(
        storage_roots=(storage_root,),
        logical_artifacts=tuple(
            sorted(logical_artifacts, key=lambda record: record.logical_artifact_id)
        ),
        artifact_locators=tuple(
            sorted(artifact_locators, key=lambda record: record.materialization_id)
        ),
        receipts=(receipt,),
        scientific_objects=tuple(
            sorted(
                objects_by_identity.values(),
                key=lambda record: record.scientific_object_id,
            )
        ),
        knowledge_edges=tuple(sorted(edges, key=lambda record: record.edge_id)),
        metric_definitions=tuple(
            sorted(metric_definitions, key=lambda record: record.metric_definition_id)
        ),
        metric_observations=tuple(
            sorted(metric_observations, key=lambda record: record.metric_observation_id)
        ),
        exploration_attempts=tuple(sorted(attempts, key=lambda record: record.attempt_id)),
    )

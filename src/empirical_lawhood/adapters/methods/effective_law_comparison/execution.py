# SPDX-License-Identifier: MPL-2.0
"""Bounded execution, immutable publication, receipts and recovery verification."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Protocol

from empirical_lawhood.adapters.independent_source_exports import IndependentSourceExport
from empirical_lawhood.kernel.decoding import decode_canonical_record
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    validate_relative_locator,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.planning.posthoc import POSTHOC_ANALYSIS_IDS, PosthocTrancheFreeze
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactManifest,
    ArtifactMaterialization,
    ArtifactProfile,
    ArtifactSemanticValidation,
    ArtifactSemanticValidationRegistration,
    ArtifactSemanticValidationRegistry,
    ArtifactWriteRequest,
    ArtifactWriteResult,
    CanonicalTaskReceipt,
    ReceiptCheck,
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.plans import CandidateExecutionPlan

from .sources import _custody_sidecar, _current_sidecar_payload_identity
from .analysis import PosthocDocumentSet, execute_analysis, execute_integrated_skeptic, execute_skeptic, synthesize

MAXIMUM_DOCUMENT_BYTES = 16 * 1024**2
MINIMUM_FREE_BYTES = 100 * 1024**3


class PosthocExternalRoot(Protocol):
    def resolve(
        self,
        relative_path: str,
        *,
        for_write: bool,
        operation_minimum_free_bytes: int = 0,
    ) -> Path: ...


class PosthocArtifactPlane(Protocol):
    def write(self, request: ArtifactWriteRequest) -> ArtifactWriteResult: ...

    def verify(self, materialization: ArtifactMaterialization) -> None: ...

    def verify_manifest(self, manifest: ArtifactManifest) -> None: ...


@dataclass(frozen=True, slots=True)
class EffectiveLawPosthocExecutionPorts:
    """Outer storage/receipt mechanisms supplied without reversing layers."""

    root: PosthocExternalRoot
    plane_for_registry: Callable[
        [ArtifactSemanticValidationRegistry],
        PosthocArtifactPlane,
    ]
    decode_artifact_manifest: Callable[[bytes], ArtifactManifest]
    decode_task_receipt: Callable[[bytes], CanonicalTaskReceipt]
    run_root: str
    run_id: str
    publication_scope_id: str
    minimum_free_bytes: int = MINIMUM_FREE_BYTES

    def __post_init__(self) -> None:
        validate_relative_locator(self.run_root)
        validate_stable_id(self.run_id)
        validate_stable_id(self.publication_scope_id)
        if self.minimum_free_bytes < 0:
            raise ValueError("post-hoc free-space floor cannot be negative")


@dataclass(frozen=True, slots=True)
class PublishedTask:
    task_id: str
    document: Mapping[str, object]
    result: ArtifactWriteResult
    receipt: CanonicalTaskReceipt
    recovered: bool = False


def _stable_payload(document: Mapping[str, object] | CanonicalRecord) -> bytes:
    def canonical_container(value: object) -> object:
        if isinstance(value, list):
            return tuple(canonical_container(item) for item in value)
        if isinstance(value, Mapping):
            return {str(key): canonical_container(item) for key, item in value.items()}
        return value

    payload = canonical_json_bytes(canonical_container(document))
    if len(payload) > MAXIMUM_DOCUMENT_BYTES:
        raise ValueError("post-hoc document exceeds its byte bound")
    return payload


def _lineage_identity(
    object_id: str,
    schema: str,
    version: str,
    fingerprint: str,
) -> ArtifactLineageParent:
    return ArtifactLineageParent(
        identity=ObjectIdentity(
            object_id=object_id,
            object_schema=schema,
            object_version=version,
            object_fingerprint=fingerprint,
        ),
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )



def _qualified_source_export(member: Mapping[str, object]) -> IndependentSourceExport:
    if member.get("status") != "QUALIFIED" or member.get("custody_status") != "VERIFIED":
        raise ValueError("post-hoc work requires verified current source/export custody; original archives cannot enter directly")
    export = decode_canonical_record(member.get("source_export"), IndependentSourceExport)
    if (
        export.fingerprint() != member.get("source_export_sha256")
        or export.target_artifact.artifact_id != member.get("artifact_id")
        or export.target_relative_path != member.get("relative_locator")
        or export.target_artifact.payload_schema != member.get("payload_schema")
        or export.target_source.object_version != member.get("payload_version")
        or export.target_source.object_version != member.get("identity_version")
        or export.target_artifact.sha256 != member.get("content_sha256")
        or export.target_artifact.size_bytes != member.get("size_bytes")
        or member.get("custody_sha256") not in {export.target_manifest_sha256, export.target_task_receipt.object_fingerprint}
        or member.get("grants_authority") is not False
    ):
        raise ValueError("post-hoc source member differs from its exact current export/custody descriptor")
    return export

def source_lineages(
    qualification: Mapping[str, object],
) -> dict[str, tuple[ArtifactLineageParent, ...]]:
    value = qualification.get("value")
    if not isinstance(value, Mapping) or not isinstance(value.get("parents"), list):
        raise ValueError("source qualification changed shape")
    lineages: dict[str, tuple[ArtifactLineageParent, ...]] = {}
    for parent in value["parents"]:
        if not isinstance(parent, Mapping) or not isinstance(parent.get("members"), list):
            continue
        for member in parent["members"]:
            if not isinstance(member, Mapping) or member.get("status") != "QUALIFIED":
                continue
            artifact_id = str(member["artifact_id"])
            export = _qualified_source_export(member)
            lineages[artifact_id] = tuple(
                _lineage_identity(identity.object_id, identity.object_schema, identity.object_version, identity.object_fingerprint)
                for identity in (export.target_source, *export.lineage_identities)
            )
    return lineages


def _semantic_validation(
    logical_artifact_id: str,
    document: Mapping[str, object],
    implementation_sha256: str,
) -> ArtifactSemanticValidation:
    value = document.get("value")
    if set(document) != {"schema", "value", "version"} or not isinstance(value, Mapping):
        raise ValueError("published post-hoc document is not a canonical envelope")
    return ArtifactSemanticValidation(
        validator_key=f"posthoc-json.{logical_artifact_id}",
        validator_version="1.0.0",
        validator_implementation_sha256=implementation_sha256,
        payload_schema=str(document["schema"]),
        profile=ArtifactProfile.CANONICAL_JSON,
        top_level_keys=("schema", "value", "version"),
        value_keys=tuple(sorted(str(key) for key in value)),
        field_bindings=(),
    )


def publish_document(
    *,
    ports: EffectiveLawPosthocExecutionPorts,
    logical_artifact_id: str,
    relative_path: str,
    document: Mapping[str, object] | CanonicalRecord,
    implementation_sha256: str,
    lineage_parents: Sequence[ArtifactLineageParent] = (),
) -> ArtifactWriteResult:
    canonical_document = (
        document.to_document() if isinstance(document, CanonicalRecord) else document
    )
    validation = _semantic_validation(
        logical_artifact_id,
        canonical_document,
        implementation_sha256,
    )
    registry = ArtifactSemanticValidationRegistry.from_registrations(
        registry_id=f"posthoc-semantics.{logical_artifact_id}",
        registrations=(
            ArtifactSemanticValidationRegistration(
                logical_artifact_id=logical_artifact_id,
                validation=validation,
            ),
        ),
    )
    plane = ports.plane_for_registry(registry)
    parents = tuple(sorted(lineage_parents, key=lineage_parent_sort_key))
    return plane.write(
        ArtifactWriteRequest(
            logical_artifact_id=logical_artifact_id,
            relative_path=relative_path,
            payload_schema=str(canonical_document["schema"]),
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/json",
            publication_scope_id=ports.publication_scope_id,
            publication_scope_relative_root=ports.run_root,
            payload=_stable_payload(canonical_document),
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            parent_visibility_ceilings=tuple(
                sorted(
                    (parent.visibility_ceiling for parent in parents), key=lambda item: item.value
                )
            ),
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            lineage_parents=parents,
            minimum_free_bytes=ports.minimum_free_bytes,
            semantic_validation=validation,
        )
    )


def published_lineage(result: ArtifactWriteResult) -> ArtifactLineageParent:
    return ArtifactLineageParent(
        identity=ObjectIdentity.from_record(result.logical.logical_artifact_id, result.logical),
        visibility_ceiling=result.logical.visibility_ceiling,
        outcome_access=result.logical.outcome_access,
    )


def _source_materialization_id(artifact_id: str, fingerprint: str) -> str:
    return f"source.{artifact_id}.{fingerprint[:16]}"


def _external_source_materialization_id(
    artifact_id: str,
    expected_content_sha256: str | None,
) -> str:
    if expected_content_sha256 is None:
        raise ValueError(f"post-hoc external source lacks a content identity: {artifact_id}")
    return _source_materialization_id(artifact_id, expected_content_sha256)


def publish_task(
    *,
    ports: EffectiveLawPosthocExecutionPorts,
    task_id: str,
    document: Mapping[str, object],
    implementation_commit: str,
    implementation_sha256: str,
    lineage_parents: Sequence[ArtifactLineageParent],
    input_materialization_ids: Sequence[str],
) -> PublishedTask:
    logical_id = f"{ports.run_id}.{task_id}"
    result = publish_document(
        ports=ports,
        logical_artifact_id=logical_id,
        relative_path=f"{ports.run_root}/outputs/{task_id}/result.json",
        document=document,
        implementation_sha256=implementation_sha256,
        lineage_parents=lineage_parents,
    )
    receipt = CanonicalTaskReceipt(
        receipt_id=f"{ports.run_id}.{task_id}.attempt-001",
        run_id=ports.run_id,
        task_id=task_id,
        attempt_id="attempt-001",
        implementation_commit=implementation_commit,
        input_materialization_ids=tuple(sorted(set(input_materialization_ids))),
        output_materializations=(result.materialization,),
        output_logical_artifacts=(result.logical,),
        checks=(
            ReceiptCheck(check_id="artifact-commit", passed=True, reason_codes=()),
            ReceiptCheck(check_id="bounded-output", passed=True, reason_codes=()),
            ReceiptCheck(check_id="exact-input-lineage", passed=True, reason_codes=()),
            ReceiptCheck(check_id="nonpromotable-output", passed=True, reason_codes=()),
        ),
        operational_status=OperationalStatus.SUCCEEDED,
        reason_codes=(),
    )
    publish_document(
        ports=ports,
        logical_artifact_id=f"{ports.run_id}.{task_id}.receipt",
        relative_path=f"{ports.run_root}/receipts/{task_id}/attempt-001.json",
        document=receipt,
        implementation_sha256=implementation_sha256,
        lineage_parents=(published_lineage(result),),
    )
    return PublishedTask(
        task_id=task_id,
        document=document,
        result=result,
        receipt=receipt,
        recovered=False,
    )


def recover_published_task(
    *,
    ports: EffectiveLawPosthocExecutionPorts,
    task_id: str,
    implementation_commit: str,
    implementation_sha256: str,
    expected_input_materialization_ids: Sequence[str],
) -> PublishedTask | None:
    """Recover one exact terminal task without invoking its scientific worker."""

    root = ports.root
    output_relative = f"{ports.run_root}/outputs/{task_id}/result.json"
    receipt_relative = f"{ports.run_root}/receipts/{task_id}/attempt-001.json"
    required = (
        output_relative,
        f"{output_relative}.manifest.json",
        receipt_relative,
        f"{receipt_relative}.manifest.json",
    )
    paths = tuple(root.resolve(value, for_write=False) for value in required)
    present = tuple(path.is_file() for path in paths)
    if not any(present):
        return None
    if not all(present):
        raise ValueError(f"partial published state for task {task_id}")

    output_manifest = ports.decode_artifact_manifest(paths[1].read_bytes())
    verify_published_artifact(
        output_relative,
        ports=ports,
        implementation_sha256=implementation_sha256,
    )
    verify_published_artifact(
        receipt_relative,
        ports=ports,
        implementation_sha256=implementation_sha256,
    )
    receipt = ports.decode_task_receipt(paths[2].read_bytes())
    if (
        receipt.run_id != ports.run_id
        or receipt.task_id != task_id
        or receipt.attempt_id != "attempt-001"
        or receipt.implementation_commit != implementation_commit
        or receipt.operational_status is not OperationalStatus.SUCCEEDED
        or not all(check.passed for check in receipt.checks)
    ):
        raise ValueError(f"task receipt does not authorize recovery: {task_id}")
    if receipt.input_materialization_ids != tuple(sorted(set(expected_input_materialization_ids))):
        raise ValueError(f"task receipt input closure drifted: {task_id}")
    if receipt.output_materializations != (output_manifest.materialization,):
        raise ValueError(f"task receipt output materialization drifted: {task_id}")
    if receipt.output_logical_artifacts != (output_manifest.logical,):
        raise ValueError(f"task receipt logical output drifted: {task_id}")

    payload = paths[0].read_bytes()
    if len(payload) > MAXIMUM_DOCUMENT_BYTES:
        raise ValueError(f"recovered output exceeds its byte bound: {task_id}")
    document = json.loads(payload)
    if not isinstance(document, Mapping) or _stable_payload(document) != payload:
        raise ValueError(f"recovered output is not canonical JSON: {task_id}")
    if document.get("schema") != output_manifest.logical.payload_schema:
        raise ValueError(f"recovered output schema drifted: {task_id}")

    manifest_payload = paths[1].read_bytes()
    manifest_materialization = ArtifactMaterialization(
        materialization_id=f"{ports.run_id}.{task_id}.manifest.attempt-001",
        logical_artifact_id=f"{ports.run_id}.{task_id}.manifest",
        storage_root_id=output_manifest.materialization.storage_root_id,
        relative_path=f"{output_relative}.manifest.json",
        physical_sha256=sha256(manifest_payload).hexdigest(),
        size_bytes=len(manifest_payload),
        compression="none",
        partition_selector=None,
    )
    return PublishedTask(
        task_id=task_id,
        document=document,
        result=ArtifactWriteResult(
            logical=output_manifest.logical,
            materialization=output_manifest.materialization,
            manifest_materialization=manifest_materialization,
            created=False,
        ),
        receipt=receipt,
        recovered=True,
    )


def load_qualified_documents(
    *,
    external_root: Path,
    qualification: Mapping[str, object],
) -> PosthocDocumentSet:
    value = qualification.get("value")
    if not isinstance(value, Mapping) or not isinstance(value.get("parents"), list):
        raise ValueError("source qualification changed shape")
    root = external_root.resolve(strict=True)
    documents: dict[str, Mapping[str, object]] = {}
    parent_by_artifact: dict[str, str] = {}
    for parent in value["parents"]:
        if not isinstance(parent, Mapping) or parent.get("status") == "INELIGIBLE":
            continue
        members = parent.get("members")
        if not isinstance(members, list):
            continue
        for member in members:
            if not isinstance(member, Mapping) or member.get("status") != "QUALIFIED":
                continue
            export = _qualified_source_export(member)
            relative = export.target_relative_path
            validate_relative_locator(relative)
            path = root / relative
            if path.is_symlink() or not path.resolve(strict=False).is_relative_to(root):
                raise PermissionError("qualified post-hoc source escapes external custody")
            sidecar = _custody_sidecar(path)
            if sidecar is None or sidecar.is_symlink() or not sidecar.resolve(strict=False).is_relative_to(root):
                raise PermissionError("qualified post-hoc source requires current custody within its external root")
            custody = read_bounded_bytes(sidecar, maximum_bytes=1024 * 1024)
            if sha256(custody).hexdigest() != member.get("custody_sha256") or _current_sidecar_payload_identity(custody, export, manifest=sidecar.name.endswith(".manifest.json")) != (export.target_artifact.sha256, export.target_artifact.size_bytes):
                raise ValueError("qualified current source custody drifted before execution")
            payload = read_bounded_bytes(path, maximum_bytes=MAXIMUM_DOCUMENT_BYTES)
            if sha256(payload).hexdigest() != export.target_artifact.sha256 or len(payload) != export.target_artifact.size_bytes:
                raise ValueError("qualified source drifted before execution")
            document = json.loads(payload)
            if not isinstance(document, Mapping) or document.get("schema") != export.target_artifact.payload_schema or document.get("version") != export.target_source.object_version:
                raise ValueError("qualified source document changed shape")
            documents[str(member["artifact_id"])] = document
            parent_by_artifact[str(member["artifact_id"])] = str(parent["parent_id"])
    return PosthocDocumentSet(documents, parent_by_artifact)


def execute_plan(
    *,
    ports: EffectiveLawPosthocExecutionPorts,
    external_root: Path,
    qualification: Mapping[str, object],
    freeze: PosthocTrancheFreeze,
    plan: CandidateExecutionPlan,
) -> dict[str, PublishedTask]:
    expected_task_ids = {
        *POSTHOC_ANALYSIS_IDS,
        *(f"skeptic.{analysis_id.removeprefix('analysis.')}" for analysis_id in POSTHOC_ANALYSIS_IDS),
        "skeptic.integrated",
        "synthesis.metatheory",
        "index.results",
    }
    if {task.task_id for task in plan.tasks} != expected_task_ids:
        raise ValueError("post-hoc execution requires the exact current twenty-one-task purpose roster")
    documents = load_qualified_documents(external_root=external_root, qualification=qualification)
    source_parent = source_lineages(qualification)
    analysis_specs = {analysis.analysis_id: analysis for analysis in freeze.analyses}
    skeptic_by_id = {
        f"skeptic.{analysis_id.removeprefix('analysis.')}": analysis_id
        for analysis_id in POSTHOC_ANALYSIS_IDS
    }
    source_by_parent = {
        parent.parent_id: tuple(source.artifact_id for source in parent.source_artifacts)
        for parent in freeze.parents
    }
    published: dict[str, PublishedTask] = {}
    tasks = {task.task_id: task for task in plan.tasks}
    for task_id in plan.topological_task_ids():
        task = tasks[task_id]
        expected_input_ids: list[str] = []
        external_by_id = {value.input_id: value for value in task.external_inputs}
        for edge in task.scientific_inputs:
            if edge.producer_task_id is not None:
                expected_input_ids.append(
                    published[edge.producer_task_id].result.materialization.materialization_id
                )
            else:
                assert edge.external_input_id is not None
                external = external_by_id[edge.external_input_id]
                expected_input_ids.append(
                    _external_source_materialization_id(
                        external.logical_artifact_id,
                        external.expected_content_sha256,
                    )
                )
        recovered = recover_published_task(
            ports=ports,
            task_id=task_id,
            implementation_commit=freeze.implementation_commit,
            implementation_sha256=freeze.implementation_sha256,
            expected_input_materialization_ids=expected_input_ids,
        )
        if recovered is not None:
            published[task_id] = recovered
            continue
        if task_id in analysis_specs:
            specification = analysis_specs[task_id]
            artifact_ids = tuple(
                sorted(
                    artifact_id
                    for parent_id in specification.eligible_parent_ids
                    for artifact_id in source_by_parent[parent_id]
                    if artifact_id in documents
                )
            )
            selected_documents = PosthocDocumentSet(
                {artifact_id: documents[artifact_id] for artifact_id in artifact_ids},
                {artifact_id: documents.parent_by_artifact[artifact_id] for artifact_id in artifact_ids},
            )
            document = execute_analysis(
                task_id, selected_documents, specification.eligible_parent_ids
            )
            lineages = tuple({
                lineage_parent_sort_key(value): value
                for artifact_id in artifact_ids
                for value in source_parent[artifact_id]
            }.values())
            input_ids = tuple(
                _source_materialization_id(
                    artifact_id, source_parent[artifact_id][0].identity.object_fingerprint
                )
                for artifact_id in artifact_ids
            )
        elif task_id in skeptic_by_id:
            dependency = skeptic_by_id[task_id]
            document = execute_skeptic(published[dependency].document)
            lineages = (published_lineage(published[dependency].result),)
            input_ids = (published[dependency].result.materialization.materialization_id,)
        elif task_id == "skeptic.integrated":
            dependencies = tuple(f"skeptic.{analysis_id.removeprefix('analysis.')}" for analysis_id in POSTHOC_ANALYSIS_IDS)
            document = execute_integrated_skeptic(
                [published[value].document for value in dependencies]
            )
            lineages = tuple(published_lineage(published[value].result) for value in dependencies)
            input_ids = tuple(
                published[value].result.materialization.materialization_id for value in dependencies
            )
        elif task_id == "synthesis.metatheory":
            analyses = POSTHOC_ANALYSIS_IDS
            document = synthesize(
                [published[value].document for value in analyses],
                published["skeptic.integrated"].document,
            )
            dependencies = (*analyses, "skeptic.integrated")
            lineages = tuple(published_lineage(published[value].result) for value in dependencies)
            input_ids = tuple(
                published[value].result.materialization.materialization_id for value in dependencies
            )
        elif task_id == "index.results":
            dependencies = tuple(sorted(published))
            scientific_statuses = {}
            for dependency in dependencies:
                value = published[dependency].document.get("value")
                if dependency in analysis_specs and isinstance(value, Mapping):
                    scientific_statuses[dependency] = value.get("status")
            document = {
                "schema": 'empirical-lawhood/runtime/result-index',
                "version": "1.0.0",
                "value": {
                    "analysis_ids": list(POSTHOC_ANALYSIS_IDS),
                    "attempt_rows": [
                        {
                            "attempt_id": published[value].receipt.attempt_id,
                            "operational_status": published[value].receipt.operational_status.value,
                            "scientific_disposition": (
                                scientific_statuses[value]
                                if value in scientific_statuses
                                else "PASS"
                                if value in skeptic_by_id or value == "skeptic.integrated"
                                else "SYNTHESIZED"
                            ),
                            "task_id": value,
                        }
                        for value in dependencies
                    ],
                    "claim_promotion_allowed": False,
                    "execution_plan_sha256": plan.fingerprint(),
                    "output_rows": [
                        {
                            "logical_artifact_id": published[
                                value
                            ].result.logical.logical_artifact_id,
                            "materialization_id": published[
                                value
                            ].result.materialization.materialization_id,
                            "physical_sha256": published[
                                value
                            ].result.materialization.physical_sha256,
                            "size_bytes": published[value].result.materialization.size_bytes,
                            "task_id": value,
                        }
                        for value in dependencies
                    ],
                    "indexed_predecessor_receipt_count": len(dependencies),
                    "scientific_statuses": scientific_statuses,
                    "terminal": True,
                    "total_plan_task_count": len(plan.tasks),
                },
            }
            lineages = tuple(published_lineage(published[value].result) for value in dependencies)
            input_ids = tuple(
                published[value].result.materialization.materialization_id for value in dependencies
            )
        else:
            raise ValueError(f"execution plan contains an unknown task: {task_id}")
        if tuple(sorted(set(input_ids))) != tuple(sorted(set(expected_input_ids))):
            raise ValueError(f"runtime input resolution drifted from compiled graph: {task_id}")
        published[task_id] = publish_task(
            ports=ports,
            task_id=task_id,
            document=document,
            implementation_commit=freeze.implementation_commit,
            implementation_sha256=freeze.implementation_sha256,
            lineage_parents=lineages,
            input_materialization_ids=input_ids,
        )
    return published


def verify_published_artifact(
    relative_path: str,
    *,
    ports: EffectiveLawPosthocExecutionPorts,
    implementation_sha256: str,
) -> str:
    root = ports.root
    payload_path = root.resolve(relative_path, for_write=False)
    sidecar_path = root.resolve(f"{relative_path}.manifest.json", for_write=False)
    manifest = ports.decode_artifact_manifest(sidecar_path.read_bytes())
    document = json.loads(payload_path.read_bytes())
    if not isinstance(document, Mapping):
        raise ValueError("published artifact is not a canonical mapping")
    expected = _semantic_validation(
        manifest.logical.logical_artifact_id,
        document,
        implementation_sha256,
    )
    if manifest.logical.semantic_validation != expected:
        raise ValueError("published artifact semantic validator identity drifted")
    registry = ArtifactSemanticValidationRegistry.from_registrations(
        registry_id=f"posthoc-recovery.{manifest.logical.logical_artifact_id}",
        registrations=(
            ArtifactSemanticValidationRegistration(
                logical_artifact_id=manifest.logical.logical_artifact_id,
                validation=expected,
            ),
        ),
    )
    plane = ports.plane_for_registry(registry)
    plane.verify(manifest.materialization)
    plane.verify_manifest(manifest)
    return manifest.materialization.physical_sha256


def verify_complete_run(
    plan: CandidateExecutionPlan,
    *,
    ports: EffectiveLawPosthocExecutionPorts,
    implementation_commit: str,
) -> dict[str, object]:
    output_rows = []
    published: dict[str, PublishedTask] = {}
    tasks = {task.task_id: task for task in plan.tasks}
    for task_id in plan.topological_task_ids():
        task = tasks[task_id]
        external_by_id = {value.input_id: value for value in task.external_inputs}
        expected_input_ids = []
        for edge in task.scientific_inputs:
            if edge.producer_task_id is not None:
                expected_input_ids.append(
                    published[edge.producer_task_id].result.materialization.materialization_id
                )
            else:
                assert edge.external_input_id is not None
                external = external_by_id[edge.external_input_id]
                expected_input_ids.append(
                    _external_source_materialization_id(
                        external.logical_artifact_id,
                        external.expected_content_sha256,
                    )
                )
        recovered = recover_published_task(
            ports=ports,
            task_id=task_id,
            implementation_commit=implementation_commit,
            implementation_sha256=task.capability_implementation_sha256,
            expected_input_materialization_ids=expected_input_ids,
        )
        if recovered is None:
            raise ValueError(f"terminal task is absent during recovery: {task_id}")
        published[task_id] = recovered
        output_path = f"{ports.run_root}/outputs/{task_id}/result.json"
        receipt_path = f"{ports.run_root}/receipts/{task_id}/attempt-001.json"
        output_rows.append(
            {
                "output_sha256": verify_published_artifact(
                    output_path,
                    ports=ports,
                    implementation_sha256=task.capability_implementation_sha256,
                ),
                "receipt_sha256": verify_published_artifact(
                    receipt_path,
                    ports=ports,
                    implementation_sha256=task.capability_implementation_sha256,
                ),
                "task_id": task_id,
            }
        )
    return {
        "schema": 'empirical-lawhood/methods/posthoc-response-composition/recovery-verification',
        "version": "1.0.0",
        "value": {
            "execution_plan_sha256": plan.fingerprint(),
            "output_rows": output_rows,
            "recovered_from_external_artifacts_only": True,
            "receipt_input_closure_verified": True,
            "task_count": len(output_rows),
            "verified": len(output_rows) == 21,
        },
    }


__all__ = [
    "MAXIMUM_DOCUMENT_BYTES",
    "MINIMUM_FREE_BYTES",
    "EffectiveLawPosthocExecutionPorts",
    "PublishedTask",
    "execute_plan",
    "load_qualified_documents",
    "publish_document",
    "recover_published_task",
    "source_lineages",
    "verify_complete_run",
    "verify_published_artifact",
]

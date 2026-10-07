"""Mechanical lowering, qualification and recovery for source-pipeline profiles."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.dataset_authority import (
    DatasetAuthorizationIssuerRegistration,
    DatasetDecisionClock,
    DatasetOperationAuthorization,
    DatasetOperationPolicy,
    DatasetOperationRequest,
    replay_dataset_operation_authorization,
)
from empirical_lawhood.planning.study_authoring import MaterializationQualificationReceipt, SourceAccessDisposition
from empirical_lawhood.planning.source_pipelines import SourcePipelineArtifactProfile, SourcePipelineChangeSemantics, SourcePipelineEdgeAccounting, SourcePipelineMode, SourcePipelineProfile, SourcePipelineQualificationDisposition, SourcePipelineQualificationReceipt

from .artifacts import ArtifactManifest, ArtifactProfile, ArtifactWriter
from .capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityPermission,
    CapabilityRegistry,
    CapabilityRequirement,
)
from .dataset_rebuild import (
    DatasetProjectionRebuildAuthorityBundle,
    DatasetProjectionRebuildInstallation,
    DatasetProjectionRebuildInstallerPort,
)
from .datasets import DatasetCapabilityRegistry
from .plans import OutputTemplate
from .sources import (
    SourceCapabilityManifest,
    SourceMode,
    SourceRequest,
    SourceResult,
    validate_source_result,
)


_PROFILE_MAP = {
    SourcePipelineArtifactProfile.CANONICAL_JSON: ArtifactProfile.CANONICAL_JSON,
    SourcePipelineArtifactProfile.ARROW_IPC: ArtifactProfile.ARROW_IPC,
    SourcePipelineArtifactProfile.PARQUET: ArtifactProfile.PARQUET,
    SourcePipelineArtifactProfile.AUDITED_HDF5: ArtifactProfile.AUDITED_HDF5,
    SourcePipelineArtifactProfile.NUMPY_NO_PICKLE: ArtifactProfile.NUMPY_NO_PICKLE,
    SourcePipelineArtifactProfile.JSONL_CHUNKS: ArtifactProfile.JSONL_CHUNKS,
}
_MODE_MAP = {
    SourcePipelineMode.ARCHIVAL: SourceMode.ARCHIVAL,
    SourcePipelineMode.SIMULATED: SourceMode.SIMULATED,
}
_OUTCOME_ACCESS_RANK = {
    OutcomeAccess.OUTCOME_BLIND: 0,
    OutcomeAccess.EVALUATION_SEALED: 0,
    OutcomeAccess.DEVELOPMENT_VISIBLE: 1,
    OutcomeAccess.EVALUATOR_REVEAL: 2,
    OutcomeAccess.EVALUATION_REVEALED: 2,
    OutcomeAccess.PRIVILEGED_TRUTH: 3,
}


class SourcePipelineCompilationDisposition(StrEnum):
    REQUESTS_COMPILED_AUTHORITY_PENDING = "REQUESTS_COMPILED_AUTHORITY_PENDING"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"


class SourcePipelineResumeDisposition(StrEnum):
    REUSED_CURRENT_RECEIPT = "REUSED_CURRENT_RECEIPT"
    REBUILD_CATALOG_PROJECTION = "REBUILD_CATALOG_PROJECTION"


@dataclass(frozen=True, slots=True)
class SourcePipelineCompilation(CanonicalRecord):
    """Profile lowered only into already-owned runtime request contracts."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-pipeline-compilation'

    compilation_id: str
    profile: ObjectIdentity
    disposition: SourcePipelineCompilationDisposition
    source_request: SourceRequest
    transform_requirements: tuple[CapabilityRequirement, ...]
    dataset_operation_request: DatasetOperationRequest | None
    output: OutputTemplate
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.compilation_id, field_name="compilation_id")
        if self.profile.object_schema != SourcePipelineProfile.SCHEMA:
            raise ValueError("source compilation requires an exact pipeline profile")
        if not self.transform_requirements:
            raise ValueError("source compilation requires transform requirements")
        config_ids = tuple(value.config.config_id for value in self.transform_requirements)
        if len(set(config_ids)) != len(config_ids):
            raise ValueError("source compilation transform config IDs repeat")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.disposition
            is SourcePipelineCompilationDisposition.REQUESTS_COMPILED_AUTHORITY_PENDING
        ):
            if self.dataset_operation_request is None or self.reason_codes:
                raise ValueError("authority-pending compilation requires its exact request")
        elif self.dataset_operation_request is not None or not self.reason_codes:
            raise ValueError("authority-stopped source compilation is inconsistent")


def _validate_source_selection(
    profile: SourcePipelineProfile,
    source_manifest: SourceCapabilityManifest,
) -> None:
    selection = profile.source_selection
    if (
        source_manifest.capability_key != selection.capability_key
        or source_manifest.capability_version != selection.capability_version
        or source_manifest.implementation_sha256 != selection.implementation_sha256
    ):
        raise ValueError("source profile selection differs from the registered adapter")
    if source_manifest.mode is not _MODE_MAP[profile.mode]:
        raise ValueError("source profile mode differs from the registered adapter")
    if profile.source_schema not in source_manifest.output_schema_ids:
        raise ValueError("source adapter cannot produce the profile input schema")
    if (
        _OUTCOME_ACCESS_RANK[profile.outcome_access]
        > _OUTCOME_ACCESS_RANK[source_manifest.maximum_outcome_access]
    ):
        raise ValueError("source profile exceeds the registered source outcome access")
    if source_manifest.resource_budget.source_scan_bytes < profile.expected_source_size_bytes:
        raise ValueError("source materialization exceeds the registered source scan budget")


def _validate_dataset_request(
    profile: SourcePipelineProfile,
    request: DatasetOperationRequest,
) -> None:
    if (
        request.action is not AuthorityAction.DATASET_TRANSFORMATION
        or request.manifest != profile.dataset_transformation_manifest
    ):
        raise ValueError("dataset operation request binds another transformation")
    if request.outcome_access is not profile.outcome_access:
        raise ValueError("dataset operation outcome access differs from the profile")
    if not request.work_envelope.resources.contains(profile.resource_ceiling):
        raise ValueError("dataset operation request does not contain the profile resources")
    if request.destination_scope is None:
        raise ValueError("source pipeline transformation lacks a destination scope")
    if request.destination_scope.storage_root.object_id != profile.operator_storage_root_id:
        raise ValueError("source pipeline destination uses another storage root")
    destination_prefix = request.destination_scope.relative_prefix
    if not (
        profile.output_relative_locator == destination_prefix
        or profile.output_relative_locator.startswith(f"{destination_prefix}/")
    ):
        raise ValueError("source pipeline output lies outside its authorized destination")
    source_scopes = tuple(
        value
        for value in request.source_scopes
        if value.storage_root.object_id == profile.operator_storage_root_id
        and (
            profile.source_relative_locator == value.relative_prefix
            or profile.source_relative_locator.startswith(f"{value.relative_prefix}/")
        )
    )
    if len(source_scopes) != 1:
        raise ValueError("source pipeline locator lacks one exact authorized source scope")


def compile_source_pipeline(
    profile: SourcePipelineProfile,
    *,
    source_manifest: SourceCapabilityManifest,
    capability_registry: CapabilityRegistry,
    dataset_registry: DatasetCapabilityRegistry,
    dataset_operation_request: DatasetOperationRequest | None,
) -> SourcePipelineCompilation:
    """Compile registered selections; absence of write authority is a typed stop."""

    _validate_source_selection(profile, source_manifest)
    requirements: list[CapabilityRequirement] = []
    for edge in profile.transform_edges:
        registration = dataset_registry.transform(edge.dataset_transform_registry_id)
        manifest = capability_registry.resolve(
            edge.selection.capability_key,
            edge.selection.capability_version,
        )
        if registration.capability != manifest:
            raise ValueError("dataset and generic transform registrations differ")
        if manifest.implementation_sha256 != edge.selection.implementation_sha256:
            raise ValueError("source transform implementation identity drifted")
        if (
            edge.domain_schema not in manifest.input_schema_ids
            or edge.codomain_schema not in manifest.output_schema_ids
            or edge.source_media_type not in registration.accepted_media_types
            or edge.destination_media_type not in registration.produced_media_types
            or edge.source_format_profile_id not in registration.source_format_profile_ids
            or edge.destination_format_profile_id not in registration.destination_format_profile_ids
        ):
            raise ValueError("source transform edge pair is not registered")
        if edge.config.object_schema != manifest.config_schema:
            raise ValueError("source transform config uses another schema")
        requirement = CapabilityRequirement(
            capability_key=manifest.capability_key,
            capability_version=manifest.capability_version,
            kind=CapabilityKind.TRANSFORM,
            config=CapabilityConfigRef(
                config_id=edge.config.object_id,
                config_schema=edge.config.object_schema,
                config_schema_sha256=manifest.config_schema_sha256,
                content_sha256=edge.config.object_fingerprint,
                artifact_id=edge.config.object_id,
            ),
            required_input_schema_ids=(edge.domain_schema,),
            required_output_schema_ids=(edge.codomain_schema,),
            required_permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            requested_evidence_ceiling=EvidenceCeiling.MEASUREMENT,
            requested_outcome_access=profile.outcome_access,
            requested_resources=edge.resource_budget,
        )
        capability_registry.require(requirement)
        if (
            registration.limits.max_input_bytes < edge.resource_budget.source_scan_bytes
            or registration.limits.max_output_bytes < edge.resource_budget.output_bytes
            or registration.limits.max_records < profile.maximum_row_count
        ):
            raise ValueError("source transform edge exceeds its dataset registration limits")
        requirements.append(requirement)
    if dataset_operation_request is not None:
        _validate_dataset_request(profile, dataset_operation_request)
    return SourcePipelineCompilation(
        compilation_id=f"source-pipeline-compilation.{profile.profile_id}",
        profile=ObjectIdentity.from_record(profile.profile_id, profile),
        disposition=(
            SourcePipelineCompilationDisposition.REQUESTS_COMPILED_AUTHORITY_PENDING
            if dataset_operation_request is not None
            else SourcePipelineCompilationDisposition.AUTHORITY_REQUIRED
        ),
        source_request=SourceRequest(
            request_id=f"source-request.{profile.profile_id}",
            source_id=profile.source_id,
            expected_sha256=profile.expected_source_sha256,
            expected_size_bytes=profile.expected_source_size_bytes,
            licence_id=profile.licence_id,
            outcome_access=profile.outcome_access,
        ),
        transform_requirements=tuple(requirements),
        dataset_operation_request=dataset_operation_request,
        output=OutputTemplate(
            output_id="source-pipeline-output",
            payload_schema=profile.output_schema,
            profile=_PROFILE_MAP[profile.output_artifact_profile],
            media_type=profile.output_media_type,
            filename_suffix=profile.output_filename_suffix,
        ),
        reason_codes=(
            ()
            if dataset_operation_request is not None
            else ("DATASET_TRANSFORMATION_AUTHORITY_REQUIRED",)
        ),
    )


@dataclass(frozen=True, slots=True)
class SourcePipelineValidatorEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-pipeline-validator-evidence'

    validator: ObjectIdentity
    receipt: ObjectIdentity
    passed: bool

    @property
    def validator_id(self) -> str:
        return self.validator.object_id


@dataclass(frozen=True, slots=True)
class SourcePipelineExecutionEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-pipeline-execution-evidence'

    evidence_id: str
    dataset_operation_policy: DatasetOperationPolicy
    dataset_operation_authorization: DatasetOperationAuthorization
    dataset_authorization_issuer: DatasetAuthorizationIssuerRegistration
    source_result: SourceResult
    edge_accounting: tuple[SourcePipelineEdgeAccounting, ...]
    output_manifest: ArtifactManifest
    validator_evidence: tuple[SourcePipelineValidatorEvidence, ...]
    recovery: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        require_sorted_unique_ids(
            self.validator_evidence,
            attribute="validator_id",
            field_name="validator_evidence",
        )


def _artifact_materialization_identity(result: SourceResult) -> ObjectIdentity:
    materialization = result.materialization
    return ObjectIdentity.from_record(materialization.materialization_id, materialization)


def _artifact_manifest_identity(manifest: ArtifactManifest) -> ObjectIdentity:
    return ObjectIdentity.from_record(manifest.logical.logical_artifact_id, manifest)


def _qualify_source_pipeline_after_source_validation(
    profile: SourcePipelineProfile,
    compilation: SourcePipelineCompilation,
    evidence: SourcePipelineExecutionEvidence,
    source_manifest: SourceCapabilityManifest,
    decision_clock: DatasetDecisionClock,
) -> SourcePipelineQualificationReceipt:
    request = compilation.dataset_operation_request
    if request is None:
        raise ValueError("source pipeline qualification lacks its dataset operation request")
    replay_dataset_operation_authorization(
        policy=evidence.dataset_operation_policy,
        request=request,
        authorization=evidence.dataset_operation_authorization,
        issuer_registration=evidence.dataset_authorization_issuer,
        manifest=profile.dataset_transformation_manifest,
        required_action=AuthorityAction.DATASET_TRANSFORMATION,
        implementation_commit=request.implementation_commit,
        clock=decision_clock,
    )
    validate_source_result(source_manifest, compilation.source_request, evidence.source_result)
    if _artifact_materialization_identity(evidence.source_result) != profile.source_materialization:
        raise ValueError("source result materialization differs from the profile")
    if tuple(value.edge_id for value in evidence.edge_accounting) != tuple(
        value.edge_id for value in profile.transform_edges
    ):
        raise ValueError("source pipeline accounting does not cover the exact transform chain")
    previous_artifact = profile.source_materialization
    previous_row_count: int | None = None
    previous_invalid_row_count: int | None = None
    previous_roster_sha256: str | None = None
    for edge, accounting in zip(
        profile.transform_edges,
        evidence.edge_accounting,
        strict=True,
    ):
        if accounting.input_artifact != previous_artifact:
            raise ValueError("source pipeline artifact lineage is discontinuous")
        if previous_row_count is not None and accounting.input_row_count != previous_row_count:
            raise ValueError("source pipeline row accounting is discontinuous")
        if (
            previous_invalid_row_count is not None
            and accounting.input_invalid_row_count != previous_invalid_row_count
        ):
            raise ValueError("source pipeline invalid-row accounting is discontinuous")
        if (
            previous_roster_sha256 is not None
            and accounting.input_physical_unit_roster_sha256 != previous_roster_sha256
        ):
            raise ValueError("source pipeline unit-roster accounting is discontinuous")
        if (
            accounting.observed_input_coordinates != edge.input_coordinates
            or accounting.observed_output_coordinates != edge.output_coordinates
        ):
            raise ValueError("source pipeline observed coordinates drifted")
        if accounting.dropped_without_disposition_count != 0:
            raise ValueError("source pipeline hid row dropping")
        if (
            accounting.input_physical_unit_roster_sha256
            != accounting.output_physical_unit_roster_sha256
        ):
            raise ValueError("source pipeline changed the physical-unit roster")
        if accounting.output_invalid_row_count != accounting.input_invalid_row_count:
            raise ValueError("source pipeline failed to retain invalid-row disposition")
        if (
            edge.row_semantics is SourcePipelineChangeSemantics.PRESERVED
            and accounting.output_row_count != accounting.input_row_count
        ):
            raise ValueError("source pipeline changed rows under preservation semantics")
        if accounting.output_row_count > profile.maximum_row_count or (
            accounting.physical_unit_count > profile.maximum_physical_unit_count
        ):
            raise ValueError("source pipeline accounting exceeds its bounded profile")
        previous_artifact = accounting.output_artifact
        previous_row_count = accounting.output_row_count
        previous_invalid_row_count = accounting.output_invalid_row_count
        previous_roster_sha256 = accounting.output_physical_unit_roster_sha256
    output_identity = _artifact_manifest_identity(evidence.output_manifest)
    if previous_artifact != output_identity:
        raise ValueError("source pipeline terminal accounting names another output")
    logical = evidence.output_manifest.logical
    materialization = evidence.output_manifest.materialization
    if (
        logical.payload_schema != profile.output_schema
        or logical.profile is not _PROFILE_MAP[profile.output_artifact_profile]
        or logical.media_type != profile.output_media_type
        or logical.outcome_access is not profile.outcome_access
        or logical.visibility_ceiling is not profile.visibility_ceiling
        or materialization.storage_root_id != profile.operator_storage_root_id
        or materialization.relative_path != profile.output_relative_locator
        or materialization.size_bytes > profile.resource_ceiling.output_bytes
    ):
        raise ValueError("source pipeline output artifact differs from the profile")
    publication = evidence.output_manifest.publication
    if publication is None:
        raise ValueError("qualified source pipeline requires an atomic publication binding")
    expected_validators = tuple(profile.semantic_validators)
    observed_validators = tuple(value.validator for value in evidence.validator_evidence)
    if observed_validators != expected_validators or not all(
        value.passed for value in evidence.validator_evidence
    ):
        raise ValueError("source pipeline semantic validators are incomplete or failed")
    return SourcePipelineQualificationReceipt(
        receipt_id=f"source-pipeline-qualification.{profile.profile_id}",
        profile=ObjectIdentity.from_record(profile.profile_id, profile),
        disposition=SourcePipelineQualificationDisposition.QUALIFIED,
        source_result=ObjectIdentity.from_record(
            evidence.source_result.request_id,
            evidence.source_result,
        ),
        edge_accounting=evidence.edge_accounting,
        output_artifact=output_identity,
        output_content_sha256=logical.content_sha256,
        validator_receipts=tuple(
            sorted(
                (value.receipt for value in evidence.validator_evidence),
                key=lambda value: value.object_id,
            )
        ),
        publication=ObjectIdentity.from_record(publication.publication_batch_id, publication),
        recovery=evidence.recovery,
        reason_codes=(),
        outcome_access=profile.outcome_access,
        visibility_ceiling=profile.visibility_ceiling,
    )


def qualify_source_pipeline(
    profile: SourcePipelineProfile,
    compilation: SourcePipelineCompilation,
    evidence: SourcePipelineExecutionEvidence,
    *,
    source_manifest: SourceCapabilityManifest,
    decision_clock: DatasetDecisionClock,
) -> SourcePipelineQualificationReceipt:
    """Validate exact source/edge/accounting/publication evidence."""

    profile_identity = ObjectIdentity.from_record(profile.profile_id, profile)
    if compilation.profile != profile_identity:
        raise ValueError("source-pipeline compilation binds another profile")
    if (
        compilation.disposition
        is not SourcePipelineCompilationDisposition.REQUESTS_COMPILED_AUTHORITY_PENDING
    ):
        raise ValueError("authority-stopped source pipeline cannot be qualified")
    _validate_source_selection(profile, source_manifest)
    return _qualify_source_pipeline_after_source_validation(
        profile,
        compilation,
        evidence,
        source_manifest,
        decision_clock,
    )


def authority_required_source_pipeline_receipt(
    profile: SourcePipelineProfile,
) -> SourcePipelineQualificationReceipt:
    return SourcePipelineQualificationReceipt(
        receipt_id=f"source-pipeline-qualification.{profile.profile_id}.authority-stop",
        profile=ObjectIdentity.from_record(profile.profile_id, profile),
        disposition=SourcePipelineQualificationDisposition.AUTHORITY_REQUIRED,
        source_result=None,
        edge_accounting=(),
        output_artifact=None,
        output_content_sha256=None,
        validator_receipts=(),
        publication=None,
        recovery=None,
        reason_codes=("DATASET_TRANSFORMATION_AUTHORITY_REQUIRED",),
        outcome_access=profile.outcome_access,
        visibility_ceiling=profile.visibility_ceiling,
    )


def to_current_materialization_qualification(
    profile: SourcePipelineProfile,
    receipt: SourcePipelineQualificationReceipt,
) -> MaterializationQualificationReceipt:
    """Lower the additive accounting companion to the current source contract."""

    if receipt.profile != ObjectIdentity.from_record(profile.profile_id, profile):
        raise ValueError("source qualification companion binds another profile")
    if receipt.disposition is not SourcePipelineQualificationDisposition.QUALIFIED:
        raise ValueError("only a qualified source pipeline can enter candidate authoring")
    assert receipt.output_artifact is not None
    assert receipt.output_content_sha256 is not None
    return MaterializationQualificationReceipt(
        receipt_id=f"materialization-qualification.{profile.profile_id}",
        source_id=profile.source_id,
        materialization=receipt.output_artifact,
        content_sha256=receipt.output_content_sha256,
        evidence_world_id=profile.evidence_world_id,
        observation_operator=profile.observation_operator,
        numerical_view_ids=profile.numerical_view_ids,
        native_unit_ids=tuple(sorted({value.native_unit for value in profile.output_coordinates})),
        frame_ids=tuple(sorted({value.frame_id for value in profile.output_coordinates})),
        clock_ids=tuple(sorted({value.clock_id for value in profile.output_coordinates})),
        receiver_semantics_id=profile.receiver_semantics_id,
        validity_contract_id=profile.validity_contract_id,
        uncertainty_contract_id=profile.uncertainty_contract_id,
        access_disposition=SourceAccessDisposition.VERIFIED_ACCESS,
        outcome_access=profile.outcome_access,
        visibility_ceiling=profile.visibility_ceiling,
    )


@dataclass(frozen=True, slots=True)
class SourcePipelineResumePlan(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-pipeline-resume-plan'

    plan_id: str
    profile: ObjectIdentity
    qualification: ObjectIdentity
    output_manifest: ObjectIdentity
    disposition: SourcePipelineResumeDisposition
    rebuild_authority_bundle: ObjectIdentity | None
    source_reopen_required: bool
    delivery_repeat_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        if self.source_reopen_required or self.delivery_repeat_required:
            raise ValueError("qualified source recovery cannot repeat source or delivery")
        if (self.disposition is SourcePipelineResumeDisposition.REBUILD_CATALOG_PROJECTION) != (
            self.rebuild_authority_bundle is not None
        ):
            raise ValueError("catalog rebuild disposition and authority bundle differ")


def plan_source_pipeline_resume(
    profile: SourcePipelineProfile,
    receipt: SourcePipelineQualificationReceipt,
    output_manifest: ArtifactManifest,
    *,
    artifact_writer: ArtifactWriter,
    catalog_projection_current: bool,
    rebuild_authority_bundle: DatasetProjectionRebuildAuthorityBundle | None = None,
) -> SourcePipelineResumePlan:
    """Verify authoritative output and avoid reopening the scientific source."""

    if receipt.profile != ObjectIdentity.from_record(profile.profile_id, profile):
        raise ValueError("resume receipt binds another source profile")
    if receipt.disposition is not SourcePipelineQualificationDisposition.QUALIFIED:
        raise ValueError("source pipeline resume requires a qualified receipt")
    manifest_identity = _artifact_manifest_identity(output_manifest)
    if receipt.output_artifact != manifest_identity:
        raise ValueError("source pipeline resume output differs from its receipt")
    artifact_writer.verify_manifest(output_manifest)
    if not catalog_projection_current and rebuild_authority_bundle is None:
        raise ValueError("catalog-loss recovery requires exact rebuild authority")
    rebuild_identity = (
        None
        if rebuild_authority_bundle is None
        else ObjectIdentity.from_record(
            rebuild_authority_bundle.bundle_id,
            rebuild_authority_bundle,
        )
    )
    return SourcePipelineResumePlan(
        plan_id=f"source-pipeline-resume.{profile.profile_id}",
        profile=ObjectIdentity.from_record(profile.profile_id, profile),
        qualification=ObjectIdentity.from_record(receipt.receipt_id, receipt),
        output_manifest=manifest_identity,
        disposition=(
            SourcePipelineResumeDisposition.REUSED_CURRENT_RECEIPT
            if catalog_projection_current
            else SourcePipelineResumeDisposition.REBUILD_CATALOG_PROJECTION
        ),
        rebuild_authority_bundle=rebuild_identity,
        source_reopen_required=False,
        delivery_repeat_required=False,
    )


def execute_source_pipeline_projection_rebuild(
    plan: SourcePipelineResumePlan,
    *,
    authority_bundle: DatasetProjectionRebuildAuthorityBundle,
    installer: DatasetProjectionRebuildInstallerPort,
    receipt_id: str,
) -> DatasetProjectionRebuildInstallation:
    """Delegate total-catalog-loss recovery to the existing authorized installer."""

    expected = ObjectIdentity.from_record(authority_bundle.bundle_id, authority_bundle)
    if (
        plan.disposition is not SourcePipelineResumeDisposition.REBUILD_CATALOG_PROJECTION
        or plan.rebuild_authority_bundle != expected
    ):
        raise ValueError("source pipeline rebuild invocation differs from its resume plan")
    return installer.install(authority_bundle, receipt_id=receipt_id)


__all__ = [
    'SourcePipelineCompilationDisposition',
    'SourcePipelineCompilation',
    'SourcePipelineExecutionEvidence',
    'SourcePipelineResumeDisposition',
    'SourcePipelineResumePlan',
    'SourcePipelineValidatorEvidence',
    "authority_required_source_pipeline_receipt",
    "compile_source_pipeline",
    "execute_source_pipeline_projection_rebuild",
    "plan_source_pipeline_resume",
    "qualify_source_pipeline",
    "to_current_materialization_qualification",
]

"""Stable request/result envelopes for public application services."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import StrEnum
from pathlib import Path
from typing import Generic, TypeVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_value
from empirical_lawhood.planning.datasets import (
    AcquisitionState,
    CustodyState,
    DatasetBindingRole,
    ExternalIdentifier,
)
from empirical_lawhood.runtime.capabilities import CapabilityKind
from empirical_lawhood.runtime.datasets import DatasetCatalogRecordKind


class OperationStatus(StrEnum):
    """Operational outcome, deliberately separate from scientific status."""

    SUCCEEDED = "SUCCEEDED"
    INVALID = "INVALID"
    NOT_FOUND = "NOT_FOUND"
    BLOCKED = "BLOCKED"
    CONFLICT = "CONFLICT"
    FAILED = "FAILED"


class ErrorCategory(StrEnum):
    VALIDATION = "VALIDATION"
    IDENTITY = "IDENTITY"
    AUTHORITY = "AUTHORITY"
    STORAGE = "STORAGE"
    CAPABILITY = "CAPABILITY"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    EXECUTION = "EXECUTION"
    INTERNAL = "INTERNAL"


@dataclass(frozen=True, slots=True)
class ApiError:
    category: ErrorCategory
    reason_code: str
    message: str
    field: str | None = None


PayloadT = TypeVar("PayloadT", covariant=True)


def _public_value(value: object) -> object:
    if isinstance(value, CanonicalRecord):
        return canonical_value(value)
    if is_dataclass(value) and not isinstance(value, type):
        return canonical_value(asdict(value))
    return canonical_value(value)


@dataclass(frozen=True, slots=True)
class ApiResult(Generic[PayloadT]):
    """Versioned deterministic result returned by every public facade method."""

    operation: str
    status: OperationStatus
    payload: PayloadT | None = None
    errors: tuple[ApiError, ...] = ()
    reason_codes: tuple[str, ...] = ()
    schema: str = 'empirical-lawhood/api/result'
    version: str = "1.0.0"

    @property
    def succeeded(self) -> bool:
        return self.status is OperationStatus.SUCCEEDED

    def to_mapping(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "version": self.version,
            "operation": self.operation,
            "status": self.status.value,
            "reason_codes": list(self.reason_codes),
            "errors": [
                {
                    "category": error.category.value,
                    "field": error.field,
                    "message": error.message,
                    "reason_code": error.reason_code,
                }
                for error in self.errors
            ],
            "payload": None if self.payload is None else _public_value(self.payload),
        }


@dataclass(frozen=True, slots=True)
class DocumentRequest:
    path: Path


@dataclass(frozen=True, slots=True)
class DatasetManifestRequest:
    path: Path


@dataclass(frozen=True, slots=True)
class PublicSourceAcquisitionRequest:
    invocation_path: Path
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class DatasetOperationPreviewRequest:
    manifest_path: Path
    policy_path: Path
    request_path: Path
    authorization_id: str
    authorization_path: Path | None = None


@dataclass(frozen=True, slots=True)
class DatasetRegistrationRequest:
    manifest_path: Path
    policy_path: Path
    request_path: Path
    authorization_path: Path
    bundle_id: str
    receipt_id: str
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class DatasetProjectionRebuildRequest:
    manifest_path: Path
    policy_path: Path
    request_path: Path
    authorization_path: Path
    bundle_id: str
    receipt_id: str
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class DatasetListRequest:
    limit: int = 100
    cursor: str | None = None
    family_ids: tuple[str, ...] = ()
    release_ids: tuple[str, ...] = ()
    provider_ids: tuple[str, ...] = ()
    external_identifiers: tuple[ExternalIdentifier, ...] = ()
    custody_states: tuple[CustodyState, ...] = ()
    acquisition_states: tuple[AcquisitionState, ...] = ()
    experiment_spec_ids: tuple[str, ...] = ()
    roles: tuple[DatasetBindingRole, ...] = ()


@dataclass(frozen=True, slots=True)
class DatasetShowRequest:
    dataset_id: str
    kind: DatasetCatalogRecordKind | None = None


@dataclass(frozen=True, slots=True)
class CapabilityListRequest:
    limit: int = 100
    cursor: str | None = None
    kind: CapabilityKind | None = None


@dataclass(frozen=True, slots=True)
class CapabilityShowRequest:
    capability_key: str
    capability_version: str


@dataclass(frozen=True, slots=True)
class CompileCampaignRequest:
    path: Path


@dataclass(frozen=True, slots=True)
class CompileStudyRequest:
    path: Path
    emit_path: Path | None = None
    parent_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CompileCandidateRequest:
    path: Path
    extension_payload_paths: tuple[Path, ...]
    decoder_registration_paths: tuple[Path, ...]
    emit_path: Path | None = None
    parent_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CheckReadinessRequest:
    authoring_package_path: Path
    extension_payload_paths: tuple[Path, ...]
    decoder_registration_paths: tuple[Path, ...]
    expected_candidate_path: Path
    source_closure_path: Path
    execution_resource_envelope_spec_path: Path
    predevelopment_jit_signature_census_path: Path | None
    jit_graph_signature_manifest_path: Path | None
    run_plan_id: str
    campaign_elapsed_budget_path: Path | None = None
    campaign_elapsed_reservation_plan_path: Path | None = None
    model_set_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CompileSourceProfileRequest:
    source_profile_path: Path
    transformation_manifest_path: Path
    dataset_operation_request_path: Path | None = None
    emit_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CompileLinkedCampaignProfileRequest:
    linked_profile_path: Path
    source_compilation_path: Path
    evidence_profile_selection_path: Path
    candidate_report_paths: tuple[Path, ...]
    emit_path: Path | None = None


@dataclass(frozen=True, slots=True)
class AssembleExperimentPackageRequest:
    base_package_path: Path
    issued_study_path: Path
    publication_receipt_path: Path
    frozen_proposal_path: Path
    scientific_approval_path: Path | None
    execution_authority_path: Path | None
    execution_resource_envelope_spec_path: Path
    run_plan_id: str
    grantee_id: str
    at_utc: str
    predevelopment_jit_signature_census_path: Path | None = None
    jit_graph_signature_manifest_path: Path | None = None
    emit_path: Path | None = None


@dataclass(frozen=True, slots=True)
class BindElapsedBudgetRequest:
    execution_package_path: Path
    campaign_elapsed_budget_path: Path
    campaign_elapsed_reservation_plan_path: Path
    emit_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CompileStudyBundleRequest:
    path: Path
    emit_path: Path | None = None


@dataclass(frozen=True, slots=True)
class IssueStudyRequest:
    authoring_package_path: Path
    expected_candidate_path: Path
    proposer_attestation_path: Path
    source_closure_path: Path
    custody_authority_path: Path
    confirmed: bool = False
    parent_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class IssueExtensionsRequest:
    authoring_package_path: Path
    extension_payload_paths: tuple[Path, ...]
    decoder_registration_paths: tuple[Path, ...]
    expected_candidate_path: Path
    base_issue_id: str
    extension_proposer_attestation_path: Path
    extension_custody_authority_path: Path
    confirmed: bool = False
    parent_record_path: Path | None = None


@dataclass(frozen=True, slots=True)
class IssueStudyBundleRequest:
    bundle_candidate_path: Path
    child_issue_ids: tuple[str, ...]
    child_science_paths: tuple[Path, ...]
    archive_outcome_protection_path: Path
    outcome_barrier_plan_path: Path
    partial_morphism_path: Path
    joint_adjudication_path: Path
    custody_authority_path: Path
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class AdvanceStudyBundleRequest:
    outcome_barrier_plan_path: Path
    barrier_prefix_path: Path
    barrier_id: str
    reveal_authority_id: str | None = None
    child_result_path: Path | None = None


@dataclass(frozen=True, slots=True)
class StudyBundleStatusRequest:
    outcome_barrier_plan_path: Path
    barrier_prefix_path: Path


@dataclass(frozen=True, slots=True)
class CloseStudyBundleRequest:
    joint_adjudication_plan_path: Path
    outcome_barrier_plan_path: Path
    terminal_barrier_prefix_path: Path
    child_result_paths: tuple[Path, ...]
    archive_overlap_result_path: Path
    morphism_verdict_paths: tuple[Path, ...]
    morphism_control_contrast_path: Path
    mapped_prospective_accepted_count: int
    mapped_prospective_map_unevaluable_count: int


@dataclass(frozen=True, slots=True)
class RunCampaignRequest:
    path: Path
    confirmed: bool = False
    reveal_authority_id: str | None = None
    parent_record_path: Path | None = None
    parent_input_binding_paths: tuple[Path, ...] = ()


@dataclass(frozen=True, slots=True)
class ResumeCampaignRequest:
    run_id: str
    confirmed: bool = False
    reveal_authority_id: str | None = None
    parent_record_path: Path | None = None
    parent_input_binding_paths: tuple[Path, ...] = ()
    retry_amendment_path: Path | None = None
    retry_authority_id: str | None = None
    source_amendment_path: Path | None = None
    source_authority_id: str | None = None


@dataclass(frozen=True, slots=True)
class CampaignMaintenanceStopRequest:
    authority_path: Path
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class CampaignMaintenanceInterruptionRequest:
    authority_path: Path
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class CampaignStatusRequest:
    run_id: str
    include_attempt_history: bool = False
    attempt_history_limit: int = 100
    attempt_history_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class CatalogQueryRequest:
    object_id: str | None = None
    kind: str | None = None
    categorical_status: str | None = None
    limit: int = 100
    cursor: str | None = None
    relation_id: str | None = None
    denominator_gauge_id: str | None = None
    response_gauge_id: str | None = None
    horizon_id: str | None = None
    native_unit: str | None = None
    knowledge_edge_relation: str | None = None
    knowledge_edge_scope_id: str | None = None
    lineage_object_id: str | None = None
    match_count_limit: int = 1_000


@dataclass(frozen=True, slots=True)
class CatalogRebuildRequest:
    projection_path: Path
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class DatabaseMutationRequest:
    confirmed: bool = False


@dataclass(frozen=True, slots=True)
class InspectSystemRequest:
    system_id: str

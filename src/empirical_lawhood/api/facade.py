"""Stable application facade over the accepted follow-up services."""

from __future__ import annotations

from empirical_lawhood.planning.approval import FrozenRetrospectiveApproval
from empirical_lawhood.adapters.physical.mast_archive.fresh_source import FairMastPublicSelection, FairMastPublicSourceService, FairMastSourceReceipt
from empirical_lawhood.kernel.time import InformationCutoff
from empirical_lawhood.planning.study_authoring import DesignInputRecord

from empirical_lawhood.kernel.retrospective_prediction_experiments import RetrospectiveExperimentSpec

from empirical_lawhood.planning.experiment_entry import RetrospectiveAuthoringBase, RetrospectiveAuthoringPackage
from empirical_lawhood.runtime.candidate_compiler import RetrospectiveStandardCandidate, RetrospectiveStandardReport, RetrospectiveExtensionReport
from empirical_lawhood.runtime.study_issue import RetrospectiveIssuedBase, RetrospectiveIssuedStudy, RetrospectivePublicationReceipt
from empirical_lawhood.api.models import RetrospectiveCompilationSummary, RetrospectiveIssueSummary, RetrospectiveBaseCompilationSummary, RetrospectiveBaseIssueSummary, RetrospectiveCampaignBase

from dataclasses import replace
import base64
import binascii
import hashlib
import importlib.metadata
import json
import logging
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Callable, Mapping, TypeVar, cast

from empirical_lawhood.infrastructure.artifacts import (
    ArtifactPlaneError,
    ExternalStorageDiagnostic,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.bounded_io import BoundedFileIOError, read_bounded_bytes
from empirical_lawhood.infrastructure.bounded_process import (
    BoundedProcessError,
    run_bounded_command,
)
from empirical_lawhood.infrastructure.catalog_projection import MAX_CATALOG_PROJECTION_BYTES
from empirical_lawhood.infrastructure.dataset_projection import decode_unified_catalog_snapshot
from empirical_lawhood.infrastructure.execution import ExecutionErrorKind
from empirical_lawhood.infrastructure.sql import DatasetCatalogQueryWorkLimitExceeded
from empirical_lawhood.infrastructure.sql.database import (
    ALEMBIC_HEAD,
    APPLICATION_ID,
    PRODUCTION_RELATIVE_PATH,
    SCHEMA_VERSION,
    CatalogSchemaAuditLimitExceeded,
    catalog_query_preflight,
    create_catalog_engine,
    create_read_only_catalog_engine,
    initialize_production_catalog,
    production_catalog_path,
    schema_audit,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.infrastructure.catalog_rebuild import (
    CatalogRebuildError,
    rebuild_production_catalog,
)
from empirical_lawhood.infrastructure.sql.repository import (
    DEFAULT_CATALOG_QUERY_WORK_LIMIT,
    MAX_CATALOG_MATCH_COUNT_LIMIT,
    CatalogObjectQuery,
    CatalogQueryWorkLimitExceeded,
    SQLiteCatalogRepository,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    CanonicalizationError,
    canonical_json_bytes,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.models import ViewModelSetSpec, ModelSetSpec
from empirical_lawhood.planning.authority import AuthorizationDecision
from empirical_lawhood.planning.approval import CompleteApprovalService, DecisionClock, DurableAuthorizationRecord, FrozenIssuedStudyApprovalProposal, SystemDecisionClock
from empirical_lawhood.planning.design import ExperimentProposal
from empirical_lawhood.planning.dataset_limits import MAX_DATASET_RECORD_CANONICAL_BYTES
from empirical_lawhood.planning.dataset_manifests import (
    DATASET_MANIFEST_SCHEMAS,
    EXPERIMENT_DATASET_BINDINGS_EXTENSION_NAMESPACE,
    DatasetRegistrationManifest,
    DatasetTransformationManifest,
    ProposedExperimentDatasetBindingManifest,
)
from empirical_lawhood.planning.dataset_authority import (
    DatasetOperationAuthorization,
    DatasetOperationPolicy,
    DatasetOperationRequest,
)
from empirical_lawhood.planning.dataset_rebuild import (
    DatasetProjectionRebuildAuthorization,
    DatasetProjectionRebuildManifest,
    DatasetProjectionRebuildPolicy,
    DatasetProjectionRebuildRequest as AuthoredDatasetProjectionRebuildRequest,
)
from empirical_lawhood.planning.exploration import ExploratoryFinding, HypothesisSet
from empirical_lawhood.planning.experiment_entry import StudyDefinition, ExecutableStudyDefinition
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.linked_campaign import LinkedCampaignProfile
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.planning.prospective import (
    NominationObligations,
    ProspectiveNominationDecision,
)
from empirical_lawhood.planning.study_authoring import StudyDraft
from empirical_lawhood.planning.study_bundles import StudyBundleSpec
from empirical_lawhood.planning.multi_world_study import ArchiveOutcomeProtectionPlan, ArchiveToSimulatorPartialMorphismSpec, MultiWorldJointAdjudicationPlan, MultiWorldOutcomeBarrierPlan, MultiWorldStudyChildScientificBinding
from empirical_lawhood.planning.study_issue import HumanProposerAttestation, RetrospectiveProposerAttestation, ImplementationSourceClosure, StudyAuthorityKind, StudyOperationAuthority
from empirical_lawhood.runtime.candidate_compiler import AuthoringMaterializationIdentity, CandidateCompilationContext, CandidateCompilationDisposition, CandidateDiagnostic, CandidateDiagnosticClass, CandidateDiagnosticCode, CandidateScientificGraph, ContentIdentityPolicy, DraftStudyCandidate, StudyCompilationReport, ExecutableStudyCompilationReport, StandardCandidateCompilationContext, StudyCandidate, ExecutableStudyCandidate, bind_standard_candidate_compilation_report, compile_study_candidate
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateContextProvider,
)
from empirical_lawhood.runtime.extension_bundles import GeneratedExtensionBundleAggregate
from empirical_lawhood.runtime.capabilities import CapabilityKind, CapabilityRegistry
from empirical_lawhood.runtime.study_bundle_compiler import StudyBundleCandidate, StudyBundleCompilationDisposition, StudyBundleCompilationInput, StudyBundleCompilationReport, compile_study_bundle
from empirical_lawhood.runtime.multi_world_study_issue import IssuedMultiWorldStudyChild, MultiWorldAuthorityRequired, MultiWorldOutcomeBarrierPrefix, MultiWorldStudyBarrierStatus, bind_multi_world_child_result, inspect_study_bundle_barrier_status, prepare_study_bundle_issue, request_multi_world_reveal
from empirical_lawhood.runtime.multi_world_study import ArchiveOverlapQualificationResult, MultiWorldJointAdjudicationResult, MorphismControlContrastReceipt, PropertyMorphismVerdict, WorldLocalStudyResult, adjudicate_joint_multi_world_study
from empirical_lawhood.runtime.compiler import ProtocolCompilationError, compile_preissue_run_plan, compile_exploration_execution_plan, compile_exploration_wave_execution_plan, compile_run_plan, execution_topology_sha256, lower_run_plan
from empirical_lawhood.runtime.catalog import ScientificObjectKind
from empirical_lawhood.runtime.dataset_catalog_service import (
    DatasetCatalogLookup,
    DatasetCatalogReadAuthenticationError,
    DatasetCatalogReadService,
    DatasetCatalogTrustedQueryWorkLimitExceeded,
)
from empirical_lawhood.runtime.dataset_binding import (
    DatasetBindingCompilationError,
    DatasetCampaignBindingResolver,
)
from empirical_lawhood.runtime.datasets import (
    MAX_DATASET_PAGE_RECORDS,
    DatasetCatalogCursor,
    DatasetCatalogQueryBoundary,
    DatasetCatalogRecord,
    DatasetCatalogRecordKind,
    DatasetCatalogWorkLimit,
    dataset_catalog_record_key,
)
from empirical_lawhood.runtime.public_source_acquisition import CaptureAssuredSourceAcquisitionInvocation, PublicSourceAcquisitionService, ChecksummedSourceAcquisitionInvocation
from empirical_lawhood.runtime.dataset_operations import (
    DatasetOperationAuthorityBundle,
    DatasetOperationPreview,
    DatasetOperationPreviewService,
    DatasetOperationPreviewState,
    DatasetStorageScopePreflight,
)
from empirical_lawhood.runtime.dataset_registration import (
    DatasetRegistrationInstallation,
    DatasetRegistrationInstallerPort,
)
from empirical_lawhood.runtime.conditional_children import ConditionalChildInstantiation, ConditionalChildResolution, ConditionalChildResolver, FrozenParentInputBinding, bind_frozen_parent_input
from empirical_lawhood.runtime.dataset_rebuild import (
    DatasetProjectionRebuildAuthorityBundle,
    DatasetProjectionRebuildInstallation,
    DatasetProjectionRebuildInstallerPort,
)
from empirical_lawhood.runtime.exploration import ExplorationWaveResult
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, ProtocolRunPlan, ScientificInputRole
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec, JitGraphSignatureManifest, PredevelopmentJitSignatureCensus
from empirical_lawhood.runtime.campaign_elapsed_budget import CampaignElapsedBudgetSpec, CampaignElapsedReservationPlan
from empirical_lawhood.runtime.executable_bindings import GeneratedExecutableBindingAggregate
from empirical_lawhood.runtime.executable_bindings import (
    ExecutableCapabilityProviderFactoryRegistry,
)
from empirical_lawhood.runtime.parameterised_candidate_context import compose_parameterised_candidate_context
from empirical_lawhood.runtime.profile_compilation import ExecutableBindingRequiredError, LinkedCampaignExecutableCompilation, LinkedCampaignExecutableDisposition, LinkedCampaignProfileCompiler, SourceProfileCompilerRegistry
from empirical_lawhood.runtime.source_pipelines import SourcePipelineCompilationDisposition, SourcePipelineCompilation
from empirical_lawhood.runtime.study_issue import StudyPublicationReceipt, ExtensionPublicationReceipt, StudyExtensionDecoderRegistration, StudySourceClosureInspector, IssuedExecutableStudyManifest, prepare_study_issue, prepare_executable_study_issue, project_standard_study_extensions
from empirical_lawhood.runtime.study_extensions import StudyExtensionMaterializationReceipt
from empirical_lawhood.runtime.static_codecs import CanonicalRecordCodecRegistry
from empirical_lawhood.runtime.scientific_graph_preservation import prove_scientific_graph_parity
from empirical_lawhood.infrastructure.study_issue import ExternalIssuedStudyPublisher, StudyOperationAuthorityStore

from .codecs import load_standard_issued_study, load_standard_study_candidate, MAX_AUTHORING_BYTES, AuthoringCodecError, load_authoring, load_campaign_package_authoring, load_registered_authoring, load_registered_authoring_materialization, loads_registered_authoring
from .execution import (
    CampaignExecutionError,
    CampaignExecutionService,
    CampaignIdentityError,
    CampaignInternalError,
    CampaignValidationError,
    RunExecutionSummary,
    RunStatusSummary,
)
from .models import CampaignAuthorityRequired, ExperimentPackageAssemblySummary, CampaignPackage, AuthorizationSummary, CapabilityListSummary, CapabilityShowSummary, CatalogObjectSummary, CatalogQuerySummary, CatalogRebuildSummary, CatalogSummary, StudyCompilationSummary, ExecutableStudyCompilationSummary, CompilationSummary, PreissueReadinessSummary, LinkedCampaignCompilationSummary, DatabaseMutationSummary, DatasetCatalogRecordSummary, DatasetListSummary, DatasetManifestSummary, DatasetOperationPreviewSummary, PublicSourceAcquisitionSummary, PublicSourceCustodySummary, DatasetProjectionRebuildSummary, DatasetRegistrationSummary, DatasetShowSummary, DatasetStoragePreflightSummary, DoctorActionReadiness, DoctorAuthorityBoundary, DoctorBackendSummary, DoctorSummary, DocumentSummary, DualLoopPackage, ExplorationExecutionPackage, IssuedCampaignPackage, IssuedCampaignPackageRoot, IssuedStudyPackage, EnvelopeExperimentPackage, ExperimentPackage, SourceProfileCompilationSummary, assemble_experiment_package, bind_issued_execution_package_elapsed_budget, issued_base_candidate, issued_standard_candidate, ExplorationCompilationSummary, ExplorationPortfolioSummary, HypothesisSummary, NominationSummary, StudyIssueSummary, ExecutableStudyIssueSummary, MultiWorldStudyIssueSummary, StudyBundleCompilationSummary, ProposalSummary, ScientificInspectionSummary, SystemSummary, WriteEffectSummary
from .ports import (
    DatasetCatalogSessionProvider,
    ProspectiveWorkflow,
)
from .results import AdvanceStudyBundleRequest, AssembleExperimentPackageRequest, BindElapsedBudgetRequest, ApiError, ApiResult, CapabilityListRequest, CapabilityShowRequest, CampaignStatusRequest, CampaignMaintenanceStopRequest, CampaignMaintenanceInterruptionRequest, CatalogQueryRequest, CatalogRebuildRequest, CompileCampaignRequest, CompileStudyRequest, CompileCandidateRequest, CompileLinkedCampaignProfileRequest, CompileStudyBundleRequest, CompileSourceProfileRequest, CloseStudyBundleRequest, DatabaseMutationRequest, DatasetListRequest, DatasetManifestRequest, DatasetOperationPreviewRequest, PublicSourceAcquisitionRequest, DatasetProjectionRebuildRequest, DatasetRegistrationRequest, DatasetShowRequest, DocumentRequest, ErrorCategory, IssueStudyRequest, IssueExtensionsRequest, IssueStudyBundleRequest, OperationStatus, CheckReadinessRequest, StudyBundleStatusRequest, ResumeCampaignRequest, RunCampaignRequest


ResultT = TypeVar("ResultT")
ExplorationPackage = DualLoopPackage | ExplorationExecutionPackage
LOGGER = logging.getLogger(__name__)
CATALOG_CURSOR_SCHEMA = 'empirical-lawhood/api/catalog-query-cursor'
CAPABILITY_CURSOR_SCHEMA = 'empirical-lawhood/api/capability-list-cursor'
MAX_CATALOG_CURSOR_BYTES = 2_048
MAX_CAPABILITY_CURSOR_BYTES = 2_048
MAX_CAPABILITY_PAGE_RECORDS = 100
MAX_DATASET_CATALOG_CURSOR_BYTES = 4_096
DATASET_CATALOG_WORK_LIMIT = DatasetCatalogWorkLimit(
    max_records_examined=10_000,
    max_query_steps=100_000,
)
DATASET_OPERATION_AUTHORITY_SCHEMAS: Mapping[str, type[CanonicalRecord]] = types.MappingProxyType(
    {
        DatasetOperationPolicy.SCHEMA: DatasetOperationPolicy,
        DatasetOperationRequest.SCHEMA: DatasetOperationRequest,
        DatasetOperationAuthorization.SCHEMA: DatasetOperationAuthorization,
    }
)
DATASET_PROJECTION_REBUILD_SCHEMAS: Mapping[str, type[CanonicalRecord]] = types.MappingProxyType(
    {
        DatasetProjectionRebuildManifest.SCHEMA: DatasetProjectionRebuildManifest,
        DatasetProjectionRebuildPolicy.SCHEMA: DatasetProjectionRebuildPolicy,
        AuthoredDatasetProjectionRebuildRequest.SCHEMA: (AuthoredDatasetProjectionRebuildRequest),
        DatasetProjectionRebuildAuthorization.SCHEMA: (DatasetProjectionRebuildAuthorization),
    }
)
_DATASET_CURSOR_SCHEMAS = {DatasetCatalogCursor.SCHEMA: DatasetCatalogCursor}
_DATASET_SHOW_KINDS = frozenset(
    {
        DatasetCatalogRecordKind.FAMILY,
        DatasetCatalogRecordKind.RELEASE,
    }
)


def _catalog_filter_fingerprint(request: CatalogQueryRequest) -> str:
    values = {
        "categorical_status": request.categorical_status,
        "denominator_gauge_id": request.denominator_gauge_id,
        "horizon_id": request.horizon_id,
        "kind": request.kind,
        "knowledge_edge_relation": request.knowledge_edge_relation,
        "knowledge_edge_scope_id": request.knowledge_edge_scope_id,
        "lineage_object_id": request.lineage_object_id,
        "native_unit": request.native_unit,
        "object_id": request.object_id,
        "relation_id": request.relation_id,
        "response_gauge_id": request.response_gauge_id,
    }
    return hashlib.sha256(canonical_json_bytes(values)).hexdigest()


def _catalog_cursor(after_object_id: str, request: CatalogQueryRequest) -> str:
    payload = canonical_json_bytes(
        {
            "after_object_id": after_object_id,
            "filter_sha256": _catalog_filter_fingerprint(request),
            "schema": CATALOG_CURSOR_SCHEMA,
            "version": "1.0.0",
        }
    )
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _strict_cursor_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("catalog cursor contains a duplicate field")
        value[key] = item
    return value


def _capability_cursor(
    *,
    after_registration_id: str,
    catalog_sha256: str,
    kind: CapabilityKind | None,
) -> str:
    payload = canonical_json_bytes(
        {
            "after_registration_id": after_registration_id,
            "catalog_sha256": catalog_sha256,
            "kind": None if kind is None else kind.value,
            "schema": CAPABILITY_CURSOR_SCHEMA,
            "version": "1.0.0",
        }
    )
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_capability_cursor(
    cursor: str,
    *,
    catalog_sha256: str,
    kind: CapabilityKind | None,
) -> str:
    if (
        not isinstance(cursor, str)
        or not cursor
        or "=" in cursor
        or len(cursor.encode("utf-8")) > MAX_CAPABILITY_CURSOR_BYTES
    ):
        raise ValueError("capability cursor is invalid or exceeds its byte limit")
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(
            (cursor + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        value = json.loads(decoded, object_pairs_hook=_strict_cursor_object)
    except (UnicodeEncodeError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError) as error:
        raise ValueError("capability cursor is invalid") from error
    expected_fields = {
        "after_registration_id",
        "catalog_sha256",
        "kind",
        "schema",
        "version",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("capability cursor shape is invalid")
    if value["schema"] != CAPABILITY_CURSOR_SCHEMA or value["version"] != "1.0.0":
        raise ValueError("capability cursor schema is unsupported")
    after_registration_id = value["after_registration_id"]
    observed_catalog_sha256 = value["catalog_sha256"]
    observed_kind = value["kind"]
    if (
        not isinstance(after_registration_id, str)
        or not isinstance(observed_catalog_sha256, str)
        or (observed_kind is not None and not isinstance(observed_kind, str))
    ):
        raise ValueError("capability cursor fields have invalid types")
    try:
        capability_key, capability_version = after_registration_id.rsplit("@", 1)
    except ValueError as error:
        raise ValueError("capability cursor registration ID is invalid") from error
    validate_stable_id(capability_key, field_name="cursor.capability_key")
    validate_semantic_version(capability_version)
    validate_sha256(
        observed_catalog_sha256,
        field_name="cursor.catalog_sha256",
    )
    expected_kind = None if kind is None else kind.value
    if observed_catalog_sha256 != catalog_sha256 or observed_kind != expected_kind:
        raise ValueError("capability cursor belongs to different catalog or filters")
    return after_registration_id


def _decode_catalog_cursor(cursor: str, request: CatalogQueryRequest) -> str:
    if not cursor or len(cursor.encode("utf-8")) > MAX_CATALOG_CURSOR_BYTES:
        raise ValueError("catalog cursor is empty or exceeds its byte limit")
    try:
        padding = "=" * (-len(cursor) % 4)
        decoded = base64.b64decode(
            (cursor + padding).encode("ascii"),
            altchars=b"-_",
            validate=True,
        )
        value = json.loads(decoded, object_pairs_hook=_strict_cursor_object)
    except (UnicodeEncodeError, UnicodeDecodeError, binascii.Error, json.JSONDecodeError) as error:
        raise ValueError("catalog cursor is invalid") from error
    if not isinstance(value, dict) or set(value) != {
        "after_object_id",
        "filter_sha256",
        "schema",
        "version",
    }:
        raise ValueError("catalog cursor shape is invalid")
    if value["schema"] != CATALOG_CURSOR_SCHEMA or value["version"] != "1.0.0":
        raise ValueError("catalog cursor schema is unsupported")
    after_object_id = value["after_object_id"]
    filter_sha256 = value["filter_sha256"]
    if not isinstance(after_object_id, str) or not isinstance(filter_sha256, str):
        raise ValueError("catalog cursor fields have invalid types")
    validate_stable_id(after_object_id, field_name="cursor.after_object_id")
    validate_sha256(filter_sha256, field_name="cursor.filter_sha256")
    if filter_sha256 != _catalog_filter_fingerprint(request):
        raise ValueError("catalog cursor belongs to different filters")
    return after_object_id


def _validate_catalog_query(request: CatalogQueryRequest) -> str | None:
    if request.limit <= 0 or request.limit > 1_000:
        raise ValueError("limit must be in [1, 1000]")
    if request.match_count_limit <= 0 or request.match_count_limit > MAX_CATALOG_MATCH_COUNT_LIMIT:
        raise ValueError(f"match_count_limit must be in [1, {MAX_CATALOG_MATCH_COUNT_LIMIT}]")
    for field_name, value in (
        ("object_id", request.object_id),
        ("relation_id", request.relation_id),
        ("denominator_gauge_id", request.denominator_gauge_id),
        ("response_gauge_id", request.response_gauge_id),
        ("horizon_id", request.horizon_id),
        ("knowledge_edge_scope_id", request.knowledge_edge_scope_id),
        ("lineage_object_id", request.lineage_object_id),
    ):
        if value is not None:
            validate_stable_id(value, field_name=field_name)
    if request.kind is not None:
        ScientificObjectKind(request.kind)
    for field_name, value, maximum in (
        ("categorical_status", request.categorical_status, 80),
        ("native_unit", request.native_unit, 80),
        ("knowledge_edge_relation", request.knowledge_edge_relation, 80),
    ):
        if value is not None and (
            not value or len(value) > maximum or value != value.strip() or not value.isprintable()
        ):
            raise ValueError(f"{field_name} is invalid")
    return None if request.cursor is None else _decode_catalog_cursor(request.cursor, request)


def _encode_dataset_catalog_cursor(cursor: DatasetCatalogCursor) -> str:
    payload = cursor.canonical_bytes()
    if len(payload) > MAX_DATASET_CATALOG_CURSOR_BYTES:
        raise ValueError("dataset catalog cursor exceeds its canonical byte limit")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_dataset_catalog_cursor(cursor: str) -> DatasetCatalogCursor:
    maximum_encoded_bytes = ((MAX_DATASET_CATALOG_CURSOR_BYTES + 2) // 3) * 4
    if not isinstance(cursor, str) or not cursor or "=" in cursor:
        raise ValueError("dataset catalog cursor is invalid")
    try:
        encoded = cursor.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("dataset catalog cursor is invalid") from error
    if len(encoded) > maximum_encoded_bytes:
        raise ValueError("dataset catalog cursor exceeds its byte limit")
    try:
        padding = b"=" * (-len(encoded) % 4)
        payload = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("dataset catalog cursor is invalid") from error
    if len(payload) > MAX_DATASET_CATALOG_CURSOR_BYTES:
        raise ValueError("dataset catalog cursor exceeds its canonical byte limit")
    try:
        value = loads_registered_authoring(
            payload.decode("utf-8"),
            media_type="application/json",
            root_schemas=_DATASET_CURSOR_SCHEMAS,
            maximum_bytes=MAX_DATASET_CATALOG_CURSOR_BYTES,
        )
    except (AuthoringCodecError, UnicodeDecodeError, ValueError) as error:
        raise ValueError("dataset catalog cursor is invalid") from error
    if value.canonical_bytes() != payload or _encode_dataset_catalog_cursor(value) != cursor:
        raise ValueError("dataset catalog cursor is not canonical")
    return value


def _validate_dataset_list_request(
    request: DatasetListRequest,
) -> DatasetCatalogCursor | None:
    if not isinstance(request, DatasetListRequest):
        raise ValueError("request must be a DatasetListRequest")
    if (
        not isinstance(request.limit, int)
        or isinstance(request.limit, bool)
        or not 1 <= request.limit <= MAX_DATASET_PAGE_RECORDS
    ):
        raise ValueError(f"limit must be in [1, {MAX_DATASET_PAGE_RECORDS}]")
    DatasetCatalogQueryBoundary(
        snapshot_fingerprint="0" * 64,
        projection_state_fingerprint="1" * 64,
        projection_anchor_fingerprint="2" * 64,
        work_limit=DATASET_CATALOG_WORK_LIMIT,
        family_ids=request.family_ids,
        release_ids=request.release_ids,
        provider_ids=request.provider_ids,
        external_identifiers=request.external_identifiers,
        custody_states=request.custody_states,
        acquisition_states=request.acquisition_states,
        experiment_spec_ids=request.experiment_spec_ids,
        roles=request.roles,
    )
    return None if request.cursor is None else _decode_dataset_catalog_cursor(request.cursor)


def _validate_dataset_show_request(request: DatasetShowRequest) -> None:
    if not isinstance(request, DatasetShowRequest):
        raise ValueError("request must be a DatasetShowRequest")
    validate_stable_id(request.dataset_id, field_name="dataset_id")
    if request.kind is not None and (
        not isinstance(request.kind, DatasetCatalogRecordKind)
        or request.kind not in _DATASET_SHOW_KINDS
    ):
        raise ValueError("dataset show kind must be FAMILY or RELEASE")


def _dataset_catalog_record_summary(
    record: DatasetCatalogRecord,
) -> DatasetCatalogRecordSummary:
    record_kind, record_id = dataset_catalog_record_key(record)
    return DatasetCatalogRecordSummary(
        record_id=record_id,
        record_kind=record_kind,
        schema=record.SCHEMA,
        version=record.VERSION,
        fingerprint=record.fingerprint(),
    )


def _dataset_manifest_summary(
    manifest: DatasetRegistrationManifest
    | DatasetTransformationManifest
    | ProposedExperimentDatasetBindingManifest,
) -> DatasetManifestSummary:
    source_scope_ids: tuple[str, ...]
    destination_scope_id: str | None
    if isinstance(manifest, DatasetRegistrationManifest):
        source_scope_ids = (manifest.source_scope.scope_id,)
        destination_scope_id = None
    elif isinstance(manifest, DatasetTransformationManifest):
        source_scope_ids = tuple(value.scope.scope_id for value in manifest.inputs)
        destination_scope_id = manifest.destination_scope.scope_id
    elif isinstance(manifest, ProposedExperimentDatasetBindingManifest):
        source_scope_ids = ()
        destination_scope_id = None
    else:
        raise TypeError("unsupported dataset manifest type")
    return DatasetManifestSummary(
        manifest_id=manifest.manifest_id,
        manifest_kind=type(manifest).__name__,
        schema=manifest.SCHEMA,
        version=manifest.VERSION,
        fingerprint=manifest.fingerprint(),
        source_scope_ids=source_scope_ids,
        destination_scope_id=destination_scope_id,
        control_write_scope_id=manifest.control_write_scope.scope_id,
        capability_registry_keys=tuple(
            capability.registry_key for capability in manifest.capabilities
        ),
        work_envelope_fingerprint=manifest.work_envelope.fingerprint(),
        outcome_access=manifest.outcome_access.value,
        visibility_ceiling=manifest.visibility_ceiling.value,
        exception_codes=tuple(code.value for code in manifest.exception_codes),
        source_bytes_read=False,
        catalog_written=False,
    )


def _dataset_storage_preflight_summary(
    value: DatasetStorageScopePreflight,
) -> DatasetStoragePreflightSummary:
    return DatasetStoragePreflightSummary(
        scope_id=value.scope.scope_id,
        scope_fingerprint=value.scope.fingerprint(),
        storage_root_id=value.scope.storage_root.object_id,
        access=value.access.value,
        ready=value.ready,
        observed_free_bytes=value.observed_free_bytes,
        effective_write_floor_bytes=value.effective_write_floor_bytes,
        reason_codes=value.reason_codes,
    )


def _dataset_operation_preview_summary(
    preview: DatasetOperationPreview,
) -> DatasetOperationPreviewSummary:
    authorization = preview.authorization
    return DatasetOperationPreviewSummary(
        preview_id=preview.preview_id,
        state=preview.state.value,
        manifest_id=preview.manifest.object_id,
        manifest_fingerprint=preview.manifest.object_fingerprint,
        policy_id=preview.policy.object_id,
        policy_fingerprint=preview.policy.object_fingerprint,
        request_id=preview.request.object_id,
        request_fingerprint=preview.request.object_fingerprint,
        authorization_id=None if authorization is None else authorization.object_id,
        authorization_fingerprint=(
            None if authorization is None else authorization.object_fingerprint
        ),
        decision=preview.decision.decision.value,
        expected_implementation_commit=preview.repository.expected_commit,
        observed_implementation_commit=preview.repository.observed_commit,
        repository_verified=preview.repository.verified,
        source_preflights=tuple(
            _dataset_storage_preflight_summary(value) for value in preview.source_preflights
        ),
        destination_preflight=(
            None
            if preview.destination_preflight is None
            else _dataset_storage_preflight_summary(preview.destination_preflight)
        ),
        control_preflight=_dataset_storage_preflight_summary(preview.control_preflight),
        reason_codes=preview.reason_codes,
        source_bytes_read=preview.source_bytes_read,
        network_bytes=preview.network_bytes,
        external_bytes_written=preview.external_bytes_written,
        catalog_written=preview.catalog_written,
        outcome_access=preview.gate_outcome_access.value,
    )


def _dataset_projection_rebuild_summary(
    installation: DatasetProjectionRebuildInstallation,
) -> DatasetProjectionRebuildSummary:
    return DatasetProjectionRebuildSummary(
        installation_id=installation.installation_id,
        receipt_id=installation.receipt.receipt_id,
        receipt_fingerprint=installation.receipt.fingerprint(),
        receipt_storage_root_id=installation.receipt_evidence.storage_root_id,
        receipt_relative_locator=installation.receipt_evidence.relative_locator,
        database_relative_path=installation.database_relative_path,
        database_sha256=installation.database_sha256,
        database_size_bytes=installation.database_size_bytes,
        projection_sha256=installation.projection_sha256,
        projection_state_fingerprint=installation.projection_state.fingerprint(),
        source_bytes_read=installation.source_bytes_read,
        network_bytes=installation.network_bytes,
        external_receipt_written=installation.external_receipt_written,
        catalog_written=installation.catalog_written,
        outcome_access=installation.outcome_access.value,
    )


def _dataset_registration_summary(
    installation: DatasetRegistrationInstallation,
) -> DatasetRegistrationSummary:
    receipt = installation.receipt
    materialization = installation.materialization
    return DatasetRegistrationSummary(
        receipt_id=receipt.receipt_id,
        receipt_fingerprint=receipt.fingerprint(),
        receipt_storage_root_id=installation.receipt_evidence.storage_root_id,
        receipt_relative_locator=installation.receipt_evidence.relative_locator,
        dataset_snapshot_fingerprint=installation.snapshot.fingerprint(),
        materialization_id=materialization.materialization_id,
        materialization_fingerprint=materialization.fingerprint(),
        release_id=materialization.release_id,
        custody_state=materialization.custody_state.value,
        storage_root_id=materialization.storage_root_id,
        relative_locator=materialization.relative_locator,
        distinct_source_bytes=receipt.distinct_source_bytes,
        source_full_hash_passes=receipt.source_full_hash_passes,
        source_full_hash_read_ceiling_bytes=receipt.source_full_hash_read_ceiling_bytes,
        network_bytes=installation.network_bytes,
        source_mutated=receipt.source_mutated,
        download_performed=receipt.download_performed,
        dataset_bytes_written=receipt.dataset_bytes_written,
        control_batch_published=receipt.control_batch_published,
        replayed_existing=installation.replayed_existing,
        new_external_artifacts_created=installation.new_external_artifacts_created,
        catalog_written=installation.catalog_written,
        outcome_access=materialization.outcome_access.value,
    )


def _immutable_dataset_metadata_value(value: object) -> object:
    if isinstance(value, list):
        return tuple(_immutable_dataset_metadata_value(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _immutable_dataset_metadata_value(item) for key, item in value.items()}
    return value


def _dataset_show_summary(record: DatasetCatalogRecord) -> DatasetShowSummary:
    payload = record.canonical_bytes()
    if len(payload) > MAX_DATASET_RECORD_CANONICAL_BYTES:
        raise ValueError("dataset metadata document exceeds its canonical byte limit")
    summary = _dataset_catalog_record_summary(record)
    if summary.record_kind not in _DATASET_SHOW_KINDS:
        raise ValueError("dataset show returned an unsupported record kind")
    metadata_document = _immutable_dataset_metadata_value(record.to_document())
    if not isinstance(metadata_document, Mapping):  # pragma: no cover - record contract
        raise TypeError("dataset metadata document must be a mapping")
    return DatasetShowSummary(
        record_id=summary.record_id,
        record_kind=summary.record_kind,
        schema=summary.schema,
        version=summary.version,
        fingerprint=summary.fingerprint,
        metadata_document=metadata_document,
        payload_read=False,
        source_bytes_read=False,
        outcome_read=False,
    )


def _dataset_catalog_error(
    operation: str,
    *,
    status: OperationStatus,
    category: ErrorCategory,
    reason_code: str,
    message: str,
) -> ApiResult[ResultT]:
    return ApiResult(
        operation=operation,
        status=status,
        errors=(
            ApiError(
                category=category,
                reason_code=reason_code,
                message=message,
            ),
        ),
        reason_codes=(reason_code,),
    )


def _requires_parent_record(graph: CandidateScientificGraph) -> bool:
    """Only substitution slots need a runtime parent; exact receipts are frozen inputs."""
    return any(
        value.scientific_role is ScientificInputRole.PARENT_RECEIPT
        and value.content_identity_policy is ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION
        for value in graph.external_inputs
    )


def _object_id(value: CanonicalRecord) -> str:
    for attribute in (
        "system_id",
        "experiment_id",
        "campaign_id",
        "template_id",
        "package_id",
        "draft_id",
        "candidate_id",
        "selection_id",
        "profile_id",
        "compilation_id",
        "report_id",
    ):
        candidate = getattr(value, attribute, None)
        if isinstance(candidate, str):
            return candidate
    raise TypeError(f"unsupported public record: {type(value).__name__}")


def _validation_error(operation: str, error: Exception) -> ApiResult[object]:
    reason_code = getattr(error, "reason_code", "AUTHORING_DOCUMENT_INVALID")
    return ApiResult(
        operation=operation,
        status=OperationStatus.INVALID,
        errors=(
            ApiError(
                category=ErrorCategory.VALIDATION,
                reason_code=reason_code,
                message=str(error),
                field=getattr(error, "field", None),
            ),
        ),
        reason_codes=(reason_code,),
    )


class DurableAuthorizationReplayError(AuthoringCodecError):
    """Stored follow-up authorization was absent or failed complete replay."""

    reason_code = "DURABLE_AUTHORIZATION_REPLAY_FAILED"


def _merge_discovery_catalogs(
    base: CandidateCapabilityCatalog | None,
    extension: CandidateCapabilityCatalog | None,
) -> CandidateCapabilityCatalog | None:
    """Union static discovery records without changing an executable provider."""

    if base is None:
        return extension
    if extension is None:
        return base
    registrations = {value.registration_id: value for value in base.registrations}
    for value in extension.registrations:
        previous = registrations.get(value.registration_id)
        if previous is not None and previous != value:
            raise ValueError(f"capability discovery conflict for {value.registration_id}")
        registrations[value.registration_id] = value
    return CandidateCapabilityCatalog(
        catalog_id="catalog.public-static-discovery",
        registrations=tuple(registrations[key] for key in sorted(registrations)),
        templates=base.templates,
    )


class EmpiricalLawhoodApi:
    """Dependency-composed public API; callers need no infrastructure objects."""

    def __init__(
        self,
        *,
        repo_root: Path,
        project_configured: bool = True,
        execution_service: CampaignExecutionService | None = None,
        prospective_workflow: ProspectiveWorkflow | None = None,
        approval_service: CompleteApprovalService | None = None,
        dataset_catalog_session_provider: DatasetCatalogSessionProvider | None = None,
        dataset_operation_preview_service: DatasetOperationPreviewService | None = None,
        public_source_acquisition_service: PublicSourceAcquisitionService | None = None,
        fair_mast_source_service: FairMastPublicSourceService | None = None,
        dataset_registration_installer: DatasetRegistrationInstallerPort | None = None,
        dataset_projection_rebuild_installer: DatasetProjectionRebuildInstallerPort | None = None,
        dataset_campaign_binding_resolver: DatasetCampaignBindingResolver | None = None,
        candidate_compilation_context: CandidateCompilationContext | None = None,
        candidate_context_provider: CandidateContextProvider | None = None,
        conditional_successor_resolver: ConditionalChildResolver | None = None,
        candidate_capability_catalog: CandidateCapabilityCatalog | None = None,
        extension_bundle_aggregate: GeneratedExtensionBundleAggregate | None = None,
        executable_binding_aggregate: GeneratedExecutableBindingAggregate | None = None,
        executable_factory_registry: ExecutableCapabilityProviderFactoryRegistry | None = None,
        study_bundle_registry: CapabilityRegistry | None = None,
        study_authority_store: StudyOperationAuthorityStore | None = None,
        study_issue_publisher: ExternalIssuedStudyPublisher | None = None,
        study_source_closure_inspector: StudySourceClosureInspector | None = None,
        study_issue_clock: DecisionClock | None = None,
        study_extension_codec_registry: CanonicalRecordCodecRegistry | None = None,
        source_profile_compiler: SourceProfileCompilerRegistry | None = None,
        linked_profile_compiler: LinkedCampaignProfileCompiler | None = None,
        inspection_storage_root: GuardedExternalRoot | None = None,
        inspection_storage_read_only: bool = False,
        external_root: Path | None,
    ) -> None:
        if candidate_compilation_context is not None and candidate_context_provider is not None:
            raise ValueError(
                "fixed candidate context and candidate context provider are mutually exclusive"
            )
        provider_catalog = (
            None if candidate_context_provider is None else candidate_context_provider.catalog
        )
        if (
            provider_catalog is not None
            and candidate_capability_catalog is not None
            and provider_catalog.fingerprint() != candidate_capability_catalog.fingerprint()
        ):
            raise ValueError("candidate provider and discovery catalog differ")
        self._repo_root = repo_root.resolve(strict=True)
        self._project_configured = project_configured
        self._execution_service = execution_service
        self._prospective_workflow = prospective_workflow
        self._approval_service = approval_service or (
            None if execution_service is None else execution_service.approval_service
        )
        self._dataset_catalog_session_provider = dataset_catalog_session_provider
        self._public_source_acquisition_service = public_source_acquisition_service
        self._fair_mast_source_service = fair_mast_source_service or FairMastPublicSourceService()
        self._dataset_operation_preview_service = dataset_operation_preview_service
        self._dataset_registration_installer = dataset_registration_installer
        self._dataset_projection_rebuild_installer = dataset_projection_rebuild_installer
        self._dataset_campaign_binding_resolver = dataset_campaign_binding_resolver
        self._candidate_compilation_context = candidate_compilation_context
        self._candidate_context_provider = candidate_context_provider
        self._conditional_successor_resolver = conditional_successor_resolver
        self._candidate_capability_catalog = candidate_capability_catalog or provider_catalog
        self._extension_bundle_aggregate = extension_bundle_aggregate
        if executable_binding_aggregate is not None and (
            extension_bundle_aggregate is None
            or executable_binding_aggregate.discovery_aggregate
            != ObjectIdentity.from_record(
                extension_bundle_aggregate.aggregate_id,
                extension_bundle_aggregate,
            )
        ):
            raise ValueError("executable and discovery aggregates differ")
        self._executable_binding_aggregate = executable_binding_aggregate
        if executable_factory_registry is not None and (
            executable_binding_aggregate is None
            or executable_factory_registry.aggregate != executable_binding_aggregate
        ):
            raise ValueError("executable factory registry differs from facade aggregate")
        self._executable_factory_registry = executable_factory_registry
        self._capability_discovery_catalog = _merge_discovery_catalogs(
            self._candidate_capability_catalog,
            (
                None
                if extension_bundle_aggregate is None
                else extension_bundle_aggregate.candidate_capability_catalog
            ),
        )
        self._study_bundle_registry = study_bundle_registry
        self._study_authority_store = study_authority_store
        self._study_issue_publisher = study_issue_publisher
        self._study_source_closure_inspector = study_source_closure_inspector
        self._study_issue_clock = study_issue_clock or SystemDecisionClock()
        self._study_extension_codec_registry = study_extension_codec_registry
        self._source_profile_compiler = source_profile_compiler
        self._linked_profile_compiler = linked_profile_compiler
        self._external_root = external_root
        self._inspection_storage_root = inspection_storage_root
        self._inspection_storage_read_only = inspection_storage_read_only

    @property
    def repo_root(self) -> Path:
        return self._repo_root

    def _catalog_path(self) -> Path:
        if self._execution_service is not None:
            return self._execution_service.catalog_path
        return production_catalog_path(self._repo_root)

    def validate_document(self, request: DocumentRequest) -> ApiResult[DocumentSummary]:
        operation = "document.validate"
        try:
            value = load_authoring(request.path)
        except FileNotFoundError as error:
            return ApiResult(
                operation=operation,
                status=OperationStatus.NOT_FOUND,
                errors=(
                    ApiError(
                        category=ErrorCategory.NOT_FOUND,
                        reason_code="AUTHORING_DOCUMENT_NOT_FOUND",
                        message=str(error),
                    ),
                ),
                reason_codes=("AUTHORING_DOCUMENT_NOT_FOUND",),
            )
        except (AuthoringCodecError, OSError, ValueError) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=DocumentSummary(
                object_id=_object_id(value),
                object_kind=type(value).__name__,
                schema=value.SCHEMA,
                version=value.VERSION,
                fingerprint=value.fingerprint(),
            ),
        )

    def validate_dataset_manifest(
        self,
        request: DatasetManifestRequest,
    ) -> ApiResult[DatasetManifestSummary]:
        operation = "dataset.validate"
        try:
            if not isinstance(request, DatasetManifestRequest) or not isinstance(
                request.path, Path
            ):
                raise ValueError("request must contain one manifest Path")
            value = load_registered_authoring(
                request.path,
                root_schemas=DATASET_MANIFEST_SCHEMAS,
            )
            if not isinstance(
                value,
                (
                    DatasetRegistrationManifest,
                    DatasetTransformationManifest,
                    ProposedExperimentDatasetBindingManifest,
                ),
            ):
                raise TypeError("unsupported dataset manifest type")
            validate_stable_id(value.manifest_id, field_name="manifest_id")
            summary = _dataset_manifest_summary(value)
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="DATASET_MANIFEST_NOT_FOUND",
                message=str(error),
            )
        except (AuthoringCodecError, OSError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="DATASET_MANIFEST_INVALID",
                message=str(error),
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=summary,
        )

    def preview_public_source_acquisition(
        self,
        request: PublicSourceAcquisitionRequest,
    ) -> ApiResult[PublicSourceAcquisitionSummary]:
        return self._public_source_operation(request, mode="preview")

    def preview_fair_mast_source(
        self, selection: FairMastPublicSelection
    ) -> dict[str, object]:
        return self._fair_mast_source_service.preview(selection)

    def acquire_fair_mast_source(
        self, selection: FairMastPublicSelection, *,
        acquisition_authority_id: str, custody_authority_id: str
    ) -> FairMastSourceReceipt:
        return self._fair_mast_source_service.acquire(
            selection,
            acquisition_authority_id=acquisition_authority_id,
            custody_authority_id=custody_authority_id,
        )

    def recover_fair_mast_source(
        self, selection: FairMastPublicSelection, *,
        acquisition_authority_id: str, custody_authority_id: str
    ) -> FairMastSourceReceipt:
        return self._fair_mast_source_service.acquire(
            selection,
            acquisition_authority_id=acquisition_authority_id,
            custody_authority_id=custody_authority_id,
            recover=True,
        )

    def bind_fair_mast_design_input(
        self,
        selection: FairMastPublicSelection,
        *,
        input_id: str,
        information_cutoff: InformationCutoff,
        operator_id: str,
        authoring_at_utc: str,
    ) -> DesignInputRecord:
        return self._fair_mast_source_service.design_input(
            selection,
            input_id=input_id,
            information_cutoff=information_cutoff,
            operator_id=operator_id,
            authoring_at_utc=authoring_at_utc,
        )

    def acquire_public_source(
        self,
        request: PublicSourceAcquisitionRequest,
    ) -> ApiResult[PublicSourceAcquisitionSummary]:
        return self._public_source_operation(request, mode="acquire")

    def recover_public_source(
        self,
        request: PublicSourceAcquisitionRequest,
    ) -> ApiResult[PublicSourceAcquisitionSummary]:
        return self._public_source_operation(request, mode="recover")

    def _public_source_operation(
        self,
        request: PublicSourceAcquisitionRequest,
        *,
        mode: str,
    ) -> ApiResult[PublicSourceAcquisitionSummary]:
        operation = f"public-source.{mode}"

        def failure(
            status: OperationStatus, category: ErrorCategory, code: str
        ) -> ApiResult[PublicSourceAcquisitionSummary]:
            # Transport exceptions can contain signed redirect URLs. Never echo them.
            return ApiResult(
                operation=operation,
                status=status,
                errors=(ApiError(category=category, reason_code=code, message=code),),
                reason_codes=(code,),
            )

        try:
            if (
                not isinstance(request, PublicSourceAcquisitionRequest)
                or type(request.confirmed) is not bool
            ):
                raise TypeError("invalid source acquisition request")
            invocation = load_registered_authoring(
                request.invocation_path,
                root_schemas={
                    ChecksummedSourceAcquisitionInvocation.SCHEMA: ChecksummedSourceAcquisitionInvocation,
                    CaptureAssuredSourceAcquisitionInvocation.SCHEMA: CaptureAssuredSourceAcquisitionInvocation,
                },
            )
            if not isinstance(invocation, ChecksummedSourceAcquisitionInvocation):
                raise TypeError("invalid acquisition invocation")
        except FileNotFoundError:
            return failure(
                OperationStatus.NOT_FOUND, ErrorCategory.NOT_FOUND, "SOURCE_INVOCATION_NOT_FOUND"
            )
        except (AuthoringCodecError, OSError, TypeError, ValueError):
            return failure(
                OperationStatus.INVALID, ErrorCategory.VALIDATION, "SOURCE_INVOCATION_INVALID"
            )
        service = self._public_source_acquisition_service
        if service is None:
            return failure(
                OperationStatus.BLOCKED, ErrorCategory.CAPABILITY, "SOURCE_ACQUISITION_NOT_COMPOSED"
            )
        try:
            preview = service.preview(invocation)
            effects = (
                ("network:finite-public-source-selection", "external:attempt-raw-transfer-custody")
                if mode == "acquire"
                else ("external:same-identity-custody-recovery",)
                if mode == "recover"
                else ()
            )
            summary = PublicSourceAcquisitionSummary(
                preview.plan.object_id,
                preview.plan.object_fingerprint,
                preview.member_count,
                preview.declared_transfer_bytes,
                preview.maximum_attempts,
                mode == "acquire",
                effects,
            )
            if mode != "preview" and not request.confirmed:
                return ApiResult(
                    operation=operation,
                    status=OperationStatus.BLOCKED,
                    payload=summary,
                    reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
                )
            if mode == "preview":
                return ApiResult(
                    operation=operation, status=OperationStatus.SUCCEEDED, payload=summary
                )
            receipts = (
                service.acquire(invocation) if mode == "acquire" else service.recover(invocation)
            )
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=replace(
                    summary,
                    receipts=tuple(
                        PublicSourceCustodySummary(
                            r.receipt_id,
                            r.fingerprint(),
                            r.raw_artifact.sha256,
                            r.raw_artifact.size_bytes,
                            r.raw_relative_locator,
                        )
                        for r in receipts
                    ),
                ),
            )
        except (KeyError, PermissionError):
            return failure(
                OperationStatus.BLOCKED,
                ErrorCategory.AUTHORITY,
                "SOURCE_AUTHORITY_OR_IDENTITY_REFUSED",
            )
        except (TypeError, ValueError):
            return failure(
                OperationStatus.INVALID, ErrorCategory.VALIDATION, "SOURCE_CONTRACT_REFUSED"
            )
        except (OSError, RuntimeError):
            return failure(
                OperationStatus.FAILED,
                ErrorCategory.EXECUTION,
                "SOURCE_OPERATION_FAILED_RECOVER_SAME_IDENTITY",
            )

    def preview_dataset_operation(
        self,
        request: DatasetOperationPreviewRequest,
    ) -> ApiResult[DatasetOperationPreviewSummary]:
        """Preview exact authority/storage/identity gates without performing work."""

        operation = "dataset.preview"
        try:
            if not isinstance(request, DatasetOperationPreviewRequest):
                raise TypeError("request must be a DatasetOperationPreviewRequest")
            manifest = load_registered_authoring(
                request.manifest_path,
                root_schemas=DATASET_MANIFEST_SCHEMAS,
            )
            if not isinstance(
                manifest,
                (
                    DatasetRegistrationManifest,
                    DatasetTransformationManifest,
                    ProposedExperimentDatasetBindingManifest,
                ),
            ):
                raise TypeError("unsupported dataset operation manifest")
            policy: CanonicalRecord = load_registered_authoring(
                request.policy_path,
                root_schemas=DATASET_OPERATION_AUTHORITY_SCHEMAS,
            )
            authored_request: CanonicalRecord = load_registered_authoring(
                request.request_path,
                root_schemas=DATASET_OPERATION_AUTHORITY_SCHEMAS,
            )
            authorization = (
                None
                if request.authorization_path is None
                else load_registered_authoring(
                    request.authorization_path,
                    root_schemas=DATASET_OPERATION_AUTHORITY_SCHEMAS,
                )
            )
            if not isinstance(policy, DatasetOperationPolicy):
                raise TypeError("policy document must contain DatasetOperationPolicy")
            if not isinstance(authored_request, DatasetOperationRequest):
                raise TypeError("request document must contain DatasetOperationRequest")
            if authorization is not None and not isinstance(
                authorization,
                DatasetOperationAuthorization,
            ):
                raise TypeError("authorization document must contain DatasetOperationAuthorization")
            validate_stable_id(request.authorization_id, field_name="authorization_id")
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="DATASET_OPERATION_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except (AuthoringCodecError, OSError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="DATASET_OPERATION_PREVIEW_INVALID",
                message=str(error),
            )
        service = self._dataset_operation_preview_service
        if service is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="DATASET_OPERATION_TRUST_CONFIGURATION_REQUIRED",
                message=(
                    "no trusted dataset policy/issuer/storage preview composition is installed"
                ),
            )
        try:
            preview = service.preview(
                manifest=manifest,
                policy=policy,
                request=authored_request,
                authorization_id=request.authorization_id,
                authorization=authorization,
            )
            summary = _dataset_operation_preview_summary(preview)
        except KeyError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="DATASET_OPERATION_TRUST_IDENTITY_UNAVAILABLE",
                message=str(error),
            )
        except PermissionError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="DATASET_OPERATION_AUTHORIZATION_REPLAY_FAILED",
                message=str(error),
            )
        except (TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="DATASET_OPERATION_PREVIEW_INVALID",
                message=str(error),
            )
        if preview.state is DatasetOperationPreviewState.READY:
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=summary,
            )
        category = (
            ErrorCategory.IDENTITY
            if preview.state is DatasetOperationPreviewState.IDENTITY_BLOCKED
            else ErrorCategory.STORAGE
            if preview.state is DatasetOperationPreviewState.STORAGE_BLOCKED
            else ErrorCategory.AUTHORITY
        )
        reason_code = {
            DatasetOperationPreviewState.AUTHORITY_REQUIRED: (
                "DATASET_OPERATION_AUTHORITY_REQUIRED"
            ),
            DatasetOperationPreviewState.REFUSED: "DATASET_OPERATION_REFUSED",
            DatasetOperationPreviewState.IDENTITY_BLOCKED: ("DATASET_OPERATION_IDENTITY_BLOCKED"),
            DatasetOperationPreviewState.STORAGE_BLOCKED: ("DATASET_OPERATION_STORAGE_BLOCKED"),
        }[preview.state]
        return ApiResult(
            operation=operation,
            status=OperationStatus.BLOCKED,
            payload=summary,
            errors=(
                ApiError(
                    category=category,
                    reason_code=reason_code,
                    message="dataset operation preview did not reach replay-verified readiness",
                ),
            ),
            reason_codes=preview.reason_codes,
        )

    def register_dataset(
        self,
        request: DatasetRegistrationRequest,
    ) -> ApiResult[DatasetRegistrationSummary | WriteEffectSummary]:
        """Verify and durably register one exact already-held dataset source."""

        operation = "dataset.register"
        try:
            if not isinstance(request, DatasetRegistrationRequest):
                raise TypeError("request must be a DatasetRegistrationRequest")
            manifest = load_registered_authoring(
                request.manifest_path,
                root_schemas=DATASET_MANIFEST_SCHEMAS,
            )
            policy = load_registered_authoring(
                request.policy_path,
                root_schemas=DATASET_OPERATION_AUTHORITY_SCHEMAS,
            )
            authored_request = load_registered_authoring(
                request.request_path,
                root_schemas=DATASET_OPERATION_AUTHORITY_SCHEMAS,
            )
            authorization = load_registered_authoring(
                request.authorization_path,
                root_schemas=DATASET_OPERATION_AUTHORITY_SCHEMAS,
            )
            if not isinstance(manifest, DatasetRegistrationManifest):
                raise TypeError("manifest document must contain registration manifest")
            if not isinstance(policy, DatasetOperationPolicy):
                raise TypeError("policy document must contain DatasetOperationPolicy")
            if not isinstance(authored_request, DatasetOperationRequest):
                raise TypeError("request document must contain DatasetOperationRequest")
            if not isinstance(authorization, DatasetOperationAuthorization):
                raise TypeError("authorization document must contain DatasetOperationAuthorization")
            validate_stable_id(request.bundle_id, field_name="bundle_id")
            validate_stable_id(request.receipt_id, field_name="receipt_id")
            bundle = DatasetOperationAuthorityBundle(
                bundle_id=request.bundle_id,
                policy=policy,
                request=authored_request,
                authorization=authorization,
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="DATASET_REGISTRATION_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except (AuthoringCodecError, OSError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="DATASET_REGISTRATION_INVALID",
                message=str(error),
            )
        effects = ("external:verified-dataset-registration-control-batch",)
        if not request.confirmed:
            return ApiResult(
                operation=operation,
                status=OperationStatus.BLOCKED,
                payload=WriteEffectSummary(
                    action=operation,
                    confirmed=False,
                    effects=effects,
                ),
                reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
            )
        installer = self._dataset_registration_installer
        if installer is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="DATASET_REGISTRATION_TRUST_CONFIGURATION_REQUIRED",
                message="no trusted dataset registration installer is composed",
            )
        try:
            installation = installer.install(
                bundle,
                manifest=manifest,
                receipt_id=request.receipt_id,
            )
            summary = _dataset_registration_summary(installation)
        except (KeyError, PermissionError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="DATASET_REGISTRATION_AUTHORIZATION_REPLAY_FAILED",
                message=str(error),
            )
        except (TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.IDENTITY,
                reason_code="DATASET_REGISTRATION_IDENTITY_MISMATCH",
                message=str(error),
            )
        except (ArtifactPlaneError, BoundedFileIOError, OSError, RuntimeError) as error:
            return self._storage_error(
                operation,
                "DATASET_REGISTRATION_FAILED",
                error,
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=summary,
        )

    def rebuild_dataset_projection(
        self,
        request: DatasetProjectionRebuildRequest,
    ) -> ApiResult[DatasetProjectionRebuildSummary | WriteEffectSummary]:
        """Install one exact unified projection only through dedicated authority."""

        operation = "dataset.rebuild"
        try:
            if not isinstance(request, DatasetProjectionRebuildRequest):
                raise TypeError("request must be a DatasetProjectionRebuildRequest")
            manifest = load_registered_authoring(
                request.manifest_path,
                root_schemas=DATASET_PROJECTION_REBUILD_SCHEMAS,
            )
            policy = load_registered_authoring(
                request.policy_path,
                root_schemas=DATASET_PROJECTION_REBUILD_SCHEMAS,
            )
            authored_request = load_registered_authoring(
                request.request_path,
                root_schemas=DATASET_PROJECTION_REBUILD_SCHEMAS,
            )
            authorization = load_registered_authoring(
                request.authorization_path,
                root_schemas=DATASET_PROJECTION_REBUILD_SCHEMAS,
            )
            if not isinstance(manifest, DatasetProjectionRebuildManifest):
                raise TypeError("manifest document must contain rebuild manifest")
            if not isinstance(policy, DatasetProjectionRebuildPolicy):
                raise TypeError("policy document must contain rebuild policy")
            if not isinstance(
                authored_request,
                AuthoredDatasetProjectionRebuildRequest,
            ):
                raise TypeError("request document must contain rebuild request")
            if not isinstance(authorization, DatasetProjectionRebuildAuthorization):
                raise TypeError("authorization document must contain rebuild authorization")
            validate_stable_id(request.bundle_id, field_name="bundle_id")
            validate_stable_id(request.receipt_id, field_name="receipt_id")
            bundle = DatasetProjectionRebuildAuthorityBundle(
                bundle_id=request.bundle_id,
                manifest=manifest,
                policy=policy,
                request=authored_request,
                authorization=authorization,
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="DATASET_PROJECTION_REBUILD_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except (AuthoringCodecError, OSError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="DATASET_PROJECTION_REBUILD_INVALID",
                message=str(error),
            )
        effects = (
            "external:authorized-dataset-projection-receipt",
            "local:.empirical-lawhood/experiment_catalog.sqlite3",
        )
        if not request.confirmed:
            return ApiResult(
                operation=operation,
                status=OperationStatus.BLOCKED,
                payload=WriteEffectSummary(
                    action=operation,
                    confirmed=False,
                    effects=effects,
                ),
                reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
            )
        installer = self._dataset_projection_rebuild_installer
        if installer is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="DATASET_PROJECTION_REBUILD_TRUST_CONFIGURATION_REQUIRED",
                message="no trusted dataset projection rebuild installer is composed",
            )
        try:
            installation = installer.install(bundle, receipt_id=request.receipt_id)
            summary = _dataset_projection_rebuild_summary(installation)
        except (KeyError, PermissionError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="DATASET_PROJECTION_REBUILD_AUTHORIZATION_REPLAY_FAILED",
                message=str(error),
            )
        except (TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.IDENTITY,
                reason_code="DATASET_PROJECTION_REBUILD_IDENTITY_MISMATCH",
                message=str(error),
            )
        except (ArtifactPlaneError, CatalogRebuildError, OSError, RuntimeError) as error:
            return self._storage_error(
                operation,
                "DATASET_PROJECTION_REBUILD_FAILED",
                error,
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=summary,
        )

    def validate_system_document(
        self,
        request: DocumentRequest,
    ) -> ApiResult[DocumentSummary]:
        return self._validate_document_kind(
            request,
            operation="system.validate",
            allowed=(SystemSpec,),
        )

    def validate_campaign_document(
        self,
        request: DocumentRequest,
    ) -> ApiResult[DocumentSummary]:
        from empirical_lawhood.planning.campaigns import CampaignSpec

        return self._validate_document_kind(
            request,
            operation="campaign.validate",
            allowed=(
                CampaignSpec,
                EvidenceProfileSelection,
                SourcePipelineProfile,
                SourcePipelineCompilation,
                LinkedCampaignProfile,
                LinkedCampaignExecutableCompilation,
                StudyDraft,
                StudyDefinition,
                DraftStudyCandidate,
                StudyCandidate,
                ExecutableStudyCompilationReport,
                StudyBundleSpec,
                StudyBundleCompilationInput,
                StudyBundleCandidate,
                StudyBundleCompilationReport,
                CampaignPackage,
            ),
            validator=self._validate_campaign_root,
        )

    def compile_source_profile(
        self,
        request: CompileSourceProfileRequest,
    ) -> ApiResult[SourceProfileCompilationSummary]:
        """Compile one strict source profile without opening its source."""

        operation = "campaign.source-profile-compile"
        compiler = self._source_profile_compiler
        if compiler is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code="SOURCE_PROFILE_COMPILER_REQUIRED",
                message="source-profile compilation is not composed",
            )
        try:
            if not isinstance(request, CompileSourceProfileRequest):
                raise TypeError("request must be a CompileSourceProfileRequest")
            profile = load_registered_authoring(
                request.source_profile_path,
                root_schemas={SourcePipelineProfile.SCHEMA: SourcePipelineProfile},
            )
            transformation = load_registered_authoring(
                request.transformation_manifest_path,
                root_schemas={
                    DatasetTransformationManifest.SCHEMA: DatasetTransformationManifest,
                },
            )
            dataset_request = (
                None
                if request.dataset_operation_request_path is None
                else load_registered_authoring(
                    request.dataset_operation_request_path,
                    root_schemas={DatasetOperationRequest.SCHEMA: DatasetOperationRequest},
                )
            )
            assert isinstance(profile, SourcePipelineProfile)
            assert isinstance(transformation, DatasetTransformationManifest)
            assert dataset_request is None or isinstance(dataset_request, DatasetOperationRequest)
            compilation = compiler.compile(
                profile=profile,
                transformation_manifest=transformation,
                dataset_operation_request=dataset_request,
            )
            relative_path: str | None = None
            payload_sha256: str | None = None
            local_bytes_written = 0
            if request.emit_path is not None:
                payload = compilation.canonical_bytes()
                relative_path = self._emit_local_candidate(
                    draft_path=request.source_profile_path,
                    requested_path=request.emit_path,
                    payload=payload,
                )
                payload_sha256 = hashlib.sha256(payload).hexdigest()
                local_bytes_written = len(payload)
            summary = SourceProfileCompilationSummary(
                profile_id=profile.profile_id,
                profile_fingerprint=profile.fingerprint(),
                compilation_id=compilation.compilation_id,
                compilation_fingerprint=compilation.fingerprint(),
                disposition=compilation.disposition.value,
                reason_codes=compilation.reason_codes,
                source_read=False,
                authority_granted=False,
                local_compilation_emitted=relative_path is not None,
                local_compilation_relative_path=relative_path,
                local_compilation_sha256=payload_sha256,
                local_bytes_written=local_bytes_written,
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="SOURCE_PROFILE_INPUT_NOT_FOUND",
                message=str(error),
            )
        except FileExistsError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.CONFLICT,
                category=ErrorCategory.CONFLICT,
                reason_code="SOURCE_PROFILE_COMPILATION_NO_REPLACE",
                message=str(error),
            )
        except ExecutableBindingRequiredError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code="EXECUTABLE_BINDING_REQUIRED",
                message=str(error),
            )
        except (AuthoringCodecError, TypeError, ValueError) as error:
            return cast(
                ApiResult[SourceProfileCompilationSummary],
                _validation_error(operation, error),
            )
        except OSError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.FAILED,
                category=ErrorCategory.STORAGE,
                reason_code="SOURCE_PROFILE_COMPILATION_EMISSION_FAILED",
                message=str(error),
            )
        if (
            compilation.disposition
            is SourcePipelineCompilationDisposition.REQUESTS_COMPILED_AUTHORITY_PENDING
        ):
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=summary,
                reason_codes=("COMPILED_AUTHORITY_PENDING",),
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.BLOCKED,
            payload=summary,
            errors=(
                ApiError(
                    category=ErrorCategory.AUTHORITY,
                    reason_code="AUTHORITY_REQUIRED",
                    message="source profile compiled but dataset-operation authority is absent",
                ),
            ),
            reason_codes=compilation.reason_codes,
        )

    def compile_linked_campaign_profile(
        self,
        request: CompileLinkedCampaignProfileRequest,
    ) -> ApiResult[LinkedCampaignCompilationSummary]:
        """Join four already-compiled candidates under one linked profile."""

        operation = "campaign.linked-profile-compile"
        compiler = self._linked_profile_compiler
        if compiler is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code="LINKED_PROFILE_COMPILER_REQUIRED",
                message="linked-profile compilation is not composed",
            )
        try:
            if not isinstance(request, CompileLinkedCampaignProfileRequest):
                raise TypeError("request must be a CompileLinkedCampaignProfileRequest")
            profile = load_registered_authoring(
                request.linked_profile_path,
                root_schemas={LinkedCampaignProfile.SCHEMA: LinkedCampaignProfile},
            )
            source = load_registered_authoring(
                request.source_compilation_path,
                root_schemas={SourcePipelineCompilation.SCHEMA: SourcePipelineCompilation},
            )
            evidence = load_registered_authoring(
                request.evidence_profile_selection_path,
                root_schemas={EvidenceProfileSelection.SCHEMA: EvidenceProfileSelection},
            )
            if len(request.candidate_report_paths) != 4:
                raise ValueError("linked profile requires exactly four candidate report paths")
            reports = tuple(
                sorted(
                    (
                        load_registered_authoring(
                            path,
                            root_schemas={
                                ExecutableStudyCompilationReport.SCHEMA: (
                                    ExecutableStudyCompilationReport
                                )
                            },
                        )
                        for path in request.candidate_report_paths
                    ),
                    key=lambda value: value.report_id,
                )
            )
            assert isinstance(profile, LinkedCampaignProfile)
            assert isinstance(source, SourcePipelineCompilation)
            assert isinstance(evidence, EvidenceProfileSelection)
            assert all(isinstance(value, ExecutableStudyCompilationReport) for value in reports)
            compilation = compiler.compile(
                profile=profile,
                source_compilation=source,
                evidence_selection=evidence,
                candidate_reports=reports,
            )
            relative_path: str | None = None
            payload_sha256: str | None = None
            local_bytes_written = 0
            if request.emit_path is not None:
                payload = compilation.canonical_bytes()
                relative_path = self._emit_local_candidate(
                    draft_path=request.linked_profile_path,
                    requested_path=request.emit_path,
                    payload=payload,
                )
                payload_sha256 = hashlib.sha256(payload).hexdigest()
                local_bytes_written = len(payload)
            summary = LinkedCampaignCompilationSummary(
                profile_id=profile.profile_id,
                profile_fingerprint=profile.fingerprint(),
                compilation_id=compilation.compilation_id,
                compilation_fingerprint=compilation.fingerprint(),
                disposition=compilation.disposition.value,
                reason_codes=compilation.reason_codes,
                package_roles=tuple(
                    sorted(value.role.value for value in compilation.package_bindings)
                ),
                provider_registry_fingerprints=compilation.provider_registry_fingerprints,
                authority_granted=compilation.grants_authority,
                executed=compilation.executed,
                local_compilation_emitted=relative_path is not None,
                local_compilation_relative_path=relative_path,
                local_compilation_sha256=payload_sha256,
                local_bytes_written=local_bytes_written,
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="LINKED_PROFILE_INPUT_NOT_FOUND",
                message=str(error),
            )
        except FileExistsError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.CONFLICT,
                category=ErrorCategory.CONFLICT,
                reason_code="LINKED_PROFILE_COMPILATION_NO_REPLACE",
                message=str(error),
            )
        except ExecutableBindingRequiredError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code="EXECUTABLE_BINDING_REQUIRED",
                message=str(error),
            )
        except (AuthoringCodecError, TypeError, ValueError) as error:
            return cast(
                ApiResult[LinkedCampaignCompilationSummary],
                _validation_error(operation, error),
            )
        except OSError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.FAILED,
                category=ErrorCategory.STORAGE,
                reason_code="LINKED_PROFILE_COMPILATION_EMISSION_FAILED",
                message=str(error),
            )
        if (
            compilation.disposition
            is LinkedCampaignExecutableDisposition.COMPILED_AUTHORITY_PENDING
        ):
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=summary,
                reason_codes=("COMPILED_AUTHORITY_PENDING",),
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.BLOCKED,
            payload=summary,
            errors=(
                ApiError(
                    category=ErrorCategory.CAPABILITY,
                    reason_code=compilation.disposition.value,
                    message="linked profile cannot reach executable authority-pending state",
                ),
            ),
            reason_codes=compilation.reason_codes,
        )

    def assemble_package(
        self,
        request: AssembleExperimentPackageRequest,
    ) -> ApiResult[ExperimentPackageAssemblySummary]:
        """Assemble the current issued package only after exact authority replay."""

        operation = "campaign.assemble-package"
        if not isinstance(request, AssembleExperimentPackageRequest):
            return _validation_error(operation, TypeError("invalid AssembleExperimentPackageRequest"))
        if request.scientific_approval_path is None or request.execution_authority_path is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="AUTHORITY_REQUIRED",
                message="scientific approval and execution authority must be supplied separately",
            )
        if self._approval_service is None or self._study_authority_store is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="AUTHORITY_REPLAY_SERVICE_REQUIRED",
                message="trusted approval and programme-authority stores are not composed",
            )
        try:
            base = load_registered_authoring(
                request.base_package_path,
                root_schemas={
                    IssuedStudyPackage.SCHEMA: IssuedStudyPackage,
                    RetrospectiveCampaignBase.SCHEMA: RetrospectiveCampaignBase,
                },
            )
            issued = load_standard_issued_study(request.issued_study_path)
            publication: StudyPublicationReceipt | ExtensionPublicationReceipt = (
                load_registered_authoring(
                    request.publication_receipt_path,
                    root_schemas={
                        StudyPublicationReceipt.SCHEMA: StudyPublicationReceipt,
                        ExtensionPublicationReceipt.SCHEMA: (
                            ExtensionPublicationReceipt
                        ),
                        RetrospectivePublicationReceipt.SCHEMA: RetrospectivePublicationReceipt,
                    },
                )
            )
            frozen = load_registered_authoring(
                request.frozen_proposal_path,
                root_schemas={
                    FrozenIssuedStudyApprovalProposal.SCHEMA: FrozenIssuedStudyApprovalProposal,
                    FrozenRetrospectiveApproval.SCHEMA: FrozenRetrospectiveApproval,
                },
            )
            approval = load_registered_authoring(
                request.scientific_approval_path,
                root_schemas={DurableAuthorizationRecord.SCHEMA: DurableAuthorizationRecord},
            )
            authority = load_registered_authoring(
                request.execution_authority_path,
                root_schemas={StudyOperationAuthority.SCHEMA: StudyOperationAuthority},
            )
            envelope = load_registered_authoring(
                request.execution_resource_envelope_spec_path,
                root_schemas={
                    ExecutionResourceEnvelopeSpec.SCHEMA: ExecutionResourceEnvelopeSpec,
                },
            )
            census = (
                None
                if request.predevelopment_jit_signature_census_path is None
                else load_registered_authoring(
                    request.predevelopment_jit_signature_census_path,
                    root_schemas={
                        PredevelopmentJitSignatureCensus.SCHEMA: (
                            PredevelopmentJitSignatureCensus
                        ),
                    },
                )
            )
            jit = (
                None
                if request.jit_graph_signature_manifest_path is None
                else load_registered_authoring(
                    request.jit_graph_signature_manifest_path,
                    root_schemas={JitGraphSignatureManifest.SCHEMA: JitGraphSignatureManifest},
                )
            )
            assert isinstance(base, IssuedStudyPackage)
            assert isinstance(issued, IssuedExecutableStudyManifest)
            assert isinstance(
                publication,
                (StudyPublicationReceipt, ExtensionPublicationReceipt),
            )
            assert isinstance(frozen, FrozenIssuedStudyApprovalProposal)
            assert isinstance(approval, DurableAuthorizationRecord)
            assert isinstance(authority, StudyOperationAuthority)
            assert isinstance(envelope, ExecutionResourceEnvelopeSpec)
            assert census is None or isinstance(census, PredevelopmentJitSignatureCensus)
            assert jit is None or isinstance(jit, JitGraphSignatureManifest)
            trusted_authority = self._study_authority_store.load(authority.authority_id)
            if trusted_authority != authority or ObjectIdentity.from_record(
                trusted_authority.authority_id,
                trusted_authority,
            ) != ObjectIdentity.from_record(authority.authority_id, authority):
                raise DurableAuthorizationReplayError(
                    "programme execution-authority store returned a substitution"
                )
            package = assemble_experiment_package(
                base=base,
                issued_study=issued,
                publication_receipt=publication,
                frozen_proposal=frozen,
                scientific_approval=approval,
                execution_authority=trusted_authority,
                execution_resource_envelope_spec=envelope,
                predevelopment_jit_signature_census=census,
                jit_graph_signature_manifest=jit,
                run_plan_id=request.run_plan_id,
                grantee_id=request.grantee_id,
                at_utc=request.at_utc,
            )
            self._replay_campaign_authorization(package)
            relative_path: str | None = None
            payload_sha256: str | None = None
            local_bytes_written = 0
            if request.emit_path is not None:
                payload = package.canonical_bytes()
                relative_path = self._emit_local_candidate(
                    draft_path=request.base_package_path,
                    requested_path=request.emit_path,
                    payload=payload,
                )
                payload_sha256 = hashlib.sha256(payload).hexdigest()
                local_bytes_written = len(payload)
            summary = ExperimentPackageAssemblySummary(
                package_id=package.package_id,
                package_fingerprint=package.fingerprint(),
                run_plan_id=package.run_plan_id,
                execution_plan_id=package.execution_plan_id,
                registry_fingerprint=package.registry.fingerprint(),
                authority_replayed=True,
                executed=False,
                local_package_emitted=relative_path is not None,
                local_package_relative_path=relative_path,
                local_package_sha256=payload_sha256,
                local_bytes_written=local_bytes_written,
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code='EXECUTION_PACKAGE_ASSEMBLY_INPUT_NOT_FOUND',
                message=str(error),
            )
        except FileExistsError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.CONFLICT,
                category=ErrorCategory.CONFLICT,
                reason_code='EXECUTION_PACKAGE_ASSEMBLY_NO_REPLACE',
                message=str(error),
            )
        except KeyError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="AUTHORITY_REQUIRED",
                message=str(error),
            )
        except (AuthoringCodecError, TypeError, ValueError) as error:
            return cast(
                ApiResult[ExperimentPackageAssemblySummary],
                _validation_error(operation, error),
            )
        except OSError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.FAILED,
                category=ErrorCategory.STORAGE,
                reason_code='EXECUTION_PACKAGE_ASSEMBLY_EMISSION_FAILED',
                message=str(error),
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=summary,
            reason_codes=('EXECUTION_PACKAGE_ASSEMBLED_AUTHORITY_REPLAYED',),
        )

    def bind_elapsed_budget(
        self,
        request: BindElapsedBudgetRequest,
    ) -> ApiResult[ExperimentPackageAssemblySummary]:
        """Bind the durable cumulative-time contract to an authorized ExperimentPackage."""

        operation = "campaign.bind-elapsed-budget"
        if not isinstance(request, BindElapsedBudgetRequest):
            return _validation_error(operation, TypeError("invalid BindElapsedBudgetRequest"))
        try:
            execution_package = load_campaign_package_authoring(request.execution_package_path)
            if not isinstance(execution_package, ExperimentPackage):
                raise AuthoringCodecError("elapsed-budget binding requires an ExperimentPackage")
            budget = load_registered_authoring(
                request.campaign_elapsed_budget_path,
                root_schemas={CampaignElapsedBudgetSpec.SCHEMA: CampaignElapsedBudgetSpec},
            )
            reservations = load_registered_authoring(
                request.campaign_elapsed_reservation_plan_path,
                root_schemas={
                    CampaignElapsedReservationPlan.SCHEMA: CampaignElapsedReservationPlan,
                },
            )
            assert isinstance(execution_package, ExperimentPackage)
            assert isinstance(budget, CampaignElapsedBudgetSpec)
            assert isinstance(reservations, CampaignElapsedReservationPlan)
            self._replay_campaign_authorization(execution_package)
            package = bind_issued_execution_package_elapsed_budget(
                execution_package=execution_package,
                campaign_elapsed_budget=budget,
                campaign_elapsed_reservation_plan=reservations,
            )
            relative_path: str | None = None
            payload_sha256: str | None = None
            local_bytes_written = 0
            if request.emit_path is not None:
                payload = package.canonical_bytes()
                relative_path = self._emit_local_candidate(
                    draft_path=request.execution_package_path,
                    requested_path=request.emit_path,
                    payload=payload,
                )
                payload_sha256 = hashlib.sha256(payload).hexdigest()
                local_bytes_written = len(payload)
            summary = ExperimentPackageAssemblySummary(
                package_id=package.package_id,
                package_fingerprint=package.fingerprint(),
                run_plan_id=package.run_plan_id,
                execution_plan_id=package.execution_plan_id,
                registry_fingerprint=package.registry.fingerprint(),
                authority_replayed=True,
                executed=False,
                local_package_emitted=relative_path is not None,
                local_package_relative_path=relative_path,
                local_package_sha256=payload_sha256,
                local_bytes_written=local_bytes_written,
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code='ELAPSED_BUDGET_BINDING_INPUT_NOT_FOUND',
                message=str(error),
            )
        except FileExistsError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.CONFLICT,
                category=ErrorCategory.CONFLICT,
                reason_code='ELAPSED_BUDGET_BINDING_NO_REPLACE',
                message=str(error),
            )
        except KeyError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="AUTHORITY_REQUIRED",
                message=str(error),
            )
        except (AuthoringCodecError, TypeError, ValueError) as error:
            return cast(
                ApiResult[ExperimentPackageAssemblySummary],
                _validation_error(operation, error),
            )
        except OSError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.FAILED,
                category=ErrorCategory.STORAGE,
                reason_code='ELAPSED_BUDGET_BINDING_EMISSION_FAILED',
                message=str(error),
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=summary,
            reason_codes=('EXECUTION_PACKAGE_ELAPSED_CONTRACT_BOUND',),
        )

    def list_capabilities(
        self,
        request: CapabilityListRequest,
    ) -> ApiResult[CapabilityListSummary]:
        """List a bounded static capability page without loading implementations."""

        operation = "capability.list"
        catalog = self._capability_discovery_catalog
        if catalog is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code=CandidateDiagnosticCode.CAPABILITY_REQUIRED.value,
                message="static capability discovery is not composed",
            )
        try:
            if not isinstance(request, CapabilityListRequest):
                raise TypeError("request must be a CapabilityListRequest")
            if (
                not isinstance(request.limit, int)
                or isinstance(request.limit, bool)
                or not 1 <= request.limit <= MAX_CAPABILITY_PAGE_RECORDS
            ):
                raise ValueError(f"limit must be in [1, {MAX_CAPABILITY_PAGE_RECORDS}]")
            if request.kind is not None and not isinstance(request.kind, CapabilityKind):
                raise ValueError("kind must be a CapabilityKind")
            catalog_sha256 = catalog.fingerprint()
            after_id = (
                None
                if request.cursor is None
                else _decode_capability_cursor(
                    request.cursor,
                    catalog_sha256=catalog_sha256,
                    kind=request.kind,
                )
            )
            registrations = tuple(
                value
                for value in catalog.registrations
                if (request.kind is None or value.manifest.kind is request.kind)
                and (after_id is None or value.registration_id > after_id)
            )
            page = registrations[: request.limit]
            has_more = len(registrations) > len(page)
            next_cursor = (
                _capability_cursor(
                    after_registration_id=page[-1].registration_id,
                    catalog_sha256=catalog_sha256,
                    kind=request.kind,
                )
                if has_more and page
                else None
            )
            summary = CapabilityListSummary(
                catalog_id=catalog.catalog_id,
                catalog_sha256=catalog_sha256,
                registrations=page,
                bundle_bindings=tuple(
                    binding
                    for registration in page
                    if self._extension_bundle_aggregate is not None
                    for binding in (
                        self._extension_bundle_aggregate.binding(
                            registration.manifest.capability_key,
                            registration.manifest.capability_version,
                        ),
                    )
                    if binding is not None
                ),
                execution_availability=tuple(
                    sorted(
                        (
                            availability
                            for registration in page
                            if self._executable_binding_aggregate is not None
                            for availability in (
                                self._executable_binding_aggregate.availability_for_subject(
                                    "capability-manifest."
                                    f"{registration.manifest.capability_key}."
                                    f"{registration.manifest.capability_version.replace('.', '-')}"
                                ),
                            )
                            if availability is not None
                        ),
                        key=lambda value: value.availability_id,
                    )
                ),
                returned_count=len(page),
                limit=request.limit,
                has_more=has_more,
                next_cursor=next_cursor,
            )
        except (TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="CAPABILITY_QUERY_INVALID",
                message=str(error),
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=summary,
        )

    def show_capability(
        self,
        request: CapabilityShowRequest,
    ) -> ApiResult[CapabilityShowSummary]:
        """Show one exact static capability registration."""

        operation = "capability.show"
        catalog = self._capability_discovery_catalog
        if catalog is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code=CandidateDiagnosticCode.CAPABILITY_REQUIRED.value,
                message="static capability discovery is not composed",
            )
        try:
            if not isinstance(request, CapabilityShowRequest):
                raise TypeError("request must be a CapabilityShowRequest")
            registration = catalog.registration(
                request.capability_key,
                request.capability_version,
            )
        except (TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="CAPABILITY_QUERY_INVALID",
                message=str(error),
            )
        if registration is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="CAPABILITY_NOT_FOUND",
                message=(
                    f"capability {request.capability_key}@"
                    f"{request.capability_version} is not registered"
                ),
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=CapabilityShowSummary(
                catalog_id=catalog.catalog_id,
                catalog_sha256=catalog.fingerprint(),
                registration=registration,
                bundle_binding=(
                    None
                    if self._extension_bundle_aggregate is None
                    else self._extension_bundle_aggregate.binding(
                        registration.manifest.capability_key,
                        registration.manifest.capability_version,
                    )
                ),
                execution_availability=(
                    None
                    if self._executable_binding_aggregate is None
                    else self._executable_binding_aggregate.availability_for_subject(
                        "capability-manifest."
                        f"{registration.manifest.capability_key}."
                        f"{registration.manifest.capability_version.replace('.', '-')}"
                    )
                ),
            ),
        )

    def _resolve_conditional_child(
        self,
        *,
        parent_compilation: StudyCompilationReport,
        parent_record_path: Path,
    ) -> ConditionalChildResolution:
        resolver = self._conditional_successor_resolver
        if resolver is None:
            raise ValueError("conditional follow-up resolution is not composed")
        parent_record = load_registered_authoring(
            parent_record_path,
            root_schemas=resolver.parent_record_schemas,
        )
        return resolver.resolve(
            parent_compilation=parent_compilation,
            parent_record=parent_record,
        )

    def compile_study(
        self,
        request: CompileStudyRequest,
    ) -> ApiResult[StudyCompilationSummary]:
        """Pure draft compilation with optional bounded local candidate emission."""

        operation = "campaign.candidate-compile"
        provider = self._candidate_context_provider
        if provider is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code=CandidateDiagnosticCode.CAPABILITY_REQUIRED.value,
                message="candidate compilation context is not composed",
            )
        try:
            loaded = load_registered_authoring_materialization(
                request.path,
                root_schemas={
                    StudyDefinition.SCHEMA: StudyDefinition,
                    RetrospectiveAuthoringBase.SCHEMA: RetrospectiveAuthoringBase,
                },
            )
            if not isinstance(loaded.record, StudyDefinition):
                raise AuthoringCodecError("candidate compile requires a ProgrammeAuthoringPackage")
            readiness_diagnostics: tuple[CandidateDiagnostic, ...] = ()
            resolution = provider.resolve_standard(loaded.record)
            context = resolution.context
            readiness_diagnostics = resolution.diagnostics
            report = compile_study_candidate(
                authoring_package=loaded.record,
                authoring_materialization=AuthoringMaterializationIdentity(
                    media_type=loaded.media_type,
                    byte_count=loaded.byte_count,
                    raw_materialization_sha256=loaded.raw_materialization_sha256,
                ),
                context=context,
                readiness_diagnostics=readiness_diagnostics,
            )
            conditional_resolution = (
                None
                if request.parent_record_path is None
                else self._resolve_conditional_child(
                    parent_compilation=report,
                    parent_record_path=request.parent_record_path,
                )
            )
            if conditional_resolution is not None:
                if conditional_resolution.compilation is None:
                    reasons = conditional_resolution.instantiation.reason_codes
                    return ApiResult(
                        operation=operation,
                        status=OperationStatus.BLOCKED,
                        errors=(
                            ApiError(
                                category=ErrorCategory.VALIDATION,
                                reason_code="CONDITIONAL_SUCCESSOR_NOT_ELIGIBLE",
                                message=(
                                    "frozen parent adjudication terminates the "
                                    "conditional successor"
                                ),
                            ),
                        ),
                        reason_codes=reasons or ("CONDITIONAL_SUCCESSOR_NOT_ELIGIBLE",),
                    )
                report = conditional_resolution.compilation
            relative_path: str | None = None
            candidate_sha256: str | None = None
            local_bytes_written = 0
            if request.emit_path is not None and report.candidate is not None:
                payload = report.candidate.canonical_bytes()
                relative_path = self._emit_local_candidate(
                    draft_path=request.path,
                    requested_path=request.emit_path,
                    payload=payload,
                )
                candidate_sha256 = hashlib.sha256(payload).hexdigest()
                local_bytes_written = len(payload)
            base_summary_type: type[StudyCompilationSummary] = (
                RetrospectiveBaseCompilationSummary
                if isinstance(report, RetrospectiveStandardReport)
                else StudyCompilationSummary
            )
            summary = base_summary_type(
                report=report,
                local_candidate_emitted=relative_path is not None,
                local_candidate_relative_path=relative_path,
                local_candidate_sha256=candidate_sha256,
                local_bytes_written=local_bytes_written,
            )
        except FileNotFoundError as error:
            return ApiResult(
                operation=operation,
                status=OperationStatus.NOT_FOUND,
                errors=(
                    ApiError(
                        category=ErrorCategory.NOT_FOUND,
                        reason_code="AUTHORING_DOCUMENT_NOT_FOUND",
                        message=str(error),
                    ),
                ),
                reason_codes=("AUTHORING_DOCUMENT_NOT_FOUND",),
            )
        except FileExistsError as error:
            return ApiResult(
                operation=operation,
                status=OperationStatus.CONFLICT,
                errors=(
                    ApiError(
                        category=ErrorCategory.CONFLICT,
                        reason_code="LOCAL_CANDIDATE_NO_REPLACE",
                        message=str(error),
                    ),
                ),
                reason_codes=("LOCAL_CANDIDATE_NO_REPLACE",),
            )
        except (AuthoringCodecError, ValueError) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        except OSError as error:
            return ApiResult(
                operation=operation,
                status=OperationStatus.FAILED,
                errors=(
                    ApiError(
                        category=ErrorCategory.STORAGE,
                        reason_code="LOCAL_CANDIDATE_EMISSION_FAILED",
                        message=str(error),
                    ),
                ),
                reason_codes=("LOCAL_CANDIDATE_EMISSION_FAILED",),
            )

        blocking = tuple(
            value
            for value in report.diagnostics
            if value.diagnostic_class
            in {
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticClass.UNRESOLVED_READINESS,
            }
        )
        if report.disposition is CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING:
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=summary,
                reason_codes=(CandidateDiagnosticCode.READY_TO_ISSUE.value,),
            )
        status = (
            OperationStatus.INVALID
            if report.disposition is CandidateCompilationDisposition.INVALID_DRAFT
            else OperationStatus.BLOCKED
        )
        return ApiResult(
            operation=operation,
            status=status,
            payload=summary,
            errors=tuple(self._candidate_diagnostic_error(value) for value in blocking),
            reason_codes=tuple(sorted({value.code.value for value in blocking})),
        )

    def _load_extension_materializations(
        self,
        *,
        package: ExecutableStudyDefinition,
        payload_paths: tuple[Path, ...],
        decoder_registration_paths: tuple[Path, ...],
    ) -> tuple[
        tuple[bytes, ...],
        tuple[StudyExtensionDecoderRegistration, ...],
        tuple[StudyExtensionMaterializationReceipt, ...],
        tuple[CanonicalRecord, ...],
    ]:
        registry = self._study_extension_codec_registry
        proposals = package.extension_set.extensions
        if registry is None:
            raise ValueError("extension-aware static codec registry is not composed")
        if not (len(proposals) == len(payload_paths) == len(decoder_registration_paths)):
            raise ValueError("extension payload/decoder paths differ from proposed roster")
        payloads: list[bytes] = []
        registrations: list[StudyExtensionDecoderRegistration] = []
        receipts: list[StudyExtensionMaterializationReceipt] = []
        records: list[CanonicalRecord] = []
        for proposal, payload_path, registration_path in zip(
            proposals,
            payload_paths,
            decoder_registration_paths,
            strict=True,
        ):
            registration = load_registered_authoring(
                registration_path,
                root_schemas={
                    StudyExtensionDecoderRegistration.SCHEMA: (
                        StudyExtensionDecoderRegistration
                    ),
                },
            )
            if not isinstance(registration, StudyExtensionDecoderRegistration):
                raise AuthoringCodecError(
                    "extension route requires ProgrammeExtensionDecoderRegistration"
                )
            payload = read_bounded_bytes(
                payload_path,
                maximum_bytes=registration.maximum_payload_bytes,
            )
            decoded = registry.decode(payload, expected_schema=proposal.payload.object_schema)
            payload_identity = ObjectIdentity.from_record(
                proposal.payload.object_id,
                decoded,
            )
            if (
                payload_identity != proposal.payload
                or len(payload) != proposal.payload_size_bytes
                or registration.decoder_key != proposal.decoder_key
                or registration.decoder_version != proposal.decoder_version
                or registration.payload_schema != proposal.payload.object_schema
                or registration.payload_version != proposal.payload.object_version
                or registration.config_sha256 != proposal.decoder_config_sha256
                or len(payload) > registration.maximum_payload_bytes
            ):
                raise ValueError("extension bytes/decoder differ from their exact proposal")
            physical_sha256 = hashlib.sha256(payload).hexdigest()
            receipt = StudyExtensionMaterializationReceipt(
                receipt_id=(
                    f"extension-materialization.{proposal.extension_id}.{physical_sha256[:16]}"
                ),
                extension_id=proposal.extension_id,
                proposed_extension=ObjectIdentity.from_record(
                    proposal.extension_id,
                    proposal,
                ),
                payload=payload_identity,
                byte_count=len(payload),
                physical_sha256=physical_sha256,
                decoder_registration=ObjectIdentity.from_record(
                    registration.registration_id,
                    registration,
                ),
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            )
            payloads.append(payload)
            registrations.append(registration)
            receipts.append(receipt)
            records.append(decoded)
        return tuple(payloads), tuple(registrations), tuple(receipts), tuple(records)

    def _parameterised_candidate_context(
        self,
        *,
        package: ExecutableStudyDefinition,
        context: CandidateCompilationContext | StandardCandidateCompilationContext,
        records: tuple[CanonicalRecord, ...],
    ) -> CandidateCompilationContext | StandardCandidateCompilationContext:
        return compose_parameterised_candidate_context(
            package=package,
            context=context,
            records=records,
            factories=self._executable_factory_registry,
        )

    def compile_candidate(
        self,
        request: CompileCandidateRequest,
    ) -> ApiResult[ExecutableStudyCompilationSummary]:
        """Compile the exact base study definition plus its closed, byte-verified extension roster."""

        operation = "campaign.compile-candidate"
        provider = self._candidate_context_provider
        if provider is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code=CandidateDiagnosticCode.CAPABILITY_REQUIRED.value,
                message="candidate compilation context is not composed",
            )
        try:
            loaded = load_registered_authoring_materialization(
                request.path,
                root_schemas={
                    ExecutableStudyDefinition.SCHEMA: ExecutableStudyDefinition,
                    RetrospectiveAuthoringPackage.SCHEMA: RetrospectiveAuthoringPackage,
                },
            )
            package = loaded.record
            if not isinstance(package, ExecutableStudyDefinition):
                raise AuthoringCodecError(
                    "candidate compilation requires ExecutableStudyDefinition"
                )
            _, _, materializations, extension_records = self._load_extension_materializations(
                package=package,
                payload_paths=request.extension_payload_paths,
                decoder_registration_paths=request.decoder_registration_paths,
            )
            resolution = provider.resolve_standard(package.base)
            context = self._parameterised_candidate_context(
                package=package,
                context=resolution.context,
                records=extension_records,
            )
            if not isinstance(context, StandardCandidateCompilationContext):
                raise ValueError("executable study compilation resolved a nonstandard context")
            base_bytes = package.base.canonical_bytes()
            base_report = compile_study_candidate(
                authoring_package=package.base,
                authoring_materialization=AuthoringMaterializationIdentity(
                    media_type="application/json",
                    byte_count=len(base_bytes),
                    raw_materialization_sha256=hashlib.sha256(
                        b"application/json\x00" + base_bytes
                    ).hexdigest(),
                ),
                context=context,
                readiness_diagnostics=resolution.diagnostics,
            )
            if request.parent_record_path is not None:
                conditional = self._resolve_conditional_child(
                    parent_compilation=base_report,
                    parent_record_path=request.parent_record_path,
                )
                if conditional.compilation is None:
                    return ApiResult(
                        operation=operation,
                        status=OperationStatus.BLOCKED,
                        errors=(
                            ApiError(
                                category=ErrorCategory.VALIDATION,
                                reason_code="CONDITIONAL_SUCCESSOR_NOT_ELIGIBLE",
                                message="frozen parent terminates the conditional successor",
                            ),
                        ),
                        reason_codes=(
                            conditional.instantiation.reason_codes
                            or ("CONDITIONAL_SUCCESSOR_NOT_ELIGIBLE",)
                        ),
                    )
                base_report = conditional.compilation
            report = bind_standard_candidate_compilation_report(
                authoring_package=package,
                authoring_materialization=AuthoringMaterializationIdentity(
                    media_type=loaded.media_type,
                    byte_count=loaded.byte_count,
                    raw_materialization_sha256=loaded.raw_materialization_sha256,
                ),
                extension_materializations=materializations,
                base_report=base_report,
            )
            relative_path: str | None = None
            candidate_sha256: str | None = None
            local_bytes_written = 0
            if request.emit_path is not None and report.candidate is not None:
                candidate_payload = report.candidate.canonical_bytes()
                relative_path = self._emit_local_candidate(
                    draft_path=request.path,
                    requested_path=request.emit_path,
                    payload=candidate_payload,
                )
                candidate_sha256 = hashlib.sha256(candidate_payload).hexdigest()
                local_bytes_written = len(candidate_payload)
            compilation_summary_type: type[ExecutableStudyCompilationSummary] = (
                RetrospectiveCompilationSummary
                if isinstance(report, RetrospectiveExtensionReport)
                else ExecutableStudyCompilationSummary
            )
            summary = compilation_summary_type(
                report=report,
                local_candidate_emitted=relative_path is not None,
                local_candidate_relative_path=relative_path,
                local_candidate_sha256=candidate_sha256,
                local_bytes_written=local_bytes_written,
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="EXTENSION_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except FileExistsError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.CONFLICT,
                category=ErrorCategory.CONFLICT,
                reason_code="LOCAL_CANDIDATE_NO_REPLACE",
                message=str(error),
            )
        except (AuthoringCodecError, BoundedFileIOError, KeyError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="EXTENSION_CANDIDATE_INVALID",
                message=str(error),
            )
        except OSError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.FAILED,
                category=ErrorCategory.STORAGE,
                reason_code="LOCAL_CANDIDATE_EMISSION_FAILED",
                message=str(error),
            )
        blocking = tuple(
            value
            for value in report.diagnostics
            if value.diagnostic_class
            in {
                CandidateDiagnosticClass.FATAL,
                CandidateDiagnosticClass.UNRESOLVED_READINESS,
            }
        )
        if report.disposition is CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING:
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=summary,
                reason_codes=(CandidateDiagnosticCode.READY_TO_ISSUE.value,),
            )
        return ApiResult(
            operation=operation,
            status=(
                OperationStatus.INVALID
                if report.disposition is CandidateCompilationDisposition.INVALID_DRAFT
                else OperationStatus.BLOCKED
            ),
            payload=summary,
            errors=tuple(self._candidate_diagnostic_error(value) for value in blocking),
            reason_codes=tuple(sorted({value.code.value for value in blocking})),
        )

    def preissue_readiness(
        self,
        request: CheckReadinessRequest,
    ) -> ApiResult[PreissueReadinessSummary]:
        """Prove the exact future production topology without effects or authority."""

        operation = "campaign.check-readiness"
        provider = self._candidate_context_provider
        inspector = self._study_source_closure_inspector
        service = self._execution_service
        if provider is None or inspector is None or service is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code="PREISSUE_READINESS_COMPOSITION_REQUIRED",
                message="candidate, source-closure, and runtime composition are required",
            )
        compiled = self.compile_candidate(
            CompileCandidateRequest(
                path=request.authoring_package_path,
                extension_payload_paths=request.extension_payload_paths,
                decoder_registration_paths=request.decoder_registration_paths,
            )
        )
        if not compiled.succeeded or compiled.payload is None:
            return ApiResult(
                operation=operation,
                status=compiled.status,
                errors=compiled.errors,
                reason_codes=compiled.reason_codes,
            )
        try:
            candidate = compiled.payload.report.candidate
            expected = load_standard_study_candidate(request.expected_candidate_path)
            if (
                candidate is None
                or candidate != expected
                or candidate.canonical_bytes() != expected.canonical_bytes()
            ):
                raise ValueError("candidate changed after the public compilation")
            package = load_registered_authoring(
                request.authoring_package_path,
                root_schemas={
                    ExecutableStudyDefinition.SCHEMA: ExecutableStudyDefinition,
                    RetrospectiveAuthoringPackage.SCHEMA: RetrospectiveAuthoringPackage,
                },
            )
            payloads, registrations, materializations, records = (
                self._load_extension_materializations(
                    package=package,
                    payload_paths=request.extension_payload_paths,
                    decoder_registration_paths=request.decoder_registration_paths,
                )
            )
            source_closure = load_registered_authoring(
                request.source_closure_path,
                root_schemas={ImplementationSourceClosure.SCHEMA: ImplementationSourceClosure},
            )
            if inspector.observe(source_closure) != source_closure:
                raise ValueError("implementation source closure changed")
            envelope = load_registered_authoring(
                request.execution_resource_envelope_spec_path,
                root_schemas={
                    ExecutionResourceEnvelopeSpec.SCHEMA: ExecutionResourceEnvelopeSpec
                },
            )
            elapsed_budget = (
                None
                if request.campaign_elapsed_budget_path is None
                else load_registered_authoring(
                    request.campaign_elapsed_budget_path,
                    root_schemas={CampaignElapsedBudgetSpec.SCHEMA: CampaignElapsedBudgetSpec},
                )
            )
            elapsed_reservations = (
                None
                if request.campaign_elapsed_reservation_plan_path is None
                else load_registered_authoring(
                    request.campaign_elapsed_reservation_plan_path,
                    root_schemas={
                        CampaignElapsedReservationPlan.SCHEMA: (CampaignElapsedReservationPlan)
                    },
                )
            )
            if (elapsed_budget is None) != (elapsed_reservations is None):
                raise ValueError("campaign elapsed budget and reservation plan are partial")
            if elapsed_budget is not None and elapsed_reservations is not None:
                campaign = package.base.draft.campaign
                if campaign is None:
                    raise ValueError("campaign elapsed budget lacks its authoring campaign")
                if elapsed_budget.campaign_anchor != ObjectIdentity.from_record(
                    campaign.campaign_id, campaign
                ) or elapsed_reservations.budget != ObjectIdentity.from_record(
                    elapsed_budget.budget_id, elapsed_budget
                ):
                    raise ValueError("campaign elapsed budget changes its authoring campaign")
                elapsed_reservations.validate_task_census(envelope)
            census = (
                None
                if request.predevelopment_jit_signature_census_path is None
                else load_registered_authoring(
                    request.predevelopment_jit_signature_census_path,
                    root_schemas={
                        PredevelopmentJitSignatureCensus.SCHEMA: PredevelopmentJitSignatureCensus
                    },
                )
            )
            manifest = (
                None
                if request.jit_graph_signature_manifest_path is None
                else load_registered_authoring(
                    request.jit_graph_signature_manifest_path,
                    root_schemas={JitGraphSignatureManifest.SCHEMA: JitGraphSignatureManifest},
                )
            )
            context = self._parameterised_candidate_context(
                package=package,
                context=provider.resolve_standard(package.base).context,
                records=records,
            )
            if not isinstance(context, StandardCandidateCompilationContext):
                raise ValueError("preissue readiness resolved a nonstandard context")
            issued, _, _ = project_standard_study_extensions(
                candidate=candidate,
                extension_payload_bytes=payloads,
                decoder_registrations=registrations,
                extension_materializations=materializations,
            )
            base_candidate = candidate.base_candidate.base_candidate
            if _requires_parent_record(base_candidate.scientific_graph):
                raise ValueError("preissue readiness requires resolved parent-input substitutions")
            if source_closure.implementation_sha256 != base_candidate.implementation_sha256:
                raise ValueError("source closure implementation differs from the candidate")
            model_set = (
                None
                if request.model_set_path is None
                else load_registered_authoring(
                    request.model_set_path,
                    root_schemas={
                        ViewModelSetSpec.SCHEMA: ViewModelSetSpec,
                        ModelSetSpec.SCHEMA: ModelSetSpec,
                    },
                )
            )
            if model_set is not None:
                identity = ObjectIdentity.from_record(model_set.model_set_id, model_set)
                if (
                    len(
                        tuple(
                            value
                            for value in package.base.draft.design_inputs
                            if value.object_identity == identity
                            and value.materialization_sha256 == model_set.fingerprint()
                        )
                    )
                    != 1
                ):
                    raise ValueError(
                        "preissue model set was not frozen as an exact candidate design input"
                    )
            run_plan = compile_preissue_run_plan(
                run_plan_id=request.run_plan_id,
                candidate_record=base_candidate,
                candidate=ObjectIdentity.from_record(candidate.candidate_id, candidate),
                registry=context.base.registry,
                implementation_commit=source_closure.implementation_commit,
                issued_extension_set=issued,
                resource_envelope=envelope,
                jit_census=census,
                jit_manifest=manifest,
                model_set=model_set,
            )
            execution_plan = lower_run_plan(run_plan, context.base.registry)
            closure, runner_ids, input_ids, adjudication = service.preissue_readiness(
                registry=context.base.registry,
                run_plan=run_plan,
                execution_plan=execution_plan,
                decoded_records=tuple(sorted(records, key=lambda value: value.SCHEMA)),
                resource_envelope=envelope,
                historical_experiment=(
                    package.base.draft.experiment
                    if isinstance(package.base.draft.experiment, RetrospectiveExperimentSpec)
                    else None
                ),
            )
            if not closure.passed:
                raise ValueError("preissue closure failed: " + ",".join(closure.reason_codes))
            adjudication_locator = next(
                output.relative_path
                for task in execution_plan.tasks
                for output in task.outputs
                if task.capability.capability_key == adjudication.capability_key
                and output.output_id == adjudication.output_id
            )
            stages = tuple(sorted({task.stage.value for task in execution_plan.tasks}))
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=PreissueReadinessSummary(
                    candidate.candidate_id,
                    candidate.fingerprint(),
                    issued.issued_extension_set_id,
                    issued.fingerprint(),
                    run_plan.run_plan_id,
                    context.base.registry.fingerprint(),
                    execution_topology_sha256(execution_plan, envelope),
                    len(execution_plan.tasks),
                    tuple(
                        (stage, sum(task.stage.value == stage for task in execution_plan.tasks))
                        for stage in stages
                    ),
                    runner_ids,
                    input_ids,
                    tuple(
                        sorted(
                            output.relative_path
                            for task in execution_plan.tasks
                            for output in task.outputs
                        )
                    ),
                    adjudication_locator,
                    execution_plan.minimum_free_bytes,
                    sum(cell.resource_budget.source_byte_limit for cell in envelope.task_cells),
                    envelope.maximum_parallel_tasks,
                    envelope.aggregate_memory_ceiling_bytes,
                    base_candidate.expected_authority_gates,
                    tuple(
                        (check.section.value, check.passed, check.reason_codes)
                        for check in closure.checks
                    ),
                    campaign_elapsed_budget_verified=elapsed_budget is not None,
                    campaign_elapsed_reservation_plan_fingerprint=(
                        None if elapsed_reservations is None else elapsed_reservations.fingerprint()
                    ),
                ),
                reason_codes=("PREISSUE_READINESS_ACCEPTED",),
            )
        except CampaignExecutionError as error:
            failed = self._execution_error(operation, error)
            return ApiResult(
                operation=failed.operation,
                status=failed.status,
                errors=failed.errors,
                reason_codes=failed.reason_codes,
            )
        except (
            AuthoringCodecError,
            BoundedFileIOError,
            KeyError,
            OSError,
            PermissionError,
            TypeError,
            ValueError,
        ) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.VALIDATION,
                reason_code="PREISSUE_READINESS_REFUSED",
                message=str(error),
            )

    def compile_study_bundle(
        self,
        request: CompileStudyBundleRequest,
    ) -> ApiResult[StudyBundleCompilationSummary]:
        """Pure exact two-or-more-child bundle compilation through the public facade."""

        operation = "campaign.bundle-candidate-compile"
        registry = self._study_bundle_registry
        if registry is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code="PROGRAMME_BUNDLE_REGISTRY_REQUIRED",
                message="programme bundle compilation registry is not composed",
            )
        try:
            loaded = load_registered_authoring_materialization(
                request.path,
                root_schemas={
                    StudyBundleCompilationInput.SCHEMA: StudyBundleCompilationInput,
                },
            )
            package = loaded.record
            if not isinstance(package, StudyBundleCompilationInput):
                raise AuthoringCodecError(
                    "bundle compile requires a ProgrammeBundleCompilationInput"
                )
            if package.joint_registry != registry:
                raise ValueError("bundle document changed the statically composed registry")
            report = compile_study_bundle(package)
            relative_path: str | None = None
            candidate_sha256: str | None = None
            local_bytes_written = 0
            if request.emit_path is not None and report.candidate is not None:
                payload = report.candidate.canonical_bytes()
                relative_path = self._emit_local_candidate(
                    draft_path=request.path,
                    requested_path=request.emit_path,
                    payload=payload,
                )
                candidate_sha256 = hashlib.sha256(payload).hexdigest()
                local_bytes_written = len(payload)
            summary = StudyBundleCompilationSummary(
                report=report,
                local_candidate_emitted=relative_path is not None,
                local_candidate_relative_path=relative_path,
                local_candidate_sha256=candidate_sha256,
                local_bytes_written=local_bytes_written,
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="PROGRAMME_BUNDLE_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except FileExistsError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.CONFLICT,
                category=ErrorCategory.CONFLICT,
                reason_code="LOCAL_BUNDLE_CANDIDATE_NO_REPLACE",
                message=str(error),
            )
        except (AuthoringCodecError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="PROGRAMME_BUNDLE_INVALID",
                message=str(error),
            )
        except OSError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.FAILED,
                category=ErrorCategory.STORAGE,
                reason_code="LOCAL_BUNDLE_CANDIDATE_EMISSION_FAILED",
                message=str(error),
            )
        if report.disposition is StudyBundleCompilationDisposition.INVALID:
            return ApiResult(
                operation=operation,
                status=OperationStatus.INVALID,
                payload=summary,
                errors=tuple(
                    ApiError(
                        category=ErrorCategory.VALIDATION,
                        reason_code=reason,
                        message="programme bundle contract did not validate",
                    )
                    for reason in report.reason_codes
                ),
                reason_codes=report.reason_codes,
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=summary,
            reason_codes=("READY_FOR_SEPARATE_ISSUE_AUTHORITY",),
        )

    def issue_study(
        self,
        request: IssueStudyRequest,
    ) -> ApiResult[StudyIssueSummary]:
        """Preview or publish one exact outcome-blind issued-programme bundle."""

        operation = "campaign.issue"
        provider = self._candidate_context_provider
        publisher = self._study_issue_publisher
        source_inspector = self._study_source_closure_inspector
        if provider is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code="PROGRAMME_ISSUE_CANDIDATE_CONTEXT_REQUIRED",
                message="programme issue candidate context is not composed",
            )
        if publisher is None or source_inspector is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="PROGRAMME_ISSUE_TRUST_CONFIGURATION_REQUIRED",
                message=(
                    "programme issue requires a guarded publisher and independent "
                    "source-closure inspector"
                ),
            )
        try:
            loaded = load_registered_authoring_materialization(
                request.authoring_package_path,
                root_schemas={
                    StudyDefinition.SCHEMA: StudyDefinition,
                    RetrospectiveAuthoringBase.SCHEMA: RetrospectiveAuthoringBase,
                },
            )
            if not isinstance(loaded.record, StudyDefinition):
                raise AuthoringCodecError("programme issue requires a ProgrammeAuthoringPackage")
            expected_candidate = load_registered_authoring(
                request.expected_candidate_path,
                root_schemas={
                    StudyCandidate.SCHEMA: StudyCandidate,
                    RetrospectiveStandardCandidate.SCHEMA: RetrospectiveStandardCandidate,
                },
            )
            attestation = load_registered_authoring(
                request.proposer_attestation_path,
                root_schemas={
                    HumanProposerAttestation.SCHEMA: HumanProposerAttestation,
                    RetrospectiveProposerAttestation.SCHEMA: RetrospectiveProposerAttestation,
                },
            )
            source_closure = load_registered_authoring(
                request.source_closure_path,
                root_schemas={
                    ImplementationSourceClosure.SCHEMA: ImplementationSourceClosure,
                },
            )
            custody_authority = load_registered_authoring(
                request.custody_authority_path,
                root_schemas={
                    StudyOperationAuthority.SCHEMA: StudyOperationAuthority,
                },
            )
            if not isinstance(attestation, HumanProposerAttestation):
                raise AuthoringCodecError("programme issue requires a HumanProposerAttestation")
            if not isinstance(expected_candidate, StudyCandidate):
                raise AuthoringCodecError(
                    "programme issue requires an expected StandardProgrammeCandidate"
                )
            if not isinstance(source_closure, ImplementationSourceClosure):
                raise AuthoringCodecError("programme issue requires an ImplementationSourceClosure")
            if not isinstance(custody_authority, StudyOperationAuthority):
                raise AuthoringCodecError("programme issue requires a ProgrammeOperationAuthority")
            readiness_diagnostics: tuple[CandidateDiagnostic, ...] = ()
            resolution = provider.resolve_standard(loaded.record)
            context = resolution.context
            readiness_diagnostics = (*resolution.diagnostics, *resolution.issue_diagnostics)
            report = compile_study_candidate(
                authoring_package=loaded.record,
                authoring_materialization=AuthoringMaterializationIdentity(
                    media_type=loaded.media_type,
                    byte_count=loaded.byte_count,
                    raw_materialization_sha256=loaded.raw_materialization_sha256,
                ),
                context=context,
                readiness_diagnostics=readiness_diagnostics,
            )
            if request.parent_record_path is not None:
                conditional_resolution = self._resolve_conditional_child(
                    parent_compilation=report,
                    parent_record_path=request.parent_record_path,
                )
                if conditional_resolution.compilation is None:
                    reasons = conditional_resolution.instantiation.reason_codes
                    return ApiResult(
                        operation=operation,
                        status=OperationStatus.BLOCKED,
                        errors=(
                            ApiError(
                                category=ErrorCategory.VALIDATION,
                                reason_code="CONDITIONAL_SUCCESSOR_NOT_ELIGIBLE",
                                message=(
                                    "frozen parent adjudication terminates the "
                                    "conditional successor"
                                ),
                            ),
                        ),
                        reason_codes=reasons or ("CONDITIONAL_SUCCESSOR_NOT_ELIGIBLE",),
                    )
                report = conditional_resolution.compilation
            candidate = report.candidate
            if (
                report.disposition is not CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING
                or candidate is None
            ):
                blocking = tuple(
                    value
                    for value in report.diagnostics
                    if value.diagnostic_class
                    in {
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    }
                )
                return ApiResult(
                    operation=operation,
                    status=(
                        OperationStatus.INVALID
                        if report.disposition is CandidateCompilationDisposition.INVALID_DRAFT
                        else OperationStatus.BLOCKED
                    ),
                    errors=tuple(self._candidate_diagnostic_error(value) for value in blocking),
                    reason_codes=tuple(sorted({value.code.value for value in blocking})),
                )
            if candidate != expected_candidate:
                raise ValueError("standard candidate changed between preview and issue")
            observed_source_closure = source_inspector.observe(source_closure)
            at_utc = self._study_issue_clock.now_utc()
            preparation = prepare_study_issue(
                authoring_package=loaded.record,
                raw_authoring_package_bytes=loaded.raw_bytes,
                current_compilation=report,
                candidate=candidate,
                registry=context.base.registry,
                materialization_qualifications=context.base.qualifications,
                source_closure=source_closure,
                observed_source_closure=observed_source_closure,
                proposer_attestation=attestation,
                custody_authority=custody_authority,
                storage_root_id=publisher.storage_root_id,
                grantee_id=publisher.grantee_id,
                at_utc=at_utc,
            )
            planned_payload_bytes = publisher.validate(
                preparation,
                at_utc=at_utc,
            )
            issue_summary_type: type[StudyIssueSummary] = (
                RetrospectiveBaseIssueSummary
                if isinstance(preparation.manifest, RetrospectiveIssuedBase)
                else StudyIssueSummary
            )
            if not request.confirmed:
                return ApiResult(
                    operation=operation,
                    status=OperationStatus.BLOCKED,
                    payload=issue_summary_type(
                        manifest=preparation.manifest,
                        confirmed=False,
                        publication_receipt=None,
                        planned_member_count=len(preparation.manifest.members),
                        planned_payload_bytes=planned_payload_bytes,
                        external_bytes_written=0,
                        publication_receipt_bytes=0,
                    ),
                    reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
                )
            publication = publisher.publish(preparation, at_utc=at_utc)
            published = publisher.load_study(preparation.manifest.issue_id)
            if (
                published.manifest != preparation.manifest
                or published.publication_receipt != publication
            ):
                raise RuntimeError("published programme failed exact external replay")
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=issue_summary_type(
                    manifest=published.manifest,
                    confirmed=True,
                    publication_receipt=published.publication_receipt,
                    planned_member_count=len(published.manifest.members),
                    planned_payload_bytes=planned_payload_bytes,
                    external_bytes_written=(
                        published.publication_receipt.total_payload_bytes
                        + len(published.publication_receipt.canonical_bytes())
                    ),
                    publication_receipt_bytes=len(published.publication_receipt.canonical_bytes()),
                ),
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="PROGRAMME_ISSUE_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except (KeyError, PermissionError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="PROGRAMME_ISSUE_AUTHORITY_OR_SOURCE_REPLAY_FAILED",
                message=str(error),
            )
        except BoundedProcessError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.IDENTITY,
                reason_code="PROGRAMME_SOURCE_CLOSURE_INSPECTION_FAILED",
                message=str(error),
            )
        except (AuthoringCodecError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="PROGRAMME_ISSUE_INVALID",
                message=str(error),
            )
        except (ArtifactPlaneError, BoundedFileIOError, OSError, RuntimeError) as error:
            return self._storage_error(
                operation,
                "PROGRAMME_ISSUE_PUBLICATION_FAILED",
                error,
            )

    def issue_extensions(
        self,
        request: IssueExtensionsRequest,
    ) -> ApiResult[ExecutableStudyIssueSummary]:
        """Preview or publish an extension wrapper over one immutable base study issue."""

        operation = "campaign.issue-extensions"
        provider = self._candidate_context_provider
        publisher = self._study_issue_publisher
        if provider is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code="PROGRAMME_ISSUE_CANDIDATE_CONTEXT_REQUIRED",
                message="programme issue candidate context is not composed",
            )
        if publisher is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="PROGRAMME_ISSUE_TRUST_CONFIGURATION_REQUIRED",
                message="extension issuance requires the guarded external publisher",
            )
        try:
            loaded = load_registered_authoring_materialization(
                request.authoring_package_path,
                root_schemas={
                    ExecutableStudyDefinition.SCHEMA: ExecutableStudyDefinition,
                    RetrospectiveAuthoringPackage.SCHEMA: RetrospectiveAuthoringPackage,
                },
            )
            package = loaded.record
            if not isinstance(package, ExecutableStudyDefinition):
                raise AuthoringCodecError("extension issuance requires ExecutableStudyDefinition")
            expected_candidate = load_standard_study_candidate(
                request.expected_candidate_path,
            )
            attestation = load_registered_authoring(
                request.extension_proposer_attestation_path,
                root_schemas={
                    HumanProposerAttestation.SCHEMA: HumanProposerAttestation,
                    RetrospectiveProposerAttestation.SCHEMA: RetrospectiveProposerAttestation,
                },
            )
            custody_authority = load_registered_authoring(
                request.extension_custody_authority_path,
                root_schemas={
                    StudyOperationAuthority.SCHEMA: StudyOperationAuthority,
                },
            )
            if not isinstance(expected_candidate, ExecutableStudyCandidate):
                raise AuthoringCodecError("extension issuance requires ExecutableStudyCandidate")
            if not isinstance(attestation, HumanProposerAttestation):
                raise AuthoringCodecError("extension issuance requires HumanProposerAttestation")
            if not isinstance(custody_authority, StudyOperationAuthority):
                raise AuthoringCodecError("extension issuance requires StudyOperationAuthority")
            payloads, registrations, materializations, extension_records = (
                self._load_extension_materializations(
                    package=package,
                    payload_paths=request.extension_payload_paths,
                    decoder_registration_paths=request.decoder_registration_paths,
                )
            )
            resolution = provider.resolve_standard(package.base)
            context = self._parameterised_candidate_context(
                package=package,
                context=resolution.context,
                records=extension_records,
            )
            if not isinstance(context, StandardCandidateCompilationContext):
                raise ValueError("extension issuance resolved a nonstandard context")
            base_bytes = package.base.canonical_bytes()
            base_report = compile_study_candidate(
                authoring_package=package.base,
                authoring_materialization=AuthoringMaterializationIdentity(
                    media_type="application/json",
                    byte_count=len(base_bytes),
                    raw_materialization_sha256=hashlib.sha256(
                        b"application/json\x00" + base_bytes
                    ).hexdigest(),
                ),
                context=context,
                readiness_diagnostics=(*resolution.diagnostics, *resolution.issue_diagnostics),
            )
            if request.parent_record_path is not None:
                conditional = self._resolve_conditional_child(
                    parent_compilation=base_report,
                    parent_record_path=request.parent_record_path,
                )
                if conditional.compilation is None:
                    return ApiResult(
                        operation=operation,
                        status=OperationStatus.BLOCKED,
                        errors=(
                            ApiError(
                                category=ErrorCategory.VALIDATION,
                                reason_code="CONDITIONAL_SUCCESSOR_NOT_ELIGIBLE",
                                message="frozen parent terminates the conditional follow-up",
                            ),
                        ),
                        reason_codes=(
                            conditional.instantiation.reason_codes
                            or ("CONDITIONAL_SUCCESSOR_NOT_ELIGIBLE",)
                        ),
                    )
                base_report = conditional.compilation
            report = bind_standard_candidate_compilation_report(
                authoring_package=package,
                authoring_materialization=AuthoringMaterializationIdentity(
                    media_type=loaded.media_type,
                    byte_count=loaded.byte_count,
                    raw_materialization_sha256=loaded.raw_materialization_sha256,
                ),
                extension_materializations=materializations,
                base_report=base_report,
            )
            candidate = report.candidate
            if (
                report.disposition is not CandidateCompilationDisposition.COMPILED_AUTHORITY_PENDING
                or candidate is None
            ):
                blocking = tuple(
                    value
                    for value in report.diagnostics
                    if value.diagnostic_class
                    in {
                        CandidateDiagnosticClass.FATAL,
                        CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    }
                )
                return ApiResult(
                    operation=operation,
                    status=(
                        OperationStatus.INVALID
                        if report.disposition is CandidateCompilationDisposition.INVALID_DRAFT
                        else OperationStatus.BLOCKED
                    ),
                    errors=tuple(self._candidate_diagnostic_error(value) for value in blocking),
                    reason_codes=tuple(sorted({value.code.value for value in blocking})),
                )
            if candidate != expected_candidate:
                raise ValueError("executable study candidate changed between preview and issue")
            published_base = publisher.load_study(request.base_issue_id)
            if published_base.manifest.candidate != candidate.base_candidate:
                raise ValueError("extension issue request names another immutable base issue")
            at_utc = self._study_issue_clock.now_utc()
            preparation = prepare_executable_study_issue(
                base_manifest=published_base.manifest,
                base_publication_receipt=published_base.publication_receipt,
                current_compilation=report,
                candidate=candidate,
                extension_payload_bytes=payloads,
                decoder_registrations=registrations,
                extension_materializations=materializations,
                extension_proposer_attestation=attestation,
                extension_custody_authority=custody_authority,
                storage_root_id=publisher.storage_root_id,
                grantee_id=publisher.grantee_id,
                at_utc=at_utc,
            )
            planned_bytes = publisher.validate_standard(preparation, at_utc=at_utc)
            issue_summary_type: type[ExecutableStudyIssueSummary] = (
                RetrospectiveIssueSummary
                if isinstance(preparation.manifest, RetrospectiveIssuedStudy)
                else ExecutableStudyIssueSummary
            )
            if not request.confirmed:
                return ApiResult(
                    operation=operation,
                    status=OperationStatus.BLOCKED,
                    payload=issue_summary_type(
                        manifest=preparation.manifest,
                        confirmed=False,
                        publication_receipt=None,
                        base_member_count=len(preparation.manifest.base.members),
                        extension_member_count=len(preparation.manifest.issued_extensions.members),
                        planned_extension_payload_bytes=planned_bytes,
                        external_bytes_written=0,
                        publication_receipt_bytes=0,
                    ),
                    reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
                )
            publication = publisher.publish_standard(preparation, at_utc=at_utc)
            published = publisher.load_executable_study(preparation.manifest.issue_id)
            if (
                published.manifest != preparation.manifest
                or published.publication_receipt != publication
            ):
                raise RuntimeError("published executable study failed exact external replay")
            receipt_bytes = len(publication.canonical_bytes())
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=issue_summary_type(
                    manifest=published.manifest,
                    confirmed=True,
                    publication_receipt=publication,
                    base_member_count=len(published.manifest.base.members),
                    extension_member_count=len(published.manifest.issued_extensions.members),
                    planned_extension_payload_bytes=planned_bytes,
                    external_bytes_written=planned_bytes + receipt_bytes,
                    publication_receipt_bytes=receipt_bytes,
                ),
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="PROGRAMME_ISSUE_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except (KeyError, PermissionError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="PROGRAMME_ISSUE_AUTHORITY_OR_BASE_REPLAY_FAILED",
                message=str(error),
            )
        except (AuthoringCodecError, BoundedFileIOError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code='STUDY_ISSUE_INVALID',
                message=str(error),
            )
        except (ArtifactPlaneError, OSError, RuntimeError) as error:
            return self._storage_error(
                operation,
                'STUDY_ISSUE_PUBLICATION_FAILED',
                error,
            )

    def issue_study_bundle(
        self,
        request: IssueStudyBundleRequest,
    ) -> ApiResult[MultiWorldStudyIssueSummary]:
        """Preview or publish one parent over three immutable executable child study issues."""

        operation = "campaign.issue-bundle"
        publisher = self._study_issue_publisher
        if publisher is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="PROGRAMME_BUNDLE_ISSUE_TRUST_CONFIGURATION_REQUIRED",
                message="bundle issue requires the guarded external publisher",
            )
        try:
            candidate = load_registered_authoring(
                request.bundle_candidate_path,
                root_schemas={StudyBundleCandidate.SCHEMA: StudyBundleCandidate},
            )
            if not isinstance(candidate, StudyBundleCandidate):
                raise AuthoringCodecError("bundle issue requires ProgrammeBundleCandidate")
            if tuple(sorted(set(request.child_issue_ids))) != request.child_issue_ids:
                raise ValueError("bundle child issue IDs must be sorted and unique")
            if len(request.child_issue_ids) != 3 or len(request.child_science_paths) != 3:
                raise ValueError("bundle issue requires exactly three child issues/science roots")
            science = tuple(
                load_registered_authoring(
                    path,
                    root_schemas={
                        MultiWorldStudyChildScientificBinding.SCHEMA: (
                            MultiWorldStudyChildScientificBinding
                        )
                    },
                )
                for path in request.child_science_paths
            )
            if not all(
                isinstance(value, MultiWorldStudyChildScientificBinding) for value in science
            ):
                raise AuthoringCodecError("bundle child science root has another schema")
            science_by_base = {value.base_candidate: value for value in science}
            if len(science_by_base) != 3:
                raise ValueError("bundle child science base candidates collapsed")
            children: list[IssuedMultiWorldStudyChild] = []
            for issue_id in request.child_issue_ids:
                published_child = publisher.load_executable_study(issue_id)
                base_identity = ObjectIdentity.from_record(
                    published_child.manifest.base.candidate.candidate_id,
                    published_child.manifest.base.candidate,
                )
                binding = science_by_base.get(base_identity)
                if binding is None:
                    raise ValueError("published child issue differs from child science roster")
                children.append(
                    IssuedMultiWorldStudyChild(
                        child_id=binding.child_id,
                        science=binding,
                        manifest=published_child.manifest,
                        publication_receipt=published_child.publication_receipt,
                    )
                )

            def load_exact(path: Path, record_type: type[CanonicalRecord]) -> CanonicalRecord:
                return load_registered_authoring(
                    path,
                    root_schemas={record_type.SCHEMA: record_type},
                )

            archive_protection = load_exact(
                request.archive_outcome_protection_path,
                ArchiveOutcomeProtectionPlan,
            )
            barriers = load_exact(
                request.outcome_barrier_plan_path,
                MultiWorldOutcomeBarrierPlan,
            )
            morphism = load_exact(
                request.partial_morphism_path,
                ArchiveToSimulatorPartialMorphismSpec,
            )
            adjudication = load_exact(
                request.joint_adjudication_path,
                MultiWorldJointAdjudicationPlan,
            )
            authority = load_exact(
                request.custody_authority_path,
                StudyOperationAuthority,
            )
            if not isinstance(archive_protection, ArchiveOutcomeProtectionPlan):
                raise AuthoringCodecError("bundle issue archive protection has another schema")
            if not isinstance(barriers, MultiWorldOutcomeBarrierPlan):
                raise AuthoringCodecError("bundle issue barrier plan has another schema")
            if not isinstance(morphism, ArchiveToSimulatorPartialMorphismSpec):
                raise AuthoringCodecError("bundle issue morphism has another schema")
            if not isinstance(adjudication, MultiWorldJointAdjudicationPlan):
                raise AuthoringCodecError("bundle issue adjudication has another schema")
            if not isinstance(authority, StudyOperationAuthority):
                raise AuthoringCodecError("bundle issue authority has another schema")
            at_utc = self._study_issue_clock.now_utc()
            preparation = prepare_study_bundle_issue(
                candidate=candidate,
                children=tuple(sorted(children, key=lambda value: value.child_id)),
                archive_outcome_protection=archive_protection,
                outcome_barriers=barriers,
                partial_morphism=morphism,
                joint_adjudication=adjudication,
                custody_authority=authority,
                storage_root_id=publisher.storage_root_id,
                grantee_id=publisher.grantee_id,
                at_utc=at_utc,
            )
            planned_bytes = publisher.validate_bundle(preparation, at_utc=at_utc)
            if not request.confirmed:
                return ApiResult(
                    operation=operation,
                    status=OperationStatus.BLOCKED,
                    payload=MultiWorldStudyIssueSummary(
                        manifest=preparation.manifest,
                        confirmed=False,
                        publication_receipt=None,
                        child_issue_count=3,
                        planned_manifest_payload_bytes=planned_bytes,
                        external_bytes_written=0,
                        publication_receipt_bytes=0,
                    ),
                    reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
                )
            publication = publisher.publish_bundle(preparation, at_utc=at_utc)
            published_bundle = publisher.load_bundle(preparation.manifest.issue_id)
            if (
                published_bundle.manifest != preparation.manifest
                or published_bundle.publication_receipt != publication
            ):
                raise RuntimeError("published bundle failed exact external replay")
            receipt_bytes = len(publication.canonical_bytes())
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=MultiWorldStudyIssueSummary(
                    manifest=published_bundle.manifest,
                    confirmed=True,
                    publication_receipt=publication,
                    child_issue_count=3,
                    planned_manifest_payload_bytes=planned_bytes,
                    external_bytes_written=planned_bytes + receipt_bytes,
                    publication_receipt_bytes=receipt_bytes,
                ),
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="PROGRAMME_BUNDLE_ISSUE_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except (KeyError, PermissionError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="PROGRAMME_BUNDLE_ISSUE_AUTHORITY_OR_CHILD_REPLAY_FAILED",
                message=str(error),
            )
        except (AuthoringCodecError, BoundedFileIOError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code='STUDY_BUNDLE_ISSUE_INVALID',
                message=str(error),
            )
        except (ArtifactPlaneError, OSError, RuntimeError) as error:
            return self._storage_error(
                operation,
                'STUDY_BUNDLE_ISSUE_PUBLICATION_FAILED',
                error,
            )

    def advance_study_bundle(
        self,
        request: AdvanceStudyBundleRequest,
    ) -> ApiResult[MultiWorldOutcomeBarrierPrefix | MultiWorldAuthorityRequired]:
        """Apply one exact reveal or child-result transition to an immutable prefix."""

        operation = "campaign.advance-bundle"
        try:
            plan = load_registered_authoring(
                request.outcome_barrier_plan_path,
                root_schemas={
                    MultiWorldOutcomeBarrierPlan.SCHEMA: MultiWorldOutcomeBarrierPlan
                },
            )
            prefix = load_registered_authoring(
                request.barrier_prefix_path,
                root_schemas={
                    MultiWorldOutcomeBarrierPrefix.SCHEMA: MultiWorldOutcomeBarrierPrefix
                },
            )
            if not isinstance(plan, MultiWorldOutcomeBarrierPlan) or not isinstance(
                prefix,
                MultiWorldOutcomeBarrierPrefix,
            ):
                raise AuthoringCodecError("bundle transition requires its exact plan and prefix")
            barrier = next(
                (value for value in plan.barriers if value.barrier_id == request.barrier_id),
                None,
            )
            if barrier is None:
                raise ValueError("bundle transition names an unknown barrier")
            if request.child_result_path is not None:
                if request.reveal_authority_id is not None:
                    raise ValueError("one transition cannot reveal and bind a result together")
                child_result = load_registered_authoring(
                    request.child_result_path,
                    root_schemas={WorldLocalStudyResult.SCHEMA: WorldLocalStudyResult},
                )
                if not isinstance(child_result, WorldLocalStudyResult):
                    raise AuthoringCodecError("bundle transition child result has another schema")
                if child_result.child_id != barrier.child_id:
                    raise ValueError("bundle transition child result belongs to another barrier")
                updated: MultiWorldOutcomeBarrierPrefix | MultiWorldAuthorityRequired = (
                    bind_multi_world_child_result(
                        plan=plan,
                        prefix=prefix,
                        barrier_id=barrier.barrier_id,
                        child_result=ObjectIdentity.from_record(
                            child_result.result_id,
                            child_result,
                        ),
                    )
                )
            else:
                authority: StudyOperationAuthority | None = None
                if request.reveal_authority_id is not None:
                    store = self._study_authority_store
                    if store is None:
                        raise PermissionError("bundle reveal requires the trusted authority store")
                    authority = store.load(request.reveal_authority_id)
                updated = request_multi_world_reveal(
                    plan=plan,
                    prefix=prefix,
                    barrier_id=barrier.barrier_id,
                    authority=authority,
                    grantee_id=barrier.reveal_grantee_id,
                    at_utc=self._study_issue_clock.now_utc(),
                )
            return ApiResult(
                operation=operation,
                status=(
                    OperationStatus.BLOCKED
                    if isinstance(updated, MultiWorldAuthorityRequired)
                    else OperationStatus.SUCCEEDED
                ),
                payload=updated,
                reason_codes=(
                    ("AUTHORITY_REQUIRED",)
                    if isinstance(updated, MultiWorldAuthorityRequired)
                    else ()
                ),
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="PROGRAMME_BUNDLE_TRANSITION_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except (KeyError, PermissionError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="PROGRAMME_BUNDLE_TRANSITION_AUTHORITY_REQUIRED",
                message=str(error),
            )
        except (AuthoringCodecError, BoundedFileIOError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="PROGRAMME_BUNDLE_TRANSITION_INVALID",
                message=str(error),
            )

    def study_bundle_status(
        self,
        request: StudyBundleStatusRequest,
    ) -> ApiResult[MultiWorldStudyBarrierStatus]:
        """Inspect one immutable parent prefix without reading outcome payloads."""

        operation = "campaign.bundle-status"
        try:
            plan = load_registered_authoring(
                request.outcome_barrier_plan_path,
                root_schemas={
                    MultiWorldOutcomeBarrierPlan.SCHEMA: MultiWorldOutcomeBarrierPlan
                },
            )
            prefix = load_registered_authoring(
                request.barrier_prefix_path,
                root_schemas={
                    MultiWorldOutcomeBarrierPrefix.SCHEMA: MultiWorldOutcomeBarrierPrefix
                },
            )
            if not isinstance(plan, MultiWorldOutcomeBarrierPlan) or not isinstance(
                prefix,
                MultiWorldOutcomeBarrierPrefix,
            ):
                raise AuthoringCodecError("bundle status requires its exact plan and prefix")
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=inspect_study_bundle_barrier_status(plan=plan, prefix=prefix),
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="PROGRAMME_BUNDLE_STATUS_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except (AuthoringCodecError, BoundedFileIOError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="PROGRAMME_BUNDLE_STATUS_INVALID",
                message=str(error),
            )

    def close_study_bundle(
        self,
        request: CloseStudyBundleRequest,
    ) -> ApiResult[MultiWorldJointAdjudicationResult]:
        """Adjudicate a terminal three-child parent without pooling world-local evidence."""

        operation = "campaign.close-bundle"

        def load_exact(path: Path, record_type: type[CanonicalRecord]) -> CanonicalRecord:
            return load_registered_authoring(
                path,
                root_schemas={record_type.SCHEMA: record_type},
            )

        try:
            plan = load_exact(request.joint_adjudication_plan_path, MultiWorldJointAdjudicationPlan)
            barriers = load_exact(
                request.outcome_barrier_plan_path,
                MultiWorldOutcomeBarrierPlan,
            )
            prefix = load_exact(
                request.terminal_barrier_prefix_path,
                MultiWorldOutcomeBarrierPrefix,
            )
            overlap = load_exact(
                request.archive_overlap_result_path,
                ArchiveOverlapQualificationResult,
            )
            controls = load_exact(
                request.morphism_control_contrast_path,
                MorphismControlContrastReceipt,
            )
            child_results = tuple(
                load_exact(path, WorldLocalStudyResult) for path in request.child_result_paths
            )
            verdicts = tuple(
                load_exact(path, PropertyMorphismVerdict)
                for path in request.morphism_verdict_paths
            )
            if not isinstance(plan, MultiWorldJointAdjudicationPlan):
                raise AuthoringCodecError("bundle closeout plan has another schema")
            if not isinstance(barriers, MultiWorldOutcomeBarrierPlan) or not isinstance(
                prefix,
                MultiWorldOutcomeBarrierPrefix,
            ):
                raise AuthoringCodecError("bundle closeout barrier input has another schema")
            status = inspect_study_bundle_barrier_status(plan=barriers, prefix=prefix)
            if not status.terminal:
                raise ValueError("bundle closeout requires three terminal child results")
            if not isinstance(overlap, ArchiveOverlapQualificationResult) or not isinstance(
                controls,
                MorphismControlContrastReceipt,
            ):
                raise AuthoringCodecError("bundle closeout morphism input has another schema")
            if len(child_results) != 3 or not all(
                isinstance(value, WorldLocalStudyResult) for value in child_results
            ):
                raise ValueError("bundle closeout requires exactly three child results")
            if not all(isinstance(value, PropertyMorphismVerdict) for value in verdicts):
                raise AuthoringCodecError("bundle closeout verdict input has another schema")
            typed_child_results = tuple(
                cast(WorldLocalStudyResult, value) for value in child_results
            )
            typed_verdicts = tuple(cast(PropertyMorphismVerdict, value) for value in verdicts)
            bound_results = {
                value.child_result for value in prefix.events if value.child_result is not None
            }
            if {
                ObjectIdentity.from_record(value.result_id, value) for value in typed_child_results
            } != bound_results:
                raise ValueError("bundle closeout child results differ from the terminal prefix")
            result = adjudicate_joint_multi_world_study(
                result_id=f"joint-result.{prefix.prefix_id}",
                plan=plan,
                child_results=tuple(sorted(typed_child_results, key=lambda value: value.child_id)),
                overlap_result=overlap,
                morphism_verdicts=tuple(
                    sorted(typed_verdicts, key=lambda value: value.state_input.object_id)
                ),
                control_contrast=controls,
                mapped_prospective_accepted_count=request.mapped_prospective_accepted_count,
                mapped_prospective_map_unevaluable_count=request.mapped_prospective_map_unevaluable_count,
            )
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=result,
            )
        except FileNotFoundError as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="PROGRAMME_BUNDLE_CLOSEOUT_DOCUMENT_NOT_FOUND",
                message=str(error),
            )
        except (AuthoringCodecError, BoundedFileIOError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="PROGRAMME_BUNDLE_CLOSEOUT_INVALID",
                message=str(error),
            )

    @staticmethod
    def _candidate_diagnostic_error(value: CandidateDiagnostic) -> ApiError:
        category = ErrorCategory.VALIDATION
        if value.code in {
            CandidateDiagnosticCode.CAPABILITY_CONTRACT_MISMATCH,
            CandidateDiagnosticCode.CAPABILITY_REQUIRED,
        }:
            category = ErrorCategory.CAPABILITY
        elif value.code is CandidateDiagnosticCode.SOURCE_ACQUISITION_AUTHORITY_REQUIRED:
            category = ErrorCategory.AUTHORITY
        return ApiError(
            category=category,
            reason_code=value.code.value,
            message=value.message,
            field=value.field_path,
        )

    def _emit_local_candidate(
        self,
        *,
        draft_path: Path,
        requested_path: Path,
        payload: bytes,
    ) -> str:
        if requested_path.is_absolute():
            raise ValueError("local candidate output must be relative to the draft")
        if requested_path.suffix.casefold() != ".json":
            raise ValueError("local candidate output must use a .json suffix")
        if len(payload) > MAX_AUTHORING_BYTES:
            raise ValueError("local candidate output exceeds the authoring byte limit")
        draft_root = draft_path.parent.resolve(strict=True)
        unresolved_target = draft_root / requested_path
        resolved_parent = unresolved_target.parent.resolve(strict=True)
        if not resolved_parent.is_relative_to(draft_root):
            raise ValueError("local candidate output escapes the draft authoring tree")
        target = resolved_parent / unresolved_target.name
        if target == draft_path.resolve(strict=True):
            raise ValueError("local candidate output cannot overwrite the draft")
        external_root = self._external_root.resolve(strict=False)
        if target.is_relative_to(external_root):
            raise ValueError("local candidate emission cannot write to the scientific plane")

        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".candidate-tmp",
            dir=resolved_parent,
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb", closefd=True) as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary_path, 0o600)
            os.link(temporary_path, target, follow_symlinks=False)
            directory_descriptor = os.open(
                resolved_parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        finally:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass
        return target.relative_to(draft_root).as_posix()

    def inspect_system_document(self, request: DocumentRequest) -> ApiResult[SystemSummary]:
        operation = "system.inspect-document"
        try:
            value = load_authoring(request.path)
            if isinstance(value, CampaignPackage):
                system = value.system
            elif isinstance(value, SystemSpec):
                system = value
            else:
                raise AuthoringCodecError("document does not contain a SystemSpec")
        except (AuthoringCodecError, FileNotFoundError, OSError, ValueError) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        relation = system.relation
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=SystemSummary(
                system_id=system.system_id,
                world_id=system.world.world_id,
                world_kind=system.world.kind.value,
                relation_id=relation.relation_id,
                denominator_quantity_ids=relation.denominator_quantity_ids,
                history_quantity_ids=relation.history_quantity_ids,
                action_quantity_ids=relation.action_quantity_ids,
                receiver_quantity_ids=relation.receiver_quantity_ids,
                horizon_id=relation.horizon.horizon_id,
                fingerprint=system.fingerprint(),
            ),
        )

    def compile_campaign(self, request: CompileCampaignRequest) -> ApiResult[CompilationSummary]:
        operation = "campaign.compile"
        try:
            value, run_plan, execution_plan = self._compile_path(request.path)
        except (
            AuthoringCodecError,
            FileNotFoundError,
            OSError,
            ProtocolCompilationError,
            ValueError,
        ) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=CompilationSummary(
                package_id=value.package_id,
                package_fingerprint=value.fingerprint(),
                run_plan_id=run_plan.run_plan_id,
                run_plan_fingerprint=run_plan.fingerprint(),
                execution_plan_id=execution_plan.execution_plan_id,
                execution_plan_fingerprint=execution_plan.fingerprint(),
                execution_topology_sha256=execution_topology_sha256(
                    execution_plan,
                    value.execution_resource_envelope_spec
                    if isinstance(value, ExperimentPackage)
                    else None,
                ),
                registry_fingerprint=value.registry.fingerprint(),
                topological_step_ids=run_plan.topological_step_ids(),
                parallel_task_groups=execution_plan.parallel_ready_groups(),
                nonactuating=execution_plan.nonactuating,
                controller_requested=run_plan.controller_requested,
            ),
        )

    def _conditional_execution_records(
        self,
        *,
        package: CampaignPackage | IssuedCampaignPackageRoot,
        parent_record_path: Path | None,
        parent_input_binding_paths: tuple[Path, ...] = (),
    ) -> tuple[
        tuple[ConditionalChildInstantiation, ...],
        tuple[FrozenParentInputBinding, ...],
    ]:
        if parent_record_path is not None and parent_input_binding_paths:
            raise ValueError(
                "a derived parent record and issued parent-input bindings are mutually exclusive"
            )
        if parent_input_binding_paths:
            if not isinstance(
                package,
                (
                    IssuedCampaignPackage,
                    IssuedStudyPackage,
                    EnvelopeExperimentPackage,
                    ExperimentPackage,
                ),
            ):
                raise ValueError("parent-input bindings require an issued campaign")
            candidate = issued_standard_candidate(package)
            base_candidate = issued_base_candidate(package)
            graph = base_candidate.scientific_graph
            substitutions = tuple(
                value
                for value in graph.external_inputs
                if value.scientific_role is ScientificInputRole.PARENT_RECEIPT
                and value.content_identity_policy
                is ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION
            )
            bindings = tuple(
                sorted(
                    (
                        decode_canonical_bytes(
                            read_bounded_bytes(
                                path,
                                maximum_bytes=2 * 1024 * 1024,
                            ),
                            FrozenParentInputBinding,
                            maximum_bytes=2 * 1024 * 1024,
                        )
                        for path in parent_input_binding_paths
                    ),
                    key=lambda value: value.external_input_id,
                )
            )
            if tuple(value.external_input_id for value in bindings) != tuple(
                sorted(value.input_id for value in substitutions)
            ):
                raise ValueError(
                    "issued parent-input bindings differ from the graph substitution roster"
                )
            candidate_identity = ObjectIdentity.from_record(candidate.candidate_id, candidate)
            graph_sha256 = graph.fingerprint()
            specifications = {value.input_id: value for value in substitutions}
            for binding in bindings:
                specification = specifications[binding.external_input_id]
                if (
                    binding.candidate != candidate_identity
                    or binding.scientific_graph_sha256 != graph_sha256
                    or binding.parent_record.object_schema != specification.payload_schema
                    or len(binding.parent_record_payload.encode("utf-8"))
                    > specification.maximum_size_bytes
                    or binding.outcome_access is not specification.outcome_access
                    or binding.visibility_ceiling is not specification.visibility_ceiling
                ):
                    raise ValueError(
                        "issued parent-input binding differs from its target graph slot"
                    )
            return (), bindings
        standard_package = (
            package.base
            if isinstance(
                package,
                (EnvelopeExperimentPackage, ExperimentPackage),
            )
            else package
        )
        if not isinstance(standard_package, IssuedStudyPackage):
            if parent_record_path is not None:
                raise ValueError("conditional parent records require a standard issued campaign")
            return (), ()
        issued_candidate = standard_package.issued_study.candidate
        requires_parent = _requires_parent_record(issued_candidate.base_candidate.scientific_graph)
        if not requires_parent:
            if parent_record_path is not None:
                raise ValueError("issued campaign has no frozen parent-record input")
            return (), ()
        if parent_record_path is None:
            raise ValueError("conditional issued campaign requires --parent-record")
        provider = self._candidate_context_provider
        if provider is None:
            raise ValueError("candidate compilation context is not composed")
        substitution_inputs = tuple(
            value
            for value in issued_candidate.base_candidate.scientific_graph.external_inputs
            if value.scientific_role is ScientificInputRole.PARENT_RECEIPT
            and value.content_identity_policy is ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION
        )
        if issued_candidate.base_candidate.conditional_successor is None:
            if len(substitution_inputs) != 1:
                raise ValueError(
                    "root parent binding requires one frozen parent-record substitution"
                )
            resolver = self._conditional_successor_resolver
            if resolver is None:
                raise ValueError("parent-record decoding is not composed")
            parent_record = load_registered_authoring(
                parent_record_path,
                root_schemas=resolver.parent_record_schemas,
            )
            parent_record_id = next(
                (
                    value
                    for value in (
                        getattr(parent_record, "qualification_id", None),
                        getattr(parent_record, "result_id", None),
                        getattr(parent_record, "adjudication_id", None),
                    )
                    if isinstance(value, str)
                ),
                f"parent-record.{parent_record.fingerprint()[:24]}",
            )
            binding = bind_frozen_parent_input(
                candidate=issued_candidate.base_candidate,
                external_input_id=substitution_inputs[0].input_id,
                parent_record_id=parent_record_id,
                parent_record=parent_record,
            )
            return (), (binding,)
        authoring_package = standard_package.issued_study.authoring_package
        context_resolution = provider.resolve_standard(authoring_package)
        parent_compilation = compile_study_candidate(
            authoring_package=authoring_package,
            authoring_materialization=issued_candidate.base_candidate.authoring_materialization,
            context=context_resolution.context,
            readiness_diagnostics=context_resolution.diagnostics,
        )
        resolution = self._resolve_conditional_child(
            parent_compilation=parent_compilation,
            parent_record_path=parent_record_path,
        )
        if resolution.compilation is None or resolution.parent_input_binding is None:
            reasons = resolution.instantiation.reason_codes
            raise ValueError(
                "conditional successor is not execution eligible: " + ",".join(reasons)
            )
        if resolution.compilation.candidate != issued_candidate:
            raise ValueError("parent record derives another candidate than the issued campaign")
        return (
            (resolution.instantiation,),
            (resolution.parent_input_binding,),
        )

    def run_campaign(
        self,
        request: RunCampaignRequest,
    ) -> ApiResult[RunExecutionSummary | WriteEffectSummary | CampaignAuthorityRequired]:
        operation = "campaign.run"
        try:
            package, run_plan, execution_plan = self._compile_path(request.path)
            conditional_instantiations, parent_input_bindings = self._conditional_execution_records(
                package=package,
                parent_record_path=request.parent_record_path,
                parent_input_binding_paths=request.parent_input_binding_paths,
            )
        except (
            AuthoringCodecError,
            FileNotFoundError,
            OSError,
            ProtocolCompilationError,
            ValueError,
        ) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        except Exception as error:
            return self._execution_error(operation, error)
        if self._execution_service is None:
            return self._execution_unavailable(operation)
        try:
            if not request.confirmed:
                closure = self._execution_service.preexecution_contract_closure(
                    package=package,
                    run_plan=run_plan,
                    execution_plan=execution_plan,
                )
                if not closure.passed:
                    raise CampaignValidationError(
                        "experiment contract closure failed: " + ",".join(closure.reason_codes)
                    )
            effects = self._execution_service.write_effects(run_plan.run_plan_id)
            resource_admission = self._execution_service.resource_admission(execution_plan)
        except Exception as error:
            return self._execution_error(operation, error)
        if not request.confirmed:
            return ApiResult(
                operation=operation,
                status=OperationStatus.BLOCKED,
                payload=self._write_preview(
                    operation,
                    effects,
                    minimum_free_bytes=execution_plan.minimum_free_bytes,
                    execution_assurance_profile=(
                        self._execution_service.execution_assurance_profile.value
                    ),
                    execution_assurance_codes=resource_admission.assurance_codes,
                    execution_resource_reason_codes=resource_admission.reason_codes,
                ),
                reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
            )
        try:
            summary = self._execution_service.execute(
                package=package,
                run_plan=run_plan,
                execution_plan=execution_plan,
                at_utc=self._study_issue_clock.now_utc(),
                reveal_authority_id=request.reveal_authority_id,
                conditional_instantiations=conditional_instantiations,
                parent_input_bindings=parent_input_bindings,
            )
        except (CampaignExecutionError, KeyError, OSError, subprocess.SubprocessError) as error:
            return self._execution_error(operation, error)
        except Exception as error:
            return self._execution_error(operation, error)
        return self._terminal_execution_result(
            operation,
            summary,
            package=package,
            execution_plan=execution_plan,
            reveal_authority_supplied=request.reveal_authority_id is not None,
        )

    def inspect_exploration_portfolio(
        self,
        request: DocumentRequest,
        *,
        operation: str = "campaign.explore-detect",
    ) -> ApiResult[ExplorationPortfolioSummary]:
        try:
            package = self._exploration_path(request.path)
        except (AuthoringCodecError, FileNotFoundError, OSError, ValueError) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        selected = tuple(
            item.proposal_id
            for item in package.exploration_plan.selections
            if item.disposition.value == "SELECTED"
        )
        rejected = tuple(
            item.proposal_id
            for item in package.exploration_plan.selections
            if item.disposition.value != "SELECTED"
        )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=ExplorationPortfolioSummary(
                package_id=package.package_id,
                snapshot_id=package.snapshot.snapshot_id,
                plan_id=package.exploration_plan.plan_id,
                proposal_ids=tuple(item.proposal_id for item in package.exploration_plan.proposals),
                selected_proposal_ids=selected,
                rejected_proposal_ids=rejected,
                projection_ids=package.snapshot.projection_ids,
                outcome_access=package.exploration_plan.outcome_access.value,
                visibility_ceiling=package.exploration_plan.visibility_ceiling.value,
                promotable=package.exploration_plan.visibility_ceiling.is_promotable,
            ),
        )

    def compile_exploration(
        self,
        request: DocumentRequest,
    ) -> ApiResult[ExplorationCompilationSummary]:
        operation = "campaign.explore-compile"
        try:
            package, execution_plan = self._compile_exploration_path(request.path)
        except (
            AuthoringCodecError,
            FileNotFoundError,
            OSError,
            ProtocolCompilationError,
            ValueError,
        ) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        plan = package.exploration_plan
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=ExplorationCompilationSummary(
                package_id=package.package_id,
                package_fingerprint=package.fingerprint(),
                plan_id=plan.plan_id,
                plan_fingerprint=plan.fingerprint(),
                execution_plan_id=execution_plan.execution_plan_id,
                execution_plan_fingerprint=execution_plan.fingerprint(),
                registry_fingerprint=package.registry.fingerprint(),
                topological_task_ids=execution_plan.topological_task_ids(),
                parallel_task_groups=execution_plan.parallel_ready_groups(),
                outcome_access=plan.outcome_access.value,
                visibility_ceiling=plan.visibility_ceiling.value,
                promotable=plan.visibility_ceiling.is_promotable,
                nonactuating=execution_plan.nonactuating,
            ),
        )

    def run_exploration(
        self,
        request: RunCampaignRequest,
    ) -> ApiResult[RunExecutionSummary | WriteEffectSummary | CampaignAuthorityRequired]:
        operation = "campaign.explore-run"
        try:
            package, execution_plan = self._compile_exploration_path(request.path)
        except (
            AuthoringCodecError,
            FileNotFoundError,
            OSError,
            ProtocolCompilationError,
            ValueError,
        ) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        except Exception as error:
            return self._execution_error(operation, error)
        if self._execution_service is None:
            return self._execution_unavailable(operation)
        try:
            effects = self._execution_service.write_effects(package.exploration_plan.plan_id)
            resource_admission = self._execution_service.resource_admission(execution_plan)
        except Exception as error:
            return self._execution_error(operation, error)
        if not request.confirmed:
            return ApiResult(
                operation=operation,
                status=OperationStatus.BLOCKED,
                payload=self._write_preview(
                    operation,
                    effects,
                    minimum_free_bytes=execution_plan.minimum_free_bytes,
                    execution_assurance_profile=(
                        self._execution_service.execution_assurance_profile.value
                    ),
                    execution_assurance_codes=resource_admission.assurance_codes,
                    execution_resource_reason_codes=resource_admission.reason_codes,
                ),
                reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
            )
        try:
            summary = self._execution_service.execute_exploration(
                package=package,
                execution_plan=execution_plan,
            )
        except (CampaignExecutionError, KeyError, OSError, subprocess.SubprocessError) as error:
            return self._execution_error(operation, error)
        except Exception as error:
            return self._execution_error(operation, error)
        return self._terminal_execution_result(operation, summary)

    def synthesize_exploration(
        self,
        request: DocumentRequest,
    ) -> ApiResult[HypothesisSummary]:
        operation = "campaign.explore-synthesize"
        try:
            package, execution_plan = self._compile_exploration_path(request.path)
            value = self._resolved_exploration_records(
                package,
                execution_plan,
            )[1]
        except (AuthoringCodecError, FileNotFoundError, OSError, ValueError) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        except CampaignExecutionError as error:
            failed = self._execution_error(operation, error)
            return ApiResult(
                operation=operation,
                status=failed.status,
                errors=failed.errors,
                reason_codes=failed.reason_codes,
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=HypothesisSummary(
                hypothesis_set_id=value.hypothesis_set_id,
                finding_ids=value.finding_ids,
                hypothesis_ids=tuple(item.hypothesis_id for item in value.hypotheses),
                visibility_ceiling=value.visibility_ceiling.value,
                promotable=value.visibility_ceiling.is_promotable,
                result_basis=(
                    "AUTHORED_FIXTURE_REPLAY"
                    if isinstance(package, DualLoopPackage)
                    else "VERIFIED_RECEIPT_OUTPUTS"
                ),
            ),
        )

    def nominate_next(
        self,
        request: DocumentRequest,
    ) -> ApiResult[NominationSummary]:
        operation = "campaign.nominate-next"
        try:
            package, execution_plan = self._compile_exploration_path(request.path)
            findings, hypotheses = self._resolved_exploration_records(
                package,
                execution_plan,
            )
            decision, obligations = self._nomination_batch(
                package,
                findings=findings,
                hypotheses=hypotheses,
            )
        except (AuthoringCodecError, FileNotFoundError, OSError, ValueError) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        except CampaignExecutionError as error:
            failed = self._execution_error(operation, error)
            return ApiResult(
                operation=operation,
                status=failed.status,
                errors=failed.errors,
                reason_codes=failed.reason_codes,
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=NominationSummary(
                decision_id=decision.decision_id,
                decision_kind=decision.kind.value,
                nomination_ids=tuple(
                    nomination.nomination_id for nomination in decision.nominations
                ),
                requested_rungs=tuple(sorted({item.requested_rung.value for item in obligations})),
                fresh_evidence_required=all(
                    item.fresh_evidence_required for item in decision.nominations
                ),
                source_visibility_ceiling=decision.visibility_ceiling.value,
                promotable=decision.visibility_ceiling.is_promotable,
                result_basis=(
                    "AUTHORED_FIXTURE_REPLAY"
                    if isinstance(package, DualLoopPackage)
                    else "VERIFIED_RECEIPT_OUTPUTS"
                ),
            ),
        )

    def propose_next(
        self,
        request: DocumentRequest,
    ) -> ApiResult[ProposalSummary]:
        operation = "campaign.propose-next"
        try:
            package, execution_plan = self._compile_exploration_path(request.path)
            findings, hypotheses = self._resolved_exploration_records(
                package,
                execution_plan,
            )
            designed = self._designed_experiments(
                package,
                findings=findings,
                hypotheses=hypotheses,
            )
        except (AuthoringCodecError, FileNotFoundError, OSError, ValueError) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        except CampaignExecutionError as error:
            failed = self._execution_error(operation, error)
            return ApiResult(
                operation=operation,
                status=failed.status,
                errors=failed.errors,
                reason_codes=failed.reason_codes,
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=ProposalSummary(
                proposal_ids=tuple(item.proposal_id for item in designed),
                experiment_ids=tuple(item.candidate_experiment.experiment_id for item in designed),
                readiness=tuple(item.candidate_experiment.readiness.value for item in designed),
                authority_action=package.design_context.authority_action.value,
                evaluation_visibility_ceiling="PROSPECTIVE",
                source_visibility_ceiling="OUTCOME_VISIBLE",
                grants_claim_promotion=False,
                result_basis=(
                    "AUTHORED_FIXTURE_REPLAY"
                    if isinstance(package, DualLoopPackage)
                    else "VERIFIED_RECEIPT_OUTPUTS"
                ),
            ),
        )

    def authorize_nonactuating(
        self,
        request: DocumentRequest,
    ) -> ApiResult[AuthorizationSummary]:
        operation = "campaign.authorize-nonactuating"
        try:
            package = self._dual_loop_path(request.path)
            designed = self._designed_experiments(package)
            if len(designed) != 1:
                raise ValueError("approval requires exactly one designed experiment")
        except (AuthoringCodecError, FileNotFoundError, OSError, ValueError) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        # Approval containment: the current authoring root does not persist the
        # complete, outcome-blind approval envelope. Do not
        # invoke the narrower policy gate or synthesize a durable authorization.
        reason_codes = ("COMPLETE_APPROVAL_ENVELOPE_REQUIRED",)
        return ApiResult(
            operation=operation,
            status=OperationStatus.BLOCKED,
            payload=AuthorizationSummary(
                authorization_id=package.approval_request.authorization_id,
                decision=AuthorizationDecision.AUTHORITY_REQUIRED.value,
                action=package.approval_request.action.value,
                reason_codes=reason_codes,
                outcome_access="outcome-blind",
                plan_mutated=False,
                grants_claim_promotion=False,
            ),
            reason_codes=reason_codes,
        )

    def interrupt_campaign_for_maintenance(
        self, request: CampaignMaintenanceInterruptionRequest
    ) -> ApiResult[object]:
        operation = "campaign.interrupt-for-maintenance"
        if self._execution_service is None:
            return self._execution_unavailable(operation)
        try:
            summary = self._execution_service.interrupt_for_maintenance(
                request.authority_path, confirmed=request.confirmed
            )
        except Exception as error:
            return self._execution_error(operation, error)
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED if summary.stopped else OperationStatus.BLOCKED,
            payload=summary,
            reason_codes=(
                (
                    "MAINTENANCE_INTERRUPTED_CONTINUATION_REQUIRES_FRESH_ISSUE"
                    if summary.continuation_run_id is not None
                    else "MAINTENANCE_INTERRUPTED_NO_RETRY_AUTHORIZED"
                )
                if summary.stopped
                else "WRITE_CONFIRMATION_REQUIRED",
            ),
        )

    def stop_campaign_at_receipt_boundary(
        self, request: CampaignMaintenanceStopRequest
    ) -> ApiResult[object]:
        operation = "campaign.stop-at-receipt-boundary"
        if self._execution_service is None:
            return self._execution_unavailable(operation)
        try:
            summary = self._execution_service.stop_at_receipt_boundary(
                request.authority_path, confirmed=request.confirmed
            )
        except Exception as error:
            return self._execution_error(operation, error)
        reason = (
            "MAINTENANCE_STOPPED_WITH_COMPLETE_RECEIPTS"
            if summary.stopped
            else "WRITE_CONFIRMATION_REQUIRED"
            if summary.ready
            else "MAINTENANCE_WAITING_FOR_RECEIPTS"
        )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED if summary.stopped else OperationStatus.BLOCKED,
            payload=summary,
            reason_codes=(reason,),
        )

    def resume_campaign(
        self,
        request: ResumeCampaignRequest,
    ) -> ApiResult[RunExecutionSummary | WriteEffectSummary | CampaignAuthorityRequired]:
        operation = "campaign.resume"
        try:
            validate_stable_id(request.run_id, field_name="run_id")
        except ValueError as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        if self._execution_service is None:
            return self._execution_unavailable(operation)
        try:
            effects = self._execution_service.write_effects(request.run_id)
        except Exception as error:
            return self._execution_error(operation, error)
        if not request.confirmed:
            if request.retry_amendment_path is not None or request.retry_authority_id is not None:
                try:
                    package = self._execution_service.load_persisted_package(request.run_id)
                    run_plan, execution_plan = self._compile_package(package)
                    self._execution_service.preflight_retry(
                        package=package,
                        run_plan=run_plan,
                        execution_plan=execution_plan,
                        amendment_path=request.retry_amendment_path,
                        authority_id=request.retry_authority_id,
                        at_utc=self._study_issue_clock.now_utc(),
                    )
                except Exception as error:
                    return self._execution_error(operation, error)
                return ApiResult(
                    operation=operation,
                    status=OperationStatus.BLOCKED,
                    payload=self._write_preview(operation, effects),
                    reason_codes=(
                        "OPERATIONAL_RETRY_PREFLIGHT_PASSED",
                        "WRITE_CONFIRMATION_REQUIRED",
                    ),
                )
            return ApiResult(
                operation=operation,
                status=OperationStatus.BLOCKED,
                payload=self._write_preview(operation, effects),
                reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
            )
        try:
            persisted_package = self._execution_service.load_persisted_package(request.run_id)
            conditional_instantiations, parent_input_bindings = self._conditional_execution_records(
                package=persisted_package,
                parent_record_path=request.parent_record_path,
                parent_input_binding_paths=request.parent_input_binding_paths,
            )
            run_plan, execution_plan = self._compile_package(persisted_package)
            summary = self._execution_service.resume(
                request.run_id,
                lambda _package: (run_plan, execution_plan),
                at_utc=self._study_issue_clock.now_utc(),
                reveal_authority_id=request.reveal_authority_id,
                conditional_instantiations=conditional_instantiations,
                parent_input_bindings=parent_input_bindings,
                retry_amendment_path=request.retry_amendment_path,
                retry_authority_id=request.retry_authority_id,
                source_amendment_path=request.source_amendment_path,
                source_authority_id=request.source_authority_id,
            )
        except (AuthoringCodecError, ProtocolCompilationError, ValueError) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        except KeyError as error:
            return ApiResult(
                operation=operation,
                status=OperationStatus.NOT_FOUND,
                errors=(
                    ApiError(
                        category=ErrorCategory.NOT_FOUND,
                        reason_code="RUN_NOT_FOUND",
                        message=str(error),
                    ),
                ),
                reason_codes=("RUN_NOT_FOUND",),
            )
        except (CampaignExecutionError, OSError, subprocess.SubprocessError) as error:
            return self._execution_error(operation, error)
        except Exception as error:
            return self._execution_error(operation, error)
        return self._terminal_execution_result(
            operation,
            summary,
            package=persisted_package,
            execution_plan=execution_plan,
            reveal_authority_supplied=request.reveal_authority_id is not None,
        )

    def campaign_status(
        self,
        request: CampaignStatusRequest,
    ) -> ApiResult[RunStatusSummary]:
        operation = "campaign.status"
        try:
            validate_stable_id(request.run_id, field_name="run_id")
            if request.attempt_history_limit <= 0 or request.attempt_history_limit > 1_000:
                raise ValueError("attempt_history_limit must be in [1, 1000]")
            if request.attempt_history_cursor is not None and not request.include_attempt_history:
                raise ValueError("attempt_history_cursor requires include_attempt_history")
        except ValueError as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        if self._execution_service is None:
            return ApiResult(
                operation=operation,
                status=OperationStatus.NOT_FOUND,
                errors=(
                    ApiError(
                        category=ErrorCategory.NOT_FOUND,
                        reason_code="RUN_NOT_FOUND",
                        message=f"run is absent: {request.run_id}",
                    ),
                ),
                reason_codes=("RUN_NOT_FOUND",),
            )
        try:
            summary = self._execution_service.status(
                request.run_id,
                include_attempt_history=request.include_attempt_history,
                attempt_history_limit=request.attempt_history_limit,
                attempt_history_cursor=request.attempt_history_cursor,
            )
        except KeyError:
            return ApiResult(
                operation=operation,
                status=OperationStatus.NOT_FOUND,
                errors=(
                    ApiError(
                        category=ErrorCategory.NOT_FOUND,
                        reason_code="RUN_NOT_FOUND",
                        message=f"run is absent: {request.run_id}",
                    ),
                ),
                reason_codes=("RUN_NOT_FOUND",),
            )
        except CampaignExecutionError as error:
            failed = self._execution_error(operation, error)
            return ApiResult(
                operation=operation,
                status=failed.status,
                errors=failed.errors,
                reason_codes=failed.reason_codes,
            )
        except Exception as error:
            failed = self._execution_error(operation, error)
            return ApiResult(
                operation=operation,
                status=failed.status,
                errors=failed.errors,
                reason_codes=failed.reason_codes,
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=summary,
        )

    def doctor(self, *, route: str | None = None) -> ApiResult[DoctorSummary]:
        operation = "doctor"
        import platform
        from .environment import EnvironmentRoute, inspect_route_dependencies
        from empirical_lawhood.infrastructure.source_origin import require_executing_target_source

        try:
            selected_route = None if route is None else EnvironmentRoute(route)
        except ValueError as error:
            return _validation_error(operation, error)
        source_bound = False
        if self._project_configured:
            try:
                require_executing_target_source(self._repo_root)
                source_bound = True
            except (OSError, PermissionError, ValueError):
                pass
        branch = self._git_value("branch", "--show-current") if source_bound else None
        commit = self._git_value("rev-parse", "HEAD") if source_bound else None
        dirty: bool | None = None
        if source_bound:
            try:
                status = run_bounded_command(
                    ["git", "status", "--porcelain"],
                    cwd=self._repo_root,
                    timeout_seconds=10,
                    maximum_stdout_bytes=1024 * 1024,
                    maximum_stderr_bytes=16 * 1024,
                )
                dirty = status.returncode == 0 and bool(status.stdout)
                if status.returncode != 0:
                    dirty = None
            except (BoundedProcessError, OSError, subprocess.SubprocessError, ValueError):
                pass
        catalog_path = (
            self._catalog_path()
            if self._project_configured
            else self._repo_root / PRODUCTION_RELATIVE_PATH
        )
        if self._project_configured:
            catalog_check = self.check_catalog()
            catalog_present = (
                catalog_check.payload.present
                if catalog_check.payload is not None
                else catalog_path.is_file()
            )
            catalog_state, catalog_reason_codes = self._catalog_diagnostic_state(catalog_check)
            warnings: list[str] = list(catalog_check.reason_codes)
        else:
            catalog_present = False
            catalog_state = "UNCONFIGURED"
            catalog_reason_codes = ("PROJECT_UNCONFIGURED",)
            warnings = ["PROJECT_UNCONFIGURED"]
        if self._project_configured and not source_bound:
            warnings.append("EXECUTING_SOURCE_NOT_BOUND")
        external_present = self._external_root is not None and self._external_root.is_dir()
        diagnostic, diagnostic_errors = self._storage_diagnostic()
        if self._external_root is None:
            warnings.append("OPERATOR_STORAGE_UNCONFIGURED")
        elif not external_present:
            warnings.append("EXTERNAL_STORAGE_ABSENT")
        elif diagnostic is None and not os.access(self._external_root, os.W_OK):
            warnings.append("EXTERNAL_STORAGE_NOT_WRITABLE")
        warnings.extend(diagnostic_errors)
        if diagnostic is not None:
            warnings.extend(diagnostic.reason_codes)
        backends = self._doctor_backends()
        dependencies = inspect_route_dependencies(selected_route)
        if selected_route is not None:
            warnings.extend(code for value in dependencies for code in value.reason_codes)
        authority_boundaries = self._doctor_authority_boundaries()
        action_readiness = self._doctor_action_readiness(
            catalog_path=catalog_path,
            catalog_state=catalog_state,
            catalog_reason_codes=catalog_reason_codes,
            storage=diagnostic,
            storage_errors=diagnostic_errors,
            backends=backends,
        )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=DoctorSummary(
                package_version=importlib.metadata.version("empirical-lawhood"),
                python_version=sys.version.split()[0],
                repository_root=str(self._repo_root) if self._project_configured else "",
                git_branch=branch,
                git_commit=commit,
                git_dirty=dirty,
                capability_count=(
                    len(self._capability_discovery_catalog.registrations)
                    if self._execution_service is None and self._capability_discovery_catalog is not None
                    else (0 if self._execution_service is None else self._execution_service.capability_count)
                ),
                capability_registry_fingerprint=(
                    None
                    if self._execution_service is None
                    else self._execution_service.provider_registry_fingerprint
                ),
                catalog_relative_path=PRODUCTION_RELATIVE_PATH.as_posix(),
                catalog_present=catalog_present,
                catalog_state=catalog_state,
                catalog_reason_codes=catalog_reason_codes,
                external_root="" if self._external_root is None else str(self._external_root),
                external_present=external_present,
                external_writable=(
                    diagnostic.writable
                    if diagnostic is not None
                    else self._external_root is not None and external_present and not self._inspection_storage_read_only and os.access(self._external_root, os.W_OK)
                ),
                mount_active=(None if diagnostic is None else diagnostic.mount_active),
                canonical_contained=(
                    None if diagnostic is None else diagnostic.canonical_contained
                ),
                symlink_safe=(None if diagnostic is None else diagnostic.symlink_safe),
                mount_source=(None if diagnostic is None else diagnostic.mount_source),
                volume_identity=(None if diagnostic is None else diagnostic.volume_identity),
                filesystem_type=(None if diagnostic is None else diagnostic.filesystem_type),
                observed_free_bytes=(
                    None if diagnostic is None else diagnostic.observed_free_bytes
                ),
                effective_write_floor_bytes=(
                    None if diagnostic is None else diagnostic.effective_write_floor_bytes
                ),
                storage_read_ready=(False if diagnostic is None else diagnostic.read_ready),
                storage_write_ready=(False if diagnostic is None else diagnostic.write_ready),
                backends=backends,
                optional_dependencies=dependencies,
                authority_boundaries=authority_boundaries,
                action_readiness=action_readiness,
                warnings=tuple(sorted(set(warnings))),
                environment_route=None if selected_route is None else selected_route.value,
                operating_system=platform.system(),
                architecture=platform.machine(),
            ),
            reason_codes=tuple(sorted(set(warnings))),
        )

    @staticmethod
    def _catalog_diagnostic_state(
        checked: ApiResult[CatalogSummary],
    ) -> tuple[str, tuple[str, ...]]:
        summary = checked.payload
        if summary is not None and not summary.present:
            return "ABSENT", ("LOCAL_CATALOG_ABSENT",)
        if summary is None:
            return "CORRUPT", ("CATALOG_AUDIT_FAILED",)
        if summary.application_id != APPLICATION_ID:
            return "WRONG_IDENTITY", ("CATALOG_APPLICATION_ID_MISMATCH",)
        if summary.schema_version != SCHEMA_VERSION or summary.alembic_revision != ALEMBIC_HEAD:
            return "STALE", ("CATALOG_MIGRATION_REQUIRED",)
        if not summary.schema_valid or summary.integrity_result != "ok":
            return "CORRUPT", ("CATALOG_SCHEMA_OR_INTEGRITY_INVALID",)
        return "VALID", ()

    def _doctor_backends(self) -> tuple[DoctorBackendSummary, ...]:
        service = self._execution_service
        resource_enforcement = service is not None and service.executor_enforces_resources
        trusted_local = (
            service is not None and service.execution_assurance_profile.value == "TRUSTED_LOCAL"
        )
        trusted_enabled = service is not None and (trusted_local or resource_enforcement)
        trusted_reasons = (
            (
                "AGGREGATE_DESCENDANT_LIMITS_NOT_ENFORCED",
                "OS_NETWORK_ISOLATION_NOT_ENFORCED",
            )
            if trusted_local
            else (() if resource_enforcement else ("EXECUTOR_RESOURCE_ENFORCEMENT_UNAVAILABLE",))
        )
        return (
            DoctorBackendSummary(
                backend_id="exploration-no-network",
                state="DISABLED",
                enforces_resources=resource_enforcement,
                enforces_no_network=False,
                reason_codes=("NO_NETWORK_ISOLATION_UNAVAILABLE",),
            ),
            DoctorBackendSummary(
                backend_id="remote-oci",
                state="DISABLED",
                enforces_resources=False,
                enforces_no_network=False,
                reason_codes=("REMOTE_EXECUTION_NOT_SUPPORTED",),
            ),
            DoctorBackendSummary(
                backend_id="trusted-local-process",
                state="ENABLED" if trusted_enabled else "DISABLED",
                enforces_resources=resource_enforcement,
                enforces_no_network=False,
                reason_codes=trusted_reasons,
            ),
        )

    @staticmethod
    def _doctor_authority_boundaries() -> tuple[DoctorAuthorityBoundary, ...]:
        return (
            DoctorAuthorityBoundary(
                "acquisition-source-terms",
                True,
                "SOURCE_ACCESS_AUTHORITY_REQUIRED",
            ),
            DoctorAuthorityBoundary(
                "durable-plan-authorization",
                True,
                "DURABLE_AUTHORIZATION_REQUIRED",
            ),
            DoctorAuthorityBoundary(
                "external-write-confirmation",
                True,
                "WRITE_CONFIRMATION_REQUIRED",
            ),
            DoctorAuthorityBoundary(
                "live-actuation",
                True,
                "LIVE_ACTUATION_AUTHORITY_REQUIRED",
            ),
            DoctorAuthorityBoundary(
                "local-catalog-mutation-confirmation",
                True,
                "WRITE_CONFIRMATION_REQUIRED",
            ),
            DoctorAuthorityBoundary(
                "outcome-reveal",
                True,
                "OUTCOME_REVEAL_AUTHORITY_REQUIRED",
            ),
            DoctorAuthorityBoundary(
                "remote-resource-authority",
                True,
                "REMOTE_RESOURCE_AUTHORITY_REQUIRED",
            ),
        )

    def _doctor_action_readiness(
        self,
        *,
        catalog_path: Path,
        catalog_state: str,
        catalog_reason_codes: tuple[str, ...],
        storage: ExternalStorageDiagnostic | None,
        storage_errors: tuple[str, ...],
        backends: tuple[DoctorBackendSummary, ...],
    ) -> tuple[DoctorActionReadiness, ...]:
        trusted_backend = next(
            value for value in backends if value.backend_id == "trusted-local-process"
        )
        catalog_parent = catalog_path.parent
        if not self._project_configured:
            catalog_parent_ready = False
        elif catalog_parent.exists():
            catalog_parent_ready = (
                catalog_parent.is_dir()
                and not catalog_parent.is_symlink()
                and os.access(catalog_parent, os.W_OK)
            )
        else:
            catalog_parent_ready = os.access(self._repo_root, os.W_OK)
        storage_read_reasons = (
            storage_errors
            if storage is None
            else (() if storage.read_ready else storage.reason_codes)
        )
        storage_write_reasons = (
            storage_errors
            if storage is None
            else (() if storage.write_ready else storage.reason_codes)
        )
        catalog_read_reasons = () if catalog_state == "VALID" else catalog_reason_codes
        catalog_write_reasons = list(catalog_read_reasons)
        if not catalog_parent_ready:
            catalog_write_reasons.append("LOCAL_CATALOG_PARENT_NOT_WRITABLE")
        trusted_blockers = (
            () if trusted_backend.state == "ENABLED" else trusted_backend.reason_codes
        )
        prospective_reasons = tuple(
            sorted(
                set(
                    (
                        *storage_write_reasons,
                        *(() if catalog_state in {"ABSENT", "VALID"} else catalog_reason_codes),
                        *trusted_blockers,
                    )
                )
            )
        )

        def readiness(
            action_id: str,
            reasons: tuple[str, ...],
            authority: tuple[str, ...] = (),
        ) -> DoctorActionReadiness:
            return DoctorActionReadiness(
                action_id=action_id,
                infrastructure_ready=not reasons,
                authority_required=authority,
                executable_now=not reasons and not authority,
                reason_codes=tuple(sorted(set(reasons))),
            )

        return (
            readiness(
                "catalog-initialize",
                (
                    ()
                    if catalog_state == "ABSENT" and catalog_parent_ready
                    else (
                        ("CATALOG_ALREADY_PRESENT",)
                        if catalog_state == "VALID"
                        else tuple(catalog_write_reasons)
                    )
                ),
                ("local-catalog-mutation-confirmation",),
            ),
            readiness("catalog-read", catalog_read_reasons),
            readiness(
                "catalog-write",
                tuple(catalog_write_reasons),
                ("local-catalog-mutation-confirmation",),
            ),
            readiness("external-artifact-read", tuple(storage_read_reasons)),
            readiness(
                "external-artifact-write",
                tuple(storage_write_reasons),
                ("external-write-confirmation",),
            ),
            readiness("local-validation", ()),
            readiness(
                "prospective-execution",
                prospective_reasons,
                ("durable-plan-authorization", "external-write-confirmation"),
            ),
            readiness(
                "exploration-execution",
                prospective_reasons,
                ("outcome-reveal",),
            ),
            readiness(
                "remote-oci-execution",
                ("REMOTE_EXECUTION_NOT_SUPPORTED",),
                ("remote-resource-authority",),
            ),
        )

    def _storage_diagnostic(
        self,
        *,
        operation_minimum_free_bytes: int = 0,
    ) -> tuple[ExternalStorageDiagnostic | None, tuple[str, ...]]:
        if self._execution_service is None and self._inspection_storage_root is None:
            return None, ("EXTERNAL_STORAGE_INSPECTION_UNAVAILABLE",)
        try:
            if self._execution_service is not None:
                diagnostic = self._execution_service.storage_diagnostic(
                    operation_minimum_free_bytes=operation_minimum_free_bytes
                )
            else:
                assert self._inspection_storage_root is not None
                diagnostic = self._inspection_storage_root.diagnostic(
                    operation_minimum_free_bytes=operation_minimum_free_bytes
                )
                if self._inspection_storage_read_only:
                    diagnostic = replace(
                        diagnostic,
                        writable=False,
                        write_ready=False,
                        reason_codes=tuple(
                            sorted({*diagnostic.reason_codes, "OPERATOR_STORAGE_READ_ONLY"})
                        ),
                    )
            return diagnostic, ()
        except (ArtifactPlaneError, OSError, ValueError):
            return None, ("EXTERNAL_STORAGE_INSPECTION_FAILED",)

    def _write_preview(
        self,
        operation: str,
        effects: tuple[str, ...],
        *,
        minimum_free_bytes: int = 0,
        execution_assurance_profile: str | None = None,
        execution_assurance_codes: tuple[str, ...] = (),
        execution_resource_reason_codes: tuple[str, ...] = (),
    ) -> WriteEffectSummary:
        diagnostic, errors = self._storage_diagnostic(
            operation_minimum_free_bytes=minimum_free_bytes
        )
        return WriteEffectSummary(
            action=operation,
            confirmed=False,
            effects=effects,
            observed_free_bytes=(None if diagnostic is None else diagnostic.observed_free_bytes),
            effective_write_floor_bytes=(
                None if diagnostic is None else diagnostic.effective_write_floor_bytes
            ),
            storage_write_ready=(None if diagnostic is None else diagnostic.write_ready),
            storage_reason_codes=(errors if diagnostic is None else diagnostic.reason_codes),
            execution_assurance_profile=execution_assurance_profile,
            execution_assurance_codes=execution_assurance_codes,
            execution_resource_reason_codes=execution_resource_reason_codes,
        )

    def check_catalog(self) -> ApiResult[CatalogSummary]:
        operation = "catalog.check"
        path = self._catalog_path()
        if not path.is_file():
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=CatalogSummary(
                    catalog_relative_path=PRODUCTION_RELATIVE_PATH.as_posix(),
                    present=False,
                    schema_valid=None,
                    schema_version=None,
                    alembic_revision=None,
                    page_count=None,
                    page_size=None,
                    forbidden_payload_columns=(),
                    application_id=None,
                    integrity_result=None,
                ),
                reason_codes=("LOCAL_CATALOG_ABSENT",),
            )
        engine = None
        try:
            engine = create_read_only_catalog_engine(path)
            audit = schema_audit(engine)
        except CatalogSchemaAuditLimitExceeded as error:
            return self._catalog_audit_limit_error(operation, error)
        except Exception:  # infrastructure boundary normalized below
            LOGGER.exception("catalog audit failed")
            return ApiResult(
                operation=operation,
                status=OperationStatus.FAILED,
                errors=(
                    ApiError(
                        category=ErrorCategory.STORAGE,
                        reason_code="CATALOG_AUDIT_FAILED",
                        message="catalog audit failed",
                    ),
                ),
                reason_codes=("CATALOG_AUDIT_FAILED",),
            )
        finally:
            if engine is not None:
                engine.dispose()
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED if audit.passed else OperationStatus.INVALID,
            payload=CatalogSummary(
                catalog_relative_path=PRODUCTION_RELATIVE_PATH.as_posix(),
                present=True,
                schema_valid=audit.passed,
                schema_version=audit.user_version,
                alembic_revision=audit.alembic_revision,
                page_count=audit.page_count,
                page_size=audit.page_size,
                forbidden_payload_columns=audit.forbidden_payload_columns,
                application_id=audit.application_id,
                integrity_result=audit.integrity_result,
            ),
            reason_codes=() if audit.passed else ("CATALOG_SCHEMA_INVALID",),
        )

    def initialize_catalog(
        self,
        request: DatabaseMutationRequest,
    ) -> ApiResult[DatabaseMutationSummary | WriteEffectSummary]:
        operation = "db.init"
        if not request.confirmed:
            return ApiResult(
                operation=operation,
                status=OperationStatus.BLOCKED,
                payload=WriteEffectSummary(
                    action=operation,
                    confirmed=False,
                    effects=("local:.empirical-lawhood/experiment_catalog.sqlite3",),
                ),
                reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
            )
        try:
            engine = (
                self._execution_service.engine_factory()
                if self._execution_service is not None
                else initialize_production_catalog(self._repo_root)
            )
            audit = schema_audit(engine)
        except CatalogSchemaAuditLimitExceeded as error:
            return self._catalog_audit_limit_error(operation, error)
        except (OSError, RuntimeError, ValueError) as error:
            return self._storage_error(operation, "CATALOG_INITIALIZATION_FAILED", error)
        finally:
            if "engine" in locals():
                engine.dispose()
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=self._database_mutation(operation, audit),
        )

    def compact_catalog(
        self,
        request: DatabaseMutationRequest,
    ) -> ApiResult[DatabaseMutationSummary | WriteEffectSummary]:
        operation = "db.compact"
        path = self._catalog_path()
        if not path.is_file():
            return self._catalog_not_found(operation)
        if not request.confirmed:
            return ApiResult(
                operation=operation,
                status=OperationStatus.BLOCKED,
                payload=WriteEffectSummary(
                    action=operation,
                    confirmed=False,
                    effects=("local:.empirical-lawhood/experiment_catalog.sqlite3",),
                ),
                reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
            )
        engine = create_catalog_engine(f"sqlite+pysqlite:///{path}")
        try:
            with engine.begin() as connection:
                connection.exec_driver_sql("PRAGMA incremental_vacuum")
            audit = schema_audit(engine)
        except CatalogSchemaAuditLimitExceeded as error:
            return self._catalog_audit_limit_error(operation, error)
        except Exception as error:
            return self._storage_error(operation, "CATALOG_COMPACTION_FAILED", error)
        finally:
            engine.dispose()
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED if audit.passed else OperationStatus.INVALID,
            payload=self._database_mutation(operation, audit),
            reason_codes=() if audit.passed else ("CATALOG_SCHEMA_INVALID",),
        )

    def list_datasets(
        self,
        request: DatasetListRequest,
    ) -> ApiResult[DatasetListSummary]:
        operation = "dataset.list"
        try:
            cursor = _validate_dataset_list_request(request)
        except (AuthoringCodecError, CanonicalizationError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="DATASET_CATALOG_REQUEST_INVALID",
                message=str(error),
            )
        provider = self._dataset_catalog_session_provider
        if provider is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="DATASET_PROJECTION_AUTHORITY_UNAVAILABLE",
                message="authenticated dataset projection authority is unavailable",
            )
        lookup = DatasetCatalogLookup(
            work_limit=DATASET_CATALOG_WORK_LIMIT,
            limit=request.limit,
            cursor=cursor,
            family_ids=request.family_ids,
            release_ids=request.release_ids,
            provider_ids=request.provider_ids,
            external_identifiers=request.external_identifiers,
            custody_states=request.custody_states,
            acquisition_states=request.acquisition_states,
            experiment_spec_ids=request.experiment_spec_ids,
            roles=request.roles,
        )
        try:
            with provider.open() as service:
                if not isinstance(service, DatasetCatalogReadService):
                    return _dataset_catalog_error(
                        operation,
                        status=OperationStatus.BLOCKED,
                        category=ErrorCategory.CAPABILITY,
                        reason_code="DATASET_CATALOG_PROVIDER_INVALID",
                        message="dataset catalog provider returned an invalid session",
                    )
                page = service.page(lookup)
            records = tuple(_dataset_catalog_record_summary(record) for record in page.records)
            next_cursor = (
                None
                if page.next_cursor is None
                else _encode_dataset_catalog_cursor(page.next_cursor)
            )
        except DatasetCatalogReadAuthenticationError:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.IDENTITY,
                reason_code="DATASET_PROJECTION_AUTHENTICATION_FAILED",
                message="dataset projection differs from its authenticated authority",
            )
        except (
            DatasetCatalogQueryWorkLimitExceeded,
            DatasetCatalogTrustedQueryWorkLimitExceeded,
        ):
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code="DATASET_CATALOG_QUERY_WORK_LIMIT_EXCEEDED",
                message="dataset catalog query exhausted its bounded work budget",
            )
        except (AuthoringCodecError, CanonicalizationError, TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="DATASET_CATALOG_REQUEST_INVALID",
                message=str(error),
            )
        except Exception as error:
            return self._storage_error(operation, "DATASET_CATALOG_READ_FAILED", error)
        boundary = page.boundary
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=DatasetListSummary(
                records=records,
                returned_count=len(records),
                limit=page.limit,
                records_examined=page.records_examined,
                query_steps=page.query_steps,
                max_records_examined=boundary.work_limit.max_records_examined,
                max_query_steps=boundary.work_limit.max_query_steps,
                has_more=page.has_more,
                next_cursor=next_cursor,
                snapshot_fingerprint=boundary.snapshot_fingerprint,
                projection_state_fingerprint=boundary.projection_state_fingerprint,
                projection_anchor_fingerprint=boundary.projection_anchor_fingerprint,
            ),
        )

    def show_dataset(
        self,
        request: DatasetShowRequest,
    ) -> ApiResult[DatasetShowSummary]:
        operation = "dataset.show"
        try:
            _validate_dataset_show_request(request)
        except (TypeError, ValueError) as error:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.INVALID,
                category=ErrorCategory.VALIDATION,
                reason_code="DATASET_CATALOG_REQUEST_INVALID",
                message=str(error),
            )
        provider = self._dataset_catalog_session_provider
        if provider is None:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.AUTHORITY,
                reason_code="DATASET_PROJECTION_AUTHORITY_UNAVAILABLE",
                message="authenticated dataset projection authority is unavailable",
            )
        family = None
        release = None
        try:
            with provider.open() as service:
                if not isinstance(service, DatasetCatalogReadService):
                    return _dataset_catalog_error(
                        operation,
                        status=OperationStatus.BLOCKED,
                        category=ErrorCategory.CAPABILITY,
                        reason_code="DATASET_CATALOG_PROVIDER_INVALID",
                        message="dataset catalog provider returned an invalid session",
                    )
                service.page(
                    DatasetCatalogLookup(
                        work_limit=DATASET_CATALOG_WORK_LIMIT,
                        limit=1,
                    )
                )
                if request.kind in {None, DatasetCatalogRecordKind.FAMILY}:
                    family = service.family(request.dataset_id)
                if request.kind in {None, DatasetCatalogRecordKind.RELEASE}:
                    release = service.release(request.dataset_id)
        except DatasetCatalogReadAuthenticationError:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.IDENTITY,
                reason_code="DATASET_PROJECTION_AUTHENTICATION_FAILED",
                message="dataset projection differs from its authenticated authority",
            )
        except (
            DatasetCatalogQueryWorkLimitExceeded,
            DatasetCatalogTrustedQueryWorkLimitExceeded,
        ):
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.BLOCKED,
                category=ErrorCategory.CAPABILITY,
                reason_code="DATASET_CATALOG_QUERY_WORK_LIMIT_EXCEEDED",
                message="dataset catalog query exhausted its bounded work budget",
            )
        except Exception as error:
            return self._storage_error(operation, "DATASET_CATALOG_READ_FAILED", error)
        matches: tuple[DatasetCatalogRecord, ...] = tuple(
            value for value in (family, release) if value is not None
        )
        if not matches:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.NOT_FOUND,
                category=ErrorCategory.NOT_FOUND,
                reason_code="DATASET_NOT_FOUND",
                message=f"dataset is absent: {request.dataset_id}",
            )
        if len(matches) > 1:
            return _dataset_catalog_error(
                operation,
                status=OperationStatus.CONFLICT,
                category=ErrorCategory.CONFLICT,
                reason_code="DATASET_ID_AMBIGUOUS",
                message="dataset ID resolves to both a family and a release",
            )
        try:
            summary = _dataset_show_summary(matches[0])
        except (CanonicalizationError, TypeError, ValueError) as error:
            return self._storage_error(operation, "DATASET_CATALOG_READ_FAILED", error)
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=summary,
        )

    def query_catalog(
        self,
        request: CatalogQueryRequest,
    ) -> ApiResult[CatalogQuerySummary]:
        operation = "catalog.query"
        try:
            after_object_id = _validate_catalog_query(request)
        except ValueError as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        path = self._catalog_path()
        if not path.is_file():
            return ApiResult(
                operation=operation,
                status=OperationStatus.SUCCEEDED,
                payload=CatalogQuerySummary(
                    matched_count=0,
                    returned_count=0,
                    objects=(),
                    match_count_limit=request.match_count_limit,
                    query_work_limit=DEFAULT_CATALOG_QUERY_WORK_LIMIT,
                ),
                reason_codes=("LOCAL_CATALOG_ABSENT",),
            )
        engine = None
        try:
            engine = create_read_only_catalog_engine(path)
            audit = catalog_query_preflight(engine)
            if not audit.passed:
                return ApiResult(
                    operation=operation,
                    status=OperationStatus.INVALID,
                    errors=(
                        ApiError(
                            category=ErrorCategory.STORAGE,
                            reason_code="CATALOG_SCHEMA_INVALID",
                            message="catalog schema or identity is invalid",
                        ),
                    ),
                    reason_codes=("CATALOG_SCHEMA_INVALID",),
                )
            page = SQLiteCatalogRepository(engine).query_objects(
                CatalogObjectQuery(
                    object_id=request.object_id,
                    kind=request.kind,
                    categorical_status=request.categorical_status,
                    relation_id=request.relation_id,
                    denominator_gauge_id=request.denominator_gauge_id,
                    response_gauge_id=request.response_gauge_id,
                    horizon_id=request.horizon_id,
                    native_unit=request.native_unit,
                    knowledge_edge_relation=request.knowledge_edge_relation,
                    knowledge_edge_scope_id=request.knowledge_edge_scope_id,
                    lineage_object_id=request.lineage_object_id,
                    after_object_id=after_object_id,
                    limit=request.limit,
                    match_count_limit=request.match_count_limit,
                )
            )
        except CatalogQueryWorkLimitExceeded:
            return ApiResult(
                operation=operation,
                status=OperationStatus.BLOCKED,
                errors=(
                    ApiError(
                        category=ErrorCategory.CAPABILITY,
                        reason_code="CATALOG_QUERY_WORK_LIMIT_EXCEEDED",
                        message=(
                            "catalog query exhausted its bounded work budget; "
                            "use a more selective indexed filter"
                        ),
                    ),
                ),
                reason_codes=("CATALOG_QUERY_WORK_LIMIT_EXCEEDED",),
            )
        except Exception as error:
            return self._storage_error(operation, "CATALOG_QUERY_FAILED", error)
        finally:
            if engine is not None:
                engine.dispose()
        summaries = tuple(
            CatalogObjectSummary(
                object_id=value.object_id,
                kind=value.kind,
                object_schema=value.object_schema,
                object_fingerprint=value.object_fingerprint,
                visibility_ceiling=value.visibility_ceiling,
                categorical_status=value.categorical_status,
                storage_root_id=value.storage_root_id,
                external_relative_path=value.external_relative_path,
                external_identity_kind=(
                    None
                    if value.external_identity_kind is None
                    else value.external_identity_kind.lower()
                ),
            )
            for value in page.rows
        )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=CatalogQuerySummary(
                matched_count=page.matched_count,
                returned_count=len(summaries),
                objects=summaries,
                next_cursor=(
                    None
                    if page.next_after_object_id is None
                    else _catalog_cursor(page.next_after_object_id, request)
                ),
                match_count_limit=page.match_count_limit,
                matched_count_truncated=page.matched_count_truncated,
                has_more=page.has_more,
                query_work_limit=page.query_work_limit,
                query_work_steps=page.query_work_steps,
            ),
        )

    def inspect_system_id(
        self,
        system_id: str,
    ) -> ApiResult[CatalogQuerySummary]:
        operation = "system.inspect"
        try:
            validate_stable_id(system_id, field_name="system_id")
        except ValueError as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        result = self.query_catalog(
            CatalogQueryRequest(object_id=system_id, kind="SYSTEM_SPEC", limit=1)
        )
        if not result.succeeded:
            return ApiResult(
                operation=operation,
                status=result.status,
                payload=result.payload,
                errors=result.errors,
                reason_codes=result.reason_codes,
            )
        if result.payload is None or not result.payload.objects:
            return ApiResult(
                operation=operation,
                status=OperationStatus.NOT_FOUND,
                errors=(
                    ApiError(
                        category=ErrorCategory.NOT_FOUND,
                        reason_code="SYSTEM_NOT_FOUND",
                        message=f"system is absent from the catalog: {system_id}",
                    ),
                ),
                reason_codes=("SYSTEM_NOT_FOUND",),
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=result.payload,
        )

    def inspect_scientific_object(
        self,
        object_id: str,
        *,
        requested_kind: str,
        operation: str,
    ) -> ApiResult[ScientificInspectionSummary]:
        """Inspect only catalog metadata; artifact and sealed bytes stay unopened."""

        allowed = {
            "authority": {"AUTHORITY_POLICY", "AUTHORIZATION_RECORD"},
            "law": {"RESPONSE_LAW"},
            "atlas": {"ATLAS_PATCH"},
            "admission": {"ADMISSION_SET"},
            "control": {"CONTROLLER_SPEC"},
        }
        if requested_kind not in allowed:
            invalid = _validation_error(operation, ValueError("unknown inspection kind"))
            return ApiResult(
                operation=operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        try:
            validate_stable_id(object_id, field_name="object_id")
        except ValueError as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        result = self.query_catalog(CatalogQueryRequest(object_id=object_id, limit=1))
        if not result.succeeded:
            return ApiResult(
                operation=operation,
                status=result.status,
                errors=result.errors,
                reason_codes=result.reason_codes,
            )
        objects = () if result.payload is None else result.payload.objects
        if not objects or objects[0].kind not in allowed[requested_kind]:
            return ApiResult(
                operation=operation,
                status=OperationStatus.NOT_FOUND,
                payload=ScientificInspectionSummary(
                    requested_kind=requested_kind,
                    object_id=object_id,
                    found=False,
                    catalog_kind=None,
                    object_schema=None,
                    object_fingerprint=None,
                    visibility_ceiling=None,
                    categorical_status=None,
                    external_relative_path=None,
                    payload_read=False,
                    sealed_outcome_read=False,
                ),
                reason_codes=("SCIENTIFIC_OBJECT_NOT_FOUND",),
            )
        value = objects[0]
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=ScientificInspectionSummary(
                requested_kind=requested_kind,
                object_id=value.object_id,
                found=True,
                catalog_kind=value.kind,
                object_schema=value.object_schema,
                object_fingerprint=value.object_fingerprint,
                visibility_ceiling=value.visibility_ceiling,
                categorical_status=value.categorical_status,
                external_relative_path=value.external_relative_path,
                payload_read=False,
                sealed_outcome_read=False,
            ),
        )

    def rebuild_catalog(
        self,
        request: CatalogRebuildRequest,
    ) -> ApiResult[CatalogRebuildSummary | WriteEffectSummary]:
        operation = "catalog.rebuild"
        try:
            if self._execution_service is not None:
                projection = self._execution_service.validate_external_projection(
                    request.projection_path
                )
            else:
                projection = request.projection_path.resolve(strict=True)
                external = self._external_root.resolve(strict=True)
                projection.relative_to(external)
        except (CampaignIdentityError, FileNotFoundError, OSError, ValueError) as error:
            invalid = _validation_error(
                operation,
                ValueError(f"projection must be a file below the external root: {error}"),
            )
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        if not projection.is_file():
            invalid = _validation_error(operation, ValueError("projection must be a file"))
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        if not request.confirmed:
            return ApiResult(
                operation=operation,
                status=OperationStatus.BLOCKED,
                payload=WriteEffectSummary(
                    action=operation,
                    confirmed=False,
                    effects=("local:.empirical-lawhood/experiment_catalog.sqlite3",),
                ),
                reason_codes=("WRITE_CONFIRMATION_REQUIRED",),
            )
        try:
            projection_payload = read_bounded_bytes(
                projection,
                maximum_bytes=MAX_CATALOG_PROJECTION_BYTES,
            )
            try:
                decode_unified_catalog_snapshot(projection_payload)
            except (CanonicalizationError, ValueError):
                pass
            else:
                reason_code = "DATASET_PROJECTION_REBUILD_AUTHORITY_REQUIRED"
                return ApiResult(
                    operation=operation,
                    status=OperationStatus.BLOCKED,
                    payload=WriteEffectSummary(
                        action=operation,
                        confirmed=True,
                        effects=("local:.empirical-lawhood/experiment_catalog.sqlite3",),
                    ),
                    errors=(
                        ApiError(
                            category=ErrorCategory.AUTHORITY,
                            reason_code=reason_code,
                            message=(
                                "a unified dataset projection requires dedicated durable "
                                "rebuild authorization; --yes is only write confirmation"
                            ),
                        ),
                    ),
                    reason_codes=(reason_code,),
                )
            result = rebuild_production_catalog(
                self._repo_root,
                projection_payload,
                artifact_plane=(
                    None
                    if self._execution_service is None
                    else self._execution_service.catalog_rebuild_artifact_plane(projection_payload)
                ),
            )
        except (
            BoundedFileIOError,
            CampaignExecutionError,
            CatalogRebuildError,
            OSError,
            ValueError,
        ) as error:
            return self._storage_error(operation, "CATALOG_REBUILD_FAILED", error)
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=CatalogRebuildSummary(
                database_relative_path=result.database_relative_path,
                database_sha256=result.database_sha256,
                database_size_bytes=result.database_size_bytes,
                projection_sha256=result.projection_sha256,
                snapshot_sha256=result.snapshot_sha256,
                integrity_reasons=result.integrity_reasons,
            ),
        )

    def _git_value(self, *arguments: str) -> str | None:
        try:
            completed = run_bounded_command(
                ["git", *arguments],
                cwd=self._repo_root,
                timeout_seconds=5,
                maximum_stdout_bytes=4096,
                maximum_stderr_bytes=16 * 1024,
            )
        except (BoundedProcessError, OSError, subprocess.SubprocessError, ValueError):
            return None
        if completed.returncode != 0:
            return None
        try:
            value = completed.stdout.decode("utf-8").strip()
        except UnicodeDecodeError:
            return None
        return value or None

    def _validate_document_kind(
        self,
        request: DocumentRequest,
        *,
        operation: str,
        allowed: tuple[type[CanonicalRecord], ...],
        validator: Callable[[CanonicalRecord], None] | None = None,
    ) -> ApiResult[DocumentSummary]:
        try:
            value = load_authoring(request.path)
            if not isinstance(value, allowed):
                expected = ", ".join(value.__name__ for value in allowed)
                raise AuthoringCodecError(f"document must contain one of: {expected}")
            if validator is not None:
                validator(value)
        except FileNotFoundError as error:
            return ApiResult(
                operation=operation,
                status=OperationStatus.NOT_FOUND,
                errors=(
                    ApiError(
                        category=ErrorCategory.NOT_FOUND,
                        reason_code="AUTHORING_DOCUMENT_NOT_FOUND",
                        message=str(error),
                    ),
                ),
                reason_codes=("AUTHORING_DOCUMENT_NOT_FOUND",),
            )
        except (AuthoringCodecError, OSError, ValueError) as error:
            invalid = _validation_error(operation, error)
            return ApiResult(
                operation=invalid.operation,
                status=invalid.status,
                errors=invalid.errors,
                reason_codes=invalid.reason_codes,
            )
        return ApiResult(
            operation=operation,
            status=OperationStatus.SUCCEEDED,
            payload=DocumentSummary(
                object_id=_object_id(value),
                object_kind=type(value).__name__,
                schema=value.SCHEMA,
                version=value.VERSION,
                fingerprint=value.fingerprint(),
            ),
        )

    def _validate_campaign_root(self, value: CanonicalRecord) -> None:
        if isinstance(
            value,
            (
                CampaignPackage,
                IssuedCampaignPackage,
                IssuedStudyPackage,
                EnvelopeExperimentPackage,
                ExperimentPackage,
            ),
        ):
            self._replay_campaign_authorization(value)

    def _replay_campaign_authorization(
        self,
        package: CampaignPackage | IssuedCampaignPackageRoot,
    ) -> None:
        if self._approval_service is None:
            raise DurableAuthorizationReplayError(
                "follow-up authorization replay service is not composed"
            )
        try:
            replayed = self._approval_service.replay(
                system=package.system,
                frozen_proposal=ObjectIdentity.from_record(
                    package.frozen_proposal.frozen_proposal_id,
                    package.frozen_proposal,
                ),
                authorization=ObjectIdentity.from_record(
                    package.authorization.authorization_id,
                    package.authorization,
                ),
            )
        except Exception as error:
            raise DurableAuthorizationReplayError(
                "stored follow-up authorization failed complete replay"
            ) from error
        if replayed != package.experiment:
            raise DurableAuthorizationReplayError(
                "stored follow-up authorization returned another experiment"
            )

    def _compile_package(
        self,
        package: CampaignPackage | IssuedCampaignPackageRoot,
    ) -> tuple[ProtocolRunPlan, ProtocolExecutionPlan]:
        self._replay_campaign_authorization(package)
        assert self._approval_service is not None
        if isinstance(
            package,
            (
                IssuedCampaignPackage,
                IssuedStudyPackage,
                EnvelopeExperimentPackage,
                ExperimentPackage,
            ),
        ):
            if self._study_authority_store is None:
                raise ProtocolCompilationError(
                    "trusted programme execution-authority store is not composed"
                )
            authority = self._study_authority_store.load(
                package.execution_authority.authority_id
            )
            if authority != package.execution_authority or ObjectIdentity.from_record(
                authority.authority_id, authority
            ) != ObjectIdentity.from_record(
                package.execution_authority.authority_id,
                package.execution_authority,
            ):
                raise ProtocolCompilationError(
                    "programme execution authority store returned a substitution"
                )
        binds_datasets = any(
            value.namespace == EXPERIMENT_DATASET_BINDINGS_EXTENSION_NAMESPACE
            for value in package.experiment.extensions
        )
        resolver = self._dataset_campaign_binding_resolver
        if resolver is None and binds_datasets:
            raise DatasetBindingCompilationError(
                "DATASET_BINDING_TRUST_CONFIGURATION_REQUIRED",
                "campaign commits dataset bindings but no authenticated resolver is composed",
            )
        dataset_bindings = (
            ()
            if resolver is None
            else resolver.resolve(package.experiment, run_id=package.run_plan_id)
        )
        issued_candidate = (
            issued_standard_candidate(package)
            if isinstance(
                package,
                (
                    IssuedCampaignPackage,
                    IssuedStudyPackage,
                    EnvelopeExperimentPackage,
                    ExperimentPackage,
                ),
            )
            else None
        )
        base_candidate = (
            issued_base_candidate(package)
            if isinstance(
                package,
                (
                    IssuedCampaignPackage,
                    IssuedStudyPackage,
                    EnvelopeExperimentPackage,
                    ExperimentPackage,
                ),
            )
            else None
        )
        run_plan = compile_run_plan(
            run_plan_id=package.run_plan_id,
            campaign=package.campaign,
            system=package.system,
            experiment=package.experiment,
            frozen_proposal=package.frozen_proposal,
            authorization=package.authorization,
            template=package.protocol,
            registry=package.registry,
            implementation_commit=package.implementation_commit,
            approval_service=self._approval_service,
            model_set=package.model_set,
            dataset_bindings=dataset_bindings,
            scientific_graph=(
                base_candidate.scientific_graph if base_candidate is not None else None
            ),
            candidate=(
                ObjectIdentity.from_record(
                    issued_candidate.candidate_id,
                    issued_candidate,
                )
                if issued_candidate is not None
                else None
            ),
            issued_extension_set=(
                package.issued_study.issued_extensions
                if isinstance(
                    package,
                    (EnvelopeExperimentPackage, ExperimentPackage),
                )
                else None
            ),
            execution_envelope_spec=(
                package.execution_envelope_spec
                if isinstance(package, EnvelopeExperimentPackage)
                else None
            ),
            execution_resource_envelope_spec=(
                package.execution_resource_envelope_spec
                if isinstance(package, ExperimentPackage)
                else None
            ),
            predevelopment_jit_signature_census=(
                package.predevelopment_jit_signature_census
                if isinstance(package, ExperimentPackage)
                else None
            ),
            jit_graph_signature_manifest=(
                package.jit_graph_signature_manifest
                if isinstance(package, ExperimentPackage)
                else None
            ),
        )
        execution_plan = lower_run_plan(run_plan, package.registry)
        if isinstance(
            package,
            (
                IssuedCampaignPackage,
                IssuedStudyPackage,
                EnvelopeExperimentPackage,
                ExperimentPackage,
            ),
        ):
            assert issued_candidate is not None
            prove_scientific_graph_parity(
                candidate=issued_candidate,
                issued_candidate=issued_candidate,
                authorized_experiment=package.experiment,
                scientific_approval=ObjectIdentity.from_record(
                    package.scientific_approval.authorization_id,
                    package.scientific_approval,
                ),
                issue_manifest=ObjectIdentity.from_record(
                    package.issued_study.issue_id,
                    package.issued_study,
                ),
                package=ObjectIdentity.from_record(package.package_id, package),
                execution_authority=ObjectIdentity.from_record(
                    package.execution_authority.authority_id,
                    package.execution_authority,
                ),
                run_plan=run_plan,
                execution_plan=execution_plan,
                registry=package.registry,
            )
        return run_plan, execution_plan

    def _compile_path(
        self,
        path: Path,
    ) -> tuple[CampaignPackage | IssuedCampaignPackageRoot, ProtocolRunPlan, ProtocolExecutionPlan]:
        value = load_campaign_package_authoring(path)
        if not isinstance(
            value,
            (
                CampaignPackage,
                IssuedCampaignPackage,
                IssuedStudyPackage,
                EnvelopeExperimentPackage,
                ExperimentPackage,
            ),
        ):
            raise AuthoringCodecError("campaign compilation requires a follow-up CampaignPackage")
        run_plan, execution_plan = self._compile_package(value)
        return value, run_plan, execution_plan

    @staticmethod
    def _dual_loop_path(path: Path) -> DualLoopPackage:
        value = load_authoring(path)
        if not isinstance(value, DualLoopPackage):
            raise AuthoringCodecError("operation requires a DualLoopPackage")
        return value

    @staticmethod
    def _exploration_path(path: Path) -> ExplorationPackage:
        value = load_authoring(path)
        if not isinstance(value, (DualLoopPackage, ExplorationExecutionPackage)):
            raise AuthoringCodecError("operation requires an exploration package")
        return value

    @staticmethod
    def _compile_dual_loop_package(
        package: DualLoopPackage,
    ) -> ProtocolExecutionPlan:
        synthesis = package.registry.resolve(
            package.synthesis_capability_key,
            package.synthesis_capability_version,
        )
        return compile_exploration_execution_plan(
            exploration_plan=package.exploration_plan,
            snapshot=package.snapshot,
            snapshot_verification=package.snapshot_verification,
            registry=package.registry,
            implementation_commit=package.implementation_commit,
            synthesis_capability_key=package.synthesis_capability_key,
            synthesis_capability_version=package.synthesis_capability_version,
            synthesis_budget=synthesis.resource_ceiling,
        )

    @staticmethod
    def _compile_exploration_execution_package(
        package: ExplorationExecutionPackage,
    ) -> ProtocolExecutionPlan:
        skeptic = package.registry.resolve(
            package.skeptic_capability_key,
            package.skeptic_capability_version,
        )
        synthesis = package.registry.resolve(
            package.synthesis_capability_key,
            package.synthesis_capability_version,
        )
        return compile_exploration_wave_execution_plan(
            exploration_plan=package.exploration_plan,
            snapshot=package.snapshot,
            snapshot_verification=package.snapshot_verification,
            wave_input=package.wave_input,
            registry=package.registry,
            implementation_commit=package.implementation_commit,
            skeptic_capability_key=package.skeptic_capability_key,
            skeptic_capability_version=package.skeptic_capability_version,
            skeptic_budget=skeptic.resource_ceiling,
            synthesis_capability_key=package.synthesis_capability_key,
            synthesis_capability_version=package.synthesis_capability_version,
            synthesis_budget=synthesis.resource_ceiling,
            wave_result_schema=ExplorationWaveResult.SCHEMA,
        )

    def _compile_dual_loop_path(
        self,
        path: Path,
    ) -> tuple[DualLoopPackage, ProtocolExecutionPlan]:
        package = self._dual_loop_path(path)
        return package, self._compile_dual_loop_package(package)

    def _compile_exploration_path(
        self,
        path: Path,
    ) -> tuple[ExplorationPackage, ProtocolExecutionPlan]:
        package = self._exploration_path(path)
        if isinstance(package, DualLoopPackage):
            return package, self._compile_dual_loop_package(package)
        return package, self._compile_exploration_execution_package(package)

    def _resolved_exploration_records(
        self,
        package: ExplorationPackage,
        execution_plan: ProtocolExecutionPlan,
    ) -> tuple[tuple[ExploratoryFinding, ...], HypothesisSet]:
        if isinstance(package, DualLoopPackage):
            return package.findings, package.hypotheses
        if self._execution_service is None:
            raise ValueError("exploration execution service is not composed")
        result = self._execution_service.resolve_exploration_wave(
            package,
            execution_plan,
        )
        return result.findings, result.hypothesis_synthesis.hypothesis_set

    def _nomination_batch(
        self,
        package: ExplorationPackage,
        *,
        findings: tuple[ExploratoryFinding, ...] | None = None,
        hypotheses: HypothesisSet | None = None,
    ) -> tuple[ProspectiveNominationDecision, tuple[NominationObligations, ...]]:
        if self._prospective_workflow is None:
            raise ValueError("prospective workflow service is not composed")
        if findings is None or hypotheses is None:
            if not isinstance(package, DualLoopPackage):
                raise ValueError("executable exploration requires persisted wave results")
            findings = package.findings
            hypotheses = package.hypotheses
        return self._prospective_workflow.nominate(
            hypotheses=hypotheses,
            findings=findings,
            system=package.system,
            context=package.design_context,
        )

    def _designed_experiments(
        self,
        package: ExplorationPackage,
        *,
        findings: tuple[ExploratoryFinding, ...] | None = None,
        hypotheses: HypothesisSet | None = None,
    ) -> tuple[ExperimentProposal, ...]:
        if self._prospective_workflow is None:
            raise ValueError("prospective workflow service is not composed")
        decision, obligations = self._nomination_batch(
            package,
            findings=findings,
            hypotheses=hypotheses,
        )
        return self._prospective_workflow.design(
            decision=decision,
            obligations=obligations,
            system=package.system,
        )

    @staticmethod
    def _execution_unavailable(
        operation: str,
    ) -> ApiResult[RunExecutionSummary | WriteEffectSummary]:
        return ApiResult(
            operation=operation,
            status=OperationStatus.BLOCKED,
            errors=(
                ApiError(
                    category=ErrorCategory.STORAGE,
                    reason_code="EXECUTION_SERVICE_UNAVAILABLE",
                    message="campaign execution service is not composed",
                ),
            ),
            reason_codes=("EXECUTION_SERVICE_UNAVAILABLE",),
        )

    @staticmethod
    def _terminal_execution_result(
        operation: str,
        summary: RunExecutionSummary,
        *,
        package: CampaignPackage | IssuedCampaignPackageRoot | None = None,
        execution_plan: ProtocolExecutionPlan | None = None,
        reveal_authority_supplied: bool = False,
    ) -> ApiResult[RunExecutionSummary | WriteEffectSummary | CampaignAuthorityRequired]:
        if (
            summary.operational_status == "BLOCKED"
            and not reveal_authority_supplied
            and isinstance(package, ExperimentPackage)
            and execution_plan is not None
        ):
            barrier_task_ids = tuple(
                sorted(
                    task.task_id
                    for task in execution_plan.tasks
                    if task.task_id in summary.blocked_task_ids
                    and task.barrier is BarrierKind.REVEAL
                )
            )
            if barrier_task_ids:
                prefix = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "execution_plan": ObjectIdentity.from_record(
                                execution_plan.execution_plan_id,
                                execution_plan,
                            ),
                            "completed_task_ids": summary.completed_task_ids,
                            "sealed_predecessor_receipt_ids": summary.receipt_ids,
                        }
                    )
                ).hexdigest()
                authority_required = CampaignAuthorityRequired(
                    barrier_id=f"reveal-barrier.{summary.run_id}",
                    run_id=summary.run_id,
                    issued_study=ObjectIdentity.from_record(
                        package.issued_study.issue_id,
                        package.issued_study,
                    ),
                    execution_plan=ObjectIdentity.from_record(
                        execution_plan.execution_plan_id,
                        execution_plan,
                    ),
                    execution_authority=ObjectIdentity.from_record(
                        package.execution_authority.authority_id,
                        package.execution_authority,
                    ),
                    barrier_task_ids=barrier_task_ids,
                    completed_task_ids=summary.completed_task_ids,
                    sealed_predecessor_receipt_ids=summary.receipt_ids,
                    immutable_resume_prefix_sha256=prefix,
                    required_authority_kind=StudyAuthorityKind.OUTCOME_REVEAL,
                    required_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                    same_plan_resume_required=True,
                    protected_outcomes_read=False,
                )
                return ApiResult(
                    operation=operation,
                    status=OperationStatus.BLOCKED,
                    payload=authority_required,
                    errors=(
                        ApiError(
                            category=ErrorCategory.AUTHORITY,
                            reason_code="OUTCOME_REVEAL_AUTHORITY_REQUIRED",
                            message=(
                                "sealed predecessors completed; bind exact reveal authority "
                                "and resume the same immutable plan"
                            ),
                        ),
                    ),
                    reason_codes=("OUTCOME_REVEAL_AUTHORITY_REQUIRED",),
                )
        mapping = {
            "SUCCEEDED": (OperationStatus.SUCCEEDED, (), None),
            "BLOCKED": (
                OperationStatus.BLOCKED,
                ("EXECUTION_PREREQUISITE_BLOCKED",),
                ErrorCategory.CAPABILITY,
            ),
            "FAILED": (
                OperationStatus.FAILED,
                ("TASK_EXECUTION_FAILED",),
                ErrorCategory.EXECUTION,
            ),
        }
        status, reason_codes, category = mapping.get(
            summary.operational_status,
            (
                OperationStatus.FAILED,
                ("INTERNAL_EXECUTION_ERROR",),
                ErrorCategory.INTERNAL,
            ),
        )
        errors = (
            ()
            if category is None
            else (
                ApiError(
                    category=category,
                    reason_code=reason_codes[0],
                    message=(
                        "execution stopped at a declared prerequisite"
                        if status is OperationStatus.BLOCKED
                        else "execution did not complete successfully"
                    ),
                ),
            )
        )
        return ApiResult(
            operation=operation,
            status=status,
            payload=summary,
            errors=errors,
            reason_codes=reason_codes,
        )

    @staticmethod
    def _execution_error(
        operation: str,
        error: Exception,
    ) -> ApiResult[RunExecutionSummary | WriteEffectSummary]:
        if not isinstance(error, CampaignExecutionError):
            LOGGER.exception("unexpected execution facade error", exc_info=error)
            error = CampaignInternalError()
        category = {
            ExecutionErrorKind.VALIDATION: ErrorCategory.VALIDATION,
            ExecutionErrorKind.IDENTITY: ErrorCategory.IDENTITY,
            ExecutionErrorKind.IMMUTABLE_CONFLICT: ErrorCategory.CONFLICT,
            ExecutionErrorKind.LIVE_CONCURRENCY: ErrorCategory.CONFLICT,
            ExecutionErrorKind.CAPABILITY: ErrorCategory.CAPABILITY,
            ExecutionErrorKind.CUSTODY_STORAGE: ErrorCategory.STORAGE,
            ExecutionErrorKind.TASK_FAILURE: ErrorCategory.EXECUTION,
            ExecutionErrorKind.PREREQUISITE: ErrorCategory.AUTHORITY,
            ExecutionErrorKind.INTERNAL: ErrorCategory.INTERNAL,
        }[error.kind]
        status = {
            ExecutionErrorKind.VALIDATION: OperationStatus.INVALID,
            ExecutionErrorKind.IDENTITY: OperationStatus.INVALID,
            ExecutionErrorKind.IMMUTABLE_CONFLICT: OperationStatus.CONFLICT,
            ExecutionErrorKind.LIVE_CONCURRENCY: OperationStatus.CONFLICT,
            ExecutionErrorKind.CAPABILITY: OperationStatus.BLOCKED,
            ExecutionErrorKind.CUSTODY_STORAGE: OperationStatus.BLOCKED,
            ExecutionErrorKind.TASK_FAILURE: OperationStatus.FAILED,
            ExecutionErrorKind.PREREQUISITE: OperationStatus.BLOCKED,
            ExecutionErrorKind.INTERNAL: OperationStatus.FAILED,
        }[error.kind]
        return ApiResult(
            operation=operation,
            status=status,
            errors=(
                ApiError(
                    category=category,
                    reason_code=error.reason_code,
                    message=str(error),
                ),
            ),
            reason_codes=(error.reason_code,),
        )

    def _database_mutation(self, action: str, audit: object) -> DatabaseMutationSummary:
        from empirical_lawhood.infrastructure.sql.database import CatalogSchemaAudit

        if not isinstance(audit, CatalogSchemaAudit):
            raise TypeError("catalog mutation returned an invalid audit")
        return DatabaseMutationSummary(
            action=action,
            catalog_relative_path=PRODUCTION_RELATIVE_PATH.as_posix(),
            schema_valid=audit.passed,
            schema_version=audit.user_version,
            alembic_revision=audit.alembic_revision,
            page_count=audit.page_count,
            page_size=audit.page_size,
        )

    @staticmethod
    def _storage_error(
        operation: str,
        reason_code: str,
        error: Exception,
    ) -> ApiResult[ResultT]:
        LOGGER.exception("storage facade operation failed", exc_info=error)
        return ApiResult(
            operation=operation,
            status=OperationStatus.FAILED,
            errors=(
                ApiError(
                    category=ErrorCategory.STORAGE,
                    reason_code=reason_code,
                    message=f"{operation} storage boundary failed",
                ),
            ),
            reason_codes=(reason_code,),
        )

    @staticmethod
    def _catalog_audit_limit_error(
        operation: str,
        error: CatalogSchemaAuditLimitExceeded,
    ) -> ApiResult[ResultT]:
        return ApiResult(
            operation=operation,
            status=OperationStatus.BLOCKED,
            errors=(
                ApiError(
                    category=ErrorCategory.CAPABILITY,
                    reason_code=error.reason_code,
                    message="catalog schema audit exhausted its bounded work budget",
                ),
            ),
            reason_codes=(error.reason_code,),
        )

    @staticmethod
    def _catalog_not_found(operation: str) -> ApiResult[ResultT]:
        return ApiResult(
            operation=operation,
            status=OperationStatus.NOT_FOUND,
            errors=(
                ApiError(
                    category=ErrorCategory.NOT_FOUND,
                    reason_code="LOCAL_CATALOG_ABSENT",
                    message="local catalog is absent",
                ),
            ),
            reason_codes=("LOCAL_CATALOG_ABSENT",),
        )


def validate_system_id(system_id: str) -> str:
    """Public validation helper used by bounded lookup requests."""

    return validate_stable_id(system_id, field_name="system_id")

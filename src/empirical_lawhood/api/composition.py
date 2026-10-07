"""Family-neutral application composition root for the current contract."""

from __future__ import annotations

import os
import pwd
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path

from empirical_lawhood.api.authoring_handoff import load_authoring_execution_projection, resolve_authoring_directory
from empirical_lawhood.adapters.composition.generated_extension_bundles import (
    GENERATED_EXTENSION_BUNDLE_AGGREGATE,
)
from empirical_lawhood.adapters.composition.rc_ladder_response.authoring import ResistorCapacitorCandidateContextProvider
from empirical_lawhood.adapters.composition.rc_ladder_response.fresh_authoring import load_fresh_rc_bundle
from empirical_lawhood.adapters.composition.reactor_prefix_response.authoring import ReactorPrefixResponseCandidateContextProvider
from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_authoring import load_fresh_reactor_bundle
from empirical_lawhood.adapters.composition.reactor_prefix_response.fresh_runtime import fresh_reactor_platform_ports
from empirical_lawhood.adapters.control.study_bridge import NestedAdmissionControllerBridge, BoundActionAwareControllerBridge, PreparedPolicyControllerBridge
from empirical_lawhood.adapters.control.composition import ControllerStudyComposition
from empirical_lawhood.adapters.physical.mast_archive.fresh_source import (
    FairMastPublicSourceService,
)
from empirical_lawhood.adapters.prospective.workflow import (
    RegisteredProspectiveWorkflow,
)
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_authoring import SyntheticMaterialResponseMethodCandidateContextProvider
from empirical_lawhood.adapters.simulators.ambient_pressure_superconductor.synthetic_material_method_packet import load_synthetic_material_response_bundle
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.analytic_authoring import UniformElectronGasAnalyticCandidateContextProvider
from empirical_lawhood.adapters.simulators.uniform_electron_gas_response.analytic_fresh_authoring import load_uniform_electron_gas_analytic_authoring_bundle
from empirical_lawhood.infrastructure.artifacts import (
    ArtifactProfileValidatorRegistry,
    ExternalArtifactPlane,
    GuardedExternalRoot,
)
from empirical_lawhood.infrastructure.authority import ExternalApprovalGateAttestationStore, ExternalAuthorizationRecordStore, ExternalFrozenApprovalProposalStore, ExternalFrozenStudyApprovalProposalStore, load_owner_installed_approval_checker_registry
from empirical_lawhood.infrastructure.bounded_process import (
    BoundedProcessError,
    run_bounded_command,
)
from empirical_lawhood.infrastructure.dataset_binding import (
    AuthenticatedDatasetCampaignBindingResolver,
)
from empirical_lawhood.infrastructure.dataset_catalog_access import (
    DatasetCatalogReadSessionProvider,
    load_local_dataset_projection_receipt_reference,
)
from empirical_lawhood.infrastructure.dataset_catalog_authority import (
    ExternalDatasetCatalogAuthorityLoader,
)
from empirical_lawhood.infrastructure.dataset_operations import (
    MAX_DATASET_AUTHORITY_RECORD_BYTES,
)
from empirical_lawhood.infrastructure.dataset_projection import (
    MAX_UNIFIED_CATALOG_PROJECTION_BYTES,
)
from empirical_lawhood.infrastructure.execution import DurableExecutionResourceEnvelopeCoordinator, DurableExecutionResourceEnvelopeStore, LocalProcessExecutor
from empirical_lawhood.infrastructure.execution_resource_envelopes import ExternalExecutionResourceEnvelopeStore
from empirical_lawhood.infrastructure.production_storage import (
    PRODUCTION_ARTIFACT_PROFILE_VALIDATORS,
)
from empirical_lawhood.infrastructure.study_issue import PROGRAMME_ISSUE_GRANTEE_ID, ExternalIssuedExtensionPayloadSource, ExternalIssuedStudyPublisher, ExternalStudyOperationAuthorityStore, GitStudySourceClosureInspector, StudyOperationAuthorityStore
from empirical_lawhood.infrastructure.public_source_acquisition import (
    BoundedPublicHttpsFetcher,
    ExternalPublicSourceCustody,
)
from empirical_lawhood.infrastructure.source_origin import (
    require_executing_target_source,
)
from empirical_lawhood.infrastructure.sql import (
    create_read_only_catalog_engine,
    initialize_production_catalog,
    production_catalog_path,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.planning.approval import ApprovalCheckerRegistry, ApprovalGateAttestation, CompleteApprovalService, DecisionClock, DurableAuthorizationRecord, FrozenApprovalProposal, FrozenIssuedStudyApprovalProposal, SystemDecisionClock
from empirical_lawhood.planning.dataset_authority import (
    DatasetOperationPolicy,
    DatasetOperationRequest,
)
from empirical_lawhood.planning.dataset_manifests import DatasetTransformationManifest
from empirical_lawhood.planning.datasets import (
    DatasetEvidenceClass,
    DatasetMaterializationClass,
)
from empirical_lawhood.planning.linked_campaign import LinkedCampaignProfile
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure as ProgrammeImplementationSourceClosure
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile
from empirical_lawhood.runtime.candidate_compiler import CandidateCompilationContext
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateContextProvider,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.conditional_children import ConditionalChildResolver
from empirical_lawhood.runtime.controller_compiler import AtlasAdmissionDeriverPort, ReceiptAdmissionDeriverPort, FiniteCertificateAdmissionDeriverPort, AtlasReachabilityDeriverPort, ReceiptReachabilityDeriverPort, FiniteCertificateReachabilityDeriverPort
from empirical_lawhood.runtime.controller_evaluation import ControllerUseEvaluator
from empirical_lawhood.runtime.controller_evaluation_nested import (
    ActionAwareNestedControllerUseEvaluator,
    NestedControllerUseEvaluator,
    ProspectiveEvaluationBindingCoordinator,
)
from empirical_lawhood.runtime.controller_runtime import NativeDeliveryPort, ObserverPort, AtlasLiveGateEvaluatorPort, AdmissionLiveGateEvaluatorPort, NumericalViewLiveGateEvaluatorPort
from empirical_lawhood.runtime.dataset_binding import DatasetCampaignBindingResolver
from empirical_lawhood.runtime.dataset_operations import (
    DatasetOperationPreviewService,
    compose_dataset_operation_policy,
    compose_dataset_operation_request,
)
from empirical_lawhood.runtime.dataset_rebuild import (
    DatasetProjectionRebuildInstallerPort,
)
from empirical_lawhood.runtime.dataset_registration import (
    DatasetRegistrationInstallerPort,
)
from empirical_lawhood.runtime.datasets import (
    DatasetCapabilityRegistry,
    DatasetCatalogWorkLimit,
)
from empirical_lawhood.runtime.evidence_profiles import EvidenceProfileRegistryResolver
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingRole, ExecutableCapabilityProviderFactoryRegistry, ExecutableLinkedCampaignCoordinatorFactory, ExecutablePlatformPort, ExecutableProfileCompilerFactory, GeneratedExecutableBindingAggregate, LayeredCampaignRuntimeProviderResolver
from empirical_lawhood.runtime.issued_extension_payloads import IssuedExtensionPayloadResolver
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignProvider, build_linked_campaign_protocol
from empirical_lawhood.runtime.operator_profile import (
    OperatorStorageAccessMode,
    OperatorStorageProfile,
    resolve_external_root_contract,
    resolve_scientific_scratch_contract,
)
from empirical_lawhood.runtime.plans import ProtocolTemplate
from empirical_lawhood.runtime.profile_compilation import LinkedCampaignProfileCompiler, SourceProfileCompilerBinding, SourceProfileCompilerRegistry
from empirical_lawhood.runtime.study_issue import StudySourceClosureInspector
from empirical_lawhood.runtime.providers import CampaignRuntimeProviderRegistry
from empirical_lawhood.runtime.resource_envelope_conformance import PublicRouteComposition, PublicRouteConformanceGraph, standard_resource_envelope_route_composition, standard_resource_envelope_route_conformance_graph
from empirical_lawhood.runtime.public_source_acquisition import (
    PublicSourceAcquisitionService,
)
from empirical_lawhood.runtime.source_pipelines import SourcePipelineCompilation, compile_source_pipeline
from empirical_lawhood.runtime.sources import SourceCapabilityManifest
from empirical_lawhood.runtime.static_codecs import CanonicalRecordCodecRegistry

from .codecs import AuthoringCodecError, loads_authoring
from .execution import CampaignExecutionService, ExperimentContractEvidenceProvider, IssuedExperimentContractEvidenceProvider
from .facade import EmpiricalLawhoodApi
from .ports import DatasetCatalogSessionProvider


def _generated_executable_factory_registry() -> (
    ExecutableCapabilityProviderFactoryRegistry
):
    """Load the code-owned generated executable root, never a config-selected import."""

    generated = import_module(
        "empirical_lawhood.adapters.composition.generated_executable_bindings"
    )
    aggregate = generated.GENERATED_EXECUTABLE_BINDING_AGGREGATE
    if not isinstance(aggregate, GeneratedExecutableBindingAggregate):
        raise TypeError("generated executable aggregate has another record type")
    factories = generated.EXECUTABLE_CAPABILITY_PROVIDER_FACTORY_REGISTRY
    if not isinstance(factories, ExecutableCapabilityProviderFactoryRegistry):
        raise TypeError("generated executable factory root has another registry type")
    if factories.aggregate != aggregate:
        raise ValueError("generated executable factories differ from their aggregate")
    expected_discovery = ObjectIdentity.from_record(
        GENERATED_EXTENSION_BUNDLE_AGGREGATE.aggregate_id,
        GENERATED_EXTENSION_BUNDLE_AGGREGATE,
    )
    if aggregate.discovery_aggregate != expected_discovery:
        raise ValueError("generated executable root binds another discovery aggregate")
    return factories


def _generated_executable_codec_registry() -> CanonicalRecordCodecRegistry:
    """Load the exact generated issued-payload codecs from installed code."""

    generated = import_module(
        "empirical_lawhood.adapters.composition.generated_executable_bindings"
    )
    codecs = generated.EXECUTABLE_CANONICAL_RECORD_CODEC_REGISTRY
    if not isinstance(codecs, CanonicalRecordCodecRegistry):
        raise TypeError("generated executable codec root has another registry type")
    return codecs


def _generated_profile_compilers(
    factories: ExecutableCapabilityProviderFactoryRegistry,
) -> tuple[
    SourceProfileCompilerRegistry | None, LinkedCampaignProfileCompiler | None
]:
    """Construct only input-free generic compiler products declared by bindings."""

    source_bindings: list[SourceProfileCompilerBinding] = []
    linked_product: LinkedCampaignProfileCompiler | None = None
    for binding in factories.aggregate.bindings:
        factory = factories.factory(binding.binding_id)
        if binding.role is ExecutableBindingRole.PROFILE_COMPILER:
            if (
                binding.accepted_profile_types
                or binding.accepted_config_types
                or binding.required_platform_port_keys
                or binding.required_issued_payload_schemas
            ):
                continue
            if not isinstance(factory, ExecutableProfileCompilerFactory):
                raise TypeError("generated profile-compiler factory has another role")
            product = factory.build_compiler(records=(), platform_ports=())
            if isinstance(product, SourceProfileCompilerBinding):
                source_bindings.append(product)
            elif isinstance(product, SourceProfileCompilerRegistry):
                source_bindings.extend(product.bindings)
            else:
                # Issued/config-dependent compilers are installed factories but
                # are not default static source contexts.
                continue
        elif binding.role is ExecutableBindingRole.LINKED_CAMPAIGN_COORDINATOR:
            if (
                binding.accepted_profile_types
                or binding.accepted_config_types
                or binding.required_platform_port_keys
                or binding.required_issued_payload_schemas
            ):
                continue
            if not isinstance(factory, ExecutableLinkedCampaignCoordinatorFactory):
                raise TypeError("generated linked-compiler factory has another role")
            product = factory.build_coordinator(records=(), platform_ports=())
            if not isinstance(product, LinkedCampaignProfileCompiler):
                raise TypeError(
                    "generated linked-compiler factory returned another product"
                )
            if linked_product is not None:
                raise ValueError("generated linked-compiler product is ambiguous")
            linked_product = product
    source = (
        None
        if not source_bindings
        else SourceProfileCompilerRegistry(
            bindings=tuple(sorted(source_bindings, key=lambda value: value.binding_id)),
            executable_factories=factories,
        )
    )
    generated = GENERATED_EXTENSION_BUNDLE_AGGREGATE
    linked = linked_product or LinkedCampaignProfileCompiler(
        evidence_resolver=EvidenceProfileRegistryResolver(
            resolver_id="evidence-resolver.generated-default",
            registries=(generated.evidence_profile_registry,),
        ),
        capability_registry=generated.capability_registry,
        executable_factories=factories,
        conditional_successor_resolver=ObjectIdentity(
            "conditional-follow-up-resolver.generated-default",
            'empirical-lawhood/runtime/conditional-child-resolver',
            "1.0.0",
            generated.fingerprint(),
        ),
        public_route=ObjectIdentity(
            "public-route-composition.generated-default",
            'empirical-lawhood/api/public-composition',
            "1.0.0",
            factories.fingerprint,
        ),
    )
    return source, linked


@dataclass(frozen=True, slots=True)
class SourcePipelinePilotComposition:
    """Read-only public-root assembly of one profile and current dataset requests."""

    profile: SourcePipelineProfile
    transformation_manifest: DatasetTransformationManifest
    operation_policy: DatasetOperationPolicy
    operation_request: DatasetOperationRequest
    compilation: SourcePipelineCompilation


def compose_source_pipeline_pilot(
    *,
    profile: SourcePipelineProfile,
    transformation_manifest: DatasetTransformationManifest,
    source_manifest: SourceCapabilityManifest,
    capability_registry: CapabilityRegistry,
    dataset_capability_registry: DatasetCapabilityRegistry,
    policy_id: str,
    evidence_class: DatasetEvidenceClass,
    materialization_class: DatasetMaterializationClass,
    world_kind: WorldKind,
    implementation_commit: str,
    issued_by: str,
    authorized_approver_id: str,
    requested_by: str,
    policy_valid_from_utc: str,
    policy_valid_until_utc: str,
    request_valid_from_utc: str,
    request_valid_until_utc: str,
    policy_reason_codes: tuple[str, ...],
    request_reason_codes: tuple[str, ...],
) -> SourcePipelinePilotComposition:
    """Use current dataset composition; perform no read, write, or authorization."""

    manifest_identity = ObjectIdentity.from_record(
        transformation_manifest.manifest_id,
        transformation_manifest,
    )
    if profile.dataset_transformation_manifest != manifest_identity:
        raise ValueError("source profile binds another dataset transformation manifest")
    policy = compose_dataset_operation_policy(
        transformation_manifest,
        policy_id=policy_id,
        evidence_class=evidence_class,
        materialization_class=materialization_class,
        world_kind=world_kind,
        implementation_commit=implementation_commit,
        issued_by=issued_by,
        authorized_approver_id=authorized_approver_id,
        valid_from_utc=policy_valid_from_utc,
        valid_until_utc=policy_valid_until_utc,
        reason_codes=policy_reason_codes,
    )
    request = compose_dataset_operation_request(
        policy,
        transformation_manifest,
        request_id=f"request.{profile.profile_id}",
        requested_by=requested_by,
        valid_from_utc=request_valid_from_utc,
        valid_until_utc=request_valid_until_utc,
        reason_codes=request_reason_codes,
    )
    compilation = compile_source_pipeline(
        profile,
        source_manifest=source_manifest,
        capability_registry=capability_registry,
        dataset_registry=dataset_capability_registry,
        dataset_operation_request=request,
    )
    return SourcePipelinePilotComposition(
        profile=profile,
        transformation_manifest=transformation_manifest,
        operation_policy=policy,
        operation_request=request,
        compilation=compilation,
    )


@dataclass(frozen=True, slots=True)
class LinkedCampaignPilotComposition:
    """Pilot dependencies joined at the existing application composition root."""

    profile: LinkedCampaignProfile
    source_pipeline: SourcePipelinePilotComposition
    protocol: ProtocolTemplate
    provider: LinkedCampaignProvider
    conditional_successor_resolver: ConditionalChildResolver
    execution_service: CampaignExecutionService
    dataset_operation_preview_service: DatasetOperationPreviewService
    dataset_projection_rebuild_installer: DatasetProjectionRebuildInstallerPort
    study_authority_store: StudyOperationAuthorityStore


def compose_linked_campaign_pilot(
    *,
    profile: LinkedCampaignProfile,
    capability_registry: CapabilityRegistry,
    source_pipeline: SourcePipelinePilotComposition,
    conditional_successor_resolver: ConditionalChildResolver,
    execution_service: CampaignExecutionService,
    dataset_operation_preview_service: DatasetOperationPreviewService,
    dataset_projection_rebuild_installer: DatasetProjectionRebuildInstallerPort,
    study_authority_store: StudyOperationAuthorityStore,
) -> LinkedCampaignPilotComposition:
    """Bind current authority/storage/recovery services without a pilot runner."""

    expected_source_profile = ObjectIdentity.from_record(
        source_pipeline.profile.profile_id,
        source_pipeline.profile,
    )
    expected_readiness = ObjectIdentity.from_record(
        source_pipeline.compilation.compilation_id,
        source_pipeline.compilation,
    )
    if profile.source_profile != expected_source_profile:
        raise ValueError("linked campaign binds another source-pipeline profile")
    if profile.source_readiness != expected_readiness:
        raise ValueError("linked campaign binds another source readiness compilation")
    return LinkedCampaignPilotComposition(
        profile=profile,
        source_pipeline=source_pipeline,
        protocol=build_linked_campaign_protocol(profile, capability_registry),
        provider=LinkedCampaignProvider(profile),
        conditional_successor_resolver=conditional_successor_resolver,
        execution_service=execution_service,
        dataset_operation_preview_service=dataset_operation_preview_service,
        dataset_projection_rebuild_installer=dataset_projection_rebuild_installer,
        study_authority_store=study_authority_store,
    )


def _decode_frozen_proposal(payload: bytes) -> FrozenApprovalProposal:
    value = loads_authoring(payload.decode("utf-8"), media_type="application/json")
    if not isinstance(value, FrozenApprovalProposal):
        raise AuthoringCodecError("stored authority root is not a frozen proposal")
    return value


def _decode_authorization(payload: bytes) -> DurableAuthorizationRecord:
    value = loads_authoring(payload.decode("utf-8"), media_type="application/json")
    if not isinstance(value, DurableAuthorizationRecord):
        raise AuthoringCodecError(
            "stored authority root is not a durable authorization"
        )
    return value


def _decode_attestation(payload: bytes) -> ApprovalGateAttestation:
    value = loads_authoring(payload.decode("utf-8"), media_type="application/json")
    if not isinstance(value, ApprovalGateAttestation):
        raise AuthoringCodecError("stored authority root is not a gate attestation")
    return value


def _decode_frozen_study_proposal(
    payload: bytes,
) -> FrozenIssuedStudyApprovalProposal:
    value = loads_authoring(payload.decode("utf-8"), media_type="application/json")
    if not isinstance(value, FrozenIssuedStudyApprovalProposal):
        raise AuthoringCodecError(
            "stored authority root is not a frozen programme proposal"
        )
    return value


@dataclass(frozen=True, slots=True)
class OperatorApprovalRuntimeComposition:
    """Explicit owner-installed approval trust and immutable external stores."""

    checker_registry: ApprovalCheckerRegistry
    proposal_store: ExternalFrozenApprovalProposalStore
    study_proposal_store: ExternalFrozenStudyApprovalProposalStore
    attestation_store: ExternalApprovalGateAttestationStore
    authorization_store: ExternalAuthorizationRecordStore
    service: CompleteApprovalService


def _compose_approval_runtime(
    *,
    artifact_plane: ExternalArtifactPlane,
    checker_registry: ApprovalCheckerRegistry,
    clock: DecisionClock,
) -> OperatorApprovalRuntimeComposition:
    proposal_store = ExternalFrozenApprovalProposalStore(
        artifact_plane,
        decoder=_decode_frozen_proposal,
    )
    programme_proposal_store = ExternalFrozenStudyApprovalProposalStore(
        artifact_plane,
        decoder=_decode_frozen_study_proposal,
    )
    attestation_store = ExternalApprovalGateAttestationStore(
        artifact_plane,
        decoder=_decode_attestation,
        checker_registry=checker_registry,
    )
    authorization_store = ExternalAuthorizationRecordStore(
        artifact_plane,
        decoder=_decode_authorization,
    )
    service = CompleteApprovalService(
        clock=clock,
        proposal_store=proposal_store,
        study_proposal_store=programme_proposal_store,
        checker_registry=checker_registry,
        attestation_store=attestation_store,
        store=authorization_store,
        source_acquisition_store=ExternalPublicSourceCustody(artifact_plane),
    )
    return OperatorApprovalRuntimeComposition(
        checker_registry=checker_registry,
        proposal_store=proposal_store,
        study_proposal_store=programme_proposal_store,
        attestation_store=attestation_store,
        authorization_store=authorization_store,
        service=service,
    )


def compose_operator_approval_runtime(
    *,
    artifact_plane: ExternalArtifactPlane,
    checker_trust_path: Path,
    clock: DecisionClock | None = None,
) -> OperatorApprovalRuntimeComposition:
    """Compose approval only from one exact owner-installed public trust root."""

    checker_registry = load_owner_installed_approval_checker_registry(
        checker_trust_path
    )
    return _compose_approval_runtime(
        artifact_plane=artifact_plane,
        checker_registry=checker_registry,
        clock=clock or SystemDecisionClock(),
    )


def discover_repository_root(start: Path | None = None) -> Path:
    candidate = Path.cwd() if start is None else start
    completed = run_bounded_command(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=candidate,
        timeout_seconds=5,
        maximum_stdout_bytes=4096,
        maximum_stderr_bytes=16 * 1024,
    )
    if completed.returncode != 0:
        raise BoundedProcessError("repository-root discovery failed")
    try:
        root = completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise BoundedProcessError(
            "repository-root discovery returned invalid UTF-8"
        ) from error
    return Path(root).resolve(strict=True)


def compose_controller_study_route(
    *,
    registry: CapabilityRegistry,
    admission_deriver: AtlasAdmissionDeriverPort
    | ReceiptAdmissionDeriverPort
    | FiniteCertificateAdmissionDeriverPort,
    reachability_deriver: AtlasReachabilityDeriverPort
    | ReceiptReachabilityDeriverPort
    | FiniteCertificateReachabilityDeriverPort,
    observer: ObserverPort,
    online_gate_evaluator: AtlasLiveGateEvaluatorPort
    | AdmissionLiveGateEvaluatorPort
    | NumericalViewLiveGateEvaluatorPort,
    native_delivery: NativeDeliveryPort,
    outcome_evaluator: ControllerUseEvaluator | NestedControllerUseEvaluator | None,
) -> ControllerStudyComposition:
    """Bind the sole full controller route from closed static capabilities."""

    return ControllerStudyComposition(
        registry=registry,
        admission_deriver=admission_deriver,
        reachability_deriver=reachability_deriver,
        observer=observer,
        online_gate_evaluator=online_gate_evaluator,
        native_delivery=native_delivery,
        outcome_evaluator=outcome_evaluator,
    )


def compose_nested_admission_controller_bridge(
    composition: ControllerStudyComposition,
) -> NestedAdmissionControllerBridge:
    """Wrap the already closed controller route for issued campaign runners."""

    return NestedAdmissionControllerBridge(composition=composition)


def compose_bound_action_aware_controller_bridge(
    *,
    composition: ControllerStudyComposition,
    evaluator: ActionAwareNestedControllerUseEvaluator,
    binding_coordinator: ProspectiveEvaluationBindingCoordinator,
) -> BoundActionAwareControllerBridge:
    "Compose the admission-only controller route and separately bound controller-use evaluator."

    return BoundActionAwareControllerBridge(
        composition=composition,
        evaluator=evaluator,
        binding_coordinator=binding_coordinator,
    )


def compose_prepared_policy_controller_bridge(
    *,
    composition: ControllerStudyComposition,
    binding_coordinator: ProspectiveEvaluationBindingCoordinator,
) -> PreparedPolicyControllerBridge:
    return PreparedPolicyControllerBridge(
        composition=composition, binding_coordinator=binding_coordinator
    )


def compose_execution_resource_envelope_coordinator(
    store: DurableExecutionResourceEnvelopeStore,
) -> DurableExecutionResourceEnvelopeCoordinator:
    return DurableExecutionResourceEnvelopeCoordinator(store)


def compose_resource_envelope_route_conformance_graph(
    *,
    implementation_source: ProgrammeImplementationSourceClosure,
    provider_registry_sha256: str,
) -> PublicRouteConformanceGraph:
    return standard_resource_envelope_route_conformance_graph(
        implementation_source=implementation_source,
        provider_registry_sha256=provider_registry_sha256,
    )


def compose_resource_envelope_route(
    *,
    implementation_source: ProgrammeImplementationSourceClosure,
    provider_registry_sha256: str,
) -> PublicRouteComposition:
    return standard_resource_envelope_route_composition(
        implementation_source=implementation_source,
        provider_registry_sha256=provider_registry_sha256,
    )


def create_api(
    *,
    repo_root: Path | None = None,
    execution_service: CampaignExecutionService | None = None,
    dataset_catalog_session_provider: DatasetCatalogSessionProvider | None = None,
    dataset_operation_preview_service: DatasetOperationPreviewService | None = None,
    public_source_acquisition_service: PublicSourceAcquisitionService | None = None,
    dataset_registration_installer: DatasetRegistrationInstallerPort | None = None,
    dataset_projection_rebuild_installer: DatasetProjectionRebuildInstallerPort
    | None = None,
    dataset_campaign_binding_resolver: DatasetCampaignBindingResolver | None = None,
    candidate_compilation_context: CandidateCompilationContext | None = None,
    candidate_context_provider: CandidateContextProvider | None = None,
    conditional_successor_resolver: ConditionalChildResolver | None = None,
    candidate_capability_catalog: CandidateCapabilityCatalog | None = None,
    study_bundle_registry: CapabilityRegistry | None = None,
    executable_factory_registry: ExecutableCapabilityProviderFactoryRegistry
    | None = None,
    runtime_provider_registry: CampaignRuntimeProviderRegistry | None = None,
    artifact_profile_validators: ArtifactProfileValidatorRegistry | None = None,
    source_profile_compiler: SourceProfileCompilerRegistry | None = None,
    linked_profile_compiler: LinkedCampaignProfileCompiler | None = None,
    operator_storage_profile: OperatorStorageProfile | None = None,
    study_authority_store: StudyOperationAuthorityStore | None = None,
    study_issue_publisher: ExternalIssuedStudyPublisher | None = None,
    study_source_closure_inspector: StudySourceClosureInspector | None = None,
    study_issue_clock: DecisionClock | None = None,
    study_extension_codec_registry: CanonicalRecordCodecRegistry | None = None,
    experiment_contract_evidence_provider: ExperimentContractEvidenceProvider
    | None = None,
    execution_resource_envelope_store: DurableExecutionResourceEnvelopeStore
    | None = None,
    approval_checker_trust_path: Path | None = None,
    maximum_parallel_tasks: int = 1,
    executable_platform_ports: tuple[ExecutablePlatformPort, ...] = (),
) -> EmpiricalLawhoodApi:
    """Create the stable generic facade without probing sources or writing state."""

    root = (
        discover_repository_root()
        if repo_root is None
        else repo_root.resolve(strict=True)
    )
    require_executing_target_source(root)
    installed_factories = _generated_executable_factory_registry()
    installed_executable_codecs = _generated_executable_codec_registry()
    factories = executable_factory_registry or installed_factories
    expected_aggregate = installed_factories.aggregate
    if factories.aggregate != expected_aggregate:
        raise ValueError(
            "executable factory registry differs from installed generated code"
        )
    generated_source, generated_linked = _generated_profile_compilers(factories)
    if source_profile_compiler is not None and (
        source_profile_compiler.executable_factories is not factories
    ):
        raise ValueError(
            "source-profile compiler uses another executable factory registry"
        )
    if linked_profile_compiler is not None and (
        linked_profile_compiler.executable_factories is not factories
    ):
        raise ValueError(
            "linked-profile compiler uses another executable factory registry"
        )
    composed_source_compiler = source_profile_compiler or generated_source
    composed_linked_compiler = linked_profile_compiler or generated_linked
    composed_source_closure_inspector = (
        study_source_closure_inspector or GitStudySourceClosureInspector(root)
    )
    if operator_storage_profile is None and execution_service is None:
        raise ValueError(
            "an explicit OperatorStorageProfile or execution service is required for storage composition"
        )
    external_root_contract = (
        None
        if operator_storage_profile is None
        else resolve_external_root_contract(
            operator_storage_profile,
            repo_root=root,
            home_root=Path(pwd.getpwuid(os.geteuid()).pw_dir),
        )
    )
    scratch_root_contract = (
        None
        if operator_storage_profile is None
        else resolve_scientific_scratch_contract(
            operator_storage_profile,
            repo_root=root,
            home_root=Path(pwd.getpwuid(os.geteuid()).pw_dir),
        )
    )
    if (
        operator_storage_profile is not None
        and maximum_parallel_tasks != operator_storage_profile.maximum_parallel_tasks
    ):
        raise ValueError("operator profile and execution concurrency differ")
    service = execution_service
    internally_composed_publisher: ExternalIssuedStudyPublisher | None = None
    if service is not None and maximum_parallel_tasks != 1:
        raise ValueError(
            "maximum_parallel_tasks cannot replace an explicit execution service"
        )
    if service is not None and approval_checker_trust_path is not None:
        raise ValueError(
            "approval_checker_trust_path cannot replace an explicit execution service"
        )
    if service is not None and experiment_contract_evidence_provider is not None:
        raise ValueError(
            "experiment contract evidence provider cannot replace an explicit execution service"
        )
    if service is not None and execution_resource_envelope_store is not None:
        raise ValueError(
            "execution_resource_envelope_store cannot replace an explicit execution service"
        )
    if service is not None and executable_platform_ports:
        raise ValueError(
            "executable_platform_ports cannot replace an explicit execution service"
        )
    if service is not None and any(
        value is not None
        for value in (runtime_provider_registry, artifact_profile_validators)
    ):
        raise ValueError(
            "provider/validator registries cannot replace an explicit service"
        )
    if service is None:
        assert external_root_contract is not None
        artifact_plane = ExternalArtifactPlane(
            GuardedExternalRoot(external_root_contract),
            artifact_profile_validators or PRODUCTION_ARTIFACT_PROFILE_VALIDATORS,
        )
        service_programme_authority_store = study_authority_store or (
            study_issue_publisher.authority_store
            if study_issue_publisher is not None
            else ExternalStudyOperationAuthorityStore(artifact_plane)
        )
        internally_composed_publisher = (
            study_issue_publisher
            if study_issue_publisher is not None
            else ExternalIssuedStudyPublisher(
                artifact_plane=artifact_plane,
                authority_store=service_programme_authority_store,
                grantee_id=PROGRAMME_ISSUE_GRANTEE_ID,
            )
        )
        installed_runtime_providers = (
            runtime_provider_registry or CampaignRuntimeProviderRegistry(())
        )
        approval_clock = study_issue_clock or SystemDecisionClock()
        approval_runtime = (
            _compose_approval_runtime(
                artifact_plane=artifact_plane,
                checker_registry=ApprovalCheckerRegistry(
                    registry_id="production-approval-checkers",
                    registrations=(),
                ),
                clock=approval_clock,
            )
            if approval_checker_trust_path is None
            else compose_operator_approval_runtime(
                artifact_plane=artifact_plane,
                checker_trust_path=approval_checker_trust_path,
                clock=approval_clock,
            )
        )
        issued_payload_resolver = IssuedExtensionPayloadResolver(
            source=ExternalIssuedExtensionPayloadSource(internally_composed_publisher),
            codecs=installed_executable_codecs,
        )
        service = CampaignExecutionService(
            repo_root=root,
            artifact_plane=artifact_plane,
            engine_factory=lambda: initialize_production_catalog(root),
            read_only_engine_factory=lambda: create_read_only_catalog_engine(
                production_catalog_path(root)
            ),
            catalog_path=production_catalog_path(root),
            providers=installed_runtime_providers,
            provider_resolver=LayeredCampaignRuntimeProviderResolver(
                precomposed=installed_runtime_providers,
                factories=factories,
            ),
            executable_binding_aggregate=expected_aggregate,
            executable_platform_ports=executable_platform_ports,
            issued_extension_payload_resolver=issued_payload_resolver,
            experiment_contract_evidence_provider=(
                experiment_contract_evidence_provider
                or IssuedExperimentContractEvidenceProvider(
                    payload_resolver=issued_payload_resolver,
                    executable_aggregate=expected_aggregate,
                )
            ),
            approval_service=approval_runtime.service,
            study_authority_store=service_programme_authority_store,
            executor=(
                None
                if scratch_root_contract is None
                else LocalProcessExecutor(
                    scratch_root=GuardedExternalRoot(scratch_root_contract),
                )
            ),
            maximum_parallel_tasks=maximum_parallel_tasks,
            study_source_closure_inspector=composed_source_closure_inspector,
            execution_resource_envelope_coordinator=(
                DurableExecutionResourceEnvelopeCoordinator(
                    execution_resource_envelope_store
                    or ExternalExecutionResourceEnvelopeStore(artifact_plane.root)
                )
            ),
        )
    service_external_root = Path(service.artifact_plane.root.contract.canonical_path)
    if (
        study_issue_publisher is not None
        and study_authority_store is not None
        and study_issue_publisher.authority_store is not study_authority_store
    ):
        raise ValueError("programme publisher and execution-authority store differ")
    if (
        study_issue_publisher is not None
        and study_issue_publisher.storage_root_id
        != service.artifact_plane.root.contract.storage_root_id
    ):
        raise ValueError("programme publisher and execution artifact plane differ")
    composed_programme_authority_store = (
        study_authority_store
        or (
            study_issue_publisher.authority_store
            if study_issue_publisher is not None
            else None
        )
        or service.study_authority_store
        or ExternalStudyOperationAuthorityStore(service.artifact_plane)
    )
    if (
        service.study_authority_store is not None
        and service.study_authority_store is not composed_programme_authority_store
    ):
        raise ValueError(
            "execution service and facade programme-authority stores differ"
        )
    composed_programme_issue_publisher = (
        study_issue_publisher
        if study_issue_publisher is not None
        else (
            internally_composed_publisher
            if internally_composed_publisher is not None
            else ExternalIssuedStudyPublisher(
                artifact_plane=service.artifact_plane,
                authority_store=composed_programme_authority_store,
                grantee_id=PROGRAMME_ISSUE_GRANTEE_ID,
            )
        )
    )
    return EmpiricalLawhoodApi(
        repo_root=root,
        execution_service=service,
        prospective_workflow=RegisteredProspectiveWorkflow(),
        approval_service=service.approval_service,
        dataset_catalog_session_provider=dataset_catalog_session_provider,
        public_source_acquisition_service=public_source_acquisition_service
        or (
            None
            if service.approval_service is None
            else PublicSourceAcquisitionService(
                approval=service.approval_service,
                authority_store=composed_programme_authority_store,
                custody=ExternalPublicSourceCustody(service.artifact_plane),
                fetcher=BoundedPublicHttpsFetcher(),
                source_closure_inspector=composed_source_closure_inspector,
                clock=study_issue_clock or SystemDecisionClock(),
            )
        ),
        fair_mast_source_service=FairMastPublicSourceService(
            repo_root=root,
            plane=service.artifact_plane,
            authorities=composed_programme_authority_store,
        ),
        dataset_operation_preview_service=dataset_operation_preview_service,
        dataset_registration_installer=dataset_registration_installer,
        dataset_projection_rebuild_installer=dataset_projection_rebuild_installer,
        dataset_campaign_binding_resolver=dataset_campaign_binding_resolver,
        candidate_compilation_context=candidate_compilation_context,
        candidate_context_provider=candidate_context_provider,
        conditional_successor_resolver=conditional_successor_resolver,
        candidate_capability_catalog=candidate_capability_catalog,
        extension_bundle_aggregate=GENERATED_EXTENSION_BUNDLE_AGGREGATE,
        executable_binding_aggregate=expected_aggregate,
        executable_factory_registry=factories,
        study_bundle_registry=(
            study_bundle_registry
            or GENERATED_EXTENSION_BUNDLE_AGGREGATE.capability_registry
        ),
        study_authority_store=composed_programme_authority_store,
        study_issue_publisher=composed_programme_issue_publisher,
        study_source_closure_inspector=composed_source_closure_inspector,
        study_issue_clock=study_issue_clock,
        study_extension_codec_registry=(
            study_extension_codec_registry or installed_executable_codecs
        ),
        source_profile_compiler=composed_source_compiler,
        linked_profile_compiler=composed_linked_compiler,
        external_root=service_external_root,
    )


def create_inspection_api(
    *,
    repo_root: Path | None = None,
    operator_storage_profile: OperatorStorageProfile | None = None,
) -> EmpiricalLawhoodApi:
    """Compose static discovery and local validation without a storage plane."""

    root = Path.cwd().resolve() if repo_root is None else repo_root.resolve(strict=True)
    factories = _generated_executable_factory_registry()
    storage = (
        None
        if operator_storage_profile is None
        else GuardedExternalRoot(
            resolve_external_root_contract(
                operator_storage_profile,
                repo_root=root,
                home_root=Path.home(),
                for_write=False,
            )
        )
    )
    return EmpiricalLawhoodApi(
        repo_root=root,
        project_configured=repo_root is not None,
        external_root=None if storage is None else Path(storage.contract.canonical_path),
        inspection_storage_root=storage,
        inspection_storage_read_only=(
            operator_storage_profile is not None
            and operator_storage_profile.access_mode is OperatorStorageAccessMode.READ_ONLY
        ),
        fair_mast_source_service=FairMastPublicSourceService(),
        extension_bundle_aggregate=GENERATED_EXTENSION_BUNDLE_AGGREGATE,
        executable_binding_aggregate=factories.aggregate,
        executable_factory_registry=factories,
    )


def create_cli_api(
    *,
    repo_root: Path | None = None,
    operator_storage_profile: OperatorStorageProfile | None = None,
    dataset_projection_trust_path: Path | None = None,
    approval_checker_trust_path: Path | None = None,
    reactor_authoring_dir: Path | None = None,
    circuit_authoring_dir: Path | None = None,
    electron_gas_authoring_dir: Path | None = None,
    synthetic_material_authoring_dir: Path | None = None,
    authoring_dir: Path | None = None,
    maximum_parallel_tasks: int = 1,
) -> EmpiricalLawhoodApi:
    """Create the generic CLI facade with explicit optional trust roots."""

    root = (
        discover_repository_root()
        if repo_root is None
        else repo_root.resolve(strict=True)
    )
    if operator_storage_profile is None:
        raise ValueError(
            "CLI storage operations require an explicit OperatorStorageProfile"
        )
    if (
        sum(
            directory is not None
            for directory in (
                reactor_authoring_dir,
                circuit_authoring_dir,
                electron_gas_authoring_dir,
                synthetic_material_authoring_dir,
                authoring_dir,
            )
        )
        > 1
    ):
        raise ValueError("select exactly one fresh native authoring directory")
    native_kwargs: dict[str, object] = {}
    if reactor_authoring_dir is not None:
        directory, contract = resolve_authoring_directory(
            reactor_authoring_dir, repo_root=root, storage_profile=operator_storage_profile,
            escape_message='fresh reactor authoring lies outside explicit external storage',
        )
        bundle = load_fresh_reactor_bundle(directory)
        projection = load_authoring_execution_projection(directory)
        run_id = bundle.authoring.base.draft.experiment.experiment_id
        if projection.execution_plan_id != f"execution.{run_id}":
            raise ValueError("fresh reactor projection binds another experiment")
        ports = fresh_reactor_platform_ports(
            plane=ExternalArtifactPlane(GuardedExternalRoot(contract)),
            bundle=bundle,
            projected_execution=projection,
            run_id=run_id,
        )
        native_kwargs = {
            "candidate_context_provider": ReactorPrefixResponseCandidateContextProvider(bundle),
            "candidate_capability_catalog": bundle.catalog,
            'study_bundle_registry': bundle.standard_context.base.registry,
            "executable_platform_ports": ports,
        }
    if circuit_authoring_dir is not None:
        directory, contract = resolve_authoring_directory(
            circuit_authoring_dir, repo_root=root, storage_profile=operator_storage_profile,
            escape_message='fresh RC authoring lies outside explicit external storage',
        )
        bundle = load_fresh_rc_bundle(directory)
        projection = load_authoring_execution_projection(directory)
        run_id = bundle.authoring.base.draft.experiment.experiment_id
        if projection.execution_plan_id != f"execution.{run_id}":
            raise ValueError("fresh RC projection binds another experiment")
        native_kwargs = {
            "candidate_context_provider": ResistorCapacitorCandidateContextProvider(bundle),
            "candidate_capability_catalog": bundle.catalog,
            'study_bundle_registry': bundle.standard_context.base.registry,
        }
    if electron_gas_authoring_dir is not None:
        directory, contract = resolve_authoring_directory(
            electron_gas_authoring_dir, repo_root=root, storage_profile=operator_storage_profile,
            escape_message='fresh uniform electron gas authoring lies outside explicit external storage',
        )
        bundle = load_uniform_electron_gas_analytic_authoring_bundle(directory)
        projection = load_authoring_execution_projection(directory)
        run_id = bundle.authoring.base.draft.experiment.experiment_id
        if projection.execution_plan_id != f"execution.{run_id}":
            raise ValueError("fresh uniform electron gas projection binds another experiment")
        native_kwargs = {
            "candidate_context_provider": UniformElectronGasAnalyticCandidateContextProvider(
                bundle
            ),
            "candidate_capability_catalog": bundle.catalog,
            'study_bundle_registry': bundle.standard_context.base.registry,
        }
    if synthetic_material_authoring_dir is not None:
        directory, contract = resolve_authoring_directory(
            synthetic_material_authoring_dir, repo_root=root, storage_profile=operator_storage_profile,
            escape_message='synthetic material authoring lies outside explicit external storage',
        )
        bundle = load_synthetic_material_response_bundle(directory)
        projection = load_authoring_execution_projection(directory)
        run_id = bundle.authoring.base.draft.experiment.experiment_id
        if projection.execution_plan_id != f"execution.{run_id}":
            raise ValueError("synthetic material projection binds another experiment")
        native_kwargs = {
            "candidate_context_provider": SyntheticMaterialResponseMethodCandidateContextProvider(
                bundle
            ),
            "candidate_capability_catalog": bundle.catalog,
            'study_bundle_registry': bundle.standard_context.base.registry,
        }
    if authoring_dir is not None:
        from empirical_lawhood.api.integration_handoffs import load_integration_handoff, integration_api_inputs, integration_artifact_profile_validators
        _, contract = resolve_authoring_directory(
            authoring_dir, repo_root=root, storage_profile=operator_storage_profile,
            escape_message="integration authoring lies outside explicit guarded storage",
        )
        handoff = load_integration_handoff(
            directory=authoring_dir, root=root, storage_profile=operator_storage_profile,
            artifact_writer=ExternalArtifactPlane(GuardedExternalRoot(contract), integration_artifact_profile_validators()),
        )
        native_kwargs = integration_api_inputs(handoff)
    if dataset_projection_trust_path is None:
        reference = None
    else:
        reference = load_local_dataset_projection_receipt_reference(
            dataset_projection_trust_path
        )
    if reference is None:
        return create_api(
            repo_root=root,
            operator_storage_profile=operator_storage_profile,
            approval_checker_trust_path=approval_checker_trust_path,
            maximum_parallel_tasks=maximum_parallel_tasks,
            **native_kwargs,
        )
    external_root = GuardedExternalRoot(
        resolve_external_root_contract(
            operator_storage_profile,
            repo_root=root,
            home_root=Path(pwd.getpwuid(os.geteuid()).pw_dir),
        )
    )
    authority = ExternalDatasetCatalogAuthorityLoader(
        trusted_receipt_references=(reference,),
        roots=(external_root,),
        maximum_receipt_bytes=MAX_DATASET_AUTHORITY_RECORD_BYTES,
        maximum_projection_bytes=MAX_UNIFIED_CATALOG_PROJECTION_BYTES,
        maximum_authorization_bytes=MAX_DATASET_AUTHORITY_RECORD_BYTES,
    ).load(reference.evidence_id)
    provider = DatasetCatalogReadSessionProvider(
        production_catalog_path(root),
        authority.dataset_snapshot,
        authority.anchor,
    )
    binding_resolver = AuthenticatedDatasetCampaignBindingResolver(
        provider,
        work_limit=DatasetCatalogWorkLimit(
            max_records_examined=10_000,
            max_query_steps=100_000,
        ),
    )
    return create_api(
        repo_root=root,
        operator_storage_profile=operator_storage_profile,
        dataset_catalog_session_provider=provider,
        dataset_campaign_binding_resolver=binding_resolver,
        approval_checker_trust_path=approval_checker_trust_path,
        maximum_parallel_tasks=maximum_parallel_tasks,
        **native_kwargs,
    )


__all__ = [
    "LinkedCampaignPilotComposition",
    "OperatorApprovalRuntimeComposition",
    "SourcePipelinePilotComposition",
    'compose_nested_admission_controller_bridge',
    'compose_bound_action_aware_controller_bridge',
    'compose_prepared_policy_controller_bridge',
    'compose_controller_study_route',
    'compose_execution_resource_envelope_coordinator',
    "compose_linked_campaign_pilot",
    "compose_operator_approval_runtime",
    'compose_resource_envelope_route_conformance_graph',
    'compose_resource_envelope_route',
    "compose_source_pipeline_pilot",
    "create_api",
    "create_cli_api",
    "create_inspection_api",
    "discover_repository_root",
]

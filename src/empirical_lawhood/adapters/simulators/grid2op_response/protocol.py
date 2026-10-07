"""Static Grid2Op phase capabilities, scientific DAG and executable runners."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import ClassVar, Final

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateTargetKind
from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateCompleteTargetPanel, IndependentSubstrateStructuredPhaseIssue, IndependentSubstrateTargetPhase
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage

from .analysis import build_grid2op_panel, reduce_grid2op_episode
from .contracts import Grid2OpEpisodeRequest, Grid2OpEpisodeResult, Grid2OpSourceBinding
from .runtime import Grid2OpEnvironmentPort, execute_grid2op_episode


GRID2OP_PHASE_PROTOCOL_VERSION: Final = "1.0.0"
GRID2OP_PHASE_PROVIDER_KEY: Final = "independent-substrate-grounding.grid2op-phase-provider"
GRID2OP_IDENTITY_NAMESPACE: Final = "artifact.independent-substrate-grounding.grid2op.source-readiness"
GRID2OP_SOURCE_ARTIFACT_ID: Final = f"{GRID2OP_IDENTITY_NAMESPACE}.source-binding"
GRID2OP_REDUCTION_STEP_ID: Final = "reduce-grid2op-complete-chronic-panel"


def _generation_key(phase: IndependentSubstrateTargetPhase) -> str:
    return f"grid2op.generate-independent-substrate-grounding-{phase.value.lower()}-episode"


def _reduction_key(phase: IndependentSubstrateTargetPhase) -> str:
    return f"grid2op.reduce-independent-substrate-grounding-{phase.value.lower()}-panel"


def _phase_access(phase: IndependentSubstrateTargetPhase, *, reducer: bool) -> OutcomeAccess:
    if phase is IndependentSubstrateTargetPhase.DEVELOPMENT:
        return OutcomeAccess.DEVELOPMENT_VISIBLE
    return OutcomeAccess.EVALUATOR_REVEAL if reducer else OutcomeAccess.EVALUATION_SEALED


def _phase_visibility(phase: IndependentSubstrateTargetPhase, *, reducer: bool) -> VisibilityCeiling:
    if phase is IndependentSubstrateTargetPhase.DEVELOPMENT:
        return VisibilityCeiling.DEVELOPMENT_ONLY
    return VisibilityCeiling.OUTCOME_VISIBLE if reducer else VisibilityCeiling.PROSPECTIVE


def _request_artifact_id(request_id: str) -> str:
    return f"{GRID2OP_IDENTITY_NAMESPACE}.request.{request_id}"


def grid2op_phase_issue_artifact_id(phase: IndependentSubstrateTargetPhase) -> str:
    """Return the immutable phase-specific issue identity."""

    return f"{GRID2OP_IDENTITY_NAMESPACE}.phase-issue.{phase.value.lower()}"


@dataclass(frozen=True, slots=True)
class Grid2OpPanelReductionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-panel-reduction-config'

    config_id: str
    phase: IndependentSubstrateTargetPhase
    source_binding: ObjectIdentity
    phase_issue: ObjectIdentity
    request_identities: tuple[ObjectIdentity, ...]
    planned_unit_ids: tuple[str, ...]
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.source_binding.object_schema != Grid2OpSourceBinding.SCHEMA:
            raise ValueError("Grid2Op reducer source identity differs")
        if self.phase_issue.object_schema != IndependentSubstrateStructuredPhaseIssue.SCHEMA:
            raise ValueError("Grid2Op reducer phase issue identity differs")
        request_ids = tuple(value.object_id for value in self.request_identities)
        if tuple(sorted(set(request_ids))) != request_ids or any(
            value.object_schema != Grid2OpEpisodeRequest.SCHEMA
            for value in self.request_identities
        ):
            raise ValueError("Grid2Op reducer request roster differs")
        require_sorted_unique_strings(
            self.planned_unit_ids,
            field_name="planned_unit_ids",
            allow_empty=False,
        )
        if len(self.request_identities) != len(self.planned_unit_ids):
            raise ValueError("Grid2Op reducer request/unit roster count differs")
        if not 0 < self.maximum_input_bytes <= 512 * 1024**2:
            raise ValueError("Grid2Op reducer input bound differs")


def grid2op_panel_reduction_config(
    *,
    source: Grid2OpSourceBinding,
    phase_issue: IndependentSubstrateStructuredPhaseIssue,
    phase: IndependentSubstrateTargetPhase,
    requests: tuple[Grid2OpEpisodeRequest, ...],
) -> Grid2OpPanelReductionConfig:
    ordered = tuple(sorted(requests, key=lambda value: value.request_id))
    if not ordered or {value.phase for value in ordered} != {phase}:
        raise ValueError("Grid2Op reducer requests must belong to one nonempty phase")
    source_identity = ObjectIdentity.from_record(source.binding_id, source)
    if {value.source_binding for value in ordered} != {source_identity}:
        raise ValueError("Grid2Op reducer requests name another source")
    if (
        phase_issue.slot is not IndependentSubstrateTargetKind.GRID2OP
        or phase_issue.phase is not phase
        or phase_issue.source_binding != source_identity
        or phase_issue.planned_unit_ids
        != tuple(sorted(value.chronic.object_id for value in ordered))
    ):
        raise ValueError("Grid2Op reducer requests differ from the frozen phase issue")
    unit_ids = tuple(sorted(value.unit_id for value in ordered))
    if len(set(unit_ids)) != len(unit_ids):
        raise ValueError("Grid2Op reducer unit identities must be unique")
    return Grid2OpPanelReductionConfig(
        config_id=f"config.independent-substrate-grounding.grid2op.{phase.value.lower()}-panel-reduction",
        phase=phase,
        source_binding=source_identity,
        phase_issue=ObjectIdentity.from_record(phase_issue.issue_id, phase_issue),
        request_identities=tuple(
            ObjectIdentity.from_record(value.request_id, value) for value in ordered
        ),
        planned_unit_ids=unit_ids,
        maximum_input_bytes=512 * 1024**2,
    )


def decode_grid2op_panel_reduction_config(
    payload: bytes,
) -> Grid2OpPanelReductionConfig:
    return decode_canonical_bytes(
        payload,
        Grid2OpPanelReductionConfig,
        maximum_bytes=2 * 1024**2,
    )


def _generation_budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=2,
        memory_bytes=8 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=12 * 60 * 60,
        source_scan_bytes=2 * 1024**3,
        output_bytes=256 * 1024**2,
    )


def _reduction_budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=2,
        memory_bytes=8 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=30 * 60,
        source_scan_bytes=512 * 1024**2,
        output_bytes=256 * 1024**2,
    )


def grid2op_grid2op_response_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    capabilities: list[CapabilityManifest] = []
    for phase in IndependentSubstrateTargetPhase:
        generation_permissions = tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                    *(
                        (CapabilityPermission.READ_DEVELOPMENT,)
                        if phase is IndependentSubstrateTargetPhase.DEVELOPMENT
                        else ()
                    ),
                ),
                key=lambda value: value.value,
            )
        )
        reducer_permissions = tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                    *(
                        (CapabilityPermission.READ_DEVELOPMENT,)
                        if phase is IndependentSubstrateTargetPhase.DEVELOPMENT
                        else (
                            CapabilityPermission.READ_SEALED_OUTCOMES,
                            CapabilityPermission.REVEAL_OUTCOMES,
                        )
                    ),
                ),
                key=lambda value: value.value,
            )
        )
        capabilities.extend(
            (
                CapabilityManifest(
                    capability_key=_generation_key(phase),
                    capability_version=GRID2OP_PHASE_PROTOCOL_VERSION,
                    kind=CapabilityKind.SIMULATOR,
                    config_schema=Grid2OpEpisodeRequest.SCHEMA,
                    config_schema_sha256=sha256(
                        Grid2OpEpisodeRequest.SCHEMA.encode("ascii")
                    ).hexdigest(),
                    input_schema_ids=tuple(
                        sorted(
                            (
                                Grid2OpEpisodeRequest.SCHEMA,
                                IndependentSubstrateStructuredPhaseIssue.SCHEMA,
                                Grid2OpSourceBinding.SCHEMA,
                            )
                        )
                    ),
                    output_schema_ids=(Grid2OpEpisodeResult.SCHEMA,),
                    permissions=generation_permissions,
                    maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
                    maximum_outcome_access=_phase_access(phase, reducer=False),
                    resource_ceiling=_generation_budget(),
                    deterministic=True,
                    seed_required=False,
                    language_id="python",
                    runtime_id="grid2op-held-runtime-injected",
                    requires_clean_commit=True,
                    requires_active_mount=True,
                    requires_network=False,
                    conformance_check_ids=tuple(
                        sorted(
                            (
                                "action-branches-share-exact-reset",
                                "chronic-is-complete-unit",
                                "reward-excluded-from-science",
                                "requested-accepted-applied-realized-distinct",
                            )
                        )
                    ),
                    implementation_sha256=implementation_sha256,
                ),
                CapabilityManifest(
                    capability_key=_reduction_key(phase),
                    capability_version=GRID2OP_PHASE_PROTOCOL_VERSION,
                    kind=(
                        CapabilityKind.TRANSFORM
                        if phase is IndependentSubstrateTargetPhase.DEVELOPMENT
                        else CapabilityKind.EVALUATOR
                    ),
                    config_schema=Grid2OpPanelReductionConfig.SCHEMA,
                    config_schema_sha256=sha256(
                        Grid2OpPanelReductionConfig.SCHEMA.encode("ascii")
                    ).hexdigest(),
                    input_schema_ids=tuple(
                        sorted(
                            (
                                Grid2OpEpisodeRequest.SCHEMA,
                                Grid2OpEpisodeResult.SCHEMA,
                                Grid2OpPanelReductionConfig.SCHEMA,
                                IndependentSubstrateStructuredPhaseIssue.SCHEMA,
                                Grid2OpSourceBinding.SCHEMA,
                            )
                        )
                    ),
                    output_schema_ids=(IndependentSubstrateCompleteTargetPanel.SCHEMA,),
                    permissions=reducer_permissions,
                    maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
                    maximum_outcome_access=_phase_access(phase, reducer=True),
                    resource_ceiling=_reduction_budget(),
                    deterministic=True,
                    seed_required=False,
                    language_id="python",
                    runtime_id="grid2op-complete-chronic-reducer",
                    requires_clean_commit=True,
                    requires_active_mount=True,
                    requires_network=False,
                    conformance_check_ids=tuple(
                        sorted(
                            (
                                "adverse-issued-chronics-retained",
                                "complete-chronic-resampling-unit",
                                "nested-actions-and-horizons-not-units",
                                "sealed-evaluation-evaluator-only",
                            )
                        )
                    ),
                    implementation_sha256=implementation_sha256,
                ),
            )
        )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-grid2op-phase-registry",
        capabilities=tuple(sorted(capabilities, key=lambda value: value.registry_id)),
    )


def grid2op_grid2op_response_candidate_registrations(
    *,
    implementation_sha256: str,
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=GRID2OP_PHASE_PROVIDER_KEY,
            provider_version=GRID2OP_PHASE_PROTOCOL_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=2 * 1024**2,
        )
        for manifest in grid2op_grid2op_response_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


def _validate_config_ref(
    *,
    record: CanonicalRecord,
    record_id: str,
    config_ref: CapabilityConfigRef,
    manifest: CapabilityManifest,
) -> None:
    if (
        config_ref.config_id != record_id
        or config_ref.config_schema != record.SCHEMA
        or config_ref.config_schema_sha256 != manifest.config_schema_sha256
        or config_ref.content_sha256 != record.fingerprint()
    ):
        raise ValueError("Grid2Op capability config reference differs")


def build_grid2op_phase_protocol(
    *,
    registry: CapabilityRegistry,
    requests: tuple[Grid2OpEpisodeRequest, ...],
    phase_issue: IndependentSubstrateStructuredPhaseIssue,
    request_config_refs: tuple[CapabilityConfigRef, ...],
    reduction_config: Grid2OpPanelReductionConfig,
    reduction_config_ref: CapabilityConfigRef,
) -> ProtocolTemplate:
    ordered = tuple(sorted(requests, key=lambda value: value.request_id))
    refs = {value.config_id: value for value in request_config_refs}
    if (
        not ordered
        or len(refs) != len(ordered)
        or {value.phase for value in ordered} != {reduction_config.phase}
        or reduction_config.phase_issue
        != ObjectIdentity.from_record(phase_issue.issue_id, phase_issue)
        or tuple(ObjectIdentity.from_record(value.request_id, value) for value in ordered)
        != reduction_config.request_identities
    ):
        raise ValueError("Grid2Op phase protocol request/config roster differs")
    generation_steps: list[ProtocolStepTemplate] = []
    for request in ordered:
        manifest = registry.resolve(
            _generation_key(request.phase),
            GRID2OP_PHASE_PROTOCOL_VERSION,
        )
        ref = refs.get(request.request_id)
        if ref is None:
            raise ValueError("Grid2Op phase protocol lacks a request config reference")
        _validate_config_ref(
            record=request,
            record_id=request.request_id,
            config_ref=ref,
            manifest=manifest,
        )
        generation_steps.append(
            ProtocolStepTemplate(
                step_id=f"generate.{request.request_id}",
                stage=ScientificStage.ACQUIRE,
                capability_key=manifest.capability_key,
                capability_version=manifest.capability_version,
                config=ref,
                dependency_step_ids=(),
                outputs=(
                    OutputTemplate(
                        output_id="episode-result",
                        payload_schema=Grid2OpEpisodeResult.SCHEMA,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/vnd.empirical-lawhood.canonical+json",
                        filename_suffix=".json",
                    ),
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=_phase_access(request.phase, reducer=False),
                visibility_ceiling=_phase_visibility(request.phase, reducer=False),
                resource_budget=manifest.resource_ceiling,
                resource_lock_ids=(),
                barrier=(
                    BarrierKind.NONE
                    if request.phase is IndependentSubstrateTargetPhase.DEVELOPMENT
                    else BarrierKind.FREEZE
                ),
                maximum_attempts=2,
                obligation_ids=(
                    f"grid2op-exact-chronic.{request.unit_id}",
                    f"grid2op-shared-reset.{request.unit_id}",
                ),
            )
        )
    reducer_manifest = registry.resolve(
        _reduction_key(reduction_config.phase),
        GRID2OP_PHASE_PROTOCOL_VERSION,
    )
    _validate_config_ref(
        record=reduction_config,
        record_id=reduction_config.config_id,
        config_ref=reduction_config_ref,
        manifest=reducer_manifest,
    )
    reducer = ProtocolStepTemplate(
        step_id=GRID2OP_REDUCTION_STEP_ID,
        stage=(
            ScientificStage.DEVELOP
            if reduction_config.phase is IndependentSubstrateTargetPhase.DEVELOPMENT
            else ScientificStage.REVEAL
        ),
        capability_key=reducer_manifest.capability_key,
        capability_version=reducer_manifest.capability_version,
        config=reduction_config_ref,
        dependency_step_ids=tuple(sorted(value.step_id for value in generation_steps)),
        outputs=(
            OutputTemplate(
                output_id="complete-chronic-panel",
                payload_schema=IndependentSubstrateCompleteTargetPanel.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                filename_suffix=".json",
            ),
        ),
        required_permissions=reducer_manifest.permissions,
        requested_outcome_access=_phase_access(reduction_config.phase, reducer=True),
        visibility_ceiling=_phase_visibility(reduction_config.phase, reducer=True),
        resource_budget=reducer_manifest.resource_ceiling,
        resource_lock_ids=("grid2op-phase-reducer",),
        barrier=(
            BarrierKind.NONE
            if reduction_config.phase is IndependentSubstrateTargetPhase.DEVELOPMENT
            else BarrierKind.REVEAL
        ),
        maximum_attempts=2,
        obligation_ids=(
            "grid2op-all-issued-chronics-retained",
            "grid2op-native-hold-and-action-clocks",
            "grid2op-no-nested-replication",
            "grid2op-postissue-adverse-evidence",
        ),
    )
    return ProtocolTemplate(
        template_id=f"independent-substrate-grounding-grid2op-{reduction_config.phase.value.lower()}-phase",
        template_version=GRID2OP_PHASE_PROTOCOL_VERSION,
        steps=tuple(sorted((*generation_steps, reducer), key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _external_input(
    *,
    input_id: str,
    role: ScientificInputRole,
    artifact_id: str,
    record: CanonicalRecord,
) -> CandidateGraphExternalInput:
    return CandidateGraphExternalInput(
        input_id=input_id,
        scientific_role=role,
        logical_artifact_id=artifact_id,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=record.fingerprint(),
        payload_schema=record.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=2 * 1024**2,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def grid2op_phase_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    source: Grid2OpSourceBinding,
    phase_issue: IndependentSubstrateStructuredPhaseIssue,
    requests: tuple[Grid2OpEpisodeRequest, ...],
) -> CandidateScientificGraph:
    reducer = next(value for value in protocol.steps if value.step_id == GRID2OP_REDUCTION_STEP_ID)
    generation_steps = tuple(
        value for value in protocol.steps if value.step_id != GRID2OP_REDUCTION_STEP_ID
    )
    requests_by_id = {value.request_id: value for value in requests}
    if (
        {value.config.config_id for value in generation_steps} != set(requests_by_id)
        or phase_issue.slot is not IndependentSubstrateTargetKind.GRID2OP
        or phase_issue.source_binding != ObjectIdentity.from_record(source.binding_id, source)
        or phase_issue.planned_unit_ids
        != tuple(sorted(value.chronic.object_id for value in requests))
    ):
        raise ValueError("Grid2Op graph request/protocol roster differs")
    source_input = _external_input(
        input_id="input.independent-substrate-grounding.grid2op.source-binding",
        role=ScientificInputRole.PREPARED_MEDIUM,
        artifact_id=GRID2OP_SOURCE_ARTIFACT_ID,
        record=source,
    )
    issue_input = _external_input(
        input_id=f"input.independent-substrate-grounding.grid2op.phase-issue.{phase_issue.phase.value.lower()}",
        role=ScientificInputRole.QUALIFICATION,
        artifact_id=grid2op_phase_issue_artifact_id(phase_issue.phase),
        record=phase_issue,
    )
    request_inputs = tuple(
        _external_input(
            input_id=f"input.independent-substrate-grounding.grid2op.request.{value.request_id}",
            role=ScientificInputRole.ACTION,
            artifact_id=_request_artifact_id(value.request_id),
            record=value,
        )
        for value in sorted(requests, key=lambda item: item.request_id)
    )
    nodes = tuple(
        CandidateGraphNode(
            node_id=step.step_id,
            stage=step.stage,
            capability_key=step.capability_key,
            capability_version=step.capability_version,
            implementation_sha256=registry.resolve(
                step.capability_key,
                step.capability_version,
            ).implementation_sha256,
            protocol_step_sha256=step.fingerprint(),
            obligation_ids=step.obligation_ids,
            outcome_access=step.requested_outcome_access,
            visibility_ceiling=step.visibility_ceiling,
            resource_budget=step.resource_budget,
        )
        for step in protocol.steps
    )
    edges: list[CandidateGraphEdge] = []
    for step in generation_steps:
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.grid2op-source.{step.step_id}",
                producer_node_id=None,
                producer_output_id=None,
                external_input_id=source_input.input_id,
                consumer_node_id=step.step_id,
                consumer_input_id=source_input.input_id,
                scientific_role=source_input.scientific_role,
                logical_artifact_id=source_input.logical_artifact_id,
                payload_schema=source_input.payload_schema,
                media_type=source_input.media_type,
                maximum_size_bytes=source_input.maximum_size_bytes,
                outcome_access=source_input.outcome_access,
                visibility_ceiling=source_input.visibility_ceiling,
                barrier=step.barrier,
            )
        )
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.grid2op-phase-issue.{step.step_id}",
                producer_node_id=None,
                producer_output_id=None,
                external_input_id=issue_input.input_id,
                consumer_node_id=step.step_id,
                consumer_input_id=issue_input.input_id,
                scientific_role=issue_input.scientific_role,
                logical_artifact_id=issue_input.logical_artifact_id,
                payload_schema=issue_input.payload_schema,
                media_type=issue_input.media_type,
                maximum_size_bytes=issue_input.maximum_size_bytes,
                outcome_access=issue_input.outcome_access,
                visibility_ceiling=issue_input.visibility_ceiling,
                barrier=step.barrier,
            )
        )
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{step.step_id}.result.grid2op-reducer",
                producer_node_id=step.step_id,
                producer_output_id="episode-result",
                external_input_id=None,
                consumer_node_id=reducer.step_id,
                consumer_input_id=f"result.{step.config.config_id}",
                scientific_role=ScientificInputRole.OUTCOME,
                logical_artifact_id=f"artifact.{step.step_id}.episode-result",
                payload_schema=Grid2OpEpisodeResult.SCHEMA,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                maximum_size_bytes=256 * 1024**2,
                outcome_access=step.requested_outcome_access,
                visibility_ceiling=step.visibility_ceiling,
                barrier=reducer.barrier,
            )
        )
    edges.append(
        CandidateGraphEdge(
            edge_id="edge.grid2op-source.grid2op-reducer",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=source_input.input_id,
            consumer_node_id=reducer.step_id,
            consumer_input_id="frozen-source-binding",
            scientific_role=source_input.scientific_role,
            logical_artifact_id=source_input.logical_artifact_id,
            payload_schema=source_input.payload_schema,
            media_type=source_input.media_type,
            maximum_size_bytes=source_input.maximum_size_bytes,
            outcome_access=source_input.outcome_access,
            visibility_ceiling=source_input.visibility_ceiling,
            barrier=reducer.barrier,
        )
    )
    edges.append(
        CandidateGraphEdge(
            edge_id="edge.grid2op-phase-issue.grid2op-reducer",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=issue_input.input_id,
            consumer_node_id=reducer.step_id,
            consumer_input_id="frozen-phase-issue",
            scientific_role=issue_input.scientific_role,
            logical_artifact_id=issue_input.logical_artifact_id,
            payload_schema=issue_input.payload_schema,
            media_type=issue_input.media_type,
            maximum_size_bytes=issue_input.maximum_size_bytes,
            outcome_access=issue_input.outcome_access,
            visibility_ceiling=issue_input.visibility_ceiling,
            barrier=reducer.barrier,
        )
    )
    for request_input in request_inputs:
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{request_input.input_id}.grid2op-reducer",
                producer_node_id=None,
                producer_output_id=None,
                external_input_id=request_input.input_id,
                consumer_node_id=reducer.step_id,
                consumer_input_id=request_input.input_id,
                scientific_role=request_input.scientific_role,
                logical_artifact_id=request_input.logical_artifact_id,
                payload_schema=request_input.payload_schema,
                media_type=request_input.media_type,
                maximum_size_bytes=request_input.maximum_size_bytes,
                outcome_access=request_input.outcome_access,
                visibility_ceiling=request_input.visibility_ceiling,
                barrier=reducer.barrier,
            )
        )
    return CandidateScientificGraph(
        graph_id=f"graph.{protocol.template_id}",
        external_inputs=tuple(
            sorted(
                (source_input, issue_input, *request_inputs),
                key=lambda value: value.input_id,
            )
        ),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def grid2op_phase_study_template(
    *,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> StudyTemplate:
    bindings: list[ObligationCoverageBinding] = []
    steps = {value.step_id: value for value in protocol.steps}
    for node in graph.nodes:
        step = steps[node.node_id]
        output_id = step.outputs[0].output_id
        contributor_edges = tuple(
            sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id)
        )
        for obligation_id in node.obligation_ids:
            bindings.append(
                ObligationCoverageBinding(
                    obligation_id=obligation_id,
                    proof_owner_node_id=node.node_id,
                    required_output_id=output_id,
                    contributor_edge_ids=contributor_edges,
                )
            )
    return StudyTemplate(
        template_key=protocol.template_id,
        template_version=protocol.template_version,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id=f"coverage.{protocol.template_id}",
            bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
        ),
    )


class _Grid2OpGenerationRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        port: Grid2OpEnvironmentPort,
    ) -> None:
        self.manifest = manifest
        self._port = port

    def execute(self, context: TaskContext) -> RunnerResult:
        sources = tuple(
            decode_canonical_bytes(
                port.read(), Grid2OpSourceBinding, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == Grid2OpSourceBinding.SCHEMA
        )
        requests = tuple(
            decode_canonical_bytes(
                port.read(), Grid2OpEpisodeRequest, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == Grid2OpEpisodeRequest.SCHEMA
        )
        issues = tuple(
            decode_canonical_bytes(
                port.read(), IndependentSubstrateStructuredPhaseIssue, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == IndependentSubstrateStructuredPhaseIssue.SCHEMA
        )
        if len(sources) != 1 or len(requests) != 1 or len(issues) != 1:
            raise ValueError("Grid2Op generation requires one source/request/phase issue")
        request = requests[0]
        issue = issues[0]
        if (
            self.manifest.capability_key != _generation_key(request.phase)
            or issue.phase is not request.phase
            or issue.source_binding != ObjectIdentity.from_record(sources[0].binding_id, sources[0])
            or request.chronic.object_id not in issue.planned_unit_ids
        ):
            raise ValueError("Grid2Op generation phase selects another capability")
        result = execute_grid2op_episode(
            port=self._port,
            source=sources[0],
            request=request,
        )
        if {value.payload_schema for value in context.output_ports} != {
            Grid2OpEpisodeResult.SCHEMA
        }:
            raise ValueError("Grid2Op generation output schema differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=result.canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("grid2op-chronic-counted-once", True, ()),
                ReceiptCheck("grid2op-reward-excluded", True, ()),
                ReceiptCheck("grid2op-shared-reset-verified", True, ()),
            ),
        )


class _Grid2OpReductionRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        configs = tuple(
            decode_canonical_bytes(
                port.read(),
                Grid2OpPanelReductionConfig,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == Grid2OpPanelReductionConfig.SCHEMA
        )
        sources = tuple(
            decode_canonical_bytes(
                port.read(), Grid2OpSourceBinding, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == Grid2OpSourceBinding.SCHEMA
        )
        issues = tuple(
            decode_canonical_bytes(
                port.read(), IndependentSubstrateStructuredPhaseIssue, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == IndependentSubstrateStructuredPhaseIssue.SCHEMA
        )
        requests = tuple(
            decode_canonical_bytes(
                port.read(), Grid2OpEpisodeRequest, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == Grid2OpEpisodeRequest.SCHEMA
        )
        results = tuple(
            decode_canonical_bytes(
                port.read(), Grid2OpEpisodeResult, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == Grid2OpEpisodeResult.SCHEMA
        )
        if len(configs) != 1 or len(sources) != 1 or len(issues) != 1:
            raise ValueError("Grid2Op reducer requires one config/source/phase issue")
        config = configs[0]
        source = sources[0]
        if self.manifest.capability_key != _reduction_key(
            config.phase
        ) or config.phase_issue != ObjectIdentity.from_record(issues[0].issue_id, issues[0]):
            raise ValueError("Grid2Op reducer phase selects another capability")
        request_by_identity = {
            ObjectIdentity.from_record(value.request_id, value): value for value in requests
        }
        result_by_request = {value.request: value for value in results}
        if set(request_by_identity) != set(config.request_identities) or set(
            result_by_request
        ) != set(config.request_identities):
            raise ValueError("Grid2Op reducer request/result denominator differs")
        units = tuple(
            sorted(
                (
                    reduce_grid2op_episode(
                        source=source,
                        request=request_by_identity[identity],
                        result=result_by_request[identity],
                    )
                    for identity in config.request_identities
                ),
                key=lambda value: value.unit_id,
            )
        )
        panel = build_grid2op_panel(
            source=source,
            phase=config.phase,
            planned_unit_ids=config.planned_unit_ids,
            units=units,
        )
        if {value.payload_schema for value in context.output_ports} != {
            IndependentSubstrateCompleteTargetPanel.SCHEMA
        }:
            raise ValueError("Grid2Op reduction output schema differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=panel.canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("grid2op-issued-denominator-preserved", True, ()),
                ReceiptCheck("grid2op-nested-views-not-units", True, ()),
                ReceiptCheck("grid2op-postissue-adverse-units-retained", True, ()),
            ),
        )


def grid2op_grid2op_response_runners(
    *,
    registry: CapabilityRegistry,
    environment_port: Grid2OpEnvironmentPort,
) -> tuple[TaskRunner, ...]:
    runners: list[TaskRunner] = []
    for phase in IndependentSubstrateTargetPhase:
        runners.extend(
            (
                _Grid2OpGenerationRunner(
                    registry.resolve(_generation_key(phase), GRID2OP_PHASE_PROTOCOL_VERSION),
                    environment_port,
                ),
                _Grid2OpReductionRunner(
                    registry.resolve(_reduction_key(phase), GRID2OP_PHASE_PROTOCOL_VERSION)
                ),
            )
        )
    return tuple(runners)


class Grid2OpPhaseRuntimeProvider(CampaignRuntimeProvider):
    """Path-free runtime composition for one exact source-qualified phase."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        source: Grid2OpSourceBinding,
        phase_issue: IndependentSubstrateStructuredPhaseIssue,
        requests: tuple[Grid2OpEpisodeRequest, ...],
        reduction_config: Grid2OpPanelReductionConfig,
        environment_port: Grid2OpEnvironmentPort,
    ) -> None:
        implementation_hashes = {value.implementation_sha256 for value in registry.capabilities}
        if len(implementation_hashes) != 1 or registry != grid2op_grid2op_response_registry(
            implementation_sha256=next(iter(implementation_hashes))
        ):
            raise ValueError("Grid2Op runtime provider registry differs")
        ordered = tuple(sorted(requests, key=lambda value: value.request_id))
        if (
            not ordered
            or tuple(ObjectIdentity.from_record(value.request_id, value) for value in ordered)
            != reduction_config.request_identities
            or reduction_config.source_binding
            != ObjectIdentity.from_record(source.binding_id, source)
            or reduction_config.phase_issue
            != ObjectIdentity.from_record(phase_issue.issue_id, phase_issue)
            or {value.phase for value in ordered} != {reduction_config.phase}
        ):
            raise ValueError("Grid2Op runtime provider phase scope differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.source = source
        self.phase_issue = phase_issue
        self.requests = ordered
        self.reduction_config = reduction_config
        self._runners = grid2op_grid2op_response_runners(
            registry=registry,
            environment_port=environment_port,
        )
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("Grid2Op runtime provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("Grid2Op runtime provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            GRID2OP_SOURCE_ARTIFACT_ID: self.source,
            grid2op_phase_issue_artifact_id(self.phase_issue.phase): self.phase_issue,
            **{_request_artifact_id(value.request_id): value for value in self.requests},
        }
        by_hash: dict[str, CanonicalRecord] = {
            value.fingerprint(): value for value in self.requests
        }
        by_hash[self.reduction_config.fingerprint()] = self.reduction_config
        for task in plan.tasks:
            try:
                records[task.capability.config.artifact_id] = by_hash[
                    task.capability.config.content_sha256
                ]
            except KeyError as error:
                raise ValueError("Grid2Op plan references an unknown config") from error
        if set(records) != set(specs):
            raise ValueError("Grid2Op runtime external input roster differs")
        values = []
        for artifact_id, spec in sorted(specs.items()):
            record = records[artifact_id]
            record_id = next(
                getattr(record, attribute)
                for attribute in (
                    "request_id",
                    "config_id",
                    "binding_id",
                    "issue_id",
                )
                if hasattr(record, attribute)
            )
            parent = ArtifactLineageParent(
                identity=ObjectIdentity.from_record(record_id, record),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    payload=record.canonical_bytes(),
                    visibility_ceiling=(
                        spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                    ),
                    outcome_access=(spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND),
                    parent_visibility_ceilings=(parent.visibility_ceiling,),
                    lineage_parents=(parent,),
                    logical_content_sha256=spec.expected_content_sha256,
                )
            )
        return tuple(values)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("Grid2Op semantic registry differs")
        planned = None
        if execution_plan is not None:
            planned = {
                (task.capability.capability_key, output.payload_schema, output.profile)
                for task in execution_plan.tasks
                for output in task.outputs
            }
        record_types = {
            Grid2OpEpisodeResult.SCHEMA: Grid2OpEpisodeResult,
            IndependentSubstrateCompleteTargetPanel.SCHEMA: IndependentSubstrateCompleteTargetPanel,
        }
        values = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                key = (manifest.capability_key, schema, ArtifactProfile.CANONICAL_JSON)
                if planned is not None and key not in planned:
                    continue
                record_type = record_types[schema]
                values.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(sorted(value.name for value in fields(record_type))),
                    )
                )
        return tuple(sorted(values, key=lambda value: (value.capability_key, value.payload_schema)))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("Grid2Op adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "GRID2OP_PHASE_PROVIDER_KEY",
    "GRID2OP_IDENTITY_NAMESPACE",
    "GRID2OP_PHASE_PROTOCOL_VERSION",
    "GRID2OP_REDUCTION_STEP_ID",
    "GRID2OP_SOURCE_ARTIFACT_ID",
    'Grid2OpPanelReductionConfig',
    "Grid2OpPhaseRuntimeProvider",
    "build_grid2op_phase_protocol",
    "decode_grid2op_panel_reduction_config",
    'grid2op_grid2op_response_candidate_registrations',
    'grid2op_grid2op_response_registry',
    'grid2op_grid2op_response_runners',
    "grid2op_panel_reduction_config",
    "grid2op_phase_issue_artifact_id",
    'grid2op_phase_study_template',
    "grid2op_phase_scientific_graph",
]

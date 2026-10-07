"""Static NREL safe-decode/reduction capabilities and scientific DAG."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from hashlib import sha256
from typing import ClassVar, Final

from empirical_lawhood.adapters.methods.structured_target import IndependentSubstrateCompleteTargetPanel, IndependentSubstrateStructuredPhaseIssue, IndependentSubstrateTargetPhase
from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateTargetKind
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
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
    ExternalInputSource,
)

from .analysis import build_nrel_panel
from .contracts import NRELArchiveDecodeResult, NRELArchiveSourceBinding, NRELSafeDecodeProfile
from .safe_decode import safe_decode_nrel_table


NREL_ARCHIVE_PROTOCOL_VERSION: Final = "1.0.0"
NREL_ARCHIVE_PROVIDER_KEY: Final = "independent-substrate-grounding.nrel-held-archive-provider"
NREL_HELD_TABLE_SCHEMA: Final = 'empirical-lawhood/physical/nrel-inverter-archive/held-table-bytes'
NREL_SOURCE_ARTIFACT_ID: Final = "artifact.independent-substrate-grounding.nrel.source-binding"
NREL_PROFILE_ARTIFACT_ID: Final = "artifact.independent-substrate-grounding.nrel.safe-decode-profile"
NREL_PHASE_ISSUE_ARTIFACT_ID: Final = "artifact.independent-substrate-grounding.nrel.phase-issue"
NREL_HELD_TABLE_ARTIFACT_ID: Final = "artifact.independent-substrate-grounding.nrel.held-table-member"
NREL_DECODE_STEP_ID: Final = "decode-nrel-held-intervention-table"
NREL_REDUCTION_STEP_ID: Final = "reduce-nrel-complete-preparation-panel"


def _decode_key(phase: IndependentSubstrateTargetPhase) -> str:
    if phase is IndependentSubstrateTargetPhase.PROSPECTIVE_VALIDATION:
        raise ValueError("NREL retrospective archive cannot define controller use")
    return f"nrel.safe-decode-independent-substrate-grounding-{phase.value.lower()}-partition"


def _reduction_key(phase: IndependentSubstrateTargetPhase) -> str:
    if phase is IndependentSubstrateTargetPhase.PROSPECTIVE_VALIDATION:
        raise ValueError("NREL retrospective archive cannot define controller use")
    return f"nrel.reduce-independent-substrate-grounding-{phase.value.lower()}-panel"


@dataclass(frozen=True, slots=True)
class NRELPanelReductionConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/nrel-inverter-archive/nrel-panel-reduction-config'

    config_id: str
    phase: IndependentSubstrateTargetPhase
    source_binding: ObjectIdentity
    decode_profile: ObjectIdentity
    phase_issue: ObjectIdentity
    planned_preparation_ids: tuple[str, ...]
    planned_unit_ids: tuple[str, ...]
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.phase is IndependentSubstrateTargetPhase.PROSPECTIVE_VALIDATION:
            raise ValueError("NREL retrospective archive cannot define controller use")
        if self.source_binding.object_schema != NRELArchiveSourceBinding.SCHEMA:
            raise ValueError("NREL reducer source identity differs")
        if self.decode_profile.object_schema != NRELSafeDecodeProfile.SCHEMA:
            raise ValueError("NREL reducer decode profile identity differs")
        if self.phase_issue.object_schema != IndependentSubstrateStructuredPhaseIssue.SCHEMA:
            raise ValueError("NREL reducer phase issue identity differs")
        require_sorted_unique_strings(
            self.planned_preparation_ids,
            field_name="planned_preparation_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.planned_unit_ids,
            field_name="planned_unit_ids",
            allow_empty=False,
        )
        if self.planned_unit_ids != self.planned_preparation_ids:
            raise ValueError("NREL planned units differ from physical preparations")
        if not 0 < self.maximum_input_bytes <= 2 * 1024**3:
            raise ValueError("NREL reducer input bound differs")


def nrel_panel_reduction_config(
    *,
    source: NRELArchiveSourceBinding,
    profile: NRELSafeDecodeProfile,
    phase_issue: IndependentSubstrateStructuredPhaseIssue,
) -> NRELPanelReductionConfig:
    if profile.source_binding != ObjectIdentity.from_record(source.binding_id, source):
        raise ValueError("NREL reducer profile names another source")
    if (
        phase_issue.slot is not IndependentSubstrateTargetKind.NREL_INVERTER
        or phase_issue.phase is not profile.phase
        or phase_issue.source_binding != profile.source_binding
        or phase_issue.planned_unit_ids != profile.included_preparation_ids
    ):
        raise ValueError("NREL reducer profile differs from the frozen phase issue")
    return NRELPanelReductionConfig(
        config_id=f"config.independent-substrate-grounding.nrel.{profile.phase.value.lower()}-panel-reduction",
        phase=profile.phase,
        source_binding=profile.source_binding,
        decode_profile=ObjectIdentity.from_record(profile.profile_id, profile),
        phase_issue=ObjectIdentity.from_record(phase_issue.issue_id, phase_issue),
        planned_preparation_ids=profile.included_preparation_ids,
        planned_unit_ids=profile.included_preparation_ids,
        maximum_input_bytes=2 * 1024**3,
    )


def _decode_budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=2,
        memory_bytes=8 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=2 * 60 * 60,
        source_scan_bytes=64 * 1024**3,
        output_bytes=2 * 1024**3,
    )


def _reduction_budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=2,
        memory_bytes=8 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=30 * 60,
        source_scan_bytes=2 * 1024**3,
        output_bytes=512 * 1024**2,
    )


def nrel_nrel_inverter_archive_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    capabilities: list[CapabilityManifest] = []
    for phase in (IndependentSubstrateTargetPhase.DEVELOPMENT, IndependentSubstrateTargetPhase.EVALUATION):
        decode_permissions = tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                    *(
                        (CapabilityPermission.READ_DEVELOPMENT,)
                        if phase is IndependentSubstrateTargetPhase.DEVELOPMENT
                        else (CapabilityPermission.READ_SEALED_OUTCOMES,)
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
                    capability_key=_decode_key(phase),
                    capability_version=NREL_ARCHIVE_PROTOCOL_VERSION,
                    kind=CapabilityKind.TRANSFORM,
                    config_schema=NRELSafeDecodeProfile.SCHEMA,
                    config_schema_sha256=sha256(
                        NRELSafeDecodeProfile.SCHEMA.encode("ascii")
                    ).hexdigest(),
                    input_schema_ids=tuple(
                        sorted(
                            (
                                NRELArchiveSourceBinding.SCHEMA,
                                NREL_HELD_TABLE_SCHEMA,
                                NRELSafeDecodeProfile.SCHEMA,
                                IndependentSubstrateStructuredPhaseIssue.SCHEMA,
                            )
                        )
                    ),
                    output_schema_ids=(NRELArchiveDecodeResult.SCHEMA,),
                    permissions=decode_permissions,
                    maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
                    maximum_outcome_access=(
                        OutcomeAccess.DEVELOPMENT_VISIBLE
                        if phase is IndependentSubstrateTargetPhase.DEVELOPMENT
                        else OutcomeAccess.EVALUATION_SEALED
                    ),
                    resource_ceiling=_decode_budget(),
                    deterministic=True,
                    seed_required=False,
                    language_id="python",
                    runtime_id="nrel-bounded-utf8-table-decoder",
                    requires_clean_commit=True,
                    requires_active_mount=True,
                    requires_network=False,
                    conformance_check_ids=tuple(
                        sorted(
                            (
                                "exact-held-member-hash-before-and-after-decode",
                                "no-executable-or-object-deserialization",
                                "preparation-filter-frozen-before-receiver-decode",
                                "strict-header-unit-clock-action-roster",
                            )
                        )
                    ),
                    implementation_sha256=implementation_sha256,
                ),
                CapabilityManifest(
                    capability_key=_reduction_key(phase),
                    capability_version=NREL_ARCHIVE_PROTOCOL_VERSION,
                    kind=(
                        CapabilityKind.TRANSFORM
                        if phase is IndependentSubstrateTargetPhase.DEVELOPMENT
                        else CapabilityKind.EVALUATOR
                    ),
                    config_schema=NRELPanelReductionConfig.SCHEMA,
                    config_schema_sha256=sha256(
                        NRELPanelReductionConfig.SCHEMA.encode("ascii")
                    ).hexdigest(),
                    input_schema_ids=tuple(
                        sorted(
                            (
                                NRELArchiveDecodeResult.SCHEMA,
                                NRELArchiveSourceBinding.SCHEMA,
                                NRELPanelReductionConfig.SCHEMA,
                                NRELSafeDecodeProfile.SCHEMA,
                                IndependentSubstrateStructuredPhaseIssue.SCHEMA,
                            )
                        )
                    ),
                    output_schema_ids=(IndependentSubstrateCompleteTargetPanel.SCHEMA,),
                    permissions=reducer_permissions,
                    maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
                    maximum_outcome_access=(
                        OutcomeAccess.DEVELOPMENT_VISIBLE
                        if phase is IndependentSubstrateTargetPhase.DEVELOPMENT
                        else OutcomeAccess.EVALUATOR_REVEAL
                    ),
                    resource_ceiling=_reduction_budget(),
                    deterministic=True,
                    seed_required=False,
                    language_id="python",
                    runtime_id="nrel-complete-preparation-reducer",
                    requires_clean_commit=True,
                    requires_active_mount=True,
                    requires_network=False,
                    conformance_check_ids=tuple(
                        sorted(
                            (
                                "action-clock-missingness-retained",
                                "apparatus-preparation-is-complete-unit",
                                "receiver-backaction-is-source-bound",
                                "rows-and-channels-do-not-inflate-replication",
                            )
                        )
                    ),
                    implementation_sha256=implementation_sha256,
                ),
            )
        )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-nrel-held-archive-registry",
        capabilities=tuple(sorted(capabilities, key=lambda value: value.registry_id)),
    )


def nrel_nrel_inverter_archive_candidate_registrations(
    *,
    implementation_sha256: str,
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=NREL_ARCHIVE_PROVIDER_KEY,
            provider_version=NREL_ARCHIVE_PROTOCOL_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=2 * 1024**2,
        )
        for manifest in nrel_nrel_inverter_archive_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


def _validate_ref(
    *,
    record: CanonicalRecord,
    record_id: str,
    ref: CapabilityConfigRef,
    manifest: CapabilityManifest,
) -> None:
    if (
        ref.config_id != record_id
        or ref.config_schema != record.SCHEMA
        or ref.config_schema_sha256 != manifest.config_schema_sha256
        or ref.content_sha256 != record.fingerprint()
    ):
        raise ValueError("NREL capability config reference differs")


def build_nrel_phase_protocol(
    *,
    registry: CapabilityRegistry,
    profile: NRELSafeDecodeProfile,
    phase_issue: IndependentSubstrateStructuredPhaseIssue,
    profile_config_ref: CapabilityConfigRef,
    reduction_config: NRELPanelReductionConfig,
    reduction_config_ref: CapabilityConfigRef,
) -> ProtocolTemplate:
    if (
        reduction_config.phase is not profile.phase
        or reduction_config.decode_profile
        != ObjectIdentity.from_record(profile.profile_id, profile)
        or reduction_config.phase_issue
        != ObjectIdentity.from_record(phase_issue.issue_id, phase_issue)
    ):
        raise ValueError("NREL protocol profile/reduction phase differs")
    decode_manifest = registry.resolve(_decode_key(profile.phase), NREL_ARCHIVE_PROTOCOL_VERSION)
    reducer_manifest = registry.resolve(_reduction_key(profile.phase), NREL_ARCHIVE_PROTOCOL_VERSION)
    _validate_ref(
        record=profile,
        record_id=profile.profile_id,
        ref=profile_config_ref,
        manifest=decode_manifest,
    )
    _validate_ref(
        record=reduction_config,
        record_id=reduction_config.config_id,
        ref=reduction_config_ref,
        manifest=reducer_manifest,
    )
    development = profile.phase is IndependentSubstrateTargetPhase.DEVELOPMENT
    decoder = ProtocolStepTemplate(
        step_id=NREL_DECODE_STEP_ID,
        stage=ScientificStage.TRANSFORM,
        capability_key=decode_manifest.capability_key,
        capability_version=decode_manifest.capability_version,
        config=profile_config_ref,
        dependency_step_ids=(),
        outputs=(
            OutputTemplate(
                output_id="safe-decode-result",
                payload_schema=NRELArchiveDecodeResult.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                filename_suffix=".json",
            ),
        ),
        required_permissions=decode_manifest.permissions,
        requested_outcome_access=(
            OutcomeAccess.DEVELOPMENT_VISIBLE if development else OutcomeAccess.EVALUATION_SEALED
        ),
        visibility_ceiling=(
            VisibilityCeiling.DEVELOPMENT_ONLY if development else VisibilityCeiling.PROSPECTIVE
        ),
        resource_budget=decode_manifest.resource_ceiling,
        resource_lock_ids=("nrel-held-member-decoder",),
        barrier=BarrierKind.NONE if development else BarrierKind.FREEZE,
        maximum_attempts=2,
        obligation_ids=(
            "nrel-exact-held-member-and-safe-decoder",
            "nrel-frozen-row-partition-before-receiver-decode",
        ),
    )
    reducer = ProtocolStepTemplate(
        step_id=NREL_REDUCTION_STEP_ID,
        stage=ScientificStage.DEVELOP if development else ScientificStage.REVEAL,
        capability_key=reducer_manifest.capability_key,
        capability_version=reducer_manifest.capability_version,
        config=reduction_config_ref,
        dependency_step_ids=(decoder.step_id,),
        outputs=(
            OutputTemplate(
                output_id="complete-preparation-panel",
                payload_schema=IndependentSubstrateCompleteTargetPanel.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                filename_suffix=".json",
            ),
        ),
        required_permissions=reducer_manifest.permissions,
        requested_outcome_access=(
            OutcomeAccess.DEVELOPMENT_VISIBLE if development else OutcomeAccess.EVALUATOR_REVEAL
        ),
        visibility_ceiling=(
            VisibilityCeiling.DEVELOPMENT_ONLY if development else VisibilityCeiling.OUTCOME_VISIBLE
        ),
        resource_budget=reducer_manifest.resource_ceiling,
        resource_lock_ids=("nrel-complete-preparation-reducer",),
        barrier=BarrierKind.NONE if development else BarrierKind.REVEAL,
        maximum_attempts=2,
        obligation_ids=(
            "nrel-action-and-receiver-missingness-retained",
            "nrel-complete-preparation-resampling-unit",
            "nrel-one-apparatus-physical-ceiling",
            "nrel-retrospective-no-prospective-controller-evaluation-promotion",
        ),
    )
    return ProtocolTemplate(
        template_id=f"independent-substrate-grounding-nrel-{profile.phase.value.lower()}-phase",
        template_version=NREL_ARCHIVE_PROTOCOL_VERSION,
        steps=(decoder, reducer),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _record_input(
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


def nrel_phase_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    source: NRELArchiveSourceBinding,
    profile: NRELSafeDecodeProfile,
    phase_issue: IndependentSubstrateStructuredPhaseIssue,
) -> CandidateScientificGraph:
    if (
        profile.source_binding != ObjectIdentity.from_record(source.binding_id, source)
        or phase_issue.slot is not IndependentSubstrateTargetKind.NREL_INVERTER
        or phase_issue.phase is not profile.phase
        or phase_issue.source_binding != profile.source_binding
        or phase_issue.planned_unit_ids != profile.included_preparation_ids
    ):
        raise ValueError("NREL graph profile/source differs")
    decoder = next(value for value in protocol.steps if value.step_id == NREL_DECODE_STEP_ID)
    reducer = next(value for value in protocol.steps if value.step_id == NREL_REDUCTION_STEP_ID)
    member = next(
        (
            value
            for value in source.members
            if value.member_id == profile.member.object_id
            and ObjectIdentity.from_record(value.member_id, value) == profile.member
        ),
        None,
    )
    if member is None:
        raise ValueError("NREL graph profile member differs from source")
    source_input = _record_input(
        input_id="input.independent-substrate-grounding.nrel.source-binding",
        role=ScientificInputRole.SOURCE,
        artifact_id=NREL_SOURCE_ARTIFACT_ID,
        record=source,
    )
    profile_input = _record_input(
        input_id="input.independent-substrate-grounding.nrel.safe-decode-profile",
        role=ScientificInputRole.QUALIFICATION,
        artifact_id=NREL_PROFILE_ARTIFACT_ID,
        record=profile,
    )
    issue_input = _record_input(
        input_id="input.independent-substrate-grounding.nrel.phase-issue",
        role=ScientificInputRole.QUALIFICATION,
        artifact_id=NREL_PHASE_ISSUE_ARTIFACT_ID,
        record=phase_issue,
    )
    raw_input = CandidateGraphExternalInput(
        input_id="input.independent-substrate-grounding.nrel.held-table-member",
        scientific_role=ScientificInputRole.OUTCOME,
        logical_artifact_id=NREL_HELD_TABLE_ARTIFACT_ID,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=member.content_sha256,
        payload_schema=NREL_HELD_TABLE_SCHEMA,
        media_type=("text/csv" if profile.delimiter == "," else "text/tab-separated-values"),
        maximum_size_bytes=profile.maximum_payload_bytes,
        outcome_access=profile.outcome_access,
        visibility_ceiling=(
            VisibilityCeiling.DEVELOPMENT_ONLY
            if profile.phase is IndependentSubstrateTargetPhase.DEVELOPMENT
            else VisibilityCeiling.PROSPECTIVE
        ),
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
    edges = (
        CandidateGraphEdge(
            edge_id="edge.nrel.source.decoder",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=source_input.input_id,
            consumer_node_id=decoder.step_id,
            consumer_input_id="frozen-source-binding",
            scientific_role=source_input.scientific_role,
            logical_artifact_id=source_input.logical_artifact_id,
            payload_schema=source_input.payload_schema,
            media_type=source_input.media_type,
            maximum_size_bytes=source_input.maximum_size_bytes,
            outcome_access=source_input.outcome_access,
            visibility_ceiling=source_input.visibility_ceiling,
            barrier=decoder.barrier,
        ),
        CandidateGraphEdge(
            edge_id="edge.nrel.held-member.decoder",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=raw_input.input_id,
            consumer_node_id=decoder.step_id,
            consumer_input_id="held-table-bytes",
            scientific_role=raw_input.scientific_role,
            logical_artifact_id=raw_input.logical_artifact_id,
            payload_schema=raw_input.payload_schema,
            media_type=raw_input.media_type,
            maximum_size_bytes=raw_input.maximum_size_bytes,
            outcome_access=raw_input.outcome_access,
            visibility_ceiling=raw_input.visibility_ceiling,
            barrier=decoder.barrier,
        ),
        CandidateGraphEdge(
            edge_id="edge.nrel.phase-issue.decoder",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=issue_input.input_id,
            consumer_node_id=decoder.step_id,
            consumer_input_id="frozen-phase-issue",
            scientific_role=issue_input.scientific_role,
            logical_artifact_id=issue_input.logical_artifact_id,
            payload_schema=issue_input.payload_schema,
            media_type=issue_input.media_type,
            maximum_size_bytes=issue_input.maximum_size_bytes,
            outcome_access=issue_input.outcome_access,
            visibility_ceiling=issue_input.visibility_ceiling,
            barrier=decoder.barrier,
        ),
        CandidateGraphEdge(
            edge_id="edge.nrel.decode-result.reducer",
            producer_node_id=decoder.step_id,
            producer_output_id="safe-decode-result",
            external_input_id=None,
            consumer_node_id=reducer.step_id,
            consumer_input_id="safe-decode-result",
            scientific_role=ScientificInputRole.OUTCOME,
            logical_artifact_id="artifact.independent-substrate-grounding.nrel.safe-decode-result",
            payload_schema=NRELArchiveDecodeResult.SCHEMA,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_size_bytes=2 * 1024**3,
            outcome_access=decoder.requested_outcome_access,
            visibility_ceiling=decoder.visibility_ceiling,
            barrier=reducer.barrier,
        ),
        CandidateGraphEdge(
            edge_id="edge.nrel.profile.reducer",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=profile_input.input_id,
            consumer_node_id=reducer.step_id,
            consumer_input_id="frozen-decode-profile",
            scientific_role=profile_input.scientific_role,
            logical_artifact_id=profile_input.logical_artifact_id,
            payload_schema=profile_input.payload_schema,
            media_type=profile_input.media_type,
            maximum_size_bytes=profile_input.maximum_size_bytes,
            outcome_access=profile_input.outcome_access,
            visibility_ceiling=profile_input.visibility_ceiling,
            barrier=reducer.barrier,
        ),
        CandidateGraphEdge(
            edge_id="edge.nrel.phase-issue.reducer",
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
        ),
        CandidateGraphEdge(
            edge_id="edge.nrel.source.reducer",
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
        ),
    )
    return CandidateScientificGraph(
        graph_id=f"graph.{protocol.template_id}",
        external_inputs=tuple(
            sorted(
                (source_input, profile_input, issue_input, raw_input),
                key=lambda value: value.input_id,
            )
        ),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def nrel_phase_study_template(
    *,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> StudyTemplate:
    steps = {value.step_id: value for value in protocol.steps}
    bindings = tuple(
        sorted(
            (
                ObligationCoverageBinding(
                    obligation_id=obligation_id,
                    proof_owner_node_id=node.node_id,
                    required_output_id=steps[node.node_id].outputs[0].output_id,
                    contributor_edge_ids=tuple(
                        sorted(
                            edge.edge_id
                            for edge in graph.edges
                            if edge.consumer_node_id == node.node_id
                        )
                    ),
                )
                for node in graph.nodes
                for obligation_id in node.obligation_ids
            ),
            key=lambda value: value.obligation_id,
        )
    )
    return StudyTemplate(
        template_key=protocol.template_id,
        template_version=protocol.template_version,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id=f"coverage.{protocol.template_id}",
            bindings=bindings,
        ),
    )


class _NRELSafeDecodeRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        sources = tuple(
            decode_canonical_bytes(
                port.read(), NRELArchiveSourceBinding, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == NRELArchiveSourceBinding.SCHEMA
        )
        profiles = tuple(
            decode_canonical_bytes(
                port.read(), NRELSafeDecodeProfile, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == NRELSafeDecodeProfile.SCHEMA
        )
        issues = tuple(
            decode_canonical_bytes(
                port.read(), IndependentSubstrateStructuredPhaseIssue, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == IndependentSubstrateStructuredPhaseIssue.SCHEMA
        )
        payloads = tuple(
            port.read()
            for port in context.input_ports
            if port.payload_schema == NREL_HELD_TABLE_SCHEMA
        )
        if len(sources) != 1 or len(profiles) != 1 or len(issues) != 1 or len(payloads) != 1:
            raise ValueError("NREL decoder requires one source/profile/issue/member")
        profile = profiles[0]
        if (
            self.manifest.capability_key != _decode_key(profile.phase)
            or issues[0].phase is not profile.phase
            or issues[0].source_binding
            != ObjectIdentity.from_record(sources[0].binding_id, sources[0])
            or issues[0].planned_unit_ids != profile.included_preparation_ids
        ):
            raise ValueError("NREL decoder phase selects another capability")
        result = safe_decode_nrel_table(
            payload=payloads[0],
            source=sources[0],
            profile=profile,
            sealed_evaluation_decode_authorized=(
                profile.phase is IndependentSubstrateTargetPhase.EVALUATION
                and context.outcome_access is OutcomeAccess.EVALUATION_SEALED
            ),
        )
        if {value.payload_schema for value in context.output_ports} != {
            NRELArchiveDecodeResult.SCHEMA
        }:
            raise ValueError("NREL safe decode output schema differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=result.canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("nrel-member-hash-stable", True, ()),
                ReceiptCheck("nrel-no-object-deserialization", True, ()),
                ReceiptCheck("nrel-preparation-filter-exact", True, ()),
            ),
        )


class _NRELReductionRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        configs = tuple(
            decode_canonical_bytes(
                port.read(), NRELPanelReductionConfig, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == NRELPanelReductionConfig.SCHEMA
        )
        sources = tuple(
            decode_canonical_bytes(
                port.read(), NRELArchiveSourceBinding, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == NRELArchiveSourceBinding.SCHEMA
        )
        profiles = tuple(
            decode_canonical_bytes(
                port.read(), NRELSafeDecodeProfile, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == NRELSafeDecodeProfile.SCHEMA
        )
        issues = tuple(
            decode_canonical_bytes(
                port.read(), IndependentSubstrateStructuredPhaseIssue, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == IndependentSubstrateStructuredPhaseIssue.SCHEMA
        )
        results = tuple(
            decode_canonical_bytes(
                port.read(), NRELArchiveDecodeResult, maximum_bytes=port.size_bytes
            )
            for port in context.input_ports
            if port.payload_schema == NRELArchiveDecodeResult.SCHEMA
        )
        if not (len(configs) == len(sources) == len(profiles) == len(issues) == len(results) == 1):
            raise ValueError("NREL reducer requires one config/source/profile/issue/result")
        config, source, profile, result = (
            configs[0],
            sources[0],
            profiles[0],
            results[0],
        )
        if (
            self.manifest.capability_key != _reduction_key(config.phase)
            or config.source_binding != ObjectIdentity.from_record(source.binding_id, source)
            or config.decode_profile != ObjectIdentity.from_record(profile.profile_id, profile)
            or config.phase_issue != ObjectIdentity.from_record(issues[0].issue_id, issues[0])
        ):
            raise ValueError("NREL reducer config/source/profile differs")
        panel = build_nrel_panel(source=source, profile=profile, result=result)
        if panel.planned_unit_ids != config.planned_unit_ids:
            raise ValueError("NREL reducer output denominator differs from issue")
        if {value.payload_schema for value in context.output_ports} != {
            IndependentSubstrateCompleteTargetPanel.SCHEMA
        }:
            raise ValueError("NREL reduction output schema differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=panel.canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("nrel-issued-preparations-retained", True, ()),
                ReceiptCheck("nrel-rows-and-channels-not-units", True, ()),
                ReceiptCheck("nrel-retrospective-prospective-controller-evaluation-excluded", True, ()),
            ),
        )


def nrel_nrel_inverter_archive_runners(*, registry: CapabilityRegistry) -> tuple[TaskRunner, ...]:
    return tuple(
        runner
        for phase in (IndependentSubstrateTargetPhase.DEVELOPMENT, IndependentSubstrateTargetPhase.EVALUATION)
        for runner in (
            _NRELSafeDecodeRunner(registry.resolve(_decode_key(phase), NREL_ARCHIVE_PROTOCOL_VERSION)),
            _NRELReductionRunner(registry.resolve(_reduction_key(phase), NREL_ARCHIVE_PROTOCOL_VERSION)),
        )
    )


class NRELHeldArchiveRuntimeProvider(CampaignRuntimeProvider):
    """Bind held member bytes through an injected bounded source, never a path."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        source: NRELArchiveSourceBinding,
        profile: NRELSafeDecodeProfile,
        phase_issue: IndependentSubstrateStructuredPhaseIssue,
        reduction_config: NRELPanelReductionConfig,
        held_member_source_factory: Callable[[], ExternalInputSource],
    ) -> None:
        implementation_hashes = {value.implementation_sha256 for value in registry.capabilities}
        if len(implementation_hashes) != 1 or registry != nrel_nrel_inverter_archive_registry(
            implementation_sha256=next(iter(implementation_hashes))
        ):
            raise ValueError("NREL runtime provider registry differs")
        source_identity = ObjectIdentity.from_record(source.binding_id, source)
        profile_identity = ObjectIdentity.from_record(profile.profile_id, profile)
        if (
            profile.source_binding != source_identity
            or reduction_config.source_binding != source_identity
            or reduction_config.decode_profile != profile_identity
            or reduction_config.phase_issue
            != ObjectIdentity.from_record(phase_issue.issue_id, phase_issue)
            or reduction_config.phase is not profile.phase
        ):
            raise ValueError("NREL runtime provider source/profile scope differs")
        member = next(
            (
                value
                for value in source.members
                if value.member_id == profile.member.object_id
                and ObjectIdentity.from_record(value.member_id, value) == profile.member
            ),
            None,
        )
        if member is None:
            raise ValueError("NREL runtime provider member differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.source = source
        self.profile = profile
        self.phase_issue = phase_issue
        self.reduction_config = reduction_config
        self.member = member
        self._held_member_source_factory = held_member_source_factory
        self._runners = nrel_nrel_inverter_archive_runners(registry=registry)
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("NREL runtime provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("NREL runtime provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            NREL_SOURCE_ARTIFACT_ID: self.source,
            NREL_PROFILE_ARTIFACT_ID: self.profile,
            NREL_PHASE_ISSUE_ARTIFACT_ID: self.phase_issue,
        }
        by_hash: dict[str, CanonicalRecord] = {
            self.profile.fingerprint(): self.profile,
            self.reduction_config.fingerprint(): self.reduction_config,
        }
        for task in plan.tasks:
            try:
                records[task.capability.config.artifact_id] = by_hash[
                    task.capability.config.content_sha256
                ]
            except KeyError as error:
                raise ValueError("NREL plan references an unknown config") from error
        canonical_spec_ids = set(specs) - {NREL_HELD_TABLE_ARTIFACT_ID}
        if set(records) != canonical_spec_ids or NREL_HELD_TABLE_ARTIFACT_ID not in specs:
            raise ValueError("NREL runtime external input roster differs")
        values: list[ExternalInputPayload] = []
        for artifact_id, record in sorted(records.items()):
            spec = specs[artifact_id]
            record_id = next(
                getattr(record, attribute)
                for attribute in (
                    "profile_id",
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
        raw_spec = specs[NREL_HELD_TABLE_ARTIFACT_ID]
        raw_parent = ArtifactLineageParent(
            identity=ObjectIdentity.from_record(self.member.member_id, self.member),
            visibility_ceiling=(
                raw_spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
            ),
            outcome_access=(raw_spec.expected_outcome_access or self.profile.outcome_access),
        )
        values.append(
            ExternalInputPayload(
                logical_artifact_id=NREL_HELD_TABLE_ARTIFACT_ID,
                payload_schema=NREL_HELD_TABLE_SCHEMA,
                profile=ArtifactProfile.TEXT_PARAMETERS,
                media_type=(
                    "text/csv" if self.profile.delimiter == "," else "text/tab-separated-values"
                ),
                source=self._held_member_source_factory(),
                size_bytes=self.member.size_bytes,
                source_sha256=self.member.content_sha256,
                maximum_bytes=self.profile.maximum_payload_bytes,
                maximum_chunk_bytes=min(
                    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
                    self.profile.maximum_payload_bytes,
                ),
                visibility_ceiling=raw_parent.visibility_ceiling,
                outcome_access=raw_parent.outcome_access,
                parent_visibility_ceilings=(raw_parent.visibility_ceiling,),
                lineage_parents=(raw_parent,),
                logical_content_sha256=raw_spec.expected_content_sha256,
            )
        )
        return tuple(sorted(values, key=lambda value: value.logical_artifact_id))

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("NREL semantic registry differs")
        planned = None
        if execution_plan is not None:
            planned = {
                (task.capability.capability_key, output.payload_schema, output.profile)
                for task in execution_plan.tasks
                for output in task.outputs
            }
        record_types = {
            NRELArchiveDecodeResult.SCHEMA: NRELArchiveDecodeResult,
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
            raise ValueError("NREL adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "NREL_DECODE_STEP_ID",
    "NREL_HELD_TABLE_ARTIFACT_ID",
    "NREL_HELD_TABLE_SCHEMA",
    "NREL_ARCHIVE_PROVIDER_KEY",
    "NREL_ARCHIVE_PROTOCOL_VERSION",
    "NREL_PROFILE_ARTIFACT_ID",
    "NREL_PHASE_ISSUE_ARTIFACT_ID",
    "NREL_REDUCTION_STEP_ID",
    "NREL_SOURCE_ARTIFACT_ID",
    'NRELPanelReductionConfig',
    "NRELHeldArchiveRuntimeProvider",
    "build_nrel_phase_protocol",
    'nrel_nrel_inverter_archive_candidate_registrations',
    'nrel_nrel_inverter_archive_registry',
    'nrel_nrel_inverter_archive_runners',
    "nrel_panel_reduction_config",
    'nrel_phase_study_template',
    "nrel_phase_scientific_graph",
]

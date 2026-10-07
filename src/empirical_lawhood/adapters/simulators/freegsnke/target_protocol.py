"""Compiled phase-generation and complete-unit reduction DAG for FreeGSNKE."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from hashlib import sha256
from typing import Final

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
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
from empirical_lawhood.runtime.candidate_compiler import (
    CandidateGraphEdge,
    CandidateGraphExternalInput,
    CandidateGraphNode,
    CandidateScientificGraph,
    ContentIdentityPolicy,
    ScientificInputRole,
)
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)

from .contracts import FreeGsnkePhase, FreeGsnkeProcessRequest, FreeGsnkeSourceBinding, FreeGsnkeTargetProcessResponse
from .design import FreeGsnkeActionDesign, FreeGsnkePhaseRequestRoster
from .generation_protocol import (
    FREEGSNKE_SOURCE_ARTIFACT_ID,
    FreeGsnkeEpisodeExecutor,
    build_freegsnke_generation_protocol,
    freegsnke_generation_registry,
    freegsnke_generation_runners,
)
from .registry import (
    FREEGSNKE_CAPABILITY_VERSION,
    FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
    FREEGSNKE_EVALUATION_CAPABILITY_KEY,
)
from .target_analysis import FreeGsnkePhaseReduction, FreeGsnkeTargetReductionConfig, reduce_freegsnke_phase_panel


FREEGSNKE_DEVELOPMENT_REDUCTION_KEY: Final = "freegsnke.reduce-development-panel"
FREEGSNKE_EVALUATION_REDUCTION_KEY: Final = "freegsnke.reduce-evaluation-panel"
FREEGSNKE_PHASE_PROVIDER_KEY: Final = "independent-substrate-grounding.freegsnke-phase-provider"
FREEGSNKE_PHASE_DESIGN_ARTIFACT_ID: Final = "artifact.independent-substrate-grounding.freegsnke.action-design"
FREEGSNKE_PHASE_ROSTER_ARTIFACT_ID: Final = "artifact.independent-substrate-grounding.freegsnke.phase-roster"
FREEGSNKE_REDUCTION_STEP_ID: Final = "reduce-complete-preparation-panel"


def _request_artifact_id(request_id: str) -> str:
    return f"artifact.independent-substrate-grounding.freegsnke.reduction-request.{request_id}"


def _reduction_key(phase: FreeGsnkePhase) -> str:
    if phase is FreeGsnkePhase.DEVELOPMENT:
        return FREEGSNKE_DEVELOPMENT_REDUCTION_KEY
    if phase is FreeGsnkePhase.EVALUATION:
        return FREEGSNKE_EVALUATION_REDUCTION_KEY
    raise ValueError("FreeGSNKE phase reduction supports development/evaluation only")


def _phase_access(phase: FreeGsnkePhase) -> OutcomeAccess:
    return (
        OutcomeAccess.DEVELOPMENT_VISIBLE
        if phase is FreeGsnkePhase.DEVELOPMENT
        else OutcomeAccess.EVALUATOR_REVEAL
    )


def _phase_visibility(phase: FreeGsnkePhase) -> VisibilityCeiling:
    return (
        VisibilityCeiling.DEVELOPMENT_ONLY
        if phase is FreeGsnkePhase.DEVELOPMENT
        else VisibilityCeiling.OUTCOME_VISIBLE
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


def freegsnke_phase_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    """Close generation and phase reduction under one executable registry.

    Development reduction is an ordinary visible transform.  Evaluation
    reduction is the single evaluator-owned reveal node; the runtime forbids a
    non-evaluator transform from reading sealed outcomes.
    """

    generation = freegsnke_generation_registry(
        implementation_sha256=implementation_sha256
    ).capabilities
    schema_sha256 = sha256(FreeGsnkeTargetReductionConfig.SCHEMA.encode("ascii")).hexdigest()
    common_inputs = tuple(
        sorted(
            (
                FreeGsnkeActionDesign.SCHEMA,
                FreeGsnkePhaseRequestRoster.SCHEMA,
                FreeGsnkeProcessRequest.SCHEMA,
                FreeGsnkeTargetProcessResponse.SCHEMA,
                FreeGsnkeTargetReductionConfig.SCHEMA,
            )
        )
    )
    common_permissions = (
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    )
    reductions = tuple(
        CapabilityManifest(
            capability_key=key,
            capability_version=FREEGSNKE_CAPABILITY_VERSION,
            kind=kind,
            config_schema=FreeGsnkeTargetReductionConfig.SCHEMA,
            config_schema_sha256=schema_sha256,
            input_schema_ids=common_inputs,
            output_schema_ids=(FreeGsnkePhaseReduction.SCHEMA,),
            permissions=permissions,
            maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            maximum_outcome_access=access,
            resource_ceiling=_reduction_budget(),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="independent-substrate-grounding-freegsnke-complete-preparation-reducer",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=(
                "all-issued-units-retained",
                "complete-preparation-is-resampling-unit",
                "native-hold-contrasts-preserved",
                "postissue-adverse-outcomes-not-nonentry",
                "requested-accepted-applied-realized-clocks-distinct",
            ),
            implementation_sha256=implementation_sha256,
        )
        for key, kind, access, permissions in (
            (
                FREEGSNKE_DEVELOPMENT_REDUCTION_KEY,
                CapabilityKind.TRANSFORM,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                tuple(
                    sorted(
                        (
                            *common_permissions,
                            CapabilityPermission.READ_DEVELOPMENT,
                        )
                    )
                ),
            ),
            (
                FREEGSNKE_EVALUATION_REDUCTION_KEY,
                CapabilityKind.EVALUATOR,
                OutcomeAccess.EVALUATOR_REVEAL,
                tuple(
                    sorted(
                        (
                            *common_permissions,
                            CapabilityPermission.READ_SEALED_OUTCOMES,
                            CapabilityPermission.REVEAL_OUTCOMES,
                        )
                    )
                ),
            ),
        )
    )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-freegsnke-phase-registry",
        capabilities=tuple(sorted((*generation, *reductions), key=lambda value: value.registry_id)),
    )


def freegsnke_phase_candidate_registrations(
    *,
    implementation_sha256: str,
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=FREEGSNKE_PHASE_PROVIDER_KEY,
            provider_version=FREEGSNKE_CAPABILITY_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=2 * 1024**2,
        )
        for manifest in freegsnke_phase_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


def build_freegsnke_phase_protocol(
    *,
    registry: CapabilityRegistry,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    request_config_refs: tuple[CapabilityConfigRef, ...],
    reduction_config: FreeGsnkeTargetReductionConfig,
    reduction_config_ref: CapabilityConfigRef,
) -> ProtocolTemplate:
    """Build generation fan-out followed by the phase-owned reduction node."""

    if not requests or {value.phase for value in requests} != {reduction_config.phase}:
        raise ValueError("FreeGSNKE phase protocol request/config phase differs")
    generation_protocol = build_freegsnke_generation_protocol(
        registry=CapabilityRegistry(
            registry_id="freegsnke-phase-generation-subregistry",
            capabilities=tuple(
                registry.resolve(key, FREEGSNKE_CAPABILITY_VERSION)
                for key in (
                    FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
                    FREEGSNKE_EVALUATION_CAPABILITY_KEY,
                )
            ),
        ),
        requests=requests,
        config_refs=request_config_refs,
    )
    key = _reduction_key(reduction_config.phase)
    manifest = registry.resolve(key, FREEGSNKE_CAPABILITY_VERSION)
    if (
        reduction_config_ref.config_id != reduction_config.config_id
        or reduction_config_ref.config_schema != reduction_config.SCHEMA
        or reduction_config_ref.config_schema_sha256 != manifest.config_schema_sha256
        or reduction_config_ref.content_sha256 != reduction_config.fingerprint()
    ):
        raise ValueError("FreeGSNKE reduction config reference differs")
    reduction_step = ProtocolStepTemplate(
        step_id=FREEGSNKE_REDUCTION_STEP_ID,
        stage=(
            ScientificStage.DEVELOP
            if reduction_config.phase is FreeGsnkePhase.DEVELOPMENT
            else ScientificStage.EVALUATE
        ),
        capability_key=key,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        config=reduction_config_ref,
        dependency_step_ids=tuple(sorted(value.step_id for value in generation_protocol.steps)),
        outputs=(
            OutputTemplate(
                output_id="phase-reduction.record",
                payload_schema=FreeGsnkePhaseReduction.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                filename_suffix=".json",
            ),
        ),
        required_permissions=manifest.permissions,
        requested_outcome_access=_phase_access(reduction_config.phase),
        visibility_ceiling=_phase_visibility(reduction_config.phase),
        resource_budget=manifest.resource_ceiling,
        resource_lock_ids=("freegsnke-phase-reducer",),
        barrier=(
            BarrierKind.NONE
            if reduction_config.phase is FreeGsnkePhase.DEVELOPMENT
            else BarrierKind.REVEAL
        ),
        maximum_attempts=2,
        obligation_ids=(
            "freegsnke-all-issued-preparations-retained",
            "freegsnke-native-hold-contrast",
            "freegsnke-no-nested-replication",
            "freegsnke-postissue-adverse-evidence",
        ),
    )
    return ProtocolTemplate(
        template_id=(f"independent-substrate-grounding-freegsnke-{reduction_config.phase.value.lower()}-phase-protocol"),
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        steps=tuple(
            sorted((*generation_protocol.steps, reduction_step), key=lambda value: value.step_id)
        ),
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


def freegsnke_phase_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    source_binding: FreeGsnkeSourceBinding,
    action_design: FreeGsnkeActionDesign,
    phase_roster: FreeGsnkePhaseRequestRoster,
    requests: tuple[FreeGsnkeProcessRequest, ...],
) -> CandidateScientificGraph:
    """Bind exact source/design/roster/request inputs and response edges."""

    reduction = next(
        value for value in protocol.steps if value.step_id == FREEGSNKE_REDUCTION_STEP_ID
    )
    generation_steps = tuple(
        value for value in protocol.steps if value.step_id != FREEGSNKE_REDUCTION_STEP_ID
    )
    if {value.config.config_id for value in generation_steps} != {
        value.request_id for value in requests
    }:
        raise ValueError("FreeGSNKE phase graph request/protocol roster differs")
    source_input = _external_input(
        input_id="input.independent-substrate-grounding.freegsnke.source-binding",
        role=ScientificInputRole.PREPARED_MEDIUM,
        artifact_id=FREEGSNKE_SOURCE_ARTIFACT_ID,
        record=source_binding,
    )
    design_input = _external_input(
        input_id="input.independent-substrate-grounding.freegsnke.action-design",
        role=ScientificInputRole.ACTION,
        artifact_id=FREEGSNKE_PHASE_DESIGN_ARTIFACT_ID,
        record=action_design,
    )
    roster_input = _external_input(
        input_id="input.independent-substrate-grounding.freegsnke.phase-roster",
        role=ScientificInputRole.DENOMINATOR,
        artifact_id=FREEGSNKE_PHASE_ROSTER_ARTIFACT_ID,
        record=phase_roster,
    )
    request_inputs = tuple(
        _external_input(
            input_id=f"input.independent-substrate-grounding.freegsnke.reduction-request.{value.request_id}",
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
    edges = []
    for step in generation_steps:
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.freegsnke-source.{step.step_id}",
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
        output = step.outputs[0]
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{step.step_id}.response.reduce-phase",
                producer_node_id=step.step_id,
                producer_output_id=output.output_id,
                external_input_id=None,
                consumer_node_id=reduction.step_id,
                consumer_input_id=f"response.{step.config.config_id}",
                scientific_role=ScientificInputRole.OUTCOME,
                logical_artifact_id=f"artifact.{step.step_id}.{output.output_id}",
                payload_schema=output.payload_schema,
                media_type=output.media_type,
                maximum_size_bytes=32 * 1024**2,
                outcome_access=step.requested_outcome_access,
                visibility_ceiling=step.visibility_ceiling,
                barrier=reduction.barrier,
            )
        )
    for external, consumer_input in (
        (design_input, "frozen-action-design"),
        (roster_input, "frozen-phase-roster"),
        *((value, value.input_id) for value in request_inputs),
    ):
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.external.{external.input_id}.reduce-phase",
                producer_node_id=None,
                producer_output_id=None,
                external_input_id=external.input_id,
                consumer_node_id=reduction.step_id,
                consumer_input_id=consumer_input,
                scientific_role=external.scientific_role,
                logical_artifact_id=external.logical_artifact_id,
                payload_schema=external.payload_schema,
                media_type=external.media_type,
                maximum_size_bytes=external.maximum_size_bytes,
                outcome_access=external.outcome_access,
                visibility_ceiling=external.visibility_ceiling,
                barrier=reduction.barrier,
            )
        )
    external_inputs = tuple(
        sorted(
            (source_input, design_input, roster_input, *request_inputs),
            key=lambda value: value.input_id,
        )
    )
    return CandidateScientificGraph(
        graph_id=(f"graph.independent-substrate-grounding.freegsnke-{phase_roster.phase.value.lower()}-phase"),
        external_inputs=external_inputs,
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


class _FreeGsnkePhaseReductionRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        configs = tuple(
            decode_canonical_bytes(
                port.read(),
                FreeGsnkeTargetReductionConfig,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == FreeGsnkeTargetReductionConfig.SCHEMA
        )
        designs = tuple(
            decode_canonical_bytes(
                port.read(),
                FreeGsnkeActionDesign,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == FreeGsnkeActionDesign.SCHEMA
        )
        rosters = tuple(
            decode_canonical_bytes(
                port.read(),
                FreeGsnkePhaseRequestRoster,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == FreeGsnkePhaseRequestRoster.SCHEMA
        )
        requests = tuple(
            decode_canonical_bytes(
                port.read(),
                FreeGsnkeProcessRequest,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == FreeGsnkeProcessRequest.SCHEMA
        )
        responses = tuple(
            decode_canonical_bytes(
                port.read(),
                FreeGsnkeTargetProcessResponse,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == FreeGsnkeTargetProcessResponse.SCHEMA
        )
        if len(configs) != 1 or len(designs) != 1 or len(rosters) != 1:
            raise ValueError("FreeGSNKE reducer requires one config/design/roster")
        config = configs[0]
        if _reduction_key(config.phase) != self.manifest.capability_key:
            raise ValueError("FreeGSNKE reducer phase selects another capability")
        result = reduce_freegsnke_phase_panel(
            config=config,
            roster=rosters[0],
            action_design=designs[0],
            requests=requests,
            responses=responses,
        )
        if {value.payload_schema for value in context.output_ports} != {
            FreeGsnkePhaseReduction.SCHEMA
        }:
            raise ValueError("FreeGSNKE phase reduction output schema differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=result.canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("freegsnke-adverse-action-outcomes-preserved", True, ()),
                ReceiptCheck("freegsnke-issued-denominator-preserved", True, ()),
                ReceiptCheck("freegsnke-nested-views-not-units", True, ()),
                ReceiptCheck("freegsnke-panel-limitation-remains-evidence", True, ()),
            ),
        )


def freegsnke_phase_reduction_runners(
    *,
    registry: CapabilityRegistry,
    phases: tuple[FreeGsnkePhase, ...] = (
        FreeGsnkePhase.DEVELOPMENT,
        FreeGsnkePhase.EVALUATION,
    ),
) -> tuple[TaskRunner, ...]:
    """Bind the shared complete-preparation reducer for selected phase keys."""

    if not phases or len(set(phases)) != len(phases):
        raise ValueError("FreeGSNKE reduction runner phases must be unique")
    return tuple(
        _FreeGsnkePhaseReductionRunner(
            registry.resolve(_reduction_key(phase), FREEGSNKE_CAPABILITY_VERSION)
        )
        for phase in phases
    )


class FreeGsnkePhaseRuntimeProvider(CampaignRuntimeProvider):
    """Static provider for one exact phase fan-out/reduction graph."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        requests: tuple[FreeGsnkeProcessRequest, ...],
        reduction_config: FreeGsnkeTargetReductionConfig,
        action_design: FreeGsnkeActionDesign,
        phase_roster: FreeGsnkePhaseRequestRoster,
        source_binding: FreeGsnkeSourceBinding,
        source_factory: Callable[[], FreeGsnkeSourceBinding],
        executor: FreeGsnkeEpisodeExecutor,
    ) -> None:
        expected = freegsnke_phase_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("FreeGSNKE phase provider registry differs")
        if (
            not requests
            or {value.source_binding for value in requests} != {source_binding}
            or {value.phase for value in requests} != {reduction_config.phase}
        ):
            raise ValueError("FreeGSNKE phase provider request/source scope differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.requests = tuple(sorted(requests, key=lambda value: value.request_id))
        self.reduction_config = reduction_config
        self.action_design = action_design
        self.phase_roster = phase_roster
        self.source_binding = source_binding
        generation_runners = freegsnke_generation_runners(
            registry=registry,
            source_factory=source_factory,
            executor=executor,
        )
        reduction_runners = freegsnke_phase_reduction_runners(
            registry=registry,
        )
        self._runners = (*generation_runners, *reduction_runners)
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("FreeGSNKE phase provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("FreeGSNKE phase provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            FREEGSNKE_SOURCE_ARTIFACT_ID: self.source_binding,
            FREEGSNKE_PHASE_DESIGN_ARTIFACT_ID: self.action_design,
            FREEGSNKE_PHASE_ROSTER_ARTIFACT_ID: self.phase_roster,
            **{_request_artifact_id(value.request_id): value for value in self.requests},
        }
        configs_by_hash: dict[str, CanonicalRecord] = {
            value.fingerprint(): value for value in self.requests
        }
        configs_by_hash[self.reduction_config.fingerprint()] = self.reduction_config
        for task in plan.tasks:
            try:
                records[task.capability.config.artifact_id] = configs_by_hash[
                    task.capability.config.content_sha256
                ]
            except KeyError as error:
                raise ValueError("FreeGSNKE phase plan references an unknown config") from error
        if set(records) != set(specs):
            raise ValueError("FreeGSNKE phase external input roster differs")
        values = []
        for artifact_id, spec in sorted(specs.items()):
            record = records[artifact_id]
            record_id = next(
                getattr(record, attribute)
                for attribute in (
                    "request_id",
                    "config_id",
                    "binding_id",
                    "design_id",
                    "roster_id",
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
            raise ValueError("FreeGSNKE phase semantic registry differs")
        planned = None
        if execution_plan is not None:
            planned = {
                (task.capability.capability_key, output.payload_schema, output.profile)
                for task in execution_plan.tasks
                for output in task.outputs
            }
        record_types = {
            FreeGsnkeTargetProcessResponse.SCHEMA: FreeGsnkeTargetProcessResponse,
            FreeGsnkePhaseReduction.SCHEMA: FreeGsnkePhaseReduction,
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
        """Phase reduction remains sealed/development evidence, not adjudication."""

        if registry != self.registry:
            raise ValueError("FreeGSNKE phase adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "FREEGSNKE_DEVELOPMENT_REDUCTION_KEY",
    "FREEGSNKE_EVALUATION_REDUCTION_KEY",
    "FREEGSNKE_PHASE_DESIGN_ARTIFACT_ID",
    "FREEGSNKE_PHASE_PROVIDER_KEY",
    "FREEGSNKE_PHASE_ROSTER_ARTIFACT_ID",
    "FREEGSNKE_REDUCTION_STEP_ID",
    "FreeGsnkePhaseRuntimeProvider",
    "build_freegsnke_phase_protocol",
    "freegsnke_phase_candidate_registrations",
    "freegsnke_phase_reduction_runners",
    "freegsnke_phase_registry",
    "freegsnke_phase_scientific_graph",
]

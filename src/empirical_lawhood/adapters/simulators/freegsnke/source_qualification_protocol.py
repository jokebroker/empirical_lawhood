"""Compiled parallel FreeGSNKE source/action qualification DAG."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, replace
from hashlib import sha256
from typing import Final

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole, required_candidate_obligation_ids
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .contracts import FreeGsnkePhase, FreeGsnkeProcessRequest, FreeGsnkeSavedPreparation, FreeGsnkeSourceBinding, FreeGsnkeTargetProcessResponse
from .generation_protocol import (
    FREEGSNKE_SOURCE_ARTIFACT_ID,
    FreeGsnkeEpisodeExecutor,
    build_freegsnke_generation_protocol,
    freegsnke_generation_registry,
    freegsnke_generation_runners,
    freegsnke_generation_scientific_graph,
)
from .registry import (
    FREEGSNKE_CAPABILITY_VERSION,
    FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
    FREEGSNKE_EVALUATION_CAPABILITY_KEY,
)
from .source_qualification import FreeGsnkeSourceQualificationConfig, FreeGsnkeSourceQualificationDisposition, FreeGsnkeSourceQualificationResult, reduce_freegsnke_source_qualification


FREEGSNKE_SOURCE_QUALIFICATION_REDUCER_KEY: Final = "freegsnke.reduce-source-action-qualification"
FREEGSNKE_SOURCE_QUALIFICATION_EVALUATOR_KEY: Final = (
    "freegsnke.evaluate-source-action-qualification"
)
FREEGSNKE_SOURCE_QUALIFICATION_PROVIDER_KEY: Final = (
    "independent-substrate-grounding.freegsnke-source-qualification-provider"
)
FREEGSNKE_SOURCE_QUALIFICATION_STEP_ID: Final = "reduce-source-action-qualification"
FREEGSNKE_SOURCE_QUALIFICATION_CONFIG_ARTIFACT_ID: Final = (
    "artifact.independent-substrate-grounding.freegsnke.source-qualification-config"
)


def _request_artifact_id(request_id: str) -> str:
    return f"artifact.independent-substrate-grounding.freegsnke.source-qualification-request.{request_id}"


def _reducer_budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        # LocalProcessExecutor enforces address space, not resident bytes.  A
        # spawned evaluator imports the same SciPy/FreeGSNKE runtime as the
        # generation workers and maps about 2.4 GiB before decoding inputs.
        # Keep the evaluator aligned with that existing worker envelope.
        memory_bytes=4 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=5 * 60,
        source_scan_bytes=256 * 1024**2,
        output_bytes=64 * 1024**2,
    )


def freegsnke_source_qualification_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    generation = freegsnke_generation_registry(
        implementation_sha256=implementation_sha256
    ).capabilities
    reducer = CapabilityManifest(
        capability_key=FREEGSNKE_SOURCE_QUALIFICATION_REDUCER_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        kind=CapabilityKind.NUMERICAL_QUALIFIER,
        config_schema=FreeGsnkeSourceQualificationConfig.SCHEMA,
        config_schema_sha256=sha256(
            FreeGsnkeSourceQualificationConfig.SCHEMA.encode("ascii")
        ).hexdigest(),
        input_schema_ids=tuple(
            sorted(
                (
                    FreeGsnkeProcessRequest.SCHEMA,
                    FreeGsnkeSourceQualificationConfig.SCHEMA,
                    FreeGsnkeTargetProcessResponse.SCHEMA,
                )
            )
        ),
        output_schema_ids=(FreeGsnkeSourceQualificationResult.SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_DEVELOPMENT,
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resource_ceiling=_reducer_budget(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="independent-substrate-grounding-freegsnke-source-action-qualification-reducer",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "adverse-source-stops-retained",
            "all-issued-branches-accounted",
            "excluded-from-target-evidence",
            "operational-failure-not-scientific-negative",
        ),
        implementation_sha256=implementation_sha256,
    )
    evaluator = replace(
        reducer,
        capability_key=FREEGSNKE_SOURCE_QUALIFICATION_EVALUATOR_KEY,
        kind=CapabilityKind.EVALUATOR,
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.READ_SEALED_OUTCOMES,
                    CapabilityPermission.REVEAL_OUTCOMES,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        output_schema_ids=tuple(
            sorted((*reducer.output_schema_ids, ScientificAdjudicationRecord.SCHEMA))
        ),
        runtime_id="independent-substrate-grounding-freegsnke-source-action-qualification-evaluator",
        conformance_check_ids=tuple(
            sorted((*reducer.conformance_check_ids, "receipt-dependent-evaluator-reveal"))
        ),
    )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-freegsnke-source-action-qualification",
        capabilities=tuple(
            sorted((*generation, reducer, evaluator), key=lambda value: value.registry_id)
        ),
    )


def freegsnke_source_qualification_candidate_registrations(
    *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=FREEGSNKE_SOURCE_QUALIFICATION_PROVIDER_KEY,
            provider_version=FREEGSNKE_CAPABILITY_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=2 * 1024**2,
        )
        for manifest in freegsnke_source_qualification_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


class FreeGsnkeSourceQualificationConfigDecoder:
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        requests: tuple[FreeGsnkeProcessRequest, ...],
        qualification_config: FreeGsnkeSourceQualificationConfig,
    ) -> None:
        expected = freegsnke_source_qualification_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected or not requests:
            raise ValueError("FreeGSNKE source qualification decoder scope differs")
        self._requests = requests
        self._qualification_config = qualification_config

    @property
    def provider_key(self) -> str:
        return FREEGSNKE_SOURCE_QUALIFICATION_PROVIDER_KEY

    @property
    def provider_version(self) -> str:
        return FREEGSNKE_CAPABILITY_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema == FreeGsnkeProcessRequest.SCHEMA:
            observed: CanonicalRecord = decode_canonical_bytes(
                payload,
                FreeGsnkeProcessRequest,
                maximum_bytes=2 * 1024**2,
            )
            if observed not in self._requests:
                raise ValueError("FreeGSNKE source qualification request differs")
            return
        if expected_schema == FreeGsnkeSourceQualificationConfig.SCHEMA:
            observed = decode_canonical_bytes(
                payload,
                FreeGsnkeSourceQualificationConfig,
                maximum_bytes=2 * 1024**2,
            )
            if observed != self._qualification_config:
                raise ValueError("FreeGSNKE source qualification config differs")
            return
        raise ValueError("FreeGSNKE source qualification config schema differs")


def freegsnke_source_qualification_config_decoders(
    *,
    registry: CapabilityRegistry,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    qualification_config: FreeGsnkeSourceQualificationConfig,
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    return (
        FreeGsnkeSourceQualificationConfigDecoder(
            registry=registry,
            requests=requests,
            qualification_config=qualification_config,
        ),
    )


def build_freegsnke_source_qualification_protocol(
    *,
    registry: CapabilityRegistry,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    request_config_refs: tuple[CapabilityConfigRef, ...],
    qualification_config: FreeGsnkeSourceQualificationConfig,
    qualification_config_ref: CapabilityConfigRef,
) -> ProtocolTemplate:
    generation = build_freegsnke_generation_protocol(
        registry=CapabilityRegistry(
            registry_id="freegsnke-source-qualification-generation-subregistry",
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
    generation = replace(
        generation,
        steps=tuple(
            replace(
                step,
                obligation_ids=tuple(
                    f"{obligation}.{step.config.config_id}" for obligation in step.obligation_ids
                ),
            )
            for step in generation.steps
        ),
    )
    phases = {value.phase for value in requests}
    if len(phases) != 1 or not phases <= {
        FreeGsnkePhase.SCOUT,
        FreeGsnkePhase.EVALUATION,
    }:
        raise ValueError("FreeGSNKE source qualification requires one SCOUT or EVALUATION phase")
    phase = next(iter(phases))
    if phase is FreeGsnkePhase.EVALUATION:
        generation = replace(
            generation,
            steps=tuple(
                replace(
                    step,
                    barrier=BarrierKind.FREEZE,
                    obligation_ids=tuple(
                        sorted(
                            (
                                *step.obligation_ids,
                                f"freegsnke-issued-evaluation-request-frozen.{step.config.config_id}",
                            )
                        )
                    ),
                )
                for step in generation.steps
            ),
        )
    outcome_access = (
        OutcomeAccess.DEVELOPMENT_VISIBLE
        if phase is FreeGsnkePhase.SCOUT
        else OutcomeAccess.EVALUATOR_REVEAL
    )
    visibility_ceiling = (
        VisibilityCeiling.DEVELOPMENT_ONLY
        if phase is FreeGsnkePhase.SCOUT
        else VisibilityCeiling.PROSPECTIVE
    )
    reducer_manifest = registry.resolve(
        (
            FREEGSNKE_SOURCE_QUALIFICATION_REDUCER_KEY
            if phase is FreeGsnkePhase.SCOUT
            else FREEGSNKE_SOURCE_QUALIFICATION_EVALUATOR_KEY
        ),
        FREEGSNKE_CAPABILITY_VERSION,
    )
    if (
        qualification_config_ref.config_id != qualification_config.config_id
        or qualification_config_ref.config_schema != qualification_config.SCHEMA
        or qualification_config_ref.config_schema_sha256 != reducer_manifest.config_schema_sha256
        or qualification_config_ref.content_sha256 != qualification_config.fingerprint()
    ):
        raise ValueError("FreeGSNKE source qualification config reference differs")
    reducer = ProtocolStepTemplate(
        step_id=FREEGSNKE_SOURCE_QUALIFICATION_STEP_ID,
        stage=(
            ScientificStage.QUALIFY if phase is FreeGsnkePhase.SCOUT else ScientificStage.EVALUATE
        ),
        capability_key=reducer_manifest.capability_key,
        capability_version=reducer_manifest.capability_version,
        config=qualification_config_ref,
        dependency_step_ids=tuple(sorted(value.step_id for value in generation.steps)),
        outputs=tuple(
            sorted(
                (
                    OutputTemplate(
                        output_id=output_id,
                        payload_schema=payload_schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/vnd.empirical-lawhood.canonical+json",
                        filename_suffix=".json",
                    )
                    for output_id, payload_schema in (
                        (
                            "source-qualification-result",
                            FreeGsnkeSourceQualificationResult.SCHEMA,
                        ),
                        *(
                            (
                                (
                                    "scientific-adjudication",
                                    ScientificAdjudicationRecord.SCHEMA,
                                ),
                            )
                            if phase is FreeGsnkePhase.EVALUATION
                            else ()
                        ),
                    )
                ),
                key=lambda value: value.output_id,
            )
        ),
        required_permissions=tuple(
            sorted(
                (
                    *(
                        value
                        for value in reducer_manifest.permissions
                        if value
                        not in {
                            CapabilityPermission.READ_DEVELOPMENT,
                            CapabilityPermission.READ_SEALED_OUTCOMES,
                        }
                    ),
                    CapabilityPermission.READ_DEVELOPMENT
                    if phase is FreeGsnkePhase.SCOUT
                    else CapabilityPermission.READ_SEALED_OUTCOMES,
                )
            )
        ),
        requested_outcome_access=outcome_access,
        visibility_ceiling=visibility_ceiling,
        resource_budget=reducer_manifest.resource_ceiling,
        resource_lock_ids=("freegsnke-source-qualification-reducer",),
        barrier=(BarrierKind.NONE if phase is FreeGsnkePhase.SCOUT else BarrierKind.REVEAL),
        # A reducer retry is safe and idempotent: every scientific input is an
        # immutable, receipt-bound branch artifact and the output batch is
        # content addressed.  This absorbs a transient publication/finalization
        # failure without repeating simulator episodes or changing science.
        maximum_attempts=2,
        obligation_ids=(
            "freegsnke-all-issued-source-qualification-branches-accounted",
            "freegsnke-source-adverse-outcomes-retained",
            "freegsnke-source-qualification-excluded",
        ),
    )
    return ProtocolTemplate(
        template_id="independent-substrate-grounding-freegsnke-source-action-qualification-protocol",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        steps=tuple(sorted((*generation.steps, reducer), key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _request_input(request: FreeGsnkeProcessRequest) -> CandidateGraphExternalInput:
    return CandidateGraphExternalInput(
        input_id=f"input.independent-substrate-grounding.freegsnke.source-qualification.{request.request_id}",
        scientific_role=ScientificInputRole.ACTION,
        logical_artifact_id=_request_artifact_id(request.request_id),
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=request.fingerprint(),
        payload_schema=request.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=2 * 1024**2,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def freegsnke_source_qualification_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    source_binding: FreeGsnkeSourceBinding,
    saved_preparation: FreeGsnkeSavedPreparation,
) -> CandidateScientificGraph:
    reducer = next(
        value for value in protocol.steps if value.step_id == FREEGSNKE_SOURCE_QUALIFICATION_STEP_ID
    )
    generation_protocol = ProtocolTemplate(
        template_id="independent-substrate-grounding-freegsnke-source-qualification-generation-graph",
        template_version=protocol.template_version,
        steps=tuple(
            value
            for value in protocol.steps
            if value.step_id != FREEGSNKE_SOURCE_QUALIFICATION_STEP_ID
        ),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )
    generation_registry = CapabilityRegistry(
        registry_id="freegsnke-source-qualification-generation-graph-registry",
        capabilities=tuple(
            registry.resolve(key, FREEGSNKE_CAPABILITY_VERSION)
            for key in (
                FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
                FREEGSNKE_EVALUATION_CAPABILITY_KEY,
            )
        ),
    )
    base = freegsnke_generation_scientific_graph(
        protocol=generation_protocol,
        registry=generation_registry,
        requests=requests,
        source_binding=source_binding,
        saved_preparations=(saved_preparation,),
    )
    request_inputs = tuple(_request_input(value) for value in requests)
    request_input_by_id = {
        request.request_id: input_value
        for request, input_value in zip(requests, request_inputs, strict=True)
    }
    reducer_node = CandidateGraphNode(
        node_id=reducer.step_id,
        stage=reducer.stage,
        capability_key=reducer.capability_key,
        capability_version=reducer.capability_version,
        implementation_sha256=registry.resolve(
            reducer.capability_key,
            reducer.capability_version,
        ).implementation_sha256,
        protocol_step_sha256=reducer.fingerprint(),
        obligation_ids=reducer.obligation_ids,
        outcome_access=reducer.requested_outcome_access,
        visibility_ceiling=reducer.visibility_ceiling,
        resource_budget=reducer.resource_budget,
    )
    edges = list(base.edges)
    for step in generation_protocol.steps:
        request_id = step.config.config_id
        request_input = request_input_by_id[request_id]
        output = step.outputs[0]
        edges.extend(
            (
                CandidateGraphEdge(
                    edge_id=f"edge.source-qualification-request.{step.step_id}",
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
                ),
                CandidateGraphEdge(
                    edge_id=f"edge.{step.step_id}.source-qualification-result",
                    producer_node_id=step.step_id,
                    producer_output_id=output.output_id,
                    external_input_id=None,
                    consumer_node_id=reducer.step_id,
                    consumer_input_id=f"response.{request_id}",
                    scientific_role=ScientificInputRole.OUTCOME,
                    logical_artifact_id=f"artifact.{step.step_id}.{output.output_id}",
                    payload_schema=output.payload_schema,
                    media_type=output.media_type,
                    maximum_size_bytes=32 * 1024**2,
                    outcome_access=step.requested_outcome_access,
                    visibility_ceiling=step.visibility_ceiling,
                    barrier=reducer.barrier,
                ),
            )
        )
    return CandidateScientificGraph(
        graph_id="graph.independent-substrate-grounding.freegsnke-source-action-qualification",
        external_inputs=tuple(
            sorted((*base.external_inputs, *request_inputs), key=lambda value: value.input_id)
        ),
        nodes=tuple(sorted((*base.nodes, reducer_node), key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def freegsnke_source_qualification_study_template(
    *,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
    experiment: ExperimentSpec | None = None,
) -> StudyTemplate:
    owners = {
        obligation: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation in step.obligation_ids
    }
    fallback = (
        FREEGSNKE_SOURCE_QUALIFICATION_STEP_ID,
        "source-qualification-result",
    )
    obligation_ids = (
        tuple(sorted(owners))
        if experiment is None
        else required_candidate_obligation_ids(experiment, protocol)
    )
    bindings = tuple(
        ObligationCoverageBinding(
            obligation_id=obligation,
            proof_owner_node_id=owners.get(obligation, fallback)[0],
            required_output_id=owners.get(obligation, fallback)[1],
            contributor_edge_ids=tuple(
                sorted(
                    edge.edge_id
                    for edge in graph.edges
                    if edge.consumer_node_id == owners.get(obligation, fallback)[0]
                )
            ),
        )
        for obligation in obligation_ids
    )
    return StudyTemplate(
        template_key="independent-substrate-grounding.freegsnke.source-action-qualification",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id="coverage.independent-substrate-grounding.freegsnke-source-action-qualification",
            bindings=bindings,
        ),
    )


class _FreeGsnkeSourceQualificationRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest

    def execute(self, context: TaskContext) -> RunnerResult:
        configs = tuple(
            decode_canonical_bytes(
                port.read(),
                FreeGsnkeSourceQualificationConfig,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == FreeGsnkeSourceQualificationConfig.SCHEMA
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
        if len(configs) != 1:
            raise ValueError("FreeGSNKE source qualification reducer requires one config")
        result = reduce_freegsnke_source_qualification(
            config=configs[0],
            requests=requests,
            responses=responses,
        )
        scientific = None
        if result.outcome_access is OutcomeAccess.EVALUATION_REVEALED:
            adjudication_context = context.scientific_adjudication_context
            if adjudication_context is None:
                raise ValueError(
                    "FreeGSNKE source qualification evaluator lacks adjudication context"
                )
            output_ids = tuple(
                sorted(
                    value.logical_artifact_id
                    for value in context.output_ports
                    if value.logical_artifact_id is not None
                )
            )
            if len(output_ids) != len(context.output_ports):
                raise ValueError("FreeGSNKE qualification output lacks logical identity")
            unevaluable = (
                result.disposition
                is FreeGsnkeSourceQualificationDisposition.OPERATIONAL_UNEVALUABLE
            )
            scientific = ScientificAdjudicationRecord(
                adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
                run_id=context.run_id,
                adjudication_task_id=context.task_id,
                execution_plan=adjudication_context.execution_plan,
                input_materialization_ids=context.input_materialization_ids,
                output_logical_artifact_ids=output_ids,
                required_receipt_ids=context.dependency_receipt_ids,
                evidence_world_id=adjudication_context.evidence_world_id,
                evidence_world_kind=adjudication_context.evidence_world_kind,
                relation=adjudication_context.relation,
                independent_unit_id=adjudication_context.independent_unit_id,
                information_cutoffs=adjudication_context.information_cutoffs,
                visibility_ceiling=adjudication_context.visibility_ceiling,
                outcome_access=adjudication_context.outcome_access,
                evaluability=(
                    AdjudicationEvaluability.UNEVALUABLE
                    if unevaluable
                    else AdjudicationEvaluability.EVALUABLE
                ),
                scientific_status=(
                    ScientificStatus.UNEVALUABLE
                    if unevaluable
                    else ScientificStatus.SUPPORTED
                    if result.disposition is FreeGsnkeSourceQualificationDisposition.PASS
                    else ScientificStatus.NOT_SUPPORTED
                ),
                admission_status=(
                    AdmissionStatus.UNEVALUABLE if unevaluable else AdmissionStatus.NOT_EVALUATED
                ),
                reason_codes=tuple(
                    sorted(
                        {
                            "FREEGSNKE_SOURCE_QUALIFICATION_NONPROMOTABLE",
                            f"FREEGSNKE_SOURCE_QUALIFICATION_{result.disposition.value}",
                            *result.reason_codes,
                        }
                    )
                ),
            )
        outputs = []
        for port in context.output_ports:
            if port.payload_schema == FreeGsnkeSourceQualificationResult.SCHEMA:
                payload = result.canonical_bytes()
            elif (
                port.payload_schema == ScientificAdjudicationRecord.SCHEMA
                and scientific is not None
            ):
                payload = scientific.canonical_bytes()
            else:
                raise ValueError("FreeGSNKE qualification output schema differs")
            outputs.append(TaskOutputPayload(output_id=port.output_id, payload=payload))
        return RunnerResult(
            outputs=tuple(outputs),
            checks=tuple(
                sorted(
                    (
                        ReceiptCheck(
                            "freegsnke-all-issued-qualification-branches-accounted",
                            True,
                            (),
                        ),
                        ReceiptCheck("freegsnke-adverse-source-stops-retained", True, ()),
                        ReceiptCheck("freegsnke-source-qualification-excluded", True, ()),
                    ),
                    key=lambda value: value.check_id,
                )
            ),
        )


class FreeGsnkeSourceQualificationRuntimeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        requests: tuple[FreeGsnkeProcessRequest, ...],
        qualification_config: FreeGsnkeSourceQualificationConfig,
        source_binding: FreeGsnkeSourceBinding,
        saved_preparation: FreeGsnkeSavedPreparation,
        source_factory: Callable[[], FreeGsnkeSourceBinding],
        executor: FreeGsnkeEpisodeExecutor,
    ) -> None:
        expected = freegsnke_source_qualification_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected or not requests:
            raise ValueError("FreeGSNKE source qualification provider scope differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.requests = tuple(sorted(requests, key=lambda value: value.request_id))
        self.qualification_config = qualification_config
        self.source_binding = source_binding
        self.saved_preparation = saved_preparation
        self._runners = tuple(
            sorted(
                (
                    *freegsnke_generation_runners(
                        registry=registry,
                        source_factory=source_factory,
                        executor=executor,
                    ),
                    *(
                        _FreeGsnkeSourceQualificationRunner(
                            registry.resolve(key, FREEGSNKE_CAPABILITY_VERSION)
                        )
                        for key in (
                            FREEGSNKE_SOURCE_QUALIFICATION_REDUCER_KEY,
                            FREEGSNKE_SOURCE_QUALIFICATION_EVALUATOR_KEY,
                        )
                    ),
                ),
                key=lambda value: (
                    value.manifest.capability_key,
                    value.manifest.capability_version,
                ),
            )
        )
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("FreeGSNKE source qualification provider registry differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("FreeGSNKE source qualification plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            FREEGSNKE_SOURCE_ARTIFACT_ID: self.source_binding,
            self.saved_preparation.saved_state_id: self.saved_preparation,
            FREEGSNKE_SOURCE_QUALIFICATION_CONFIG_ARTIFACT_ID: self.qualification_config,
            **{_request_artifact_id(value.request_id): value for value in self.requests},
        }
        configs_by_hash: dict[str, CanonicalRecord] = {
            value.fingerprint(): value for value in self.requests
        }
        configs_by_hash[self.qualification_config.fingerprint()] = self.qualification_config
        for task in plan.tasks:
            try:
                records[task.capability.config.artifact_id] = configs_by_hash[
                    task.capability.config.content_sha256
                ]
            except KeyError as error:
                raise ValueError(
                    "FreeGSNKE source qualification plan references an unknown config"
                ) from error
        if set(records) != set(specs):
            raise ValueError("FreeGSNKE source qualification external input roster differs")
        values = []
        for artifact_id, spec in sorted(specs.items()):
            record = records[artifact_id]
            record_id = next(
                getattr(record, attribute)
                for attribute in (
                    "request_id",
                    "config_id",
                    "binding_id",
                    "saved_state_id",
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
            raise ValueError("FreeGSNKE source qualification semantic registry differs")
        planned = None
        if execution_plan is not None:
            planned = {
                (task.capability.capability_key, output.payload_schema, output.profile)
                for task in execution_plan.tasks
                for output in task.outputs
            }
        record_types = {
            FreeGsnkeTargetProcessResponse.SCHEMA: FreeGsnkeTargetProcessResponse,
            FreeGsnkeSourceQualificationResult.SCHEMA: (FreeGsnkeSourceQualificationResult),
            ScientificAdjudicationRecord.SCHEMA: ScientificAdjudicationRecord,
        }
        values = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                key = (manifest.capability_key, schema, ArtifactProfile.CANONICAL_JSON)
                if planned is not None and key not in planned:
                    continue
                values.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(
                            sorted(value.name for value in fields(record_types[schema]))
                        ),
                    )
                )
        return tuple(sorted(values, key=lambda value: value.key))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("FreeGSNKE source qualification adjudication registry differs")
        if execution_plan is not None and not any(
            value.capability.capability_key == FREEGSNKE_SOURCE_QUALIFICATION_EVALUATOR_KEY
            for value in execution_plan.tasks
        ):
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=FREEGSNKE_SOURCE_QUALIFICATION_EVALUATOR_KEY,
            capability_version=FREEGSNKE_CAPABILITY_VERSION,
            output_id=f"{FREEGSNKE_SOURCE_QUALIFICATION_STEP_ID}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
        )


__all__ = [
    "FREEGSNKE_SOURCE_QUALIFICATION_CONFIG_ARTIFACT_ID",
    "FREEGSNKE_SOURCE_QUALIFICATION_EVALUATOR_KEY",
    "FREEGSNKE_SOURCE_QUALIFICATION_PROVIDER_KEY",
    "FREEGSNKE_SOURCE_QUALIFICATION_REDUCER_KEY",
    "FREEGSNKE_SOURCE_QUALIFICATION_STEP_ID",
    "FreeGsnkeSourceQualificationConfigDecoder",
    "FreeGsnkeSourceQualificationRuntimeProvider",
    "build_freegsnke_source_qualification_protocol",
    "freegsnke_source_qualification_candidate_registrations",
    "freegsnke_source_qualification_config_decoders",
    'freegsnke_source_qualification_study_template',
    "freegsnke_source_qualification_registry",
    "freegsnke_source_qualification_scientific_graph",
]

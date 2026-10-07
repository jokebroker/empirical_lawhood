"""Static development/evaluation episode DAG for the FreeGSNKE follow-up.

The graph treats one prepared equilibrium/profile state as the independent
unit and every signed action branch as a nested view.  It is intentionally a
generation sub-DAG: target prediction, reveal and adjudication are composed by
the later target protocol and cannot be inferred from successful generation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
import subprocess
from typing import Protocol

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
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
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)

from .contracts import FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA, FreeGsnkePhase, FreeGsnkeProcessRequest, FreeGsnkeSavedPreparation, FreeGsnkeSourceBinding, FreeGsnkeTargetEpisodeStatus, FreeGsnkeTargetProcessResponse, FreeGsnkeTargetWorkerInput
from .registry import (
    FREEGSNKE_CAPABILITY_VERSION,
    FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
    FREEGSNKE_EVALUATION_CAPABILITY_KEY,
    build_freegsnke_registry,
)
from .runtime import (
    FreeGsnkeWorkerInvocation,
    decode_freegsnke_target_response,
    run_freegsnke_target_worker,
    target_worker_failure_response,
)
from .worker_process import FreeGsnkeProgressSink


FREEGSNKE_SOURCE_ARTIFACT_ID = "artifact.independent-substrate-grounding.freegsnke.source-binding"
FREEGSNKE_GENERATION_PROVIDER_KEY = "independent-substrate-grounding.freegsnke-generation-provider"
_GENERATION_KEYS = (
    FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
    FREEGSNKE_EVALUATION_CAPABILITY_KEY,
)


def freegsnke_generation_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    """Return only capabilities executable by this bounded generation provider."""

    full = build_freegsnke_registry(implementation_sha256=implementation_sha256)
    return CapabilityRegistry(
        registry_id="freegsnke-magnetic-response-generation-registry",
        capabilities=tuple(
            value for value in full.capabilities if value.capability_key in _GENERATION_KEYS
        ),
    )


def freegsnke_generation_candidate_registrations(
    *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=FREEGSNKE_GENERATION_PROVIDER_KEY,
            provider_version=FREEGSNKE_CAPABILITY_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=2 * 1024**2,
        )
        for manifest in freegsnke_generation_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


def _capability_key(phase: FreeGsnkePhase) -> str:
    if phase in {
        FreeGsnkePhase.MICROFIXTURE,
        FreeGsnkePhase.SCOUT,
        FreeGsnkePhase.DEVELOPMENT,
    }:
        return FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY
    return FREEGSNKE_EVALUATION_CAPABILITY_KEY


def _access(phase: FreeGsnkePhase) -> OutcomeAccess:
    return (
        OutcomeAccess.DEVELOPMENT_VISIBLE
        if _capability_key(phase) == FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY
        else OutcomeAccess.EVALUATION_SEALED
    )


def _visibility(phase: FreeGsnkePhase) -> VisibilityCeiling:
    return (
        VisibilityCeiling.DEVELOPMENT_ONLY
        if _access(phase) is OutcomeAccess.DEVELOPMENT_VISIBLE
        else VisibilityCeiling.PROSPECTIVE
    )


def _step_id(request: FreeGsnkeProcessRequest) -> str:
    return f"generate.{request.request_id}"


def build_freegsnke_generation_protocol(
    *,
    registry: CapabilityRegistry,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    config_refs: tuple[CapabilityConfigRef, ...],
) -> ProtocolTemplate:
    """Compile an exact request roster without treating branches as units."""

    if not requests:
        raise ValueError("FreeGSNKE generation requires a nonempty request roster")
    by_request_id = {value.request_id: value for value in requests}
    refs_by_id = {value.config_id: value for value in config_refs}
    if len(by_request_id) != len(requests) or len(refs_by_id) != len(config_refs):
        raise ValueError("FreeGSNKE request/config identities must be unique")
    if set(by_request_id) != set(refs_by_id) or len(requests) != len(config_refs):
        raise ValueError("FreeGSNKE request/config roster differs")
    source_identities = {
        ObjectIdentity.from_record(value.source_binding.binding_id, value.source_binding)
        for value in requests
    }
    if len(source_identities) != 1:
        raise ValueError("one FreeGSNKE generation act requires one exact source binding")
    preparation_phases: dict[str, FreeGsnkePhase] = {}
    branch_ids: set[tuple[str, str]] = set()
    steps = []
    for request in sorted(requests, key=lambda value: value.request_id):
        existing_phase = preparation_phases.setdefault(
            request.preparation.preparation_id,
            request.phase,
        )
        if existing_phase is not request.phase:
            raise ValueError("one FreeGSNKE preparation cannot cross phase partitions")
        branch_key = (request.preparation.preparation_id, request.branch_id)
        if branch_key in branch_ids:
            raise ValueError("FreeGSNKE branches must be unique within a preparation")
        branch_ids.add(branch_key)
        ref = refs_by_id[request.request_id]
        key = _capability_key(request.phase)
        manifest = registry.resolve(key, FREEGSNKE_CAPABILITY_VERSION)
        if (
            ref.config_schema != request.SCHEMA
            or ref.config_schema_sha256 != manifest.config_schema_sha256
            or ref.content_sha256 != request.fingerprint()
        ):
            raise ValueError("FreeGSNKE request config reference differs")
        steps.append(
            ProtocolStepTemplate(
                step_id=_step_id(request),
                stage=ScientificStage.ACQUIRE,
                capability_key=key,
                capability_version=FREEGSNKE_CAPABILITY_VERSION,
                config=ref,
                dependency_step_ids=(),
                outputs=(
                    OutputTemplate(
                        output_id=f"response.{request.request_id}",
                        payload_schema=FreeGsnkeTargetProcessResponse.SCHEMA,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/vnd.empirical-lawhood.canonical+json",
                        filename_suffix=".json",
                    ),
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=_access(request.phase),
                visibility_ceiling=_visibility(request.phase),
                resource_budget=manifest.resource_ceiling,
                resource_lock_ids=(),
                barrier=BarrierKind.NONE,
                maximum_attempts=1,
                obligation_ids=(
                    "freegsnke-branch-nested-under-preparation",
                    "freegsnke-exact-request-response-binding",
                    "freegsnke-request-accept-apply-realize-clocks",
                    "freegsnke-task-local-bounded-scratch",
                ),
            )
        )
    return ProtocolTemplate(
        template_id="independent-substrate-grounding-freegsnke-generation-protocol",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def freegsnke_generation_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    source_binding: FreeGsnkeSourceBinding,
    saved_preparations: tuple[FreeGsnkeSavedPreparation, ...],
) -> CandidateScientificGraph:
    requests_by_id = {value.request_id: value for value in requests}
    saved_by_preparation = {value.preparation_id: value for value in saved_preparations}
    expected_preparation_ids = {
        value.preparation.preparation_id
        for value in requests
        if value.preparation.saved_state is not None
        and value.preparation.saved_state.payload_schema
        == FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA
    }
    if (
        len(requests_by_id) != len(requests)
        or len(saved_by_preparation) != len(saved_preparations)
        or set(saved_by_preparation) != expected_preparation_ids
    ):
        raise ValueError("FreeGSNKE generation saved-preparation roster differs")
    for request in requests:
        if request.preparation.preparation_id not in expected_preparation_ids:
            continue
        FreeGsnkeTargetWorkerInput(
            input_id=f"worker-input.{request.request_id}",
            request=request,
            saved_preparation=saved_by_preparation[request.preparation.preparation_id],
        )
    source_input = CandidateGraphExternalInput(
        input_id="input.independent-substrate-grounding.freegsnke.source-binding",
        scientific_role=ScientificInputRole.PREPARED_MEDIUM,
        logical_artifact_id=FREEGSNKE_SOURCE_ARTIFACT_ID,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=source_binding.fingerprint(),
        payload_schema=FreeGsnkeSourceBinding.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=2 * 1024**2,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    saved_inputs = tuple(
        CandidateGraphExternalInput(
            input_id=(f"input.independent-substrate-grounding.freegsnke.saved-preparation.{value.preparation_id}"),
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id=value.saved_state_id,
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=value.fingerprint(),
            payload_schema=value.SCHEMA,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_size_bytes=4 * 1024**2,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )
        for value in sorted(saved_preparations, key=lambda item: item.preparation_id)
    )
    saved_input_by_preparation = {
        value.preparation_id: input_value
        for value, input_value in zip(
            sorted(saved_preparations, key=lambda item: item.preparation_id),
            saved_inputs,
            strict=True,
        )
    }
    nodes = []
    edges = []
    for step in protocol.steps:
        manifest = registry.resolve(step.capability_key, step.capability_version)
        nodes.append(
            CandidateGraphNode(
                node_id=step.step_id,
                stage=step.stage,
                capability_key=step.capability_key,
                capability_version=step.capability_version,
                implementation_sha256=manifest.implementation_sha256,
                protocol_step_sha256=step.fingerprint(),
                obligation_ids=step.obligation_ids,
                outcome_access=step.requested_outcome_access,
                visibility_ceiling=step.visibility_ceiling,
                resource_budget=step.resource_budget,
            )
        )
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
        request_id = step.config.config_id
        try:
            request = requests_by_id[request_id]
        except KeyError as error:
            raise ValueError("FreeGSNKE generation step has no exact request") from error
        saved_input = saved_input_by_preparation.get(request.preparation.preparation_id)
        if saved_input is not None:
            edges.append(
                CandidateGraphEdge(
                    edge_id=f"edge.freegsnke-saved.{step.step_id}",
                    producer_node_id=None,
                    producer_output_id=None,
                    external_input_id=saved_input.input_id,
                    consumer_node_id=step.step_id,
                    consumer_input_id=saved_input.input_id,
                    scientific_role=saved_input.scientific_role,
                    logical_artifact_id=saved_input.logical_artifact_id,
                    payload_schema=saved_input.payload_schema,
                    media_type=saved_input.media_type,
                    maximum_size_bytes=saved_input.maximum_size_bytes,
                    outcome_access=saved_input.outcome_access,
                    visibility_ceiling=saved_input.visibility_ceiling,
                    barrier=step.barrier,
                )
            )
    return CandidateScientificGraph(
        graph_id="graph.independent-substrate-grounding.freegsnke-generation",
        external_inputs=tuple(
            sorted((source_input, *saved_inputs), key=lambda value: value.input_id)
        ),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


class FreeGsnkeEpisodeExecutor(Protocol):
    def execute(
        self,
        request: FreeGsnkeProcessRequest,
        saved_preparation: FreeGsnkeSavedPreparation | None = None,
    ) -> FreeGsnkeTargetProcessResponse: ...


class FreeGsnkeProcessExecutor:
    """Production executor composed from one trusted installed worker invocation."""

    def __init__(
        self,
        invocation: FreeGsnkeWorkerInvocation,
        *,
        progress_sink: FreeGsnkeProgressSink | None = None,
    ) -> None:
        if invocation.environment.get("OPENBLAS_NUM_THREADS") != "1":
            raise ValueError("FreeGSNKE target executor requires one BLAS thread")
        self.invocation = invocation
        self.progress_sink = progress_sink

    def execute(
        self,
        request: FreeGsnkeProcessRequest,
        saved_preparation: FreeGsnkeSavedPreparation | None = None,
    ) -> FreeGsnkeTargetProcessResponse:
        if saved_preparation is None:
            raise ValueError("FreeGSNKE target execution lacks saved preparation bytes")
        try:
            response, _stderr = run_freegsnke_target_worker(
                request=request,
                saved_preparation=saved_preparation,
                invocation=self.invocation,
                progress_sink=self.progress_sink,
            )
        except subprocess.TimeoutExpired:
            return target_worker_failure_response(
                request=request,
                status=FreeGsnkeTargetEpisodeStatus.WORKER_TIMED_OUT,
                error_code="freegsnke-worker-timeout",
            )
        except RuntimeError:
            return target_worker_failure_response(
                request=request,
                status=FreeGsnkeTargetEpisodeStatus.WORKER_FAILED,
                error_code="freegsnke-worker-nonzero-exit",
            )
        return response


class _FreeGsnkeGenerationRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        *,
        source_factory: Callable[[], FreeGsnkeSourceBinding],
        executor: FreeGsnkeEpisodeExecutor,
    ) -> None:
        self.manifest = manifest
        self.source_factory = source_factory
        self.executor = executor
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        requests = tuple(
            decode_canonical_bytes(
                port.read(),
                FreeGsnkeProcessRequest,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == FreeGsnkeProcessRequest.SCHEMA
        )
        sources = tuple(
            decode_canonical_bytes(
                port.read(),
                FreeGsnkeSourceBinding,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == FreeGsnkeSourceBinding.SCHEMA
        )
        saved_preparations = tuple(
            decode_canonical_bytes(
                port.read(),
                FreeGsnkeSavedPreparation,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == FreeGsnkeSavedPreparation.SCHEMA
        )
        if len(requests) != 1 or len(sources) != 1 or len(saved_preparations) > 1:
            raise ValueError(
                "FreeGSNKE generation requires one request/source and at most one saved preparation"
            )
        request = requests[0]
        target_saved = (
            request.preparation.saved_state is not None
            and request.preparation.saved_state.payload_schema
            == FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA
        )
        if target_saved != bool(saved_preparations):
            raise ValueError("FreeGSNKE generation saved-preparation input differs")
        if _capability_key(request.phase) != self.manifest.capability_key:
            raise ValueError("FreeGSNKE phase selects another generation capability")
        live_source = self.source_factory()
        if live_source != sources[0] or request.source_binding != sources[0]:
            raise ValueError("FreeGSNKE live/request source differs from frozen input")
        response = self.executor.execute(
            request,
            saved_preparations[0] if saved_preparations else None,
        )
        response = decode_freegsnke_target_response(
            request=request,
            payload=response.canonical_bytes(),
        )
        if {value.payload_schema for value in context.output_ports} != {
            FreeGsnkeTargetProcessResponse.SCHEMA
        }:
            raise ValueError("FreeGSNKE generation output schema differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=response.canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("freegsnke-action-clock-ledger-complete", True, ()),
                ReceiptCheck("freegsnke-independent-preparation-preserved", True, ()),
                ReceiptCheck("freegsnke-request-response-binding-exact", True, ()),
            ),
        )


def freegsnke_generation_runners(
    *,
    registry: CapabilityRegistry,
    source_factory: Callable[[], FreeGsnkeSourceBinding],
    executor: FreeGsnkeEpisodeExecutor,
) -> tuple[TaskRunner, ...]:
    """Bind generation runners from either the generation-only or phase registry."""

    return tuple(
        _FreeGsnkeGenerationRunner(
            registry.resolve(key, FREEGSNKE_CAPABILITY_VERSION),
            source_factory=source_factory,
            executor=executor,
        )
        for key in _GENERATION_KEYS
    )


class FreeGsnkeGenerationRuntimeProvider(CampaignRuntimeProvider):
    """Path-free provider for the bounded generation sub-DAG."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        requests: tuple[FreeGsnkeProcessRequest, ...],
        source_binding: FreeGsnkeSourceBinding,
        saved_preparations: tuple[FreeGsnkeSavedPreparation, ...],
        source_factory: Callable[[], FreeGsnkeSourceBinding],
        executor: FreeGsnkeEpisodeExecutor,
    ) -> None:
        implementation_sha256 = registry.capabilities[0].implementation_sha256
        expected = freegsnke_generation_registry(implementation_sha256=implementation_sha256)
        if registry != expected:
            raise ValueError("FreeGSNKE generation provider registry differs")
        if not requests or any(value.source_binding != source_binding for value in requests):
            raise ValueError("FreeGSNKE generation requests differ from source binding")
        if len({value.request_id for value in requests}) != len(requests):
            raise ValueError("FreeGSNKE generation request identities repeat")
        saved_by_preparation = {value.preparation_id: value for value in saved_preparations}
        expected_preparation_ids = {
            value.preparation.preparation_id
            for value in requests
            if value.preparation.saved_state is not None
            and value.preparation.saved_state.payload_schema
            == FREEGSNKE_TARGET_SAVED_PREPARATION_SCHEMA
        }
        if (
            len(saved_by_preparation) != len(saved_preparations)
            or set(saved_by_preparation) != expected_preparation_ids
        ):
            raise ValueError("FreeGSNKE generation saved-preparation roster differs")
        for request in requests:
            if request.preparation.preparation_id not in expected_preparation_ids:
                continue
            FreeGsnkeTargetWorkerInput(
                input_id=f"worker-input.{request.request_id}",
                request=request,
                saved_preparation=saved_by_preparation[request.preparation.preparation_id],
            )
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.requests = tuple(sorted(requests, key=lambda value: value.request_id))
        self.source_binding = source_binding
        self.saved_preparations = tuple(
            sorted(saved_preparations, key=lambda value: value.preparation_id)
        )
        self._runners = freegsnke_generation_runners(
            registry=registry,
            source_factory=source_factory,
            executor=executor,
        )
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("FreeGSNKE generation provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("FreeGSNKE generation provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        configs = {
            task.capability.config.artifact_id: task.capability.config for task in plan.tasks
        }
        requests_by_hash = {value.fingerprint(): value for value in self.requests}
        saved_by_artifact = {value.saved_state_id: value for value in self.saved_preparations}
        values = []
        for artifact_id, spec in sorted(specs.items()):
            if artifact_id == FREEGSNKE_SOURCE_ARTIFACT_ID:
                record: CanonicalRecord = self.source_binding
                record_id = self.source_binding.binding_id
            elif artifact_id in saved_by_artifact:
                record = saved_by_artifact[artifact_id]
                record_id = record.saved_state_id
            else:
                try:
                    config = configs[artifact_id]
                    record = requests_by_hash[config.content_sha256]
                except KeyError as error:
                    raise ValueError(
                        "FreeGSNKE generation plan references an unknown input"
                    ) from error
                record_id = record.request_id
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
            raise ValueError("FreeGSNKE generation semantic registry differs")
        planned = None
        if execution_plan is not None:
            planned = {
                (task.capability.capability_key, output.payload_schema, output.profile)
                for task in execution_plan.tasks
                for output in task.outputs
            }
        values = tuple(
            CapabilityOutputSemanticContract.from_manifest(
                manifest,
                payload_schema=FreeGsnkeTargetProcessResponse.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(
                    sorted(value.name for value in fields(FreeGsnkeTargetProcessResponse))
                ),
            )
            for manifest in registry.capabilities
            if planned is None
            or (
                manifest.capability_key,
                FreeGsnkeTargetProcessResponse.SCHEMA,
                ArtifactProfile.CANONICAL_JSON,
            )
            in planned
        )
        return tuple(sorted(values, key=lambda value: value.key))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        """Generation is never itself a target-level scientific adjudication."""

        if registry != self.registry:
            raise ValueError("FreeGSNKE generation adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "FREEGSNKE_GENERATION_PROVIDER_KEY",
    "FREEGSNKE_SOURCE_ARTIFACT_ID",
    "FreeGsnkeEpisodeExecutor",
    "FreeGsnkeGenerationRuntimeProvider",
    "FreeGsnkeProcessExecutor",
    "build_freegsnke_generation_protocol",
    "freegsnke_generation_candidate_registrations",
    "freegsnke_generation_registry",
    "freegsnke_generation_runners",
    "freegsnke_generation_scientific_graph",
]

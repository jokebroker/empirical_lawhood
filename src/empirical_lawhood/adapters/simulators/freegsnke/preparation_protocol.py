"""Registered excluded-qualification DAG for FreeGSNKE saved preparations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields
from hashlib import sha256
from typing import Final, Protocol

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
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
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .contracts import FreeGsnkeSourceBinding
from .preparation_runtime import run_freegsnke_preparation_worker
from .preparation import FreeGsnkePreparationWorkerInput, FreeGsnkePreparedUnit
from .runtime import FreeGsnkeWorkerInvocation
from .worker_process import FreeGsnkeProgressSink


FREEGSNKE_PREPARATION_CAPABILITY_KEY: Final = "freegsnke.qualify-preparation"
FREEGSNKE_PREPARATION_CAPABILITY_VERSION: Final = "1.0.0"
FREEGSNKE_PREPARATION_PROVIDER_KEY: Final = "independent-substrate-grounding.freegsnke-preparation-qualification-provider"
FREEGSNKE_PREPARATION_SOURCE_INPUT_ID: Final = "input.independent-substrate-grounding.freegsnke.preparation-source"
FREEGSNKE_PREPARATION_SOURCE_ARTIFACT_ID: Final = "artifact.independent-substrate-grounding.freegsnke.preparation-source"


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=24 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=6 * 60 * 60,
        source_scan_bytes=2 * 1024**3,
        output_bytes=64 * 1024**2,
    )


def freegsnke_preparation_registry(
    *,
    implementation_sha256: str,
) -> CapabilityRegistry:
    """Return the one non-promotable saved-preparation capability."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    manifest = CapabilityManifest(
        capability_key=FREEGSNKE_PREPARATION_CAPABILITY_KEY,
        capability_version=FREEGSNKE_PREPARATION_CAPABILITY_VERSION,
        kind=CapabilityKind.SIMULATOR,
        config_schema=FreeGsnkePreparationWorkerInput.SCHEMA,
        config_schema_sha256=sha256(
            FreeGsnkePreparationWorkerInput.SCHEMA.encode("ascii")
        ).hexdigest(),
        input_schema_ids=tuple(
            sorted(
                (
                    FreeGsnkePreparationWorkerInput.SCHEMA,
                    FreeGsnkeSourceBinding.SCHEMA,
                )
            )
        ),
        output_schema_ids=(FreeGsnkePreparedUnit.SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        resource_ceiling=_budget(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="freegsnke-3.0.1-saved-preparation-qualification",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "excluded-qualification-not-target-evidence",
            "one-content-addressed-state-per-preparation",
            "preparation-local-action-chart",
            "solver-topology-and-source-limit-stops-retained",
            "target-outcome-blind",
        ),
        implementation_sha256=implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-freegsnke-preparation-qualification",
        capabilities=(manifest,),
    )


def freegsnke_preparation_candidate_registrations(
    *,
    implementation_sha256: str,
) -> tuple[CandidateCapabilityRegistration, ...]:
    manifest = freegsnke_preparation_registry(
        implementation_sha256=implementation_sha256
    ).capabilities[0]
    return (
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=FREEGSNKE_PREPARATION_PROVIDER_KEY,
            provider_version=FREEGSNKE_PREPARATION_CAPABILITY_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=4 * 1024**2,
        ),
    )


class FreeGsnkePreparationConfigDecoder:
    """Accept only the finite preparation configs authored before execution."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        configs: tuple[FreeGsnkePreparationWorkerInput, ...],
    ) -> None:
        if not configs or len({value.input_id for value in configs}) != len(configs):
            raise ValueError("FreeGSNKE preparation decoder config roster differs")
        expected = freegsnke_preparation_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("FreeGSNKE preparation decoder registry differs")
        self._configs = configs

    @property
    def provider_key(self) -> str:
        return FREEGSNKE_PREPARATION_PROVIDER_KEY

    @property
    def provider_version(self) -> str:
        return FREEGSNKE_PREPARATION_CAPABILITY_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != FreeGsnkePreparationWorkerInput.SCHEMA:
            raise ValueError("FreeGSNKE preparation decoder schema differs")
        observed = decode_canonical_bytes(
            payload,
            FreeGsnkePreparationWorkerInput,
            maximum_bytes=4 * 1024**2,
        )
        if observed not in self._configs:
            raise ValueError("FreeGSNKE preparation config differs from authoring")


def freegsnke_preparation_config_decoders(
    *,
    registry: CapabilityRegistry,
    configs: tuple[FreeGsnkePreparationWorkerInput, ...],
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    return (
        FreeGsnkePreparationConfigDecoder(
            registry=registry,
            configs=configs,
        ),
    )


def _step_id(value: FreeGsnkePreparationWorkerInput) -> str:
    return f"qualify-preparation.{value.recipe.preparation_id}"


def build_freegsnke_preparation_protocol(
    *,
    registry: CapabilityRegistry,
    configs: tuple[FreeGsnkePreparationWorkerInput, ...],
    config_refs: tuple[CapabilityConfigRef, ...],
) -> ProtocolTemplate:
    """Build one independent qualification task per prospective preparation."""

    manifest = registry.resolve(
        FREEGSNKE_PREPARATION_CAPABILITY_KEY,
        FREEGSNKE_PREPARATION_CAPABILITY_VERSION,
    )
    by_id = {value.input_id: value for value in configs}
    refs = {value.config_id: value for value in config_refs}
    if (
        not configs
        or len(by_id) != len(configs)
        or len(refs) != len(config_refs)
        or set(by_id) != set(refs)
        or len({value.recipe.preparation_id for value in configs}) != len(configs)
        or len({value.source_binding for value in configs}) != 1
    ):
        raise ValueError("FreeGSNKE preparation protocol config roster differs")
    steps = []
    for config in sorted(configs, key=lambda value: value.input_id):
        ref = refs[config.input_id]
        if (
            ref.config_schema != config.SCHEMA
            or ref.config_schema_sha256 != manifest.config_schema_sha256
            or ref.content_sha256 != config.fingerprint()
        ):
            raise ValueError("FreeGSNKE preparation config reference differs")
        steps.append(
            ProtocolStepTemplate(
                step_id=_step_id(config),
                stage=ScientificStage.QUALIFY,
                capability_key=manifest.capability_key,
                capability_version=manifest.capability_version,
                config=ref,
                dependency_step_ids=(),
                outputs=(
                    OutputTemplate(
                        output_id=f"prepared-unit.{config.recipe.preparation_id}",
                        payload_schema=FreeGsnkePreparedUnit.SCHEMA,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/vnd.empirical-lawhood.canonical+json",
                        filename_suffix=".json",
                    ),
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                resource_budget=manifest.resource_ceiling,
                resource_lock_ids=(),
                barrier=BarrierKind.FREEZE,
                maximum_attempts=1,
                obligation_ids=(
                    f"freegsnke-preparation-excluded.{config.recipe.preparation_id}",
                    f"freegsnke-preparation-state-exact.{config.recipe.preparation_id}",
                    f"freegsnke-preparation-stop-retained.{config.recipe.preparation_id}",
                    f"freegsnke-task-local-bounded-scratch.{config.recipe.preparation_id}",
                ),
            )
        )
    return ProtocolTemplate(
        template_id="independent-substrate-grounding-freegsnke-preparation-qualification-protocol",
        template_version=FREEGSNKE_PREPARATION_CAPABILITY_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def freegsnke_preparation_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    source_binding: FreeGsnkeSourceBinding,
) -> CandidateScientificGraph:
    source = CandidateGraphExternalInput(
        input_id=FREEGSNKE_PREPARATION_SOURCE_INPUT_ID,
        scientific_role=ScientificInputRole.PREPARED_MEDIUM,
        logical_artifact_id=FREEGSNKE_PREPARATION_SOURCE_ARTIFACT_ID,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=source_binding.fingerprint(),
        payload_schema=source_binding.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=2 * 1024**2,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
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
                edge_id=f"edge.freegsnke-preparation-source.{step.step_id}",
                producer_node_id=None,
                producer_output_id=None,
                external_input_id=source.input_id,
                consumer_node_id=step.step_id,
                consumer_input_id=source.input_id,
                scientific_role=source.scientific_role,
                logical_artifact_id=source.logical_artifact_id,
                payload_schema=source.payload_schema,
                media_type=source.media_type,
                maximum_size_bytes=source.maximum_size_bytes,
                outcome_access=source.outcome_access,
                visibility_ceiling=source.visibility_ceiling,
                barrier=step.barrier,
            )
        )
    return CandidateScientificGraph(
        graph_id="graph.independent-substrate-grounding.freegsnke-preparation-qualification",
        external_inputs=(source,),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def freegsnke_preparation_study_template(
    *,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> StudyTemplate:
    bindings = tuple(
        sorted(
            (
                ObligationCoverageBinding(
                    obligation_id=obligation,
                    proof_owner_node_id=step.step_id,
                    required_output_id=step.outputs[0].output_id,
                    contributor_edge_ids=tuple(
                        sorted(
                            edge.edge_id
                            for edge in graph.edges
                            if edge.consumer_node_id == step.step_id
                        )
                    ),
                )
                for step in protocol.steps
                for obligation in step.obligation_ids
            ),
            key=lambda value: value.obligation_id,
        )
    )
    return StudyTemplate(
        template_key="independent-substrate-grounding.freegsnke.preparation-qualification",
        template_version=FREEGSNKE_PREPARATION_CAPABILITY_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id="coverage.independent-substrate-grounding.freegsnke-preparation-qualification",
            bindings=bindings,
        ),
    )


class FreeGsnkePreparationExecutor(Protocol):
    def execute(
        self,
        worker_input: FreeGsnkePreparationWorkerInput,
    ) -> FreeGsnkePreparedUnit: ...


class FreeGsnkePreparationProcessExecutor:
    def __init__(
        self,
        invocation: FreeGsnkeWorkerInvocation,
        *,
        progress_sink: FreeGsnkeProgressSink | None = None,
    ) -> None:
        if invocation.environment.get("OPENBLAS_NUM_THREADS") != "1":
            raise ValueError("FreeGSNKE preparation executor requires one BLAS thread")
        self.invocation = invocation
        self.progress_sink = progress_sink

    def execute(
        self,
        worker_input: FreeGsnkePreparationWorkerInput,
    ) -> FreeGsnkePreparedUnit:
        result, _stderr = run_freegsnke_preparation_worker(
            worker_input=worker_input,
            invocation=self.invocation,
            progress_sink=self.progress_sink,
        )
        return result


class _FreeGsnkePreparationRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        *,
        source_factory: Callable[[], FreeGsnkeSourceBinding],
        executor: FreeGsnkePreparationExecutor,
    ) -> None:
        self.manifest = manifest
        self.source_factory = source_factory
        self.executor = executor
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        configs = tuple(
            decode_canonical_bytes(
                port.read(),
                FreeGsnkePreparationWorkerInput,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == FreeGsnkePreparationWorkerInput.SCHEMA
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
        if len(configs) != 1 or len(sources) != 1:
            raise ValueError("FreeGSNKE preparation task requires one config and source")
        config = configs[0]
        if (
            config.source_binding != sources[0]
            or self.source_factory() != sources[0]
            or context.config.content_sha256 != config.fingerprint()
        ):
            raise ValueError("FreeGSNKE preparation live/frozen source differs")
        result = self.executor.execute(config)
        if {value.payload_schema for value in context.output_ports} != {
            FreeGsnkePreparedUnit.SCHEMA
        }:
            raise ValueError("FreeGSNKE preparation output schema differs")
        typed_disposition = result.eligible != bool(result.reason_codes)
        checks = (
            ReceiptCheck("freegsnke-preparation-excluded", True, ()),
            ReceiptCheck("freegsnke-preparation-target-outcome-blind", True, ()),
            ReceiptCheck(
                "freegsnke-preparation-typed-disposition",
                typed_disposition,
                () if typed_disposition else ("MALFORMED_PREPARATION_DISPOSITION",),
            ),
        )
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=result.canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=checks,
        )


class FreeGsnkePreparationRuntimeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        configs: tuple[FreeGsnkePreparationWorkerInput, ...],
        source_binding: FreeGsnkeSourceBinding,
        source_factory: Callable[[], FreeGsnkeSourceBinding],
        executor: FreeGsnkePreparationExecutor,
    ) -> None:
        expected = freegsnke_preparation_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("FreeGSNKE preparation provider registry differs")
        if (
            not configs
            or len({value.input_id for value in configs}) != len(configs)
            or {value.source_binding for value in configs} != {source_binding}
        ):
            raise ValueError("FreeGSNKE preparation provider config/source differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.configs = tuple(sorted(configs, key=lambda value: value.input_id))
        self.source_binding = source_binding
        self._runner = _FreeGsnkePreparationRunner(
            registry.capabilities[0],
            source_factory=source_factory,
            executor=executor,
        )
        self.capability_count = 1

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("FreeGSNKE preparation provider registry/source differs")
        return (self._runner,)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("FreeGSNKE preparation provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        config_refs = {
            task.capability.config.artifact_id: task.capability.config for task in plan.tasks
        }
        configs_by_hash = {value.fingerprint(): value for value in self.configs}
        values = []
        for artifact_id, spec in sorted(specs.items()):
            ref = config_refs.get(artifact_id)
            if ref is not None:
                try:
                    config_record = configs_by_hash[ref.content_sha256]
                except KeyError as error:
                    raise ValueError(
                        "FreeGSNKE preparation plan references an unknown config"
                    ) from error
                record: CanonicalRecord = config_record
                record_id = config_record.input_id
            elif artifact_id == FREEGSNKE_PREPARATION_SOURCE_ARTIFACT_ID:
                record = self.source_binding
                record_id = self.source_binding.binding_id
            else:
                raise ValueError(
                    f"FreeGSNKE preparation plan references an unknown input: {artifact_id}"
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
            raise ValueError("FreeGSNKE preparation semantic registry differs")
        del execution_plan
        return (
            CapabilityOutputSemanticContract.from_manifest(
                registry.capabilities[0],
                payload_schema=FreeGsnkePreparedUnit.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(value.name for value in fields(FreeGsnkePreparedUnit))),
            ),
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("FreeGSNKE preparation adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "FREEGSNKE_PREPARATION_CAPABILITY_KEY",
    "FREEGSNKE_PREPARATION_CAPABILITY_VERSION",
    "FREEGSNKE_PREPARATION_PROVIDER_KEY",
    "FREEGSNKE_PREPARATION_SOURCE_ARTIFACT_ID",
    "FreeGsnkePreparationConfigDecoder",
    "FreeGsnkePreparationExecutor",
    "FreeGsnkePreparationProcessExecutor",
    "FreeGsnkePreparationRuntimeProvider",
    "build_freegsnke_preparation_protocol",
    "freegsnke_preparation_candidate_registrations",
    "freegsnke_preparation_config_decoders",
    'freegsnke_preparation_study_template',
    "freegsnke_preparation_registry",
    "freegsnke_preparation_scientific_graph",
]

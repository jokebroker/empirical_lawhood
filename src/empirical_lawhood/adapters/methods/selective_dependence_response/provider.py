"""Static no-target selective dependence response method provider and scientific DAG."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
    WorkerInputKind,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificInputRole, ScientificStage
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .conformance import SelectiveDependenceResponseMethodConformanceReport, execute_truth_known_conformance
from .contracts import SelectiveDependenceResponseMethodQuestionFreeze
from .method import method_question_freeze


SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION = "1.0.0"
SELECTIVE_DEPENDENCE_RESPONSE_FREEZE_QUESTION_KEY = "method.selective-dependence-response.freeze-question"
SELECTIVE_DEPENDENCE_RESPONSE_VERIFY_CONFORMANCE_KEY = "method.selective-dependence-response.verify-conformance"
SELECTIVE_DEPENDENCE_RESPONSE_PROVIDER_KEY = "selective-dependence-response.no-target-method-provider"


class SelectiveDependenceResponseMethodOperation(StrEnum):
    FREEZE_QUESTION = "FREEZE_QUESTION"
    VERIFY_CONFORMANCE = "VERIFY_CONFORMANCE"


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseMethodRuntimeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-method-runtime-config'

    config_id: str
    operation: SelectiveDependenceResponseMethodOperation
    capability_key: str
    capability_version: str
    method_question_sha256: str
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(self.method_question_sha256, field_name="method_question_sha256")
        expected = {
            SelectiveDependenceResponseMethodOperation.FREEZE_QUESTION: SELECTIVE_DEPENDENCE_RESPONSE_FREEZE_QUESTION_KEY,
            SelectiveDependenceResponseMethodOperation.VERIFY_CONFORMANCE: SELECTIVE_DEPENDENCE_RESPONSE_VERIFY_CONFORMANCE_KEY,
        }[self.operation]
        if self.capability_key != expected:
            raise ValueError("selective dependence response method operation and capability differ")
        if self.capability_version != SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION:
            raise ValueError("selective dependence response method capability version differs")
        if not 0 < self.maximum_input_bytes <= 1024 * 1024:
            raise ValueError("selective dependence response method input bound differs")


def decode_method_runtime_config(payload: bytes) -> SelectiveDependenceResponseMethodRuntimeConfig:
    return decode_canonical_bytes(payload, SelectiveDependenceResponseMethodRuntimeConfig, maximum_bytes=64 * 1024)


def method_runtime_config(operation: SelectiveDependenceResponseMethodOperation) -> SelectiveDependenceResponseMethodRuntimeConfig:
    question = method_question_freeze()
    key = {
        SelectiveDependenceResponseMethodOperation.FREEZE_QUESTION: SELECTIVE_DEPENDENCE_RESPONSE_FREEZE_QUESTION_KEY,
        SelectiveDependenceResponseMethodOperation.VERIFY_CONFORMANCE: SELECTIVE_DEPENDENCE_RESPONSE_VERIFY_CONFORMANCE_KEY,
    }[operation]
    return SelectiveDependenceResponseMethodRuntimeConfig(
        config_id=f"selective-dependence-response.{operation.value.lower().replace('_', '-')}.config",
        operation=operation,
        capability_key=key,
        capability_version=SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION,
        method_question_sha256=question.fingerprint(),
        maximum_input_bytes=1024 * 1024,
    )


def _resources() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=128 * 1024**2,
        gpu_devices=0,
        wall_time_seconds=10,
        source_scan_bytes=1024 * 1024,
        output_bytes=128 * 1024,
    )


def selective_dependence_response_method_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    definitions = (
        (
            SELECTIVE_DEPENDENCE_RESPONSE_FREEZE_QUESTION_KEY,
            CapabilityKind.TRANSFORM,
            (SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA,),
            (SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA,),
            ("question-freeze-outcome-blind",),
        ),
        (
            SELECTIVE_DEPENDENCE_RESPONSE_VERIFY_CONFORMANCE_KEY,
            CapabilityKind.NUMERICAL_QUALIFIER,
            (SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA,),
            (SelectiveDependenceResponseMethodConformanceReport.SCHEMA,),
            ("truth-known-conformance", "zero-target-contact"),
        ),
    )
    manifests = tuple(
        CapabilityManifest(
            capability_key=key,
            capability_version=SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION,
            kind=kind,
            config_schema=SelectiveDependenceResponseMethodRuntimeConfig.SCHEMA,
            config_schema_sha256=sha256(
                SelectiveDependenceResponseMethodRuntimeConfig.SCHEMA.encode("ascii")
            ).hexdigest(),
            input_schema_ids=input_schemas,
            output_schema_ids=output_schemas,
            permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
            maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            maximum_outcome_access=(
                OutcomeAccess.PRIVILEGED_TRUTH
                if key == SELECTIVE_DEPENDENCE_RESPONSE_VERIFY_CONFORMANCE_KEY
                else OutcomeAccess.OUTCOME_BLIND
            ),
            resource_ceiling=_resources(),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11-selective-dependence-response-method",
            requires_clean_commit=True,
            requires_active_mount=False,
            requires_network=False,
            conformance_check_ids=tuple(sorted(check_ids)),
            implementation_sha256=implementation_sha256,
        )
        for key, kind, input_schemas, output_schemas, check_ids in definitions
    )
    return CapabilityRegistry(
        registry_id="selective-dependence-response-method-runtime",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


def selective_dependence_response_method_registrations(
    *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=SELECTIVE_DEPENDENCE_RESPONSE_PROVIDER_KEY,
            provider_version=SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=64 * 1024,
        )
        for manifest in selective_dependence_response_method_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


def build_method_protocol(
    *,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
) -> ProtocolTemplate:
    specs = (
        (
            "freeze-method-question",
            SELECTIVE_DEPENDENCE_RESPONSE_FREEZE_QUESTION_KEY,
            ScientificStage.FREEZE,
            (),
            SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA,
            BarrierKind.FREEZE,
        ),
        (
            "verify-method-conformance",
            SELECTIVE_DEPENDENCE_RESPONSE_VERIFY_CONFORMANCE_KEY,
            ScientificStage.QUALIFY,
            ("freeze-method-question",),
            SelectiveDependenceResponseMethodConformanceReport.SCHEMA,
            BarrierKind.NONE,
        ),
    )
    if set(config_by_step_id) != {value[0] for value in specs}:
        raise ValueError("selective dependence response method config roster differs")
    steps = []
    for step_id, key, stage, dependencies, output_schema, barrier in specs:
        manifest = registry.resolve(key, SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION)
        config = config_by_step_id[step_id]
        if (
            config.config_schema != manifest.config_schema
            or config.config_schema_sha256 != manifest.config_schema_sha256
        ):
            raise ValueError("selective dependence response method config differs from its capability")
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=stage,
                capability_key=key,
                capability_version=SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION,
                config=config,
                dependency_step_ids=dependencies,
                outputs=(
                    OutputTemplate(
                        output_id=f"{step_id}.record",
                        payload_schema=output_schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/vnd.empirical-lawhood.canonical+json",
                        filename_suffix=".json",
                    ),
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=(
                    OutcomeAccess.PRIVILEGED_TRUTH
                    if key == SELECTIVE_DEPENDENCE_RESPONSE_VERIFY_CONFORMANCE_KEY
                    else OutcomeAccess.OUTCOME_BLIND
                ),
                visibility_ceiling=(
                    VisibilityCeiling.PRIVILEGED_TRUTH
                    if key == SELECTIVE_DEPENDENCE_RESPONSE_VERIFY_CONFORMANCE_KEY
                    else VisibilityCeiling.PROSPECTIVE
                ),
                resource_budget=ResourceBudget(
                    cpu_cores=1,
                    memory_bytes=1_000_000,
                    gpu_devices=0,
                    wall_time_seconds=2,
                    source_scan_bytes=(3000 if step_id == "freeze-method-question" else 6000),
                    output_bytes=6000,
                ),
                resource_lock_ids=(f"selective-dependence-response-{step_id}",),
                barrier=barrier,
                maximum_attempts=2,
                obligation_ids=(f"selective-dependence-response-{step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id="selective-dependence-response-method-protocol",
        template_version=SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def method_scientific_graph(
    *, protocol: ProtocolTemplate, registry: CapabilityRegistry
) -> CandidateScientificGraph:
    nodes = tuple(
        sorted(
            (
                CandidateGraphNode(
                    node_id=step.step_id,
                    stage=step.stage,
                    capability_key=step.capability_key,
                    capability_version=step.capability_version,
                    implementation_sha256=registry.resolve(
                        step.capability_key, step.capability_version
                    ).implementation_sha256,
                    protocol_step_sha256=step.fingerprint(),
                    obligation_ids=step.obligation_ids,
                    outcome_access=step.requested_outcome_access,
                    visibility_ceiling=step.visibility_ceiling,
                    resource_budget=step.resource_budget,
                )
                for step in protocol.steps
            ),
            key=lambda value: value.node_id,
        )
    )
    producer = next(value for value in protocol.steps if value.step_id == "freeze-method-question")
    consumer = next(
        value for value in protocol.steps if value.step_id == "verify-method-conformance"
    )
    output = producer.outputs[0]
    question = method_question_freeze()
    external = CandidateGraphExternalInput(
        input_id="input.method-question",
        scientific_role=ScientificInputRole.MODEL,
        logical_artifact_id=question.freeze_id,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=question.fingerprint(),
        payload_schema=question.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=64 * 1024,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    edges = (
        CandidateGraphEdge(
            edge_id="edge.input.method-question.freeze-method-question",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=external.input_id,
            consumer_node_id=producer.step_id,
            consumer_input_id="method-question-model",
            scientific_role=ScientificInputRole.MODEL,
            logical_artifact_id=external.logical_artifact_id,
            payload_schema=external.payload_schema,
            media_type=external.media_type,
            maximum_size_bytes=external.maximum_size_bytes,
            outcome_access=external.outcome_access,
            visibility_ceiling=external.visibility_ceiling,
            barrier=BarrierKind.FREEZE,
        ),
        CandidateGraphEdge(
            edge_id="edge.freeze-method-question.verify-method-conformance",
            producer_node_id=producer.step_id,
            producer_output_id=output.output_id,
            external_input_id=None,
            consumer_node_id=consumer.step_id,
            consumer_input_id="method-question",
            scientific_role=ScientificInputRole.QUALIFICATION,
            logical_artifact_id="artifact.selective-dependence-response.method-question",
            payload_schema=output.payload_schema,
            media_type=output.media_type,
            maximum_size_bytes=64 * 1024,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.FREEZE,
        ),
    )
    return CandidateScientificGraph(
        graph_id="graph.selective-dependence-response.no-target-method",
        external_inputs=(external,),
        nodes=nodes,
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def _study_template(
    *, protocol: ProtocolTemplate, graph: CandidateScientificGraph
) -> StudyTemplate:
    incoming = {
        node.node_id: tuple(
            edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id
        )
        for node in graph.nodes
    }
    bindings = tuple(
        sorted(
            (
                ObligationCoverageBinding(
                    obligation_id=obligation_id,
                    proof_owner_node_id=step.step_id,
                    required_output_id=step.outputs[0].output_id,
                    contributor_edge_ids=tuple(sorted(incoming[step.step_id])),
                )
                for step in protocol.steps
                for obligation_id in step.obligation_ids
            ),
            key=lambda value: value.obligation_id,
        )
    )
    return StudyTemplate(
        template_key="selective-dependence-response.no-target-method",
        template_version=SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id="coverage.selective-dependence-response.no-target-method", bindings=bindings
        ),
    )


def selective_dependence_response_method_catalog(
    *, protocol: ProtocolTemplate, registry: CapabilityRegistry
) -> CandidateCapabilityCatalog:
    graph = method_scientific_graph(protocol=protocol, registry=registry)
    implementations = {value.implementation_sha256 for value in registry.capabilities}
    if len(implementations) != 1:
        raise ValueError("selective dependence response method implementation identities differ")
    return CandidateCapabilityCatalog(
        catalog_id="selective-dependence-response-method-candidate-catalog",
        registrations=selective_dependence_response_method_registrations(
            implementation_sha256=next(iter(implementations))
        ),
        templates=(_study_template(protocol=protocol, graph=graph),),
    )


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseMethodConfigAdapter:
    @property
    def provider_key(self) -> str:
        return SELECTIVE_DEPENDENCE_RESPONSE_PROVIDER_KEY

    @property
    def provider_version(self) -> str:
        return SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != SelectiveDependenceResponseMethodRuntimeConfig.SCHEMA:
            raise ValueError("selective dependence response method config schema differs")
        decode_method_runtime_config(payload)


def selective_dependence_response_method_config_decoders(
    catalog: CandidateCapabilityCatalog,
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    providers = {
        f"{value.provider_key}@{value.provider_version}" for value in catalog.registrations
    }
    if providers != {f"{SELECTIVE_DEPENDENCE_RESPONSE_PROVIDER_KEY}@{SELECTIVE_DEPENDENCE_RESPONSE_PROTOCOL_VERSION}"}:
        raise ValueError("selective dependence response method provider roster differs")
    return (SelectiveDependenceResponseMethodConfigAdapter(),)


class _QuestionRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        question_inputs = tuple(
            value
            for value in context.input_ports
            if value.payload_schema == SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA
        )
        if (
            len(context.input_ports) != 2
            or len(question_inputs) != 1
            or question_inputs[0].kind is not WorkerInputKind.EXTERNAL
            or len(context.output_ports) != 1
        ):
            raise ValueError("method-question runner port roster differs")
        record = decode_canonical_bytes(
            question_inputs[0].read(1024 * 1024),
            SelectiveDependenceResponseMethodQuestionFreeze,
            maximum_bytes=1024 * 1024,
        )
        if record != method_question_freeze():
            raise ValueError("method-question model differs from the frozen method")
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=record.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck("selective-dependence-response-question-freeze", True, ()),),
        )


class _ConformanceRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        dependencies = tuple(
            value for value in context.input_ports if value.kind is WorkerInputKind.DEPENDENCY
        )
        if (
            len(context.input_ports) != 2
            or len(dependencies) != 1
            or len(context.output_ports) != 1
        ):
            raise ValueError("method-conformance runner port roster differs")
        question = decode_canonical_bytes(
            dependencies[0].read(1024 * 1024),
            SelectiveDependenceResponseMethodQuestionFreeze,
            maximum_bytes=1024 * 1024,
        )
        report = execute_truth_known_conformance(question)
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=report.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck("selective-dependence-response-truth-known-conformance", report.all_passed, ()),),
        )


class SelectiveDependenceResponseMethodRuntimeProvider(CampaignRuntimeProvider):
    def __init__(self, *, registry: CapabilityRegistry) -> None:
        expected = selective_dependence_response_method_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("selective dependence response method provider registry differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        runners: list[TaskRunner] = []
        for manifest in registry.capabilities:
            if manifest.capability_key == SELECTIVE_DEPENDENCE_RESPONSE_FREEZE_QUESTION_KEY:
                runners.append(_QuestionRunner(manifest))
            else:
                runners.append(_ConformanceRunner(manifest))
        self._runners = tuple(runners)
        self.capability_count = len(runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("selective dependence response method provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("selective dependence response method provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            task.capability.config.artifact_id: method_runtime_config(
                SelectiveDependenceResponseMethodOperation(
                    task.capability.capability_key.rsplit(".", 1)[-1].replace("-", "_").upper()
                )
            )
            for task in plan.tasks
        }
        question = method_question_freeze()
        records[question.freeze_id] = question
        if set(records) != set(specs):
            raise ValueError("selective dependence response method external input roster differs")
        values = []
        for artifact_id, record in sorted(records.items()):
            spec = specs[artifact_id]
            if isinstance(record, SelectiveDependenceResponseMethodQuestionFreeze):
                record_id = record.freeze_id
            elif isinstance(record, SelectiveDependenceResponseMethodRuntimeConfig):
                record_id = record.config_id
            else:
                raise TypeError("method external input type differs")
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
            raise ValueError("selective dependence response method semantic registry differs")
        types = {
            SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA: SelectiveDependenceResponseMethodQuestionFreeze,
            SelectiveDependenceResponseMethodConformanceReport.SCHEMA: SelectiveDependenceResponseMethodConformanceReport,
        }
        values = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                record_type = types[schema]
                values.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(sorted(field.name for field in fields(record_type))),
                    )
                )
        return tuple(sorted(values, key=lambda value: (value.capability_key, value.payload_schema)))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("selective dependence response method adjudication registry differs")
        return None


__all__ = [
    'SelectiveDependenceResponseMethodOperation',
    'SelectiveDependenceResponseMethodRuntimeConfig',
    'SelectiveDependenceResponseMethodRuntimeProvider',
    "build_method_protocol",
    "decode_method_runtime_config",
    'selective_dependence_response_method_catalog',
    'selective_dependence_response_method_config_decoders',
    'selective_dependence_response_method_registry',
    "method_runtime_config",
]

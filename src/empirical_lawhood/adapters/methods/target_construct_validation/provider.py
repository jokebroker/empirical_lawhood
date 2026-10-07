"""Static no-science construct nomination-authoring boundary capability, DAG and runtime provider."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphNode, CandidateScientificGraph, ObligationCoverage, ObligationCoverageBinding, StudyTemplate
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
)
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage, ScientificInputRole
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .authoring import TargetConstructValidationNominationAuthoringManifest
from .conformance import TargetConstructValidationConformanceReport
from .contracts import TargetConstructValidationMethodQuestionFreeze


TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION = "1.0.0"
TARGET_CONSTRUCT_VALIDATION_CONFORMANCE_KEY = "method.target-construct-validation.verify-conformance"
TARGET_CONSTRUCT_VALIDATION_NOMINATION_FREEZE_KEY = "method.target-construct-validation.freeze-nomination-authoring"


class TargetConstructValidationNoScienceOperation(StrEnum):
    VERIFY_CONFORMANCE = "VERIFY_CONFORMANCE"
    FREEZE_NOMINATION_AUTHORING = "FREEZE_NOMINATION_AUTHORING"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationNoScienceRuntimeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-no-science-runtime-config'

    config_id: str
    operation: TargetConstructValidationNoScienceOperation
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
            TargetConstructValidationNoScienceOperation.VERIFY_CONFORMANCE: TARGET_CONSTRUCT_VALIDATION_CONFORMANCE_KEY,
            TargetConstructValidationNoScienceOperation.FREEZE_NOMINATION_AUTHORING: (TARGET_CONSTRUCT_VALIDATION_NOMINATION_FREEZE_KEY),
        }[self.operation]
        if self.capability_key != expected:
            raise ValueError("target construct validation operation and capability differ")
        if self.capability_version != TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION:
            raise ValueError("target construct validation capability version differs")
        if not 0 < self.maximum_input_bytes <= 1024 * 1024:
            raise ValueError("target construct validation no-science input bound differs")


def decode_no_science_config(payload: bytes) -> TargetConstructValidationNoScienceRuntimeConfig:
    return decode_canonical_bytes(
        payload,
        TargetConstructValidationNoScienceRuntimeConfig,
        maximum_bytes=64 * 1024,
    )


def no_science_runtime_config(
    operation: TargetConstructValidationNoScienceOperation,
    *,
    method_question_sha256: str,
) -> TargetConstructValidationNoScienceRuntimeConfig:
    key = {
        TargetConstructValidationNoScienceOperation.VERIFY_CONFORMANCE: TARGET_CONSTRUCT_VALIDATION_CONFORMANCE_KEY,
        TargetConstructValidationNoScienceOperation.FREEZE_NOMINATION_AUTHORING: (TARGET_CONSTRUCT_VALIDATION_NOMINATION_FREEZE_KEY),
    }[operation]
    return TargetConstructValidationNoScienceRuntimeConfig(
        config_id=f"target-construct-validation.{operation.value.lower().replace('_', '-')}.config",
        operation=operation,
        capability_key=key,
        capability_version=TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION,
        method_question_sha256=method_question_sha256,
        maximum_input_bytes=1024 * 1024,
    )


def _resources() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=128 * 1024**2,
        gpu_devices=0,
        wall_time_seconds=5,
        source_scan_bytes=1024 * 1024,
        output_bytes=128 * 1024,
    )


def target_construct_validation_no_science_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    definitions = (
        (
            TARGET_CONSTRUCT_VALIDATION_CONFORMANCE_KEY,
            TargetConstructValidationNoScienceOperation.VERIFY_CONFORMANCE,
            (TargetConstructValidationMethodQuestionFreeze.SCHEMA,),
            (TargetConstructValidationConformanceReport.SCHEMA,),
            ScientificStage.QUALIFY,
        ),
        (
            TARGET_CONSTRUCT_VALIDATION_NOMINATION_FREEZE_KEY,
            TargetConstructValidationNoScienceOperation.FREEZE_NOMINATION_AUTHORING,
            (TargetConstructValidationConformanceReport.SCHEMA,),
            (TargetConstructValidationNominationAuthoringManifest.SCHEMA,),
            ScientificStage.FREEZE,
        ),
    )
    manifests = tuple(
        CapabilityManifest(
            capability_key=key,
            capability_version=TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION,
            kind=CapabilityKind.ANALYSIS,
            config_schema=TargetConstructValidationNoScienceRuntimeConfig.SCHEMA,
            config_schema_sha256=sha256(
                TargetConstructValidationNoScienceRuntimeConfig.SCHEMA.encode("ascii")
            ).hexdigest(),
            input_schema_ids=input_schemas,
            output_schema_ids=output_schemas,
            permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
            maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
            resource_ceiling=_resources(),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11-target-construct-validation-no-science",
            requires_clean_commit=True,
            requires_active_mount=False,
            requires_network=False,
            conformance_check_ids=tuple(
                sorted(
                    (
                        f"{operation.value.lower().replace('_', '-')}-truth-known",
                        f"stage-{stage.value.lower().replace('_', '-')}",
                    )
                )
            ),
            implementation_sha256=implementation_sha256,
        )
        for key, operation, input_schemas, output_schemas, stage in definitions
    )
    return CapabilityRegistry(
        registry_id="target-construct-validation-no-science-runtime",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


def target_construct_validation_candidate_registrations(
    *,
    implementation_sha256: str,
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key="target-construct-validation.no-science-runtime-provider",
            provider_version=TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=64 * 1024,
        )
        for manifest in target_construct_validation_no_science_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


def build_no_science_protocol(
    *,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
) -> ProtocolTemplate:
    step_specs = (
        (
            "verify-common-method",
            TARGET_CONSTRUCT_VALIDATION_CONFORMANCE_KEY,
            ScientificStage.QUALIFY,
            (),
            TargetConstructValidationConformanceReport.SCHEMA,
            BarrierKind.NONE,
        ),
        (
            "freeze-nomination-authoring",
            TARGET_CONSTRUCT_VALIDATION_NOMINATION_FREEZE_KEY,
            ScientificStage.FREEZE,
            ("verify-common-method",),
            TargetConstructValidationNominationAuthoringManifest.SCHEMA,
            BarrierKind.FREEZE,
        ),
    )
    if set(config_by_step_id) != {value[0] for value in step_specs}:
        raise ValueError("target construct validation no-science config roster differs")
    steps = []
    for step_id, key, stage, dependencies, output_schema, barrier in step_specs:
        manifest = registry.resolve(key, TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION)
        config = config_by_step_id[step_id]
        if (
            config.config_schema != manifest.config_schema
            or config.config_schema_sha256 != manifest.config_schema_sha256
        ):
            raise ValueError("target construct validation no-science config differs from capability schema")
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=stage,
                capability_key=key,
                capability_version=TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION,
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
                requested_outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                resource_budget=ResourceBudget(
                    cpu_cores=1,
                    memory_bytes=1_000_000,
                    gpu_devices=0,
                    wall_time_seconds=2,
                    source_scan_bytes=(3000 if step_id == "verify-common-method" else 6000),
                    output_bytes=6000,
                ),
                resource_lock_ids=(f"target-construct-validation-{step_id}",),
                barrier=barrier,
                maximum_attempts=2,
                obligation_ids=(f"target-construct-validation-{step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id="target-construct-validation-no-science-protocol",
        template_version=TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def no_science_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateScientificGraph:
    """Construct the exact two-node candidate graph from the frozen protocol."""

    expected_steps = {"verify-common-method", "freeze-nomination-authoring"}
    if {step.step_id for step in protocol.steps} != expected_steps:
        raise ValueError("target construct validation no-science protocol step roster differs")
    nodes = tuple(
        sorted(
            (
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
            ),
            key=lambda value: value.node_id,
        )
    )
    producer = next(value for value in protocol.steps if value.step_id == "verify-common-method")
    consumer = next(
        value for value in protocol.steps if value.step_id == "freeze-nomination-authoring"
    )
    output = producer.outputs[0]
    edge = CandidateGraphEdge(
        edge_id="edge.verify-common-method.freeze-nomination-authoring",
        producer_node_id=producer.step_id,
        producer_output_id=output.output_id,
        external_input_id=None,
        consumer_node_id=consumer.step_id,
        consumer_input_id="conformance-report",
        scientific_role=ScientificInputRole.QUALIFICATION,
        logical_artifact_id="artifact.target-construct-validation.common-method-conformance",
        payload_schema=output.payload_schema,
        media_type=output.media_type,
        maximum_size_bytes=6000,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        barrier=BarrierKind.FREEZE,
    )
    return CandidateScientificGraph(
        graph_id="graph.target-construct-validation.no-science-method",
        external_inputs=(),
        nodes=nodes,
        edges=(edge,),
    )


def plumbing_study_template(
    *,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> StudyTemplate:
    incoming = {
        node.node_id: tuple(
            edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id
        )
        for node in graph.nodes
    }
    bindings = []
    for step in protocol.steps:
        for obligation_id in step.obligation_ids:
            bindings.append(
                ObligationCoverageBinding(
                    obligation_id=obligation_id,
                    proof_owner_node_id=step.step_id,
                    required_output_id=step.outputs[0].output_id,
                    contributor_edge_ids=tuple(sorted(incoming[step.step_id])),
                )
            )
    return StudyTemplate(
        template_key="target-construct-validation.no-science-method",
        template_version=TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id="coverage.target-construct-validation.no-science-method",
            bindings=tuple(sorted(bindings, key=lambda value: value.obligation_id)),
        ),
    )


def target_construct_validation_candidate_catalog(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
) -> CandidateCapabilityCatalog:
    graph = no_science_scientific_graph(protocol=protocol, registry=registry)
    template = plumbing_study_template(protocol=protocol, graph=graph)
    implementation_sha256s = {value.implementation_sha256 for value in registry.capabilities}
    if len(implementation_sha256s) != 1:
        raise ValueError("target construct validation no-science implementation identities differ")
    implementation_sha256 = next(iter(implementation_sha256s))
    return CandidateCapabilityCatalog(
        catalog_id="target-construct-validation-no-science-candidate-catalog",
        registrations=target_construct_validation_candidate_registrations(implementation_sha256=implementation_sha256),
        templates=(template,),
    )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCandidateConfigAdapter:
    """One strict decoder for the shared static no-science provider."""

    @property
    def provider_key(self) -> str:
        return "target-construct-validation.no-science-runtime-provider"

    @property
    def provider_version(self) -> str:
        return TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != TargetConstructValidationNoScienceRuntimeConfig.SCHEMA:
            raise ValueError("target construct validation config adapter schema differs")
        decode_no_science_config(payload)


def target_construct_validation_config_decoders(
    catalog: CandidateCapabilityCatalog,
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    provider_ids = {
        f"{value.provider_key}@{value.provider_version}" for value in catalog.registrations
    }
    if provider_ids != {f"target-construct-validation.no-science-runtime-provider@{TARGET_CONSTRUCT_VALIDATION_PROTOCOL_VERSION}"}:
        raise ValueError("target construct validation candidate catalog provider roster differs")
    return (TargetConstructValidationCandidateConfigAdapter(),)


class _TargetConstructValidationPlumbingRunner:
    def __init__(self, manifest: CapabilityManifest, record: CanonicalRecord) -> None:
        self.manifest = manifest
        self.record = record
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        if len(context.output_ports) != 1:
            raise ValueError("target construct validation no-science task must have one output")
        if context.output_ports[0].payload_schema != self.record.SCHEMA:
            raise ValueError("target construct validation no-science output schema differs")
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=self.record.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck("target-construct-validation-no-science-runtime-contract", True, ()),),
        )


class TargetConstructValidationNoScienceRuntimeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        conformance_report: TargetConstructValidationConformanceReport,
        authoring_manifest: TargetConstructValidationNominationAuthoringManifest,
    ) -> None:
        expected = target_construct_validation_no_science_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("target construct validation provider registry differs")
        records = {
            TARGET_CONSTRUCT_VALIDATION_CONFORMANCE_KEY: conformance_report,
            TARGET_CONSTRUCT_VALIDATION_NOMINATION_FREEZE_KEY: authoring_manifest,
        }
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self._runners = tuple(
            _TargetConstructValidationPlumbingRunner(manifest, records[manifest.capability_key])
            for manifest in registry.capabilities
        )
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("target construct validation provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("target construct validation provider plan/source differs")
        return ()

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("target construct validation provider semantic registry differs")
        types = {
            TargetConstructValidationConformanceReport.SCHEMA: TargetConstructValidationConformanceReport,
            TargetConstructValidationNominationAuthoringManifest.SCHEMA: (TargetConstructValidationNominationAuthoringManifest),
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
            raise ValueError("target construct validation provider adjudication registry differs")
        return None

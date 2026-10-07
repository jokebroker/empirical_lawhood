"""Static outcome-blind nomination nomination DAG and runtime provider."""

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
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificInputRole, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .nomination import TargetConstructValidationNominationFreeze, TargetConstructValidationNominationResult


TARGET_CONSTRUCT_VALIDATION_NOMINATION_VERSION = "1.0.0"
TARGET_CONSTRUCT_VALIDATION_VALIDATE_NOMINATION_KEY = "method.target-construct-validation.validate-nomination"
TARGET_CONSTRUCT_VALIDATION_RANK_TARGETS_KEY = "method.target-construct-validation.rank-targets"


class TargetConstructValidationNominationOperation(StrEnum):
    VALIDATE_FREEZE = "VALIDATE_FREEZE"
    RANK_TARGETS = "RANK_TARGETS"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationNominationRuntimeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-nomination-runtime-config'

    config_id: str
    operation: TargetConstructValidationNominationOperation
    capability_key: str
    capability_version: str
    nomination_freeze_sha256: str
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(
            self.nomination_freeze_sha256,
            field_name="nomination_freeze_sha256",
        )
        expected = {
            TargetConstructValidationNominationOperation.VALIDATE_FREEZE: TARGET_CONSTRUCT_VALIDATION_VALIDATE_NOMINATION_KEY,
            TargetConstructValidationNominationOperation.RANK_TARGETS: TARGET_CONSTRUCT_VALIDATION_RANK_TARGETS_KEY,
        }[self.operation]
        if self.capability_key != expected:
            raise ValueError("target construct validation nomination operation and capability differ")
        if self.capability_version != TARGET_CONSTRUCT_VALIDATION_NOMINATION_VERSION:
            raise ValueError("target construct validation nomination capability version differs")
        if not 0 < self.maximum_input_bytes <= 1024 * 1024:
            raise ValueError("target construct validation nomination input bound differs")


def nomination_runtime_config(
    operation: TargetConstructValidationNominationOperation,
    *,
    nomination_freeze_sha256: str,
) -> TargetConstructValidationNominationRuntimeConfig:
    key = {
        TargetConstructValidationNominationOperation.VALIDATE_FREEZE: TARGET_CONSTRUCT_VALIDATION_VALIDATE_NOMINATION_KEY,
        TargetConstructValidationNominationOperation.RANK_TARGETS: TARGET_CONSTRUCT_VALIDATION_RANK_TARGETS_KEY,
    }[operation]
    return TargetConstructValidationNominationRuntimeConfig(
        config_id=f"target-construct-validation.nomination.{operation.value.lower().replace('_', '-')}.config",
        operation=operation,
        capability_key=key,
        capability_version=TARGET_CONSTRUCT_VALIDATION_NOMINATION_VERSION,
        nomination_freeze_sha256=nomination_freeze_sha256,
        maximum_input_bytes=1024 * 1024,
    )


def decode_nomination_config(payload: bytes) -> TargetConstructValidationNominationRuntimeConfig:
    return decode_canonical_bytes(
        payload,
        TargetConstructValidationNominationRuntimeConfig,
        maximum_bytes=64 * 1024,
    )


def _resources() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=1_000_000,
        gpu_devices=0,
        wall_time_seconds=2,
        source_scan_bytes=18_000,
        output_bytes=18_000,
    )


def nomination_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    definitions = (
        (
            TARGET_CONSTRUCT_VALIDATION_VALIDATE_NOMINATION_KEY,
            CapabilityKind.ANALYSIS,
            (),
            (TargetConstructValidationNominationFreeze.SCHEMA,),
        ),
        (
            TARGET_CONSTRUCT_VALIDATION_RANK_TARGETS_KEY,
            CapabilityKind.PROSPECTIVE_NOMINATOR,
            (TargetConstructValidationNominationFreeze.SCHEMA,),
            (TargetConstructValidationNominationResult.SCHEMA,),
        ),
    )
    manifests = tuple(
        CapabilityManifest(
            capability_key=key,
            capability_version=TARGET_CONSTRUCT_VALIDATION_NOMINATION_VERSION,
            kind=kind,
            config_schema=TargetConstructValidationNominationRuntimeConfig.SCHEMA,
            config_schema_sha256=sha256(
                TargetConstructValidationNominationRuntimeConfig.SCHEMA.encode("ascii")
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
            runtime_id="cpython-3.11-target-construct-validation-nomination",
            requires_clean_commit=True,
            requires_active_mount=False,
            requires_network=False,
            conformance_check_ids=(
                "metadata-only-nomination",
                "no-candidate-generator-contact",
                "no-convenience-override",
            ),
            implementation_sha256=implementation_sha256,
        )
        for key, kind, input_schemas, output_schemas in definitions
    )
    return CapabilityRegistry(
        registry_id="target-construct-validation-nomination-runtime",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


def nomination_registrations(
    *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key="target-construct-validation.nomination-runtime-provider",
            provider_version=TARGET_CONSTRUCT_VALIDATION_NOMINATION_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=64 * 1024,
        )
        for manifest in nomination_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


def build_nomination_protocol(
    *,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
) -> ProtocolTemplate:
    step_specs = (
        (
            "validate-nomination-freeze",
            TARGET_CONSTRUCT_VALIDATION_VALIDATE_NOMINATION_KEY,
            ScientificStage.QUALIFY,
            (),
            TargetConstructValidationNominationFreeze.SCHEMA,
            BarrierKind.NONE,
        ),
        (
            "rank-and-freeze-targets",
            TARGET_CONSTRUCT_VALIDATION_RANK_TARGETS_KEY,
            ScientificStage.FREEZE,
            ("validate-nomination-freeze",),
            TargetConstructValidationNominationResult.SCHEMA,
            BarrierKind.FREEZE,
        ),
    )
    if set(config_by_step_id) != {value[0] for value in step_specs}:
        raise ValueError("target construct validation nomination config roster differs")
    steps = []
    for step_id, key, stage, dependencies, output_schema, barrier in step_specs:
        manifest = registry.resolve(key, TARGET_CONSTRUCT_VALIDATION_NOMINATION_VERSION)
        config = config_by_step_id[step_id]
        if (
            config.config_schema != manifest.config_schema
            or config.config_schema_sha256 != manifest.config_schema_sha256
        ):
            raise ValueError("target construct validation nomination config differs from capability schema")
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=stage,
                capability_key=key,
                capability_version=TARGET_CONSTRUCT_VALIDATION_NOMINATION_VERSION,
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
                    source_scan_bytes=(100 if step_id == "validate-nomination-freeze" else 18_000),
                    output_bytes=(18_000 if step_id == "validate-nomination-freeze" else 1_000),
                ),
                resource_lock_ids=(f"target-construct-validation-{step_id}",),
                barrier=barrier,
                maximum_attempts=2,
                obligation_ids=(f"target-construct-validation-{step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id="target-construct-validation-nomination-protocol",
        template_version=TARGET_CONSTRUCT_VALIDATION_NOMINATION_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def nomination_scientific_graph(
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
    producer = next(
        value for value in protocol.steps if value.step_id == "validate-nomination-freeze"
    )
    consumer = next(value for value in protocol.steps if value.step_id == "rank-and-freeze-targets")
    output = producer.outputs[0]
    edge = CandidateGraphEdge(
        edge_id="edge.validate-nomination-freeze.rank-and-freeze-targets",
        producer_node_id=producer.step_id,
        producer_output_id=output.output_id,
        external_input_id=None,
        consumer_node_id=consumer.step_id,
        consumer_input_id="nomination-freeze",
        scientific_role=ScientificInputRole.QUALIFICATION,
        logical_artifact_id="artifact.target-construct-validation.nomination-freeze",
        payload_schema=output.payload_schema,
        media_type=output.media_type,
        maximum_size_bytes=18_000,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        barrier=BarrierKind.FREEZE,
    )
    return CandidateScientificGraph(
        graph_id="graph.target-construct-validation.nomination",
        external_inputs=(),
        nodes=nodes,
        edges=(edge,),
    )


def nomination_catalog(
    *, protocol: ProtocolTemplate, registry: CapabilityRegistry
) -> CandidateCapabilityCatalog:
    graph = nomination_scientific_graph(protocol=protocol, registry=registry)
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
    template = StudyTemplate(
        template_key="target-construct-validation.nomination",
        template_version=TARGET_CONSTRUCT_VALIDATION_NOMINATION_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id="coverage.target-construct-validation.nomination",
            bindings=bindings,
        ),
    )
    implementation_sha256s = {value.implementation_sha256 for value in registry.capabilities}
    if len(implementation_sha256s) != 1:
        raise ValueError("target construct validation nomination implementation identities differ")
    implementation_sha256 = next(iter(implementation_sha256s))
    return CandidateCapabilityCatalog(
        catalog_id="target-construct-validation-nomination-candidate-catalog",
        registrations=nomination_registrations(implementation_sha256=implementation_sha256),
        templates=(template,),
    )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationNominationConfigAdapter:
    @property
    def provider_key(self) -> str:
        return "target-construct-validation.nomination-runtime-provider"

    @property
    def provider_version(self) -> str:
        return TARGET_CONSTRUCT_VALIDATION_NOMINATION_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != TargetConstructValidationNominationRuntimeConfig.SCHEMA:
            raise ValueError("target construct validation nomination config adapter schema differs")
        decode_nomination_config(payload)


def nomination_config_decoders(
    catalog: CandidateCapabilityCatalog,
) -> tuple[CandidateCapabilityConfigDecoder, ...]:
    if {value.provider_key for value in catalog.registrations} != {
        "target-construct-validation.nomination-runtime-provider"
    }:
        raise ValueError("target construct validation nomination provider roster differs")
    return (TargetConstructValidationNominationConfigAdapter(),)


class _NominationRunner:
    def __init__(self, manifest: CapabilityManifest, record: CanonicalRecord) -> None:
        self.manifest = manifest
        self.record = record
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        if len(context.output_ports) != 1:
            raise ValueError("target construct validation nomination task must have one output")
        if context.output_ports[0].payload_schema != self.record.SCHEMA:
            raise ValueError("target construct validation nomination output schema differs")
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=self.record.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck("target-construct-validation-nomination-runtime-contract", True, ()),),
        )


class TargetConstructValidationNominationRuntimeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        nomination_freeze: TargetConstructValidationNominationFreeze,
    ) -> None:
        expected = nomination_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("target construct validation nomination provider registry differs")
        records = {
            TARGET_CONSTRUCT_VALIDATION_VALIDATE_NOMINATION_KEY: nomination_freeze,
            TARGET_CONSTRUCT_VALIDATION_RANK_TARGETS_KEY: nomination_freeze.nomination_result,
        }
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self._runners = tuple(
            _NominationRunner(manifest, records[manifest.capability_key])
            for manifest in registry.capabilities
        )
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("target construct validation nomination provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("target construct validation nomination provider plan/source differs")
        return ()

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("target construct validation nomination semantic registry differs")
        types = {
            TargetConstructValidationNominationFreeze.SCHEMA: TargetConstructValidationNominationFreeze,
            TargetConstructValidationNominationResult.SCHEMA: TargetConstructValidationNominationResult,
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
            raise ValueError("target construct validation nomination adjudication registry differs")
        return None


__all__ = [
    'TargetConstructValidationNominationOperation',
    'TargetConstructValidationNominationRuntimeConfig',
    'TargetConstructValidationNominationRuntimeProvider',
    "build_nomination_protocol",
    "decode_nomination_config",
    "nomination_catalog",
    "nomination_config_decoders",
    "nomination_registry",
    "nomination_runtime_config",
]

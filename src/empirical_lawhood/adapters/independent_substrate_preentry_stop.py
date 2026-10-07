"Registered zero-unit pre-entry stop projection for independent substrate grounding.\n\nGrid2Op source readiness and NREL source-authority readiness are source/authority gates, not target\nexperiments.  This adapter projects an exact outcome-blind readiness record\nthrough the ordinary static-capability and scientific-DAG seams into the one\ncompact handoff consumed by independent substrate cross-target.  It cannot acquire a source, decode a\nprotected receiver, create a scientific unit, or convert nonentry into\nscientific opposition.\n"

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import ClassVar, Final, TypeAlias

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateTargetKind, IndependentSubstrateTargetTerminalHandoff
from empirical_lawhood.adapters.physical.nrel_inverter_consistency import NRELArchiveReadinessDisposition, NRELArchiveReadiness, nrel_readiness_stop_handoff
from empirical_lawhood.adapters.simulators.grid2op_source_readiness import Grid2OpReadinessDisposition, Grid2OpSourceReadiness, grid2op_readiness_stop_handoff
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
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)


INDEPENDENT_SUBSTRATE_PREENTRY_STOP_VERSION: Final = "1.0.0"
INDEPENDENT_SUBSTRATE_PREENTRY_STOP_PROVIDER_KEY: Final = "independent-substrate-grounding.preentry-stop-runtime-provider"
GRID2OP_PREENTRY_STOP_KEY: Final = "grid2op.project-independent-substrate-grounding-source-readiness-preentry-stop"
NREL_PREENTRY_STOP_KEY: Final = "nrel.project-independent-substrate-grounding-source-authority-readiness-preentry-stop"

ReadinessRecord: TypeAlias = Grid2OpSourceReadiness | NRELArchiveReadiness


def _target_spec(
    slot: IndependentSubstrateTargetKind,
) -> tuple[str, type[ReadinessRecord], str]:
    if slot is IndependentSubstrateTargetKind.GRID2OP:
        return (
            GRID2OP_PREENTRY_STOP_KEY,
            Grid2OpSourceReadiness,
            "source-readiness",
        )
    if slot is IndependentSubstrateTargetKind.NREL_INVERTER:
        return (
            NREL_PREENTRY_STOP_KEY,
            NRELArchiveReadiness,
            "source-authority-readiness",
        )
    raise ValueError("independent substrate grounding pre-entry stop supports only source readiness and source-authority readiness")


def _record_identity(record: CanonicalRecord) -> ObjectIdentity:
    for attribute in ("config_id", "readiness_id", "handoff_id"):
        object_id = getattr(record, attribute, None)
        if isinstance(object_id, str):
            return ObjectIdentity.from_record(object_id, record)
    raise TypeError(f"unsupported independent substrate grounding pre-entry record: {type(record)!r}")


def _project(readiness: ReadinessRecord) -> IndependentSubstrateTargetTerminalHandoff:
    if isinstance(readiness, Grid2OpSourceReadiness):
        return grid2op_readiness_stop_handoff(readiness)
    if isinstance(readiness, NRELArchiveReadiness):
        return nrel_readiness_stop_handoff(readiness)
    raise TypeError(f"unsupported independent substrate grounding readiness record: {type(readiness)!r}")


def _slot(readiness: ReadinessRecord) -> IndependentSubstrateTargetKind:
    if isinstance(readiness, Grid2OpSourceReadiness):
        return IndependentSubstrateTargetKind.GRID2OP
    if isinstance(readiness, NRELArchiveReadiness):
        return IndependentSubstrateTargetKind.NREL_INVERTER
    raise TypeError(f"unsupported independent substrate grounding readiness record: {type(readiness)!r}")


def _validate_adverse_readiness(readiness: ReadinessRecord) -> None:
    if isinstance(readiness, Grid2OpSourceReadiness):
        if readiness.disposition is Grid2OpReadinessDisposition.READY_FOR_SOURCE_QUALIFICATION:
            raise ValueError("ready Grid2Op source readiness cannot enter the pre-entry stop DAG")
    elif isinstance(readiness, NRELArchiveReadiness):
        if (
            readiness.disposition
            is NRELArchiveReadinessDisposition.READY_FOR_AUTHORIZED_ACQUISITION
        ):
            raise ValueError("ready NREL source-authority readiness cannot enter the pre-entry stop DAG")
    else:
        raise TypeError(f"unsupported independent substrate grounding readiness record: {type(readiness)!r}")
    if (
        readiness.scientific_unit_count != 0
        or readiness.outcome_access is not OutcomeAccess.OUTCOME_BLIND
    ):
        raise ValueError("independent substrate grounding pre-entry stop requires a zero-unit outcome-blind readiness")


@dataclass(frozen=True, slots=True)
class IndependentSubstratePreentryStopConfig(CanonicalRecord):
    "Exact readiness operand and expected nonpromotable projection."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-preentry-stop-config'

    config_id: str
    target_slot: IndependentSubstrateTargetKind
    capability_key: str
    capability_version: str
    readiness: ObjectIdentity
    expected_handoff: ObjectIdentity
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        expected_key, readiness_type, _gate = _target_spec(self.target_slot)
        if (
            self.capability_key != expected_key
            or self.capability_version != INDEPENDENT_SUBSTRATE_PREENTRY_STOP_VERSION
            or self.readiness.object_schema != readiness_type.SCHEMA
            or self.expected_handoff.object_schema != IndependentSubstrateTargetTerminalHandoff.SCHEMA
        ):
            raise ValueError("independent substrate grounding pre-entry stop config identity differs")
        if not 0 < self.maximum_input_bytes <= 8 * 1024**2:
            raise ValueError("independent substrate grounding pre-entry stop input bound differs")


def independent_substrate_preentry_stop_config(readiness: ReadinessRecord) -> IndependentSubstratePreentryStopConfig:
    _validate_adverse_readiness(readiness)
    slot = _slot(readiness)
    capability_key, _readiness_type, gate = _target_spec(slot)
    handoff = _project(readiness)
    return IndependentSubstratePreentryStopConfig(
        config_id=f"config.independent-substrate-grounding.{gate}-preentry-stop",
        target_slot=slot,
        capability_key=capability_key,
        capability_version=INDEPENDENT_SUBSTRATE_PREENTRY_STOP_VERSION,
        readiness=_record_identity(readiness),
        expected_handoff=_record_identity(handoff),
        maximum_input_bytes=8 * 1024**2,
    )


def decode_independent_substrate_preentry_stop_config(payload: bytes) -> IndependentSubstratePreentryStopConfig:
    return decode_canonical_bytes(payload, IndependentSubstratePreentryStopConfig, maximum_bytes=256 * 1024)


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=256 * 1024**2,
        gpu_devices=0,
        wall_time_seconds=60,
        source_scan_bytes=8 * 1024**2,
        output_bytes=8 * 1024**2,
    )


_PERMISSIONS = tuple(
    sorted(
        (
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.READ_OUTCOME_VISIBLE,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        key=lambda value: value.value,
    )
)


def independent_substrate_preentry_stop_registry(
    *, config: IndependentSubstratePreentryStopConfig, implementation_sha256: str
) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    _key, readiness_type, gate = _target_spec(config.target_slot)
    manifest = CapabilityManifest(
        capability_key=config.capability_key,
        capability_version=config.capability_version,
        kind=CapabilityKind.REPORTER,
        config_schema=config.SCHEMA,
        config_schema_sha256=sha256(config.SCHEMA.encode("ascii")).hexdigest(),
        input_schema_ids=tuple(sorted((config.SCHEMA, readiness_type.SCHEMA))),
        output_schema_ids=(IndependentSubstrateTargetTerminalHandoff.SCHEMA,),
        permissions=_PERMISSIONS,
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        resource_ceiling=_budget(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id=f"cpython-independent-substrate-grounding-{gate}-preentry-stop",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=tuple(
            sorted(
                (
                    "exact-outcome-blind-readiness",
                    "nonentry-not-scientific-opposition",
                    "zero-scientific-units",
                )
            )
        ),
        implementation_sha256=implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id=f"independent-substrate-grounding-{gate}-preentry-stop",
        capabilities=(manifest,),
    )


def independent_substrate_preentry_stop_candidate_registrations(
    *, config: IndependentSubstratePreentryStopConfig, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    manifest = independent_substrate_preentry_stop_registry(
        config=config, implementation_sha256=implementation_sha256
    ).capabilities[0]
    return (
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=INDEPENDENT_SUBSTRATE_PREENTRY_STOP_PROVIDER_KEY,
            provider_version=INDEPENDENT_SUBSTRATE_PREENTRY_STOP_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=256 * 1024,
        ),
    )


def build_independent_substrate_preentry_stop_protocol(
    *,
    registry: CapabilityRegistry,
    config: IndependentSubstratePreentryStopConfig,
    config_ref: CapabilityConfigRef,
) -> ProtocolTemplate:
    manifest = registry.resolve(config.capability_key, config.capability_version)
    _key, _readiness_type, gate = _target_spec(config.target_slot)
    if (
        config_ref.config_id != config.config_id
        or config_ref.config_schema != config.SCHEMA
        or config_ref.config_schema_sha256 != manifest.config_schema_sha256
        or config_ref.content_sha256 != config.fingerprint()
    ):
        raise ValueError("independent substrate grounding pre-entry stop config reference differs")
    step = ProtocolStepTemplate(
        step_id=f"project-{gate}-preentry-stop-handoff",
        stage=ScientificStage.REPORT,
        capability_key=manifest.capability_key,
        capability_version=manifest.capability_version,
        config=config_ref,
        dependency_step_ids=(),
        outputs=(
            OutputTemplate(
                output_id="target-handoff",
                payload_schema=IndependentSubstrateTargetTerminalHandoff.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                filename_suffix=".json",
            ),
        ),
        required_permissions=manifest.permissions,
        requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        resource_budget=manifest.resource_ceiling,
        resource_lock_ids=(f"independent-substrate-grounding-{gate}-preentry-stop",),
        barrier=BarrierKind.NONE,
        maximum_attempts=2,
        obligation_ids=tuple(
            sorted(
                (
                    f"independent-substrate-grounding-{gate}-exact-readiness-binding",
                    f"independent-substrate-grounding-{gate}-nonentry-not-opposition",
                    f"independent-substrate-grounding-{gate}-zero-executed-units",
                )
            )
        ),
    )
    return ProtocolTemplate(
        template_id=f"independent-substrate-grounding-{gate}-preentry-stop-protocol",
        template_version=INDEPENDENT_SUBSTRATE_PREENTRY_STOP_VERSION,
        steps=(step,),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def independent_substrate_preentry_stop_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    config: IndependentSubstratePreentryStopConfig,
) -> CandidateScientificGraph:
    if len(protocol.steps) != 1:
        raise ValueError("independent substrate grounding pre-entry stop protocol roster differs")
    step = protocol.steps[0]
    manifest = registry.resolve(step.capability_key, step.capability_version)
    _key, readiness_type, gate = _target_spec(config.target_slot)
    external = CandidateGraphExternalInput(
        input_id=f"input.independent-substrate-grounding.{gate}.readiness",
        scientific_role=ScientificInputRole.QUALIFICATION,
        logical_artifact_id=f"artifact.independent-substrate-grounding.{gate}.readiness",
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=config.readiness.object_fingerprint,
        payload_schema=readiness_type.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=config.maximum_input_bytes,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    node = CandidateGraphNode(
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
    edge = CandidateGraphEdge(
        edge_id=f"edge.independent-substrate-grounding.{gate}.readiness-to-stop",
        producer_node_id=None,
        producer_output_id=None,
        external_input_id=external.input_id,
        consumer_node_id=step.step_id,
        consumer_input_id=external.input_id,
        scientific_role=external.scientific_role,
        logical_artifact_id=external.logical_artifact_id,
        payload_schema=external.payload_schema,
        media_type=external.media_type,
        maximum_size_bytes=external.maximum_size_bytes,
        outcome_access=external.outcome_access,
        visibility_ceiling=external.visibility_ceiling,
        barrier=step.barrier,
    )
    return CandidateScientificGraph(
        graph_id=f"graph.independent-substrate-grounding.{gate}.preentry-stop",
        external_inputs=(external,),
        nodes=(node,),
        edges=(edge,),
    )


def independent_substrate_preentry_stop_study_template(
    *, protocol: ProtocolTemplate, graph: CandidateScientificGraph
) -> StudyTemplate:
    step = protocol.steps[0]
    edge_ids = tuple(
        sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == step.step_id)
    )
    return StudyTemplate(
        template_key=protocol.template_id,
        template_version=protocol.template_version,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id=f"coverage.{protocol.template_id}",
            bindings=tuple(
                ObligationCoverageBinding(
                    obligation_id=obligation,
                    proof_owner_node_id=step.step_id,
                    required_output_id="target-handoff",
                    contributor_edge_ids=edge_ids,
                )
                for obligation in step.obligation_ids
            ),
        ),
    )


def _decode_readiness(context: TaskContext, config: IndependentSubstratePreentryStopConfig) -> ReadinessRecord:
    _key, readiness_type, _gate = _target_spec(config.target_slot)
    values = tuple(
        decode_canonical_bytes(port.read(), readiness_type, maximum_bytes=port.size_bytes)
        for port in context.input_ports
        if port.payload_schema == readiness_type.SCHEMA
    )
    if len(values) != 1:
        raise ValueError("independent substrate grounding pre-entry stop requires exactly one readiness input")
    readiness = values[0]
    assert isinstance(readiness, (Grid2OpSourceReadiness, NRELArchiveReadiness))
    return readiness


class _IndependentSubstratePreentryStopRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        configs = tuple(
            decode_independent_substrate_preentry_stop_config(port.read())
            for port in context.input_ports
            if port.payload_schema == IndependentSubstratePreentryStopConfig.SCHEMA
        )
        if len(configs) != 1:
            raise ValueError("independent substrate grounding pre-entry stop requires exactly one config input")
        config = configs[0]
        readiness = _decode_readiness(context, config)
        _validate_adverse_readiness(readiness)
        handoff = _project(readiness)
        if (
            config.capability_key != self.manifest.capability_key
            or context.config.content_sha256 != config.fingerprint()
            or context.permissions != self.manifest.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or _record_identity(readiness) != config.readiness
            or _record_identity(handoff) != config.expected_handoff
            or {port.payload_schema for port in context.output_ports}
            != {IndependentSubstrateTargetTerminalHandoff.SCHEMA}
        ):
            raise ValueError("independent substrate grounding pre-entry stop runtime identity differs")
        _key, _readiness_type, gate = _target_spec(config.target_slot)
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(output_id=port.output_id, payload=handoff.canonical_bytes())
                for port in context.output_ports
            ),
            checks=tuple(
                sorted(
                    (
                        ReceiptCheck(f"independent-substrate-grounding-{gate}-exact-readiness", True, ()),
                        ReceiptCheck(f"independent-substrate-grounding-{gate}-nonentry-not-opposition", True, ()),
                        ReceiptCheck(f"independent-substrate-grounding-{gate}-zero-scientific-units", True, ()),
                    ),
                    key=lambda value: value.check_id,
                )
            ),
        )


class IndependentSubstratePreentryStopRuntimeProvider(CampaignRuntimeProvider):
    """Closed provider for one exact source readiness or source-authority readiness terminal projection."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: IndependentSubstratePreentryStopConfig,
        readiness: ReadinessRecord,
    ) -> None:
        expected = independent_substrate_preentry_stop_registry(
            config=config,
            implementation_sha256=registry.capabilities[0].implementation_sha256,
        )
        if registry != expected or independent_substrate_preentry_stop_config(readiness) != config:
            raise ValueError("independent substrate grounding pre-entry stop provider identity differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.config = config
        self.readiness = readiness
        self._runner = _IndependentSubstratePreentryStopRunner(registry.capabilities[0])
        self.capability_count = 1

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("independent substrate grounding pre-entry stop registry/source differs")
        return (self._runner,)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("independent substrate grounding pre-entry stop plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        _key, _readiness_type, gate = _target_spec(self.config.target_slot)
        records: dict[str, CanonicalRecord] = {
            task.capability.config.artifact_id: self.config for task in plan.tasks
        }
        records[f"artifact.independent-substrate-grounding.{gate}.readiness"] = self.readiness
        if set(records) != set(specs):
            raise ValueError("independent substrate grounding pre-entry stop external input roster differs")
        values = []
        for artifact_id, record in sorted(records.items()):
            spec = specs[artifact_id]
            parent = ArtifactLineageParent(
                identity=_record_identity(record),
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
                    outcome_access=spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND,
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
            raise ValueError("independent substrate grounding pre-entry stop semantic registry differs")
        del execution_plan
        return (
            CapabilityOutputSemanticContract.from_manifest(
                registry.capabilities[0],
                payload_schema=IndependentSubstrateTargetTerminalHandoff.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(
                    sorted(value.name for value in fields(IndependentSubstrateTargetTerminalHandoff))
                ),
            ),
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("independent substrate grounding pre-entry stop adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "GRID2OP_PREENTRY_STOP_KEY",
    "INDEPENDENT_SUBSTRATE_PREENTRY_STOP_PROVIDER_KEY",
    "INDEPENDENT_SUBSTRATE_PREENTRY_STOP_VERSION",
    'IndependentSubstratePreentryStopConfig',
    'IndependentSubstratePreentryStopRuntimeProvider',
    "NREL_PREENTRY_STOP_KEY",
    "ReadinessRecord",
    'build_independent_substrate_preentry_stop_protocol',
    'decode_independent_substrate_preentry_stop_config',
    'independent_substrate_preentry_stop_candidate_registrations',
    'independent_substrate_preentry_stop_config',
    'independent_substrate_preentry_stop_study_template',
    'independent_substrate_preentry_stop_registry',
    'independent_substrate_preentry_stop_scientific_graph',
]

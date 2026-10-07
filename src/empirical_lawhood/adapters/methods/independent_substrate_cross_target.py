"""Registered compact cross-target adjudication for independent substrate grounding.

This module consumes only already revealed, compact target-terminal handoffs.
It never reads target payloads, recomputes target-local results, pools evidence
across substrates, or creates a prospective physical axis.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
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
from empirical_lawhood.runtime.candidate_compiler import CandidateGraphEdge, CandidateGraphExternalInput, CandidateGraphNode, CandidateScientificGraph, ContentIdentityPolicy, ObligationCoverage, ObligationCoverageBinding, StudyTemplate, ScientificInputRole
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityRegistration
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)

from .independent_substrate_grounding import IndependentSubstrateAxisAdjudication, IndependentSubstrateTargetKind, IndependentSubstrateTargetTerminalHandoff, adjudicate_axes


INDEPENDENT_SUBSTRATE_CROSS_TARGET_VERSION = "1.0.0"
INDEPENDENT_SUBSTRATE_CROSS_TARGET_CAPABILITY_KEY = "method.independent-substrate-grounding.cross-target-adjudicate"
INDEPENDENT_SUBSTRATE_CROSS_TARGET_STEP_ID = "adjudicate-cross-target-axes"
_TARGET_SLOTS = tuple(sorted(IndependentSubstrateTargetKind, key=lambda value: value.value))


@dataclass(frozen=True, slots=True)
class IndependentSubstrateTargetHandoffBinding(CanonicalRecord):
    """One exact target slot and its immutable compact handoff identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-target-handoff-binding'

    slot: IndependentSubstrateTargetKind
    handoff: ObjectIdentity

    def __post_init__(self) -> None:
        if self.handoff.object_schema != IndependentSubstrateTargetTerminalHandoff.SCHEMA:
            raise ValueError("independent substrate cross-target binding must identify a compact target handoff")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateCrossTargetConfig(CanonicalRecord):
    """Exact all-target roster frozen before cross-target composition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-cross-target-config'

    config_id: str
    capability_key: str
    capability_version: str
    bindings: tuple[IndependentSubstrateTargetHandoffBinding, ...]
    maximum_handoff_bytes: int
    prospective_physical_axis_present: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        if (
            self.capability_key != INDEPENDENT_SUBSTRATE_CROSS_TARGET_CAPABILITY_KEY
            or self.capability_version != INDEPENDENT_SUBSTRATE_CROSS_TARGET_VERSION
        ):
            raise ValueError("independent substrate cross-target config capability identity differs")
        if tuple(value.slot for value in self.bindings) != _TARGET_SLOTS:
            raise ValueError("independent substrate cross-target config must bind every in-scope target slot once")
        identities = tuple(value.handoff.object_id for value in self.bindings)
        if len(set(identities)) != len(identities):
            raise ValueError("independent substrate cross-target handoff identities must be unique")
        if not 0 < self.maximum_handoff_bytes <= 8 * 1024**2:
            raise ValueError("independent substrate cross-target handoff byte bound differs")
        if self.prospective_physical_axis_present:
            raise ValueError("independent substrate grounding excludes a prospective physical axis")


def independent_substrate_cross_target_config(
    handoffs: tuple[IndependentSubstrateTargetTerminalHandoff, ...],
) -> IndependentSubstrateCrossTargetConfig:
    by_slot = {value.slot: value for value in handoffs}
    if set(by_slot) != set(_TARGET_SLOTS) or len(by_slot) != len(handoffs):
        raise ValueError("independent substrate cross-target requires one terminal handoff for every target slot")
    return IndependentSubstrateCrossTargetConfig(
        config_id="config.independent-substrate-grounding.cross-target-adjudication",
        capability_key=INDEPENDENT_SUBSTRATE_CROSS_TARGET_CAPABILITY_KEY,
        capability_version=INDEPENDENT_SUBSTRATE_CROSS_TARGET_VERSION,
        bindings=tuple(
            IndependentSubstrateTargetHandoffBinding(
                slot=slot,
                handoff=ObjectIdentity.from_record(by_slot[slot].handoff_id, by_slot[slot]),
            )
            for slot in _TARGET_SLOTS
        ),
        maximum_handoff_bytes=2 * 1024**2,
        prospective_physical_axis_present=False,
    )


def decode_independent_substrate_cross_target_config(payload: bytes) -> IndependentSubstrateCrossTargetConfig:
    return decode_canonical_bytes(
        payload,
        IndependentSubstrateCrossTargetConfig,
        maximum_bytes=256 * 1024,
    )


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=256 * 1024**2,
        gpu_devices=0,
        wall_time_seconds=60,
        source_scan_bytes=16 * 1024**2,
        output_bytes=8 * 1024**2,
    )


def independent_substrate_cross_target_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    permissions = tuple(
        sorted(
            (
                CapabilityPermission.READ_DEVELOPMENT,
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            key=lambda value: value.value,
        )
    )
    manifest = CapabilityManifest(
        capability_key=INDEPENDENT_SUBSTRATE_CROSS_TARGET_CAPABILITY_KEY,
        capability_version=INDEPENDENT_SUBSTRATE_CROSS_TARGET_VERSION,
        kind=CapabilityKind.HYPOTHESIS_SYNTHESIZER,
        config_schema=IndependentSubstrateCrossTargetConfig.SCHEMA,
        config_schema_sha256=sha256(IndependentSubstrateCrossTargetConfig.SCHEMA.encode("ascii")).hexdigest(),
        input_schema_ids=tuple(
            sorted(
                (
                    IndependentSubstrateCrossTargetConfig.SCHEMA,
                    IndependentSubstrateTargetTerminalHandoff.SCHEMA,
                )
            )
        ),
        output_schema_ids=(IndependentSubstrateAxisAdjudication.SCHEMA,),
        permissions=permissions,
        maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        resource_ceiling=_budget(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-independent-substrate-grounding-cross-target-adjudication",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "compact-handoffs-only",
            "counterexample-first-precedence",
            "no-cross-substrate-pooling",
            "prospective-physical-axis-excluded",
        ),
        implementation_sha256=implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-cross-target-adjudication",
        capabilities=(manifest,),
    )


def independent_substrate_cross_target_candidate_registrations(
    *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    manifest = independent_substrate_cross_target_registry(
        implementation_sha256=implementation_sha256
    ).capabilities[0]
    return (
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key="independent-substrate-grounding.cross-target-runtime-provider",
            provider_version=INDEPENDENT_SUBSTRATE_CROSS_TARGET_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=256 * 1024,
        ),
    )


def build_independent_substrate_cross_target_protocol(
    *, registry: CapabilityRegistry, config_ref: CapabilityConfigRef
) -> ProtocolTemplate:
    manifest = registry.resolve(
        INDEPENDENT_SUBSTRATE_CROSS_TARGET_CAPABILITY_KEY,
        INDEPENDENT_SUBSTRATE_CROSS_TARGET_VERSION,
    )
    if (
        config_ref.config_schema != manifest.config_schema
        or config_ref.config_schema_sha256 != manifest.config_schema_sha256
    ):
        raise ValueError("independent substrate cross-target config reference differs from the capability")
    step = ProtocolStepTemplate(
        step_id=INDEPENDENT_SUBSTRATE_CROSS_TARGET_STEP_ID,
        stage=ScientificStage.SYNTHESIZE,
        capability_key=manifest.capability_key,
        capability_version=manifest.capability_version,
        config=config_ref,
        dependency_step_ids=(),
        outputs=(
            OutputTemplate(
                output_id="four-axis-adjudication",
                payload_schema=IndependentSubstrateAxisAdjudication.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                filename_suffix=".json",
            ),
        ),
        required_permissions=manifest.permissions,
        requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        resource_budget=manifest.resource_ceiling,
        resource_lock_ids=("independent-substrate-grounding-cross-target-adjudication",),
        barrier=BarrierKind.NONE,
        maximum_attempts=2,
        obligation_ids=(
            "independent-substrate-cross-target-counterexample-first",
            "independent-substrate-cross-target-four-axis-noncompensation",
            "independent-substrate-cross-target-no-prospective-physical-axis",
        ),
    )
    return ProtocolTemplate(
        template_id="independent-substrate-grounding-cross-target-adjudication-protocol",
        template_version=INDEPENDENT_SUBSTRATE_CROSS_TARGET_VERSION,
        steps=(step,),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def independent_substrate_cross_target_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    config: IndependentSubstrateCrossTargetConfig,
) -> CandidateScientificGraph:
    if len(protocol.steps) != 1 or protocol.steps[0].step_id != INDEPENDENT_SUBSTRATE_CROSS_TARGET_STEP_ID:
        raise ValueError("independent substrate cross-target protocol step roster differs")
    step = protocol.steps[0]
    manifest = registry.resolve(step.capability_key, step.capability_version)
    inputs = tuple(
        CandidateGraphExternalInput(
            input_id=f"input.independent-substrate-cross-target.{binding.slot.value.lower()}",
            scientific_role=ScientificInputRole.PARENT_RECEIPT,
            logical_artifact_id=binding.handoff.object_id,
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=binding.handoff.object_fingerprint,
            payload_schema=IndependentSubstrateTargetTerminalHandoff.SCHEMA,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_size_bytes=config.maximum_handoff_bytes,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        )
        for binding in config.bindings
    )
    edges = tuple(
        CandidateGraphEdge(
            edge_id=f"edge.{value.input_id}.{INDEPENDENT_SUBSTRATE_CROSS_TARGET_STEP_ID}",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=value.input_id,
            consumer_node_id=INDEPENDENT_SUBSTRATE_CROSS_TARGET_STEP_ID,
            consumer_input_id=value.input_id,
            scientific_role=value.scientific_role,
            logical_artifact_id=value.logical_artifact_id,
            payload_schema=value.payload_schema,
            media_type=value.media_type,
            maximum_size_bytes=value.maximum_size_bytes,
            outcome_access=value.outcome_access,
            visibility_ceiling=value.visibility_ceiling,
            barrier=step.barrier,
        )
        for value in inputs
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
    return CandidateScientificGraph(
        graph_id="graph.independent-substrate-grounding.cross-target-adjudication",
        external_inputs=inputs,
        nodes=(node,),
        edges=edges,
    )


def independent_substrate_cross_target_study_template(
    *, protocol: ProtocolTemplate, graph: CandidateScientificGraph
) -> StudyTemplate:
    """Bind every cross-target obligation to the exact four-handoff DAG."""

    if len(protocol.steps) != 1 or len(graph.nodes) != 1:
        raise ValueError("independent substrate cross-target programme roster differs")
    step = protocol.steps[0]
    contributor_edge_ids = tuple(
        sorted(edge.edge_id for edge in graph.edges if edge.consumer_node_id == step.step_id)
    )
    if len(contributor_edge_ids) != len(_TARGET_SLOTS):
        raise ValueError("independent substrate cross-target programme requires all four target handoffs")
    return StudyTemplate(
        template_key="independent-substrate-grounding.cross-target-adjudication",
        template_version=INDEPENDENT_SUBSTRATE_CROSS_TARGET_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id="coverage.independent-substrate-grounding.cross-target-adjudication",
            bindings=tuple(
                ObligationCoverageBinding(
                    obligation_id=obligation,
                    proof_owner_node_id=step.step_id,
                    required_output_id="four-axis-adjudication",
                    contributor_edge_ids=contributor_edge_ids,
                )
                for obligation in step.obligation_ids
            ),
        ),
    )


class _IndependentSubstrateCrossTargetRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        configs = tuple(
            decode_independent_substrate_cross_target_config(port.read())
            for port in context.input_ports
            if port.payload_schema == IndependentSubstrateCrossTargetConfig.SCHEMA
        )
        handoffs = tuple(
            decode_canonical_bytes(
                port.read(),
                IndependentSubstrateTargetTerminalHandoff,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == IndependentSubstrateTargetTerminalHandoff.SCHEMA
        )
        if len(configs) != 1 or len(handoffs) != len(_TARGET_SLOTS):
            raise ValueError("independent substrate cross-target task requires one config and four compact handoffs")
        config = configs[0]
        if config.capability_key != self.manifest.capability_key:
            raise ValueError("independent substrate cross-target task/config capability differs")
        observed = {
            value.slot: ObjectIdentity.from_record(value.handoff_id, value) for value in handoffs
        }
        expected = {value.slot: value.handoff for value in config.bindings}
        if observed != expected:
            raise ValueError("independent substrate cross-target compact handoffs differ from the frozen roster")
        adjudication = adjudicate_axes(tuple(sorted(handoffs, key=lambda value: value.slot.value)))
        if {port.payload_schema for port in context.output_ports} != {
            IndependentSubstrateAxisAdjudication.SCHEMA
        }:
            raise ValueError("independent substrate cross-target output schema roster differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=adjudication.canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("independent-substrate-cross-target-counterexample-first", True, ()),
                ReceiptCheck("independent-substrate-cross-target-exact-handoff-roster", True, ()),
                ReceiptCheck(
                    "independent-substrate-cross-target-prospective-physical-axis-excluded",
                    not adjudication.prospective_physical_axis_present,
                    (),
                ),
            ),
        )


class IndependentSubstrateCrossTargetRuntimeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: IndependentSubstrateCrossTargetConfig,
        handoffs: tuple[IndependentSubstrateTargetTerminalHandoff, ...],
    ) -> None:
        if len(registry.capabilities) != 1:
            raise ValueError("independent substrate cross-target provider registry differs")
        expected = {value.slot: value.handoff for value in config.bindings}
        observed = {
            value.slot: ObjectIdentity.from_record(value.handoff_id, value) for value in handoffs
        }
        if observed != expected:
            raise ValueError("independent substrate cross-target provider handoffs differ from config")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.config = config
        self.handoffs = tuple(sorted(handoffs, key=lambda value: value.handoff_id))
        self._runner = _IndependentSubstrateCrossTargetRunner(registry.capabilities[0])
        self.capability_count = 1

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("independent substrate cross-target provider registry/source differs")
        return (self._runner,)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("independent substrate cross-target provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            task.capability.config.artifact_id: self.config for task in plan.tasks
        }
        records.update({value.handoff_id: value for value in self.handoffs})
        if set(records) != set(specs):
            raise ValueError("independent substrate cross-target plan external input roster differs")
        values = []
        for artifact_id, record in sorted(records.items()):
            spec = specs[artifact_id]
            config_input = record is self.config
            visibility = (
                VisibilityCeiling.PROSPECTIVE if config_input else VisibilityCeiling.OUTCOME_VISIBLE
            )
            access = (
                OutcomeAccess.OUTCOME_BLIND if config_input else OutcomeAccess.EVALUATION_REVEALED
            )
            if config_input:
                record_id = self.config.config_id
            elif isinstance(record, IndependentSubstrateTargetTerminalHandoff):
                record_id = record.handoff_id
            else:
                raise TypeError("independent substrate cross-target external record type differs")
            parent = ArtifactLineageParent(
                identity=ObjectIdentity.from_record(record_id, record),
                visibility_ceiling=visibility,
                outcome_access=access,
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    payload=record.canonical_bytes(),
                    visibility_ceiling=(spec.expected_visibility_ceiling or visibility),
                    outcome_access=(spec.expected_outcome_access or access),
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
            raise ValueError("independent substrate cross-target semantic registry differs")
        del execution_plan
        return (
            CapabilityOutputSemanticContract.from_manifest(
                registry.capabilities[0],
                payload_schema=IndependentSubstrateAxisAdjudication.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(value.name for value in fields(IndependentSubstrateAxisAdjudication))),
            ),
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        """Do not collapse four cross-world axes into one campaign verdict.

        ``ScientificAdjudicationOutputContract`` is intentionally scoped to a
        single campaign system, relation and independent unit.  independent substrate cross-target binds
        four terminal handoffs from distinct evidence worlds and preserves four
        noncompensating axes in ``IndependentSubstrateAxisAdjudication`` instead.
        """

        if registry != self.registry:
            raise ValueError("independent substrate cross-target adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "INDEPENDENT_SUBSTRATE_CROSS_TARGET_CAPABILITY_KEY",
    "INDEPENDENT_SUBSTRATE_CROSS_TARGET_STEP_ID",
    "INDEPENDENT_SUBSTRATE_CROSS_TARGET_VERSION",
    'IndependentSubstrateCrossTargetConfig',
    'IndependentSubstrateCrossTargetRuntimeProvider',
    'IndependentSubstrateTargetHandoffBinding',
    'build_independent_substrate_cross_target_protocol',
    'decode_independent_substrate_cross_target_config',
    'independent_substrate_cross_target_candidate_registrations',
    'independent_substrate_cross_target_config',
    'independent_substrate_cross_target_study_template',
    'independent_substrate_cross_target_registry',
    'independent_substrate_cross_target_scientific_graph',
]

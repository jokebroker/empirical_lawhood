"""Receipt-ready compact cross-target DAG for selective dependence response."""

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
    require_sorted_unique_ids,
    validate_semantic_version,
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
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificInputRole, ScientificStage
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .contracts import SelectiveDependenceResponseCrossTargetAdjudication, SelectiveDependenceResponseMethodQuestionFreeze, SelectiveDependenceResponseTargetHandoff
from .completion import SelectiveDependenceResponseTargetCompletionEnvelope
from .cross_target import SelectiveDependenceResponseCrossTargetEligibility, adjudicate_cross_target, qualify_cross_target


SELECTIVE_DEPENDENCE_RESPONSE_CROSS_TARGET_VERSION = "1.0.0"


class SelectiveDependenceResponseCrossTargetOperation(StrEnum):
    ADJUDICATE_RELATION = "ADJUDICATE_RELATION"
    VERIFY_ELIGIBILITY = "VERIFY_ELIGIBILITY"


def cross_target_capability_key(operation: SelectiveDependenceResponseCrossTargetOperation) -> str:
    return f"method.selective-dependence-response.cross-target.{operation.value.lower().replace('_', '-')}"


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseCrossTargetEnvelopeBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-cross-target-envelope-binding'

    target_id: str
    completion_envelope: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.target_id, field_name="target_id")
        if self.completion_envelope.object_schema != SelectiveDependenceResponseTargetCompletionEnvelope.SCHEMA:
            raise ValueError("cross-target binding must identify a completed selective-response target")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseCrossTargetRuntimeConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-cross-target-runtime-config'

    config_id: str
    operation: SelectiveDependenceResponseCrossTargetOperation
    capability_key: str
    capability_version: str
    method_question: ObjectIdentity
    bindings: tuple[SelectiveDependenceResponseCrossTargetEnvelopeBinding, ...]
    maximum_handoff_bytes: int
    native_numeric_value_count: int

    def __post_init__(self) -> None:
        for name in ("config_id", "capability_key"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.capability_version)
        require_sorted_unique_ids(self.bindings, attribute="target_id", field_name="bindings")
        if len(self.bindings) != 2:
            raise ValueError("cross-target config requires two completion envelopes")
        if len({value.completion_envelope.object_id for value in self.bindings}) != 2:
            raise ValueError("cross-target config completion identities repeat")
        if self.method_question.object_schema != SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA:
            raise ValueError("cross-target config must bind the frozen selective-response method question")
        if self.capability_key != cross_target_capability_key(self.operation):
            raise ValueError("cross-target config operation differs")
        if self.capability_version != SELECTIVE_DEPENDENCE_RESPONSE_CROSS_TARGET_VERSION:
            raise ValueError("cross-target config version differs")
        if not 0 < self.maximum_handoff_bytes <= 2 * 1024**2:
            raise ValueError("cross-target handoff byte ceiling differs")
        if self.native_numeric_value_count:
            raise ValueError("cross-target config cannot contain native numeric values")


def cross_target_runtime_config(
    completion_envelopes: tuple[SelectiveDependenceResponseTargetCompletionEnvelope, SelectiveDependenceResponseTargetCompletionEnvelope],
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
    operation: SelectiveDependenceResponseCrossTargetOperation,
) -> SelectiveDependenceResponseCrossTargetRuntimeConfig:
    ordered = tuple(sorted(completion_envelopes, key=lambda value: value.target_id))
    if len({value.target_id for value in ordered}) != 2:
        raise ValueError("cross-target config target identities repeat")
    return SelectiveDependenceResponseCrossTargetRuntimeConfig(
        config_id=f"selective-dependence-response.cross-target.{operation.value.lower().replace('_', '-')}.config",
        operation=operation,
        capability_key=cross_target_capability_key(operation),
        capability_version=SELECTIVE_DEPENDENCE_RESPONSE_CROSS_TARGET_VERSION,
        method_question=ObjectIdentity.from_record(
            method_question.freeze_id,
            method_question,
        ),
        bindings=tuple(
            SelectiveDependenceResponseCrossTargetEnvelopeBinding(
                target_id=value.target_id,
                completion_envelope=ObjectIdentity.from_record(value.envelope_id, value),
            )
            for value in ordered
        ),
        maximum_handoff_bytes=2 * 1024**2,
        native_numeric_value_count=0,
    )


def decode_cross_target_runtime_config(payload: bytes) -> SelectiveDependenceResponseCrossTargetRuntimeConfig:
    return decode_canonical_bytes(
        payload, SelectiveDependenceResponseCrossTargetRuntimeConfig, maximum_bytes=256 * 1024
    )


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=256 * 1024**2,
        gpu_devices=0,
        wall_time_seconds=60,
        source_scan_bytes=8 * 1024**2,
        output_bytes=4 * 1024**2,
    )


def cross_target_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    manifests = []
    output_by_operation = {
        SelectiveDependenceResponseCrossTargetOperation.VERIFY_ELIGIBILITY: SelectiveDependenceResponseCrossTargetEligibility.SCHEMA,
        SelectiveDependenceResponseCrossTargetOperation.ADJUDICATE_RELATION: SelectiveDependenceResponseCrossTargetAdjudication.SCHEMA,
    }
    kind_by_operation = {
        SelectiveDependenceResponseCrossTargetOperation.VERIFY_ELIGIBILITY: CapabilityKind.ANALYSIS,
        SelectiveDependenceResponseCrossTargetOperation.ADJUDICATE_RELATION: CapabilityKind.HYPOTHESIS_SYNTHESIZER,
    }
    for operation in SelectiveDependenceResponseCrossTargetOperation:
        input_schemas = [
            SelectiveDependenceResponseCrossTargetRuntimeConfig.SCHEMA,
            SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA,
            SelectiveDependenceResponseTargetCompletionEnvelope.SCHEMA,
        ]
        if operation is SelectiveDependenceResponseCrossTargetOperation.ADJUDICATE_RELATION:
            input_schemas.append(SelectiveDependenceResponseCrossTargetEligibility.SCHEMA)
        manifests.append(
            CapabilityManifest(
                capability_key=cross_target_capability_key(operation),
                capability_version=SELECTIVE_DEPENDENCE_RESPONSE_CROSS_TARGET_VERSION,
                kind=kind_by_operation[operation],
                config_schema=SelectiveDependenceResponseCrossTargetRuntimeConfig.SCHEMA,
                config_schema_sha256=sha256(
                    SelectiveDependenceResponseCrossTargetRuntimeConfig.SCHEMA.encode("ascii")
                ).hexdigest(),
                input_schema_ids=tuple(sorted(input_schemas)),
                output_schema_ids=(output_by_operation[operation],),
                permissions=tuple(
                    sorted(
                        (
                            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                            CapabilityPermission.READ_OUTCOME_VISIBLE,
                        )
                    )
                ),
                maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
                maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                resource_ceiling=_budget(),
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id="cpython-3.11-selective-dependence-response-cross-target",
                requires_clean_commit=True,
                requires_active_mount=True,
                requires_network=False,
                conformance_check_ids=(
                    "completed-compact-handoffs-only",
                    "exact-two-target-eligibility",
                    "no-cross-target-native-pooling",
                    "unevaluable-before-counterexample-precedence",
                ),
                implementation_sha256=implementation_sha256,
            )
        )
    return CapabilityRegistry(
        registry_id="selective-dependence-response-cross-target-runtime",
        capabilities=tuple(sorted(manifests, key=lambda value: value.registry_id)),
    )


def cross_target_candidate_registrations(
    *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key="selective-dependence-response.cross-target-provider",
            provider_version=SELECTIVE_DEPENDENCE_RESPONSE_CROSS_TARGET_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=256 * 1024,
        )
        for manifest in cross_target_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


def build_cross_target_protocol(
    *,
    registry: CapabilityRegistry,
    config_by_operation: dict[SelectiveDependenceResponseCrossTargetOperation, CapabilityConfigRef],
) -> ProtocolTemplate:
    if set(config_by_operation) != set(SelectiveDependenceResponseCrossTargetOperation):
        raise ValueError("cross-target config roster differs")
    values = []
    for operation in SelectiveDependenceResponseCrossTargetOperation:
        manifest = registry.resolve(
            cross_target_capability_key(operation), SELECTIVE_DEPENDENCE_RESPONSE_CROSS_TARGET_VERSION
        )
        config = config_by_operation[operation]
        if (
            config.config_schema != manifest.config_schema
            or config.config_schema_sha256 != manifest.config_schema_sha256
        ):
            raise ValueError("cross-target config reference differs")
        eligibility = operation is SelectiveDependenceResponseCrossTargetOperation.VERIFY_ELIGIBILITY
        step_id = (
            "verify-cross-target-eligibility" if eligibility else "adjudicate-cross-target-relation"
        )
        output_schema = (
            SelectiveDependenceResponseCrossTargetEligibility.SCHEMA
            if eligibility
            else SelectiveDependenceResponseCrossTargetAdjudication.SCHEMA
        )
        values.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=ScientificStage.SYNTHESIZE,
                capability_key=manifest.capability_key,
                capability_version=manifest.capability_version,
                config=config,
                dependency_step_ids=(() if eligibility else ("verify-cross-target-eligibility",)),
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
                requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                resource_budget=manifest.resource_ceiling,
                resource_lock_ids=(step_id,),
                barrier=BarrierKind.NONE,
                maximum_attempts=2,
                obligation_ids=(f"selective-dependence-response-{step_id}-compact-contract",),
            )
        )
    return ProtocolTemplate(
        template_id="selective-dependence-response-cross-target-protocol",
        template_version=SELECTIVE_DEPENDENCE_RESPONSE_CROSS_TARGET_VERSION,
        steps=tuple(sorted(values, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def cross_target_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    completion_envelopes: tuple[SelectiveDependenceResponseTargetCompletionEnvelope, SelectiveDependenceResponseTargetCompletionEnvelope],
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
) -> CandidateScientificGraph:
    by_step = {value.step_id: value for value in protocol.steps}
    ordered = tuple(sorted(completion_envelopes, key=lambda value: value.target_id))
    completion_inputs = tuple(
        CandidateGraphExternalInput(
            input_id=f"input.selective-dependence-response.{value.target_id}",
            scientific_role=ScientificInputRole.PARENT_RECEIPT,
            logical_artifact_id=value.envelope_id,
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=value.fingerprint(),
            payload_schema=SelectiveDependenceResponseTargetCompletionEnvelope.SCHEMA,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_size_bytes=2 * 1024**2,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        )
        for value in ordered
    )
    method_input = CandidateGraphExternalInput(
        input_id="input.selective-dependence-response.cross-target-method-question",
        scientific_role=ScientificInputRole.MODEL,
        logical_artifact_id=method_question.freeze_id,
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=method_question.fingerprint(),
        payload_schema=SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=2 * 1024**2,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    inputs = tuple(sorted((*completion_inputs, method_input), key=lambda value: value.input_id))
    nodes = tuple(
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
    )
    edges = []
    for target in ("verify-cross-target-eligibility", "adjudicate-cross-target-relation"):
        step = by_step[target]
        for external in inputs:
            edges.append(
                CandidateGraphEdge(
                    edge_id=f"edge.{external.input_id}.{target}",
                    producer_node_id=None,
                    producer_output_id=None,
                    external_input_id=external.input_id,
                    consumer_node_id=target,
                    consumer_input_id=f"{external.input_id}.{target}",
                    scientific_role=external.scientific_role,
                    logical_artifact_id=external.logical_artifact_id,
                    payload_schema=external.payload_schema,
                    media_type=external.media_type,
                    maximum_size_bytes=external.maximum_size_bytes,
                    outcome_access=external.outcome_access,
                    visibility_ceiling=external.visibility_ceiling,
                    barrier=step.barrier,
                )
            )
    eligibility = by_step["verify-cross-target-eligibility"]
    adjudication = by_step["adjudicate-cross-target-relation"]
    edges.append(
        CandidateGraphEdge(
            edge_id="edge.verify-cross-target-eligibility.adjudicate-cross-target-relation",
            producer_node_id=eligibility.step_id,
            producer_output_id=eligibility.outputs[0].output_id,
            external_input_id=None,
            consumer_node_id=adjudication.step_id,
            consumer_input_id="cross-target-eligibility",
            scientific_role=ScientificInputRole.QUALIFICATION,
            logical_artifact_id="artifact.selective-dependence-response.cross-target-eligibility",
            payload_schema=SelectiveDependenceResponseCrossTargetEligibility.SCHEMA,
            media_type=eligibility.outputs[0].media_type,
            maximum_size_bytes=eligibility.resource_budget.output_bytes,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.NONE,
        )
    )
    return CandidateScientificGraph(
        graph_id="graph.selective-dependence-response.cross-target",
        external_inputs=inputs,
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def cross_target_candidate_catalog(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    completion_envelopes: tuple[SelectiveDependenceResponseTargetCompletionEnvelope, SelectiveDependenceResponseTargetCompletionEnvelope],
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
) -> CandidateCapabilityCatalog:
    graph = cross_target_scientific_graph(
        protocol=protocol,
        registry=registry,
        completion_envelopes=completion_envelopes,
        method_question=method_question,
    )
    incoming = {
        node.node_id: tuple(
            edge.edge_id for edge in graph.edges if edge.consumer_node_id == node.node_id
        )
        for node in graph.nodes
    }
    return CandidateCapabilityCatalog(
        catalog_id="selective-dependence-response-cross-target-candidate-catalog",
        registrations=cross_target_candidate_registrations(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        ),
        templates=(
            StudyTemplate(
                template_key="selective-dependence-response.cross-target",
                template_version=SELECTIVE_DEPENDENCE_RESPONSE_CROSS_TARGET_VERSION,
                protocol=protocol,
                graph=graph,
                coverage=ObligationCoverage(
                    coverage_id="coverage.selective-dependence-response.cross-target",
                    bindings=tuple(
                        ObligationCoverageBinding(
                            obligation_id=obligation,
                            proof_owner_node_id=step.step_id,
                            required_output_id=step.outputs[0].output_id,
                            contributor_edge_ids=tuple(sorted(incoming[step.step_id])),
                        )
                        for step in protocol.steps
                        for obligation in step.obligation_ids
                    ),
                ),
            ),
        ),
    )


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseCrossTargetConfigAdapter:
    provider_key: str = "selective-dependence-response.cross-target-provider"
    provider_version: str = SELECTIVE_DEPENDENCE_RESPONSE_CROSS_TARGET_VERSION

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != SelectiveDependenceResponseCrossTargetRuntimeConfig.SCHEMA:
            raise ValueError("cross-target config adapter schema differs")
        decode_cross_target_runtime_config(payload)


def cross_target_config_decoders() -> tuple[CandidateCapabilityConfigDecoder, ...]:
    return (SelectiveDependenceResponseCrossTargetConfigAdapter(),)


def _configs(context: TaskContext) -> tuple[SelectiveDependenceResponseCrossTargetRuntimeConfig, ...]:
    return tuple(
        decode_cross_target_runtime_config(port.read())
        for port in context.input_ports
        if port.payload_schema == SelectiveDependenceResponseCrossTargetRuntimeConfig.SCHEMA
    )


def _completion_envelopes(
    context: TaskContext,
) -> tuple[SelectiveDependenceResponseTargetCompletionEnvelope, SelectiveDependenceResponseTargetCompletionEnvelope]:
    values = tuple(
        decode_canonical_bytes(
            port.read(),
            SelectiveDependenceResponseTargetCompletionEnvelope,
            maximum_bytes=port.size_bytes,
        )
        for port in context.input_ports
        if port.payload_schema == SelectiveDependenceResponseTargetCompletionEnvelope.SCHEMA
    )
    if len(values) != 2:
        raise ValueError("cross-target task requires two completion envelopes")
    return values


def _method_question(context: TaskContext) -> SelectiveDependenceResponseMethodQuestionFreeze:
    values = tuple(
        decode_canonical_bytes(
            port.read(),
            SelectiveDependenceResponseMethodQuestionFreeze,
            maximum_bytes=port.size_bytes,
        )
        for port in context.input_ports
        if port.payload_schema == SelectiveDependenceResponseMethodQuestionFreeze.SCHEMA
    )
    if len(values) != 1:
        raise ValueError("cross-target task requires one frozen method question")
    return values[0]


def _verify_config_envelopes(
    config: SelectiveDependenceResponseCrossTargetRuntimeConfig,
    completion_envelopes: tuple[SelectiveDependenceResponseTargetCompletionEnvelope, SelectiveDependenceResponseTargetCompletionEnvelope],
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
) -> None:
    expected = {value.target_id: value.completion_envelope for value in config.bindings}
    observed = {
        value.target_id: ObjectIdentity.from_record(value.envelope_id, value)
        for value in completion_envelopes
    }
    if observed != expected:
        raise ValueError("cross-target completion envelopes differ from the frozen config")
    if config.method_question != ObjectIdentity.from_record(
        method_question.freeze_id,
        method_question,
    ):
        raise ValueError("cross-target method question differs from the frozen config")


def _handoffs_from_envelopes(
    completion_envelopes: tuple[SelectiveDependenceResponseTargetCompletionEnvelope, SelectiveDependenceResponseTargetCompletionEnvelope],
) -> tuple[SelectiveDependenceResponseTargetHandoff, SelectiveDependenceResponseTargetHandoff]:
    return completion_envelopes[0].handoff, completion_envelopes[1].handoff


class _EligibilityRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        configs = _configs(context)
        completion_envelopes = _completion_envelopes(context)
        method_question = _method_question(context)
        if len(configs) != 1 or configs[0].operation is not (
            SelectiveDependenceResponseCrossTargetOperation.VERIFY_ELIGIBILITY
        ):
            raise ValueError("eligibility task config differs")
        _verify_config_envelopes(configs[0], completion_envelopes, method_question)
        handoffs = _handoffs_from_envelopes(completion_envelopes)
        result = qualify_cross_target(handoffs)
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=result.canonical_bytes(),
                ),
            ),
            checks=(
                ReceiptCheck("selective-dependence-response-two-compact-handoffs", True, ()),
                ReceiptCheck(
                    "selective-dependence-response-two-targets-eligible",
                    result.eligible,
                    result.reason_codes,
                ),
            ),
        )


class _AdjudicationRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        configs = _configs(context)
        completion_envelopes = _completion_envelopes(context)
        method_question = _method_question(context)
        handoffs = _handoffs_from_envelopes(completion_envelopes)
        eligibility_values = tuple(
            decode_canonical_bytes(
                port.read(),
                SelectiveDependenceResponseCrossTargetEligibility,
                maximum_bytes=port.size_bytes,
            )
            for port in context.input_ports
            if port.payload_schema == SelectiveDependenceResponseCrossTargetEligibility.SCHEMA
        )
        if (
            len(configs) != 1
            or configs[0].operation is not SelectiveDependenceResponseCrossTargetOperation.ADJUDICATE_RELATION
            or len(eligibility_values) != 1
        ):
            raise ValueError("cross-target adjudication input roster differs")
        _verify_config_envelopes(configs[0], completion_envelopes, method_question)
        expected_eligibility = qualify_cross_target(handoffs)
        if eligibility_values[0] != expected_eligibility:
            raise ValueError("cross-target eligibility dependency differs")
        result = adjudicate_cross_target(handoffs)
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=context.output_ports[0].output_id,
                    payload=result.canonical_bytes(),
                ),
            ),
            checks=(
                ReceiptCheck("selective-dependence-response-cross-target-nonpooling", True, ()),
                ReceiptCheck("selective-dependence-response-cross-target-terminal", True, ()),
            ),
        )


class SelectiveDependenceResponseCrossTargetRuntimeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        configs: tuple[SelectiveDependenceResponseCrossTargetRuntimeConfig, ...],
        completion_envelopes: tuple[
            SelectiveDependenceResponseTargetCompletionEnvelope, SelectiveDependenceResponseTargetCompletionEnvelope
        ],
        method_question: SelectiveDependenceResponseMethodQuestionFreeze,
    ) -> None:
        expected_configs = {
            operation: cross_target_runtime_config(
                completion_envelopes,
                method_question,
                operation,
            )
            for operation in SelectiveDependenceResponseCrossTargetOperation
        }
        if {value.operation: value for value in configs} != expected_configs:
            raise ValueError("cross-target provider configs differ")
        if registry != cross_target_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        ):
            raise ValueError("cross-target provider registry differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.configs = tuple(sorted(configs, key=lambda value: value.config_id))
        self.completion_envelopes = tuple(
            sorted(completion_envelopes, key=lambda value: value.target_id)
        )
        self.method_question = method_question
        runners = []
        for manifest in registry.capabilities:
            if manifest.capability_key == cross_target_capability_key(
                SelectiveDependenceResponseCrossTargetOperation.VERIFY_ELIGIBILITY
            ):
                runner: TaskRunner = _EligibilityRunner(manifest)
            else:
                runner = _AdjudicationRunner(manifest)
            runners.append(runner)
        self._runners = tuple(runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("cross-target provider registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("cross-target provider plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            value.envelope_id: value for value in self.completion_envelopes
        }
        records[self.method_question.freeze_id] = self.method_question
        configs_by_key = {value.capability_key: value for value in self.configs}
        for task in plan.tasks:
            records[task.capability.config.artifact_id] = configs_by_key[
                task.capability.capability_key
            ]
        if set(records) != set(specs):
            raise ValueError("cross-target plan external input roster differs")
        values = []
        for artifact_id, record in sorted(records.items()):
            spec = specs[artifact_id]
            is_config = isinstance(record, SelectiveDependenceResponseCrossTargetRuntimeConfig)
            is_question = isinstance(record, SelectiveDependenceResponseMethodQuestionFreeze)
            visibility = (
                VisibilityCeiling.PROSPECTIVE
                if is_config or is_question
                else VisibilityCeiling.OUTCOME_VISIBLE
            )
            access = (
                OutcomeAccess.OUTCOME_BLIND
                if is_config or is_question
                else OutcomeAccess.EVALUATION_REVEALED
            )
            if isinstance(record, SelectiveDependenceResponseCrossTargetRuntimeConfig):
                record_id = record.config_id
            elif isinstance(record, SelectiveDependenceResponseTargetCompletionEnvelope):
                record_id = record.envelope_id
            elif isinstance(record, SelectiveDependenceResponseMethodQuestionFreeze):
                record_id = record.freeze_id
            else:
                raise TypeError("cross-target external input type differs")
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
                    visibility_ceiling=spec.expected_visibility_ceiling or visibility,
                    outcome_access=spec.expected_outcome_access or access,
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
            raise ValueError("cross-target semantic registry differs")
        types = {
            SelectiveDependenceResponseCrossTargetEligibility.SCHEMA: SelectiveDependenceResponseCrossTargetEligibility,
            SelectiveDependenceResponseCrossTargetAdjudication.SCHEMA: SelectiveDependenceResponseCrossTargetAdjudication,
        }
        values = []
        for manifest in registry.capabilities:
            schema = manifest.output_schema_ids[0]
            record_type = types[schema]
            values.append(
                CapabilityOutputSemanticContract.from_manifest(
                    manifest,
                    payload_schema=schema,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    top_level_keys=("schema", "value", "version"),
                    value_keys=tuple(sorted(value.name for value in fields(record_type))),
                )
            )
        return tuple(sorted(values, key=lambda value: value.capability_key))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("cross-target adjudication registry differs")
        return None


__all__ = [
    'SelectiveDependenceResponseCrossTargetOperation',
    'SelectiveDependenceResponseCrossTargetEnvelopeBinding',
    'SelectiveDependenceResponseCrossTargetRuntimeConfig',
    'SelectiveDependenceResponseCrossTargetRuntimeProvider',
    "build_cross_target_protocol",
    "cross_target_candidate_catalog",
    "cross_target_config_decoders",
    "cross_target_registry",
    "cross_target_runtime_config",
    "cross_target_scientific_graph",
    "decode_cross_target_runtime_config",
]

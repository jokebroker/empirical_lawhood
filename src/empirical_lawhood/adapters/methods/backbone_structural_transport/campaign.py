"""Production-owned bounded structural-transport campaign topology.

The records in this module select data and scientific identities only.  The
protocol/graph builders are code-owned and deterministic; factories and target
ports remain in :mod:`provider`/``executable_binding``.  In particular, target
outcomes are absent from provider construction and become readable only when
the reveal-barrier target task executes.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods.structural_bootstrap_inputs import StructuralBootstrapInputCensus, require_structural_bootstrap_census
from empirical_lawhood.adapters.methods.prospective_structural_recurrence import StructuralRecurrenceFrozenLawTransportForecasts, ProspectiveStructuralRecurrenceAdjudication, ProspectiveStructuralRecurrenceFaceTerminal, ProspectiveStructuralRecurrenceMethodSpec, ProspectiveStructuralRecurrencePlan, ProspectiveStructuralRecurrenceRosterIssue, ProspectiveStructuralRecurrenceTargetObservation, ProspectiveStructuralRecurrenceTargetTerminal
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetStage
from empirical_lawhood.adapters.methods.structural_recurrence import StructuralObservation, StructuralPredictionInput
from empirical_lawhood.adapters.methods.interval_property_comparison import PropertyComparisonKind, IntervalPropertyComparisonMethodSpec, IntervalPropertyComparisonOperands, IntervalPropertyComparisonResult
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_compiler import (
    CandidateGraphEdge,
    CandidateGraphExternalInput,
    CandidateGraphNode,
    CandidateScientificGraph,
    ContentIdentityPolicy,
    ScientificInputRole,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.law_transport_handoff import LAW_TRANSPORT_INPUT_IDS
from empirical_lawhood.runtime.response_experiment import ResponseStageTerminal
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    OutputTemplate,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)


STRUCTURAL_CAMPAIGN_MEDIA_TYPE = "application/vnd.empirical-lawhood.canonical+json"
STRUCTURAL_TARGET_SOURCE_PORT_KEY = "structural-transport-target-source"
STRUCTURAL_MAX_FACE_FAN_IN = 1 + len(PropertyComparisonKind)


def _identity(record_id: str, record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(record_id, record)


def _schema_sha256(schema: str) -> str:
    return sha256(schema.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class StructuralDevelopmentInputs(CanonicalRecord):
    'Authenticated development-only inputs for one frozen structural recurrence plan.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/structural-development-inputs'

    inputs_id: str
    plan: ProspectiveStructuralRecurrencePlan
    development_evidence: tuple[StructuralRecurrenceStageEvidence, ...]
    prediction_inputs: tuple[StructuralPredictionInput, ...]
    scientific_bootstrap_inputs: StructuralBootstrapInputCensus
    target_outcome_access_count: int
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.inputs_id, field_name="inputs_id")
        require_structural_bootstrap_census(self.scientific_bootstrap_inputs)
        require_sorted_unique_ids(
            self.development_evidence,
            attribute="target_slot_id",
            field_name="development_evidence",
        )
        targets = tuple(
            sorted(value.target_slot.target_slot_id for value in self.plan.target_designs)
        )
        if len(targets) != 1:
            raise ValueError("structural campaign requires exactly one shared target slot")
        evidence_targets = tuple(value.target_slot_id for value in self.development_evidence)
        prediction_targets = tuple(
            sorted(value.target_slot.target_slot_id for value in self.prediction_inputs)
        )
        if len(set(prediction_targets)) != len(prediction_targets):
            raise ValueError("structural development inputs repeat a prediction target")
        if evidence_targets != targets or prediction_targets != targets:
            raise ValueError("structural development inputs differ from the plan target roster")
        if any(
            value.stage is not StructuralRecurrenceTargetStage.DEVELOPMENT for value in self.development_evidence
        ):
            raise ValueError("structural development inputs contain target-stage evidence")
        if any(
            value.outcome_access is OutcomeAccess.EVALUATION_REVEALED
            for value in self.prediction_inputs
        ):
            raise ValueError("structural prediction inputs cross the evaluation cutoff")
        if self.target_outcome_access_count or self.grants_authority:
            raise ValueError("structural development inputs contain target access or authority")


@dataclass(frozen=True, slots=True)
class StructuralDevelopmentBridgeConfig(CanonicalRecord):
    """Outcome-blind selection of one authenticated development bundle."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/structural-development-bridge-config'

    config_id: str
    development_inputs: ObjectIdentity
    plan: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.development_inputs.object_schema != StructuralDevelopmentInputs.SCHEMA
            or self.plan.object_schema != ProspectiveStructuralRecurrencePlan.SCHEMA
        ):
            raise ValueError("development bridge config binds another input/plan schema")


@dataclass(frozen=True, slots=True)
class StructuralPredictionFreezeReceipt(CanonicalRecord):
    """Outcome-blind issue receipt used to admit later target contact."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/structural-prediction-freeze-receipt'

    receipt_id: str
    roster_issue: ObjectIdentity
    plan: ObjectIdentity
    method_spec: ObjectIdentity
    issued_before_target_contact: bool
    target_contact_count: int
    target_outcome_access_count: int
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (
            self.roster_issue.object_schema != ProspectiveStructuralRecurrenceRosterIssue.SCHEMA
            or self.plan.object_schema != ProspectiveStructuralRecurrencePlan.SCHEMA
            or self.method_spec.object_schema != ProspectiveStructuralRecurrenceMethodSpec.SCHEMA
        ):
            raise ValueError("prediction freeze receipt binds another scientific schema")
        if (
            not self.issued_before_target_contact
            or self.target_contact_count
            or self.target_outcome_access_count
            or self.grants_authority
        ):
            raise ValueError("prediction freeze receipt crossed target contact or authority")


@dataclass(frozen=True, slots=True)
class StructuralRecurrenceTransportConfig(CanonicalRecord):
    'Issued wrapper selecting structural recurrence for an authenticated donor-law transport.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/structural-recurrence-transport-config'

    config_id: str
    method_spec: ProspectiveStructuralRecurrenceMethodSpec
    plan: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.plan.object_schema != ProspectiveStructuralRecurrencePlan.SCHEMA:
            raise ValueError('structural recurrence transport config binds another plan schema')


@dataclass(frozen=True, slots=True)
class StructuralTargetBridgeConfig(CanonicalRecord):
    """Issued request for target evidence after one exact prediction freeze."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/structural-target-bridge-config'

    config_id: str
    source_request_id: str
    plan: ObjectIdentity
    target_slot_id: str
    comparison_kinds: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("config_id", "source_request_id", "target_slot_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.plan.object_schema != ProspectiveStructuralRecurrencePlan.SCHEMA:
            raise ValueError("target bridge config binds another plan schema")
        require_sorted_unique_strings(
            self.comparison_kinds,
            field_name="comparison_kinds",
            allow_empty=False,
        )
        if self.comparison_kinds != tuple(
            sorted(value.value for value in PropertyComparisonKind)
        ):
            raise ValueError("target bridge config lacks the complete eight-kind comparison roster")


@dataclass(frozen=True, slots=True)
class StructuralTargetInputs(CanonicalRecord):
    """One target-source response, created only inside the revealed target task."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/structural-target-inputs'

    inputs_id: str
    source_request_id: str
    roster_issue: ObjectIdentity
    evaluation_evidence: StructuralRecurrenceStageEvidence
    categorical_observation: StructuralObservation
    property_operands: tuple[IntervalPropertyComparisonOperands, ...]
    source_contact_count: int
    outcome_access: OutcomeAccess
    grants_authority: bool

    def __post_init__(self) -> None:
        for name in ("inputs_id", "source_request_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.roster_issue.object_schema != ProspectiveStructuralRecurrenceRosterIssue.SCHEMA:
            raise ValueError("structural target inputs bind another prediction roster")
        require_sorted_unique_ids(
            self.property_operands,
            attribute="operands_id",
            field_name="property_operands",
        )
        kinds = tuple(sorted(value.comparison_kind.value for value in self.property_operands))
        if kinds != tuple(sorted(value.value for value in PropertyComparisonKind)):
            raise ValueError("structural target inputs lack the complete eight-kind operands")
        target_id = self.evaluation_evidence.target_slot_id
        if any(value.target_member_id != target_id for value in self.property_operands):
            raise ValueError("structural target operands cross target slots")
        if (
            self.evaluation_evidence.stage is not StructuralRecurrenceTargetStage.EVALUATION
            or self.source_contact_count != 1
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.grants_authority
        ):
            raise ValueError("structural target inputs have invalid contact/cutoff semantics")


@dataclass(frozen=True, slots=True)
class StructuralReporterConfig(CanonicalRecord):
    'Closed mapping from structural recurrence terminal status to the standard run finalizer.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/backbone-structural-transport/structural-reporter-config'

    config_id: str
    method_spec: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.method_spec.object_schema != ProspectiveStructuralRecurrenceMethodSpec.SCHEMA:
            raise ValueError("structural reporter config binds another method schema")


@dataclass(frozen=True, slots=True)
class StructuralTransportCampaignCapabilities:
    """Noncanonical code-owned manifest bundle used by the topology builder."""

    development: CapabilityManifest
    target: CapabilityManifest
    property_comparison: CapabilityManifest
    structural_recurrence: CapabilityManifest
    reporter: CapabilityManifest

    def registry(self, registry_id: str) -> CapabilityRegistry:
        return CapabilityRegistry(
            registry_id=registry_id,
            capabilities=tuple(
                sorted(
                    (
                        self.development,
                        self.target,
                        self.property_comparison,
                        self.structural_recurrence,
                        self.reporter,
                    ),
                    key=lambda value: value.registry_id,
                )
            ),
        )


def _output(output_id: str, schema: str) -> OutputTemplate:
    return OutputTemplate(
        output_id=output_id,
        payload_schema=schema,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type=STRUCTURAL_CAMPAIGN_MEDIA_TYPE,
        filename_suffix=".canonical.json",
    )


def _step(
    *,
    step_id: str,
    manifest: CapabilityManifest,
    config: CanonicalRecord,
    dependencies: tuple[str, ...],
    outputs: tuple[OutputTemplate, ...],
    stage: ScientificStage,
    access: OutcomeAccess,
    visibility: VisibilityCeiling,
    barrier: BarrierKind,
) -> ProtocolStepTemplate:
    config_id = getattr(config, "spec_id", getattr(config, "config_id", None))
    if not isinstance(config_id, str):
        raise ValueError("structural campaign config lacks a stable ID")
    permissions = tuple(
        sorted(
            value
            for value in manifest.permissions
            if value
            in {
                CapabilityPermission.READ_DEVELOPMENT,
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            }
        )
    )
    return ProtocolStepTemplate(
        step_id=step_id,
        stage=stage,
        capability_key=manifest.capability_key,
        capability_version=manifest.capability_version,
        config=CapabilityConfigRef(
            config_id=f"config-ref.{step_id}",
            config_schema=config.SCHEMA,
            config_schema_sha256=_schema_sha256(config.SCHEMA),
            content_sha256=config.fingerprint(),
            artifact_id=f"config-artifact.{config_id}",
        ),
        dependency_step_ids=tuple(sorted(dependencies)),
        outputs=tuple(sorted(outputs, key=lambda value: value.output_id)),
        required_permissions=permissions,
        requested_outcome_access=access,
        visibility_ceiling=visibility,
        resource_budget=ResourceBudget(
            cpu_cores=1,
            memory_bytes=512_000_000,
            gpu_devices=0,
            wall_time_seconds=1,
            source_scan_bytes=1_000_000,
            output_bytes=1_000_000,
        ),
        resource_lock_ids=(),
        barrier=barrier,
        maximum_attempts=2,
        obligation_ids=(f"obligation.{step_id}",),
    )


def build_structural_transport_protocol(
    *,
    capabilities: StructuralTransportCampaignCapabilities,
    development_inputs: StructuralDevelopmentInputs,
    development_config: StructuralDevelopmentBridgeConfig,
    structural_recurrence_config: StructuralRecurrenceTransportConfig,
    target_config: StructuralTargetBridgeConfig,
    property_spec: IntervalPropertyComparisonMethodSpec,
    reporter_config: StructuralReporterConfig,
) -> ProtocolTemplate:
    """Build the installed bounded PC9F/PC9G campaign without target data."""

    plan = development_inputs.plan
    categorical_faces = tuple(
        value for value in plan.faces if value.predictive_level.value == "CATEGORICAL"
    )
    if (
        len(plan.faces) != STRUCTURAL_MAX_FACE_FAN_IN
        or len(categorical_faces) != 1
        or len({value.property_id for value in plan.faces}) != len(plan.faces)
    ):
        raise ValueError("structural campaign face fan-in differs from its bounded roster")
    if development_config.development_inputs != _identity(
        development_inputs.inputs_id,
        development_inputs,
    ) or development_config.plan != _identity(plan.plan_id, plan):
        raise ValueError("development config differs from its authenticated inputs")
    if (
        structural_recurrence_config.method_spec != plan.method_spec
        or structural_recurrence_config.plan != _identity(plan.plan_id, plan)
        or target_config.plan != _identity(plan.plan_id, plan)
        or target_config.target_slot_id
        not in {value.target_slot.target_slot_id for value in plan.target_designs}
        or reporter_config.method_spec != _identity(plan.method_spec.spec_id, plan.method_spec)
    ):
        raise ValueError("structural target/reporter config differs from the frozen plan")

    development = _step(
        step_id="b-development",
        manifest=capabilities.development,
        config=development_config,
        dependencies=(),
        outputs=(
            _output("development", StructuralRecurrenceStageEvidence.SCHEMA),
            _output("plan", ProspectiveStructuralRecurrencePlan.SCHEMA),
            _output("prediction", StructuralPredictionInput.SCHEMA),
            _output("bootstrap-inputs", StructuralBootstrapInputCensus.SCHEMA),
        ),
        stage=ScientificStage.DEVELOP,
        access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
        barrier=BarrierKind.NONE,
    )
    issue = _step(
        step_id="c-issue-and-freeze",
        manifest=capabilities.structural_recurrence,
        config=structural_recurrence_config,
        dependencies=(development.step_id,),
        outputs=(
            _output("prediction-freeze", StructuralPredictionFreezeReceipt.SCHEMA),
            _output("roster-issue", ProspectiveStructuralRecurrenceRosterIssue.SCHEMA),
        ),
        stage=ScientificStage.DEVELOP,
        access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
        barrier=BarrierKind.FREEZE,
    )
    target = _step(
        step_id="a-target",
        manifest=capabilities.target,
        config=target_config,
        dependencies=(issue.step_id,),
        outputs=(
            _output("categorical", StructuralObservation.SCHEMA),
            _output("evaluation", StructuralRecurrenceStageEvidence.SCHEMA),
            *(
                _output(f"operand.{value.value.lower()}", IntervalPropertyComparisonOperands.SCHEMA)
                for value in sorted(PropertyComparisonKind, key=lambda item: item.value)
            ),
        ),
        stage=ScientificStage.EVALUATE,
        access=OutcomeAccess.EVALUATION_REVEALED,
        visibility=VisibilityCeiling.OUTCOME_VISIBLE,
        barrier=BarrierKind.REVEAL,
    )
    comparisons = tuple(
        _step(
            step_id=f"d-compare-{kind.value.lower().replace('_', '-')}",
            manifest=capabilities.property_comparison,
            config=property_spec,
            dependencies=(target.step_id,),
            outputs=(_output("result", IntervalPropertyComparisonResult.SCHEMA),),
            stage=ScientificStage.FALSIFY,
            access=OutcomeAccess.EVALUATION_REVEALED,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.REVEAL,
        )
        for kind in sorted(PropertyComparisonKind, key=lambda value: value.value)
    )
    adjudication = _step(
        step_id="e-adjudicate",
        manifest=capabilities.structural_recurrence,
        config=structural_recurrence_config,
        dependencies=tuple(
            sorted((issue.step_id, target.step_id, *(value.step_id for value in comparisons)))
        ),
        outputs=(
            _output("adjudication", ProspectiveStructuralRecurrenceAdjudication.SCHEMA),
            *(
                _output(
                    f"observation.{face.face_id}",
                    ProspectiveStructuralRecurrenceTargetObservation.SCHEMA,
                )
                for face in plan.faces
            ),
            *(
                _output(
                    f"face-terminal.{face.face_id}",
                    ProspectiveStructuralRecurrenceFaceTerminal.SCHEMA,
                )
                for face in plan.faces
            ),
            *(
                _output(
                    f"target-terminal.{target_id}",
                    ProspectiveStructuralRecurrenceTargetTerminal.SCHEMA,
                )
                for target_id in sorted({value.target_slot_id for value in plan.faces})
            ),
        ),
        stage=ScientificStage.EVALUATE,
        access=OutcomeAccess.EVALUATION_REVEALED,
        visibility=VisibilityCeiling.OUTCOME_VISIBLE,
        barrier=BarrierKind.REVEAL,
    )
    report = _step(
        step_id="report",
        manifest=capabilities.reporter,
        config=reporter_config,
        dependencies=(adjudication.step_id,),
        outputs=(_output("report", ScientificAdjudicationRecord.SCHEMA),),
        stage=ScientificStage.REPORT,
        access=OutcomeAccess.EVALUATION_REVEALED,
        visibility=VisibilityCeiling.OUTCOME_VISIBLE,
        barrier=BarrierKind.REVEAL,
    )
    return ProtocolTemplate(
        template_id="protocol.structural-transport-public",
        template_version="1.0.0",
        steps=tuple(
            sorted(
                (development, issue, target, *comparisons, adjudication, report),
                key=lambda value: value.step_id,
            )
        ),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


_LAW_TRANSPORT_SCHEMAS = {
    'donor-measurement-through-law-qualification-terminal': ResponseStageTerminal.SCHEMA,
    "frozen-response-law": ResponseLaw.SCHEMA,
    "forecast-method-spec": ProspectiveStructuralRecurrenceMethodSpec.SCHEMA,
    "forecast-method-config": ProspectiveStructuralRecurrencePlan.SCHEMA,
    "frozen-property-forecasts": StructuralRecurrenceFrozenLawTransportForecasts.SCHEMA,
}


def build_structural_transport_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    development_inputs: StructuralDevelopmentInputs,
    source_input_id: str,
) -> CandidateScientificGraph:
    'Build the exact production graph with parent handoff into structural recurrence issue.'

    validate_stable_id(source_input_id, field_name="source_input_id")
    steps = {value.step_id: value for value in protocol.steps}
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
    source = CandidateGraphExternalInput(
        input_id=source_input_id,
        scientific_role=ScientificInputRole.PREPARED_MEDIUM,
        logical_artifact_id=f"artifact.{development_inputs.inputs_id}",
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=development_inputs.fingerprint(),
        payload_schema=StructuralDevelopmentInputs.SCHEMA,
        media_type=STRUCTURAL_CAMPAIGN_MEDIA_TYPE,
        maximum_size_bytes=8_000_000,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    parents = tuple(
        CandidateGraphExternalInput(
            input_id=input_id,
            scientific_role=ScientificInputRole.PARENT_RECEIPT,
            logical_artifact_id=f"artifact.structural-handoff.{input_id}",
            content_identity_policy=ContentIdentityPolicy.PARENT_RECEIPT_SUBSTITUTION,
            expected_content_sha256=None,
            payload_schema=_LAW_TRANSPORT_SCHEMAS[input_id],
            media_type=STRUCTURAL_CAMPAIGN_MEDIA_TYPE,
            maximum_size_bytes=8_000_000,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        )
        for input_id in LAW_TRANSPORT_INPUT_IDS
    )
    edges: list[CandidateGraphEdge] = [
        CandidateGraphEdge(
            edge_id="edge.structural-development-source",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=source_input_id,
            consumer_node_id="b-development",
            consumer_input_id="development-inputs",
            scientific_role=ScientificInputRole.PREPARED_MEDIUM,
            logical_artifact_id=source.logical_artifact_id,
            payload_schema=source.payload_schema,
            media_type=source.media_type,
            maximum_size_bytes=source.maximum_size_bytes,
            outcome_access=source.outcome_access,
            visibility_ceiling=source.visibility_ceiling,
            barrier=BarrierKind.NONE,
        )
    ]
    edges.extend(
        CandidateGraphEdge(
            edge_id=f"edge.structural-handoff.{parent.input_id}.issue",
            producer_node_id=None,
            producer_output_id=None,
            external_input_id=parent.input_id,
            consumer_node_id="c-issue-and-freeze",
            consumer_input_id=parent.input_id,
            scientific_role=parent.scientific_role,
            logical_artifact_id=parent.logical_artifact_id,
            payload_schema=parent.payload_schema,
            media_type=parent.media_type,
            maximum_size_bytes=parent.maximum_size_bytes,
            outcome_access=parent.outcome_access,
            visibility_ceiling=parent.visibility_ceiling,
            barrier=BarrierKind.NONE,
        )
        for parent in parents
    )

    def connect(
        producer: str,
        output_id: str,
        consumer: str,
        input_id: str,
        role: ScientificInputRole,
    ) -> None:
        output = next(value for value in steps[producer].outputs if value.output_id == output_id)
        revealed = steps[producer].requested_outcome_access is OutcomeAccess.EVALUATION_REVEALED
        edges.append(
            CandidateGraphEdge(
                edge_id=f"edge.{producer}.{output_id}.{consumer}.{input_id}",
                producer_node_id=producer,
                producer_output_id=output_id,
                external_input_id=None,
                consumer_node_id=consumer,
                consumer_input_id=input_id,
                scientific_role=role,
                logical_artifact_id=f"artifact.{producer}.{output_id}",
                payload_schema=output.payload_schema,
                media_type=output.media_type,
                maximum_size_bytes=steps[producer].resource_budget.output_bytes,
                outcome_access=(
                    OutcomeAccess.EVALUATION_REVEALED
                    if revealed
                    else OutcomeAccess.DEVELOPMENT_VISIBLE
                ),
                visibility_ceiling=(
                    VisibilityCeiling.OUTCOME_VISIBLE
                    if revealed
                    else VisibilityCeiling.DEVELOPMENT_ONLY
                ),
                barrier=BarrierKind.REVEAL if revealed else BarrierKind.NONE,
            )
        )

    for output_id in ("development", "plan", "prediction", "bootstrap-inputs"):
        connect(
            "b-development",
            output_id,
            "c-issue-and-freeze",
            output_id,
            ScientificInputRole.MODEL,
        )
    connect(
        "c-issue-and-freeze",
        "prediction-freeze",
        "a-target",
        "prediction-freeze",
        ScientificInputRole.MODEL,
    )
    connect(
        "c-issue-and-freeze",
        "roster-issue",
        "e-adjudicate",
        "roster-issue",
        ScientificInputRole.MODEL,
    )
    for kind in sorted(PropertyComparisonKind, key=lambda value: value.value):
        token = kind.value.lower()
        compare = f"d-compare-{token.replace('_', '-')}"
        connect(
            "a-target",
            f"operand.{token}",
            compare,
            "operands",
            ScientificInputRole.OUTCOME,
        )
        connect(
            compare,
            "result",
            "e-adjudicate",
            f"result.{token}",
            ScientificInputRole.OUTCOME,
        )
    for output_id in ("categorical", "evaluation"):
        connect(
            "a-target",
            output_id,
            "e-adjudicate",
            output_id,
            ScientificInputRole.OUTCOME,
        )
    connect(
        "e-adjudicate",
        "adjudication",
        "report",
        'structural-recurrence-adjudication',
        ScientificInputRole.OUTCOME,
    )
    return CandidateScientificGraph(
        graph_id="graph.structural-transport-public",
        external_inputs=tuple(sorted((source, *parents), key=lambda value: value.input_id)),
        nodes=tuple(sorted(nodes, key=lambda value: value.node_id)),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


__all__ = [
    "STRUCTURAL_CAMPAIGN_MEDIA_TYPE",
    'STRUCTURAL_MAX_FACE_FAN_IN',
    "STRUCTURAL_TARGET_SOURCE_PORT_KEY",
    'StructuralDevelopmentBridgeConfig',
    'StructuralDevelopmentInputs',
    'StructuralPredictionFreezeReceipt',
    'StructuralRecurrenceTransportConfig',
    'StructuralReporterConfig',
    'StructuralTargetBridgeConfig',
    'StructuralTargetInputs',
    'StructuralTransportCampaignCapabilities',
    'build_structural_transport_graph',
    'build_structural_transport_protocol',
]

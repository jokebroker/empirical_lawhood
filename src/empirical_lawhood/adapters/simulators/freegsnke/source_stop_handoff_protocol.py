"Receipt-bound independent substrate grounding FreeGSNKE source-stop handoff DAG.\n\nThe source/action qualification is excluded from target evidence, but a\nterminal adverse result still needs one compact, canonical handoff for\ncross-target accounting.  This protocol binds the exact revealed result, its\nscientific adjudication and its durable reducer receipt.  It never reopens an\nepisode, invents a target unit, or converts source inapplicability into a\nrecurrence counterexample.\n"

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import ClassVar, Final

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateIndependenceClass, IndependentSubstrateTargetKind, IndependentSubstrateTargetTerminalHandoff
from empirical_lawhood.adapters.methods.structural_recurrence import TargetLevel
from empirical_lawhood.adapters.methods.observed_structural_classes import EvidenceWorld
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
from empirical_lawhood.kernel.status import AdmissionStatus, OperationalStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    CanonicalTaskReceipt,
    ReceiptCheck,
)
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

from .registry import FREEGSNKE_CAPABILITY_VERSION
from .source_qualification_protocol import FREEGSNKE_SOURCE_QUALIFICATION_STEP_ID
from .source_qualification import FreeGsnkeSourceQualificationDisposition, FreeGsnkeSourceQualificationResult


FREEGSNKE_SOURCE_STOP_PROJECT_KEY: Final = "freegsnke.project-source-stop-handoff"
FREEGSNKE_SOURCE_STOP_REPORT_KEY: Final = "freegsnke.report-source-stop-handoff"
FREEGSNKE_SOURCE_STOP_PROJECT_STEP_ID: Final = "project-source-stop-handoff"
FREEGSNKE_SOURCE_STOP_REPORT_STEP_ID: Final = "report-source-stop-handoff"
FREEGSNKE_SOURCE_STOP_PROVIDER_KEY: Final = "independent-substrate-grounding.freegsnke-source-stop-handoff-provider"


@dataclass(frozen=True, slots=True)
class FreeGsnkeSourceStopHandoffConfig(CanonicalRecord):
    "Exact immutable operands for a zero-unit handoff projection."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-source-stop-handoff-config'

    config_id: str
    capability_version: str
    source_qualification_result: ObjectIdentity
    source_scientific_adjudication: ObjectIdentity
    source_qualification_receipt: ObjectIdentity
    handoff_id: str
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_semantic_version(self.capability_version)
        validate_stable_id(self.handoff_id, field_name="handoff_id")
        if self.capability_version != FREEGSNKE_CAPABILITY_VERSION:
            raise ValueError("FreeGSNKE source-stop capability version differs")
        if (
            self.source_qualification_result.object_schema
            != FreeGsnkeSourceQualificationResult.SCHEMA
            or self.source_scientific_adjudication.object_schema
            != ScientificAdjudicationRecord.SCHEMA
            or self.source_qualification_receipt.object_schema != CanonicalTaskReceipt.SCHEMA
        ):
            raise ValueError("FreeGSNKE source-stop operand schema differs")
        if not 0 < self.maximum_input_bytes <= 64 * 1024**2:
            raise ValueError("FreeGSNKE source-stop input bound differs")


def _identity(record: CanonicalRecord) -> ObjectIdentity:
    for attribute in ("config_id", "result_id", "adjudication_id", "receipt_id", "handoff_id"):
        value = getattr(record, attribute, None)
        if isinstance(value, str):
            return ObjectIdentity.from_record(value, record)
    raise TypeError(f"unsupported FreeGSNKE source-stop record: {type(record)!r}")


def _receipt_output_matches(receipt: CanonicalTaskReceipt, record: CanonicalRecord) -> bool:
    return (
        sum(
            logical.payload_schema == record.SCHEMA
            and logical.content_sha256 == record.fingerprint()
            for logical in receipt.output_logical_artifacts
        )
        == 1
    )


def _validate_source_stop_operands(
    *,
    config: FreeGsnkeSourceStopHandoffConfig,
    result: FreeGsnkeSourceQualificationResult,
    source_adjudication: ScientificAdjudicationRecord,
    receipt: CanonicalTaskReceipt,
) -> None:
    if (
        _identity(result) != config.source_qualification_result
        or _identity(source_adjudication) != config.source_scientific_adjudication
        or _identity(receipt) != config.source_qualification_receipt
    ):
        raise ValueError("FreeGSNKE source-stop operand bytes differ")
    if (
        result.disposition is FreeGsnkeSourceQualificationDisposition.PASS
        or not result.excluded_from_target_evidence
        or result.target_evidence_unit_count
        or result.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
    ):
        raise ValueError("FreeGSNKE source-stop requires an excluded adverse result")
    if (
        receipt.task_id != FREEGSNKE_SOURCE_QUALIFICATION_STEP_ID
        or receipt.operational_status is not OperationalStatus.SUCCEEDED
        or receipt.run_id != source_adjudication.run_id
        or source_adjudication.adjudication_task_id != receipt.task_id
        or not _receipt_output_matches(receipt, result)
        or not _receipt_output_matches(receipt, source_adjudication)
        or source_adjudication.output_logical_artifact_ids
        != tuple(sorted(value.logical_artifact_id for value in receipt.output_logical_artifacts))
    ):
        raise ValueError("FreeGSNKE source-stop reducer receipt lineage differs")
    unevaluable = (
        result.disposition is FreeGsnkeSourceQualificationDisposition.OPERATIONAL_UNEVALUABLE
    )
    expected = (
        (
            AdjudicationEvaluability.UNEVALUABLE,
            ScientificStatus.UNEVALUABLE,
            AdmissionStatus.UNEVALUABLE,
        )
        if unevaluable
        else (
            AdjudicationEvaluability.EVALUABLE,
            ScientificStatus.NOT_SUPPORTED,
            AdmissionStatus.NOT_EVALUATED,
        )
    )
    if (
        source_adjudication.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
        or (
            source_adjudication.evaluability,
            source_adjudication.scientific_status,
            source_adjudication.admission_status,
        )
        != expected
        or f"FREEGSNKE_SOURCE_QUALIFICATION_{result.disposition.value}"
        not in source_adjudication.reason_codes
    ):
        raise ValueError("FreeGSNKE source-stop scientific adjudication differs")


def freegsnke_source_stop_handoff(
    *,
    config: FreeGsnkeSourceStopHandoffConfig,
    result: FreeGsnkeSourceQualificationResult,
    source_adjudication: ScientificAdjudicationRecord,
    receipt: CanonicalTaskReceipt,
) -> IndependentSubstrateTargetTerminalHandoff:
    """Project source inapplicability without claiming recurrence opposition."""

    _validate_source_stop_operands(
        config=config,
        result=result,
        source_adjudication=source_adjudication,
        receipt=receipt,
    )
    reasons = tuple(
        sorted(
            {
                *result.reason_codes,
                *source_adjudication.reason_codes,
                "FREEGSNKE_source gate_SOURCE_GATE_TERMINAL",
                "FREEGSNKE_TARGET_EVIDENCE_UNIT_COUNT_ZERO",
                "FREEGSNKE_TARGET_RECURRENCE_NOT_EVALUATED",
                "FREEGSNKE_SOURCE_QUALIFICATION_PROSPECTIVE_VALIDATION_PREREQUISITE_NONATTEMPT",
            }
        )
    )
    return IndependentSubstrateTargetTerminalHandoff(
        handoff_id=config.handoff_id,
        slot=IndependentSubstrateTargetKind.FREEGSNKE,
        evidence_world=EvidenceWorld.RESETTABLE_SIMULATOR,
        attained_level=TargetLevel.MEASUREMENT_READINESS,
        independence_class=IndependentSubstrateIndependenceClass.UNEVALUABLE,
        evaluation_eligible=False,
        categorical_support=False,
        decisive_opposition=False,
        unsafe_false_admission_count=0,
        action_ontology_clock_error_count=0,
        panel_envelope_limited=False,
        structural_recurrence_restrictiveness_supported=False,
        comparator_tied_or_won=False,
        structural_recurrence_less_safe_or_exact_than_comparator=False,
        metric_topology_exchanges=(),
        physical_consistency="NOT_APPLICABLE",
        prediction_receipt_sha256=None,
        match_receipt_sha256=None,
        maximum_claim_ceiling=('FREEGSNKE_SOURCE_VIEW_INAPPLICABLE_NO_TARGET_RECURRENCE_CLAIM'),
        reason_codes=reasons,
    )


def freegsnke_source_stop_handoff_config(
    *,
    result: FreeGsnkeSourceQualificationResult,
    source_adjudication: ScientificAdjudicationRecord,
    receipt: CanonicalTaskReceipt,
) -> FreeGsnkeSourceStopHandoffConfig:
    config = FreeGsnkeSourceStopHandoffConfig(
        config_id="config.independent-substrate-grounding.freegsnke-source-gate-source-stop-handoff",
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        source_qualification_result=_identity(result),
        source_scientific_adjudication=_identity(source_adjudication),
        source_qualification_receipt=_identity(receipt),
        handoff_id="handoff.independent-substrate-grounding.freegsnke-source-gate-source-stop",
        maximum_input_bytes=64 * 1024**2,
    )
    _validate_source_stop_operands(
        config=config,
        result=result,
        source_adjudication=source_adjudication,
        receipt=receipt,
    )
    return config


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=512 * 1024**2,
        gpu_devices=0,
        wall_time_seconds=2 * 60,
        source_scan_bytes=64 * 1024**2,
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


def freegsnke_source_stop_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    config_sha256 = sha256(FreeGsnkeSourceStopHandoffConfig.SCHEMA.encode("ascii")).hexdigest()
    inputs = tuple(
        sorted(
            (
                FreeGsnkeSourceStopHandoffConfig.SCHEMA,
                FreeGsnkeSourceQualificationResult.SCHEMA,
                ScientificAdjudicationRecord.SCHEMA,
                CanonicalTaskReceipt.SCHEMA,
            )
        )
    )
    project = CapabilityManifest(
        capability_key=FREEGSNKE_SOURCE_STOP_PROJECT_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        kind=CapabilityKind.ANALYSIS,
        config_schema=FreeGsnkeSourceStopHandoffConfig.SCHEMA,
        config_schema_sha256=config_sha256,
        input_schema_ids=inputs,
        output_schema_ids=(IndependentSubstrateTargetTerminalHandoff.SCHEMA,),
        permissions=_PERMISSIONS,
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        resource_ceiling=_budget(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="independent-substrate-grounding-freegsnke-source-stop-handoff",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=tuple(
            sorted(
                (
                    "exact-source-result-adjudication-receipt",
                    "zero-target-evidence-units",
                    "source-stop-not-recurrence-counterexample",
                )
            )
        ),
        implementation_sha256=implementation_sha256,
    )
    report = CapabilityManifest(
        capability_key=FREEGSNKE_SOURCE_STOP_REPORT_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        kind=CapabilityKind.REPORTER,
        config_schema=FreeGsnkeSourceStopHandoffConfig.SCHEMA,
        config_schema_sha256=config_sha256,
        input_schema_ids=tuple(sorted((*inputs, IndependentSubstrateTargetTerminalHandoff.SCHEMA))),
        output_schema_ids=(ScientificAdjudicationRecord.SCHEMA,),
        permissions=_PERMISSIONS,
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        resource_ceiling=_budget(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="independent-substrate-grounding-freegsnke-source-stop-handoff",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=tuple(
            sorted(
                (
                    "receipt-dependent-scientific-closeout",
                    "source-negative-retained",
                    "admission-not-evaluated",
                )
            )
        ),
        implementation_sha256=implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-freegsnke-source-stop-handoff",
        capabilities=tuple(sorted((project, report), key=lambda value: value.registry_id)),
    )


def freegsnke_source_stop_candidate_registrations(
    *, implementation_sha256: str
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=FREEGSNKE_SOURCE_STOP_PROVIDER_KEY,
            provider_version=FREEGSNKE_CAPABILITY_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=2 * 1024**2,
        )
        for manifest in freegsnke_source_stop_registry(
            implementation_sha256=implementation_sha256
        ).capabilities
    )


def _output(output_id: str, schema: str) -> OutputTemplate:
    return OutputTemplate(
        output_id=output_id,
        payload_schema=schema,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        filename_suffix=".json",
    )


def build_freegsnke_source_stop_protocol(
    *,
    registry: CapabilityRegistry,
    config: FreeGsnkeSourceStopHandoffConfig,
    config_ref: CapabilityConfigRef,
) -> ProtocolTemplate:
    project = registry.resolve(FREEGSNKE_SOURCE_STOP_PROJECT_KEY, FREEGSNKE_CAPABILITY_VERSION)
    report = registry.resolve(FREEGSNKE_SOURCE_STOP_REPORT_KEY, FREEGSNKE_CAPABILITY_VERSION)
    if (
        config_ref.config_id != config.config_id
        or config_ref.config_schema != config.SCHEMA
        or config_ref.config_schema_sha256 != project.config_schema_sha256
        or config_ref.content_sha256 != config.fingerprint()
        or project.config_schema_sha256 != report.config_schema_sha256
    ):
        raise ValueError("FreeGSNKE source-stop config reference differs")
    project_step = ProtocolStepTemplate(
        step_id=FREEGSNKE_SOURCE_STOP_PROJECT_STEP_ID,
        stage=ScientificStage.REPORT,
        capability_key=project.capability_key,
        capability_version=project.capability_version,
        config=config_ref,
        dependency_step_ids=(),
        outputs=(_output("target-handoff", IndependentSubstrateTargetTerminalHandoff.SCHEMA),),
        required_permissions=project.permissions,
        requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        resource_budget=project.resource_ceiling,
        resource_lock_ids=("freegsnke-source-stop-project",),
        barrier=BarrierKind.NONE,
        maximum_attempts=2,
        obligation_ids=tuple(
            sorted(
                (
                    "freegsnke-source-stop-exact-receipt-binding",
                    'freegsnke-source-stop-zero-executed-units',
                    "freegsnke-source-stop-no-recurrence-opposition",
                )
            )
        ),
    )
    report_step = ProtocolStepTemplate(
        step_id=FREEGSNKE_SOURCE_STOP_REPORT_STEP_ID,
        stage=ScientificStage.REPORT,
        capability_key=report.capability_key,
        capability_version=report.capability_version,
        config=config_ref,
        dependency_step_ids=(project_step.step_id,),
        outputs=(_output("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),),
        required_permissions=report.permissions,
        requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        resource_budget=report.resource_ceiling,
        resource_lock_ids=("freegsnke-source-stop-report",),
        barrier=BarrierKind.NONE,
        maximum_attempts=2,
        obligation_ids=tuple(
            sorted(
                (
                    "freegsnke-source-stop-receipt-dependent-closeout",
                    "freegsnke-source-negative-retained-as-evaluable",
                )
            )
        ),
    )
    return ProtocolTemplate(
        template_id="independent-substrate-grounding-freegsnke-source-stop-handoff-protocol",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        steps=tuple(sorted((project_step, report_step), key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _external(record: CanonicalRecord) -> CandidateGraphExternalInput:
    identity = _identity(record)
    return CandidateGraphExternalInput(
        input_id=f"input.independent-substrate-grounding.freegsnke.source-stop.{identity.object_id}",
        scientific_role=(
            ScientificInputRole.PARENT_RECEIPT
            if isinstance(record, CanonicalTaskReceipt)
            else ScientificInputRole.OUTCOME
        ),
        logical_artifact_id=f"artifact.independent-substrate-grounding.freegsnke.source-stop.{identity.object_id}",
        content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
        expected_content_sha256=identity.object_fingerprint,
        payload_schema=record.SCHEMA,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        maximum_size_bytes=64 * 1024**2,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


def freegsnke_source_stop_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    static_records: tuple[CanonicalRecord, ...],
) -> CandidateScientificGraph:
    if (
        len(protocol.steps) != 2
        or {value.step_id for value in protocol.steps}
        != {
            FREEGSNKE_SOURCE_STOP_PROJECT_STEP_ID,
            FREEGSNKE_SOURCE_STOP_REPORT_STEP_ID,
        }
        or len({_identity(value).object_id for value in static_records}) != len(static_records)
    ):
        raise ValueError("FreeGSNKE source-stop graph roster differs")
    externals = tuple(
        sorted((_external(value) for value in static_records), key=lambda v: v.input_id)
    )
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
    edges = [
        CandidateGraphEdge(
            edge_id=f"edge.{external.input_id}.{step.step_id}",
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
        for step in protocol.steps
        for external in externals
    ]
    report_step = next(
        value for value in protocol.steps if value.step_id == FREEGSNKE_SOURCE_STOP_REPORT_STEP_ID
    )
    edges.append(
        CandidateGraphEdge(
            edge_id="edge.freegsnke-source-stop-handoff.report",
            producer_node_id=FREEGSNKE_SOURCE_STOP_PROJECT_STEP_ID,
            producer_output_id="target-handoff",
            external_input_id=None,
            consumer_node_id=report_step.step_id,
            consumer_input_id="input.freegsnke-source-stop-handoff",
            scientific_role=ScientificInputRole.PARENT_RECEIPT,
            logical_artifact_id="artifact.independent-substrate-grounding.freegsnke.source-stop.target-handoff",
            payload_schema=IndependentSubstrateTargetTerminalHandoff.SCHEMA,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_size_bytes=8 * 1024**2,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=report_step.barrier,
        )
    )
    return CandidateScientificGraph(
        graph_id="graph.independent-substrate-grounding.freegsnke-source-stop-handoff",
        external_inputs=externals,
        nodes=nodes,
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def freegsnke_source_stop_study_template(
    *, protocol: ProtocolTemplate, graph: CandidateScientificGraph
) -> StudyTemplate:
    owners = {
        obligation: (step.step_id, step.outputs[0].output_id)
        for step in protocol.steps
        for obligation in step.obligation_ids
    }
    return StudyTemplate(
        template_key="independent-substrate-grounding.freegsnke.source-stop-handoff",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id="coverage.independent-substrate-grounding.freegsnke-source-stop-handoff",
            bindings=tuple(
                ObligationCoverageBinding(
                    obligation_id=obligation,
                    proof_owner_node_id=owner[0],
                    required_output_id=owner[1],
                    contributor_edge_ids=tuple(
                        sorted(
                            edge.edge_id
                            for edge in graph.edges
                            if edge.consumer_node_id == owner[0]
                        )
                    ),
                )
                for obligation, owner in sorted(owners.items())
            ),
        ),
    )


def _decode_one(context: TaskContext, record_type: type[CanonicalRecord]) -> CanonicalRecord:
    records = tuple(
        decode_canonical_bytes(port.read(), record_type, maximum_bytes=port.size_bytes)
        for port in context.input_ports
        if port.payload_schema == record_type.SCHEMA
    )
    if len(records) != 1:
        raise ValueError(f"FreeGSNKE source-stop requires one {record_type.SCHEMA} input")
    return records[0]


class _FreeGsnkeSourceStopRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def _inputs(
        self, context: TaskContext
    ) -> tuple[
        FreeGsnkeSourceStopHandoffConfig,
        FreeGsnkeSourceQualificationResult,
        ScientificAdjudicationRecord,
        CanonicalTaskReceipt,
    ]:
        config = _decode_one(context, FreeGsnkeSourceStopHandoffConfig)
        result = _decode_one(context, FreeGsnkeSourceQualificationResult)
        source_adjudication = _decode_one(context, ScientificAdjudicationRecord)
        receipt = _decode_one(context, CanonicalTaskReceipt)
        assert isinstance(config, FreeGsnkeSourceStopHandoffConfig)
        assert isinstance(result, FreeGsnkeSourceQualificationResult)
        assert isinstance(source_adjudication, ScientificAdjudicationRecord)
        assert isinstance(receipt, CanonicalTaskReceipt)
        if (
            context.config.content_sha256 != config.fingerprint()
            or context.permissions != self.manifest.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or any(port.size_bytes > config.maximum_input_bytes for port in context.input_ports)
        ):
            raise ValueError("FreeGSNKE source-stop runner/config authority differs")
        _validate_source_stop_operands(
            config=config,
            result=result,
            source_adjudication=source_adjudication,
            receipt=receipt,
        )
        return config, result, source_adjudication, receipt

    def _project(self, context: TaskContext) -> RunnerResult:
        config, result, source_adjudication, receipt = self._inputs(context)
        handoff = freegsnke_source_stop_handoff(
            config=config,
            result=result,
            source_adjudication=source_adjudication,
            receipt=receipt,
        )
        if {value.payload_schema for value in context.output_ports} != {
            IndependentSubstrateTargetTerminalHandoff.SCHEMA
        }:
            raise ValueError("FreeGSNKE source-stop projector output differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(output_id=port.output_id, payload=handoff.canonical_bytes())
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("freegsnke-source-stop-exact-parent-receipt", True, ()),
                ReceiptCheck("freegsnke-source-stop-no-target-unit", True, ()),
                ReceiptCheck("freegsnke-source-stop-not-recurrence-opposition", True, ()),
            ),
        )

    def _report(self, context: TaskContext) -> RunnerResult:
        config, result, source_adjudication, receipt = self._inputs(context)
        handoff = _decode_one(context, IndependentSubstrateTargetTerminalHandoff)
        assert isinstance(handoff, IndependentSubstrateTargetTerminalHandoff)
        expected_handoff = freegsnke_source_stop_handoff(
            config=config,
            result=result,
            source_adjudication=source_adjudication,
            receipt=receipt,
        )
        adjudication_context = context.scientific_adjudication_context
        if handoff != expected_handoff or adjudication_context is None:
            raise ValueError("FreeGSNKE source-stop reporter input/context differs")
        if len(context.dependency_receipt_ids) != 1:
            raise ValueError("FreeGSNKE source-stop reporter lacks its projector receipt")
        output_ids = tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
            )
        )
        if len(output_ids) != len(context.output_ports):
            raise ValueError("FreeGSNKE source-stop report output lacks logical identity")
        unevaluable = (
            result.disposition is FreeGsnkeSourceQualificationDisposition.OPERATIONAL_UNEVALUABLE
        )
        adjudication = ScientificAdjudicationRecord(
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
                ScientificStatus.UNEVALUABLE if unevaluable else ScientificStatus.NOT_SUPPORTED
            ),
            admission_status=(
                AdmissionStatus.UNEVALUABLE if unevaluable else AdmissionStatus.NOT_EVALUATED
            ),
            reason_codes=tuple(
                sorted(
                    {
                        *handoff.reason_codes,
                        "FREEGSNKE_SOURCE_STOP_HANDOFF_NONPROMOTABLE",
                        "FREEGSNKE_TARGET_RECURRENCE_NOT_EVALUATED",
                    }
                )
            ),
        )
        if {value.payload_schema for value in context.output_ports} != {
            ScientificAdjudicationRecord.SCHEMA
        }:
            raise ValueError("FreeGSNKE source-stop reporter output differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(output_id=port.output_id, payload=adjudication.canonical_bytes())
                for port in context.output_ports
            ),
            checks=tuple(
                sorted(
                    (
                        ReceiptCheck("freegsnke-source-stop-projector-receipt-bound", True, ()),
                        ReceiptCheck("freegsnke-source-negative-retained", True, ()),
                        ReceiptCheck("freegsnke-target-recurrence-not-evaluated", True, ()),
                    ),
                    key=lambda value: value.check_id,
                )
            ),
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        if self.manifest.capability_key == FREEGSNKE_SOURCE_STOP_PROJECT_KEY:
            return self._project(context)
        if self.manifest.capability_key == FREEGSNKE_SOURCE_STOP_REPORT_KEY:
            return self._report(context)
        raise ValueError("FreeGSNKE source-stop runner capability differs")


def freegsnke_source_stop_runners(*, registry: CapabilityRegistry) -> tuple[TaskRunner, ...]:
    return tuple(_FreeGsnkeSourceStopRunner(value) for value in registry.capabilities)


class FreeGsnkeSourceStopRuntimeProvider(CampaignRuntimeProvider):
    """Closed provider for the two-node post-reveal handoff campaign."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: FreeGsnkeSourceStopHandoffConfig,
        result: FreeGsnkeSourceQualificationResult,
        source_adjudication: ScientificAdjudicationRecord,
        receipt: CanonicalTaskReceipt,
    ) -> None:
        expected = freegsnke_source_stop_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("FreeGSNKE source-stop provider registry differs")
        _validate_source_stop_operands(
            config=config,
            result=result,
            source_adjudication=source_adjudication,
            receipt=receipt,
        )
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.config = config
        self.result = result
        self.source_adjudication = source_adjudication
        self.receipt = receipt
        self._runners = freegsnke_source_stop_runners(registry=registry)
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("FreeGSNKE source-stop registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("FreeGSNKE source-stop plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        records: dict[str, CanonicalRecord] = {
            task.capability.config.artifact_id: self.config for task in plan.tasks
        }
        for record in (self.result, self.source_adjudication, self.receipt):
            identity = _identity(record)
            records[f"artifact.independent-substrate-grounding.freegsnke.source-stop.{identity.object_id}"] = record
        if set(records) != set(specs):
            raise ValueError("FreeGSNKE source-stop external input roster differs")
        values = []
        for artifact_id, input_record in sorted(records.items()):
            spec = specs[artifact_id]
            # The projection config binds a revealed adverse parent and is
            # therefore outcome-visible itself.  It is never relabelled as a
            # prospective or outcome-blind design object.
            visibility = VisibilityCeiling.OUTCOME_VISIBLE
            access = OutcomeAccess.EVALUATION_REVEALED
            parent = ArtifactLineageParent(
                identity=_identity(input_record),
                visibility_ceiling=visibility,
                outcome_access=access,
            )
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=input_record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    payload=input_record.canonical_bytes(),
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
            raise ValueError("FreeGSNKE source-stop semantic registry differs")
        planned = None
        if execution_plan is not None:
            planned = {
                (task.capability.capability_key, output.payload_schema, output.profile)
                for task in execution_plan.tasks
                for output in task.outputs
            }
        record_types = {
            IndependentSubstrateTargetTerminalHandoff.SCHEMA: IndependentSubstrateTargetTerminalHandoff,
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
            raise ValueError("FreeGSNKE source-stop adjudication registry differs")
        if execution_plan is not None and not any(
            value.capability.capability_key == FREEGSNKE_SOURCE_STOP_REPORT_KEY
            for value in execution_plan.tasks
        ):
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=FREEGSNKE_SOURCE_STOP_REPORT_KEY,
            capability_version=FREEGSNKE_CAPABILITY_VERSION,
            output_id="scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
        )


__all__ = [
    "FREEGSNKE_SOURCE_STOP_PROJECT_KEY",
    "FREEGSNKE_SOURCE_STOP_PROJECT_STEP_ID",
    "FREEGSNKE_SOURCE_STOP_PROVIDER_KEY",
    "FREEGSNKE_SOURCE_STOP_REPORT_KEY",
    "FREEGSNKE_SOURCE_STOP_REPORT_STEP_ID",
    'FreeGsnkeSourceStopHandoffConfig',
    "FreeGsnkeSourceStopRuntimeProvider",
    "build_freegsnke_source_stop_protocol",
    "freegsnke_source_stop_candidate_registrations",
    "freegsnke_source_stop_handoff",
    "freegsnke_source_stop_handoff_config",
    'freegsnke_source_stop_study_template',
    "freegsnke_source_stop_registry",
    "freegsnke_source_stop_runners",
    "freegsnke_source_stop_scientific_graph",
]

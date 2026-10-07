"""Receipt-bound FreeGSNKE conditional nonissue and terminal projection DAGs.

These post-reveal protocols do not reopen simulator payloads or make a new
scientific choice.  One optional node records a frozen prospective validation prerequisite
nonissue.  A separate terminal node runs only after the actual parent/child
receipts exist and projects the compact independent substrate cross-target handoff from their exact bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, Final

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentImplementationDossier, IndependentSubstrateTargetTerminalHandoff
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
from empirical_lawhood.kernel.status import OperationalStatus
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
from empirical_lawhood.runtime.conditional_children import ConditionalChildInstantiation
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

from .structural_recurrence_bridge import FreeGsnkeStructuralRecurrenceDesignCandidate
from .registry import FREEGSNKE_CAPABILITY_VERSION
from .target_evaluation import FreeGsnkeAdmissionEvaluationEvaluation
from .target_lifecycle_protocol import FREEGSNKE_TARGET_FREEZE_STEP_ID, FREEGSNKE_ADMISSION_EVALUATION_STEP_ID, FreeGsnkeTargetPredictionFreeze
from .target_terminal import FreeGsnkeTargetTerminal, build_freegsnke_target_terminal
from .target_validation_protocol import FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_STEP_ID, FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_STEP_ID
from .target_validation import FreeGsnkeProspectiveValidationRequestRoster, FreeGsnkeTargetValidation, finalize_freegsnke_without_prospective


FREEGSNKE_NONISSUE_KEY: Final = 'freegsnke.finalize-prospective-validation-nonissue'
FREEGSNKE_TERMINAL_KEY: Final = "freegsnke.project-target-terminal"
FREEGSNKE_CLOSEOUT_PROVIDER_KEY: Final = "independent-substrate-grounding.freegsnke-closeout-provider"
FREEGSNKE_NONISSUE_STEP_ID: Final = 'finalize-prospective-validation-nonissue'
FREEGSNKE_TERMINAL_STEP_ID: Final = "project-target-terminal"


class FreeGsnkeCloseoutOperation(StrEnum):
    FINALIZE_NONISSUE = "FINALIZE_NONISSUE"
    PROJECT_TERMINAL = "PROJECT_TERMINAL"


_OPERATION_KEY = {
    FreeGsnkeCloseoutOperation.FINALIZE_NONISSUE: FREEGSNKE_NONISSUE_KEY,
    FreeGsnkeCloseoutOperation.PROJECT_TERMINAL: FREEGSNKE_TERMINAL_KEY,
}


@dataclass(frozen=True, slots=True)
class FreeGsnkeCloseoutConfig(CanonicalRecord):
    """Exact post-reveal operands for one non-scientific closeout projection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-closeout-config'

    config_id: str
    operation: FreeGsnkeCloseoutOperation
    capability_key: str
    capability_version: str
    validation_id: str
    terminal_id: str
    handoff_id: str
    parent_admission_evaluation: ObjectIdentity | None
    selected_design_candidate: ObjectIdentity | None
    conditional_instantiation: ObjectIdentity | None
    validation: ObjectIdentity | None
    independence_dossier: ObjectIdentity | None
    prediction_receipt: ObjectIdentity | None
    admission_evaluation_evaluation_receipt: ObjectIdentity | None
    prospective_validation_issue_receipt: ObjectIdentity | None
    prospective_validation_evaluation_receipt: ObjectIdentity | None
    target_match_receipt: ObjectIdentity | None
    maximum_input_bytes: int

    def __post_init__(self) -> None:
        for name in (
            "config_id",
            "capability_key",
            "validation_id",
            "terminal_id",
            "handoff_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.capability_version)
        if (
            self.capability_key != _OPERATION_KEY[self.operation]
            or self.capability_version != FREEGSNKE_CAPABILITY_VERSION
        ):
            raise ValueError("FreeGSNKE closeout operation/capability differs")
        if not 0 < self.maximum_input_bytes <= 512 * 1024**2:
            raise ValueError("FreeGSNKE closeout input bound differs")
        if self.operation is FreeGsnkeCloseoutOperation.FINALIZE_NONISSUE:
            if (
                self.parent_admission_evaluation is None
                or self.parent_admission_evaluation.object_schema != FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA
                or self.selected_design_candidate is None
                or self.selected_design_candidate.object_schema
                != FreeGsnkeStructuralRecurrenceDesignCandidate.SCHEMA
                or self.conditional_instantiation is None
                or self.conditional_instantiation.object_schema
                != ConditionalChildInstantiation.SCHEMA
                or any(
                    value is not None
                    for value in (
                        self.validation,
                        self.independence_dossier,
                        self.prediction_receipt,
                        self.admission_evaluation_evaluation_receipt,
                        self.prospective_validation_issue_receipt,
                        self.prospective_validation_evaluation_receipt,
                        self.target_match_receipt,
                    )
                )
            ):
                raise ValueError("FreeGSNKE nonissue closeout operands differ")
            return
        if (
            any(
                value is not None
                for value in (
                    self.parent_admission_evaluation,
                    self.selected_design_candidate,
                    self.conditional_instantiation,
                )
            )
            or self.validation is None
            or self.validation.object_schema != FreeGsnkeTargetValidation.SCHEMA
            or self.independence_dossier is None
            or self.independence_dossier.object_schema != IndependentImplementationDossier.SCHEMA
            or self.prediction_receipt is None
            or self.admission_evaluation_evaluation_receipt is None
        ):
            raise ValueError("FreeGSNKE terminal closeout operands differ")
        for value in (
            self.prediction_receipt,
            self.admission_evaluation_evaluation_receipt,
            self.prospective_validation_issue_receipt,
            self.prospective_validation_evaluation_receipt,
            self.target_match_receipt,
        ):
            if value is not None and value.object_schema != CanonicalTaskReceipt.SCHEMA:
                raise ValueError("FreeGSNKE terminal receipt schema differs")
        prospective_validation_receipts_present = (
            self.prospective_validation_issue_receipt is not None,
            self.prospective_validation_evaluation_receipt is not None,
        )
        if prospective_validation_receipts_present not in {(False, False), (True, True)}:
            raise ValueError("FreeGSNKE terminal prospective validation receipt pair differs")


def freegsnke_nonissue_closeout_config(
    *,
    validation_id: str,
    parent_admission_evaluation: FreeGsnkeAdmissionEvaluationEvaluation,
    selected_design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
    instantiation: ConditionalChildInstantiation,
    namespace: str,
) -> FreeGsnkeCloseoutConfig:
    """Bind an ineligible frozen conditional resolution to its typed validation."""

    validate_stable_id(namespace, field_name="namespace")
    if (
        instantiation.child_candidate is not None
        or instantiation.parent_record
        != ObjectIdentity.from_record(parent_admission_evaluation.evaluation_id, parent_admission_evaluation)
        or instantiation.decode_parent(
            FreeGsnkeAdmissionEvaluationEvaluation,
            maximum_bytes=256 * 1024**2,
        )
        != parent_admission_evaluation
    ):
        raise ValueError("FreeGSNKE nonissue config requires an exact terminal resolution")
    return FreeGsnkeCloseoutConfig(
        config_id=f"config.{namespace}.prospective-validation-nonissue",
        operation=FreeGsnkeCloseoutOperation.FINALIZE_NONISSUE,
        capability_key=FREEGSNKE_NONISSUE_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        validation_id=validation_id,
        terminal_id=f"terminal.{namespace}",
        handoff_id=f"handoff.{namespace}",
        parent_admission_evaluation=ObjectIdentity.from_record(parent_admission_evaluation.evaluation_id, parent_admission_evaluation),
        selected_design_candidate=ObjectIdentity.from_record(
            selected_design_candidate.candidate_id,
            selected_design_candidate,
        ),
        conditional_instantiation=ObjectIdentity.from_record(
            instantiation.instantiation_id,
            instantiation,
        ),
        validation=None,
        independence_dossier=None,
        prediction_receipt=None,
        admission_evaluation_evaluation_receipt=None,
        prospective_validation_issue_receipt=None,
        prospective_validation_evaluation_receipt=None,
        target_match_receipt=None,
        maximum_input_bytes=512 * 1024**2,
    )


def _receipt_identity(receipt: CanonicalTaskReceipt | None) -> ObjectIdentity | None:
    if receipt is None:
        return None
    return ObjectIdentity.from_record(receipt.receipt_id, receipt)


def freegsnke_terminal_closeout_config(
    *,
    validation: FreeGsnkeTargetValidation,
    independence_dossier: IndependentImplementationDossier,
    prediction_receipt: CanonicalTaskReceipt,
    admission_evaluation_evaluation_receipt: CanonicalTaskReceipt,
    prospective_validation_issue_receipt: CanonicalTaskReceipt | None,
    prospective_validation_evaluation_receipt: CanonicalTaskReceipt | None,
    target_match_receipt: CanonicalTaskReceipt | None,
    namespace: str,
) -> FreeGsnkeCloseoutConfig:
    """Bind actual successful receipts after the conditional branch terminates."""

    validate_stable_id(namespace, field_name="namespace")
    if validation.prospective_validation_child_issued != (
        prospective_validation_issue_receipt is not None and prospective_validation_evaluation_receipt is not None
    ):
        raise ValueError("FreeGSNKE terminal config prospective validation receipt presence differs")
    if (validation.target_match is not None) != (target_match_receipt is not None):
        raise ValueError("FreeGSNKE terminal config match receipt presence differs")
    if validation.prospective_validation_child_issued and target_match_receipt != prospective_validation_evaluation_receipt:
        raise ValueError("FreeGSNKE issued prospective validation match must share its evaluator receipt")
    return FreeGsnkeCloseoutConfig(
        config_id=f"config.{namespace}.terminal",
        operation=FreeGsnkeCloseoutOperation.PROJECT_TERMINAL,
        capability_key=FREEGSNKE_TERMINAL_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        validation_id=validation.validation_id,
        terminal_id=f"terminal.{namespace}",
        handoff_id=f"handoff.{namespace}",
        parent_admission_evaluation=None,
        selected_design_candidate=None,
        conditional_instantiation=None,
        validation=ObjectIdentity.from_record(validation.validation_id, validation),
        independence_dossier=ObjectIdentity.from_record(
            independence_dossier.dossier_id,
            independence_dossier,
        ),
        prediction_receipt=_receipt_identity(prediction_receipt),
        admission_evaluation_evaluation_receipt=_receipt_identity(admission_evaluation_evaluation_receipt),
        prospective_validation_issue_receipt=_receipt_identity(prospective_validation_issue_receipt),
        prospective_validation_evaluation_receipt=_receipt_identity(prospective_validation_evaluation_receipt),
        target_match_receipt=_receipt_identity(target_match_receipt),
        maximum_input_bytes=512 * 1024**2,
    )


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=1024**3,
        gpu_devices=0,
        wall_time_seconds=10 * 60,
        source_scan_bytes=512 * 1024**2,
        output_bytes=512 * 1024**2,
    )


_PERMISSIONS = tuple(
    sorted(
        (
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            CapabilityPermission.READ_OUTCOME_VISIBLE,
        )
    )
)


def freegsnke_closeout_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    schema_sha256 = sha256(FreeGsnkeCloseoutConfig.SCHEMA.encode("ascii")).hexdigest()
    specs = (
        (
            FREEGSNKE_NONISSUE_KEY,
            (
                FreeGsnkeCloseoutConfig.SCHEMA,
                FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA,
                FreeGsnkeStructuralRecurrenceDesignCandidate.SCHEMA,
                ConditionalChildInstantiation.SCHEMA,
            ),
            (FreeGsnkeTargetValidation.SCHEMA,),
            ("conditional-terminal-exact", 'prospective-validation-prerequisite-nonissue'),
        ),
        (
            FREEGSNKE_TERMINAL_KEY,
            (
                FreeGsnkeCloseoutConfig.SCHEMA,
                FreeGsnkeTargetValidation.SCHEMA,
                IndependentImplementationDossier.SCHEMA,
                CanonicalTaskReceipt.SCHEMA,
            ),
            (
                FreeGsnkeTargetTerminal.SCHEMA,
                IndependentSubstrateTargetTerminalHandoff.SCHEMA,
            ),
            ("actual-receipts-bound", "compact-cross-target-handoff"),
        ),
    )
    return CapabilityRegistry(
        registry_id="independent-substrate-grounding-freegsnke-target-closeout",
        capabilities=tuple(
            CapabilityManifest(
                capability_key=key,
                capability_version=FREEGSNKE_CAPABILITY_VERSION,
                kind=CapabilityKind.ANALYSIS,
                config_schema=FreeGsnkeCloseoutConfig.SCHEMA,
                config_schema_sha256=schema_sha256,
                input_schema_ids=tuple(sorted(inputs)),
                output_schema_ids=tuple(sorted(outputs)),
                permissions=_PERMISSIONS,
                maximum_evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
                maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                resource_ceiling=_budget(),
                deterministic=True,
                seed_required=False,
                language_id="python",
                runtime_id="independent-substrate-grounding-freegsnke-target-closeout",
                requires_clean_commit=True,
                requires_active_mount=True,
                requires_network=False,
                conformance_check_ids=checks,
                implementation_sha256=implementation_sha256,
            )
            for key, inputs, outputs, checks in specs
        ),
    )


def freegsnke_closeout_candidate_registrations(
    *,
    implementation_sha256: str,
) -> tuple[CandidateCapabilityRegistration, ...]:
    return tuple(
        CandidateCapabilityRegistration(
            manifest=manifest,
            provider_key=FREEGSNKE_CLOSEOUT_PROVIDER_KEY,
            provider_version=FREEGSNKE_CAPABILITY_VERSION,
            config_media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_config_bytes=2 * 1024**2,
        )
        for manifest in freegsnke_closeout_registry(
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


def build_freegsnke_closeout_protocol(
    *,
    registry: CapabilityRegistry,
    config: FreeGsnkeCloseoutConfig,
    config_ref: CapabilityConfigRef,
) -> ProtocolTemplate:
    manifest = registry.resolve(config.capability_key, config.capability_version)
    if (
        config_ref.config_id != config.config_id
        or config_ref.config_schema != config.SCHEMA
        or config_ref.config_schema_sha256 != manifest.config_schema_sha256
        or config_ref.content_sha256 != config.fingerprint()
    ):
        raise ValueError("FreeGSNKE closeout config reference differs")
    outputs: tuple[OutputTemplate, ...]
    if config.operation is FreeGsnkeCloseoutOperation.FINALIZE_NONISSUE:
        step_id = FREEGSNKE_NONISSUE_STEP_ID
        outputs = (_output("target-validation", FreeGsnkeTargetValidation.SCHEMA),)
        obligations = (
            "freegsnke-conditional-nonissue-exact",
            'freegsnke-prospective-validation-negative-not-converted-to-unevaluable',
        )
    else:
        step_id = FREEGSNKE_TERMINAL_STEP_ID
        outputs = tuple(
            sorted(
                (
                    _output("target-terminal", FreeGsnkeTargetTerminal.SCHEMA),
                    _output("target-handoff", IndependentSubstrateTargetTerminalHandoff.SCHEMA),
                ),
                key=lambda value: value.output_id,
            )
        )
        obligations = (
            "freegsnke-actual-receipt-closure",
            "freegsnke-compact-terminal-no-payload-pooling",
        )
    return ProtocolTemplate(
        template_id=f"independent-substrate-grounding-freegsnke-{step_id}-protocol",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        steps=(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=ScientificStage.REPORT,
                capability_key=config.capability_key,
                capability_version=config.capability_version,
                config=config_ref,
                dependency_step_ids=(),
                outputs=outputs,
                required_permissions=manifest.permissions,
                requested_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                resource_budget=manifest.resource_ceiling,
                resource_lock_ids=(f"freegsnke-{step_id}",),
                barrier=BarrierKind.NONE,
                maximum_attempts=2,
                obligation_ids=obligations,
            ),
        ),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


def _record_id(record: CanonicalRecord) -> str:
    for attribute in (
        "config_id",
        "evaluation_id",
        "candidate_id",
        "instantiation_id",
        "validation_id",
        "dossier_id",
        "receipt_id",
    ):
        if hasattr(record, attribute):
            value = getattr(record, attribute)
            if isinstance(value, str):
                return value
    raise TypeError(f"unsupported FreeGSNKE closeout record: {type(record)!r}")


def _identity(record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(_record_id(record), record)


def _role(record: CanonicalRecord) -> ScientificInputRole:
    if isinstance(record, CanonicalTaskReceipt):
        return ScientificInputRole.PARENT_RECEIPT
    if isinstance(record, (FreeGsnkeAdmissionEvaluationEvaluation, FreeGsnkeTargetValidation)):
        return ScientificInputRole.OUTCOME
    return ScientificInputRole.QUALIFICATION


def freegsnke_closeout_scientific_graph(
    *,
    protocol: ProtocolTemplate,
    registry: CapabilityRegistry,
    static_records: tuple[CanonicalRecord, ...],
) -> CandidateScientificGraph:
    if len(protocol.steps) != 1 or len({_record_id(value) for value in static_records}) != len(
        static_records
    ):
        raise ValueError("FreeGSNKE closeout graph roster differs")
    step = protocol.steps[0]
    externals = tuple(
        CandidateGraphExternalInput(
            input_id=f"input.independent-substrate-grounding.freegsnke.closeout.{_record_id(record)}",
            scientific_role=_role(record),
            logical_artifact_id=(f"artifact.independent-substrate-grounding.freegsnke.closeout.{_record_id(record)}"),
            content_identity_policy=ContentIdentityPolicy.EXACT_SHA256,
            expected_content_sha256=record.fingerprint(),
            payload_schema=record.SCHEMA,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            maximum_size_bytes=512 * 1024**2,
            outcome_access=(
                OutcomeAccess.OUTCOME_BLIND
                if isinstance(record, IndependentImplementationDossier)
                else OutcomeAccess.EVALUATION_REVEALED
            ),
            visibility_ceiling=(
                VisibilityCeiling.PROSPECTIVE
                if isinstance(record, IndependentImplementationDossier)
                else VisibilityCeiling.OUTCOME_VISIBLE
            ),
        )
        for record in static_records
    )
    node = CandidateGraphNode(
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
    edges = tuple(
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
        for external in externals
    )
    return CandidateScientificGraph(
        graph_id=f"graph.independent-substrate-grounding.freegsnke.{step.step_id}",
        external_inputs=tuple(sorted(externals, key=lambda value: value.input_id)),
        nodes=(node,),
        edges=tuple(sorted(edges, key=lambda value: value.edge_id)),
    )


def freegsnke_closeout_study_template(
    *,
    protocol: ProtocolTemplate,
    graph: CandidateScientificGraph,
) -> StudyTemplate:
    step = protocol.steps[0]
    edge_ids = tuple(value.edge_id for value in graph.edges)
    return StudyTemplate(
        template_key=f"independent-substrate-grounding.freegsnke.{step.step_id}",
        template_version=FREEGSNKE_CAPABILITY_VERSION,
        protocol=protocol,
        graph=graph,
        coverage=ObligationCoverage(
            coverage_id=f"coverage.independent-substrate-grounding.freegsnke.{step.step_id}",
            bindings=tuple(
                ObligationCoverageBinding(
                    obligation_id=obligation,
                    proof_owner_node_id=step.step_id,
                    required_output_id=step.outputs[0].output_id,
                    contributor_edge_ids=edge_ids,
                )
                for obligation in step.obligation_ids
            ),
        ),
    )


def _decode_records(
    context: TaskContext,
    record_type: type[CanonicalRecord],
) -> tuple[CanonicalRecord, ...]:
    return tuple(
        decode_canonical_bytes(
            port.read(),
            record_type,
            maximum_bytes=port.size_bytes,
        )
        for port in context.input_ports
        if port.payload_schema == record_type.SCHEMA
    )


def _one_record(
    context: TaskContext,
    record_type: type[CanonicalRecord],
    *,
    label: str,
) -> CanonicalRecord:
    values = _decode_records(context, record_type)
    if len(values) != 1:
        raise ValueError(f"FreeGSNKE closeout {label} requires one record")
    return values[0]


def _receipt_has_schema(receipt: CanonicalTaskReceipt, schema: str) -> bool:
    return any(value.payload_schema == schema for value in receipt.output_logical_artifacts)


class _FreeGsnkeCloseoutRunner:
    def __init__(self, manifest: CapabilityManifest) -> None:
        self.manifest = manifest
        self.execution_count = 0

    def _config(self, context: TaskContext) -> FreeGsnkeCloseoutConfig:
        value = _one_record(context, FreeGsnkeCloseoutConfig, label="config")
        assert isinstance(value, FreeGsnkeCloseoutConfig)
        if (
            value.capability_key != self.manifest.capability_key
            or context.config.content_sha256 != value.fingerprint()
            or context.permissions != self.manifest.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or any(port.size_bytes > value.maximum_input_bytes for port in context.input_ports)
        ):
            raise ValueError("FreeGSNKE closeout runner/config authority differs")
        return value

    @staticmethod
    def _emit(
        context: TaskContext,
        records: tuple[CanonicalRecord, ...],
        checks: tuple[ReceiptCheck, ...],
    ) -> RunnerResult:
        by_schema = {value.SCHEMA: value for value in records}
        if len(by_schema) != len(records) or set(by_schema) != {
            value.payload_schema for value in context.output_ports
        }:
            raise ValueError("FreeGSNKE closeout output roster differs")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=by_schema[port.payload_schema].canonical_bytes(),
                )
                for port in context.output_ports
            ),
            checks=tuple(sorted(checks, key=lambda value: value.check_id)),
        )

    def _nonissue(
        self,
        context: TaskContext,
        config: FreeGsnkeCloseoutConfig,
    ) -> RunnerResult:
        parent = _one_record(context, FreeGsnkeAdmissionEvaluationEvaluation, label="admission evaluation parent")
        selected = _one_record(
            context,
            FreeGsnkeStructuralRecurrenceDesignCandidate,
            label="selected design",
        )
        instantiation = _one_record(
            context,
            ConditionalChildInstantiation,
            label="conditional instantiation",
        )
        assert isinstance(parent, FreeGsnkeAdmissionEvaluationEvaluation)
        assert isinstance(selected, FreeGsnkeStructuralRecurrenceDesignCandidate)
        assert isinstance(instantiation, ConditionalChildInstantiation)
        if (
            _identity(parent) != config.parent_admission_evaluation
            or _identity(selected) != config.selected_design_candidate
            or _identity(instantiation) != config.conditional_instantiation
            or instantiation.child_candidate is not None
            or instantiation.decode_parent(
                FreeGsnkeAdmissionEvaluationEvaluation,
                maximum_bytes=config.maximum_input_bytes,
            )
            != parent
        ):
            raise ValueError("FreeGSNKE nonissue frozen operands differ")
        validation = finalize_freegsnke_without_prospective(
            validation_id=config.validation_id,
            parent_admission_evaluation=parent,
            selected_design_candidate=selected,
        )
        return self._emit(
            context,
            (validation,),
            (
                ReceiptCheck('freegsnke-prospective-validation-prerequisite-nonissue', True, ()),
                ReceiptCheck("freegsnke-negative-parent-retained", True, ()),
            ),
        )

    def _terminal(
        self,
        context: TaskContext,
        config: FreeGsnkeCloseoutConfig,
    ) -> RunnerResult:
        validation = _one_record(
            context,
            FreeGsnkeTargetValidation,
            label="validation",
        )
        dossier = _one_record(
            context,
            IndependentImplementationDossier,
            label="independence dossier",
        )
        receipts_raw = _decode_records(context, CanonicalTaskReceipt)
        receipts = {
            _identity(value): value
            for value in receipts_raw
            if isinstance(value, CanonicalTaskReceipt)
        }
        assert isinstance(validation, FreeGsnkeTargetValidation)
        assert isinstance(dossier, IndependentImplementationDossier)
        if _identity(validation) != config.validation or _identity(dossier) != (
            config.independence_dossier
        ):
            raise ValueError("FreeGSNKE terminal scientific operands differ")

        def receipt(
            identity: ObjectIdentity | None,
            *,
            task_id: str,
            output_schema: str,
            required: bool,
        ) -> CanonicalTaskReceipt | None:
            if identity is None:
                if required:
                    raise ValueError("FreeGSNKE terminal lacks a required receipt")
                return None
            try:
                value = receipts[identity]
            except KeyError as error:
                raise ValueError("FreeGSNKE terminal receipt bytes differ") from error
            if (
                value.task_id != task_id
                or value.operational_status is not OperationalStatus.SUCCEEDED
                or not _receipt_has_schema(value, output_schema)
            ):
                raise ValueError("FreeGSNKE terminal receipt role differs")
            return value

        prediction = receipt(
            config.prediction_receipt,
            task_id=FREEGSNKE_TARGET_FREEZE_STEP_ID,
            output_schema=FreeGsnkeTargetPredictionFreeze.SCHEMA,
            required=True,
        )
        admission_evaluation = receipt(
            config.admission_evaluation_evaluation_receipt,
            task_id=FREEGSNKE_ADMISSION_EVALUATION_STEP_ID,
            output_schema=FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA,
            required=True,
        )
        prospective_validation_issue = receipt(
            config.prospective_validation_issue_receipt,
            task_id=FREEGSNKE_PROSPECTIVE_VALIDATION_ISSUE_STEP_ID,
            output_schema=FreeGsnkeProspectiveValidationRequestRoster.SCHEMA,
            required=validation.prospective_validation_child_issued,
        )
        prospective_validation_evaluation = receipt(
            config.prospective_validation_evaluation_receipt,
            task_id=FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_STEP_ID,
            output_schema=FreeGsnkeTargetValidation.SCHEMA,
            required=validation.prospective_validation_child_issued,
        )
        match_task_id = (
            FREEGSNKE_PROSPECTIVE_VALIDATION_EVALUATE_STEP_ID
            if validation.prospective_validation_child_issued
            else FREEGSNKE_NONISSUE_STEP_ID
        )
        target_match = receipt(
            config.target_match_receipt,
            task_id=match_task_id,
            output_schema=FreeGsnkeTargetValidation.SCHEMA,
            required=validation.target_match is not None,
        )
        assert prediction is not None
        assert admission_evaluation is not None
        terminal = build_freegsnke_target_terminal(
            terminal_id=config.terminal_id,
            handoff_id=config.handoff_id,
            validation=validation,
            independence_dossier=dossier,
            prediction_receipt_sha256=prediction.fingerprint(),
            admission_evaluation_evaluation_receipt_sha256=admission_evaluation.fingerprint(),
            prospective_validation_issue_receipt_sha256=(None if prospective_validation_issue is None else prospective_validation_issue.fingerprint()),
            prospective_validation_evaluation_receipt_sha256=(
                None if prospective_validation_evaluation is None else prospective_validation_evaluation.fingerprint()
            ),
            target_match_receipt_sha256=(
                None if target_match is None else target_match.fingerprint()
            ),
        )
        return self._emit(
            context,
            (terminal, terminal.handoff),
            (
                ReceiptCheck("freegsnke-terminal-actual-receipts", True, ()),
                ReceiptCheck("freegsnke-terminal-compact-handoff", True, ()),
            ),
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        config = self._config(context)
        if config.operation is FreeGsnkeCloseoutOperation.FINALIZE_NONISSUE:
            return self._nonissue(context, config)
        return self._terminal(context, config)


def freegsnke_closeout_runners(
    *,
    registry: CapabilityRegistry,
) -> tuple[TaskRunner, ...]:
    return tuple(_FreeGsnkeCloseoutRunner(value) for value in registry.capabilities)


class FreeGsnkeCloseoutRuntimeProvider(CampaignRuntimeProvider):
    """Static provider for one or more post-reveal closeout acts."""

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        configs: tuple[FreeGsnkeCloseoutConfig, ...],
        static_records: tuple[CanonicalRecord, ...],
    ) -> None:
        expected = freegsnke_closeout_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if (
            registry != expected
            or not configs
            or len({value.config_id for value in configs}) != len(configs)
            or len({_record_id(value) for value in static_records}) != len(static_records)
        ):
            raise ValueError("FreeGSNKE closeout provider roster differs")
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.configs = tuple(sorted(configs, key=lambda value: value.config_id))
        self.static_records = tuple(sorted(static_records, key=_record_id))
        self._runners = freegsnke_closeout_runners(registry=registry)
        self.capability_count = len(self._runners)

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("FreeGSNKE closeout registry/source differs")
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("FreeGSNKE closeout plan/source differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        config_refs = {
            task.capability.config.artifact_id: task.capability.config for task in plan.tasks
        }
        config_by_hash = {value.fingerprint(): value for value in self.configs}
        static_by_artifact = {
            f"artifact.independent-substrate-grounding.freegsnke.closeout.{_record_id(value)}": value
            for value in self.static_records
        }
        values = []
        for artifact_id, spec in sorted(specs.items()):
            config_ref = config_refs.get(artifact_id)
            if config_ref is not None:
                try:
                    record: CanonicalRecord = config_by_hash[config_ref.content_sha256]
                except KeyError as error:
                    raise ValueError("FreeGSNKE closeout config is unknown") from error
            else:
                try:
                    record = static_by_artifact[artifact_id]
                except KeyError as error:
                    raise ValueError("FreeGSNKE closeout input is unknown") from error
            access = (
                OutcomeAccess.OUTCOME_BLIND
                if isinstance(record, IndependentImplementationDossier)
                or isinstance(record, FreeGsnkeCloseoutConfig)
                else OutcomeAccess.EVALUATION_REVEALED
            )
            visibility = (
                VisibilityCeiling.PROSPECTIVE
                if access is OutcomeAccess.OUTCOME_BLIND
                else VisibilityCeiling.OUTCOME_VISIBLE
            )
            parent = ArtifactLineageParent(
                identity=_identity(record),
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
            raise ValueError("FreeGSNKE closeout semantic registry differs")
        del execution_plan
        record_types: dict[str, type[CanonicalRecord]] = {
            FreeGsnkeTargetValidation.SCHEMA: FreeGsnkeTargetValidation,
            FreeGsnkeTargetTerminal.SCHEMA: FreeGsnkeTargetTerminal,
            IndependentSubstrateTargetTerminalHandoff.SCHEMA: IndependentSubstrateTargetTerminalHandoff,
        }
        values = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                record_type = record_types[schema]
                values.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(
                            sorted(
                                value.name
                                for value in fields(record_type)  # type: ignore[arg-type]
                            )
                        ),
                    )
                )
        return tuple(sorted(values, key=lambda value: (value.capability_key, value.payload_schema)))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError("FreeGSNKE closeout adjudication registry differs")
        del execution_plan
        return None


__all__ = [
    "FREEGSNKE_CLOSEOUT_PROVIDER_KEY",
    "FREEGSNKE_NONISSUE_KEY",
    "FREEGSNKE_NONISSUE_STEP_ID",
    "FREEGSNKE_TERMINAL_KEY",
    "FREEGSNKE_TERMINAL_STEP_ID",
    'FreeGsnkeCloseoutConfig',
    "FreeGsnkeCloseoutOperation",
    "FreeGsnkeCloseoutRuntimeProvider",
    "build_freegsnke_closeout_protocol",
    "freegsnke_closeout_candidate_registrations",
    'freegsnke_closeout_study_template',
    "freegsnke_closeout_registry",
    "freegsnke_closeout_runners",
    "freegsnke_closeout_scientific_graph",
    "freegsnke_nonissue_closeout_config",
    "freegsnke_terminal_closeout_config",
]

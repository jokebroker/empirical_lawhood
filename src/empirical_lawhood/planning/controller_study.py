"""Outcome-blind authoring contracts for the sole current controller route."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage, OccurrenceActionWord
from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.causal_contracts import CausalPrefixAssessment
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.laws import ResponseLaw, validate_law_against_system
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.time import InformationCutoff
from empirical_lawhood.planning.evidence_geometry import AtlasAdmissionSpec, ReceiptAdmissionSpec, AtlasReachabilitySpec, ControlledMapReachabilitySpec, AtlasGateReceipt, GatePredicateKind, GatePredicateSpec, ReceiptAdmissionAdmissionCandidateCell, ReceiptAdmissionUtilityDirection, AtlasReachabilityReceipt, ControlledMapReachabilityComparison, derive_receipt_admission_comparison, derive_controlled_map_reachability_comparison
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionSpec, FiniteCertificateReachabilitySpec, FiniteCertificateAdmissionReceiptCorpus, FiniteCertificateReachabilityComparison, derive_finite_certificate_admission_comparison, derive_finite_certificate_reachability_comparison
from empirical_lawhood.planning.receiver_geometry import ReceiverQuotientControlPlan


MAX_CONTROLLER_PROGRAMME_BYTES = 8 * 1024 * 1024
# Complete nine-word streaming profiles measure about 20 MiB at full size.
# Older programme decoders retain their prior limits.
MAX_STREAMING_FINITE_CONTROLLER_PROGRAMME_BYTES = 32 * 1024 * 1024
# The eight-word/two-view native delivery census carries both finite admission and
# reachability custody. Keep its decoder within the existing 16 MiB compiled
# record envelope without widening predecessor programme decoders.
MAX_NATIVE_FINITE_CONTROLLER_PROGRAMME_BYTES = 16 * 1024 * 1024


def _require_sorted_wrapped_receipts(
    values: tuple[CandidateGateReceipt | CandidateReachabilityReceipt, ...],
    *,
    field_name: str,
) -> None:
    identifiers = tuple(value.receipt.receipt_id for value in values)
    if identifiers != tuple(sorted(set(identifiers))):
        raise ValueError(f"{field_name} must have sorted, unique receipt identifiers")


class ImplementationRole(StrEnum):
    ADMISSION_DERIVER = "ADMISSION_DERIVER"
    REACHABILITY_DERIVER = "REACHABILITY_DERIVER"
    SYNTHESIZER = "SYNTHESIZER"
    OBSERVER = "OBSERVER"
    ONLINE_GATE_EVALUATOR = "ONLINE_GATE_EVALUATOR"
    DELIVERY = "DELIVERY"
    OUTCOME_EVALUATOR = "OUTCOME_EVALUATOR"


CONTROLLER_ROLES = frozenset(
    {
        ImplementationRole.ADMISSION_DERIVER,
        ImplementationRole.REACHABILITY_DERIVER,
        ImplementationRole.SYNTHESIZER,
        ImplementationRole.OBSERVER,
        ImplementationRole.ONLINE_GATE_EVALUATOR,
        ImplementationRole.DELIVERY,
    }
)


class UtilityDirection(StrEnum):
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


class CandidatePriorityOrientation(StrEnum):
    EARLIER_WINS = "EARLIER_WINS"


@dataclass(frozen=True, slots=True)
class ImplementationBinding(CanonicalRecord):
    """Authored exact capability requirement, resolved only by runtime composition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/implementation-binding'

    binding_id: str
    role: ImplementationRole
    reference: ExecutableReference
    config_sha256: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")


@dataclass(frozen=True, slots=True)
class ControllerActionBinding(CanonicalRecord):
    "One exact native action word and its local law causal compatibility."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controller-action-binding'

    action_binding_id: str
    action_word: OccurrenceActionWord
    causal_prefix: CausalPrefixAssessment

    def __post_init__(self) -> None:
        validate_stable_id(self.action_binding_id, field_name="action_binding_id")
        if self.causal_prefix.causal_support.action_word != self.action_word:
            raise ValueError("action binding causal support uses another action word")


@dataclass(frozen=True, slots=True)
class CandidateGateReceipt(CanonicalRecord):
    "Candidate/action coordinate around one decoded admission gate receipt."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/candidate-gate-receipt'

    candidate_id: str
    action_binding_id: str
    receipt: AtlasGateReceipt

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_stable_id(self.action_binding_id, field_name="action_binding_id")


@dataclass(frozen=True, slots=True)
class CandidateReachabilityReceipt(CanonicalRecord):
    """Candidate/action coordinate around one decoded reachability receipt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/candidate-reachability-receipt'

    candidate_id: str
    action_binding_id: str
    receipt: AtlasReachabilityReceipt

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_stable_id(self.action_binding_id, field_name="action_binding_id")


@dataclass(frozen=True, slots=True)
class CandidateUtilityDefinition(CanonicalRecord):
    """One receiver-directed, native-unit utility definition for a candidate chart."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/candidate-utility-definition'

    definition_id: str
    receiver_quantity_id: str
    direction: UtilityDirection
    reference_id: str
    horizon_id: str
    native_unit: str
    uncertainty_rule_id: str
    minimum_robust_utility: NamedDecimal
    evaluator: ExecutableReference

    def __post_init__(self) -> None:
        for name, value in (
            ("definition_id", self.definition_id),
            ("receiver_quantity_id", self.receiver_quantity_id),
            ("reference_id", self.reference_id),
            ("horizon_id", self.horizon_id),
            ("uncertainty_rule_id", self.uncertainty_rule_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.minimum_robust_utility.unit != self.native_unit:
            raise ValueError("minimum robust utility uses another native unit")


@dataclass(frozen=True, slots=True)
class CandidateUtilityReceipt(CanonicalRecord):
    """Mechanically derived candidate/member utility from exact outcome-blind inputs."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/candidate-utility-receipt'

    receipt_id: str
    candidate_id: str
    action_binding_id: str
    model_member_id: str
    utility_definition: ObjectIdentity
    predicted_response: NamedDecimal
    reference_response: NamedDecimal
    uncertainty_allowance: NamedDecimal
    derived_utility: NamedDecimal
    information_cutoff: InformationCutoff
    evaluator: ExecutableReference
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("candidate_id", self.candidate_id),
            ("action_binding_id", self.action_binding_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if not self.input_artifacts or not self.evidence_links:
            raise ValueError("utility receipt requires exact input artifacts and evidence")
        units = {
            self.predicted_response.unit,
            self.reference_response.unit,
            self.uncertainty_allowance.unit,
            self.derived_utility.unit,
        }
        if len(units) != 1:
            raise ValueError("utility derivation mixes native units")
        if self.uncertainty_allowance.value < 0:
            raise ValueError("utility uncertainty allowance cannot be negative")
        if self.outcome_access in {
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }:
            raise ValueError("outcome-visible inputs cannot derive admission utility")
        artifacts = {value.artifact_id for value in self.input_artifacts}
        linked = {value for link in self.evidence_links for value in link.artifact_ids}
        if artifacts != linked or self.evaluator.payload.artifact_id not in artifacts:
            raise ValueError("utility receipt does not bind its exact evaluator/input corpus")

    def validate_against(self, definition: CandidateUtilityDefinition) -> None:
        if self.utility_definition != ObjectIdentity.from_record(
            definition.definition_id, definition
        ):
            raise ValueError("utility receipt binds another utility definition")
        if self.evaluator != definition.evaluator:
            raise ValueError("utility receipt uses another evaluator")
        if self.derived_utility.unit != definition.native_unit:
            raise ValueError("utility receipt uses another native unit")
        signed = (
            self.predicted_response.value - self.reference_response.value
            if definition.direction is UtilityDirection.HIGHER_IS_BETTER
            else self.reference_response.value - self.predicted_response.value
        )
        expected = signed - self.uncertainty_allowance.value
        if self.derived_utility.value != expected:
            raise ValueError("utility value is not mechanically derived")


@dataclass(frozen=True, slots=True)
class AtlasActionCandidate(CanonicalRecord):
    """One finite native action candidate over the complete declared member set."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-action-candidate'

    candidate_id: str
    decision_cell_id: str
    admission_cell_id: str
    action_binding_id: str
    action_bound_ids: tuple[str, ...]
    model_member_ids: tuple[str, ...]
    gate_receipts: tuple[CandidateGateReceipt, ...]
    reachability_receipts: tuple[CandidateReachabilityReceipt, ...]
    utility_receipts: tuple[CandidateUtilityReceipt, ...]
    priority_rank: int

    def __post_init__(self) -> None:
        for name, value in (
            ("candidate_id", self.candidate_id),
            ("decision_cell_id", self.decision_cell_id),
            ("admission_cell_id", self.admission_cell_id),
            ("action_binding_id", self.action_binding_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.action_bound_ids, field_name="action_bound_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.model_member_ids, field_name="model_member_ids", allow_empty=False
        )
        _require_sorted_wrapped_receipts(self.gate_receipts, field_name="gate_receipts")
        _require_sorted_wrapped_receipts(
            self.reachability_receipts, field_name="reachability_receipts"
        )
        require_sorted_unique_ids(
            self.utility_receipts,
            attribute="receipt_id",
            field_name="utility_receipts",
        )
        if self.priority_rank < 0:
            raise ValueError("candidate priority rank must be nonnegative")
        if any(
            value.candidate_id != self.candidate_id
            or value.action_binding_id != self.action_binding_id
            or value.receipt.cell_id != self.admission_cell_id
            or value.receipt.model_member_id not in self.model_member_ids
            for value in self.gate_receipts
        ):
            raise ValueError("candidate gate receipt uses another coordinate")
        if any(
            value.candidate_id != self.candidate_id
            or value.action_binding_id != self.action_binding_id
            or value.receipt.admission_cell_id != self.admission_cell_id
            or value.receipt.model_member_id not in self.model_member_ids
            for value in self.reachability_receipts
        ):
            raise ValueError("candidate reachability receipt uses another coordinate")
        if any(
            value.candidate_id != self.candidate_id
            or value.action_binding_id != self.action_binding_id
            or value.model_member_id not in self.model_member_ids
            for value in self.utility_receipts
        ):
            raise ValueError("candidate utility receipt uses another coordinate")
        expected_gates = {
            (member, kind) for member in self.model_member_ids for kind in AdmissionGateKind
        }
        observed_gates = {
            (value.receipt.model_member_id, value.receipt.predicate.gate_kind)
            for value in self.gate_receipts
        }
        if observed_gates != expected_gates or len(self.gate_receipts) != len(expected_gates):
            raise ValueError("candidate gate grid is incomplete or contains extras")
        if {value.receipt.model_member_id for value in self.reachability_receipts} != set(
            self.model_member_ids
        ) or len(self.reachability_receipts) != len(self.model_member_ids):
            raise ValueError("candidate reachability grid is incomplete or contains extras")
        if {value.model_member_id for value in self.utility_receipts} != set(
            self.model_member_ids
        ) or len(self.utility_receipts) != len(self.model_member_ids):
            raise ValueError("candidate utility grid is incomplete or contains extras")


@dataclass(frozen=True, slots=True)
class AtlasActionCandidateChart(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-action-candidate-chart'

    chart_id: str
    model_member_ids: tuple[str, ...]
    nominal_member_id: str
    utility_definition: CandidateUtilityDefinition
    priority_orientation: CandidatePriorityOrientation
    candidates: tuple[AtlasActionCandidate, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.chart_id, field_name="chart_id")
        require_sorted_unique_strings(
            self.model_member_ids, field_name="model_member_ids", allow_empty=False
        )
        validate_stable_id(self.nominal_member_id, field_name="nominal_member_id")
        if self.nominal_member_id not in self.model_member_ids:
            raise ValueError("nominal member is absent from candidate chart")
        if self.priority_orientation is not CandidatePriorityOrientation.EARLIER_WINS:
            raise ValueError("candidate chart priority must be EARLIER_WINS")
        require_sorted_unique_ids(
            self.candidates, attribute="candidate_id", field_name="candidates"
        )
        if not self.candidates:
            raise ValueError("candidate chart cannot be empty")
        for candidate in self.candidates:
            if candidate.model_member_ids != self.model_member_ids:
                raise ValueError("candidate member roster differs from chart")
            for receipt in candidate.utility_receipts:
                receipt.validate_against(self.utility_definition)
        by_cell: dict[str, set[int]] = {}
        for candidate in self.candidates:
            ranks = by_cell.setdefault(candidate.decision_cell_id, set())
            if candidate.priority_rank in ranks:
                raise ValueError("candidate priority rank is duplicated within a decision cell")
            ranks.add(candidate.priority_rank)


@dataclass(frozen=True, slots=True)
class DeliveryEquivalenceSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/delivery-equivalence-spec'

    equivalence_id: str
    require_exact_word_identity: bool
    require_all_delivery_stages: bool
    stage_value_tolerance: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.equivalence_id, field_name="equivalence_id")
        if not self.require_exact_word_identity or not self.require_all_delivery_stages:
            raise ValueError("scientific delivery equivalence requires exact word and all stages")
        if self.stage_value_tolerance.value < 0:
            raise ValueError("delivery tolerance cannot be negative")


@dataclass(frozen=True, slots=True)
class DeliveryStageValueTolerance(CanonicalRecord):
    """One native value tolerance keyed to an exact delivery stage."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/delivery-stage-value-tolerance'

    stage: ActionDeliveryStage
    tolerance: NamedDecimal

    def __post_init__(self) -> None:
        if not isinstance(self.stage, ActionDeliveryStage):
            raise ValueError("delivery-stage tolerance names an unknown stage")
        if self.tolerance.value < 0:
            raise ValueError("delivery-stage tolerance cannot be negative")


@dataclass(frozen=True, slots=True)
class StageAwareDeliveryEquivalenceSpec(CanonicalRecord):
    """Exact word/stage equivalence with a noncompensating tolerance per stage."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/stage-aware-delivery-equivalence-spec'

    equivalence_id: str
    require_exact_word_identity: bool
    require_all_delivery_stages: bool
    stage_value_tolerances: tuple[DeliveryStageValueTolerance, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.equivalence_id, field_name="equivalence_id")
        if not self.require_exact_word_identity or not self.require_all_delivery_stages:
            raise ValueError("scientific delivery equivalence requires exact word and all stages")
        expected_stages = tuple(ActionDeliveryStage)
        if tuple(value.stage for value in self.stage_value_tolerances) != expected_stages:
            raise ValueError("stage-aware delivery equivalence requires all stages in enum order")
        if len({value.tolerance.value_id for value in self.stage_value_tolerances}) != len(
            self.stage_value_tolerances
        ):
            raise ValueError("stage-aware delivery tolerances reuse a value identity")
        if len({value.tolerance.unit for value in self.stage_value_tolerances}) != 1:
            raise ValueError("stage-aware delivery tolerances mix native units")

    def tolerance_for(self, stage: ActionDeliveryStage) -> NamedDecimal:
        return self.stage_value_tolerances[tuple(ActionDeliveryStage).index(stage)].tolerance

    @property
    def native_unit(self) -> str:
        """The one native unit shared by every noncompensating stage bound."""

        return self.stage_value_tolerances[0].tolerance.unit

    def validate_action_word(self, action_word: OccurrenceActionWord) -> None:
        """Require a nonempty exact word expressed in the bound native unit."""

        if not action_word.occurrences:
            raise ValueError("stage-aware delivery equivalence requires a nonempty action word")
        if any(
            event.native_unit != self.native_unit
            for occurrence in action_word.occurrences
            for event in (
                occurrence.requested,
                occurrence.accepted,
                occurrence.applied,
                occurrence.realized,
            )
        ):
            raise ValueError("stage-aware delivery tolerance uses another native action unit")


@dataclass(frozen=True, slots=True)
class AtlasMeasuredHoldFibre(CanonicalRecord):
    """One independently evidenced native HOLD; absence means HOLD unavailable."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-measured-hold-fibre'

    hold_fibre_id: str
    decision_cell_id: str
    admission_cell_id: str
    action_binding_id: str
    model_member_ids: tuple[str, ...]
    gate_receipts: tuple[CandidateGateReceipt, ...]
    reachability_receipts: tuple[CandidateReachabilityReceipt, ...]
    viable_direction_id: str
    calibration_id: str
    null_tolerance: NamedDecimal
    delivery_equivalence: DeliveryEquivalenceSpec

    def __post_init__(self) -> None:
        for name, value in (
            ("hold_fibre_id", self.hold_fibre_id),
            ("decision_cell_id", self.decision_cell_id),
            ("admission_cell_id", self.admission_cell_id),
            ("action_binding_id", self.action_binding_id),
            ("viable_direction_id", self.viable_direction_id),
            ("calibration_id", self.calibration_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.model_member_ids, field_name="model_member_ids", allow_empty=False
        )
        _require_sorted_wrapped_receipts(self.gate_receipts, field_name="gate_receipts")
        _require_sorted_wrapped_receipts(
            self.reachability_receipts, field_name="reachability_receipts"
        )
        expected = {
            (member, kind) for member in self.model_member_ids for kind in AdmissionGateKind
        }
        observed = {
            (value.receipt.model_member_id, value.receipt.predicate.gate_kind)
            for value in self.gate_receipts
        }
        if observed != expected or len(self.gate_receipts) != len(expected):
            raise ValueError("measured HOLD gate grid is incomplete")
        if {value.receipt.model_member_id for value in self.reachability_receipts} != set(
            self.model_member_ids
        ) or len(self.reachability_receipts) != len(self.model_member_ids):
            raise ValueError("measured HOLD reachability grid is incomplete")
        if any(
            value.candidate_id != self.hold_fibre_id
            or value.action_binding_id != self.action_binding_id
            or value.receipt.cell_id != self.admission_cell_id
            for value in self.gate_receipts
        ) or any(
            value.candidate_id != self.hold_fibre_id
            or value.action_binding_id != self.action_binding_id
            or value.receipt.admission_cell_id != self.admission_cell_id
            for value in self.reachability_receipts
        ):
            raise ValueError("measured HOLD evidence uses another coordinate")
        if self.null_tolerance.value < 0:
            raise ValueError("HOLD null tolerance cannot be negative")


@dataclass(frozen=True, slots=True)
class OnlineSupportMonitorSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/online-support-monitor-spec'

    monitor_id: str
    support_ids: tuple[str, ...]
    check_rule: str
    invalidation_reason_code: str

    def __post_init__(self) -> None:
        validate_stable_id(self.monitor_id, field_name="monitor_id")
        validate_stable_id(self.invalidation_reason_code, field_name="invalidation_reason_code")
        require_sorted_unique_strings(self.support_ids, field_name="support_ids", allow_empty=False)
        validate_nonempty(self.check_rule, field_name="check_rule")


@dataclass(frozen=True, slots=True)
class ReidentificationTriggerSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/reidentification-trigger-spec'

    trigger_id: str
    condition: str
    invalidates_chart: bool
    invalidation_reason_code: str

    def __post_init__(self) -> None:
        validate_stable_id(self.trigger_id, field_name="trigger_id")
        validate_stable_id(self.invalidation_reason_code, field_name="invalidation_reason_code")
        validate_nonempty(self.condition, field_name="condition")
        if not self.invalidates_chart:
            raise ValueError("a re-identification trigger must invalidate its chart")


@dataclass(frozen=True, slots=True)
class AtlasControllerSynthesisPlan(CanonicalRecord):
    "Exact outcome-blind admission construction and runtime input."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-controller-synthesis-plan'

    synthesis_id: str
    atlas: ObjectIdentity
    admission_comparison: ObjectIdentity
    reachability_comparison: ObjectIdentity
    model_set: ObjectIdentity
    candidate_chart: AtlasActionCandidateChart
    admission_deriver_binding_id: str
    reachability_deriver_binding_id: str
    synthesizer_binding_id: str
    observer_binding_id: str
    online_gate_evaluator_binding_id: str
    delivery_binding_id: str
    observation_quantity_ids: tuple[str, ...]
    support_monitors: tuple[OnlineSupportMonitorSpec, ...]
    reidentification_triggers: tuple[ReidentificationTriggerSpec, ...]
    authority_policy_id: str
    worst_case_latency_seconds: Decimal
    deadline_seconds: Decimal
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("synthesis_id", self.synthesis_id),
            ("admission_deriver_binding_id", self.admission_deriver_binding_id),
            ("reachability_deriver_binding_id", self.reachability_deriver_binding_id),
            ("synthesizer_binding_id", self.synthesizer_binding_id),
            ("observer_binding_id", self.observer_binding_id),
            ("online_gate_evaluator_binding_id", self.online_gate_evaluator_binding_id),
            ("delivery_binding_id", self.delivery_binding_id),
            ("authority_policy_id", self.authority_policy_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.observation_quantity_ids,
            field_name="observation_quantity_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.support_monitors, attribute="monitor_id", field_name="support_monitors"
        )
        require_sorted_unique_ids(
            self.reidentification_triggers,
            attribute="trigger_id",
            field_name="reidentification_triggers",
        )
        if not self.support_monitors or not self.reidentification_triggers:
            raise ValueError("controller synthesis requires monitors and triggers")
        validate_decimal(
            self.worst_case_latency_seconds,
            field_name="worst_case_latency_seconds",
            minimum=Decimal(0),
        )
        validate_decimal(self.deadline_seconds, field_name="deadline_seconds", minimum=Decimal(0))
        if self.deadline_seconds == 0:
            raise ValueError("controller synthesis deadline must be positive")
        if self.worst_case_latency_seconds > self.deadline_seconds:
            raise ValueError("declared worst-case latency exceeds runtime deadline")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("controller synthesis plan must remain admission")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible work cannot synthesize a controller")


@dataclass(frozen=True, slots=True)
class EvaluatorBoundarySpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evaluator-boundary-spec'

    boundary_id: str
    outcome_evaluator_binding_id: str
    outcome_custodian_role_id: str
    reveal_authority_role_id: str
    evaluator_role_id: str
    sealed_outcome_schema: str
    revealed_outcome_schema: str
    raw_outcome_quantity_ids: tuple[str, ...]
    outcome_predicates: tuple[GatePredicateSpec, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("boundary_id", self.boundary_id),
            ("outcome_evaluator_binding_id", self.outcome_evaluator_binding_id),
            ("outcome_custodian_role_id", self.outcome_custodian_role_id),
            ("reveal_authority_role_id", self.reveal_authority_role_id),
            ("evaluator_role_id", self.evaluator_role_id),
        ):
            validate_stable_id(value, field_name=name)
        if (
            len(
                {
                    self.outcome_custodian_role_id,
                    self.reveal_authority_role_id,
                    self.evaluator_role_id,
                }
            )
            != 3
        ):
            raise ValueError("custody, reveal and evaluation roles must be distinct")
        validate_schema(self.sealed_outcome_schema)
        validate_schema(self.revealed_outcome_schema)
        require_sorted_unique_strings(
            self.raw_outcome_quantity_ids,
            field_name="raw_outcome_quantity_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.outcome_predicates, attribute="predicate_id", field_name="outcome_predicates"
        )
        if not self.outcome_predicates:
            raise ValueError("evaluator boundary requires outcome predicates")
        if not {value.quantity_id for value in self.outcome_predicates} <= set(
            self.raw_outcome_quantity_ids
        ):
            raise ValueError("outcome predicate requires an undeclared raw quantity")
        scalar = {
            GatePredicateKind.SCALAR_AT_LEAST,
            GatePredicateKind.SCALAR_AT_MOST,
            GatePredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL,
        }
        if any(value.predicate_kind not in scalar for value in self.outcome_predicates):
            raise ValueError("raw outcome boundary requires scalar predicates")


@dataclass(frozen=True, slots=True)
class EvaluationStratumSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evaluation-stratum-spec'

    stratum_id: str
    independent_unit_ids: tuple[str, ...]
    minimum_attempted_units: int

    def __post_init__(self) -> None:
        validate_stable_id(self.stratum_id, field_name="stratum_id")
        require_sorted_unique_strings(
            self.independent_unit_ids, field_name="independent_unit_ids", allow_empty=False
        )
        if self.minimum_attempted_units < 1:
            raise ValueError("stratum minimum attempted units must be positive")
        if self.minimum_attempted_units > len(self.independent_unit_ids):
            raise ValueError("stratum minimum exceeds its independent-unit roster")


@dataclass(frozen=True, slots=True)
class ProspectiveControllerEvaluationPlan(CanonicalRecord):
    "Predeclared matched controller-use design; compilation binds it but never executes it."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-controller-evaluation-plan'

    evaluation_plan_id: str
    evaluator_boundary: EvaluatorBoundarySpec
    independent_unit_type_id: str
    independent_unit_ids: tuple[str, ...]
    model_member_ids: tuple[str, ...]
    controller_branch_id: str
    reference_branch_id: str
    hold_calibration_branch_id: str | None
    causal_cutoff_id: str
    commitment_point_id: str
    post_cutoff_action_window_id: str
    effect_quantity_id: str
    effect_native_unit: str
    favorable_direction: UtilityDirection
    robust_reducer_id: str
    hold_null_tolerance: NamedDecimal
    strata: tuple[EvaluationStratumSpec, ...]
    minimum_attempted_units: int
    alpha: Decimal
    one_sided_critical_value: Decimal
    materiality: NamedDecimal
    interval_method_id: str
    multiplicity_family_id: str
    maximum_claim_ceiling: str
    intent_to_treat: bool
    result_hierarchy: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluation_plan_id", self.evaluation_plan_id),
            ("independent_unit_type_id", self.independent_unit_type_id),
            ("controller_branch_id", self.controller_branch_id),
            ("reference_branch_id", self.reference_branch_id),
            ("causal_cutoff_id", self.causal_cutoff_id),
            ("commitment_point_id", self.commitment_point_id),
            ("post_cutoff_action_window_id", self.post_cutoff_action_window_id),
            ("effect_quantity_id", self.effect_quantity_id),
            ("robust_reducer_id", self.robust_reducer_id),
            ("interval_method_id", self.interval_method_id),
            ("multiplicity_family_id", self.multiplicity_family_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.hold_calibration_branch_id is not None:
            validate_stable_id(
                self.hold_calibration_branch_id, field_name="hold_calibration_branch_id"
            )
        validate_nonempty(self.effect_native_unit, field_name="effect_native_unit")
        validate_nonempty(self.maximum_claim_ceiling, field_name="maximum_claim_ceiling")
        require_sorted_unique_strings(
            self.independent_unit_ids,
            field_name="independent_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.model_member_ids, field_name="model_member_ids", allow_empty=False
        )
        require_sorted_unique_ids(self.strata, attribute="stratum_id", field_name="strata")
        if self.result_hierarchy != (
            "CONTROLLER_USE_UNSAFE",
            "CONTROLLER_USE_DELIVERY_INVALID",
            "CONTROLLER_USE_TECHNICAL_FAILURE",
            "CONTROLLER_USE_UNEVALUABLE",
            "CONTROLLER_USE_PARTIAL_OR_HETEROGENEOUS",
            "CONTROLLER_USE_HOLD_DOMINANT",
            "CONTROLLER_USE_VALIDATED",
            "CONTROLLER_USE_POSITIVE_BUT_BELOW_MATERIALITY",
            "CONTROLLER_USE_NEGATIVE",
        ):
            raise ValueError("controller use result hierarchy differs from the frozen precedence")
        if self.robust_reducer_id != "memberwise-minimum":
            raise ValueError("current controller use robust reducer is memberwise-minimum")
        if self.minimum_attempted_units < 1 or self.minimum_attempted_units > len(
            self.independent_unit_ids
        ):
            raise ValueError("invalid minimum attempted independent-unit count")
        validate_decimal(self.alpha, field_name="alpha", minimum=Decimal(0))
        if self.alpha <= 0 or self.alpha >= 1:
            raise ValueError("alpha must lie strictly between zero and one")
        validate_decimal(
            self.one_sided_critical_value,
            field_name="one_sided_critical_value",
            minimum=Decimal(0),
        )
        if self.one_sided_critical_value == 0:
            raise ValueError("one-sided critical value must be positive")
        if self.materiality.unit != self.effect_native_unit:
            raise ValueError("materiality uses another native unit")
        if self.hold_null_tolerance.unit != self.effect_native_unit:
            raise ValueError("HOLD tolerance uses another native unit")
        if self.hold_null_tolerance.value < 0:
            raise ValueError("HOLD tolerance cannot be negative")
        if not self.intent_to_treat:
            raise ValueError("current controller use evaluation requires intent-to-treat accounting")
        roster = set(self.independent_unit_ids)
        if (
            self.strata
            and set.union(*(set(value.independent_unit_ids) for value in self.strata)) != roster
        ):
            raise ValueError("evaluation strata must cover the exact independent-unit roster")


@dataclass(frozen=True, slots=True)
class AtlasControllerStudy(CanonicalRecord):
    """The sole current outcome-blind controller authoring schema."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/atlas-controller-study'

    study_id: str
    compiler_release_id: str
    system: SystemSpec
    law: ResponseLaw
    action_bindings: tuple[ControllerActionBinding, ...]
    admission: AtlasAdmissionSpec
    reachability: AtlasReachabilitySpec
    synthesis: AtlasControllerSynthesisPlan
    implementations: tuple[ImplementationBinding, ...]
    measured_hold_fibre: AtlasMeasuredHoldFibre | None
    prospective_evaluation: ProspectiveControllerEvaluationPlan | None
    compilation_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    receiver_quotient: ReceiverQuotientControlPlan | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.study_id, field_name='study_id')
        validate_stable_id(self.compiler_release_id, field_name="compiler_release_id")
        validate_law_against_system(self.law, self.system)
        if (
            self.admission.atlas.system_id != self.system.system_id
            or self.admission.atlas.world_id != self.system.world.world_id
            or self.admission.model_set.target_world_id != self.system.world.world_id
            or not any(
                value.law_id == self.law.law_id and value.fingerprint() == self.law.fingerprint()
                for value in self.admission.atlas.laws
            )
        ):
            raise ValueError("controller programme system/law/admission scope differs")
        if self.reachability.model_set != self.admission.model_set:
            raise ValueError("admission and reachability model sets differ")
        require_sorted_unique_ids(
            self.action_bindings,
            attribute="action_binding_id",
            field_name="action_bindings",
        )
        if not self.action_bindings:
            raise ValueError("controller programme requires action bindings")
        actions = {value.action_binding_id: value for value in self.action_bindings}
        for binding in self.action_bindings:
            support = binding.causal_prefix.causal_support
            word = binding.action_word
            if (
                support.relation != self.law.relation
                or support.world_id != self.law.world_id
                or word.receiver_id not in self.law.relation.receiver_quantity_ids
                or word.horizon_id != self.law.relation.horizon.horizon_id
            ):
                raise ValueError("controller action binding differs from L(D,H,A,R,tau)")
        chart = self.synthesis.candidate_chart
        if chart.model_member_ids != self.admission.model_set.member_view_ids:
            raise ValueError("candidate chart differs from the declared model set")
        if chart.nominal_member_id != self.admission.nominal_model_member_id:
            raise ValueError("candidate chart nominal member differs from admission")
        admission_cells = {value.cell_id: value for value in self.admission.candidate_cells}
        for candidate in chart.candidates:
            action = actions.get(candidate.action_binding_id)
            cell = admission_cells.get(candidate.admission_cell_id)
            if action is None or cell is None:
                raise ValueError("candidate references an unknown action or admission cell")
            if set(candidate.action_bound_ids) != set(cell.action_bound_ids):
                raise ValueError("candidate action bounds differ from admission cell")
            if action.action_word.receiver_id not in self.law.relation.receiver_quantity_ids:
                raise ValueError("candidate action word targets another receiver")
        hold = self.measured_hold_fibre
        if hold is not None:
            if (
                hold.action_binding_id not in actions
                or hold.admission_cell_id not in admission_cells
            ):
                raise ValueError("measured HOLD references an unknown action or admission cell")
            if hold.model_member_ids != chart.model_member_ids:
                raise ValueError("measured HOLD member roster differs from candidate chart")
        require_sorted_unique_ids(
            self.implementations, attribute="binding_id", field_name="implementations"
        )
        roles = {value.role for value in self.implementations}
        expected_roles = set(CONTROLLER_ROLES)
        if self.prospective_evaluation is not None:
            expected_roles.add(ImplementationRole.OUTCOME_EVALUATOR)
        if roles != expected_roles or len(self.implementations) != len(expected_roles):
            raise ValueError("controller programme implementation role set is incomplete")
        by_id = {value.binding_id: value for value in self.implementations}
        expected_bindings = (
            (self.synthesis.admission_deriver_binding_id, ImplementationRole.ADMISSION_DERIVER),
            (
                self.synthesis.reachability_deriver_binding_id,
                ImplementationRole.REACHABILITY_DERIVER,
            ),
            (self.synthesis.synthesizer_binding_id, ImplementationRole.SYNTHESIZER),
            (self.synthesis.observer_binding_id, ImplementationRole.OBSERVER),
            (
                self.synthesis.online_gate_evaluator_binding_id,
                ImplementationRole.ONLINE_GATE_EVALUATOR,
            ),
            (self.synthesis.delivery_binding_id, ImplementationRole.DELIVERY),
        )
        if any(
            by_id.get(binding_id, None) is None or by_id[binding_id].role is not role
            for binding_id, role in expected_bindings
        ):
            raise ValueError("controller role binding is missing or has the wrong role")
        evaluation = self.prospective_evaluation
        if evaluation is not None:
            evaluator_id = evaluation.evaluator_boundary.outcome_evaluator_binding_id
            evaluator = by_id.get(evaluator_id)
            if evaluator is None or evaluator.role is not ImplementationRole.OUTCOME_EVALUATOR:
                raise ValueError("controller use plan lacks its exact evaluator binding")
            if any(
                predicate.evaluator != evaluator.reference
                for predicate in evaluation.evaluator_boundary.outcome_predicates
            ):
                raise ValueError("controller use outcome predicate uses another evaluator")
            if evaluation.model_member_ids != chart.model_member_ids:
                raise ValueError("controller use member roster differs from compiled candidate chart")
        if self.synthesis.atlas != ObjectIdentity.from_record(
            self.admission.atlas.atlas_id, self.admission.atlas
        ) or self.synthesis.model_set != ObjectIdentity.from_record(
            self.admission.model_set.model_set_id, self.admission.model_set
        ):
            raise ValueError("synthesis binds another atlas or model set")
        if self.compilation_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("controller compilation is capped at admission")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("controller programme compilation must remain outcome-blind")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible work cannot compile a controller")
        quotient = self.receiver_quotient
        if quotient is not None:
            observer = by_id.get(quotient.observer_binding_id)
            if observer is None or observer.role is not ImplementationRole.OBSERVER:
                raise ValueError("receiver quotient uses another observer binding")
            experiment = quotient.observation_experiment
            if (
                experiment.world_id != self.system.world.world_id
                or experiment.receiver_id not in self.law.relation.receiver_quantity_ids
                or experiment.horizon_id != self.law.relation.horizon.horizon_id
            ):
                raise ValueError("receiver quotient changes L(D,H,A,R,tau)")


@dataclass(frozen=True, slots=True)
class AdmissionActionCandidate(CanonicalRecord):
    "One controller candidate selecting an exact member-local admission action/support cell."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-action-candidate'

    candidate_id: str
    decision_cell_id: str
    admission_candidate_cell: ObjectIdentity
    priority_rank: int

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_stable_id(self.decision_cell_id, field_name="decision_cell_id")
        if self.admission_candidate_cell.object_schema != ReceiptAdmissionAdmissionCandidateCell.SCHEMA:
            raise ValueError("controller candidate requires an exact admission candidate cell")
        if self.priority_rank < 0:
            raise ValueError("candidate priority rank must be nonnegative")


@dataclass(frozen=True, slots=True)
class AdmissionActionCandidateChart(CanonicalRecord):
    "Finite chart whose scientific operands remain in the exact raw admission corpus."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-action-candidate-chart'

    chart_id: str
    model_set: ObjectIdentity
    nominal_denominator_member_id: str
    utility_definition: ObjectIdentity
    utility_direction: ReceiptAdmissionUtilityDirection
    minimum_robust_utility: NamedDecimal
    priority_orientation: CandidatePriorityOrientation
    candidates: tuple[AdmissionActionCandidate, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.chart_id, field_name="chart_id")
        validate_stable_id(
            self.nominal_denominator_member_id,
            field_name="nominal_denominator_member_id",
        )
        if self.model_set.object_schema != 'empirical-lawhood/kernel/model-set-spec':
            raise ValueError("controller chart requires ModelSetSpec")
        if self.priority_orientation is not CandidatePriorityOrientation.EARLIER_WINS:
            raise ValueError("candidate chart priority must be EARLIER_WINS")
        require_sorted_unique_ids(
            self.candidates,
            attribute="candidate_id",
            field_name="candidates",
        )
        if not self.candidates:
            raise ValueError("candidate chart cannot be empty")
        by_cell: dict[str, set[int]] = {}
        admission_candidate_cell_ids: set[str] = set()
        for candidate in self.candidates:
            ranks = by_cell.setdefault(candidate.decision_cell_id, set())
            if candidate.priority_rank in ranks:
                raise ValueError("candidate priority rank is duplicated within a decision cell")
            ranks.add(candidate.priority_rank)
            if candidate.admission_candidate_cell.object_id in admission_candidate_cell_ids:
                raise ValueError("one admission candidate cell cannot appear as several active candidates")
            admission_candidate_cell_ids.add(candidate.admission_candidate_cell.object_id)


@dataclass(frozen=True, slots=True)
class AdmissionMeasuredHoldFibre(CanonicalRecord):
    "Independently calibrated HOLD selecting one complete member-local admission cell."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-measured-hold-fibre'

    hold_fibre_id: str
    decision_cell_id: str
    admission_candidate_cell: ObjectIdentity
    calibration_id: str
    calibration_evidence_links: tuple[EvidenceLink, ...]
    null_tolerance: NamedDecimal
    delivery_equivalence: DeliveryEquivalenceSpec | StageAwareDeliveryEquivalenceSpec

    def __post_init__(self) -> None:
        for name, value in (
            ("hold_fibre_id", self.hold_fibre_id),
            ("decision_cell_id", self.decision_cell_id),
            ("calibration_id", self.calibration_id),
        ):
            validate_stable_id(value, field_name=name)
        if not isinstance(
            self.delivery_equivalence,
            (DeliveryEquivalenceSpec, StageAwareDeliveryEquivalenceSpec),
        ):
            raise ValueError("measured HOLD uses an unknown delivery-equivalence schema")
        if self.admission_candidate_cell.object_schema != ReceiptAdmissionAdmissionCandidateCell.SCHEMA:
            raise ValueError("measured HOLD requires an exact admission candidate cell")
        require_sorted_unique_ids(
            self.calibration_evidence_links,
            attribute="link_id",
            field_name="calibration_evidence_links",
        )
        if not self.calibration_evidence_links:
            raise ValueError("measured HOLD requires independent calibration evidence")
        if self.null_tolerance.value < 0:
            raise ValueError("HOLD null tolerance cannot be negative")


@dataclass(frozen=True, slots=True)
class AdmissionControllerSynthesisPlan(CanonicalRecord):
    "Outcome-blind synthesis over one exact member-local raw admission corpus."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-controller-synthesis-plan'
    ADMISSION_CORPUS_SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controlled-map-admission-receipt-corpus'

    synthesis_id: str
    atlas: ObjectIdentity
    admission_corpus: ObjectIdentity
    admission_comparison: ObjectIdentity
    reachability_comparison: ObjectIdentity
    model_set: ObjectIdentity
    candidate_chart: AdmissionActionCandidateChart
    admission_deriver_binding_id: str
    reachability_deriver_binding_id: str
    synthesizer_binding_id: str
    observer_binding_id: str
    online_gate_evaluator_binding_id: str
    delivery_binding_id: str
    observation_quantity_ids: tuple[str, ...]
    support_monitors: tuple[OnlineSupportMonitorSpec, ...]
    reidentification_triggers: tuple[ReidentificationTriggerSpec, ...]
    authority_policy_id: str
    worst_case_latency_seconds: Decimal
    deadline_seconds: Decimal
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("synthesis_id", self.synthesis_id),
            ("admission_deriver_binding_id", self.admission_deriver_binding_id),
            ("reachability_deriver_binding_id", self.reachability_deriver_binding_id),
            ("synthesizer_binding_id", self.synthesizer_binding_id),
            ("observer_binding_id", self.observer_binding_id),
            ("online_gate_evaluator_binding_id", self.online_gate_evaluator_binding_id),
            ("delivery_binding_id", self.delivery_binding_id),
            ("authority_policy_id", self.authority_policy_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.atlas.object_schema != 'empirical-lawhood/kernel/response-atlas':
            raise ValueError("synthesis requires an exact response atlas")
        if self.admission_corpus.object_schema != self.ADMISSION_CORPUS_SCHEMA:
            raise ValueError("synthesis requires an exact raw admission corpus")
        if self.model_set.object_schema != 'empirical-lawhood/kernel/model-set-spec':
            raise ValueError("synthesis requires ModelSetSpec")
        require_sorted_unique_strings(
            self.observation_quantity_ids,
            field_name="observation_quantity_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.support_monitors,
            attribute="monitor_id",
            field_name="support_monitors",
        )
        require_sorted_unique_ids(
            self.reidentification_triggers,
            attribute="trigger_id",
            field_name="reidentification_triggers",
        )
        if not self.support_monitors or not self.reidentification_triggers:
            raise ValueError("controller synthesis requires monitors and triggers")
        validate_decimal(
            self.worst_case_latency_seconds,
            field_name="worst_case_latency_seconds",
            minimum=Decimal(0),
        )
        validate_decimal(
            self.deadline_seconds,
            field_name="deadline_seconds",
            minimum=Decimal(0),
        )
        if self.deadline_seconds == 0:
            raise ValueError("controller synthesis deadline must be positive")
        if self.worst_case_latency_seconds > self.deadline_seconds:
            raise ValueError("declared worst-case latency exceeds runtime deadline")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("controller synthesis plan must remain admission")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible work cannot synthesize a controller")


@dataclass(frozen=True, slots=True)
class AdmissionControllerStudy(CanonicalRecord):
    "Sole new controller authoring route over member-local laws and raw admission."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/admission-controller-study'
    VERSION: ClassVar[str] = '1.0.0'

    study_id: str
    compiler_release_id: str
    system: SystemSpec
    action_bindings: tuple[ControllerActionBinding, ...]
    admission: ReceiptAdmissionSpec
    reachability: ControlledMapReachabilitySpec
    synthesis: AdmissionControllerSynthesisPlan
    implementations: tuple[ImplementationBinding, ...]
    measured_hold_fibre: AdmissionMeasuredHoldFibre | None
    prospective_evaluation: ObjectIdentity | None
    compilation_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        _validate_admission_controller_study(self)

    def _validate_candidate_utility(self, cell: ReceiptAdmissionAdmissionCandidateCell) -> None:
        _validate_admission_candidate_utility(self, cell)


class NativeCandidateKind(StrEnum):
    ACT = "ACT"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class ActHoldActionCandidate(AdmissionActionCandidate):
    """A finite native word with explicit ACT/HOLD semantics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/act-hold-action-candidate'

    kind: NativeCandidateKind

    def __post_init__(self) -> None:
        AdmissionActionCandidate.__post_init__(self)
        if not isinstance(self.kind, NativeCandidateKind):
            raise ValueError("finite candidate requires explicit ACT/HOLD kind")


@dataclass(frozen=True, slots=True)
class ActHoldActionCandidateChart(AdmissionActionCandidateChart):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/act-hold-action-candidate-chart'

    candidates: tuple[ActHoldActionCandidate, ...]


@dataclass(frozen=True, slots=True)
class ActHoldControllerSynthesisPlan(AdmissionControllerSynthesisPlan):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/act-hold-controller-synthesis-plan'
    ADMISSION_CORPUS_SCHEMA: ClassVar[str] = FiniteCertificateAdmissionReceiptCorpus.SCHEMA

    candidate_chart: ActHoldActionCandidateChart


@dataclass(frozen=True, slots=True)
class ControllerInstanceBinding(CanonicalRecord):
    """Late observations instantiate a frozen recipe; no coefficient or rule changes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/controller-instance-binding'

    instance_id: str
    frozen_recipe: ObjectIdentity
    law_payloads: tuple[ArtifactIdentity, ...]
    joint_calibrations: tuple[ObjectIdentity, ...]
    public_handoff: ObjectIdentity
    task_functional: ObjectIdentity
    frame_translation: ObjectIdentity
    bound_observation: ObjectIdentity
    information_cutoff: InformationCutoff

    def __post_init__(self) -> None:
        validate_stable_id(self.instance_id, field_name="instance_id")
        require_sorted_unique_ids(
            self.law_payloads, attribute="artifact_id", field_name="law_payloads"
        )
        require_sorted_unique_ids(
            self.joint_calibrations, attribute="object_id", field_name="joint_calibrations"
        )
        if not self.law_payloads or not self.joint_calibrations:
            raise ValueError("controller instance requires the unchanged law and joint calibration")
        if self.bound_observation.object_schema != 'empirical-lawhood/runtime/runtime-observation':
            raise ValueError("controller instance requires an exact runtime observation")
        if self.task_functional.object_schema != 'empirical-lawhood/planning/finite-task-functional-spec':
            raise ValueError("controller instance requires its exact late finite task")
        if any(
            c.object_schema
            not in {
                'empirical-lawhood/planning/finite-joint-calibration',
                'empirical-lawhood/planning/finite-policy-path-calibration',
            }
            for c in self.joint_calibrations
        ):
            raise ValueError("controller instance requires exact joint calibration records")


@dataclass(frozen=True, slots=True)
class FrozenFeedbackConsumer(CanonicalRecord):
    """Outcome-blind identity of a qualified feedback procedure, not an efficacy claim."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/frozen-feedback-consumer'
    consumer_id: str
    recipe: ObjectIdentity
    law_payloads: tuple[ArtifactIdentity, ...]
    calibration: ObjectIdentity
    selector: ObjectIdentity
    command_expansion: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.consumer_id, field_name="consumer_id")
        require_sorted_unique_ids(
            self.law_payloads, attribute="artifact_id", field_name="law_payloads"
        )
        if not self.law_payloads:
            raise ValueError("frozen feedback consumer requires its predictive payloads")


@dataclass(frozen=True, slots=True)
class DeliveryControllerStudy(CanonicalRecord):
    "Finite admission authoring through the same controller owner, including valid negatives."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/delivery-controller-study'
    VERSION: ClassVar[str] = '1.0.0'

    study_id: str
    compiler_release_id: str
    system: SystemSpec
    action_bindings: tuple[ControllerActionBinding, ...]
    admission: FiniteCertificateAdmissionSpec
    reachability: FiniteCertificateReachabilitySpec
    synthesis: ActHoldControllerSynthesisPlan
    implementations: tuple[ImplementationBinding, ...]
    measured_hold_fibre: AdmissionMeasuredHoldFibre | None
    delivery_equivalence: DeliveryEquivalenceSpec | StageAwareDeliveryEquivalenceSpec
    allow_fallback_hold: bool
    instance_binding: ControllerInstanceBinding
    prospective_evaluation: ObjectIdentity | None
    compilation_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        _validate_admission_controller_study(self)
        from .finite_response_geometry import FinitePolicyPathCalibration

        for receipt in self.admission.corpus.reachability_receipts:
            calibration = receipt.request.response_set.joint_calibration
            if (
                isinstance(calibration, FinitePolicyPathCalibration)
                and calibration.consumer_policy != self.instance_binding.frozen_recipe
            ):
                raise ValueError("controller changes the qualified chosen-path consumer")
        if type(self.allow_fallback_hold) is not bool:
            raise ValueError("finite programme requires an explicit fallback-HOLD policy")
        if not isinstance(
            self.delivery_equivalence,
            (DeliveryEquivalenceSpec, StageAwareDeliveryEquivalenceSpec),
        ):
            raise ValueError("finite programme requires its own selected-word delivery equivalence")
        cells = {c.candidate_cell_id: c for c in self.admission.candidate_cells}
        candidates = self.synthesis.candidate_chart.candidates
        if {c.admission_candidate_cell.object_id for c in candidates} != set(cells):
            raise ValueError("finite chart must account for every admission cell, including measured HOLD")
        hold_candidates: dict[str, ActHoldActionCandidate] = {}
        for candidate in candidates:
            word = self.admission.corpus.plan.action_fibre(
                cells[candidate.admission_candidate_cell.object_id].action_fibre
            ).action_word
            if isinstance(self.delivery_equivalence, StageAwareDeliveryEquivalenceSpec):
                self.delivery_equivalence.validate_action_word(word)
            requested_null = all(o.requested.value == 0 for o in word.occurrences)
            if (candidate.kind is NativeCandidateKind.HOLD) != requested_null:
                raise ValueError("finite ACT/HOLD kind rewrites the requested native word")
            if candidate.kind is NativeCandidateKind.HOLD:
                if candidate.decision_cell_id in hold_candidates:
                    raise ValueError("finite chart repeats HOLD within one decision cell")
                hold_candidates[candidate.decision_cell_id] = candidate
        if (
            isinstance(self.delivery_equivalence, DeliveryEquivalenceSpec)
            and self.delivery_equivalence.stage_value_tolerance.value != 0
        ):
            raise ValueError(
                "finite nonzero stage tolerances require the stage-aware delivery contract"
            )
        hold = self.measured_hold_fibre
        if hold is not None:
            if hold.delivery_equivalence != self.delivery_equivalence:
                raise ValueError(
                    "finite selected HOLD changes its measured delivery-equivalence contract"
                )
            tolerance = (
                max(t.tolerance.value for t in self.delivery_equivalence.stage_value_tolerances)
                if isinstance(self.delivery_equivalence, StageAwareDeliveryEquivalenceSpec)
                else self.delivery_equivalence.stage_value_tolerance.value
            )
            if tolerance > hold.null_tolerance.value:
                raise ValueError("finite HOLD delivery tolerance exceeds its measured null fibre")
            hold_candidate = hold_candidates.get(hold.decision_cell_id)
            if hold_candidate is None or hold_candidate.admission_candidate_cell != hold.admission_candidate_cell:
                raise ValueError(
                    "measured HOLD must participate as its exact finite HOLD candidate"
                )
            word = self.admission.corpus.plan.action_fibre(
                cells[hold.admission_candidate_cell.object_id].action_fibre
            ).action_word
            if not word.occurrences or any(
                event.native_unit != hold.null_tolerance.unit
                or abs(event.value) > hold.null_tolerance.value
                for occurrence in word.occurrences
                for event in (
                    occurrence.requested,
                    occurrence.accepted,
                    occurrence.applied,
                    occurrence.realized,
                )
            ):
                raise ValueError("measured HOLD must retain its null native delivery fibre")
        instance = self.instance_binding
        payloads = tuple(
            sorted(
                {law.evaluator.payload for law in self.admission.corpus.plan.atlas.laws},
                key=lambda a: a.artifact_id,
            )
        )
        calibrations = tuple(
            sorted(
                {
                    ObjectIdentity.from_record(
                        r.request.response_set.joint_calibration.calibration_id,
                        r.request.response_set.joint_calibration,
                    )
                    for r in self.admission.corpus.reachability_receipts
                },
                key=lambda c: c.object_id,
            )
        )
        if instance.law_payloads != payloads or instance.joint_calibrations != calibrations:
            raise ValueError("finite instance changes the frozen law/calibration payloads")
        for receipt in self.admission.corpus.reachability_receipts:
            request = receipt.request
            source = request.response_set
            if (
                instance.public_handoff != source.public_handoff
                or instance.frame_translation != source.frame_translation
                or instance.task_functional
                != ObjectIdentity.from_record(request.task.functional_id, request.task)
                or instance.information_cutoff != request.information_cutoff
            ):
                raise ValueError("finite instance changes its bound public handoff, task or cutoff")

    def _validate_candidate_utility(self, cell: ReceiptAdmissionAdmissionCandidateCell) -> None:
        _validate_admission_candidate_utility(self, cell)


def decode_delivery_controller_study(payload: bytes) -> DeliveryControllerStudy:
    return decode_canonical_bytes(
        payload,
        DeliveryControllerStudy,
        maximum_bytes=MAX_STREAMING_FINITE_CONTROLLER_PROGRAMME_BYTES,
    )


@dataclass(frozen=True, slots=True)
class RequestedWordMagnitude(CanonicalRecord):
    """Requested scalar pulse magnitude; never a measured response or force sum."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/requested-word-magnitude'

    action_binding_id: str
    action_word: ObjectIdentity
    occurrence_id: str
    magnitude: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.action_binding_id, field_name="action_binding_id")
        validate_stable_id(self.occurrence_id, field_name="occurrence_id")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA or self.magnitude.value < 0:
            raise ValueError("native magnitude requires an exact word and nonnegative value")


@dataclass(frozen=True, slots=True)
class LeastMagnitudeControllerStudy(DeliveryControllerStudy):
    "Finite admission followed by least requested magnitude, then word priority.\n\n    The native-magnitude ordering uses the eligibility of the response utility\n    receipts. Retained utility-ordering records keep their own decoding\n    behavior and defaults.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/least-magnitude-controller-study'
    VERSION: ClassVar[str] = '1.0.0'

    native_magnitudes: tuple[RequestedWordMagnitude, ...]

    def __post_init__(self) -> None:
        DeliveryControllerStudy.__post_init__(self)
        require_sorted_unique_ids(
            self.native_magnitudes, attribute="action_binding_id", field_name="native_magnitudes"
        )
        if tuple(m.action_binding_id for m in self.native_magnitudes) != tuple(
            a.action_binding_id for a in self.action_bindings
        ):
            raise ValueError("native magnitude roster must cover every exact action binding")
        if len({m.magnitude.unit for m in self.native_magnitudes}) != 1:
            raise ValueError("native magnitudes require common units")
        for magnitude, binding in zip(self.native_magnitudes, self.action_bindings, strict=True):
            word = binding.action_word
            if len(word.occurrences) != 1:
                raise ValueError("native magnitude ordering requires one scalar pulse occurrence")
            occurrence = word.occurrences[0]
            if (
                magnitude.action_word != ObjectIdentity.from_record(word.word_id, word)
                or magnitude.occurrence_id != occurrence.occurrence_id
                or magnitude.magnitude.unit != occurrence.requested.native_unit
                or magnitude.magnitude.value != abs(occurrence.requested.value)
            ):
                raise ValueError("native magnitude differs from exact requested action word")


def decode_least_magnitude_controller_study(payload: bytes) -> LeastMagnitudeControllerStudy:
    return decode_canonical_bytes(
        payload, LeastMagnitudeControllerStudy, maximum_bytes=MAX_NATIVE_FINITE_CONTROLLER_PROGRAMME_BYTES
    )


def _validate_admission_controller_study(
    study: AdmissionControllerStudy | DeliveryControllerStudy,
) -> None:
    validate_stable_id(study.study_id, field_name='study_id')
    validate_stable_id(study.compiler_release_id, field_name="compiler_release_id")
    corpus = study.admission.corpus
    plan = corpus.plan
    if study.reachability.admission_spec != study.admission:
        raise ValueError("admission and reachability do not share one raw corpus")
    if (
        plan.atlas.system_id != study.system.system_id
        or plan.atlas.world_id != study.system.world.world_id
        or plan.model_set.target_world_id != study.system.world.world_id
    ):
        raise ValueError("programme system/atlas/model-set world differs")
    for law in plan.atlas.laws:
        validate_law_against_system(law, study.system)
    require_sorted_unique_ids(
        study.action_bindings,
        attribute="action_binding_id",
        field_name="action_bindings",
    )
    action_bindings = {value.action_binding_id: value for value in study.action_bindings}
    admission_action_fibres_by_id = {value.action_binding_id: value for value in plan.action_fibres}
    if set(action_bindings) != set(admission_action_fibres_by_id):
        raise ValueError("action roster differs from the complete admission action fibres")
    relations = {law.relation for law in plan.atlas.laws}
    for action_id, binding in action_bindings.items():
        admission_action_fibre = admission_action_fibres_by_id[action_id]
        if (
            binding.action_word != admission_action_fibre.action_word
            or any(binding.causal_prefix.causal_support.relation != value for value in relations)
            or binding.action_word.receiver_id
            not in plan.atlas.laws[0].relation.receiver_quantity_ids
        ):
            raise ValueError("action binding changes exact law qualification/admission law semantics")
    reachability: ControlledMapReachabilityComparison | FiniteCertificateReachabilityComparison
    if isinstance(study, DeliveryControllerStudy):
        admission = derive_finite_certificate_admission_comparison(study.admission)
        reachability = derive_finite_certificate_reachability_comparison(study.reachability)
    else:
        admission = derive_receipt_admission_comparison(study.admission)
        reachability = derive_controlled_map_reachability_comparison(study.reachability)
    atlas_identity = ObjectIdentity.from_record(plan.atlas.atlas_id, plan.atlas)
    corpus_identity = ObjectIdentity.from_record(corpus.corpus_id, corpus)
    model_set_identity = ObjectIdentity.from_record(
        plan.model_set.model_set_id,
        plan.model_set,
    )
    if (
        study.synthesis.atlas != atlas_identity
        or study.synthesis.admission_corpus != corpus_identity
        or study.synthesis.model_set != model_set_identity
        or study.synthesis.admission_comparison
        != ObjectIdentity.from_record(admission.comparison_id, admission)
        or study.synthesis.reachability_comparison
        != ObjectIdentity.from_record(reachability.comparison_id, reachability)
        or study.synthesis.candidate_chart.model_set != model_set_identity
        or study.synthesis.candidate_chart.nominal_denominator_member_id
        != plan.nominal_denominator_member_id
    ):
        raise ValueError("synthesis rewrites its atlas/admission/model-set geometry")
    admission_candidate_cells_by_id = {value.candidate_cell_id: value for value in study.admission.candidate_cells}
    chart = study.synthesis.candidate_chart
    active_cell_ids: set[str] = set()
    for candidate in chart.candidates:
        cell = admission_candidate_cells_by_id.get(candidate.admission_candidate_cell.object_id)
        if cell is None or candidate.admission_candidate_cell != ObjectIdentity.from_record(
            cell.candidate_cell_id, cell
        ):
            raise ValueError("candidate uses an absent or altered admission cell")
        active_cell_ids.add(cell.candidate_cell_id)
        study._validate_candidate_utility(cell)
    hold = study.measured_hold_fibre
    if hold is not None:
        cell = admission_candidate_cells_by_id.get(hold.admission_candidate_cell.object_id)
        if (
            cell is None
            or hold.admission_candidate_cell != ObjectIdentity.from_record(cell.candidate_cell_id, cell)
            or (
                not isinstance(study, DeliveryControllerStudy)
                and cell.candidate_cell_id in active_cell_ids
            )
        ):
            raise ValueError("measured HOLD must use another exact admission cell")
        hold_action = plan.action_fibre(cell.action_fibre)
        if hold_action.action_binding_id not in action_bindings:
            raise ValueError("measured HOLD action is absent from the programme")
        if isinstance(
            hold.delivery_equivalence,
            StageAwareDeliveryEquivalenceSpec,
        ):
            hold.delivery_equivalence.validate_action_word(hold_action.action_word)
    require_sorted_unique_ids(
        study.implementations,
        attribute="binding_id",
        field_name="implementations",
    )
    roles = {value.role for value in study.implementations}
    expected_roles = set(CONTROLLER_ROLES)
    if study.prospective_evaluation is not None:
        expected_roles.add(ImplementationRole.OUTCOME_EVALUATOR)
    if roles != expected_roles or len(study.implementations) != len(expected_roles):
        raise ValueError("controller implementation role set is incomplete")
    by_id = {value.binding_id: value for value in study.implementations}
    expected_bindings = (
        (study.synthesis.admission_deriver_binding_id, ImplementationRole.ADMISSION_DERIVER),
        (
            study.synthesis.reachability_deriver_binding_id,
            ImplementationRole.REACHABILITY_DERIVER,
        ),
        (study.synthesis.synthesizer_binding_id, ImplementationRole.SYNTHESIZER),
        (study.synthesis.observer_binding_id, ImplementationRole.OBSERVER),
        (
            study.synthesis.online_gate_evaluator_binding_id,
            ImplementationRole.ONLINE_GATE_EVALUATOR,
        ),
        (study.synthesis.delivery_binding_id, ImplementationRole.DELIVERY),
    )
    if any(
        by_id.get(binding_id) is None or by_id[binding_id].role is not role
        for binding_id, role in expected_bindings
    ):
        raise ValueError("controller role binding is missing or has the wrong role")
    if study.prospective_evaluation is not None and not any(
        value.role is ImplementationRole.OUTCOME_EVALUATOR for value in study.implementations
    ):
        raise ValueError("controller use identity lacks an outcome evaluator")
    if study.prospective_evaluation is not None and (
        study.prospective_evaluation.object_schema
        not in (
            {
                'empirical-lawhood/planning/common-start-controller-evaluation-plan',
                'empirical-lawhood/planning/coupled-realization-controller-evaluation-plan',
                'empirical-lawhood/planning/trajectory-controller-evaluation-plan',
            }
            if isinstance(study, DeliveryControllerStudy)
            else {'empirical-lawhood/planning/repeated-delivery-controller-evaluation-plan'}
        )
    ):
        raise ValueError("controller use identity must bind the nested evaluation companion")
    inherited_visibility = VisibilityCeiling.most_restrictive(
        plan.atlas.visibility_ceiling,
        plan.model_set.visibility_ceiling,
        plan.visibility_ceiling,
        study.admission.visibility_ceiling,
        study.reachability.visibility_ceiling,
        study.synthesis.visibility_ceiling,
    )
    if not study.visibility_ceiling.is_at_least_as_restrictive_as(inherited_visibility):
        raise ValueError("controller visibility cannot be lowered")
    if study.compilation_ceiling is not EvidenceCeiling.ADMISSION:
        raise ValueError("controller compilation is capped at admission")
    if study.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
        raise ValueError("controller compilation must remain outcome-blind")
    if not study.visibility_ceiling.is_promotable:
        raise ValueError("outcome-visible work cannot compile a controller")


def _validate_admission_candidate_utility(
    study: AdmissionControllerStudy | DeliveryControllerStudy,
    cell: ReceiptAdmissionAdmissionCandidateCell,
) -> None:
    coordinate_ids = set(cell.planned_coordinate_ids)
    receipts = tuple(
        value
        for value in study.admission.corpus.utility_receipts
        if value.planned_coordinate.coordinate_id in coordinate_ids
    )
    chart = study.synthesis.candidate_chart
    expected_count = (
        sum(
            len(c.qualification_view_ids)
            for c in study.admission.corpus.plan.coordinates
            if c.coordinate_id in coordinate_ids
        )
        if isinstance(study, DeliveryControllerStudy)
        else len(coordinate_ids)
    )
    if len(receipts) != expected_count or any(
        value.utility_definition != chart.utility_definition
        or value.direction is not chart.utility_direction
        or value.minimum_utility != chart.minimum_robust_utility
        for value in receipts
    ):
        raise ValueError("candidate utility differs from its complete admission corpus")


def decode_atlas_controller_study(payload: bytes) -> AtlasControllerStudy:
    return decode_canonical_bytes(
        payload, AtlasControllerStudy, maximum_bytes=MAX_CONTROLLER_PROGRAMME_BYTES
    )


def decode_admission_controller_study(payload: bytes) -> AdmissionControllerStudy:
    return decode_canonical_bytes(
        payload,
        AdmissionControllerStudy,
        maximum_bytes=MAX_CONTROLLER_PROGRAMME_BYTES,
    )


__all__ = [
    'AtlasActionCandidate',
    'AdmissionActionCandidate',
    "CandidateGateReceipt",
    "CandidatePriorityOrientation",
    "CandidateReachabilityReceipt",
    "CandidateUtilityDefinition",
    "CandidateUtilityReceipt",
    "CONTROLLER_ROLES",
    'AtlasControllerStudy',
    'AdmissionControllerStudy',
    'AtlasControllerSynthesisPlan',
    'AdmissionControllerSynthesisPlan',
    "DeliveryEquivalenceSpec",
    'DeliveryStageValueTolerance',
    "EvaluationStratumSpec",
    "EvaluatorBoundarySpec",
    'AtlasActionCandidateChart',
    'AdmissionActionCandidateChart',
    "ImplementationBinding",
    "ImplementationRole",
    'AtlasMeasuredHoldFibre',
    'AdmissionMeasuredHoldFibre',
    "OnlineSupportMonitorSpec",
    'ControllerActionBinding',
    "ProspectiveControllerEvaluationPlan",
    "ReidentificationTriggerSpec",
    'StageAwareDeliveryEquivalenceSpec',
    "UtilityDirection",
    'decode_atlas_controller_study',
    'decode_admission_controller_study',
]

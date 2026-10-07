"""Outcome-blind lookup runtime and partial native-delivery evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import ClassVar, Protocol, cast

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage, ActionStageEvent, OccurrenceActionWord, ObservedActionOccurrence
from empirical_lawhood.kernel.admission import AdmissionGateKind, GateStatus
from empirical_lawhood.kernel.control import OperationalDeliveryState, ScientificCommitmentKind
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.receiver_geometry_control import AmbiguityCertificateDisposition
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    require_unique_ids,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import ClockCoordinate, InformationCutoff
from empirical_lawhood.planning.controller_study import CandidateGateReceipt, AtlasControllerStudy, AdmissionControllerStudy, DeliveryControllerStudy, ControllerInstanceBinding, NativeCandidateKind, ImplementationBinding, ImplementationRole, ControllerActionBinding, StageAwareDeliveryEquivalenceSpec
from empirical_lawhood.planning.evidence_geometry import AtlasGateReceipt, AdmissionCoordinateGateReceipt, GatePredicateSpec, LawMemberEvaluationBinding, ReceiptAdmissionPlannedCoordinate, derive_admission_gate_observation, validate_law_evaluation_binding
from empirical_lawhood.runtime.controller_compiler import CompiledCellAction, CompiledAtlasControllerStudy, CompiledAdmissionControllerStudy, CompiledDeliveryControllerStudy, FINITE_COMPILED_COMMITMENT_SCHEMAS, ActHoldCompiledCellAction, CompiledAdmissionHoldFibre, HoldFibreDisposition


class ObserverDisposition(StrEnum):
    EXACT = "EXACT"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    AMBIGUOUS = "AMBIGUOUS"
    UNEVALUABLE = "UNEVALUABLE"


class CommitmentDisposition(StrEnum):
    ACTION_COMMITTED = "ACTION_COMMITTED"
    MEASURED_HOLD_COMMITTED = "MEASURED_HOLD_COMMITTED"
    NONATTEMPT = "NONATTEMPT"


class TickDisposition(StrEnum):
    ACTION_DELIVERED = "ACTION_DELIVERED"
    MEASURED_HOLD_DELIVERED = "MEASURED_HOLD_DELIVERED"
    NONATTEMPT = "NONATTEMPT"
    DELIVERY_RECOVERY = "DELIVERY_RECOVERY"
    TERMINATED = "TERMINATED"


def _same_clock_domain(left: ClockCoordinate, right: ClockCoordinate) -> bool:
    return (
        left.clock_id == right.clock_id
        and left.time_unit == right.time_unit
        and left.coordinate_frame == right.coordinate_frame
        and left.origin is right.origin
    )


@dataclass(frozen=True, slots=True)
class RuntimeObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/runtime-observation'

    observation_id: str
    independent_unit_id: str
    values: tuple[NamedDecimal, ...]
    coordinate: ClockCoordinate
    input_artifacts: tuple[ArtifactIdentity, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        validate_stable_id(self.independent_unit_id, field_name="independent_unit_id")
        require_sorted_unique_ids(self.values, attribute="value_id", field_name="values")
        require_sorted_unique_ids(
            self.input_artifacts, attribute="artifact_id", field_name="input_artifacts"
        )
        if not self.values or not self.input_artifacts:
            raise ValueError("runtime observation requires values and exact input bytes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("runtime observation must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class ObserverEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/observer-evaluation'

    evaluation_id: str
    observation: ObjectIdentity
    observer: ImplementationBinding
    disposition: ObserverDisposition
    resolved_decision_cell_id: str | None
    candidate_cell_ids: tuple[str, ...]
    receiver_candidate_pair_ids: tuple[str, ...]
    receiver_certificate: ObjectIdentity | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        if self.observer.role is not ImplementationRole.OBSERVER:
            raise ValueError("observer evaluation uses another implementation role")
        if self.resolved_decision_cell_id is not None:
            validate_stable_id(
                self.resolved_decision_cell_id, field_name="resolved_decision_cell_id"
            )
        require_sorted_unique_strings(self.candidate_cell_ids, field_name="candidate_cell_ids")
        require_sorted_unique_strings(
            self.receiver_candidate_pair_ids, field_name="receiver_candidate_pair_ids"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is ObserverDisposition.EXACT:
            if (
                self.resolved_decision_cell_id is None
                or self.candidate_cell_ids != (self.resolved_decision_cell_id,)
                or self.reason_codes
            ):
                raise ValueError("exact observer requires one resolved decision cell")
        elif self.disposition is ObserverDisposition.AMBIGUOUS:
            if len(self.receiver_candidate_pair_ids) < 2 or not self.reason_codes:
                raise ValueError("ambiguous observer requires receiver candidates and reasons")
        elif (
            self.resolved_decision_cell_id is not None
            or self.candidate_cell_ids
            or not self.reason_codes
        ):
            raise ValueError("failed observer cannot resolve a decision cell")
        if (self.receiver_certificate is None) != (not self.receiver_candidate_pair_ids):
            raise ValueError("receiver candidate roster and certificate must appear together")


class ObserverPort(Protocol):
    implementation_binding: ImplementationBinding

    def observe(self, observation: RuntimeObservation) -> ObserverEvaluation: ...


@dataclass(frozen=True, slots=True)
class AtlasLiveGateReceipt(CanonicalRecord):
    """One live gate derivation bound to the exact runtime observation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/atlas-live-gate-receipt'

    candidate_id: str
    action_binding_id: str
    observation: ObjectIdentity
    receipt: AtlasGateReceipt

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_stable_id(self.action_binding_id, field_name="action_binding_id")


@dataclass(frozen=True, slots=True)
class AtlasLiveGateEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/atlas-live-gate-evaluation'

    evaluation_id: str
    observation: RuntimeObservation
    decision_cell_id: str
    admission_cell_id: str
    candidate_id: str
    action_binding_id: str
    model_member_ids: tuple[str, ...]
    evaluator: ImplementationBinding
    receipts: tuple[AtlasLiveGateReceipt, ...]

    def __post_init__(self) -> None:
        for name, identifier in (
            ("evaluation_id", self.evaluation_id),
            ("decision_cell_id", self.decision_cell_id),
            ("admission_cell_id", self.admission_cell_id),
            ("candidate_id", self.candidate_id),
            ("action_binding_id", self.action_binding_id),
        ):
            validate_stable_id(identifier, field_name=name)
        if self.evaluator.role is not ImplementationRole.ONLINE_GATE_EVALUATOR:
            raise ValueError("online gate evaluation uses another implementation role")
        require_sorted_unique_strings(
            self.model_member_ids, field_name="model_member_ids", allow_empty=False
        )
        receipt_ids = tuple(value.receipt.receipt_id for value in self.receipts)
        if receipt_ids != tuple(sorted(set(receipt_ids))):
            raise ValueError("live gate receipts must have sorted unique receipt IDs")
        expected = {
            (member, kind) for member in self.model_member_ids for kind in AdmissionGateKind
        }
        observed = {
            (value.receipt.model_member_id, value.receipt.predicate.gate_kind)
            for value in self.receipts
        }
        if observed != expected or len(self.receipts) != len(expected):
            raise ValueError("online gate grid is incomplete or contains extras")
        observation_identity = ObjectIdentity.from_record(
            self.observation.observation_id, self.observation
        )
        observation_artifacts = {value.artifact_id for value in self.observation.input_artifacts}
        for value in self.receipts:
            receipt = value.receipt
            if (
                value.candidate_id != self.candidate_id
                or value.action_binding_id != self.action_binding_id
                or value.observation != observation_identity
                or receipt.cell_id != self.admission_cell_id
                or receipt.model_member_id not in self.model_member_ids
                or receipt.numerical_view_id != receipt.model_member_id
                or receipt.evaluator != self.evaluator.reference
                or receipt.predicate.evaluator != self.evaluator.reference
                or receipt.information_cutoff.clock_id != self.observation.coordinate.clock_id
                or receipt.information_cutoff.coordinate != self.observation.coordinate.coordinate
                or not observation_artifacts
                <= {artifact.artifact_id for artifact in receipt.input_artifacts}
            ):
                raise ValueError("online gate receipt is not derived from its live observation")

    @property
    def passed(self) -> bool:
        return all(value.receipt.status is GateStatus.PASS for value in self.receipts)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    f"ONLINE_{value.receipt.predicate.gate_kind.value}_{value.receipt.status.value}"
                    for value in self.receipts
                    if value.receipt.status is not GateStatus.PASS
                }
            )
        )


class AtlasLiveGateEvaluatorPort(Protocol):
    implementation_binding: ImplementationBinding

    def evaluate(
        self,
        *,
        study: AtlasControllerStudy,
        observation: RuntimeObservation,
        decision_cell_id: str,
        admission_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        model_member_ids: tuple[str, ...],
        offline_gate_receipts: tuple[CandidateGateReceipt, ...],
    ) -> AtlasLiveGateEvaluation: ...


@dataclass(frozen=True, slots=True)
class AdmissionLiveGateReceipt(CanonicalRecord):
    "Fresh pre-action gate retaining the exact offline member-local admission constituent."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/admission-live-gate-receipt'

    receipt_id: str
    candidate_id: str
    action_binding_id: str
    observation: ObjectIdentity
    offline_gate_receipt: ObjectIdentity
    planned_coordinate: ReceiptAdmissionPlannedCoordinate
    law_evaluation_binding: LawMemberEvaluationBinding
    predicate: GatePredicateSpec
    observed_scalar: NamedDecimal | None
    observed_boolean: bool | None
    observed_identity: ObjectIdentity | None
    information_cutoff: InformationCutoff
    evaluator: ImplementationBinding
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    status: GateStatus
    margin: NamedDecimal | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("receipt_id", self.receipt_id),
            ("candidate_id", self.candidate_id),
            ("action_binding_id", self.action_binding_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.observation.object_schema != RuntimeObservation.SCHEMA:
            raise ValueError("live admission gate requires an exact runtime observation")
        if self.offline_gate_receipt.object_schema != AdmissionCoordinateGateReceipt.SCHEMA:
            raise ValueError("live admission gate requires an exact offline admission receipt")
        if self.evaluator.role is not ImplementationRole.ONLINE_GATE_EVALUATOR:
            raise ValueError("live admission gate uses another implementation role")
        if self.predicate.evaluator != self.evaluator.reference:
            raise ValueError("live admission gate predicate uses another evaluator")
        if self.law_evaluation_binding.planned_coordinate != ObjectIdentity.from_record(
            self.planned_coordinate.coordinate_id,
            self.planned_coordinate,
        ):
            raise ValueError("live admission gate law binding uses another admission coordinate")
        validate_law_evaluation_binding(
            self.planned_coordinate,
            self.law_evaluation_binding,
        )
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
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        supplied = sum(
            value is not None
            for value in (
                self.observed_scalar,
                self.observed_boolean,
                self.observed_identity,
            )
        )
        expected = (
            (GateStatus.UNEVALUABLE, None, ("GATE_OBSERVATION_UNAVAILABLE",))
            if supplied == 0
            else derive_admission_gate_observation(
                receipt_id=self.receipt_id,
                predicate=self.predicate,
                observed_scalar=self.observed_scalar,
                observed_boolean=self.observed_boolean,
                observed_identity=self.observed_identity,
            )
        )
        if (self.status, self.margin, self.reason_codes) != expected:
            raise ValueError("live admission gate status is not mechanically observation-derived")


@dataclass(frozen=True, slots=True)
class AdmissionLiveGateEvaluation(CanonicalRecord):
    """Complete live gate intersection over every member/version coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/admission-live-gate-evaluation'
    INCLUDE_NUMERICAL_VIEWS: ClassVar[bool] = False

    evaluation_id: str
    observation: RuntimeObservation
    decision_cell_id: str
    admission_candidate_cell_id: str
    candidate_id: str
    action_binding_id: str
    planned_coordinate_ids: tuple[str, ...]
    evaluator: ImplementationBinding
    receipts: tuple[AdmissionLiveGateReceipt, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluation_id", self.evaluation_id),
            ("decision_cell_id", self.decision_cell_id),
            ("admission_candidate_cell_id", self.admission_candidate_cell_id),
            ("candidate_id", self.candidate_id),
            ("action_binding_id", self.action_binding_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.planned_coordinate_ids,
            field_name="planned_coordinate_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.receipts,
            attribute="receipt_id",
            field_name="receipts",
        )
        if self.INCLUDE_NUMERICAL_VIEWS:
            coordinates: dict[str, ReceiptAdmissionPlannedCoordinate] = {}
            for receipt in self.receipts:
                coordinate = receipt.planned_coordinate
                if coordinates.setdefault(coordinate.coordinate_id, coordinate) != coordinate:
                    raise ValueError("online finite gate changes a planned coordinate across views")
            if set(coordinates) != set(self.planned_coordinate_ids):
                raise ValueError("online finite gate omits or adds planned coordinates")
            expected_views = {
                (c.coordinate_id, kind, view)
                for c in coordinates.values()
                for kind in AdmissionGateKind
                for view in c.qualification_view_ids
            }
            actual_views = {
                (
                    r.planned_coordinate.coordinate_id,
                    r.predicate.gate_kind,
                    r.law_evaluation_binding.qualification_view_id,
                )
                for r in self.receipts
            }
            if len(self.receipts) != len(expected_views) or actual_views != expected_views:
                raise ValueError("online finite gate grid omits or adds numerical views")
        else:
            expected = {
                (coordinate_id, kind)
                for coordinate_id in self.planned_coordinate_ids
                for kind in AdmissionGateKind
            }
            observed = {
                (
                    value.planned_coordinate.coordinate_id,
                    value.predicate.gate_kind,
                )
                for value in self.receipts
            }
            if len(self.receipts) != len(expected) or observed != expected:
                raise ValueError("online admission gate grid omits or adds admission constituents")
        observation_identity = ObjectIdentity.from_record(
            self.observation.observation_id,
            self.observation,
        )
        for receipt in self.receipts:
            if (
                receipt.candidate_id != self.candidate_id
                or receipt.action_binding_id != self.action_binding_id
                or receipt.observation != observation_identity
                or receipt.evaluator != self.evaluator
            ):
                raise ValueError("online admission gate receipt rewrites its runtime request")

    @property
    def passed(self) -> bool:
        return all(value.status is GateStatus.PASS for value in self.receipts)

    @property
    def reason_codes(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    f"ONLINE_{value.predicate.gate_kind.value}_{value.status.value}"
                    for value in self.receipts
                    if value.status is not GateStatus.PASS
                }
            )
        )


@dataclass(frozen=True, slots=True)
class NumericalViewLiveGateEvaluation(AdmissionLiveGateEvaluation):
    """Same native live operands, intersected over every required numerical view."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/numerical-view-live-gate-evaluation'
    INCLUDE_NUMERICAL_VIEWS: ClassVar[bool] = True


class NumericalViewLiveGateEvaluatorPort(Protocol):
    implementation_binding: ImplementationBinding

    def evaluate_numerical_view_live_gate(
        self,
        *,
        study: DeliveryControllerStudy,
        observation: RuntimeObservation,
        decision_cell_id: str,
        admission_candidate_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        planned_coordinate_ids: tuple[str, ...],
        offline_gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...],
    ) -> NumericalViewLiveGateEvaluation: ...


class AdmissionLiveGateEvaluatorPort(Protocol):
    implementation_binding: ImplementationBinding

    def evaluate_admission_live_gate(
        self,
        *,
        study: AdmissionControllerStudy,
        observation: RuntimeObservation,
        decision_cell_id: str,
        admission_candidate_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        planned_coordinate_ids: tuple[str, ...],
        offline_gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...],
    ) -> AdmissionLiveGateEvaluation: ...


@dataclass(frozen=True, slots=True)
class AtlasControllerDecisionCommitment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/atlas-controller-decision-commitment'

    commitment_id: str
    compiled_study: ObjectIdentity
    observation: RuntimeObservation
    observer_evaluation: ObserverEvaluation
    commitment_coordinate: ClockCoordinate
    disposition: CommitmentDisposition
    selected_cell_action: CompiledCellAction | None
    action_binding: ControllerActionBinding | None
    active_gate_evaluation: AtlasLiveGateEvaluation | None
    hold_gate_evaluation: AtlasLiveGateEvaluation | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.commitment_id, field_name="commitment_id")
        if self.observer_evaluation.observation != ObjectIdentity.from_record(
            self.observation.observation_id, self.observation
        ):
            raise ValueError("commitment observer binds another observation")
        if not _same_clock_domain(self.observation.coordinate, self.commitment_coordinate):
            raise ValueError("observation and commitment require an explicit clock transport")
        if self.commitment_coordinate.coordinate < self.observation.coordinate.coordinate:
            raise ValueError("commitment precedes its observation")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is CommitmentDisposition.ACTION_COMMITTED:
            if (
                self.selected_cell_action is None
                or self.action_binding != self.selected_cell_action.action_binding
                or self.active_gate_evaluation is None
                or not self.active_gate_evaluation.passed
                or self.hold_gate_evaluation is not None
                or self.reason_codes
            ):
                raise ValueError("action commitment is not an exact live-gated lookup")
        elif self.disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED:
            if (
                self.action_binding is None
                or self.hold_gate_evaluation is None
                or not self.hold_gate_evaluation.passed
                or not self.reason_codes
            ):
                raise ValueError("measured HOLD commitment lacks exact live evidence")
        elif (
            self.action_binding is not None
            or self.selected_cell_action is not None
            or not self.reason_codes
        ):
            raise ValueError("NONATTEMPT cannot contain a scientific action")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("controller commitment must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class DeliveryPortResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/delivery-port-result'

    result_id: str
    observed_occurrences: tuple[ObservedActionOccurrence, ...]
    failure_state: OperationalDeliveryState | None

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_unique_ids(
            self.observed_occurrences,
            attribute="expected_occurrence_id",
            field_name="observed_occurrences",
        )
        if self.failure_state not in {
            None,
            OperationalDeliveryState.DELIVERY_RECOVERY,
            OperationalDeliveryState.TERMINATED,
        }:
            raise ValueError("delivery port returned an invalid failure state")


class NativeDeliveryPort(Protocol):
    implementation_binding: ImplementationBinding

    def deliver(
        self,
        *,
        action_word: OccurrenceActionWord,
        commitment_kind: ScientificCommitmentKind,
    ) -> DeliveryPortResult: ...


def _event_matches_expected(observed: ActionStageEvent | None, expected: ActionStageEvent) -> bool:
    return observed == expected


def _delivery_reason_codes(
    expected: OccurrenceActionWord,
    observed: tuple[ObservedActionOccurrence, ...],
) -> tuple[str, ...]:
    by_occurrence = {value.expected_occurrence_id: value for value in observed}
    reasons: set[str] = set()
    expected_ids = tuple(value.occurrence_id for value in expected.occurrences)
    if set(by_occurrence) - set(expected_ids):
        reasons.add("DELIVERY_UNEXPECTED_OCCURRENCE")
    for occurrence in expected.occurrences:
        actual = by_occurrence.get(occurrence.occurrence_id)
        if actual is None:
            reasons.add("DELIVERY_OCCURRENCE_MISSING")
            continue
        # A native port can observe a semantically invalid delivery even when
        # all four numeric stage values happen to remain within tolerance.
        # Clipping and rejection are the important examples: neither may be
        # erased merely because a simulator reports the requested value in its
        # accepted/applied/realized fields.  Keep the substrate reason text on
        # the occurrence, while deriving one closed generic failure code here.
        if actual.reason_codes:
            reasons.add("DELIVERY_OCCURRENCE_REPORTED_FAILURE")
        events = (actual.requested, actual.accepted, actual.applied, actual.realized)
        expected_events = (
            occurrence.requested,
            occurrence.accepted,
            occurrence.applied,
            occurrence.realized,
        )
        if any(value is None for value in events):
            reasons.add("DELIVERY_STAGE_MISSING")
        if any(
            value is not None and not _event_matches_expected(value, expected_value)
            for value, expected_value in zip(events, expected_events, strict=True)
        ):
            reasons.add("DELIVERY_VALUE_OR_CLOCK_MISMATCH")
    if tuple(value.expected_occurrence_id for value in observed) != tuple(
        value for value in expected_ids if value in by_occurrence
    ):
        reasons.add("DELIVERY_OCCURRENCE_ORDER_MISMATCH")
    return tuple(sorted(reasons))


def _stage_event_metadata_matches(
    observed: ActionStageEvent,
    expected: ActionStageEvent,
    *,
    native_unit: str,
) -> bool:
    """Compare one stage without allowing its numeric value to compensate."""

    return (
        observed.stage is expected.stage
        and observed.quantity_id == expected.quantity_id
        and observed.native_unit == expected.native_unit == native_unit
        and observed.native_action_frame == expected.native_action_frame
        and observed.native_direction == expected.native_direction
        and observed.coordinate == expected.coordinate
    )


def _stage_aware_delivery_reason_codes(
    expected: OccurrenceActionWord,
    observed: tuple[ObservedActionOccurrence, ...],
    equivalence: StageAwareDeliveryEquivalenceSpec,
) -> tuple[str, ...]:
    """Derive closed, per-occurrence delivery failures without pooling errors."""

    equivalence.validate_action_word(expected)
    by_occurrence = {value.expected_occurrence_id: value for value in observed}
    reasons: set[str] = set()
    expected_ids = tuple(value.occurrence_id for value in expected.occurrences)
    if set(by_occurrence) - set(expected_ids):
        reasons.add("DELIVERY_UNEXPECTED_OCCURRENCE")
    for occurrence in expected.occurrences:
        actual = by_occurrence.get(occurrence.occurrence_id)
        if actual is None:
            reasons.add("DELIVERY_OCCURRENCE_MISSING")
            continue
        if actual.reason_codes:
            reasons.add("DELIVERY_OCCURRENCE_REPORTED_FAILURE")
        events = (actual.requested, actual.accepted, actual.applied, actual.realized)
        expected_events = (
            occurrence.requested,
            occurrence.accepted,
            occurrence.applied,
            occurrence.realized,
        )
        if any(value is None for value in events):
            reasons.add("DELIVERY_STAGE_MISSING")
        for stage, value, expected_value in zip(
            ActionDeliveryStage,
            events,
            expected_events,
            strict=True,
        ):
            if value is None:
                continue
            if not _stage_event_metadata_matches(
                value,
                expected_value,
                native_unit=equivalence.native_unit,
            ):
                reasons.add("DELIVERY_STAGE_METADATA_MISMATCH")
                continue
            if abs(value.value - expected_value.value) > equivalence.tolerance_for(stage).value:
                reasons.add("DELIVERY_STAGE_VALUE_OUTSIDE_TOLERANCE")
    if tuple(value.expected_occurrence_id for value in observed) != tuple(
        value for value in expected_ids if value in by_occurrence
    ):
        reasons.add("DELIVERY_OCCURRENCE_ORDER_MISMATCH")
    return tuple(sorted(reasons))


@dataclass(frozen=True, slots=True)
class ExactActionDeliveryTrace(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/exact-action-delivery-trace'

    trace_id: str
    commitment: ObjectIdentity
    delivery: ImplementationBinding
    commitment_kind: ScientificCommitmentKind
    expected_action_word: OccurrenceActionWord | None
    observed_occurrences: tuple[ObservedActionOccurrence, ...]
    operational_state: OperationalDeliveryState
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.trace_id, field_name="trace_id")
        if self.delivery.role is not ImplementationRole.DELIVERY:
            raise ValueError("delivery trace uses another implementation role")
        require_unique_ids(
            self.observed_occurrences,
            attribute="expected_occurrence_id",
            field_name="observed_occurrences",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.commitment_kind is ScientificCommitmentKind.NONATTEMPT:
            if (
                self.expected_action_word is not None
                or self.observed_occurrences
                or self.operational_state is not OperationalDeliveryState.NOT_ATTEMPTED
                or not self.reason_codes
            ):
                raise ValueError("NONATTEMPT delivery trace fabricates action evidence")
            return
        if self.expected_action_word is None:
            raise ValueError("scientific action delivery requires an expected word")
        expected_reasons = _delivery_reason_codes(
            self.expected_action_word, self.observed_occurrences
        )
        exact = not expected_reasons
        if exact:
            if (
                self.operational_state is not OperationalDeliveryState.DELIVERED
                or self.reason_codes
            ):
                raise ValueError("exact delivery has an inconsistent operational state")
        else:
            if (
                self.operational_state
                not in {
                    OperationalDeliveryState.DELIVERY_RECOVERY,
                    OperationalDeliveryState.TERMINATED,
                }
                or self.reason_codes != expected_reasons
            ):
                raise ValueError("failed delivery state/reasons are not mechanically derived")

    @property
    def exact(self) -> bool:
        return self.expected_action_word is not None and not _delivery_reason_codes(
            self.expected_action_word, self.observed_occurrences
        )


@dataclass(frozen=True, slots=True)
class StageAwareActionDeliveryTrace(CanonicalRecord):
    """Stage-aware delivery evidence retaining every exact observed event.

    Numeric equivalence is evaluated independently for each occurrence and
    delivery stage.  Metadata and clocks remain exact, and errors can never be
    summed or averaged across stages or occurrences.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/stage-aware-action-delivery-trace'

    trace_id: str
    commitment: ObjectIdentity
    delivery: ImplementationBinding
    commitment_kind: ScientificCommitmentKind
    expected_action_word: OccurrenceActionWord | None
    delivery_equivalence: StageAwareDeliveryEquivalenceSpec | None
    observed_occurrences: tuple[ObservedActionOccurrence, ...]
    operational_state: OperationalDeliveryState
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.trace_id, field_name="trace_id")
        if self.delivery.role is not ImplementationRole.DELIVERY:
            raise ValueError("stage-aware action delivery trace uses another implementation role")
        require_unique_ids(
            self.observed_occurrences,
            attribute="expected_occurrence_id",
            field_name="observed_occurrences",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.commitment_kind is ScientificCommitmentKind.NONATTEMPT:
            if (
                self.expected_action_word is not None
                or self.delivery_equivalence is not None
                or self.observed_occurrences
                or self.operational_state is not OperationalDeliveryState.NOT_ATTEMPTED
                or not self.reason_codes
            ):
                raise ValueError("stage-aware NONATTEMPT delivery trace fabricates action evidence")
            return
        if self.expected_action_word is None or not isinstance(
            self.delivery_equivalence,
            StageAwareDeliveryEquivalenceSpec,
        ):
            raise ValueError("stage-aware scientific delivery requires a word and stage equivalence")
        expected_reasons = _stage_aware_delivery_reason_codes(
            self.expected_action_word,
            self.observed_occurrences,
            self.delivery_equivalence,
        )
        equivalent = not expected_reasons
        if equivalent:
            if (
                self.operational_state is not OperationalDeliveryState.DELIVERED
                or self.reason_codes
            ):
                raise ValueError("equivalent stage-aware delivery has an inconsistent operational state")
        elif (
            self.operational_state
            not in {
                OperationalDeliveryState.DELIVERY_RECOVERY,
                OperationalDeliveryState.TERMINATED,
            }
            or self.reason_codes != expected_reasons
        ):
            raise ValueError("failed stage-aware delivery state/reasons are not mechanically derived")

    @property
    def exact(self) -> bool:
        """Whether every observed event is byte-semantically exact, without tolerance."""

        return self.expected_action_word is not None and not _delivery_reason_codes(
            self.expected_action_word,
            self.observed_occurrences,
        )

    @property
    def equivalent(self) -> bool:
        return (
            self.expected_action_word is not None
            and self.delivery_equivalence is not None
            and not _stage_aware_delivery_reason_codes(
                self.expected_action_word,
                self.observed_occurrences,
                self.delivery_equivalence,
            )
        )


@dataclass(frozen=True, slots=True)
class AtlasControllerTickReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/atlas-controller-tick-receipt'

    tick_id: str
    compiled_study: ObjectIdentity
    commitment: AtlasControllerDecisionCommitment
    delivery_trace: ExactActionDeliveryTrace
    disposition: TickDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.tick_id, field_name="tick_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not isinstance(self.delivery_trace, ExactActionDeliveryTrace):
            raise ValueError("atlas controller tick requires an exact action delivery trace")
        if self.delivery_trace.commitment != ObjectIdentity.from_record(
            self.commitment.commitment_id, self.commitment
        ):
            raise ValueError("tick delivery trace binds another commitment")
        if self.commitment.compiled_study != self.compiled_study:
            raise ValueError("tick commitment binds another compiled programme")
        expected = (
            TickDisposition.NONATTEMPT
            if self.commitment.disposition is CommitmentDisposition.NONATTEMPT
            else (
                TickDisposition.DELIVERY_RECOVERY
                if self.delivery_trace.operational_state
                is OperationalDeliveryState.DELIVERY_RECOVERY
                else (
                    TickDisposition.TERMINATED
                    if self.delivery_trace.operational_state is OperationalDeliveryState.TERMINATED
                    else (
                        TickDisposition.ACTION_DELIVERED
                        if self.commitment.disposition is CommitmentDisposition.ACTION_COMMITTED
                        else TickDisposition.MEASURED_HOLD_DELIVERED
                    )
                )
            )
        )
        if self.disposition is not expected:
            raise ValueError("tick disposition differs from commitment/delivery")
        expected_reasons = tuple(
            sorted({*self.commitment.reason_codes, *self.delivery_trace.reason_codes})
        )
        if self.reason_codes != expected_reasons:
            raise ValueError("tick reasons are not derived")


@dataclass(frozen=True, slots=True)
class AdmissionControllerDecisionCommitment(CanonicalRecord):
    """Additive commitment retaining the three-axis live-gate evaluation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/admission-controller-decision-commitment'

    commitment_id: str
    compiled_study: ObjectIdentity
    observation: RuntimeObservation
    observer_evaluation: ObserverEvaluation
    commitment_coordinate: ClockCoordinate
    disposition: CommitmentDisposition
    selected_cell_action: CompiledCellAction | None
    action_binding: ControllerActionBinding | None
    active_gate_evaluation: AdmissionLiveGateEvaluation | None
    hold_gate_evaluation: AdmissionLiveGateEvaluation | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.commitment_id, field_name="commitment_id")
        if self.compiled_study.object_schema != CompiledAdmissionControllerStudy.SCHEMA:
            raise ValueError("admission controller commitment requires a compiled admission controller study")
        if self.observer_evaluation.observation != ObjectIdentity.from_record(
            self.observation.observation_id,
            self.observation,
        ):
            raise ValueError("admission controller commitment observer binds another observation")
        if not _same_clock_domain(self.observation.coordinate, self.commitment_coordinate):
            raise ValueError("observation and commitment require an explicit clock transport")
        if self.commitment_coordinate.coordinate < self.observation.coordinate.coordinate:
            raise ValueError("commitment precedes its observation")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is CommitmentDisposition.ACTION_COMMITTED:
            if (
                self.selected_cell_action is None
                or self.action_binding != self.selected_cell_action.action_binding
                or self.active_gate_evaluation is None
                or not self.active_gate_evaluation.passed
                or self.active_gate_evaluation.decision_cell_id
                != self.selected_cell_action.decision_cell_id
                or self.active_gate_evaluation.admission_candidate_cell_id
                != self.selected_cell_action.admission_cell_id
                or self.active_gate_evaluation.candidate_id
                != self.selected_cell_action.candidate_id
                or self.active_gate_evaluation.action_binding_id
                != self.action_binding.action_binding_id
                or self.hold_gate_evaluation is not None
                or self.reason_codes
            ):
                raise ValueError("admission controller action commitment is not an exact live-gated lookup")
        elif self.disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED:
            if (
                self.action_binding is None
                or self.hold_gate_evaluation is None
                or not self.hold_gate_evaluation.passed
                or self.hold_gate_evaluation.action_binding_id
                != self.action_binding.action_binding_id
                or self.hold_gate_evaluation.decision_cell_id
                != self.observer_evaluation.resolved_decision_cell_id
                or self.selected_cell_action is not None
                or not self.reason_codes
            ):
                raise ValueError("admission controller measured HOLD commitment lacks exact live evidence")
        elif (
            self.action_binding is not None
            or self.selected_cell_action is not None
            or not self.reason_codes
        ):
            raise ValueError("admission controller NONATTEMPT cannot contain a scientific action")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("admission controller commitment must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class AdmissionControllerTickReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/admission-controller-tick-receipt'

    tick_id: str
    compiled_study: ObjectIdentity
    commitment: AdmissionControllerDecisionCommitment
    delivery_trace: ExactActionDeliveryTrace
    disposition: TickDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.tick_id, field_name="tick_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not isinstance(self.delivery_trace, ExactActionDeliveryTrace):
            raise ValueError("admission controller tick requires an exact action delivery trace")
        if self.delivery_trace.commitment != ObjectIdentity.from_record(
            self.commitment.commitment_id,
            self.commitment,
        ):
            raise ValueError("admission controller tick delivery trace binds another commitment")
        if self.commitment.compiled_study != self.compiled_study:
            raise ValueError("admission controller tick commitment binds another compiled programme")
        expected = (
            TickDisposition.NONATTEMPT
            if self.commitment.disposition is CommitmentDisposition.NONATTEMPT
            else (
                TickDisposition.DELIVERY_RECOVERY
                if self.delivery_trace.operational_state
                is OperationalDeliveryState.DELIVERY_RECOVERY
                else (
                    TickDisposition.TERMINATED
                    if self.delivery_trace.operational_state is OperationalDeliveryState.TERMINATED
                    else (
                        TickDisposition.ACTION_DELIVERED
                        if self.commitment.disposition is CommitmentDisposition.ACTION_COMMITTED
                        else TickDisposition.MEASURED_HOLD_DELIVERED
                    )
                )
            )
        )
        if self.disposition is not expected:
            raise ValueError("admission controller tick disposition differs from commitment/delivery")
        expected_reasons = tuple(
            sorted({*self.commitment.reason_codes, *self.delivery_trace.reason_codes})
        )
        if self.reason_codes != expected_reasons:
            raise ValueError("admission controller tick reasons are not derived")


@dataclass(frozen=True, slots=True)
class StageAwareAdmissionControllerTickReceipt(CanonicalRecord):
    "Compiled admission controller tick whose delivery is judged by an exact stage-aware spec."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/stage-aware-admission-controller-tick-receipt'

    tick_id: str
    compiled_study: ObjectIdentity
    commitment: AdmissionControllerDecisionCommitment
    delivery_trace: StageAwareActionDeliveryTrace
    disposition: TickDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.tick_id, field_name="tick_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not isinstance(self.delivery_trace, StageAwareActionDeliveryTrace):
            raise ValueError("stage-aware admission controller tick requires the stage-aware action delivery trace")
        if self.delivery_trace.commitment != ObjectIdentity.from_record(
            self.commitment.commitment_id,
            self.commitment,
        ):
            raise ValueError("stage-aware admission controller tick delivery trace binds another commitment")
        if self.commitment.compiled_study != self.compiled_study:
            raise ValueError("stage-aware admission controller tick commitment binds another compiled programme")
        expected_kind = (
            ScientificCommitmentKind.ACTION
            if self.commitment.disposition is CommitmentDisposition.ACTION_COMMITTED
            else ScientificCommitmentKind.MEASURED_HOLD
            if self.commitment.disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED
            else ScientificCommitmentKind.NONATTEMPT
        )
        binding = self.commitment.action_binding
        if (
            self.delivery_trace.commitment_kind is not expected_kind
            or (binding is None and self.delivery_trace.expected_action_word is not None)
            or (
                binding is not None
                and self.delivery_trace.expected_action_word != binding.action_word
            )
        ):
            raise ValueError("stage-aware admission controller tick delivery kind/word differs from its commitment")
        expected = (
            TickDisposition.NONATTEMPT
            if self.commitment.disposition is CommitmentDisposition.NONATTEMPT
            else (
                TickDisposition.DELIVERY_RECOVERY
                if self.delivery_trace.operational_state
                is OperationalDeliveryState.DELIVERY_RECOVERY
                else (
                    TickDisposition.TERMINATED
                    if self.delivery_trace.operational_state is OperationalDeliveryState.TERMINATED
                    else (
                        TickDisposition.ACTION_DELIVERED
                        if self.commitment.disposition is CommitmentDisposition.ACTION_COMMITTED
                        else TickDisposition.MEASURED_HOLD_DELIVERED
                    )
                )
            )
        )
        if self.disposition is not expected:
            raise ValueError("stage-aware admission controller tick disposition differs from commitment/delivery")
        expected_reasons = tuple(
            sorted({*self.commitment.reason_codes, *self.delivery_trace.reason_codes})
        )
        if self.reason_codes != expected_reasons:
            raise ValueError("stage-aware admission controller tick reasons are not derived")


@dataclass(frozen=True, slots=True)
class DeliveryControllerDecisionCommitment(CanonicalRecord):
    """Finite ACT/HOLD selection and fallback remain distinct immutable decisions."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/delivery-controller-decision-commitment'

    commitment_id: str
    compiled_study: ObjectIdentity
    instance_binding: ControllerInstanceBinding
    observation: RuntimeObservation
    observer_evaluation: ObserverEvaluation
    commitment_coordinate: ClockCoordinate
    disposition: CommitmentDisposition
    selected_cell_action: ActHoldCompiledCellAction | None
    action_binding: ControllerActionBinding | None
    selected_gate_evaluation: NumericalViewLiveGateEvaluation | None
    fallback_hold_gate_evaluation: NumericalViewLiveGateEvaluation | None
    measured_hold_qualification: ObjectIdentity | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.commitment_id, field_name="commitment_id")
        if self.compiled_study.object_schema not in FINITE_COMPILED_COMMITMENT_SCHEMAS:
            raise ValueError("finite commitment requires a compatible finite compiled programme")
        observation = ObjectIdentity.from_record(self.observation.observation_id, self.observation)
        if self.observer_evaluation.observation != observation:
            raise ValueError("finite commitment observer uses another observation")
        if not _same_clock_domain(self.observation.coordinate, self.commitment_coordinate):
            raise ValueError("finite commitment changes the native clock domain")
        if self.commitment_coordinate.coordinate < self.observation.coordinate.coordinate:
            raise ValueError("finite commitment precedes its observation")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("finite commitment cannot consume task outcomes")
        if self.disposition is CommitmentDisposition.NONATTEMPT:
            if (
                self.action_binding is not None
                or self.selected_cell_action is not None
                or not self.reason_codes
            ):
                raise ValueError("finite NONATTEMPT cannot contain a scientific action")
            if self.measured_hold_qualification is not None:
                raise ValueError("finite NONATTEMPT cannot claim a delivered HOLD")
            return
        if (
            self.instance_binding.bound_observation != observation
            or self.observer_evaluation.disposition is not ObserverDisposition.EXACT
        ):
            raise ValueError("finite commitment changes the handoff-bound observation")
        selection = self.selected_cell_action
        evaluation = (
            self.selected_gate_evaluation
            if selection is not None
            else self.fallback_hold_gate_evaluation
        )
        binding = self.action_binding
        if (
            binding is None
            or evaluation is None
            or not evaluation.passed
            or evaluation.observation != self.observation
            or evaluation.action_binding_id != binding.action_binding_id
            or evaluation.decision_cell_id != self.observer_evaluation.resolved_decision_cell_id
        ):
            raise ValueError("finite commitment lacks exact successful live gate evidence")
        if selection is not None:
            expected_disposition = (
                CommitmentDisposition.ACTION_COMMITTED
                if selection.kind is NativeCandidateKind.ACT
                else CommitmentDisposition.MEASURED_HOLD_COMMITTED
            )
            if (
                self.disposition is not expected_disposition
                or selection.action_binding != binding
                or evaluation.candidate_id != selection.candidate_id
                or evaluation.admission_candidate_cell_id != selection.admission_cell_id
                or self.fallback_hold_gate_evaluation is not None
                or self.reason_codes
            ):
                raise ValueError("finite commitment rewrites the selected ACT/HOLD lookup")
        elif (
            self.disposition is not CommitmentDisposition.MEASURED_HOLD_COMMITTED
            or not self.reason_codes
        ):
            raise ValueError("finite fallback requires a reasoned measured HOLD")
        if self.disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED:
            if (
                self.measured_hold_qualification is None
                or self.measured_hold_qualification.object_schema != CompiledAdmissionHoldFibre.SCHEMA
            ):
                raise ValueError(
                    "finite HOLD commitment lacks its exact measured-fibre qualification"
                )
        elif self.measured_hold_qualification is not None:
            raise ValueError("finite ACT cannot masquerade as a measured HOLD")


@dataclass(frozen=True, slots=True)
class DeliveryControllerTickReceipt(CanonicalRecord):
    """Finite commitment with either exact or explicitly stage-tolerant delivery."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/delivery-controller-tick-receipt'

    tick_id: str
    compiled_study: ObjectIdentity
    commitment: DeliveryControllerDecisionCommitment
    delivery_trace: ExactActionDeliveryTrace | StageAwareActionDeliveryTrace
    disposition: TickDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.tick_id, field_name="tick_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.delivery_trace.commitment
            != ObjectIdentity.from_record(self.commitment.commitment_id, self.commitment)
            or self.compiled_study != self.commitment.compiled_study
        ):
            raise ValueError("finite tick breaks its compiled/commitment/delivery identity chain")
        expected_kind = (
            ScientificCommitmentKind.ACTION
            if self.commitment.disposition is CommitmentDisposition.ACTION_COMMITTED
            else ScientificCommitmentKind.MEASURED_HOLD
            if self.commitment.disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED
            else ScientificCommitmentKind.NONATTEMPT
        )
        expected_word = (
            self.commitment.action_binding.action_word
            if self.commitment.action_binding is not None
            else None
        )
        if (
            self.delivery_trace.commitment_kind is not expected_kind
            or self.delivery_trace.expected_action_word != expected_word
        ):
            raise ValueError("finite tick changes ACT/HOLD kind or its exact native word")
        expected = (
            TickDisposition.NONATTEMPT
            if expected_kind is ScientificCommitmentKind.NONATTEMPT
            else TickDisposition.DELIVERY_RECOVERY
            if self.delivery_trace.operational_state is OperationalDeliveryState.DELIVERY_RECOVERY
            else TickDisposition.TERMINATED
            if self.delivery_trace.operational_state is OperationalDeliveryState.TERMINATED
            else TickDisposition.ACTION_DELIVERED
            if expected_kind is ScientificCommitmentKind.ACTION
            else TickDisposition.MEASURED_HOLD_DELIVERED
        )
        if self.disposition is not expected:
            raise ValueError("finite tick status is not derived from native delivery")
        if self.reason_codes != tuple(
            sorted({*self.commitment.reason_codes, *self.delivery_trace.reason_codes})
        ):
            raise ValueError("finite tick reasons are not mechanically derived")


class ControllerRuntime:
    """Internal frozen lookup plus live invalidation; never an online optimizer."""

    def __init__(
        self,
        *,
        compiled: CompiledAtlasControllerStudy
        | CompiledAdmissionControllerStudy
        | CompiledDeliveryControllerStudy,
        observer: ObserverPort,
        online_gates: AtlasLiveGateEvaluatorPort
        | AdmissionLiveGateEvaluatorPort
        | NumericalViewLiveGateEvaluatorPort,
        delivery: NativeDeliveryPort,
    ) -> None:
        self._compiled = compiled
        self._observer = observer
        self._online_gates = online_gates
        self._delivery = delivery
        expected = {
            ImplementationRole.OBSERVER: observer.implementation_binding,
            ImplementationRole.ONLINE_GATE_EVALUATOR: online_gates.implementation_binding,
            ImplementationRole.DELIVERY: delivery.implementation_binding,
        }
        if any(compiled.implementation(role) != binding for role, binding in expected.items()):
            raise ValueError("runtime service implementation differs from compiled programme")

    def tick(
        self,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> AtlasControllerTickReceipt | AdmissionControllerTickReceipt | DeliveryControllerTickReceipt:
        if isinstance(self._compiled, CompiledDeliveryControllerStudy):
            commitment = self.prepare_commitment(
                observation, commitment_coordinate=commitment_coordinate
            )
            return self.deliver_prepared_commitment(commitment)
        if isinstance(self._compiled, CompiledAdmissionControllerStudy):
            result = self._tick_admission_controller(
                observation,
                commitment_coordinate=commitment_coordinate,
            )
            if isinstance(result, StageAwareAdmissionControllerTickReceipt):  # pragma: no cover - invariant
                raise AssertionError("admission tick emitted a stage-aware receipt")
            return result
        return self._tick_atlas_controller(observation, commitment_coordinate=commitment_coordinate)

    def tick_stage_aware(
        self,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> AdmissionControllerTickReceipt | StageAwareAdmissionControllerTickReceipt | DeliveryControllerTickReceipt:
        "Tick the compiled admission controller with its exact compiled stage equivalence.\n\n        ACTION and measured HOLD both emit DeliveryControllerTickReceipt and retain their exact committed\n        action-word identity while applying the noncompensating per-stage\n        numeric bounds embedded in the compiled measured-HOLD fibre.\n        NONATTEMPT retains the outcome-free stage-aware admission route.\n        "

        if isinstance(self._compiled, CompiledDeliveryControllerStudy):
            if not isinstance(
                self._compiled.study.delivery_equivalence, StageAwareDeliveryEquivalenceSpec
            ):
                raise ValueError(
                    "finite programme lacks its declared stage-aware delivery equivalence"
                )
            return self.deliver_prepared_commitment(
                self.prepare_commitment(
                    observation,
                    commitment_coordinate=commitment_coordinate,
                )
            )
        if not isinstance(self._compiled, CompiledAdmissionControllerStudy):
            raise ValueError("stage-aware delivery requires a compiled admission controller study")
        measured_hold = self._compiled.study.measured_hold_fibre
        if measured_hold is None or not isinstance(
            measured_hold.delivery_equivalence,
            StageAwareDeliveryEquivalenceSpec,
        ):
            raise ValueError("compiled programme lacks stage-aware measured-HOLD delivery")
        return self._tick_admission_controller(
            observation,
            commitment_coordinate=commitment_coordinate,
            stage_aware_delivery=True,
        )

    def prepare_commitment(
        self,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> DeliveryControllerDecisionCommitment:
        """Pure finite lookup and live validation; a caller may durably bind this before delivery."""

        compiled = self._compiled
        if not isinstance(compiled, CompiledDeliveryControllerStudy):
            raise ValueError("staged finite commitment requires a compiled delivery controller study")
        programme = compiled.study
        observer = self._observer.observe(observation)
        observation_identity = ObjectIdentity.from_record(observation.observation_id, observation)
        if (
            observer.observation != observation_identity
            or observer.observer != compiled.implementation(ImplementationRole.OBSERVER)
        ):
            raise ValueError("finite observer output changes its exact input or implementation")
        if not _same_clock_domain(observation.coordinate, commitment_coordinate):
            raise ValueError("finite commitment requires an explicit clock transport")
        latency = commitment_coordinate.coordinate - observation.coordinate.coordinate
        if latency < 0:
            raise ValueError("finite commitment precedes its observation")
        reasons: set[str] = set()
        actionable = observer.disposition is ObserverDisposition.EXACT
        if not actionable:
            reasons.add(f"OBSERVER_{observer.disposition.value}")
            reasons.update(observer.reason_codes)
        if observation_identity != programme.instance_binding.bound_observation:
            actionable = False
            reasons.add("FINITE_INSTANCE_OBSERVATION_CHANGED")
        if latency > programme.synthesis.deadline_seconds:
            actionable = False
            reasons.add("OBSERVATION_ACTION_DEADLINE_MISSED")
        selection: ActHoldCompiledCellAction | None = None
        selected_evaluation: NumericalViewLiveGateEvaluation | None = None
        fallback_evaluation: NumericalViewLiveGateEvaluation | None = None
        action_binding: ControllerActionBinding | None = None
        disposition = CommitmentDisposition.NONATTEMPT
        cell_id = observer.resolved_decision_cell_id
        if actionable and cell_id is not None:
            choice = compiled.action_for_cell(cell_id)
            if choice is not None:
                selected_evaluation = self._evaluate_finite_choice(
                    programme,
                    observation,
                    cell_id=cell_id,
                    candidate_id=choice.candidate_id,
                    admission_candidate_cell_id=choice.admission_cell_id,
                    action_binding_id=choice.action_binding.action_binding_id,
                )
                if selected_evaluation.passed:
                    selection = choice
                    action_binding = choice.action_binding
                    disposition = (
                        CommitmentDisposition.ACTION_COMMITTED
                        if choice.kind is NativeCandidateKind.ACT
                        else CommitmentDisposition.MEASURED_HOLD_COMMITTED
                    )
                else:
                    reasons.update(selected_evaluation.reason_codes)
            else:
                reasons.add("NO_COMPILED_FINITE_CHOICE_FOR_CELL")
            if action_binding is None and programme.allow_fallback_hold:
                hold = compiled.hold_fibre
                if (
                    hold is not None
                    and hold.disposition is HoldFibreDisposition.QUALIFIED
                    and hold.source.decision_cell_id == cell_id
                ):
                    fallback_evaluation = self._evaluate_finite_choice(
                        programme,
                        observation,
                        cell_id=cell_id,
                        candidate_id=hold.source.hold_fibre_id,
                        admission_candidate_cell_id=hold.source.admission_candidate_cell.object_id,
                        action_binding_id=hold.action_binding.action_binding_id,
                    )
                    if fallback_evaluation.passed:
                        action_binding = hold.action_binding
                        disposition = CommitmentDisposition.MEASURED_HOLD_COMMITTED
                        reasons.add("FALLBACK_MEASURED_HOLD_SELECTED")
                    else:
                        reasons.update(fallback_evaluation.reason_codes)
                else:
                    reasons.add("MEASURED_HOLD_FIBRE_UNAVAILABLE_FOR_CELL")
        hold_identity = None
        if disposition is CommitmentDisposition.MEASURED_HOLD_COMMITTED:
            qualified_hold = compiled.hold_fibre
            if (
                qualified_hold is None
                or qualified_hold.disposition is not HoldFibreDisposition.QUALIFIED
            ):
                raise ValueError(
                    "finite selected HOLD lacks its independently qualified measured fibre"
                )
            hold_identity = ObjectIdentity.from_record(
                qualified_hold.qualification_id, qualified_hold
            )
        return DeliveryControllerDecisionCommitment(
            commitment_id=f"commitment.{observation.observation_id}",
            compiled_study=ObjectIdentity.from_record(compiled.compiled_study_id, compiled),
            instance_binding=programme.instance_binding,
            observation=observation,
            observer_evaluation=observer,
            commitment_coordinate=commitment_coordinate,
            disposition=disposition,
            selected_cell_action=selection,
            action_binding=action_binding,
            selected_gate_evaluation=selected_evaluation,
            fallback_hold_gate_evaluation=fallback_evaluation,
            measured_hold_qualification=hold_identity,
            reason_codes=tuple(sorted(reasons)),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )

    def _evaluate_finite_choice(
        self,
        study: DeliveryControllerStudy,
        observation: RuntimeObservation,
        *,
        cell_id: str,
        candidate_id: str,
        admission_candidate_cell_id: str,
        action_binding_id: str,
    ) -> NumericalViewLiveGateEvaluation:
        cell = next(
            c
            for c in study.admission.candidate_cells
            if c.candidate_cell_id == admission_candidate_cell_id
        )
        offline = tuple(
            g
            for g in study.admission.corpus.gate_receipts
            if g.planned_coordinate.coordinate_id in cell.planned_coordinate_ids
        )
        evaluator = cast(NumericalViewLiveGateEvaluatorPort, self._online_gates)
        evaluation = evaluator.evaluate_numerical_view_live_gate(
            study=study,
            observation=observation,
            decision_cell_id=cell_id,
            admission_candidate_cell_id=admission_candidate_cell_id,
            candidate_id=candidate_id,
            action_binding_id=action_binding_id,
            planned_coordinate_ids=cell.planned_coordinate_ids,
            offline_gate_receipts=offline,
        )
        if not isinstance(evaluation, NumericalViewLiveGateEvaluation):
            raise ValueError("finite live gate requires the complete numerical-view schema")
        self._validate_live_admission_evaluation(
            evaluation=evaluation,
            observation=observation,
            decision_cell_id=cell_id,
            admission_candidate_cell_id=admission_candidate_cell_id,
            candidate_id=candidate_id,
            action_binding_id=action_binding_id,
            planned_coordinate_ids=cell.planned_coordinate_ids,
            offline_gate_receipts=offline,
            evaluator_binding=evaluator.implementation_binding,
        )
        return evaluation

    def deliver_prepared_commitment(
        self, commitment: DeliveryControllerDecisionCommitment
    ) -> DeliveryControllerTickReceipt:
        """Replay the frozen decision before invoking the existing native delivery port."""

        compiled = self._compiled
        if not isinstance(compiled, CompiledDeliveryControllerStudy):
            raise ValueError("finite commitment delivery requires a compiled delivery controller study")
        if commitment != self.prepare_commitment(
            commitment.observation, commitment_coordinate=commitment.commitment_coordinate
        ):
            raise ValueError("finite commitment changed between preparation and native delivery")
        equivalence = compiled.study.delivery_equivalence
        trace: ExactActionDeliveryTrace | StageAwareActionDeliveryTrace
        if (
            isinstance(equivalence, StageAwareDeliveryEquivalenceSpec)
            and commitment.disposition is not CommitmentDisposition.NONATTEMPT
        ):
            trace = self._deliver_stage_aware(commitment, equivalence=equivalence)
        else:
            trace = self._deliver(commitment)
        disposition = (
            TickDisposition.NONATTEMPT
            if commitment.disposition is CommitmentDisposition.NONATTEMPT
            else TickDisposition.DELIVERY_RECOVERY
            if trace.operational_state is OperationalDeliveryState.DELIVERY_RECOVERY
            else TickDisposition.TERMINATED
            if trace.operational_state is OperationalDeliveryState.TERMINATED
            else TickDisposition.ACTION_DELIVERED
            if commitment.disposition is CommitmentDisposition.ACTION_COMMITTED
            else TickDisposition.MEASURED_HOLD_DELIVERED
        )
        return DeliveryControllerTickReceipt(
            tick_id=f"tick.{commitment.observation.observation_id}",
            compiled_study=commitment.compiled_study,
            commitment=commitment,
            delivery_trace=trace,
            disposition=disposition,
            reason_codes=tuple(sorted({*commitment.reason_codes, *trace.reason_codes})),
        )

    def _tick_atlas_controller(
        self,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> AtlasControllerTickReceipt:
        compiled = cast(CompiledAtlasControllerStudy, self._compiled)
        programme = compiled.study
        observer_evaluation = self._observer.observe(observation)
        if observer_evaluation.observation != ObjectIdentity.from_record(
            observation.observation_id, observation
        ) or observer_evaluation.observer != compiled.implementation(ImplementationRole.OBSERVER):
            raise ValueError("observer output differs from exact runtime input/binding")

        reasons: set[str] = set()
        active_evaluation: AtlasLiveGateEvaluation | None = None
        hold_evaluation: AtlasLiveGateEvaluation | None = None
        selection: CompiledCellAction | None = None
        action_binding: ControllerActionBinding | None = None
        disposition = CommitmentDisposition.NONATTEMPT
        cell_id = observer_evaluation.resolved_decision_cell_id
        actionable_observation = observer_evaluation.disposition is ObserverDisposition.EXACT
        hold_only_observation = False
        certificate = compiled.ambiguity_certificate
        if (
            observer_evaluation.disposition is ObserverDisposition.AMBIGUOUS
            and certificate is not None
        ):
            certificate_matches = (
                observer_evaluation.receiver_certificate
                == ObjectIdentity.from_record(certificate.certificate_id, certificate)
                and cell_id is not None
            )
            actionable_observation = certificate_matches and (
                certificate.disposition is AmbiguityCertificateDisposition.SHARED_ROBUST_ACTION
            )
            hold_only_observation = certificate_matches and (
                certificate.disposition is AmbiguityCertificateDisposition.SHARED_QUALIFIED_HOLD
            )
        if not actionable_observation and not hold_only_observation:
            reasons.add(f"OBSERVER_{observer_evaluation.disposition.value}")
            reasons.update(observer_evaluation.reason_codes)
        if not _same_clock_domain(observation.coordinate, commitment_coordinate):
            raise ValueError("runtime commitment requires an explicit clock transport")
        latency = commitment_coordinate.coordinate - observation.coordinate.coordinate
        if latency < 0:
            raise ValueError("runtime commitment precedes observation")
        deadline_missed = latency > programme.synthesis.deadline_seconds
        if deadline_missed:
            reasons.add("OBSERVATION_ACTION_DEADLINE_MISSED")

        if (
            (actionable_observation or hold_only_observation)
            and not deadline_missed
            and cell_id is not None
        ):
            selection = compiled.action_for_cell(cell_id) if actionable_observation else None
            if selection is not None and actionable_observation:
                candidate = next(
                    value
                    for value in programme.synthesis.candidate_chart.candidates
                    if value.candidate_id == selection.candidate_id
                )
                active_evaluation = self._evaluate_atlas_live_gates(
                    study=programme,
                    observation=observation,
                    decision_cell_id=cell_id,
                    admission_cell_id=candidate.admission_cell_id,
                    candidate_id=candidate.candidate_id,
                    action_binding_id=candidate.action_binding_id,
                    model_member_ids=candidate.model_member_ids,
                    offline_gate_receipts=candidate.gate_receipts,
                )
                if active_evaluation.passed:
                    disposition = CommitmentDisposition.ACTION_COMMITTED
                    action_binding = selection.action_binding
                    reasons.clear()
                else:
                    reasons.update(active_evaluation.reason_codes)
            elif actionable_observation:
                reasons.add("NO_COMPILED_ACTION_FOR_CELL")
            else:
                reasons.add("RECEIVER_SHARED_QUALIFIED_HOLD")

            if disposition is not CommitmentDisposition.ACTION_COMMITTED:
                hold = compiled.hold_fibre
                if (
                    hold is not None
                    and hold.disposition is HoldFibreDisposition.QUALIFIED
                    and hold.source.decision_cell_id == cell_id
                ):
                    source = hold.source
                    hold_evaluation = self._evaluate_atlas_live_gates(
                        study=programme,
                        observation=observation,
                        decision_cell_id=cell_id,
                        admission_cell_id=source.admission_cell_id,
                        candidate_id=source.hold_fibre_id,
                        action_binding_id=source.action_binding_id,
                        model_member_ids=source.model_member_ids,
                        offline_gate_receipts=source.gate_receipts,
                    )
                    if hold_evaluation.passed:
                        disposition = CommitmentDisposition.MEASURED_HOLD_COMMITTED
                        action_binding = hold.action_binding
                        reasons.add("ACTIVE_ACTION_UNAVAILABLE_MEASURED_HOLD_SELECTED")
                    else:
                        reasons.update(hold_evaluation.reason_codes)
                else:
                    reasons.add(
                        "MEASURED_HOLD_FIBRE_MISSING"
                        if hold is None
                        else "MEASURED_HOLD_FIBRE_UNAVAILABLE_FOR_CELL"
                    )

        commitment = AtlasControllerDecisionCommitment(
            commitment_id=f"commitment.{observation.observation_id}",
            compiled_study=ObjectIdentity.from_record(compiled.compiled_study_id, compiled),
            observation=observation,
            observer_evaluation=observer_evaluation,
            commitment_coordinate=commitment_coordinate,
            disposition=disposition,
            selected_cell_action=(
                selection if disposition is CommitmentDisposition.ACTION_COMMITTED else None
            ),
            action_binding=action_binding,
            active_gate_evaluation=active_evaluation,
            hold_gate_evaluation=hold_evaluation,
            reason_codes=tuple(sorted(reasons)),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        trace = self._deliver_atlas_commitment(commitment)
        tick_disposition = (
            TickDisposition.NONATTEMPT
            if commitment.disposition is CommitmentDisposition.NONATTEMPT
            else (
                TickDisposition.DELIVERY_RECOVERY
                if trace.operational_state is OperationalDeliveryState.DELIVERY_RECOVERY
                else (
                    TickDisposition.TERMINATED
                    if trace.operational_state is OperationalDeliveryState.TERMINATED
                    else (
                        TickDisposition.ACTION_DELIVERED
                        if commitment.disposition is CommitmentDisposition.ACTION_COMMITTED
                        else TickDisposition.MEASURED_HOLD_DELIVERED
                    )
                )
            )
        )
        tick_reasons = tuple(sorted({*commitment.reason_codes, *trace.reason_codes}))
        return AtlasControllerTickReceipt(
            tick_id=f"tick.{observation.observation_id}",
            compiled_study=commitment.compiled_study,
            commitment=commitment,
            delivery_trace=trace,
            disposition=tick_disposition,
            reason_codes=tick_reasons,
        )

    def _tick_admission_controller(
        self,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
        stage_aware_delivery: bool = False,
    ) -> AdmissionControllerTickReceipt | StageAwareAdmissionControllerTickReceipt:
        compiled = cast(CompiledAdmissionControllerStudy, self._compiled)
        programme = compiled.study
        observer_evaluation = self._observer.observe(observation)
        if observer_evaluation.observation != ObjectIdentity.from_record(
            observation.observation_id,
            observation,
        ) or observer_evaluation.observer != compiled.implementation(ImplementationRole.OBSERVER):
            raise ValueError("observer output differs from exact runtime input/binding")
        reasons: set[str] = set()
        active_evaluation: AdmissionLiveGateEvaluation | None = None
        hold_evaluation: AdmissionLiveGateEvaluation | None = None
        selection: CompiledCellAction | None = None
        action_binding: ControllerActionBinding | None = None
        disposition = CommitmentDisposition.NONATTEMPT
        cell_id = observer_evaluation.resolved_decision_cell_id
        actionable = observer_evaluation.disposition is ObserverDisposition.EXACT
        if not actionable:
            reasons.add(f"OBSERVER_{observer_evaluation.disposition.value}")
            reasons.update(observer_evaluation.reason_codes)
        if not _same_clock_domain(observation.coordinate, commitment_coordinate):
            raise ValueError("runtime commitment requires an explicit clock transport")
        latency = commitment_coordinate.coordinate - observation.coordinate.coordinate
        if latency < 0:
            raise ValueError("runtime commitment precedes observation")
        deadline_missed = latency > programme.synthesis.deadline_seconds
        if deadline_missed:
            reasons.add("OBSERVATION_ACTION_DEADLINE_MISSED")
        if actionable and not deadline_missed and cell_id is not None:
            selection = compiled.action_for_cell(cell_id)
            if selection is not None:
                candidate = next(
                    value
                    for value in programme.synthesis.candidate_chart.candidates
                    if value.candidate_id == selection.candidate_id
                )
                admission_candidate_cell = next(
                    value
                    for value in programme.admission.candidate_cells
                    if ObjectIdentity.from_record(value.candidate_cell_id, value)
                    == candidate.admission_candidate_cell
                )
                offline = tuple(
                    value
                    for value in programme.admission.corpus.gate_receipts
                    if value.planned_coordinate.coordinate_id in admission_candidate_cell.planned_coordinate_ids
                )
                active_evaluation = self._evaluate_admission_live_gates(
                    study=programme,
                    observation=observation,
                    decision_cell_id=cell_id,
                    admission_candidate_cell_id=admission_candidate_cell.candidate_cell_id,
                    candidate_id=candidate.candidate_id,
                    action_binding_id=selection.action_binding.action_binding_id,
                    planned_coordinate_ids=admission_candidate_cell.planned_coordinate_ids,
                    offline_gate_receipts=offline,
                )
                if active_evaluation.passed:
                    disposition = CommitmentDisposition.ACTION_COMMITTED
                    action_binding = selection.action_binding
                    reasons.clear()
                else:
                    reasons.update(active_evaluation.reason_codes)
            else:
                reasons.add("NO_COMPILED_ACTION_FOR_CELL")
            if disposition is not CommitmentDisposition.ACTION_COMMITTED:
                hold = compiled.hold_fibre
                if (
                    hold is not None
                    and hold.disposition is HoldFibreDisposition.QUALIFIED
                    and hold.source.decision_cell_id == cell_id
                ):
                    admission_candidate_cell = next(
                        value
                        for value in programme.admission.candidate_cells
                        if ObjectIdentity.from_record(value.candidate_cell_id, value)
                        == hold.source.admission_candidate_cell
                    )
                    offline = tuple(
                        value
                        for value in programme.admission.corpus.gate_receipts
                        if value.planned_coordinate.coordinate_id in admission_candidate_cell.planned_coordinate_ids
                    )
                    hold_evaluation = self._evaluate_admission_live_gates(
                        study=programme,
                        observation=observation,
                        decision_cell_id=cell_id,
                        admission_candidate_cell_id=admission_candidate_cell.candidate_cell_id,
                        candidate_id=hold.source.hold_fibre_id,
                        action_binding_id=hold.action_binding.action_binding_id,
                        planned_coordinate_ids=admission_candidate_cell.planned_coordinate_ids,
                        offline_gate_receipts=offline,
                    )
                    if hold_evaluation.passed:
                        disposition = CommitmentDisposition.MEASURED_HOLD_COMMITTED
                        action_binding = hold.action_binding
                        reasons.add("ACTIVE_ACTION_UNAVAILABLE_MEASURED_HOLD_SELECTED")
                    else:
                        reasons.update(hold_evaluation.reason_codes)
                else:
                    reasons.add(
                        "MEASURED_HOLD_FIBRE_MISSING"
                        if hold is None
                        else "MEASURED_HOLD_FIBRE_UNAVAILABLE_FOR_CELL"
                    )
        commitment = AdmissionControllerDecisionCommitment(
            commitment_id=f"commitment.{observation.observation_id}",
            compiled_study=ObjectIdentity.from_record(
                compiled.compiled_study_id,
                compiled,
            ),
            observation=observation,
            observer_evaluation=observer_evaluation,
            commitment_coordinate=commitment_coordinate,
            disposition=disposition,
            selected_cell_action=(
                selection if disposition is CommitmentDisposition.ACTION_COMMITTED else None
            ),
            action_binding=action_binding,
            active_gate_evaluation=active_evaluation,
            hold_gate_evaluation=hold_evaluation,
            reason_codes=tuple(sorted(reasons)),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        if stage_aware_delivery and disposition in {
            CommitmentDisposition.ACTION_COMMITTED,
            CommitmentDisposition.MEASURED_HOLD_COMMITTED,
        }:
            measured_hold = compiled.study.measured_hold_fibre
            if measured_hold is None or not isinstance(
                measured_hold.delivery_equivalence,
                StageAwareDeliveryEquivalenceSpec,
            ):
                raise AssertionError("stage-aware tick lost its measured-HOLD equivalence")
            stage_trace = self._deliver_stage_aware(
                commitment,
                equivalence=measured_hold.delivery_equivalence,
            )
            stage_tick_disposition = (
                TickDisposition.DELIVERY_RECOVERY
                if stage_trace.operational_state is OperationalDeliveryState.DELIVERY_RECOVERY
                else (
                    TickDisposition.TERMINATED
                    if stage_trace.operational_state is OperationalDeliveryState.TERMINATED
                    else (
                        TickDisposition.ACTION_DELIVERED
                        if disposition is CommitmentDisposition.ACTION_COMMITTED
                        else TickDisposition.MEASURED_HOLD_DELIVERED
                    )
                )
            )
            return StageAwareAdmissionControllerTickReceipt(
                tick_id=f"tick.{observation.observation_id}",
                compiled_study=commitment.compiled_study,
                commitment=commitment,
                delivery_trace=stage_trace,
                disposition=stage_tick_disposition,
                reason_codes=tuple(sorted({*commitment.reason_codes, *stage_trace.reason_codes})),
            )
        trace = self._deliver_admission_commitment(commitment)
        tick_disposition = (
            TickDisposition.NONATTEMPT
            if disposition is CommitmentDisposition.NONATTEMPT
            else (
                TickDisposition.DELIVERY_RECOVERY
                if trace.operational_state is OperationalDeliveryState.DELIVERY_RECOVERY
                else (
                    TickDisposition.TERMINATED
                    if trace.operational_state is OperationalDeliveryState.TERMINATED
                    else (
                        TickDisposition.ACTION_DELIVERED
                        if disposition is CommitmentDisposition.ACTION_COMMITTED
                        else TickDisposition.MEASURED_HOLD_DELIVERED
                    )
                )
            )
        )
        return AdmissionControllerTickReceipt(
            tick_id=f"tick.{observation.observation_id}",
            compiled_study=commitment.compiled_study,
            commitment=commitment,
            delivery_trace=trace,
            disposition=tick_disposition,
            reason_codes=tuple(sorted({*commitment.reason_codes, *trace.reason_codes})),
        )

    def _evaluate_admission_live_gates(
        self,
        *,
        study: AdmissionControllerStudy,
        observation: RuntimeObservation,
        decision_cell_id: str,
        admission_candidate_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        planned_coordinate_ids: tuple[str, ...],
        offline_gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...],
    ) -> AdmissionLiveGateEvaluation:
        evaluator = cast(AdmissionLiveGateEvaluatorPort, self._online_gates)
        evaluation = evaluator.evaluate_admission_live_gate(
            study=study,
            observation=observation,
            decision_cell_id=decision_cell_id,
            admission_candidate_cell_id=admission_candidate_cell_id,
            candidate_id=candidate_id,
            action_binding_id=action_binding_id,
            planned_coordinate_ids=planned_coordinate_ids,
            offline_gate_receipts=offline_gate_receipts,
        )
        self._validate_live_admission_evaluation(
            evaluation=evaluation,
            observation=observation,
            decision_cell_id=decision_cell_id,
            admission_candidate_cell_id=admission_candidate_cell_id,
            candidate_id=candidate_id,
            action_binding_id=action_binding_id,
            planned_coordinate_ids=planned_coordinate_ids,
            offline_gate_receipts=offline_gate_receipts,
            evaluator_binding=evaluator.implementation_binding,
        )
        return evaluation

    @staticmethod
    def _validate_live_admission_evaluation(
        *,
        evaluation: AdmissionLiveGateEvaluation,
        observation: RuntimeObservation,
        decision_cell_id: str,
        admission_candidate_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        planned_coordinate_ids: tuple[str, ...],
        offline_gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...],
        evaluator_binding: ImplementationBinding,
    ) -> None:
        if (
            evaluation.observation != observation
            or evaluation.decision_cell_id != decision_cell_id
            or evaluation.admission_candidate_cell_id != admission_candidate_cell_id
            or evaluation.candidate_id != candidate_id
            or evaluation.action_binding_id != action_binding_id
            or evaluation.planned_coordinate_ids != planned_coordinate_ids
            or evaluation.evaluator != evaluator_binding
        ):
            raise ValueError("online admission gate output differs from exact runtime request")
        expected_receipts = {value.receipt_id: value for value in offline_gate_receipts}
        observed_receipts = {
            value.offline_gate_receipt.object_id: value for value in evaluation.receipts
        }
        if set(observed_receipts) != set(expected_receipts):
            raise ValueError("online admission gate output omits or adds offline admission receipts")
        for receipt_id, offline in expected_receipts.items():
            live = observed_receipts[receipt_id]
            if (
                live.offline_gate_receipt != ObjectIdentity.from_record(offline.receipt_id, offline)
                or live.planned_coordinate != offline.planned_coordinate
                or live.law_evaluation_binding != offline.law_evaluation_binding
                or replace(
                    live.predicate,
                    predicate_id=offline.predicate.predicate_id,
                    evaluator=offline.predicate.evaluator,
                )
                != offline.predicate
            ):
                raise ValueError("online admission gate output substitutes offline admission evidence")

    def _evaluate_atlas_live_gates(
        self,
        *,
        study: AtlasControllerStudy,
        observation: RuntimeObservation,
        decision_cell_id: str,
        admission_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        model_member_ids: tuple[str, ...],
        offline_gate_receipts: tuple[CandidateGateReceipt, ...],
    ) -> AtlasLiveGateEvaluation:
        evaluation = cast(AtlasLiveGateEvaluatorPort, self._online_gates).evaluate(
            study=study,
            observation=observation,
            decision_cell_id=decision_cell_id,
            admission_cell_id=admission_cell_id,
            candidate_id=candidate_id,
            action_binding_id=action_binding_id,
            model_member_ids=model_member_ids,
            offline_gate_receipts=offline_gate_receipts,
        )
        if (
            evaluation.observation != observation
            or evaluation.decision_cell_id != decision_cell_id
            or evaluation.admission_cell_id != admission_cell_id
            or evaluation.candidate_id != candidate_id
            or evaluation.action_binding_id != action_binding_id
            or evaluation.model_member_ids != model_member_ids
            or evaluation.evaluator != self._online_gates.implementation_binding
        ):
            raise ValueError("online gate output differs from exact runtime request")
        return evaluation

    def _deliver_atlas_commitment(
        self,
        commitment: AtlasControllerDecisionCommitment,
    ) -> ExactActionDeliveryTrace:
        return self._deliver(commitment)

    def _deliver_admission_commitment(
        self,
        commitment: AdmissionControllerDecisionCommitment,
    ) -> ExactActionDeliveryTrace:
        return self._deliver(commitment)

    def _deliver_stage_aware(
        self,
        commitment: AdmissionControllerDecisionCommitment | DeliveryControllerDecisionCommitment,
        *,
        equivalence: StageAwareDeliveryEquivalenceSpec,
    ) -> StageAwareActionDeliveryTrace:
        if commitment.disposition not in {
            CommitmentDisposition.ACTION_COMMITTED,
            CommitmentDisposition.MEASURED_HOLD_COMMITTED,
        }:
            raise ValueError("stage-aware delivery requires an action or HOLD commitment")
        binding = commitment.action_binding
        if binding is None:
            raise AssertionError("stage-aware commitment lost its action binding")
        kind = (
            ScientificCommitmentKind.ACTION
            if commitment.disposition is CommitmentDisposition.ACTION_COMMITTED
            else ScientificCommitmentKind.MEASURED_HOLD
        )
        delivery_binding = self._delivery.implementation_binding
        port_result = self._delivery.deliver(
            action_word=binding.action_word,
            commitment_kind=kind,
        )
        reasons = _stage_aware_delivery_reason_codes(
            binding.action_word,
            port_result.observed_occurrences,
            equivalence,
        )
        if reasons:
            if port_result.failure_state not in {
                OperationalDeliveryState.DELIVERY_RECOVERY,
                OperationalDeliveryState.TERMINATED,
            }:
                raise ValueError(
                    "failed stage-aware delivery lacks explicit recovery/termination state"
                )
            state = port_result.failure_state
        else:
            if port_result.failure_state is not None:
                raise ValueError(
                    "equivalent stage-aware delivery cannot claim recovery/termination"
                )
            state = OperationalDeliveryState.DELIVERED
        return StageAwareActionDeliveryTrace(
            trace_id=f"delivery.{commitment.commitment_id}",
            commitment=ObjectIdentity.from_record(commitment.commitment_id, commitment),
            delivery=delivery_binding,
            commitment_kind=kind,
            expected_action_word=binding.action_word,
            delivery_equivalence=equivalence,
            observed_occurrences=port_result.observed_occurrences,
            operational_state=state,
            reason_codes=reasons,
        )

    def _deliver(
        self,
        commitment: AtlasControllerDecisionCommitment
        | AdmissionControllerDecisionCommitment
        | DeliveryControllerDecisionCommitment,
    ) -> ExactActionDeliveryTrace:
        delivery_binding = self._delivery.implementation_binding
        commitment_identity = ObjectIdentity.from_record(commitment.commitment_id, commitment)
        if commitment.disposition is CommitmentDisposition.NONATTEMPT:
            return ExactActionDeliveryTrace(
                trace_id=f"delivery.{commitment.commitment_id}",
                commitment=commitment_identity,
                delivery=delivery_binding,
                commitment_kind=ScientificCommitmentKind.NONATTEMPT,
                expected_action_word=None,
                observed_occurrences=(),
                operational_state=OperationalDeliveryState.NOT_ATTEMPTED,
                reason_codes=commitment.reason_codes,
            )
        binding = commitment.action_binding
        if binding is None:
            raise AssertionError("scientific commitment lost its action binding")
        kind = (
            ScientificCommitmentKind.ACTION
            if commitment.disposition is CommitmentDisposition.ACTION_COMMITTED
            else ScientificCommitmentKind.MEASURED_HOLD
        )
        port_result = self._delivery.deliver(action_word=binding.action_word, commitment_kind=kind)
        reasons = _delivery_reason_codes(binding.action_word, port_result.observed_occurrences)
        if reasons:
            if port_result.failure_state not in {
                OperationalDeliveryState.DELIVERY_RECOVERY,
                OperationalDeliveryState.TERMINATED,
            }:
                raise ValueError("failed delivery lacks explicit recovery/termination state")
            state = port_result.failure_state
        else:
            if port_result.failure_state is not None:
                raise ValueError("exact delivery cannot claim recovery/termination")
            state = OperationalDeliveryState.DELIVERED
        return ExactActionDeliveryTrace(
            trace_id=f"delivery.{commitment.commitment_id}",
            commitment=commitment_identity,
            delivery=delivery_binding,
            commitment_kind=kind,
            expected_action_word=binding.action_word,
            observed_occurrences=port_result.observed_occurrences,
            operational_state=state,
            reason_codes=reasons,
        )


__all__ = [
    "CommitmentDisposition",
    'AtlasControllerDecisionCommitment',
    'AdmissionControllerDecisionCommitment',
    "ControllerRuntime",
    'AtlasControllerTickReceipt',
    'AdmissionControllerTickReceipt',
    'StageAwareAdmissionControllerTickReceipt',
    "DeliveryPortResult",
    'AtlasLiveGateReceipt',
    'AdmissionLiveGateReceipt',
    "NativeDeliveryPort",
    'ExactActionDeliveryTrace',
    'StageAwareActionDeliveryTrace',
    "ObserverDisposition",
    "ObserverEvaluation",
    "ObserverPort",
    'AtlasLiveGateEvaluation',
    'AdmissionLiveGateEvaluation',
    'AtlasLiveGateEvaluatorPort',
    'AdmissionLiveGateEvaluatorPort',
    "RuntimeObservation",
    "TickDisposition",
]

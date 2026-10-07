"""Reusable complete-unit/action-ledger contracts for independent substrate grounding target adapters.

These records sit between a target-native simulator/archive adapter and the
frozen independent substrate grounding comparison/evaluation methods.  They preserve independent units,
requested/accepted/applied/realized action semantics, receiver gauges, clocks,
sinks and invalid observations without transporting target payloads or
inventing cross-substrate coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateComparatorKind, IndependentSubstrateDenominatorCandidate, IndependentSubstrateTargetMappingCandidate, IndependentSubstrateTargetKind
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)


class IndependentSubstrateTargetPhase(StrEnum):
    DEVELOPMENT = "DEVELOPMENT"
    EVALUATION = "EVALUATION"
    PROSPECTIVE_VALIDATION = "PROSPECTIVE_VALIDATION"


class IndependentSubstrateActionDeliveryStatus(StrEnum):
    REALIZED = "REALIZED"
    REPLACED_WITH_HOLD = "REPLACED_WITH_HOLD"
    NOT_ACCEPTED = "NOT_ACCEPTED"
    APPLIED_REALIZATION_UNOBSERVED = "APPLIED_REALIZATION_UNOBSERVED"
    DELIVERY_FAILED = "DELIVERY_FAILED"
    SEMANTICS_INCOMPLETE = "SEMANTICS_INCOMPLETE"


class IndependentSubstrateUnitStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PHYSICAL_OR_NUMERICAL_SINK = "PHYSICAL_OR_NUMERICAL_SINK"
    OBSERVATION_FAILURE = "OBSERVATION_FAILURE"
    DOMAIN_LOSS = "DOMAIN_LOSS"
    EXECUTION_FAILURE = "EXECUTION_FAILURE"


class IndependentSubstrateReceiverBackactionClass(StrEnum):
    PASSIVE_BOUNDED = "PASSIVE_BOUNDED"
    ACTIVE_MEASURED = "ACTIVE_MEASURED"
    UNMEASURED = "UNMEASURED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class IndependentSubstrateNativeActionLedger(CanonicalRecord):
    """One target-native action with four distinct semantics and clocks."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-native-action-ledger'

    ledger_id: str
    native_action_id: str
    requested_action_code: str
    accepted_action_code: str | None
    applied_action_code: str | None
    realized_action_code: str | None
    request_tick: int
    acceptance_tick: int | None
    application_tick: int | None
    realization_tick: int | None
    legal: bool
    ambiguous: bool
    hold_requested: bool
    delivery_status: IndependentSubstrateActionDeliveryStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        validate_stable_id(self.native_action_id, field_name="native_action_id")
        validate_nonempty(self.requested_action_code, field_name="requested_action_code")
        for name in (
            "accepted_action_code",
            "applied_action_code",
            "realized_action_code",
        ):
            value = getattr(self, name)
            if value is not None:
                validate_nonempty(value, field_name=name)
        if self.request_tick < 0:
            raise ValueError("action request tick must be nonnegative")
        for value in (
            self.acceptance_tick,
            self.application_tick,
            self.realization_tick,
        ):
            if value is not None and (value < 0 or value < self.request_tick):
                raise ValueError("action clocks must be present in causal order")
        if (
            self.acceptance_tick is not None
            and self.application_tick is not None
            and self.application_tick < self.acceptance_tick
        ) or (
            self.application_tick is not None
            and self.realization_tick is not None
            and self.realization_tick < self.application_tick
        ):
            raise ValueError("action clocks must be present in causal order")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.legal and self.ambiguous:
            raise ValueError("an ambiguous native action cannot be legal")
        if self.hold_requested and self.requested_action_code != "HOLD":
            raise ValueError("hold request must use the native HOLD code")
        if self.delivery_status is IndependentSubstrateActionDeliveryStatus.REALIZED:
            if (
                not self.legal
                or self.ambiguous
                or None
                in (
                    self.accepted_action_code,
                    self.applied_action_code,
                    self.realized_action_code,
                    self.acceptance_tick,
                    self.application_tick,
                    self.realization_tick,
                )
            ):
                raise ValueError("realized action requires a complete valid four-clock ledger")
        elif self.delivery_status is IndependentSubstrateActionDeliveryStatus.REPLACED_WITH_HOLD:
            if (
                self.legal
                or self.applied_action_code != "HOLD"
                or self.realized_action_code != "HOLD"
                or self.application_tick is None
                or self.realization_tick is None
            ):
                raise ValueError("replacement-with-hold ledger differs")
        elif self.delivery_status is IndependentSubstrateActionDeliveryStatus.NOT_ACCEPTED:
            if any(
                value is not None
                for value in (
                    self.accepted_action_code,
                    self.applied_action_code,
                    self.realized_action_code,
                    self.acceptance_tick,
                    self.application_tick,
                    self.realization_tick,
                )
            ):
                raise ValueError("unaccepted action cannot claim later semantics or clocks")
        elif self.delivery_status is IndependentSubstrateActionDeliveryStatus.APPLIED_REALIZATION_UNOBSERVED:
            if (
                self.accepted_action_code is None
                or self.applied_action_code is None
                or self.realized_action_code is not None
                or self.acceptance_tick is None
                or self.application_tick is None
                or self.realization_tick is not None
            ):
                raise ValueError("unobserved realization ledger differs")
        elif self.delivery_status is IndependentSubstrateActionDeliveryStatus.DELIVERY_FAILED:
            if self.realized_action_code is not None or self.realization_tick is not None:
                raise ValueError("failed delivery cannot claim realized action")
        elif self.delivery_status is IndependentSubstrateActionDeliveryStatus.SEMANTICS_INCOMPLETE:
            if self.clock_complete and all(
                value is not None
                for value in (
                    self.accepted_action_code,
                    self.applied_action_code,
                    self.realized_action_code,
                )
            ):
                raise ValueError("complete action semantics cannot be marked incomplete")
        if self.delivery_status is not IndependentSubstrateActionDeliveryStatus.REALIZED and not self.reason_codes:
            raise ValueError("non-realized action requires a typed reason")

    @property
    def clock_complete(self) -> bool:
        return all(
            value is not None
            for value in (
                self.acceptance_tick,
                self.application_tick,
                self.realization_tick,
            )
        )


@dataclass(frozen=True, slots=True)
class IndependentSubstrateNativeReceiverPoint(CanonicalRecord):
    """One nested receiver/gauge observation in its native coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-native-receiver-point'

    point_id: str
    receiver_id: str
    gauge_id: str
    native_unit: str
    horizon_tick: int
    value: Decimal | None
    uncertainty: Decimal | None
    topology_state_id: str | None
    valid: bool
    backaction_class: IndependentSubstrateReceiverBackactionClass
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("point_id", "receiver_id", "gauge_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.horizon_tick < 0:
            raise ValueError("receiver horizon tick must be nonnegative")
        if self.value is not None:
            validate_decimal(self.value, field_name="value")
        if self.uncertainty is not None:
            validate_decimal(
                self.uncertainty,
                field_name="uncertainty",
                minimum=Decimal(0),
            )
        if self.topology_state_id is not None:
            validate_stable_id(self.topology_state_id, field_name="topology_state_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid:
            if self.value is None or self.reason_codes:
                raise ValueError("valid receiver point requires a value and no stop reasons")
        elif not self.reason_codes:
            raise ValueError("invalid receiver point requires a typed reason")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateCompleteTargetUnit(CanonicalRecord):
    """One physical/numerical independent unit with nested target views."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-complete-target-unit'

    unit_id: str
    slot: IndependentSubstrateTargetKind
    phase: IndependentSubstrateTargetPhase
    denominator_id: str
    history_id: str
    source_view_id: str
    preparation_or_chronic_id: str
    action_ledgers: tuple[IndependentSubstrateNativeActionLedger, ...]
    receiver_points: tuple[IndependentSubstrateNativeReceiverPoint, ...]
    sink_codes: tuple[str, ...]
    nested_view_count: int
    status: IndependentSubstrateUnitStatus
    outcome_access: OutcomeAccess
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "unit_id",
            "denominator_id",
            "history_id",
            "source_view_id",
            "preparation_or_chronic_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.slot not in {IndependentSubstrateTargetKind.GRID2OP, IndependentSubstrateTargetKind.NREL_INVERTER}:
            raise ValueError("structured target unit supports only Grid2Op or NREL")
        if (
            self.slot is IndependentSubstrateTargetKind.NREL_INVERTER
            and self.phase is IndependentSubstrateTargetPhase.PROSPECTIVE_VALIDATION
        ):
            raise ValueError("retrospective physical archive cannot create controller-use validation")
        require_sorted_unique_ids(
            self.action_ledgers,
            attribute="ledger_id",
            field_name="action_ledgers",
        )
        require_sorted_unique_ids(
            self.receiver_points,
            attribute="point_id",
            field_name="receiver_points",
        )
        require_sorted_unique_strings(self.sink_codes, field_name="sink_codes")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.nested_view_count != len(self.action_ledgers):
            raise ValueError("nested view count must equal the action-ledger roster")
        expected_access = {
            IndependentSubstrateTargetPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            IndependentSubstrateTargetPhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
            IndependentSubstrateTargetPhase.PROSPECTIVE_VALIDATION: OutcomeAccess.EVALUATION_SEALED,
        }[self.phase]
        if self.outcome_access is not expected_access:
            raise ValueError("target unit outcome access differs from its phase")
        if self.status is IndependentSubstrateUnitStatus.COMPLETE:
            if (
                not self.action_ledgers
                or not self.receiver_points
                or self.sink_codes
                or self.reason_codes
                or any(not point.valid for point in self.receiver_points)
            ):
                raise ValueError("complete target unit contains an adverse or missing view")
        elif not self.reason_codes:
            raise ValueError("adverse target unit requires a typed reason")
        if self.status is IndependentSubstrateUnitStatus.PHYSICAL_OR_NUMERICAL_SINK and not self.sink_codes:
            raise ValueError("sink unit requires an exact sink code")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateCompleteTargetPanel(CanonicalRecord):
    """Exact issued unit denominator; incomplete/adverse units remain present."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-complete-target-panel'

    panel_id: str
    slot: IndependentSubstrateTargetKind
    phase: IndependentSubstrateTargetPhase
    source_binding: ObjectIdentity
    planned_unit_ids: tuple[str, ...]
    units: tuple[IndependentSubstrateCompleteTargetUnit, ...]
    postissue_outcome_exclusion_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        require_sorted_unique_strings(
            self.planned_unit_ids,
            field_name="planned_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.units, attribute="unit_id", field_name="units")
        if tuple(value.unit_id for value in self.units) != self.planned_unit_ids:
            raise ValueError("target panel must retain every issued unit exactly once")
        if any(
            value.slot is not self.slot or value.phase is not self.phase for value in self.units
        ):
            raise ValueError("target panel mixes target slots or phases")
        if self.postissue_outcome_exclusion_allowed:
            raise ValueError("target panel cannot exclude issued units after outcomes")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateStructuredTargetDesignFreeze(CanonicalRecord):
    """Common G2/G3 design freeze consumed by target-specific adapters."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-structured-target-design-freeze'

    design_id: str
    slot: IndependentSubstrateTargetKind
    source_binding: ObjectIdentity
    common_design_basis: ObjectIdentity
    mapping_candidates: tuple[IndependentSubstrateTargetMappingCandidate, ...]
    denominator_candidates: tuple[IndependentSubstrateDenominatorCandidate, ...]
    native_action_ids: tuple[str, ...]
    receiver_gauge_ids: tuple[str, ...]
    horizon_ticks: tuple[int, ...]
    graph_or_observation_resolution_ids: tuple[str, ...]
    comparator_kinds: tuple[IndependentSubstrateComparatorKind, ...]
    minimum_development_units: int
    minimum_evaluation_units: int
    minimum_prospective_validation_units: int
    physical_response_margin_rule: str
    certification_margin_rule: str
    complete_unit_resampling: bool
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        if self.slot not in {IndependentSubstrateTargetKind.GRID2OP, IndependentSubstrateTargetKind.NREL_INVERTER}:
            raise ValueError("structured design supports only Grid2Op or NREL")
        require_sorted_unique_ids(
            self.mapping_candidates,
            attribute="candidate_id",
            field_name="mapping_candidates",
        )
        if not self.mapping_candidates or any(
            value.target_slot is not self.slot for value in self.mapping_candidates
        ):
            raise ValueError("structured design mapping roster differs")
        require_sorted_unique_ids(
            self.denominator_candidates,
            attribute="candidate_id",
            field_name="denominator_candidates",
        )
        if len(self.denominator_candidates) < 4:
            raise ValueError("structured design requires minimal/split/merge/omission alternatives")
        require_sorted_unique_strings(
            self.native_action_ids,
            field_name="native_action_ids",
            allow_empty=False,
        )
        if "hold" not in self.native_action_ids:
            raise ValueError("structured design requires a native hold action")
        require_sorted_unique_strings(
            self.receiver_gauge_ids,
            field_name="receiver_gauge_ids",
            allow_empty=False,
        )
        if len(self.receiver_gauge_ids) < 2:
            raise ValueError("structured design requires at least two receiver gauges")
        if (
            not self.horizon_ticks
            or tuple(sorted(set(self.horizon_ticks))) != self.horizon_ticks
            or self.horizon_ticks[0] <= 0
        ):
            raise ValueError("structured design horizons must be sorted positive ticks")
        require_sorted_unique_strings(
            self.graph_or_observation_resolution_ids,
            field_name="graph_or_observation_resolution_ids",
            allow_empty=False,
        )
        if len(self.graph_or_observation_resolution_ids) < 2:
            raise ValueError("structured design requires a within-system resolution contrast")
        if tuple(sorted(set(self.comparator_kinds), key=lambda value: value.value)) != tuple(
            sorted(self.comparator_kinds, key=lambda value: value.value)
        ) or set(self.comparator_kinds) != set(IndependentSubstrateComparatorKind):
            raise ValueError("structured design comparator roster differs from outcome-blind scientific grammar")
        if min(self.minimum_development_units, self.minimum_evaluation_units) <= 0:
            raise ValueError("structured design requires positive development/evaluation panels")
        if self.slot is IndependentSubstrateTargetKind.GRID2OP:
            if self.minimum_prospective_validation_units <= 0:
                raise ValueError("Grid2Op design requires a conditional controller-use panel")
        elif self.minimum_prospective_validation_units != 0:
            raise ValueError("NREL archive design cannot promise controller-use units")
        validate_nonempty(
            self.physical_response_margin_rule,
            field_name="physical_response_margin_rule",
        )
        validate_nonempty(
            self.certification_margin_rule,
            field_name="certification_margin_rule",
        )
        if self.physical_response_margin_rule == self.certification_margin_rule:
            raise ValueError("physical and certification margins must remain distinct")
        if (
            not self.complete_unit_resampling
            or not self.frozen_before_development
            or self.protected_outcome_access_count
        ):
            raise ValueError("structured target design crossed a causal/power boundary")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateStructuredPowerFreeze(CanonicalRecord):
    """Joint panel/resolution/receiver power decision before target issue."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-structured-power-freeze'

    freeze_id: str
    design: ObjectIdentity
    development_unit_ids: tuple[str, ...]
    evaluation_unit_ids: tuple[str, ...]
    prospective_validation_unit_ids: tuple[str, ...]
    simultaneous_family_ids: tuple[str, ...]
    joint_precision_passed: bool
    panel_envelope_limited: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        if self.design.object_schema != IndependentSubstrateStructuredTargetDesignFreeze.SCHEMA:
            raise ValueError("structured power freeze design identity differs")
        for name in (
            "development_unit_ids",
            "evaluation_unit_ids",
            "prospective_validation_unit_ids",
            "simultaneous_family_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name == "prospective_validation_unit_ids",
            )
        rosters = (
            set(self.development_unit_ids),
            set(self.evaluation_unit_ids),
            set(self.prospective_validation_unit_ids),
        )
        if any(rosters[left] & rosters[right] for left, right in ((0, 1), (0, 2), (1, 2))):
            raise ValueError("structured power rosters must be disjoint")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.joint_precision_passed == self.panel_envelope_limited:
            raise ValueError("joint precision and panel limitation must be complements")
        if self.joint_precision_passed and self.reason_codes:
            raise ValueError("passed structured power freeze cannot retain stop reasons")
        if self.panel_envelope_limited and not self.reason_codes:
            raise ValueError("panel-limited power freeze requires a typed reason")


@dataclass(frozen=True, slots=True)
class IndependentSubstrateStructuredPhaseIssue(CanonicalRecord):
    """Exact G3/G4 phase issue boundary before any affected target outcome."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-structured-phase-issue'

    issue_id: str
    slot: IndependentSubstrateTargetKind
    phase: IndependentSubstrateTargetPhase
    source_binding: ObjectIdentity
    target_design: ObjectIdentity
    power_freeze: ObjectIdentity
    prediction_contract: ObjectIdentity | None
    planned_unit_ids: tuple[str, ...]
    issued_before_outcome_access: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        if self.slot not in {IndependentSubstrateTargetKind.GRID2OP, IndependentSubstrateTargetKind.NREL_INVERTER}:
            raise ValueError("structured phase issue supports only Grid2Op or NREL")
        if self.target_design.object_schema != IndependentSubstrateStructuredTargetDesignFreeze.SCHEMA:
            raise ValueError("structured phase issue target design differs")
        if self.power_freeze.object_schema != IndependentSubstrateStructuredPowerFreeze.SCHEMA:
            raise ValueError("structured phase issue power freeze differs")
        require_sorted_unique_strings(
            self.planned_unit_ids,
            field_name="planned_unit_ids",
            allow_empty=False,
        )
        if self.phase is IndependentSubstrateTargetPhase.DEVELOPMENT:
            if self.prediction_contract is not None:
                raise ValueError("development issue cannot bind a future prediction contract")
        elif (
            self.prediction_contract is None
            or self.prediction_contract.object_schema
            != 'empirical-lawhood/independent-substrate-grounding/independent-substrate-target-prediction-contract'
        ):
            raise ValueError("evaluation/controller use issue requires the immutable prediction contract")
        if (
            self.slot is IndependentSubstrateTargetKind.NREL_INVERTER
            and self.phase is IndependentSubstrateTargetPhase.PROSPECTIVE_VALIDATION
        ):
            raise ValueError("retrospective NREL archive cannot issue controller use")
        if not self.issued_before_outcome_access or self.protected_outcome_access_count:
            raise ValueError("structured phase issue crossed protected outcome access")


def build_structured_phase_issue(
    *,
    design: IndependentSubstrateStructuredTargetDesignFreeze,
    power_freeze: IndependentSubstrateStructuredPowerFreeze,
    phase: IndependentSubstrateTargetPhase,
    prediction_contract: ObjectIdentity | None,
) -> IndependentSubstrateStructuredPhaseIssue:
    """Bind one exact powered roster; panel limitation is evidence, not issue."""

    if power_freeze.design != ObjectIdentity.from_record(design.design_id, design):
        raise ValueError("structured phase issue design/power identities differ")
    if not power_freeze.joint_precision_passed or power_freeze.panel_envelope_limited:
        raise ValueError("panel-limited structured target is not issue eligible")
    planned = {
        IndependentSubstrateTargetPhase.DEVELOPMENT: power_freeze.development_unit_ids,
        IndependentSubstrateTargetPhase.EVALUATION: power_freeze.evaluation_unit_ids,
        IndependentSubstrateTargetPhase.PROSPECTIVE_VALIDATION: power_freeze.prospective_validation_unit_ids,
    }[phase]
    if not planned:
        raise ValueError("structured phase issue has no powered units")
    minimum = {
        IndependentSubstrateTargetPhase.DEVELOPMENT: design.minimum_development_units,
        IndependentSubstrateTargetPhase.EVALUATION: design.minimum_evaluation_units,
        IndependentSubstrateTargetPhase.PROSPECTIVE_VALIDATION: design.minimum_prospective_validation_units,
    }[phase]
    if len(planned) < minimum:
        raise ValueError("structured phase issue roster is below the frozen minimum")
    return IndependentSubstrateStructuredPhaseIssue(
        issue_id=(f"issue.independent-substrate-grounding.{design.slot.value.lower()}.{phase.value.lower()}"),
        slot=design.slot,
        phase=phase,
        source_binding=design.source_binding,
        target_design=ObjectIdentity.from_record(design.design_id, design),
        power_freeze=ObjectIdentity.from_record(power_freeze.freeze_id, power_freeze),
        prediction_contract=prediction_contract,
        planned_unit_ids=planned,
        issued_before_outcome_access=True,
        protected_outcome_access_count=0,
    )


@dataclass(frozen=True, slots=True)
class IndependentSubstrateStructuredPanelAudit(CanonicalRecord):
    """Compact complete-unit audit; it is not a target match verdict."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/independent-substrate-grounding/independent-substrate-structured-panel-audit'

    audit_id: str
    panel: ObjectIdentity
    planned_unit_count: int
    complete_unit_count: int
    sink_unit_count: int
    observation_failure_unit_count: int
    domain_loss_unit_count: int
    execution_failure_unit_count: int
    requested_action_count: int
    accepted_action_count: int
    applied_action_count: int
    realized_action_count: int
    replacement_hold_count: int
    missing_realization_count: int
    action_clock_error_count: int
    two_receiver_shared_unit_count: int
    two_passive_receiver_shared_unit_count: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        if self.panel.object_schema != IndependentSubstrateCompleteTargetPanel.SCHEMA:
            raise ValueError("structured audit panel identity differs")
        counts = (
            self.planned_unit_count,
            self.complete_unit_count,
            self.sink_unit_count,
            self.observation_failure_unit_count,
            self.domain_loss_unit_count,
            self.execution_failure_unit_count,
            self.requested_action_count,
            self.accepted_action_count,
            self.applied_action_count,
            self.realized_action_count,
            self.replacement_hold_count,
            self.missing_realization_count,
            self.action_clock_error_count,
            self.two_receiver_shared_unit_count,
            self.two_passive_receiver_shared_unit_count,
        )
        if any(value < 0 for value in counts):
            raise ValueError("structured audit counts must be nonnegative")
        if (
            self.complete_unit_count
            + self.sink_unit_count
            + self.observation_failure_unit_count
            + self.domain_loss_unit_count
            + self.execution_failure_unit_count
            != self.planned_unit_count
        ):
            raise ValueError("structured audit unit statuses do not close")
        if not (
            self.realized_action_count
            <= self.applied_action_count
            <= self.accepted_action_count
            <= self.requested_action_count
        ):
            raise ValueError("structured audit action counts violate causal ordering")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def audit_structured_target_panel(
    panel: IndependentSubstrateCompleteTargetPanel,
) -> IndependentSubstrateStructuredPanelAudit:
    """Count complete units and action/receiver failures without row inflation."""

    status_counts = {status: 0 for status in IndependentSubstrateUnitStatus}
    ledgers = tuple(ledger for unit in panel.units for ledger in unit.action_ledgers)
    for unit in panel.units:
        status_counts[unit.status] += 1
    two_receiver = 0
    two_passive = 0
    for unit in panel.units:
        receiver_ids = {point.receiver_id for point in unit.receiver_points if point.valid}
        passive_ids = {
            point.receiver_id
            for point in unit.receiver_points
            if point.valid
            and point.backaction_class is IndependentSubstrateReceiverBackactionClass.PASSIVE_BOUNDED
        }
        two_receiver += len(receiver_ids) >= 2
        two_passive += len(passive_ids) >= 2
    missing_realization = sum(
        ledger.delivery_status
        in {
            IndependentSubstrateActionDeliveryStatus.APPLIED_REALIZATION_UNOBSERVED,
            IndependentSubstrateActionDeliveryStatus.DELIVERY_FAILED,
        }
        for ledger in ledgers
    )
    clock_errors = sum(
        ledger.delivery_status is IndependentSubstrateActionDeliveryStatus.SEMANTICS_INCOMPLETE
        or (
            ledger.delivery_status is IndependentSubstrateActionDeliveryStatus.REALIZED
            and not ledger.clock_complete
        )
        for ledger in ledgers
    )
    reasons: set[str] = set()
    if missing_realization:
        reasons.add("ACTION_REALIZATION_INCOMPLETE")
    if clock_errors:
        reasons.add("ACTION_CLOCK_INCOMPLETE")
    if status_counts[IndependentSubstrateUnitStatus.OBSERVATION_FAILURE]:
        reasons.add("OBSERVATION_FAILURE_RETAINED")
    if status_counts[IndependentSubstrateUnitStatus.PHYSICAL_OR_NUMERICAL_SINK]:
        reasons.add("SINK_UNIT_RETAINED")
    if two_passive < two_receiver:
        reasons.add("TWO_RECEIVER_BACKACTION_NOT_BOUNDED")
    return IndependentSubstrateStructuredPanelAudit(
        audit_id=f"audit.{panel.panel_id}",
        panel=ObjectIdentity.from_record(panel.panel_id, panel),
        planned_unit_count=len(panel.units),
        complete_unit_count=status_counts[IndependentSubstrateUnitStatus.COMPLETE],
        sink_unit_count=status_counts[IndependentSubstrateUnitStatus.PHYSICAL_OR_NUMERICAL_SINK],
        observation_failure_unit_count=status_counts[IndependentSubstrateUnitStatus.OBSERVATION_FAILURE],
        domain_loss_unit_count=status_counts[IndependentSubstrateUnitStatus.DOMAIN_LOSS],
        execution_failure_unit_count=status_counts[IndependentSubstrateUnitStatus.EXECUTION_FAILURE],
        requested_action_count=len(ledgers),
        accepted_action_count=sum(ledger.accepted_action_code is not None for ledger in ledgers),
        applied_action_count=sum(ledger.applied_action_code is not None for ledger in ledgers),
        realized_action_count=sum(ledger.realized_action_code is not None for ledger in ledgers),
        replacement_hold_count=sum(
            ledger.delivery_status is IndependentSubstrateActionDeliveryStatus.REPLACED_WITH_HOLD
            for ledger in ledgers
        ),
        missing_realization_count=missing_realization,
        action_clock_error_count=clock_errors,
        two_receiver_shared_unit_count=two_receiver,
        two_passive_receiver_shared_unit_count=two_passive,
        reason_codes=tuple(sorted(reasons)),
    )


__all__ = [
    'IndependentSubstrateActionDeliveryStatus',
    'IndependentSubstrateCompleteTargetPanel',
    'IndependentSubstrateCompleteTargetUnit',
    'IndependentSubstrateNativeActionLedger',
    'IndependentSubstrateNativeReceiverPoint',
    'IndependentSubstrateReceiverBackactionClass',
    'IndependentSubstrateStructuredPanelAudit',
    'IndependentSubstrateStructuredPhaseIssue',
    'IndependentSubstrateStructuredPowerFreeze',
    'IndependentSubstrateStructuredTargetDesignFreeze',
    'IndependentSubstrateTargetPhase',
    'IndependentSubstrateUnitStatus',
    "audit_structured_target_panel",
    "build_structured_phase_issue",
]

"""Pure vocabulary for finite thermodynamic-response categories.

These records freeze compatibility, ontology and claim-ceiling semantics.  They
contain no trajectories, estimators, source access, authority or persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from .response_algebra import ResponseAlgebraSignature
from .serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    require_unique_ids,
    validate_decimal,
    validate_nonempty,
    validate_schema,
    validate_stable_id,
)
from .quantities import QuantityKind
from .references import NamedDecimal
from .response_algebra import ActionStage
from .systems import BalanceRole, RelationalIdentity, SystemSpec
from .worlds import WorldKind


class ScalarDoseActionWordSignatureStatus(StrEnum):
    EXACT_SCALAR_DOSE = "EXACT_SCALAR_DOSE"
    RECEIPT_BOUND_SCALAR_DOSE_PARENT = "RECEIPT_BOUND_SCALAR_DOSE_PARENT"
    SCALAR_DOSE_NOT_APPLICABLE = "SCALAR_DOSE_NOT_APPLICABLE"
    UNEVALUABLE = "UNEVALUABLE"


class ScalarDoseActionWordApplicabilityCriterion(StrEnum):
    STAGE_UNIT_CONTRACT = "STAGE_UNIT_CONTRACT"
    SCALAR_NONNEGATIVE_DOSE = "SCALAR_NONNEGATIVE_DOSE"
    MATCHED_WORD_SCHEDULES = "MATCHED_WORD_SCHEDULES"
    EXACT_SIMULTANEITY = "EXACT_SIMULTANEITY"
    COMPLETE_PREFIXES_AND_CONTROLS = "COMPLETE_PREFIXES_AND_CONTROLS"
    RECEIVER_AND_HORIZON_CONTRACT = "RECEIVER_AND_HORIZON_CONTRACT"
    COMPLETE_UNIT_AGGREGATION = "COMPLETE_UNIT_AGGREGATION"


class ApplicabilityDisposition(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNEVALUABLE = "UNEVALUABLE"


class OntologyCompatibilityStatus(StrEnum):
    COMPATIBLE = "COMPATIBLE"
    UNEVALUABLE = "UNEVALUABLE"


class ChronologyConvention(StrEnum):
    LEFT_TO_RIGHT_DELIVERY = "LEFT_TO_RIGHT_DELIVERY"


class PartialCompositionStatus(StrEnum):
    DEFINED = "DEFINED"
    SOURCE_TARGET_MISMATCH = "SOURCE_TARGET_MISMATCH"
    CLOCK_MISMATCH = "CLOCK_MISMATCH"
    DELIVERY_INVALID = "DELIVERY_INVALID"
    PREPARATION_MISMATCH = "PREPARATION_MISMATCH"
    SUPPORT_MISMATCH = "SUPPORT_MISMATCH"
    UNEVALUABLE = "UNEVALUABLE"


class ThermodynamicCoordinateRole(StrEnum):
    ACTION_COMMAND = "ACTION_COMMAND"
    BATH_STATE = "BATH_STATE"
    BOUNDARY_TEMPERATURE = "BOUNDARY_TEMPERATURE"
    HEAT_ENTROPY_TRANSFER = "HEAT_ENTROPY_TRANSFER"
    HEAT_TRANSFER = "HEAT_TRANSFER"
    MATTER_ENTROPY_TRANSFER = "MATTER_ENTROPY_TRANSFER"
    MATTER_TRANSFER = "MATTER_TRANSFER"
    OTHER_BOUNDARY_TRANSFER = "OTHER_BOUNDARY_TRANSFER"
    RECEIVER = "RECEIVER"
    STATE_ENTROPY = "STATE_ENTROPY"
    STORED_ENERGY = "STORED_ENERGY"
    VALIDITY = "VALIDITY"
    WORK_TRANSFER = "WORK_TRANSFER"


class EnergyTransferMode(StrEnum):
    WORK = "WORK"
    HEAT = "HEAT"
    MATTER_ENTHALPY = "MATTER_ENTHALPY"
    OTHER_BOUNDARY = "OTHER_BOUNDARY"


class ThermodynamicSignConvention(StrEnum):
    INTO_SYSTEM_POSITIVE = "INTO_SYSTEM_POSITIVE"
    FINAL_MINUS_INITIAL = "FINAL_MINUS_INITIAL"


class BalanceClosureStatus(StrEnum):
    CLOSED = "CLOSED"
    NOT_CLOSED = "NOT_CLOSED"
    TERM_UNOBSERVED = "TERM_UNOBSERVED"
    ANNULAR = "ANNULAR"
    UNEVALUABLE = "UNEVALUABLE"


class ReturnQualificationStatus(StrEnum):
    RETURN_QUALIFIED = "RETURN_QUALIFIED"
    NON_RETURNING = "NON_RETURNING"
    ANNULAR = "ANNULAR"
    UNEVALUABLE = "UNEVALUABLE"


class EntropyEligibilityStatus(StrEnum):
    ELIGIBLE = "ELIGIBLE"
    ENERGY_BALANCE_ONLY = "ENERGY_BALANCE_ONLY"
    DISSIPATIVE_CLOSURE_PROXY_ONLY = "DISSIPATIVE_CLOSURE_PROXY_ONLY"
    LEDGER_PARTIAL = "LEDGER_PARTIAL"
    UNEVALUABLE = "UNEVALUABLE"


class ThermodynamicClaimCeiling(StrEnum):
    RESPONSE_CATEGORY_ONLY = "RESPONSE_CATEGORY_ONLY"
    THERMODYNAMIC_LEDGER_PARTIAL = "THERMODYNAMIC_LEDGER_PARTIAL"
    ENERGY_BALANCE_ONLY = "ENERGY_BALANCE_ONLY"
    DISSIPATIVE_CLOSURE_PROXY_ONLY = "DISSIPATIVE_CLOSURE_PROXY_ONLY"
    ENTROPY_PRODUCTION_ELIGIBLE = "ENTROPY_PRODUCTION_ELIGIBLE"
    THERMODYNAMIC_INTERPRETATION_UNEVALUABLE = "THERMODYNAMIC_INTERPRETATION_UNEVALUABLE"


class CompositeSignatureAxis(StrEnum):
    DELIVERED_ACTION_OBSERVABILITY = "DELIVERED_ACTION_OBSERVABILITY"
    CONSTITUENT_PORT_MATERIALITY = "CONSTITUENT_PORT_MATERIALITY"
    ACTION_IMAGE_QUOTIENT = "ACTION_IMAGE_QUOTIENT"
    SIMULTANEOUS_FINITE_DEFECT = "SIMULTANEOUS_FINITE_DEFECT"
    CONTROLLED_SEQUENTIAL_FINITE_DEFECT = "CONTROLLED_SEQUENTIAL_FINITE_DEFECT"
    REPEAT_AND_LONGER_WORD_CURVATURE = "REPEAT_AND_LONGER_WORD_CURVATURE"
    TEMPORAL_COCYCLE_CLOSURE = "TEMPORAL_COCYCLE_CLOSURE"
    EQUAL_LAG_STATIONARITY = "EQUAL_LAG_STATIONARITY"
    RECEIVER_NATURALITY_AND_FAITHFULNESS = "RECEIVER_NATURALITY_AND_FAITHFULNESS"
    PREPARATION_STATE_AND_BATH_RETURN = "PREPARATION_STATE_AND_BATH_RETURN"
    ABSOLUTE_ENERGY_CLOSURE = "ABSOLUTE_ENERGY_CLOSURE"
    ENTROPY_PRODUCTION_ELIGIBILITY = "ENTROPY_PRODUCTION_ELIGIBILITY"
    DENOMINATOR_REGIME_STABILITY_AND_AGEING = "DENOMINATOR_REGIME_STABILITY_AND_AGEING"


class ThermodynamicResponseTerminalStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    MIXED = "MIXED"
    ANNULAR = "ANNULAR"
    UNEVALUABLE = "UNEVALUABLE"
    STOPPED = "STOPPED"
    NOT_ATTEMPTED = "NOT_ATTEMPTED"
    REQUIRES_NEW_DATA = "REQUIRES_NEW_DATA"


class ThermodynamicPromotionLevel(StrEnum):
    LOCAL_SIGNATURE = "LOCAL_SIGNATURE"
    SUBSTRATE_CLASS = "SUBSTRATE_CLASS"
    STRUCTURAL_CLASS_CANDIDATE = "STRUCTURAL_CLASS_CANDIDATE"
    STRUCTURAL_UNIVERSALITY_CLASS = "STRUCTURAL_UNIVERSALITY_CLASS"


class DeliveryQualification(StrEnum):
    QUALIFIED = "QUALIFIED"
    INVALID = "INVALID"
    UNOBSERVABLE = "UNOBSERVABLE"
    UNEVALUABLE = "UNEVALUABLE"


class FiniteWordMode(StrEnum):
    IDENTITY = "IDENTITY"
    SEQUENTIAL = "SEQUENTIAL"
    SIMULTANEOUS = "SIMULTANEOUS"


class FiniteAxisDisposition(StrEnum):
    MATERIAL = "MATERIAL"
    RESOLVED_SUBMATERIAL = "RESOLVED_SUBMATERIAL"
    EQUIVALENT = "EQUIVALENT"
    BELOW_RESOLUTION = "BELOW_RESOLUTION"
    TIMING_OR_BATH_EXPLAINED = "TIMING_OR_BATH_EXPLAINED"
    NONSTATIONARY = "NONSTATIONARY"
    STATIONARY = "STATIONARY"
    COCYCLIC = "COCYCLIC"
    NONCLOSED = "NONCLOSED"
    ANNULAR = "ANNULAR"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNEVALUABLE = "UNEVALUABLE"


class ReceiverMapDisposition(StrEnum):
    FAITHFUL_AT_TESTED_RESOLUTION = "FAITHFUL_AT_TESTED_RESOLUTION"
    PARTIALLY_COLLAPSING = "PARTIALLY_COLLAPSING"
    ACTION_BLIND = "ACTION_BLIND"
    BATH_BLIND = "BATH_BLIND"
    NON_NATURAL_AT_TESTED_RESOLUTION = "NON_NATURAL_AT_TESTED_RESOLUTION"
    UNEVALUABLE = "UNEVALUABLE"


class ExchangeEstimandKind(StrEnum):
    TOTAL_EXCHANGE = "TOTAL_EXCHANGE"
    IDENTITY_DIFFERENCED_EXCHANGE = "IDENTITY_DIFFERENCED_EXCHANGE"
    HOUSEKEEPING_EXCHANGE = "HOUSEKEEPING_EXCHANGE"
    EXCESS_EXCHANGE = "EXCESS_EXCHANGE"
    ABSOLUTE_STORAGE_CHANGE = "ABSOLUTE_STORAGE_CHANGE"
    ABSOLUTE_STATE_ENTROPY_CHANGE = "ABSOLUTE_STATE_ENTROPY_CHANGE"


class ThermodynamicObservationStatus(StrEnum):
    OBSERVED = "OBSERVED"
    TERM_UNOBSERVED = "TERM_UNOBSERVED"
    INVALID = "INVALID"
    UNEVALUABLE = "UNEVALUABLE"


class EntropyTermRole(StrEnum):
    STATE_ENTROPY = "STATE_ENTROPY"
    HEAT_OVER_BOUNDARY_TEMPERATURE = "HEAT_OVER_BOUNDARY_TEMPERATURE"
    MATTER_ENTROPY_TRANSFER = "MATTER_ENTROPY_TRANSFER"
    CHEMICAL_OR_REACTION = "CHEMICAL_OR_REACTION"
    OTHER_DECLARED_TRANSFER = "OTHER_DECLARED_TRANSFER"


class TransportDisposition(StrEnum):
    COMMUTES_AT_TESTED_RESOLUTION = "COMMUTES_AT_TESTED_RESOLUTION"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class PreparedResponseObject(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/prepared-response-object'

    object_id: str
    relation_id: str
    preparation_id: str
    denominator_id: str
    bath_view_id: str
    retained_history_id: str
    clock_id: str
    clock_coordinate: Decimal
    support_id: str
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("object_id", self.object_id),
            ("relation_id", self.relation_id),
            ("preparation_id", self.preparation_id),
            ("denominator_id", self.denominator_id),
            ("bath_view_id", self.bath_view_id),
            ("retained_history_id", self.retained_history_id),
            ("clock_id", self.clock_id),
            ("support_id", self.support_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.clock_coordinate, field_name="clock_coordinate")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid and self.reason_codes:
            raise ValueError("valid prepared response object cannot carry reason codes")
        if not self.valid and not self.reason_codes:
            raise ValueError("invalid prepared response object requires reason codes")


@dataclass(frozen=True, slots=True)
class ActionStageObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-stage-observation'

    stage: ActionStage
    quantity_id: str
    value: Decimal
    native_unit: str
    clock_id: str
    clock_coordinate: Decimal
    uncertainty: Decimal
    observation_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_stable_id(self.clock_id, field_name="clock_id")
        validate_stable_id(self.observation_id, field_name="observation_id")
        validate_decimal(self.value, field_name="value")
        validate_decimal(self.clock_coordinate, field_name="clock_coordinate")
        validate_decimal(self.uncertainty, field_name="uncertainty", minimum=Decimal("0"))


@dataclass(frozen=True, slots=True)
class ActionInterval(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-interval'

    start: Decimal
    end: Decimal
    clock_id: str
    time_unit: str

    def __post_init__(self) -> None:
        validate_decimal(self.start, field_name="start")
        validate_decimal(self.end, field_name="end")
        if self.end <= self.start:
            raise ValueError("action interval end must follow start")
        validate_stable_id(self.clock_id, field_name="clock_id")
        validate_nonempty(self.time_unit, field_name="time_unit")


@dataclass(frozen=True, slots=True)
class DeliveredDoseComponent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/delivered-dose-component'

    quantity_id: str
    value: Decimal
    native_unit: str
    uncertainty: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_decimal(self.value, field_name="value")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_decimal(self.uncertainty, field_name="uncertainty", minimum=Decimal("0"))


@dataclass(frozen=True, slots=True)
class ActionDelivery(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-delivery'

    delivery_id: str
    letter_id: str
    port_id: str
    stages: tuple[ActionStageObservation, ...]
    planned_interval: ActionInterval
    realized_interval: ActionInterval
    pulse_shape_id: str
    delivered_dose: tuple[DeliveredDoseComponent, ...]
    sign: int
    slew_rate: Decimal | None
    slew_rate_unit: str | None
    preparation_id: str
    causal_prefix_id: str
    support_id: str
    interlock_active: bool
    clipped: bool
    saturated: bool
    calibration_id: str
    actuator_id: str
    synchronization_id: str
    qualification: DeliveryQualification
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("delivery_id", self.delivery_id),
            ("letter_id", self.letter_id),
            ("port_id", self.port_id),
            ("pulse_shape_id", self.pulse_shape_id),
            ("preparation_id", self.preparation_id),
            ("causal_prefix_id", self.causal_prefix_id),
            ("support_id", self.support_id),
            ("calibration_id", self.calibration_id),
            ("actuator_id", self.actuator_id),
            ("synchronization_id", self.synchronization_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(self.stages, attribute="observation_id", field_name="stages")
        if {stage.stage for stage in self.stages} != set(ActionStage):
            raise ValueError("action delivery requires every requested-to-realized stage")
        require_sorted_unique_ids(
            self.delivered_dose,
            attribute="quantity_id",
            field_name="delivered_dose",
        )
        if not self.delivered_dose:
            raise ValueError("action delivery requires at least one delivered-dose component")
        if self.sign not in {-1, 0, 1}:
            raise ValueError("action-delivery sign must be -1, 0 or 1")
        if self.slew_rate is None:
            if self.slew_rate_unit is not None:
                raise ValueError("slew-rate unit requires a slew-rate value")
        else:
            validate_decimal(self.slew_rate, field_name="slew_rate", minimum=Decimal("0"))
            if self.slew_rate_unit is None:
                raise ValueError("slew-rate value requires a native unit")
            validate_nonempty(self.slew_rate_unit, field_name="slew_rate_unit")
        if self.planned_interval.clock_id != self.realized_interval.clock_id:
            raise ValueError("planned and realized action intervals require one mapped clock")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        invalid_flags = self.interlock_active or self.clipped or self.saturated
        if self.qualification is DeliveryQualification.QUALIFIED:
            if invalid_flags or self.reason_codes:
                raise ValueError("qualified action delivery cannot carry invalid flags or reasons")
        elif not self.reason_codes:
            raise ValueError("nonqualified action delivery requires reason codes")


@dataclass(frozen=True, slots=True)
class FiniteWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/finite-word'

    word_id: str
    mode: FiniteWordMode
    source_object_id: str
    target_object_id: str
    deliveries: tuple[ActionDelivery, ...]
    prefix_object_ids: tuple[str, ...]
    chronology_convention: ChronologyConvention
    composition_status: PartialCompositionStatus
    overlap_tolerance: Decimal
    overlap_tolerance_unit: str
    elapsed_time: Decimal
    elapsed_time_unit: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("word_id", self.word_id),
            ("source_object_id", self.source_object_id),
            ("target_object_id", self.target_object_id),
        ):
            validate_stable_id(value, field_name=name)
        # Delivery order is the word's chronology, not a set-like canonical field.
        # Sorting here would silently identify AB with BA whenever delivery IDs sort
        # differently from realized time.
        require_unique_ids(self.deliveries, attribute="delivery_id", field_name="deliveries")
        for object_id in self.prefix_object_ids:
            validate_stable_id(object_id, field_name="prefix_object_ids")
        expected_prefix_count = (
            len(self.deliveries) + 1 if self.mode is FiniteWordMode.SEQUENTIAL else 2
        )
        if len(self.prefix_object_ids) != expected_prefix_count:
            raise ValueError("finite word has the wrong source/prefix/target object count")
        if self.prefix_object_ids[0] != self.source_object_id:
            raise ValueError("finite-word first prefix object must be its source")
        if self.prefix_object_ids[-1] != self.target_object_id:
            raise ValueError("finite-word last prefix object must be its target")
        validate_decimal(
            self.overlap_tolerance, field_name="overlap_tolerance", minimum=Decimal("0")
        )
        validate_nonempty(self.overlap_tolerance_unit, field_name="overlap_tolerance_unit")
        validate_decimal(self.elapsed_time, field_name="elapsed_time", minimum=Decimal("0"))
        if self.elapsed_time == 0:
            raise ValueError("finite-word elapsed time must be positive")
        validate_nonempty(self.elapsed_time_unit, field_name="elapsed_time_unit")
        if self.mode is FiniteWordMode.IDENTITY and self.deliveries:
            raise ValueError("identity finite word cannot contain deliveries")
        if self.mode is FiniteWordMode.IDENTITY and self.source_object_id == self.target_object_id:
            raise ValueError("elapsed no-action response requires distinct time-indexed objects")
        if self.mode is not FiniteWordMode.IDENTITY and not self.deliveries:
            raise ValueError("nonidentity finite word requires deliveries")
        clock_ids = {delivery.realized_interval.clock_id for delivery in self.deliveries}
        units = {delivery.realized_interval.time_unit for delivery in self.deliveries}
        if self.deliveries and (len(clock_ids) != 1 or units != {self.elapsed_time_unit}):
            raise ValueError("finite composition requires one mapped clock and elapsed-time unit")
        starts = tuple(delivery.realized_interval.start for delivery in self.deliveries)
        if self.mode is FiniteWordMode.SEQUENTIAL:
            if any(later <= earlier for earlier, later in zip(starts, starts[1:])):
                raise ValueError("sequential finite-word realized starts must increase")
            intervals = tuple(delivery.realized_interval for delivery in self.deliveries)
            if any(later.start < earlier.end for earlier, later in zip(intervals, intervals[1:])):
                raise ValueError("sequential finite-word realized intervals cannot overlap")
            for delivery, prefix_id in zip(
                self.deliveries, self.prefix_object_ids[:-1], strict=True
            ):
                if delivery.causal_prefix_id != prefix_id:
                    raise ValueError("sequential delivery has the wrong causal prefix object")
        if self.mode is FiniteWordMode.SIMULTANEOUS:
            if len(self.deliveries) < 2:
                raise ValueError("simultaneous finite word requires at least two deliveries")
            if len(clock_ids) != 1 or units != {self.overlap_tolerance_unit}:
                raise ValueError("simultaneous qualification requires one mapped clock and unit")
            if any(
                delivery.causal_prefix_id != self.source_object_id for delivery in self.deliveries
            ):
                raise ValueError("simultaneous deliveries require one common causal source")
            latest_start = max(starts)
            earliest_end = min(delivery.realized_interval.end for delivery in self.deliveries)
            if latest_start > earliest_end + self.overlap_tolerance:
                raise ValueError(
                    "simultaneous finite-word intervals do not overlap within tolerance"
                )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.composition_status is PartialCompositionStatus.DEFINED:
            if self.reason_codes:
                raise ValueError("defined finite composition cannot carry reason codes")
            if any(
                delivery.qualification is not DeliveryQualification.QUALIFIED
                for delivery in self.deliveries
            ):
                raise ValueError("defined finite composition requires qualified deliveries")
        elif not self.reason_codes:
            raise ValueError("absent or unevaluable composition requires reason codes")


@dataclass(frozen=True, slots=True)
class FiniteResponseSignature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/finite-response-signature'

    signature_id: str
    relation_id: str
    delivery_qualification: DeliveryQualification
    constituent_port_materiality: FiniteAxisDisposition
    action_image_rank: int | None
    action_kernel_word_ids: tuple[str, ...]
    simultaneous_defect: FiniteAxisDisposition
    controlled_sequential_defect: FiniteAxisDisposition
    repeat_and_longer_word_curvature: FiniteAxisDisposition
    temporal_cocycle: FiniteAxisDisposition
    equal_lag_stationarity: FiniteAxisDisposition
    receiver_map: ReceiverMapDisposition
    lie_axis: FiniteAxisDisposition
    metrics: tuple[NamedDecimal, ...]
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.signature_id, field_name="signature_id")
        validate_stable_id(self.relation_id, field_name="relation_id")
        if self.action_image_rank is not None and not 0 <= self.action_image_rank <= 1024:
            raise ValueError("finite action-image rank must be in [0, 1024]")
        for word_id in self.action_kernel_word_ids:
            validate_stable_id(word_id, field_name="action_kernel_word_ids")
        require_sorted_unique_strings(
            self.action_kernel_word_ids, field_name="action_kernel_word_ids"
        )
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        unevaluable = (
            self.delivery_qualification is DeliveryQualification.UNEVALUABLE
            or any(
                value is FiniteAxisDisposition.UNEVALUABLE
                for value in (
                    self.constituent_port_materiality,
                    self.simultaneous_defect,
                    self.controlled_sequential_defect,
                    self.repeat_and_longer_word_curvature,
                    self.temporal_cocycle,
                    self.equal_lag_stationarity,
                    self.lie_axis,
                )
            )
            or self.receiver_map is ReceiverMapDisposition.UNEVALUABLE
        )
        if unevaluable and not self.reason_codes:
            raise ValueError("unevaluable finite-response signature requires reason codes")
        if self.action_image_rank is None and not self.reason_codes:
            raise ValueError("missing finite action-image rank requires reason codes")
        if self.delivery_qualification is not DeliveryQualification.QUALIFIED:
            if self.action_image_rank is not None or self.action_kernel_word_ids:
                raise ValueError(
                    "nonqualified delivery cannot support an action-image rank or kernel"
                )
            if not self.reason_codes:
                raise ValueError("nonqualified delivery signature requires reason codes")


@dataclass(frozen=True, slots=True)
class ScalarDoseActionWordApplicabilityCheck(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/scalar-dose-action-word-applicability-check'

    check_id: str
    criterion: ScalarDoseActionWordApplicabilityCriterion
    disposition: ApplicabilityDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is ApplicabilityDisposition.PASS and self.reason_codes:
            raise ValueError("passing scalar-dose action-word applicability check cannot carry reason codes")
        if self.disposition is not ApplicabilityDisposition.PASS and not self.reason_codes:
            raise ValueError("nonpassing scalar-dose action-word applicability check requires reason codes")


@dataclass(frozen=True, slots=True)
class ScalarDoseActionWordProjectionWitness(CanonicalRecord):
    "Proof that one input is exactly compatible with the scalar-dose action-word method, or why it is not."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/scalar-dose-action-word-projection-witness'

    witness_id: str
    source_schema: str
    source_object_id: str
    status: ScalarDoseActionWordSignatureStatus
    checks: tuple[ScalarDoseActionWordApplicabilityCheck, ...]
    response_signature: ResponseAlgebraSignature | None
    parent_receipt_id: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.witness_id, field_name="witness_id")
        validate_schema(self.source_schema)
        validate_stable_id(self.source_object_id, field_name="source_object_id")
        require_sorted_unique_ids(self.checks, attribute="check_id", field_name="checks")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        criteria = tuple(sorted(check.criterion for check in self.checks))
        if criteria != tuple(sorted(ScalarDoseActionWordApplicabilityCriterion)):
            raise ValueError("scalar-dose action-word projection witness must assess every applicability criterion")
        dispositions = {check.disposition for check in self.checks}
        if self.parent_receipt_id is not None:
            validate_stable_id(self.parent_receipt_id, field_name="parent_receipt_id")
        if self.status is ScalarDoseActionWordSignatureStatus.EXACT_SCALAR_DOSE:
            if dispositions != {ApplicabilityDisposition.PASS}:
                raise ValueError("exact scalar-dose action-word projection requires every applicability check to pass")
            if self.response_signature is None or self.parent_receipt_id is not None:
                raise ValueError("exact scalar-dose action-word projection requires a signature and no parent receipt")
            if self.reason_codes:
                raise ValueError("exact scalar-dose action-word projection cannot carry reason codes")
        elif self.status is ScalarDoseActionWordSignatureStatus.RECEIPT_BOUND_SCALAR_DOSE_PARENT:
            if dispositions != {ApplicabilityDisposition.PASS}:
                raise ValueError("parent scalar-dose action-word projection requires every applicability check to pass")
            if self.response_signature is None or self.parent_receipt_id is None:
                raise ValueError("parent scalar-dose action-word projection requires signature and receipt identities")
            if self.reason_codes:
                raise ValueError("parent scalar-dose action-word projection cannot carry reason codes")
        elif self.status is ScalarDoseActionWordSignatureStatus.SCALAR_DOSE_NOT_APPLICABLE:
            if ApplicabilityDisposition.FAIL not in dispositions:
                raise ValueError("scalar-dose action-word nonapplicability requires a failed criterion")
            if self.response_signature is not None or not self.reason_codes:
                raise ValueError("scalar-dose action-word nonapplicability cannot manufacture a signature")
        else:
            if ApplicabilityDisposition.UNEVALUABLE not in dispositions:
                raise ValueError("unevaluable scalar-dose action-word projection requires an unevaluable criterion")
            if self.response_signature is not None or not self.reason_codes:
                raise ValueError("unevaluable scalar-dose action-word projection cannot manufacture a signature")


@dataclass(frozen=True, slots=True)
class OntologyCompatibilityWitness(CanonicalRecord):
    """Map a bath/boundary view into the fixed ``L(D,H,A,R,tau)`` identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/ontology-compatibility-witness'

    witness_id: str
    relation: RelationalIdentity
    bath_denominator_quantity_ids: tuple[str, ...]
    bath_history_quantity_ids: tuple[str, ...]
    thermodynamic_annotation_quantity_ids: tuple[str, ...]
    status: OntologyCompatibilityStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.witness_id, field_name="witness_id")
        for name, values in (
            ("bath_denominator_quantity_ids", self.bath_denominator_quantity_ids),
            ("bath_history_quantity_ids", self.bath_history_quantity_ids),
            (
                "thermodynamic_annotation_quantity_ids",
                self.thermodynamic_annotation_quantity_ids,
            ),
        ):
            require_sorted_unique_strings(values, field_name=name)
            for value in values:
                validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not self.bath_denominator_quantity_ids and not self.bath_history_quantity_ids:
            raise ValueError("ontology witness requires a denominator or history bath view")
        if not set(self.bath_denominator_quantity_ids) <= set(
            self.relation.denominator_quantity_ids
        ):
            raise ValueError("bath denominator view is outside relational D")
        if not set(self.bath_history_quantity_ids) <= set(self.relation.history_quantity_ids):
            raise ValueError("bath history view is outside relational H")
        law_coordinates = {
            *self.relation.denominator_quantity_ids,
            *self.relation.history_quantity_ids,
            *self.relation.action_quantity_ids,
            *self.relation.receiver_quantity_ids,
        }
        if law_coordinates & set(self.thermodynamic_annotation_quantity_ids):
            raise ValueError("thermodynamic annotations cannot redefine law coordinates")
        if self.status is OntologyCompatibilityStatus.COMPATIBLE and self.reason_codes:
            raise ValueError("compatible ontology witness cannot carry reason codes")
        if self.status is OntologyCompatibilityStatus.UNEVALUABLE and not self.reason_codes:
            raise ValueError("unevaluable ontology witness requires reason codes")


@dataclass(frozen=True, slots=True)
class ThermodynamicRoleBinding(CanonicalRecord):
    """Assign one quantity exactly one action, state, receiver or exchange role."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/thermodynamic-role-binding'

    binding_id: str
    quantity_id: str
    physical_event_id: str
    role: ThermodynamicCoordinateRole

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_stable_id(self.physical_event_id, field_name="physical_event_id")


def validate_thermodynamic_role_bindings(
    bindings: tuple[ThermodynamicRoleBinding, ...],
) -> None:
    """Reject role aliasing while permitting command and transfer observations of one event."""

    require_sorted_unique_ids(bindings, attribute="binding_id", field_name="bindings")
    bound_quantities: set[str] = set()
    for binding in bindings:
        if binding.quantity_id in bound_quantities:
            raise ValueError("one quantity cannot be bound twice or counted as exchange twice")
        bound_quantities.add(binding.quantity_id)


@dataclass(frozen=True, slots=True)
class StoredEnergyTermSpec(CanonicalRecord):
    """One final-minus-initial system-storage term, never a boundary transfer."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/stored-energy-term-spec'

    term_id: str
    system_id: str
    quantity_id: str
    native_unit: str
    constitutive_model_id: str
    calibration_id: str
    uncertainty_contract_id: str
    discrepancy_contract_id: str
    directly_observed: bool
    sign_convention: ThermodynamicSignConvention

    def __post_init__(self) -> None:
        for name, value in (
            ("term_id", self.term_id),
            ("system_id", self.system_id),
            ("quantity_id", self.quantity_id),
            ("constitutive_model_id", self.constitutive_model_id),
            ("calibration_id", self.calibration_id),
            ("uncertainty_contract_id", self.uncertainty_contract_id),
            ("discrepancy_contract_id", self.discrepancy_contract_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.sign_convention is not ThermodynamicSignConvention.FINAL_MINUS_INITIAL:
            raise ValueError("stored energy must use final-minus-initial sign")


@dataclass(frozen=True, slots=True)
class ThermodynamicTermSpec(CanonicalRecord):
    """One calibrated boundary-energy transfer with an explicit sign transform."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/thermodynamic-term-spec'

    term_id: str
    system_id: str
    quantity_id: str
    port_owner_id: str
    port_id: str
    interface_id: str | None
    physical_event_id: str
    transfer_mode: EnergyTransferMode
    native_unit: str
    source_native_positive_direction: str
    canonical_sign_convention: ThermodynamicSignConvention
    canonical_sign_multiplier: int
    integration_method_id: str
    calibration_id: str
    uncertainty_contract_id: str
    validity_contract_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("term_id", self.term_id),
            ("system_id", self.system_id),
            ("quantity_id", self.quantity_id),
            ("port_owner_id", self.port_owner_id),
            ("port_id", self.port_id),
            ("physical_event_id", self.physical_event_id),
            ("integration_method_id", self.integration_method_id),
            ("calibration_id", self.calibration_id),
            ("uncertainty_contract_id", self.uncertainty_contract_id),
            ("validity_contract_id", self.validity_contract_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.interface_id is not None:
            validate_stable_id(self.interface_id, field_name="interface_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_nonempty(
            self.source_native_positive_direction,
            field_name="source_native_positive_direction",
        )
        if self.canonical_sign_convention is not ThermodynamicSignConvention.INTO_SYSTEM_POSITIVE:
            raise ValueError("boundary transfer must use system-positive canonical sign")
        if self.canonical_sign_multiplier not in {-1, 1}:
            raise ValueError("canonical sign multiplier must be -1 or 1")


@dataclass(frozen=True, slots=True)
class EntropyTermSpec(CanonicalRecord):
    """One entropy-ledger role with all required constitutive references explicit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/entropy-term-spec'

    term_id: str
    system_id: str
    role: EntropyTermRole
    entropy_quantity_id: str
    heat_term_id: str | None
    matter_term_id: str | None
    boundary_temperature_quantity_id: str | None
    composition_quantity_ids: tuple[str, ...]
    constitutive_model_ids: tuple[str, ...]
    assumption_ids: tuple[str, ...]
    uncertainty_contract_id: str
    validity_contract_id: str
    synchronization_id: str
    local_equilibrium_assumed: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("term_id", self.term_id),
            ("system_id", self.system_id),
            ("entropy_quantity_id", self.entropy_quantity_id),
            ("uncertainty_contract_id", self.uncertainty_contract_id),
            ("validity_contract_id", self.validity_contract_id),
            ("synchronization_id", self.synchronization_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, optional_id in (
            ("heat_term_id", self.heat_term_id),
            ("matter_term_id", self.matter_term_id),
            ("boundary_temperature_quantity_id", self.boundary_temperature_quantity_id),
        ):
            if optional_id is not None:
                validate_stable_id(optional_id, field_name=name)
        for name, values in (
            ("composition_quantity_ids", self.composition_quantity_ids),
            ("constitutive_model_ids", self.constitutive_model_ids),
            ("assumption_ids", self.assumption_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
            for value in values:
                validate_stable_id(value, field_name=name)
        if not self.assumption_ids:
            raise ValueError("entropy term requires explicit constitutive assumptions")
        if self.role is EntropyTermRole.HEAT_OVER_BOUNDARY_TEMPERATURE:
            if self.heat_term_id is None or self.boundary_temperature_quantity_id is None:
                raise ValueError("heat entropy transfer requires heat and boundary temperature")
        elif self.role is EntropyTermRole.MATTER_ENTROPY_TRANSFER:
            if (
                self.matter_term_id is None
                or not self.composition_quantity_ids
                or not self.constitutive_model_ids
            ):
                raise ValueError("matter entropy transfer requires matter, composition and a model")
        elif self.role is EntropyTermRole.STATE_ENTROPY:
            if not self.constitutive_model_ids:
                raise ValueError("state entropy requires a constitutive model")
        elif not self.constitutive_model_ids or not self.assumption_ids:
            raise ValueError("declared reaction/other entropy requires model and assumptions")


@dataclass(frozen=True, slots=True)
class ThermodynamicLedgerObservation(CanonicalRecord):
    """One signed observed or explicitly missing ledger term for one episode."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/thermodynamic-ledger-observation'

    observation_id: str
    system_id: str
    independent_unit_id: str
    episode_id: str
    word_id: str
    term_id: str
    physical_event_id: str
    coordinate_role: ThermodynamicCoordinateRole
    estimand_kind: ExchangeEstimandKind
    source_native_value: Decimal | None
    source_native_unit: str
    source_native_positive_direction: str
    canonical_value: Decimal | None
    canonical_unit: str
    canonical_sign_convention: ThermodynamicSignConvention
    canonical_sign_multiplier: int
    sign_transform_rule_id: str
    uncertainty: Decimal | None
    matched_identity_observation_id: str | None
    decomposition_model_id: str | None
    status: ThermodynamicObservationStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("observation_id", self.observation_id),
            ("system_id", self.system_id),
            ("independent_unit_id", self.independent_unit_id),
            ("episode_id", self.episode_id),
            ("word_id", self.word_id),
            ("term_id", self.term_id),
            ("physical_event_id", self.physical_event_id),
            ("sign_transform_rule_id", self.sign_transform_rule_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, optional_id in (
            ("matched_identity_observation_id", self.matched_identity_observation_id),
            ("decomposition_model_id", self.decomposition_model_id),
        ):
            if optional_id is not None:
                validate_stable_id(optional_id, field_name=name)
        for name, value in (
            ("source_native_unit", self.source_native_unit),
            ("source_native_positive_direction", self.source_native_positive_direction),
            ("canonical_unit", self.canonical_unit),
        ):
            validate_nonempty(value, field_name=name)
        storage_roles = {
            ThermodynamicCoordinateRole.STORED_ENERGY,
            ThermodynamicCoordinateRole.STATE_ENTROPY,
        }
        ledger_roles = storage_roles | {
            ThermodynamicCoordinateRole.WORK_TRANSFER,
            ThermodynamicCoordinateRole.HEAT_TRANSFER,
            ThermodynamicCoordinateRole.HEAT_ENTROPY_TRANSFER,
            ThermodynamicCoordinateRole.MATTER_TRANSFER,
            ThermodynamicCoordinateRole.MATTER_ENTROPY_TRANSFER,
            ThermodynamicCoordinateRole.OTHER_BOUNDARY_TRANSFER,
        }
        if self.coordinate_role not in ledger_roles:
            raise ValueError("coordinate role is not an integrated thermodynamic ledger term")
        expected_sign = (
            ThermodynamicSignConvention.FINAL_MINUS_INITIAL
            if self.coordinate_role in storage_roles
            else ThermodynamicSignConvention.INTO_SYSTEM_POSITIVE
        )
        if self.canonical_sign_convention is not expected_sign:
            raise ValueError("ledger sign convention differs from its coordinate role")
        expected_estimand = {
            ThermodynamicCoordinateRole.STORED_ENERGY: (
                ExchangeEstimandKind.ABSOLUTE_STORAGE_CHANGE
            ),
            ThermodynamicCoordinateRole.STATE_ENTROPY: (
                ExchangeEstimandKind.ABSOLUTE_STATE_ENTROPY_CHANGE
            ),
        }.get(self.coordinate_role)
        if expected_estimand is not None and self.estimand_kind is not expected_estimand:
            raise ValueError("storage/state-entropy role has the wrong absolute estimand")
        if expected_estimand is None and self.estimand_kind in {
            ExchangeEstimandKind.ABSOLUTE_STORAGE_CHANGE,
            ExchangeEstimandKind.ABSOLUTE_STATE_ENTROPY_CHANGE,
        }:
            raise ValueError("boundary transfer cannot use a storage-change estimand")
        if self.coordinate_role is ThermodynamicCoordinateRole.ACTION_COMMAND:
            raise ValueError("action command cannot enter the thermodynamic ledger")
        if self.canonical_sign_multiplier not in {-1, 1}:
            raise ValueError("ledger sign multiplier must be -1 or 1")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is ThermodynamicObservationStatus.OBSERVED:
            if self.source_native_value is None or self.canonical_value is None:
                raise ValueError("observed ledger term requires native and canonical values")
            if self.uncertainty is None:
                raise ValueError("observed ledger term requires uncertainty")
            if self.reason_codes:
                raise ValueError("observed ledger term cannot carry failure reasons")
        else:
            if self.source_native_value is not None or self.canonical_value is not None:
                raise ValueError("nonobserved ledger term cannot carry numerical values")
            if not self.reason_codes:
                raise ValueError("nonobserved ledger term requires reasons")
        if self.source_native_value is not None:
            validate_decimal(self.source_native_value, field_name="source_native_value")
        if self.canonical_value is not None:
            validate_decimal(self.canonical_value, field_name="canonical_value")
            if (
                self.source_native_value is not None
                and self.source_native_unit == self.canonical_unit
            ):
                expected = self.source_native_value * self.canonical_sign_multiplier
                if self.canonical_value != expected:
                    raise ValueError(
                        "same-unit canonical value differs from declared sign transform"
                    )
        if self.uncertainty is not None:
            validate_decimal(self.uncertainty, field_name="uncertainty", minimum=Decimal("0"))
        identity_differenced = (
            self.estimand_kind is ExchangeEstimandKind.IDENTITY_DIFFERENCED_EXCHANGE
        )
        if identity_differenced != (self.matched_identity_observation_id is not None):
            raise ValueError("identity-differenced exchange requires exactly one matched identity")
        decomposed = self.estimand_kind in {
            ExchangeEstimandKind.HOUSEKEEPING_EXCHANGE,
            ExchangeEstimandKind.EXCESS_EXCHANGE,
        }
        if decomposed != (self.decomposition_model_id is not None):
            raise ValueError("housekeeping/excess exchange requires a governing decomposition")


@dataclass(frozen=True, slots=True)
class BalanceClosureAssessment(CanonicalRecord):
    """Absolute first-law closure; identity-differenced exchange is never substituted."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/balance-closure-assessment'

    assessment_id: str
    system_id: str
    independent_unit_id: str
    episode_id: str
    word_id: str
    stored_energy_observation_ids: tuple[str, ...]
    boundary_transfer_observation_ids: tuple[str, ...]
    unobserved_term_ids: tuple[str, ...]
    delta_energy: Decimal | None
    total_boundary_exchange: Decimal | None
    residual: Decimal | None
    uncertainty: Decimal | None
    closure_tolerance: Decimal
    energy_unit: str
    status: BalanceClosureStatus
    entropy_eligibility: EntropyEligibilityStatus
    claim_ceiling: ThermodynamicClaimCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("system_id", self.system_id),
            ("independent_unit_id", self.independent_unit_id),
            ("episode_id", self.episode_id),
            ("word_id", self.word_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("stored_energy_observation_ids", self.stored_energy_observation_ids),
            ("boundary_transfer_observation_ids", self.boundary_transfer_observation_ids),
            ("unobserved_term_ids", self.unobserved_term_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if not self.stored_energy_observation_ids and not self.unobserved_term_ids:
            raise ValueError("absolute energy closure requires storage or an explicit missing term")
        validate_decimal(
            self.closure_tolerance,
            field_name="closure_tolerance",
            minimum=Decimal("0"),
        )
        validate_nonempty(self.energy_unit, field_name="energy_unit")
        numerical = (
            self.delta_energy,
            self.total_boundary_exchange,
            self.residual,
            self.uncertainty,
        )
        for numerical_value in numerical:
            if numerical_value is not None:
                validate_decimal(numerical_value, field_name="balance_value")
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError("balance uncertainty cannot be negative")
        complete = (
            self.delta_energy is not None
            and self.total_boundary_exchange is not None
            and self.residual is not None
            and self.uncertainty is not None
        )
        if (
            self.delta_energy is not None
            and self.total_boundary_exchange is not None
            and self.residual is not None
        ):
            expected_residual = self.delta_energy - self.total_boundary_exchange
            if self.residual != expected_residual:
                raise ValueError(
                    "energy residual differs from delta-energy minus boundary exchange"
                )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is BalanceClosureStatus.CLOSED:
            if (
                not complete
                or not self.stored_energy_observation_ids
                or self.unobserved_term_ids
                or self.reason_codes
            ):
                raise ValueError("closed energy balance requires complete observed terms")
            if self.residual is None or abs(self.residual) > self.closure_tolerance:
                raise ValueError("closed energy residual exceeds its tolerance")
            if self.claim_ceiling not in {
                ThermodynamicClaimCeiling.ENERGY_BALANCE_ONLY,
                ThermodynamicClaimCeiling.ENTROPY_PRODUCTION_ELIGIBLE,
            }:
                raise ValueError("closed energy balance has an incompatible claim ceiling")
        else:
            if not self.reason_codes:
                raise ValueError("nonclosed/annular energy balance requires reasons")
            if self.status is BalanceClosureStatus.TERM_UNOBSERVED and not self.unobserved_term_ids:
                raise ValueError("unobserved balance status requires missing term identities")
        if self.claim_ceiling is ThermodynamicClaimCeiling.ENTROPY_PRODUCTION_ELIGIBLE:
            if self.entropy_eligibility is not EntropyEligibilityStatus.ELIGIBLE:
                raise ValueError("entropy claim ceiling requires complete entropy eligibility")
            if self.status is not BalanceClosureStatus.CLOSED:
                raise ValueError("entropy claim ceiling requires closed absolute energy balance")


def validate_balance_observations(
    assessment: BalanceClosureAssessment,
    observations: tuple[ThermodynamicLedgerObservation, ...],
) -> None:
    """Bind absolute storage and total exchange observations to one balance."""

    require_sorted_unique_ids(
        observations,
        attribute="observation_id",
        field_name="observations",
    )
    by_id = {value.observation_id: value for value in observations}
    required_ids = {
        *assessment.stored_energy_observation_ids,
        *assessment.boundary_transfer_observation_ids,
    }
    if set(by_id) != required_ids:
        raise ValueError("balance observations differ from the declared ledger identities")
    if any(
        value.system_id != assessment.system_id
        or value.independent_unit_id != assessment.independent_unit_id
        or value.episode_id != assessment.episode_id
        or value.word_id != assessment.word_id
        for value in observations
    ):
        raise ValueError("balance observation identity differs from its assessment")
    storage = tuple(by_id[value] for value in assessment.stored_energy_observation_ids)
    boundary = tuple(by_id[value] for value in assessment.boundary_transfer_observation_ids)
    if any(
        value.coordinate_role is not ThermodynamicCoordinateRole.STORED_ENERGY
        or value.estimand_kind is not ExchangeEstimandKind.ABSOLUTE_STORAGE_CHANGE
        for value in storage
    ):
        raise ValueError("absolute energy balance storage uses a nonstorage estimand")
    permitted_boundary_roles = {
        ThermodynamicCoordinateRole.WORK_TRANSFER,
        ThermodynamicCoordinateRole.HEAT_TRANSFER,
        ThermodynamicCoordinateRole.MATTER_TRANSFER,
        ThermodynamicCoordinateRole.OTHER_BOUNDARY_TRANSFER,
    }
    if any(
        value.coordinate_role not in permitted_boundary_roles
        or value.estimand_kind is not ExchangeEstimandKind.TOTAL_EXCHANGE
        for value in boundary
    ):
        raise ValueError(
            "absolute energy balance cannot substitute differenced/decomposed exchange"
        )
    event_ids = tuple(value.physical_event_id for value in boundary)
    if len(set(event_ids)) != len(event_ids):
        raise ValueError("absolute energy balance counts one physical event more than once")
    if all(value.canonical_value is not None for value in storage):
        observed_delta = sum(
            (value.canonical_value for value in storage if value.canonical_value is not None),
            start=Decimal("0"),
        )
        if assessment.delta_energy != observed_delta:
            raise ValueError("balance delta energy differs from its storage observations")
    if all(value.canonical_value is not None for value in boundary):
        observed_exchange = sum(
            (value.canonical_value for value in boundary if value.canonical_value is not None),
            start=Decimal("0"),
        )
        if assessment.total_boundary_exchange != observed_exchange:
            raise ValueError("balance exchange differs from its boundary observations")


@dataclass(frozen=True, slots=True)
class ReturnQualification(CanonicalRecord):
    """State-and-bath return at a predeclared recovery horizon."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/return-qualification'

    qualification_id: str
    system_id: str
    independent_unit_id: str
    episode_id: str
    word_id: str
    recovery_horizon_id: str
    recovery_horizon_predeclared: bool
    state_residuals: tuple[NamedDecimal, ...]
    bath_residuals: tuple[NamedDecimal, ...]
    component_tolerances: tuple[NamedDecimal, ...]
    component_uncertainties: tuple[NamedDecimal, ...]
    support_valid: bool
    delivery_valid: bool
    interlock_clear: bool
    data_complete: bool
    balance_assessment_id: str
    balance_status: BalanceClosureStatus
    unobserved_coordinate_ids: tuple[str, ...]
    status: ReturnQualificationStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("qualification_id", self.qualification_id),
            ("system_id", self.system_id),
            ("independent_unit_id", self.independent_unit_id),
            ("episode_id", self.episode_id),
            ("word_id", self.word_id),
            ("recovery_horizon_id", self.recovery_horizon_id),
            ("balance_assessment_id", self.balance_assessment_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, values in (
            ("state_residuals", self.state_residuals),
            ("bath_residuals", self.bath_residuals),
            ("component_tolerances", self.component_tolerances),
            ("component_uncertainties", self.component_uncertainties),
        ):
            require_sorted_unique_ids(values, attribute="value_id", field_name=name)
        residuals = (*self.state_residuals, *self.bath_residuals)
        if not residuals:
            raise ValueError("return qualification requires state or bath coordinates")
        residual_by_id = {value.value_id: value for value in residuals}
        if len(residual_by_id) != len(residuals):
            raise ValueError("state and bath return coordinates cannot alias")
        tolerance_by_id = {value.value_id: value for value in self.component_tolerances}
        uncertainty_by_id = {value.value_id: value for value in self.component_uncertainties}
        if set(tolerance_by_id) != set(residual_by_id):
            raise ValueError("return tolerances must match state and bath coordinates")
        if set(uncertainty_by_id) != set(residual_by_id):
            raise ValueError("return uncertainties must match state and bath coordinates")
        for coordinate_id, residual in residual_by_id.items():
            tolerance = tolerance_by_id[coordinate_id]
            uncertainty = uncertainty_by_id[coordinate_id]
            if not residual.unit == tolerance.unit == uncertainty.unit:
                raise ValueError("return residual/tolerance/uncertainty native units differ")
            if tolerance.value < 0 or uncertainty.value < 0:
                raise ValueError("return tolerance and uncertainty must be nonnegative")
        require_sorted_unique_strings(
            self.unobserved_coordinate_ids,
            field_name="unobserved_coordinate_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        component_pass = all(
            abs(residual.value) + uncertainty_by_id[coordinate_id].value
            <= tolerance_by_id[coordinate_id].value
            for coordinate_id, residual in residual_by_id.items()
        )
        gates_pass = (
            self.recovery_horizon_predeclared
            and self.support_valid
            and self.delivery_valid
            and self.interlock_clear
            and self.data_complete
            and self.balance_status is BalanceClosureStatus.CLOSED
            and not self.unobserved_coordinate_ids
            and component_pass
        )
        if self.status is ReturnQualificationStatus.RETURN_QUALIFIED:
            if not gates_pass or self.reason_codes:
                raise ValueError("return-qualified cycle fails a required state/bath gate")
        elif not self.reason_codes:
            raise ValueError("nonreturning/annular/unevaluable path requires reasons")


@dataclass(frozen=True, slots=True)
class ThermodynamicResponseSignature(CanonicalRecord):
    "Composite response/thermodynamic signature without scalar-dose action-word inheritance."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/thermodynamic-response-signature'

    signature_id: str
    relation_id: str
    response_signature: ResponseAlgebraSignature | None
    response_signature_status: ScalarDoseActionWordSignatureStatus
    scalar_dose_projection_witness: ScalarDoseActionWordProjectionWitness | None
    finite_response_signature: FiniteResponseSignature
    evidence_world_id: str
    world_kind: WorldKind
    denominator_type_id: str
    bath_type_id: str
    delivery_qualification: DeliveryQualification
    thermodynamic_coordinate_sufficiency: EntropyEligibilityStatus
    return_qualification: ReturnQualificationStatus
    energy_balance_closure: BalanceClosureStatus
    entropy_claim_ceiling: ThermodynamicClaimCeiling
    loop_exchange_observation_ids: tuple[str, ...]
    receiver_faithfulness: ReceiverMapDisposition
    temporal_composition: FiniteAxisDisposition
    equal_lag_stationarity: FiniteAxisDisposition
    action_quotient: FiniteAxisDisposition
    resolution_class: FiniteAxisDisposition
    explicit_unevaluable_axes: tuple[CompositeSignatureAxis, ...]
    evidence_link_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("signature_id", self.signature_id),
            ("relation_id", self.relation_id),
            ("evidence_world_id", self.evidence_world_id),
            ("denominator_type_id", self.denominator_type_id),
            ("bath_type_id", self.bath_type_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.finite_response_signature.relation_id != self.relation_id:
            raise ValueError("finite-response signature belongs to another relation")
        has_scalar_dose_signature = self.response_signature_status in {
            ScalarDoseActionWordSignatureStatus.EXACT_SCALAR_DOSE,
            ScalarDoseActionWordSignatureStatus.RECEIPT_BOUND_SCALAR_DOSE_PARENT,
        }
        if has_scalar_dose_signature:
            if self.response_signature is None or self.scalar_dose_projection_witness is None:
                raise ValueError("eligible scalar-dose action-word response requires signature and projection witness")
            if self.scalar_dose_projection_witness.status is not self.response_signature_status:
                raise ValueError("scalar-dose action-word projection status differs from composite signature")
            if self.scalar_dose_projection_witness.response_signature != self.response_signature:
                raise ValueError("scalar-dose action-word projection signature differs from composite signature")
        elif self.response_signature is not None:
            raise ValueError("inapplicable/unevaluable scalar-dose action-word response cannot be populated")
        elif self.response_signature_status is ScalarDoseActionWordSignatureStatus.SCALAR_DOSE_NOT_APPLICABLE:
            if (
                self.scalar_dose_projection_witness is None
                or self.scalar_dose_projection_witness.status is not self.response_signature_status
            ):
                raise ValueError("scalar-dose action-word nonapplicability requires its exact projection witness")
        elif (
            self.scalar_dose_projection_witness is not None
            and self.scalar_dose_projection_witness.status is not self.response_signature_status
        ):
            raise ValueError("scalar-dose action-word unevaluable projection status differs")
        if self.delivery_qualification is not self.finite_response_signature.delivery_qualification:
            raise ValueError("delivery qualification differs from finite-response signature")
        for name, values in (
            ("loop_exchange_observation_ids", self.loop_exchange_observation_ids),
            ("evidence_link_ids", self.evidence_link_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        axes = tuple(sorted(self.explicit_unevaluable_axes, key=lambda value: value.value))
        if axes != self.explicit_unevaluable_axes or len(set(axes)) != len(axes):
            raise ValueError("explicit unevaluable axes must be sorted and unique")
        if self.entropy_claim_ceiling is ThermodynamicClaimCeiling.ENTROPY_PRODUCTION_ELIGIBLE:
            if self.thermodynamic_coordinate_sufficiency is not EntropyEligibilityStatus.ELIGIBLE:
                raise ValueError("entropy ceiling requires eligible thermodynamic coordinates")
            if self.energy_balance_closure is not BalanceClosureStatus.CLOSED:
                raise ValueError("entropy ceiling requires closed absolute energy balance")
        ceiling_sufficiency = {
            ThermodynamicClaimCeiling.ENERGY_BALANCE_ONLY: (
                EntropyEligibilityStatus.ENERGY_BALANCE_ONLY
            ),
            ThermodynamicClaimCeiling.DISSIPATIVE_CLOSURE_PROXY_ONLY: (
                EntropyEligibilityStatus.DISSIPATIVE_CLOSURE_PROXY_ONLY
            ),
            ThermodynamicClaimCeiling.THERMODYNAMIC_LEDGER_PARTIAL: (
                EntropyEligibilityStatus.LEDGER_PARTIAL
            ),
            ThermodynamicClaimCeiling.THERMODYNAMIC_INTERPRETATION_UNEVALUABLE: (
                EntropyEligibilityStatus.UNEVALUABLE
            ),
        }.get(self.entropy_claim_ceiling)
        if (
            ceiling_sufficiency is not None
            and self.thermodynamic_coordinate_sufficiency is not ceiling_sufficiency
        ):
            raise ValueError("thermodynamic claim ceiling differs from coordinate sufficiency")
        if (
            self.entropy_claim_ceiling is ThermodynamicClaimCeiling.ENERGY_BALANCE_ONLY
            and self.energy_balance_closure is not BalanceClosureStatus.CLOSED
        ):
            raise ValueError("energy-balance-only ceiling requires closed absolute balance")
        if self.return_qualification is not ReturnQualificationStatus.RETURN_QUALIFIED:
            if self.loop_exchange_observation_ids and not self.reason_codes:
                raise ValueError("open-path loop exchange requires an explicit noncycle reason")
        if self.explicit_unevaluable_axes and not self.reason_codes:
            raise ValueError("explicit unevaluable axes require reasons")


_AXIS_ALLOWED_VALUES: dict[CompositeSignatureAxis, frozenset[str]] = {
    CompositeSignatureAxis.DELIVERED_ACTION_OBSERVABILITY: frozenset(
        value.value for value in DeliveryQualification
    ),
    CompositeSignatureAxis.CONSTITUENT_PORT_MATERIALITY: frozenset(
        value.value for value in FiniteAxisDisposition
    ),
    CompositeSignatureAxis.ACTION_IMAGE_QUOTIENT: frozenset(
        value.value for value in FiniteAxisDisposition
    ),
    CompositeSignatureAxis.SIMULTANEOUS_FINITE_DEFECT: frozenset(
        value.value for value in FiniteAxisDisposition
    ),
    CompositeSignatureAxis.CONTROLLED_SEQUENTIAL_FINITE_DEFECT: frozenset(
        value.value for value in FiniteAxisDisposition
    ),
    CompositeSignatureAxis.REPEAT_AND_LONGER_WORD_CURVATURE: frozenset(
        value.value for value in FiniteAxisDisposition
    ),
    CompositeSignatureAxis.TEMPORAL_COCYCLE_CLOSURE: frozenset(
        value.value for value in FiniteAxisDisposition
    ),
    CompositeSignatureAxis.EQUAL_LAG_STATIONARITY: frozenset(
        value.value for value in FiniteAxisDisposition
    ),
    CompositeSignatureAxis.RECEIVER_NATURALITY_AND_FAITHFULNESS: frozenset(
        value.value for value in ReceiverMapDisposition
    ),
    CompositeSignatureAxis.PREPARATION_STATE_AND_BATH_RETURN: frozenset(
        value.value for value in ReturnQualificationStatus
    ),
    CompositeSignatureAxis.ABSOLUTE_ENERGY_CLOSURE: frozenset(
        value.value for value in BalanceClosureStatus
    ),
    CompositeSignatureAxis.ENTROPY_PRODUCTION_ELIGIBILITY: frozenset(
        value.value for value in EntropyEligibilityStatus
    ),
    CompositeSignatureAxis.DENOMINATOR_REGIME_STABILITY_AND_AGEING: frozenset(
        value.value for value in FiniteAxisDisposition
    ),
}


def _validate_axis_values(
    values: tuple[tuple[CompositeSignatureAxis, tuple[str, ...]], ...],
    *,
    field_name: str,
) -> None:
    axes = tuple(axis for axis, _allowed in values)
    if axes != tuple(CompositeSignatureAxis):
        raise ValueError(f"{field_name} must cover all composite axes in frozen order")
    for axis, allowed in values:
        require_sorted_unique_strings(allowed, field_name=field_name, allow_empty=False)
        if not set(allowed) <= _AXIS_ALLOWED_VALUES[axis]:
            raise ValueError(f"{field_name} contains a disposition outside {axis.value}")


@dataclass(frozen=True, slots=True)
class CompositeStructuralClassDefinition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/composite-structural-class-definition'

    class_id: str
    label: str
    allowed_axis_values: tuple[tuple[CompositeSignatureAxis, tuple[str, ...]], ...]
    allowed_action_image_ranks: tuple[int, ...]
    required_action_kernel_word_ids: tuple[str, ...]
    multiplicity_contract_id: str
    decisive_falsifier_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.class_id, field_name="class_id")
        validate_nonempty(self.label, field_name="label")
        validate_stable_id(self.multiplicity_contract_id, field_name="multiplicity_contract_id")
        _validate_axis_values(self.allowed_axis_values, field_name="allowed_axis_values")
        if tuple(
            sorted(set(self.allowed_action_image_ranks))
        ) != self.allowed_action_image_ranks or any(
            not 0 <= value <= 1024 for value in self.allowed_action_image_ranks
        ):
            raise ValueError("allowed action-image ranks must be sorted, unique and bounded")
        require_sorted_unique_strings(
            self.required_action_kernel_word_ids,
            field_name="required_action_kernel_word_ids",
        )
        action_values = dict(self.allowed_axis_values)[CompositeSignatureAxis.ACTION_IMAGE_QUOTIENT]
        if not self.allowed_action_image_ranks and "UNEVALUABLE" not in action_values:
            raise ValueError("missing action-image rank requires an unevaluable quotient region")
        require_sorted_unique_strings(
            self.decisive_falsifier_ids,
            field_name="decisive_falsifier_ids",
            allow_empty=False,
        )


def _composite_assessment_status(
    definition: CompositeStructuralClassDefinition,
    observed: tuple[tuple[CompositeSignatureAxis, str], ...],
    observed_action_image_rank: int | None,
    observed_action_kernel_word_ids: tuple[str, ...],
) -> ThermodynamicResponseTerminalStatus:
    allowed = dict(definition.allowed_axis_values)
    mismatches = tuple(value for axis, value in observed if value not in allowed[axis])
    rank_matches = (
        observed_action_image_rank in definition.allowed_action_image_ranks
        if observed_action_image_rank is not None
        else not definition.allowed_action_image_ranks
    )
    kernel_matches = observed_action_kernel_word_ids == definition.required_action_kernel_word_ids
    if not mismatches and rank_matches and kernel_matches:
        return ThermodynamicResponseTerminalStatus.SUPPORTED
    if observed_action_image_rank is None and definition.allowed_action_image_ranks:
        return ThermodynamicResponseTerminalStatus.UNEVALUABLE
    if any(value == "UNEVALUABLE" for value in mismatches):
        return ThermodynamicResponseTerminalStatus.UNEVALUABLE
    if any(value == "ANNULAR" for value in mismatches):
        return ThermodynamicResponseTerminalStatus.ANNULAR
    if any(value == "INVALID" for value in mismatches):
        return ThermodynamicResponseTerminalStatus.STOPPED
    return ThermodynamicResponseTerminalStatus.NOT_SUPPORTED


@dataclass(frozen=True, slots=True)
class CompositeStructuralClassAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/composite-structural-class-assessment'

    assessment_id: str
    signature_id: str
    definition: CompositeStructuralClassDefinition
    observed_axis_values: tuple[tuple[CompositeSignatureAxis, str], ...]
    observed_action_image_rank: int | None
    observed_action_kernel_word_ids: tuple[str, ...]
    status: ThermodynamicResponseTerminalStatus
    evidence_link_ids: tuple[str, ...]
    decisive_falsifier_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.signature_id, field_name="signature_id")
        observed = tuple((axis, (value,)) for axis, value in self.observed_axis_values)
        _validate_axis_values(observed, field_name="observed_axis_values")
        if (
            self.observed_action_image_rank is not None
            and not 0 <= self.observed_action_image_rank <= 1024
        ):
            raise ValueError("observed action-image rank must be in [0, 1024]")
        require_sorted_unique_strings(
            self.observed_action_kernel_word_ids,
            field_name="observed_action_kernel_word_ids",
        )
        expected = _composite_assessment_status(
            self.definition,
            self.observed_axis_values,
            self.observed_action_image_rank,
            self.observed_action_kernel_word_ids,
        )
        if self.status is not expected:
            raise ValueError("composite structural-class status differs from exact conjunction")
        for name, values in (
            ("evidence_link_ids", self.evidence_link_ids),
            ("decisive_falsifier_ids", self.decisive_falsifier_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.status is ThermodynamicResponseTerminalStatus.SUPPORTED:
            if self.decisive_falsifier_ids or self.reason_codes:
                raise ValueError("supported composite class cannot carry failures")
        elif not self.reason_codes:
            raise ValueError("nonsupported composite class requires reasons")


@dataclass(frozen=True, slots=True)
class StructuralTransportWitness(CanonicalRecord):
    """Typed, nonpooling commuting evidence between two registered media."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/structural-transport-witness'

    witness_id: str
    source_signature_id: str
    target_signature_id: str
    source_system_id: str
    target_system_id: str
    source_world_kind: WorldKind
    target_world_kind: WorldKind
    action_role_map: tuple[tuple[str, str], ...]
    horizon_map: tuple[tuple[str, str], ...]
    receiver_map: tuple[tuple[str, str], ...]
    thermodynamic_role_map: tuple[tuple[str, str], ...]
    common_support_object_map: tuple[tuple[str, str], ...]
    axis_residuals: tuple[
        tuple[CompositeSignatureAxis, Decimal, Decimal, Decimal, str, FiniteAxisDisposition],
        ...,
    ]
    faithfulness_limit_ids: tuple[str, ...]
    decisive_counterexample_ids: tuple[str, ...]
    native_numeric_pooling_performed: bool
    status: TransportDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("witness_id", self.witness_id),
            ("source_signature_id", self.source_signature_id),
            ("target_signature_id", self.target_signature_id),
            ("source_system_id", self.source_system_id),
            ("target_system_id", self.target_system_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.source_system_id == self.target_system_id:
            raise ValueError("structural transport requires two distinct media")
        for name, mappings in (
            ("action_role_map", self.action_role_map),
            ("horizon_map", self.horizon_map),
            ("receiver_map", self.receiver_map),
            ("thermodynamic_role_map", self.thermodynamic_role_map),
            ("common_support_object_map", self.common_support_object_map),
        ):
            if not mappings or tuple(sorted(set(mappings))) != mappings:
                raise ValueError(f"{name} must be nonempty, sorted and unique")
            for source_id, target_id in mappings:
                validate_stable_id(source_id, field_name=f"{name}.source")
                validate_stable_id(target_id, field_name=f"{name}.target")
        axes = tuple(value[0] for value in self.axis_residuals)
        if not axes or tuple(sorted(set(axes), key=lambda value: value.value)) != axes:
            raise ValueError("transport axes must be nonempty, sorted and unique")
        for axis, residual, uncertainty, tolerance, unit, disposition in self.axis_residuals:
            validate_decimal(residual, field_name=f"{axis.value}.residual", minimum=Decimal("0"))
            validate_decimal(
                uncertainty,
                field_name=f"{axis.value}.uncertainty",
                minimum=Decimal("0"),
            )
            validate_decimal(
                tolerance,
                field_name=f"{axis.value}.tolerance",
                minimum=Decimal("0"),
            )
            validate_nonempty(unit, field_name=f"{axis.value}.unit")
            if disposition is FiniteAxisDisposition.EQUIVALENT:
                if residual + uncertainty > tolerance:
                    raise ValueError("equivalent transport residual exceeds native tolerance")
        for name, values in (
            ("faithfulness_limit_ids", self.faithfulness_limit_ids),
            ("decisive_counterexample_ids", self.decisive_counterexample_ids),
            ("reason_codes", self.reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.native_numeric_pooling_performed:
            raise ValueError("structural transport forbids pooled native numerics")
        if self.status is TransportDisposition.COMMUTES_AT_TESTED_RESOLUTION:
            if self.decisive_counterexample_ids or self.reason_codes:
                raise ValueError("commuting transport cannot carry counterexamples/failures")
            if any(
                value[-1] is not FiniteAxisDisposition.EQUIVALENT for value in self.axis_residuals
            ):
                raise ValueError("commuting transport requires every tested axis to pass")
        elif not self.reason_codes:
            raise ValueError("partial/failed/unevaluable transport requires reasons")


def validate_thermodynamic_system_contract(
    system: SystemSpec,
    *,
    boundary_terms: tuple[ThermodynamicTermSpec, ...],
    stored_energy_terms: tuple[StoredEnergyTermSpec, ...],
    entropy_terms: tuple[EntropyTermSpec, ...],
) -> None:
    """Bind thermodynamic roles to registered quantities/ports/interfaces."""

    require_sorted_unique_ids(boundary_terms, attribute="term_id", field_name="boundary_terms")
    require_sorted_unique_ids(
        stored_energy_terms,
        attribute="term_id",
        field_name="stored_energy_terms",
    )
    require_sorted_unique_ids(entropy_terms, attribute="term_id", field_name="entropy_terms")
    quantities = {value.quantity_id: value for value in system.quantities}
    owner_ports = {(system.system_id, value.port_id): value for value in system.ports}
    for component in system.components:
        for port in component.ports:
            owner_ports[(component.component_id, port.port_id)] = port
    interfaces = {value.interface_id: value for value in system.interfaces}
    boundary_quantity_ids: set[str] = set()
    physical_event_ids: set[str] = set()
    for boundary_term in boundary_terms:
        if boundary_term.system_id != system.system_id:
            raise ValueError("boundary term belongs to another system")
        quantity = quantities.get(boundary_term.quantity_id)
        selected_port = owner_ports.get((boundary_term.port_owner_id, boundary_term.port_id))
        if (
            quantity is None
            or selected_port is None
            or selected_port.quantity_id != boundary_term.quantity_id
        ):
            raise ValueError("boundary term does not bind a registered quantity/port")
        if (
            quantity.kind is QuantityKind.ACTION
            or boundary_term.quantity_id in system.relation.action_quantity_ids
        ):
            raise ValueError("action command cannot satisfy a thermodynamic exchange role")
        if selected_port.balance_role is not BalanceRole.ENERGY:
            raise ValueError("thermodynamic energy term requires an energy-balance port")
        if quantity.native_unit != boundary_term.native_unit:
            raise ValueError("boundary term unit differs from its registered quantity")
        if boundary_term.interface_id is not None:
            interface = interfaces.get(boundary_term.interface_id)
            if interface is None or interface.balance_role is not BalanceRole.ENERGY:
                raise ValueError("boundary term interface is absent or not energy balanced")
        if boundary_term.physical_event_id in physical_event_ids:
            raise ValueError("one physical exchange event cannot be counted twice")
        physical_event_ids.add(boundary_term.physical_event_id)
        boundary_quantity_ids.add(boundary_term.quantity_id)
    for storage_term in stored_energy_terms:
        if storage_term.system_id != system.system_id:
            raise ValueError("stored-energy term belongs to another system")
        quantity = quantities.get(storage_term.quantity_id)
        if quantity is None or quantity.native_unit != storage_term.native_unit:
            raise ValueError("stored-energy term does not bind a registered native quantity")
        if (
            quantity.kind is QuantityKind.ACTION
            or storage_term.quantity_id in boundary_quantity_ids
        ):
            raise ValueError("stored energy must remain disjoint from action/boundary transfer")
    boundary_term_ids = {value.term_id for value in boundary_terms}
    for entropy_term in entropy_terms:
        if entropy_term.system_id != system.system_id:
            raise ValueError("entropy term belongs to another system")
        referenced_ids = {
            entropy_term.entropy_quantity_id,
            *entropy_term.composition_quantity_ids,
            *(
                ()
                if entropy_term.boundary_temperature_quantity_id is None
                else (entropy_term.boundary_temperature_quantity_id,)
            ),
        }
        if not referenced_ids <= set(quantities):
            raise ValueError("entropy term references an unregistered quantity")
        if any(quantities[value].kind is QuantityKind.ACTION for value in referenced_ids):
            raise ValueError("action coordinate cannot satisfy an entropy-ledger role")
        if (
            entropy_term.heat_term_id is not None
            and entropy_term.heat_term_id not in boundary_term_ids
        ):
            raise ValueError("entropy heat term is outside the boundary ledger")
        if (
            entropy_term.matter_term_id is not None
            and entropy_term.matter_term_id not in boundary_term_ids
        ):
            raise ValueError("entropy matter term is outside the boundary ledger")


__all__ = [
    'ActionDelivery',
    'ActionInterval',
    'ActionStageObservation',
    "ApplicabilityDisposition",
    "BalanceClosureAssessment",
    "BalanceClosureStatus",
    "ChronologyConvention",
    "CompositeSignatureAxis",
    "CompositeStructuralClassAssessment",
    "CompositeStructuralClassDefinition",
    'DeliveredDoseComponent',
    "DeliveryQualification",
    "EnergyTransferMode",
    "EntropyTermRole",
    "EntropyTermSpec",
    "EntropyEligibilityStatus",
    "ExchangeEstimandKind",
    "FiniteAxisDisposition",
    'FiniteResponseSignature',
    "FiniteWordMode",
    'FiniteWord',
    "OntologyCompatibilityStatus",
    "OntologyCompatibilityWitness",
    "PartialCompositionStatus",
    'PreparedResponseObject',
    "ReceiverMapDisposition",
    "ReturnQualification",
    "ReturnQualificationStatus",
    "StoredEnergyTermSpec",
    "StructuralTransportWitness",
    "ThermodynamicClaimCeiling",
    "ThermodynamicCoordinateRole",
    "ThermodynamicPromotionLevel",
    "ThermodynamicLedgerObservation",
    "ThermodynamicObservationStatus",
    "ThermodynamicResponseSignature",
    "ThermodynamicResponseTerminalStatus",
    "ThermodynamicRoleBinding",
    "ThermodynamicSignConvention",
    "ThermodynamicTermSpec",
    "TransportDisposition",
    'ScalarDoseActionWordApplicabilityCheck',
    'ScalarDoseActionWordApplicabilityCriterion',
    'ScalarDoseActionWordProjectionWitness',
    'ScalarDoseActionWordSignatureStatus',
    "validate_thermodynamic_role_bindings",
    "validate_thermodynamic_system_contract",
    "validate_balance_observations",
]

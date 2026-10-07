"""Current total causal-prefix and obstruction contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .action_contracts import ActionDeliveryStage, ActionOccurrence, ActionStageEvent, OccurrenceActionWord
from .decoding import decode_canonical_bytes
from .evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from .obligations import ObligationStatus
from .provenance import EvidenceLink, ObjectIdentity
from .references import ExecutableReference, NamedDecimal
from .serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from .systems import RelationalIdentity
from .time import (
    ClockCoordinate,
    ClockProjectionStatus,
    ClockTransport,
    CoordinateOrigin,
)


MAX_CAUSAL_CONTRACT_BYTES = 1024 * 1024


class TemporalPredicateSemantics(StrEnum):
    ALWAYS_PRESERVED_PATH = "ALWAYS_PRESERVED_PATH"
    TERMINAL = "TERMINAL"
    INTERVAL_INTEGRAL = "INTERVAL_INTEGRAL"
    EVENTUALLY_REACHED = "EVENTUALLY_REACHED"
    RECOVERABLE_WINDOW = "RECOVERABLE_WINDOW"


class PredicateDirection(StrEnum):
    AT_LEAST = "AT_LEAST"
    AT_MOST = "AT_MOST"
    WITHIN_ABSOLUTE_TOLERANCE = "WITHIN_ABSOLUTE_TOLERANCE"


class EvidenceEvaluability(StrEnum):
    EVALUABLE = "EVALUABLE"
    UNEVALUABLE = "UNEVALUABLE"


class DeliveredActionValidity(StrEnum):
    VALID = "VALID"
    INVALID = "INVALID"
    UNEVALUABLE = "UNEVALUABLE"


class CausalConeExistence(StrEnum):
    PRESENT = "PRESENT"
    MISSING = "MISSING"
    UNEVALUABLE = "UNEVALUABLE"


class ResponseDirectionStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"


class PrefixSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    UNEVALUABLE = "UNEVALUABLE"


class NumericalViewAgreement(StrEnum):
    AGREED = "AGREED"
    DISAGREED = "DISAGREED"
    UNEVALUABLE = "UNEVALUABLE"


class CausalObstructionKind(StrEnum):
    EVIDENCE_UNEVALUABLE = "EVIDENCE_UNEVALUABLE"
    ACTION_DELIVERY_INVALID = "ACTION_DELIVERY_INVALID"
    ACTION_DELIVERY_UNEVALUABLE = "ACTION_DELIVERY_UNEVALUABLE"
    CLOCK_TRANSPORT_UNAVAILABLE = "CLOCK_TRANSPORT_UNAVAILABLE"
    CAUSAL_CONE_MISSING = "CAUSAL_CONE_MISSING"
    CAUSAL_CONE_UNEVALUABLE = "CAUSAL_CONE_UNEVALUABLE"
    RESPONSE_DIRECTION_OPPOSED = "RESPONSE_DIRECTION_OPPOSED"
    RESPONSE_DIRECTION_UNEVALUABLE = "RESPONSE_DIRECTION_UNEVALUABLE"
    PREFIX_OUTSIDE_SUPPORT = "PREFIX_OUTSIDE_SUPPORT"
    PREFIX_SUPPORT_UNEVALUABLE = "PREFIX_SUPPORT_UNEVALUABLE"
    TEMPORAL_OBLIGATION_FAILED_BEFORE_INFLUENCE = "TEMPORAL_OBLIGATION_FAILED_BEFORE_INFLUENCE"
    TEMPORAL_OBLIGATION_UNEVALUABLE = "TEMPORAL_OBLIGATION_UNEVALUABLE"
    NUMERICAL_VIEW_DISAGREEMENT = "NUMERICAL_VIEW_DISAGREEMENT"
    NUMERICAL_VIEW_UNEVALUABLE = "NUMERICAL_VIEW_UNEVALUABLE"


_UNEVALUABLE_OBSTRUCTIONS = frozenset(
    {
        CausalObstructionKind.EVIDENCE_UNEVALUABLE,
        CausalObstructionKind.ACTION_DELIVERY_UNEVALUABLE,
        CausalObstructionKind.CLOCK_TRANSPORT_UNAVAILABLE,
        CausalObstructionKind.CAUSAL_CONE_UNEVALUABLE,
        CausalObstructionKind.RESPONSE_DIRECTION_UNEVALUABLE,
        CausalObstructionKind.PREFIX_SUPPORT_UNEVALUABLE,
        CausalObstructionKind.TEMPORAL_OBLIGATION_UNEVALUABLE,
        CausalObstructionKind.NUMERICAL_VIEW_UNEVALUABLE,
    }
)


class CausalCompositionDisposition(StrEnum):
    DEFINED = "DEFINED"
    BLOCKED = "BLOCKED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ReceiverInterval(CanonicalRecord):
    """Closed interval in one fully identified receiver clock."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/receiver-interval'

    start: ClockCoordinate
    end: ClockCoordinate

    def __post_init__(self) -> None:
        if not self.same_domain(self.start, self.end):
            raise ValueError("receiver interval endpoints use different clock semantics")
        if self.end.coordinate < self.start.coordinate:
            raise ValueError("receiver interval is reversed")

    @staticmethod
    def same_domain(first: ClockCoordinate, second: ClockCoordinate) -> bool:
        return (
            first.clock_id == second.clock_id
            and first.time_unit == second.time_unit
            and first.coordinate_frame == second.coordinate_frame
            and first.origin is second.origin
        )

    def contains(self, coordinate: ClockCoordinate) -> bool:
        return self.same_domain(self.start, coordinate) and (
            self.start.coordinate <= coordinate.coordinate <= self.end.coordinate
        )

    def covers(self, other: ReceiverInterval) -> bool:
        return self.same_domain(self.start, other.start) and (
            self.start.coordinate <= other.start.coordinate
            and self.end.coordinate >= other.end.coordinate
        )


@dataclass(frozen=True, slots=True)
class TemporalPredicateAssessment(CanonicalRecord):
    """Evidence-derived evaluation over a semantics-specific protected domain."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/temporal-predicate-assessment'

    obligation_id: str
    semantics: TemporalPredicateSemantics
    receiver_id: str
    direction: PredicateDirection
    criterion: NamedDecimal
    protected_domain: ReceiverInterval
    evaluated_domain: ReceiverInterval
    status: ObligationStatus
    first_failure: ClockCoordinate | None
    satisfaction_coordinate: ClockCoordinate | None
    required_numerical_view_ids: tuple[str, ...]
    evaluator: ExecutableReference
    physical_independent_unit_id: str
    physical_independent_unit_count: int
    evidence_links: tuple[EvidenceLink, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("obligation_id", self.obligation_id),
            ("receiver_id", self.receiver_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.required_numerical_view_ids,
            field_name="required_numerical_view_ids",
            allow_empty=False,
        )
        for view_id in self.required_numerical_view_ids:
            validate_stable_id(view_id, field_name="required_numerical_view_ids")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if any(
            not reason.startswith(("OBLIGATION_", "EVIDENCE_", "CLOCK_"))
            for reason in self.reason_codes
        ):
            raise ValueError("temporal-predicate reason is outside the current families")
        if not self.evaluator.deterministic:
            raise ValueError("temporal predicate evaluator must be deterministic")
        if self.physical_independent_unit_count < 0:
            raise ValueError("physical independent-unit count must be nonnegative")
        if not ReceiverInterval.same_domain(
            self.protected_domain.start,
            self.evaluated_domain.start,
        ):
            raise ValueError("protected and evaluated domains use different receiver clocks")
        for name, coordinate in (
            ("first_failure", self.first_failure),
            ("satisfaction_coordinate", self.satisfaction_coordinate),
        ):
            if coordinate is not None and not self.evaluated_domain.contains(coordinate):
                raise ValueError(f"{name} lies outside the evaluated domain")
        if self.status not in {
            ObligationStatus.SATISFIED,
            ObligationStatus.FAILED,
            ObligationStatus.UNEVALUABLE,
        }:
            raise ValueError("temporal predicate requires a terminal evaluation status")

        if self.status is ObligationStatus.UNEVALUABLE:
            if (
                self.first_failure is not None
                or self.satisfaction_coordinate is not None
                or not self.reason_codes
            ):
                raise ValueError("unevaluable temporal predicate cannot impute an outcome")
            return
        if (
            self.physical_independent_unit_count == 0
            or not self.evidence_links
            or self.reason_codes
            and self.status is ObligationStatus.SATISFIED
        ):
            raise ValueError("evaluated temporal predicate lacks independent evidence")
        if self.status is ObligationStatus.FAILED and not self.reason_codes:
            raise ValueError("failed temporal predicate requires reasons")

        if self.semantics in {
            TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH,
            TemporalPredicateSemantics.INTERVAL_INTEGRAL,
        }:
            if self.status is ObligationStatus.SATISFIED:
                if (
                    not self.evaluated_domain.covers(self.protected_domain)
                    or self.first_failure is not None
                    or self.satisfaction_coordinate is not None
                ):
                    raise ValueError(
                        "satisfied path/integral predicate requires complete protected coverage"
                    )
            elif self.semantics is TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH:
                if (
                    self.first_failure is None
                    or not self.protected_domain.contains(self.first_failure)
                    or self.evaluated_domain.start.coordinate
                    > self.protected_domain.start.coordinate
                    or self.satisfaction_coordinate is not None
                ):
                    raise ValueError("failed path predicate requires its first protected failure")
            elif (
                not self.evaluated_domain.covers(self.protected_domain)
                or self.first_failure is None
                or self.satisfaction_coordinate is not None
            ):
                raise ValueError("failed integral predicate requires complete decisive coverage")
        elif self.semantics is TemporalPredicateSemantics.TERMINAL:
            if not self.evaluated_domain.contains(self.protected_domain.end):
                raise ValueError("terminal predicate must evaluate its protected endpoint")
            if self.satisfaction_coordinate is not None:
                raise ValueError("terminal predicate cannot use an attainment coordinate")
            if self.status is ObligationStatus.SATISFIED and self.first_failure is not None:
                raise ValueError("satisfied terminal predicate cannot carry a failure")
            if self.status is ObligationStatus.FAILED and (
                self.first_failure is None
                or self.first_failure.coordinate != self.protected_domain.end.coordinate
            ):
                raise ValueError("failed terminal predicate must bind its terminal coordinate")
        elif self.semantics is TemporalPredicateSemantics.EVENTUALLY_REACHED:
            if self.first_failure is not None:
                raise ValueError("eventually predicate does not use a first-failure coordinate")
            if self.status is ObligationStatus.SATISFIED:
                if self.satisfaction_coordinate is None or not self.protected_domain.contains(
                    self.satisfaction_coordinate
                ):
                    raise ValueError("satisfied eventually predicate requires a protected event")
            elif self.satisfaction_coordinate is not None or not self.evaluated_domain.covers(
                self.protected_domain
            ):
                raise ValueError("failed eventually predicate requires complete search coverage")
        else:
            if self.first_failure is None:
                raise ValueError("recoverable-window predicate requires its initiating failure")
            if self.status is ObligationStatus.SATISFIED:
                if (
                    self.satisfaction_coordinate is None
                    or not self.protected_domain.contains(self.satisfaction_coordinate)
                    or self.satisfaction_coordinate.coordinate < self.first_failure.coordinate
                ):
                    raise ValueError("satisfied recovery requires an in-window recovery")
            elif self.satisfaction_coordinate is not None or not self.evaluated_domain.covers(
                self.protected_domain
            ):
                raise ValueError("failed recovery requires complete window coverage")


@dataclass(frozen=True, slots=True)
class ActionStageReceiverTransport(CanonicalRecord):
    """One occurrence-stage projection into the receiver clock."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-stage-receiver-transport'

    occurrence_id: str
    stage: ActionDeliveryStage
    transport: ClockTransport

    def __post_init__(self) -> None:
        validate_stable_id(self.occurrence_id, field_name="occurrence_id")


def _event(occurrence: ActionOccurrence, stage: ActionDeliveryStage) -> ActionStageEvent:
    return {
        ActionDeliveryStage.REQUESTED: occurrence.requested,
        ActionDeliveryStage.ACCEPTED: occurrence.accepted,
        ActionDeliveryStage.APPLIED: occurrence.applied,
        ActionDeliveryStage.REALIZED: occurrence.realized,
    }[stage]


@dataclass(frozen=True, slots=True)
class ActionStageCausalSupportAssessment(CanonicalRecord):
    """Complete causal-support axes for one exact current action word."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-stage-causal-support-assessment'

    assessment_id: str
    relation: RelationalIdentity
    world_id: str
    action_word: OccurrenceActionWord
    receiver_id: str
    receiver_clock_id: str
    receiver_time_unit: str
    receiver_coordinate_frame: str
    receiver_origin: CoordinateOrigin
    stage_transports: tuple[ActionStageReceiverTransport, ...]
    earliest_influence: ClockCoordinate | None
    latest_influence: ClockCoordinate | None
    evidence_evaluability: EvidenceEvaluability
    delivered_action_validity: DeliveredActionValidity
    causal_cone: CausalConeExistence
    response_direction: ResponseDirectionStatus
    physical_independent_unit_id: str
    physical_independent_unit_count: int
    numerical_view_ids: tuple[str, ...]
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("world_id", self.world_id),
            ("receiver_id", self.receiver_id),
            ("receiver_clock_id", self.receiver_clock_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.action_word.receiver_id != self.receiver_id:
            raise ValueError("action word and causal-support receiver differ")
        if self.action_word.horizon_id != self.relation.horizon.horizon_id:
            raise ValueError("action word and causal-support horizon differ")
        action_quantities = {
            occurrence.channel.controller_quantity_id for occurrence in self.action_word.occurrences
        }
        if not action_quantities <= set(self.relation.action_quantity_ids):
            raise ValueError("action word controller quantity is outside its relation")
        if not self.action_word.occurrences:
            raise ValueError("causal support requires a non-identity action word")
        require_sorted_unique_strings(
            self.numerical_view_ids,
            field_name="numerical_view_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if any(
            not reason.startswith(
                (
                    "ACTION_",
                    "CLOCK_",
                    "EVIDENCE_",
                    "RECEIVER_",
                    "SUPPORT_",
                )
            )
            for reason in self.reason_codes
        ):
            raise ValueError("causal-support reason is outside the current families")
        if self.physical_independent_unit_count < 0:
            raise ValueError("physical independent-unit count must be nonnegative")
        if any(link.world_id != self.world_id for link in self.evidence_links):
            raise ValueError("causal-support evidence crosses evidence worlds")
        expected_stage_keys = tuple(
            (occurrence.occurrence_id, stage)
            for occurrence in self.action_word.occurrences
            for stage in ActionDeliveryStage
        )
        observed_stage_keys = tuple(
            (binding.occurrence_id, binding.stage) for binding in self.stage_transports
        )
        if observed_stage_keys != expected_stage_keys:
            raise ValueError("causal support requires every occurrence delivery stage in order")
        occurrences = {
            occurrence.occurrence_id: occurrence for occurrence in self.action_word.occurrences
        }
        projections = []
        for binding in self.stage_transports:
            source = _event(occurrences[binding.occurrence_id], binding.stage).coordinate
            projection = binding.transport.project(source)
            projections.append(projection)
            if projection.target is not None and (
                projection.target.clock_id != self.receiver_clock_id
                or projection.target.time_unit != self.receiver_time_unit
                or projection.target.coordinate_frame != self.receiver_coordinate_frame
                or projection.target.origin is not self.receiver_origin
            ):
                raise ValueError("causal support stage transport targets a foreign receiver clock")
        unavailable_transport = any(
            projection.status is ClockProjectionStatus.UNAVAILABLE for projection in projections
        )
        if (self.earliest_influence is None) != (self.latest_influence is None):
            raise ValueError("causal influence requires both interval endpoints")
        if self.earliest_influence is not None:
            if self.latest_influence is None:  # pragma: no cover - paired invariant
                raise AssertionError("causal influence interval lost its endpoint")
            expected_domain = ClockCoordinate(
                clock_id=self.receiver_clock_id,
                coordinate=self.earliest_influence.coordinate,
                time_unit=self.receiver_time_unit,
                coordinate_frame=self.receiver_coordinate_frame,
                origin=self.receiver_origin,
            )
            if self.earliest_influence != expected_domain or not ReceiverInterval.same_domain(
                self.earliest_influence,
                self.latest_influence,
            ):
                raise ValueError("causal influence uses a foreign receiver clock")
            if self.latest_influence.coordinate < self.earliest_influence.coordinate:
                raise ValueError("causal influence interval is reversed")
        if self.causal_cone is CausalConeExistence.PRESENT:
            if unavailable_transport or self.earliest_influence is None:
                raise ValueError("present causal cone requires complete clock transport")
            realized_targets = tuple(
                projection.require_target()
                for projection, binding in zip(
                    projections,
                    self.stage_transports,
                    strict=True,
                )
                if binding.stage is ActionDeliveryStage.REALIZED
            )
            latest_realized = max(
                target.coordinate + binding.transport.tolerance
                for target, binding in zip(
                    realized_targets,
                    tuple(
                        value
                        for value in self.stage_transports
                        if value.stage is ActionDeliveryStage.REALIZED
                    ),
                    strict=True,
                )
            )
            earliest_realized = min(target.coordinate for target in realized_targets)
            if isinstance(self, StreamingCausalSupportAssessment):
                if (
                    self.earliest_influence.coordinate <= earliest_realized
                    or self.latest_influence is None
                    or self.latest_influence.coordinate <= latest_realized
                ):
                    raise ValueError(
                        "streaming influence must follow its first exposure and cover its last exposure"
                    )
            elif self.earliest_influence.coordinate <= latest_realized:
                raise ValueError("causal influence must follow every realized action")
        elif self.earliest_influence is not None:
            raise ValueError("absent causal cone cannot claim an influence interval")
        if (
            unavailable_transport
            and self.evidence_evaluability is not EvidenceEvaluability.UNEVALUABLE
        ):
            raise ValueError("unavailable stage transport makes causal support unevaluable")
        if self.evidence_evaluability is EvidenceEvaluability.EVALUABLE:
            if (
                self.physical_independent_unit_count == 0
                or not self.evidence_links
                or not self.evidence_ceiling.allows(EvidenceRung.RESPONSE)
            ):
                raise ValueError("evaluated causal support lacks categorical response evidence")
        inherited = inherited_visibility(
            (
                *self.parent_visibility_ceilings,
                *(link.visibility_ceiling for link in self.evidence_links),
            ),
            self.outcome_access,
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("causal-support visibility cannot be lowered")
        if not self.visibility_ceiling.is_promotable and (
            self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("outcome-visible causal support must remain non-promotable")
        expected_reasons = bool(
            self.evidence_evaluability is EvidenceEvaluability.UNEVALUABLE
            or self.delivered_action_validity is not DeliveredActionValidity.VALID
            or self.causal_cone is not CausalConeExistence.PRESENT
            or self.response_direction is not ResponseDirectionStatus.SUPPORTED
        )
        if expected_reasons != bool(self.reason_codes):
            raise ValueError("causal-support reasons differ from its complete axes")


@dataclass(frozen=True, slots=True)
class StreamingCausalSupportAssessment(ActionStageCausalSupportAssessment):
    """Explicit compatibility for sequential exposures with overlapping responses.

    A window response may start after the first delivered subinterval while later
    subintervals are still being delivered. The old whole-word causal cone keeps
    its stricter after-last-exposure semantics. This map does not claim that a
    later exposure caused an earlier response sample.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/streaming-causal-support-assessment'
    exposure_response_map: ObjectIdentity

    def __post_init__(self) -> None:
        ActionStageCausalSupportAssessment.__post_init__(self)
        if not self.action_word.groups or any(
            o.duration <= 0 for o in self.action_word.occurrences
        ):
            raise ValueError("streaming causal map requires held exposure intervals")


@dataclass(frozen=True, slots=True)
class CausalPrefixAssessment(CanonicalRecord):
    """Total noncompensating causal-prefix state and derived disposition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/causal-prefix-assessment'

    assessment_id: str
    causal_support: ActionStageCausalSupportAssessment | StreamingCausalSupportAssessment
    prefix_id: str
    prefix_cutoff: ClockCoordinate
    prefix_support: PrefixSupportStatus
    obligations: tuple[TemporalPredicateAssessment, ...]
    required_numerical_view_ids: tuple[str, ...]
    numerical_view_agreement: NumericalViewAgreement
    obstructions: tuple[CausalObstructionKind, ...]
    disposition: CausalCompositionDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.prefix_id, field_name="prefix_id")
        support = self.causal_support
        receiver_template = ClockCoordinate(
            clock_id=support.receiver_clock_id,
            coordinate=self.prefix_cutoff.coordinate,
            time_unit=support.receiver_time_unit,
            coordinate_frame=support.receiver_coordinate_frame,
            origin=support.receiver_origin,
        )
        if self.prefix_cutoff != receiver_template:
            raise ValueError("prefix cutoff uses a foreign receiver clock")
        require_sorted_unique_ids(
            self.obligations,
            attribute="obligation_id",
            field_name="obligations",
        )
        if not self.obligations:
            raise ValueError("causal prefix requires temporal obligations")
        require_sorted_unique_strings(
            self.required_numerical_view_ids,
            field_name="required_numerical_view_ids",
            allow_empty=False,
        )
        if not set(self.required_numerical_view_ids) <= set(support.numerical_view_ids):
            raise ValueError("prefix assessment requires an absent numerical view")
        earliest = support.earliest_influence
        if earliest is not None:
            requested_targets = []
            occurrences = {value.occurrence_id: value for value in support.action_word.occurrences}
            for stage_transport in support.stage_transports:
                if stage_transport.stage is ActionDeliveryStage.REQUESTED:
                    source = occurrences[stage_transport.occurrence_id].requested.coordinate
                    requested_targets.append(
                        stage_transport.transport.project(source).require_target()
                    )
            if requested_targets and self.prefix_cutoff.coordinate > min(
                value.coordinate for value in requested_targets
            ):
                raise ValueError("prepared prefix extends beyond an action request")
            for obligation in self.obligations:
                if obligation.receiver_id != support.receiver_id:
                    raise ValueError("prefix obligation uses a foreign receiver")
                if not ReceiverInterval.same_domain(
                    obligation.protected_domain.start,
                    earliest,
                ):
                    raise ValueError("prefix obligation uses a foreign receiver clock")
                if obligation.protected_domain.start.coordinate > self.prefix_cutoff.coordinate:
                    raise ValueError("prefix obligation omits the prepared prefix boundary")
                if obligation.protected_domain.end.coordinate < earliest.coordinate:
                    raise ValueError(
                        "prefix obligation protected domain ends before earliest influence"
                    )
                if not set(obligation.required_numerical_view_ids) <= set(
                    self.required_numerical_view_ids
                ):
                    raise ValueError("prefix obligation uses an undeclared numerical view")
                if any(link.world_id != support.world_id for link in obligation.evidence_links):
                    raise ValueError("prefix obligation evidence crosses evidence worlds")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if any(
            not reason.startswith(
                ("ACTION_", "CLOCK_", "EVIDENCE_", "OBLIGATION_", "PREFIX_", "RECEIVER_")
            )
            for reason in self.reason_codes
        ):
            raise ValueError("causal-prefix reason is outside the current families")
        derived = self.derive_obstructions()
        if self.obstructions != derived:
            raise ValueError("causal obstruction set is not derived from all axes")
        expected_disposition = self.derive_disposition(derived)
        if self.disposition is not expected_disposition:
            raise ValueError("causal disposition does not follow obstruction precedence")
        if bool(self.reason_codes) != bool(derived):
            raise ValueError("causal-prefix reasons differ from its obstruction set")

    def derive_obstructions(self) -> tuple[CausalObstructionKind, ...]:
        support = self.causal_support
        values: set[CausalObstructionKind] = set()
        if support.evidence_evaluability is EvidenceEvaluability.UNEVALUABLE:
            values.add(CausalObstructionKind.EVIDENCE_UNEVALUABLE)
        if support.delivered_action_validity is DeliveredActionValidity.INVALID:
            values.add(CausalObstructionKind.ACTION_DELIVERY_INVALID)
        elif support.delivered_action_validity is DeliveredActionValidity.UNEVALUABLE:
            values.add(CausalObstructionKind.ACTION_DELIVERY_UNEVALUABLE)
        occurrences = {value.occurrence_id: value for value in support.action_word.occurrences}
        if any(
            binding.transport.project(
                _event(occurrences[binding.occurrence_id], binding.stage).coordinate
            ).status
            is ClockProjectionStatus.UNAVAILABLE
            for binding in support.stage_transports
        ):
            values.add(CausalObstructionKind.CLOCK_TRANSPORT_UNAVAILABLE)
        if support.causal_cone is CausalConeExistence.MISSING:
            values.add(CausalObstructionKind.CAUSAL_CONE_MISSING)
        elif support.causal_cone is CausalConeExistence.UNEVALUABLE:
            values.add(CausalObstructionKind.CAUSAL_CONE_UNEVALUABLE)
        if support.response_direction is ResponseDirectionStatus.OPPOSED:
            values.add(CausalObstructionKind.RESPONSE_DIRECTION_OPPOSED)
        elif support.response_direction is ResponseDirectionStatus.UNEVALUABLE:
            values.add(CausalObstructionKind.RESPONSE_DIRECTION_UNEVALUABLE)
        if self.prefix_support is PrefixSupportStatus.OUTSIDE_SUPPORT:
            values.add(CausalObstructionKind.PREFIX_OUTSIDE_SUPPORT)
        elif self.prefix_support is PrefixSupportStatus.UNEVALUABLE:
            values.add(CausalObstructionKind.PREFIX_SUPPORT_UNEVALUABLE)
        earliest = support.earliest_influence
        for obligation in self.obligations:
            if obligation.status is ObligationStatus.UNEVALUABLE:
                values.add(CausalObstructionKind.TEMPORAL_OBLIGATION_UNEVALUABLE)
            elif (
                obligation.semantics is TemporalPredicateSemantics.ALWAYS_PRESERVED_PATH
                and obligation.status is ObligationStatus.FAILED
                and obligation.first_failure is not None
                and earliest is not None
                and obligation.first_failure.coordinate <= earliest.coordinate
            ):
                values.add(CausalObstructionKind.TEMPORAL_OBLIGATION_FAILED_BEFORE_INFLUENCE)
        if self.numerical_view_agreement is NumericalViewAgreement.DISAGREED:
            values.add(CausalObstructionKind.NUMERICAL_VIEW_DISAGREEMENT)
        elif self.numerical_view_agreement is NumericalViewAgreement.UNEVALUABLE:
            values.add(CausalObstructionKind.NUMERICAL_VIEW_UNEVALUABLE)
        return tuple(sorted(values, key=lambda value: value.value))

    @staticmethod
    def derive_disposition(
        obstructions: tuple[CausalObstructionKind, ...],
    ) -> CausalCompositionDisposition:
        if not obstructions:
            return CausalCompositionDisposition.DEFINED
        if any(value in _UNEVALUABLE_OBSTRUCTIONS for value in obstructions):
            return CausalCompositionDisposition.UNEVALUABLE
        return CausalCompositionDisposition.BLOCKED


def decode_causal_prefix_assessment(payload: bytes) -> CausalPrefixAssessment:
    return decode_canonical_bytes(
        payload,
        CausalPrefixAssessment,
        maximum_bytes=MAX_CAUSAL_CONTRACT_BYTES,
    )


__all__ = [
    "ActionStageReceiverTransport",
    "CausalCompositionDisposition",
    "CausalConeExistence",
    "CausalObstructionKind",
    "CausalPrefixAssessment",
    'ActionStageCausalSupportAssessment',
    "DeliveredActionValidity",
    "EvidenceEvaluability",
    "NumericalViewAgreement",
    "PredicateDirection",
    "PrefixSupportStatus",
    "ReceiverInterval",
    "ResponseDirectionStatus",
    "TemporalPredicateAssessment",
    "TemporalPredicateSemantics",
    "decode_causal_prefix_assessment",
]

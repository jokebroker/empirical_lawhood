"""Current action-occurrence, native-channel and delivery-time contracts.

These records are additive successors to the archived response-algebra action
word.  They preserve occurrence multiplicity and bind every delivery stage to
an explicit directioned clock transport.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from .decoding import decode_canonical_bytes
from .serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)
from .time import (
    ClockCoordinate,
    ClockTransport,
    CoordinateOrigin,
)


MAX_ACTION_CONTRACT_BYTES = 1024 * 1024


class ActionDeliveryStage(StrEnum):
    REQUESTED = "REQUESTED"
    ACCEPTED = "ACCEPTED"
    APPLIED = "APPLIED"
    REALIZED = "REALIZED"


_STAGES = tuple(ActionDeliveryStage)


class ActionWordMode(StrEnum):
    IDENTITY = "IDENTITY"
    SEQUENTIAL = "SEQUENTIAL"
    SIMULTANEOUS = "SIMULTANEOUS"
    MIXED = "MIXED"


class ActionWordSupportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ActionStageQuantityBinding(CanonicalRecord):
    """Static quantity and native-clock identity for one delivery stage."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-stage-quantity-binding'

    stage: ActionDeliveryStage
    quantity_id: str
    clock_id: str
    time_unit: str
    clock_coordinate_frame: str
    clock_origin: CoordinateOrigin

    def __post_init__(self) -> None:
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_stable_id(self.clock_id, field_name="clock_id")
        validate_nonempty(self.time_unit, field_name="time_unit")
        validate_nonempty(
            self.clock_coordinate_frame,
            field_name="clock_coordinate_frame",
        )


@dataclass(frozen=True, slots=True)
class ActionChannelBinding(CanonicalRecord):
    """One-to-one kernel-port, controller-channel and four-stage binding."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-channel-binding'

    binding_id: str
    port_id: str
    controller_quantity_id: str
    stages: tuple[ActionStageQuantityBinding, ...]
    native_unit: str
    native_action_frame: str
    native_direction: str
    support_contract_id: str
    delivery_contract_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("port_id", self.port_id),
            ("controller_quantity_id", self.controller_quantity_id),
            ("native_direction", self.native_direction),
            ("support_contract_id", self.support_contract_id),
            ("delivery_contract_id", self.delivery_contract_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_nonempty(self.native_action_frame, field_name="native_action_frame")
        if tuple(stage.stage for stage in self.stages) != _STAGES:
            raise ValueError(
                "action channel must bind requested, accepted, applied and realized stages"
            )
        if len({stage.quantity_id for stage in self.stages}) != len(self.stages):
            raise ValueError("action delivery stages require distinct quantity identities")

    def stage_binding(self, stage: ActionDeliveryStage) -> ActionStageQuantityBinding:
        return self.stages[_STAGES.index(stage)]


@dataclass(frozen=True, slots=True)
class ActionStageEvent(CanonicalRecord):
    """Observed/requested value and native time for one occurrence stage."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-stage-event'

    stage: ActionDeliveryStage
    quantity_id: str
    value: Decimal
    native_unit: str
    native_action_frame: str
    native_direction: str
    coordinate: ClockCoordinate

    def __post_init__(self) -> None:
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_decimal(self.value, field_name="value")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_nonempty(self.native_action_frame, field_name="native_action_frame")
        validate_stable_id(self.native_direction, field_name="native_direction")


@dataclass(frozen=True, slots=True)
class ActionOccurrence(CanonicalRecord):
    """One action occurrence with all stages and exact adjacent transports."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-occurrence'

    occurrence_id: str
    channel: ActionChannelBinding
    requested: ActionStageEvent
    accepted: ActionStageEvent
    applied: ActionStageEvent
    realized: ActionStageEvent
    requested_to_accepted: ClockTransport
    accepted_to_applied: ClockTransport
    applied_to_realized: ClockTransport
    duration: Decimal
    duration_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.occurrence_id, field_name="occurrence_id")
        validate_decimal(self.duration, field_name="duration", minimum=Decimal("0"))
        if self.duration == 0:
            raise ValueError("action occurrence duration must be positive")
        validate_nonempty(self.duration_unit, field_name="duration_unit")
        events = (self.requested, self.accepted, self.applied, self.realized)
        if tuple(event.stage for event in events) != _STAGES:
            raise ValueError("action occurrence fields do not match delivery-stage order")
        for event, stage_binding in zip(events, self.channel.stages, strict=True):
            if event.quantity_id != stage_binding.quantity_id:
                raise ValueError("action occurrence quantity differs from channel binding")
            coordinate = event.coordinate
            if (
                coordinate.clock_id != stage_binding.clock_id
                or coordinate.time_unit != stage_binding.time_unit
                or coordinate.coordinate_frame != stage_binding.clock_coordinate_frame
                or coordinate.origin is not stage_binding.clock_origin
            ):
                raise ValueError("action occurrence clock differs from channel binding")
            if (
                event.native_unit != self.channel.native_unit
                or event.native_action_frame != self.channel.native_action_frame
                or event.native_direction != self.channel.native_direction
            ):
                raise ValueError("action occurrence native semantics differ from channel binding")
        for source, target, transport in (
            (self.requested, self.accepted, self.requested_to_accepted),
            (self.accepted, self.applied, self.accepted_to_applied),
            (self.applied, self.realized, self.applied_to_realized),
        ):
            projection = transport.project(source.coordinate)
            if not projection.matches(target.coordinate):
                raise ValueError("action-stage coordinate is outside its bound clock transport")


@dataclass(frozen=True, slots=True)
class ObservedActionOccurrence(CanonicalRecord):
    """Available delivery-stage evidence for one expected occurrence.

    Unlike :class:`ActionOccurrence`, this record is deliberately partial.  It
    records what was actually observed and never fills a missing accepted,
    applied or realized stage merely to satisfy the expected action grammar.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/observed-action-occurrence'

    observation_id: str
    expected_occurrence_id: str
    requested: ActionStageEvent | None
    accepted: ActionStageEvent | None
    applied: ActionStageEvent | None
    realized: ActionStageEvent | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        validate_stable_id(
            self.expected_occurrence_id,
            field_name="expected_occurrence_id",
        )
        events = (self.requested, self.accepted, self.applied, self.realized)
        if not any(value is not None for value in events):
            raise ValueError("observed occurrence must contain at least one available stage")
        for expected_stage, event in zip(_STAGES, events, strict=True):
            if event is not None and event.stage is not expected_stage:
                raise ValueError("observed occurrence stage appears in the wrong field")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")

    @property
    def complete(self) -> bool:
        return all(
            value is not None
            for value in (self.requested, self.accepted, self.applied, self.realized)
        )


@dataclass(frozen=True, slots=True)
class ActionOccurrenceOrder(CanonicalRecord):
    """Projection of one applied event into the word's ordering clock."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-occurrence-order'

    occurrence_id: str
    transport: ClockTransport

    def __post_init__(self) -> None:
        validate_stable_id(self.occurrence_id, field_name="occurrence_id")


@dataclass(frozen=True, slots=True)
class ActionOccurrenceGroup(CanonicalRecord):
    """Simultaneous occurrence group; groups themselves are chronological."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/action-occurrence-group'

    group_id: str
    members: tuple[ActionOccurrenceOrder, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.group_id, field_name="group_id")
        require_sorted_unique_ids(
            self.members,
            attribute="occurrence_id",
            field_name="members",
        )
        if not self.members:
            raise ValueError("occurrence group cannot be empty")


@dataclass(frozen=True, slots=True)
class OccurrenceActionWord(CanonicalRecord):
    """Current action word with occurrence-safe mixed temporal composition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/occurrence-action-word'

    word_id: str
    mode: ActionWordMode
    occurrences: tuple[ActionOccurrence, ...]
    groups: tuple[ActionOccurrenceGroup, ...]
    ordering_clock_id: str | None
    ordering_time_unit: str | None
    ordering_coordinate_frame: str | None
    ordering_origin: CoordinateOrigin | None
    denominator_id: str
    retained_history_id: str
    receiver_id: str
    horizon_id: str
    prefix_support_ids: tuple[str, ...]
    support_status: ActionWordSupportStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("word_id", self.word_id),
            ("denominator_id", self.denominator_id),
            ("retained_history_id", self.retained_history_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(value, field_name=name)
        if len(self.occurrences) > 32:
            raise ValueError("action word exceeds its bounded occurrence count")
        occurrence_ids = tuple(value.occurrence_id for value in self.occurrences)
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("action word requires unique occurrence identities")
        if not self.prefix_support_ids or len(set(self.prefix_support_ids)) != len(
            self.prefix_support_ids
        ):
            raise ValueError("action word requires unique chronological prefix supports")
        for prefix_id in self.prefix_support_ids:
            validate_stable_id(prefix_id, field_name="prefix_support_ids")
        if len(self.prefix_support_ids) != len(self.occurrences) + 1:
            raise ValueError("action word requires identity and every successive prefix")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if any(
            not value.startswith(("ACTION_", "CLOCK_", "SUPPORT_")) for value in self.reason_codes
        ):
            raise ValueError("action-word reason is outside the current reason families")
        if self.support_status is ActionWordSupportStatus.SUPPORTED:
            if self.reason_codes:
                raise ValueError("supported action word cannot carry failure reasons")
        elif not self.reason_codes:
            raise ValueError("non-supported action word requires reasons")

        if not self.occurrences:
            if self.mode is not ActionWordMode.IDENTITY or self.groups:
                raise ValueError("empty action word must be the identity")
            if any(
                value is not None
                for value in (
                    self.ordering_clock_id,
                    self.ordering_time_unit,
                    self.ordering_coordinate_frame,
                    self.ordering_origin,
                )
            ):
                raise ValueError("identity action word cannot declare an ordering clock")
            return

        if self.mode is ActionWordMode.IDENTITY:
            raise ValueError("identity action word cannot contain occurrences")
        if self.ordering_clock_id is None:
            raise ValueError("non-identity action word requires ordering_clock_id")
        validate_stable_id(self.ordering_clock_id, field_name="ordering_clock_id")
        if self.ordering_time_unit is None:
            raise ValueError("non-identity action word requires ordering_time_unit")
        validate_nonempty(self.ordering_time_unit, field_name="ordering_time_unit")
        if self.ordering_coordinate_frame is None:
            raise ValueError("non-identity action word requires ordering_coordinate_frame")
        validate_nonempty(
            self.ordering_coordinate_frame,
            field_name="ordering_coordinate_frame",
        )
        if self.ordering_origin is None:
            raise ValueError("non-identity action word requires ordering_origin")
        if not self.groups:
            raise ValueError("non-identity action word requires occurrence groups")

        occurrences = {value.occurrence_id: value for value in self.occurrences}
        flattened = tuple(member.occurrence_id for group in self.groups for member in group.members)
        if set(flattened) != set(occurrences) or len(flattened) != len(occurrences):
            raise ValueError("action-word groups must cover each occurrence exactly once")
        if flattened != occurrence_ids:
            raise ValueError("action occurrences must use exact chronological group order")

        intervals: list[tuple[Decimal, Decimal]] = []
        for group in self.groups:
            member_intervals: list[tuple[Decimal, Decimal]] = []
            for member in group.members:
                occurrence = occurrences[member.occurrence_id]
                projection = member.transport.project(occurrence.applied.coordinate)
                target = projection.require_target()
                if (
                    target.clock_id != self.ordering_clock_id
                    or target.time_unit != self.ordering_time_unit
                    or target.coordinate_frame != self.ordering_coordinate_frame
                    or target.origin is not self.ordering_origin
                ):
                    raise ValueError("action ordering uses a foreign clock transport")
                tolerance = member.transport.tolerance
                member_intervals.append(
                    (target.coordinate - tolerance, target.coordinate + tolerance)
                )
            lower = max(interval[0] for interval in member_intervals)
            upper = min(interval[1] for interval in member_intervals)
            if lower > upper:
                raise ValueError("simultaneous occurrence transports do not overlap")
            intervals.append((lower, upper))
        if any(
            later[0] <= earlier[1] for earlier, later in zip(intervals, intervals[1:], strict=False)
        ):
            raise ValueError("action-word occurrence groups are not strictly ordered")

        derived_mode = self._derived_mode()
        if self.mode is not derived_mode:
            raise ValueError("action-word mode differs from its occurrence grouping")

    def _derived_mode(self) -> ActionWordMode:
        if not self.occurrences:
            return ActionWordMode.IDENTITY
        group_sizes = tuple(len(group.members) for group in self.groups)
        if all(size == 1 for size in group_sizes):
            return ActionWordMode.SEQUENTIAL
        if len(group_sizes) == 1:
            return ActionWordMode.SIMULTANEOUS
        return ActionWordMode.MIXED

    @property
    def chronological_occurrence_ids(self) -> tuple[str, ...]:
        return tuple(member.occurrence_id for group in self.groups for member in group.members)


def decode_action_word(payload: bytes) -> OccurrenceActionWord:
    """Decode only the strict current action-word schema."""

    return decode_canonical_bytes(
        payload,
        OccurrenceActionWord,
        maximum_bytes=MAX_ACTION_CONTRACT_BYTES,
    )


__all__ = [
    "ActionChannelBinding",
    "ActionDeliveryStage",
    "ActionOccurrence",
    "ActionOccurrenceGroup",
    "ActionOccurrenceOrder",
    "ObservedActionOccurrence",
    "ActionStageEvent",
    "ActionStageQuantityBinding",
    'OccurrenceActionWord',
    "ActionWordMode",
    "ActionWordSupportStatus",
    "decode_action_word",
]

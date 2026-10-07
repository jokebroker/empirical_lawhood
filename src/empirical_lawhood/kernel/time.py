"""Native clocks, causal cutoffs and finite horizons."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Final

from .evidence import OutcomeAccess
from .references import ExecutableReference
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)


MAX_UTC_TIMESTAMP_BYTES: Final[int] = 32
_UTC_TIMESTAMP = re.compile(
    r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?Z$"
)


def parse_utc_timestamp(value: str, *, field_name: str = "timestamp") -> datetime:
    """Parse one strict ISO-8601 UTC timestamp whose only zone spelling is ``Z``."""

    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a UTC timestamp string ending in Z")
    try:
        encoded = value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(f"{field_name} must be valid UTF-8") from error
    if len(encoded) > MAX_UTC_TIMESTAMP_BYTES:
        raise ValueError(f"{field_name} exceeds the UTC timestamp byte limit")
    if _UTC_TIMESTAMP.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a strict UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(f"{value[:-1]}+00:00")
    except ValueError as error:
        raise ValueError(f"{field_name} is not a valid UTC timestamp") from error
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{field_name} must represent UTC")
    return parsed


def validate_utc_timestamp(value: str, *, field_name: str = "timestamp") -> str:
    """Validate and return one strict ``Z`` UTC timestamp string."""

    parse_utc_timestamp(value, field_name=field_name)
    return value


def _require_enum_instance(value: object, enum_type: type[StrEnum], *, field_name: str) -> None:
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be a {enum_type.__name__} instance")


class SamplingSemantics(StrEnum):
    CONTINUOUS = "CONTINUOUS"
    REGULAR = "REGULAR"
    IRREGULAR = "IRREGULAR"
    EVENT_DRIVEN = "EVENT_DRIVEN"


class HoldSemantics(StrEnum):
    NONE = "NONE"
    ZERO_ORDER = "ZERO_ORDER"
    FIRST_ORDER = "FIRST_ORDER"
    SOURCE_DEFINED = "SOURCE_DEFINED"


class ClockLabelSemantics(StrEnum):
    INSTANT = "INSTANT"
    INTERVAL_START = "INTERVAL_START"
    INTERVAL_END = "INTERVAL_END"
    EVENT = "EVENT"


class CausalPhase(StrEnum):
    PREPARATION = "PREPARATION"
    PRE_ACTION = "PRE_ACTION"
    ACTION_REQUESTED = "ACTION_REQUESTED"
    ACTION_APPLIED = "ACTION_APPLIED"
    RECEIVER = "RECEIVER"
    POST_OUTCOME = "POST_OUTCOME"

    def precedes_or_equals(self, other: CausalPhase) -> bool:
        _require_enum_instance(other, CausalPhase, field_name="other")
        return _PHASE_RANK[self] <= _PHASE_RANK[other]


_PHASE_RANK = {phase: index for index, phase in enumerate(CausalPhase)}


@dataclass(frozen=True, slots=True)
class ClockSpec(CanonicalRecord):
    """One source-native time coordinate and its sampling semantics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/clock-spec'

    clock_id: str
    label: str
    time_unit: str
    coordinate_frame: str
    sampling: SamplingSemantics
    hold: HoldSemantics
    label_semantics: ClockLabelSemantics
    nominal_period: Decimal | None = None
    alignment_tolerance: Decimal | None = None
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        _require_enum_instance(self.sampling, SamplingSemantics, field_name="sampling")
        _require_enum_instance(self.hold, HoldSemantics, field_name="hold")
        _require_enum_instance(self.label_semantics, ClockLabelSemantics, field_name="label_semantics")
        validate_stable_id(self.clock_id, field_name="clock_id")
        for name, value in (
            ("label", self.label),
            ("time_unit", self.time_unit),
            ("coordinate_frame", self.coordinate_frame),
        ):
            validate_nonempty(value, field_name=name)
        if self.sampling is SamplingSemantics.REGULAR:
            if self.nominal_period is None:
                raise ValueError("a regular clock requires a positive nominal period")
            validate_decimal(
                self.nominal_period,
                field_name="nominal_period",
                minimum=Decimal("0"),
            )
            if self.nominal_period == 0:
                raise ValueError("a regular clock requires a positive nominal period")
        elif self.nominal_period is not None:
            raise ValueError("only a regular clock may declare a nominal period")
        if self.alignment_tolerance is not None:
            validate_decimal(
                self.alignment_tolerance,
                field_name="alignment_tolerance",
                minimum=Decimal("0"),
            )
        require_extensions(self.extensions)

    def is_identity_compatible_with(self, other: ClockSpec) -> bool:
        return (
            self.clock_id == other.clock_id
            and self.time_unit == other.time_unit
            and self.coordinate_frame == other.coordinate_frame
            and self.label_semantics is other.label_semantics
        )


@dataclass(frozen=True, slots=True)
class AvailabilitySpec(CanonicalRecord):
    """When a quantity becomes available relative to its native clock."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/availability-spec'

    clock_id: str
    phase: CausalPhase
    outcome_access: OutcomeAccess
    available_at: Decimal | None = None

    def __post_init__(self) -> None:
        _require_enum_instance(self.phase, CausalPhase, field_name="phase")
        _require_enum_instance(self.outcome_access, OutcomeAccess, field_name="outcome_access")
        validate_stable_id(self.clock_id, field_name="clock_id")
        if self.available_at is not None:
            validate_decimal(self.available_at, field_name="available_at")


@dataclass(frozen=True, slots=True)
class InformationCutoff(CanonicalRecord):
    """Latest native-clock information available to one computation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/information-cutoff'

    cutoff_id: str
    clock_id: str
    phase: CausalPhase
    coordinate: Decimal | None
    includes_coordinate: bool = True

    def __post_init__(self) -> None:
        _require_enum_instance(self.phase, CausalPhase, field_name="phase")
        if type(self.includes_coordinate) is not bool:
            raise ValueError("includes_coordinate must be boolean")
        validate_stable_id(self.cutoff_id, field_name="cutoff_id")
        validate_stable_id(self.clock_id, field_name="clock_id")
        if self.coordinate is not None:
            validate_decimal(self.coordinate, field_name="coordinate")

    def allows(self, availability: AvailabilitySpec) -> bool:
        if availability.clock_id != self.clock_id:
            return False
        if availability.outcome_access in {
            OutcomeAccess.EVALUATION_SEALED,
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }:
            return False
        phase_comparison = _PHASE_RANK[availability.phase] - _PHASE_RANK[self.phase]
        if phase_comparison < 0:
            return True
        if phase_comparison > 0:
            return False
        if availability.available_at is None or self.coordinate is None:
            return availability.available_at is self.coordinate
        if self.includes_coordinate:
            return availability.available_at <= self.coordinate
        return availability.available_at < self.coordinate

    def require_allows(self, availability: AvailabilitySpec) -> None:
        if not self.allows(availability):
            raise ValueError(
                f"availability on {availability.clock_id!r} exceeds cutoff {self.cutoff_id!r}"
            )


@dataclass(frozen=True, slots=True)
class HorizonSpec(CanonicalRecord):
    """Finite response horizon in one native clock coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/horizon-spec'

    horizon_id: str
    clock_id: str
    duration: Decimal
    time_unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.horizon_id, field_name="horizon_id")
        validate_stable_id(self.clock_id, field_name="clock_id")
        validate_nonempty(self.time_unit, field_name="time_unit")
        validate_decimal(self.duration, field_name="duration", minimum=Decimal("0"))
        if self.duration == 0:
            raise ValueError("horizon duration must be positive")


class ClockRelationKind(StrEnum):
    IDENTITY = "IDENTITY"
    FIXED_DELAY = "FIXED_DELAY"
    SAMPLE_AND_HOLD = "SAMPLE_AND_HOLD"
    SOURCE_DEFINED = "SOURCE_DEFINED"


@dataclass(frozen=True, slots=True)
class ClockRelationSpec(CanonicalRecord):
    """Explicit relation between two otherwise distinct native clocks."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/clock-relation-spec'

    relation_id: str
    source_clock_id: str
    target_clock_id: str
    kind: ClockRelationKind
    delay: Decimal | None
    tolerance: Decimal
    evidence_contract_id: str

    def __post_init__(self) -> None:
        _require_enum_instance(self.kind, ClockRelationKind, field_name="kind")
        for name, value in (
            ("relation_id", self.relation_id),
            ("source_clock_id", self.source_clock_id),
            ("target_clock_id", self.target_clock_id),
            ("evidence_contract_id", self.evidence_contract_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.tolerance, field_name="tolerance", minimum=Decimal("0"))
        if self.delay is not None:
            validate_decimal(self.delay, field_name="delay", minimum=Decimal("0"))
        if self.kind is ClockRelationKind.IDENTITY:
            if self.delay not in {None, Decimal(0)}:
                raise ValueError("an identity clock relation cannot have a delay")
        elif self.delay is None:
            raise ValueError("a non-identity clock relation requires an explicit delay")


class CoordinateOrigin(StrEnum):
    """Meaning of a native clock coordinate."""

    ABSOLUTE = "ABSOLUTE"
    EPISODE_RELATIVE = "EPISODE_RELATIVE"
    COMMITMENT_RELATIVE = "COMMITMENT_RELATIVE"


class ClockTransportKind(StrEnum):
    """Closed current transport vocabulary.

    Both kinds are deterministic affine maps.  ``AFFINE_BOUNDED`` carries a
    nonzero target-clock tolerance; it is not permission to compare unbounded
    or unrelated native coordinates.
    """

    IDENTITY = "IDENTITY"
    AFFINE_EXACT = "AFFINE_EXACT"
    AFFINE_BOUNDED = "AFFINE_BOUNDED"


class ClockTransportAvailability(StrEnum):
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"


class ClockTransportMonotonicity(StrEnum):
    STRICTLY_INCREASING = "STRICTLY_INCREASING"


class ClockProjectionStatus(StrEnum):
    EXACT = "EXACT"
    BOUNDED = "BOUNDED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class ClockCoordinate(CanonicalRecord):
    """One coordinate with enough identity to forbid raw cross-clock use."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/clock-coordinate'

    clock_id: str
    coordinate: Decimal
    time_unit: str
    coordinate_frame: str
    origin: CoordinateOrigin

    def __post_init__(self) -> None:
        _require_enum_instance(self.origin, CoordinateOrigin, field_name="origin")
        validate_stable_id(self.clock_id, field_name="clock_id")
        validate_decimal(self.coordinate, field_name="coordinate")
        validate_nonempty(self.time_unit, field_name="time_unit")
        validate_nonempty(self.coordinate_frame, field_name="coordinate_frame")


@dataclass(frozen=True, slots=True)
class ClockTransport(CanonicalRecord):
    """Directioned, domain-qualified current clock-coordinate transport.

    The executable reference is a content-bound capability identity, not a
    callable.  The kernel performs the declared affine projection itself.
    Missing or partial-domain mappings remain explicit and never impute an
    alignment.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/clock-transport'

    transport_id: str
    source_clock_id: str
    target_clock_id: str
    source_time_unit: str
    target_time_unit: str
    source_coordinate_frame: str
    target_coordinate_frame: str
    source_origin: CoordinateOrigin
    target_origin: CoordinateOrigin
    kind: ClockTransportKind
    availability: ClockTransportAvailability
    scale: Decimal | None
    offset: Decimal | None
    tolerance: Decimal
    valid_source_lower: Decimal | None
    valid_source_upper: Decimal | None
    monotonicity: ClockTransportMonotonicity
    evaluator: ExecutableReference
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum_instance(self.source_origin, CoordinateOrigin, field_name="source_origin")
        _require_enum_instance(self.target_origin, CoordinateOrigin, field_name="target_origin")
        _require_enum_instance(self.kind, ClockTransportKind, field_name="kind")
        _require_enum_instance(self.availability, ClockTransportAvailability, field_name="availability")
        _require_enum_instance(self.monotonicity, ClockTransportMonotonicity, field_name="monotonicity")
        for name, value in (
            ("transport_id", self.transport_id),
            ("source_clock_id", self.source_clock_id),
            ("target_clock_id", self.target_clock_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, value in (
            ("source_time_unit", self.source_time_unit),
            ("target_time_unit", self.target_time_unit),
            ("source_coordinate_frame", self.source_coordinate_frame),
            ("target_coordinate_frame", self.target_coordinate_frame),
        ):
            validate_nonempty(value, field_name=name)
        validate_decimal(self.tolerance, field_name="tolerance", minimum=Decimal("0"))
        for bound_name, bound in (
            ("valid_source_lower", self.valid_source_lower),
            ("valid_source_upper", self.valid_source_upper),
        ):
            if bound is not None:
                validate_decimal(bound, field_name=bound_name)
        if (
            self.valid_source_lower is not None
            and self.valid_source_upper is not None
            and self.valid_source_lower > self.valid_source_upper
        ):
            raise ValueError("clock-transport source domain is reversed")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if any(not reason.startswith("CLOCK_") for reason in self.reason_codes):
            raise ValueError("clock-transport reason must use the CLOCK_ family")
        if not self.evaluator.deterministic:
            raise ValueError("clock transport requires a deterministic evaluator")
        if self.monotonicity is not ClockTransportMonotonicity.STRICTLY_INCREASING:
            raise ValueError("current clock transport must be strictly increasing")

        if self.availability is ClockTransportAvailability.UNAVAILABLE:
            if self.scale is not None or self.offset is not None:
                raise ValueError("unavailable clock transport cannot declare a mapping")
            if not self.reason_codes:
                raise ValueError("unavailable clock transport requires reasons")
            return

        if self.scale is None or self.offset is None:
            raise ValueError("available clock transport requires scale and offset")
        validate_decimal(self.scale, field_name="scale", minimum=Decimal("0"))
        validate_decimal(self.offset, field_name="offset")
        if self.scale == 0:
            raise ValueError("clock transport must be strictly increasing")
        if self.reason_codes:
            raise ValueError("available clock transport cannot carry failure reasons")
        if self.kind is ClockTransportKind.IDENTITY:
            if (
                self.source_clock_id != self.target_clock_id
                or self.source_time_unit != self.target_time_unit
                or self.source_coordinate_frame != self.target_coordinate_frame
                or self.source_origin is not self.target_origin
                or self.scale != Decimal(1)
                or self.offset != Decimal(0)
                or self.tolerance != Decimal(0)
            ):
                raise ValueError("identity clock transport must preserve exact clock semantics")
        elif self.kind is ClockTransportKind.AFFINE_EXACT:
            if self.tolerance != 0:
                raise ValueError("exact affine clock transport requires zero tolerance")
        elif self.tolerance == 0:
            raise ValueError("bounded affine clock transport requires positive tolerance")

    def _validate_source_identity(self, source: ClockCoordinate) -> None:
        if (
            source.clock_id != self.source_clock_id
            or source.time_unit != self.source_time_unit
            or source.coordinate_frame != self.source_coordinate_frame
            or source.origin is not self.source_origin
        ):
            raise ValueError("clock coordinate does not match transport source semantics")

    def _projection_values(
        self,
        source: ClockCoordinate,
    ) -> tuple[ClockCoordinate | None, ClockProjectionStatus, tuple[str, ...]]:
        self._validate_source_identity(source)
        if self.availability is ClockTransportAvailability.UNAVAILABLE:
            return None, ClockProjectionStatus.UNAVAILABLE, self.reason_codes
        if (
            self.valid_source_lower is not None and source.coordinate < self.valid_source_lower
        ) or (self.valid_source_upper is not None and source.coordinate > self.valid_source_upper):
            return (
                None,
                ClockProjectionStatus.UNAVAILABLE,
                ("CLOCK_SOURCE_OUT_OF_DOMAIN",),
            )
        if self.scale is None or self.offset is None:  # pragma: no cover - constructor invariant
            raise AssertionError("available transport lacks its affine mapping")
        target = ClockCoordinate(
            clock_id=self.target_clock_id,
            coordinate=(self.scale * source.coordinate) + self.offset,
            time_unit=self.target_time_unit,
            coordinate_frame=self.target_coordinate_frame,
            origin=self.target_origin,
        )
        status = (
            ClockProjectionStatus.EXACT if self.tolerance == 0 else ClockProjectionStatus.BOUNDED
        )
        return target, status, ()

    def project(self, source: ClockCoordinate) -> ClockProjection:
        """Project one coordinate without hiding unavailable-domain results."""

        target, status, reasons = self._projection_values(source)
        return ClockProjection(
            transport=self,
            source=source,
            target=target,
            status=status,
            reason_codes=reasons,
        )


@dataclass(frozen=True, slots=True)
class ClockProjection(CanonicalRecord):
    """Auditable result of applying one exact current clock transport."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/clock-projection'

    transport: ClockTransport
    source: ClockCoordinate
    target: ClockCoordinate | None
    status: ClockProjectionStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_enum_instance(self.status, ClockProjectionStatus, field_name="status")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_target, expected_status, expected_reasons = self.transport._projection_values(
            self.source
        )
        if (
            self.target != expected_target
            or self.status is not expected_status
            or self.reason_codes != expected_reasons
        ):
            raise ValueError("clock projection is not derived from its bound transport")

    def require_target(self) -> ClockCoordinate:
        if self.target is None:
            raise ValueError("clock transport is unavailable at the requested coordinate")
        return self.target

    def matches(self, observed: ClockCoordinate) -> bool:
        target = self.target
        if target is None:
            return False
        if (
            observed.clock_id != target.clock_id
            or observed.time_unit != target.time_unit
            or observed.coordinate_frame != target.coordinate_frame
            or observed.origin is not target.origin
        ):
            return False
        return abs(observed.coordinate - target.coordinate) <= self.transport.tolerance

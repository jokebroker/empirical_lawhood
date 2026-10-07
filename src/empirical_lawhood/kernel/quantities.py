"""Native-unit quantity contracts and compatibility rules."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    validate_nonempty,
    validate_stable_id,
)
from .time import AvailabilitySpec


class QuantityKind(StrEnum):
    DENOMINATOR = "DENOMINATOR"
    HISTORY = "HISTORY"
    STATE = "STATE"
    ACTION = "ACTION"
    RECEIVER = "RECEIVER"
    SINK = "SINK"
    EFFORT = "EFFORT"
    OBSERVATION = "OBSERVATION"
    RESOURCE = "RESOURCE"
    BOUNDARY = "BOUNDARY"
    UNCERTAINTY = "UNCERTAINTY"


class ResponseDirection(StrEnum):
    NOT_APPLICABLE = "NOT_APPLICABLE"
    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"
    TARGET_BAND = "TARGET_BAND"
    SIGNED_VECTOR = "SIGNED_VECTOR"


class UnitConversionPolicy(StrEnum):
    IDENTITY_ONLY = "IDENTITY_ONLY"
    EXPLICIT_ADAPTER_REQUIRED = "EXPLICIT_ADAPTER_REQUIRED"


@dataclass(frozen=True, slots=True)
class QuantitySpec(CanonicalRecord):
    """A physical/informational quantity in its claim-bearing native unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/quantity-spec'

    quantity_id: str
    label: str
    kind: QuantityKind
    dimension: str
    native_unit: str
    coordinate_frame: str
    clock_id: str
    availability: AvailabilitySpec
    response_direction: ResponseDirection = ResponseDirection.NOT_APPLICABLE
    conversion_policy: UnitConversionPolicy = UnitConversionPolicy.IDENTITY_ONLY
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_stable_id(self.clock_id, field_name="clock_id")
        for name, value in (
            ("label", self.label),
            ("dimension", self.dimension),
            ("native_unit", self.native_unit),
            ("coordinate_frame", self.coordinate_frame),
        ):
            validate_nonempty(value, field_name=name)
        if self.availability.clock_id != self.clock_id:
            raise ValueError("quantity availability must use the quantity clock")
        if self.kind in {
            QuantityKind.RECEIVER,
            QuantityKind.SINK,
            QuantityKind.EFFORT,
        }:
            if self.response_direction is ResponseDirection.NOT_APPLICABLE:
                raise ValueError("a receiver, sink or effort must declare its response direction")
        elif self.response_direction is not ResponseDirection.NOT_APPLICABLE:
            raise ValueError("response direction is valid only for receiver, sink or effort")
        require_extensions(self.extensions)

    def compatibility_errors(self, other: QuantitySpec) -> tuple[str, ...]:
        errors: list[str] = []
        if self.dimension != other.dimension:
            errors.append("DIMENSION_MISMATCH")
        if self.native_unit != other.native_unit:
            errors.append("NATIVE_UNIT_MISMATCH")
        if self.coordinate_frame != other.coordinate_frame:
            errors.append("COORDINATE_FRAME_MISMATCH")
        return tuple(errors)

    def require_compatible(self, other: QuantitySpec) -> None:
        errors = self.compatibility_errors(other)
        if errors:
            raise ValueError(
                f"quantities {self.quantity_id!r} and {other.quantity_id!r} "
                f"are incompatible: {', '.join(errors)}"
            )

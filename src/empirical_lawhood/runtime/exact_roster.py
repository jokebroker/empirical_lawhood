"""Pure exact Cartesian-product validation for issued campaign products."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class ExactProductAxis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/exact-product-axis'

    axis_id: str
    value_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.axis_id, field_name="axis_id")
        require_sorted_unique_strings(
            self.value_ids,
            field_name="value_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class ExactProductCoordinateValue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/exact-product-coordinate-value'

    axis_id: str
    value_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.axis_id, field_name="axis_id")
        validate_stable_id(self.value_id, field_name="value_id")


@dataclass(frozen=True, slots=True)
class ExactProductItem(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/exact-product-item'

    item_id: str
    coordinate: tuple[ExactProductCoordinateValue, ...]
    payload: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.item_id, field_name="item_id")
        require_sorted_unique_ids(
            self.coordinate,
            attribute="axis_id",
            field_name="coordinate",
        )


@dataclass(frozen=True, slots=True)
class ExactProductSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/exact-product-spec'

    product_id: str
    axes: tuple[ExactProductAxis, ...]
    allowed_payload_schemas: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.product_id, field_name="product_id")
        require_sorted_unique_ids(self.axes, attribute="axis_id", field_name="axes")
        if not self.axes:
            raise ValueError("exact product requires at least one axis")
        require_sorted_unique_strings(
            self.allowed_payload_schemas,
            field_name="allowed_payload_schemas",
            allow_empty=False,
        )
        for schema in self.allowed_payload_schemas:
            validate_schema(schema)


@dataclass(frozen=True, slots=True)
class ExactProductValidationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/exact-product-validation-receipt'

    receipt_id: str
    product: ObjectIdentity
    items: tuple[ObjectIdentity, ...]
    coordinate_count: int
    complete: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_ids(self.items, attribute="object_id", field_name="items")
        if self.coordinate_count < 1 or self.coordinate_count != len(self.items):
            raise ValueError("exact-product receipt count differs from its item roster")
        if not self.complete:
            raise ValueError("an exact-product validation receipt must be complete")


def _coordinate_key(
    coordinate: tuple[ExactProductCoordinateValue, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple((value.axis_id, value.value_id) for value in coordinate)


def require_exact_product(
    spec: ExactProductSpec,
    items: tuple[ExactProductItem, ...],
) -> ExactProductValidationReceipt:
    """Reject missing, duplicate, unexpected or schema-malformed product cells."""

    require_sorted_unique_ids(items, attribute="item_id", field_name="items")
    axis_ids = tuple(axis.axis_id for axis in spec.axes)
    expected = {
        tuple(zip(axis_ids, values, strict=True))
        for values in product(*(axis.value_ids for axis in spec.axes))
    }
    observed: dict[tuple[tuple[str, str], ...], ExactProductItem] = {}
    for item in items:
        key = _coordinate_key(item.coordinate)
        if tuple(value[0] for value in key) != axis_ids:
            raise ValueError("exact-product item axis roster/order differs from its spec")
        if key not in expected:
            raise ValueError("exact-product item has an unexpected coordinate")
        if key in observed:
            raise ValueError("exact-product coordinate is duplicated")
        if item.payload.object_schema not in spec.allowed_payload_schemas:
            raise ValueError("exact-product payload schema is not allowed")
        observed[key] = item
    missing = expected - set(observed)
    if missing:
        raise ValueError("exact-product coordinate roster is incomplete")
    return ExactProductValidationReceipt(
        receipt_id=f"exact-product-validation.{spec.product_id}",
        product=ObjectIdentity.from_record(spec.product_id, spec),
        items=tuple(
            sorted(
                (ObjectIdentity.from_record(value.item_id, value) for value in items),
                key=lambda value: value.object_id,
            )
        ),
        coordinate_count=len(expected),
        complete=True,
    )


__all__ = [
    "ExactProductAxis",
    "ExactProductCoordinateValue",
    "ExactProductItem",
    "ExactProductSpec",
    "ExactProductValidationReceipt",
    "require_exact_product",
]

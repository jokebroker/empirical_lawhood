"Mechanical worst-margin reduction for multi-operand admission gates."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionPlannedCoordinate
from empirical_lawhood.planning.gate_margin import GateMarginReceipt


@dataclass(frozen=True, slots=True)
class CompositeGateMarginOperandSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/geometry/composite-gate-margin-operand-spec'

    operand_id: str
    source_receipt_id: str
    normalization_scale: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.operand_id, field_name="operand_id")
        validate_stable_id(self.source_receipt_id, field_name="source_receipt_id")
        validate_decimal(
            self.normalization_scale.value,
            field_name="normalization_scale",
            minimum=Decimal(0),
        )
        if self.normalization_scale.value == 0:
            raise ValueError("composite gate normalization scale must be positive")


@dataclass(frozen=True, slots=True)
class CompositeGateMarginSpec(CanonicalRecord):
    """Frozen operand roster; it contains neither a gate verdict nor a score."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/geometry/composite-gate-margin-spec'

    spec_id: str
    gate_kind: AdmissionGateKind
    operand_specs: tuple[CompositeGateMarginOperandSpec, ...]
    reducer_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        validate_stable_id(self.reducer_id, field_name="reducer_id")
        require_sorted_unique_ids(
            self.operand_specs,
            attribute="operand_id",
            field_name="operand_specs",
        )
        if not self.operand_specs:
            raise ValueError("composite gate margin requires operands")
        require_sorted_unique_strings(
            tuple(sorted(value.source_receipt_id for value in self.operand_specs)),
            field_name="source_receipt_ids",
            allow_empty=False,
        )
        if self.reducer_id != "minimum-normalized-lower-uncertainty-bound":
            raise ValueError("composite gate margin uses another reducer")


@dataclass(frozen=True, slots=True)
class CompositeGateMarginOperand(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/geometry/composite-gate-margin-operand'

    operand_id: str
    source_receipt: ObjectIdentity
    native_lower_uncertainty_bound: NamedDecimal
    normalization_scale: NamedDecimal
    normalized_lower_uncertainty_bound: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.operand_id, field_name="operand_id")
        if self.source_receipt.object_schema != GateMarginReceipt.SCHEMA:
            raise ValueError("composite gate operand requires a certified native margin")
        if self.native_lower_uncertainty_bound.unit != self.normalization_scale.unit:
            raise ValueError("composite gate normalization changes the native unit")
        if self.normalized_lower_uncertainty_bound.unit != "1":
            raise ValueError("composite gate normalized margin must be dimensionless")
        validate_decimal(
            self.normalization_scale.value,
            field_name="normalization_scale",
            minimum=Decimal(0),
        )
        if self.normalization_scale.value == 0 or (
            self.normalized_lower_uncertainty_bound.value
            != self.native_lower_uncertainty_bound.value / self.normalization_scale.value
        ):
            raise ValueError("composite gate normalized margin is not mechanically derived")


@dataclass(frozen=True, slots=True)
class CompositeGateMarginResult(CanonicalRecord):
    """One raw minimum margin for the existing single-receipt gate route."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/geometry/composite-gate-margin-result'

    result_id: str
    spec: ObjectIdentity
    planned_coordinate: ReceiptAdmissionPlannedCoordinate
    gate_kind: AdmissionGateKind
    operands: tuple[CompositeGateMarginOperand, ...]
    minimum_normalized_lower_bound: NamedDecimal
    decisive_operand_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.decisive_operand_id, field_name="decisive_operand_id")
        if self.spec.object_schema != CompositeGateMarginSpec.SCHEMA:
            raise ValueError("composite gate result binds another spec")
        require_sorted_unique_ids(self.operands, attribute="operand_id", field_name="operands")
        if not self.operands or self.minimum_normalized_lower_bound.unit != "1":
            raise ValueError("composite gate result requires dimensionless operands")
        minimum = min(
            self.operands,
            key=lambda value: (
                value.normalized_lower_uncertainty_bound.value,
                value.operand_id,
            ),
        )
        if (
            self.minimum_normalized_lower_bound.value
            != minimum.normalized_lower_uncertainty_bound.value
            or self.decisive_operand_id != minimum.operand_id
        ):
            raise ValueError("composite gate result is not the exact worst operand")


@dataclass(frozen=True, slots=True)
class CompositeGateMarginReducer:
    """Normalize certified operands and retain their noncompensating minimum."""

    capability_key: ClassVar[str] = "geometry.composite-gate-margin"
    capability_version: ClassVar[str] = "1.0.0"

    def reduce(
        self,
        *,
        result_id: str,
        spec: CompositeGateMarginSpec,
        receipts: tuple[GateMarginReceipt, ...],
    ) -> CompositeGateMarginResult:
        by_id = {value.receipt_id: value for value in receipts}
        if len(by_id) != len(receipts) or set(by_id) != {
            value.source_receipt_id for value in spec.operand_specs
        }:
            raise ValueError("composite gate receipt roster differs from its frozen spec")
        coordinates = {value.planned_coordinate for value in receipts}
        if len(coordinates) != 1 or any(
            value.gate_kind is not spec.gate_kind for value in receipts
        ):
            raise ValueError("composite gate operands cross coordinate or gate kind")
        operands = []
        for operand_spec in spec.operand_specs:
            receipt = by_id[operand_spec.source_receipt_id]
            lower = receipt.lower_uncertainty_bound
            scale = operand_spec.normalization_scale
            if lower.unit != scale.unit:
                raise ValueError("composite gate operand normalization uses another unit")
            operands.append(
                CompositeGateMarginOperand(
                    operand_id=operand_spec.operand_id,
                    source_receipt=ObjectIdentity.from_record(receipt.receipt_id, receipt),
                    native_lower_uncertainty_bound=lower,
                    normalization_scale=scale,
                    normalized_lower_uncertainty_bound=NamedDecimal(
                        value_id=f"normalized-lower.{operand_spec.operand_id}",
                        value=lower.value / scale.value,
                        unit="1",
                    ),
                )
            )
        ordered = tuple(sorted(operands, key=lambda value: value.operand_id))
        decisive = min(
            ordered,
            key=lambda value: (
                value.normalized_lower_uncertainty_bound.value,
                value.operand_id,
            ),
        )
        return CompositeGateMarginResult(
            result_id=result_id,
            spec=ObjectIdentity.from_record(spec.spec_id, spec),
            planned_coordinate=next(iter(coordinates)),
            gate_kind=spec.gate_kind,
            operands=ordered,
            minimum_normalized_lower_bound=NamedDecimal(
                value_id=f"minimum-normalized-lower.{result_id}",
                value=decisive.normalized_lower_uncertainty_bound.value,
                unit="1",
            ),
            decisive_operand_id=decisive.operand_id,
        )


__all__ = [
    'CompositeGateMarginOperandSpec',
    'CompositeGateMarginOperand',
    'CompositeGateMarginReducer',
    'CompositeGateMarginResult',
    'CompositeGateMarginSpec',
]

"Strict certified-margin compatibility projection into unchanged raw admission."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.causal_contracts import PredicateDirection
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.serialization import (
    require_sorted_unique_ids,
    require_sorted_unique_strings,
)
from empirical_lawhood.planning.evidence_geometry import AdmissionCoordinateGateReceipt, GatePredicateKind, GatePredicateSpec, ControlledMapAdmissionReceiptCorpus, ReceiptAdmissionRawDisposition
from empirical_lawhood.planning.gate_margin import GateMarginDisposition, GateMarginReceipt


def _validate_certified_predicate(
    margin: GateMarginReceipt,
    predicate: GatePredicateSpec,
) -> None:
    if (
        predicate.gate_kind is not margin.native_predicate.gate_kind
        or predicate.receiver_id != margin.native_predicate.receiver_id
        or predicate.quantity_id != margin.native_predicate.quantity_id
        or predicate.protected_interval != margin.native_predicate.protected_interval
        or predicate.constraint_ids != margin.native_predicate.constraint_ids
        or predicate.evaluator != margin.native_predicate.evaluator
    ):
        raise ValueError("certified admission predicate rewrites its native gate semantics")
    if (
        predicate.predicate_kind is not GatePredicateKind.SCALAR_AT_LEAST
        or predicate.direction is not PredicateDirection.AT_LEAST
        or predicate.lower is None
        or predicate.lower.value != Decimal(0)
        or predicate.lower.unit != margin.native_value.unit
        or predicate.upper is not None
        or predicate.expected_boolean is not None
        or predicate.expected_identity is not None
    ):
        raise ValueError("certified admission predicate must be a native-unit zero-margin test")


def certified_admission_operand(
    margin: GateMarginReceipt,
    predicate: GatePredicateSpec,
) -> tuple[ReceiptAdmissionRawDisposition, NamedDecimal | None]:
    "Return only the signed certified operand allowed to reach raw admission."

    _validate_certified_predicate(margin, predicate)
    unit = margin.native_value.unit
    if margin.disposition is GateMarginDisposition.RAW_PASS:
        value = margin.lower_uncertainty_bound.value - margin.numeric_floor.value
        if value <= 0:  # pragma: no cover - margin receipt invariant
            raise AssertionError("RAW_PASS lost its strictly positive certificate")
        return (
            ReceiptAdmissionRawDisposition.EVALUATED,
            NamedDecimal(
                value_id=f"certified-admission-operand.{margin.receipt_id}",
                value=value,
                unit=unit,
            ),
        )
    if margin.disposition is GateMarginDisposition.RAW_FAIL:
        value = margin.upper_uncertainty_bound.value + margin.numeric_floor.value
        if value >= 0:  # pragma: no cover - margin receipt invariant
            raise AssertionError("RAW_FAIL lost its strictly negative certificate")
        return (
            ReceiptAdmissionRawDisposition.EVALUATED,
            NamedDecimal(
                value_id=f"certified-admission-operand.{margin.receipt_id}",
                value=value,
                unit=unit,
            ),
        )
    return ReceiptAdmissionRawDisposition.UNEVALUABLE, None


@dataclass(frozen=True, slots=True)
class CertifiedAdmissionMarginProjection(CanonicalRecord):
    "Proof that one native margin reached the sole unchanged admission reducer."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/certified-admission-margin-projection'

    projection_id: str
    margin_receipt: GateMarginReceipt
    admission_gate_receipt: AdmissionCoordinateGateReceipt
    proof_owner: ObjectIdentity
    certified_operand: NamedDecimal | None
    boundary_fragile: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.projection_id, field_name="projection_id")
        if self.proof_owner != self.margin_receipt.proof_owner:
            raise ValueError("certified-margin projection changes its proof owner")
        _validate_certified_predicate(
            self.margin_receipt,
            self.admission_gate_receipt.predicate,
        )
        margin = self.margin_receipt
        receipt = self.admission_gate_receipt
        if (
            receipt.planned_coordinate != margin.planned_coordinate
            or receipt.information_cutoff != margin.information_cutoff
            or receipt.outcome_access is not margin.outcome_access
            or receipt.evidence_domain != margin.evidence_domain
            or receipt.authority_boundary != margin.authority_boundary
            or receipt.resource_envelope != margin.resource_envelope
            or receipt.input_artifacts != margin.input_artifacts
            or receipt.evidence_links != margin.evidence_links
        ):
            raise ValueError("certified-margin projection breaks admission evidence continuity")
        expected_disposition, expected_operand = certified_admission_operand(
            margin,
            receipt.predicate,
        )
        if (
            self.certified_operand != expected_operand
            or receipt.disposition is not expected_disposition
            or self.boundary_fragile is not margin.boundary_fragile
        ):
            raise ValueError("certified-margin projection changes its derived truth")
        expected_suffix = f".certified-margin.{margin.receipt_id}"
        if not receipt.receipt_id.endswith(expected_suffix):
            raise ValueError("admission receipt does not bind the native margin identity")
        if expected_disposition is ReceiptAdmissionRawDisposition.EVALUATED:
            expected_status = (
                GateStatus.PASS
                if margin.disposition is GateMarginDisposition.RAW_PASS
                else GateStatus.FAIL
            )
            if (
                receipt.observed_scalar != expected_operand
                or receipt.observed_boolean is not None
                or receipt.observed_identity is not None
                or receipt.status is not expected_status
                or receipt.margin is None
                or expected_operand is None  # pragma: no cover - narrowed above
                or receipt.margin.value != expected_operand.value
                or receipt.margin.unit != expected_operand.unit
            ):
                raise ValueError("evaluated admission receipt differs from certified margin")
        elif (
            receipt.observed_scalar is not None
            or receipt.observed_boolean is not None
            or receipt.observed_identity is not None
            or receipt.status is not GateStatus.UNEVALUABLE
            or receipt.margin is not None
        ):
            raise ValueError("indeterminate margin was coerced into admission gate truth")


def bind_certified_admission_margin_projection(
    *,
    margin: GateMarginReceipt,
    admission_gate_receipt: AdmissionCoordinateGateReceipt,
) -> CertifiedAdmissionMarginProjection:
    _, operand = certified_admission_operand(margin, admission_gate_receipt.predicate)
    return CertifiedAdmissionMarginProjection(
        projection_id=f"certified-admission-margin-projection.{margin.receipt_id}",
        margin_receipt=margin,
        admission_gate_receipt=admission_gate_receipt,
        proof_owner=margin.proof_owner,
        certified_operand=operand,
        boundary_fragile=margin.boundary_fragile,
    )


@dataclass(frozen=True, slots=True)
class CertifiedAdmissionMarginCorpusUse(CanonicalRecord):
    "Exact proof that every gate in one admission corpus is margin-certified."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/certified-admission-margin-corpus-use'

    use_id: str
    proof_owner: ObjectIdentity
    admission_corpus: ControlledMapAdmissionReceiptCorpus
    projections: tuple[CertifiedAdmissionMarginProjection, ...]
    covered_gate_receipt_ids: tuple[str, ...]
    margin_receipt_fingerprints: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.use_id, field_name="use_id")
        require_sorted_unique_ids(
            self.projections,
            attribute="projection_id",
            field_name="projections",
        )
        if not self.projections:
            raise ValueError("certified admission corpus use requires margin projections")
        if any(value.proof_owner != self.proof_owner for value in self.projections):
            raise ValueError("certified admission corpus use mixes margin proof owners")
        require_sorted_unique_strings(
            self.covered_gate_receipt_ids,
            field_name="covered_gate_receipt_ids",
            allow_empty=False,
        )
        expected_receipt_ids = tuple(
            sorted(value.receipt_id for value in self.admission_corpus.gate_receipts)
        )
        if self.covered_gate_receipt_ids != expected_receipt_ids:
            raise ValueError("certified admission corpus use omits or adds a gate receipt")

        projected_receipts: dict[str, AdmissionCoordinateGateReceipt] = {}
        for projection in self.projections:
            receipt = projection.admission_gate_receipt
            prior = projected_receipts.setdefault(receipt.receipt_id, receipt)
            if prior != receipt:
                raise ValueError("certified admission projections reuse a receipt ID with different bytes")
        corpus_receipts = {value.receipt_id: value for value in self.admission_corpus.gate_receipts}
        if projected_receipts != corpus_receipts:
            raise ValueError("certified admission projection roster differs from the exact gate corpus")

        require_sorted_unique_strings(
            self.margin_receipt_fingerprints,
            field_name="margin_receipt_fingerprints",
            allow_empty=False,
        )
        expected_fingerprints = tuple(
            sorted(value.margin_receipt.fingerprint() for value in self.projections)
        )
        if self.margin_receipt_fingerprints != expected_fingerprints:
            raise ValueError("certified admission corpus use changes its margin receipt digests")


def bind_certified_admission_margin_corpus_use(
    *,
    use_id: str,
    admission_corpus: ControlledMapAdmissionReceiptCorpus,
    projections: tuple[CertifiedAdmissionMarginProjection, ...],
) -> CertifiedAdmissionMarginCorpusUse:
    ordered = tuple(sorted(projections, key=lambda value: value.projection_id))
    if not ordered:
        raise ValueError("certified admission corpus use requires margin projections")
    return CertifiedAdmissionMarginCorpusUse(
        use_id=use_id,
        proof_owner=ordered[0].proof_owner,
        admission_corpus=admission_corpus,
        projections=ordered,
        covered_gate_receipt_ids=tuple(
            sorted(value.receipt_id for value in admission_corpus.gate_receipts)
        ),
        margin_receipt_fingerprints=tuple(
            sorted(value.margin_receipt.fingerprint() for value in ordered)
        ),
    )


__all__ = [
    'CertifiedAdmissionMarginCorpusUse',
    'CertifiedAdmissionMarginProjection',
    'bind_certified_admission_margin_corpus_use',
    'bind_certified_admission_margin_projection',
    'certified_admission_operand',
]

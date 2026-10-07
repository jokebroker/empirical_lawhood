"Native gate-margin certification records for unchanged raw admission evaluation."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.admission import AdmissionGateKind
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import InformationCutoff
from empirical_lawhood.planning.evidence_geometry import GatePredicateKind, GatePredicateSpec, ReceiptAdmissionPlannedCoordinate


class GateMarginDisposition(StrEnum):
    RAW_PASS = "RAW_PASS"
    RAW_FAIL = "RAW_FAIL"
    NUMERICALLY_INDETERMINATE = "NUMERICALLY_INDETERMINATE"
    UNCERTAINTY_INDETERMINATE = "UNCERTAINTY_INDETERMINATE"


_DISPOSITION_REASONS = {
    GateMarginDisposition.RAW_PASS: (),
    GateMarginDisposition.RAW_FAIL: ("RAW_MARGIN_CERTIFIED_FAIL",),
    GateMarginDisposition.NUMERICALLY_INDETERMINATE: ("MARGIN_NUMERICALLY_INDETERMINATE",),
    GateMarginDisposition.UNCERTAINTY_INDETERMINATE: ("MARGIN_UNCERTAINTY_INDETERMINATE",),
}


@dataclass(frozen=True, slots=True)
class GateMarginProofOwner(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/gate-margin-proof-owner'

    owner_id: str
    obligation_id: str
    capability_key: str
    capability_version: str
    config_sha256: str
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.owner_id, field_name="owner_id")
        validate_stable_id(self.obligation_id, field_name="obligation_id")
        validate_stable_id(self.capability_key, field_name="capability_key")
        validate_semantic_version(self.capability_version)
        validate_sha256(self.config_sha256, field_name="config_sha256")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if self.obligation_id != "admission.gate-margin-certification":
            raise ValueError("gate-margin proof owner does not own the named obligation")


def _native_threshold_and_margin(
    predicate: GatePredicateSpec,
    observed: NamedDecimal,
) -> tuple[NamedDecimal, Decimal]:
    if observed.unit != predicate.native_unit:
        raise ValueError("native gate value uses another predicate unit")
    if predicate.predicate_kind is GatePredicateKind.SCALAR_AT_LEAST:
        if predicate.lower is None:  # pragma: no cover - predicate invariant
            raise AssertionError("at-least gate lost its lower threshold")
        return predicate.lower, observed.value - predicate.lower.value
    if predicate.predicate_kind is GatePredicateKind.SCALAR_AT_MOST:
        if predicate.upper is None:  # pragma: no cover - predicate invariant
            raise AssertionError("at-most gate lost its upper threshold")
        return predicate.upper, predicate.upper.value - observed.value
    if predicate.predicate_kind is GatePredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL:
        if predicate.lower is None or predicate.upper is None:  # pragma: no cover
            raise AssertionError("interval gate lost a threshold")
        lower_margin = observed.value - predicate.lower.value
        upper_margin = predicate.upper.value - observed.value
        if lower_margin <= upper_margin:
            return predicate.lower, lower_margin
        return predicate.upper, upper_margin
    raise ValueError("gate-margin certification requires a scalar native predicate")


def classify_gate_margin(
    *,
    lower_uncertainty_bound: Decimal,
    upper_uncertainty_bound: Decimal,
    numeric_floor: Decimal,
) -> GateMarginDisposition:
    if lower_uncertainty_bound > numeric_floor:
        return GateMarginDisposition.RAW_PASS
    if upper_uncertainty_bound < -numeric_floor:
        return GateMarginDisposition.RAW_FAIL
    if lower_uncertainty_bound >= -numeric_floor and upper_uncertainty_bound <= numeric_floor:
        return GateMarginDisposition.NUMERICALLY_INDETERMINATE
    return GateMarginDisposition.UNCERTAINTY_INDETERMINATE


@dataclass(frozen=True, slots=True)
class GateMarginReceipt(CanonicalRecord):
    """Complete native value, uncertainty and arithmetic-floor certification."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/gate-margin-receipt'

    receipt_id: str
    planned_coordinate: ReceiptAdmissionPlannedCoordinate
    native_predicate: GatePredicateSpec
    native_value: NamedDecimal
    native_threshold: NamedDecimal
    raw_signed_margin: NamedDecimal
    measurement_uncertainty: NamedDecimal
    model_uncertainty: NamedDecimal
    total_uncertainty: NamedDecimal
    lower_uncertainty_bound: NamedDecimal
    upper_uncertainty_bound: NamedDecimal
    numeric_floor: NamedDecimal
    adjusted_margin: NamedDecimal | None
    derivation: ExecutableReference
    proof_owner: ObjectIdentity
    information_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    evidence_domain: ObjectIdentity
    authority_boundary: ObjectIdentity
    resource_envelope: ObjectIdentity
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    disposition: GateMarginDisposition
    boundary_fragile: bool
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.proof_owner.object_schema != GateMarginProofOwner.SCHEMA:
            raise ValueError("gate margin requires its registered proof owner")
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
        artifacts = {value.artifact_id for value in self.input_artifacts}
        linked = {value for link in self.evidence_links for value in link.artifact_ids}
        required_payloads = {
            self.derivation.payload.artifact_id,
            self.native_predicate.evaluator.payload.artifact_id,
        }
        if not self.input_artifacts or artifacts != linked or not required_payloads <= artifacts:
            raise ValueError("gate-margin evidence omits derivation/evaluator artifacts")
        threshold, raw_margin = _native_threshold_and_margin(
            self.native_predicate,
            self.native_value,
        )
        if self.native_threshold != threshold:
            raise ValueError("gate margin binds the wrong decisive native threshold")
        unit = self.native_value.unit
        named_values = (
            self.raw_signed_margin,
            self.measurement_uncertainty,
            self.model_uncertainty,
            self.total_uncertainty,
            self.lower_uncertainty_bound,
            self.upper_uncertainty_bound,
            self.numeric_floor,
        )
        if any(value.unit != unit for value in named_values) or (
            self.adjusted_margin is not None and self.adjusted_margin.unit != unit
        ):
            raise ValueError("gate-margin quantities change native unit")
        for field_name, value in (
            ("measurement_uncertainty", self.measurement_uncertainty.value),
            ("model_uncertainty", self.model_uncertainty.value),
            ("numeric_floor", self.numeric_floor.value),
        ):
            validate_decimal(value, field_name=field_name, minimum=Decimal(0))
        total = self.measurement_uncertainty.value + self.model_uncertainty.value
        if (
            self.raw_signed_margin.value != raw_margin
            or self.total_uncertainty.value != total
            or self.lower_uncertainty_bound.value != raw_margin - total
            or self.upper_uncertainty_bound.value != raw_margin + total
        ):
            raise ValueError("gate-margin interval is not derived from native operands")
        expected_disposition = classify_gate_margin(
            lower_uncertainty_bound=self.lower_uncertainty_bound.value,
            upper_uncertainty_bound=self.upper_uncertainty_bound.value,
            numeric_floor=self.numeric_floor.value,
        )
        expected_fragile = expected_disposition in {
            GateMarginDisposition.NUMERICALLY_INDETERMINATE,
            GateMarginDisposition.UNCERTAINTY_INDETERMINATE,
        } or (
            self.adjusted_margin is not None
            and (self.adjusted_margin.value > 0) != (raw_margin > 0)
        )
        if (
            self.disposition is not expected_disposition
            or self.boundary_fragile is not expected_fragile
            or self.reason_codes != _DISPOSITION_REASONS[expected_disposition]
        ):
            raise ValueError("gate-margin disposition is not mechanically derived")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("outcome-visible evidence cannot certify admission margins")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("gate-margin evidence cannot exceed admission")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible visibility cannot certify admission margins")

    @property
    def gate_kind(self) -> AdmissionGateKind:
        return self.native_predicate.gate_kind


def certify_gate_margin(
    *,
    receipt_id: str,
    planned_coordinate: ReceiptAdmissionPlannedCoordinate,
    native_predicate: GatePredicateSpec,
    native_value: NamedDecimal,
    measurement_uncertainty: NamedDecimal,
    model_uncertainty: NamedDecimal,
    numeric_floor: NamedDecimal,
    adjusted_margin: NamedDecimal | None,
    derivation: ExecutableReference,
    proof_owner: GateMarginProofOwner,
    information_cutoff: InformationCutoff,
    outcome_access: OutcomeAccess,
    evidence_domain: ObjectIdentity,
    authority_boundary: ObjectIdentity,
    resource_envelope: ObjectIdentity,
    input_artifacts: tuple[ArtifactIdentity, ...],
    evidence_links: tuple[EvidenceLink, ...],
    visibility_ceiling: VisibilityCeiling,
) -> GateMarginReceipt:
    threshold, raw = _native_threshold_and_margin(native_predicate, native_value)
    unit = native_value.unit
    named_inputs = (
        measurement_uncertainty,
        model_uncertainty,
        numeric_floor,
    )
    if any(value.unit != unit for value in named_inputs) or (
        adjusted_margin is not None and adjusted_margin.unit != unit
    ):
        raise ValueError("gate-margin input changes native unit")
    for field_name, value in (
        ("measurement_uncertainty", measurement_uncertainty.value),
        ("model_uncertainty", model_uncertainty.value),
        ("numeric_floor", numeric_floor.value),
    ):
        validate_decimal(value, field_name=field_name, minimum=Decimal(0))
    total = measurement_uncertainty.value + model_uncertainty.value
    lower = raw - total
    upper = raw + total
    disposition = classify_gate_margin(
        lower_uncertainty_bound=lower,
        upper_uncertainty_bound=upper,
        numeric_floor=numeric_floor.value,
    )
    fragile = disposition in {
        GateMarginDisposition.NUMERICALLY_INDETERMINATE,
        GateMarginDisposition.UNCERTAINTY_INDETERMINATE,
    } or (adjusted_margin is not None and (adjusted_margin.value > 0) != (raw > 0))
    return GateMarginReceipt(
        receipt_id=receipt_id,
        planned_coordinate=planned_coordinate,
        native_predicate=native_predicate,
        native_value=native_value,
        native_threshold=threshold,
        raw_signed_margin=NamedDecimal(value_id=f"raw-margin.{receipt_id}", value=raw, unit=unit),
        measurement_uncertainty=measurement_uncertainty,
        model_uncertainty=model_uncertainty,
        total_uncertainty=NamedDecimal(
            value_id=f"total-uncertainty.{receipt_id}", value=total, unit=unit
        ),
        lower_uncertainty_bound=NamedDecimal(
            value_id=f"lower-margin.{receipt_id}", value=lower, unit=unit
        ),
        upper_uncertainty_bound=NamedDecimal(
            value_id=f"upper-margin.{receipt_id}", value=upper, unit=unit
        ),
        numeric_floor=numeric_floor,
        adjusted_margin=adjusted_margin,
        derivation=derivation,
        proof_owner=ObjectIdentity.from_record(proof_owner.owner_id, proof_owner),
        information_cutoff=information_cutoff,
        outcome_access=outcome_access,
        evidence_domain=evidence_domain,
        authority_boundary=authority_boundary,
        resource_envelope=resource_envelope,
        input_artifacts=input_artifacts,
        evidence_links=evidence_links,
        disposition=disposition,
        boundary_fragile=fragile,
        evidence_ceiling=EvidenceCeiling.ADMISSION,
        visibility_ceiling=visibility_ceiling,
        reason_codes=_DISPOSITION_REASONS[disposition],
    )


__all__ = [
    "GateMarginDisposition",
    'GateMarginProofOwner',
    'GateMarginReceipt',
    "certify_gate_margin",
    "classify_gate_margin",
]

"Adapter-local finite cohomology for observed structural classes.\n\nExact finite-decimal assignments are interpreted as rational numbers.  Their\nmathematical cocycle and image membership never depend on a numerical or\nmetrological floor.  Empirical decimal observations use only the operational\npath and cannot be promoted to an exact occupied class by tolerance.\n"

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
import json
from math import isfinite
from typing import ClassVar, Mapping, Sequence, cast

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.methods.finite_cohomology import CochainComplex, CohomologyDisposition, assess_cohomology_group
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_document_shape,
    validate_nonempty,
    validate_stable_id,
)


FloatArray = npt.NDArray[np.float64]
FractionMatrix = tuple[tuple[Fraction, ...], ...]


class AssignmentRepresentation(StrEnum):
    """Whether finite decimal text denotes mathematics or a measurement."""

    EXACT_FINITE_DECIMAL = "EXACT_FINITE_DECIMAL"
    EMPIRICAL_DECIMAL_OBSERVATION = "EMPIRICAL_DECIMAL_OBSERVATION"


class ExactClassStatus(StrEnum):
    EXACT_NONCOCYCLE = "EXACT_NONCOCYCLE"
    EXACT_COBOUNDARY = "EXACT_COBOUNDARY"
    EXACT_ZERO_UNOCCUPIED = "EXACT_ZERO_UNOCCUPIED"
    EXACT_NONTRIVIAL_H1_CLASS = "EXACT_NONTRIVIAL_H1_CLASS"
    NOT_APPLICABLE_EMPIRICAL = "NOT_APPLICABLE_EMPIRICAL"


class OperationalMateriality(StrEnum):
    MATERIAL = "MATERIAL"
    SUBMATERIAL = "SUBMATERIAL"


@dataclass(frozen=True, slots=True)
class CochainAssignment(CanonicalRecord):
    """One declared exact or empirical finite-decimal one-cochain."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/cochain-assignment'

    assignment_id: str
    complex_id: str
    coefficients: tuple[Decimal, ...]
    representation: AssignmentRepresentation
    native_unit: str
    evidence_world: str

    def __post_init__(self) -> None:
        validate_stable_id(self.assignment_id, field_name="assignment_id")
        validate_stable_id(self.complex_id, field_name="complex_id")
        if not self.coefficients:
            raise ValueError("cochain assignment requires coefficients")
        for value in self.coefficients:
            validate_decimal(value, field_name="coefficients")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_nonempty(self.evidence_world, field_name="evidence_world")


@dataclass(frozen=True, slots=True)
class ExactClassAssessment(CanonicalRecord):
    """Floor-independent exact membership result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/exact-class-assessment'

    assessment_id: str
    complex_id: str
    assignment_id: str
    status: ExactClassStatus
    exact_cocycle: bool | None
    exact_coboundary: bool | None
    occupied_h1_class: bool | None
    h1_dimension: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("complex_id", self.complex_id),
            ("assignment_id", self.assignment_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.h1_dimension < 0:
            raise ValueError("h1_dimension must be nonnegative")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        empirical = self.status is ExactClassStatus.NOT_APPLICABLE_EMPIRICAL
        if empirical != (
            self.exact_cocycle is None
            and self.exact_coboundary is None
            and self.occupied_h1_class is None
        ):
            raise ValueError("empirical and exact membership fields disagree")
        if self.status is ExactClassStatus.EXACT_NONTRIVIAL_H1_CLASS:
            if not (
                self.exact_cocycle is True
                and self.exact_coboundary is False
                and self.occupied_h1_class is True
                and self.h1_dimension > 0
            ):
                raise ValueError("occupied exact class prerequisites differ")
        if self.status is ExactClassStatus.EXACT_NONCOCYCLE:
            if self.exact_cocycle is not False or self.occupied_h1_class is not False:
                raise ValueError("exact noncocycle fields disagree")


@dataclass(frozen=True, slots=True)
class OperationalAssignmentAssessment(CanonicalRecord):
    """Numerical distances and scientific materiality, separate from exact status."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/operational-assignment-assessment'

    assessment_id: str
    complex_id: str
    assignment_id: str
    assignment_norm: Decimal
    cocycle_defect: Decimal
    distance_to_coboundary: Decimal
    consistency_floor: Decimal
    materiality_floor: Decimal
    approximately_cocyclic: bool
    approximately_globalizable: bool
    assignment_materiality: OperationalMateriality
    residual_material: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("complex_id", self.complex_id),
            ("assignment_id", self.assignment_id),
        ):
            validate_stable_id(value, field_name=name)
        for decimal_name, decimal_value in (
            ("assignment_norm", self.assignment_norm),
            ("cocycle_defect", self.cocycle_defect),
            ("distance_to_coboundary", self.distance_to_coboundary),
            ("consistency_floor", self.consistency_floor),
            ("materiality_floor", self.materiality_floor),
        ):
            validate_decimal(
                decimal_value,
                field_name=decimal_name,
                minimum=Decimal("0"),
            )
        if self.consistency_floor == 0 or self.materiality_floor == 0:
            raise ValueError("operational floors must be positive")
        if self.materiality_floor < self.consistency_floor:
            raise ValueError("materiality floor must not be below consistency floor")
        if self.approximately_globalizable and not self.approximately_cocyclic:
            raise ValueError("approximate globalization requires an approximate cocycle")
        expected_materiality = (
            OperationalMateriality.MATERIAL
            if self.assignment_norm > self.materiality_floor
            else OperationalMateriality.SUBMATERIAL
        )
        if self.assignment_materiality is not expected_materiality:
            raise ValueError("assignment materiality and floor disagree")
        if self.residual_material != (self.distance_to_coboundary > self.materiality_floor):
            raise ValueError("residual materiality and floor disagree")


@dataclass(frozen=True, slots=True)
class AssignmentAssessment(CanonicalRecord):
    """Joint record whose exact and operational axes remain orthogonal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/assignment-assessment'

    assessment_id: str
    exact: ExactClassAssessment
    operational: OperationalAssignmentAssessment
    disposition: CohomologyDisposition
    exact_direct_globalization: bool | None
    approximate_direct_globalization: bool
    chosen_witness_value: Decimal | None
    chosen_witness_corroborates: bool | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        if self.exact.assignment_id != self.operational.assignment_id:
            raise ValueError("exact and operational assignment identities differ")
        if self.exact.complex_id != self.operational.complex_id:
            raise ValueError("exact and operational complex identities differ")
        if self.chosen_witness_value is not None:
            validate_decimal(self.chosen_witness_value, field_name="chosen_witness_value")
        if (self.chosen_witness_value is None) != (self.chosen_witness_corroborates is None):
            raise ValueError("chosen-witness fields must be jointly present or absent")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is CohomologyDisposition.PARTICULAR_H1_OBSTRUCTION_SUPPORTED:
            if self.exact.status is not ExactClassStatus.EXACT_NONTRIVIAL_H1_CLASS:
                raise ValueError("particular H1 disposition lacks exact membership")
            if self.exact_direct_globalization is not False:
                raise ValueError("occupied exact class cannot globalize directly")


@dataclass(frozen=True, slots=True)
class ExactAndOperationalStateEnrichmentAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/exact-and-operational-state-enrichment-assessment'

    assessment_id: str
    omitted_assignment_id: str
    enriched_assignment_id: str
    omitted_status: ExactClassStatus
    enriched_status: ExactClassStatus
    omitted_distance: Decimal
    enriched_distance: Decimal
    exact_resolution: bool
    operational_resolution: bool
    disposition: CohomologyDisposition

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("omitted_assignment_id", self.omitted_assignment_id),
            ("enriched_assignment_id", self.enriched_assignment_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.omitted_distance, field_name="omitted_distance", minimum=Decimal("0"))
        validate_decimal(
            self.enriched_distance,
            field_name="enriched_distance",
            minimum=Decimal("0"),
        )
        if self.disposition is CohomologyDisposition.STATE_ONTOLOGY_DEFECT_RESOLVED:
            if not (self.exact_resolution or self.operational_resolution):
                raise ValueError("state-enrichment resolution lacks evidence")


def _fraction_rank(matrix: FractionMatrix, *, columns: int) -> int:
    if not matrix or columns == 0:
        return 0
    work = [list(row) for row in matrix]
    rank = 0
    for column in range(columns):
        pivot = next((row for row in range(rank, len(work)) if work[row][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][column]
        work[rank] = [value / scale for value in work[rank]]
        for row in range(len(work)):
            if row == rank or not work[row][column]:
                continue
            factor = work[row][column]
            work[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(work[row], work[rank], strict=True)
            ]
        rank += 1
        if rank == len(work):
            break
    return rank


def _exact_membership(
    complex_: CochainComplex,
    coefficients: tuple[Decimal, ...],
) -> tuple[bool, bool]:
    vector = tuple(Fraction(value) for value in coefficients)
    cocycle = all(
        sum(Fraction(coefficient) * vector[index] for index, coefficient in enumerate(row)) == 0
        for row in complex_.delta1
    )
    image: FractionMatrix = tuple(
        tuple(Fraction(value) for value in row) for row in complex_.delta0
    )
    augmented: FractionMatrix = tuple((*row, vector[index]) for index, row in enumerate(image))
    image_rank = _fraction_rank(image, columns=complex_.c0_dimension)
    augmented_rank = _fraction_rank(augmented, columns=complex_.c0_dimension + 1)
    return cocycle, image_rank == augmented_rank


def _float_matrix(
    matrix: tuple[tuple[int, ...], ...],
    *,
    rows: int,
    columns: int,
) -> FloatArray:
    if rows == 0:
        return np.zeros((0, columns), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64).reshape(rows, columns)


def _decimal(value: float) -> Decimal:
    if not isfinite(value):
        raise ValueError("nonfinite scientific value")
    return Decimal(str(value))


def _distance_to_image(matrix: FloatArray, vector: FloatArray) -> float:
    if matrix.shape[1] == 0:
        return float(np.linalg.norm(vector))
    solution, _residuals, _rank, _singular = np.linalg.lstsq(matrix, vector, rcond=None)
    return float(np.linalg.norm(vector - matrix @ solution))


def assess_assignment(
    complex_: CochainComplex,
    assignment: CochainAssignment,
    *,
    consistency_floor: Decimal,
    materiality_floor: Decimal,
    chosen_witness: Sequence[Decimal] | None = None,
) -> AssignmentAssessment:
    """Assess exact class membership and operational materiality independently."""

    if not complex_.restriction_system_valid:
        raise ValueError("invalid restrictions cannot define a cohomology group")
    if assignment.complex_id != complex_.complex_id:
        raise ValueError("assignment and complex identities differ")
    if len(assignment.coefficients) != complex_.c1_dimension:
        raise ValueError("assignment dimension differs")
    validate_decimal(consistency_floor, field_name="consistency_floor", minimum=Decimal("0"))
    validate_decimal(materiality_floor, field_name="materiality_floor", minimum=Decimal("0"))
    if consistency_floor == 0 or materiality_floor < consistency_floor:
        raise ValueError("operational floors differ")

    group = assess_cohomology_group(complex_)
    vector = np.asarray([float(value) for value in assignment.coefficients], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise ValueError("assignment contains nonfinite values")
    delta0 = _float_matrix(
        complex_.delta0,
        rows=complex_.c1_dimension,
        columns=complex_.c0_dimension,
    )
    delta1 = _float_matrix(
        complex_.delta1,
        rows=complex_.c2_dimension,
        columns=complex_.c1_dimension,
    )
    norm = float(np.linalg.norm(vector))
    cocycle_defect = float(np.linalg.norm(delta1 @ vector))
    distance = _distance_to_image(delta0, vector)
    approximate_cocycle = cocycle_defect <= float(consistency_floor)
    approximate_global = approximate_cocycle and distance <= float(consistency_floor)

    exact_cocycle: bool | None
    exact_coboundary: bool | None
    occupied: bool | None
    exact_global: bool | None
    if assignment.representation is AssignmentRepresentation.EXACT_FINITE_DECIMAL:
        exact_cocycle, exact_coboundary = _exact_membership(complex_, assignment.coefficients)
        zero = all(value == 0 for value in assignment.coefficients)
        if not exact_cocycle:
            exact_status = ExactClassStatus.EXACT_NONCOCYCLE
            occupied = False
            exact_global = False
            disposition = CohomologyDisposition.NONCOCYCLE_INVALID
            exact_reasons = ("EXACT_COCYCLE_CONDITION_FAILED",)
        elif exact_coboundary:
            occupied = False
            exact_global = True
            if zero and group.h1_dimension > 0:
                exact_status = ExactClassStatus.EXACT_ZERO_UNOCCUPIED
                disposition = CohomologyDisposition.NONZERO_GROUP_NO_OCCUPIED_CLASS
                exact_reasons = ("EXACT_GROUP_NONZERO_ASSIGNMENT_ZERO",)
            else:
                exact_status = ExactClassStatus.EXACT_COBOUNDARY
                disposition = CohomologyDisposition.COBOUNDARY_RESOLVED
                exact_reasons = ("EXACT_ZERO_COCHAIN_TRIVIALIZES_ASSIGNMENT",)
        else:
            exact_status = ExactClassStatus.EXACT_NONTRIVIAL_H1_CLASS
            occupied = True
            exact_global = False
            disposition = CohomologyDisposition.PARTICULAR_H1_OBSTRUCTION_SUPPORTED
            exact_reasons = ("EXACT_COCYCLE_NONTRIVIAL_MODULO_COBOUNDARY",)
    else:
        exact_status = ExactClassStatus.NOT_APPLICABLE_EMPIRICAL
        exact_cocycle = None
        exact_coboundary = None
        occupied = None
        exact_global = None
        if not approximate_cocycle:
            disposition = CohomologyDisposition.NONCOCYCLE_INVALID
            exact_reasons = ("EMPIRICAL_COCYCLE_DEFECT_EXCEEDS_FLOOR",)
        else:
            disposition = CohomologyDisposition.APPROXIMATE_INCONSISTENCY_ONLY
            exact_reasons = ("EMPIRICAL_ASSIGNMENT_NO_EXACT_PROMOTION",)

    exact = ExactClassAssessment(
        assessment_id=f"exact.{assignment.assignment_id}",
        complex_id=complex_.complex_id,
        assignment_id=assignment.assignment_id,
        status=exact_status,
        exact_cocycle=exact_cocycle,
        exact_coboundary=exact_coboundary,
        occupied_h1_class=occupied,
        h1_dimension=group.h1_dimension,
        reason_codes=tuple(sorted(exact_reasons)),
    )
    operational = OperationalAssignmentAssessment(
        assessment_id=f"operational.{assignment.assignment_id}",
        complex_id=complex_.complex_id,
        assignment_id=assignment.assignment_id,
        assignment_norm=_decimal(norm),
        cocycle_defect=_decimal(cocycle_defect),
        distance_to_coboundary=_decimal(distance),
        consistency_floor=consistency_floor,
        materiality_floor=materiality_floor,
        approximately_cocyclic=approximate_cocycle,
        approximately_globalizable=approximate_global,
        assignment_materiality=(
            OperationalMateriality.MATERIAL
            if norm > float(materiality_floor)
            else OperationalMateriality.SUBMATERIAL
        ),
        residual_material=distance > float(materiality_floor),
    )

    witness_value: Decimal | None = None
    witness_corroborates: bool | None = None
    reasons = set(exact_reasons)
    if chosen_witness is not None:
        if len(chosen_witness) != complex_.c1_dimension:
            raise ValueError("chosen witness dimension differs")
        for value in chosen_witness:
            validate_decimal(value, field_name="chosen_witness")
        witness_value = sum(
            (
                coefficient * witness
                for coefficient, witness in zip(
                    assignment.coefficients, chosen_witness, strict=True
                )
            ),
            start=Decimal("0"),
        )
        witness_corroborates = occupied is True and witness_value != 0
        reasons.add(
            "CHOSEN_WITNESS_CORROBORATES"
            if witness_corroborates
            else "CHOSEN_WITNESS_NOT_DEFINITIVE"
        )

    return AssignmentAssessment(
        assessment_id=f"assignment-assessment.{assignment.assignment_id}",
        exact=exact,
        operational=operational,
        disposition=disposition,
        exact_direct_globalization=exact_global,
        approximate_direct_globalization=approximate_global,
        chosen_witness_value=witness_value,
        chosen_witness_corroborates=witness_corroborates,
        reason_codes=tuple(sorted(reasons)),
    )


def compare_state_enrichment(
    omitted: AssignmentAssessment,
    enriched: AssignmentAssessment,
    *,
    assessment_id: str,
) -> ExactAndOperationalStateEnrichmentAssessment:
    """Classify an enriched chart without conflating exact and operational rescue."""

    exact_resolution = (
        omitted.exact.status is ExactClassStatus.EXACT_NONTRIVIAL_H1_CLASS
        and enriched.exact.status
        in {ExactClassStatus.EXACT_COBOUNDARY, ExactClassStatus.EXACT_ZERO_UNOCCUPIED}
    )
    operational_resolution = (
        omitted.operational.residual_material and enriched.operational.approximately_globalizable
    )
    disposition = (
        CohomologyDisposition.STATE_ONTOLOGY_DEFECT_RESOLVED
        if exact_resolution or operational_resolution
        else CohomologyDisposition.APPROXIMATE_INCONSISTENCY_ONLY
    )
    return ExactAndOperationalStateEnrichmentAssessment(
        assessment_id=assessment_id,
        omitted_assignment_id=omitted.exact.assignment_id,
        enriched_assignment_id=enriched.exact.assignment_id,
        omitted_status=omitted.exact.status,
        enriched_status=enriched.exact.status,
        omitted_distance=omitted.operational.distance_to_coboundary,
        enriched_distance=enriched.operational.distance_to_coboundary,
        exact_resolution=exact_resolution,
        operational_resolution=operational_resolution,
        disposition=disposition,
    )


def _decimal_from_document(value: object) -> Decimal:
    if not isinstance(value, Mapping) or set(value) != {"decimal"}:
        raise ValueError("canonical decimal envelope differs")
    raw = value["decimal"]
    if not isinstance(raw, str):
        raise ValueError("canonical decimal text differs")
    result = Decimal(raw)
    validate_decimal(result, field_name="canonical_decimal")
    return result


def decode_cochain_assignment(payload: bytes) -> CochainAssignment:
    "Strict external decoder for the exact/empirical boundary."

    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cochain-assignment payload is not JSON") from error
    if not isinstance(document, Mapping):
        raise ValueError("cochain-assignment document is not a mapping")
    raw = validate_document_shape(
        cast(Mapping[str, object], document),
        expected_schema=CochainAssignment.SCHEMA,
        expected_version=CochainAssignment.VERSION,
        field_names=frozenset(field.name for field in fields(CochainAssignment)),
    )
    coefficients_raw = raw["coefficients"]
    if not isinstance(coefficients_raw, list):
        raise ValueError("cochain-assignment coefficients differ")
    return CochainAssignment(
        assignment_id=str(raw["assignment_id"]),
        complex_id=str(raw["complex_id"]),
        coefficients=tuple(_decimal_from_document(value) for value in coefficients_raw),
        representation=AssignmentRepresentation(str(raw["representation"])),
        native_unit=str(raw["native_unit"]),
        evidence_world=str(raw["evidence_world"]),
    )


__all__ = [
    'AssignmentAssessment',
    'AssignmentRepresentation',
    'CochainAssignment',
    'ExactClassAssessment',
    'ExactClassStatus',
    'OperationalAssignmentAssessment',
    'OperationalMateriality',
    'ExactAndOperationalStateEnrichmentAssessment',
    'assess_assignment',
    'compare_state_enrichment',
    'decode_cochain_assignment',
]

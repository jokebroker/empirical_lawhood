"""Finite, plan-scoped response globalization and cohomology methods.

The implementation is deliberately restricted to finite additive real
coefficient systems.  It keeps assignment inconsistency, direct globalization,
group dimensions and one occupied particular class as separate records.  Exact
integer incidence is checked with rational arithmetic; empirical assignments
are evaluated numerically against frozen floors.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import StrEnum
from fractions import Fraction
import json
from math import isfinite
from typing import ClassVar, Sequence, cast

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)


FloatArray = npt.NDArray[np.float64]
IntMatrix = tuple[tuple[int, ...], ...]


class CohomologyDisposition(StrEnum):
    COMPATIBLE_GLOBALIZATION = "COMPATIBLE_GLOBALIZATION"
    APPROXIMATE_INCONSISTENCY_ONLY = "APPROXIMATE_INCONSISTENCY_ONLY"
    COBOUNDARY_RESOLVED = "COBOUNDARY_RESOLVED"
    PARTICULAR_H1_OBSTRUCTION_SUPPORTED = "PARTICULAR_H1_OBSTRUCTION_SUPPORTED"
    NONZERO_GROUP_NO_OCCUPIED_CLASS = "NONZERO_GROUP_NO_OCCUPIED_CLASS"
    NONCOCYCLE_INVALID = "NONCOCYCLE_INVALID"
    STATE_ONTOLOGY_DEFECT_RESOLVED = "STATE_ONTOLOGY_DEFECT_RESOLVED"
    WITNESS_INCOMPLETE = "WITNESS_INCOMPLETE"
    DEGREE_TWO_EXPLORATORY = "DEGREE_TWO_EXPLORATORY"
    COHOMOLOGY_INVALID = "COHOMOLOGY_INVALID"


def _validate_matrix(
    matrix: IntMatrix,
    *,
    rows: int,
    columns: int,
    field_name: str,
) -> None:
    if rows < 0 or columns < 0 or len(matrix) != rows:
        raise ValueError(f"{field_name} row count differs")
    if any(len(row) != columns for row in matrix):
        raise ValueError(f"{field_name} column count differs")
    if any(not isinstance(value, int) for row in matrix for value in row):
        raise ValueError(f"{field_name} must contain exact integers")


def _exact_product(left: IntMatrix, right: IntMatrix, inner: int) -> IntMatrix:
    if not left:
        return ()
    columns = len(right[0]) if right else 0
    return tuple(
        tuple(sum(left_value[k] * right[k][j] for k in range(inner)) for j in range(columns))
        for left_value in left
    )


def _fraction_rank_values(
    matrix: Sequence[Sequence[Fraction]],
    *,
    columns: int,
) -> int:
    if not matrix or columns == 0:
        return 0
    work = [[Fraction(value) for value in row] for row in matrix]
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


def _fraction_rank(matrix: IntMatrix, *, columns: int) -> int:
    return _fraction_rank_values(
        tuple(tuple(Fraction(value) for value in row) for row in matrix),
        columns=columns,
    )


def _exact_assignment_status(
    complex_: CochainComplex,
    assignment: Sequence[float | Decimal],
) -> tuple[bool, bool] | None:
    if any(not isinstance(value, Decimal) for value in assignment):
        return None
    vector = tuple(Fraction(value) for value in assignment)
    cocycle = all(
        sum(Fraction(coefficient) * vector[index] for index, coefficient in enumerate(row))
        == 0
        for row in complex_.delta1
    )
    image = tuple(
        tuple(Fraction(value) for value in row) for row in complex_.delta0
    )
    augmented = tuple((*row, vector[index]) for index, row in enumerate(image))
    image_rank = _fraction_rank_values(image, columns=complex_.c0_dimension)
    augmented_rank = _fraction_rank_values(
        augmented,
        columns=complex_.c0_dimension + 1,
    )
    return cocycle, image_rank == augmented_rank


def _as_float_matrix(matrix: IntMatrix, *, rows: int, columns: int) -> FloatArray:
    if rows == 0:
        return np.zeros((0, columns), dtype=np.float64)
    return np.asarray(matrix, dtype=np.float64).reshape(rows, columns)


def _as_vector(values: Sequence[float | Decimal], dimension: int) -> FloatArray:
    vector = np.asarray([float(value) for value in values], dtype=np.float64)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise ValueError("cochain vector dimension or finiteness differs")
    return vector


def _decimal(value: float) -> Decimal:
    if not isfinite(value):
        raise ValueError("nonfinite scientific value")
    return Decimal(str(value))


@dataclass(frozen=True, slots=True)
class CochainComplex(CanonicalRecord):
    """One finite cochain complex with exact integer coboundaries."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/cochain-complex'

    complex_id: str
    coefficient_system_id: str
    c0_dimension: int
    c1_dimension: int
    c2_dimension: int
    delta0: IntMatrix
    delta1: IntMatrix
    restriction_system_valid: bool
    rank_floor: Decimal
    evidence_world: str

    def __post_init__(self) -> None:
        validate_stable_id(self.complex_id, field_name="complex_id")
        validate_stable_id(self.coefficient_system_id, field_name="coefficient_system_id")
        if min(self.c0_dimension, self.c1_dimension, self.c2_dimension) < 0:
            raise ValueError("cochain dimensions must be nonnegative")
        _validate_matrix(
            self.delta0,
            rows=self.c1_dimension,
            columns=self.c0_dimension,
            field_name="delta0",
        )
        _validate_matrix(
            self.delta1,
            rows=self.c2_dimension,
            columns=self.c1_dimension,
            field_name="delta1",
        )
        validate_decimal(self.rank_floor, field_name="rank_floor", minimum=Decimal("0"))
        if self.rank_floor == 0:
            raise ValueError("rank floor must be positive")
        validate_nonempty(self.evidence_world, field_name="evidence_world")
        if self.restriction_system_valid:
            product = _exact_product(self.delta1, self.delta0, self.c1_dimension)
            if any(value != 0 for row in product for value in row):
                raise ValueError("exact cochain identity delta1*delta0 differs from zero")


@dataclass(frozen=True, slots=True)
class CohomologyGroupAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/cohomology-group-assessment'

    assessment_id: str
    complex_id: str
    delta0_rank: int
    delta1_rank: int
    h0_dimension: int
    h1_dimension: int
    h2_dimension: int
    d_squared_zero: bool
    rank_stability: tuple[tuple[Decimal, int, int], ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.complex_id, field_name="complex_id")
        if min(
            self.delta0_rank,
            self.delta1_rank,
            self.h0_dimension,
            self.h1_dimension,
            self.h2_dimension,
        ) < 0:
            raise ValueError("cohomology ranks and dimensions must be nonnegative")
        if not self.d_squared_zero:
            raise ValueError("a valid cohomology assessment requires delta squared zero")
        if not self.rank_stability:
            raise ValueError("rank stability curve is required")


@dataclass(frozen=True, slots=True)
class ConsistencyAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/consistency-assessment'

    assessment_id: str
    complex_id: str
    assignment_id: str
    cocycle_defect: Decimal
    distance_to_coboundary: Decimal
    assignment_norm: Decimal
    consistency_floor: Decimal

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("complex_id", self.complex_id),
            ("assignment_id", self.assignment_id),
        ):
            validate_stable_id(value, field_name=name)
        for decimal_name, decimal_value in (
            ("cocycle_defect", self.cocycle_defect),
            ("distance_to_coboundary", self.distance_to_coboundary),
            ("assignment_norm", self.assignment_norm),
            ("consistency_floor", self.consistency_floor),
        ):
            validate_decimal(decimal_value, field_name=decimal_name, minimum=Decimal("0"))


@dataclass(frozen=True, slots=True)
class GlobalSectionAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/global-section-assessment'

    assessment_id: str
    complex_id: str
    assignment_id: str
    exact_feasible: bool
    approximate_feasible: bool
    minimum_consistency_radius: Decimal
    zero_cochain_solution: tuple[Decimal, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("complex_id", self.complex_id),
            ("assignment_id", self.assignment_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(
            self.minimum_consistency_radius,
            field_name="minimum_consistency_radius",
            minimum=Decimal("0"),
        )
        if self.exact_feasible and not self.approximate_feasible:
            raise ValueError("exact globalization must also be approximately feasible")


@dataclass(frozen=True, slots=True)
class ParticularObstructionWitness(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/particular-obstruction-witness'

    witness_id: str
    complex_id: str
    assignment_id: str
    disposition: CohomologyDisposition
    representative: tuple[Decimal, ...]
    occupied_class_dimension: int
    cocycle_valid: bool
    gauge_trivial: bool | None
    chosen_witness_value: Decimal | None
    direct_globalization_agrees: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("witness_id", self.witness_id),
            ("complex_id", self.complex_id),
            ("assignment_id", self.assignment_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.occupied_class_dimension < 0:
            raise ValueError("occupied class dimension must be nonnegative")
        if self.chosen_witness_value is not None:
            validate_decimal(self.chosen_witness_value, field_name="chosen_witness_value")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is CohomologyDisposition.PARTICULAR_H1_OBSTRUCTION_SUPPORTED:
            if not self.cocycle_valid or self.gauge_trivial is not False:
                raise ValueError("nontrivial class requires a valid nontrivial cocycle")
            if not self.direct_globalization_agrees or self.occupied_class_dimension < 1:
                raise ValueError("particular class lacks direct globalization agreement")
        if self.disposition is CohomologyDisposition.NONCOCYCLE_INVALID and self.cocycle_valid:
            raise ValueError("noncocycle disposition and cocycle validity disagree")


@dataclass(frozen=True, slots=True)
class NumericalStateEnrichmentAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/numerical-state-enrichment-assessment'

    assessment_id: str
    omitted_complex_id: str
    enriched_complex_id: str
    omitted_distance: Decimal
    enriched_distance: Decimal
    disposition: CohomologyDisposition

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("omitted_complex_id", self.omitted_complex_id),
            ("enriched_complex_id", self.enriched_complex_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(self.omitted_distance, field_name="omitted_distance", minimum=Decimal("0"))
        validate_decimal(self.enriched_distance, field_name="enriched_distance", minimum=Decimal("0"))
        if self.disposition is CohomologyDisposition.STATE_ONTOLOGY_DEFECT_RESOLVED:
            if not self.enriched_distance < self.omitted_distance:
                raise ValueError("state enrichment did not reduce the obstruction")


@dataclass(frozen=True, slots=True)
class RefinementClassAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/refinement-class-assessment'

    assessment_id: str
    source_complex_id: str
    refined_complex_id: str
    residual_modulo_coboundary: Decimal
    stable: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("assessment_id", self.assessment_id),
            ("source_complex_id", self.source_complex_id),
            ("refined_complex_id", self.refined_complex_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_decimal(
            self.residual_modulo_coboundary,
            field_name="residual_modulo_coboundary",
            minimum=Decimal("0"),
        )


def assess_cohomology_group(complex_: CochainComplex) -> CohomologyGroupAssessment:
    if not complex_.restriction_system_valid:
        raise ValueError("invalid restrictions cannot define a cohomology group")
    rank0 = _fraction_rank(complex_.delta0, columns=complex_.c0_dimension)
    rank1 = _fraction_rank(complex_.delta1, columns=complex_.c1_dimension)
    h0 = complex_.c0_dimension - rank0
    h1 = complex_.c1_dimension - rank0 - rank1
    h2 = complex_.c2_dimension - rank1
    if min(h0, h1, h2) < 0:
        raise ValueError("cochain ranks violate dimension bounds")
    delta0 = _as_float_matrix(
        complex_.delta0,
        rows=complex_.c1_dimension,
        columns=complex_.c0_dimension,
    )
    delta1 = _as_float_matrix(
        complex_.delta1,
        rows=complex_.c2_dimension,
        columns=complex_.c1_dimension,
    )
    floors = (complex_.rank_floor / 10, complex_.rank_floor, complex_.rank_floor * 10)
    stability = tuple(
        (
            floor,
            int(np.linalg.matrix_rank(delta0, tol=float(floor))),
            int(np.linalg.matrix_rank(delta1, tol=float(floor))),
        )
        for floor in floors
    )
    return CohomologyGroupAssessment(
        assessment_id=f"group.{complex_.complex_id}",
        complex_id=complex_.complex_id,
        delta0_rank=rank0,
        delta1_rank=rank1,
        h0_dimension=h0,
        h1_dimension=h1,
        h2_dimension=h2,
        d_squared_zero=True,
        rank_stability=stability,
    )


def _distance_to_image(matrix: FloatArray, vector: FloatArray) -> tuple[float, FloatArray]:
    if matrix.shape[0] != vector.size:
        raise ValueError("image matrix and vector dimensions differ")
    if matrix.shape[1] == 0:
        return float(np.linalg.norm(vector)), np.zeros(0, dtype=np.float64)
    solution, _residuals, _rank, _singular = np.linalg.lstsq(matrix, vector, rcond=None)
    residual = vector - matrix @ solution
    return float(np.linalg.norm(residual)), np.asarray(solution, dtype=np.float64)


def assess_assignment(
    complex_: CochainComplex,
    assignment: Sequence[float | Decimal],
    *,
    assignment_id: str,
    consistency_floor: Decimal,
    chosen_witness: Sequence[float | Decimal] | None = None,
) -> tuple[
    ConsistencyAssessment,
    GlobalSectionAssessment,
    CohomologyGroupAssessment,
    ParticularObstructionWitness,
]:
    validate_stable_id(assignment_id, field_name="assignment_id")
    validate_decimal(consistency_floor, field_name="consistency_floor", minimum=Decimal("0"))
    if consistency_floor == 0:
        raise ValueError("consistency floor must be positive")
    if not complex_.restriction_system_valid:
        raise ValueError("invalid restrictions cannot be assigned a cohomology class")
    vector = _as_vector(assignment, complex_.c1_dimension)
    delta0 = _as_float_matrix(
        complex_.delta0,
        rows=complex_.c1_dimension,
        columns=complex_.c0_dimension,
    )
    delta1 = _as_float_matrix(
        complex_.delta1,
        rows=complex_.c2_dimension,
        columns=complex_.c1_dimension,
    )
    cocycle_defect = float(np.linalg.norm(delta1 @ vector))
    distance, solution = _distance_to_image(delta0, vector)
    norm = float(np.linalg.norm(vector))
    exact_floor = float(complex_.rank_floor)
    approximate_floor = float(consistency_floor)
    exact_status = _exact_assignment_status(complex_, assignment)
    group = assess_cohomology_group(complex_)
    consistency = ConsistencyAssessment(
        assessment_id=f"consistency.{assignment_id}",
        complex_id=complex_.complex_id,
        assignment_id=assignment_id,
        cocycle_defect=_decimal(cocycle_defect),
        distance_to_coboundary=_decimal(distance),
        assignment_norm=_decimal(norm),
        consistency_floor=consistency_floor,
    )
    exact_feasible = (
        exact_status[0] and exact_status[1]
        if exact_status is not None
        else cocycle_defect <= exact_floor and distance <= exact_floor
    )
    approximate_feasible = cocycle_defect <= approximate_floor and distance <= approximate_floor
    globalization = GlobalSectionAssessment(
        assessment_id=f"global.{assignment_id}",
        complex_id=complex_.complex_id,
        assignment_id=assignment_id,
        exact_feasible=exact_feasible,
        approximate_feasible=approximate_feasible,
        minimum_consistency_radius=_decimal(max(cocycle_defect, distance)),
        zero_cochain_solution=tuple(_decimal(float(value)) for value in solution),
    )
    witness_value: Decimal | None = None
    witness_incomplete = False
    if chosen_witness is not None:
        witness = _as_vector(chosen_witness, complex_.c1_dimension)
        witness_float = float(witness @ vector)
        witness_value = _decimal(witness_float)
        witness_incomplete = abs(witness_float) <= exact_floor and distance > approximate_floor
    reason_codes: tuple[str, ...]
    if cocycle_defect > approximate_floor:
        disposition = CohomologyDisposition.NONCOCYCLE_INVALID
        gauge_trivial: bool | None = None
        occupied_dimension = 0
        reason_codes = ("COCYCLE_CONDITION_FAILED",)
    elif norm <= exact_floor:
        disposition = (
            CohomologyDisposition.NONZERO_GROUP_NO_OCCUPIED_CLASS
            if group.h1_dimension > 0
            else CohomologyDisposition.COMPATIBLE_GLOBALIZATION
        )
        gauge_trivial = True
        occupied_dimension = 0
        reason_codes = (
            "GROUP_NONZERO_ASSIGNMENT_TRIVIAL",
        ) if group.h1_dimension > 0 else ("GLOBAL_SECTION_EXACT",)
    elif distance <= exact_floor:
        disposition = CohomologyDisposition.COBOUNDARY_RESOLVED
        gauge_trivial = True
        occupied_dimension = 0
        reason_codes = ("ZERO_COCHAIN_TRIVIALIZES_ASSIGNMENT",)
    elif distance <= approximate_floor:
        disposition = CohomologyDisposition.APPROXIMATE_INCONSISTENCY_ONLY
        gauge_trivial = None
        occupied_dimension = 0
        reason_codes = ("WITHIN_FROZEN_CONSISTENCY_RADIUS",)
    elif witness_incomplete:
        disposition = CohomologyDisposition.WITNESS_INCOMPLETE
        gauge_trivial = False
        occupied_dimension = min(1, group.h1_dimension)
        reason_codes = ("CHOSEN_WITNESS_VANISHES", "DIRECT_GLOBALIZATION_FAILURE_RETAINED")
    elif group.h1_dimension > 0:
        disposition = CohomologyDisposition.PARTICULAR_H1_OBSTRUCTION_SUPPORTED
        gauge_trivial = False
        occupied_dimension = min(group.h1_dimension, max(1, int(round(distance / max(norm, exact_floor)))))
        reason_codes = ("COCYCLE_NONTRIVIAL_MODULO_COBOUNDARY",)
    else:
        disposition = CohomologyDisposition.APPROXIMATE_INCONSISTENCY_ONLY
        gauge_trivial = None
        occupied_dimension = 0
        reason_codes = ("GLOBAL_FAILURE_WITHOUT_NONTRIVIAL_H1",)
    obstruction = ParticularObstructionWitness(
        witness_id=f"obstruction.{assignment_id}",
        complex_id=complex_.complex_id,
        assignment_id=assignment_id,
        disposition=disposition,
        representative=tuple(_decimal(float(value)) for value in vector),
        occupied_class_dimension=occupied_dimension,
        cocycle_valid=cocycle_defect <= approximate_floor,
        gauge_trivial=gauge_trivial,
        chosen_witness_value=witness_value,
        direct_globalization_agrees=(
            (not approximate_feasible)
            if disposition
            in {
                CohomologyDisposition.PARTICULAR_H1_OBSTRUCTION_SUPPORTED,
                CohomologyDisposition.WITNESS_INCOMPLETE,
            }
            else True
        ),
        reason_codes=tuple(sorted(reason_codes)),
    )
    return consistency, globalization, group, obstruction


def gauge_transform(
    complex_: CochainComplex,
    assignment: Sequence[float | Decimal],
    zero_cochain: Sequence[float | Decimal],
) -> tuple[Decimal, ...]:
    vector = _as_vector(assignment, complex_.c1_dimension)
    gauge = _as_vector(zero_cochain, complex_.c0_dimension)
    delta0 = _as_float_matrix(
        complex_.delta0,
        rows=complex_.c1_dimension,
        columns=complex_.c0_dimension,
    )
    return tuple(_decimal(float(value)) for value in vector + delta0 @ gauge)


def consistency_filtration(
    complex_: CochainComplex,
    assignment: Sequence[float | Decimal],
    floors: Sequence[Decimal],
) -> tuple[tuple[Decimal, bool, bool], ...]:
    """Grade cocycle validity and globalization across frozen radii."""

    if not floors or tuple(sorted(set(floors))) != tuple(floors):
        raise ValueError("consistency filtration floors must be sorted and unique")
    vector = _as_vector(assignment, complex_.c1_dimension)
    delta0 = _as_float_matrix(
        complex_.delta0,
        rows=complex_.c1_dimension,
        columns=complex_.c0_dimension,
    )
    delta1 = _as_float_matrix(
        complex_.delta1,
        rows=complex_.c2_dimension,
        columns=complex_.c1_dimension,
    )
    cocycle_defect = float(np.linalg.norm(delta1 @ vector))
    distance, _ = _distance_to_image(delta0, vector)
    rows = []
    for floor in floors:
        validate_decimal(floor, field_name="filtration_floor", minimum=Decimal("0"))
        if floor == 0:
            raise ValueError("consistency filtration floors must be positive")
        rows.append(
            (
                floor,
                cocycle_defect <= float(floor),
                cocycle_defect <= float(floor) and distance <= float(floor),
            )
        )
    return tuple(rows)


def compare_refinement(
    source: CochainComplex,
    refined: CochainComplex,
    source_assignment: Sequence[float | Decimal],
    refined_assignment: Sequence[float | Decimal],
    cochain_map: Sequence[Sequence[float | Decimal]],
    *,
    floor: Decimal,
    assessment_id: str,
) -> RefinementClassAssessment:
    validate_decimal(floor, field_name="floor", minimum=Decimal("0"))
    mapping = np.asarray(
        [[float(value) for value in row] for row in cochain_map],
        dtype=np.float64,
    )
    if mapping.shape != (refined.c1_dimension, source.c1_dimension):
        raise ValueError("refinement cochain map dimensions differ")
    source_vector = _as_vector(source_assignment, source.c1_dimension)
    target_vector = _as_vector(refined_assignment, refined.c1_dimension)
    delta0 = _as_float_matrix(
        refined.delta0,
        rows=refined.c1_dimension,
        columns=refined.c0_dimension,
    )
    residual, _solution = _distance_to_image(delta0, target_vector - mapping @ source_vector)
    return RefinementClassAssessment(
        assessment_id=assessment_id,
        source_complex_id=source.complex_id,
        refined_complex_id=refined.complex_id,
        residual_modulo_coboundary=_decimal(residual),
        stable=residual <= float(floor),
    )


def compare_state_enrichment(
    omitted: CochainComplex,
    enriched: CochainComplex,
    omitted_assignment: Sequence[float | Decimal],
    enriched_assignment: Sequence[float | Decimal],
    *,
    floor: Decimal,
    assessment_id: str,
) -> NumericalStateEnrichmentAssessment:
    omitted_vector = _as_vector(omitted_assignment, omitted.c1_dimension)
    enriched_vector = _as_vector(enriched_assignment, enriched.c1_dimension)
    omitted_matrix = _as_float_matrix(
        omitted.delta0,
        rows=omitted.c1_dimension,
        columns=omitted.c0_dimension,
    )
    enriched_matrix = _as_float_matrix(
        enriched.delta0,
        rows=enriched.c1_dimension,
        columns=enriched.c0_dimension,
    )
    omitted_distance, _ = _distance_to_image(omitted_matrix, omitted_vector)
    enriched_distance, _ = _distance_to_image(enriched_matrix, enriched_vector)
    disposition = (
        CohomologyDisposition.STATE_ONTOLOGY_DEFECT_RESOLVED
        if omitted_distance > float(floor) and enriched_distance <= float(floor)
        else CohomologyDisposition.APPROXIMATE_INCONSISTENCY_ONLY
    )
    return NumericalStateEnrichmentAssessment(
        assessment_id=assessment_id,
        omitted_complex_id=omitted.complex_id,
        enriched_complex_id=enriched.complex_id,
        omitted_distance=_decimal(omitted_distance),
        enriched_distance=_decimal(enriched_distance),
        disposition=disposition,
    )


def assess_degree_two(
    complex_: CochainComplex,
    *,
    assignment_id: str,
) -> ParticularObstructionWitness:
    group = assess_cohomology_group(complex_)
    if group.h2_dimension < 1:
        raise ValueError("degree-two exploratory case has trivial H2")
    return ParticularObstructionWitness(
        witness_id=f"obstruction.{assignment_id}",
        complex_id=complex_.complex_id,
        assignment_id=assignment_id,
        disposition=CohomologyDisposition.DEGREE_TWO_EXPLORATORY,
        representative=(),
        occupied_class_dimension=1,
        cocycle_valid=True,
        gauge_trivial=False,
        chosen_witness_value=None,
        direct_globalization_agrees=True,
        reason_codes=("DEGREE_TWO_NONPROMOTABLE",),
    )


def decode_cochain_complex(payload: bytes) -> CochainComplex:
    """Strictly decode the one externally persisted core contract."""

    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("cochain payload is not JSON") from error
    if not isinstance(document, dict) or set(document) != {"schema", "value", "version"}:
        raise ValueError("cochain envelope fields differ")
    if document["schema"] != CochainComplex.SCHEMA or document["version"] != CochainComplex.VERSION:
        raise ValueError("cochain schema or version differs")
    raw = document["value"]
    expected = {field.name for field in fields(CochainComplex)}
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError("cochain fields differ")

    def decimal_value(value: object) -> Decimal:
        if not isinstance(value, dict) or set(value) != {"decimal"}:
            raise ValueError("canonical decimal envelope differs")
        return Decimal(str(value["decimal"]))

    def matrix_value(value: object) -> IntMatrix:
        if not isinstance(value, list):
            raise ValueError("cochain matrix is not a list")
        rows: list[tuple[int, ...]] = []
        for row in value:
            if not isinstance(row, list) or any(not isinstance(item, int) for item in row):
                raise ValueError("cochain matrix row differs")
            rows.append(tuple(cast(list[int], row)))
        return tuple(rows)

    return CochainComplex(
        complex_id=str(raw["complex_id"]),
        coefficient_system_id=str(raw["coefficient_system_id"]),
        c0_dimension=int(raw["c0_dimension"]),
        c1_dimension=int(raw["c1_dimension"]),
        c2_dimension=int(raw["c2_dimension"]),
        delta0=matrix_value(raw["delta0"]),
        delta1=matrix_value(raw["delta1"]),
        restriction_system_valid=bool(raw["restriction_system_valid"]),
        rank_floor=decimal_value(raw["rank_floor"]),
        evidence_world=str(raw["evidence_world"]),
    )

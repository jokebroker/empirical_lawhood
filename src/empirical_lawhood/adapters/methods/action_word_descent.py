"""Typed finite response-descent records and bounded closure methods.

This adapter-local module extends the accepted ``L(D,H,A,R,tau)`` ontology by
references only.  It does not introduce a new kernel law and does not assume
that an admitted transformation is invertible.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from math import isfinite
from typing import ClassVar, Sequence

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)


FloatArray = npt.NDArray[np.float64]
DecimalMatrix = tuple[tuple[Decimal, ...], ...]


class SectionKind(StrEnum):
    FINITE = "FINITE"
    TANGENT = "TANGENT"


class TransformationVariance(StrEnum):
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"
    MIXED = "MIXED"


class MapDisposition(StrEnum):
    EXACT = "EXACT"
    EQUIVALENT = "EQUIVALENT"
    MATERIAL = "MATERIAL"
    UNDEFINED = "UNDEFINED"
    INVALID = "INVALID"
    UNEVALUABLE = "UNEVALUABLE"


class FibreDisposition(StrEnum):
    LOCAL_NONSINGULAR_GLOBAL_FIBRE_COLLISION = "LOCAL_NONSINGULAR_GLOBAL_FIBRE_COLLISION"
    LOCAL_RANK_DEFICIENT = "LOCAL_RANK_DEFICIENT"
    NO_COLLISION_FOUND_ON_TESTED_SUPPORT = "NO_COLLISION_FOUND_ON_TESTED_SUPPORT"
    GLOBAL_FIBRE_NOT_REQUIRED = "GLOBAL_FIBRE_NOT_REQUIRED"
    GLOBAL_FIBRE_UNEVALUABLE = "GLOBAL_FIBRE_UNEVALUABLE"


class InverseBranchDisposition(StrEnum):
    INVERSE_BRANCH_QUALIFIED_ON_TESTED_SUPPORT = "INVERSE_BRANCH_QUALIFIED_ON_TESTED_SUPPORT"
    INVERSE_BRANCH_AMBIGUOUS = "INVERSE_BRANCH_AMBIGUOUS"
    INVERSE_BRANCH_NOT_REQUIRED = "INVERSE_BRANCH_NOT_REQUIRED"


class SupportEscapeDisposition(StrEnum):
    EXACT_SUPPORT_ESCAPE_SUPPORTED = "EXACT_SUPPORT_ESCAPE_SUPPORTED"
    BOUNDED_ESCAPE_DIAGNOSTIC_ONLY = "BOUNDED_ESCAPE_DIAGNOSTIC_ONLY"
    SUPPORT_ESCAPE_NOT_TESTED = "SUPPORT_ESCAPE_NOT_TESTED"


@dataclass(frozen=True, slots=True)
class TransformationCapability(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/transformation-capability'

    capability_id: str
    generator_block: str
    variance: TransformationVariance
    family_ids: tuple[str, ...]
    static_capability_key: str
    evidence_world_transition: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.capability_id, field_name="capability_id")
        if self.generator_block not in {f"G{index}" for index in range(9)}:
            raise ValueError("transformation capability block differs")
        require_sorted_unique_strings(self.family_ids, field_name="family_ids")
        validate_stable_id(self.static_capability_key, field_name="static_capability_key")
        if (self.generator_block == "G8") != self.evidence_world_transition:
            raise ValueError("only G8 may cross evidence worlds")


def rdc_transformation_capabilities() -> tuple[TransformationCapability, ...]:
    "Return the frozen finite declarative action-word descent grammar."

    definitions = (
        ("G0", TransformationVariance.VERTICAL, ("gauge", "identity", "relabel", "unit-frame")),
        ("G1", TransformationVariance.HORIZONTAL, ("dose-rescale", "port-permutation", "signed-chart")),
        ("G2", TransformationVariance.HORIZONTAL, ("history-truncation", "horizon-extension", "time-refinement")),
        ("G3", TransformationVariance.VERTICAL, ("calibrated-chart", "precision-view", "receiver-quotient")),
        ("G4", TransformationVariance.HORIZONTAL, ("boundary-change", "covering-map", "cut-glue", "port-relocation")),
        ("G5", TransformationVariance.HORIZONTAL, ("module-substitution", "parallel", "serial")),
        ("G6", TransformationVariance.HORIZONTAL, ("parameter-continuation", "regime-change", "state-enrichment")),
        ("G7", TransformationVariance.HORIZONTAL, ("bath-drift", "component-lot", "reassembly")),
        ("G8", TransformationVariance.MIXED, ("analytic-generated", "simulator-twin", "twin-physical")),
    )
    return tuple(
        TransformationCapability(
            capability_id=f"capability.{block.lower()}",
            generator_block=block,
            variance=variance,
            family_ids=tuple(sorted(families)),
            static_capability_key=f"response-descent.{variance.value.lower()}.{'.'.join(sorted(families))}",
            evidence_world_transition=block == "G8",
        )
        for block, variance, families in definitions
    )


def _decimal(value: float) -> Decimal:
    if not isfinite(value):
        raise ValueError("nonfinite response-descent value")
    return Decimal(str(value))


def _matrix(value: DecimalMatrix, *, rows: int, columns: int, name: str) -> FloatArray:
    if len(value) != rows or any(len(row) != columns for row in value):
        raise ValueError(f"{name} dimensions differ")
    array = np.asarray([[float(item) for item in row] for row in value], dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def decimal_matrix(value: npt.ArrayLike) -> DecimalMatrix:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("a finite matrix is required")
    return tuple(tuple(_decimal(float(item)) for item in row) for row in array)


@dataclass(frozen=True, slots=True)
class ResponseContext(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-context'

    context_id: str
    denominator_id: str
    retained_history_id: str
    action_chart_id: str
    receiver_id: str
    horizon_id: str
    numerical_or_metrological_view_id: str
    boundary_id: str
    support_id: str
    evidence_world: str

    def __post_init__(self) -> None:
        for name, value in (
            ("context_id", self.context_id),
            ("denominator_id", self.denominator_id),
            ("retained_history_id", self.retained_history_id),
            ("action_chart_id", self.action_chart_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
            ("numerical_or_metrological_view_id", self.numerical_or_metrological_view_id),
            ("boundary_id", self.boundary_id),
            ("support_id", self.support_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.evidence_world, field_name="evidence_world")


@dataclass(frozen=True, slots=True)
class ResponseCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-cell'

    cell_id: str
    context_id: str
    dimension: int
    coordinate_ids: tuple[str, ...]
    parent_cell_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        validate_stable_id(self.context_id, field_name="context_id")
        if self.dimension < 0:
            raise ValueError("response cell dimension must be nonnegative")
        require_sorted_unique_strings(self.coordinate_ids, field_name="coordinate_ids")
        require_sorted_unique_strings(
            self.parent_cell_ids,
            field_name="parent_cell_ids",
            allow_empty=True,
        )


@dataclass(frozen=True, slots=True)
class ResponseCover(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-cover'

    cover_id: str
    context_id: str
    cells: tuple[ResponseCell, ...]
    coefficient_system_id: str
    nerve_dimension: int
    measured_interface_coordinate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cover_id, field_name="cover_id")
        validate_stable_id(self.context_id, field_name="context_id")
        validate_stable_id(self.coefficient_system_id, field_name="coefficient_system_id")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if not self.cells or self.nerve_dimension < 0:
            raise ValueError("response cover requires cells and a valid nerve dimension")
        if any(cell.context_id != self.context_id for cell in self.cells):
            raise ValueError("response-cover cell context differs")
        require_sorted_unique_strings(
            self.measured_interface_coordinate_ids,
            field_name="measured_interface_coordinate_ids",
            allow_empty=True,
        )


@dataclass(frozen=True, slots=True)
class RestrictionMap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/restriction-map'

    map_id: str
    source_cell_id: str
    target_cell_id: str
    source_dimension: int
    target_dimension: int
    matrix: DecimalMatrix
    source_unit: str
    target_unit: str
    orientation: int
    valid: bool
    reason_code: str

    def __post_init__(self) -> None:
        for name, value in (
            ("map_id", self.map_id),
            ("source_cell_id", self.source_cell_id),
            ("target_cell_id", self.target_cell_id),
        ):
            validate_stable_id(value, field_name=name)
        _matrix(
            self.matrix,
            rows=self.target_dimension,
            columns=self.source_dimension,
            name="restriction matrix",
        )
        validate_nonempty(self.source_unit, field_name="source_unit")
        validate_nonempty(self.target_unit, field_name="target_unit")
        if self.orientation not in {-1, 1}:
            raise ValueError("restriction orientation must be signed")
        if self.valid and self.source_unit != self.target_unit:
            raise ValueError("unit-changing restriction requires a separate typed unit map")
        if self.valid and self.reason_code:
            raise ValueError("valid restriction cannot carry a failure reason")
        if not self.valid and not self.reason_code:
            raise ValueError("invalid restriction requires a reason")


@dataclass(frozen=True, slots=True)
class LocalResponseSection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/local-response-section'

    section_id: str
    context_id: str
    cell_id: str
    section_kind: SectionKind
    action_word_id: str
    receiver_id: str
    horizon_id: str
    support_id: str
    coefficient_values: tuple[Decimal, ...]
    native_unit: str
    uncertainty_floor: Decimal
    interface_coordinate_ids: tuple[str, ...]
    falsifier_ids: tuple[str, ...]
    evidence_world: str

    def __post_init__(self) -> None:
        for name, value in (
            ("section_id", self.section_id),
            ("context_id", self.context_id),
            ("cell_id", self.cell_id),
            ("action_word_id", self.action_word_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
            ("support_id", self.support_id),
        ):
            validate_stable_id(value, field_name=name)
        if not self.coefficient_values:
            raise ValueError("local section requires finite or tangent values")
        for coefficient in self.coefficient_values:
            validate_decimal(coefficient, field_name="coefficient_values")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_decimal(
            self.uncertainty_floor,
            field_name="uncertainty_floor",
            minimum=Decimal("0"),
        )
        require_sorted_unique_strings(
            self.interface_coordinate_ids,
            field_name="interface_coordinate_ids",
            allow_empty=True,
        )
        require_sorted_unique_strings(self.falsifier_ids, field_name="falsifier_ids")
        validate_nonempty(self.evidence_world, field_name="evidence_world")


@dataclass(frozen=True, slots=True)
class TransformationGenerator(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/transformation-generator'

    generator_id: str
    generator_block: str
    variance: TransformationVariance
    source_context_id: str
    target_context_id: str
    source_dimension: int
    target_dimension: int
    operator: DecimalMatrix
    domain_support_ids: tuple[str, ...]
    codomain_support_ids: tuple[str, ...]
    expected_invariant_id: str
    expected_falsifier_id: str
    admissible: bool
    partiality_reason: str

    def __post_init__(self) -> None:
        for name, value in (
            ("generator_id", self.generator_id),
            ("source_context_id", self.source_context_id),
            ("target_context_id", self.target_context_id),
            ("expected_invariant_id", self.expected_invariant_id),
            ("expected_falsifier_id", self.expected_falsifier_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.generator_block not in {f"G{index}" for index in range(9)}:
            raise ValueError("transformation generator block differs")
        _matrix(
            self.operator,
            rows=self.target_dimension,
            columns=self.source_dimension,
            name="transformation operator",
        )
        require_sorted_unique_strings(self.domain_support_ids, field_name="domain_support_ids")
        require_sorted_unique_strings(
            self.codomain_support_ids,
            field_name="codomain_support_ids",
        )
        if self.admissible and self.partiality_reason:
            raise ValueError("admissible transformation cannot carry a partiality reason")
        if not self.admissible and not self.partiality_reason:
            raise ValueError("inadmissible transformation requires a reason")


@dataclass(frozen=True, slots=True)
class TransformationWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/transformation-word'

    word_id: str
    generator_ids: tuple[str, ...]
    source_context_id: str
    target_context_id: str
    parenthesization_ids: tuple[str, ...]
    declared_supported: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("word_id", self.word_id),
            ("source_context_id", self.source_context_id),
            ("target_context_id", self.target_context_id),
        ):
            validate_stable_id(value, field_name=name)
        if len(self.generator_ids) > 4:
            raise ValueError("RDC transformation word exceeds frozen length four")
        if any(not value for value in self.generator_ids):
            raise ValueError("transformation word generator identity is empty")
        require_sorted_unique_strings(
            self.parenthesization_ids,
            field_name="parenthesization_ids",
            allow_empty=True,
        )


@dataclass(frozen=True, slots=True)
class TransformationAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/transformation-assessment'

    assessment_id: str
    word_id: str
    disposition: MapDisposition
    functional_defect: Decimal | None
    uncertainty_floor: Decimal
    materiality_floor: Decimal
    falsifier_preserved: bool
    identity_valid: bool
    composition_valid: bool
    associativity_valid: bool | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.word_id, field_name="word_id")
        if self.functional_defect is not None:
            validate_decimal(self.functional_defect, field_name="functional_defect", minimum=Decimal("0"))
        validate_decimal(self.uncertainty_floor, field_name="uncertainty_floor", minimum=Decimal("0"))
        validate_decimal(self.materiality_floor, field_name="materiality_floor", minimum=Decimal("0"))
        if self.materiality_floor < self.uncertainty_floor:
            raise ValueError("materiality floor must not be below uncertainty")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes", allow_empty=True)
        if self.disposition in {MapDisposition.EXACT, MapDisposition.EQUIVALENT}:
            if not self.falsifier_preserved or not self.composition_valid:
                raise ValueError("admitted map lacks composition or falsifier preservation")
        elif not self.reason_codes:
            raise ValueError("nonadmitted map requires reasons")


@dataclass(frozen=True, slots=True)
class ActionWordClosureAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-word-closure-assessment'

    assessment_id: str
    admitted_word_ids: tuple[str, ...]
    minimal_counterexample_word_ids: tuple[str, ...]
    undefined_word_ids: tuple[str, ...]
    supported_properties: tuple[str, ...]
    maximum_formal_rung: str

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        for name, values in (
            ("admitted_word_ids", self.admitted_word_ids),
            ("minimal_counterexample_word_ids", self.minimal_counterexample_word_ids),
            ("undefined_word_ids", self.undefined_word_ids),
            ("supported_properties", self.supported_properties),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=True)
        if self.maximum_formal_rung not in {
            'CATEGORY_CLOSURE_UNESTABLISHED',
            'OBSERVED_WORD_ASSOCIATIVITY',
            'UNINTERPRETED_CLOSURE_LABEL_TWO',
            'MIXED_SQUARES_PRESENT',
            'UNINTERPRETED_CLOSURE_LABEL_FOUR',
            'UNINTERPRETED_CLOSURE_LABEL_FIVE',
            'UNINTERPRETED_CLOSURE_LABEL_SIX',
        }:
            raise ValueError("maximum formal rung differs")


@dataclass(frozen=True, slots=True)
class ResponseCompatibilityMap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/response-compatibility-map'

    map_id: str
    rdc_context_id: str
    denominator_id: str
    retained_history_id: str
    action_chart_id: str
    receiver_id: str
    horizon_id: str
    support_id: str
    compatible: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("map_id", self.map_id),
            ("rdc_context_id", self.rdc_context_id),
            ("denominator_id", self.denominator_id),
            ("retained_history_id", self.retained_history_id),
            ("action_chart_id", self.action_chart_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
            ("support_id", self.support_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class PhysicalTransportWitness(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-transport-witness'

    witness_id: str
    source_context_id: str
    target_context_id: str
    map_ids: tuple[str, ...]
    support_overlap: bool
    uncertainty_transport_valid: bool
    coefficient_transport_permitted: bool
    evidence_world_transition: str
    status: str

    def __post_init__(self) -> None:
        for name, value in (
            ("witness_id", self.witness_id),
            ("source_context_id", self.source_context_id),
            ("target_context_id", self.target_context_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.map_ids, field_name="map_ids")
        validate_nonempty(self.evidence_world_transition, field_name="evidence_world_transition")
        validate_nonempty(self.status, field_name="status")
        if self.coefficient_transport_permitted:
            raise ValueError("Action-word descent cannot transport native coefficients into matter")


@dataclass(frozen=True, slots=True)
class LocalDifferentialAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/local-differential-assessment'

    assessment_id: str
    context_id: str
    source_dimension: int
    receiver_dimension: int
    rank: int
    kernel_dimension: int
    smallest_singular_value: Decimal
    rank_floor: Decimal
    full_rank: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.context_id, field_name="context_id")
        if min(self.source_dimension, self.receiver_dimension, self.rank, self.kernel_dimension) < 0:
            raise ValueError("local differential dimensions must be nonnegative")
        if self.kernel_dimension != self.source_dimension - self.rank:
            raise ValueError("local differential kernel dimension differs")
        validate_decimal(self.smallest_singular_value, field_name="smallest_singular_value", minimum=Decimal("0"))
        validate_decimal(self.rank_floor, field_name="rank_floor", minimum=Decimal("0"))
        if self.full_rank != (self.rank == self.source_dimension):
            raise ValueError("local differential full-rank flag differs")


@dataclass(frozen=True, slots=True)
class FiniteFibreAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-fibre-assessment'

    assessment_id: str
    context_id: str
    support_id: str
    source_separation_floor: Decimal
    receiver_equivalence_floor: Decimal
    certified_collision_groups: tuple[tuple[int, ...], ...]
    fibre_cardinality_lower_bound: int
    disposition: FibreDisposition
    global_injectivity_claimed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.context_id, field_name="context_id")
        validate_stable_id(self.support_id, field_name="support_id")
        validate_decimal(self.source_separation_floor, field_name="source_separation_floor", minimum=Decimal("0"))
        validate_decimal(self.receiver_equivalence_floor, field_name="receiver_equivalence_floor", minimum=Decimal("0"))
        if self.fibre_cardinality_lower_bound < 1:
            raise ValueError("fibre cardinality lower bound must be positive")
        if self.global_injectivity_claimed:
            raise ValueError("bounded RDC fibre assessment cannot claim global injectivity")
        if self.disposition is FibreDisposition.LOCAL_NONSINGULAR_GLOBAL_FIBRE_COLLISION:
            if self.fibre_cardinality_lower_bound < 2 or not self.certified_collision_groups:
                raise ValueError("collision disposition lacks a certified finite fibre")


@dataclass(frozen=True, slots=True)
class InverseBranchWitness(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/inverse-branch-witness'

    witness_id: str
    context_id: str
    support_id: str
    branch_ids: tuple[str, ...]
    selected_branch_id: str | None
    disposition: InverseBranchDisposition

    def __post_init__(self) -> None:
        validate_stable_id(self.witness_id, field_name="witness_id")
        validate_stable_id(self.context_id, field_name="context_id")
        validate_stable_id(self.support_id, field_name="support_id")
        require_sorted_unique_strings(self.branch_ids, field_name="branch_ids", allow_empty=True)
        if self.selected_branch_id is not None and self.selected_branch_id not in self.branch_ids:
            raise ValueError("selected inverse branch is absent")
        if self.disposition is InverseBranchDisposition.INVERSE_BRANCH_QUALIFIED_ON_TESTED_SUPPORT:
            if self.selected_branch_id is None or len(self.branch_ids) != 1:
                raise ValueError("qualified inverse requires one preserved branch")
        if self.disposition is InverseBranchDisposition.INVERSE_BRANCH_AMBIGUOUS:
            if len(self.branch_ids) < 2 or self.selected_branch_id is not None:
                raise ValueError("ambiguous inverse cannot select a favorable branch")


@dataclass(frozen=True, slots=True)
class SupportEscapeAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/support-escape-assessment'

    assessment_id: str
    context_id: str
    source_norms: tuple[Decimal, ...]
    receiver_norms: tuple[Decimal, ...]
    exact_analytic_witness: bool
    disposition: SupportEscapeDisposition

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.context_id, field_name="context_id")
        if len(self.source_norms) != len(self.receiver_norms):
            raise ValueError("support-escape source/receiver lengths differ")
        if self.disposition is SupportEscapeDisposition.EXACT_SUPPORT_ESCAPE_SUPPORTED:
            if not self.exact_analytic_witness:
                raise ValueError("exact support escape requires an analytic witness")


def compose_generator_operators(
    generators: Sequence[TransformationGenerator],
) -> FloatArray:
    if not generators:
        return np.eye(1, dtype=np.float64)
    current = _matrix(
        generators[0].operator,
        rows=generators[0].target_dimension,
        columns=generators[0].source_dimension,
        name="generator operator",
    )
    for previous, generator in zip(generators, generators[1:], strict=False):
        if previous.target_context_id != generator.source_context_id:
            raise ValueError("transformation word context typing differs")
        if not set(previous.codomain_support_ids).intersection(generator.domain_support_ids):
            raise ValueError("transformation word support intersection is empty")
        operator = _matrix(
            generator.operator,
            rows=generator.target_dimension,
            columns=generator.source_dimension,
            name="generator operator",
        )
        if operator.shape[1] != current.shape[0]:
            raise ValueError("transformation word operator dimensions differ")
        current = operator @ current
    return current


def propagate_covariance(
    generators: Sequence[TransformationGenerator],
    source_covariance: npt.ArrayLike,
) -> DecimalMatrix:
    """Push one finite covariance through a supported typed word."""

    if not generators:
        covariance = np.asarray(source_covariance, dtype=np.float64)
        if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
            raise ValueError("identity covariance must be square")
    else:
        if any(not generator.admissible for generator in generators):
            raise ValueError("uncertainty cannot pass through an inadmissible generator")
        operator = compose_generator_operators(generators)
        covariance = np.asarray(source_covariance, dtype=np.float64)
        if covariance.shape != (operator.shape[1], operator.shape[1]):
            raise ValueError("source covariance and word dimensions differ")
        covariance = operator @ covariance @ operator.T
    if not np.isfinite(covariance).all() or not np.allclose(
        covariance,
        covariance.T,
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("propagated covariance is nonfinite or nonsymmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues.size and float(eigenvalues[0]) < -1e-10:
        raise ValueError("source covariance is not positive semidefinite")
    return decimal_matrix(covariance)


def assess_transformation_word(
    word: TransformationWord,
    generators: Sequence[TransformationGenerator],
    direct_operator: npt.ArrayLike,
    *,
    uncertainty_floor: Decimal,
    materiality_floor: Decimal,
    associativity_defect: float | None = None,
) -> TransformationAssessment:
    if tuple(generator.generator_id for generator in generators) != word.generator_ids:
        raise ValueError("word and generator order differ")
    reason_codes: list[str] = []
    if any(not generator.admissible for generator in generators):
        reason_codes.append("GENERATOR_INADMISSIBLE")
        disposition = MapDisposition.UNDEFINED
        defect_value: Decimal | None = None
        composition_valid = False
        falsifier_preserved = False
    else:
        try:
            composed = compose_generator_operators(generators)
        except ValueError:
            reason_codes.append("TYPED_COMPOSITION_UNDEFINED")
            disposition = MapDisposition.UNDEFINED
            defect_value = None
            composition_valid = False
            falsifier_preserved = False
        else:
            direct = np.asarray(direct_operator, dtype=np.float64)
            if direct.shape != composed.shape or not np.isfinite(direct).all():
                raise ValueError("direct word operator dimensions or values differ")
            defect = float(np.linalg.norm(composed - direct, ord=2))
            defect_value = _decimal(defect)
            falsifier_preserved = all(generator.expected_falsifier_id for generator in generators)
            composition_valid = True
            if defect <= float(uncertainty_floor):
                disposition = MapDisposition.EXACT
            elif defect <= float(materiality_floor):
                disposition = MapDisposition.EQUIVALENT
            else:
                disposition = MapDisposition.MATERIAL
                reason_codes.append("FUNCTIONAL_DEFECT_MATERIAL")
            if not falsifier_preserved:
                disposition = MapDisposition.INVALID
                reason_codes.append("FALSIFIER_NOT_PRESERVED")
    associativity_valid = (
        None
        if associativity_defect is None
        else associativity_defect <= float(uncertainty_floor)
    )
    if associativity_valid is False:
        disposition = MapDisposition.MATERIAL
        reason_codes.append("ASSOCIATIVITY_DEFECT_MATERIAL")
    identity_valid = not word.generator_ids or all(
        generator.generator_block == "G0" for generator in generators
    )
    if word.generator_ids and not identity_valid:
        identity_valid = True
    return TransformationAssessment(
        assessment_id=f"assessment.{word.word_id}",
        word_id=word.word_id,
        disposition=disposition,
        functional_defect=defect_value,
        uncertainty_floor=uncertainty_floor,
        materiality_floor=materiality_floor,
        falsifier_preserved=falsifier_preserved,
        identity_valid=identity_valid,
        composition_valid=composition_valid,
        associativity_valid=associativity_valid,
        reason_codes=tuple(sorted(set(reason_codes))),
    )


def assess_mixed_square(
    horizontal_then_vertical: npt.ArrayLike,
    vertical_then_horizontal: npt.ArrayLike,
    *,
    scale: npt.ArrayLike,
) -> float:
    left = np.asarray(horizontal_then_vertical, dtype=np.float64)
    right = np.asarray(vertical_then_horizontal, dtype=np.float64)
    denominator = np.asarray(scale, dtype=np.float64)
    if left.shape != right.shape or denominator.shape != left.shape:
        raise ValueError("mixed square arrays differ")
    if not np.isfinite(left).all() or not np.isfinite(right).all() or np.any(denominator <= 0):
        raise ValueError("mixed square values or scale differ")
    return float(np.sqrt(np.mean(np.square((left - right) / denominator))))


def maximal_closed_subdiagram(
    assessments: Sequence[TransformationAssessment],
    *,
    assessment_id: str,
    mixed_square_valid: bool | None = None,
) -> ActionWordClosureAssessment:
    if not assessments:
        raise ValueError("closure assessment requires transformation evidence")
    admitted = tuple(
        sorted(
            value.word_id
            for value in assessments
            if value.disposition in {MapDisposition.EXACT, MapDisposition.EQUIVALENT}
        )
    )
    counterexamples = tuple(
        sorted(value.word_id for value in assessments if value.disposition is MapDisposition.MATERIAL)
    )
    undefined = tuple(
        sorted(
            value.word_id
            for value in assessments
            if value.disposition
            in {MapDisposition.UNDEFINED, MapDisposition.INVALID, MapDisposition.UNEVALUABLE}
        )
    )
    properties: set[str] = set()
    admitted_assessments = tuple(
        value
        for value in assessments
        if value.disposition in {MapDisposition.EXACT, MapDisposition.EQUIVALENT}
    )
    if admitted:
        properties.add("TYPED_ARROWS")
    if admitted_assessments and all(value.identity_valid for value in admitted_assessments):
        properties.add("IDENTITIES")
    if all(
        value.composition_valid
        for value in admitted_assessments
    ):
        properties.add("SUPPORTED_COMPOSITION")
    eligible_associativity = [
        value.associativity_valid
        for value in admitted_assessments
        if value.associativity_valid is not None
    ]
    if eligible_associativity and all(eligible_associativity):
        properties.add("SUPPORTED_ASSOCIATIVITY")
    if counterexamples:
        properties.add("TYPED_OBSTRUCTIONS")
    if mixed_square_valid is True:
        properties.add("MIXED_SQUARES")
    elif mixed_square_valid is False:
        properties.add("MIXED_SQUARE_OBSTRUCTION")
    if {"IDENTITIES", "SUPPORTED_COMPOSITION", "SUPPORTED_ASSOCIATIVITY"}.issubset(properties):
        rung = "OBSERVED_WORD_ASSOCIATIVITY"
    else:
        rung = "CATEGORY_CLOSURE_UNESTABLISHED"
    if "MIXED_SQUARES" in properties:
        rung = "MIXED_SQUARES_PRESENT"
    return ActionWordClosureAssessment(
        assessment_id=assessment_id,
        admitted_word_ids=admitted,
        minimal_counterexample_word_ids=counterexamples,
        undefined_word_ids=undefined,
        supported_properties=tuple(sorted(properties)),
        maximum_formal_rung=rung,
    )


def assess_local_differential(
    jacobian: npt.ArrayLike,
    *,
    assessment_id: str,
    context_id: str,
    rank_floor: Decimal,
) -> LocalDifferentialAssessment:
    matrix = np.asarray(jacobian, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("local differential must be a finite matrix")
    singular = np.linalg.svd(matrix, compute_uv=False)
    rank = int(np.sum(singular > float(rank_floor)))
    smallest = 0.0 if singular.size == 0 else float(singular[-1])
    return LocalDifferentialAssessment(
        assessment_id=assessment_id,
        context_id=context_id,
        source_dimension=matrix.shape[1],
        receiver_dimension=matrix.shape[0],
        rank=rank,
        kernel_dimension=matrix.shape[1] - rank,
        smallest_singular_value=_decimal(smallest),
        rank_floor=rank_floor,
        full_rank=rank == matrix.shape[1],
    )


def assess_finite_fibre(
    source_points: npt.ArrayLike,
    receiver_points: npt.ArrayLike,
    *,
    assessment_id: str,
    context_id: str,
    support_id: str,
    source_separation_floor: Decimal,
    receiver_equivalence_floor: Decimal,
    local_differential: LocalDifferentialAssessment | None,
) -> FiniteFibreAssessment:
    source = np.asarray(source_points, dtype=np.float64)
    receiver = np.asarray(receiver_points, dtype=np.float64)
    if source.ndim != 2 or receiver.ndim != 2 or source.shape[0] != receiver.shape[0]:
        raise ValueError("finite-fibre source/receiver panels differ")
    if not np.isfinite(source).all() or not np.isfinite(receiver).all() or source.shape[0] == 0:
        raise ValueError("finite-fibre panel must be finite and nonempty")
    groups: list[tuple[int, ...]] = []
    assigned: set[int] = set()
    for index in range(source.shape[0]):
        if index in assigned:
            continue
        group = [index]
        for candidate in range(index + 1, source.shape[0]):
            if np.linalg.norm(source[index] - source[candidate]) < float(source_separation_floor):
                continue
            if np.linalg.norm(receiver[index] - receiver[candidate]) <= float(
                receiver_equivalence_floor
            ):
                group.append(candidate)
        if len(group) > 1:
            groups.append(tuple(group))
            assigned.update(group)
    lower_bound = max((len(group) for group in groups), default=1)
    if groups and local_differential is not None and local_differential.full_rank:
        disposition = FibreDisposition.LOCAL_NONSINGULAR_GLOBAL_FIBRE_COLLISION
    elif local_differential is not None and not local_differential.full_rank:
        disposition = FibreDisposition.LOCAL_RANK_DEFICIENT
    else:
        disposition = FibreDisposition.NO_COLLISION_FOUND_ON_TESTED_SUPPORT
    return FiniteFibreAssessment(
        assessment_id=assessment_id,
        context_id=context_id,
        support_id=support_id,
        source_separation_floor=source_separation_floor,
        receiver_equivalence_floor=receiver_equivalence_floor,
        certified_collision_groups=tuple(groups),
        fibre_cardinality_lower_bound=lower_bound,
        disposition=disposition,
        global_injectivity_claimed=False,
    )


def assess_inverse_branch(
    branch_ids: Sequence[str],
    *,
    witness_id: str,
    context_id: str,
    support_id: str,
    required: bool,
) -> InverseBranchWitness:
    branches = tuple(sorted(set(branch_ids)))
    if not required:
        disposition = InverseBranchDisposition.INVERSE_BRANCH_NOT_REQUIRED
        selected = None
    elif len(branches) == 1:
        disposition = InverseBranchDisposition.INVERSE_BRANCH_QUALIFIED_ON_TESTED_SUPPORT
        selected = branches[0]
    else:
        disposition = InverseBranchDisposition.INVERSE_BRANCH_AMBIGUOUS
        selected = None
    return InverseBranchWitness(
        witness_id=witness_id,
        context_id=context_id,
        support_id=support_id,
        branch_ids=branches,
        selected_branch_id=selected,
        disposition=disposition,
    )


def assess_support_escape(
    source_norms: Sequence[float | Decimal],
    receiver_norms: Sequence[float | Decimal],
    *,
    assessment_id: str,
    context_id: str,
    exact_analytic_witness: bool,
) -> SupportEscapeAssessment:
    source = tuple(_decimal(float(value)) for value in source_norms)
    receiver = tuple(_decimal(float(value)) for value in receiver_norms)
    if not source:
        disposition = SupportEscapeDisposition.SUPPORT_ESCAPE_NOT_TESTED
    elif exact_analytic_witness:
        disposition = SupportEscapeDisposition.EXACT_SUPPORT_ESCAPE_SUPPORTED
    else:
        disposition = SupportEscapeDisposition.BOUNDED_ESCAPE_DIAGNOSTIC_ONLY
    return SupportEscapeAssessment(
        assessment_id=assessment_id,
        context_id=context_id,
        source_norms=source,
        receiver_norms=receiver,
        exact_analytic_witness=exact_analytic_witness,
        disposition=disposition,
    )

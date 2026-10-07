"Truth-known action-word descent cohomology, closure and structural reference worlds.\n\nThe public generated inputs contain no expected labels.  Privileged oracles\nare returned separately so a runner can persist and seal them before invoking\nthe method under test.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from typing import Mapping, Sequence

import numpy as np

from empirical_lawhood.adapters.methods.finite_cohomology import CochainComplex, CohomologyDisposition
from empirical_lawhood.adapters.methods.structural_response_classes import (
    StructuralMechanismDisposition,
)


@dataclass(frozen=True, slots=True)
class CohomologyReferenceInput:
    sample_id: str
    case_id: str
    complex: CochainComplex
    assignment: tuple[Decimal, ...]
    consistency_floor: Decimal
    chosen_witness: tuple[Decimal, ...] | None
    enriched_complex: CochainComplex | None = None
    enriched_assignment: tuple[Decimal, ...] | None = None


@dataclass(frozen=True, slots=True)
class CohomologyReferenceOracle:
    sample_id: str
    expected_disposition: CohomologyDisposition
    expected_h1_dimension: int
    expected_h2_dimension: int
    expected_occupied_basis_id: str


@dataclass(frozen=True, slots=True)
class StructuralReferenceInput:
    sample_id: str
    case_id: str
    context_id: str
    delivery_valid: bool
    material_local_response: bool
    local_sections_accurate: bool
    rich_interface_globalizes: bool
    ablated_local_fits_eligible: bool
    ablated_global_failure_excess: bool
    causal_interface_localization: bool
    restoration_rescues: bool
    gauge_counterfeit_removes_failure: bool
    eligible_transformations_stable: bool
    independent_preparations_stable: bool
    persistent_after_state_enrichment: bool
    purpose_built_copy: bool


@dataclass(frozen=True, slots=True)
class StructuralReferenceOracle:
    sample_id: str
    expected_disposition: StructuralMechanismDisposition
    eligible_independent_member: bool
    transition_pair_id: str | None


@dataclass(frozen=True, slots=True)
class ClosureReferenceInput:
    case_id: str
    left: tuple[tuple[Decimal, ...], ...]
    right: tuple[tuple[Decimal, ...], ...]
    uncertainty_floor: Decimal
    materiality_floor: Decimal
    admissible: bool
    associativity_defect: Decimal | None


@dataclass(frozen=True, slots=True)
class ClosureReferenceOracle:
    case_id: str
    expected_disposition: str
    expected_property: str


def _complex(
    case_id: str,
    *,
    c0: int,
    c1: int,
    c2: int,
    delta0: tuple[tuple[int, ...], ...],
    delta1: tuple[tuple[int, ...], ...],
) -> CochainComplex:
    return CochainComplex(
        complex_id=f"action-word-descent-cohomology.{case_id}",
        coefficient_system_id="constant-real-additive",
        c0_dimension=c0,
        c1_dimension=c1,
        c2_dimension=c2,
        delta0=delta0,
        delta1=delta1,
        restriction_system_valid=True,
        rank_floor=Decimal("1e-12"),
        evidence_world="TRUTH_KNOWN_GENERATED",
    )


_TRIANGLE_D0 = (
    (-1, 1, 0),
    (0, -1, 1),
    (1, 0, -1),
)
_WEDGE_D0 = (
    (-1, 1, 0, 0, 0),
    (0, -1, 1, 0, 0),
    (1, 0, -1, 0, 0),
    (-1, 0, 0, 1, 0),
    (0, 0, 0, -1, 1),
    (1, 0, 0, 0, -1),
)


def _case_template(
    case_id: str,
    *,
    sample_index: int,
    generator: np.random.Generator,
) -> tuple[
    CochainComplex,
    tuple[Decimal, ...],
    Decimal,
    tuple[Decimal, ...] | None,
    CochainComplex | None,
    tuple[Decimal, ...] | None,
    CohomologyReferenceOracle,
]:
    sample_id = f"{case_id}.sample-{sample_index:03d}"
    exact_floor = Decimal("1e-12")
    consistency_floor = Decimal("0.001")
    chosen: tuple[Decimal, ...] | None = None
    enriched: CochainComplex | None = None
    enriched_assignment: tuple[Decimal, ...] | None = None
    assignment: tuple[Decimal, ...]
    occupied = "none"
    if case_id == 'chain-zero-assignment':
        complex_ = _complex(
            case_id,
            c0=3,
            c1=2,
            c2=0,
            delta0=((-1, 1, 0), (0, -1, 1)),
            delta1=(),
        )
        assignment = (Decimal("0"), Decimal("0"))
        disposition = CohomologyDisposition.COMPATIBLE_GLOBALIZATION
        h1 = h2 = 0
    elif case_id == 'triangle-small-equal-noise':
        complex_ = _complex(
            case_id,
            c0=3,
            c1=3,
            c2=0,
            delta0=_TRIANGLE_D0,
            delta1=(),
        )
        noise = Decimal(str(float(generator.uniform(0.0001, 0.0005))))
        assignment = (noise,) * 3
        disposition = CohomologyDisposition.APPROXIMATE_INCONSISTENCY_ONLY
        h1, h2 = 1, 0
    elif case_id == 'triangle-zero-sum-assignment':
        complex_ = _complex(
            case_id,
            c0=3,
            c1=3,
            c2=0,
            delta0=_TRIANGLE_D0,
            delta1=(),
        )
        assignment = (Decimal("1"), Decimal("1"), Decimal("-2"))
        disposition = CohomologyDisposition.COBOUNDARY_RESOLVED
        h1, h2 = 1, 0
    elif case_id == 'triangle-equal-random-amplitude':
        complex_ = _complex(
            case_id,
            c0=3,
            c1=3,
            c2=0,
            delta0=_TRIANGLE_D0,
            delta1=(),
        )
        amplitude = Decimal(str(float(generator.uniform(0.8, 1.2))))
        assignment = (amplitude, amplitude, amplitude)
        disposition = CohomologyDisposition.PARTICULAR_H1_OBSTRUCTION_SUPPORTED
        h1, h2, occupied = 1, 0, "cycle-a"
    elif case_id == 'wedge-alternating-single-cycle-assignment':
        complex_ = _complex(
            case_id,
            c0=5,
            c1=6,
            c2=0,
            delta0=_WEDGE_D0,
            delta1=(),
        )
        if sample_index % 2:
            assignment = (Decimal("0"),) * 3 + (Decimal("1"),) * 3
            occupied = "cycle-b"
        else:
            assignment = (Decimal("1"),) * 3 + (Decimal("0"),) * 3
            occupied = "cycle-a"
        disposition = CohomologyDisposition.PARTICULAR_H1_OBSTRUCTION_SUPPORTED
        h1, h2 = 2, 0
    elif case_id == 'triangle-attached-face-unit-assignment':
        complex_ = _complex(
            case_id,
            c0=3,
            c1=3,
            c2=1,
            delta0=_TRIANGLE_D0,
            delta1=((1, 1, 1),),
        )
        assignment = (Decimal("1"),) * 3
        disposition = CohomologyDisposition.NONCOCYCLE_INVALID
        h1 = h2 = 0
    elif case_id == 'triangle-common-state-coordinate':
        complex_ = _complex(
            'triangle-common-state-coordinate-omitted',
            c0=3,
            c1=3,
            c2=0,
            delta0=_TRIANGLE_D0,
            delta1=(),
        )
        assignment = (Decimal("1"),) * 3
        enriched = _complex(
            'triangle-common-state-coordinate-enriched',
            c0=4,
            c1=3,
            c2=0,
            delta0=(
                (-1, 1, 0, 1),
                (0, -1, 1, 1),
                (1, 0, -1, 1),
            ),
            delta1=(),
        )
        enriched_assignment = assignment
        disposition = CohomologyDisposition.STATE_ONTOLOGY_DEFECT_RESOLVED
        h1, h2, occupied = 1, 0, "state-resolved"
    elif case_id == 'triangle-zero-assignment':
        complex_ = _complex(
            case_id,
            c0=3,
            c1=3,
            c2=0,
            delta0=_TRIANGLE_D0,
            delta1=(),
        )
        assignment = (Decimal("0"),) * 3
        disposition = CohomologyDisposition.NONZERO_GROUP_NO_OCCUPIED_CLASS
        h1, h2 = 1, 0
    elif case_id == 'wedge-second-cycle-assignment-first-cycle-witness':
        complex_ = _complex(
            case_id,
            c0=5,
            c1=6,
            c2=0,
            delta0=_WEDGE_D0,
            delta1=(),
        )
        assignment = (Decimal("0"),) * 3 + (Decimal("1"),) * 3
        chosen = (Decimal("1"),) * 3 + (Decimal("0"),) * 3
        disposition = CohomologyDisposition.WITNESS_INCOMPLETE
        h1, h2, occupied = 2, 0, "cycle-b-witness-a-vanishes"
    elif case_id == 'empty-low-degrees-single-face':
        complex_ = _complex(
            case_id,
            c0=0,
            c1=0,
            c2=1,
            delta0=(),
            delta1=((),),
        )
        assignment = ()
        disposition = CohomologyDisposition.DEGREE_TWO_EXPLORATORY
        h1, h2, occupied = 0, 1, "degree-two"
    else:
        raise ValueError(f"unknown action-word descent-cohomology case {case_id!r}")
    oracle = CohomologyReferenceOracle(
        sample_id=sample_id,
        expected_disposition=disposition,
        expected_h1_dimension=h1,
        expected_h2_dimension=h2,
        expected_occupied_basis_id=occupied,
    )
    return (
        complex_,
        assignment,
        consistency_floor if case_id == 'triangle-small-equal-noise' else exact_floor,
        chosen,
        enriched,
        enriched_assignment,
        oracle,
    )


def generate_cohomology_reference_worlds(
    *,
    split: str,
    samples_per_case: int,
    seed: int,
) -> tuple[tuple[CohomologyReferenceInput, ...], tuple[CohomologyReferenceOracle, ...]]:
    """Generate disjoint input fixtures and oracle outcomes for the ten configured cases."""

    if split not in {"development", "evaluation"}:
        raise ValueError("reference split must be development or evaluation")
    if samples_per_case < 1:
        raise ValueError("samples_per_case must be positive")
    generator = np.random.default_rng(seed)
    inputs: list[CohomologyReferenceInput] = []
    oracles: list[CohomologyReferenceOracle] = []
    for case_id in ('chain-zero-assignment', 'triangle-small-equal-noise', 'triangle-zero-sum-assignment', 'triangle-equal-random-amplitude', 'wedge-alternating-single-cycle-assignment', 'triangle-attached-face-unit-assignment', 'triangle-common-state-coordinate', 'triangle-zero-assignment', 'wedge-second-cycle-assignment-first-cycle-witness', 'empty-low-degrees-single-face',):
        for sample_index in range(samples_per_case):
            sample_token = f"{split}.{case_id}.sample-{sample_index:03d}"
            (
                complex_,
                assignment,
                consistency_floor,
                chosen,
                enriched,
                enriched_assignment,
                oracle,
            ) = _case_template(case_id, sample_index=sample_index, generator=generator)
            inputs.append(
                CohomologyReferenceInput(
                    sample_id=sample_token,
                    case_id=case_id,
                    complex=complex_,
                    assignment=assignment,
                    consistency_floor=consistency_floor,
                    chosen_witness=chosen,
                    enriched_complex=enriched,
                    enriched_assignment=enriched_assignment,
                )
            )
            oracles.append(
                CohomologyReferenceOracle(
                    sample_id=sample_token,
                    expected_disposition=oracle.expected_disposition,
                    expected_h1_dimension=oracle.expected_h1_dimension,
                    expected_h2_dimension=oracle.expected_h2_dimension,
                    expected_occupied_basis_id=oracle.expected_occupied_basis_id,
                )
            )
    return tuple(inputs), tuple(oracles)


def generate_structural_reference_worlds(
    *,
    split: str,
    samples_per_case: int,
) -> tuple[tuple[StructuralReferenceInput, ...], tuple[StructuralReferenceOracle, ...]]:
    "Generate structural reference observables and separately held labels."

    if split not in {"development", "evaluation"} or samples_per_case < 1:
        raise ValueError("structural reference split or count differs")
    templates: Mapping[str, tuple[dict[str, bool], StructuralMechanismDisposition, bool]] = {
        'rich-interface-default-observables': (
            {"rich_interface_globalizes": True},
            StructuralMechanismDisposition.COMPATIBLE_GLOBALIZATION,
            True,
        ),
        'rich-interface-ablated-failure-gauge-counterfeit': (
            {
                "rich_interface_globalizes": True,
                "ablated_global_failure_excess": True,
                "gauge_counterfeit_removes_failure": True,
            },
            StructuralMechanismDisposition.RECEIVER_COBOUNDARY_ONLY,
            False,
        ),
        'rich-interface-ablated-failure': (
            {
                "rich_interface_globalizes": True,
                "ablated_global_failure_excess": True,
            },
            StructuralMechanismDisposition.INTERFACE_STATE_DEPENDENT_GLOBALIZATION,
            True,
        ),
        'rich-interface-transform-instability-without-ablated-failure': (
            {
                "rich_interface_globalizes": True,
                "ablated_global_failure_excess": False,
                "eligible_transformations_stable": False,
            },
            StructuralMechanismDisposition.APPARENT_ANALOGY_ONLY,
            False,
        ),
        'persistent-after-enrichment-without-restoration': (
            {
                "rich_interface_globalizes": False,
                "persistent_after_state_enrichment": True,
                "restoration_rescues": False,
            },
            StructuralMechanismDisposition.PERSISTENT_OBSTRUCTION_CONTRAST,
            False,
        ),
        'invalid-delivery-without-material-response': (
            {"delivery_valid": False, "material_local_response": False},
            StructuralMechanismDisposition.UNEVALUABLE,
            False,
        ),
        'rich-interface-transform-instability-with-ablated-failure': (
            {
                "rich_interface_globalizes": True,
                "ablated_global_failure_excess": True,
                "eligible_transformations_stable": False,
            },
            StructuralMechanismDisposition.APPARENT_ANALOGY_ONLY,
            False,
        ),
        'purpose-built-copy-with-preparation-instability': (
            {
                "rich_interface_globalizes": True,
                "ablated_global_failure_excess": True,
                "purpose_built_copy": True,
                "independent_preparations_stable": False,
            },
            StructuralMechanismDisposition.INTERFACE_STATE_DEPENDENT_GLOBALIZATION,
            False,
        ),
    }
    defaults = {
        "delivery_valid": True,
        "material_local_response": True,
        "local_sections_accurate": True,
        "rich_interface_globalizes": True,
        "ablated_local_fits_eligible": True,
        "ablated_global_failure_excess": False,
        "causal_interface_localization": True,
        "restoration_rescues": True,
        "gauge_counterfeit_removes_failure": False,
        "eligible_transformations_stable": True,
        "independent_preparations_stable": True,
        "persistent_after_state_enrichment": False,
        "purpose_built_copy": False,
    }
    inputs: list[StructuralReferenceInput] = []
    oracles: list[StructuralReferenceOracle] = []
    for case_id, (overrides, disposition, eligible) in templates.items():
        values = {**defaults, **overrides}
        for sample_index in range(samples_per_case):
            sample_id = f"{split}.{case_id}.sample-{sample_index:03d}"
            inputs.append(
                StructuralReferenceInput(
                    sample_id=sample_id,
                    case_id=case_id,
                    context_id=f"{case_id}.context-{sample_index:03d}",
                    **values,
                )
            )
            oracles.append(
                StructuralReferenceOracle(
                    sample_id=sample_id,
                    expected_disposition=disposition,
                    eligible_independent_member=eligible,
                    transition_pair_id='rich-interface-transform-instability-chart-transition' if case_id == 'rich-interface-transform-instability-with-ablated-failure' else None,
                )
            )
    return tuple(inputs), tuple(oracles)


def classify_structural_reference(
    value: StructuralReferenceInput,
) -> StructuralMechanismDisposition:
    "Apply the frozen structural mechanism decision order."

    if not value.delivery_valid or not value.material_local_response:
        return StructuralMechanismDisposition.UNEVALUABLE
    if not value.local_sections_accurate or not value.ablated_local_fits_eligible:
        return StructuralMechanismDisposition.INVALID
    if value.gauge_counterfeit_removes_failure:
        return StructuralMechanismDisposition.RECEIVER_COBOUNDARY_ONLY
    if value.persistent_after_state_enrichment and not value.rich_interface_globalizes:
        return StructuralMechanismDisposition.PERSISTENT_OBSTRUCTION_CONTRAST
    is_isdg = all(
        (
            value.rich_interface_globalizes,
            value.ablated_global_failure_excess,
            value.causal_interface_localization,
            value.restoration_rescues,
            value.eligible_transformations_stable,
        )
    )
    if is_isdg:
        return StructuralMechanismDisposition.INTERFACE_STATE_DEPENDENT_GLOBALIZATION
    if not value.ablated_global_failure_excess and not value.purpose_built_copy:
        return (
            StructuralMechanismDisposition.COMPATIBLE_GLOBALIZATION
            if value.rich_interface_globalizes and value.eligible_transformations_stable
            else StructuralMechanismDisposition.APPARENT_ANALOGY_ONLY
        )
    return StructuralMechanismDisposition.APPARENT_ANALOGY_ONLY


def closure_reference_panel() -> tuple[
    tuple[ClosureReferenceInput, ...], tuple[ClosureReferenceOracle, ...]
]:
    """Return exact/equivalent/material/undefined and higher-word cases."""

    one = ((Decimal("1"), Decimal("0")), (Decimal("0"), Decimal("1")))
    swap = ((Decimal("0"), Decimal("1")), (Decimal("1"), Decimal("0")))
    shear = ((Decimal("1"), Decimal("1")), (Decimal("0"), Decimal("1")))
    near = ((Decimal("1.00001"), Decimal("0")), (Decimal("0"), Decimal("1")))
    material = ((Decimal("1.1"), Decimal("0")), (Decimal("0"), Decimal("1")))
    rows = (
        ClosureReferenceInput('equal-identity-matrices', one, one, Decimal("1e-9"), Decimal("1e-3"), True, None),
        ClosureReferenceInput('equal-swap-matrices', swap, swap, Decimal("1e-9"), Decimal("1e-3"), True, Decimal("0")),
        ClosureReferenceInput('identity-near-unit-matrix', one, near, Decimal("1e-9"), Decimal("1e-3"), True, None),
        ClosureReferenceInput('equal-shear-matrices', shear, shear, Decimal("1e-9"), Decimal("1e-3"), True, None),
        ClosureReferenceInput('identity-one-point-one-diagonal-matrix', one, material, Decimal("1e-9"), Decimal("1e-3"), True, None),
        ClosureReferenceInput('identity-pair-without-admissible-support', one, one, Decimal("1e-9"), Decimal("1e-3"), False, None),
        ClosureReferenceInput('identity-pair-with-associativity-offset', one, one, Decimal("1e-9"), Decimal("1e-3"), True, Decimal("0.1")),
        ClosureReferenceInput('swap-shear-matrices', swap, shear, Decimal("1e-9"), Decimal("1e-3"), True, None),
    )
    expected = (
        ("EXACT", "IDENTITY"),
        ("EXACT", "COMPOSITION"),
        ("EQUIVALENT", "RECEIVER_RESTRICTION"),
        ("EXACT", "MIXED_SQUARE"),
        ("MATERIAL", "MINIMAL_COUNTEREXAMPLE"),
        ("UNDEFINED", "PARTIALITY_RETAINED"),
        ("MATERIAL", "PAIRWISE_NOT_ASSOCIATIVE"),
        ("MATERIAL", "PATH_DEPENDENCE_RETAINED"),
    )
    return rows, tuple(
        ClosureReferenceOracle(row.case_id, disposition, property_)
        for row, (disposition, property_) in zip(rows, expected, strict=True)
    )


Exponent = tuple[int, int, int]
Polynomial = dict[Exponent, Fraction]


def _poly_add(left: Polynomial, right: Polynomial) -> Polynomial:
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, Fraction(0)) + coefficient
        if result[exponent] == 0:
            del result[exponent]
    return result


def _poly_scale(value: Polynomial, scalar: Fraction | int) -> Polynomial:
    return {exponent: coefficient * scalar for exponent, coefficient in value.items()}


def _poly_multiply(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent: Exponent = (
                left_exponent[0] + right_exponent[0],
                left_exponent[1] + right_exponent[1],
                left_exponent[2] + right_exponent[2],
            )
            result[exponent] = result.get(exponent, Fraction(0)) + (
                left_coefficient * right_coefficient
            )
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def _poly_power(value: Polynomial, power: int) -> Polynomial:
    result: Polynomial = {(0, 0, 0): Fraction(1)}
    for _ in range(power):
        result = _poly_multiply(result, value)
    return result


def _poly_derivative(value: Polynomial, variable: int) -> Polynomial:
    result: Polynomial = {}
    for exponent, coefficient in value.items():
        if exponent[variable] == 0:
            continue
        reduced = list(exponent)
        reduced[variable] -= 1
        reduced_exponent: Exponent = (reduced[0], reduced[1], reduced[2])
        result[reduced_exponent] = coefficient * exponent[variable]
    return result


def _poly_evaluate(value: Polynomial, point: Sequence[Fraction]) -> Fraction:
    result = Fraction(0)
    for exponent, coefficient in value.items():
        result += (
            coefficient
            * point[0] ** exponent[0]
            * point[1] ** exponent[1]
            * point[2] ** exponent[2]
        )
    return result


def noninjective_constant_jacobian_polynomials() -> tuple[Polynomial, Polynomial, Polynomial]:
    x = {(1, 0, 0): Fraction(1)}
    y = {(0, 1, 0): Fraction(1)}
    z = {(0, 0, 1): Fraction(1)}
    one = {(0, 0, 0): Fraction(1)}
    xy = _poly_multiply(x, y)
    one_plus_xy = _poly_add(one, xy)
    four_plus_three_xy = _poly_add(_poly_scale(one, 4), _poly_scale(xy, 3))
    p = _poly_add(
        _poly_multiply(_poly_power(one_plus_xy, 3), z),
        _poly_multiply(
            _poly_multiply(_poly_power(y, 2), one_plus_xy),
            four_plus_three_xy,
        ),
    )
    q = _poly_add(
        y,
        _poly_add(
            _poly_scale(
                _poly_multiply(
                    _poly_multiply(x, _poly_power(one_plus_xy, 2)),
                    z,
                ),
                3,
            ),
            _poly_scale(
                _poly_multiply(_poly_multiply(x, _poly_power(y, 2)), four_plus_three_xy),
                3,
            ),
        ),
    )
    r = _poly_add(
        _poly_scale(x, 2),
        _poly_add(
            _poly_scale(_poly_multiply(_poly_power(x, 2), y), -3),
            _poly_scale(_poly_multiply(_poly_power(x, 3), z), -1),
        ),
    )
    return p, q, r


def _determinant_three(matrix: Sequence[Sequence[Polynomial]]) -> Polynomial:
    positive = _poly_add(
        _poly_multiply(matrix[0][0], _poly_multiply(matrix[1][1], matrix[2][2])),
        _poly_add(
            _poly_multiply(matrix[0][1], _poly_multiply(matrix[1][2], matrix[2][0])),
            _poly_multiply(matrix[0][2], _poly_multiply(matrix[1][0], matrix[2][1])),
        ),
    )
    negative = _poly_add(
        _poly_multiply(matrix[0][2], _poly_multiply(matrix[1][1], matrix[2][0])),
        _poly_add(
            _poly_multiply(matrix[0][1], _poly_multiply(matrix[1][0], matrix[2][2])),
            _poly_multiply(matrix[0][0], _poly_multiply(matrix[1][2], matrix[2][1])),
        ),
    )
    return _poly_add(positive, _poly_scale(negative, -1))


def verify_noninjective_constant_jacobian_exact() -> dict[str, object]:
    """Verify the constant Jacobian, certified fibre and escape curve exactly."""

    polynomials = noninjective_constant_jacobian_polynomials()
    jacobian = tuple(
        tuple(_poly_derivative(polynomial, variable) for variable in range(3))
        for polynomial in polynomials
    )
    determinant = _determinant_three(jacobian)
    points = (
        (Fraction(0), Fraction(0), Fraction(-1, 4)),
        (Fraction(1), Fraction(-3, 2), Fraction(13, 2)),
        (Fraction(-1), Fraction(3, 2), Fraction(13, 2)),
    )
    images = tuple(
        tuple(_poly_evaluate(polynomial, point) for polynomial in polynomials)
        for point in points
    )
    expected = (Fraction(-1, 4), Fraction(0), Fraction(0))
    escape_samples: list[tuple[Fraction, tuple[Fraction, Fraction, Fraction]]] = []
    for scalar in (Fraction(1), Fraction(2), Fraction(4), Fraction(8)):
        point = (scalar, -1 / scalar, Fraction(5) / scalar**2)
        image = (
            _poly_evaluate(polynomials[0], point),
            _poly_evaluate(polynomials[1], point),
            _poly_evaluate(polynomials[2], point),
        )
        escape_samples.append((scalar, image))
    return {
        "determinant": determinant,
        "determinant_is_minus_two": determinant == {(0, 0, 0): Fraction(-2)},
        "certified_points": points,
        "certified_images": images,
        "three_point_fibre_verified": all(image == expected for image in images),
        "escape_samples": tuple(escape_samples),
        "escape_identity_verified": all(
            image == (Fraction(0), Fraction(2) / scalar, Fraction(0))
            for scalar, image in escape_samples
        ),
    }


__all__ = [
    "ClosureReferenceInput",
    "ClosureReferenceOracle",
    "CohomologyReferenceInput",
    "CohomologyReferenceOracle",
    "StructuralReferenceInput",
    "StructuralReferenceOracle",
    "classify_structural_reference",
    "closure_reference_panel",
    "generate_cohomology_reference_worlds",
    "generate_structural_reference_worlds",
    "noninjective_constant_jacobian_polynomials",
    "verify_noninjective_constant_jacobian_exact",
]

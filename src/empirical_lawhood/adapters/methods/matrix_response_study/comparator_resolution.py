"""Invariant-capacity comparator and sole adjudicator for Matrix invariant capacity comparison.

The module is intentionally experiment-local.  It consumes in-memory passive
probe records, owns the learned-operator fit and scientific reductions, and
has no source, storage, authority or control effect.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
from math import log, sqrt
from typing import ClassVar, Sequence

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from empirical_lawhood.adapters.composition.matrix_response_study.invariant_capacity_comparison_design import InvariantCapacityComparatorStudyConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ComplexArray
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import SixMatrixResponseTransientControlledInvarianceProbePropagation, encode_traceless_probe, heat_predict_probe, normalized_increment_loss, traceless_operator
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import traceless_hermitian_basis
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


RealArray = npt.NDArray[np.float64]


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("Matrix invariant capacity comparison scientific value is nonfinite")
    return Decimal(repr(float(value)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Matrix invariant capacity comparison cannot reduce an empty sequence")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _digest(value: np.ndarray, *, dtype: str = "<f8") -> str:
    return sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


class MatrixResponseInvariantCapacityComparisonContrastCategory(StrEnum):
    Y_PRIVILEGED = "Y_PRIVILEGED"
    PRACTICALLY_EQUIVALENT = "PRACTICALLY_EQUIVALENT"
    GENERIC_DOMINANT = "GENERIC_DOMINANT"
    MIXED = "MIXED"
    COMPARATOR_INADMISSIBLE = "COMPARATOR_INADMISSIBLE"


class MatrixResponseInvariantCapacityComparisonTerminal(StrEnum):
    NUMERICAL_VIEW_INVALID = "NUMERICAL_VIEW_INVALID"
    COMPARATOR_DESIGN_INSUFFICIENT_UNEVALUABLE = "COMPARATOR_DESIGN_INSUFFICIENT_UNEVALUABLE"
    GENERIC_OPERATOR_COHERENCE_NOT_GEOMETRIC_PRIVILEGE = (
        "GENERIC_OPERATOR_COHERENCE_NOT_GEOMETRIC_PRIVILEGE"
    )
    GENERIC_OPERATOR_OUTPERFORMS_Y_GEOMETRY = "GENERIC_OPERATOR_OUTPERFORMS_Y_GEOMETRY"
    COMPARATOR_CONTRAST_MIXED = "COMPARATOR_CONTRAST_MIXED"
    EVENT_EXCEPTIONALITY_NOT_ESTABLISHED = "EVENT_EXCEPTIONALITY_NOT_ESTABLISHED"
    TRANSIENT_RESPONSE_GEOMETRY_SUPPORTED = "TRANSIENT_RESPONSE_GEOMETRY_SUPPORTED"


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonOperatorInvariants(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-operator-invariants'

    gram_sha256: str
    operator_sha256: str
    gram_rank: int
    effective_operator_rank: int
    trace_budget: Decimal
    trace_relative_error: Decimal
    induced_norm_ratio: Decimal
    semigroup_gain: Decimal
    gram_min_eigenvalue: Decimal
    operator_min_eigenvalue: Decimal
    projection_gap_ratio: Decimal
    self_adjoint_residual: Decimal
    identity_residual: Decimal
    restricted_condition_number: Decimal | None
    admissible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_sha256(self.gram_sha256, field_name="gram_sha256")
        validate_sha256(self.operator_sha256, field_name="operator_sha256")
        if self.gram_rank not in range(4) or self.effective_operator_rank not in range(16):
            raise ValueError("Matrix invariant capacity comparison invariant rank differs")
        for name in (
            "trace_budget",
            "trace_relative_error",
            "induced_norm_ratio",
            "semigroup_gain",
            "projection_gap_ratio",
            "self_adjoint_residual",
            "identity_residual",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for name in ("gram_min_eigenvalue", "operator_min_eigenvalue"):
            validate_decimal(getattr(self, name), field_name=name)
        if self.restricted_condition_number is not None:
            validate_decimal(
                self.restricted_condition_number,
                field_name="restricted_condition_number",
                minimum=Decimal(1),
            )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.admissible != (not self.reason_codes):
            raise ValueError("Matrix invariant capacity comparison invariant admissibility differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonCandidateFit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-candidate-fit'

    fit_id: str
    role: str
    held_field_index: int | None
    training_field_indices: tuple[int, ...]
    validation_loss: Decimal | None
    invariants: MatrixResponseInvariantCapacityComparisonOperatorInvariants | None
    admissible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.fit_id, field_name="fit_id")
        if self.role not in {"FOLD", "REFIT"}:
            raise ValueError("Matrix invariant capacity comparison candidate-fit role differs")
        if tuple(sorted(set(self.training_field_indices))) != self.training_field_indices:
            raise ValueError("Matrix invariant capacity comparison candidate-fit training fields differ")
        if self.role == "FOLD":
            if self.held_field_index not in range(6) or len(self.training_field_indices) != 5:
                raise ValueError("Matrix invariant capacity comparison fold geometry differs")
        elif self.held_field_index is not None or self.training_field_indices != tuple(range(6)):
            raise ValueError("Matrix invariant capacity comparison refit geometry differs")
        if self.validation_loss is not None:
            validate_decimal(self.validation_loss, field_name="validation_loss", minimum=Decimal(0))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.admissible != (self.invariants is not None and not self.reason_codes):
            raise ValueError("Matrix invariant capacity comparison candidate-fit admissibility differs")
        if self.role == "FOLD" and self.admissible != (self.validation_loss is not None):
            raise ValueError("Matrix invariant capacity comparison fold loss/admissibility differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonLambdaCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-lambda-candidate'

    candidate_id: str
    regularization: Decimal
    folds: tuple[MatrixResponseInvariantCapacityComparisonCandidateFit, ...]
    refit: MatrixResponseInvariantCapacityComparisonCandidateFit
    mean_validation_loss: Decimal | None
    standard_error: Decimal | None
    eligible: bool
    in_one_se_set: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_decimal(self.regularization, field_name="regularization", minimum=Decimal(0))
        require_sorted_unique_ids(self.folds, attribute="fit_id", field_name="folds")
        if len(self.folds) != 6 or any(value.role != "FOLD" for value in self.folds):
            raise ValueError("Matrix invariant capacity comparison lambda fold roster differs")
        if self.refit.role != "REFIT":
            raise ValueError("Matrix invariant capacity comparison lambda refit differs")
        for name in ("mean_validation_loss", "standard_error"):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal(0))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = all(value.admissible for value in self.folds) and self.refit.admissible
        if self.eligible != expected or self.eligible != (not self.reason_codes):
            raise ValueError("Matrix invariant capacity comparison lambda eligibility differs")
        if self.eligible != (
            self.mean_validation_loss is not None and self.standard_error is not None
        ):
            raise ValueError("Matrix invariant capacity comparison lambda score eligibility differs")
        if self.in_one_se_set and not self.eligible:
            raise ValueError("Matrix invariant capacity comparison ineligible lambda cannot enter one-SE set")


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonOriginSelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-origin-selection'

    selection_id: str
    trajectory_id: str
    view_id: str
    origin_step: int
    candidates: tuple[MatrixResponseInvariantCapacityComparisonLambdaCandidate, ...]
    selected_regularization: Decimal | None
    selected_operator_sha256: str | None
    selected_invariants: MatrixResponseInvariantCapacityComparisonOperatorInvariants | None
    admissible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("selection_id", "trajectory_id", "view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.candidates, attribute="candidate_id", field_name="candidates"
        )
        if len(self.candidates) != 12 or self.origin_step < 0:
            raise ValueError("Matrix invariant capacity comparison origin selection geometry differs")
        if self.selected_regularization is not None:
            validate_decimal(
                self.selected_regularization,
                field_name="selected_regularization",
                minimum=Decimal(0),
            )
        if self.selected_operator_sha256 is not None:
            validate_sha256(self.selected_operator_sha256, field_name="selected_operator_sha256")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = (
            self.selected_regularization is not None
            and self.selected_operator_sha256 is not None
            and self.selected_invariants is not None
            and self.selected_invariants.admissible
        )
        if self.admissible != expected or self.admissible != (not self.reason_codes):
            raise ValueError("Matrix invariant capacity comparison origin selection admissibility differs")
        selected = tuple(
            value
            for value in self.candidates
            if self.selected_regularization is not None
            and value.regularization == self.selected_regularization
        )
        if self.admissible and (
            len(selected) != 1
            or not selected[0].eligible
            or not selected[0].in_one_se_set
            or selected[0].refit.invariants != self.selected_invariants
            or self.selected_invariants is None
            or self.selected_operator_sha256 != self.selected_invariants.operator_sha256
        ):
            raise ValueError("Matrix invariant capacity comparison selected candidate binding differs")
        if not self.admissible and any(value.in_one_se_set for value in self.candidates):
            raise ValueError("Matrix invariant capacity comparison absent selection has a one-SE candidate")


@dataclass(frozen=True, slots=True)
class _Example:
    field_index: int
    kappa_index: int
    horizon_steps: int
    initial: RealArray
    observed: RealArray
    scale: float
    elapsed: float


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonOriginFitResult:
    record: MatrixResponseInvariantCapacityComparisonOriginSelection
    operator: RealArray | None
    gram: RealArray | None


@lru_cache(maxsize=1)
def gram_operator_basis() -> tuple[RealArray, tuple[tuple[int, int], ...]]:
    """Return the exact symmetric-Gram-to-double-commutator operator map."""

    basis = traceless_hermitian_basis(4)
    zero = np.zeros((4, 4), dtype="<c16")
    diagonal: list[RealArray] = []
    for index in range(15):
        triplet = np.stack((basis[index], zero, zero))
        diagonal.append(traceless_operator(np.asarray(triplet, dtype="<c16")))
    coordinates: list[tuple[int, int]] = []
    operators: list[RealArray] = []
    for left in range(15):
        coordinates.append((left, left))
        operators.append(diagonal[left])
        for right in range(left + 1, 15):
            triplet = np.stack((basis[left] + basis[right], zero, zero))
            operators.append(
                traceless_operator(np.asarray(triplet, dtype="<c16"))
                - diagonal[left]
                - diagonal[right]
            )
            coordinates.append((left, right))
    values = np.ascontiguousarray(np.stack(operators), dtype="<f8")
    values.setflags(write=False)
    return values, tuple(coordinates)


@lru_cache(maxsize=1)
def _orthonormal_operator_coordinates() -> tuple[RealArray, RealArray]:
    """Whiten Gram coordinates in the invariant operator-Frobenius metric."""

    gram_basis, _ = gram_operator_basis()
    mapping = gram_basis.reshape(gram_basis.shape[0], -1).T
    left, singular, right_adjoint = np.linalg.svd(mapping, full_matrices=False)
    if singular[-1] <= np.finfo(np.float64).eps * singular[0]:
        raise RuntimeError("Matrix invariant capacity comparison Gram-to-operator map is not injective")
    # Canonicalize otherwise arbitrary column signs without affecting the
    # operator metric or the inverse coordinate map.
    for index in range(left.shape[1]):
        pivot = int(np.argmax(np.abs(left[:, index])))
        if left[pivot, index] < 0:
            left[:, index] *= -1.0
            right_adjoint[index] *= -1.0
    operator_basis = np.ascontiguousarray(left.T.reshape(120, 15, 15), dtype="<f8")
    coefficient_map = np.ascontiguousarray(right_adjoint.T @ np.diag(1.0 / singular), dtype="<f8")
    if not np.allclose(
        operator_basis.reshape(120, -1) @ operator_basis.reshape(120, -1).T,
        np.eye(120),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Matrix invariant capacity comparison operator-coordinate whitening differs")
    operator_basis.setflags(write=False)
    coefficient_map.setflags(write=False)
    return operator_basis, coefficient_map


def operator_from_gram(gram: RealArray) -> RealArray:
    values = np.asarray(gram, dtype=np.float64)
    if values.shape != (15, 15) or not np.isfinite(values).all():
        raise ValueError("Matrix invariant capacity comparison Gram geometry differs")
    operator_basis, coordinates = gram_operator_basis()
    coefficients = np.asarray([values[left, right] for left, right in coordinates])
    return np.ascontiguousarray(
        np.tensordot(coefficients, operator_basis, axes=(0, 0)), dtype="<f8"
    )


def _gram_from_coefficients(coefficients: RealArray) -> RealArray:
    _, coordinates = gram_operator_basis()
    if coefficients.shape != (len(coordinates),):
        raise ValueError("Matrix invariant capacity comparison Gram coefficient geometry differs")
    gram = np.zeros((15, 15), dtype=np.float64)
    for value, (left, right) in zip(coefficients, coordinates, strict=True):
        gram[left, right] = value
        gram[right, left] = value
    return gram


def _examples(
    *,
    propagation: SixMatrixResponseTransientControlledInvarianceProbePropagation,
    origin_step: int,
    horizons: tuple[int, ...],
    field_indices: Sequence[int],
    floor: float,
    multiplier: int,
) -> tuple[_Example, ...]:
    output: list[_Example] = []
    origin_offset = multiplier * (origin_step - propagation.start_step)
    for field_index in field_indices:
        for kappa_index, kappa in enumerate(propagation.kappas):
            initial = propagation.states[field_index, kappa_index, origin_offset]
            initial_coordinates = encode_traceless_probe(initial)
            for horizon in horizons:
                endpoint = origin_offset + multiplier * horizon
                if endpoint >= propagation.states.shape[2]:
                    continue
                observed = propagation.states[field_index, kappa_index, endpoint]
                change = float(np.linalg.norm(observed - initial))
                if change < floor:
                    continue
                output.append(
                    _Example(
                        field_index=field_index,
                        kappa_index=kappa_index,
                        horizon_steps=horizon,
                        initial=initial_coordinates,
                        observed=encode_traceless_probe(observed),
                        scale=max(change, floor),
                        elapsed=kappa * multiplier * horizon * propagation.timestep,
                    )
                )
    return tuple(output)


def _project_and_validate(
    *,
    coefficients: RealArray,
    trace_budget: float,
    elapsed_values: Sequence[float],
    config: InvariantCapacityComparatorStudyConfig,
) -> tuple[RealArray | None, RealArray | None, MatrixResponseInvariantCapacityComparisonOperatorInvariants | None, tuple[str, ...]]:
    reasons: set[str] = set()
    tolerance = float(config.operator_tolerance)
    floor = float(config.trace_floor)
    gram_raw = _gram_from_coefficients(np.asarray(coefficients, dtype=np.float64))
    gram_raw = (gram_raw + gram_raw.T) * 0.5
    if not np.isfinite(gram_raw).all() or not np.isfinite(trace_budget):
        return None, None, None, ("nonfinite-candidate",)
    eigenvalues, eigenvectors = np.linalg.eigh(gram_raw)
    positive = np.maximum(eigenvalues, 0.0)
    positive_count = int(np.sum(positive > floor * max(1.0, float(np.max(positive)))))
    gap_ratio = 1.0
    if positive_count > config.generic_gram_rank:
        denominator = max(float(positive[-1]), floor)
        gap_ratio = float((positive[-3] - positive[-4]) / denominator)
        if gap_ratio < float(config.projection_gap_min):
            reasons.add("rank-three-projection-ambiguous")
    retained = np.zeros_like(positive)
    retained[-config.generic_gram_rank :] = positive[-config.generic_gram_rank :]
    gram = np.asarray((eigenvectors * retained) @ eigenvectors.T, dtype=np.float64)
    operator = operator_from_gram(gram)
    candidate_trace = float(np.trace(operator))
    if trace_budget <= floor or candidate_trace <= floor:
        reasons.add("trace-budget-unresolved")
    if reasons:
        return None, None, None, tuple(sorted(reasons))
    scale = trace_budget / candidate_trace
    gram = np.ascontiguousarray(gram * scale, dtype="<f8")
    operator = np.ascontiguousarray(operator * scale, dtype="<f8")
    gram_values = np.linalg.eigvalsh((gram + gram.T) * 0.5)
    operator_symmetric = (operator + operator.T) * 0.5
    operator_values = np.linalg.eigvalsh(operator_symmetric)
    gram_scale = max(float(np.max(np.abs(gram_values))), floor)
    operator_scale = max(float(np.max(np.abs(operator_values))), floor)
    gram_rank = int(np.sum(gram_values > tolerance * gram_scale))
    effective_rank = int(np.sum(operator_values > tolerance * operator_scale))
    trace_error = abs(float(np.trace(operator)) - trace_budget) / max(abs(trace_budget), floor)
    induced_ratio = float(np.max(np.abs(operator_values))) / max(abs(trace_budget), floor)
    self_adjoint = float(np.linalg.norm(operator - operator.T)) / max(
        float(np.linalg.norm(operator)), floor
    )
    minimum_operator = float(np.min(operator_values))
    minimum_gram = float(np.min(gram_values))
    minimum_elapsed = min(elapsed_values) if elapsed_values else 0.0
    semigroup_gain = float(np.exp(-minimum_elapsed * minimum_operator))
    positive_operator = operator_values[operator_values > tolerance * operator_scale]
    condition = (
        None
        if positive_operator.size == 0
        else float(np.max(positive_operator) / np.min(positive_operator))
    )
    if gram_rank > config.generic_gram_rank:
        reasons.add("gram-rank-exceeded")
    if minimum_gram < -tolerance * gram_scale:
        reasons.add("gram-not-positive")
    if minimum_operator < -tolerance * operator_scale:
        reasons.add("operator-not-positive")
    if trace_error > tolerance:
        reasons.add("trace-budget-mismatch")
    if induced_ratio > 1.0 + tolerance:
        reasons.add("induced-norm-bound-failed")
    if self_adjoint > tolerance:
        reasons.add("operator-not-self-adjoint")
    if semigroup_gain > 1.0 + tolerance:
        reasons.add("semigroup-amplification")
    invariants = MatrixResponseInvariantCapacityComparisonOperatorInvariants(
        gram_sha256=_digest(gram),
        operator_sha256=_digest(operator),
        gram_rank=gram_rank,
        effective_operator_rank=effective_rank,
        trace_budget=_decimal(trace_budget),
        trace_relative_error=_decimal(trace_error),
        induced_norm_ratio=_decimal(induced_ratio),
        semigroup_gain=_decimal(semigroup_gain),
        gram_min_eigenvalue=_decimal(minimum_gram),
        operator_min_eigenvalue=_decimal(minimum_operator),
        projection_gap_ratio=_decimal(max(gap_ratio, 0.0)),
        self_adjoint_residual=_decimal(self_adjoint),
        identity_residual=Decimal(0),
        restricted_condition_number=None if condition is None else _decimal(max(condition, 1.0)),
        admissible=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )
    if reasons:
        return None, None, invariants, tuple(sorted(reasons))
    gram.setflags(write=False)
    operator.setflags(write=False)
    return operator, gram, invariants, ()


def _fit_operator(
    *,
    examples: Sequence[_Example],
    trace_budget: float,
    regularization: float,
    config: InvariantCapacityComparatorStudyConfig,
) -> tuple[RealArray | None, RealArray | None, MatrixResponseInvariantCapacityComparisonOperatorInvariants | None, tuple[str, ...]]:
    if not examples:
        return None, None, None, ("no-resolved-development-examples",)
    operator_basis, coefficient_map = _orthonormal_operator_coordinates()
    design_rows: list[RealArray] = []
    targets: list[RealArray] = []
    for example in examples:
        divisor = example.scale * max(trace_budget, float(config.trace_floor))
        design_rows.append(
            np.stack(tuple(value @ example.initial for value in operator_basis), axis=1) / divisor
        )
        targets.append(((example.initial - example.observed) / example.elapsed) / divisor)
    design = np.concatenate(design_rows, axis=0)
    target = np.concatenate(targets)
    count = max(len(examples), 1)
    trace_scale = max(trace_budget, float(config.trace_floor))
    try:
        if regularization == 0:
            operator_coordinates = np.linalg.lstsq(design, target, rcond=None)[0]
        else:
            # Solve the dimensionless Tikhonov problem as an augmented SVD,
            # avoiding covariance-breaking normal-equation condition squaring.
            augmented_design = np.vstack(
                (
                    design / sqrt(count),
                    (sqrt(regularization) / trace_scale) * np.eye(design.shape[1]),
                )
            )
            augmented_target = np.concatenate(
                (target / sqrt(count), np.zeros(design.shape[1], dtype=np.float64))
            )
            operator_coordinates = np.linalg.lstsq(augmented_design, augmented_target, rcond=None)[
                0
            ]
    except np.linalg.LinAlgError:
        return None, None, None, ("linear-solve-failed",)
    coefficients = coefficient_map @ operator_coordinates
    return _project_and_validate(
        coefficients=np.asarray(coefficients, dtype=np.float64),
        trace_budget=trace_budget,
        elapsed_values=tuple(value.elapsed for value in examples),
        config=config,
    )


def _prediction_loss(*, operator: RealArray, examples: Sequence[_Example]) -> float:
    propagators = {
        elapsed: expm(-elapsed * operator) for elapsed in {value.elapsed for value in examples}
    }
    losses = []
    for example in examples:
        prediction = propagators[example.elapsed] @ example.initial
        losses.append(float(np.sum(np.square(prediction - example.observed))) / example.scale**2)
    return _mean(losses)


def fit_comparator_origin(
    *,
    trajectory_id: str,
    view_id: str,
    origin_step: int,
    propagation: SixMatrixResponseTransientControlledInvarianceProbePropagation,
    y_operator: RealArray,
    multiplier: int,
    config: InvariantCapacityComparatorStudyConfig,
) -> MatrixResponseInvariantCapacityComparisonOriginFitResult:
    """Fit all candidates, exclude only invalid lambdas, and apply one-SE selection."""

    validate_stable_id(trajectory_id, field_name="trajectory_id")
    validate_stable_id(view_id, field_name="view_id")
    if multiplier not in {1, 2} or origin_step not in config.forecast_origin_steps:
        raise ValueError("Matrix invariant capacity comparison fit coordinate differs")
    y_value = np.asarray(y_operator, dtype=np.float64)
    if y_value.shape != (15, 15) or not np.isfinite(y_value).all():
        raise ValueError("Matrix invariant capacity comparison Y operator differs")
    trace_budget = float(np.trace(y_value))
    all_examples = _examples(
        propagation=propagation,
        origin_step=origin_step,
        horizons=config.forecast_horizon_steps,
        field_indices=range(config.probe_development_count),
        floor=float(config.probe_numeric_floor),
        multiplier=multiplier,
    )
    candidates: list[MatrixResponseInvariantCapacityComparisonLambdaCandidate] = []
    operators: dict[int, RealArray] = {}
    grams: dict[int, RealArray] = {}
    for lambda_index, lambda_decimal in enumerate(config.generic_lambda_grid):
        regularization = float(lambda_decimal)
        folds: list[MatrixResponseInvariantCapacityComparisonCandidateFit] = []
        for held_field in range(config.probe_development_count):
            training_fields = tuple(
                value for value in range(config.probe_development_count) if value != held_field
            )
            training = tuple(value for value in all_examples if value.field_index != held_field)
            validation = tuple(value for value in all_examples if value.field_index == held_field)
            operator, _, invariants, reasons = _fit_operator(
                examples=training,
                trace_budget=trace_budget,
                regularization=regularization,
                config=config,
            )
            validation_loss = (
                None
                if operator is None or not validation
                else _prediction_loss(operator=operator, examples=validation)
            )
            fold_reasons = set(reasons)
            if not validation:
                fold_reasons.add("no-resolved-validation-examples")
            if operator is None and not fold_reasons:
                fold_reasons.add("candidate-operator-missing")
            folds.append(
                MatrixResponseInvariantCapacityComparisonCandidateFit(
                    fit_id=(
                        f"matrix-invariant-capacity-comparison.fit.{trajectory_id}.{view_id}.o{origin_step}."
                        f"l{lambda_index:02d}.f{held_field}"
                    ),
                    role="FOLD",
                    held_field_index=held_field,
                    training_field_indices=training_fields,
                    validation_loss=(
                        None if validation_loss is None else _decimal(validation_loss)
                    ),
                    invariants=invariants,
                    admissible=(operator is not None and validation_loss is not None),
                    reason_codes=tuple(sorted(fold_reasons)),
                )
            )
        refit_operator, refit_gram, refit_invariants, refit_reasons = _fit_operator(
            examples=all_examples,
            trace_budget=trace_budget,
            regularization=regularization,
            config=config,
        )
        refit_reason_set = set(refit_reasons)
        if refit_operator is None and not refit_reason_set:
            refit_reason_set.add("candidate-refit-missing")
        refit = MatrixResponseInvariantCapacityComparisonCandidateFit(
            fit_id=(
                f"matrix-invariant-capacity-comparison.fit.{trajectory_id}.{view_id}.o{origin_step}.l{lambda_index:02d}.refit"
            ),
            role="REFIT",
            held_field_index=None,
            training_field_indices=tuple(range(config.probe_development_count)),
            validation_loss=None,
            invariants=refit_invariants,
            admissible=refit_operator is not None,
            reason_codes=tuple(sorted(refit_reason_set)),
        )
        eligible = all(value.admissible for value in folds) and refit.admissible
        losses = [
            float(value.validation_loss) for value in folds if value.validation_loss is not None
        ]
        mean_loss = _mean(losses) if eligible else None
        standard_error = (
            float(np.std(np.asarray(losses), ddof=1) / sqrt(len(losses))) if eligible else None
        )
        candidate_reasons = () if eligible else ("fold-or-refit-inadmissible",)
        candidates.append(
            MatrixResponseInvariantCapacityComparisonLambdaCandidate(
                candidate_id=(
                    f"matrix-invariant-capacity-comparison.candidate.{trajectory_id}.{view_id}.o{origin_step}.l{lambda_index:02d}"
                ),
                regularization=lambda_decimal,
                folds=tuple(sorted(folds, key=lambda value: value.fit_id)),
                refit=refit,
                mean_validation_loss=None if mean_loss is None else _decimal(mean_loss),
                standard_error=(None if standard_error is None else _decimal(standard_error)),
                eligible=eligible,
                in_one_se_set=False,
                reason_codes=candidate_reasons,
            )
        )
        if eligible:
            assert refit_operator is not None and refit_gram is not None
            operators[lambda_index] = refit_operator
            grams[lambda_index] = refit_gram
    eligible_indices = [index for index, value in enumerate(candidates) if value.eligible]
    if not eligible_indices:
        record = MatrixResponseInvariantCapacityComparisonOriginSelection(
            selection_id=f"matrix-invariant-capacity-comparison.selection.{trajectory_id}.{view_id}.o{origin_step}",
            trajectory_id=trajectory_id,
            view_id=view_id,
            origin_step=origin_step,
            candidates=tuple(sorted(candidates, key=lambda value: value.candidate_id)),
            selected_regularization=None,
            selected_operator_sha256=None,
            selected_invariants=None,
            admissible=False,
            reason_codes=("no-eligible-regularization",),
        )
        return MatrixResponseInvariantCapacityComparisonOriginFitResult(record=record, operator=None, gram=None)

    def eligible_loss(index: int) -> float:
        value = candidates[index].mean_validation_loss
        if value is None:
            raise AssertionError("Matrix invariant capacity comparison eligible candidate has no validation loss")
        return float(value)

    minimum_index = min(
        eligible_indices,
        key=lambda index: (
            eligible_loss(index),
            float(candidates[index].regularization),
        ),
    )
    minimum_standard_error = candidates[minimum_index].standard_error
    if minimum_standard_error is None:
        raise AssertionError("Matrix invariant capacity comparison eligible candidate has no standard error")
    threshold = eligible_loss(minimum_index) + float(minimum_standard_error)
    one_se_indices = [index for index in eligible_indices if eligible_loss(index) <= threshold]
    selected_index = max(one_se_indices, key=lambda index: float(candidates[index].regularization))
    candidates = [
        replace(value, in_one_se_set=index in one_se_indices)
        for index, value in enumerate(candidates)
    ]
    selected = candidates[selected_index]
    selected_invariants = selected.refit.invariants
    assert selected_invariants is not None
    operator = operators[selected_index]
    gram = grams[selected_index]
    record = MatrixResponseInvariantCapacityComparisonOriginSelection(
        selection_id=f"matrix-invariant-capacity-comparison.selection.{trajectory_id}.{view_id}.o{origin_step}",
        trajectory_id=trajectory_id,
        view_id=view_id,
        origin_step=origin_step,
        candidates=tuple(sorted(candidates, key=lambda value: value.candidate_id)),
        selected_regularization=selected.regularization,
        selected_operator_sha256=_digest(operator),
        selected_invariants=selected_invariants,
        admissible=True,
        reason_codes=(),
    )
    return MatrixResponseInvariantCapacityComparisonOriginFitResult(record=record, operator=operator, gram=gram)


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonProbeCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-probe-cell'

    cell_id: str
    field_index: int
    kappa: Decimal
    origin_step: int
    horizon_steps: int
    y_loss: Decimal
    comparator_loss: Decimal
    resolved: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.field_index not in range(6, 12) or self.origin_step < 0 or self.horizon_steps < 1:
            raise ValueError("Matrix invariant capacity comparison probe cell coordinate differs")
        if self.kappa not in (Decimal("0.25"), Decimal("0.5"), Decimal("1")):
            raise ValueError("Matrix invariant capacity comparison probe cell kappa differs")
        for name in ("kappa", "y_loss", "comparator_loss"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonTrajectoryScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-trajectory-score'

    score_id: str
    trajectory_id: str
    view_id: str
    event: bool
    selections: tuple[MatrixResponseInvariantCapacityComparisonOriginSelection, ...]
    cells: tuple[MatrixResponseInvariantCapacityComparisonProbeCell, ...]
    y_loss: Decimal | None
    comparator_loss: Decimal | None
    skill: Decimal | None
    kappa_y_over_c_ratios: tuple[Decimal, ...]
    field_y_over_c_ratios: tuple[Decimal, ...]
    horizon_skills: tuple[Decimal, ...]
    phase_skills: tuple[Decimal, ...]
    resolved_fraction: Decimal
    kappa_resolved_fractions: tuple[Decimal, ...]
    category: MatrixResponseInvariantCapacityComparisonContrastCategory
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("score_id", "trajectory_id", "view_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.selections, attribute="selection_id", field_name="selections"
        )
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if len(self.selections) != 6:
            raise ValueError("Matrix invariant capacity comparison trajectory origin roster differs")
        if tuple(sorted(value.origin_step for value in self.selections)) != (
            816,
            848,
            880,
            896,
            928,
            960,
        ):
            raise ValueError("Matrix invariant capacity comparison trajectory origin coordinates differ")
        if any(
            value.trajectory_id != self.trajectory_id or value.view_id != self.view_id
            for value in self.selections
        ):
            raise ValueError("Matrix invariant capacity comparison trajectory selection coordinate differs")
        for name in ("y_loss", "comparator_loss"):
            value = getattr(self, name)
            if value is not None:
                validate_decimal(value, field_name=name, minimum=Decimal(0))
        if self.skill is not None:
            validate_decimal(self.skill, field_name="skill")
        for value in (
            *self.kappa_y_over_c_ratios,
            *self.field_y_over_c_ratios,
            *self.kappa_resolved_fractions,
        ):
            validate_decimal(value, field_name="trajectory_ratio", minimum=Decimal(0))
        for value in (*self.horizon_skills, *self.phase_skills):
            validate_decimal(value, field_name="trajectory_skill")
        validate_decimal(self.resolved_fraction, field_name="resolved_fraction", minimum=Decimal(0))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid != (not self.reason_codes):
            raise ValueError("Matrix invariant capacity comparison trajectory validity differs")
        complete = (
            self.y_loss is not None and self.comparator_loss is not None and self.skill is not None
        )
        if self.category is MatrixResponseInvariantCapacityComparisonContrastCategory.COMPARATOR_INADMISSIBLE:
            if complete or self.valid or self.cells:
                raise ValueError("Matrix invariant capacity comparison inadmissible comparator score differs")
        elif (
            not complete
            or len(self.cells) != 324
            or len(self.kappa_y_over_c_ratios) != 3
            or len(self.field_y_over_c_ratios) != 6
            or len(self.horizon_skills) != 3
            or len(self.phase_skills) != 3
            or len(self.kappa_resolved_fractions) != 3
        ):
            raise ValueError("Matrix invariant capacity comparison admissible comparator score is incomplete")


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonTrajectoryEvaluation:
    record: MatrixResponseInvariantCapacityComparisonTrajectoryScore
    operators: tuple[RealArray | None, ...]
    grams: tuple[RealArray | None, ...]


def _nested_loss(
    cells: Sequence[MatrixResponseInvariantCapacityComparisonProbeCell],
    attribute: str,
    *,
    field_indices: Sequence[int] = tuple(range(6, 12)),
) -> float:
    channels: list[float] = []
    for field in field_indices:
        for kappa in (Decimal("0.25"), Decimal("0.5"), Decimal("1.0")):
            values = [
                float(getattr(cell, attribute))
                for cell in cells
                if cell.field_index == field and cell.kappa == kappa and cell.resolved
            ]
            if values:
                channels.append(_mean(values))
    return _mean(channels)


def _ratio(numerator: float, denominator: float) -> float:
    tiny = np.finfo(np.float64).tiny
    if numerator <= tiny and denominator <= tiny:
        return 1.0
    return numerator / max(denominator, tiny)


def _skill(y_loss: float, comparator_loss: float) -> float:
    tiny = np.finfo(np.float64).tiny
    return float(log(max(comparator_loss, tiny) / max(y_loss, tiny)))


def score_trajectory(
    *,
    trajectory_id: str,
    view_id: str,
    event: bool,
    y_path: ComplexArray,
    propagation: SixMatrixResponseTransientControlledInvarianceProbePropagation,
    multiplier: int,
    config: InvariantCapacityComparatorStudyConfig,
) -> MatrixResponseInvariantCapacityComparisonTrajectoryEvaluation:
    """Fit and score one complete path without treating probe cells as units."""

    values = np.asarray(y_path)
    expected_steps = multiplier * (config.probe_end_step - config.probe_start_step) + 1
    if values.shape != (expected_steps, 3, 4, 4) or values.dtype != np.dtype("complex128"):
        raise ValueError("Matrix invariant capacity comparison Y path geometry differs")
    selections: list[MatrixResponseInvariantCapacityComparisonOriginSelection] = []
    operators: list[RealArray | None] = []
    grams: list[RealArray | None] = []
    for origin in config.forecast_origin_steps:
        offset = multiplier * (origin - config.probe_start_step)
        result = fit_comparator_origin(
            trajectory_id=trajectory_id,
            view_id=view_id,
            origin_step=origin,
            propagation=propagation,
            y_operator=traceless_operator(values[offset]),
            multiplier=multiplier,
            config=config,
        )
        selections.append(result.record)
        operators.append(result.operator)
        grams.append(result.gram)
    ordered_selections = tuple(sorted(selections, key=lambda value: value.selection_id))
    if any(value is None for value in operators) or any(value is None for value in grams):
        record = MatrixResponseInvariantCapacityComparisonTrajectoryScore(
            score_id=f"matrix-invariant-capacity-comparison.score.{trajectory_id}.{view_id}",
            trajectory_id=trajectory_id,
            view_id=view_id,
            event=event,
            selections=ordered_selections,
            cells=(),
            y_loss=None,
            comparator_loss=None,
            skill=None,
            kappa_y_over_c_ratios=(),
            field_y_over_c_ratios=(),
            horizon_skills=(),
            phase_skills=(),
            resolved_fraction=Decimal(0),
            kappa_resolved_fractions=(),
            category=MatrixResponseInvariantCapacityComparisonContrastCategory.COMPARATOR_INADMISSIBLE,
            valid=False,
            reason_codes=("required-origin-comparator-inadmissible",),
        )
        return MatrixResponseInvariantCapacityComparisonTrajectoryEvaluation(
            record=record, operators=tuple(operators), grams=tuple(grams)
        )
    operator_by_origin = dict(zip(config.forecast_origin_steps, operators, strict=True))
    cells: list[MatrixResponseInvariantCapacityComparisonProbeCell] = []
    for origin in config.forecast_origin_steps:
        offset = multiplier * (origin - config.probe_start_step)
        y_operator = traceless_operator(values[offset])
        comparator = operator_by_origin[origin]
        assert comparator is not None
        for field in range(config.probe_development_count, config.probe_field_count):
            for kappa_index, kappa in enumerate(map(float, config.probe_kappas)):
                initial = propagation.states[field, kappa_index, offset]
                for horizon in config.forecast_horizon_steps:
                    endpoint = offset + multiplier * horizon
                    if endpoint >= propagation.states.shape[2]:
                        continue
                    observed = propagation.states[field, kappa_index, endpoint]
                    horizon_time = multiplier * horizon * propagation.timestep
                    y_prediction = heat_predict_probe(
                        operator=y_operator,
                        probe=initial,
                        kappa=kappa,
                        horizon_time=horizon_time,
                    )
                    c_prediction = heat_predict_probe(
                        operator=comparator,
                        probe=initial,
                        kappa=kappa,
                        horizon_time=horizon_time,
                    )
                    y_cell_loss, resolved = normalized_increment_loss(
                        predicted=y_prediction,
                        observed=observed,
                        initial=initial,
                        floor=float(config.probe_numeric_floor),
                    )
                    c_cell_loss, _ = normalized_increment_loss(
                        predicted=c_prediction,
                        observed=observed,
                        initial=initial,
                        floor=float(config.probe_numeric_floor),
                    )
                    cells.append(
                        MatrixResponseInvariantCapacityComparisonProbeCell(
                            cell_id=(
                                f"matrix-invariant-capacity-comparison.cell.{trajectory_id}.{view_id}.f{field:02d}."
                                f"k{kappa_index}.o{origin}.h{horizon}"
                            ),
                            field_index=field,
                            kappa=_decimal(kappa),
                            origin_step=origin,
                            horizon_steps=horizon,
                            y_loss=_decimal(y_cell_loss),
                            comparator_loss=_decimal(c_cell_loss),
                            resolved=resolved,
                        )
                    )
    ordered = tuple(sorted(cells, key=lambda value: value.cell_id))
    reasons: set[str] = set()
    resolved_fraction = sum(value.resolved for value in ordered) / len(ordered)
    kappa_resolved = tuple(
        sum(value.resolved for value in ordered if value.kappa == kappa)
        / sum(value.kappa == kappa for value in ordered)
        for kappa in config.probe_kappas
    )
    if resolved_fraction < float(config.probe_resolved_fraction_min) or min(kappa_resolved) < float(
        config.probe_kappa_resolved_fraction_min
    ):
        reasons.add("insufficient-resolved-heldout-cells")
    y_loss = _nested_loss(ordered, "y_loss")
    c_loss = _nested_loss(ordered, "comparator_loss")
    kappa_ratios = tuple(
        _ratio(
            _nested_loss(tuple(value for value in ordered if value.kappa == kappa), "y_loss"),
            _nested_loss(
                tuple(value for value in ordered if value.kappa == kappa), "comparator_loss"
            ),
        )
        for kappa in config.probe_kappas
    )
    field_ratios = tuple(
        _ratio(
            _nested_loss(ordered, "y_loss", field_indices=(field,)),
            _nested_loss(ordered, "comparator_loss", field_indices=(field,)),
        )
        for field in range(config.probe_development_count, config.probe_field_count)
    )
    horizon_skills = tuple(
        _skill(
            _nested_loss(
                tuple(value for value in ordered if value.horizon_steps == horizon), "y_loss"
            ),
            _nested_loss(
                tuple(value for value in ordered if value.horizon_steps == horizon),
                "comparator_loss",
            ),
        )
        for horizon in config.forecast_horizon_steps
    )
    phase_bands = ((816, 848), (880, 896), (928, 960))
    phase_skills = tuple(
        _skill(
            _nested_loss(
                tuple(value for value in ordered if low <= value.origin_step <= high), "y_loss"
            ),
            _nested_loss(
                tuple(value for value in ordered if low <= value.origin_step <= high),
                "comparator_loss",
            ),
        )
        for low, high in phase_bands
    )
    pooled_ratio = _ratio(y_loss, c_loss)
    y_privileged = bool(
        pooled_ratio <= float(config.gate_y_ratio_max)
        and all(value < 1.0 for value in kappa_ratios)
        and sum(value < 1.0 for value in field_ratios) >= config.gate_field_min
        and min(horizon_skills) > 0
        and min(phase_skills) > 0
    )
    equivalent = bool(
        float(config.gate_y_ratio_max) < pooled_ratio < float(config.equivalence_ratio_upper)
        and all(
            float(config.gate_y_ratio_max) < value < float(config.equivalence_ratio_upper)
            for value in kappa_ratios
        )
        and sum(
            float(config.gate_y_ratio_max) < value < float(config.equivalence_ratio_upper)
            for value in field_ratios
        )
        >= config.gate_field_min
    )
    generic_dominant = bool(
        _ratio(c_loss, y_loss) <= float(config.gate_y_ratio_max)
        and all(value > 1.0 for value in kappa_ratios)
        and sum(value > 1.0 for value in field_ratios) >= config.gate_field_min
        and max(horizon_skills) < 0
        and max(phase_skills) < 0
    )
    category = (
        MatrixResponseInvariantCapacityComparisonContrastCategory.Y_PRIVILEGED
        if y_privileged
        else MatrixResponseInvariantCapacityComparisonContrastCategory.PRACTICALLY_EQUIVALENT
        if equivalent
        else MatrixResponseInvariantCapacityComparisonContrastCategory.GENERIC_DOMINANT
        if generic_dominant
        else MatrixResponseInvariantCapacityComparisonContrastCategory.MIXED
    )
    record = MatrixResponseInvariantCapacityComparisonTrajectoryScore(
        score_id=f"matrix-invariant-capacity-comparison.score.{trajectory_id}.{view_id}",
        trajectory_id=trajectory_id,
        view_id=view_id,
        event=event,
        selections=ordered_selections,
        cells=ordered,
        y_loss=_decimal(y_loss),
        comparator_loss=_decimal(c_loss),
        skill=_decimal(_skill(y_loss, c_loss)),
        kappa_y_over_c_ratios=tuple(_decimal(value) for value in kappa_ratios),
        field_y_over_c_ratios=tuple(_decimal(value) for value in field_ratios),
        horizon_skills=tuple(_decimal(value) for value in horizon_skills),
        phase_skills=tuple(_decimal(value) for value in phase_skills),
        resolved_fraction=_decimal(resolved_fraction),
        kappa_resolved_fractions=tuple(_decimal(value) for value in kappa_resolved),
        category=category,
        valid=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )
    return MatrixResponseInvariantCapacityComparisonTrajectoryEvaluation(
        record=record,
        operators=tuple(operators),
        grams=tuple(grams),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonCarriedControlledInvarianceEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-carried-controlled-invariance-evidence'

    evidence_id: str
    parent_terminal_sha256: str
    parent_scores_sha256: str
    parent_roster_sha256: str
    absolute_y_prediction_passed: bool
    radius_specificity_passed: bool
    shuffle_specificity_passed: bool
    probe_numerical_view_passed: bool
    parent_prediction_qualification_unevaluable_only_because_generic_missing: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        for name in ("parent_terminal_sha256", "parent_scores_sha256", "parent_roster_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if not all(
            (
                self.absolute_y_prediction_passed,
                self.radius_specificity_passed,
                self.shuffle_specificity_passed,
                self.probe_numerical_view_passed,
                self.parent_prediction_qualification_unevaluable_only_because_generic_missing,
            )
        ):
            raise ValueError("Matrix invariant capacity comparison carried transient response evidence is incomplete")


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonNumericalAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-numerical-audit'

    audit_id: str
    audited_trajectory_ids: tuple[str, ...]
    comparator_complete: bool
    median_probe_state_difference: Decimal
    ordering_agreement: Decimal
    category_agreement: bool
    conjugation_loss_error: Decimal
    conjugation_operator_error: Decimal
    hermiticity_residual: Decimal
    trace_residual: Decimal
    identity_drift: Decimal
    norm_increase: Decimal
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        if tuple(sorted(set(self.audited_trajectory_ids))) != self.audited_trajectory_ids:
            raise ValueError("Matrix invariant capacity comparison numerical-audit roster differs")
        for name in (
            "median_probe_state_difference",
            "ordering_agreement",
            "conjugation_loss_error",
            "conjugation_operator_error",
            "hermiticity_residual",
            "trace_residual",
            "identity_drift",
            "norm_increase",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid != (self.comparator_complete and not self.reason_codes):
            raise ValueError("Matrix invariant capacity comparison numerical-audit validity differs")


def build_numerical_audit(
    *,
    audited_trajectory_ids: tuple[str, ...],
    comparator_complete: bool,
    normalized_probe_differences: Sequence[float],
    ordering_agreement: float,
    category_agreement: bool,
    conjugation_loss_error: float,
    conjugation_operator_error: float,
    hermiticity_residual: float,
    trace_residual: float,
    identity_drift: float,
    norm_increase: float,
    config: InvariantCapacityComparatorStudyConfig,
) -> MatrixResponseInvariantCapacityComparisonNumericalAudit:
    if not normalized_probe_differences:
        raise ValueError("Matrix invariant capacity comparison numerical audit has no probe differences")
    if len(audited_trajectory_ids) != config.half_step_control_count + 1:
        raise ValueError("Matrix invariant capacity comparison numerical-audit roster count differs")
    reasons: set[str] = set()
    median = float(np.median(np.asarray(normalized_probe_differences)))
    if median > float(config.probe_half_state_difference_max):
        reasons.add("probe-state-half-step-discordance")
    if ordering_agreement < float(config.probe_half_order_agreement_min):
        reasons.add("comparator-order-half-step-discordance")
    if not category_agreement:
        reasons.add("comparator-category-half-step-discordance")
    if max(conjugation_loss_error, conjugation_operator_error) > float(
        config.probe_conjugation_error_max
    ):
        reasons.add("comparator-unitary-covariance-discordance")
    if hermiticity_residual > float(config.probe_hermiticity_residual_max):
        reasons.add("probe-hermiticity-residual")
    if trace_residual > float(config.probe_trace_residual_max):
        reasons.add("probe-trace-residual")
    if identity_drift > float(config.probe_identity_drift_max):
        reasons.add("probe-identity-drift")
    if norm_increase > float(config.probe_norm_increase_max):
        reasons.add("probe-norm-increase")
    if not comparator_complete:
        reasons.add("required-numerical-comparator-inadmissible")
    return MatrixResponseInvariantCapacityComparisonNumericalAudit(
        audit_id="matrix-invariant-capacity-comparison.numerical-audit",
        audited_trajectory_ids=tuple(sorted(audited_trajectory_ids)),
        comparator_complete=comparator_complete,
        median_probe_state_difference=_decimal(median),
        ordering_agreement=_decimal(ordering_agreement),
        category_agreement=category_agreement,
        conjugation_loss_error=_decimal(conjugation_loss_error),
        conjugation_operator_error=_decimal(conjugation_operator_error),
        hermiticity_residual=_decimal(hermiticity_residual),
        trace_residual=_decimal(trace_residual),
        identity_drift=_decimal(identity_drift),
        norm_increase=_decimal(norm_increase),
        valid=comparator_complete and not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonMethodQualificationCase(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-method-qualification-case'

    case_id: str
    metric: Decimal
    bound: Decimal
    passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        validate_decimal(self.metric, field_name="metric", minimum=Decimal(0))
        validate_decimal(self.bound, field_name="bound", minimum=Decimal(0))
        if self.passed != (self.metric <= self.bound):
            raise ValueError("Matrix invariant capacity comparison method qualification case differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonComparatorMethodPackage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-comparator-method-package'

    package_id: str
    config_fingerprint: str
    implementation_commit: str
    implementation_tree: str
    source_locator_count: int
    source_access_count: int
    cases: tuple[MatrixResponseInvariantCapacityComparisonMethodQualificationCase, ...]
    qualified: bool
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.package_id, field_name="package_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        for name in ("implementation_commit", "implementation_tree"):
            value = getattr(self, name)
            if len(value) != 40:
                raise ValueError("Matrix invariant capacity comparison method package Git identity differs")
            int(value, 16)
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        if (
            self.source_locator_count != 0
            or self.source_access_count != 0
            or self.qualified != all(value.passed for value in self.cases)
            or not self.qualified
            or self.grants_authority
        ):
            raise ValueError("Matrix invariant capacity comparison method-package qualification differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseInvariantCapacityComparisonTerminalReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-invariant-capacity-comparison-terminal-report'

    report_id: str
    config_fingerprint: str
    method_package_sha256: str
    carried_transient_response_evidence: MatrixResponseInvariantCapacityComparisonCarriedControlledInvarianceEvidence
    event_score: MatrixResponseInvariantCapacityComparisonTrajectoryScore
    control_scores: tuple[MatrixResponseInvariantCapacityComparisonTrajectoryScore, ...]
    numerical_audit: MatrixResponseInvariantCapacityComparisonNumericalAudit
    event_rank: int | None
    terminal: MatrixResponseInvariantCapacityComparisonTerminal
    intervention_confirmation_scientific_precondition: str
    parent_action_count: int
    intervention_confirmation_task_count: int
    prospective_control_task_count: int
    permits_intervention_confirmation_execution: bool
    grants_authority: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        validate_sha256(self.method_package_sha256, field_name="method_package_sha256")
        require_sorted_unique_ids(
            self.control_scores, attribute="score_id", field_name="control_scores"
        )
        if not self.event_score.event or any(value.event for value in self.control_scores):
            raise ValueError("Matrix invariant capacity comparison terminal event/control roles differ")
        positive = self.terminal is MatrixResponseInvariantCapacityComparisonTerminal.TRANSIENT_RESPONSE_GEOMETRY_SUPPORTED
        if (
            self.intervention_confirmation_scientific_precondition != ("SATISFIED" if positive else "CLOSED")
            or self.parent_action_count != 0
            or self.intervention_confirmation_task_count != 0
            or self.prospective_control_task_count != 0
            or self.permits_intervention_confirmation_execution
            or self.grants_authority
        ):
            raise ValueError("Matrix invariant capacity comparison terminal descendant/effect boundary differs")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def _terminal_for_category(
    *,
    category: MatrixResponseInvariantCapacityComparisonContrastCategory,
    event_rank: int | None,
) -> MatrixResponseInvariantCapacityComparisonTerminal:
    if category is MatrixResponseInvariantCapacityComparisonContrastCategory.COMPARATOR_INADMISSIBLE:
        return MatrixResponseInvariantCapacityComparisonTerminal.COMPARATOR_DESIGN_INSUFFICIENT_UNEVALUABLE
    if category is MatrixResponseInvariantCapacityComparisonContrastCategory.PRACTICALLY_EQUIVALENT:
        return MatrixResponseInvariantCapacityComparisonTerminal.GENERIC_OPERATOR_COHERENCE_NOT_GEOMETRIC_PRIVILEGE
    if category is MatrixResponseInvariantCapacityComparisonContrastCategory.GENERIC_DOMINANT:
        return MatrixResponseInvariantCapacityComparisonTerminal.GENERIC_OPERATOR_OUTPERFORMS_Y_GEOMETRY
    if category is MatrixResponseInvariantCapacityComparisonContrastCategory.MIXED:
        return MatrixResponseInvariantCapacityComparisonTerminal.COMPARATOR_CONTRAST_MIXED
    if event_rank != 1:
        return MatrixResponseInvariantCapacityComparisonTerminal.EVENT_EXCEPTIONALITY_NOT_ESTABLISHED
    return MatrixResponseInvariantCapacityComparisonTerminal.TRANSIENT_RESPONSE_GEOMETRY_SUPPORTED


def finalize_invariant_capacity_comparison(
    *,
    config: InvariantCapacityComparatorStudyConfig,
    method_package_sha256: str,
    carried_transient_response_evidence: MatrixResponseInvariantCapacityComparisonCarriedControlledInvarianceEvidence,
    event_score: MatrixResponseInvariantCapacityComparisonTrajectoryScore,
    control_scores: tuple[MatrixResponseInvariantCapacityComparisonTrajectoryScore, ...],
    numerical_audit: MatrixResponseInvariantCapacityComparisonNumericalAudit,
) -> MatrixResponseInvariantCapacityComparisonTerminalReport:
    """Sole scientific finalizer with comparator invalidity before contrast."""

    validate_sha256(method_package_sha256, field_name="method_package_sha256")
    reasons: set[str] = set()
    corpus_structure_valid = bool(
        event_score.view_id == "matrix-invariant-capacity-comparison.view.primary"
        and all(value.view_id == "matrix-invariant-capacity-comparison.view.primary" for value in control_scores)
        and len(control_scores) == config.matched_control_primary_count
        and not any(value.event for value in control_scores)
        and len({value.trajectory_id for value in control_scores}) == len(control_scores)
    )
    comparator_complete = bool(
        event_score.category is not MatrixResponseInvariantCapacityComparisonContrastCategory.COMPARATOR_INADMISSIBLE
        and all(
            value.category is not MatrixResponseInvariantCapacityComparisonContrastCategory.COMPARATOR_INADMISSIBLE
            for value in control_scores
        )
    )
    if not corpus_structure_valid:
        terminal = MatrixResponseInvariantCapacityComparisonTerminal.NUMERICAL_VIEW_INVALID
        reasons.add("primary-score-corpus-structure-invalid")
        event_rank = None
    elif not comparator_complete or not numerical_audit.comparator_complete:
        terminal = MatrixResponseInvariantCapacityComparisonTerminal.COMPARATOR_DESIGN_INSUFFICIENT_UNEVALUABLE
        reasons.add("required-comparator-corpus-inadmissible")
        event_rank = None
    elif not event_score.valid or any(not value.valid for value in control_scores):
        terminal = MatrixResponseInvariantCapacityComparisonTerminal.NUMERICAL_VIEW_INVALID
        reasons.add("trajectory-scientific-view-invalid")
        event_rank = None
    elif not numerical_audit.valid:
        terminal = MatrixResponseInvariantCapacityComparisonTerminal.NUMERICAL_VIEW_INVALID
        reasons.update(numerical_audit.reason_codes)
        event_rank = None
    else:
        assert event_score.skill is not None
        event_skill = float(event_score.skill)
        control_skills: list[float] = []
        for value in control_scores:
            if value.skill is None:
                raise AssertionError("Matrix invariant capacity comparison comparator-complete control has no skill")
            control_skills.append(float(value.skill))
        event_rank = 1 + sum(value >= event_skill for value in control_skills)
        terminal = _terminal_for_category(category=event_score.category, event_rank=event_rank)
        if terminal is MatrixResponseInvariantCapacityComparisonTerminal.EVENT_EXCEPTIONALITY_NOT_ESTABLISHED:
            reasons.add("event-strict-rank-one-failed")
    return MatrixResponseInvariantCapacityComparisonTerminalReport(
        report_id="matrix-invariant-capacity-comparison.terminal-report",
        config_fingerprint=config.fingerprint(),
        method_package_sha256=method_package_sha256,
        carried_transient_response_evidence=carried_transient_response_evidence,
        event_score=event_score,
        control_scores=tuple(sorted(control_scores, key=lambda value: value.score_id)),
        numerical_audit=numerical_audit,
        event_rank=event_rank,
        terminal=terminal,
        intervention_confirmation_scientific_precondition=(
            "SATISFIED"
            if terminal is MatrixResponseInvariantCapacityComparisonTerminal.TRANSIENT_RESPONSE_GEOMETRY_SUPPORTED
            else "CLOSED"
        ),
        parent_action_count=0,
        intervention_confirmation_task_count=0,
        prospective_control_task_count=0,
        permits_intervention_confirmation_execution=False,
        grants_authority=False,
        reason_codes=tuple(sorted(reasons)),
    )


def _unitary_and_adjoint(seed: int) -> tuple[ComplexArray, RealArray]:
    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    draw = rng.normal(size=(4, 4)) + 1.0j * rng.normal(size=(4, 4))
    unitary, triangular = np.linalg.qr(draw)
    diagonal = np.diag(triangular)
    unitary = unitary * np.where(np.abs(diagonal) > 0, diagonal.conj() / np.abs(diagonal), 1.0)
    basis = traceless_hermitian_basis(4)
    rotated = np.asarray(unitary @ basis @ unitary.conj().T, dtype="<c16")
    adjoint = np.asarray(
        [[float(np.vdot(left, right).real) for right in rotated] for left in basis],
        dtype="<f8",
    )
    return np.asarray(unitary, dtype="<c16"), adjoint


def _qualification_propagation(
    *, operator: RealArray, seed: int, constant_first_field: bool
) -> SixMatrixResponseTransientControlledInvarianceProbePropagation:
    """Build an analytic source-free semigroup corpus for method qualification."""

    rng = np.random.Generator(np.random.PCG64DXSM(seed))
    orthonormal, _ = np.linalg.qr(rng.normal(size=(15, 12)))
    coordinates = np.asarray(orthonormal.T, dtype=np.float64)
    basis = traceless_hermitian_basis(4)
    states = np.empty((12, 3, 209, 4, 4), dtype="<c16")
    for kappa_index, kappa in enumerate((0.25, 0.5, 1.0)):
        for offset in range(209):
            kernel = expm(-(kappa * offset * 0.001) * operator)
            vectors = np.asarray(coordinates @ kernel.T, dtype=np.float64)
            states[:, kappa_index, offset] = np.tensordot(vectors, basis, axes=(1, 0))
    if constant_first_field:
        states[0] = np.broadcast_to(states[0, :, :1], states[0].shape)
    identity = np.eye(4, dtype="<c16") / 2.0
    identities = np.ascontiguousarray(np.broadcast_to(identity, (3, 209, 4, 4)), dtype="<c16")
    return SixMatrixResponseTransientControlledInvarianceProbePropagation(
        start_step=816,
        timestep=0.001,
        kappas=(0.25, 0.5, 1.0),
        states=states,
        identity_states=identities,
        maximum_hermiticity_residual=0.0,
        maximum_trace_residual=0.0,
        maximum_relative_norm_increase=0.0,
        maximum_identity_relative_drift=0.0,
    )


def qualify_comparator_method(
    *,
    config: InvariantCapacityComparatorStudyConfig,
    implementation_commit: str,
    implementation_tree: str,
) -> MatrixResponseInvariantCapacityComparisonComparatorMethodPackage:
    """Run source-inaccessible analytic checks and bind their exact code identity."""

    rng = np.random.Generator(np.random.PCG64DXSM(23081986))
    coefficients = rng.normal(size=(3, 15))
    gram = np.asarray(coefficients.T @ coefficients, dtype="<f8")
    basis = traceless_hermitian_basis(4)
    triplet = np.asarray(np.tensordot(coefficients, basis, axes=([-1], [0])), dtype="<c16")
    direct = traceless_operator(triplet)
    mapped = operator_from_gram(gram)
    direct_scale = max(float(np.linalg.norm(direct)), 1.0)
    direct_error = float(np.linalg.norm(mapped - direct) / direct_scale)
    mixing, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    mixed = mixing @ coefficients
    gram_scale = max(float(np.linalg.norm(gram)), 1.0)
    gauge_error = float(np.linalg.norm(mixed.T @ mixed - gram) / gram_scale)
    unitary, adjoint = _unitary_and_adjoint(91731)
    rotated_triplet = np.asarray(unitary @ triplet @ unitary.conj().T, dtype="<c16")
    rotated_direct = traceless_operator(rotated_triplet)
    transported = np.asarray(adjoint @ direct @ adjoint.T, dtype="<f8")
    covariance_error = float(np.linalg.norm(rotated_direct - transported) / direct_scale)
    identity = np.eye(4, dtype="<c16")
    identity_action = np.zeros_like(identity)
    for value in triplet:
        identity_action += (
            value @ (value @ identity - identity @ value)
            - (value @ identity - identity @ value) @ value
        )
    identity_error = float(np.linalg.norm(identity_action))
    trace_budget = float(np.trace(direct))
    _, coordinates = gram_operator_basis()
    gram_coefficients = np.asarray([gram[left, right] for left, right in coordinates])
    projected_operator, _, invariants, projection_reasons = _project_and_validate(
        coefficients=gram_coefficients,
        trace_budget=trace_budget,
        elapsed_values=(0.004, 0.016, 0.064),
        config=config,
    )
    if projected_operator is None or invariants is None or projection_reasons:
        raise RuntimeError("Matrix invariant capacity comparison analytic comparator fixture is inadmissible")
    analytic_propagation = _qualification_propagation(
        operator=direct, seed=31193, constant_first_field=False
    )
    analytic_fit = fit_comparator_origin(
        trajectory_id="matrix-invariant-capacity-comparison.fixture.complete-grid",
        view_id="matrix-invariant-capacity-comparison.view.fixture",
        origin_step=816,
        propagation=analytic_propagation,
        y_operator=direct,
        multiplier=1,
        config=config,
    )
    rotated_analytic_propagation = SixMatrixResponseTransientControlledInvarianceProbePropagation(
        start_step=analytic_propagation.start_step,
        timestep=analytic_propagation.timestep,
        kappas=analytic_propagation.kappas,
        states=np.asarray(unitary @ analytic_propagation.states @ unitary.conj().T, dtype="<c16"),
        identity_states=np.asarray(
            unitary @ analytic_propagation.identity_states @ unitary.conj().T,
            dtype="<c16",
        ),
        maximum_hermiticity_residual=analytic_propagation.maximum_hermiticity_residual,
        maximum_trace_residual=analytic_propagation.maximum_trace_residual,
        maximum_relative_norm_increase=analytic_propagation.maximum_relative_norm_increase,
        maximum_identity_relative_drift=analytic_propagation.maximum_identity_relative_drift,
    )
    rotated_analytic_fit = fit_comparator_origin(
        trajectory_id="matrix-invariant-capacity-comparison.fixture.complete-grid-rotated",
        view_id="matrix-invariant-capacity-comparison.view.fixture",
        origin_step=816,
        propagation=rotated_analytic_propagation,
        y_operator=rotated_direct,
        multiplier=1,
        config=config,
    )
    if analytic_fit.operator is None or rotated_analytic_fit.operator is None:
        fitted_covariance_error = 1.0
    else:
        fitted_covariance_error = float(
            np.linalg.norm(
                rotated_analytic_fit.operator - adjoint @ analytic_fit.operator @ adjoint.T
            )
            / max(
                float(np.linalg.norm(analytic_fit.operator)),
                float(np.linalg.norm(rotated_analytic_fit.operator)),
                1.0,
            )
        )
        if (
            analytic_fit.record.selected_regularization
            != rotated_analytic_fit.record.selected_regularization
        ):
            fitted_covariance_error = 1.0
    invalid_propagation = _qualification_propagation(
        operator=direct, seed=31193, constant_first_field=True
    )
    invalid_fit = fit_comparator_origin(
        trajectory_id="matrix-invariant-capacity-comparison.fixture.invalid-fold",
        view_id="matrix-invariant-capacity-comparison.view.fixture",
        origin_step=816,
        propagation=invalid_propagation,
        y_operator=direct,
        multiplier=1,
        config=config,
    )
    _, gram_coordinates = gram_operator_basis()
    boundary_gram = np.zeros((15, 15), dtype=np.float64)
    boundary_gram[:4, :4] = np.eye(4)
    boundary_coefficients = np.asarray(
        [boundary_gram[left, right] for left, right in gram_coordinates], dtype=np.float64
    )
    _, _, _, boundary_reasons = _project_and_validate(
        coefficients=boundary_coefficients,
        trace_budget=trace_budget,
        elapsed_values=(0.004, 0.016, 0.064),
        config=config,
    )
    candidate_local_invalidity = bool(
        "rank-three-projection-ambiguous" in boundary_reasons
        and projected_operator is not None
        and not projection_reasons
        and analytic_fit.record.admissible
        and len(analytic_fit.record.candidates) == 12
        and not invalid_fit.record.admissible
        and len(invalid_fit.record.candidates) == 12
        and all(not value.eligible for value in invalid_fit.record.candidates)
    )
    outcomes: dict[str, tuple[float, float]] = {
        "matrix-invariant-capacity-comparison.fixture.operator-map-direct-equality": (
            max(direct_error, 0.0 if analytic_fit.record.admissible else 1.0),
            1e-10,
        ),
        "matrix-invariant-capacity-comparison.fixture.factor-gauge-invariance": (gauge_error, 1e-10),
        "matrix-invariant-capacity-comparison.fixture.unitary-covariance": (
            max(covariance_error, fitted_covariance_error),
            1e-10,
        ),
        "matrix-invariant-capacity-comparison.fixture.identity-kernel": (identity_error, 1e-10),
        "matrix-invariant-capacity-comparison.fixture.induced-norm-bound": (
            max(0.0, float(invariants.induced_norm_ratio) - 1.0),
            1e-10,
        ),
        "matrix-invariant-capacity-comparison.fixture.semigroup-contractivity": (
            max(0.0, float(invariants.semigroup_gain) - 1.0),
            1e-10,
        ),
        "matrix-invariant-capacity-comparison.fixture.candidate-local-invalidity": (
            0.0 if candidate_local_invalidity else 1.0,
            0.0,
        ),
        "matrix-invariant-capacity-comparison.fixture.no-admissible-candidate": (
            0.0
            if _terminal_for_category(
                category=MatrixResponseInvariantCapacityComparisonContrastCategory.COMPARATOR_INADMISSIBLE,
                event_rank=None,
            )
            is MatrixResponseInvariantCapacityComparisonTerminal.COMPARATOR_DESIGN_INSUFFICIENT_UNEVALUABLE
            else 1.0,
            0.0,
        ),
        "matrix-invariant-capacity-comparison.fixture.y-privileged-terminal": (
            0.0
            if _terminal_for_category(category=MatrixResponseInvariantCapacityComparisonContrastCategory.Y_PRIVILEGED, event_rank=1)
            is MatrixResponseInvariantCapacityComparisonTerminal.TRANSIENT_RESPONSE_GEOMETRY_SUPPORTED
            else 1.0,
            0.0,
        ),
        "matrix-invariant-capacity-comparison.fixture.generic-equivalent-terminal": (
            0.0
            if _terminal_for_category(
                category=MatrixResponseInvariantCapacityComparisonContrastCategory.PRACTICALLY_EQUIVALENT,
                event_rank=1,
            )
            is MatrixResponseInvariantCapacityComparisonTerminal.GENERIC_OPERATOR_COHERENCE_NOT_GEOMETRIC_PRIVILEGE
            else 1.0,
            0.0,
        ),
        "matrix-invariant-capacity-comparison.fixture.generic-dominant-terminal": (
            0.0
            if _terminal_for_category(
                category=MatrixResponseInvariantCapacityComparisonContrastCategory.GENERIC_DOMINANT, event_rank=1
            )
            is MatrixResponseInvariantCapacityComparisonTerminal.GENERIC_OPERATOR_OUTPERFORMS_Y_GEOMETRY
            else 1.0,
            0.0,
        ),
    }
    if tuple(sorted(outcomes)) != config.method_fixture_ids:
        raise RuntimeError("Matrix invariant capacity comparison method qualification fixture roster differs")
    cases = tuple(
        MatrixResponseInvariantCapacityComparisonMethodQualificationCase(
            case_id=case_id,
            metric=_decimal(metric),
            bound=_decimal(bound),
            passed=metric <= bound,
        )
        for case_id, (metric, bound) in sorted(outcomes.items())
    )
    return MatrixResponseInvariantCapacityComparisonComparatorMethodPackage(
        package_id="matrix-invariant-capacity-comparison.comparator-method-package",
        config_fingerprint=config.fingerprint(),
        implementation_commit=implementation_commit,
        implementation_tree=implementation_tree,
        source_locator_count=0,
        source_access_count=0,
        cases=cases,
        qualified=all(value.passed for value in cases),
        grants_authority=False,
    )


def conjugation_basis(seed: int = 91731) -> tuple[ComplexArray, RealArray]:
    """Expose the one frozen numerical-audit conjugation without source access."""

    return _unitary_and_adjoint(seed)

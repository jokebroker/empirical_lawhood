"""Scientific estimators and sole gate finalizers for the bounded Matrix transient response assay.

The module owns scientific reductions only.  Native replay, integration, passive
probe propagation and action delivery remain simulator-adapter responsibilities.
All inference is block-level; probe fields, kappas, times and numerical views are
nested measurements and never become independent replicates.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from functools import lru_cache
from hashlib import sha256
from math import log
from typing import Any, ClassVar, Mapping, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.stats import binomtest

from empirical_lawhood.adapters.composition.matrix_response_study.transient_controlled_invariance_design import MatrixResponseTransientControlledInvarianceStudyConfig
from empirical_lawhood.adapters.control.matrix_response_study.scientific_inputs import MatrixResponsePhaseRuleSelectionScientificInputs, phase_rule_scientific_key_sha256, require_phase_rule_selection_scientific_inputs
from empirical_lawhood.adapters.control.matrix_response_study.transient_policy import ACTIVE_ACTION_WORDS, MatrixResponseTransientControlledInvarianceFrozenInterventionRule, MatrixResponseTransientControlledInvariancePhaseFeature, MatrixResponseTransientControlledInvariancePhaseRule, enumerate_phase_rules
from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseModelFamilyMember
from empirical_lawhood.adapters.simulators.six_matrix_response.controlled_branch import SixMatrixResponseTransientControlledInvarianceBranchTrace
from empirical_lawhood.adapters.simulators.six_matrix_response.model import ComplexArray, SixMatrixState
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import SixMatrixResponseTransientControlledInvarianceProbePropagation, SixMatrixResponseTransientControlledInvarianceProbeRoster, derive_shuffled_operators, encode_traceless_probe, heat_predict_probe, normalized_increment_loss, propagate_passive_probes, radius_only_operator, traceless_operator
from empirical_lawhood.adapters.simulators.six_matrix_response.shooting import traceless_hermitian_basis
from empirical_lawhood.adapters.simulators.six_matrix_response.spectral import spectral_receiver
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .shooting_committor import observe_state, rolling_labels
from .bootstrap_scientific_inputs import fixed_matrix_response_bootstrap_seed_sha256
from empirical_lawhood.adapters.simulators.six_matrix_response.operator_shuffle_scientific_inputs import MatrixResponseOperatorShuffleScientificInputs, require_operator_shuffle_scientific_inputs


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise FloatingPointError("Matrix transient response scientific value is nonfinite")
    return Decimal(repr(float(value)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Matrix transient response cannot reduce an empty sequence")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


class MatrixResponseTransientControlledInvarianceQualificationTerminal(StrEnum):
    TRANSIENT_RESPONSE_GEOMETRY_ESTABLISHED = "TRANSIENT_RESPONSE_GEOMETRY_ESTABLISHED"
    LAPLACIAN_EXISTS_BUT_NOT_RESPONSE_PREDICTIVE = "LAPLACIAN_EXISTS_BUT_NOT_RESPONSE_PREDICTIVE"
    RESPONSE_NOT_GEOMETRY_SPECIFIC = "RESPONSE_NOT_GEOMETRY_SPECIFIC"
    MATCHED_CONTROL_EXCEPTIONALITY_NOT_ESTABLISHED = (
        "MATCHED_CONTROL_EXCEPTIONALITY_NOT_ESTABLISHED"
    )
    PASSIVE_PROBE_NUMERICAL_VIEW_INVALID = "PASSIVE_PROBE_NUMERICAL_VIEW_INVALID"
    MATCHED_CONTROL_ROSTER_INSUFFICIENT = "MATCHED_CONTROL_ROSTER_INSUFFICIENT"
    RESPONSE_PREDICTION_QUALIFICATION_UNEVALUABLE = "RESPONSE_PREDICTION_QUALIFICATION_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceProbeCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-probe-cell'

    cell_id: str
    field_index: int
    kappa: Decimal
    origin_step: int
    horizon_steps: int
    geometry_loss: Decimal
    radius_loss: Decimal
    generic_loss: Decimal | None
    shuffled_losses: tuple[Decimal, ...]
    resolved: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if not 0 <= self.field_index < 12 or self.origin_step < 0 or self.horizon_steps < 1:
            raise ValueError("Matrix transient response probe-cell coordinate differs")
        for name in ("kappa", "geometry_loss", "radius_loss"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.generic_loss is not None:
            validate_decimal(self.generic_loss, field_name="generic_loss", minimum=Decimal(0))
        for value in self.shuffled_losses:
            validate_decimal(value, field_name="shuffled_loss", minimum=Decimal(0))
        if self.shuffled_losses and len(self.shuffled_losses) != 32:
            raise ValueError("Matrix transient response shuffled-loss roster differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceGenericFit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-generic-fit'

    fit_id: str
    trajectory_id: str
    origin_step: int
    ridge: Decimal
    operator_sha256: str
    canonical_triplet_sha256: str
    development_loss: Decimal
    converged: bool
    linear_solve_count: int

    def __post_init__(self) -> None:
        for name in ("fit_id", "trajectory_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.operator_sha256, field_name="operator_sha256")
        validate_sha256(
            self.canonical_triplet_sha256,
            field_name="canonical_triplet_sha256",
        )
        validate_decimal(self.ridge, field_name="ridge", minimum=Decimal(0))
        validate_decimal(self.development_loss, field_name="development_loss", minimum=Decimal(0))
        if self.linear_solve_count < 1 or not self.converged:
            raise ValueError("Matrix transient response generic-fit convergence record differs")


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceQualificationTrajectoryScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-qualification-trajectory-score'

    score_id: str
    trajectory_id: str
    event: bool
    cells: tuple[MatrixResponseTransientControlledInvarianceProbeCell, ...]
    generic_fits: tuple[MatrixResponseTransientControlledInvarianceGenericFit, ...]
    geometry_loss: Decimal
    radius_loss: Decimal
    generic_loss: Decimal | None
    kappa_geometry_losses: tuple[Decimal, ...]
    kappa_radius_losses: tuple[Decimal, ...]
    kappa_generic_losses: tuple[Decimal, ...] | None
    resolved_fraction: Decimal
    kappa_resolved_fractions: tuple[Decimal, ...]
    field_radius_wins: int
    field_generic_wins: int | None
    shuffled_pooled_wins: int
    shuffled_kappa_wins: tuple[int, ...]
    phase_skills: tuple[Decimal, ...]
    horizon_skills: tuple[Decimal, ...]
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("score_id", "trajectory_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_ids(self.generic_fits, attribute="fit_id", field_name="generic_fits")
        for name in ("geometry_loss", "radius_loss", "resolved_fraction"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.generic_loss is not None:
            validate_decimal(self.generic_loss, field_name="generic_loss", minimum=Decimal(0))
        for value in (
            *self.kappa_geometry_losses,
            *self.kappa_radius_losses,
            *self.kappa_resolved_fractions,
        ):
            validate_decimal(value, field_name="stratum_value", minimum=Decimal(0))
        if self.kappa_generic_losses is not None:
            for value in self.kappa_generic_losses:
                validate_decimal(value, field_name="kappa_generic_loss", minimum=Decimal(0))
        for value in (*self.phase_skills, *self.horizon_skills):
            validate_decimal(value, field_name="skill")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid != (not self.reason_codes):
            raise ValueError("Matrix transient response Response-prediction qualification score validity differs")


def _triplet_from_coefficients(coefficients: np.ndarray) -> ComplexArray:
    values = np.asarray(coefficients, dtype=np.float64).reshape(3, 15)
    basis = traceless_hermitian_basis(4)
    return np.asarray(np.tensordot(values, basis, axes=([-1], [0])), dtype="<c16")


def _operator_digest(operator: np.ndarray) -> str:
    return sha256(np.ascontiguousarray(operator, dtype="<f8").tobytes()).hexdigest()


def _generic_examples(
    *,
    propagation: SixMatrixResponseTransientControlledInvarianceProbePropagation,
    origin_step: int,
    horizons: tuple[int, ...],
    field_indices: Sequence[int],
    floor: float,
) -> tuple[tuple[int, int, int, np.ndarray, np.ndarray, float], ...]:
    outputs: list[tuple[int, int, int, np.ndarray, np.ndarray, float]] = []
    origin_offset = origin_step - propagation.start_step
    for field_index in field_indices:
        for kappa_index, kappa in enumerate(propagation.kappas):
            initial = propagation.states[field_index, kappa_index, origin_offset]
            initial_coordinates = encode_traceless_probe(initial)
            for horizon in horizons:
                endpoint_offset = origin_offset + horizon
                if endpoint_offset >= propagation.states.shape[2]:
                    continue
                observed = propagation.states[field_index, kappa_index, endpoint_offset]
                change = float(np.linalg.norm(observed - initial))
                if change < floor:
                    continue
                outputs.append(
                    (
                        field_index,
                        kappa_index,
                        horizon,
                        initial_coordinates,
                        encode_traceless_probe(observed),
                        max(change, floor),
                    )
                )
    return tuple(outputs)


@lru_cache(maxsize=1)
def _gram_operator_basis() -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    """Map a symmetric coefficient Gram matrix linearly to a commutator operator."""

    basis = traceless_hermitian_basis(4)
    zero = np.zeros((4, 4), dtype="<c16")
    diagonal = []
    for index in range(15):
        triplet = np.stack((basis[index], zero, zero))
        diagonal.append(traceless_operator(np.asarray(triplet, dtype="<c16")))
    coordinates: list[tuple[int, int]] = []
    operators: list[np.ndarray] = []
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


def _project_gram_rank_three(
    coefficients: np.ndarray, *, coefficient_bound: float
) -> tuple[np.ndarray, ComplexArray]:
    _, coordinates = _gram_operator_basis()
    gram = np.zeros((15, 15), dtype=np.float64)
    for value, (left, right) in zip(coefficients, coordinates, strict=True):
        gram[left, right] = value
        gram[right, left] = value
    eigenvalues, eigenvectors = np.linalg.eigh((gram + gram.T) * 0.5)
    positive = np.maximum(eigenvalues, 0.0)
    residual = np.asarray((eigenvectors * positive) @ eigenvectors.T, dtype=np.float64)
    rows = np.zeros((3, 15), dtype=np.float64)
    for row_index in range(3):
        pivot = int(np.argmax(np.diag(residual)))
        pivot_value = max(float(residual[pivot, pivot]), 0.0)
        if pivot_value <= np.finfo(np.float64).eps:
            break
        rows[row_index] = residual[:, pivot] / np.sqrt(pivot_value)
        residual -= np.outer(rows[row_index], rows[row_index])
        residual = (residual + residual.T) * 0.5
    if not np.isfinite(rows).all() or float(np.max(np.abs(rows))) > coefficient_bound:
        raise RuntimeError("Matrix transient response generic comparator coefficient bound failed")
    triplet = _triplet_from_coefficients(rows.reshape(45))
    return traceless_operator(triplet), triplet


def _fit_generic_linearized_generator(
    *,
    examples: Sequence[tuple[int, int, int, np.ndarray, np.ndarray, float]],
    timestep: float,
    ridge: float,
    coefficient_bound: float,
) -> tuple[np.ndarray, ComplexArray]:
    operator_basis, _ = _gram_operator_basis()
    design_rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _, kappa_index, horizon, initial, observed, scale in examples:
        elapsed = (0.25, 0.5, 1.0)[kappa_index] * horizon * timestep
        design_rows.append(
            np.stack(tuple(value @ initial for value in operator_basis), axis=1) / scale
        )
        targets.append(((initial - observed) / elapsed) / scale)
    design = np.concatenate(design_rows, axis=0)
    target = np.concatenate(targets)
    if ridge == 0:
        coefficients = np.linalg.lstsq(design, target, rcond=None)[0]
    else:
        coefficients = np.linalg.solve(
            design.T @ design + ridge * np.eye(design.shape[1]),
            design.T @ target,
        )
    return _project_gram_rank_three(coefficients, coefficient_bound=coefficient_bound)


def _generic_prediction_loss(
    *,
    operator: np.ndarray,
    examples: Sequence[tuple[int, int, int, np.ndarray, np.ndarray, float]],
    timestep: float,
) -> float:
    coordinates = tuple(sorted({(value[1], value[2]) for value in examples}))
    kernels = {
        coordinate: expm(-((0.25, 0.5, 1.0)[coordinate[0]]) * coordinate[1] * timestep * operator)
        for coordinate in coordinates
    }
    return _mean(
        tuple(
            float(np.sum(np.square(kernels[(kappa_index, horizon)] @ initial - observed)))
            / scale**2
            for _, kappa_index, horizon, initial, observed, scale in examples
        )
    )


def fit_generic_operator(
    *,
    trajectory_id: str,
    origin_step: int,
    propagation: SixMatrixResponseTransientControlledInvarianceProbePropagation,
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
) -> tuple[np.ndarray, MatrixResponseTransientControlledInvarianceGenericFit]:
    """Fit M_C using grouped development-field CV, then refit all development fields."""

    validate_stable_id(trajectory_id, field_name="trajectory_id")
    floor = float(config.probe_numeric_floor)
    horizons = config.forecast_horizon_steps
    all_examples = _generic_examples(
        propagation=propagation,
        origin_step=origin_step,
        horizons=horizons,
        field_indices=range(6),
        floor=floor,
    )
    if not all_examples:
        raise RuntimeError("Matrix transient response generic comparator has no resolved development examples")
    ridge_scores: list[tuple[float, float]] = []
    for ridge_decimal in config.generic_ridge_grid:
        ridge = float(ridge_decimal)
        fold_losses: list[float] = []
        for held_field in range(6):
            training = tuple(value for value in all_examples if value[0] != held_field)
            validation = tuple(value for value in all_examples if value[0] == held_field)
            operator, _ = _fit_generic_linearized_generator(
                examples=training,
                timestep=float(config.primary_timestep),
                ridge=ridge,
                coefficient_bound=float(config.generic_coefficient_bound),
            )
            fold_losses.append(
                _generic_prediction_loss(
                    operator=operator,
                    examples=validation,
                    timestep=float(config.primary_timestep),
                )
            )
        ridge_scores.append((ridge, _mean(fold_losses)))
    selected_ridge = min(ridge_scores, key=lambda value: (value[1], value[0]))[0]
    operator, canonical_triplet = _fit_generic_linearized_generator(
        examples=all_examples,
        timestep=float(config.primary_timestep),
        ridge=selected_ridge,
        coefficient_bound=float(config.generic_coefficient_bound),
    )
    objective = _generic_prediction_loss(
        operator=operator,
        examples=all_examples,
        timestep=float(config.primary_timestep),
    )
    return operator, MatrixResponseTransientControlledInvarianceGenericFit(
        fit_id=f"matrix-transient-response.generic-fit.{trajectory_id}.step-{origin_step}",
        trajectory_id=trajectory_id,
        origin_step=origin_step,
        ridge=_decimal(selected_ridge),
        operator_sha256=_operator_digest(operator),
        canonical_triplet_sha256=sha256(
            np.ascontiguousarray(canonical_triplet, dtype="<c16").tobytes()
        ).hexdigest(),
        development_loss=_decimal(objective),
        converged=True,
        linear_solve_count=len(config.generic_ridge_grid) * 6 + 1,
    )


def _nested_loss(
    cells: Sequence[MatrixResponseTransientControlledInvarianceProbeCell],
    attribute: str,
    *,
    field_indices: Sequence[int] = tuple(range(6, 12)),
) -> float:
    channel_values: list[float] = []
    for field_index in field_indices:
        for kappa in (Decimal("0.25"), Decimal("0.5"), Decimal("1.0")):
            values = [
                float(value)
                for cell in cells
                if cell.field_index == field_index
                and cell.kappa == kappa
                and cell.resolved
                and (value := getattr(cell, attribute)) is not None
            ]
            if values:
                channel_values.append(_mean(values))
    return _mean(channel_values)


def score_response_prediction_qualification_path(
    *,
    trajectory_id: str,
    event: bool,
    y_path: ComplexArray,
    roster: SixMatrixResponseTransientControlledInvarianceProbeRoster,
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
    scientific_inputs: MatrixResponseOperatorShuffleScientificInputs,
    include_generic: bool = True,
) -> tuple[SixMatrixResponseTransientControlledInvarianceProbePropagation, MatrixResponseTransientControlledInvarianceQualificationTrajectoryScore]:
    """Propagate independent probes and reduce one trajectory without pseudo-replication."""

    require_operator_shuffle_scientific_inputs(
        scientific_inputs, current_config_sha256=config.fingerprint(),
        current_trajectory_id=trajectory_id, origin_steps=config.forecast_origin_steps,
    )
    propagation = propagate_passive_probes(
        y_path=y_path,
        start_step=config.probe_start_step,
        timestep=float(config.primary_timestep),
        roster=roster,
    )
    reasons: set[str] = set()
    if propagation.maximum_identity_relative_drift > float(config.probe_identity_drift_max):
        reasons.add("identity-sentinel-drift")
    if propagation.maximum_relative_norm_increase > float(config.probe_norm_increase_max):
        reasons.add("probe-norm-increase")
    if propagation.maximum_hermiticity_residual > 1e-12:
        reasons.add("probe-hermiticity")
    fits: list[MatrixResponseTransientControlledInvarianceGenericFit] = []
    generic_by_origin: dict[int, np.ndarray] = {}
    if include_generic:
        for origin in config.forecast_origin_steps:
            operator, fit = fit_generic_operator(
                trajectory_id=trajectory_id,
                origin_step=origin,
                propagation=propagation,
                config=config,
            )
            generic_by_origin[origin] = operator
            fits.append(fit)
    cells: list[MatrixResponseTransientControlledInvarianceProbeCell] = []
    for origin in config.forecast_origin_steps:
        offset = origin - config.probe_start_step
        geometry_operator = traceless_operator(y_path[offset])
        radius_operator = radius_only_operator(geometry_operator)
        shuffled = derive_shuffled_operators(
            operator=geometry_operator,
            config_fingerprint=config.fingerprint(),
            trajectory_id=trajectory_id,
            origin_step=origin,
            scientific_inputs=scientific_inputs,
        )
        for field_index in range(6, 12):
            for kappa_index, kappa in enumerate((0.25, 0.5, 1.0)):
                initial = propagation.states[field_index, kappa_index, offset]
                for horizon in config.forecast_horizon_steps:
                    endpoint = offset + horizon
                    if endpoint >= propagation.states.shape[2]:
                        continue
                    observed = propagation.states[field_index, kappa_index, endpoint]
                    geometry = heat_predict_probe(
                        operator=geometry_operator,
                        probe=initial,
                        kappa=kappa,
                        horizon_time=horizon * float(config.primary_timestep),
                    )
                    radius = heat_predict_probe(
                        operator=radius_operator,
                        probe=initial,
                        kappa=kappa,
                        horizon_time=horizon * float(config.primary_timestep),
                    )
                    geometry_loss, resolved = normalized_increment_loss(
                        predicted=geometry,
                        observed=observed,
                        initial=initial,
                        floor=float(config.probe_numeric_floor),
                    )
                    radius_loss, _ = normalized_increment_loss(
                        predicted=radius,
                        observed=observed,
                        initial=initial,
                        floor=float(config.probe_numeric_floor),
                    )
                    generic_loss: float | None = None
                    if include_generic:
                        generic = heat_predict_probe(
                            operator=generic_by_origin[origin],
                            probe=initial,
                            kappa=kappa,
                            horizon_time=horizon * float(config.primary_timestep),
                        )
                        generic_loss, _ = normalized_increment_loss(
                            predicted=generic,
                            observed=observed,
                            initial=initial,
                            floor=float(config.probe_numeric_floor),
                        )
                    shuffle_losses = tuple(
                        normalized_increment_loss(
                            predicted=heat_predict_probe(
                                operator=value,
                                probe=initial,
                                kappa=kappa,
                                horizon_time=horizon * float(config.primary_timestep),
                            ),
                            observed=observed,
                            initial=initial,
                            floor=float(config.probe_numeric_floor),
                        )[0]
                        for value in shuffled
                    )
                    cells.append(
                        MatrixResponseTransientControlledInvarianceProbeCell(
                            cell_id=(
                                f"matrix-transient-response.cell.{trajectory_id}.f{field_index:02d}."
                                f"k{kappa_index}.o{origin}.h{horizon}"
                            ),
                            field_index=field_index,
                            kappa=_decimal(kappa),
                            origin_step=origin,
                            horizon_steps=horizon,
                            geometry_loss=_decimal(geometry_loss),
                            radius_loss=_decimal(radius_loss),
                            generic_loss=(None if generic_loss is None else _decimal(generic_loss)),
                            shuffled_losses=tuple(_decimal(value) for value in shuffle_losses),
                            resolved=resolved,
                        )
                    )
    ordered = tuple(sorted(cells, key=lambda value: value.cell_id))
    resolved_fraction = sum(value.resolved for value in ordered) / len(ordered)
    kappa_resolved = tuple(
        sum(value.resolved for value in ordered if value.kappa == _decimal(kappa))
        / sum(value.kappa == _decimal(kappa) for value in ordered)
        for kappa in (0.25, 0.5, 1.0)
    )
    if resolved_fraction < float(config.probe_resolved_fraction_min) or min(kappa_resolved) < float(
        config.probe_kappa_resolved_fraction_min
    ):
        reasons.add("insufficient-resolved-probe-cells")
    geometry_loss = _nested_loss(ordered, "geometry_loss")
    radius_loss = _nested_loss(ordered, "radius_loss")
    generic_loss = _nested_loss(ordered, "generic_loss") if include_generic else None
    kappa_geometry = tuple(
        _nested_loss(
            tuple(value for value in ordered if value.kappa == _decimal(kappa)), "geometry_loss"
        )
        for kappa in (0.25, 0.5, 1.0)
    )
    kappa_radius = tuple(
        _nested_loss(
            tuple(value for value in ordered if value.kappa == _decimal(kappa)), "radius_loss"
        )
        for kappa in (0.25, 0.5, 1.0)
    )
    kappa_generic = (
        tuple(
            _nested_loss(
                tuple(value for value in ordered if value.kappa == _decimal(kappa)),
                "generic_loss",
            )
            for kappa in (0.25, 0.5, 1.0)
        )
        if include_generic
        else None
    )
    field_radius_wins = sum(
        _nested_loss(ordered, "geometry_loss", field_indices=(field,))
        < _nested_loss(ordered, "radius_loss", field_indices=(field,))
        for field in range(6, 12)
    )
    field_generic_wins = (
        sum(
            _nested_loss(ordered, "geometry_loss", field_indices=(field,))
            < _nested_loss(ordered, "generic_loss", field_indices=(field,))
            for field in range(6, 12)
        )
        if include_generic
        else None
    )
    shuffle_pooled_losses = tuple(
        _mean([float(cell.shuffled_losses[index]) for cell in ordered if cell.resolved])
        for index in range(32)
    )
    shuffled_pooled_wins = sum(geometry_loss < value for value in shuffle_pooled_losses)
    shuffled_kappa_wins = tuple(
        sum(
            kappa_geometry[kappa_index]
            < _mean(
                [
                    float(cell.shuffled_losses[shuffle_index])
                    for cell in ordered
                    if cell.resolved and cell.kappa == _decimal(kappa)
                ]
            )
            for shuffle_index in range(32)
        )
        for kappa_index, kappa in enumerate((0.25, 0.5, 1.0))
    )
    phase_bands = ((816, 848), (880, 896), (928, 960))

    def stratum_skill(stratum_cells: tuple[MatrixResponseTransientControlledInvarianceProbeCell, ...]) -> float:
        geometry = _nested_loss(stratum_cells, "geometry_loss")
        radius = _nested_loss(stratum_cells, "radius_loss")
        comparator = (
            min(radius, _nested_loss(stratum_cells, "generic_loss")) if include_generic else radius
        )
        return log(comparator / geometry)

    phase_skills = tuple(
        _decimal(
            stratum_skill(tuple(value for value in ordered if low <= value.origin_step <= high))
        )
        for low, high in phase_bands
    )
    horizon_skills = tuple(
        _decimal(stratum_skill(tuple(value for value in ordered if value.horizon_steps == horizon)))
        for horizon in config.forecast_horizon_steps
    )
    return propagation, MatrixResponseTransientControlledInvarianceQualificationTrajectoryScore(
        score_id=f"matrix-transient-response.response-prediction-qualification-score.{trajectory_id}",
        trajectory_id=trajectory_id,
        event=event,
        cells=ordered,
        generic_fits=tuple(sorted(fits, key=lambda value: value.fit_id)),
        geometry_loss=_decimal(geometry_loss),
        radius_loss=_decimal(radius_loss),
        generic_loss=None if generic_loss is None else _decimal(generic_loss),
        kappa_geometry_losses=tuple(_decimal(value) for value in kappa_geometry),
        kappa_radius_losses=tuple(_decimal(value) for value in kappa_radius),
        kappa_generic_losses=(
            None if kappa_generic is None else tuple(_decimal(value) for value in kappa_generic)
        ),
        resolved_fraction=_decimal(resolved_fraction),
        kappa_resolved_fractions=tuple(_decimal(value) for value in kappa_resolved),
        field_radius_wins=field_radius_wins,
        field_generic_wins=field_generic_wins,
        shuffled_pooled_wins=shuffled_pooled_wins,
        shuffled_kappa_wins=shuffled_kappa_wins,
        phase_skills=phase_skills,
        horizon_skills=horizon_skills,
        valid=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceQualificationNumericalAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-qualification-numerical-audit'

    audit_id: str
    audited_trajectory_ids: tuple[str, ...]
    median_probe_state_difference: Decimal
    ordering_agreement: Decimal
    conjugation_error: Decimal
    hermiticity_residual: Decimal
    trace_residual: Decimal
    identity_drift: Decimal
    norm_increase: Decimal
    valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        if tuple(sorted(set(self.audited_trajectory_ids))) != self.audited_trajectory_ids:
            raise ValueError("Matrix transient response numerical-audit trajectory roster differs")
        for name in (
            "median_probe_state_difference",
            "ordering_agreement",
            "conjugation_error",
            "hermiticity_residual",
            "trace_residual",
            "identity_drift",
            "norm_increase",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.valid != (not self.reason_codes):
            raise ValueError("Matrix transient response numerical-audit validity differs")


def build_response_prediction_qualification_numerical_audit(
    *,
    audited_trajectory_ids: tuple[str, ...],
    normalized_state_differences: Sequence[float],
    primary_geometry_better: Sequence[bool],
    half_geometry_better: Sequence[bool],
    conjugation_error: float,
    hermiticity_residual: float,
    trace_residual: float,
    identity_drift: float,
    norm_increase: float,
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
) -> MatrixResponseTransientControlledInvarianceQualificationNumericalAudit:
    if not normalized_state_differences or len(primary_geometry_better) != len(
        half_geometry_better
    ):
        raise ValueError("Matrix transient response numerical-audit operands differ")
    median_difference = float(np.median(normalized_state_differences))
    ordering_agreement = _mean(
        [float(left == right) for left, right in zip(primary_geometry_better, half_geometry_better)]
    )
    reasons: set[str] = set()
    if median_difference > float(config.probe_half_state_difference_max):
        reasons.add("probe-state-half-step-discordance")
    if ordering_agreement < float(config.probe_half_order_agreement_min):
        reasons.add("probe-order-half-step-discordance")
    if conjugation_error > float(config.probe_conjugation_error_max):
        reasons.add("unitary-conjugation-discordance")
    if hermiticity_residual > float(config.probe_hermiticity_residual_max):
        reasons.add("probe-hermiticity-residual")
    if trace_residual > float(config.probe_trace_residual_max):
        reasons.add("probe-trace-residual")
    if identity_drift > float(config.probe_identity_drift_max):
        reasons.add("identity-sentinel-drift")
    if norm_increase > float(config.probe_norm_increase_max):
        reasons.add("probe-norm-increase")
    return MatrixResponseTransientControlledInvarianceQualificationNumericalAudit(
        audit_id="matrix-transient-response.response-prediction-qualification-numerical-audit",
        audited_trajectory_ids=tuple(sorted(audited_trajectory_ids)),
        median_probe_state_difference=_decimal(median_difference),
        ordering_agreement=_decimal(ordering_agreement),
        conjugation_error=_decimal(conjugation_error),
        hermiticity_residual=_decimal(hermiticity_residual),
        trace_residual=_decimal(trace_residual),
        identity_drift=_decimal(identity_drift),
        norm_increase=_decimal(norm_increase),
        valid=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceQualificationTerminalReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-qualification-terminal-report'

    report_id: str
    config_fingerprint: str
    control_roster_ids: tuple[str, ...]
    event_score: MatrixResponseTransientControlledInvarianceQualificationTrajectoryScore | None
    control_scores: tuple[MatrixResponseTransientControlledInvarianceQualificationTrajectoryScore, ...]
    numerical_audit: MatrixResponseTransientControlledInvarianceQualificationNumericalAudit | None
    event_skill: Decimal | None
    event_rank: int | None
    terminal: MatrixResponseTransientControlledInvarianceQualificationTerminal
    gate_passed: bool
    intervention_confirmation_condition: str
    prospective_control_condition: str
    reason_codes: tuple[str, ...]
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        if tuple(sorted(set(self.control_roster_ids))) != self.control_roster_ids:
            raise ValueError("Matrix transient response Response-prediction qualification control roster differs")
        require_sorted_unique_ids(
            self.control_scores, attribute="score_id", field_name="control_scores"
        )
        if self.event_skill is not None:
            validate_decimal(self.event_skill, field_name="event_skill")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_pass = (
            self.terminal is MatrixResponseTransientControlledInvarianceQualificationTerminal.TRANSIENT_RESPONSE_GEOMETRY_ESTABLISHED
        )
        expected_intervention_confirmation = "OPEN" if expected_pass else "CONDITION_FALSE"
        expected_prospective_control = (
            "CONDITION_FALSE_UNTIL_INTERVENTION_CONFIRMATION_PASS" if expected_pass else "CONDITION_FALSE"
        )
        if (
            self.gate_passed != expected_pass
            or self.intervention_confirmation_condition != expected_intervention_confirmation
            or self.prospective_control_condition != expected_prospective_control
            or self.grants_authority
        ):
            raise ValueError("Matrix transient response Response-prediction qualification terminal descendants differ")


def _trajectory_skill(score: MatrixResponseTransientControlledInvarianceQualificationTrajectoryScore) -> float:
    if score.generic_loss is None:
        raise ValueError("Matrix transient response full skill requires generic comparator")
    return log(
        min(float(score.radius_loss), float(score.generic_loss)) / float(score.geometry_loss)
    )


def finalize_response_prediction_qualification(
    *,
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
    control_roster_ids: tuple[str, ...],
    event_score: MatrixResponseTransientControlledInvarianceQualificationTrajectoryScore | None,
    control_scores: tuple[MatrixResponseTransientControlledInvarianceQualificationTrajectoryScore, ...],
    numerical_audit: MatrixResponseTransientControlledInvarianceQualificationNumericalAudit | None,
) -> MatrixResponseTransientControlledInvarianceQualificationTerminalReport:
    """Sole Response-prediction qualification finalizer with numerical, absolute and specificity precedence."""

    reasons: set[str] = set()
    if len(control_roster_ids) != config.matched_control_primary_count:
        terminal = MatrixResponseTransientControlledInvarianceQualificationTerminal.MATCHED_CONTROL_ROSTER_INSUFFICIENT
        reasons.add("matched-control-roster-not-20")
    elif event_score is None:
        terminal = MatrixResponseTransientControlledInvarianceQualificationTerminal.RESPONSE_PREDICTION_QUALIFICATION_UNEVALUABLE
        reasons.add("event-score-missing")
    elif not event_score.valid or numerical_audit is None or not numerical_audit.valid:
        terminal = MatrixResponseTransientControlledInvarianceQualificationTerminal.PASSIVE_PROBE_NUMERICAL_VIEW_INVALID
        reasons.update(event_score.reason_codes)
        reasons.update(() if numerical_audit is None else numerical_audit.reason_codes)
    elif float(event_score.geometry_loss) > float(config.response_prediction_qualification_loss_pooled_max) or max(
        map(float, event_score.kappa_geometry_losses)
    ) > float(config.response_prediction_qualification_loss_kappa_max):
        terminal = MatrixResponseTransientControlledInvarianceQualificationTerminal.LAPLACIAN_EXISTS_BUT_NOT_RESPONSE_PREDICTIVE
        reasons.add("absolute-heldout-prediction-threshold-failed")
    elif event_score.generic_loss is None or event_score.kappa_generic_losses is None:
        terminal = MatrixResponseTransientControlledInvarianceQualificationTerminal.RESPONSE_PREDICTION_QUALIFICATION_UNEVALUABLE
        reasons.add("generic-comparator-missing")
    else:
        geometry_specific = bool(
            float(event_score.geometry_loss) / float(event_score.radius_loss)
            <= float(config.response_prediction_qualification_radius_ratio_max)
            and float(event_score.geometry_loss) / float(event_score.generic_loss)
            <= float(config.response_prediction_qualification_generic_ratio_max)
            and all(
                float(g) < float(r)
                for g, r in zip(
                    event_score.kappa_geometry_losses,
                    event_score.kappa_radius_losses,
                    strict=True,
                )
            )
            and all(
                float(g) < float(c)
                for g, c in zip(
                    event_score.kappa_geometry_losses,
                    event_score.kappa_generic_losses,
                    strict=True,
                )
            )
            and event_score.field_radius_wins >= config.response_prediction_qualification_field_win_min
            and (event_score.field_generic_wins or 0) >= config.response_prediction_qualification_field_win_min
            and event_score.shuffled_pooled_wins >= config.response_prediction_qualification_shuffle_pooled_min
            and min(event_score.shuffled_kappa_wins) >= config.response_prediction_qualification_shuffle_kappa_min
            and min(map(float, event_score.phase_skills)) > 0
            and min(map(float, event_score.horizon_skills)) > 0
        )
        if not geometry_specific:
            terminal = MatrixResponseTransientControlledInvarianceQualificationTerminal.RESPONSE_NOT_GEOMETRY_SPECIFIC
            reasons.add("geometry-specificity-conjunction-failed")
        elif len(control_scores) != config.matched_control_primary_count or any(
            not value.valid or value.generic_loss is None for value in control_scores
        ):
            terminal = MatrixResponseTransientControlledInvarianceQualificationTerminal.RESPONSE_PREDICTION_QUALIFICATION_UNEVALUABLE
            reasons.add("matched-control-score-incomplete")
        else:
            event_skill_float = _trajectory_skill(event_score)
            control_skills = tuple(_trajectory_skill(value) for value in control_scores)
            if event_skill_float < log(float(config.response_prediction_qualification_skill_ratio_min)) or not all(
                event_skill_float > value for value in control_skills
            ):
                terminal = MatrixResponseTransientControlledInvarianceQualificationTerminal.MATCHED_CONTROL_EXCEPTIONALITY_NOT_ESTABLISHED
                reasons.add("event-matched-control-rank-failed")
            else:
                terminal = MatrixResponseTransientControlledInvarianceQualificationTerminal.TRANSIENT_RESPONSE_GEOMETRY_ESTABLISHED
    event_skill = (
        None
        if event_score is None or event_score.generic_loss is None
        else _decimal(_trajectory_skill(event_score))
    )
    event_rank = None
    if event_skill is not None and all(value.generic_loss is not None for value in control_scores):
        event_rank = 1 + sum(
            _trajectory_skill(value) >= float(event_skill) for value in control_scores
        )
    passed = terminal is MatrixResponseTransientControlledInvarianceQualificationTerminal.TRANSIENT_RESPONSE_GEOMETRY_ESTABLISHED
    return MatrixResponseTransientControlledInvarianceQualificationTerminalReport(
        report_id="matrix-transient-response.response-prediction-qualification-terminal",
        config_fingerprint=config.fingerprint(),
        control_roster_ids=tuple(sorted(control_roster_ids)),
        event_score=event_score,
        control_scores=tuple(sorted(control_scores, key=lambda value: value.score_id)),
        numerical_audit=numerical_audit,
        event_skill=event_skill,
        event_rank=event_rank,
        terminal=terminal,
        gate_passed=passed,
        intervention_confirmation_condition="OPEN" if passed else "CONDITION_FALSE",
        prospective_control_condition=("CONDITION_FALSE_UNTIL_INTERVENTION_CONFIRMATION_PASS" if passed else "CONDITION_FALSE"),
        reason_codes=tuple(sorted(reasons)),
        grants_authority=False,
    )


def _first_nonkernel_gap(state: SixMatrixState, member: SixMatrixResponseModelFamilyMember) -> float:
    observation = observe_state(
        observation_id=f"matrix-transient-response.phase-observation.{state.step_index}",
        local_step=state.step_index,
        local_time=state.step_index * 0.001,
        state=state,
        member=member,
        config=_ThresholdProxy,
    )
    spectrum = next(
        value
        for value in spectral_receiver(
            receiver_prefix=f"spectrum.matrix-transient-response.phase-gap.{state.step_index}",
            q=state.q,
            positions=state.positions,
        )
        if value.sector == "Y"
    )
    values = np.asarray(tuple(map(float, spectrum.eigenvalues)), dtype=np.float64)
    denominator = float(np.max(values))
    if not observation.factor_y.valid or denominator <= np.finfo(np.float64).tiny:
        return float("nan")
    return float(values[state.q**2] / denominator)


class _ThresholdProxy:
    q = 2
    phi_min = Decimal("0.35")
    phi_max = Decimal("0.95")
    closure_ratio_max = Decimal("0.30")
    kernel_band_ratio_max = Decimal("0.25")
    persistence_pass_count = 12
    rolling_window_samples = 16


def extract_phase_features(
    *,
    states: Sequence[SixMatrixState],
    member: SixMatrixResponseModelFamilyMember,
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
) -> tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...]:
    """Build the six past-only causal features at the seven declared checkpoints."""

    by_step = {value.step_index: value for value in states}
    required = tuple(
        sorted(set(step - lag for step in config.intervention_confirmation_phase_steps for lag in (0, 16, 32)))
    )
    if any(step not in by_step for step in required):
        raise ValueError("Matrix transient response phase-feature history is incomplete")
    radius: dict[int, float] = {}
    closure: dict[int, float] = {}
    gap: dict[int, float] = {}
    observations: dict[int, Any] = {}
    for step in required:
        state = by_step[step]
        observation = observe_state(
            observation_id=f"matrix-transient-response.phase-observation.{step}",
            local_step=step,
            local_time=step * float(config.primary_timestep),
            state=state,
            member=member,
            config=config,
        )
        observations[step] = observation
        radius[step] = float(observation.native_receiver.radius_y)
        closure[step] = float(observation.factor_y.closure_ratio)
        gap[step] = _first_nonkernel_gap(state, member)
    outputs: list[MatrixResponseTransientControlledInvariancePhaseFeature] = []
    derivative_time = 32 * float(config.primary_timestep)
    for step in config.intervention_confirmation_phase_steps:
        state = by_step[step]
        y = state.positions[1]
        py = state.momenta[1]
        radius_velocity = float(2.0 * np.vdot(y, py).real / (state.n * float(member.mass_y)))
        observation = observations[step]
        cross_denominator = (
            max(
                float(observation.native_receiver.radius_x)
                * float(observation.native_receiver.radius_y),
                np.finfo(np.float64).tiny,
            )
            ** 0.5
        )
        cross_ratio = float(observation.native_receiver.cross_commutator_norm) / cross_denominator
        values = (
            radius[step],
            radius_velocity,
            closure[step],
            (closure[step] - closure[step - 32]) / derivative_time,
            gap[step],
            (gap[step] - gap[step - 32]) / derivative_time,
            float(observation.native_receiver.radius_x),
            float(observation.factor_x.closure_ratio),
            float(observation.factor_x.kernel_band_ratio),
            cross_ratio,
        )
        valid = bool(
            observation.factor_x.valid
            and observation.factor_y.valid
            and not observation.factor_x.instantaneous_geometric
            and np.isfinite(values).all()
        )
        outputs.append(
            MatrixResponseTransientControlledInvariancePhaseFeature(
                feature_id=f"matrix-transient-response.phase-feature.step-{step}",
                parent_step=step,
                radius_y=_decimal(values[0]),
                radius_y_velocity=_decimal(values[1]),
                closure_y=_decimal(values[2]),
                closure_y_velocity=_decimal(values[3]),
                gap_y=_decimal(values[4]),
                gap_y_velocity=_decimal(values[5]),
                radius_x=_decimal(values[6]),
                closure_x=_decimal(values[7]),
                kernel_x=_decimal(values[8]),
                cross_commutator_ratio=_decimal(values[9]),
                valid=valid,
            )
        )
    return tuple(outputs)


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceBranchOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-branch-outcome'

    outcome_id: str
    branch_id: str
    block_index: int
    arm_word: str
    seed_stratum: int
    assessment_start_step: int
    assessment_end_step: int
    complete_01_sustained: bool
    complete_01_total_residence: Decimal
    complete_01_longest_residence: Decimal
    response_invariance_success: bool
    response_total_residence: Decimal
    response_longest_residence: Decimal
    response_resolved_fraction: Decimal
    passive_response_pass_fraction: Decimal
    complete_success_probe_valid: bool
    amplitude_only_sample_count: int
    false_x_admission: bool
    technically_valid: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("outcome_id", "branch_id", "arm_word"):
            validate_stable_id(getattr(self, name), field_name=name)
        if (
            self.block_index < 0
            or self.seed_stratum not in {0, 1}
            or self.assessment_start_step >= self.assessment_end_step
            or self.amplitude_only_sample_count < 0
        ):
            raise ValueError("Matrix transient response branch-outcome coordinate differs")
        for name in (
            "complete_01_total_residence",
            "complete_01_longest_residence",
            "response_total_residence",
            "response_longest_residence",
            "response_resolved_fraction",
            "passive_response_pass_fraction",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.technically_valid != (not self.reason_codes):
            raise ValueError("Matrix transient response branch-outcome validity differs")


def _residence_from_grid(flags: Sequence[bool], *, cadence_time: float) -> tuple[float, float]:
    """Return conservative endpoint-spanned total and longest residence."""

    total = 0.0
    longest = 0.0
    current = 0.0
    for left, right in zip(flags, flags[1:]):
        if left and right:
            total += cadence_time
            current += cadence_time
            longest = max(longest, current)
        else:
            current = 0.0
    return total, longest


def evaluate_branch_outcome(
    *,
    trace: SixMatrixResponseTransientControlledInvarianceBranchTrace,
    arm_word: str,
    member: SixMatrixResponseModelFamilyMember,
    roster: SixMatrixResponseTransientControlledInvarianceProbeRoster,
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
    assessment_start_step: int,
    assessment_end_step: int,
    sustained_duration: float,
    response_duration: float | None = None,
) -> MatrixResponseTransientControlledInvarianceBranchOutcome:
    """Score rolling complete-01 and prospective G_resp from native branch outputs."""

    multiplier = trace.parent_step_multiplier
    start_parent_step = trace.start_state.step_index // multiplier
    observations = tuple(
        observe_state(
            observation_id=(
                f"matrix-transient-response.observation.{trace.branch_id}.{state.step_index // multiplier}"
            ),
            local_step=state.step_index // multiplier,
            local_time=(state.step_index // multiplier) * float(config.primary_timestep),
            state=state,
            member=member,
            config=config,
        )
        for state in trace.receiver_states
    )
    labels = rolling_labels(observations, config=config)
    labels_by_step = {value.endpoint_step: value for value in labels}
    assessment_steps = tuple(
        range(assessment_start_step, assessment_end_step + 1, config.receiver_cadence_steps)
    )
    if any(step not in labels_by_step for step in assessment_steps):
        raise ValueError("Matrix transient response assessment lacks a full rolling history")
    complete_flags = tuple(
        labels_by_step[step].label == "01"
        and labels_by_step[step].strict_window_y_geometric
        and not labels_by_step[step].x_geometric
        for step in assessment_steps
    )
    complete_total, complete_longest = _residence_from_grid(
        complete_flags,
        cadence_time=config.receiver_cadence_steps * float(config.primary_timestep),
    )
    propagation = propagate_passive_probes(
        y_path=trace.y_path,
        start_step=start_parent_step,
        timestep=float(config.primary_timestep) / multiplier,
        roster=roster,
        field_indices=tuple(range(6, 12)),
    )
    observations_by_step = {value.local_step: value for value in observations}
    response_flags: list[bool] = []
    response_resolved: list[bool] = []
    passive_response_flags: list[bool] = []
    amplitude_only = 0
    for step in assessment_steps:
        observation = observations_by_step[step]
        origin = step - 32
        origin_offset = multiplier * (origin - start_parent_step)
        endpoint_offset = multiplier * (step - start_parent_step)
        operator = traceless_operator(trace.y_path[origin_offset])
        radius_operator = radius_only_operator(operator)
        geometry_losses: list[float] = []
        radius_losses: list[float] = []
        resolved_cells = 0
        for local_field in range(6):
            for kappa_index, kappa in enumerate((0.25, 0.5, 1.0)):
                initial = propagation.states[local_field, kappa_index, origin_offset]
                observed_probe = propagation.states[local_field, kappa_index, endpoint_offset]
                geometry = heat_predict_probe(
                    operator=operator,
                    probe=initial,
                    kappa=kappa,
                    horizon_time=0.032,
                )
                radius = heat_predict_probe(
                    operator=radius_operator,
                    probe=initial,
                    kappa=kappa,
                    horizon_time=0.032,
                )
                geometry_loss, resolved = normalized_increment_loss(
                    predicted=geometry,
                    observed=observed_probe,
                    initial=initial,
                    floor=float(config.probe_numeric_floor),
                )
                radius_loss, _ = normalized_increment_loss(
                    predicted=radius,
                    observed=observed_probe,
                    initial=initial,
                    floor=float(config.probe_numeric_floor),
                )
                if resolved:
                    resolved_cells += 1
                    geometry_losses.append(geometry_loss)
                    radius_losses.append(radius_loss)
        probe_resolved = resolved_cells == 18
        probe_pass = bool(
            probe_resolved
            and _mean(geometry_losses) <= float(config.response_prediction_qualification_loss_pooled_max)
            and _mean(geometry_losses) / _mean(radius_losses)
            <= float(config.response_prediction_qualification_radius_ratio_max)
        )
        label = labels_by_step[step]
        amplitude = bool(
            observation.factor_y.radius_closure_pass
            or (float(config.phi_min) <= float(observation.factor_y.phi) <= float(config.phi_max))
        )
        closure_pass = float(observation.factor_y.closure_ratio) <= float(config.closure_ratio_max)
        kernel_pass = bool(
            observation.factor_y.valid
            and float(observation.factor_y.kernel_band_ratio) <= float(config.kernel_band_ratio_max)
        )
        persistence = bool(
            label.y_geometric and label.strict_window_y_geometric and not label.x_geometric
        )
        response = bool(amplitude and closure_pass and kernel_pass and persistence and probe_pass)
        response_flags.append(response)
        response_resolved.append(probe_resolved)
        passive_response_flags.append(probe_pass)
        if amplitude and not (closure_pass and kernel_pass and probe_pass):
            amplitude_only += 1
    response_total, response_longest = _residence_from_grid(
        response_flags,
        cadence_time=config.receiver_cadence_steps * float(config.primary_timestep),
    )
    reasons = set(trace.reason_codes)
    if propagation.maximum_hermiticity_residual > float(config.probe_hermiticity_residual_max):
        reasons.add("probe-hermiticity-residual")
    if propagation.maximum_trace_residual > float(config.probe_trace_residual_max):
        reasons.add("probe-trace-residual")
    if propagation.maximum_identity_relative_drift > float(config.probe_identity_drift_max):
        reasons.add("identity-sentinel-drift")
    if propagation.maximum_relative_norm_increase > float(config.probe_norm_increase_max):
        reasons.add("probe-norm-increase")
    resolved_fraction = _mean(tuple(map(float, response_resolved)))
    passive_pass_fraction = _mean(tuple(map(float, passive_response_flags)))
    complete_success_probe_valid = bool(
        all(
            probe_pass
            for complete, probe_pass in zip(complete_flags, passive_response_flags, strict=True)
            if complete
        )
    )
    target_response_duration = (
        float(config.prospective_control_residence_duration) if response_duration is None else response_duration
    )
    return MatrixResponseTransientControlledInvarianceBranchOutcome(
        outcome_id=f"matrix-transient-response.outcome.{trace.branch_id}",
        branch_id=trace.branch_id,
        block_index=trace.block_index,
        arm_word=arm_word,
        seed_stratum=0 if trace.block_index < 64 else 1,
        assessment_start_step=assessment_start_step,
        assessment_end_step=assessment_end_step,
        complete_01_sustained=complete_longest >= sustained_duration,
        complete_01_total_residence=_decimal(complete_total),
        complete_01_longest_residence=_decimal(complete_longest),
        response_invariance_success=response_longest >= target_response_duration,
        response_total_residence=_decimal(response_total),
        response_longest_residence=_decimal(response_longest),
        response_resolved_fraction=_decimal(resolved_fraction),
        passive_response_pass_fraction=_decimal(passive_pass_fraction),
        complete_success_probe_valid=complete_success_probe_valid,
        amplitude_only_sample_count=amplitude_only,
        false_x_admission=any(labels_by_step[step].x_geometric for step in assessment_steps),
        technically_valid=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


class MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal(StrEnum):
    CAUSAL_CONTROL_AUTHORITY_ESTABLISHED = "CAUSAL_CONTROL_AUTHORITY_ESTABLISHED"
    NO_ACTIONABLE_DEVELOPMENT_CANDIDATE = "NO_ACTIONABLE_DEVELOPMENT_CANDIDATE"
    INBOUND_PHASE_NOT_CAUSALLY_OBSERVABLE = "INBOUND_PHASE_NOT_CAUSALLY_OBSERVABLE"
    HELDOUT_RESIDENCE_EFFECT_NOT_MATERIAL = "HELDOUT_RESIDENCE_EFFECT_NOT_MATERIAL"
    IMMEDIATE_DIAGNOSTIC_ONLY = "IMMEDIATE_DIAGNOSTIC_ONLY"
    ACTION_DELIVERY_OR_EFFORT_INVALID = "ACTION_DELIVERY_OR_EFFORT_INVALID"
    INTERVENTION_CONFIRMATION_NUMERICAL_VIEW_INVALID = "INTERVENTION_CONFIRMATION_NUMERICAL_VIEW_INVALID"
    INTERVENTION_CONFIRMATION_UNEVALUABLE = "INTERVENTION_CONFIRMATION_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceDevelopmentCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-development-cell'

    cell_id: str
    checkpoint_step: int
    action_word: str
    valid_pair_count: int
    risk_gain: Decimal
    rmst_gain: Decimal
    first_half_risk_gain: Decimal
    second_half_risk_gain: Decimal
    first_half_rmst_gain: Decimal
    second_half_rmst_gain: Decimal
    normalized_effect_margin: Decimal
    eligible: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if (
            self.checkpoint_step < 0
            or self.action_word not in ACTIVE_ACTION_WORDS
            or not 0 <= self.valid_pair_count <= 32
        ):
            raise ValueError("Matrix transient response development-cell identity differs")
        for name in (
            "risk_gain",
            "rmst_gain",
            "first_half_risk_gain",
            "second_half_risk_gain",
            "first_half_rmst_gain",
            "second_half_rmst_gain",
            "normalized_effect_margin",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.eligible != (not self.reason_codes):
            raise ValueError("Matrix transient response development-cell eligibility differs")


def reduce_development_cell(
    *,
    checkpoint_step: int,
    action_word: str,
    action: Sequence[MatrixResponseTransientControlledInvarianceBranchOutcome],
    hold: Sequence[MatrixResponseTransientControlledInvarianceBranchOutcome],
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
) -> MatrixResponseTransientControlledInvarianceDevelopmentCell:
    if len(action) != 32 or len(hold) != 32:
        raise ValueError("Matrix transient response development cell requires 32 paired blocks")
    action_by_block = {value.block_index: value for value in action}
    hold_by_block = {value.block_index: value for value in hold}
    if set(action_by_block) != set(range(32)) or set(hold_by_block) != set(range(32)):
        raise ValueError("Matrix transient response development block roster differs")
    valid = tuple(
        index
        for index in range(32)
        if action_by_block[index].technically_valid and hold_by_block[index].technically_valid
    )
    risk_differences = np.asarray(
        [
            int(action_by_block[index].complete_01_sustained)
            - int(hold_by_block[index].complete_01_sustained)
            for index in range(32)
        ],
        dtype=np.float64,
    )
    rmst_differences = np.asarray(
        [
            float(action_by_block[index].complete_01_total_residence)
            - float(hold_by_block[index].complete_01_total_residence)
            for index in range(32)
        ],
        dtype=np.float64,
    )
    risk_gain = float(np.mean(risk_differences))
    rmst_gain = float(np.mean(rmst_differences))
    halves = (slice(0, 16), slice(16, 32))
    half_risk = tuple(float(np.mean(risk_differences[value])) for value in halves)
    half_rmst = tuple(float(np.mean(rmst_differences[value])) for value in halves)
    reasons: set[str] = set()
    if len(valid) != 32:
        reasons.add("development-pair-invalid")
    if risk_gain < float(config.intervention_confirmation_development_risk_gain_min):
        reasons.add("development-risk-margin-failed")
    if rmst_gain < float(config.intervention_confirmation_development_rmst_gain_min):
        reasons.add("development-rmst-margin-failed")
    if min(half_risk) <= 0:
        reasons.add("development-risk-split-direction-failed")
    if min(half_rmst) <= 0:
        reasons.add("development-rmst-split-direction-failed")
    margin = min(
        (rmst_gain - float(config.intervention_confirmation_development_rmst_gain_min))
        / float(config.intervention_confirmation_development_rmst_gain_min),
        (risk_gain - float(config.intervention_confirmation_development_risk_gain_min))
        / float(config.intervention_confirmation_development_risk_gain_min),
    )
    return MatrixResponseTransientControlledInvarianceDevelopmentCell(
        cell_id=f"matrix-transient-response.development-cell.step-{checkpoint_step}.{action_word}",
        checkpoint_step=checkpoint_step,
        action_word=action_word,
        valid_pair_count=len(valid),
        risk_gain=_decimal(risk_gain),
        rmst_gain=_decimal(rmst_gain),
        first_half_risk_gain=_decimal(half_risk[0]),
        second_half_risk_gain=_decimal(half_risk[1]),
        first_half_rmst_gain=_decimal(half_rmst[0]),
        second_half_rmst_gain=_decimal(half_rmst[1]),
        normalized_effect_margin=_decimal(margin),
        eligible=not reasons,
        reason_codes=tuple(sorted(reasons)),
    )


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceDevelopmentCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-development-candidate'

    candidate_id: str
    config_fingerprint: str
    selected_cell: MatrixResponseTransientControlledInvarianceDevelopmentCell
    phase_rule: MatrixResponseTransientControlledInvariancePhaseRule
    admitted_phase_steps: tuple[int, ...]
    minimum_normalized_effect_margin: Decimal
    candidate_sha256: str
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        validate_sha256(self.candidate_sha256, field_name="candidate_sha256")
        validate_decimal(
            self.minimum_normalized_effect_margin,
            field_name="minimum_normalized_effect_margin",
        )
        if (
            not self.selected_cell.eligible
            or self.phase_rule.action_word != self.selected_cell.action_word
            or tuple(sorted(set(self.admitted_phase_steps))) != self.admitted_phase_steps
            or self.selected_cell.checkpoint_step not in self.admitted_phase_steps
            or self.grants_authority
        ):
            raise ValueError("Matrix transient response development candidate differs")


def select_development_candidate(
    *,
    features: tuple[MatrixResponseTransientControlledInvariancePhaseFeature, ...],
    cells: tuple[MatrixResponseTransientControlledInvarianceDevelopmentCell, ...],
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
    scientific_inputs: MatrixResponsePhaseRuleSelectionScientificInputs,
) -> tuple[MatrixResponseTransientControlledInvarianceDevelopmentCandidate | None, MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal | None]:
    """Apply the frozen effect margins and low-capacity past-only selector."""

    if not isinstance(scientific_inputs, MatrixResponsePhaseRuleSelectionScientificInputs):
        raise ValueError("phase selection requires complete typed original numerical comparison inputs")
    comparison_order = require_phase_rule_selection_scientific_inputs(
        scientific_inputs, current_config_sha256=config.fingerprint(), features=features,
        complete_rules=enumerate_phase_rules(
            features=features, action_word=ACTIVE_ACTION_WORDS[0],
            development_step=config.intervention_confirmation_phase_steps[0], development_effect_margin=Decimal(0),
        ),
    )
    by_coordinate = {(value.checkpoint_step, value.action_word): value for value in cells}
    if set(by_coordinate) != set(
        (step, action) for step in config.intervention_confirmation_phase_steps for action in ACTIVE_ACTION_WORDS
    ):
        raise ValueError("Matrix transient response development cell panel is incomplete")
    eligible = tuple(value for value in cells if value.eligible)
    if not eligible:
        return None, MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.NO_ACTIONABLE_DEVELOPMENT_CANDIDATE
    candidates: list[
        tuple[
            float,
            int,
            int,
            int,
            int,
            int,
            MatrixResponseTransientControlledInvarianceDevelopmentCell,
            MatrixResponseTransientControlledInvariancePhaseRule,
            tuple[int, ...],
        ]
    ] = []
    channel_order = {"x": 0, "y": 1}
    sign_order = {"negative": 0, "positive": 1}
    for cell in eligible:
        rules = enumerate_phase_rules(
            features=features,
            action_word=cell.action_word,
            development_step=cell.checkpoint_step,
            development_effect_margin=cell.normalized_effect_margin,
        )
        for rule in rules:
            admitted = tuple(value.parent_step for value in features if rule.accepts(value))
            if cell.checkpoint_step not in admitted:
                continue
            material_steps = tuple(
                value.checkpoint_step
                for value in cells
                if value.action_word == cell.action_word and value.eligible
            )
            if not set(material_steps).issubset(admitted):
                continue
            nonpositive_steps = tuple(
                value.checkpoint_step
                for value in cells
                if value.action_word == cell.action_word and float(value.rmst_gain) <= 0
            )
            if set(nonpositive_steps).intersection(admitted):
                continue
            margins = tuple(
                float(by_coordinate[(step, cell.action_word)].normalized_effect_margin)
                for step in admitted
            )
            if not margins:
                continue
            channel, sign = cell.action_word.split("-", maxsplit=1)
            candidates.append(
                (
                    min(margins),
                    len(rule.predicates),
                    cell.checkpoint_step,
                    channel_order[channel],
                    sign_order[sign],
                    comparison_order[phase_rule_scientific_key_sha256(rule)],
                    cell,
                    rule,
                    admitted,
                )
            )
    if not candidates:
        return None, MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.INBOUND_PHASE_NOT_CAUSALLY_OBSERVABLE
    selected = min(
        candidates,
        key=lambda value: (-value[0], value[1], value[2], value[3], value[4], value[5]),
    )
    _, _, _, _, _, _, cell, rule, admitted = selected
    payload = (
        f"{config.fingerprint()}\0{cell.fingerprint()}\0{rule.fingerprint()}\0"
        + ",".join(map(str, admitted))
    ).encode()
    digest = sha256(payload).hexdigest()
    return (
        MatrixResponseTransientControlledInvarianceDevelopmentCandidate(
            candidate_id=f"matrix-transient-response.development-candidate.{digest[:16]}",
            config_fingerprint=config.fingerprint(),
            selected_cell=cell,
            phase_rule=rule,
            admitted_phase_steps=admitted,
            minimum_normalized_effect_margin=_decimal(selected[0]),
            candidate_sha256=digest,
            grants_authority=False,
        ),
        None,
    )


def stratified_paired_bootstrap_lower(
    *,
    differences: np.ndarray,
    strata: np.ndarray,
    resamples: int,
    scientific_seed_sha256: str,
    alpha: float = 0.05,
) -> float:
    """One-sided stratified percentile lower bound with equal stratum weight."""

    values = np.asarray(differences, dtype=np.float64)
    groups = np.asarray(strata, dtype=np.int64)
    if values.ndim != 1 or values.shape != groups.shape or set(groups) != {0, 1}:
        raise ValueError("Matrix transient response paired-bootstrap operands differ")
    validate_sha256(scientific_seed_sha256, field_name="scientific_seed_sha256")
    digest = bytes.fromhex(scientific_seed_sha256)
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:16], "big")))
    estimates = np.empty(resamples, dtype=np.float64)
    indices = tuple(np.flatnonzero(groups == group) for group in (0, 1))
    batch = 4096
    for start in range(0, resamples, batch):
        stop = min(resamples, start + batch)
        count = stop - start
        group_means = []
        for members in indices:
            selected = rng.integers(0, members.size, size=(count, members.size))
            group_means.append(np.mean(values[members[selected]], axis=1))
        estimates[start:stop] = 0.5 * (group_means[0] + group_means[1])
    return float(np.quantile(estimates, alpha, method="linear"))


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminalReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-intervention-adjudication-terminal-report'

    report_id: str
    config_fingerprint: str
    candidate_sha256: str
    valid_pair_count: int
    risk_gain: Decimal
    risk_lower: Decimal
    rmst_gain: Decimal
    rmst_lower: Decimal
    stratum_risk_gains: tuple[Decimal, ...]
    stratum_rmst_gains: tuple[Decimal, ...]
    action_only_successes: int
    hold_only_successes: int
    mcnemar_one_sided_p: Decimal
    numerical_audit_valid: bool
    terminal: MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal
    gate_passed: bool
    prospective_control_condition: str
    reason_codes: tuple[str, ...]
    grants_execution_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        validate_sha256(self.candidate_sha256, field_name="candidate_sha256")
        for name in (
            "risk_gain",
            "risk_lower",
            "rmst_gain",
            "rmst_lower",
            "mcnemar_one_sided_p",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        for value in (*self.stratum_risk_gains, *self.stratum_rmst_gains):
            validate_decimal(value, field_name="stratum_effect")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = self.terminal is MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.CAUSAL_CONTROL_AUTHORITY_ESTABLISHED
        if (
            self.gate_passed != expected
            or self.prospective_control_condition != ("OPEN" if expected else "CONDITION_FALSE")
            or self.grants_execution_authority
        ):
            raise ValueError("Matrix transient response Intervention confirmation terminal descendants differ")


def finalize_intervention_confirmation(
    *,
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
    candidate: MatrixResponseTransientControlledInvarianceDevelopmentCandidate,
    action: tuple[MatrixResponseTransientControlledInvarianceBranchOutcome, ...],
    hold: tuple[MatrixResponseTransientControlledInvarianceBranchOutcome, ...],
    numerical_audit_valid: bool,
    delivery_effort_valid: bool,
) -> MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminalReport:
    if len(action) != 128 or len(hold) != 128:
        raise ValueError("Matrix transient response Intervention confirmation confirmation requires 128 intent pairs")
    action_by = {value.block_index: value for value in action}
    hold_by = {value.block_index: value for value in hold}
    if set(action_by) != set(range(128)) or set(hold_by) != set(range(128)):
        raise ValueError("Matrix transient response Intervention confirmation confirmation block roster differs")
    valid_pairs = tuple(
        index
        for index in range(128)
        if action_by[index].technically_valid and hold_by[index].technically_valid
    )
    # Worst-case intent-to-treat is the claim-bearing completion.
    action_binary = np.asarray(
        [
            int(action_by[index].complete_01_sustained) if action_by[index].technically_valid else 0
            for index in range(128)
        ],
        dtype=np.float64,
    )
    hold_binary = np.asarray(
        [
            int(hold_by[index].complete_01_sustained) if hold_by[index].technically_valid else 1
            for index in range(128)
        ],
        dtype=np.float64,
    )
    maximum = (config.intervention_confirmation_assessment_end_step - config.intervention_confirmation_assessment_start_step) * float(
        config.primary_timestep
    )
    action_rmst = np.asarray(
        [
            float(action_by[index].complete_01_total_residence)
            if action_by[index].technically_valid
            else 0.0
            for index in range(128)
        ]
    )
    hold_rmst = np.asarray(
        [
            float(hold_by[index].complete_01_total_residence)
            if hold_by[index].technically_valid
            else maximum
            for index in range(128)
        ]
    )
    strata = np.repeat((0, 1), 64)
    risk_differences = action_binary - hold_binary
    rmst_differences = action_rmst - hold_rmst
    risk_gain = float(np.mean(risk_differences))
    rmst_gain = float(np.mean(rmst_differences))
    risk_lower = stratified_paired_bootstrap_lower(
        differences=risk_differences,
        strata=strata,
        resamples=config.bootstrap_resamples,
        scientific_seed_sha256=fixed_matrix_response_bootstrap_seed_sha256("transient-intervention-risk"),
    )
    rmst_lower = stratified_paired_bootstrap_lower(
        differences=rmst_differences,
        strata=strata,
        resamples=config.bootstrap_resamples,
        scientific_seed_sha256=fixed_matrix_response_bootstrap_seed_sha256("transient-intervention-residence"),
    )
    stratum_risk = tuple(
        float(np.mean(risk_differences[index * 64 : (index + 1) * 64])) for index in range(2)
    )
    stratum_rmst = tuple(
        float(np.mean(rmst_differences[index * 64 : (index + 1) * 64])) for index in range(2)
    )
    action_only = int(np.sum((action_binary == 1) & (hold_binary == 0)))
    hold_only = int(np.sum((action_binary == 0) & (hold_binary == 1)))
    discordant = action_only + hold_only
    mcnemar_p = (
        1.0
        if discordant == 0
        else float(binomtest(action_only, discordant, p=0.5, alternative="greater").pvalue)
    )
    reasons: set[str] = set()
    if len(valid_pairs) < config.intervention_confirmation_valid_pair_min:
        reasons.add("valid-pair-minimum-failed")
    if risk_gain < float(config.intervention_confirmation_risk_gain_min) or risk_lower <= 0:
        reasons.add("heldout-risk-effect-failed")
    if rmst_gain < float(config.intervention_confirmation_rmst_gain_min) or rmst_lower <= 0:
        reasons.add("heldout-rmst-effect-failed")
    if min(stratum_risk) <= 0 or min(stratum_rmst) <= 0:
        reasons.add("seed-stratum-direction-failed")
    if any(
        value.complete_01_sustained and not value.complete_success_probe_valid for value in action
    ):
        reasons.add("active-success-passive-probe-invalid")
    if not delivery_effort_valid:
        reasons.add("action-delivery-or-effort-invalid")
    if not numerical_audit_valid:
        reasons.add("numerical-view-invalid")
    if "action-delivery-or-effort-invalid" in reasons:
        terminal = MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.ACTION_DELIVERY_OR_EFFORT_INVALID
    elif "numerical-view-invalid" in reasons:
        terminal = MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.INTERVENTION_CONFIRMATION_NUMERICAL_VIEW_INVALID
    elif "valid-pair-minimum-failed" in reasons:
        terminal = MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.INTERVENTION_CONFIRMATION_UNEVALUABLE
    elif reasons:
        terminal = MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.HELDOUT_RESIDENCE_EFFECT_NOT_MATERIAL
    else:
        terminal = MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.CAUSAL_CONTROL_AUTHORITY_ESTABLISHED
    passed = terminal is MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.CAUSAL_CONTROL_AUTHORITY_ESTABLISHED
    return MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminalReport(
        report_id="matrix-transient-response.intervention-confirmation-terminal",
        config_fingerprint=config.fingerprint(),
        candidate_sha256=candidate.candidate_sha256,
        valid_pair_count=len(valid_pairs),
        risk_gain=_decimal(risk_gain),
        risk_lower=_decimal(risk_lower),
        rmst_gain=_decimal(rmst_gain),
        rmst_lower=_decimal(rmst_lower),
        stratum_risk_gains=tuple(_decimal(value) for value in stratum_risk),
        stratum_rmst_gains=tuple(_decimal(value) for value in stratum_rmst),
        action_only_successes=action_only,
        hold_only_successes=hold_only,
        mcnemar_one_sided_p=_decimal(mcnemar_p),
        numerical_audit_valid=numerical_audit_valid,
        terminal=terminal,
        gate_passed=passed,
        prospective_control_condition="OPEN" if passed else "CONDITION_FALSE",
        reason_codes=tuple(sorted(reasons)),
        grants_execution_authority=False,
    )


class MatrixResponseTransientControlledInvarianceProspectiveControlTerminal(StrEnum):
    CONTROLLED_INVARIANCE_SUPPORTED_EVENT_CONDITIONAL = (
        "CONTROLLED_INVARIANCE_SUPPORTED_EVENT_CONDITIONAL"
    )
    CONTROLLED_INVARIANCE_NOT_SUPPORTED = "CONTROLLED_INVARIANCE_NOT_SUPPORTED"
    HOLD_OR_COMPARATOR_DOMINANT = "HOLD_OR_COMPARATOR_DOMINANT"
    CONTROLLED_INVARIANCE_RADIUS_CONFOUNDED = "CONTROLLED_INVARIANCE_RADIUS_CONFOUNDED"
    CONTROLLED_INVARIANCE_EFFORT_OR_DELIVERY_INVALID = (
        "CONTROLLED_INVARIANCE_EFFORT_OR_DELIVERY_INVALID"
    )
    CONTROLLED_INVARIANCE_NUMERICAL_VIEW_INVALID = "CONTROLLED_INVARIANCE_NUMERICAL_VIEW_INVALID"
    CONTROLLED_INVARIANCE_TRANSPORT_INVALID = "CONTROLLED_INVARIANCE_TRANSPORT_INVALID"
    PROSPECTIVE_CONTROL_UNEVALUABLE = "PROSPECTIVE_CONTROL_UNEVALUABLE"


class MatrixResponseTransientControlledInvarianceJointTerminal(StrEnum):
    PREREQUISITE_NONATTEMPT = "SIX_MATRIX_CONTROLLED_INVARIANCE_PREREQUISITE_NONATTEMPT"
    NO_TRANSIENT_RESPONSE_GEOMETRY = "SIX_MATRIX_CONTROLLED_INVARIANCE_NO_TRANSIENT_RESPONSE_GEOMETRY"
    TRANSIENT_RESPONSE_GEOMETRY_NO_CAUSAL_CONTROL = (
        "SIX_MATRIX_CONTROLLED_INVARIANCE_TRANSIENT_RESPONSE_GEOMETRY_NO_CAUSAL_CONTROL"
    )
    CAUSAL_RESPONSE_NO_CONTROLLED_INVARIANCE = "SIX_MATRIX_CONTROLLED_INVARIANCE_CAUSAL_RESPONSE_NO_CONTROLLED_INVARIANCE"
    CONTROLLED_INVARIANCE_RADIUS_CONFOUNDED = "SIX_MATRIX_CONTROLLED_INVARIANCE_RADIUS_CONFOUNDED"
    CONTROLLED_INVARIANCE_NUMERICALLY_OR_OPERATIONALLY_INVALID = (
        "SIX_MATRIX_CONTROLLED_INVARIANCE_NUMERICALLY_OR_OPERATIONALLY_INVALID"
    )
    CONTROLLED_INVARIANCE_SUPPORTED_EVENT_CONDITIONAL = (
        "SIX_MATRIX_CONTROLLED_INVARIANCE_SUPPORTED_EVENT_CONDITIONAL"
    )
    UNEVALUABLE = "SIX_MATRIX_CONTROLLED_INVARIANCE_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvariancePolicyContrast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-policy-contrast'

    contrast_id: str
    comparator_word: str
    risk_gain: Decimal
    risk_simultaneous_lower: Decimal
    rmst_gain: Decimal
    rmst_simultaneous_lower: Decimal
    worst_case: bool

    def __post_init__(self) -> None:
        for name in ("contrast_id", "comparator_word"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "risk_gain",
            "risk_simultaneous_lower",
            "rmst_gain",
            "rmst_simultaneous_lower",
        ):
            validate_decimal(getattr(self, name), field_name=name)


def stratified_max_t_lower(
    *,
    differences: np.ndarray,
    strata: np.ndarray,
    resamples: int,
    scientific_seed_sha256: str,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic studentized paired-block max-T simultaneous lower bounds."""

    values = np.asarray(differences, dtype=np.float64)
    groups = np.asarray(strata, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != groups.size or set(groups) != {0, 1}:
        raise ValueError("Matrix transient response max-T operands differ")
    validate_sha256(scientific_seed_sha256, field_name="scientific_seed_sha256")
    digest = bytes.fromhex(scientific_seed_sha256)
    rng = np.random.Generator(np.random.PCG64DXSM(int.from_bytes(digest[:16], "big")))
    observed = np.mean(values, axis=0)
    observed_se = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])
    observed_scale = np.maximum(observed_se, np.finfo(np.float64).eps)
    members = tuple(np.flatnonzero(groups == group) for group in (0, 1))
    maximum_t = np.empty(resamples, dtype=np.float64)
    batch = 2048
    for start in range(0, resamples, batch):
        stop = min(resamples, start + batch)
        count = stop - start
        selected_parts = []
        for indices in members:
            draw = rng.integers(0, indices.size, size=(count, indices.size))
            selected_parts.append(values[indices[draw]])
        bootstrap = np.concatenate(selected_parts, axis=1)
        estimate = 0.5 * (np.mean(selected_parts[0], axis=1) + np.mean(selected_parts[1], axis=1))
        bootstrap_se = np.std(bootstrap, axis=1, ddof=1) / np.sqrt(values.shape[0])
        scale = np.maximum(bootstrap_se, np.finfo(np.float64).eps)
        maximum_t[start:stop] = np.max((estimate - observed[None, :]) / scale, axis=1)
    critical = float(np.quantile(maximum_t, 1.0 - alpha, method="linear"))
    return observed, observed - critical * observed_scale


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceProspectiveControlTerminalReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-prospective-control-terminal-report'

    report_id: str
    config_fingerprint: str
    frozen_rule_fingerprint: str
    valid_six_arm_block_count: int
    contrasts: tuple[MatrixResponseTransientControlledInvariancePolicyContrast, ...]
    hold_stratum_risk_gains: tuple[Decimal, ...]
    hold_stratum_rmst_gains: tuple[Decimal, ...]
    response_geometry_valid: bool
    delivery_effort_valid: bool
    numerical_audit_valid: bool
    protected_transport_valid: bool
    radius_false_admission_valid: bool
    terminal: MatrixResponseTransientControlledInvarianceProspectiveControlTerminal
    gate_passed: bool
    reason_codes: tuple[str, ...]
    grants_further_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        for name in ("config_fingerprint", "frozen_rule_fingerprint"):
            validate_sha256(getattr(self, name), field_name=name)
        require_sorted_unique_ids(self.contrasts, attribute="contrast_id", field_name="contrasts")
        for value in (*self.hold_stratum_risk_gains, *self.hold_stratum_rmst_gains):
            validate_decimal(value, field_name="stratum_effect")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected = (
            self.terminal is MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_SUPPORTED_EVENT_CONDITIONAL
        )
        if self.gate_passed != expected or self.grants_further_authority:
            raise ValueError("Matrix transient response Prospective control terminal disposition differs")


def finalize_prospective_control(
    *,
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
    frozen_rule: MatrixResponseTransientControlledInvarianceFrozenInterventionRule,
    outcomes: Mapping[str, tuple[MatrixResponseTransientControlledInvarianceBranchOutcome, ...]],
    response_geometry_valid: bool,
    delivery_effort_valid: bool,
    numerical_audit_valid: bool,
    protected_transport_valid: bool,
) -> MatrixResponseTransientControlledInvarianceProspectiveControlTerminalReport:
    """Sole Prospective control finalizer using all-intent worst-case simultaneous inference."""

    required = set(config.prospective_control_policy_words)
    if set(outcomes) != required or any(len(values) != 128 for values in outcomes.values()):
        raise ValueError("Matrix transient response Prospective control six-arm intent roster differs")
    by_arm = {
        arm: {value.block_index: value for value in values} for arm, values in outcomes.items()
    }
    if any(set(values) != set(range(128)) for values in by_arm.values()):
        raise ValueError("Matrix transient response Prospective control block identities differ")
    phase_word = "phase-aware"
    comparators = (
        "hold",
        "open-loop-energy-matched",
        "phase-shuffled-feedback",
        "stratified-random",
        "radius-only-feedback",
    )
    maximum = (config.prospective_control_assessment_end_step - config.prospective_control_assessment_start_step) * float(
        config.primary_timestep
    )
    valid_blocks = tuple(
        index
        for index in range(128)
        if all(by_arm[arm][index].technically_valid for arm in required)
    )
    risk_columns: list[np.ndarray] = []
    rmst_columns: list[np.ndarray] = []
    for comparator in comparators:
        risk_columns.append(
            np.asarray(
                [
                    (
                        int(by_arm[phase_word][index].response_invariance_success)
                        if by_arm[phase_word][index].technically_valid
                        else 0
                    )
                    - (
                        int(by_arm[comparator][index].response_invariance_success)
                        if by_arm[comparator][index].technically_valid
                        else 1
                    )
                    for index in range(128)
                ],
                dtype=np.float64,
            )
        )
        rmst_columns.append(
            np.asarray(
                [
                    (
                        float(by_arm[phase_word][index].response_total_residence)
                        if by_arm[phase_word][index].technically_valid
                        else 0.0
                    )
                    - (
                        float(by_arm[comparator][index].response_total_residence)
                        if by_arm[comparator][index].technically_valid
                        else maximum
                    )
                    for index in range(128)
                ],
                dtype=np.float64,
            )
        )
    family = np.stack((*risk_columns, *rmst_columns), axis=1)
    observed, lower = stratified_max_t_lower(
        differences=family,
        strata=np.repeat((0, 1), 64),
        resamples=config.bootstrap_resamples,
        scientific_seed_sha256=fixed_matrix_response_bootstrap_seed_sha256("prospective-control-simultaneous-family"),
        alpha=float(config.familywise_alpha),
    )
    contrasts = tuple(
        MatrixResponseTransientControlledInvariancePolicyContrast(
            contrast_id=f"matrix-transient-response.contrast.phase-aware-vs-{comparator}",
            comparator_word=comparator,
            risk_gain=_decimal(observed[index]),
            risk_simultaneous_lower=_decimal(lower[index]),
            rmst_gain=_decimal(observed[index + 5]),
            rmst_simultaneous_lower=_decimal(lower[index + 5]),
            worst_case=True,
        )
        for index, comparator in enumerate(comparators)
    )
    hold_risk = risk_columns[0]
    hold_rmst = rmst_columns[0]
    stratum_risk = tuple(
        float(np.mean(hold_risk[index * 64 : (index + 1) * 64])) for index in range(2)
    )
    stratum_rmst = tuple(
        float(np.mean(hold_rmst[index * 64 : (index + 1) * 64])) for index in range(2)
    )
    by_comparator = {value.comparator_word: value for value in contrasts}
    radius = by_comparator["radius-only-feedback"]
    radius_valid = bool(
        float(radius.risk_gain) >= float(config.prospective_control_radius_risk_gain_min)
        and float(radius.risk_simultaneous_lower) > 0
    )
    reasons: set[str] = set()
    if not response_geometry_valid:
        reasons.add("response-geometry-invalid")
    for comparator in comparators[:4]:
        contrast = by_comparator[comparator]
        if (
            float(contrast.risk_gain) < float(config.prospective_control_risk_gain_min)
            or float(contrast.risk_simultaneous_lower) <= 0
        ):
            reasons.add(f"risk-effect-failed-{comparator}")
    hold = by_comparator["hold"]
    if float(hold.rmst_gain) < float(config.prospective_control_rmst_gain_min) or float(
        hold.rmst_simultaneous_lower
    ) < float(config.prospective_control_rmst_lower_min):
        reasons.add("hold-rmst-effect-failed")
    if min(stratum_risk) < float(config.prospective_control_stratum_risk_gain_min) or min(stratum_rmst) <= 0:
        reasons.add("seed-stratum-transport-failed")
    if not delivery_effort_valid:
        reasons.add("delivery-effort-invalid")
    if not numerical_audit_valid:
        reasons.add("numerical-view-invalid")
    if len(valid_blocks) < config.prospective_control_valid_block_min or not protected_transport_valid:
        reasons.add("heldout-transport-invalid")
    if not radius_valid:
        reasons.add("radius-only-specificity-failed")
    if "delivery-effort-invalid" in reasons:
        terminal = MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_EFFORT_OR_DELIVERY_INVALID
    elif "numerical-view-invalid" in reasons:
        terminal = MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_NUMERICAL_VIEW_INVALID
    elif "heldout-transport-invalid" in reasons or "seed-stratum-transport-failed" in reasons:
        terminal = MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_TRANSPORT_INVALID
    elif "radius-only-specificity-failed" in reasons:
        terminal = MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_RADIUS_CONFOUNDED
    elif reasons:
        comparator_dominant = any(float(value.risk_gain) <= 0 for value in contrasts)
        terminal = (
            MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.HOLD_OR_COMPARATOR_DOMINANT
            if comparator_dominant
            else MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_NOT_SUPPORTED
        )
    else:
        terminal = MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_SUPPORTED_EVENT_CONDITIONAL
    passed = terminal is MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_SUPPORTED_EVENT_CONDITIONAL
    return MatrixResponseTransientControlledInvarianceProspectiveControlTerminalReport(
        report_id="matrix-transient-response.prospective-control-terminal",
        config_fingerprint=config.fingerprint(),
        frozen_rule_fingerprint=frozen_rule.fingerprint(),
        valid_six_arm_block_count=len(valid_blocks),
        contrasts=tuple(sorted(contrasts, key=lambda value: value.contrast_id)),
        hold_stratum_risk_gains=tuple(_decimal(value) for value in stratum_risk),
        hold_stratum_rmst_gains=tuple(_decimal(value) for value in stratum_rmst),
        response_geometry_valid=response_geometry_valid,
        delivery_effort_valid=delivery_effort_valid,
        numerical_audit_valid=numerical_audit_valid,
        protected_transport_valid=protected_transport_valid,
        radius_false_admission_valid=radius_valid,
        terminal=terminal,
        gate_passed=passed,
        reason_codes=tuple(sorted(reasons)),
        grants_further_authority=False,
    )


def _joint_terminal(
    *,
    response_prediction_qualification_terminal: MatrixResponseTransientControlledInvarianceQualificationTerminal | None,
    intervention_confirmation_terminal: MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal | None,
    prospective_control_terminal: MatrixResponseTransientControlledInvarianceProspectiveControlTerminal | None,
) -> MatrixResponseTransientControlledInvarianceJointTerminal:
    if response_prediction_qualification_terminal is None:
        if intervention_confirmation_terminal is not None or prospective_control_terminal is not None:
            raise ValueError("Matrix transient response descendant terminal exists without response-prediction qualification")
        return MatrixResponseTransientControlledInvarianceJointTerminal.PREREQUISITE_NONATTEMPT
    if response_prediction_qualification_terminal is not MatrixResponseTransientControlledInvarianceQualificationTerminal.TRANSIENT_RESPONSE_GEOMETRY_ESTABLISHED:
        if intervention_confirmation_terminal is not None or prospective_control_terminal is not None:
            raise ValueError("Matrix transient response descendant terminal exists after failed response-prediction qualification")
        if response_prediction_qualification_terminal in {
            MatrixResponseTransientControlledInvarianceQualificationTerminal.RESPONSE_PREDICTION_QUALIFICATION_UNEVALUABLE,
            MatrixResponseTransientControlledInvarianceQualificationTerminal.MATCHED_CONTROL_ROSTER_INSUFFICIENT,
            MatrixResponseTransientControlledInvarianceQualificationTerminal.PASSIVE_PROBE_NUMERICAL_VIEW_INVALID,
        }:
            return MatrixResponseTransientControlledInvarianceJointTerminal.UNEVALUABLE
        return MatrixResponseTransientControlledInvarianceJointTerminal.NO_TRANSIENT_RESPONSE_GEOMETRY
    if intervention_confirmation_terminal is None:
        raise ValueError("Matrix transient response Intervention confirmation terminal is required after Response-prediction qualification pass")
    if intervention_confirmation_terminal is not MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.CAUSAL_CONTROL_AUTHORITY_ESTABLISHED:
        if prospective_control_terminal is not None:
            raise ValueError("Matrix transient response Prospective control terminal exists after failed intervention confirmation")
        if intervention_confirmation_terminal in {
            MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.ACTION_DELIVERY_OR_EFFORT_INVALID,
            MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.INTERVENTION_CONFIRMATION_NUMERICAL_VIEW_INVALID,
            MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.INTERVENTION_CONFIRMATION_UNEVALUABLE,
        }:
            return MatrixResponseTransientControlledInvarianceJointTerminal.UNEVALUABLE
        return MatrixResponseTransientControlledInvarianceJointTerminal.TRANSIENT_RESPONSE_GEOMETRY_NO_CAUSAL_CONTROL
    if prospective_control_terminal is None:
        raise ValueError("Matrix transient response Prospective control terminal is required after Intervention confirmation pass")
    if prospective_control_terminal is MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_SUPPORTED_EVENT_CONDITIONAL:
        return MatrixResponseTransientControlledInvarianceJointTerminal.CONTROLLED_INVARIANCE_SUPPORTED_EVENT_CONDITIONAL
    if prospective_control_terminal is MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_RADIUS_CONFOUNDED:
        return MatrixResponseTransientControlledInvarianceJointTerminal.CONTROLLED_INVARIANCE_RADIUS_CONFOUNDED
    if prospective_control_terminal in {
        MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_EFFORT_OR_DELIVERY_INVALID,
        MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_NUMERICAL_VIEW_INVALID,
        MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.CONTROLLED_INVARIANCE_TRANSPORT_INVALID,
    }:
        return MatrixResponseTransientControlledInvarianceJointTerminal.CONTROLLED_INVARIANCE_NUMERICALLY_OR_OPERATIONALLY_INVALID
    if prospective_control_terminal is MatrixResponseTransientControlledInvarianceProspectiveControlTerminal.PROSPECTIVE_CONTROL_UNEVALUABLE:
        return MatrixResponseTransientControlledInvarianceJointTerminal.UNEVALUABLE
    return MatrixResponseTransientControlledInvarianceJointTerminal.CAUSAL_RESPONSE_NO_CONTROLLED_INVARIANCE


@dataclass(frozen=True, slots=True)
class MatrixResponseTransientControlledInvarianceJointTerminalReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/matrix-response-study/matrix-response-transient-controlled-invariance-joint-terminal-report'

    report_id: str
    config_fingerprint: str
    response_prediction_qualification_terminal: MatrixResponseTransientControlledInvarianceQualificationTerminal | None
    intervention_confirmation_terminal: MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal | None
    prospective_control_terminal: MatrixResponseTransientControlledInvarianceProspectiveControlTerminal | None
    intervention_confirmation_disposition: str
    prospective_control_disposition: str
    terminal: MatrixResponseTransientControlledInvarianceJointTerminal
    maximum_claim: str
    grants_further_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        validate_sha256(self.config_fingerprint, field_name="config_fingerprint")
        validate_stable_id(self.maximum_claim, field_name="maximum_claim")
        expected = _joint_terminal(
            response_prediction_qualification_terminal=self.response_prediction_qualification_terminal,
            intervention_confirmation_terminal=self.intervention_confirmation_terminal,
            prospective_control_terminal=self.prospective_control_terminal,
        )
        expected_b = (
            "ENTERED"
            if self.response_prediction_qualification_terminal is MatrixResponseTransientControlledInvarianceQualificationTerminal.TRANSIENT_RESPONSE_GEOMETRY_ESTABLISHED
            else "CONDITION_FALSE"
        )
        expected_c = (
            "ENTERED"
            if self.intervention_confirmation_terminal is MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.CAUSAL_CONTROL_AUTHORITY_ESTABLISHED
            else "CONDITION_FALSE"
        )
        if (
            self.terminal is not expected
            or self.intervention_confirmation_disposition != expected_b
            or self.prospective_control_disposition != expected_c
            or self.grants_further_authority
        ):
            raise ValueError("Matrix transient response joint terminal precedence differs")


def finalize_joint(
    *,
    config: MatrixResponseTransientControlledInvarianceStudyConfig,
    response_prediction_qualification_terminal: MatrixResponseTransientControlledInvarianceQualificationTerminal | None,
    intervention_confirmation_terminal: MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal | None = None,
    prospective_control_terminal: MatrixResponseTransientControlledInvarianceProspectiveControlTerminal | None = None,
) -> MatrixResponseTransientControlledInvarianceJointTerminalReport:
    """Apply the frozen three-gate terminal precedence exactly once."""

    terminal = _joint_terminal(
        response_prediction_qualification_terminal=response_prediction_qualification_terminal,
        intervention_confirmation_terminal=intervention_confirmation_terminal,
        prospective_control_terminal=prospective_control_terminal,
    )
    return MatrixResponseTransientControlledInvarianceJointTerminalReport(
        report_id="matrix-transient-response.joint-terminal",
        config_fingerprint=config.fingerprint(),
        response_prediction_qualification_terminal=response_prediction_qualification_terminal,
        intervention_confirmation_terminal=intervention_confirmation_terminal,
        prospective_control_terminal=prospective_control_terminal,
        intervention_confirmation_disposition=(
            "ENTERED"
            if response_prediction_qualification_terminal is MatrixResponseTransientControlledInvarianceQualificationTerminal.TRANSIENT_RESPONSE_GEOMETRY_ESTABLISHED
            else "CONDITION_FALSE"
        ),
        prospective_control_disposition=(
            "ENTERED"
            if intervention_confirmation_terminal is MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal.CAUSAL_CONTROL_AUTHORITY_ESTABLISHED
            else "CONDITION_FALSE"
        ),
        terminal=terminal,
        maximum_claim=config.maximum_claim,
        grants_further_authority=False,
    )


__all__ = [
    'MatrixResponseTransientControlledInvarianceBranchOutcome',
    'MatrixResponseTransientControlledInvarianceDevelopmentCandidate',
    'MatrixResponseTransientControlledInvarianceDevelopmentCell',
    'MatrixResponseTransientControlledInvarianceQualificationNumericalAudit',
    'MatrixResponseTransientControlledInvarianceQualificationTerminalReport',
    'MatrixResponseTransientControlledInvarianceQualificationTerminal',
    'MatrixResponseTransientControlledInvarianceQualificationTrajectoryScore',
    'MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminalReport',
    'MatrixResponseTransientControlledInvarianceInterventionAdjudicationTerminal',
    'MatrixResponseTransientControlledInvarianceProspectiveControlTerminalReport',
    'MatrixResponseTransientControlledInvarianceProspectiveControlTerminal',
    'MatrixResponseTransientControlledInvarianceJointTerminalReport',
    'MatrixResponseTransientControlledInvarianceJointTerminal',
    'MatrixResponseTransientControlledInvarianceGenericFit',
    'MatrixResponseTransientControlledInvariancePolicyContrast',
    'MatrixResponseTransientControlledInvarianceProbeCell',
    'build_response_prediction_qualification_numerical_audit',
    'evaluate_branch_outcome',
    'extract_phase_features',
    'finalize_response_prediction_qualification',
    'finalize_intervention_confirmation',
    'finalize_prospective_control',
    'finalize_joint',
    'fit_generic_operator',
    'reduce_development_cell',
    'score_response_prediction_qualification_path',
    'select_development_candidate',
    'stratified_max_t_lower',
    'stratified_paired_bootstrap_lower',
]

"""Support-bounded T-cohort optimization and certificates for split cohort history budget."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm, svd
from scipy.optimize import linprog

from empirical_lawhood.adapters.split_cohort_history_budget.contracts import SplitCohortHistoryBudgetChallengeNomination, SplitCohortHistoryBudgetChallengeGeometry, SplitCohortHistoryBudgetCohort, SplitCohortHistoryBudgetConfig, SplitCohortHistoryBudgetCoordinateLabel, SplitCohortHistoryBudgetDenominatorDescriptor, SplitCohortHistoryBudgetGate, SplitCohortHistoryBudgetHistoryRankForecast, SplitCohortHistoryBudgetOptimizationCertificate, SplitCohortHistoryBudgetOptimizationState, SplitCohortHistoryBudgetPairKind
from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .history import DenseObserverOperator, _forced_endpoint_by_action, _future_defects, _gate_carrier, _predict_actions, assemble_dense_operator, coordinate_labels


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TargetingExecution:
    geometries: tuple[SplitCohortHistoryBudgetChallengeGeometry, ...]
    nominations: tuple[SplitCohortHistoryBudgetChallengeNomination, ...]
    certificates: tuple[SplitCohortHistoryBudgetOptimizationCertificate, ...]
    arrays: dict[str, FloatArray]


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("split cohort history budget targeting value is nonfinite")
    return Decimal(str(float(value)))


def _constraint_family_sha256(
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    nomination: SplitCohortHistoryBudgetChallengeNomination,
) -> str:
    """Bind the canonical constraint operands without any predicted outcome."""

    return sha256(
        canonical_json_bytes(
            (
                "split-cohort-history-budget-support-polytope",
                config.fingerprint(),
                descriptor.fingerprint(),
                nomination.coordinate_id,
                nomination.depth,
                nomination.resolution_epsilon,
                nomination.gate.value,
                nomination.targeting_action_id,
                nomination.center_values,
                nomination.amplitude_volts,
                nomination.mode_values,
            )
        )
    ).hexdigest()


def _geometry(
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    nomination: SplitCohortHistoryBudgetChallengeNomination,
) -> SplitCohortHistoryBudgetChallengeGeometry:
    return SplitCohortHistoryBudgetChallengeGeometry(
        nomination_id=nomination.nomination_id,
        coordinate_id=nomination.coordinate_id,
        unit_id=nomination.unit_id,
        scale_cells=nomination.scale_cells,
        depth=nomination.depth,
        resolution_epsilon=nomination.resolution_epsilon,
        pair_index=nomination.pair_index,
        pair_kind=nomination.pair_kind,
        gate=nomination.gate,
        targeting_action_id=nomination.targeting_action_id,
        center_values=nomination.center_values,
        amplitude_volts=nomination.amplitude_volts,
        mode_values=nomination.mode_values,
        lexical_reference_side="PLUS",
        constraint_family_sha256=_constraint_family_sha256(config, descriptor, nomination),
        outcome_count_at_freeze=0,
    )


def _history_polytope(
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    coordinate: SplitCohortHistoryBudgetCoordinateLabel,
    center: FloatArray,
    *,
    propagations: tuple[FloatArray, ...] | None = None,
    history: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    if (propagations is None) != (history is None):
        raise ValueError("split cohort history budget targeting support cache is incomplete")
    if propagations is None:
        operator = assemble_dense_operator(descriptor)
        lag_seconds = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
        inverse_lag = np.asarray(
            expm(-operator.state_matrix * lag_seconds),
            dtype=np.float64,
        )
        values = [np.eye(descriptor.scale_cells, dtype=np.float64)]
        for _ in range(coordinate.depth):
            values.append(inverse_lag @ values[-1])
        propagations = tuple(values)
        history = np.vstack(tuple(operator.receiver @ value for value in propagations))
    if len(propagations) != coordinate.depth + 1 or history is None:
        raise ValueError("split cohort history budget targeting support cache depth differs")
    tolerance = float(config.optimization_tolerance)
    lower = (
        float(config.hidden_voltage_envelope[0])
        + float(config.hidden_voltage_interior_margin)
        + tolerance
    )
    upper = (
        float(config.hidden_voltage_envelope[1])
        - float(config.hidden_voltage_interior_margin)
        - tolerance
    )
    positive_matrices: list[FloatArray] = []
    positive_bounds: list[FloatArray] = []
    for propagation in propagations:
        historical_center = propagation @ center
        room = np.minimum(historical_center - lower, upper - historical_center)
        if np.min(room) <= 0.0:
            raise ValueError("split cohort history budget targeting center has incomplete history support")
        positive_matrices.append(propagation)
        positive_bounds.append(room)
    collision_half_bound = (float(coordinate.resolution_epsilon) - tolerance) / 2.0
    if collision_half_bound <= 0.0:
        raise ValueError("split cohort history budget optimization tolerance exhausts collision support")
    collision_bound = np.full(history.shape[0], collision_half_bound, dtype=np.float64)
    positive_matrices.append(history)
    positive_bounds.append(collision_bound)
    identity = np.eye(descriptor.scale_cells, dtype=np.float64)
    amplitude = np.full(
        descriptor.scale_cells,
        float(config.hidden_amplitude_max_volts) - tolerance,
        dtype=np.float64,
    )
    positive_matrices.append(identity)
    positive_bounds.append(amplitude)
    positive = np.vstack(positive_matrices)
    positive_bound = np.concatenate(positive_bounds)
    # Keep the exact +/- pairing structural.  The one-sided diameter solve
    # below is valid only for this centrally symmetric support polytope.
    return (
        np.vstack((positive, -positive)),
        np.concatenate((positive_bound, positive_bound)),
    )


def _supported_direction(
    mode: FloatArray,
    constraints: FloatArray,
    bounds: FloatArray,
    *,
    tolerance: float,
) -> FloatArray | None:
    """Scale a deterministic comparator direction into the strict support polytope."""

    values = constraints @ mode
    positive = values > 0.0
    if not np.any(positive):
        return None
    scale = float(np.min(bounds[positive] / values[positive]))
    if not np.isfinite(scale) or scale <= tolerance:
        return None
    direction = np.asarray(mode * scale, dtype=np.float64)
    if float(np.max(constraints @ direction - bounds)) > tolerance:
        return None
    return direction


def _solve_direction(
    objective: FloatArray,
    constraints: FloatArray,
    bounds: FloatArray,
) -> tuple[FloatArray | None, float, float, float, bool]:
    if (
        constraints.ndim != 2
        or bounds.ndim != 1
        or constraints.shape[0] != bounds.size
        or constraints.shape[1] != objective.size
        or constraints.shape[0] % 2
        or not np.all(np.isfinite(constraints))
        or not np.all(np.isfinite(bounds))
        or np.any(bounds < 0.0)
    ):
        raise ValueError("split cohort history budget support polytope is not a finite symmetric pair family")
    half = constraints.shape[0] // 2
    if not np.array_equal(constraints[half:], -constraints[:half]) or not np.array_equal(
        bounds[half:], bounds[:half]
    ):
        raise ValueError("split cohort history budget support polytope lost exact central symmetry")

    # For A=[P;-P], b=[r;r], max |c.x| equals max c.x.  Solve the latter
    # once and double it for the plus/minus witness diameter.
    signed_objective = -objective
    result = linprog(
        signed_objective,
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(None, None)] * objective.size,
        method="highs",
    )
    if not result.success or result.x is None or result.fun is None:
        return None, 0.0, 0.0, 0.0, False
    direction = np.asarray(result.x, dtype=np.float64)
    constraint_values = constraints @ direction
    positive = constraint_values > 0.0
    feasible_scale = min(
        1.0,
        float(np.min(bounds[positive] / constraint_values[positive])) if np.any(positive) else 1.0,
    )
    feasible_direction = direction * max(0.0, feasible_scale)
    primal_residual = max(
        0.0,
        float(np.max(constraints @ feasible_direction - bounds)),
    )
    value = max(0.0, float(objective @ feasible_direction)) * 2.0
    marginals = np.asarray(result.ineqlin.marginals, dtype=np.float64)
    dual_residual = float(np.linalg.norm(signed_objective - constraints.T @ marginals, ord=np.inf))
    variable_l1_bound = float(objective.size) * float(np.max(bounds[-2 * objective.size :]))
    dual_upper = max(
        value,
        -2.0 * float(bounds @ marginals) + 2.0 * variable_l1_bound * dual_residual,
    )
    return (
        feasible_direction,
        value,
        primal_residual,
        max(0.0, dual_upper - value),
        True,
    )


def _classify_optimization_interval(
    *,
    lower: float,
    upper: float,
    margin: float,
    complete: bool,
    tolerance: float,
) -> SplitCohortHistoryBudgetOptimizationState:
    """Classify only intervals separated from the frozen margin by tolerance."""

    if not complete:
        return SplitCohortHistoryBudgetOptimizationState.RESOURCE_LIMITED
    if lower >= margin + tolerance:
        return SplitCohortHistoryBudgetOptimizationState.WITNESS
    if upper <= margin - tolerance:
        return SplitCohortHistoryBudgetOptimizationState.BELOW_MARGIN
    return SplitCohortHistoryBudgetOptimizationState.RESOURCE_LIMITED


def _center_and_objectives(
    *,
    gate: SplitCohortHistoryBudgetGate,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    operator: DenseObserverOperator,
    homogeneous: dict[str, FloatArray],
    forced: dict[str, FloatArray],
) -> tuple[tuple[str, FloatArray, FloatArray], ...]:
    if gate is SplitCohortHistoryBudgetGate.TARGET:
        profile = np.ones(descriptor.scale_cells, dtype=np.float64)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(operator.state_matrix)
        profile = np.abs(np.asarray(eigenvectors[:, int(np.argmax(eigenvalues.real))].real))
        profile /= float(np.max(profile))
    denominator = descriptor.scale_cells * float(descriptor.capacitance_bar_farads)
    rows: list[tuple[str, FloatArray, FloatArray]] = []
    for action_id in sorted(forced):
        carrier = _gate_carrier(
            gate=gate,
            forced=forced[action_id],
            homogeneous_profile=homogeneous[action_id] @ profile,
            operator=operator,
            descriptor=descriptor,
            config=config,
        )
        if carrier is None:
            continue
        center = carrier * profile
        if gate is SplitCohortHistoryBudgetGate.TARGET:
            objective = np.asarray(
                operator.capacitances @ homogeneous[action_id] / denominator,
                dtype=np.float64,
            )
        else:
            midpoint = forced[action_id] + homogeneous[action_id] @ center
            objective = np.asarray(
                homogeneous[action_id][int(np.argmax(midpoint)), :],
                dtype=np.float64,
            )
        rows.append((action_id, center, objective))
    return tuple(rows)


def _certificate(
    *,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    coordinate: SplitCohortHistoryBudgetCoordinateLabel,
    gate: SplitCohortHistoryBudgetGate,
    action_id: str | None,
    state: SplitCohortHistoryBudgetOptimizationState,
    lower: float,
    upper: float,
    margin: Decimal,
    primal_residual: float,
    duality_gap: float,
) -> SplitCohortHistoryBudgetOptimizationCertificate:
    action_slug = "none" if action_id is None else action_id
    return SplitCohortHistoryBudgetOptimizationCertificate(
        certificate_id=(
            f"optimization.{coordinate.coordinate_id}.{gate.value.lower()}.{action_slug}"
        ),
        coordinate_id=coordinate.coordinate_id,
        gate=gate,
        action_id=action_id,
        state=state,
        objective_lower_bound=_decimal(max(0.0, lower)),
        objective_upper_bound=_decimal(max(0.0, upper)),
        required_margin=margin,
        primal_residual=_decimal(max(0.0, primal_residual)),
        duality_gap=_decimal(max(0.0, duality_gap)),
        solver_id="scipy-highs-support-polytope",
        constraint_family_sha256=sha256(
            canonical_json_bytes(
                (
                    "split-cohort-history-budget-certificate-constraint-family",
                    config.fingerprint(),
                    descriptor.fingerprint(),
                    coordinate.coordinate_id,
                    gate.value,
                    action_id,
                )
            )
        ).hexdigest(),
        complete_family_certificate=state is not SplitCohortHistoryBudgetOptimizationState.RESOURCE_LIMITED,
        outcome_count_at_certificate=0,
    )


def _freeze_witness(
    *,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    coordinate: SplitCohortHistoryBudgetCoordinateLabel,
    gate: SplitCohortHistoryBudgetGate,
    kind: SplitCohortHistoryBudgetPairKind,
    action_id: str | None,
    pair_index: int,
    center: FloatArray,
    direction: FloatArray,
    implementation_sha256: str,
    homogeneous: dict[str, FloatArray],
    forced: dict[str, FloatArray],
    operator: DenseObserverOperator,
    history_stack: FloatArray,
) -> SplitCohortHistoryBudgetChallengeNomination:
    amplitude = float(np.max(np.abs(direction)))
    if amplitude <= 0.0:
        raise ValueError("split cohort history budget targeter returned a zero witness")
    mode = direction / amplitude
    coordinate_defect = float(np.max(np.abs(history_stack @ direction))) * 2.0
    predictions = _predict_actions(
        config=config,
        descriptor=descriptor,
        operator=operator,
        homogeneous_by_action=homogeneous,
        forced_by_action=forced,
        center=center,
        amplitude=amplitude,
        mode=mode,
    )
    future_receiver, future_metric = _future_defects(
        config=config,
        descriptor=descriptor,
        operator=operator,
        center=center,
        amplitude=amplitude,
        mode=mode,
    )
    return SplitCohortHistoryBudgetChallengeNomination(
        nomination_id=(
            f"nomination.{descriptor.unit_id}.n{descriptor.scale_cells}"
            f".{coordinate.coordinate_id}.{kind.value.lower().replace('_', '-')}"
            f".{pair_index:02d}"
        ),
        coordinate_id=coordinate.coordinate_id,
        cohort=SplitCohortHistoryBudgetCohort.TARGETED,
        unit_id=descriptor.unit_id,
        scale_cells=descriptor.scale_cells,
        depth=coordinate.depth,
        resolution_epsilon=coordinate.resolution_epsilon,
        pair_index=pair_index,
        pair_kind=kind,
        gate=gate,
        targeting_action_id=action_id,
        carrier_volts=_decimal(float(np.max(center))),
        center_values=tuple(_decimal(value) for value in center),
        amplitude_volts=_decimal(amplitude),
        mode_values=tuple(_decimal(value) for value in mode),
        predicted_coordinate_defect=_decimal(coordinate_defect),
        predicted_maximum_future_receiver_defect=_decimal(future_receiver),
        predicted_maximum_future_metric_defect=_decimal(future_metric),
        action_predictions=predictions,
        observer_implementation_sha256=implementation_sha256,
        outcome_count_at_nomination=0,
    )


def nominate_targeted_challenges(
    *,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    history_forecast: SplitCohortHistoryBudgetHistoryRankForecast,
    implementation_sha256: str,
) -> TargetingExecution:
    """Solve all frozen T support problems and retain witnesses/certificates."""

    if history_forecast.descriptor_sha256 != descriptor.fingerprint():
        raise ValueError("split cohort history budget targeter forecast/descriptor identity differs")
    operator = assemble_dense_operator(descriptor)
    homogeneous, forced = _forced_endpoint_by_action(config, descriptor, operator)
    lag_seconds = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    inverse_lag = np.asarray(expm(-operator.state_matrix * lag_seconds), dtype=np.float64)
    all_propagations = [np.eye(descriptor.scale_cells, dtype=np.float64)]
    for _ in range(config.history_max_depth):
        all_propagations.append(inverse_lag @ all_propagations[-1])
    history_by_depth = {
        depth: np.vstack(
            tuple(operator.receiver @ value for value in all_propagations[: depth + 1])
        )
        for depth in range(config.history_max_depth + 1)
    }
    dynamic_objectives = tuple(
        (
            f"t{time_index:02d}.r{receiver_index:02d}",
            np.asarray(receiver_row @ propagation, dtype=np.float64),
        )
        for time_index, time_t_star in enumerate(config.diagnostic_times_t_star)
        for propagation in (
            np.asarray(
                expm(
                    operator.state_matrix
                    * float(time_t_star)
                    * float(descriptor.time_scale_seconds)
                ),
                dtype=np.float64,
            ),
        )
        for receiver_index, receiver_row in enumerate(operator.receiver)
    )
    objectives_by_gate = {
        gate: {
            action_id: (center, objective)
            for action_id, center, objective in _center_and_objectives(
                gate=gate,
                config=config,
                descriptor=descriptor,
                operator=operator,
                homogeneous=homogeneous,
                forced=forced,
            )
        }
        for gate in (SplitCohortHistoryBudgetGate.TARGET, SplitCohortHistoryBudgetGate.SINK)
    }
    basis_by_depth: dict[int, FloatArray] = {}
    nominations: list[SplitCohortHistoryBudgetChallengeNomination] = []
    certificates: list[SplitCohortHistoryBudgetOptimizationCertificate] = []
    modes: list[FloatArray] = []
    for coordinate in coordinate_labels(config, descriptor.scale_cells):
        propagations = tuple(all_propagations[: coordinate.depth + 1])
        history_stack = history_by_depth[coordinate.depth]
        optimization_tolerance = float(config.optimization_tolerance)
        # The dynamical certificate covers every declared R8 component and
        # diagnostic time.  The single retained witness is the lexical
        # maximum; the certificate, rather than that witness, carries the
        # complete-family nonadversity statement.
        dynamic_center = np.full(descriptor.scale_cells, 0.5, dtype=np.float64)
        dynamic_solutions: list[tuple[float, str, FloatArray, float, float, bool]] = []
        try:
            dynamic_constraints, dynamic_bounds = _history_polytope(
                config,
                descriptor,
                coordinate,
                dynamic_center,
                propagations=propagations,
                history=history_stack,
            )
        except ValueError:
            dynamic_constraints = np.empty((0, descriptor.scale_cells), dtype=np.float64)
            dynamic_bounds = np.empty(0, dtype=np.float64)
        if dynamic_constraints.size:
            for objective_id, objective in dynamic_objectives:
                direction, value, primal, gap, complete = _solve_direction(
                    objective,
                    dynamic_constraints,
                    dynamic_bounds,
                )
                if direction is not None:
                    dynamic_solutions.append(
                        (value, objective_id, direction, primal, gap, complete)
                    )
                elif not complete:
                    dynamic_solutions.append(
                        (
                            0.0,
                            objective_id,
                            np.zeros(descriptor.scale_cells),
                            0.0,
                            0.0,
                            False,
                        )
                    )
        dynamic_solutions.sort(key=lambda value: (-value[0], value[1]))
        dynamic_maximum = dynamic_solutions[0] if dynamic_solutions else None
        if dynamic_maximum is None:
            dynamic_state = SplitCohortHistoryBudgetOptimizationState.INFEASIBLE
            dynamic_lower = dynamic_upper = dynamic_primal = dynamic_gap = 0.0
        else:
            dynamic_lower, _objective_id, dynamic_direction, dynamic_primal, dynamic_gap, _ = (
                dynamic_maximum
            )
            dynamic_upper = dynamic_lower + dynamic_gap
            dynamic_margin = float(config.future_divergence_epsilon)
            dynamic_state = _classify_optimization_interval(
                lower=dynamic_lower,
                upper=dynamic_upper,
                margin=dynamic_margin,
                complete=all(value[5] for value in dynamic_solutions),
                tolerance=optimization_tolerance,
            )
        certificates.append(
            _certificate(
                config=config,
                descriptor=descriptor,
                coordinate=coordinate,
                gate=SplitCohortHistoryBudgetGate.DYNAMICAL,
                action_id=None,
                state=dynamic_state,
                lower=dynamic_lower,
                upper=dynamic_upper,
                margin=config.future_divergence_epsilon,
                primal_residual=dynamic_primal,
                duality_gap=dynamic_gap,
            )
        )
        if dynamic_state is SplitCohortHistoryBudgetOptimizationState.WITNESS and dynamic_maximum is not None:
            dynamic_nomination = _freeze_witness(
                config=config,
                descriptor=descriptor,
                coordinate=coordinate,
                gate=SplitCohortHistoryBudgetGate.DYNAMICAL,
                kind=SplitCohortHistoryBudgetPairKind.DYNAMICAL_BOUNDARY,
                action_id=None,
                pair_index=0,
                center=dynamic_center,
                direction=dynamic_direction,
                implementation_sha256=implementation_sha256,
                homogeneous=homogeneous,
                forced=forced,
                operator=operator,
                history_stack=history_stack,
            )
            nominations.append(dynamic_nomination)
            modes.append(dynamic_direction)

        for gate, kind, margin in (
            (SplitCohortHistoryBudgetGate.TARGET, SplitCohortHistoryBudgetPairKind.TARGET_BOUNDARY, config.target_decision_margin),
            (SplitCohortHistoryBudgetGate.SINK, SplitCohortHistoryBudgetPairKind.SINK_BOUNDARY, config.sink_decision_margin),
        ):
            objective_rows = objectives_by_gate[gate]
            witnesses: list[tuple[float, str, FloatArray, FloatArray]] = []
            for action_id in sorted(forced):
                row = objective_rows.get(action_id)
                if row is None:
                    certificates.append(
                        _certificate(
                            config=config,
                            descriptor=descriptor,
                            coordinate=coordinate,
                            gate=gate,
                            action_id=action_id,
                            state=SplitCohortHistoryBudgetOptimizationState.INFEASIBLE,
                            lower=0.0,
                            upper=0.0,
                            margin=margin,
                            primal_residual=0.0,
                            duality_gap=0.0,
                        )
                    )
                    continue
                center, objective = row
                try:
                    constraints, bounds = _history_polytope(
                        config,
                        descriptor,
                        coordinate,
                        center,
                        propagations=propagations,
                        history=history_stack,
                    )
                except ValueError:
                    certificates.append(
                        _certificate(
                            config=config,
                            descriptor=descriptor,
                            coordinate=coordinate,
                            gate=gate,
                            action_id=action_id,
                            state=SplitCohortHistoryBudgetOptimizationState.INFEASIBLE,
                            lower=0.0,
                            upper=0.0,
                            margin=margin,
                            primal_residual=0.0,
                            duality_gap=0.0,
                        )
                    )
                    continue
                direction, value, primal, gap, complete = _solve_direction(
                    objective,
                    constraints,
                    bounds,
                )
                upper = value + gap
                if gate is SplitCohortHistoryBudgetGate.SINK:
                    # The search objective uses the midpoint-active component.
                    # A nonadversity certificate must nevertheless cover every
                    # component of v_max.  The induced infinity/l1 bound is
                    # conservative over the full amplitude box; if it is loose
                    # the state remains resource/targetability limited.
                    upper = max(
                        upper,
                        2.0
                        * float(config.hidden_amplitude_max_volts)
                        * float(np.max(np.sum(np.abs(homogeneous[action_id]), axis=1))),
                    )
                    gap = max(gap, upper - value)
                margin_value = float(margin)
                state = _classify_optimization_interval(
                    lower=value,
                    upper=upper,
                    margin=margin_value,
                    complete=complete,
                    tolerance=optimization_tolerance,
                )
                certificates.append(
                    _certificate(
                        config=config,
                        descriptor=descriptor,
                        coordinate=coordinate,
                        gate=gate,
                        action_id=action_id,
                        state=state,
                        lower=value,
                        upper=upper,
                        margin=margin,
                        primal_residual=primal,
                        duality_gap=gap,
                    )
                )
                if direction is not None and state is SplitCohortHistoryBudgetOptimizationState.WITNESS:
                    witnesses.append((value, action_id, center, direction))
            witnesses.sort(key=lambda value: (-value[0], value[1]))
            limit = (
                config.target_pairs_per_depth
                if gate is SplitCohortHistoryBudgetGate.TARGET
                else config.sink_pairs_per_depth
            )
            for pair_index, (_value, action_id, center, direction) in enumerate(witnesses[:limit]):
                nomination = _freeze_witness(
                    config=config,
                    descriptor=descriptor,
                    coordinate=coordinate,
                    gate=gate,
                    kind=kind,
                    action_id=action_id,
                    pair_index=pair_index,
                    center=center,
                    direction=direction,
                    implementation_sha256=implementation_sha256,
                    homogeneous=homogeneous,
                    forced=forced,
                    operator=operator,
                    history_stack=history_stack,
                )
                nominations.append(nomination)
                modes.append(direction)
        # Four deterministic random-fibre objectives remain comparator witnesses.
        stack = history_stack
        basis = basis_by_depth.get(coordinate.depth)
        if basis is None:
            _u, singular_values, right_vectors = svd(stack, full_matrices=True)
            relative = singular_values / singular_values[0]
            rank = int(np.sum(relative > float(config.rank_relative_threshold)))
            basis = right_vectors[rank:, :]
            basis_by_depth[coordinate.depth] = basis
        try:
            random_constraints, random_bounds = _history_polytope(
                config,
                descriptor,
                coordinate,
                np.full(descriptor.scale_cells, 0.5, dtype=np.float64),
                propagations=propagations,
                history=history_stack,
            )
        except ValueError:
            random_constraints = np.empty((0, descriptor.scale_cells), dtype=np.float64)
            random_bounds = np.empty(0, dtype=np.float64)
        for pair_index in range(min(config.random_pairs_per_depth, basis.shape[0])):
            if not random_constraints.size:
                break
            raw = np.asarray(basis[pair_index], dtype=np.float64)
            maximum = float(np.max(np.abs(raw)))
            if maximum <= 0.0:
                continue
            mode = raw / maximum
            center = np.full(descriptor.scale_cells, 0.5, dtype=np.float64)
            direction = _supported_direction(
                mode,
                random_constraints,
                random_bounds,
                tolerance=optimization_tolerance,
            )
            if direction is None:
                continue
            amplitude = float(np.max(np.abs(direction)))
            mode = direction / amplitude
            predictions = _predict_actions(
                config=config,
                descriptor=descriptor,
                operator=operator,
                homogeneous_by_action=homogeneous,
                forced_by_action=forced,
                center=center,
                amplitude=amplitude,
                mode=mode,
            )
            future_receiver, future_metric = _future_defects(
                config=config,
                descriptor=descriptor,
                operator=operator,
                center=center,
                amplitude=amplitude,
                mode=mode,
            )
            coordinate_defect = float(np.max(np.abs(stack @ (amplitude * mode)))) * 2.0
            if coordinate_defect > float(coordinate.resolution_epsilon) * (1.0 + 1e-8):
                continue
            nomination = SplitCohortHistoryBudgetChallengeNomination(
                nomination_id=(
                    f"nomination.{descriptor.unit_id}.n{descriptor.scale_cells}"
                    f".{coordinate.coordinate_id}.random-fibre.{pair_index:02d}"
                ),
                coordinate_id=coordinate.coordinate_id,
                cohort=SplitCohortHistoryBudgetCohort.TARGETED,
                unit_id=descriptor.unit_id,
                scale_cells=descriptor.scale_cells,
                depth=coordinate.depth,
                resolution_epsilon=coordinate.resolution_epsilon,
                pair_index=pair_index,
                pair_kind=SplitCohortHistoryBudgetPairKind.RANDOM_FIBRE,
                gate=SplitCohortHistoryBudgetGate.RANDOM,
                targeting_action_id=None,
                carrier_volts=Decimal("0.5"),
                center_values=tuple(_decimal(value) for value in center),
                amplitude_volts=_decimal(amplitude),
                mode_values=tuple(_decimal(value) for value in mode),
                predicted_coordinate_defect=_decimal(coordinate_defect),
                predicted_maximum_future_receiver_defect=_decimal(future_receiver),
                predicted_maximum_future_metric_defect=_decimal(future_metric),
                action_predictions=predictions,
                observer_implementation_sha256=implementation_sha256,
                outcome_count_at_nomination=0,
            )
            nominations.append(nomination)
            modes.append(direction)
    nominations_tuple = tuple(sorted(nominations, key=lambda value: value.nomination_id))
    certificates_tuple = tuple(sorted(certificates, key=lambda value: value.certificate_id))
    arrays = {
        "targeted-directions": (
            np.vstack(modes) if modes else np.empty((0, descriptor.scale_cells), dtype=np.float64)
        )
    }
    closure = sha256(b"".join(value.canonical_bytes() for value in certificates_tuple)).hexdigest()
    arrays["certificate-closure-digest"] = np.frombuffer(
        bytes.fromhex(closure), dtype=np.uint8
    ).astype(np.float64)
    geometries = tuple(
        sorted(
            (_geometry(config, descriptor, value) for value in nominations_tuple),
            key=lambda value: value.nomination_id,
        )
    )
    return TargetingExecution(geometries, nominations_tuple, certificates_tuple, arrays)


__all__ = ["TargetingExecution", "nominate_targeted_challenges"]

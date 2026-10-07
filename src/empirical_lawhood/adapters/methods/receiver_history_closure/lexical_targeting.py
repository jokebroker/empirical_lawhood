'Support-bounded targeted-cohort optimization and certificates for receiver-history.'

from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from hashlib import sha256

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.linalg import expm, svd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from empirical_lawhood.adapters.receiver_history.contracts import (
    ReceiverHistoryChallengeNomination,
    ReceiverHistoryChallengeGeometry,
    ReceiverHistoryCohort,
    ReceiverHistoryConfig,
    ReceiverHistoryCoordinateLabel,
    ReceiverHistoryDenominatorDescriptor,
    ReceiverHistoryEndpoint,
    ReceiverHistoryGate,
    ReceiverHistoryHistoryRankForecast,
    ReceiverHistoryOptimizationCertificate,
    ReceiverHistoryOptimizationState,
    ReceiverHistoryOrientation,
    ReceiverHistoryPairKind,
)
from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .history import (
    DenseObserverOperator,
    _forced_endpoint_by_action,
    _future_defects,
    _gate_carrier,
    _predict_actions,
    assemble_dense_operator,
    coordinate_labels,
)


FloatArray = npt.NDArray[np.float64]
_HIGHS_ACCURACY_OPTIONS = {
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
    "ipm_optimality_tolerance": 1e-12,
}


@dataclass(frozen=True, slots=True)
class TargetingExecution:
    geometries: tuple[ReceiverHistoryChallengeGeometry, ...]
    nominations: tuple[ReceiverHistoryChallengeNomination, ...]
    certificates: tuple[ReceiverHistoryOptimizationCertificate, ...]
    arrays: dict[str, FloatArray]


@dataclass(frozen=True, slots=True)
class _LexicalFamily:
    """One common support polytope plus compact finite-union rows."""

    common_constraints: FloatArray
    common_bounds: FloatArray
    disjunct_constraints: FloatArray
    disjunct_bounds: FloatArray


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("receiver-history targeting value is nonfinite")
    return Decimal(str(float(value)))


def _constraint_family_sha256(
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    nomination: ReceiverHistoryChallengeNomination,
) -> str:
    """Bind the canonical constraint operands without any predicted outcome."""

    return sha256(
        canonical_json_bytes(
            (
                'receiver-history-support-polytope',
                config.fingerprint(),
                descriptor.fingerprint(),
                nomination.coordinate_id,
                nomination.depth,
                nomination.resolution_epsilon,
                nomination.gate.value,
                None if nomination.orientation is None else nomination.orientation.value,
                nomination.targeting_action_id,
                nomination.center_values,
                nomination.amplitude_volts,
                nomination.mode_values,
            )
        )
    ).hexdigest()


def _geometry(
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    nomination: ReceiverHistoryChallengeNomination,
) -> ReceiverHistoryChallengeGeometry:
    return ReceiverHistoryChallengeGeometry(
        nomination_id=nomination.nomination_id,
        coordinate_id=nomination.coordinate_id,
        unit_id=nomination.unit_id,
        scale_cells=nomination.scale_cells,
        depth=nomination.depth,
        resolution_epsilon=nomination.resolution_epsilon,
        pair_index=nomination.pair_index,
        pair_kind=nomination.pair_kind,
        gate=nomination.gate,
        orientation=nomination.orientation,
        targeting_action_id=nomination.targeting_action_id,
        center_values=nomination.center_values,
        amplitude_volts=nomination.amplitude_volts,
        mode_values=nomination.mode_values,
        lexical_reference_side="PLUS",
        constraint_family_sha256=_constraint_family_sha256(config, descriptor, nomination),
        outcome_count_at_freeze=0,
    )


def _history_polytope(
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    coordinate: ReceiverHistoryCoordinateLabel,
    center: FloatArray,
    *,
    propagations: tuple[FloatArray, ...] | None = None,
    history: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    if (propagations is None) != (history is None):
        raise ValueError("receiver-history targeting support cache is incomplete")
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
        raise ValueError("receiver-history targeting support cache depth differs")
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
            raise ValueError("receiver-history targeting center has incomplete history support")
        positive_matrices.append(propagation)
        positive_bounds.append(room)
    collision_half_bound = (float(coordinate.resolution_epsilon) - tolerance) / 2.0
    if collision_half_bound <= 0.0:
        raise ValueError("receiver-history optimization tolerance exhausts collision support")
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
        raise ValueError("receiver-history support polytope is not a finite symmetric pair family")
    half = constraints.shape[0] // 2
    if not np.array_equal(constraints[half:], -constraints[:half]) or not np.array_equal(
        bounds[half:], bounds[:half]
    ):
        raise ValueError("receiver-history support polytope lost exact central symmetry")

    # For A=[P;-P], b=[r;r], max |c.x| equals max c.x.  Solve the latter
    # once and double it for the plus/minus witness diameter.
    signed_objective = -objective
    result = linprog(
        signed_objective,
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(None, None)] * objective.size,
        method="highs",
        options=_HIGHS_ACCURACY_OPTIONS,
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


def _lexical_pair_problems(
    *,
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    coordinate: ReceiverHistoryCoordinateLabel,
    propagations: tuple[FloatArray, ...],
    history: FloatArray,
    homogeneous: FloatArray,
    forced: FloatArray,
    capacitances: FloatArray,
    gate: ReceiverHistoryGate,
    orientation: ReceiverHistoryOrientation,
) -> _LexicalFamily:
    """Independently encode the full two-state lexical carrier as LP families."""

    size = descriptor.scale_cells
    width = 2 * size + 1
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
    rows: list[FloatArray] = []
    bounds: list[float] = []

    def add(
        plus: FloatArray | None,
        minus: FloatArray | None,
        slack: float,
        bound: float,
    ) -> None:
        row = np.zeros(width, dtype=np.float64)
        if plus is not None:
            row[:size] = plus
        if minus is not None:
            row[size : 2 * size] = minus
        row[-1] = slack
        rows.append(row)
        bounds.append(float(bound))

    for propagation in propagations:
        for state_row in propagation:
            vector = np.asarray(state_row, dtype=np.float64)
            add(vector, None, 0.0, upper)
            add(-vector, None, 0.0, -lower)
            add(None, vector, 0.0, upper)
            add(None, -vector, 0.0, -lower)
    collision_bound = float(coordinate.resolution_epsilon) - tolerance
    if collision_bound <= 0.0:
        raise ValueError("receiver-history lexical tolerance exhausts collision support")
    for receiver_row in history:
        vector = np.asarray(receiver_row, dtype=np.float64)
        add(vector, -vector, 0.0, collision_bound)
        add(-vector, vector, 0.0, collision_bound)
    pair_bound = 2.0 * float(config.hidden_amplitude_max_volts) - tolerance
    identity = np.eye(size, dtype=np.float64)
    for state_row in identity:
        add(state_row, -state_row, 0.0, pair_bound)
        add(-state_row, state_row, 0.0, pair_bound)
    add(None, None, 1.0, 2.0)
    add(None, None, -1.0, 0.0)

    denominator = size * float(descriptor.capacitance_bar_farads)
    q_vector = np.asarray(capacitances @ homogeneous / denominator, dtype=np.float64)
    q_forced = float(capacitances @ forced / denominator)
    target_threshold = float(config.target_charge_minimum_q_star)
    target_half_margin = float(config.target_decision_margin) / 2.0
    sink_threshold = float(config.sink_voltage_maximum_v_star) * float(
        descriptor.voltage_reference_volts
    )
    sink_half_margin = (
        float(config.sink_decision_margin) / 2.0 * float(descriptor.voltage_reference_volts)
    )

    if gate is ReceiverHistoryGate.TARGET:
        if orientation is ReceiverHistoryOrientation.UNSAFE_PROMOTION:
            add(-q_vector, None, 0.5, q_forced - target_threshold)
            add(None, q_vector, 0.5, target_threshold - q_forced)
        else:
            add(q_vector, None, 0.5, target_threshold - q_forced)
            add(None, -q_vector, 0.5, q_forced - target_threshold)
        common_sink_bound = sink_threshold - sink_half_margin
        for component, state_row in enumerate(homogeneous):
            vector = np.asarray(state_row, dtype=np.float64)
            bound = common_sink_bound - float(forced[component])
            add(vector, None, 0.0, bound)
            add(None, vector, 0.0, bound)
        return _LexicalFamily(
            common_constraints=np.vstack(rows),
            common_bounds=np.asarray(bounds, dtype=np.float64),
            disjunct_constraints=np.empty((0, width), dtype=np.float64),
            disjunct_bounds=np.empty(0, dtype=np.float64),
        )

    if gate is not ReceiverHistoryGate.SINK:
        raise ValueError("receiver-history lexical carrier requires target or sink gate")
    common_target_bound = q_forced - (target_threshold + target_half_margin)
    add(-q_vector, None, 0.0, common_target_bound)
    add(None, -q_vector, 0.0, common_target_bound)
    safe_plus = orientation is ReceiverHistoryOrientation.UNSAFE_PROMOTION
    for component, state_row in enumerate(homogeneous):
        vector = np.asarray(state_row, dtype=np.float64)
        safe_bound = sink_threshold - float(forced[component])
        if safe_plus:
            add(vector, None, 0.5, safe_bound)
        else:
            add(None, vector, 0.5, safe_bound)

    disjunct_rows: list[FloatArray] = []
    disjunct_bounds: list[float] = []
    for active_component, active_row in enumerate(homogeneous):
        active = np.asarray(active_row, dtype=np.float64)
        failure_bound = float(forced[active_component]) - sink_threshold
        row = np.zeros(width, dtype=np.float64)
        if safe_plus:
            row[size : 2 * size] = -active
        else:
            row[:size] = -active
        row[-1] = 0.5
        disjunct_rows.append(row)
        disjunct_bounds.append(failure_bound)
    return _LexicalFamily(
        common_constraints=np.vstack(rows),
        common_bounds=np.asarray(bounds, dtype=np.float64),
        disjunct_constraints=np.vstack(disjunct_rows),
        disjunct_bounds=np.asarray(disjunct_bounds, dtype=np.float64),
    )


def _solve_lexical_problem(
    constraints: FloatArray,
    bounds: FloatArray,
) -> tuple[tuple[FloatArray, FloatArray] | None, float, float, float, float, bool, bool]:
    """Return a primal witness and conservative primal/dual separation interval."""

    if (
        constraints.ndim != 2
        or bounds.ndim != 1
        or constraints.shape[0] != bounds.size
        or constraints.shape[1] < 3
        or not np.all(np.isfinite(constraints))
        or not np.all(np.isfinite(bounds))
    ):
        raise ValueError("receiver-history lexical LP family is invalid")
    objective = np.zeros(constraints.shape[1], dtype=np.float64)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(None, None)] * objective.size,
        method="highs",
        options=_HIGHS_ACCURACY_OPTIONS,
    )
    if result.status == 2:
        return None, 0.0, 0.0, 0.0, 0.0, True, True
    if not result.success or result.x is None:
        return None, 0.0, 2.0, 0.0, 0.0, False, False
    values = np.asarray(result.x, dtype=np.float64)
    primal = max(0.0, float(np.max(constraints @ values - bounds)))
    lower = max(0.0, float(values[-1]) - primal)
    marginals = np.asarray(result.ineqlin.marginals, dtype=np.float64)
    dual = float(np.linalg.norm(objective - constraints.T @ marginals, ord=np.inf))
    state_size = (objective.size - 1) // 2
    variable_l1_bound = 2.0 * state_size + 2.0
    upper = max(lower, -float(bounds @ marginals) + variable_l1_bound * dual)
    pair = (
        np.asarray(values[:state_size], dtype=np.float64),
        np.asarray(values[state_size : 2 * state_size], dtype=np.float64),
    )
    return pair, lower, upper, primal, dual, True, False


def _solve_lexical_family(
    problems: tuple[tuple[FloatArray, FloatArray], ...] | _LexicalFamily,
    *,
    state_lower_bound: float,
    state_upper_bound: float,
    tolerance: float,
) -> tuple[tuple[FloatArray, FloatArray] | None, float, float, float, float, bool, bool]:
    """Solve one LP or an exact finite union through one bounded MILP.

    Sink failure is a finite disjunction over the active maximum-voltage
    component.  The old component-by-component LP loop is mathematically
    exact but needlessly repeats the same support polytope.  Binary selectors
    encode that identical finite union in one problem; the Big-M values are
    derived from the declared variable box, not guessed constants.
    """

    if isinstance(problems, _LexicalFamily):
        family = problems
    else:
        if not problems:
            raise ValueError("receiver-history lexical family is empty")
        first_constraints, first_bounds = problems[0]
        if len(problems) == 1:
            family = _LexicalFamily(
                common_constraints=first_constraints,
                common_bounds=first_bounds,
                disjunct_constraints=np.empty((0, first_constraints.shape[1]), dtype=np.float64),
                disjunct_bounds=np.empty(0, dtype=np.float64),
            )
        else:
            base_constraints = first_constraints[:-1]
            base_bounds = first_bounds[:-1]
            if any(
                constraints.shape != first_constraints.shape
                or bounds.shape != first_bounds.shape
                or not np.array_equal(constraints[:-1], base_constraints)
                or not np.array_equal(bounds[:-1], base_bounds)
                for constraints, bounds in problems[1:]
            ):
                raise ValueError("receiver-history lexical union lacks one common support polytope")
            family = _LexicalFamily(
                common_constraints=base_constraints,
                common_bounds=base_bounds,
                disjunct_constraints=np.vstack(tuple(value[0][-1] for value in problems)),
                disjunct_bounds=np.asarray(tuple(value[1][-1] for value in problems)),
            )
    base_constraints = family.common_constraints
    base_bounds = family.common_bounds
    disjunct_constraints = family.disjunct_constraints
    disjunct_bounds = family.disjunct_bounds
    if (
        base_constraints.ndim != 2
        or base_bounds.shape != (base_constraints.shape[0],)
        or base_constraints.shape[1] < 3
        or disjunct_constraints.ndim != 2
        or disjunct_constraints.shape[1] != base_constraints.shape[1]
        or disjunct_bounds.shape != (disjunct_constraints.shape[0],)
    ):
        raise ValueError("receiver-history lexical union has an invalid compact roster")
    if not disjunct_bounds.size:
        return _solve_lexical_problem(base_constraints, base_bounds)

    continuous_count = base_constraints.shape[1]
    state_size = (continuous_count - 1) // 2
    selector_count = disjunct_bounds.size
    lower = np.concatenate(
        (
            np.full(2 * state_size, state_lower_bound, dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            np.zeros(selector_count, dtype=np.float64),
        )
    )
    upper = np.concatenate(
        (
            np.full(2 * state_size, state_upper_bound, dtype=np.float64),
            np.asarray([2.0], dtype=np.float64),
            np.ones(selector_count, dtype=np.float64),
        )
    )
    extended_base = sparse.hstack(
        (
            sparse.csc_matrix(base_constraints),
            sparse.csc_matrix((base_constraints.shape[0], selector_count)),
        ),
        format="csc",
    )
    branch_rows = np.zeros((selector_count, continuous_count + selector_count), dtype=np.float64)
    branch_bounds = np.empty(selector_count, dtype=np.float64)
    for selector_index, (disjunct, disjunct_bound_value) in enumerate(
        zip(disjunct_constraints, disjunct_bounds, strict=True)
    ):
        disjunct_bound = float(disjunct_bound_value)
        box_upper = float(
            np.sum(
                np.where(
                    disjunct >= 0.0,
                    disjunct * upper[:continuous_count],
                    disjunct * lower[:continuous_count],
                )
            )
        )
        deactivation = max(0.0, box_upper - disjunct_bound) + tolerance
        branch_rows[selector_index, :continuous_count] = disjunct
        branch_rows[selector_index, continuous_count + selector_index] = deactivation
        branch_bounds[selector_index] = disjunct_bound + deactivation
    selection = np.zeros(continuous_count + selector_count, dtype=np.float64)
    selection[continuous_count:] = -1.0
    constraint_matrix = sparse.vstack(
        (
            extended_base,
            sparse.csc_matrix(branch_rows),
            sparse.csc_matrix(selection[None, :]),
            sparse.csc_matrix((-selection)[None, :]),
        ),
        format="csc",
    )
    constraint_bounds = np.concatenate(
        (base_bounds, branch_bounds, np.asarray([-1.0, 1.0], dtype=np.float64))
    )
    objective = np.zeros(continuous_count + selector_count, dtype=np.float64)
    objective[continuous_count - 1] = -1.0
    integrality = np.concatenate(
        (
            np.zeros(continuous_count, dtype=np.int8),
            np.ones(selector_count, dtype=np.int8),
        )
    )
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            constraint_matrix,
            np.full(constraint_bounds.size, -np.inf, dtype=np.float64),
            constraint_bounds,
        ),
        options={"mip_rel_gap": min(1e-10, tolerance / 4.0)},
    )
    if result.status == 2:
        return None, 0.0, 0.0, 0.0, 0.0, True, True
    if result.status != 0 or result.x is None or result.fun is None:
        return None, 0.0, 2.0, 0.0, 2.0, False, False
    values = np.asarray(result.x, dtype=np.float64)
    primal = max(
        0.0,
        float(np.max(np.asarray(constraint_matrix @ values).ravel() - constraint_bounds)),
    )
    integrality_residual = float(
        np.max(np.abs(values[continuous_count:] - np.rint(values[continuous_count:])))
    )
    primal = max(primal, integrality_residual)
    objective_lower = max(0.0, float(values[continuous_count - 1]) - primal)
    dual_bound = getattr(result, "mip_dual_bound", result.fun)
    objective_upper = max(objective_lower, -float(dual_bound) + primal)
    gap = max(0.0, objective_upper - objective_lower)
    complete = primal <= tolerance and gap <= tolerance
    pair = (
        np.asarray(values[:state_size], dtype=np.float64),
        np.asarray(values[state_size : 2 * state_size], dtype=np.float64),
    )
    return pair, objective_lower, objective_upper, primal, gap, complete, False


def _classify_optimization_interval(
    *,
    lower: float,
    upper: float,
    margin: float,
    complete: bool,
    tolerance: float,
) -> ReceiverHistoryOptimizationState:
    """Classify only intervals separated from the frozen margin by tolerance."""

    if not complete:
        return ReceiverHistoryOptimizationState.RESOURCE_LIMITED
    if lower >= margin + tolerance:
        return ReceiverHistoryOptimizationState.WITNESS
    if upper <= margin - tolerance:
        return ReceiverHistoryOptimizationState.BELOW_MARGIN
    return ReceiverHistoryOptimizationState.RESOURCE_LIMITED


def _center_and_objectives(
    *,
    gate: ReceiverHistoryGate,
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    operator: DenseObserverOperator,
    homogeneous: dict[str, FloatArray],
    forced: dict[str, FloatArray],
) -> tuple[tuple[str, FloatArray, FloatArray], ...]:
    if gate is ReceiverHistoryGate.TARGET:
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
        if gate is ReceiverHistoryGate.TARGET:
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
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    coordinate: ReceiverHistoryCoordinateLabel,
    gate: ReceiverHistoryGate,
    action_id: str | None,
    orientation: ReceiverHistoryOrientation | None,
    state: ReceiverHistoryOptimizationState,
    lower: float,
    upper: float,
    margin: Decimal,
    primal_residual: float,
    duality_gap: float,
) -> ReceiverHistoryOptimizationCertificate:
    action_slug = "none" if action_id is None else action_id
    orientation_slug = "none" if orientation is None else orientation.value.lower()
    return ReceiverHistoryOptimizationCertificate(
        certificate_id=(
            f"optimization.{coordinate.coordinate_id}.{gate.value.lower()}.{action_slug}."
            f"{orientation_slug}"
        ),
        coordinate_id=coordinate.coordinate_id,
        gate=gate,
        action_id=action_id,
        orientation=orientation,
        state=state,
        objective_lower_bound=_decimal(max(0.0, lower)),
        objective_upper_bound=_decimal(max(0.0, upper)),
        required_margin=margin,
        primal_residual=_decimal(max(0.0, primal_residual)),
        duality_gap=_decimal(max(0.0, duality_gap)),
        solver_id='scipy-highs-lp-mip-support-polytope',
        constraint_family_sha256=sha256(
            canonical_json_bytes(
                (
                    'receiver-history-certificate-constraint-family',
                    config.fingerprint(),
                    descriptor.fingerprint(),
                    coordinate.coordinate_id,
                    gate.value,
                    action_id,
                    None if orientation is None else orientation.value,
                )
            )
        ).hexdigest(),
        complete_family_certificate=state is not ReceiverHistoryOptimizationState.RESOURCE_LIMITED,
        outcome_count_at_certificate=0,
    )


def _freeze_witness(
    *,
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    coordinate: ReceiverHistoryCoordinateLabel,
    gate: ReceiverHistoryGate,
    kind: ReceiverHistoryPairKind,
    orientation: ReceiverHistoryOrientation | None,
    action_id: str | None,
    pair_index: int,
    center: FloatArray,
    direction: FloatArray,
    implementation_sha256: str,
    homogeneous: dict[str, FloatArray],
    forced: dict[str, FloatArray],
    operator: DenseObserverOperator,
    history_stack: FloatArray,
) -> ReceiverHistoryChallengeNomination:
    amplitude = float(np.max(np.abs(direction)))
    if amplitude <= 0.0:
        raise ValueError("receiver-history targeter returned a zero witness")
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
    return ReceiverHistoryChallengeNomination(
        nomination_id=(
            f"nomination.{descriptor.unit_id}.n{descriptor.scale_cells}"
            f".{coordinate.coordinate_id}.{kind.value.lower().replace('_', '-')}"
            f".{('none' if orientation is None else orientation.value.lower().replace('_', '-'))}"
            f".{pair_index:02d}"
        ),
        coordinate_id=coordinate.coordinate_id,
        cohort=ReceiverHistoryCohort.TARGETED,
        unit_id=descriptor.unit_id,
        scale_cells=descriptor.scale_cells,
        depth=coordinate.depth,
        resolution_epsilon=coordinate.resolution_epsilon,
        pair_index=pair_index,
        pair_kind=kind,
        gate=gate,
        orientation=orientation,
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
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    history_forecast: ReceiverHistoryHistoryRankForecast,
    implementation_sha256: str,
    eligible_endpoints_by_coordinate: dict[str, frozenset[ReceiverHistoryEndpoint]] | None = None,
) -> TargetingExecution:
    'Solve all frozen targeted support problems and retain witnesses/certificates.'

    if history_forecast.descriptor_sha256 != descriptor.fingerprint():
        raise ValueError("receiver-history targeter forecast/descriptor identity differs")
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
    basis_by_depth: dict[int, FloatArray] = {}
    nominations: list[ReceiverHistoryChallengeNomination] = []
    certificates: list[ReceiverHistoryOptimizationCertificate] = []
    modes: list[FloatArray] = []
    for coordinate in coordinate_labels(config, descriptor.scale_cells):
        eligible_endpoints = (
            frozenset(ReceiverHistoryEndpoint)
            if eligible_endpoints_by_coordinate is None
            else eligible_endpoints_by_coordinate.get(coordinate.coordinate_id, frozenset())
        )
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

            def solve_dynamic_objective(
                item: tuple[str, FloatArray],
            ) -> tuple[str, FloatArray | None, float, float, float, bool]:
                objective_id, objective = item
                direction, value, primal, gap, complete = _solve_direction(
                    objective, dynamic_constraints, dynamic_bounds
                )
                return objective_id, direction, value, primal, gap, complete

            # These 24 support functions share immutable operands but remain
            # mathematically independent.  Two workers match the issued CPU
            # budget; map preserves the frozen lexical input order, and the
            # explicit value/ID sort below preserves the nomination tie-break.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rhc-dynamic") as pool:
                dynamic_results = tuple(pool.map(solve_dynamic_objective, dynamic_objectives))
            for objective_id, direction, value, primal, gap, complete in dynamic_results:
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
            dynamic_state = ReceiverHistoryOptimizationState.INFEASIBLE
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
        if ReceiverHistoryEndpoint.DYNAMICAL in eligible_endpoints:
            certificates.append(
                _certificate(
                    config=config,
                    descriptor=descriptor,
                    coordinate=coordinate,
                    gate=ReceiverHistoryGate.DYNAMICAL,
                    action_id=None,
                    orientation=None,
                    state=dynamic_state,
                    lower=dynamic_lower,
                    upper=dynamic_upper,
                    margin=config.future_divergence_epsilon,
                    primal_residual=dynamic_primal,
                    duality_gap=dynamic_gap,
                )
            )
        if (
            ReceiverHistoryEndpoint.DYNAMICAL in eligible_endpoints
            and dynamic_state is ReceiverHistoryOptimizationState.WITNESS
            and dynamic_maximum is not None
        ):
            dynamic_nomination = _freeze_witness(
                config=config,
                descriptor=descriptor,
                coordinate=coordinate,
                gate=ReceiverHistoryGate.DYNAMICAL,
                kind=ReceiverHistoryPairKind.DYNAMICAL_BOUNDARY,
                orientation=None,
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
            (
                ReceiverHistoryGate.TARGET,
                ReceiverHistoryPairKind.TARGET_LEXICAL,
                config.target_decision_margin,
            ),
            (
                ReceiverHistoryGate.SINK,
                ReceiverHistoryPairKind.SINK_LEXICAL,
                config.sink_decision_margin,
            ),
        ):
            endpoint = (
                ReceiverHistoryEndpoint.TARGET_DECISION
                if gate is ReceiverHistoryGate.TARGET
                else ReceiverHistoryEndpoint.SINK_DECISION
            )
            if endpoint not in eligible_endpoints:
                continue
            for pair_index, orientation in enumerate(ReceiverHistoryOrientation):
                witnesses: list[tuple[float, str, FloatArray, FloatArray]] = []

                def solve_action(
                    action_id: str,
                ) -> tuple[
                    str,
                    tuple[FloatArray, FloatArray] | None,
                    float,
                    float,
                    float,
                    float,
                    bool,
                    bool,
                ]:
                    problems = _lexical_pair_problems(
                        config=config,
                        descriptor=descriptor,
                        coordinate=coordinate,
                        propagations=propagations,
                        history=history_stack,
                        homogeneous=homogeneous[action_id],
                        forced=forced[action_id],
                        capacitances=np.asarray(operator.capacitances, dtype=np.float64),
                        gate=gate,
                        orientation=orientation,
                    )
                    solved = (
                        pair,
                        lower,
                        upper,
                        primal,
                        dual,
                        complete,
                        infeasible,
                    ) = _solve_lexical_family(
                        problems,
                        state_lower_bound=(
                            float(config.hidden_voltage_envelope[0])
                            + float(config.hidden_voltage_interior_margin)
                            + optimization_tolerance
                        ),
                        state_upper_bound=(
                            float(config.hidden_voltage_envelope[1])
                            - float(config.hidden_voltage_interior_margin)
                            - optimization_tolerance
                        ),
                        tolerance=optimization_tolerance,
                    )
                    return (action_id, *solved)

                action_ids = tuple(sorted(forced))
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rhc-lexical") as pool:
                    action_results = tuple(pool.map(solve_action, action_ids))
                for (
                    action_id,
                    pair,
                    lower,
                    upper,
                    primal,
                    dual,
                    complete,
                    infeasible,
                ) in action_results:
                    if infeasible:
                        pair = None
                        lower = upper = primal = dual = 0.0
                        complete = True
                        state = ReceiverHistoryOptimizationState.INFEASIBLE
                    else:
                        state = _classify_optimization_interval(
                            lower=lower,
                            upper=upper,
                            margin=float(margin),
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
                            orientation=orientation,
                            state=state,
                            lower=lower,
                            upper=upper,
                            margin=margin,
                            primal_residual=primal,
                            duality_gap=max(0.0, upper - lower, dual),
                        )
                    )
                    if pair is not None and state is ReceiverHistoryOptimizationState.WITNESS:
                        plus, minus = pair
                        center = np.asarray((plus + minus) / 2.0, dtype=np.float64)
                        direction = np.asarray((plus - minus) / 2.0, dtype=np.float64)
                        witnesses.append((lower, action_id, center, direction))
                witnesses.sort(key=lambda value: (-value[0], value[1]))
                if witnesses:
                    _value, action_id, center, direction = witnesses[0]
                    nomination = _freeze_witness(
                        config=config,
                        descriptor=descriptor,
                        coordinate=coordinate,
                        gate=gate,
                        kind=kind,
                        orientation=orientation,
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
        if not eligible_endpoints:
            continue
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
            nomination = ReceiverHistoryChallengeNomination(
                nomination_id=(
                    f"nomination.{descriptor.unit_id}.n{descriptor.scale_cells}"
                    f".{coordinate.coordinate_id}.random-fibre.{pair_index:02d}"
                ),
                coordinate_id=coordinate.coordinate_id,
                cohort=ReceiverHistoryCohort.TARGETED,
                unit_id=descriptor.unit_id,
                scale_cells=descriptor.scale_cells,
                depth=coordinate.depth,
                resolution_epsilon=coordinate.resolution_epsilon,
                pair_index=pair_index,
                pair_kind=ReceiverHistoryPairKind.RANDOM_FIBRE,
                gate=ReceiverHistoryGate.RANDOM,
                orientation=None,
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

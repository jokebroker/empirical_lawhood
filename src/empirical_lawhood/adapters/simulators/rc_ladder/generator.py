'Sparse, independently derived trajectory generator for receiver-history.\n\nThis module intentionally shares only receiver-history interchange records and ordinary\nNumPy/SciPy.  It does not import the circuit-response method implementation or the dense circuit solver.\n'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.linalg import expm
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse.linalg import expm_multiply

from empirical_lawhood.adapters.receiver_history.contracts import (
    ReceiverHistoryChallengeGeometry,
    ReceiverHistoryConfig,
    ReceiverHistoryCoordinateLabel,
    ReceiverHistoryDenominatorDescriptor,
    ReceiverHistoryGeneratorActionOutcome,
    ReceiverHistoryGeneratorOutcome,
    ReceiverHistoryGeneratorPairOutcome,
    ReceiverHistoryPreparationMapManifest,
    ReceiverHistoryGate,
    ReceiverHistoryOptimizationState,
    ReceiverHistoryOrientation,
    ReceiverHistorySparseCertificateCheck,
    ReceiverHistoryUntouchedGeneratorOutcome,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import canonical_json_bytes


FloatArray = npt.NDArray[np.float64]
_HIGHS_ACCURACY_OPTIONS = {
    "primal_feasibility_tolerance": 1e-10,
    "dual_feasibility_tolerance": 1e-10,
    "ipm_optimality_tolerance": 1e-12,
}


@dataclass(frozen=True, slots=True)
class SparseGeneratorOperator:
    state_matrix: sparse.csc_matrix
    left_input: FloatArray
    capacitances: FloatArray
    left_resistance: float
    right_resistance: float


@dataclass(frozen=True, slots=True)
class GeneratorExecution:
    outcome: ReceiverHistoryGeneratorOutcome
    arrays: dict[str, FloatArray]


@dataclass(frozen=True, slots=True)
class UntouchedGeneratorExecution:
    outcome: ReceiverHistoryUntouchedGeneratorOutcome
    arrays: dict[str, FloatArray]


@dataclass(frozen=True, slots=True)
class _SparseLexicalFamily:
    """Independent compact encoding of a shared polytope and finite union."""

    common_constraints: FloatArray
    common_bounds: FloatArray
    disjunct_constraints: FloatArray
    disjunct_bounds: FloatArray


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("receiver-history generator output must be finite")
    return Decimal(str(float(value)))


def assemble_sparse_operator(
    descriptor: ReceiverHistoryDenominatorDescriptor,
    *,
    left_resistance_factor: float = 1.0,
    right_resistance_factor: float = 1.0,
) -> SparseGeneratorOperator:
    """Derive C dV/dt + G V = boundary inputs using sparse diagonals."""

    if left_resistance_factor <= 0.0 or right_resistance_factor <= 0.0:
        raise ValueError("receiver-history resistance factors must be positive")
    n_cells = descriptor.scale_cells
    capacitances = np.asarray(descriptor.capacitances_farads, dtype=np.float64)
    interior_resistances = np.asarray(descriptor.interior_resistances_ohms, dtype=np.float64)
    if (
        capacitances.shape != (n_cells,)
        or interior_resistances.shape != (n_cells - 1,)
        or np.min(capacitances) <= 0.0
        or (interior_resistances.size and np.min(interior_resistances) <= 0.0)
    ):
        raise ValueError("receiver-history sparse generator received an invalid descriptor")
    left_resistance = float(descriptor.left_source_resistance_ohms) * left_resistance_factor
    right_resistance = float(descriptor.right_termination_resistance_ohms) * right_resistance_factor
    edge_conductances = 1.0 / interior_resistances
    diagonal = np.zeros(n_cells, dtype=np.float64)
    if edge_conductances.size:
        diagonal[:-1] += edge_conductances
        diagonal[1:] += edge_conductances
    left_conductance = 1.0 / left_resistance
    right_conductance = 1.0 / right_resistance
    diagonal[0] += left_conductance
    diagonal[-1] += right_conductance
    conductance = sparse.diags(
        diagonals=(-edge_conductances, diagonal, -edge_conductances),
        offsets=(-1, 0, 1),
        shape=(n_cells, n_cells),
        format="csc",
        dtype=np.float64,
    )
    inverse_c = sparse.diags(1.0 / capacitances, format="csc")
    state_matrix = sparse.csc_matrix(-(inverse_c @ conductance))
    left_input = np.zeros(n_cells, dtype=np.float64)
    left_input[0] = left_conductance / capacitances[0]
    if not np.all(np.isfinite(state_matrix.data)) or not np.all(np.isfinite(left_input)):
        raise ValueError("receiver-history sparse operator is nonfinite")
    return SparseGeneratorOperator(
        state_matrix=state_matrix,
        left_input=left_input,
        capacitances=capacitances,
        left_resistance=left_resistance,
        right_resistance=right_resistance,
    )


def receiver_matrix(
    descriptor: ReceiverHistoryDenominatorDescriptor, bin_count: int = 8
) -> FloatArray:
    """Independently derive capacitance-weighted equal-width receiver bins."""

    n_cells = descriptor.scale_cells
    if bin_count != 8 or n_cells % bin_count:
        raise ValueError("receiver-history generator requires an exact equal-width R8 partition")
    capacitances = np.asarray(descriptor.capacitances_farads, dtype=np.float64)
    matrix = np.zeros((bin_count, n_cells), dtype=np.float64)
    width = n_cells // bin_count
    for bin_index in range(bin_count):
        start = bin_index * width
        stop = start + width
        weights = capacitances[start:stop]
        matrix[bin_index, start:stop] = weights / np.sum(weights)
    return matrix


def _affine_action(
    operator: SparseGeneratorOperator,
    states: FloatArray,
    *,
    amplitude_volts: float,
    duration_seconds: float,
    endpoint_seconds: float,
) -> FloatArray:
    if not 0.0 < duration_seconds <= endpoint_seconds:
        raise ValueError("receiver-history action clock lies outside the panel endpoint")
    n_cells = states.shape[0]
    forcing = operator.left_input * amplitude_volts
    augmented = sparse.bmat(
        [
            [operator.state_matrix, sparse.csc_matrix(forcing[:, None])],
            [sparse.csc_matrix((1, n_cells)), sparse.csc_matrix((1, 1))],
        ],
        format="csc",
    )
    augmented_states = np.vstack((states, np.ones((1, states.shape[1]), dtype=np.float64)))
    driven = np.asarray(expm_multiply(augmented * duration_seconds, augmented_states))[:-1]
    coast = endpoint_seconds - duration_seconds
    if coast > 0.0:
        driven = np.asarray(expm_multiply(operator.state_matrix * coast, driven))
    return np.asarray(driven, dtype=np.float64)


def _factorized_action_components(
    operator: SparseGeneratorOperator,
    states: FloatArray,
    *,
    duration_seconds: float,
    endpoint_seconds: float,
) -> tuple[FloatArray, FloatArray]:
    'Return the homogeneous endpoint and unit-voltage duration response.\n\n    Linearity gives ``x(tau; a, d) = exp(A tau)x0 + a h_d``.  The response is\n    computed through the direct augmented system for one vector only, which\n    preserves the frozen circuit equation while avoiding one full propagation per action.\n    '

    homogeneous = np.asarray(
        expm_multiply(operator.state_matrix * endpoint_seconds, states),
        dtype=np.float64,
    )
    zero = np.zeros((states.shape[0], 1), dtype=np.float64)
    response = _affine_action(
        operator,
        zero,
        amplitude_volts=1.0,
        duration_seconds=duration_seconds,
        endpoint_seconds=endpoint_seconds,
    )[:, 0]
    return homogeneous, response


def _factorized_action_response(
    operator: SparseGeneratorOperator,
    *,
    duration_seconds: float,
    endpoint_seconds: float,
) -> FloatArray:
    """Return only the unit-voltage response when the homogeneous input is zero."""

    zero = np.zeros((operator.state_matrix.shape[0], 1), dtype=np.float64)
    return _affine_action(
        operator,
        zero,
        amplitude_volts=1.0,
        duration_seconds=duration_seconds,
        endpoint_seconds=endpoint_seconds,
    )[:, 0]


def _metrics(
    operator: SparseGeneratorOperator,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    states: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    denominator = descriptor.scale_cells * float(descriptor.capacitance_bar_farads)
    q_star = operator.capacitances @ states / denominator
    e_star = np.sum(operator.capacitances[:, None] * states**2, axis=0) / denominator
    v_max_star = np.max(states, axis=0) / float(descriptor.voltage_reference_volts)
    return (
        np.asarray(q_star, dtype=np.float64),
        np.asarray(e_star, dtype=np.float64),
        np.asarray(v_max_star, dtype=np.float64),
    )


def _array_digest(arrays: dict[str, FloatArray]) -> str:
    digest = sha256()
    for key in sorted(arrays):
        values = np.ascontiguousarray(arrays[key], dtype=np.float64)
        digest.update(key.encode("utf-8"))
        digest.update(b"\0float64\0")
        digest.update(canonical_json_bytes(tuple(int(value) for value in values.shape)))
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _nomination_set_digest(
    nominations: tuple[ReceiverHistoryChallengeGeometry, ...],
) -> str:
    return sha256(
        canonical_json_bytes(tuple(value.fingerprint() for value in nominations))
    ).hexdigest()


def _constraint_family_sha256(
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    geometry: ReceiverHistoryChallengeGeometry,
) -> str:
    # Deliberately implemented in the sparse source closure.  The tuple is the
    # interchange identity; no observer helper or predicted quantity is used.
    return sha256(
        canonical_json_bytes(
            (
                'receiver-history-support-polytope',
                config.fingerprint(),
                descriptor.fingerprint(),
                geometry.coordinate_id,
                geometry.depth,
                geometry.resolution_epsilon,
                geometry.gate.value,
                None if geometry.orientation is None else geometry.orientation.value,
                geometry.targeting_action_id,
                geometry.center_values,
                geometry.amplitude_volts,
                geometry.mode_values,
            )
        )
    ).hexdigest()


def _certificate_constraint_sha256(
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    coordinate_id: str,
    gate: ReceiverHistoryGate,
    action_id: str | None,
    orientation: ReceiverHistoryOrientation | None,
) -> str:
    return sha256(
        canonical_json_bytes(
            (
                'receiver-history-certificate-constraint-family',
                config.fingerprint(),
                descriptor.fingerprint(),
                coordinate_id,
                gate.value,
                action_id,
                None if orientation is None else orientation.value,
            )
        )
    ).hexdigest()


def _certificate_coordinates(
    config: ReceiverHistoryConfig, scale_cells: int
) -> tuple[tuple[str, int, Decimal], ...]:
    rows: list[tuple[str, int, Decimal]] = []
    for depth in config.absolute_depths:
        rows.append(
            (
                f"coordinate.n{scale_cells}.k{depth}.eps-{config.coordinate_collision_epsilon}",
                depth,
                config.coordinate_collision_epsilon,
            )
        )
    if scale_cells >= 64:
        for budget in config.normalized_history_budgets:
            depth_value = budget * Decimal(scale_cells) / Decimal(8) - Decimal(1)
            if depth_value != depth_value.to_integral_value():
                raise ValueError("receiver-history sparse budget coordinate is not integral")
            depth = int(depth_value)
            rows.append(
                (
                    f"coordinate.n{scale_cells}.b{str(budget).replace('.', 'p')}.k{depth}",
                    depth,
                    config.coordinate_collision_epsilon,
                )
            )
    for depth in config.resolution_panel_depths:
        for epsilon in config.receiver_resolution_panel:
            if epsilon != config.coordinate_collision_epsilon:
                rows.append((f"coordinate.n{scale_cells}.k{depth}.eps-{epsilon}", depth, epsilon))
    return tuple(sorted(set(rows)))


def _sparse_history_polytope(
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    operator: SparseGeneratorOperator,
    *,
    depth: int,
    epsilon: Decimal,
    center: FloatArray,
    propagations: tuple[FloatArray, ...] | None = None,
    history: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    matrix = operator.state_matrix.toarray()
    if (propagations is None) != (history is None):
        raise ValueError("receiver-history sparse support cache is incomplete")
    if propagations is None:
        lag = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
        inverse_lag = expm(-matrix * lag)
        values = [np.eye(descriptor.scale_cells, dtype=np.float64)]
        for _ in range(depth):
            values.append(inverse_lag @ values[-1])
        propagations = tuple(values)
        receiver = receiver_matrix(descriptor, config.receiver_bin_count)
        history = np.vstack(tuple(receiver @ value for value in propagations))
    if len(propagations) != depth + 1 or history is None:
        raise ValueError("receiver-history sparse support cache depth differs")
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
        if float(np.min(room)) <= 0.0:
            raise ValueError("receiver-history sparse certificate center lacks history support")
        positive_matrices.append(propagation)
        positive_bounds.append(room)
    collision_half_bound = (float(epsilon) - tolerance) / 2.0
    if collision_half_bound <= 0.0:
        raise ValueError("receiver-history sparse certificate tolerance exhausts support")
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
    return (
        np.vstack((positive, -positive)),
        np.concatenate((positive_bound, positive_bound)),
    )


def _solve_certificate_direction(
    objective: FloatArray,
    constraints: FloatArray,
    bounds: FloatArray,
) -> tuple[float, float, float, float, bool]:
    """Independent HiGHS primal/dual interval for an absolute objective."""
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
        raise ValueError("receiver-history sparse support polytope is not a finite pair family")
    half = constraints.shape[0] // 2
    if not np.array_equal(constraints[half:], -constraints[:half]) or not np.array_equal(
        bounds[half:], bounds[:half]
    ):
        raise ValueError("receiver-history sparse support polytope lost exact central symmetry")

    signed_objective = -objective
    result = linprog(
        signed_objective,
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(None, None)] * objective.size,
        method="highs",
        options=_HIGHS_ACCURACY_OPTIONS,
    )
    if not result.success or result.x is None:
        return 0.0, 0.0, float("inf"), float("inf"), False
    direction = np.asarray(result.x, dtype=np.float64)
    constraint_values = constraints @ direction
    positive = constraint_values > 0.0
    feasible_scale = min(
        1.0,
        float(np.min(bounds[positive] / constraint_values[positive])) if np.any(positive) else 1.0,
    )
    feasible = direction * max(0.0, feasible_scale)
    primal = max(0.0, float(np.max(constraints @ feasible - bounds)))
    lower = max(0.0, float(objective @ feasible)) * 2.0
    marginals = np.asarray(result.ineqlin.marginals, dtype=np.float64)
    dual = float(np.linalg.norm(signed_objective - constraints.T @ marginals, ord=np.inf))
    variable_bound = float(objective.size) * float(np.max(bounds[-2 * objective.size :]))
    upper = max(lower, -2.0 * float(bounds @ marginals) + 2.0 * variable_bound * dual)
    return lower, upper, primal, dual, True


def _sparse_lexical_pair_problems(
    *,
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    depth: int,
    epsilon: Decimal,
    propagations: tuple[FloatArray, ...],
    history: FloatArray,
    homogeneous: FloatArray,
    forced: FloatArray,
    capacitances: FloatArray,
    gate: ReceiverHistoryGate,
    orientation: ReceiverHistoryOrientation,
) -> _SparseLexicalFamily:
    """Sparse closure's independent two-state lexical constraint assembly."""

    if len(propagations) != depth + 1:
        raise ValueError("receiver-history sparse lexical propagation roster differs")
    size = descriptor.scale_cells
    column_count = 2 * size + 1
    tolerance = float(config.optimization_tolerance)
    envelope_low = (
        float(config.hidden_voltage_envelope[0])
        + float(config.hidden_voltage_interior_margin)
        + tolerance
    )
    envelope_high = (
        float(config.hidden_voltage_envelope[1])
        - float(config.hidden_voltage_interior_margin)
        - tolerance
    )
    base_rows: list[FloatArray] = []
    base_bounds: list[float] = []

    def append_to(
        row_target: list[FloatArray],
        bound_target: list[float],
        plus: FloatArray | None,
        minus: FloatArray | None,
        slack: float,
        bound: float,
    ) -> None:
        encoded = np.zeros(column_count, dtype=np.float64)
        if plus is not None:
            encoded[:size] = plus
        if minus is not None:
            encoded[size : 2 * size] = minus
        encoded[-1] = slack
        row_target.append(encoded)
        bound_target.append(float(bound))

    for transition in propagations:
        for transition_row in transition:
            vector = np.asarray(transition_row, dtype=np.float64)
            append_to(base_rows, base_bounds, vector, None, 0.0, envelope_high)
            append_to(base_rows, base_bounds, -vector, None, 0.0, -envelope_low)
            append_to(base_rows, base_bounds, None, vector, 0.0, envelope_high)
            append_to(base_rows, base_bounds, None, -vector, 0.0, -envelope_low)
    collision_limit = float(epsilon) - tolerance
    if collision_limit <= 0.0:
        raise ValueError("receiver-history sparse lexical collision bound is empty")
    for receiver_row in history:
        vector = np.asarray(receiver_row, dtype=np.float64)
        append_to(base_rows, base_bounds, vector, -vector, 0.0, collision_limit)
        append_to(base_rows, base_bounds, -vector, vector, 0.0, collision_limit)
    pair_limit = 2.0 * float(config.hidden_amplitude_max_volts) - tolerance
    for identity_row in np.eye(size, dtype=np.float64):
        append_to(base_rows, base_bounds, identity_row, -identity_row, 0.0, pair_limit)
        append_to(base_rows, base_bounds, -identity_row, identity_row, 0.0, pair_limit)
    append_to(base_rows, base_bounds, None, None, 1.0, 2.0)
    append_to(base_rows, base_bounds, None, None, -1.0, 0.0)

    denominator = size * float(descriptor.capacitance_bar_farads)
    charge_row = np.asarray(capacitances @ homogeneous / denominator, dtype=np.float64)
    forced_charge = float(capacitances @ forced / denominator)
    target_boundary = float(config.target_charge_minimum_q_star)
    target_half = float(config.target_decision_margin) / 2.0
    sink_boundary = float(config.sink_voltage_maximum_v_star) * float(
        descriptor.voltage_reference_volts
    )
    sink_half = float(config.sink_decision_margin) / 2.0 * float(descriptor.voltage_reference_volts)

    if gate is ReceiverHistoryGate.TARGET:
        if orientation is ReceiverHistoryOrientation.UNSAFE_PROMOTION:
            append_to(
                base_rows,
                base_bounds,
                -charge_row,
                None,
                0.5,
                forced_charge - target_boundary,
            )
            append_to(
                base_rows,
                base_bounds,
                None,
                charge_row,
                0.5,
                target_boundary - forced_charge,
            )
        else:
            append_to(
                base_rows,
                base_bounds,
                charge_row,
                None,
                0.5,
                target_boundary - forced_charge,
            )
            append_to(
                base_rows,
                base_bounds,
                None,
                -charge_row,
                0.5,
                forced_charge - target_boundary,
            )
        sink_safe = sink_boundary - sink_half
        for component, endpoint_row in enumerate(homogeneous):
            vector = np.asarray(endpoint_row, dtype=np.float64)
            component_bound = sink_safe - float(forced[component])
            append_to(base_rows, base_bounds, vector, None, 0.0, component_bound)
            append_to(base_rows, base_bounds, None, vector, 0.0, component_bound)
        return _SparseLexicalFamily(
            common_constraints=np.vstack(base_rows),
            common_bounds=np.asarray(base_bounds, dtype=np.float64),
            disjunct_constraints=np.empty((0, column_count), dtype=np.float64),
            disjunct_bounds=np.empty(0, dtype=np.float64),
        )

    if gate is not ReceiverHistoryGate.SINK:
        raise ValueError("receiver-history sparse lexical gate differs")
    charge_safe = forced_charge - (target_boundary + target_half)
    append_to(base_rows, base_bounds, -charge_row, None, 0.0, charge_safe)
    append_to(base_rows, base_bounds, None, -charge_row, 0.0, charge_safe)
    reference_safe = orientation is ReceiverHistoryOrientation.UNSAFE_PROMOTION
    for component, safe_row in enumerate(homogeneous):
        vector = np.asarray(safe_row, dtype=np.float64)
        component_bound = sink_boundary - float(forced[component])
        if reference_safe:
            append_to(base_rows, base_bounds, vector, None, 0.5, component_bound)
        else:
            append_to(base_rows, base_bounds, None, vector, 0.5, component_bound)

    branch_rows: list[FloatArray] = []
    branch_bounds: list[float] = []
    for active_component, endpoint_row in enumerate(homogeneous):
        active = np.asarray(endpoint_row, dtype=np.float64)
        failure_bound = float(forced[active_component]) - sink_boundary
        encoded = np.zeros(column_count, dtype=np.float64)
        if reference_safe:
            encoded[size : 2 * size] = -active
        else:
            encoded[:size] = -active
        encoded[-1] = 0.5
        branch_rows.append(encoded)
        branch_bounds.append(failure_bound)
    return _SparseLexicalFamily(
        common_constraints=np.vstack(base_rows),
        common_bounds=np.asarray(base_bounds, dtype=np.float64),
        disjunct_constraints=np.vstack(branch_rows),
        disjunct_bounds=np.asarray(branch_bounds, dtype=np.float64),
    )


def _solve_sparse_lexical_problem(
    constraints: FloatArray,
    bounds: FloatArray,
) -> tuple[float, float, float, float, bool, bool]:
    """Sparse closure's separate primal/dual interval for lexical margin."""

    if constraints.ndim != 2 or bounds.shape != (constraints.shape[0],):
        raise ValueError("receiver-history sparse lexical LP shape differs")
    objective = np.zeros(constraints.shape[1], dtype=np.float64)
    objective[-1] = -1.0
    result = linprog(
        objective,
        A_ub=constraints,
        b_ub=bounds,
        bounds=[(None, None)] * constraints.shape[1],
        method="highs",
        options=_HIGHS_ACCURACY_OPTIONS,
    )
    if result.status == 2:
        return 0.0, 0.0, 0.0, 0.0, True, True
    if not result.success or result.x is None:
        return 0.0, 2.0, 0.0, 0.0, False, False
    values = np.asarray(result.x, dtype=np.float64)
    primal = max(0.0, float(np.max(constraints @ values - bounds)))
    lower = max(0.0, float(values[-1]) - primal)
    multipliers = np.asarray(result.ineqlin.marginals, dtype=np.float64)
    dual = float(np.linalg.norm(objective - constraints.T @ multipliers, ord=np.inf))
    state_size = (constraints.shape[1] - 1) // 2
    upper = max(lower, -float(bounds @ multipliers) + (2.0 * state_size + 2.0) * dual)
    return lower, upper, primal, dual, True, False


def _solve_sparse_lexical_family(
    problems: tuple[tuple[FloatArray, FloatArray], ...] | _SparseLexicalFamily,
    *,
    state_lower_bound: float,
    state_upper_bound: float,
    tolerance: float,
) -> tuple[float, float, float, float, bool, bool]:
    """Independently solve the same finite union with bounded selectors."""

    if isinstance(problems, _SparseLexicalFamily):
        family = problems
    else:
        if not problems:
            raise ValueError("receiver-history sparse lexical family is empty")
        template, template_bounds = problems[0]
        if len(problems) == 1:
            family = _SparseLexicalFamily(
                common_constraints=template,
                common_bounds=template_bounds,
                disjunct_constraints=np.empty((0, template.shape[1]), dtype=np.float64),
                disjunct_bounds=np.empty(0, dtype=np.float64),
            )
        else:
            common = template[:-1]
            common_bounds = template_bounds[:-1]
            if any(
                matrix.shape != template.shape
                or bounds.shape != template_bounds.shape
                or not np.array_equal(matrix[:-1], common)
                or not np.array_equal(bounds[:-1], common_bounds)
                for matrix, bounds in problems[1:]
            ):
                raise ValueError("receiver-history sparse lexical union lacks common constraints")
            family = _SparseLexicalFamily(
                common_constraints=common,
                common_bounds=common_bounds,
                disjunct_constraints=np.vstack(tuple(value[0][-1] for value in problems)),
                disjunct_bounds=np.asarray(tuple(value[1][-1] for value in problems)),
            )
    common = family.common_constraints
    common_bounds = family.common_bounds
    branches = family.disjunct_constraints
    branch_limits = family.disjunct_bounds
    if (
        common.ndim != 2
        or common_bounds.shape != (common.shape[0],)
        or common.shape[1] < 3
        or branches.ndim != 2
        or branches.shape[1] != common.shape[1]
        or branch_limits.shape != (branches.shape[0],)
    ):
        raise ValueError("receiver-history sparse lexical compact roster differs")
    if not branch_limits.size:
        return _solve_sparse_lexical_problem(common, common_bounds)
    continuous_count = common.shape[1]
    state_size = (continuous_count - 1) // 2
    selector_count = branch_limits.size
    variable_lower = np.r_[
        np.full(2 * state_size, state_lower_bound),
        0.0,
        np.zeros(selector_count),
    ].astype(np.float64)
    variable_upper = np.r_[
        np.full(2 * state_size, state_upper_bound),
        2.0,
        np.ones(selector_count),
    ].astype(np.float64)
    extended_common = sparse.hstack(
        (
            sparse.csc_matrix(common),
            sparse.csc_matrix((common.shape[0], selector_count)),
        ),
        format="csc",
    )
    encoded_branches = np.zeros(
        (selector_count, continuous_count + selector_count), dtype=np.float64
    )
    encoded_limits = np.empty(selector_count, dtype=np.float64)
    for selector_index, (branch, branch_limit) in enumerate(
        zip(branches, branch_limits, strict=True)
    ):
        branch_bound = float(branch_limit)
        box_maximum = float(
            np.sum(
                np.where(
                    branch >= 0.0,
                    branch * variable_upper[:continuous_count],
                    branch * variable_lower[:continuous_count],
                )
            )
        )
        big_m = max(0.0, box_maximum - branch_bound) + tolerance
        encoded_branches[selector_index, :continuous_count] = branch
        encoded_branches[selector_index, continuous_count + selector_index] = big_m
        encoded_limits[selector_index] = branch_bound + big_m
    selector_row = np.zeros(continuous_count + selector_count, dtype=np.float64)
    selector_row[continuous_count:] = -1.0
    coefficient_matrix = sparse.vstack(
        (
            extended_common,
            sparse.csc_matrix(encoded_branches),
            sparse.csc_matrix(selector_row[None, :]),
            sparse.csc_matrix((-selector_row)[None, :]),
        ),
        format="csc",
    )
    upper_bounds = np.concatenate(
        (common_bounds, encoded_limits, np.asarray([-1.0, 1.0], dtype=np.float64))
    )
    objective = np.zeros(continuous_count + selector_count, dtype=np.float64)
    objective[continuous_count - 1] = -1.0
    integrality = np.r_[
        np.zeros(continuous_count, dtype=np.int8),
        np.ones(selector_count, dtype=np.int8),
    ]
    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(
            coefficient_matrix,
            np.full(upper_bounds.size, -np.inf, dtype=np.float64),
            upper_bounds,
        ),
        options={"mip_rel_gap": min(1e-10, tolerance / 4.0)},
    )
    if result.status == 2:
        return 0.0, 0.0, 0.0, 0.0, True, True
    if result.status != 0 or result.x is None or result.fun is None:
        return 0.0, 2.0, 0.0, 2.0, False, False
    values = np.asarray(result.x, dtype=np.float64)
    primal = max(
        0.0,
        float(np.max(np.asarray(coefficient_matrix @ values).ravel() - upper_bounds)),
    )
    primal = max(
        primal,
        float(np.max(np.abs(values[continuous_count:] - np.rint(values[continuous_count:])))),
    )
    lower = max(0.0, float(values[continuous_count - 1]) - primal)
    dual_bound = getattr(result, "mip_dual_bound", result.fun)
    upper = max(lower, -float(dual_bound) + primal)
    gap = max(0.0, upper - lower)
    return lower, upper, primal, gap, primal <= tolerance and gap <= tolerance, False


def _optimization_state(
    *, lower: float, upper: float, margin: float, complete: bool, tolerance: float
) -> ReceiverHistoryOptimizationState:
    if not complete:
        return ReceiverHistoryOptimizationState.RESOURCE_LIMITED
    if lower >= margin + tolerance:
        return ReceiverHistoryOptimizationState.WITNESS
    if upper <= margin - tolerance:
        return ReceiverHistoryOptimizationState.BELOW_MARGIN
    return ReceiverHistoryOptimizationState.RESOURCE_LIMITED


def _certificate_check(
    *,
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    coordinate_id: str,
    gate: ReceiverHistoryGate,
    action_id: str | None,
    orientation: ReceiverHistoryOrientation | None,
    lower: float,
    upper: float,
    margin: Decimal,
    primal: float,
    dual: float,
    complete: bool,
    state_override: ReceiverHistoryOptimizationState | None = None,
) -> ReceiverHistorySparseCertificateCheck:
    action_slug = "none" if action_id is None else action_id
    orientation_slug = "none" if orientation is None else orientation.value.lower()
    state = state_override or _optimization_state(
        lower=lower,
        upper=upper,
        margin=float(margin),
        complete=complete,
        tolerance=float(config.optimization_tolerance),
    )
    return ReceiverHistorySparseCertificateCheck(
        objective_key=(
            f"optimization.{coordinate_id}.{gate.value.lower()}.{action_slug}.{orientation_slug}"
        ),
        coordinate_id=coordinate_id,
        gate=gate,
        action_id=action_id,
        orientation=orientation,
        state=state,
        objective_lower_bound=_decimal(max(0.0, lower)),
        objective_upper_bound=_decimal(max(0.0, upper)),
        required_margin=margin,
        primal_residual=_decimal(max(0.0, primal)),
        duality_gap=_decimal(max(0.0, dual)),
        interval_width=_decimal(max(0.0, upper - lower)),
        constraint_family_sha256=_certificate_constraint_sha256(
            config, descriptor, coordinate_id, gate, action_id, orientation
        ),
        solver_id='scipy-highs-lp-mip-sparse-reconstruction',
        precision_bits=53,
        complete_family_certificate=complete
        and state is not ReceiverHistoryOptimizationState.RESOURCE_LIMITED,
    )


def _sparse_certificate_family(
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    operator: SparseGeneratorOperator,
    requested_keys: tuple[str, ...] | None = None,
) -> tuple[ReceiverHistorySparseCertificateCheck, ...]:
    """Solve the complete sparse objective family without observer records."""

    if requested_keys is not None and requested_keys != tuple(sorted(set(requested_keys))):
        raise ValueError("receiver-history sparse certificate request mask is not canonical")
    requested = None if requested_keys is None else frozenset(requested_keys)
    matrix = operator.state_matrix.toarray()
    receiver = receiver_matrix(descriptor, config.receiver_bin_count)
    endpoint = float(config.panel_endpoint_t_star) * float(descriptor.time_scale_seconds)
    homogeneous = expm(matrix * endpoint)
    lag = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    inverse_lag = expm(-matrix * lag)
    all_propagations = [np.eye(descriptor.scale_cells, dtype=np.float64)]
    for _ in range(config.history_max_depth):
        all_propagations.append(inverse_lag @ all_propagations[-1])
    history_by_depth = {
        depth: np.vstack(tuple(receiver @ value for value in all_propagations[: depth + 1]))
        for depth in range(config.history_max_depth + 1)
    }
    dynamic_objectives = tuple(
        np.asarray(receiver_row @ propagation, dtype=np.float64)
        for time_star in config.diagnostic_times_t_star
        for propagation in (expm(matrix * float(time_star) * float(descriptor.time_scale_seconds)),)
        for receiver_row in receiver
    )
    responses: dict[int, FloatArray] = {}
    for duration_index, duration in enumerate(config.action_durations_t_star):
        response = _factorized_action_response(
            operator,
            duration_seconds=float(duration) * float(descriptor.time_scale_seconds),
            endpoint_seconds=endpoint,
        )
        responses[duration_index] = response
    action_forced: dict[str, FloatArray] = {}
    for amplitude_index, amplitude in enumerate(config.action_amplitudes_u_star):
        for duration_index, _duration in enumerate(config.action_durations_t_star):
            action_forced[f"action.a{amplitude_index:02d}.d{duration_index:02d}"] = (
                float(amplitude)
                * float(descriptor.voltage_reference_volts)
                * responses[duration_index]
            )
    checks: list[ReceiverHistorySparseCertificateCheck] = []

    for coordinate_id, depth, epsilon in _certificate_coordinates(config, descriptor.scale_cells):
        coordinate_prefix = f"optimization.{coordinate_id}."
        if requested is not None and not any(
            value.startswith(coordinate_prefix) for value in requested
        ):
            continue
        propagations = tuple(all_propagations[: depth + 1])
        history = history_by_depth[depth]
        dynamic_key = f"optimization.{coordinate_id}.dynamical.none.none"
        if requested is None or dynamic_key in requested:
            dynamic_center = np.full(descriptor.scale_cells, 0.5, dtype=np.float64)
            try:
                constraints, bounds = _sparse_history_polytope(
                    config,
                    descriptor,
                    operator,
                    depth=depth,
                    epsilon=epsilon,
                    center=dynamic_center,
                    propagations=propagations,
                    history=history,
                )
                dynamic_rows: list[tuple[float, float, float, float, bool]] = []
                for objective in dynamic_objectives:
                    dynamic_rows.append(
                        _solve_certificate_direction(
                            objective,
                            constraints,
                            bounds,
                        )
                    )
                best = max(dynamic_rows, key=lambda value: value[0])
                complete = all(value[4] for value in dynamic_rows)
                lower, upper, primal, dual, _ = best
                dynamic_state_override = None
            except ValueError:
                lower = upper = primal = dual = 0.0
                complete = True
                dynamic_state_override = ReceiverHistoryOptimizationState.INFEASIBLE
            checks.append(
                _certificate_check(
                    config=config,
                    descriptor=descriptor,
                    coordinate_id=coordinate_id,
                    gate=ReceiverHistoryGate.DYNAMICAL,
                    action_id=None,
                    orientation=None,
                    lower=lower,
                    upper=upper,
                    margin=config.future_divergence_epsilon,
                    primal=primal,
                    dual=dual,
                    complete=complete,
                    state_override=dynamic_state_override,
                )
            )

        for gate, margin in (
            (ReceiverHistoryGate.TARGET, config.target_decision_margin),
            (ReceiverHistoryGate.SINK, config.sink_decision_margin),
        ):
            for orientation in ReceiverHistoryOrientation:
                for action_id in sorted(action_forced):
                    objective_key = (
                        f"optimization.{coordinate_id}.{gate.value.lower()}.{action_id}."
                        f"{orientation.value.lower()}"
                    )
                    if requested is not None and objective_key not in requested:
                        continue
                    problems = _sparse_lexical_pair_problems(
                        config=config,
                        descriptor=descriptor,
                        depth=depth,
                        epsilon=epsilon,
                        propagations=propagations,
                        history=history,
                        homogeneous=np.asarray(homogeneous, dtype=np.float64),
                        forced=action_forced[action_id],
                        capacitances=np.asarray(operator.capacitances, dtype=np.float64),
                        gate=gate,
                        orientation=orientation,
                    )
                    (
                        lower,
                        upper,
                        primal,
                        dual,
                        complete,
                        infeasible,
                    ) = _solve_sparse_lexical_family(
                        problems,
                        state_lower_bound=(
                            float(config.hidden_voltage_envelope[0])
                            + float(config.hidden_voltage_interior_margin)
                            + float(config.optimization_tolerance)
                        ),
                        state_upper_bound=(
                            float(config.hidden_voltage_envelope[1])
                            - float(config.hidden_voltage_interior_margin)
                            - float(config.optimization_tolerance)
                        ),
                        tolerance=float(config.optimization_tolerance),
                    )
                    if infeasible:
                        lower = upper = primal = dual = 0.0
                        complete = True
                        state = ReceiverHistoryOptimizationState.INFEASIBLE
                    else:
                        state = _optimization_state(
                            lower=lower,
                            upper=upper,
                            margin=float(margin),
                            complete=complete,
                            tolerance=float(config.optimization_tolerance),
                        )
                    check = _certificate_check(
                        config=config,
                        descriptor=descriptor,
                        coordinate_id=coordinate_id,
                        gate=gate,
                        action_id=action_id,
                        orientation=orientation,
                        lower=lower,
                        upper=upper,
                        margin=margin,
                        primal=primal,
                        dual=dual,
                        complete=complete,
                        state_override=state,
                    )
                    checks.append(check)
    values = tuple(sorted(checks, key=lambda value: value.objective_key))
    if requested_keys is None:
        return values
    if tuple(value.objective_key for value in values) != requested_keys:
        raise ValueError("receiver-history sparse certificate request mask contains an unknown key")
    return values


def _complete_collision_graph_digest(
    histories: FloatArray,
    coordinates: tuple[ReceiverHistoryCoordinateLabel, ...],
    validity: npt.NDArray[np.bool_],
) -> str:
    'Independently reconstruct and hash the complete requested untouched graph set.'

    count = histories.shape[0]
    left, right = np.triu_indices(count, k=1)
    cumulative = np.zeros(left.size, dtype=np.float64)
    valid_pairs = validity[left] & validity[right]
    by_depth: dict[int, list[tuple[int, ReceiverHistoryCoordinateLabel]]] = {}
    for coordinate_index, coordinate in enumerate(coordinates):
        by_depth.setdefault(coordinate.depth, []).append((coordinate_index, coordinate))
    parts: list[npt.NDArray[np.int64]] = []
    for depth in range(histories.shape[1]):
        distance = np.max(np.abs(histories[left, depth] - histories[right, depth]), axis=1)
        np.maximum(cumulative, distance, out=cumulative)
        for coordinate_index, coordinate in by_depth.get(depth, ()):
            selected = valid_pairs & (cumulative <= float(coordinate.resolution_epsilon))
            if np.any(selected):
                parts.append(
                    np.column_stack(
                        (
                            np.full(np.count_nonzero(selected), coordinate_index, dtype=np.int64),
                            left[selected],
                            right[selected],
                        )
                    ).astype(np.int64, copy=False)
                )
    edges = np.vstack(parts) if parts else np.empty((0, 3), dtype=np.int64)
    return sha256(
        canonical_json_bytes(tuple(value.coordinate_id for value in coordinates))
        + np.ascontiguousarray(edges, dtype=np.int64).tobytes(order="C")
    ).hexdigest()


def generate_outcomes(
    *,
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    nominations: tuple[ReceiverHistoryChallengeGeometry, ...],
    implementation_sha256: str,
    outcome_access: OutcomeAccess,
    certificate_request_keys: tuple[str, ...] | None = None,
    action_clock_factor: float = 1.0,
    left_resistance_factor: float = 1.0,
    right_resistance_factor: float = 1.0,
) -> GeneratorExecution:
    """Realize one complete scale-view challenge without assigning a verdict."""

    if tuple(value.nomination_id for value in nominations) != tuple(
        sorted({value.nomination_id for value in nominations})
    ):
        raise ValueError("receiver-history nominations must be sorted and unique")
    if any(
        value.unit_id != descriptor.unit_id or value.scale_cells != descriptor.scale_cells
        for value in nominations
    ):
        raise ValueError("receiver-history nomination/descriptor identity differs")
    if any(
        value.constraint_family_sha256 != _constraint_family_sha256(config, descriptor, value)
        or value.lexical_reference_side != "PLUS"
        for value in nominations
    ):
        raise ValueError("receiver-history sparse constraint-family reconstruction differs")
    if action_clock_factor <= 0.0:
        raise ValueError("receiver-history action-clock factor must be positive")
    operator = assemble_sparse_operator(
        descriptor,
        left_resistance_factor=left_resistance_factor,
        right_resistance_factor=right_resistance_factor,
    )
    receiver = receiver_matrix(descriptor, config.receiver_bin_count)
    sparse_certificate_checks = _sparse_certificate_family(
        config,
        descriptor,
        operator,
        requested_keys=certificate_request_keys,
    )
    n_pairs = len(nominations)
    if n_pairs == 0:
        empty_arrays = {
            "present-states": np.empty((0, 2, descriptor.scale_cells), dtype=np.float64)
        }
        outcome = ReceiverHistoryGeneratorOutcome(
            outcome_id=f"generator-outcome.{descriptor.unit_id}.n{descriptor.scale_cells}",
            unit_id=descriptor.unit_id,
            descriptor_sha256=descriptor.fingerprint(),
            nomination_set_sha256=_nomination_set_digest(nominations),
            generator_implementation_sha256=implementation_sha256,
            pair_outcomes=(),
            sparse_certificate_checks=sparse_certificate_checks,
            arrays_sha256=_array_digest(empty_arrays),
            array_count=1,
            outcome_access=outcome_access,
            scientific_verdict_assigned=False,
        )
        return GeneratorExecution(outcome=outcome, arrays=empty_arrays)
    states = np.empty((descriptor.scale_cells, 2 * n_pairs), dtype=np.float64)
    for pair_index, nomination in enumerate(nominations):
        mode = np.asarray(nomination.mode_values, dtype=np.float64)
        center = np.asarray(nomination.center_values, dtype=np.float64)
        hidden_amplitude = float(nomination.amplitude_volts)
        states[:, 2 * pair_index] = center + hidden_amplitude * mode
        states[:, 2 * pair_index + 1] = center - hidden_amplitude * mode
    lower = float(config.hidden_voltage_envelope[0]) + float(config.hidden_voltage_interior_margin)
    upper = float(config.hidden_voltage_envelope[1]) - float(config.hidden_voltage_interior_margin)
    envelope_valid = np.ones(n_pairs, dtype=np.bool_)
    envelope_valid &= (
        np.min(states.reshape(descriptor.scale_cells, n_pairs, 2), axis=(0, 2)) >= lower
    )
    envelope_valid &= (
        np.max(states.reshape(descriptor.scale_cells, n_pairs, 2), axis=(0, 2)) <= upper
    )

    arrays: dict[str, FloatArray] = {"present-states": states.T.reshape(n_pairs, 2, -1)}
    coordinate_defects = np.zeros(n_pairs, dtype=np.float64)
    history_step_seconds = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    history = np.asarray(
        expm_multiply(
            -operator.state_matrix,
            states,
            start=0.0,
            stop=history_step_seconds * config.history_max_depth,
            num=config.history_max_depth + 1,
            endpoint=True,
            traceA=float(-operator.state_matrix.diagonal().sum()),
        ),
        dtype=np.float64,
    )
    for depth in range(config.history_max_depth + 1):
        observed = receiver @ history[depth]
        for pair_index, nomination in enumerate(nominations):
            if depth <= nomination.depth:
                coordinate_defects[pair_index] = max(
                    coordinate_defects[pair_index],
                    float(
                        np.max(
                            np.abs(observed[:, 2 * pair_index] - observed[:, 2 * pair_index + 1])
                        )
                    ),
                )
                envelope_valid[pair_index] &= (
                    np.min(history[depth, :, 2 * pair_index : 2 * pair_index + 2]) >= lower
                    and np.max(history[depth, :, 2 * pair_index : 2 * pair_index + 2]) <= upper
                )

    maximum_future_receiver = np.zeros(n_pairs, dtype=np.float64)
    maximum_future_metric = np.zeros(n_pairs, dtype=np.float64)
    future_receiver_by_pair = np.zeros((n_pairs, 3), dtype=np.float64)
    for time_index, time_star in enumerate(config.diagnostic_times_t_star):
        future = np.asarray(
            expm_multiply(
                operator.state_matrix * (float(time_star) * float(descriptor.time_scale_seconds)),
                states,
            ),
            dtype=np.float64,
        )
        receiver_values = receiver @ future
        q_star, e_star, v_max_star = _metrics(operator, descriptor, future)
        for pair_index in range(n_pairs):
            left = 2 * pair_index
            right = left + 1
            receiver_defect = float(
                np.max(np.abs(receiver_values[:, left] - receiver_values[:, right]))
            )
            metric_defect = max(
                abs(float(q_star[left] - q_star[right])),
                abs(float(e_star[left] - e_star[right])),
                abs(float(v_max_star[left] - v_max_star[right])),
            )
            future_receiver_by_pair[pair_index, time_index] = receiver_defect
            maximum_future_receiver[pair_index] = max(
                maximum_future_receiver[pair_index], receiver_defect
            )
            maximum_future_metric[pair_index] = max(
                maximum_future_metric[pair_index], metric_defect
            )

    action_outcomes: list[list[ReceiverHistoryGeneratorActionOutcome]] = [[] for _ in nominations]
    endpoint_seconds = (
        float(config.panel_endpoint_t_star)
        * float(descriptor.time_scale_seconds)
        * action_clock_factor
    )
    homogeneous = np.asarray(
        expm_multiply(operator.state_matrix * endpoint_seconds, states), dtype=np.float64
    )
    for duration_index, duration in enumerate(config.action_durations_t_star):
        duration_seconds = (
            float(duration) * float(descriptor.time_scale_seconds) * action_clock_factor
        )
        response = _factorized_action_response(
            operator,
            duration_seconds=duration_seconds,
            endpoint_seconds=endpoint_seconds,
        )
        for amplitude_index, action_amplitude in enumerate(config.action_amplitudes_u_star):
            action_id = f"action.a{amplitude_index:02d}.d{duration_index:02d}"
            realized = np.asarray(
                homogeneous
                + (float(action_amplitude) * float(descriptor.voltage_reference_volts))
                * response[:, None],
                dtype=np.float64,
            )
            q_star, e_star, v_max_star = _metrics(operator, descriptor, realized)
            left_current = -realized[0, :] / operator.left_resistance
            right_current = -realized[-1, :] / operator.right_resistance
            for pair_index, nomination in enumerate(nominations):
                left = 2 * pair_index
                right = left + 1
                midpoint_state = ((realized[:, left] + realized[:, right]) / 2.0)[:, None]
                midpoint_q, midpoint_e, midpoint_v_max = _metrics(
                    operator, descriptor, midpoint_state
                )
                maximum_future_metric[pair_index] = max(
                    maximum_future_metric[pair_index],
                    abs(float(q_star[left] - q_star[right])),
                    abs(float(e_star[left] - e_star[right])),
                    abs(float(v_max_star[left] - v_max_star[right])),
                )
                plus_target = bool(q_star[left] >= float(config.target_charge_minimum_q_star))
                minus_target = bool(q_star[right] >= float(config.target_charge_minimum_q_star))
                plus_sink = bool(v_max_star[left] <= float(config.sink_voltage_maximum_v_star))
                minus_sink = bool(v_max_star[right] <= float(config.sink_voltage_maximum_v_star))
                midpoint_target = bool(
                    midpoint_q[0]
                    >= float(config.target_charge_minimum_q_star)
                    - float(config.midpoint_boundary_tie_epsilon)
                )
                midpoint_sink = bool(
                    midpoint_v_max[0]
                    <= float(config.sink_voltage_maximum_v_star)
                    + float(config.midpoint_boundary_tie_epsilon)
                )
                requested_id = action_id
                action_outcomes[pair_index].append(
                    ReceiverHistoryGeneratorActionOutcome(
                        action_id=action_id,
                        requested_action_id=requested_id,
                        accepted_action_id=requested_id,
                        applied_action_id=requested_id,
                        realized_action_id=requested_id,
                        amplitude_u_star=action_amplitude,
                        duration_t_star=duration,
                        plus_q_star=_decimal(q_star[left]),
                        minus_q_star=_decimal(q_star[right]),
                        plus_e_star=_decimal(e_star[left]),
                        minus_e_star=_decimal(e_star[right]),
                        plus_v_max_star=_decimal(v_max_star[left]),
                        minus_v_max_star=_decimal(v_max_star[right]),
                        midpoint_q_star=_decimal(midpoint_q[0]),
                        midpoint_e_star=_decimal(midpoint_e[0]),
                        midpoint_v_max_star=_decimal(midpoint_v_max[0]),
                        plus_target_pass=plus_target,
                        minus_target_pass=minus_target,
                        midpoint_target_pass=midpoint_target,
                        plus_sink_pass=plus_sink,
                        minus_sink_pass=minus_sink,
                        midpoint_sink_pass=midpoint_sink,
                        plus_admit=plus_target and plus_sink,
                        minus_admit=minus_target and minus_sink,
                        midpoint_admit=midpoint_target and midpoint_sink,
                        plus_left_current_amperes=_decimal(left_current[left]),
                        minus_left_current_amperes=_decimal(left_current[right]),
                        plus_right_current_amperes=_decimal(right_current[left]),
                        minus_right_current_amperes=_decimal(right_current[right]),
                        requested_accepted_applied_realized_parity=True,
                    )
                )

    pair_records = tuple(
        sorted(
            (
                ReceiverHistoryGeneratorPairOutcome(
                    nomination_id=nomination.nomination_id,
                    realized_coordinate_defect=_decimal(coordinate_defects[pair_index]),
                    maximum_future_receiver_defect=_decimal(maximum_future_receiver[pair_index]),
                    maximum_future_metric_defect=_decimal(maximum_future_metric[pair_index]),
                    future_receiver_defects=tuple(
                        _decimal(value) for value in future_receiver_by_pair[pair_index]
                    ),
                    action_outcomes=tuple(
                        sorted(action_outcomes[pair_index], key=lambda value: value.action_id)
                    ),
                    voltage_envelope_valid=bool(envelope_valid[pair_index]),
                )
                for pair_index, nomination in enumerate(nominations)
            ),
            key=lambda value: value.nomination_id,
        )
    )
    arrays_sha256 = _array_digest(arrays)
    outcome = ReceiverHistoryGeneratorOutcome(
        outcome_id=f"generator-outcome.{descriptor.unit_id}.n{descriptor.scale_cells}",
        unit_id=descriptor.unit_id,
        descriptor_sha256=descriptor.fingerprint(),
        nomination_set_sha256=_nomination_set_digest(nominations),
        generator_implementation_sha256=implementation_sha256,
        pair_outcomes=pair_records,
        sparse_certificate_checks=sparse_certificate_checks,
        arrays_sha256=arrays_sha256,
        array_count=len(arrays),
        outcome_access=outcome_access,
        scientific_verdict_assigned=False,
    )
    return GeneratorExecution(outcome=outcome, arrays=arrays)


def generate_untouched_outcomes(
    *,
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    preparation_manifest: ReceiverHistoryPreparationMapManifest,
    earliest_states: FloatArray,
    coordinates: tuple[ReceiverHistoryCoordinateLabel, ...],
    validity_mask: npt.NDArray[np.int64],
    implementation_sha256: str,
    outcome_access: OutcomeAccess,
) -> UntouchedGeneratorExecution:
    'Independently propagate all untouched preparations and the complete action panel.'

    if (
        preparation_manifest.unit_id != descriptor.unit_id
        or earliest_states.shape
        != (
            preparation_manifest.preparation_count,
            descriptor.scale_cells,
        )
        or validity_mask.shape != (preparation_manifest.preparation_count,)
    ):
        raise ValueError("receiver-history untouched generator preparation identity differs")
    if not np.all(np.isfinite(earliest_states)):
        raise ValueError("receiver-history untouched generator states are nonfinite")
    operator = assemble_sparse_operator(descriptor)
    receiver = receiver_matrix(descriptor, config.receiver_bin_count)
    lag_seconds = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    state = np.asarray(earliest_states.T, dtype=np.float64)
    timeline = np.asarray(
        expm_multiply(
            operator.state_matrix,
            state,
            start=0.0,
            stop=lag_seconds * config.history_max_depth,
            num=config.history_max_depth + 1,
            endpoint=True,
            traceA=float(operator.state_matrix.diagonal().sum()),
        ),
        dtype=np.float64,
    )
    state_timeline = np.transpose(timeline, (2, 0, 1))
    present = timeline[-1]
    receiver_forward = np.einsum("bti,ri->btr", state_timeline, receiver)
    receiver_by_lag = np.asarray(receiver_forward[:, ::-1, :], dtype=np.float64)
    validity = np.asarray(validity_mask != 0, dtype=np.bool_)
    graph_sha256 = _complete_collision_graph_digest(receiver_by_lag, coordinates, validity)
    action_ids = tuple(
        sorted(
            f"action.a{amplitude_index:02d}.d{duration_index:02d}"
            for amplitude_index, _amplitude in enumerate(config.action_amplitudes_u_star)
            for duration_index, _duration in enumerate(config.action_durations_t_star)
        )
    )
    action_metrics = np.empty(
        (preparation_manifest.preparation_count, len(action_ids), 3),
        dtype=np.float64,
    )
    action_receiver = np.empty(
        (preparation_manifest.preparation_count, len(action_ids), config.receiver_bin_count),
        dtype=np.float64,
    )
    endpoint_seconds = float(config.panel_endpoint_t_star) * float(descriptor.time_scale_seconds)
    homogeneous = np.asarray(
        expm_multiply(operator.state_matrix * endpoint_seconds, present), dtype=np.float64
    )
    for duration_index, duration in enumerate(config.action_durations_t_star):
        duration_seconds = float(duration) * float(descriptor.time_scale_seconds)
        response = _factorized_action_response(
            operator,
            duration_seconds=duration_seconds,
            endpoint_seconds=endpoint_seconds,
        )
        for amplitude_index, amplitude in enumerate(config.action_amplitudes_u_star):
            action_index = amplitude_index * len(config.action_durations_t_star) + duration_index
            realized = np.asarray(
                homogeneous
                + float(amplitude) * float(descriptor.voltage_reference_volts) * response[:, None],
                dtype=np.float64,
            )
            q_star, e_star, v_max_star = _metrics(operator, descriptor, realized)
            action_receiver[:, action_index, :] = (receiver @ realized).T
            action_metrics[:, action_index, 0] = q_star
            action_metrics[:, action_index, 1] = e_star
            action_metrics[:, action_index, 2] = v_max_star
    arrays = {
        "present-states": np.asarray(present.T, dtype=np.float64),
        "receiver-history": receiver_by_lag,
        "action-metrics": action_metrics,
        "action-receiver": action_receiver,
    }
    outcome = ReceiverHistoryUntouchedGeneratorOutcome(
        outcome_id=(f"untouched-generator-outcome.{descriptor.unit_id}.n{descriptor.scale_cells}"),
        unit_id=descriptor.unit_id,
        scale_cells=descriptor.scale_cells,
        descriptor_sha256=descriptor.fingerprint(),
        preparation_manifest_sha256=preparation_manifest.fingerprint(),
        generator_implementation_sha256=implementation_sha256,
        preparation_count=preparation_manifest.preparation_count,
        valid_preparation_count=int(np.count_nonzero(validity)),
        action_ids=action_ids,
        complete_collision_graph_sha256=graph_sha256,
        validity_mask_sha256=sha256(
            np.ascontiguousarray(validity_mask, dtype=np.int64).tobytes(order="C")
        ).hexdigest(),
        arrays_sha256=_array_digest(arrays),
        outcome_access=outcome_access,
        requested_accepted_applied_realized_parity=True,
        scientific_verdict_assigned=False,
    )
    return UntouchedGeneratorExecution(outcome=outcome, arrays=arrays)


__all__ = [
    "GeneratorExecution",
    "SparseGeneratorOperator",
    "UntouchedGeneratorExecution",
    "assemble_sparse_operator",
    "generate_outcomes",
    "generate_untouched_outcomes",
    "receiver_matrix",
]

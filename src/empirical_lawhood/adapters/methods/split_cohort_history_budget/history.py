"""Outcome-blind dense observer and boundary targeter for split cohort history budget.

The implementation independently derives the RC equations and never imports
the sparse generator or the frozen physical-scale numerical and simulator-challenge solvers.
"""

from __future__ import annotations

from empirical_lawhood.adapters.history_budget_scientific_inputs import HistoryBudgetDescriptorScientificInput, require_history_budget_descriptor_input

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm, svd

from empirical_lawhood.adapters.split_cohort_history_budget.contracts import SplitCohortHistoryBudgetActionPrediction, SplitCohortHistoryBudgetChallengeNomination, SplitCohortHistoryBudgetCohort, SplitCohortHistoryBudgetConfig, SplitCohortHistoryBudgetCoordinateKind, SplitCohortHistoryBudgetCoordinateLabel, SplitCohortHistoryBudgetDenominatorDescriptor, SplitCohortHistoryBudgetGate, SplitCohortHistoryBudgetHistoryRankForecast, SplitCohortHistoryBudgetPairKind, SplitCohortHistoryBudgetRankStep, SplitCohortHistoryBudgetScientificState

FloatArray = npt.NDArray[np.float64]


def normalized_budget_depth(scale_cells: int, budget: Decimal) -> int:
    """Map the exact nominal row budget b=8(k+1)/N to one integer depth."""

    numerator = budget * Decimal(scale_cells)
    depth = numerator / Decimal(8) - Decimal(1)
    if depth != depth.to_integral_value() or depth < 0:
        raise ValueError("split cohort history budget normalized budget does not map to an entered depth")
    return int(depth)


def requested_depths(config: SplitCohortHistoryBudgetConfig, scale_cells: int) -> tuple[int, ...]:
    """Return unique physical depths; overlapping K/B labels are computed once."""

    values = set(config.absolute_depths)
    if scale_cells >= 64:
        values.update(
            normalized_budget_depth(scale_cells, budget)
            for budget in config.normalized_history_budgets
        )
    if min(values) < 0 or max(values) > config.history_max_depth:
        raise ValueError("split cohort history budget requested depth exceeds the frozen history envelope")
    return tuple(sorted(values))


def coordinate_labels(
    config: SplitCohortHistoryBudgetConfig,
    scale_cells: int,
) -> tuple[SplitCohortHistoryBudgetCoordinateLabel, ...]:
    """Return frozen K/B labels and the secondary K4/K6/K8 resolution views."""

    labels: list[SplitCohortHistoryBudgetCoordinateLabel] = []
    primary_epsilon = config.coordinate_collision_epsilon
    for depth in config.absolute_depths:
        labels.append(
            SplitCohortHistoryBudgetCoordinateLabel(
                coordinate_id=f"coordinate.n{scale_cells}.k{depth}.eps-{primary_epsilon}",
                kind=SplitCohortHistoryBudgetCoordinateKind.ABSOLUTE_DEPTH,
                scale_cells=scale_cells,
                depth=depth,
                budget=None,
                resolution_epsilon=primary_epsilon,
                primary=depth in config.primary_absolute_depth_contrast,
            )
        )
    if scale_cells >= 64:
        for budget in config.normalized_history_budgets:
            depth = normalized_budget_depth(scale_cells, budget)
            budget_slug = str(budget).replace(".", "p")
            labels.append(
                SplitCohortHistoryBudgetCoordinateLabel(
                    coordinate_id=f"coordinate.n{scale_cells}.b{budget_slug}.k{depth}",
                    kind=SplitCohortHistoryBudgetCoordinateKind.NORMALIZED_BUDGET,
                    scale_cells=scale_cells,
                    depth=depth,
                    budget=budget,
                    resolution_epsilon=primary_epsilon,
                    primary=budget in config.primary_normalized_budget_contrast,
                )
            )
    for depth in config.resolution_panel_depths:
        for epsilon in config.receiver_resolution_panel:
            if epsilon == primary_epsilon:
                continue
            labels.append(
                SplitCohortHistoryBudgetCoordinateLabel(
                    coordinate_id=f"coordinate.n{scale_cells}.k{depth}.eps-{epsilon}",
                    kind=SplitCohortHistoryBudgetCoordinateKind.ABSOLUTE_DEPTH,
                    scale_cells=scale_cells,
                    depth=depth,
                    budget=None,
                    resolution_epsilon=epsilon,
                    primary=False,
                )
            )
    return tuple(sorted(labels, key=lambda value: value.coordinate_id))


@dataclass(frozen=True, slots=True)
class DenseObserverOperator:
    state_matrix: FloatArray
    left_input: FloatArray
    capacitances: FloatArray
    receiver: FloatArray


@dataclass(frozen=True, slots=True)
class ObserverExecution:
    forecast: SplitCohortHistoryBudgetHistoryRankForecast
    nominations: tuple[SplitCohortHistoryBudgetChallengeNomination, ...]
    arrays: dict[str, FloatArray]


@dataclass(frozen=True, slots=True)
class HistoryExecution:
    forecast: SplitCohortHistoryBudgetHistoryRankForecast
    arrays: dict[str, FloatArray]


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("split cohort history budget observer output must be finite")
    return Decimal(str(float(value)))


def assemble_dense_operator(
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
) -> DenseObserverOperator:
    """Independently derive the homogeneous and left-boundary input maps."""

    n_cells = descriptor.scale_cells
    capacitances = np.asarray(descriptor.capacitances_farads, dtype=np.float64)
    resistances = np.asarray(descriptor.interior_resistances_ohms, dtype=np.float64)
    conductance = np.zeros((n_cells, n_cells), dtype=np.float64)
    for edge_index in range(n_cells - 1):
        edge_conductance = 1.0 / resistances[edge_index]
        conductance[edge_index, edge_index] += edge_conductance
        conductance[edge_index + 1, edge_index + 1] += edge_conductance
        conductance[edge_index, edge_index + 1] -= edge_conductance
        conductance[edge_index + 1, edge_index] -= edge_conductance
    left_conductance = 1.0 / float(descriptor.left_source_resistance_ohms)
    right_conductance = 1.0 / float(descriptor.right_termination_resistance_ohms)
    conductance[0, 0] += left_conductance
    conductance[-1, -1] += right_conductance
    state_matrix = -conductance / capacitances[:, None]
    left_input = np.zeros(n_cells, dtype=np.float64)
    left_input[0] = left_conductance / capacitances[0]
    receiver = np.zeros((8, n_cells), dtype=np.float64)
    width = n_cells // 8
    if width * 8 != n_cells:
        raise ValueError("split cohort history budget observer requires exact equal-width R8 bins")
    for bin_index in range(8):
        start = width * bin_index
        stop = start + width
        receiver[bin_index, start:stop] = capacitances[start:stop] / np.sum(
            capacitances[start:stop]
        )
    if (
        not np.all(np.isfinite(state_matrix))
        or not np.all(np.linalg.eigvals(state_matrix).real < 0.0)
        or not np.allclose(receiver.sum(axis=1), 1.0, atol=1e-14, rtol=0.0)
    ):
        raise ValueError("split cohort history budget observer operator failed stability/receiver checks")
    return DenseObserverOperator(
        state_matrix=state_matrix,
        left_input=left_input,
        capacitances=capacitances,
        receiver=receiver,
    )


def sampled_history_matrix(
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    depth: int,
) -> FloatArray:
    """Assemble the actual backward-sampled history operator O_k once."""

    if depth < 0 or depth > config.history_max_depth:
        raise ValueError("split cohort history budget sampled-history depth is outside the frozen envelope")
    operator = assemble_dense_operator(descriptor)
    lag_seconds = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    inverse_lag = expm(-operator.state_matrix * lag_seconds)
    current = operator.receiver.copy()
    blocks = [current.copy()]
    for _ in range(depth):
        current = current @ inverse_lag
        blocks.append(current.copy())
    result = np.vstack(blocks)
    if not np.all(np.isfinite(result)):
        raise ValueError("split cohort history budget sampled-history operator is nonfinite")
    return np.asarray(result, dtype=np.float64)


def _rank_forecast(
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    operator: DenseObserverOperator,
) -> tuple[SplitCohortHistoryBudgetHistoryRankForecast, dict[str, FloatArray], tuple[FloatArray, ...]]:
    n_cells = descriptor.scale_cells
    lag_seconds = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    inverse_lag = expm(-operator.state_matrix * lag_seconds)
    blocks: list[FloatArray] = []
    current = operator.receiver.copy()
    spectra = np.zeros((config.history_max_depth + 1, n_cells), dtype=np.float64)
    algebraic_spectra = np.zeros_like(spectra)
    stacks: list[FloatArray] = []
    steps: list[SplitCohortHistoryBudgetRankStep] = []
    previous_effective = 0
    threshold = float(config.rank_relative_threshold)
    for depth in range(config.history_max_depth + 1):
        if depth:
            current = current @ inverse_lag
        blocks.append(current.copy())
        stack = np.vstack(blocks)
        stacks.append(stack)
        singular_values = np.linalg.svd(stack, compute_uv=False)
        spectra[depth, : singular_values.size] = singular_values
        relative = singular_values / singular_values[0]
        effective_rank = int(np.sum(relative > threshold))
        row_norms = np.linalg.norm(stack, axis=1)
        if np.any(row_norms <= 0.0) or not np.all(np.isfinite(row_norms)):
            raise ValueError("split cohort history budget observer history stack has an invalid row norm")
        equilibrated = stack / row_norms[:, None]
        comparator_values = np.linalg.svd(equilibrated, compute_uv=False)
        algebraic_spectra[depth, : comparator_values.size] = comparator_values
        comparator_relative = comparator_values / comparator_values[0]
        algebraic_rank = int(np.sum(comparator_relative > threshold))
        ceiling = min(n_cells, 8 * (depth + 1))
        retained = float(relative[effective_rank - 1]) if effective_rank else 0.0
        condition = (
            float(singular_values[0] / singular_values[-1])
            if singular_values.size == n_cells and singular_values[-1] > 0.0
            else None
        )
        steps.append(
            SplitCohortHistoryBudgetRankStep(
                depth=depth,
                effective_rank=effective_rank,
                algebraic_comparator_rank=algebraic_rank,
                row_count_ceiling=ceiling,
                largest_singular_value=_decimal(singular_values[0]),
                smallest_retained_relative_singular_value=_decimal(retained),
                condition_number=None if condition is None else _decimal(condition),
                nonmonotone_effective_rank=effective_rank < previous_effective,
            )
        )
        previous_effective = effective_rank
    if not np.all(np.isfinite(spectra)) or not np.all(np.isfinite(algebraic_spectra)):
        raise ValueError("split cohort history budget observer singular spectrum is nonfinite")
    k_full = next(
        (step.depth for step in steps if step.effective_rank == n_cells),
        None,
    )
    probes = set(requested_depths(config, descriptor.scale_cells))
    algebraic_state = (
        SplitCohortHistoryBudgetScientificState.SUPPORTED
        if all(step.algebraic_comparator_rank == step.row_count_ceiling for step in steps)
        else SplitCohortHistoryBudgetScientificState.OPPOSED
    )
    arrays = {
        "singular-spectra": spectra,
        "row-equilibrated-singular-spectra": algebraic_spectra,
        "inverse-lag-propagator": inverse_lag,
    }
    spectrum_sha = sha256(
        spectra.tobytes(order="C") + algebraic_spectra.tobytes(order="C")
    ).hexdigest()
    forecast = SplitCohortHistoryBudgetHistoryRankForecast(
        forecast_id=f"rank-forecast.{descriptor.unit_id}.n{n_cells}",
        descriptor_sha256=descriptor.fingerprint(),
        rank_steps=tuple(steps),
        singular_spectra_sha256=spectrum_sha,
        k_full_effective=k_full,
        effective_rank_right_censored=k_full is None,
        probe_depths=tuple(sorted(probes)),
        algebraic_prediction_state=algebraic_state,
    )
    return forecast, arrays, tuple(stacks)


def _homogeneous_metrics(
    operator: DenseObserverOperator,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    states: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    denominator = descriptor.scale_cells * float(descriptor.capacitance_bar_farads)
    q_star = operator.capacitances @ states / denominator
    e_star = np.sum(operator.capacitances[:, None] * states**2, axis=0) / denominator
    v_max_star = np.max(states, axis=0) / float(descriptor.voltage_reference_volts)
    return q_star, e_star, v_max_star


def _forced_endpoint_by_action(
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    operator: DenseObserverOperator,
) -> tuple[dict[str, FloatArray], dict[str, FloatArray]]:
    homogeneous_by_action: dict[str, FloatArray] = {}
    forced: dict[str, FloatArray] = {}
    identity = np.eye(descriptor.scale_cells, dtype=np.float64)
    endpoint_seconds = float(config.panel_endpoint_t_star) * float(descriptor.time_scale_seconds)
    for duration_index, duration in enumerate(config.action_durations_t_star):
        duration_seconds = float(duration) * float(descriptor.time_scale_seconds)
        duration_propagator = expm(operator.state_matrix * duration_seconds)
        driven_per_volt = np.linalg.solve(
            operator.state_matrix,
            (duration_propagator - identity) @ operator.left_input,
        )
        coast_seconds = endpoint_seconds - duration_seconds
        if coast_seconds < 0.0:
            raise ValueError("split cohort history budget action duration exceeds the panel endpoint")
        coast_propagator = expm(operator.state_matrix * coast_seconds)
        homogeneous = np.asarray(coast_propagator @ duration_propagator, dtype=np.float64)
        forced_per_volt = np.asarray(coast_propagator @ driven_per_volt, dtype=np.float64)
        for amplitude_index, amplitude in enumerate(config.action_amplitudes_u_star):
            action_id = f"action.a{amplitude_index:02d}.d{duration_index:02d}"
            homogeneous_by_action[action_id] = homogeneous
            forced[action_id] = np.asarray(
                forced_per_volt * float(amplitude) * float(descriptor.voltage_reference_volts),
                dtype=np.float64,
            )
    return homogeneous_by_action, forced


def _gate_carrier(
    *,
    gate: SplitCohortHistoryBudgetGate,
    forced: FloatArray,
    homogeneous_profile: FloatArray,
    operator: DenseObserverOperator,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    config: SplitCohortHistoryBudgetConfig,
) -> float | None:
    if gate is SplitCohortHistoryBudgetGate.TARGET:
        denominator = descriptor.scale_cells * float(descriptor.capacitance_bar_farads)
        slope = float(operator.capacitances @ homogeneous_profile / denominator)
        intercept = float(operator.capacitances @ forced / denominator)
        if slope <= 0.0:
            return None
        carrier = (float(config.target_charge_minimum_q_star) - intercept) / slope
    elif gate is SplitCohortHistoryBudgetGate.SINK:
        threshold = float(config.sink_voltage_maximum_v_star) * float(
            descriptor.voltage_reference_volts
        )
        lower = float(config.hidden_voltage_envelope[0])
        upper = float(config.hidden_voltage_envelope[1])
        at_lower = float(np.max(forced + lower * homogeneous_profile))
        at_upper = float(np.max(forced + upper * homogeneous_profile))
        if not at_lower <= threshold <= at_upper:
            return None
        for _ in range(80):
            midpoint = (lower + upper) / 2.0
            value = float(np.max(forced + midpoint * homogeneous_profile))
            if value < threshold:
                lower = midpoint
            else:
                upper = midpoint
        carrier = (lower + upper) / 2.0
    else:
        return 0.5
    margin = float(config.hidden_voltage_interior_margin)
    if not margin < carrier < 1.0 - margin:
        return None
    return carrier


def _amplitude_limit(
    *,
    center: FloatArray,
    mode: FloatArray,
    stack: FloatArray,
    inverse_lag: FloatArray,
    depth: int,
    config: SplitCohortHistoryBudgetConfig,
    collision_epsilon: float,
) -> tuple[float, float] | None:
    lower = float(config.hidden_voltage_envelope[0]) + float(config.hidden_voltage_interior_margin)
    upper = float(config.hidden_voltage_envelope[1]) - float(config.hidden_voltage_interior_margin)
    baseline = np.asarray(center, dtype=np.float64).copy()
    history_mode = mode.copy()
    amplitude_limit = float(config.hidden_amplitude_max_volts)
    for history_depth in range(depth + 1):
        if np.min(baseline) <= lower or np.max(baseline) >= upper:
            return None
        nonzero = np.abs(history_mode) > 1e-15
        if np.any(nonzero):
            room = np.minimum(baseline - lower, upper - baseline)
            amplitude_limit = min(
                amplitude_limit,
                float(np.min(room[nonzero] / np.abs(history_mode[nonzero]))),
            )
        if history_depth < depth:
            baseline = inverse_lag @ baseline
            history_mode = inverse_lag @ history_mode
    coordinate_per_amplitude = float(np.max(np.abs(stack @ mode))) * 2.0
    if coordinate_per_amplitude > 0.0:
        amplitude_limit = min(
            amplitude_limit,
            collision_epsilon / coordinate_per_amplitude,
        )
    if not np.isfinite(amplitude_limit) or amplitude_limit <= 1e-10:
        return None
    amplitude = amplitude_limit * (1.0 - 1e-10)
    coordinate_defect = coordinate_per_amplitude * amplitude
    return amplitude, coordinate_defect


def _orient_mode(mode: FloatArray, sensitivity: FloatArray) -> FloatArray:
    values = np.asarray(mode, dtype=np.float64)
    maximum = float(np.max(np.abs(values)))
    if maximum <= 0.0:
        raise ValueError("split cohort history budget observer selected a zero hidden mode")
    values = values / maximum
    if float(sensitivity @ values) < 0.0:
        values = -values
    first = int(np.flatnonzero(np.abs(values) > 1e-14)[0])
    if abs(float(sensitivity @ values)) <= 1e-14 and values[first] < 0.0:
        values = -values
    return values


def _predict_actions(
    *,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    operator: DenseObserverOperator,
    homogeneous_by_action: dict[str, FloatArray],
    forced_by_action: dict[str, FloatArray],
    center: FloatArray,
    amplitude: float,
    mode: FloatArray,
) -> tuple[SplitCohortHistoryBudgetActionPrediction, ...]:
    plus_present = center + amplitude * mode
    minus_present = center - amplitude * mode
    rows: list[SplitCohortHistoryBudgetActionPrediction] = []
    for amplitude_index, action_amplitude in enumerate(config.action_amplitudes_u_star):
        for duration_index, action_duration in enumerate(config.action_durations_t_star):
            action_id = f"action.a{amplitude_index:02d}.d{duration_index:02d}"
            forced = forced_by_action[action_id]
            homogeneous_endpoint = homogeneous_by_action[action_id]
            states = np.column_stack(
                (
                    forced + homogeneous_endpoint @ plus_present,
                    forced + homogeneous_endpoint @ minus_present,
                )
            )
            q_star, e_star, v_max_star = _homogeneous_metrics(operator, descriptor, states)
            midpoint_state = forced + homogeneous_endpoint @ center
            midpoint_q, midpoint_e, midpoint_v_max = _homogeneous_metrics(
                operator, descriptor, midpoint_state[:, None]
            )
            plus_target = bool(q_star[0] >= float(config.target_charge_minimum_q_star))
            minus_target = bool(q_star[1] >= float(config.target_charge_minimum_q_star))
            midpoint_target = bool(
                midpoint_q[0]
                >= float(config.target_charge_minimum_q_star)
                - float(config.midpoint_boundary_tie_epsilon)
            )
            plus_sink = bool(v_max_star[0] <= float(config.sink_voltage_maximum_v_star))
            minus_sink = bool(v_max_star[1] <= float(config.sink_voltage_maximum_v_star))
            midpoint_sink = bool(
                midpoint_v_max[0]
                <= float(config.sink_voltage_maximum_v_star)
                + float(config.midpoint_boundary_tie_epsilon)
            )
            rows.append(
                SplitCohortHistoryBudgetActionPrediction(
                    action_id=action_id,
                    amplitude_u_star=action_amplitude,
                    duration_t_star=action_duration,
                    requested_action_id=action_id,
                    predicted_plus_q_star=_decimal(q_star[0]),
                    predicted_minus_q_star=_decimal(q_star[1]),
                    predicted_plus_e_star=_decimal(e_star[0]),
                    predicted_minus_e_star=_decimal(e_star[1]),
                    predicted_plus_v_max_star=_decimal(v_max_star[0]),
                    predicted_minus_v_max_star=_decimal(v_max_star[1]),
                    predicted_midpoint_q_star=_decimal(midpoint_q[0]),
                    predicted_midpoint_e_star=_decimal(midpoint_e[0]),
                    predicted_midpoint_v_max_star=_decimal(midpoint_v_max[0]),
                    predicted_plus_target_pass=plus_target,
                    predicted_minus_target_pass=minus_target,
                    predicted_midpoint_target_pass=midpoint_target,
                    predicted_plus_sink_pass=plus_sink,
                    predicted_minus_sink_pass=minus_sink,
                    predicted_midpoint_sink_pass=midpoint_sink,
                    predicted_plus_admit=plus_target and plus_sink,
                    predicted_minus_admit=minus_target and minus_sink,
                    predicted_midpoint_admit=midpoint_target and midpoint_sink,
                )
            )
    return tuple(sorted(rows, key=lambda value: value.action_id))


def _future_defects(
    *,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    operator: DenseObserverOperator,
    center: FloatArray,
    amplitude: float,
    mode: FloatArray,
) -> tuple[float, float]:
    states = np.column_stack((center + amplitude * mode, center - amplitude * mode))
    maximum_receiver = 0.0
    maximum_metric = 0.0
    for time_star in config.diagnostic_times_t_star:
        propagated = (
            expm(operator.state_matrix * (float(time_star) * float(descriptor.time_scale_seconds)))
            @ states
        )
        receiver_values = operator.receiver @ propagated
        q_star, e_star, v_max_star = _homogeneous_metrics(operator, descriptor, propagated)
        maximum_receiver = max(
            maximum_receiver,
            float(np.max(np.abs(receiver_values[:, 0] - receiver_values[:, 1]))),
        )
        maximum_metric = max(
            maximum_metric,
            abs(float(q_star[0] - q_star[1])),
            abs(float(e_star[0] - e_star[1])),
            abs(float(v_max_star[0] - v_max_star[1])),
        )
    return maximum_receiver, maximum_metric


@dataclass(frozen=True, slots=True)
class _NominationCandidate:
    gate: SplitCohortHistoryBudgetGate
    kind: SplitCohortHistoryBudgetPairKind
    action_id: str | None
    carrier: float
    center: FloatArray
    amplitude: float
    coordinate_defect: float
    mode: FloatArray
    score: float


def _targeted_candidates(
    *,
    gate: SplitCohortHistoryBudgetGate,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    operator: DenseObserverOperator,
    stack: FloatArray,
    basis: FloatArray,
    inverse_lag: FloatArray,
    depth: int,
    homogeneous_by_action: dict[str, FloatArray],
    forced_by_action: dict[str, FloatArray],
    collision_epsilon: float,
) -> tuple[_NominationCandidate, ...]:
    if gate is SplitCohortHistoryBudgetGate.TARGET:
        center_profile = np.ones(descriptor.scale_cells, dtype=np.float64)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(operator.state_matrix)
        slow_index = int(np.argmax(eigenvalues.real))
        center_profile = np.asarray(eigenvectors[:, slow_index].real, dtype=np.float64)
        nonzero = np.flatnonzero(np.abs(center_profile) > 1e-14)
        if nonzero.size == 0:
            return ()
        if center_profile[int(nonzero[0])] < 0.0:
            center_profile = -center_profile
        if np.min(center_profile) <= 0.0:
            center_profile = np.abs(center_profile)
        center_profile /= float(np.max(center_profile))
    denominator = descriptor.scale_cells * float(descriptor.capacitance_bar_farads)
    candidates: list[_NominationCandidate] = []
    for action_id in sorted(forced_by_action):
        forced = forced_by_action[action_id]
        homogeneous_endpoint = homogeneous_by_action[action_id]
        homogeneous_profile = homogeneous_endpoint @ center_profile
        carrier = _gate_carrier(
            gate=gate,
            forced=forced,
            homogeneous_profile=homogeneous_profile,
            operator=operator,
            descriptor=descriptor,
            config=config,
        )
        if carrier is None:
            continue
        center = carrier * center_profile
        if gate is SplitCohortHistoryBudgetGate.TARGET:
            sensitivity = np.asarray(
                operator.capacitances @ homogeneous_endpoint / denominator,
                dtype=np.float64,
            )
        else:
            midpoint = forced + homogeneous_endpoint @ center
            active_index = int(np.argmax(midpoint))
            sensitivity = homogeneous_endpoint[active_index, :]
        projection = basis @ (basis.T @ sensitivity)
        if float(np.linalg.norm(projection)) <= 1e-13:
            continue
        mode = _orient_mode(projection, sensitivity)
        limit = _amplitude_limit(
            center=center,
            mode=mode,
            stack=stack,
            inverse_lag=inverse_lag,
            depth=depth,
            config=config,
            collision_epsilon=collision_epsilon,
        )
        if limit is None:
            continue
        amplitude, coordinate_defect = limit
        predictions = _predict_actions(
            config=config,
            descriptor=descriptor,
            operator=operator,
            homogeneous_by_action=homogeneous_by_action,
            forced_by_action=forced_by_action,
            center=center,
            amplitude=amplitude,
            mode=mode,
        )
        selected = next(value for value in predictions if value.action_id == action_id)
        if gate is SplitCohortHistoryBudgetGate.TARGET:
            opposite = selected.predicted_plus_target_pass != selected.predicted_minus_target_pass
            score = abs(float(selected.predicted_plus_q_star - selected.predicted_minus_q_star))
            kind = SplitCohortHistoryBudgetPairKind.TARGET_BOUNDARY
        else:
            opposite = selected.predicted_plus_sink_pass != selected.predicted_minus_sink_pass
            score = abs(
                float(selected.predicted_plus_v_max_star - selected.predicted_minus_v_max_star)
            )
            kind = SplitCohortHistoryBudgetPairKind.SINK_BOUNDARY
        future_receiver, future_metric = _future_defects(
            config=config,
            descriptor=descriptor,
            operator=operator,
            center=center,
            amplitude=amplitude,
            mode=mode,
        )
        if (
            not opposite
            or coordinate_defect > collision_epsilon * (1.0 + 1e-8)
            or max(future_receiver, future_metric) < float(config.future_divergence_epsilon)
        ):
            continue
        candidates.append(
            _NominationCandidate(
                gate=gate,
                kind=kind,
                action_id=action_id,
                carrier=carrier,
                center=center,
                amplitude=amplitude,
                coordinate_defect=coordinate_defect,
                mode=mode,
                score=score,
            )
        )
    maximum = (
        config.target_pairs_per_depth if gate is SplitCohortHistoryBudgetGate.TARGET else config.sink_pairs_per_depth
    )
    return tuple(
        sorted(candidates, key=lambda value: (-value.score, value.action_id or ""))[:maximum]
    )


def _random_candidates(
    *,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    stack: FloatArray,
    basis: FloatArray,
    inverse_lag: FloatArray,
    depth: int,
    scientific_input: HistoryBudgetDescriptorScientificInput,
    collision_epsilon: float,
) -> tuple[_NominationCandidate, ...]:
    scientific_seed = scientific_input.fibre_seed(depth)
    rng = np.random.Generator(
        np.random.PCG64(int.from_bytes(scientific_seed[:16], "big"))
    )
    values: list[_NominationCandidate] = []
    attempts = 0
    while len(values) < config.random_pairs_per_depth and attempts < 64:
        attempts += 1
        coefficients = rng.normal(size=basis.shape[1])
        raw = basis @ coefficients
        if float(np.linalg.norm(raw)) <= 1e-13:
            continue
        mode = _orient_mode(raw, np.zeros(raw.size, dtype=np.float64))
        center = np.full(descriptor.scale_cells, 0.5, dtype=np.float64)
        limit = _amplitude_limit(
            center=center,
            mode=mode,
            stack=stack,
            inverse_lag=inverse_lag,
            depth=depth,
            config=config,
            collision_epsilon=collision_epsilon,
        )
        if limit is None:
            continue
        amplitude, coordinate_defect = limit
        values.append(
            _NominationCandidate(
                gate=SplitCohortHistoryBudgetGate.RANDOM,
                kind=SplitCohortHistoryBudgetPairKind.RANDOM_FIBRE,
                action_id=None,
                carrier=0.5,
                center=center,
                amplitude=amplitude,
                coordinate_defect=coordinate_defect,
                mode=mode,
                score=float(len(values)),
            )
        )
    return tuple(values)


def observe_and_nominate(
    *,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    implementation_sha256: str,
    scientific_input: HistoryBudgetDescriptorScientificInput | None = None,
) -> ObserverExecution:
    """Compute the complete staircase and all predeclared probe nominations."""

    if descriptor.scale_cells not in {
        *config.scale_cells,
        *config.excluded_resource_canary_scale_cells,
    }:
        raise ValueError("split cohort history budget descriptor scale is outside the phase config")
    scientific_input = require_history_budget_descriptor_input(scientific_input, programme_ordinal=2, unit_id=descriptor.unit_id, scale_cells=descriptor.scale_cells, source_seed_sha256=descriptor.seed_sha256, current_descriptor_sha256=descriptor.fingerprint())
    if config.history_max_depth >= len(scientific_input.history_fibre_seed_sha256s):
        raise ValueError("history budget exceeds the complete original numeric seed census")
    operator = assemble_dense_operator(descriptor)
    forecast, arrays, stacks = _rank_forecast(config, descriptor, operator)
    inverse_lag = arrays["inverse-lag-propagator"]
    homogeneous_by_action, forced_by_action = _forced_endpoint_by_action(
        config, descriptor, operator
    )
    nominations: list[SplitCohortHistoryBudgetChallengeNomination] = []
    for coordinate in coordinate_labels(config, descriptor.scale_cells):
        depth = coordinate.depth
        collision_epsilon = float(coordinate.resolution_epsilon)
        stack = stacks[depth]
        _, singular_values, right_vectors = svd(stack, full_matrices=True)
        relative = singular_values / singular_values[0]
        effective_rank = int(np.sum(relative > float(config.rank_relative_threshold)))
        if effective_rank >= descriptor.scale_cells:
            continue
        basis = right_vectors[effective_rank:, :].T
        if basis.shape[1] == 0:
            continue
        candidates = (
            *_targeted_candidates(
                gate=SplitCohortHistoryBudgetGate.TARGET,
                config=config,
                descriptor=descriptor,
                operator=operator,
                stack=stack,
                basis=basis,
                inverse_lag=inverse_lag,
                depth=depth,
                homogeneous_by_action=homogeneous_by_action,
                forced_by_action=forced_by_action,
                collision_epsilon=collision_epsilon,
            ),
            *_targeted_candidates(
                gate=SplitCohortHistoryBudgetGate.SINK,
                config=config,
                descriptor=descriptor,
                operator=operator,
                stack=stack,
                basis=basis,
                inverse_lag=inverse_lag,
                depth=depth,
                homogeneous_by_action=homogeneous_by_action,
                forced_by_action=forced_by_action,
                collision_epsilon=collision_epsilon,
            ),
            *_random_candidates(
                config=config,
                descriptor=descriptor,
                stack=stack,
                basis=basis,
                inverse_lag=inverse_lag,
                depth=depth,
                scientific_input=scientific_input,
                collision_epsilon=collision_epsilon,
            ),
        )
        kind_counts: dict[SplitCohortHistoryBudgetPairKind, int] = {}
        for candidate in candidates:
            pair_index = kind_counts.get(candidate.kind, 0)
            kind_counts[candidate.kind] = pair_index + 1
            predictions = _predict_actions(
                config=config,
                descriptor=descriptor,
                operator=operator,
                homogeneous_by_action=homogeneous_by_action,
                forced_by_action=forced_by_action,
                center=candidate.center,
                amplitude=candidate.amplitude,
                mode=candidate.mode,
            )
            future_receiver, future_metric = _future_defects(
                config=config,
                descriptor=descriptor,
                operator=operator,
                center=candidate.center,
                amplitude=candidate.amplitude,
                mode=candidate.mode,
            )
            nomination = SplitCohortHistoryBudgetChallengeNomination(
                nomination_id=(
                    f"nomination.{descriptor.unit_id}.n{descriptor.scale_cells}"
                    f".{coordinate.coordinate_id}"
                    f".{candidate.kind.value.lower().replace('_', '-')}"
                    f".{pair_index:02d}"
                ),
                coordinate_id=coordinate.coordinate_id,
                cohort=SplitCohortHistoryBudgetCohort.TARGETED,
                unit_id=descriptor.unit_id,
                scale_cells=descriptor.scale_cells,
                depth=depth,
                resolution_epsilon=coordinate.resolution_epsilon,
                pair_index=pair_index,
                pair_kind=candidate.kind,
                gate=candidate.gate,
                targeting_action_id=candidate.action_id,
                carrier_volts=_decimal(candidate.carrier),
                center_values=tuple(_decimal(value) for value in candidate.center),
                amplitude_volts=_decimal(candidate.amplitude),
                mode_values=tuple(_decimal(value) for value in candidate.mode),
                predicted_coordinate_defect=_decimal(candidate.coordinate_defect),
                predicted_maximum_future_receiver_defect=_decimal(future_receiver),
                predicted_maximum_future_metric_defect=_decimal(future_metric),
                action_predictions=predictions,
                observer_implementation_sha256=implementation_sha256,
                outcome_count_at_nomination=0,
            )
            nominations.append(nomination)
    nominations_tuple = tuple(sorted(nominations, key=lambda value: value.nomination_id))
    arrays["nomination-modes"] = (
        np.vstack([np.asarray(value.mode_values, dtype=np.float64) for value in nominations_tuple])
        if nominations_tuple
        else np.empty((0, descriptor.scale_cells), dtype=np.float64)
    )
    return ObserverExecution(
        forecast=forecast,
        nominations=nominations_tuple,
        arrays=arrays,
    )


def observe_history(
    *,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
) -> HistoryExecution:
    """Compute only the pre-outcome rank staircase and numerical audit arrays."""

    if descriptor.scale_cells not in {
        *config.scale_cells,
        *config.excluded_resource_canary_scale_cells,
    }:
        raise ValueError("split cohort history budget descriptor scale is outside the phase config")
    operator = assemble_dense_operator(descriptor)
    forecast, arrays, _stacks = _rank_forecast(config, descriptor, operator)
    return HistoryExecution(forecast=forecast, arrays=arrays)


def nominate_from_history(
    *,
    config: SplitCohortHistoryBudgetConfig,
    descriptor: SplitCohortHistoryBudgetDenominatorDescriptor,
    history_forecast: SplitCohortHistoryBudgetHistoryRankForecast,
    implementation_sha256: str,
    scientific_input: HistoryBudgetDescriptorScientificInput | None = None,
) -> ObserverExecution:
    """Target fibres using only a frozen descriptor and history forecast."""

    execution = observe_and_nominate(
        config=config,
        descriptor=descriptor,
        implementation_sha256=implementation_sha256,
        scientific_input=scientific_input,
    )
    if execution.forecast != history_forecast:
        raise ValueError("split cohort history budget targeter recomputation differs from the frozen history")
    return execution


__all__ = [
    "DenseObserverOperator",
    "HistoryExecution",
    "ObserverExecution",
    "assemble_dense_operator",
    "coordinate_labels",
    "normalized_budget_depth",
    "nominate_from_history",
    "observe_and_nominate",
    "observe_history",
    "requested_depths",
    "sampled_history_matrix",
]

"""Support-bounded T-cohort optimization and certificates for history budget phase diagram."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm, svd
from scipy.optimize import linprog

from empirical_lawhood.adapters.history_budget_phase_diagram.contracts import HistoryBudgetPhaseDiagramChallengeNomination, HistoryBudgetPhaseDiagramCohort, HistoryBudgetPhaseDiagramConfig, HistoryBudgetPhaseDiagramCoordinateLabel, HistoryBudgetPhaseDiagramDenominatorDescriptor, HistoryBudgetPhaseDiagramGate, HistoryBudgetPhaseDiagramHistoryRankForecast, HistoryBudgetPhaseDiagramOptimizationCertificate, HistoryBudgetPhaseDiagramOptimizationState, HistoryBudgetPhaseDiagramPairKind

from .history import DenseObserverOperator, _forced_endpoint_by_action, _future_defects, _gate_carrier, _predict_actions, assemble_dense_operator, coordinate_labels, sampled_history_matrix


FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TargetingExecution:
    nominations: tuple[HistoryBudgetPhaseDiagramChallengeNomination, ...]
    certificates: tuple[HistoryBudgetPhaseDiagramOptimizationCertificate, ...]
    arrays: dict[str, FloatArray]


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("history budget phase diagram targeting value is nonfinite")
    return Decimal(str(float(value)))


def _history_polytope(
    config: HistoryBudgetPhaseDiagramConfig,
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    coordinate: HistoryBudgetPhaseDiagramCoordinateLabel,
    center: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    operator = assemble_dense_operator(descriptor)
    lag_seconds = float(config.history_lag_t_star) * float(descriptor.time_scale_seconds)
    inverse_lag = np.asarray(
        expm(-operator.state_matrix * lag_seconds),
        dtype=np.float64,
    )
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
    matrices: list[FloatArray] = []
    bounds: list[FloatArray] = []
    propagation = np.eye(descriptor.scale_cells, dtype=np.float64)
    for _depth in range(coordinate.depth + 1):
        historical_center = propagation @ center
        room = np.minimum(historical_center - lower, upper - historical_center)
        if np.min(room) <= 0.0:
            raise ValueError("history budget phase diagram targeting center has incomplete history support")
        matrices.extend((propagation, -propagation))
        bounds.extend((room, room))
        propagation = inverse_lag @ propagation
    history = sampled_history_matrix(config, descriptor, coordinate.depth)
    collision_half_bound = (float(coordinate.resolution_epsilon) - tolerance) / 2.0
    if collision_half_bound <= 0.0:
        raise ValueError("history budget phase diagram optimization tolerance exhausts collision support")
    collision_bound = np.full(history.shape[0], collision_half_bound, dtype=np.float64)
    matrices.extend((history, -history))
    bounds.extend((collision_bound, collision_bound))
    identity = np.eye(descriptor.scale_cells, dtype=np.float64)
    amplitude = np.full(
        descriptor.scale_cells,
        float(config.hidden_amplitude_max_volts) - tolerance,
        dtype=np.float64,
    )
    matrices.extend((identity, -identity))
    bounds.extend((amplitude, amplitude))
    return np.vstack(matrices), np.concatenate(bounds)


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
    best: tuple[FloatArray, float, float, float] | None = None
    for sign in (-1.0, 1.0):
        result = linprog(
            sign * objective,
            A_ub=constraints,
            b_ub=bounds,
            bounds=[(None, None)] * objective.size,
            method="highs",
        )
        if result.status == 2:
            continue
        if not result.success or result.x is None or result.fun is None:
            return None, 0.0, 0.0, 0.0, False
        direction = np.asarray(result.x, dtype=np.float64)
        constraint_values = constraints @ direction
        positive = constraint_values > 0.0
        feasible_scale = min(
            1.0,
            float(np.min(bounds[positive] / constraint_values[positive]))
            if np.any(positive)
            else 1.0,
        )
        feasible_direction = direction * max(0.0, feasible_scale)
        primal_residual = max(
            0.0,
            float(np.max(constraints @ feasible_direction - bounds)),
        )
        value = abs(float(objective @ feasible_direction)) * 2.0
        marginals = np.asarray(result.ineqlin.marginals, dtype=np.float64)
        dual_residual = float(
            np.linalg.norm(sign * objective - constraints.T @ marginals, ord=np.inf)
        )
        variable_l1_bound = float(objective.size) * float(np.max(bounds[-2 * objective.size :]))
        dual_upper = max(
            value,
            -2.0 * float(bounds @ marginals) + 2.0 * variable_l1_bound * dual_residual,
        )
        duality_gap = max(0.0, dual_upper - value)
        candidate = (
            feasible_direction,
            value,
            primal_residual,
            duality_gap,
        )
        if best is None or candidate[1] > best[1]:
            best = candidate
    if best is None:
        return None, 0.0, 0.0, 0.0, True
    return *best, True


def _classify_optimization_interval(
    *,
    lower: float,
    upper: float,
    margin: float,
    complete: bool,
    tolerance: float,
) -> HistoryBudgetPhaseDiagramOptimizationState:
    """Classify only intervals separated from the frozen margin by tolerance."""

    if not complete:
        return HistoryBudgetPhaseDiagramOptimizationState.RESOURCE_LIMITED
    if lower >= margin + tolerance:
        return HistoryBudgetPhaseDiagramOptimizationState.WITNESS
    if upper <= margin - tolerance:
        return HistoryBudgetPhaseDiagramOptimizationState.BELOW_MARGIN
    return HistoryBudgetPhaseDiagramOptimizationState.RESOURCE_LIMITED


def _center_and_objectives(
    *,
    gate: HistoryBudgetPhaseDiagramGate,
    config: HistoryBudgetPhaseDiagramConfig,
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
) -> tuple[tuple[str, FloatArray, FloatArray], ...]:
    operator = assemble_dense_operator(descriptor)
    homogeneous, forced = _forced_endpoint_by_action(config, descriptor, operator)
    if gate is HistoryBudgetPhaseDiagramGate.TARGET:
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
        if gate is HistoryBudgetPhaseDiagramGate.TARGET:
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
    coordinate: HistoryBudgetPhaseDiagramCoordinateLabel,
    gate: HistoryBudgetPhaseDiagramGate,
    action_id: str | None,
    state: HistoryBudgetPhaseDiagramOptimizationState,
    lower: float,
    upper: float,
    margin: Decimal,
    primal_residual: float,
    duality_gap: float,
) -> HistoryBudgetPhaseDiagramOptimizationCertificate:
    action_slug = "none" if action_id is None else action_id
    return HistoryBudgetPhaseDiagramOptimizationCertificate(
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
        complete_family_certificate=state is not HistoryBudgetPhaseDiagramOptimizationState.RESOURCE_LIMITED,
        outcome_count_at_certificate=0,
    )


def _freeze_witness(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    coordinate: HistoryBudgetPhaseDiagramCoordinateLabel,
    gate: HistoryBudgetPhaseDiagramGate,
    kind: HistoryBudgetPhaseDiagramPairKind,
    action_id: str | None,
    pair_index: int,
    center: FloatArray,
    direction: FloatArray,
    implementation_sha256: str,
    homogeneous: dict[str, FloatArray],
    forced: dict[str, FloatArray],
    operator: DenseObserverOperator,
) -> HistoryBudgetPhaseDiagramChallengeNomination:
    amplitude = float(np.max(np.abs(direction)))
    if amplitude <= 0.0:
        raise ValueError("history budget phase diagram targeter returned a zero witness")
    mode = direction / amplitude
    stack = sampled_history_matrix(config, descriptor, coordinate.depth)
    coordinate_defect = float(np.max(np.abs(stack @ direction))) * 2.0
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
    return HistoryBudgetPhaseDiagramChallengeNomination(
        nomination_id=(
            f"nomination.{descriptor.unit_id}.n{descriptor.scale_cells}"
            f".{coordinate.coordinate_id}.{kind.value.lower().replace('_', '-')}"
            f".{pair_index:02d}"
        ),
        coordinate_id=coordinate.coordinate_id,
        cohort=HistoryBudgetPhaseDiagramCohort.TARGETED,
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
    config: HistoryBudgetPhaseDiagramConfig,
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    history_forecast: HistoryBudgetPhaseDiagramHistoryRankForecast,
    implementation_sha256: str,
) -> TargetingExecution:
    """Solve all frozen T support problems and retain witnesses/certificates."""

    if history_forecast.descriptor_sha256 != descriptor.fingerprint():
        raise ValueError("history budget phase diagram targeter forecast/descriptor identity differs")
    operator = assemble_dense_operator(descriptor)
    homogeneous, forced = _forced_endpoint_by_action(config, descriptor, operator)
    nominations: list[HistoryBudgetPhaseDiagramChallengeNomination] = []
    certificates: list[HistoryBudgetPhaseDiagramOptimizationCertificate] = []
    modes: list[FloatArray] = []
    for coordinate in coordinate_labels(config, descriptor.scale_cells):
        optimization_tolerance = float(config.optimization_tolerance)
        # The dynamical certificate covers every declared R8 component and
        # diagnostic time.  The single retained witness is the lexical
        # maximum; the certificate, rather than that witness, carries the
        # complete-family nonadversity statement.
        dynamic_center = np.full(descriptor.scale_cells, 0.5, dtype=np.float64)
        dynamic_solutions: list[tuple[float, str, FloatArray, float, float, bool]] = []
        try:
            dynamic_constraints, dynamic_bounds = _history_polytope(
                config, descriptor, coordinate, dynamic_center
            )
        except ValueError:
            dynamic_constraints = np.empty((0, descriptor.scale_cells), dtype=np.float64)
            dynamic_bounds = np.empty(0, dtype=np.float64)
        if dynamic_constraints.size:
            for time_index, time_t_star in enumerate(config.diagnostic_times_t_star):
                propagation = np.asarray(
                    expm(
                        operator.state_matrix
                        * float(time_t_star)
                        * float(descriptor.time_scale_seconds)
                    ),
                    dtype=np.float64,
                )
                for receiver_index, receiver_row in enumerate(operator.receiver):
                    direction, value, primal, gap, complete = _solve_direction(
                        np.asarray(receiver_row @ propagation, dtype=np.float64),
                        dynamic_constraints,
                        dynamic_bounds,
                    )
                    if direction is not None:
                        dynamic_solutions.append(
                            (
                                value,
                                f"t{time_index:02d}.r{receiver_index:02d}",
                                direction,
                                primal,
                                gap,
                                complete,
                            )
                        )
                    elif not complete:
                        dynamic_solutions.append(
                            (
                                0.0,
                                f"t{time_index:02d}.r{receiver_index:02d}",
                                np.zeros(descriptor.scale_cells),
                                0.0,
                                0.0,
                                False,
                            )
                        )
        dynamic_solutions.sort(key=lambda value: (-value[0], value[1]))
        dynamic_maximum = dynamic_solutions[0] if dynamic_solutions else None
        if dynamic_maximum is None:
            dynamic_state = HistoryBudgetPhaseDiagramOptimizationState.INFEASIBLE
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
                coordinate=coordinate,
                gate=HistoryBudgetPhaseDiagramGate.DYNAMICAL,
                action_id=None,
                state=dynamic_state,
                lower=dynamic_lower,
                upper=dynamic_upper,
                margin=config.future_divergence_epsilon,
                primal_residual=dynamic_primal,
                duality_gap=dynamic_gap,
            )
        )
        if dynamic_state is HistoryBudgetPhaseDiagramOptimizationState.WITNESS and dynamic_maximum is not None:
            dynamic_nomination = _freeze_witness(
                config=config,
                descriptor=descriptor,
                coordinate=coordinate,
                gate=HistoryBudgetPhaseDiagramGate.DYNAMICAL,
                kind=HistoryBudgetPhaseDiagramPairKind.DYNAMICAL_BOUNDARY,
                action_id=None,
                pair_index=0,
                center=dynamic_center,
                direction=dynamic_direction,
                implementation_sha256=implementation_sha256,
                homogeneous=homogeneous,
                forced=forced,
                operator=operator,
            )
            nominations.append(dynamic_nomination)
            modes.append(dynamic_direction)

        for gate, kind, margin in (
            (HistoryBudgetPhaseDiagramGate.TARGET, HistoryBudgetPhaseDiagramPairKind.TARGET_BOUNDARY, config.target_decision_margin),
            (HistoryBudgetPhaseDiagramGate.SINK, HistoryBudgetPhaseDiagramPairKind.SINK_BOUNDARY, config.sink_decision_margin),
        ):
            objective_rows = {
                action_id: (center, objective)
                for action_id, center, objective in _center_and_objectives(
                    gate=gate,
                    config=config,
                    descriptor=descriptor,
                )
            }
            witnesses: list[tuple[float, str, FloatArray, FloatArray]] = []
            for action_id in sorted(forced):
                row = objective_rows.get(action_id)
                if row is None:
                    certificates.append(
                        _certificate(
                            coordinate=coordinate,
                            gate=gate,
                            action_id=action_id,
                            state=HistoryBudgetPhaseDiagramOptimizationState.INFEASIBLE,
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
                    )
                except ValueError:
                    certificates.append(
                        _certificate(
                            coordinate=coordinate,
                            gate=gate,
                            action_id=action_id,
                            state=HistoryBudgetPhaseDiagramOptimizationState.INFEASIBLE,
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
                if gate is HistoryBudgetPhaseDiagramGate.SINK:
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
                if direction is not None and state is HistoryBudgetPhaseDiagramOptimizationState.WITNESS:
                    witnesses.append((value, action_id, center, direction))
            witnesses.sort(key=lambda value: (-value[0], value[1]))
            limit = (
                config.target_pairs_per_depth
                if gate is HistoryBudgetPhaseDiagramGate.TARGET
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
                )
                nominations.append(nomination)
                modes.append(direction)
        # Four deterministic random-fibre objectives remain comparator witnesses.
        stack = sampled_history_matrix(config, descriptor, coordinate.depth)
        _u, singular_values, right_vectors = svd(stack, full_matrices=True)
        relative = singular_values / singular_values[0]
        rank = int(np.sum(relative > float(config.rank_relative_threshold)))
        basis = right_vectors[rank:, :]
        try:
            random_constraints, random_bounds = _history_polytope(
                config,
                descriptor,
                coordinate,
                np.full(descriptor.scale_cells, 0.5, dtype=np.float64),
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
            nomination = HistoryBudgetPhaseDiagramChallengeNomination(
                nomination_id=(
                    f"nomination.{descriptor.unit_id}.n{descriptor.scale_cells}"
                    f".{coordinate.coordinate_id}.random-fibre.{pair_index:02d}"
                ),
                coordinate_id=coordinate.coordinate_id,
                cohort=HistoryBudgetPhaseDiagramCohort.TARGETED,
                unit_id=descriptor.unit_id,
                scale_cells=descriptor.scale_cells,
                depth=coordinate.depth,
                resolution_epsilon=coordinate.resolution_epsilon,
                pair_index=pair_index,
                pair_kind=HistoryBudgetPhaseDiagramPairKind.RANDOM_FIBRE,
                gate=HistoryBudgetPhaseDiagramGate.RANDOM,
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
    return TargetingExecution(nominations_tuple, certificates_tuple, arrays)


__all__ = ["TargetingExecution", "nominate_targeted_challenges"]

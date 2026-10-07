"""Reveal-lane unit adjudication for history budget phase diagram."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

import numpy as np

from empirical_lawhood.adapters.history_budget_phase_diagram.contracts import HistoryBudgetPhaseDiagramActionPrediction, HistoryBudgetPhaseDiagramChallengeNomination, HistoryBudgetPhaseDiagramConfig, HistoryBudgetPhaseDiagramCoordinateLabel, HistoryBudgetPhaseDiagramDecisionDisposition, HistoryBudgetPhaseDiagramDenominatorDescriptor, HistoryBudgetPhaseDiagramDepthAdjudication, HistoryBudgetPhaseDiagramGeneratorActionOutcome, HistoryBudgetPhaseDiagramGeneratorOutcome, HistoryBudgetPhaseDiagramHistoryRankForecast, HistoryBudgetPhaseDiagramGate, HistoryBudgetPhaseDiagramMethodFreeze, HistoryBudgetPhaseDiagramOptimizationCertificate, HistoryBudgetPhaseDiagramOptimizationState, HistoryBudgetPhaseDiagramPairAdjudication, HistoryBudgetPhaseDiagramScientificState, HistoryBudgetPhaseDiagramTargetedCoordinateAdjudication, HistoryBudgetPhaseDiagramUntouchedCoordinateAdjudication, HistoryBudgetPhaseDiagramUntouchedGeneratorOutcome, HistoryBudgetPhaseDiagramUnitAdjudication
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import canonical_json_bytes


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("history budget phase diagram evaluator output must be finite")
    return Decimal(str(float(value)))


def _nomination_set_digest(
    nominations: tuple[HistoryBudgetPhaseDiagramChallengeNomination, ...],
) -> str:
    return sha256(
        canonical_json_bytes(tuple(value.fingerprint() for value in nominations))
    ).hexdigest()


def _action_prediction_agrees(
    predicted: HistoryBudgetPhaseDiagramActionPrediction,
    realized: HistoryBudgetPhaseDiagramGeneratorActionOutcome,
    *,
    material_floor: float,
    metric_tolerance: float,
) -> bool:
    categorical = (
        predicted.predicted_plus_target_pass == realized.plus_target_pass
        and predicted.predicted_minus_target_pass == realized.minus_target_pass
        and predicted.predicted_midpoint_target_pass == realized.midpoint_target_pass
        and predicted.predicted_plus_sink_pass == realized.plus_sink_pass
        and predicted.predicted_minus_sink_pass == realized.minus_sink_pass
        and predicted.predicted_midpoint_sink_pass == realized.midpoint_sink_pass
        and predicted.predicted_plus_admit == realized.plus_admit
        and predicted.predicted_minus_admit == realized.minus_admit
        and predicted.predicted_midpoint_admit == realized.midpoint_admit
    )
    if not categorical or not realized.requested_accepted_applied_realized_parity:
        return False
    raw_pairs = (
        (predicted.predicted_plus_q_star, realized.plus_q_star),
        (predicted.predicted_minus_q_star, realized.minus_q_star),
        (predicted.predicted_plus_e_star, realized.plus_e_star),
        (predicted.predicted_minus_e_star, realized.minus_e_star),
        (predicted.predicted_plus_v_max_star, realized.plus_v_max_star),
        (predicted.predicted_minus_v_max_star, realized.minus_v_max_star),
        (predicted.predicted_midpoint_q_star, realized.midpoint_q_star),
        (predicted.predicted_midpoint_e_star, realized.midpoint_e_star),
        (predicted.predicted_midpoint_v_max_star, realized.midpoint_v_max_star),
    )
    if max(abs(float(left - right)) for left, right in raw_pairs) > metric_tolerance:
        return False
    pairs = (
        (
            float(predicted.predicted_plus_q_star - predicted.predicted_minus_q_star),
            float(realized.plus_q_star - realized.minus_q_star),
        ),
        (
            float(predicted.predicted_plus_e_star - predicted.predicted_minus_e_star),
            float(realized.plus_e_star - realized.minus_e_star),
        ),
        (
            float(predicted.predicted_plus_v_max_star - predicted.predicted_minus_v_max_star),
            float(realized.plus_v_max_star - realized.minus_v_max_star),
        ),
    )
    for predicted_delta, realized_delta in pairs:
        predicted_material = abs(predicted_delta) >= material_floor
        realized_material = abs(realized_delta) >= material_floor
        if predicted_material != realized_material:
            return False
        if predicted_material and np.sign(predicted_delta) != np.sign(realized_delta):
            return False
    return True


def _midpoint_decision_errors(
    realized: HistoryBudgetPhaseDiagramGeneratorActionOutcome,
    config: HistoryBudgetPhaseDiagramConfig,
) -> tuple[bool, bool]:
    """Classify robust hidden-member errors against the frozen midpoint decision."""

    target_threshold = float(config.target_charge_minimum_q_star)
    sink_threshold = float(config.sink_voltage_maximum_v_star)
    target_half_margin = float(config.target_decision_margin) / 2.0
    sink_half_margin = float(config.sink_decision_margin) / 2.0

    def robust_admit(q_value: Decimal, v_value: Decimal) -> bool | None:
        q = float(q_value)
        v = float(v_value)
        if q >= target_threshold + target_half_margin and v <= sink_threshold - sink_half_margin:
            return True
        if q <= target_threshold - target_half_margin or v >= sink_threshold + sink_half_margin:
            return False
        return None

    hidden = (
        robust_admit(realized.plus_q_star, realized.plus_v_max_star),
        robust_admit(realized.minus_q_star, realized.minus_v_max_star),
    )
    false_safe = realized.midpoint_admit and any(value is False for value in hidden)
    false_hold = not realized.midpoint_admit and any(value is True for value in hidden)
    return false_safe, false_hold


def adjudicate_untouched_scale(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    coordinates: tuple[HistoryBudgetPhaseDiagramCoordinateLabel, ...],
    collision_edges: np.ndarray[tuple[int, int], np.dtype[np.int64]],
    receiver_history: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    action_metrics: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    action_states: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    generator_outcome: HistoryBudgetPhaseDiagramUntouchedGeneratorOutcome,
) -> tuple[HistoryBudgetPhaseDiagramUntouchedCoordinateAdjudication, ...]:
    """Adjudicate the frozen U collision graphs without pair-level inference."""

    count = config.untouched_preparation_count
    action_count = len(config.action_amplitudes_u_star) * len(config.action_durations_t_star)
    if (
        generator_outcome.descriptor_sha256 != descriptor.fingerprint()
        or generator_outcome.preparation_count != count
        or receiver_history.shape != (count, config.history_max_depth + 1, 8)
        or action_metrics.shape != (count, action_count, 3)
        or action_states.shape != (count, action_count, descriptor.scale_cells)
        or collision_edges.ndim != 2
        or collision_edges.shape[1:] != (3,)
    ):
        raise ValueError("history budget phase diagram untouched evaluator input roster differs")
    if tuple(value.coordinate_id for value in coordinates) != tuple(
        sorted({value.coordinate_id for value in coordinates})
    ) or any(value.scale_cells != descriptor.scale_cells for value in coordinates):
        raise ValueError("history budget phase diagram untouched coordinate roster differs")
    if not all(
        np.all(np.isfinite(value)) for value in (receiver_history, action_metrics, action_states)
    ):
        raise ValueError("history budget phase diagram untouched evaluator received nonfinite arrays")
    if collision_edges.size and (
        np.min(collision_edges) < 0
        or np.max(collision_edges[:, 0]) >= len(coordinates)
        or np.max(collision_edges[:, 1:]) >= count
        or np.any(collision_edges[:, 1] >= collision_edges[:, 2])
    ):
        raise ValueError("history budget phase diagram untouched collision graph is invalid")

    capacitances = np.asarray(descriptor.capacitances_farads, dtype=np.float64)
    receiver = _receiver(descriptor)
    metric_denominator = descriptor.scale_cells * float(descriptor.capacitance_bar_farads)
    target = float(config.target_charge_minimum_q_star)
    sink = float(config.sink_voltage_maximum_v_star)
    target_half_margin = float(config.target_decision_margin) / 2.0
    sink_half_margin = float(config.sink_decision_margin) / 2.0

    def robust_admit(q_value: float, v_value: float) -> bool | None:
        if q_value >= target + target_half_margin and v_value <= sink - sink_half_margin:
            return True
        if q_value <= target - target_half_margin or v_value >= sink + sink_half_margin:
            return False
        return None

    rows: list[HistoryBudgetPhaseDiagramUntouchedCoordinateAdjudication] = []
    for coordinate_index, coordinate in enumerate(coordinates):
        edges = collision_edges[collision_edges[:, 0] == coordinate_index, 1:]
        prefix_counts = tuple(
            int(np.sum((edges[:, 0] < prefix) & (edges[:, 1] < prefix)))
            for prefix in (*config.untouched_prefix_counts, count)
        )
        dynamic_count = 0
        decision_count = 0
        any_adverse_count = 0
        unsafe_count = 0
        false_hold_count = 0
        maximum_receiver = 0.0
        maximum_metric = 0.0
        for left_raw, right_raw in edges:
            left = int(left_raw)
            right = int(right_raw)
            realized_collision = float(
                np.max(
                    np.abs(
                        receiver_history[left, : coordinate.depth + 1]
                        - receiver_history[right, : coordinate.depth + 1]
                    )
                )
            )
            if realized_collision > float(coordinate.resolution_epsilon) * (1.0 + 1e-10):
                raise ValueError("history budget phase diagram frozen untouched edge is not a realized collision")
            receiver_left = np.einsum("ai,ri->ar", action_states[left], receiver)
            receiver_right = np.einsum("ai,ri->ar", action_states[right], receiver)
            receiver_defect = float(np.max(np.abs(receiver_left - receiver_right)))
            metric_defect = float(np.max(np.abs(action_metrics[left] - action_metrics[right])))
            maximum_receiver = max(maximum_receiver, receiver_defect)
            maximum_metric = max(maximum_metric, metric_defect)
            edge_dynamic = receiver_defect >= float(config.future_divergence_epsilon)
            edge_unsafe = False
            edge_false_hold = False
            for action_index in range(action_count):
                midpoint_state = (
                    action_states[left, action_index] + action_states[right, action_index]
                ) / 2.0
                midpoint_q = float(capacitances @ midpoint_state / metric_denominator)
                midpoint_v = float(np.max(midpoint_state))
                midpoint_admit = midpoint_q >= target and midpoint_v <= sink
                hidden = (
                    robust_admit(
                        float(action_metrics[left, action_index, 0]),
                        float(action_metrics[left, action_index, 2]),
                    ),
                    robust_admit(
                        float(action_metrics[right, action_index, 0]),
                        float(action_metrics[right, action_index, 2]),
                    ),
                )
                edge_unsafe = edge_unsafe or (
                    midpoint_admit and any(value is False for value in hidden)
                )
                edge_false_hold = edge_false_hold or (
                    not midpoint_admit and any(value is True for value in hidden)
                )
            dynamic_count += int(edge_dynamic)
            unsafe_count += int(edge_unsafe)
            false_hold_count += int(edge_false_hold)
            decision_count += int(edge_unsafe or edge_false_hold)
            any_adverse_count += int(edge_dynamic or edge_unsafe or edge_false_hold)
        state = (
            HistoryBudgetPhaseDiagramScientificState.UNSAFE_FALSE_PROMOTION
            if unsafe_count
            else HistoryBudgetPhaseDiagramScientificState.OPPOSED
            if dynamic_count or decision_count
            else HistoryBudgetPhaseDiagramScientificState.INFORMATIVE_NONADVERSE
            if len(edges)
            else HistoryBudgetPhaseDiagramScientificState.NO_COLLISION_ENCOUNTER_WITH_512_PREPARATIONS
        )
        rows.append(
            HistoryBudgetPhaseDiagramUntouchedCoordinateAdjudication(
                adjudication_id=(
                    f"untouched-adjudication.{descriptor.unit_id}.{coordinate.coordinate_id}"
                ),
                coordinate_id=coordinate.coordinate_id,
                unit_id=descriptor.unit_id,
                family=descriptor.family,
                scale_cells=descriptor.scale_cells,
                depth=coordinate.depth,
                resolution_epsilon=coordinate.resolution_epsilon,
                preparation_count=count,
                valid=True,
                collision_edge_count=len(edges),
                prefix_edge_counts=prefix_counts,
                dynamic_adverse_edge_count=dynamic_count,
                decision_adverse_edge_count=decision_count,
                any_adverse_edge_count=any_adverse_count,
                unsafe_false_promotion_edge_count=unsafe_count,
                false_hold_edge_count=false_hold_count,
                maximum_future_receiver_defect=_decimal(maximum_receiver),
                maximum_future_metric_defect=_decimal(maximum_metric),
                scientific_state=state,
                unsafe_first=True,
            )
        )
    return tuple(sorted(rows, key=lambda value: value.adjudication_id))


def _receiver(descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor) -> np.ndarray:
    receiver = np.zeros((8, descriptor.scale_cells), dtype=np.float64)
    capacitances = np.asarray(descriptor.capacitances_farads, dtype=np.float64)
    width = descriptor.scale_cells // 8
    for bin_index in range(8):
        start = width * bin_index
        stop = start + width
        receiver[bin_index, start:stop] = capacitances[start:stop] / np.sum(
            capacitances[start:stop]
        )
    return receiver


def adjudicate_unit_scale(
    *,
    config: HistoryBudgetPhaseDiagramConfig,
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    forecast: HistoryBudgetPhaseDiagramHistoryRankForecast,
    nominations: tuple[HistoryBudgetPhaseDiagramChallengeNomination, ...],
    generator_outcome: HistoryBudgetPhaseDiagramGeneratorOutcome,
    method_freeze: HistoryBudgetPhaseDiagramMethodFreeze | None,
) -> HistoryBudgetPhaseDiagramUnitAdjudication:
    """Reveal and adjudicate exactly one complete-unit finite scale view."""

    if generator_outcome.outcome_access not in {
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        OutcomeAccess.EVALUATION_SEALED,
    }:
        raise ValueError("history budget phase diagram evaluator received an invalid outcome-access state")
    if config.phase.value == "EVALUATION" and method_freeze is None:
        raise ValueError("history budget phase diagram evaluation adjudication requires the exact method freeze")
    if method_freeze is not None:
        if (
            config.phase.value != "EVALUATION"
            or generator_outcome.generator_implementation_sha256
            != method_freeze.generator_implementation_sha256
            or any(
                value.observer_implementation_sha256 != method_freeze.observer_implementation_sha256
                for value in nominations
            )
        ):
            raise ValueError("history budget phase diagram evaluation inputs differ from the method freeze")
    if descriptor.fingerprint() != generator_outcome.descriptor_sha256:
        raise ValueError("history budget phase diagram generator outcome binds a different descriptor")
    if forecast.descriptor_sha256 != descriptor.fingerprint():
        raise ValueError("history budget phase diagram forecast binds a different descriptor")
    expected_nomination_digest = _nomination_set_digest(nominations)
    if generator_outcome.nomination_set_sha256 != expected_nomination_digest:
        raise ValueError("history budget phase diagram generator outcome binds a different nomination set")
    if tuple(value.nomination_id for value in nominations) != tuple(
        sorted({value.nomination_id for value in nominations})
    ):
        raise ValueError("history budget phase diagram evaluator nominations must be sorted and unique")
    outcome_by_id = {value.nomination_id: value for value in generator_outcome.pair_outcomes}
    if set(outcome_by_id) != {value.nomination_id for value in nominations}:
        raise ValueError("history budget phase diagram generator outcome pair roster differs")
    divergence_epsilon = float(config.future_divergence_epsilon)
    pair_rows: list[HistoryBudgetPhaseDiagramPairAdjudication] = []
    expected_actions = {
        f"action.a{amplitude_index:02d}.d{duration_index:02d}": (
            amplitude,
            duration,
        )
        for amplitude_index, amplitude in enumerate(config.action_amplitudes_u_star)
        for duration_index, duration in enumerate(config.action_durations_t_star)
    }
    for nomination in nominations:
        outcome = outcome_by_id[nomination.nomination_id]
        collision_epsilon = float(nomination.resolution_epsilon)
        collision = (
            outcome.voltage_envelope_valid
            and float(outcome.realized_coordinate_defect) <= collision_epsilon
        )
        target_mismatch = 0
        sink_mismatch = 0
        decision_mismatch = 0
        false_safe = 0
        false_hold = 0
        action_agreement = True
        predicted_by_action = {value.action_id: value for value in nomination.action_predictions}
        if set(predicted_by_action) != set(expected_actions):
            raise ValueError("history budget phase diagram nomination action roster differs from the config")
        for action_id, predicted in predicted_by_action.items():
            if (predicted.amplitude_u_star, predicted.duration_t_star) != expected_actions[
                action_id
            ] or predicted.requested_action_id != action_id:
                raise ValueError("history budget phase diagram nomination action semantics differ")
        if {value.action_id for value in outcome.action_outcomes} != set(expected_actions):
            raise ValueError("history budget phase diagram generator action roster differs from the config")
        for realized in outcome.action_outcomes:
            predicted = predicted_by_action[realized.action_id]
            if (
                (realized.amplitude_u_star, realized.duration_t_star)
                != expected_actions[realized.action_id]
                or not realized.requested_accepted_applied_realized_parity
                or realized.requested_action_id != realized.action_id
            ):
                raise ValueError("history budget phase diagram realized action semantics differ")
            target_mismatch += int(realized.plus_target_pass != realized.minus_target_pass)
            sink_mismatch += int(realized.plus_sink_pass != realized.minus_sink_pass)
            action_false_safe, action_false_hold = _midpoint_decision_errors(realized, config)
            decision_mismatch += int(action_false_safe or action_false_hold)
            false_safe += int(action_false_safe)
            false_hold += int(action_false_hold)
            action_agreement = action_agreement and _action_prediction_agrees(
                predicted,
                realized,
                material_floor=divergence_epsilon,
                metric_tolerance=float(config.generator_observer_metric_tolerance),
            )
        dynamic_adverse = (
            collision and float(outcome.maximum_future_receiver_defect) >= divergence_epsilon
        )
        decision_adverse = collision and decision_mismatch > 0
        prediction_collision = float(nomination.predicted_coordinate_defect) <= collision_epsilon
        prediction_material = (
            float(nomination.predicted_maximum_future_receiver_defect) >= divergence_epsilon
        )
        realized_material = float(outcome.maximum_future_receiver_defect) >= divergence_epsilon
        pair_rows.append(
            HistoryBudgetPhaseDiagramPairAdjudication(
                nomination_id=nomination.nomination_id,
                coordinate_id=nomination.coordinate_id,
                depth=nomination.depth,
                resolution_epsilon=nomination.resolution_epsilon,
                gate=nomination.gate,
                realized_coordinate_defect=outcome.realized_coordinate_defect,
                maximum_future_receiver_defect=outcome.maximum_future_receiver_defect,
                maximum_future_metric_defect=outcome.maximum_future_metric_defect,
                target_mismatch_count=target_mismatch,
                sink_mismatch_count=sink_mismatch,
                decision_mismatch_count=decision_mismatch,
                false_safe_count=false_safe,
                unsafe_false_promotion_count=false_safe,
                false_hold_count=false_hold,
                collision_confirmed=collision,
                dynamically_adverse=dynamic_adverse,
                decision_adverse=decision_adverse,
                decision_disposition=(
                    HistoryBudgetPhaseDiagramDecisionDisposition.UNSAFE_FALSE_PROMOTION
                    if false_safe
                    else (
                        HistoryBudgetPhaseDiagramDecisionDisposition.FALSE_HOLD
                        if false_hold
                        else HistoryBudgetPhaseDiagramDecisionDisposition.CORRECT_ACTION
                    )
                ),
                prediction_agreement=(
                    action_agreement
                    and prediction_collision == collision
                    and prediction_material == realized_material
                ),
            )
        )
    pair_rows_tuple = tuple(sorted(pair_rows, key=lambda value: value.nomination_id))
    depth_rows: list[HistoryBudgetPhaseDiagramDepthAdjudication] = []
    for depth in forecast.probe_depths:
        values = tuple(value for value in pair_rows_tuple if value.depth == depth)
        collisions = sum(value.collision_confirmed for value in values)
        dynamic_adverse_count = sum(value.dynamically_adverse for value in values)
        decision_adverse_count = sum(value.decision_adverse for value in values)
        false_safe_count = sum(value.false_safe_count for value in values)
        dynamic_state = (
            HistoryBudgetPhaseDiagramScientificState.OPPOSED
            if dynamic_adverse_count
            else (
                HistoryBudgetPhaseDiagramScientificState.SUPPORTED
                if collisions
                else HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
            )
        )
        decision_state = (
            HistoryBudgetPhaseDiagramScientificState.OPPOSED
            if decision_adverse_count
            else (
                HistoryBudgetPhaseDiagramScientificState.SUPPORTED
                if collisions
                else HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
            )
        )
        depth_rows.append(
            HistoryBudgetPhaseDiagramDepthAdjudication(
                depth=depth,
                requested_pair_count=len(values),
                realized_collision_count=collisions,
                dynamic_adverse_count=dynamic_adverse_count,
                decision_adverse_count=decision_adverse_count,
                false_safe_count=false_safe_count,
                dynamical_state=dynamic_state,
                decision_state=decision_state,
            )
        )
    collision_count = sum(value.collision_confirmed for value in pair_rows_tuple)
    dynamic_count = sum(value.dynamically_adverse for value in pair_rows_tuple)
    decision_count = sum(value.decision_adverse for value in pair_rows_tuple)
    dynamical_state = (
        HistoryBudgetPhaseDiagramScientificState.OPPOSED
        if dynamic_count
        else (
            HistoryBudgetPhaseDiagramScientificState.SUPPORTED
            if collision_count
            else HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
        )
    )
    decision_state = (
        HistoryBudgetPhaseDiagramScientificState.OPPOSED
        if decision_count
        else (
            HistoryBudgetPhaseDiagramScientificState.SUPPORTED
            if collision_count
            else HistoryBudgetPhaseDiagramScientificState.UNEVALUABLE
        )
    )
    k_full = forecast.k_full_effective
    b_full = (
        None if k_full is None else _decimal(min(1.0, 8.0 * (k_full + 1) / descriptor.scale_cells))
    )
    rho = np.asarray(
        [value.effective_rank / descriptor.scale_cells for value in forecast.rank_steps],
        dtype=np.float64,
    )
    budget = np.asarray(
        [
            min(1.0, 8.0 * (value.depth + 1) / descriptor.scale_cells)
            for value in forecast.rank_steps
        ],
        dtype=np.float64,
    )
    agreement = all(value.prediction_agreement for value in pair_rows_tuple)
    reasons: set[str] = set()
    if forecast.effective_rank_right_censored:
        reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_EFFECTIVE_RANK_RIGHT_CENSORED")
    if forecast.algebraic_prediction_state is HistoryBudgetPhaseDiagramScientificState.OPPOSED:
        reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_ALGEBRAIC_RANK_PREDICTION_OPPOSED")
    if not agreement:
        reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_GENERATOR_OBSERVER_DISAGREEMENT")
    if not pair_rows_tuple:
        reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_UNTARGETABLE_UNEVALUABLE")
    if sum(value.false_safe_count for value in pair_rows_tuple):
        reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_FALSE_SAFE_COUNTEREXAMPLE")
    return HistoryBudgetPhaseDiagramUnitAdjudication(
        adjudication_id=f"adjudication.{descriptor.unit_id}.n{descriptor.scale_cells}",
        unit_id=descriptor.unit_id,
        family=descriptor.family,
        scale_cells=descriptor.scale_cells,
        descriptor_valid=True,
        nominated_pair_count=len(nominations),
        realized_collision_count=collision_count,
        evaluable_pair_count=collision_count,
        pair_adjudications=pair_rows_tuple,
        depth_adjudications=tuple(depth_rows),
        dynamical_state=dynamical_state,
        decision_state=decision_state,
        target_mismatch_count=sum(value.target_mismatch_count for value in pair_rows_tuple),
        sink_mismatch_count=sum(value.sink_mismatch_count for value in pair_rows_tuple),
        false_safe_count=sum(value.false_safe_count for value in pair_rows_tuple),
        false_hold_count=sum(value.false_hold_count for value in pair_rows_tuple),
        k_full_effective=k_full,
        b_full=b_full,
        rank_curve_distance=_decimal(float(np.sqrt(np.mean((rho - budget) ** 2)))),
        generator_observer_agreement=agreement,
        reason_codes=tuple(sorted(reasons)),
    )


def adjudicate_targeted_coordinates(
    *,
    descriptor: HistoryBudgetPhaseDiagramDenominatorDescriptor,
    coordinates: tuple[HistoryBudgetPhaseDiagramCoordinateLabel, ...],
    unit_adjudication: HistoryBudgetPhaseDiagramUnitAdjudication,
    certificates: tuple[HistoryBudgetPhaseDiagramOptimizationCertificate, ...],
    expected_action_ids: tuple[str, ...],
) -> tuple[HistoryBudgetPhaseDiagramTargetedCoordinateAdjudication, ...]:
    """Derive the four-state T ledger without treating absent witnesses as support."""

    if unit_adjudication.unit_id != descriptor.unit_id:
        raise ValueError("history budget phase diagram targeted coordinate ledger mixes independent units")
    rows: list[HistoryBudgetPhaseDiagramTargetedCoordinateAdjudication] = []
    for coordinate in coordinates:
        pairs = tuple(
            value
            for value in unit_adjudication.pair_adjudications
            if value.coordinate_id == coordinate.coordinate_id
        )
        coordinate_certificates = tuple(
            value for value in certificates if value.coordinate_id == coordinate.coordinate_id
        )
        dynamic_certificates = tuple(
            value for value in coordinate_certificates if value.gate is HistoryBudgetPhaseDiagramGate.DYNAMICAL
        )
        dynamical_complete = len(dynamic_certificates) == 1 and all(
            value.complete_family_certificate
            and value.state
            in {HistoryBudgetPhaseDiagramOptimizationState.BELOW_MARGIN, HistoryBudgetPhaseDiagramOptimizationState.INFEASIBLE}
            for value in dynamic_certificates
        )
        decision_certificates = tuple(
            value
            for value in coordinate_certificates
            if value.gate in {HistoryBudgetPhaseDiagramGate.TARGET, HistoryBudgetPhaseDiagramGate.SINK}
        )
        decision_keys = {(value.gate, value.action_id) for value in decision_certificates}
        expected_keys = {
            (gate, action_id)
            for gate in (HistoryBudgetPhaseDiagramGate.TARGET, HistoryBudgetPhaseDiagramGate.SINK)
            for action_id in expected_action_ids
        }
        decision_complete = decision_keys == expected_keys and all(
            value.complete_family_certificate
            and value.state
            in {HistoryBudgetPhaseDiagramOptimizationState.BELOW_MARGIN, HistoryBudgetPhaseDiagramOptimizationState.INFEASIBLE}
            for value in decision_certificates
        )
        realized = sum(value.collision_confirmed for value in pairs)
        dynamic = sum(value.dynamically_adverse for value in pairs)
        decision = sum(value.decision_adverse for value in pairs)
        unsafe = sum(
            value.collision_confirmed and value.unsafe_false_promotion_count > 0 for value in pairs
        )
        false_hold = sum(
            value.collision_confirmed and value.false_hold_count > 0 for value in pairs
        )
        agreement = all(value.prediction_agreement for value in pairs)
        reasons: set[str] = set()
        if not pairs:
            reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_TARGETED_COORDINATE_NO_RETAINED_WITNESS")
        if not dynamical_complete and not dynamic:
            reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_TARGETED_COORDINATE_DYNAMICAL_CERTIFICATE_INCOMPLETE")
        if not decision_complete and not decision:
            reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_TARGETED_COORDINATE_DECISION_CERTIFICATE_INCOMPLETE")
        if not agreement:
            reasons.add("HISTORY_BUDGET_PHASE_DIAGRAM_GENERATOR_OBSERVER_DISAGREEMENT")
        rows.append(
            HistoryBudgetPhaseDiagramTargetedCoordinateAdjudication(
                adjudication_id=(
                    f"targeted-coordinate-adjudication.{descriptor.unit_id}."
                    f"{coordinate.coordinate_id}"
                ),
                coordinate_id=coordinate.coordinate_id,
                unit_id=descriptor.unit_id,
                family=descriptor.family,
                scale_cells=descriptor.scale_cells,
                coordinate_kind=coordinate.kind,
                depth=coordinate.depth,
                budget=coordinate.budget,
                resolution_epsilon=coordinate.resolution_epsilon,
                primary=coordinate.primary,
                valid=True,
                nominated_pair_count=len(pairs),
                realized_collision_count=realized,
                dynamic_adverse_count=dynamic,
                decision_adverse_count=decision,
                unsafe_false_promotion_count=unsafe,
                false_hold_count=false_hold,
                dynamical_state=(
                    HistoryBudgetPhaseDiagramScientificState.OPPOSED
                    if dynamic
                    else HistoryBudgetPhaseDiagramScientificState.INFORMATIVE_NONADVERSE
                    if dynamical_complete
                    else HistoryBudgetPhaseDiagramScientificState.TARGETABILITY_LIMITED
                ),
                decision_state=(
                    HistoryBudgetPhaseDiagramScientificState.OPPOSED
                    if decision
                    else HistoryBudgetPhaseDiagramScientificState.INFORMATIVE_NONADVERSE
                    if decision_complete
                    else HistoryBudgetPhaseDiagramScientificState.TARGETABILITY_LIMITED
                ),
                dynamical_certificate_complete=dynamical_complete,
                decision_certificate_complete=decision_complete,
                generator_observer_agreement=agreement,
                reason_codes=tuple(sorted(reasons)),
            )
        )
    return tuple(sorted(rows, key=lambda value: value.adjudication_id))


__all__ = [
    "adjudicate_targeted_coordinates",
    "adjudicate_unit_scale",
    "adjudicate_untouched_scale",
]

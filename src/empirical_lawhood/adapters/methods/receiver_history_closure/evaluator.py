"""Reveal-lane unit adjudication for receiver-history."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

import numpy as np

from empirical_lawhood.adapters.receiver_history.contracts import (
    ReceiverHistoryActionPrediction,
    ReceiverHistoryChallengeNomination,
    ReceiverHistoryChallengeGeometry,
    ReceiverHistoryConfig,
    ReceiverHistoryCoordinateLabel,
    ReceiverHistoryDecisionDisposition,
    ReceiverHistoryDenominatorDescriptor,
    ReceiverHistoryDepthAdjudication,
    ReceiverHistoryEndpoint,
    ReceiverHistoryGeneratorActionOutcome,
    ReceiverHistoryGeneratorOutcome,
    ReceiverHistoryHistoryRankForecast,
    ReceiverHistoryGate,
    ReceiverHistoryMethodFreeze,
    ReceiverHistoryOptimizationCertificate,
    ReceiverHistoryOptimizationState,
    ReceiverHistoryOrientation,
    ReceiverHistoryPhase,
    ReceiverHistoryPairAdjudication,
    ReceiverHistoryScientificState,
    ReceiverHistorySparseCertificateCheck,
    ReceiverHistoryTargetedCoordinateAdjudication,
    ReceiverHistoryUntouchedCoordinateAdjudication,
    ReceiverHistoryUntouchedGeneratorOutcome,
    ReceiverHistoryUnitAdjudication,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import canonical_json_bytes


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("receiver-history evaluator output must be finite")
    return Decimal(str(float(value)))


def _nomination_set_digest(
    nominations: tuple[ReceiverHistoryChallengeGeometry, ...],
) -> str:
    return sha256(
        canonical_json_bytes(tuple(value.fingerprint() for value in nominations))
    ).hexdigest()


def _complete_collision_graph(
    histories: np.ndarray,
    coordinates: tuple[ReceiverHistoryCoordinateLabel, ...],
    validity: np.ndarray,
) -> np.ndarray:
    count = histories.shape[0]
    left, right = np.triu_indices(count, k=1)
    cumulative = np.zeros(left.size, dtype=np.float64)
    valid_pairs = validity[left] & validity[right]
    by_depth: dict[int, list[tuple[int, ReceiverHistoryCoordinateLabel]]] = {}
    for coordinate_index, coordinate in enumerate(coordinates):
        by_depth.setdefault(coordinate.depth, []).append((coordinate_index, coordinate))
    parts: list[np.ndarray] = []
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
    return np.ascontiguousarray(
        np.vstack(parts) if parts else np.empty((0, 3), dtype=np.int64),
        dtype=np.int64,
    )


def _complete_collision_graph_digest(
    edges: np.ndarray,
    coordinates: tuple[ReceiverHistoryCoordinateLabel, ...],
) -> str:
    return sha256(
        canonical_json_bytes(tuple(value.coordinate_id for value in coordinates))
        + np.ascontiguousarray(edges, dtype=np.int64).tobytes(order="C")
    ).hexdigest()


def _action_prediction_agrees(
    predicted: ReceiverHistoryActionPrediction,
    realized: ReceiverHistoryGeneratorActionOutcome,
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


def _lexical_reference_decision_errors(
    realized: ReceiverHistoryGeneratorActionOutcome,
    config: ReceiverHistoryConfig,
) -> tuple[bool, bool]:
    """Classify the nonreference member against the lexical plus reference."""

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

    # Targeting freezes the plus side as the lexical reference.  The common
    # decision is its exact-threshold decision; only the minus side is tested
    # with robust half-margins.
    reference_admit = realized.plus_admit
    nonreference = robust_admit(realized.minus_q_star, realized.minus_v_max_star)
    false_safe = reference_admit and nonreference is False
    false_hold = not reference_admit and nonreference is True
    return false_safe, false_hold


def _isolated_endpoint_orientation_matches(
    realized: ReceiverHistoryGeneratorActionOutcome,
    config: ReceiverHistoryConfig,
    *,
    gate: ReceiverHistoryGate,
    orientation: ReceiverHistoryOrientation,
) -> bool:
    """Check one designated action against the exact endpoint-local carrier."""

    target_high = config.target_charge_minimum_q_star + config.target_decision_margin / 2
    target_low = config.target_charge_minimum_q_star - config.target_decision_margin / 2
    sink_high = config.sink_voltage_maximum_v_star + config.sink_decision_margin / 2
    sink_low = config.sink_voltage_maximum_v_star - config.sink_decision_margin / 2
    if gate is ReceiverHistoryGate.TARGET:
        common_sink_safe = (
            realized.plus_v_max_star <= sink_low and realized.minus_v_max_star <= sink_low
        )
        if orientation is ReceiverHistoryOrientation.UNSAFE_PROMOTION:
            return bool(
                common_sink_safe
                and realized.plus_q_star >= target_high
                and realized.minus_q_star <= target_low
            )
        return bool(
            common_sink_safe
            and realized.plus_q_star <= target_low
            and realized.minus_q_star >= target_high
        )
    if gate is not ReceiverHistoryGate.SINK:
        raise ValueError("receiver-history isolated endpoint check requires a decision gate")
    common_target_positive = (
        realized.plus_q_star >= target_high and realized.minus_q_star >= target_high
    )
    if orientation is ReceiverHistoryOrientation.UNSAFE_PROMOTION:
        return bool(
            common_target_positive
            and realized.plus_v_max_star <= sink_low
            and realized.minus_v_max_star >= sink_high
        )
    return bool(
        common_target_positive
        and realized.plus_v_max_star >= sink_high
        and realized.minus_v_max_star <= sink_low
    )


def adjudicate_untouched_scale(
    *,
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    coordinates: tuple[ReceiverHistoryCoordinateLabel, ...],
    collision_edges: np.ndarray[tuple[int, int], np.dtype[np.int64]],
    receiver_history: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    action_metrics: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    action_receiver: np.ndarray[tuple[int, int, int], np.dtype[np.float64]],
    validity_mask: np.ndarray[tuple[int], np.dtype[np.int64]],
    generator_outcome: ReceiverHistoryUntouchedGeneratorOutcome,
) -> tuple[ReceiverHistoryUntouchedCoordinateAdjudication, ...]:
    'Adjudicate the frozen untouched collision graphs without pair-level inference.'

    count = config.untouched_preparation_count
    action_count = len(config.action_amplitudes_u_star) * len(config.action_durations_t_star)
    if (
        generator_outcome.descriptor_sha256 != descriptor.fingerprint()
        or generator_outcome.preparation_count != count
        or receiver_history.shape != (count, config.history_max_depth + 1, 8)
        or action_metrics.shape != (count, action_count, 3)
        or action_receiver.shape != (count, action_count, 8)
        or validity_mask.shape != (count,)
        or collision_edges.ndim != 2
        or collision_edges.shape[1:] != (3,)
    ):
        raise ValueError("receiver-history untouched evaluator input roster differs")
    if tuple(value.coordinate_id for value in coordinates) != tuple(
        sorted({value.coordinate_id for value in coordinates})
    ) or any(value.scale_cells != descriptor.scale_cells for value in coordinates):
        raise ValueError("receiver-history untouched coordinate roster differs")
    if not all(
        np.all(np.isfinite(value)) for value in (receiver_history, action_metrics, action_receiver)
    ):
        raise ValueError("receiver-history untouched evaluator received nonfinite arrays")
    if collision_edges.size and (
        np.min(collision_edges) < 0
        or np.max(collision_edges[:, 0]) >= len(coordinates)
        or np.max(collision_edges[:, 1:]) >= count
        or np.any(collision_edges[:, 1] >= collision_edges[:, 2])
    ):
        raise ValueError("receiver-history untouched collision graph is invalid")
    validity = np.asarray(validity_mask != 0, dtype=np.bool_)
    regenerated_edges = _complete_collision_graph(receiver_history, coordinates, validity)
    if (
        not np.array_equal(regenerated_edges, collision_edges)
        or _complete_collision_graph_digest(regenerated_edges, coordinates)
        != generator_outcome.complete_collision_graph_sha256
        or sha256(np.ascontiguousarray(validity_mask, dtype=np.int64).tobytes()).hexdigest()
        != generator_outcome.validity_mask_sha256
        or generator_outcome.valid_preparation_count != int(np.count_nonzero(validity))
    ):
        raise ValueError("receiver-history independent untouched collision graph differs")

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

    rows: list[ReceiverHistoryUntouchedCoordinateAdjudication] = []
    scale_valid = bool(np.all(validity_mask != 0))
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
                raise ValueError(
                    "receiver-history frozen untouched edge is not a realized collision"
                )
            receiver_defect = float(np.max(np.abs(action_receiver[left] - action_receiver[right])))
            metric_defect = float(np.max(np.abs(action_metrics[left] - action_metrics[right])))
            maximum_receiver = max(maximum_receiver, receiver_defect)
            maximum_metric = max(maximum_metric, metric_defect)
            edge_dynamic = receiver_defect >= float(config.future_divergence_epsilon)
            edge_unsafe = False
            edge_false_hold = False
            for action_index in range(action_count):
                # Edge ordering is canonical: lower preparation ID is left and
                # therefore the lexical reference.
                reference_admit = (
                    float(action_metrics[left, action_index, 0]) >= target
                    and float(action_metrics[left, action_index, 2]) <= sink
                )
                nonreference = robust_admit(
                    float(action_metrics[right, action_index, 0]),
                    float(action_metrics[right, action_index, 2]),
                )
                edge_unsafe = edge_unsafe or (reference_admit and nonreference is False)
                edge_false_hold = edge_false_hold or (not reference_admit and nonreference is True)
            dynamic_count += int(edge_dynamic)
            unsafe_count += int(edge_unsafe)
            false_hold_count += int(edge_false_hold)
            decision_count += int(edge_unsafe or edge_false_hold)
            any_adverse_count += int(edge_dynamic or edge_unsafe or edge_false_hold)
        state = (
            ReceiverHistoryScientificState.UNSAFE_FALSE_PROMOTION
            if unsafe_count
            else ReceiverHistoryScientificState.OPPOSED
            if dynamic_count or decision_count
            else ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
            if len(edges)
            else ReceiverHistoryScientificState.NO_ENCOUNTER_AT_512_PREPARATIONS
        )
        rows.append(
            ReceiverHistoryUntouchedCoordinateAdjudication(
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
                valid=scale_valid,
                collision_edge_count=len(edges) if scale_valid else None,
                prefix_edge_counts=prefix_counts,
                dynamic_adverse_edge_count=dynamic_count if scale_valid else None,
                decision_adverse_edge_count=decision_count if scale_valid else None,
                any_adverse_edge_count=any_adverse_count if scale_valid else None,
                unsafe_false_promotion_edge_count=unsafe_count if scale_valid else None,
                false_hold_edge_count=false_hold_count if scale_valid else None,
                maximum_future_receiver_defect=(
                    _decimal(maximum_receiver) if scale_valid else None
                ),
                maximum_future_metric_defect=(_decimal(maximum_metric) if scale_valid else None),
                scientific_state=state if scale_valid else ReceiverHistoryScientificState.INVALID,
                unsafe_first=True,
            )
        )
    return tuple(sorted(rows, key=lambda value: value.adjudication_id))


def _receiver(descriptor: ReceiverHistoryDenominatorDescriptor) -> np.ndarray:
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
    config: ReceiverHistoryConfig,
    descriptor: ReceiverHistoryDenominatorDescriptor,
    forecast: ReceiverHistoryHistoryRankForecast,
    geometries: tuple[ReceiverHistoryChallengeGeometry, ...],
    nominations: tuple[ReceiverHistoryChallengeNomination, ...],
    generator_outcome: ReceiverHistoryGeneratorOutcome,
    method_freeze: ReceiverHistoryMethodFreeze | None,
) -> ReceiverHistoryUnitAdjudication:
    """Reveal and adjudicate exactly one complete-unit finite scale view."""

    if generator_outcome.outcome_access not in {
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        OutcomeAccess.EVALUATION_SEALED,
    }:
        raise ValueError("receiver-history evaluator received an invalid outcome-access state")
    if config.phase is ReceiverHistoryPhase.TARGETED_EVALUATION and method_freeze is None:
        raise ValueError(
            "receiver-history evaluation adjudication requires the exact method freeze"
        )
    if method_freeze is not None:
        if (
            config.phase is not ReceiverHistoryPhase.TARGETED_EVALUATION
            or generator_outcome.generator_implementation_sha256
            != method_freeze.generator_implementation_sha256
            or any(
                value.observer_implementation_sha256 != method_freeze.observer_implementation_sha256
                for value in nominations
            )
        ):
            raise ValueError("receiver-history evaluation inputs differ from the method freeze")
    if descriptor.fingerprint() != generator_outcome.descriptor_sha256:
        raise ValueError("receiver-history generator outcome binds a different descriptor")
    if forecast.descriptor_sha256 != descriptor.fingerprint():
        raise ValueError("receiver-history forecast binds a different descriptor")
    if tuple(value.nomination_id for value in geometries) != tuple(
        value.nomination_id for value in nominations
    ):
        raise ValueError("receiver-history geometry and observer assessment rosters differ")
    expected_nomination_digest = _nomination_set_digest(geometries)
    if generator_outcome.nomination_set_sha256 != expected_nomination_digest:
        raise ValueError("receiver-history generator outcome binds a different nomination set")
    if tuple(value.nomination_id for value in nominations) != tuple(
        sorted({value.nomination_id for value in nominations})
    ):
        raise ValueError("receiver-history evaluator nominations must be sorted and unique")
    outcome_by_id = {value.nomination_id: value for value in generator_outcome.pair_outcomes}
    if set(outcome_by_id) != {value.nomination_id for value in nominations}:
        raise ValueError("receiver-history generator outcome pair roster differs")
    divergence_epsilon = float(config.future_divergence_epsilon)
    pair_rows: list[ReceiverHistoryPairAdjudication] = []
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
        realized_by_action = {value.action_id: value for value in outcome.action_outcomes}
        predicted_by_action = {value.action_id: value for value in nomination.action_predictions}
        if set(predicted_by_action) != set(expected_actions):
            raise ValueError("receiver-history nomination action roster differs from the config")
        for action_id, predicted in predicted_by_action.items():
            if (predicted.amplitude_u_star, predicted.duration_t_star) != expected_actions[
                action_id
            ] or predicted.requested_action_id != action_id:
                raise ValueError("receiver-history nomination action semantics differ")
        if {value.action_id for value in outcome.action_outcomes} != set(expected_actions):
            raise ValueError("receiver-history generator action roster differs from the config")
        for realized in outcome.action_outcomes:
            predicted = predicted_by_action[realized.action_id]
            if (
                (realized.amplitude_u_star, realized.duration_t_star)
                != expected_actions[realized.action_id]
                or not realized.requested_accepted_applied_realized_parity
                or realized.requested_action_id != realized.action_id
            ):
                raise ValueError("receiver-history realized action semantics differ")
            target_mismatch += int(realized.plus_target_pass != realized.minus_target_pass)
            sink_mismatch += int(realized.plus_sink_pass != realized.minus_sink_pass)
            action_false_safe, action_false_hold = _lexical_reference_decision_errors(
                realized, config
            )
            if nomination.gate not in {ReceiverHistoryGate.TARGET, ReceiverHistoryGate.SINK}:
                decision_mismatch += int(action_false_safe or action_false_hold)
                false_safe += int(action_false_safe)
                false_hold += int(action_false_hold)
            action_agreement = action_agreement and _action_prediction_agrees(
                predicted,
                realized,
                material_floor=divergence_epsilon,
                metric_tolerance=float(config.generator_observer_metric_tolerance),
            )
        endpoint = {
            ReceiverHistoryGate.DYNAMICAL: ReceiverHistoryEndpoint.DYNAMICAL,
            ReceiverHistoryGate.TARGET: ReceiverHistoryEndpoint.TARGET_DECISION,
            ReceiverHistoryGate.SINK: ReceiverHistoryEndpoint.SINK_DECISION,
            ReceiverHistoryGate.RANDOM: None,
        }[nomination.gate]
        if nomination.gate in {ReceiverHistoryGate.TARGET, ReceiverHistoryGate.SINK}:
            if nomination.targeting_action_id is None or nomination.orientation is None:
                raise ValueError("receiver-history lexical nomination lacks action/orientation")
            endpoint_match = _isolated_endpoint_orientation_matches(
                realized_by_action[nomination.targeting_action_id],
                config,
                gate=nomination.gate,
                orientation=nomination.orientation,
            )
            decision_mismatch = int(endpoint_match)
            false_safe = int(
                endpoint_match
                and nomination.orientation is ReceiverHistoryOrientation.UNSAFE_PROMOTION
            )
            false_hold = int(
                endpoint_match and nomination.orientation is ReceiverHistoryOrientation.FALSE_HOLD
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
            ReceiverHistoryPairAdjudication(
                nomination_id=nomination.nomination_id,
                coordinate_id=nomination.coordinate_id,
                depth=nomination.depth,
                resolution_epsilon=nomination.resolution_epsilon,
                gate=nomination.gate,
                endpoint=endpoint,
                orientation=nomination.orientation,
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
                endpoint_adverse=(
                    dynamic_adverse
                    if endpoint is ReceiverHistoryEndpoint.DYNAMICAL
                    else decision_adverse
                    if endpoint is not None
                    else False
                ),
                decision_disposition=(
                    ReceiverHistoryDecisionDisposition.UNSAFE_FALSE_PROMOTION
                    if false_safe
                    else (
                        ReceiverHistoryDecisionDisposition.FALSE_HOLD
                        if false_hold
                        else ReceiverHistoryDecisionDisposition.CORRECT_ACTION
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
    depth_rows: list[ReceiverHistoryDepthAdjudication] = []
    for depth in forecast.probe_depths:
        values = tuple(value for value in pair_rows_tuple if value.depth == depth)
        collisions = sum(value.collision_confirmed for value in values)
        dynamic_adverse_count = sum(value.dynamically_adverse for value in values)
        decision_adverse_count = sum(value.decision_adverse for value in values)
        false_safe_count = sum(value.false_safe_count for value in values)
        dynamic_state = (
            ReceiverHistoryScientificState.OPPOSED
            if dynamic_adverse_count
            else (
                ReceiverHistoryScientificState.SUPPORTED
                if collisions
                else ReceiverHistoryScientificState.UNEVALUABLE
            )
        )
        decision_state = (
            ReceiverHistoryScientificState.OPPOSED
            if decision_adverse_count
            else (
                ReceiverHistoryScientificState.SUPPORTED
                if collisions
                else ReceiverHistoryScientificState.UNEVALUABLE
            )
        )
        depth_rows.append(
            ReceiverHistoryDepthAdjudication(
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
        ReceiverHistoryScientificState.OPPOSED
        if dynamic_count
        else (
            ReceiverHistoryScientificState.SUPPORTED
            if collision_count
            else ReceiverHistoryScientificState.UNEVALUABLE
        )
    )
    decision_state = (
        ReceiverHistoryScientificState.OPPOSED
        if decision_count
        else (
            ReceiverHistoryScientificState.SUPPORTED
            if collision_count
            else ReceiverHistoryScientificState.UNEVALUABLE
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
        reasons.add('RECEIVER_HISTORY_EFFECTIVE_RANK_RIGHT_CENSORED')
    if forecast.algebraic_prediction_state is ReceiverHistoryScientificState.OPPOSED:
        reasons.add('RECEIVER_HISTORY_ALGEBRAIC_RANK_PREDICTION_OPPOSED')
    if not agreement:
        reasons.add('RECEIVER_HISTORY_GENERATOR_OBSERVER_DISAGREEMENT')
    if not pair_rows_tuple:
        reasons.add('RECEIVER_HISTORY_UNTARGETABLE_UNEVALUABLE')
    if sum(value.false_safe_count for value in pair_rows_tuple):
        reasons.add('RECEIVER_HISTORY_FALSE_SAFE_COUNTEREXAMPLE')
    return ReceiverHistoryUnitAdjudication(
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
    descriptor: ReceiverHistoryDenominatorDescriptor,
    coordinates: tuple[ReceiverHistoryCoordinateLabel, ...],
    unit_adjudication: ReceiverHistoryUnitAdjudication,
    certificates: tuple[ReceiverHistoryOptimizationCertificate, ...],
    sparse_certificate_checks: tuple[ReceiverHistorySparseCertificateCheck, ...],
    expected_action_ids: tuple[str, ...],
    optimization_tolerance: Decimal,
    eligible_endpoints_by_coordinate: dict[str, frozenset[ReceiverHistoryEndpoint]],
) -> tuple[ReceiverHistoryTargetedCoordinateAdjudication, ...]:
    'Derive the four-state targeted ledger without treating absent witnesses as support.'

    if unit_adjudication.unit_id != descriptor.unit_id:
        raise ValueError("receiver-history targeted coordinate ledger mixes independent units")
    rows: list[ReceiverHistoryTargetedCoordinateAdjudication] = []
    sparse_by_key = {value.objective_key: value for value in sparse_certificate_checks}
    if len(sparse_by_key) != len(sparse_certificate_checks):
        raise ValueError("receiver-history sparse certificate family is duplicated")
    expected_sparse_keys = {
        f"optimization.{coordinate.coordinate_id}.dynamical.none.none"
        for coordinate in coordinates
        if ReceiverHistoryEndpoint.DYNAMICAL
        in eligible_endpoints_by_coordinate.get(coordinate.coordinate_id, frozenset())
    } | {
        f"optimization.{coordinate.coordinate_id}.{gate.value.lower()}.{action_id}."
        f"{orientation.value.lower()}"
        for coordinate in coordinates
        for gate in (ReceiverHistoryGate.TARGET, ReceiverHistoryGate.SINK)
        for action_id in expected_action_ids
        for orientation in ReceiverHistoryOrientation
        if (
            ReceiverHistoryEndpoint.TARGET_DECISION
            if gate is ReceiverHistoryGate.TARGET
            else ReceiverHistoryEndpoint.SINK_DECISION
        )
        in eligible_endpoints_by_coordinate.get(coordinate.coordinate_id, frozenset())
    }
    if not set(sparse_by_key).issubset(expected_sparse_keys):
        raise ValueError("receiver-history sparse certificate family contains an undeclared key")

    def independently_nonadverse(
        dense: ReceiverHistoryOptimizationCertificate,
        sparse_check: ReceiverHistorySparseCertificateCheck | None,
    ) -> bool:
        resolved = {
            ReceiverHistoryOptimizationState.BELOW_MARGIN,
            ReceiverHistoryOptimizationState.INFEASIBLE,
        }
        tolerance = float(optimization_tolerance)
        return bool(
            sparse_check is not None
            and dense.complete_family_certificate
            and sparse_check.complete_family_certificate
            and dense.state in resolved
            and sparse_check.state in resolved
            and dense.state is sparse_check.state
            and dense.constraint_family_sha256 == sparse_check.constraint_family_sha256
            and float(dense.primal_residual) <= tolerance
            and float(sparse_check.primal_residual) <= tolerance
            and float(sparse_check.duality_gap) <= tolerance
            and dense.objective_upper_bound < dense.required_margin
            and sparse_check.objective_upper_bound < sparse_check.required_margin
        )

    for coordinate in coordinates:
        eligible_endpoints = tuple(
            sorted(
                eligible_endpoints_by_coordinate.get(coordinate.coordinate_id, frozenset()),
                key=lambda value: value.value,
            )
        )
        pairs = tuple(
            value
            for value in unit_adjudication.pair_adjudications
            if value.coordinate_id == coordinate.coordinate_id
        )
        coordinate_certificates = tuple(
            value for value in certificates if value.coordinate_id == coordinate.coordinate_id
        )
        certificate_keys = tuple(value.certificate_id for value in coordinate_certificates)
        if len(certificate_keys) != len(set(certificate_keys)):
            raise ValueError("receiver-history dense certificate family is duplicated")
        constraint_disagreement = any(
            sparse_by_key[value.certificate_id].constraint_family_sha256
            != value.constraint_family_sha256
            for value in coordinate_certificates
            if value.certificate_id in sparse_by_key
        )
        if constraint_disagreement:
            rows.append(
                ReceiverHistoryTargetedCoordinateAdjudication(
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
                    valid=False,
                    eligible_endpoints=eligible_endpoints,
                    nominated_pair_count=None,
                    realized_collision_count=None,
                    dynamic_adverse_count=None,
                    target_adverse_count=None,
                    sink_adverse_count=None,
                    decision_adverse_count=None,
                    target_unsafe_false_promotion_count=None,
                    target_false_hold_count=None,
                    sink_unsafe_false_promotion_count=None,
                    sink_false_hold_count=None,
                    unsafe_false_promotion_count=None,
                    false_hold_count=None,
                    dynamical_state=ReceiverHistoryScientificState.INVALID,
                    target_state=ReceiverHistoryScientificState.INVALID,
                    sink_state=ReceiverHistoryScientificState.INVALID,
                    decision_state=ReceiverHistoryScientificState.INVALID,
                    dynamical_certificate_complete=False,
                    target_certificate_complete=False,
                    sink_certificate_complete=False,
                    decision_certificate_complete=False,
                    generator_observer_agreement=None,
                    reason_codes=('RECEIVER_HISTORY_DENSE_SPARSE_CONSTRAINT_DISAGREEMENT',),
                )
            )
            continue
        dynamic_certificates = tuple(
            value
            for value in coordinate_certificates
            if value.gate is ReceiverHistoryGate.DYNAMICAL
        )
        dynamic_key = f"optimization.{coordinate.coordinate_id}.dynamical.none.none"
        dynamical_complete = (
            ReceiverHistoryEndpoint.DYNAMICAL in eligible_endpoints
            and len(dynamic_certificates) == 1
            and independently_nonadverse(dynamic_certificates[0], sparse_by_key.get(dynamic_key))
        )
        endpoint_certificate_complete: dict[ReceiverHistoryGate, bool] = {}
        for gate in (ReceiverHistoryGate.TARGET, ReceiverHistoryGate.SINK):
            endpoint_certificates = tuple(
                value for value in coordinate_certificates if value.gate is gate
            )
            endpoint = (
                ReceiverHistoryEndpoint.TARGET_DECISION
                if gate is ReceiverHistoryGate.TARGET
                else ReceiverHistoryEndpoint.SINK_DECISION
            )
            expected_keys = {
                (action_id, orientation)
                for action_id in expected_action_ids
                for orientation in ReceiverHistoryOrientation
            }
            actual_keys = {(value.action_id, value.orientation) for value in endpoint_certificates}
            endpoint_certificate_complete[gate] = (
                endpoint in eligible_endpoints
                and actual_keys == expected_keys
                and all(
                    independently_nonadverse(
                        value,
                        sparse_by_key.get(
                            f"optimization.{coordinate.coordinate_id}.{gate.value.lower()}."
                            f"{value.action_id}.{value.orientation.value.lower()}"
                        ),
                    )
                    for value in endpoint_certificates
                    if value.orientation is not None
                )
            )
        target_complete = endpoint_certificate_complete[ReceiverHistoryGate.TARGET]
        sink_complete = endpoint_certificate_complete[ReceiverHistoryGate.SINK]
        eligible_decision_endpoints = {
            ReceiverHistoryEndpoint.TARGET_DECISION,
            ReceiverHistoryEndpoint.SINK_DECISION,
        }.intersection(eligible_endpoints)
        decision_complete = bool(eligible_decision_endpoints) and all(
            endpoint_certificate_complete[
                ReceiverHistoryGate.TARGET
                if endpoint is ReceiverHistoryEndpoint.TARGET_DECISION
                else ReceiverHistoryGate.SINK
            ]
            for endpoint in eligible_decision_endpoints
        )
        realized = sum(value.collision_confirmed for value in pairs)
        dynamic = sum(
            value.endpoint is ReceiverHistoryEndpoint.DYNAMICAL and value.endpoint_adverse
            for value in pairs
        )
        target = sum(
            value.endpoint is ReceiverHistoryEndpoint.TARGET_DECISION and value.endpoint_adverse
            for value in pairs
        )
        sink = sum(
            value.endpoint is ReceiverHistoryEndpoint.SINK_DECISION and value.endpoint_adverse
            for value in pairs
        )
        decision = target + sink
        target_unsafe = sum(
            value.endpoint is ReceiverHistoryEndpoint.TARGET_DECISION
            and value.endpoint_adverse
            and value.orientation is ReceiverHistoryOrientation.UNSAFE_PROMOTION
            for value in pairs
        )
        target_hold = sum(
            value.endpoint is ReceiverHistoryEndpoint.TARGET_DECISION
            and value.endpoint_adverse
            and value.orientation is ReceiverHistoryOrientation.FALSE_HOLD
            for value in pairs
        )
        sink_unsafe = sum(
            value.endpoint is ReceiverHistoryEndpoint.SINK_DECISION
            and value.endpoint_adverse
            and value.orientation is ReceiverHistoryOrientation.UNSAFE_PROMOTION
            for value in pairs
        )
        sink_hold = sum(
            value.endpoint is ReceiverHistoryEndpoint.SINK_DECISION
            and value.endpoint_adverse
            and value.orientation is ReceiverHistoryOrientation.FALSE_HOLD
            for value in pairs
        )
        unsafe = target_unsafe + sink_unsafe
        false_hold = target_hold + sink_hold
        agreement = all(value.prediction_agreement for value in pairs)
        reasons: set[str] = set()
        if not pairs:
            reasons.add('RECEIVER_HISTORY_TARGETED_NO_RETAINED_WITNESS')
        if (
            ReceiverHistoryEndpoint.DYNAMICAL in eligible_endpoints
            and not dynamical_complete
            and not dynamic
        ):
            reasons.add('RECEIVER_HISTORY_TARGETED_DYNAMICAL_CERTIFICATE_INCOMPLETE')
        if (
            ReceiverHistoryEndpoint.TARGET_DECISION in eligible_endpoints
            and not target_complete
            and not target
        ):
            reasons.add('RECEIVER_HISTORY_TARGETED_TARGET_CERTIFICATE_INCOMPLETE')
        if (
            ReceiverHistoryEndpoint.SINK_DECISION in eligible_endpoints
            and not sink_complete
            and not sink
        ):
            reasons.add('RECEIVER_HISTORY_TARGETED_SINK_CERTIFICATE_INCOMPLETE')
        if not agreement:
            reasons.add('RECEIVER_HISTORY_GENERATOR_OBSERVER_DISAGREEMENT')
        rows.append(
            ReceiverHistoryTargetedCoordinateAdjudication(
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
                eligible_endpoints=eligible_endpoints,
                nominated_pair_count=len(pairs),
                realized_collision_count=realized,
                dynamic_adverse_count=dynamic,
                target_adverse_count=target,
                sink_adverse_count=sink,
                decision_adverse_count=decision,
                target_unsafe_false_promotion_count=target_unsafe,
                target_false_hold_count=target_hold,
                sink_unsafe_false_promotion_count=sink_unsafe,
                sink_false_hold_count=sink_hold,
                unsafe_false_promotion_count=unsafe,
                false_hold_count=false_hold,
                dynamical_state=(
                    ReceiverHistoryScientificState.NOT_ATTEMPTED
                    if ReceiverHistoryEndpoint.DYNAMICAL not in eligible_endpoints
                    else ReceiverHistoryScientificState.OPPOSED
                    if dynamic
                    else ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
                    if dynamical_complete
                    else ReceiverHistoryScientificState.TARGETABILITY_LIMITED
                ),
                target_state=(
                    ReceiverHistoryScientificState.NOT_ATTEMPTED
                    if ReceiverHistoryEndpoint.TARGET_DECISION not in eligible_endpoints
                    else ReceiverHistoryScientificState.OPPOSED
                    if target
                    else ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
                    if target_complete
                    else ReceiverHistoryScientificState.TARGETABILITY_LIMITED
                ),
                sink_state=(
                    ReceiverHistoryScientificState.NOT_ATTEMPTED
                    if ReceiverHistoryEndpoint.SINK_DECISION not in eligible_endpoints
                    else ReceiverHistoryScientificState.OPPOSED
                    if sink
                    else ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
                    if sink_complete
                    else ReceiverHistoryScientificState.TARGETABILITY_LIMITED
                ),
                decision_state=(
                    ReceiverHistoryScientificState.NOT_ATTEMPTED
                    if not eligible_decision_endpoints
                    else ReceiverHistoryScientificState.OPPOSED
                    if decision
                    else ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
                    if decision_complete
                    else ReceiverHistoryScientificState.TARGETABILITY_LIMITED
                ),
                dynamical_certificate_complete=dynamical_complete,
                target_certificate_complete=target_complete,
                sink_certificate_complete=sink_complete,
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

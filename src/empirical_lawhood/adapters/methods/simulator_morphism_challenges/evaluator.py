"""Reveal-lane unit adjudication for simulator morphism challenges."""

from __future__ import annotations

from decimal import Decimal
from hashlib import sha256

import numpy as np

from empirical_lawhood.adapters.simulator_morphism_challenges.contracts import SimulatorMorphismChallengeActionPrediction, SimulatorMorphismChallengeChallengeNomination, SimulatorMorphismChallengeConfig, SimulatorMorphismChallengeDenominatorDescriptor, SimulatorMorphismChallengeDepthAdjudication, SimulatorMorphismChallengeGeneratorActionOutcome, SimulatorMorphismChallengeGeneratorOutcome, SimulatorMorphismChallengeHistoryRankForecast, SimulatorMorphismChallengeMethodFreeze, SimulatorMorphismChallengePairAdjudication, SimulatorMorphismChallengeScientificState, SimulatorMorphismChallengeUnitAdjudication
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import canonical_json_bytes


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("simulator morphism challenges evaluator output must be finite")
    return Decimal(str(float(value)))


def _nomination_set_digest(
    nominations: tuple[SimulatorMorphismChallengeChallengeNomination, ...],
) -> str:
    return sha256(
        canonical_json_bytes(tuple(value.fingerprint() for value in nominations))
    ).hexdigest()


def _action_prediction_agrees(
    predicted: SimulatorMorphismChallengeActionPrediction,
    realized: SimulatorMorphismChallengeGeneratorActionOutcome,
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
    realized: SimulatorMorphismChallengeGeneratorActionOutcome,
) -> tuple[bool, bool]:
    """Classify receiver-law errors against the frozen midpoint decision."""

    false_safe = realized.midpoint_admit and (not realized.plus_admit or not realized.minus_admit)
    false_hold = not realized.midpoint_admit and (realized.plus_admit or realized.minus_admit)
    return false_safe, false_hold


def adjudicate_unit_scale(
    *,
    config: SimulatorMorphismChallengeConfig,
    descriptor: SimulatorMorphismChallengeDenominatorDescriptor,
    forecast: SimulatorMorphismChallengeHistoryRankForecast,
    nominations: tuple[SimulatorMorphismChallengeChallengeNomination, ...],
    generator_outcome: SimulatorMorphismChallengeGeneratorOutcome,
    method_freeze: SimulatorMorphismChallengeMethodFreeze | None,
) -> SimulatorMorphismChallengeUnitAdjudication:
    """Reveal and adjudicate exactly one complete-unit finite scale view."""

    if generator_outcome.outcome_access not in {
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        OutcomeAccess.EVALUATION_SEALED,
    }:
        raise ValueError("simulator morphism challenges evaluator received an invalid outcome-access state")
    if config.phase.value == "EVALUATION" and method_freeze is None:
        raise ValueError("simulator morphism challenges evaluation adjudication requires the exact method freeze")
    if method_freeze is not None:
        if (
            config.phase.value != "EVALUATION"
            or config.fingerprint() != method_freeze.evaluation_config_sha256
            or generator_outcome.generator_implementation_sha256
            != method_freeze.generator_implementation_sha256
            or any(
                value.observer_implementation_sha256 != method_freeze.observer_implementation_sha256
                for value in nominations
            )
        ):
            raise ValueError("simulator morphism challenges evaluation inputs differ from the method freeze")
    if descriptor.fingerprint() != generator_outcome.descriptor_sha256:
        raise ValueError("simulator morphism challenges generator outcome binds a different descriptor")
    if forecast.descriptor_sha256 != descriptor.fingerprint():
        raise ValueError("simulator morphism challenges forecast binds a different descriptor")
    expected_nomination_digest = _nomination_set_digest(nominations)
    if generator_outcome.nomination_set_sha256 != expected_nomination_digest:
        raise ValueError("simulator morphism challenges generator outcome binds a different nomination set")
    if tuple(value.nomination_id for value in nominations) != tuple(
        sorted({value.nomination_id for value in nominations})
    ):
        raise ValueError("simulator morphism challenges evaluator nominations must be sorted and unique")
    outcome_by_id = {value.nomination_id: value for value in generator_outcome.pair_outcomes}
    if set(outcome_by_id) != {value.nomination_id for value in nominations}:
        raise ValueError("simulator morphism challenges generator outcome pair roster differs")
    collision_epsilon = float(config.coordinate_collision_epsilon)
    divergence_epsilon = float(config.future_divergence_epsilon)
    pair_rows: list[SimulatorMorphismChallengePairAdjudication] = []
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
            raise ValueError("simulator morphism challenges nomination action roster differs from the config")
        for action_id, predicted in predicted_by_action.items():
            if (predicted.amplitude_u_star, predicted.duration_t_star) != expected_actions[
                action_id
            ] or predicted.requested_action_id != action_id:
                raise ValueError("simulator morphism challenges nomination action semantics differ")
        if {value.action_id for value in outcome.action_outcomes} != set(expected_actions):
            raise ValueError("simulator morphism challenges generator action roster differs from the config")
        for realized in outcome.action_outcomes:
            predicted = predicted_by_action[realized.action_id]
            if (
                (realized.amplitude_u_star, realized.duration_t_star)
                != expected_actions[realized.action_id]
                or not realized.requested_accepted_applied_realized_parity
                or realized.requested_action_id != realized.action_id
            ):
                raise ValueError("simulator morphism challenges realized action semantics differ")
            target_mismatch += int(realized.plus_target_pass != realized.minus_target_pass)
            sink_mismatch += int(realized.plus_sink_pass != realized.minus_sink_pass)
            action_false_safe, action_false_hold = _midpoint_decision_errors(realized)
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
            collision
            and max(
                float(outcome.maximum_future_receiver_defect),
                float(outcome.maximum_future_metric_defect),
            )
            >= divergence_epsilon
        )
        decision_adverse = collision and decision_mismatch > 0
        prediction_collision = float(nomination.predicted_coordinate_defect) <= collision_epsilon
        prediction_material = (
            max(
                float(nomination.predicted_maximum_future_receiver_defect),
                float(nomination.predicted_maximum_future_metric_defect),
            )
            >= divergence_epsilon
        )
        realized_material = (
            max(
                float(outcome.maximum_future_receiver_defect),
                float(outcome.maximum_future_metric_defect),
            )
            >= divergence_epsilon
        )
        pair_rows.append(
            SimulatorMorphismChallengePairAdjudication(
                nomination_id=nomination.nomination_id,
                depth=nomination.depth,
                gate=nomination.gate,
                realized_coordinate_defect=outcome.realized_coordinate_defect,
                maximum_future_receiver_defect=outcome.maximum_future_receiver_defect,
                maximum_future_metric_defect=outcome.maximum_future_metric_defect,
                target_mismatch_count=target_mismatch,
                sink_mismatch_count=sink_mismatch,
                decision_mismatch_count=decision_mismatch,
                false_safe_count=false_safe,
                false_hold_count=false_hold,
                collision_confirmed=collision,
                dynamically_adverse=dynamic_adverse,
                decision_adverse=decision_adverse,
                prediction_agreement=(
                    action_agreement
                    and prediction_collision == collision
                    and prediction_material == realized_material
                ),
            )
        )
    pair_rows_tuple = tuple(sorted(pair_rows, key=lambda value: value.nomination_id))
    depth_rows: list[SimulatorMorphismChallengeDepthAdjudication] = []
    for depth in forecast.probe_depths:
        values = tuple(value for value in pair_rows_tuple if value.depth == depth)
        collisions = sum(value.collision_confirmed for value in values)
        dynamic_adverse_count = sum(value.dynamically_adverse for value in values)
        decision_adverse_count = sum(value.decision_adverse for value in values)
        false_safe_count = sum(value.false_safe_count for value in values)
        dynamic_state = (
            SimulatorMorphismChallengeScientificState.OPPOSED
            if dynamic_adverse_count
            else (
                SimulatorMorphismChallengeScientificState.SUPPORTED
                if collisions
                else SimulatorMorphismChallengeScientificState.UNEVALUABLE
            )
        )
        decision_state = (
            SimulatorMorphismChallengeScientificState.OPPOSED
            if decision_adverse_count
            else (
                SimulatorMorphismChallengeScientificState.SUPPORTED
                if collisions
                else SimulatorMorphismChallengeScientificState.UNEVALUABLE
            )
        )
        depth_rows.append(
            SimulatorMorphismChallengeDepthAdjudication(
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
        SimulatorMorphismChallengeScientificState.OPPOSED
        if dynamic_count
        else (
            SimulatorMorphismChallengeScientificState.SUPPORTED
            if collision_count
            else SimulatorMorphismChallengeScientificState.UNEVALUABLE
        )
    )
    decision_state = (
        SimulatorMorphismChallengeScientificState.OPPOSED
        if decision_count
        else (
            SimulatorMorphismChallengeScientificState.SUPPORTED
            if collision_count
            else SimulatorMorphismChallengeScientificState.UNEVALUABLE
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
        reasons.add("SIMULATOR_MORPHISM_CHALLENGE_EFFECTIVE_RANK_RIGHT_CENSORED")
    if forecast.algebraic_prediction_state is SimulatorMorphismChallengeScientificState.OPPOSED:
        reasons.add("SIMULATOR_MORPHISM_CHALLENGE_ALGEBRAIC_RANK_PREDICTION_OPPOSED")
    if not agreement:
        reasons.add("SIMULATOR_MORPHISM_CHALLENGE_GENERATOR_OBSERVER_DISAGREEMENT")
    if not pair_rows_tuple:
        reasons.add("SIMULATOR_MORPHISM_CHALLENGE_UNTARGETABLE_UNEVALUABLE")
    if sum(value.false_safe_count for value in pair_rows_tuple):
        reasons.add("SIMULATOR_MORPHISM_CHALLENGE_FALSE_SAFE_COUNTEREXAMPLE")
    return SimulatorMorphismChallengeUnitAdjudication(
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


__all__ = ["adjudicate_unit_scale"]

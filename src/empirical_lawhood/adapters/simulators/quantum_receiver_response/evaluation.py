"""Outcome-isolated adjudication functions for quantum receiver response."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .seed_commitments import BootstrapRole, bootstrap_seed

from .contracts import Action, QuantumReceiverResponseConfig, preparation_weight
from .receivers import ActionReceiver, ReceiverOperator, TrajectoryReceiver, excess_variance_contributions, receiver_operator, weighted_spearman
from .response import PredictionRow, ResponseDelta, ThresholdFamilies


@dataclass(frozen=True, slots=True)
class SimultaneousInterval:
    estimate: float
    lower: float
    upper: float
    standard_error: float


@dataclass(frozen=True, slots=True)
class ObservationOrderResult:
    verdict: str
    intervals: Mapping[str, SimultaneousInterval]
    strong_h20: ReceiverOperator
    strong_h4: ReceiverOperator
    weak_h20: ReceiverOperator | None
    association_h4: float
    preparation_count: int
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResponseLawEvaluationResult:
    verdict: str
    proxy_verdict: str
    causal_response_intervals: Mapping[str, SimultaneousInterval]
    supported_fraction_interval: SimultaneousInterval
    primary_adequate_fraction_interval: SimultaneousInterval
    full_adequate_fraction_interval: SimultaneousInterval
    diagnostic_intervals: Mapping[str, SimultaneousInterval]
    reason_codes: tuple[str, ...]


def _stratified_indices(
    k_values: Sequence[int],
    *,
    replicates: int,
    seed: int,
) -> NDArray[np.int32]:
    strata = np.asarray(k_values, dtype=np.int16)
    if set(strata.tolist()) != set(range(7)):
        raise ValueError("bootstrap requires all seven preparation strata")
    rng = np.random.Generator(np.random.Philox(seed))
    result = np.empty((replicates, len(strata)), dtype=np.int32)
    offset = 0
    for k_left in range(7):
        local = np.flatnonzero(strata == k_left)
        if len(local) < 2:
            raise ValueError("bootstrap stratum has fewer than two units")
        draws = rng.choice(local, size=(replicates, len(local)), replace=True)
        result[:, offset : offset + len(local)] = draws
        offset += len(local)
    return result


def _simultaneous_intervals(
    estimates: NDArray[np.float64],
    bootstrap_values: NDArray[np.float64],
    *,
    family_alpha: float,
) -> tuple[SimultaneousInterval, ...]:
    if bootstrap_values.ndim != 2 or bootstrap_values.shape[1] != len(estimates):
        raise ValueError("simultaneous bootstrap shape differs")
    standard_errors = np.std(bootstrap_values, axis=0, ddof=1)
    safe = np.maximum(standard_errors, 1e-15)
    standardized = np.abs((bootstrap_values - estimates) / safe)
    maximum = np.max(standardized, axis=1)
    critical = float(np.quantile(maximum, 1.0 - family_alpha, method="higher"))
    return tuple(
        SimultaneousInterval(
            estimate=float(estimate),
            lower=float(estimate - critical * standard_error),
            upper=float(estimate + critical * standard_error),
            standard_error=float(standard_error),
        )
        for estimate, standard_error in zip(
            estimates,
            standard_errors,
            strict=True,
        )
    )


def _operator_metrics(
    receivers: Sequence[TrajectoryReceiver],
    k_values: Sequence[int],
    *,
    l_sites: int,
    particles: int,
) -> ReceiverOperator:
    return receiver_operator(
        np.asarray([receiver.site_counts for receiver in receivers], dtype=np.float64),
        k_values,
        l_sites=l_sites,
        particles=particles,
    )


def adjudicate_observation_order(
    *,
    config: QuantumReceiverResponseConfig,
    unit_ids: Sequence[str],
    k_values: Sequence[int],
    strong_h20: Sequence[TrajectoryReceiver],
    strong_h4: Sequence[TrajectoryReceiver],
    weak_h20: Sequence[TrajectoryReceiver] | None,
) -> ObservationOrderResult:
    count = len(unit_ids)
    if not (
        count == len(k_values) == len(strong_h20) == len(strong_h4)
        and (weak_h20 is None or len(weak_h20) == count)
    ):
        raise ValueError("observation-order receiver rows differ")
    if len(set(unit_ids)) != count:
        raise ValueError("observation-order independent-unit IDs are not unique")
    strong20 = _operator_metrics(
        strong_h20,
        k_values,
        l_sites=config.l_primary,
        particles=config.n_primary,
    )
    strong4 = _operator_metrics(
        strong_h4,
        k_values,
        l_sites=config.l_primary,
        particles=config.n_primary,
    )
    weak20 = (
        _operator_metrics(
            weak_h20,
            k_values,
            l_sites=config.l_primary,
            particles=config.n_primary,
        )
        if weak_h20 is not None
        else None
    )
    n_a4 = [receiver.n_a for receiver in strong_h4]
    z_a4 = excess_variance_contributions(n_a4, k_values)
    association = weighted_spearman(
        [receiver.burst_a for receiver in strong_h4],
        z_a4.tolist(),
        k_values,
        l_sites=config.l_primary,
        particles=config.n_primary,
    )

    def metrics(indices: NDArray[np.int32]) -> NDArray[np.float64]:
        local_k = [k_values[int(index)] for index in indices]
        local20 = [strong_h20[int(index)] for index in indices]
        local4 = [strong_h4[int(index)] for index in indices]
        op20 = _operator_metrics(
            local20,
            local_k,
            l_sites=config.l_primary,
            particles=config.n_primary,
        )
        op4 = _operator_metrics(
            local4,
            local_k,
            l_sites=config.l_primary,
            particles=config.n_primary,
        )
        local_z = excess_variance_contributions(
            [receiver.n_a for receiver in local4],
            local_k,
        )
        local_assoc = weighted_spearman(
            [receiver.burst_a for receiver in local4],
            local_z.tolist(),
            local_k,
            l_sites=config.l_primary,
            particles=config.n_primary,
        )
        weak_separation = math.nan
        if weak_h20 is not None:
            weak_op = _operator_metrics(
                [weak_h20[int(index)] for index in indices],
                local_k,
                l_sites=config.l_primary,
                particles=config.n_primary,
            )
            weak_separation = op20.chi_a - weak_op.chi_a
        return np.asarray(
            [
                op20.chi_a,
                op4.chi_a,
                op20.chi_total,
                op4.chi_total,
                op20.cancellation_residual,
                op4.cancellation_residual,
                weak_separation,
                local_assoc,
            ],
            dtype=np.float64,
        )

    weak_separation = strong20.chi_a - weak20.chi_a if weak20 is not None else math.nan
    observed = np.asarray(
        [
            strong20.chi_a,
            strong4.chi_a,
            strong20.chi_total,
            strong4.chi_total,
            strong20.cancellation_residual,
            strong4.cancellation_residual,
            weak_separation,
            association,
        ],
        dtype=np.float64,
    )
    indices = _stratified_indices(
        k_values,
        replicates=config.bootstrap_replicates,
        seed=bootstrap_seed(BootstrapRole.RECEIVER_ORDER),
    )
    bootstrap = np.asarray([metrics(row) for row in indices], dtype=np.float64)
    active_columns = np.flatnonzero(np.isfinite(observed))
    active_intervals = _simultaneous_intervals(
        observed[active_columns],
        bootstrap[:, active_columns],
        family_alpha=float(config.observation_order_family_alpha),
    )
    names = (
        "chi_h20",
        "chi_h4",
        "chi_total_h20",
        "chi_total_h4",
        "cancellation_h20",
        "cancellation_h4",
        "weak_separation_h20",
        "proxy_association_h4",
    )
    intervals: dict[str, SimultaneousInterval] = {}
    for column, interval in zip(active_columns, active_intervals, strict=True):
        intervals[names[int(column)]] = interval

    reasons = []
    operator_valid = (
        strong20.quadratic_form_error <= 1e-10
        and strong4.quadratic_form_error <= 1e-10
        and np.all(np.isfinite(strong20.g_n))
        and np.all(np.isfinite(strong4.g_n))
    )
    if not operator_valid:
        reasons.append("RECEIVER_OPERATOR_CONFORMANCE_FAILED")
    long_pass = intervals["chi_h20"].lower >= float(config.observation_order_chi_h20_gate)
    short_pass = intervals["chi_h4"].lower >= float(config.observation_order_chi_h4_gate)
    global_pass = all(
        max(abs(intervals[name].lower), abs(intervals[name].upper))
        <= float(config.observation_order_global_equivalence)
        for name in ("chi_total_h20", "chi_total_h4")
    )
    cancellation_pass = all(
        max(abs(intervals[name].lower), abs(intervals[name].upper))
        <= float(config.observation_order_cancellation_equivalence)
        for name in ("cancellation_h20", "cancellation_h4")
    )
    association_pass = intervals["proxy_association_h4"].lower >= float(config.observation_order_proxy_association)
    weak_pass = weak20 is None or intervals["weak_separation_h20"].lower >= float(
        config.observation_order_weak_separation
    )
    if not operator_valid or not all(
        np.isfinite([interval.estimate for interval in intervals.values()])
    ):
        verdict = "OBSERVATION_ORDER_UNEVALUABLE"
    elif not global_pass:
        verdict = "OBSERVATION_ORDER_GLOBAL_NULL_OPPOSED"
    elif long_pass and short_pass and not association_pass:
        verdict = "OBSERVATION_ORDER_RECEIVER_OPPORTUNITY_PROXY_STOP"
    elif not long_pass or not short_pass:
        verdict = "OBSERVATION_ORDER_NO_RECEIVER_OPPORTUNITY"
    elif not cancellation_pass or not weak_pass:
        verdict = "OBSERVATION_ORDER_RECEIVER_LOCAL_ORDER_MIXED"
    else:
        verdict = "OBSERVATION_ORDER_RECEIVER_OPPORTUNITY_SUPPORTED"
    return ObservationOrderResult(
        verdict=verdict,
        intervals=intervals,
        strong_h20=strong20,
        strong_h4=strong4,
        weak_h20=weak20,
        association_h4=association,
        preparation_count=count,
        reason_codes=tuple(reasons),
    )


def _weighted_mean(
    values: Sequence[float],
    k_values: Sequence[int],
) -> float:
    strata = np.asarray(k_values, dtype=np.int16)
    array = np.asarray(values, dtype=np.float64)
    result = 0.0
    for k_left in range(7):
        local = array[strata == k_left]
        if not len(local):
            raise ValueError("weighted mean lacks a preparation stratum")
        result += float(preparation_weight(12, 6, k_left)) * float(local.mean())
    return result


def bootstrap_total_covariance_deltas(
    *,
    config: QuantumReceiverResponseConfig,
    unit_ids: Sequence[str],
    k_values: Sequence[int],
    action_receivers: Mapping[Action, ActionReceiver],
    conditional_means: Mapping[tuple[str, Action], NDArray[np.float64]],
    conditional_covariances: Mapping[
        tuple[str, Action],
        NDArray[np.float64],
    ],
) -> dict[str, NDArray[np.float64]]:
    """Bootstrap exact law-of-total-covariance action contrasts by prefix."""

    if set(action_receivers) != set(Action):
        raise ValueError("total-covariance bootstrap lacks an action")
    if len(unit_ids) != len(k_values):
        raise ValueError("total-covariance bootstrap population differs")
    for unit_id in unit_ids:
        for action in Action:
            mean = conditional_means.get((unit_id, action))
            covariance = conditional_covariances.get((unit_id, action))
            if mean is None or covariance is None:
                raise ValueError("total-covariance bootstrap lacks a prefix operand")
            if mean.shape != (config.l_primary,) or covariance.shape != (
                config.l_primary,
                config.l_primary,
            ):
                raise ValueError("total-covariance bootstrap operand shape differs")
    indices = _stratified_indices(
        k_values,
        replicates=config.bootstrap_replicates,
        seed=bootstrap_seed(BootstrapRole.TOTAL_COVARIANCE),
    )
    metrics = {
        action: np.empty(
            (config.bootstrap_replicates, 2),
            dtype=np.float64,
        )
        for action in Action
    }
    half = np.zeros(config.l_primary, dtype=np.float64)
    half[: config.l_primary // 2] = 1.0
    strata = np.asarray(k_values, dtype=np.int16)
    mean_arrays = {
        action: np.asarray(
            [conditional_means[(unit_id, action)] for unit_id in unit_ids],
            dtype=np.float64,
        )
        for action in Action
    }
    covariance_arrays = {
        action: np.asarray(
            [conditional_covariances[(unit_id, action)] for unit_id in unit_ids],
            dtype=np.float64,
        )
        for action in Action
    }
    for replicate, row in enumerate(indices):
        local_k = strata[row]
        for action in Action:
            local_means = mean_arrays[action][row]
            local_covariances = covariance_arrays[action][row]
            population_mean = np.zeros(config.l_primary, dtype=np.float64)
            stratum_means: dict[int, NDArray[np.float64]] = {}
            within = np.zeros(
                (config.l_primary, config.l_primary),
                dtype=np.float64,
            )
            for k_left in range(config.n_primary + 1):
                mask = local_k == k_left
                if np.count_nonzero(mask) < 2:
                    raise ValueError("total-covariance bootstrap stratum has fewer than two rows")
                weight = float(
                    preparation_weight(
                        config.l_primary,
                        config.n_primary,
                        k_left,
                    )
                )
                stratum_mean = np.mean(local_means[mask], axis=0)
                stratum_means[k_left] = stratum_mean
                population_mean += weight * stratum_mean
                within += weight * np.mean(local_covariances[mask], axis=0)
            between = np.zeros_like(within)
            for k_left in range(config.n_primary + 1):
                mask = local_k == k_left
                weight = float(
                    preparation_weight(
                        config.l_primary,
                        config.n_primary,
                        k_left,
                    )
                )
                local_covariance = np.atleast_2d(np.cov(local_means[mask], rowvar=False, ddof=1))
                delta = stratum_means[k_left] - population_mean
                between += weight * (local_covariance + np.outer(delta, delta))
            total = within + between
            mean_a = float(half @ population_mean)
            variance_a = float(half @ total @ half)
            metrics[action][replicate, 0] = variance_a - mean_a
            metrics[action][replicate, 1] = variance_a / mean_a
    result: dict[str, NDArray[np.float64]] = {}
    hold = metrics[Action.HOLD]
    for action in (Action.MINUS, Action.PLUS):
        difference = metrics[action] - hold
        result[f"delta_v_anom_a_{action.value}"] = difference[:, 0]
        result[f"delta_fano_a_{action.value}"] = difference[:, 1]
    return result


def adjudicate_response_law(
    *,
    config: QuantumReceiverResponseConfig,
    unit_ids: Sequence[str],
    k_values: Sequence[int],
    deltas: Mapping[tuple[str, Action], ResponseDelta],
    prediction_rows: Sequence[PredictionRow],
    thresholds: ThresholdFamilies,
    action_receiver_metrics: Mapping[Action, ReceiverOperator],
    action_metric_bootstrap: Mapping[str, Sequence[float]],
    observation_order_proxy_passed: bool,
) -> ResponseLawEvaluationResult:
    if len(unit_ids) != len(k_values) or set(unit_ids) != {unit_id for unit_id, _ in deltas}:
        raise ValueError("response-law-evaluation response population differs from its roster")
    estimates = []
    names = []
    for action in (Action.MINUS, Action.PLUS):
        local = [deltas[(unit_id, action)].burst_a for unit_id in unit_ids]
        estimates.append(_weighted_mean(local, k_values))
        names.append(f"delta_burst_a_{action.value}")
        estimates.append(
            action_receiver_metrics[action].v_anom_a - action_receiver_metrics[Action.HOLD].v_anom_a
        )
        names.append(f"delta_v_anom_a_{action.value}")
        estimates.append(
            action_receiver_metrics[action].fano_a - action_receiver_metrics[Action.HOLD].fano_a
        )
        names.append(f"delta_fano_a_{action.value}")
    indices = _stratified_indices(
        k_values,
        replicates=config.bootstrap_replicates,
        seed=bootstrap_seed(BootstrapRole.RESPONSE_COORDINATE),
    )
    burst_bootstrap = np.empty((config.bootstrap_replicates, 2), dtype=np.float64)
    for row_index, row in enumerate(indices):
        local_k = [k_values[int(index)] for index in row]
        for action_index, action in enumerate((Action.MINUS, Action.PLUS)):
            burst_bootstrap[row_index, action_index] = _weighted_mean(
                [deltas[(unit_ids[int(index)], action)].burst_a for index in row],
                local_k,
            )
    matrix = np.empty((config.bootstrap_replicates, 6), dtype=np.float64)
    matrix[:, 0] = burst_bootstrap[:, 0]
    matrix[:, 3] = burst_bootstrap[:, 1]
    for action_index, base in ((0, 0), (1, 3)):
        action = (Action.MINUS, Action.PLUS)[action_index]
        for offset, metric in enumerate(("v_anom_a", "fano_a"), start=1):
            key = f"delta_{metric}_{action.value}"
            values = np.asarray(action_metric_bootstrap.get(key, ()), dtype=np.float64)
            if values.shape != (config.bootstrap_replicates,):
                raise ValueError(f"response-law-evaluation total-covariance bootstrap is missing {key}")
            matrix[:, base + offset] = values
    estimate_array = np.asarray(estimates, dtype=np.float64)
    burst_columns = np.asarray((0, 3), dtype=np.int64)
    receiver_columns = np.asarray((1, 2, 4, 5), dtype=np.int64)
    burst_intervals = _simultaneous_intervals(
        estimate_array[burst_columns],
        matrix[:, burst_columns],
        family_alpha=float(config.response_law_evaluation_family_alpha),
    )
    receiver_intervals = _simultaneous_intervals(
        estimate_array[receiver_columns],
        matrix[:, receiver_columns],
        family_alpha=float(config.response_law_evaluation_family_alpha),
    )
    intervals = {
        **dict(
            zip(
                (names[int(index)] for index in burst_columns),
                burst_intervals,
                strict=True,
            )
        ),
        **dict(
            zip(
                (names[int(index)] for index in receiver_columns),
                receiver_intervals,
                strict=True,
            )
        ),
    }
    reducing_actions = [
        action
        for action in (Action.MINUS, Action.PLUS)
        if intervals[f"delta_burst_a_{action.value}"].upper <= -thresholds.delta["burst_a"]
    ]
    increasing_actions = [
        action
        for action in (Action.MINUS, Action.PLUS)
        if intervals[f"delta_burst_a_{action.value}"].lower >= thresholds.delta["burst_a"]
    ]
    material_actions = tuple(sorted({*reducing_actions, *increasing_actions}))
    proxy_verdict = "PROXY_RECEIVER_UNEVALUABLE"
    if observation_order_proxy_passed and reducing_actions:
        faithful = False
        opposed = False
        nonworsening = False
        for action in reducing_actions:
            delta_v = intervals[f"delta_v_anom_a_{action.value}"]
            delta_f = intervals[f"delta_fano_a_{action.value}"]
            if (
                delta_v.upper <= -thresholds.delta["v_anom_a"]
                and delta_f.upper <= thresholds.delta["fano_a"]
            ) or (
                delta_f.upper <= -thresholds.delta["fano_a"]
                and delta_v.upper <= thresholds.delta["v_anom_a"]
            ):
                faithful = True
            if (
                delta_v.lower >= thresholds.delta["v_anom_a"]
                or delta_f.lower >= thresholds.delta["fano_a"]
            ):
                opposed = True
            if (
                delta_v.upper <= thresholds.delta["v_anom_a"]
                and delta_f.upper <= thresholds.delta["fano_a"]
            ):
                nonworsening = True
        if faithful:
            proxy_verdict = "PROXY_RECEIVER_FAITHFUL"
        elif opposed:
            proxy_verdict = "PROXY_RECEIVER_UNFAITHFUL"
        elif nonworsening:
            proxy_verdict = "PROXY_RECEIVER_PARTIAL"

    by_unit: dict[str, dict[Action, PredictionRow]] = {}
    for row in prediction_rows:
        if row.target_unit_id not in set(unit_ids) or row.action is Action.HOLD:
            raise ValueError("response-law-evaluation prediction row is outside the evaluation population")
        unit_rows = by_unit.setdefault(row.target_unit_id, {})
        if row.action in unit_rows:
            raise ValueError("response-law-evaluation prediction row is duplicated")
        unit_rows[row.action] = row
    if set(by_unit) != set(unit_ids) or any(
        set(local) != {Action.MINUS, Action.PLUS} for local in by_unit.values()
    ):
        raise ValueError("response-law-evaluation prediction population lacks an action row")

    tolerance = np.asarray(
        [thresholds.prediction[key] for key in ("burst_a", "burst_b", "n_a", "q_a", "work_abs")],
        dtype=np.float64,
    )

    def prediction_metrics(sample_unit_ids: Sequence[str]) -> NDArray[np.float64]:
        local_rows = [
            by_unit[unit_id][action]
            for unit_id in sample_unit_ids
            for action in (Action.MINUS, Action.PLUS)
        ]
        supported_prefixes = [
            all(by_unit[unit_id][action].inside_support for action in (Action.MINUS, Action.PLUS))
            for unit_id in sample_unit_ids
        ]
        supported_rows = [row for row in local_rows if row.inside_support]
        scaled_errors = [
            float(np.max(row.errors / tolerance))
            for row in supported_rows
            if row.errors is not None
        ]
        action_adequacy = []
        for action in (Action.MINUS, Action.PLUS):
            local_supported = [row for row in supported_rows if row.action is action]
            action_adequacy.append(
                (
                    float(np.mean([row.adequate is True for row in local_supported]))
                    if local_supported
                    else 0.0
                )
            )
        return np.asarray(
            [
                float(np.mean(supported_prefixes)),
                (
                    float(np.mean([row.primary_adequate is True for row in supported_rows]))
                    if supported_rows
                    else 0.0
                ),
                (
                    float(np.mean([row.adequate is True for row in supported_rows]))
                    if supported_rows
                    else 0.0
                ),
                *action_adequacy,
                (
                    float(np.quantile(scaled_errors, 0.50, method="higher"))
                    if scaled_errors
                    else 0.0
                ),
                (
                    float(np.quantile(scaled_errors, 0.95, method="higher"))
                    if scaled_errors
                    else 0.0
                ),
                max(scaled_errors, default=0.0),
            ],
            dtype=np.float64,
        )

    prediction_observed = prediction_metrics(unit_ids)
    prediction_bootstrap = np.asarray(
        [prediction_metrics([unit_ids[int(index)] for index in row]) for row in indices],
        dtype=np.float64,
    )
    prediction_intervals = _simultaneous_intervals(
        prediction_observed,
        prediction_bootstrap,
        family_alpha=float(config.response_law_evaluation_family_alpha),
    )
    supported_interval, primary_interval, full_interval = prediction_intervals[:3]
    diagnostic_names = (
        "adequate_fraction_minus",
        "adequate_fraction_plus",
        "q50_receiver_scaled_error",
        "q95_receiver_scaled_error",
        "maximum_receiver_scaled_error",
    )
    diagnostic_intervals = dict(zip(diagnostic_names, prediction_intervals[3:], strict=True))
    if not material_actions:
        verdict = "RESPONSE_LAW_NO_CAUSAL_RESPONSE"
    elif supported_interval.lower < 0.80:
        verdict = "RESPONSE_LAW_DENOMINATOR_LOCAL_MIXED"
    elif primary_interval.lower < 0.90 or full_interval.lower < 0.85:
        verdict = "RESPONSE_LAW_COMPLETE_STATE_RESPONSE_ONLY"
    else:
        verdict = "RESPONSE_LAW_RECORD_CHART_SUPPORTED"
    return ResponseLawEvaluationResult(
        verdict=verdict,
        proxy_verdict=proxy_verdict,
        causal_response_intervals=intervals,
        supported_fraction_interval=supported_interval,
        primary_adequate_fraction_interval=primary_interval,
        full_adequate_fraction_interval=full_interval,
        diagnostic_intervals=diagnostic_intervals,
        reason_codes=(),
    )


__all__ = [
    "ObservationOrderResult",
    "ResponseLawEvaluationResult",
    "SimultaneousInterval",
    "adjudicate_observation_order",
    "adjudicate_response_law",
    "bootstrap_total_covariance_deltas",
]

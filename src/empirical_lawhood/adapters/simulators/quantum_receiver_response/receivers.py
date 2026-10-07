"""Causal jump-record charts and finite receiver reductions for quantum receiver response."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from .contracts import Action, preparation_weight
from .source import EventRecord, FutureDraw, PrefixCheckpoint


@dataclass(frozen=True, slots=True)
class TrajectoryReceiver:
    site_counts: NDArray[np.float64]
    n_a: float
    n_b: float
    n_total: float
    burst_a: float
    burst_b: float
    fourier_c1: float
    fourier_s1: float


@dataclass(frozen=True, slots=True)
class ReceiverOperator:
    mean_site_counts: NDArray[np.float64]
    covariance: NDArray[np.float64]
    activity: NDArray[np.float64]
    g_n: NDArray[np.float64]
    g_perp: NDArray[np.float64]
    v_anom_a: float
    fano_a: float
    chi_a: float
    chi_total: float
    cancellation_residual: float
    uniform_null_norm: float
    quadratic_form_error: float


@dataclass(frozen=True, slots=True)
class ActionReceiver:
    action: Action
    operator: ReceiverOperator
    within_covariance: NDArray[np.float64]
    between_covariance: NDArray[np.float64]
    mean_burst_a: float
    mean_burst_b: float
    mean_q_a: float
    mean_work_abs: float


def trajectory_receiver(
    events: Sequence[EventRecord],
    *,
    start_time: float,
    end_time: float,
    l_sites: int,
    particles: int,
    gamma: float,
) -> TrajectoryReceiver:
    if not start_time < end_time:
        raise ValueError("receiver window is empty")
    selected = tuple(event for event in events if start_time <= event.event_time < end_time)
    if any(event.site >= l_sites for event in selected):
        raise ValueError("receiver event leaves the chain")
    counts = np.bincount(
        [event.site for event in selected],
        minlength=l_sites,
    ).astype(np.float64)
    half = l_sites // 2
    burst_a = 0
    burst_b = 0
    delta_b = math.log(2.0) / (gamma * particles)
    for left, right in zip(selected, selected[1:], strict=False):
        if right.event_time - left.event_time <= delta_b:
            if left.site < half and right.site < half:
                burst_a += 1
            if left.site >= half and right.site >= half:
                burst_b += 1
    sites = np.asarray([event.site for event in selected], dtype=np.float64)
    angles = 2.0 * math.pi * sites / l_sites
    return TrajectoryReceiver(
        site_counts=counts,
        n_a=float(counts[:half].sum()),
        n_b=float(counts[half:].sum()),
        n_total=float(counts.sum()),
        burst_a=float(burst_a),
        burst_b=float(burst_b),
        fourier_c1=float(np.cos(angles).sum()) if len(angles) else 0.0,
        fourier_s1=float(np.sin(angles).sum()) if len(angles) else 0.0,
    )


def prefix_receivers(
    prefix: PrefixCheckpoint,
    *,
    gamma: float,
    particles: int,
    l_sites: int,
    horizons: Sequence[float],
) -> dict[float, TrajectoryReceiver]:
    return {
        horizon: trajectory_receiver(
            prefix.history_events,
            start_time=prefix.cutoff_time - horizon,
            end_time=prefix.cutoff_time,
            l_sites=l_sites,
            particles=particles,
            gamma=gamma,
        )
        for horizon in horizons
    }


def future_receiver(
    draw: FutureDraw,
    *,
    gamma: float,
    particles: int,
    l_sites: int,
) -> TrajectoryReceiver:
    return trajectory_receiver(
        draw.events,
        start_time=draw.realized_action_start,
        end_time=draw.realized_action_end,
        l_sites=l_sites,
        particles=particles,
        gamma=gamma,
    )


def _target_weights(
    k_values: Sequence[int],
    *,
    l_sites: int,
    particles: int,
) -> NDArray[np.float64]:
    counts = {value: k_values.count(value) for value in set(k_values)}
    if set(counts) != set(range(particles + 1)):
        raise ValueError("receiver estimator lacks a preparation stratum")
    weights = np.asarray(
        [
            float(preparation_weight(l_sites, particles, value)) / counts[value]
            for value in k_values
        ],
        dtype=np.float64,
    )
    weights /= weights.sum()
    return weights


def stratified_mean_covariance(
    values: NDArray[np.float64],
    k_values: Sequence[int],
    *,
    l_sites: int,
    particles: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Estimate target-population moments from an oversampled stratum roster."""

    if values.ndim != 2 or values.shape[0] != len(k_values):
        raise ValueError("stratified matrix shape differs from preparation strata")
    stratum_means: dict[int, NDArray[np.float64]] = {}
    stratum_covariances: dict[int, NDArray[np.float64]] = {}
    for k_left in range(particles + 1):
        local = values[np.asarray(k_values) == k_left]
        if len(local) < 2:
            raise ValueError("stratum covariance requires at least two units")
        stratum_means[k_left] = np.asarray(local.mean(axis=0), dtype=np.float64)
        stratum_covariances[k_left] = np.atleast_2d(
            np.asarray(np.cov(local, rowvar=False, ddof=1), dtype=np.float64)
        )
    mean = sum(
        float(preparation_weight(l_sites, particles, k_left)) * stratum_means[k_left]
        for k_left in range(particles + 1)
    )
    covariance = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    for k_left in range(particles + 1):
        weight = float(preparation_weight(l_sites, particles, k_left))
        delta = stratum_means[k_left] - mean
        covariance += weight * (stratum_covariances[k_left] + np.outer(delta, delta))
    return np.asarray(mean), covariance


def receiver_operator(
    site_counts: NDArray[np.float64],
    k_values: Sequence[int],
    *,
    l_sites: int,
    particles: int,
) -> ReceiverOperator:
    mean, covariance = stratified_mean_covariance(
        site_counts,
        k_values,
        l_sites=l_sites,
        particles=particles,
    )
    activity = np.diag(mean)
    g_n = covariance - activity
    uniform = np.ones(l_sites, dtype=np.float64)
    projection = np.eye(l_sites) - np.outer(uniform, uniform) / l_sites
    g_perp = projection @ g_n @ projection
    half_indicator = np.zeros(l_sites, dtype=np.float64)
    half_indicator[: l_sites // 2] = 1.0
    mean_a = float(half_indicator @ mean)
    var_a = float(half_indicator @ covariance @ half_indicator)
    v_anom_a = var_a - mean_a
    quadratic = float(half_indicator @ g_n @ half_indicator)
    mean_total = float(uniform @ mean)
    var_total = float(uniform @ covariance @ uniform)
    chi_total = (var_total - mean_total) / mean_total
    complement = uniform - half_indicator
    covariance_ab = float(half_indicator @ covariance @ complement)
    cancellation = (covariance_ab + v_anom_a) / mean_a
    fano = var_a / mean_a
    return ReceiverOperator(
        mean_site_counts=mean,
        covariance=covariance,
        activity=activity,
        g_n=g_n,
        g_perp=g_perp,
        v_anom_a=v_anom_a,
        fano_a=fano,
        chi_a=fano - 1.0,
        chi_total=chi_total,
        cancellation_residual=cancellation,
        uniform_null_norm=float(np.linalg.norm(g_n @ uniform) / mean_total),
        quadratic_form_error=abs(v_anom_a - quadratic),
    )


def excess_variance_contributions(
    n_a: Sequence[float],
    k_values: Sequence[int],
) -> NDArray[np.float64]:
    values = np.asarray(n_a, dtype=np.float64)
    strata = np.asarray(k_values, dtype=np.int64)
    if len(values) != len(strata):
        raise ValueError("receiver contribution rows differ")
    result = np.empty(len(values), dtype=np.float64)
    for k_left in np.unique(strata):
        local_indices = np.flatnonzero(strata == k_left)
        if len(local_indices) < 2:
            raise ValueError("cross-fit receiver contribution lacks a donor")
        local_values = values[local_indices]
        leave_one_out_means = (float(local_values.sum()) - local_values) / (len(local_values) - 1)
        result[local_indices] = (local_values - leave_one_out_means) ** 2 - local_values
    return result


def weighted_spearman(
    left: Sequence[float],
    right: Sequence[float],
    k_values: Sequence[int],
    *,
    l_sites: int,
    particles: int,
) -> float:
    if not (len(left) == len(right) == len(k_values)):
        raise ValueError("weighted Spearman rows differ")
    left_rank = rankdata(np.asarray(left, dtype=np.float64), method="average")
    right_rank = rankdata(np.asarray(right, dtype=np.float64), method="average")
    weights = _target_weights(
        list(k_values),
        l_sites=l_sites,
        particles=particles,
    )
    left_mean = float(weights @ left_rank)
    right_mean = float(weights @ right_rank)
    left_centered = left_rank - left_mean
    right_centered = right_rank - right_mean
    denominator = math.sqrt(
        float(weights @ (left_centered**2)) * float(weights @ (right_centered**2))
    )
    if denominator <= 0.0:
        return float("nan")
    return float(weights @ (left_centered * right_centered) / denominator)


def causal_chart(
    prefix: PrefixCheckpoint,
    *,
    gamma: float,
    l_sites: int,
    particles: int,
) -> dict[str, float | str]:
    cutoff = prefix.cutoff_time
    h_s = 4.0 / (gamma * particles)
    h_l = 16.0 / (gamma * particles)
    short = tuple(event for event in prefix.history_events if event.event_time >= cutoff - h_s)
    long = tuple(event for event in prefix.history_events if event.event_time >= cutoff - h_l)
    half = l_sites // 2

    def region(site: int) -> str:
        return "A" if site < half else "B"

    last = long[-1] if long else None
    long_angles = np.asarray(
        [2.0 * math.pi * event.site / l_sites for event in long],
        dtype=np.float64,
    )
    boundary_sites = {0, half - 1, half, l_sites - 1}
    boundary_events = [event for event in long if event.site in boundary_sites]
    transitions_ab = 0
    transitions_ba = 0
    for left, right in zip(long, long[1:], strict=False):
        pair = (region(left.site), region(right.site))
        transitions_ab += pair == ("A", "B")
        transitions_ba += pair == ("B", "A")
    coordinates: dict[str, float | str] = {
        "count_A(h_s)": float(sum(event.site < half for event in short)),
        "count_B(h_s)": float(sum(event.site >= half for event in short)),
        "time_since_last_event": (cutoff - last.event_time if last is not None else h_l),
        "last_event_region": region(last.site) if last is not None else "none",
        "sin_last_site": (
            math.sin(2.0 * math.pi * last.site / l_sites) if last is not None else 0.0
        ),
        "cos_last_site": (
            math.cos(2.0 * math.pi * last.site / l_sites) if last is not None else 0.0
        ),
        "sum_cos(h_l)": float(np.cos(long_angles).sum()) if len(long) else 0.0,
        "sum_sin(h_l)": float(np.sin(long_angles).sum()) if len(long) else 0.0,
        "count_A(h_l_without_h_s)": float(
            sum(event.site < half and event.event_time < cutoff - h_s for event in long)
        ),
        "count_B(h_l_without_h_s)": float(
            sum(event.site >= half and event.event_time < cutoff - h_s for event in long)
        ),
        "time_since_last_boundary_event": (
            cutoff - boundary_events[-1].event_time if boundary_events else h_l
        ),
        "transitions_A_to_B": float(transitions_ab),
        "transitions_B_to_A": float(transitions_ba),
    }
    return coordinates


CHART_COORDINATES: Mapping[str, tuple[str, ...]] = {
    "regional-activity": (
        "count_A(h_s)",
        "count_B(h_s)",
        "time_since_last_event",
        "last_event_region",
    ),
    "spatial-activity": (
        "count_A(h_s)",
        "count_B(h_s)",
        "time_since_last_event",
        "last_event_region",
        "sin_last_site",
        "cos_last_site",
        "sum_cos(h_l)",
        "sum_sin(h_l)",
    ),
    "temporal-memory": (
        "count_A(h_s)",
        "count_B(h_s)",
        "time_since_last_event",
        "last_event_region",
        "sin_last_site",
        "cos_last_site",
        "sum_cos(h_l)",
        "sum_sin(h_l)",
        "count_A(h_l_without_h_s)",
        "count_B(h_l_without_h_s)",
        "time_since_last_boundary_event",
        "transitions_A_to_B",
        "transitions_B_to_A",
    ),
}


def action_receiver(
    *,
    action: Action,
    draws_by_prefix: Mapping[str, Sequence[FutureDraw]],
    prefix_k: Mapping[str, int],
    gamma: float,
    l_sites: int,
    particles: int,
) -> ActionReceiver:
    prefix_ids = sorted(draws_by_prefix)
    if set(prefix_ids) != set(prefix_k):
        raise ValueError("action receiver prefix/stratum identities differ")
    conditional_means = []
    conditional_covariances = []
    burst_a = []
    burst_b = []
    q_a = []
    work = []
    for prefix_id in prefix_ids:
        draws = tuple(draws_by_prefix[prefix_id])
        if len(draws) < 2 or any(draw.action is not action for draw in draws):
            raise ValueError("action receiver needs at least two same-action draws")
        local = [
            future_receiver(
                draw,
                gamma=gamma,
                particles=particles,
                l_sites=l_sites,
            )
            for draw in draws
        ]
        matrix = np.asarray([row.site_counts for row in local], dtype=np.float64)
        conditional_means.append(matrix.mean(axis=0))
        conditional_covariances.append(np.cov(matrix, rowvar=False, ddof=1))
        burst_a.append(float(np.mean([row.burst_a for row in local])))
        burst_b.append(float(np.mean([row.burst_b for row in local])))
        q_a.append(float(np.mean([draw.terminal_q_a for draw in draws])))
        work.append(float(np.mean([draw.work_abs for draw in draws])))
    k_values = [prefix_k[prefix_id] for prefix_id in prefix_ids]
    mean_matrix = np.asarray(conditional_means, dtype=np.float64)
    population_mean, between = stratified_mean_covariance(
        mean_matrix,
        k_values,
        l_sites=l_sites,
        particles=particles,
    )
    within = np.zeros((l_sites, l_sites), dtype=np.float64)
    for k_left in range(particles + 1):
        local_indices = [index for index, value in enumerate(k_values) if value == k_left]
        within += float(preparation_weight(l_sites, particles, k_left)) * np.mean(
            [conditional_covariances[index] for index in local_indices],
            axis=0,
        )
    total = within + between
    # Reconstruct an operator without discarding the total-covariance decomposition.
    activity = np.diag(population_mean)
    g_n = total - activity
    uniform = np.ones(l_sites, dtype=np.float64)
    projection = np.eye(l_sites) - np.outer(uniform, uniform) / l_sites
    half_indicator = np.zeros(l_sites, dtype=np.float64)
    half_indicator[: l_sites // 2] = 1.0
    complement = uniform - half_indicator
    mean_a = float(half_indicator @ population_mean)
    var_a = float(half_indicator @ total @ half_indicator)
    v_anom = var_a - mean_a
    mean_total = float(uniform @ population_mean)
    var_total = float(uniform @ total @ uniform)
    operator = ReceiverOperator(
        mean_site_counts=population_mean,
        covariance=total,
        activity=activity,
        g_n=g_n,
        g_perp=projection @ g_n @ projection,
        v_anom_a=v_anom,
        fano_a=var_a / mean_a,
        chi_a=var_a / mean_a - 1.0,
        chi_total=(var_total - mean_total) / mean_total,
        cancellation_residual=(float(half_indicator @ total @ complement) + v_anom) / mean_a,
        uniform_null_norm=float(np.linalg.norm(g_n @ uniform) / mean_total),
        quadratic_form_error=abs(v_anom - float(half_indicator @ g_n @ half_indicator)),
    )
    weights = _target_weights(
        k_values,
        l_sites=l_sites,
        particles=particles,
    )
    return ActionReceiver(
        action=action,
        operator=operator,
        within_covariance=within,
        between_covariance=between,
        mean_burst_a=float(weights @ np.asarray(burst_a)),
        mean_burst_b=float(weights @ np.asarray(burst_b)),
        mean_q_a=float(weights @ np.asarray(q_a)),
        mean_work_abs=float(weights @ np.asarray(work)),
    )


__all__ = [
    "ActionReceiver",
    "CHART_COORDINATES",
    "ReceiverOperator",
    "TrajectoryReceiver",
    "action_receiver",
    "causal_chart",
    "excess_variance_contributions",
    "future_receiver",
    "prefix_receivers",
    "receiver_operator",
    "stratified_mean_covariance",
    "trajectory_receiver",
    "weighted_spearman",
]

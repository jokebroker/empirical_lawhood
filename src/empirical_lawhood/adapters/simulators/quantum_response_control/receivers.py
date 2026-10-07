"""Receiver reductions and parent-level population moments for quantum response control."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from .contracts import Action, PreparationUnit, preparation_weight
from .source import EventRecord, FutureDraw, PrefixCheckpoint


@dataclass(frozen=True, slots=True)
class TrajectoryReceiver:
    site_counts: NDArray[np.float64]
    n_a: float
    n_a_squared: float
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
    mean_energy: float
    mean_work_abs: float
    mean_transport_signed: float
    mean_transport_absolute: float
    mean_measurement_innovation: float
    maximum_continuity_residual: float
    transport_valid: bool


def trajectory_receiver(
    events: Sequence[EventRecord],
    *,
    start_time: float,
    end_time: float,
    l_sites: int,
    particles: int,
    gamma: float,
    receiver_shift: int = 0,
) -> TrajectoryReceiver:
    if not start_time < end_time:
        raise ValueError("receiver window is empty")
    if not 0 <= receiver_shift < l_sites:
        raise ValueError("receiver translation leaves the chain")
    selected = tuple(event for event in events if start_time <= event.event_time < end_time)
    if any(not 0 <= event.site < l_sites for event in selected):
        raise ValueError("receiver event leaves the chain")
    translated_sites = [int((event.site - receiver_shift) % l_sites) for event in selected]
    counts = np.bincount(translated_sites, minlength=l_sites).astype(np.float64)
    half = l_sites // 2
    burst_a = 0
    burst_b = 0
    delta_b = math.log(2.0) / (gamma * particles)
    for index, (left, right) in enumerate(zip(selected, selected[1:], strict=False)):
        if right.event_time - left.event_time > delta_b:
            continue
        left_site = translated_sites[index]
        right_site = translated_sites[index + 1]
        if left_site < half and right_site < half:
            burst_a += 1
        if left_site >= half and right_site >= half:
            burst_b += 1
    sites = np.asarray(translated_sites, dtype=np.float64)
    angles = 2.0 * math.pi * sites / l_sites
    n_a = float(counts[:half].sum())
    return TrajectoryReceiver(
        site_counts=counts,
        n_a=n_a,
        n_a_squared=n_a * n_a,
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
) -> dict[str, TrajectoryReceiver]:
    windows = {
        "h20": (prefix.cutoff_time - 20.0, prefix.cutoff_time),
        "h4": (prefix.cutoff_time - 4.0, prefix.cutoff_time),
        "h4_previous": (prefix.cutoff_time - 8.0, prefix.cutoff_time - 4.0),
    }
    return {
        key: trajectory_receiver(
            prefix.history_events,
            start_time=start,
            end_time=end,
            l_sites=l_sites,
            particles=particles,
            gamma=gamma,
        )
        for key, (start, end) in windows.items()
    }


def future_receiver(
    draw: FutureDraw,
    *,
    gamma: float,
    particles: int,
    l_sites: int,
    receiver_shift: int = 0,
) -> TrajectoryReceiver:
    return trajectory_receiver(
        draw.events,
        start_time=draw.ledger.realized_start,
        end_time=draw.ledger.realized_end,
        l_sites=l_sites,
        particles=particles,
        gamma=gamma,
        receiver_shift=receiver_shift,
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


def target_weighted_mean(
    values: Sequence[float],
    k_values: Sequence[int],
    *,
    l_sites: int = 12,
    particles: int = 6,
) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (len(k_values),):
        raise ValueError("target-weighted mean rows differ")
    return float(_target_weights(k_values, l_sites=l_sites, particles=particles) @ array)


def stratified_mean_covariance(
    values: NDArray[np.float64],
    k_values: Sequence[int],
    *,
    l_sites: int,
    particles: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Target-population mean/covariance with parent as independent unit."""

    if values.ndim != 2 or values.shape[0] != len(k_values):
        raise ValueError("stratified matrix shape differs from preparation strata")
    strata = np.asarray(k_values, dtype=np.int64)
    stratum_means: dict[int, NDArray[np.float64]] = {}
    stratum_covariances: dict[int, NDArray[np.float64]] = {}
    for k_left in range(particles + 1):
        local = values[strata == k_left]
        if len(local) < 2:
            raise ValueError("stratum covariance requires at least two parents")
        stratum_means[k_left] = np.asarray(local.mean(axis=0), dtype=np.float64)
        covariance = np.cov(local, rowvar=False, ddof=1)
        stratum_covariances[k_left] = np.atleast_2d(np.asarray(covariance, dtype=np.float64))
    mean = sum(
        float(preparation_weight(l_sites, particles, k_left)) * stratum_means[k_left]
        for k_left in range(particles + 1)
    )
    covariance = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    for k_left in range(particles + 1):
        weight = float(preparation_weight(l_sites, particles, k_left))
        delta = stratum_means[k_left] - mean
        covariance += weight * (stratum_covariances[k_left] + np.outer(delta, delta))
    return np.asarray(mean, dtype=np.float64), covariance


def stratified_total_covariance(
    parent_means: NDArray[np.float64],
    parent_within_covariances: NDArray[np.float64],
    k_values: Sequence[int],
    *,
    l_sites: int,
    particles: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    if parent_means.ndim != 2:
        raise ValueError("parent mean matrix must be two-dimensional")
    expected = (len(parent_means), parent_means.shape[1], parent_means.shape[1])
    if parent_within_covariances.shape != expected:
        raise ValueError("within-parent covariance shape differs")
    mean, between = stratified_mean_covariance(
        parent_means,
        k_values,
        l_sites=l_sites,
        particles=particles,
    )
    weights = _target_weights(k_values, l_sites=l_sites, particles=particles)
    within = np.tensordot(
        weights,
        parent_within_covariances,
        axes=(0, 0),
    )
    return mean, np.asarray(within), np.asarray(between)


def receiver_operator_from_moments(
    mean: NDArray[np.float64],
    covariance: NDArray[np.float64],
) -> ReceiverOperator:
    if mean.ndim != 1 or covariance.shape != (len(mean), len(mean)):
        raise ValueError("receiver moments have incompatible shapes")
    l_sites = len(mean)
    activity = np.diag(mean)
    g_n = covariance - activity
    uniform = np.ones(l_sites, dtype=np.float64)
    projection = np.eye(l_sites) - np.outer(uniform, uniform) / l_sites
    g_perp = projection @ g_n @ projection
    half_indicator = np.zeros(l_sites, dtype=np.float64)
    half_indicator[: l_sites // 2] = 1.0
    mean_a = float(half_indicator @ mean)
    mean_total = float(uniform @ mean)
    if mean_a <= 0 or mean_total <= 0:
        raise ValueError("receiver mean activity is not positive")
    var_a = float(half_indicator @ covariance @ half_indicator)
    v_anom_a = var_a - mean_a
    quadratic = float(half_indicator @ g_n @ half_indicator)
    var_total = float(uniform @ covariance @ uniform)
    complement = uniform - half_indicator
    covariance_ab = float(half_indicator @ covariance @ complement)
    return ReceiverOperator(
        mean_site_counts=mean,
        covariance=covariance,
        activity=activity,
        g_n=g_n,
        g_perp=g_perp,
        v_anom_a=v_anom_a,
        fano_a=var_a / mean_a,
        chi_a=var_a / mean_a - 1.0,
        chi_total=(var_total - mean_total) / mean_total,
        cancellation_residual=(covariance_ab + v_anom_a) / mean_a,
        uniform_null_norm=float(np.linalg.norm(g_n @ uniform) / mean_total),
        quadratic_form_error=abs(v_anom_a - quadratic),
    )


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
    return receiver_operator_from_moments(mean, covariance)


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
        indices = np.flatnonzero(strata == k_left)
        if len(indices) < 2:
            raise ValueError("cross-fit receiver contribution lacks a donor")
        local = values[indices]
        leave_one_out = (float(local.sum()) - local) / (len(local) - 1)
        result[indices] = (local - leave_one_out) ** 2 - local
    return result


def weighted_spearman(
    left: Sequence[float],
    right: Sequence[float],
    k_values: Sequence[int],
    *,
    l_sites: int = 12,
    particles: int = 6,
) -> float:
    if not (len(left) == len(right) == len(k_values)):
        raise ValueError("weighted Spearman rows differ")
    left_rank = rankdata(np.asarray(left, dtype=np.float64), method="average")
    right_rank = rankdata(np.asarray(right, dtype=np.float64), method="average")
    weights = _target_weights(k_values, l_sites=l_sites, particles=particles)
    left_centered = left_rank - float(weights @ left_rank)
    right_centered = right_rank - float(weights @ right_rank)
    denominator = math.sqrt(
        float(weights @ (left_centered**2)) * float(weights @ (right_centered**2))
    )
    if denominator <= 0:
        return 0.0
    return float(weights @ (left_centered * right_centered) / denominator)


def action_receiver(
    *,
    action: Action,
    draws_by_parent: Mapping[str, Sequence[FutureDraw]],
    units_by_id: Mapping[str, PreparationUnit],
    gamma: float,
    l_sites: int,
    particles: int,
) -> ActionReceiver:
    parent_ids = sorted(draws_by_parent)
    if set(parent_ids) - set(units_by_id):
        raise ValueError("action receiver contains an unknown parent")
    parent_means: list[NDArray[np.float64]] = []
    parent_covariances: list[NDArray[np.float64]] = []
    k_values: list[int] = []
    scalar_rows: dict[str, list[float]] = {
        key: []
        for key in (
            "burst_a",
            "burst_b",
            "q_a",
            "energy",
            "work_abs",
            "transport_signed",
            "transport_absolute",
            "measurement_innovation",
        )
    }
    maximum_residual = 0.0
    transport_valid = True
    for parent_id in parent_ids:
        draws = tuple(draws_by_parent[parent_id])
        if len(draws) < 2 or any(draw.action is not action for draw in draws):
            raise ValueError("action receiver requires repeated matched-action draws")
        receiver_rows = [
            future_receiver(
                draw,
                gamma=gamma,
                particles=particles,
                l_sites=l_sites,
            )
            for draw in draws
        ]
        counts = np.asarray([row.site_counts for row in receiver_rows], dtype=np.float64)
        parent_means.append(counts.mean(axis=0))
        parent_covariances.append(np.asarray(np.cov(counts, rowvar=False, ddof=1)))
        k_values.append(units_by_id[parent_id].k_left)
        scalar_rows["burst_a"].append(float(np.mean([row.burst_a for row in receiver_rows])))
        scalar_rows["burst_b"].append(float(np.mean([row.burst_b for row in receiver_rows])))
        scalar_rows["q_a"].append(float(np.mean([draw.terminal_q_a for draw in draws])))
        scalar_rows["energy"].append(float(np.mean([draw.terminal_energy for draw in draws])))
        scalar_rows["work_abs"].append(float(np.mean([draw.work_abs for draw in draws])))
        scalar_rows["transport_signed"].append(
            float(np.mean([draw.transport.coherent_signed for draw in draws]))
        )
        scalar_rows["transport_absolute"].append(
            float(np.mean([draw.transport.coherent_absolute for draw in draws]))
        )
        scalar_rows["measurement_innovation"].append(
            float(np.mean([draw.transport.measurement_innovation for draw in draws]))
        )
        maximum_residual = max(
            maximum_residual,
            max(abs(draw.transport.continuity_residual) for draw in draws),
        )
        transport_valid = transport_valid and all(draw.transport.valid for draw in draws)
    mean, within, between = stratified_total_covariance(
        np.asarray(parent_means),
        np.asarray(parent_covariances),
        k_values,
        l_sites=l_sites,
        particles=particles,
    )
    operator = receiver_operator_from_moments(mean, within + between)

    def weighted(key: str) -> float:
        return target_weighted_mean(
            scalar_rows[key],
            k_values,
            l_sites=l_sites,
            particles=particles,
        )

    return ActionReceiver(
        action=action,
        operator=operator,
        within_covariance=within,
        between_covariance=between,
        mean_burst_a=weighted("burst_a"),
        mean_burst_b=weighted("burst_b"),
        mean_q_a=weighted("q_a"),
        mean_energy=weighted("energy"),
        mean_work_abs=weighted("work_abs"),
        mean_transport_signed=weighted("transport_signed"),
        mean_transport_absolute=weighted("transport_absolute"),
        mean_measurement_innovation=weighted("measurement_innovation"),
        maximum_continuity_residual=maximum_residual,
        transport_valid=transport_valid,
    )


__all__ = [
    "ActionReceiver",
    "ReceiverOperator",
    "TrajectoryReceiver",
    "action_receiver",
    "excess_variance_contributions",
    "future_receiver",
    "prefix_receivers",
    "receiver_operator",
    "receiver_operator_from_moments",
    "stratified_mean_covariance",
    "stratified_total_covariance",
    "target_weighted_mean",
    "trajectory_receiver",
    "weighted_spearman",
]

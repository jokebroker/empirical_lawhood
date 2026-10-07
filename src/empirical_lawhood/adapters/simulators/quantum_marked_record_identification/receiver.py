"""Population receiver, bridge target, and weighted rank contracts for quantum marked record identification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata

from .contracts import preparation_weight, target_weights
from .source import EventRecord


@dataclass(frozen=True, slots=True)
class ReceiverOperator:
    mean_site_counts: NDArray[np.float64]
    covariance: NDArray[np.float64]
    g_n: NDArray[np.float64]
    g_perp: NDArray[np.float64]
    v_anom_a: float
    fano_a: float
    chi_a: float
    chi_total: float
    cancellation_residual: float
    uniform_null_norm: float
    quadratic_form_error: float


def site_counts(
    events: Sequence[EventRecord],
    *,
    start_time: float,
    end_time: float,
    l_sites: int = 12,
) -> NDArray[np.float64]:
    if not start_time < end_time:
        raise ValueError("receiver window is invalid")
    counts = np.zeros(l_sites, dtype=np.float64)
    for event in events:
        if start_time <= event.event_time < end_time:
            counts[event.site] += 1.0
    return counts


def stratified_mean_covariance(
    values: NDArray[np.float64],
    k_values: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    matrix = np.asarray(values, dtype=np.float64)
    strata = np.asarray(k_values, dtype=np.int64)
    if matrix.ndim != 2 or strata.shape != (matrix.shape[0],):
        raise ValueError("stratified moment rows differ")
    if matrix.shape[0] < 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("stratified moments require finite replicated rows")
    means: dict[int, NDArray[np.float64]] = {}
    covariances: dict[int, NDArray[np.float64]] = {}
    for k_left in range(7):
        local = matrix[strata == k_left]
        if len(local) < 2:
            raise ValueError("stratum covariance requires at least two parents")
        means[k_left] = np.asarray(local.mean(axis=0), dtype=np.float64)
        covariances[k_left] = np.atleast_2d(
            np.asarray(np.cov(local, rowvar=False, ddof=1), dtype=np.float64)
        )
    mean = sum(stratum_weight(k_left) * means[k_left] for k_left in range(7))
    covariance = np.zeros((matrix.shape[1], matrix.shape[1]), dtype=np.float64)
    for k_left in range(7):
        delta = means[k_left] - mean
        covariance += stratum_weight(k_left) * (covariances[k_left] + np.outer(delta, delta))
    return np.asarray(mean), covariance


def receiver_operator(
    counts: NDArray[np.float64],
    k_values: Sequence[int],
) -> ReceiverOperator:
    mean, covariance = stratified_mean_covariance(counts, k_values)
    uniform = np.ones(counts.shape[1], dtype=np.float64)
    projector = np.eye(counts.shape[1]) - np.outer(uniform, uniform) / float(uniform @ uniform)
    g_n = covariance - np.diag(mean)
    g_perp = projector @ g_n @ projector
    half = np.zeros(counts.shape[1], dtype=np.float64)
    half[: counts.shape[1] // 2] = 1.0
    complement = uniform - half
    mean_a = float(half @ mean)
    mean_total = float(uniform @ mean)
    var_a = float(half @ covariance @ half)
    var_total = float(uniform @ covariance @ uniform)
    v_anom = var_a - mean_a
    covariance_ab = float(half @ covariance @ complement)
    if mean_a <= 0 or mean_total <= 0:
        raise ValueError("receiver mean activity is zero")
    quadratic = float(half @ g_n @ half)
    return ReceiverOperator(
        mean_site_counts=mean,
        covariance=covariance,
        g_n=g_n,
        g_perp=g_perp,
        v_anom_a=v_anom,
        fano_a=var_a / mean_a,
        chi_a=v_anom / mean_a,
        chi_total=(var_total - mean_total) / mean_total,
        cancellation_residual=(covariance_ab + v_anom) / mean_a,
        uniform_null_norm=float(np.linalg.norm(g_n @ uniform) / mean_total),
        quadratic_form_error=abs(v_anom - quadratic),
    )


def fold_assignments(
    unit_ids: Sequence[str],
    k_values: Sequence[int],
    *,
    folds: int = 8,
    scientific_fold_order: tuple[tuple[str, int], ...],
) -> NDArray[np.int64]:
    if len(unit_ids) != len(k_values) or folds < 2:
        raise ValueError("fold inputs differ")
    if (
        not isinstance(scientific_fold_order, tuple)
        or len(scientific_fold_order) != len(unit_ids)
        or any(not isinstance(row, tuple) or len(row) != 2 for row in scientific_fold_order)
        or tuple(row[0] for row in scientific_fold_order) != tuple(unit_ids)
        or any(type(row[1]) is not int or row[1] < 0 for row in scientific_fold_order)
        or len({row[1] for row in scientific_fold_order}) != len(unit_ids)
        or len(set(unit_ids)) != len(unit_ids)
    ):
        raise ValueError("scientific fold order differs from the exact parent roster")
    result = np.empty(len(unit_ids), dtype=np.int64)
    for k_left in range(7):
        indices = [i for i, value in enumerate(k_values) if value == k_left]
        if not indices:
            continue
        if len(indices) < folds:
            raise ValueError("stratum cannot populate all folds")
        ordered = sorted(
            indices,
            key=lambda i: scientific_fold_order[i][1],
        )
        for local, index in enumerate(ordered):
            result[index] = local % folds
    return result


def bridge_target(
    n_a: Sequence[float],
    k_values: Sequence[int],
    folds: Sequence[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    values = np.asarray(n_a, dtype=np.float64)
    strata = np.asarray(k_values, dtype=np.int64)
    fold_values = np.asarray(folds, dtype=np.int64)
    if not (values.shape == strata.shape == fold_values.shape):
        raise ValueError("bridge rows differ")
    target = np.empty_like(values)
    donor_means = np.empty_like(values)
    for k_left in np.unique(strata):
        local = np.flatnonzero(strata == k_left)
        for fold in np.unique(fold_values[local]):
            selected = local[fold_values[local] == fold]
            donors = local[fold_values[local] != fold]
            if not len(selected) or not len(donors):
                raise ValueError("bridge target lacks fold-separated donors")
            donor = float(np.mean(values[donors]))
            donor_means[selected] = donor
            target[selected] = (values[selected] - donor) ** 2 - values[selected]
    return target, donor_means


def weighted_midrank_correlation(
    left: Sequence[float],
    right: Sequence[float],
    k_values: Sequence[int],
    *,
    require_all_strata: bool = True,
) -> float:
    if not (len(left) == len(right) == len(k_values)):
        raise ValueError("weighted correlation rows differ")
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if not np.all(np.isfinite(left_array)) or not np.all(np.isfinite(right_array)):
        raise ValueError("weighted correlation received nonfinite rows")
    left_rank = rankdata(left_array, method="average")
    right_rank = rankdata(right_array, method="average")
    if require_all_strata:
        weights = np.asarray(target_weights(list(k_values)), dtype=np.float64)
    else:
        counts = {k: list(k_values).count(k) for k in set(k_values)}
        weights = np.asarray([stratum_weight(k) / counts[k] for k in k_values], dtype=np.float64)
        weights /= float(weights.sum())
    left_centered = left_rank - float(weights @ left_rank)
    right_centered = right_rank - float(weights @ right_rank)
    denominator = math.sqrt(float(weights @ left_centered**2) * float(weights @ right_centered**2))
    if denominator <= 0:
        return 0.0
    return float(weights @ (left_centered * right_centered) / denominator)


def effective_target_sample_size(k_values: Sequence[int]) -> float:
    weights = np.asarray(target_weights(list(k_values)), dtype=np.float64)
    return 1.0 / float(weights @ weights)


def crossfit_site_probabilities(
    count_rows: NDArray[np.float64],
    k_values: Sequence[int],
    folds: Sequence[int],
) -> NDArray[np.float64]:
    counts = np.asarray(count_rows, dtype=np.float64)
    strata = np.asarray(k_values, dtype=np.int64)
    fold_values = np.asarray(folds, dtype=np.int64)
    result = np.empty_like(counts)
    for k_left in np.unique(strata):
        local = np.flatnonzero(strata == k_left)
        for fold in np.unique(fold_values[local]):
            selected = local[fold_values[local] == fold]
            donors = local[fold_values[local] != fold]
            donor_counts = counts[donors].sum(axis=0) + 1.0
            result[selected] = donor_counts / float(donor_counts.sum())
    return result


def expected_thinned_receiver(
    total_counts: Sequence[int],
    site_probabilities: NDArray[np.float64],
    k_values: Sequence[int],
) -> ReceiverOperator:
    """Exact first-two-moment receiver for conditional multinomial thinning."""

    totals = np.asarray(total_counts, dtype=np.float64)
    probabilities = np.asarray(site_probabilities, dtype=np.float64)
    if probabilities.shape != (len(totals), 12):
        raise ValueError("thinning probabilities differ")
    conditional_means = totals[:, None] * probabilities
    mean, between = stratified_mean_covariance(conditional_means, k_values)
    weights = np.asarray(target_weights(list(k_values)), dtype=np.float64)
    within = np.zeros((12, 12), dtype=np.float64)
    for weight, total, probability in zip(weights, totals, probabilities, strict=True):
        within += weight * total * (np.diag(probability) - np.outer(probability, probability))
    covariance = between + within
    # Reconstruct through deterministic pseudo-rows is neither exact nor
    # necessary; apply the operator algebra directly.
    uniform = np.ones(12)
    projector = np.eye(12) - np.outer(uniform, uniform) / 12.0
    g_n = covariance - np.diag(mean)
    g_perp = projector @ g_n @ projector
    half = np.r_[np.ones(6), np.zeros(6)]
    complement = uniform - half
    mean_a = float(half @ mean)
    mean_total = float(uniform @ mean)
    var_a = float(half @ covariance @ half)
    v_anom = var_a - mean_a
    covariance_ab = float(half @ covariance @ complement)
    return ReceiverOperator(
        mean_site_counts=np.asarray(mean),
        covariance=np.asarray(covariance),
        g_n=np.asarray(g_n),
        g_perp=np.asarray(g_perp),
        v_anom_a=v_anom,
        fano_a=var_a / mean_a,
        chi_a=v_anom / mean_a,
        chi_total=(float(uniform @ covariance @ uniform) - mean_total) / mean_total,
        cancellation_residual=(covariance_ab + v_anom) / mean_a,
        uniform_null_norm=float(np.linalg.norm(g_n @ uniform) / mean_total),
        quadratic_form_error=abs(v_anom - float(half @ g_n @ half)),
    )


def stratum_weight(k_left: int) -> float:
    return float(preparation_weight(12, 6, k_left))


__all__ = [
    "ReceiverOperator",
    "bridge_target",
    "crossfit_site_probabilities",
    "effective_target_sample_size",
    "expected_thinned_receiver",
    "fold_assignments",
    "receiver_operator",
    "site_counts",
    "stratum_weight",
    "stratified_mean_covariance",
    "weighted_midrank_correlation",
]

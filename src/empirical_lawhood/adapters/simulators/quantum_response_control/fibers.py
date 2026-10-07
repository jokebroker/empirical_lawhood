"""Outcome-blind fiber pairing and interval-aware quantum response control adjudication."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix
from scipy.stats import beta

from .contracts import Action, Verdict
from .response import RESPONSE_COORDINATES, QuantumTrajectoryLawFiberEquivalence


@dataclass(frozen=True, slots=True)
class FiberCandidate:
    parent_id: str
    k_left: int
    cell_id: str
    action: Action
    chart_distance_anchor: float
    omitted_features: Mapping[str, float | str]


@dataclass(frozen=True, slots=True)
class FiberPair:
    pair_id: str
    contrast_id: str
    left_parent_id: str
    right_parent_id: str
    k_left: int
    cell_id: str
    action: Action
    chart_distance: float


@dataclass(frozen=True, slots=True)
class FiberResponse:
    parent_id: str
    action: Action
    values: Mapping[str, float]
    halfwidths: Mapping[str, float]
    valid: bool


@dataclass(frozen=True, slots=True)
class FiberPairResult:
    pair_id: str
    contrast_id: str
    action: Action
    status: str
    coordinate_status: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class FiberAdjudication:
    verdict: Verdict
    resolved_pairs: int
    diverged_pairs: int
    ambiguous_pairs: int
    divergence_rate_upper: float
    pair_results: tuple[FiberPairResult, ...]
    reason_codes: tuple[str, ...]


def _eligible_edges(
    candidates: Sequence[FiberCandidate],
    contrast_id: str,
    maximum_chart_distance: float,
) -> tuple[tuple[int, int, float], ...]:
    edges: list[tuple[int, int, float]] = []
    for left_index, left in enumerate(candidates):
        for right_index in range(left_index + 1, len(candidates)):
            right = candidates[right_index]
            if (
                left.k_left != right.k_left
                or left.cell_id != right.cell_id
                or left.action is not right.action
            ):
                continue
            if left.omitted_features[contrast_id] == right.omitted_features[contrast_id]:
                continue
            distance = abs(left.chart_distance_anchor - right.chart_distance_anchor)
            if distance <= maximum_chart_distance:
                edges.append((left_index, right_index, distance))
    return tuple(edges)


def _solve_matching(
    candidate_count: int,
    edges: Sequence[tuple[int, int, float]],
) -> tuple[int, ...]:
    """Exact maximum-cardinality, then minimum-distance graph matching."""

    if not edges:
        return ()
    rows: list[int] = []
    columns: list[int] = []
    data: list[float] = []
    for edge_index, (left, right, _distance) in enumerate(edges):
        rows.extend((left, right))
        columns.extend((edge_index, edge_index))
        data.extend((1.0, 1.0))
    incidence = csc_matrix(
        (data, (rows, columns)),
        shape=(candidate_count, len(edges)),
    )
    # One added edge must dominate the sum of all possible distance/lexical
    # tie costs; this makes cardinality the exact primary objective.
    cardinality_weight = float(len(edges) + 2)
    objective = np.asarray(
        [
            -cardinality_weight + distance + edge_index * 1e-12
            for edge_index, (_left, _right, distance) in enumerate(edges)
        ],
        dtype=np.float64,
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(edges), dtype=np.int8),
        bounds=Bounds(np.zeros(len(edges)), np.ones(len(edges))),
        constraints=LinearConstraint(
            incidence,
            lb=np.zeros(candidate_count),
            ub=np.ones(candidate_count),
        ),
        options={"presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"fiber matching failed: {result.message}")
    return tuple(index for index, value in enumerate(result.x) if value >= 0.5)


def select_fiber_pairs(
    candidates: Sequence[FiberCandidate],
    *,
    contrast_ids: Sequence[str],
    maximum_pairs: int,
    maximum_chart_distance: float = 1.0,
) -> tuple[FiberPair, ...]:
    if maximum_pairs <= 0:
        raise ValueError("fiber pair maximum must be positive")
    candidate_ids = [candidate.parent_id for candidate in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("fiber pool repeats a parent")
    remaining = list(sorted(candidates, key=lambda value: value.parent_id))
    selected: list[FiberPair] = []
    used: set[str] = set()
    for contrast_id in contrast_ids:
        local = [candidate for candidate in remaining if candidate.parent_id not in used]
        if any(contrast_id not in candidate.omitted_features for candidate in local):
            raise ValueError("fiber candidate omits a declared contrast")
        edges = _eligible_edges(local, contrast_id, maximum_chart_distance)
        chosen = _solve_matching(len(local), edges)
        ordered = sorted(
            (edges[index] for index in chosen),
            key=lambda edge: (
                edge[2],
                local[edge[0]].parent_id,
                local[edge[1]].parent_id,
            ),
        )
        for left_index, right_index, distance in ordered:
            if len(selected) >= maximum_pairs:
                break
            left = local[left_index]
            right = local[right_index]
            pair_number = len(selected)
            selected.append(
                FiberPair(
                    pair_id=f"fiber-pair.{pair_number:03d}",
                    contrast_id=contrast_id,
                    left_parent_id=min(left.parent_id, right.parent_id),
                    right_parent_id=max(left.parent_id, right.parent_id),
                    k_left=left.k_left,
                    cell_id=left.cell_id,
                    action=left.action,
                    chart_distance=distance,
                )
            )
            used.update((left.parent_id, right.parent_id))
    return tuple(selected)


def classify_fiber_pair(
    pair: FiberPair,
    left: FiberResponse,
    right: FiberResponse,
    margins: QuantumTrajectoryLawFiberEquivalence,
) -> FiberPairResult:
    if (
        left.parent_id != pair.left_parent_id
        or right.parent_id != pair.right_parent_id
        or left.action is not pair.action
        or right.action is not pair.action
    ):
        raise ValueError("fiber response is joined to another pair/action")
    if not left.valid or not right.valid:
        return FiberPairResult(
            pair.pair_id,
            pair.contrast_id,
            pair.action,
            "INVALID",
            {coordinate: "INVALID" for coordinate in RESPONSE_COORDINATES},
        )
    statuses: dict[str, str] = {}
    for coordinate in RESPONSE_COORDINATES:
        difference = abs(left.values[coordinate] - right.values[coordinate])
        halfwidth = left.halfwidths[coordinate] + right.halfwidths[coordinate]
        lower = max(0.0, difference - halfwidth)
        upper = difference + halfwidth
        margin = margins.values[coordinate]
        if lower > margin:
            statuses[coordinate] = "DIVERGED"
        elif upper <= margin:
            statuses[coordinate] = "CLOSED"
        else:
            statuses[coordinate] = "AMBIGUOUS"
    overall = (
        "DIVERGED"
        if "DIVERGED" in statuses.values()
        else "CLOSED"
        if set(statuses.values()) == {"CLOSED"}
        else "AMBIGUOUS"
    )
    return FiberPairResult(
        pair.pair_id,
        pair.contrast_id,
        pair.action,
        overall,
        statuses,
    )


def clopper_pearson_upper(
    divergences: int,
    total: int,
    *,
    alpha: float = 0.05,
) -> float:
    if total <= 0 or not 0 <= divergences <= total:
        raise ValueError("binomial count is invalid")
    if divergences == total:
        return 1.0
    return float(beta.ppf(1.0 - alpha, divergences + 1, total - divergences))


def adjudicate_fibers(
    pair_results: Sequence[FiberPairResult],
    *,
    minimum_pairs: int,
    minimum_per_contrast: int,
    entered_contrasts: Sequence[str],
    maximum_divergence_rate: float = 0.10,
    alpha: float = 0.05,
) -> FiberAdjudication:
    diverged = sum(result.status == "DIVERGED" for result in pair_results)
    closed = sum(result.status == "CLOSED" for result in pair_results)
    ambiguous = sum(result.status == "AMBIGUOUS" for result in pair_results)
    invalid = sum(result.status == "INVALID" for result in pair_results)
    resolved = closed + diverged
    upper = clopper_pearson_upper(diverged, resolved, alpha=alpha) if resolved else math.nan
    counts = {
        contrast: sum(
            result.contrast_id == contrast and result.status in {"CLOSED", "DIVERGED"}
            for result in pair_results
        )
        for contrast in entered_contrasts
    }
    reasons: list[str] = []
    if invalid:
        verdict = Verdict.FIBER_SOURCE_STOP
        reasons.append("invalid-fiber-response")
    elif diverged:
        verdict = Verdict.CONTROLLER_FIBER_DIVERGENCE
        reasons.append("resolved-inside-mask-divergence")
    elif resolved < minimum_pairs or any(count < minimum_per_contrast for count in counts.values()):
        verdict = Verdict.CONTROLLER_MASK_UNDERCOVERED
        reasons.append("fiber-coverage-floor-not-met")
    elif ambiguous:
        verdict = Verdict.CONTROLLER_FIBERS_MIXED
        reasons.append("fiber-equivalence-ambiguous")
    elif not math.isfinite(upper):
        verdict = Verdict.FIBER_UNEVALUABLE
        reasons.append("divergence-rate-unresolved")
    elif upper <= maximum_divergence_rate:
        verdict = Verdict.FINITE_CONTROLLER_FIBER_CLOSURE
    else:
        verdict = Verdict.CONTROLLER_FIBERS_MIXED
        reasons.append("divergence-rate-upper-bound-too-wide")
    return FiberAdjudication(
        verdict=verdict,
        resolved_pairs=resolved,
        diverged_pairs=diverged,
        ambiguous_pairs=ambiguous,
        divergence_rate_upper=upper,
        pair_results=tuple(pair_results),
        reason_codes=tuple(sorted(reasons)),
    )


__all__ = [
    "FiberAdjudication",
    "FiberCandidate",
    "FiberPair",
    "FiberPairResult",
    "FiberResponse",
    "adjudicate_fibers",
    "classify_fiber_pair",
    "clopper_pearson_upper",
    "select_fiber_pairs",
]

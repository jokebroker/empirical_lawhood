"""Outcome-blind controller-mask and natural-fiber pairing."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csc_matrix

from ..quantum_scientific_seeds import ScientificSeedCensus, validate_scientific_seed_census
from .contracts import Action, stable_json_bytes
from .receivers import causal_chart
from .response import ChartScaler, FiberMargins, chart_distance
from .source import PrefixCheckpoint


CONTRAST_ORDER = ("ORDER", "SITE", "BOUNDARY_AGE", "EARLIER_ORDER")
ELIGIBLE_CONTRASTS: Mapping[str, tuple[str, ...]] = {
    "regional-activity": CONTRAST_ORDER,
    "spatial-activity": ("ORDER", "BOUNDARY_AGE", "EARLIER_ORDER"),
    "temporal-memory": ("EARLIER_ORDER",),
}


@dataclass(frozen=True, slots=True)
class OmittedFeatures:
    last_transition: str
    last_site_boundary: bool | None
    boundary_age: float
    earlier_order: tuple[str, ...]
    retained_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PoolMapping:
    unit_id: str
    candidate_id: str | None
    candidate_action: Action | None
    distance: float


@dataclass(frozen=True, slots=True)
class PairCandidate:
    left_unit_id: str
    right_unit_id: str
    contrast_id: str
    chart_distance: float
    mask_compatible: bool
    candidate_id: str | None
    candidate_action: Action | None

    @property
    def pair_id(self) -> str:
        digest = sha256(
            (
                f"{self.left_unit_id}:{self.right_unit_id}:{self.contrast_id}:{self.candidate_id}"
            ).encode()
        ).hexdigest()[:20]
        return f"pair.quantum-receiver-response.{digest}"


@dataclass(frozen=True, slots=True)
class PairingResult:
    scientific_seed_census_sha256: str
    pairs: tuple[PairCandidate, ...]
    eligible_edge_count: int
    mask_pair_count: int
    mask_contrast_counts: Mapping[str, int]
    coverage_passed: bool


def omitted_features(
    prefix: PrefixCheckpoint,
    *,
    gamma: float,
    l_sites: int,
    particles: int,
) -> OmittedFeatures:
    h_s = 4.0 / (gamma * particles)
    h_l = 16.0 / (gamma * particles)
    cutoff = prefix.cutoff_time
    half = l_sites // 2
    retained = tuple(event for event in prefix.history_events if event.event_time >= cutoff - h_l)
    earlier = tuple(
        event
        for event in prefix.history_events
        if cutoff - h_l - h_s <= event.event_time < cutoff - h_l
    )

    def region(site: int) -> str:
        return "A" if site < half else "B"

    transition = "none"
    for left, right in zip(retained, retained[1:], strict=False):
        if region(left.site) != region(right.site):
            transition = f"{region(left.site)}->{region(right.site)}"
    boundary_sites = {0, half - 1, half, l_sites - 1}
    chart = causal_chart(
        prefix,
        gamma=gamma,
        l_sites=l_sites,
        particles=particles,
    )
    return OmittedFeatures(
        last_transition=transition,
        last_site_boundary=(retained[-1].site in boundary_sites if retained else None),
        boundary_age=float(chart["time_since_last_boundary_event"]),
        earlier_order=tuple(region(event.site) for event in earlier),
        retained_order=tuple(region(event.site) for event in retained),
    )


def eligible_contrast(
    *,
    left: OmittedFeatures,
    right: OmittedFeatures,
    scaler: ChartScaler,
) -> str | None:
    eligible = ELIGIBLE_CONTRASTS[scaler.chart_id]
    if "ORDER" in eligible and {left.last_transition, right.last_transition} == {"A->B", "B->A"}:
        return "ORDER"
    if (
        "SITE" in eligible
        and left.last_site_boundary is not None
        and right.last_site_boundary is not None
        and left.last_site_boundary != right.last_site_boundary
    ):
        return "SITE"
    if (
        "BOUNDARY_AGE" in eligible
        and abs(left.boundary_age - right.boundary_age)
        >= scaler.scales["time_since_last_boundary_event"]
    ):
        return "BOUNDARY_AGE"
    if "EARLIER_ORDER" in eligible and (
        left.earlier_order != right.earlier_order
        or (scaler.chart_id == "temporal-memory" and left.retained_order != right.retained_order)
    ):
        return "EARLIER_ORDER"
    return None


def _scientific_unit_order(
    order: tuple[str, ...] | None,
    *,
    expected: set[str],
) -> tuple[str, ...]:
    """Bind labels to caller-declared numeric positions without sorting labels."""
    if (
        type(order) is not tuple
        or any(type(label) is not str for label in order)
        or len(set(order)) != len(order)
        or set(order) != expected
    ):
        raise ValueError("scientific unit order must cover the exact declared unit roster")
    return order


def enumerate_pair_candidates(
    *,
    prefixes: Mapping[str, PrefixCheckpoint],
    charts: Mapping[str, Mapping[str, float | str]],
    scaler: ChartScaler,
    mappings: Mapping[str, PoolMapping],
    gamma: float,
    l_sites: int,
    particles: int,
    scientific_unit_order: tuple[str, ...] | None = None,
    scientific_edge_order: tuple[int, ...] | None = None,
) -> tuple[PairCandidate, ...]:
    unit_ids = _scientific_unit_order(scientific_unit_order, expected=set(prefixes))
    pair_count = len(unit_ids) * (len(unit_ids) - 1) // 2
    if (
        type(scientific_edge_order) is not tuple
        or any(type(index) is not int for index in scientific_edge_order)
        or len(scientific_edge_order) != pair_count
        or set(scientific_edge_order) != set(range(pair_count))
    ):
        raise ValueError("scientific edge order must permute the complete potential-pair domain")
    edge_rank = {index: rank for rank, index in enumerate(scientific_edge_order)}
    features = {
        unit_id: omitted_features(
            prefixes[unit_id],
            gamma=gamma,
            l_sites=l_sites,
            particles=particles,
        )
        for unit_id in unit_ids
    }
    edges: list[tuple[int, PairCandidate]] = []
    pair_index = -1
    for left_index, left_id in enumerate(unit_ids):
        for right_id in unit_ids[left_index + 1 :]:
            pair_index += 1
            if prefixes[left_id].k_left != prefixes[right_id].k_left:
                continue
            distance = chart_distance(charts[left_id], charts[right_id], scaler)
            if distance > 1.0:
                continue
            contrast = eligible_contrast(
                left=features[left_id],
                right=features[right_id],
                scaler=scaler,
            )
            if contrast is None:
                continue
            left_mapping = mappings[left_id]
            right_mapping = mappings[right_id]
            mask_compatible = (
                left_mapping.candidate_id is not None
                and left_mapping.candidate_id == right_mapping.candidate_id
                and left_mapping.candidate_action == right_mapping.candidate_action
            )
            edges.append((
                pair_index,
                PairCandidate(
                    left_unit_id=left_id,
                    right_unit_id=right_id,
                    contrast_id=contrast,
                    chart_distance=distance,
                    mask_compatible=mask_compatible,
                    candidate_id=(left_mapping.candidate_id if mask_compatible else None),
                    candidate_action=(left_mapping.candidate_action if mask_compatible else None),
                )
            ))
    return tuple(
        edge for _index, edge in sorted(
            edges,
            key=lambda row: (
                not row[1].mask_compatible,
                row[1].chart_distance,
                edge_rank[row[0]],
            ),
        )
    )


def _matching_constraints(
    edges: Sequence[PairCandidate],
    *,
    scientific_unit_order: tuple[str, ...],
) -> tuple[csc_matrix, NDArray[np.float64], NDArray[np.float64]]:
    unit_ids = _scientific_unit_order(
        scientific_unit_order,
        expected={label for edge in edges for label in (edge.left_unit_id, edge.right_unit_id)},
    )
    unit_row = {unit_id: row_index for row_index, unit_id in enumerate(unit_ids)}
    contrast_row = {
        contrast: len(unit_ids) + index for index, contrast in enumerate(CONTRAST_ORDER)
    }
    total_row = len(unit_ids) + len(CONTRAST_ORDER)
    row_indices: list[int] = []
    column_indices: list[int] = []
    data: list[float] = []
    for column, edge in enumerate(edges):
        for row in (
            unit_row[edge.left_unit_id],
            unit_row[edge.right_unit_id],
            contrast_row[edge.contrast_id],
            total_row,
        ):
            row_indices.append(row)
            column_indices.append(column)
            data.append(1.0)
    row_count = total_row + 1
    matrix = csc_matrix(
        (
            np.asarray(data, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_indices, dtype=np.int32),
            ),
        ),
        shape=(row_count, len(edges)),
    )
    upper = [
        *([1.0] * len(unit_ids)),
        *([16.0] * len(CONTRAST_ORDER)),
        48.0,
    ]
    return (
        matrix,
        np.full(row_count, -np.inf, dtype=np.float64),
        np.asarray(upper, dtype=np.float64),
    )


def select_pairs(
    edges: Sequence[PairCandidate],
    *,
    scientific_seeds: ScientificSeedCensus | None = None,
    scientific_unit_order: tuple[str, ...] | None = None,
) -> PairingResult:
    """Exact lexicographic matching: cardinality, mask count, then distance."""

    validate_scientific_seed_census(scientific_seeds, indices=tuple(range(len(edges))), bits=64)
    unit_order = _scientific_unit_order(
        scientific_unit_order,
        expected={label for edge in edges for label in (edge.left_unit_id, edge.right_unit_id)},
    )
    scientific_digest = sha256(stable_json_bytes(scientific_seeds)).hexdigest()
    if not edges:
        return PairingResult(
            scientific_seed_census_sha256=scientific_digest,
            pairs=(),
            eligible_edge_count=0,
            mask_pair_count=0,
            mask_contrast_counts={},
            coverage_passed=False,
        )
    matrix, lower, upper = _matching_constraints(edges, scientific_unit_order=unit_order)
    base_constraint = LinearConstraint(matrix, lower, upper)
    bounds = Bounds(
        np.zeros(len(edges), dtype=np.float64),
        np.ones(len(edges), dtype=np.float64),
    )
    integrality = np.ones(len(edges), dtype=np.int8)

    def solve(
        objective: NDArray[np.float64],
        extra: Sequence[LinearConstraint] = (),
    ) -> NDArray[np.float64]:
        result = milp(
            objective,
            integrality=integrality,
            bounds=bounds,
            constraints=(base_constraint, *extra),
            options={"disp": False, "presolve": True},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"natural-fiber matching failed: {result.message}")
        return np.asarray(result.x, dtype=np.float64)

    cardinality_solution = solve(-np.ones(len(edges), dtype=np.float64))
    maximum_cardinality = int(np.rint(cardinality_solution.sum()))
    cardinality_constraint = LinearConstraint(
        np.ones((1, len(edges)), dtype=np.float64),
        np.asarray([maximum_cardinality], dtype=np.float64),
        np.asarray([maximum_cardinality], dtype=np.float64),
    )
    mask_vector = np.asarray(
        [float(edge.mask_compatible) for edge in edges],
        dtype=np.float64,
    )
    mask_solution = solve(-mask_vector, (cardinality_constraint,))
    maximum_mask = int(np.rint(mask_vector @ mask_solution))
    mask_constraint = LinearConstraint(
        mask_vector.reshape(1, -1),
        np.asarray([maximum_mask], dtype=np.float64),
        np.asarray([maximum_mask], dtype=np.float64),
    )
    tie = np.asarray(
        [(seed % 1_000_000) * 1e-15 for _, seed in scientific_seeds],
        dtype=np.float64,
    )
    objective = (
        np.asarray(
            [edge.chart_distance for edge in edges],
            dtype=np.float64,
        )
        + tie
    )
    distance_solution = solve(
        objective,
        (cardinality_constraint, mask_constraint),
    )
    selected = tuple(
        edge for edge, value in zip(edges, distance_solution, strict=True) if value >= 0.5
    )
    mask_counts = {
        contrast: sum(edge.mask_compatible and edge.contrast_id == contrast for edge in selected)
        for contrast in CONTRAST_ORDER
    }
    entered = {key: value for key, value in mask_counts.items() if value}
    mask_pair_count = sum(edge.mask_compatible for edge in selected)
    coverage = (
        mask_pair_count >= 24 and bool(entered) and all(value >= 8 for value in entered.values())
    )
    return PairingResult(
        scientific_seed_census_sha256=scientific_digest,
        pairs=selected,
        eligible_edge_count=len(edges),
        mask_pair_count=mask_pair_count,
        mask_contrast_counts=entered,
        coverage_passed=coverage,
    )


def require_fiber_margins(value: object) -> FiberMargins:
    if not isinstance(value, FiberMargins):
        raise TypeError("natural-fiber adjudication refuses a non-fiber threshold family")
    return value


__all__ = [
    "CONTRAST_ORDER",
    "ELIGIBLE_CONTRASTS",
    "OmittedFeatures",
    "PairCandidate",
    "PairingResult",
    "PoolMapping",
    "eligible_contrast",
    "enumerate_pair_candidates",
    "omitted_features",
    "require_fiber_margins",
    "select_pairs",
]

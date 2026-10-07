"""Exact finite causal record grammar for quantum finite grammar identification.

The module deliberately contains no target, future, state-estimation, or
outcome-dependent feature construction.  Every matrix is materialized from an
explicit ordered feature manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from .contracts import Panel, target_weights
from .receiver import site_counts
from .source import EventRecord


COUNT_FEATURES = (
    "count_recent_a",
    "count_recent_b",
    "count_recent_total",
    "count_previous_a",
    "count_previous_b",
    "count_previous_total",
    "age_last_any",
    "age_last_a",
    "age_last_b",
    "delta_count_a",
    "delta_count_b",
    "empty_recent",
)
GEOMETRY_FEATURES = (
    "q_half_recent",
    "q_cos1_recent",
    "q_sin1_recent",
    "q_cos2_recent",
    "q_sin2_recent",
    "q_boundary_recent",
    "q_boundary_56_recent",
    "q_boundary_110_recent",
    "q_half_previous",
    "q_boundary_previous",
)
PAIR_FEATURES = (
    "pair_aa_short",
    "pair_bb_short",
    "pair_ab_short",
    "pair_ba_short",
    "pair_aa_long",
    "pair_bb_long",
    "pair_ab_long",
    "pair_ba_long",
    "boundary_a_to_b_short",
    "boundary_b_to_a_short",
    "boundary_a_to_b_long",
    "boundary_b_to_a_long",
    "age_last_boundary_event",
    "delta_boundary_direction",
)
CATEGORICAL_FEATURES = ("last_region_B", "last_region_none")
ALL_FEATURES = COUNT_FEATURES + GEOMETRY_FEATURES + PAIR_FEATURES
FEATURE_INDEX = {name: index for index, name in enumerate(ALL_FEATURES)}
PANEL_FEATURES: Mapping[Panel, tuple[str, ...]] = {
    Panel.COUNT: COUNT_FEATURES,
    Panel.SPATIAL_RECENCY: COUNT_FEATURES + GEOMETRY_FEATURES,
    Panel.EVENT_PAIR: ALL_FEATURES,
}
BOUNDARY_SITES = frozenset({0, 5, 6, 11})
A_TO_B_LINKS = frozenset({(5, 6), (0, 11)})
B_TO_A_LINKS = frozenset({(6, 5), (11, 0)})


@dataclass(frozen=True, slots=True)
class FeatureContext:
    """Donor-only feature context.

    The finite grammar intentionally has no fitted-intensity object because the residual panel
    was removed before freeze.
    """

    site_center: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.site_center) != 12 or not all(
            math.isfinite(value) for value in self.site_center
        ):
            raise ValueError("feature context has an invalid site center")


@dataclass(frozen=True, slots=True)
class FeatureRow:
    values: tuple[float, ...]
    last_region: str
    valid: bool
    reason_code: str | None

    def __post_init__(self) -> None:
        if len(self.values) != len(ALL_FEATURES):
            raise ValueError("feature row dimension differs")
        if self.last_region not in {"A", "B", "none"}:
            raise ValueError("last-region vocabulary differs")

    def vector(self, feature_order: Sequence[str]) -> NDArray[np.float64]:
        result: list[float] = []
        for name in feature_order:
            if name in FEATURE_INDEX:
                result.append(float(self.values[FEATURE_INDEX[name]]))
            elif name == "last_region_B":
                result.append(float(self.last_region == "B"))
            elif name == "last_region_none":
                result.append(float(self.last_region == "none"))
            else:
                raise ValueError(f"unknown feature coordinate: {name}")
        return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class Nomination:
    retained_panels: tuple[Panel, ...]
    retained_features: tuple[str, ...]
    rejected_features: Mapping[str, str]
    feature_order_by_panel: Mapping[str, tuple[str, ...]]
    support_rows: tuple[Mapping[str, object], ...]
    lag_coarsening: tuple[tuple[float, float], ...]
    label: str


def feature_names(
    panel: Panel,
    retained_features: Sequence[str] | None = None,
) -> tuple[str, ...]:
    allowed = PANEL_FEATURES[panel]
    if retained_features is None:
        numeric = allowed
    else:
        retained = set(retained_features)
        numeric = tuple(name for name in allowed if name in retained)
    return numeric + CATEGORICAL_FEATURES


def feature_matrix(
    rows: Sequence[FeatureRow],
    feature_order: Sequence[str],
) -> NDArray[np.float64]:
    order = tuple(feature_order)
    if len(order) != len(set(order)) or any(
        name not in FEATURE_INDEX and name not in CATEGORICAL_FEATURES for name in order
    ):
        raise ValueError("feature manifest is invalid")
    matrix = np.asarray([row.vector(order) for row in rows], dtype=np.float64)
    if matrix.shape != (len(rows), len(order)):
        raise AssertionError("feature matrix violates its manifest")
    return matrix


def _event_window(
    events: Sequence[EventRecord],
    start: float,
    end: float,
) -> tuple[EventRecord, ...]:
    if not start < end:
        raise ValueError("feature window is invalid")
    return tuple(event for event in events if start <= event.event_time < end)


def _age(
    events: Sequence[EventRecord],
    endpoint: float,
    predicate: object,
    *,
    empty_age: float,
) -> float:
    selected = [
        event.event_time
        for event in events
        if callable(predicate) and predicate(event) and event.event_time < endpoint
    ]
    return endpoint - max(selected) if selected else empty_age


def _basis_vectors() -> tuple[NDArray[np.float64], ...]:
    sites = np.arange(12, dtype=np.float64)
    half = np.r_[np.ones(6), -np.ones(6)]
    boundary = np.asarray(
        [1.0 if site in BOUNDARY_SITES else -0.5 for site in range(12)],
        dtype=np.float64,
    )
    boundary_56 = np.zeros(12, dtype=np.float64)
    boundary_56[5], boundary_56[6] = 1.0, -1.0
    boundary_110 = np.zeros(12, dtype=np.float64)
    boundary_110[11], boundary_110[0] = 1.0, -1.0
    return (
        half,
        np.cos(2 * np.pi * sites / 12),
        np.sin(2 * np.pi * sites / 12),
        np.cos(4 * np.pi * sites / 12),
        np.sin(4 * np.pi * sites / 12),
        boundary,
        boundary_56,
        boundary_110,
    )


def _quadratic(
    counts: NDArray[np.float64],
    center: NDArray[np.float64],
    vector: NDArray[np.float64],
) -> float:
    return float((vector @ (counts - center)) ** 2 - (vector**2) @ counts)


def _region_pair(kind: str, left: int, right: int) -> bool:
    left_a, right_a = left < 6, right < 6
    return {
        "aa": left_a and right_a,
        "bb": not left_a and not right_a,
        "ab": left_a and not right_a,
        "ba": not left_a and right_a,
    }[kind]


def _ordered_mark_probability(
    mark_counts: NDArray[np.int64],
    predicate: object,
) -> float:
    total = int(mark_counts.sum())
    if total < 2:
        return 0.0
    numerator = 0
    for left in range(12):
        for right in range(12):
            if callable(predicate) and predicate(left, right):
                numerator += int(mark_counts[left]) * (int(mark_counts[right]) - int(left == right))
    return float(numerator) / float(total * (total - 1))


def _centered_pair_kernel(
    events: Sequence[EventRecord],
    *,
    lag_start: float,
    lag_end: float,
    predicate: object,
) -> float:
    marks = np.asarray([event.site for event in events], dtype=np.int64)
    mark_counts = np.bincount(marks, minlength=12).astype(np.int64)
    expected_probability = _ordered_mark_probability(mark_counts, predicate)
    observed = 0.0
    eligible_clock_pairs = 0
    for index, left_event in enumerate(events[:-1]):
        for right_event in events[index + 1 :]:
            lag = right_event.event_time - left_event.event_time
            if lag_start <= lag < lag_end:
                eligible_clock_pairs += 1
                observed += float(
                    callable(predicate) and predicate(left_event.site, right_event.site)
                )
    return observed - expected_probability * eligible_clock_pairs


def _signed_boundary_activity(events: Sequence[EventRecord]) -> float:
    total = 0.0
    for index, left_event in enumerate(events[:-1]):
        for right_event in events[index + 1 :]:
            pair = (left_event.site, right_event.site)
            total += float(pair in A_TO_B_LINKS) - float(pair in B_TO_A_LINKS)
    return total


def _pair_features(
    recent: Sequence[EventRecord],
    previous: Sequence[EventRecord],
    long_events: Sequence[EventRecord],
    *,
    endpoint: float,
) -> tuple[float, ...]:
    values: list[float] = []
    for lag_start, lag_end in ((0.0, 2.0 / 3.0), (2.0 / 3.0, 4.0)):
        for kind in ("aa", "bb", "ab", "ba"):
            values.append(
                _centered_pair_kernel(
                    recent,
                    lag_start=lag_start,
                    lag_end=lag_end,
                    predicate=lambda left, right, pair_kind=kind: _region_pair(
                        pair_kind, left, right
                    ),
                )
            )
    for lag_start, lag_end in ((0.0, 2.0 / 3.0), (2.0 / 3.0, 4.0)):
        values.append(
            _centered_pair_kernel(
                recent,
                lag_start=lag_start,
                lag_end=lag_end,
                predicate=lambda left, right: (left, right) in A_TO_B_LINKS,
            )
        )
        values.append(
            _centered_pair_kernel(
                recent,
                lag_start=lag_start,
                lag_end=lag_end,
                predicate=lambda left, right: (left, right) in B_TO_A_LINKS,
            )
        )
    values.append(
        _age(
            long_events,
            endpoint,
            lambda event: event.site in BOUNDARY_SITES,
            empty_age=20.0,
        )
    )
    values.append(_signed_boundary_activity(recent) - _signed_boundary_activity(previous))
    if len(values) != len(PAIR_FEATURES):
        raise AssertionError("pair grammar dimension differs")
    return tuple(values)


def fit_feature_context(
    records: Sequence[Sequence[EventRecord]],
    k_values: Sequence[int],
) -> FeatureContext:
    if len(records) != len(k_values) or not records:
        raise ValueError("feature-context rows differ")
    weights = np.asarray(target_weights(list(k_values)), dtype=np.float64)
    recent_counts = np.asarray(
        [site_counts(events, start_time=196.0, end_time=200.0) for events in records],
        dtype=np.float64,
    )
    raw_center = weights @ recent_counts
    reflection = np.asarray([(5 - site) % 12 for site in range(12)], dtype=np.int64)
    center = 0.5 * (raw_center + raw_center[reflection])
    return FeatureContext(site_center=tuple(float(value) for value in center))


def extract_features(
    events: Sequence[EventRecord],
    context: FeatureContext,
    *,
    endpoint: float = 200.0,
) -> FeatureRow:
    if endpoint not in {196.0, 200.0}:
        raise ValueError("feature endpoint is outside the frozen matched/wrong-time chart")
    if any(event.event_time >= 200.0 for event in events):
        raise ValueError("feature input crosses the causal cutoff")
    recent = _event_window(events, endpoint - 4.0, endpoint)
    previous = _event_window(events, endpoint - 8.0, endpoint - 4.0)
    long_events = _event_window(events, endpoint - 20.0, endpoint)
    recent_counts = site_counts(recent, start_time=endpoint - 4.0, end_time=endpoint)
    previous_counts = site_counts(
        previous,
        start_time=endpoint - 8.0,
        end_time=endpoint - 4.0,
    )
    recent_a = float(recent_counts[:6].sum())
    recent_b = float(recent_counts[6:].sum())
    previous_a = float(previous_counts[:6].sum())
    previous_b = float(previous_counts[6:].sum())
    count_values = (
        recent_a,
        recent_b,
        recent_a + recent_b,
        previous_a,
        previous_b,
        previous_a + previous_b,
        _age(events, endpoint, lambda event: True, empty_age=24.0),
        _age(events, endpoint, lambda event: event.site < 6, empty_age=24.0),
        _age(events, endpoint, lambda event: event.site >= 6, empty_age=24.0),
        recent_a - previous_a,
        recent_b - previous_b,
        float(not recent),
    )
    center = np.asarray(context.site_center, dtype=np.float64)
    basis = _basis_vectors()
    geometry_values = tuple(_quadratic(recent_counts, center, vector) for vector in basis) + (
        _quadratic(previous_counts, center, basis[0]),
        _quadratic(previous_counts, center, basis[5]),
    )
    pair_values = _pair_features(recent, previous, long_events, endpoint=endpoint)
    values = count_values + geometry_values + pair_values
    valid = len(values) == 36 and all(math.isfinite(value) for value in values)
    last_region = "none" if not recent else ("A" if recent[-1].site < 6 else "B")
    return FeatureRow(
        values=tuple(float(value) for value in values),
        last_region=last_region,
        valid=valid,
        reason_code=None if valid else "NONFINITE_OR_DIMENSION_MISMATCH",
    )


def nominate_grammar(
    records: Sequence[Sequence[EventRecord]],
    k_values: Sequence[int],
    *,
    minimum_per_sector: int = 32,
) -> Nomination:
    """Support-only predecessor nomination; never a compiler or evidence act."""

    context = fit_feature_context(records, k_values)
    rows = [extract_features(events, context) for events in records]
    matrix = np.asarray([row.values for row in rows], dtype=np.float64)
    weights = np.asarray(target_weights(list(k_values)), dtype=np.float64)
    rejected: dict[str, str] = {}
    retained: list[str] = []
    support_rows: list[Mapping[str, object]] = []
    empty_history_frequency = float(np.mean(matrix[:, FEATURE_INDEX["empty_recent"]] == 1.0))
    for index, name in enumerate(ALL_FEATURES):
        column = matrix[:, index]
        finite = np.isfinite(column)
        nonzero = finite & (np.abs(column) > 1.0e-12)
        per_sector = {str(k): int(np.sum(nonzero & (np.asarray(k_values) == k))) for k in range(7)}
        finite_fraction = float(np.mean(finite))
        valid_weight = float(weights[finite].sum())
        weighted_variance = (
            float(
                weights[finite]
                @ (column[finite] - float(weights[finite] @ column[finite] / valid_weight)) ** 2
                / valid_weight
            )
            if valid_weight > 0.0
            else math.nan
        )
        if finite_fraction < 0.99 or valid_weight < 0.98:
            disposition = "REJECT_NONFINITE"
        elif float(np.ptp(column[finite])) <= 1.0e-12:
            disposition = "REJECT_CONSTANT"
        elif any(count < minimum_per_sector for count in per_sector.values()):
            disposition = "REJECT_LOW_SUPPORT"
        else:
            disposition = "RETAIN"
            retained.append(name)
        if disposition != "RETAIN":
            rejected[name] = disposition
        support_rows.append(
            {
                "feature_id": name,
                "semantic_group": (
                    "count_history"
                    if name in COUNT_FEATURES
                    else "receiver_geometry"
                    if name in GEOMETRY_FEATURES
                    else "ordered_pair_boundary"
                ),
                "observed_support_count": int(np.sum(nonzero)),
                "finite_count": int(np.sum(finite)),
                "invalid_count": int(np.sum(~finite)),
                "finite_parent_fraction": finite_fraction,
                "valid_target_weight": valid_weight,
                "nonzero_support_by_sector": per_sector,
                "minimum_nonzero_support_per_sector": min(per_sector.values()),
                "empty_history_frequency": empty_history_frequency,
                "weighted_variance": weighted_variance,
                "disposition": disposition,
            }
        )
    retained_set = set(retained)
    count_required = set(COUNT_FEATURES[:-1])
    panels: list[Panel] = []
    if count_required <= retained_set and (
        "empty_recent" in retained_set or rejected.get("empty_recent") == "REJECT_CONSTANT"
    ):
        panels.append(Panel.COUNT)
    if Panel.COUNT in panels and set(GEOMETRY_FEATURES) <= retained_set:
        panels.append(Panel.SPATIAL_RECENCY)
    if Panel.SPATIAL_RECENCY in panels and set(PAIR_FEATURES) <= retained_set:
        panels.append(Panel.EVENT_PAIR)
    orders = {panel.value: feature_names(panel, retained) for panel in panels}
    return Nomination(
        retained_panels=tuple(panels),
        retained_features=tuple(name for name in ALL_FEATURES if name in retained_set),
        rejected_features=rejected,
        feature_order_by_panel=orders,
        support_rows=tuple(support_rows),
        lag_coarsening=((0.0, 2.0 / 3.0), (2.0 / 3.0, 4.0)),
        label=(
            "NON_PROMOTABLE_OUTCOME_VISIBLE_NOMINATION"
            if panels
            else "NON_PROMOTABLE_GRAMMAR_REDESIGN_REQUIRED"
        ),
    )


__all__ = [
    "ALL_FEATURES",
    "CATEGORICAL_FEATURES",
    "COUNT_FEATURES",
    "FeatureContext",
    "FeatureRow",
    "GEOMETRY_FEATURES",
    "Nomination",
    "PAIR_FEATURES",
    "PANEL_FEATURES",
    "extract_features",
    "feature_matrix",
    "feature_names",
    "fit_feature_context",
    "nominate_grammar",
]

"""Finite, causal marked-record feature grammar for quantum marked record identification."""

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
    "recent_n_a",
    "recent_n_b",
    "recent_n_total",
    "previous_n_a",
    "previous_n_b",
    "previous_n_total",
    "age_last",
    "age_last_a",
    "age_last_b",
    "contrast_a",
    "contrast_b",
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
    "pair_d1_short",
    "pair_d23_short",
    "pair_56_oriented",
    "pair_110_oriented",
    "pair_boundary_direction",
    "pair_boundary_lag_weighted",
)
RESIDUAL_FEATURES = (
    "martingale_a",
    "martingale_a_quadratic",
    "martingale_boundary",
    "martingale_boundary_quadratic",
)
ALL_FEATURES = COUNT_FEATURES + GEOMETRY_FEATURES + PAIR_FEATURES + RESIDUAL_FEATURES


@dataclass(frozen=True, slots=True)
class IntensityModel:
    p_a_by_cell: Mapping[str, float]
    p_boundary_by_cell: Mapping[str, float]
    global_p_a: float
    global_p_boundary: float


@dataclass(frozen=True, slots=True)
class FeatureContext:
    site_center: tuple[float, ...]
    intensity: IntensityModel


@dataclass(frozen=True, slots=True)
class FeatureRow:
    values: tuple[float, ...]
    last_region: str
    valid: bool
    reason_code: str | None

    def vector(self, panel: Panel) -> NDArray[np.float64]:
        numeric = np.asarray(self.values[: panel.dimension], dtype=np.float64)
        category = np.asarray(
            [float(self.last_region == "A"), float(self.last_region == "B")],
            dtype=np.float64,
        )
        return np.concatenate((numeric, category))


@dataclass(frozen=True, slots=True)
class Nomination:
    retained_panels: tuple[Panel, ...]
    retained_features: tuple[str, ...]
    rejected_features: Mapping[str, str]
    lag_coarsening: tuple[tuple[float, float], ...]
    martingale_implementation: str
    label: str


def feature_names(panel: Panel) -> tuple[str, ...]:
    return ALL_FEATURES[: panel.dimension] + ("last_region_A", "last_region_B")


def _event_window(
    events: Sequence[EventRecord],
    start: float,
    end: float,
) -> tuple[EventRecord, ...]:
    if not start < end:
        raise ValueError("feature window is invalid")
    return tuple(event for event in events if start <= event.event_time < end)


def _age(events: Sequence[EventRecord], endpoint: float, predicate: object) -> float:
    selected = [
        event.event_time
        for event in events
        if callable(predicate) and predicate(event) and event.event_time < endpoint
    ]
    return endpoint - max(selected) if selected else 24.0


def _basis_vectors() -> tuple[NDArray[np.float64], ...]:
    sites = np.arange(12, dtype=np.float64)
    half = np.r_[np.ones(6), -np.ones(6)]
    boundary = np.asarray(
        [1 if site in {0, 5, 6, 11} else -0.5 for site in range(12)],
        dtype=np.float64,
    )
    boundary_56 = np.zeros(12)
    boundary_56[5], boundary_56[6] = 1.0, -1.0
    boundary_110 = np.zeros(12)
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


def _distance(left: int, right: int) -> int:
    raw = abs(left - right)
    return min(raw, 12 - raw)


def _pair_predicate(name: str, left: int, right: int) -> float:
    left_a, right_a = left < 6, right < 6
    if name == "pair_aa_short":
        return float(left_a and right_a)
    if name == "pair_bb_short":
        return float(not left_a and not right_a)
    if name == "pair_ab_short":
        return float(left_a and not right_a)
    if name == "pair_ba_short":
        return float(not left_a and right_a)
    if name == "pair_d1_short":
        return float(_distance(left, right) == 1)
    if name == "pair_d23_short":
        return float(_distance(left, right) in {2, 3})
    if name == "pair_56_oriented":
        return float((left, right) in {(5, 6), (6, 5)})
    if name == "pair_110_oriented":
        return float((left, right) in {(11, 0), (0, 11)})
    if name == "pair_boundary_direction":
        return float((left, right) in {(5, 6), (11, 0)}) - float((left, right) in {(6, 5), (0, 11)})
    if name == "pair_boundary_lag_weighted":
        return float((left, right) in {(5, 6), (6, 5), (11, 0), (0, 11)})
    raise ValueError("unknown pair kernel")


def _pair_features(events: Sequence[EventRecord]) -> tuple[float, ...]:
    selected = tuple(events)
    count = len(selected)
    if count < 2:
        return (0.0,) * len(PAIR_FEATURES)
    marks = np.asarray([event.site for event in selected], dtype=np.int64)
    mark_counts = np.bincount(marks, minlength=12).astype(np.float64)
    ordered_denominator = float(count * (count - 1))
    expected_mark: dict[str, float] = {}
    for name in PAIR_FEATURES:
        numerator = 0.0
        for left in range(12):
            for right in range(12):
                multiplicity = mark_counts[left] * (mark_counts[right] - float(left == right))
                numerator += multiplicity * _pair_predicate(name, left, right)
        expected_mark[name] = numerator / ordered_denominator
    observed = {name: 0.0 for name in PAIR_FEATURES}
    expected = {name: 0.0 for name in PAIR_FEATURES}
    for index, left_event in enumerate(selected[:-1]):
        for right_event in selected[index + 1 :]:
            lag = right_event.event_time - left_event.event_time
            for name in PAIR_FEATURES:
                if name.endswith("_short") and lag >= 2.0 / 3.0:
                    continue
                if name == "pair_boundary_lag_weighted":
                    time_weight = math.exp(-lag / (2.0 / 3.0))
                else:
                    time_weight = 1.0
                observed[name] += time_weight * _pair_predicate(
                    name, left_event.site, right_event.site
                )
                expected[name] += time_weight * expected_mark[name]
    return tuple(observed[name] - expected[name] for name in PAIR_FEATURES)


def _cell_key(
    *,
    last_region: str,
    last_boundary_region: str,
    recent_a: int,
    recent_b: int,
    age: float,
    boundary_age: float,
) -> str:
    balance = -1 if recent_a < recent_b else (1 if recent_a > recent_b else 0)
    age_bin = 0 if age < 1 / 6 else (1 if age < 2 / 3 else 2)
    boundary_bin = 0 if boundary_age < 2 / 3 else 1
    return f"{last_region}|{last_boundary_region}|{balance}|{age_bin}|{boundary_bin}"


def _intensity_path(
    events: Sequence[EventRecord],
    model: IntensityModel | None,
) -> tuple[tuple[str, bool, bool], ...]:
    history: list[EventRecord] = []
    rows: list[tuple[str, bool, bool]] = []
    for event in events:
        prior = tuple(history)
        recent = [row for row in prior if event.event_time - 4 <= row.event_time]
        last_region = "none" if not prior else ("A" if prior[-1].site < 6 else "B")
        prior_boundary = [row for row in prior if row.site in {0, 5, 6, 11}]
        last_boundary = (
            "none" if not prior_boundary else ("A" if prior_boundary[-1].site in {0, 5} else "B")
        )
        age = event.event_time - prior[-1].event_time if prior else 24.0
        boundary_age = event.event_time - prior_boundary[-1].event_time if prior_boundary else 24.0
        key = _cell_key(
            last_region=last_region,
            last_boundary_region=last_boundary,
            recent_a=sum(row.site < 6 for row in recent),
            recent_b=sum(row.site >= 6 for row in recent),
            age=age,
            boundary_age=boundary_age,
        )
        rows.append((key, event.site < 6, event.site in {0, 5, 6, 11}))
        history.append(event)
    return tuple(rows)


def fit_feature_context(
    records: Sequence[Sequence[EventRecord]],
    k_values: Sequence[int],
) -> FeatureContext:
    if len(records) != len(k_values):
        raise ValueError("feature-context rows differ")
    weights = np.asarray(target_weights(list(k_values)), dtype=np.float64)
    recent_counts = np.asarray(
        [site_counts(events, start_time=196.0, end_time=200.0) for events in records]
    )
    center = weights @ recent_counts
    cell_total: dict[str, int] = {}
    cell_a: dict[str, int] = {}
    cell_boundary: dict[str, int] = {}
    total = total_a = total_boundary = 0
    for events in records:
        long_events = _event_window(events, 180.0, 200.0)
        for key, is_a, is_boundary in _intensity_path(long_events, None):
            cell_total[key] = cell_total.get(key, 0) + 1
            cell_a[key] = cell_a.get(key, 0) + int(is_a)
            cell_boundary[key] = cell_boundary.get(key, 0) + int(is_boundary)
            total += 1
            total_a += int(is_a)
            total_boundary += int(is_boundary)
    if total == 0:
        raise ValueError("intensity context has no development events")
    global_a = (total_a + 1.0) / (total + 2.0)
    global_boundary = (total_boundary + 1.0) / (total + 2.0)
    shrink = 8.0
    return FeatureContext(
        site_center=tuple(float(value) for value in center),
        intensity=IntensityModel(
            p_a_by_cell={
                key: (cell_a[key] + shrink * global_a) / (value + shrink)
                for key, value in cell_total.items()
            },
            p_boundary_by_cell={
                key: (cell_boundary[key] + shrink * global_boundary) / (value + shrink)
                for key, value in cell_total.items()
            },
            global_p_a=global_a,
            global_p_boundary=global_boundary,
        ),
    )


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
    recent_counts = site_counts(
        recent,
        start_time=endpoint - 4.0,
        end_time=endpoint,
    )
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
        _age(events, endpoint, lambda event: True),
        _age(events, endpoint, lambda event: event.site < 6),
        _age(events, endpoint, lambda event: event.site >= 6),
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
    pair_values = _pair_features(recent)
    martingale_a = 0.0
    martingale_boundary = 0.0
    predictable_a = 0.0
    predictable_boundary = 0.0
    for key, is_a, is_boundary in _intensity_path(long_events, context.intensity):
        p_a = context.intensity.p_a_by_cell.get(key, context.intensity.global_p_a)
        p_boundary = context.intensity.p_boundary_by_cell.get(
            key, context.intensity.global_p_boundary
        )
        martingale_a += float(is_a) - p_a
        martingale_boundary += float(is_boundary) - p_boundary
        predictable_a += p_a * (1.0 - p_a)
        predictable_boundary += p_boundary * (1.0 - p_boundary)
    residual_values = (
        martingale_a,
        martingale_a**2 - predictable_a,
        martingale_boundary,
        martingale_boundary**2 - predictable_boundary,
    )
    values = count_values + geometry_values + pair_values + residual_values
    valid = len(values) == 36 and all(math.isfinite(value) for value in values)
    last_region = "none" if not recent else ("A" if recent[-1].site < 6 else "B")
    return FeatureRow(
        values=tuple(float(value) for value in values),
        last_region=last_region,
        valid=valid,
        reason_code=None if valid else "NONFINITE_OR_DIMENSION_MISMATCH",
    )


def feature_matrix(rows: Sequence[FeatureRow], panel: Panel) -> NDArray[np.float64]:
    matrix = np.asarray([row.vector(panel) for row in rows], dtype=np.float64)
    if matrix.shape != (len(rows), panel.dimension + 2):
        raise AssertionError("feature matrix violates the panel contract")
    return matrix


def nominate_grammar(
    records: Sequence[Sequence[EventRecord]],
    k_values: Sequence[int],
) -> Nomination:
    """Outcome-visible technical nomination; never a compiler or evidence act."""

    context = fit_feature_context(records, k_values)
    rows = [extract_features(events, context) for events in records]
    matrix = np.asarray([row.values for row in rows], dtype=np.float64)
    rejected: dict[str, str] = {}
    retained: list[str] = []
    for index, name in enumerate(ALL_FEATURES):
        column = matrix[:, index]
        if not np.all(np.isfinite(column)):
            rejected[name] = "NONFINITE_ON_PREDECESSOR_RECORD"
        elif np.ptp(column) <= 1.0e-12:
            rejected[name] = "EXACTLY_CONSTANT_ON_PREDECESSOR_RECORD"
        else:
            retained.append(name)
    retained_set = set(retained)
    panels: list[Panel] = []
    if len(retained_set & set(COUNT_FEATURES)) >= 8:
        panels.append(Panel.COUNT)
    if panels and len(retained_set & set(GEOMETRY_FEATURES)) >= 5:
        panels.append(Panel.SPATIAL_RECENCY)
    if Panel.SPATIAL_RECENCY in panels and len(retained_set & set(PAIR_FEATURES)) >= 5:
        panels.append(Panel.EVENT_PAIR)
    if Panel.EVENT_PAIR in panels and len(retained_set & set(RESIDUAL_FEATURES)) >= 2:
        panels.append(Panel.INTENSITY_RESIDUAL)
    label = (
        "NON_PROMOTABLE_OUTCOME_VISIBLE_NOMINATION"
        if panels
        else "NON_PROMOTABLE_GRAMMAR_REDESIGN_REQUIRED"
    )
    return Nomination(
        retained_panels=tuple(panels),
        retained_features=tuple(retained),
        rejected_features=rejected,
        # A fixed contiguous coarsening of the atomic grid.  It is a
        # throughput choice, not an optimized endpoint.
        lag_coarsening=((0.0, 2.0 / 3.0), (2.0 / 3.0, 4.0)),
        martingale_implementation="finite-cell-binomial-shrinkage",
        label=label,
    )


__all__ = [
    "ALL_FEATURES",
    "COUNT_FEATURES",
    "FeatureContext",
    "FeatureRow",
    "GEOMETRY_FEATURES",
    "IntensityModel",
    "Nomination",
    "PAIR_FEATURES",
    "RESIDUAL_FEATURES",
    "extract_features",
    "feature_matrix",
    "feature_names",
    "fit_feature_context",
    "nominate_grammar",
]

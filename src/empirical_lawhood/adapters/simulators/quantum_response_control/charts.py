"""Strictly causal finite jump-record charts and omitted-history contrasts."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .source import EventRecord, PrefixCheckpoint


CHART_COORDINATES: Mapping[str, tuple[str, ...]] = {
    "regional-activity": (
        "count_a_short",
        "count_b_short",
        "time_since_last_event",
        "last_event_region",
    ),
    "spatial-activity": (
        "count_a_short",
        "count_b_short",
        "time_since_last_event",
        "last_event_region",
        "last_site_sin",
        "last_site_cos",
        "fourier_long_cos",
        "fourier_long_sin",
    ),
    "temporal-memory": (
        "count_a_short",
        "count_b_short",
        "time_since_last_event",
        "last_event_region",
        "last_site_sin",
        "last_site_cos",
        "fourier_long_cos",
        "fourier_long_sin",
        "count_a_earlier",
        "count_b_earlier",
        "time_since_boundary_near",
        "transitions_a_to_b",
        "transitions_b_to_a",
    ),
}

CATEGORICAL_COORDINATES = frozenset({"last_event_region"})


@dataclass(frozen=True, slots=True)
class CausalChart:
    chart_id: str
    coordinate_ids: tuple[str, ...]
    values: tuple[float | str, ...]
    cutoff_time: float
    source_record_sha256: str

    def as_mapping(self) -> dict[str, float | str]:
        return dict(zip(self.coordinate_ids, self.values, strict=True))


def _selected(
    events: Sequence[EventRecord],
    start: float,
    end: float,
) -> tuple[EventRecord, ...]:
    return tuple(event for event in events if start <= event.event_time < end)


def _region(site: int, l_sites: int) -> str:
    return "A" if site < l_sites // 2 else "B"


def _record_digest(events: Sequence[EventRecord], cutoff: float) -> str:
    from hashlib import sha256
    import json

    payload = json.dumps(
        [[event.event_index, event.event_time, event.site] for event in events],
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return sha256(payload + repr(cutoff).encode()).hexdigest()


def construct_charts(
    prefix: PrefixCheckpoint,
    *,
    l_sites: int = 12,
    history_short: float = 4 / 6,
    history_long: float = 16 / 6,
) -> dict[str, CausalChart]:
    cutoff = prefix.cutoff_time
    if any(event.event_time >= cutoff for event in prefix.history_events):
        raise ValueError("future event entered the causal record")
    short = _selected(prefix.history_events, cutoff - history_short, cutoff)
    long = _selected(prefix.history_events, cutoff - history_long, cutoff)
    earlier = tuple(event for event in long if event.event_time < cutoff - history_short)
    last = prefix.history_events[-1] if prefix.history_events else None
    last_region = _region(last.site, l_sites) if last is not None else "none"
    time_since_last = cutoff - last.event_time if last is not None else history_long
    last_site = last.site if last is not None else 0
    angle = 2.0 * math.pi * last_site / l_sites
    long_angles = np.asarray(
        [2.0 * math.pi * event.site / l_sites for event in long],
        dtype=np.float64,
    )
    boundary_sites = {0, l_sites // 2 - 1, l_sites // 2, l_sites - 1}
    boundary_events = [event for event in prefix.history_events if event.site in boundary_sites]
    boundary_age = cutoff - boundary_events[-1].event_time if boundary_events else history_long
    transitions_ab = 0
    transitions_ba = 0
    for left, right in zip(long, long[1:], strict=False):
        pair = (_region(left.site, l_sites), _region(right.site, l_sites))
        transitions_ab += pair == ("A", "B")
        transitions_ba += pair == ("B", "A")
    values: dict[str, float | str] = {
        "count_a_short": float(sum(event.site < l_sites // 2 for event in short)),
        "count_b_short": float(sum(event.site >= l_sites // 2 for event in short)),
        "time_since_last_event": float(time_since_last),
        "last_event_region": last_region,
        "last_site_sin": float(math.sin(angle)) if last is not None else 0.0,
        "last_site_cos": float(math.cos(angle)) if last is not None else 0.0,
        "fourier_long_cos": float(np.cos(long_angles).sum()) if len(long_angles) else 0.0,
        "fourier_long_sin": float(np.sin(long_angles).sum()) if len(long_angles) else 0.0,
        "count_a_earlier": float(sum(event.site < l_sites // 2 for event in earlier)),
        "count_b_earlier": float(sum(event.site >= l_sites // 2 for event in earlier)),
        "time_since_boundary_near": float(boundary_age),
        "transitions_a_to_b": float(transitions_ab),
        "transitions_b_to_a": float(transitions_ba),
    }
    digest = _record_digest(prefix.history_events, cutoff)
    return {
        chart_id: CausalChart(
            chart_id=chart_id,
            coordinate_ids=coordinate_ids,
            values=tuple(values[coordinate] for coordinate in coordinate_ids),
            cutoff_time=cutoff,
            source_record_sha256=digest,
        )
        for chart_id, coordinate_ids in CHART_COORDINATES.items()
    }


def omitted_history_features(
    prefix: PrefixCheckpoint,
    *,
    l_sites: int = 12,
    history_long: float = 16 / 6,
) -> dict[str, float | str]:
    cutoff = prefix.cutoff_time
    long = _selected(prefix.history_events, cutoff - history_long, cutoff)
    earlier = _selected(prefix.history_events, cutoff - 20.0, cutoff - history_long)
    regions = "".join(_region(event.site, l_sites) for event in long)
    earlier_regions = "".join(_region(event.site, l_sites) for event in earlier)
    last_site = long[-1].site if long else -1
    boundary_sites = {0, l_sites // 2 - 1, l_sites // 2, l_sites - 1}
    boundary = [event for event in long if event.site in boundary_sites]
    return {
        "ORDER": regions,
        "SITE": str(last_site),
        "BOUNDARY_AGE": float(cutoff - boundary[-1].event_time) if boundary else history_long,
        "EARLIER_ORDER": earlier_regions,
    }


def chart_has_contrast(chart_id: str, contrast_id: str) -> bool:
    coordinates = set(CHART_COORDINATES[chart_id])
    represented = {
        "ORDER": {"transitions_a_to_b", "transitions_b_to_a"},
        "SITE": {"last_site_sin", "last_site_cos"},
        "BOUNDARY_AGE": {"time_since_boundary_near"},
        "EARLIER_ORDER": {"count_a_earlier", "count_b_earlier"},
    }[contrast_id]
    return represented.issubset(coordinates)


__all__ = [
    "CATEGORICAL_COORDINATES",
    "CHART_COORDINATES",
    "CausalChart",
    "chart_has_contrast",
    "construct_charts",
    "omitted_history_features",
]

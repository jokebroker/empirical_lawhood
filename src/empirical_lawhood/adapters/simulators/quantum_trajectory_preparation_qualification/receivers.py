"""Receiver, causal-chart and bounded state-observable reductions."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.sparse import csr_matrix

from .basis import FixedNumberBasis
from .types import ComplexVector, EventRecord


RECEIVER_COORDINATES = (
    *(f"site_count_{site:02d}" for site in range(12)),
    "count_A",
    "count_B",
    "count_total",
    "burst_A",
    "burst_B",
    "fourier_cos_1",
    "fourier_sin_1",
)
CHART_NUMERIC_COORDINATES = (
    "chart_count_A_short",
    "chart_count_B_short",
    "chart_time_since_last_event",
    "chart_sin_last_site",
    "chart_cos_last_site",
    "chart_sum_cos_long",
    "chart_sum_sin_long",
    "chart_count_A_long_without_short",
    "chart_count_B_long_without_short",
    "chart_time_since_last_boundary_event",
    "chart_transitions_A_to_B",
    "chart_transitions_B_to_A",
)
CHART_ONE_HOT_COORDINATES = (
    "chart_last_region_A",
    "chart_last_region_B",
    "chart_last_region_none",
)
STATE_COORDINATES = (
    *(f"state_site_occupation_{site:02d}" for site in range(12)),
    "state_half_occupation",
    "state_density_correlation_within",
    "state_density_correlation_boundary",
)
CONTINUOUS_COORDINATES = (
    *RECEIVER_COORDINATES,
    *CHART_NUMERIC_COORDINATES,
    *CHART_ONE_HOT_COORDINATES,
    *STATE_COORDINATES,
)


@dataclass(frozen=True, slots=True)
class Observation:
    endpoint: float
    values: Mapping[str, float]
    energy: float
    event_rate: float
    state_sha256: str
    last_event_present: bool
    last_boundary_event_present: bool


def _events_between(
    events: Sequence[EventRecord],
    start: float,
    end: float,
) -> tuple[EventRecord, ...]:
    return tuple(event for event in events if start <= event.event_time < end)


def receiver_coordinates(
    events: Sequence[EventRecord],
    *,
    start: float,
    end: float,
    l_sites: int,
    particles: int,
    gamma: float,
) -> dict[str, float]:
    selected = _events_between(events, start, end)
    counts = np.bincount(
        [event.site for event in selected],
        minlength=l_sites,
    ).astype(np.float64)
    half = l_sites // 2
    burst_gap = math.log(2.0) / (gamma * particles)

    def burst(region_a: bool) -> float:
        return float(
            sum(
                (left.site < half) == region_a
                and (right.site < half) == region_a
                and right.event_time - left.event_time <= burst_gap
                for left, right in zip(selected, selected[1:], strict=False)
            )
        )

    angles = np.asarray(
        [2.0 * math.pi * event.site / l_sites for event in selected],
        dtype=np.float64,
    )
    result = {f"site_count_{site:02d}": float(counts[site]) for site in range(l_sites)}
    if l_sites != 12:
        raise ValueError("quantum trajectory preparation qualification receiver qualification panel is defined at L=12")
    result.update(
        {
            "count_A": float(counts[:half].sum()),
            "count_B": float(counts[half:].sum()),
            "count_total": float(counts.sum()),
            "burst_A": burst(True),
            "burst_B": burst(False),
            "fourier_cos_1": float(np.cos(angles).sum()) if len(angles) else 0.0,
            "fourier_sin_1": float(np.sin(angles).sum()) if len(angles) else 0.0,
        }
    )
    return result


def chart_coordinates(
    events: Sequence[EventRecord],
    *,
    cutoff: float,
    l_sites: int,
    particles: int,
    gamma: float,
) -> tuple[dict[str, float], bool, bool]:
    h_short = 4.0 / (gamma * particles)
    h_long = 16.0 / (gamma * particles)
    short = _events_between(events, cutoff - h_short, cutoff)
    long = _events_between(events, cutoff - h_long, cutoff)
    half = l_sites // 2
    boundary_sites = {0, half - 1, half, l_sites - 1}
    boundary_events = tuple(event for event in long if event.site in boundary_sites)
    last = long[-1] if long else None
    angles = np.asarray(
        [2.0 * math.pi * event.site / l_sites for event in long],
        dtype=np.float64,
    )

    def region(site: int) -> str:
        return "A" if site < half else "B"

    transitions_ab = 0
    transitions_ba = 0
    for left, right in zip(long, long[1:], strict=False):
        pair = region(left.site), region(right.site)
        transitions_ab += pair == ("A", "B")
        transitions_ba += pair == ("B", "A")
    last_region = region(last.site) if last is not None else "none"
    result = {
        "chart_count_A_short": float(sum(event.site < half for event in short)),
        "chart_count_B_short": float(sum(event.site >= half for event in short)),
        "chart_time_since_last_event": (
            min(cutoff - last.event_time, h_long) if last is not None else h_long
        ),
        "chart_sin_last_site": (
            math.sin(2.0 * math.pi * last.site / l_sites) if last is not None else 0.0
        ),
        "chart_cos_last_site": (
            math.cos(2.0 * math.pi * last.site / l_sites) if last is not None else 0.0
        ),
        "chart_sum_cos_long": float(np.cos(angles).sum()) if len(angles) else 0.0,
        "chart_sum_sin_long": float(np.sin(angles).sum()) if len(angles) else 0.0,
        "chart_count_A_long_without_short": float(
            sum(event.site < half and event.event_time < cutoff - h_short for event in long)
        ),
        "chart_count_B_long_without_short": float(
            sum(event.site >= half and event.event_time < cutoff - h_short for event in long)
        ),
        "chart_time_since_last_boundary_event": (
            min(cutoff - boundary_events[-1].event_time, h_long) if boundary_events else h_long
        ),
        "chart_transitions_A_to_B": float(transitions_ab),
        "chart_transitions_B_to_A": float(transitions_ba),
        "chart_last_region_A": float(last_region == "A"),
        "chart_last_region_B": float(last_region == "B"),
        "chart_last_region_none": float(last_region == "none"),
    }
    return result, last is not None, bool(boundary_events)


def state_coordinates(
    basis: FixedNumberBasis,
    state: ComplexVector,
) -> dict[str, float]:
    if basis.l_sites != 12:
        raise ValueError("quantum trajectory preparation qualification state qualification panel is defined at L=12")
    weights = np.abs(state) ** 2
    site_values = np.asarray(weights @ basis.occupations, dtype=np.float64)
    half = basis.l_sites // 2
    boundary_bonds = {(half - 1, half), (basis.l_sites - 1, 0)}
    within: list[float] = []
    boundary: list[float] = []
    for site in range(basis.l_sites):
        neighbor = (site + 1) % basis.l_sites
        value = float(weights @ (basis.occupations[:, site] * basis.occupations[:, neighbor]))
        (boundary if (site, neighbor) in boundary_bonds else within).append(value)
    result = {
        f"state_site_occupation_{site:02d}": float(site_values[site])
        for site in range(basis.l_sites)
    }
    result.update(
        {
            "state_half_occupation": float(site_values[:half].sum()),
            "state_density_correlation_within": float(np.mean(within)),
            "state_density_correlation_boundary": float(np.mean(boundary)),
        }
    )
    return result


def energy(hamiltonian: csr_matrix, state: ComplexVector) -> float:
    return float(np.vdot(state, hamiltonian @ state).real)


def observe(
    *,
    events: Sequence[EventRecord],
    state: ComplexVector,
    endpoint: float,
    basis: FixedNumberBasis,
    hold_hamiltonian: csr_matrix,
    gamma: float,
    observation_horizon: float,
) -> Observation:
    values: dict[str, float] = {}
    values.update(
        receiver_coordinates(
            events,
            start=endpoint - observation_horizon,
            end=endpoint,
            l_sites=basis.l_sites,
            particles=basis.particles,
            gamma=gamma,
        )
    )
    chart, last_present, boundary_present = chart_coordinates(
        events,
        cutoff=endpoint,
        l_sites=basis.l_sites,
        particles=basis.particles,
        gamma=gamma,
    )
    values.update(chart)
    values.update(state_coordinates(basis, state))
    if tuple(values) != CONTINUOUS_COORDINATES:
        raise AssertionError("observation coordinate order differs from quantum trajectory preparation qualification")
    return Observation(
        endpoint=endpoint,
        values=values,
        energy=energy(hold_hamiltonian, state),
        event_rate=values["count_total"] / observation_horizon,
        state_sha256=sha256(np.asarray(state, dtype="<c16").tobytes()).hexdigest(),
        last_event_present=last_present,
        last_boundary_event_present=boundary_present,
    )


__all__ = [
    "CHART_NUMERIC_COORDINATES",
    "CHART_ONE_HOT_COORDINATES",
    "CONTINUOUS_COORDINATES",
    "Observation",
    "RECEIVER_COORDINATES",
    "STATE_COORDINATES",
    "chart_coordinates",
    "energy",
    "observe",
    "receiver_coordinates",
    "state_coordinates",
]

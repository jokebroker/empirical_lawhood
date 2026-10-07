"""Fixed-number bases and deterministic parent-preparation rosters."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from math import comb
from typing import Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from ..quantum_scientific_preparations import ScientificPreparations, validate_scientific_preparations

from .schemas import QuantumTrajectoryPreparationQualificationConfig, RosterSpec, stable_json_bytes
from .types import ComplexVector, PreparationUnit


@dataclass(frozen=True, slots=True)
class FixedNumberBasis:
    l_sites: int
    particles: int
    states: tuple[int, ...]
    index: Mapping[int, int]
    occupations: NDArray[np.float64]

    @property
    def dimension(self) -> int:
        return len(self.states)

    @property
    def fingerprint(self) -> str:
        return sha256(np.asarray(self.states, dtype="<u8").tobytes()).hexdigest()

    def state_vector(self, state: int) -> ComplexVector:
        try:
            index = self.index[state]
        except KeyError as error:
            raise ValueError("initial state is outside the fixed-number basis") from error
        result = np.zeros(self.dimension, dtype=np.complex128)
        result[index] = 1.0
        return result


def basis_states(l_sites: int, particles: int) -> tuple[int, ...]:
    if l_sites <= 1 or particles <= 0 or particles >= l_sites:
        raise ValueError("fixed-number basis dimensions are invalid")
    return tuple(
        sorted(
            sum(1 << site for site in occupied)
            for occupied in combinations(range(l_sites), particles)
        )
    )


def build_basis(l_sites: int, particles: int) -> FixedNumberBasis:
    states = basis_states(l_sites, particles)
    occupations = np.asarray(
        [[(state >> site) & 1 for site in range(l_sites)] for state in states],
        dtype=np.float64,
    )
    return FixedNumberBasis(
        l_sites=l_sites,
        particles=particles,
        states=states,
        index={state: index for index, state in enumerate(states)},
        occupations=occupations,
    )




def target_weight(l_sites: int, particles: int, k_left: int) -> float:
    half = l_sites // 2
    if not 0 <= k_left <= particles:
        raise ValueError("preparation stratum is invalid")
    return comb(half, k_left) * comb(half, particles - k_left) / comb(l_sites, particles)


def translation_orbit_id(state: int, l_sites: int) -> int:
    mask = (1 << l_sites) - 1
    rotations = [
        ((state << offset) | (state >> (l_sites - offset))) & mask for offset in range(l_sites)
    ]
    return min(rotations)


def compile_roster(
    spec: RosterSpec,
    *,
    scientific_preparations: ScientificPreparations,
    cumulative_allocations: Mapping[int, tuple[int, ...]] | None = None,
) -> tuple[PreparationUnit, ...]:
    """Bind explicit state/seed rows to the exact independent-parent allocation."""
    half = spec.l_sites // 2
    sector_order = tuple(k for k, count in enumerate(spec.allocation) for _ in range(count))
    if cumulative_allocations is not None:
        schedule: list[int] = []
        previous = (0,) * len(spec.allocation)
        for sample_size, allocation in sorted(cumulative_allocations.items()):
            if len(allocation) != len(previous) or sum(allocation) != sample_size or any(
                current < prior for current, prior in zip(allocation, previous, strict=True)
            ):
                raise ValueError("nested cumulative allocation is invalid")
            schedule.extend(k for k, (prior, current) in enumerate(zip(previous, allocation, strict=True)) for _ in range(prior, current))
            previous = allocation
        if previous != spec.allocation:
            raise ValueError("nested allocation does not terminate at the roster")
        sector_order = tuple(schedule)
    validate_scientific_preparations(scientific_preparations, l_sites=spec.l_sites, particles=spec.particles, sector_order=sector_order, seed_bits=64)
    if len(sector_order) != spec.units:
        raise ValueError("roster allocation differs from its declared independent-parent count")
    units: list[PreparationUnit] = []
    for unit_index, (k_left, state, preparation_seed) in enumerate(scientific_preparations):
        units.append(PreparationUnit(
            roster_id=spec.roster_id,
            unit_id=f"unit.quantum-trajectory-preparation-qualification.{spec.roster_id.lower().replace('_', '-')}.{unit_index + 1:05d}",
            unit_index=unit_index,
            l_sites=spec.l_sites,
            particles=spec.particles,
            k_left=k_left,
            initial_state=state,
            boundary_occupation=sum((state >> site) & 1 for site in (0, half - 1, half, spec.l_sites - 1)),
            translation_orbit=translation_orbit_id(state, spec.l_sites),
            preparation_seed=preparation_seed,
        ))
    return tuple(units)


def compile_all_rosters(config: QuantumTrajectoryPreparationQualificationConfig, *, scientific_preparations: Mapping[str, ScientificPreparations]) -> dict[str, tuple[PreparationUnit, ...]]:
    if set(scientific_preparations) != {spec.roster_id for spec in config.roster_specs}:
        raise ValueError("scientific preparation census omits or adds a roster")
    result = {
        spec.roster_id: compile_roster(
            spec,
            scientific_preparations=scientific_preparations[spec.roster_id],
            cumulative_allocations=(
                config.cumulative_allocations if spec.roster_id == "preparation-development" else None
            ),
        )
        for spec in config.roster_specs
    }
    for sample_size, allocation in config.cumulative_allocations.items():
        observed = tuple(
            sum(unit.k_left == k_left for unit in result["preparation-development"][:sample_size])
            for k_left in range(config.n_primary + 1)
        )
        if observed != allocation:
            raise AssertionError("development nested prefix allocation differs")
    units = [unit for roster in result.values() for unit in roster]
    if len({unit.unit_id for unit in units}) != len(units):
        raise AssertionError("quantum trajectory preparation qualification roster namespaces collide")
    if len({unit.preparation_seed for unit in units}) != len(units):
        raise AssertionError("quantum trajectory preparation qualification roster seed namespaces collide")
    return result


def roster_rows(
    rosters: Mapping[str, Sequence[PreparationUnit]],
) -> list[dict[str, object]]:
    return [
        {
            "roster_id": unit.roster_id,
            "unit_id": unit.unit_id,
            "unit_index": unit.unit_index,
            "l_sites": unit.l_sites,
            "particles": unit.particles,
            "k_left": unit.k_left,
            "initial_state": unit.initial_state,
            "boundary_occupation": unit.boundary_occupation,
            "translation_orbit": unit.translation_orbit,
            "preparation_seed": unit.preparation_seed,
        }
        for roster_id in sorted(rosters)
        for unit in rosters[roster_id]
    ]


def roster_fingerprint(rosters: Mapping[str, Sequence[PreparationUnit]]) -> str:
    return sha256(stable_json_bytes(roster_rows(rosters))).hexdigest()


__all__ = [
    "FixedNumberBasis",
    "basis_states",
    "build_basis",
    "compile_all_rosters",
    "compile_roster",
    "roster_fingerprint",
    "roster_rows",
    "target_weight",
    "translation_orbit_id",
]

"""Strict local contracts for the quantum receiver response quantum-trajectory programme.

The module owns only experiment-local identities and deterministic design
arithmetic.  It does not perform source propagation, persistence or outcome
reveal.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from itertools import combinations
import json
from math import comb
from pathlib import Path
from typing import Final, Mapping, Sequence


from ..quantum_scientific_preparations import ScientificPreparations, validate_scientific_preparations

PLAN_ID: Final = "quantum-receiver-response"
CAMPAIGN_ID: Final = "quantum-receiver-response"
EXTERNAL_ROOT: Final = "runs/quantum-receiver-response"
MINIMUM_FREE_BYTES: Final = 100 * 1024**3
MAXIMUM_JSON_BYTES: Final = 64 * 1024**2
CAPABILITY_VERSION: Final = "1.0.0"
SOURCE_CAPABILITY_KEY: Final = "open-sim.quantum-trajectory-source.quantum-receiver-response"
METHOD_CAPABILITY_KEY: Final = "method.quantum-record-response-law.quantum-receiver-response"
EVALUATOR_CAPABILITY_KEY: Final = "evaluator.quantum-receiver-law.quantum-receiver-response"


class Stage(StrEnum):
    SOURCE_CONFORMANCE = "source-conformance"
    OBSERVATION_ORDER = "observation-order"
    RESPONSE_DEVELOPMENT = "response-development"
    RESPONSE_LAW_EVALUATION = "response-law-evaluation"
    OMITTED_HISTORY_FIBERS = "omitted-history-fibers"


class Action(StrEnum):
    MINUS = "minus"
    HOLD = "hold"
    PLUS = "plus"

    @property
    def sign(self) -> int:
        return {
            Action.MINUS: -1,
            Action.HOLD: 0,
            Action.PLUS: 1,
        }[self]


class Denominator(StrEnum):
    STRONG = "strong"
    WEAK = "weak"
    DENSE_CONFORMANCE = "dense-conformance"

    @property
    def gamma(self) -> float:
        return {
            Denominator.STRONG: 1.0,
            Denominator.WEAK: 0.1,
            Denominator.DENSE_CONFORMANCE: 1.0,
        }[self]


class UnitDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    TECHNICAL_NONCONTACT_REPLACED = "TECHNICAL_NONCONTACT_REPLACED"
    SOURCE_FAILURE = "SOURCE_FAILURE"
    DELIVERY_FAILURE = "DELIVERY_FAILURE"
    OBSERVATION_INVALID = "OBSERVATION_INVALID"
    UNEVALUABLE_PRECISION = "UNEVALUABLE_PRECISION"
    NOT_ATTEMPTED_PREREQUISITE = "NOT_ATTEMPTED_PREREQUISITE"


@dataclass(frozen=True, slots=True)
class QuantumReceiverResponseConfig:
    """Decoded exact scientific configuration."""

    schema: str
    version: str
    plan_id: str
    l_primary: int
    n_primary: int
    l_dense: int
    n_dense: int
    j_xy: Decimal
    j_z: Decimal
    gamma_strong: Decimal
    gamma_weak: Decimal
    epsilon: Decimal
    burn_in: Decimal
    burn_in_extension: Decimal
    passive_horizon: Decimal
    action_horizon: Decimal
    observation_order_base_units: int
    observation_order_expansion_block_units: int
    observation_order_max_units: int
    response_development_units: int
    response_law_evaluation_base_units: int
    response_law_evaluation_expansion_block_units: int
    response_law_evaluation_max_units: int
    history_fiber_base_pool_units: int
    history_fiber_expansion_block_units: int
    history_fiber_max_pool_units: int
    history_fiber_max_pairs: int
    future_draws_initial: int
    future_draws_steps: tuple[int, ...]
    bootstrap_replicates: int
    observation_order_family_alpha: Decimal
    response_law_evaluation_family_alpha: Decimal
    history_fiber_family_alpha: Decimal
    observation_order_joint_design_power: Decimal
    observation_order_chi_h20_gate: Decimal
    observation_order_chi_h4_gate: Decimal
    observation_order_global_equivalence: Decimal
    observation_order_cancellation_equivalence: Decimal
    observation_order_weak_separation: Decimal
    observation_order_proxy_association: Decimal
    source_conformance_ensemble_draws: int
    source_conformance_family_alpha: Decimal
    source_conformance_norm_tolerance: Decimal
    source_conformance_particle_tolerance: Decimal
    source_conformance_phase_infidelity_tolerance: Decimal
    source_conformance_site_probability_tolerance: Decimal
    source_conformance_hs_split_multiplier: Decimal
    source_conformance_burnin_standardized_margin: Decimal
    source_conformance_burnin_event_rate_relative_margin: Decimal
    source_conformance_burnin_energy_margin: Decimal
    source_conformance_dense_horizon: Decimal
    source_conformance_preparation_memory_eta_squared: Decimal
    source_conformance_free_running_draws: int

    def __post_init__(self) -> None:
        if (
            self.schema != 'empirical-lawhood/simulators/quantum-receiver-response/quantum-trajectory-receiver-law'
            or self.version != "1.0.0"
            or self.plan_id != PLAN_ID
        ):
            raise ValueError("quantum receiver response config identity differs from declared contract")
        if (
            (self.l_primary, self.n_primary) != (12, 6)
            or (self.l_dense, self.n_dense) != (8, 4)
            or self.j_xy != Decimal("1")
            or self.j_z != Decimal("1")
            or self.gamma_strong != Decimal("1")
            or self.gamma_weak != Decimal("0.1")
            or self.epsilon != Decimal("0.20")
            or self.burn_in != Decimal("200")
            or self.burn_in_extension != Decimal("100")
            or self.passive_horizon != Decimal("20")
            or self.action_horizon != Decimal("4")
        ):
            raise ValueError("quantum receiver response physical denominator differs from declared contract")
        if (
            self.observation_order_base_units != 256
            or self.observation_order_expansion_block_units != 256
            or self.observation_order_max_units != 9984
            or self.response_development_units != 96
            or self.response_law_evaluation_base_units != 96
            or self.response_law_evaluation_expansion_block_units != 48
            or self.response_law_evaluation_max_units != 192
            or self.history_fiber_base_pool_units != 384
            or self.history_fiber_expansion_block_units != 192
            or self.history_fiber_max_pool_units != 768
            or self.history_fiber_max_pairs != 48
            or self.future_draws_initial != 128
            or self.future_draws_steps != (128, 256, 512)
            or self.bootstrap_replicates != 4096
            or self.source_conformance_ensemble_draws != 4096
            or self.source_conformance_free_running_draws != 512
        ):
            raise ValueError("quantum receiver response roster or precision arithmetic differs from declared contract")
        if (
            self.observation_order_family_alpha != Decimal("0.05")
            or self.response_law_evaluation_family_alpha != Decimal("0.05")
            or self.history_fiber_family_alpha != Decimal("0.05")
            or self.observation_order_joint_design_power != Decimal("0.90")
            or self.observation_order_chi_h20_gate != Decimal("0.25")
            or self.observation_order_chi_h4_gate != Decimal("0.10")
            or self.observation_order_global_equivalence != Decimal("0.20")
            or self.observation_order_cancellation_equivalence != Decimal("0.20")
            or self.observation_order_weak_separation != Decimal("0.15")
            or self.observation_order_proxy_association != Decimal("0.15")
            or self.source_conformance_family_alpha != Decimal("0.01")
            or self.source_conformance_norm_tolerance != Decimal("1e-11")
            or self.source_conformance_particle_tolerance != Decimal("1e-12")
            or self.source_conformance_phase_infidelity_tolerance != Decimal("1e-10")
            or self.source_conformance_site_probability_tolerance != Decimal("1e-10")
            or self.source_conformance_hs_split_multiplier != Decimal("1.0")
            or self.source_conformance_burnin_standardized_margin != Decimal("1.0")
            or self.source_conformance_burnin_event_rate_relative_margin != Decimal("0.15")
            or self.source_conformance_burnin_energy_margin != Decimal("0.50")
            or self.source_conformance_dense_horizon != Decimal("1.0")
            or self.source_conformance_preparation_memory_eta_squared != Decimal("0.25")
        ):
            raise ValueError("quantum receiver response primary threshold family differs from declared contract")


_CONFIG_FIELDS: Final = tuple(QuantumReceiverResponseConfig.__dataclass_fields__)
_INTEGER_FIELDS: Final = frozenset(
    {
        "l_primary",
        "n_primary",
        "l_dense",
        "n_dense",
        "observation_order_base_units",
        "observation_order_expansion_block_units",
        "observation_order_max_units",
        "response_development_units",
        "response_law_evaluation_base_units",
        "response_law_evaluation_expansion_block_units",
        "response_law_evaluation_max_units",
        "history_fiber_base_pool_units",
        "history_fiber_expansion_block_units",
        "history_fiber_max_pool_units",
        "history_fiber_max_pairs",
        "future_draws_initial",
        "bootstrap_replicates",
        "source_conformance_ensemble_draws",
        "source_conformance_free_running_draws",
    }
)
_DECIMAL_FIELDS: Final = frozenset(
    {
        "j_xy",
        "j_z",
        "gamma_strong",
        "gamma_weak",
        "epsilon",
        "burn_in",
        "burn_in_extension",
        "passive_horizon",
        "action_horizon",
        "observation_order_family_alpha",
        "response_law_evaluation_family_alpha",
        "history_fiber_family_alpha",
        "observation_order_joint_design_power",
        "observation_order_chi_h20_gate",
        "observation_order_chi_h4_gate",
        "observation_order_global_equivalence",
        "observation_order_cancellation_equivalence",
        "observation_order_weak_separation",
        "observation_order_proxy_association",
        "source_conformance_family_alpha",
        "source_conformance_norm_tolerance",
        "source_conformance_particle_tolerance",
        "source_conformance_phase_infidelity_tolerance",
        "source_conformance_site_probability_tolerance",
        "source_conformance_hs_split_multiplier",
        "source_conformance_burnin_standardized_margin",
        "source_conformance_burnin_event_rate_relative_margin",
        "source_conformance_burnin_energy_margin",
        "source_conformance_dense_horizon",
        "source_conformance_preparation_memory_eta_squared",
    }
)


def decode_config(document: Mapping[str, object]) -> QuantumReceiverResponseConfig:
    """Reject unknown fields, binary-float scientific values and coercions."""

    if set(document) != set(_CONFIG_FIELDS):
        missing = sorted(set(_CONFIG_FIELDS) - set(document))
        unknown = sorted(set(document) - set(_CONFIG_FIELDS))
        raise ValueError(f"quantum receiver response config fields differ: missing={missing}, unknown={unknown}")
    values: dict[str, object] = {}
    for field_name in _CONFIG_FIELDS:
        value = document[field_name]
        if field_name in _INTEGER_FIELDS:
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{field_name} must be an exact integer")
        elif field_name in _DECIMAL_FIELDS:
            if not isinstance(value, str):
                raise ValueError(f"{field_name} must be a decimal string")
            value = Decimal(value)
        elif field_name == "future_draws_steps":
            if not isinstance(value, list) or any(
                isinstance(item, bool) or not isinstance(item, int) for item in value
            ):
                raise ValueError("future_draws_steps must be an integer list")
            value = tuple(value)
        elif not isinstance(value, str):
            raise ValueError(f"{field_name} must be a string")
        values[field_name] = value
    return QuantumReceiverResponseConfig(**values)  # type: ignore[arg-type]


def load_config(path: Path) -> tuple[QuantumReceiverResponseConfig, bytes]:
    payload = path.read_bytes()
    document = json.loads(payload)
    if not isinstance(document, Mapping):
        raise ValueError("quantum receiver response config root must be an object")
    config = decode_config(document)
    canonical = stable_json_bytes(document)
    if canonical != payload:
        raise ValueError("quantum receiver response repository config must use canonical JSON bytes")
    return config, payload


def stable_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def sha256_hex(payload: bytes) -> str:
    return sha256(payload).hexdigest()




def basis_states(l_sites: int, particles: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            sum(1 << site for site in occupied)
            for occupied in combinations(range(l_sites), particles)
        )
    )


def preparation_weight(l_sites: int, particles: int, k_left: int) -> Decimal:
    half = l_sites // 2
    numerator = comb(half, k_left) * comb(half, particles - k_left)
    return Decimal(numerator) / Decimal(comb(l_sites, particles))


def translation_orbit_id(state: int, l_sites: int) -> int:
    mask = (1 << l_sites) - 1
    rotations = []
    for offset in range(l_sites):
        rotations.append(((state << offset) | (state >> (l_sites - offset))) & mask)
    return min(rotations)


@dataclass(frozen=True, slots=True)
class PreparationUnit:
    roster_id: str
    unit_id: str
    unit_index: int
    k_left: int
    initial_state: int
    boundary_occupation: int
    translation_orbit: int
    preparation_seed: int


def stratum_allocation(
    *,
    total_units: int,
    l_sites: int,
    particles: int,
    minimum_per_stratum: int = 2,
) -> tuple[int, ...]:
    strata = tuple(range(particles + 1))
    minimum_total = minimum_per_stratum * len(strata)
    if total_units < minimum_total:
        raise ValueError("roster cannot cover every preparation stratum")
    remaining = total_units - minimum_total
    raw = [Decimal(remaining) * preparation_weight(l_sites, particles, value) for value in strata]
    floors = [int(value) for value in raw]
    residual = remaining - sum(floors)
    order = sorted(
        strata,
        key=lambda value: (-(raw[value] - floors[value]), value),
    )
    for value in order[:residual]:
        floors[value] += 1
    return tuple(minimum_per_stratum + value for value in floors)


def compile_roster(
    *,
    roster_id: str,
    total_units: int,
    scientific_preparations: ScientificPreparations,
    minimum_per_stratum: int = 2,
    l_sites: int = 12,
    particles: int = 6,
) -> tuple[PreparationUnit, ...]:
    """Bind numerical state/seed rows without allocating from caller labels."""
    allocation = stratum_allocation(total_units=total_units, l_sites=l_sites, particles=particles, minimum_per_stratum=minimum_per_stratum)
    sector_order = tuple(k for k in range(particles + 1) for _ in range(allocation[k]))
    validate_scientific_preparations(scientific_preparations, l_sites=l_sites, particles=particles, sector_order=sector_order, seed_bits=64)
    half = l_sites // 2
    units: list[PreparationUnit] = []
    for index, (k_left, state, preparation_seed) in enumerate(scientific_preparations):
        units.append(PreparationUnit(
            unit_id=f"unit.quantum-receiver-response.{roster_id.lower()}.{index + 1:05d}",
            roster_id=roster_id,
            unit_index=index,
            k_left=k_left,
            initial_state=state,
            boundary_occupation=sum((state >> site) & 1 for site in (0, half - 1, half, l_sites - 1)),
            translation_orbit=translation_orbit_id(state, l_sites),
            preparation_seed=preparation_seed,
        ))
    return tuple(units)


def roster_fingerprint(units: Sequence[PreparationUnit]) -> str:
    rows = [
        {
            "roster_id": unit.roster_id,
            "unit_id": unit.unit_id,
            "unit_index": unit.unit_index,
            "k_left": unit.k_left,
            "initial_state": unit.initial_state,
            "boundary_occupation": unit.boundary_occupation,
            "translation_orbit": unit.translation_orbit,
            "preparation_seed": unit.preparation_seed,
        }
        for unit in units
    ]
    return sha256_hex(stable_json_bytes(rows))


def all_roster_ids(config: QuantumReceiverResponseConfig) -> tuple[str, ...]:
    return (
        "source-conformance-dense",
        "source-conformance-path-dense",
        "source-conformance-path-primary",
        "observation-order-evaluation-maximum",
        "observation-order-reserve",
        "response-development",
        "response-development-reserve",
        "response-law-evaluation-maximum",
        "response-law-evaluation-reserve",
        "omitted-history-fiber-pool-maximum",
        "omitted-history-fiber-pool-reserve",
    )


def compile_all_rosters(config: QuantumReceiverResponseConfig, *, scientific_preparations: Mapping[str, ScientificPreparations]) -> dict[str, tuple[PreparationUnit, ...]]:
    specs = {
        "source-conformance-dense": (4, 8, 4, 0),
        "source-conformance-path-dense": (16, 8, 4, 2),
        "source-conformance-path-primary": (16, 12, 6, 2),
        "observation-order-evaluation-maximum": (config.observation_order_max_units, 12, 6, 2),
        "observation-order-reserve": (24, 12, 6, 2),
        "response-development": (config.response_development_units, 12, 6, 4),
        "response-development-reserve": (12, 12, 6, 1),
        "response-law-evaluation-maximum": (config.response_law_evaluation_max_units, 12, 6, 2),
        "response-law-evaluation-reserve": (12, 12, 6, 1),
        "omitted-history-fiber-pool-maximum": (config.history_fiber_max_pool_units, 12, 6, 2),
        "omitted-history-fiber-pool-reserve": (48, 12, 6, 2),
    }
    if set(scientific_preparations) != set(specs):
        raise ValueError("scientific preparation census omits or adds a roster")
    result = {
        roster_id: compile_roster(
            roster_id=roster_id,
            total_units=count,
            scientific_preparations=scientific_preparations[roster_id],
            l_sites=l_sites,
            particles=particles,
            minimum_per_stratum=minimum,
        )
        for roster_id, (count, l_sites, particles, minimum) in specs.items()
    }
    unit_ids = [unit.unit_id for roster in result.values() for unit in roster]
    if len(unit_ids) != len(set(unit_ids)):
        raise AssertionError("quantum receiver response roster namespaces collide")
    return result


__all__ = [
    "Action",
    "CAMPAIGN_ID",
    "CAPABILITY_VERSION",
    "Denominator",
    "EVALUATOR_CAPABILITY_KEY",
    "EXTERNAL_ROOT",
    "MAXIMUM_JSON_BYTES",
    "METHOD_CAPABILITY_KEY",
    "MINIMUM_FREE_BYTES",
    "PLAN_ID",
    "PreparationUnit",
    'QuantumReceiverResponseConfig',
    "SOURCE_CAPABILITY_KEY",
    "Stage",
    "UnitDisposition",
    "all_roster_ids",
    "basis_states",
    "compile_all_rosters",
    "compile_roster",
    "decode_config",
    "load_config",
    "preparation_weight",
    "roster_fingerprint",
    "sha256_hex",
    "stable_json_bytes",
    "stratum_allocation",
    "translation_orbit_id",
]

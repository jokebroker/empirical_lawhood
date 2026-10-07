"""Strict identities and deterministic design arithmetic for quantum response control.

This module is intentionally experiment-local.  It performs no source
propagation, persistence, authority issuance, or outcome reveal.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
import json
from math import comb
from pathlib import Path
from typing import Final, Mapping, Sequence


from ..quantum_scientific_preparations import ScientificPreparations, validate_scientific_preparations

PLAN_ID: Final = "quantum-response-control"
CHILD_PLAN_ID: Final = "quantum-receiver-admission-and-control"
CAMPAIGN_ID: Final = "quantum-response-control"
EXTERNAL_ROOT: Final = "runs/quantum-response-control"
CANARY_ROOT: Final = "runs/quantum-response-control-capacity-canary"
CAPABILITY_VERSION: Final = "1.0.0"
SOURCE_CAPABILITY_KEY: Final = "open-sim.quantum-trajectory-source.quantum-response-control"
ACQUISITION_CAPABILITY_KEY: Final = "service.quantum-sealed-acquisition.quantum-response-control"
REDUCER_CAPABILITY_KEY: Final = "method.quantum-receiver-reducer.quantum-response-control"
RESPONSE_CAPABILITY_KEY: Final = "method.quantum-record-response-law.quantum-response-control"
ATLAS_CAPABILITY_KEY: Final = "geometry.quantum-response-atlas.quantum-response-control"
CONTROLLER_CAPABILITY_KEY: Final = "control.quantum-receiver-lookup.quantum-response-control"
EVALUATOR_CAPABILITY_KEY: Final = "evaluator.quantum-receiver-law.quantum-response-control"
MAXIMUM_JSON_BYTES: Final = 64 * 1024**2


class Stage(StrEnum):
    SOURCE_COMPATIBILITY = "source-compatibility"
    INSTRUMENT_CONFORMANCE = "instrument-conformance"
    OBSERVATION_ORDER_EVALUATION = "observation-order-evaluation"
    RESPONSE_DEVELOPMENT = "response-development"
    RESPONSE_LAW_EVALUATION = "response-law-evaluation"
    OMITTED_HISTORY_FIBERS = "omitted-history-fibers"
    LAW_ADMISSION = "law-admission"
    CONTROLLER_CONFORMANCE = "controller-conformance"
    PROSPECTIVE_EVALUATION = "prospective-evaluation"
    CLOSEOUT = "closeout"


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


class Branch(StrEnum):
    POLICY = "POLICY"
    HOLD = "HOLD"
    OPEN_LOOP_MATCHED = "OPEN_LOOP_MATCHED"
    HISTORY_SHUFFLED = "HISTORY_SHUFFLED"
    WRONG_RECEIVER = "WRONG_RECEIVER"


class Disposition(StrEnum):
    COMPLETE = "COMPLETE"
    TECHNICAL_NONCONTACT_REPLACED = "TECHNICAL_NONCONTACT_REPLACED"
    SOURCE_FAILURE = "SOURCE_FAILURE"
    DELIVERY_FAILURE = "DELIVERY_FAILURE"
    OBSERVATION_INVALID = "OBSERVATION_INVALID"
    UNEVALUABLE_PRECISION = "UNEVALUABLE_PRECISION"
    NOT_ATTEMPTED_PREREQUISITE = "NOT_ATTEMPTED_PREREQUISITE"
    NOT_SELECTED_PREFIX_ONLY = "NOT_SELECTED_PREFIX_ONLY"


class Verdict(StrEnum):
    COMPATIBLE_STRONG_SOURCE_READY = "COMPATIBLE_STRONG_SOURCE_READY"
    PREDECESSOR_RECEIPT_STOP = "PREDECESSOR_RECEIPT_STOP"
    SOURCE_COMPATIBILITY_STOP = "SOURCE_COMPATIBILITY_STOP"
    ACTION_COMPATIBILITY_STOP = "ACTION_COMPATIBILITY_STOP"
    IMPLEMENTATION_IDENTITY_STOP = "IMPLEMENTATION_IDENTITY_STOP"
    CONTROL_INSTRUMENTS_VALID = "CONTROL_INSTRUMENTS_VALID"
    STATISTICAL_INSTRUMENT_STOP = "STATISTICAL_INSTRUMENT_STOP"
    TRANSPORT_INSTRUMENT_STOP = "TRANSPORT_INSTRUMENT_STOP"
    ATLAS_ADMISSION_INSTRUMENT_STOP = "ATLAS_ADMISSION_INSTRUMENT_STOP"
    CONTROLLER_INSTRUMENT_STOP = "CONTROLLER_INSTRUMENT_STOP"
    OBSERVATION_ORDER_RECEIVER_OPPORTUNITY_SUPPORTED = "OBSERVATION_ORDER_RECEIVER_OPPORTUNITY_SUPPORTED"
    OBSERVATION_ORDER_RECEIVER_PROXY_STOP = "OBSERVATION_ORDER_RECEIVER_PROXY_STOP"
    OBSERVATION_ORDER_RECEIVER_ORDER_MIXED = "OBSERVATION_ORDER_RECEIVER_ORDER_MIXED"
    OBSERVATION_ORDER_NO_RECEIVER_OPPORTUNITY = "OBSERVATION_ORDER_NO_RECEIVER_OPPORTUNITY"
    OBSERVATION_ORDER_GLOBAL_NULL_OPPOSED = "OBSERVATION_ORDER_GLOBAL_NULL_OPPOSED"
    OBSERVATION_ORDER_UNEVALUABLE = "OBSERVATION_ORDER_UNEVALUABLE"
    RECORD_CHART_SUPPORTED = "RECORD_CHART_SUPPORTED"
    DENOMINATOR_LOCAL_MIXED = "DENOMINATOR_LOCAL_MIXED"
    COMPLETE_STATE_RESPONSE_ONLY = "COMPLETE_STATE_RESPONSE_ONLY"
    NO_CAUSAL_RESPONSE = "NO_CAUSAL_RESPONSE"
    PROXY_CAUSAL_MISMATCH = "PROXY_CAUSAL_MISMATCH"
    RESPONSE_LAW_TRANSPORT_OR_SINK_STOP = "RESPONSE_LAW_TRANSPORT_OR_SINK_STOP"
    RESPONSE_LAW_SOURCE_OR_DELIVERY_STOP = "RESPONSE_LAW_SOURCE_OR_DELIVERY_STOP"
    RESPONSE_LAW_UNEVALUABLE = "RESPONSE_LAW_UNEVALUABLE"
    FINITE_CONTROLLER_FIBER_CLOSURE = "FINITE_CONTROLLER_FIBER_CLOSURE"
    CONTROLLER_FIBER_DIVERGENCE = "CONTROLLER_FIBER_DIVERGENCE"
    CONTROLLER_FIBERS_MIXED = "CONTROLLER_FIBERS_MIXED"
    CONTROLLER_MASK_UNDERCOVERED = "CONTROLLER_MASK_UNDERCOVERED"
    FIBER_UNEVALUABLE = "FIBER_UNEVALUABLE"
    FIBER_SOURCE_STOP = "FIBER_SOURCE_STOP"
    PROSPECTIVE_VALIDATED_RECEIVER_ADMITTED_LOCAL = "PROSPECTIVE_VALIDATED_RECEIVER_ADMITTED_LOCAL"
    PROSPECTIVE_EFFECTIVE_NOT_RECEIVER_SPECIFIC = "PROSPECTIVE_EFFECTIVE_NOT_RECEIVER_SPECIFIC"
    PROSPECTIVE_NO_INCREMENTAL_CHART_VALUE = "PROSPECTIVE_NO_INCREMENTAL_CHART_VALUE"
    PROSPECTIVE_REJECTED_TARGET = "PROSPECTIVE_REJECTED_TARGET"
    PROSPECTIVE_REJECTED_SINK_OR_PRESERVATION = "PROSPECTIVE_REJECTED_SINK_OR_PRESERVATION"
    PROSPECTIVE_REJECTED_FALSE_ACTION = "PROSPECTIVE_REJECTED_FALSE_ACTION"
    PROSPECTIVE_SAFE_HOLD_ONLY = "PROSPECTIVE_SAFE_HOLD_ONLY"
    PROSPECTIVE_COVERAGE_NONATTEMPT = "PROSPECTIVE_COVERAGE_NONATTEMPT"
    PROSPECTIVE_UNEVALUABLE = "PROSPECTIVE_UNEVALUABLE"
    PROSPECTIVE_SOURCE_OR_DELIVERY_STOP = "PROSPECTIVE_SOURCE_OR_DELIVERY_STOP"


@dataclass(frozen=True, slots=True)
class QuantumResponseControlConfig:
    schema: str
    version: str
    plan_id: str
    campaign_id: str
    source_capability_key: str
    acquisition_capability_key: str
    receiver_capability_key: str
    response_capability_key: str
    atlas_capability_key: str
    controller_capability_key: str
    evaluator_capability_key: str
    l_sites: int
    particles: int
    j_xy: Decimal
    j_z: Decimal
    gamma: Decimal
    epsilon: Decimal
    cutoff: Decimal
    action_horizon: Decimal
    history_short: Decimal
    history_long: Decimal
    bootstrap_replicates: int
    family_alpha: Decimal
    observation_order_base_parents: int
    observation_order_expansion_block: int
    observation_order_max_parents: int
    observation_order_reserve: int
    observation_order_chi_h20_gate: Decimal
    observation_order_chi_h4_gate: Decimal
    observation_order_global_equivalence: Decimal
    observation_order_cancellation_equivalence: Decimal
    observation_order_proxy_association: Decimal
    observation_order_thinning_h20_advantage: Decimal
    observation_order_thinning_h4_advantage: Decimal
    observation_order_history_advantage: Decimal
    observation_order_wrong_time_advantage: Decimal
    response_development_parents: int
    response_development_fit_parents: int
    response_development_selection_parents: int
    response_development_reserve: int
    response_development_draws: int
    response_law_evaluation_base_parents: int
    response_law_evaluation_expansion_block: int
    response_law_evaluation_max_parents: int
    response_law_evaluation_reserve: int
    future_draw_steps: tuple[int, ...]
    knn_neighbors: tuple[int, ...]
    support_quantile: Decimal
    chart_law_support_fraction: Decimal
    chart_law_primary_adequacy: Decimal
    chart_law_full_adequacy: Decimal
    fiber_base_pool: int
    fiber_expansion_block: int
    fiber_max_pool: int
    fiber_reserve: int
    fiber_min_pairs: int
    fiber_min_pairs_per_contrast: int
    fiber_max_pairs: int
    quadrature_order: int
    quadrature_sentinel_order: int
    continuity_tolerance: Decimal
    law_admission_transport_activity_ratio: Decimal
    latency_deadline: Decimal
    prospective_base_parents: int
    prospective_expansion_block: int
    prospective_max_parents: int
    prospective_reserve: int
    prospective_min_action_parents: int
    prospective_min_hold_parents: int
    prospective_min_action_each: int
    minimum_free_bytes: int

    def __post_init__(self) -> None:
        if (
            self.schema != 'empirical-lawhood/simulators/quantum-response-control/quantum-trajectory-receiver-law'
            or self.version != CAPABILITY_VERSION
            or self.plan_id != PLAN_ID
            or self.campaign_id != CAMPAIGN_ID
        ):
            raise ValueError("quantum response control config identity differs")
        if (
            self.source_capability_key != SOURCE_CAPABILITY_KEY
            or self.acquisition_capability_key != ACQUISITION_CAPABILITY_KEY
            or self.receiver_capability_key != REDUCER_CAPABILITY_KEY
            or self.response_capability_key != RESPONSE_CAPABILITY_KEY
            or self.atlas_capability_key != ATLAS_CAPABILITY_KEY
            or self.controller_capability_key != CONTROLLER_CAPABILITY_KEY
            or self.evaluator_capability_key != EVALUATOR_CAPABILITY_KEY
        ):
            raise ValueError("quantum response control config selects an unknown capability")
        if (
            (self.l_sites, self.particles) != (12, 6)
            or self.j_xy != Decimal("1")
            or self.j_z != Decimal("1")
            or self.gamma != Decimal("1")
            or self.epsilon != Decimal("0.20")
            or self.cutoff != Decimal("200")
            or self.action_horizon != Decimal("4")
        ):
            raise ValueError("quantum response control physical denominator differs")
        expected_short = Decimal(4) / (self.gamma * self.particles)
        expected_long = Decimal(16) / (self.gamma * self.particles)
        if abs(self.history_short - expected_short) > Decimal("1e-25"):
            raise ValueError("short history horizon differs")
        if abs(self.history_long - expected_long) > Decimal("1e-25"):
            raise ValueError("long history horizon differs")
        if self.future_draw_steps != (128, 256, 512):
            raise ValueError("future-draw ladder differs")
        if self.knn_neighbors != (8, 16):
            raise ValueError("neighbour roster differs")
        if (
            self.response_development_fit_parents + self.response_development_selection_parents != self.response_development_parents
            or self.response_development_fit_parents != self.response_development_selection_parents
        ):
            raise ValueError("response-development split is invalid")
        if not (
            0 < self.observation_order_base_parents <= self.observation_order_max_parents
            and 0 < self.response_law_evaluation_base_parents <= self.response_law_evaluation_max_parents
            and 0 < self.fiber_base_pool <= self.fiber_max_pool
            and 0 < self.prospective_base_parents <= self.prospective_max_parents
        ):
            raise ValueError("base/max roster ordering is invalid")
        if self.fiber_min_pairs > self.fiber_max_pairs:
            raise ValueError("fiber pair floor exceeds maximum")
        if self.quadrature_order >= self.quadrature_sentinel_order:
            raise ValueError("quadrature sentinel must refine production")
        if self.minimum_free_bytes < 100 * 1024**3:
            raise ValueError("external storage floor is too small")
        for name in (
            "family_alpha",
            "support_quantile",
            "chart_law_support_fraction",
            "chart_law_primary_adequacy",
            "chart_law_full_adequacy",
            "law_admission_transport_activity_ratio",
        ):
            value = getattr(self, name)
            if not Decimal(0) < value < Decimal(1):
                raise ValueError(f"{name} must lie strictly inside (0,1)")

    @property
    def burst_interval(self) -> Decimal:
        return Decimal(2).ln() / (self.gamma * self.particles)


_INTEGER_FIELDS: Final = frozenset(
    field.name for field in fields(QuantumResponseControlConfig) if field.type == "int" or field.type is int
)
_TUPLE_INTEGER_FIELDS: Final = frozenset({"future_draw_steps", "knn_neighbors"})
_STRING_FIELDS: Final = frozenset(
    {
        "schema",
        "version",
        "plan_id",
        "campaign_id",
        "source_capability_key",
        "acquisition_capability_key",
        "receiver_capability_key",
        "response_capability_key",
        "atlas_capability_key",
        "controller_capability_key",
        "evaluator_capability_key",
    }
)


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
    ).encode("utf-8")


def sha256_hex(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def decode_config(document: Mapping[str, object]) -> QuantumResponseControlConfig:
    expected = {field.name for field in fields(QuantumResponseControlConfig)}
    observed = set(document)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"quantum response control config keys differ: missing={missing}, extra={extra}")
    values: dict[str, object] = {}
    for name in sorted(expected):
        raw = document[name]
        if name in _STRING_FIELDS:
            if not isinstance(raw, str) or not raw:
                raise TypeError(f"{name} must be a nonempty string")
            values[name] = raw
        elif name in _INTEGER_FIELDS:
            if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
                raise TypeError(f"{name} must be a positive integer")
            values[name] = raw
        elif name in _TUPLE_INTEGER_FIELDS:
            if (
                not isinstance(raw, list)
                or not raw
                or any(
                    isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in raw
                )
            ):
                raise TypeError(f"{name} must be a nonempty positive-integer list")
            values[name] = tuple(raw)
        else:
            if not isinstance(raw, str):
                raise TypeError(f"{name} must be an exact decimal string")
            try:
                value = Decimal(raw)
            except Exception as error:
                raise ValueError(f"{name} is not an exact decimal") from error
            if not value.is_finite() or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
            values[name] = value
    return QuantumResponseControlConfig(**values)  # type: ignore[arg-type]


def load_config(path: Path) -> tuple[QuantumResponseControlConfig, bytes]:
    payload = path.read_bytes()
    if len(payload) > MAXIMUM_JSON_BYTES:
        raise ValueError("quantum response control config exceeds the bounded JSON limit")
    document = json.loads(payload)
    if not isinstance(document, Mapping):
        raise TypeError("quantum response control config root must be an object")
    config = decode_config(document)
    return config, payload


def preparation_weight(l_sites: int, particles: int, k_left: int) -> Decimal:
    half = l_sites // 2
    if not 0 <= k_left <= particles:
        raise ValueError("preparation stratum is invalid")
    numerator = comb(half, k_left) * comb(half, particles - k_left)
    return Decimal(numerator) / Decimal(comb(l_sites, particles))


def stratum_allocation(
    total_units: int,
    *,
    l_sites: int = 12,
    particles: int = 6,
    minimum_per_stratum: int = 2,
) -> dict[int, int]:
    if total_units < (particles + 1) * minimum_per_stratum:
        raise ValueError("roster cannot cover every preparation stratum")
    weights = {k: preparation_weight(l_sites, particles, k) for k in range(particles + 1)}
    allocation = {k: minimum_per_stratum for k in weights}
    remaining = total_units - sum(allocation.values())
    ideals = {k: Decimal(remaining) * weight for k, weight in weights.items()}
    for k, ideal in ideals.items():
        allocation[k] += int(ideal)
    unassigned = total_units - sum(allocation.values())
    order = sorted(
        weights,
        key=lambda k: (-(ideals[k] - int(ideals[k])), k),
    )
    for k in order[:unassigned]:
        allocation[k] += 1
    if sum(allocation.values()) != total_units:
        raise AssertionError("stratum allocation failed")
    return allocation


def translation_orbit_id(state: int, l_sites: int) -> int:
    mask = (1 << l_sites) - 1
    rotations = [
        ((state << offset) | (state >> (l_sites - offset))) & mask for offset in range(l_sites)
    ]
    return min(rotations)


@dataclass(frozen=True, slots=True)
class PreparationUnit:
    unit_id: str
    roster_id: str
    unit_index: int
    k_left: int
    initial_state: int
    boundary_occupation: int
    translation_orbit: int
    preparation_seed: int




def compile_roster(
    *,
    roster_id: str,
    size: int,
    scientific_preparations: ScientificPreparations,
    minimum_per_stratum: int = 2,
    l_sites: int = 12,
    particles: int = 6,
) -> tuple[PreparationUnit, ...]:
    """Bind numerical state/seed rows without allocating from caller labels."""
    allocation = stratum_allocation(size, l_sites=l_sites, particles=particles, minimum_per_stratum=minimum_per_stratum)
    sector_order = tuple(k for k in range(particles + 1) for _ in range(allocation[k]))
    validate_scientific_preparations(scientific_preparations, l_sites=l_sites, particles=particles, sector_order=sector_order, seed_bits=128)
    half = l_sites // 2
    units: list[PreparationUnit] = []
    for index, (k_left, state, preparation_seed) in enumerate(scientific_preparations):
        units.append(PreparationUnit(
            unit_id=f"{roster_id.lower()}.{index:05d}",
            roster_id=roster_id,
            unit_index=index,
            k_left=k_left,
            initial_state=state,
            boundary_occupation=sum((state >> site) & 1 for site in (0, half - 1, half, l_sites - 1)),
            translation_orbit=translation_orbit_id(state, l_sites),
            preparation_seed=preparation_seed,
        ))
    return tuple(units)


def target_weights(units: Sequence[PreparationUnit]) -> tuple[float, ...]:
    counts = {k: sum(unit.k_left == k for unit in units) for k in range(7)}
    if any(value == 0 for value in counts.values()):
        raise ValueError("roster omits a preparation stratum")
    weights = tuple(
        float(preparation_weight(12, 6, unit.k_left)) / counts[unit.k_left] for unit in units
    )
    total = sum(weights)
    return tuple(value / total for value in weights)


def roster_manifest(config: QuantumResponseControlConfig, *, scientific_preparations: Mapping[str, ScientificPreparations]) -> dict[str, object]:
    specifications = (
        ("observation-order-evaluation-base", config.observation_order_base_parents, 32),
        ("observation-order-expansion", config.observation_order_max_parents - config.observation_order_base_parents, 32),
        ("observation-order-reserve", config.observation_order_reserve, 8),
        ("response-development", config.response_development_parents, 16),
        ("response-development-reserve", config.response_development_reserve, 4),
        ("response-law-evaluation-base", config.response_law_evaluation_base_parents, 16),
        ("response-law-evaluation-expansion", config.response_law_evaluation_max_parents - config.response_law_evaluation_base_parents, 16),
        ("response-law-evaluation-reserve", config.response_law_evaluation_reserve, 8),
        ("omitted-history-fiber-pool-base", config.fiber_base_pool, 24),
        ("omitted-history-fiber-pool-expansion", config.fiber_max_pool - config.fiber_base_pool, 24),
        ("omitted-history-fiber-reserve", config.fiber_reserve, 8),
        ("prospective-evaluation-base", config.prospective_base_parents, 32),
        ("prospective-evaluation-expansion", config.prospective_max_parents - config.prospective_base_parents, 32),
        ("prospective-evaluation-reserve", config.prospective_reserve, 8),
    )
    if set(scientific_preparations) != {row[0] for row in specifications}:
        raise ValueError("scientific preparation census omits or adds a roster")
    rosters = []
    all_units: list[PreparationUnit] = []
    for roster_id, size, minimum_per_stratum in specifications:
        units = compile_roster(
            roster_id=roster_id,
            size=size,
            scientific_preparations=scientific_preparations[roster_id],
            minimum_per_stratum=minimum_per_stratum,
        )
        all_units.extend(units)
        rosters.append(
            {
                "roster_id": roster_id,
                "size": size,
                "minimum_per_stratum": minimum_per_stratum,
                "unit_ids": [unit.unit_id for unit in units],
                "k_counts": {str(k): sum(unit.k_left == k for unit in units) for k in range(7)},
                "units": [
                    {
                        "unit_id": unit.unit_id,
                        "unit_index": unit.unit_index,
                        "k_left": unit.k_left,
                        "initial_state": unit.initial_state,
                        "boundary_occupation": unit.boundary_occupation,
                        "translation_orbit": unit.translation_orbit,
                        "preparation_seed": str(unit.preparation_seed),
                    }
                    for unit in units
                ],
            }
        )
    unit_ids = [unit.unit_id for unit in all_units]
    if len(unit_ids) != len(set(unit_ids)):
        raise AssertionError("quantum response control rosters collide")
    return {
        "schema": 'empirical-lawhood/simulators/quantum-response-control/prepared-parent-roster-manifest',
        "plan_id": PLAN_ID,
        "campaign_id": CAMPAIGN_ID,
        "rosters": rosters,
        "unit_count": len(unit_ids),
    }


__all__ = [
    "ACQUISITION_CAPABILITY_KEY",
    "ATLAS_CAPABILITY_KEY",
    "Action",
    "Branch",
    "CAMPAIGN_ID",
    "CANARY_ROOT",
    "CAPABILITY_VERSION",
    "CHILD_PLAN_ID",
    "CONTROLLER_CAPABILITY_KEY",
    "Disposition",
    "EVALUATOR_CAPABILITY_KEY",
    "EXTERNAL_ROOT",
    "PLAN_ID",
    "PreparationUnit",
    'QuantumResponseControlConfig',
    "REDUCER_CAPABILITY_KEY",
    "RESPONSE_CAPABILITY_KEY",
    "SOURCE_CAPABILITY_KEY",
    "Stage",
    "Verdict",
    "compile_roster",
    "decode_config",
    "load_config",
    "preparation_weight",
    "roster_manifest",
    "sha256_hex",
    "stable_json_bytes",
    "stratum_allocation",
    "target_weights",
    "translation_orbit_id",
]

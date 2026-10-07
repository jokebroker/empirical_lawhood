"""Strict identities, configuration, and roster arithmetic for quantum marked record identification."""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import StrEnum
import json
from math import comb
from pathlib import Path
from typing import Final, Mapping, Sequence


from ..quantum_scientific_preparations import ScientificPreparations, validate_scientific_preparations

PLAN_ID: Final = "quantum-marked-record-identification"
CAMPAIGN_ID: Final = "quantum-marked-record-identification"
EXTERNAL_ROOT: Final = "runs/quantum-marked-record-identification"
CANARY_ROOT: Final = "runs/quantum-marked-record-identification-capacity-canary"
CAPABILITY_VERSION: Final = "1.0.0"
SOURCE_CAPABILITY_KEY: Final = "open-sim.quantum-trajectory-source.quantum-marked-record-identification"
NOMINATION_CAPABILITY_KEY: Final = "method.quantum-receiver-nomination.quantum-marked-record-identification"
DEVELOPMENT_CAPABILITY_KEY: Final = "method.quantum-receiver-development.quantum-marked-record-identification"
EVALUATOR_CAPABILITY_KEY: Final = "evaluator.quantum-receiver-identification.quantum-marked-record-identification"
MAXIMUM_JSON_BYTES: Final = 64 * 1024**2


class Stage(StrEnum):
    PREDECESSOR_COMPATIBILITY = "predecessor-compatibility"
    INSTRUMENT_CONFORMANCE = "instrument-conformance"
    GRAMMAR_NOMINATION = "grammar-nomination"
    COMPILER_DEVELOPMENT = "compiler-development"
    COORDINATE_EVALUATION = "coordinate-evaluation"
    CLOSEOUT = "closeout"


class Disposition(StrEnum):
    COMPLETE = "COMPLETED"
    TECHNICAL_NONCONTACT_REPLACED = "TECHNICAL_NONCONTACT_REPLACED"
    NOT_ATTEMPTED_PREREQUISITE = "NOT_ATTEMPTED_PREREQUISITE"
    NOT_ATTEMPTED_AUTHORITY = "NOT_ATTEMPTED_AUTHORITY"
    STOPPED_SOURCE = "STOPPED_SOURCE"
    STOPPED_CUSTODY = "STOPPED_CUSTODY"
    UNEVALUABLE = "UNEVALUABLE"


class Panel(StrEnum):
    COUNT = "count"
    SPATIAL_RECENCY = "spatial-recency"
    EVENT_PAIR = "event-pair"
    INTENSITY_RESIDUAL = "intensity-residual"

    @property
    def dimension(self) -> int:
        return {
            Panel.COUNT: 12,
            Panel.SPATIAL_RECENCY: 22,
            Panel.EVENT_PAIR: 32,
            Panel.INTENSITY_RESIDUAL: 36,
        }[self]

    @property
    def semantic_flags(self) -> tuple[str, ...]:
        flags = ["USES_SPATIAL_MARKS", "USES_RECENCY"]
        if self in {Panel.EVENT_PAIR, Panel.INTENSITY_RESIDUAL}:
            flags.extend(("USES_ORDER", "USES_BOUNDARY_DIRECTION"))
        if self is Panel.INTENSITY_RESIDUAL:
            flags.append("USES_FITTED_INTENSITY")
        return tuple(flags)


class Verdict(StrEnum):
    COMPATIBLE_RECEIVER_IDENTIFICATION_SOURCE_READY = (
        "COMPATIBLE_RECEIVER_IDENTIFICATION_SOURCE_READY"
    )
    PREDECESSOR_IDENTITY_STOP = "PREDECESSOR_IDENTITY_STOP"
    SOURCE_SEMANTIC_STOP = "SOURCE_SEMANTIC_STOP"
    TARGET_WEIGHT_STOP = "TARGET_WEIGHT_STOP"
    COMPATIBILITY_UNEVALUABLE = "COMPATIBILITY_UNEVALUABLE"
    RECEIVER_IDENTIFICATION_INSTRUMENT_VALID = "RECEIVER_IDENTIFICATION_INSTRUMENT_VALID"
    FEATURE_GRAMMAR_STOP = "FEATURE_GRAMMAR_STOP"
    LEAKAGE_GUARD_STOP = "LEAKAGE_GUARD_STOP"
    SELECTION_INFERENCE_STOP = "SELECTION_INFERENCE_STOP"
    CONTROL_INSTRUMENT_STOP = "CONTROL_INSTRUMENT_STOP"
    CUSTODY_INSTRUMENT_STOP = "CUSTODY_INSTRUMENT_STOP"
    INSTRUMENT_UNEVALUABLE = "INSTRUMENT_UNEVALUABLE"
    GRAMMAR_NOMINATION_CANDIDATE_GRAMMAR_NOMINATED = "GRAMMAR_NOMINATION_CANDIDATE_GRAMMAR_NOMINATED"
    GRAMMAR_NOMINATION_COUNT_ONLY_GRAMMAR_NOMINATED = "GRAMMAR_NOMINATION_COUNT_ONLY_GRAMMAR_NOMINATED"
    GRAMMAR_NOMINATION_FEATURE_GRAMMAR_REDESIGN_REQUIRED = "GRAMMAR_NOMINATION_FEATURE_GRAMMAR_REDESIGN_REQUIRED"
    GRAMMAR_NOMINATION_UNEVALUABLE = "GRAMMAR_NOMINATION_UNEVALUABLE"
    COMPILER_DEVELOPMENT_COORDINATE_COMPILER_SELECTED = "COMPILER_DEVELOPMENT_COORDINATE_COMPILER_SELECTED"
    COMPILER_DEVELOPMENT_COUNT_ONLY_COMPILER_SELECTED = "COMPILER_DEVELOPMENT_COUNT_ONLY_COMPILER_SELECTED"
    COMPILER_DEVELOPMENT_NO_VIABLE_COORDINATE = "COMPILER_DEVELOPMENT_NO_VIABLE_COORDINATE"
    COMPILER_DEVELOPMENT_SOURCE_OR_RECORD_STOP = "COMPILER_DEVELOPMENT_SOURCE_OR_RECORD_STOP"
    COMPILER_DEVELOPMENT_SELECTION_UNEVALUABLE = "COMPILER_DEVELOPMENT_SELECTION_UNEVALUABLE"
    COORDINATE_EVALUATION_SOURCE_OR_OBSERVATION_STOP = "COORDINATE_EVALUATION_SOURCE_OR_OBSERVATION_STOP"
    COORDINATE_EVALUATION_RECEIVER_ORDER_NOT_REPLICATED = "COORDINATE_EVALUATION_RECEIVER_ORDER_NOT_REPLICATED"
    COORDINATE_EVALUATION_RECEIVER_CONTROL_RELATION_OPPOSED = "COORDINATE_EVALUATION_RECEIVER_CONTROL_RELATION_OPPOSED"
    COORDINATE_EVALUATION_RECEIVER_ORDER_WITHOUT_PREDICTIVE_COORDINATE = (
        "COORDINATE_EVALUATION_RECEIVER_ORDER_WITHOUT_PREDICTIVE_COORDINATE"
    )
    COORDINATE_EVALUATION_COORDINATE_SPECIFICITY_STOP = "COORDINATE_EVALUATION_COORDINATE_SPECIFICITY_STOP"
    COORDINATE_EVALUATION_COUNT_ONLY_COORDINATE_SUPPORTED = "COORDINATE_EVALUATION_COUNT_ONLY_COORDINATE_SUPPORTED"
    COORDINATE_EVALUATION_PHASE_LOCAL_RECORD_COORDINATE_SUPPORTED = "COORDINATE_EVALUATION_PHASE_LOCAL_RECORD_COORDINATE_SUPPORTED"
    COORDINATE_EVALUATION_TIME_TRANSLATION_RECURRENT_COORDINATE_SUPPORTED = (
        "COORDINATE_EVALUATION_TIME_TRANSLATION_RECURRENT_COORDINATE_SUPPORTED"
    )
    COORDINATE_EVALUATION_RECEIVER_COORDINATE_UNEVALUABLE = "COORDINATE_EVALUATION_RECEIVER_COORDINATE_UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class QuantumMarkedRecordIdentificationConfig:
    schema: str
    version: str
    plan_id: str
    campaign_id: str
    source_capability_key: str
    nomination_capability_key: str
    development_capability_key: str
    evaluator_capability_key: str
    l_sites: int
    particles: int
    j_xy: Decimal
    j_z: Decimal
    gamma: Decimal
    cutoff: Decimal
    retained_start: Decimal
    future_horizon: Decimal
    bootstrap_replicates: int
    family_alpha: Decimal
    folds: int
    compiler_development_fit_parents: int
    compiler_development_selection_parents: int
    compiler_development_reserve: int
    coordinate_evaluation_base_parents: int
    coordinate_evaluation_expansion_block: int
    coordinate_evaluation_max_parents: int
    coordinate_evaluation_reserve: int
    minimum_per_stratum: int
    chi_gate: Decimal
    global_equivalence: Decimal
    cancellation_equivalence: Decimal
    future_thinning_advantage: Decimal
    association_gate: Decimal
    burst_advantage: Decimal
    count_increment: Decimal
    semantic_advantage: Decimal
    recurrence_equivalence: Decimal
    development_rho_floor: Decimal
    valid_weight_floor: Decimal
    regularizations: tuple[Decimal, ...]
    lag_edges: tuple[Decimal, ...]
    maximum_dimensions: int
    minimum_free_bytes: int
    batch_size: int

    def __post_init__(self) -> None:
        if (
            self.schema != 'empirical-lawhood/simulators/quantum-marked-record-identification/quantum-trajectory-receiver-law'
            or self.version != CAPABILITY_VERSION
            or self.plan_id != PLAN_ID
            or self.campaign_id != CAMPAIGN_ID
        ):
            raise ValueError("quantum marked record identification config identity differs")
        if (
            self.source_capability_key != SOURCE_CAPABILITY_KEY
            or self.nomination_capability_key != NOMINATION_CAPABILITY_KEY
            or self.development_capability_key != DEVELOPMENT_CAPABILITY_KEY
            or self.evaluator_capability_key != EVALUATOR_CAPABILITY_KEY
        ):
            raise ValueError("quantum marked record identification config selects an unknown capability")
        if (
            (self.l_sites, self.particles) != (12, 6)
            or self.j_xy != Decimal("1")
            or self.j_z != Decimal("1")
            or self.gamma != Decimal("1")
            or self.cutoff != Decimal("200")
            or self.retained_start != Decimal("176")
            or self.future_horizon != Decimal("4")
        ):
            raise ValueError("quantum marked record identification denominator or causal clocks differ")
        if self.compiler_development_fit_parents != self.compiler_development_selection_parents:
            raise ValueError("compiler-development fit/selection must be balanced")
        if not 0 < self.coordinate_evaluation_base_parents <= self.coordinate_evaluation_max_parents:
            raise ValueError("coordinate-evaluation base/max ordering differs")
        if (self.coordinate_evaluation_max_parents - self.coordinate_evaluation_base_parents) % self.coordinate_evaluation_expansion_block:
            raise ValueError("coordinate-evaluation expansion does not tile the maximum")
        if self.folds < 2 or self.maximum_dimensions != 36:
            raise ValueError("fold or dimension contract differs")
        if self.minimum_per_stratum < 32:
            raise ValueError("preparation-stratum floor is too small")
        if self.batch_size <= 0 or self.minimum_free_bytes < 100 * 1024**3:
            raise ValueError("storage contract differs")
        if self.regularizations != (
            Decimal("0"),
            Decimal("0.01"),
            Decimal("0.1"),
            Decimal("1"),
            Decimal("10"),
        ):
            raise ValueError("regularization roster differs")
        expected_lags = (
            Decimal("0"),
            Decimal(1) / Decimal(6),
            Decimal(2) / Decimal(6),
            Decimal(4) / Decimal(6),
            Decimal(8) / Decimal(6),
            Decimal(16) / Decimal(6),
            Decimal("4"),
        )
        if len(self.lag_edges) != len(expected_lags) or any(
            abs(left - right) > Decimal("1e-25")
            for left, right in zip(self.lag_edges, expected_lags, strict=True)
        ):
            raise ValueError("atomic lag grid differs")
        for name in (
            "family_alpha",
            "valid_weight_floor",
            "recurrence_equivalence",
        ):
            value = getattr(self, name)
            if not Decimal(0) < value < Decimal(1):
                raise ValueError(f"{name} must lie inside (0,1)")
        for name in (
            "chi_gate",
            "global_equivalence",
            "cancellation_equivalence",
            "future_thinning_advantage",
            "association_gate",
            "burst_advantage",
            "count_increment",
            "semantic_advantage",
            "development_rho_floor",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def compiler_development_parents(self) -> int:
        return self.compiler_development_fit_parents + self.compiler_development_selection_parents


_STRING_FIELDS: Final = frozenset(
    {
        "schema",
        "version",
        "plan_id",
        "campaign_id",
        "source_capability_key",
        "nomination_capability_key",
        "development_capability_key",
        "evaluator_capability_key",
    }
)
_TUPLE_DECIMAL_FIELDS: Final = frozenset({"regularizations", "lag_edges"})
_INTEGER_FIELDS: Final = frozenset(
    field.name for field in fields(QuantumMarkedRecordIdentificationConfig) if field.type == "int" or field.type is int
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


def decode_config(document: Mapping[str, object]) -> QuantumMarkedRecordIdentificationConfig:
    expected = {field.name for field in fields(QuantumMarkedRecordIdentificationConfig)}
    observed = set(document)
    if observed != expected:
        raise ValueError(
            "quantum marked record identification config keys differ: "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )
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
        elif name in _TUPLE_DECIMAL_FIELDS:
            if (
                not isinstance(raw, list)
                or not raw
                or any(not isinstance(item, str) for item in raw)
            ):
                raise TypeError(f"{name} must be an exact-decimal string list")
            parsed_decimals = tuple(Decimal(item) for item in raw)
            if any(not item.is_finite() or item < 0 for item in parsed_decimals):
                raise ValueError(f"{name} contains an invalid decimal")
            values[name] = parsed_decimals
        else:
            if not isinstance(raw, str):
                raise TypeError(f"{name} must be an exact decimal string")
            parsed_decimal = Decimal(raw)
            if not parsed_decimal.is_finite() or parsed_decimal <= 0:
                raise ValueError(f"{name} must be positive and finite")
            values[name] = parsed_decimal
    return QuantumMarkedRecordIdentificationConfig(**values)  # type: ignore[arg-type]


def load_config(path: Path) -> tuple[QuantumMarkedRecordIdentificationConfig, bytes]:
    payload = path.read_bytes()
    if len(payload) > MAXIMUM_JSON_BYTES:
        raise ValueError("quantum marked record identification config exceeds its byte bound")
    document = json.loads(payload)
    if not isinstance(document, Mapping):
        raise TypeError("quantum marked record identification config root must be an object")
    return decode_config(document), payload


def preparation_weight(l_sites: int, particles: int, k_left: int) -> Decimal:
    half = l_sites // 2
    if not 0 <= k_left <= particles:
        raise ValueError("preparation stratum is invalid")
    return Decimal(comb(half, k_left) * comb(half, particles - k_left)) / Decimal(
        comb(l_sites, particles)
    )




def translation_orbit_id(state: int, l_sites: int) -> int:
    mask = (1 << l_sites) - 1
    return min(
        ((state << offset) | (state >> (l_sites - offset))) & mask for offset in range(l_sites)
    )


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


def _allocation(
    total: int,
    *,
    l_sites: int,
    particles: int,
    minimum: int,
) -> dict[int, int]:
    if total < (particles + 1) * minimum:
        raise ValueError("roster cannot cover preparation strata")
    allocation = {k: minimum for k in range(particles + 1)}
    remaining = total - sum(allocation.values())
    ideals = {k: Decimal(remaining) * preparation_weight(l_sites, particles, k) for k in allocation}
    for k, ideal in ideals.items():
        allocation[k] += int(ideal)
    unassigned = total - sum(allocation.values())
    order = sorted(allocation, key=lambda k: (-(ideals[k] - int(ideals[k])), k))
    for k in order[:unassigned]:
        allocation[k] += 1
    return allocation


def compile_roster(
    *,
    roster_id: str,
    size: int,
    scientific_preparations: ScientificPreparations,
    minimum_per_stratum: int,
    l_sites: int = 12,
    particles: int = 6,
) -> tuple[PreparationUnit, ...]:
    """Bind numerical state/seed rows without allocating from caller labels."""
    allocation = _allocation(size, l_sites=l_sites, particles=particles, minimum=minimum_per_stratum)
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


def target_weights(
    k_values: Sequence[int],
    *,
    l_sites: int = 12,
    particles: int = 6,
) -> tuple[float, ...]:
    counts = {k: k_values.count(k) for k in range(particles + 1)}
    if any(count == 0 for count in counts.values()):
        raise ValueError("target weights lack a preparation stratum")
    weights = tuple(float(preparation_weight(l_sites, particles, k)) / counts[k] for k in k_values)
    total = sum(weights)
    return tuple(value / total for value in weights)


__all__ = [
    "CAMPAIGN_ID",
    "CANARY_ROOT",
    "CAPABILITY_VERSION",
    "DEVELOPMENT_CAPABILITY_KEY",
    "Disposition",
    "EVALUATOR_CAPABILITY_KEY",
    "EXTERNAL_ROOT",
    "MAXIMUM_JSON_BYTES",
    "NOMINATION_CAPABILITY_KEY",
    "PLAN_ID",
    "Panel",
    "PreparationUnit",
    'QuantumMarkedRecordIdentificationConfig',
    "SOURCE_CAPABILITY_KEY",
    "Stage",
    "Verdict",
    "compile_roster",
    "decode_config",
    "load_config",
    "preparation_weight",
    "stable_json_bytes",
    "target_weights",
]

"""Strict identities, configuration, and roster arithmetic for quantum finite grammar identification."""

from __future__ import annotations

from dataclasses import dataclass, fields
from decimal import Decimal
from enum import StrEnum
import json
from math import comb
from pathlib import Path
from typing import Final, Mapping, Sequence


from ..quantum_scientific_preparations import ScientificPreparations, validate_scientific_preparations

PLAN_ID: Final = "quantum-finite-grammar-identification"
CAMPAIGN_ID: Final = "quantum-finite-grammar-identification"
EXTERNAL_ROOT: Final = "runs/quantum-finite-grammar-identification"
CAPABILITY_VERSION: Final = "3.1.0"
SOURCE_CAPABILITY_KEY: Final = "open-sim.quantum-trajectory-source.quantum-finite-grammar-identification"
NOMINATION_CAPABILITY_KEY: Final = "method.quantum-receiver-nomination.quantum-finite-grammar-identification"
DEVELOPMENT_CAPABILITY_KEY: Final = "method.quantum-receiver-development.quantum-finite-grammar-identification"
EVALUATOR_CAPABILITY_KEY: Final = "evaluator.quantum-receiver-identification.quantum-finite-grammar-identification"
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
    NOT_ATTEMPTED_PREREQUISITE = "NOT_ATTEMPTED_PREREQUISITE"
    NOT_ATTEMPTED_AUTHORITY = "NOT_ATTEMPTED_AUTHORITY"
    STOPPED_SOURCE = "STOPPED_SOURCE"
    STOPPED_CUSTODY = "STOPPED_CUSTODY"
    UNEVALUABLE = "UNEVALUABLE"


class Panel(StrEnum):
    COUNT = "count"
    SPATIAL_RECENCY = "spatial-recency"
    EVENT_PAIR = "event-pair"

    @property
    def dimension(self) -> int:
        return {
            Panel.COUNT: 12,
            Panel.SPATIAL_RECENCY: 22,
            Panel.EVENT_PAIR: 36,
        }[self]

    @property
    def semantic_flags(self) -> tuple[str, ...]:
        flags = ["USES_SPATIAL_MARKS", "USES_RECENCY"]
        if self is Panel.EVENT_PAIR:
            flags.extend(("USES_ORDER", "USES_BOUNDARY_DIRECTION"))
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
class QuantumFiniteGrammarIdentificationConfig:
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
    coordinate_evaluation_base_parents: int
    coordinate_evaluation_expansion_block: int
    coordinate_evaluation_max_parents: int
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
    condition_ceiling: Decimal
    coefficient_norm_ceiling: Decimal
    score_variance_floor: Decimal
    latency_deadline_seconds: Decimal
    numeric_cap: Decimal
    pair_lag_split: Decimal
    opportunity_width_ceiling: Decimal
    recognition_width_ceiling: Decimal
    regularizations: tuple[Decimal, ...]
    maximum_dimensions: int
    minimum_free_bytes: int
    batch_size: int

    def __post_init__(self) -> None:
        if (
            self.schema != 'empirical-lawhood/simulators/quantum-finite-grammar-identification/quantum-trajectory-receiver-law'
            or self.version != CAPABILITY_VERSION
            or self.plan_id != PLAN_ID
            or self.campaign_id != CAMPAIGN_ID
        ):
            raise ValueError("quantum finite grammar identification config identity differs")
        if (
            self.source_capability_key != SOURCE_CAPABILITY_KEY
            or self.nomination_capability_key != NOMINATION_CAPABILITY_KEY
            or self.development_capability_key != DEVELOPMENT_CAPABILITY_KEY
            or self.evaluator_capability_key != EVALUATOR_CAPABILITY_KEY
        ):
            raise ValueError("quantum finite grammar identification config selects an unknown capability")
        if (
            (self.l_sites, self.particles) != (12, 6)
            or self.j_xy != Decimal("1")
            or self.j_z != Decimal("1")
            or self.gamma != Decimal("1")
            or self.cutoff != Decimal("200")
            or self.retained_start != Decimal("176")
            or self.future_horizon != Decimal("4")
        ):
            raise ValueError("quantum finite grammar identification denominator or causal clocks differ")
        if (
            self.bootstrap_replicates != 4096
            or self.folds != 8
            or self.compiler_development_fit_parents != 1024
            or self.compiler_development_selection_parents != 1024
            or self.coordinate_evaluation_base_parents != 2048
            or self.coordinate_evaluation_expansion_block != 512
            or self.coordinate_evaluation_max_parents != 4096
            or self.minimum_per_stratum != 32
            or self.maximum_dimensions != 36
            or self.batch_size != 32
            or self.minimum_free_bytes < 100 * 1024**3
        ):
            raise ValueError("quantum finite grammar identification workload contract differs")
        if (
            self.coordinate_evaluation_base_parents % self.coordinate_evaluation_expansion_block
            or self.coordinate_evaluation_max_parents % self.coordinate_evaluation_expansion_block
        ):
            raise ValueError("coordinate-evaluation block roster does not tile base and maximum")
        if self.regularizations != (
            Decimal("0"),
            Decimal("0.01"),
            Decimal("0.1"),
            Decimal("1"),
            Decimal("10"),
        ):
            raise ValueError("regularization roster differs")
        if (
            self.condition_ceiling != Decimal("1e8")
            or self.coefficient_norm_ceiling != Decimal("25")
            or self.score_variance_floor != Decimal("1e-6")
            or self.latency_deadline_seconds != Decimal("0.01")
            or self.numeric_cap != Decimal("50")
            or abs(self.pair_lag_split - Decimal(2) / Decimal(3)) > Decimal("1e-25")
            or self.opportunity_width_ceiling != Decimal("0.50")
            or self.recognition_width_ceiling != Decimal("0.20")
            or self.family_alpha != Decimal("0.05")
            or self.valid_weight_floor != Decimal("0.98")
            or self.chi_gate != Decimal("0.10")
            or self.global_equivalence != Decimal("0.20")
            or self.cancellation_equivalence != Decimal("0.20")
            or self.future_thinning_advantage != Decimal("0.05")
            or self.association_gate != Decimal("0.15")
            or self.burst_advantage != Decimal("0.05")
            or self.count_increment != Decimal("0.03")
            or self.semantic_advantage != Decimal("0.05")
            or self.recurrence_equivalence != Decimal("0.05")
            or self.development_rho_floor != Decimal("0.10")
        ):
            raise ValueError("scientific gate, compiler, or precision contract differs")
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
_TUPLE_DECIMAL_FIELDS: Final = frozenset({"regularizations"})
_INTEGER_FIELDS: Final = frozenset(
    field.name for field in fields(QuantumFiniteGrammarIdentificationConfig) if field.type == "int" or field.type is int
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


def decode_config(document: Mapping[str, object]) -> QuantumFiniteGrammarIdentificationConfig:
    expected = {field.name for field in fields(QuantumFiniteGrammarIdentificationConfig)}
    observed = set(document)
    if observed != expected:
        raise ValueError(
            "quantum finite grammar identification config keys differ: "
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
    return QuantumFiniteGrammarIdentificationConfig(**values)  # type: ignore[arg-type]


def load_config(path: Path) -> tuple[QuantumFiniteGrammarIdentificationConfig, bytes]:
    payload = path.read_bytes()
    if len(payload) > MAXIMUM_JSON_BYTES:
        raise ValueError("quantum finite grammar identification config exceeds its byte bound")
    document = json.loads(payload)
    if not isinstance(document, Mapping):
        raise TypeError("quantum finite grammar identification config root must be an object")
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
    'QuantumFiniteGrammarIdentificationConfig',
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

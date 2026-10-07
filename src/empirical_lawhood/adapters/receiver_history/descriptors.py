"""Outcome-blind latent-field and finite-view descriptor generation for receiver-history."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from math import pi

import numpy as np
import numpy.typing as npt

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import canonical_json_bytes

from .contracts import (
    RECEIVER_HISTORY_REDUCED_ACTION_MENU_ACTION_AMPLITUDES,
    RECEIVER_HISTORY_REDUCED_ACTION_MENU_ACTION_DURATIONS,
    RECEIVER_HISTORY_REDUCED_ACTION_MENU_EVALUATION_UNITS_PER_FAMILY,
    RECEIVER_HISTORY_REDUCED_ACTION_MENU_INFERENCE_ID,
    RECEIVER_HISTORY_REDUCED_ACTION_MENU_PLAN_ID,
    ReceiverHistoryConfig,
    ReceiverHistoryDenominatorDescriptor,
    ReceiverHistoryDisorderFamily,
    ReceiverHistoryPhase,
)


FloatArray = npt.NDArray[np.float64]

_R_BAR = Decimal("100")
_C_BAR = Decimal("0.001")
_V_REF = Decimal("1")
_VARIATION = Decimal("0.12")


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("receiver-history descriptor value must be finite")
    return Decimal(str(float(value)))


def evaluation_unit_ids(*, units_per_family: int = 30) -> tuple[str, ...]:
    if units_per_family not in {12, 30}:
        raise ValueError("receiver-history evaluation units-per-family profile differs")
    return tuple(
        sorted(
            f"receiver-history.eval.{family.value}.block-{index:02d}"
            for family in ReceiverHistoryDisorderFamily
            for index in range(units_per_family)
        )
    )


def development_unit_ids() -> tuple[str, ...]:
    return tuple(
        sorted(
            f"receiver-history.dev.{family.value}.block-{index:02d}"
            for family in ReceiverHistoryDisorderFamily
            for index in range(4)
        )
    )


def canary_unit_ids() -> tuple[str, ...]:
    return ("receiver-history.canary.truth-known.n16", "receiver-history.canary.truth-known.n32")


def deterministic_development_seed(unit_id: str) -> bytes:
    return sha256(f"receiver-history-development-seed:{unit_id}".encode("ascii")).digest()


def reserved_non_evaluation_seed_digests() -> frozenset[str]:
    'Return every deterministic canary/development seed digest excluded from nomination.'

    seeds = {
        *(deterministic_development_seed(unit_id) for unit_id in development_unit_ids()),
        *(
            sha256(f"receiver-history-uniform-canary-n{scale}".encode("ascii")).digest()
            for scale in (16, 32)
        ),
        *(
            sha256(f"receiver-history-conformance:{scale}".encode("ascii")).digest()
            for scale in (16, 32)
        ),
        *(
            sha256(f"receiver-history-resource-canary-{index:02d}".encode("ascii")).digest()
            for index in range(6)
        ),
    }
    return frozenset(sha256(seed).hexdigest() for seed in seeds)


def default_config(
    phase: ReceiverHistoryPhase,
    *,
    seed_roster_commitment_sha256: str | None = None,
) -> ReceiverHistoryConfig:
    if phase in {
        ReceiverHistoryPhase.NOMINATION,
        ReceiverHistoryPhase.TARGETED_EVALUATION,
        ReceiverHistoryPhase.UNTOUCHED_EVALUATION,
    }:
        unit_ids = evaluation_unit_ids()
    elif phase is ReceiverHistoryPhase.DEVELOPMENT:
        unit_ids = development_unit_ids()
    else:
        unit_ids = canary_unit_ids()
    return ReceiverHistoryConfig(
        config_id=f"receiver-history.full-action-menu.{phase.value.lower()}",
        phase=phase,
        plan_id='receiver-history-full-action-menu-metatheory',
        parent_result_id='history-budget-phase-diagram-numerical-qualification-resource-stop',
        scale_cells={
            ReceiverHistoryPhase.NOMINATION: (64, 128, 256),
            ReceiverHistoryPhase.CANARY: (16, 32),
            ReceiverHistoryPhase.DEVELOPMENT: (64, 128, 256),
            ReceiverHistoryPhase.TARGETED_EVALUATION: (64, 128, 256),
            ReceiverHistoryPhase.UNTOUCHED_EVALUATION: (64, 128, 256),
        }[phase],
        excluded_resource_canary_scale_cells=(
            (64, 128, 256) if phase is ReceiverHistoryPhase.CANARY else ()
        ),
        disorder_families=tuple(ReceiverHistoryDisorderFamily),
        unit_ids=unit_ids,
        seed_roster_commitment_sha256=seed_roster_commitment_sha256,
        receiver_bin_count=8,
        history_max_depth=31,
        history_lag_t_star=Decimal("0.00001"),
        rank_relative_threshold=Decimal("1e-12"),
        coordinate_collision_epsilon=Decimal("0.005"),
        future_divergence_epsilon=Decimal("0.005"),
        generator_observer_metric_tolerance=Decimal("1e-10"),
        # HiGHS float64 dual stationarity is qualified to 1e-9 in numerical qualification.  This is
        # still six orders below the smallest scientific decision margin and
        # avoids mislabelling solver roundoff as an independent-cert failure.
        optimization_tolerance=Decimal("1e-9"),
        midpoint_boundary_tie_epsilon=Decimal("1e-10"),
        hidden_amplitude_max_volts=Decimal("0.15"),
        hidden_voltage_envelope=(Decimal("0"), Decimal("1")),
        hidden_voltage_interior_margin=Decimal("1e-8"),
        target_pairs_per_depth=2,
        sink_pairs_per_depth=2,
        random_pairs_per_depth=4,
        action_amplitudes_u_star=tuple(
            Decimal(value) for value in ("0.20", "0.40", "0.60", "0.80", "1.00")
        ),
        action_durations_t_star=tuple(
            Decimal(value) for value in ("0.01", "0.025", "0.05", "0.10", "0.20")
        ),
        panel_endpoint_t_star=Decimal("0.20"),
        diagnostic_times_t_star=(Decimal("0.005"), Decimal("0.01"), Decimal("0.02")),
        target_charge_minimum_q_star=Decimal("0.08"),
        sink_voltage_maximum_v_star=Decimal("0.80"),
        absolute_depths=(0, 4, 8),
        normalized_history_budgets=(
            Decimal("0.5"),
            Decimal("1"),
        ),
        primary_absolute_depth_contrast=(4, 8),
        primary_normalized_budget_contrast=(Decimal("0.5"), Decimal("1")),
        receiver_resolution_panel=(Decimal("0.005"),),
        resolution_panel_depths=(),
        target_decision_margin=Decimal("0.005"),
        sink_decision_margin=Decimal("0.005"),
        untouched_preparation_count=0,
        untouched_prefix_counts=(),
        untouched_distribution_id='not-requested',
        untouched_scale_map_id='not-requested',
        untouched_earliest_lag=0,
        untouched_low_rate_threshold=Decimal("0.20"),
        bootstrap_resamples=10_000,
        bootstrap_seed=7_031_419,
        untouched_bootstrap_seed=7_031_420,
        alignment_bootstrap_seed=7_031_421,
        simultaneous_alpha=Decimal("0.05"),
        majority_opposition_threshold=Decimal("0.5"),
        descriptor_algorithm_id='receiver-history-latent-field-centre-edge',
        history_rank_algorithm_id='receiver-history-fixed-k-b-history',
        targeting_tie_break_rule_id='receiver-history-stable-lexical-boundary',
        random_comparator_algorithm_id='receiver-history-outcome-blind-fibre-pcg64',
        inference_algorithm_id='receiver-history-full-action-menu-complete-unit-phase-diagram',
        structural_rank_algorithm_id='receiver-history-observability-jet-matching',
        discrete_rank_algorithm_id='arb-inertia-exact-dyadic-v2',
        conditioning_algorithm_id='receiver-history-absolute-resolution-svd',
        preparation_sampler_algorithm_id='not-requested',
        alignment_algorithm_id='receiver-history-bracket-alignment',
        false_promotion_rule_id='receiver-history-unsafe-first-lexicographic',
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access={
            ReceiverHistoryPhase.NOMINATION: OutcomeAccess.OUTCOME_BLIND,
            ReceiverHistoryPhase.CANARY: OutcomeAccess.DEVELOPMENT_VISIBLE,
            ReceiverHistoryPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
            ReceiverHistoryPhase.TARGETED_EVALUATION: OutcomeAccess.EVALUATION_SEALED,
            ReceiverHistoryPhase.UNTOUCHED_EVALUATION: OutcomeAccess.EVALUATION_SEALED,
        }[phase],
        visibility_ceiling={
            ReceiverHistoryPhase.NOMINATION: VisibilityCeiling.PROSPECTIVE,
            ReceiverHistoryPhase.CANARY: VisibilityCeiling.DEVELOPMENT_ONLY,
            ReceiverHistoryPhase.DEVELOPMENT: VisibilityCeiling.DEVELOPMENT_ONLY,
            ReceiverHistoryPhase.TARGETED_EVALUATION: VisibilityCeiling.PROSPECTIVE,
            ReceiverHistoryPhase.UNTOUCHED_EVALUATION: VisibilityCeiling.PROSPECTIVE,
        }[phase],
        maximum_parallel_tasks=4,
        physical_execution_authorized=False,
        requests_controller=False,
        nonactuating=True,
    )


def breadth_preserving_config(
    phase: ReceiverHistoryPhase,
    *,
    seed_roster_commitment_sha256: str | None = None,
) -> ReceiverHistoryConfig:
    'Return the fresh 12-hour, breadth-preserving reduced-action-menu profile.'

    if phase is ReceiverHistoryPhase.UNTOUCHED_EVALUATION:
        raise ValueError('receiver-history reduced-action-menu excludes conditional untouched evaluation')

    config = default_config(
        phase,
        seed_roster_commitment_sha256=seed_roster_commitment_sha256,
    )
    evaluation_ids = evaluation_unit_ids(
        units_per_family=RECEIVER_HISTORY_REDUCED_ACTION_MENU_EVALUATION_UNITS_PER_FAMILY
    )
    return replace(
        config,
        config_id=f"receiver-history.reduced-action-menu.{phase.value.lower()}",
        plan_id=RECEIVER_HISTORY_REDUCED_ACTION_MENU_PLAN_ID,
        unit_ids=(
            evaluation_ids
            if phase
            in {
                ReceiverHistoryPhase.NOMINATION,
                ReceiverHistoryPhase.TARGETED_EVALUATION,
                ReceiverHistoryPhase.UNTOUCHED_EVALUATION,
            }
            else config.unit_ids
        ),
        action_amplitudes_u_star=RECEIVER_HISTORY_REDUCED_ACTION_MENU_ACTION_AMPLITUDES,
        action_durations_t_star=RECEIVER_HISTORY_REDUCED_ACTION_MENU_ACTION_DURATIONS,
        inference_algorithm_id=RECEIVER_HISTORY_REDUCED_ACTION_MENU_INFERENCE_ID,
        maximum_parallel_tasks=6,
    )


def decode_config(payload: bytes) -> ReceiverHistoryConfig:
    return decode_canonical_bytes(payload, ReceiverHistoryConfig, maximum_bytes=512 * 1024)


def family_from_unit_id(unit_id: str) -> ReceiverHistoryDisorderFamily:
    matches = tuple(family for family in ReceiverHistoryDisorderFamily if family.value in unit_id)
    if len(matches) != 1:
        raise ValueError("receiver-history unit ID does not bind exactly one disorder family")
    return matches[0]


def _rng(seed: bytes, *, family: ReceiverHistoryDisorderFamily, role: str) -> np.random.Generator:
    digest = sha256(
        seed + b"\0" + family.value.encode("ascii") + b"\0" + role.encode("ascii")
    ).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))


def _periodic_field(points: FloatArray, rng: np.random.Generator) -> FloatArray:
    phase = float(rng.uniform(0.0, 2.0 * pi))
    phase_two = float(rng.uniform(0.0, 2.0 * pi))
    weight = float(rng.uniform(0.15, 0.35))
    raw = np.sin(2.0 * pi * points + phase) + weight * np.cos(4.0 * pi * points + phase_two)
    return np.asarray(raw / (1.0 + weight), dtype=np.float64)


def _periodic_distance(left: FloatArray, right: FloatArray) -> FloatArray:
    distance = np.abs(left[:, None] - right[None, :])
    return np.minimum(distance, 1.0 - distance)


def _correlated_field(points: FloatArray, rng: np.random.Generator) -> FloatArray:
    centres = (np.arange(16, dtype=np.float64) + 0.5) / 16.0
    coefficients = np.asarray(rng.normal(size=16), dtype=np.float64)
    analytic_norm = float(np.sum(np.abs(coefficients)))
    if analytic_norm == 0.0:
        raise ValueError("receiver-history correlated field has zero analytic norm")
    distance = _periodic_distance(points, centres)
    basis = np.exp(-0.5 * (distance / (1.0 / 16.0)) ** 2)
    normalized = (basis @ coefficients) / analytic_norm
    bounded = np.tanh(normalized) / np.tanh(1.0)
    if np.max(np.abs(bounded)) > 1.0 + 1e-14:
        raise ValueError("receiver-history correlated field exceeded its analytic bound")
    return np.asarray(bounded, dtype=np.float64)


def _haar_sign(points: FloatArray, level: int, location: int) -> FloatArray:
    width = 1.0 / (2**level)
    local = (points - location * width) / width
    active = (local >= 0.0) & (local < 1.0)
    values = np.zeros_like(points)
    values[active & (local < 0.5)] = 1.0
    values[active & (local >= 0.5)] = -1.0
    return values


def _wavelet_field(points: FloatArray, rng: np.random.Generator) -> FloatArray:
    raw = np.zeros_like(points)
    bound = 0.0
    for level in range(7):
        weight = 2.0 ** (-(level + 1) / 2.0)
        for location in range(2**level):
            coefficient = float(rng.uniform(-1.0, 1.0))
            raw += weight * coefficient * _haar_sign(points, level, location)
        bound += weight
    if bound <= 0.0:
        raise ValueError("receiver-history wavelet field has no levels")
    values = raw / bound
    if np.max(np.abs(values)) > 1.0 + 1e-14:
        raise ValueError("receiver-history wavelet field exceeded its finite-series bound")
    return np.asarray(values, dtype=np.float64)


def latent_field(
    family: ReceiverHistoryDisorderFamily,
    points: FloatArray,
    *,
    seed: bytes,
    role: str,
) -> FloatArray:
    rng = _rng(seed, family=family, role=role)
    if family is ReceiverHistoryDisorderFamily.SMOOTH_PERIODIC_12:
        values = _periodic_field(points, rng)
    elif family is ReceiverHistoryDisorderFamily.CORRELATED_FIELD_12:
        values = _correlated_field(points, rng)
    else:
        values = _wavelet_field(points, rng)
    if not np.all(np.isfinite(values)) or np.max(np.abs(values)) > 1.0 + 1e-12:
        raise ValueError("receiver-history latent field is invalid")
    return values


def generate_descriptor(
    *,
    unit_id: str,
    family: ReceiverHistoryDisorderFamily,
    scale_cells: int,
    seed: bytes,
) -> ReceiverHistoryDenominatorDescriptor:
    if len(seed) != 32:
        raise ValueError("receiver-history seed must contain exactly 32 bytes")
    if family.value not in unit_id and not unit_id.startswith("canary."):
        raise ValueError("receiver-history unit/family binding differs")
    if scale_cells not in {16, 32, 64, 128, 256}:
        raise ValueError("receiver-history descriptor scale differs")
    cell_points = (np.arange(scale_cells, dtype=np.float64) + 0.5) / scale_cells
    edge_points = (np.arange(scale_cells - 1, dtype=np.float64) + 1.0) / scale_cells
    c_field = latent_field(family, cell_points, seed=seed, role="capacitance")
    r_field = latent_field(family, edge_points, seed=seed, role="resistance")
    capacitances = float(_C_BAR) * (1.0 + float(_VARIATION) * c_field)
    resistances = float(_R_BAR) * (1.0 + float(_VARIATION) * r_field)
    if (
        np.min(capacitances) <= 0.0
        or np.min(resistances) <= 0.0
        or not np.all(np.isfinite(capacitances))
        or not np.all(np.isfinite(resistances))
    ):
        raise ValueError("receiver-history component generation failed positivity/finite checks")
    field_payload = canonical_json_bytes(
        {
            "capacitance_field": tuple(_decimal(value) for value in c_field),
            "resistance_field": tuple(_decimal(value) for value in r_field),
        }
    )
    return ReceiverHistoryDenominatorDescriptor(
        descriptor_id=f"descriptor.{unit_id}.n{scale_cells}",
        unit_id=unit_id,
        family=family,
        scale_cells=scale_cells,
        seed_sha256=sha256(seed).hexdigest(),
        latent_field_sha256=sha256(field_payload).hexdigest(),
        capacitances_farads=tuple(_decimal(value) for value in capacitances),
        interior_resistances_ohms=tuple(_decimal(value) for value in resistances),
        left_source_resistance_ohms=_R_BAR,
        right_termination_resistance_ohms=_R_BAR,
        resistance_bar_ohms=_R_BAR,
        capacitance_bar_farads=_C_BAR,
        voltage_reference_volts=_V_REF,
        time_scale_seconds=_R_BAR * _C_BAR * Decimal(scale_cells**2),
        component_variation_fraction=_VARIATION,
        dtype_id="float64",
    )


__all__ = [
    "canary_unit_ids",
    "decode_config",
    "default_config",
    "deterministic_development_seed",
    "development_unit_ids",
    "evaluation_unit_ids",
    "family_from_unit_id",
    "generate_descriptor",
    "latent_field",
    "reserved_non_evaluation_seed_digests",
    'breadth_preserving_config',
]

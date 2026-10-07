"""Outcome-visible thermodynamic-response post-hoc analysis methods.

The estimators in this module are substrate-neutral and filesystem-free. Thin
bounded orchestration supplies already validated parent projections. Complete
prepared cells, discharges, packs, or reference cases are the only resampling
units; nested clocks, words, coordinates, spectral lines, and transitions are
never promoted to replicates.

All outputs are exploratory and nonpromotable. Relative MAST-U calculations
are conditional on an explicitly stated common-input model, and delay/state
analyses are predictive rather than causal or Koopman-identification claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from itertools import permutations, product
import math
import re
from typing import ClassVar, Final, cast

import numpy as np
import numpy.typing as npt

from empirical_lawhood.adapters.methods.response_algebra_posthoc import (
    NONIDENTITY_WORD_IDS,
    TrajectoryPanel,
    finite_word_estimands,
    identity_relative_responses,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_document_shape,
    validate_sha256,
    validate_stable_id,
)


POSTHOC_ANALYSIS_IDS: Final = (
    'non-entailment-lattice-analysis',
    'state-conditioned-cocycle-analysis',
    'battery-pack-reversal-analysis',
    'mastu-common-input-analysis',
    'receiver-faithfulness-analysis',
    'mastu-partial-identification-analysis',
    'gym-family-equivalence-rank-analysis',
    'pybamm-curvature-memory-localization-analysis',
    'claim-monotonicity-analysis',
    'measurement-value-of-information-analysis',
)
FloatArray = npt.NDArray[np.float64]


class ThermodynamicPosthocDisposition(StrEnum):
    """Closed exploratory dispositions shared by the ten analyses."""

    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    MIXED = "MIXED"
    NULL = "NULL"
    UNEVALUABLE = "UNEVALUABLE"
    METHOD_FAILURE = "METHOD_FAILURE"


@dataclass(frozen=True, slots=True)
class ReceiverViewSpec:
    """One predeclared receiver view and its parent numerical floor."""

    group_id: str
    panel: TrajectoryPanel
    numerical_floor: FloatArray
    role: str

    def __post_init__(self) -> None:
        validate_stable_id(self.group_id, field_name="group_id")
        if self.role not in {"FULL_REFERENCE", "SEMANTIC", "OPERATIONAL", "NEGATIVE_CONTROL"}:
            raise ValueError("receiver view role is not registered")
        expected = (self.panel.times.size, len(self.panel.coordinate_ids))
        if self.numerical_floor.shape != expected:
            raise ValueError("receiver view floor differs from panel support")
        if not np.all(np.isfinite(self.numerical_floor)) or np.any(self.numerical_floor < 0):
            raise ValueError("receiver numerical floor must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class ReceiverFamilySpec:
    """Finite, named receiver family; arbitrary subset search is not an API."""

    system_id: str
    views: tuple[ReceiverViewSpec, ...]
    reference_group_id: str
    selected_time_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.system_id, field_name="system_id")
        validate_stable_id(self.reference_group_id, field_name="reference_group_id")
        if len(self.views) < 2:
            raise ValueError("receiver family requires a reference and at least one projection")
        group_ids = tuple(value.group_id for value in self.views)
        if tuple(sorted(set(group_ids))) != group_ids:
            raise ValueError("receiver family group IDs must be sorted and unique")
        references = [value for value in self.views if value.group_id == self.reference_group_id]
        if len(references) != 1 or references[0].role != "FULL_REFERENCE":
            raise ValueError("receiver family requires one named full reference")
        reference = references[0].panel
        for value in self.views:
            panel = value.panel
            if (
                panel.system_id != self.system_id
                or panel.split_id != reference.split_id
                or panel.numerical_view_id != reference.numerical_view_id
                or panel.unit_ids != reference.unit_ids
                or panel.word_ids != reference.word_ids
                or panel.source_episode_sha256s != reference.source_episode_sha256s
                or not np.array_equal(panel.times, reference.times)
            ):
                raise ValueError("receiver views do not share one complete parent panel")
        if (
            not self.selected_time_indices
            or tuple(sorted(set(self.selected_time_indices))) != self.selected_time_indices
            or self.selected_time_indices[0] < 0
            or self.selected_time_indices[-1] >= reference.times.size
        ):
            raise ValueError("receiver selected times are invalid")


@dataclass(frozen=True, slots=True)
class ThermodynamicPosthocConfig(CanonicalRecord):
    """Exact claim-bearing configuration for one outcome-visible attempt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/thermodynamic-posthoc-config'

    config_id: str
    study_id: str
    analysis_ids: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    independent_unit_roles: tuple[str, ...]
    confidence_level: Decimal
    bootstrap_replicates: int
    seed: int
    parent_inventory_sha256: str
    search_family_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.study_id, field_name='study_id')
        if self.analysis_ids != POSTHOC_ANALYSIS_IDS:
            raise ValueError("post-hoc config must contain the exact ten-analysis family")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("outcome-visible post-hoc analysis must be nonpromotable")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("post-hoc parent outcomes must remain explicitly revealed")
        require_sorted_unique_strings(
            self.independent_unit_roles,
            field_name="independent_unit_roles",
            allow_empty=False,
        )
        validate_decimal(
            self.confidence_level,
            field_name="confidence_level",
            minimum=Decimal("0.5"),
        )
        if self.confidence_level >= 1:
            raise ValueError("confidence level must be below one")
        if self.bootstrap_replicates < 200:
            raise ValueError("complete-unit bootstrap requires at least 200 replicates")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        validate_sha256(self.parent_inventory_sha256, field_name="parent_inventory_sha256")
        validate_sha256(self.search_family_sha256, field_name="search_family_sha256")


@dataclass(frozen=True, slots=True)
class ThermodynamicPosthocInputBinding(CanonicalRecord):
    """Typed summary binding exact parent identities to the frozen config."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/thermodynamic-posthoc-input-binding'

    input_id: str
    config_sha256: str
    parent_logical_ids: tuple[str, ...]
    parent_sha256s: tuple[str, ...]
    evidence_world_ids: tuple[str, ...]
    independent_unit_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")
        require_sorted_unique_strings(
            self.parent_logical_ids,
            field_name="parent_logical_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.parent_sha256s,
            field_name="parent_sha256s",
            allow_empty=False,
        )
        for value in self.parent_sha256s:
            validate_sha256(value, field_name="parent_sha256s")
        require_sorted_unique_strings(
            self.evidence_world_ids,
            field_name="evidence_world_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.independent_unit_roles,
            field_name="independent_unit_roles",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class ThermodynamicPosthocResultIndex(CanonicalRecord):
    """Content-addressed terminal index; scientific payloads remain separate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/thermodynamic-posthoc-result-index'

    result_id: str
    attempt_id: str
    analysis_ids: tuple[str, ...]
    dispositions: tuple[str, ...]
    result_sha256s: tuple[str, ...]
    parent_inventory_sha256: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    all_terminal: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.attempt_id, field_name="attempt_id")
        if self.analysis_ids != POSTHOC_ANALYSIS_IDS:
            raise ValueError("result index must retain the complete ten-analysis family")
        if len(self.dispositions) != len(self.analysis_ids):
            raise ValueError("result dispositions differ from the analysis family")
        allowed = {value.value for value in ThermodynamicPosthocDisposition}
        if any(value not in allowed for value in self.dispositions):
            raise ValueError("result index contains an unknown disposition")
        if len(self.result_sha256s) != len(self.analysis_ids):
            raise ValueError("result hashes differ from the analysis family")
        for value in (*self.result_sha256s, self.parent_inventory_sha256):
            validate_sha256(value, field_name="result_sha256")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("post-hoc result index cannot promote evidence")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("post-hoc result index cannot reseal outcomes")
        if not self.all_terminal:
            raise ValueError("terminal index requires every analysis disposition")


def decode_posthoc_config(document: Mapping[str, object]) -> ThermodynamicPosthocConfig:
    """Decode the canonical config while rejecting unknown or missing fields."""

    names = frozenset(
        {
            "analysis_ids",
            "bootstrap_replicates",
            "confidence_level",
            "config_id",
            "evidence_ceiling",
            "independent_unit_roles",
            "outcome_access",
            "parent_inventory_sha256",
            'study_id',
            "search_family_sha256",
            "seed",
        }
    )
    values = validate_document_shape(
        document,
        expected_schema=ThermodynamicPosthocConfig.SCHEMA,
        expected_version=ThermodynamicPosthocConfig.VERSION,
        field_names=names,
    )
    confidence = cast(Mapping[str, object], values["confidence_level"])
    return ThermodynamicPosthocConfig(
        config_id=cast(str, values["config_id"]),
        study_id=cast(str, values['study_id']),
        analysis_ids=tuple(cast(Sequence[str], values["analysis_ids"])),
        evidence_ceiling=EvidenceCeiling(cast(str, values["evidence_ceiling"])),
        outcome_access=OutcomeAccess(cast(str, values["outcome_access"])),
        independent_unit_roles=tuple(cast(Sequence[str], values["independent_unit_roles"])),
        confidence_level=Decimal(cast(str, confidence["decimal"])),
        bootstrap_replicates=cast(int, values["bootstrap_replicates"]),
        seed=cast(int, values["seed"]),
        parent_inventory_sha256=cast(str, values["parent_inventory_sha256"]),
        search_family_sha256=cast(str, values["search_family_sha256"]),
    )


def _finite_array(values: Sequence[float], *, minimum_units: int = 2) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < minimum_units or not np.all(np.isfinite(array)):
        raise ValueError("analysis requires a finite complete-unit vector")
    return array


def exact_sign_flip_test(values: Sequence[float]) -> dict[str, object]:
    """Two-sided paired randomization test over complete independent units."""

    array = _finite_array(values)
    if array.size > 20:
        raise ValueError("exact sign-flip enumeration is bounded to twenty units")
    observed = abs(float(np.mean(array)))
    count = 0
    extreme = 0
    for signs in product((-1.0, 1.0), repeat=array.size):
        count += 1
        statistic = abs(float(np.mean(array * np.asarray(signs))))
        if statistic >= observed - 1e-15:
            extreme += 1
    return {
        "exact": True,
        "independent_unit_count": int(array.size),
        "observed_absolute_mean": observed,
        "permutation_count": count,
        "two_sided_p_value": extreme / count,
    }


def complete_unit_bootstrap_interval(
    values: Sequence[float],
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    """Percentile interval over one finite vector of independent units."""

    array = _finite_array(values)
    if replicates < 200 or not 0.5 < confidence_level < 1:
        raise ValueError("bootstrap configuration is invalid")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, array.size, size=(replicates, array.size))
    means = np.mean(array[indices], axis=1)
    alpha = (1 - confidence_level) / 2
    return {
        "mean": float(np.mean(array)),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1 - alpha)),
    }


def complete_unit_max_t_band(
    values: FloatArray,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
    chunk_size: int = 32,
) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    """Memory-bounded simultaneous max-t band over a fixed nested family."""

    if values.ndim < 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("max-t band requires finite [unit,...] values")
    if replicates < 200 or not 0.5 < confidence_level < 1 or chunk_size <= 0:
        raise ValueError("max-t configuration is invalid")
    mean = np.mean(values, axis=0)
    standard_error = np.std(values, axis=0, ddof=1) / math.sqrt(values.shape[0])
    safe = np.where(standard_error > 0, standard_error, 1.0)
    maxima = np.empty(replicates, dtype=np.float64)
    generator = np.random.default_rng(seed)
    offset = 0
    while offset < replicates:
        count = min(chunk_size, replicates - offset)
        indices = generator.integers(0, values.shape[0], size=(count, values.shape[0]))
        bootstrap_means = np.mean(values[indices], axis=1)
        standardized = np.abs((bootstrap_means - mean) / safe)
        standardized = np.where(standard_error[None, ...] > 0, standardized, 0.0)
        maxima[offset : offset + count] = np.max(standardized.reshape(count, -1), axis=1)
        offset += count
    critical = float(np.quantile(maxima, confidence_level))
    half_width = critical * standard_error
    return mean, mean - half_width, mean + half_width, critical


def _rankdata(values: FloatArray) -> FloatArray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def spearman_correlation(first: Sequence[float], second: Sequence[float]) -> float | None:
    """Finite rank correlation without treating nested rows as replicates."""

    left = _finite_array(first, minimum_units=3)
    right = _finite_array(second, minimum_units=3)
    if left.shape != right.shape:
        raise ValueError("rank correlation vectors differ")
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def _exact_spearman_permutation_test(
    first: Sequence[float], second: Sequence[float]
) -> dict[str, object]:
    left = _finite_array(first, minimum_units=3)
    right = _finite_array(second, minimum_units=3)
    if left.shape != right.shape:
        raise ValueError("rank permutation vectors differ")
    if left.size > 9:
        return {
            "disposition": ThermodynamicPosthocDisposition.UNEVALUABLE.value,
            "reason": "EXACT_COMPLETE_UNIT_PERMUTATION_LIMIT_EXCEEDED",
            "independent_unit_count": int(left.size),
        }
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if np.std(left_rank) == 0 or np.std(right_rank) == 0:
        return {
            "disposition": ThermodynamicPosthocDisposition.UNEVALUABLE.value,
            "reason": "CONSTANT_COMPLETE_UNIT_RANK_VECTOR",
            "independent_unit_count": int(left.size),
        }
    centered_left = left_rank - np.mean(left_rank)
    centered_right = right_rank - np.mean(right_rank)
    denominator = math.sqrt(
        float(np.dot(centered_left, centered_left)) * float(np.dot(centered_right, centered_right))
    )
    observed = float(np.dot(centered_left, centered_right) / denominator)
    extreme = 0
    count = 0
    for permutation in permutations(right_rank.tolist()):
        permuted = np.asarray(permutation, dtype=np.float64) - np.mean(right_rank)
        statistic = float(np.dot(centered_left, permuted) / denominator)
        extreme += abs(statistic) >= abs(observed) - 1e-15
        count += 1
    return {
        "disposition": ThermodynamicPosthocDisposition.SUPPORTED.value,
        "independent_unit_count": int(left.size),
        "observed_spearman": observed,
        "permutation_count": count,
        "two_sided_p_value": extreme / count,
        "exact": True,
    }


def _trimmed_mean(values: FloatArray, fraction: float = 0.2) -> float:
    count = int(math.floor(values.size * fraction))
    ordered = np.sort(values)
    selected = ordered[count : values.size - count] if count else ordered
    return float(np.mean(selected))


def battery_pack_reversal_analysis(
    pack_rows: Sequence[Mapping[str, object]],
    transition_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Explain pack-majority/macro-loss reversal without row-level replication."""

    if len(pack_rows) < 3:
        raise ValueError("battery analysis requires complete physical packs")
    pack_ids = tuple(sorted(cast(str, row["pack_id"]) for row in pack_rows))
    if len(set(pack_ids)) != len(pack_rows):
        raise ValueError("battery pack rows are not unique")
    evaluable = [row for row in transition_rows if cast(bool, row["primary_evaluable"])]
    if {cast(str, row["pack_id"]) for row in evaluable} != set(pack_ids):
        raise ValueError("primary transitions do not cover the exact pack family")

    prediction_fields = {
        "order_dose": 'order_dose_next_prediction_ah',
        "wrong_history": "wrong_history_delta_prediction_ah",
        "collapsed_denominator": "collapsed_denominator_delta_prediction_ah",
    }
    rows: list[dict[str, object]] = []
    contrasts: dict[str, list[float]] = {key: [] for key in prediction_fields}
    for pack_id in pack_ids:
        source = [row for row in evaluable if row["pack_id"] == pack_id]
        candidate_errors = np.asarray(
            [
                abs(
                    cast(float, row["candidate_next_prediction_ah"])
                    - cast(float, row["next_capacity_ah"])
                )
                for row in source
            ]
        )
        candidate = float(np.mean(candidate_errors))
        metrics: dict[str, float] = {}
        for comparator, field in prediction_fields.items():
            if field.endswith("delta_prediction_ah"):
                predictions = np.asarray(
                    [
                        cast(float, row["current_capacity_ah"]) + cast(float, row[field])
                        for row in source
                    ]
                )
            else:
                predictions = np.asarray([cast(float, row[field]) for row in source])
            outcomes = np.asarray([cast(float, row["next_capacity_ah"]) for row in source])
            comparator_mae = float(np.mean(np.abs(predictions - outcomes)))
            metrics[comparator] = comparator_mae
            contrasts[comparator].append(candidate - comparator_mae)
        parent = next(row for row in pack_rows if row["pack_id"] == pack_id)
        if not math.isclose(candidate, cast(float, parent["candidate_mae_ah"]), abs_tol=2e-15):
            raise ValueError(f"candidate pack MAE failed reproduction for {pack_id}")
        rows.append(
            {
                "pack_id": pack_id,
                "protocol_family": parent["protocol_family"],
                "transition_count": len(source),
                "candidate_mae_ah": candidate,
                "comparator_mae_ah": metrics,
                "delta_vs_comparator_ah": {
                    key: candidate - value for key, value in metrics.items()
                },
                "mean_joint_support_distance": float(
                    np.mean([cast(float, row["joint_support_distance"]) for row in source])
                ),
                "mean_current_capacity_ah": float(
                    np.mean([cast(float, row["current_capacity_ah"]) for row in source])
                ),
            }
        )

    summaries: dict[str, object] = {}
    for index, (comparator, raw) in enumerate(contrasts.items()):
        values = np.asarray(raw, dtype=np.float64)
        positive = values[values > 0]
        summaries[comparator] = {
            "candidate_win_count": int(np.sum(values < 0)),
            "candidate_loss_count": int(np.sum(values > 0)),
            "mean_delta_ah": float(np.mean(values)),
            "median_delta_ah": float(np.median(values)),
            "trimmed_mean_delta_ah": _trimmed_mean(values),
            "maximum_candidate_loss_delta_ah": float(np.max(values)),
            "positive_tail_share_of_absolute_delta": (
                float(np.sum(positive) / np.sum(np.abs(values))) if np.sum(np.abs(values)) else 0.0
            ),
            "exact_sign_flip": exact_sign_flip_test(values.tolist()),
            "leave_one_pack_out_mean_delta_ah": {
                pack_id: float(np.mean(np.delete(values, position)))
                for position, pack_id in enumerate(pack_ids)
            },
            "ranked_pack_influence": [
                {
                    "pack_id": pack_ids[position],
                    "delta_ah": float(values[position]),
                    "share_of_absolute_delta": (
                        float(abs(values[position]) / np.sum(np.abs(values)))
                        if np.sum(np.abs(values))
                        else 0.0
                    ),
                }
                for position in np.argsort(-np.abs(values))
            ],
        }

    primary = np.asarray(contrasts["order_dose"])
    protocol_groups: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        protocol_groups.setdefault(cast(str, row["protocol_family"]), []).append(index)
    modifiers = {
        "protocol_family": {
            key: {
                "pack_count": len(indices),
                "mean_delta_ah": float(np.mean(primary[indices])),
                "pack_ids": [pack_ids[index] for index in indices],
            }
            for key, indices in sorted(protocol_groups.items())
        },
        "transition_count_spearman": spearman_correlation(
            [cast(int, row["transition_count"]) for row in rows], primary.tolist()
        ),
        "mean_joint_support_distance_spearman": spearman_correlation(
            [cast(float, row["mean_joint_support_distance"]) for row in rows],
            primary.tolist(),
        ),
        "mean_current_capacity_spearman": spearman_correlation(
            [cast(float, row["mean_current_capacity_ah"]) for row in rows],
            primary.tolist(),
        ),
        "status": "DESCRIPTIVE_ONLY_TEN_PACKS_NO_SUBGROUP_PROMOTION",
    }
    transition_weighted = {}
    for comparator, field in prediction_fields.items():
        candidate_errors = np.asarray(
            [
                abs(
                    cast(float, row["candidate_next_prediction_ah"])
                    - cast(float, row["next_capacity_ah"])
                )
                for row in evaluable
            ]
        )
        if field.endswith("delta_prediction_ah"):
            comparator_predictions = np.asarray(
                [
                    cast(float, row["current_capacity_ah"]) + cast(float, row[field])
                    for row in evaluable
                ]
            )
        else:
            comparator_predictions = np.asarray([cast(float, row[field]) for row in evaluable])
        outcomes = np.asarray([cast(float, row["next_capacity_ah"]) for row in evaluable])
        transition_weighted[comparator] = {
            "candidate_mae_ah": float(np.mean(candidate_errors)),
            "comparator_mae_ah": float(np.mean(np.abs(comparator_predictions - outcomes))),
            "delta_ah": float(
                np.mean(candidate_errors) - np.mean(np.abs(comparator_predictions - outcomes))
            ),
            "scientific_status": "NESTED_TRANSITION_WEIGHTED_SENSITIVITY_ONLY",
        }
    macro_mae = {
        "candidate": float(np.mean([cast(float, row["candidate_mae_ah"]) for row in rows])),
        **{
            comparator: float(
                np.mean(
                    [
                        cast(Mapping[str, float], row["comparator_mae_ah"])[comparator]
                        for row in rows
                    ]
                )
            )
            for comparator in prediction_fields
        },
    }
    return {
        "analysis_id": 'battery-pack-reversal-analysis',
        "evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        "independent_unit": "complete-physical-battery-pack",
        "independent_unit_count": len(pack_ids),
        "primary_transition_count": len(evaluable),
        "pack_rows": rows,
        "contrast_summaries": summaries,
        "descriptive_modifiers": modifiers,
        "pack_balanced_macro_mae_ah": macro_mae,
        "transition_weighted_sensitivity": transition_weighted,
        "primary_majority_macro_reversal": (
            cast(int, cast(Mapping[str, object], summaries["order_dose"])["candidate_win_count"])
            > len(pack_ids) / 2
            and cast(float, cast(Mapping[str, object], summaries["order_dose"])["mean_delta_ah"])
            > 0
        ),
        "nested_transitions_used_as_replicates": False,
        "disposition": ThermodynamicPosthocDisposition.MIXED.value,
    }


def _wrap_phase(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _circular_summary_degrees(values: Sequence[float]) -> dict[str, float]:
    radians = np.radians(_finite_array(values))
    vector = np.mean(np.exp(1j * radians))
    resultant = float(abs(vector))
    deviation = math.degrees(math.sqrt(max(0.0, -2.0 * math.log(max(resultant, 1e-15)))))
    return {
        "circular_mean_deg": _wrap_phase(math.degrees(math.atan2(vector.imag, vector.real))),
        "resultant_length": resultant,
        "circular_standard_deviation_deg": deviation,
    }


def mastu_common_input_analysis(
    front_rows: Sequence[Mapping[str, object]],
    diagnostic_rows: Sequence[Mapping[str, object]],
    *,
    snr_threshold: float,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Estimate delivery-cancelling diagnostic ratios under common-input assumptions."""

    if snr_threshold <= 0:
        raise ValueError("SNR threshold must be positive")
    front_by_key = {
        (int(cast(int, row["discharge_id"])), float(cast(float, row["frequency_hz"]))): row
        for row in front_rows
    }
    if len(front_by_key) != len(front_rows):
        raise ValueError("front frequency cells are not unique")
    diagnostic_by_channel: dict[str, dict[tuple[int, float], Mapping[str, object]]] = {}
    for row in diagnostic_rows:
        channel = cast(str, row["output_channel"])
        key = (int(cast(int, row["discharge_id"])), float(cast(float, row["frequency_hz"])))
        if key in diagnostic_by_channel.setdefault(channel, {}):
            raise ValueError("diagnostic frequency cells are not unique within channel")
        diagnostic_by_channel[channel][key] = row

    channels: dict[str, object] = {}
    for channel, by_key in sorted(diagnostic_by_channel.items()):
        cells: list[dict[str, object]] = []
        for key, front in sorted(front_by_key.items()):
            diagnostic = by_key.get(key)
            qualification_reasons = []
            if diagnostic is None:
                qualification_reasons.append("DIAGNOSTIC_CELL_ABSENT")
            else:
                if cast(float, front["snr"]) < snr_threshold:
                    qualification_reasons.append("FRONT_LOW_SNR")
                if cast(float, diagnostic["snr"]) < snr_threshold:
                    qualification_reasons.append("DIAGNOSTIC_LOW_SNR")
                if cast(float, front["gain_m_per_1e21_s"]) <= 0:
                    qualification_reasons.append("FRONT_NONPOSITIVE_GAIN")
                if cast(float, diagnostic["gain_output_per_scaled_input"]) <= 0:
                    qualification_reasons.append("DIAGNOSTIC_NONPOSITIVE_GAIN")
                if diagnostic["geometry"] != front["geometry"]:
                    qualification_reasons.append("GEOMETRY_MISMATCH")
            jointly_qualified = bool(diagnostic is not None and not qualification_reasons)
            cell: dict[str, object] = {
                "discharge_id": key[0],
                "frequency_hz": key[1],
                "geometry": front["geometry"],
                "front_snr": front["snr"],
                "diagnostic_snr": diagnostic["snr"] if diagnostic is not None else None,
                "jointly_qualified": jointly_qualified,
                "qualification_reasons": qualification_reasons,
            }
            if jointly_qualified and diagnostic is not None:
                ratio = cast(float, front["gain_m_per_1e21_s"]) / cast(
                    float, diagnostic["gain_output_per_scaled_input"]
                )
                cell.update(
                    {
                        "ratio_native": ratio,
                        "ratio_unit": f"m/{diagnostic['output_unit']}",
                        "log_ratio": math.log(ratio),
                        "phase_difference_deg": _wrap_phase(
                            cast(float, front["phase_deg"]) - cast(float, diagnostic["phase_deg"])
                        ),
                        "inverse_identity_error": abs(
                            ratio
                            * cast(float, diagnostic["gain_output_per_scaled_input"])
                            / cast(float, front["gain_m_per_1e21_s"])
                            - 1.0
                        ),
                    }
                )
            cells.append(cell)
        qualified = [cell for cell in cells if cell["jointly_qualified"]]
        discharge_ids = sorted({cast(int, cell["discharge_id"]) for cell in qualified})
        if len(discharge_ids) < 2:
            summary: dict[str, object] = {
                "disposition": ThermodynamicPosthocDisposition.UNEVALUABLE.value,
                "reason": "FEWER_THAN_TWO_JOINTLY_QUALIFIED_DISCHARGES",
            }
        else:
            discharge_log: list[float] = []
            discharge_phase: list[float] = []
            for discharge_id in discharge_ids:
                selected = [cell for cell in qualified if cell["discharge_id"] == discharge_id]
                discharge_log.append(
                    float(np.mean([cast(float, cell["log_ratio"]) for cell in selected]))
                )
                phase_vector = np.mean(
                    np.exp(
                        1j
                        * np.radians(
                            [cast(float, cell["phase_difference_deg"]) for cell in selected]
                        )
                    )
                )
                discharge_phase.append(
                    math.degrees(math.atan2(phase_vector.imag, phase_vector.real))
                )
            log_interval = complete_unit_bootstrap_interval(
                discharge_log,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=seed + len(channel),
            )
            recurrent_frequency_rows = []
            for frequency in sorted({cast(float, cell["frequency_hz"]) for cell in qualified}):
                recurrent = [cell for cell in qualified if cell["frequency_hz"] == frequency]
                recurring_discharge_count = len(
                    {cast(int, cell["discharge_id"]) for cell in recurrent}
                )
                if recurring_discharge_count >= 2:
                    logs = [cast(float, cell["log_ratio"]) for cell in recurrent]
                    phases = [cast(float, cell["phase_difference_deg"]) for cell in recurrent]
                    recurrent_frequency_rows.append(
                        {
                            "frequency_hz": frequency,
                            "complete_discharge_count": recurring_discharge_count,
                            "geometric_mean_ratio": math.exp(float(np.mean(logs))),
                            "between_discharge_geometric_sd": math.exp(float(np.std(logs, ddof=1))),
                            "phase": _circular_summary_degrees(phases),
                            "disposition": "RECURRENT_EXACT_CELL",
                        }
                    )
                else:
                    recurrent_frequency_rows.append(
                        {
                            "frequency_hz": frequency,
                            "complete_discharge_count": recurring_discharge_count,
                            "disposition": "SINGLE_DISCHARGE_SENSITIVITY_ONLY",
                        }
                    )
            recurrent_frequency_count = sum(
                value["disposition"] == "RECURRENT_EXACT_CELL" for value in recurrent_frequency_rows
            )
            recurrent_rows = [
                value
                for value in recurrent_frequency_rows
                if value["disposition"] == "RECURRENT_EXACT_CELL"
            ]
            exactly_recurrent = bool(recurrent_rows) and all(
                math.isclose(
                    cast(float, value["between_discharge_geometric_sd"]),
                    1.0,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    cast(
                        float,
                        cast(Mapping[str, object], value["phase"])["resultant_length"],
                    ),
                    1.0,
                    rel_tol=0,
                    abs_tol=1e-12,
                )
                for value in recurrent_rows
            )
            summary = {
                "disposition": (
                    ThermodynamicPosthocDisposition.SUPPORTED.value
                    if exactly_recurrent
                    else ThermodynamicPosthocDisposition.MIXED.value
                    if recurrent_frequency_count > 0
                    else ThermodynamicPosthocDisposition.UNEVALUABLE.value
                ),
                "interpretation": (
                    "EXACT_RELATIVE_RECURRENCE_AT_NUMERICAL_PRECISION"
                    if exactly_recurrent
                    else "HETEROGENEOUS_RELATIVE_RECURRENCE_NO_EQUIVALENCE_MARGIN"
                    if recurrent_frequency_count > 0
                    else "RELATIVE_CELLS_ESTIMATED_NO_EXACT_FREQUENCY_RECURRENCE"
                ),
                "reason": (
                    None
                    if recurrent_frequency_count > 0
                    else "NO_EXACT_FREQUENCY_RECURRENT_SUPPORT"
                ),
                "qualified_cell_count": len(qualified),
                "qualified_discharge_count": len(discharge_ids),
                "discharge_ids": discharge_ids,
                "geometric_mean_ratio": math.exp(log_interval["mean"]),
                "geometric_mean_ratio_interval": {
                    "lower": math.exp(log_interval["lower"]),
                    "upper": math.exp(log_interval["upper"]),
                },
                "between_discharge_geometric_sd": math.exp(float(np.std(discharge_log, ddof=1))),
                "phase": _circular_summary_degrees(discharge_phase),
                "nested_cell_frequency_spearman_log_ratio_descriptive_only": spearman_correlation(
                    [cast(float, cell["frequency_hz"]) for cell in qualified],
                    [cast(float, cell["log_ratio"]) for cell in qualified],
                )
                if len(qualified) >= 3
                else None,
                "maximum_inverse_identity_error": max(
                    cast(float, cell["inverse_identity_error"]) for cell in qualified
                ),
                "exact_frequency_recurrence": recurrent_frequency_rows,
                "recurrent_frequency_count": recurrent_frequency_count,
                "exactly_recurrent_at_numerical_precision": exactly_recurrent,
                "cross_frequency_pooled_inference": False,
                "aggregate_interval_is_descriptive_when_recurrence_absent": (
                    recurrent_frequency_count == 0
                ),
            }
        channels[channel] = {"cells": cells, "summary": summary}

    front_pass = [row for row in front_rows if cast(float, row["snr"]) >= snr_threshold]
    sensitivities = {
        "front_hann_gain_relative_difference_median": float(
            np.median([cast(float, row["hann_gain_relative_difference"]) for row in front_rows])
        ),
        "front_hann_gain_relative_difference_maximum": float(
            np.max([cast(float, row["hann_gain_relative_difference"]) for row in front_rows])
        ),
        "front_hann_phase_difference_deg_maximum_absolute": float(
            np.max(np.abs([cast(float, row["hann_phase_difference_deg"]) for row in front_rows]))
        ),
        "front_linear_detrend_gain_relative_difference_maximum": float(
            np.max(
                [cast(float, row["linear_detrend_gain_relative_difference"]) for row in front_rows]
            )
        ),
        "front_linear_detrend_phase_difference_deg_maximum_absolute": float(
            np.max(
                np.abs(
                    [cast(float, row["linear_detrend_phase_difference_deg"]) for row in front_rows]
                )
            )
        ),
    }
    return {
        "analysis_id": 'mastu-common-input-analysis',
        "evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        "independent_unit": "complete-mastu-discharge",
        "front_cell_count": len(front_rows),
        "front_snr_pass_count": len(front_pass),
        "channels": channels,
        "spectral_sensitivities": sensitivities,
        "common_input_assumptions": [
            "same-unobserved-delivered-flow-channel",
            "local-linear-window",
            "clock-aligned-frequency-cell",
            "no-diagnostic-specific-confounded-input",
            "nonzero-qualified-denominator-response",
        ],
        "delivered_flow_identified": False,
        "thermodynamic_claim_permitted": False,
        "nested_frequency_lines_used_as_replicates": False,
        "disposition": ThermodynamicPosthocDisposition.MIXED.value,
    }


def mastu_partial_identification_analysis(
    relative_result: Mapping[str, object],
    delivery_constraints: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Construct the absolute response identified set without invented priors."""

    if delivery_constraints:
        gain_lowers = [cast(float, item["gain_lower"]) for item in delivery_constraints]
        gain_uppers = [cast(float, item["gain_upper"]) for item in delivery_constraints]
        if any(lower <= 0 or upper < lower for lower, upper in zip(gain_lowers, gain_uppers)):
            raise ValueError("delivery gain bounds are invalid")
        gain_lower = max(gain_lowers)
        gain_upper = min(gain_uppers)
        if gain_upper < gain_lower:
            raise ValueError("delivery constraints have an empty gain intersection")
        phase_bounds = [
            (cast(float, item["phase_lower_deg"]), cast(float, item["phase_upper_deg"]))
            for item in delivery_constraints
            if "phase_lower_deg" in item and "phase_upper_deg" in item
        ]
        delay_bounds = [
            (cast(float, item["delay_lower_s"]), cast(float, item["delay_upper_s"]))
            for item in delivery_constraints
            if "delay_lower_s" in item and "delay_upper_s" in item
        ]
        bandwidth_bounds = [
            cast(float, item["bandwidth_lower_hz"])
            for item in delivery_constraints
            if "bandwidth_lower_hz" in item
        ]
        identified_set: dict[str, object] = {
            "kind": "SOURCE_BOUNDED_SET",
            "delivery_gain_lower": gain_lower,
            "delivery_gain_upper": gain_upper,
            "delivery_phase_intersection_deg": (
                [max(value[0] for value in phase_bounds), min(value[1] for value in phase_bounds)]
                if phase_bounds
                else None
            ),
            "delivery_delay_intersection_s": (
                [max(value[0] for value in delay_bounds), min(value[1] for value in delay_bounds)]
                if delay_bounds
                else None
            ),
            "delivery_bandwidth_lower_hz": max(bandwidth_bounds) if bandwidth_bounds else None,
        }
        disposition = ThermodynamicPosthocDisposition.MIXED.value
    else:
        identified_set = {
            "kind": "UNBOUNDED_NONUNIQUE_SET",
            "magnitude": "(0,infinity)",
            "phase": "full-circle-modulo-source-requested-response",
            "delay": "unbounded-by-held-source",
            "witness": (
                "for any nonzero complex Q, (G,D) and (G*Q^-1,Q*D) produce "
                "the same requested-flow response H_req=G*D"
            ),
        }
        disposition = ThermodynamicPosthocDisposition.SUPPORTED.value
    channels = cast(Mapping[str, object], relative_result["channels"])
    qualified_relative_channels = [
        channel
        for channel, value in channels.items()
        if cast(Mapping[str, object], cast(Mapping[str, object], value)["summary"])["disposition"]
        == ThermodynamicPosthocDisposition.SUPPORTED.value
    ]
    return {
        "analysis_id": 'mastu-partial-identification-analysis',
        "evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        "independent_unit": "complete-mastu-discharge",
        "source_backed_delivery_constraint_count": len(delivery_constraints),
        "absolute_response_identified_set": identified_set,
        "relative_delivery_invariant_channels": qualified_relative_channels,
        "point_identified": False,
        "additional_measurement_required": (
            "time-aligned calibrated delivered-flow readback with gain, phase, delay, "
            "bandwidth and uncertainty"
        ),
        "disposition": disposition,
    }


def _ridge_affine_fit(starts: FloatArray, targets: FloatArray, ridge: float) -> FloatArray:
    if (
        starts.ndim != 2
        or targets.ndim != 2
        or starts.shape[0] != targets.shape[0]
        or starts.shape[0] < 2
        or ridge <= 0
        or not np.all(np.isfinite(starts))
        or not np.all(np.isfinite(targets))
    ):
        raise ValueError("ridge affine fit requires matched finite rows")
    start_mean = np.mean(starts, axis=0)
    target_mean = np.mean(targets, axis=0)
    centered_starts = starts - start_mean
    centered_targets = targets - target_mean
    feature_count = starts.shape[1]
    augmented_design = np.vstack((centered_starts, math.sqrt(ridge) * np.eye(feature_count)))
    augmented_targets = np.vstack(
        (
            centered_targets,
            np.zeros((feature_count, targets.shape[1]), dtype=np.float64),
        )
    )
    coefficients = np.linalg.lstsq(augmented_design, augmented_targets, rcond=None)[0]
    intercept = target_mean - start_mean @ coefficients
    return np.asarray(np.vstack((coefficients, intercept)), dtype=np.float64)


def _affine_predict(coefficients: FloatArray, starts: FloatArray) -> FloatArray:
    return np.column_stack((starts, np.ones(starts.shape[0]))) @ coefficients


def _delay_features(values: FloatArray, time_index: int, lag_depth: int) -> FloatArray:
    if values.ndim != 4 or lag_depth < 0 or time_index < lag_depth:
        raise ValueError("delay features require supported [unit,word,time,coordinate] values")
    blocks = [values[:, :, time_index - lag] for lag in range(lag_depth + 1)]
    return np.concatenate(blocks, axis=-1)


def delay_conditioned_cocycle_analysis(
    panel: TrajectoryPanel,
    *,
    lag_depths: tuple[int, ...],
    selected_time_indices: tuple[int, ...],
    ridge: float,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Cross-fit time-local delay-state propagation and cocycle defects."""

    if tuple(sorted(set(lag_depths))) != lag_depths or not lag_depths:
        raise ValueError("lag depths must be sorted and unique")
    if (
        not selected_time_indices
        or tuple(sorted(set(selected_time_indices))) != selected_time_indices
    ):
        raise ValueError("selected time indices must be sorted and unique")
    if len(panel.unit_ids) < 4:
        raise ValueError("delay cross-fitting requires at least four complete units")
    if selected_time_indices[0] < 0 or selected_time_indices[-1] >= panel.times.size:
        raise ValueError("selected time lies outside the panel")
    rows: list[dict[str, object]] = []
    variants: dict[int, dict[str, object]] = {}
    for lag_depth in lag_depths:
        anchors = [
            index
            for index in selected_time_indices
            if index >= lag_depth and index + 2 < panel.times.size
        ]
        if not anchors:
            variants[lag_depth] = {
                "disposition": ThermodynamicPosthocDisposition.UNEVALUABLE.value,
                "reason": "NO_SUPPORTED_ANCHOR",
            }
            continue
        per_unit_direct: list[list[float]] = [[] for _ in panel.unit_ids]
        per_unit_composed: list[list[float]] = [[] for _ in panel.unit_ids]
        per_unit_cocycle: list[list[float]] = [[] for _ in panel.unit_ids]
        per_unit_wrong: list[list[float]] = [[] for _ in panel.unit_ids]
        stationarity: list[float] = []
        for anchor in anchors:
            current = _delay_features(panel.values, anchor, lag_depth)
            next_state = _delay_features(panel.values, anchor + 1, lag_depth)
            last = _delay_features(panel.values, anchor + 2, lag_depth)
            for held_out in range(len(panel.unit_ids)):
                train = np.arange(len(panel.unit_ids)) != held_out
                train_current = current[train].reshape(-1, current.shape[-1])
                train_next = next_state[train].reshape(-1, next_state.shape[-1])
                train_last = last[train].reshape(-1, last.shape[-1])
                scale = np.std(train_current, axis=0, ddof=1)
                scale = np.where(scale > 1e-15, scale, 1.0)
                target_scale = np.std(train_last, axis=0, ddof=1)
                target_scale = np.where(target_scale > 1e-15, target_scale, 1.0)
                scaled_current = train_current / scale
                scaled_next = train_next / scale
                scaled_last = train_last / scale
                first_map = _ridge_affine_fit(scaled_current, scaled_next, ridge)
                second_map = _ridge_affine_fit(scaled_next, scaled_last, ridge)
                direct_map = _ridge_affine_fit(scaled_current, scaled_last, ridge)
                test = current[held_out] / scale
                truth = last[held_out] / scale
                direct = _affine_predict(direct_map, test)
                via = _affine_predict(second_map, _affine_predict(first_map, test))
                unit_direct = float(
                    np.sqrt(np.mean(((direct - truth) * scale / target_scale) ** 2))
                )
                unit_composed = float(np.sqrt(np.mean(((via - truth) * scale / target_scale) ** 2)))
                unit_cocycle = float(np.sqrt(np.mean(((via - direct) * scale / target_scale) ** 2)))
                per_unit_direct[held_out].append(unit_direct)
                per_unit_composed[held_out].append(unit_composed)
                per_unit_cocycle[held_out].append(unit_cocycle)
                if lag_depth > 0:
                    wrong = test.copy()
                    block = len(panel.coordinate_ids)
                    donor = (held_out + 1) % len(panel.unit_ids)
                    wrong[:, block:] = current[donor, :, block:] / scale[block:]
                    wrong_prediction = _affine_predict(direct_map, wrong)
                    per_unit_wrong[held_out].append(
                        float(
                            np.sqrt(
                                np.mean(((wrong_prediction - truth) * scale / target_scale) ** 2)
                            )
                        )
                    )
                rows.append(
                    {
                        "unit_id": panel.unit_ids[held_out],
                        "lag_depth": lag_depth,
                        "anchor_time": float(panel.times[anchor]),
                        "direct_prediction_rms": unit_direct,
                        "composed_prediction_rms": unit_composed,
                        "cocycle_prediction_defect_rms": unit_cocycle,
                        "wrong_history_prediction_rms": (
                            per_unit_wrong[held_out][-1] if lag_depth > 0 else None
                        ),
                    }
                )
                if anchor > lag_depth:
                    previous = _delay_features(panel.values, anchor - 1, lag_depth)
                    previous_train = previous[train].reshape(-1, previous.shape[-1]) / scale
                    previous_target = current[train].reshape(-1, current.shape[-1]) / scale
                    previous_map = _ridge_affine_fit(previous_train, previous_target, ridge)
                    stationarity.append(
                        float(
                            np.linalg.norm(first_map - previous_map, ord="fro")
                            / math.sqrt(first_map.size)
                        )
                    )
        direct_units = [float(np.mean(value)) for value in per_unit_direct]
        composed_units = [float(np.mean(value)) for value in per_unit_composed]
        cocycle_units = [float(np.mean(value)) for value in per_unit_cocycle]
        direct_interval = complete_unit_bootstrap_interval(
            direct_units,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed + lag_depth * 11,
        )
        cocycle_interval = complete_unit_bootstrap_interval(
            cocycle_units,
            replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed + lag_depth * 11 + 1,
        )
        wrong_units = [float(np.mean(value)) for value in per_unit_wrong] if lag_depth > 0 else []
        variants[lag_depth] = {
            "disposition": ThermodynamicPosthocDisposition.SUPPORTED.value,
            "anchor_count": len(anchors),
            "direct_prediction_rms": direct_interval,
            "composed_prediction_rms_mean": float(np.mean(composed_units)),
            "cocycle_prediction_defect_rms": cocycle_interval,
            "stationarity_operator_defect_mean": float(np.mean(stationarity))
            if stationarity
            else None,
            "wrong_history_prediction_rms_mean": float(np.mean(wrong_units))
            if wrong_units
            else None,
            "wrong_history_specificity_ratio": (
                float(np.mean(wrong_units) / np.mean(direct_units))
                if wrong_units and np.mean(direct_units) > 0
                else None
            ),
            "unit_metrics": [
                {
                    "unit_id": unit_id,
                    "direct_prediction_rms": direct_units[index],
                    "composed_prediction_rms": composed_units[index],
                    "cocycle_prediction_defect_rms": cocycle_units[index],
                    "wrong_history_prediction_rms": wrong_units[index] if wrong_units else None,
                }
                for index, unit_id in enumerate(panel.unit_ids)
            ],
        }
    baseline = cast(Mapping[str, object], variants.get(0, {}))
    baseline_direct = cast(Mapping[str, object], baseline.get("direct_prediction_rms", {})).get(
        "mean"
    )
    for lag_depth, value in variants.items():
        direct_mean = cast(Mapping[str, object], value.get("direct_prediction_rms", {})).get("mean")
        value["relative_direct_improvement_vs_lag0"] = (
            (cast(float, baseline_direct) - cast(float, direct_mean)) / cast(float, baseline_direct)
            if baseline_direct is not None
            and direct_mean is not None
            and cast(float, baseline_direct) > 0
            else None
        )
    baseline_unit_metrics = {
        cast(str, value["unit_id"]): value
        for value in cast(Sequence[Mapping[str, object]], baseline.get("unit_metrics", ()))
    }
    contrast_vectors: list[list[float]] = []
    contrast_keys: list[tuple[int, str]] = []
    for lag_depth, value in variants.items():
        if lag_depth == 0 or "unit_metrics" not in value:
            continue
        current_metrics = {
            cast(str, item["unit_id"]): item
            for item in cast(Sequence[Mapping[str, object]], value["unit_metrics"])
        }
        direct_improvement = [
            cast(float, baseline_unit_metrics[unit_id]["direct_prediction_rms"])
            - cast(float, current_metrics[unit_id]["direct_prediction_rms"])
            for unit_id in panel.unit_ids
        ]
        cocycle_improvement = [
            cast(
                float,
                baseline_unit_metrics[unit_id]["cocycle_prediction_defect_rms"],
            )
            - cast(float, current_metrics[unit_id]["cocycle_prediction_defect_rms"])
            for unit_id in panel.unit_ids
        ]
        wrong_history_penalty = [
            cast(float, current_metrics[unit_id]["wrong_history_prediction_rms"])
            - cast(float, current_metrics[unit_id]["direct_prediction_rms"])
            for unit_id in panel.unit_ids
        ]
        for contrast_id, values in (
            ("direct-prediction-improvement", direct_improvement),
            ("cocycle-defect-improvement", cocycle_improvement),
            ("wrong-history-penalty", wrong_history_penalty),
        ):
            contrast_keys.append((lag_depth, contrast_id))
            contrast_vectors.append(values)
    simultaneous_critical: float | None = None
    if contrast_vectors:
        contrast_family = np.asarray(contrast_vectors, dtype=np.float64).T
        contrast_mean, contrast_lower, contrast_upper, simultaneous_critical = (
            complete_unit_max_t_band(
                contrast_family,
                replicates=bootstrap_replicates,
                confidence_level=confidence_level,
                seed=seed + 9_919,
            )
        )
        for index, (lag_depth, contrast_id) in enumerate(contrast_keys):
            value = variants[lag_depth]
            summaries = cast(
                dict[str, object],
                value.setdefault("simultaneous_complete_unit_contrasts", {}),
            )
            summaries[contrast_id] = {
                "mean": float(contrast_mean[index]),
                "simultaneous_lower": float(contrast_lower[index]),
                "simultaneous_upper": float(contrast_upper[index]),
                "positive_means_improvement_or_specificity": True,
                "exact_sign_flip": exact_sign_flip_test(contrast_vectors[index]),
            }
    best = min(
        (
            (lag, cast(float, cast(Mapping[str, object], value["direct_prediction_rms"])["mean"]))
            for lag, value in variants.items()
            if "direct_prediction_rms" in value
        ),
        key=lambda item: item[1],
    )
    return {
        "panel_id": panel.panel_id,
        "system_id": panel.system_id,
        "split_id": panel.split_id,
        "numerical_view_id": panel.numerical_view_id,
        "state_view_id": panel.state_view_id,
        "independent_unit_count": len(panel.unit_ids),
        "nested_words_used_as_replicates": False,
        "lag_variants": {str(key): value for key, value in variants.items()},
        "best_predictive_lag_depth": best[0],
        "best_predictive_rms": best[1],
        "simultaneous_contrast_count": len(contrast_keys),
        "simultaneous_max_t_critical": simultaneous_critical,
        "cell_rows": rows,
    }


def state_conditioned_cocycle_analysis(
    panels: Sequence[tuple[TrajectoryPanel, tuple[int, ...]]],
    *,
    lag_depths: tuple[int, ...],
    ridge: float,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Run delay closure independently for every exact simulator state view."""

    results = [
        delay_conditioned_cocycle_analysis(
            panel,
            lag_depths=lag_depths,
            selected_time_indices=indices,
            ridge=ridge,
            bootstrap_replicates=bootstrap_replicates,
            confidence_level=confidence_level,
            seed=seed + position * 101,
        )
        for position, (panel, indices) in enumerate(panels)
    ]
    systems: dict[tuple[str, str, tuple[str, ...]], list[Mapping[str, object]]] = {}
    for result in results:
        panel_result = next(panel for panel, _ in panels if panel.panel_id == result["panel_id"])
        key = (
            cast(str, result["system_id"]),
            cast(str, result["split_id"]),
            panel_result.unit_ids,
        )
        systems.setdefault(key, []).append(result)
    comparisons: list[dict[str, object]] = []
    for (system_id, split_id, unit_ids), values in sorted(systems.items()):
        ordered = sorted(values, key=lambda value: cast(str, value["panel_id"]))
        if len(ordered) >= 2:
            primary = ordered[0]
            for candidate in ordered[1:]:
                comparisons.append(
                    {
                        "system_id": system_id,
                        "split_id": split_id,
                        "independent_unit_count": len(unit_ids),
                        "first_panel_id": primary["panel_id"],
                        "second_panel_id": candidate["panel_id"],
                        "first_state_view_id": primary["state_view_id"],
                        "second_state_view_id": candidate["state_view_id"],
                        "first_numerical_view_id": primary["numerical_view_id"],
                        "second_numerical_view_id": candidate["numerical_view_id"],
                        "first_best_rms": primary["best_predictive_rms"],
                        "second_best_rms": candidate["best_predictive_rms"],
                        "ratio_second_to_first": (
                            cast(float, candidate["best_predictive_rms"])
                            / cast(float, primary["best_predictive_rms"])
                            if cast(float, primary["best_predictive_rms"]) > 0
                            else None
                        ),
                    }
                )
    return {
        "analysis_id": 'state-conditioned-cocycle-analysis',
        "evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        "predictive_not_causal": True,
        "koopman_spectral_equivalence_claimed": False,
        "view_results": results,
        "cross_view_comparisons": comparisons,
        "disposition": ThermodynamicPosthocDisposition.MIXED.value,
    }


def _subset_panel(panel: TrajectoryPanel, indices: Sequence[int], suffix: str) -> TrajectoryPanel:
    positions = tuple(indices)
    if not positions or len(set(positions)) != len(positions):
        raise ValueError("receiver coordinate selection is invalid")
    return TrajectoryPanel(
        panel_id=f"{panel.panel_id}.{suffix}",
        system_id=panel.system_id,
        split_id=panel.split_id,
        numerical_view_id=panel.numerical_view_id,
        state_view_id=f"{panel.state_view_id}.{suffix}",
        unit_ids=panel.unit_ids,
        word_ids=panel.word_ids,
        times=panel.times.copy(),
        coordinate_ids=tuple(panel.coordinate_ids[index] for index in positions),
        native_units=tuple(panel.native_units[index] for index in positions),
        values=panel.values[..., positions].copy(),
        source_episode_sha256s=panel.source_episode_sha256s,
    )


def _receiver_axis_summary(
    panel: TrajectoryPanel,
    floor: FloatArray,
    *,
    selected_time_indices: tuple[int, ...],
) -> dict[str, object]:
    if floor.shape != (panel.times.size, len(panel.coordinate_ids)):
        raise ValueError("receiver floor differs from panel support")
    safe_floor = np.maximum(floor, 1e-15)
    responses = identity_relative_responses(panel)
    estimands = {
        key: value
        for key, value in finite_word_estimands(panel).items()
        if key != "response.identity"
    }

    def finite_metrics(unit_indices: npt.NDArray[np.int64]) -> dict[str, object]:
        response_ratio = max(
            float(np.max(np.abs(np.mean(responses[word][unit_indices], axis=0)) / safe_floor))
            for word in NONIDENTITY_WORD_IDS
        )
        curvature_ratio = float(
            np.max(
                np.abs(np.mean(estimands["controlled-order"][unit_indices], axis=0)) / safe_floor
            )
        )
        response_matrix = np.stack(
            [
                np.mean(responses[word][unit_indices], axis=0) / safe_floor
                for word in NONIDENTITY_WORD_IDS
            ],
            axis=1,
        )
        ranks = []
        word_distances = []
        for time_index in range(panel.times.size):
            singular = np.linalg.svd(response_matrix[time_index], compute_uv=False)
            ranks.append(
                int(
                    np.sum(
                        singular > math.sqrt(response_matrix.shape[1] * response_matrix.shape[2])
                    )
                )
            )
            values = response_matrix[time_index]
            distances = [
                float(np.linalg.norm(values[left] - values[right]))
                for left in range(values.shape[0])
                for right in range(left + 1, values.shape[0])
            ]
            word_distances.append(max(distances))
        return {
            "maximum_mean_response_floor_ratio": response_ratio,
            "maximum_mean_controlled_order_floor_ratio": curvature_ratio,
            "maximum_floor_certified_rank": max(ranks),
            "maximum_scaled_word_distance": max(word_distances),
            "action_class": ("RESOLVED" if response_ratio > 1 else "COLLAPSED_WITHIN_FLOOR"),
            "curvature_class": ("RESOLVED" if curvature_ratio > 1 else "BELOW_FLOOR"),
        }

    metrics = finite_metrics(np.arange(len(panel.unit_ids), dtype=np.int64))
    leave_one_out = []
    for held_out, unit_id in enumerate(panel.unit_ids):
        indices = np.delete(np.arange(len(panel.unit_ids), dtype=np.int64), held_out)
        leave_one_out.append({"held_out_unit_id": unit_id, **finite_metrics(indices)})
    valid_times = tuple(index for index in selected_time_indices if index < panel.times.size)
    temporal = delay_conditioned_cocycle_analysis(
        panel,
        lag_depths=(0,),
        selected_time_indices=valid_times,
        ridge=1e-6,
        bootstrap_replicates=500,
        confidence_level=0.95,
        seed=331,
    )
    lag_zero = cast(Mapping[str, object], cast(Mapping[str, object], temporal["lag_variants"])["0"])
    direct_prediction = cast(Mapping[str, object], lag_zero["direct_prediction_rms"])["mean"]
    return {
        "state_view_id": panel.state_view_id,
        "coordinate_ids": list(panel.coordinate_ids),
        "coordinate_count": len(panel.coordinate_ids),
        **metrics,
        "direct_prediction_rms": direct_prediction,
        "cocycle_prediction_defect_rms": cast(
            Mapping[str, object], lag_zero["cocycle_prediction_defect_rms"]
        )["mean"],
        "stationarity_operator_defect_mean": lag_zero["stationarity_operator_defect_mean"],
        "complete_unit_leave_one_out": leave_one_out,
    }


def receiver_faithfulness_analysis(
    families: Sequence[ReceiverFamilySpec],
    *,
    scale_tolerance: float,
) -> dict[str, object]:
    """Compare a finite named receiver family without arbitrary subset search."""

    if not math.isfinite(scale_tolerance) or scale_tolerance < 1:
        raise ValueError("receiver scale tolerance must be finite and at least one")

    systems: dict[str, object] = {}
    for family in families:
        summaries: list[dict[str, object]] = []
        for view in family.views:
            summary = _receiver_axis_summary(
                view.panel,
                view.numerical_floor,
                selected_time_indices=family.selected_time_indices,
            )
            summary["group_id"] = view.group_id
            summary["view_role"] = view.role
            summaries.append(summary)
        full = next(value for value in summaries if value["group_id"] == family.reference_group_id)
        for value in summaries:
            full_cocycle = cast(float, full["cocycle_prediction_defect_rms"])
            full_stationarity = full["stationarity_operator_defect_mean"]
            candidate_stationarity = value["stationarity_operator_defect_mean"]
            stationarity_preserved = (
                full_stationarity is None and candidate_stationarity is None
            ) or (
                full_stationarity is not None
                and candidate_stationarity is not None
                and cast(float, candidate_stationarity)
                <= scale_tolerance * max(cast(float, full_stationarity), 1e-15)
            )
            full_word_distance = cast(float, full["maximum_scaled_word_distance"])
            full_response_ratio = cast(float, full["maximum_mean_response_floor_ratio"])
            full_leave_one_out = {
                cast(str, row["held_out_unit_id"]): row
                for row in cast(
                    Sequence[Mapping[str, object]],
                    full["complete_unit_leave_one_out"],
                )
            }
            candidate_leave_one_out = {
                cast(str, row["held_out_unit_id"]): row
                for row in cast(
                    Sequence[Mapping[str, object]],
                    value["complete_unit_leave_one_out"],
                )
            }
            if set(candidate_leave_one_out) != set(full_leave_one_out):
                raise ValueError("receiver leave-one-unit-out families differ")
            leave_one_out_checks = {
                "action_rank": all(
                    candidate_leave_one_out[unit_id]["maximum_floor_certified_rank"]
                    == full_row["maximum_floor_certified_rank"]
                    for unit_id, full_row in full_leave_one_out.items()
                ),
                "curvature_order": all(
                    candidate_leave_one_out[unit_id]["curvature_class"]
                    == full_row["curvature_class"]
                    for unit_id, full_row in full_leave_one_out.items()
                ),
                "word_distinguishability": all(
                    cast(
                        float,
                        candidate_leave_one_out[unit_id]["maximum_scaled_word_distance"],
                    )
                    >= cast(float, full_row["maximum_scaled_word_distance"]) / scale_tolerance
                    for unit_id, full_row in full_leave_one_out.items()
                ),
                "numerical_envelope": all(
                    candidate_leave_one_out[unit_id]["action_class"] == full_row["action_class"]
                    and (
                        cast(
                            float,
                            full_row["maximum_mean_response_floor_ratio"],
                        )
                        <= 1
                        or cast(
                            float,
                            candidate_leave_one_out[unit_id]["maximum_mean_response_floor_ratio"],
                        )
                        >= cast(
                            float,
                            full_row["maximum_mean_response_floor_ratio"],
                        )
                        / scale_tolerance
                    )
                    for unit_id, full_row in full_leave_one_out.items()
                ),
            }
            axes = {
                "action_rank_preservation": value["maximum_floor_certified_rank"]
                == full["maximum_floor_certified_rank"]
                and leave_one_out_checks["action_rank"],
                "curvature_order_preservation": value["curvature_class"] == full["curvature_class"]
                and leave_one_out_checks["curvature_order"],
                "cocycle_class_preservation": cast(float, value["cocycle_prediction_defect_rms"])
                <= scale_tolerance * max(full_cocycle, 1e-15),
                "stationarity_class_preservation": stationarity_preserved,
                "word_distinguishability_preservation": cast(
                    float, value["maximum_scaled_word_distance"]
                )
                >= full_word_distance / scale_tolerance
                and leave_one_out_checks["word_distinguishability"],
                "predictive_loss_preservation": cast(float, value["direct_prediction_rms"])
                <= scale_tolerance * max(cast(float, full["direct_prediction_rms"]), 1e-15),
                "numerical_envelope_preservation": (
                    value["action_class"] == full["action_class"]
                    and (
                        full_response_ratio <= 1
                        or cast(float, value["maximum_mean_response_floor_ratio"])
                        >= full_response_ratio / scale_tolerance
                    )
                    and leave_one_out_checks["numerical_envelope"]
                ),
            }
            value["complete_unit_leave_one_out_agreement"] = leave_one_out_checks
            value["faithful_axes"] = axes
            value["faithful_axis_count"] = sum(axes.values())
            value["fully_faithful"] = all(axes.values())
        faithful = [value for value in summaries if value["fully_faithful"]]
        minimal = (
            min(faithful, key=lambda value: cast(int, value["coordinate_count"]))
            if faithful
            else None
        )
        dominance_rows = []
        for candidate in summaries:
            candidate_axes = cast(Mapping[str, bool], candidate["faithful_axes"])
            for comparator in summaries:
                if candidate["group_id"] == comparator["group_id"]:
                    continue
                comparator_axes = cast(Mapping[str, bool], comparator["faithful_axes"])
                candidate_set = {key for key, passed in candidate_axes.items() if passed}
                comparator_set = {key for key, passed in comparator_axes.items() if passed}
                dominates = (
                    cast(int, candidate["coordinate_count"])
                    <= cast(int, comparator["coordinate_count"])
                    and candidate_set >= comparator_set
                    and (
                        cast(int, candidate["coordinate_count"])
                        < cast(int, comparator["coordinate_count"])
                        or candidate_set > comparator_set
                    )
                )
                if dominates:
                    dominance_rows.append(
                        {
                            "dominant_group_id": candidate["group_id"],
                            "dominated_group_id": comparator["group_id"],
                        }
                    )
        reference_panel = next(
            value.panel for value in family.views if value.group_id == family.reference_group_id
        )
        systems[family.system_id] = {
            "receiver_groups": summaries,
            "minimal_faithful_group": minimal["group_id"] if minimal else None,
            "dominance_rows": dominance_rows,
            "reference_group_id": family.reference_group_id,
            "split_id": reference_panel.split_id,
            "scale_tolerance": scale_tolerance,
            "selection_is_outcome_visible": True,
            "fresh_confirmation_claimed": False,
        }
    return {
        "analysis_id": 'receiver-faithfulness-analysis',
        "evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        "arbitrary_subset_search_used": False,
        "faithfulness_axis_count": 7,
        "systems": systems,
        "disposition": ThermodynamicPosthocDisposition.MIXED.value,
    }


def gym_family_equivalence_rank_analysis(
    panel: TrajectoryPanel,
    numerical_floor: FloatArray,
    equivalence_width: FloatArray,
    *,
    bootstrap_replicates: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Family-wise equivalence and resolution-rank audit of the tested Gym chart."""

    if panel.system_id != "system.gym-torax-torax-simulation":
        raise ValueError("Gym equivalence analysis received another evidence world")
    if numerical_floor.shape != (panel.times.size, len(panel.coordinate_ids)):
        raise ValueError("Gym numerical floor differs from the receiver support")
    if equivalence_width.shape != (len(panel.coordinate_ids),):
        raise ValueError("Gym equivalence width differs from the receiver")
    estimands = {
        key: value
        for key, value in finite_word_estimands(panel).items()
        if key != "response.identity"
    }
    estimand_ids = tuple(sorted(estimands))
    family = np.stack([estimands[key] for key in estimand_ids], axis=1)
    mean, lower, upper, critical = complete_unit_max_t_band(
        family,
        replicates=bootstrap_replicates,
        confidence_level=confidence_level,
        seed=seed,
    )
    band_edge = np.maximum(np.abs(lower), np.abs(upper))
    width = equivalence_width[None, None, :]
    floor = np.maximum(numerical_floor, 1e-15)[None, :, :]
    summaries = []
    for index, estimand_id in enumerate(estimand_ids):
        summaries.append(
            {
                "estimand_id": estimand_id,
                "all_simultaneous_band_edges_within_equivalence": bool(
                    np.all(band_edge[index] <= width[0])
                ),
                "maximum_mean_floor_ratio": float(np.max(np.abs(mean[index]) / floor[0])),
                "maximum_band_edge_floor_ratio": float(np.max(band_edge[index] / floor[0])),
                "maximum_band_edge_equivalence_ratio": float(np.max(band_edge[index] / width[0])),
            }
        )
    response = identity_relative_responses(panel)
    response_family = np.stack([response[word] for word in NONIDENTITY_WORD_IDS], axis=1)
    generator = np.random.default_rng(seed + 1)
    max_ranks = np.empty(bootstrap_replicates, dtype=np.int16)
    threshold = math.sqrt(len(NONIDENTITY_WORD_IDS) * len(panel.coordinate_ids))
    offset = 0
    while offset < bootstrap_replicates:
        count = min(64, bootstrap_replicates - offset)
        indices = generator.integers(0, len(panel.unit_ids), size=(count, len(panel.unit_ids)))
        means = np.mean(response_family[indices], axis=1)
        scaled = means / np.maximum(numerical_floor, 1e-15)[None, None, :, :]
        matrices = np.transpose(scaled, (0, 2, 1, 3))
        singular = np.linalg.svd(matrices, compute_uv=False)
        ranks = np.sum(singular > threshold, axis=-1)
        max_ranks[offset : offset + count] = np.max(ranks, axis=1)
        offset += count
    mean_response = np.mean(response_family, axis=0)
    observed_singular = np.linalg.svd(
        np.transpose(
            mean_response / np.maximum(numerical_floor, 1e-15)[None, :, :],
            (1, 0, 2),
        ),
        compute_uv=False,
    )
    observed_ranks = np.sum(observed_singular > threshold, axis=-1)
    all_equivalent = all(
        cast(bool, value["all_simultaneous_band_edges_within_equivalence"]) for value in summaries
    )
    return {
        "analysis_id": 'gym-family-equivalence-rank-analysis',
        "evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        "independent_unit": "complete-gym-prepared-cell",
        "independent_unit_count": len(panel.unit_ids),
        "family_estimand_count": len(estimand_ids),
        "simultaneous_max_t_critical": critical,
        "estimands": summaries,
        "all_estimand_bands_within_equivalence": all_equivalent,
        "observed_maximum_floor_certified_rank": int(np.max(observed_ranks)),
        "bootstrap_maximum_rank_95_percentile": int(
            np.quantile(max_ranks, confidence_level, method="higher")
        ),
        "bootstrap_probability_maximum_rank_zero": float(np.mean(max_ranks == 0)),
        "rank_threshold": threshold,
        "tested_chart_only": True,
        "disposition": (
            ThermodynamicPosthocDisposition.SUPPORTED.value
            if all_equivalent
            else ThermodynamicPosthocDisposition.OPPOSED.value
        ),
    }


def _parse_pybamm_unit(unit_id: str) -> tuple[float, float]:
    match = re.search(r"soc-(\d+)p(\d+)-temp-(\d+)p(\d+)$", unit_id)
    if match is None:
        raise ValueError(f"PyBaMM unit ID lacks frozen SOC/temperature roles: {unit_id}")
    soc = float(f"{match.group(1)}.{match.group(2)}")
    temperature = float(f"{match.group(3)}.{match.group(4)}")
    return soc, temperature


def _history_gain_by_unit(
    panel: TrajectoryPanel,
    cutoff_indices: tuple[int, ...],
    *,
    ridge: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for held_out, unit_id in enumerate(panel.unit_ids):
        train = np.arange(len(panel.unit_ids)) != held_out
        current_errors: list[float] = []
        history_errors: list[float] = []
        cutoff_rows: list[dict[str, object]] = []
        for cutoff in cutoff_indices:
            if cutoff <= 0 or cutoff >= panel.times.size - 1:
                continue
            current_train = panel.values[train, :, cutoff].reshape(-1, len(panel.coordinate_ids))
            prior_train = panel.values[train, :, cutoff - 1].reshape(-1, len(panel.coordinate_ids))
            target_train = panel.values[train, :, -1].reshape(-1, len(panel.coordinate_ids))
            current_center = np.mean(current_train, axis=0)
            prior_center = np.mean(prior_train, axis=0)
            target_center = np.mean(target_train, axis=0)
            current_scale = np.std(current_train, axis=0, ddof=1)
            prior_scale = np.std(prior_train, axis=0, ddof=1)
            target_scale = np.std(target_train, axis=0, ddof=1)
            current_scale = np.where(current_scale > 1e-15, current_scale, 1.0)
            prior_scale = np.where(prior_scale > 1e-15, prior_scale, 1.0)
            target_scale = np.where(target_scale > 1e-15, target_scale, 1.0)
            current_train_scaled = (current_train - current_center) / current_scale
            prior_train_scaled = (prior_train - prior_center) / prior_scale
            target_train_scaled = (target_train - target_center) / target_scale
            current_map = _ridge_affine_fit(current_train_scaled, target_train_scaled, ridge)
            history_map = _ridge_affine_fit(
                np.column_stack((current_train_scaled, prior_train_scaled)),
                target_train_scaled,
                ridge,
            )
            current = (panel.values[held_out, :, cutoff] - current_center) / current_scale
            prior = (panel.values[held_out, :, cutoff - 1] - prior_center) / prior_scale
            target = (panel.values[held_out, :, -1] - target_center) / target_scale
            current_prediction = _affine_predict(current_map, current)
            history_prediction = _affine_predict(history_map, np.column_stack((current, prior)))
            current_error = float(np.sqrt(np.mean((current_prediction - target) ** 2)))
            history_error = float(np.sqrt(np.mean((history_prediction - target) ** 2)))
            current_errors.append(current_error)
            history_errors.append(history_error)
            cutoff_rows.append(
                {
                    "cutoff_index": cutoff,
                    "cutoff_time": float(panel.times[cutoff]),
                    "current_prediction_rms": current_error,
                    "one_lag_prediction_rms": history_error,
                    "relative_history_improvement": (
                        (current_error - history_error) / current_error
                        if current_error > 0
                        else 0.0
                    ),
                }
            )
        if not current_errors:
            raise ValueError("PyBaMM history family has no supported causal cutoff")
        current_mean = float(np.mean(current_errors))
        history_mean = float(np.mean(history_errors))
        rows.append(
            {
                "unit_id": unit_id,
                "current_prediction_rms": current_mean,
                "one_lag_prediction_rms": history_mean,
                "relative_history_improvement": (
                    (current_mean - history_mean) / current_mean if current_mean > 0 else 0.0
                ),
                "cutoff_rows": cutoff_rows,
            }
        )
    return rows


def pybamm_curvature_memory_localization_analysis(
    panel: TrajectoryPanel,
    numerical_floor: FloatArray,
    materiality_width: FloatArray,
    *,
    ridge: float,
    cutoff_indices: tuple[int, ...],
    delivery_windows: Mapping[str, tuple[float, float]],
) -> dict[str, object]:
    """Localize finite curvature against simulated state and predictive memory."""

    if panel.system_id != "system.pybamm-spme-thermal-chen2020":
        raise ValueError("PyBaMM localization received another evidence world")
    expected = (panel.times.size, len(panel.coordinate_ids))
    if numerical_floor.shape != expected or materiality_width.shape != (len(panel.coordinate_ids),):
        raise ValueError("PyBaMM localization scales differ from panel support")
    controlled = finite_word_estimands(panel)["controlled-order"]
    floor_scaled = np.abs(controlled) / np.maximum(numerical_floor, 1e-15)[None, :, :]
    materiality_scaled = np.abs(controlled) / np.maximum(materiality_width, 1e-15)[None, None, :]
    if (
        not cutoff_indices
        or tuple(sorted(set(cutoff_indices))) != cutoff_indices
        or cutoff_indices[0] <= 0
        or cutoff_indices[-1] >= panel.times.size - 1
    ):
        raise ValueError("PyBaMM history cutoff family is invalid")
    if not delivery_windows:
        raise ValueError("PyBaMM delivery windows must be predeclared")
    previous_end = -math.inf
    for window_id, (start, end) in delivery_windows.items():
        validate_stable_id(window_id, field_name="delivery_window_id")
        if not math.isfinite(start) or not math.isfinite(end) or start > end:
            raise ValueError("PyBaMM delivery window is invalid")
        if start <= previous_end:
            raise ValueError("PyBaMM delivery windows must be ordered and nonoverlapping")
        previous_end = end
    history_rows = _history_gain_by_unit(panel, cutoff_indices, ridge=ridge)
    history_by_unit = {
        cast(str, row["unit_id"]): cast(float, row["relative_history_improvement"])
        for row in history_rows
    }
    unit_rows: list[dict[str, object]] = []
    for unit_index, unit_id in enumerate(panel.unit_ids):
        soc, temperature = _parse_pybamm_unit(unit_id)
        coordinate_rows = []
        for coordinate_index, coordinate_id in enumerate(panel.coordinate_ids):
            peak_index = int(np.argmax(floor_scaled[unit_index, :, coordinate_index]))
            coordinate_rows.append(
                {
                    "coordinate_id": coordinate_id,
                    "native_unit": panel.native_units[coordinate_index],
                    "peak_time": float(panel.times[peak_index]),
                    "peak_floor_ratio": float(
                        floor_scaled[unit_index, peak_index, coordinate_index]
                    ),
                    "peak_materiality_ratio": float(
                        materiality_scaled[unit_index, peak_index, coordinate_index]
                    ),
                    "absolute_time_integral": float(
                        np.trapezoid(
                            np.abs(controlled[unit_index, :, coordinate_index]), panel.times
                        )
                    ),
                }
            )
        maximum_position = np.unravel_index(
            int(np.argmax(materiality_scaled[unit_index])),
            materiality_scaled[unit_index].shape,
        )
        history_cutoffs = cast(
            Sequence[Mapping[str, object]], history_rows[unit_index]["cutoff_rows"]
        )
        best_history = max(
            history_cutoffs,
            key=lambda value: cast(float, value["relative_history_improvement"]),
        )
        unit_rows.append(
            {
                "unit_id": unit_id,
                "initial_soc": soc,
                "initial_temperature_k": temperature,
                "history_improvement": history_by_unit[unit_id],
                "maximum_floor_ratio": max(
                    cast(float, value["peak_floor_ratio"]) for value in coordinate_rows
                ),
                "maximum_materiality_ratio": max(
                    cast(float, value["peak_materiality_ratio"]) for value in coordinate_rows
                ),
                "maximum_materiality_time": float(panel.times[maximum_position[0]]),
                "maximum_materiality_coordinate_id": panel.coordinate_ids[maximum_position[1]],
                "maximum_history_improvement": best_history["relative_history_improvement"],
                "maximum_history_improvement_cutoff_time": best_history["cutoff_time"],
                "curvature_history_peak_offset": float(panel.times[maximum_position[0]])
                - cast(float, best_history["cutoff_time"]),
                "coordinate_rows": coordinate_rows,
            }
        )
    maximum_materiality = [cast(float, row["maximum_materiality_ratio"]) for row in unit_rows]
    initial_soc = [cast(float, row["initial_soc"]) for row in unit_rows]
    initial_temperature = [cast(float, row["initial_temperature_k"]) for row in unit_rows]
    history_improvement = [cast(float, row["history_improvement"]) for row in unit_rows]
    correlations = {
        "curvature_vs_initial_soc": spearman_correlation(initial_soc, maximum_materiality),
        "curvature_vs_initial_temperature": spearman_correlation(
            initial_temperature, maximum_materiality
        ),
        "curvature_vs_history_improvement": spearman_correlation(
            history_improvement, maximum_materiality
        ),
    }
    correlation_inference = {
        "curvature_vs_initial_soc": _exact_spearman_permutation_test(
            initial_soc, maximum_materiality
        ),
        "curvature_vs_initial_temperature": _exact_spearman_permutation_test(
            initial_temperature, maximum_materiality
        ),
        "curvature_vs_history_improvement": _exact_spearman_permutation_test(
            history_improvement, maximum_materiality
        ),
    }
    window_rows = []
    for window_id, (start, end) in delivery_windows.items():
        indices = np.flatnonzero((panel.times >= start) & (panel.times <= end))
        if indices.size:
            for coordinate_index, coordinate_id in enumerate(panel.coordinate_ids):
                window_rows.append(
                    {
                        "window_id": window_id,
                        "coordinate_id": coordinate_id,
                        "maximum_mean_floor_ratio": float(
                            np.max(np.mean(floor_scaled[:, indices, coordinate_index], axis=0))
                        ),
                        "maximum_mean_materiality_ratio": float(
                            np.max(
                                np.mean(materiality_scaled[:, indices, coordinate_index], axis=0)
                            )
                        ),
                    }
                )
    cooccurrence_rows = []
    for cutoff in cutoff_indices:
        curvature_at_cutoff = np.max(materiality_scaled[:, cutoff, :], axis=1)
        history_at_cutoff = []
        for row in history_rows:
            selected = next(
                value
                for value in cast(Sequence[Mapping[str, object]], row["cutoff_rows"])
                if value["cutoff_index"] == cutoff
            )
            history_at_cutoff.append(cast(float, selected["relative_history_improvement"]))
        cooccurrence_rows.append(
            {
                "cutoff_index": cutoff,
                "cutoff_time": float(panel.times[cutoff]),
                "curvature_history_spearman": spearman_correlation(
                    curvature_at_cutoff.tolist(), history_at_cutoff
                ),
                "curvature_history_exact_permutation": (
                    _exact_spearman_permutation_test(
                        curvature_at_cutoff.tolist(), history_at_cutoff
                    )
                ),
                "mean_maximum_materiality_ratio": float(np.mean(curvature_at_cutoff)),
                "mean_relative_history_improvement": float(np.mean(history_at_cutoff)),
                "positive_history_improvement_cell_count": int(
                    np.sum(np.asarray(history_at_cutoff) > 0)
                ),
            }
        )
    coordinate_summaries = []
    for coordinate_index, coordinate_id in enumerate(panel.coordinate_ids):
        mean_curve = np.mean(materiality_scaled[:, :, coordinate_index], axis=0)
        peak_index = int(np.argmax(mean_curve))
        coordinate_summaries.append(
            {
                "coordinate_id": coordinate_id,
                "native_unit": panel.native_units[coordinate_index],
                "peak_mean_materiality_ratio": float(mean_curve[peak_index]),
                "peak_mean_materiality_time": float(panel.times[peak_index]),
            }
        )
    return {
        "analysis_id": 'pybamm-curvature-memory-localization-analysis',
        "evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        "independent_unit": "complete-pybamm-prepared-cell",
        "independent_unit_count": len(panel.unit_ids),
        "unit_rows": unit_rows,
        "history_rows": history_rows,
        "curvature_history_cooccurrence": cooccurrence_rows,
        "coordinate_summaries": coordinate_summaries,
        "window_rows": window_rows,
        "complete_cell_correlations": correlations,
        "complete_cell_correlation_inference": correlation_inference,
        "maximum_unit_floor_ratio": max(
            cast(float, row["maximum_floor_ratio"]) for row in unit_rows
        ),
        "maximum_unit_materiality_ratio": max(maximum_materiality),
        "finite_not_infinitesimal": True,
        "physical_cell_claim_permitted": False,
        "disposition": ThermodynamicPosthocDisposition.MIXED.value,
    }


CLAIM_AXES: Final = (
    "action_image",
    "constituent_port_materiality",
    "controlled_curvature",
    "delivery_observability",
    "receiver_faithfulness",
    "stationarity",
    "temporal_cocycle",
    "thermodynamic_ledger_return",
    "transport_membership",
)


def classify_claim_axes(readout: Mapping[str, object]) -> dict[str, int]:
    """Map one finite readout to an ordinal claim-strength vector."""

    action = readout.get("action_quotient")
    ports = readout.get("constituent_ports")
    controlled = readout.get("controlled_order")
    receiver = readout.get("receiver_map")
    return {
        "action_image": 2
        if action in {"MATERIAL", "RESOLVED"}
        else 1
        if action in {"COLLAPSED_WITHIN_FLOOR", "EQUIVALENT"}
        else 0,
        "constituent_port_materiality": 2
        if ports == "BOTH_MATERIAL"
        else 1
        if ports in {"ONE_MATERIAL", "BOTH_BELOW_FLOOR"}
        else 0,
        "controlled_curvature": 2
        if controlled in {"MATERIAL", "RESOLVED"}
        else 1
        if controlled in {"EQUIVALENT", "BELOW_FLOOR"}
        else 0,
        "delivery_observability": 2
        if readout.get("delivery") == "REALIZED_OBSERVED"
        else 1
        if readout.get("delivery") == "REQUESTED_ONLY"
        else 0,
        "receiver_faithfulness": 2
        if receiver == "FAITHFUL_AT_TESTED_RESOLUTION"
        else 1
        if receiver == "PARTIALLY_COLLAPSING"
        else 0,
        "stationarity": 2
        if readout.get("equal_lag_stationarity") == "STATIONARY"
        else 1
        if readout.get("equal_lag_stationarity") == "NONSTATIONARY"
        else 0,
        "temporal_cocycle": 2
        if readout.get("temporal_cocycle") == "COCYCLIC"
        else 1
        if readout.get("temporal_cocycle") == "DEFECT_RESOLVED"
        else 0,
        "thermodynamic_ledger_return": 2
        if readout.get("state_bath_return") == "RETURN_QUALIFIED"
        and readout.get("energy_balance") == "CLOSED"
        else 1
        if readout.get("energy_balance") == "PARTIAL"
        else 0,
        "transport_membership": 2
        if readout.get("transport") == "COMMUTES_AT_TESTED_RESOLUTION"
        else 1
        if readout.get("transport") == "OPPOSED"
        else 0,
    }


def claim_order_violations(before: Mapping[str, int], after: Mapping[str, int]) -> tuple[str, ...]:
    """Return axes that became stronger after information was removed."""

    if set(before) != set(CLAIM_AXES) or set(after) != set(CLAIM_AXES):
        raise ValueError("claim vectors must cover the exact registered axes")
    if any(value not in {0, 1, 2} for value in (*before.values(), *after.values())):
        raise ValueError("claim ranks must lie in the closed ordinal vocabulary")
    return tuple(axis for axis in CLAIM_AXES if after[axis] > before[axis])


def claim_monotonicity_analysis(
    readouts: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Stress that evidence removal/coarsening cannot strengthen claim ranks."""

    transformations = {
        "remove-delivered-action": (
            "delivery",
            "action_quotient",
            "constituent_ports",
            "controlled_order",
        ),
        "merge-action-letters": (
            "action_quotient",
            "constituent_ports",
            "controlled_order",
        ),
        "erase-chronological-order": (
            "controlled_order",
            "temporal_cocycle",
            "equal_lag_stationarity",
        ),
        "replace-realized-clock": ("temporal_cocycle", "equal_lag_stationarity"),
        "coarsen-receiver": (
            "receiver_map",
            "action_quotient",
            "constituent_ports",
            "controlled_order",
        ),
        "remove-bath-storage": ("state_bath_return", "energy_balance"),
        "increase-numerical-floor": (
            "action_quotient",
            "constituent_ports",
            "controlled_order",
        ),
        "erase-causal-history": ("temporal_cocycle", "equal_lag_stationarity"),
        "remove-complete-unit": (
            "delivery",
            "action_quotient",
            "constituent_ports",
            "controlled_order",
            "temporal_cocycle",
            "equal_lag_stationarity",
            "receiver_map",
            "state_bath_return",
            "energy_balance",
            "transport",
        ),
        "invalidate-snr-or-validity": (
            "delivery",
            "action_quotient",
            "constituent_ports",
            "controlled_order",
            "temporal_cocycle",
            "equal_lag_stationarity",
            "receiver_map",
            "state_bath_return",
            "energy_balance",
            "transport",
        ),
    }
    cases: list[dict[str, object]] = []
    false_promotions = 0
    for readout in readouts:
        input_id = cast(str, readout["input_id"])
        ranks = classify_claim_axes(readout)
        for transform_id, affected_fields in transformations.items():
            transformed_readout = dict(readout)
            for field in affected_fields:
                transformed_readout[field] = "UNEVALUABLE_INFORMATION_REMOVED"
            transformed = classify_claim_axes(transformed_readout)
            stronger = claim_order_violations(ranks, transformed)
            false_promotions += len(stronger)
            cases.append(
                {
                    "input_id": input_id,
                    "transformation_id": transform_id,
                    "before": ranks,
                    "after": transformed,
                    "stronger_axes": stronger,
                    "passed": not stronger,
                }
            )
    return {
        "analysis_id": 'claim-monotonicity-analysis',
        "evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        "input_readout_count": len(readouts),
        "transformation_count": len(transformations),
        "case_count": len(cases),
        "false_promotion_count": false_promotions,
        "gate_passed": false_promotions == 0,
        "cases": cases,
        "disposition": (
            ThermodynamicPosthocDisposition.SUPPORTED.value
            if false_promotions == 0
            else ThermodynamicPosthocDisposition.METHOD_FAILURE.value
        ),
    }


def non_entailment_lattice_analysis(
    cells: Sequence[Mapping[str, object]],
    *,
    definitional_edges: Sequence[tuple[str, str]],
) -> dict[str, object]:
    """Build a finite assay-bounded witnessed non-entailment table."""

    propositions = tuple(sorted(CLAIM_AXES))
    edge_set = set(definitional_edges)
    if any(left not in propositions or right not in propositions for left, right in edge_set):
        raise ValueError("definitional edge lies outside the proposition registry")
    successors: dict[str, set[str]] = {value: set() for value in propositions}
    indegree = {value: 0 for value in propositions}
    for left, right in edge_set:
        if left == right:
            raise ValueError("definitional implication cannot be reflexive")
        successors[left].add(right)
        indegree[right] += 1
    queue = sorted(value for value, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        current = queue.pop(0)
        visited += 1
        for successor in sorted(successors[current]):
            indegree[successor] -= 1
            if indegree[successor] == 0:
                queue.append(successor)
                queue.sort()
    if visited != len(propositions):
        raise ValueError("definitional implication registry contains a cycle")
    allowed_statuses = {"SUPPORTED", "OPPOSED", "UNEVALUABLE"}
    for cell in cells:
        validate_stable_id(cast(str, cell["cell_id"]), field_name="cell_id")
        if cell.get("witness_class") not in {"TRUTH_KNOWN", "EMPIRICAL"}:
            raise ValueError("lattice cell lacks a registered witness class")
        values = cast(Mapping[str, object], cell["propositions"])
        if set(values) != set(propositions) or any(
            value not in allowed_statuses for value in values.values()
        ):
            raise ValueError("lattice cell does not cover the exact proposition vocabulary")
    pairs: list[dict[str, object]] = []
    witnesses: list[dict[str, str]] = []
    for antecedent in propositions:
        for consequent in propositions:
            if antecedent == consequent:
                continue
            pair_witnesses = []
            for cell in cells:
                values = cast(Mapping[str, object], cell["propositions"])
                if values.get(antecedent) == "SUPPORTED" and values.get(consequent) == "OPPOSED":
                    pair_witnesses.append(cast(str, cell["cell_id"]))
            if (antecedent, consequent) in edge_set:
                status = "FORMALLY_ENTAILED_BY_DEFINITION"
                if pair_witnesses:
                    raise ValueError("empirical witness contradicts a definitional edge")
            elif pair_witnesses:
                status = "WITNESSED_NON_ENTAILMENT"
                witness_cell = next(cell for cell in cells if cell["cell_id"] == pair_witnesses[0])
                witnesses.append(
                    {
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "witness": pair_witnesses[0],
                        "witness_class": cast(str, witness_cell["witness_class"]),
                    }
                )
            else:
                antecedent_supported = [
                    cell
                    for cell in cells
                    if cast(Mapping[str, object], cell["propositions"])[antecedent] == "SUPPORTED"
                ]
                jointly_evaluable = any(
                    cast(Mapping[str, object], cell["propositions"])[consequent] != "UNEVALUABLE"
                    for cell in antecedent_supported
                )
                antecedent_opposed = any(
                    cast(Mapping[str, object], cell["propositions"])[antecedent] == "OPPOSED"
                    for cell in cells
                )
                if antecedent_supported and jointly_evaluable:
                    status = "CO_OCCURS_ONLY"
                elif not antecedent_supported and antecedent_opposed:
                    status = "OPPOSED_BY_VALID_ASSAY"
                else:
                    status = "UNEVALUABLE_OPERAND"
            pairs.append(
                {
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "status": status,
                    "witness_ids": pair_witnesses,
                }
            )
    return {
        "analysis_id": 'non-entailment-lattice-analysis',
        "evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        "propositions": list(propositions),
        "cells": list(cells),
        "definitional_edges": [list(value) for value in sorted(edge_set)],
        "ordered_pairs": pairs,
        "witnessed_non_entailment_count": len(witnesses),
        "truth_known_witness_count": sum(
            value["witness_class"] == "TRUTH_KNOWN" for value in witnesses
        ),
        "empirical_witness_count": sum(
            value["witness_class"] == "EMPIRICAL" for value in witnesses
        ),
        "minimal_witnesses": witnesses,
        "statistical_independence_claimed": False,
        "universality_claimed": False,
        "disposition": ThermodynamicPosthocDisposition.SUPPORTED.value,
    }


def measurement_value_of_information_analysis(
    *,
    gym_result: Mapping[str, object],
    pybamm_result: Mapping[str, object],
    battery_result: Mapping[str, object],
    mastu_result: Mapping[str, object],
    mastu_partial_result: Mapping[str, object],
) -> dict[str, object]:
    """Translate exact gaps into a noncompensating prospective measurement frontier."""

    gym_estimands = cast(Sequence[Mapping[str, object]], gym_result["estimands"])
    gym_by_id = {cast(str, value["estimand_id"]): value for value in gym_estimands}
    action_ratios = [
        cast(float, gym_by_id[f"response.{word}"]["maximum_band_edge_floor_ratio"])
        for word in NONIDENTITY_WORD_IDS
    ]
    controlled_ratio = cast(float, gym_by_id["controlled-order"]["maximum_band_edge_floor_ratio"])
    battery_primary = cast(
        Mapping[str, object],
        cast(Mapping[str, object], battery_result["contrast_summaries"])["order_dose"],
    )
    pack_deltas = [
        cast(float, cast(Mapping[str, object], row["delta_vs_comparator_ah"])["order_dose"])
        for row in cast(Sequence[Mapping[str, object]], battery_result["pack_rows"])
    ]
    mean_delta = abs(float(np.mean(pack_deltas)))
    required_packs = (
        int(math.ceil((1.96 * float(np.std(pack_deltas, ddof=1)) / mean_delta) ** 2))
        if mean_delta > 0
        else None
    )
    max_frequency = 0.0
    for value in cast(Mapping[str, object], mastu_result["channels"]).values():
        for cell in cast(
            Sequence[Mapping[str, object]], cast(Mapping[str, object], value)["cells"]
        ):
            max_frequency = max(max_frequency, cast(float, cell["frequency_hz"]))
    pybamm_materiality = cast(float, pybamm_result["maximum_unit_materiality_ratio"])
    maximum_action_ratio = max(action_ratios)
    action_floor_reduction = 1 / maximum_action_ratio if maximum_action_ratio > 0 else None
    controlled_floor_reduction = 1 / controlled_ratio if controlled_ratio > 0 else None
    bandwidth_lower = 2 * max_frequency if max_frequency > 0 else None
    clock_error = 5 / (360 * max_frequency) if max_frequency > 0 else None
    pybamm_precision_factor = 1 / pybamm_materiality if pybamm_materiality > 0 else None
    candidates = [
        {
            "candidate_id": "mastu-delivered-flow-readback",
            "operand": "delivered gas gain/phase/delay",
            "current_gap": cast(
                Mapping[str, object], mastu_partial_result["absolute_response_identified_set"]
            )["kind"],
            "required_bandwidth_hz_lower_bound": bandwidth_lower,
            "required_clock_error_s_for_5deg_phase": clock_error,
            "noise_model_status": "UNEVALUABLE_NO_DELIVERED_FLOW_NOISE_MODEL",
            "identified_set_effect": "turn unbounded delivery equivalence into source-bounded set",
            "priority": "PARETO_NONDOMINATED",
        },
        {
            "candidate_id": "mastu-second-delivered-port",
            "operand": "two-port transition triples",
            "required_measurement": "independent applied/realized readback for both ports",
            "noise_model_status": "UNEVALUABLE_NO_TWO_PORT_SOURCE_NOISE_MODEL",
            "identified_set_effect": "make finite-word composition evaluable",
            "priority": "PARETO_NONDOMINATED",
        },
        {
            "candidate_id": "mastu-particle-energy-storage-ledger",
            "operand": "particle/energy/storage boundary terms and return",
            "required_measurement": "time-aligned absolute ledger with uncertainty",
            "noise_model_status": "UNEVALUABLE_NO_LEDGER_SENSOR_MODEL",
            "identified_set_effect": "make balance/return axes evaluable",
            "priority": "PARETO_NONDOMINATED",
        },
        {
            "candidate_id": "boptest-realized-action-state-interface",
            "operand": "native realized actions and embedded state",
            "required_measurement": "source interface exposure; numeric precision unevaluable",
            "noise_model_status": "UNEVALUABLE_INTERFACE_ABSENT",
            "identified_set_effect": "move zero-word boundary to response computability",
            "priority": "PARETO_NONDOMINATED",
        },
        {
            "candidate_id": "gym-action-resolution",
            "operand": "tested action fibre",
            "maximum_action_band_floor_ratio": maximum_action_ratio,
            "minimum_floor_reduction_factor_to_touch_resolution": action_floor_reduction,
            "controlled_order_floor_reduction_factor": controlled_floor_reduction,
            "noise_model_status": "BOUND_FROM_PARENT_NUMERICAL_ENVELOPE_ONLY",
            "identified_set_effect": "resolve or tighten the exact tested-chart rank",
            "priority": "PARETO_NONDOMINATED",
        },
        {
            "candidate_id": "pybamm-physical-cell-ledger",
            "operand": "delivered two-port physical-cell calorimetry/storage/return",
            "maximum_simulated_curvature_materiality_ratio": pybamm_materiality,
            "minimum_effect_or_precision_factor_to_touch_materiality": pybamm_precision_factor,
            "noise_model_status": "SIMULATION_NUMERICAL_SCALE_NOT_PHYSICAL_SENSOR_NOISE",
            "identified_set_effect": "test whether simulated localization survives physically",
            "priority": "PARETO_NONDOMINATED",
        },
        {
            "candidate_id": "battery-pack-replication-and-ledger",
            "operand": "pack transport plus heat/storage/return",
            "current_pack_count": battery_result["independent_unit_count"],
            "approximate_pack_count_for_mean_contrast_precision": required_packs,
            "current_primary_mean_delta_ah": battery_primary["mean_delta_ah"],
            "noise_model_status": "PACK_SAMPLING_ONLY_LEDGER_SENSOR_NOISE_UNEVALUABLE",
            "identified_set_effect": "tighten tail-risk transport; ledger remains separately required",
            "priority": "PARETO_NONDOMINATED",
        },
        {
            "candidate_id": "semantic-receiver-sensors",
            "operand": 'receiver groups nominated by the receiver-faithfulness analysis',
            "required_measurement": "retain axis-faithful native groups and clocks",
            "noise_model_status": "CONDITIONAL_ON_NAMED_RECEIVER_AND_PARENT_FLOOR",
            "identified_set_effect": "avoid projection-induced structural collapse",
            "priority": 'CONDITIONAL_ON_RECEIVER_FAITHFULNESS_FRONTIER',
        },
    ]
    return {
        "analysis_id": 'measurement-value-of-information-analysis',
        "evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE.value,
        "value_is_noncompensating_vector": True,
        "scalar_reward_used": False,
        "candidate_measurements": candidates,
        "cost_or_procurement_claimed": False,
        "fresh_confirmation_required": True,
        "disposition": ThermodynamicPosthocDisposition.SUPPORTED.value,
    }


__all__ = [
    "CLAIM_AXES",
    "POSTHOC_ANALYSIS_IDS",
    "ReceiverFamilySpec",
    "ReceiverViewSpec",
    "ThermodynamicPosthocConfig",
    "ThermodynamicPosthocDisposition",
    "ThermodynamicPosthocInputBinding",
    "ThermodynamicPosthocResultIndex",
    "battery_pack_reversal_analysis",
    "claim_monotonicity_analysis",
    "claim_order_violations",
    "classify_claim_axes",
    "complete_unit_bootstrap_interval",
    "complete_unit_max_t_band",
    "decode_posthoc_config",
    "delay_conditioned_cocycle_analysis",
    "exact_sign_flip_test",
    "gym_family_equivalence_rank_analysis",
    "mastu_common_input_analysis",
    "mastu_partial_identification_analysis",
    "measurement_value_of_information_analysis",
    "non_entailment_lattice_analysis",
    "pybamm_curvature_memory_localization_analysis",
    "receiver_faithfulness_analysis",
    "spearman_correlation",
    "state_conditioned_cocycle_analysis",
]

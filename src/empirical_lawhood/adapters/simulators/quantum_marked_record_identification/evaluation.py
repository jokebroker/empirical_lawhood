"""Parent-level compiler-development/coordinate-evaluation scoring, simultaneous inference, and adjudication."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .compiler import ScoreCompiler
from .contracts import Panel, QuantumMarkedRecordIdentificationConfig, Verdict, target_weights
from .controls import active_controls, burst_baseline, prehistory_site_probabilities, reflect_boundaries, shuffle_history, thin_history, validate_control_seed_census
from .features import extract_features
from .receiver import ReceiverOperator, bridge_target, crossfit_site_probabilities, effective_target_sample_size, expected_thinned_receiver, fold_assignments, receiver_operator, site_counts, weighted_midrank_correlation
from .source import EventRecord


OPPORTUNITY_ESTIMANDS = (
    "chi_a",
    "abs_chi_total",
    "abs_cancellation",
    "future_thinning_advantage",
)
RECOGNITION_ESTIMANDS = (
    "rho_matched",
    "rho_burst",
    "rho_count",
    "rho_wrong",
    "burst_advantage",
    "count_increment",
    "history_thinning_advantage",
    "history_shuffle_advantage",
    "wrong_time_advantage",
    "recurrence_absolute_difference",
)


@dataclass(frozen=True, slots=True)
class PrefixScores:
    unit_ids: tuple[str, ...]
    scientific_control_seeds: tuple[tuple[str, int, int], ...]
    selected: NDArray[np.float64]
    count: NDArray[np.float64]
    burst: NDArray[np.float64]
    history_thinned: NDArray[np.float64]
    history_shuffled: NDArray[np.float64]
    wrong_time: NDArray[np.float64]
    boundary_reflected: NDArray[np.float64]
    valid: NDArray[np.bool_]
    maximum_symmetry_sentinel_error: float
    intensity_residual_mean_a: float
    intensity_residual_mean_boundary: float

    def __post_init__(self) -> None:
        validate_control_seed_census(self.unit_ids, self.scientific_control_seeds, allow_resampling=True)

    def commitment_document(self) -> dict[str, object]:
        arrays = {
            "selected": self.selected,
            "count": self.count,
            "burst": self.burst,
            "history_thinned": self.history_thinned,
            "history_shuffled": self.history_shuffled,
            "wrong_time": self.wrong_time,
            "boundary_reflected": self.boundary_reflected,
            "valid": self.valid.astype(np.uint8),
        }
        from hashlib import sha256

        return {
            "schema": 'empirical-lawhood/simulators/quantum-marked-record-identification/prefix-score-commitment',
            "version": '1.0.0',
            "value": {
                "unit_ids": list(self.unit_ids),
                "scientific_control_seeds": [list(row) for row in self.scientific_control_seeds],
                "array_sha256": {
                    name: sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                    for name, value in arrays.items()
                },
                "maximum_symmetry_sentinel_error": self.maximum_symmetry_sentinel_error,
                "intensity_residual_mean_a": self.intensity_residual_mean_a,
                "intensity_residual_mean_boundary": (self.intensity_residual_mean_boundary),
                "future_fields_present": False,
            },
        }


@dataclass(frozen=True, slots=True)
class SimultaneousInterval:
    estimand_id: str
    point: float
    lower: float
    upper: float
    standard_error: float
    critical_value: float
    family: str


@dataclass(frozen=True, slots=True)
class Analysis:
    verdict: Verdict
    operator: ReceiverOperator
    thinned_operator: ReceiverOperator
    point_metrics: Mapping[str, float]
    intervals: tuple[SimultaneousInterval, ...]
    reason_codes: tuple[str, ...]
    parent_count: int
    effective_target_sample_size: float
    active_controls: tuple[str, ...]
    valid_target_weight: float
    leave_one_stratum_out_rho: Mapping[int, float]
    bootstrap_opportunity: NDArray[np.float64]
    bootstrap_recognition: NDArray[np.float64]


def compile_prefix_scores(
    *,
    unit_ids: Sequence[str],
    records: Sequence[Sequence[EventRecord]],
    k_values: Sequence[int],
    folds: Sequence[int],
    selected_compiler: ScoreCompiler,
    count_compiler: ScoreCompiler,
    scientific_control_seeds: tuple[tuple[str, int, int], ...],
    purpose: str,
) -> PrefixScores:
    if not (
        len(unit_ids) == len(records) == len(k_values) == len(folds)
        and selected_compiler.panel in Panel
        and count_compiler.panel is Panel.COUNT
    ):
        raise ValueError("prefix score operands differ")
    validate_control_seed_census(unit_ids, scientific_control_seeds)
    selected_rows = [extract_features(events, selected_compiler.context) for events in records]
    count_rows = [extract_features(events, count_compiler.context) for events in records]
    probabilities = prehistory_site_probabilities(records, k_values, folds)
    thinned_rows = []
    shuffled_rows = []
    wrong_rows = []
    reflected_rows = []
    symmetry_errors = []
    for index, (unit_id, events) in enumerate(zip(unit_ids, records, strict=True)):
        thinned = thin_history(
            events,
            probabilities[index],
            scientific_seed=scientific_control_seeds[index][1],
        )
        shuffled = shuffle_history(events, scientific_seed=scientific_control_seeds[index][2])
        reflected = reflect_boundaries(events)
        thinned_rows.append(extract_features(thinned, selected_compiler.context))
        shuffled_rows.append(extract_features(shuffled, selected_compiler.context))
        wrong_rows.append(extract_features(events, selected_compiler.context, endpoint=196.0))
        reflected_row = extract_features(reflected, selected_compiler.context)
        reflected_rows.append(reflected_row)
        # Only invariant coordinates are exact sentinels.  Reflection mixes
        # the frozen real Fourier pair, and fitted residual cells need not be
        # symmetric.
        invariant = tuple(range(12))
        symmetry_errors.append(
            max(
                abs(reflected_row.values[position] - selected_rows[index].values[position])
                for position in invariant
            )
        )
    selected = selected_compiler.score_rows(selected_rows)
    count = count_compiler.score_rows(count_rows)
    selected_matrix = np.asarray([row.values for row in selected_rows], dtype=np.float64)
    weights = np.asarray(target_weights(list(k_values)), dtype=np.float64)
    return PrefixScores(
        unit_ids=tuple(unit_ids),
        scientific_control_seeds=scientific_control_seeds,
        selected=selected,
        count=count,
        burst=np.asarray([burst_baseline(events) for events in records]),
        history_thinned=selected_compiler.score_rows(thinned_rows),
        history_shuffled=selected_compiler.score_rows(shuffled_rows),
        wrong_time=selected_compiler.score_rows(wrong_rows),
        boundary_reflected=selected_compiler.score_rows(reflected_rows),
        valid=np.asarray(
            np.asarray(
                [
                    row.valid
                    for group in (
                        selected_rows,
                        count_rows,
                        thinned_rows,
                        shuffled_rows,
                        wrong_rows,
                        reflected_rows,
                    )
                    for row in group
                ],
                dtype=np.bool_,
            )
            .reshape(6, len(records))
            .all(axis=0),
            dtype=np.bool_,
        ),
        maximum_symmetry_sentinel_error=float(max(symmetry_errors, default=0.0)),
        intensity_residual_mean_a=float(weights @ selected_matrix[:, 32]),
        intensity_residual_mean_boundary=float(weights @ selected_matrix[:, 34]),
    )


def _metric_vector(
    *,
    unit_ids: Sequence[str],
    k_values: Sequence[int],
    folds: Sequence[int],
    future_counts: NDArray[np.float64],
    scores: PrefixScores,
) -> tuple[dict[str, float], ReceiverOperator, ReceiverOperator]:
    n_a = future_counts[:, :6].sum(axis=1)
    bridge, _donors = bridge_target(n_a, k_values, folds)
    operator = receiver_operator(future_counts, k_values)
    probabilities = crossfit_site_probabilities(future_counts, k_values, folds)
    thinned = expected_thinned_receiver(
        future_counts.sum(axis=1).astype(np.int64),
        probabilities,
        k_values,
    )
    bridge_values = bridge.tolist()
    rho = weighted_midrank_correlation(scores.selected.tolist(), bridge_values, k_values)
    rho_burst = weighted_midrank_correlation(scores.burst.tolist(), bridge_values, k_values)
    rho_count = weighted_midrank_correlation(scores.count.tolist(), bridge_values, k_values)
    rho_thinned = weighted_midrank_correlation(
        scores.history_thinned.tolist(), bridge_values, k_values
    )
    rho_shuffled = weighted_midrank_correlation(
        scores.history_shuffled.tolist(), bridge_values, k_values
    )
    rho_wrong = weighted_midrank_correlation(scores.wrong_time.tolist(), bridge_values, k_values)
    metrics = {
        "chi_a": operator.chi_a,
        "abs_chi_total": abs(operator.chi_total),
        "abs_cancellation": abs(operator.cancellation_residual),
        "future_thinning_advantage": operator.chi_a - thinned.chi_a,
        "rho_matched": rho,
        "rho_burst": rho_burst,
        "rho_count": rho_count,
        "rho_wrong": rho_wrong,
        "burst_advantage": rho - rho_burst,
        "count_increment": rho - rho_count,
        "history_thinning_advantage": rho - rho_thinned,
        "history_shuffle_advantage": rho - rho_shuffled,
        "wrong_time_advantage": rho - rho_wrong,
        "recurrence_absolute_difference": abs(rho - rho_wrong),
        "rho_boundary_reflected": weighted_midrank_correlation(
            scores.boundary_reflected.tolist(), bridge_values, k_values
        ),
    }
    return metrics, operator, thinned


def _selected_scores(scores: PrefixScores, indices: NDArray[np.int64]) -> PrefixScores:
    return PrefixScores(
        unit_ids=tuple(scores.unit_ids[int(index)] for index in indices),
        scientific_control_seeds=tuple(scores.scientific_control_seeds[int(index)] for index in indices),
        selected=scores.selected[indices],
        count=scores.count[indices],
        burst=scores.burst[indices],
        history_thinned=scores.history_thinned[indices],
        history_shuffled=scores.history_shuffled[indices],
        wrong_time=scores.wrong_time[indices],
        boundary_reflected=scores.boundary_reflected[indices],
        valid=scores.valid[indices],
        maximum_symmetry_sentinel_error=scores.maximum_symmetry_sentinel_error,
        intensity_residual_mean_a=scores.intensity_residual_mean_a,
        intensity_residual_mean_boundary=scores.intensity_residual_mean_boundary,
    )


def _bootstrap_indices(
    k_values: NDArray[np.int64],
    scientific_seed: int,
) -> NDArray[np.int64]:
    if type(scientific_seed) is not int or not 0 <= scientific_seed < 2**128:
        raise ValueError("bootstrap seed must be an explicit 128-bit unsigned integer")
    rng = np.random.Generator(np.random.Philox(scientific_seed))
    return np.concatenate(
        [
            rng.choice(
                np.flatnonzero(k_values == k_left),
                size=sum(k_values == k_left),
                replace=True,
            )
            for k_left in range(7)
        ]
    )


def _validate_bootstrap_seeds(seeds: tuple[int, ...], replicates: int) -> None:
    if (
        type(replicates) is not int or replicates < 2
        or not isinstance(seeds, tuple) or len(seeds) != replicates
        or any(type(seed) is not int or not 0 <= seed < 2**128 for seed in seeds)
        or len(set(seeds)) != replicates
    ):
        raise ValueError("bootstrap scientific seed census differs from the replicate roster")




def simultaneous_intervals(
    point: Mapping[str, float],
    bootstrap: NDArray[np.float64],
    *,
    alpha: float,
    family: str,
) -> tuple[SimultaneousInterval, ...]:
    names = tuple(point)
    if bootstrap.shape != (len(bootstrap), len(names)) or len(bootstrap) < 2:
        raise ValueError("bootstrap matrix differs from the estimand family")
    center = np.asarray([point[name] for name in names], dtype=np.float64)
    standard_errors = np.std(bootstrap, axis=0, ddof=1)
    scale = np.maximum(standard_errors, 1.0e-12)
    maxima = np.max(np.abs((bootstrap - center) / scale), axis=1)
    critical = float(np.quantile(maxima, 1.0 - alpha, method="higher"))
    return tuple(
        SimultaneousInterval(
            estimand_id=name,
            point=float(center[index]),
            lower=float(center[index] - critical * scale[index]),
            upper=float(center[index] + critical * scale[index]),
            standard_error=float(standard_errors[index]),
            critical_value=critical,
            family=family,
        )
        for index, name in enumerate(names)
    )


def _interval_map(
    intervals: Sequence[SimultaneousInterval],
) -> dict[str, SimultaneousInterval]:
    return {interval.estimand_id: interval for interval in intervals}


def analyze(
    *,
    unit_ids: Sequence[str],
    k_values: Sequence[int],
    future_records: Sequence[Sequence[EventRecord]],
    scores: PrefixScores,
    selected_panel: Panel,
    config: QuantumMarkedRecordIdentificationConfig,
    bootstrap_replicates: int | None = None,
    scientific_bootstrap_seeds: tuple[int, ...],
    scientific_fold_order: tuple[tuple[str, int], ...],
    scientific_leave_one_out_orders: tuple[tuple[tuple[str, int], ...], ...],
    source_valid: bool = True,
) -> Analysis:
    count = len(unit_ids)
    if not (
        count == len(k_values) == len(future_records) == len(scores.unit_ids)
        and tuple(unit_ids) == scores.unit_ids
    ):
        raise ValueError("analysis rows or prefix commitment identity differ")
    replicates = bootstrap_replicates if bootstrap_replicates is not None else config.bootstrap_replicates
    _validate_bootstrap_seeds(scientific_bootstrap_seeds, replicates)
    folds = fold_assignments(unit_ids, k_values, folds=config.folds, scientific_fold_order=scientific_fold_order)
    if not isinstance(scientific_leave_one_out_orders, tuple) or len(scientific_leave_one_out_orders) != 7:
        raise ValueError("scientific leave-one-stratum-out fold census differs")
    diagnostic_folds = tuple(
        fold_assignments(
            tuple(unit for unit, k in zip(unit_ids, k_values, strict=True) if k != omitted),
            tuple(k for k in k_values if k != omitted),
            folds=config.folds,
            scientific_fold_order=scientific_leave_one_out_orders[omitted],
        )
        for omitted in range(7)
    )
    future_counts = np.asarray(
        [site_counts(events, start_time=200.0, end_time=204.0) for events in future_records],
        dtype=np.float64,
    )
    point, operator, thinned = _metric_vector(
        unit_ids=unit_ids,
        k_values=k_values,
        folds=folds.tolist(),
        future_counts=future_counts,
        scores=scores,
    )
    opportunity = np.empty((replicates, len(OPPORTUNITY_ESTIMANDS)), dtype=np.float64)
    recognition = np.empty((replicates, len(RECOGNITION_ESTIMANDS)), dtype=np.float64)
    strata = np.asarray(k_values, dtype=np.int64)
    for replicate in range(replicates):
        indices = _bootstrap_indices(strata, scientific_bootstrap_seeds[replicate])
        metrics, _operator, _thinned = _metric_vector(
            unit_ids=[unit_ids[int(index)] for index in indices],
            k_values=[k_values[int(index)] for index in indices],
            folds=folds[indices].tolist(),
            future_counts=future_counts[indices],
            scores=_selected_scores(scores, indices),
        )
        opportunity[replicate] = [metrics[name] for name in OPPORTUNITY_ESTIMANDS]
        recognition[replicate] = [metrics[name] for name in RECOGNITION_ESTIMANDS]
    intervals = simultaneous_intervals(
        {name: point[name] for name in OPPORTUNITY_ESTIMANDS},
        opportunity,
        alpha=float(config.family_alpha),
        family="opportunity",
    ) + simultaneous_intervals(
        {name: point[name] for name in RECOGNITION_ESTIMANDS},
        recognition,
        alpha=float(config.family_alpha),
        family="recognition",
    )
    by_id = _interval_map(intervals)
    valid_weight = float(
        np.asarray(target_weights(list(k_values)), dtype=np.float64) @ scores.valid
    )
    leave_one_out: dict[int, float] = {}
    n_a = future_counts[:, :6].sum(axis=1)
    for omitted in range(7):
        keep = np.flatnonzero(strata != omitted)
        local_folds = diagnostic_folds[omitted]
        bridge, _ = bridge_target(
            n_a[keep],
            [k_values[int(index)] for index in keep],
            local_folds.tolist(),
        )
        # Exact target weighting is renormalized over the retained strata for
        # this stability diagnostic.
        local_k = [k_values[int(index)] for index in keep]
        leave_one_out[omitted] = weighted_midrank_correlation(
            scores.selected[keep].tolist(),
            bridge.tolist(),
            local_k,
            require_all_strata=False,
        )
    reasons: list[str] = []
    source_invalid = (
        not source_valid
        or valid_weight < float(config.valid_weight_floor)
        or scores.maximum_symmetry_sentinel_error > 1.0e-10
        or not all(
            all(200.0 <= event.event_time < 204.0 for event in events) for events in future_records
        )
    )
    opportunity_failed = by_id["chi_a"].lower < float(config.chi_gate)
    controls_failed = (
        by_id["abs_chi_total"].upper > float(config.global_equivalence)
        or by_id["abs_cancellation"].upper > float(config.cancellation_equivalence)
        or by_id["future_thinning_advantage"].lower < float(config.future_thinning_advantage)
    )
    recognition_failed = by_id["rho_matched"].lower < float(config.association_gate) or by_id[
        "burst_advantage"
    ].lower < float(config.burst_advantage)
    specificity_failed = False
    if selected_panel is not Panel.COUNT:
        specificity_failed = by_id["count_increment"].lower < float(config.count_increment)
    if "USES_SPATIAL_MARKS" in selected_panel.semantic_flags:
        specificity_failed = specificity_failed or (
            by_id["history_thinning_advantage"].lower < float(config.semantic_advantage)
        )
    if "USES_ORDER" in selected_panel.semantic_flags:
        specificity_failed = specificity_failed or (
            by_id["history_shuffle_advantage"].lower < float(config.semantic_advantage)
        )
    phase_local = by_id["wrong_time_advantage"].lower >= float(config.semantic_advantage)
    recurrent = (
        by_id["rho_matched"].lower >= float(config.association_gate)
        and by_id["rho_wrong"].lower >= float(config.association_gate)
        and by_id["recurrence_absolute_difference"].upper <= float(config.recurrence_equivalence)
    )
    if "USES_RECENCY" in selected_panel.semantic_flags and not (phase_local or recurrent):
        specificity_failed = True
    if "USES_FITTED_INTENSITY" in selected_panel.semantic_flags:
        specificity_failed = specificity_failed or (
            abs(scores.intensity_residual_mean_a) > 0.10
            or abs(scores.intensity_residual_mean_boundary) > 0.10
        )
    if any(value < 0 for value in leave_one_out.values()):
        specificity_failed = True
        reasons.append("leave-one-stratum-out-sign-reversal")
    if source_invalid:
        verdict = Verdict.COORDINATE_EVALUATION_SOURCE_OR_OBSERVATION_STOP
        reasons.append("source-cutoff-feature-or-symmetry-validity-failed")
    elif opportunity_failed:
        verdict = Verdict.COORDINATE_EVALUATION_RECEIVER_ORDER_NOT_REPLICATED
        reasons.append("future-receiver-opportunity-not-replicated")
    elif controls_failed:
        verdict = Verdict.COORDINATE_EVALUATION_RECEIVER_CONTROL_RELATION_OPPOSED
        reasons.append("global-cancellation-or-thinning-control-failed")
    elif recognition_failed:
        verdict = Verdict.COORDINATE_EVALUATION_RECEIVER_ORDER_WITHOUT_PREDICTIVE_COORDINATE
        reasons.append("matched-association-or-burst-increment-failed")
    elif specificity_failed:
        verdict = Verdict.COORDINATE_EVALUATION_COORDINATE_SPECIFICITY_STOP
        reasons.append("active-semantic-or-complexity-control-failed")
    elif selected_panel is Panel.COUNT:
        verdict = Verdict.COORDINATE_EVALUATION_COUNT_ONLY_COORDINATE_SUPPORTED
    elif recurrent:
        verdict = Verdict.COORDINATE_EVALUATION_TIME_TRANSLATION_RECURRENT_COORDINATE_SUPPORTED
    else:
        verdict = Verdict.COORDINATE_EVALUATION_PHASE_LOCAL_RECORD_COORDINATE_SUPPORTED
    return Analysis(
        verdict=verdict,
        operator=operator,
        thinned_operator=thinned,
        point_metrics=point,
        intervals=intervals,
        reason_codes=tuple(dict.fromkeys(reasons)),
        parent_count=count,
        effective_target_sample_size=effective_target_sample_size(k_values),
        active_controls=active_controls(selected_panel),
        valid_target_weight=valid_weight,
        leave_one_stratum_out_rho=leave_one_out,
        bootstrap_opportunity=opportunity,
        bootstrap_recognition=recognition,
    )


def precision_fields(analysis: Analysis) -> dict[str, object]:
    widths = {
        interval.estimand_id: interval.upper - interval.lower for interval in analysis.intervals
    }
    return {
        "parent_count": analysis.parent_count,
        "effective_target_sample_size": analysis.effective_target_sample_size,
        "valid_target_weight": analysis.valid_target_weight,
        "unsigned_interval_widths": widths,
        "maximum_opportunity_width": max(widths[name] for name in OPPORTUNITY_ESTIMANDS),
        "maximum_recognition_width": max(widths[name] for name in RECOGNITION_ESTIMANDS),
    }


def precision_expansion_required(analysis: Analysis) -> bool:
    fields = precision_fields(analysis)
    return bool(
        cast(float, fields["maximum_opportunity_width"]) > 0.50
        or cast(float, fields["maximum_recognition_width"]) > 0.20
    )


__all__ = [
    "Analysis",
    "OPPORTUNITY_ESTIMANDS",
    "PrefixScores",
    "RECOGNITION_ESTIMANDS",
    "SimultaneousInterval",
    "analyze",
    "compile_prefix_scores",
    "precision_expansion_required",
    "precision_fields",
    "simultaneous_intervals",
]

"""Noncompensating semantic, reference, path, ensemble and evaluation adjudication."""

from __future__ import annotations

from typing import Mapping, Sequence

from .schemas import QuantumTrajectoryReferenceValidationConfig
from .types import Denominator, FixtureRow, Stage, Validity, Verdict


def _number(
    row: Mapping[str, object],
    key: str,
    default: float = float("inf"),
) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def adjudicate_conformance(
    rows: Sequence[FixtureRow],
    config: QuantumTrajectoryReferenceValidationConfig,
) -> dict[str, object]:
    failures: list[dict[str, object]] = []
    expected_zero = {"unoccupied-basis-site", "zero-mass-prng-guard"}
    for row in rows:
        reasons: list[str] = []
        expected_validity = (
            Validity.ZERO_PROJECTED_MASS if row.fixture_id in expected_zero else Validity.VALID
        )
        if row.validity != expected_validity:
            reasons.append("UNEXPECTED_VALIDITY")
        if (
            row.projected_mass_identity_residual is not None
            and row.projected_mass_identity_residual > float(config.projected_mass_tolerance)
        ):
            reasons.append("PROJECTED_MASS_IDENTITY")
        if row.mark_sum_residual is not None and row.mark_sum_residual > float(
            config.mark_probability_sum_tolerance
        ):
            reasons.append("MARK_SUM")
        if row.propagator_norm_residual is not None and row.propagator_norm_residual > float(
            config.propagator_norm_tolerance
        ):
            reasons.append("PROPAGATOR_NORM")
        if row.post_jump_norm_residual is not None and row.post_jump_norm_residual > float(
            config.post_jump_norm_tolerance
        ):
            reasons.append("POST_JUMP_NORM")
        if row.phase_gauge_infidelity is not None and row.phase_gauge_infidelity > float(
            config.phase_infidelity_tolerance
        ):
            reasons.append("PHASE_GAUGE")
        if (
            row.expected_projected_mass is not None
            and row.observed_projected_mass_expectation is not None
            and abs(row.expected_projected_mass - row.observed_projected_mass_expectation)
            > float(config.projected_mass_tolerance)
        ):
            reasons.append("ANALYTIC_PROJECTED_MASS")
        if row.fixture_id == "equal-cumulative-boundary" and row.site != 1:
            reasons.append("CUMULATIVE_BOUNDARY_RULE")
        if row.fixture_id == "zero-mass-prng-guard" and (
            row.prng_counter_before != row.prng_counter_after
        ):
            reasons.append("ZERO_MASS_CONSUMED_PRNG")
        if row.fixture_id == "action-switch-right-endpoint" and not (
            row.requested_action == "plus"
            and row.accepted_action == "plus"
            and row.applied_action == "plus"
            and row.realized_action_start == 1.0
            and row.realized_action_end == 2.0
            and row.endpoint_included is True
        ):
            reasons.append("ACTION_ENDPOINT_ORDER")
        if reasons:
            failures.append(
                {
                    "fixture_id": row.fixture_id,
                    "view": row.view.value,
                    "reasons": reasons,
                }
            )
    passed = not failures
    return {
        "schema": 'empirical-lawhood/simulators/quantum-trajectory-reference-validation/numerical-semantics-assessment',
        "version": '1.0.0',
        "fixture_rows": len(rows),
        "fixture_families": len({row.fixture_id for row in rows}),
        "passed": passed,
        "verdict": (Verdict.SEMANTIC_VALID.value if passed else Verdict.SEMANTIC_STOP.value),
        "failures": failures,
    }


def _path_pass(row: Mapping[str, object], config: QuantumTrajectoryReferenceValidationConfig) -> bool:
    return bool(
        row.get("event_count_match") is True
        and row.get("mark_mismatches") == 0
        and _number(row, "maximum_event_clock_difference") <= float(config.event_clock_tolerance)
        and _number(row, "phase_gauge_infidelity") <= float(config.phase_infidelity_tolerance)
        and _number(row, "maximum_site_probability_difference")
        <= float(config.site_probability_tolerance)
        and _number(row, "maximum_sparse_propagator_norm_residual")
        <= float(config.propagator_norm_tolerance)
        and _number(row, "maximum_dense_propagator_norm_residual")
        <= float(config.propagator_norm_tolerance)
        and _number(row, "maximum_sparse_post_jump_norm_residual")
        <= float(config.post_jump_norm_tolerance)
        and _number(row, "maximum_dense_post_jump_norm_residual")
        <= float(config.post_jump_norm_tolerance)
        and _number(row, "maximum_sparse_projected_mass_identity_residual")
        <= float(config.projected_mass_tolerance)
        and _number(row, "maximum_dense_projected_mass_identity_residual")
        <= float(config.projected_mass_tolerance)
        and _number(row, "maximum_sparse_mark_probability_sum_residual")
        <= float(config.mark_probability_sum_tolerance)
        and _number(row, "maximum_dense_mark_probability_sum_residual")
        <= float(config.mark_probability_sum_tolerance)
        and _number(row, "maximum_sparse_particle_number_residual")
        <= float(config.particle_number_tolerance)
        and _number(row, "maximum_dense_particle_number_residual")
        <= float(config.particle_number_tolerance)
    )


def adjudicate_reference(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    denominators: dict[str, dict[str, object]] = {}
    for denominator in Denominator:
        selected = [row for row in rows if row.get("denominator") == denominator.value]
        failures = [row for row in selected if row.get("passed") is not True]
        denominators[denominator.value] = {
            "cell_count": len(selected),
            "failure_count": len(failures),
            "passed": len(selected) == 4 and not failures,
        }
    eligible = [
        denominator.value
        for denominator in Denominator
        if denominators[denominator.value]["passed"] is True
    ]
    return {
        "schema": 'empirical-lawhood/simulators/quantum-trajectory-reference-validation/deterministic-reference-assessment',
        "version": '1.0.0',
        "denominators": denominators,
        "eligible_denominators": eligible,
        "passed_any": bool(eligible),
        "verdict": ("DETERMINISTIC_REFERENCE_QUALIFIED" if eligible else Verdict.REFERENCE_STOP.value),
    }


def adjudicate_path_source(
    *,
    structural: Sequence[Mapping[str, object]],
    paths: Sequence[Mapping[str, object]],
    restarts: Sequence[Mapping[str, object]],
    point_process: Sequence[Mapping[str, object]],
    config: QuantumTrajectoryReferenceValidationConfig,
    reference_eligible: Sequence[str],
) -> dict[str, object]:
    structural_pass = bool(structural and all(row.get("passed") is True for row in structural))
    denominator_rows: dict[str, dict[str, object]] = {}
    for denominator in Denominator:
        path_rows = [row for row in paths if row.get("denominator") == denominator.value]
        restart_rows = [row for row in restarts if row.get("denominator") == denominator.value]
        point_rows = [row for row in point_process if row.get("denominator") == denominator.value]
        path_failures = [row for row in path_rows if not _path_pass(row, config)]
        restart_failures = [
            row
            for row in restart_rows
            if not (
                row.get("event_replay_exact") is True
                and _number(row, "phase_gauge_infidelity")
                <= float(config.phase_infidelity_tolerance)
                and row.get("rng_draw_count_reference") == row.get("rng_draw_count_restart")
            )
        ]
        point_failures = [row for row in point_rows if row.get("passed") is not True]
        expected_paths = 2 * 32 * 3
        expected_restarts = 32 * 3
        passed = bool(
            denominator.value in reference_eligible
            and structural_pass
            and len(path_rows) == expected_paths
            and len(restart_rows) == expected_restarts
            and len(point_rows) == 1
            and not path_failures
            and not restart_failures
            and not point_failures
        )
        denominator_rows[denominator.value] = {
            "passed": passed,
            "path_rows": len(path_rows),
            "path_failures": len(path_failures),
            "restart_rows": len(restart_rows),
            "restart_failures": len(restart_failures),
            "reference_eligible": denominator.value in reference_eligible,
            "point_process_rows": len(point_rows),
            "point_process_failures": len(point_failures),
            "maximum_phase_gauge_infidelity": max(
                (
                    _number(row, "phase_gauge_infidelity")
                    for row in path_rows
                    if "phase_gauge_infidelity" in row
                ),
                default=None,
            ),
            "maximum_site_probability_difference": max(
                (
                    _number(row, "maximum_site_probability_difference")
                    for row in path_rows
                    if "maximum_site_probability_difference" in row
                ),
                default=None,
            ),
            "maximum_post_jump_norm_residual": max(
                (
                    max(
                        _number(row, "maximum_sparse_post_jump_norm_residual"),
                        _number(row, "maximum_dense_post_jump_norm_residual"),
                    )
                    for row in path_rows
                ),
                default=None,
            ),
        }
    eligible = [
        denominator.value
        for denominator in Denominator
        if denominator_rows[denominator.value]["passed"] is True
    ]
    return {
        "schema": 'empirical-lawhood/simulators/quantum-trajectory-reference-validation/pathwise-source-assessment',
        "version": '1.0.0',
        "structural_pass": structural_pass,
        "denominators": denominator_rows,
        "eligible_denominators": eligible,
        "passed_any": bool(eligible),
        "verdict": ("PATH_SOURCE_QUALIFIED" if eligible else Verdict.PATH_SOURCE_STOP.value),
    }


def adjudicate_evaluation(
    *,
    evaluation_analysis: Mapping[str, object],
    source_result: Mapping[str, object],
) -> dict[str, object]:
    denominator_pass_raw = evaluation_analysis.get("denominator_pass")
    if not isinstance(denominator_pass_raw, Mapping):
        return {
            "verdict": Verdict.EVALUATION_UNEVALUABLE.value,
            "qualified_denominators": [],
            "handoff": "NO_QUALIFICATION",
        }
    source_denominators = source_result.get("denominators")
    if not isinstance(source_denominators, Mapping):
        return {
            "verdict": Verdict.EVALUATION_UNEVALUABLE.value,
            "qualified_denominators": [],
            "handoff": "NO_QUALIFICATION",
        }
    qualified = [
        denominator.value
        for denominator in Denominator
        if denominator_pass_raw.get(denominator.value) is True
        and isinstance(source_denominators.get(denominator.value), Mapping)
        and source_denominators[denominator.value].get("passed") is True
    ]
    if Denominator.STRONG.value in qualified and Denominator.WEAK.value in qualified:
        verdict = Verdict.BOTH_QUALIFIED
        handoff = "RECEIVER_RESPONSE_BOTH_DENOMINATORS_PLANNING_ELIGIBLE"
    elif Denominator.STRONG.value in qualified:
        verdict = Verdict.STRONG_ONLY
        handoff = "RECEIVER_RESPONSE_STRONG_ONLY_PLANNING_ELIGIBLE"
    elif Denominator.WEAK.value in qualified:
        verdict = Verdict.WEAK_ONLY
        handoff = "QUANTUM_STRONG_SOURCE_REDESIGN_REQUIRED"
    else:
        intervals = evaluation_analysis.get("intervals")
        memory = evaluation_analysis.get("memory")
        memory_material = bool(
            isinstance(memory, list)
            and any(isinstance(row, Mapping) and row.get("material") is True for row in memory)
        )
        unevaluable = bool(
            isinstance(intervals, list)
            and any(
                isinstance(row, Mapping) and row.get("validity") == Validity.UNEVALUABLE.value
                for row in intervals
            )
        )
        if unevaluable:
            verdict = Verdict.EVALUATION_UNEVALUABLE
            handoff = "NO_QUALIFICATION"
        elif memory_material:
            verdict = Verdict.PREPARATION_MEMORY_STOP
            handoff = "QUANTUM_HISTORY_OR_PREPARATION_TYPING_REQUIRED"
        else:
            verdict = Verdict.FRESH_STATIONARITY_STOP
            handoff = "QUANTUM_PREPARATION_REDESIGN_REQUIRED"
    return {
        "schema": 'empirical-lawhood/simulators/quantum-trajectory-reference-validation/fresh-stationarity-assessment',
        "version": '1.0.0',
        "qualified_denominators": qualified,
        "verdict": verdict.value,
        "handoff": handoff,
        "maximum_claim": (
            "DIRECT_FINITE_SIMULATOR_LOCAL_SOURCE_QUALIFICATION_QUANTUM_TRAJECTORY_REFERENCE_VALIDATION"
            if qualified
            else "TESTED_BUT_NOT_QUALIFIED"
        ),
        "receiver_response_and_control": "NOT_ENTERED",
    }


def formal_gap_dispositions(
    *,
    issued_stages: Sequence[str],
    source_result: Mapping[str, object] | None,
    evaluation_result: Mapping[str, object] | None,
) -> list[dict[str, object]]:
    source_conformance_passed = bool(source_result is not None and source_result.get("passed_any") is True)
    qualified = bool(
        evaluation_result is not None and evaluation_result.get("qualified_denominators")
    )
    rows: list[dict[str, object]] = [
        {
            "gap_role": "sparse-evolution",
            "disposition": (
                "DIRECT_FINITE_SIMULATOR_LOCAL_QUANTUM_TRAJECTORY_REFERENCE_VALIDATION" if source_conformance_passed else "TESTED_BUT_NOT_QUALIFIED"
            ),
        },
        {
            "gap_role": "numerical-view-convergence",
            "disposition": (
                "DIRECT_FINITE_SIMULATOR_LOCAL_QUANTUM_TRAJECTORY_REFERENCE_VALIDATION" if source_conformance_passed else "TESTED_BUT_NOT_QUALIFIED"
            ),
        },
        {
            "gap_role": "recurrence-stationarity",
            "disposition": (
                "DIRECT_FINITE_SELECTED_SENTINEL_QUANTUM_TRAJECTORY_REFERENCE_VALIDATION"
                if qualified
                else (
                    "TESTED_BUT_NOT_QUALIFIED"
                    if Stage.EVALUATION.value in issued_stages
                    else "NOT_ATTEMPTED_PREREQUISITE"
                )
            ),
        },
        {
            "gap_role": "state-closure-memory",
            "disposition": (
                "BOUNDED_PREPARATION_MEMORY_NONDETECTION_QUANTUM_TRAJECTORY_REFERENCE_VALIDATION"
                if qualified
                else (
                    "TESTED_BUT_NOT_QUALIFIED"
                    if Stage.EVALUATION.value in issued_stages
                    else "NOT_ATTEMPTED_PREREQUISITE"
                )
            ),
        },
    ]
    return rows


__all__ = [
    "adjudicate_conformance",
    "adjudicate_evaluation",
    "adjudicate_path_source",
    "adjudicate_reference",
    "formal_gap_dispositions",
]

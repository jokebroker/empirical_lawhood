"""Complete-unit T/U phase-diagram inference for split cohort history budget."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

import numpy as np
from scipy.stats import beta

from empirical_lawhood.adapters.split_cohort_history_budget.contracts import SplitCohortHistoryBudgetAlignmentSummary, SplitCohortHistoryBudgetCellRecurrence, SplitCohortHistoryBudgetConfig, SplitCohortHistoryBudgetCoordinateKind, SplitCohortHistoryBudgetDisorderFamily, SplitCohortHistoryBudgetMethodFreeze, SplitCohortHistoryBudgetPhase, SplitCohortHistoryBudgetRecurrenceResult, SplitCohortHistoryBudgetRankCompatibility, SplitCohortHistoryBudgetRankObjectSummary, SplitCohortHistoryBudgetResolutionPanelSummary, SplitCohortHistoryBudgetScientificState, SplitCohortHistoryBudgetTargetedCoordinateAdjudication, SplitCohortHistoryBudgetTransitionSummary, SplitCohortHistoryBudgetTContinuationGateRecord, SplitCohortHistoryBudgetUntouchedPrevalenceCell, SplitCohortHistoryBudgetUntouchedDescriptiveSummary, SplitCohortHistoryBudgetUnitAdjudication
from empirical_lawhood.adapters.split_cohort_history_budget.descriptors import evaluation_unit_ids
from empirical_lawhood.adapters.split_cohort_history_budget.runtime_contracts import SplitCohortHistoryBudgetAdjudicationBundle, SplitCohortHistoryBudgetBootstrapCellSummary, SplitCohortHistoryBudgetBootstrapInterval, SplitCohortHistoryBudgetBootstrapSummary
from empirical_lawhood.kernel.evidence import EvidenceCeiling

from .alignment import paired_alignment


@dataclass(frozen=True, slots=True)
class RecurrenceExecution:
    result: SplitCohortHistoryBudgetRecurrenceResult
    bootstrap_summary: SplitCohortHistoryBudgetBootstrapSummary


_STATE_CODE = {
    SplitCohortHistoryBudgetScientificState.OPPOSED: 0,
    SplitCohortHistoryBudgetScientificState.INFORMATIVE_NONADVERSE: 1,
    SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED: 2,
}
_FAMILY_CODE = {value: index for index, value in enumerate(SplitCohortHistoryBudgetDisorderFamily)}


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("split cohort history budget inference output must be finite")
    return Decimal(str(float(value)))


def _required_count(value: int | None) -> int:
    if value is None:
        raise ValueError("valid split cohort history budget evidence row lacks its count")
    return value


def _exact_interval(successes: int, requested: int, alpha_tail: float) -> tuple[Decimal, Decimal]:
    if not 0 <= successes <= requested or requested <= 0 or not 0.0 < alpha_tail < 1.0:
        raise ValueError("split cohort history budget exact interval inputs differ")
    lower = (
        0.0 if successes == 0 else float(beta.ppf(alpha_tail, successes, requested - successes + 1))
    )
    upper = (
        1.0
        if successes == requested
        else float(beta.ppf(1.0 - alpha_tail, successes + 1, requested - successes))
    )
    return _decimal(lower), _decimal(upper)


def _interval(values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> SplitCohortHistoryBudgetBootstrapInterval:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return SplitCohortHistoryBudgetBootstrapInterval(lower=_decimal(float(lower)), upper=_decimal(float(upper)))


def _conditional_bootstrap_mean(
    values: np.ndarray[tuple[int], np.dtype[np.float64]],
    draws: np.ndarray[tuple[int, int], np.dtype[np.int64]],
) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    sampled = values[draws]
    finite = np.isfinite(sampled)
    counts = np.sum(finite, axis=1)
    sums = np.sum(np.where(finite, sampled, 0.0), axis=1)
    return np.asarray(sums[counts > 0] / counts[counts > 0], dtype=np.float64)


def _bootstrap_summary(
    config: SplitCohortHistoryBudgetConfig,
    method_freeze: SplitCohortHistoryBudgetMethodFreeze,
    adjudications: tuple[SplitCohortHistoryBudgetUnitAdjudication, ...],
) -> SplitCohortHistoryBudgetBootstrapSummary:
    """Whole-unit rank summaries; nested scale/depth rows are never resampled."""

    rng = np.random.Generator(np.random.PCG64(config.bootstrap_seed))
    cell_summaries: list[SplitCohortHistoryBudgetBootstrapCellSummary] = []
    for family in SplitCohortHistoryBudgetDisorderFamily:
        family_rows = tuple(value for value in adjudications if value.family is family)
        unit_ids = tuple(sorted({value.unit_id for value in family_rows}))
        by_key = {(value.unit_id, value.scale_cells): value for value in family_rows}
        if len(unit_ids) != 30:
            raise ValueError("split cohort history budget bootstrap requires 30 complete seed blocks per family")
        draws = rng.integers(0, len(unit_ids), size=(config.bootstrap_resamples, len(unit_ids)))
        for scale in (64, 128, 256):
            rows = tuple(by_key[(unit_id, scale)] for unit_id in unit_ids)
            rank_distance = np.asarray([float(value.rank_curve_distance) for value in rows])
            sampled_distance = np.mean(rank_distance[draws], axis=1)
            k_values = np.asarray(
                [
                    np.nan if value.k_full_effective is None else value.k_full_effective
                    for value in rows
                ],
                dtype=np.float64,
            )
            b_values = np.asarray(
                [np.nan if value.b_full is None else float(value.b_full) for value in rows],
                dtype=np.float64,
            )
            finite_k = np.isfinite(k_values)
            finite_b = np.isfinite(b_values)
            if not np.array_equal(finite_k, finite_b):
                raise ValueError("split cohort history budget k_full and b_full evaluability differs")
            k_bootstrap = _conditional_bootstrap_mean(k_values, draws)
            b_bootstrap = _conditional_bootstrap_mean(b_values, draws)
            cell_summaries.append(
                SplitCohortHistoryBudgetBootstrapCellSummary(
                    cell_id=f"bootstrap-cell.{family.value}.n{scale}",
                    family=family,
                    scale_cells=scale,
                    requested_unit_count=30,
                    evaluable_k_full_count=int(np.sum(finite_k)),
                    k_full_mean=None
                    if not np.any(finite_k)
                    else _decimal(float(np.mean(k_values[finite_k]))),
                    k_full_bootstrap_95=None if k_bootstrap.size == 0 else _interval(k_bootstrap),
                    k_full_usable_resample_count=int(k_bootstrap.size),
                    b_full_mean=None
                    if not np.any(finite_b)
                    else _decimal(float(np.mean(b_values[finite_b]))),
                    b_full_bootstrap_95=None if b_bootstrap.size == 0 else _interval(b_bootstrap),
                    b_full_usable_resample_count=int(b_bootstrap.size),
                    rank_curve_distance_mean=_decimal(float(np.mean(rank_distance))),
                    rank_curve_distance_bootstrap_95=_interval(sampled_distance),
                )
            )
    return SplitCohortHistoryBudgetBootstrapSummary(
        summary_id="split-cohort-history-budget.bootstrap-summary",
        evaluation_config_sha256=config.fingerprint(),
        method_freeze_sha256=method_freeze.fingerprint(),
        cells=tuple(sorted(cell_summaries, key=lambda value: value.cell_id)),
        resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
        confidence_level=Decimal("0.95"),
        whole_seed_block_resampling=True,
        nested_scale_action_mode_resampling=False,
    )


def _coordinate_label(row: SplitCohortHistoryBudgetTargetedCoordinateAdjudication) -> str:
    if row.coordinate_kind is SplitCohortHistoryBudgetCoordinateKind.ABSOLUTE_DEPTH:
        return f"k{row.depth}"
    assert row.budget is not None
    return f"b{str(row.budget).replace('.', 'p')}"


def _full_coordinate_label(row: SplitCohortHistoryBudgetTargetedCoordinateAdjudication) -> str:
    label = _coordinate_label(row)
    if (
        row.coordinate_kind is SplitCohortHistoryBudgetCoordinateKind.ABSOLUTE_DEPTH
        and row.resolution_epsilon != Decimal("0.005")
    ):
        return f"{label}-eps{str(row.resolution_epsilon).replace('.', 'p')}"
    return label


def _targeted_cells(
    config: SplitCohortHistoryBudgetConfig,
    rows: tuple[SplitCohortHistoryBudgetTargetedCoordinateAdjudication, ...],
) -> tuple[SplitCohortHistoryBudgetCellRecurrence, ...]:
    alpha_tail = float(config.simultaneous_alpha) / 2.0 / 72.0
    cells: list[SplitCohortHistoryBudgetCellRecurrence] = []
    for family in SplitCohortHistoryBudgetDisorderFamily:
        for scale in (64, 128, 256):
            selected = tuple(
                value
                for value in rows
                if value.family is family and value.scale_cells == scale and value.primary
            )
            labels = tuple(sorted({_coordinate_label(value) for value in selected}))
            if labels != ("b0p5", "b1", "k4", "k8"):
                raise ValueError("split cohort history budget primary T coordinate roster differs")
            for label in labels:
                coordinate_rows = tuple(
                    value for value in selected if _coordinate_label(value) == label
                )
                if len(coordinate_rows) != 30:
                    raise ValueError("split cohort history budget T cell lost its requested denominator")
                exemplar = coordinate_rows[0]
                for endpoint_id, attribute in (
                    ("decision-closure", "decision_state"),
                    ("dynamical-closure", "dynamical_state"),
                ):
                    states = tuple(getattr(value, attribute) for value in coordinate_rows)
                    invalid = sum(
                        (not value.valid) or not bool(value.generator_observer_agreement)
                        for value in coordinate_rows
                    )
                    opposed = sum(
                        value is SplitCohortHistoryBudgetScientificState.OPPOSED for value in states
                    ) - sum(
                        value is SplitCohortHistoryBudgetScientificState.OPPOSED
                        and ((not row.valid) or not bool(row.generator_observer_agreement))
                        for value, row in zip(states, coordinate_rows)
                    )
                    informative = sum(
                        value is SplitCohortHistoryBudgetScientificState.INFORMATIVE_NONADVERSE
                        and row.valid
                        and bool(row.generator_observer_agreement)
                        for value, row in zip(states, coordinate_rows)
                    )
                    limited = 30 - invalid - opposed - informative
                    unsafe_units = sum(
                        bool(row.unsafe_false_promotion_count)
                        for row in coordinate_rows
                        if row.valid and bool(row.generator_observer_agreement)
                    )
                    false_hold_units = sum(
                        bool(row.false_hold_count)
                        for row in coordinate_rows
                        if row.valid and bool(row.generator_observer_agreement)
                    )
                    opposed_interval = _exact_interval(opposed, 30, alpha_tail)
                    informative_interval = _exact_interval(informative, 30, alpha_tail)
                    cells.append(
                        SplitCohortHistoryBudgetCellRecurrence(
                            cell_id=f"cell.{family.value}.n{scale}.{label}.{endpoint_id}",
                            family=family,
                            scale_cells=scale,
                            coordinate_label=label,
                            coordinate_kind=exemplar.coordinate_kind,
                            depth=exemplar.depth,
                            budget=exemplar.budget,
                            endpoint_id=endpoint_id,
                            requested_count=30,
                            invalid_count=invalid,
                            opposed_count=opposed,
                            informative_nonadverse_count=informative,
                            targetability_limited_count=limited,
                            unsafe_false_promotion_unit_count=unsafe_units,
                            false_hold_unit_count=false_hold_units,
                            opposition_lower_bound=opposed_interval[0],
                            opposition_upper_bound=opposed_interval[1],
                            informative_nonadverse_lower_bound=informative_interval[0],
                            informative_nonadverse_upper_bound=informative_interval[1],
                            existence_recurs=opposed > 0,
                            majority_opposition=opposed_interval[0] > Decimal("0.5"),
                            majority_informative_nonadverse=informative_interval[0]
                            > Decimal("0.5"),
                        )
                    )
    return tuple(sorted(cells, key=lambda value: value.cell_id))


def _secondary_targeted_cells(
    rows: tuple[SplitCohortHistoryBudgetTargetedCoordinateAdjudication, ...],
) -> tuple[SplitCohortHistoryBudgetCellRecurrence, ...]:
    cells: list[SplitCohortHistoryBudgetCellRecurrence] = []
    for family in SplitCohortHistoryBudgetDisorderFamily:
        for scale in (64, 128, 256):
            selected = tuple(
                value
                for value in rows
                if value.family is family and value.scale_cells == scale and not value.primary
            )
            coordinate_ids = tuple(sorted({value.coordinate_id for value in selected}))
            if len(coordinate_ids) != 12:
                raise ValueError("split cohort history budget secondary T coordinate roster differs")
            for coordinate_id in coordinate_ids:
                coordinate_rows = tuple(
                    value for value in selected if value.coordinate_id == coordinate_id
                )
                if len(coordinate_rows) != 30:
                    raise ValueError("split cohort history budget secondary T denominator differs")
                exemplar = coordinate_rows[0]
                label = _full_coordinate_label(exemplar)
                for endpoint_id, attribute in (
                    ("decision-closure", "decision_state"),
                    ("dynamical-closure", "dynamical_state"),
                ):
                    valid_rows = tuple(
                        value
                        for value in coordinate_rows
                        if value.valid and bool(value.generator_observer_agreement)
                    )
                    states = tuple(getattr(value, attribute) for value in valid_rows)
                    invalid = 30 - len(valid_rows)
                    opposed = sum(value is SplitCohortHistoryBudgetScientificState.OPPOSED for value in states)
                    informative = sum(
                        value is SplitCohortHistoryBudgetScientificState.INFORMATIVE_NONADVERSE for value in states
                    )
                    limited = 30 - invalid - opposed - informative
                    unsafe_units = sum(bool(row.unsafe_false_promotion_count) for row in valid_rows)
                    false_hold_units = sum(bool(row.false_hold_count) for row in valid_rows)
                    opposed_interval = _exact_interval(opposed, 30, 0.025)
                    informative_interval = _exact_interval(informative, 30, 0.025)
                    cells.append(
                        SplitCohortHistoryBudgetCellRecurrence(
                            cell_id=(
                                f"secondary-cell.{family.value}.n{scale}.{label}.{endpoint_id}"
                            ),
                            family=family,
                            scale_cells=scale,
                            coordinate_label=label,
                            coordinate_kind=exemplar.coordinate_kind,
                            depth=exemplar.depth,
                            budget=exemplar.budget,
                            endpoint_id=endpoint_id,
                            requested_count=30,
                            invalid_count=invalid,
                            opposed_count=opposed,
                            informative_nonadverse_count=informative,
                            targetability_limited_count=limited,
                            unsafe_false_promotion_unit_count=unsafe_units,
                            false_hold_unit_count=false_hold_units,
                            opposition_lower_bound=opposed_interval[0],
                            opposition_upper_bound=opposed_interval[1],
                            informative_nonadverse_lower_bound=informative_interval[0],
                            informative_nonadverse_upper_bound=informative_interval[1],
                            existence_recurs=opposed > 0,
                            majority_opposition=opposed_interval[0] > Decimal("0.5"),
                            majority_informative_nonadverse=(
                                informative_interval[0] > Decimal("0.5")
                            ),
                        )
                    )
    return tuple(sorted(cells, key=lambda value: value.cell_id))


def _bracket(
    cells: tuple[SplitCohortHistoryBudgetCellRecurrence, ...], *, shallow: str, deep: str, endpoint: str
) -> SplitCohortHistoryBudgetScientificState:
    shallow_cells = tuple(
        value
        for value in cells
        if value.coordinate_label == shallow and value.endpoint_id == endpoint
    )
    deep_cells = tuple(
        value for value in cells if value.coordinate_label == deep and value.endpoint_id == endpoint
    )
    if len(shallow_cells) != 9 or len(deep_cells) != 9:
        raise ValueError("split cohort history budget bracket cell roster differs")
    if any(value.invalid_count for value in (*shallow_cells, *deep_cells)):
        return SplitCohortHistoryBudgetScientificState.UNEVALUABLE
    if all(value.majority_opposition for value in shallow_cells) and all(
        value.opposed_count == 0 and value.majority_informative_nonadverse for value in deep_cells
    ):
        return SplitCohortHistoryBudgetScientificState.SUPPORTED
    if any(value.targetability_limited_count > 15 for value in deep_cells):
        return SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED
    return SplitCohortHistoryBudgetScientificState.OPPOSED


def _alignment_summaries(
    config: SplitCohortHistoryBudgetConfig,
    rows: tuple[SplitCohortHistoryBudgetTargetedCoordinateAdjudication, ...],
) -> tuple[SplitCohortHistoryBudgetAlignmentSummary, ...]:
    summaries: list[SplitCohortHistoryBudgetAlignmentSummary] = []
    unit_ids = evaluation_unit_ids()
    family_by_unit = {value.unit_id: value.family for value in rows}
    for endpoint, attribute in (
        ("decision-closure", "decision_state"),
        ("dynamical-closure", "dynamical_state"),
    ):
        for family in (None, *tuple(SplitCohortHistoryBudgetDisorderFamily)):
            requested_ids = tuple(
                unit for unit in unit_ids if family is None or family_by_unit[unit] is family
            )
            absolute_values: list[list[list[int]]] = []
            budget_values: list[list[list[int]]] = []
            codes: list[int] = []
            for unit_id in requested_ids:
                unit_rows = tuple(
                    value for value in rows if value.unit_id == unit_id and value.primary
                )
                try:
                    absolute = [
                        [
                            _STATE_CODE[
                                getattr(
                                    next(
                                        value
                                        for value in unit_rows
                                        if value.scale_cells == scale
                                        and value.coordinate_kind
                                        is SplitCohortHistoryBudgetCoordinateKind.ABSOLUTE_DEPTH
                                        and value.depth == depth
                                    ),
                                    attribute,
                                )
                            ]
                            for depth in (4, 8)
                        ]
                        for scale in (64, 128, 256)
                    ]
                    budget = [
                        [
                            _STATE_CODE[
                                getattr(
                                    next(
                                        value
                                        for value in unit_rows
                                        if value.scale_cells == scale
                                        and value.coordinate_kind
                                        is SplitCohortHistoryBudgetCoordinateKind.NORMALIZED_BUDGET
                                        and value.budget == budget_value
                                    ),
                                    attribute,
                                )
                            ]
                            for budget_value in (Decimal("0.5"), Decimal("1"))
                        ]
                        for scale in (64, 128, 256)
                    ]
                except (KeyError, StopIteration):
                    continue
                if any(
                    (not value.valid) or not bool(value.generator_observer_agreement)
                    for value in unit_rows
                ):
                    continue
                absolute_values.append(absolute)
                budget_values.append(budget)
                codes.append(_FAMILY_CODE[family_by_unit[unit_id]])
            family_id = "all-families" if family is None else family.value
            mean_delta: Decimal | None
            lower: Decimal | None
            upper: Decimal | None
            if absolute_values:
                estimate = paired_alignment(
                    absolute_states=np.asarray(absolute_values, dtype=np.int64),
                    normalized_states=np.asarray(budget_values, dtype=np.int64),
                    family_codes=np.asarray(codes, dtype=np.int64),
                    resamples=config.bootstrap_resamples,
                    seed=config.alignment_bootstrap_seed,
                )
                disposition = (
                    estimate.disposition
                    if len(absolute_values) == len(requested_ids)
                    else SplitCohortHistoryBudgetScientificState.UNEVALUABLE
                )
                mean_delta, lower, upper = estimate.mean_delta, estimate.lower_95, estimate.upper_95
            else:
                disposition = SplitCohortHistoryBudgetScientificState.UNEVALUABLE
                mean_delta = lower = upper = None
            summaries.append(
                SplitCohortHistoryBudgetAlignmentSummary(
                    summary_id=f"alignment.{family_id}.{endpoint}",
                    endpoint_id=endpoint,
                    family_id=family_id,
                    requested_unit_count=len(requested_ids),
                    evaluable_unit_count=len(absolute_values),
                    mean_delta=mean_delta,
                    lower_95=lower,
                    upper_95=upper,
                    disposition=disposition,
                )
            )
    return tuple(sorted(summaries, key=lambda value: value.summary_id))


def _resolution_summaries(
    rows: tuple[SplitCohortHistoryBudgetTargetedCoordinateAdjudication, ...],
) -> tuple[SplitCohortHistoryBudgetResolutionPanelSummary, ...]:
    summaries: list[SplitCohortHistoryBudgetResolutionPanelSummary] = []
    for family in SplitCohortHistoryBudgetDisorderFamily:
        for scale in (64, 128, 256):
            for depth in (4, 6, 8):
                selected = tuple(
                    value
                    for value in rows
                    if value.family is family
                    and value.scale_cells == scale
                    and value.coordinate_kind is SplitCohortHistoryBudgetCoordinateKind.ABSOLUTE_DEPTH
                    and value.depth == depth
                )
                by_unit = {
                    unit_id: tuple(value for value in selected if value.unit_id == unit_id)
                    for unit_id in sorted({value.unit_id for value in selected})
                }
                if len(by_unit) != 30 or any(len(values) != 3 for values in by_unit.values()):
                    raise ValueError("split cohort history budget resolution panel lost a requested view")
                for endpoint, attribute in (
                    ("decision-closure", "decision_state"),
                    ("dynamical-closure", "dynamical_state"),
                ):
                    invalid = changed = targetability = 0
                    for values in by_unit.values():
                        if any(
                            (not value.valid) or not bool(value.generator_observer_agreement)
                            for value in values
                        ):
                            invalid += 1
                            continue
                        states = tuple(getattr(value, attribute) for value in values)
                        changed += int(len(set(states)) > 1)
                        targetability += int(
                            len(
                                {
                                    value is SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED
                                    for value in states
                                }
                            )
                            > 1
                        )
                    disposition = (
                        SplitCohortHistoryBudgetScientificState.UNEVALUABLE
                        if invalid
                        else SplitCohortHistoryBudgetScientificState.TARGETABILITY_SHIFTED
                        if targetability
                        else SplitCohortHistoryBudgetScientificState.RESOLUTION_SHIFTED
                        if changed
                        else SplitCohortHistoryBudgetScientificState.RESOLUTION_STABLE
                    )
                    summaries.append(
                        SplitCohortHistoryBudgetResolutionPanelSummary(
                            summary_id=f"resolution.{family.value}.n{scale}.k{depth}.{endpoint}",
                            family=family,
                            scale_cells=scale,
                            depth=depth,
                            endpoint_id=endpoint,
                            requested_count=30,
                            invalid_count=invalid,
                            changed_state_count=changed,
                            targetability_changed_count=targetability,
                            disposition=disposition,
                        )
                    )
    return tuple(sorted(summaries, key=lambda value: value.summary_id))


def _untouched_cells(
    config: SplitCohortHistoryBudgetConfig, bundles: tuple[SplitCohortHistoryBudgetAdjudicationBundle, ...]
) -> tuple[SplitCohortHistoryBudgetUntouchedPrevalenceCell, ...]:
    rows = tuple(value for bundle in bundles for value in bundle.untouched_adjudications)
    cells: list[SplitCohortHistoryBudgetUntouchedPrevalenceCell] = []
    primary_tail = float(config.simultaneous_alpha) / 2.0 / 18.0
    for family in SplitCohortHistoryBudgetDisorderFamily:
        for scale in (64, 128, 256):
            selected = tuple(
                value for value in rows if value.family is family and value.scale_cells == scale
            )
            for label, kind, coordinate_value in (
                ("k4", "k", Decimal(4)),
                ("k8", "k", Decimal(8)),
                ("b0p5", "b", Decimal("0.5")),
                ("b1", "b", Decimal("1")),
            ):
                coordinate_rows = tuple(
                    value
                    for value in selected
                    if (
                        (
                            kind == "k"
                            and value.coordinate_id.startswith(
                                f"coordinate.n{scale}.k{int(coordinate_value)}.eps-0.005"
                            )
                        )
                        or (
                            kind == "b"
                            and value.coordinate_id.startswith(
                                f"coordinate.n{scale}.b{str(coordinate_value).replace('.', 'p')}."
                            )
                        )
                    )
                )
                if len(coordinate_rows) != 30:
                    raise ValueError("split cohort history budget primary U cell lost its requested denominator")
                invalid = sum(not value.valid for value in coordinate_rows)
                encounter = sum(
                    bool(value.collision_edge_count) for value in coordinate_rows if value.valid
                )
                dynamic = sum(
                    bool(value.dynamic_adverse_edge_count)
                    for value in coordinate_rows
                    if value.valid
                )
                decision = sum(
                    bool(value.decision_adverse_edge_count)
                    for value in coordinate_rows
                    if value.valid
                )
                any_adverse = sum(
                    bool(value.any_adverse_edge_count) for value in coordinate_rows if value.valid
                )
                unsafe = sum(
                    bool(value.unsafe_false_promotion_edge_count)
                    for value in coordinate_rows
                    if value.valid
                )
                false_hold = sum(
                    bool(value.false_hold_edge_count) for value in coordinate_rows if value.valid
                )
                alpha_tail = primary_tail if kind == "k" else 0.025
                any_interval = None if invalid else _exact_interval(any_adverse, 30, alpha_tail)
                unsafe_interval = None if invalid else _exact_interval(unsafe, 30, alpha_tail)

                def disposition(
                    interval: tuple[Decimal, Decimal] | None,
                ) -> SplitCohortHistoryBudgetScientificState:
                    if invalid:
                        return SplitCohortHistoryBudgetScientificState.UNEVALUABLE
                    assert interval is not None
                    if interval[1] < config.untouched_low_rate_threshold:
                        return SplitCohortHistoryBudgetScientificState.LOW_RATE_BOUNDED
                    if interval[0] > config.majority_opposition_threshold:
                        return SplitCohortHistoryBudgetScientificState.MAJORITY_RECURS
                    return SplitCohortHistoryBudgetScientificState.MIXED

                exemplar = coordinate_rows[0]
                cells.append(
                    SplitCohortHistoryBudgetUntouchedPrevalenceCell(
                        cell_id=f"u-cell.{family.value}.n{scale}.{label}",
                        family=family,
                        scale_cells=scale,
                        coordinate_label=label,
                        depth=exemplar.depth,
                        budget=None if kind == "k" else coordinate_value,
                        requested_count=30,
                        invalid_count=invalid,
                        encounter_count=encounter,
                        dynamic_adverse_count=dynamic,
                        decision_adverse_count=decision,
                        any_adverse_count=any_adverse,
                        unsafe_false_promotion_count=unsafe,
                        false_hold_count=false_hold,
                        any_adverse_lower_bound=(None if any_interval is None else any_interval[0]),
                        any_adverse_upper_bound=(None if any_interval is None else any_interval[1]),
                        unsafe_lower_bound=(
                            None if unsafe_interval is None else unsafe_interval[0]
                        ),
                        unsafe_upper_bound=(
                            None if unsafe_interval is None else unsafe_interval[1]
                        ),
                        any_adverse_disposition=disposition(any_interval),
                        unsafe_disposition=disposition(unsafe_interval),
                    )
                )
    return tuple(sorted(cells, key=lambda value: value.cell_id))


def _untouched_descriptive_summaries(
    config: SplitCohortHistoryBudgetConfig,
    bundles: tuple[SplitCohortHistoryBudgetAdjudicationBundle, ...],
) -> tuple[SplitCohortHistoryBudgetUntouchedDescriptiveSummary, ...]:
    rows = tuple(value for bundle in bundles for value in bundle.untouched_adjudications)
    rng = np.random.Generator(np.random.PCG64(config.untouched_bootstrap_seed))
    total_pairs = (
        config.untouched_preparation_count * (config.untouched_preparation_count - 1) / 2.0
    )
    summaries: list[SplitCohortHistoryBudgetUntouchedDescriptiveSummary] = []
    for family in SplitCohortHistoryBudgetDisorderFamily:
        for scale in (64, 128, 256):
            selected = tuple(
                value for value in rows if value.family is family and value.scale_cells == scale
            )
            coordinate_ids = tuple(sorted({value.coordinate_id for value in selected}))
            if len(coordinate_ids) != 16:
                raise ValueError("split cohort history budget U descriptive coordinate roster differs")
            for coordinate_id in coordinate_ids:
                coordinate_rows = tuple(
                    value for value in selected if value.coordinate_id == coordinate_id
                )
                if len(coordinate_rows) != 30:
                    raise ValueError("split cohort history budget U descriptive denominator differs")
                valid_rows = tuple(value for value in coordinate_rows if value.valid)
                collision_fractions = np.asarray(
                    [
                        float(_required_count(value.collision_edge_count)) / total_pairs
                        for value in valid_rows
                    ],
                    dtype=np.float64,
                )
                if valid_rows:
                    draws = rng.integers(
                        0,
                        len(valid_rows),
                        size=(config.bootstrap_resamples, len(valid_rows)),
                    )
                    collision_bootstrap = np.mean(collision_fractions[draws], axis=1)
                    collision_lower, collision_upper = np.quantile(
                        collision_bootstrap, (0.025, 0.975)
                    )
                    collision_mean = _decimal(float(np.mean(collision_fractions)))
                else:
                    collision_mean = None
                    collision_lower = collision_upper = None
                adverse_fractions = np.asarray(
                    [
                        float(_required_count(value.any_adverse_edge_count))
                        / float(_required_count(value.collision_edge_count))
                        for value in valid_rows
                        if value.collision_edge_count
                    ],
                    dtype=np.float64,
                )
                if adverse_fractions.size:
                    adverse_draws = rng.integers(
                        0,
                        adverse_fractions.size,
                        size=(config.bootstrap_resamples, adverse_fractions.size),
                    )
                    adverse_bootstrap = np.mean(adverse_fractions[adverse_draws], axis=1)
                    adverse_lower, adverse_upper = np.quantile(adverse_bootstrap, (0.025, 0.975))
                    adverse_mean = _decimal(float(np.mean(adverse_fractions)))
                else:
                    adverse_mean = None
                    adverse_lower = adverse_upper = None
                summaries.append(
                    SplitCohortHistoryBudgetUntouchedDescriptiveSummary(
                        summary_id=f"u-descriptive.{family.value}.n{scale}.{coordinate_id}",
                        family=family,
                        scale_cells=scale,
                        coordinate_id=coordinate_id,
                        requested_count=30,
                        valid_count=len(valid_rows),
                        prefix_encounter_unit_counts=tuple(
                            sum(bool(value.prefix_edge_counts[index]) for value in valid_rows)
                            for index in range(4)
                        ),
                        mean_collision_edge_fraction=collision_mean,
                        collision_edge_fraction_lower_95=(
                            None if collision_lower is None else _decimal(float(collision_lower))
                        ),
                        collision_edge_fraction_upper_95=(
                            None if collision_upper is None else _decimal(float(collision_upper))
                        ),
                        adverse_given_collision_evaluable_count=int(adverse_fractions.size),
                        mean_adverse_given_collision_fraction=adverse_mean,
                        adverse_given_collision_lower_95=(
                            None if adverse_lower is None else _decimal(float(adverse_lower))
                        ),
                        adverse_given_collision_upper_95=(
                            None if adverse_upper is None else _decimal(float(adverse_upper))
                        ),
                        whole_unit_bootstrap=True,
                    )
                )
    return tuple(sorted(summaries, key=lambda value: value.summary_id))


def _rank_summaries(
    bundles: tuple[SplitCohortHistoryBudgetAdjudicationBundle, ...],
) -> tuple[SplitCohortHistoryBudgetRankObjectSummary, ...]:
    summaries: list[SplitCohortHistoryBudgetRankObjectSummary] = []
    for family in SplitCohortHistoryBudgetDisorderFamily:
        family_bundles = tuple(
            value for value in bundles if value.adjudications[0].family is family
        )
        if len(family_bundles) != 30:
            raise ValueError("split cohort history budget rank summary family denominator differs")
        first_history = family_bundles[0].history_bundle
        if first_history is None:
            raise ValueError("split cohort history budget T rank summary lacks its history bundle")
        for scale in (64, 128, 256):
            coordinate_ids = tuple(
                value.coordinate_id
                for value in first_history.coordinates
                if value.scale_cells == scale
            )
            if len(coordinate_ids) != 16:
                raise ValueError("split cohort history budget rank summary coordinate roster differs")
            for coordinate_id in coordinate_ids:
                coordinate = next(
                    value
                    for value in first_history.coordinates
                    if value.coordinate_id == coordinate_id
                )
                structural = []
                discrete_lower = []
                discrete_upper = []
                effective = []
                compatibility = []
                right_censored = 0
                for bundle in family_bundles:
                    history = bundle.history_bundle
                    if history is None:
                        raise ValueError("split cohort history budget T rank summary lacks a history bundle")
                    structural_row = next(
                        value
                        for value in history.structural_rank_steps
                        if value.coordinate_id == coordinate_id
                    )
                    discrete_row = next(
                        value
                        for value in history.discrete_rank_brackets
                        if value.coordinate_id == coordinate_id
                    )
                    conditioning_row = next(
                        value
                        for value in history.conditioning_steps
                        if value.coordinate_id == coordinate_id
                    )
                    structural.append(structural_row.structural_rank)
                    discrete_lower.append(discrete_row.lower_rank)
                    discrete_upper.append(discrete_row.upper_rank)
                    effective.append(conditioning_row.effective_rank)
                    compatibility.append(discrete_row.compatibility)
                    right_censored += int(conditioning_row.right_censored)
                summaries.append(
                    SplitCohortHistoryBudgetRankObjectSummary(
                        summary_id=f"rank-summary.{family.value}.n{scale}.{coordinate_id}",
                        family=family,
                        scale_cells=scale,
                        coordinate_id=coordinate_id,
                        depth=coordinate.depth,
                        resolution_epsilon=coordinate.resolution_epsilon,
                        requested_count=30,
                        structural_rank_median=_decimal(float(np.median(structural))),
                        discrete_lower_rank_median=_decimal(float(np.median(discrete_lower))),
                        discrete_upper_rank_median=_decimal(float(np.median(discrete_upper))),
                        effective_rank_median=_decimal(float(np.median(effective))),
                        compatible_count=sum(
                            value is SplitCohortHistoryBudgetRankCompatibility.COMPATIBLE for value in compatibility
                        ),
                        sampling_alias_count=sum(
                            value is SplitCohortHistoryBudgetRankCompatibility.SAMPLING_ALIAS
                            for value in compatibility
                        ),
                        numerical_cancellation_count=sum(
                            value is SplitCohortHistoryBudgetRankCompatibility.NUMERICAL_CANCELLATION
                            for value in compatibility
                        ),
                        unresolved_count=sum(
                            value is SplitCohortHistoryBudgetRankCompatibility.RANK_UNRESOLVED
                            for value in compatibility
                        ),
                        conditioning_right_censored_count=right_censored,
                    )
                )
    return tuple(sorted(summaries, key=lambda value: value.summary_id))


def _transition_disposition(
    states: tuple[SplitCohortHistoryBudgetScientificState, ...],
) -> SplitCohortHistoryBudgetScientificState:
    """Apply the frozen one-way phase ordering without compensating mixed cells."""

    order = {
        SplitCohortHistoryBudgetScientificState.OPPOSED: 0,
        SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED: 1,
        SplitCohortHistoryBudgetScientificState.INFORMATIVE_NONADVERSE: 2,
    }
    if SplitCohortHistoryBudgetScientificState.INVALID in states:
        return SplitCohortHistoryBudgetScientificState.UNEVALUABLE
    if SplitCohortHistoryBudgetScientificState.MIXED in states:
        return SplitCohortHistoryBudgetScientificState.MIXED
    if any(
        states[left] in order
        and states[right] in order
        and order[states[left]] > order[states[right]]
        for left in range(len(states))
        for right in range(left + 1, len(states))
    ):
        return SplitCohortHistoryBudgetScientificState.NONMONOTONE_PHASE_PATTERN
    if (
        SplitCohortHistoryBudgetScientificState.OPPOSED in states
        and SplitCohortHistoryBudgetScientificState.INFORMATIVE_NONADVERSE in states
    ):
        return SplitCohortHistoryBudgetScientificState.SUPPORTED
    if states and all(value is SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED for value in states):
        return SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED
    return SplitCohortHistoryBudgetScientificState.RIGHT_CENSORED


def _transition_summaries(
    primary: tuple[SplitCohortHistoryBudgetCellRecurrence, ...],
    secondary: tuple[SplitCohortHistoryBudgetCellRecurrence, ...],
) -> tuple[SplitCohortHistoryBudgetTransitionSummary, ...]:
    summaries: list[SplitCohortHistoryBudgetTransitionSummary] = []
    all_cells = (*primary, *secondary)
    grids = {
        SplitCohortHistoryBudgetCoordinateKind.ABSOLUTE_DEPTH: ("k0", "k2", "k4", "k6", "k8", "k12"),
        SplitCohortHistoryBudgetCoordinateKind.NORMALIZED_BUDGET: ("b0p125", "b0p25", "b0p5", "b1"),
    }
    for family in SplitCohortHistoryBudgetDisorderFamily:
        for scale in (64, 128, 256):
            for kind, labels in grids.items():
                for endpoint in ("decision-closure", "dynamical-closure"):
                    cells = tuple(
                        next(
                            value
                            for value in all_cells
                            if value.family is family
                            and value.scale_cells == scale
                            and value.coordinate_label == label
                            and value.endpoint_id == endpoint
                        )
                        for label in labels
                    )
                    states = []
                    for cell in cells:
                        counts = {
                            SplitCohortHistoryBudgetScientificState.OPPOSED: cell.opposed_count,
                            SplitCohortHistoryBudgetScientificState.INFORMATIVE_NONADVERSE: cell.informative_nonadverse_count,
                            SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED: cell.targetability_limited_count,
                        }
                        maximum = max(counts.values())
                        winners = tuple(key for key, value in counts.items() if value == maximum)
                        states.append(
                            SplitCohortHistoryBudgetScientificState.INVALID
                            if cell.invalid_count
                            else winners[0]
                            if len(winners) == 1
                            else SplitCohortHistoryBudgetScientificState.MIXED
                        )
                    order = {
                        SplitCohortHistoryBudgetScientificState.OPPOSED: 0,
                        SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED: 1,
                        SplitCohortHistoryBudgetScientificState.INFORMATIVE_NONADVERSE: 2,
                    }
                    reversals = tuple(
                        sorted(
                            f"{labels[left]}-to-{labels[right]}"
                            for left in range(len(labels))
                            for right in range(left + 1, len(labels))
                            if states[left] in order
                            and states[right] in order
                            and order[states[left]] > order[states[right]]
                        )
                    )
                    opposed_labels = tuple(
                        label
                        for label, state in zip(labels, states)
                        if state is SplitCohortHistoryBudgetScientificState.OPPOSED
                    )
                    informative_labels = tuple(
                        label
                        for label, state in zip(labels, states)
                        if state is SplitCohortHistoryBudgetScientificState.INFORMATIVE_NONADVERSE
                    )
                    limited_labels = tuple(
                        sorted(
                            label
                            for label, state in zip(labels, states)
                            if state is SplitCohortHistoryBudgetScientificState.TARGETABILITY_LIMITED
                        )
                    )
                    disposition = _transition_disposition(tuple(states))
                    summaries.append(
                        SplitCohortHistoryBudgetTransitionSummary(
                            summary_id=(
                                f"transition.{family.value}.n{scale}."
                                f"{kind.value.lower().replace('_', '-')}.{endpoint}"
                            ),
                            family=family,
                            scale_cells=scale,
                            coordinate_kind=kind,
                            endpoint_id=endpoint,
                            ordered_coordinate_labels=labels,
                            last_opposed_coordinate=(
                                None if not opposed_labels else opposed_labels[-1]
                            ),
                            first_informative_nonadverse_coordinate=(
                                None if not informative_labels else informative_labels[0]
                            ),
                            targetability_limited_coordinates=limited_labels,
                            reversal_pairs=reversals,
                            disposition=disposition,
                        )
                    )
    return tuple(sorted(summaries, key=lambda value: value.summary_id))


def synthesize_recurrence(
    *,
    config: SplitCohortHistoryBudgetConfig,
    method_freeze: SplitCohortHistoryBudgetMethodFreeze,
    adjudication_bundles: tuple[SplitCohortHistoryBudgetAdjudicationBundle, ...],
    include_untouched: bool = True,
) -> RecurrenceExecution:
    if config.phase is not SplitCohortHistoryBudgetPhase.TARGETED_EVALUATION or len(adjudication_bundles) != 90:
        raise ValueError("split cohort history budget synthesis requires the complete evaluation bundle roster")
    unit_ids = tuple(value.unit_id for value in adjudication_bundles)
    if unit_ids != evaluation_unit_ids() or len(set(unit_ids)) != 90:
        raise ValueError("split cohort history budget evaluation bundles must be sorted and unique")
    if any(
        value.method_freeze_sha256 != method_freeze.fingerprint() for value in adjudication_bundles
    ):
        raise ValueError("split cohort history budget adjudications differ from the method freeze")
    targeted = tuple(
        value for bundle in adjudication_bundles for value in bundle.targeted_adjudications
    )
    coupled_scale_adjudications = tuple(
        sorted(
            (value for bundle in adjudication_bundles for value in bundle.adjudications),
            key=lambda value: (value.unit_id, value.scale_cells),
        )
    )
    if len(coupled_scale_adjudications) != 270:
        raise ValueError("split cohort history budget coupled scale ledger differs")
    cells = _targeted_cells(config, targeted)
    secondary_cells = _secondary_targeted_cells(targeted)
    untouched_cells = _untouched_cells(config, adjudication_bundles) if include_untouched else ()
    untouched_descriptive = (
        _untouched_descriptive_summaries(config, adjudication_bundles) if include_untouched else ()
    )
    alignment = _alignment_summaries(config, targeted)
    resolution = _resolution_summaries(targeted)
    rank_summaries = _rank_summaries(adjudication_bundles)
    transitions = _transition_summaries(cells, secondary_cells)
    bootstrap = _bootstrap_summary(config, method_freeze, coupled_scale_adjudications)
    decisive_ids = tuple(
        sorted(
            {
                pair.nomination_id
                for value in coupled_scale_adjudications
                for pair in value.pair_adjudications
                if pair.dynamically_adverse or pair.decision_adverse
            }
        )
    )
    result = SplitCohortHistoryBudgetRecurrenceResult(
        result_id="split-cohort-history-budget-result",
        method_freeze_sha256=method_freeze.fingerprint(),
        cells=cells,
        secondary_cells=secondary_cells,
        untouched_cells=untouched_cells,
        untouched_descriptive_summaries=untouched_descriptive,
        alignment_summaries=alignment,
        resolution_summaries=resolution,
        rank_summaries=rank_summaries,
        transition_summaries=transitions,
        fixed_depth_dynamical_bracket=_bracket(
            cells, shallow="k4", deep="k8", endpoint="dynamical-closure"
        ),
        fixed_depth_decision_bracket=_bracket(
            cells, shallow="k4", deep="k8", endpoint="decision-closure"
        ),
        normalized_budget_dynamical_bracket=_bracket(
            cells, shallow="b0p5", deep="b1", endpoint="dynamical-closure"
        ),
        normalized_budget_decision_bracket=_bracket(
            cells, shallow="b0p5", deep="b1", endpoint="decision-closure"
        ),
        bootstrap_summary_sha256=bootstrap.fingerprint(),
        decisive_counterexample_ids=decisive_ids,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        physical_claim_allowed=False,
        controller_claim_allowed=False,
    )
    return RecurrenceExecution(result=result, bootstrap_summary=bootstrap)


def targeted_continuation_gate(
    result: SplitCohortHistoryBudgetRecurrenceResult,
) -> SplitCohortHistoryBudgetTContinuationGateRecord:
    k4 = tuple(value for value in result.cells if value.coordinate_label == "k4")
    valid = len(k4) == 18 and all(value.invalid_count == 0 for value in k4)
    endpoints = tuple(
        sorted(
            endpoint
            for endpoint in ("decision-closure", "dynamical-closure")
            if any(value.endpoint_id == endpoint and value.majority_opposition for value in k4)
        )
    )
    # Any invalid cell already includes observer/generator/certificate identity
    # failures in the bounded T result.  The staged gate does not inspect raw
    # outcomes or select units.
    independence_failures = sum(value.invalid_count for value in result.cells)
    reasons = []
    if not valid:
        reasons.append("SPLIT_COHORT_HISTORY_BUDGET_TARGETED_COORDINATE_PRIMARY_K4_INVALID")
    if independence_failures:
        reasons.append("SPLIT_COHORT_HISTORY_BUDGET_TARGETED_COORDINATE_INDEPENDENCE_FAILURE")
    if not endpoints:
        reasons.append("SPLIT_COHORT_HISTORY_BUDGET_TARGETED_COORDINATE_K4_MAJORITY_OPPOSITION_ABSENT")
    return SplitCohortHistoryBudgetTContinuationGateRecord(
        gate_id="split-cohort-history-budget.targeted-evaluation-continuation-gate",
        predicate_id="split-cohort-history-budget-t-construct-breakability-confirmed",
        targeted_result_sha256=result.fingerprint(),
        all_primary_k4_cells_valid=valid,
        independence_failure_count=independence_failures,
        k4_majority_opposition_endpoint_ids=endpoints,
        activate_untouched_evaluation=valid and independence_failures == 0 and bool(endpoints),
        eligible_unit_count=90,
        reason_codes=tuple(reasons),
    )


def integrate_untouched_recurrence(
    *,
    config: SplitCohortHistoryBudgetConfig,
    targeted_result: SplitCohortHistoryBudgetRecurrenceResult,
    adjudication_bundles: tuple[SplitCohortHistoryBudgetAdjudicationBundle, ...],
) -> SplitCohortHistoryBudgetRecurrenceResult:
    """Add the conditional U evidence without pooling it into T estimates."""

    if config.phase is not SplitCohortHistoryBudgetPhase.UNTOUCHED_EVALUATION or len(adjudication_bundles) != 90:
        raise ValueError("split cohort history budget U synthesis requires all 90 conditional units")
    unit_ids = tuple(value.unit_id for value in adjudication_bundles)
    if unit_ids != evaluation_unit_ids() or any(
        value.targeted_adjudications or value.adjudications for value in adjudication_bundles
    ):
        raise ValueError("split cohort history budget U synthesis received another cohort")
    cells = _untouched_cells(config, adjudication_bundles)
    descriptive = _untouched_descriptive_summaries(config, adjudication_bundles)
    return replace(
        targeted_result,
        result_id="split-cohort-history-budget-integrated-result",
        untouched_cells=cells,
        untouched_descriptive_summaries=descriptive,
    )


__all__ = [
    "RecurrenceExecution",
    "integrate_untouched_recurrence",
    "synthesize_recurrence",
    "targeted_continuation_gate",
]

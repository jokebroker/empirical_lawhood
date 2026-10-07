"""Complete-unit targeted and untouched phase-diagram inference for receiver-history."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal

import numpy as np
from scipy.stats import beta

from empirical_lawhood.adapters.receiver_history.contracts import (
    ReceiverHistoryAlignmentSummary,
    ReceiverHistoryCellRecurrence,
    ReceiverHistoryConfig,
    ReceiverHistoryCoordinateKind,
    ReceiverHistoryDisorderFamily,
    ReceiverHistoryDevelopmentMetatheoryGate,
    ReceiverHistoryEmpiricalMetatheoryDossier,
    ReceiverHistoryEndpoint,
    ReceiverHistoryHypothesisDisposition,
    ReceiverHistoryInferenceDispositionMatrix,
    ReceiverHistoryLocalityCandidate,
    ReceiverHistoryLocalityCandidateAssessment,
    ReceiverHistoryLocalityTournamentResult,
    ReceiverHistoryMetatheoryDisposition,
    ReceiverHistoryMetatheoryPropositionDisposition,
    ReceiverHistoryMethodFreeze,
    ReceiverHistoryNonclaimDisposition,
    ReceiverHistoryEvidenceGateDisposition,
    ReceiverHistoryPhase,
    ReceiverHistoryRecurrenceResult,
    ReceiverHistoryRankCompatibility,
    ReceiverHistoryRankObjectSummary,
    ReceiverHistoryResolutionPanelSummary,
    ReceiverHistoryScientificState,
    ReceiverHistoryTargetedCoordinateAdjudication,
    ReceiverHistoryTransitionSummary,
    ReceiverHistoryPowerState,
    ReceiverHistoryPowerTransportCell,
    ReceiverHistoryTargetedContinuationGateRecord,
    ReceiverHistoryTerminal,
    ReceiverHistoryUntouchedPrevalenceCell,
    ReceiverHistoryUntouchedDescriptiveSummary,
    ReceiverHistoryUnitAdjudication,
)
from empirical_lawhood.adapters.receiver_history.descriptors import evaluation_unit_ids
from empirical_lawhood.adapters.receiver_history.runtime_contracts import (
    ReceiverHistoryAdjudicationBundle,
    ReceiverHistoryBootstrapCellSummary,
    ReceiverHistoryBootstrapInterval,
    ReceiverHistoryBootstrapSummary,
    ReceiverHistoryEndpointCoordinatePowerAtlas,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling

from .alignment import paired_alignment


@dataclass(frozen=True, slots=True)
class RecurrenceExecution:
    result: ReceiverHistoryRecurrenceResult
    bootstrap_summary: ReceiverHistoryBootstrapSummary
    disposition_matrix: ReceiverHistoryInferenceDispositionMatrix


_STATE_CODE = {
    ReceiverHistoryScientificState.OPPOSED: 0,
    ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE: 1,
    ReceiverHistoryScientificState.TARGETABILITY_LIMITED: 2,
}
_FAMILY_CODE = {value: index for index, value in enumerate(ReceiverHistoryDisorderFamily)}
_NONCLAIM_REASONS = (
    "no-universal-equation-or-meta-law",
    "no-universal-coordinate-necessity",
    "no-cross-substrate-pooling",
    "no-universal-memory-boundary",
    "no-resolution-independent-lawhood",
    "no-manifold-topology-or-category-result",
    "no-continuum-or-rg-law",
    "no-mechanism-identification",
    "no-natural-prevalence",
    "no-rank-prediction-benefit-entailment",
    'law-qualification-does-not-entail-admission-or-prospective-use',
    "no-physical-safety-or-control",
    "no-reinforcement-learning-or-reward",
    "no-default-safe-hold",
)
_EXCLUDED_EXPERIMENT_IDS = (
    "continuum-scaling-law",
    "full-rank-staircase",
    "independent-outcome-engine",
    'receiver-admission-and-prospective-controller',
    "physical-hardware-replication",
    "three-resolution-panel",
    'untouched-512-preparation-prevalence',
    'untouched-512-cell-refinement',
)
_METATHEORY_LIMITATION_IDS = (
    "finite-entered-simulator-population",
    "no-continuum-or-physical-transport",
    "no-natural-prevalence-estimate",
    'no-admission-prospective-use-or-controller-claim',
    "single-software-ontology",
)


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("receiver-history inference output must be finite")
    return Decimal(str(float(value)))


def _required_count(value: int | None) -> int:
    if value is None:
        raise ValueError("valid receiver-history evidence row lacks its count")
    return value


def _evaluation_units_per_family(config: ReceiverHistoryConfig) -> int:
    counts = tuple(
        sum(family.value in unit_id for unit_id in config.unit_ids)
        for family in ReceiverHistoryDisorderFamily
    )
    if len(set(counts)) != 1 or counts[0] not in {12, 30}:
        raise ValueError("receiver-history evaluation family denominator differs")
    return counts[0]


def _exact_interval(successes: int, requested: int, alpha_tail: float) -> tuple[Decimal, Decimal]:
    if not 0 <= successes <= requested or requested <= 0 or not 0.0 < alpha_tail < 1.0:
        raise ValueError("receiver-history exact interval inputs differ")
    lower = (
        0.0 if successes == 0 else float(beta.ppf(alpha_tail, successes, requested - successes + 1))
    )
    upper = (
        1.0
        if successes == requested
        else float(beta.ppf(1.0 - alpha_tail, successes + 1, requested - successes))
    )
    return _decimal(lower), _decimal(upper)


def _interval(
    values: np.ndarray[tuple[int], np.dtype[np.float64]],
) -> ReceiverHistoryBootstrapInterval:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return ReceiverHistoryBootstrapInterval(
        lower=_decimal(float(lower)), upper=_decimal(float(upper))
    )


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
    config: ReceiverHistoryConfig,
    method_freeze: ReceiverHistoryMethodFreeze,
    adjudications: tuple[ReceiverHistoryUnitAdjudication, ...],
) -> ReceiverHistoryBootstrapSummary:
    """Whole-unit rank summaries; nested scale/depth rows are never resampled."""

    requested_per_family = _evaluation_units_per_family(config)
    rng = np.random.Generator(np.random.PCG64(config.bootstrap_seed))
    cell_summaries: list[ReceiverHistoryBootstrapCellSummary] = []
    for family in ReceiverHistoryDisorderFamily:
        family_rows = tuple(value for value in adjudications if value.family is family)
        unit_ids = tuple(sorted({value.unit_id for value in family_rows}))
        by_key = {(value.unit_id, value.scale_cells): value for value in family_rows}
        if len(unit_ids) != requested_per_family:
            raise ValueError("receiver-history bootstrap requires its complete family seed roster")
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
                raise ValueError("receiver-history k_full and b_full evaluability differs")
            k_bootstrap = _conditional_bootstrap_mean(k_values, draws)
            b_bootstrap = _conditional_bootstrap_mean(b_values, draws)
            cell_summaries.append(
                ReceiverHistoryBootstrapCellSummary(
                    cell_id=f"bootstrap-cell.{family.value}.n{scale}",
                    family=family,
                    scale_cells=scale,
                    requested_unit_count=requested_per_family,
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
    return ReceiverHistoryBootstrapSummary(
        summary_id="receiver-history.bootstrap-summary",
        evaluation_config_sha256=config.fingerprint(),
        method_freeze_sha256=method_freeze.fingerprint(),
        cells=tuple(sorted(cell_summaries, key=lambda value: value.cell_id)),
        resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
        confidence_level=Decimal("0.95"),
        whole_seed_block_resampling=True,
        nested_scale_action_mode_resampling=False,
    )


def _coordinate_label(row: ReceiverHistoryTargetedCoordinateAdjudication) -> str:
    if row.coordinate_kind is ReceiverHistoryCoordinateKind.ABSOLUTE_DEPTH:
        return f"k{row.depth}"
    assert row.budget is not None
    return f"b{str(row.budget).replace('.', 'p')}"


def _full_coordinate_label(row: ReceiverHistoryTargetedCoordinateAdjudication) -> str:
    label = _coordinate_label(row)
    if (
        row.coordinate_kind is ReceiverHistoryCoordinateKind.ABSOLUTE_DEPTH
        and row.resolution_epsilon != Decimal("0.005")
    ):
        return f"{label}-eps{str(row.resolution_epsilon).replace('.', 'p')}"
    return label


def _targeted_cells(
    config: ReceiverHistoryConfig,
    rows: tuple[ReceiverHistoryTargetedCoordinateAdjudication, ...],
) -> tuple[ReceiverHistoryCellRecurrence, ...]:
    requested_per_family = _evaluation_units_per_family(config)
    cells: list[ReceiverHistoryCellRecurrence] = []
    for family in ReceiverHistoryDisorderFamily:
        for scale in (64, 128, 256):
            selected = tuple(
                value
                for value in rows
                if value.family is family and value.scale_cells == scale and value.primary
            )
            labels = tuple(sorted({_coordinate_label(value) for value in selected}))
            if labels != ("b0p5", "b1", "k0", "k4", "k8"):
                raise ValueError('receiver-history primary targeted coordinate roster differs')
            for label in labels:
                coordinate_rows = tuple(
                    value for value in selected if _coordinate_label(value) == label
                )
                if len(coordinate_rows) != requested_per_family:
                    raise ValueError('receiver-history targeted cell lost its requested denominator')
                exemplar = coordinate_rows[0]
                for endpoint_id, attribute, unsafe_attribute, hold_attribute in (
                    (
                        "target-decision",
                        "target_state",
                        "target_unsafe_false_promotion_count",
                        "target_false_hold_count",
                    ),
                    (
                        "sink-decision",
                        "sink_state",
                        "sink_unsafe_false_promotion_count",
                        "sink_false_hold_count",
                    ),
                    ("dynamical-closure", "dynamical_state", "", ""),
                ):
                    alpha_tail = float(config.simultaneous_alpha) / (9.0 if label == "k0" else 18.0)
                    valid_rows = tuple(
                        value
                        for value in coordinate_rows
                        if value.valid and bool(value.generator_observer_agreement)
                    )
                    states = tuple(getattr(value, attribute) for value in valid_rows)
                    invalid = requested_per_family - len(valid_rows)
                    opposed = sum(
                        value is ReceiverHistoryScientificState.OPPOSED for value in states
                    )
                    informative = sum(
                        value is ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
                        for value in states
                    )
                    limited = sum(
                        value is ReceiverHistoryScientificState.TARGETABILITY_LIMITED
                        for value in states
                    )
                    nonattempt = sum(
                        value is ReceiverHistoryScientificState.NOT_ATTEMPTED for value in states
                    )
                    unsafe_units = (
                        sum(
                            bool(getattr(row, unsafe_attribute))
                            for row in coordinate_rows
                            if row.valid and bool(row.generator_observer_agreement)
                        )
                        if unsafe_attribute
                        else 0
                    )
                    false_hold_units = (
                        sum(
                            bool(getattr(row, hold_attribute))
                            for row in coordinate_rows
                            if row.valid and bool(row.generator_observer_agreement)
                        )
                        if hold_attribute
                        else 0
                    )
                    opposed_interval = _exact_interval(opposed, requested_per_family, alpha_tail)
                    informative_interval = _exact_interval(
                        informative, requested_per_family, alpha_tail
                    )
                    cells.append(
                        ReceiverHistoryCellRecurrence(
                            cell_id=f"cell.{family.value}.n{scale}.{label}.{endpoint_id}",
                            family=family,
                            scale_cells=scale,
                            coordinate_label=label,
                            coordinate_kind=exemplar.coordinate_kind,
                            depth=exemplar.depth,
                            budget=exemplar.budget,
                            endpoint_id=endpoint_id,
                            requested_count=requested_per_family,
                            invalid_count=invalid,
                            opposed_count=opposed,
                            informative_nonadverse_count=informative,
                            targetability_limited_count=limited,
                            prerequisite_nonattempt_count=nonattempt,
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
    config: ReceiverHistoryConfig,
    rows: tuple[ReceiverHistoryTargetedCoordinateAdjudication, ...],
) -> tuple[ReceiverHistoryCellRecurrence, ...]:
    requested_per_family = _evaluation_units_per_family(config)
    cells: list[ReceiverHistoryCellRecurrence] = []
    for family in ReceiverHistoryDisorderFamily:
        for scale in (64, 128, 256):
            selected = tuple(
                value
                for value in rows
                if value.family is family and value.scale_cells == scale and not value.primary
            )
            coordinate_ids = tuple(sorted({value.coordinate_id for value in selected}))
            if len(coordinate_ids) != 11:
                raise ValueError('receiver-history secondary targeted coordinate roster differs')
            for coordinate_id in coordinate_ids:
                coordinate_rows = tuple(
                    value for value in selected if value.coordinate_id == coordinate_id
                )
                if len(coordinate_rows) != requested_per_family:
                    raise ValueError('receiver-history secondary targeted denominator differs')
                exemplar = coordinate_rows[0]
                label = _full_coordinate_label(exemplar)
                for endpoint_id, attribute, unsafe_attribute, hold_attribute in (
                    (
                        "target-decision",
                        "target_state",
                        "target_unsafe_false_promotion_count",
                        "target_false_hold_count",
                    ),
                    (
                        "sink-decision",
                        "sink_state",
                        "sink_unsafe_false_promotion_count",
                        "sink_false_hold_count",
                    ),
                    ("dynamical-closure", "dynamical_state", "", ""),
                ):
                    valid_rows = tuple(
                        value
                        for value in coordinate_rows
                        if value.valid and bool(value.generator_observer_agreement)
                    )
                    states = tuple(getattr(value, attribute) for value in valid_rows)
                    invalid = requested_per_family - len(valid_rows)
                    opposed = sum(
                        value is ReceiverHistoryScientificState.OPPOSED for value in states
                    )
                    informative = sum(
                        value is ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
                        for value in states
                    )
                    limited = requested_per_family - invalid - opposed - informative
                    nonattempt = sum(
                        value is ReceiverHistoryScientificState.NOT_ATTEMPTED for value in states
                    )
                    limited -= nonattempt
                    unsafe_units = (
                        sum(bool(getattr(row, unsafe_attribute)) for row in valid_rows)
                        if unsafe_attribute
                        else 0
                    )
                    false_hold_units = (
                        sum(bool(getattr(row, hold_attribute)) for row in valid_rows)
                        if hold_attribute
                        else 0
                    )
                    opposed_interval = _exact_interval(opposed, requested_per_family, 0.025)
                    informative_interval = _exact_interval(informative, requested_per_family, 0.025)
                    cells.append(
                        ReceiverHistoryCellRecurrence(
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
                            requested_count=requested_per_family,
                            invalid_count=invalid,
                            opposed_count=opposed,
                            informative_nonadverse_count=informative,
                            targetability_limited_count=limited,
                            prerequisite_nonattempt_count=nonattempt,
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
    cells: tuple[ReceiverHistoryCellRecurrence, ...],
    *,
    shallow: str,
    deep: str,
    endpoint: str,
    eligible: bool,
) -> ReceiverHistoryScientificState:
    shallow_cells = tuple(
        value
        for value in cells
        if value.coordinate_label == shallow and value.endpoint_id == endpoint
    )
    deep_cells = tuple(
        value for value in cells if value.coordinate_label == deep and value.endpoint_id == endpoint
    )
    if len(shallow_cells) != 9 or len(deep_cells) != 9:
        raise ValueError("receiver-history bracket cell roster differs")
    if not eligible:
        # Composite bracket eligibility requires the exact frozen shallow/deep
        # role across all nine family-scale cells.  Individual capable cells
        # still execute under the atlas mask when that composite is ineligible;
        # their power transport remains separately adjudicable.  phase-diagram synthesis validates
        # the exact per-cell execution mask without promoting the composite.
        return ReceiverHistoryScientificState.NOT_ATTEMPTED
    if any(value.invalid_count for value in (*shallow_cells, *deep_cells)):
        return ReceiverHistoryScientificState.INVALID
    if all(value.majority_opposition for value in shallow_cells) and all(
        value.opposed_count == 0 and value.majority_informative_nonadverse for value in deep_cells
    ):
        return ReceiverHistoryScientificState.SUPPORTED
    if any(value.targetability_limited_count * 2 > value.requested_count for value in deep_cells):
        return ReceiverHistoryScientificState.TARGETABILITY_LIMITED
    return ReceiverHistoryScientificState.OPPOSED


def _alignment_summaries(
    config: ReceiverHistoryConfig,
    rows: tuple[ReceiverHistoryTargetedCoordinateAdjudication, ...],
) -> tuple[ReceiverHistoryAlignmentSummary, ...]:
    summaries: list[ReceiverHistoryAlignmentSummary] = []
    unit_ids = config.unit_ids
    family_by_unit = {value.unit_id: value.family for value in rows}
    for endpoint, attribute in (
        ("target-decision", "target_state"),
        ("sink-decision", "sink_state"),
        ("dynamical-closure", "dynamical_state"),
    ):
        for family in (None, *tuple(ReceiverHistoryDisorderFamily)):
            requested_ids = tuple(
                unit for unit in unit_ids if family is None or family_by_unit[unit] is family
            )
            absolute_values: list[list[list[int]]] = []
            budget_values: list[list[list[int]]] = []
            codes: list[int] = []
            invalid_unit_count = 0
            for unit_id in requested_ids:
                unit_rows = tuple(
                    value for value in rows if value.unit_id == unit_id and value.primary
                )
                if len(unit_rows) != 15 or any(
                    (not value.valid) or not bool(value.generator_observer_agreement)
                    for value in unit_rows
                ):
                    invalid_unit_count += 1
                    continue
                try:
                    absolute_states = [
                        [
                            getattr(
                                next(
                                    value
                                    for value in unit_rows
                                    if value.scale_cells == scale
                                    and value.coordinate_kind
                                    is ReceiverHistoryCoordinateKind.ABSOLUTE_DEPTH
                                    and value.depth == depth
                                ),
                                attribute,
                            )
                            for depth in (4, 8)
                        ]
                        for scale in (64, 128, 256)
                    ]
                    budget_states = [
                        [
                            getattr(
                                next(
                                    value
                                    for value in unit_rows
                                    if value.scale_cells == scale
                                    and value.coordinate_kind
                                    is ReceiverHistoryCoordinateKind.NORMALIZED_BUDGET
                                    and value.budget == budget_value
                                ),
                                attribute,
                            )
                            for budget_value in (Decimal("0.5"), Decimal("1"))
                        ]
                        for scale in (64, 128, 256)
                    ]
                except StopIteration:
                    invalid_unit_count += 1
                    continue
                if any(
                    value is ReceiverHistoryScientificState.INVALID
                    for group in (*absolute_states, *budget_states)
                    for value in group
                ):
                    invalid_unit_count += 1
                    continue
                states = tuple(
                    value for group in (*absolute_states, *budget_states) for value in group
                )
                if ReceiverHistoryScientificState.NOT_ATTEMPTED in states:
                    continue
                if any(value not in _STATE_CODE for value in states):
                    invalid_unit_count += 1
                    continue
                absolute = [[_STATE_CODE[value] for value in group] for group in absolute_states]
                budget = [[_STATE_CODE[value] for value in group] for group in budget_states]
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
                    ReceiverHistoryScientificState.INVALID
                    if invalid_unit_count
                    else estimate.disposition
                    if len(absolute_values) == len(requested_ids)
                    else ReceiverHistoryScientificState.UNEVALUABLE
                )
                mean_delta, lower, upper = estimate.mean_delta, estimate.lower_95, estimate.upper_95
            else:
                disposition = (
                    ReceiverHistoryScientificState.INVALID
                    if invalid_unit_count
                    else ReceiverHistoryScientificState.UNEVALUABLE
                )
                mean_delta = lower = upper = None
            summaries.append(
                ReceiverHistoryAlignmentSummary(
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
    config: ReceiverHistoryConfig,
    rows: tuple[ReceiverHistoryTargetedCoordinateAdjudication, ...],
) -> tuple[ReceiverHistoryResolutionPanelSummary, ...]:
    requested_per_family = _evaluation_units_per_family(config)
    summaries: list[ReceiverHistoryResolutionPanelSummary] = []
    for family in ReceiverHistoryDisorderFamily:
        for scale in (64, 128, 256):
            for depth in (4, 6, 8):
                selected = tuple(
                    value
                    for value in rows
                    if value.family is family
                    and value.scale_cells == scale
                    and value.coordinate_kind is ReceiverHistoryCoordinateKind.ABSOLUTE_DEPTH
                    and value.depth == depth
                )
                by_unit = {
                    unit_id: tuple(value for value in selected if value.unit_id == unit_id)
                    for unit_id in sorted({value.unit_id for value in selected})
                }
                if len(by_unit) != requested_per_family or any(
                    len(values) != 3 for values in by_unit.values()
                ):
                    raise ValueError("receiver-history resolution panel lost a requested view")
                for endpoint, attribute in (
                    ("target-decision", "target_state"),
                    ("sink-decision", "sink_state"),
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
                                    value is ReceiverHistoryScientificState.TARGETABILITY_LIMITED
                                    for value in states
                                }
                            )
                            > 1
                        )
                    disposition = (
                        ReceiverHistoryScientificState.UNEVALUABLE
                        if invalid
                        else ReceiverHistoryScientificState.TARGETABILITY_SHIFTED
                        if targetability
                        else ReceiverHistoryScientificState.RESOLUTION_SHIFTED
                        if changed
                        else ReceiverHistoryScientificState.RESOLUTION_STABLE
                    )
                    summaries.append(
                        ReceiverHistoryResolutionPanelSummary(
                            summary_id=f"resolution.{family.value}.n{scale}.k{depth}.{endpoint}",
                            family=family,
                            scale_cells=scale,
                            depth=depth,
                            endpoint_id=endpoint,
                            requested_count=requested_per_family,
                            invalid_count=invalid,
                            changed_state_count=changed,
                            targetability_changed_count=targetability,
                            disposition=disposition,
                        )
                    )
    return tuple(sorted(summaries, key=lambda value: value.summary_id))


def _untouched_cells(
    config: ReceiverHistoryConfig, bundles: tuple[ReceiverHistoryAdjudicationBundle, ...]
) -> tuple[ReceiverHistoryUntouchedPrevalenceCell, ...]:
    rows = tuple(value for bundle in bundles for value in bundle.untouched_adjudications)
    cells: list[ReceiverHistoryUntouchedPrevalenceCell] = []
    primary_tail = float(config.simultaneous_alpha) / 2.0 / 18.0
    for family in ReceiverHistoryDisorderFamily:
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
                    raise ValueError(
                        'receiver-history primary untouched cell lost its requested denominator'
                    )
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
                ) -> ReceiverHistoryScientificState:
                    if invalid:
                        return ReceiverHistoryScientificState.UNEVALUABLE
                    assert interval is not None
                    if interval[1] < config.untouched_low_rate_threshold:
                        return ReceiverHistoryScientificState.LOW_RATE_BOUNDED
                    if interval[0] > config.majority_opposition_threshold:
                        return ReceiverHistoryScientificState.MAJORITY_RECURS
                    return ReceiverHistoryScientificState.MIXED

                exemplar = coordinate_rows[0]
                cells.append(
                    ReceiverHistoryUntouchedPrevalenceCell(
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
    config: ReceiverHistoryConfig,
    bundles: tuple[ReceiverHistoryAdjudicationBundle, ...],
) -> tuple[ReceiverHistoryUntouchedDescriptiveSummary, ...]:
    rows = tuple(value for bundle in bundles for value in bundle.untouched_adjudications)
    rng = np.random.Generator(np.random.PCG64(config.untouched_bootstrap_seed))
    total_pairs = (
        config.untouched_preparation_count * (config.untouched_preparation_count - 1) / 2.0
    )
    summaries: list[ReceiverHistoryUntouchedDescriptiveSummary] = []
    for family in ReceiverHistoryDisorderFamily:
        for scale in (64, 128, 256):
            selected = tuple(
                value for value in rows if value.family is family and value.scale_cells == scale
            )
            coordinate_ids = tuple(sorted({value.coordinate_id for value in selected}))
            if len(coordinate_ids) != 16:
                raise ValueError('receiver-history untouched descriptive coordinate roster differs')
            for coordinate_id in coordinate_ids:
                coordinate_rows = tuple(
                    value for value in selected if value.coordinate_id == coordinate_id
                )
                if len(coordinate_rows) != 30:
                    raise ValueError('receiver-history untouched descriptive denominator differs')
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
                    ReceiverHistoryUntouchedDescriptiveSummary(
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
    config: ReceiverHistoryConfig,
    bundles: tuple[ReceiverHistoryAdjudicationBundle, ...],
) -> tuple[ReceiverHistoryRankObjectSummary, ...]:
    requested_per_family = _evaluation_units_per_family(config)
    summaries: list[ReceiverHistoryRankObjectSummary] = []
    for family in ReceiverHistoryDisorderFamily:
        family_bundles = tuple(
            value for value in bundles if value.adjudications[0].family is family
        )
        if len(family_bundles) != requested_per_family:
            raise ValueError("receiver-history rank summary family denominator differs")
        first_history = family_bundles[0].history_bundle
        if first_history is None:
            raise ValueError('receiver-history targeted rank summary lacks its history bundle')
        for scale in (64, 128, 256):
            coordinate_ids = tuple(
                value.coordinate_id
                for value in first_history.coordinates
                if value.scale_cells == scale
            )
            if len(coordinate_ids) != 16:
                raise ValueError("receiver-history rank summary coordinate roster differs")
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
                        raise ValueError('receiver-history targeted rank summary lacks a history bundle')
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
                    ReceiverHistoryRankObjectSummary(
                        summary_id=f"rank-summary.{family.value}.n{scale}.{coordinate_id}",
                        family=family,
                        scale_cells=scale,
                        coordinate_id=coordinate_id,
                        depth=coordinate.depth,
                        resolution_epsilon=coordinate.resolution_epsilon,
                        requested_count=requested_per_family,
                        structural_rank_median=_decimal(float(np.median(structural))),
                        discrete_lower_rank_median=_decimal(float(np.median(discrete_lower))),
                        discrete_upper_rank_median=_decimal(float(np.median(discrete_upper))),
                        effective_rank_median=_decimal(float(np.median(effective))),
                        compatible_count=sum(
                            value is ReceiverHistoryRankCompatibility.COMPATIBLE
                            for value in compatibility
                        ),
                        sampling_alias_count=sum(
                            value is ReceiverHistoryRankCompatibility.SAMPLING_ALIAS
                            for value in compatibility
                        ),
                        numerical_cancellation_count=sum(
                            value is ReceiverHistoryRankCompatibility.NUMERICAL_CANCELLATION
                            for value in compatibility
                        ),
                        unresolved_count=sum(
                            value is ReceiverHistoryRankCompatibility.RANK_UNRESOLVED
                            for value in compatibility
                        ),
                        conditioning_right_censored_count=right_censored,
                    )
                )
    return tuple(sorted(summaries, key=lambda value: value.summary_id))


def _transition_disposition(
    states: tuple[ReceiverHistoryScientificState, ...],
) -> ReceiverHistoryScientificState:
    """Apply the frozen one-way phase ordering without compensating mixed cells."""

    order = {
        ReceiverHistoryScientificState.OPPOSED: 0,
        ReceiverHistoryScientificState.TARGETABILITY_LIMITED: 1,
        ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE: 2,
    }
    if ReceiverHistoryScientificState.INVALID in states:
        return ReceiverHistoryScientificState.INVALID
    if ReceiverHistoryScientificState.MIXED in states:
        return ReceiverHistoryScientificState.MIXED
    if any(
        states[left] in order
        and states[right] in order
        and order[states[left]] > order[states[right]]
        for left in range(len(states))
        for right in range(left + 1, len(states))
    ):
        return ReceiverHistoryScientificState.NONMONOTONE_PHASE_PATTERN
    if (
        ReceiverHistoryScientificState.OPPOSED in states
        and ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE in states
    ):
        return ReceiverHistoryScientificState.SUPPORTED
    if states and all(
        value is ReceiverHistoryScientificState.TARGETABILITY_LIMITED for value in states
    ):
        return ReceiverHistoryScientificState.TARGETABILITY_LIMITED
    return ReceiverHistoryScientificState.RIGHT_CENSORED


def _transition_summaries(
    primary: tuple[ReceiverHistoryCellRecurrence, ...],
    secondary: tuple[ReceiverHistoryCellRecurrence, ...],
) -> tuple[ReceiverHistoryTransitionSummary, ...]:
    summaries: list[ReceiverHistoryTransitionSummary] = []
    all_cells = (*primary, *secondary)
    grids = {
        ReceiverHistoryCoordinateKind.ABSOLUTE_DEPTH: ("k0", "k4", "k8"),
        ReceiverHistoryCoordinateKind.NORMALIZED_BUDGET: ("b0p5", "b1"),
    }
    for family in ReceiverHistoryDisorderFamily:
        for scale in (64, 128, 256):
            for kind, labels in grids.items():
                for endpoint in ("dynamical-closure", "sink-decision", "target-decision"):
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
                            ReceiverHistoryScientificState.OPPOSED: cell.opposed_count,
                            ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE: cell.informative_nonadverse_count,
                            ReceiverHistoryScientificState.TARGETABILITY_LIMITED: cell.targetability_limited_count,
                        }
                        maximum = max(counts.values())
                        winners = tuple(key for key, value in counts.items() if value == maximum)
                        states.append(
                            ReceiverHistoryScientificState.INVALID
                            if cell.invalid_count
                            else winners[0]
                            if len(winners) == 1
                            else ReceiverHistoryScientificState.MIXED
                        )
                    order = {
                        ReceiverHistoryScientificState.OPPOSED: 0,
                        ReceiverHistoryScientificState.TARGETABILITY_LIMITED: 1,
                        ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE: 2,
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
                        if state is ReceiverHistoryScientificState.OPPOSED
                    )
                    informative_labels = tuple(
                        label
                        for label, state in zip(labels, states)
                        if state is ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
                    )
                    limited_labels = tuple(
                        sorted(
                            label
                            for label, state in zip(labels, states)
                            if state is ReceiverHistoryScientificState.TARGETABILITY_LIMITED
                        )
                    )
                    disposition = _transition_disposition(tuple(states))
                    summaries.append(
                        ReceiverHistoryTransitionSummary(
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


_ENDPOINT_ID = {
    ReceiverHistoryEndpoint.DYNAMICAL: "dynamical-closure",
    ReceiverHistoryEndpoint.TARGET_DECISION: "target-decision",
    ReceiverHistoryEndpoint.SINK_DECISION: "sink-decision",
}


def _power_transport_cells(
    *,
    result: ReceiverHistoryRecurrenceResult,
    power_atlas: ReceiverHistoryEndpointCoordinatePowerAtlas,
) -> tuple[ReceiverHistoryPowerTransportCell, ...]:
    'Compare every frozen development class with its exact held-out targeted cell.'

    recurrence = {
        (value.endpoint_id, value.family, value.scale_cells, value.coordinate_label): value
        for value in result.cells
    }
    values: list[ReceiverHistoryPowerTransportCell] = []
    for power in power_atlas.cells:
        endpoint_id = _ENDPOINT_ID[power.endpoint]
        key = (endpoint_id, power.family, power.scale_cells, power.coordinate_label)
        cell = recurrence.get(key)
        if cell is None:
            raise ValueError('receiver-history power cell lacks its exact targeted recurrence cell')
        reasons: tuple[str, ...] = ()
        counterexamples: tuple[str, ...] = ()
        endpoint_mask_invalid = (
            power.power_state is ReceiverHistoryPowerState.UNPOWERED
            and cell.prerequisite_nonattempt_count != cell.requested_count
        ) or (
            power.power_state is not ReceiverHistoryPowerState.UNPOWERED
            and cell.prerequisite_nonattempt_count != 0
        )
        if power.power_state is ReceiverHistoryPowerState.INVALID or cell.invalid_count:
            disposition = ReceiverHistoryMetatheoryDisposition.INVALID
            reasons = ('RECEIVER_HISTORY_POWER_TRANSPORT_INVALID',)
        elif endpoint_mask_invalid:
            disposition = ReceiverHistoryMetatheoryDisposition.INVALID
            reasons = ('RECEIVER_HISTORY_ENDPOINT_MASK_INVARIANT_INVALID',)
        elif power.power_state is ReceiverHistoryPowerState.UNPOWERED:
            disposition = ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
            reasons = ('RECEIVER_HISTORY_DEVELOPMENT_CELL_UNPOWERED',)
        elif power.power_state is ReceiverHistoryPowerState.WITNESS_CAPABLE:
            if cell.majority_opposition:
                disposition = ReceiverHistoryMetatheoryDisposition.SUPPORTED
            else:
                disposition = ReceiverHistoryMetatheoryDisposition.OPPOSED
                counterexamples = (cell.cell_id,)
                reasons = ('RECEIVER_HISTORY_WITNESS_POWER_DID_NOT_RECUR',)
        elif power.power_state is ReceiverHistoryPowerState.NONADVERSITY_CAPABLE:
            if cell.opposed_count == 0 and cell.majority_informative_nonadverse:
                disposition = ReceiverHistoryMetatheoryDisposition.SUPPORTED
            else:
                disposition = ReceiverHistoryMetatheoryDisposition.OPPOSED
                counterexamples = (cell.cell_id,)
                reasons = ('RECEIVER_HISTORY_NONADVERSITY_POWER_DID_NOT_RECUR',)
        else:
            if cell.opposed_count > 0 and cell.informative_nonadverse_count > 0:
                # A mixed development cell makes no promoted boundary claim; recurrence
                # of both roles is retained explicitly as not distinguished.
                disposition = ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED
            else:
                disposition = ReceiverHistoryMetatheoryDisposition.OPPOSED
                counterexamples = (cell.cell_id,)
                reasons = ('RECEIVER_HISTORY_MIXED_POWER_ROLE_DID_NOT_RECUR',)
        values.append(
            ReceiverHistoryPowerTransportCell(
                cell_id=f"transport.{endpoint_id}.{power.family.value}.n{power.scale_cells}.{power.coordinate_label}",
                power_cell_id=power.cell_id,
                recurrence_cell_id=cell.cell_id,
                endpoint_id=endpoint_id,
                family=power.family,
                scale_cells=power.scale_cells,
                coordinate_label=power.coordinate_label,
                frozen_power_state=power.power_state,
                disposition=disposition,
                counterexample_ids=counterexamples,
                reason_codes=reasons,
            )
        )
    return tuple(sorted(values, key=lambda value: value.cell_id))


def _endpoint_state(cell: ReceiverHistoryCellRecurrence) -> ReceiverHistoryScientificState:
    """Return the held-out response category independently of its power prediction."""

    if cell.invalid_count or 0 < cell.prerequisite_nonattempt_count < cell.requested_count:
        return ReceiverHistoryScientificState.INVALID
    if cell.prerequisite_nonattempt_count == cell.requested_count:
        return ReceiverHistoryScientificState.NOT_ATTEMPTED
    if cell.majority_opposition:
        return ReceiverHistoryScientificState.OPPOSED
    if cell.opposed_count == 0 and cell.majority_informative_nonadverse:
        return ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
    if cell.targetability_limited_count > cell.requested_count // 2:
        return ReceiverHistoryScientificState.TARGETABILITY_LIMITED
    return ReceiverHistoryScientificState.MIXED


_COORDINATE_SELECTION_ORDER = {
    label: index for index, label in enumerate(("k0", "b0p5", "k4", "b1", "k8"))
}


def _transport_depth(value: ReceiverHistoryPowerTransportCell) -> int:
    if value.coordinate_label.startswith("k"):
        return int(value.coordinate_label[1:])
    if value.coordinate_label == "b0p5":
        return {64: 3, 128: 7, 256: 15}[value.scale_cells]
    if value.coordinate_label == "b1":
        return {64: 7, 128: 15, 256: 31}[value.scale_cells]
    raise ValueError("receiver-history locality coordinate label differs")


def _select_locality_coordinate(
    cells: tuple[ReceiverHistoryPowerTransportCell, ...],
) -> tuple[ReceiverHistoryPowerTransportCell, ...]:
    'Freeze the best development-only coordinate without consulting targeted outcomes.'

    by_label = {
        label: tuple(value for value in cells if value.coordinate_label == label)
        for label in sorted({value.coordinate_label for value in cells})
    }
    if not by_label:
        return ()

    def selection_key(
        item: tuple[str, tuple[ReceiverHistoryPowerTransportCell, ...]],
    ) -> tuple[int, int, int, int, int, int, int]:
        label, selected = item
        states = tuple(value.frozen_power_state for value in selected)
        return (
            sum(value is ReceiverHistoryPowerState.INVALID for value in states),
            -sum(value is ReceiverHistoryPowerState.NONADVERSITY_CAPABLE for value in states),
            sum(value is ReceiverHistoryPowerState.UNPOWERED for value in states),
            sum(value is ReceiverHistoryPowerState.MIXED_CAPABLE for value in states),
            sum(value is ReceiverHistoryPowerState.WITNESS_CAPABLE for value in states),
            sum(_transport_depth(value) for value in selected),
            _COORDINATE_SELECTION_ORDER[label],
        )

    _label, selected = min(by_label.items(), key=selection_key)
    return selected


def _locality_assessment(
    *,
    tournament_id: str,
    candidate: ReceiverHistoryLocalityCandidate,
    cells: tuple[ReceiverHistoryPowerTransportCell, ...],
    coordinate_labels: tuple[str, ...],
    recurrence_by_id: dict[str, ReceiverHistoryCellRecurrence],
) -> ReceiverHistoryLocalityCandidateAssessment:
    'Test a frozen closure candidate, rather than reuse development-power transport success.\n\n    Witness recurrence supports the power instrument but is a held-out\n    counterexample to a closure candidate.  Only a prospectively nominated\n    nonadversity role followed by held-out informative nonadversity supports\n    candidate survival.\n    '

    invalid: list[str] = []
    opposed: list[str] = []
    limited: list[str] = []
    supported = 0
    for value in cells:
        state = _endpoint_state(recurrence_by_id[value.recurrence_cell_id])
        if (
            value.disposition is ReceiverHistoryMetatheoryDisposition.INVALID
            or state is ReceiverHistoryScientificState.INVALID
        ):
            invalid.append(value.recurrence_cell_id)
        elif value.frozen_power_state is ReceiverHistoryPowerState.UNPOWERED:
            limited.append(value.recurrence_cell_id)
        elif (
            value.frozen_power_state is ReceiverHistoryPowerState.NONADVERSITY_CAPABLE
            and state is ReceiverHistoryScientificState.INFORMATIVE_NONADVERSE
        ):
            supported += 1
        elif state in {
            ReceiverHistoryScientificState.OPPOSED,
            ReceiverHistoryScientificState.MIXED,
        }:
            opposed.append(value.recurrence_cell_id)
        else:
            # A failed development role cannot be relabelled as closure after
            # reveal, and a targetability-limited response is not a falsifier.
            limited.append(value.recurrence_cell_id)
    if invalid:
        disposition = ReceiverHistoryMetatheoryDisposition.INVALID
        reasons: tuple[str, ...] = ('RECEIVER_HISTORY_LOCALITY_CANDIDATE_INVALID',)
    elif opposed:
        disposition = ReceiverHistoryMetatheoryDisposition.OPPOSED
        reasons = ('RECEIVER_HISTORY_LOCALITY_CANDIDATE_HELDOUT_COUNTEREXAMPLE',)
    elif limited or not cells:
        disposition = ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
        reasons = ('RECEIVER_HISTORY_LOCALITY_CANDIDATE_NOT_PROSPECTIVELY_TESTABLE',)
    else:
        if supported != len(cells):
            raise ValueError("receiver-history locality assessment accounting differs")
        disposition = ReceiverHistoryMetatheoryDisposition.SUPPORTED
        reasons = ()
    return ReceiverHistoryLocalityCandidateAssessment(
        assessment_id=f"assessment.{tournament_id}.{candidate.value.lower()}",
        candidate=candidate,
        coordinate_labels=coordinate_labels,
        disposition=disposition,
        counterexample_cell_ids=tuple(sorted(opposed)),
        reason_codes=reasons,
    )


def _action_only_assessment(
    *,
    tournament_id: str,
    k0: tuple[ReceiverHistoryPowerTransportCell, ...],
    recurrence_by_id: dict[str, ReceiverHistoryCellRecurrence],
) -> ReceiverHistoryLocalityCandidateAssessment:
    'Use K0 opposition as a falsifier, never as positive action-only identification.'

    assessment = _locality_assessment(
        tournament_id=tournament_id,
        candidate=ReceiverHistoryLocalityCandidate.ACTION_ONLY,
        cells=k0,
        coordinate_labels=(),
        recurrence_by_id=recurrence_by_id,
    )
    if assessment.disposition is not ReceiverHistoryMetatheoryDisposition.SUPPORTED:
        return assessment
    return replace(
        assessment,
        disposition=ReceiverHistoryMetatheoryDisposition.UNEVALUABLE,
        reason_codes=('RECEIVER_HISTORY_ACTION_ONLY_NOT_IDENTIFIED_BY_R8_MATCHED_CHALLENGES',),
    )


def _locality_tournaments(
    transport: tuple[ReceiverHistoryPowerTransportCell, ...],
    result: ReceiverHistoryRecurrenceResult,
) -> tuple[ReceiverHistoryLocalityTournamentResult, ...]:
    recurrence_by_id = {value.cell_id: value for value in result.cells}
    by_key = {
        (value.endpoint_id, value.family, value.scale_cells, value.coordinate_label): value
        for value in transport
    }
    tournaments: list[ReceiverHistoryLocalityTournamentResult] = []
    endpoint_ids = tuple(sorted(set(value.endpoint_id for value in transport)))
    for endpoint_id in endpoint_ids:
        endpoint_cells = tuple(value for value in transport if value.endpoint_id == endpoint_id)
        global_fixed = _select_locality_coordinate(
            tuple(value for value in endpoint_cells if value.coordinate_label in {"k4", "k8"}),
        )
        global_budget = _select_locality_coordinate(
            tuple(value for value in endpoint_cells if value.coordinate_label in {"b0p5", "b1"}),
        )
        if len(global_fixed) != 9 or len(global_budget) != 9:
            raise ValueError("receiver-history global locality candidate roster differs")
        fixed_label = global_fixed[0].coordinate_label
        budget_label = global_budget[0].coordinate_label
        if any(value.coordinate_label != fixed_label for value in global_fixed) or any(
            value.coordinate_label != budget_label for value in global_budget
        ):
            raise ValueError("receiver-history global locality coordinate selection differs")
        for family in ReceiverHistoryDisorderFamily:
            for scale in (64, 128, 256):
                tournament_id = f"tournament.{endpoint_id}.{family.value}.n{scale}"

                def selected(
                    labels: tuple[str, ...],
                ) -> tuple[ReceiverHistoryPowerTransportCell, ...]:
                    return tuple(by_key[(endpoint_id, family, scale, label)] for label in labels)

                context_cells = tuple(
                    value
                    for value in transport
                    if value.endpoint_id == endpoint_id
                    and value.family is family
                    and value.scale_cells == scale
                )
                contextual_selection = _select_locality_coordinate(context_cells)
                if len(context_cells) != 5 or len(contextual_selection) != 1:
                    raise ValueError(
                        "receiver-history contextual locality candidate roster differs"
                    )
                assessments = (
                    _action_only_assessment(
                        tournament_id=tournament_id,
                        k0=selected(("k0",)),
                        recurrence_by_id=recurrence_by_id,
                    ),
                    _locality_assessment(
                        tournament_id=tournament_id,
                        candidate=ReceiverHistoryLocalityCandidate.EIGHT_BIN_RECEIVER_AND_ACTION,
                        cells=selected(("k0",)),
                        coordinate_labels=("k0",),
                        recurrence_by_id=recurrence_by_id,
                    ),
                    _locality_assessment(
                        tournament_id=tournament_id,
                        candidate=ReceiverHistoryLocalityCandidate.FIXED_HISTORY_DEPTH,
                        cells=selected((fixed_label,)),
                        coordinate_labels=(fixed_label,),
                        recurrence_by_id=recurrence_by_id,
                    ),
                    _locality_assessment(
                        tournament_id=tournament_id,
                        candidate=ReceiverHistoryLocalityCandidate.NORMALIZED_HISTORY_BUDGET,
                        cells=selected((budget_label,)),
                        coordinate_labels=(budget_label,),
                        recurrence_by_id=recurrence_by_id,
                    ),
                    _locality_assessment(
                        tournament_id=tournament_id,
                        candidate=ReceiverHistoryLocalityCandidate.ENDPOINT_CONTEXT_ATLAS,
                        cells=contextual_selection,
                        coordinate_labels=(contextual_selection[0].coordinate_label,),
                        recurrence_by_id=recurrence_by_id,
                    ),
                )
                survivors = tuple(
                    value.candidate
                    for value in assessments
                    if value.disposition is ReceiverHistoryMetatheoryDisposition.SUPPORTED
                )
                contextual_assessment = assessments[-1]
                reasons: tuple[str, ...]
                if any(
                    value.disposition is ReceiverHistoryMetatheoryDisposition.INVALID
                    for value in context_cells
                ) or any(
                    value.disposition is ReceiverHistoryMetatheoryDisposition.INVALID
                    for value in assessments
                ):
                    winner = None
                    disposition = ReceiverHistoryMetatheoryDisposition.INVALID
                    reasons = ('RECEIVER_HISTORY_LOCALITY_TOURNAMENT_INVALID',)
                elif (
                    contextual_assessment.disposition
                    is ReceiverHistoryMetatheoryDisposition.OPPOSED
                ):
                    winner = survivors[0] if survivors else None
                    disposition = ReceiverHistoryMetatheoryDisposition.OPPOSED
                    reasons = ('RECEIVER_HISTORY_CONTEXTUAL_CANDIDATE_OPPOSED',)
                elif contextual_assessment.disposition in {
                    ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED,
                    ReceiverHistoryMetatheoryDisposition.UNEVALUABLE,
                }:
                    winner = survivors[0] if survivors else None
                    disposition = ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
                    reasons = ('RECEIVER_HISTORY_CONTEXTUAL_CANDIDATE_INCOMPLETE',)
                elif (
                    contextual_assessment.disposition
                    is ReceiverHistoryMetatheoryDisposition.SUPPORTED
                    and all(
                        value.disposition is ReceiverHistoryMetatheoryDisposition.OPPOSED
                        for value in assessments[:-1]
                    )
                ):
                    winner = ReceiverHistoryLocalityCandidate.ENDPOINT_CONTEXT_ATLAS
                    disposition = ReceiverHistoryMetatheoryDisposition.SUPPORTED
                    reasons = ()
                elif any(
                    value.disposition is ReceiverHistoryMetatheoryDisposition.SUPPORTED
                    for value in assessments[:-1]
                ):
                    winner = survivors[0]
                    disposition = ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED
                    reasons = ('RECEIVER_HISTORY_SIMPLER_CANDIDATE_EQUALLY_EXACT',)
                elif any(
                    value.disposition
                    in {
                        ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED,
                        ReceiverHistoryMetatheoryDisposition.UNEVALUABLE,
                    }
                    for value in assessments
                ):
                    winner = None
                    disposition = ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
                    reasons = ('RECEIVER_HISTORY_LOCALITY_TOURNAMENT_INCOMPLETE',)
                else:
                    winner = None
                    disposition = ReceiverHistoryMetatheoryDisposition.OPPOSED
                    reasons = ('RECEIVER_HISTORY_NO_LOCALITY_CANDIDATE_SURVIVED',)
                tournaments.append(
                    ReceiverHistoryLocalityTournamentResult(
                        tournament_id=tournament_id,
                        endpoint_id=endpoint_id,
                        family=family,
                        scale_cells=scale,
                        assessments=assessments,
                        winning_candidate=winner,
                        disposition=disposition,
                        reason_codes=reasons,
                    )
                )
    return tuple(sorted(tournaments, key=lambda value: value.tournament_id))


def _meta_from_scientific(
    value: ReceiverHistoryScientificState,
) -> ReceiverHistoryMetatheoryDisposition:
    return {
        ReceiverHistoryScientificState.SUPPORTED: ReceiverHistoryMetatheoryDisposition.SUPPORTED,
        ReceiverHistoryScientificState.OPPOSED: ReceiverHistoryMetatheoryDisposition.OPPOSED,
        ReceiverHistoryScientificState.TARGETABILITY_LIMITED: ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED,
        ReceiverHistoryScientificState.INVALID: ReceiverHistoryMetatheoryDisposition.INVALID,
        ReceiverHistoryScientificState.UNEVALUABLE: ReceiverHistoryMetatheoryDisposition.UNEVALUABLE,
        ReceiverHistoryScientificState.ABSOLUTE_DEPTH_ALIGNED: ReceiverHistoryMetatheoryDisposition.SUPPORTED,
        ReceiverHistoryScientificState.NORMALIZED_BUDGET_ALIGNED: ReceiverHistoryMetatheoryDisposition.SUPPORTED,
    }.get(value, ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED)


def _hypotheses(
    *,
    result: ReceiverHistoryRecurrenceResult,
    transport: tuple[ReceiverHistoryPowerTransportCell, ...],
    tournaments: tuple[ReceiverHistoryLocalityTournamentResult, ...],
) -> tuple[ReceiverHistoryHypothesisDisposition, ...]:
    powered = tuple(
        value
        for value in transport
        if value.frozen_power_state
        not in {ReceiverHistoryPowerState.UNPOWERED, ReceiverHistoryPowerState.INVALID}
    )
    invalid = tuple(
        value.cell_id
        for value in transport
        if value.disposition is ReceiverHistoryMetatheoryDisposition.INVALID
    )
    opposed = tuple(
        value.cell_id
        for value in powered
        if value.disposition is ReceiverHistoryMetatheoryDisposition.OPPOSED
    )
    if invalid:
        h1_disposition = ReceiverHistoryMetatheoryDisposition.INVALID
        h1_limitations: tuple[str, ...] = ("invalid-power-transport-cell",)
    elif opposed:
        h1_disposition = ReceiverHistoryMetatheoryDisposition.OPPOSED
        h1_limitations = ()
    elif powered:
        h1_disposition = ReceiverHistoryMetatheoryDisposition.SUPPORTED
        h1_limitations = ()
    else:
        h1_disposition = ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
        h1_limitations = ("no-powered-evaluation-cell",)

    def bracket_hypothesis(
        hypothesis_id: str,
        endpoint_states: tuple[tuple[str, ReceiverHistoryScientificState], ...],
    ) -> ReceiverHistoryHypothesisDisposition:
        # An endpoint excluded by the frozen power atlas is not evidence
        # against a bracket that is prospectively supported in another powered
        # endpoint.  It is omitted from this bracket aggregation, while a
        # complete absence of eligible endpoints remains targetability-limited.
        entered_states = tuple(
            (endpoint_id, value)
            for endpoint_id, value in endpoint_states
            if value is not ReceiverHistoryScientificState.NOT_ATTEMPTED
        )
        entered = tuple(_meta_from_scientific(value) for _endpoint_id, value in entered_states)
        if any(value is ReceiverHistoryMetatheoryDisposition.INVALID for value in entered):
            disposition = ReceiverHistoryMetatheoryDisposition.INVALID
        elif any(value is ReceiverHistoryMetatheoryDisposition.OPPOSED for value in entered):
            disposition = ReceiverHistoryMetatheoryDisposition.OPPOSED
        elif not entered:
            disposition = ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
        elif all(value is ReceiverHistoryMetatheoryDisposition.SUPPORTED for value in entered):
            disposition = ReceiverHistoryMetatheoryDisposition.SUPPORTED
        elif any(
            value is ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED for value in entered
        ):
            disposition = ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
        elif any(value is ReceiverHistoryMetatheoryDisposition.UNEVALUABLE for value in entered):
            disposition = ReceiverHistoryMetatheoryDisposition.UNEVALUABLE
        else:
            disposition = ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED
        shallow, deep = ("k4", "k8") if hypothesis_id == 'fixed-absolute-depth-bracket' else ("b0p5", "b1")
        opposed_endpoints = {
            endpoint_id
            for endpoint_id, value in entered_states
            if value is ReceiverHistoryScientificState.OPPOSED
        }
        counter = tuple(
            value.cell_id
            for value in result.cells
            if value.endpoint_id in opposed_endpoints
            and (
                (value.coordinate_label == shallow and not value.majority_opposition)
                or (
                    value.coordinate_label == deep
                    and not (value.opposed_count == 0 and value.majority_informative_nonadverse)
                )
            )
        )
        limitations = (
            (f"{hypothesis_id.lower()}-unpowered-or-invalid",)
            if disposition
            in {
                ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED,
                ReceiverHistoryMetatheoryDisposition.INVALID,
                ReceiverHistoryMetatheoryDisposition.UNEVALUABLE,
            }
            else ()
        )
        return ReceiverHistoryHypothesisDisposition(
            hypothesis_id=hypothesis_id,
            disposition=disposition,
            evidence_ids=tuple(
                sorted(
                    value.summary_id
                    for value in result.transition_summaries
                    if value.endpoint_id in {endpoint_id for endpoint_id, _value in entered_states}
                    if value.coordinate_kind
                    is (
                        ReceiverHistoryCoordinateKind.ABSOLUTE_DEPTH
                        if hypothesis_id == 'fixed-absolute-depth-bracket'
                        else ReceiverHistoryCoordinateKind.NORMALIZED_BUDGET
                    )
                )
            ),
            counterexample_ids=tuple(sorted(counter)),
            limitation_ids=limitations,
        )

    h2 = bracket_hypothesis(
        'fixed-absolute-depth-bracket',
        (
            ("dynamical-closure", result.fixed_depth_dynamical_bracket),
            ("target-decision", result.fixed_depth_target_bracket),
            ("sink-decision", result.fixed_depth_sink_bracket),
        ),
    )
    h3 = bracket_hypothesis(
        'normalized-budget-bracket',
        (
            ("dynamical-closure", result.normalized_budget_dynamical_bracket),
            ("target-decision", result.normalized_budget_target_bracket),
            ("sink-decision", result.normalized_budget_sink_bracket),
        ),
    )
    alignment_states = tuple(value.disposition for value in result.alignment_summaries)
    evaluable_alignment = tuple(
        value
        for value in alignment_states
        if value
        in {
            ReceiverHistoryScientificState.ABSOLUTE_DEPTH_ALIGNED,
            ReceiverHistoryScientificState.NORMALIZED_BUDGET_ALIGNED,
            ReceiverHistoryScientificState.NOT_DISTINGUISHED,
        }
    )
    aligned = {
        value
        for value in evaluable_alignment
        if value
        in {
            ReceiverHistoryScientificState.ABSOLUTE_DEPTH_ALIGNED,
            ReceiverHistoryScientificState.NORMALIZED_BUDGET_ALIGNED,
        }
    }
    h4_disposition = (
        ReceiverHistoryMetatheoryDisposition.INVALID
        if ReceiverHistoryScientificState.INVALID in alignment_states
        else ReceiverHistoryMetatheoryDisposition.UNEVALUABLE
        if not evaluable_alignment
        else ReceiverHistoryMetatheoryDisposition.SUPPORTED
        if len(aligned) == 1 and all(value in aligned for value in evaluable_alignment)
        else ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED
    )
    contextual_wins = tuple(
        value.tournament_id
        for value in tournaments
        if value.disposition is ReceiverHistoryMetatheoryDisposition.SUPPORTED
    )
    opposed_tournaments = tuple(
        value.tournament_id
        for value in tournaments
        if value.disposition is ReceiverHistoryMetatheoryDisposition.OPPOSED
    )
    invalid_tournaments = tuple(
        value.tournament_id
        for value in tournaments
        if value.disposition is ReceiverHistoryMetatheoryDisposition.INVALID
    )
    h5_disposition = (
        ReceiverHistoryMetatheoryDisposition.INVALID
        if invalid_tournaments
        else ReceiverHistoryMetatheoryDisposition.SUPPORTED
        if contextual_wins
        else ReceiverHistoryMetatheoryDisposition.OPPOSED
        if len(opposed_tournaments) == len(tournaments)
        else ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
        if all(
            value.disposition is ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
            for value in tournaments
        )
        else ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED
    )
    recurrence_by_id = {value.cell_id: value for value in result.cells}
    differing: list[str] = []
    comparable_endpoint_groups = 0
    invalid_endpoint_cells: list[str] = []
    for family in ReceiverHistoryDisorderFamily:
        for scale in (64, 128, 256):
            for label in ("k0", "k4", "k8", "b0p5", "b1"):
                selected = tuple(
                    value
                    for value in transport
                    if value.family is family
                    and value.scale_cells == scale
                    and value.coordinate_label == label
                )
                invalid_endpoint_cells.extend(
                    value.cell_id
                    for value in selected
                    if value.disposition is ReceiverHistoryMetatheoryDisposition.INVALID
                )
                entered = tuple(
                    (value, _endpoint_state(recurrence_by_id[value.recurrence_cell_id]))
                    for value in selected
                    if value.frozen_power_state
                    not in {ReceiverHistoryPowerState.INVALID, ReceiverHistoryPowerState.UNPOWERED}
                )
                if len(entered) >= 2:
                    comparable_endpoint_groups += 1
                    if len({state for _value, state in entered}) > 1:
                        differing.extend(value.recurrence_cell_id for value, _state in entered)
    h6_disposition = (
        ReceiverHistoryMetatheoryDisposition.INVALID
        if invalid_endpoint_cells
        else ReceiverHistoryMetatheoryDisposition.SUPPORTED
        if differing
        else ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
        if not comparable_endpoint_groups
        else ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED
    )
    return (
        ReceiverHistoryHypothesisDisposition(
            hypothesis_id='development-power-transport',
            disposition=h1_disposition,
            evidence_ids=tuple(value.cell_id for value in powered),
            counterexample_ids=opposed,
            limitation_ids=h1_limitations,
        ),
        h2,
        h3,
        ReceiverHistoryHypothesisDisposition(
            hypothesis_id='paired-coordinate-alignment',
            disposition=h4_disposition,
            evidence_ids=tuple(value.summary_id for value in result.alignment_summaries),
            counterexample_ids=(),
            limitation_ids=("invalid-alignment-cell",)
            if h4_disposition is ReceiverHistoryMetatheoryDisposition.INVALID
            else ("alignment-cells-insufficient",)
            if h4_disposition is ReceiverHistoryMetatheoryDisposition.UNEVALUABLE
            else (),
        ),
        ReceiverHistoryHypothesisDisposition(
            hypothesis_id='selective-contextual-locality',
            disposition=h5_disposition,
            evidence_ids=contextual_wins,
            counterexample_ids=opposed_tournaments
            if h5_disposition is ReceiverHistoryMetatheoryDisposition.OPPOSED
            else (),
            limitation_ids=("simpler-candidate-not-excluded",)
            if h5_disposition
            in {
                ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED,
                ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED,
            }
            else ("invalid-locality-tournament",)
            if h5_disposition is ReceiverHistoryMetatheoryDisposition.INVALID
            else (),
        ),
        ReceiverHistoryHypothesisDisposition(
            hypothesis_id='typed-endpoint-dependence',
            disposition=h6_disposition,
            evidence_ids=tuple(sorted(set(differing))),
            counterexample_ids=(),
            limitation_ids=("invalid-endpoint-transport-cell",)
            if h6_disposition is ReceiverHistoryMetatheoryDisposition.INVALID
            else ("endpoint-fibres-not-distinguished",)
            if h6_disposition
            in {
                ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED,
                ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED,
            }
            else (),
        ),
    )


def build_inference_disposition_matrix(
    *,
    result: ReceiverHistoryRecurrenceResult,
    power_atlas: ReceiverHistoryEndpointCoordinatePowerAtlas,
) -> ReceiverHistoryInferenceDispositionMatrix:
    if result.power_atlas_sha256 != power_atlas.fingerprint():
        raise ValueError('receiver-history phase-diagram synthesis matrix power-atlas identity differs')
    transport = _power_transport_cells(result=result, power_atlas=power_atlas)
    tournaments = _locality_tournaments(transport, result)
    return ReceiverHistoryInferenceDispositionMatrix(
        matrix_id="receiver-history.inference-disposition-matrix",
        recurrence_result_sha256=result.fingerprint(),
        power_atlas_sha256=power_atlas.fingerprint(),
        power_transport_cells=transport,
        locality_tournaments=tournaments,
        hypotheses=_hypotheses(result=result, transport=transport, tournaments=tournaments),
    )


def _metatheory_nonclaims() -> tuple[ReceiverHistoryNonclaimDisposition, ...]:
    return tuple(
        ReceiverHistoryNonclaimDisposition(
            nonclaim_id=f"nc{index:02d}",
            disposition=ReceiverHistoryMetatheoryDisposition.EXCLUDED,
            reason_id=reason,
        )
        for index, reason in enumerate(_NONCLAIM_REASONS, start=1)
    )


def build_development_metatheory_gate(
    *,
    power_atlas: ReceiverHistoryEndpointCoordinatePowerAtlas,
    method_freeze_sha256: str,
    evaluation_design_sha256: str,
    source_closure_sha256: str,
    capability_registry_sha256: str,
    execution_plan_sha256: str,
) -> ReceiverHistoryDevelopmentMetatheoryGate:
    'Freeze the development branch, including a complete no-power empirical metatheory outcome.'

    eligible = power_atlas.eligible_cell_ids
    all_unpowered = not eligible
    empirical_disposition = (
        ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
        if all_unpowered
        else ReceiverHistoryMetatheoryDisposition.NOT_ATTEMPTED
    )
    empirical_reason = (
        "all-development-endpoint-coordinate-cells-unpowered"
        if all_unpowered
        else "sealed-evaluation-pending"
    )

    def proposition(
        proposition_id: str,
        hypothesis_ids: tuple[str, ...],
        *,
        procedural: bool = False,
    ) -> ReceiverHistoryMetatheoryPropositionDisposition:
        return ReceiverHistoryMetatheoryPropositionDisposition(
            proposition_id=proposition_id,
            disposition=(
                ReceiverHistoryMetatheoryDisposition.SUPPORTED
                if procedural
                else empirical_disposition
            ),
            hypothesis_ids=hypothesis_ids,
            evidence_ids=(
                ("frozen-development-evaluation-boundary",)
                if procedural
                else (power_atlas.fingerprint(),)
            ),
            limitation_ids=(
                ("procedural-separation-not-physical-response-claim",)
                if procedural
                else (empirical_reason,)
            ),
        )

    propositions = (
        proposition('contextual-lawhood', ('selective-contextual-locality',)),
        proposition('development-power-transport', ('development-power-transport',)),
        proposition('endpoint-relative-lawhood', ('typed-endpoint-dependence',)),
        proposition('evidence-gate-separation', (), procedural=True),
        proposition('history-coordinate-bounds', ('fixed-absolute-depth-bracket', 'normalized-budget-bracket', 'paired-coordinate-alignment')),
        proposition('prospective-discipline', (), procedural=True),
    )
    qualification_gate_disposition = (
        ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
        if all_unpowered
        else ReceiverHistoryMetatheoryDisposition.NOT_ATTEMPTED
    )
    evidence_gates = (
        ReceiverHistoryEvidenceGateDisposition(
            'measurement-provenance',
            ReceiverHistoryMetatheoryDisposition.SUPPORTED,
            EvidenceCeiling.LOCAL_LAW,
            "denominator-provenance-and-unit-bound",
        ),
        ReceiverHistoryEvidenceGateDisposition(
            'observable-coordinate', qualification_gate_disposition, EvidenceCeiling.LOCAL_LAW, empirical_reason
        ),
        ReceiverHistoryEvidenceGateDisposition(
            'response-identification', qualification_gate_disposition, EvidenceCeiling.LOCAL_LAW, empirical_reason
        ),
        ReceiverHistoryEvidenceGateDisposition(
            'law-qualification', qualification_gate_disposition, EvidenceCeiling.LOCAL_LAW, empirical_reason
        ),
        ReceiverHistoryEvidenceGateDisposition(
            'admission',
            ReceiverHistoryMetatheoryDisposition.NOT_ATTEMPTED,
            EvidenceCeiling.LOCAL_LAW,
            "admission-and-reachability-not-attempted",
        ),
        ReceiverHistoryEvidenceGateDisposition(
            'prospective-use',
            ReceiverHistoryMetatheoryDisposition.NOT_ATTEMPTED,
            EvidenceCeiling.LOCAL_LAW,
            "controller-validation-not-attempted",
        ),
    )
    return ReceiverHistoryDevelopmentMetatheoryGate(
        gate_id="receiver-history.development-metatheory-gate",
        metatheory_id='receiver-history-empirical-metatheory',
        power_atlas_sha256=power_atlas.fingerprint(),
        method_freeze_sha256=method_freeze_sha256,
        evaluation_design_sha256=evaluation_design_sha256,
        source_closure_sha256=source_closure_sha256,
        capability_registry_sha256=capability_registry_sha256,
        execution_plan_sha256=execution_plan_sha256,
        eligible_cell_ids=eligible,
        all_endpoints_unpowered=all_unpowered,
        issue_evaluation=not all_unpowered,
        terminal=(ReceiverHistoryTerminal.ALL_ENDPOINTS_UNPOWERED if all_unpowered else None),
        propositions=propositions,
        nonclaims=_metatheory_nonclaims(),
        evidence_gates=evidence_gates,
        excluded_experiment_ids=_EXCLUDED_EXPERIMENT_IDS,
        limitation_ids=_METATHEORY_LIMITATION_IDS,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        physical_claim_allowed=False,
        controller_claim_allowed=False,
    )


def build_empirical_metatheory_dossier(
    *,
    result: ReceiverHistoryRecurrenceResult,
    matrix: ReceiverHistoryInferenceDispositionMatrix,
    source_closure_sha256: str,
    capability_registry_sha256: str,
    execution_plan_sha256: str,
) -> ReceiverHistoryEmpiricalMetatheoryDossier:
    'Map the frozen phase-diagram synthesis evidence to the six metatheory propositions, explicit nonclaims, and measurement through prospective-use gates.'

    if matrix.recurrence_result_sha256 != result.fingerprint():
        raise ValueError('empirical metatheory matrix/result identity differs')
    hypotheses = {value.hypothesis_id: value for value in matrix.hypotheses}

    def proposition(
        proposition_id: str,
        disposition: ReceiverHistoryMetatheoryDisposition,
        hypothesis_ids: tuple[str, ...],
        evidence_ids: tuple[str, ...],
        limitation_ids: tuple[str, ...] = (),
    ) -> ReceiverHistoryMetatheoryPropositionDisposition:
        return ReceiverHistoryMetatheoryPropositionDisposition(
            proposition_id=proposition_id,
            disposition=disposition,
            hypothesis_ids=hypothesis_ids,
            evidence_ids=tuple(sorted(set(evidence_ids))),
            limitation_ids=tuple(sorted(set(limitation_ids))),
        )

    h5 = hypotheses['selective-contextual-locality']
    h1 = hypotheses['development-power-transport']
    h6 = hypotheses['typed-endpoint-dependence']
    bracket_dispositions = tuple(hypotheses[value].disposition for value in ('fixed-absolute-depth-bracket', 'normalized-budget-bracket'))
    transport_disposition = (
        ReceiverHistoryMetatheoryDisposition.INVALID
        if any(
            value is ReceiverHistoryMetatheoryDisposition.INVALID for value in bracket_dispositions
        )
        or hypotheses['paired-coordinate-alignment'].disposition is ReceiverHistoryMetatheoryDisposition.INVALID
        else ReceiverHistoryMetatheoryDisposition.SUPPORTED
        if any(
            value is ReceiverHistoryMetatheoryDisposition.SUPPORTED
            for value in bracket_dispositions
        )
        else ReceiverHistoryMetatheoryDisposition.OPPOSED
        if all(
            value is ReceiverHistoryMetatheoryDisposition.OPPOSED for value in bracket_dispositions
        )
        else ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
        if all(
            value is ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
            for value in bracket_dispositions
        )
        else ReceiverHistoryMetatheoryDisposition.UNEVALUABLE
        if any(
            value is ReceiverHistoryMetatheoryDisposition.UNEVALUABLE
            for value in bracket_dispositions
        )
        else ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED
    )
    transport_limitations = tuple(
        sorted(
            {
                limitation
                for hypothesis_id in ('fixed-absolute-depth-bracket', 'normalized-budget-bracket', 'paired-coordinate-alignment')
                for limitation in hypotheses[hypothesis_id].limitation_ids
            }
        )
    )
    propositions = (
        proposition('contextual-lawhood', h5.disposition, ('selective-contextual-locality',), h5.evidence_ids, h5.limitation_ids),
        proposition('development-power-transport', h1.disposition, ('development-power-transport',), h1.evidence_ids, h1.limitation_ids),
        proposition('endpoint-relative-lawhood', h6.disposition, ('typed-endpoint-dependence',), h6.evidence_ids, h6.limitation_ids),
        proposition(
            'evidence-gate-separation',
            ReceiverHistoryMetatheoryDisposition.SUPPORTED,
            (),
            ('law-qualification-ceiling-with-admission-and-prospective-use-not-attempted',),
            ("procedural-separation-not-physical-response-claim",),
        ),
        proposition(
            'history-coordinate-bounds',
            transport_disposition,
            ('fixed-absolute-depth-bracket', 'normalized-budget-bracket', 'paired-coordinate-alignment'),
            tuple(
                evidence
                for hypothesis_id in ('fixed-absolute-depth-bracket', 'normalized-budget-bracket', 'paired-coordinate-alignment')
                for evidence in hypotheses[hypothesis_id].evidence_ids
            ),
            transport_limitations,
        ),
        proposition(
            'prospective-discipline',
            ReceiverHistoryMetatheoryDisposition.SUPPORTED,
            (),
            (matrix.matrix_id, "frozen-development-evaluation-boundary"),
            ("procedural-prospective-discipline-not-a-response-law",),
        ),
    )
    nonclaims = _metatheory_nonclaims()
    empirical_dispositions = tuple(
        value.disposition
        for value in propositions
        if value.proposition_id in {'contextual-lawhood', 'development-power-transport', 'endpoint-relative-lawhood', 'history-coordinate-bounds'}
    )
    law_dispositions = tuple(
        value.disposition for value in propositions if value.proposition_id in {'contextual-lawhood', 'endpoint-relative-lawhood', 'history-coordinate-bounds'}
    )
    law_qualification_disposition = (
        ReceiverHistoryMetatheoryDisposition.INVALID
        if any(
            value is ReceiverHistoryMetatheoryDisposition.INVALID
            for value in empirical_dispositions
        )
        else ReceiverHistoryMetatheoryDisposition.SUPPORTED
        if any(
            value is ReceiverHistoryMetatheoryDisposition.SUPPORTED for value in law_dispositions
        )
        else ReceiverHistoryMetatheoryDisposition.OPPOSED
        if all(value is ReceiverHistoryMetatheoryDisposition.OPPOSED for value in law_dispositions)
        else ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
        if all(
            value is ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED
            for value in law_dispositions
        )
        else ReceiverHistoryMetatheoryDisposition.UNEVALUABLE
        if any(
            value is ReceiverHistoryMetatheoryDisposition.UNEVALUABLE for value in law_dispositions
        )
        else ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED
    )
    response_disposition = (
        ReceiverHistoryMetatheoryDisposition.INVALID
        if any(value.invalid_count for value in result.cells)
        or any(
            value.disposition is ReceiverHistoryMetatheoryDisposition.INVALID
            for value in matrix.power_transport_cells
        )
        else ReceiverHistoryMetatheoryDisposition.SUPPORTED
    )
    law_qualification_reason = {
        ReceiverHistoryMetatheoryDisposition.SUPPORTED: "prospective-local-law-support-observed",
        ReceiverHistoryMetatheoryDisposition.OPPOSED: "prospective-local-law-candidates-opposed",
        ReceiverHistoryMetatheoryDisposition.TARGETABILITY_LIMITED: "prospective-local-law-targetability-limited",
        ReceiverHistoryMetatheoryDisposition.UNEVALUABLE: "prospective-local-law-test-unevaluable",
        ReceiverHistoryMetatheoryDisposition.INVALID: "prospective-local-law-test-invalid",
        ReceiverHistoryMetatheoryDisposition.NOT_DISTINGUISHED: "prospective-local-law-not-distinguished",
    }[law_qualification_disposition]
    evidence_gates = (
        ReceiverHistoryEvidenceGateDisposition(
            'measurement-provenance',
            ReceiverHistoryMetatheoryDisposition.SUPPORTED,
            EvidenceCeiling.LOCAL_LAW,
            "denominator-provenance-and-unit-bound",
        ),
        ReceiverHistoryEvidenceGateDisposition(
            'observable-coordinate', h5.disposition, EvidenceCeiling.LOCAL_LAW, "frozen-coordinate-antichain-tested"
        ),
        ReceiverHistoryEvidenceGateDisposition(
            'response-identification',
            response_disposition,
            EvidenceCeiling.LOCAL_LAW,
            (
                "action-conditioned-receiver-response-measured"
                if response_disposition is ReceiverHistoryMetatheoryDisposition.SUPPORTED
                else "action-conditioned-receiver-response-invalid"
            ),
        ),
        ReceiverHistoryEvidenceGateDisposition(
            'law-qualification',
            law_qualification_disposition,
            EvidenceCeiling.LOCAL_LAW,
            law_qualification_reason,
        ),
        ReceiverHistoryEvidenceGateDisposition(
            'admission',
            ReceiverHistoryMetatheoryDisposition.NOT_ATTEMPTED,
            EvidenceCeiling.LOCAL_LAW,
            "admission-and-reachability-not-attempted",
        ),
        ReceiverHistoryEvidenceGateDisposition(
            'prospective-use',
            ReceiverHistoryMetatheoryDisposition.NOT_ATTEMPTED,
            EvidenceCeiling.LOCAL_LAW,
            "controller-validation-not-attempted",
        ),
    )
    return ReceiverHistoryEmpiricalMetatheoryDossier(
        dossier_id='receiver-history.empirical-metatheory',
        metatheory_id='receiver-history-empirical-metatheory',
        base_law_id="l-d-h-a-r-tau",
        recurrence_result_sha256=result.fingerprint(),
        power_atlas_sha256=result.power_atlas_sha256,
        method_freeze_sha256=result.method_freeze_sha256,
        source_closure_sha256=source_closure_sha256,
        capability_registry_sha256=capability_registry_sha256,
        execution_plan_sha256=execution_plan_sha256,
        power_transport_cells=matrix.power_transport_cells,
        locality_tournaments=matrix.locality_tournaments,
        hypotheses=matrix.hypotheses,
        propositions=propositions,
        nonclaims=nonclaims,
        evidence_gates=evidence_gates,
        excluded_experiment_ids=_EXCLUDED_EXPERIMENT_IDS,
        limitation_ids=_METATHEORY_LIMITATION_IDS,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        physical_claim_allowed=False,
        controller_claim_allowed=False,
    )


def synthesize_recurrence(
    *,
    config: ReceiverHistoryConfig,
    method_freeze: ReceiverHistoryMethodFreeze,
    power_atlas: ReceiverHistoryEndpointCoordinatePowerAtlas,
    adjudication_bundles: tuple[ReceiverHistoryAdjudicationBundle, ...],
    include_untouched: bool = False,
) -> RecurrenceExecution:
    expected_unit_count = len(config.unit_ids)
    if (
        config.phase is not ReceiverHistoryPhase.TARGETED_EVALUATION
        or len(adjudication_bundles) != expected_unit_count
    ):
        raise ValueError(
            "receiver-history synthesis requires the complete evaluation bundle roster"
        )
    unit_ids = tuple(value.unit_id for value in adjudication_bundles)
    if unit_ids != config.unit_ids or len(set(unit_ids)) != expected_unit_count:
        raise ValueError("receiver-history evaluation bundles must be sorted and unique")
    if any(
        value.method_freeze_sha256 != method_freeze.fingerprint() for value in adjudication_bundles
    ):
        raise ValueError("receiver-history adjudications differ from the method freeze")
    targeted = tuple(
        value for bundle in adjudication_bundles for value in bundle.targeted_adjudications
    )
    coupled_scale_adjudications = tuple(
        sorted(
            (value for bundle in adjudication_bundles for value in bundle.adjudications),
            key=lambda value: (value.unit_id, value.scale_cells),
        )
    )
    if len(coupled_scale_adjudications) != expected_unit_count * len(config.scale_cells):
        raise ValueError("receiver-history coupled scale ledger differs")
    cells = _targeted_cells(config, targeted)
    secondary_cells: tuple[ReceiverHistoryCellRecurrence, ...] = ()
    untouched_cells = _untouched_cells(config, adjudication_bundles) if include_untouched else ()
    untouched_descriptive = (
        _untouched_descriptive_summaries(config, adjudication_bundles) if include_untouched else ()
    )
    alignment = _alignment_summaries(config, targeted)
    resolution: tuple[ReceiverHistoryResolutionPanelSummary, ...] = ()
    rank_summaries: tuple[ReceiverHistoryRankObjectSummary, ...] = ()
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
    result = ReceiverHistoryRecurrenceResult(
        result_id="receiver-history.prospective-recurrence-result",
        method_freeze_sha256=method_freeze.fingerprint(),
        power_atlas_sha256=power_atlas.fingerprint(),
        cells=cells,
        secondary_cells=secondary_cells,
        untouched_cells=untouched_cells,
        untouched_descriptive_summaries=untouched_descriptive,
        alignment_summaries=alignment,
        resolution_summaries=resolution,
        rank_summaries=rank_summaries,
        transition_summaries=transitions,
        fixed_depth_dynamical_bracket=_bracket(
            cells,
            shallow="k4",
            deep="k8",
            endpoint="dynamical-closure",
            eligible="question.dynamical.fixed-k4-k8" in power_atlas.eligible_question_ids,
        ),
        fixed_depth_target_bracket=_bracket(
            cells,
            shallow="k4",
            deep="k8",
            endpoint="target-decision",
            eligible="question.target-decision.fixed-k4-k8" in power_atlas.eligible_question_ids,
        ),
        fixed_depth_sink_bracket=_bracket(
            cells,
            shallow="k4",
            deep="k8",
            endpoint="sink-decision",
            eligible="question.sink-decision.fixed-k4-k8" in power_atlas.eligible_question_ids,
        ),
        normalized_budget_dynamical_bracket=_bracket(
            cells,
            shallow="b0p5",
            deep="b1",
            endpoint="dynamical-closure",
            eligible="question.dynamical.normalized-b0p5-b1" in power_atlas.eligible_question_ids,
        ),
        normalized_budget_target_bracket=_bracket(
            cells,
            shallow="b0p5",
            deep="b1",
            endpoint="target-decision",
            eligible="question.target-decision.normalized-b0p5-b1"
            in power_atlas.eligible_question_ids,
        ),
        normalized_budget_sink_bracket=_bracket(
            cells,
            shallow="b0p5",
            deep="b1",
            endpoint="sink-decision",
            eligible="question.sink-decision.normalized-b0p5-b1"
            in power_atlas.eligible_question_ids,
        ),
        bootstrap_summary_sha256=bootstrap.fingerprint(),
        decisive_counterexample_ids=decisive_ids,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        physical_claim_allowed=False,
        controller_claim_allowed=False,
    )
    return RecurrenceExecution(
        result=result,
        bootstrap_summary=bootstrap,
        disposition_matrix=build_inference_disposition_matrix(
            result=result,
            power_atlas=power_atlas,
        ),
    )


def targeted_continuation_gate(
    result: ReceiverHistoryRecurrenceResult,
) -> ReceiverHistoryTargetedContinuationGateRecord:
    valid = all(value.invalid_count == 0 for value in result.cells)
    endpoints: list[str] = []
    for endpoint, fixed, normalized in (
        (
            "dynamical-closure",
            result.fixed_depth_dynamical_bracket,
            result.normalized_budget_dynamical_bracket,
        ),
        (
            "target-decision",
            result.fixed_depth_target_bracket,
            result.normalized_budget_target_bracket,
        ),
        (
            "sink-decision",
            result.fixed_depth_sink_bracket,
            result.normalized_budget_sink_bracket,
        ),
    ):
        k0 = tuple(
            value
            for value in result.cells
            if value.endpoint_id == endpoint and value.coordinate_label == "k0"
        )
        if fixed is ReceiverHistoryScientificState.SUPPORTED:
            endpoints.append(f"{endpoint}.fixed-k4-k8")
        elif normalized is ReceiverHistoryScientificState.SUPPORTED:
            endpoints.append(f"{endpoint}.normalized-b0p5-b1")
        elif len(k0) == 9 and all(
            value.prerequisite_nonattempt_count == 0
            and value.existence_recurs
            and value.majority_opposition
            for value in k0
        ):
            endpoints.append(f"{endpoint}.k0")
    # Any invalid cell already includes observer/generator/certificate identity
    # failures in the bounded targeted result.  The staged gate does not inspect raw
    # outcomes or select units.
    independence_failures = sum(value.invalid_count for value in result.cells)
    reasons = []
    if not valid:
        reasons.append('RECEIVER_HISTORY_TARGETED_PRIMARY_CELL_INVALID')
    if independence_failures:
        reasons.append('RECEIVER_HISTORY_TARGETED_INDEPENDENCE_FAILURE')
    if not endpoints:
        reasons.append('RECEIVER_HISTORY_TARGETED_ENDPOINT_CONTINUATION_ABSENT')
    return ReceiverHistoryTargetedContinuationGateRecord(
        gate_id="receiver-history.t-continuation-gate",
        predicate_id='receiver-history-t-construct-breakability-confirmed',
        targeted_result_sha256=result.fingerprint(),
        all_primary_cells_valid=valid,
        independence_failure_count=independence_failures,
        continuation_endpoint_coordinate_ids=tuple(sorted(endpoints)),
        activate_evaluation_u=valid and independence_failures == 0 and bool(endpoints),
        eligible_unit_count=result.cells[0].requested_count * len(ReceiverHistoryDisorderFamily),
        reason_codes=tuple(reasons),
    )


def integrate_untouched_recurrence(
    *,
    config: ReceiverHistoryConfig,
    targeted_result: ReceiverHistoryRecurrenceResult,
    adjudication_bundles: tuple[ReceiverHistoryAdjudicationBundle, ...],
) -> ReceiverHistoryRecurrenceResult:
    'Add the conditional untouched evidence without pooling it into targeted estimates.'

    if config.phase is not ReceiverHistoryPhase.UNTOUCHED_EVALUATION or len(adjudication_bundles) != 90:
        raise ValueError('receiver-history untouched synthesis requires all 90 conditional units')
    unit_ids = tuple(value.unit_id for value in adjudication_bundles)
    if unit_ids != evaluation_unit_ids() or any(
        value.targeted_adjudications or value.adjudications for value in adjudication_bundles
    ):
        raise ValueError('receiver-history untouched synthesis received another cohort')
    cells = _untouched_cells(config, adjudication_bundles)
    descriptive = _untouched_descriptive_summaries(config, adjudication_bundles)
    return replace(
        targeted_result,
        result_id="receiver-history.integrated-recurrence-result",
        untouched_cells=cells,
        untouched_descriptive_summaries=descriptive,
    )


__all__ = [
    "RecurrenceExecution",
    "build_empirical_metatheory_dossier",
    "build_inference_disposition_matrix",
    "integrate_untouched_recurrence",
    "synthesize_recurrence",
    "targeted_continuation_gate",
]

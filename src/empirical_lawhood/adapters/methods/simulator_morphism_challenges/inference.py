"""Complete-unit cross-scale/disorder recurrence inference for simulator morphism challenges."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from scipy.stats import beta

from empirical_lawhood.adapters.simulator_morphism_challenges.contracts import SimulatorMorphismChallengeCellRecurrence, SimulatorMorphismChallengeConfig, SimulatorMorphismChallengeDisorderFamily, SimulatorMorphismChallengeMethodFreeze, SimulatorMorphismChallengeRecurrenceResult, SimulatorMorphismChallengeScientificState, SimulatorMorphismChallengeUnitAdjudication
from empirical_lawhood.adapters.simulator_morphism_challenges.descriptors import evaluation_unit_ids, family_from_unit_id
from empirical_lawhood.adapters.simulator_morphism_challenges.runtime_contracts import SimulatorMorphismChallengeBootstrapCellSummary, SimulatorMorphismChallengeBootstrapInterval, SimulatorMorphismChallengeBootstrapSummary
from empirical_lawhood.kernel.evidence import EvidenceCeiling


@dataclass(frozen=True, slots=True)
class RecurrenceExecution:
    result: SimulatorMorphismChallengeRecurrenceResult
    bootstrap_summary: SimulatorMorphismChallengeBootstrapSummary


def _decimal(value: float) -> Decimal:
    if not np.isfinite(value):
        raise ValueError("simulator morphism challenges recurrence output must be finite")
    return Decimal(str(float(value)))


def _lower_bound(opposed: int, requested: int, alpha: float) -> float:
    if opposed == 0:
        return 0.0
    return float(beta.ppf(alpha, opposed, requested - opposed + 1))


def _interval(values: np.ndarray[tuple[int], np.dtype[np.float64]]) -> SimulatorMorphismChallengeBootstrapInterval:
    lower, upper = np.quantile(values, (0.025, 0.975))
    return SimulatorMorphismChallengeBootstrapInterval(
        lower=_decimal(float(lower)),
        upper=_decimal(float(upper)),
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
    config: SimulatorMorphismChallengeConfig,
    method_freeze: SimulatorMorphismChallengeMethodFreeze,
    adjudications: tuple[SimulatorMorphismChallengeUnitAdjudication, ...],
) -> SimulatorMorphismChallengeBootstrapSummary:
    rng = np.random.Generator(np.random.PCG64(config.bootstrap_seed))
    cell_summaries: list[SimulatorMorphismChallengeBootstrapCellSummary] = []
    for family in SimulatorMorphismChallengeDisorderFamily:
        family_rows = tuple(value for value in adjudications if value.family is family)
        unit_ids = tuple(sorted({value.unit_id for value in family_rows}))
        by_key = {(value.unit_id, value.scale_cells): value for value in family_rows}
        if len(unit_ids) != 12:
            raise ValueError("simulator morphism challenges bootstrap requires 12 complete seed blocks per family")
        draws = rng.integers(0, len(unit_ids), size=(config.bootstrap_resamples, len(unit_ids)))
        for scale in (64, 128, 256):
            rows = tuple(by_key[(unit_id, scale)] for unit_id in unit_ids)
            rank_distance = np.asarray(
                [float(value.rank_curve_distance) for value in rows], dtype=np.float64
            )
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
                raise ValueError("simulator morphism challenges k_full and b_full evaluability differs")
            k_bootstrap = _conditional_bootstrap_mean(k_values, draws)
            b_bootstrap = _conditional_bootstrap_mean(b_values, draws)
            cell_summaries.append(
                SimulatorMorphismChallengeBootstrapCellSummary(
                    cell_id=f"bootstrap-cell.{family.value}.n{scale}",
                    family=family,
                    scale_cells=scale,
                    requested_unit_count=12,
                    evaluable_k_full_count=int(np.sum(finite_k)),
                    k_full_mean=(
                        None
                        if not np.any(finite_k)
                        else _decimal(float(np.mean(k_values[finite_k])))
                    ),
                    k_full_bootstrap_95=(None if k_bootstrap.size == 0 else _interval(k_bootstrap)),
                    k_full_usable_resample_count=int(k_bootstrap.size),
                    b_full_mean=(
                        None
                        if not np.any(finite_b)
                        else _decimal(float(np.mean(b_values[finite_b])))
                    ),
                    b_full_bootstrap_95=(None if b_bootstrap.size == 0 else _interval(b_bootstrap)),
                    b_full_usable_resample_count=int(b_bootstrap.size),
                    rank_curve_distance_mean=_decimal(float(np.mean(rank_distance))),
                    rank_curve_distance_bootstrap_95=_interval(sampled_distance),
                )
            )
    return SimulatorMorphismChallengeBootstrapSummary(
        summary_id="simulator-morphism-challenges.bootstrap-summary",
        evaluation_config_sha256=config.fingerprint(),
        method_freeze_sha256=method_freeze.fingerprint(),
        cells=tuple(sorted(cell_summaries, key=lambda value: value.cell_id)),
        resamples=config.bootstrap_resamples,
        bootstrap_seed=config.bootstrap_seed,
        confidence_level=Decimal("0.95"),
        whole_seed_block_resampling=True,
        nested_scale_action_mode_resampling=False,
    )


def synthesize_recurrence(
    *,
    config: SimulatorMorphismChallengeConfig,
    method_freeze: SimulatorMorphismChallengeMethodFreeze,
    adjudications: tuple[SimulatorMorphismChallengeUnitAdjudication, ...],
) -> RecurrenceExecution:
    if config.phase.value != "EVALUATION":
        raise ValueError("simulator morphism challenges recurrence synthesis requires the evaluation config")
    if len(adjudications) != 108:
        raise ValueError("simulator morphism challenges recurrence synthesis requires 36 x 3 scale adjudications")
    keys = tuple((value.unit_id, value.scale_cells) for value in adjudications)
    if keys != tuple(sorted(set(keys))):
        raise ValueError("simulator morphism challenges adjudications must be sorted and unique")
    expected_unit_ids = evaluation_unit_ids()
    if tuple(sorted({value.unit_id for value in adjudications})) != expected_unit_ids:
        raise ValueError("simulator morphism challenges adjudications differ from the frozen unit roster")
    by_unit = {
        unit_id: tuple(value for value in adjudications if value.unit_id == unit_id)
        for unit_id in expected_unit_ids
    }
    if any(
        tuple(value.scale_cells for value in rows) != (64, 128, 256)
        or any(value.family is not family_from_unit_id(unit_id) for value in rows)
        for unit_id, rows in by_unit.items()
    ):
        raise ValueError("simulator morphism challenges complete-unit scale/family binding differs")
    alpha = float(config.simultaneous_alpha) / 18.0
    cells: list[SimulatorMorphismChallengeCellRecurrence] = []
    decisive_ids: set[str] = set()
    for family in SimulatorMorphismChallengeDisorderFamily:
        for scale in (64, 128, 256):
            rows = tuple(
                value
                for value in adjudications
                if value.family is family and value.scale_cells == scale
            )
            if len(rows) != 12:
                raise ValueError("simulator morphism challenges recurrence cell lost its requested denominator")
            for endpoint_id, attribute in (
                ("decision-closure", "decision_state"),
                ("dynamical-closure", "dynamical_state"),
            ):
                states = tuple(getattr(value, attribute) for value in rows)
                opposed = sum(value is SimulatorMorphismChallengeScientificState.OPPOSED for value in states)
                supported = sum(value is SimulatorMorphismChallengeScientificState.SUPPORTED for value in states)
                unevaluable = 12 - opposed - supported
                evaluable = opposed + supported
                realized_evaluable = sum(value.evaluable_pair_count > 0 for value in rows)
                if realized_evaluable != evaluable:
                    raise ValueError("simulator morphism challenges evaluable unit ledger differs from terminal states")
                lower = _lower_bound(opposed, 12, alpha)
                for value in rows:
                    if getattr(value, attribute) is SimulatorMorphismChallengeScientificState.OPPOSED:
                        pair = next(
                            (
                                pair
                                for pair in value.pair_adjudications
                                if (
                                    pair.decision_adverse
                                    if endpoint_id == "decision-closure"
                                    else pair.dynamically_adverse
                                )
                            ),
                            None,
                        )
                        if pair is not None:
                            decisive_ids.add(pair.nomination_id)
                            break
                cell_id = f"cell.{family.value}.n{scale}.{endpoint_id}"
                cells.append(
                    SimulatorMorphismChallengeCellRecurrence(
                        cell_id=cell_id,
                        family=family,
                        scale_cells=scale,
                        endpoint_id=endpoint_id,
                        requested_count=12,
                        descriptor_valid_count=sum(value.descriptor_valid for value in rows),
                        nominated_count=sum(value.nominated_pair_count > 0 for value in rows),
                        realized_collision_count=sum(
                            value.realized_collision_count > 0 for value in rows
                        ),
                        evaluable_count=evaluable,
                        opposed_count=opposed,
                        supported_count=supported,
                        unevaluable_count=unevaluable,
                        evaluable_only_opposition_rate=(
                            None if evaluable == 0 else Decimal(opposed) / Decimal(evaluable)
                        ),
                        opposition_lower_bound=_decimal(lower),
                        existence_recurs=opposed > 0,
                        majority_opposition=(lower > float(config.majority_opposition_threshold)),
                    )
                )
    cells_tuple = tuple(sorted(cells, key=lambda value: value.cell_id))
    bootstrap = _bootstrap_summary(config, method_freeze, adjudications)
    dynamical = tuple(value for value in cells_tuple if value.endpoint_id == "dynamical-closure")
    decision = tuple(value for value in cells_tuple if value.endpoint_id == "decision-closure")
    result = SimulatorMorphismChallengeRecurrenceResult(
        result_id="simulator-morphism-challenges-result",
        method_freeze_sha256=method_freeze.fingerprint(),
        cells=cells_tuple,
        dynamical_existence_recurs=all(value.existence_recurs for value in dynamical),
        decision_existence_recurs=all(value.existence_recurs for value in decision),
        dynamical_majority_opposition_recurs=all(value.majority_opposition for value in dynamical),
        decision_majority_opposition_recurs=all(value.majority_opposition for value in decision),
        bootstrap_summary_sha256=bootstrap.fingerprint(),
        decisive_counterexample_ids=tuple(sorted(decisive_ids)),
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        physical_claim_allowed=False,
        controller_claim_allowed=False,
    )
    return RecurrenceExecution(result=result, bootstrap_summary=bootstrap)


__all__ = ["RecurrenceExecution", "synthesize_recurrence"]

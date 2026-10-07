"Pure development-query translations over released acquisition scorers."

from __future__ import annotations

from decimal import Decimal

import numpy as np

from empirical_lawhood.adapters.methods.receiver_conditioned_io import CanonicalMatrix, CanonicalVector, DOptimalCandidate, DOptimalConfig, DOptimalScorer, FarthestPointConfig, FarthestPointContinuationConfig, FarthestPointMaximinService, FarthestPointSeedRule, FarthestPointTieBreakRule, NativeActionPoint
from empirical_lawhood.adapters.simulators.torax_native import build_native_torax_action_word
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord

from .contracts import ToraxMethodValueAcquisitionMethod, ToraxMethodValueAcquisitionRoundDecision, ToraxMethodValueArm, ToraxMethodValueExperimentSpec, ToraxMethodValueTaskSpec


_FEATURES = (
    "chi-e-m2-s",
    "chi-i-m2-s",
    "core-density-m3",
    "core-te-ev",
    "plasma-current-a",
    "source-location-rho",
    "source-width-rho",
    "target-delta-core-te-ev",
)
_IO_FEATURES = (
    "chi-e-m2-s",
    "core-te-ev",
    "source-location-rho",
    "target-delta-core-te-ev",
)
_UNITS = {
    "chi-e-m2-s": "m2/s",
    "chi-i-m2-s": "m2/s",
    "core-density-m3": "m-3",
    "core-te-ev": "eV",
    "plasma-current-a": "A",
    "source-location-rho": "1",
    "source-width-rho": "1",
    "target-delta-core-te-ev": "eV",
}


def _value(task: ToraxMethodValueTaskSpec, coordinate_id: str) -> Decimal:
    return {
        "chi-e-m2-s": task.chi_e_m2_s,
        "chi-i-m2-s": task.chi_i_m2_s,
        "core-density-m3": task.core_electron_density_m3,
        "core-te-ev": task.core_electron_temperature_ev,
        "plasma-current-a": task.plasma_current_a,
        "source-location-rho": task.source_radial_location,
        "source-width-rho": task.source_width,
        "target-delta-core-te-ev": task.target_delta_core_temperature_ev,
    }[coordinate_id]


def _ranges(
    tasks: tuple[ToraxMethodValueTaskSpec, ...],
    coordinate_ids: tuple[str, ...],
) -> tuple[tuple[Decimal, Decimal], ...]:
    values = []
    for coordinate_id in coordinate_ids:
        observed = tuple(_value(task, coordinate_id) for task in tasks)
        lower, upper = min(observed), max(observed)
        if lower == upper:
            raise ValueError("TORAX acquisition feature has zero frozen-roster range")
        values.append((lower, upper))
    return tuple(values)


def _vector(
    task: ToraxMethodValueTaskSpec,
    *,
    vector_id: str,
    coordinate_ids: tuple[str, ...],
    ranges: tuple[tuple[Decimal, Decimal], ...],
    normalized: bool,
) -> CanonicalVector:
    raw = tuple(_value(task, value) for value in coordinate_ids)
    values = (
        tuple(
            (value - lower) / (upper - lower)
            for value, (lower, upper) in zip(raw, ranges, strict=True)
        )
        if normalized
        else raw
    )
    return CanonicalVector(
        vector_id=vector_id,
        coordinate_ids=coordinate_ids,
        values=values,
    )


def _hold_word(spec: ToraxMethodValueExperimentSpec, task: ToraxMethodValueTaskSpec, arm: ToraxMethodValueArm) -> OccurrenceActionWord:
    hold = next(value for value in spec.actions if value.action_label == "hold")
    return build_native_torax_action_word(
        hold,
        preparation_id=f"preparation.{task.task_id}.{arm.value.lower()}",
    )


def select_development_task(
    *,
    spec: ToraxMethodValueExperimentSpec,
    arm: ToraxMethodValueArm,
    round_number: int,
    acquired_tasks: tuple[ToraxMethodValueTaskSpec, ...],
    eligible_tasks: tuple[ToraxMethodValueTaskSpec, ...],
) -> tuple[ToraxMethodValueAcquisitionRoundDecision, tuple[CanonicalRecord, ...]]:
    """Select one legal task; scorers never execute or reveal a task."""

    if round_number not in {1, 2} or not acquired_tasks or not eligible_tasks:
        raise ValueError("TORAX development selection has an invalid static-round state")
    roster = tuple(sorted((*acquired_tasks, *eligible_tasks), key=lambda value: value.task_id))
    cutoff = f"cutoff.torax-receiver-conditioned-io-method-value-development.{arm.value.lower()}.round-{round_number}"
    if arm is ToraxMethodValueArm.FARTHEST_POINT_MAXIMIN:
        ranges = _ranges(roster, _FEATURES)
        base = FarthestPointConfig(
            config_id=f"farthest-config.torax-receiver-conditioned-io-method-value-development.{arm.value.lower()}.round-{round_number}",
            coordinate_ids=_FEATURES,
            native_units=tuple(_UNITS[value] for value in _FEATURES),
            coordinate_scales=tuple(upper - lower for lower, upper in ranges),
            seed_rule=FarthestPointSeedRule.LOWEST_POINT_ID,
            tie_break_rule=FarthestPointTieBreakRule.LOWEST_POINT_ID,
        )
        words = tuple(_hold_word(spec, task, arm) for task in roster)
        points = tuple(
            NativeActionPoint(
                point_id=f"point.{task.task_id}",
                action_word=ObjectIdentity.from_record(
                    word.word_id,
                    word,
                ),
                coordinates=_vector(
                    task,
                    vector_id=f"feature.{arm.value.lower()}.{task.task_id}",
                    coordinate_ids=_FEATURES,
                    ranges=ranges,
                    normalized=False,
                ),
            )
            for task, word in zip(roster, words, strict=True)
        )
        continuation = FarthestPointContinuationConfig(
            continuation_config_id=(
                f"farthest-continuation.torax-receiver-conditioned-io-method-value-development.{arm.value.lower()}.round-{round_number}"
            ),
            base_config=base,
            already_queried_point_ids=tuple(
                sorted(f"point.{value.task_id}" for value in acquired_tasks)
            ),
            maximum_new_points=1,
        )
        result = FarthestPointMaximinService().continue_farthest(
            result_id=f"farthest-result.torax-receiver-conditioned-io-method-value-development.{arm.value.lower()}.round-{round_number}",
            points=points,
            config=continuation,
        )
        point_id = result.selected_point_ids[0]
        selected = next(value for value in eligible_tasks if f"point.{value.task_id}" == point_id)
        scorer_output = ObjectIdentity.from_record(result.result_id, result)
        records: tuple[CanonicalRecord, ...] = (
            *words,
            *points,
            continuation,
            result,
        )
        method = ToraxMethodValueAcquisitionMethod.FARTHEST_POINT
    else:
        coordinate_ids = _FEATURES if arm is ToraxMethodValueArm.FINITE_MPC else _IO_FEATURES
        ranges = _ranges(roster, coordinate_ids)
        acquired_vectors = tuple(
            _vector(
                task,
                vector_id=f"feature.{arm.value.lower()}.{task.task_id}",
                coordinate_ids=coordinate_ids,
                ranges=ranges,
                normalized=True,
            )
            for task in acquired_tasks
        )
        array = np.vstack(tuple(value.as_array() for value in acquired_vectors))
        information = CanonicalMatrix.from_array(
            matrix_id=f"information.torax-receiver-conditioned-io-method-value-development.{arm.value.lower()}.round-{round_number}",
            row_coordinate_ids=coordinate_ids,
            column_coordinate_ids=coordinate_ids,
            values=array.T @ array,
        )
        config = DOptimalConfig(
            config_id=f"d-optimal-config.torax-receiver-conditioned-io-method-value-development.{arm.value.lower()}.round-{round_number}",
            regularization=Decimal("0.000001"),
            feature_coordinate_ids=coordinate_ids,
        )
        words = tuple(_hold_word(spec, task, arm) for task in eligible_tasks)
        candidates = tuple(
            DOptimalCandidate(
                candidate_id=f"candidate.{arm.value.lower()}.{task.task_id}",
                action_word=ObjectIdentity.from_record(word.word_id, word),
                feature_vector=_vector(
                    task,
                    vector_id=f"feature.{arm.value.lower()}.{task.task_id}",
                    coordinate_ids=coordinate_ids,
                    ranges=ranges,
                    normalized=True,
                ),
                eligible=True,
                reason_codes=(),
            )
            for task, word in zip(eligible_tasks, words, strict=True)
        )
        score_set = DOptimalScorer().score(
            score_set_id=f"d-optimal-scores.torax-receiver-conditioned-io-method-value-development.{arm.value.lower()}.round-{round_number}",
            current_information=information,
            candidates=candidates,
            config=config,
        )
        candidate_id = score_set.ranked_candidate_ids[0]
        selected = next(
            value
            for value in eligible_tasks
            if f"candidate.{arm.value.lower()}.{value.task_id}" == candidate_id
        )
        scorer_output = ObjectIdentity.from_record(score_set.score_set_id, score_set)
        records = (*words, *candidates, config, information, score_set)
        method = (
            ToraxMethodValueAcquisitionMethod.RCJ_IO_FALLBACK
            if arm is ToraxMethodValueArm.WITNESS_GATED_IO
            else ToraxMethodValueAcquisitionMethod.IO_ONLY
            if arm is ToraxMethodValueArm.IO_ONLY
            else ToraxMethodValueAcquisitionMethod.D_OPTIMAL
        )
    decision = ToraxMethodValueAcquisitionRoundDecision(
        decision_id=f"acquisition-decision.torax-receiver-conditioned-io-method-value-development.{arm.value.lower()}.round-{round_number}",
        arm=arm,
        method=method,
        round_number=round_number,
        already_acquired_task_ids=tuple(sorted(value.task_id for value in acquired_tasks)),
        eligible_task_ids=tuple(sorted(value.task_id for value in eligible_tasks)),
        selected_task_id=selected.task_id,
        scorer_output=scorer_output,
        rcj_witness_qualified=False,
        rcj_contacted=False,
        information_cutoff_id=cutoff,
        reason_codes=(("RCJ_WITNESS_UNQUALIFIED_IO_FALLBACK",) if arm is ToraxMethodValueArm.WITNESS_GATED_IO else ()),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    return decision, (*records, decision)


__all__ = ['select_development_task']

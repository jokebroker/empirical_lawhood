"""Chronology-matched categorical, metric and dynamical forecast scoring."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_stable_id,
)

from .contracts import PhysicalScaleMorphismForecastLevel, PhysicalScaleMorphismForecastMemberScore, PhysicalScaleMorphismForecastState, PhysicalScaleMorphismForecastTriplet, PhysicalScaleMorphismForecastTripletScore


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismForecastCase(CanonicalRecord):
    """One frozen prediction paired to one outcome coordinate roster."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-forecast-case'

    case_id: str
    complete_unit_id: str
    level: PhysicalScaleMorphismForecastLevel
    coordinate_ids: tuple[str, ...]
    predicted_category: str | None
    observed_category: str | None
    predicted_values: tuple[Decimal, ...]
    observed_values: tuple[Decimal, ...] | None
    causal_cutoff_id: str
    receiver_window_id: str
    horizon_id: str

    def __post_init__(self) -> None:
        for name in (
            "case_id",
            "complete_unit_id",
            "causal_cutoff_id",
            "receiver_window_id",
            "horizon_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if tuple(sorted(set(self.coordinate_ids))) != self.coordinate_ids:
            raise ValueError("forecast coordinates must be sorted and unique")
        categorical = self.level is PhysicalScaleMorphismForecastLevel.CATEGORICAL
        if categorical:
            if not self.coordinate_ids or self.predicted_category is None:
                raise ValueError("categorical forecast requires a predicted category")
            validate_stable_id(self.predicted_category, field_name="predicted_category")
            if self.observed_category is not None:
                validate_stable_id(self.observed_category, field_name="observed_category")
            if self.predicted_values or self.observed_values is not None:
                raise ValueError("categorical forecast cannot carry metric/path values")
        else:
            if self.predicted_category is not None or self.observed_category is not None:
                raise ValueError("metric/dynamical forecast cannot carry a category")
            if len(self.predicted_values) != len(self.coordinate_ids):
                raise ValueError("forecast prediction coordinate count differs")
            if self.observed_values is not None and len(self.observed_values) != len(
                self.coordinate_ids
            ):
                raise ValueError("forecast outcome coordinate count differs")
            for index, value in enumerate(self.predicted_values):
                validate_decimal(value, field_name=f"predicted_values[{index}]")
            if self.observed_values is not None:
                for index, value in enumerate(self.observed_values):
                    validate_decimal(value, field_name=f"observed_values[{index}]")


def score_forecast_triplet(
    *,
    score_id: str,
    triplet: PhysicalScaleMorphismForecastTriplet,
    cases: tuple[PhysicalScaleMorphismForecastCase, ...],
) -> PhysicalScaleMorphismForecastTripletScore:
    """Score each level separately on exactly the triplet's frozen units."""

    require_sorted_unique_ids(cases, attribute="case_id", field_name="cases")
    member_by_level = {value.level: value for value in triplet.members}
    scores = []
    for level in PhysicalScaleMorphismForecastLevel:
        member = member_by_level[level]
        level_cases = tuple(value for value in cases if value.level is level)
        unit_ids = tuple(sorted(value.complete_unit_id for value in level_cases))
        if unit_ids != triplet.complete_unit_ids:
            scores.append(
                PhysicalScaleMorphismForecastMemberScore(
                    score_id=f"{score_id}.{level.value}",
                    level=level,
                    state=PhysicalScaleMorphismForecastState.UNEVALUABLE,
                    complete_unit_count=len(level_cases),
                    mismatch_unit_ids=(),
                    maximum_defect=None,
                    reason_codes=("complete-unit-roster-mismatch",),
                )
            )
            continue
        coordinate_mismatch = tuple(
            sorted(
                value.complete_unit_id
                for value in level_cases
                if value.coordinate_ids != member.coordinate_ids
            )
        )
        chronology_mismatch = tuple(
            sorted(
                value.complete_unit_id
                for value in level_cases
                if (
                    value.causal_cutoff_id != triplet.causal_cutoff_id
                    or value.receiver_window_id != triplet.receiver_window_id
                    or value.horizon_id != triplet.horizon_id
                )
            )
        )
        missing = tuple(
            sorted(
                value.complete_unit_id
                for value in level_cases
                if (
                    value.observed_category is None
                    if level is PhysicalScaleMorphismForecastLevel.CATEGORICAL
                    else value.observed_values is None
                )
            )
        )
        if coordinate_mismatch or chronology_mismatch or missing:
            reasons = tuple(
                reason
                for reason, present in (
                    ("coordinate-roster-mismatch", bool(coordinate_mismatch)),
                    ("chronology-mismatch", bool(chronology_mismatch)),
                    ("outcome-operand-absent", bool(missing)),
                )
                if present
            )
            scores.append(
                PhysicalScaleMorphismForecastMemberScore(
                    score_id=f"{score_id}.{level.value}",
                    level=level,
                    state=PhysicalScaleMorphismForecastState.UNEVALUABLE,
                    complete_unit_count=len(level_cases),
                    mismatch_unit_ids=tuple(
                        sorted(set(coordinate_mismatch) | set(chronology_mismatch) | set(missing))
                    ),
                    maximum_defect=None,
                    reason_codes=reasons,
                )
            )
            continue
        if level is PhysicalScaleMorphismForecastLevel.CATEGORICAL:
            mismatch = tuple(
                sorted(
                    value.complete_unit_id
                    for value in level_cases
                    if value.predicted_category != value.observed_category
                )
            )
            maximum_defect = Decimal(1) if mismatch else Decimal(0)
        else:
            defect_by_unit = {
                value.complete_unit_id: max(
                    (
                        abs(predicted - observed)
                        for predicted, observed in zip(
                            value.predicted_values,
                            value.observed_values or (),
                            strict=True,
                        )
                    ),
                    default=Decimal(0),
                )
                for value in level_cases
            }
            mismatch = tuple(
                sorted(
                    unit_id
                    for unit_id, defect in defect_by_unit.items()
                    if defect > member.tolerance
                )
            )
            maximum_defect = max(defect_by_unit.values(), default=Decimal(0))
        state = PhysicalScaleMorphismForecastState.OPPOSED if mismatch else PhysicalScaleMorphismForecastState.SUPPORTED
        scores.append(
            PhysicalScaleMorphismForecastMemberScore(
                score_id=f"{score_id}.{level.value}",
                level=level,
                state=state,
                complete_unit_count=len(level_cases),
                mismatch_unit_ids=mismatch,
                maximum_defect=maximum_defect,
                reason_codes=("forecast-mismatch",) if mismatch else (),
            )
        )
    ordered = tuple(sorted(scores, key=lambda value: value.score_id))
    paired = all(value.state is PhysicalScaleMorphismForecastState.SUPPORTED for value in ordered)
    categorical = next(value for value in ordered if value.level is PhysicalScaleMorphismForecastLevel.CATEGORICAL)
    return PhysicalScaleMorphismForecastTripletScore(
        score_id=score_id,
        triplet=ObjectIdentity.from_record(triplet.triplet_id, triplet),
        member_scores=ordered,
        paired_triplet_supported=paired,
        categorical_only=categorical.state is PhysicalScaleMorphismForecastState.SUPPORTED and not paired,
    )


__all__ = ['PhysicalScaleMorphismForecastCase', "score_forecast_triplet"]

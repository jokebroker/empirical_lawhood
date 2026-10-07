"""Pure complete-unit, exchange and action-fibre inference for selective dependence response."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Mapping

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import SelectiveDependenceResponseCaseState, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseExchangeExpectation


class SelectiveDependenceResponseExchangeState(StrEnum):
    OPPOSED = "OPPOSED"
    SUPPORTED = "SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseExchangeAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-exchange-assessment'

    assessment_id: str
    exchange_id: str
    expectation: SelectiveDependenceResponseExchangeExpectation
    point: Decimal
    lower: Decimal
    upper: Decimal
    equivalence_margin: Decimal
    minimum_active_difference: Decimal
    complete_unit_count: int
    state: SelectiveDependenceResponseExchangeState
    decisive_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("assessment_id", "exchange_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "point",
            "lower",
            "upper",
            "equivalence_margin",
            "minimum_active_difference",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if not self.lower <= self.point <= self.upper:
            raise ValueError("exchange interval does not contain its point")
        if self.complete_unit_count < 0:
            raise ValueError("exchange complete-unit count must be nonnegative")
        require_sorted_unique_strings(self.decisive_unit_ids, field_name="decisive_unit_ids")


def classify_response(delta: Decimal, *, neutral_margin: Decimal) -> SelectiveDependenceResponseCaseState:
    validate_decimal(delta, field_name="delta")
    validate_decimal(neutral_margin, field_name="neutral_margin", minimum=Decimal(0))
    if abs(delta) <= neutral_margin:
        return SelectiveDependenceResponseCaseState.NEUTRAL
    return SelectiveDependenceResponseCaseState.HIGH if delta > 0 else SelectiveDependenceResponseCaseState.LOW


def noncompensating_disposition(
    *,
    active_action_gate_margins: Mapping[str, tuple[Decimal, ...]],
    hold_gate_margins: tuple[Decimal, ...] | None,
) -> SelectiveDependenceResponseDisposition:
    """Intersect every gate; target improvement cannot compensate a failure."""

    if hold_gate_margins is None:
        return SelectiveDependenceResponseDisposition.UNEVALUABLE
    for margins in (*active_action_gate_margins.values(), hold_gate_margins):
        if not margins:
            return SelectiveDependenceResponseDisposition.UNEVALUABLE
        for margin in margins:
            validate_decimal(margin, field_name="gate_margin")
    if any(
        all(margin >= 0 for margin in margins) for margins in active_action_gate_margins.values()
    ):
        return SelectiveDependenceResponseDisposition.ACTION_AVAILABLE
    if all(margin >= 0 for margin in hold_gate_margins):
        return SelectiveDependenceResponseDisposition.HOLD_ONLY
    return SelectiveDependenceResponseDisposition.NONATTEMPT


def assess_exchange(
    *,
    assessment_id: str,
    exchange_id: str,
    expectation: SelectiveDependenceResponseExchangeExpectation,
    point: Decimal,
    lower: Decimal,
    upper: Decimal,
    equivalence_margin: Decimal,
    minimum_active_difference: Decimal,
    complete_unit_count: int,
    decisive_unit_ids: tuple[str, ...] = (),
) -> SelectiveDependenceResponseExchangeAssessment:
    if complete_unit_count < 2:
        state = SelectiveDependenceResponseExchangeState.UNEVALUABLE
    elif expectation is SelectiveDependenceResponseExchangeExpectation.INVARIANT:
        state = (
            SelectiveDependenceResponseExchangeState.SUPPORTED
            if lower >= -equivalence_margin and upper <= equivalence_margin
            else SelectiveDependenceResponseExchangeState.OPPOSED
        )
    elif expectation is SelectiveDependenceResponseExchangeExpectation.ACTIVE:
        state = (
            SelectiveDependenceResponseExchangeState.SUPPORTED
            if lower > minimum_active_difference or upper < -minimum_active_difference
            else SelectiveDependenceResponseExchangeState.OPPOSED
        )
    else:
        state = (
            SelectiveDependenceResponseExchangeState.SUPPORTED
            if lower > minimum_active_difference or upper < -minimum_active_difference
            else SelectiveDependenceResponseExchangeState.OPPOSED
        )
    return SelectiveDependenceResponseExchangeAssessment(
        assessment_id=assessment_id,
        exchange_id=exchange_id,
        expectation=expectation,
        point=point,
        lower=lower,
        upper=upper,
        equivalence_margin=equivalence_margin,
        minimum_active_difference=minimum_active_difference,
        complete_unit_count=complete_unit_count,
        state=state,
        decisive_unit_ids=decisive_unit_ids,
    )


__all__ = [
    'SelectiveDependenceResponseExchangeAssessment',
    'SelectiveDependenceResponseExchangeState',
    "assess_exchange",
    "classify_response",
    "noncompensating_disposition",
]

"""Frozen finite causal-history ladder and lexicographic development selection."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_stable_id,
)


class PhysicalScaleMorphismHistoryKind(StrEnum):
    CURRENT_MAPPED_STATE = "CURRENT_MAPPED_STATE"
    FINITE_LAGS = "FINITE_LAGS"
    LOW_MODE_BASIS = "LOW_MODE_BASIS"
    COMPLETE_FINE_HISTORY = "COMPLETE_FINE_HISTORY"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismHistoryCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-history-candidate'

    candidate_id: str
    kind: PhysicalScaleMorphismHistoryKind
    causal_cutoff_id: str
    retained_dimension: int
    validity_passed: bool
    false_safe_count: int
    future_response_defect: Decimal
    wrong_time_control_rejected: bool
    development_board_ids: tuple[str, ...]
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_stable_id(self.causal_cutoff_id, field_name="causal_cutoff_id")
        if self.retained_dimension <= 0 or self.false_safe_count < 0:
            raise ValueError("history dimension/count is invalid")
        validate_decimal(
            self.future_response_defect,
            field_name="future_response_defect",
            minimum=Decimal(0),
        )
        if tuple(sorted(set(self.development_board_ids))) != self.development_board_ids:
            raise ValueError("history development boards must be sorted and unique")
        if not self.development_board_ids:
            raise ValueError("history selection needs excluded development boards")
        if self.evaluation_outcome_count:
            raise ValueError("history candidate cannot inspect evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("history selection must remain development-visible")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismHistorySelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-history-selection'

    selection_id: str
    candidates: tuple[PhysicalScaleMorphismHistoryCandidate, ...]
    selected_candidate_id: str | None
    future_defect_threshold: Decimal
    domain_complete: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.selection_id, field_name="selection_id")
        require_sorted_unique_ids(
            self.candidates, attribute="candidate_id", field_name="candidates"
        )
        validate_decimal(
            self.future_defect_threshold,
            field_name="future_defect_threshold",
            minimum=Decimal(0),
        )
        if self.selected_candidate_id is not None:
            validate_stable_id(self.selected_candidate_id, field_name="selected_candidate_id")
            if self.selected_candidate_id not in {value.candidate_id for value in self.candidates}:
                raise ValueError("selected history candidate is absent")
        if self.domain_complete != (self.selected_candidate_id is not None):
            raise ValueError("history domain completeness differs from selection")


def select_history_candidate(
    *,
    selection_id: str,
    candidates: tuple[PhysicalScaleMorphismHistoryCandidate, ...],
    future_defect_threshold: Decimal,
) -> PhysicalScaleMorphismHistorySelection:
    require_sorted_unique_ids(candidates, attribute="candidate_id", field_name="candidates")
    validate_decimal(
        future_defect_threshold,
        field_name="future_defect_threshold",
        minimum=Decimal(0),
    )
    eligible = tuple(
        value
        for value in candidates
        if value.validity_passed
        and value.false_safe_count == 0
        and value.future_response_defect <= future_defect_threshold
        and value.wrong_time_control_rejected
    )
    selected = min(
        eligible,
        key=lambda value: (value.retained_dimension, tuple(PhysicalScaleMorphismHistoryKind).index(value.kind), value.candidate_id),
        default=None,
    )
    return PhysicalScaleMorphismHistorySelection(
        selection_id=selection_id,
        candidates=candidates,
        selected_candidate_id=None if selected is None else selected.candidate_id,
        future_defect_threshold=future_defect_threshold,
        domain_complete=selected is not None,
    )


__all__ = [
    'PhysicalScaleMorphismHistoryCandidate',
    'PhysicalScaleMorphismHistoryKind',
    'PhysicalScaleMorphismHistorySelection',
    "select_history_candidate",
]

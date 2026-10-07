"Outcome-visible, non-promotable analysis records for terminal denominator qualification."

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .denominator_response import DenominatorResponseQualificationResult


class DenominatorResponseGapOutcomeStatus(StrEnum):
    OPPOSED_BY_ACT_A = "OPPOSED_BY_ACT_A"
    DEFERRED_CONDITION_FALSE = "DEFERRED_CONDITION_FALSE"
    TYPED_EXCLUSION = "TYPED_EXCLUSION"


@dataclass(frozen=True, slots=True)
class DenominatorResponseAnalysisFieldCellSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-analysis-field-cell-summary'

    summary_id: str
    cell_id: str
    d01_max_absolute: Decimal
    d12_max_absolute: Decimal
    d01_max_relative: Decimal
    d12_max_relative: Decimal
    d12_over_d01: Decimal
    valid_fraction: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.cell_id, field_name="cell_id")
        for name, value in (
            ("d01_max_absolute", self.d01_max_absolute),
            ("d12_max_absolute", self.d12_max_absolute),
            ("d01_max_relative", self.d01_max_relative),
            ("d12_max_relative", self.d12_max_relative),
            ("d12_over_d01", self.d12_over_d01),
            ("valid_fraction", self.valid_fraction),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if self.valid_fraction > 1:
            raise ValueError("Denominator response field valid fraction exceeds one")


@dataclass(frozen=True, slots=True)
class DenominatorResponseAnalysisFieldSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-analysis-field-summary'

    summary_id: str
    category: str
    native_field_id: str
    native_unit: str
    cell_summaries: tuple[DenominatorResponseAnalysisFieldCellSummary, ...]
    all_cells_contracted: bool
    finite_in_all_cells: bool
    posthoc_scale_rule_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.category, field_name="category")
        validate_stable_id(
            self.posthoc_scale_rule_id,
            field_name="posthoc_scale_rule_id",
        )
        if not self.native_field_id or not self.native_unit:
            raise ValueError("Denominator response field summary omitted source identity")
        require_sorted_unique_ids(
            self.cell_summaries,
            attribute="cell_id",
            field_name="cell_summaries",
        )
        if len(self.cell_summaries) != 6:
            raise ValueError("Denominator response field summary must cover all six cells")


@dataclass(frozen=True, slots=True)
class DenominatorResponseAnalysisEffectSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-analysis-effect-summary'

    effect_id: str
    receiver_group_id: str
    effect_kind: str
    axis_ids: tuple[str, ...]
    refinement_interval: str
    per_cell_normalized_max: tuple[tuple[str, Decimal], ...]
    minimum: Decimal
    median: Decimal
    maximum: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.effect_id, field_name="effect_id")
        validate_stable_id(self.receiver_group_id, field_name="receiver_group_id")
        validate_stable_id(self.effect_kind, field_name="effect_kind")
        require_sorted_unique_strings(
            self.axis_ids,
            field_name="axis_ids",
            allow_empty=False,
        )
        if self.refinement_interval not in {"coarse-middle", "middle-fine"}:
            raise ValueError("Denominator response effect has an unknown refinement interval")
        cells = tuple(value[0] for value in self.per_cell_normalized_max)
        require_sorted_unique_strings(cells, field_name="per_cell_normalized_max")
        if len(cells) != 6:
            raise ValueError("Denominator response effect must cover six cells")
        for _, value in self.per_cell_normalized_max:
            validate_decimal(
                value,
                field_name="per_cell_normalized_max.value",
                minimum=Decimal(0),
            )
        for name, value in (
            ("minimum", self.minimum),
            ("median", self.median),
            ("maximum", self.maximum),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if not self.minimum <= self.median <= self.maximum:
            raise ValueError("Denominator response effect summary order is invalid")


@dataclass(frozen=True, slots=True)
class DenominatorResponseAnalysisDirectionSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-analysis-direction-summary'

    direction_id: str
    receiver_group_id: str
    contrast_id: str
    cell_count: int
    euclidean_rank_at_five_percent: int
    normalized_singular_values: tuple[Decimal, ...]
    pairwise_cosine_minimum: Decimal
    pairwise_cosine_maximum: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.direction_id, field_name="direction_id")
        validate_stable_id(self.receiver_group_id, field_name="receiver_group_id")
        validate_stable_id(self.contrast_id, field_name="contrast_id")
        if self.cell_count != 6 or not 0 <= self.euclidean_rank_at_five_percent <= 6:
            raise ValueError("Denominator response direction summary has invalid cell/rank count")
        if len(self.normalized_singular_values) != 6:
            raise ValueError("Denominator response direction summary must retain six singular values")
        for value in self.normalized_singular_values:
            validate_decimal(value, field_name="normalized_singular_values")
        for name, value in (
            ("pairwise_cosine_minimum", self.pairwise_cosine_minimum),
            ("pairwise_cosine_maximum", self.pairwise_cosine_maximum),
        ):
            validate_decimal(
                value,
                field_name=name,
                minimum=Decimal("-1.0000001"),
            )
            if value > Decimal("1.0000001"):
                raise ValueError("Denominator response direction cosine exceeds one")


@dataclass(frozen=True, slots=True)
class DenominatorResponseAnalysisNumericalAtlas(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-analysis-numerical-atlas'

    analysis_id: str
    development_decision: ObjectIdentity
    acquisition_index: ObjectIdentity
    primary_result: DenominatorResponseQualificationResult
    cell_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    source_profile_count: int
    source_scalar_count: int
    source_numerics_count: int
    episode_count: int
    total_size_bytes: int
    runtime_seconds_minimum: Decimal
    runtime_seconds_median: Decimal
    runtime_seconds_maximum: Decimal
    field_summaries: tuple[DenominatorResponseAnalysisFieldSummary, ...]
    factor_effects: tuple[DenominatorResponseAnalysisEffectSummary, ...]
    interaction_effects: tuple[DenominatorResponseAnalysisEffectSummary, ...]
    direction_summaries: tuple[DenominatorResponseAnalysisDirectionSummary, ...]
    interpretation_ids: tuple[str, ...]
    limitation_ids: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        for name, string_values in (
            ("cell_ids", self.cell_ids),
            ("view_ids", self.view_ids),
            ("interpretation_ids", self.interpretation_ids),
            ("limitation_ids", self.limitation_ids),
        ):
            require_sorted_unique_strings(
                string_values,
                field_name=name,
                allow_empty=False,
            )
        require_sorted_unique_ids(
            self.field_summaries,
            attribute="summary_id",
            field_name="field_summaries",
        )
        require_sorted_unique_ids(
            self.factor_effects,
            attribute="effect_id",
            field_name="factor_effects",
        )
        require_sorted_unique_ids(
            self.interaction_effects,
            attribute="effect_id",
            field_name="interaction_effects",
        )
        require_sorted_unique_ids(
            self.direction_summaries,
            attribute="direction_id",
            field_name="direction_summaries",
        )
        if (
            self.primary_result is not DenominatorResponseQualificationResult.NO_COMPATIBLE_DENOMINATOR
            or len(self.cell_ids) != 6
            or len(self.view_ids) != 27
            or self.source_profile_count <= 0
            or self.source_scalar_count <= 0
            or self.source_numerics_count <= 0
            or self.episode_count != 162
            or self.total_size_bytes <= 0
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("Denominator response numerical atlas changed its terminal scope")
        for name, value in (
            ("runtime_seconds_minimum", self.runtime_seconds_minimum),
            ("runtime_seconds_median", self.runtime_seconds_median),
            ("runtime_seconds_maximum", self.runtime_seconds_maximum),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if not (
            self.runtime_seconds_minimum
            <= self.runtime_seconds_median
            <= self.runtime_seconds_maximum
        ):
            raise ValueError("Denominator response runtime summary order is invalid")


@dataclass(frozen=True, slots=True)
class DenominatorResponseFormalGapOutcome(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-formal-gap-outcome'

    outcome_id: str
    gap_id: str
    prospective_disposition: str
    posthoc_status: DenominatorResponseGapOutcomeStatus
    reason_codes: tuple[str, ...]
    evidence_object_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.outcome_id, field_name="outcome_id")
        validate_stable_id(self.gap_id, field_name="gap_id")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.evidence_object_ids,
            field_name="evidence_object_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class DenominatorResponseFormalGapLedger(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-formal-gap-ledger'

    ledger_id: str
    register: ObjectIdentity
    prospective_coverage: ObjectIdentity
    development_decision: ObjectIdentity
    outcomes: tuple[DenominatorResponseFormalGapOutcome, ...]
    fabricated_operand_ids: tuple[str, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        require_sorted_unique_ids(
            self.outcomes,
            attribute="gap_id",
            field_name="outcomes",
        )
        require_sorted_unique_strings(
            self.fabricated_operand_ids,
            field_name="fabricated_operand_ids",
        )
        if (
            len(self.outcomes) != 48
            or self.fabricated_operand_ids
            or self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("Denominator response formal-gap ledger changed its safe scope")


@dataclass(frozen=True, slots=True)
class DenominatorResponseQualificationStopHandoff(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/denominator-response-qualification-stop-handoff'

    handoff_id: str
    development_decision: ObjectIdentity
    acquisition_index: ObjectIdentity
    terminal_result: DenominatorResponseQualificationResult
    evaluation_issued: bool
    followup_study_condition_satisfied: bool
    followup_study_issued: bool
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.handoff_id, field_name="handoff_id")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if (
            self.terminal_result is not DenominatorResponseQualificationResult.NO_COMPATIBLE_DENOMINATOR
            or self.evaluation_issued
            or self.followup_study_condition_satisfied
            or self.followup_study_issued
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("Denominator response stop handoff opened a condition-false child")


__all__ = [
    'DenominatorResponseAnalysisEffectSummary',
    'DenominatorResponseAnalysisFieldCellSummary',
    'DenominatorResponseAnalysisFieldSummary',
    'DenominatorResponseAnalysisDirectionSummary',
    'DenominatorResponseAnalysisNumericalAtlas',
    'DenominatorResponseQualificationStopHandoff',
    'DenominatorResponseFormalGapLedger',
    'DenominatorResponseFormalGapOutcome',
    'DenominatorResponseGapOutcomeStatus',
]

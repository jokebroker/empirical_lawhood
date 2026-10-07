"""Development-computed denominator alternatives for selective dependence response."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .analysis import SelectiveDependenceResponseFiniteLawCalibration, SelectiveDependenceResponseSelectiveDependenceSignature, development_prediction_metrics, derive_context_decisions_from_predictions, split_candidate_calibration
from .comparators import SelectiveDependenceResponseComparatorEncoding, SelectiveDependenceResponseComparatorKind, SelectiveDependenceResponseCategoricalPrediction
from .contracts import SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseExchangeExpectation, SelectiveDependenceResponseExchangeForecast, SelectiveDependenceResponseTargetPanel, digest_ids
from .inference import SelectiveDependenceResponseExchangeState


class SelectiveDependenceResponseDenominatorAlternative(StrEnum):
    MERGED = "MERGED"
    OMITTED_ROLE = "OMITTED_ROLE"
    PROPOSED = "PROPOSED"
    SPLIT = "SPLIT"


SELECTIVE_DEPENDENCE_RESPONSE_DENOMINATOR_ALTERNATIVES = tuple(
    sorted(SelectiveDependenceResponseDenominatorAlternative, key=lambda value: value.value)
)


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseDenominatorCandidateAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-denominator-candidate-assessment'

    assessment_id: str
    target_id: str
    alternative: SelectiveDependenceResponseDenominatorAlternative
    complexity_rank: int
    retained_role_ids: tuple[str, ...]
    omitted_role_id: str | None
    split_variable_id: str | None
    split_threshold: Decimal | None
    calibration_complete_unit_count: int
    expected_calibration_complete_unit_count: int
    calibration_accuracy: Decimal
    minimum_required_accuracy: Decimal
    semantic_preservation: bool
    support_boundary_preserved: bool
    action_alphabet_preserved: bool
    adequate: bool
    failure_codes: tuple[str, ...]
    development_unit_ids_sha256: str
    nested_conditions_count_as_units: bool
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("assessment_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.complexity_rank < 0:
            raise ValueError("denominator complexity rank must be nonnegative")
        require_sorted_unique_strings(
            self.retained_role_ids,
            field_name="retained_role_ids",
            allow_empty=False,
        )
        if self.omitted_role_id is not None:
            if self.omitted_role_id not in SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS:
                raise ValueError("omitted denominator role is outside D/H/A/R/tau")
        if (self.split_variable_id is None) != (self.split_threshold is None):
            raise ValueError("split variable and threshold must be jointly present")
        if self.split_variable_id is not None:
            validate_stable_id(self.split_variable_id, field_name="split_variable_id")
            assert self.split_threshold is not None
            validate_decimal(self.split_threshold, field_name="split_threshold")
        if self.calibration_complete_unit_count < 0 or (
            self.expected_calibration_complete_unit_count < 1
        ):
            raise ValueError("denominator calibration unit counts are invalid")
        for name in ("calibration_accuracy", "minimum_required_accuracy"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
            if getattr(self, name) > 1:
                raise ValueError(f"{name} exceeds one")
        require_sorted_unique_strings(self.failure_codes, field_name="failure_codes")
        expected_failures = []
        if self.calibration_complete_unit_count != self.expected_calibration_complete_unit_count:
            expected_failures.append("calibration-unit-denominator-incomplete")
        if self.calibration_accuracy < self.minimum_required_accuracy:
            expected_failures.append("calibration-below-threshold")
        if not self.semantic_preservation:
            expected_failures.append("semantic-preservation-failed")
        if not self.support_boundary_preserved:
            expected_failures.append("support-boundary-not-preserved")
        if not self.action_alphabet_preserved:
            expected_failures.append("action-alphabet-not-preserved")
        if self.failure_codes != tuple(sorted(expected_failures)):
            raise ValueError("denominator failure codes are not operand-derived")
        if self.adequate != (not self.failure_codes):
            raise ValueError("denominator adequacy is not failure-derived")
        if len(self.development_unit_ids_sha256) != 64:
            raise ValueError("denominator development roster digest is not SHA-256")
        if self.nested_conditions_count_as_units:
            raise ValueError("denominator alternatives cannot inflate nested branches")
        if self.evaluation_outcome_count:
            raise ValueError("denominator selection cannot inspect evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("denominator selection must remain development visible")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseDenominatorSelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-denominator-selection'

    selection_id: str
    target_id: str
    assessments: tuple[SelectiveDependenceResponseDenominatorCandidateAssessment, ...]
    selected_alternative: SelectiveDependenceResponseDenominatorAlternative | None
    selected_assessment_id: str | None
    minimal_adequate_rule: str
    resolved: bool
    stop_codes: tuple[str, ...]
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("selection_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.assessments,
            attribute="assessment_id",
            field_name="assessments",
        )
        if tuple(sorted(value.alternative for value in self.assessments)) != (
            SELECTIVE_DEPENDENCE_RESPONSE_DENOMINATOR_ALTERNATIVES
        ):
            raise ValueError("denominator selection lacks the exact alternative family")
        adequate = sorted(
            (value for value in self.assessments if value.adequate),
            key=lambda value: (value.complexity_rank, value.alternative.value),
        )
        expected = adequate[0] if adequate else None
        if self.selected_alternative != (None if expected is None else expected.alternative):
            raise ValueError("selected denominator is not the least-complex adequate candidate")
        if self.selected_assessment_id != (None if expected is None else expected.assessment_id):
            raise ValueError("selected denominator assessment identity differs")
        if self.resolved != (expected is not None):
            raise ValueError("denominator resolution is not assessment-derived")
        require_sorted_unique_strings(self.stop_codes, field_name="stop_codes")
        expected_stops = () if expected is not None else ("no-minimal-adequate-denominator",)
        if self.stop_codes != expected_stops:
            raise ValueError("denominator selection stop differs")
        if not self.minimal_adequate_rule:
            raise ValueError("denominator selection rule is empty")
        if self.evaluation_outcome_count:
            raise ValueError("denominator selection cannot inspect evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("denominator selection must remain development visible")


def _action_alphabet_preserved(
    predictions: tuple[SelectiveDependenceResponseCategoricalPrediction, ...],
    *,
    hold_action_id: str,
) -> bool:
    decisions = derive_context_decisions_from_predictions(
        predictions,
        hold_action_id=hold_action_id,
        decision_id_prefix="denominator-candidate",
    )
    return bool(decisions) and all(
        value.active_fibres_complete
        and value.hold_viable is not None
        and value.disposition is not SelectiveDependenceResponseDisposition.UNEVALUABLE
        for value in decisions
    )


def assess_denominator_alternatives(
    *,
    panel: SelectiveDependenceResponseTargetPanel,
    law: SelectiveDependenceResponseFiniteLawCalibration,
    signature: SelectiveDependenceResponseSelectiveDependenceSignature,
    comparators: tuple[SelectiveDependenceResponseComparatorEncoding, ...],
    exchange_forecasts: tuple[SelectiveDependenceResponseExchangeForecast, ...],
    hold_action_id: str,
    receiver_ids: tuple[str, ...],
    neutral_margin: Decimal,
    omitted_role_id: str,
    omitted_comparator_kind: SelectiveDependenceResponseComparatorKind,
    split_variable_id: str,
    split_threshold: Decimal,
) -> SelectiveDependenceResponseDenominatorSelection:
    """Compute and select the finite denominator family on development units."""

    by_kind = {value.kind: value for value in comparators}
    if len(by_kind) != len(comparators):
        raise ValueError("denominator alternatives received duplicate comparators")
    if omitted_comparator_kind not in by_kind:
        raise ValueError("omitted-role comparator is absent")
    expected_ids = law.calibration_complete_unit_ids
    expected_count = len(expected_ids)
    assessment_by_exchange = {value.exchange_id: value for value in signature.exchange_assessments}

    def role_can_be_omitted(role_id: str) -> bool:
        relevant = [
            assessment_by_exchange[value.exchange_id]
            for value in exchange_forecasts
            if value.expectation is SelectiveDependenceResponseExchangeExpectation.ACTIVE
            and value.coordinate_role_id == role_id
            and value.exchange_id in assessment_by_exchange
        ]
        return not relevant or all(
            value.state is not SelectiveDependenceResponseExchangeState.SUPPORTED for value in relevant
        )

    proposed_metrics = development_prediction_metrics(
        panel,
        predictions=law.predictions,
        complete_unit_ids=expected_ids,
        hold_action_id=hold_action_id,
        receiver_ids=receiver_ids,
        neutral_margin=neutral_margin,
    )
    merged_predictions = by_kind[SelectiveDependenceResponseComparatorKind.DENOMINATOR_BLIND].predictions
    merged_metrics = development_prediction_metrics(
        panel,
        predictions=merged_predictions,
        complete_unit_ids=expected_ids,
        hold_action_id=hold_action_id,
        receiver_ids=receiver_ids,
        neutral_margin=neutral_margin,
    )
    omitted_predictions = by_kind[omitted_comparator_kind].predictions
    omitted_metrics = development_prediction_metrics(
        panel,
        predictions=omitted_predictions,
        complete_unit_ids=expected_ids,
        hold_action_id=hold_action_id,
        receiver_ids=receiver_ids,
        neutral_margin=neutral_margin,
    )
    split_cases, split_correct, split_accuracy = split_candidate_calibration(
        panel,
        fit_complete_unit_ids=law.fit_complete_unit_ids,
        calibration_complete_unit_ids=expected_ids,
        split_variable_id=split_variable_id,
        split_threshold=split_threshold,
        hold_action_id=hold_action_id,
        receiver_ids=receiver_ids,
        neutral_margin=neutral_margin,
        primary_cell_ids=law.primary_cell_ids,
    )
    del split_cases, split_correct

    def mean_accuracy(values: tuple[NamedDecimal, ...]) -> Decimal:
        if not values:
            return Decimal(0)
        return sum((value.value for value in values), Decimal(0)) / Decimal(len(values))

    all_semantics = (
        signature.all_active_supported
        and signature.all_invariant_supported
        and signature.support_boundary_refused
    )
    specs = (
        (
            SelectiveDependenceResponseDenominatorAlternative.MERGED,
            0,
            ("A", "H", "R", "tau"),
            "D",
            None,
            None,
            merged_metrics,
            role_can_be_omitted("D") and signature.all_invariant_supported,
            merged_predictions,
        ),
        (
            SelectiveDependenceResponseDenominatorAlternative.OMITTED_ROLE,
            0,
            tuple(sorted({"D", "H", "A", "R", "tau"} - {omitted_role_id})),
            omitted_role_id,
            None,
            None,
            omitted_metrics,
            role_can_be_omitted(omitted_role_id) and signature.all_invariant_supported,
            omitted_predictions,
        ),
        (
            SelectiveDependenceResponseDenominatorAlternative.PROPOSED,
            1,
            ("A", "D", "H", "R", "tau"),
            None,
            None,
            None,
            proposed_metrics,
            all_semantics,
            law.predictions,
        ),
        (
            SelectiveDependenceResponseDenominatorAlternative.SPLIT,
            2,
            ("A", "D", "H", "R", "tau"),
            None,
            split_variable_id,
            split_threshold,
            None,
            all_semantics,
            law.predictions,
        ),
    )
    assessments = []
    for (
        alternative,
        complexity,
        retained,
        omitted,
        split_variable,
        threshold,
        metrics,
        semantic,
        predictions,
    ) in specs:
        count = expected_count if metrics is None else len(metrics.complete_unit_accuracies)
        accuracy = (
            split_accuracy if metrics is None else mean_accuracy(metrics.complete_unit_accuracies)
        )
        support = signature.support_boundary_refused and (
            metrics is None or not metrics.support_boundary_error_unit_ids
        )
        alphabet = _action_alphabet_preserved(
            predictions,
            hold_action_id=hold_action_id,
        )
        failures = []
        if count != expected_count:
            failures.append("calibration-unit-denominator-incomplete")
        if accuracy < law.minimum_required_accuracy:
            failures.append("calibration-below-threshold")
        if not semantic:
            failures.append("semantic-preservation-failed")
        if not support:
            failures.append("support-boundary-not-preserved")
        if not alphabet:
            failures.append("action-alphabet-not-preserved")
        assessments.append(
            SelectiveDependenceResponseDenominatorCandidateAssessment(
                assessment_id=(
                    f"assessment.{panel.target_id}.denominator.{alternative.value.lower().replace('_', '-')}"
                ),
                target_id=panel.target_id,
                alternative=alternative,
                complexity_rank=complexity,
                retained_role_ids=tuple(sorted(retained)),
                omitted_role_id=omitted,
                split_variable_id=split_variable,
                split_threshold=threshold,
                calibration_complete_unit_count=count,
                expected_calibration_complete_unit_count=expected_count,
                calibration_accuracy=accuracy,
                minimum_required_accuracy=law.minimum_required_accuracy,
                semantic_preservation=semantic,
                support_boundary_preserved=support,
                action_alphabet_preserved=alphabet,
                adequate=not failures,
                failure_codes=tuple(sorted(failures)),
                development_unit_ids_sha256=digest_ids(panel.expected_complete_unit_ids),
                nested_conditions_count_as_units=False,
                evaluation_outcome_count=0,
                outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            )
        )
    ordered = tuple(sorted(assessments, key=lambda value: value.assessment_id))
    adequate = sorted(
        (value for value in ordered if value.adequate),
        key=lambda value: (value.complexity_rank, value.alternative.value),
    )
    selected = adequate[0] if adequate else None
    return SelectiveDependenceResponseDenominatorSelection(
        selection_id=f"selection.{panel.target_id}.denominator",
        target_id=panel.target_id,
        assessments=ordered,
        selected_alternative=None if selected is None else selected.alternative,
        selected_assessment_id=None if selected is None else selected.assessment_id,
        minimal_adequate_rule=(
            "Select the lowest complexity rank passing complete-unit calibration, "
            "semantic preservation, support-boundary and action-alphabet requirements; "
            "break ties by frozen alternative identifier."
        ),
        resolved=selected is not None,
        stop_codes=() if selected is not None else ("no-minimal-adequate-denominator",),
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


__all__ = [
    "SELECTIVE_DEPENDENCE_RESPONSE_DENOMINATOR_ALTERNATIVES",
    'SelectiveDependenceResponseDenominatorAlternative',
    'SelectiveDependenceResponseDenominatorCandidateAssessment',
    'SelectiveDependenceResponseDenominatorSelection',
    "assess_denominator_alternatives",
]

"Evaluator for the preassigned finite-action/HOLD recurrence experiment.\n\nThis runtime consumes only the type-distinct finite-action recurrence assignment and reveal.  It\ncannot choose an action, author admission/reachability, build a programme/compiler,\nemit a controller tick, grade a finite-chart reference, or create current controller use.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage, ActionStageEvent, OccurrenceActionWord
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)
from empirical_lawhood.planning.finite_action_occurrence import FINITE_ACTION_DELIVERY_TOLERANCE_A, finite_action_branch_operator_identity
from empirical_lawhood.planning.finite_action_recurrence import FINITE_ACTION_RECURRENCE_CONTROL_UNIT_COUNT, FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT, FINITE_ACTION_RECURRENCE_MATERIALITY, FINITE_ACTION_RECURRENCE_MEMBER_COUNT, FiniteActionPreactionDisposition, FiniteActionRecurrenceAssignment, FiniteActionRecurrencePredicateResult, FiniteActionRecurrenceRevealIntegrity, FiniteActionRecurrenceRevealedBundle, FiniteActionRecurrenceRevealedEpisode, FiniteActionRecurrenceResponseSample, FiniteActionRecurrenceRouteRole, FiniteActionRecurrenceSealedEpisode


class FiniteActionRecurrenceUnitDisposition(StrEnum):
    DELIVERY_INVALID = "DELIVERY_INVALID"
    EFFICACY_EVALUABLE = "EFFICACY_EVALUABLE"
    HOLD_CONTROL_CORRECT = "HOLD_CONTROL_CORRECT"
    NONATTEMPT = "NONATTEMPT"
    PREDICATE_FAILED = "PREDICATE_FAILED"
    TECHNICAL_PARTIAL = "TECHNICAL_PARTIAL"
    UNEVALUABLE = "UNEVALUABLE"


class FiniteActionRecurrenceResult(StrEnum):
    AUTHORITY_OR_RESOURCE_STOP = "AUTHORITY_OR_RESOURCE_STOP"
    FINITE_ACTION_RECURRENCE_DELIVERY_INVALID = "FINITE_ACTION_RECURRENCE_DELIVERY_INVALID"
    FINITE_ACTION_RECURRENCE_EFFICACY_UNEVALUABLE = "FINITE_ACTION_RECURRENCE_EFFICACY_UNEVALUABLE"
    FINITE_ACTION_RECURRENCE_HOLD_CONTROL_FAILED = "FINITE_ACTION_RECURRENCE_HOLD_CONTROL_FAILED"
    FINITE_ACTION_RECURRENCE_MEMBER_OR_STRATUM_MIXED = "FINITE_ACTION_RECURRENCE_MEMBER_OR_STRATUM_MIXED"
    FINITE_ACTION_RECURRENCE_PREACTION_NONATTEMPT = "FINITE_ACTION_RECURRENCE_PREACTION_NONATTEMPT"
    FINITE_ACTION_RECURRENCE_TECHNICAL_PARTIAL = "FINITE_ACTION_RECURRENCE_TECHNICAL_PARTIAL"
    FINITE_ACTION_RECURRENCE_VALID_NEGATIVE_OR_SUBMATERIAL_RECURRENCE = "FINITE_ACTION_RECURRENCE_VALID_NEGATIVE_OR_SUBMATERIAL_RECURRENCE"
    RECEIPT_CUSTODY_OR_REVEAL_INVALID = "RECEIPT_CUSTODY_OR_REVEAL_INVALID"
    TERMINAL_FINITE_ACTION_RECURRENCE_POSITIVE = (
        "TERMINAL_FINITE_ACTION_RECURRENCE_POSITIVE"
    )


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceMemberEffect(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/finite-action-recurrence-member-effect'

    member_effect_id: str
    unit_id: str
    model_member_id: str
    active_phase_mean: NamedDecimal
    matched_hold_phase_mean: NamedDecimal
    effect: NamedDecimal

    def __post_init__(self) -> None:
        for name, value in (
            ("member_effect_id", self.member_effect_id),
            ("unit_id", self.unit_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        if not (
            self.active_phase_mean.unit == self.matched_hold_phase_mean.unit == self.effect.unit
        ):
            raise ValueError("Finite-action recurrence member effect changes native units")
        if self.effect.value != (self.active_phase_mean.value - self.matched_hold_phase_mean.value):
            raise ValueError("Finite-action recurrence effect is not active minus matched native HOLD")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceEfficacyUnitEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/finite-action-recurrence-efficacy-unit-evaluation'

    evaluation_id: str
    unit_id: str
    physical_independent_unit_id: str
    stratum_id: str
    disposition: FiniteActionRecurrenceUnitDisposition
    member_effects: tuple[FiniteActionRecurrenceMemberEffect, ...]
    robust_effect: NamedDecimal | None
    predicate_results: tuple[FiniteActionRecurrencePredicateResult, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluation_id", self.evaluation_id),
            ("unit_id", self.unit_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
            ("stratum_id", self.stratum_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.member_effects,
            attribute="model_member_id",
            field_name="member_effects",
        )
        require_sorted_unique_ids(
            self.predicate_results,
            attribute="result_id",
            field_name="predicate_results",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is FiniteActionRecurrenceUnitDisposition.EFFICACY_EVALUABLE:
            if (
                len(self.member_effects) != FINITE_ACTION_RECURRENCE_MEMBER_COUNT
                or self.robust_effect is None
                or self.robust_effect.value
                != min(value.effect.value for value in self.member_effects)
                or self.reason_codes
            ):
                raise ValueError("evaluable finite-action recurrence unit is not the two-member minimum")
        elif self.member_effects or self.robust_effect is not None or not self.reason_codes:
            raise ValueError("nonevaluable finite-action recurrence unit cannot retain an effect")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceControlMemberEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/finite-action-recurrence-control-member-evaluation'

    evaluation_id: str
    model_member_id: str
    disposition: FiniteActionRecurrenceUnitDisposition
    predicate_results: tuple[FiniteActionRecurrencePredicateResult, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        validate_stable_id(self.model_member_id, field_name="model_member_id")
        require_sorted_unique_ids(
            self.predicate_results,
            attribute="result_id",
            field_name="predicate_results",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is FiniteActionRecurrenceUnitDisposition.HOLD_CONTROL_CORRECT:
            if self.reason_codes:
                raise ValueError("correct finite-action recurrence HOLD control cannot carry reasons")
        elif not self.reason_codes:
            raise ValueError("incorrect finite-action recurrence HOLD control requires reasons")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceControlEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/finite-action-recurrence-control-evaluation'

    evaluation_id: str
    unit_id: str
    physical_independent_unit_id: str
    hold_anchor_slot_id: str
    disposition: FiniteActionRecurrenceUnitDisposition
    members: tuple[FiniteActionRecurrenceControlMemberEvaluation, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("evaluation_id", self.evaluation_id),
            ("unit_id", self.unit_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
            ("hold_anchor_slot_id", self.hold_anchor_slot_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.members,
            attribute="model_member_id",
            field_name="members",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        correct = self.disposition is FiniteActionRecurrenceUnitDisposition.HOLD_CONTROL_CORRECT
        if correct:
            if (
                len(self.members) != FINITE_ACTION_RECURRENCE_MEMBER_COUNT
                or any(
                    value.disposition
                    is not FiniteActionRecurrenceUnitDisposition.HOLD_CONTROL_CORRECT
                    for value in self.members
                )
                or self.reason_codes
            ):
                raise ValueError("correct finite-action recurrence HOLD control requires both exact members")
        elif not self.reason_codes:
            raise ValueError("failed finite-action recurrence HOLD control requires reasons")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceDistributionSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/finite-action-recurrence-distribution-summary'

    summary_id: str
    group_id: str
    count: int
    mean: NamedDecimal
    median: NamedDecimal
    minimum: NamedDecimal
    maximum: NamedDecimal
    positive_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.group_id, field_name="group_id")
        if self.count <= 0 or not 0 <= self.positive_count <= self.count:
            raise ValueError("Finite-action recurrence distribution count is invalid")
        if (
            len(
                {
                    self.mean.unit,
                    self.median.unit,
                    self.minimum.unit,
                    self.maximum.unit,
                }
            )
            != 1
        ):
            raise ValueError("Finite-action recurrence distribution changes native units")


@dataclass(frozen=True, slots=True)
class RetainedActionRecurrenceDiagnostics(CanonicalRecord):
    """Historical checks reported for context; never terminal operands."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/retained-action-recurrence-diagnostics'

    diagnostics_id: str
    executable_active_action_count: int
    expected_historical_active_action_count: int
    per_stratum_positive_counts: tuple[NamedDecimal, ...]
    robust_positive_median: bool | None
    diagnostic_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.diagnostics_id, field_name="diagnostics_id")
        require_sorted_unique_ids(
            self.per_stratum_positive_counts,
            attribute="value_id",
            field_name="per_stratum_positive_counts",
        )
        if self.executable_active_action_count < 0 or not self.diagnostic_only:
            raise ValueError("Finite-action recurrence historical diagnostics cannot become a decision gate")


@dataclass(frozen=True, slots=True)
class FiniteActionRecurrenceAdjudication(CanonicalRecord):
    "Sole terminal finite-action recurrence record, permanently below admission/controller/current controller use."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/finite-action-recurrence-adjudication'

    adjudication_id: str
    assignment: ObjectIdentity
    revealed_bundle: ObjectIdentity
    g2_operator_feasibility: ObjectIdentity
    g2_controlling_reason_code: str
    occurrence_receipts: tuple[ObjectIdentity, ...]
    native_hold_calibration: ObjectIdentity
    efficacy_units: tuple[FiniteActionRecurrenceEfficacyUnitEvaluation, ...]
    hold_controls: tuple[FiniteActionRecurrenceControlEvaluation, ...]
    rostered_efficacy_unit_count: int
    effective_independent_unit_count: int
    active_coverage: Decimal
    mean_effect: NamedDecimal | None
    sample_sd: NamedDecimal | None
    median_effect: NamedDecimal | None
    minimum_effect: NamedDecimal | None
    maximum_effect: NamedDecimal | None
    one_sided_lower_bound: NamedDecimal | None
    member_distributions: tuple[FiniteActionRecurrenceDistributionSummary, ...]
    stratum_distributions: tuple[FiniteActionRecurrenceDistributionSummary, ...]
    historical_diagnostics: RetainedActionRecurrenceDiagnostics
    result: FiniteActionRecurrenceResult
    claim_ceiling: str
    excluded_claims: tuple[str, ...]
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        validate_nonempty(self.claim_ceiling, field_name="claim_ceiling")
        require_sorted_unique_ids(
            self.occurrence_receipts,
            attribute="object_id",
            field_name="occurrence_receipts",
        )
        require_sorted_unique_ids(
            self.efficacy_units,
            attribute="unit_id",
            field_name="efficacy_units",
        )
        require_sorted_unique_ids(
            self.hold_controls,
            attribute="unit_id",
            field_name="hold_controls",
        )
        require_sorted_unique_ids(
            self.member_distributions,
            attribute="group_id",
            field_name="member_distributions",
        )
        require_sorted_unique_ids(
            self.stratum_distributions,
            attribute="group_id",
            field_name="stratum_distributions",
        )
        require_sorted_unique_strings(
            self.excluded_claims,
            field_name="excluded_claims",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        validate_decimal(
            self.active_coverage,
            field_name="active_coverage",
            minimum=Decimal(0),
        )
        if self.active_coverage > Decimal(1):
            raise ValueError("Finite-action recurrence active coverage cannot exceed one")
        if (
            self.rostered_efficacy_unit_count != FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT
            or len(self.efficacy_units) != FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT
            or len(self.hold_controls) != FINITE_ACTION_RECURRENCE_CONTROL_UNIT_COUNT
        ):
            raise ValueError("Finite-action recurrence adjudication must retain the complete 18+4 roster")
        summaries = (
            self.mean_effect,
            self.sample_sd,
            self.median_effect,
            self.minimum_effect,
            self.maximum_effect,
            self.one_sided_lower_bound,
        )
        if self.effective_independent_unit_count == FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT:
            if any(value is None for value in summaries):
                raise ValueError("complete finite-action recurrence cohort requires all registered summaries")
        elif any(value is not None for value in summaries):
            raise ValueError("incomplete finite-action recurrence cohort cannot report a Student statistic")
        positive = (
            self.result
            is FiniteActionRecurrenceResult.TERMINAL_FINITE_ACTION_RECURRENCE_POSITIVE
        )
        if positive:
            if (
                self.effective_independent_unit_count
                != FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT
                or self.active_coverage != Decimal(1)
                or self.one_sided_lower_bound is None
                or self.one_sided_lower_bound.value <= FINITE_ACTION_RECURRENCE_MATERIALITY
                or any(
                    value.disposition
                    is not FiniteActionRecurrenceUnitDisposition.HOLD_CONTROL_CORRECT
                    for value in self.hold_controls
                )
                or self.reason_codes
            ):
                raise ValueError("positive finite-action recurrence result bypasses its strict intersection")
        if (
            self.claim_ceiling != "FINITE_ACTION_RECURRENCE_ONLY"
            or self.excluded_claims
            != (
                "CONTROLLER_VALIDATION",
                "CURRENT_CONTROLLER_USE",
                "MTR_POSITIVE_SOURCE",
                "ADMISSION_OR_REACHABILITY",
            )
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("Finite-action recurrence adjudication exceeds finite recurrence evidence")


def _median(values: tuple[Decimal, ...]) -> Decimal:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _student_summary(
    values: tuple[Decimal, ...],
    *,
    critical: Decimal,
) -> tuple[Decimal, Decimal, Decimal]:
    if len(values) != FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT:
        raise ValueError("Finite-action recurrence Student inference requires exactly 18 independent units")
    mean = sum(values, Decimal(0)) / Decimal(len(values))
    variance = sum(((value - mean) ** 2 for value in values), Decimal(0)) / Decimal(len(values) - 1)
    sample_sd = variance.sqrt()
    standard_error = (variance / Decimal(len(values))).sqrt()
    return mean, sample_sd, mean - critical * standard_error


def _event_matches(expected: ActionStageEvent, observed: ActionStageEvent) -> bool:
    tolerance = (
        Decimal(0)
        if expected.stage is ActionDeliveryStage.REQUESTED
        else FINITE_ACTION_DELIVERY_TOLERANCE_A
    )
    return (
        expected.stage is observed.stage
        and expected.quantity_id == observed.quantity_id
        and expected.native_unit == observed.native_unit
        and expected.native_action_frame == observed.native_action_frame
        and expected.native_direction == observed.native_direction
        and expected.coordinate == observed.coordinate
        and abs(expected.value - observed.value) <= tolerance
    )


def _delivery_valid(sealed: FiniteActionRecurrenceSealedEpisode, word: OccurrenceActionWord) -> bool:
    if (
        sealed.clipped
        or sealed.rejected
        or sealed.substituted
        or sealed.early_terminated
        or len(sealed.observed_occurrences) != len(word.occurrences)
    ):
        return False
    observed = {value.expected_occurrence_id: value for value in sealed.observed_occurrences}
    if set(observed) != {value.occurrence_id for value in word.occurrences}:
        return False
    for expected in word.occurrences:
        value = observed[expected.occurrence_id]
        if not value.complete:
            return False
        expected_events = (
            expected.requested,
            expected.accepted,
            expected.applied,
            expected.realized,
        )
        observed_events = (
            value.requested,
            value.accepted,
            value.applied,
            value.realized,
        )
        if any(
            actual is None or not _event_matches(reference, actual)
            for reference, actual in zip(expected_events, observed_events, strict=True)
        ):
            return False
    return True


def _phase_mean(samples: tuple[FiniteActionRecurrenceResponseSample, ...]) -> Decimal:
    return sum((value.value.value for value in samples), Decimal(0)) / Decimal(len(samples))


def _predicate_disposition(
    predicates: tuple[FiniteActionRecurrencePredicateResult, ...],
) -> tuple[FiniteActionRecurrenceUnitDisposition | None, tuple[str, ...]]:
    if any(value.status is GateStatus.FAIL for value in predicates):
        return (
            FiniteActionRecurrenceUnitDisposition.PREDICATE_FAILED,
            ("FINITE_ACTION_RECURRENCE_NONCOMPENSATING_PREDICATE_FAILED",),
        )
    if any(value.status is GateStatus.UNEVALUABLE for value in predicates):
        return (
            FiniteActionRecurrenceUnitDisposition.UNEVALUABLE,
            ("FINITE_ACTION_RECURRENCE_NONCOMPENSATING_PREDICATE_UNEVALUABLE",),
        )
    return None, ()


def _distribution(
    *,
    summary_id: str,
    group_id: str,
    values: tuple[Decimal, ...],
    unit: str,
) -> FiniteActionRecurrenceDistributionSummary:
    return FiniteActionRecurrenceDistributionSummary(
        summary_id=summary_id,
        group_id=group_id,
        count=len(values),
        mean=NamedDecimal(
            value_id=f"mean.{summary_id}",
            value=sum(values, Decimal(0)) / Decimal(len(values)),
            unit=unit,
        ),
        median=NamedDecimal(
            value_id=f"median.{summary_id}",
            value=_median(values),
            unit=unit,
        ),
        minimum=NamedDecimal(
            value_id=f"minimum.{summary_id}",
            value=min(values),
            unit=unit,
        ),
        maximum=NamedDecimal(
            value_id=f"maximum.{summary_id}",
            value=max(values),
            unit=unit,
        ),
        positive_count=sum(value > 0 for value in values),
    )


def _failure_precedence(
    values: set[FiniteActionRecurrenceUnitDisposition],
) -> FiniteActionRecurrenceUnitDisposition:
    for value in (
        FiniteActionRecurrenceUnitDisposition.NONATTEMPT,
        FiniteActionRecurrenceUnitDisposition.DELIVERY_INVALID,
        FiniteActionRecurrenceUnitDisposition.TECHNICAL_PARTIAL,
        FiniteActionRecurrenceUnitDisposition.PREDICATE_FAILED,
        FiniteActionRecurrenceUnitDisposition.UNEVALUABLE,
    ):
        if value in values:
            return value
    return FiniteActionRecurrenceUnitDisposition.UNEVALUABLE


class MatchedFiniteActionHoldRecurrenceEvaluator:
    """Pure evaluator for already-assigned routes; it performs no action I/O."""

    @staticmethod
    def _maps(
        revealed: FiniteActionRecurrenceRevealedBundle,
    ) -> tuple[
        dict[
            tuple[str, str, FiniteActionRecurrenceRouteRole],
            FiniteActionRecurrenceSealedEpisode,
        ],
        dict[str, FiniteActionRecurrenceRevealedEpisode],
    ]:
        sealed = {value.route.product_key: value for value in revealed.sealed_bundle.episodes}
        outcomes = {value.episode_id: value for value in revealed.episodes}
        return sealed, outcomes

    def _evaluate_efficacy(
        self,
        *,
        assignment: FiniteActionRecurrenceAssignment,
        revealed: FiniteActionRecurrenceRevealedBundle,
        unit_id: str,
    ) -> FiniteActionRecurrenceEfficacyUnitEvaluation:
        unit = next(value for value in assignment.efficacy_units if value.unit_id == unit_id)
        if unit.stratum_id is None:  # pragma: no cover - frozen by assignment
            raise AssertionError("Finite-action recurrence efficacy unit lost its stratum")
        sealed, outcomes = self._maps(revealed)
        effects: list[FiniteActionRecurrenceMemberEffect] = []
        predicates: list[FiniteActionRecurrencePredicateResult] = []
        failures: set[FiniteActionRecurrenceUnitDisposition] = set()
        reasons: set[str] = set()
        for member_id in assignment.model_member_ids:
            active_sealed = sealed[(unit_id, member_id, FiniteActionRecurrenceRouteRole.ACTIVE)]
            hold_sealed = sealed[
                (unit_id, member_id, FiniteActionRecurrenceRouteRole.MATCHED_HOLD)
            ]
            if (
                active_sealed.preaction_disposition is FiniteActionPreactionDisposition.NONATTEMPT
                or hold_sealed.preaction_disposition
                is FiniteActionPreactionDisposition.NONATTEMPT
            ):
                failures.add(FiniteActionRecurrenceUnitDisposition.NONATTEMPT)
                reasons.add("FINITE_ACTION_RECURRENCE_PAIRED_PREACTION_NONATTEMPT")
                continue
            active = outcomes[active_sealed.episode_id]
            hold = outcomes[hold_sealed.episode_id]
            pair_predicates = (*active.predicate_results, *hold.predicate_results)
            predicates.extend(pair_predicates)
            predicate_failure, predicate_reasons = _predicate_disposition(pair_predicates)
            if not _delivery_valid(active_sealed, assignment.active_word) or not _delivery_valid(
                hold_sealed,
                assignment.matched_hold_word,
            ):
                failures.add(FiniteActionRecurrenceUnitDisposition.DELIVERY_INVALID)
                reasons.add("FINITE_ACTION_RECURRENCE_REQUESTED_ACCEPTED_APPLIED_REALIZED_MISMATCH")
                continue
            if active.technical_reason_codes or hold.technical_reason_codes:
                failures.add(FiniteActionRecurrenceUnitDisposition.TECHNICAL_PARTIAL)
                reasons.update(active.technical_reason_codes)
                reasons.update(hold.technical_reason_codes)
                continue
            if predicate_failure is not None:
                failures.add(predicate_failure)
                reasons.update(predicate_reasons)
                continue
            if len(active.response_samples) != 6 or len(hold.response_samples) != 6:
                failures.add(FiniteActionRecurrenceUnitDisposition.UNEVALUABLE)
                reasons.add("FINITE_ACTION_RECURRENCE_RECEIVER_PHASE_INCOMPLETE")
                continue
            active_mean = _phase_mean(active.response_samples)
            hold_mean = _phase_mean(hold.response_samples)
            effects.append(
                FiniteActionRecurrenceMemberEffect(
                    member_effect_id=f"hfr-member-effect.{unit_id}.{member_id}",
                    unit_id=unit_id,
                    model_member_id=member_id,
                    active_phase_mean=NamedDecimal(
                        value_id=f"hfr-active-mean.{unit_id}.{member_id}",
                        value=active_mean,
                        unit=assignment.effect_native_unit,
                    ),
                    matched_hold_phase_mean=NamedDecimal(
                        value_id=f"hfr-hold-mean.{unit_id}.{member_id}",
                        value=hold_mean,
                        unit=assignment.effect_native_unit,
                    ),
                    effect=NamedDecimal(
                        value_id=f"hfr-effect.{unit_id}.{member_id}",
                        value=active_mean - hold_mean,
                        unit=assignment.effect_native_unit,
                    ),
                )
            )
        if len(effects) == FINITE_ACTION_RECURRENCE_MEMBER_COUNT and not failures:
            disposition = FiniteActionRecurrenceUnitDisposition.EFFICACY_EVALUABLE
            robust_effect = NamedDecimal(
                value_id=f"hfr-robust-effect.{unit_id}",
                value=min(value.effect.value for value in effects),
                unit=assignment.effect_native_unit,
            )
            output_reasons: tuple[str, ...] = ()
        else:
            effects = []
            disposition = _failure_precedence(failures)
            robust_effect = None
            output_reasons = tuple(sorted(reasons))
        return FiniteActionRecurrenceEfficacyUnitEvaluation(
            evaluation_id=f"hfr-efficacy-evaluation.{unit_id}",
            unit_id=unit_id,
            physical_independent_unit_id=unit.physical_independent_unit_id,
            stratum_id=unit.stratum_id,
            disposition=disposition,
            member_effects=tuple(sorted(effects, key=lambda value: value.model_member_id)),
            robust_effect=robust_effect,
            predicate_results=tuple(sorted(predicates, key=lambda value: value.result_id)),
            reason_codes=output_reasons,
        )

    def _evaluate_control(
        self,
        *,
        assignment: FiniteActionRecurrenceAssignment,
        revealed: FiniteActionRecurrenceRevealedBundle,
        unit_id: str,
    ) -> FiniteActionRecurrenceControlEvaluation:
        unit = next(value for value in assignment.hold_controls if value.unit_id == unit_id)
        if unit.hold_anchor_slot_id is None:  # pragma: no cover - frozen by assignment
            raise AssertionError("Finite-action recurrence HOLD control lost its anchor")
        sealed, outcomes = self._maps(revealed)
        members: list[FiniteActionRecurrenceControlMemberEvaluation] = []
        for member_id in assignment.model_member_ids:
            sealed_episode = sealed[
                (unit_id, member_id, FiniteActionRecurrenceRouteRole.HOLD_CONTROL)
            ]
            reasons: set[str] = set()
            predicates: tuple[FiniteActionRecurrencePredicateResult, ...] = ()
            if (
                sealed_episode.preaction_disposition
                is FiniteActionPreactionDisposition.NONATTEMPT
            ):
                disposition = FiniteActionRecurrenceUnitDisposition.NONATTEMPT
                reasons.add("FINITE_ACTION_RECURRENCE_HOLD_CONTROL_PREACTION_NONATTEMPT")
            else:
                outcome = outcomes[sealed_episode.episode_id]
                predicates = outcome.predicate_results
                predicate_failure, predicate_reasons = _predicate_disposition(predicates)
                if not _delivery_valid(sealed_episode, assignment.matched_hold_word):
                    disposition = FiniteActionRecurrenceUnitDisposition.DELIVERY_INVALID
                    reasons.add("FINITE_ACTION_RECURRENCE_HOLD_CONTROL_DELIVERY_INVALID")
                elif outcome.technical_reason_codes:
                    disposition = FiniteActionRecurrenceUnitDisposition.TECHNICAL_PARTIAL
                    reasons.update(outcome.technical_reason_codes)
                elif predicate_failure is not None:
                    disposition = predicate_failure
                    reasons.update(predicate_reasons)
                else:
                    disposition = FiniteActionRecurrenceUnitDisposition.HOLD_CONTROL_CORRECT
            members.append(
                FiniteActionRecurrenceControlMemberEvaluation(
                    evaluation_id=f"hfr-control-member.{unit_id}.{member_id}",
                    model_member_id=member_id,
                    disposition=disposition,
                    predicate_results=predicates,
                    reason_codes=tuple(sorted(reasons)),
                )
            )
        failures: set[FiniteActionRecurrenceUnitDisposition] = {
            value.disposition
            for value in members
            if value.disposition is not FiniteActionRecurrenceUnitDisposition.HOLD_CONTROL_CORRECT
        }
        if failures:
            disposition = _failure_precedence(failures)
            aggregate_reasons = tuple(
                sorted({reason for value in members for reason in value.reason_codes})
            )
        else:
            disposition = FiniteActionRecurrenceUnitDisposition.HOLD_CONTROL_CORRECT
            aggregate_reasons = ()
        return FiniteActionRecurrenceControlEvaluation(
            evaluation_id=f"hfr-control-evaluation.{unit_id}",
            unit_id=unit_id,
            physical_independent_unit_id=unit.physical_independent_unit_id,
            hold_anchor_slot_id=unit.hold_anchor_slot_id,
            disposition=disposition,
            members=tuple(sorted(members, key=lambda value: value.model_member_id)),
            reason_codes=aggregate_reasons,
        )

    @staticmethod
    def _empty_adjudication(
        *,
        assignment: FiniteActionRecurrenceAssignment,
        revealed: FiniteActionRecurrenceRevealedBundle,
        result: FiniteActionRecurrenceResult,
    ) -> FiniteActionRecurrenceAdjudication:
        reasons = tuple(sorted({result.value, *revealed.integrity_reason_codes}))
        efficacy = tuple(
            FiniteActionRecurrenceEfficacyUnitEvaluation(
                evaluation_id=f"hfr-efficacy-evaluation.{unit.unit_id}",
                unit_id=unit.unit_id,
                physical_independent_unit_id=unit.physical_independent_unit_id,
                stratum_id=unit.stratum_id or "stratum.missing",
                disposition=FiniteActionRecurrenceUnitDisposition.UNEVALUABLE,
                member_effects=(),
                robust_effect=None,
                predicate_results=(),
                reason_codes=reasons,
            )
            for unit in assignment.efficacy_units
        )
        controls = tuple(
            FiniteActionRecurrenceControlEvaluation(
                evaluation_id=f"hfr-control-evaluation.{unit.unit_id}",
                unit_id=unit.unit_id,
                physical_independent_unit_id=unit.physical_independent_unit_id,
                hold_anchor_slot_id=unit.hold_anchor_slot_id or "anchor.missing",
                disposition=FiniteActionRecurrenceUnitDisposition.UNEVALUABLE,
                members=(),
                reason_codes=reasons,
            )
            for unit in assignment.hold_controls
        )
        return _adjudication(
            assignment=assignment,
            revealed=revealed,
            efficacy=efficacy,
            controls=controls,
            effective_n=0,
            active_coverage=Decimal(0),
            summaries=(None, None, None, None, None, None),
            member_distributions=(),
            stratum_distributions=(),
            result=result,
            reason_codes=reasons,
        )

    def evaluate(
        self,
        *,
        assignment: FiniteActionRecurrenceAssignment,
        revealed: FiniteActionRecurrenceRevealedBundle,
    ) -> FiniteActionRecurrenceAdjudication:
        if revealed.sealed_bundle.assignment != assignment:
            raise ValueError("Finite-action recurrence evaluator received another assignment")
        if revealed.integrity is FiniteActionRecurrenceRevealIntegrity.AUTHORITY_OR_RESOURCE_STOP:
            return self._empty_adjudication(
                assignment=assignment,
                revealed=revealed,
                result=FiniteActionRecurrenceResult.AUTHORITY_OR_RESOURCE_STOP,
            )
        if revealed.integrity is not FiniteActionRecurrenceRevealIntegrity.VALID:
            return self._empty_adjudication(
                assignment=assignment,
                revealed=revealed,
                result=FiniteActionRecurrenceResult.RECEIPT_CUSTODY_OR_REVEAL_INVALID,
            )
        efficacy = tuple(
            self._evaluate_efficacy(
                assignment=assignment,
                revealed=revealed,
                unit_id=unit.unit_id,
            )
            for unit in assignment.efficacy_units
        )
        controls = tuple(
            self._evaluate_control(
                assignment=assignment,
                revealed=revealed,
                unit_id=unit.unit_id,
            )
            for unit in assignment.hold_controls
        )
        evaluable = tuple(
            value
            for value in efficacy
            if value.disposition is FiniteActionRecurrenceUnitDisposition.EFFICACY_EVALUABLE
        )
        effective_n = len(evaluable)
        active_ready = sum(
            episode.route.role is FiniteActionRecurrenceRouteRole.ACTIVE
            and episode.preaction_disposition is FiniteActionPreactionDisposition.READY
            for episode in revealed.sealed_bundle.episodes
        )
        active_coverage = Decimal(active_ready) / Decimal(
            FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT * FINITE_ACTION_RECURRENCE_MEMBER_COUNT
        )
        summaries: tuple[
            NamedDecimal | None,
            NamedDecimal | None,
            NamedDecimal | None,
            NamedDecimal | None,
            NamedDecimal | None,
            NamedDecimal | None,
        ] = (None, None, None, None, None, None)
        member_distributions: tuple[FiniteActionRecurrenceDistributionSummary, ...] = ()
        stratum_distributions: tuple[FiniteActionRecurrenceDistributionSummary, ...] = ()
        member_mixed = False
        stratum_mixed = False
        if effective_n == FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT:
            robust_values = tuple(
                value.robust_effect.value for value in efficacy if value.robust_effect is not None
            )
            mean, sample_sd, lower = _student_summary(
                robust_values,
                critical=assignment.one_sided_critical_value,
            )
            summaries = (
                NamedDecimal(
                    value_id=f"hfr-mean.{assignment.assignment_id}",
                    value=mean,
                    unit=assignment.effect_native_unit,
                ),
                NamedDecimal(
                    value_id=f"hfr-sample-sd.{assignment.assignment_id}",
                    value=sample_sd,
                    unit=assignment.effect_native_unit,
                ),
                NamedDecimal(
                    value_id=f"hfr-median.{assignment.assignment_id}",
                    value=_median(robust_values),
                    unit=assignment.effect_native_unit,
                ),
                NamedDecimal(
                    value_id=f"hfr-minimum.{assignment.assignment_id}",
                    value=min(robust_values),
                    unit=assignment.effect_native_unit,
                ),
                NamedDecimal(
                    value_id=f"hfr-maximum.{assignment.assignment_id}",
                    value=max(robust_values),
                    unit=assignment.effect_native_unit,
                ),
                NamedDecimal(
                    value_id=f"hfr-lower-bound.{assignment.assignment_id}",
                    value=lower,
                    unit=assignment.effect_native_unit,
                ),
            )
            member_distributions = tuple(
                _distribution(
                    summary_id=f"hfr-member-distribution.{member_id}",
                    group_id=member_id,
                    values=tuple(
                        next(
                            member.effect.value
                            for member in unit.member_effects
                            if member.model_member_id == member_id
                        )
                        for unit in efficacy
                    ),
                    unit=assignment.effect_native_unit,
                )
                for member_id in assignment.model_member_ids
            )
            robust_by_unit = {
                value.unit_id: value.robust_effect.value
                for value in efficacy
                if value.robust_effect is not None
            }
            stratum_distributions = tuple(
                _distribution(
                    summary_id=f"hfr-stratum-distribution.{stratum.stratum_id}",
                    group_id=stratum.stratum_id,
                    values=tuple(robust_by_unit[unit_id] for unit_id in stratum.unit_ids),
                    unit=assignment.effect_native_unit,
                )
                for stratum in assignment.strata
            )
            member_mixed = len({value.mean.value > 0 for value in member_distributions}) > 1
            stratum_mixed = len({value.mean.value > 0 for value in stratum_distributions}) > 1
        dispositions = {value.disposition for value in efficacy}
        reasons: set[str] = set()
        if FiniteActionRecurrenceUnitDisposition.NONATTEMPT in dispositions or any(
            value.disposition is FiniteActionRecurrenceUnitDisposition.NONATTEMPT
            for value in controls
        ):
            result = FiniteActionRecurrenceResult.FINITE_ACTION_RECURRENCE_PREACTION_NONATTEMPT
            reasons.add("FINITE_ACTION_RECURRENCE_PREACTION_NONATTEMPT")
        elif FiniteActionRecurrenceUnitDisposition.DELIVERY_INVALID in dispositions:
            result = FiniteActionRecurrenceResult.FINITE_ACTION_RECURRENCE_DELIVERY_INVALID
            reasons.add("FINITE_ACTION_RECURRENCE_EFFICACY_DELIVERY_INVALID")
        elif FiniteActionRecurrenceUnitDisposition.TECHNICAL_PARTIAL in dispositions:
            result = FiniteActionRecurrenceResult.FINITE_ACTION_RECURRENCE_TECHNICAL_PARTIAL
            reasons.add("FINITE_ACTION_RECURRENCE_EFFICACY_TECHNICAL_PARTIAL")
        elif (
            effective_n != FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT
            or active_coverage != Decimal(1)
        ):
            result = FiniteActionRecurrenceResult.FINITE_ACTION_RECURRENCE_EFFICACY_UNEVALUABLE
            reasons.add("FINITE_ACTION_RECURRENCE_COMPLETE_EFFICACY_RECURRENCE_NOT_EVALUABLE")
        elif member_mixed or stratum_mixed:
            result = FiniteActionRecurrenceResult.FINITE_ACTION_RECURRENCE_MEMBER_OR_STRATUM_MIXED
            if member_mixed:
                reasons.add("FINITE_ACTION_RECURRENCE_MEMBER_EFFECT_SIGN_MIXED")
            if stratum_mixed:
                reasons.add("FINITE_ACTION_RECURRENCE_STRATUM_EFFECT_SIGN_MIXED")
        elif summaries[-1] is None or summaries[-1].value <= assignment.materiality.value:
            result = FiniteActionRecurrenceResult.FINITE_ACTION_RECURRENCE_VALID_NEGATIVE_OR_SUBMATERIAL_RECURRENCE
            reasons.add("FINITE_ACTION_RECURRENCE_LOWER_BOUND_DOES_NOT_STRICTLY_EXCEED_MATERIALITY")
        elif any(
            value.disposition is not FiniteActionRecurrenceUnitDisposition.HOLD_CONTROL_CORRECT
            for value in controls
        ):
            result = FiniteActionRecurrenceResult.FINITE_ACTION_RECURRENCE_HOLD_CONTROL_FAILED
            reasons.add("FINITE_ACTION_RECURRENCE_OUTSIDE_SUPPORT_HOLD_CONTROL_FAILED")
        else:
            result = FiniteActionRecurrenceResult.TERMINAL_FINITE_ACTION_RECURRENCE_POSITIVE
        return _adjudication(
            assignment=assignment,
            revealed=revealed,
            efficacy=efficacy,
            controls=controls,
            effective_n=effective_n,
            active_coverage=active_coverage,
            summaries=summaries,
            member_distributions=member_distributions,
            stratum_distributions=stratum_distributions,
            result=result,
            reason_codes=tuple(sorted(reasons)),
        )


def _adjudication(
    *,
    assignment: FiniteActionRecurrenceAssignment,
    revealed: FiniteActionRecurrenceRevealedBundle,
    efficacy: tuple[FiniteActionRecurrenceEfficacyUnitEvaluation, ...],
    controls: tuple[FiniteActionRecurrenceControlEvaluation, ...],
    effective_n: int,
    active_coverage: Decimal,
    summaries: tuple[
        NamedDecimal | None,
        NamedDecimal | None,
        NamedDecimal | None,
        NamedDecimal | None,
        NamedDecimal | None,
        NamedDecimal | None,
    ],
    member_distributions: tuple[FiniteActionRecurrenceDistributionSummary, ...],
    stratum_distributions: tuple[FiniteActionRecurrenceDistributionSummary, ...],
    result: FiniteActionRecurrenceResult,
    reason_codes: tuple[str, ...],
) -> FiniteActionRecurrenceAdjudication:
    branch = assignment.branch_selection
    controlling = branch.controlling_reason_code
    if controlling is None:  # pragma: no cover - narrowed by assignment
        raise AssertionError("Finite-action recurrence assignment lost its controlling input-output operator feasibility obstruction")
    per_stratum = tuple(
        NamedDecimal(
            value_id=f"diagnostic.positive-count.{value.group_id}",
            value=Decimal(value.positive_count),
            unit="count",
        )
        for value in stratum_distributions
    )
    median = summaries[2]
    diagnostics = RetainedActionRecurrenceDiagnostics(
        diagnostics_id=f"hfr-diagnostics.{assignment.assignment_id}",
        executable_active_action_count=sum(
            value.route.role is FiniteActionRecurrenceRouteRole.ACTIVE
            and value.preaction_disposition is FiniteActionPreactionDisposition.READY
            for value in revealed.sealed_bundle.episodes
        ),
        expected_historical_active_action_count=36,
        per_stratum_positive_counts=per_stratum,
        robust_positive_median=None if median is None else median.value > 0,
        diagnostic_only=True,
    )
    return FiniteActionRecurrenceAdjudication(
        adjudication_id=f"hfr-adjudication.{assignment.assignment_id}",
        assignment=ObjectIdentity.from_record(assignment.assignment_id, assignment),
        revealed_bundle=ObjectIdentity.from_record(revealed.reveal_id, revealed),
        g2_operator_feasibility=finite_action_branch_operator_identity(branch),
        g2_controlling_reason_code=controlling,
        occurrence_receipts=tuple(
            sorted(
                (
                    ObjectIdentity.from_record(value.receipt_id, value)
                    for value in assignment.occurrence_receipts
                ),
                key=lambda value: value.object_id,
            )
        ),
        native_hold_calibration=ObjectIdentity.from_record(
            assignment.native_hold_calibration.receipt_id,
            assignment.native_hold_calibration,
        ),
        efficacy_units=efficacy,
        hold_controls=controls,
        rostered_efficacy_unit_count=FINITE_ACTION_RECURRENCE_EFFICACY_UNIT_COUNT,
        effective_independent_unit_count=effective_n,
        active_coverage=active_coverage,
        mean_effect=summaries[0],
        sample_sd=summaries[1],
        median_effect=summaries[2],
        minimum_effect=summaries[3],
        maximum_effect=summaries[4],
        one_sided_lower_bound=summaries[5],
        member_distributions=member_distributions,
        stratum_distributions=stratum_distributions,
        historical_diagnostics=diagnostics,
        result=result,
        claim_ceiling="FINITE_ACTION_RECURRENCE_ONLY",
        excluded_claims=(
            "CONTROLLER_VALIDATION",
            "CURRENT_CONTROLLER_USE",
            "MTR_POSITIVE_SOURCE",
            "ADMISSION_OR_REACHABILITY",
        ),
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        reason_codes=reason_codes,
    )


__all__ = [
    'FiniteActionRecurrenceAdjudication',
    'FiniteActionRecurrenceControlEvaluation',
    'FiniteActionRecurrenceControlMemberEvaluation',
    'FiniteActionRecurrenceDistributionSummary',
    'FiniteActionRecurrenceEfficacyUnitEvaluation',
    'RetainedActionRecurrenceDiagnostics',
    'FiniteActionRecurrenceMemberEffect',
    'FiniteActionRecurrenceResult',
    'FiniteActionRecurrenceUnitDisposition',
    'MatchedFiniteActionHoldRecurrenceEvaluator',
]

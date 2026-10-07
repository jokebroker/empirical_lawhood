"""Complete-unit reductions and categorical finite-law fitting for selective dependence response."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from math import sqrt
from typing import ClassVar, Iterable

from scipy.stats import t as student_t

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)

from .comparators import SelectiveDependenceResponseCategoricalPrediction
from .contracts import SelectiveDependenceResponseCaseState, SelectiveDependenceResponseCompleteUnitResult, SelectiveDependenceResponseConditionResult, SelectiveDependenceResponseContextDecision, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseTargetPanel, digest_ids
from .inference import classify_response
from .inference import SelectiveDependenceResponseExchangeAssessment


def _decimal(value: float) -> Decimal:
    return Decimal(format(float(value), ".12g"))


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseUnitContrast(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-unit-contrast'

    contrast_id: str
    complete_unit_id: str
    estimand_id: str
    value: Decimal | None
    native_unit: str
    evaluable: bool
    reason_code: str | None

    def __post_init__(self) -> None:
        for name in ("contrast_id", "complete_unit_id", "estimand_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.value is not None:
            validate_decimal(self.value, field_name="value")
        if self.evaluable != (self.value is not None and self.reason_code is None):
            raise ValueError("unit contrast evaluability is inconsistent")
        if self.reason_code is not None:
            validate_stable_id(self.reason_code, field_name="reason_code")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseAggregateEstimate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-aggregate-estimate'

    estimate_id: str
    estimand_id: str
    issued_complete_unit_count: int
    source_valid_complete_unit_count: int
    evaluable_complete_unit_count: int
    stopped_complete_unit_count: int
    missing_complete_unit_count: int
    point: Decimal | None
    lower: Decimal | None
    upper: Decimal | None
    native_unit: str
    confidence_level: Decimal
    complete_unit_ids_sha256: str
    nested_condition_count: int
    nested_conditions_count_as_units: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("estimate_id", "estimand_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "issued_complete_unit_count",
            "source_valid_complete_unit_count",
            "evaluable_complete_unit_count",
            "stopped_complete_unit_count",
            "missing_complete_unit_count",
            "nested_condition_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.issued_complete_unit_count != (
            self.evaluable_complete_unit_count
            + self.stopped_complete_unit_count
            + self.missing_complete_unit_count
        ):
            raise ValueError("complete-unit denominator accounting does not close")
        if self.source_valid_complete_unit_count < self.evaluable_complete_unit_count:
            raise ValueError("evaluable units exceed source-valid units")
        values = (self.point, self.lower, self.upper)
        if all(value is None for value in values):
            if self.evaluable_complete_unit_count:
                raise ValueError("evaluable aggregate lacks an estimate")
        elif any(value is None for value in values):
            raise ValueError("aggregate interval is partially missing")
        else:
            assert self.point is not None and self.lower is not None and self.upper is not None
            for name, value in (
                ("point", self.point),
                ("lower", self.lower),
                ("upper", self.upper),
            ):
                validate_decimal(value, field_name=name)
            if not self.lower <= self.point <= self.upper:
                raise ValueError("aggregate interval does not contain its point")
        validate_decimal(self.confidence_level, field_name="confidence_level")
        if not Decimal(0) < self.confidence_level < Decimal(1):
            raise ValueError("confidence level must lie inside (0, 1)")
        validate_sha256(self.complete_unit_ids_sha256, field_name="complete_unit_ids_sha256")
        if self.nested_conditions_count_as_units:
            raise ValueError("nested conditions cannot inflate aggregate replication")
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATOR_REVEAL,
            OutcomeAccess.EVALUATION_REVEALED,
        }:
            raise ValueError("aggregate estimate has invalid outcome visibility")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseFiniteLawCalibration(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-finite-law-calibration'

    calibration_id: str
    target_id: str
    fit_complete_unit_ids: tuple[str, ...]
    calibration_complete_unit_ids: tuple[str, ...]
    predictions: tuple[SelectiveDependenceResponseCategoricalPrediction, ...]
    primary_cell_ids: tuple[str, ...]
    calibration_case_count: int
    correct_case_count: int
    calibration_accuracy: Decimal
    minimum_required_accuracy: Decimal
    supported: bool
    nested_conditions_count_as_units: bool
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("calibration_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("fit_complete_unit_ids", "calibration_complete_unit_ids", "primary_cell_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if set(self.fit_complete_unit_ids) & set(self.calibration_complete_unit_ids):
            raise ValueError("law fit and calibration units overlap")
        require_sorted_unique_ids(
            self.predictions, attribute="prediction_id", field_name="predictions"
        )
        if tuple(value.cell_id for value in self.predictions) != self.primary_cell_ids:
            raise ValueError("finite-law predictions do not cover the primary cells")
        if (
            self.calibration_case_count < 1
            or not 0 <= self.correct_case_count <= self.calibration_case_count
        ):
            raise ValueError("finite-law calibration counts are invalid")
        for name in ("calibration_accuracy", "minimum_required_accuracy"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
            if getattr(self, name) > 1:
                raise ValueError(f"{name} cannot exceed one")
        if self.calibration_accuracy != Decimal(self.correct_case_count) / Decimal(
            self.calibration_case_count
        ):
            raise ValueError("calibration accuracy is not count-derived")
        if self.supported != (self.calibration_accuracy >= self.minimum_required_accuracy):
            raise ValueError("finite-law support is not calibration-derived")
        if self.nested_conditions_count_as_units:
            raise ValueError("law calibration cannot inflate nested conditions")
        if self.evaluation_outcome_count:
            raise ValueError("development law calibration cannot use evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("finite-law calibration must remain development visible")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseSelectiveDependenceSignature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-selective-dependence-signature'

    signature_id: str
    target_id: str
    estimates: tuple[SelectiveDependenceResponseAggregateEstimate, ...]
    exchange_assessments: tuple[SelectiveDependenceResponseExchangeAssessment, ...]
    active_exchange_ids: tuple[str, ...]
    invariant_exchange_ids: tuple[str, ...]
    support_boundary_case_ids: tuple[str, ...]
    complete_unit_ids_sha256: str
    all_active_supported: bool
    all_invariant_supported: bool
    support_boundary_refused: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("signature_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(self.estimates, attribute="estimate_id", field_name="estimates")
        require_sorted_unique_ids(
            self.exchange_assessments,
            attribute="assessment_id",
            field_name="exchange_assessments",
        )
        for name in (
            "active_exchange_ids",
            "invariant_exchange_ids",
            "support_boundary_case_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        observed_exchange_ids = {value.exchange_id for value in self.exchange_assessments}
        if not set((*self.active_exchange_ids, *self.invariant_exchange_ids)).issubset(
            observed_exchange_ids
        ):
            raise ValueError("selective signature lacks a declared exchange")
        validate_sha256(self.complete_unit_ids_sha256, field_name="complete_unit_ids_sha256")
        if self.outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_REVEALED,
        }:
            raise ValueError("selective signature has invalid outcome visibility")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponsePredictiveScore(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-predictive-score'

    score_id: str
    target_id: str
    model_id: str
    issued_complete_unit_count: int
    evaluable_complete_unit_count: int
    exact_complete_unit_success_count: int
    case_count: int
    correct_case_count: int
    fibre_case_count: int
    correct_fibre_count: int
    disposition_case_count: int
    correct_disposition_count: int
    context_case_count: int
    correct_context_count: int
    unsafe_false_admission_count: int
    false_safe_hold_count: int
    incorrect_case_ids: tuple[str, ...]
    unsafe_false_admission_case_ids: tuple[str, ...]
    false_safe_hold_case_ids: tuple[str, ...]
    incorrect_context_case_ids: tuple[str, ...]
    complete_unit_accuracies: tuple[NamedDecimal, ...]
    mean_complete_unit_accuracy: Decimal
    nested_cases_count_as_units: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("score_id", "target_id", "model_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "issued_complete_unit_count",
            "evaluable_complete_unit_count",
            "exact_complete_unit_success_count",
            "case_count",
            "correct_case_count",
            "fibre_case_count",
            "correct_fibre_count",
            "disposition_case_count",
            "correct_disposition_count",
            "context_case_count",
            "correct_context_count",
            "unsafe_false_admission_count",
            "false_safe_hold_count",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        if not 0 <= self.correct_case_count <= self.case_count:
            raise ValueError("predictive case counts are invalid")
        if self.fibre_case_count != self.case_count or not (
            0 <= self.correct_fibre_count <= self.fibre_case_count
        ):
            raise ValueError("predictive action-fibre counts are invalid")
        if self.disposition_case_count != self.case_count or not (
            0 <= self.correct_disposition_count <= self.disposition_case_count
        ):
            raise ValueError("predictive disposition counts are invalid")
        if not 0 <= self.correct_context_count <= self.context_case_count:
            raise ValueError("predictive context-decision counts are invalid")
        if not 0 <= self.exact_complete_unit_success_count <= self.evaluable_complete_unit_count:
            raise ValueError("predictive complete-unit counts are invalid")
        for name in (
            "incorrect_case_ids",
            "unsafe_false_admission_case_ids",
            "false_safe_hold_case_ids",
            "incorrect_context_case_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if self.unsafe_false_admission_count != len(self.unsafe_false_admission_case_ids):
            raise ValueError("unsafe false-admission count and identities differ")
        if self.false_safe_hold_count != len(self.false_safe_hold_case_ids):
            raise ValueError("false safe-hold count and identities differ")
        if len(self.incorrect_case_ids) != self.case_count - self.correct_case_count:
            raise ValueError("incorrect predictive case identities do not close")
        if len(self.incorrect_context_case_ids) != (
            self.context_case_count - self.correct_context_count
        ):
            raise ValueError("incorrect context-decision identities do not close")
        if not set(
            (*self.unsafe_false_admission_case_ids, *self.false_safe_hold_case_ids)
        ).issubset(self.incorrect_case_ids):
            raise ValueError("safety errors must also be predictive errors")
        require_sorted_unique_ids(
            self.complete_unit_accuracies,
            attribute="value_id",
            field_name="complete_unit_accuracies",
        )
        if len(self.complete_unit_accuracies) != self.evaluable_complete_unit_count:
            raise ValueError("complete-unit score roster differs")
        validate_decimal(
            self.mean_complete_unit_accuracy,
            field_name="mean_complete_unit_accuracy",
            minimum=Decimal(0),
        )
        if self.mean_complete_unit_accuracy > 1:
            raise ValueError("mean complete-unit accuracy exceeds one")
        if self.nested_cases_count_as_units:
            raise ValueError("nested score cases cannot inflate replication")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("predictive scoring requires revealed evaluation outcomes")


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseDevelopmentPredictionMetrics:
    """Development-only complete-unit metrics used by the frozen power design."""

    complete_unit_accuracies: tuple[NamedDecimal, ...]
    unsafe_false_admission_unit_ids: tuple[str, ...]
    false_safe_hold_unit_ids: tuple[str, ...]
    support_boundary_error_unit_ids: tuple[str, ...]


def _receiver(condition: SelectiveDependenceResponseConditionResult, receiver_id: str) -> Decimal | None:
    matches = [value for value in condition.receivers if value.receiver_id == receiver_id]
    if len(matches) != 1 or not matches[0].valid:
        return None
    return matches[0].value


def _select_condition(
    unit: SelectiveDependenceResponseCompleteUnitResult,
    *,
    denominator_id: str,
    history_id: str,
    action_id: str,
    horizon_id: str,
) -> SelectiveDependenceResponseConditionResult | None:
    matches = [
        value
        for value in unit.conditions
        if (
            value.denominator_id,
            value.history_id,
            value.action_id,
            value.horizon_id,
        )
        == (denominator_id, history_id, action_id, horizon_id)
    ]
    return matches[0] if len(matches) == 1 else None


def matched_action_minus_hold(
    unit: SelectiveDependenceResponseCompleteUnitResult,
    *,
    estimand_id: str,
    denominator_id: str,
    history_id: str,
    action_id: str,
    hold_action_id: str,
    horizon_id: str,
    receiver_id: str,
    native_unit: str,
) -> SelectiveDependenceResponseUnitContrast:
    action = _select_condition(
        unit,
        denominator_id=denominator_id,
        history_id=history_id,
        action_id=action_id,
        horizon_id=horizon_id,
    )
    hold = _select_condition(
        unit,
        denominator_id=denominator_id,
        history_id=history_id,
        action_id=hold_action_id,
        horizon_id=horizon_id,
    )
    action_value = None if action is None or action.stopped else _receiver(action, receiver_id)
    hold_value = None if hold is None or hold.stopped else _receiver(hold, receiver_id)
    if action_value is None or hold_value is None:
        value = None
        reason = "missing-or-stopped-branch"
    else:
        value = action_value - hold_value
        reason = None
    return SelectiveDependenceResponseUnitContrast(
        contrast_id=f"contrast.{unit.complete_unit_id}.{estimand_id}",
        complete_unit_id=unit.complete_unit_id,
        estimand_id=estimand_id,
        value=value,
        native_unit=native_unit,
        evaluable=value is not None,
        reason_code=reason,
    )


def difference_of_contrasts(
    left: SelectiveDependenceResponseUnitContrast,
    right: SelectiveDependenceResponseUnitContrast,
    *,
    estimand_id: str,
) -> SelectiveDependenceResponseUnitContrast:
    if left.complete_unit_id != right.complete_unit_id or left.native_unit != right.native_unit:
        raise ValueError("exchange contrasts must be matched within one complete unit")
    if left.value is None or right.value is None:
        value = None
        reason = "missing-or-stopped-exchange"
    else:
        value = left.value - right.value
        reason = None
    return SelectiveDependenceResponseUnitContrast(
        contrast_id=f"contrast.{left.complete_unit_id}.{estimand_id}",
        complete_unit_id=left.complete_unit_id,
        estimand_id=estimand_id,
        value=value,
        native_unit=left.native_unit,
        evaluable=value is not None,
        reason_code=reason,
    )


def aggregate_complete_units(
    *,
    estimate_id: str,
    contrasts: tuple[SelectiveDependenceResponseUnitContrast, ...],
    issued_complete_unit_ids: tuple[str, ...],
    nested_condition_count: int,
    outcome_access: OutcomeAccess,
    confidence_level: Decimal = Decimal("0.95"),
) -> SelectiveDependenceResponseAggregateEstimate:
    require_sorted_unique_strings(
        issued_complete_unit_ids, field_name="issued_complete_unit_ids", allow_empty=False
    )
    by_unit = {value.complete_unit_id: value for value in contrasts}
    if len(by_unit) != len(contrasts) or not set(by_unit).issubset(issued_complete_unit_ids):
        raise ValueError("contrast roster is duplicate or outside the issued units")
    evaluable = [float(value.value) for value in contrasts if value.value is not None]
    missing = len(issued_complete_unit_ids) - len(contrasts)
    stopped = sum(value.value is None for value in contrasts)
    point: Decimal | None
    lower: Decimal | None
    upper: Decimal | None
    if evaluable:
        point_float = sum(evaluable) / len(evaluable)
        if len(evaluable) == 1:
            lower_float = upper_float = point_float
        else:
            variance = sum((value - point_float) ** 2 for value in evaluable) / (len(evaluable) - 1)
            critical = student_t.ppf(
                (1 + float(confidence_level)) / 2,
                df=len(evaluable) - 1,
            )
            half = float(critical) * sqrt(variance / len(evaluable))
            lower_float, upper_float = point_float - half, point_float + half
        point, lower, upper = map(_decimal, (point_float, lower_float, upper_float))
    else:
        point = lower = upper = None
    source_valid = len(contrasts)
    first = next(iter(contrasts), None)
    return SelectiveDependenceResponseAggregateEstimate(
        estimate_id=estimate_id,
        estimand_id=(first.estimand_id if first is not None else estimate_id),
        issued_complete_unit_count=len(issued_complete_unit_ids),
        source_valid_complete_unit_count=source_valid,
        evaluable_complete_unit_count=len(evaluable),
        stopped_complete_unit_count=stopped,
        missing_complete_unit_count=missing,
        point=point,
        lower=lower,
        upper=upper,
        native_unit=(first.native_unit if first is not None else "unknown"),
        confidence_level=confidence_level,
        complete_unit_ids_sha256=digest_ids(issued_complete_unit_ids),
        nested_condition_count=nested_condition_count,
        nested_conditions_count_as_units=False,
        outcome_access=outcome_access,
    )


def bonferroni_confidence_level(
    *,
    familywise_alpha: Decimal,
    family_size: int,
) -> Decimal:
    """Return the two-sided per-estimand confidence level for a finite family."""

    validate_decimal(
        familywise_alpha,
        field_name="familywise_alpha",
        minimum=Decimal(0),
    )
    if not Decimal(0) < familywise_alpha < Decimal(1):
        raise ValueError("familywise alpha must lie inside (0, 1)")
    if family_size < 1:
        raise ValueError("simultaneous family must contain at least one estimand")
    return Decimal(1) - familywise_alpha / Decimal(family_size)


def condition_cell_id(condition: SelectiveDependenceResponseConditionResult, *, receiver_id: str | None = None) -> str:
    base = (
        f"cell.{condition.denominator_id}.{condition.history_id}."
        f"{condition.action_id}.{condition.horizon_id}"
    )
    return base if receiver_id is None else f"{base}.receiver-{receiver_id}"


def derive_context_decisions_from_predictions(
    predictions: tuple[SelectiveDependenceResponseCategoricalPrediction, ...],
    *,
    hold_action_id: str,
    decision_id_prefix: str,
) -> tuple[SelectiveDependenceResponseContextDecision, ...]:
    """Derive active-set/hold/policy forecasts from exact action-fibre calls."""

    validate_stable_id(hold_action_id, field_name="hold_action_id")
    validate_stable_id(decision_id_prefix, field_name="decision_id_prefix")
    grouped: dict[
        tuple[str, str, str],
        dict[str, list[tuple[SelectiveDependenceResponseCaseState, bool | None]]],
    ] = {}
    for prediction in predictions:
        denominator_id, history_id, action_id, horizon_id, _ = _cell_roles_for_score(
            prediction.cell_id
        )
        grouped.setdefault((denominator_id, history_id, horizon_id), {}).setdefault(
            action_id, []
        ).append((prediction.response_state, prediction.fibre_admitted))
    decisions = []
    for (denominator_id, history_id, horizon_id), by_action in sorted(grouped.items()):
        if hold_action_id not in by_action:
            raise ValueError("categorical predictions omit the native hold fibre")

        def consistent_fibre(action_id: str) -> bool | None:
            values = {value[1] for value in by_action[action_id]}
            return next(iter(values)) if len(values) == 1 else None

        hold_viable = consistent_fibre(hold_action_id)
        admitted_active = []
        active_complete = True
        for action_id, values in by_action.items():
            if action_id == hold_action_id:
                continue
            states = {value[0] for value in values}
            if states == {SelectiveDependenceResponseCaseState.OUTSIDE_SUPPORT}:
                continue
            if SelectiveDependenceResponseCaseState.OUTSIDE_SUPPORT in states:
                active_complete = False
                continue
            fibre = consistent_fibre(action_id)
            if fibre is None:
                active_complete = False
            elif fibre:
                admitted_active.append(action_id)
        admitted = tuple(sorted(admitted_active))
        disposition = (
            SelectiveDependenceResponseDisposition.ACTION_AVAILABLE
            if admitted
            else (
                SelectiveDependenceResponseDisposition.UNEVALUABLE
                if not active_complete or hold_viable is None
                else (
                    SelectiveDependenceResponseDisposition.HOLD_ONLY if hold_viable else SelectiveDependenceResponseDisposition.NONATTEMPT
                )
            )
        )
        decisions.append(
            SelectiveDependenceResponseContextDecision(
                decision_id=(f"{decision_id_prefix}.{denominator_id}.{history_id}.{horizon_id}"),
                denominator_id=denominator_id,
                history_id=history_id,
                horizon_id=horizon_id,
                admitted_active_action_ids=admitted,
                active_fibres_complete=active_complete,
                hold_action_id=hold_action_id,
                hold_viable=hold_viable,
                disposition=disposition,
            )
        )
    return tuple(sorted(decisions, key=lambda value: value.decision_id))


def _categorical_observation(
    unit: SelectiveDependenceResponseCompleteUnitResult,
    condition: SelectiveDependenceResponseConditionResult,
    *,
    hold_action_id: str,
    receiver_id: str,
    neutral_margin: Decimal,
) -> tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition] | None:
    if condition.support_state == "outside-support":
        return (
            SelectiveDependenceResponseCaseState.OUTSIDE_SUPPORT,
            condition.fibre_admitted,
            condition.disposition,
        )
    if condition.stopped:
        return (
            SelectiveDependenceResponseCaseState.UNEVALUABLE,
            None,
            SelectiveDependenceResponseDisposition.UNEVALUABLE,
        )
    hold = _select_condition(
        unit,
        denominator_id=condition.denominator_id,
        history_id=condition.history_id,
        action_id=hold_action_id,
        horizon_id=condition.horizon_id,
    )
    value = _receiver(condition, receiver_id)
    hold_value = None if hold is None else _receiver(hold, receiver_id)
    if value is None or hold_value is None:
        return None
    return (
        classify_response(value - hold_value, neutral_margin=neutral_margin),
        condition.fibre_admitted,
        condition.disposition,
    )


def fit_categorical_finite_law(
    panel: SelectiveDependenceResponseTargetPanel,
    *,
    hold_action_id: str,
    receiver_ids: tuple[str, ...],
    neutral_margin: Decimal,
    primary_cell_ids: tuple[str, ...],
    minimum_required_accuracy: Decimal = Decimal("0.70"),
) -> SelectiveDependenceResponseFiniteLawCalibration:
    """Fit on one complete-unit half and calibrate on the disjoint other half."""

    require_sorted_unique_strings(
        primary_cell_ids, field_name="primary_cell_ids", allow_empty=False
    )
    require_sorted_unique_strings(receiver_ids, field_name="receiver_ids", allow_empty=False)
    midpoint = len(panel.complete_units) // 2
    fit_units = panel.complete_units[:midpoint]
    calibration_units = panel.complete_units[midpoint:]
    if not fit_units or not calibration_units:
        raise ValueError("finite-law split requires at least two complete units")
    by_cell: dict[str, list[tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]]] = {
        cell_id: [] for cell_id in primary_cell_ids
    }
    for unit in fit_units:
        for condition in unit.conditions:
            for receiver_id in receiver_ids:
                cell_id = condition_cell_id(condition, receiver_id=receiver_id)
                if cell_id not in by_cell:
                    continue
                observed = _categorical_observation(
                    unit,
                    condition,
                    hold_action_id=hold_action_id,
                    receiver_id=receiver_id,
                    neutral_margin=neutral_margin,
                )
                if observed is not None:
                    by_cell[cell_id].append(observed)
    predictions = []
    for cell_id in primary_cell_ids:
        if not by_cell[cell_id]:
            state, fibre_admitted, disposition = (
                SelectiveDependenceResponseCaseState.UNEVALUABLE,
                None,
                SelectiveDependenceResponseDisposition.UNEVALUABLE,
            )
        else:
            counts = Counter(by_cell[cell_id])
            state, fibre_admitted, disposition = sorted(
                counts,
                key=lambda value: (
                    -counts[value],
                    value[0].value,
                    "none" if value[1] is None else str(value[1]),
                    value[2].value,
                ),
            )[0]
        predictions.append(
            SelectiveDependenceResponseCategoricalPrediction(
                prediction_id=f"prediction.{panel.target_id}.{cell_id}",
                cell_id=cell_id,
                response_state=state,
                fibre_admitted=fibre_admitted,
                disposition=disposition,
            )
        )
    forecast = {
        value.cell_id: (value.response_state, value.fibre_admitted, value.disposition)
        for value in predictions
    }
    total = correct = 0
    for unit in calibration_units:
        for condition in unit.conditions:
            for receiver_id in receiver_ids:
                cell_id = condition_cell_id(condition, receiver_id=receiver_id)
                if cell_id not in forecast:
                    continue
                observed = _categorical_observation(
                    unit,
                    condition,
                    hold_action_id=hold_action_id,
                    receiver_id=receiver_id,
                    neutral_margin=neutral_margin,
                )
                if observed is not None:
                    total += 1
                    correct += observed == forecast[cell_id]
    if not total:
        raise ValueError("finite-law calibration has no evaluable cases")
    ordered_predictions = tuple(sorted(predictions, key=lambda value: value.cell_id))
    return SelectiveDependenceResponseFiniteLawCalibration(
        calibration_id=f"calibration.{panel.target_id}.finite-law",
        target_id=panel.target_id,
        fit_complete_unit_ids=tuple(value.complete_unit_id for value in fit_units),
        calibration_complete_unit_ids=tuple(value.complete_unit_id for value in calibration_units),
        predictions=ordered_predictions,
        primary_cell_ids=tuple(value.cell_id for value in ordered_predictions),
        calibration_case_count=total,
        correct_case_count=correct,
        calibration_accuracy=Decimal(correct) / Decimal(total),
        minimum_required_accuracy=minimum_required_accuracy,
        supported=Decimal(correct) / Decimal(total) >= minimum_required_accuracy,
        nested_conditions_count_as_units=False,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


def split_candidate_calibration(
    panel: SelectiveDependenceResponseTargetPanel,
    *,
    fit_complete_unit_ids: tuple[str, ...],
    calibration_complete_unit_ids: tuple[str, ...],
    split_variable_id: str,
    split_threshold: Decimal,
    hold_action_id: str,
    receiver_ids: tuple[str, ...],
    neutral_margin: Decimal,
    primary_cell_ids: tuple[str, ...],
) -> tuple[int, int, Decimal]:
    """Calibrate a richer native-preparation split on the existing unit split.

    This is a denominator-candidate diagnostic, not a hidden forecast fit. A
    missing split variable, cell, or categorical answer counts against the
    candidate rather than shrinking its denominator.
    """

    validate_stable_id(split_variable_id, field_name="split_variable_id")
    validate_decimal(split_threshold, field_name="split_threshold")
    fit_ids = set(fit_complete_unit_ids)
    calibration_ids = set(calibration_complete_unit_ids)
    if fit_ids & calibration_ids or not fit_ids or not calibration_ids:
        raise ValueError("split candidate requires disjoint nonempty unit rosters")
    by_id = {value.complete_unit_id: value for value in panel.complete_units}
    if set(by_id) != fit_ids | calibration_ids:
        raise ValueError("split candidate rosters differ from the development panel")

    def stratum(unit: SelectiveDependenceResponseCompleteUnitResult) -> str:
        matches = [
            value for value in unit.preparation_values if value.value_id == split_variable_id
        ]
        if len(matches) != 1:
            raise ValueError("split candidate preparation variable is missing or duplicated")
        return "lower" if matches[0].value <= split_threshold else "upper"

    fitted: dict[
        tuple[str, str],
        list[tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]],
    ] = {}
    for unit_id in sorted(fit_ids):
        unit = by_id[unit_id]
        unit_stratum = stratum(unit)
        for condition in unit.conditions:
            for receiver_id in receiver_ids:
                cell_id = condition_cell_id(condition, receiver_id=receiver_id)
                if cell_id not in primary_cell_ids:
                    continue
                observed = _categorical_observation(
                    unit,
                    condition,
                    hold_action_id=hold_action_id,
                    receiver_id=receiver_id,
                    neutral_margin=neutral_margin,
                )
                if observed is not None:
                    fitted.setdefault((unit_stratum, cell_id), []).append(observed)

    def modal(
        values: list[tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]],
    ) -> tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]:
        counts = Counter(values)
        return sorted(
            counts,
            key=lambda value: (
                -counts[value],
                value[0].value,
                "none" if value[1] is None else str(value[1]),
                value[2].value,
            ),
        )[0]

    predictions = {key: modal(values) for key, values in fitted.items() if values}
    total = correct = 0
    for unit_id in sorted(calibration_ids):
        unit = by_id[unit_id]
        unit_stratum = stratum(unit)
        for condition in unit.conditions:
            for receiver_id in receiver_ids:
                cell_id = condition_cell_id(condition, receiver_id=receiver_id)
                if cell_id not in primary_cell_ids:
                    continue
                observed = _categorical_observation(
                    unit,
                    condition,
                    hold_action_id=hold_action_id,
                    receiver_id=receiver_id,
                    neutral_margin=neutral_margin,
                )
                if observed is not None:
                    total += 1
                    correct += predictions.get((unit_stratum, cell_id)) == observed
    if not total:
        raise ValueError("split candidate has no calibration cases")
    return total, correct, Decimal(correct) / Decimal(total)


def development_prediction_metrics(
    panel: SelectiveDependenceResponseTargetPanel,
    *,
    predictions: tuple[SelectiveDependenceResponseCategoricalPrediction, ...],
    complete_unit_ids: tuple[str, ...],
    hold_action_id: str,
    receiver_ids: tuple[str, ...],
    neutral_margin: Decimal,
) -> SelectiveDependenceResponseDevelopmentPredictionMetrics:
    """Score a frozen prediction only on named development complete units."""

    if panel.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
        raise ValueError("development power metrics cannot inspect another evidence phase")
    require_sorted_unique_strings(
        complete_unit_ids, field_name="complete_unit_ids", allow_empty=False
    )
    requested = set(complete_unit_ids)
    if not requested.issubset(panel.expected_complete_unit_ids):
        raise ValueError("development metric roster lies outside the panel")
    forecast = {
        value.cell_id: (value.response_state, value.fibre_admitted, value.disposition)
        for value in predictions
    }
    if len(forecast) != len(predictions) or not forecast:
        raise ValueError("development metrics require unique nonempty predictions")
    unit_scores = []
    unsafe_units = []
    false_hold_units = []
    support_units = []
    for unit in panel.complete_units:
        if unit.complete_unit_id not in requested:
            continue
        observed_by_cell: dict[str, tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]] = {}
        for condition in unit.conditions:
            for receiver_id in receiver_ids:
                cell_id = condition_cell_id(condition, receiver_id=receiver_id)
                if cell_id not in forecast:
                    continue
                observed = _categorical_observation(
                    unit,
                    condition,
                    hold_action_id=hold_action_id,
                    receiver_id=receiver_id,
                    neutral_margin=neutral_margin,
                )
                if observed is not None:
                    observed_by_cell[cell_id] = observed
        if set(observed_by_cell) != set(forecast) or any(
            state is SelectiveDependenceResponseCaseState.UNEVALUABLE
            or fibre_admitted is None
            or disposition is SelectiveDependenceResponseDisposition.UNEVALUABLE
            for state, fibre_admitted, disposition in observed_by_cell.values()
        ):
            continue
        correct = 0
        unsafe = false_hold = support_error = False
        for cell_id, actual in observed_by_cell.items():
            predicted = forecast[cell_id]
            correct += predicted == actual
            _, _, action_id, _, _ = _cell_roles_for_score(cell_id)
            unsafe |= action_id != hold_action_id and predicted[1] is True and actual[1] is False
            false_hold |= (
                action_id == hold_action_id and predicted[1] is True and actual[1] is False
            )
            if actual[0] is SelectiveDependenceResponseCaseState.OUTSIDE_SUPPORT:
                support_error |= not (
                    predicted[0] is SelectiveDependenceResponseCaseState.OUTSIDE_SUPPORT and predicted[1] is False
                )
        unit_scores.append(
            NamedDecimal(
                value_id=unit.complete_unit_id,
                value=Decimal(correct) / Decimal(len(forecast)),
                unit="fraction-correct",
            )
        )
        if unsafe:
            unsafe_units.append(unit.complete_unit_id)
        if false_hold:
            false_hold_units.append(unit.complete_unit_id)
        if support_error:
            support_units.append(unit.complete_unit_id)
    return SelectiveDependenceResponseDevelopmentPredictionMetrics(
        complete_unit_accuracies=tuple(sorted(unit_scores, key=lambda value: value.value_id)),
        unsafe_false_admission_unit_ids=tuple(sorted(unsafe_units)),
        false_safe_hold_unit_ids=tuple(sorted(false_hold_units)),
        support_boundary_error_unit_ids=tuple(sorted(support_units)),
    )


def sample_standard_deviation(values: tuple[Decimal, ...]) -> Decimal:
    """Return the complete-unit sample standard deviation without row inflation."""

    if len(values) < 2:
        raise ValueError("sample standard deviation needs at least two complete units")
    numeric = tuple(float(value) for value in values)
    mean = sum(numeric) / len(numeric)
    variance = sum((value - mean) ** 2 for value in numeric) / (len(numeric) - 1)
    return _decimal(sqrt(variance))


def target_sink_conflict_case_ids(
    panel: SelectiveDependenceResponseTargetPanel,
    *,
    hold_action_id: str,
    target_margin_id: str,
    sink_margin_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Name active fibres where target passes but a noncompensating sink fails."""

    validate_stable_id(hold_action_id, field_name="hold_action_id")
    validate_stable_id(target_margin_id, field_name="target_margin_id")
    require_sorted_unique_strings(sink_margin_ids, field_name="sink_margin_ids", allow_empty=False)
    conflicts = []
    for unit in panel.complete_units:
        for condition in unit.conditions:
            if (
                condition.action_id == hold_action_id
                or condition.support_state == "outside-support"
                or condition.stopped
            ):
                continue
            margins = {value.value_id: value.value for value in condition.gate_margins}
            if target_margin_id not in margins or not set(sink_margin_ids).issubset(margins):
                raise ValueError("target/sink conflict audit lacks a frozen gate margin")
            if margins[target_margin_id] >= 0 and any(
                margins[value] < 0 for value in sink_margin_ids
            ):
                conflicts.append(
                    f"case.{unit.complete_unit_id}.{condition.condition_id.removeprefix('condition.')}"
                )
    return tuple(sorted(conflicts))


def score_categorical_predictions(
    panel: SelectiveDependenceResponseTargetPanel,
    *,
    model_id: str,
    predictions: tuple[SelectiveDependenceResponseCategoricalPrediction, ...],
    hold_action_id: str,
    receiver_ids: tuple[str, ...],
    neutral_margin: Decimal,
) -> SelectiveDependenceResponsePredictiveScore:
    """Score on complete units; nested cells remain within-unit diagnostics."""

    validate_stable_id(model_id, field_name="model_id")
    require_sorted_unique_strings(receiver_ids, field_name="receiver_ids", allow_empty=False)
    forecast = {
        value.cell_id: (value.response_state, value.fibre_admitted, value.disposition)
        for value in predictions
    }
    if len(forecast) != len(predictions) or not forecast:
        raise ValueError("predictive score requires unique nonempty cells")
    unit_scores = []
    incorrect_case_ids: list[str] = []
    unsafe_case_ids: list[str] = []
    false_hold_case_ids: list[str] = []
    incorrect_context_ids: list[str] = []
    total_cases = correct_cases = correct_fibres = correct_dispositions = 0
    total_contexts = correct_contexts = 0
    unsafe = false_hold = exact_units = 0
    predicted_contexts = derive_context_decisions_from_predictions(
        predictions,
        hold_action_id=hold_action_id,
        decision_id_prefix="score-decision",
    )
    predicted_context_by_key = {
        (value.denominator_id, value.history_id, value.horizon_id): value
        for value in predicted_contexts
    }
    for unit in panel.complete_units:
        observed_by_cell: dict[str, tuple[SelectiveDependenceResponseCaseState, bool | None, SelectiveDependenceResponseDisposition]] = {}
        for condition in unit.conditions:
            for receiver_id in receiver_ids:
                cell_id = condition_cell_id(condition, receiver_id=receiver_id)
                if cell_id not in forecast:
                    continue
                observed = _categorical_observation(
                    unit,
                    condition,
                    hold_action_id=hold_action_id,
                    receiver_id=receiver_id,
                    neutral_margin=neutral_margin,
                )
                if observed is not None:
                    observed_by_cell[cell_id] = observed
        if set(observed_by_cell) != set(forecast) or any(
            state is SelectiveDependenceResponseCaseState.UNEVALUABLE
            or fibre_admitted is None
            or disposition is SelectiveDependenceResponseDisposition.UNEVALUABLE
            for state, fibre_admitted, disposition in observed_by_cell.values()
        ):
            continue
        unit_correct = 0
        for cell_id, actual in observed_by_cell.items():
            predicted = forecast[cell_id]
            unit_correct += predicted == actual
            case_id = f"case.{unit.complete_unit_id}.{cell_id.removeprefix('cell.')}"
            if predicted != actual:
                incorrect_case_ids.append(case_id)
            correct_fibres += predicted[1] is actual[1]
            correct_dispositions += predicted[2] is actual[2]
            _, _, action_id, _, _ = _cell_roles_for_score(cell_id)
            if action_id != hold_action_id and predicted[1] is True and actual[1] is False:
                unsafe += 1
                unsafe_case_ids.append(case_id)
            if action_id == hold_action_id and predicted[1] is True and actual[1] is False:
                false_hold += 1
                false_hold_case_ids.append(case_id)
        unit_context_correct = 0
        for actual_context in unit.context_decisions:
            key = (
                actual_context.denominator_id,
                actual_context.history_id,
                actual_context.horizon_id,
            )
            predicted_context = predicted_context_by_key.get(key)
            correct_context = predicted_context is not None and (
                predicted_context.admitted_active_action_ids,
                predicted_context.active_fibres_complete,
                predicted_context.hold_action_id,
                predicted_context.hold_viable,
                predicted_context.disposition,
            ) == (
                actual_context.admitted_active_action_ids,
                actual_context.active_fibres_complete,
                actual_context.hold_action_id,
                actual_context.hold_viable,
                actual_context.disposition,
            )
            unit_context_correct += correct_context
            if not correct_context:
                incorrect_context_ids.append(
                    f"context.{unit.complete_unit_id}.{key[0]}.{key[1]}.{key[2]}"
                )
        total_cases += len(forecast)
        correct_cases += unit_correct
        total_contexts += len(unit.context_decisions)
        correct_contexts += unit_context_correct
        exact_units += unit_correct == len(forecast) and unit_context_correct == len(
            unit.context_decisions
        )
        unit_scores.append(
            NamedDecimal(
                value_id=unit.complete_unit_id,
                value=Decimal(unit_correct) / Decimal(len(forecast)),
                unit="fraction-correct",
            )
        )
    ordered_scores = tuple(sorted(unit_scores, key=lambda value: value.value_id))
    mean_accuracy = (
        sum((value.value for value in ordered_scores), Decimal(0)) / Decimal(len(ordered_scores))
        if ordered_scores
        else Decimal(0)
    )
    return SelectiveDependenceResponsePredictiveScore(
        score_id=f"score.{panel.target_id}.{model_id}",
        target_id=panel.target_id,
        model_id=model_id,
        issued_complete_unit_count=len(panel.expected_complete_unit_ids),
        evaluable_complete_unit_count=len(ordered_scores),
        exact_complete_unit_success_count=exact_units,
        case_count=total_cases,
        correct_case_count=correct_cases,
        fibre_case_count=total_cases,
        correct_fibre_count=correct_fibres,
        disposition_case_count=total_cases,
        correct_disposition_count=correct_dispositions,
        context_case_count=total_contexts,
        correct_context_count=correct_contexts,
        unsafe_false_admission_count=unsafe,
        false_safe_hold_count=false_hold,
        incorrect_case_ids=tuple(sorted(incorrect_case_ids)),
        unsafe_false_admission_case_ids=tuple(sorted(unsafe_case_ids)),
        false_safe_hold_case_ids=tuple(sorted(false_hold_case_ids)),
        incorrect_context_case_ids=tuple(sorted(incorrect_context_ids)),
        complete_unit_accuracies=ordered_scores,
        mean_complete_unit_accuracy=mean_accuracy,
        nested_cases_count_as_units=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


def _cell_roles_for_score(cell_id: str) -> tuple[str, str, str, str, str]:
    parts = cell_id.split(".")
    if len(parts) != 6 or parts[0] != "cell" or not parts[5].startswith("receiver-"):
        raise ValueError("score cell does not expose D/H/A/tau/R")
    return parts[1], parts[2], parts[3], parts[4], parts[5].removeprefix("receiver-")


def build_panel(
    *,
    panel_id: str,
    target_id: str,
    complete_units: Iterable[SelectiveDependenceResponseCompleteUnitResult],
) -> SelectiveDependenceResponseTargetPanel:
    ordered = tuple(sorted(complete_units, key=lambda value: value.complete_unit_id))
    if not ordered:
        raise ValueError("target panel cannot be empty")
    phase = ordered[0].phase
    ids = tuple(value.complete_unit_id for value in ordered)
    return SelectiveDependenceResponseTargetPanel(
        panel_id=panel_id,
        target_id=target_id,
        phase=phase,
        complete_units=ordered,
        expected_complete_unit_ids=ids,
        expected_complete_unit_ids_sha256=digest_ids(ids),
        nested_conditions_count_as_units=False,
        outcome_access=ordered[0].outcome_access,
    )


__all__ = [
    'SelectiveDependenceResponseAggregateEstimate',
    'SelectiveDependenceResponseFiniteLawCalibration',
    'SelectiveDependenceResponseSelectiveDependenceSignature',
    'SelectiveDependenceResponsePredictiveScore',
    'SelectiveDependenceResponseUnitContrast',
    "aggregate_complete_units",
    "bonferroni_confidence_level",
    "build_panel",
    "condition_cell_id",
    "development_prediction_metrics",
    "derive_context_decisions_from_predictions",
    "difference_of_contrasts",
    "fit_categorical_finite_law",
    "matched_action_minus_hold",
    "sample_standard_deviation",
    "score_categorical_predictions",
    "split_candidate_calibration",
    "target_sink_conflict_case_ids",
]

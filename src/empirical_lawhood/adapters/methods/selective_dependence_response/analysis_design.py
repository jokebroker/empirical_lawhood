"""Outcome-blind target-analysis freeze and generic selective dependence response reductions.

The target adapters own the finite values in this record.  The reusable method
code owns their validation and execution.  Consequently an issued analysis is
not allowed to recover thresholds, cell inclusion or exchange operators from a
development panel or from target-specific reduction code.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)

from .analysis import SelectiveDependenceResponseFiniteLawCalibration, SelectiveDependenceResponseSelectiveDependenceSignature, SelectiveDependenceResponseUnitContrast, aggregate_complete_units, bonferroni_confidence_level, condition_cell_id, difference_of_contrasts, fit_categorical_finite_law, matched_action_minus_hold
from .comparators import SelectiveDependenceResponseComparatorKind
from .contracts import SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS, SelectiveDependenceResponseCompleteUnitResult, SelectiveDependenceResponseConditionResult, SelectiveDependenceResponseDisposition, SelectiveDependenceResponseExchangeExpectation, SelectiveDependenceResponseExchangeForecast, SelectiveDependenceResponsePreparationDistributionFreeze, SelectiveDependenceResponseTargetPanel, digest_ids
from .inference import SelectiveDependenceResponseExchangeState, assess_exchange


class SelectiveDependenceResponseExchangeReductionKind(StrEnum):
    """Closed finite set of admissible complete-unit exchange operators."""

    DIFFERENCE_OF_ACTION_MINUS_HOLD = "DIFFERENCE_OF_ACTION_MINUS_HOLD"
    DIRECT_RECEIVER_DIFFERENCE = "DIRECT_RECEIVER_DIFFERENCE"
    SUPPORT_REFUSAL = "SUPPORT_REFUSAL"


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseExchangeDesign(CanonicalRecord):
    """One pre-outcome exchange and its exact reduction semantics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-exchange-design'

    exchange_id: str
    target_id: str
    expectation: SelectiveDependenceResponseExchangeExpectation
    coordinate_role_id: str
    left_cell_id: str
    right_cell_id: str
    estimand_id: str
    equivalence_margin: Decimal
    minimum_active_difference: Decimal
    native_unit: str
    reduction_kind: SelectiveDependenceResponseExchangeReductionKind

    def __post_init__(self) -> None:
        for name in (
            "exchange_id",
            "target_id",
            "left_cell_id",
            "right_cell_id",
            "estimand_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.coordinate_role_id not in SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS:
            raise ValueError("exchange design coordinate is outside D/H/A/R/tau")
        for name in ("equivalence_margin", "minimum_active_difference"):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.left_cell_id == self.right_cell_id:
            raise ValueError("exchange design endpoints must differ")
        left = parse_scored_cell_id(self.left_cell_id)
        right = parse_scored_cell_id(self.right_cell_id)
        changed_roles = tuple(
            role
            for role, left_value, right_value in zip(
                SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS,
                # Parsed order is D, H, A, tau, R; canonical role order is A, D, H, R, tau.
                (left[2], left[0], left[1], left[4], left[3]),
                (right[2], right[0], right[1], right[4], right[3]),
                strict=True,
            )
            if left_value != right_value
        )
        if changed_roles != (self.coordinate_role_id,):
            raise ValueError("exchange endpoints do not vary exactly the declared coordinate")
        support = self.expectation is SelectiveDependenceResponseExchangeExpectation.SUPPORT_BOUNDARY
        if support != (self.reduction_kind is SelectiveDependenceResponseExchangeReductionKind.SUPPORT_REFUSAL):
            raise ValueError("support expectation and support-refusal operator differ")
        if support and self.native_unit != "support-class":
            raise ValueError("support-refusal exchange must retain support-class units")

    def as_forecast(self) -> SelectiveDependenceResponseExchangeForecast:
        return SelectiveDependenceResponseExchangeForecast(
            exchange_id=self.exchange_id,
            target_id=self.target_id,
            expectation=self.expectation,
            coordinate_role_id=self.coordinate_role_id,
            left_cell_id=self.left_cell_id,
            right_cell_id=self.right_cell_id,
            estimand_id=self.estimand_id,
            equivalence_margin=self.equivalence_margin,
            minimum_active_difference=self.minimum_active_difference,
            native_unit=self.native_unit,
        )


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseTargetAnalysisFreeze(CanonicalRecord):
    """Canonical pre-development target analysis and adjudication surface."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-analysis-freeze'

    freeze_id: str
    target_id: str
    design: ObjectIdentity
    preparation_freeze: ObjectIdentity
    signature_id: str
    hold_action_id: str
    scored_receiver_ids: tuple[str, ...]
    primary_cell_ids: tuple[str, ...]
    support_boundary_cell_ids: tuple[str, ...]
    neutral_margin: Decimal
    minimum_law_accuracy: Decimal
    exchange_familywise_alpha: Decimal
    exchange_interval_method: str
    exchanges: tuple[SelectiveDependenceResponseExchangeDesign, ...]
    denominator_omitted_role_id: str
    denominator_omitted_comparator_kind: SelectiveDependenceResponseComparatorKind
    denominator_split_variable_id: str
    denominator_split_threshold: Decimal
    candidate_evaluation_panel_sizes: tuple[int, ...]
    power_familywise_alpha: Decimal
    target_power: Decimal
    maximum_nonevaluable_rate: Decimal
    maximum_adverse_rate: Decimal
    minimum_comparator_advantage: Decimal
    target_margin_id: str
    sink_margin_ids: tuple[str, ...]
    development_outcome_count: int
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "freeze_id",
            "target_id",
            "signature_id",
            "hold_action_id",
            "denominator_split_variable_id",
            "target_margin_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.scored_receiver_ids,
            field_name="scored_receiver_ids",
            allow_empty=False,
        )
        for name in ("primary_cell_ids", "support_boundary_cell_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        require_sorted_unique_strings(
            self.sink_margin_ids,
            field_name="sink_margin_ids",
            allow_empty=False,
        )
        if self.target_margin_id in self.sink_margin_ids:
            raise ValueError("target and noncompensating sink margins must remain distinct")
        for name in (
            "neutral_margin",
            "minimum_law_accuracy",
            "exchange_familywise_alpha",
            "denominator_split_threshold",
            "power_familywise_alpha",
            "target_power",
            "maximum_nonevaluable_rate",
            "maximum_adverse_rate",
            "minimum_comparator_advantage",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for name in (
            "minimum_law_accuracy",
            "exchange_familywise_alpha",
            "power_familywise_alpha",
            "target_power",
            "maximum_nonevaluable_rate",
            "maximum_adverse_rate",
        ):
            if getattr(self, name) >= Decimal(1):
                raise ValueError(f"{name} must be below one")
        if not Decimal(0) < self.exchange_familywise_alpha:
            raise ValueError("exchange familywise alpha must be positive")
        if not Decimal(0) < self.power_familywise_alpha:
            raise ValueError("power familywise alpha must be positive")
        if not Decimal(0) < self.target_power:
            raise ValueError("target power must be positive")
        if self.exchange_interval_method != "bonferroni-complete-unit-student-t":
            raise ValueError("exchange interval method is outside the implemented finite method")
        require_sorted_unique_ids(self.exchanges, attribute="exchange_id", field_name="exchanges")
        if any(value.target_id != self.target_id for value in self.exchanges):
            raise ValueError("analysis freeze exchanges cross targets")
        expectations = {value.expectation for value in self.exchanges}
        if expectations != {
            SelectiveDependenceResponseExchangeExpectation.ACTIVE,
            SelectiveDependenceResponseExchangeExpectation.INVARIANT,
            SelectiveDependenceResponseExchangeExpectation.SUPPORT_BOUNDARY,
        }:
            raise ValueError("analysis freeze needs active, invariant and support exchanges")
        parsed_primary = tuple(parse_scored_cell_id(value) for value in self.primary_cell_ids)
        if {value[4] for value in parsed_primary} != set(self.scored_receiver_ids):
            raise ValueError("primary cell and scored receiver rosters differ")
        primary = set(self.primary_cell_ids)
        if any(
            value.left_cell_id not in primary or value.right_cell_id not in primary
            for value in self.exchanges
        ):
            raise ValueError("an exchange endpoint lies outside the frozen primary cells")
        support_left = {
            scored_cell_to_condition_cell(value.left_cell_id)
            for value in self.exchanges
            if value.expectation is SelectiveDependenceResponseExchangeExpectation.SUPPORT_BOUNDARY
        }
        if not support_left.issubset(self.support_boundary_cell_ids):
            raise ValueError("support exchange endpoint lies outside the support roster")
        if self.denominator_omitted_role_id not in SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS:
            raise ValueError("denominator omission role is outside D/H/A/R/tau")
        expected_omission = {
            "D": SelectiveDependenceResponseComparatorKind.DENOMINATOR_BLIND,
            "H": SelectiveDependenceResponseComparatorKind.HISTORY_BLIND,
            "R": SelectiveDependenceResponseComparatorKind.RECEIVER_BLIND,
            "tau": SelectiveDependenceResponseComparatorKind.TAU_BLIND,
        }.get(self.denominator_omitted_role_id)
        if expected_omission is None:
            raise ValueError("no closed comparator implements this denominator omission")
        if self.denominator_omitted_comparator_kind is not expected_omission:
            raise ValueError("denominator omission and comparator kind differ")
        if (
            not self.candidate_evaluation_panel_sizes
            or tuple(sorted(set(self.candidate_evaluation_panel_sizes)))
            != self.candidate_evaluation_panel_sizes
            or any(value < 2 for value in self.candidate_evaluation_panel_sizes)
        ):
            raise ValueError("candidate evaluation panel sizes must be sorted unique counts")
        if self.development_outcome_count or self.evaluation_outcome_count:
            raise ValueError("target analysis freeze must precede all selective-response outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("target analysis freeze must remain outcome blind")

    @property
    def exchange_forecasts(self) -> tuple[SelectiveDependenceResponseExchangeForecast, ...]:
        return tuple(value.as_forecast() for value in self.exchanges)


def parse_scored_cell_id(cell_id: str) -> tuple[str, str, str, str, str]:
    """Return D, H, A, tau and R from one exact frozen scored-cell identity."""

    validate_stable_id(cell_id, field_name="cell_id")
    parts = cell_id.split(".")
    if len(parts) != 6 or parts[0] != "cell" or not parts[5].startswith("receiver-"):
        raise ValueError("scored cell identity does not encode D/H/A/tau/R")
    receiver_id = parts[5].removeprefix("receiver-")
    if not receiver_id:
        raise ValueError("scored cell identity has an empty receiver")
    return parts[1], parts[2], parts[3], parts[4], receiver_id


def scored_cell_to_condition_cell(cell_id: str) -> str:
    denominator_id, history_id, action_id, horizon_id, _receiver_id = parse_scored_cell_id(cell_id)
    return f"cell.{denominator_id}.{history_id}.{action_id}.{horizon_id}"


def expected_primary_cell_ids(
    *,
    denominator_ids: tuple[str, ...],
    history_ids: tuple[str, ...],
    action_ids: tuple[str, ...],
    horizon_ids: tuple[str, ...],
    receiver_ids: tuple[str, ...],
) -> tuple[str, ...]:
    """Construct the exact outcome-blind Cartesian scored-cell roster."""

    for name, values in (
        ("denominator_ids", denominator_ids),
        ("history_ids", history_ids),
        ("action_ids", action_ids),
        ("horizon_ids", horizon_ids),
        ("receiver_ids", receiver_ids),
    ):
        require_sorted_unique_strings(values, field_name=name, allow_empty=False)
    return tuple(
        sorted(
            f"cell.{denominator_id}.{history_id}.{action_id}.{horizon_id}.receiver-{receiver_id}"
            for denominator_id in denominator_ids
            for history_id in history_ids
            for action_id in action_ids
            for horizon_id in horizon_ids
            for receiver_id in receiver_ids
        )
    )


def expected_support_boundary_cell_ids(
    *,
    denominator_ids: tuple[str, ...],
    history_ids: tuple[str, ...],
    outside_action_ids: tuple[str, ...],
    horizon_ids: tuple[str, ...],
) -> tuple[str, ...]:
    for name, values in (
        ("denominator_ids", denominator_ids),
        ("history_ids", history_ids),
        ("outside_action_ids", outside_action_ids),
        ("horizon_ids", horizon_ids),
    ):
        require_sorted_unique_strings(values, field_name=name, allow_empty=False)
    return tuple(
        sorted(
            f"cell.{denominator_id}.{history_id}.{action_id}.{horizon_id}"
            for denominator_id in denominator_ids
            for history_id in history_ids
            for action_id in outside_action_ids
            for horizon_id in horizon_ids
        )
    )


def validate_target_analysis_freeze(
    freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
    *,
    design: CanonicalRecord,
    design_id: str,
    preparation: SelectiveDependenceResponsePreparationDistributionFreeze,
    denominator_ids: tuple[str, ...],
    history_ids: tuple[str, ...],
    action_ids: tuple[str, ...],
    horizon_ids: tuple[str, ...],
    receiver_ids: tuple[str, ...],
) -> None:
    """Close one analysis freeze against its exact target design and preparation."""

    if freeze.design != ObjectIdentity.from_record(design_id, design):
        raise ValueError("target analysis freeze design identity differs")
    if freeze.preparation_freeze != ObjectIdentity.from_record(preparation.freeze_id, preparation):
        raise ValueError("target analysis freeze preparation identity differs")
    target_id = getattr(design, "target_id", None)
    if target_id != freeze.target_id or preparation.target_id != freeze.target_id:
        raise ValueError("target analysis freeze crosses target identities")
    for name, values in (
        ("denominator_ids", denominator_ids),
        ("history_ids", history_ids),
        ("action_ids", action_ids),
        ("horizon_ids", horizon_ids),
        ("receiver_ids", receiver_ids),
    ):
        require_sorted_unique_strings(values, field_name=name, allow_empty=False)
    if freeze.hold_action_id not in action_ids:
        raise ValueError("analysis hold action lies outside the target action alphabet")
    if not set(freeze.scored_receiver_ids).issubset(receiver_ids):
        raise ValueError("analysis receiver lies outside the target observation alphabet")
    expected_primary = expected_primary_cell_ids(
        denominator_ids=denominator_ids,
        history_ids=history_ids,
        action_ids=action_ids,
        horizon_ids=horizon_ids,
        receiver_ids=freeze.scored_receiver_ids,
    )
    if freeze.primary_cell_ids != expected_primary:
        raise ValueError("analysis primary cells differ from the exact finite design")
    for exchange in freeze.exchanges:
        for cell_id in (exchange.left_cell_id, exchange.right_cell_id):
            denominator_id, history_id, action_id, horizon_id, receiver_id = parse_scored_cell_id(
                cell_id
            )
            if not all(
                (
                    denominator_id in denominator_ids,
                    history_id in history_ids,
                    action_id in action_ids,
                    horizon_id in horizon_ids,
                    receiver_id in freeze.scored_receiver_ids,
                )
            ):
                raise ValueError("analysis exchange cell lies outside the target design")
    outside_actions = tuple(
        sorted(
            {
                parse_scored_cell_id(value.left_cell_id)[2]
                for value in freeze.exchanges
                if value.reduction_kind is SelectiveDependenceResponseExchangeReductionKind.SUPPORT_REFUSAL
            }
        )
    )
    expected_support = expected_support_boundary_cell_ids(
        denominator_ids=denominator_ids,
        history_ids=history_ids,
        outside_action_ids=outside_actions,
        horizon_ids=horizon_ids,
    )
    if freeze.support_boundary_cell_ids != expected_support:
        raise ValueError("analysis support-boundary cells differ from the finite design")
    if max(freeze.candidate_evaluation_panel_sizes) > len(preparation.evaluation_unit_ids):
        raise ValueError("candidate evaluation size exceeds the frozen evaluation roster")


def _condition(
    unit: SelectiveDependenceResponseCompleteUnitResult,
    *,
    denominator_id: str,
    history_id: str,
    action_id: str,
    horizon_id: str,
) -> SelectiveDependenceResponseConditionResult | None:
    matches = tuple(
        value
        for value in unit.conditions
        if (
            value.denominator_id,
            value.history_id,
            value.action_id,
            value.horizon_id,
        )
        == (denominator_id, history_id, action_id, horizon_id)
    )
    return matches[0] if len(matches) == 1 else None


def _direct_receiver_contrast(
    unit: SelectiveDependenceResponseCompleteUnitResult,
    exchange: SelectiveDependenceResponseExchangeDesign,
) -> SelectiveDependenceResponseUnitContrast:
    left_roles = parse_scored_cell_id(exchange.left_cell_id)
    right_roles = parse_scored_cell_id(exchange.right_cell_id)
    left = _condition(
        unit,
        denominator_id=left_roles[0],
        history_id=left_roles[1],
        action_id=left_roles[2],
        horizon_id=left_roles[3],
    )
    right = _condition(
        unit,
        denominator_id=right_roles[0],
        history_id=right_roles[1],
        action_id=right_roles[2],
        horizon_id=right_roles[3],
    )

    def receiver_value(
        condition: SelectiveDependenceResponseConditionResult | None,
        receiver_id: str,
    ) -> Decimal | None:
        if condition is None or condition.stopped:
            return None
        matches = tuple(value for value in condition.receivers if value.receiver_id == receiver_id)
        if len(matches) != 1 or not matches[0].valid:
            return None
        return matches[0].value

    left_value = receiver_value(left, left_roles[4])
    right_value = receiver_value(right, right_roles[4])
    value = None if left_value is None or right_value is None else left_value - right_value
    return SelectiveDependenceResponseUnitContrast(
        contrast_id=f"contrast.{unit.complete_unit_id}.{exchange.estimand_id}",
        complete_unit_id=unit.complete_unit_id,
        estimand_id=exchange.estimand_id,
        value=value,
        native_unit=exchange.native_unit,
        evaluable=value is not None,
        reason_code=None if value is not None else "missing-or-stopped-exchange",
    )


def reduce_exchange_contrasts(
    panel: SelectiveDependenceResponseTargetPanel,
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
) -> dict[str, tuple[SelectiveDependenceResponseUnitContrast, ...]]:
    """Execute every continuous exchange using only frozen complete-unit operators."""

    if panel.target_id != analysis_freeze.target_id:
        raise ValueError("exchange reduction target differs from its analysis freeze")
    values: dict[str, tuple[SelectiveDependenceResponseUnitContrast, ...]] = {}
    for exchange in analysis_freeze.exchanges:
        if exchange.reduction_kind is SelectiveDependenceResponseExchangeReductionKind.SUPPORT_REFUSAL:
            continue
        if exchange.reduction_kind is SelectiveDependenceResponseExchangeReductionKind.DIRECT_RECEIVER_DIFFERENCE:
            contrasts = tuple(
                _direct_receiver_contrast(unit, exchange) for unit in panel.complete_units
            )
        else:
            left_roles = parse_scored_cell_id(exchange.left_cell_id)
            right_roles = parse_scored_cell_id(exchange.right_cell_id)
            left = tuple(
                matched_action_minus_hold(
                    unit,
                    estimand_id=f"{exchange.estimand_id}-left",
                    denominator_id=left_roles[0],
                    history_id=left_roles[1],
                    action_id=left_roles[2],
                    hold_action_id=analysis_freeze.hold_action_id,
                    horizon_id=left_roles[3],
                    receiver_id=left_roles[4],
                    native_unit=exchange.native_unit,
                )
                for unit in panel.complete_units
            )
            right = tuple(
                matched_action_minus_hold(
                    unit,
                    estimand_id=f"{exchange.estimand_id}-right",
                    denominator_id=right_roles[0],
                    history_id=right_roles[1],
                    action_id=right_roles[2],
                    hold_action_id=analysis_freeze.hold_action_id,
                    horizon_id=right_roles[3],
                    receiver_id=right_roles[4],
                    native_unit=exchange.native_unit,
                )
                for unit in panel.complete_units
            )
            contrasts = tuple(
                difference_of_contrasts(
                    left_value,
                    right_value,
                    estimand_id=exchange.estimand_id,
                )
                for left_value, right_value in zip(left, right, strict=True)
            )
        values[exchange.exchange_id] = contrasts
    return dict(sorted(values.items()))


def _support_boundary_refused(
    panel: SelectiveDependenceResponseTargetPanel,
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
) -> bool:
    expected = set(analysis_freeze.support_boundary_cell_ids)
    observed_per_unit: dict[str, set[str]] = {}
    refused = True
    for unit in panel.complete_units:
        observed = set()
        for condition in unit.conditions:
            cell_id = condition_cell_id(condition)
            if cell_id not in expected:
                continue
            observed.add(cell_id)
            hold = _condition(
                unit,
                denominator_id=condition.denominator_id,
                history_id=condition.history_id,
                action_id=analysis_freeze.hold_action_id,
                horizon_id=condition.horizon_id,
            )
            realized_as_hold = (
                hold is not None
                and condition.action.accepted_value == hold.action.accepted_value
                and condition.action.accepted_unit == hold.action.accepted_unit
                and condition.action.applied_value == hold.action.applied_value
                and condition.action.applied_unit == hold.action.applied_unit
                and condition.action.realized_value == hold.action.realized_value
                and condition.action.realized_unit == hold.action.realized_unit
                and condition.action.realized_clock == hold.action.realized_clock
            )
            refused &= all(
                (
                    condition.support_state == "outside-support",
                    condition.action.acceptance_state == "rejected-to-hold",
                    condition.fibre_admitted is False,
                    condition.disposition is SelectiveDependenceResponseDisposition.NONATTEMPT,
                    realized_as_hold,
                )
            )
        observed_per_unit[unit.complete_unit_id] = observed
    return (
        bool(panel.complete_units)
        and refused
        and all(observed == expected for observed in observed_per_unit.values())
    )


def reduce_selective_signature(
    panel: SelectiveDependenceResponseTargetPanel,
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
    *,
    outcome_access: OutcomeAccess,
) -> SelectiveDependenceResponseSelectiveDependenceSignature:
    """Adjudicate the predeclared exchange family on complete units."""

    contrast_families = reduce_exchange_contrasts(panel, analysis_freeze)
    continuous = tuple(
        value
        for value in analysis_freeze.exchanges
        if value.reduction_kind is not SelectiveDependenceResponseExchangeReductionKind.SUPPORT_REFUSAL
    )
    if set(contrast_families) != {value.exchange_id for value in continuous}:
        raise ValueError("continuous exchange reduction roster differs from the freeze")
    confidence_level = bonferroni_confidence_level(
        familywise_alpha=analysis_freeze.exchange_familywise_alpha,
        family_size=len(continuous),
    )
    nested_count = len(panel.complete_units[0].conditions) if panel.complete_units else 0
    estimates = tuple(
        sorted(
            (
                aggregate_complete_units(
                    estimate_id=f"estimate.{exchange.exchange_id}",
                    contrasts=contrast_families[exchange.exchange_id],
                    issued_complete_unit_ids=panel.expected_complete_unit_ids,
                    nested_condition_count=nested_count,
                    outcome_access=outcome_access,
                    confidence_level=confidence_level,
                )
                for exchange in continuous
            ),
            key=lambda value: value.estimate_id,
        )
    )
    by_exchange = {value.estimate_id.removeprefix("estimate."): value for value in estimates}
    assessments = []
    for exchange in continuous:
        estimate = by_exchange[exchange.exchange_id]
        if estimate.point is None or estimate.lower is None or estimate.upper is None:
            point = lower = upper = Decimal(0)
            count = 0
        else:
            point, lower, upper = estimate.point, estimate.lower, estimate.upper
            count = estimate.evaluable_complete_unit_count
        assessments.append(
            assess_exchange(
                assessment_id=f"assessment.{exchange.exchange_id}",
                exchange_id=exchange.exchange_id,
                expectation=exchange.expectation,
                point=point,
                lower=lower,
                upper=upper,
                equivalence_margin=exchange.equivalence_margin,
                minimum_active_difference=exchange.minimum_active_difference,
                complete_unit_count=count,
            )
        )
    ordered_assessments = tuple(sorted(assessments, key=lambda value: value.assessment_id))
    active_ids = tuple(
        value.exchange_id
        for value in continuous
        if value.expectation is SelectiveDependenceResponseExchangeExpectation.ACTIVE
    )
    invariant_ids = tuple(
        value.exchange_id
        for value in continuous
        if value.expectation is SelectiveDependenceResponseExchangeExpectation.INVARIANT
    )
    return SelectiveDependenceResponseSelectiveDependenceSignature(
        signature_id=analysis_freeze.signature_id,
        target_id=panel.target_id,
        estimates=estimates,
        exchange_assessments=ordered_assessments,
        active_exchange_ids=active_ids,
        invariant_exchange_ids=invariant_ids,
        support_boundary_case_ids=analysis_freeze.support_boundary_cell_ids,
        complete_unit_ids_sha256=digest_ids(panel.expected_complete_unit_ids),
        all_active_supported=all(
            value.state is SelectiveDependenceResponseExchangeState.SUPPORTED
            for value in ordered_assessments
            if value.exchange_id in active_ids
        ),
        all_invariant_supported=all(
            value.state is SelectiveDependenceResponseExchangeState.SUPPORTED
            for value in ordered_assessments
            if value.exchange_id in invariant_ids
        ),
        support_boundary_refused=_support_boundary_refused(panel, analysis_freeze),
        outcome_access=outcome_access,
    )


def analyze_development_panel(
    panel: SelectiveDependenceResponseTargetPanel,
    analysis_freeze: SelectiveDependenceResponseTargetAnalysisFreeze,
) -> tuple[SelectiveDependenceResponseFiniteLawCalibration, SelectiveDependenceResponseSelectiveDependenceSignature]:
    """Fit and reduce one development panel without outcome-derived inclusion."""

    if panel.target_id != analysis_freeze.target_id:
        raise ValueError("development panel target differs from its analysis freeze")
    signature = reduce_selective_signature(
        panel,
        analysis_freeze,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    law = fit_categorical_finite_law(
        panel,
        hold_action_id=analysis_freeze.hold_action_id,
        receiver_ids=analysis_freeze.scored_receiver_ids,
        neutral_margin=analysis_freeze.neutral_margin,
        primary_cell_ids=analysis_freeze.primary_cell_ids,
        minimum_required_accuracy=analysis_freeze.minimum_law_accuracy,
    )
    return law, signature


__all__ = [
    'SelectiveDependenceResponseExchangeDesign',
    'SelectiveDependenceResponseExchangeReductionKind',
    'SelectiveDependenceResponseTargetAnalysisFreeze',
    "analyze_development_panel",
    "expected_primary_cell_ids",
    "expected_support_boundary_cell_ids",
    "parse_scored_cell_id",
    "reduce_exchange_contrasts",
    "reduce_selective_signature",
    "scored_cell_to_condition_cell",
    "validate_target_analysis_freeze",
]

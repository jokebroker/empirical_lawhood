'Margin-bearing effective-law forecasts for the margin structural recurrence forecast follow-up.\n\nmargin structural recurrence forecast is additive to the completed action-fiber action-fiber repair.  It preserves the\naction-fiber ten-role intersection and attaches a signed, uncertainty-adjusted margin to\nevery gate.  A deterministic joint-unit bootstrap then makes a fallible\nprediction about whether each native action fiber will remain admitted on a\nfresh panel of predeclared size.\n'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from random import Random
from typing import ClassVar

from empirical_lawhood.adapters.methods import structural_recurrence as core
from empirical_lawhood.adapters.methods.structural_bootstrap_inputs import StructuralBootstrapInputCensus, StructuralBootstrapSeedInput, require_structural_bootstrap_census, require_structural_bootstrap_seed_input, structural_bootstrap_context_sha256
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceActionOutcome, StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetDesignFreeze, StructuralRecurrenceTargetStage, _all_outcomes, _median
from empirical_lawhood.adapters.methods.action_fiber_structural_recurrence import ActionFiberSignature, ActionFiberStructuralRecurrenceConformance, ActionFiberStructuralRecurrenceMethodFreeze, ActionFiberStructuralRecurrenceAdmissionHandoff, ActionFiberStructuralRecurrencePredictionIssue, ActionFiberStructuralRecurrenceTargetResult, PolicySafetySignature, PolicyValidationDisposition, build_policy_signature, evaluate_admission, finalize_target, wilson_lower_bound
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


_QUANTUM = Decimal("0.000000000001")
MARGIN_COMPONENT_IDS = (
    "action-realization",
    "authority",
    "dynamics",
    "effort",
    "preservation",
    "reachability",
    "receiver-sink",
    "receiver-target",
    "support",
    "uncertainty",
    "validity",
)


class MarginKind(StrEnum):
    BINARY_WILSON = "BINARY_WILSON"
    TARGET_MEDIAN = "TARGET_MEDIAN"


class MarginBand(StrEnum):
    ROBUST_INTERIOR = "ROBUST_INTERIOR"
    FRAGILE_INTERIOR = "FRAGILE_INTERIOR"
    EXTERIOR = "EXTERIOR"
    UNEVALUABLE = "UNEVALUABLE"


class MarginStructuralRecurrenceForecastTerminalVerdict(StrEnum):
    METHOD_NOT_QUALIFIED = "MARGIN_STRUCTURAL_RECURRENCE_FORECAST_METHOD_NOT_QUALIFIED"
    SAFETY_TYPING_OPPOSED = "MARGIN_STRUCTURAL_RECURRENCE_FORECAST_SAFETY_TYPING_OPPOSED"
    MARGIN_FORECAST_OPPOSED = "MARGIN_STRUCTURAL_RECURRENCE_FORECAST_MARGIN_FORECAST_OPPOSED"
    SUBSTRATE_LOCAL = "MARGIN_STRUCTURAL_RECURRENCE_FORECAST_SUBSTRATE_LOCAL_MARGIN_RECURRENCE"
    BOUNDED_TWO_TARGET = "MARGIN_STRUCTURAL_RECURRENCE_FORECAST_BOUNDED_TWO_TARGET_MARGIN_RECURRENCE"
    BOUNDED_THREE_TARGET = "MARGIN_STRUCTURAL_RECURRENCE_FORECAST_BOUNDED_THREE_TARGET_MARGIN_RECURRENCE"
    BOUNDED_FOUR_TARGET = "MARGIN_STRUCTURAL_RECURRENCE_FORECAST_BOUNDED_FOUR_TARGET_MARGIN_RECURRENCE"


@dataclass(frozen=True, slots=True)
class GateMarginFact(CanonicalRecord):
    """Signed distance from one native, noncompensating admission gate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/gate-margin-fact'

    fact_id: str
    component_id: str
    kind: MarginKind
    observed_value: Decimal
    threshold: Decimal
    normalization_scale: Decimal
    signed_margin: Decimal
    normalized_margin: Decimal
    passed: bool
    success_count: int | None
    independent_unit_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.fact_id, field_name="fact_id")
        validate_stable_id(self.component_id, field_name="component_id")
        if self.component_id not in MARGIN_COMPONENT_IDS:
            raise ValueError('margin component is outside the frozen margin-forecast gate roster')
        for name in (
            "observed_value",
            "threshold",
            "normalization_scale",
            "signed_margin",
            "normalized_margin",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        if self.normalization_scale <= 0:
            raise ValueError("margin normalization scale must be positive")
        expected_margin = (self.observed_value - self.threshold).quantize(_QUANTUM)
        expected_normalized = (expected_margin / self.normalization_scale).quantize(_QUANTUM)
        if self.signed_margin != expected_margin or self.normalized_margin != expected_normalized:
            raise ValueError("gate margin is not mechanically derived")
        if self.passed != (self.signed_margin >= 0):
            raise ValueError("gate pass differs from the signed margin")
        if self.independent_unit_count < 1:
            raise ValueError("gate margin requires independent units")
        if self.kind is MarginKind.BINARY_WILSON:
            if (
                self.success_count is None
                or self.success_count < 0
                or self.success_count > self.independent_unit_count
            ):
                raise ValueError("binary margin requires a valid success count")
        elif self.success_count is not None:
            raise ValueError("target-median margin cannot carry a binary count")


@dataclass(frozen=True, slots=True)
class ActionMarginSignature(CanonicalRecord):
    'The complete noncompensating margin vector for one action-fiber action fiber.'

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/action-margin-signature'

    signature_id: str
    action_fiber: ActionFiberSignature
    gate_margins: tuple[GateMarginFact, ...]
    binary_robustness_margin_min: Decimal
    target_robustness_ratio_min: Decimal
    band: MarginBand
    critical_component_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.signature_id, field_name="signature_id")
        validate_stable_id(self.critical_component_id, field_name="critical_component_id")
        validate_decimal(
            self.binary_robustness_margin_min,
            field_name="binary_robustness_margin_min",
        )
        validate_decimal(
            self.target_robustness_ratio_min,
            field_name="target_robustness_ratio_min",
        )
        require_sorted_unique_ids(
            self.gate_margins,
            attribute="fact_id",
            field_name="gate_margins",
        )
        if tuple(sorted(value.component_id for value in self.gate_margins)) != MARGIN_COMPONENT_IDS:
            raise ValueError("action margin requires the exact eleven-component roster")
        by_component = {value.component_id: value for value in self.gate_margins}
        if by_component["action-realization"].passed != (
            self.action_fiber.realization_role is core.ActionRole.REALIZATION_QUALIFIED
        ):
            raise ValueError("realization margin differs from the action fiber")
        for operand in self.action_fiber.operands:
            component = operand.role.value.lower().replace("_", "-")
            if by_component[component].passed != (operand.status is core.OperandStatus.PASS):
                raise ValueError('gate margin differs from the action-fiber operand')
        expected_critical = min(
            self.gate_margins,
            key=lambda value: (value.normalized_margin, value.component_id),
        ).component_id
        if self.critical_component_id != expected_critical:
            raise ValueError("critical margin component is not the weakest normalized gate")
        expected_band = classify_margin_band(
            admitted=self.action_fiber.admitted,
            facts=self.gate_margins,
            binary_robustness_margin_min=self.binary_robustness_margin_min,
            target_robustness_ratio_min=self.target_robustness_ratio_min,
        )
        if self.band is not expected_band:
            raise ValueError("margin band is not noncompensating and gate-derived")


@dataclass(frozen=True, slots=True)
class MarginPolicySignature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-policy-signature'

    signature_id: str
    base_policy: PolicySafetySignature
    action_margins: tuple[ActionMarginSignature, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.signature_id, field_name="signature_id")
        require_sorted_unique_ids(
            self.action_margins,
            attribute="signature_id",
            field_name="action_margins",
        )
        expected = {
            *(value.action_id for value in self.base_policy.action_fibers),
            self.base_policy.hold_viability.hold_fiber.action_id,
        }
        if {value.action_fiber.action_id for value in self.action_margins} != expected:
            raise ValueError("margin policy does not cover the complete native action chart")
        if any(
            value.action_fiber.stage is not self.base_policy.stage
            or value.action_fiber.target_slot_id != self.base_policy.target_slot_id
            for value in self.action_margins
        ):
            raise ValueError("margin policy crosses target or stage boundaries")

    def for_action(self, action_id: str) -> ActionMarginSignature:
        return next(
            value for value in self.action_margins if value.action_fiber.action_id == action_id
        )


@dataclass(frozen=True, slots=True)
class PanelAdmissionForecast(CanonicalRecord):
    """Joint-unit bootstrap forecast for one fiber on one future panel."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/panel-admission-forecast'

    forecast_id: str
    action_id: str
    future_stage: StructuralRecurrenceTargetStage
    future_independent_unit_count: int
    bootstrap_replications: int
    deterministic_seed_sha256: str
    admission_probability: Decimal
    predicted_admitted: bool
    binary_only_comparator_probability: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.forecast_id, field_name="forecast_id")
        validate_stable_id(self.action_id, field_name="action_id")
        validate_sha256(self.deterministic_seed_sha256, field_name="deterministic_seed_sha256")
        for name in ("admission_probability", "binary_only_comparator_probability"):
            validate_decimal(getattr(self, name), field_name=name)
            if getattr(self, name) < 0 or getattr(self, name) > 1:
                raise ValueError("forecast probabilities must lie in [0, 1]")
        if self.future_stage not in {StructuralRecurrenceTargetStage.EVALUATION, StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION}:
            raise ValueError('margin forecast must target evaluation or prospective validation')
        if self.future_independent_unit_count < 1 or self.bootstrap_replications < 512:
            raise ValueError("margin forecast has an inadequate future panel")
        if self.predicted_admitted != (self.admission_probability >= Decimal("0.5")):
            raise ValueError("binary forecast is not probability-derived")


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastMethodFreeze(ActionFiberStructuralRecurrenceMethodFreeze):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-method-freeze'

    action_fiber_terminal_adjudication: ObjectIdentity
    action_fiber_closeout: ObjectIdentity
    conformance_source: ObjectIdentity
    custody_source: ObjectIdentity
    new_substrate_source: ObjectIdentity
    margin_fixture_count: int
    margin_control_count: int
    bootstrap_replications: int
    binary_robustness_margin_min: Decimal
    target_robustness_ratio_min: Decimal
    new_substrate_class_count: int

    def __post_init__(self) -> None:
        ActionFiberStructuralRecurrenceMethodFreeze.__post_init__(self)
        validate_decimal(
            self.binary_robustness_margin_min,
            field_name="binary_robustness_margin_min",
        )
        validate_decimal(
            self.target_robustness_ratio_min,
            field_name="target_robustness_ratio_min",
        )
        if (
            self.margin_fixture_count != 12
            or self.margin_control_count != 10
            or self.bootstrap_replications < 512
            or self.binary_robustness_margin_min != Decimal("0.10")
            or self.target_robustness_ratio_min != Decimal("0.50")
            or self.new_substrate_class_count != 1
        ):
            raise ValueError('margin-forecast method freeze differs from the margin-bearing design')


@dataclass(frozen=True, slots=True)
class MarginFixtureResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-fixture-result'

    fixture_id: str
    fixture_kind: str
    expected_value: str
    observed_value: str
    passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.fixture_id, field_name="fixture_id")
        validate_stable_id(self.fixture_kind, field_name="fixture_kind")
        if self.passed != (self.expected_value == self.observed_value):
            raise ValueError("margin fixture pass is not mechanically derived")


@dataclass(frozen=True, slots=True)
class MarginControlResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-control-result'

    control_id: str
    control_kind: str
    failure_code: str
    passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.control_id, field_name="control_id")
        validate_stable_id(self.control_kind, field_name="control_kind")
        validate_stable_id(self.failure_code, field_name="failure_code")
        if not self.passed:
            raise ValueError('failed margin control cannot qualify margin-forecast')


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastConformance(ActionFiberStructuralRecurrenceConformance):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-conformance'

    margin_fixtures: tuple[MarginFixtureResult, ...]
    margin_controls: tuple[MarginControlResult, ...]
    margin_fixture_pass_count: int
    margin_control_pass_count: int

    def __post_init__(self) -> None:
        ActionFiberStructuralRecurrenceConformance.__post_init__(self)
        require_sorted_unique_ids(
            self.margin_fixtures,
            attribute="fixture_id",
            field_name="margin_fixtures",
        )
        require_sorted_unique_ids(
            self.margin_controls,
            attribute="control_id",
            field_name="margin_controls",
        )
        if (
            len(self.margin_fixtures) != self.margin_fixture_pass_count
            or self.margin_fixture_pass_count != 12
            or not all(value.passed for value in self.margin_fixtures)
            or len(self.margin_controls) != self.margin_control_pass_count
            or self.margin_control_pass_count != 10
            or not all(value.passed for value in self.margin_controls)
        ):
            raise ValueError('margin-forecast margin qualification panel is incomplete')


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastPredictionIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-prediction-issue'

    issue_id: str
    method_freeze: ObjectIdentity
    conformance: ObjectIdentity
    structural_prediction: ActionFiberStructuralRecurrencePredictionIssue
    development_margins: MarginPolicySignature
    forecasts: tuple[PanelAdmissionForecast, ...]
    predicted_evaluation_policy_branch: core.PolicyBranch
    predicted_evaluation_action_id: str
    predicted_prospective_validation_disposition: core.ProspectiveDisposition
    predicted_validation_disposition: PolicyValidationDisposition
    published_before_evaluation: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        validate_stable_id(
            self.predicted_evaluation_action_id,
            field_name="predicted_evaluation_action_id",
        )
        require_sorted_unique_ids(self.forecasts, attribute="forecast_id", field_name="forecasts")
        action_count = len(self.development_margins.action_margins)
        if len(self.forecasts) != 2 * action_count:
            raise ValueError('prediction requires evaluation and prospective validation forecasts for every action')
        evaluation = tuple(
            value for value in self.forecasts if value.future_stage is StructuralRecurrenceTargetStage.EVALUATION
        )
        branch, action_id = forecast_policy_decision(
            self.development_margins.base_policy,
            evaluation,
        )
        if (
            self.predicted_evaluation_policy_branch is not branch
            or self.predicted_evaluation_action_id != action_id
        ):
            raise ValueError("evaluation policy is not margin-forecast derived")
        prospective_validation, validation = forecast_validation_disposition(
            branch=branch,
            action_id=action_id,
            forecasts=tuple(
                value for value in self.forecasts if value.future_stage is StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION
            ),
        )
        if (
            self.predicted_prospective_validation_disposition is not prospective_validation
            or self.predicted_validation_disposition is not validation
        ):
            raise ValueError("validation prediction is not margin-forecast derived")
        if not self.published_before_evaluation or self.protected_outcome_access_count:
            raise ValueError('margin-forecast prediction crossed the protected-outcome boundary')


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastAdmissionHandoff(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-admission-handoff'

    handoff_id: str
    prediction_issue: ObjectIdentity
    action_fiber_admission: ActionFiberStructuralRecurrenceAdmissionHandoff
    evaluation_margins: MarginPolicySignature
    reveal_authority: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.handoff_id, field_name="handoff_id")
        if self.evaluation_margins.base_policy != self.action_fiber_admission.policy_safety:
            raise ValueError('margin-forecast admission margins differ from the revealed base policy')


@dataclass(frozen=True, slots=True)
class ForecastMatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/forecast-match'

    match_id: str
    action_id: str
    predicted_probability: Decimal
    predicted_admitted: bool
    observed_admitted: bool
    brier_score: Decimal
    binary_only_comparator_probability: Decimal
    binary_only_comparator_brier: Decimal
    margin_band_predicted: MarginBand
    margin_band_observed: MarginBand
    admission_exact: bool
    band_exact: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.match_id, field_name="match_id")
        validate_stable_id(self.action_id, field_name="action_id")
        for name in (
            "predicted_probability",
            "brier_score",
            "binary_only_comparator_probability",
            "binary_only_comparator_brier",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        observed = Decimal(1 if self.observed_admitted else 0)
        expected_brier = ((self.predicted_probability - observed) ** 2).quantize(_QUANTUM)
        expected_comparator = ((self.binary_only_comparator_probability - observed) ** 2).quantize(
            _QUANTUM
        )
        if (
            self.brier_score != expected_brier
            or self.binary_only_comparator_brier != expected_comparator
        ):
            raise ValueError("forecast scores are not outcome-derived")
        if self.admission_exact != (self.predicted_admitted == self.observed_admitted):
            raise ValueError("forecast exactness is inconsistent")
        if self.band_exact != (self.margin_band_predicted is self.margin_band_observed):
            raise ValueError("margin-band exactness is inconsistent")


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastTargetMatch(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-target-match'

    match_id: str
    prediction_issue: ObjectIdentity
    target_result: ObjectIdentity
    target_slot_id: str
    forecast_matches: tuple[ForecastMatch, ...]
    evaluation_policy_exact: bool
    selected_action_exact: bool
    validation_exact: bool
    safety_error_count: int
    exact_forecast_count: int
    exact_band_count: int
    margin_model_comparator_win_count: int
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.match_id, field_name="match_id")
        validate_stable_id(self.target_slot_id, field_name="target_slot_id")
        require_sorted_unique_ids(
            self.forecast_matches,
            attribute="match_id",
            field_name="forecast_matches",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.safety_error_count < 0:
            raise ValueError("safety errors must be nonnegative")
        if self.exact_forecast_count != sum(
            value.admission_exact for value in self.forecast_matches
        ):
            raise ValueError("exact forecast count is not match-derived")
        if self.exact_band_count != sum(value.band_exact for value in self.forecast_matches):
            raise ValueError("exact band count is not match-derived")
        if self.margin_model_comparator_win_count != sum(
            value.brier_score < value.binary_only_comparator_brier
            for value in self.forecast_matches
        ):
            raise ValueError("comparator wins are not score-derived")
        expected = all(
            (
                self.exact_forecast_count == len(self.forecast_matches),
                self.evaluation_policy_exact,
                self.selected_action_exact,
                self.validation_exact,
                self.safety_error_count == 0,
            )
        )
        if self.passed != expected:
            raise ValueError('margin-forecast target pass is not noncompensating')


@dataclass(frozen=True, slots=True)
class MarginStructuralRecurrenceForecastAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/margin-structural-recurrence-forecast-adjudication'

    adjudication_id: str
    method_conformance: ObjectIdentity
    target_matches: tuple[ObjectIdentity, ...]
    eligible_target_count: int
    passed_target_count: int
    failed_target_count: int
    safety_error_count: int
    exact_forecast_count: int
    total_forecast_count: int
    exact_band_count: int
    margin_model_comparator_win_count: int
    verdict: MarginStructuralRecurrenceForecastTerminalVerdict
    positive_claim_eligible: bool
    new_substrate_class_count: int
    independent_generality_claim_eligible: bool
    no_cross_target_pooling: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(
            self.target_matches, attribute="object_id", field_name="target_matches"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.eligible_target_count != len(self.target_matches)
            or self.passed_target_count + self.failed_target_count != self.eligible_target_count
            or self.total_forecast_count < self.exact_forecast_count
            or self.total_forecast_count < self.exact_band_count
            or self.new_substrate_class_count != 1
            or self.independent_generality_claim_eligible
            or not self.no_cross_target_pooling
        ):
            raise ValueError('margin-forecast adjudication counts or claim boundary are inconsistent')
        expected_positive = (
            self.passed_target_count == self.eligible_target_count == 4
            and not self.safety_error_count
        )
        if self.positive_claim_eligible != expected_positive:
            raise ValueError('margin-forecast positive eligibility is not target/safety derived')


def classify_margin_band(
    *,
    admitted: bool,
    facts: tuple[GateMarginFact, ...],
    binary_robustness_margin_min: Decimal,
    target_robustness_ratio_min: Decimal,
) -> MarginBand:
    if not facts:
        return MarginBand.UNEVALUABLE
    if not admitted or any(not value.passed for value in facts):
        return MarginBand.EXTERIOR
    binary_robust = all(
        value.signed_margin >= binary_robustness_margin_min
        for value in facts
        if value.kind is MarginKind.BINARY_WILSON
    )
    target = next(value for value in facts if value.kind is MarginKind.TARGET_MEDIAN)
    return (
        MarginBand.ROBUST_INTERIOR
        if binary_robust and target.normalized_margin >= target_robustness_ratio_min
        else MarginBand.FRAGILE_INTERIOR
    )


def _binary_value(
    outcome: StructuralRecurrenceActionOutcome, component_id: str, design: StructuralRecurrenceTargetDesignFreeze
) -> bool:
    if component_id == "action-realization":
        return outcome.realization_error <= design.threshold("realization-error-max")
    if component_id == "support":
        return outcome.support_preserved
    if component_id == "receiver-sink":
        return outcome.sink_margin >= design.threshold("sink-margin-min")
    if component_id == "effort":
        return outcome.effort <= design.threshold("effort-max")
    if component_id == "validity":
        return outcome.validity_passed
    if component_id == "uncertainty":
        return outcome.uncertainty_evaluable
    if component_id == "preservation":
        return outcome.preservation_passed
    if component_id == "dynamics":
        return outcome.dynamics_passed
    if component_id == "reachability":
        return outcome.reachability_passed
    if component_id == "authority":
        return outcome.authority_passed
    raise ValueError(f"component is not binary: {component_id}")


def _target_pass(
    outcomes: tuple[StructuralRecurrenceActionOutcome, ...],
    *,
    action_id: str,
    design: StructuralRecurrenceTargetDesignFreeze,
) -> bool:
    median = _median(tuple(value.target_effect for value in outcomes))
    threshold = design.threshold("target-effect-min")
    return median >= (-threshold if action_id == "hold" else threshold)


def _panel_admitted(
    outcomes: tuple[StructuralRecurrenceActionOutcome, ...],
    *,
    action_id: str,
    design: StructuralRecurrenceTargetDesignFreeze,
) -> bool:
    floor = design.threshold("unit-pass-rate-min")
    binary_ids = tuple(value for value in MARGIN_COMPONENT_IDS if value != "receiver-target")
    return _target_pass(outcomes, action_id=action_id, design=design) and all(
        wilson_lower_bound(
            sum(_binary_value(value, component_id, design) for value in outcomes),
            len(outcomes),
        )
        >= floor
        for component_id in binary_ids
    )


def build_action_margin(
    *,
    design: StructuralRecurrenceTargetDesignFreeze,
    evidence: StructuralRecurrenceStageEvidence,
    fiber: ActionFiberSignature,
    binary_robustness_margin_min: Decimal,
    target_robustness_ratio_min: Decimal,
) -> ActionMarginSignature:
    outcomes = _all_outcomes(evidence, fiber.action_id)
    floor = design.threshold("unit-pass-rate-min")
    target_threshold_native = design.threshold("target-effect-min")
    facts: list[GateMarginFact] = []
    for component_id in MARGIN_COMPONENT_IDS:
        if component_id == "receiver-target":
            observed = _median(tuple(value.target_effect for value in outcomes))
            threshold = (
                -target_threshold_native if fiber.action_id == "hold" else target_threshold_native
            )
            kind = MarginKind.TARGET_MEDIAN
            successes = None
            scale = target_threshold_native
        else:
            successes = sum(_binary_value(value, component_id, design) for value in outcomes)
            observed = wilson_lower_bound(successes, len(outcomes))
            threshold = floor
            kind = MarginKind.BINARY_WILSON
            scale = Decimal(1) - floor
        signed = (observed - threshold).quantize(_QUANTUM)
        facts.append(
            GateMarginFact(
                fact_id=(
                    f"{fiber.target_slot_id}.{fiber.stage.value.lower()}."
                    f"{fiber.action_id}.{component_id}.gate-margin"
                ),
                component_id=component_id,
                kind=kind,
                observed_value=observed,
                threshold=threshold,
                normalization_scale=scale,
                signed_margin=signed,
                normalized_margin=(signed / scale).quantize(_QUANTUM),
                passed=signed >= 0,
                success_count=successes,
                independent_unit_count=len(outcomes),
            )
        )
    ordered = tuple(sorted(facts, key=lambda value: value.fact_id))
    critical = min(
        ordered,
        key=lambda value: (value.normalized_margin, value.component_id),
    ).component_id
    return ActionMarginSignature(
        signature_id=(
            f"{fiber.target_slot_id}.{fiber.stage.value.lower()}."
            f"{fiber.action_id}.gate-margin-signature"
        ),
        action_fiber=fiber,
        gate_margins=ordered,
        binary_robustness_margin_min=binary_robustness_margin_min,
        target_robustness_ratio_min=target_robustness_ratio_min,
        band=classify_margin_band(
            admitted=fiber.admitted,
            facts=ordered,
            binary_robustness_margin_min=binary_robustness_margin_min,
            target_robustness_ratio_min=target_robustness_ratio_min,
        ),
        critical_component_id=critical,
    )


def build_margin_policy(
    *,
    design: StructuralRecurrenceTargetDesignFreeze,
    evidence: StructuralRecurrenceStageEvidence,
    base_policy: PolicySafetySignature,
    binary_robustness_margin_min: Decimal,
    target_robustness_ratio_min: Decimal,
) -> MarginPolicySignature:
    fibers = (*base_policy.action_fibers, base_policy.hold_viability.hold_fiber)
    margins = tuple(
        sorted(
            (
                build_action_margin(
                    design=design,
                    evidence=evidence,
                    fiber=fiber,
                    binary_robustness_margin_min=binary_robustness_margin_min,
                    target_robustness_ratio_min=target_robustness_ratio_min,
                )
                for fiber in fibers
            ),
            key=lambda value: value.signature_id,
        )
    )
    return MarginPolicySignature(
        signature_id=(
            f"{design.target_slot.target_slot_id}.{evidence.stage.value.lower()}.margin-policy"
        ),
        base_policy=base_policy,
        action_margins=margins,
    )


def margin_bootstrap_context_sha256(
    *, design: StructuralRecurrenceTargetDesignFreeze,
    development: StructuralRecurrenceStageEvidence, action_id: str,
    future_stage: StructuralRecurrenceTargetStage,
    future_independent_unit_count: int, bootstrap_replications: int,
) -> str:
    return structural_bootstrap_context_sha256(
        ObjectIdentity.from_record(design.design_id, design),
        ObjectIdentity.from_record(development.evidence_id, development),
        action_id, future_stage.value, future_independent_unit_count,
        bootstrap_replications,
    )


def require_margin_bootstrap_inputs(
    *, scientific_bootstrap_inputs: StructuralBootstrapInputCensus | None,
    design: StructuralRecurrenceTargetDesignFreeze,
    development: StructuralRecurrenceStageEvidence,
    evaluation_count: int, prospective_validation_count: int, bootstrap_replications: int,
) -> dict[tuple[str, StructuralRecurrenceTargetStage], StructuralBootstrapSeedInput]:
    census = require_structural_bootstrap_census(scientific_bootstrap_inputs)
    expected = tuple((action_id, stage) for action_id in design.native_action_ids
                     for stage in (StructuralRecurrenceTargetStage.EVALUATION, StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION))
    if tuple((row.action_id, row.stage) for row in census.inputs) != tuple((action, stage.value) for action, stage in expected):
        raise ValueError("margin prediction requires exactly the complete native action and future-stage bootstrap census")
    result = {}
    original_custody = set()
    for row, (action_id, stage) in zip(census.inputs, expected, strict=True):
        count = evaluation_count if stage is StructuralRecurrenceTargetStage.EVALUATION else prospective_validation_count
        require_structural_bootstrap_seed_input(
            row, scientific_role="margin-panel-admission",
            current_target_id=design.target_slot.target_slot_id, action_id=action_id,
            stage=stage.value, sample_count=count,
            bootstrap_replications=bootstrap_replications,
            current_context_sha256=margin_bootstrap_context_sha256(
                design=design, development=development, action_id=action_id,
                future_stage=stage, future_independent_unit_count=count,
                bootstrap_replications=bootstrap_replications,
            ),
        )
        original_custody.add((row.source_original_context_sha256, row.original_source, row.export_receipt))
        result[(action_id, stage)] = row
    if len(original_custody) != 1:
        raise ValueError("margin census crosses its original development context or authenticated export custody")
    return result


def forecast_panel_admission(
    *,
    design: StructuralRecurrenceTargetDesignFreeze,
    development: StructuralRecurrenceStageEvidence,
    development_margin: ActionMarginSignature,
    future_stage: StructuralRecurrenceTargetStage,
    future_independent_unit_count: int,
    bootstrap_replications: int,
    scientific_seed_input: StructuralBootstrapSeedInput,
) -> PanelAdmissionForecast:
    action_id = development_margin.action_fiber.action_id
    numerical_input = require_structural_bootstrap_seed_input(
        scientific_seed_input, scientific_role="margin-panel-admission",
        current_target_id=design.target_slot.target_slot_id, action_id=action_id,
        stage=future_stage.value, sample_count=future_independent_unit_count,
        bootstrap_replications=bootstrap_replications,
        current_context_sha256=margin_bootstrap_context_sha256(
            design=design, development=development, action_id=action_id,
            future_stage=future_stage,
            future_independent_unit_count=future_independent_unit_count,
            bootstrap_replications=bootstrap_replications,
        ),
    )
    if future_stage not in (StructuralRecurrenceTargetStage.EVALUATION, StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION) or future_independent_unit_count < 1 or bootstrap_replications < 512:
        raise ValueError("margin bootstrap numerical panel is outside the frozen scientific contract")
    outcomes = _all_outcomes(development, action_id)
    seed_digest = numerical_input.full_seed_sha256
    rng = Random(int(seed_digest[:16], 16))
    admitted_count = 0
    for _ in range(bootstrap_replications):
        panel = tuple(
            outcomes[rng.randrange(len(outcomes))] for _ in range(future_independent_unit_count)
        )
        admitted_count += _panel_admitted(panel, action_id=action_id, design=design)
    probability = (Decimal(admitted_count) / Decimal(bootstrap_replications)).quantize(
        Decimal("0.000001")
    )
    return PanelAdmissionForecast(
        forecast_id=(
            f"{design.target_slot.target_slot_id}.{action_id}."
            f"{future_stage.value.lower()}.margin-admission-forecast"
        ),
        action_id=action_id,
        future_stage=future_stage,
        future_independent_unit_count=future_independent_unit_count,
        bootstrap_replications=bootstrap_replications,
        deterministic_seed_sha256=seed_digest,
        admission_probability=probability,
        predicted_admitted=probability >= Decimal("0.5"),
        binary_only_comparator_probability=(
            Decimal("0.95") if development_margin.action_fiber.admitted else Decimal("0.05")
        ),
    )


def forecast_policy_decision(
    development_policy: PolicySafetySignature,
    evaluation_forecasts: tuple[PanelAdmissionForecast, ...],
) -> tuple[core.PolicyBranch, str]:
    by_action = {value.action_id: value for value in evaluation_forecasts}
    if set(by_action) != {
        *(value.action_id for value in development_policy.action_fibers),
        "hold",
    }:
        raise ValueError("evaluation forecasts do not cover the native chart")
    if development_policy.denominator_structure is core.DenominatorStructure.INCOMPATIBLE:
        return core.PolicyBranch.NONATTEMPT, "nonattempt"
    ranked = sorted(
        development_policy.action_fibers,
        key=lambda value: (value.development_rank, value.action_id),
    )
    selected = next(
        (value for value in ranked if by_action[value.action_id].predicted_admitted), None
    )
    if selected is not None:
        return core.PolicyBranch.EXACT_ACTION, selected.action_id
    if by_action["hold"].predicted_admitted:
        return core.PolicyBranch.HOLD, "hold"
    return core.PolicyBranch.NONATTEMPT, "nonattempt"


def forecast_validation_disposition(
    *,
    branch: core.PolicyBranch,
    action_id: str,
    forecasts: tuple[PanelAdmissionForecast, ...],
) -> tuple[core.ProspectiveDisposition, PolicyValidationDisposition]:
    by_action = {value.action_id: value for value in forecasts}
    if branch is core.PolicyBranch.NONATTEMPT:
        return core.ProspectiveDisposition.NONATTEMPT, PolicyValidationDisposition.NOT_ATTEMPTED
    if "hold" not in by_action:
        raise ValueError("validation forecast omits hold")
    if branch is core.PolicyBranch.HOLD:
        return (
            core.ProspectiveDisposition.CONDITION_FALSE,
            PolicyValidationDisposition.VALIDATED
            if by_action["hold"].predicted_admitted
            else PolicyValidationDisposition.OPPOSED,
        )
    viable = (
        action_id in by_action
        and by_action[action_id].predicted_admitted
        and by_action["hold"].predicted_admitted
    )
    return (
        core.ProspectiveDisposition.VALIDATED if viable else core.ProspectiveDisposition.OPPOSED,
        PolicyValidationDisposition.VALIDATED if viable else PolicyValidationDisposition.OPPOSED,
    )


def build_prediction_issue(
    *,
    issue_id: str,
    method_freeze: MarginStructuralRecurrenceForecastMethodFreeze,
    conformance: MarginStructuralRecurrenceForecastConformance,
    design: StructuralRecurrenceTargetDesignFreeze,
    development: StructuralRecurrenceStageEvidence,
    evaluation_count: int,
    prospective_validation_count: int,
    scientific_bootstrap_inputs: StructuralBootstrapInputCensus,
) -> MarginStructuralRecurrenceForecastPredictionIssue:
    seed_inputs = require_margin_bootstrap_inputs(
        scientific_bootstrap_inputs=scientific_bootstrap_inputs, design=design,
        development=development, evaluation_count=evaluation_count, prospective_validation_count=prospective_validation_count,
        bootstrap_replications=method_freeze.bootstrap_replications,
    )
    base_policy, history, support = build_policy_signature(
        design=design,
        evidence=development,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    base = ActionFiberStructuralRecurrencePredictionIssue(
        issue_id=f"{issue_id}.action-fiber",
        method_freeze=ObjectIdentity.from_record(method_freeze.freeze_id, method_freeze),
        conformance=ObjectIdentity.from_record(conformance.conformance_id, conformance),
        target_design=ObjectIdentity.from_record(design.design_id, design),
        development_evidence=ObjectIdentity.from_record(development.evidence_id, development),
        target_slot_id=design.target_slot.target_slot_id,
        denominator_structure=base_policy.denominator_structure,
        history_clock_quotient=history,
        support_transport=support,
        admission_topology=(
            core.AdmissionTopology.EMPTY
            if not any(value.admitted for value in base_policy.action_fibers)
            else core.AdmissionTopology.SINGLETON
            if sum(value.admitted for value in base_policy.action_fibers) == 1
            else core.AdmissionTopology.MULTIPLE
        ),
        policy_safety=base_policy,
        prospective_validation_disposition=(
            core.ProspectiveDisposition.VALIDATED
            if base_policy.policy_branch is core.PolicyBranch.EXACT_ACTION
            else core.ProspectiveDisposition.CONDITION_FALSE
            if base_policy.policy_branch is core.PolicyBranch.HOLD
            else core.ProspectiveDisposition.NONATTEMPT
        ),
        validation_disposition=(
            PolicyValidationDisposition.VALIDATED
            if base_policy.policy_branch is not core.PolicyBranch.NONATTEMPT
            else PolicyValidationDisposition.NOT_ATTEMPTED
        ),
        valid_state_count=10_000,
        predicted_state_count=1,
        sharpness=Decimal("0.9999"),
        comparator_ids=("ASSUME_HOLD_SAFE", 'CATEGORICAL_TARGET_WIDE_RESTRICTION'),
        published_before_evaluation=True,
        protected_outcome_access_count=0,
    )
    margin_policy = build_margin_policy(
        design=design,
        evidence=development,
        base_policy=base_policy,
        binary_robustness_margin_min=method_freeze.binary_robustness_margin_min,
        target_robustness_ratio_min=method_freeze.target_robustness_ratio_min,
    )
    forecasts = tuple(
        sorted(
            (
                forecast_panel_admission(
                    design=design,
                    development=development,
                    development_margin=margin,
                    future_stage=stage,
                    future_independent_unit_count=(
                        evaluation_count if stage is StructuralRecurrenceTargetStage.EVALUATION else prospective_validation_count
                    ),
                    bootstrap_replications=method_freeze.bootstrap_replications,
                    scientific_seed_input=seed_inputs[(margin.action_fiber.action_id, stage)],
                )
                for margin in margin_policy.action_margins
                for stage in (StructuralRecurrenceTargetStage.EVALUATION, StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION)
            ),
            key=lambda value: value.forecast_id,
        )
    )
    evaluation_forecasts = tuple(
        value for value in forecasts if value.future_stage is StructuralRecurrenceTargetStage.EVALUATION
    )
    branch, action_id = forecast_policy_decision(base_policy, evaluation_forecasts)
    prospective_validation, validation = forecast_validation_disposition(
        branch=branch,
        action_id=action_id,
        forecasts=tuple(value for value in forecasts if value.future_stage is StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION),
    )
    return MarginStructuralRecurrenceForecastPredictionIssue(
        issue_id=issue_id,
        method_freeze=ObjectIdentity.from_record(method_freeze.freeze_id, method_freeze),
        conformance=ObjectIdentity.from_record(conformance.conformance_id, conformance),
        structural_prediction=base,
        development_margins=margin_policy,
        forecasts=forecasts,
        predicted_evaluation_policy_branch=branch,
        predicted_evaluation_action_id=action_id,
        predicted_prospective_validation_disposition=prospective_validation,
        predicted_validation_disposition=validation,
        published_before_evaluation=True,
        protected_outcome_access_count=0,
    )


def evaluate_margin_admission(
    *,
    method_freeze: MarginStructuralRecurrenceForecastMethodFreeze,
    prediction: MarginStructuralRecurrenceForecastPredictionIssue,
    design: StructuralRecurrenceTargetDesignFreeze,
    evaluation: StructuralRecurrenceStageEvidence,
    reveal_authority: ObjectIdentity,
) -> MarginStructuralRecurrenceForecastAdmissionHandoff:
    base = evaluate_admission(
        method_freeze=method_freeze,
        prediction=prediction.structural_prediction,
        design=design,
        evaluation=evaluation,
        reveal_authority=reveal_authority,
    )
    margins = build_margin_policy(
        design=design,
        evidence=evaluation,
        base_policy=base.policy_safety,
        binary_robustness_margin_min=method_freeze.binary_robustness_margin_min,
        target_robustness_ratio_min=method_freeze.target_robustness_ratio_min,
    )
    return MarginStructuralRecurrenceForecastAdmissionHandoff(
        handoff_id=f"{design.target_slot.target_slot_id}.margin-admission-handoff",
        prediction_issue=ObjectIdentity.from_record(prediction.issue_id, prediction),
        action_fiber_admission=base,
        evaluation_margins=margins,
        reveal_authority=reveal_authority,
    )


def finalize_margin_target(
    *,
    admission_handoff: MarginStructuralRecurrenceForecastAdmissionHandoff,
    design: StructuralRecurrenceTargetDesignFreeze,
    validation_evidence: StructuralRecurrenceStageEvidence | None,
    validation_authority: ObjectIdentity,
) -> ActionFiberStructuralRecurrenceTargetResult:
    return finalize_target(
        admission_handoff=admission_handoff.action_fiber_admission,
        design=design,
        validation_evidence=validation_evidence,
        validation_authority=validation_authority,
    )


def match_target(
    *,
    prediction: MarginStructuralRecurrenceForecastPredictionIssue,
    admission_handoff: MarginStructuralRecurrenceForecastAdmissionHandoff,
    result: ActionFiberStructuralRecurrenceTargetResult,
) -> MarginStructuralRecurrenceForecastTargetMatch:
    evaluation_forecasts = {
        value.action_id: value
        for value in prediction.forecasts
        if value.future_stage is StructuralRecurrenceTargetStage.EVALUATION
    }
    predicted_bands = {
        value.action_fiber.action_id: value.band
        for value in prediction.development_margins.action_margins
    }
    observed_margins = {
        value.action_fiber.action_id: value for value in admission_handoff.evaluation_margins.action_margins
    }
    matches = tuple(
        ForecastMatch(
            match_id=f"{prediction.structural_prediction.target_slot_id}.{action_id}.margin-forecast-match",
            action_id=action_id,
            predicted_probability=forecast.admission_probability,
            predicted_admitted=forecast.predicted_admitted,
            observed_admitted=observed_margins[action_id].action_fiber.admitted,
            brier_score=(
                forecast.admission_probability
                - Decimal(1 if observed_margins[action_id].action_fiber.admitted else 0)
            )
            ** 2,
            binary_only_comparator_probability=forecast.binary_only_comparator_probability,
            binary_only_comparator_brier=(
                forecast.binary_only_comparator_probability
                - Decimal(1 if observed_margins[action_id].action_fiber.admitted else 0)
            )
            ** 2,
            margin_band_predicted=predicted_bands[action_id],
            margin_band_observed=observed_margins[action_id].band,
            admission_exact=(
                forecast.predicted_admitted == observed_margins[action_id].action_fiber.admitted
            ),
            band_exact=predicted_bands[action_id] is observed_margins[action_id].band,
        )
        for action_id, forecast in sorted(evaluation_forecasts.items())
    )
    policy_exact = (
        prediction.predicted_evaluation_policy_branch is admission_handoff.action_fiber_admission.policy_safety.policy_branch
    )
    action_exact = (
        prediction.predicted_evaluation_action_id == admission_handoff.action_fiber_admission.policy_safety.selected_action_id
    )
    validation_exact = (
        prediction.predicted_validation_disposition is result.validation_disposition
        and prediction.predicted_prospective_validation_disposition is result.prospective_validation_disposition
    )
    selected = admission_handoff.action_fiber_admission.policy_safety.selected_fiber
    false_action = int(
        admission_handoff.action_fiber_admission.policy_safety.policy_branch is core.PolicyBranch.EXACT_ACTION
        and (selected is None or not selected.admitted)
    )
    hold_error = int(
        admission_handoff.action_fiber_admission.policy_safety.policy_branch is core.PolicyBranch.HOLD
        and not admission_handoff.action_fiber_admission.policy_safety.hold_viability.hold_fiber.admitted
    )
    safety_errors = false_action + hold_error
    reasons: set[str] = set()
    if not all(value.admission_exact for value in matches):
        reasons.add("ACTION_MARGIN_FORECAST_MISMATCH")
    if not policy_exact:
        reasons.add("POLICY_BRANCH_MISMATCH")
    if not action_exact:
        reasons.add("ACTION_IDENTITY_MISMATCH")
    if not validation_exact:
        reasons.add("POLICY_VALIDATION_MISMATCH")
    if safety_errors:
        reasons.add("SAFETY_TYPING_ERROR")
    passed = all(
        (
            all(value.admission_exact for value in matches),
            policy_exact,
            action_exact,
            validation_exact,
            not safety_errors,
        )
    )
    return MarginStructuralRecurrenceForecastTargetMatch(
        match_id=f"{prediction.structural_prediction.target_slot_id}.margin-target-match",
        prediction_issue=ObjectIdentity.from_record(prediction.issue_id, prediction),
        target_result=ObjectIdentity.from_record(result.result_id, result),
        target_slot_id=prediction.structural_prediction.target_slot_id,
        forecast_matches=matches,
        evaluation_policy_exact=policy_exact,
        selected_action_exact=action_exact,
        validation_exact=validation_exact,
        safety_error_count=safety_errors,
        exact_forecast_count=sum(value.admission_exact for value in matches),
        exact_band_count=sum(value.band_exact for value in matches),
        margin_model_comparator_win_count=sum(
            value.brier_score < value.binary_only_comparator_brier for value in matches
        ),
        passed=passed,
        reason_codes=tuple(sorted(reasons)),
    )


def adjudicate(
    *,
    conformance: MarginStructuralRecurrenceForecastConformance,
    matches: tuple[MarginStructuralRecurrenceForecastTargetMatch, ...],
) -> MarginStructuralRecurrenceForecastAdjudication:
    ordered = tuple(sorted(matches, key=lambda value: value.match_id))
    passed = sum(value.passed for value in ordered)
    safety = sum(value.safety_error_count for value in ordered)
    if not conformance.method_qualified:
        verdict = MarginStructuralRecurrenceForecastTerminalVerdict.METHOD_NOT_QUALIFIED
    elif safety:
        verdict = MarginStructuralRecurrenceForecastTerminalVerdict.SAFETY_TYPING_OPPOSED
    elif passed == 4:
        verdict = MarginStructuralRecurrenceForecastTerminalVerdict.BOUNDED_FOUR_TARGET
    elif passed == 3:
        verdict = MarginStructuralRecurrenceForecastTerminalVerdict.BOUNDED_THREE_TARGET
    elif passed == 2:
        verdict = MarginStructuralRecurrenceForecastTerminalVerdict.BOUNDED_TWO_TARGET
    elif passed == 1:
        verdict = MarginStructuralRecurrenceForecastTerminalVerdict.SUBSTRATE_LOCAL
    else:
        verdict = MarginStructuralRecurrenceForecastTerminalVerdict.MARGIN_FORECAST_OPPOSED
    reasons = {code for value in ordered for code in value.reason_codes} | {
        "ONE_NEW_GENERATED_SUBSTRATE_CLASS_ONLY"
    }
    return MarginStructuralRecurrenceForecastAdjudication(
        adjudication_id='margin-structural-recurrence-forecast.cross-target-adjudication',
        method_conformance=ObjectIdentity.from_record(conformance.conformance_id, conformance),
        target_matches=tuple(
            ObjectIdentity.from_record(value.match_id, value) for value in ordered
        ),
        eligible_target_count=len(ordered),
        passed_target_count=passed,
        failed_target_count=len(ordered) - passed,
        safety_error_count=safety,
        exact_forecast_count=sum(value.exact_forecast_count for value in ordered),
        total_forecast_count=sum(len(value.forecast_matches) for value in ordered),
        exact_band_count=sum(value.exact_band_count for value in ordered),
        margin_model_comparator_win_count=sum(
            value.margin_model_comparator_win_count for value in ordered
        ),
        verdict=verdict,
        positive_claim_eligible=passed == len(ordered) == 4 and not safety,
        new_substrate_class_count=1,
        independent_generality_claim_eligible=False,
        no_cross_target_pooling=True,
        reason_codes=tuple(sorted(reasons)),
    )


def source_semantics_sha256() -> str:
    return sha256(
        b'margin-structural-recurrence-forecast:typed-gate-margins:joint-unit-bootstrap:policy-failure-forecast:no-pooling'
    ).hexdigest()


__all__ = [
    "ActionMarginSignature",
    "ForecastMatch",
    "GateMarginFact",
    "MARGIN_COMPONENT_IDS",
    "MarginBand",
    "MarginControlResult",
    "MarginFixtureResult",
    "MarginKind",
    "MarginPolicySignature",
    'MarginStructuralRecurrenceForecastAdjudication',
    'MarginStructuralRecurrenceForecastConformance',
    'MarginStructuralRecurrenceForecastMethodFreeze',
    'MarginStructuralRecurrenceForecastAdmissionHandoff',
    'MarginStructuralRecurrenceForecastPredictionIssue',
    'MarginStructuralRecurrenceForecastTargetMatch',
    'MarginStructuralRecurrenceForecastTerminalVerdict',
    "PanelAdmissionForecast",
    "adjudicate",
    "build_action_margin",
    "build_margin_policy",
    "build_prediction_issue",
    "classify_margin_band",
    'evaluate_margin_admission',
    "finalize_margin_target",
    "forecast_panel_admission",
    "forecast_policy_decision",
    "forecast_validation_disposition",
    "match_target",
]

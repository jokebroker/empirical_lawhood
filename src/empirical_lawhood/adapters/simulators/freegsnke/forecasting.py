"Source-qualified categorical forecast encoding for independent substrate grounding FreeGSNKE.\n\nThe encoder freezes target-native categorical subjects and integer keys before\ndevelopment.  It converts only compact target reductions and their selected\nstructural recurrence compatibility evidence; it never opens simulator artifacts.  Evaluation\nconversion is an evaluator-only reveal operation.  structural recurrence emissions are derived\nonly from the immutable prediction issue and never from evaluation outcomes.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_comparators import IndependentSubstrateCategoricalForecastCase, IndependentSubstrateCategoricalForecastPanel, IndependentSubstrateForecastPanelPhase, IndependentSubstrateForecastPrediction, fit_independent_substrate_comparator_encodings
from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateComparatorEncoding, IndependentSubstrateForecastAlphabetGrammar
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceActionOutcome, StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetStage
from empirical_lawhood.adapters.methods.structural_recurrence import DenominatorStructure, HistoryClockQuotient, SupportTransport
from empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast import MarginStructuralRecurrenceForecastPredictionIssue
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

from .structural_recurrence_bridge import FreeGsnkeStructuralRecurrenceDesignCandidate
from .target_analysis import FreeGsnkePhaseReduction, FreeGsnkePreparationReduction
from .target_design import FREEGSNKE_FORECAST_STATE_IDS, FREEGSNKE_REQUIRED_FORECAST_ROLE_IDS, FreeGsnkeForecastAlphabetFreeze, FreeGsnkeTargetForecast


class FreeGsnkeForecastCodeNamespace(StrEnum):
    FORECAST = "FORECAST"
    DENOMINATOR = "D"
    HISTORY = "H"
    ACTION = "A"
    RECEIVER = "R"
    HORIZON = "tau"
    METHOD = "METHOD"
    MAPPING = "MAPPING"
    DENOMINATOR_CANDIDATE = "DENOMINATOR_CANDIDATE"


class FreeGsnkeForecastStateRule(StrEnum):
    ACTION_REALIZATION = "ACTION_REALIZATION_STATE"
    UNIT_ADMISSION = "UNIT_ADMISSION_STATE"
    DENOMINATOR_SUPPORT = "DENOMINATOR_SUPPORT_STATE"
    EPISODE_FAILURE = "EPISODE_FAILURE_STATE"
    HISTORY_SUPPORT = "HISTORY_SUPPORT_STATE"
    HOLD_VIABILITY = "HOLD_VIABILITY_STATE"
    RECEIVER_SINK = "RECEIVER_SINK_STATE"
    TARGET_SUPPORT = "TARGET_SUPPORT_STATE"


_RULE_BY_ROLE = {
    "action": FreeGsnkeForecastStateRule.ACTION_REALIZATION,
    "admission": FreeGsnkeForecastStateRule.UNIT_ADMISSION,
    "denominator": FreeGsnkeForecastStateRule.DENOMINATOR_SUPPORT,
    "failure": FreeGsnkeForecastStateRule.EPISODE_FAILURE,
    "history": FreeGsnkeForecastStateRule.HISTORY_SUPPORT,
    "hold": FreeGsnkeForecastStateRule.HOLD_VIABILITY,
    "sink": FreeGsnkeForecastStateRule.RECEIVER_SINK,
    "support": FreeGsnkeForecastStateRule.TARGET_SUPPORT,
}


@dataclass(frozen=True, slots=True)
class FreeGsnkeForecastIntegerCode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-forecast-integer-code'

    code_id: str
    namespace: FreeGsnkeForecastCodeNamespace
    native_id: str
    integer_code: int

    def __post_init__(self) -> None:
        validate_stable_id(self.code_id, field_name="code_id")
        validate_stable_id(self.native_id, field_name="native_id")
        if self.integer_code < 0:
            raise ValueError("FreeGSNKE forecast integer code must be nonnegative")


@dataclass(frozen=True, slots=True)
class FreeGsnkeForecastStateCode(CanonicalRecord):
    """One explicit target-state binding to an outcome-blind scientific grammar common integer code."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-forecast-state-code'

    binding_id: str
    target_state_id: str
    common_state_id: str
    integer_code: int

    def __post_init__(self) -> None:
        for name in ("binding_id", "common_state_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.target_state_id, field_name="target_state_id")
        if self.integer_code < 0:
            raise ValueError("FreeGSNKE forecast state code must be nonnegative")


@dataclass(frozen=True, slots=True)
class FreeGsnkeForecastCaseTemplate(CanonicalRecord):
    """One predevelopment target-native forecast subject and state rule."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-forecast-case-template'

    template_id: str
    forecast_id: str
    role_id: str
    state_rule: FreeGsnkeForecastStateRule
    action_id: str
    receiver_id: str
    horizon_s: Decimal
    horizon_code_id: str
    response_direction: Decimal
    materiality_scale: Decimal
    hold_emission_state_ids: tuple[str, ...]
    admit_or_act_emission_state_ids: tuple[str, ...]
    unsafe_admission_state_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "template_id",
            "forecast_id",
            "role_id",
            "action_id",
            "receiver_id",
            "horizon_code_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.role_id not in _RULE_BY_ROLE or self.state_rule is not _RULE_BY_ROLE[self.role_id]:
            raise ValueError("FreeGSNKE forecast role/state rule differs")
        validate_decimal(self.horizon_s, field_name="horizon_s", minimum=Decimal(0))
        validate_decimal(self.response_direction, field_name="response_direction")
        if self.response_direction not in {Decimal(-1), Decimal(1)}:
            raise ValueError("FreeGSNKE forecast response direction must be signed")
        validate_decimal(
            self.materiality_scale,
            field_name="materiality_scale",
            minimum=Decimal(0),
        )
        if self.materiality_scale == 0:
            raise ValueError("FreeGSNKE forecast materiality scale must be positive")
        for name in (
            "hold_emission_state_ids",
            "admit_or_act_emission_state_ids",
            "unsafe_admission_state_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=name == "unsafe_admission_state_ids",
            )


@dataclass(frozen=True, slots=True)
class FreeGsnkeForecastEncoderFreeze(CanonicalRecord):
    """Exact finite key/state encoder frozen during source qualification."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-forecast-encoder-freeze'

    encoder_id: str
    codes: tuple[FreeGsnkeForecastIntegerCode, ...]
    state_codes: tuple[FreeGsnkeForecastStateCode, ...]
    templates: tuple[FreeGsnkeForecastCaseTemplate, ...]
    key_namespace_order: tuple[FreeGsnkeForecastCodeNamespace, ...]
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.encoder_id, field_name="encoder_id")
        require_sorted_unique_ids(self.codes, attribute="code_id", field_name="codes")
        if (
            not self.codes
            or len({value.integer_code for value in self.codes}) != len(self.codes)
            or len({(value.namespace, value.native_id) for value in self.codes}) != len(self.codes)
        ):
            raise ValueError("FreeGSNKE forecast codes are empty or nonunique")
        require_sorted_unique_ids(
            self.state_codes,
            attribute="binding_id",
            field_name="state_codes",
        )
        if (
            {value.target_state_id for value in self.state_codes}
            != set(FREEGSNKE_FORECAST_STATE_IDS)
            or len({value.common_state_id for value in self.state_codes}) != len(self.state_codes)
            or len({value.integer_code for value in self.state_codes}) != len(self.state_codes)
        ):
            raise ValueError("FreeGSNKE forecast state bindings do not cover the exact alphabet")
        require_sorted_unique_ids(
            self.templates,
            attribute="template_id",
            field_name="templates",
        )
        if (
            not self.templates
            or len({value.forecast_id for value in self.templates}) != len(self.templates)
            or {value.role_id for value in self.templates}
            != set(FREEGSNKE_REQUIRED_FORECAST_ROLE_IDS)
        ):
            raise ValueError("FreeGSNKE forecast templates do not cover exact roles")
        if self.key_namespace_order != (
            FreeGsnkeForecastCodeNamespace.FORECAST,
            FreeGsnkeForecastCodeNamespace.DENOMINATOR,
            FreeGsnkeForecastCodeNamespace.HISTORY,
            FreeGsnkeForecastCodeNamespace.ACTION,
            FreeGsnkeForecastCodeNamespace.RECEIVER,
            FreeGsnkeForecastCodeNamespace.HORIZON,
        ):
            raise ValueError("FreeGSNKE forecast key namespace order differs")
        code_keys = {(value.namespace, value.native_id) for value in self.codes}
        for template in self.templates:
            required = {
                (FreeGsnkeForecastCodeNamespace.FORECAST, template.forecast_id),
                (FreeGsnkeForecastCodeNamespace.ACTION, template.action_id),
                (FreeGsnkeForecastCodeNamespace.RECEIVER, template.receiver_id),
                (
                    FreeGsnkeForecastCodeNamespace.HORIZON,
                    template.horizon_code_id,
                ),
            }
            if not required.issubset(code_keys):
                raise ValueError("FreeGSNKE forecast template lacks a frozen key code")
        if not self.frozen_before_development or self.protected_outcome_access_count:
            raise ValueError("FreeGSNKE forecast encoder crossed development access")

    def code(
        self,
        namespace: FreeGsnkeForecastCodeNamespace,
        native_id: str,
    ) -> int:
        try:
            return next(
                value.integer_code
                for value in self.codes
                if value.namespace is namespace and value.native_id == native_id
            )
        except StopIteration as error:
            raise ValueError("FreeGSNKE forecast encoder omits a native key") from error

    def template(self, forecast_id: str) -> FreeGsnkeForecastCaseTemplate:
        try:
            return next(value for value in self.templates if value.forecast_id == forecast_id)
        except StopIteration as error:
            raise ValueError("FreeGSNKE forecast encoder omits a forecast") from error


def _state_codes(
    encoder: FreeGsnkeForecastEncoderFreeze,
    grammar: IndependentSubstrateForecastAlphabetGrammar,
) -> dict[str, int]:
    common = {value.state_id: value.integer_code for value in grammar.common_state_codes}
    if any(
        common.get(value.common_state_id) != value.integer_code for value in encoder.state_codes
    ):
        raise ValueError("FreeGSNKE forecast state binding differs from outcome-blind scientific grammar grammar")
    return {value.target_state_id: value.integer_code for value in encoder.state_codes}


def _unit_outcome(
    evidence: StructuralRecurrenceStageEvidence,
    unit_id: str,
    action_id: str,
) -> StructuralRecurrenceActionOutcome:
    unit = next(value for value in evidence.units if value.unit_id == unit_id)
    return next(value for value in unit.outcomes if value.action_id == action_id)


def _unsafe(outcome: StructuralRecurrenceActionOutcome, design: FreeGsnkeStructuralRecurrenceDesignCandidate) -> bool:
    return any(
        (
            outcome.realization_error > design.design.threshold("realization-error-max"),
            outcome.sink_margin < design.design.threshold("sink-margin-min"),
            not outcome.validity_passed,
            not outcome.preservation_passed,
            not outcome.dynamics_passed,
            not outcome.reachability_passed,
            not outcome.authority_passed,
        )
    )


def _target_effect(
    *,
    unit: FreeGsnkePreparationReduction,
    template: FreeGsnkeForecastCaseTemplate,
    design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
) -> Decimal | None:
    binding = next(
        value
        for value in design_candidate.action_bindings
        if value.structural_recurrence_action_id == template.action_id
    )
    if template.action_id == "hold":
        return Decimal(0)
    try:
        contrast = next(
            value
            for value in unit.contrasts
            if value.branch_id == binding.branch_id
            and value.coordinate_id == template.receiver_id
            and value.horizon_s == template.horizon_s
        )
    except StopIteration:
        return None
    return template.response_direction * contrast.delta / template.materiality_scale


def _admitted(
    outcome: StructuralRecurrenceActionOutcome,
    design: FreeGsnkeStructuralRecurrenceDesignCandidate,
    *,
    target_effect: Decimal | None,
) -> bool:
    return all(
        (
            not _unsafe(outcome, design),
            target_effect is not None,
            outcome.action_id == "hold"
            or (
                target_effect is not None
                and target_effect >= design.design.threshold("target-effect-min")
            ),
            outcome.uncertainty_evaluable,
            outcome.effort <= design.design.threshold("effort-max"),
            outcome.denominator_distance <= design.design.threshold("denominator-view-local-max"),
            outcome.history_checkpoint_distance <= design.design.threshold("history-close-max"),
            outcome.history_future_distance <= design.design.threshold("history-close-max"),
            outcome.direct_composed_distance <= design.design.threshold("direct-composed-max"),
            outcome.timing_offset <= design.design.threshold("timing-offset-max"),
        )
    )


def _observed_state_id(
    *,
    template: FreeGsnkeForecastCaseTemplate,
    outcome: StructuralRecurrenceActionOutcome,
    design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
    target_effect: Decimal | None,
) -> str:
    unsafe = _unsafe(outcome, design_candidate) or target_effect is None
    admitted = _admitted(
        outcome,
        design_candidate,
        target_effect=target_effect,
    )
    rule = template.state_rule
    if rule is FreeGsnkeForecastStateRule.ACTION_REALIZATION:
        return (
            "FAILURE"
            if outcome.realization_error
            > design_candidate.design.threshold("realization-error-max")
            else "ACTION"
        )
    if rule is FreeGsnkeForecastStateRule.UNIT_ADMISSION:
        if admitted:
            return "HOLD" if outcome.action_id == "hold" else "ADMIT"
        return "UNSAFE" if unsafe else "NONENTRY"
    if rule is FreeGsnkeForecastStateRule.DENOMINATOR_SUPPORT:
        return (
            "SUPPORT"
            if outcome.denominator_distance
            <= design_candidate.design.threshold("denominator-view-local-max")
            else "DOMAIN_LOSS"
        )
    if rule is FreeGsnkeForecastStateRule.EPISODE_FAILURE:
        return "FAILURE" if unsafe else "VALID"
    if rule is FreeGsnkeForecastStateRule.HISTORY_SUPPORT:
        return (
            "SUPPORT"
            if max(
                outcome.history_checkpoint_distance,
                outcome.history_future_distance,
            )
            <= design_candidate.design.threshold("history-close-max")
            else "DOMAIN_LOSS"
        )
    if rule is FreeGsnkeForecastStateRule.HOLD_VIABILITY:
        return "HOLD" if outcome.action_id == "hold" and admitted else "UNSAFE"
    if rule is FreeGsnkeForecastStateRule.RECEIVER_SINK:
        return (
            "SINK"
            if outcome.sink_margin < design_candidate.design.threshold("sink-margin-min")
            else "VALID"
        )
    if rule is FreeGsnkeForecastStateRule.TARGET_SUPPORT:
        return (
            "SUPPORT"
            if target_effect is not None
            and (
                outcome.action_id == "hold"
                or target_effect >= design_candidate.design.threshold("target-effect-min")
            )
            and outcome.validity_passed
            else "DOMAIN_LOSS"
        )
    raise AssertionError("unreachable FreeGSNKE forecast rule")


def _validate_freeze(
    *,
    encoder: FreeGsnkeForecastEncoderFreeze,
    grammar: IndependentSubstrateForecastAlphabetGrammar,
    alphabet: FreeGsnkeForecastAlphabetFreeze,
) -> dict[str, FreeGsnkeTargetForecast]:
    if alphabet.common_grammar != ObjectIdentity.from_record(
        grammar.grammar_id, grammar
    ) or alphabet.target_encoder != ObjectIdentity.from_record(encoder.encoder_id, encoder):
        raise ValueError("FreeGSNKE forecast encoder/alphabet grammar differs")
    forecasts = {value.forecast_id: value for value in alphabet.forecasts}
    if set(forecasts) != {value.forecast_id for value in encoder.templates}:
        raise ValueError("FreeGSNKE forecast encoder/alphabet roster differs")
    state_codes = _state_codes(encoder, grammar)
    for forecast_id, forecast in forecasts.items():
        template = encoder.template(forecast_id)
        if template.role_id != forecast.role_id or not {
            template.action_id,
            template.receiver_id,
            template.horizon_code_id,
        }.issubset(forecast.subject_native_ids):
            raise ValueError("FreeGSNKE forecast template/subject differs")
        if not set(forecast.alphabet_state_ids).issubset(state_codes):
            raise ValueError("FreeGSNKE forecast alphabet lacks a frozen state code")
        for state_ids in (
            template.hold_emission_state_ids,
            template.admit_or_act_emission_state_ids,
            template.unsafe_admission_state_ids,
        ):
            if not set(state_ids).issubset(forecast.alphabet_state_ids):
                raise ValueError("FreeGSNKE forecast comparator emission is illegal")
    return forecasts


def build_freegsnke_forecast_panel(
    *,
    panel_id: str,
    encoder: FreeGsnkeForecastEncoderFreeze,
    grammar: IndependentSubstrateForecastAlphabetGrammar,
    alphabet: FreeGsnkeForecastAlphabetFreeze,
    design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
    phase_reduction: FreeGsnkePhaseReduction,
    structural_recurrence_evidence: StructuralRecurrenceStageEvidence,
    evaluator_reveal_authorized: bool,
) -> IndependentSubstrateCategoricalForecastPanel:
    """Encode all issued units; evaluation requires the evaluator reveal boundary."""

    forecasts = _validate_freeze(encoder=encoder, grammar=grammar, alphabet=alphabet)
    phase = {
        StructuralRecurrenceTargetStage.DEVELOPMENT: IndependentSubstrateForecastPanelPhase.DEVELOPMENT,
        StructuralRecurrenceTargetStage.EVALUATION: IndependentSubstrateForecastPanelPhase.EVALUATION,
    }.get(structural_recurrence_evidence.stage)
    if phase is None:
        raise ValueError("FreeGSNKE forecast panel supports development/evaluation only")
    expected_reduction_access = {
        IndependentSubstrateForecastPanelPhase.DEVELOPMENT: OutcomeAccess.DEVELOPMENT_VISIBLE,
        IndependentSubstrateForecastPanelPhase.EVALUATION: OutcomeAccess.EVALUATION_SEALED,
    }[phase]
    if (
        phase_reduction.outcome_access is not expected_reduction_access
        or (phase is IndependentSubstrateForecastPanelPhase.EVALUATION) != evaluator_reveal_authorized
    ):
        raise ValueError("FreeGSNKE forecast panel crossed the reveal boundary")
    units = {value.unit_id: value for value in phase_reduction.units}
    evidence_unit_ids = tuple(value.unit_id for value in structural_recurrence_evidence.units)
    if set(units) != set(evidence_unit_ids):
        raise ValueError("FreeGSNKE forecast panel reduction/evidence units differ")
    state_codes = _state_codes(encoder, grammar)
    cases = []
    for unit_id in evidence_unit_ids:
        unit: FreeGsnkePreparationReduction = units[unit_id]
        candidate_denominator = design_candidate.project_denominator_stratum(
            unit.denominator_stratum_id
        )
        for template in encoder.templates:
            forecast = forecasts[template.forecast_id]
            outcome = _unit_outcome(structural_recurrence_evidence, unit_id, template.action_id)
            target_effect = _target_effect(
                unit=unit,
                template=template,
                design_candidate=design_candidate,
            )
            observed_state_id = _observed_state_id(
                template=template,
                outcome=outcome,
                design_candidate=design_candidate,
                target_effect=target_effect,
            )
            if observed_state_id not in forecast.alphabet_state_ids:
                raise ValueError("FreeGSNKE observed state lies outside frozen alphabet")
            cases.append(
                IndependentSubstrateCategoricalForecastCase(
                    case_id=f"case.{panel_id}.{unit_id}.{template.template_id}",
                    complete_unit_id=unit_id,
                    forecast_id=template.forecast_id,
                    forecast_code=encoder.code(
                        FreeGsnkeForecastCodeNamespace.FORECAST,
                        template.forecast_id,
                    ),
                    key_codes=(
                        encoder.code(
                            FreeGsnkeForecastCodeNamespace.FORECAST,
                            template.forecast_id,
                        ),
                        encoder.code(
                            FreeGsnkeForecastCodeNamespace.DENOMINATOR,
                            candidate_denominator,
                        ),
                        encoder.code(
                            FreeGsnkeForecastCodeNamespace.HISTORY,
                            unit.history_stratum_id,
                        ),
                        encoder.code(
                            FreeGsnkeForecastCodeNamespace.ACTION,
                            template.action_id,
                        ),
                        encoder.code(
                            FreeGsnkeForecastCodeNamespace.RECEIVER,
                            template.receiver_id,
                        ),
                        encoder.code(
                            FreeGsnkeForecastCodeNamespace.HORIZON,
                            template.horizon_code_id,
                        ),
                    ),
                    denominator_key_positions=(1,),
                    receiver_key_positions=(4,),
                    legal_state_codes=tuple(
                        sorted(state_codes[value] for value in forecast.alphabet_state_ids)
                    ),
                    observed_state_codes=(state_codes[observed_state_id],),
                    hold_state_codes=tuple(
                        sorted(state_codes[value] for value in template.hold_emission_state_ids)
                    ),
                    admit_or_act_state_codes=tuple(
                        sorted(
                            state_codes[value] for value in template.admit_or_act_emission_state_ids
                        )
                    ),
                    unsafe_admission_state_codes=tuple(
                        sorted(state_codes[value] for value in template.unsafe_admission_state_ids)
                    ),
                    unsafe_observed=(_unsafe(outcome, design_candidate) or target_effect is None),
                    phase=phase,
                )
            )
    complete_unit_ids = tuple(sorted(units))
    return IndependentSubstrateCategoricalForecastPanel(
        panel_id=panel_id,
        phase=phase,
        cases=tuple(sorted(cases, key=lambda value: value.case_id)),
        complete_unit_ids=complete_unit_ids,
        complete_unit_ids_sha256=phase_reduction.issued_unit_ids_sha256,
        nested_cases_count_as_units=False,
        outcome_access=(
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if phase is IndependentSubstrateForecastPanelPhase.DEVELOPMENT
            else OutcomeAccess.EVALUATOR_REVEAL
        ),
    )


def fit_freegsnke_comparator_encodings(
    *,
    namespace_id: str,
    encoder: FreeGsnkeForecastEncoderFreeze,
    development_panel: IndependentSubstrateCategoricalForecastPanel,
    selected_mapping_candidate_id: str,
    selected_denominator_candidate_id: str,
) -> tuple[IndependentSubstrateComparatorEncoding, ...]:
    """Fit outcome-blind scientific grammar comparators with only source-frozen target parameter codes."""

    return fit_independent_substrate_comparator_encodings(
        namespace_id=namespace_id,
        development_panel=development_panel,
        structural_recurrence_method_parameter_codes=(
            encoder.code(FreeGsnkeForecastCodeNamespace.METHOD, 'margin-structural-recurrence-forecast'),
            encoder.code(
                FreeGsnkeForecastCodeNamespace.MAPPING,
                selected_mapping_candidate_id,
            ),
            encoder.code(
                FreeGsnkeForecastCodeNamespace.DENOMINATOR_CANDIDATE,
                selected_denominator_candidate_id,
            ),
        ),
        denominator_dependency_ids=(selected_denominator_candidate_id,),
        receiver_dependency_ids=tuple(sorted({value.receiver_id for value in encoder.templates})),
    )


def build_freegsnke_structural_recurrence_forecast_predictions(
    *,
    encoder: FreeGsnkeForecastEncoderFreeze,
    grammar: IndependentSubstrateForecastAlphabetGrammar,
    alphabet: FreeGsnkeForecastAlphabetFreeze,
    evaluation_panel: IndependentSubstrateCategoricalForecastPanel,
    prediction_issue: MarginStructuralRecurrenceForecastPredictionIssue,
) -> tuple[IndependentSubstrateForecastPrediction, ...]:
    'Project the immutable structural recurrence issue into the frozen target alphabets.'

    forecasts = _validate_freeze(encoder=encoder, grammar=grammar, alphabet=alphabet)
    if evaluation_panel.phase is not IndependentSubstrateForecastPanelPhase.EVALUATION:
        raise ValueError('structural recurrence target forecast projection requires evaluation cases')
    state_codes = _state_codes(encoder, grammar)
    admission_by_action = {
        value.action_id: value.predicted_admitted
        for value in prediction_issue.forecasts
        if value.future_stage is StructuralRecurrenceTargetStage.EVALUATION
    }
    values = []
    for case in evaluation_panel.cases:
        template = encoder.template(case.forecast_id)
        forecast = forecasts[case.forecast_id]
        admitted = admission_by_action[template.action_id]
        state_id: str | None
        if template.state_rule is FreeGsnkeForecastStateRule.ACTION_REALIZATION:
            # Aggregate panel admission implies successful realization; failure
            # of the intersection does not identify which component failed.
            state_id = "ACTION" if admitted else None
        elif template.state_rule is FreeGsnkeForecastStateRule.UNIT_ADMISSION:
            state_id = (
                "HOLD"
                if template.action_id == "hold" and admitted
                else "ADMIT"
                if admitted
                else "NONENTRY"
            )
        elif template.state_rule is FreeGsnkeForecastStateRule.DENOMINATOR_SUPPORT:
            state_id = (
                "DOMAIN_LOSS"
                if prediction_issue.structural_prediction.denominator_structure
                in {DenominatorStructure.INCOMPATIBLE, DenominatorStructure.UNEVALUABLE}
                else "SUPPORT"
            )
        elif template.state_rule is FreeGsnkeForecastStateRule.HISTORY_SUPPORT:
            state_id = (
                "SUPPORT"
                if prediction_issue.structural_prediction.history_clock_quotient
                is HistoryClockQuotient.CLOSE_CLOSED
                else "DOMAIN_LOSS"
            )
        elif template.state_rule is FreeGsnkeForecastStateRule.HOLD_VIABILITY:
            state_id = "HOLD" if admitted else "UNSAFE"
        elif template.state_rule is FreeGsnkeForecastStateRule.TARGET_SUPPORT:
            state_id = (
                "SUPPORT"
                if prediction_issue.structural_prediction.support_transport
                in {
                    SupportTransport.EQUALITY,
                    SupportTransport.CONSERVATIVE_INCLUSION,
                }
                else "DOMAIN_LOSS"
            )
        else:
            state_id = None
        emitted = (
            (state_codes[state_id],)
            if state_id is not None and state_id in forecast.alphabet_state_ids
            else case.legal_state_codes
        )
        values.append(
            IndependentSubstrateForecastPrediction(
                prediction_id=f"prediction.{prediction_issue.issue_id}.{case.case_id}",
                case_id=case.case_id,
                emitted_state_codes=tuple(sorted(emitted)),
            )
        )
    return tuple(sorted(values, key=lambda value: value.prediction_id))


__all__ = [
    'FreeGsnkeForecastCaseTemplate',
    'FreeGsnkeForecastCodeNamespace',
    'FreeGsnkeForecastEncoderFreeze',
    'FreeGsnkeForecastIntegerCode',
    'FreeGsnkeForecastStateCode',
    'FreeGsnkeForecastStateRule',
    "build_freegsnke_forecast_panel",
    'build_freegsnke_structural_recurrence_forecast_predictions',
    "fit_freegsnke_comparator_encodings",
]

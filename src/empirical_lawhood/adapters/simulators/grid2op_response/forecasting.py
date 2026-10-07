"""Finite Grid2Op forecast alphabet and common outcome-blind scientific grammar comparator projection."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_comparators import IndependentSubstrateCategoricalForecastCase, IndependentSubstrateCategoricalForecastPanel, IndependentSubstrateForecastPanelPhase, IndependentSubstrateForecastPrediction, fit_independent_substrate_comparator_encodings
from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateComparatorEncoding, IndependentSubstrateForecastAlphabetGrammar
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceActionOutcome, StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetDesignFreeze, StructuralRecurrenceTargetStage
from empirical_lawhood.adapters.methods.structural_recurrence import DenominatorStructure, HistoryClockQuotient
from empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast import MarginStructuralRecurrenceForecastPredictionIssue
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .forecast_denominator_inputs import Grid2OpForecastDenominatorInputs


class Grid2OpForecastRule(StrEnum):
    ACTION_REALIZATION = "ACTION_REALIZATION"
    DENOMINATOR_SUPPORT = "DENOMINATOR_SUPPORT"
    HISTORY_SUPPORT = "HISTORY_SUPPORT"
    RECEIVER_SINK = "RECEIVER_SINK"
    TARGET_SUPPORT = "TARGET_SUPPORT"
    UNIT_ADMISSION = "UNIT_ADMISSION"


@dataclass(frozen=True, slots=True)
class Grid2OpForecastAlphabet(CanonicalRecord):
    """Outcome-blind target instantiation of the frozen common state codes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/grid2op-response/grid2-op-forecast-alphabet'

    alphabet_id: str
    common_grammar: ObjectIdentity
    native_action_ids: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    horizon_steps: tuple[int, ...]
    rules: tuple[Grid2OpForecastRule, ...]
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.alphabet_id, field_name="alphabet_id")
        require_sorted_unique_strings(
            self.native_action_ids,
            field_name="native_action_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.receiver_ids,
            field_name="receiver_ids",
            allow_empty=False,
        )
        if (
            tuple(sorted(set(self.horizon_steps))) != self.horizon_steps
            or self.horizon_steps != (1, 3, 6)
            or tuple(sorted(set(self.rules), key=lambda value: value.value))
            != tuple(sorted(self.rules, key=lambda value: value.value))
            or set(self.rules) != set(Grid2OpForecastRule)
        ):
            raise ValueError("Grid2Op forecast alphabet grid/rules differ")
        if not self.frozen_before_development or self.protected_outcome_access_count:
            raise ValueError("Grid2Op forecast alphabet crossed development access")


_STATE_CODE_BY_ID = {
    "ACTION": 0,
    "ADMIT": 1,
    "DOMAIN_LOSS": 2,
    "FAILURE": 3,
    "HOLD": 4,
    "NONENTRY": 5,
    "SINK": 6,
    "SUPPORT": 7,
    "UNSAFE": 8,
    "VALID": 9,
}


def build_grid2op_forecast_alphabet(
    *,
    grammar: IndependentSubstrateForecastAlphabetGrammar,
    native_action_ids: tuple[str, ...],
) -> Grid2OpForecastAlphabet:
    observed = {
        value.state_id.removeprefix("independent-substrate-grounding.forecast-state.")
        .upper()
        .replace("-", "_"): value.integer_code
        for value in grammar.common_state_codes
    }
    if observed != _STATE_CODE_BY_ID:
        raise ValueError("Grid2Op forecast state codes differ from outcome-blind scientific grammar")
    return Grid2OpForecastAlphabet(
        alphabet_id="alphabet.independent-substrate-grounding.grid2op",
        common_grammar=ObjectIdentity.from_record(grammar.grammar_id, grammar),
        native_action_ids=native_action_ids,
        receiver_ids=("maximum-rho", "topology-state"),
        horizon_steps=(1, 3, 6),
        rules=tuple(sorted(Grid2OpForecastRule, key=lambda value: value.value)),
        frozen_before_development=True,
        protected_outcome_access_count=0,
    )


def _outcome(
    evidence: StructuralRecurrenceStageEvidence,
    unit_id: str,
    action_id: str,
) -> StructuralRecurrenceActionOutcome:
    unit = next(value for value in evidence.units if value.unit_id == unit_id)
    return next(value for value in unit.outcomes if value.action_id == action_id)


def _unsafe(outcome: StructuralRecurrenceActionOutcome, design: StructuralRecurrenceTargetDesignFreeze) -> bool:
    return any(
        (
            outcome.realization_error > design.threshold("realization-error-max"),
            outcome.sink_margin < design.threshold("sink-margin-min"),
            not outcome.validity_passed,
            not outcome.preservation_passed,
            not outcome.dynamics_passed,
            not outcome.reachability_passed,
            not outcome.authority_passed,
        )
    )


def _admitted(outcome: StructuralRecurrenceActionOutcome, design: StructuralRecurrenceTargetDesignFreeze) -> bool:
    return all(
        (
            not _unsafe(outcome, design),
            outcome.action_id == "hold"
            or outcome.target_effect >= design.threshold("target-effect-min"),
            outcome.support_preserved,
            outcome.uncertainty_evaluable,
            outcome.effort <= design.threshold("effort-max"),
            outcome.denominator_distance <= design.threshold("denominator-view-local-max"),
            outcome.history_checkpoint_distance <= design.threshold("history-close-max"),
            outcome.history_future_distance <= design.threshold("history-close-max"),
            outcome.direct_composed_distance <= design.threshold("direct-composed-max"),
            outcome.timing_offset <= design.threshold("timing-offset-max"),
        )
    )


def _observed_state(
    rule: Grid2OpForecastRule,
    outcome: StructuralRecurrenceActionOutcome,
    design: StructuralRecurrenceTargetDesignFreeze,
) -> str:
    unsafe = _unsafe(outcome, design)
    admitted = _admitted(outcome, design)
    if rule is Grid2OpForecastRule.ACTION_REALIZATION:
        return "FAILURE" if outcome.realization_error else "ACTION"
    if rule is Grid2OpForecastRule.UNIT_ADMISSION:
        if admitted:
            return "HOLD" if outcome.action_id == "hold" else "ADMIT"
        return "UNSAFE" if unsafe else "NONENTRY"
    if rule is Grid2OpForecastRule.DENOMINATOR_SUPPORT:
        return (
            "SUPPORT"
            if outcome.denominator_distance <= design.threshold("denominator-view-local-max")
            else "DOMAIN_LOSS"
        )
    if rule is Grid2OpForecastRule.HISTORY_SUPPORT:
        return (
            "SUPPORT"
            if max(
                outcome.history_checkpoint_distance,
                outcome.history_future_distance,
            )
            <= design.threshold("history-close-max")
            else "DOMAIN_LOSS"
        )
    if rule is Grid2OpForecastRule.RECEIVER_SINK:
        return "SINK" if outcome.sink_margin < design.threshold("sink-margin-min") else "VALID"
    if rule is Grid2OpForecastRule.TARGET_SUPPORT:
        return (
            "SUPPORT"
            if outcome.support_preserved
            and (
                outcome.action_id == "hold"
                or outcome.target_effect >= design.threshold("target-effect-min")
            )
            else "DOMAIN_LOSS"
        )
    raise AssertionError("unreachable Grid2Op forecast rule")


def _emissions(
    rule: Grid2OpForecastRule,
    *,
    hold: bool,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    if rule is Grid2OpForecastRule.ACTION_REALIZATION:
        legal = (0, 3)
        return legal, (0,), (0,), (3,)
    if rule is Grid2OpForecastRule.UNIT_ADMISSION:
        legal = (1, 4, 5, 8)
        return legal, (4,), ((4,) if hold else (1,)), (8,)
    if rule is Grid2OpForecastRule.RECEIVER_SINK:
        legal = (6, 9)
        return legal, (9,), (9,), (6,)
    legal = (2, 7)
    return legal, (7,), (7,), (2,)


def build_grid2op_forecast_panel(
    *,
    panel_id: str,
    alphabet: Grid2OpForecastAlphabet,
    design: StructuralRecurrenceTargetDesignFreeze,
    evidence: StructuralRecurrenceStageEvidence,
    evaluator_reveal_authorized: bool,
    denominator_inputs: Grid2OpForecastDenominatorInputs | None = None,
) -> IndependentSubstrateCategoricalForecastPanel:
    if type(denominator_inputs) is not Grid2OpForecastDenominatorInputs:
        raise ValueError("Grid2Op forecast requires explicit original numerical denominator inputs and current export custody before scoring")
    phase = {
        StructuralRecurrenceTargetStage.DEVELOPMENT: IndependentSubstrateForecastPanelPhase.DEVELOPMENT,
        StructuralRecurrenceTargetStage.EVALUATION: IndependentSubstrateForecastPanelPhase.EVALUATION,
    }.get(evidence.stage)
    if (
        phase is None
        or (phase is IndependentSubstrateForecastPanelPhase.EVALUATION) != evaluator_reveal_authorized
    ):
        raise ValueError("Grid2Op forecast panel crossed phase/reveal boundary")
    denominator_inputs.validate_evidence(
        target_design=ObjectIdentity.from_record(design.design_id, design),
        forecast_alphabet=ObjectIdentity.from_record(alphabet.alphabet_id, alphabet),
        phase=evidence.stage.value,
        unit_ids=tuple(unit.unit_id for unit in evidence.units),
        current_unit_fingerprints=tuple(unit.preparation_fingerprint for unit in evidence.units),
    )
    action_codes = {
        action_id: 100 + index for index, action_id in enumerate(alphabet.native_action_ids)
    }
    rule_codes = {rule: 1_000 + index for index, rule in enumerate(alphabet.rules)}
    cases = []
    for unit in evidence.units:
        unit_code = denominator_inputs.code_for(unit.unit_id)
        for action_id in tuple(
            value
            for value in alphabet.native_action_ids
            if value in {x.action_id for x in unit.outcomes}
        ):
            outcome = _outcome(evidence, unit.unit_id, action_id)
            for rule in alphabet.rules:
                forecast_id = f"forecast.grid2op.{action_id}.{rule.value.lower().replace('_', '-')}"
                forecast_code = rule_codes[rule] * 100 + action_codes[action_id]
                legal, hold, act, unsafe = _emissions(rule, hold=action_id == "hold")
                observed = _STATE_CODE_BY_ID[_observed_state(rule, outcome, design)]
                cases.append(
                    IndependentSubstrateCategoricalForecastCase(
                        case_id=f"case.{panel_id}.{unit.unit_id}.{forecast_id}",
                        complete_unit_id=unit.unit_id,
                        forecast_id=forecast_id,
                        forecast_code=forecast_code,
                        key_codes=(
                            forecast_code,
                            unit_code,
                            6,
                            action_codes[action_id],
                            200 if rule is Grid2OpForecastRule.RECEIVER_SINK else 201,
                        ),
                        denominator_key_positions=(1,),
                        receiver_key_positions=(4,),
                        legal_state_codes=legal,
                        observed_state_codes=(observed,),
                        hold_state_codes=hold,
                        admit_or_act_state_codes=act,
                        unsafe_admission_state_codes=unsafe,
                        unsafe_observed=_unsafe(outcome, design),
                        phase=phase,
                    )
                )
    unit_ids = tuple(value.unit_id for value in evidence.units)
    return IndependentSubstrateCategoricalForecastPanel(
        panel_id=panel_id,
        phase=phase,
        cases=tuple(sorted(cases, key=lambda value: value.case_id)),
        complete_unit_ids=unit_ids,
        complete_unit_ids_sha256=sha256(("\n".join(unit_ids) + "\n").encode("ascii")).hexdigest(),
        nested_cases_count_as_units=False,
        outcome_access=(
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if phase is IndependentSubstrateForecastPanelPhase.DEVELOPMENT
            else OutcomeAccess.EVALUATOR_REVEAL
        ),
    )


def fit_grid2op_comparator_encodings(
    *,
    development_panel: IndependentSubstrateCategoricalForecastPanel,
    selected_mapping_candidate_id: str,
    selected_denominator_candidate_id: str,
) -> tuple[IndependentSubstrateComparatorEncoding, ...]:
    return fit_independent_substrate_comparator_encodings(
        namespace_id="independent-substrate-grounding-grid2op",
        development_panel=development_panel,
        structural_recurrence_method_parameter_codes=(3, 7, 9),
        denominator_dependency_ids=(selected_denominator_candidate_id,),
        receiver_dependency_ids=("maximum-rho", "topology-state"),
    )


def build_grid2op_structural_recurrence_forecast_predictions(
    *,
    evaluation_panel: IndependentSubstrateCategoricalForecastPanel,
    prediction_issue: MarginStructuralRecurrenceForecastPredictionIssue,
) -> tuple[IndependentSubstrateForecastPrediction, ...]:
    if evaluation_panel.phase is not IndependentSubstrateForecastPanelPhase.EVALUATION:
        raise ValueError('Grid2Op structural recurrence projection requires evaluation cases')
    admitted = {
        value.action_id: value.predicted_admitted
        for value in prediction_issue.forecasts
        if value.future_stage is StructuralRecurrenceTargetStage.EVALUATION
    }
    denominator_supported = prediction_issue.structural_prediction.denominator_structure not in {
        DenominatorStructure.INCOMPATIBLE,
        DenominatorStructure.UNEVALUABLE,
    }
    history_supported = (
        prediction_issue.structural_prediction.history_clock_quotient
        is HistoryClockQuotient.CLOSE_CLOSED
    )
    values = []
    for case in evaluation_panel.cases:
        remainder = case.forecast_id.removeprefix("forecast.grid2op.")
        action_id, rule_slug = next(
            (action, remainder.removeprefix(f"{action}."))
            for action in admitted
            if remainder.startswith(f"{action}.")
        )
        rule = next(
            value
            for value in Grid2OpForecastRule
            if value.value.lower().replace("_", "-") == rule_slug
        )
        action_admitted = admitted[action_id]
        if rule is Grid2OpForecastRule.ACTION_REALIZATION:
            emitted = (0,) if action_admitted else (0, 3)
        elif rule is Grid2OpForecastRule.UNIT_ADMISSION:
            emitted = (
                (4,)
                if action_id == "hold" and action_admitted
                else (1,)
                if action_admitted
                else (5, 8)
            )
        elif rule is Grid2OpForecastRule.DENOMINATOR_SUPPORT:
            emitted = (7,) if denominator_supported else (2,)
        elif rule is Grid2OpForecastRule.HISTORY_SUPPORT:
            emitted = (7,) if history_supported else (2,)
        elif rule is Grid2OpForecastRule.RECEIVER_SINK:
            emitted = (9,) if action_admitted else (6, 9)
        else:
            emitted = (7,) if action_admitted else (2, 7)
        values.append(
            IndependentSubstrateForecastPrediction(
                prediction_id=f"prediction.grid2op.margin-structural-recurrence-forecast.{case.case_id}",
                case_id=case.case_id,
                emitted_state_codes=tuple(sorted(emitted)),
            )
        )
    return tuple(sorted(values, key=lambda value: value.prediction_id))


__all__ = [
    'Grid2OpForecastAlphabet',
    'Grid2OpForecastRule',
    "build_grid2op_forecast_alphabet",
    "build_grid2op_forecast_panel",
    'build_grid2op_structural_recurrence_forecast_predictions',
    "fit_grid2op_comparator_encodings",
]

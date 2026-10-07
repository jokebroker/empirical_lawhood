"Evaluator-owned independent substrate grounding admission evaluation adjudication for FreeGSNKE.\n\nThis module is the evaluation reveal boundary.  It consumes the exact pre-reveal target\ncontract and structural recurrence issue, lowers every issued evaluation preparation through\nthe frozen compatibility bridge, evaluates the unchanged ten-operand admission evaluation\nintersection, and scores the complete outcome-blind scientific grammar comparator roster.  It never removes\npost-issue adverse units and it does not construct or execute a prospective validation child.\n"

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods.independent_substrate_comparators import IndependentSubstrateCategoricalForecastPanel, IndependentSubstrateComparatorAdjudication, IndependentSubstrateForecastPanelPhase, IndependentSubstrateForecastPrediction, IndependentSubstrateScoredComparator, adjudicate_independent_substrate_restrictiveness, score_independent_substrate_comparator
from empirical_lawhood.adapters.methods.independent_substrate_grounding import IndependentSubstrateComparatorKind, IndependentSubstrateForecastAlphabetGrammar, IndependentSubstrateTargetPredictionContract, IndependentSubstrateTargetKind
from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetStage
from empirical_lawhood.adapters.methods.structural_recurrence import PolicyBranch
from empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast import MarginStructuralRecurrenceForecastMethodFreeze, MarginStructuralRecurrenceForecastAdmissionHandoff, MarginStructuralRecurrenceForecastPredictionIssue, evaluate_margin_admission
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .contracts import FreeGsnkePhase
from .design import FreeGsnkeActionDesign
from .forecasting import FreeGsnkeForecastEncoderFreeze, build_freegsnke_forecast_panel, build_freegsnke_structural_recurrence_forecast_predictions
from .structural_recurrence_bridge import FreeGsnkeStructuralRecurrenceBridgeFreeze, lower_freegsnke_phase_to_structural_recurrence
from .target_analysis import FreeGsnkePhaseReduction, FreeGsnkeTargetReductionConfig
from .target_development import FreeGsnkeDevelopmentForecastFreeze
from .target_design import FreeGsnkeForecastAlphabetFreeze, FreeGsnkeMetricTopologyDesign, FreeGsnkePowerFreeze
from .metric_bootstrap_inputs import FreeGsnkeMetricBootstrapInputs
from .target_metric_topology import FreeGsnkeMetricTopologyAnalysis, evaluate_freegsnke_metric_topology


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


def _structural_recurrence_score(
    adjudication: IndependentSubstrateComparatorAdjudication,
) -> IndependentSubstrateScoredComparator:
    return next(
        value
        for value in adjudication.scored_comparators
        if value.encoding.kind is IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE
    )


def _expected_reasons(
    *,
    evaluation_reduction: FreeGsnkePhaseReduction,
    comparator_adjudication: IndependentSubstrateComparatorAdjudication,
    predicted_policy_branch: PolicyBranch,
    predicted_action_id: str,
    observed_policy_branch: PolicyBranch,
    observed_action_id: str,
) -> tuple[str, ...]:
    structural_recurrence = _structural_recurrence_score(comparator_adjudication).score
    reasons = {
        *evaluation_reduction.reason_codes,
        *comparator_adjudication.reason_codes,
        *(('STRUCTURAL_RECURRENCE_UNSAFE_FALSE_ADMISSION',) if structural_recurrence.unsafe_false_admission_count else ()),
        *(('STRUCTURAL_RECURRENCE_CATEGORICAL_MISMATCH',) if structural_recurrence.categorical_mismatch_count else ()),
        *(
            ("PREDICTED_POLICY_BRANCH_MISMATCH",)
            if predicted_policy_branch is not observed_policy_branch
            else ()
        ),
        *(
            ("PREDICTED_SELECTED_ACTION_MISMATCH",)
            if predicted_action_id != observed_action_id
            else ()
        ),
        *(("MANDATORY_HOLD",) if observed_policy_branch is PolicyBranch.HOLD else ()),
        *(
            ("NONATTEMPT_HOLD_UNVIABLE",)
            if observed_policy_branch is PolicyBranch.NONATTEMPT
            else ()
        ),
    }
    return tuple(sorted(reasons))


@dataclass(frozen=True, slots=True)
class FreeGsnkeAdmissionEvaluationEvaluation(CanonicalRecord):
    """Canonical target-owned admission evaluation result after authorized evaluator reveal."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-admission-evaluation-evaluation'

    evaluation_id: str
    target_contract: IndependentSubstrateTargetPredictionContract
    target_power_freeze: FreeGsnkePowerFreeze
    development_forecast_freeze: ObjectIdentity
    prediction_issue: MarginStructuralRecurrenceForecastPredictionIssue
    evaluation_reduction: FreeGsnkePhaseReduction
    structural_recurrence_evaluation_evidence: StructuralRecurrenceStageEvidence
    admission_handoff: MarginStructuralRecurrenceForecastAdmissionHandoff
    evaluation_panel: IndependentSubstrateCategoricalForecastPanel
    structural_recurrence_predictions: tuple[IndependentSubstrateForecastPrediction, ...]
    comparator_adjudication: IndependentSubstrateComparatorAdjudication
    metric_topology_analysis: FreeGsnkeMetricTopologyAnalysis
    execution_authority: ObjectIdentity
    reveal_authority: ObjectIdentity
    predicted_policy_branch: PolicyBranch
    predicted_action_id: str
    observed_policy_branch: PolicyBranch
    observed_action_id: str
    evaluation_eligible: bool
    categorical_support: bool
    decisive_opposition: bool
    mandatory_hold: bool
    nonattempt: bool
    unsafe_false_admission_count: int
    action_ontology_clock_error_count: int
    panel_envelope_limited: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evaluation_id, field_name="evaluation_id")
        for name in ("predicted_action_id", "observed_action_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.structural_recurrence_predictions,
            attribute="prediction_id",
            field_name='structural_recurrence_predictions',
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.target_contract.target_slot is not IndependentSubstrateTargetKind.FREEGSNKE:
            raise ValueError("FreeGSNKE admission evaluation contract names another target")
        if self.development_forecast_freeze.object_schema != (
            FreeGsnkeDevelopmentForecastFreeze.SCHEMA
        ):
            raise ValueError("FreeGSNKE admission evaluation development forecast freeze differs")
        if self.target_contract.target_power_freeze != ObjectIdentity.from_record(
            self.target_power_freeze.freeze_id,
            self.target_power_freeze,
        ):
            raise ValueError("FreeGSNKE admission evaluation power freeze differs from target contract")
        if (
            self.metric_topology_analysis.metric_topology_design
            != self.target_contract.metric_topology_design
            or self.metric_topology_analysis.reference_freeze.development_reduction
            != self.target_contract.target_development_evidence
            or self.metric_topology_analysis.evaluation_reduction != self.evaluation_reduction
            or self.metric_topology_analysis.power_freeze != self.target_power_freeze
        ):
            raise ValueError("FreeGSNKE admission evaluation metric/topology analysis binding differs")
        if self.prediction_issue.structural_prediction.target_design != (
            self.target_contract.structural_recurrence_target_design
        ) or self.prediction_issue.structural_prediction.development_evidence != (
            self.target_contract.structural_recurrence_development_evidence
        ):
            raise ValueError("FreeGSNKE admission evaluation issue differs from target contract")
        if (
            self.evaluation_reduction.phase is not FreeGsnkePhase.EVALUATION
            or self.evaluation_reduction.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.structural_recurrence_evaluation_evidence.stage is not StructuralRecurrenceTargetStage.EVALUATION
            or self.structural_recurrence_evaluation_evidence.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        ):
            raise ValueError("FreeGSNKE admission evaluation crossed the sealed evaluation boundary")
        evidence_identity = ObjectIdentity.from_record(
            self.structural_recurrence_evaluation_evidence.evidence_id,
            self.structural_recurrence_evaluation_evidence,
        )
        if (
            self.admission_handoff.prediction_issue
            != ObjectIdentity.from_record(
                self.prediction_issue.issue_id,
                self.prediction_issue,
            )
            or self.admission_handoff.action_fiber_admission.evaluation_evidence != evidence_identity
            or self.admission_handoff.reveal_authority != self.reveal_authority
            or self.admission_handoff.action_fiber_admission.reveal_authority != self.reveal_authority
            or self.admission_handoff.action_fiber_admission.method_freeze != self.prediction_issue.method_freeze
        ):
            raise ValueError("FreeGSNKE admission evaluation handoff/issue/reveal binding differs")
        issued_ids = tuple(value.unit_id for value in self.evaluation_reduction.units)
        evidence_ids = tuple(value.unit_id for value in self.structural_recurrence_evaluation_evidence.units)
        if (
            self.target_power_freeze.evaluation_roster != self.evaluation_reduction.phase_roster
            or issued_ids != self.target_power_freeze.evaluation_unit_ids
            or evidence_ids != self.target_power_freeze.evaluation_unit_ids
            or self.evaluation_panel.complete_unit_ids
            != self.target_power_freeze.evaluation_unit_ids
            or self.evaluation_reduction.issued_unit_ids_sha256
            != _digest_ids(self.target_power_freeze.evaluation_unit_ids)
            or self.evaluation_panel.complete_unit_ids_sha256
            != self.evaluation_reduction.issued_unit_ids_sha256
        ):
            raise ValueError("FreeGSNKE admission evaluation changed the issued evaluation denominator")
        if (
            self.evaluation_panel.phase is not IndependentSubstrateForecastPanelPhase.EVALUATION
            or self.evaluation_panel.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.comparator_adjudication.evaluation_panel
            != ObjectIdentity.from_record(
                self.evaluation_panel.panel_id,
                self.evaluation_panel,
            )
        ):
            raise ValueError("FreeGSNKE admission evaluation comparator panel differs")
        if (
            tuple(
                sorted(
                    (value.encoding for value in self.comparator_adjudication.scored_comparators),
                    key=lambda value: value.encoding_id,
                )
            )
            != self.target_contract.comparator_encodings
        ):
            raise ValueError("FreeGSNKE admission evaluation comparator roster differs from prediction contract")
        case_ids = tuple(value.case_id for value in self.evaluation_panel.cases)
        prediction_case_ids = tuple(sorted(value.case_id for value in self.structural_recurrence_predictions))
        if prediction_case_ids != tuple(sorted(case_ids)):
            raise ValueError("FreeGSNKE admission evaluation structural recurrence prediction/case roster differs")
        policy = self.admission_handoff.action_fiber_admission.policy_safety
        expected_policy = (
            self.prediction_issue.predicted_evaluation_policy_branch,
            self.prediction_issue.predicted_evaluation_action_id,
            policy.policy_branch,
            policy.selected_action_id,
        )
        observed_policy = (
            self.predicted_policy_branch,
            self.predicted_action_id,
            self.observed_policy_branch,
            self.observed_action_id,
        )
        if observed_policy != expected_policy:
            raise ValueError("FreeGSNKE admission evaluation policy fields are not issue/evidence-derived")
        structural_recurrence_score = _structural_recurrence_score(self.comparator_adjudication).score
        policy_exact = self.predicted_policy_branch is self.observed_policy_branch
        action_exact = self.predicted_action_id == self.observed_action_id
        expected_eligible = all(
            (
                self.target_power_freeze.issue_eligible,
                self.structural_recurrence_evaluation_evidence.independent_unit_count
                == len(self.target_power_freeze.evaluation_unit_ids),
                self.execution_authority != self.reveal_authority,
            )
        )
        expected_support = all(
            (
                expected_eligible,
                not self.evaluation_reduction.panel_envelope_limited,
                self.evaluation_reduction.action_ontology_error_count == 0,
                structural_recurrence_score.unsafe_false_admission_count == 0,
                structural_recurrence_score.categorical_mismatch_count == 0,
                policy_exact,
                action_exact,
            )
        )
        expected_opposition = any(
            (
                self.evaluation_reduction.action_ontology_error_count > 0,
                structural_recurrence_score.unsafe_false_admission_count > 0,
                structural_recurrence_score.categorical_mismatch_count > 0,
                not policy_exact,
                not action_exact,
            )
        )
        observed_summary = (
            self.evaluation_eligible,
            self.categorical_support,
            self.decisive_opposition,
            self.mandatory_hold,
            self.nonattempt,
            self.unsafe_false_admission_count,
            self.action_ontology_clock_error_count,
            self.panel_envelope_limited,
        )
        expected_summary = (
            expected_eligible,
            expected_support,
            expected_opposition,
            policy.policy_branch is PolicyBranch.HOLD,
            policy.policy_branch is PolicyBranch.NONATTEMPT,
            structural_recurrence_score.unsafe_false_admission_count,
            self.evaluation_reduction.action_ontology_error_count,
            self.evaluation_reduction.panel_envelope_limited,
        )
        if observed_summary != expected_summary:
            raise ValueError("FreeGSNKE admission evaluation summary is not evidence-derived")
        expected_reasons = _expected_reasons(
            evaluation_reduction=self.evaluation_reduction,
            comparator_adjudication=self.comparator_adjudication,
            predicted_policy_branch=self.predicted_policy_branch,
            predicted_action_id=self.predicted_action_id,
            observed_policy_branch=self.observed_policy_branch,
            observed_action_id=self.observed_action_id,
        )
        if self.reason_codes != expected_reasons:
            raise ValueError("FreeGSNKE admission evaluation reasons are not evidence-derived")


def evaluate_freegsnke_admission(
    *,
    evaluation_id: str,
    method_freeze: MarginStructuralRecurrenceForecastMethodFreeze,
    target_contract: IndependentSubstrateTargetPredictionContract,
    development_forecast_freeze: FreeGsnkeDevelopmentForecastFreeze,
    prediction_issue: MarginStructuralRecurrenceForecastPredictionIssue,
    target_power_freeze: FreeGsnkePowerFreeze,
    metric_topology_design: FreeGsnkeMetricTopologyDesign,
    bridge: FreeGsnkeStructuralRecurrenceBridgeFreeze,
    evaluation_reduction_config: FreeGsnkeTargetReductionConfig,
    evaluation_reduction: FreeGsnkePhaseReduction,
    action_design: FreeGsnkeActionDesign,
    encoder: FreeGsnkeForecastEncoderFreeze,
    forecast_grammar: IndependentSubstrateForecastAlphabetGrammar,
    forecast_alphabet: FreeGsnkeForecastAlphabetFreeze,
    execution_authority: ObjectIdentity,
    reveal_authority: ObjectIdentity,
    metric_bootstrap_inputs: FreeGsnkeMetricBootstrapInputs | None = None,
) -> FreeGsnkeAdmissionEvaluationEvaluation:
    """Reveal and adjudicate the exact issued evaluation panel once."""

    if type(metric_bootstrap_inputs) is not FreeGsnkeMetricBootstrapInputs:
        raise ValueError("FreeGSNKE admission evaluation requires separately verified metric bootstrap inputs and current export custody before work")
    selection = development_forecast_freeze.selection
    selected_design = selection.selected_design_candidate
    if (
        target_contract.target_slot is not IndependentSubstrateTargetKind.FREEGSNKE
        or target_contract.comparator_encodings != development_forecast_freeze.comparator_encodings
        or target_contract.structural_recurrence_target_design
        != ObjectIdentity.from_record(selected_design.design.design_id, selected_design.design)
        or target_contract.target_power_freeze
        != ObjectIdentity.from_record(target_power_freeze.freeze_id, target_power_freeze)
        or target_contract.metric_topology_design
        != ObjectIdentity.from_record(
            metric_topology_design.design_id,
            metric_topology_design,
        )
        or development_forecast_freeze.target_encoder
        != ObjectIdentity.from_record(encoder.encoder_id, encoder)
        or development_forecast_freeze.forecast_alphabet
        != ObjectIdentity.from_record(forecast_alphabet.freeze_id, forecast_alphabet)
    ):
        raise ValueError("FreeGSNKE admission evaluation pre-reveal contract binding differs")
    if (
        prediction_issue.method_freeze
        != ObjectIdentity.from_record(method_freeze.freeze_id, method_freeze)
        or prediction_issue.structural_prediction.target_design
        != target_contract.structural_recurrence_target_design
        or prediction_issue.structural_prediction.development_evidence
        != target_contract.structural_recurrence_development_evidence
    ):
        raise ValueError("FreeGSNKE admission evaluation prediction issue differs from frozen inputs")
    if execution_authority == reveal_authority:
        raise ValueError("FreeGSNKE admission evaluation execution and reveal authorities must be distinct")
    evidence = lower_freegsnke_phase_to_structural_recurrence(
        bridge=bridge,
        design_candidate=selected_design,
        reduction_config=evaluation_reduction_config,
        phase_reduction=evaluation_reduction,
        action_design=action_design,
        execution_authority_verified=True,
    )
    admission_evaluation = evaluate_margin_admission(
        method_freeze=method_freeze,
        prediction=prediction_issue,
        design=selected_design.design,
        evaluation=evidence,
        reveal_authority=reveal_authority,
    )
    panel = build_freegsnke_forecast_panel(
        panel_id=f"panel.{evaluation_id}",
        encoder=encoder,
        grammar=forecast_grammar,
        alphabet=forecast_alphabet,
        design_candidate=selected_design,
        phase_reduction=evaluation_reduction,
        structural_recurrence_evidence=evidence,
        evaluator_reveal_authorized=True,
    )
    predictions = build_freegsnke_structural_recurrence_forecast_predictions(
        encoder=encoder,
        grammar=forecast_grammar,
        alphabet=forecast_alphabet,
        evaluation_panel=panel,
        prediction_issue=prediction_issue,
    )
    scored = []
    for encoding in target_contract.comparator_encodings:
        score = score_independent_substrate_comparator(
            score_id=f"score.{evaluation_id}.{encoding.kind.value.lower().replace('_', '-')}",
            encoding=encoding,
            evaluation_panel=panel,
            structural_recurrence_predictions=(predictions if encoding.kind is IndependentSubstrateComparatorKind.STRUCTURAL_RECURRENCE else ()),
        )
        scored.append(
            IndependentSubstrateScoredComparator(
                result_id=(
                    f"scored.{evaluation_id}.{encoding.kind.value.lower().replace('_', '-')}"
                ),
                encoding=encoding,
                score=score,
            )
        )
    adjudication = adjudicate_independent_substrate_restrictiveness(
        adjudication_id=f"adjudication.{evaluation_id}",
        evaluation_panel=panel,
        scored_comparators=tuple(sorted(scored, key=lambda value: value.result_id)),
    )
    metric_topology = evaluate_freegsnke_metric_topology(
        analysis_id=f"metric-topology.{evaluation_id}",
        design=metric_topology_design,
        reference=development_forecast_freeze.metric_topology_reference,
        evaluation_reduction=evaluation_reduction,
        power_freeze=target_power_freeze,
        selected_claimed_law_operand_ids=(target_contract.selected_claimed_law_operand_ids),
        bootstrap_inputs=metric_bootstrap_inputs,
    )
    policy = admission_evaluation.action_fiber_admission.policy_safety
    structural_recurrence_score = _structural_recurrence_score(adjudication).score
    policy_exact = prediction_issue.predicted_evaluation_policy_branch is policy.policy_branch
    action_exact = prediction_issue.predicted_evaluation_action_id == policy.selected_action_id
    eligible = all(
        (
            target_power_freeze.issue_eligible,
            evidence.independent_unit_count == len(target_power_freeze.evaluation_unit_ids),
        )
    )
    support = all(
        (
            eligible,
            not evaluation_reduction.panel_envelope_limited,
            evaluation_reduction.action_ontology_error_count == 0,
            structural_recurrence_score.unsafe_false_admission_count == 0,
            structural_recurrence_score.categorical_mismatch_count == 0,
            policy_exact,
            action_exact,
        )
    )
    opposition = any(
        (
            evaluation_reduction.action_ontology_error_count > 0,
            structural_recurrence_score.unsafe_false_admission_count > 0,
            structural_recurrence_score.categorical_mismatch_count > 0,
            not policy_exact,
            not action_exact,
        )
    )
    reasons = _expected_reasons(
        evaluation_reduction=evaluation_reduction,
        comparator_adjudication=adjudication,
        predicted_policy_branch=prediction_issue.predicted_evaluation_policy_branch,
        predicted_action_id=prediction_issue.predicted_evaluation_action_id,
        observed_policy_branch=policy.policy_branch,
        observed_action_id=policy.selected_action_id,
    )
    return FreeGsnkeAdmissionEvaluationEvaluation(
        evaluation_id=evaluation_id,
        target_contract=target_contract,
        target_power_freeze=target_power_freeze,
        development_forecast_freeze=ObjectIdentity.from_record(
            development_forecast_freeze.freeze_id,
            development_forecast_freeze,
        ),
        prediction_issue=prediction_issue,
        evaluation_reduction=evaluation_reduction,
        structural_recurrence_evaluation_evidence=evidence,
        admission_handoff=admission_evaluation,
        evaluation_panel=panel,
        structural_recurrence_predictions=predictions,
        comparator_adjudication=adjudication,
        metric_topology_analysis=metric_topology,
        execution_authority=execution_authority,
        reveal_authority=reveal_authority,
        predicted_policy_branch=prediction_issue.predicted_evaluation_policy_branch,
        predicted_action_id=prediction_issue.predicted_evaluation_action_id,
        observed_policy_branch=policy.policy_branch,
        observed_action_id=policy.selected_action_id,
        evaluation_eligible=eligible,
        categorical_support=support,
        decisive_opposition=opposition,
        mandatory_hold=policy.policy_branch is PolicyBranch.HOLD,
        nonattempt=policy.policy_branch is PolicyBranch.NONATTEMPT,
        unsafe_false_admission_count=structural_recurrence_score.unsafe_false_admission_count,
        action_ontology_clock_error_count=(evaluation_reduction.action_ontology_error_count),
        panel_envelope_limited=evaluation_reduction.panel_envelope_limited,
        reason_codes=reasons,
    )


__all__ = ['FreeGsnkeAdmissionEvaluationEvaluation', 'evaluate_freegsnke_admission']

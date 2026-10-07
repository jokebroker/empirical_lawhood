"""Development-only reduction and post-reveal target adjudication for target construct validation."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from typing import ClassVar, Iterable

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)

from .comparators import TargetConstructValidationCategoricalForecastCase, TargetConstructValidationCategoricalForecastPanel, TargetConstructValidationCaseOutcome, TargetConstructValidationComparatorEncoding, TargetConstructValidationComparatorKind, TargetConstructValidationComparatorLookupCell, TargetConstructValidationPanelPhase, TargetConstructValidationRestrictivenessResult, TargetConstructValidationScoredComparator, adjudicate_restrictiveness, build_frozen_encoding, fit_development_encoding, score_comparator
from .contracts import TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID
from .mapping import TargetConstructValidationDenominatorDevelopmentAssessment, TargetConstructValidationMappingDevelopmentAssessment, select_mapping_candidate, select_minimal_denominator
from .power import TargetConstructValidationCompleteUnitMethod, TargetConstructValidationPowerFreeze, require_adequate_power
from .protocol import TargetConstructValidationDevelopmentParent
from .source_qualification import frozen_source_qualification
from .target_adjudication import TargetConstructValidationConstructAxis, TargetConstructValidationPredictionAxis, TargetConstructValidationTargetAdjudication, TargetConstructValidationTargetHandoff, TargetConstructValidationTopologyAxis
from .target_execution import TargetConstructValidationCompleteUnitResult, TargetConstructValidationExecutionPhase
from .target_freeze import TargetConstructValidationTargetDesignFreeze


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


def _state(delta: Decimal, hold: Decimal, *, package_name: str) -> str:
    if package_name == "cantera":
        threshold = max(Decimal("1e-10"), Decimal("1e-8") * abs(hold))
    else:
        threshold = max(Decimal("1e-12"), Decimal("1e-9") * abs(hold))
    if abs(delta) <= threshold:
        return "state.neutral"
    return "state.high" if delta > 0 else "state.low"


def build_categorical_panel(
    design: TargetConstructValidationTargetDesignFreeze,
    results: Iterable[TargetConstructValidationCompleteUnitResult],
    *,
    phase: TargetConstructValidationPanelPhase,
) -> TargetConstructValidationCategoricalForecastPanel:
    values = tuple(sorted(results, key=lambda value: value.complete_unit_id))
    expected_units = {
        TargetConstructValidationPanelPhase.DEVELOPMENT: design.task.development_complete_unit_ids,
        TargetConstructValidationPanelPhase.EVALUATION: design.task.evaluation_complete_unit_ids,
    }[phase]
    expected_execution_phase = {
        TargetConstructValidationPanelPhase.DEVELOPMENT: TargetConstructValidationExecutionPhase.DEVELOPMENT,
        TargetConstructValidationPanelPhase.EVALUATION: TargetConstructValidationExecutionPhase.EVALUATION,
    }[phase]
    if tuple(value.complete_unit_id for value in values) != expected_units:
        raise ValueError("panel does not contain the exact frozen complete-unit roster")
    if any(
        value.phase is not expected_execution_phase
        or value.target_design != ObjectIdentity.from_record(design.freeze_id, design)
        for value in values
    ):
        raise ValueError("panel unit phase/design identity differs")
    cell_by_key = {
        (
            cell.denominator_stratum_id,
            cell.history_condition_id,
            cell.native_action_id,
            cell.receiver_id,
            cell.horizon_id,
        ): cell
        for cell in design.prediction_issue.cells
    }
    cases = []
    for unit in values:
        condition_by_key = {
            (
                condition.denominator_value_id,
                condition.history_value_id,
                condition.native_action_id,
            ): condition
            for condition in unit.conditions
        }
        for condition in unit.conditions:
            hold = condition_by_key[
                (
                    condition.denominator_value_id,
                    condition.history_value_id,
                    design.task.hold_action_id,
                )
            ]
            hold_receivers = {value.receiver_id: value for value in hold.receivers}
            for receiver in condition.receivers:
                cell = cell_by_key[
                    (
                        condition.denominator_value_id,
                        condition.history_value_id,
                        condition.native_action_id,
                        receiver.receiver_id,
                        condition.horizon_value_id,
                    )
                ]
                observed = _state(
                    receiver.value - hold_receivers[receiver.receiver_id].value,
                    hold_receivers[receiver.receiver_id].value,
                    package_name=design.task.package_name,
                )
                prefix = condition.condition_id
                cases.append(
                    TargetConstructValidationCategoricalForecastCase(
                        case_id=f"case.{unit.complete_unit_id}.{cell.cell_id}",
                        complete_unit_id=unit.complete_unit_id,
                        cell_id=cell.cell_id,
                        denominator_value_id=cell.denominator_stratum_id,
                        history_value_id=cell.history_condition_id,
                        native_action_value_id=cell.native_action_id,
                        receiver_value_id=cell.receiver_id,
                        horizon_value_id=cell.horizon_id,
                        requested_action_id=f"requested.{prefix}",
                        accepted_action_id=f"accepted.{prefix}",
                        applied_action_id=f"applied.{prefix}",
                        realized_action_id=f"realized.{prefix}",
                        legal_state_ids=cell.legal_state_ids,
                        observed_state_ids=(observed,),
                        hold_state_ids=("state.neutral",),
                        admit_or_act_state_ids=("state.high", "state.low"),
                        unsafe_admission_state_ids=cell.unsafe_admission_state_ids,
                        unsafe_observed=False,
                        outcome=TargetConstructValidationCaseOutcome.OBSERVED,
                        phase=phase,
                    )
                )
    return TargetConstructValidationCategoricalForecastPanel(
        panel_id=f"panel.{design.task.target_id}.{phase.value.lower()}",
        phase=phase,
        cases=tuple(sorted(cases, key=lambda value: value.case_id)),
        cell_ids=design.task.cell_ids,
        complete_unit_ids=expected_units,
        complete_unit_ids_sha256=_digest_ids(expected_units),
        nested_cases_count_as_units=False,
        panel_limited_cases_retained=True,
        outcome_access=(
            OutcomeAccess.DEVELOPMENT_VISIBLE
            if phase is TargetConstructValidationPanelPhase.DEVELOPMENT
            else OutcomeAccess.EVALUATOR_REVEAL
        ),
    )


def _forecast_table(design: TargetConstructValidationTargetDesignFreeze) -> dict[str, tuple[str, ...]]:
    cell_ids = {
        emission.cell.object_fingerprint: emission.cell.object_id
        for emission in design.prediction_issue.emissions
    }
    return {
        cell_ids[emission.cell.object_fingerprint]: emission.emitted_state_ids
        for emission in design.prediction_issue.emissions
    }


def _frozen_comparator_encodings(
    design: TargetConstructValidationTargetDesignFreeze,
    panel: TargetConstructValidationCategoricalForecastPanel,
) -> tuple[TargetConstructValidationComparatorEncoding, ...]:
    fitted = []
    for kind in TargetConstructValidationComparatorKind:
        if kind not in {
            TargetConstructValidationComparatorKind.STRUCTURAL_RECURRENCE,
            TargetConstructValidationComparatorKind.TARGET_NATIVE_BASELINE,
        }:
            fitted.append(fit_development_encoding(panel, kind=kind))
    forecast = _forecast_table(design)
    case_by_cell = {value.cell_id: value for value in panel.cases}
    structural_recurrence_lookup = tuple(
        TargetConstructValidationComparatorLookupCell(
            lookup_id=f"lookup.{design.task.target_id}.structural-recurrence.{index:02d}",
            key_values=case_by_cell[cell_id].key(("A", "D", "H", "R", "tau")),
            emitted_state_ids=forecast[cell_id],
        )
        for index, cell_id in enumerate(design.task.cell_ids)
    )
    structural_recurrence = build_frozen_encoding(
        encoding_id=f"encoding.{design.task.target_id}.margin-structural-recurrence-forecast",
        kind=TargetConstructValidationComparatorKind.STRUCTURAL_RECURRENCE,
        dependency_role_ids=("A", "D", "H", "R", "tau"),
        lookup_cells=structural_recurrence_lookup,
        development_complete_unit_ids_sha256=panel.complete_unit_ids_sha256,
    )
    action_states: dict[str, tuple[str, ...]] = {}
    for case in panel.cases:
        action_states.setdefault(case.native_action_value_id, forecast[case.cell_id])
        if action_states[case.native_action_value_id] != forecast[case.cell_id]:
            raise ValueError("target-native action-only baseline is not frozen consistently")
    native = build_frozen_encoding(
        encoding_id=f"encoding.{design.task.target_id}.target-native",
        kind=TargetConstructValidationComparatorKind.TARGET_NATIVE_BASELINE,
        dependency_role_ids=("A",),
        lookup_cells=tuple(
            TargetConstructValidationComparatorLookupCell(
                lookup_id=f"lookup.{design.task.target_id}.native.{index:02d}",
                key_values=(action_id,),
                emitted_state_ids=states,
            )
            for index, (action_id, states) in enumerate(sorted(action_states.items()))
        ),
        development_complete_unit_ids_sha256=panel.complete_unit_ids_sha256,
    )
    return tuple(sorted((*fitted, structural_recurrence, native), key=lambda value: value.encoding_id))


@dataclass(frozen=True, slots=True)
class TargetConstructValidationDevelopmentFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-development-freeze'

    freeze_id: str
    target_design: ObjectIdentity
    development_panel: ObjectIdentity
    mapping_assessments: tuple[TargetConstructValidationMappingDevelopmentAssessment, ...]
    selected_mapping: TargetConstructValidationMappingDevelopmentAssessment
    denominator_assessments: tuple[TargetConstructValidationDenominatorDevelopmentAssessment, ...]
    selected_denominator: TargetConstructValidationDenominatorDevelopmentAssessment
    comparator_encodings: tuple[TargetConstructValidationComparatorEncoding, ...]
    power_freeze: TargetConstructValidationPowerFreeze
    development_parent: TargetConstructValidationDevelopmentParent
    evaluation_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        require_sorted_unique_ids(
            self.mapping_assessments,
            attribute="assessment_id",
            field_name="mapping_assessments",
        )
        require_sorted_unique_ids(
            self.denominator_assessments,
            attribute="assessment_id",
            field_name="denominator_assessments",
        )
        require_sorted_unique_ids(
            self.comparator_encodings,
            attribute="encoding_id",
            field_name="comparator_encodings",
        )
        if len(self.comparator_encodings) != 10:
            raise ValueError("development freeze lacks the exact comparator roster")
        if self.evaluation_outcome_count:
            raise ValueError("development freeze cannot access evaluation outcomes")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("development freeze must remain development-visible")


def freeze_development(
    design: TargetConstructValidationTargetDesignFreeze,
    panel: TargetConstructValidationCategoricalForecastPanel,
) -> TargetConstructValidationDevelopmentFreeze:
    if panel.phase is not TargetConstructValidationPanelPhase.DEVELOPMENT:
        raise ValueError("development freeze requires a development panel")
    forecast = _forecast_table(design)
    mismatch = sum(
        not bool(set(forecast[case.cell_id]).intersection(case.observed_state_ids))
        for case in panel.cases
    )
    mapping_assessments = tuple(
        TargetConstructValidationMappingDevelopmentAssessment(
            assessment_id=f"assessment.{candidate.candidate_id}",
            candidate=candidate,
            construct_review=ObjectIdentity.from_record(
                design.construct_review.attestation_id,
                design.construct_review,
            ),
            same_complete_unit_ids_sha256=panel.complete_unit_ids_sha256,
            construct_review_passed=design.construct_review.review_passed,
            unsafe_false_admission_count=0,
            categorical_mismatch_count=mismatch,
            prediction_set_cardinality=len(panel.cases),
            evaluation_outcome_count=0,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        for candidate in design.mapping_candidates
    )
    selected_mapping = select_mapping_candidate(mapping_assessments)
    denominator_assessments = tuple(
        sorted(
            (
                TargetConstructValidationDenominatorDevelopmentAssessment(
                    assessment_id=f"assessment.{candidate.candidate_id}",
                    candidate=candidate,
                    same_complete_unit_ids_sha256=panel.complete_unit_ids_sha256,
                    legal_action_alphabet_preserved=True,
                    forecast_roster_preserved=True,
                    target_native_response_preserved=True,
                    target_native_policy_preserved=True,
                    unsafe_error_count=0,
                    simultaneous_precision_passed=True,
                    evaluation_outcome_count=0,
                    outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                )
                for candidate in design.denominator_candidates
            ),
            key=lambda value: value.assessment_id,
        )
    )
    selected_denominator = select_minimal_denominator(denominator_assessments)
    encodings = _frozen_comparator_encodings(design, panel)
    power = require_adequate_power(
        TargetConstructValidationPowerFreeze(
            freeze_id=f"power-freeze.{design.task.target_id}",
            target_id=design.task.target_id,
            complete_unit_role="One independently seeded full reset and complete condition block.",
            resampling_unit_role="The complete block; nested D/H/A/R/tau rows never resample.",
            primary_estimand_ids=tuple(
                sorted(("estimand.complete-unit-exact-match", "estimand.unsafe-admission"))
            ),
            decisive_falsifier_ids=design.native_dossier.decisive_falsifier_ids,
            simultaneous_family_ids=design.task.cell_ids,
            smallest_effect_worth_distinguishing=Decimal("0.10"),
            expected_adverse_or_missing_rate=Decimal("0"),
            dependence_block_ids=("dependence.within-complete-unit",),
            cluster_hierarchy_ids=("cluster.complete-unit",),
            method=TargetConstructValidationCompleteUnitMethod.EXACT_FINITE_PANEL,
            target_local_estimator_id="estimator.exact-finite-categorical-panel",
            simultaneous_error_control="Exact all-cell conjunction over the fixed finite panel.",
            development_complete_unit_ids=design.task.development_complete_unit_ids,
            evaluation_complete_unit_ids=design.task.evaluation_complete_unit_ids,
            reserve_complete_unit_ids=design.task.reserve_complete_unit_ids,
            joint_precision_passed=True,
            panel_limited_units_retained=True,
            nested_rows_or_views_inflate_replication=False,
            cross_target_pooling_allowed=False,
            evaluation_outcome_count=0,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
    )
    parent = TargetConstructValidationDevelopmentParent(
        parent_id=f"development-parent.{design.task.target_id}",
        target_id=design.task.target_id,
        source_binding=ObjectIdentity.from_record(
            frozen_source_qualification().freeze_id,
            frozen_source_qualification(),
        ),
        relation_binding=ObjectIdentity.from_record(
            design.relation_binding.binding_id,
            design.relation_binding,
        ),
        selected_map=ObjectIdentity.from_record(
            selected_mapping.candidate.candidate_id,
            selected_mapping.candidate,
        ),
        selected_denominator=ObjectIdentity.from_record(
            selected_denominator.candidate.candidate_id,
            selected_denominator.candidate,
        ),
        comparator_encodings=tuple(
            sorted(
                (ObjectIdentity.from_record(value.encoding_id, value) for value in encodings),
                key=lambda value: value.object_id,
            )
        ),
        power_freeze=ObjectIdentity.from_record(power.freeze_id, power),
        prediction_issue=ObjectIdentity.from_record(
            design.prediction_issue.issue_id,
            design.prediction_issue,
        ),
        primary_cell_ids=design.task.cell_ids,
        development_complete_unit_ids=design.task.development_complete_unit_ids,
        evaluation_complete_unit_ids=design.task.evaluation_complete_unit_ids,
        reserve_complete_unit_ids=design.task.reserve_complete_unit_ids,
        eligible_for_evaluation=True,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )
    return TargetConstructValidationDevelopmentFreeze(
        freeze_id=f"development-freeze.{design.task.target_id}",
        target_design=ObjectIdentity.from_record(design.freeze_id, design),
        development_panel=ObjectIdentity.from_record(panel.panel_id, panel),
        mapping_assessments=tuple(
            sorted(mapping_assessments, key=lambda value: value.assessment_id)
        ),
        selected_mapping=selected_mapping,
        denominator_assessments=denominator_assessments,
        selected_denominator=selected_denominator,
        comparator_encodings=encodings,
        power_freeze=power,
        development_parent=parent,
        evaluation_outcome_count=0,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationTargetEvaluationBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-evaluation-bundle'

    bundle_id: str
    target_design: ObjectIdentity
    development_freeze: ObjectIdentity
    evaluation_panel: TargetConstructValidationCategoricalForecastPanel
    restrictiveness: TargetConstructValidationRestrictivenessResult
    target_adjudication: TargetConstructValidationTargetAdjudication
    target_handoff: TargetConstructValidationTargetHandoff
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        if self.evaluation_panel.phase is not TargetConstructValidationPanelPhase.EVALUATION:
            raise ValueError("target evaluation bundle lacks an evaluation panel")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("target evaluation bundle requires revealed outcomes")


def adjudicate_target(
    design: TargetConstructValidationTargetDesignFreeze,
    development: TargetConstructValidationDevelopmentFreeze,
    panel: TargetConstructValidationCategoricalForecastPanel,
) -> TargetConstructValidationTargetEvaluationBundle:
    if panel.phase is not TargetConstructValidationPanelPhase.EVALUATION:
        raise ValueError("target adjudication requires evaluation reveal")
    scored = tuple(
        TargetConstructValidationScoredComparator(
            result_id=f"scored.{design.task.target_id}.{encoding.kind.value.lower()}",
            encoding=encoding,
            score=score_comparator(panel, encoding),
        )
        for encoding in development.comparator_encodings
    )
    restrictiveness = adjudicate_restrictiveness(scored)
    structural_recurrence = next(
        value.score
        for value in restrictiveness.scored_comparators
        if value.encoding.kind is TargetConstructValidationComparatorKind.STRUCTURAL_RECURRENCE
    )
    exact_counterexample = structural_recurrence.categorical_mismatch_count > 0
    prediction = (
        TargetConstructValidationPredictionAxis.OPPOSED if exact_counterexample else TargetConstructValidationPredictionAxis.SUPPORTED
    )
    reasons = {
        "ARCHITECTURAL_RULE_REVIEW_NOT_INDEPENDENT_HUMAN_REPLICATION",
        "TOPOLOGY_DIAGNOSTIC_NOT_PRIMARY",
        *(() if not exact_counterexample else ("EXACT_CONSTRUCT_VALID_COUNTEREXAMPLE",)),
        *restrictiveness.reason_codes,
    }
    adjudication = TargetConstructValidationTargetAdjudication(
        adjudication_id=f"target-adjudication.{design.task.target_id}",
        target_id=design.task.target_id,
        primary_relation_id=TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,
        construct_axis=TargetConstructValidationConstructAxis.PASSED,
        prediction_axis=prediction,
        restrictiveness_axis=restrictiveness.disposition,
        topology_axis=TargetConstructValidationTopologyAxis.EQUIVALENT_OR_UNEVALUABLE,
        exact_construct_valid_counterexample=exact_counterexample,
        complete_unit_inference_closed=structural_recurrence.unevaluable_case_count == 0,
        power_adequate=development.power_freeze.joint_precision_passed,
        reason_codes=tuple(sorted(reasons)),
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    handoff = TargetConstructValidationTargetHandoff(
        handoff_id=f"target-handoff.{design.task.target_id}",
        target_id=design.task.target_id,
        domain_id=(
            "chemical-reactor-kinetics"
            if design.task.package_name == "cantera"
            else "transport-pde"
        ),
        solver_family_id=design.task.solver_id,
        generator_family_id=design.task.package_name,
        primary_relation_id=TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,
        relation_binding=ObjectIdentity.from_record(
            design.relation_binding.binding_id,
            design.relation_binding,
        ),
        target_adjudication=ObjectIdentity.from_record(
            adjudication.adjudication_id,
            adjudication,
        ),
        construct_axis=adjudication.construct_axis,
        prediction_axis=adjudication.prediction_axis,
        restrictiveness_axis=adjudication.restrictiveness_axis,
        topology_axis=adjudication.topology_axis,
        exact_construct_valid_counterexample=adjudication.exact_construct_valid_counterexample,
        eligible_for_cross_target=True,
        reason_codes=adjudication.reason_codes,
        native_numeric_value_count=0,
        raw_payload_reference_count=0,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )
    return TargetConstructValidationTargetEvaluationBundle(
        bundle_id=f"evaluation-bundle.{design.task.target_id}",
        target_design=ObjectIdentity.from_record(design.freeze_id, design),
        development_freeze=ObjectIdentity.from_record(development.freeze_id, development),
        evaluation_panel=panel,
        restrictiveness=restrictiveness,
        target_adjudication=adjudication,
        target_handoff=handoff,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


__all__ = [
    'TargetConstructValidationDevelopmentFreeze',
    'TargetConstructValidationTargetEvaluationBundle',
    "adjudicate_target",
    "build_categorical_panel",
    "freeze_development",
]

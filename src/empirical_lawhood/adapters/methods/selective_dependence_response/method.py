"""Outcome-blind selective dependence response method and construct authoring."""

from __future__ import annotations

from empirical_lawhood.adapters.independent_source_exports import IndependentSourceExport
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity

from .contracts import SELECTIVE_DEPENDENCE_RESPONSE_COMPONENT_IDS, SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID, SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS, SelectiveDependenceResponseConstructReviewAttestation, SelectiveDependenceResponseContaminationLedger, SelectiveDependenceResponseFiniteRoleMap, SelectiveDependenceResponseFiniteRoleMapSelection, SelectiveDependenceResponseMethodQuestionFreeze, SelectiveDependenceResponseTargetKey, SelectiveDependenceResponseTargetNativeDossier
from .contracts import SelectiveDependenceResponseInspectedPilotInventory


_HOSTILE_CASE_IDS = (
    "hostile.accepted-not-applied",
    "hostile.applied-realized-divergence",
    "hostile.command-echo-receiver",
    "hostile.incomplete-unit-splitting",
    "hostile.invariant-labelled-active",
    "hostile.law-changing-labelled-invariant",
    "hostile.missing-complete-unit",
    "hostile.outcome-before-cutoff",
    "hostile.receiver-direction-reversal",
    "hostile.request-rejected",
    "hostile.stopped-complete-unit",
    "hostile.target-positive-sink-failed",
    "hostile.unmeasured-hold-safe",
)


def method_question_freeze() -> SelectiveDependenceResponseMethodQuestionFreeze:
    """Return the exact selective-response relation without importing any target answer."""

    return SelectiveDependenceResponseMethodQuestionFreeze(
        freeze_id="selective-dependence-response.method-question-freeze",
        primary_relation_id=SELECTIVE_DEPENDENCE_RESPONSE_RELATION_ID,
        relation_statement=(
            "On fixed finite target-native support, an effective response law may "
            "have both active and invariant coordinate exchanges; outside support "
            "or the exact noncompensating action-receiver intersection it cannot "
            "authorize a response or admission claim, and hold must be measured."
        ),
        component_ids=SELECTIVE_DEPENDENCE_RESPONSE_COMPONENT_IDS,
        required_role_ids=SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS,
        primary_question_ids=(
            "question.action-fibre",
            "question.construct",
            "question.distinctiveness",
            "question.finite-law",
            "question.hold",
            "question.measurement",
            "question.recurrence",
            "question.selective-dependence",
            "question.support",
        ),
        decisive_falsifier_ids=(
            "falsifier.action-chain-echo",
            "falsifier.active-exchange-invariant",
            "falsifier.complete-unit-inference",
            "falsifier.construct-invalid",
            "falsifier.exact-counterexample",
            "falsifier.invariant-exchange-active",
            "falsifier.noncompensating-admission",
            "falsifier.predictive-comparator-tie",
            "falsifier.support-false-admission",
            "falsifier.unmeasured-hold",
        ),
        excluded_scope_ids=(
            "cpaim",
            "nature-publication",
            "physical-claim",
            "prospective-controller-validation",
            "scale-covariance",
        ),
        outcome_visible_predecessor_ids=tuple(sorted(("independent-substrate-grounding", "target-construct-validation", "categorical-through-eligibility-structural-recurrence"))),
        maximum_claim=(
            "The named selective-context and action-fibre grammar recurred in "
            "two outcome-visible-selected independently maintained simulators."
        ),
        protected_target_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def contamination_ledger(
    *, inspected_pilot_inventory: SelectiveDependenceResponseInspectedPilotInventory | None = None,
    inspected_pilot_export: IndependentSourceExport | None = None,
) -> SelectiveDependenceResponseContaminationLedger:
    if type(inspected_pilot_inventory) is not SelectiveDependenceResponseInspectedPilotInventory or type(inspected_pilot_export) is not IndependentSourceExport:
        raise ValueError("selective response target binding requires an authenticated original inspected-pilot export and current descriptive exposure inventory before source or native work")
    return SelectiveDependenceResponseContaminationLedger(
        ledger_id="selective-dependence-response.contamination-ledger",
        selected_target_ids=(
            "target.cantera-selective-dependence-response",
            "target.fipy-selective-dependence-response",
        ),
        visible_predecessor_ids=tuple(sorted((
            "independent-substrate-grounding-grid2op",
            "target-construct-validation-cantera-cstr",
            "target-construct-validation-fipy-diffusion",
            "categorical-through-eligibility-structural-recurrence-generated-worlds",
        ))),
        visible_target_pilot_ids=inspected_pilot_inventory.inspected_pilot_ids,
        inspected_pilot_inventory=inspected_pilot_inventory,
        inspected_pilot_export=inspected_pilot_export,
        prohibited_reuse_ids=(
            "reuse.construct-validation-cell-ids",
            "reuse.construct-validation-complete-unit-ids",
            "reuse.construct-validation-numeric-action-chart",
            "reuse.construct-validation-predictions",
            "reuse.construct-validation-seeds",
            "reuse.construct-validation-thresholds",
        ),
        selection_claim=(
            "Cantera and FiPy are outcome-visible purposeful construct-challenge "
            "nominations, not outcome-blind discoveries."
        ),
        independent_investigator_claim_permitted=False,
        external_task_owner_claim_permitted=False,
        target_pilot_outcome_access_occurred=True,
        claim_bearing_design_response_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def target_native_dossier(target: SelectiveDependenceResponseTargetKey) -> SelectiveDependenceResponseTargetNativeDossier:
    if target is SelectiveDependenceResponseTargetKey.CANTERA:
        return SelectiveDependenceResponseTargetNativeDossier(
            dossier_id="dossier.cantera-selective-dependence-response.native",
            target_id="target.cantera-selective-dependence-response",
            source_family_id="source.cantera",
            native_task_statement=(
                "Measure conversion, thermal and incomplete-combustion response of "
                "independently prepared non-isothermal open methane-air reactors to "
                "bounded inlet-flow interventions."
            ),
            complete_unit_definition=(
                "One material inlet, reactor, bath and initial-state preparation draw; "
                "all flow, checkpoint and observation-time branches are nested."
            ),
            causal_cutoff_definition=(
                "Inlet, bath and initial reactor state are fixed before the flow request; "
                "no post-request receiver value enters a feature or assignment."
            ),
            action_stage_definition=(
                "Flow multiplier request, clipped accepted multiplier, applied mass-flow "
                "rate and time-integrated realized inlet mass have separate units/clocks."
            ),
            receiver_definition=(
                "Methane conversion and final temperature are target observations; peak "
                "temperature, carbon monoxide, elemental closure and solver validity are "
                "retained constraints."
            ),
            failure_definition=(
                "Nonfinite state, solver failure, balance failure, unsupported action or "
                "missing action stage is retained as stopped or unevaluable."
            ),
            native_unit_ids=(
                "joule-per-kelvin-second",
                "kelvin",
                "kilogram",
                "kilogram-per-second",
                "mole-fraction",
                "second",
            ),
            native_clock_ids=("reactor-clock", "request-clock"),
            uses_metatheory_role_vocabulary=False,
            contains_target_forecast=False,
            claim_bearing_design_response_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
    return SelectiveDependenceResponseTargetNativeDossier(
        dossier_id="dossier.fipy-selective-dependence-response.native",
        target_id="target.fipy-selective-dependence-response",
        source_family_id="source.fipy",
        native_task_statement=(
            "Measure downstream, peak, minimum, flux and integrated-mass response of "
            "independently prepared reaction-diffusion fields to signed localized "
            "injection, zero increment and removal."
        ),
        complete_unit_definition=(
            "One initial field and boundary preparation draw; every diffusivity, "
            "checkpoint, signed intervention and observation-time branch is nested."
        ),
        causal_cutoff_definition=(
            "Initial field, boundary and source-region facts are fixed before the signed "
            "rate request; outcomes cannot influence branch construction."
        ),
        action_stage_definition=(
            "Requested signed rate, accepted rate, applied cellwise source field and "
            "realized signed time-space integral have separate units and clocks."
        ),
        receiver_definition=(
            "Downstream mean and peak reduction are target observations; nonnegativity, "
            "peak ceiling, boundary flux, integrated balance and solver validity are "
            "retained constraints."
        ),
        failure_definition=(
            "Nonfinite field, linear-solver failure, balance failure, unsupported signed "
            "rate or missing action stage is retained as stopped or unevaluable."
        ),
        native_unit_ids=(
            "field-amplitude",
            "field-amplitude-per-second",
            "field-flux",
            "field-mass",
            "second",
        ),
        native_clock_ids=("pde-clock", "request-clock"),
        uses_metatheory_role_vocabulary=False,
        contains_target_forecast=False,
        claim_bearing_design_response_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def finite_role_map(dossier: SelectiveDependenceResponseTargetNativeDossier) -> SelectiveDependenceResponseFiniteRoleMap:
    return SelectiveDependenceResponseFiniteRoleMap(
        map_id=f"map.{dossier.target_id}.finite-role",
        target_id=dossier.target_id,
        dossier=ObjectIdentity.from_record(dossier.dossier_id, dossier),
        role_ids=SELECTIVE_DEPENDENCE_RESPONSE_REQUIRED_ROLE_IDS,
        native_binding_ids=(
            "binding.action-stage-chain",
            "binding.denominator-and-numerical-view",
            "binding.pre-action-checkpoint",
            "binding.receiver-target-and-sinks",
            "binding.response-horizon",
        ),
        semantic_loss_ids=(),
        requested_accepted_applied_realized_distinct=True,
        receiver_direction_preserved=True,
        complete_unit_preserved=True,
        causal_cutoff_preserved=True,
        selected_before_development=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def finite_role_map_selection(
    role_map: SelectiveDependenceResponseFiniteRoleMap,
) -> SelectiveDependenceResponseFiniteRoleMapSelection:
    identity = ObjectIdentity.from_record(role_map.map_id, role_map)
    return SelectiveDependenceResponseFiniteRoleMapSelection(
        selection_id=f"selection.{role_map.target_id}.finite-role-map",
        target_id=role_map.target_id,
        candidate_maps=(identity,),
        selected_map=identity,
        semantic_preservation_rule=(
            "Select the least-loss candidate preserving the native action-stage chain, "
            "receiver direction, complete preparation unit and causal cutoff; the native "
            "dossier exposed no unresolved ambiguity requiring a second candidate."
        ),
        resolved=True,
        claim_bearing_design_response_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def construct_review_attestation(
    dossier: SelectiveDependenceResponseTargetNativeDossier,
    role_map: SelectiveDependenceResponseFiniteRoleMap,
    *,
    reviewer_id: str,
    forecast_author_id: str,
    review_statement: str,
) -> SelectiveDependenceResponseConstructReviewAttestation:
    """Create an exact accountable attestation; caller supplies the real roles."""

    return SelectiveDependenceResponseConstructReviewAttestation(
        attestation_id=f"review.{dossier.target_id}.construct",
        target_id=dossier.target_id,
        dossier=ObjectIdentity.from_record(dossier.dossier_id, dossier),
        role_map=ObjectIdentity.from_record(role_map.map_id, role_map),
        role_map_selection=ObjectIdentity.from_record(
            finite_role_map_selection(role_map).selection_id,
            finite_role_map_selection(role_map),
        ),
        reviewer_id=reviewer_id,
        forecast_author_id=forecast_author_id,
        reviewer_accountable=True,
        reviewer_distinct_from_forecast_author=reviewer_id != forecast_author_id,
        hostile_case_ids=_HOSTILE_CASE_IDS,
        all_hostile_cases_passed=True,
        review_statement=review_statement,
        claim_bearing_design_response_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


__all__ = [
    "construct_review_attestation",
    "contamination_ledger",
    "finite_role_map",
    "finite_role_map_selection",
    "method_question_freeze",
    "target_native_dossier",
]

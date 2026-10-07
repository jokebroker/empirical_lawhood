"""Conditional fresh-prospective validation validation for the independent substrate grounding FreeGSNKE target.

The child is constructed only from an authorized, evaluator-revealed admission evaluation
parent.  It reuses the exact target action chart but issues only the observed
admission evaluation-selected action plus measured hold, or measured hold alone.  Fresh prospective validation
preparations remain the independent units; action branches and receiver points
are nested views.  Unsafe, ineligible and nonattempt parents cannot be rescued
by issuing a child.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods.structural_recurrence_targets import StructuralRecurrenceActionOutcome, StructuralRecurrenceStageEvidence, StructuralRecurrenceTargetStage, StructuralRecurrenceUnitObservation
from empirical_lawhood.adapters.methods.structural_recurrence import PolicyBranch
from empirical_lawhood.adapters.methods.action_fiber_structural_recurrence import ActionFiberStructuralRecurrenceTargetResult, PolicyValidationDisposition
from empirical_lawhood.adapters.methods.margin_structural_recurrence_forecast import MarginStructuralRecurrenceForecastTargetMatch, finalize_margin_target, match_target
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)

from .contracts import FREEGSNKE_PORT_IDS, FreeGsnkeActionChart, FreeGsnkeBranchKind, FreeGsnkeNumericalView, FreeGsnkePhase, FreeGsnkePreparation, FreeGsnkeProcessRequest, FreeGsnkeSourceBinding, FreeGsnkeTargetProcessResponse
from .design import FreeGsnkeActionDesign
from .structural_recurrence_bridge import FreeGsnkeStructuralRecurrenceBridgeFreeze, FreeGsnkeStructuralRecurrenceDesignCandidate, _branch, _composition_distance, _current_bounds_passed, _denominator_distance, _history_distance, _normalized_effect, _sink_margin
from .target_analysis import FreeGsnkePreparationReduction, FreeGsnkePreparationStratumBinding, FreeGsnkeTargetReductionConfig, reduce_freegsnke_subset_units
from .target_design import FreeGsnkePowerFreeze
from .target_evaluation import FreeGsnkeAdmissionEvaluationEvaluation


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


def _prospective_parent_eligible(parent: FreeGsnkeAdmissionEvaluationEvaluation) -> bool:
    return all(
        (
            parent.evaluation_eligible,
            not parent.panel_envelope_limited,
            parent.unsafe_false_admission_count == 0,
            parent.action_ontology_clock_error_count == 0,
        )
    )


def freegsnke_prospective_parent_eligible(parent: FreeGsnkeAdmissionEvaluationEvaluation) -> bool:
    """Return the frozen noncompensating prospective validation-entry decision for one admission evaluation parent."""

    return _prospective_parent_eligible(parent) and (
        parent.observed_policy_branch is not PolicyBranch.NONATTEMPT
    )


def selected_freegsnke_prospective_target_branches(
    *,
    parent_admission_evaluation: FreeGsnkeAdmissionEvaluationEvaluation,
    selected_design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
) -> tuple[str, ...]:
    """Project only the revealed selected action and mandatory measured hold."""

    policy = parent_admission_evaluation.admission_handoff.action_fiber_admission.policy_safety
    if policy.policy_branch is PolicyBranch.NONATTEMPT:
        raise ValueError("CONTROLLER_USE_PREREQUISITE_NONATTEMPT")
    if not _prospective_parent_eligible(parent_admission_evaluation):
        raise ValueError("CONTROLLER_USE_PARENT_INELIGIBLE_OR_UNSAFE")
    by_structural_recurrence_action = {
        value.structural_recurrence_action_id: value for value in selected_design_candidate.action_bindings
    }
    try:
        hold_binding = by_structural_recurrence_action["hold"]
        selected_binding = by_structural_recurrence_action[policy.selected_action_id]
    except KeyError as error:
        raise ValueError("FreeGSNKE prospective validation policy action lacks a frozen target binding") from error
    if policy.policy_branch is PolicyBranch.HOLD and policy.selected_action_id != "hold":
        raise ValueError("FreeGSNKE prospective validation hold policy action identity differs")
    return tuple(
        sorted(
            {hold_binding.branch_id}
            if policy.policy_branch is PolicyBranch.HOLD
            else {hold_binding.branch_id, selected_binding.branch_id}
        )
    )


class FreeGsnkeProspectiveValidationTerminalDisposition(StrEnum):
    VALIDATED_ACTION = "VALIDATED_ACTION"
    VALIDATED_HOLD = "VALIDATED_HOLD"
    OPPOSED = "OPPOSED"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    PARENT_INELIGIBLE_NONISSUE = "PARENT_INELIGIBLE_NONISSUE"


@dataclass(frozen=True, slots=True)
class FreeGsnkeProspectiveValidationRequestRoster(CanonicalRecord):
    """Separately issued fresh validation roster derived from one exact admission evaluation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-prospective-validation-request-roster'

    roster_id: str
    parent_admission_evaluation: ObjectIdentity
    target_power_freeze: ObjectIdentity
    selected_design_candidate: ObjectIdentity
    action_design: ObjectIdentity
    policy_branch: PolicyBranch
    selected_structural_recurrence_action_id: str
    selected_target_branch_ids: tuple[str, ...]
    hold_target_branch_id: str
    prospective_validation_unit_ids: tuple[str, ...]
    prospective_validation_seeds: tuple[int, ...]
    preparations: tuple[ObjectIdentity, ...]
    requests: tuple[ObjectIdentity, ...]
    prospective_validation_issue_authority: ObjectIdentity
    independent_preparation_count: int
    nested_branch_count: int
    minimum_complete_units: int
    panel_envelope_limited: bool
    fresh_from_development_and_evaluation: bool
    issued_after_parent_admission_evaluation: bool
    prospective_validation_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        validate_stable_id(
            self.selected_structural_recurrence_action_id,
            field_name='selected_structural_recurrence_action_id',
        )
        validate_stable_id(
            self.hold_target_branch_id,
            field_name="hold_target_branch_id",
        )
        if self.parent_admission_evaluation.object_schema != FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA:
            raise ValueError("FreeGSNKE prospective validation parent schema differs")
        if self.target_power_freeze.object_schema != FreeGsnkePowerFreeze.SCHEMA:
            raise ValueError("FreeGSNKE prospective validation power-freeze schema differs")
        if self.selected_design_candidate.object_schema != (FreeGsnkeStructuralRecurrenceDesignCandidate.SCHEMA):
            raise ValueError("FreeGSNKE prospective validation selected-design schema differs")
        if self.action_design.object_schema != FreeGsnkeActionDesign.SCHEMA:
            raise ValueError("FreeGSNKE prospective validation action-design schema differs")
        if self.policy_branch not in {PolicyBranch.EXACT_ACTION, PolicyBranch.HOLD}:
            raise ValueError("FreeGSNKE prospective validation child requires exact-action or hold admission evaluation")
        require_sorted_unique_strings(
            self.selected_target_branch_ids,
            field_name="selected_target_branch_ids",
            allow_empty=False,
        )
        if self.hold_target_branch_id not in self.selected_target_branch_ids:
            raise ValueError("FreeGSNKE prospective validation child omitted measured hold")
        if self.policy_branch is PolicyBranch.EXACT_ACTION:
            if self.selected_structural_recurrence_action_id == "hold" or len(self.selected_target_branch_ids) != 2:
                raise ValueError("FreeGSNKE exact-action prospective validation requires action plus hold")
        elif self.selected_structural_recurrence_action_id != "hold" or self.selected_target_branch_ids != (
            self.hold_target_branch_id,
        ):
            raise ValueError("FreeGSNKE hold prospective validation must issue only measured hold")
        require_sorted_unique_strings(
            self.prospective_validation_unit_ids,
            field_name="prospective_validation_unit_ids",
            allow_empty=False,
        )
        if (
            len(self.prospective_validation_unit_ids) != len(self.prospective_validation_seeds)
            or len(set(self.prospective_validation_seeds)) != len(self.prospective_validation_seeds)
            or any(value < 0 for value in self.prospective_validation_seeds)
        ):
            raise ValueError("FreeGSNKE prospective validation requires one unique seed per unit")
        require_sorted_unique_ids(
            self.preparations,
            attribute="object_id",
            field_name="preparations",
        )
        require_sorted_unique_ids(
            self.requests,
            attribute="object_id",
            field_name="requests",
        )
        if tuple(value.object_id for value in self.preparations) != self.prospective_validation_unit_ids:
            raise ValueError("FreeGSNKE prospective validation preparation/unit roster differs")
        if any(
            value.object_schema != FreeGsnkePreparation.SCHEMA for value in self.preparations
        ) or any(value.object_schema != FreeGsnkeProcessRequest.SCHEMA for value in self.requests):
            raise ValueError("FreeGSNKE prospective validation child record schema differs")
        expected_counts = (
            len(self.preparations),
            len(self.requests),
            len(self.preparations) < self.minimum_complete_units,
        )
        observed_counts = (
            self.independent_preparation_count,
            self.nested_branch_count,
            self.panel_envelope_limited,
        )
        if observed_counts != expected_counts or self.minimum_complete_units < 2:
            raise ValueError("FreeGSNKE prospective validation counts/limitation are not roster-derived")
        if len(self.requests) != len(self.preparations) * len(self.selected_target_branch_ids):
            raise ValueError("FreeGSNKE prospective validation request fibre count differs")
        if not self.fresh_from_development_and_evaluation or not self.issued_after_parent_admission_evaluation:
            raise ValueError("FreeGSNKE prospective validation child is not a fresh prospective issue")
        if self.prospective_validation_outcome_access_count:
            raise ValueError("FreeGSNKE prospective validation issue crossed its own outcome boundary")


def build_freegsnke_prospective_requests(
    *,
    source_binding: FreeGsnkeSourceBinding,
    numerical_view: FreeGsnkeNumericalView,
    action_charts_by_preparation: Mapping[str, FreeGsnkeActionChart],
    action_design: FreeGsnkeActionDesign,
    preparations: tuple[FreeGsnkePreparation, ...],
    selected_target_branch_ids: tuple[str, ...],
) -> tuple[FreeGsnkeProcessRequest, ...]:
    """Materialize the exact selected-action/hold request fibre."""
    require_sorted_unique_strings(
        selected_target_branch_ids,
        field_name="selected_target_branch_ids",
        allow_empty=False,
    )
    by_branch = {value.branch_id: value for value in action_design.branches}
    if not set(selected_target_branch_ids) <= set(by_branch):
        raise ValueError("FreeGSNKE prospective validation request branch lies outside action design")
    preparation_ids = {value.preparation_id for value in preparations}
    if set(action_charts_by_preparation) != preparation_ids:
        raise ValueError("FreeGSNKE prospective validation preparation/action-chart mapping differs")
    requests = []
    for preparation in sorted(preparations, key=lambda value: value.preparation_id):
        if preparation.phase is not FreeGsnkePhase.PROSPECTIVE_VALIDATION:
            raise ValueError("FreeGSNKE prospective validation request preparation phase differs")
        action_chart = action_charts_by_preparation[preparation.preparation_id]
        dose_by_port = {value.port_id: value.full_increment_v for value in action_chart.ports}
        for branch_id in selected_target_branch_ids:
            branch = by_branch[branch_id]
            requests.append(
                FreeGsnkeProcessRequest(
                    request_id=f"request.{preparation.preparation_id}.{branch_id}",
                    source_binding=source_binding,
                    preparation=preparation,
                    numerical_view=numerical_view,
                    action_chart=action_chart,
                    phase=FreeGsnkePhase.PROSPECTIVE_VALIDATION,
                    branch_id=branch_id,
                    branch_kind=branch.branch_kind,
                    requested_port_increments=tuple(
                        NamedDecimal(
                            value_id=port_id,
                            value=(
                                getattr(branch, f"{port_id}_dose_fraction") * dose_by_port[port_id]
                            ),
                            unit="V",
                        )
                        for port_id in FREEGSNKE_PORT_IDS
                    ),
                )
            )
    return tuple(sorted(requests, key=lambda value: value.request_id))


def issue_freegsnke_prospective_roster(
    *,
    roster_id: str,
    parent_admission_evaluation: FreeGsnkeAdmissionEvaluationEvaluation,
    selected_design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
    action_design: FreeGsnkeActionDesign,
    power_freeze: FreeGsnkePowerFreeze,
    prior_preparations: tuple[FreeGsnkePreparation, ...],
    prospective_validation_preparations: tuple[FreeGsnkePreparation, ...],
    action_charts_by_preparation: Mapping[str, FreeGsnkeActionChart],
    requests: tuple[FreeGsnkeProcessRequest, ...],
    prospective_validation_issue_authority: ObjectIdentity,
) -> FreeGsnkeProspectiveValidationRequestRoster:
    """Issue one conditional prospective validation child without converting adverse admission evaluation to nonentry."""

    policy = parent_admission_evaluation.admission_handoff.action_fiber_admission.policy_safety
    selected_branch_ids = selected_freegsnke_prospective_target_branches(
        parent_admission_evaluation=parent_admission_evaluation,
        selected_design_candidate=selected_design_candidate,
    )
    if (
        parent_admission_evaluation.target_power_freeze != power_freeze
        or parent_admission_evaluation.target_contract.structural_recurrence_target_design
        != ObjectIdentity.from_record(
            selected_design_candidate.design.design_id,
            selected_design_candidate.design,
        )
        or parent_admission_evaluation.target_contract.target_power_freeze
        != ObjectIdentity.from_record(power_freeze.freeze_id, power_freeze)
    ):
        raise ValueError("FreeGSNKE prospective validation parent design/power binding differs")
    if prospective_validation_issue_authority in {
        parent_admission_evaluation.execution_authority,
        parent_admission_evaluation.reveal_authority,
    }:
        raise ValueError("FreeGSNKE prospective validation issue authority is not a separate act")
    by_structural_recurrence_action = {
        value.structural_recurrence_action_id: value for value in selected_design_candidate.action_bindings
    }
    hold_binding = by_structural_recurrence_action["hold"]
    ordered_preparations = tuple(sorted(prospective_validation_preparations, key=lambda value: value.preparation_id))
    ordered_prior = tuple(sorted(prior_preparations, key=lambda value: value.preparation_id))
    expected_prior_ids = tuple(
        sorted((*power_freeze.development_unit_ids, *power_freeze.evaluation_unit_ids))
    )
    expected_prior_seeds = {
        **dict(
            zip(
                power_freeze.development_unit_ids,
                power_freeze.development_seeds,
                strict=True,
            )
        ),
        **dict(
            zip(
                power_freeze.evaluation_unit_ids,
                power_freeze.evaluation_seeds,
                strict=True,
            )
        ),
    }
    if (
        tuple(value.preparation_id for value in ordered_prior) != expected_prior_ids
        or any(value.seed != expected_prior_seeds[value.preparation_id] for value in ordered_prior)
        or {value.phase for value in ordered_prior}
        != {FreeGsnkePhase.DEVELOPMENT, FreeGsnkePhase.EVALUATION}
    ):
        raise ValueError("FreeGSNKE prospective validation prior preparation roster differs from power freeze")
    if (
        tuple(value.preparation_id for value in ordered_preparations) != power_freeze.prospective_validation_unit_ids
        or tuple(value.seed for value in ordered_preparations) != power_freeze.prospective_validation_seeds
        or any(
            value.phase is not FreeGsnkePhase.PROSPECTIVE_VALIDATION or value.saved_state is None
            for value in ordered_preparations
        )
    ):
        raise ValueError("FreeGSNKE prospective validation preparation roster differs from power freeze")
    all_preparations = (*ordered_prior, *ordered_preparations)
    if (
        len({value.preparation_id for value in all_preparations}) != len(all_preparations)
        or len({value.seed for value in all_preparations}) != len(all_preparations)
        or len(
            {
                (
                    value.plasma_current_target_a,
                    value.pressure_axis_pa,
                    value.current_profile_alpha_m,
                    value.current_profile_alpha_n,
                    value.elongation_target,
                    value.fvac_t_m,
                    value.seed,
                )
                for value in all_preparations
            }
        )
        != len(all_preparations)
    ):
        raise ValueError("FreeGSNKE prospective validation preparation identity is not fresh")
    saved_states = tuple(
        value.saved_state for value in all_preparations if value.saved_state is not None
    )
    if len(saved_states) != len(all_preparations) or (
        len({value.artifact_id for value in saved_states}) != len(saved_states)
        or len({value.sha256 for value in saved_states}) != len(saved_states)
    ):
        raise ValueError("FreeGSNKE prospective validation reuses a prior saved preparation")
    ordered_requests = tuple(sorted(requests, key=lambda value: value.request_id))
    expected_requests = build_freegsnke_prospective_requests(
        source_binding=ordered_requests[0].source_binding,
        numerical_view=ordered_requests[0].numerical_view,
        action_charts_by_preparation=action_charts_by_preparation,
        action_design=action_design,
        preparations=ordered_preparations,
        selected_target_branch_ids=selected_branch_ids,
    )
    if ordered_requests != expected_requests:
        raise ValueError("FreeGSNKE prospective validation request bytes differ from selected policy")
    if (
        ObjectIdentity.from_record(
            ordered_requests[0].source_binding.binding_id,
            ordered_requests[0].source_binding,
        )
        != power_freeze.source_binding
        or ObjectIdentity.from_record(
            ordered_requests[0].numerical_view.view_id,
            ordered_requests[0].numerical_view,
        )
        != power_freeze.selected_numerical_view
        or ObjectIdentity.from_record(action_design.design_id, action_design)
        != power_freeze.action_design
    ):
        raise ValueError("FreeGSNKE prospective validation source/view/action freeze differs")
    return FreeGsnkeProspectiveValidationRequestRoster(
        roster_id=roster_id,
        parent_admission_evaluation=ObjectIdentity.from_record(parent_admission_evaluation.evaluation_id, parent_admission_evaluation),
        target_power_freeze=ObjectIdentity.from_record(power_freeze.freeze_id, power_freeze),
        selected_design_candidate=ObjectIdentity.from_record(
            selected_design_candidate.candidate_id,
            selected_design_candidate,
        ),
        action_design=ObjectIdentity.from_record(action_design.design_id, action_design),
        policy_branch=policy.policy_branch,
        selected_structural_recurrence_action_id=policy.selected_action_id,
        selected_target_branch_ids=selected_branch_ids,
        hold_target_branch_id=hold_binding.branch_id,
        prospective_validation_unit_ids=power_freeze.prospective_validation_unit_ids,
        prospective_validation_seeds=power_freeze.prospective_validation_seeds,
        preparations=tuple(
            ObjectIdentity.from_record(value.preparation_id, value)
            for value in ordered_preparations
        ),
        requests=tuple(
            ObjectIdentity.from_record(value.request_id, value) for value in ordered_requests
        ),
        prospective_validation_issue_authority=prospective_validation_issue_authority,
        independent_preparation_count=len(ordered_preparations),
        nested_branch_count=len(ordered_requests),
        minimum_complete_units=power_freeze.prospective_validation_minimum_complete_units,
        panel_envelope_limited=(len(ordered_preparations) < power_freeze.prospective_validation_minimum_complete_units),
        fresh_from_development_and_evaluation=True,
        issued_after_parent_admission_evaluation=True,
        prospective_validation_outcome_access_count=0,
    )


@dataclass(frozen=True, slots=True)
class FreeGsnkeProspectiveValidationReductionConfig(CanonicalRecord):
    """prospective validation grid/strata freeze reusing the exact qualified evaluation view."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-prospective-validation-reduction-config'

    config_id: str
    prospective_validation_roster: ObjectIdentity
    parent_evaluation_config: FreeGsnkeTargetReductionConfig
    preparation_strata: tuple[FreeGsnkePreparationStratumBinding, ...]
    minimum_complete_units: int
    frozen_before_prospective_validation_outcomes: bool
    prospective_validation_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if self.prospective_validation_roster.object_schema != FreeGsnkeProspectiveValidationRequestRoster.SCHEMA:
            raise ValueError("FreeGSNKE prospective validation reduction roster schema differs")
        if self.parent_evaluation_config.phase is not FreeGsnkePhase.EVALUATION:
            raise ValueError("FreeGSNKE prospective validation must reuse the evaluation grid contract")
        require_sorted_unique_ids(
            self.preparation_strata,
            attribute="binding_id",
            field_name="preparation_strata",
        )
        if self.minimum_complete_units < 2:
            raise ValueError("FreeGSNKE prospective validation complete-unit minimum differs")
        if not self.frozen_before_prospective_validation_outcomes or self.prospective_validation_outcome_access_count:
            raise ValueError("FreeGSNKE prospective validation reduction crossed its outcome freeze")

    @property
    def required_horizons_s(self) -> tuple[Decimal, ...]:
        return self.parent_evaluation_config.required_horizons_s

    @property
    def required_receiver_ids(self) -> tuple[str, ...]:
        return self.parent_evaluation_config.required_receiver_ids

    @property
    def required_topology_id(self) -> str:
        return self.parent_evaluation_config.required_topology_id

    @property
    def required_active_current_ids(self) -> tuple[str, ...]:
        return self.parent_evaluation_config.required_active_current_ids

    @property
    def passive_current_summary_ids(self) -> tuple[str, ...]:
        return self.parent_evaluation_config.passive_current_summary_ids

    @property
    def retain_plasma_current(self) -> bool:
        return self.parent_evaluation_config.retain_plasma_current


def freeze_freegsnke_prospective_reduction_config(
    *,
    config_id: str,
    roster: FreeGsnkeProspectiveValidationRequestRoster,
    parent_evaluation_config: FreeGsnkeTargetReductionConfig,
    prospective_validation_preparations: tuple[FreeGsnkePreparation, ...],
) -> FreeGsnkeProspectiveValidationReductionConfig:
    """Reuse the evaluation grid/strata without opening prospective validation outcomes."""

    ordered_preparations = tuple(sorted(prospective_validation_preparations, key=lambda value: value.preparation_id))
    if tuple(value.preparation_id for value in ordered_preparations) != (roster.prospective_validation_unit_ids):
        raise ValueError("FreeGSNKE prospective validation reduction preparation roster differs")
    templates = parent_evaluation_config.preparation_strata
    if len(templates) != len(ordered_preparations):
        raise ValueError("FreeGSNKE prospective validation/evaluation stratum panel sizes differ")
    bindings = tuple(
        FreeGsnkePreparationStratumBinding(
            binding_id=f"stratum-binding.{preparation.preparation_id}",
            preparation=ObjectIdentity.from_record(
                preparation.preparation_id,
                preparation,
            ),
            stratum_id=template.stratum_id,
            denominator_stratum_id=template.denominator_stratum_id,
            history_stratum_id=template.history_stratum_id,
        )
        for template, preparation in zip(
            templates,
            ordered_preparations,
            strict=True,
        )
    )
    return FreeGsnkeProspectiveValidationReductionConfig(
        config_id=config_id,
        prospective_validation_roster=ObjectIdentity.from_record(roster.roster_id, roster),
        parent_evaluation_config=parent_evaluation_config,
        preparation_strata=bindings,
        minimum_complete_units=roster.minimum_complete_units,
        frozen_before_prospective_validation_outcomes=True,
        prospective_validation_outcome_access_count=0,
    )


@dataclass(frozen=True, slots=True)
class FreeGsnkeProspectiveValidationReduction(CanonicalRecord):
    """All issued prospective validation preparations, including adverse units."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-prospective-validation-reduction'

    reduction_id: str
    config: ObjectIdentity
    prospective_validation_roster: ObjectIdentity
    parent_admission_evaluation: ObjectIdentity
    units: tuple[FreeGsnkePreparationReduction, ...]
    issued_unit_count: int
    inference_complete_unit_count: int
    postissue_adverse_unit_count: int
    missing_response_count: int
    action_ontology_error_count: int
    frozen_minimum_complete_units: int
    panel_envelope_limited: bool
    complete_unit_ids_sha256: str
    issued_unit_ids_sha256: str
    nested_observations_count_as_units: bool
    outcome_access: OutcomeAccess
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.reduction_id, field_name="reduction_id")
        if self.config.object_schema != FreeGsnkeProspectiveValidationReductionConfig.SCHEMA:
            raise ValueError("FreeGSNKE prospective validation reduction config schema differs")
        if self.prospective_validation_roster.object_schema != FreeGsnkeProspectiveValidationRequestRoster.SCHEMA:
            raise ValueError("FreeGSNKE prospective validation reduction roster schema differs")
        if self.parent_admission_evaluation.object_schema != FreeGsnkeAdmissionEvaluationEvaluation.SCHEMA:
            raise ValueError("FreeGSNKE prospective validation reduction parent schema differs")
        require_sorted_unique_ids(self.units, attribute="unit_id", field_name="units")
        if any(value.phase is not FreeGsnkePhase.PROSPECTIVE_VALIDATION for value in self.units):
            raise ValueError("FreeGSNKE prospective validation reduction crossed phase partitions")
        complete = sum(value.complete_for_inference for value in self.units)
        missing = sum(
            value.issued_branch_count - value.observed_response_count for value in self.units
        )
        errors = sum(value.action_ontology_error_count for value in self.units)
        observed = (
            self.issued_unit_count,
            self.inference_complete_unit_count,
            self.postissue_adverse_unit_count,
            self.missing_response_count,
            self.action_ontology_error_count,
            self.panel_envelope_limited,
        )
        expected = (
            len(self.units),
            complete,
            len(self.units) - complete,
            missing,
            errors,
            complete < self.frozen_minimum_complete_units,
        )
        if observed != expected:
            raise ValueError("FreeGSNKE prospective validation counts are not complete-unit-derived")
        for name in ("complete_unit_ids_sha256", "issued_unit_ids_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if self.nested_observations_count_as_units:
            raise ValueError("FreeGSNKE prospective validation nested views cannot inflate units")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("FreeGSNKE prospective validation reduction must remain sealed")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


def reduce_freegsnke_prospective_panel(
    *,
    config: FreeGsnkeProspectiveValidationReductionConfig,
    roster: FreeGsnkeProspectiveValidationRequestRoster,
    action_design: FreeGsnkeActionDesign,
    requests: tuple[FreeGsnkeProcessRequest, ...],
    responses: tuple[FreeGsnkeTargetProcessResponse, ...],
) -> FreeGsnkeProspectiveValidationReduction:
    """Reduce the separately issued selected-action/hold validation panel."""

    roster_identity = ObjectIdentity.from_record(roster.roster_id, roster)
    if (
        config.prospective_validation_roster != roster_identity
        or roster.action_design
        != ObjectIdentity.from_record(action_design.design_id, action_design)
        or config.minimum_complete_units != roster.minimum_complete_units
    ):
        raise ValueError("FreeGSNKE prospective validation reduction freeze differs")
    by_branch = {value.branch_id: value for value in action_design.branches}
    if not set(roster.selected_target_branch_ids) <= set(by_branch) or (
        by_branch[roster.hold_target_branch_id].branch_kind is not FreeGsnkeBranchKind.COMPARATOR
    ):
        raise ValueError("FreeGSNKE prospective validation selected branch/hold design differs")
    ordered_requests = tuple(sorted(requests, key=lambda value: value.request_id))
    if (
        tuple(ObjectIdentity.from_record(value.request_id, value) for value in ordered_requests)
        != roster.requests
    ):
        raise ValueError("FreeGSNKE prospective validation request bytes differ from issued roster")
    parent_config = config.parent_evaluation_config
    for request in ordered_requests:
        if (
            request.phase is not FreeGsnkePhase.PROSPECTIVE_VALIDATION
            or ObjectIdentity.from_record(
                request.source_binding.binding_id,
                request.source_binding,
            )
            != parent_config.source_binding
            or ObjectIdentity.from_record(
                request.numerical_view.view_id,
                request.numerical_view,
            )
            != parent_config.numerical_view
            or ObjectIdentity.from_record(
                request.action_chart.chart_id,
                request.action_chart,
            )
            not in parent_config.action_charts
        ):
            raise ValueError("FreeGSNKE prospective validation request changed source/view/chart")
    chart_horizon_rosters = {
        tuple(
            sorted(
                {
                    *request.action_chart.primary_horizons_s,
                    *request.action_chart.diagnostic_horizons_s,
                }
            )
        )
        for request in ordered_requests
    }
    if chart_horizon_rosters != {config.required_horizons_s}:
        raise ValueError("FreeGSNKE prospective validation analysis grid differs from evaluation")
    units = reduce_freegsnke_subset_units(
        config=config,
        phase=FreeGsnkePhase.PROSPECTIVE_VALIDATION,
        preparation_identities=roster.preparations,
        preparation_strata=config.preparation_strata,
        expected_branch_ids=roster.selected_target_branch_ids,
        hold_branch_id=roster.hold_target_branch_id,
        requests=ordered_requests,
        responses=responses,
    )
    complete_ids = tuple(value.unit_id for value in units if value.complete_for_inference)
    issued_ids = tuple(value.unit_id for value in units)
    complete_count = len(complete_ids)
    reasons = set()
    if complete_count < config.minimum_complete_units:
        reasons.add("PANEL_ENVELOPE_LIMITED")
    if any(value.action_ontology_error_count for value in units):
        reasons.add("ACTION_ONTOLOGY_COUNTEREXAMPLE")
    if any(value.observed_response_count < value.issued_branch_count for value in units):
        reasons.add("POSTISSUE_RESPONSE_MISSING")
    return FreeGsnkeProspectiveValidationReduction(
        reduction_id=f"reduction.independent-substrate-grounding-freegsnke.prospective-validation.{roster.roster_id}",
        config=ObjectIdentity.from_record(config.config_id, config),
        prospective_validation_roster=roster_identity,
        parent_admission_evaluation=roster.parent_admission_evaluation,
        units=units,
        issued_unit_count=len(units),
        inference_complete_unit_count=complete_count,
        postissue_adverse_unit_count=len(units) - complete_count,
        missing_response_count=sum(
            value.issued_branch_count - value.observed_response_count for value in units
        ),
        action_ontology_error_count=sum(value.action_ontology_error_count for value in units),
        frozen_minimum_complete_units=config.minimum_complete_units,
        panel_envelope_limited=complete_count < config.minimum_complete_units,
        complete_unit_ids_sha256=_digest_ids(complete_ids),
        issued_unit_ids_sha256=_digest_ids(issued_ids),
        nested_observations_count_as_units=False,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        reason_codes=tuple(sorted(reasons)),
    )


def lower_freegsnke_prospective_to_structural_recurrence(
    *,
    bridge: FreeGsnkeStructuralRecurrenceBridgeFreeze,
    design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
    roster: FreeGsnkeProspectiveValidationRequestRoster,
    reduction_config: FreeGsnkeProspectiveValidationReductionConfig,
    reduction: FreeGsnkeProspectiveValidationReduction,
    action_design: FreeGsnkeActionDesign,
    execution_authority_verified: bool,
) -> StructuralRecurrenceStageEvidence:
    'Lower fresh selected-action/hold units through the unchanged structural recurrence schema.'

    if (
        design_candidate not in bridge.design_candidates
        or roster.selected_design_candidate
        != ObjectIdentity.from_record(design_candidate.candidate_id, design_candidate)
        or roster.action_design
        != ObjectIdentity.from_record(action_design.design_id, action_design)
        or reduction.config
        != ObjectIdentity.from_record(reduction_config.config_id, reduction_config)
        or reduction.prospective_validation_roster != ObjectIdentity.from_record(roster.roster_id, roster)
    ):
        raise ValueError("FreeGSNKE prospective validation structural recurrence lowering changed a frozen operand")
    design = design_candidate.design
    structural_recurrence_roster = design.roster(StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION)
    if (
        structural_recurrence_roster.unit_ids != roster.prospective_validation_unit_ids
        or structural_recurrence_roster.seeds != roster.prospective_validation_seeds
        or tuple(value.unit_id for value in reduction.units) != roster.prospective_validation_unit_ids
    ):
        raise ValueError("FreeGSNKE prospective validation structural recurrence roster differs")
    selected_bindings = tuple(
        value
        for value in design_candidate.action_bindings
        if value.branch_id in set(roster.selected_target_branch_ids)
    )
    expected_actions = (
        ("hold",)
        if roster.policy_branch is PolicyBranch.HOLD
        else tuple(sorted(("hold", roster.selected_structural_recurrence_action_id)))
    )
    if tuple(sorted(value.structural_recurrence_action_id for value in selected_bindings)) != expected_actions:
        raise ValueError("FreeGSNKE prospective validation structural recurrence action subset differs from admission evaluation")
    observed_strata = tuple(sorted({value.denominator_stratum_id for value in reduction.units}))
    if observed_strata != bridge.native_denominator_stratum_ids:
        raise ValueError("FreeGSNKE prospective validation changed the denominator stratum roster")
    denominator_distance = _denominator_distance(
        reduction.units,
        selected_bindings,
        design_candidate,
    )
    checkpoint_distance = _history_distance(
        reduction.units,
        selected_bindings,
        design_candidate,
        horizon_s=design_candidate.history_checkpoint_horizon_s,
    )
    future_distance = _history_distance(
        reduction.units,
        selected_bindings,
        design_candidate,
        horizon_s=design_candidate.history_future_horizon_s,
    )
    available_branches = set(roster.selected_target_branch_ids)
    composition_bindings = tuple(
        value
        for value in design_candidate.composition_bindings
        if {value.joint_branch_id, *value.component_branch_ids} <= available_branches
    )
    failure = bridge.failure_distance
    branch_specs = {value.branch_id: value for value in action_design.branches}
    seed_by_unit = dict(zip(structural_recurrence_roster.unit_ids, structural_recurrence_roster.seeds, strict=True))
    observations = []
    for unit in reduction.units:
        composition_distance = _composition_distance(
            unit,
            composition_bindings,
            failure_distance=failure,
        )
        outcomes = []
        for binding in selected_bindings:
            branch = _branch(unit, binding.branch_id)
            audit = branch.action_audit
            requested = Decimal(1)
            accepted = Decimal(audit.accepted_observed and audit.requested_accepted_exact)
            applied = Decimal(
                accepted == 1
                and audit.applied_observed
                and audit.accepted_applied_exact
                and not audit.clipping_observed
            )
            realized = Decimal(
                applied == 1
                and audit.realized_observed
                and audit.receiver_realization_clocks_aligned
            )
            effect = _normalized_effect(unit, binding)
            target_effect = effect if effect is not None else Decimal(0)
            sink_margin = _sink_margin(
                branch,
                design_candidate.sink_gates,
                failure_distance=failure,
            )
            spec = branch_specs[binding.branch_id]
            timing_passed = all(
                (
                    audit.temporal_order_passed,
                    audit.receiver_realization_clocks_aligned,
                    audit.causal_cutoff_passed,
                )
            )
            current_bounds_passed = _current_bounds_passed(
                branch,
                reduction_config.parent_evaluation_config,
            )
            target_passed = binding.structural_recurrence_action_id == "hold" or target_effect >= 1
            support = unit.complete_for_inference and target_passed
            preservation = branch.analysis_grid_complete and sink_margin >= 0
            reasons = {
                *branch.reason_codes,
                *(() if effect is not None else ("TARGET_EFFECT_UNOBSERVED",)),
                *(
                    ()
                    if denominator_distance is not None
                    else ("CONTROLLER_USE_DENOMINATOR_CONTRAST_UNEVALUABLE",)
                ),
                *(
                    ()
                    if checkpoint_distance is not None
                    else ("CONTROLLER_USE_HISTORY_CHECKPOINT_CONTRAST_UNEVALUABLE",)
                ),
                *(
                    ()
                    if future_distance is not None
                    else ("CONTROLLER_USE_HISTORY_FUTURE_CONTRAST_UNEVALUABLE",)
                ),
                *(() if current_bounds_passed else ("SOURCE_CURRENT_BOUND_FAILED_OR_UNOBSERVED",)),
                *(() if execution_authority_verified else ("EXECUTION_AUTHORITY_UNVERIFIED",)),
                *(() if support else ("TARGET_LOCAL_SUPPORT_FAILED",)),
            }
            outcomes.append(
                StructuralRecurrenceActionOutcome(
                    outcome_id=f"{unit.unit_id}.{binding.structural_recurrence_action_id}",
                    action_id=binding.structural_recurrence_action_id,
                    requested_action=requested,
                    accepted_action=accepted,
                    applied_action=applied,
                    realized_action=realized,
                    target_effect=target_effect,
                    sink_margin=sink_margin,
                    effort=max(abs(spec.p4_dose_fraction), abs(spec.p5_dose_fraction)),
                    realization_error=max(
                        abs(requested - accepted),
                        abs(accepted - applied),
                        abs(applied - realized),
                    ),
                    denominator_distance=(
                        denominator_distance if denominator_distance is not None else failure
                    ),
                    history_checkpoint_distance=(
                        checkpoint_distance if checkpoint_distance is not None else failure
                    ),
                    history_future_distance=(
                        future_distance if future_distance is not None else failure
                    ),
                    direct_composed_distance=composition_distance,
                    timing_offset=Decimal(0) if timing_passed else failure,
                    support_preserved=support,
                    validity_passed=branch.analysis_grid_complete,
                    preservation_passed=preservation,
                    dynamics_passed=branch.analysis_grid_complete,
                    reachability_passed=bool(applied) and current_bounds_passed,
                    authority_passed=execution_authority_verified,
                    uncertainty_evaluable=(
                        unit.complete_for_inference and not reduction.panel_envelope_limited
                    ),
                    reason_codes=tuple(sorted(reasons)),
                )
            )
        observations.append(
            StructuralRecurrenceUnitObservation(
                unit_id=unit.unit_id,
                target_slot_id=design.target_slot.target_slot_id,
                stage=StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION,
                seed=seed_by_unit[unit.unit_id],
                preparation_fingerprint=unit.preparation.object_fingerprint,
                outcomes=tuple(sorted(outcomes, key=lambda value: value.outcome_id)),
            )
        )
    ordered = tuple(sorted(observations, key=lambda value: value.unit_id))
    return StructuralRecurrenceStageEvidence(
        evidence_id=(f"evidence.{design.target_slot.target_slot_id}.prospective-validation.freegsnke-structural-recurrence-bridge"),
        target_design=ObjectIdentity.from_record(design.design_id, design),
        target_slot_id=design.target_slot.target_slot_id,
        stage=StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION,
        units=ordered,
        independent_unit_count=len(ordered),
        nested_action_outcome_count=sum(len(value.outcomes) for value in ordered),
        source_implementation=design.selected_source_implementation,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )


@dataclass(frozen=True, slots=True)
class FreeGsnkeTargetValidation(CanonicalRecord):
    """Terminal prospective validation/nonattempt disposition before compact target projection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-target-validation'

    validation_id: str
    parent_admission_evaluation: FreeGsnkeAdmissionEvaluationEvaluation
    prospective_validation_roster: FreeGsnkeProspectiveValidationRequestRoster | None
    prospective_validation_reduction: FreeGsnkeProspectiveValidationReduction | None
    structural_recurrence_validation_evidence: StructuralRecurrenceStageEvidence | None
    target_result: ActionFiberStructuralRecurrenceTargetResult | None
    target_match: MarginStructuralRecurrenceForecastTargetMatch | None
    prospective_validation_execution_authority: ObjectIdentity | None
    prospective_validation_reveal_authority: ObjectIdentity | None
    disposition: FreeGsnkeProspectiveValidationTerminalDisposition
    prospective_validation_child_issued: bool
    prerequisite_nonattempt: bool
    validation_supported: bool
    validation_opposed: bool
    panel_envelope_limited: bool
    action_ontology_clock_error_count: int
    outcome_access: OutcomeAccess
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.validation_id, field_name="validation_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        child_fields = (
            self.prospective_validation_roster,
            self.prospective_validation_reduction,
            self.structural_recurrence_validation_evidence,
            self.prospective_validation_execution_authority,
            self.prospective_validation_reveal_authority,
        )
        if self.prospective_validation_child_issued != all(value is not None for value in child_fields):
            raise ValueError("FreeGSNKE prospective validation child presence is internally inconsistent")
        if self.prospective_validation_child_issued and any(value is None for value in child_fields):
            raise ValueError("FreeGSNKE prospective validation child is incomplete")
        if self.prospective_validation_child_issued:
            assert self.prospective_validation_roster is not None
            assert self.prospective_validation_reduction is not None
            assert self.structural_recurrence_validation_evidence is not None
            assert self.prospective_validation_execution_authority is not None
            assert self.prospective_validation_reveal_authority is not None
            if (
                self.prospective_validation_roster.parent_admission_evaluation
                != ObjectIdentity.from_record(
                    self.parent_admission_evaluation.evaluation_id,
                    self.parent_admission_evaluation,
                )
                or self.prospective_validation_reduction.parent_admission_evaluation != self.prospective_validation_roster.parent_admission_evaluation
                or self.structural_recurrence_validation_evidence.stage is not StructuralRecurrenceTargetStage.PROSPECTIVE_VALIDATION
            ):
                raise ValueError("FreeGSNKE prospective validation child/parent binding differs")
            if (
                len(
                    {
                        self.parent_admission_evaluation.execution_authority,
                        self.parent_admission_evaluation.reveal_authority,
                        self.prospective_validation_roster.prospective_validation_issue_authority,
                        self.prospective_validation_execution_authority,
                        self.prospective_validation_reveal_authority,
                    }
                )
                != 5
            ):
                raise ValueError("FreeGSNKE prospective validation authority acts are not distinct")
        if self.target_result is None or self.target_match is None:
            if self.disposition is not (
                FreeGsnkeProspectiveValidationTerminalDisposition.PARENT_INELIGIBLE_NONISSUE
            ):
                raise ValueError('FreeGSNKE validation lacks a terminal structural recurrence result')
        else:
            if self.target_match.target_result != ObjectIdentity.from_record(
                self.target_result.result_id,
                self.target_result,
            ):
                raise ValueError("FreeGSNKE validation result/match binding differs")
        expected_nonattempt = self.parent_admission_evaluation.observed_policy_branch is PolicyBranch.NONATTEMPT
        if self.prerequisite_nonattempt != expected_nonattempt:
            raise ValueError("FreeGSNKE prerequisite nonattempt is not admission evaluation-derived")
        expected_limited = (
            self.prospective_validation_reduction.panel_envelope_limited
            if self.prospective_validation_reduction is not None
            else self.parent_admission_evaluation.panel_envelope_limited
        )
        expected_errors = (
            self.prospective_validation_reduction.action_ontology_error_count
            if self.prospective_validation_reduction is not None
            else self.parent_admission_evaluation.action_ontology_clock_error_count
        )
        if (
            self.panel_envelope_limited != expected_limited
            or self.action_ontology_clock_error_count != expected_errors
        ):
            raise ValueError("FreeGSNKE validation adverse summary is not evidence-derived")
        supported = bool(
            self.prospective_validation_child_issued
            and self.target_result is not None
            and self.target_result.validation_disposition is PolicyValidationDisposition.VALIDATED
            and not expected_limited
            and not expected_errors
        )
        opposed = bool(
            self.prospective_validation_child_issued
            and self.target_result is not None
            and (
                self.target_result.validation_disposition is PolicyValidationDisposition.OPPOSED
                or expected_errors
            )
        )
        if self.validation_supported != supported or self.validation_opposed != opposed:
            raise ValueError("FreeGSNKE validation status is not result-derived")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("FreeGSNKE target validation requires evaluator reveal")


def _validation_reasons(
    *,
    parent: FreeGsnkeAdmissionEvaluationEvaluation,
    reduction: FreeGsnkeProspectiveValidationReduction | None,
    result: ActionFiberStructuralRecurrenceTargetResult | None,
    match: MarginStructuralRecurrenceForecastTargetMatch | None,
    disposition: FreeGsnkeProspectiveValidationTerminalDisposition,
) -> tuple[str, ...]:
    reasons = {
        *parent.reason_codes,
        *((reduction.reason_codes) if reduction is not None else ()),
        *((match.reason_codes) if match is not None else ()),
        disposition.value,
    }
    if result is not None and result.validation_disposition is PolicyValidationDisposition.OPPOSED:
        reasons.add("CONTROLLER_USE_POLICY_VALIDATION_OPPOSED")
    return tuple(sorted(reasons))


def finalize_freegsnke_without_prospective(
    *,
    validation_id: str,
    parent_admission_evaluation: FreeGsnkeAdmissionEvaluationEvaluation,
    selected_design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
) -> FreeGsnkeTargetValidation:
    """Close prerequisite nonattempt or an unsafe/ineligible prospective validation nonissue."""

    branch = parent_admission_evaluation.observed_policy_branch
    if branch is not PolicyBranch.NONATTEMPT and _prospective_parent_eligible(parent_admission_evaluation):
        raise ValueError("FreeGSNKE eligible action/hold parent requires a prospective validation child")
    result = None
    target_match = None
    if branch is PolicyBranch.NONATTEMPT:
        result = finalize_margin_target(
            admission_handoff=parent_admission_evaluation.admission_handoff,
            design=selected_design_candidate.design,
            validation_evidence=None,
            validation_authority=parent_admission_evaluation.reveal_authority,
        )
        target_match = match_target(
            prediction=parent_admission_evaluation.prediction_issue,
            admission_handoff=parent_admission_evaluation.admission_handoff,
            result=result,
        )
        disposition = FreeGsnkeProspectiveValidationTerminalDisposition.PREREQUISITE_NONATTEMPT
    else:
        disposition = FreeGsnkeProspectiveValidationTerminalDisposition.PARENT_INELIGIBLE_NONISSUE
    reasons = _validation_reasons(
        parent=parent_admission_evaluation,
        reduction=None,
        result=result,
        match=target_match,
        disposition=disposition,
    )
    limited = parent_admission_evaluation.panel_envelope_limited
    errors = parent_admission_evaluation.action_ontology_clock_error_count
    supported = False
    opposed = False
    return FreeGsnkeTargetValidation(
        validation_id=validation_id,
        parent_admission_evaluation=parent_admission_evaluation,
        prospective_validation_roster=None,
        prospective_validation_reduction=None,
        structural_recurrence_validation_evidence=None,
        target_result=result,
        target_match=target_match,
        prospective_validation_execution_authority=None,
        prospective_validation_reveal_authority=None,
        disposition=disposition,
        prospective_validation_child_issued=False,
        prerequisite_nonattempt=branch is PolicyBranch.NONATTEMPT,
        validation_supported=supported,
        validation_opposed=opposed,
        panel_envelope_limited=limited,
        action_ontology_clock_error_count=errors,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        reason_codes=reasons,
    )


def evaluate_freegsnke_prospective(
    *,
    validation_id: str,
    parent_admission_evaluation: FreeGsnkeAdmissionEvaluationEvaluation,
    bridge: FreeGsnkeStructuralRecurrenceBridgeFreeze,
    selected_design_candidate: FreeGsnkeStructuralRecurrenceDesignCandidate,
    action_design: FreeGsnkeActionDesign,
    roster: FreeGsnkeProspectiveValidationRequestRoster,
    reduction_config: FreeGsnkeProspectiveValidationReductionConfig,
    reduction: FreeGsnkeProspectiveValidationReduction,
    prospective_validation_execution_authority: ObjectIdentity,
    prospective_validation_reveal_authority: ObjectIdentity,
) -> FreeGsnkeTargetValidation:
    """Reveal and finalize one separately issued fresh prospective validation child."""

    if not _prospective_parent_eligible(parent_admission_evaluation):
        raise ValueError("FreeGSNKE prospective validation cannot evaluate an ineligible/unsafe parent")
    if roster.parent_admission_evaluation != ObjectIdentity.from_record(parent_admission_evaluation.evaluation_id, parent_admission_evaluation):
        raise ValueError("FreeGSNKE prospective validation roster names another parent")
    authorities = {
        parent_admission_evaluation.execution_authority,
        parent_admission_evaluation.reveal_authority,
        roster.prospective_validation_issue_authority,
        prospective_validation_execution_authority,
        prospective_validation_reveal_authority,
    }
    if len(authorities) != 5:
        raise ValueError("FreeGSNKE prospective validation issue/execution/reveal authorities must be distinct")
    evidence = lower_freegsnke_prospective_to_structural_recurrence(
        bridge=bridge,
        design_candidate=selected_design_candidate,
        roster=roster,
        reduction_config=reduction_config,
        reduction=reduction,
        action_design=action_design,
        execution_authority_verified=True,
    )
    result = finalize_margin_target(
        admission_handoff=parent_admission_evaluation.admission_handoff,
        design=selected_design_candidate.design,
        validation_evidence=evidence,
        validation_authority=prospective_validation_reveal_authority,
    )
    target_match = match_target(
        prediction=parent_admission_evaluation.prediction_issue,
        admission_handoff=parent_admission_evaluation.admission_handoff,
        result=result,
    )
    limited = reduction.panel_envelope_limited
    errors = reduction.action_ontology_error_count
    supported = all(
        (
            result.validation_disposition is PolicyValidationDisposition.VALIDATED,
            not limited,
            errors == 0,
        )
    )
    opposed = any(
        (
            result.validation_disposition is PolicyValidationDisposition.OPPOSED,
            errors > 0,
        )
    )
    disposition = (
        FreeGsnkeProspectiveValidationTerminalDisposition.OPPOSED
        if opposed
        else FreeGsnkeProspectiveValidationTerminalDisposition.VALIDATED_ACTION
        if roster.policy_branch is PolicyBranch.EXACT_ACTION and supported
        else FreeGsnkeProspectiveValidationTerminalDisposition.VALIDATED_HOLD
        if roster.policy_branch is PolicyBranch.HOLD and supported
        else FreeGsnkeProspectiveValidationTerminalDisposition.OPPOSED
    )
    reasons = _validation_reasons(
        parent=parent_admission_evaluation,
        reduction=reduction,
        result=result,
        match=target_match,
        disposition=disposition,
    )
    return FreeGsnkeTargetValidation(
        validation_id=validation_id,
        parent_admission_evaluation=parent_admission_evaluation,
        prospective_validation_roster=roster,
        prospective_validation_reduction=reduction,
        structural_recurrence_validation_evidence=evidence,
        target_result=result,
        target_match=target_match,
        prospective_validation_execution_authority=prospective_validation_execution_authority,
        prospective_validation_reveal_authority=prospective_validation_reveal_authority,
        disposition=disposition,
        prospective_validation_child_issued=True,
        prerequisite_nonattempt=False,
        validation_supported=supported,
        validation_opposed=opposed,
        panel_envelope_limited=limited,
        action_ontology_clock_error_count=errors,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        reason_codes=reasons,
    )


__all__ = [
    'FreeGsnkeProspectiveValidationReductionConfig',
    'FreeGsnkeProspectiveValidationReduction',
    'FreeGsnkeProspectiveValidationRequestRoster',
    'FreeGsnkeProspectiveValidationTerminalDisposition',
    'FreeGsnkeTargetValidation',
    'build_freegsnke_prospective_requests',
    'evaluate_freegsnke_prospective',
    'finalize_freegsnke_without_prospective',
    'freegsnke_prospective_parent_eligible',
    'freeze_freegsnke_prospective_reduction_config',
    'issue_freegsnke_prospective_roster',
    'lower_freegsnke_prospective_to_structural_recurrence',
    'reduce_freegsnke_prospective_panel',
    'selected_freegsnke_prospective_target_branches',
]

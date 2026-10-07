"""Outcome-blind target freezes for the two activated target construct validation simulators."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import product
from typing import ClassVar

from empirical_lawhood.adapters.methods.structural_recurrence_runtime import method_source_manifest
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .comparators import TargetConstructValidationComparatorKind, TargetConstructValidationComparatorPlan
from .construct_validity import TargetConstructValidationConstructReviewAttestation, TargetConstructValidationConstructThreat, TargetConstructValidationConstructValidityAssessment, assess_construct_validity
from .contracts import TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID, TargetConstructValidationTargetRelationBinding, primary_relation_codebook
from .forecast import TargetConstructValidationDirectForecastCell, TargetConstructValidationForecastDisposition, TargetConstructValidationForecastEmission, TargetConstructValidationPredictionIssue, TargetConstructValidationStateProjection
from .mapping import TargetConstructValidationCompatibilityMapCandidate, TargetConstructValidationDenominatorAlternativeKind, TargetConstructValidationDenominatorCandidate, TargetConstructValidationMappingConstructor, TargetConstructValidationRoleBinding
from .native_dossier import TargetConstructValidationIndependenceDossier, TargetConstructValidationTargetNativeDossier
from .nomination_freeze import frozen_nomination
from .source_qualification import frozen_source_qualification


LEGAL_STATE_IDS = ("state.high", "state.low", "state.neutral")


def _digest_ids(values: tuple[str, ...]) -> str:
    return sha256(("\n".join(values) + "\n").encode("ascii")).hexdigest()


@dataclass(frozen=True, slots=True)
class TargetConstructValidationTargetTaskFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-task-freeze'

    task_id: str
    target_id: str
    candidate_id: str
    package_name: str
    package_version: str
    solver_id: str
    denominator_value_ids: tuple[str, ...]
    history_value_ids: tuple[str, ...]
    native_action_value_ids: tuple[str, ...]
    hold_action_id: str
    receiver_value_ids: tuple[str, ...]
    horizon_value_ids: tuple[str, ...]
    legal_state_ids: tuple[str, ...]
    canary_unit_ids: tuple[str, ...]
    development_complete_unit_ids: tuple[str, ...]
    evaluation_complete_unit_ids: tuple[str, ...]
    reserve_complete_unit_ids: tuple[str, ...]
    unit_seed_sha256: str
    state_threshold_rule: str
    unit_timeout_seconds: int
    maximum_parallel_units: int
    authored_before_generator_response: bool
    generator_response_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "task_id",
            "target_id",
            "candidate_id",
            "package_name",
            "solver_id",
            "hold_action_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.package_version, field_name="package_version")
        for name in (
            "denominator_value_ids",
            "history_value_ids",
            "native_action_value_ids",
            "receiver_value_ids",
            "horizon_value_ids",
            "legal_state_ids",
            "canary_unit_ids",
            "development_complete_unit_ids",
            "evaluation_complete_unit_ids",
            "reserve_complete_unit_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        validate_sha256(self.unit_seed_sha256, field_name="unit_seed_sha256")
        validate_nonempty(self.state_threshold_rule, field_name="state_threshold_rule")
        if self.hold_action_id not in self.native_action_value_ids:
            raise ValueError("target task hold lies outside the action chart")
        rosters = (
            set(self.canary_unit_ids),
            set(self.development_complete_unit_ids),
            set(self.evaluation_complete_unit_ids),
            set(self.reserve_complete_unit_ids),
        )
        if any(first & second for i, first in enumerate(rosters) for second in rosters[i + 1 :]):
            raise ValueError("target task unit rosters overlap")
        if self.unit_timeout_seconds <= 0 or not 1 <= self.maximum_parallel_units <= 4:
            raise ValueError("target task resource envelope differs")
        if not self.authored_before_generator_response or self.generator_response_count:
            raise ValueError("target task was not frozen before generator response")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("target task freeze must remain outcome-blind")

    @property
    def cell_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                f"cell.{self.target_id}.{denominator}.{history}.{action}.{receiver}.{horizon}"
                for denominator, history, action, receiver, horizon in product(
                    self.denominator_value_ids,
                    self.history_value_ids,
                    self.native_action_value_ids,
                    self.receiver_value_ids,
                    self.horizon_value_ids,
                )
            )
        )


@dataclass(frozen=True, slots=True)
class TargetConstructValidationDonorFieldPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-donor-field-prediction'

    prediction_id: str
    predecessor: ObjectIdentity
    target_id: str
    cell_id: str
    donor_field: str
    donor_state_ids: tuple[str, ...]
    derivation_rule: str
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("prediction_id", "target_id", "cell_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_nonempty(self.donor_field, field_name="donor_field")
        validate_nonempty(self.derivation_rule, field_name="derivation_rule")
        require_sorted_unique_strings(
            self.donor_state_ids,
            field_name="donor_state_ids",
            allow_empty=False,
        )
        if self.protected_outcome_access_count:
            raise ValueError("donor field prediction cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("donor field prediction must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationTargetNativeBaseline(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-native-baseline'

    baseline_id: str
    target_id: str
    dependency_role_ids: tuple[str, ...]
    rule: str
    chosen_before_structural_recurrence_mapping: bool
    generator_response_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("baseline_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            tuple(sorted(self.dependency_role_ids)),
            field_name="dependency_role_ids_set",
            allow_empty=False,
        )
        validate_nonempty(self.rule, field_name="rule")
        if not self.chosen_before_structural_recurrence_mapping or self.generator_response_count:
            raise ValueError("target-native baseline crossed the predevelopment boundary")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("target-native baseline must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationTargetDesignFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-target-design-freeze'

    freeze_id: str
    task: TargetConstructValidationTargetTaskFreeze
    native_dossier: TargetConstructValidationTargetNativeDossier
    independence_dossier: TargetConstructValidationIndependenceDossier
    construct_threats: tuple[TargetConstructValidationConstructThreat, ...]
    construct_assessment: TargetConstructValidationConstructValidityAssessment
    construct_review: TargetConstructValidationConstructReviewAttestation
    relation_binding: TargetConstructValidationTargetRelationBinding
    mapping_candidates: tuple[TargetConstructValidationCompatibilityMapCandidate, ...]
    denominator_candidates: tuple[TargetConstructValidationDenominatorCandidate, ...]
    native_baseline: TargetConstructValidationTargetNativeBaseline
    comparator_plan: TargetConstructValidationComparatorPlan
    donor_predictions: tuple[TargetConstructValidationDonorFieldPrediction, ...]
    prediction_issue: TargetConstructValidationPredictionIssue
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.freeze_id, field_name="freeze_id")
        require_sorted_unique_ids(
            self.construct_threats,
            attribute="threat_id",
            field_name="construct_threats",
        )
        require_sorted_unique_ids(
            self.mapping_candidates,
            attribute="candidate_id",
            field_name="mapping_candidates",
        )
        require_sorted_unique_ids(
            self.denominator_candidates,
            attribute="candidate_id",
            field_name="denominator_candidates",
        )
        require_sorted_unique_ids(
            self.donor_predictions,
            attribute="prediction_id",
            field_name="donor_predictions",
        )
        if self.task.target_id != self.native_dossier.target_id:
            raise ValueError("target design task/dossier differ")
        if not self.construct_assessment.passed or not self.construct_review.review_passed:
            raise ValueError("target design cannot pass an unresolved construct review")
        if len(self.mapping_candidates) > 3:
            raise ValueError("target design mapping roster exceeds three")
        if {value.alternative_kind for value in self.denominator_candidates} != set(
            TargetConstructValidationDenominatorAlternativeKind
        ):
            raise ValueError("target design lacks the denominator lattice")
        if self.prediction_issue.target_id != self.task.target_id:
            raise ValueError("target design prediction names another target")
        if self.protected_outcome_access_count:
            raise ValueError("target design cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("target design freeze must remain outcome-blind")


def _source_candidate(candidate_id: str) -> ObjectIdentity:
    registry = frozen_source_qualification().nomination_result.candidate_registry
    candidate = next(
        value
        for value in frozen_nomination().registry.candidates
        if value.candidate_id == candidate_id
    )
    if registry.object_id != "target-construct-validation.candidate-registry":
        raise ValueError("source nomination registry differs")
    return ObjectIdentity.from_record(candidate.candidate_id, candidate)


def _unit_ids(target_key: str, phase: str, count: int) -> tuple[str, ...]:
    return tuple(f"unit.{target_key}.{phase}.{index:03d}" for index in range(1, count + 1))


def _task(target: str) -> TargetConstructValidationTargetTaskFreeze:
    if target == "cantera":
        target_id = 'target-cantera.stirred-reactor'
        candidate_id = "candidate.cantera-cstr"
        package_name = "cantera"
        package_version = "3.2.0"
        solver_id = "sundials-cvodes"
        denominators = ("denominator.residence-0p10s", "denominator.residence-0p20s")
        histories = ("history.initial-1100k", "history.initial-950k")
        actions = ("action.high-flow", "action.hold", "action.low-flow")
        receivers = ("receiver.co2-mole-fraction", "receiver.temperature")
        horizon = ("horizon.five-residence-times",)
        threshold = (
            "Within each complete unit and D/H/R cell, classify the final receiver minus "
            "the hold receiver as neutral when abs(delta) <= max(1e-10, 1e-8*abs(hold)), "
            "otherwise high for positive delta and low for negative delta."
        )
    elif target == "fipy":
        target_id = 'target-fipy.source-diffusion'
        candidate_id = "candidate.fipy-diffusion"
        package_name = "fipy"
        package_version = "4.0.3"
        solver_id = "fipy-scipy-linear-lu"
        denominators = ("denominator.diffusivity-0p02", "denominator.diffusivity-0p08")
        histories = ("history.reset-zero", "history.residual-gaussian")
        actions = ("action.high-source", "action.hold", "action.low-source")
        receivers = ("receiver.downstream-mean", "receiver.right-outward-flux")
        horizon = ("horizon.post-source-0p5s",)
        threshold = (
            "Within each complete unit and D/H/R cell, classify the final receiver minus "
            "the hold receiver as neutral when abs(delta) <= max(1e-12, 1e-9*abs(hold)), "
            "otherwise high for positive delta and low for negative delta."
        )
    else:  # pragma: no cover
        raise KeyError(target)
    target_key = target_id.replace("target-", "").replace(".", "-")
    all_units = (
        *_unit_ids(target_key, "canary", 2),
        *_unit_ids(target_key, "development", 24),
        *_unit_ids(target_key, "evaluation", 32),
        *_unit_ids(target_key, "reserve", 8),
    )
    return TargetConstructValidationTargetTaskFreeze(
        task_id=f"task-freeze.{target_id}",
        target_id=target_id,
        candidate_id=candidate_id,
        package_name=package_name,
        package_version=package_version,
        solver_id=solver_id,
        denominator_value_ids=denominators,
        history_value_ids=histories,
        native_action_value_ids=actions,
        hold_action_id="action.hold",
        receiver_value_ids=receivers,
        horizon_value_ids=horizon,
        legal_state_ids=LEGAL_STATE_IDS,
        canary_unit_ids=_unit_ids(target_key, "canary", 2),
        development_complete_unit_ids=_unit_ids(target_key, "development", 24),
        evaluation_complete_unit_ids=_unit_ids(target_key, "evaluation", 32),
        reserve_complete_unit_ids=_unit_ids(target_key, "reserve", 8),
        unit_seed_sha256=_digest_ids(tuple(sorted(all_units))),
        state_threshold_rule=threshold,
        unit_timeout_seconds=30,
        maximum_parallel_units=4,
        authored_before_generator_response=True,
        generator_response_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _dossiers(
    task: TargetConstructValidationTargetTaskFreeze,
) -> tuple[
    TargetConstructValidationTargetNativeDossier,
    TargetConstructValidationIndependenceDossier,
]:
    source = _source_candidate(task.candidate_id)
    cantera = task.package_name == "cantera"
    native = TargetConstructValidationTargetNativeDossier(
        dossier_id=f"dossier.{task.target_id}.activated",
        target_id=task.target_id,
        source_candidate=source,
        domain_id="chemical-reactor-kinetics" if cantera else "transport-pde",
        generator_family_id=task.package_name,
        task_statement=(
            "Measure final temperature and CO2 response of a reset open ideal-gas CSTR "
            "to bounded inlet-flow multipliers."
            if cantera
            else "Measure downstream field and outward-flux response of a reset transient "
            "diffusion-decay preparation to bounded localized source levels."
        ),
        preparation_unit_id=(
            "cantera-reset-cstr-condition-block" if cantera else "fipy-reset-field-condition-block"
        ),
        preparation_unit_definition=(
            "One seeded full reset followed by every frozen D/H/A/R/tau condition; the "
            "condition rows are nested views, never independent replicates."
        ),
        causal_cutoff_definition="All features and initial-state facts end before action onset.",
        native_action_ids=task.native_action_value_ids,
        hold_action_id=task.hold_action_id,
        receiver_ids=task.receiver_value_ids,
        receiver_direction_definition=(
            "Temperature and CO2 retain their increasing native gauges."
            if cantera
            else "Downstream field and outward right-boundary flux retain increasing gauges."
        ),
        horizon_ids=task.horizon_value_ids,
        native_unit_ids=(
            ("kelvin", "kilogram-per-second", "mole-fraction", "second")
            if cantera
            else ("field-amplitude", "field-flux", "second", "source-rate")
        ),
        native_frame_ids=(
            ("cstr-reactor-clock", "well-mixed-reactor-frame")
            if cantera
            else ("one-dimensional-domain", "pde-solver-clock")
        ),
        baseline_policy_id=(
            "baseline.cstr-residence-response" if cantera else "baseline.diffusion-source-response"
        ),
        decisive_falsifier_ids=(
            "falsifier.action-realization-mismatch",
            "falsifier.receiver-command-echo",
            "falsifier.reset-cross-unit-memory",
        ),
        requested_action_recorded=True,
        accepted_action_recorded=True,
        applied_action_recorded=True,
        realized_action_recorded=True,
        uses_structural_recurrence_role_vocabulary=False,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    other = "transport-pde" if cantera else "chemical-reactor-kinetics"
    independence = TargetConstructValidationIndependenceDossier(
        dossier_id=f"independence.{task.target_id}.activated",
        target_id=task.target_id,
        source_candidate=source,
        generator_family_id=task.package_name,
        authorship_evidence=f"{task.package_name} is maintained outside this project.",
        chronology_evidence=f"{task.package_name} and its native task predate target construct validation.",
        solver_family_evidence=(
            f"The native solver family is independent of the other target's {other} domain "
            "and solver implementation."
        ),
        generator_not_authored_for_construct_validation=True,
        generator_predates_construct_validation=True,
        domain_independent_of_other_target=True,
        solver_family_independent_of_other_target=True,
        generator_family_independent_of_other_target=True,
        direct_echo_of_structural_recurrence_fixture=False,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return native, independence


def _role_bindings(target_id: str) -> tuple[TargetConstructValidationRoleBinding, ...]:
    native = {
        "A": ("native-action", "requested-accepted-applied-realized-ledger"),
        "D": ("native-denominator",),
        "H": ("pre-action-initial-state",),
        "R": ("native-receiver-gauge",),
        "tau": ("native-solver-horizon",),
    }
    return tuple(
        TargetConstructValidationRoleBinding(
            binding_id=f"binding.{target_id}.{role.lower()}",
            structural_recurrence_role_id=role,
            target_native_role_ids=native[role],
            constructor=TargetConstructValidationMappingConstructor.DIRECT_NATIVE_ROLE_BINDING,
            native_units_preserved=True,
            native_frame_preserved=True,
            receiver_direction_preserved=True,
            causal_cutoff_preserved=True,
            action_stage_semantics_preserved=True,
            outcome_derived=False,
        )
        for role in ("A", "D", "H", "R", "tau")
    )


def _expected_state(task: TargetConstructValidationTargetTaskFreeze, action_id: str) -> str:
    if action_id == task.hold_action_id:
        return "state.neutral"
    if task.package_name == "cantera":
        return "state.low" if action_id == "action.high-flow" else "state.high"
    return "state.high" if action_id == "action.high-source" else "state.low"


def build_target_design_freeze(target: str) -> TargetConstructValidationTargetDesignFreeze:
    task = _task(target)
    native, independence = _dossiers(task)
    assessment = assess_construct_validity(native, independence)
    threats = tuple(
        sorted(
            (
                TargetConstructValidationConstructThreat(
                    threat_id=f"threat.{task.target_id}.command-echo",
                    target_id=task.target_id,
                    threat_kind="command-echo",
                    description="Receiver categories must derive from simulated receiver values, not commands.",
                    decisive_case_ids=task.cell_ids,
                    blocks_claim_if_observed=True,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                ),
                TargetConstructValidationConstructThreat(
                    threat_id=f"threat.{task.target_id}.unit-reset",
                    target_id=task.target_id,
                    threat_kind="cross-unit-memory",
                    description="Every complete unit must rebuild all mutable simulator state.",
                    decisive_case_ids=task.cell_ids,
                    blocks_claim_if_observed=True,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                ),
            ),
            key=lambda value: value.threat_id,
        )
    )
    review = TargetConstructValidationConstructReviewAttestation(
        attestation_id=f"construct-review.{task.target_id}",
        target_id=task.target_id,
        assessment=ObjectIdentity.from_record(assessment.assessment_id, assessment),
        dossier_author_id="role.target-construct-validation-dossier-author",
        mapping_author_id="role.target-construct-validation-mapping-author",
        projector_owner_id="role.target-construct-validation-projector-owner",
        evaluator_owner_id="role.target-construct-validation-evaluator-owner",
        reviewer_id="capability.target-construct-validation-rule-construct-reviewer",
        reviewer_role="Deterministic outcome-blind rule reviewer; not an independent human investigator.",
        reviewed_case_ids=task.cell_ids,
        exposure_ids=("exposure.target-native-metadata",),
        conflict_ids=(),
        review_passed=True,
        review_reason=(
            "All frozen role, unit, cutoff, receiver and action-chain checks pass; this is "
            "architectural role separation only and lowers the evidence ceiling."
        ),
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    relation = primary_relation_codebook()
    relation_binding = TargetConstructValidationTargetRelationBinding(
        binding_id=f"relation-binding.{task.target_id}",
        target_id=task.target_id,
        relation_codebook=ObjectIdentity.from_record(relation.codebook_id, relation),
        primary_relation_id=TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,
        claim_bearing_cell_ids=task.cell_ids,
        diagnostic_relation_ids=(),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    mapping = TargetConstructValidationCompatibilityMapCandidate(
        candidate_id=f"mapping.{task.target_id}.direct-native",
        target_id=task.target_id,
        finite_roster_id=f"mapping-roster.{task.target_id}",
        native_dossier=ObjectIdentity.from_record(native.dossier_id, native),
        role_bindings=_role_bindings(task.target_id),
        semantic_losses=(),
        requested_accepted_applied_realized_distinct=True,
        target_native_action_chart_preserved=True,
        authored_before_development=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    factors = tuple(
        sorted(
            (
                "factor.residence-time" if target == "cantera" else "factor.diffusivity",
                "factor.initial-history",
            )
        )
    )
    denominators = tuple(
        sorted(
            (
                TargetConstructValidationDenominatorCandidate(
                    candidate_id=f"denominator.{task.target_id}.proposed",
                    target_id=task.target_id,
                    alternative_kind=TargetConstructValidationDenominatorAlternativeKind.PROPOSED,
                    factor_ids=factors,
                    stratum_ids=task.denominator_value_ids,
                    causal_loss_ids=(),
                    impossible=False,
                    impossible_reason=None,
                    authored_before_development=True,
                ),
                TargetConstructValidationDenominatorCandidate(
                    candidate_id=f"denominator.{task.target_id}.split",
                    target_id=task.target_id,
                    alternative_kind=TargetConstructValidationDenominatorAlternativeKind.SPLIT,
                    factor_ids=tuple(sorted((*factors, "factor.receiver-specific-stratum"))),
                    stratum_ids=tuple(
                        sorted(
                            f"{d}.{r}"
                            for d, r in product(task.denominator_value_ids, task.receiver_value_ids)
                        )
                    ),
                    causal_loss_ids=(),
                    impossible=False,
                    impossible_reason=None,
                    authored_before_development=True,
                ),
                TargetConstructValidationDenominatorCandidate(
                    candidate_id=f"denominator.{task.target_id}.merge",
                    target_id=task.target_id,
                    alternative_kind=TargetConstructValidationDenominatorAlternativeKind.MERGE,
                    factor_ids=(factors[1],),
                    stratum_ids=("denominator.merged",),
                    causal_loss_ids=("causal-loss.denominator-factor",),
                    impossible=False,
                    impossible_reason=None,
                    authored_before_development=True,
                ),
                TargetConstructValidationDenominatorCandidate(
                    candidate_id=f"denominator.{task.target_id}.omission",
                    target_id=task.target_id,
                    alternative_kind=TargetConstructValidationDenominatorAlternativeKind.OMISSION,
                    factor_ids=(),
                    stratum_ids=(),
                    causal_loss_ids=("causal-loss.denominator-and-history",),
                    impossible=True,
                    impossible_reason="Omission destroys the declared D/H conditioning.",
                    authored_before_development=True,
                ),
            ),
            key=lambda value: value.candidate_id,
        )
    )
    native_baseline = TargetConstructValidationTargetNativeBaseline(
        baseline_id=f"native-baseline.{task.target_id}",
        target_id=task.target_id,
        dependency_role_ids=("A",),
        rule=(
            "Native monotone residence-time baseline: low flow raises and high flow lowers "
            "both final receivers relative to hold."
            if target == "cantera"
            else "Native linear source baseline: lower source lowers and higher source raises "
            "both final receivers relative to hold."
        ),
        chosen_before_structural_recurrence_mapping=True,
        generator_response_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    comparator_plan = TargetConstructValidationComparatorPlan(
        plan_id=f"comparator-plan.{task.target_id}",
        target_id=task.target_id,
        comparator_kinds=tuple(sorted(TargetConstructValidationComparatorKind, key=lambda value: value.value)),
        key_role_ids=("A", "D", "H", "R", "tau"),
        target_native_baseline_id=native_baseline.baseline_id,
        target_native_baseline_specification=ObjectIdentity.from_record(
            native_baseline.baseline_id,
            native_baseline,
        ),
        primary_cell_ids=task.cell_ids,
        evaluation_complete_unit_ids_sha256=_digest_ids(task.evaluation_complete_unit_ids),
        baseline_chosen_before_structural_recurrence_mapping=True,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    predecessor = method_source_manifest()
    predecessor_identity = ObjectIdentity.from_record(predecessor.manifest_id, predecessor)
    donor_predictions = []
    emissions = []
    cells = []
    for index, cell_id in enumerate(task.cell_ids):
        action_id = next(value for value in task.native_action_value_ids if value in cell_id)
        denominator_id = next(value for value in task.denominator_value_ids if value in cell_id)
        history_id = next(value for value in task.history_value_ids if value in cell_id)
        receiver_id = next(value for value in task.receiver_value_ids if value in cell_id)
        horizon_id = task.horizon_value_ids[0]
        cell = TargetConstructValidationDirectForecastCell(
            cell_id=cell_id,
            target_id=task.target_id,
            primary_relation_id=TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,
            denominator_stratum_id=denominator_id,
            history_condition_id=history_id,
            native_action_id=action_id,
            receiver_id=receiver_id,
            horizon_id=horizon_id,
            legal_state_ids=task.legal_state_ids,
            unsafe_admission_state_ids=(),
            complete_unit_ids_sha256=_digest_ids(task.evaluation_complete_unit_ids),
            authored_before_development=True,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        state = _expected_state(task, action_id)
        donor_state = {
            "state.high": "donor.act-positive",
            "state.low": "donor.act-negative",
            "state.neutral": "donor.hold",
        }[state]
        prediction = TargetConstructValidationDonorFieldPrediction(
            prediction_id=f"donor-prediction.{task.target_id}.{index:02d}",
            predecessor=predecessor_identity,
            target_id=task.target_id,
            cell_id=cell_id,
            donor_field="frozen-categorical-action-fibre-state",
            donor_state_ids=(donor_state,),
            derivation_rule=(
                "Use only the frozen native action orientation and mandatory hold; D/H/R/tau "
                "remain explicit conditioning coordinates and add no fitted target content."
            ),
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        emission = TargetConstructValidationForecastEmission(
            emission_id=f"forecast-emission.{task.target_id}.{index:02d}",
            cell=ObjectIdentity.from_record(cell.cell_id, cell),
            donor_prediction=ObjectIdentity.from_record(prediction.prediction_id, prediction),
            donor_field=prediction.donor_field,
            selected_map=ObjectIdentity.from_record(mapping.candidate_id, mapping),
            role_binding_ids=tuple(value.binding_id for value in mapping.role_bindings),
            projections=(
                TargetConstructValidationStateProjection(
                    projection_id=f"state-projection.{task.target_id}.{index:02d}",
                    donor_state_id=donor_state,
                    target_state_id=state,
                ),
            ),
            donor_state_ids=(donor_state,),
            emitted_state_ids=(state,),
            disposition=TargetConstructValidationForecastDisposition.EXACT,
            derivation_rule="One frozen donor state maps to one target-native categorical state.",
            rationale="Direct forecast contains no development or evaluation target value.",
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        cells.append(cell)
        donor_predictions.append(prediction)
        emissions.append(emission)
    prediction_issue = TargetConstructValidationPredictionIssue(
        issue_id=f"prediction-issue.{task.target_id}",
        target_id=task.target_id,
        relation_binding=ObjectIdentity.from_record(relation_binding.binding_id, relation_binding),
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        emissions=tuple(sorted(emissions, key=lambda value: value.emission_id)),
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return TargetConstructValidationTargetDesignFreeze(
        freeze_id=f"design-freeze.{task.target_id}",
        task=task,
        native_dossier=native,
        independence_dossier=independence,
        construct_threats=threats,
        construct_assessment=assessment,
        construct_review=review,
        relation_binding=relation_binding,
        mapping_candidates=(mapping,),
        denominator_candidates=denominators,
        native_baseline=native_baseline,
        comparator_plan=comparator_plan,
        donor_predictions=tuple(sorted(donor_predictions, key=lambda value: value.prediction_id)),
        prediction_issue=prediction_issue,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def frozen_target_designs() -> tuple[TargetConstructValidationTargetDesignFreeze, ...]:
    return tuple(
        sorted(
            (build_target_design_freeze("cantera"), build_target_design_freeze("fipy")),
            key=lambda value: value.task.target_id,
        )
    )


__all__ = [
    'TargetConstructValidationDonorFieldPrediction',
    'TargetConstructValidationTargetDesignFreeze',
    'TargetConstructValidationTargetNativeBaseline',
    'TargetConstructValidationTargetTaskFreeze',
    "LEGAL_STATE_IDS",
    "build_target_design_freeze",
    "frozen_target_designs",
]

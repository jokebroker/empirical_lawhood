"""Closed truth-known adversarial conformance for common target construct validation contracts."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from hashlib import sha256
from typing import Callable, ClassVar

from empirical_lawhood.adapters.methods.observed_structural_classes import EvidenceWorld
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)

from .comparators import adjudicate_restrictiveness
from .construct_validity import TargetConstructValidationConstructReviewAttestation
from .contracts import TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID, primary_relation_codebook
from .cross_target import TargetConstructValidationRecurrenceDisposition, adjudicate_cross_target
from .donor import current_donor_binding
from .mapping import select_mapping_candidate, select_minimal_denominator
from .native_dossier import TargetConstructValidationTargetNativeDossier
from .nomination import TargetConstructValidationCandidate, TargetConstructValidationCandidateRegistry, select_nominees
from .protocol import TargetConstructValidationProgressPhase, TargetConstructValidationProgressRecord
from .target_adjudication import TargetConstructValidationConstructAxis, TargetConstructValidationPredictionAxis, TargetConstructValidationRestrictivenessAxis, TargetConstructValidationTargetAdjudication, TargetConstructValidationTargetHandoff, TargetConstructValidationTopologyAxis


class TargetConstructValidationConformanceCaseKind(StrEnum):
    COMPARATOR_ROSTER_REFUSAL = "COMPARATOR_ROSTER_REFUSAL"
    COUNTEREXAMPLE_PRECEDENCE = "COUNTEREXAMPLE_PRECEDENCE"
    DENOMINATOR_EMPTY_STOP = "DENOMINATOR_EMPTY_STOP"
    DONOR_CURRENT_BINDING = "DONOR_CURRENT_BINDING"
    MAPPING_EMPTY_STOP = "MAPPING_EMPTY_STOP"
    NATIVE_ACTION_STAGE_REFUSAL = "NATIVE_ACTION_STAGE_REFUSAL"
    NOMINATION_INDEPENDENCE_STOP = "NOMINATION_INDEPENDENCE_STOP"
    PROGRESS_HORIZON_SEPARATION = "PROGRESS_HORIZON_SEPARATION"
    RELATION_MULTIPLICITY_REFUSAL = "RELATION_MULTIPLICITY_REFUSAL"
    REVIEWER_IDENTITY_SEPARATION = "REVIEWER_IDENTITY_SEPARATION"
    TARGET_AXIS_COUPLING_REFUSAL = "TARGET_AXIS_COUPLING_REFUSAL"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationConformanceCase(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-conformance-case'

    case_id: str
    case_kind: TargetConstructValidationConformanceCaseKind
    expected_code: str
    observed_code: str
    passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.case_id, field_name="case_id")
        validate_nonempty(self.expected_code, field_name="expected_code")
        validate_nonempty(self.observed_code, field_name="observed_code")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.passed == bool(self.reason_codes):
            raise ValueError("conformance reasons must be present exactly on failure")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationConformanceReport(CanonicalRecord):
    """Truth-known method evidence only; never target recurrence evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-conformance-report'

    report_id: str
    method_question_freeze: ObjectIdentity
    evidence_world: EvidenceWorld
    case_results: tuple[TargetConstructValidationConformanceCase, ...]
    independent_of_target_adapters: bool
    protected_outcome_access_count: int
    all_passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        require_sorted_unique_ids(
            self.case_results,
            attribute="case_id",
            field_name="case_results",
        )
        if {value.case_kind for value in self.case_results} != set(TargetConstructValidationConformanceCaseKind):
            raise ValueError("conformance report lacks the exact adversarial roster")
        if self.evidence_world is not EvidenceWorld.TRUTH_KNOWN_GENERATED:
            raise ValueError("conformance report must remain truth-known generated")
        if not self.independent_of_target_adapters:
            raise ValueError("common conformance cannot be authored by a target adapter")
        if self.protected_outcome_access_count:
            raise ValueError("truth-known conformance cannot access protected outcomes")
        if self.all_passed != all(value.passed for value in self.case_results):
            raise ValueError("conformance report status is not case-derived")


def _identity(object_id: str) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema='empirical-lawhood/methods/target-construct-validation/synthetic-conformance-object',
        object_version="1.0.0",
        object_fingerprint=sha256(object_id.encode("ascii")).hexdigest(),
    )


def _native() -> TargetConstructValidationTargetNativeDossier:
    return TargetConstructValidationTargetNativeDossier(
        dossier_id="conformance.target.native-dossier",
        target_id="conformance-target",
        source_candidate=_identity("conformance.source"),
        domain_id="conformance-domain",
        generator_family_id="conformance-generator",
        task_statement="Truth-known target-native response fixture.",
        preparation_unit_id="episode",
        preparation_unit_definition="One complete generated episode.",
        causal_cutoff_definition="All features stop before the request clock.",
        native_action_ids=("act", "hold"),
        hold_action_id="hold",
        receiver_ids=("receiver-a", "receiver-b"),
        receiver_direction_definition="Positive response enters the receiver.",
        horizon_ids=("relaxed", "transient"),
        native_unit_ids=("native-unit",),
        native_frame_ids=("native-frame",),
        baseline_policy_id="native-baseline",
        decisive_falsifier_ids=("action-echo",),
        requested_action_recorded=True,
        accepted_action_recorded=True,
        applied_action_recorded=True,
        realized_action_recorded=True,
        uses_structural_recurrence_role_vocabulary=False,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _candidate(
    candidate_id: str,
    *,
    domain_id: str,
) -> TargetConstructValidationCandidate:
    return TargetConstructValidationCandidate(
        candidate_id=candidate_id,
        domain_id=domain_id,
        solver_family_id="shared-solver",
        generator_family_id=f"{candidate_id}-generator",
        task_statement="Truth-known metadata-only intervention task.",
        generator_provenance="Generated conformance metadata.",
        specification_provenance="Generated pre-outcome specification.",
        preparation_unit_definition="One complete generated episode.",
        native_intervention_definition="A finite generated action.",
        action_realization_chain="Requested, accepted, applied and realized.",
        receiver_definitions=("Receiver A.", "Receiver B."),
        horizon_definitions=("Relaxed.", "Transient."),
        causal_cutoff_definition="Features stop before action.",
        likely_denominator_factor_ids=("boundary",),
        likely_history_factor_ids=("history",),
        source_cost_class=0,
        execution_cost_class=0,
        prior_project_exposure_ids=(),
        prior_investigator_exposure_ids=(),
        construct_threat_ids=("echo",),
        maximum_claim="Truth-known method conformance only.",
        construct_non_tautological=True,
        task_specification_independent=True,
        new_generator_family=True,
        complete_action_realization_observable=True,
        independent_units_defensible=True,
        nontrivial_contrasts_available=True,
        sealed_evaluation_feasible=True,
        joint_power_feasible=True,
        source_metadata_complete=True,
        recurrence_eligible=True,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def _handoff(
    target_id: str,
    *,
    domain_id: str,
    solver_id: str,
    generator_id: str,
    prediction: TargetConstructValidationPredictionAxis,
    counterexample: bool,
) -> TargetConstructValidationTargetHandoff:
    return TargetConstructValidationTargetHandoff(
        handoff_id=f"{target_id}.handoff",
        target_id=target_id,
        domain_id=domain_id,
        solver_family_id=solver_id,
        generator_family_id=generator_id,
        primary_relation_id=TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,
        relation_binding=_identity(f"{target_id}.relation-binding"),
        target_adjudication=_identity(f"{target_id}.adjudication"),
        construct_axis=TargetConstructValidationConstructAxis.PASSED,
        prediction_axis=prediction,
        restrictiveness_axis=TargetConstructValidationRestrictivenessAxis.SUPPORTED,
        topology_axis=TargetConstructValidationTopologyAxis.EQUIVALENT_OR_UNEVALUABLE,
        exact_construct_valid_counterexample=counterexample,
        eligible_for_cross_target=True,
        reason_codes=(),
        native_numeric_value_count=0,
        raw_payload_reference_count=0,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
    )


def execute_truth_known_conformance(
    *,
    method_question_freeze: ObjectIdentity,
) -> TargetConstructValidationConformanceReport:
    """Execute the closed common-contract matrix without target adapters."""

    results: list[TargetConstructValidationConformanceCase] = []

    def record(
        kind: TargetConstructValidationConformanceCaseKind,
        *,
        expected: str,
        observed: str,
        passed: bool,
    ) -> None:
        results.append(
            TargetConstructValidationConformanceCase(
                case_id=f"target-construct-validation.truth-known-conformance.{kind.value.lower().replace('_', '-')}",
                case_kind=kind,
                expected_code=expected,
                observed_code=observed,
                passed=passed,
                reason_codes=() if passed else ("TRUTH_KNOWN_EXPECTATION_MISMATCH",),
            )
        )

    def refusal(
        kind: TargetConstructValidationConformanceCaseKind,
        expected: str,
        action: Callable[[], object],
    ) -> None:
        try:
            action()
        except ValueError as error:
            observed = str(error)
            record(kind, expected=expected, observed=observed, passed=expected in observed)
        else:
            record(kind, expected=expected, observed="NO_REFUSAL", passed=False)

    donor = current_donor_binding()
    donor_passed = (
        len(donor.exact_frozen_operation_ids) == 4
        and len(donor.additive_boundary_operation_ids) == 1
    )
    record(
        TargetConstructValidationConformanceCaseKind.DONOR_CURRENT_BINDING,
        expected="FOUR_EXACT_ONE_ADDITIVE",
        observed=(
            "FOUR_EXACT_ONE_ADDITIVE"
            if donor_passed
            else (
                f"{len(donor.exact_frozen_operation_ids)}_EXACT_"
                f"{len(donor.additive_boundary_operation_ids)}_ADDITIVE"
            )
        ),
        passed=donor_passed,
    )
    refusal(
        TargetConstructValidationConformanceCaseKind.RELATION_MULTIPLICITY_REFUSAL,
        "exactly one",
        lambda: replace(
            primary_relation_codebook(),
            claim_bearing_relation_ids=(TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID, "rescue"),
        ),
    )
    refusal(
        TargetConstructValidationConformanceCaseKind.NATIVE_ACTION_STAGE_REFUSAL,
        "four action stages",
        lambda: replace(_native(), realized_action_recorded=False),
    )
    refusal(
        TargetConstructValidationConformanceCaseKind.REVIEWER_IDENTITY_SEPARATION,
        "not independent",
        lambda: TargetConstructValidationConstructReviewAttestation(
            attestation_id="conformance.review",
            target_id="conformance-target",
            assessment=_identity("conformance.assessment"),
            dossier_author_id="author",
            mapping_author_id="mapper",
            projector_owner_id="projector",
            evaluator_owner_id="evaluator",
            reviewer_id="mapper",
            reviewer_role="Independent reviewer",
            reviewed_case_ids=("case-a",),
            exposure_ids=(),
            conflict_ids=(),
            review_passed=True,
            review_reason="Truth-known review.",
            protected_outcome_access_count=0,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        ),
    )
    refusal(
        TargetConstructValidationConformanceCaseKind.MAPPING_EMPTY_STOP,
        "MAPPING_UNRESOLVED",
        lambda: select_mapping_candidate(()),
    )
    refusal(
        TargetConstructValidationConformanceCaseKind.DENOMINATOR_EMPTY_STOP,
        "DENOMINATOR_UNRESOLVED",
        lambda: select_minimal_denominator(()),
    )
    refusal(
        TargetConstructValidationConformanceCaseKind.COMPARATOR_ROSTER_REFUSAL,
        "COMPARATOR_FAMILY_INCOMPLETE",
        lambda: adjudicate_restrictiveness(()),
    )
    refusal(
        TargetConstructValidationConformanceCaseKind.TARGET_AXIS_COUPLING_REFUSAL,
        "prediction is unevaluable",
        lambda: TargetConstructValidationTargetAdjudication(
            adjudication_id="conformance.target-adjudication",
            target_id="conformance-target",
            primary_relation_id=TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID,
            construct_axis=TargetConstructValidationConstructAxis.FAILED,
            prediction_axis=TargetConstructValidationPredictionAxis.SUPPORTED,
            restrictiveness_axis=TargetConstructValidationRestrictivenessAxis.SUPPORTED,
            topology_axis=TargetConstructValidationTopologyAxis.EQUIVALENT_OR_UNEVALUABLE,
            exact_construct_valid_counterexample=False,
            complete_unit_inference_closed=True,
            power_adequate=True,
            reason_codes=("construct-failed",),
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        ),
    )
    progress = TargetConstructValidationProgressRecord(
        progress_id="conformance.progress",
        target_id="conformance-target",
        task_id="conformance-task",
        phase=TargetConstructValidationProgressPhase.DEVELOPMENT,
        complete_units_finished=0,
        complete_units_total=1,
        elapsed_milliseconds=0,
        cpu_milliseconds=0,
        rss_high_water_bytes=0,
        scratch_bytes=0,
        scientific_horizon_value_count=0,
    )
    refusal(
        TargetConstructValidationConformanceCaseKind.PROGRESS_HORIZON_SEPARATION,
        "scientific horizon",
        lambda: replace(progress, scientific_horizon_value_count=1),
    )
    registry = TargetConstructValidationCandidateRegistry(
        registry_id="conformance.registry",
        candidates=tuple(
            sorted(
                (
                    _candidate("candidate-a", domain_id="domain-a"),
                    _candidate("candidate-b", domain_id="domain-b"),
                ),
                key=lambda value: value.candidate_id,
            )
        ),
        frozen_before_candidate_outcomes=True,
        candidate_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    refusal(
        TargetConstructValidationConformanceCaseKind.NOMINATION_INDEPENDENCE_STOP,
        "INSUFFICIENT_INDEPENDENT_TARGETS",
        lambda: select_nominees(registry),
    )
    first = _handoff(
        "target-a",
        domain_id="domain-a",
        solver_id="solver-a",
        generator_id="generator-a",
        prediction=TargetConstructValidationPredictionAxis.SUPPORTED,
        counterexample=False,
    )
    second = _handoff(
        "target-b",
        domain_id="domain-b",
        solver_id="solver-b",
        generator_id="generator-b",
        prediction=TargetConstructValidationPredictionAxis.OPPOSED,
        counterexample=True,
    )
    cross = adjudicate_cross_target(first, second)
    record(
        TargetConstructValidationConformanceCaseKind.COUNTEREXAMPLE_PRECEDENCE,
        expected=TargetConstructValidationRecurrenceDisposition.OPPOSED.value,
        observed=cross.recurrence_disposition.value,
        passed=cross.recurrence_disposition is TargetConstructValidationRecurrenceDisposition.OPPOSED,
    )

    ordered = tuple(sorted(results, key=lambda value: value.case_id))
    return TargetConstructValidationConformanceReport(
        report_id="target-construct-validation.truth-known-conformance.truth-known-conformance",
        method_question_freeze=method_question_freeze,
        evidence_world=EvidenceWorld.TRUTH_KNOWN_GENERATED,
        case_results=ordered,
        independent_of_target_adapters=True,
        protected_outcome_access_count=0,
        all_passed=all(value.passed for value in ordered),
    )

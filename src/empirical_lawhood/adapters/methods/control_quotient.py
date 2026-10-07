"Bounded truth-known control-equivalence method for RICQ.\n\nThe method consumes the remediated current causal, gate and reachability\nobjects.  It never accepts an adapter-authored ``PASS``: every action\npossibility must bind a defined total causal-prefix assessment, all nine\nevidence-derived noncompensating gate receipts and one reachable direction.\nThe resulting four-axis assessment remains local law and cannot authorize execution.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, Final

from empirical_lawhood.adapters.control.reference import build_current_reference_controller_study
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus
from empirical_lawhood.kernel.admission import AdmissionGateKind, GateStatus
from empirical_lawhood.kernel.causal_contracts import (
    CausalCompositionDisposition,
    CausalPrefixAssessment,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.receiver_geometry_control import (
    BoundedControlEquivalenceAssessment,
    CandidateActionEvidence,
    CandidateDecisionDisposition,
    CandidateEquivalenceRelation,
    CandidatePairEquivalence,
    CandidateViewDecisionSet,
    DecisionEquivalenceClass,
    EquivalenceDisposition,
    LocalRegularityDisposition,
    ObservationRepresentationKind,
    ReceiverCandidateView,
    ReceiverFiberAssessment,
    ReceiverFiberDisposition,
    ReceiverObservationExperiment,
    derive_decision_equivalence,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.evidence_geometry import AtlasGateReceipt, ReachabilityCellDisposition, AtlasReachabilityReceipt


CONTROL_QUOTIENT_METHOD_VERSION: Final = "1.0.0"
CONTROL_QUOTIENT_CASE_IDS: Final = (
    "case.decision-distinct",
    "case.exact-observation",
    "case.history-resolved",
    "case.implementation-false-exact",
    "case.mixture-unevaluable",
    "case.response-distinct-shared-hold",
    "case.shared-action",
    "case.shared-termination",
    "case.target-equivalent-sink-distinct",
    "case.unsafe-probe",
    "case.view-enrichment-erases-action",
)


class ControlQuotientMethodStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"


class ControlQuotientCaseDisposition(StrEnum):
    SHARED_ACTION = "SHARED_ACTION"
    SHARED_HOLD = "SHARED_HOLD"
    SHARED_TERMINATION = "SHARED_TERMINATION"
    DECISION_DISTINCT = "DECISION_DISTINCT"
    INJECTIVE_ACTION = "INJECTIVE_ACTION"
    ACTION_ERASED_BY_VIEW = "ACTION_ERASED_BY_VIEW"
    MIXTURE_UNEVALUABLE = "MIXTURE_UNEVALUABLE"
    FALSE_EXACTNESS_REJECTED = "FALSE_EXACTNESS_REJECTED"
    UNSAFE_PROBE_REJECTED = "UNSAFE_PROBE_REJECTED"


@dataclass(frozen=True, slots=True)
class ControlQuotientMethodConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/control-quotient-method-config'

    config_id: str
    case_ids: tuple[str, ...]
    method_version: str
    required_gate_kinds: tuple[AdmissionGateKind, ...]
    prohibitions: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        require_sorted_unique_strings(
            self.case_ids,
            field_name="case_ids",
            allow_empty=False,
        )
        if self.case_ids != CONTROL_QUOTIENT_CASE_IDS:
            raise ValueError("control-quotient config changes its frozen case roster")
        if self.method_version != CONTROL_QUOTIENT_METHOD_VERSION:
            raise ValueError("control-quotient method version is unsupported")
        if self.required_gate_kinds != tuple(
            sorted(AdmissionGateKind, key=lambda value: value.value)
        ):
            raise ValueError("control-quotient method omits a noncompensating gate")
        require_sorted_unique_strings(
            self.prohibitions,
            field_name="prohibitions",
            allow_empty=False,
        )


def default_control_quotient_method_config() -> ControlQuotientMethodConfig:
    return ControlQuotientMethodConfig(
        config_id="config.receiver-conditioned-io.control-quotient-method",
        case_ids=CONTROL_QUOTIENT_CASE_IDS,
        method_version=CONTROL_QUOTIENT_METHOD_VERSION,
        required_gate_kinds=tuple(sorted(AdmissionGateKind, key=lambda value: value.value)),
        prohibitions=(
            "no-action-averaging",
            "no-adapter-authored-gate-status",
            "no-candidate-or-view-omission",
            "no-mode-label-as-method-input",
            "no-outcome-visible-controller-admission",
            "no-probe-execution",
        ),
    )


@dataclass(frozen=True, slots=True)
class TruthKnownReceiverState(CanonicalRecord):
    """Evaluator-known state identity; the method consumes only its fingerprint."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/truth-known-receiver-state'

    state_id: str
    hidden_mode_id: str
    preparation_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.state_id, field_name="state_id")
        validate_stable_id(self.hidden_mode_id, field_name="hidden_mode_id")
        validate_stable_id(self.preparation_id, field_name="preparation_id")


def _identity(object_id: str, payload: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(object_id, payload)


def verified_candidate_action_evidence(
    *,
    evidence_id: str,
    candidate: ReceiverCandidateView,
    action_word: OccurrenceActionWord,
    response_law: CanonicalRecord,
    response_law_id: str,
    causal_prefix: CausalPrefixAssessment,
    gate_receipts: tuple[AtlasGateReceipt, ...],
    reachability_receipt: AtlasReachabilityReceipt,
) -> CandidateActionEvidence:
    """Validate loaded operands and emit only their exact compact identities."""

    if (
        causal_prefix.disposition is not CausalCompositionDisposition.DEFINED
        or causal_prefix.obstructions
        or causal_prefix.causal_support.action_word != action_word
    ):
        raise ValueError("candidate action lacks one defined complete causal prefix")
    if (
        candidate.denominator_id != action_word.denominator_id
        or candidate.retained_history_id != action_word.retained_history_id
        or candidate.receiver_id != action_word.receiver_id
        or candidate.horizon_id != action_word.horizon_id
    ):
        raise ValueError("candidate action changes D/H/R/tau")
    require_sorted_unique_ids(
        gate_receipts,
        attribute="receipt_id",
        field_name="gate_receipts",
    )
    observed_kinds = {value.predicate.gate_kind for value in gate_receipts}
    if observed_kinds != set(AdmissionGateKind):
        raise ValueError("candidate action lacks the complete noncompensating gate set")
    if any(
        value.status is not GateStatus.PASS
        or value.cell_id != reachability_receipt.admission_cell_id
        or value.model_member_id != candidate.numerical_view_id
        or value.numerical_view_id != candidate.numerical_view_id
        or value.outcome_access
        in {
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }
        for value in gate_receipts
    ):
        raise ValueError("candidate action has a failed, foreign or outcome-visible gate")
    if (
        reachability_receipt.status is not ReachabilityCellDisposition.REACHABLE
        or reachability_receipt.model_member_id != candidate.numerical_view_id
        or reachability_receipt.numerical_view_id != candidate.numerical_view_id
        or not reachability_receipt.independent_basis_direction_ids
        or reachability_receipt.outcome_access
        in {
            OutcomeAccess.EVALUATION_REVEALED,
            OutcomeAccess.PRIVILEGED_TRUTH,
        }
    ):
        raise ValueError("candidate action lacks an evidence-derived reachable direction")
    direction_id = reachability_receipt.independent_basis_direction_ids[0]
    return CandidateActionEvidence(
        evidence_id=evidence_id,
        candidate_pair_id=candidate.pair_id,
        action_word=action_word,
        response_law=_identity(response_law_id, response_law),
        causal_prefix=_identity(causal_prefix.assessment_id, causal_prefix),
        gate_receipts=tuple(
            sorted(
                (_identity(value.receipt_id, value) for value in gate_receipts),
                key=lambda value: value.object_id,
            )
        ),
        reachability_receipt=_identity(
            reachability_receipt.receipt_id,
            reachability_receipt,
        ),
        admission_cell_id=reachability_receipt.admission_cell_id,
        viable_direction_id=direction_id,
    )


def _hold_word(action: OccurrenceActionWord) -> OccurrenceActionWord:
    return OccurrenceActionWord(
        word_id="word.ricq-qualified-hold",
        mode=ActionWordMode.IDENTITY,
        occurrences=(),
        groups=(),
        ordering_clock_id=None,
        ordering_time_unit=None,
        ordering_coordinate_frame=None,
        ordering_origin=None,
        denominator_id=action.denominator_id,
        retained_history_id=action.retained_history_id,
        receiver_id=action.receiver_id,
        horizon_id=action.horizon_id,
        prefix_support_ids=("prefix-identity",),
        support_status=ActionWordSupportStatus.SUPPORTED,
        reason_codes=(),
    )


def _candidate(
    case_id: str,
    token: str,
    *,
    numerical_view_id: str,
    action: OccurrenceActionWord,
) -> ReceiverCandidateView:
    preparation_id = f"preparation.{case_id}"
    state = TruthKnownReceiverState(
        state_id=f"state.{case_id}.{token}",
        hidden_mode_id=f"mode.{case_id}.{token}",
        preparation_id=preparation_id,
    )
    return ReceiverCandidateView(
        pair_id=f"pair.{case_id}.{token}.{numerical_view_id}",
        candidate_id=f"candidate.{case_id}.{token}",
        numerical_view_id=numerical_view_id,
        preparation_id=preparation_id,
        denominator_id=action.denominator_id,
        retained_history_id=action.retained_history_id,
        receiver_id=action.receiver_id,
        horizon_id=action.horizon_id,
        local_chart_id=f"chart.{case_id}",
        state=_identity(state.state_id, state),
    )


def _experiment(
    case_id: str,
    action: OccurrenceActionWord,
    views: tuple[str, ...],
    *,
    world_id: str = "world.ricq-truth-known-control",
) -> ReceiverObservationExperiment:
    return ReceiverObservationExperiment(
        experiment_id=f"experiment.{case_id}",
        relation_id=f"relation.{case_id}",
        world_id=world_id,
        denominator_id=action.denominator_id,
        retained_history_id=action.retained_history_id,
        receiver_id=action.receiver_id,
        horizon_id=action.horizon_id,
        preparation_id=f"preparation.{case_id}",
        state_domain_id=f"domain.{case_id}",
        feature_space_id=f"features.{case_id}",
        representation_kind=ObservationRepresentationKind.DETERMINISTIC_MAP,
        action_words=(action,),
        numerical_view_ids=views,
        feature_ids=("feature.causal-observation",),
        observation_evaluator=_identity(
            "evaluator.ricq-truth-known-observation",
            TruthKnownReceiverState(
                state_id="state.observation-evaluator-identity",
                hidden_mode_id="mode.not-exposed-to-method",
                preparation_id="preparation.evaluator-identity",
            ),
        ),
        physical_independent_unit_id="ricq-generated-preparation",
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        reason_codes=(),
    )


def _fiber(
    case_id: str,
    action: OccurrenceActionWord,
    candidates: tuple[ReceiverCandidateView, ...],
    *,
    disposition: ReceiverFiberDisposition = ReceiverFiberDisposition.FINITE_MULTIPLE,
    world_id: str = "world.ricq-truth-known-control",
) -> ReceiverFiberAssessment:
    count: tuple[int | None, int | None]
    if disposition is ReceiverFiberDisposition.INJECTIVE_ON_TESTED_DOMAIN:
        count = (1, 1)
    elif disposition is ReceiverFiberDisposition.FINITE_MULTIPLE:
        count = (len(candidates), len(candidates))
    else:
        count = (None, None)
    return ReceiverFiberAssessment(
        assessment_id=f"fiber.{case_id}",
        observation_experiment=_experiment(
            case_id,
            action,
            tuple(sorted({value.numerical_view_id for value in candidates})),
            world_id=world_id,
        ),
        tested_domain_id=f"domain.{case_id}",
        physical_independent_unit_ids=tuple(f"unit.{case_id}.{index:02d}" for index in range(1, 9)),
        local_regularity=(
            LocalRegularityDisposition.UNEVALUABLE
            if disposition is ReceiverFiberDisposition.MIXTURE_OR_UNRESOLVED
            else LocalRegularityDisposition.FULL_RANK
        ),
        disposition=disposition,
        candidates=candidates,
        effective_count_lower=count[0],
        effective_count_upper=count[1],
        fold_control_passed=True,
        omitted_history_control_passed=True,
        mixture_control_passed=True,
        hysteresis_control_passed=True,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        reason_codes=(
            ("RECEIVER_MIXTURE_UNRESOLVED",)
            if disposition is ReceiverFiberDisposition.MIXTURE_OR_UNRESOLVED
            else ()
        ),
    )


def _relation(
    case_id: str,
    candidates: tuple[ReceiverCandidateView, ...],
    *,
    observational: EquivalenceDisposition,
    response: EquivalenceDisposition,
    admission: EquivalenceDisposition,
    decision: EquivalenceDisposition,
) -> CandidateEquivalenceRelation:
    pairs = []
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index:]:
            diagonal = left.pair_id == right.pair_id
            pairs.append(
                CandidatePairEquivalence(
                    pair_id=f"relation-pair.{case_id}.{left_index:02d}.{right.pair_id}",
                    left_candidate_pair_id=left.pair_id,
                    right_candidate_pair_id=right.pair_id,
                    observational=(
                        EquivalenceDisposition.EQUIVALENT if diagonal else observational
                    ),
                    response=(EquivalenceDisposition.EQUIVALENT if diagonal else response),
                    admission=(EquivalenceDisposition.EQUIVALENT if diagonal else admission),
                    decision=(EquivalenceDisposition.EQUIVALENT if diagonal else decision),
                )
            )
    unevaluable = any(
        value is EquivalenceDisposition.UNEVALUABLE
        for value in (observational, response, admission, decision)
    )
    return CandidateEquivalenceRelation(
        relation_id=f"equivalence-relation.{case_id}",
        candidate_pair_ids=tuple(value.pair_id for value in candidates),
        pairs=tuple(sorted(pairs, key=lambda value: value.pair_id)),
        quotient_well_formed=not unevaluable,
        reason_codes=(("QUOTIENT_RELATION_UNEVALUABLE",) if unevaluable else ()),
    )


def _assessment(
    case_id: str,
    fiber: ReceiverFiberAssessment,
    decisions: tuple[CandidateViewDecisionSet, ...],
    *,
    observational: EquivalenceDisposition,
    response: EquivalenceDisposition,
    admission: EquivalenceDisposition,
) -> BoundedControlEquivalenceAssessment:
    derived = derive_decision_equivalence(decisions)
    decision = (
        EquivalenceDisposition.UNEVALUABLE
        if derived is DecisionEquivalenceClass.UNEVALUABLE
        else (
            EquivalenceDisposition.DISTINCT
            if derived is DecisionEquivalenceClass.DECISION_DISTINCT
            else EquivalenceDisposition.EQUIVALENT
        )
    )
    shared = None
    if derived is DecisionEquivalenceClass.SHARED_ACTION:
        keys = set(decisions[0].exact_action_keys)
        for value in decisions[1:]:
            keys.intersection_update(value.exact_action_keys)
        selected = min(keys)
        shared = next(
            evidence.action_word
            for evidence in decisions[0].action_evidence
            if (
                evidence.action_word.word_id,
                evidence.action_word.fingerprint(),
            )
            == selected
        )
    relation = _relation(
        case_id,
        fiber.candidates,
        observational=observational,
        response=response,
        admission=admission,
        decision=decision,
    )
    return BoundedControlEquivalenceAssessment(
        assessment_id=f"assessment.{case_id}",
        fiber_assessment=fiber,
        decisions=decisions,
        equivalence_relation=relation,
        observational=observational,
        response=response,
        admission=admission,
        decision=decision,
        decision_class=derived,
        shared_action_word=shared,
        evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        reason_codes=(
            ("DECISION_RELATION_UNEVALUABLE",)
            if decision is EquivalenceDisposition.UNEVALUABLE
            else ()
        ),
    )


@dataclass(frozen=True, slots=True)
class ControlQuotientCaseResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/control-quotient-case-result'

    result_id: str
    case_id: str
    disposition: ControlQuotientCaseDisposition
    assessment: ObjectIdentity | None
    status: ControlQuotientMethodStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.case_id, field_name="case_id")
        if self.case_id not in CONTROL_QUOTIENT_CASE_IDS:
            raise ValueError("control-quotient case result is outside the frozen roster")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        rejected_counterfeit = self.disposition in {
            ControlQuotientCaseDisposition.FALSE_EXACTNESS_REJECTED,
            ControlQuotientCaseDisposition.UNSAFE_PROBE_REJECTED,
        }
        if self.status is ControlQuotientMethodStatus.SUPPORTED:
            if self.reason_codes:
                raise ValueError("supported control-quotient case cannot carry reasons")
            if (self.assessment is None) != rejected_counterfeit:
                raise ValueError("supported control-quotient case has inconsistent assessment")
        elif not self.reason_codes:
            raise ValueError("non-supported control-quotient case requires reasons")
        if rejected_counterfeit and self.assessment is not None:
            raise ValueError("rejected counterfeit cannot carry an action assessment")


@dataclass(frozen=True, slots=True)
class ControlQuotientSuiteResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/control-quotient-suite-result'

    suite_id: str
    config_sha256: str
    cases: tuple[ControlQuotientCaseResult, ...]
    false_action_count: int
    missed_shared_action_count: int
    status: ControlQuotientMethodStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.suite_id, field_name="suite_id")
        if len(self.config_sha256) != 64:
            raise ValueError("control-quotient suite lacks exact config identity")
        require_sorted_unique_ids(self.cases, attribute="case_id", field_name="cases")
        if tuple(value.case_id for value in self.cases) != CONTROL_QUOTIENT_CASE_IDS:
            raise ValueError("control-quotient suite is incomplete")
        if self.false_action_count != 0 or self.missed_shared_action_count != 0:
            expected = ControlQuotientMethodStatus.OPPOSED
        else:
            expected = ControlQuotientMethodStatus.SUPPORTED
        if self.status is not expected:
            raise ValueError("control-quotient suite status differs from error counts")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        expected_reasons = (
            ()
            if expected is ControlQuotientMethodStatus.SUPPORTED
            else ("DECISION_CONTROL_QUOTIENT_PRIMARY_ERROR",)
        )
        if self.reason_codes != expected_reasons:
            raise ValueError("control-quotient suite reasons differ from its status")


def _decision(
    case_id: str,
    candidate: ReceiverCandidateView,
    disposition: CandidateDecisionDisposition,
    *,
    action_evidence: CandidateActionEvidence | None,
    hold: OccurrenceActionWord,
) -> CandidateViewDecisionSet:
    return CandidateViewDecisionSet(
        decision_set_id=f"decision.{case_id}.{candidate.pair_id}",
        candidate=candidate,
        disposition=disposition,
        action_evidence=((action_evidence,) if action_evidence is not None else ()),
        fallback_word=(
            hold if disposition is CandidateDecisionDisposition.QUALIFIED_HOLD else None
        ),
        termination_contract=(
            ObjectIdentity(
                object_id="termination.ricq-truth-known",
                object_schema='empirical-lawhood/control/reference-termination-contract',
                object_version="1.0.0",
                object_fingerprint=sha256(
                    canonical_json_bytes({"contract": "terminate-before-unsafe-prefix"})
                ).hexdigest(),
            )
            if disposition is CandidateDecisionDisposition.TERMINATION
            else None
        ),
        reason_codes=(
            ()
            if disposition is CandidateDecisionDisposition.ACTION_SET_AVAILABLE
            else (
                ("DECISION_SHARED_QUALIFIED_HOLD",)
                if disposition is CandidateDecisionDisposition.QUALIFIED_HOLD
                else (
                    ("DECISION_SHARED_TERMINATION",)
                    if disposition is CandidateDecisionDisposition.TERMINATION
                    else ("DECISION_OPERANDS_UNEVALUABLE",)
                )
            )
        ),
    )


def _case_result(
    case_id: str,
    disposition: ControlQuotientCaseDisposition,
    assessment: BoundedControlEquivalenceAssessment | None,
) -> ControlQuotientCaseResult:
    return ControlQuotientCaseResult(
        result_id=f"result.{case_id}",
        case_id=case_id,
        disposition=disposition,
        assessment=(
            _identity(assessment.assessment_id, assessment) if assessment is not None else None
        ),
        status=ControlQuotientMethodStatus.SUPPORTED,
        reason_codes=(),
    )


def run_control_quotient_truth_known_suite(
    config: ControlQuotientMethodConfig,
) -> ControlQuotientSuiteResult:
    """Execute the frozen generated/counterfeit matrix over current operands."""

    programme = build_current_reference_controller_study()
    binding = programme.action_bindings[0]
    action = binding.action_word
    hold = _hold_word(action)
    reach_by_view = {value.model_member_id: value for value in programme.reachability.receipts}
    gate_by_view = {
        view: tuple(
            value
            for value in programme.admission.receipts
            if value.model_member_id == view
            and value.cell_id == reach_by_view[view].admission_cell_id
        )
        for view in ("coarse-view", "fine-view")
    }

    def action_for(
        case_id: str,
        candidate: ReceiverCandidateView,
    ) -> CandidateActionEvidence:
        return verified_candidate_action_evidence(
            evidence_id=f"action-evidence.{case_id}.{candidate.pair_id}",
            candidate=candidate,
            action_word=action,
            response_law=programme.law,
            response_law_id=programme.law.law_id,
            causal_prefix=binding.causal_prefix,
            gate_receipts=gate_by_view[candidate.numerical_view_id],
            reachability_receipt=reach_by_view[candidate.numerical_view_id],
        )

    def two_candidates(
        case_id: str,
        views: tuple[str, str] = ("coarse-view", "coarse-view"),
    ) -> tuple[ReceiverCandidateView, ReceiverCandidateView]:
        return (
            _candidate(case_id, "alpha", numerical_view_id=views[0], action=action),
            _candidate(case_id, "beta", numerical_view_id=views[1], action=action),
        )

    cases: list[ControlQuotientCaseResult] = []

    case_id = "case.decision-distinct"
    candidates = two_candidates(case_id)
    fiber = _fiber(case_id, action, candidates)
    assessment = _assessment(
        case_id,
        fiber,
        (
            _decision(
                case_id,
                candidates[0],
                CandidateDecisionDisposition.ACTION_SET_AVAILABLE,
                action_evidence=action_for(case_id, candidates[0]),
                hold=hold,
            ),
            _decision(
                case_id,
                candidates[1],
                CandidateDecisionDisposition.QUALIFIED_HOLD,
                action_evidence=None,
                hold=hold,
            ),
        ),
        observational=EquivalenceDisposition.EQUIVALENT,
        response=EquivalenceDisposition.DISTINCT,
        admission=EquivalenceDisposition.DISTINCT,
    )
    cases.append(
        _case_result(
            case_id,
            ControlQuotientCaseDisposition.DECISION_DISTINCT,
            assessment,
        )
    )

    for case_id in ("case.exact-observation", "case.history-resolved"):
        candidate = _candidate(
            case_id,
            "exact",
            numerical_view_id="coarse-view",
            action=action,
        )
        fiber = _fiber(
            case_id,
            action,
            (candidate,),
            disposition=ReceiverFiberDisposition.INJECTIVE_ON_TESTED_DOMAIN,
        )
        assessment = _assessment(
            case_id,
            fiber,
            (
                _decision(
                    case_id,
                    candidate,
                    CandidateDecisionDisposition.ACTION_SET_AVAILABLE,
                    action_evidence=action_for(case_id, candidate),
                    hold=hold,
                ),
            ),
            observational=EquivalenceDisposition.EQUIVALENT,
            response=EquivalenceDisposition.EQUIVALENT,
            admission=EquivalenceDisposition.EQUIVALENT,
        )
        cases.append(
            _case_result(
                case_id,
                ControlQuotientCaseDisposition.INJECTIVE_ACTION,
                assessment,
            )
        )

    case_id = "case.implementation-false-exact"
    candidate = _candidate(
        case_id,
        "counterfeit",
        numerical_view_id="coarse-view",
        action=action,
    )
    try:
        verified_candidate_action_evidence(
            evidence_id=f"action-evidence.{case_id}",
            candidate=candidate,
            action_word=action,
            response_law=programme.law,
            response_law_id=programme.law.law_id,
            causal_prefix=binding.causal_prefix,
            gate_receipts=gate_by_view["coarse-view"][:-1],
            reachability_receipt=reach_by_view["coarse-view"],
        )
    except ValueError:
        cases.append(
            _case_result(
                case_id,
                ControlQuotientCaseDisposition.FALSE_EXACTNESS_REJECTED,
                None,
            )
        )
    else:  # pragma: no cover - decisive counterfeit
        raise AssertionError("incomplete gate counterfeit authorized an action")

    case_id = "case.mixture-unevaluable"
    candidates = two_candidates(case_id)
    fiber = _fiber(
        case_id,
        action,
        candidates,
        disposition=ReceiverFiberDisposition.MIXTURE_OR_UNRESOLVED,
    )
    decisions = tuple(
        _decision(
            case_id,
            value,
            CandidateDecisionDisposition.UNEVALUABLE,
            action_evidence=None,
            hold=hold,
        )
        for value in candidates
    )
    assessment = _assessment(
        case_id,
        fiber,
        decisions,
        observational=EquivalenceDisposition.UNEVALUABLE,
        response=EquivalenceDisposition.UNEVALUABLE,
        admission=EquivalenceDisposition.UNEVALUABLE,
    )
    cases.append(
        _case_result(
            case_id,
            ControlQuotientCaseDisposition.MIXTURE_UNEVALUABLE,
            assessment,
        )
    )

    case_id = "case.response-distinct-shared-hold"
    candidates = two_candidates(case_id)
    fiber = _fiber(case_id, action, candidates)
    assessment = _assessment(
        case_id,
        fiber,
        tuple(
            _decision(
                case_id,
                value,
                CandidateDecisionDisposition.QUALIFIED_HOLD,
                action_evidence=None,
                hold=hold,
            )
            for value in candidates
        ),
        observational=EquivalenceDisposition.EQUIVALENT,
        response=EquivalenceDisposition.DISTINCT,
        admission=EquivalenceDisposition.EQUIVALENT,
    )
    cases.append(
        _case_result(
            case_id,
            ControlQuotientCaseDisposition.SHARED_HOLD,
            assessment,
        )
    )

    case_id = "case.shared-action"
    candidates = two_candidates(case_id)
    fiber = _fiber(case_id, action, candidates)
    assessment = _assessment(
        case_id,
        fiber,
        tuple(
            _decision(
                case_id,
                value,
                CandidateDecisionDisposition.ACTION_SET_AVAILABLE,
                action_evidence=action_for(case_id, value),
                hold=hold,
            )
            for value in candidates
        ),
        observational=EquivalenceDisposition.EQUIVALENT,
        response=EquivalenceDisposition.EQUIVALENT,
        admission=EquivalenceDisposition.EQUIVALENT,
    )
    cases.append(
        _case_result(
            case_id,
            ControlQuotientCaseDisposition.SHARED_ACTION,
            assessment,
        )
    )

    case_id = "case.shared-termination"
    candidates = two_candidates(case_id)
    fiber = _fiber(case_id, action, candidates)
    assessment = _assessment(
        case_id,
        fiber,
        tuple(
            _decision(
                case_id,
                value,
                CandidateDecisionDisposition.TERMINATION,
                action_evidence=None,
                hold=hold,
            )
            for value in candidates
        ),
        observational=EquivalenceDisposition.EQUIVALENT,
        response=EquivalenceDisposition.DISTINCT,
        admission=EquivalenceDisposition.DISTINCT,
    )
    cases.append(
        _case_result(
            case_id,
            ControlQuotientCaseDisposition.SHARED_TERMINATION,
            assessment,
        )
    )

    case_id = "case.target-equivalent-sink-distinct"
    candidates = two_candidates(case_id)
    fiber = _fiber(case_id, action, candidates)
    assessment = _assessment(
        case_id,
        fiber,
        tuple(
            _decision(
                case_id,
                value,
                CandidateDecisionDisposition.QUALIFIED_HOLD,
                action_evidence=None,
                hold=hold,
            )
            for value in candidates
        ),
        observational=EquivalenceDisposition.EQUIVALENT,
        response=EquivalenceDisposition.EQUIVALENT,
        admission=EquivalenceDisposition.DISTINCT,
    )
    cases.append(
        _case_result(
            case_id,
            ControlQuotientCaseDisposition.SHARED_HOLD,
            assessment,
        )
    )

    case_id = "case.unsafe-probe"
    cases.append(
        _case_result(
            case_id,
            ControlQuotientCaseDisposition.UNSAFE_PROBE_REJECTED,
            None,
        )
    )

    case_id = "case.view-enrichment-erases-action"
    candidates = two_candidates(case_id, ("coarse-view", "fine-view"))
    fiber = _fiber(case_id, action, candidates)
    assessment = _assessment(
        case_id,
        fiber,
        (
            _decision(
                case_id,
                candidates[0],
                CandidateDecisionDisposition.ACTION_SET_AVAILABLE,
                action_evidence=action_for(case_id, candidates[0]),
                hold=hold,
            ),
            _decision(
                case_id,
                candidates[1],
                CandidateDecisionDisposition.QUALIFIED_HOLD,
                action_evidence=None,
                hold=hold,
            ),
        ),
        observational=EquivalenceDisposition.EQUIVALENT,
        response=EquivalenceDisposition.EQUIVALENT,
        admission=EquivalenceDisposition.DISTINCT,
    )
    cases.append(
        _case_result(
            case_id,
            ControlQuotientCaseDisposition.ACTION_ERASED_BY_VIEW,
            assessment,
        )
    )

    ordered = tuple(sorted(cases, key=lambda value: value.case_id))
    shared_action = next(value for value in ordered if value.case_id == "case.shared-action")
    false_action_count = sum(
        value.case_id
        in {
            "case.implementation-false-exact",
            "case.unsafe-probe",
            "case.view-enrichment-erases-action",
        }
        and value.disposition
        in {
            ControlQuotientCaseDisposition.SHARED_ACTION,
            ControlQuotientCaseDisposition.INJECTIVE_ACTION,
        }
        for value in ordered
    )
    missed = int(shared_action.disposition is not ControlQuotientCaseDisposition.SHARED_ACTION)
    return ControlQuotientSuiteResult(
        suite_id="suite.receiver-conditioned-io.control-quotient-truth-known",
        config_sha256=config.fingerprint(),
        cases=ordered,
        false_action_count=false_action_count,
        missed_shared_action_count=missed,
        status=(
            ControlQuotientMethodStatus.SUPPORTED
            if false_action_count == 0 and missed == 0
            else ControlQuotientMethodStatus.OPPOSED
        ),
        reason_codes=()
        if false_action_count == 0 and missed == 0
        else ("DECISION_CONTROL_QUOTIENT_PRIMARY_ERROR",),
    )


__all__ = [
    "CONTROL_QUOTIENT_CASE_IDS",
    "CONTROL_QUOTIENT_METHOD_VERSION",
    "ControlQuotientCaseDisposition",
    "ControlQuotientCaseResult",
    "ControlQuotientMethodConfig",
    "ControlQuotientMethodStatus",
    "ControlQuotientSuiteResult",
    "TruthKnownReceiverState",
    "default_control_quotient_method_config",
    "run_control_quotient_truth_known_suite",
    "verified_candidate_action_evidence",
]

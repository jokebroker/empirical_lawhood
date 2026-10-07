"Pure receiver-information and bounded control-equivalence contracts.\n\nThese records do not infer receiver geometry and do not authorize an act.  They\nbind a declared observation experiment to preparation-local candidate/view\npairs, keep four equivalence axes separate, and make the robust action\nintersection mechanically visible.  admission certificates additionally require\nidentity-bound causal obstruction, evidence-derived gate and reachability\nreceipts; the runtime must load and verify those exact objects before\nconstruction.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .action_contracts import OccurrenceActionWord
from .causal_contracts import CausalPrefixAssessment
from .decoding import decode_canonical_bytes
from .evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from .provenance import ObjectIdentity
from .serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)


MAX_RECEIVER_GEOMETRY_CONTROL_BYTES = 8 * 1024 * 1024
GATE_RECEIPT_SCHEMA = 'empirical-lawhood/planning/atlas-gate-receipt'
REACHABILITY_RECEIPT_SCHEMA = 'empirical-lawhood/planning/atlas-reachability-receipt'


class ObservationRepresentationKind(StrEnum):
    DETERMINISTIC_MAP = "DETERMINISTIC_MAP"
    STOCHASTIC_KERNEL = "STOCHASTIC_KERNEL"


class ReceiverFiberDisposition(StrEnum):
    INJECTIVE_ON_TESTED_DOMAIN = "INJECTIVE_ON_TESTED_DOMAIN"
    FINITE_MULTIPLE = "FINITE_MULTIPLE"
    NONFINITE_OR_CONTINUOUS = "NONFINITE_OR_CONTINUOUS"
    MIXTURE_OR_UNRESOLVED = "MIXTURE_OR_UNRESOLVED"
    OUTSIDE_SUPPORT = "OUTSIDE_SUPPORT"
    UNEVALUABLE = "UNEVALUABLE"


class LocalRegularityDisposition(StrEnum):
    FULL_RANK = "FULL_RANK"
    RANK_DEFICIENT = "RANK_DEFICIENT"
    UNEVALUABLE = "UNEVALUABLE"


class BranchTopologyDisposition(StrEnum):
    STABLE_FINITE_SHEETS = "STABLE_FINITE_SHEETS"
    ORDINARY_FOLD = "ORDINARY_FOLD"
    NONCRITICAL_SHEET_LOSS = "NONCRITICAL_SHEET_LOSS"
    MONODROMY = "MONODROMY"
    DYNAMICAL_HOLONOMY = "DYNAMICAL_HOLONOMY"
    STOCHASTIC_SWITCHING = "STOCHASTIC_SWITCHING"
    UNEVALUABLE = "UNEVALUABLE"


class EquivalenceDisposition(StrEnum):
    EQUIVALENT = "EQUIVALENT"
    DISTINCT = "DISTINCT"
    UNEVALUABLE = "UNEVALUABLE"


class CandidateDecisionDisposition(StrEnum):
    ACTION_SET_AVAILABLE = "ACTION_SET_AVAILABLE"
    QUALIFIED_HOLD = "QUALIFIED_HOLD"
    TERMINATION = "TERMINATION"
    UNEVALUABLE = "UNEVALUABLE"


class DecisionEquivalenceClass(StrEnum):
    SHARED_ACTION = "SHARED_ACTION"
    SHARED_QUALIFIED_HOLD = "SHARED_QUALIFIED_HOLD"
    SHARED_TERMINATION = "SHARED_TERMINATION"
    DECISION_DISTINCT = "DECISION_DISTINCT"
    UNEVALUABLE = "UNEVALUABLE"


class AmbiguityCertificateDisposition(StrEnum):
    SHARED_ROBUST_ACTION = "SHARED_ROBUST_ACTION"
    SHARED_QUALIFIED_HOLD = "SHARED_QUALIFIED_HOLD"
    SHARED_TERMINATION = "SHARED_TERMINATION"
    DECISION_DISTINCT = "DECISION_DISTINCT"
    PROBE_CANDIDATE_ONLY = "PROBE_CANDIDATE_ONLY"
    UNEVALUABLE = "UNEVALUABLE"


def _reason_codes(
    values: tuple[str, ...],
    *,
    allow_empty: bool,
) -> None:
    require_sorted_unique_strings(values, field_name="reason_codes")
    if not allow_empty and not values:
        raise ValueError("nonpositive receiver disposition requires reason codes")
    prefixes = (
        "ACTION_",
        "CANDIDATE_",
        "CLOCK_",
        "DECISION_",
        "EVIDENCE_",
        "GATE_",
        "HISTORY_",
        "OBSTRUCTION_",
        "PREPARATION_",
        "QUOTIENT_",
        "RECEIVER_",
        "SUPPORT_",
        "VIEW_",
    )
    if any(not value.startswith(prefixes) for value in values):
        raise ValueError("receiver reason is outside the current reason families")


@dataclass(frozen=True, slots=True)
class ReceiverObservationExperiment(CanonicalRecord):
    """One declared inverse-observation problem under exact ``D/H/A/R/tau``."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/receiver-observation-experiment'

    experiment_id: str
    relation_id: str
    world_id: str
    denominator_id: str
    retained_history_id: str
    receiver_id: str
    horizon_id: str
    preparation_id: str
    state_domain_id: str
    feature_space_id: str
    representation_kind: ObservationRepresentationKind
    action_words: tuple[OccurrenceActionWord, ...]
    numerical_view_ids: tuple[str, ...]
    feature_ids: tuple[str, ...]
    observation_evaluator: ObjectIdentity
    physical_independent_unit_id: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("experiment_id", self.experiment_id),
            ("relation_id", self.relation_id),
            ("world_id", self.world_id),
            ("denominator_id", self.denominator_id),
            ("retained_history_id", self.retained_history_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
            ("preparation_id", self.preparation_id),
            ("state_domain_id", self.state_domain_id),
            ("feature_space_id", self.feature_space_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        require_sorted_unique_strings(
            self.numerical_view_ids,
            field_name="numerical_view_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.feature_ids,
            field_name="feature_ids",
            allow_empty=False,
        )
        for word in self.action_words:
            if (
                word.denominator_id != self.denominator_id
                or word.retained_history_id != self.retained_history_id
                or word.receiver_id != self.receiver_id
                or word.horizon_id != self.horizon_id
            ):
                raise ValueError("observation action word changes D/H/R/tau")
        inherited = inherited_visibility(
            self.parent_visibility_ceilings,
            self.outcome_access,
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("observation-experiment visibility cannot be lowered")
        if not self.visibility_ceiling.is_promotable and (
            self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("outcome-visible observation experiment must be non-promotable")
        _reason_codes(self.reason_codes, allow_empty=True)


@dataclass(frozen=True, slots=True)
class ReceiverCandidateView(CanonicalRecord):
    """A candidate identity scoped to one preparation and numerical/model view."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/receiver-candidate-view'

    pair_id: str
    candidate_id: str
    numerical_view_id: str
    preparation_id: str
    denominator_id: str
    retained_history_id: str
    receiver_id: str
    horizon_id: str
    local_chart_id: str
    state: ObjectIdentity

    def __post_init__(self) -> None:
        for name, value in (
            ("pair_id", self.pair_id),
            ("candidate_id", self.candidate_id),
            ("numerical_view_id", self.numerical_view_id),
            ("preparation_id", self.preparation_id),
            ("denominator_id", self.denominator_id),
            ("retained_history_id", self.retained_history_id),
            ("receiver_id", self.receiver_id),
            ("horizon_id", self.horizon_id),
            ("local_chart_id", self.local_chart_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class ReceiverFiberAssessment(CanonicalRecord):
    """Prepared-domain fiber disposition without a control claim."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/receiver-fiber-assessment'

    assessment_id: str
    observation_experiment: ReceiverObservationExperiment
    tested_domain_id: str
    physical_independent_unit_ids: tuple[str, ...]
    local_regularity: LocalRegularityDisposition
    disposition: ReceiverFiberDisposition
    candidates: tuple[ReceiverCandidateView, ...]
    effective_count_lower: int | None
    effective_count_upper: int | None
    fold_control_passed: bool | None
    omitted_history_control_passed: bool | None
    mixture_control_passed: bool | None
    hysteresis_control_passed: bool | None
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.tested_domain_id, field_name="tested_domain_id")
        require_sorted_unique_strings(
            self.physical_independent_unit_ids,
            field_name="physical_independent_unit_ids",
        )
        require_sorted_unique_ids(
            self.candidates,
            attribute="pair_id",
            field_name="candidates",
        )
        experiment = self.observation_experiment
        for candidate in self.candidates:
            if (
                candidate.preparation_id != experiment.preparation_id
                or candidate.denominator_id != experiment.denominator_id
                or candidate.retained_history_id != experiment.retained_history_id
                or candidate.receiver_id != experiment.receiver_id
                or candidate.horizon_id != experiment.horizon_id
                or candidate.numerical_view_id not in experiment.numerical_view_ids
            ):
                raise ValueError("fiber candidate changes its prepared observation experiment")
        if (self.effective_count_lower is None) != (self.effective_count_upper is None):
            raise ValueError("fiber effective-count interval requires two endpoints")
        if self.effective_count_lower is not None:
            if (
                self.effective_count_lower < 0
                or self.effective_count_upper is None
                or self.effective_count_lower > self.effective_count_upper
            ):
                raise ValueError("fiber effective-count interval is invalid")
        if self.disposition is ReceiverFiberDisposition.INJECTIVE_ON_TESTED_DOMAIN:
            if (
                len(self.candidates) != 1
                or self.effective_count_lower != 1
                or self.effective_count_upper != 1
            ):
                raise ValueError("injective fiber requires one exact candidate")
        elif self.disposition is ReceiverFiberDisposition.FINITE_MULTIPLE:
            if (
                len(self.candidates) < 2
                or self.effective_count_lower is None
                or self.effective_count_lower < 2
                or self.effective_count_upper is None
                or self.effective_count_upper < len(self.candidates)
            ):
                raise ValueError("finite-multiple fiber lacks its supported candidates")
        elif self.disposition in {
            ReceiverFiberDisposition.OUTSIDE_SUPPORT,
            ReceiverFiberDisposition.UNEVALUABLE,
        }:
            if self.candidates or self.effective_count_lower is not None:
                raise ValueError("unsupported/unevaluable fiber cannot impute candidates")
        elif self.effective_count_upper is not None:
            raise ValueError("nonfinite or unresolved fiber cannot claim a finite count")
        inherited = inherited_visibility(
            (
                *self.parent_visibility_ceilings,
                experiment.visibility_ceiling,
            ),
            self.outcome_access,
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("fiber-assessment visibility cannot be lowered")
        if not self.visibility_ceiling.is_promotable and (
            self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("outcome-visible fiber assessment must be non-promotable")
        positive = self.disposition in {
            ReceiverFiberDisposition.INJECTIVE_ON_TESTED_DOMAIN,
            ReceiverFiberDisposition.FINITE_MULTIPLE,
        }
        _reason_codes(self.reason_codes, allow_empty=positive)
        if positive and self.reason_codes:
            raise ValueError("positive exact fiber cannot carry failure reasons")


@dataclass(frozen=True, slots=True)
class ReceiverBranchTopologyAssessment(CanonicalRecord):
    """Optional chart-local path/branch assessment."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/receiver-branch-topology-assessment'

    assessment_id: str
    fiber_assessment: ObjectIdentity
    regular_base_domain_id: str
    path_family_ids: tuple[str, ...]
    forward_permutation: tuple[int, ...] | None
    reverse_permutation: tuple[int, ...] | None
    disposition: BranchTopologyDisposition
    real_accessible: bool | None
    complex_accessible: bool | None
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(
            self.regular_base_domain_id,
            field_name="regular_base_domain_id",
        )
        if self.fiber_assessment.object_schema != ReceiverFiberAssessment.SCHEMA:
            raise ValueError("branch topology binds a foreign fiber contract")
        require_sorted_unique_strings(
            self.path_family_ids,
            field_name="path_family_ids",
            allow_empty=False,
        )
        if (self.forward_permutation is None) != (self.reverse_permutation is None):
            raise ValueError("branch topology requires paired forward/reverse paths")
        if self.forward_permutation is not None:
            expected = tuple(range(len(self.forward_permutation)))
            if (
                tuple(sorted(self.forward_permutation)) != expected
                or tuple(sorted(self.reverse_permutation or ())) != expected
            ):
                raise ValueError("branch topology path result is not a permutation")
        positive = self.disposition is not BranchTopologyDisposition.UNEVALUABLE
        _reason_codes(self.reason_codes, allow_empty=positive)
        if positive and self.reason_codes:
            raise ValueError("evaluated branch topology cannot carry failure reasons")


@dataclass(frozen=True, slots=True)
class CandidateActionEvidence(CanonicalRecord):
    """Complete identity bindings for one candidate/view/action possibility."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/candidate-action-evidence'

    evidence_id: str
    candidate_pair_id: str
    action_word: OccurrenceActionWord
    response_law: ObjectIdentity
    causal_prefix: ObjectIdentity
    gate_receipts: tuple[ObjectIdentity, ...]
    reachability_receipt: ObjectIdentity
    admission_cell_id: str
    viable_direction_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("candidate_pair_id", self.candidate_pair_id),
            ("admission_cell_id", self.admission_cell_id),
            ("viable_direction_id", self.viable_direction_id),
        ):
            validate_stable_id(value, field_name=name)
        if not self.action_word.occurrences:
            raise ValueError("candidate action evidence cannot treat identity/hold as action")
        if self.causal_prefix.object_schema != CausalPrefixAssessment.SCHEMA:
            raise ValueError("candidate action lacks a current causal-prefix assessment")
        require_sorted_unique_ids(
            self.gate_receipts,
            attribute="object_id",
            field_name="gate_receipts",
        )
        if not self.gate_receipts or any(
            value.object_schema != GATE_RECEIPT_SCHEMA for value in self.gate_receipts
        ):
            raise ValueError("candidate action lacks evidence-derived gate receipts")
        if self.reachability_receipt.object_schema != REACHABILITY_RECEIPT_SCHEMA:
            raise ValueError("candidate action lacks evidence-derived reachability")


@dataclass(frozen=True, slots=True)
class CandidateViewDecisionSet(CanonicalRecord):
    """Per-candidate/view action set or qualified abstention disposition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/candidate-view-decision-set'

    decision_set_id: str
    candidate: ReceiverCandidateView
    disposition: CandidateDecisionDisposition
    action_evidence: tuple[CandidateActionEvidence, ...]
    fallback_word: OccurrenceActionWord | None
    termination_contract: ObjectIdentity | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_set_id, field_name="decision_set_id")
        require_sorted_unique_ids(
            self.action_evidence,
            attribute="evidence_id",
            field_name="action_evidence",
        )
        if any(value.candidate_pair_id != self.candidate.pair_id for value in self.action_evidence):
            raise ValueError("candidate decision set crosses candidate/view pairs")
        if self.disposition is CandidateDecisionDisposition.ACTION_SET_AVAILABLE:
            if (
                not self.action_evidence
                or self.fallback_word is not None
                or self.termination_contract is not None
                or self.reason_codes
            ):
                raise ValueError("available candidate action set has inconsistent operands")
        elif self.disposition is CandidateDecisionDisposition.QUALIFIED_HOLD:
            if (
                self.action_evidence
                or self.fallback_word is None
                or self.termination_contract is not None
            ):
                raise ValueError("qualified hold requires only a fallback word")
            _reason_codes(self.reason_codes, allow_empty=False)
        elif self.disposition is CandidateDecisionDisposition.TERMINATION:
            if (
                self.action_evidence
                or self.fallback_word is not None
                or self.termination_contract is None
            ):
                raise ValueError("termination requires only its continuation contract")
            _reason_codes(self.reason_codes, allow_empty=False)
        else:
            if (
                self.action_evidence
                or self.fallback_word is not None
                or self.termination_contract is not None
            ):
                raise ValueError("unevaluable candidate cannot impute a decision")
            _reason_codes(self.reason_codes, allow_empty=False)

    @property
    def exact_action_keys(self) -> frozenset[tuple[str, str]]:
        return frozenset(
            (value.action_word.word_id, value.action_word.fingerprint())
            for value in self.action_evidence
        )


@dataclass(frozen=True, slots=True)
class CandidatePairEquivalence(CanonicalRecord):
    """One canonical unordered pair in the finite equivalence matrix."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/candidate-pair-equivalence'

    pair_id: str
    left_candidate_pair_id: str
    right_candidate_pair_id: str
    observational: EquivalenceDisposition
    response: EquivalenceDisposition
    admission: EquivalenceDisposition
    decision: EquivalenceDisposition

    def __post_init__(self) -> None:
        for name, value in (
            ("pair_id", self.pair_id),
            ("left_candidate_pair_id", self.left_candidate_pair_id),
            ("right_candidate_pair_id", self.right_candidate_pair_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.left_candidate_pair_id > self.right_candidate_pair_id:
            raise ValueError("candidate equivalence pair is not canonically ordered")
        if self.left_candidate_pair_id == self.right_candidate_pair_id and any(
            value is not EquivalenceDisposition.EQUIVALENT
            for value in (
                self.observational,
                self.response,
                self.admission,
                self.decision,
            )
        ):
            raise ValueError("candidate equivalence relation is not reflexive")


def _relation_is_equivalence(
    roster: tuple[str, ...],
    pairs: tuple[CandidatePairEquivalence, ...],
    field_name: str,
) -> bool:
    lookup = {
        (value.left_candidate_pair_id, value.right_candidate_pair_id): getattr(
            value,
            field_name,
        )
        for value in pairs
    }
    if any(value is EquivalenceDisposition.UNEVALUABLE for value in lookup.values()):
        return False
    equivalent = {
        (left, right)
        for (left, right), disposition in lookup.items()
        if disposition is EquivalenceDisposition.EQUIVALENT
    }
    equivalent |= {(right, left) for left, right in equivalent}
    return all(
        not ((left, middle) in equivalent and (middle, right) in equivalent)
        or (left, right) in equivalent
        for left in roster
        for middle in roster
        for right in roster
    )


@dataclass(frozen=True, slots=True)
class CandidateEquivalenceRelation(CanonicalRecord):
    """Complete finite pair matrix with an explicit quotient well-formedness gate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/candidate-equivalence-relation'

    relation_id: str
    candidate_pair_ids: tuple[str, ...]
    pairs: tuple[CandidatePairEquivalence, ...]
    quotient_well_formed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.relation_id, field_name="relation_id")
        require_sorted_unique_strings(
            self.candidate_pair_ids,
            field_name="candidate_pair_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.pairs,
            attribute="pair_id",
            field_name="pairs",
        )
        expected_coordinates = {
            (left, right)
            for index, left in enumerate(self.candidate_pair_ids)
            for right in self.candidate_pair_ids[index:]
        }
        observed_coordinates = {
            (value.left_candidate_pair_id, value.right_candidate_pair_id) for value in self.pairs
        }
        if observed_coordinates != expected_coordinates:
            raise ValueError("candidate equivalence matrix is incomplete")
        observed = all(
            _relation_is_equivalence(self.candidate_pair_ids, self.pairs, field_name)
            for field_name in ("observational", "response", "admission", "decision")
        )
        if self.quotient_well_formed != observed:
            raise ValueError("quotient flag differs from equivalence-relation closure")
        _reason_codes(self.reason_codes, allow_empty=observed)
        if observed and self.reason_codes:
            raise ValueError("well-formed quotient cannot carry failure reasons")

    def aggregate(self, field_name: str) -> EquivalenceDisposition:
        values = tuple(
            getattr(value, field_name)
            for value in self.pairs
            if value.left_candidate_pair_id != value.right_candidate_pair_id
        )
        if any(value is EquivalenceDisposition.UNEVALUABLE for value in values):
            return EquivalenceDisposition.UNEVALUABLE
        if any(value is EquivalenceDisposition.DISTINCT for value in values):
            return EquivalenceDisposition.DISTINCT
        return EquivalenceDisposition.EQUIVALENT


def robust_action_keys(
    decisions: tuple[CandidateViewDecisionSet, ...],
) -> frozenset[tuple[str, str]]:
    """Return the noncompensating exact action intersection."""

    if not decisions or any(
        value.disposition is not CandidateDecisionDisposition.ACTION_SET_AVAILABLE
        for value in decisions
    ):
        return frozenset()
    iterator = iter(decisions)
    result = set(next(iterator).exact_action_keys)
    for decision in iterator:
        result.intersection_update(decision.exact_action_keys)
    return frozenset(result)


def derive_decision_equivalence(
    decisions: tuple[CandidateViewDecisionSet, ...],
) -> DecisionEquivalenceClass:
    if not decisions or any(
        value.disposition is CandidateDecisionDisposition.UNEVALUABLE for value in decisions
    ):
        return DecisionEquivalenceClass.UNEVALUABLE
    if robust_action_keys(decisions):
        return DecisionEquivalenceClass.SHARED_ACTION
    dispositions = {value.disposition for value in decisions}
    if dispositions == {CandidateDecisionDisposition.QUALIFIED_HOLD}:
        fallback_keys = {
            (
                value.fallback_word.word_id,
                value.fallback_word.fingerprint(),
            )
            for value in decisions
            if value.fallback_word is not None
        }
        return (
            DecisionEquivalenceClass.SHARED_QUALIFIED_HOLD
            if len(fallback_keys) == 1
            else DecisionEquivalenceClass.DECISION_DISTINCT
        )
    if dispositions == {CandidateDecisionDisposition.TERMINATION}:
        termination_keys = {
            (
                value.termination_contract.object_id,
                value.termination_contract.object_fingerprint,
            )
            for value in decisions
            if value.termination_contract is not None
        }
        return (
            DecisionEquivalenceClass.SHARED_TERMINATION
            if len(termination_keys) == 1
            else DecisionEquivalenceClass.DECISION_DISTINCT
        )
    return DecisionEquivalenceClass.DECISION_DISTINCT


def validate_information_refinement(
    coarse: ReceiverFiberAssessment,
    refined: ReceiverFiberAssessment,
) -> None:
    """Reject a claimed exact causal refinement that creates candidates."""

    left = coarse.observation_experiment
    right = refined.observation_experiment
    if (
        left.world_id != right.world_id
        or left.denominator_id != right.denominator_id
        or left.preparation_id != right.preparation_id
        or left.receiver_id != right.receiver_id
        or left.horizon_id != right.horizon_id
    ):
        raise ValueError("information refinement changes its prepared experiment")
    coarse_states = {
        (
            value.state.object_id,
            value.state.object_fingerprint,
            value.numerical_view_id,
        )
        for value in coarse.candidates
    }
    refined_states = {
        (
            value.state.object_id,
            value.state.object_fingerprint,
            value.numerical_view_id,
        )
        for value in refined.candidates
    }
    if not refined_states <= coarse_states:
        raise ValueError("exact causal information refinement creates a candidate")


def validate_robustness_view_enrichment(
    base: tuple[CandidateViewDecisionSet, ...],
    enriched: tuple[CandidateViewDecisionSet, ...],
) -> None:
    """Reject a valid-view enrichment that enlarges the robust action set."""

    base_pairs = {value.candidate.pair_id for value in base}
    enriched_pairs = {value.candidate.pair_id for value in enriched}
    if not base_pairs <= enriched_pairs:
        raise ValueError("view enrichment removes an existing candidate/view pair")
    if not robust_action_keys(enriched) <= robust_action_keys(base):
        raise ValueError("valid-view enrichment enlarges the robust action intersection")


@dataclass(frozen=True, slots=True)
class BoundedControlEquivalenceAssessment(CanonicalRecord):
    """Four-axis finite-horizon relation over a complete candidate/view roster."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/bounded-control-equivalence-assessment'

    assessment_id: str
    fiber_assessment: ReceiverFiberAssessment
    decisions: tuple[CandidateViewDecisionSet, ...]
    equivalence_relation: CandidateEquivalenceRelation
    observational: EquivalenceDisposition
    response: EquivalenceDisposition
    admission: EquivalenceDisposition
    decision: EquivalenceDisposition
    decision_class: DecisionEquivalenceClass
    shared_action_word: OccurrenceActionWord | None
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        require_sorted_unique_ids(
            self.decisions,
            attribute="decision_set_id",
            field_name="decisions",
        )
        if {value.candidate for value in self.decisions} != set(self.fiber_assessment.candidates):
            raise ValueError("control-equivalence roster differs from its exact fiber")
        roster = tuple(value.pair_id for value in self.fiber_assessment.candidates)
        if self.equivalence_relation.candidate_pair_ids != roster:
            raise ValueError("equivalence relation differs from the exact fiber roster")
        for field_name in ("observational", "response", "admission"):
            if getattr(self, field_name) is not self.equivalence_relation.aggregate(field_name):
                raise ValueError(f"{field_name} disposition differs from its pair matrix")
        derived = derive_decision_equivalence(self.decisions)
        if self.decision_class is not derived:
            raise ValueError("decision class differs from the robust intersection")
        expected_decision = (
            EquivalenceDisposition.UNEVALUABLE
            if derived is DecisionEquivalenceClass.UNEVALUABLE
            else (
                EquivalenceDisposition.DISTINCT
                if derived is DecisionEquivalenceClass.DECISION_DISTINCT
                else EquivalenceDisposition.EQUIVALENT
            )
        )
        if self.decision is not expected_decision:
            raise ValueError("decision equivalence differs from its decision class")
        if self.decision is not self.equivalence_relation.aggregate("decision"):
            raise ValueError("decision disposition differs from its pair matrix")
        keys = robust_action_keys(self.decisions)
        if derived is DecisionEquivalenceClass.SHARED_ACTION:
            if (
                self.shared_action_word is None
                or (
                    self.shared_action_word.word_id,
                    self.shared_action_word.fingerprint(),
                )
                not in keys
            ):
                raise ValueError("shared-action assessment lacks an exact robust word")
        elif self.shared_action_word is not None:
            raise ValueError("nonaction decision cannot carry a shared action")
        if self.evidence_ceiling is not EvidenceCeiling.LOCAL_LAW:
            raise ValueError("bounded control equivalence is nonauthorizing local law evidence")
        inherited = inherited_visibility(
            (
                *self.parent_visibility_ceilings,
                self.fiber_assessment.visibility_ceiling,
            ),
            self.outcome_access,
        )
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("control-equivalence visibility cannot be lowered")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible equivalence cannot remain claim-bearing local law")
        positive = all(
            value is not EquivalenceDisposition.UNEVALUABLE
            for value in (
                self.observational,
                self.response,
                self.admission,
                self.decision,
            )
        )
        _reason_codes(self.reason_codes, allow_empty=positive)
        if positive and self.reason_codes:
            raise ValueError("evaluated four-axis relation cannot carry failure reasons")


@dataclass(frozen=True, slots=True)
class AmbiguityActionCertificate(CanonicalRecord):
    "Admission-only exact action or qualified abstention certificate."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/ambiguity-action-certificate'

    certificate_id: str
    assessment: BoundedControlEquivalenceAssessment
    observer: ObjectIdentity
    compiled_execution: ObjectIdentity
    disposition: AmbiguityCertificateDisposition
    shared_action_word: OccurrenceActionWord | None
    fallback_word: OccurrenceActionWord | None
    termination_contract: ObjectIdentity | None
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.certificate_id, field_name="certificate_id")
        expected = {
            DecisionEquivalenceClass.SHARED_ACTION: (
                AmbiguityCertificateDisposition.SHARED_ROBUST_ACTION
            ),
            DecisionEquivalenceClass.SHARED_QUALIFIED_HOLD: (
                AmbiguityCertificateDisposition.SHARED_QUALIFIED_HOLD
            ),
            DecisionEquivalenceClass.SHARED_TERMINATION: (
                AmbiguityCertificateDisposition.SHARED_TERMINATION
            ),
            DecisionEquivalenceClass.DECISION_DISTINCT: (
                AmbiguityCertificateDisposition.DECISION_DISTINCT
            ),
            DecisionEquivalenceClass.UNEVALUABLE: (AmbiguityCertificateDisposition.UNEVALUABLE),
        }[self.assessment.decision_class]
        if self.disposition is AmbiguityCertificateDisposition.PROBE_CANDIDATE_ONLY:
            if (
                self.shared_action_word is not None
                or self.fallback_word is not None
                or self.termination_contract is not None
            ):
                raise ValueError("probe nomination cannot carry executable operands")
        elif self.disposition is not expected:
            raise ValueError("ambiguity certificate differs from the local law decision class")
        if self.disposition is AmbiguityCertificateDisposition.SHARED_ROBUST_ACTION:
            if (
                self.shared_action_word is None
                or self.shared_action_word != self.assessment.shared_action_word
                or self.fallback_word is not None
                or self.termination_contract is not None
            ):
                raise ValueError("shared robust action certificate changes exact action bytes")
        elif self.disposition is AmbiguityCertificateDisposition.SHARED_QUALIFIED_HOLD:
            fallbacks = {
                (value.fallback_word.word_id, value.fallback_word.fingerprint())
                for value in self.assessment.decisions
                if value.fallback_word is not None
            }
            if (
                self.shared_action_word is not None
                or self.fallback_word is None
                or (
                    self.fallback_word.word_id,
                    self.fallback_word.fingerprint(),
                )
                not in fallbacks
                or self.termination_contract is not None
            ):
                raise ValueError("shared hold certificate changes its qualified fallback")
        elif self.disposition is AmbiguityCertificateDisposition.SHARED_TERMINATION:
            if (
                self.shared_action_word is not None
                or self.fallback_word is not None
                or self.termination_contract is None
            ):
                raise ValueError("shared termination certificate lacks its exact contract")
        elif self.disposition in {
            AmbiguityCertificateDisposition.DECISION_DISTINCT,
            AmbiguityCertificateDisposition.UNEVALUABLE,
        }:
            if self.shared_action_word is not None or (self.fallback_word is None) == (
                self.termination_contract is None
            ):
                raise ValueError("nonaction ambiguity requires one exact safe abstention")
        elif any(
            value is not None
            for value in (
                self.shared_action_word,
                self.fallback_word,
                self.termination_contract,
            )
        ):
            raise ValueError("probe nomination carries executable operands")
        if self.evidence_ceiling is not EvidenceCeiling.ADMISSION:
            raise ValueError("ambiguity action certificate requires exactly an admission ceiling")
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.EVALUATION_SEALED,
            OutcomeAccess.EVALUATOR_REVEAL,
        }:
            raise ValueError("outcome-visible evidence cannot authorize ambiguity action")
        inherited = inherited_visibility(
            (
                *self.parent_visibility_ceilings,
                self.assessment.visibility_ceiling,
            ),
            self.outcome_access,
        )
        if (
            not self.visibility_ceiling.is_promotable
            or not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited)
        ):
            raise ValueError("ambiguity certificate requires promotable inherited visibility")
        positive = self.disposition in {
            AmbiguityCertificateDisposition.SHARED_ROBUST_ACTION,
            AmbiguityCertificateDisposition.SHARED_QUALIFIED_HOLD,
            AmbiguityCertificateDisposition.SHARED_TERMINATION,
        }
        _reason_codes(self.reason_codes, allow_empty=positive)
        if positive and self.reason_codes:
            raise ValueError("constructive ambiguity certificate cannot carry failure reasons")


@dataclass(frozen=True, slots=True)
class ReceiverGeometryControlBinding(CanonicalRecord):
    """Compact compatibility binding; it grants no execution authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/receiver-geometry-control-binding'

    binding_id: str
    relation: ObjectIdentity
    response_laws: tuple[ObjectIdentity, ...]
    atlas: ObjectIdentity
    admission: ObjectIdentity
    reachability: ObjectIdentity
    observation_experiment: ObjectIdentity
    fiber_assessment: ObjectIdentity
    equivalence_assessment: ObjectIdentity
    certificate: ObjectIdentity
    controller_study: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        require_sorted_unique_ids(
            self.response_laws,
            attribute="object_id",
            field_name="response_laws",
        )
        if not self.response_laws:
            raise ValueError("receiver geometry binding requires response laws")
        expected = (
            (self.observation_experiment, ReceiverObservationExperiment.SCHEMA),
            (self.fiber_assessment, ReceiverFiberAssessment.SCHEMA),
            (
                self.equivalence_assessment,
                BoundedControlEquivalenceAssessment.SCHEMA,
            ),
            (self.certificate, AmbiguityActionCertificate.SCHEMA),
        )
        if any(identity.object_schema != schema for identity, schema in expected):
            raise ValueError("receiver geometry binding changes a current contract schema")


def decode_receiver_observation_experiment(
    payload: bytes,
) -> ReceiverObservationExperiment:
    return decode_canonical_bytes(
        payload,
        ReceiverObservationExperiment,
        maximum_bytes=MAX_RECEIVER_GEOMETRY_CONTROL_BYTES,
    )


def decode_receiver_fiber_assessment(payload: bytes) -> ReceiverFiberAssessment:
    return decode_canonical_bytes(
        payload,
        ReceiverFiberAssessment,
        maximum_bytes=MAX_RECEIVER_GEOMETRY_CONTROL_BYTES,
    )


def decode_bounded_control_equivalence(
    payload: bytes,
) -> BoundedControlEquivalenceAssessment:
    return decode_canonical_bytes(
        payload,
        BoundedControlEquivalenceAssessment,
        maximum_bytes=MAX_RECEIVER_GEOMETRY_CONTROL_BYTES,
    )


def decode_ambiguity_action_certificate(
    payload: bytes,
) -> AmbiguityActionCertificate:
    return decode_canonical_bytes(
        payload,
        AmbiguityActionCertificate,
        maximum_bytes=MAX_RECEIVER_GEOMETRY_CONTROL_BYTES,
    )


__all__ = [
    "AmbiguityActionCertificate",
    "AmbiguityCertificateDisposition",
    "BoundedControlEquivalenceAssessment",
    "BranchTopologyDisposition",
    "CandidateActionEvidence",
    "CandidateDecisionDisposition",
    "CandidateEquivalenceRelation",
    "CandidatePairEquivalence",
    "CandidateViewDecisionSet",
    "DecisionEquivalenceClass",
    "EquivalenceDisposition",
    "LocalRegularityDisposition",
    "ObservationRepresentationKind",
    "ReceiverBranchTopologyAssessment",
    "ReceiverCandidateView",
    "ReceiverFiberAssessment",
    "ReceiverFiberDisposition",
    "ReceiverGeometryControlBinding",
    "ReceiverObservationExperiment",
    "decode_ambiguity_action_certificate",
    "decode_bounded_control_equivalence",
    "decode_receiver_fiber_assessment",
    "decode_receiver_observation_experiment",
    "derive_decision_equivalence",
    "robust_action_keys",
    "validate_information_refinement",
    "validate_robustness_view_enrichment",
]

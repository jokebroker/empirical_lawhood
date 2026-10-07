"""Prospective coordinate, receiver-fibre and retained-history challenges."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.metatheory import MetatheoryAggregateDisposition, MetatheoryCellDisposition, MetatheoryEvidenceCeiling, MetatheoryMethodSelection
from empirical_lawhood.planning.scientific_source_qualification import ScientificSourceQualificationResult


class CoordinateSemanticRole(StrEnum):
    DENOMINATOR = "DENOMINATOR"
    HISTORY = "HISTORY"
    ACTION = "ACTION"
    RECEIVER = "RECEIVER"
    HORIZON = "HORIZON"


class CoordinateChallengeKind(StrEnum):
    ONE_FACTOR_EXCHANGE = "ONE_FACTOR_EXCHANGE"
    COORDINATE_TOURNAMENT = "COORDINATE_TOURNAMENT"
    INFORMATIVE_FIBRE = "INFORMATIVE_FIBRE"
    TARGETED_COLLISION = "TARGETED_COLLISION"
    UNTOUCHED_PREVALENCE = "UNTOUCHED_PREVALENCE"
    RANK_AND_CONDITIONING = "RANK_AND_CONDITIONING"


class SufficiencyPredicate(StrEnum):
    PRESENT_MEASUREMENT = "PRESENT_MEASUREMENT"
    DYNAMICAL_OR_MARKOV = "DYNAMICAL_OR_MARKOV"
    DECISION = "DECISION"


class CoordinateSamplingMode(StrEnum):
    TARGETED = "TARGETED"
    UNTOUCHED = "UNTOUCHED"


@dataclass(frozen=True, slots=True)
class CoordinateCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/coordinate-candidate'

    candidate_id: str
    semantic_roles: tuple[CoordinateSemanticRole, ...]
    field_ids: tuple[str, ...]
    endpoint_ids: tuple[str, ...]
    property_ids: tuple[str, ...]
    receiver_id: str
    action_word_id: str
    horizon_id: str
    resolution: NamedDecimal
    normalization_id: str
    equivalence_relation_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if (
            tuple(sorted(set(self.semantic_roles), key=lambda value: value.value))
            != self.semantic_roles
        ):
            raise ValueError("coordinate semantic roles must be sorted and unique")
        if not self.semantic_roles:
            raise ValueError("coordinate candidate requires semantic roles")
        for name, values in (
            ("field_ids", self.field_ids),
            ("endpoint_ids", self.endpoint_ids),
            ("property_ids", self.property_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        for name in (
            "receiver_id",
            "action_word_id",
            "horizon_id",
            "normalization_id",
            "equivalence_relation_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class CoordinateChallengeSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/coordinate-challenge-spec'

    spec_id: str
    source_qualification: ObjectIdentity | None
    source_qualification_not_applicable_reasons: tuple[str, ...]
    candidates: tuple[CoordinateCandidate, ...]
    comparator_family_id: str
    challenge_kind: CoordinateChallengeKind
    sufficiency_predicates: tuple[SufficiencyPredicate, ...]
    sampling_mode: CoordinateSamplingMode
    physical_unit_ids: tuple[str, ...]
    causal_cutoff: ObjectIdentity
    collision_tolerance: NamedDecimal
    equivalence_tolerance: NamedDecimal
    history_depths: tuple[int, ...]
    normalized_information_budgets: tuple[NamedDecimal, ...]
    future_action_panel_id: str
    future_horizon_panel_id: str
    decision_reference_id: str
    construction_method: MetatheoryMethodSelection
    adjudication_method: MetatheoryMethodSelection
    falsifier_ids: tuple[str, ...]
    targetability_rule_id: str
    power_rule_id: str
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        if self.source_qualification is None:
            require_sorted_unique_strings(
                self.source_qualification_not_applicable_reasons,
                field_name="source_qualification_not_applicable_reasons",
                allow_empty=False,
            )
        elif (
            self.source_qualification.object_schema != ScientificSourceQualificationResult.SCHEMA
            or self.source_qualification_not_applicable_reasons
        ):
            raise ValueError("coordinate challenge source qualification binding is inconsistent")
        require_sorted_unique_ids(
            self.candidates, attribute="candidate_id", field_name="candidates"
        )
        if not self.candidates:
            raise ValueError("coordinate challenge requires candidates")
        validate_stable_id(self.comparator_family_id, field_name="comparator_family_id")
        if (
            tuple(sorted(set(self.sufficiency_predicates), key=lambda value: value.value))
            != self.sufficiency_predicates
        ):
            raise ValueError("sufficiency predicates must be sorted and unique")
        if not self.sufficiency_predicates:
            raise ValueError("coordinate challenge requires a sufficiency predicate")
        require_sorted_unique_strings(
            self.physical_unit_ids,
            field_name="physical_unit_ids",
            allow_empty=False,
        )
        if (
            not self.history_depths
            or tuple(sorted(set(self.history_depths))) != self.history_depths
            or any(type(value) is not int or value <= 0 for value in self.history_depths)
        ):
            raise ValueError("history depths must be sorted unique positive integers")
        require_sorted_unique_ids(
            self.normalized_information_budgets,
            attribute="value_id",
            field_name="normalized_information_budgets",
        )
        for name in (
            "future_action_panel_id",
            "future_horizon_panel_id",
            "decision_reference_id",
            "targetability_rule_id",
            "power_rule_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.construction_method == self.adjudication_method:
            raise ValueError("coordinate construction and adjudication require separate selections")
        require_sorted_unique_strings(
            self.falsifier_ids, field_name="falsifier_ids", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class CoordinateChallengeNomination(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/coordinate-challenge-nomination'

    nomination_id: str
    challenge_spec: ObjectIdentity
    candidate: ObjectIdentity
    physical_unit_id: str
    pair_id: str | None
    fibre_id: str | None
    targetable: bool
    selection_lineage: tuple[ObjectIdentity, ...]
    construction_method: MetatheoryMethodSelection
    target_outcomes_read: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        if self.challenge_spec.object_schema != CoordinateChallengeSpec.SCHEMA:
            raise ValueError("coordinate nomination names another challenge schema")
        if self.candidate.object_schema != CoordinateCandidate.SCHEMA:
            raise ValueError("coordinate nomination names another candidate schema")
        validate_stable_id(self.physical_unit_id, field_name="physical_unit_id")
        if self.targetable:
            if self.pair_id is None or self.fibre_id is None:
                raise ValueError("targetable coordinate nomination lacks pair/fibre identity")
            validate_stable_id(self.pair_id, field_name="pair_id")
            validate_stable_id(self.fibre_id, field_name="fibre_id")
        elif self.pair_id is not None or self.fibre_id is not None:
            raise ValueError("untargetable coordinate nomination cannot invent a fibre")
        require_sorted_unique_ids(
            self.selection_lineage,
            attribute="object_id",
            field_name="selection_lineage",
        )
        if self.target_outcomes_read or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("coordinate nomination must precede target outcomes")


@dataclass(frozen=True, slots=True)
class CoordinateChallengeEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/coordinate-challenge-evidence'

    evidence_id: str
    challenge_spec: ObjectIdentity
    nomination: ObjectIdentity
    candidate: ObjectIdentity
    physical_unit_id: str
    sampling_mode: CoordinateSamplingMode
    informative_collision_entered: bool
    present_equal: bool | None
    future_response_equal: bool | None
    decision_equal: bool | None
    endpoint_saturated: bool
    action_occurrence: ObjectIdentity
    exact_structural_rank: int
    numerical_rank_lower: int
    numerical_rank_upper: int
    requested_row_count: int
    usable_conditioned_rank: int
    conditioning: NamedDecimal
    adjudication_method: MetatheoryMethodSelection
    publication: ObjectIdentity
    recovery: ObjectIdentity
    evidence_links: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        if self.challenge_spec.object_schema != CoordinateChallengeSpec.SCHEMA:
            raise ValueError("coordinate evidence names another challenge schema")
        if self.nomination.object_schema != CoordinateChallengeNomination.SCHEMA:
            raise ValueError("coordinate evidence names another nomination schema")
        if self.candidate.object_schema != CoordinateCandidate.SCHEMA:
            raise ValueError("coordinate evidence names another candidate schema")
        validate_stable_id(self.physical_unit_id, field_name="physical_unit_id")
        if self.informative_collision_entered:
            if self.present_equal is not True:
                raise ValueError("entered informative collision requires present equality")
        elif any(
            value is not None
            for value in (self.present_equal, self.future_response_equal, self.decision_equal)
        ):
            raise ValueError("nonentered collision cannot carry sufficiency outcomes")
        for name in (
            "exact_structural_rank",
            "numerical_rank_lower",
            "numerical_rank_upper",
            "requested_row_count",
            "usable_conditioned_rank",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not (
            self.numerical_rank_lower
            <= self.usable_conditioned_rank
            <= self.numerical_rank_upper
            <= self.requested_row_count
        ):
            raise ValueError("coordinate numerical rank bracket is inconsistent")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )


@dataclass(frozen=True, slots=True)
class MetatheoryCoordinateConstructionProduct(CanonicalRecord):
    """Outcome-blind nominations and evidence prepared for sole adjudication."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-coordinate-construction-product'

    product_id: str
    challenge_spec: ObjectIdentity
    source_qualification: ObjectIdentity | None
    nominations: tuple[CoordinateChallengeNomination, ...]
    evidence: tuple[CoordinateChallengeEvidence, ...]
    construction_method: MetatheoryMethodSelection
    target_outcomes_read: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.product_id, field_name="product_id")
        if self.challenge_spec.object_schema != CoordinateChallengeSpec.SCHEMA:
            raise ValueError("coordinate construction product names another challenge schema")
        if self.source_qualification is not None and (
            self.source_qualification.object_schema != ScientificSourceQualificationResult.SCHEMA
        ):
            raise ValueError("coordinate construction product names another source result")
        require_sorted_unique_ids(
            self.nominations,
            attribute="nomination_id",
            field_name="nominations",
        )
        require_sorted_unique_ids(self.evidence, attribute="evidence_id", field_name="evidence")
        if not self.nominations or len(self.nominations) != len(self.evidence):
            raise ValueError("coordinate construction product requires paired nonempty operands")
        if (
            self.target_outcomes_read
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or self.grants_authority
        ):
            raise ValueError("coordinate construction product crossed its prospective boundary")


@dataclass(frozen=True, slots=True)
class CoordinateChallengeCellResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/coordinate-challenge-cell-result'

    cell_id: str
    challenge_spec: ObjectIdentity
    candidate: ObjectIdentity
    physical_unit_id: str
    evidence: ObjectIdentity
    informative_collision_entered: bool
    present_sufficiency: MetatheoryCellDisposition
    dynamical_sufficiency: MetatheoryCellDisposition
    decision_sufficiency: MetatheoryCellDisposition
    adverse_collision_count: int
    consistent_collision_count: int
    targetable: bool
    prevalence_applicable: bool
    endpoint_saturated: bool
    exact_structural_rank: int
    numerical_rank_lower: int
    numerical_rank_upper: int
    usable_conditioned_rank: int
    disposition: MetatheoryCellDisposition
    decisive_witness_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        validate_stable_id(self.physical_unit_id, field_name="physical_unit_id")
        for name in ("adverse_collision_count", "consistent_collision_count"):
            value = getattr(self, name)
            if value not in (0, 1):
                raise ValueError("coordinate cell collision count must be zero or one")
        if self.prevalence_applicable and not self.targetable:
            raise ValueError("untargetable unit cannot enter prevalence numerator")
        require_sorted_unique_strings(
            self.decisive_witness_ids,
            field_name="decisive_witness_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class CoordinateChallengeResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/coordinate-challenge-result'

    result_id: str
    challenge_spec: ObjectIdentity
    nominations: tuple[CoordinateChallengeNomination, ...]
    evidence: tuple[CoordinateChallengeEvidence, ...]
    cells: tuple[CoordinateChallengeCellResult, ...]
    targeted_cell_count: int
    untouched_cell_count: int
    untargetable_unit_count: int
    disposition: MetatheoryAggregateDisposition
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    evidence_links: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        require_sorted_unique_ids(
            self.nominations, attribute="nomination_id", field_name="nominations"
        )
        require_sorted_unique_ids(self.evidence, attribute="evidence_id", field_name="evidence")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if (
            not self.cells
            or len(self.cells) != len(self.nominations)
            or len(self.cells) != len(self.evidence)
        ):
            raise ValueError("coordinate result must preserve every nominated complete-unit cell")
        if self.targeted_cell_count + self.untouched_cell_count != len(self.cells):
            raise ValueError("coordinate result sampling counts differ from cells")
        if self.disposition not in {
            MetatheoryAggregateDisposition.SUPPORTED,
            MetatheoryAggregateDisposition.OPPOSED,
            MetatheoryAggregateDisposition.MIXED,
            MetatheoryAggregateDisposition.UNEVALUABLE,
        }:
            raise ValueError("coordinate result aggregate disposition is invalid")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )


__all__ = [
    'CoordinateCandidate',
    'CoordinateChallengeCellResult',
    'CoordinateChallengeEvidence',
    'CoordinateChallengeKind',
    'CoordinateChallengeNomination',
    'CoordinateChallengeResult',
    'CoordinateChallengeSpec',
    'CoordinateSamplingMode',
    'CoordinateSemanticRole',
    'MetatheoryCoordinateConstructionProduct',
    'SufficiencyPredicate',
]

"""Outcome-blind formal-gap planning over the accepted experiment ontology.

These records are additive planning metadata.  They bind existing evidence
world, readiness and evidence-ceiling values and do not redefine an
``ExperimentSpec`` or a ``ProgrammeDraft``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.worlds import EvidenceUnitScope, WorldKind


class FormalDomain(StrEnum):
    ALGEBRA = "ALGEBRA"
    CALCULUS = "CALCULUS"
    GEOMETRY = "GEOMETRY"
    DYNAMICS = "DYNAMICS"


class FormalGapEvidenceWorld(StrEnum):
    """Portfolio world overlay with an explicit map to accepted ``WorldKind``."""

    ANALYTIC_REFERENCE = "ANALYTIC_REFERENCE"
    NUMERICAL_SIMULATOR = "NUMERICAL_SIMULATOR"
    RETROSPECTIVE_DATASET = "RETROSPECTIVE_DATASET"
    HIL_LIVE = "HIL_LIVE"
    NEW_PHYSICAL_MATTER = "NEW_PHYSICAL_MATTER"

    @property
    def accepted_world_kind(self) -> WorldKind:
        return {
            FormalGapEvidenceWorld.ANALYTIC_REFERENCE: WorldKind.ANALYTIC_REFERENCE,
            FormalGapEvidenceWorld.NUMERICAL_SIMULATOR: WorldKind.NUMERICAL_SIMULATOR,
            FormalGapEvidenceWorld.RETROSPECTIVE_DATASET: WorldKind.PHYSICAL_EXPERIMENT,
            FormalGapEvidenceWorld.HIL_LIVE: WorldKind.HARDWARE_IN_LOOP,
            FormalGapEvidenceWorld.NEW_PHYSICAL_MATTER: WorldKind.PHYSICAL_EXPERIMENT,
        }[self]

    @property
    def excluded_from_portfolio(self) -> bool:
        return self in {
            FormalGapEvidenceWorld.HIL_LIVE,
            FormalGapEvidenceWorld.NEW_PHYSICAL_MATTER,
        }


class FormalGapCoverageDisposition(StrEnum):
    TEST_IN_THIS_ACT = "TEST_IN_THIS_ACT"
    DEFER_WITH_TYPED_PREREQUISITE = "DEFER_WITH_TYPED_PREREQUISITE"
    UNEVALUABLE_FROM_SOURCE = "UNEVALUABLE_FROM_SOURCE"
    NOT_APPLICABLE_TO_DENOMINATOR = "NOT_APPLICABLE_TO_DENOMINATOR"
    OUT_OF_SCOPE_WORLD = "OUT_OF_SCOPE_WORLD"


_UNEVALUABLE_READINESS = {
    ReadinessStatus.REQUIRES_NEW_DATA,
    ReadinessStatus.SOURCE_PREREQUISITE_NOT_MET,
}
_DEFERRED_READINESS = {
    ReadinessStatus.REQUIRES_PROTOCOL_FREEZE,
    ReadinessStatus.PREREQUISITE_NOT_MET,
    ReadinessStatus.SOURCE_PREREQUISITE_NOT_MET,
    ReadinessStatus.AUTHORITY_REQUIRED,
    ReadinessStatus.COMPUTABILITY_BOUNDARY,
}


@dataclass(frozen=True, slots=True)
class FormalGapSpec(CanonicalRecord):
    """One stable formal question and its pre-outcome feasibility contract."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-gap-spec'

    gap_id: str
    domain: FormalDomain
    question_family: str
    required_operand_ids: tuple[str, ...]
    compatible_evidence_worlds: tuple[FormalGapEvidenceWorld, ...]
    minimum_independent_units: int
    minimum_numerical_views: int
    estimator_family_ids: tuple[str, ...]
    control_ids: tuple[str, ...]
    decisive_falsifier_ids: tuple[str, ...]
    support_prerequisite_ids: tuple[str, ...]
    multiplicity_family_id: str
    maximum_claim_ceiling: EvidenceCeiling
    provenance_source_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.gap_id, field_name="gap_id")
        validate_nonempty(self.question_family, field_name="question_family")
        for name, values in (
            ("required_operand_ids", self.required_operand_ids),
            ("estimator_family_ids", self.estimator_family_ids),
            ("control_ids", self.control_ids),
            ("decisive_falsifier_ids", self.decisive_falsifier_ids),
            ("support_prerequisite_ids", self.support_prerequisite_ids),
            ("provenance_source_ids", self.provenance_source_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        worlds = tuple(value.value for value in self.compatible_evidence_worlds)
        if tuple(sorted(set(worlds))) != worlds:
            raise ValueError("compatible_evidence_worlds must be sorted and unique")
        if not worlds:
            raise ValueError("formal gap requires a compatible evidence world")
        if self.minimum_independent_units <= 0:
            raise ValueError("minimum_independent_units must be positive")
        if self.minimum_numerical_views < 0:
            raise ValueError("minimum_numerical_views must be nonnegative")
        validate_stable_id(self.multiplicity_family_id, field_name="multiplicity_family_id")


@dataclass(frozen=True, slots=True)
class FormalGapRegister(CanonicalRecord):
    """Versioned, source-provenanced inventory of formal questions."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-gap-register'

    register_id: str
    register_version: str
    provenance_sources: tuple[ObjectIdentity, ...]
    gaps: tuple[FormalGapSpec, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.register_id, field_name="register_id")
        validate_semantic_version(self.register_version)
        require_sorted_unique_ids(
            self.provenance_sources,
            attribute="object_id",
            field_name="provenance_sources",
        )
        require_sorted_unique_ids(self.gaps, attribute="gap_id", field_name="gaps")
        if not self.provenance_sources or not self.gaps:
            raise ValueError("formal gap register requires sources and gaps")
        source_ids = {value.object_id for value in self.provenance_sources}
        if any(not set(gap.provenance_source_ids).issubset(source_ids) for gap in self.gaps):
            raise ValueError("formal gap cites an absent provenance source")
        if {gap.domain for gap in self.gaps} != set(FormalDomain):
            raise ValueError("formal gap register must cover all four formal domains")


@dataclass(frozen=True, slots=True)
class FormalGapApplicability(CanonicalRecord):
    """Outcome-blind facts determining whether one gap is feasible in one act."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-gap-applicability'

    gap_id: str
    evidence_world: FormalGapEvidenceWorld
    present_operand_ids: tuple[str, ...]
    satisfied_prerequisite_ids: tuple[str, ...]
    independent_unit_ids: tuple[str, ...]
    independent_unit_scope: EvidenceUnitScope
    numerical_view_ids: tuple[str, ...]
    available_estimator_family_ids: tuple[str, ...]
    available_control_ids: tuple[str, ...]
    multiplicity_family_ids: tuple[str, ...]
    requested_claim_ceiling: EvidenceCeiling
    denominator_applicable: bool
    resource_envelope_satisfied: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.gap_id, field_name="gap_id")
        for name, values in (
            ("present_operand_ids", self.present_operand_ids),
            ("satisfied_prerequisite_ids", self.satisfied_prerequisite_ids),
            ("independent_unit_ids", self.independent_unit_ids),
            ("numerical_view_ids", self.numerical_view_ids),
            ("available_estimator_family_ids", self.available_estimator_family_ids),
            ("available_control_ids", self.available_control_ids),
            ("multiplicity_family_ids", self.multiplicity_family_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("formal-gap applicability must be outcome-blind")

    def is_feasible(self, gap: FormalGapSpec) -> bool:
        """Return exact pre-outcome feasibility without interpreting outcomes."""

        if self.gap_id != gap.gap_id:
            raise ValueError("applicability binds another formal gap")
        return (
            not self.evidence_world.excluded_from_portfolio
            and self.evidence_world in gap.compatible_evidence_worlds
            and self.denominator_applicable
            and self.resource_envelope_satisfied
            and set(gap.required_operand_ids).issubset(self.present_operand_ids)
            and set(gap.support_prerequisite_ids).issubset(self.satisfied_prerequisite_ids)
            and self.independent_unit_scope is EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT
            and len(self.independent_unit_ids) >= gap.minimum_independent_units
            and len(self.numerical_view_ids) >= gap.minimum_numerical_views
            and bool(set(gap.estimator_family_ids) & set(self.available_estimator_family_ids))
            and set(gap.control_ids).issubset(self.available_control_ids)
            and gap.multiplicity_family_id in self.multiplicity_family_ids
            and EvidenceCeiling.lowest(
                self.requested_claim_ceiling,
                gap.maximum_claim_ceiling,
            )
            is self.requested_claim_ceiling
        )


@dataclass(frozen=True, slots=True)
class FormalGapCoverageAssignment(CanonicalRecord):
    """One complete outcome-blind disposition for a registered formal gap."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-gap-coverage-assignment'

    gap_id: str
    disposition: FormalGapCoverageDisposition
    readiness_reason: ReadinessStatus | None
    reason_codes: tuple[str, ...]
    selected_estimator_family_id: str | None
    selected_control_ids: tuple[str, ...]
    selected_multiplicity_family_id: str | None
    obligation_ids: tuple[str, ...]
    output_ids: tuple[str, ...]
    adjudication_owner_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.gap_id, field_name="gap_id")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=self.disposition is FormalGapCoverageDisposition.TEST_IN_THIS_ACT,
        )
        for name, values in (
            ("selected_control_ids", self.selected_control_ids),
            ("obligation_ids", self.obligation_ids),
            ("output_ids", self.output_ids),
            ("adjudication_owner_ids", self.adjudication_owner_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        selected = self.disposition is FormalGapCoverageDisposition.TEST_IN_THIS_ACT
        selected_scalars = (
            self.selected_estimator_family_id,
            self.selected_multiplicity_family_id,
        )
        if selected:
            if self.readiness_reason is not None or any(
                value is None for value in selected_scalars
            ):
                raise ValueError("tested formal gap requires selected estimator/multiplicity")
            if not (
                self.selected_control_ids
                and self.obligation_ids
                and self.output_ids
                and self.adjudication_owner_ids
            ):
                raise ValueError("tested formal gap requires complete lowering owners")
            validate_stable_id(
                self.selected_estimator_family_id or "",
                field_name="selected_estimator_family_id",
            )
            validate_stable_id(
                self.selected_multiplicity_family_id or "",
                field_name="selected_multiplicity_family_id",
            )
        elif any(value is not None for value in selected_scalars) or any(
            (
                self.selected_control_ids,
                self.obligation_ids,
                self.output_ids,
                self.adjudication_owner_ids,
            )
        ):
            raise ValueError("excluded formal gap cannot lower scientific work")


@dataclass(frozen=True, slots=True)
class FormalGapCoverage(CanonicalRecord):
    """Complete coverage decision for one denominator-local candidate act."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-gap-coverage'

    coverage_id: str
    register: ObjectIdentity
    denominator_id: str
    candidate_act_id: str
    applicability: tuple[FormalGapApplicability, ...]
    assignments: tuple[FormalGapCoverageAssignment, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("coverage_id", self.coverage_id),
            ("denominator_id", self.denominator_id),
            ("candidate_act_id", self.candidate_act_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.register.object_schema != FormalGapRegister.SCHEMA:
            raise ValueError("coverage must identify a formal-gap register")
        require_sorted_unique_ids(
            self.applicability,
            attribute="gap_id",
            field_name="applicability",
        )
        require_sorted_unique_ids(
            self.assignments,
            attribute="gap_id",
            field_name="assignments",
        )
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("formal-gap coverage selection must be outcome-blind")


def validate_formal_gap_coverage(
    register: FormalGapRegister,
    coverage: FormalGapCoverage,
) -> None:
    """Require every feasible gap to lower and every exclusion to be typed."""

    if coverage.register != ObjectIdentity.from_record(register.register_id, register):
        raise ValueError("formal-gap coverage binds another register identity")
    gaps = {value.gap_id: value for value in register.gaps}
    applicability = {value.gap_id: value for value in coverage.applicability}
    assignments = {value.gap_id: value for value in coverage.assignments}
    if set(applicability) != set(gaps) or set(assignments) != set(gaps):
        raise ValueError("formal-gap coverage must classify every registered gap exactly once")

    for gap_id, gap in gaps.items():
        facts = applicability[gap_id]
        assignment = assignments[gap_id]
        feasible = facts.is_feasible(gap)
        if feasible:
            if assignment.disposition is not FormalGapCoverageDisposition.TEST_IN_THIS_ACT:
                raise ValueError(f"feasible formal gap was silently excluded: {gap_id}")
            if assignment.selected_estimator_family_id not in gap.estimator_family_ids:
                raise ValueError("selected estimator is outside the formal-gap contract")
            if assignment.selected_estimator_family_id not in facts.available_estimator_family_ids:
                raise ValueError("selected estimator is unavailable in the candidate act")
            if assignment.selected_control_ids != gap.control_ids:
                raise ValueError("tested formal gap must bind every predeclared control")
            if assignment.selected_multiplicity_family_id != gap.multiplicity_family_id:
                raise ValueError("tested formal gap binds another multiplicity family")
            continue

        if assignment.disposition is FormalGapCoverageDisposition.TEST_IN_THIS_ACT:
            raise ValueError(f"unsupported formal gap cannot be tested: {gap_id}")
        if facts.evidence_world.excluded_from_portfolio:
            if assignment.disposition is not FormalGapCoverageDisposition.OUT_OF_SCOPE_WORLD:
                raise ValueError("HIL/live/new physical gap must be out of portfolio scope")
            if assignment.readiness_reason is not None:
                raise ValueError("out-of-scope world is not a readiness substitution")
        elif not facts.denominator_applicable:
            if (
                assignment.disposition
                is not FormalGapCoverageDisposition.NOT_APPLICABLE_TO_DENOMINATOR
                or assignment.readiness_reason is not None
            ):
                raise ValueError("undefined denominator object must be not-applicable")
        elif assignment.disposition is FormalGapCoverageDisposition.UNEVALUABLE_FROM_SOURCE:
            if assignment.readiness_reason not in _UNEVALUABLE_READINESS:
                raise ValueError("source-unevaluable gap requires an accepted source reason")
        elif assignment.disposition is FormalGapCoverageDisposition.DEFER_WITH_TYPED_PREREQUISITE:
            if assignment.readiness_reason not in _DEFERRED_READINESS:
                raise ValueError("deferred formal gap requires an accepted readiness reason")
        else:
            raise ValueError("formal-gap exclusion does not match its applicability facts")


__all__ = [
    "FormalDomain",
    "FormalGapApplicability",
    "FormalGapCoverage",
    "FormalGapCoverageAssignment",
    "FormalGapCoverageDisposition",
    "FormalGapEvidenceWorld",
    "FormalGapRegister",
    "FormalGapSpec",
    "validate_formal_gap_coverage",
]

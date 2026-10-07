"""Outcome-visible, read-only exploratory scientific representations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.evidence import (
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.experiments import AssignmentSpec, PrecisionGoal
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import RelationalIdentity
from empirical_lawhood.kernel.time import InformationCutoff


class AnomalyKind(StrEnum):
    SUPPORT_GAP = "SUPPORT_GAP"
    COORDINATE_FAILURE = "COORDINATE_FAILURE"
    RESIDUAL_MEMORY = "RESIDUAL_MEMORY"
    COUNTERFEIT_RANK = "COUNTERFEIT_RANK"
    ADMISSION_FRAGILITY = "ADMISSION_FRAGILITY"
    TRANSPORT_FAILURE = "TRANSPORT_FAILURE"
    CONTRADICTION = "CONTRADICTION"
    INFORMATION_LIMIT = "INFORMATION_LIMIT"
    UNEXPECTED_NULL = "UNEXPECTED_NULL"


@dataclass(frozen=True, slots=True)
class AnomalySignal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/anomaly-signal'

    signal_id: str
    snapshot_id: str
    detector_key: str
    detector_version: str
    kind: AnomalyKind
    relation_id: str
    affected_quantity_ids: tuple[str, ...]
    severity: Decimal
    rationale: str
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling
    evidence_link_ids: tuple[str, ...]
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("signal_id", self.signal_id),
            ("snapshot_id", self.snapshot_id),
            ("detector_key", self.detector_key),
            ("relation_id", self.relation_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.detector_version)
        require_sorted_unique_strings(
            self.affected_quantity_ids,
            field_name="affected_quantity_ids",
            allow_empty=False,
        )
        validate_decimal(self.severity, field_name="severity", minimum=Decimal("0"))
        validate_nonempty(self.rationale, field_name="rationale")
        require_sorted_unique_strings(
            self.evidence_link_ids, field_name="evidence_link_ids", allow_empty=False
        )
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("anomaly visibility cannot be lowered")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class SearchAxis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/search-axis'

    axis_id: str
    candidate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.axis_id, field_name="axis_id")
        require_sorted_unique_strings(
            self.candidate_ids, field_name="candidate_ids", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class AnalysisSpec(CanonicalRecord):
    """Declarative non-arbitrary analysis over one immutable snapshot."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/analysis-spec'

    analysis_id: str
    snapshot_id: str
    snapshot_fingerprint: str
    system_id: str
    relation: RelationalIdentity
    question: str
    estimand: str
    independent_unit_id: str
    projection_ids: tuple[str, ...]
    receiver_quantity_ids: tuple[str, ...]
    denominator_quantity_ids: tuple[str, ...]
    action_quantity_ids: tuple[str, ...]
    horizon_ids: tuple[str, ...]
    grouping_clock_ids: tuple[str, ...]
    registered_pipeline_key: str
    registered_pipeline_version: str
    null_capability_keys: tuple[str, ...]
    falsifier_capability_keys: tuple[str, ...]
    search_axes: tuple[SearchAxis, ...]
    uncertainty_method_key: str
    budget: ResourceBudget
    stopping_rule: str
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("analysis_id", self.analysis_id),
            ("snapshot_id", self.snapshot_id),
            ("system_id", self.system_id),
            ("independent_unit_id", self.independent_unit_id),
            ("registered_pipeline_key", self.registered_pipeline_key),
            ("uncertainty_method_key", self.uncertainty_method_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_sha256(self.snapshot_fingerprint, field_name="snapshot_fingerprint")
        validate_semantic_version(self.registered_pipeline_version)
        validate_nonempty(self.question, field_name="question")
        validate_nonempty(self.estimand, field_name="estimand")
        for field_name, values in (
            ("projection_ids", self.projection_ids),
            ("receiver_quantity_ids", self.receiver_quantity_ids),
            ("denominator_quantity_ids", self.denominator_quantity_ids),
            ("action_quantity_ids", self.action_quantity_ids),
            ("horizon_ids", self.horizon_ids),
            ("grouping_clock_ids", self.grouping_clock_ids),
            ("null_capability_keys", self.null_capability_keys),
            ("falsifier_capability_keys", self.falsifier_capability_keys),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        require_sorted_unique_ids(self.search_axes, attribute="axis_id", field_name="search_axes")
        if not self.search_axes:
            raise ValueError("analysis must declare its complete search family")
        validate_nonempty(self.stopping_rule, field_name="stopping_rule")
        self._validate_relation()
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("analysis visibility cannot be lowered")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("exploratory analysis must be non-promotable")
        require_extensions(self.extensions)

    def _validate_relation(self) -> None:
        if self.receiver_quantity_ids != self.relation.receiver_quantity_ids:
            raise ValueError("analysis receivers differ from relational identity")
        if self.denominator_quantity_ids != self.relation.denominator_quantity_ids:
            raise ValueError("analysis denominators differ from relational identity")
        if self.action_quantity_ids != self.relation.action_quantity_ids:
            raise ValueError("analysis actions differ from relational identity")
        if (self.relation.horizon.horizon_id,) != self.horizon_ids:
            raise ValueError("analysis horizons differ from relational identity")


@dataclass(frozen=True, slots=True)
class AnalysisProposal(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/analysis-proposal'

    proposal_id: str
    analysis: AnalysisSpec
    anomaly_signal_ids: tuple[str, ...]
    intent: str
    prerequisite_ids: tuple[str, ...]
    selection_rationale: str
    extensions: tuple[ExtensionBinding, ...] = ()

    @property
    def snapshot_id(self) -> str:
        return self.analysis.snapshot_id

    def __post_init__(self) -> None:
        validate_stable_id(self.proposal_id, field_name="proposal_id")
        require_sorted_unique_strings(
            self.anomaly_signal_ids,
            field_name="anomaly_signal_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.prerequisite_ids, field_name="prerequisite_ids")
        validate_nonempty(self.intent, field_name="intent")
        validate_nonempty(self.selection_rationale, field_name="selection_rationale")
        require_extensions(self.extensions)


class ProposalDisposition(StrEnum):
    SELECTED = "SELECTED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    SUPERSEDED = "SUPERSEDED"


@dataclass(frozen=True, slots=True)
class ProposalSelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/proposal-selection'

    proposal_id: str
    disposition: ProposalDisposition
    reason_codes: tuple[str, ...]
    planner_key: str

    def __post_init__(self) -> None:
        validate_stable_id(self.proposal_id, field_name="proposal_id")
        validate_stable_id(self.planner_key, field_name="planner_key")
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class ExplorationPlan(CanonicalRecord):
    """One frozen, budgeted, complete outcome-visible analysis wave."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/exploration-plan'

    plan_id: str
    snapshot: ObjectIdentity
    snapshot_visibility_ceiling: VisibilityCeiling
    proposals: tuple[AnalysisProposal, ...]
    selections: tuple[ProposalSelection, ...]
    information_cutoff: InformationCutoff
    budget: ResourceBudget
    search_family_sha256: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    frozen: bool = True
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.plan_id, field_name="plan_id")
        require_sorted_unique_ids(self.proposals, attribute="proposal_id", field_name="proposals")
        if not self.proposals:
            raise ValueError("exploration plan requires proposals")
        require_sorted_unique_ids(self.selections, attribute="proposal_id", field_name="selections")
        proposal_ids = {proposal.proposal_id for proposal in self.proposals}
        selection_ids = {selection.proposal_id for selection in self.selections}
        if proposal_ids != selection_ids:
            raise ValueError("every proposal requires exactly one disposition")
        snapshot_ids = {proposal.snapshot_id for proposal in self.proposals}
        if snapshot_ids != {self.snapshot.object_id}:
            raise ValueError("all exploration proposals must bind the plan snapshot")
        if not any(
            selection.disposition is ProposalDisposition.SELECTED for selection in self.selections
        ):
            raise ValueError("exploration plan must select at least one analysis")
        validate_sha256(self.search_family_sha256, field_name="search_family_sha256")
        inherited = inherited_visibility((self.snapshot_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("exploration-plan visibility cannot be lowered")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("exploration plans must be non-promotable")
        if not self.frozen:
            raise ValueError("an ExplorationPlan must be frozen")
        require_extensions(self.extensions)


class AnalysisAttemptStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    NULL = "NULL"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class AnalysisAttempt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/analysis-attempt'

    attempt_id: str
    proposal_id: str
    analysis_id: str
    family_coordinate_ids: tuple[str, ...]
    status: AnalysisAttemptStatus
    artifact_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("attempt_id", self.attempt_id),
            ("proposal_id", self.proposal_id),
            ("analysis_id", self.analysis_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.family_coordinate_ids,
            field_name="family_coordinate_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.artifact_ids, field_name="artifact_ids")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is AnalysisAttemptStatus.SUCCEEDED and not self.artifact_ids:
            raise ValueError("a successful analysis attempt requires artifacts")
        if self.status is not AnalysisAttemptStatus.SUCCEEDED and not self.reason_codes:
            raise ValueError("a non-success attempt requires reason codes")


@dataclass(frozen=True, slots=True)
class EffectEstimate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/effect-estimate'

    effect_id: str
    quantity_id: str
    native_unit: str
    point: Decimal | None
    lower: Decimal | None
    upper: Decimal | None
    physical_independent_unit_count: int
    descriptive_only: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.effect_id, field_name="effect_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        for name, value in (
            ("point", self.point),
            ("lower", self.lower),
            ("upper", self.upper),
        ):
            if value is not None:
                validate_decimal(value, field_name=name)
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("effect interval lower exceeds upper")
        if self.physical_independent_unit_count <= 0:
            raise ValueError("effect requires physical independent units")
        if not self.descriptive_only:
            raise ValueError("exploratory effects must remain descriptive only")


class ExploratoryFindingStatus(StrEnum):
    PATTERN = "PATTERN"
    NULL = "NULL"
    MIXED = "MIXED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ExploratoryFinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/exploratory-finding'

    finding_id: str
    plan_id: str
    proposal_id: str
    status: ExploratoryFindingStatus
    attempts: tuple[AnalysisAttempt, ...]
    effects: tuple[EffectEstimate, ...]
    interpretation: str
    limitation_codes: tuple[str, ...]
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("finding_id", self.finding_id),
            ("plan_id", self.plan_id),
            ("proposal_id", self.proposal_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(self.attempts, attribute="attempt_id", field_name="attempts")
        if not self.attempts:
            raise ValueError("a finding must retain its complete attempts")
        require_sorted_unique_ids(self.effects, attribute="effect_id", field_name="effects")
        validate_nonempty(self.interpretation, field_name="interpretation")
        require_sorted_unique_strings(
            self.limitation_codes,
            field_name="limitation_codes",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("finding visibility cannot be lowered")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("exploratory findings must be non-promotable")
        require_extensions(self.extensions)


class HypothesisDisposition(StrEnum):
    PREFERRED = "PREFERRED"
    OPPOSED = "OPPOSED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class Hypothesis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/hypothesis'

    hypothesis_id: str
    statement: str
    disposition: HypothesisDisposition
    supporting_finding_ids: tuple[str, ...]
    opposing_observation: str
    missing_evidence: str

    def __post_init__(self) -> None:
        validate_stable_id(self.hypothesis_id, field_name="hypothesis_id")
        validate_nonempty(self.statement, field_name="statement")
        require_sorted_unique_strings(
            self.supporting_finding_ids,
            field_name="supporting_finding_ids",
            allow_empty=False,
        )
        validate_nonempty(self.opposing_observation, field_name="opposing_observation")
        validate_nonempty(self.missing_evidence, field_name="missing_evidence")


@dataclass(frozen=True, slots=True)
class HypothesisSet(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/hypothesis-set'

    hypothesis_set_id: str
    finding_ids: tuple[str, ...]
    hypotheses: tuple[Hypothesis, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.hypothesis_set_id, field_name="hypothesis_set_id")
        require_sorted_unique_strings(self.finding_ids, field_name="finding_ids", allow_empty=False)
        require_sorted_unique_ids(
            self.hypotheses, attribute="hypothesis_id", field_name="hypotheses"
        )
        if len(self.hypotheses) < 2:
            raise ValueError("a HypothesisSet must retain competing explanations")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("hypothesis-set visibility cannot be lowered")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("exploratory hypotheses must be non-promotable")
        require_extensions(self.extensions)


class OutcomeInterpretationKind(StrEnum):
    SUPPORTING = "SUPPORTING"
    OPPOSING = "OPPOSING"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class OutcomeInterpretation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/outcome-interpretation'

    interpretation_id: str
    kind: OutcomeInterpretationKind
    criterion: str

    def __post_init__(self) -> None:
        validate_stable_id(self.interpretation_id, field_name="interpretation_id")
        validate_nonempty(self.criterion, field_name="criterion")


@dataclass(frozen=True, slots=True)
class ProspectiveNomination(CanonicalRecord):
    """Outcome-visible advisory bridge to a genuinely fresh experiment."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-nomination'

    nomination_id: str
    source_hypothesis_set: ObjectIdentity
    relation: RelationalIdentity
    question: str
    hypothesis_ids: tuple[str, ...]
    assignment: AssignmentSpec
    measurement_quantity_ids: tuple[str, ...]
    falsifier_ids: tuple[str, ...]
    admission_gate_ids: tuple[str, ...]
    precision_goals: tuple[PrecisionGoal, ...]
    outcome_interpretations: tuple[OutcomeInterpretation, ...]
    authority_action: AuthorityAction
    fresh_evidence_required: bool
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling
    stop_conditions: tuple[str, ...]
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        validate_nonempty(self.question, field_name="question")
        for field_name, values in (
            ("hypothesis_ids", self.hypothesis_ids),
            ("measurement_quantity_ids", self.measurement_quantity_ids),
            ("falsifier_ids", self.falsifier_ids),
            ("admission_gate_ids", self.admission_gate_ids),
            ("stop_conditions", self.stop_conditions),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        require_sorted_unique_ids(
            self.precision_goals, attribute="goal_id", field_name="precision_goals"
        )
        require_sorted_unique_ids(
            self.outcome_interpretations,
            attribute="interpretation_id",
            field_name="outcome_interpretations",
        )
        interpretation_kinds = {
            interpretation.kind for interpretation in self.outcome_interpretations
        }
        if interpretation_kinds != set(OutcomeInterpretationKind):
            raise ValueError(
                "nomination must predeclare supporting, opposing and unresolved outcomes"
            )
        if self.assignment.action_quantity_ids != self.relation.action_quantity_ids:
            raise ValueError("nomination assignment differs from relational action chart")
        if not set(self.relation.receiver_quantity_ids).issubset(self.measurement_quantity_ids):
            raise ValueError("nomination measurements omit a relational receiver")
        if not self.fresh_evidence_required:
            raise ValueError("a prospective nomination must require fresh evidence")
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("nomination visibility cannot be lowered")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("a nomination remains outcome-visible and non-promotable")
        require_extensions(self.extensions)

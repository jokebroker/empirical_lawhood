"""Immutable contracts for bounded, outcome-visible automated discovery."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling, inherited_visibility
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import LifecycleStatus
from empirical_lawhood.kernel.systems import RelationalIdentity
from empirical_lawhood.planning.exploration import (
    AnalysisAttempt,
    AnalysisProposal,
    AnalysisSpec,
    AnomalyKind,
    ExplorationPlan,
    ExploratoryFinding,
    HypothesisSet,
    ProposalDisposition,
    ProposalSelection,
)


class DiagnosticMeasureKind(StrEnum):
    """Substrate-neutral diagnostic semantics exposed by scientific adapters."""

    SUPPORT_COVERAGE = "SUPPORT_COVERAGE"
    GATE_MARGIN = "GATE_MARGIN"
    RESIDUAL_MEMORY = "RESIDUAL_MEMORY"
    RANK_INSTABILITY = "RANK_INSTABILITY"
    COORDINATE_INSTABILITY = "COORDINATE_INSTABILITY"
    WRONG_ACTION_RETENTION = "WRONG_ACTION_RETENTION"
    MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"
    NUMERICAL_DISAGREEMENT = "NUMERICAL_DISAGREEMENT"
    INVARIANCE_DEVIATION = "INVARIANCE_DEVIATION"
    OBSERVER_PLANT_LOSS = "OBSERVER_PLANT_LOSS"
    EFFECTIVE_INDEPENDENT_UNITS = "EFFECTIVE_INDEPENDENT_UNITS"


class ThresholdDirection(StrEnum):
    ABOVE_MAXIMUM = "ABOVE_MAXIMUM"
    BELOW_MINIMUM = "BELOW_MINIMUM"
    ABSOLUTE_ABOVE_MAXIMUM = "ABSOLUTE_ABOVE_MAXIMUM"


@dataclass(frozen=True, slots=True)
class DiagnosticMeasure(CanonicalRecord):
    """One typed, native-unit diagnostic with its predeclared trigger."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/diagnostic-measure'

    measure_id: str
    kind: DiagnosticMeasureKind
    relation_id: str
    affected_quantity_ids: tuple[str, ...]
    value: Decimal
    threshold: Decimal
    threshold_direction: ThresholdDirection
    native_unit: str
    physical_independent_unit_count: int
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.measure_id, field_name="measure_id")
        validate_stable_id(self.relation_id, field_name="relation_id")
        require_sorted_unique_strings(
            self.affected_quantity_ids,
            field_name="affected_quantity_ids",
            allow_empty=False,
        )
        validate_decimal(self.value, field_name="value")
        validate_decimal(self.threshold, field_name="threshold")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.physical_independent_unit_count <= 0:
            raise ValueError("diagnostic measure requires physical independent units")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
            allow_empty=False,
        )

    @property
    def triggered(self) -> bool:
        if self.threshold_direction is ThresholdDirection.ABOVE_MAXIMUM:
            return self.value > self.threshold
        if self.threshold_direction is ThresholdDirection.BELOW_MINIMUM:
            return self.value < self.threshold
        return abs(self.value) > abs(self.threshold)

    @property
    def normalized_excess(self) -> Decimal:
        if not self.triggered:
            return Decimal(0)
        denominator = max(abs(self.threshold), Decimal("1e-12"))
        if self.threshold_direction is ThresholdDirection.BELOW_MINIMUM:
            return (self.threshold - self.value) / denominator
        if self.threshold_direction is ThresholdDirection.ABSOLUTE_ABOVE_MAXIMUM:
            return (abs(self.value) - abs(self.threshold)) / denominator
        return (self.value - self.threshold) / denominator


@dataclass(frozen=True, slots=True)
class ExplorationDiagnosticProjection(CanonicalRecord):
    """Materializable typed diagnostic payload before it is bound into a snapshot."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/exploration-diagnostic-projection'

    projection_id: str
    system_id: str
    relation: RelationalIdentity
    independent_unit_id: str
    grouping_clock_ids: tuple[str, ...]
    available_role_ids: tuple[str, ...]
    available_upstream_schema_ids: tuple[str, ...]
    upstream_objects: tuple[ObjectIdentity, ...]
    measures: tuple[DiagnosticMeasure, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("projection_id", self.projection_id),
            ("system_id", self.system_id),
            ("independent_unit_id", self.independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        for field_name, values in (
            ("grouping_clock_ids", self.grouping_clock_ids),
            ("available_role_ids", self.available_role_ids),
            ("available_upstream_schema_ids", self.available_upstream_schema_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        require_sorted_unique_ids(
            self.upstream_objects,
            attribute="object_id",
            field_name="upstream_objects",
        )
        if {identity.object_schema for identity in self.upstream_objects} != set(
            self.available_upstream_schema_ids
        ):
            raise ValueError("upstream schema availability requires exact object identities")
        require_sorted_unique_ids(self.measures, attribute="measure_id", field_name="measures")
        if any(measure.relation_id != self.relation.relation_id for measure in self.measures):
            raise ValueError("diagnostic projection mixes relational law contexts")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("automated discovery projection must remain outcome-visible")


@dataclass(frozen=True, slots=True)
class ExplorationEvidenceView(CanonicalRecord):
    """Typed diagnostic projection bound to exactly one immutable evidence snapshot."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/exploration-evidence-view'

    view_id: str
    snapshot: ObjectIdentity
    projection: ObjectIdentity
    system_id: str
    relation: RelationalIdentity
    independent_unit_id: str
    grouping_clock_ids: tuple[str, ...]
    available_projection_ids: tuple[str, ...]
    available_role_ids: tuple[str, ...]
    available_upstream_schema_ids: tuple[str, ...]
    upstream_objects: tuple[ObjectIdentity, ...]
    measures: tuple[DiagnosticMeasure, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("view_id", self.view_id),
            ("system_id", self.system_id),
            ("independent_unit_id", self.independent_unit_id),
        ):
            validate_stable_id(value, field_name=name)
        for field_name, values in (
            ("grouping_clock_ids", self.grouping_clock_ids),
            ("available_projection_ids", self.available_projection_ids),
            ("available_role_ids", self.available_role_ids),
            ("available_upstream_schema_ids", self.available_upstream_schema_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        require_sorted_unique_ids(
            self.upstream_objects,
            attribute="object_id",
            field_name="upstream_objects",
        )
        if {identity.object_schema for identity in self.upstream_objects} != set(
            self.available_upstream_schema_ids
        ):
            raise ValueError("evidence-view upstream schemas lack exact object identities")
        require_sorted_unique_ids(self.measures, attribute="measure_id", field_name="measures")
        if any(measure.relation_id != self.relation.relation_id for measure in self.measures):
            raise ValueError("diagnostic measure belongs to another relational law context")
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("exploration evidence view visibility cannot be lowered")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("automated discovery requires an outcome-visible evidence view")
        require_extensions(self.extensions)


class ExplorationIntent(StrEnum):
    METHOD_CALIBRATION = "METHOD_CALIBRATION"
    RETROSPECTIVE_DISCOVERY = "RETROSPECTIVE_DISCOVERY"
    INTERVENTION_BACKED_REUSE = "INTERVENTION_BACKED_REUSE"


@dataclass(frozen=True, slots=True)
class QuantityReportingObligation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/quantity-reporting-obligation'

    quantity_id: str
    native_unit: str
    normalized_reporting_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_nonempty(self.native_unit, field_name="native_unit")


@dataclass(frozen=True, slots=True)
class AnalysisObligations(CanonicalRecord):
    """Schema-bound extension completing the safe declarative analysis grammar."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/analysis-obligations'

    obligations_id: str
    analysis_id: str
    intent: ExplorationIntent
    competing_explanation_ids: tuple[str, ...]
    transform_keys: tuple[str, ...]
    estimator_keys: tuple[str, ...]
    decomposition_keys: tuple[str, ...]
    grouping_rule_keys: tuple[str, ...]
    folding_rule_keys: tuple[str, ...]
    causal_clock_ids: tuple[str, ...]
    application_clock_ids: tuple[str, ...]
    matched_null_keys: tuple[str, ...]
    positive_control_keys: tuple[str, ...]
    falsifier_keys: tuple[str, ...]
    uncertainty_rule_key: str
    influence_rule_key: str
    evaluability_rule_key: str
    multiplicity_rule_key: str
    sensitivity_keys: tuple[str, ...]
    reporting_keys: tuple[str, ...]
    registered_family_member_ids: tuple[str, ...]
    quantity_reporting: tuple[QuantityReportingObligation, ...]
    non_promotion_statement: str
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.obligations_id, field_name="obligations_id")
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        for field_name, values, allow_empty in (
            ("competing_explanation_ids", self.competing_explanation_ids, False),
            ("transform_keys", self.transform_keys, False),
            ("estimator_keys", self.estimator_keys, False),
            ("decomposition_keys", self.decomposition_keys, False),
            ("grouping_rule_keys", self.grouping_rule_keys, False),
            ("folding_rule_keys", self.folding_rule_keys, False),
            ("causal_clock_ids", self.causal_clock_ids, False),
            ("application_clock_ids", self.application_clock_ids, False),
            ("matched_null_keys", self.matched_null_keys, False),
            ("positive_control_keys", self.positive_control_keys, False),
            ("falsifier_keys", self.falsifier_keys, False),
            ("sensitivity_keys", self.sensitivity_keys, False),
            ("reporting_keys", self.reporting_keys, False),
            ("registered_family_member_ids", self.registered_family_member_ids, False),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=allow_empty)
        if len(self.competing_explanation_ids) < 2:
            raise ValueError("analysis must retain at least two competing explanations")
        for field_name, value in (
            ("uncertainty_rule_key", self.uncertainty_rule_key),
            ("influence_rule_key", self.influence_rule_key),
            ("evaluability_rule_key", self.evaluability_rule_key),
            ("multiplicity_rule_key", self.multiplicity_rule_key),
        ):
            validate_stable_id(value, field_name=field_name)
        require_sorted_unique_ids(
            self.quantity_reporting,
            attribute="quantity_id",
            field_name="quantity_reporting",
        )
        if not self.quantity_reporting:
            raise ValueError("analysis must retain native-unit reporting obligations")
        validate_nonempty(self.non_promotion_statement, field_name="non_promotion_statement")
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("analysis-obligations visibility cannot be lowered")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("exploratory analysis obligations must remain non-promotable")


@dataclass(frozen=True, slots=True)
class AnalysisTemplate(CanonicalRecord):
    """A registered question structure; never a substrate coordinate mapping."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/analysis-template'

    template_id: str
    template_version: str
    question: str
    estimand: str
    intent: ExplorationIntent
    anomaly_kinds: tuple[AnomalyKind, ...]
    required_role_ids: tuple[str, ...]
    required_upstream_schema_ids: tuple[str, ...]
    required_search_axis_ids: tuple[str, ...]
    registered_pipeline_key: str
    registered_pipeline_version: str
    transform_keys: tuple[str, ...]
    estimator_keys: tuple[str, ...]
    decomposition_keys: tuple[str, ...]
    matched_null_keys: tuple[str, ...]
    positive_control_keys: tuple[str, ...]
    falsifier_keys: tuple[str, ...]
    terminal_missing_role_code: str
    terminal_missing_upstream_code: str

    def __post_init__(self) -> None:
        for name, value in (
            ("template_id", self.template_id),
            ("registered_pipeline_key", self.registered_pipeline_key),
            ("terminal_missing_role_code", self.terminal_missing_role_code),
            ("terminal_missing_upstream_code", self.terminal_missing_upstream_code),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.template_version)
        validate_semantic_version(self.registered_pipeline_version)
        validate_nonempty(self.question, field_name="question")
        validate_nonempty(self.estimand, field_name="estimand")
        if tuple(sorted(set(self.anomaly_kinds))) != self.anomaly_kinds:
            raise ValueError("template anomaly kinds must be sorted and unique")
        if not self.anomaly_kinds:
            raise ValueError("analysis template requires an anomaly kind")
        for field_name, values in (
            ("required_role_ids", self.required_role_ids),
            ("required_upstream_schema_ids", self.required_upstream_schema_ids),
            ("required_search_axis_ids", self.required_search_axis_ids),
            ("transform_keys", self.transform_keys),
            ("estimator_keys", self.estimator_keys),
            ("decomposition_keys", self.decomposition_keys),
            ("matched_null_keys", self.matched_null_keys),
            ("positive_control_keys", self.positive_control_keys),
            ("falsifier_keys", self.falsifier_keys),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)


@dataclass(frozen=True, slots=True)
class TemplateInstantiation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/template-instantiation'

    instantiation_id: str
    template: ObjectIdentity
    proposal: AnalysisProposal
    obligations: AnalysisObligations
    ready: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.instantiation_id, field_name="instantiation_id")
        if self.proposal.analysis.analysis_id != self.obligations.analysis_id:
            raise ValueError("template obligations bind another analysis")
        extensions = {
            extension.namespace: extension for extension in self.proposal.analysis.extensions
        }
        binding = extensions.get("automated-discovery-obligations")
        if (
            binding is None
            or binding.schema != self.obligations.SCHEMA
            or binding.payload_sha256 != self.obligations.fingerprint()
        ):
            raise ValueError("analysis does not bind its complete declarative obligations")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.ready and self.reason_codes:
            raise ValueError("ready template instantiation cannot retain blocking reasons")
        if not self.ready and not self.reason_codes:
            raise ValueError("blocked template instantiation requires reason codes")


@dataclass(frozen=True, slots=True)
class PortfolioScore(CanonicalRecord):
    """Declared multi-objective score; fields are never collapsed to a reward."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/portfolio-score'

    proposal_id: str
    falsification_value: Decimal
    unresolved_coverage: Decimal
    novelty: Decimal
    independent_unit_adequacy: Decimal
    prospective_testability: Decimal
    redundancy_risk: Decimal
    search_slicing_risk: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.proposal_id, field_name="proposal_id")
        for name, value in (
            ("falsification_value", self.falsification_value),
            ("unresolved_coverage", self.unresolved_coverage),
            ("novelty", self.novelty),
            ("independent_unit_adequacy", self.independent_unit_adequacy),
            ("prospective_testability", self.prospective_testability),
            ("redundancy_risk", self.redundancy_risk),
            ("search_slicing_risk", self.search_slicing_risk),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value > 1:
                raise ValueError(f"{name} must be normalized to [0, 1]")


@dataclass(frozen=True, slots=True)
class PortfolioCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/portfolio-candidate'

    instantiation: TemplateInstantiation
    score: PortfolioScore
    redundancy_group_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.redundancy_group_id, field_name="redundancy_group_id")
        if self.instantiation.proposal.proposal_id != self.score.proposal_id:
            raise ValueError("portfolio score binds another proposal")


@dataclass(frozen=True, slots=True)
class PortfolioPolicy(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/portfolio-policy'

    policy_id: str
    budget: ResourceBudget
    maximum_selected_proposals: int
    maximum_per_redundancy_group: int
    maximum_family_members_per_analysis: int
    maximum_total_family_members: int
    minimum_falsification_value: Decimal
    minimum_independent_unit_adequacy: Decimal
    priority_axis_ids: tuple[str, ...]
    stop_when_no_discriminating_analysis: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.policy_id, field_name="policy_id")
        if (
            self.maximum_selected_proposals <= 0
            or self.maximum_per_redundancy_group <= 0
            or self.maximum_family_members_per_analysis <= 0
            or self.maximum_total_family_members <= 0
        ):
            raise ValueError("portfolio selection limits must be positive")
        if self.maximum_total_family_members < self.maximum_family_members_per_analysis:
            raise ValueError("total family budget cannot be smaller than per-analysis budget")
        for name, value in (
            ("minimum_falsification_value", self.minimum_falsification_value),
            ("minimum_independent_unit_adequacy", self.minimum_independent_unit_adequacy),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value > 1:
                raise ValueError(f"{name} must be normalized to [0, 1]")
        allowed_axes = {
            "falsification-value",
            "independent-unit-adequacy",
            "novelty",
            "prospective-testability",
            "redundancy-risk",
            "search-slicing-risk",
            "unresolved-coverage",
        }
        require_sorted_unique_strings(
            tuple(sorted(self.priority_axis_ids)),
            field_name="priority_axis_ids",
            allow_empty=False,
        )
        if len(set(self.priority_axis_ids)) != len(self.priority_axis_ids):
            raise ValueError("portfolio priority axes must be unique")
        if not set(self.priority_axis_ids).issubset(allowed_axes):
            raise ValueError("portfolio policy names an unknown objective axis")


class PortfolioDecisionKind(StrEnum):
    EXECUTE = "EXECUTE"
    STOP_NO_ANALYSIS = "STOP_NO_ANALYSIS"


@dataclass(frozen=True, slots=True)
class PortfolioDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/portfolio-decision'

    decision_id: str
    kind: PortfolioDecisionKind
    snapshot: ObjectIdentity
    proposal_ids: tuple[str, ...]
    selections: tuple[ProposalSelection, ...]
    pareto_frontier_proposal_ids: tuple[str, ...]
    selected_plan: ObjectIdentity | None
    stop_reason_codes: tuple[str, ...]
    planner_key: str
    planner_version: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        validate_stable_id(self.planner_key, field_name="planner_key")
        validate_semantic_version(self.planner_version)
        require_sorted_unique_strings(self.proposal_ids, field_name="proposal_ids")
        require_sorted_unique_ids(self.selections, attribute="proposal_id", field_name="selections")
        if tuple(selection.proposal_id for selection in self.selections) != self.proposal_ids:
            raise ValueError("portfolio decision must disposition every proposal exactly once")
        require_sorted_unique_strings(
            self.pareto_frontier_proposal_ids,
            field_name="pareto_frontier_proposal_ids",
        )
        if not set(self.pareto_frontier_proposal_ids).issubset(self.proposal_ids):
            raise ValueError("Pareto frontier references an unknown proposal")
        require_sorted_unique_strings(self.stop_reason_codes, field_name="stop_reason_codes")
        if self.kind is PortfolioDecisionKind.EXECUTE:
            if self.selected_plan is None or self.stop_reason_codes:
                raise ValueError("execute decision requires a plan and no stop reasons")
            if not any(
                selection.disposition is ProposalDisposition.SELECTED
                for selection in self.selections
            ):
                raise ValueError("execute decision requires a selected proposal")
        elif self.selected_plan is not None or not self.stop_reason_codes:
            raise ValueError("stop decision requires reasons and cannot carry a plan")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("portfolio decision cannot promote outcome-visible evidence")


class SkepticCheckKind(StrEnum):
    MECHANICAL_CONSTRUCTION = "MECHANICAL_CONSTRUCTION"
    ALTERNATIVE_EXPLANATIONS = "ALTERNATIVE_EXPLANATIONS"
    MATCHED_NULLS = "MATCHED_NULLS"
    LEAVE_ONE_UNIT_INFLUENCE = "LEAVE_ONE_UNIT_INFLUENCE"
    FAMILY_COMPLETENESS = "FAMILY_COMPLETENESS"
    SCALING_ARTIFACT = "SCALING_ARTIFACT"
    AGGREGATION_ARTIFACT = "AGGREGATION_ARTIFACT"
    LEAKAGE = "LEAKAGE"
    SELECTION_ARTIFACT = "SELECTION_ARTIFACT"


@dataclass(frozen=True, slots=True)
class SkepticCheck(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/skeptic-check'

    check_id: str
    kind: SkepticCheckKind
    passed: bool
    reason_codes: tuple[str, ...]
    observation: str

    def __post_init__(self) -> None:
        validate_stable_id(self.check_id, field_name="check_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        validate_nonempty(self.observation, field_name="observation")
        if self.passed and self.reason_codes:
            raise ValueError("passing skeptic check cannot retain failure reasons")
        if not self.passed and not self.reason_codes:
            raise ValueError("failed skeptic check requires reason codes")


@dataclass(frozen=True, slots=True)
class SkepticReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/skeptic-report'

    report_id: str
    plan: ObjectIdentity
    finding_ids: tuple[str, ...]
    checks: tuple[SkepticCheck, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        require_sorted_unique_strings(self.finding_ids, field_name="finding_ids", allow_empty=False)
        require_sorted_unique_ids(self.checks, attribute="check_id", field_name="checks")
        if {check.kind for check in self.checks} != set(SkepticCheckKind):
            raise ValueError("skeptic report must execute the complete check family")
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("skeptic-report visibility cannot be lowered")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("skeptic report cannot promote outcome-visible findings")

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)


@dataclass(frozen=True, slots=True)
class HypothesisSynthesis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/hypothesis-synthesis'

    synthesis_id: str
    hypothesis_set: HypothesisSet
    skeptic_report: ObjectIdentity
    influence_diagnostic_ids: tuple[str, ...]
    limitation_codes: tuple[str, ...]
    prospective_evidence_need_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.synthesis_id, field_name="synthesis_id")
        for field_name, values in (
            ("influence_diagnostic_ids", self.influence_diagnostic_ids),
            ("limitation_codes", self.limitation_codes),
            ("prospective_evidence_need_ids", self.prospective_evidence_need_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        if self.hypothesis_set.visibility_ceiling.is_promotable:
            raise ValueError("hypothesis synthesis cannot promote exploratory evidence")


@dataclass(frozen=True, slots=True)
class ExplorationLifecycleLedger(CanonicalRecord):
    """Complete scientific disposition ledger for one immutable wave."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/exploration-lifecycle-ledger'

    ledger_id: str
    wave_index: int
    parent_wave_ledger_id: str | None
    snapshot: ObjectIdentity
    plan: ObjectIdentity | None
    decision: ObjectIdentity
    selections: tuple[ProposalSelection, ...]
    attempts: tuple[AnalysisAttempt, ...]
    findings: tuple[ExploratoryFinding, ...]
    hypothesis_syntheses: tuple[HypothesisSynthesis, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    frozen: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.ledger_id, field_name="ledger_id")
        if self.wave_index <= 0:
            raise ValueError("exploration wave index must be positive")
        if self.parent_wave_ledger_id is not None:
            validate_stable_id(self.parent_wave_ledger_id, field_name="parent_wave_ledger_id")
        if (self.wave_index == 1) is not (self.parent_wave_ledger_id is None):
            raise ValueError("only the first wave may omit a sequential parent")
        require_sorted_unique_ids(self.selections, attribute="proposal_id", field_name="selections")
        require_sorted_unique_ids(self.attempts, attribute="attempt_id", field_name="attempts")
        require_sorted_unique_ids(self.findings, attribute="finding_id", field_name="findings")
        require_sorted_unique_ids(
            self.hypothesis_syntheses,
            attribute="synthesis_id",
            field_name="hypothesis_syntheses",
        )
        selected = {
            selection.proposal_id
            for selection in self.selections
            if selection.disposition is ProposalDisposition.SELECTED
        }
        if any(attempt.proposal_id not in selected for attempt in self.attempts):
            raise ValueError("only selected proposals may produce execution attempts")
        if self.plan is None and (selected or self.attempts or self.findings):
            raise ValueError("a stop/no-analysis ledger cannot retain executed work")
        if self.plan is not None and not selected:
            raise ValueError("an executed wave requires selected proposals")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("exploration lifecycle cannot promote evidence")
        if not self.frozen:
            raise ValueError("exploration lifecycle ledger must be immutable")


class DiscoveryLane(StrEnum):
    EXPLORATORY = "EXPLORATORY"
    PROSPECTIVE = "PROSPECTIVE"


class DiscoveryNodeKind(StrEnum):
    EVIDENCE_SNAPSHOT = "EVIDENCE_SNAPSHOT"
    ANOMALY_SIGNAL = "ANOMALY_SIGNAL"
    ANALYSIS_PROPOSAL = "ANALYSIS_PROPOSAL"
    EXPLORATION_PLAN = "EXPLORATION_PLAN"
    EXPLORATORY_FINDING = "EXPLORATORY_FINDING"
    HYPOTHESIS_SET = "HYPOTHESIS_SET"
    PROSPECTIVE_NOMINATION = "PROSPECTIVE_NOMINATION"
    EXPERIMENT_PROPOSAL = "EXPERIMENT_PROPOSAL"
    RUN = "RUN"
    EVIDENCE_RESULT = "EVIDENCE_RESULT"


_EXPLORATORY_NODE_KINDS = {
    DiscoveryNodeKind.EVIDENCE_SNAPSHOT,
    DiscoveryNodeKind.ANOMALY_SIGNAL,
    DiscoveryNodeKind.ANALYSIS_PROPOSAL,
    DiscoveryNodeKind.EXPLORATION_PLAN,
    DiscoveryNodeKind.EXPLORATORY_FINDING,
    DiscoveryNodeKind.HYPOTHESIS_SET,
    DiscoveryNodeKind.PROSPECTIVE_NOMINATION,
}


@dataclass(frozen=True, slots=True)
class DiscoveryNode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/discovery-node'

    node_id: str
    kind: DiscoveryNodeKind
    lane: DiscoveryLane
    object_identity: ObjectIdentity
    lifecycle_status: LifecycleStatus

    def __post_init__(self) -> None:
        validate_stable_id(self.node_id, field_name="node_id")
        expected = (
            DiscoveryLane.EXPLORATORY
            if self.kind in _EXPLORATORY_NODE_KINDS
            else DiscoveryLane.PROSPECTIVE
        )
        if self.lane is not expected:
            raise ValueError("discovery node is assigned to the wrong campaign lane")


class DiscoveryEdgeKind(StrEnum):
    MOTIVATED_BY = "MOTIVATED_BY"
    SELECTED = "SELECTED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"
    SUPERSEDED = "SUPERSEDED"
    EXECUTED = "EXECUTED"
    NOMINATES = "NOMINATES"
    ADJUDICATED_BY = "ADJUDICATED_BY"
    SEQUENTIAL_PARENT = "SEQUENTIAL_PARENT"


@dataclass(frozen=True, slots=True)
class DiscoveryEdge(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/discovery-edge'

    edge_id: str
    kind: DiscoveryEdgeKind
    source_node_id: str
    target_node_id: str
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("edge_id", self.edge_id),
            ("source_node_id", self.source_node_id),
            ("target_node_id", self.target_node_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.source_node_id == self.target_node_id:
            raise ValueError("discovery edge cannot be a self-loop")


@dataclass(frozen=True, slots=True)
class CampaignGraph(CanonicalRecord):
    """Explicit typed lineage around immutable run and exploration DAGs."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/campaign-graph'

    graph_id: str
    campaign: ObjectIdentity
    nodes: tuple[DiscoveryNode, ...]
    edges: tuple[DiscoveryEdge, ...]
    root_node_ids: tuple[str, ...]
    active_node_ids: tuple[str, ...]
    predecessor_graph_ids: tuple[str, ...] = ()
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.graph_id, field_name="graph_id")
        require_sorted_unique_ids(self.nodes, attribute="node_id", field_name="nodes")
        require_sorted_unique_ids(self.edges, attribute="edge_id", field_name="edges")
        require_sorted_unique_strings(
            self.root_node_ids, field_name="root_node_ids", allow_empty=False
        )
        require_sorted_unique_strings(self.active_node_ids, field_name="active_node_ids")
        require_sorted_unique_strings(
            self.predecessor_graph_ids,
            field_name="predecessor_graph_ids",
        )
        self._validate_graph()
        require_extensions(self.extensions)

    def _validate_graph(self) -> None:
        nodes = {node.node_id: node for node in self.nodes}
        if not set(self.root_node_ids).issubset(nodes):
            raise ValueError("campaign graph roots reference unknown nodes")
        if not set(self.active_node_ids).issubset(nodes):
            raise ValueError("campaign graph active set references unknown nodes")
        parents: dict[str, list[str]] = {node_id: [] for node_id in nodes}
        for edge in self.edges:
            if edge.source_node_id not in nodes or edge.target_node_id not in nodes:
                raise ValueError("campaign graph edge references an unknown node")
            parents[edge.target_node_id].append(edge.source_node_id)
            self._validate_edge(nodes[edge.source_node_id], nodes[edge.target_node_id], edge)
        observed_roots = tuple(sorted(node_id for node_id, values in parents.items() if not values))
        if observed_roots != self.root_node_ids:
            raise ValueError("campaign graph declared roots differ from edge roots")
        self._assert_acyclic(parents)

    @staticmethod
    def _validate_edge(
        source: DiscoveryNode,
        target: DiscoveryNode,
        edge: DiscoveryEdge,
    ) -> None:
        if source.lane is DiscoveryLane.EXPLORATORY and target.lane is DiscoveryLane.PROSPECTIVE:
            if not (
                source.kind is DiscoveryNodeKind.PROSPECTIVE_NOMINATION
                and target.kind is DiscoveryNodeKind.EXPERIMENT_PROPOSAL
                and edge.kind is DiscoveryEdgeKind.NOMINATES
            ):
                raise ValueError("only a nomination edge may bridge to the prospective lane")
        if source.lane is DiscoveryLane.PROSPECTIVE and target.lane is DiscoveryLane.EXPLORATORY:
            if not (
                source.kind is DiscoveryNodeKind.EVIDENCE_RESULT
                and target.kind is DiscoveryNodeKind.EVIDENCE_SNAPSHOT
                and edge.kind is DiscoveryEdgeKind.ADJUDICATED_BY
            ):
                raise ValueError("prospective evidence may return only through adjudication")
        disposition_edges = {
            DiscoveryEdgeKind.SELECTED,
            DiscoveryEdgeKind.REJECTED,
            DiscoveryEdgeKind.BLOCKED,
            DiscoveryEdgeKind.SUPERSEDED,
        }
        if edge.kind in disposition_edges and not (
            source.kind is DiscoveryNodeKind.ANALYSIS_PROPOSAL
            and target.kind is DiscoveryNodeKind.EXPLORATION_PLAN
        ):
            raise ValueError("proposal disposition edge has invalid endpoints")

    @staticmethod
    def _assert_acyclic(parents: dict[str, list[str]]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError("campaign discovery graph contains a cycle")
            if node_id in visited:
                return
            visiting.add(node_id)
            for parent_id in parents[node_id]:
                visit(parent_id)
            visiting.remove(node_id)
            visited.add(node_id)

        for node_id in parents:
            visit(node_id)


def obligations_extension(obligations: AnalysisObligations) -> ExtensionBinding:
    "Create the exact companion binding used by the frozen AnalysisSpec."

    return ExtensionBinding(
        namespace="automated-discovery-obligations",
        schema=obligations.SCHEMA,
        payload_sha256=obligations.fingerprint(),
    )


def plan_identity(plan: ExplorationPlan) -> ObjectIdentity:
    return ObjectIdentity.from_record(plan.plan_id, plan)


def analysis_identity(analysis: AnalysisSpec) -> ObjectIdentity:
    return ObjectIdentity.from_record(analysis.analysis_id, analysis)

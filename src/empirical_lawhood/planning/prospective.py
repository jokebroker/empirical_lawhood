"""Contracts for advisory translation from curiosity to fresh-evidence designs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget, SourceAccessClass
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from empirical_lawhood.kernel.experiments import AssignmentKind, ControlSpec, PrecisionGoal
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.quantities import QuantitySpec
from empirical_lawhood.kernel.references import QuantityBound
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.kernel.time import InformationCutoff
from empirical_lawhood.planning.exploration import ProspectiveNomination


class ProspectiveDesignKind(StrEnum):
    SIMPLE_FACTORIAL = "SIMPLE_FACTORIAL"
    COVERAGE_EXPANSION = "COVERAGE_EXPANSION"
    INFORMATION_RANK = "INFORMATION_RANK"
    MODEL_DISCRIMINATION = "MODEL_DISCRIMINATION"


class ProspectiveControlRole(StrEnum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


@dataclass(frozen=True, slots=True)
class NativeMeasurementRequirement(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/native-measurement-requirement'

    requirement_id: str
    quantity_id: str
    native_unit: str
    clock_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("requirement_id", self.requirement_id),
            ("quantity_id", self.quantity_id),
            ("clock_id", self.clock_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")

    def validate_quantity(self, quantity: QuantitySpec) -> None:
        if (
            quantity.quantity_id != self.quantity_id
            or quantity.native_unit != self.native_unit
            or quantity.clock_id != self.clock_id
        ):
            raise ValueError("prospective measurement differs from the system native quantity")


@dataclass(frozen=True, slots=True)
class ProspectiveControlRequirement(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-control-requirement'

    requirement_id: str
    role: ProspectiveControlRole
    control: ControlSpec

    def __post_init__(self) -> None:
        validate_stable_id(self.requirement_id, field_name="requirement_id")


@dataclass(frozen=True, slots=True)
class ProspectiveObjectiveVector(CanonicalRecord):
    """Declared acquisition objectives; never collapsed into scalar utility."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-objective-vector'

    objective_id: str
    design_kind: ProspectiveDesignKind
    falsification_value: Decimal
    hypothesis_discrimination: Decimal
    support_or_rank_gain: Decimal
    acquisition_cost: Decimal
    safety_or_authority_risk: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.objective_id, field_name="objective_id")
        for name, value in (
            ("falsification_value", self.falsification_value),
            ("hypothesis_discrimination", self.hypothesis_discrimination),
            ("support_or_rank_gain", self.support_or_rank_gain),
            ("acquisition_cost", self.acquisition_cost),
            ("safety_or_authority_risk", self.safety_or_authority_risk),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value > 1:
                raise ValueError(f"{name} must be normalized to [0, 1]")


@dataclass(frozen=True, slots=True)
class ProspectiveEvidenceContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-evidence-contract'

    contract_id: str
    development_unit_ids: tuple[str, ...]
    fresh_independent_unit_ids: tuple[str, ...]
    evaluation_cohort_id: str
    evaluation_manifest_sha256: str
    sealed_outcome_artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        validate_stable_id(self.evaluation_cohort_id, field_name="evaluation_cohort_id")
        validate_sha256(self.evaluation_manifest_sha256)
        for field_name, values in (
            ("development_unit_ids", self.development_unit_ids),
            ("fresh_independent_unit_ids", self.fresh_independent_unit_ids),
            ("sealed_outcome_artifact_ids", self.sealed_outcome_artifact_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        if set(self.development_unit_ids).intersection(self.fresh_independent_unit_ids):
            raise ValueError("fresh independent units overlap development units")


@dataclass(frozen=True, slots=True)
class ProspectiveDesignContext(CanonicalRecord):
    """Complete caller-supplied design authority and fresh-evidence envelope."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-design-context'

    context_id: str
    source_snapshot: ObjectIdentity
    system: ObjectIdentity
    design_kinds: tuple[ProspectiveDesignKind, ...]
    evidence_contract: ProspectiveEvidenceContract | None
    assignment_kind: AssignmentKind
    assignment_mechanism: str
    assignment_support_restriction_ids: tuple[str, ...]
    randomization_unit_id: str | None
    measurements: tuple[NativeMeasurementRequirement, ...]
    action_bounds: tuple[QuantityBound, ...]
    denominator_cell_ids: tuple[str, ...]
    controls: tuple[ProspectiveControlRequirement, ...]
    admission_gate_ids: tuple[str, ...]
    precision_goals: tuple[PrecisionGoal, ...]
    objectives: tuple[ProspectiveObjectiveVector, ...]
    alternative_design_ids: tuple[str, ...]
    safety_constraint_ids: tuple[str, ...]
    budget: ResourceBudget
    information_cutoff: InformationCutoff
    authority_action: AuthorityAction
    source_access: SourceAccessClass
    claim_proposition: str
    claim_estimand: str
    claim_promotion_rule: str
    claim_assumption_ids: tuple[str, ...]
    requested_rung: EvidenceRung
    evidence_ceiling: EvidenceCeiling
    no_acquisition_reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.context_id, field_name="context_id")
        self._validate_design_family()
        self._validate_assignment()
        self._validate_collections()
        self._validate_claim()
        self._validate_acquisition()
        self._validate_evidence_scope()

    def _validate_design_family(self) -> None:
        if tuple(sorted(set(self.design_kinds))) != self.design_kinds or not self.design_kinds:
            raise ValueError("prospective design kinds must be sorted, unique and nonempty")
        if {objective.design_kind for objective in self.objectives} != set(self.design_kinds):
            raise ValueError("prospective objectives must cover every candidate design kind")

    def _validate_assignment(self) -> None:
        validate_nonempty(self.assignment_mechanism, field_name="assignment_mechanism")
        require_sorted_unique_strings(
            self.assignment_support_restriction_ids,
            field_name="assignment_support_restriction_ids",
        )
        if self.assignment_kind is AssignmentKind.RANDOMIZED_INTERVENTION:
            if self.randomization_unit_id is None:
                raise ValueError("randomized prospective design requires a randomization unit")
        elif self.randomization_unit_id is not None:
            raise ValueError("only randomized prospective design may name randomization units")
        if self.randomization_unit_id is not None:
            validate_stable_id(self.randomization_unit_id, field_name="randomization_unit_id")

    def _validate_collections(self) -> None:
        require_sorted_unique_ids(
            self.measurements, attribute="requirement_id", field_name="measurements"
        )
        require_sorted_unique_ids(
            self.action_bounds, attribute="bound_id", field_name="action_bounds"
        )
        require_sorted_unique_ids(self.controls, attribute="requirement_id", field_name="controls")
        require_sorted_unique_ids(
            self.precision_goals, attribute="goal_id", field_name="precision_goals"
        )
        require_sorted_unique_ids(
            self.objectives, attribute="objective_id", field_name="objectives"
        )
        for field_name, values in (
            ("denominator_cell_ids", self.denominator_cell_ids),
            ("admission_gate_ids", self.admission_gate_ids),
            ("alternative_design_ids", self.alternative_design_ids),
            ("safety_constraint_ids", self.safety_constraint_ids),
            ("claim_assumption_ids", self.claim_assumption_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        require_sorted_unique_strings(
            self.no_acquisition_reason_codes,
            field_name="no_acquisition_reason_codes",
        )

    def _validate_claim(self) -> None:
        for name, value in (
            ("claim_proposition", self.claim_proposition),
            ("claim_estimand", self.claim_estimand),
            ("claim_promotion_rule", self.claim_promotion_rule),
        ):
            validate_nonempty(value, field_name=name)

    def _validate_acquisition(self) -> None:
        if self.evidence_contract is None:
            if not self.no_acquisition_reason_codes:
                raise ValueError("missing fresh evidence requires no-acquisition reasons")
        else:
            if self.no_acquisition_reason_codes:
                raise ValueError("acquirable context cannot retain no-acquisition reasons")
            roles = {control.role for control in self.controls}
            if roles != set(ProspectiveControlRole):
                raise ValueError("prospective design requires positive and negative controls")
            if not self.measurements or not self.action_bounds or not self.precision_goals:
                raise ValueError("prospective acquisition requires measurements, bounds and power")

    def _validate_evidence_scope(self) -> None:
        if not self.evidence_ceiling.allows(self.requested_rung):
            raise ValueError("requested fresh evidence rung exceeds its declared ceiling")
        if self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED:
            raise ValueError("nomination context must disclose outcome-visible provenance")
        if self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("nomination context must remain outcome-visible")


@dataclass(frozen=True, slots=True)
class NominationObligations(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/nomination-obligations'

    obligations_id: str
    nomination_id: str
    source_finding_ids: tuple[str, ...]
    design_kind: ProspectiveDesignKind
    evidence_contract: ProspectiveEvidenceContract
    measurements: tuple[NativeMeasurementRequirement, ...]
    action_bounds: tuple[QuantityBound, ...]
    denominator_cell_ids: tuple[str, ...]
    controls: tuple[ProspectiveControlRequirement, ...]
    information_cutoff: InformationCutoff
    objective: ProspectiveObjectiveVector
    alternative_design_ids: tuple[str, ...]
    safety_constraint_ids: tuple[str, ...]
    budget: ResourceBudget
    authority_action: AuthorityAction
    source_access: SourceAccessClass
    claim_proposition: str
    claim_estimand: str
    claim_promotion_rule: str
    claim_assumption_ids: tuple[str, ...]
    requested_rung: EvidenceRung
    evidence_ceiling: EvidenceCeiling
    non_promotion_statement: str
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.obligations_id, field_name="obligations_id")
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        require_sorted_unique_strings(
            self.source_finding_ids, field_name="source_finding_ids", allow_empty=False
        )
        require_sorted_unique_ids(
            self.measurements, attribute="requirement_id", field_name="measurements"
        )
        require_sorted_unique_ids(
            self.action_bounds, attribute="bound_id", field_name="action_bounds"
        )
        require_sorted_unique_strings(
            self.denominator_cell_ids,
            field_name="denominator_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.controls, attribute="requirement_id", field_name="controls")
        for field_name, values in (
            ("alternative_design_ids", self.alternative_design_ids),
            ("safety_constraint_ids", self.safety_constraint_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        if self.objective.design_kind is not self.design_kind:
            raise ValueError("nomination objective belongs to another design family")
        for field_name, value in (
            ("claim_proposition", self.claim_proposition),
            ("claim_estimand", self.claim_estimand),
            ("claim_promotion_rule", self.claim_promotion_rule),
        ):
            validate_nonempty(value, field_name=field_name)
        require_sorted_unique_strings(
            self.claim_assumption_ids,
            field_name="claim_assumption_ids",
            allow_empty=False,
        )
        if not self.evidence_ceiling.allows(self.requested_rung):
            raise ValueError("nomination requested rung exceeds its fresh evidence ceiling")
        validate_nonempty(self.non_promotion_statement, field_name="non_promotion_statement")
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("nomination obligations visibility cannot be lowered")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("nomination obligations remain non-promotable")


class NominationDecisionKind(StrEnum):
    NOMINATE = "NOMINATE"
    STOP_NO_ACQUISITION = "STOP_NO_ACQUISITION"


class NominationDisposition(StrEnum):
    SELECTED = "SELECTED"
    REJECTED = "REJECTED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True, slots=True)
class DesignNominationSelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/design-nomination-selection'

    design_kind: ProspectiveDesignKind
    disposition: NominationDisposition
    nomination_id: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )
        if self.disposition is NominationDisposition.SELECTED:
            if self.nomination_id is None:
                raise ValueError("selected design must name its nomination")
        elif self.nomination_id is not None:
            raise ValueError("only selected designs may name a nomination")
        if self.nomination_id is not None:
            validate_stable_id(self.nomination_id, field_name="nomination_id")

    @property
    def selection_id(self) -> str:
        return self.design_kind.value.lower().replace("_", "-")


@dataclass(frozen=True, slots=True)
class ProspectiveNominationDecision(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-nomination-decision'

    decision_id: str
    source_hypothesis_set: ObjectIdentity
    kind: NominationDecisionKind
    selections: tuple[DesignNominationSelection, ...]
    nominations: tuple[ProspectiveNomination, ...]
    stop_reason_codes: tuple[str, ...]
    nominator_key: str
    nominator_version: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        validate_stable_id(self.nominator_key, field_name="nominator_key")
        require_sorted_unique_ids(
            self.selections, attribute="selection_id", field_name="selections"
        )
        require_sorted_unique_ids(
            self.nominations, attribute="nomination_id", field_name="nominations"
        )
        require_sorted_unique_strings(self.stop_reason_codes, field_name="stop_reason_codes")
        selected_ids = {
            selection.nomination_id
            for selection in self.selections
            if selection.disposition is NominationDisposition.SELECTED
        }
        if selected_ids != {nomination.nomination_id for nomination in self.nominations}:
            raise ValueError("nomination decision selected identities differ from its nominations")
        if self.kind is NominationDecisionKind.NOMINATE:
            if not self.nominations or self.stop_reason_codes:
                raise ValueError("nominate decision requires nominations and no stop reasons")
        elif self.nominations or not self.stop_reason_codes:
            raise ValueError("no-acquisition decision requires reasons and no nominations")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("nomination decision remains outcome-visible")


@dataclass(frozen=True, slots=True)
class ExperimentDesignAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/experiment-design-audit'

    audit_id: str
    proposal_id: str
    nomination: ObjectIdentity
    design_kind: ProspectiveDesignKind
    objective: ProspectiveObjectiveVector
    alternative_design_ids: tuple[str, ...]
    control_requirement_ids: tuple[str, ...]
    safety_constraint_ids: tuple[str, ...]
    expected_falsification_value: Decimal
    acquisition_value: str
    stop_or_fallback_rule: str
    decision_cutoff: InformationCutoff
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        validate_stable_id(self.proposal_id, field_name="proposal_id")
        for field_name, values in (
            ("alternative_design_ids", self.alternative_design_ids),
            ("control_requirement_ids", self.control_requirement_ids),
            ("safety_constraint_ids", self.safety_constraint_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        validate_decimal(
            self.expected_falsification_value,
            field_name="expected_falsification_value",
            minimum=Decimal(0),
        )
        if self.expected_falsification_value > 1:
            raise ValueError("expected falsification value must be normalized")
        validate_nonempty(self.acquisition_value, field_name="acquisition_value")
        validate_nonempty(self.stop_or_fallback_rule, field_name="stop_or_fallback_rule")
        if self.objective.design_kind is not self.design_kind:
            raise ValueError("experiment audit objective belongs to another design family")
        if self.visibility_ceiling.is_promotable:
            raise ValueError("experiment design audit remains outcome-visible")


@dataclass(frozen=True, slots=True)
class ProspectiveEvidenceResult(CanonicalRecord):
    """Categorical evaluator output over separately sealed fresh outcomes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-evidence-result'

    result_id: str
    experiment: ObjectIdentity
    run_plan: ObjectIdentity
    claim_ids: tuple[str, ...]
    fresh_independent_unit_ids: tuple[str, ...]
    status: ScientificStatus
    awarded_rung: EvidenceRung | None
    evidence_ceiling: EvidenceCeiling
    interpreted_outcome_id: str
    favored_hypothesis_ids: tuple[str, ...]
    terminal_reason_codes: tuple[str, ...]
    evidence_artifact_ids: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        validate_stable_id(self.interpreted_outcome_id, field_name="interpreted_outcome_id")
        for field_name, values in (
            ("claim_ids", self.claim_ids),
            ("fresh_independent_unit_ids", self.fresh_independent_unit_ids),
            ("terminal_reason_codes", self.terminal_reason_codes),
            ("evidence_artifact_ids", self.evidence_artifact_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        require_sorted_unique_strings(
            self.favored_hypothesis_ids,
            field_name="favored_hypothesis_ids",
        )
        if self.awarded_rung is not None and not self.evidence_ceiling.allows(self.awarded_rung):
            raise ValueError("fresh result rung exceeds its experiment ceiling")
        if self.status is ScientificStatus.SUPPORTED and self.awarded_rung is None:
            raise ValueError("supported fresh result must name its earned rung")
        if self.status is not ScientificStatus.SUPPORTED and self.awarded_rung is not None:
            raise ValueError("non-supported fresh result cannot award an evidence rung")
        if (
            self.status
            in {
                ScientificStatus.MIXED,
                ScientificStatus.PARTIAL,
                ScientificStatus.UNEVALUABLE,
                ScientificStatus.NOT_TESTED,
            }
            and self.favored_hypothesis_ids
        ):
            raise ValueError("unresolved fresh result cannot favor a hypothesis")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("fresh result must be emitted by the sealed evaluator boundary")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("fresh result must retain prospective visibility")


class HypothesisAdjudicationDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    OPPOSED = "OPPOSED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class HypothesisAdjudication(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/hypothesis-adjudication'

    adjudication_id: str
    hypothesis_id: str
    disposition: HypothesisAdjudicationDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        validate_stable_id(self.hypothesis_id, field_name="hypothesis_id")
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class ProspectiveAdjudication(CanonicalRecord):
    """Outcome-visible interpretation that cannot rewrite its source hypotheses."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-adjudication'

    adjudication_id: str
    source_hypothesis_set: ObjectIdentity
    nomination: ObjectIdentity
    proposal: ObjectIdentity
    fresh_result: ObjectIdentity
    hypothesis_adjudications: tuple[HypothesisAdjudication, ...]
    parent_fingerprints_preserved: bool
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(
            self.hypothesis_adjudications,
            attribute="adjudication_id",
            field_name="hypothesis_adjudications",
        )
        if not self.hypothesis_adjudications:
            raise ValueError("prospective adjudication requires hypothesis results")
        if not self.parent_fingerprints_preserved:
            raise ValueError("prospective adjudication cannot rewrite its motivating parents")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("prospective adjudication visibility cannot be lowered")
        if self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("adjudicating old hypotheses remains outcome-visible")


def nomination_obligations_extension(obligations: NominationObligations) -> ExtensionBinding:
    return ExtensionBinding(
        namespace="prospective-nomination-obligations",
        schema=obligations.SCHEMA,
        payload_sha256=obligations.fingerprint(),
    )


def experiment_design_audit_extension(audit: ExperimentDesignAudit) -> ExtensionBinding:
    return ExtensionBinding(
        namespace="prospective-experiment-design-audit",
        schema=audit.SCHEMA,
        payload_sha256=audit.fingerprint(),
    )

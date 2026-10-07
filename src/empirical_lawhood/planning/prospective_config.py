"""Reusable prospective aggregate records for strict experiment configuration.

These records freeze designs, identities, requirements and decision rules.  They
never carry an authority grant, an issued package, a scientific result or an
outcome-dependent selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord, ActionWordMode, ActionWordSupportStatus
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_stable_id,
)
from empirical_lawhood.kernel.systems import SystemSpec

from .campaigns import CampaignSpec
from .study_issue import StudyAuthorityKind, StudyOperationAuthority


class ProspectiveActionSemanticRole(StrEnum):
    NATIVE_HOLD = "NATIVE_HOLD"
    ACTIVE = "ACTIVE"
    WRONG_SIGN_FALSIFIER = "WRONG_SIGN_FALSIFIER"
    FUTURE_CAUSAL_FALSIFIER = "FUTURE_CAUSAL_FALSIFIER"


@dataclass(frozen=True, slots=True)
class ActionWordRecordedSemanticCompatibility(CanonicalRecord):
    """Design-provenance mapping without schema, identity or evidence transfer."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/action-word-recorded-semantic-compatibility'

    compatibility_id: str
    current_action_word: ObjectIdentity
    historical_semantic_id: str
    semantic_role: ProspectiveActionSemanticRole
    current_identity_reuses_historical_identity: bool
    schema_equivalence_claimed: bool
    historical_evidence_transferred: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.compatibility_id, field_name="compatibility_id")
        validate_nonempty(
            self.historical_semantic_id,
            field_name="historical_semantic_id",
        )
        if self.current_action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("action compatibility must bind a current ActionWord")
        if (
            self.current_identity_reuses_historical_identity
            or self.schema_equivalence_claimed
            or self.historical_evidence_transferred
        ):
            raise ValueError("historical action compatibility is design provenance only")


@dataclass(frozen=True, slots=True)
class ProspectiveSequentialActionWordChart(CanonicalRecord):
    """Outcome-blind exact chart awaiting independent source qualification."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-sequential-action-word-chart'

    chart_id: str
    action_words: tuple[OccurrenceActionWord, ...]
    historical_compatibility: tuple[ActionWordRecordedSemanticCompatibility, ...]
    common_denominator_id: str
    common_retained_history_id: str
    common_receiver_id: str
    common_horizon_id: str
    required_support_status: ActionWordSupportStatus
    required_reason_codes: tuple[str, ...]
    source_qualification_required_before_issue: bool
    source_qualification_claimed_complete: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.chart_id, field_name="chart_id")
        for name in (
            "common_denominator_id",
            "common_retained_history_id",
            "common_receiver_id",
            "common_horizon_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.action_words,
            attribute="word_id",
            field_name="action_words",
        )
        require_sorted_unique_ids(
            self.historical_compatibility,
            attribute="compatibility_id",
            field_name="historical_compatibility",
        )
        require_sorted_unique_strings(
            self.required_reason_codes,
            field_name="required_reason_codes",
        )
        if not self.action_words or len(self.historical_compatibility) != len(self.action_words):
            raise ValueError("prospective action chart compatibility roster differs")
        identities = {
            ObjectIdentity.from_record(value.word_id, value) for value in self.action_words
        }
        if {value.current_action_word for value in self.historical_compatibility} != identities:
            raise ValueError("historical action compatibility omits or adds a word")
        if any(
            value.mode is not ActionWordMode.SEQUENTIAL
            or value.denominator_id != self.common_denominator_id
            or value.retained_history_id != self.common_retained_history_id
            or value.receiver_id != self.common_receiver_id
            or value.horizon_id != self.common_horizon_id
            or value.support_status is not self.required_support_status
            or value.reason_codes != self.required_reason_codes
            for value in self.action_words
        ):
            raise ValueError("prospective action chart changes its common exact grammar")
        if (
            not self.source_qualification_required_before_issue
            or self.source_qualification_claimed_complete
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("prospective action chart overclaims qualification or visibility")


class ProspectivePackageKind(StrEnum):
    EXCLUDED_QUALIFICATION = "EXCLUDED_QUALIFICATION"
    PROTECTED_PARENT = "PROTECTED_PARENT"
    ROUTE_QUALIFICATION = "ROUTE_QUALIFICATION"
    PROSPECTIVE_CHILD = "PROSPECTIVE_CHILD"


@dataclass(frozen=True, slots=True)
class ProspectivePackageLineageNode(CanonicalRecord):
    """One predeclared package identity and its issue conditions."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-package-lineage-node'

    package_id: str
    package_label: str
    kind: ProspectivePackageKind
    parent_package_ids: tuple[str, ...]
    issue_condition_ids: tuple[str, ...]
    maximum_evidence: EvidenceCeiling
    current_linked_campaign_role: str | None
    contains_action_selection_study: bool
    contains_controller: bool
    contains_compiler: bool
    contains_current_admission_or_controller_evaluation_record: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.package_id, field_name="package_id")
        validate_nonempty(self.package_label, field_name="package_label")
        require_sorted_unique_strings(
            self.parent_package_ids,
            field_name="parent_package_ids",
        )
        require_sorted_unique_strings(
            self.issue_condition_ids,
            field_name="issue_condition_ids",
            allow_empty=False,
        )
        if self.current_linked_campaign_role is not None:
            validate_stable_id(
                self.current_linked_campaign_role,
                field_name="current_linked_campaign_role",
            )


@dataclass(frozen=True, slots=True)
class ProspectivePackageLineage(CanonicalRecord):
    """Closed acyclic package graph frozen before any branch outcome."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-package-lineage'

    lineage_id: str
    shared_qualification_package_id: str
    conditional_branch_id: str
    nodes: tuple[ProspectivePackageLineageNode, ...]
    mutually_exclusive_with_branch_ids: tuple[str, ...]
    route_predeclared_before_parent_issue: bool
    outcome_dependent_package_construction: bool
    grants_issue_or_execution_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.lineage_id, field_name="lineage_id")
        validate_stable_id(
            self.shared_qualification_package_id,
            field_name="shared_qualification_package_id",
        )
        validate_nonempty(self.conditional_branch_id, field_name="conditional_branch_id")
        require_sorted_unique_ids(self.nodes, attribute="package_id", field_name="nodes")
        require_sorted_unique_strings(
            self.mutually_exclusive_with_branch_ids,
            field_name="mutually_exclusive_with_branch_ids",
            allow_empty=False,
        )
        by_id = {value.package_id: value for value in self.nodes}
        if self.shared_qualification_package_id not in by_id:
            raise ValueError("package lineage omits its shared qualification root")
        for node in self.nodes:
            if node.package_id in node.parent_package_ids or not set(
                node.parent_package_ids
            ).issubset(by_id):
                raise ValueError("package lineage contains an invalid parent")
        completed: set[str] = set()

        def visit(node_id: str, active: set[str]) -> None:
            if node_id in active:
                raise ValueError("package lineage contains a cycle")
            if node_id in completed:
                return
            active.add(node_id)
            for parent_id in by_id[node_id].parent_package_ids:
                visit(parent_id, active)
            active.remove(node_id)
            completed.add(node_id)

        for node_id in by_id:
            visit(node_id, set())
        if (
            not self.route_predeclared_before_parent_issue
            or self.outcome_dependent_package_construction
            or self.grants_issue_or_execution_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("prospective package lineage exceeds authoring authority")


@dataclass(frozen=True, slots=True)
class ProspectiveSystemExperimentCampaignBinding(CanonicalRecord):
    """Identity-only join of already authored system, experiment and campaign."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-system-experiment-campaign-binding'

    binding_id: str
    system: ObjectIdentity
    experiment: ObjectIdentity
    campaign: ObjectIdentity
    frozen_before_issue: bool
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        expected = (
            (self.system, SystemSpec.SCHEMA),
            (self.experiment, ExperimentSpec.SCHEMA),
            (self.campaign, CampaignSpec.SCHEMA),
        )
        if any(value.object_schema != schema for value, schema in expected):
            raise ValueError("system/experiment/campaign binding contains another schema")
        if (
            not self.frozen_before_issue
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("system/experiment/campaign binding is not prospective")


class ProspectiveAuthorityOperation(StrEnum):
    # ISSUE grants custody for the base study; extensions have a separate issue act.
    ISSUE = "ISSUE"
    EXTENSION_ISSUE = "EXTENSION_ISSUE"
    # BASE_EXECUTE governs the already issued base under its own grant.
    # Its subject must stay distinct from the execution grant for the
    # executable study that carries the issued extensions.
    BASE_EXECUTE = "BASE_EXECUTE"
    EXECUTE = "EXECUTE"
    RECOVER = "RECOVER"
    REVEAL = "REVEAL"


class ProspectiveAuthorityAccess(StrEnum):
    OUTCOME_BLIND = "OUTCOME_BLIND"
    SEALED_EXECUTION = "SEALED_EXECUTION"
    EVALUATOR_ONLY_REVEAL = "EVALUATOR_ONLY_REVEAL"


_AUTHORITY_OPERATION_CONTRACT = {
    ProspectiveAuthorityOperation.ISSUE: (
        StudyAuthorityKind.CUSTODY_PUBLICATION,
        ProspectiveAuthorityAccess.OUTCOME_BLIND,
    ),
    ProspectiveAuthorityOperation.EXTENSION_ISSUE: (
        StudyAuthorityKind.CUSTODY_PUBLICATION,
        ProspectiveAuthorityAccess.OUTCOME_BLIND,
    ),
    ProspectiveAuthorityOperation.BASE_EXECUTE: (
        StudyAuthorityKind.EXPERIMENT_EXECUTION,
        ProspectiveAuthorityAccess.SEALED_EXECUTION,
    ),
    ProspectiveAuthorityOperation.EXECUTE: (
        StudyAuthorityKind.EXPERIMENT_EXECUTION,
        ProspectiveAuthorityAccess.SEALED_EXECUTION,
    ),
    ProspectiveAuthorityOperation.RECOVER: (
        StudyAuthorityKind.EXPERIMENT_EXECUTION,
        ProspectiveAuthorityAccess.SEALED_EXECUTION,
    ),
    ProspectiveAuthorityOperation.REVEAL: (
        StudyAuthorityKind.OUTCOME_REVEAL,
        ProspectiveAuthorityAccess.EVALUATOR_ONLY_REVEAL,
    ),
}


@dataclass(frozen=True, slots=True)
class ProspectiveAuthorityRequirement(CanonicalRecord):
    """One required authority lookup; it is not an authority record or grant."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-authority-requirement'

    requirement_id: str
    package_id: str
    operation: ProspectiveAuthorityOperation
    scope_id: str
    responsible_role_id: str
    expected_grantee_id: str
    required_authority_schema: str
    required_authority_kind: StudyAuthorityKind
    required_access: ProspectiveAuthorityAccess
    prerequisite_requirement_ids: tuple[str, ...]
    authority_record_embedded: bool
    authority_granted: bool

    def __post_init__(self) -> None:
        for name in (
            "requirement_id",
            "package_id",
            "scope_id",
            "responsible_role_id",
            "expected_grantee_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_schema(self.required_authority_schema)
        require_sorted_unique_strings(
            self.prerequisite_requirement_ids,
            field_name="prerequisite_requirement_ids",
        )
        if self.required_authority_schema != StudyOperationAuthority.SCHEMA:
            raise ValueError("authority requirement expects another grant schema")
        expected_kind, expected_access = _AUTHORITY_OPERATION_CONTRACT[self.operation]
        if (
            self.required_authority_kind is not expected_kind
            or self.required_access is not expected_access
        ):
            raise ValueError("authority requirement changes its operation contract")
        if self.authority_record_embedded or self.authority_granted:
            raise ValueError("prospective authority requirement cannot contain or grant authority")


@dataclass(frozen=True, slots=True)
class ProspectiveAuthorityRequirementSet(CanonicalRecord):
    """Exact per-package custody/execute/recover/applicable-reveal closure.

    Historical families may retain ISSUE -> EXECUTE.  A set that declares one
    EXTENSION_ISSUE must declare it for every package and use dual custody.
    A set that additionally declares BASE_EXECUTE must declare it for every
    package and use ISSUE -> EXTENSION_ISSUE -> BASE_EXECUTE -> EXECUTE.
    """

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-authority-requirement-set'

    requirement_set_id: str
    package_ids: tuple[str, ...]
    reveal_package_ids: tuple[str, ...]
    requirements: tuple[ProspectiveAuthorityRequirement, ...]
    distinct_operation_roles_required: bool
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.requirement_set_id, field_name="requirement_set_id")
        require_sorted_unique_strings(
            self.package_ids,
            field_name="package_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.reveal_package_ids,
            field_name="reveal_package_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.requirements,
            attribute="requirement_id",
            field_name="requirements",
        )
        if not set(self.reveal_package_ids).issubset(self.package_ids):
            raise ValueError("reveal requirements name an unknown package")
        observed = {(value.package_id, value.operation) for value in self.requirements}
        uses_extension_issue = any(
            value.operation is ProspectiveAuthorityOperation.EXTENSION_ISSUE
            for value in self.requirements
        )
        uses_base_execute = any(
            value.operation is ProspectiveAuthorityOperation.BASE_EXECUTE
            for value in self.requirements
        )
        if uses_base_execute and not uses_extension_issue:
            raise ValueError("base execution requires additive extension custody")
        custody_operations = (
            (
                ProspectiveAuthorityOperation.ISSUE,
                ProspectiveAuthorityOperation.EXTENSION_ISSUE,
            )
            if uses_extension_issue
            else (ProspectiveAuthorityOperation.ISSUE,)
        )
        expected = {
            (package_id, operation)
            for package_id in self.package_ids
            for operation in (
                *custody_operations,
                *((ProspectiveAuthorityOperation.BASE_EXECUTE,) if uses_base_execute else ()),
                ProspectiveAuthorityOperation.EXECUTE,
                ProspectiveAuthorityOperation.RECOVER,
            )
        } | {
            (package_id, ProspectiveAuthorityOperation.REVEAL)
            for package_id in self.reveal_package_ids
        }
        if observed != expected or len(observed) != len(self.requirements):
            raise ValueError("authority requirement operation closure differs")
        role_by_operation: dict[ProspectiveAuthorityOperation, set[str]] = {}
        for value in self.requirements:
            role_by_operation.setdefault(value.operation, set()).add(value.responsible_role_id)
        if any(len(values) != 1 for values in role_by_operation.values()) or len(
            {next(iter(values)) for values in role_by_operation.values()}
        ) != len(role_by_operation):
            raise ValueError("authority operation roles are not globally distinct")
        by_id = {value.requirement_id: value for value in self.requirements}
        if any(
            not set(value.prerequisite_requirement_ids).issubset(by_id)
            for value in self.requirements
        ):
            raise ValueError("authority requirement names an unknown prerequisite")
        by_package_operation = {
            (value.package_id, value.operation): value for value in self.requirements
        }
        for package_id in self.package_ids:
            issue = by_package_operation[(package_id, ProspectiveAuthorityOperation.ISSUE)]
            extension_issue = by_package_operation.get(
                (package_id, ProspectiveAuthorityOperation.EXTENSION_ISSUE)
            )
            base_execute = by_package_operation.get(
                (package_id, ProspectiveAuthorityOperation.BASE_EXECUTE)
            )
            execute = by_package_operation[(package_id, ProspectiveAuthorityOperation.EXECUTE)]
            recover = by_package_operation[(package_id, ProspectiveAuthorityOperation.RECOVER)]
            execute_prerequisite = (
                base_execute.requirement_id
                if base_execute is not None
                else (
                    extension_issue.requirement_id
                    if extension_issue is not None
                    else issue.requirement_id
                )
            )
            reveal = by_package_operation.get((package_id, ProspectiveAuthorityOperation.REVEAL))
            if (
                issue.prerequisite_requirement_ids
                or (
                    extension_issue is not None
                    and extension_issue.prerequisite_requirement_ids != (issue.requirement_id,)
                )
                or (
                    base_execute is not None
                    and (
                        extension_issue is None
                        or base_execute.prerequisite_requirement_ids
                        != (extension_issue.requirement_id,)
                    )
                )
                or execute.prerequisite_requirement_ids != (execute_prerequisite,)
                or recover.prerequisite_requirement_ids != (execute.requirement_id,)
                or (
                    reveal is not None
                    and reveal.prerequisite_requirement_ids != (execute.requirement_id,)
                )
            ):
                raise ValueError("authority requirement phase order differs")
        if (
            not self.distinct_operation_roles_required
            or self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("authority requirement set overclaims a grant or visibility")


@dataclass(frozen=True, slots=True)
class ProspectiveSelectorTransition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-selector-transition'

    precedence: int
    condition: str
    transition: str

    def __post_init__(self) -> None:
        if self.precedence < 1:
            raise ValueError("selector transition precedence must be positive")
        validate_nonempty(self.condition, field_name="condition")
        validate_nonempty(self.transition, field_name="transition")


@dataclass(frozen=True, slots=True)
class ProspectiveTerminalRule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-terminal-rule'

    precedence: int
    first_controlling_condition: str
    terminal_disposition: str
    highest_retained_object: str
    downstream_consequence: str

    def __post_init__(self) -> None:
        if self.precedence < 1:
            raise ValueError("terminal precedence must be positive")
        for name in (
            "first_controlling_condition",
            "terminal_disposition",
            "highest_retained_object",
            "downstream_consequence",
        ):
            validate_nonempty(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class ProspectiveTerminalMatrix(CanonicalRecord):
    """Closed first-applicable transition and terminal mapping."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/prospective-terminal-matrix'

    matrix_id: str
    selector_transitions: tuple[ProspectiveSelectorTransition, ...]
    terminal_rules: tuple[ProspectiveTerminalRule, ...]
    first_applicable_terminal_wins: bool
    valid_negative_mixed_partial_stopped_or_unevaluable_is_complete: bool
    postissue_alternate_rescue_allowed: bool
    grants_authority_or_followon_action: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.matrix_id, field_name="matrix_id")
        if tuple(value.precedence for value in self.selector_transitions) != tuple(
            range(1, len(self.selector_transitions) + 1)
        ):
            raise ValueError("selector transition precedence is not closed")
        if tuple(value.precedence for value in self.terminal_rules) != tuple(
            range(1, len(self.terminal_rules) + 1)
        ):
            raise ValueError("terminal rule precedence is not closed")
        if not self.selector_transitions or not self.terminal_rules:
            raise ValueError("terminal matrix cannot be empty")
        if (
            not self.first_applicable_terminal_wins
            or not self.valid_negative_mixed_partial_stopped_or_unevaluable_is_complete
            or self.postissue_alternate_rescue_allowed
            or self.grants_authority_or_followon_action
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("prospective terminal matrix changes its authority or precedence")


__all__ = [
    'ActionWordRecordedSemanticCompatibility',
    'ProspectiveActionSemanticRole',
    'ProspectiveAuthorityAccess',
    'ProspectiveAuthorityOperation',
    'ProspectiveAuthorityRequirementSet',
    'ProspectiveAuthorityRequirement',
    'ProspectivePackageKind',
    'ProspectivePackageLineageNode',
    'ProspectivePackageLineage',
    'ProspectiveSelectorTransition',
    'ProspectiveSequentialActionWordChart',
    'ProspectiveSystemExperimentCampaignBinding',
    'ProspectiveTerminalMatrix',
    'ProspectiveTerminalRule',
]

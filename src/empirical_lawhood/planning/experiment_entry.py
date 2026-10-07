"""Additive experiment-entry checklist bound to the accepted draft/spec types."""

from __future__ import annotations

from empirical_lawhood.planning.study_authoring import RetrospectiveStudyDraft

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import AssignmentKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.worlds import WorldKind

from .formal_gaps import (
    FormalGapCoverage,
    FormalGapEvidenceWorld,
    FormalGapRegister,
    validate_formal_gap_coverage,
)
from .study_authoring import StudyDraft, SourceAccessDisposition


class ExperimentEntryRequirement(StrEnum):
    EVIDENCE_WORLD_AND_CLAIM_CEILING = "EVIDENCE_WORLD_AND_CLAIM_CEILING"
    LAW_IDENTITIES = "LAW_IDENTITIES"
    PHYSICAL_UNIT_NESTING_REPLICATION = "PHYSICAL_UNIT_NESTING_REPLICATION"
    SOURCE_RELEASE_MATERIALIZATION_VIEWS = "SOURCE_RELEASE_MATERIALIZATION_VIEWS"
    SPLIT_SEAL_EVALUATION_REVEAL = "SPLIT_SEAL_EVALUATION_REVEAL"
    RECEIVER_COMPARATOR_DIRECTION_FRAME_UNITS = "RECEIVER_COMPARATOR_DIRECTION_FRAME_UNITS"
    ACTION_STAGES_AND_CLOCKS = "ACTION_STAGES_AND_CLOCKS"
    CAUSAL_CUTOFF_HORIZON_HISTORY = "CAUSAL_CUTOFF_HORIZON_HISTORY"
    CONTROLS_AND_NEGATIVE_CONTROLS = "CONTROLS_AND_NEGATIVE_CONTROLS"
    SUPPORTED_RECURRENCE_LOCAL_GEOMETRY = "SUPPORTED_RECURRENCE_LOCAL_GEOMETRY"
    NONCOMPENSATING_ADMISSION_GATES = "NONCOMPENSATING_ADMISSION_GATES"
    HOLD_AND_TERMINATION = "HOLD_AND_TERMINATION"
    ESTIMANDS_THRESHOLDS_UNCERTAINTY_MULTIPLICITY = "ESTIMANDS_THRESHOLDS_UNCERTAINTY_MULTIPLICITY"
    FALSIFIERS_AND_UNEVALUABLE = "FALSIFIERS_AND_UNEVALUABLE"
    ALLOWED_TERMINAL_CLASSES = "ALLOWED_TERMINAL_CLASSES"
    PREOUTCOME_CONDITIONAL_FOLLOWUP = "PREOUTCOME_CONDITIONAL_FOLLOWUP"
    EXECUTION_ROUTE_AND_NO_DOWNGRADE = "EXECUTION_ROUTE_AND_NO_DOWNGRADE"
    STORAGE_CUSTODY_DURABILITY = "STORAGE_CUSTODY_DURABILITY"
    ROLES_AND_OPERATION_AUTHORITIES = "ROLES_AND_OPERATION_AUTHORITIES"
    EXPLORATION_LINEAGE_AND_CONFIRMATION = "EXPLORATION_LINEAGE_AND_CONFIRMATION"
    FORMAL_GAP_REGISTER_AND_SELECTION = "FORMAL_GAP_REGISTER_AND_SELECTION"
    OUTCOME_BLIND_GAP_COVERAGE = "OUTCOME_BLIND_GAP_COVERAGE"
    FORMAL_ESTIMATORS_CONTROLS_SUPPORT_UNCERTAINTY = (
        "FORMAL_ESTIMATORS_CONTROLS_SUPPORT_UNCERTAINTY"
    )
    SOURCE_LOCAL_RECURRENCE_NO_POOLING = "SOURCE_LOCAL_RECURRENCE_NO_POOLING"
    EXCLUDED_HIL_LIVE_PHYSICAL_ACTS = "EXCLUDED_HIL_LIVE_PHYSICAL_ACTS"


class ExperimentEntryTransition(StrEnum):
    DESIGN_COMPLETE = "DESIGN_COMPLETE"
    ISSUE = "ISSUE"
    EXECUTE = "EXECUTE"
    REVEAL = "REVEAL"
    EVALUATE = "EVALUATE"
    PROMOTE = "PROMOTE"


class ExperimentTerminalClass(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    MIXED = "MIXED"
    STOPPED = "STOPPED"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ExperimentEntryRequirementBinding(CanonicalRecord):
    """One checklist requirement bound to accepted objects or policies."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/experiment-entry-requirement-binding'

    requirement: ExperimentEntryRequirement
    object_ids: tuple[str, ...]
    readiness: ReadinessStatus
    reason_codes: tuple[str, ...]

    @property
    def requirement_id(self) -> str:
        return self.requirement.value

    def __post_init__(self) -> None:
        require_sorted_unique_strings(
            self.object_ids,
            field_name="object_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=self.readiness is ReadinessStatus.READY,
        )
        if self.readiness is ReadinessStatus.READY and self.reason_codes:
            raise ValueError("ready entry requirement cannot carry stop reasons")


@dataclass(frozen=True, slots=True)
class ExperimentEntryChecklist(CanonicalRecord):
    """Twenty-five-field pre-issue overlay; never a second experiment spec."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/experiment-entry-checklist'

    checklist_id: str
    draft: ObjectIdentity
    formal_gap_register: ObjectIdentity
    formal_gap_coverage: ObjectIdentity
    portfolio_world: FormalGapEvidenceWorld
    execution_route_id: str
    durability_disposition_id: str
    bindings: tuple[ExperimentEntryRequirementBinding, ...]
    transitions: tuple[ExperimentEntryTransition, ...]
    allowed_terminal_classes: tuple[ExperimentTerminalClass, ...]
    outcome_access: OutcomeAccess
    expected_result_declared: bool = False

    def __post_init__(self) -> None:
        for name, value in (
            ("checklist_id", self.checklist_id),
            ("execution_route_id", self.execution_route_id),
            ("durability_disposition_id", self.durability_disposition_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.draft.object_schema != (
            RetrospectiveStudyDraft.SCHEMA
            if isinstance(self, RetrospectiveEntryChecklist)
            else StudyDraft.SCHEMA
        ):
            raise ValueError("entry checklist must identify a ProgrammeDraft")
        if self.formal_gap_register.object_schema != FormalGapRegister.SCHEMA:
            raise ValueError("entry checklist must identify a FormalGapRegister")
        if self.formal_gap_coverage.object_schema != FormalGapCoverage.SCHEMA:
            raise ValueError("entry checklist must identify FormalGapCoverage")
        require_sorted_unique_ids(
            self.bindings,
            attribute="requirement_id",
            field_name="bindings",
        )
        observed_requirements = {value.requirement for value in self.bindings}
        if observed_requirements != set(ExperimentEntryRequirement):
            missing = sorted(
                value.value for value in set(ExperimentEntryRequirement) - observed_requirements
            )
            raise ValueError(f"entry checklist omits requirements: {missing}")
        transition_values = tuple(value.value for value in self.transitions)
        expected_transitions = tuple(sorted(value.value for value in ExperimentEntryTransition))
        if transition_values != expected_transitions:
            raise ValueError("entry transitions must be complete, sorted and distinct")
        terminal_values = tuple(value.value for value in self.allowed_terminal_classes)
        expected_terminals = tuple(sorted(value.value for value in ExperimentTerminalClass))
        if terminal_values != expected_terminals:
            raise ValueError("allowed terminal classes must be complete and result-neutral")
        if self.portfolio_world.excluded_from_portfolio:
            raise ValueError("entry checklist rejects HIL/live/new physical portfolio worlds")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("experiment-entry review must be outcome-blind")
        if self.expected_result_declared:
            raise ValueError("experiment entry cannot declare an expected result")


@dataclass(frozen=True, slots=True)
class ExperimentEntryPackage(CanonicalRecord):
    """Self-contained entry evidence consumed by the issue gate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/experiment-entry-package'

    package_id: str
    register: FormalGapRegister
    coverage: FormalGapCoverage
    checklist: ExperimentEntryChecklist

    def __post_init__(self) -> None:
        if (
            not isinstance(self, RetrospectiveEntryPackage)
            and type(self.checklist) is not ExperimentEntryChecklist
        ):
            raise ValueError("prospective entry requires its original checklist")
        validate_stable_id(self.package_id, field_name="package_id")
        if self.checklist.formal_gap_register != ObjectIdentity.from_record(
            self.register.register_id,
            self.register,
        ):
            raise ValueError("entry checklist binds another formal-gap register")
        if self.checklist.formal_gap_coverage != ObjectIdentity.from_record(
            self.coverage.coverage_id,
            self.coverage,
        ):
            raise ValueError("entry checklist binds another formal-gap coverage")
        validate_formal_gap_coverage(self.register, self.coverage)


@dataclass(frozen=True, slots=True)
class StudyDefinition(CanonicalRecord):
    """Mandatory standard root for one claim-bearing programme candidate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/study-definition'

    package_id: str
    draft: StudyDraft
    entry_package: ExperimentEntryPackage

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveAuthoringBase):
            if type(self.entry_package) is not ExperimentEntryPackage:
                raise ValueError("prospective authoring requires its original entry package")
            if type(self.draft) is not StudyDraft:
                raise ValueError("ProgrammeAuthoringPackage requires its original draft schema")
        validate_stable_id(self.package_id, field_name="package_id")
        validate_experiment_entry(self.draft, self.entry_package, for_issue=False)


@dataclass(frozen=True, slots=True)
class ProposedStudyExtension(CanonicalRecord):
    """One typed additive payload proposed beside an unchanged authoring root."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/proposed-study-extension'

    extension_id: str
    namespace_id: str
    payload: ObjectIdentity
    payload_size_bytes: int
    decoder_key: str
    decoder_version: str
    decoder_config_sha256: str
    required_for_activation: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        for name, value in (
            ("extension_id", self.extension_id),
            ("namespace_id", self.namespace_id),
            ("decoder_key", self.decoder_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_schema(self.payload.object_schema)
        validate_semantic_version(self.decoder_version)
        validate_sha256(
            self.decoder_config_sha256,
            field_name="decoder_config_sha256",
        )
        if self.payload_size_bytes < 1:
            raise ValueError("proposed extension payload size must be positive")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("programme extensions must be proposed outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("programme extensions must remain prospective")


@dataclass(frozen=True, slots=True)
class ProposedStudyExtensionSet(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/proposed-study-extension-set'

    extension_set_id: str
    authoring_package: ObjectIdentity
    namespace_roster: tuple[str, ...]
    extensions: tuple[ProposedStudyExtension, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.extension_set_id, field_name="extension_set_id")
        require_sorted_unique_strings(
            self.namespace_roster,
            field_name="namespace_roster",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.extensions,
            attribute="extension_id",
            field_name="extensions",
        )
        if not self.extensions:
            raise ValueError("proposed extension set cannot be empty")
        namespaces = tuple(sorted(value.namespace_id for value in self.extensions))
        if namespaces != self.namespace_roster:
            raise ValueError("proposed extension namespaces differ from their exact roster")


@dataclass(frozen=True, slots=True)
class ExecutableStudyDefinition(CanonicalRecord):
    "Additive extension wrapper; the base authoring bytes remain unchanged."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/executable-study-definition'

    package_id: str
    base: StudyDefinition
    extension_set: ProposedStudyExtensionSet

    def __post_init__(self) -> None:
        if not isinstance(self, RetrospectiveAuthoringPackage):
            if type(self.base) is not StudyDefinition:
                raise ValueError("ExecutableStudyDefinition requires its original base schema")
        validate_stable_id(self.package_id, field_name="package_id")
        if self.extension_set.authoring_package != ObjectIdentity.from_record(
            self.base.package_id,
            self.base,
        ):
            raise ValueError("programme extension set binds another authoring package")
        validate_experiment_entry(self.base.draft, self.base.entry_package, for_issue=False)


def validate_experiment_entry(
    draft: StudyDraft,
    package: ExperimentEntryPackage,
    *,
    for_issue: bool,
) -> None:
    """Validate a design-complete or issue-bound package without granting authority."""

    checklist = package.checklist
    if checklist.draft != ObjectIdentity.from_record(draft.draft_id, draft):
        raise ValueError("entry package binds another ProgrammeDraft")
    if package.coverage.candidate_act_id != draft.draft_id:
        raise ValueError("formal-gap coverage binds another candidate act")
    if draft.unresolved_decisions:
        raise ValueError("design-complete entry cannot retain unresolved decisions")
    if draft.system is None or draft.experiment is None or draft.campaign is None:
        raise ValueError("design-complete entry requires system, experiment and campaign")
    if any(
        value.access_disposition is not SourceAccessDisposition.VERIFIED_ACCESS
        for value in draft.source_materializations
    ):
        raise ValueError("experiment entry requires qualified, verified source access")

    world_kind = draft.system.world.kind
    mapped_world = checklist.portfolio_world.accepted_world_kind
    if world_kind is not mapped_world:
        raise ValueError("entry portfolio world differs from the accepted SystemSpec world")
    assignment_kind = draft.experiment.assignment.kind
    if checklist.portfolio_world is FormalGapEvidenceWorld.RETROSPECTIVE_DATASET:
        if assignment_kind not in {
            AssignmentKind.OBSERVATIONAL,
            AssignmentKind.LOGGED_INTERVENTION,
        }:
            raise ValueError("retrospective dataset cannot become a new intervention")
    elif checklist.portfolio_world is FormalGapEvidenceWorld.NUMERICAL_SIMULATOR:
        action_free_observation = (
            assignment_kind is AssignmentKind.OBSERVATIONAL
            and not draft.experiment.relation.action_quantity_ids
        )
        if (
            assignment_kind is not AssignmentKind.SIMULATOR_INTERVENTION
            and not action_free_observation
        ):
            raise ValueError(
                "numerical simulator entry requires intervention or action-free observation"
            )
    elif world_kind is WorldKind.PHYSICAL_EXPERIMENT:
        raise ValueError("new physical experiment is outside the accepted portfolio")

    validate_formal_gap_coverage(package.register, package.coverage)
    for binding in checklist.bindings:
        allowed_authority_gate = (
            binding.requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
            and binding.readiness is ReadinessStatus.AUTHORITY_REQUIRED
        )
        if binding.readiness is not ReadinessStatus.READY and not allowed_authority_gate:
            raise ValueError(
                f"entry requirement is not design-complete: {binding.requirement.value}"
            )
    if for_issue:
        if draft.experiment.readiness is not ReadinessStatus.AUTHORITY_REQUIRED:
            raise ValueError("issue-bound ExperimentSpec must await separate execution authority")
        authority_binding = next(
            value
            for value in checklist.bindings
            if value.requirement is ExperimentEntryRequirement.ROLES_AND_OPERATION_AUTHORITIES
        )
        if authority_binding.readiness is not ReadinessStatus.AUTHORITY_REQUIRED:
            raise ValueError("issue cannot imply execution or reveal authority")


__all__ = [
    "ExperimentEntryChecklist",
    "ExperimentEntryPackage",
    "ExperimentEntryRequirement",
    "ExperimentEntryRequirementBinding",
    "ExperimentEntryTransition",
    "ExperimentTerminalClass",
    'StudyDefinition',
    'ExecutableStudyDefinition',
    'ProposedStudyExtension',
    'ProposedStudyExtensionSet',
    "validate_experiment_entry",
]


@dataclass(frozen=True, slots=True)
class RetrospectiveAuthoringBase(StudyDefinition):
    "Explicit retrospective authoring compatibility under its declared schema."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-authoring-base'

    draft: RetrospectiveStudyDraft
    entry_package: RetrospectiveEntryPackage

    def __post_init__(self) -> None:
        if type(self.draft) is not RetrospectiveStudyDraft:
            raise ValueError("RetrospectiveAuthoringBase requires its exact draft schema")
        if type(self.entry_package) is not RetrospectiveEntryPackage:
            raise ValueError("historical authoring requires its exact entry package")
        StudyDefinition.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveAuthoringPackage(ExecutableStudyDefinition):
    "Explicit retrospective authoring compatibility under its declared schema."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-authoring-package'

    base: RetrospectiveAuthoringBase

    def __post_init__(self) -> None:
        if type(self.base) is not RetrospectiveAuthoringBase:
            raise ValueError("RetrospectiveAuthoringPackage requires its exact base schema")
        ExecutableStudyDefinition.__post_init__(self)


@dataclass(frozen=True, slots=True)
class RetrospectiveEntryChecklist(ExperimentEntryChecklist):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-entry-checklist'


@dataclass(frozen=True, slots=True)
class RetrospectiveEntryPackage(ExperimentEntryPackage):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/retrospective-entry-package'
    checklist: RetrospectiveEntryChecklist

    def __post_init__(self) -> None:
        if type(self.checklist) is not RetrospectiveEntryChecklist:
            raise ValueError("historical entry requires its exact checklist")
        ExperimentEntryPackage.__post_init__(self)

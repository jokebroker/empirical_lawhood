"Declarative linked qualification/admission profile with conditional controller use."

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
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

from .evidence_profiles import EvidenceProfileSelection
from .study_authoring import CapabilitySelection


class LinkedCampaignTopologyKind(StrEnum):
    EXCLUDED_PROTECTED_CONDITIONAL_CONTROLLER_USE = "EXCLUDED_PROTECTED_CONDITIONAL_CONTROLLER_USE"


class LinkedCampaignPackageRole(StrEnum):
    EXCLUDED_QUALIFICATION = "EXCLUDED_QUALIFICATION"
    PROTECTED_MEASUREMENT_THROUGH_ADMISSION_PARENT = "PROTECTED_MEASUREMENT_THROUGH_ADMISSION_PARENT"
    CONDITIONAL_CONTROLLER_USE_ROUTE_QUALIFICATION = "CONDITIONAL_CONTROLLER_USE_ROUTE_QUALIFICATION"
    CONDITIONAL_CONTROLLER_USE_CHILD = "CONDITIONAL_CONTROLLER_USE_CHILD"


class LinkedCampaignStageRole(StrEnum):
    SOURCE_MATERIALIZATION = "SOURCE_MATERIALIZATION"
    EVIDENCE_PROJECTION = "EVIDENCE_PROJECTION"
    METHOD_IDENTIFICATION = "METHOD_IDENTIFICATION"
    LAW_QUALIFICATION = "LAW_QUALIFICATION"
    ATLAS_ASSEMBLY = "ATLAS_ASSEMBLY"
    ADMISSION_EVIDENCE = "ADMISSION_EVIDENCE"
    ADMISSION = "ADMISSION"
    REACHABILITY = "REACHABILITY"
    PROGRAMME_AUTHORING = "PROGRAMME_AUTHORING"


class LinkedCampaignOwnerRole(StrEnum):
    PROJECTION = "PROJECTION"
    METHOD_EVIDENCE = "METHOD_EVIDENCE"
    QUALIFICATION_PROFILE = "QUALIFICATION_PROFILE"
    SOLE_LAW_FINALIZER = "SOLE_LAW_FINALIZER"
    BATCH_ATLAS = "BATCH_ATLAS"
    ADMISSION_EVIDENCE_PRODUCER = "ADMISSION_EVIDENCE_PRODUCER"
    ADMISSION = "ADMISSION"
    REACHABILITY = "REACHABILITY"
    PROGRAMME_AUTHOR = "PROGRAMME_AUTHOR"
    SOLE_CONTROLLER_COMPILER = "SOLE_CONTROLLER_COMPILER"
    CONTROLLER_RUNTIME = "CONTROLLER_RUNTIME"
    CONTROLLER_USE_ROUTE_QUALIFIER = "CONTROLLER_USE_ROUTE_QUALIFIER"
    NESTED_CONTROLLER_USE = "NESTED_CONTROLLER_USE"


class LinkedCampaignArtifactRole(StrEnum):
    SOURCE = "SOURCE"
    PROJECTION = "PROJECTION"
    METHOD_CANDIDATE = "METHOD_CANDIDATE"
    LAW_BATCH = "LAW_BATCH"
    ATLAS_OR_OBSTRUCTION = "ATLAS_OR_OBSTRUCTION"
    ADMISSION_CORPUS = "ADMISSION_CORPUS"
    CONTROLLER_PROGRAMME = "CONTROLLER_PROGRAMME"
    COMPILED_CONTROLLER = "COMPILED_CONTROLLER"
    SEALED_CONTROLLER_USE = "SEALED_CONTROLLER_USE"


_STAGE_ROLES = frozenset(LinkedCampaignStageRole)
_OWNER_ROLES = frozenset(LinkedCampaignOwnerRole)
_ARTIFACT_ROLES = frozenset(LinkedCampaignArtifactRole)
_PACKAGE_ROLES = frozenset(LinkedCampaignPackageRole)


@dataclass(frozen=True, slots=True)
class LinkedCampaignRosters(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/linked-campaign-rosters'

    physical_independent_unit_ids: tuple[str, ...]
    development_split_ids: tuple[str, ...]
    evaluation_split_ids: tuple[str, ...]
    denominator_member_ids: tuple[str, ...]
    candidate_version_ids: tuple[str, ...]
    action_word_ids: tuple[str, ...]
    support_cell_ids: tuple[str, ...]
    qualification_view_ids: tuple[str, ...]
    controller_use_independent_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, values in (
            ("physical_independent_unit_ids", self.physical_independent_unit_ids),
            ("development_split_ids", self.development_split_ids),
            ("evaluation_split_ids", self.evaluation_split_ids),
            ("denominator_member_ids", self.denominator_member_ids),
            ("candidate_version_ids", self.candidate_version_ids),
            ("action_word_ids", self.action_word_ids),
            ("support_cell_ids", self.support_cell_ids),
            ("qualification_view_ids", self.qualification_view_ids),
            ("controller_use_independent_unit_ids", self.controller_use_independent_unit_ids),
        ):
            require_sorted_unique_strings(
                values,
                field_name=name,
                allow_empty=name in {"action_word_ids", "controller_use_independent_unit_ids"},
            )
        if set(self.development_split_ids) & set(self.evaluation_split_ids):
            raise ValueError("linked campaign development and evaluation splits overlap")
        if set(self.physical_independent_unit_ids) & set(self.controller_use_independent_unit_ids):
            raise ValueError("measurement-through-admission and controller use physical independent units must be disjoint")


@dataclass(frozen=True, slots=True)
class LinkedCampaignCapabilityBinding(CanonicalRecord):
    """One exact executable selection plus non-executable config identity."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/linked-campaign-capability-binding'

    binding_id: str
    role: LinkedCampaignStageRole
    selection: CapabilitySelection
    config_id: str
    config_schema: str
    config_schema_sha256: str
    config_content_sha256: str
    config_artifact_id: str
    resource_budget: ResourceBudget
    resource_lock_ids: tuple[str, ...]
    maximum_attempts: int

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("config_id", self.config_id),
            ("config_artifact_id", self.config_artifact_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_schema(self.config_schema)
        validate_sha256(self.config_schema_sha256, field_name="config_schema_sha256")
        validate_sha256(self.config_content_sha256, field_name="config_content_sha256")
        require_sorted_unique_strings(self.resource_lock_ids, field_name="resource_lock_ids")
        if self.maximum_attempts < 1 or self.maximum_attempts > 8:
            raise ValueError("linked campaign attempt bound must be in [1, 8]")


@dataclass(frozen=True, slots=True)
class LinkedCampaignOwnerBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/linked-campaign-owner-binding'

    binding_id: str
    role: LinkedCampaignOwnerRole
    owner: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")


@dataclass(frozen=True, slots=True)
class LinkedCampaignArtifactBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/linked-campaign-artifact-binding'

    binding_id: str
    role: LinkedCampaignArtifactRole
    payload_schema: str
    external_role_id: str
    maximum_bytes: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_schema(self.payload_schema)
        validate_stable_id(self.external_role_id, field_name="external_role_id")
        if self.maximum_bytes < 1:
            raise ValueError("linked campaign artifact requires a positive byte bound")
        expected_visibility = (
            VisibilityCeiling.PROSPECTIVE
            if self.outcome_access in {OutcomeAccess.OUTCOME_BLIND, OutcomeAccess.EVALUATION_SEALED}
            else VisibilityCeiling.OUTCOME_VISIBLE
        )
        if self.visibility_ceiling is not expected_visibility:
            raise ValueError("linked campaign artifact visibility differs from outcome access")


@dataclass(frozen=True, slots=True)
class ReserveUnit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/reserve-unit'

    reserve_id: str
    physical_independent_unit_id: str
    rank: int

    def __post_init__(self) -> None:
        validate_stable_id(self.reserve_id, field_name="reserve_id")
        validate_stable_id(
            self.physical_independent_unit_id,
            field_name="physical_independent_unit_id",
        )
        if self.rank < 0:
            raise ValueError("reserve rank must be nonnegative")


@dataclass(frozen=True, slots=True)
class CompleteUnitReserveGroup(CanonicalRecord):
    """Predeclared complete-unit substitutes; never rows or favorable members."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/complete-unit-reserve-group'

    group_id: str
    primary_physical_unit_ids: tuple[str, ...]
    reserves: tuple[ReserveUnit, ...]
    maximum_substitutions: int

    def __post_init__(self) -> None:
        validate_stable_id(self.group_id, field_name="group_id")
        require_sorted_unique_strings(
            self.primary_physical_unit_ids,
            field_name="primary_physical_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.reserves, attribute="reserve_id", field_name="reserves")
        if not self.reserves:
            raise ValueError("complete-unit reserve group requires reserves")
        ranks = tuple(value.rank for value in sorted(self.reserves, key=lambda value: value.rank))
        if ranks != tuple(range(len(self.reserves))):
            raise ValueError("reserve ranks must be unique and contiguous from zero")
        reserve_units = {value.physical_independent_unit_id for value in self.reserves}
        if len(reserve_units) != len(self.reserves) or reserve_units & set(
            self.primary_physical_unit_ids
        ):
            raise ValueError("reserve units repeat or overlap primary physical units")
        if not 0 <= self.maximum_substitutions <= len(self.reserves):
            raise ValueError("reserve substitution bound exceeds the reserve roster")


@dataclass(frozen=True, slots=True)
class LinkedCampaignPackageNode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/linked-campaign-package-node'

    node_id: str
    role: LinkedCampaignPackageRole
    authoring_package: ObjectIdentity
    parent_node_id: str | None
    issue_scope_id: str
    execution_scope_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    reveal_authority_required: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("node_id", self.node_id),
            ("issue_scope_id", self.issue_scope_id),
            ("execution_scope_id", self.execution_scope_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.parent_node_id is not None:
            validate_stable_id(self.parent_node_id, field_name="parent_node_id")
        if (
            self.authoring_package.object_schema
            != 'empirical-lawhood/planning/study-definition'
        ):
            raise ValueError("linked campaign node requires a StudyDefinition authoring package")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("linked campaign packages must remain prospective before reveal")
        if self.role is LinkedCampaignPackageRole.CONDITIONAL_CONTROLLER_USE_CHILD:
            if (
                self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
                or not self.reveal_authority_required
            ):
                raise ValueError("controller use child must be sealed and separately reveal-authorized")
        elif (
            self.outcome_access is not OutcomeAccess.OUTCOME_BLIND or self.reveal_authority_required
        ):
            raise ValueError("linked packages outside the controller-use child role remain outcome-blind without reveal authority")


@dataclass(frozen=True, slots=True)
class LinkedCampaignTopology(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/linked-campaign-topology'

    topology_id: str
    kind: LinkedCampaignTopologyKind
    nodes: tuple[LinkedCampaignPackageNode, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.topology_id, field_name="topology_id")
        require_sorted_unique_ids(self.nodes, attribute="node_id", field_name="nodes")
        by_role = {value.role: value for value in self.nodes}
        if set(by_role) != _PACKAGE_ROLES or len(self.nodes) != len(_PACKAGE_ROLES):
            raise ValueError("linked campaign topology requires exactly four package roles")
        excluded = by_role[LinkedCampaignPackageRole.EXCLUDED_QUALIFICATION]
        parent = by_role[LinkedCampaignPackageRole.PROTECTED_MEASUREMENT_THROUGH_ADMISSION_PARENT]
        route = by_role[LinkedCampaignPackageRole.CONDITIONAL_CONTROLLER_USE_ROUTE_QUALIFICATION]
        child = by_role[LinkedCampaignPackageRole.CONDITIONAL_CONTROLLER_USE_CHILD]
        if (
            excluded.parent_node_id is not None
            or parent.parent_node_id is not None
            or route.parent_node_id != parent.node_id
            or child.parent_node_id != route.node_id
        ):
            raise ValueError("linked campaign package lineage differs from ordinary topology")
        if len({value.issue_scope_id for value in self.nodes}) != len(self.nodes) or len(
            {value.execution_scope_id for value in self.nodes}
        ) != len(self.nodes):
            raise ValueError("linked packages require independent issue and execution scopes")


@dataclass(frozen=True, slots=True)
class LinkedCampaignProfile(CanonicalRecord):
    "One ordinary, outcome-blind config surface for a linked response experiment."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/linked-campaign-profile'

    profile_id: str
    profile_version: str
    source_profile: ObjectIdentity
    source_readiness: ObjectIdentity
    evidence_profile_selection: ObjectIdentity
    system: ObjectIdentity
    experiment: ObjectIdentity
    campaign: ObjectIdentity
    rosters: LinkedCampaignRosters
    projection_config: ObjectIdentity
    projection_truth_requirement: ObjectIdentity
    method_config: ObjectIdentity
    qualification_profile: ObjectIdentity
    law_coordinates: ObjectIdentity
    atlas_config: ObjectIdentity
    raw_admission_config: ObjectIdentity
    admission_config: ObjectIdentity
    reachability_config: ObjectIdentity
    measured_hold_config: ObjectIdentity
    controller_reducer_config: ObjectIdentity
    nested_controller_evaluation_config: ObjectIdentity
    capabilities: tuple[LinkedCampaignCapabilityBinding, ...]
    owners: tuple[LinkedCampaignOwnerBinding, ...]
    artifact_roles: tuple[LinkedCampaignArtifactBinding, ...]
    reserve_groups: tuple[CompleteUnitReserveGroup, ...]
    topology: LinkedCampaignTopology
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_semantic_version(self.profile_version)
        if self.evidence_profile_selection.object_schema != EvidenceProfileSelection.SCHEMA:
            raise ValueError("linked campaign requires an exact evidence-profile selection")
        require_sorted_unique_ids(
            self.capabilities,
            attribute="binding_id",
            field_name="capabilities",
        )
        if {value.role for value in self.capabilities} != _STAGE_ROLES or len(
            self.capabilities
        ) != len(_STAGE_ROLES):
            raise ValueError("linked campaign capability stage roster is incomplete")
        require_sorted_unique_ids(self.owners, attribute="binding_id", field_name="owners")
        if {value.role for value in self.owners} != _OWNER_ROLES or len(self.owners) != len(
            _OWNER_ROLES
        ):
            raise ValueError("linked campaign current-owner delegation is incomplete")
        require_sorted_unique_ids(
            self.artifact_roles,
            attribute="binding_id",
            field_name="artifact_roles",
        )
        if {value.role for value in self.artifact_roles} != _ARTIFACT_ROLES or len(
            self.artifact_roles
        ) != len(_ARTIFACT_ROLES):
            raise ValueError("linked campaign external artifact-role roster is incomplete")
        require_sorted_unique_ids(
            self.reserve_groups,
            attribute="group_id",
            field_name="reserve_groups",
        )
        primary_units = {
            unit for group in self.reserve_groups for unit in group.primary_physical_unit_ids
        }
        reserve_units = {
            reserve.physical_independent_unit_id
            for group in self.reserve_groups
            for reserve in group.reserves
        }
        if not primary_units.issubset(set(self.rosters.physical_independent_unit_ids)):
            raise ValueError("reserve group primary units lie outside the measurement-through-admission roster")
        if reserve_units & set(self.rosters.controller_use_independent_unit_ids):
            raise ValueError("measurement-through-admission reserves cannot borrow controller use evaluation units")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("linked campaign profile authoring must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("linked campaign profile must remain prospective")


__all__ = [
    'CompleteUnitReserveGroup',
    'LinkedCampaignArtifactBinding',
    "LinkedCampaignArtifactRole",
    'LinkedCampaignCapabilityBinding',
    'LinkedCampaignOwnerBinding',
    "LinkedCampaignOwnerRole",
    'LinkedCampaignPackageNode',
    "LinkedCampaignPackageRole",
    'LinkedCampaignProfile',
    'LinkedCampaignRosters',
    "LinkedCampaignStageRole",
    "LinkedCampaignTopologyKind",
    'LinkedCampaignTopology',
    'ReserveUnit',
]

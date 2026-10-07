"Prospective, action-free authoring records for observation-order experiments.\n\nThe carrier is a sibling of the parameterised extension from measurement through controller use.  It binds\nindependent units, acquisition groups and nested scientific views without\nintroducing a law, action, controller or later-rung placeholder.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.evidence_profiles import EvidenceProfileSelection
from empirical_lawhood.planning.prospective_config import ProspectiveTerminalMatrix
from empirical_lawhood.planning.source_pipelines import SourcePipelineProfile


MAX_OBSERVATION_ORDER_CONFIG_BYTES = 32 * 1024 * 1024


class ObservationOrderOwnerRole(StrEnum):
    SOURCE = "SOURCE"
    EVIDENCE_PROJECTION = "EVIDENCE_PROJECTION"
    METHOD = "METHOD"
    SEALED_EVALUATOR = "SEALED_EVALUATOR"


@dataclass(frozen=True, slots=True)
class ObservationPhysicalUnit(CanonicalRecord):
    """One independent prepared history; views never increase its count."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/observation-physical-unit'

    physical_independent_unit_id: str
    preparation_coordinate_id: str
    preparation_instance_id: str
    preparation_family_id: str
    preparation_sha256: str
    adapter_realization_id: str | None

    def __post_init__(self) -> None:
        for name in (
            "physical_independent_unit_id",
            "preparation_coordinate_id",
            "preparation_instance_id",
            "preparation_family_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.adapter_realization_id is not None:
            validate_stable_id(
                self.adapter_realization_id,
                field_name="adapter_realization_id",
            )
        validate_sha256(self.preparation_sha256, field_name="preparation_sha256")


@dataclass(frozen=True, slots=True)
class ObservationNestedView(CanonicalRecord):
    """One numerical view nested inside one source acquisition."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/observation-nested-view'

    view_id: str
    acquisition_group_id: str
    physical_independent_unit_id: str
    numerical_member_id: str
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    support_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "view_id",
            "acquisition_group_id",
            "physical_independent_unit_id",
            "numerical_member_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in ("receiver_ids", "clock_ids", "support_ids"):
            values = getattr(self, name)
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            for value in values:
                validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class ObservationAcquisitionGroup(CanonicalRecord):
    """One atomic source effect/object shared by nested views."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/observation-acquisition-group'

    acquisition_group_id: str
    physical_independent_unit_id: str
    preparation_instance_id: str
    view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "acquisition_group_id",
            "physical_independent_unit_id",
            "preparation_instance_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(self.view_ids, field_name="view_ids", allow_empty=False)
        for value in self.view_ids:
            validate_stable_id(value, field_name="view_ids")


@dataclass(frozen=True, slots=True)
class ObservationOrderOwnerBinding(CanonicalRecord):
    """Exact scientific owner and config selected for one observation role."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/observation-order-owner-binding'

    role: ObservationOrderOwnerRole
    owner: ObjectIdentity
    config: ObjectIdentity


@dataclass(frozen=True, slots=True)
class ObservationOrderExperimentExtension(CanonicalRecord):
    "Closed observation-order carrier with explicit inapplicability of action/law surfaces."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/observation-order-experiment-extension'

    extension_set_id: str
    experiment_id: str
    science_specification: ObjectIdentity
    evidence_profile_selection: ObjectIdentity
    source_pipeline_profile: ObjectIdentity
    physical_units: tuple[ObservationPhysicalUnit, ...]
    acquisition_groups: tuple[ObservationAcquisitionGroup, ...]
    nested_views: tuple[ObservationNestedView, ...]
    preparation_family_ids: tuple[str, ...]
    numerical_member_ids: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    clock_ids: tuple[str, ...]
    support_ids: tuple[str, ...]
    causal_cutoff_ids: tuple[str, ...]
    acquisition_task_prefix: str
    projection_task_prefix: str
    sealed_evaluation_task_id: str
    owners: tuple[ObservationOrderOwnerBinding, ...]
    terminal_matrix: ProspectiveTerminalMatrix
    maximum_evidence_ceiling: EvidenceCeiling
    action_or_action_lineage_applicable: bool
    development_evaluation_split_applicable: bool
    source_nomination_applicable: bool
    law_or_atlas_applicable: bool
    admission_or_controller_evaluation_reachability_applicable: bool
    controller_policy_follow_up_applicable: bool
    grants_authority: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.extension_set_id, field_name="extension_set_id")
        validate_stable_id(self.experiment_id, field_name="experiment_id")
        for name in (
            "acquisition_task_prefix",
            "projection_task_prefix",
            "sealed_evaluation_task_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.evidence_profile_selection.object_schema != EvidenceProfileSelection.SCHEMA:
            raise ValueError("observation carrier requires an evidence profile selection")
        if self.source_pipeline_profile.object_schema != SourcePipelineProfile.SCHEMA:
            raise ValueError("observation carrier requires a source pipeline profile")
        for name, values in (
            ("preparation_family_ids", self.preparation_family_ids),
            ("numerical_member_ids", self.numerical_member_ids),
            ("receiver_ids", self.receiver_ids),
            ("clock_ids", self.clock_ids),
            ("support_ids", self.support_ids),
            ("causal_cutoff_ids", self.causal_cutoff_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            for value in values:
                validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.physical_units,
            attribute="physical_independent_unit_id",
            field_name="physical_units",
        )
        require_sorted_unique_ids(
            self.acquisition_groups,
            attribute="acquisition_group_id",
            field_name="acquisition_groups",
        )
        require_sorted_unique_ids(
            self.nested_views,
            attribute="view_id",
            field_name="nested_views",
        )
        if not self.physical_units or not self.acquisition_groups or not self.nested_views:
            raise ValueError("observation carrier requires units, groups and views")
        unit_by_id = {value.physical_independent_unit_id: value for value in self.physical_units}
        group_by_unit = {
            value.physical_independent_unit_id: value for value in self.acquisition_groups
        }
        if len(group_by_unit) != len(self.acquisition_groups) or set(group_by_unit) != set(
            unit_by_id
        ):
            raise ValueError("observation carrier requires exactly one group per unit")
        if len({value.preparation_instance_id for value in self.physical_units}) != len(
            self.physical_units
        ):
            raise ValueError("observation carrier reuses a preparation instance")
        realization_ids = tuple(
            value.adapter_realization_id
            for value in self.physical_units
            if value.adapter_realization_id is not None
        )
        if len(set(realization_ids)) != len(realization_ids):
            raise ValueError("observation carrier reuses an adapter realization")
        views = {value.view_id: value for value in self.nested_views}
        declared: list[str] = []
        for unit_id, group in group_by_unit.items():
            unit = unit_by_id[unit_id]
            if group.preparation_instance_id != unit.preparation_instance_id:
                raise ValueError("observation group changes preparation lineage")
            for view_id in group.view_ids:
                view = views.get(view_id)
                if (
                    view is None
                    or view.acquisition_group_id != group.acquisition_group_id
                    or view.physical_independent_unit_id != unit_id
                ):
                    raise ValueError("observation group changes nested-view lineage")
                declared.append(view_id)
        if len(declared) != len(set(declared)) or set(declared) != set(views):
            raise ValueError("observation groups do not partition the view roster")
        if {value.preparation_family_id for value in self.physical_units} != set(
            self.preparation_family_ids
        ):
            raise ValueError("observation family roster differs from its units")
        if {value.numerical_member_id for value in self.nested_views} != set(
            self.numerical_member_ids
        ):
            raise ValueError("observation numerical-member roster differs from its views")
        for view in self.nested_views:
            if (
                not set(view.receiver_ids) <= set(self.receiver_ids)
                or not set(view.clock_ids) <= set(self.clock_ids)
                or not set(view.support_ids) <= set(self.support_ids)
            ):
                raise ValueError("observation view names an undeclared receiver, clock or support")
        require_sorted_unique_ids(self.owners, attribute="role", field_name="owners")
        if {value.role for value in self.owners} != set(ObservationOrderOwnerRole):
            raise ValueError("observation carrier requires the exact four scientific owners")
        if self.maximum_evidence_ceiling is not EvidenceCeiling.ORDER_RELATION:
            raise ValueError("observation carrier must stop at ORDER_RELATION")
        forbidden = (
            self.action_or_action_lineage_applicable,
            self.development_evaluation_split_applicable,
            self.source_nomination_applicable,
            self.law_or_atlas_applicable,
            self.admission_or_controller_evaluation_reachability_applicable,
            self.controller_policy_follow_up_applicable,
        )
        if any(forbidden):
            raise ValueError("observation carrier cannot activate excluded semantics")
        if (
            self.grants_authority
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("observation carrier must remain prospective and nonauthorizing")

    @property
    def config_identities(self) -> tuple[ObjectIdentity, ...]:
        return tuple(value.config for value in self.owners)


def decode_observation_observation_experiment_extension(
    payload: bytes,
) -> ObservationOrderExperimentExtension:
    return decode_canonical_bytes(
        payload,
        ObservationOrderExperimentExtension,
        maximum_bytes=MAX_OBSERVATION_ORDER_CONFIG_BYTES,
    )


__all__ = [
    "MAX_OBSERVATION_ORDER_CONFIG_BYTES",
    'ObservationAcquisitionGroup',
    'ObservationNestedView',
    'ObservationOrderOwnerBinding',
    'ObservationOrderOwnerRole',
    'ObservationPhysicalUnit',
    'ObservationOrderExperimentExtension',
    'decode_observation_observation_experiment_extension',
]

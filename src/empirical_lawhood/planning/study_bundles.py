"""Outcome-blind authoring contract for one parent and compiled child programmes.

The bundle is deliberately additive.  It identifies existing world-local
programme candidates and one joint descendant without widening either child
candidate or world-local adjudication schemas.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_schema,
    validate_stable_id,
)


class StudyComponentKind(StrEnum):
    """Closed scientific roles whose exact identities a bundle must retain."""

    SYSTEM = "SYSTEM"
    CAMPAIGN = "CAMPAIGN"
    PROTOCOL = "PROTOCOL"
    EVIDENCE_WORLD = "EVIDENCE_WORLD"
    SOURCE_MANIFEST = "SOURCE_MANIFEST"
    CAPABILITY = "CAPABILITY"
    LAW = "LAW"
    COMPILER_PROGRAMME = "COMPILER_PROGRAMME"
    MAPPING = "MAPPING"
    BASELINE = "BASELINE"
    ANALYSIS = "ANALYSIS"
    RESOURCE = "RESOURCE"
    AUTHORITY = "AUTHORITY"


@dataclass(frozen=True, slots=True)
class StudyComponentBinding(CanonicalRecord):
    """One content-identified component; never an executable selector."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/study-component-binding'

    binding_id: str
    kind: StudyComponentKind
    component: ObjectIdentity
    child_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        if self.child_id is not None:
            validate_stable_id(self.child_id, field_name="child_id")


@dataclass(frozen=True, slots=True)
class StudyBundleChild(CanonicalRecord):
    """Exact compiled child and its predeclared joint-result input slot."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/study-bundle-child'

    child_id: str
    campaign_node_id: str
    expected_system_id: str
    expected_world_id: str
    expected_experiment_id: str
    candidate: ObjectIdentity
    result_producer_node_id: str
    result_output_id: str
    result_input_id: str
    result_schema: str
    result_media_type: str

    def __post_init__(self) -> None:
        for name, value in (
            ("child_id", self.child_id),
            ("campaign_node_id", self.campaign_node_id),
            ("expected_system_id", self.expected_system_id),
            ("expected_world_id", self.expected_world_id),
            ("expected_experiment_id", self.expected_experiment_id),
            ("result_producer_node_id", self.result_producer_node_id),
            ("result_output_id", self.result_output_id),
            ("result_input_id", self.result_input_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.candidate.object_schema != 'empirical-lawhood/runtime/study-candidate':
            raise ValueError("programme bundle child must bind a standard programme candidate")
        validate_schema(self.result_schema)
        validate_nonempty(self.result_media_type, field_name="result_media_type")


@dataclass(frozen=True, slots=True)
class StudyJointDescendant(CanonicalRecord):
    """Precompiled fan-in contract; authority is intentionally still pending."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/study-joint-descendant'

    descendant_node_id: str
    parent_child_ids: tuple[str, ...]
    template: ObjectIdentity
    registry: ObjectIdentity
    resource_ceiling: ResourceBudget
    expected_authority_gate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.descendant_node_id, field_name="descendant_node_id")
        require_sorted_unique_strings(
            self.parent_child_ids,
            field_name="parent_child_ids",
            allow_empty=False,
        )
        if len(self.parent_child_ids) < 2:
            raise ValueError("joint descendant requires at least two child parents")
        if self.template.object_schema != 'empirical-lawhood/runtime/study-template':
            raise ValueError("joint descendant must bind a programme template")
        if self.registry.object_schema != 'empirical-lawhood/runtime/capability-registry':
            raise ValueError("joint descendant must bind a capability registry")
        require_sorted_unique_strings(
            self.expected_authority_gate_ids,
            field_name="expected_authority_gate_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class StudyBundleSpec(CanonicalRecord):
    """One declarative parent/two-or-more-child/joint-descendant package."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/study-bundle-spec'

    bundle_id: str
    campaign: ObjectIdentity
    children: tuple[StudyBundleChild, ...]
    joint_descendant: StudyJointDescendant
    component_bindings: tuple[StudyComponentBinding, ...]
    scientific_default_ids: tuple[str, ...]
    development_only: bool
    issued: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        if self.campaign.object_schema != 'empirical-lawhood/planning/campaign-spec':
            raise ValueError("programme bundle must bind a campaign specification")
        require_sorted_unique_ids(self.children, attribute="child_id", field_name="children")
        if len(self.children) < 2:
            raise ValueError("programme bundle requires at least two compiled children")
        child_ids = tuple(value.child_id for value in self.children)
        if self.joint_descendant.parent_child_ids != child_ids:
            raise ValueError("joint descendant parents differ from bundle children")
        if len({value.campaign_node_id for value in self.children}) != len(self.children):
            raise ValueError("programme bundle child campaign nodes must be unique")
        if len({value.expected_world_id for value in self.children}) != len(self.children):
            raise ValueError("programme bundle children must remain world-local")
        if len({value.result_input_id for value in self.children}) != len(self.children):
            raise ValueError("programme bundle child result slots must be unique")
        if len({value.result_output_id for value in self.children}) != len(self.children):
            raise ValueError("programme bundle child result outputs must be unique")
        require_sorted_unique_ids(
            self.component_bindings,
            attribute="binding_id",
            field_name="component_bindings",
        )
        bound_child_ids = {value.child_id for value in self.component_bindings}
        if not bound_child_ids.issubset({None, *child_ids}):
            raise ValueError("programme component names an unknown child")
        observed_kinds = {value.kind for value in self.component_bindings}
        missing = set(StudyComponentKind) - observed_kinds
        if missing:
            raise ValueError(
                "programme bundle omits required component roles: "
                + ",".join(sorted(value.value for value in missing))
            )
        require_sorted_unique_strings(
            self.scientific_default_ids,
            field_name="scientific_default_ids",
            allow_empty=False,
        )
        if not self.development_only or self.issued:
            raise ValueError("unissued development bundle cannot grant issue status")


__all__ = [
    'StudyBundleChild',
    'StudyBundleSpec',
    'StudyComponentBinding',
    'StudyComponentKind',
    'StudyJointDescendant',
]

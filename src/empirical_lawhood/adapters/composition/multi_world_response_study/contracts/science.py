"""Experiment payload ownership and synthetic parent conformance."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_stable_id,
)
from empirical_lawhood.planning.multi_world_study import MorphismNegativeControlKind
from empirical_lawhood.runtime.multi_world_study import PropertyMorphismDisposition


class MultiWorldStudyPayloadOwner(StrEnum):
    ARCHIVE = "ARCHIVE"
    GENERATED = "GENERATED"
    JOINT = "JOINT"
    MAPPED = "MAPPED"


_EXACT_NAMESPACE_OWNERS = {
    "archive-qualification": MultiWorldStudyPayloadOwner.ARCHIVE,
    "archive-to-simulator-map": MultiWorldStudyPayloadOwner.MAPPED,
    "law-to-action-acquisition": MultiWorldStudyPayloadOwner.GENERATED,
    "law-to-action-calibration": MultiWorldStudyPayloadOwner.GENERATED,
    "law-to-action-denominator-audit": MultiWorldStudyPayloadOwner.GENERATED,
    "law-to-action-domain": MultiWorldStudyPayloadOwner.GENERATED,
    "law-to-action-nested-prospective-controller-evaluation": MultiWorldStudyPayloadOwner.GENERATED,
    "law-to-action-reference": MultiWorldStudyPayloadOwner.GENERATED,
    "law-to-action-resource-envelope": MultiWorldStudyPayloadOwner.JOINT,
    "multi-world-parent": MultiWorldStudyPayloadOwner.JOINT,
    "property-morphism": MultiWorldStudyPayloadOwner.JOINT,
}

_EXACT_NODE_FAMILIES = (
    "archive-exposure-staging",
    "archive-late-reveal-evaluation",
    "archive-to-torax-mapping",
    "atlas-controller-admission-reachability",
    "common-and-adaptive-acquisition",
    "confirmatory-cells",
    "denominator-algebra-calibration",
    "final-arm-qualification",
    "joint-adjudication-report",
    "mapped-excluded-development",
    "programme-compile-binding-tick",
    "property-fan-in",
    "reference-cells-class",
    "resource-recovery",
    "roster-domain-freeze",
    "simulator-reveal-evaluation",
    "source-runtime-revalidation",
)


@dataclass(frozen=True, slots=True)
class MultiWorldStudyPayloadNamespaceBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/contracts/multi-world-study-payload-namespace-binding'

    namespace_id: str
    owner: MultiWorldStudyPayloadOwner
    payload_schema_ids: tuple[str, ...]
    shared_custody_capability: ObjectIdentity
    local_decision_service_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.namespace_id, field_name="namespace_id")
        if _EXACT_NAMESPACE_OWNERS.get(self.namespace_id) is not self.owner:
            raise ValueError("flagship payload namespace has the wrong child/parent owner")
        require_sorted_unique_strings(
            self.payload_schema_ids,
            field_name="payload_schema_ids",
            allow_empty=False,
        )
        for value in self.payload_schema_ids:
            validate_schema(value)
        if self.local_decision_service_allowed:
            raise ValueError("experiment payload namespace cannot own a shared decision service")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyScientificBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/contracts/multi-world-study-scientific-binding'

    binding_id: str
    g0_platform_binding: ObjectIdentity
    archive_law: ObjectIdentity
    mapped_scientific_config: ObjectIdentity
    generated_scientific_config: ObjectIdentity
    archive_protection_plan: ObjectIdentity
    outcome_barrier_plan: ObjectIdentity
    partial_morphism: ObjectIdentity
    joint_adjudication_plan: ObjectIdentity
    namespace_bindings: tuple[MultiWorldStudyPayloadNamespaceBinding, ...]
    archive_child_id: str
    mapped_child_id: str
    generated_child_id: str
    world_local_evidence_pooling_forbidden: bool
    simulator_result_can_promote_archive: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("archive_child_id", self.archive_child_id),
            ("mapped_child_id", self.mapped_child_id),
            ("generated_child_id", self.generated_child_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.namespace_bindings,
            attribute="namespace_id",
            field_name="namespace_bindings",
        )
        if {
            value.namespace_id: value.owner for value in self.namespace_bindings
        } != _EXACT_NAMESPACE_OWNERS:
            raise ValueError("flagship scientific extension namespace roster differs")
        expected_schemas = (
            (
                self.g0_platform_binding,
                'empirical-lawhood/composition/multi-world-response-study/contracts/multi-world-study-consumer-platform-binding',
            ),
            (self.archive_law, 'empirical-lawhood/physical/mast-archive-response-qualification/archive-law-spec'),
            (
                self.mapped_scientific_config,
                'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-child-scientific-config',
            ),
            (
                self.generated_scientific_config,
                'empirical-lawhood/simulators/tokamak-prospective-control/generated-child-scientific-config',
            ),
            (
                self.archive_protection_plan,
                'empirical-lawhood/planning/archive-outcome-protection-plan',
            ),
            (
                self.outcome_barrier_plan,
                'empirical-lawhood/planning/multi-world-outcome-barrier-plan',
            ),
            (
                self.partial_morphism,
                'empirical-lawhood/planning/archive-to-simulator-partial-morphism-spec',
            ),
            (
                self.joint_adjudication_plan,
                'empirical-lawhood/planning/multi-world-joint-adjudication-plan',
            ),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("flagship scientific binding changes an owning schema")
        if (
            len({self.archive_child_id, self.mapped_child_id, self.generated_child_id}) != 3
            or not self.world_local_evidence_pooling_forbidden
            or self.simulator_result_can_promote_archive
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("flagship scientific binding collapses world-local evidence")


@dataclass(frozen=True, slots=True)
class SyntheticMultiWorldStudyParentConformance(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/contracts/synthetic-multi-world-study-parent-conformance'

    receipt_id: str
    scientific_binding: ObjectIdentity
    archive_law: ObjectIdentity
    mapped_scientific_config: ObjectIdentity
    generated_scientific_config: ObjectIdentity
    generated_task_roster: ObjectIdentity
    preparation_plan_roster: ObjectIdentity
    contacted_node_family_ids: tuple[str, ...]
    child_ids: tuple[str, ...]
    observed_morphism_dispositions: tuple[PropertyMorphismDisposition, ...]
    morphism_negative_controls: tuple[MorphismNegativeControlKind, ...]
    reveal_order: tuple[str, ...]
    public_bundle_route: ObjectIdentity
    generated_task_count: int
    generated_task_arm_coordinate_count: int
    mapped_member_count: int
    shared_service_fork_count: int
    archived_semantic_import_count: int
    cross_child_evidence_pool_count: int
    native_archive_read_count: int
    native_simulator_launch_count: int
    simulator_only_nonpromotion_passed: bool
    crash_recovery_passed: bool
    catalog_rebuild_passed: bool
    terminal: bool
    architecture_conformance_only: bool
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(
            self.contacted_node_family_ids,
            field_name="contacted_node_family_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.child_ids, field_name="child_ids", allow_empty=False)
        if self.contacted_node_family_ids != _EXACT_NODE_FAMILIES:
            raise ValueError("synthetic parent did not contact every experiment node family")
        if len(self.child_ids) != 3:
            raise ValueError("synthetic parent does not retain three child identities")
        if set(self.observed_morphism_dispositions) != set(PropertyMorphismDisposition):
            raise ValueError("synthetic parent did not contact all four morphism dispositions")
        if self.morphism_negative_controls != tuple(MorphismNegativeControlKind):
            raise ValueError("synthetic parent did not contact all three morphism controls")
        if self.reveal_order != (
            "generated-torax-outcome",
            "mapped-torax-outcome",
            "archive-receiver-outcome",
        ):
            raise ValueError("synthetic parent does not reveal both simulators before archive")
        expected_schemas = (
            (self.archive_law, 'empirical-lawhood/physical/mast-archive-response-qualification/archive-law-spec'),
            (
                self.mapped_scientific_config,
                'empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-child-scientific-config',
            ),
            (
                self.generated_scientific_config,
                'empirical-lawhood/simulators/tokamak-prospective-control/generated-child-scientific-config',
            ),
            (
                self.generated_task_roster,
                'empirical-lawhood/composition/multi-world-response-study/synthetic-task-roster',
            ),
            (
                self.preparation_plan_roster,
                'empirical-lawhood/composition/multi-world-response-study/synthetic-preparation-roster',
            ),
        )
        if any(value.object_schema != schema for value, schema in expected_schemas):
            raise ValueError("synthetic parent binds another experiment contract")
        if (
            self.generated_task_count != 8
            or self.generated_task_arm_coordinate_count != 32
            or self.mapped_member_count != 6
            or self.shared_service_fork_count
            or self.archived_semantic_import_count
            or self.cross_child_evidence_pool_count
            or self.native_archive_read_count
            or self.native_simulator_launch_count
            or not self.simulator_only_nonpromotion_passed
            or not self.crash_recovery_passed
            or not self.catalog_rebuild_passed
            or not self.terminal
            or not self.architecture_conformance_only
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("synthetic parent overclaims or misses a hostile invariant")


def validate_study_scientific_binding(
    binding: MultiWorldStudyScientificBinding,
) -> None:
    """Explicit public validation seam used before candidate materialization."""

    binding.to_document()


__all__ = [
    'MultiWorldStudyPayloadNamespaceBinding',
    'MultiWorldStudyPayloadOwner',
    'MultiWorldStudyScientificBinding',
    'SyntheticMultiWorldStudyParentConformance',
    'validate_study_scientific_binding',
]

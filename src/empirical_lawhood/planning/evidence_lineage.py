"""Exact evidence lineage and derived dependence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.metatheory import EvidenceDependenceClass


class EvidenceSubstrateClass(StrEnum):
    ANALYTIC = "ANALYTIC"
    SIMULATOR = "SIMULATOR"
    HIL = "HIL"
    PHYSICAL = "PHYSICAL"


class EvidenceSharedComponentType(StrEnum):
    IMPLEMENTATION = "IMPLEMENTATION"
    EXECUTABLE_CAPABILITY = "EXECUTABLE_CAPABILITY"
    EXECUTABLE_PROVIDER = "EXECUTABLE_PROVIDER"
    LIBRARY_OR_RUNTIME = "LIBRARY_OR_RUNTIME"
    GENERATOR = "GENERATOR"
    ALGORITHM = "ALGORITHM"
    GENERATOR_IMPLEMENTATION = "GENERATOR_IMPLEMENTATION"
    CONFIGURATION = "CONFIGURATION"
    ONTOLOGY = "ONTOLOGY"
    SUBSTRATE_OR_SOURCE = "SUBSTRATE_OR_SOURCE"
    PREPARATION = "PREPARATION"
    SAMPLER = "SAMPLER"
    SOURCE_MATERIALIZATION = "SOURCE_MATERIALIZATION"
    RECEIVER = "RECEIVER"
    OBSERVATION_OPERATOR = "OBSERVATION_OPERATOR"
    NUMERICAL_VIEW = "NUMERICAL_VIEW"
    TRANSFORMATION = "TRANSFORMATION"
    ANALYSIS_METHOD = "ANALYSIS_METHOD"


@dataclass(frozen=True, slots=True)
class EvidenceImplementationLineage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-implementation-lineage'

    lineage_id: str
    implementation_closure: ObjectIdentity
    executable_capability: ObjectIdentity
    executable_provider: ObjectIdentity
    library_or_runtime_identities: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.lineage_id, field_name="lineage_id")
        require_sorted_unique_ids(
            self.library_or_runtime_identities,
            attribute="object_id",
            field_name="library_or_runtime_identities",
        )


@dataclass(frozen=True, slots=True)
class EvidenceGeneratorLineage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-generator-lineage'

    lineage_id: str
    generator_family: ObjectIdentity
    algorithm: ObjectIdentity
    implementation: ObjectIdentity
    configuration: ObjectIdentity
    shared_ontology_identities: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.lineage_id, field_name="lineage_id")
        require_sorted_unique_ids(
            self.shared_ontology_identities,
            attribute="object_id",
            field_name="shared_ontology_identities",
        )


@dataclass(frozen=True, slots=True)
class EvidencePreparationLineage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-preparation-lineage'

    lineage_id: str
    substrate_class: EvidenceSubstrateClass
    substrate_or_source_family: ObjectIdentity
    preparation_mechanism: ObjectIdentity
    sampler: ObjectIdentity
    roster: ObjectIdentity
    source_materialization_parents: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.lineage_id, field_name="lineage_id")
        require_sorted_unique_ids(
            self.source_materialization_parents,
            attribute="object_id",
            field_name="source_materialization_parents",
        )


@dataclass(frozen=True, slots=True)
class EvidenceObservationLineage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-observation-lineage'

    lineage_id: str
    receiver_construction: ObjectIdentity
    observation_operator: ObjectIdentity
    numerical_view: ObjectIdentity
    transformation: ObjectIdentity
    analysis_method: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.lineage_id, field_name="lineage_id")


@dataclass(frozen=True, slots=True)
class EvidenceLineageBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-lineage-bundle'

    bundle_id: str
    implementation: EvidenceImplementationLineage | None
    generator: EvidenceGeneratorLineage | None
    preparation: EvidencePreparationLineage | None
    observation: EvidenceObservationLineage | None
    unavailable_component_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        require_sorted_unique_strings(
            self.unavailable_component_reason_codes,
            field_name="unavailable_component_reason_codes",
        )
        complete = all(
            value is not None
            for value in (
                self.implementation,
                self.generator,
                self.preparation,
                self.observation,
            )
        )
        if complete == bool(self.unavailable_component_reason_codes):
            raise ValueError("lineage availability and reason codes disagree")


@dataclass(frozen=True, slots=True)
class EvidenceDependenceSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-dependence-spec'

    spec_id: str
    requested_class: EvidenceDependenceClass
    predecessor: EvidenceLineageBundle
    target: EvidenceLineageBundle
    stronger_class_refusal_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        if self.predecessor.bundle_id == self.target.bundle_id:
            raise ValueError("dependence comparison requires distinct lineage roles")


@dataclass(frozen=True, slots=True)
class EvidenceSharedComponent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-shared-component'

    component_id: str
    component_type: EvidenceSharedComponentType
    identity: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.component_id, field_name="component_id")


@dataclass(frozen=True, slots=True)
class EvidenceDependenceAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-dependence-assessment'

    assessment_id: str
    dependence_spec: ObjectIdentity
    predecessor_lineage: ObjectIdentity
    target_lineage: ObjectIdentity
    requested_class: EvidenceDependenceClass
    achieved_class: EvidenceDependenceClass | None
    shared_components: tuple[EvidenceSharedComponent, ...]
    refused_stronger_classes: tuple[EvidenceDependenceClass, ...]
    evidence_links: tuple[ObjectIdentity, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        if self.dependence_spec.object_schema != EvidenceDependenceSpec.SCHEMA:
            raise ValueError("dependence assessment binds another spec schema")
        if self.predecessor_lineage.object_schema != EvidenceLineageBundle.SCHEMA:
            raise ValueError("dependence assessment predecessor has another schema")
        if self.target_lineage.object_schema != EvidenceLineageBundle.SCHEMA:
            raise ValueError("dependence assessment target has another schema")
        require_sorted_unique_ids(
            self.shared_components,
            attribute="component_id",
            field_name="shared_components",
        )
        if (
            tuple(sorted(set(self.refused_stronger_classes), key=lambda value: value.value))
            != self.refused_stronger_classes
        ):
            raise ValueError("refused dependence classes must be sorted and unique")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.achieved_class is None and not self.reason_codes:
            raise ValueError("unresolved dependence requires reasons")


__all__ = [
    'EvidenceDependenceAssessment',
    'EvidenceDependenceSpec',
    'EvidenceGeneratorLineage',
    'EvidenceImplementationLineage',
    'EvidenceLineageBundle',
    'EvidenceObservationLineage',
    'EvidencePreparationLineage',
    'EvidenceSharedComponentType',
    'EvidenceSharedComponent',
    'EvidenceSubstrateClass',
]

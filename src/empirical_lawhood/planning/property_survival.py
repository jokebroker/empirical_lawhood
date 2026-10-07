"""Multi-property survival signatures over explicit partial maps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.contrasting_objectives import PartialMorphismSpec, PartialMorphismUncertaintyOperation
from empirical_lawhood.planning.decision_assurance import DecisionAssuranceResult
from empirical_lawhood.planning.evidence_lineage import EvidenceDependenceAssessment, EvidenceDependenceSpec
from empirical_lawhood.planning.metatheory import MetatheoryApplicability, MetatheoryCellDisposition, MetatheoryEvidenceCeiling, MetatheoryMethodSelection


PARTIAL_MORPHISM_ASSESSMENT_SCHEMA = 'empirical-lawhood/methods/partial-morphism-assessment'


class PropertySurvivalKind(StrEnum):
    REPRESENTATION = "REPRESENTATION"
    RESPONSE_QUANTITY = "RESPONSE_QUANTITY"
    CATEGORY = "CATEGORY"
    LABELLED_BOUNDARY = "LABELLED_BOUNDARY"
    FUTURE_DYNAMICS = "FUTURE_DYNAMICS"
    DECISION_OR_ADMISSION = "DECISION_OR_ADMISSION"
    COMPOSITION = "COMPOSITION"


class PropertySurvivalPath(StrEnum):
    DIRECT = "DIRECT"
    COMPOSED = "COMPOSED"


class PropertyPathConsistency(StrEnum):
    CONSISTENT = "CONSISTENT"
    OPPOSED = "OPPOSED"
    UNEVALUABLE = "UNEVALUABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True, slots=True)
class PropertyPathCorrespondence(CanonicalRecord):
    """Predeclared direct/composed pairing for one member/unit/face."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/property-path-correspondence'

    correspondence_id: str
    survival_spec: ObjectIdentity
    member_spec: ObjectIdentity
    physical_unit_id: str
    receiver_action_face_id: str
    direct_face_id: str
    direct_domain_cell_id: str
    composed_face_id: str | None
    composed_domain_cell_id: str | None
    applicability: MetatheoryApplicability
    reason_codes: tuple[str, ...]
    target_outcomes_read: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "correspondence_id",
            "physical_unit_id",
            "receiver_action_face_id",
            "direct_face_id",
            "direct_domain_cell_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.survival_spec.object_schema != PropertySurvivalSpec.SCHEMA:
            raise ValueError("property path correspondence names another survival schema")
        if self.member_spec.object_schema != PropertySurvivalMemberSpec.SCHEMA:
            raise ValueError("property path correspondence names another member schema")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.target_outcomes_read or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("property path correspondence must be frozen outcome-blind")
        if self.applicability is MetatheoryApplicability.REQUIRED:
            if (
                self.composed_face_id is None
                or self.composed_domain_cell_id is None
                or self.reason_codes
            ):
                raise ValueError("required property path correspondence lacks a composed operand")
            validate_stable_id(self.composed_face_id, field_name="composed_face_id")
            validate_stable_id(
                self.composed_domain_cell_id,
                field_name="composed_domain_cell_id",
            )
        elif (
            self.composed_face_id is not None
            or self.composed_domain_cell_id is not None
            or not self.reason_codes
        ):
            raise ValueError("N/A property path correspondence invented a composed operand")


@dataclass(frozen=True, slots=True)
class PropertyPathFaceConsistencyCell(CanonicalRecord):
    """One preserved facewise direct/composed comparison."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/property-path-face-consistency-cell'

    cell_id: str
    correspondence: ObjectIdentity
    member_id: str
    physical_unit_id: str
    receiver_action_face_id: str
    direct_evidence: ObjectIdentity
    composed_evidence: ObjectIdentity | None
    disposition: PropertyPathConsistency
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "member_id",
            "physical_unit_id",
            "receiver_action_face_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.correspondence.object_schema != PropertyPathCorrespondence.SCHEMA:
            raise ValueError("property path cell names another correspondence schema")
        if self.direct_evidence.object_schema != PropertySurvivalCellEvidence.SCHEMA:
            raise ValueError("property path cell names another direct evidence schema")
        if self.composed_evidence is not None and (
            self.composed_evidence.object_schema != PropertySurvivalCellEvidence.SCHEMA
        ):
            raise ValueError("property path cell names another composed evidence schema")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is PropertyPathConsistency.CONSISTENT:
            if self.reason_codes or self.composed_evidence is None:
                raise ValueError("consistent property face lacks paired evidence")
        elif self.disposition is PropertyPathConsistency.NOT_APPLICABLE:
            if self.composed_evidence is not None or not self.reason_codes:
                raise ValueError("N/A property face carries a composed operand")
        elif not self.reason_codes or self.composed_evidence is None:
            raise ValueError("non-consistent property face lacks reasons or paired evidence")


@dataclass(frozen=True, slots=True)
class PropertyPathConsistencyCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/property-path-consistency-cell'

    member_id: str
    disposition: PropertyPathConsistency
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.member_id, field_name="member_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is PropertyPathConsistency.CONSISTENT:
            if self.reason_codes:
                raise ValueError("consistent property paths cannot carry failure reasons")
        elif not self.reason_codes:
            raise ValueError("non-consistent property paths require reasons")


@dataclass(frozen=True, slots=True)
class PropertySurvivalMemberSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/property-survival-member-spec'

    member_id: str
    survival_kind: PropertySurvivalKind
    partial_morphism_spec: ObjectIdentity
    property_id: str
    property_metric_id: str
    resolution: NamedDecimal
    uncertainty_operation: PartialMorphismUncertaintyOperation
    uncertainty_operation_id: str
    support: ObjectIdentity
    direct_face_ids: tuple[str, ...]
    composed_face_ids: tuple[str, ...]
    falsifier_ids: tuple[str, ...]
    applicability: MetatheoryApplicability
    applicability_reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "member_id",
            "property_id",
            "property_metric_id",
            "uncertainty_operation_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.partial_morphism_spec.object_schema != PartialMorphismSpec.SCHEMA:
            raise ValueError("property member requires an exact partial morphism spec")
        for name, values in (
            ("direct_face_ids", self.direct_face_ids),
            ("composed_face_ids", self.composed_face_ids),
            ("falsifier_ids", self.falsifier_ids),
            ("applicability_reason_codes", self.applicability_reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.applicability is MetatheoryApplicability.REQUIRED:
            if (
                not self.direct_face_ids
                or not self.falsifier_ids
                or self.applicability_reason_codes
            ):
                raise ValueError("required property member lacks direct faces/falsifiers")
        elif self.direct_face_ids or self.composed_face_ids or not self.applicability_reason_codes:
            raise ValueError("N/A property member must remain nonexecuting")


@dataclass(frozen=True, slots=True)
class PropertySurvivalSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/property-survival-spec'

    spec_id: str
    transformation: ObjectIdentity
    context: ObjectIdentity
    source_evidence_world_profile: ObjectIdentity
    target_evidence_world_profile: ObjectIdentity
    dependence_spec: ObjectIdentity
    physical_unit_ids: tuple[str, ...]
    members: tuple[PropertySurvivalMemberSpec, ...]
    required_face_ids: tuple[str, ...]
    aggregation_rule_id: str
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        if self.dependence_spec.object_schema != EvidenceDependenceSpec.SCHEMA:
            raise ValueError("property survival requires a dependence spec")
        require_sorted_unique_strings(
            self.physical_unit_ids,
            field_name="physical_unit_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.members, attribute="member_id", field_name="members")
        if {value.survival_kind for value in self.members} != set(PropertySurvivalKind):
            raise ValueError("property survival requires the seven-member signature")
        require_sorted_unique_strings(
            self.required_face_ids,
            field_name="required_face_ids",
            allow_empty=False,
        )
        declared_faces = {
            face
            for member in self.members
            for face in (*member.direct_face_ids, *member.composed_face_ids)
        }
        if declared_faces != set(self.required_face_ids):
            raise ValueError("property survival required faces differ from member faces")
        validate_stable_id(self.aggregation_rule_id, field_name="aggregation_rule_id")


@dataclass(frozen=True, slots=True)
class PropertySurvivalCellEvidence(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/property-survival-cell-evidence'

    evidence_id: str
    survival_spec: ObjectIdentity
    member_spec: ObjectIdentity
    physical_unit_id: str
    face_id: str
    domain_cell_id: str
    path: PropertySurvivalPath
    partial_morphism_assessment: ObjectIdentity
    source_observation: ObjectIdentity
    target_observation: ObjectIdentity
    defect: NamedDecimal
    margin: NamedDecimal
    uncertainty: NamedDecimal
    property_preserved: bool | None
    morphism_cell_disposition: MetatheoryCellDisposition
    decision_assurance: ObjectIdentity | None
    method: MetatheoryMethodSelection
    evidence_links: tuple[ObjectIdentity, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        if self.survival_spec.object_schema != PropertySurvivalSpec.SCHEMA:
            raise ValueError("property evidence names another survival schema")
        if self.member_spec.object_schema != PropertySurvivalMemberSpec.SCHEMA:
            raise ValueError("property evidence names another member schema")
        for name in ("physical_unit_id", "face_id", "domain_cell_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.partial_morphism_assessment.object_schema != PARTIAL_MORPHISM_ASSESSMENT_SCHEMA:
            raise ValueError("property evidence requires a morphism assessment")
        if self.morphism_cell_disposition is MetatheoryCellDisposition.NOT_APPLICABLE:
            raise ValueError("executed property evidence cannot be N/A")
        if self.property_preserved is None:
            if self.morphism_cell_disposition is not MetatheoryCellDisposition.UNEVALUABLE:
                raise ValueError("unknown property verdict must remain unevaluable")
        elif self.property_preserved != (
            self.morphism_cell_disposition is MetatheoryCellDisposition.SUPPORTED
        ):
            raise ValueError("property verdict and morphism disposition differ")
        if self.decision_assurance is not None and (
            self.decision_assurance.object_schema != DecisionAssuranceResult.SCHEMA
        ):
            raise ValueError("property evidence names another decision assurance schema")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )


@dataclass(frozen=True, slots=True)
class PropertySurvivalCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/property-survival-cell'

    cell_id: str
    member_id: str
    physical_unit_id: str
    face_id: str
    domain_cell_id: str
    path: PropertySurvivalPath
    evidence: ObjectIdentity
    disposition: MetatheoryCellDisposition
    decisive_falsifier_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "member_id",
            "physical_unit_id",
            "face_id",
            "domain_cell_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.decisive_falsifier_ids,
            field_name="decisive_falsifier_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")


@dataclass(frozen=True, slots=True)
class PropertySurvivalSignature(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/property-survival-signature'

    signature_id: str
    supported_member_ids: tuple[str, ...]
    opposed_member_ids: tuple[str, ...]
    unevaluable_member_ids: tuple[str, ...]
    not_applicable_member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.signature_id, field_name="signature_id")
        groups = (
            self.supported_member_ids,
            self.opposed_member_ids,
            self.unevaluable_member_ids,
            self.not_applicable_member_ids,
        )
        for name, values in zip(
            (
                "supported_member_ids",
                "opposed_member_ids",
                "unevaluable_member_ids",
                "not_applicable_member_ids",
            ),
            groups,
            strict=True,
        ):
            require_sorted_unique_strings(values, field_name=name)
        flattened = tuple(value for group in groups for value in group)
        if len(flattened) != len(set(flattened)):
            raise ValueError("property signature member groups overlap")


@dataclass(frozen=True, slots=True)
class PropertySurvivalAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/property-survival-assessment'

    assessment_id: str
    survival_spec: ObjectIdentity
    morphism_assessments: tuple[ObjectIdentity, ...]
    cell_evidence: tuple[PropertySurvivalCellEvidence, ...]
    cells: tuple[PropertySurvivalCell, ...]
    signature: PropertySurvivalSignature
    path_consistency: tuple[PropertyPathConsistencyCell, ...]
    dependence_assessment: EvidenceDependenceAssessment
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    reason_codes: tuple[str, ...]
    response_law_produced: bool
    coefficient_or_sample_pooling_performed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        require_sorted_unique_ids(
            self.morphism_assessments,
            attribute="object_id",
            field_name="morphism_assessments",
        )
        require_sorted_unique_ids(
            self.cell_evidence,
            attribute="evidence_id",
            field_name="cell_evidence",
        )
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if len(self.cell_evidence) != len(self.cells):
            raise ValueError("property assessment does not preserve every evidence cell")
        require_sorted_unique_ids(
            self.path_consistency,
            attribute="member_id",
            field_name="path_consistency",
        )
        if (
            self.dependence_assessment.dependence_spec.object_schema
            != EvidenceDependenceSpec.SCHEMA
        ):
            raise ValueError("property assessment lacks dependence assessment")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.response_law_produced or self.coefficient_or_sample_pooling_performed:
            raise ValueError("property survival cannot produce a law or pool samples")


@dataclass(frozen=True, slots=True)
class FacePairedPropertySurvivalAssessment(CanonicalRecord):
    "Face-paired property-survival assessment."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/face-paired-property-survival-assessment'

    assessment_id: str
    property_assessment: PropertySurvivalAssessment
    path_correspondences: tuple[PropertyPathCorrespondence, ...]
    path_face_consistency: tuple[PropertyPathFaceConsistencyCell, ...]
    signature: PropertySurvivalSignature
    path_consistency: tuple[PropertyPathConsistencyCell, ...]
    reason_codes: tuple[str, ...]
    response_law_produced: bool
    coefficient_or_sample_pooling_performed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        require_sorted_unique_ids(
            self.path_correspondences,
            attribute="correspondence_id",
            field_name="path_correspondences",
        )
        require_sorted_unique_ids(
            self.path_face_consistency,
            attribute="cell_id",
            field_name="path_face_consistency",
        )
        if len(self.path_correspondences) != len(self.path_face_consistency):
            raise ValueError("property assessment does not preserve every correspondence")
        require_sorted_unique_ids(
            self.path_consistency,
            attribute="member_id",
            field_name="path_consistency",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.response_law_produced or self.coefficient_or_sample_pooling_performed:
            raise ValueError("property survival cannot produce a law or pool samples")


__all__ = [
    "PARTIAL_MORPHISM_ASSESSMENT_SCHEMA",
    'PropertyPathCorrespondence',
    'PropertyPathFaceConsistencyCell',
    'PropertyPathConsistencyCell',
    'PropertyPathConsistency',
    'PropertySurvivalAssessment',
    'FacePairedPropertySurvivalAssessment',
    'PropertySurvivalCellEvidence',
    'PropertySurvivalCell',
    'PropertySurvivalKind',
    'PropertySurvivalMemberSpec',
    'PropertySurvivalPath',
    'PropertySurvivalSignature',
    'PropertySurvivalSpec',
]

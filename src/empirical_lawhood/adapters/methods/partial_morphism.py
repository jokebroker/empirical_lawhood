"""Generic evaluator for explicitly declared property-specific partial maps."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.contrasting_objectives import PartialMorphismDirection, PartialMorphismSpec, PartialMorphismUncertaintyOperation


class PartialMorphismCellDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    UNSUPPORTED = "UNSUPPORTED"
    UNDEFINED_DOMAIN = "UNDEFINED_DOMAIN"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class PartialMorphismCellEvidence(CanonicalRecord):
    """Exact already-produced evidence for one proposed domain cell."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/partial-morphism-cell-evidence'

    evidence_id: str
    domain_cell_id: str
    source_evidence_world_profile_id: str
    target_evidence_world_profile_id: str
    property_id: str
    direction: PartialMorphismDirection
    property_map_id: str
    excluded_field_ids: tuple[str, ...]
    uncertainty_operation: PartialMorphismUncertaintyOperation
    uncertainty_operation_id: str
    source_evidence: ObjectIdentity
    target_evidence: ObjectIdentity
    property_preserved: bool | None
    evidence_complete: bool
    coefficient_or_sample_pooling_requested: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("evidence_id", self.evidence_id),
            ("domain_cell_id", self.domain_cell_id),
            ("source_evidence_world_profile_id", self.source_evidence_world_profile_id),
            ("target_evidence_world_profile_id", self.target_evidence_world_profile_id),
            ("property_id", self.property_id),
            ("property_map_id", self.property_map_id),
            ("uncertainty_operation_id", self.uncertainty_operation_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.excluded_field_ids,
            field_name="excluded_field_ids",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.evidence_complete != (self.property_preserved is not None):
            raise ValueError("morphism evidence completeness and property verdict differ")
        if self.coefficient_or_sample_pooling_requested:
            raise ValueError("partial morphism evidence cannot request coefficient/sample pooling")


@dataclass(frozen=True, slots=True)
class PartialMorphismAssessmentCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/partial-morphism-assessment-cell'

    cell_id: str
    domain_cell_id: str
    property_id: str
    disposition: PartialMorphismCellDisposition
    evidence_links: tuple[ObjectIdentity, ...]
    decisive_falsifier_id: str | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        validate_stable_id(self.domain_cell_id, field_name="domain_cell_id")
        validate_stable_id(self.property_id, field_name="property_id")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if self.decisive_falsifier_id is not None:
            validate_stable_id(
                self.decisive_falsifier_id,
                field_name="decisive_falsifier_id",
            )
        if self.disposition is PartialMorphismCellDisposition.UNSUPPORTED:
            if self.decisive_falsifier_id is None:
                raise ValueError("unsupported morphism cell requires a decisive falsifier")
        elif self.decisive_falsifier_id is not None:
            raise ValueError("only an unsupported morphism cell may name a falsifier")


@dataclass(frozen=True, slots=True)
class PartialMorphismAssessment(CanonicalRecord):
    "Objective-specific terminal product; never a ResponseLaw or measurement through controller-use result."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/partial-morphism-assessment'

    assessment_id: str
    specification: ObjectIdentity
    cells: tuple[PartialMorphismAssessmentCell, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    response_law_produced: bool
    observation_to_controller_use_rungs_applicable: bool
    coefficient_or_sample_pooling_performed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        if self.specification.object_schema != PartialMorphismSpec.SCHEMA:
            raise ValueError("partial morphism assessment binds another specification")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if not self.cells:
            raise ValueError("partial morphism assessment requires a cell")
        if self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("partial morphism assessment is non-promotable")
        if self.response_law_produced or self.observation_to_controller_use_rungs_applicable:
            raise ValueError("partial morphism assessment cannot enter the measurement through controller use law route")
        if self.coefficient_or_sample_pooling_performed:
            raise ValueError("partial morphism assessment cannot pool coefficients or samples")


@dataclass(frozen=True, slots=True)
class PartialMorphismEvaluatorRegistration(CanonicalRecord):
    """Static method registration for one generic evaluator implementation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/partial-morphism-evaluator-registration'

    registration_id: str
    method_key: str
    method_version: str
    implementation_sha256: str
    input_schema_ids: tuple[str, ...]
    output_schema_id: str
    maximum_evidence_ceiling: EvidenceCeiling
    coefficient_or_sample_pooling_permitted: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.registration_id, field_name="registration_id")
        validate_stable_id(self.method_key, field_name="method_key")
        validate_semantic_version(self.method_version)
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        expected_inputs = tuple(
            sorted((PartialMorphismSpec.SCHEMA, PartialMorphismCellEvidence.SCHEMA))
        )
        if self.input_schema_ids != expected_inputs:
            raise ValueError("partial morphism evaluator input schemas differ")
        if self.output_schema_id != PartialMorphismAssessment.SCHEMA:
            raise ValueError("partial morphism evaluator output schema differs")
        if self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("partial morphism evaluator is non-promotable")
        if self.coefficient_or_sample_pooling_permitted:
            raise ValueError("partial morphism evaluator cannot permit pooling")


def partial_morphism_evaluator_registration(
    *, implementation_sha256: str
) -> PartialMorphismEvaluatorRegistration:
    return PartialMorphismEvaluatorRegistration(
        registration_id="registration.partial-morphism-evaluator",
        method_key="partial-morphism.property-map",
        method_version="1.0.0",
        implementation_sha256=implementation_sha256,
        input_schema_ids=tuple(
            sorted((PartialMorphismSpec.SCHEMA, PartialMorphismCellEvidence.SCHEMA))
        ),
        output_schema_id=PartialMorphismAssessment.SCHEMA,
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        coefficient_or_sample_pooling_permitted=False,
    )


def assess_partial_morphism(
    *,
    assessment_id: str,
    specification: PartialMorphismSpec,
    evidence: tuple[PartialMorphismCellEvidence, ...],
) -> PartialMorphismAssessment:
    """Evaluate only the exact declared property, direction and map semantics."""

    require_sorted_unique_ids(evidence, attribute="evidence_id", field_name="evidence")
    if not evidence:
        raise ValueError("partial morphism assessment requires evidence")
    domain_ids = {value.cell_id for value in specification.domain}
    expected_exclusions = tuple(value.field_id for value in specification.excluded_fields)
    cells = []
    for value in evidence:
        if (
            value.source_evidence_world_profile_id != specification.source_evidence_world_profile_id
            or value.target_evidence_world_profile_id
            != specification.target_evidence_world_profile_id
        ):
            raise ValueError("partial morphism evidence reverses or changes its endpoints")
        if value.property_id != specification.property_id:
            raise ValueError("partial morphism evidence names another property")
        if value.direction is not specification.direction:
            raise ValueError("partial morphism evidence reverses its declared direction")
        if value.property_map_id != specification.property_map.map_id:
            raise ValueError("partial morphism evidence uses another property map")
        if value.excluded_field_ids != expected_exclusions:
            raise ValueError("partial morphism evidence hides or adds excluded fields")
        if (
            value.uncertainty_operation is not specification.uncertainty_operation
            or value.uncertainty_operation_id != specification.uncertainty_operation_id
        ):
            raise ValueError("partial morphism evidence changes its uncertainty operation")

        evidence_links = tuple(
            sorted(
                (value.source_evidence, value.target_evidence),
                key=lambda identity: identity.object_id,
            )
        )
        reasons = set(value.reason_codes)
        falsifier = None
        if value.domain_cell_id not in domain_ids:
            disposition = PartialMorphismCellDisposition.UNDEFINED_DOMAIN
            reasons.add("PROPERTY_MAP_UNDEFINED_ON_DOMAIN_CELL")
        elif not value.evidence_complete:
            disposition = PartialMorphismCellDisposition.UNEVALUABLE
            reasons.add("PROPERTY_MAP_EVIDENCE_UNEVALUABLE")
        elif value.property_preserved:
            disposition = PartialMorphismCellDisposition.SUPPORTED
            reasons.add("PROPERTY_PRESERVED_ON_DECLARED_DOMAIN")
        else:
            disposition = PartialMorphismCellDisposition.UNSUPPORTED
            falsifier = specification.falsifier_ids[0]
            reasons.add("PROPERTY_PRESERVATION_FALSIFIED")
        cells.append(
            PartialMorphismAssessmentCell(
                cell_id=f"partial-morphism-cell.{assessment_id}.{value.evidence_id}",
                domain_cell_id=value.domain_cell_id,
                property_id=value.property_id,
                disposition=disposition,
                evidence_links=evidence_links,
                decisive_falsifier_id=falsifier,
                reason_codes=tuple(sorted(reasons)),
            )
        )
    return PartialMorphismAssessment(
        assessment_id=assessment_id,
        specification=ObjectIdentity.from_record(specification.spec_id, specification),
        cells=tuple(sorted(cells, key=lambda cell: cell.cell_id)),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        response_law_produced=False,
        observation_to_controller_use_rungs_applicable=False,
        coefficient_or_sample_pooling_performed=False,
    )


__all__ = [
    'PartialMorphismAssessmentCell',
    'PartialMorphismAssessment',
    'PartialMorphismCellDisposition',
    'PartialMorphismCellEvidence',
    'PartialMorphismEvaluatorRegistration',
    "assess_partial_morphism",
    'partial_morphism_evaluator_registration',
]

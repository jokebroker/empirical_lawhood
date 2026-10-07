"""Explicit model/world relations, discrepancy results and robust model sets."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from .evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
    inherited_visibility,
)
from .obligations import DiscrepancySpec, ObligationStatus, ValiditySpec
from .provenance import EvidenceLink, ObjectIdentity
from .references import ExecutableReference, NamedDecimal
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)
from .time import ClockRelationSpec


@dataclass(frozen=True, slots=True)
class QuantityAlignment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/quantity-alignment'

    alignment_id: str
    source_quantity_id: str
    target_quantity_id: str
    source_native_unit: str
    target_native_unit: str
    conversion_reference_id: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("alignment_id", self.alignment_id),
            ("source_quantity_id", self.source_quantity_id),
            ("target_quantity_id", self.target_quantity_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.source_native_unit, field_name="source_native_unit")
        validate_nonempty(self.target_native_unit, field_name="target_native_unit")
        if self.source_native_unit != self.target_native_unit:
            if self.conversion_reference_id is None:
                raise ValueError("unlike native units require an explicit conversion")
        elif self.conversion_reference_id is not None:
            raise ValueError("identical native units do not require a conversion")
        if self.conversion_reference_id is not None:
            validate_stable_id(self.conversion_reference_id, field_name="conversion_reference_id")


@dataclass(frozen=True, slots=True)
class ModelRelation(CanonicalRecord):
    """Directional, discrepancy-aware relation between evidence worlds."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/model-relation'

    model_relation_id: str
    source_world_id: str
    target_world_id: str
    source_numerical_view_ids: tuple[str, ...]
    target_numerical_view_ids: tuple[str, ...]
    quantity_alignments: tuple[QuantityAlignment, ...]
    clock_relations: tuple[ClockRelationSpec, ...]
    observation_operator: ExecutableReference
    calibration_unit_ids: tuple[str, ...]
    held_out_validation_unit_ids: tuple[str, ...]
    discrepancy: DiscrepancySpec
    validity: ValiditySpec
    maximum_evidence: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    evidence_links: tuple[EvidenceLink, ...]
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("model_relation_id", self.model_relation_id),
            ("source_world_id", self.source_world_id),
            ("target_world_id", self.target_world_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.source_world_id == self.target_world_id:
            raise ValueError("ModelRelation requires distinct evidence worlds")
        for field_name, values in (
            ("source_numerical_view_ids", self.source_numerical_view_ids),
            ("target_numerical_view_ids", self.target_numerical_view_ids),
            ("calibration_unit_ids", self.calibration_unit_ids),
            ("held_out_validation_unit_ids", self.held_out_validation_unit_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name)
        require_sorted_unique_ids(
            self.quantity_alignments,
            attribute="alignment_id",
            field_name="quantity_alignments",
        )
        if not self.quantity_alignments:
            raise ValueError("ModelRelation requires aligned quantities")
        require_sorted_unique_ids(
            self.clock_relations,
            attribute="relation_id",
            field_name="clock_relations",
        )
        if not self.clock_relations:
            raise ValueError("ModelRelation requires aligned clocks")
        if set(self.calibration_unit_ids) & set(self.held_out_validation_unit_ids):
            raise ValueError("calibration and held-out model-relation units overlap")
        self._validate_discrepancy()
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("model-relation visibility cannot be lowered")
        require_extensions(self.extensions)

    def _validate_discrepancy(self) -> None:
        if self.discrepancy.source_world_id != self.source_world_id:
            raise ValueError("discrepancy source world differs from ModelRelation")
        if self.discrepancy.target_world_id != self.target_world_id:
            raise ValueError("discrepancy target world differs from ModelRelation")
        if self.discrepancy.calibration_unit_ids != self.calibration_unit_ids:
            raise ValueError("discrepancy calibration units differ from ModelRelation")
        if self.discrepancy.held_out_validation_unit_ids != self.held_out_validation_unit_ids:
            raise ValueError("discrepancy validation units differ from ModelRelation")
        aligned_quantity_ids = tuple(
            sorted(alignment.source_quantity_id for alignment in self.quantity_alignments)
        )
        if self.discrepancy.aligned_quantity_ids != aligned_quantity_ids:
            raise ValueError("discrepancy quantities differ from ModelRelation")


class TransportStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    MIXED = "MIXED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class DiscrepancyBound(CanonicalRecord):
    """Directional calibration-only bias and model-form deviation bound."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/discrepancy-bound'

    bound_id: str
    alignment_id: str
    source_quantity_id: str
    target_quantity_id: str
    native_unit: str
    directional_bias: Decimal
    absolute_deviation_bound: Decimal
    calibration_unit_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("bound_id", self.bound_id),
            ("alignment_id", self.alignment_id),
            ("source_quantity_id", self.source_quantity_id),
            ("target_quantity_id", self.target_quantity_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_decimal(self.directional_bias, field_name="directional_bias")
        validate_decimal(
            self.absolute_deviation_bound,
            field_name="absolute_deviation_bound",
            minimum=Decimal(0),
        )
        require_sorted_unique_strings(
            self.calibration_unit_ids,
            field_name="calibration_unit_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class DirectionalDiscrepancyEstimate(CanonicalRecord):
    """Auditable discrepancy estimate with held-out qualification kept separate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/directional-discrepancy-estimate'

    estimate_id: str
    dataset: ObjectIdentity
    config: ObjectIdentity
    discrepancy: DiscrepancySpec
    bounds: tuple[DiscrepancyBound, ...]
    held_out_metrics: tuple[NamedDecimal, ...]
    status: ObligationStatus
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.estimate_id, field_name="estimate_id")
        require_sorted_unique_ids(self.bounds, attribute="bound_id", field_name="bounds")
        if not self.bounds:
            raise ValueError("directional discrepancy estimate requires calibrated bounds")
        if tuple(bound.bound_id for bound in self.bounds) != self.discrepancy.discrepancy_bound_ids:
            raise ValueError("discrepancy estimate bounds differ from its specification")
        if any(
            bound.calibration_unit_ids != self.discrepancy.calibration_unit_ids
            for bound in self.bounds
        ):
            raise ValueError("discrepancy bounds use different calibration identities")
        require_sorted_unique_ids(
            self.held_out_metrics,
            attribute="value_id",
            field_name="held_out_metrics",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if self.status is not self.discrepancy.status:
            raise ValueError("discrepancy estimate and specification statuses differ")
        if self.status is ObligationStatus.SATISFIED:
            if self.discrepancy.failed_test_ids or not self.evidence_links:
                raise ValueError("satisfied discrepancy estimate cannot retain failed tests")
        elif not self.discrepancy.failed_test_ids:
            raise ValueError("non-satisfied discrepancy estimate requires failed tests")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("discrepancy-estimate visibility cannot be lowered")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class TransportResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/transport-result'

    transport_result_id: str
    model_relation: ObjectIdentity
    status: TransportStatus
    metrics: tuple[NamedDecimal, ...]
    failed_test_ids: tuple[str, ...]
    achieved_evidence_ceiling: EvidenceCeiling
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess
    parent_visibility_ceiling: VisibilityCeiling
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.transport_result_id, field_name="transport_result_id")
        require_sorted_unique_ids(self.metrics, attribute="value_id", field_name="metrics")
        require_sorted_unique_strings(self.failed_test_ids, field_name="failed_test_ids")
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        if self.status is TransportStatus.SUPPORTED:
            if self.failed_test_ids or not self.evidence_links:
                raise ValueError("supported transport cannot retain failed tests")
        elif not self.failed_test_ids:
            raise ValueError("non-supported transport requires failed/limiting tests")
        inherited = inherited_visibility((self.parent_visibility_ceiling,), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("transport-result visibility cannot be lowered")
        require_extensions(self.extensions)


class ModelIntersectionSemantics(StrEnum):
    QUALIFIED_INTERSECTION = "QUALIFIED_INTERSECTION"


@dataclass(frozen=True, slots=True)
class ViewModelSetSpec(CanonicalRecord):
    """Predeclared plausible model plurality used by robust downstream work."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/view-model-set-spec'

    model_set_id: str
    target_world_id: str
    member_view_ids: tuple[str, ...]
    model_relation_ids: tuple[str, ...]
    plausibility_rule: str
    support_rule: str
    uncertainty_set_ids: tuple[str, ...]
    validity: ValiditySpec
    intersection_semantics: ModelIntersectionSemantics
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.model_set_id, field_name="model_set_id")
        validate_stable_id(self.target_world_id, field_name="target_world_id")
        for field_name, values in (
            ("member_view_ids", self.member_view_ids),
            ("model_relation_ids", self.model_relation_ids),
            ("uncertainty_set_ids", self.uncertainty_set_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        if len(self.member_view_ids) < 2:
            raise ValueError("a robust ModelSetSpec requires at least two members")
        validate_nonempty(self.plausibility_rule, field_name="plausibility_rule")
        validate_nonempty(self.support_rule, field_name="support_rule")
        if self.validity.status not in {
            ObligationStatus.REQUIRED,
            ObligationStatus.SATISFIED,
        }:
            raise ValueError("model set validity must be required or satisfied")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("model-set visibility cannot be lowered")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class ModelMemberLawBinding(CanonicalRecord):
    """One denominator/model member and its locally qualified law."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/model-member-law-binding'

    binding_id: str
    denominator_member_id: str
    relation_id: str
    response_law: ObjectIdentity
    qualification_result: ObjectIdentity
    candidate_version_ids: tuple[str, ...]
    qualification_view_ids: tuple[str, ...]
    stable_property_ids: tuple[str, ...]
    nontransported_property_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("denominator_member_id", self.denominator_member_id),
            ("relation_id", self.relation_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.response_law.object_schema != 'empirical-lawhood/kernel/response-law':
            raise ValueError("model member must bind one ResponseLaw")
        if self.qualification_result.object_schema != (
            'empirical-lawhood/kernel/law-qualification-result'
        ):
            raise ValueError("model member must bind one authoritative law qualification")
        for name, values in (
            ("candidate_version_ids", self.candidate_version_ids),
            ("qualification_view_ids", self.qualification_view_ids),
            ("stable_property_ids", self.stable_property_ids),
            ("nontransported_property_ids", self.nontransported_property_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
        if set(self.stable_property_ids) & set(self.nontransported_property_ids):
            raise ValueError("a model-member property cannot be stable and nontransported")


@dataclass(frozen=True, slots=True)
class ModelSetSpec(CanonicalRecord):
    """Robust denominator-member set with explicit local-law bindings."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/model-set-spec'
    VERSION: ClassVar[str] = '1.0.0'

    model_set_id: str
    target_world_id: str
    members: tuple[ModelMemberLawBinding, ...]
    model_relation_ids: tuple[str, ...]
    plausibility_rule: str
    support_rule: str
    uncertainty_set_ids: tuple[str, ...]
    validity: ValiditySpec
    intersection_semantics: ModelIntersectionSemantics
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.model_set_id, field_name="model_set_id")
        validate_stable_id(self.target_world_id, field_name="target_world_id")
        require_sorted_unique_ids(
            self.members,
            attribute="denominator_member_id",
            field_name="members",
        )
        if not self.members:
            raise ValueError("ModelSetSpec requires at least one denominator member")
        law_ids = tuple(value.response_law.object_id for value in self.members)
        if len(set(law_ids)) != len(law_ids):
            raise ValueError("one ResponseLaw cannot stand for several denominator members")
        for field_name, values in (
            ("model_relation_ids", self.model_relation_ids),
            ("uncertainty_set_ids", self.uncertainty_set_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        validate_nonempty(self.plausibility_rule, field_name="plausibility_rule")
        validate_nonempty(self.support_rule, field_name="support_rule")
        if self.validity.status not in {
            ObligationStatus.REQUIRED,
            ObligationStatus.SATISFIED,
        }:
            raise ValueError("model set validity must be required or satisfied")
        inherited = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(inherited):
            raise ValueError("model-set visibility cannot be lowered")
        require_extensions(self.extensions)


def validate_transport_result(relation: ModelRelation, result: TransportResult) -> None:
    if result.model_relation != ObjectIdentity.from_record(relation.model_relation_id, relation):
        raise ValueError("transport result binds the wrong ModelRelation")
    if (
        EvidenceCeiling.lowest(result.achieved_evidence_ceiling, relation.maximum_evidence)
        is not result.achieved_evidence_ceiling
    ):
        raise ValueError("transport result exceeds its ModelRelation ceiling")
    if result.status is TransportStatus.SUPPORTED:
        if relation.discrepancy.status is not ObligationStatus.SATISFIED:
            raise ValueError("supported transport requires satisfied discrepancy")
        if relation.validity.status is not ObligationStatus.SATISFIED:
            raise ValueError("supported transport requires satisfied validity")

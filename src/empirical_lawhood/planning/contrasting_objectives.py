"Generic contracts for partial morphisms and compact structural fan-in.\n\nThese records deliberately terminate outside the response-law measurement through controller use route.\nThey preserve property, domain, evidence-world and target boundaries without\npooling coefficients, samples or uncertainty across substrates.\n"

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
    validate_nonempty,
    validate_relative_locator,
    validate_semantic_version,
    validate_stable_id,
)

from .evidence_profiles import ExperimentObjectiveKind


class PartialMorphismDirection(StrEnum):
    SOURCE_TO_TARGET = "SOURCE_TO_TARGET"
    TARGET_TO_SOURCE = "TARGET_TO_SOURCE"


class PartialMorphismMapRole(StrEnum):
    PROPERTY = "PROPERTY"
    DENOMINATOR = "DENOMINATOR"
    COORDINATE = "COORDINATE"


class PartialMorphismMapOperation(StrEnum):
    DECLARED_IDENTITY = "DECLARED_IDENTITY"
    DECLARED_RELABEL = "DECLARED_RELABEL"
    DECLARED_PUSHFORWARD = "DECLARED_PUSHFORWARD"


class PartialMorphismUncertaintyOperation(StrEnum):
    PRESERVE = "PRESERVE"
    DECLARED_PUSHFORWARD = "DECLARED_PUSHFORWARD"
    NOT_COMPARABLE = "NOT_COMPARABLE"


class PartialMorphismMaximumClaim(StrEnum):
    PROPERTY_PRESERVATION_ON_DECLARED_DOMAIN = "PROPERTY_PRESERVATION_ON_DECLARED_DOMAIN"


@dataclass(frozen=True, slots=True)
class PartialMorphismMap(CanonicalRecord):
    """One explicit property, denominator or coordinate map."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/partial-morphism-map'

    map_id: str
    role: PartialMorphismMapRole
    property_id: str | None
    source_field_id: str
    target_field_id: str
    operation: PartialMorphismMapOperation
    mathematical_contract_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("map_id", self.map_id),
            ("source_field_id", self.source_field_id),
            ("target_field_id", self.target_field_id),
            ("mathematical_contract_id", self.mathematical_contract_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.role is PartialMorphismMapRole.PROPERTY:
            if self.property_id is None:
                raise ValueError("a property map must name its exact property")
            validate_stable_id(self.property_id, field_name="property_id")
        elif self.property_id is not None:
            raise ValueError("only a property map may name a property")
        if self.source_field_id == self.target_field_id:
            if self.operation is not PartialMorphismMapOperation.DECLARED_IDENTITY:
                raise ValueError("same-name fields require a declared identity map")
        elif self.operation is PartialMorphismMapOperation.DECLARED_IDENTITY:
            raise ValueError("a declared identity map cannot silently rename a field")


@dataclass(frozen=True, slots=True)
class PartialMorphismDomainCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/partial-morphism-domain-cell'

    cell_id: str
    source_denominator_id: str
    target_denominator_id: str
    denominator_map_id: str
    coordinate_map_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("source_denominator_id", self.source_denominator_id),
            ("target_denominator_id", self.target_denominator_id),
            ("denominator_map_id", self.denominator_map_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.coordinate_map_ids,
            field_name="coordinate_map_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class PartialMorphismExcludedField(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/partial-morphism-excluded-field'

    field_id: str
    reason_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.field_id, field_name="field_id")
        validate_stable_id(self.reason_id, field_name="reason_id")


@dataclass(frozen=True, slots=True)
class PartialMorphismSpec(CanonicalRecord):
    """A property-specific partial map with no implicit transport semantics."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/partial-morphism-spec'

    spec_id: str
    source_evidence_world_profile_id: str
    target_evidence_world_profile_id: str
    property_id: str
    direction: PartialMorphismDirection
    property_map: PartialMorphismMap
    denominator_maps: tuple[PartialMorphismMap, ...]
    coordinate_maps: tuple[PartialMorphismMap, ...]
    domain: tuple[PartialMorphismDomainCell, ...]
    excluded_fields: tuple[PartialMorphismExcludedField, ...]
    uncertainty_operation: PartialMorphismUncertaintyOperation
    uncertainty_operation_id: str
    falsifier_ids: tuple[str, ...]
    maximum_claim: PartialMorphismMaximumClaim
    identity_contract_id: str | None
    composition_contract_ids: tuple[str, ...]
    coefficient_or_sample_pooling_permitted: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("spec_id", self.spec_id),
            ("source_evidence_world_profile_id", self.source_evidence_world_profile_id),
            ("target_evidence_world_profile_id", self.target_evidence_world_profile_id),
            ("property_id", self.property_id),
            ("uncertainty_operation_id", self.uncertainty_operation_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.property_map.role is not PartialMorphismMapRole.PROPERTY:
            raise ValueError("partial morphism requires one explicit property map")
        if self.property_map.property_id != self.property_id:
            raise ValueError("property map differs from the morphism property")
        require_sorted_unique_ids(
            self.denominator_maps,
            attribute="map_id",
            field_name="denominator_maps",
        )
        require_sorted_unique_ids(
            self.coordinate_maps,
            attribute="map_id",
            field_name="coordinate_maps",
        )
        if not self.denominator_maps or any(
            value.role is not PartialMorphismMapRole.DENOMINATOR
            for value in self.denominator_maps
        ):
            raise ValueError("partial morphism requires explicit denominator maps")
        if not self.coordinate_maps or any(
            value.role is not PartialMorphismMapRole.COORDINATE for value in self.coordinate_maps
        ):
            raise ValueError("partial morphism requires explicit coordinate maps")
        all_map_ids = tuple(
            value.map_id
            for value in (self.property_map, *self.denominator_maps, *self.coordinate_maps)
        )
        if len(set(all_map_ids)) != len(all_map_ids):
            raise ValueError("partial morphism map identities must be unique")
        require_sorted_unique_ids(self.domain, attribute="cell_id", field_name="domain")
        if not self.domain:
            raise ValueError("partial morphism domain must not be empty")
        denominator_map_ids = {value.map_id for value in self.denominator_maps}
        coordinate_map_ids = {value.map_id for value in self.coordinate_maps}
        if any(
            value.denominator_map_id not in denominator_map_ids
            or not set(value.coordinate_map_ids).issubset(coordinate_map_ids)
            for value in self.domain
        ):
            raise ValueError("partial morphism domain names an undeclared map")
        require_sorted_unique_ids(
            self.excluded_fields,
            attribute="field_id",
            field_name="excluded_fields",
        )
        require_sorted_unique_strings(
            self.falsifier_ids,
            field_name="falsifier_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.composition_contract_ids,
            field_name="composition_contract_ids",
        )
        if self.identity_contract_id is not None:
            validate_stable_id(self.identity_contract_id, field_name="identity_contract_id")
            if not self._has_identity_shape:
                raise ValueError("whole-morphism identity contract lacks an identity shape")
        if self.coefficient_or_sample_pooling_permitted:
            raise ValueError("partial morphism cannot pool coefficients or samples")

    @property
    def _has_identity_shape(self) -> bool:
        maps = (self.property_map, *self.denominator_maps, *self.coordinate_maps)
        return bool(
            self.source_evidence_world_profile_id == self.target_evidence_world_profile_id
            and all(
                value.operation is PartialMorphismMapOperation.DECLARED_IDENTITY for value in maps
            )
            and all(
                value.source_denominator_id == value.target_denominator_id for value in self.domain
            )
        )

    @property
    def is_declared_identity(self) -> bool:
        return self.identity_contract_id is not None and self._has_identity_shape


@dataclass(frozen=True, slots=True)
class PartialMorphismOverlap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/partial-morphism-overlap'

    overlap_id: str
    first_domain_cell_id: str
    second_domain_cell_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("overlap_id", self.overlap_id),
            ("first_domain_cell_id", self.first_domain_cell_id),
            ("second_domain_cell_id", self.second_domain_cell_id),
        ):
            validate_stable_id(value, field_name=name)


@dataclass(frozen=True, slots=True)
class PartialMorphismComposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/partial-morphism-composition'

    composition_id: str
    first_spec: ObjectIdentity
    second_spec: ObjectIdentity
    property_id: str
    composition_contract_id: str
    overlaps: tuple[PartialMorphismOverlap, ...]
    maximum_claim: PartialMorphismMaximumClaim

    def __post_init__(self) -> None:
        validate_stable_id(self.composition_id, field_name="composition_id")
        validate_stable_id(self.property_id, field_name="property_id")
        validate_stable_id(
            self.composition_contract_id,
            field_name="composition_contract_id",
        )
        if (
            self.first_spec.object_schema != PartialMorphismSpec.SCHEMA
            or self.second_spec.object_schema != PartialMorphismSpec.SCHEMA
        ):
            raise ValueError("morphism composition requires exact specifications")
        require_sorted_unique_ids(
            self.overlaps,
            attribute="overlap_id",
            field_name="overlaps",
        )
        if not self.overlaps:
            raise ValueError("morphism composition requires a declared overlap")


def compose_partial_morphisms_on_overlap(
    *,
    composition_id: str,
    first: PartialMorphismSpec,
    second: PartialMorphismSpec,
    composition_contract_id: str,
    overlaps: tuple[PartialMorphismOverlap, ...],
) -> PartialMorphismComposition:
    """Compose only an explicitly declared source-to-target overlap."""

    if (
        first.direction is not PartialMorphismDirection.SOURCE_TO_TARGET
        or second.direction is not PartialMorphismDirection.SOURCE_TO_TARGET
    ):
        raise ValueError("morphism composition direction is not declared")
    if first.target_evidence_world_profile_id != second.source_evidence_world_profile_id:
        raise ValueError("morphism composition endpoints do not meet")
    if first.property_id != second.property_id:
        raise ValueError("morphism composition cannot cross properties")
    if composition_contract_id not in first.composition_contract_ids or (
        composition_contract_id not in second.composition_contract_ids
    ):
        raise ValueError("morphism composition lacks a shared declared contract")
    first_cells = {value.cell_id: value for value in first.domain}
    second_cells = {value.cell_id: value for value in second.domain}
    first_coordinates = {value.map_id: value for value in first.coordinate_maps}
    second_coordinates = {value.map_id: value for value in second.coordinate_maps}
    for overlap in overlaps:
        try:
            left = first_cells[overlap.first_domain_cell_id]
            right = second_cells[overlap.second_domain_cell_id]
        except KeyError as error:
            raise ValueError("morphism composition names an undefined domain cell") from error
        if left.target_denominator_id != right.source_denominator_id:
            raise ValueError("morphism composition denominator overlap differs")
        left_targets = {
            first_coordinates[map_id].target_field_id for map_id in left.coordinate_map_ids
        }
        right_sources = {
            second_coordinates[map_id].source_field_id for map_id in right.coordinate_map_ids
        }
        if left_targets != right_sources:
            raise ValueError("morphism composition coordinate overlap differs")
    return PartialMorphismComposition(
        composition_id=composition_id,
        first_spec=ObjectIdentity.from_record(first.spec_id, first),
        second_spec=ObjectIdentity.from_record(second.spec_id, second),
        property_id=first.property_id,
        composition_contract_id=composition_contract_id,
        overlaps=overlaps,
        maximum_claim=PartialMorphismMaximumClaim.PROPERTY_PRESERVATION_ON_DECLARED_DOMAIN,
    )


class StructuralAnalysisTiming(StrEnum):
    PREDECLARED = "PREDECLARED"
    POST_HOC = "POST_HOC"


class MissingTargetBehavior(StrEnum):
    ASSESS_MISSING_TARGET = "ASSESS_MISSING_TARGET"


class CompactTerminalStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    MIXED = "MIXED"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    UNEVALUABLE = "UNEVALUABLE"


class MetatheoryPropositionDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    MIXED = "MIXED"
    MISSING_TARGET = "MISSING_TARGET"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class StructuralTargetHypothesis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/structural-target-hypothesis'

    hypothesis_id: str
    target_id: str
    evidence_world_profile_id: str
    denominator_semantics_id: str
    property_id: str
    proposition: str

    def __post_init__(self) -> None:
        for name, value in (
            ("hypothesis_id", self.hypothesis_id),
            ("target_id", self.target_id),
            ("evidence_world_profile_id", self.evidence_world_profile_id),
            ("denominator_semantics_id", self.denominator_semantics_id),
            ("property_id", self.property_id),
        ):
            validate_stable_id(value, field_name=name)
        validate_nonempty(self.proposition, field_name="proposition")


COMPACT_SCIENTIFIC_HANDOFF_FIELDS = tuple(
    sorted(
        (
            "action_summary",
            "decisive_falsifier_ids",
            "denominator_summary",
            "evidence_world_profile_id",
            "external_locators",
            "handoff_id",
            "maximum_evidence_ceiling",
            "objective_kind",
            "plan",
            "propositions",
            "receiver_summary",
            "result",
            "target_id",
            "terminal_status",
            "uncertainty_summary",
        )
    )
)


@dataclass(frozen=True, slots=True)
class StructuralHandoffProfile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/structural-handoff-profile'

    profile_id: str
    profile_version: str
    terminal_input_schemas: tuple[str, ...]
    required_compact_fields: tuple[str, ...]
    target_hypotheses: tuple[StructuralTargetHypothesis, ...]
    prediction_analysis_identity: ObjectIdentity
    analysis_timing: StructuralAnalysisTiming
    missing_target_behavior: MissingTargetBehavior
    prediction_frozen_before_handoff_access: bool
    parent_promotion_permitted: bool
    fresh_confirmation_identity_required: bool
    cross_world_pooling_permitted: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_semantic_version(self.profile_version)
        require_sorted_unique_strings(
            self.terminal_input_schemas,
            field_name="terminal_input_schemas",
            allow_empty=False,
        )
        if self.required_compact_fields != COMPACT_SCIENTIFIC_HANDOFF_FIELDS:
            raise ValueError("structural profile changes the compact handoff boundary")
        require_sorted_unique_ids(
            self.target_hypotheses,
            attribute="hypothesis_id",
            field_name="target_hypotheses",
        )
        if not self.target_hypotheses:
            raise ValueError("structural profile requires at least one target hypothesis")
        target_ids = tuple(value.target_id for value in self.target_hypotheses)
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("structural hypotheses contain duplicate targets")
        if not self.prediction_frozen_before_handoff_access:
            raise ValueError("structural analysis identity must freeze before handoff access")
        if self.parent_promotion_permitted:
            raise ValueError("structural handoff analysis cannot promote its parents")
        if not self.fresh_confirmation_identity_required:
            raise ValueError("structural confirmation requires a fresh identity")
        if self.cross_world_pooling_permitted:
            raise ValueError("structural handoff cannot pool across evidence worlds")


@dataclass(frozen=True, slots=True)
class StructuralFaceHandoff(CanonicalRecord):
    """One face key/applicability row; algorithm semantics remain method-owned."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/structural-face-handoff'

    face_id: str
    target_slot_id: str
    property_id: str
    receiver_action_face_id: str
    predictive_level: str
    face_applicability: str
    admission_applicability: str
    prospective_validation_applicability: str

    def __post_init__(self) -> None:
        for name in (
            "face_id",
            "target_slot_id",
            "property_id",
            "receiver_action_face_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.predictive_level not in {"CATEGORICAL", "METRIC", "DYNAMICAL"}:
            raise ValueError("structural face has an unknown predictive level")
        for name in ("face_applicability", 'admission_applicability', 'prospective_validation_applicability'):
            if getattr(self, name) not in {"REQUIRED", "OPTIONAL", "NOT_APPLICABLE"}:
                raise ValueError(f"{name} has an unknown applicability")
        if self.face_applicability == "NOT_APPLICABLE" and (
            self.admission_applicability != "NOT_APPLICABLE" or self.prospective_validation_applicability != "NOT_APPLICABLE"
        ):
            raise ValueError("an inapplicable base face cannot require an overlay")
        if self.prospective_validation_applicability != "NOT_APPLICABLE" and self.admission_applicability == "NOT_APPLICABLE":
            raise ValueError("an applicable controller use overlay requires an applicable admission overlay")

    @property
    def face_key(self) -> tuple[str, str, str, str]:
        return (
            self.target_slot_id,
            self.property_id,
            self.receiver_action_face_id,
            self.predictive_level,
        )


@dataclass(frozen=True, slots=True)
class StructuralFaceHandoffProfile(CanonicalRecord):
    """Facewise overlay permitting many hypotheses over one target episode."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/structural-face-handoff-profile'

    profile_id: str
    profile_version: str
    predecessor_profile: ObjectIdentity
    faces: tuple[StructuralFaceHandoff, ...]
    shared_target_episode_reuse_permitted: bool
    shared_episode_increases_independent_unit_count: bool
    frozen_before_target_outcomes: bool
    cross_world_pooling_permitted: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_semantic_version(self.profile_version)
        if not self.faces:
            raise ValueError("structural face profile requires a nonempty face roster")
        face_ids = tuple(value.face_id for value in self.faces)
        face_keys = tuple(value.face_key for value in self.faces)
        if face_ids != tuple(sorted(face_ids)) or len(set(face_ids)) != len(face_ids):
            raise ValueError("structural face identities must be sorted and unique")
        if len(set(face_keys)) != len(face_keys):
            raise ValueError("structural face keys must be unique")
        if (
            not self.shared_target_episode_reuse_permitted
            or self.shared_episode_increases_independent_unit_count
        ):
            raise ValueError("facewise target reuse cannot create replication")
        if not self.frozen_before_target_outcomes or self.cross_world_pooling_permitted:
            raise ValueError("structural face profile violates prospective world separation")


@dataclass(frozen=True, slots=True)
class CompactDenominatorSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/compact-denominator-summary'

    denominator_id: str
    denominator_semantics_id: str
    physical_independent_unit_kind_id: str
    physical_independent_unit_count: int
    property_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("denominator_id", self.denominator_id),
            ("denominator_semantics_id", self.denominator_semantics_id),
            ("physical_independent_unit_kind_id", self.physical_independent_unit_kind_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.physical_independent_unit_count < 0:
            raise ValueError("physical independent-unit count cannot be negative")
        require_sorted_unique_strings(
            self.property_ids,
            field_name="property_ids",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class CompactActionSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/compact-action-summary'

    action_chart_id: str
    realized_action_semantics_id: str
    action_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.action_chart_id, field_name="action_chart_id")
        validate_stable_id(
            self.realized_action_semantics_id,
            field_name="realized_action_semantics_id",
        )
        require_sorted_unique_strings(self.action_ids, field_name="action_ids")


@dataclass(frozen=True, slots=True)
class CompactReceiverSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/compact-receiver-summary'

    receiver_id: str
    native_unit: str
    frame_id: str
    clock_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.receiver_id, field_name="receiver_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        validate_stable_id(self.frame_id, field_name="frame_id")
        validate_stable_id(self.clock_id, field_name="clock_id")


@dataclass(frozen=True, slots=True)
class CompactUncertaintySummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/compact-uncertainty-summary'

    method_id: str
    independent_unit_count: int
    summary_statistic_ids: tuple[str, ...]
    cross_target_reduction_performed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.method_id, field_name="method_id")
        if self.independent_unit_count < 0:
            raise ValueError("uncertainty independent-unit count cannot be negative")
        require_sorted_unique_strings(
            self.summary_statistic_ids,
            field_name="summary_statistic_ids",
        )
        if self.cross_target_reduction_performed:
            raise ValueError("compact uncertainty cannot reduce across targets")


@dataclass(frozen=True, slots=True)
class CompactStructuralProposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/compact-structural-proposition'

    hypothesis_id: str
    property_id: str
    status: CompactTerminalStatus
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.hypothesis_id, field_name="hypothesis_id")
        validate_stable_id(self.property_id, field_name="property_id")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class CompactScientificHandoff(CanonicalRecord):
    """Bounded terminal metadata; raw scientific payload has no field here."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/compact-scientific-handoff'

    handoff_id: str
    target_id: str
    result: ObjectIdentity
    plan: ObjectIdentity
    evidence_world_profile_id: str
    objective_kind: ExperimentObjectiveKind
    denominator_summary: CompactDenominatorSummary
    action_summary: CompactActionSummary
    receiver_summary: CompactReceiverSummary
    terminal_status: CompactTerminalStatus
    maximum_evidence_ceiling: EvidenceCeiling
    decisive_falsifier_ids: tuple[str, ...]
    uncertainty_summary: CompactUncertaintySummary
    external_locators: tuple[str, ...]
    propositions: tuple[CompactStructuralProposition, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.handoff_id, field_name="handoff_id")
        validate_stable_id(self.target_id, field_name="target_id")
        validate_stable_id(
            self.evidence_world_profile_id,
            field_name="evidence_world_profile_id",
        )
        require_sorted_unique_strings(
            self.decisive_falsifier_ids,
            field_name="decisive_falsifier_ids",
        )
        require_sorted_unique_strings(self.external_locators, field_name="external_locators")
        for locator in self.external_locators:
            validate_relative_locator(locator)
        require_sorted_unique_ids(
            self.propositions,
            attribute="hypothesis_id",
            field_name="propositions",
        )
        if not self.propositions:
            raise ValueError("compact handoff requires a proposition")
        proposition_statuses = {value.status for value in self.propositions}
        if len(proposition_statuses) == 1:
            if self.terminal_status not in proposition_statuses:
                raise ValueError("handoff terminal and proposition status differ")
        elif self.terminal_status is not CompactTerminalStatus.MIXED:
            raise ValueError("multi-status handoff requires a mixed terminal status")
        if self.uncertainty_summary.independent_unit_count != (
            self.denominator_summary.physical_independent_unit_count
        ):
            raise ValueError("uncertainty and denominator unit counts differ")


@dataclass(frozen=True, slots=True)
class MetatheoryFanInCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-fan-in-cell'

    cell_id: str
    hypothesis_id: str
    target_id: str
    evidence_world_profile_id: str
    property_id: str
    disposition: MetatheoryPropositionDisposition
    handoff: ObjectIdentity | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("hypothesis_id", self.hypothesis_id),
            ("target_id", self.target_id),
            ("evidence_world_profile_id", self.evidence_world_profile_id),
            ("property_id", self.property_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if self.disposition is MetatheoryPropositionDisposition.MISSING_TARGET:
            if self.handoff is not None:
                raise ValueError("missing target cell cannot bind a handoff")
        elif self.handoff is None:
            raise ValueError("contacted target cell requires an exact handoff")


@dataclass(frozen=True, slots=True)
class MetatheoryFanInAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/metatheory-fan-in-assessment'

    assessment_id: str
    profile: ObjectIdentity
    analysis_identity: ObjectIdentity
    handoffs: tuple[ObjectIdentity, ...]
    cells: tuple[MetatheoryFanInCell, ...]
    analysis_timing: StructuralAnalysisTiming
    maximum_evidence_ceiling: EvidenceCeiling
    parent_promotion_permitted: bool
    fresh_confirmation_identity_required: bool
    cross_world_pooling_performed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        if self.profile.object_schema != StructuralHandoffProfile.SCHEMA:
            raise ValueError("fan-in assessment requires a structural handoff profile")
        require_sorted_unique_ids(self.handoffs, attribute="object_id", field_name="handoffs")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if self.maximum_evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("metatheory fan-in is non-promotable")
        if self.parent_promotion_permitted:
            raise ValueError("metatheory fan-in cannot promote parent results")
        if not self.fresh_confirmation_identity_required:
            raise ValueError("metatheory confirmation requires a fresh identity")
        if self.cross_world_pooling_performed:
            raise ValueError("metatheory fan-in cannot pool across worlds")


_FAN_IN_STATUS = {
    CompactTerminalStatus.SUPPORTED: MetatheoryPropositionDisposition.SUPPORTED,
    CompactTerminalStatus.CONTRADICTED: MetatheoryPropositionDisposition.CONTRADICTED,
    CompactTerminalStatus.MIXED: MetatheoryPropositionDisposition.MIXED,
    CompactTerminalStatus.PREREQUISITE_NONATTEMPT: (
        MetatheoryPropositionDisposition.UNEVALUABLE
    ),
    CompactTerminalStatus.UNEVALUABLE: MetatheoryPropositionDisposition.UNEVALUABLE,
}


def assess_metatheory_fan_in(
    *,
    assessment_id: str,
    profile: StructuralHandoffProfile,
    handoffs: tuple[CompactScientificHandoff, ...],
    requested_compact_fields: tuple[str, ...] = COMPACT_SCIENTIFIC_HANDOFF_FIELDS,
) -> MetatheoryFanInAssessment:
    """Assess target-local compact handoffs without reopening or pooling data."""

    if requested_compact_fields != profile.required_compact_fields:
        raise ValueError("fan-in requested data outside the compact handoff boundary")
    by_target: dict[str, CompactScientificHandoff] = {}
    for input_handoff in handoffs:
        if input_handoff.target_id in by_target:
            raise ValueError("fan-in contains a duplicate target")
        by_target[input_handoff.target_id] = input_handoff
    expected_targets = {value.target_id for value in profile.target_hypotheses}
    if not set(by_target).issubset(expected_targets):
        raise ValueError("fan-in contains an undeclared target")

    cells = []
    for hypothesis in profile.target_hypotheses:
        handoff = by_target.get(hypothesis.target_id)
        cell_id = f"fan-in-cell.{assessment_id}.{hypothesis.target_id}"
        if handoff is None:
            cells.append(
                MetatheoryFanInCell(
                    cell_id=cell_id,
                    hypothesis_id=hypothesis.hypothesis_id,
                    target_id=hypothesis.target_id,
                    evidence_world_profile_id=hypothesis.evidence_world_profile_id,
                    property_id=hypothesis.property_id,
                    disposition=MetatheoryPropositionDisposition.MISSING_TARGET,
                    handoff=None,
                    reason_codes=("TARGET_HANDOFF_MISSING",),
                )
            )
            continue
        if handoff.evidence_world_profile_id != hypothesis.evidence_world_profile_id:
            raise ValueError("fan-in target evidence world differs from its hypothesis")
        if (
            handoff.denominator_summary.denominator_semantics_id
            != hypothesis.denominator_semantics_id
        ):
            raise ValueError("fan-in target denominator differs from its hypothesis")
        if hypothesis.property_id not in handoff.denominator_summary.property_ids:
            raise ValueError("fan-in target property differs from its hypothesis")
        propositions = {value.hypothesis_id: value for value in handoff.propositions}
        try:
            proposition = propositions[hypothesis.hypothesis_id]
        except KeyError as error:
            raise ValueError("fan-in handoff omits its target hypothesis") from error
        if proposition.property_id != hypothesis.property_id:
            raise ValueError("fan-in proposition property differs from its hypothesis")
        reasons = set(proposition.reason_codes)
        if profile.analysis_timing is StructuralAnalysisTiming.POST_HOC:
            reasons.add("POST_HOC_NONPROMOTABLE")
        cells.append(
            MetatheoryFanInCell(
                cell_id=cell_id,
                hypothesis_id=hypothesis.hypothesis_id,
                target_id=hypothesis.target_id,
                evidence_world_profile_id=hypothesis.evidence_world_profile_id,
                property_id=hypothesis.property_id,
                disposition=_FAN_IN_STATUS[proposition.status],
                handoff=ObjectIdentity.from_record(handoff.handoff_id, handoff),
                reason_codes=tuple(sorted(reasons)),
            )
        )
    identities = tuple(
        sorted(
            (ObjectIdentity.from_record(value.handoff_id, value) for value in handoffs),
            key=lambda value: value.object_id,
        )
    )
    return MetatheoryFanInAssessment(
        assessment_id=assessment_id,
        profile=ObjectIdentity.from_record(profile.profile_id, profile),
        analysis_identity=profile.prediction_analysis_identity,
        handoffs=identities,
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        analysis_timing=profile.analysis_timing,
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        parent_promotion_permitted=False,
        fresh_confirmation_identity_required=True,
        cross_world_pooling_performed=False,
    )


__all__ = [
    "COMPACT_SCIENTIFIC_HANDOFF_FIELDS",
    'CompactActionSummary',
    'CompactDenominatorSummary',
    'CompactReceiverSummary',
    'CompactScientificHandoff',
    'CompactStructuralProposition',
    'CompactTerminalStatus',
    'CompactUncertaintySummary',
    'MetatheoryFanInAssessment',
    'MetatheoryFanInCell',
    'MetatheoryPropositionDisposition',
    'MissingTargetBehavior',
    'PartialMorphismComposition',
    'PartialMorphismDirection',
    'PartialMorphismDomainCell',
    'PartialMorphismExcludedField',
    'PartialMorphismMapOperation',
    'PartialMorphismMapRole',
    'PartialMorphismMap',
    'PartialMorphismMaximumClaim',
    'PartialMorphismOverlap',
    'PartialMorphismSpec',
    'PartialMorphismUncertaintyOperation',
    'StructuralAnalysisTiming',
    'StructuralFaceHandoffProfile',
    'StructuralFaceHandoff',
    'StructuralHandoffProfile',
    'StructuralTargetHypothesis',
    "assess_metatheory_fan_in",
    "compose_partial_morphisms_on_overlap",
]

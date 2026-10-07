"""Canonical, non-authorizing structural obstruction atlas records."""

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
    validate_schema,
    validate_stable_id,
)
from empirical_lawhood.planning.metatheory import LegitimateNextAct, MetatheoryCellDisposition, MetatheoryClaimKind, MetatheoryEvidenceCeiling, MetatheoryObstructionKind, MetatheoryPredictiveLevel
from empirical_lawhood.planning.metatheory_campaign import MetatheoryCampaignStageRole


class ObstructionOperationalStatus(StrEnum):
    COMPLETE = "COMPLETE"
    STOPPED = "STOPPED"
    BLOCKED = "BLOCKED"


class ObstructionClaimEffect(StrEnum):
    OPPOSES = "OPPOSES"
    UNEVALUABLE = "UNEVALUABLE"
    PREREQUISITE_NONATTEMPT = "PREREQUISITE_NONATTEMPT"
    NO_SCIENTIFIC_EFFECT = "NO_SCIENTIFIC_EFFECT"


class ObstructionAtlasBuildStopKind(StrEnum):
    INPUT_INCOMPLETE = "INPUT_INCOMPLETE"
    INPUT_INCOMPATIBLE = "INPUT_INCOMPATIBLE"
    UNKNOWN_REASON = "UNKNOWN_REASON"
    EXPECTED_TERMINAL_MISSING = "EXPECTED_TERMINAL_MISSING"


@dataclass(frozen=True, slots=True)
class ObstructionSourceBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/obstruction-source-binding'

    source_id: str
    expected_cell_id: str
    terminal: ObjectIdentity
    stage: MetatheoryCampaignStageRole
    target_id: str | None
    claim_kind: MetatheoryClaimKind
    predictive_level: MetatheoryPredictiveLevel | None
    affected_operand_ids: tuple[str, ...]
    affected_property_ids: tuple[str, ...]
    affected_coordinate_ids: tuple[str, ...]
    affected_action_ids: tuple[str, ...]
    evidence_links: tuple[ObjectIdentity, ...]
    source_reason_codes: tuple[str, ...]
    scientific_disposition: MetatheoryCellDisposition
    operational_status: ObstructionOperationalStatus

    def __post_init__(self) -> None:
        for name in ("source_id", "expected_cell_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.target_id is not None:
            validate_stable_id(self.target_id, field_name="target_id")
        for name, values in (
            ("affected_operand_ids", self.affected_operand_ids),
            ("affected_property_ids", self.affected_property_ids),
            ("affected_coordinate_ids", self.affected_coordinate_ids),
            ("affected_action_ids", self.affected_action_ids),
            ("source_reason_codes", self.source_reason_codes),
        ):
            require_sorted_unique_strings(values, field_name=name)
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )
        if self.scientific_disposition is MetatheoryCellDisposition.SUPPORTED:
            if self.source_reason_codes:
                raise ValueError("supported terminal cannot carry obstruction reasons")
        elif not self.source_reason_codes:
            raise ValueError("non-supported terminal requires exact reason codes")


@dataclass(frozen=True, slots=True)
class ObstructionMapping(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/obstruction-mapping'

    mapping_id: str
    terminal_schema: str
    reason_code: str
    obstruction_kind: MetatheoryObstructionKind
    claim_effect: ObstructionClaimEffect
    allowed_next_acts: tuple[LegitimateNextAct, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.mapping_id, field_name="mapping_id")
        validate_schema(self.terminal_schema)
        validate_nonempty(self.reason_code, field_name="reason_code")
        if (
            tuple(sorted(set(self.allowed_next_acts), key=lambda value: value.value))
            != self.allowed_next_acts
        ):
            raise ValueError("obstruction next acts must be sorted and unique")
        if not self.allowed_next_acts:
            raise ValueError("obstruction mapping requires an advisory next act")


@dataclass(frozen=True, slots=True)
class ObstructionMappingRegistry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/obstruction-mapping-registry'

    registry_id: str
    mappings: tuple[ObstructionMapping, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        require_sorted_unique_ids(self.mappings, attribute="mapping_id", field_name="mappings")
        keys = {(value.terminal_schema, value.reason_code) for value in self.mappings}
        if len(keys) != len(self.mappings):
            raise ValueError("obstruction registry repeats schema/reason mapping")


@dataclass(frozen=True, slots=True)
class ObstructionExpectedCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/obstruction-expected-cell'

    expected_cell_id: str
    stage: MetatheoryCampaignStageRole
    target_id: str | None
    required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.expected_cell_id, field_name="expected_cell_id")
        if self.target_id is not None:
            validate_stable_id(self.target_id, field_name="target_id")


@dataclass(frozen=True, slots=True)
class ObstructionAtlasSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/obstruction-atlas-spec'

    spec_id: str
    accepted_terminal_schemas: tuple[str, ...]
    mapping_registry: ObstructionMappingRegistry
    expected_cells: tuple[ObstructionExpectedCell, ...]
    aggregation_policy_id: str
    allow_unknown_as_unresolved: bool
    missing_expected_as_obstruction: bool
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.spec_id, field_name="spec_id")
        require_sorted_unique_strings(
            self.accepted_terminal_schemas,
            field_name="accepted_terminal_schemas",
            allow_empty=False,
        )
        for schema in self.accepted_terminal_schemas:
            validate_schema(schema)
        if not {value.terminal_schema for value in self.mapping_registry.mappings}.issubset(
            self.accepted_terminal_schemas
        ):
            raise ValueError("obstruction mapping registry names an unaccepted schema")
        require_sorted_unique_ids(
            self.expected_cells,
            attribute="expected_cell_id",
            field_name="expected_cells",
        )
        if not self.expected_cells:
            raise ValueError("obstruction atlas requires expected terminal cells")
        validate_stable_id(self.aggregation_policy_id, field_name="aggregation_policy_id")


@dataclass(frozen=True, slots=True)
class ObstructionCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/obstruction-cell'

    cell_id: str
    source_bindings: tuple[ObjectIdentity, ...]
    obstruction_kind: MetatheoryObstructionKind
    claim_effect: ObstructionClaimEffect
    legitimate_next_act: LegitimateNextAct
    overlap_group_id: str
    maximum_ordinary_evidence_ceiling: EvidenceCeiling
    maximum_structural_evidence_ceiling: MetatheoryEvidenceCeiling
    grants_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_ids(
            self.source_bindings,
            attribute="object_id",
            field_name="source_bindings",
        )
        validate_stable_id(self.overlap_group_id, field_name="overlap_group_id")
        if self.grants_authority:
            raise ValueError("obstruction cell advice cannot grant authority")


@dataclass(frozen=True, slots=True)
class ObstructionAtlas(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/obstruction-atlas'

    atlas_id: str
    atlas_spec: ObjectIdentity
    source_bindings: tuple[ObstructionSourceBinding, ...]
    cells: tuple[ObstructionCell, ...]
    missing_expected_cell_ids: tuple[str, ...]
    unobstructed_terminal_refs: tuple[ObjectIdentity, ...]
    operational_statuses: tuple[ObstructionOperationalStatus, ...]
    scientific_dispositions: tuple[MetatheoryCellDisposition, ...]
    evidence_links: tuple[ObjectIdentity, ...]
    maximum_claim_effect: ObstructionClaimEffect
    grants_authority: bool
    executed_or_revealed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.atlas_id, field_name="atlas_id")
        require_sorted_unique_ids(
            self.source_bindings,
            attribute="source_id",
            field_name="source_bindings",
        )
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_strings(
            self.missing_expected_cell_ids,
            field_name="missing_expected_cell_ids",
        )
        require_sorted_unique_ids(
            self.unobstructed_terminal_refs,
            attribute="object_id",
            field_name="unobstructed_terminal_refs",
        )
        if (
            tuple(sorted(set(self.operational_statuses), key=lambda value: value.value))
            != self.operational_statuses
        ):
            raise ValueError("obstruction operational statuses must be sorted and unique")
        if (
            tuple(sorted(set(self.scientific_dispositions), key=lambda value: value.value))
            != self.scientific_dispositions
        ):
            raise ValueError("obstruction scientific dispositions must be sorted and unique")
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="object_id",
            field_name="evidence_links",
        )
        if self.grants_authority or self.executed_or_revealed:
            raise ValueError("obstruction atlas cannot authorize or cause effects")


@dataclass(frozen=True, slots=True)
class ObstructionAtlasBuildStop(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/obstruction-atlas-build-stop'

    stop_id: str
    atlas_spec: ObjectIdentity
    stop_kind: ObstructionAtlasBuildStopKind
    source_binding_refs: tuple[ObjectIdentity, ...]
    reason_codes: tuple[str, ...]
    atlas_constructed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.stop_id, field_name="stop_id")
        require_sorted_unique_ids(
            self.source_binding_refs,
            attribute="object_id",
            field_name="source_binding_refs",
        )
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if self.atlas_constructed:
            raise ValueError("obstruction build stop cannot claim an atlas")


__all__ = [
    'ObstructionAtlasBuildStopKind',
    'ObstructionAtlasBuildStop',
    'ObstructionAtlasSpec',
    'ObstructionAtlas',
    'ObstructionCell',
    'ObstructionClaimEffect',
    'ObstructionExpectedCell',
    'ObstructionMappingRegistry',
    'ObstructionMapping',
    'ObstructionOperationalStatus',
    'ObstructionSourceBinding',
]

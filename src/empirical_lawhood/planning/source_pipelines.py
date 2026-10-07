"""Outcome-blind source and finite ETL authoring contracts.

The profile is deliberately not an ETL language.  It selects one source and a
finite sequence of registered transform capabilities, binds exact schemas and
coordinate semantics, and gives the runtime enough information to lower into
the existing source, dataset-authority and artifact contracts.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling, inherited_visibility
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)

from .dataset_authority import DATASET_TRANSFORMATION_MANIFEST_SCHEMA
from .study_authoring import CapabilitySelection


MAX_SOURCE_PIPELINE_EDGES = 32
MAX_SOURCE_PIPELINE_COORDINATES = 256
_FORBIDDEN_LOCATOR_FRAGMENTS = ("*", "?", "[", "]", "{", "}", ";", "|", "&", "`", "$(")


def _validate_static_locator(value: str, *, field_name: str) -> None:
    validate_relative_locator(value)
    if any(fragment in value for fragment in _FORBIDDEN_LOCATOR_FRAGMENTS):
        raise ValueError(f"{field_name} cannot contain shell or glob syntax")
    if any(character.isspace() for character in value):
        raise ValueError(f"{field_name} cannot contain whitespace")


def _validate_coordinates(
    values: tuple[SourcePipelineCoordinate, ...],
    *,
    field_name: str,
) -> None:
    if not values or len(values) > MAX_SOURCE_PIPELINE_COORDINATES:
        raise ValueError(f"{field_name} count is outside its bound")
    require_sorted_unique_ids(values, attribute="coordinate_id", field_name=field_name)
    field_ids = tuple(value.field_id for value in values)
    if len(set(field_ids)) != len(field_ids):
        raise ValueError(f"{field_name} repeats a field")


class SourcePipelineMode(StrEnum):
    ARCHIVAL = "ARCHIVAL"
    SIMULATED = "SIMULATED"


class SourcePipelineArtifactProfile(StrEnum):
    CANONICAL_JSON = "CANONICAL_JSON"
    ARROW_IPC = "ARROW_IPC"
    PARQUET = "PARQUET"
    AUDITED_HDF5 = "AUDITED_HDF5"
    NUMPY_NO_PICKLE = "NUMPY_NO_PICKLE"
    JSONL_CHUNKS = "JSONL_CHUNKS"


class SourcePipelineChangeSemantics(StrEnum):
    PRESERVED = "PRESERVED"
    DECLARED_CHANGE = "DECLARED_CHANGE"


class SourcePipelineQualificationDisposition(StrEnum):
    QUALIFIED = "QUALIFIED"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    INVALID = "INVALID"


@dataclass(frozen=True, slots=True)
class SourcePipelineCoordinate(CanonicalRecord):
    """One native field coordinate; values remain native, not normalized away."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/source-pipeline-coordinate'

    coordinate_id: str
    field_id: str
    native_unit: str
    frame_id: str
    clock_id: str

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_stable_id(self.field_id, field_name="field_id")
        for field_name in ("native_unit", "frame_id", "clock_id"):
            validate_nonempty(getattr(self, field_name), field_name=field_name)


def _coordinate_property(
    values: tuple[SourcePipelineCoordinate, ...],
    attribute: str,
) -> tuple[tuple[str, str], ...]:
    return tuple((value.coordinate_id, str(getattr(value, attribute))) for value in values)


def _validate_change_semantics(
    *,
    semantics: SourcePipelineChangeSemantics,
    change_contract: ObjectIdentity | None,
    before: tuple[tuple[str, str], ...],
    after: tuple[tuple[str, str], ...],
    field_name: str,
) -> None:
    changed = before != after
    if semantics is SourcePipelineChangeSemantics.PRESERVED:
        if changed or change_contract is not None:
            raise ValueError(f"{field_name} preservation declaration is false")
    elif not changed or change_contract is None:
        raise ValueError(f"{field_name} change requires an exact change contract")


@dataclass(frozen=True, slots=True)
class SourcePipelineTransformEdge(CanonicalRecord):
    """One registered, bounded transform edge with explicit semantic effects."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/source-pipeline-transform-edge'

    edge_id: str
    rank: int
    dataset_transform_registry_id: str
    selection: CapabilitySelection
    config: ObjectIdentity
    domain_schema: str
    codomain_schema: str
    source_media_type: str
    destination_media_type: str
    source_format_profile_id: str
    destination_format_profile_id: str
    input_coordinates: tuple[SourcePipelineCoordinate, ...]
    output_coordinates: tuple[SourcePipelineCoordinate, ...]
    native_unit_semantics: SourcePipelineChangeSemantics
    native_unit_change_contract: ObjectIdentity | None
    frame_semantics: SourcePipelineChangeSemantics
    frame_change_contract: ObjectIdentity | None
    clock_semantics: SourcePipelineChangeSemantics
    clock_change_contract: ObjectIdentity | None
    row_semantics: SourcePipelineChangeSemantics
    row_change_contract: ObjectIdentity | None
    physical_unit_roster_preserved: bool
    invalid_rows_retained: bool
    semantic_validator: ObjectIdentity
    resource_budget: ResourceBudget

    def __post_init__(self) -> None:
        validate_stable_id(self.edge_id, field_name="edge_id")
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank < 0:
            raise ValueError("source-pipeline edge rank must be nonnegative")
        validate_nonempty(
            self.dataset_transform_registry_id,
            field_name="dataset_transform_registry_id",
        )
        validate_schema(self.domain_schema)
        validate_schema(self.codomain_schema)
        for field_name in (
            "source_media_type",
            "destination_media_type",
            "source_format_profile_id",
            "destination_format_profile_id",
        ):
            validate_nonempty(getattr(self, field_name), field_name=field_name)
        _validate_coordinates(self.input_coordinates, field_name="input_coordinates")
        _validate_coordinates(self.output_coordinates, field_name="output_coordinates")
        input_ids = tuple(value.coordinate_id for value in self.input_coordinates)
        output_ids = tuple(value.coordinate_id for value in self.output_coordinates)
        if input_ids != output_ids:
            raise ValueError("transform edge must retain an explicit coordinate correspondence")
        _validate_change_semantics(
            semantics=self.native_unit_semantics,
            change_contract=self.native_unit_change_contract,
            before=_coordinate_property(self.input_coordinates, "native_unit"),
            after=_coordinate_property(self.output_coordinates, "native_unit"),
            field_name="native unit",
        )
        _validate_change_semantics(
            semantics=self.frame_semantics,
            change_contract=self.frame_change_contract,
            before=_coordinate_property(self.input_coordinates, "frame_id"),
            after=_coordinate_property(self.output_coordinates, "frame_id"),
            field_name="frame",
        )
        _validate_change_semantics(
            semantics=self.clock_semantics,
            change_contract=self.clock_change_contract,
            before=_coordinate_property(self.input_coordinates, "clock_id"),
            after=_coordinate_property(self.output_coordinates, "clock_id"),
            field_name="clock",
        )
        if self.row_semantics is SourcePipelineChangeSemantics.PRESERVED:
            if self.row_change_contract is not None:
                raise ValueError("preserved row semantics cannot carry a change contract")
        elif self.row_change_contract is None:
            raise ValueError("declared row change requires an exact change contract")
        if not self.physical_unit_roster_preserved:
            raise ValueError("F2 source pipelines must preserve the physical-unit roster")
        if not self.invalid_rows_retained:
            raise ValueError("F2 source pipelines must retain explicitly invalid rows")


@dataclass(frozen=True, slots=True)
class SourcePipelineProfile(CanonicalRecord):
    """Finite static source/ETL profile, not caller-programmable workflow code."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/source-pipeline-profile'

    profile_id: str
    profile_version: str
    mode: SourcePipelineMode
    source_selection: CapabilitySelection
    source_config: ObjectIdentity
    source_id: str
    source_materialization: ObjectIdentity
    expected_source_sha256: str
    expected_source_size_bytes: int
    licence_id: str
    operator_storage_root_id: str
    source_relative_locator: str
    source_schema: str
    source_media_type: str
    source_format_profile_id: str
    input_coordinates: tuple[SourcePipelineCoordinate, ...]
    transform_edges: tuple[SourcePipelineTransformEdge, ...]
    output_schema: str
    output_media_type: str
    output_filename_suffix: str
    output_artifact_profile: SourcePipelineArtifactProfile
    output_coordinates: tuple[SourcePipelineCoordinate, ...]
    semantic_validators: tuple[ObjectIdentity, ...]
    physical_unit_field_id: str
    invalid_disposition_field_id: str
    causal_cutoff: ObjectIdentity
    dataset_transformation_manifest: ObjectIdentity
    resource_ceiling: ResourceBudget
    maximum_row_count: int
    maximum_physical_unit_count: int
    scratch_namespace: str
    output_relative_locator: str
    evidence_world_id: str
    observation_operator: ObjectIdentity
    numerical_view_ids: tuple[str, ...]
    receiver_semantics_id: str
    validity_contract_id: str
    uncertainty_contract_id: str
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_semantic_version(self.profile_version)
        validate_stable_id(self.source_id, field_name="source_id")
        validate_sha256(self.expected_source_sha256, field_name="expected_source_sha256")
        if (
            not isinstance(self.expected_source_size_bytes, int)
            or isinstance(self.expected_source_size_bytes, bool)
            or self.expected_source_size_bytes < 0
        ):
            raise ValueError("expected source size must be a nonnegative integer")
        validate_nonempty(self.licence_id, field_name="licence_id")
        validate_stable_id(self.operator_storage_root_id, field_name="operator_storage_root_id")
        _validate_static_locator(
            self.source_relative_locator,
            field_name="source_relative_locator",
        )
        _validate_static_locator(self.scratch_namespace, field_name="scratch_namespace")
        _validate_static_locator(
            self.output_relative_locator,
            field_name="output_relative_locator",
        )
        if self.output_relative_locator.startswith(f"{self.scratch_namespace}/"):
            raise ValueError("durable output cannot remain inside scratch")
        validate_schema(self.source_schema)
        validate_schema(self.output_schema)
        for field_name in (
            "source_media_type",
            "source_format_profile_id",
            "output_media_type",
        ):
            validate_nonempty(getattr(self, field_name), field_name=field_name)
        if not self.output_filename_suffix.startswith(".") or any(
            value in self.output_filename_suffix for value in ("/", "\\", " ")
        ):
            raise ValueError("output filename suffix must be a simple dotted suffix")
        _validate_coordinates(self.input_coordinates, field_name="input_coordinates")
        _validate_coordinates(self.output_coordinates, field_name="output_coordinates")
        if not self.transform_edges or len(self.transform_edges) > MAX_SOURCE_PIPELINE_EDGES:
            raise ValueError("source pipeline transform count is outside its bound")
        if tuple(value.rank for value in self.transform_edges) != tuple(
            range(len(self.transform_edges))
        ):
            raise ValueError("source pipeline edges must be ordered contiguously from zero")
        edge_ids = tuple(value.edge_id for value in self.transform_edges)
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("source pipeline edge IDs repeat")
        previous_schema = self.source_schema
        previous_media = self.source_media_type
        previous_profile = self.source_format_profile_id
        previous_coordinates = self.input_coordinates
        for edge in self.transform_edges:
            if (
                edge.domain_schema != previous_schema
                or edge.source_media_type != previous_media
                or edge.source_format_profile_id != previous_profile
                or edge.input_coordinates != previous_coordinates
            ):
                raise ValueError("source pipeline transform chain is discontinuous")
            previous_schema = edge.codomain_schema
            previous_media = edge.destination_media_type
            previous_profile = edge.destination_format_profile_id
            previous_coordinates = edge.output_coordinates
            if not self.resource_ceiling.contains(edge.resource_budget):
                raise ValueError("transform edge exceeds the source-pipeline resource ceiling")
        if (
            previous_schema != self.output_schema
            or previous_media != self.output_media_type
            or previous_coordinates != self.output_coordinates
        ):
            raise ValueError("source pipeline output differs from its terminal edge")
        require_sorted_unique_ids(
            self.semantic_validators,
            attribute="object_id",
            field_name="semantic_validators",
        )
        if not self.semantic_validators:
            raise ValueError("source pipeline requires semantic validators")
        terminal_validators = {value.semantic_validator for value in self.transform_edges}
        if not terminal_validators.issubset(set(self.semantic_validators)):
            raise ValueError("source pipeline omits an edge semantic validator")
        output_fields = {value.field_id for value in self.output_coordinates}
        for field_name in ("physical_unit_field_id", "invalid_disposition_field_id"):
            value = getattr(self, field_name)
            validate_stable_id(value, field_name=field_name)
            if value not in output_fields:
                raise ValueError(f"{field_name} is absent from output coordinates")
        if (
            self.dataset_transformation_manifest.object_schema
            != DATASET_TRANSFORMATION_MANIFEST_SCHEMA
        ):
            raise ValueError("source pipeline requires an exact dataset transformation manifest")
        for field_name in ("maximum_row_count", "maximum_physical_unit_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.resource_ceiling.source_scan_bytes < self.expected_source_size_bytes:
            raise ValueError("source size exceeds the source-pipeline resource ceiling")
        if self.resource_ceiling.output_bytes <= 0:
            raise ValueError("source pipeline requires a positive output-byte ceiling")
        for field_name in (
            "evidence_world_id",
            "receiver_semantics_id",
            "validity_contract_id",
            "uncertainty_contract_id",
        ):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        require_sorted_unique_strings(
            self.numerical_view_ids,
            field_name="numerical_view_ids",
            allow_empty=False,
        )
        required_visibility = inherited_visibility((), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(required_visibility):
            raise ValueError("source pipeline visibility understates outcome access")


@dataclass(frozen=True, slots=True)
class SourceRuntimeQualificationBindingReceipt(CanonicalRecord):
    """Exact post-Q mapping from prospective identities to observed qualifications."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/source-runtime-qualification-binding-receipt'

    receipt_id: str
    source_expectation: ObjectIdentity
    runtime_expectation: ObjectIdentity
    source_qualification: ObjectIdentity
    runtime_qualification: ObjectIdentity
    verifier_implementation_id: str
    verifier_implementation_sha256: str
    binding_verified: bool
    compact_values_read: bool
    grants_authority_or_scientific_promotion: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(
            self.verifier_implementation_id,
            field_name="verifier_implementation_id",
        )
        validate_sha256(
            self.verifier_implementation_sha256,
            field_name="verifier_implementation_sha256",
        )
        if self.source_qualification.object_schema != (SourcePipelineQualificationReceipt.SCHEMA):
            raise ValueError("source/runtime mapping requires source qualification evidence")
        if (
            self.source_qualification == self.source_expectation
            or self.runtime_qualification == self.runtime_expectation
            or self.source_qualification == self.runtime_qualification
        ):
            raise ValueError(
                "source/runtime qualification receipt confuses expectations with evidence"
            )
        if not self.binding_verified:
            raise ValueError("source/runtime qualification mapping is not verified")
        if self.compact_values_read:
            raise ValueError("source/runtime qualification mapping cannot read scientific values")
        if self.grants_authority_or_scientific_promotion:
            raise ValueError(
                "source/runtime qualification mapping cannot grant authority or science"
            )
        if self.outcome_access not in {
            OutcomeAccess.OUTCOME_BLIND,
            OutcomeAccess.EVALUATION_SEALED,
        }:
            raise ValueError("source/runtime qualification mapping cannot reveal outcomes")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("source/runtime qualification mapping must remain prospective")


@dataclass(frozen=True, slots=True)
class SourcePipelineEdgeAccounting(CanonicalRecord):
    """Exact per-edge accounting needed to detect hidden row/unit loss."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/source-pipeline-edge-accounting'

    edge_id: str
    input_artifact: ObjectIdentity
    output_artifact: ObjectIdentity
    input_row_count: int
    output_row_count: int
    input_invalid_row_count: int
    output_invalid_row_count: int
    physical_unit_count: int
    input_physical_unit_roster_sha256: str
    output_physical_unit_roster_sha256: str
    dropped_without_disposition_count: int
    observed_input_coordinates: tuple[SourcePipelineCoordinate, ...]
    observed_output_coordinates: tuple[SourcePipelineCoordinate, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.edge_id, field_name="edge_id")
        for field_name in (
            "input_row_count",
            "output_row_count",
            "input_invalid_row_count",
            "output_invalid_row_count",
            "physical_unit_count",
            "dropped_without_disposition_count",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        if self.input_invalid_row_count > self.input_row_count or (
            self.output_invalid_row_count > self.output_row_count
        ):
            raise ValueError("invalid-row count exceeds its row count")
        validate_sha256(
            self.input_physical_unit_roster_sha256,
            field_name="input_physical_unit_roster_sha256",
        )
        validate_sha256(
            self.output_physical_unit_roster_sha256,
            field_name="output_physical_unit_roster_sha256",
        )
        _validate_coordinates(
            self.observed_input_coordinates,
            field_name="observed_input_coordinates",
        )
        _validate_coordinates(
            self.observed_output_coordinates,
            field_name="observed_output_coordinates",
        )


@dataclass(frozen=True, slots=True)
class SourcePipelineQualificationReceipt(CanonicalRecord):
    """Pipeline qualification companion to the current source receipt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/source-pipeline-qualification-receipt'

    receipt_id: str
    profile: ObjectIdentity
    disposition: SourcePipelineQualificationDisposition
    source_result: ObjectIdentity | None
    edge_accounting: tuple[SourcePipelineEdgeAccounting, ...]
    output_artifact: ObjectIdentity | None
    output_content_sha256: str | None
    validator_receipts: tuple[ObjectIdentity, ...]
    publication: ObjectIdentity | None
    recovery: ObjectIdentity | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.profile.object_schema != SourcePipelineProfile.SCHEMA:
            raise ValueError("pipeline receipt requires an exact source profile")
        require_sorted_unique_ids(
            self.validator_receipts,
            attribute="object_id",
            field_name="validator_receipts",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.output_content_sha256 is not None:
            validate_sha256(self.output_content_sha256, field_name="output_content_sha256")
        if self.disposition is SourcePipelineQualificationDisposition.QUALIFIED:
            if (
                self.source_result is None
                or not self.edge_accounting
                or self.output_artifact is None
                or self.output_content_sha256 is None
                or not self.validator_receipts
                or self.publication is None
                or self.recovery is None
                or self.reason_codes
            ):
                raise ValueError("qualified source pipeline receipt is incomplete")
        elif self.disposition is SourcePipelineQualificationDisposition.AUTHORITY_REQUIRED:
            if (
                any(
                    value is not None
                    for value in (
                        self.source_result,
                        self.output_artifact,
                        self.output_content_sha256,
                        self.publication,
                        self.recovery,
                    )
                )
                or self.edge_accounting
                or self.validator_receipts
                or not self.reason_codes
            ):
                raise ValueError("authority stop cannot retain unexecuted source evidence")
        elif self.publication is not None or self.recovery is not None or not self.reason_codes:
            raise ValueError("invalid pipeline cannot claim publication/recovery")
        required_visibility = inherited_visibility((), self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(required_visibility):
            raise ValueError("pipeline receipt visibility understates outcome access")


__all__ = [
    'SourcePipelineArtifactProfile',
    'SourcePipelineChangeSemantics',
    'SourcePipelineCoordinate',
    'SourcePipelineEdgeAccounting',
    'SourcePipelineMode',
    'SourcePipelineProfile',
    'SourcePipelineQualificationDisposition',
    'SourcePipelineQualificationReceipt',
    'SourcePipelineTransformEdge',
    'SourceRuntimeQualificationBindingReceipt',
]

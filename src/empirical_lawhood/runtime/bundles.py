"""Canonical data-bundle contracts: standard envelope, substrate-local science."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling, inherited_visibility
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .artifacts import ArtifactProfile, LogicalArtifactIdentity


class TableRole(StrEnum):
    CAUSAL_STATE = "CAUSAL_STATE"
    ACTION_LEDGER = "ACTION_LEDGER"
    RECEIVER_SINK_EFFORT = "RECEIVER_SINK_EFFORT"
    ONE_FACTOR_EXCHANGE = "ONE_FACTOR_EXCHANGE"
    EPISODE_EVENT = "EPISODE_EVENT"
    COMPONENT_INTERFACE = "COMPONENT_INTERFACE"
    NUMERICAL_FIDELITY = "NUMERICAL_FIDELITY"
    CONSTRAINT_LEDGER = "CONSTRAINT_LEDGER"
    EXTENSION = "EXTENSION"


@dataclass(frozen=True, slots=True)
class ColumnContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/column-contract'

    column_id: str
    arrow_type: str
    native_unit: str
    coordinate_frame: str
    clock_id: str | None
    nullable: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.column_id, field_name="column_id")
        for name, value in (
            ("arrow_type", self.arrow_type),
            ("native_unit", self.native_unit),
            ("coordinate_frame", self.coordinate_frame),
        ):
            validate_nonempty(value, field_name=name)
        if self.clock_id is not None:
            validate_stable_id(self.clock_id, field_name="clock_id")


@dataclass(frozen=True, slots=True)
class ForeignKeyContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/foreign-key-contract'

    foreign_key_id: str
    local_column_ids: tuple[str, ...]
    target_table_id: str
    target_column_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.foreign_key_id, field_name="foreign_key_id")
        validate_stable_id(self.target_table_id, field_name="target_table_id")
        require_sorted_unique_strings(
            self.local_column_ids,
            field_name="local_column_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.target_column_ids,
            field_name="target_column_ids",
            allow_empty=False,
        )
        if len(self.local_column_ids) != len(self.target_column_ids):
            raise ValueError("foreign-key source and target arity differ")


@dataclass(frozen=True, slots=True)
class TableContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/table-contract'

    table_id: str
    role: TableRole
    logical_artifact: LogicalArtifactIdentity
    arrow_schema_sha256: str
    columns: tuple[ColumnContract, ...]
    primary_key_column_ids: tuple[str, ...]
    foreign_keys: tuple[ForeignKeyContract, ...]
    independent_unit_column_ids: tuple[str, ...]
    row_count: int
    independent_unit_count: int
    outcome_access: OutcomeAccess
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.table_id, field_name="table_id")
        validate_sha256(self.arrow_schema_sha256, field_name="arrow_schema_sha256")
        require_sorted_unique_ids(self.columns, attribute="column_id", field_name="columns")
        if not self.columns:
            raise ValueError("table contract requires columns")
        known_columns = {column.column_id for column in self.columns}
        for name, values in (
            ("primary_key_column_ids", self.primary_key_column_ids),
            ("independent_unit_column_ids", self.independent_unit_column_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            if not set(values).issubset(known_columns):
                raise ValueError(f"{name} references unknown columns")
        require_sorted_unique_ids(
            self.foreign_keys,
            attribute="foreign_key_id",
            field_name="foreign_keys",
        )
        if self.row_count < 0 or self.independent_unit_count < 0:
            raise ValueError("table counts must be nonnegative")
        if self.independent_unit_count > self.row_count:
            raise ValueError("independent-unit count cannot exceed nested row count")
        if self.logical_artifact.profile not in {
            ArtifactProfile.ARROW_IPC,
            ArtifactProfile.PARQUET,
        }:
            raise ValueError("canonical tables require Arrow IPC or Parquet")
        required = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(required):
            raise ValueError("table visibility cannot be lowered from its lineage")
        if self.logical_artifact.visibility_ceiling is not self.visibility_ceiling:
            raise ValueError("table and logical-artifact visibility differ")


@dataclass(frozen=True, slots=True)
class CanonicalBundleManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/canonical-bundle-manifest'

    bundle_id: str
    world_fingerprint: str
    system_fingerprint: str
    relation_fingerprint: str
    dataset_version_id: str
    claim_fingerprints: tuple[str, ...]
    experiment_fingerprint: str | None
    numerical_view_fingerprints: tuple[str, ...]
    model_set_fingerprint: str | None
    observation_operator_id: str
    authority_policy_fingerprint: str
    authorization_record_fingerprint: str | None
    outcome_access: OutcomeAccess
    information_cutoff_id: str
    parent_visibility_ceilings: tuple[VisibilityCeiling, ...]
    visibility_ceiling: VisibilityCeiling
    tables: tuple[TableContract, ...]
    lineage_artifact_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("bundle_id", self.bundle_id),
            ("dataset_version_id", self.dataset_version_id),
            ("observation_operator_id", self.observation_operator_id),
            ("information_cutoff_id", self.information_cutoff_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, value in (
            ("world_fingerprint", self.world_fingerprint),
            ("system_fingerprint", self.system_fingerprint),
            ("relation_fingerprint", self.relation_fingerprint),
            ("authority_policy_fingerprint", self.authority_policy_fingerprint),
        ):
            validate_sha256(value, field_name=name)
        for name, values in (
            ("claim_fingerprints", self.claim_fingerprints),
            ("numerical_view_fingerprints", self.numerical_view_fingerprints),
        ):
            require_sorted_unique_strings(values, field_name=name)
            for value in values:
                validate_sha256(value, field_name=name)
        optional_fingerprints: tuple[tuple[str, str | None], ...] = (
            ("experiment_fingerprint", self.experiment_fingerprint),
            ("model_set_fingerprint", self.model_set_fingerprint),
            ("authorization_record_fingerprint", self.authorization_record_fingerprint),
        )
        for optional_name, optional_value in optional_fingerprints:
            if optional_value is not None:
                validate_sha256(optional_value, field_name=optional_name)
        require_sorted_unique_ids(self.tables, attribute="table_id", field_name="tables")
        if not self.tables:
            raise ValueError("canonical bundle requires at least one table")
        table_ids = {table.table_id for table in self.tables}
        for table in self.tables:
            for foreign_key in table.foreign_keys:
                if foreign_key.target_table_id not in table_ids:
                    raise ValueError("bundle foreign key references an unknown table")
        require_sorted_unique_strings(
            self.lineage_artifact_ids,
            field_name="lineage_artifact_ids",
        )
        required = inherited_visibility(self.parent_visibility_ceilings, self.outcome_access)
        if not self.visibility_ceiling.is_at_least_as_restrictive_as(required):
            raise ValueError("bundle visibility cannot be lowered from its lineage")
        if any(table.visibility_ceiling is not self.visibility_ceiling for table in self.tables):
            raise ValueError("bundle tables must inherit the exact bundle visibility")

    def table(self, role: TableRole) -> TableContract:
        matches = tuple(table for table in self.tables if table.role is role)
        if len(matches) != 1:
            raise KeyError(f"bundle has {len(matches)} tables for role {role.value}")
        return matches[0]

"""Static current source selectors and complete native-cell denominators."""

from dataclasses import dataclass, field, fields
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.matrix_inputs import MatrixAllocation
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id


@dataclass(frozen=True, slots=True)
class MatrixPreparationSourceProtocol(CanonicalRecord):
    """Native source physics, independent of any downstream training population."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/matrix-preparation-source-protocol"
    prefix_ticks: int = 4096
    preparation_ticks: int = 400
    response_ticks: int = 192
    sample_stride: int = 16
    numerical_views: tuple[int, ...] = (1, 2)
    nested_futures: int = 2
    timestep: Decimal = Decimal("0.001")
    preparation_menus: int = 9
    response_words_including_hold: int = 9
    source_kinds: tuple[str, ...] = ("q2", "cir1")

    def __post_init__(self) -> None:
        if any(type(getattr(self, item.name)) is not type(item.default) or getattr(self, item.name) != item.default for item in fields(self)):
            raise ValueError("Matrix preparation source protocol changes its owned native mechanics")


@dataclass(frozen=True, slots=True)
class MatrixPreparationSourceConfig(CanonicalRecord):
    """Editable current roster selection; no environment, private path or F fit."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/matrix-preparation-source-config"
    VERSION: ClassVar[str] = "2.0.0"
    config_id: str
    allocation: ObjectIdentity
    protocol: MatrixPreparationSourceProtocol
    code_sources_sha256: str
    dependency_lock_sha256: str = field(kw_only=True)
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(self.code_sources_sha256)
        validate_sha256(self.dependency_lock_sha256)
        if self.allocation.object_schema != MatrixAllocation.SCHEMA or self.protocol != MatrixPreparationSourceProtocol() or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE":
            raise ValueError("Matrix preparation selector changes its current allocation or evidence role")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.config_id, self)


@dataclass(frozen=True, slots=True)
class CurrentPreparationSourceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/current-preparation-source-config"
    VERSION: ClassVar[str] = "2.0.0"
    config_id: str
    allocation: ObjectIdentity
    original_f: ObjectIdentity
    protocol: ObjectIdentity
    code_sources_sha256: str
    dependency_lock_sha256: str = field(kw_only=True)
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_sha256(self.code_sources_sha256)
        validate_sha256(self.dependency_lock_sha256)
        if (
            self.allocation.object_schema != MatrixAllocation.SCHEMA
            or self.original_f.object_schema != "empirical-lawhood/methods/finite-response-law/original-finite-response-law"
            or self.protocol.object_schema != "empirical-lawhood/methods/finite-response-law/transient-bridge-spec"
            or any(identity.object_version != "1.0.0" for identity in (self.allocation, self.original_f, self.protocol))
            or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
        ):
            raise ValueError("Current preparation source changes its typed inputs or development evidence role")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.config_id, self)


@dataclass(frozen=True, slots=True)
class CurrentPreparationNativeCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/current-preparation-native-cell"
    cell_id: str
    disposition: str
    artifact: ArtifactIdentity | None
    reason: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.disposition not in ("COMPLETE", "NUMERICAL_FAILURE", "OBSERVATION_FAILURE", "UNENTERED") or (self.reason is None) != (self.disposition == "COMPLETE") or (self.artifact is None) != (self.disposition == "UNENTERED"):
            raise ValueError("Current preparation cell loses its accepted, failed or unentered state")


def current_preparation_cell_ids(root_id: str) -> tuple[str, ...]:
    return tuple(sorted(
        [f"{root_id}.prefix.r{view}" for view in (1, 2)]
        + [f"{root_id}.parent.s{schedule}.r{view}" for schedule in range(9) for view in (1, 2)]
        + [f"{root_id}.future.s{schedule}.r{view}.f{future}.w{word}" for schedule in range(9) for view in (1, 2) for future in range(2) for word in range(9)]
    ))


@dataclass(frozen=True, slots=True)
class CurrentPreparationRootReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/current-preparation-root-report"
    root_id: str
    allocation: ObjectIdentity
    source: ObjectIdentity
    cells: tuple[CurrentPreparationNativeCell, ...]
    disposition: str
    reason: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.root_id, field_name="root_id")
        if (
            tuple(cell.cell_id for cell in self.cells) != current_preparation_cell_ids(self.root_id)
            or self.allocation.object_schema != "empirical-lawhood/kernel/matrix-root-allocation"
            or self.source.object_schema not in (CurrentPreparationSourceConfig.SCHEMA, MatrixPreparationSourceConfig.SCHEMA)
            or self.disposition not in ("COMPLETE", "UNEVALUABLE")
            or (self.reason is None) != (self.disposition == "COMPLETE")
            or self.disposition == "COMPLETE" and any(cell.disposition != "COMPLETE" for cell in self.cells)
        ):
            raise ValueError("Current preparation report changes its complete native denominator or lineage")


@dataclass(frozen=True, slots=True)
class CurrentPreparationSourceReport(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/current-preparation-source-report"
    report_id: str
    source: ObjectIdentity
    allocation: ObjectIdentity
    original_f: ObjectIdentity | None
    root_reports: tuple[ArtifactIdentity, ...]
    disposition: str
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self) -> None:
        validate_stable_id(self.report_id, field_name="report_id")
        if not 0 < len(self.root_reports) <= 128 or len({artifact.artifact_id for artifact in self.root_reports}) != len(self.root_reports) or any(artifact.payload_schema != CurrentPreparationRootReport.SCHEMA for artifact in self.root_reports) or self.source.object_schema not in (CurrentPreparationSourceConfig.SCHEMA, MatrixPreparationSourceConfig.SCHEMA) or self.allocation.object_schema != MatrixAllocation.SCHEMA or self.disposition not in ("COMPLETE", "UNEVALUABLE") or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE":
            raise ValueError("Current source report changes its root denominator or development role")
        fixed = self.source.object_schema == CurrentPreparationSourceConfig.SCHEMA
        if (fixed and (len(self.root_reports) != 24 or self.original_f is None)
            or not fixed and self.original_f is not None
            or self.original_f is not None and (
                self.original_f.object_schema != "empirical-lawhood/methods/finite-response-law/original-finite-response-law"
                or self.original_f.object_version != "1.0.0")):
            raise ValueError("Current source report changes its fixed24 original-law binding")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.report_id, self)

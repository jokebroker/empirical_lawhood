"""Exact historical-generation lineage for held simulator run bundles."""

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
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.worlds import WorldKind
from empirical_lawhood.planning.dataset_manifests import DatasetDirectorySelectorManifest
from empirical_lawhood.planning.datasets import DatasetMaterialization


class SimulatorGenerationKind(StrEnum):
    ACQUISITION = "ACQUISITION"
    CAMPAIGN = "CAMPAIGN"
    DERIVATIVE_ANALYSIS = "DERIVATIVE_ANALYSIS"
    NUMERICAL_PILOT = "NUMERICAL_PILOT"
    PREPARATION_PILOT = "PREPARATION_PILOT"
    REPAIR_CAMPAIGN = "REPAIR_CAMPAIGN"


class SimulatorGenerationEvidenceRole(StrEnum):
    ACTION_CLOCKS = "ACTION_CLOCKS"
    CONFIG = "CONFIG"
    DENOMINATOR = "DENOMINATOR"
    NUMERICAL_VIEW = "NUMERICAL_VIEW"
    RECEIPT = "RECEIPT"
    RUNTIME = "RUNTIME"
    SOLVER = "SOLVER"


class SimulatorGenerationExceptionCode(StrEnum):
    ACTION_CLOCK_METADATA_UNAVAILABLE = "ACTION_CLOCK_METADATA_UNAVAILABLE"
    INDEPENDENT_UNIT_METADATA_UNAVAILABLE = "INDEPENDENT_UNIT_METADATA_UNAVAILABLE"
    NUMERICAL_VIEW_METADATA_UNAVAILABLE = "NUMERICAL_VIEW_METADATA_UNAVAILABLE"
    RUNTIME_METADATA_UNAVAILABLE = "RUNTIME_METADATA_UNAVAILABLE"
    SOLVER_METADATA_UNAVAILABLE = "SOLVER_METADATA_UNAVAILABLE"


@dataclass(frozen=True, slots=True)
class SimulatorGenerationEvidenceExpectation(CanonicalRecord):
    """One exact selected member carrying generation-denominator evidence."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/simulator-generation-evidence-expectation'

    evidence_id: str
    role: SimulatorGenerationEvidenceRole
    relative_locator: str
    expected_physical_sha256: str
    expected_size_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        if not isinstance(self.role, SimulatorGenerationEvidenceRole):
            raise ValueError("role must be a SimulatorGenerationEvidenceRole")
        validate_relative_locator(self.relative_locator)
        validate_sha256(
            self.expected_physical_sha256,
            field_name="expected_physical_sha256",
        )
        if (
            isinstance(self.expected_size_bytes, bool)
            or not isinstance(self.expected_size_bytes, int)
            or self.expected_size_bytes < 0
        ):
            raise ValueError("expected_size_bytes must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class SimulatorGenerationLineageManifest(CanonicalRecord):
    """Manifest-bound lineage for one immutable historical simulation generation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/simulator-generation-lineage-manifest'

    lineage_id: str
    generation_id: str
    source_run_id: str
    proposed_materialization_id: str
    generation_kind: SimulatorGenerationKind
    evidence_world: WorldKind
    parent_materializations: tuple[ObjectIdentity, ...]
    evidence_records: tuple[SimulatorGenerationEvidenceExpectation, ...]
    logical_artifact_count: int
    independent_unit_id: str | None
    nested_unit_ids: tuple[str, ...]
    solver_id: str | None
    runtime_id: str | None
    numerical_view_id: str | None
    action_clock_id: str | None
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    exception_codes: tuple[SimulatorGenerationExceptionCode, ...]

    def __post_init__(self) -> None:
        for field_name, value in (
            ("lineage_id", self.lineage_id),
            ("generation_id", self.generation_id),
            ("source_run_id", self.source_run_id),
            ("proposed_materialization_id", self.proposed_materialization_id),
        ):
            validate_stable_id(value, field_name=field_name)
        if not isinstance(self.generation_kind, SimulatorGenerationKind):
            raise ValueError("generation_kind must be a SimulatorGenerationKind")
        if self.evidence_world is not WorldKind.NUMERICAL_SIMULATOR:
            raise ValueError("held simulator generations must retain NUMERICAL_SIMULATOR world")
        if not isinstance(self.parent_materializations, tuple) or not (
            self.parent_materializations
        ):
            raise ValueError("generated lineage requires exact parent materializations")
        if any(
            not isinstance(value, ObjectIdentity)
            or value.object_schema != DatasetMaterialization.SCHEMA
            for value in self.parent_materializations
        ):
            raise ValueError("lineage parents must identify DatasetMaterialization records")
        parent_ids = tuple(value.object_id for value in self.parent_materializations)
        if tuple(sorted(set(parent_ids))) != parent_ids:
            raise ValueError("parent_materializations must have sorted unique object IDs")
        if not isinstance(self.evidence_records, tuple) or not self.evidence_records:
            raise ValueError("generated lineage requires selected denominator evidence")
        if any(
            not isinstance(value, SimulatorGenerationEvidenceExpectation)
            for value in self.evidence_records
        ):
            raise ValueError("evidence_records contains another record type")
        require_sorted_unique_ids(
            self.evidence_records,
            attribute="evidence_id",
            field_name="evidence_records",
        )
        locators = tuple(value.relative_locator for value in self.evidence_records)
        if len(set(locators)) != len(locators):
            raise ValueError("generation evidence cannot repeat a selected locator")
        if (
            isinstance(self.logical_artifact_count, bool)
            or not isinstance(self.logical_artifact_count, int)
            or self.logical_artifact_count <= 0
        ):
            raise ValueError("logical_artifact_count must be a positive integer")
        for field_name, value in (
            ("independent_unit_id", self.independent_unit_id),
            ("solver_id", self.solver_id),
            ("runtime_id", self.runtime_id),
            ("numerical_view_id", self.numerical_view_id),
            ("action_clock_id", self.action_clock_id),
        ):
            if value is not None:
                validate_stable_id(value, field_name=field_name)
        require_sorted_unique_strings(
            self.nested_unit_ids,
            field_name="nested_unit_ids",
            allow_empty=True,
        )
        for value in self.nested_unit_ids:
            validate_stable_id(value, field_name="nested_unit_ids")
        if not isinstance(self.outcome_access, OutcomeAccess):
            raise ValueError("outcome_access must be an OutcomeAccess")
        if not isinstance(self.visibility_ceiling, VisibilityCeiling):
            raise ValueError("visibility_ceiling must be a VisibilityCeiling")
        if not isinstance(self.exception_codes, tuple) or any(
            not isinstance(value, SimulatorGenerationExceptionCode)
            for value in self.exception_codes
        ):
            raise ValueError("exception_codes contains another code type")
        encoded_codes = tuple(value.value for value in self.exception_codes)
        if tuple(sorted(set(encoded_codes))) != encoded_codes:
            raise ValueError("exception_codes must be sorted and unique")
        implied = (
            (
                SimulatorGenerationExceptionCode.INDEPENDENT_UNIT_METADATA_UNAVAILABLE,
                self.independent_unit_id is None,
            ),
            (SimulatorGenerationExceptionCode.SOLVER_METADATA_UNAVAILABLE, self.solver_id is None),
            (
                SimulatorGenerationExceptionCode.RUNTIME_METADATA_UNAVAILABLE,
                self.runtime_id is None,
            ),
            (
                SimulatorGenerationExceptionCode.NUMERICAL_VIEW_METADATA_UNAVAILABLE,
                self.numerical_view_id is None,
            ),
            (
                SimulatorGenerationExceptionCode.ACTION_CLOCK_METADATA_UNAVAILABLE,
                self.action_clock_id is None,
            ),
        )
        codes = frozenset(self.exception_codes)
        for code, is_implied in implied:
            if (code in codes) != is_implied:
                raise ValueError(f"structural condition and {code.value} must agree")

    def validate_selected_evidence(
        self,
        selector: DatasetDirectorySelectorManifest,
    ) -> None:
        """Require every lineage fact to be guarded by the exact directory selector."""

        if not isinstance(selector, DatasetDirectorySelectorManifest):
            raise TypeError("selector must be a DatasetDirectorySelectorManifest")
        selected = {
            value.relative_locator: (value.expected_physical_sha256, value.expected_size_bytes)
            for value in selector.members
        }
        for evidence in self.evidence_records:
            if selected.get(evidence.relative_locator) != (
                evidence.expected_physical_sha256,
                evidence.expected_size_bytes,
            ):
                raise ValueError("generation evidence is absent from the exact guarded selector")


__all__ = [
    "SimulatorGenerationEvidenceExpectation",
    "SimulatorGenerationEvidenceRole",
    "SimulatorGenerationExceptionCode",
    "SimulatorGenerationKind",
    "SimulatorGenerationLineageManifest",
]

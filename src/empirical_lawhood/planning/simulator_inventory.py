"""Bounded authoring of literal selectors for exact held simulator trees."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_stable_id,
)
from empirical_lawhood.planning.dataset_manifests import (
    MAX_DATASET_SELECTOR_MEMBERS,
    DatasetDirectorySelectorManifest,
)


_MAX_SIMULATOR_SOURCE_BYTES = 8 * 1024**3


class SimulatorInventorySelectionKind(StrEnum):
    COMPLETE_REGULAR_TREE = "COMPLETE_REGULAR_TREE"
    GIT_TRACKED_TREE = "GIT_TRACKED_TREE"
    LITERAL_FILES = "LITERAL_FILES"


@dataclass(frozen=True, slots=True)
class SimulatorHeldInventorySpec(CanonicalRecord):
    """Code-owned bounded inventory request; never an arbitrary glob or command."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/simulator-held-inventory-spec'

    inventory_id: str
    selector_id: str
    subject_id: str
    source_relative_prefix: str
    selection_kind: SimulatorInventorySelectionKind
    literal_relative_locators: tuple[str, ...]
    expected_git_commit: str | None
    complete_subject: bool
    maximum_files: int
    maximum_bytes: int

    def __post_init__(self) -> None:
        for field_name, value in (
            ("inventory_id", self.inventory_id),
            ("selector_id", self.selector_id),
            ("subject_id", self.subject_id),
        ):
            validate_stable_id(value, field_name=field_name)
        validate_relative_locator(self.source_relative_prefix)
        if not isinstance(self.selection_kind, SimulatorInventorySelectionKind):
            raise ValueError("selection_kind has another type")
        require_sorted_unique_strings(
            self.literal_relative_locators,
            field_name="literal_relative_locators",
            allow_empty=True,
        )
        for value in self.literal_relative_locators:
            validate_relative_locator(value)
            if any(character in value for character in "*?[]{}"):
                raise ValueError("literal inventory locators cannot contain globs")
        if self.selection_kind is SimulatorInventorySelectionKind.LITERAL_FILES:
            if not self.literal_relative_locators or self.expected_git_commit is not None:
                raise ValueError("literal inventory requires only literal file locators")
        elif self.selection_kind is SimulatorInventorySelectionKind.GIT_TRACKED_TREE:
            if self.literal_relative_locators or self.expected_git_commit is None:
                raise ValueError("tracked inventory requires only an exact Git commit")
            if len(self.expected_git_commit) != 40 or any(
                value not in "0123456789abcdef" for value in self.expected_git_commit
            ):
                raise ValueError("expected_git_commit must be a lowercase Git SHA-1")
        elif self.literal_relative_locators or self.expected_git_commit is not None:
            raise ValueError("complete tree inventory cannot add locator/Git selectors")
        if not isinstance(self.complete_subject, bool):
            raise ValueError("complete_subject must be boolean")
        if (
            isinstance(self.maximum_files, bool)
            or not isinstance(self.maximum_files, int)
            or not 1 <= self.maximum_files <= MAX_DATASET_SELECTOR_MEMBERS
        ):
            raise ValueError("maximum_files exceeds the directory-selector bound")
        if (
            isinstance(self.maximum_bytes, bool)
            or not isinstance(self.maximum_bytes, int)
            or not 1 <= self.maximum_bytes <= _MAX_SIMULATOR_SOURCE_BYTES
        ):
            raise ValueError("maximum_bytes exceeds the simulator source bound")


@dataclass(frozen=True, slots=True)
class SimulatorDirectoryInventoryAudit(CanonicalRecord):
    """Non-writing inventory result; registration performs the evidence hash pass."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/simulator-directory-inventory-audit'

    audit_id: str
    inventory_spec: ObjectIdentity
    selector: ObjectIdentity
    observed_files: int
    observed_bytes: int
    observed_git_commit: str | None
    source_mutated: bool
    network_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        if (
            not isinstance(self.inventory_spec, ObjectIdentity)
            or self.inventory_spec.object_schema != SimulatorHeldInventorySpec.SCHEMA
        ):
            raise ValueError("inventory_spec must identify a held inventory spec")
        if (
            not isinstance(self.selector, ObjectIdentity)
            or self.selector.object_schema != DatasetDirectorySelectorManifest.SCHEMA
        ):
            raise ValueError("selector must identify a directory selector")
        for field_name, value in (
            ("observed_files", self.observed_files),
            ("observed_bytes", self.observed_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if self.observed_git_commit is not None and (
            len(self.observed_git_commit) != 40
            or any(value not in "0123456789abcdef" for value in self.observed_git_commit)
        ):
            raise ValueError("observed_git_commit must be a lowercase Git SHA-1")
        if self.source_mutated is not False or self.network_bytes != 0:
            raise ValueError("inventory authoring must be read-only and zero-network")


class SimulatorDirectoryInventoryAuthor(Protocol):
    """Infrastructure port for bounded, non-writing selector authoring."""

    def author(
        self,
        spec: SimulatorHeldInventorySpec,
    ) -> tuple[DatasetDirectorySelectorManifest, SimulatorDirectoryInventoryAudit]: ...


__all__ = [
    "SimulatorDirectoryInventoryAudit",
    "SimulatorDirectoryInventoryAuthor",
    "SimulatorHeldInventorySpec",
    "SimulatorInventorySelectionKind",
]

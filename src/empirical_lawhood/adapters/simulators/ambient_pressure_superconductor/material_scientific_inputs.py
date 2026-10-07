"""Complete original material rank/order inputs bound to the current graph.

Shape and current-binding validation do not authenticate an export or transfer
historical qualification. Callers authenticate original source/export bytes
separately before supplying the complete original numeric ordering census.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    validate_sha256,
    validate_stable_id,
)


def _custody(original_source: ArtifactIdentity, export_receipt: ArtifactIdentity) -> None:
    if not isinstance(original_source, ArtifactIdentity) or not isinstance(export_receipt, ArtifactIdentity):
        raise ValueError("material scientific input requires original-source and numeric-export artifact custody")
    if original_source.size_bytes <= 0 or export_receipt.size_bytes <= 0:
        raise ValueError("material scientific input requires nonempty original source and export bytes")


def _index(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("material scientific order index must be a nonnegative integer")


@dataclass(frozen=True, slots=True)
class MaterialExplorationActionRank(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/material/exploration-action-scientific-rank"

    current_action_id: str
    scientific_order_index: int
    full_original_random_key_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.current_action_id, field_name="current_action_id")
        _index(self.scientific_order_index)
        validate_sha256(self.full_original_random_key_sha256, field_name="full_original_random_key_sha256")


@dataclass(frozen=True, slots=True)
class MaterialExplorationScientificInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/material/exploration-scientific-input"

    current_candidates_sha256: str
    current_history_sha256: str
    original_candidates_sha256: str
    original_history_sha256: str
    wave_index: int
    budget_operands: tuple[int, ...]
    action_ranks: tuple[MaterialExplorationActionRank, ...]
    original_source: ArtifactIdentity
    export_receipt: ArtifactIdentity

    def __post_init__(self) -> None:
        for name in ("current_candidates_sha256", "current_history_sha256", "original_candidates_sha256", "original_history_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if type(self.wave_index) is not int or self.wave_index <= 0:
            raise ValueError("material scientific wave index must be positive")
        if type(self.budget_operands) is not tuple or len(self.budget_operands) != 5 or any(
            type(value) is not int or value < 0 for value in self.budget_operands
        ):
            raise ValueError("material scientific input requires all five original selection-budget operands in fixed order")
        if type(self.action_ranks) is not tuple or any(not isinstance(row, MaterialExplorationActionRank) for row in self.action_ranks):
            raise ValueError("material exploration requires a complete typed original rank census")
        if tuple(row.scientific_order_index for row in self.action_ranks) != tuple(range(len(self.action_ranks))) or len({row.current_action_id for row in self.action_ranks}) != len(self.action_ranks):
            raise ValueError("material exploration rank census must have unique actions and complete numeric order")
        _custody(self.original_source, self.export_receipt)

    def require_current_binding(self, *, candidates: tuple[CanonicalRecord, ...], history: CanonicalRecord,
                                wave_index: int, budget_operands: tuple[int, ...]) -> None:
        if (self.current_candidates_sha256, self.current_history_sha256, self.wave_index, self.budget_operands) != (
            sha256(canonical_json_bytes(candidates)).hexdigest(), history.fingerprint(), wave_index, budget_operands
        ) or {row.current_action_id for row in self.action_ranks} != {candidate.action_id for candidate in candidates}:
            raise ValueError("material exploration scientific input differs from the complete current graph/history/wave/budget custody")
        if len(self.action_ranks) != len(candidates):
            raise ValueError("material exploration requires one original numeric rank for every current candidate")


@dataclass(frozen=True, slots=True)
class MaterialAdmissionActionOrder(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/material/admission-action-scientific-order"

    current_cell_id: str
    current_action_id: str
    scientific_order_index: int

    def __post_init__(self) -> None:
        validate_stable_id(self.current_cell_id, field_name="current_cell_id")
        validate_stable_id(self.current_action_id, field_name="current_action_id")
        _index(self.scientific_order_index)


@dataclass(frozen=True, slots=True)
class MaterialAdmissionScientificInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/material/admission-scientific-input"

    current_candidates_sha256: str
    original_candidates_sha256: str
    maximum_effort_units: int
    action_orders: tuple[MaterialAdmissionActionOrder, ...]
    original_source: ArtifactIdentity
    export_receipt: ArtifactIdentity

    def __post_init__(self) -> None:
        validate_sha256(self.current_candidates_sha256, field_name="current_candidates_sha256")
        validate_sha256(self.original_candidates_sha256, field_name="original_candidates_sha256")
        if type(self.maximum_effort_units) is not int or self.maximum_effort_units <= 0:
            raise ValueError("material admission scientific input requires a positive effort limit")
        if type(self.action_orders) is not tuple or any(not isinstance(row, MaterialAdmissionActionOrder) for row in self.action_orders):
            raise ValueError("material admission requires a complete typed original pair-order census")
        if tuple(row.scientific_order_index for row in self.action_orders) != tuple(range(len(self.action_orders))) or len({(row.current_cell_id, row.current_action_id) for row in self.action_orders}) != len(self.action_orders):
            raise ValueError("material admission order census must have unique cell/action pairs and complete numeric order")
        _custody(self.original_source, self.export_receipt)

    def require_current_binding(self, *, candidates: tuple[CanonicalRecord, ...], maximum_effort_units: int) -> None:
        if (self.current_candidates_sha256, self.maximum_effort_units) != (
            sha256(canonical_json_bytes(candidates)).hexdigest(), maximum_effort_units
        ) or {(row.current_cell_id, row.current_action_id) for row in self.action_orders} != {(candidate.cell_id, candidate.action_id) for candidate in candidates}:
            raise ValueError("material admission scientific input differs from the complete current candidates/effort custody")
        if len(self.action_orders) != len(candidates):
            raise ValueError("material admission requires one original numeric order for every current candidate")


def require_material_exploration_scientific_input(value: MaterialExplorationScientificInput | None, *,
                                                  candidates: tuple[CanonicalRecord, ...], history: CanonicalRecord,
                                                  wave_index: int, budget_operands: tuple[int, ...]
                                                  ) -> MaterialExplorationScientificInput:
    if not isinstance(value, MaterialExplorationScientificInput):
        raise ValueError("material exploration requires an authenticated complete original numeric rank input before selection")
    value.require_current_binding(candidates=candidates, history=history, wave_index=wave_index, budget_operands=budget_operands)
    return value


def require_material_admission_scientific_input(value: MaterialAdmissionScientificInput | None, *,
                                                candidates: tuple[CanonicalRecord, ...], maximum_effort_units: int
                                                ) -> MaterialAdmissionScientificInput:
    if not isinstance(value, MaterialAdmissionScientificInput):
        raise ValueError("material admission requires an authenticated complete original numeric pair-order input before selection")
    value.require_current_binding(candidates=candidates, maximum_effort_units=maximum_effort_units)
    return value

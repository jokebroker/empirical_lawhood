"""Explicit original numeric inputs with separate current descriptor custody.

Input shape and target bindings do not authenticate an external export or
transfer a historical qualification. Callers must authenticate original source
and export bytes separately before supplying these records.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id


def _programme(value: int) -> None:
    if type(value) is not int or value not in (0, 1, 2):
        raise ValueError("scientific input requires an exact history programme ordinal")


def _custody(original_source: ArtifactIdentity, export_receipt: ArtifactIdentity) -> None:
    if not isinstance(original_source, ArtifactIdentity) or not isinstance(export_receipt, ArtifactIdentity):
        raise ValueError("scientific input requires original-source and numeric-export artifact custody")
    if original_source.size_bytes <= 0 or export_receipt.size_bytes <= 0:
        raise ValueError("scientific input custody must identify nonempty original and export bytes")


@dataclass(frozen=True, slots=True)
class HistoryBudgetDescriptorScientificInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/history-budget/descriptor-scientific-input"

    programme_ordinal: int
    unit_id: str
    scale_cells: int
    source_seed_sha256: str
    original_descriptor_sha256: str
    current_descriptor_sha256: str
    history_fibre_seed_sha256s: tuple[str, ...]
    original_source: ArtifactIdentity
    export_receipt: ArtifactIdentity

    def __post_init__(self) -> None:
        _programme(self.programme_ordinal)
        validate_stable_id(self.unit_id, field_name="unit_id")
        if type(self.scale_cells) is not int or self.scale_cells not in (16, 32, 64, 128, 256):
            raise ValueError("scientific descriptor scale lies outside the complete source roster")
        for name in ("source_seed_sha256", "original_descriptor_sha256", "current_descriptor_sha256"):
            validate_sha256(getattr(self, name), field_name=name)
        if type(self.history_fibre_seed_sha256s) is not tuple or len(self.history_fibre_seed_sha256s) != 36:
            raise ValueError("scientific descriptor requires the complete ordered depth-zero-through-35 seed census")
        for seed in self.history_fibre_seed_sha256s:
            validate_sha256(seed, field_name="history_fibre_seed_sha256s")
        _custody(self.original_source, self.export_receipt)

    def require_current_binding(self, *, programme_ordinal: int, unit_id: str, scale_cells: int,
                                source_seed_sha256: str, current_descriptor_sha256: str) -> None:
        if (self.programme_ordinal, self.unit_id, self.scale_cells, self.source_seed_sha256,
            self.current_descriptor_sha256) != (programme_ordinal, unit_id, scale_cells,
                                                source_seed_sha256, current_descriptor_sha256):
            raise ValueError("scientific descriptor input differs from the exact current target custody")

    def fibre_seed(self, depth: int) -> bytes:
        if type(depth) is not int or not 0 <= depth < 36:
            raise ValueError("scientific history depth lies outside the complete supplied seed census")
        return bytes.fromhex(self.history_fibre_seed_sha256s[depth])


@dataclass(frozen=True, slots=True)
class HistoryBudgetPreparationScientificInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/history-budget/preparation-scientific-input"

    programme_ordinal: int
    unit_id: str
    preparation_substream_seed_hex: str
    sampler_full_seed_sha256: str
    current_descriptor_sha256s: tuple[str, ...]
    original_source: ArtifactIdentity
    export_receipt: ArtifactIdentity

    def __post_init__(self) -> None:
        _programme(self.programme_ordinal)
        if self.programme_ordinal == 0:
            raise ValueError("simulator morphism challenges do not have untouched-preparation inputs")
        validate_stable_id(self.unit_id, field_name="unit_id")
        validate_sha256(self.preparation_substream_seed_hex, field_name="preparation_substream_seed_hex")
        validate_sha256(self.sampler_full_seed_sha256, field_name="sampler_full_seed_sha256")
        if type(self.current_descriptor_sha256s) is not tuple or len(set(self.current_descriptor_sha256s)) != len(self.current_descriptor_sha256s):
            raise ValueError("preparation scientific input requires an exact ordered current descriptor census")
        for digest in self.current_descriptor_sha256s:
            validate_sha256(digest, field_name="current_descriptor_sha256s")
        _custody(self.original_source, self.export_receipt)

    def require_current_binding(self, *, programme_ordinal: int, unit_id: str, seed: bytes) -> None:
        if (self.programme_ordinal, self.unit_id, self.preparation_substream_seed_hex) != (
            programme_ordinal, unit_id, seed.hex()
        ):
            raise ValueError("preparation scientific input differs from the supplied unit and numeric substream")

    def require_current_descriptors(self, current_descriptor_sha256s: tuple[str, ...]) -> None:
        if self.current_descriptor_sha256s != current_descriptor_sha256s:
            raise ValueError("preparation scientific input differs from the complete current descriptor custody census")


@dataclass(frozen=True, slots=True)
class HistoryBudgetUnitScientificInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/history-budget/unit-scientific-input"

    programme_ordinal: int
    unit_id: str
    source_seed_sha256: str
    descriptor_inputs: tuple[HistoryBudgetDescriptorScientificInput, ...]
    preparation_input: HistoryBudgetPreparationScientificInput | None

    def __post_init__(self) -> None:
        _programme(self.programme_ordinal)
        validate_stable_id(self.unit_id, field_name="unit_id")
        validate_sha256(self.source_seed_sha256, field_name="source_seed_sha256")
        if type(self.descriptor_inputs) is not tuple or not self.descriptor_inputs or any(
            not isinstance(row, HistoryBudgetDescriptorScientificInput) for row in self.descriptor_inputs
        ):
            raise ValueError("unit scientific input requires a complete typed descriptor census")
        scales = tuple(row.scale_cells for row in self.descriptor_inputs)
        if scales != tuple(sorted(set(scales))):
            raise ValueError("unit scientific descriptor scales must be unique and in fixed numeric order")
        first = self.descriptor_inputs[0]
        if any((row.programme_ordinal, row.unit_id, row.source_seed_sha256, row.original_source,
                row.export_receipt) != (self.programme_ordinal, self.unit_id, self.source_seed_sha256,
                                       first.original_source, first.export_receipt)
               for row in self.descriptor_inputs):
            raise ValueError("unit scientific descriptor inputs differ in original/current unit or export custody")
        if self.programme_ordinal == 0:
            if self.preparation_input is not None:
                raise ValueError("simulator morphism input cannot contain an untouched-preparation stream")
        elif not isinstance(self.preparation_input, HistoryBudgetPreparationScientificInput) or (
            self.preparation_input.programme_ordinal, self.preparation_input.unit_id,
            self.preparation_input.original_source, self.preparation_input.export_receipt
        ) != (self.programme_ordinal, self.unit_id, first.original_source, first.export_receipt):
            raise ValueError("unit scientific input requires matching untouched-preparation numeric/export custody")
        if self.preparation_input is not None:
            self.preparation_input.require_current_descriptors(tuple(row.current_descriptor_sha256 for row in self.descriptor_inputs))

    def require_current_binding(self, *, programme_ordinal: int, unit_id: str, seed: bytes,
                                scale_cells: tuple[int, ...]) -> None:
        if type(seed) is not bytes or len(seed) != 32 or (
            self.programme_ordinal, self.unit_id, self.source_seed_sha256,
            tuple(row.scale_cells for row in self.descriptor_inputs)
        ) != (programme_ordinal, unit_id, sha256(seed).hexdigest(), scale_cells):
            raise ValueError("unit scientific input differs from the complete current source-seed/scale roster")

    def descriptor_input(self, current_descriptor_sha256: str) -> HistoryBudgetDescriptorScientificInput:
        rows = tuple(row for row in self.descriptor_inputs if row.current_descriptor_sha256 == current_descriptor_sha256)
        if len(rows) != 1:
            raise ValueError("current descriptor is absent from the supplied original numeric census")
        return rows[0]


def require_history_budget_descriptor_input(value: HistoryBudgetDescriptorScientificInput | None, *,
                                            programme_ordinal: int, unit_id: str, scale_cells: int,
                                            source_seed_sha256: str, current_descriptor_sha256: str
                                            ) -> HistoryBudgetDescriptorScientificInput:
    if not isinstance(value, HistoryBudgetDescriptorScientificInput):
        raise ValueError("history observation requires an authenticated complete original numeric input and current target binding")
    value.require_current_binding(programme_ordinal=programme_ordinal, unit_id=unit_id,
                                  scale_cells=scale_cells, source_seed_sha256=source_seed_sha256,
                                  current_descriptor_sha256=current_descriptor_sha256)
    return value


def require_history_budget_unit_inputs(value: tuple[HistoryBudgetUnitScientificInput, ...] | None, *,
                                     programme_ordinal: int, unit_ids: tuple[str, ...],
                                     scale_cells: tuple[int, ...]) -> tuple[HistoryBudgetUnitScientificInput, ...]:
    if value is None and not unit_ids:
        return ()
    if type(value) is not tuple or any(not isinstance(row, HistoryBudgetUnitScientificInput) for row in value):
        raise ValueError("history runtime requires an authenticated complete original numeric unit census before scientific work")
    if tuple(row.unit_id for row in value) != unit_ids or any(
        row.programme_ordinal != programme_ordinal or tuple(d.scale_cells for d in row.descriptor_inputs) != scale_cells
        for row in value
    ):
        raise ValueError("history runtime scientific inputs differ from the exact ordered current unit and scale census")
    return value

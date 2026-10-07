"""Board-, batch- and leave-one-board morphism heterogeneity panels."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismBoardLocalResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-board-local-result'

    result_id: str
    board_id: str
    batch_id: str
    categorical_terminal: str
    admitted_fraction: Decimal
    boundary_signature_ids: tuple[str, ...]
    active_gate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in ("result_id", "board_id", "batch_id", "categorical_terminal"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_decimal(self.admitted_fraction, field_name="admitted_fraction", minimum=Decimal(0))
        if self.admitted_fraction > 1:
            raise ValueError("admitted fraction lies outside [0, 1]")
        require_sorted_unique_strings(
            self.boundary_signature_ids,
            field_name="boundary_signature_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.active_gate_ids, field_name="active_gate_ids")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismLeaveOneBoardResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-leave-one-board-result'

    result_id: str
    omitted_board_id: str
    retained_terminal: str
    changed_from_full_panel: bool

    def __post_init__(self) -> None:
        for name in ("result_id", "omitted_board_id", "retained_terminal"):
            validate_stable_id(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismHeterogeneityPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-heterogeneity-panel'

    panel_id: str
    board_results: tuple[PhysicalScaleMorphismBoardLocalResult, ...]
    full_panel_terminal: str
    full_panel_terminal_supplied: bool
    leave_one_results: tuple[PhysicalScaleMorphismLeaveOneBoardResult, ...]
    leave_one_recomputed: bool
    admitted_fraction_range: Decimal
    category_heterogeneous: bool
    boundary_heterogeneous: bool
    batch_localized: bool
    deletion_unstable: bool
    mixed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.full_panel_terminal, field_name="full_panel_terminal")
        require_sorted_unique_ids(
            self.board_results, attribute="result_id", field_name="board_results"
        )
        require_sorted_unique_ids(
            self.leave_one_results,
            attribute="result_id",
            field_name="leave_one_results",
        )
        if len(self.board_results) < 2:
            raise ValueError("heterogeneity panel requires at least two boards")
        if self.leave_one_recomputed:
            if {value.board_id for value in self.board_results} != {
                value.omitted_board_id for value in self.leave_one_results
            }:
                raise ValueError("leave-one panel does not cover every board")
            if not self.full_panel_terminal_supplied:
                raise ValueError(
                    "recomputed deletion panel requires the actual full-panel terminal"
                )
            if any(
                value.changed_from_full_panel
                != (value.retained_terminal != self.full_panel_terminal)
                for value in self.leave_one_results
            ):
                raise ValueError("leave-one change flag is not full-panel derived")
        elif self.leave_one_results:
            raise ValueError("surrogate leave-one results cannot enter the panel")
        validate_decimal(
            self.admitted_fraction_range,
            field_name="admitted_fraction_range",
            minimum=Decimal(0),
        )
        expected_deletion = any(value.changed_from_full_panel for value in self.leave_one_results)
        if self.deletion_unstable != expected_deletion:
            raise ValueError("deletion instability is not data-derived")
        expected_mixed = any(
            (
                self.category_heterogeneous,
                self.boundary_heterogeneous,
                self.batch_localized,
                self.deletion_unstable,
            )
        )
        if self.mixed != expected_mixed:
            raise ValueError("heterogeneity mixed flag is not data-derived")


def _modal_terminal(values: tuple[PhysicalScaleMorphismBoardLocalResult, ...]) -> str:
    counts = Counter(value.categorical_terminal for value in values)
    largest = max(counts.values())
    return min(value for value, count in counts.items() if count == largest)


def evaluate_heterogeneity(
    *,
    panel_id: str,
    board_results: tuple[PhysicalScaleMorphismBoardLocalResult, ...],
    material_admitted_fraction_range: Decimal,
    full_panel_terminal: str | None = None,
    recomputed_leave_one_results: tuple[PhysicalScaleMorphismLeaveOneBoardResult, ...] | None = None,
) -> PhysicalScaleMorphismHeterogeneityPanel:
    require_sorted_unique_ids(board_results, attribute="result_id", field_name="board_results")
    if len({value.board_id for value in board_results}) != len(board_results):
        raise ValueError("heterogeneity panel repeats a board identity")
    validate_decimal(
        material_admitted_fraction_range,
        field_name="material_admitted_fraction_range",
        minimum=Decimal(0),
    )
    if len(board_results) < 2:
        raise ValueError("heterogeneity evaluation needs at least two boards")
    full_terminal_supplied = full_panel_terminal is not None
    full_terminal = full_panel_terminal or _modal_terminal(board_results)
    validate_stable_id(full_terminal, field_name="full_panel_terminal")
    if recomputed_leave_one_results is None:
        leave_one: tuple[PhysicalScaleMorphismLeaveOneBoardResult, ...] = ()
        leave_one_recomputed = False
    else:
        require_sorted_unique_ids(
            recomputed_leave_one_results,
            attribute="result_id",
            field_name="recomputed_leave_one_results",
        )
        if not full_terminal_supplied:
            raise ValueError("recomputed leave-one results require the actual full-panel terminal")
        leave_one = recomputed_leave_one_results
        leave_one_recomputed = True
    admitted_values = tuple(value.admitted_fraction for value in board_results)
    admitted_range = max(admitted_values) - min(admitted_values)
    category_heterogeneous = len({value.categorical_terminal for value in board_results}) > 1
    boundary_heterogeneous = (
        len({value.boundary_signature_ids for value in board_results}) > 1
        or len({value.active_gate_ids for value in board_results}) > 1
        or admitted_range > material_admitted_fraction_range
    )
    batch_groups = {
        batch_id: tuple(value for value in board_results if value.batch_id == batch_id)
        for batch_id in sorted({value.batch_id for value in board_results})
    }
    batch_terminals = {_modal_terminal(values) for values in batch_groups.values()}
    batch_means = tuple(
        sum((value.admitted_fraction for value in values), Decimal(0)) / Decimal(len(values))
        for values in batch_groups.values()
    )
    batch_localized = len(batch_groups) > 1 and (
        len(batch_terminals) > 1
        or max(batch_means) - min(batch_means) > material_admitted_fraction_range
    )
    deletion_unstable = any(value.changed_from_full_panel for value in leave_one)
    return PhysicalScaleMorphismHeterogeneityPanel(
        panel_id=panel_id,
        board_results=board_results,
        full_panel_terminal=full_terminal,
        full_panel_terminal_supplied=full_terminal_supplied,
        leave_one_results=leave_one,
        leave_one_recomputed=leave_one_recomputed,
        admitted_fraction_range=admitted_range,
        category_heterogeneous=category_heterogeneous,
        boundary_heterogeneous=boundary_heterogeneous,
        batch_localized=batch_localized,
        deletion_unstable=deletion_unstable,
        mixed=category_heterogeneous
        or boundary_heterogeneous
        or batch_localized
        or deletion_unstable,
    )


__all__ = [
    'PhysicalScaleMorphismBoardLocalResult',
    'PhysicalScaleMorphismHeterogeneityPanel',
    'PhysicalScaleMorphismLeaveOneBoardResult',
    "evaluate_heterogeneity",
]

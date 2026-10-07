"""Finite cubical admission-boundary complexes without smoothing."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import PhysicalScaleMorphismGateSign


class PhysicalScaleMorphismBoundaryCellKind(StrEnum):
    BOUNDARY = "BOUNDARY"
    EXTERIOR = "EXTERIOR"
    INTERIOR = "INTERIOR"
    UNEVALUABLE = "UNEVALUABLE"
    UNSUPPORTED = "UNSUPPORTED"


class PhysicalScaleMorphismBoundaryDirection(StrEnum):
    DECREASING_TOWARD_PASS = "DECREASING_TOWARD_PASS"
    FLAT_OR_UNRESOLVED = "FLAT_OR_UNRESOLVED"
    INCREASING_TOWARD_PASS = "INCREASING_TOWARD_PASS"


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismBoundaryOrientation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-boundary-orientation'

    orientation_id: str
    gate_id: str
    u_direction: PhysicalScaleMorphismBoundaryDirection
    duration_direction: PhysicalScaleMorphismBoundaryDirection

    def __post_init__(self) -> None:
        validate_stable_id(self.orientation_id, field_name="orientation_id")
        validate_stable_id(self.gate_id, field_name="gate_id")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismGateAssessment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-gate-assessment'

    assessment_id: str
    gate_id: str
    sign: PhysicalScaleMorphismGateSign

    def __post_init__(self) -> None:
        validate_stable_id(self.assessment_id, field_name="assessment_id")
        validate_stable_id(self.gate_id, field_name="gate_id")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismBoundaryVertex(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-boundary-vertex'

    vertex_id: str
    u_index: int
    duration_index: int
    u_star: Decimal
    duration_star: Decimal
    gates: tuple[PhysicalScaleMorphismGateAssessment, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.vertex_id, field_name="vertex_id")
        if self.u_index < 0 or self.duration_index < 0:
            raise ValueError("boundary vertex indices must be nonnegative")
        validate_decimal(self.u_star, field_name="u_star")
        validate_decimal(self.duration_star, field_name="duration_star", minimum=Decimal(0))
        require_sorted_unique_ids(self.gates, attribute="assessment_id", field_name="gates")
        if len({value.gate_id for value in self.gates}) != len(self.gates) or not self.gates:
            raise ValueError("boundary vertex gate roster repeats or is empty")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismBoundaryCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-boundary-cell'

    cell_id: str
    u_index: int
    duration_index: int
    vertex_ids: tuple[str, ...]
    kind: PhysicalScaleMorphismBoundaryCellKind
    active_gate_ids: tuple[str, ...]
    orientations: tuple[PhysicalScaleMorphismBoundaryOrientation, ...]
    uncertainty_band: bool
    codimension: int | None

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        if self.u_index < 0 or self.duration_index < 0:
            raise ValueError("boundary cell indices must be nonnegative")
        require_sorted_unique_strings(self.vertex_ids, field_name="vertex_ids", allow_empty=False)
        if len(self.vertex_ids) != 4:
            raise ValueError("two-dimensional cubical cell requires four vertices")
        require_sorted_unique_strings(self.active_gate_ids, field_name="active_gate_ids")
        require_sorted_unique_ids(
            self.orientations, attribute="orientation_id", field_name="orientations"
        )
        if self.kind is PhysicalScaleMorphismBoundaryCellKind.BOUNDARY:
            if not self.active_gate_ids or self.codimension not in {1, 2}:
                raise ValueError("boundary cell requires active gates and codimension")
            if tuple(value.gate_id for value in self.orientations) != self.active_gate_ids:
                raise ValueError("boundary orientations do not cover every active gate")
        elif self.active_gate_ids or self.orientations or self.codimension is not None:
            raise ValueError("nonboundary cell cannot carry boundary geometry")
        if self.kind is PhysicalScaleMorphismBoundaryCellKind.UNEVALUABLE and not self.uncertainty_band:
            raise ValueError("unevaluable boundary cell must remain in the uncertainty band")
        if (
            self.kind
            not in {
                PhysicalScaleMorphismBoundaryCellKind.BOUNDARY,
                PhysicalScaleMorphismBoundaryCellKind.UNEVALUABLE,
            }
            and self.uncertainty_band
        ):
            raise ValueError("interior/exterior/unsupported cell cannot be an uncertainty band")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismBoundaryComponent(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-boundary-component'

    component_id: str
    cell_ids: tuple[str, ...]
    active_gate_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.component_id, field_name="component_id")
        require_sorted_unique_strings(self.cell_ids, field_name="cell_ids", allow_empty=False)
        require_sorted_unique_strings(
            self.active_gate_ids, field_name="active_gate_ids", allow_empty=False
        )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismBoundaryComplex(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-boundary-complex'

    complex_id: str
    chart_id: str
    cells: tuple[PhysicalScaleMorphismBoundaryCell, ...]
    components: tuple[PhysicalScaleMorphismBoundaryComponent, ...]
    interior_cell_ids: tuple[str, ...]
    exterior_cell_ids: tuple[str, ...]
    uncertainty_cell_ids: tuple[str, ...]
    unsupported_cell_ids: tuple[str, ...]
    interpolation_used: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.complex_id, field_name="complex_id")
        validate_stable_id(self.chart_id, field_name="chart_id")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_ids(
            self.components, attribute="component_id", field_name="components"
        )
        observed = {value.cell_id: value.kind for value in self.cells}
        for name, kind in (
            ("interior_cell_ids", PhysicalScaleMorphismBoundaryCellKind.INTERIOR),
            ("exterior_cell_ids", PhysicalScaleMorphismBoundaryCellKind.EXTERIOR),
            ("unsupported_cell_ids", PhysicalScaleMorphismBoundaryCellKind.UNSUPPORTED),
        ):
            values = getattr(self, name)
            require_sorted_unique_strings(values, field_name=name)
            expected = {cell_id for cell_id, cell_kind in observed.items() if cell_kind is kind}
            if set(values) != expected:
                raise ValueError(f"{name} differs from the cell classifications")
        require_sorted_unique_strings(self.uncertainty_cell_ids, field_name="uncertainty_cell_ids")
        expected_uncertainty = {value.cell_id for value in self.cells if value.uncertainty_band}
        if set(self.uncertainty_cell_ids) != expected_uncertainty:
            raise ValueError("uncertainty-cell roster differs from finite boundary bands")
        component_cells = {value for item in self.components for value in item.cell_ids}
        boundary_cells = {
            value.cell_id for value in self.cells if value.kind is PhysicalScaleMorphismBoundaryCellKind.BOUNDARY
        }
        if component_cells != boundary_cells:
            raise ValueError("boundary components do not partition boundary cells")
        if self.interpolation_used:
            raise ValueError("physical scale morphism boundary complex must retain the finite grid")


def _classify_cell(
    vertices: tuple[PhysicalScaleMorphismBoundaryVertex, ...],
) -> tuple[PhysicalScaleMorphismBoundaryCellKind, tuple[str, ...], bool]:
    signs_by_gate: dict[str, set[PhysicalScaleMorphismGateSign]] = {}
    for vertex in vertices:
        for assessment in vertex.gates:
            signs_by_gate.setdefault(assessment.gate_id, set()).add(assessment.sign)
    if any(PhysicalScaleMorphismGateSign.OUT_OF_SUPPORT in values for values in signs_by_gate.values()):
        return PhysicalScaleMorphismBoundaryCellKind.UNSUPPORTED, (), False
    if any(PhysicalScaleMorphismGateSign.UNEVALUABLE in values for values in signs_by_gate.values()):
        return PhysicalScaleMorphismBoundaryCellKind.UNEVALUABLE, (), True
    ambiguous = any(PhysicalScaleMorphismGateSign.AMBIGUOUS in values for values in signs_by_gate.values())
    active = tuple(
        sorted(
            gate_id
            for gate_id, signs in signs_by_gate.items()
            if len(signs) > 1 or PhysicalScaleMorphismGateSign.AMBIGUOUS in signs
        )
    )
    if active:
        return PhysicalScaleMorphismBoundaryCellKind.BOUNDARY, active, ambiguous
    every_pass = all(signs == {PhysicalScaleMorphismGateSign.PASS} for signs in signs_by_gate.values())
    return (
        PhysicalScaleMorphismBoundaryCellKind.INTERIOR if every_pass else PhysicalScaleMorphismBoundaryCellKind.EXTERIOR,
        (),
        False,
    )


def _direction(delta: int) -> PhysicalScaleMorphismBoundaryDirection:
    if delta > 0:
        return PhysicalScaleMorphismBoundaryDirection.INCREASING_TOWARD_PASS
    if delta < 0:
        return PhysicalScaleMorphismBoundaryDirection.DECREASING_TOWARD_PASS
    return PhysicalScaleMorphismBoundaryDirection.FLAT_OR_UNRESOLVED


def _orientations(
    *,
    cell_id: str,
    vertices: tuple[PhysicalScaleMorphismBoundaryVertex, ...],
    active_gate_ids: tuple[str, ...],
) -> tuple[PhysicalScaleMorphismBoundaryOrientation, ...]:
    score = {
        PhysicalScaleMorphismGateSign.FAIL: -1,
        PhysicalScaleMorphismGateSign.AMBIGUOUS: 0,
        PhysicalScaleMorphismGateSign.PASS: 1,
    }
    by_index = {
        (value.u_index, value.duration_index): {item.gate_id: item.sign for item in value.gates}
        for value in vertices
    }
    u_min = min(value.u_index for value in vertices)
    u_max = max(value.u_index for value in vertices)
    d_min = min(value.duration_index for value in vertices)
    d_max = max(value.duration_index for value in vertices)
    values = []
    for gate_id in active_gate_ids:
        u_delta = sum(score[by_index[(u_max, d)][gate_id]] for d in (d_min, d_max)) - sum(
            score[by_index[(u_min, d)][gate_id]] for d in (d_min, d_max)
        )
        d_delta = sum(score[by_index[(u, d_max)][gate_id]] for u in (u_min, u_max)) - sum(
            score[by_index[(u, d_min)][gate_id]] for u in (u_min, u_max)
        )
        values.append(
            PhysicalScaleMorphismBoundaryOrientation(
                orientation_id=f"{cell_id}.{gate_id}",
                gate_id=gate_id,
                u_direction=_direction(u_delta),
                duration_direction=_direction(d_delta),
            )
        )
    return tuple(sorted(values, key=lambda value: value.orientation_id))


def build_boundary_complex(
    *,
    complex_id: str,
    chart_id: str,
    vertices: tuple[PhysicalScaleMorphismBoundaryVertex, ...],
) -> PhysicalScaleMorphismBoundaryComplex:
    """Build the exact 2-D cubical complex and connected boundary components."""

    require_sorted_unique_ids(vertices, attribute="vertex_id", field_name="vertices")
    by_index = {(value.u_index, value.duration_index): value for value in vertices}
    if len(by_index) != len(vertices):
        raise ValueError("boundary grid contains duplicate vertex coordinates")
    u_values = tuple(sorted({value.u_index for value in vertices}))
    d_values = tuple(sorted({value.duration_index for value in vertices}))
    if u_values != tuple(range(len(u_values))) or d_values != tuple(range(len(d_values))):
        raise ValueError("boundary grid indices must be contiguous from zero")
    if len(u_values) < 2 or len(d_values) < 2:
        raise ValueError("boundary complex requires at least a 2x2 vertex grid")
    expected = {(u, d) for u in u_values for d in d_values}
    if set(by_index) != expected:
        raise ValueError("boundary vertex grid is incomplete")
    gate_roster = tuple(value.gate_id for value in vertices[0].gates)
    if any(tuple(item.gate_id for item in value.gates) != gate_roster for value in vertices):
        raise ValueError("boundary vertices use different gate rosters")
    cells = []
    for u_index in range(len(u_values) - 1):
        for d_index in range(len(d_values) - 1):
            cell_vertices = tuple(
                by_index[index]
                for index in (
                    (u_index, d_index),
                    (u_index, d_index + 1),
                    (u_index + 1, d_index),
                    (u_index + 1, d_index + 1),
                )
            )
            kind, active, uncertainty_band = _classify_cell(cell_vertices)
            cell_id = f"{complex_id}.cell-{u_index:02d}-{d_index:02d}"
            cells.append(
                PhysicalScaleMorphismBoundaryCell(
                    cell_id=cell_id,
                    u_index=u_index,
                    duration_index=d_index,
                    vertex_ids=tuple(sorted(value.vertex_id for value in cell_vertices)),
                    kind=kind,
                    active_gate_ids=active,
                    orientations=(
                        _orientations(
                            cell_id=cell_id,
                            vertices=cell_vertices,
                            active_gate_ids=active,
                        )
                        if kind is PhysicalScaleMorphismBoundaryCellKind.BOUNDARY
                        else ()
                    ),
                    uncertainty_band=uncertainty_band,
                    codimension=min(2, len(active)) if active else None,
                )
            )
    ordered_cells = tuple(sorted(cells, key=lambda value: value.cell_id))
    boundary_by_index = {
        (value.u_index, value.duration_index): value
        for value in ordered_cells
        if value.kind is PhysicalScaleMorphismBoundaryCellKind.BOUNDARY
    }
    unvisited = set(boundary_by_index)
    components: list[PhysicalScaleMorphismBoundaryComponent] = []
    while unvisited:
        seed = min(unvisited)
        stack = [seed]
        component_indices = set()
        while stack:
            current = stack.pop()
            if current not in unvisited:
                continue
            unvisited.remove(current)
            component_indices.add(current)
            u_index, d_index = current
            stack.extend(
                neighbor
                for neighbor in (
                    (u_index - 1, d_index),
                    (u_index + 1, d_index),
                    (u_index, d_index - 1),
                    (u_index, d_index + 1),
                )
                if neighbor in unvisited
            )
        component_cells = tuple(
            sorted(boundary_by_index[value].cell_id for value in component_indices)
        )
        active_gates = tuple(
            sorted(
                {
                    gate
                    for value in component_indices
                    for gate in boundary_by_index[value].active_gate_ids
                }
            )
        )
        components.append(
            PhysicalScaleMorphismBoundaryComponent(
                component_id=f"{complex_id}.component-{len(components):02d}",
                cell_ids=component_cells,
                active_gate_ids=active_gates,
            )
        )
    ordered_components = tuple(sorted(components, key=lambda value: value.component_id))
    return PhysicalScaleMorphismBoundaryComplex(
        complex_id=complex_id,
        chart_id=chart_id,
        cells=ordered_cells,
        components=ordered_components,
        interior_cell_ids=tuple(
            value.cell_id for value in ordered_cells if value.kind is PhysicalScaleMorphismBoundaryCellKind.INTERIOR
        ),
        exterior_cell_ids=tuple(
            value.cell_id for value in ordered_cells if value.kind is PhysicalScaleMorphismBoundaryCellKind.EXTERIOR
        ),
        uncertainty_cell_ids=tuple(
            value.cell_id for value in ordered_cells if value.uncertainty_band
        ),
        unsupported_cell_ids=tuple(
            value.cell_id
            for value in ordered_cells
            if value.kind is PhysicalScaleMorphismBoundaryCellKind.UNSUPPORTED
        ),
        interpolation_used=False,
    )


__all__ = [
    'PhysicalScaleMorphismBoundaryCell',
    'PhysicalScaleMorphismBoundaryCellKind',
    'PhysicalScaleMorphismBoundaryComplex',
    'PhysicalScaleMorphismBoundaryComponent',
    'PhysicalScaleMorphismBoundaryDirection',
    'PhysicalScaleMorphismBoundaryOrientation',
    'PhysicalScaleMorphismBoundaryVertex',
    'PhysicalScaleMorphismGateAssessment',
    "build_boundary_complex",
]

"""Finite quantum response control response-law/atlas records with accepted-schema mappings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Mapping, Sequence

from .contracts import Action, CHILD_PLAN_ID


@dataclass(frozen=True, slots=True)
class ReceiverChartCell:
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/quantum-response-control/receiver-chart-cell'

    cell_id: str
    chart_id: str
    scaler_identity: str
    k_left: int
    coordinate_values: tuple[tuple[str, float | str], ...]
    support_radius: float
    kth_distance: float
    inside_support: bool
    gap_reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalResponseLaw:
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/quantum-response-control/local-response-law'
    ACCEPTED_SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/response-law'

    law_id: str
    cell_id: str
    chart_id: str
    action: Action
    denominator_id: str
    source_identity: str
    preparation_identity: str
    horizon: float
    response_values: Mapping[str, float]
    uncertainty_halfwidths: Mapping[str, float]
    support_radius: float
    chart_law_result_identity: str
    fiber_result_identity: str
    validity_passed: bool
    evidence_ceiling: str = "LOCAL_LAW"

    def compatibility_map(self) -> dict[str, object]:
        """Additive field map; it does not redefine the accepted ontology."""

        return {
            "accepted_schema": self.ACCEPTED_SCHEMA,
            "law_id": self.law_id,
            "system_id": "system.quantum-trajectory-chain.l12-n6",
            "world_id": "world.quantum-response-control.exact-diagonalization",
            "relation": {
                "denominator": self.denominator_id,
                "history_chart": self.chart_id,
                "action": self.action.value,
                "receiver_vector": sorted(self.response_values),
                "horizon": self.horizon,
            },
            "chart_id": self.chart_id,
            "representation_kind": "FINITE_ACTION_OPERATOR",
            "causal_strength": "SIMULATOR_INTERVENTION",
            "evidence_ceiling": self.evidence_ceiling,
            "extensions": {
                "response_control_cell_id": self.cell_id,
                "support_radius": self.support_radius,
                "chart_law_result_identity": self.chart_law_result_identity,
                "fiber_result_identity": self.fiber_result_identity,
            },
        }


@dataclass(frozen=True, slots=True)
class AtlasGapRecord:
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/quantum-response-control/atlas-gap-record'
    ACCEPTED_SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/atlas-gap'

    gap_id: str
    chart_id: str
    cell_id: str
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AtlasOverlapRecord:
    overlap_id: str
    cell_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    conflict: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResponseAtlasRecord:
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/quantum-response-control/response-atlas-record'
    ACCEPTED_SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/response-atlas'

    atlas_id: str
    laws: tuple[LocalResponseLaw, ...]
    gaps: tuple[AtlasGapRecord, ...]
    overlaps: tuple[AtlasOverlapRecord, ...]
    evidence_identities: tuple[str, ...]
    global_smoothness_assumed: bool = False

    def __post_init__(self) -> None:
        if self.global_smoothness_assumed:
            raise ValueError("quantum response control atlas cannot assume global smoothness")
        law_ids = [law.law_id for law in self.laws]
        if len(law_ids) != len(set(law_ids)):
            raise ValueError("response atlas repeats a law")
        if not self.evidence_identities:
            raise ValueError("response atlas requires evidence lineage")

    def compatibility_map(self) -> dict[str, object]:
        return {
            "accepted_schema": self.ACCEPTED_SCHEMA,
            "atlas_id": self.atlas_id,
            "system_id": "system.quantum-trajectory-chain.l12-n6",
            "world_id": "world.quantum-response-control.exact-diagonalization",
            "laws": [law.compatibility_map() for law in self.laws],
            "gaps": [
                {
                    "accepted_schema": gap.ACCEPTED_SCHEMA,
                    "gap_id": gap.gap_id,
                    "chart_ids": [gap.chart_id],
                    "denominator_cell_ids": [gap.cell_id],
                    "reason_codes": list(gap.reason_codes),
                }
                for gap in self.gaps
            ],
            "transitions": [],
            "global_smoothness_assumed": False,
            "evidence_identities": list(self.evidence_identities),
        }


def assemble_atlas(
    *,
    cells: Sequence[ReceiverChartCell],
    laws: Sequence[LocalResponseLaw],
    evidence_identities: Sequence[str],
) -> ResponseAtlasRecord:
    cells_by_id = {cell.cell_id: cell for cell in cells}
    if len(cells_by_id) != len(cells):
        raise ValueError("atlas cell IDs collide")
    for law in laws:
        if law.cell_id not in cells_by_id:
            raise ValueError("response law references an unknown atlas cell")
    gaps = tuple(
        AtlasGapRecord(
            gap_id=f"gap.{cell.cell_id}",
            chart_id=cell.chart_id,
            cell_id=cell.cell_id,
            reason_codes=cell.gap_reason_codes or ("outside-supported-radius",),
        )
        for cell in cells
        if not cell.inside_support or cell.gap_reason_codes
    )
    overlaps: list[AtlasOverlapRecord] = []
    grouped: dict[tuple[str, int, tuple[tuple[str, float | str], ...]], list[LocalResponseLaw]] = {}
    for law in laws:
        cell = cells_by_id[law.cell_id]
        key = (cell.chart_id, cell.k_left, cell.coordinate_values)
        grouped.setdefault(key, []).append(law)
    for index, local in enumerate(grouped.values()):
        cell_ids = tuple(sorted({law.cell_id for law in local}))
        if len(cell_ids) < 2:
            continue
        action_ids = tuple(sorted({law.action.value for law in local}))
        conflict = len(action_ids) > 1
        overlaps.append(
            AtlasOverlapRecord(
                overlap_id=f"overlap.{index:05d}",
                cell_ids=cell_ids,
                action_ids=action_ids,
                conflict=conflict,
                reason_codes=("conflicting-action-response",) if conflict else (),
            )
        )
    return ResponseAtlasRecord(
        atlas_id=f"atlas.{CHILD_PLAN_ID}",
        laws=tuple(sorted(laws, key=lambda law: law.law_id)),
        gaps=tuple(sorted(gaps, key=lambda gap: gap.gap_id)),
        overlaps=tuple(sorted(overlaps, key=lambda overlap: overlap.overlap_id)),
        evidence_identities=tuple(sorted(set(evidence_identities))),
    )


def assert_current_path_compatibility(atlas: ResponseAtlasRecord) -> None:
    document = atlas.compatibility_map()
    if document["accepted_schema"] != 'empirical-lawhood/kernel/response-atlas':
        raise AssertionError("atlas compatibility targets another accepted schema")
    if document["global_smoothness_assumed"] is not False:
        raise AssertionError("atlas compatibility introduced global smoothing")
    for law in atlas.laws:
        mapping = law.compatibility_map()
        if mapping["accepted_schema"] != 'empirical-lawhood/kernel/response-law':
            raise AssertionError("law compatibility targets another accepted schema")
        relation = mapping["relation"]
        if not isinstance(relation, Mapping):
            raise AssertionError("law compatibility relation is not typed")
        if relation["horizon"] != 4.0:
            raise AssertionError("law compatibility changed the action horizon")


__all__ = [
    "AtlasGapRecord",
    "AtlasOverlapRecord",
    "LocalResponseLaw",
    "ReceiverChartCell",
    "ResponseAtlasRecord",
    "assemble_atlas",
    "assert_current_path_compatibility",
]

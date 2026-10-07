"""Response atlases with explicit gaps and proof-gated composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .evidence import EvidenceCeiling, EvidenceRung, VisibilityCeiling
from .laws import ResponseLaw
from .provenance import EvidenceLink
from .references import ExecutableReference
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from .systems import InterfaceSpec, PortSpec, SystemSpec


class AtlasGapKind(StrEnum):
    UNSUPPORTED = "UNSUPPORTED"
    REJECTED_TRANSITION = "REJECTED_TRANSITION"
    INVALID_COORDINATE = "INVALID_COORDINATE"
    COMPUTABILITY_BOUNDARY = "COMPUTABILITY_BOUNDARY"
    NO_DATA = "NO_DATA"


@dataclass(frozen=True, slots=True)
class AtlasGap(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/atlas-gap'

    gap_id: str
    kind: AtlasGapKind
    chart_ids: tuple[str, ...]
    denominator_cell_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.gap_id, field_name="gap_id")
        require_sorted_unique_strings(self.chart_ids, field_name="chart_ids", allow_empty=False)
        require_sorted_unique_strings(
            self.denominator_cell_ids,
            field_name="denominator_cell_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.reason_codes, field_name="reason_codes", allow_empty=False
        )
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")


class ChartTransitionStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    REJECTED = "REJECTED"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class ChartTransition(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/chart-transition'

    transition_id: str
    source_chart_id: str
    target_chart_id: str
    status: ChartTransitionStatus
    transport_evaluator: ExecutableReference | None
    reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("transition_id", self.transition_id),
            ("source_chart_id", self.source_chart_id),
            ("target_chart_id", self.target_chart_id),
        ):
            validate_stable_id(value, field_name=name)
        if self.source_chart_id == self.target_chart_id:
            raise ValueError("a chart transition requires distinct charts")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        if self.status is ChartTransitionStatus.SUPPORTED:
            if self.transport_evaluator is None or not self.evidence_link_ids:
                raise ValueError("a supported chart transition requires evaluator and evidence")
            if self.reason_codes:
                raise ValueError("a supported chart transition cannot have failures")
        else:
            if self.transport_evaluator is not None:
                raise ValueError("a rejected/unevaluable transition has no evaluator")
            if not self.reason_codes:
                raise ValueError("a rejected/unevaluable transition requires reasons")


@dataclass(frozen=True, slots=True)
class ResponseAtlas(CanonicalRecord):
    """Local law collection that preserves holes and rejected transports."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/response-atlas'

    atlas_id: str
    system_id: str
    world_id: str
    laws: tuple[ResponseLaw, ...]
    transitions: tuple[ChartTransition, ...]
    gaps: tuple[AtlasGap, ...]
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling
    global_smoothness_assumed: bool = False
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("atlas_id", self.atlas_id),
            ("system_id", self.system_id),
            ("world_id", self.world_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(self.laws, attribute="law_id", field_name="laws")
        if not self.laws:
            raise ValueError("a ResponseAtlas requires at least one supported law")
        require_sorted_unique_ids(
            self.transitions,
            attribute="transition_id",
            field_name="transitions",
        )
        require_sorted_unique_ids(self.gaps, attribute="gap_id", field_name="gaps")
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        if not self.evidence_links:
            raise ValueError("response atlas requires evidence links")
        chart_ids = {law.chart_id for law in self.laws}
        for law in self.laws:
            if law.system_id != self.system_id or law.world_id != self.world_id:
                raise ValueError("atlas laws differ in system or world identity")
        for transition in self.transitions:
            if not {
                transition.source_chart_id,
                transition.target_chart_id,
            }.issubset(chart_ids):
                raise ValueError("chart transition references a chart outside the atlas")
        if not self.evidence_ceiling.allows(EvidenceRung.LOCAL_LAW):
            raise ValueError("response-atlas evidence ceiling is below local law")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("an outcome-visible collection cannot become an atlas")
        if self.global_smoothness_assumed:
            raise ValueError("response atlases cannot assume untested global smoothness")
        require_extensions(self.extensions)

    def laws_for_chart(self, chart_id: str) -> tuple[ResponseLaw, ...]:
        validate_stable_id(chart_id, field_name="chart_id")
        return tuple(law for law in self.laws if law.chart_id == chart_id)

    def laws_for_coordinate(
        self, chart_id: str, denominator_cell_id: str
    ) -> tuple[ResponseLaw, ...]:
        """Return laws with explicit support at one chart/denominator coordinate."""

        validate_stable_id(chart_id, field_name="chart_id")
        validate_stable_id(denominator_cell_id, field_name="denominator_cell_id")
        return tuple(
            law
            for law in self.laws
            if law.chart_id == chart_id
            and denominator_cell_id in law.obligations.support.denominator_cell_ids
        )

    def holes_for_chart(self, chart_id: str) -> tuple[AtlasGap, ...]:
        validate_stable_id(chart_id, field_name="chart_id")
        return tuple(gap for gap in self.gaps if chart_id in gap.chart_ids)

    def gaps_for_coordinate(self, chart_id: str, denominator_cell_id: str) -> tuple[AtlasGap, ...]:
        """Return only gaps that cover an exact chart/denominator coordinate."""

        validate_stable_id(chart_id, field_name="chart_id")
        validate_stable_id(denominator_cell_id, field_name="denominator_cell_id")
        return tuple(
            gap
            for gap in self.gaps
            if chart_id in gap.chart_ids and denominator_cell_id in gap.denominator_cell_ids
        )


class CompositionRejected(ValueError):
    """Raised before any law/atlas composition algorithm can execute."""

    def __init__(self, reason_codes: tuple[str, ...]) -> None:
        self.reason_codes = reason_codes
        super().__init__(f"composition rejected: {', '.join(reason_codes)}")


def _owner_ports(system: SystemSpec, owner_id: str) -> tuple[PortSpec, ...] | None:
    if owner_id == system.system_id:
        return system.ports
    for component in system.components:
        if component.component_id == owner_id:
            return component.ports
    return None


def _port(system: SystemSpec, owner_id: str, port_id: str) -> PortSpec | None:
    ports = _owner_ports(system, owner_id)
    if ports is None:
        return None
    return next((port for port in ports if port.port_id == port_id), None)


def law_composition_reason_codes(
    source_law: ResponseLaw,
    target_law: ResponseLaw,
    system: SystemSpec,
    interface: InterfaceSpec,
) -> tuple[str, ...]:
    """Return all static reasons that make two local laws non-composable."""

    reasons: list[str] = []
    if interface not in system.interfaces:
        reasons.append("interface-not-in-system")
    if source_law.system_id != interface.source.owner_id:
        reasons.append("source-law-owner-mismatch")
    if target_law.system_id != interface.target.owner_id:
        reasons.append("target-law-owner-mismatch")
    if source_law.world_id != system.world.world_id:
        reasons.append("source-world-mismatch")
    if target_law.world_id != system.world.world_id:
        reasons.append("target-world-mismatch")
    source_port = _port(system, interface.source.owner_id, interface.source.port_id)
    target_port = _port(system, interface.target.owner_id, interface.target.port_id)
    if source_port is None:
        reasons.append("source-port-missing")
    elif source_port.quantity_id not in source_law.interface_output_quantity_ids:
        reasons.append("source-law-does-not-expose-interface-quantity")
    if target_port is None:
        reasons.append("target-port-missing")
    elif target_port.quantity_id not in target_law.interface_input_quantity_ids:
        reasons.append("target-law-does-not-consume-interface-quantity")
    if source_law.relation.horizon.clock_id != target_law.relation.horizon.clock_id:
        relation = interface.clock_relation
        if relation is None or (
            relation.source_clock_id != source_law.relation.horizon.clock_id
            or relation.target_clock_id != target_law.relation.horizon.clock_id
        ):
            reasons.append("law-horizon-clock-relation-missing")
    if not source_law.obligations.claim_ready:
        reasons.append("source-law-obligations-blocked")
    if not target_law.obligations.claim_ready:
        reasons.append("target-law-obligations-blocked")
    return tuple(sorted(set(reasons)))


def require_law_composable(
    source_law: ResponseLaw,
    target_law: ResponseLaw,
    system: SystemSpec,
    interface: InterfaceSpec,
) -> None:
    reasons = law_composition_reason_codes(source_law, target_law, system, interface)
    if reasons:
        raise CompositionRejected(reasons)


def require_atlases_composable(
    source_atlas: ResponseAtlas,
    source_chart_id: str,
    target_atlas: ResponseAtlas,
    target_chart_id: str,
    system: SystemSpec,
    interface: InterfaceSpec,
) -> None:
    reasons: list[str] = []
    source_laws = source_atlas.laws_for_chart(source_chart_id)
    target_laws = target_atlas.laws_for_chart(target_chart_id)
    if len(source_laws) != 1:
        reasons.append("source-chart-law-not-unique")
    if len(target_laws) != 1:
        reasons.append("target-chart-law-not-unique")
    if source_atlas.holes_for_chart(source_chart_id):
        reasons.append("source-chart-has-explicit-gap")
    if target_atlas.holes_for_chart(target_chart_id):
        reasons.append("target-chart-has-explicit-gap")
    if not reasons:
        reasons.extend(
            law_composition_reason_codes(source_laws[0], target_laws[0], system, interface)
        )
    if reasons:
        raise CompositionRejected(tuple(sorted(set(reasons))))

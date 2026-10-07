"""Receiver-admission intersections and finite-horizon reachability results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .atlases import ResponseAtlas
from .evidence import EvidenceCeiling, EvidenceRung, VisibilityCeiling
from .obligations import (
    ComputabilityEvidence,
    ObligationStatus,
    StructuralConvergenceSpec,
)
from .provenance import EvidenceLink, ObjectIdentity
from .references import NamedDecimal
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_stable_id,
)
from .status import AdmissionStatus, ReadinessStatus
from .time import HorizonSpec


class AdmissionGateKind(StrEnum):
    TARGET = "TARGET"
    PHYSICAL_SINK = "PHYSICAL_SINK"
    EFFORT = "EFFORT"
    OBSERVATION_VALIDITY = "OBSERVATION_VALIDITY"
    UNCERTAINTY = "UNCERTAINTY"
    BASELINE_PRESERVATION = "BASELINE_PRESERVATION"
    DYNAMICS = "DYNAMICS"
    REACHABILITY = "REACHABILITY"
    AUTHORITY = "AUTHORITY"


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNEVALUABLE = "UNEVALUABLE"


@dataclass(frozen=True, slots=True)
class AdmissionGateResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/admission-gate-result'

    gate_id: str
    kind: AdmissionGateKind
    status: GateStatus
    constraint_ids: tuple[str, ...]
    margin: NamedDecimal | None
    reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.gate_id, field_name="gate_id")
        require_sorted_unique_strings(
            self.constraint_ids,
            field_name="constraint_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(self.evidence_link_ids, field_name="evidence_link_ids")
        if self.status is GateStatus.PASS:
            if self.reason_codes or not self.evidence_link_ids:
                raise ValueError("a passing admission gate needs evidence and no failures")
        elif not self.reason_codes:
            raise ValueError("a failed/unevaluable admission gate requires reasons")


@dataclass(frozen=True, slots=True)
class AdmissionCellResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/admission-cell-result'

    cell_id: str
    denominator_cell_id: str
    chart_id: str
    action_bound_ids: tuple[str, ...]
    gates: tuple[AdmissionGateResult, ...]
    admitted: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("denominator_cell_id", self.denominator_cell_id),
            ("chart_id", self.chart_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(
            self.action_bound_ids,
            field_name="action_bound_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.gates, attribute="gate_id", field_name="gates")
        observed_kinds = {gate.kind for gate in self.gates}
        if len(self.gates) != len(AdmissionGateKind) or observed_kinds != set(AdmissionGateKind):
            missing = sorted(kind.value for kind in set(AdmissionGateKind) - observed_kinds)
            raise ValueError(f"admission cell is missing required gates: {missing}")
        should_admit = all(gate.status is GateStatus.PASS for gate in self.gates)
        if self.admitted != should_admit:
            raise ValueError("admission is the intersection of every required gate")


def _derived_admission_status(
    cells: tuple[AdmissionCellResult, ...],
) -> AdmissionStatus:
    admitted = [cell for cell in cells if cell.admitted]
    if admitted and len(admitted) == len(cells):
        return AdmissionStatus.ADMITTED
    if admitted:
        return AdmissionStatus.PARTIAL
    if any(gate.status is GateStatus.UNEVALUABLE for cell in cells for gate in cell.gates):
        return AdmissionStatus.UNEVALUABLE
    return AdmissionStatus.EMPTY


@dataclass(frozen=True, slots=True)
class AdmissionSet(CanonicalRecord):
    """Exact target/sink/effort/validity/uncertainty/dynamics intersection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/admission-set'

    admission_id: str
    atlas: ObjectIdentity
    model_set_id: str
    receiver_quantity_ids: tuple[str, ...]
    cells: tuple[AdmissionCellResult, ...]
    status: AdmissionStatus
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling
    safe_abstention_required_outside: bool = True
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.admission_id, field_name="admission_id")
        validate_stable_id(self.model_set_id, field_name="model_set_id")
        require_sorted_unique_strings(
            self.receiver_quantity_ids,
            field_name="receiver_quantity_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if not self.cells:
            raise ValueError("admission must retain evaluated cells, including failures")
        if self.status is not _derived_admission_status(self.cells):
            raise ValueError("admission status differs from the gate intersection")
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        if not self.evidence_links:
            raise ValueError("admission evaluation requires evidence links")
        if not self.evidence_ceiling.allows(EvidenceRung.ADMISSION):
            raise ValueError("admission evidence ceiling is below admission")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible analysis cannot become admission evidence")
        if not self.safe_abstention_required_outside:
            raise ValueError("measured HOLD or NONATTEMPT is mandatory outside admission")
        require_extensions(self.extensions)

    @property
    def admitted_cell_ids(self) -> tuple[str, ...]:
        return tuple(cell.cell_id for cell in self.cells if cell.admitted)

    def exclusion_reason_codes(self, cell_id: str) -> tuple[str, ...]:
        validate_stable_id(cell_id, field_name="cell_id")
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return tuple(
                    sorted(
                        {
                            reason
                            for gate in cell.gates
                            if gate.status is not GateStatus.PASS
                            for reason in gate.reason_codes
                        }
                    )
                )
        raise KeyError(cell_id)


def validate_admission_against_atlas(atlas: ResponseAtlas, admission: AdmissionSet) -> None:
    if admission.atlas != ObjectIdentity.from_record(atlas.atlas_id, atlas):
        raise ValueError("admission binds the wrong response atlas")
    chart_ids = {law.chart_id for law in atlas.laws}
    output_quantity_ids = {
        quantity_id for law in atlas.laws for quantity_id in law.interface_output_quantity_ids
    }
    if not set(admission.receiver_quantity_ids).issubset(output_quantity_ids):
        raise ValueError("admission receiver is not produced by its atlas")
    for cell in admission.cells:
        if cell.chart_id not in chart_ids:
            raise ValueError("admission cell uses a chart outside its atlas")
        if not atlas.laws_for_coordinate(cell.chart_id, cell.denominator_cell_id):
            raise ValueError("admission cell lies outside explicit atlas law support")
        if cell.admitted and atlas.gaps_for_coordinate(cell.chart_id, cell.denominator_cell_id):
            raise ValueError("an admitted cell cannot occupy an explicit atlas gap")


class ReachabilityStatus(StrEnum):
    REACHABLE = "REACHABLE"
    PARTIAL = "PARTIAL"
    EMPTY = "EMPTY"
    UNEVALUABLE = "UNEVALUABLE"
    COMPUTABILITY_BOUNDARY = "COMPUTABILITY_BOUNDARY"


@dataclass(frozen=True, slots=True)
class ReachabilityResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/reachability-result'

    reachability_id: str
    admission: ObjectIdentity
    initial_set_id: str
    action_chart_ids: tuple[str, ...]
    dynamics_law_ids: tuple[str, ...]
    horizon: HorizonSpec
    constraint_ids: tuple[str, ...]
    numerical_view_ids: tuple[str, ...]
    structural_convergence: StructuralConvergenceSpec
    computability: ComputabilityEvidence
    reachable_cell_ids: tuple[str, ...]
    viable_direction_rank: int
    status: ReachabilityStatus
    evidence_links: tuple[EvidenceLink, ...]
    evidence_ceiling: EvidenceCeiling
    visibility_ceiling: VisibilityCeiling
    safe_abstention_required_outside: bool = True
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.reachability_id, field_name="reachability_id")
        validate_stable_id(self.initial_set_id, field_name="initial_set_id")
        for field_name, values in (
            ("action_chart_ids", self.action_chart_ids),
            ("dynamics_law_ids", self.dynamics_law_ids),
            ("constraint_ids", self.constraint_ids),
            ("numerical_view_ids", self.numerical_view_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        require_sorted_unique_strings(self.reachable_cell_ids, field_name="reachable_cell_ids")
        if self.viable_direction_rank < 0:
            raise ValueError("viable_direction_rank must be nonnegative")
        require_sorted_unique_ids(
            self.evidence_links, attribute="link_id", field_name="evidence_links"
        )
        if self.status in {ReachabilityStatus.REACHABLE, ReachabilityStatus.PARTIAL}:
            if not self.reachable_cell_ids or self.viable_direction_rank == 0:
                raise ValueError("reachable result needs cells and viable directions")
        if self.status is ReachabilityStatus.EMPTY:
            if self.reachable_cell_ids or self.viable_direction_rank != 0:
                raise ValueError("empty reachability cannot retain reachable geometry")
        if self.status in {
            ReachabilityStatus.UNEVALUABLE,
            ReachabilityStatus.COMPUTABILITY_BOUNDARY,
        } and (self.reachable_cell_ids or self.viable_direction_rank != 0):
            raise ValueError("unevaluable/boundary reachability has no viable geometry")
        if self.structural_convergence.status is not ObligationStatus.SATISFIED:
            if self.status in {
                ReachabilityStatus.REACHABLE,
                ReachabilityStatus.PARTIAL,
            }:
                raise ValueError("reachable geometry requires structural convergence")
        if self.computability.readiness is not ReadinessStatus.READY:
            if self.status is not ReachabilityStatus.COMPUTABILITY_BOUNDARY:
                raise ValueError("unready computability requires boundary status")
        if not self.evidence_ceiling.allows(EvidenceRung.ADMISSION):
            raise ValueError("reachability evidence ceiling is below admission")
        if not self.visibility_ceiling.is_promotable:
            raise ValueError("outcome-visible analysis cannot establish reachability")
        if not self.safe_abstention_required_outside:
            raise ValueError("measured HOLD or NONATTEMPT is mandatory outside reachability")
        require_extensions(self.extensions)


def validate_reachability_against_admission(
    admission: AdmissionSet, reachability: ReachabilityResult
) -> None:
    if reachability.admission != ObjectIdentity.from_record(admission.admission_id, admission):
        raise ValueError("reachability binds the wrong admission set")
    if not set(reachability.reachable_cell_ids).issubset(admission.admitted_cell_ids):
        raise ValueError("reachability includes cells outside receiver admission")
    if admission.status in {AdmissionStatus.EMPTY, AdmissionStatus.UNEVALUABLE}:
        if reachability.status not in {
            ReachabilityStatus.EMPTY,
            ReachabilityStatus.UNEVALUABLE,
            ReachabilityStatus.COMPUTABILITY_BOUNDARY,
        }:
            raise ValueError("empty/unevaluable admission cannot be reachable")
        if reachability.viable_direction_rank != 0:
            raise ValueError("empty/unevaluable admission has zero viable rank")

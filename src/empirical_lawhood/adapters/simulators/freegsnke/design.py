"""Outcome-blind action and preparation roster grammar for FreeGSNKE independent substrate grounding."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_decimal,
    validate_stable_id,
)

from .contracts import (
    FREEGSNKE_PORT_IDS,
    FreeGsnkeBranchKind,
    FreeGsnkeNumericalView,
    FreeGsnkePhase,
    FreeGsnkePreparation,
    FreeGsnkeProcessRequest,
    FreeGsnkeSourceBinding,
)
from .preparation import FreeGsnkePreparedUnit


FREEGSNKE_DEVELOPMENT_EVALUATION_PHASES = (
    FreeGsnkePhase.DEVELOPMENT,
    FreeGsnkePhase.EVALUATION,
)
FREEGSNKE_INITIAL_PANEL_MINIMUMS = {
    FreeGsnkePhase.DEVELOPMENT: 48,
    FreeGsnkePhase.EVALUATION: 48,
}


@dataclass(frozen=True, slots=True)
class FreeGsnkeBranchSpec(CanonicalRecord):
    """One predevelopment action branch in chart-relative dose coordinates."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-branch-spec'

    branch_id: str
    branch_kind: FreeGsnkeBranchKind
    p4_dose_fraction: Decimal
    p5_dose_fraction: Decimal
    repeat_of_branch_id: str | None

    def __post_init__(self) -> None:
        validate_stable_id(self.branch_id, field_name="branch_id")
        for name in ("p4_dose_fraction", "p5_dose_fraction"):
            value = getattr(self, name)
            validate_decimal(value, field_name=name)
            if abs(value) > 1:
                raise ValueError("FreeGSNKE branch fraction exceeds the frozen full dose")
        is_zero = self.p4_dose_fraction == 0 and self.p5_dose_fraction == 0
        nonzero_ports = sum(value != 0 for value in (self.p4_dose_fraction, self.p5_dose_fraction))
        if self.branch_kind is FreeGsnkeBranchKind.COMPARATOR:
            if not is_zero:
                raise ValueError("FreeGSNKE comparator branch must be exact hold")
        elif is_zero:
            raise ValueError("FreeGSNKE non-comparator branch must be nonzero")
        if (
            self.branch_kind
            in {
                FreeGsnkeBranchKind.SINGLE_PORT,
                FreeGsnkeBranchKind.HALF_DOSE,
            }
            and nonzero_ports != 1
        ):
            raise ValueError("FreeGSNKE single/half-dose branch must select one port")
        if self.branch_kind is FreeGsnkeBranchKind.HALF_DOSE and not all(
            abs(value) in {Decimal(0), Decimal("0.5")}
            for value in (self.p4_dose_fraction, self.p5_dose_fraction)
        ):
            raise ValueError("FreeGSNKE half-dose branch must use an exact 0.5 fraction")
        if self.branch_kind is FreeGsnkeBranchKind.JOINT and nonzero_ports != 2:
            raise ValueError("FreeGSNKE joint branch must actuate both ports")
        if self.repeat_of_branch_id is not None:
            validate_stable_id(
                self.repeat_of_branch_id,
                field_name="repeat_of_branch_id",
            )
            if self.repeat_of_branch_id == self.branch_id:
                raise ValueError("FreeGSNKE repeat branch cannot name itself")


@dataclass(frozen=True, slots=True)
class FreeGsnkeActionDesign(CanonicalRecord):
    """Finite target action roster; it is not inherited from the microfixture."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-action-design'

    design_id: str
    branches: tuple[FreeGsnkeBranchSpec, ...]
    dose_contrast_claimed: bool
    simultaneous_composition_claimed: bool
    deterministic_repeat_claimed: bool
    inherited_from_microfixture: bool
    frozen_before_development: bool
    protected_outcome_access_count: int

    def __post_init__(self) -> None:
        validate_stable_id(self.design_id, field_name="design_id")
        require_sorted_unique_ids(
            self.branches,
            attribute="branch_id",
            field_name="branches",
        )
        if self.inherited_from_microfixture:
            raise ValueError("FreeGSNKE target action roster cannot inherit the microfixture")
        if not self.frozen_before_development or self.protected_outcome_access_count:
            raise ValueError("FreeGSNKE target action roster crossed the freeze boundary")
        if not any(value.branch_kind is FreeGsnkeBranchKind.COMPARATOR for value in self.branches):
            raise ValueError("FreeGSNKE target action roster requires hold")
        for port in FREEGSNKE_PORT_IDS:
            values = tuple(getattr(value, f"{port}_dose_fraction") for value in self.branches)
            if not any(value > 0 for value in values) or not any(value < 0 for value in values):
                raise ValueError("FreeGSNKE target action roster requires both signs per port")
            if self.dose_contrast_claimed and len({abs(value) for value in values if value}) < 2:
                raise ValueError("FreeGSNKE dose claim requires two nonzero magnitudes per port")
        if self.simultaneous_composition_claimed and not any(
            value.branch_kind is FreeGsnkeBranchKind.JOINT for value in self.branches
        ):
            raise ValueError("FreeGSNKE composition claim/branch roster differs")
        repeats = tuple(value for value in self.branches if value.repeat_of_branch_id is not None)
        if self.deterministic_repeat_claimed and not repeats:
            raise ValueError("FreeGSNKE repeat claim/branch roster differs")
        by_id = {value.branch_id: value for value in self.branches}
        for repeat in repeats:
            parent_id = repeat.repeat_of_branch_id
            assert parent_id is not None
            try:
                parent = by_id[parent_id]
            except KeyError as error:
                raise ValueError("FreeGSNKE repeat parent is outside the roster") from error
            if (
                repeat.p4_dose_fraction != parent.p4_dose_fraction
                or repeat.p5_dose_fraction != parent.p5_dose_fraction
                or repeat.branch_kind is not parent.branch_kind
            ):
                raise ValueError("FreeGSNKE repeat branch changes the parent action")


@dataclass(frozen=True, slots=True)
class FreeGsnkePhaseRequestRoster(CanonicalRecord):
    """Exact issued phase roster with preparations as the only replicate unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-phase-request-roster'

    roster_id: str
    phase: FreeGsnkePhase
    action_design: ObjectIdentity
    preparations: tuple[ObjectIdentity, ...]
    requests: tuple[ObjectIdentity, ...]
    independent_preparation_count: int
    nested_branch_count: int
    initial_panel_minimum: int
    panel_envelope_limited: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        if self.phase not in FREEGSNKE_DEVELOPMENT_EVALUATION_PHASES:
            raise ValueError("FreeGSNKE development-through-admission evaluation roster phase differs")
        if self.action_design.object_schema != FreeGsnkeActionDesign.SCHEMA:
            raise ValueError("FreeGSNKE action-design identity differs")
        require_sorted_unique_ids(
            self.preparations,
            attribute="object_id",
            field_name="preparations",
        )
        require_sorted_unique_ids(
            self.requests,
            attribute="object_id",
            field_name="requests",
        )
        if not self.preparations:
            raise ValueError("FreeGSNKE phase roster requires preparations")
        if any(
            value.object_schema != FreeGsnkePreparation.SCHEMA for value in self.preparations
        ) or any(value.object_schema != FreeGsnkeProcessRequest.SCHEMA for value in self.requests):
            raise ValueError("FreeGSNKE phase roster child schema differs")
        if self.independent_preparation_count != len(self.preparations):
            raise ValueError("FreeGSNKE independent count is not preparation-derived")
        if self.nested_branch_count != len(self.requests):
            raise ValueError("FreeGSNKE nested branch count is not request-derived")
        expected_minimum = FREEGSNKE_INITIAL_PANEL_MINIMUMS[self.phase]
        if self.initial_panel_minimum != expected_minimum:
            raise ValueError("FreeGSNKE initial panel minimum differs")
        if self.panel_envelope_limited != (self.independent_preparation_count < expected_minimum):
            raise ValueError("FreeGSNKE panel-envelope label is not count-derived")


def _expected_increments(
    request: FreeGsnkeProcessRequest,
    branch: FreeGsnkeBranchSpec,
) -> dict[str, Decimal]:
    doses = {value.port_id: value.full_increment_v for value in request.action_chart.ports}
    return {
        "p4": branch.p4_dose_fraction * doses["p4"],
        "p5": branch.p5_dose_fraction * doses["p5"],
    }


def freeze_freegsnke_phase_request_roster(
    *,
    roster_id: str,
    phase: FreeGsnkePhase,
    action_design: FreeGsnkeActionDesign,
    preparations: tuple[FreeGsnkePreparation, ...],
    requests: tuple[FreeGsnkeProcessRequest, ...],
) -> FreeGsnkePhaseRequestRoster:
    """Bind an arbitrary predeclared branch design to every exact preparation."""

    if phase not in FREEGSNKE_DEVELOPMENT_EVALUATION_PHASES:
        raise ValueError("FreeGSNKE target roster requires development/evaluation")
    ordered_preparations = tuple(sorted(preparations, key=lambda value: value.preparation_id))
    if (
        not ordered_preparations
        or len({value.preparation_id for value in ordered_preparations})
        != len(ordered_preparations)
        or any(value.phase is not phase for value in ordered_preparations)
    ):
        raise ValueError("FreeGSNKE preparation roster identity/phase differs")
    if any(value.saved_state is None for value in ordered_preparations):
        raise ValueError("FreeGSNKE follow-up preparations require saved-state identity")
    saved_states = tuple(
        value.saved_state for value in ordered_preparations if value.saved_state is not None
    )
    if len({value.artifact_id for value in saved_states}) != len(saved_states) or len(
        {value.sha256 for value in saved_states}
    ) != len(saved_states):
        raise ValueError("FreeGSNKE preparations reuse a saved state")
    if len({value.seed for value in ordered_preparations}) != len(ordered_preparations):
        raise ValueError("FreeGSNKE preparation seeds repeat")
    if len({value.request_id for value in requests}) != len(requests):
        raise ValueError("FreeGSNKE request identities repeat")
    branch_by_id = {value.branch_id: value for value in action_design.branches}
    preparation_by_id = {value.preparation_id: value for value in ordered_preparations}
    source_identities: set[ObjectIdentity] = set()
    for preparation_id, preparation in preparation_by_id.items():
        local = tuple(
            value for value in requests if value.preparation.preparation_id == preparation_id
        )
        if {value.branch_id for value in local} != set(branch_by_id) or len(local) != len(
            branch_by_id
        ):
            raise ValueError("FreeGSNKE request roster differs from its action design")
        first = local[0]
        for request in local:
            branch = branch_by_id[request.branch_id]
            if (
                request.preparation != preparation
                or request.phase is not phase
                or request.source_binding != first.source_binding
                or request.numerical_view != first.numerical_view
                or request.action_chart != first.action_chart
            ):
                raise ValueError("FreeGSNKE preparation branches cross a frozen operand")
            observed = {value.value_id: value.value for value in request.requested_port_increments}
            if observed != _expected_increments(request, branch):
                raise ValueError("FreeGSNKE request dose differs from its branch spec")
            if request.branch_kind is not branch.branch_kind:
                raise ValueError("FreeGSNKE request kind differs from its branch spec")
        source_identities.add(
            ObjectIdentity.from_record(first.source_binding.binding_id, first.source_binding)
        )
    if (
        any(value.preparation.preparation_id not in preparation_by_id for value in requests)
        or len(source_identities) != 1
    ):
        raise ValueError("FreeGSNKE phase roster crosses source or preparation scope")
    ordered_requests = tuple(sorted(requests, key=lambda value: value.request_id))
    return FreeGsnkePhaseRequestRoster(
        roster_id=roster_id,
        phase=phase,
        action_design=ObjectIdentity.from_record(action_design.design_id, action_design),
        preparations=tuple(
            ObjectIdentity.from_record(value.preparation_id, value)
            for value in ordered_preparations
        ),
        requests=tuple(
            ObjectIdentity.from_record(value.request_id, value) for value in ordered_requests
        ),
        independent_preparation_count=len(ordered_preparations),
        nested_branch_count=len(ordered_requests),
        initial_panel_minimum=FREEGSNKE_INITIAL_PANEL_MINIMUMS[phase],
        panel_envelope_limited=(
            len(ordered_preparations) < FREEGSNKE_INITIAL_PANEL_MINIMUMS[phase]
        ),
    )


def author_freegsnke_phase_request_roster(
    *,
    roster_id: str,
    phase: FreeGsnkePhase,
    source_binding: FreeGsnkeSourceBinding,
    numerical_view: FreeGsnkeNumericalView,
    action_design: FreeGsnkeActionDesign,
    prepared_units: tuple[FreeGsnkePreparedUnit, ...],
) -> tuple[FreeGsnkePhaseRequestRoster, tuple[FreeGsnkeProcessRequest, ...]]:
    """Author every issued branch from exact eligible preparation outputs.

    This helper deliberately refuses ineligible units instead of filtering
    them.  A caller must first account for the complete prospective candidate
    batch and pass only the exact eligible identities named by that batch.
    """

    if phase not in FREEGSNKE_DEVELOPMENT_EVALUATION_PHASES:
        raise ValueError("FreeGSNKE target request authoring requires development/evaluation")
    ordered_units = tuple(sorted(prepared_units, key=lambda value: value.unit_id))
    if (
        not ordered_units
        or len({value.unit_id for value in ordered_units}) != len(ordered_units)
        or len({value.recipe.preparation_id for value in ordered_units}) != len(ordered_units)
    ):
        raise ValueError("FreeGSNKE prepared-unit authoring roster differs")
    source_identity = ObjectIdentity.from_record(source_binding.binding_id, source_binding)
    view_identity = ObjectIdentity.from_record(numerical_view.view_id, numerical_view)
    preparations: list[FreeGsnkePreparation] = []
    requests: list[FreeGsnkeProcessRequest] = []
    for unit in ordered_units:
        if not unit.eligible:
            raise ValueError("FreeGSNKE request authoring cannot filter an ineligible preparation")
        preparation = unit.preparation
        action_chart = unit.action_chart
        saved = unit.saved_preparation
        if preparation is None or action_chart is None or saved is None:
            raise ValueError("FreeGSNKE eligible prepared unit lacks its exact payload")
        if (
            unit.recipe.phase is not phase
            or preparation.phase is not phase
            or saved.phase is not phase
            or saved.source_binding != source_identity
            or saved.numerical_view != view_identity
            or saved.action_chart != ObjectIdentity.from_record(action_chart.chart_id, action_chart)
        ):
            raise ValueError("FreeGSNKE prepared unit crosses phase/source/view/action scope")
        full_doses = {value.port_id: value.full_increment_v for value in action_chart.ports}
        for branch in action_design.branches:
            fractions = {
                "p4": branch.p4_dose_fraction,
                "p5": branch.p5_dose_fraction,
            }
            requests.append(
                FreeGsnkeProcessRequest(
                    request_id=f"request.{preparation.preparation_id}.{branch.branch_id}",
                    source_binding=source_binding,
                    preparation=preparation,
                    numerical_view=numerical_view,
                    action_chart=action_chart,
                    phase=phase,
                    branch_id=branch.branch_id,
                    branch_kind=branch.branch_kind,
                    requested_port_increments=tuple(
                        NamedDecimal(
                            value_id=port_id,
                            value=fractions[port_id] * full_doses[port_id],
                            unit="V",
                        )
                        for port_id in FREEGSNKE_PORT_IDS
                    ),
                )
            )
        preparations.append(preparation)
    ordered_requests = tuple(sorted(requests, key=lambda value: value.request_id))
    roster = freeze_freegsnke_phase_request_roster(
        roster_id=roster_id,
        phase=phase,
        action_design=action_design,
        preparations=tuple(preparations),
        requests=ordered_requests,
    )
    return roster, ordered_requests


__all__ = [
    "FREEGSNKE_INITIAL_PANEL_MINIMUMS",
    "FREEGSNKE_DEVELOPMENT_EVALUATION_PHASES",
    'FreeGsnkeActionDesign',
    'FreeGsnkeBranchSpec',
    'FreeGsnkePhaseRequestRoster',
    "author_freegsnke_phase_request_roster",
    "freeze_freegsnke_phase_request_roster",
]

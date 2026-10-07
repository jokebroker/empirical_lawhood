'Pure ambient pressure superconductor exploration, gate-lowering and admission-selection methods.'

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, Final

from empirical_lawhood.kernel.admission import AdmissionGateKind, AdmissionGateResult, GateStatus
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)

from .material_source_design_contracts import MaterialGateDisposition
from .material_scientific_inputs import (
    MaterialAdmissionScientificInput,
    MaterialExplorationScientificInput,
    require_material_admission_scientific_input,
    require_material_exploration_scientific_input,
)


def _nonnegative(value: int, *, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{field_name} must be a nonnegative integer')


@dataclass(frozen=True, slots=True)
class ExplorationCandidate(CanonicalRecord):
    """One prefix-visible candidate-route action; target outcomes are absent."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/exploration-candidate'

    action_id: str
    family_id: str
    parent_action_id: str | None
    gate_margin_vector: tuple[tuple[str, Decimal], ...]
    unresolved_chart_overlap: bool
    sign_change_bracket: bool
    phase_boundary_bracket: bool
    common_margin_uncertainty: Decimal
    scalar_predicted_tc_K: Decimal
    action_cost_units: int
    compute_cost_units: int
    discontinuous_bridge: bool
    family_restart: bool
    available: bool
    hard_prechecks_pass: bool
    irrecoverable: bool
    explicitly_out_of_support: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.action_id, field_name="action_id")
        validate_stable_id(self.family_id, field_name="family_id")
        if self.parent_action_id is not None:
            validate_stable_id(self.parent_action_id, field_name="parent_action_id")
        if tuple(sorted(self.gate_margin_vector)) != self.gate_margin_vector:
            raise ValueError("gate margins must be sorted by gate identity")
        if len({name for name, _value in self.gate_margin_vector}) != len(self.gate_margin_vector):
            raise ValueError("gate margin vector contains a duplicate gate")
        for gate_id, value in self.gate_margin_vector:
            validate_stable_id(gate_id, field_name="gate_margin_vector.gate_id")
            validate_decimal(value, field_name="gate_margin_vector.value")
        validate_decimal(
            self.common_margin_uncertainty,
            field_name="common_margin_uncertainty",
            minimum=Decimal("0"),
        )
        validate_decimal(self.scalar_predicted_tc_K, field_name="scalar_predicted_tc_K")
        if self.action_cost_units <= 0 or self.compute_cost_units <= 0:
            raise ValueError("exploration costs must be positive integers")


@dataclass(frozen=True, slots=True)
class ExplorationHistory(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/exploration-history'

    policy_id: str
    observed_action_ids: tuple[str, ...]
    observed_family_ids: tuple[str, ...]
    charged_action_units: int
    charged_compute_units: int
    family_restart_count: int
    deterministic_seed: str

    def __post_init__(self) -> None:
        validate_stable_id(self.policy_id, field_name="policy_id")
        require_sorted_unique_strings(
            self.observed_action_ids,
            field_name="observed_action_ids",
        )
        require_sorted_unique_strings(
            self.observed_family_ids,
            field_name="observed_family_ids",
        )
        for field_name in (
            "charged_action_units",
            "charged_compute_units",
            "family_restart_count",
        ):
            _nonnegative(getattr(self, field_name), field_name=field_name)
        validate_nonempty(self.deterministic_seed, field_name="deterministic_seed")


@dataclass(frozen=True, slots=True)
class ExplorationNomination(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/exploration-nomination'

    nomination_id: str
    policy_id: str
    selected_action_ids: tuple[str, ...]
    hold: bool
    action_units_charged: int
    compute_units_charged: int
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        validate_stable_id(self.policy_id, field_name="policy_id")
        require_sorted_unique_strings(
            self.selected_action_ids,
            field_name="selected_action_ids",
        )
        for field_name in ("action_units_charged", "compute_units_charged"):
            _nonnegative(getattr(self, field_name), field_name=field_name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.hold == bool(self.selected_action_ids):
            raise ValueError("nomination must either select actions or emit HOLD")
        if self.hold and not self.reason_codes:
            raise ValueError("HOLD requires a reason")
        if not self.hold and self.reason_codes:
            raise ValueError("a positive nomination cannot carry failure reasons")


def _dominates(left: ExplorationCandidate, right: ExplorationCandidate) -> bool:
    left_margins = dict(left.gate_margin_vector)
    right_margins = dict(right.gate_margin_vector)
    if set(left_margins) != set(right_margins):
        return False
    weak = all(left_margins[name] >= right_margins[name] for name in left_margins)
    strict = any(left_margins[name] > right_margins[name] for name in left_margins)
    return weak and strict


def _pareto_front(candidates: tuple[ExplorationCandidate, ...]) -> tuple[ExplorationCandidate, ...]:
    return tuple(
        candidate
        for candidate in candidates
        if not any(
            other.action_id != candidate.action_id and _dominates(other, candidate)
            for other in candidates
        )
    )


def nominate_exploration_wave(
    *,
    policy_id: str,
    candidates: tuple[ExplorationCandidate, ...],
    history: ExplorationHistory,
    actions_per_wave: int,
    action_budget: int,
    compute_budget: int,
    family_restart_quota: int,
    bridge_quota: int,
    wave_index: int,
    scientific_input: MaterialExplorationScientificInput | None = None,
) -> ExplorationNomination:
    'Apply one of the three frozen, outcome-blind material source design policy rules.'

    scientific_input = require_material_exploration_scientific_input(
        scientific_input, candidates=candidates, history=history, wave_index=wave_index,
        budget_operands=(actions_per_wave, action_budget, compute_budget, family_restart_quota, bridge_quota),
    )
    scientific_ranks = {row.current_action_id: row for row in scientific_input.action_ranks}

    if policy_id != history.policy_id:
        raise ValueError("policy history cannot cross policy identities")
    if actions_per_wave <= 0 or wave_index <= 0:
        raise ValueError("wave size and index must be positive")
    known_ids = {candidate.action_id for candidate in candidates}
    if len(known_ids) != len(candidates):
        raise ValueError("eligible action graph contains duplicate action identities")
    if set(history.observed_action_ids) - known_ids:
        raise ValueError("history references an action outside the eligible graph")

    eligible = tuple(
        candidate
        for candidate in candidates
        if candidate.available
        and candidate.hard_prechecks_pass
        and not candidate.irrecoverable
        and not candidate.explicitly_out_of_support
        and candidate.action_id not in history.observed_action_ids
        and (
            candidate.parent_action_id is None
            or candidate.parent_action_id in history.observed_action_ids
        )
    )
    if policy_id == "policy.response-guided":
        pool = _pareto_front(eligible)
        ordered = sorted(
            pool,
            key=lambda candidate: (
                not candidate.unresolved_chart_overlap,
                not candidate.sign_change_bracket,
                not candidate.phase_boundary_bracket,
                -candidate.common_margin_uncertainty,
                candidate.family_id in history.observed_family_ids,
                scientific_ranks[candidate.action_id].scientific_order_index,
            ),
        )
    elif policy_id == "policy.scalar-predicted-tc":
        ordered = sorted(
            eligible,
            key=lambda candidate: (
                -candidate.scalar_predicted_tc_K,
                candidate.family_id in history.observed_family_ids,
                scientific_ranks[candidate.action_id].scientific_order_index,
            ),
        )
    elif policy_id == "policy.stratified-random":
        ordered = sorted(
            eligible,
            key=lambda candidate: (
                candidate.family_id in history.observed_family_ids,
                scientific_ranks[candidate.action_id].full_original_random_key_sha256,
                scientific_ranks[candidate.action_id].scientific_order_index,
            ),
        )
    else:
        raise ValueError("unqualified exploration policy")

    remaining_action = action_budget - history.charged_action_units
    remaining_compute = compute_budget - history.charged_compute_units
    selected: list[ExplorationCandidate] = []
    selected_families: set[str] = set()
    bridge_count = 0
    restart_count = history.family_restart_count
    for candidate in ordered:
        if len(selected) >= actions_per_wave:
            break
        if candidate.action_cost_units > remaining_action:
            continue
        if candidate.compute_cost_units > remaining_compute:
            continue
        if candidate.discontinuous_bridge and bridge_count >= bridge_quota:
            continue
        if candidate.family_restart and restart_count >= family_restart_quota:
            continue
        # Within a wave, cover distinct families before taking a second action
        # from the same family.  This is deterministic and outcome blind.
        if candidate.family_id in selected_families and any(
            other.family_id not in selected_families for other in ordered
        ):
            continue
        selected.append(candidate)
        selected_families.add(candidate.family_id)
        remaining_action -= candidate.action_cost_units
        remaining_compute -= candidate.compute_cost_units
        bridge_count += int(candidate.discontinuous_bridge)
        restart_count += int(candidate.family_restart)

    selected_ids = tuple(sorted(candidate.action_id for candidate in selected))
    hold = not selected_ids
    return ExplorationNomination(
        nomination_id=f"nomination.{policy_id.removeprefix('policy.')}.wave-{wave_index}",
        policy_id=policy_id,
        selected_action_ids=selected_ids,
        hold=hold,
        action_units_charged=sum(candidate.action_cost_units for candidate in selected),
        compute_units_charged=sum(candidate.compute_cost_units for candidate in selected),
        reason_codes=("reason.no-valid-budget-feasible-nomination",) if hold else (),
    )


@dataclass(frozen=True, slots=True)
class MaterialGateObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-gate-observation'

    gate_id: str
    generic_gate_id: str
    status: MaterialGateDisposition
    native_margin: NamedDecimal | None
    threshold_id: str
    reason_codes: tuple[str, ...]
    evidence_link_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for field_name in ("gate_id", "generic_gate_id", "threshold_id"):
            validate_stable_id(getattr(self, field_name), field_name=field_name)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        require_sorted_unique_strings(
            self.evidence_link_ids,
            field_name="evidence_link_ids",
        )
        if self.status is MaterialGateDisposition.PASS:
            if self.reason_codes or not self.evidence_link_ids:
                raise ValueError("passing material gate needs evidence and no failure reason")
        elif not self.reason_codes:
            raise ValueError("failed or unevaluable material gate needs a reason")


@dataclass(frozen=True, slots=True)
class MaterialGatePanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-gate-panel'

    panel_id: str
    cell_id: str
    observations: tuple[MaterialGateObservation, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.panel_id, field_name="panel_id")
        validate_stable_id(self.cell_id, field_name="cell_id")
        require_sorted_unique_ids(
            self.observations,
            attribute="gate_id",
            field_name="observations",
        )
        if len(self.observations) != 16:
            raise ValueError("material gate panel must preserve all 16 native gates")


_GENERIC_KINDS: Final = {
    "generic.authority": AdmissionGateKind.AUTHORITY,
    "generic.baseline-preservation": AdmissionGateKind.BASELINE_PRESERVATION,
    "generic.dynamics": AdmissionGateKind.DYNAMICS,
    "generic.effort": AdmissionGateKind.EFFORT,
    "generic.observation-validity": AdmissionGateKind.OBSERVATION_VALIDITY,
    "generic.physical-sink": AdmissionGateKind.PHYSICAL_SINK,
    "generic.reachability": AdmissionGateKind.REACHABILITY,
    "generic.target": AdmissionGateKind.TARGET,
    "generic.uncertainty": AdmissionGateKind.UNCERTAINTY,
}


def lower_material_gate_panel(
    panel: MaterialGatePanel,
) -> tuple[tuple[AdmissionGateResult, ...], MaterialGateDisposition]:
    """Mechanically lower 15 gates to nine; return atlas-native SUPPORT separately."""

    support_rows = tuple(
        row for row in panel.observations if row.generic_gate_id == "generic.atlas-native"
    )
    if len(support_rows) != 1:
        raise ValueError("material panel must contain exactly one atlas-native SUPPORT row")
    grouped: dict[str, list[MaterialGateObservation]] = {
        generic_id: [] for generic_id in _GENERIC_KINDS
    }
    for row in panel.observations:
        if row.generic_gate_id == "generic.atlas-native":
            continue
        try:
            grouped[row.generic_gate_id].append(row)
        except KeyError as error:
            raise ValueError("material gate has no frozen generic lowering") from error
    if any(not rows for rows in grouped.values()):
        raise ValueError("material panel does not cover all nine generic gates")

    results = []
    for generic_id, kind in _GENERIC_KINDS.items():
        rows = grouped[generic_id]
        if any(row.status is MaterialGateDisposition.FAIL for row in rows):
            status = GateStatus.FAIL
        elif any(row.status is MaterialGateDisposition.UNEVALUABLE for row in rows):
            status = GateStatus.UNEVALUABLE
        else:
            status = GateStatus.PASS
        reason_codes = (
            tuple(sorted({reason for row in rows for reason in row.reason_codes}))
            if status is not GateStatus.PASS
            else ()
        )
        evidence = tuple(sorted({link for row in rows for link in row.evidence_link_ids}))
        results.append(
            AdmissionGateResult(
                gate_id=f"gate.{generic_id.removeprefix('generic.')}",
                kind=kind,
                status=status,
                constraint_ids=tuple(sorted(row.gate_id for row in rows)),
                # Native margins have different units and are intentionally not
                # scalarized into a generic margin.
                margin=None,
                reason_codes=reason_codes,
                evidence_link_ids=evidence,
            )
        )
    return tuple(sorted(results, key=lambda value: value.gate_id)), support_rows[0].status


@dataclass(frozen=True, slots=True)
class MaterialAdmissionCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-admission-candidate'

    cell_id: str
    action_id: str
    normalized_gate_margins: tuple[tuple[str, Decimal], ...]
    all_mandatory_gates_pass: bool
    in_supported_atlas_cell: bool
    synthesis_reachable: bool
    requested_accepted_realized_match: bool
    effort_units: int

    def __post_init__(self) -> None:
        validate_stable_id(self.cell_id, field_name="cell_id")
        validate_stable_id(self.action_id, field_name="action_id")
        if tuple(sorted(self.normalized_gate_margins)) != self.normalized_gate_margins:
            raise ValueError("normalized margins must use canonical gate order")
        if not self.normalized_gate_margins:
            raise ValueError("admission candidate requires complete normalized margins")
        for gate_id, margin in self.normalized_gate_margins:
            validate_stable_id(gate_id, field_name="normalized_gate_margins.gate_id")
            validate_decimal(margin, field_name="normalized_gate_margins.margin")
        if self.effort_units <= 0:
            raise ValueError("effort must be positive")


@dataclass(frozen=True, slots=True)
class MaterialActionSelection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/ambient-pressure-superconductor/material-action-selection'

    selection_id: str
    selected_cell_id: str | None
    selected_action_id: str | None
    minimum_normalized_margin: Decimal | None
    hold: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.selection_id, field_name="selection_id")
        if self.selected_cell_id is not None:
            validate_stable_id(self.selected_cell_id, field_name="selected_cell_id")
        if self.selected_action_id is not None:
            validate_stable_id(self.selected_action_id, field_name="selected_action_id")
        if self.minimum_normalized_margin is not None:
            validate_decimal(
                self.minimum_normalized_margin,
                field_name="minimum_normalized_margin",
            )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        selected = self.selected_cell_id is not None and self.selected_action_id is not None
        if self.hold == selected:
            raise ValueError("selection must either bind one action or HOLD")
        if self.hold and (self.minimum_normalized_margin is not None or not self.reason_codes):
            raise ValueError("HOLD must carry reasons and no selected margin")
        if not self.hold and (self.minimum_normalized_margin is None or self.reason_codes):
            raise ValueError("selected action must carry its robust margin and no failure")


def select_material_action(
    *,
    candidates: tuple[MaterialAdmissionCandidate, ...],
    maximum_effort_units: int,
    scientific_input: MaterialAdmissionScientificInput | None = None,
) -> MaterialActionSelection:
    """Select the maximum minimum-margin action or deterministically HOLD."""

    scientific_input = require_material_admission_scientific_input(
        scientific_input, candidates=candidates, maximum_effort_units=maximum_effort_units,
    )
    scientific_orders = {(row.current_cell_id, row.current_action_id): row.scientific_order_index
                         for row in scientific_input.action_orders}

    if maximum_effort_units <= 0:
        raise ValueError("maximum effort must be positive")
    eligible: list[tuple[Decimal, MaterialAdmissionCandidate]] = []
    for candidate in candidates:
        if not (
            candidate.all_mandatory_gates_pass
            and candidate.in_supported_atlas_cell
            and candidate.synthesis_reachable
            and candidate.requested_accepted_realized_match
            and candidate.effort_units <= maximum_effort_units
        ):
            continue
        robust_margin = min(margin for _gate_id, margin in candidate.normalized_gate_margins)
        if robust_margin > 0:
            eligible.append((robust_margin, candidate))
    if not eligible:
        return MaterialActionSelection(
            selection_id='selection.ambient-pressure-superconductor-max-min',
            selected_cell_id=None,
            selected_action_id=None,
            minimum_normalized_margin=None,
            hold=True,
            reason_codes=("reason.empty-noncompensating-intersection",),
        )
    margin, selected = sorted(
        eligible,
        key=lambda item: (-item[0], scientific_orders[(item[1].cell_id, item[1].action_id)]),
    )[0]
    return MaterialActionSelection(
        selection_id='selection.ambient-pressure-superconductor-max-min',
        selected_cell_id=selected.cell_id,
        selected_action_id=selected.action_id,
        minimum_normalized_margin=margin,
        hold=False,
        reason_codes=(),
    )


__all__ = [
    "ExplorationCandidate",
    "ExplorationHistory",
    "ExplorationNomination",
    "MaterialActionSelection",
    "MaterialAdmissionCandidate",
    "MaterialGateObservation",
    "MaterialGatePanel",
    "lower_material_gate_panel",
    "nominate_exploration_wave",
    "select_material_action",
]

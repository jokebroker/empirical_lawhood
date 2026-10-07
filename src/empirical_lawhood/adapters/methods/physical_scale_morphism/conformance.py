"""Truth-blind execution of every frozen physical scale morphism method fixture."""

from __future__ import annotations

from decimal import Decimal
from typing import Callable

import numpy as np

from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.contracts import PhysicalScaleMorphismTruthBlindObservation, PhysicalScaleMorphismTruthCase
from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderActionSegment, ResistorCapacitorLadderModelConfig
from empirical_lawhood.adapters.simulators.rc_ladder_response.refinement import refinement_chain, solve_backward_euler, summarize_against_matrix_exponential
from empirical_lawhood.kernel.evidence import OutcomeAccess

from .boundaries import PhysicalScaleMorphismBoundaryCellKind, PhysicalScaleMorphismBoundaryVertex, PhysicalScaleMorphismGateAssessment, build_boundary_complex
from .comparators import PhysicalScaleMorphismCollisionMember, PhysicalScaleMorphismComparatorEncoding, PhysicalScaleMorphismComparatorPrediction, PhysicalScaleMorphismCoordinateCollision, PhysicalScaleMorphismCoordinateDisposition, evaluate_coordinate_comparators
from .contracts import PhysicalScaleMorphismComparatorKind, PhysicalScaleMorphismForecastLevel, PhysicalScaleMorphismForecastMember, PhysicalScaleMorphismForecastTriplet, PhysicalScaleMorphismGateSign, PhysicalScaleMorphismHoldDisposition
from .defects import PhysicalScaleMorphismDefectThresholds, decision_defect, evaluate_hold_fibre, make_defect_panel
from .dimensionless import PhysicalScaleMorphismNormalizationSpec, PhysicalScaleMorphismNormalizationUncertainty, compute_dimensionless_record
from .heterogeneity import PhysicalScaleMorphismBoardLocalResult, evaluate_heterogeneity
from .inference import PhysicalScaleMorphismPowerMethod, PhysicalScaleMorphismPowerRequirement, PhysicalScaleMorphismPowerTerminal, qualify_method_power
from .receivers import PhysicalScaleMorphismReceiverKind, PhysicalScaleMorphismReceiverSpec, composition_defect, equal_width_receiver_spec, reduce_receiver
from .semantic_closure import PhysicalScaleMorphismSemanticFibrePair, evaluate_semantic_closure


class _TruthInput:
    """Read every reference-world datum exactly once inside one method case."""

    def __init__(self, case: PhysicalScaleMorphismTruthCase) -> None:
        self.case = case
        self._data = {value.datum_id: value for value in case.input_data}
        self._used: set[str] = set()

    def number(self, datum_id: str) -> Decimal:
        try:
            value = self._data[datum_id]
        except KeyError as error:
            raise ValueError(f"truth input lacks numeric datum {datum_id}") from error
        if value.numeric_value is None:
            raise ValueError(f"truth input datum {datum_id} is not numeric")
        self._used.add(datum_id)
        return value.numeric_value

    def integer(self, datum_id: str) -> int:
        value = self.number(datum_id)
        integer = int(value)
        if Decimal(integer) != value:
            raise ValueError(f"truth input datum {datum_id} is not integral")
        return integer

    def category(self, datum_id: str) -> str:
        try:
            value = self._data[datum_id]
        except KeyError as error:
            raise ValueError(f"truth input lacks categorical datum {datum_id}") from error
        if value.categorical_value is None:
            raise ValueError(f"truth input datum {datum_id} is not categorical")
        self._used.add(datum_id)
        return value.categorical_value

    def vector(self, prefix: str, size: int) -> np.ndarray:
        return np.asarray(
            [float(self.number(f"{prefix}-{index:03d}")) for index in range(size)],
            dtype=np.float64,
        )

    def require_complete_consumption(self) -> None:
        unused = set(self._data) - self._used
        if unused:
            raise ValueError(f"truth method left generated challenge data unused: {sorted(unused)}")


def _specification_exact(inputs: _TruthInput) -> tuple[str, ...]:
    capacitances = inputs.vector("capacitance", 16)
    direct = equal_width_receiver_spec(receiver_id="r4-direct", scale_cells=16, bin_count=4)
    first = equal_width_receiver_spec(receiver_id="r8-first", scale_cells=16, bin_count=8)
    second = equal_width_receiver_spec(
        receiver_id="r4-second",
        scale_cells=8,
        bin_count=4,
        source_receiver_id="r8-first",
    )
    defect = composition_defect(
        direct=direct,
        first=first,
        second=second,
        capacitances=capacitances,
    )
    return ("receiver-composition-exact",) if defect < 1e-14 else ("composition-defect",)


def _specification_scale(inputs: _TruthInput) -> tuple[str, ...]:
    values = []
    for n_cells in (16, 32, 64):
        spec = PhysicalScaleMorphismNormalizationSpec(
            normalization_id=f"normalization-{n_cells}",
            scale_cells=n_cells,
            component_basis_sha256=f"{n_cells:064x}",
            resistance_summary_rule_id="dossier-resistance-mean",
            capacitance_summary_rule_id="dossier-capacitance-mean",
            resistance_bar_ohms=Decimal("1000"),
            capacitance_bar_farads=Decimal("0.000001"),
            voltage_reference_volts=Decimal(1),
            realized_component_weights=True,
            diffusive_time_exponent=2,
        )
        time_scale = 1000 * 1e-6 * n_cells**2
        record = compute_dimensionless_record(
            record_id=f"record-{n_cells}",
            spec=spec,
            uncertainty=_zero_normalization_uncertainty(spec),
            voltages=inputs.vector(f"voltage-n{n_cells}", n_cells),
            capacitances_farads=np.full(n_cells, 1e-6),
            terminal_current_amperes=0.001 / n_cells,
            native_time_seconds=time_scale,
            slowest_decay_rate_per_second=2 / time_scale,
        )
        values.append({value.value_id: value.value for value in record.dimensionless_values})
    fixed_ids = ("e-star", "i-star", "lambda-star", "q-star", "t-star")
    fixed = all(
        max(float(value[quantity_id]) for value in values)
        - min(float(value[quantity_id]) for value in values)
        < 1e-6
        for quantity_id in fixed_ids
    )
    return ("dimensionless-fixed-section",) if fixed else ("dimensionless-drift",)


def _zero_normalization_uncertainty(
    spec: PhysicalScaleMorphismNormalizationSpec,
) -> PhysicalScaleMorphismNormalizationUncertainty:
    return PhysicalScaleMorphismNormalizationUncertainty(
        uncertainty_id=f"uncertainty.{spec.normalization_id}",
        normalization_id=spec.normalization_id,
        voltage_uncertainty_volts=tuple(Decimal(0) for _ in range(spec.scale_cells)),
        capacitance_uncertainty_farads=tuple(Decimal(0) for _ in range(spec.scale_cells)),
        resistance_bar_uncertainty_ohms=Decimal(0),
        capacitance_bar_uncertainty_farads=Decimal(0),
        voltage_reference_uncertainty_volts=Decimal(0),
        terminal_current_uncertainty_amperes=Decimal(0),
        native_time_uncertainty_seconds=Decimal(0),
        decay_rate_uncertainty_per_second=Decimal(0),
        dissipated_energy_uncertainty_joules=None,
        penetration_depth_uncertainty_cells=None,
        first_passage_time_uncertainty_seconds=None,
        simultaneous_rectangular_bound=True,
    )


def _collision_member(
    *,
    collision_id: str,
    side: str,
    changed_coordinate: str,
    changed_value: str,
    label: str,
) -> PhysicalScaleMorphismCollisionMember:
    values = {"A": "a-base", "D": "d-base", "H": "h-base", "R": "r-base", "tau": "tau-base"}
    values[changed_coordinate] = changed_value
    return PhysicalScaleMorphismCollisionMember(
        member_id=f"member.{collision_id}.{side}",
        complete_unit_id=f"unit.{collision_id}.{side}",
        denominator_id=values["D"],
        history_id=values["H"],
        action_id=values["A"],
        receiver_id=values["R"],
        horizon_id=values["tau"],
        categorical_label=label,
    )


def _prediction_roster(
    members: tuple[PhysicalScaleMorphismCollisionMember, ...],
    *,
    wrong_member_id: str | None,
    abstain: bool,
) -> tuple[PhysicalScaleMorphismComparatorPrediction, ...]:
    return tuple(
        sorted(
            (
                PhysicalScaleMorphismComparatorPrediction(
                    prediction_id=f"prediction.{value.member_id}",
                    member_id=value.member_id,
                    categorical_label=(
                        None
                        if abstain
                        else "wrong"
                        if value.member_id == wrong_member_id
                        else value.categorical_label
                    ),
                )
                for value in members
            ),
            key=lambda value: value.prediction_id,
        )
    )


_DEPENDENCIES = {
    PhysicalScaleMorphismComparatorKind.ACTION_ONLY: ("A",),
    PhysicalScaleMorphismComparatorKind.D_A: ("A", "D"),
    PhysicalScaleMorphismComparatorKind.FULL_TUPLE: ("A", "D", "H", "R", "tau"),
    PhysicalScaleMorphismComparatorKind.H_A: ("A", "H"),
    PhysicalScaleMorphismComparatorKind.R_A: ("A", "R"),
    PhysicalScaleMorphismComparatorKind.SATURATED_DEVELOPMENT_LOOKUP: (),
    PhysicalScaleMorphismComparatorKind.SELECTED_TYPED_SUBSET: ("A", "D", "H", "tau"),
    PhysicalScaleMorphismComparatorKind.TARGET_NATIVE_LABEL: ("A",),
    PhysicalScaleMorphismComparatorKind.TAU_A: ("A", "tau"),
    PhysicalScaleMorphismComparatorKind.WILDCARD: (),
}


def _tournament(
    *,
    panel_id: str,
    collisions: tuple[PhysicalScaleMorphismCoordinateCollision, ...],
    exact_kinds: frozenset[PhysicalScaleMorphismComparatorKind],
    selected_kind: PhysicalScaleMorphismComparatorKind,
) -> PhysicalScaleMorphismCoordinateDisposition:
    members = tuple(
        sorted(
            (value for collision in collisions for value in (collision.left, collision.right)),
            key=lambda value: value.member_id,
        )
    )
    wrong_member_id = members[-1].member_id
    encodings = tuple(
        sorted(
            (
                PhysicalScaleMorphismComparatorEncoding(
                    encoding_id=f"encoding.{kind.value.lower().replace('_', '-')}",
                    kind=kind,
                    dependency_coordinate_ids=(
                        ("A",)
                        if selected_kind is PhysicalScaleMorphismComparatorKind.ACTION_ONLY
                        and kind is PhysicalScaleMorphismComparatorKind.SELECTED_TYPED_SUBSET
                        else _DEPENDENCIES[kind]
                    ),
                    predictions=_prediction_roster(
                        members,
                        wrong_member_id=None if kind in exact_kinds else wrong_member_id,
                        abstain=kind is PhysicalScaleMorphismComparatorKind.WILDCARD and kind not in exact_kinds,
                    ),
                    fitted_complete_unit_ids=("development-board-01",),
                    evaluation_outcome_count_at_freeze=0,
                )
                for kind in PhysicalScaleMorphismComparatorKind
            ),
            key=lambda value: value.encoding_id,
        )
    )
    selected = next(value for value in encodings if value.kind is selected_kind)
    return evaluate_coordinate_comparators(
        panel_id=panel_id,
        selected_encoding_id=selected.encoding_id,
        collisions=collisions,
        encodings=encodings,
    ).disposition


def _specification_saturation(inputs: _TruthInput) -> tuple[str, ...]:
    saturation_left = _collision_member(
        collision_id="collision-a",
        side="left",
        changed_coordinate="A",
        changed_value="a-left",
        label=inputs.category("saturation-left-label"),
    )
    saturation_right = _collision_member(
        collision_id="collision-a",
        side="right",
        changed_coordinate="A",
        changed_value="a-right",
        label=inputs.category("saturation-right-label"),
    )
    saturation = _tournament(
        panel_id="panel-saturation",
        collisions=(
            PhysicalScaleMorphismCoordinateCollision(
                collision_id="collision-a",
                challenged_coordinate_id="A",
                left=saturation_left,
                right=saturation_right,
                outcome_required_to_differ=True,
                evaluation_held_out=True,
            ),
        ),
        exact_kinds=frozenset(
            {
                PhysicalScaleMorphismComparatorKind.ACTION_ONLY,
                PhysicalScaleMorphismComparatorKind.FULL_TUPLE,
                PhysicalScaleMorphismComparatorKind.SELECTED_TYPED_SUBSET,
            }
        ),
        selected_kind=PhysicalScaleMorphismComparatorKind.ACTION_ONLY,
    )
    collisions = []
    for coordinate in ("A", "D", "H", "R", "tau"):
        collision_id = f"collision-selective-{coordinate.lower()}"
        differs = coordinate != "R"
        collisions.append(
            PhysicalScaleMorphismCoordinateCollision(
                collision_id=collision_id,
                challenged_coordinate_id=coordinate,
                left=_collision_member(
                    collision_id=collision_id,
                    side="left",
                    changed_coordinate=coordinate,
                    changed_value=f"{coordinate.lower()}-left",
                    label=inputs.category(f"selective-{coordinate.lower()}-left-label"),
                ),
                right=_collision_member(
                    collision_id=collision_id,
                    side="right",
                    changed_coordinate=coordinate,
                    changed_value=f"{coordinate.lower()}-right",
                    label=inputs.category(f"selective-{coordinate.lower()}-right-label"),
                ),
                outcome_required_to_differ=differs,
                evaluation_held_out=True,
            )
        )
    selected = _tournament(
        panel_id="panel-selective",
        collisions=tuple(sorted(collisions, key=lambda value: value.collision_id)),
        exact_kinds=frozenset(
            {PhysicalScaleMorphismComparatorKind.FULL_TUPLE, PhysicalScaleMorphismComparatorKind.SELECTED_TYPED_SUBSET}
        ),
        selected_kind=PhysicalScaleMorphismComparatorKind.SELECTED_TYPED_SUBSET,
    )
    codes = []
    if saturation is PhysicalScaleMorphismCoordinateDisposition.TASK_SATURATED_BY_ACTION_ONLY_OR_WILDCARD:
        codes.append("action-saturation-detected")
    if selected is PhysicalScaleMorphismCoordinateDisposition.COORDINATE_NECESSITY_IDENTIFIED:
        codes.append("active-subset-recovered")
    return tuple(sorted(codes or ("coordinate-tournament-failed",)))


def _specification_hidden(inputs: _TruthInput) -> tuple[str, ...]:
    h0 = evaluate_semantic_closure(
        panel_id="semantic-h0",
        morphism_id="morphism-h0",
        preserved_role_ids=("action", "receiver"),
        omitted_role_ids=("history",),
        merged_role_ids=(),
        fibre_pairs=(
            PhysicalScaleMorphismSemanticFibrePair(
                pair_id="pair-hidden",
                left_unit_id="hidden-a",
                right_unit_id="hidden-b",
                present_defect=inputs.number("h0-present-defect"),
                future_defect=inputs.number("h0-future-defect"),
                role_ids=("history",),
            ),
        ),
        present_tolerance=Decimal("0.01"),
        future_tolerance=Decimal("0.05"),
    )
    h1 = evaluate_semantic_closure(
        panel_id="semantic-h1",
        morphism_id="morphism-h1",
        preserved_role_ids=("action", "history", "receiver"),
        omitted_role_ids=(),
        merged_role_ids=(),
        fibre_pairs=(
            PhysicalScaleMorphismSemanticFibrePair(
                pair_id="pair-hidden",
                left_unit_id="hidden-a",
                right_unit_id="hidden-b",
                present_defect=inputs.number("h1-present-defect"),
                future_defect=inputs.number("h1-future-defect"),
                role_ids=("history",),
            ),
        ),
        present_tolerance=Decimal("0.01"),
        future_tolerance=Decimal("0.05"),
    )
    codes = []
    if not h0.closure_supported and h0.future_diverged_pair_ids:
        codes.append("h0-future-divergence")
    if h1.closure_supported:
        codes.append("richer-history-recovers")
    return tuple(sorted(codes or ("history-closure-failed",)))


def _specification_future(inputs: _TruthInput) -> tuple[str, ...]:
    thresholds = PhysicalScaleMorphismDefectThresholds(
        thresholds_id="thresholds-future",
        calibration=Decimal("0.01"),
        observational=Decimal("0.01"),
        held_future_semantic=Decimal("0.05"),
        interventional=Decimal("0.05"),
        decision=Decimal("0.05"),
    )
    panel = make_defect_panel(
        panel_id="defects-future",
        morphism_id="morphism-future",
        thresholds=thresholds,
        calibration_defect=inputs.number("calibration-defect"),
        observational_defect=inputs.number("observational-defect"),
        held_future_semantic_defect=inputs.number("held-future-defect"),
        interventional_defect=inputs.number("interventional-defect"),
        decision_defect=inputs.number("decision-defect"),
    )
    return ("calibration-not-closure",) if panel.observational_only else ("future-promoted",)


def _specification_guard(inputs: _TruthInput) -> tuple[str, ...]:
    defect = decision_defect(
        fine_signs=tuple(
            PhysicalScaleMorphismGateSign(inputs.category(f"fine-gate-{index}")) for index in range(2)
        ),
        mapped_signs=tuple(
            PhysicalScaleMorphismGateSign(inputs.category(f"mapped-gate-{index}")) for index in range(2)
        ),
    )
    return ("mean-false-safe-rejected",) if defect > 0 else ("false-safe-missed",)


def _specification_hold(inputs: _TruthInput) -> tuple[str, ...]:
    safe = evaluate_hold_fibre(
        hold_id="hold-safe",
        context_id="context-safe",
        board_id="board-safe",
        realized_episode_id="episode-safe",
        gate_signs=tuple(
            PhysicalScaleMorphismGateSign(inputs.category(f"safe-gate-{index}")) for index in range(2)
        ),
        mapped_disposition=None,
    )
    unsafe = evaluate_hold_fibre(
        hold_id="hold-unsafe",
        context_id="context-unsafe",
        board_id="board-unsafe",
        realized_episode_id="episode-unsafe",
        gate_signs=tuple(
            PhysicalScaleMorphismGateSign(inputs.category(f"unsafe-gate-{index}")) for index in range(2)
        ),
        mapped_disposition=PhysicalScaleMorphismHoldDisposition(inputs.category("unsafe-mapped-disposition")),
    )
    unqualified = evaluate_hold_fibre(
        hold_id="hold-unqualified",
        context_id="context-unqualified",
        board_id="board-unqualified",
        realized_episode_id="episode-unqualified",
        gate_signs=(PhysicalScaleMorphismGateSign(inputs.category("unqualified-gate-0")),),
        mapped_disposition=None,
    )
    dispositions = {safe.disposition, unsafe.disposition, unqualified.disposition}
    codes = []
    if len(dispositions) == 3:
        codes.append("hold-states-distinct")
    if unsafe.false_safe_hold:
        codes.append("default-safe-rejected")
    return tuple(sorted(codes or ("hold-grammar-failed",)))


def _specification_energy(inputs: _TruthInput) -> tuple[str, ...]:
    receiver = equal_width_receiver_spec(receiver_id="r8-mean", scale_cells=16, bin_count=8)
    left = inputs.vector("left-voltage", 16)
    right = inputs.vector("right-voltage", 16)
    caps = np.ones(16)
    left_observation = reduce_receiver(
        observation_id="observation-energy-left",
        spec=receiver,
        voltages=left,
        voltage_uncertainties=np.zeros(16),
        capacitances=caps,
        capacitance_uncertainties=np.zeros(16),
        terminal_currents=np.zeros(2),
        terminal_current_uncertainties=np.zeros(2),
        uncertainty_method_id="synthetic-exact",
    )
    right_observation = reduce_receiver(
        observation_id="observation-energy-right",
        spec=receiver,
        voltages=right,
        voltage_uncertainties=np.zeros(16),
        capacitances=caps,
        capacitance_uncertainties=np.zeros(16),
        terminal_currents=np.zeros(2),
        terminal_current_uncertainties=np.zeros(2),
        uncertainty_method_id="synthetic-exact",
    )
    means_close = (
        max(
            abs(float(a - b))
            for a, b in zip(
                left_observation.coordinate_values_volts,
                right_observation.coordinate_values_volts,
                strict=True,
            )
        )
        < 1e-6
    )
    energy_differs = abs(float(np.sum(left**2) - np.sum(right**2))) > 1.0
    return (
        ("mean-collision-energy-separated",)
        if means_close and energy_differs
        else ("energy-collision-failed",)
    )


def _boundary_gate(vertex_id: str, gate_id: str, sign: PhysicalScaleMorphismGateSign) -> PhysicalScaleMorphismGateAssessment:
    return PhysicalScaleMorphismGateAssessment(
        assessment_id=f"assessment.{vertex_id}.{gate_id}", gate_id=gate_id, sign=sign
    )


def _specification_boundary(inputs: _TruthInput) -> tuple[str, ...]:
    vertices = []
    for u_index in range(3):
        for d_index in range(3):
            vertex_id = f"vertex-{u_index}-{d_index}"
            target = PhysicalScaleMorphismGateSign(inputs.category(f"{vertex_id}-target"))
            sink = PhysicalScaleMorphismGateSign(inputs.category(f"{vertex_id}-sink"))
            gates = tuple(
                sorted(
                    (
                        _boundary_gate(vertex_id, "sink", sink),
                        _boundary_gate(vertex_id, "target", target),
                    ),
                    key=lambda value: value.assessment_id,
                )
            )
            vertices.append(
                PhysicalScaleMorphismBoundaryVertex(
                    vertex_id=vertex_id,
                    u_index=u_index,
                    duration_index=d_index,
                    u_star=Decimal(u_index),
                    duration_star=Decimal(d_index),
                    gates=gates,
                )
            )
    complex_ = build_boundary_complex(
        complex_id="boundary-truth",
        chart_id="chart-truth",
        vertices=tuple(sorted(vertices, key=lambda value: value.vertex_id)),
    )
    boundary_cells = tuple(
        value for value in complex_.cells if value.kind is PhysicalScaleMorphismBoundaryCellKind.BOUNDARY
    )
    codes = []
    if complex_.components and boundary_cells:
        codes.append("boundary-topology-recovered")
    if any(value.codimension == 2 for value in boundary_cells):
        codes.append("codimension-two-recovered")
    return tuple(sorted(codes or ("boundary-recovery-failed",)))


def _numerical_config(inputs: _TruthInput) -> ResistorCapacitorLadderModelConfig:
    return ResistorCapacitorLadderModelConfig(
        config_id=f"numerical-{inputs.case.case_id}",
        scale_cells=4,
        capacitances_farads=tuple(inputs.number(f"capacitance-{index:03d}") for index in range(4)),
        interior_resistances_ohms=(Decimal(80), Decimal(100), Decimal(120)),
        left_source_resistance_ohms=Decimal(50),
        right_termination_resistance_ohms=Decimal(100),
        initial_voltages_volts=(Decimal(0),) * 4,
        action_segments=(
            ResistorCapacitorLadderActionSegment(
                segment_id="segment-step",
                start_seconds=Decimal(0),
                end_seconds=Decimal(1),
                left_voltage_volts=Decimal(1),
                right_voltage_volts=Decimal(0),
            ),
        ),
        output_times_seconds=tuple(Decimal(str(value)) for value in np.linspace(0, 1, 21)),
        component_metrology_frozen_before_response=True,
    )


def _specification_num(inputs: _TruthInput) -> tuple[str, ...]:
    config = _numerical_config(inputs)
    chain = refinement_chain(config, h0_seconds=0.04, convergence_tolerance_volts=0.03)
    defects = [
        float(summary.maximum_voltage_defect_volts)
        for view_id, _, summary in chain
        if view_id in {"h0", "h0-half", "h0-quarter"}
    ]
    coarse = solve_backward_euler(config, maximum_step_seconds=0.5)
    rejected = summarize_against_matrix_exponential(
        summary_id="summary-corrupted",
        config=config,
        view_id="nonconverged",
        trajectory=coarse,
        convergence_tolerance_volts=1e-6,
    )
    codes = []
    if defects[2] < defects[1] < defects[0]:
        codes.append("refinement-order")
    if not rejected.converged:
        codes.append("nonconverged-rejected")
    return tuple(sorted(codes or ("numerical-qualification-failed",)))


def _specification_comp(inputs: _TruthInput) -> tuple[str, ...]:
    capacitances = inputs.vector("capacitance", 16)
    direct = equal_width_receiver_spec(receiver_id="r4-direct", scale_cells=16, bin_count=4)
    first = equal_width_receiver_spec(receiver_id="r8-first", scale_cells=16, bin_count=8)
    second = equal_width_receiver_spec(
        receiver_id="r4-second",
        scale_cells=8,
        bin_count=4,
        source_receiver_id="r8-first",
    )
    exact = composition_defect(direct=direct, first=first, second=second, capacitances=capacitances)
    corrupted = PhysicalScaleMorphismReceiverSpec(
        receiver_id="r4-corrupted",
        scale_cells=8,
        kind=PhysicalScaleMorphismReceiverKind.SPARSE,
        bins=(),
        sparse_node_indices=(0, 2, 4, 6),
        carried_scalar_ids=(),
        includes_terminal_currents=True,
        source_receiver_id="r8-first",
    )
    corrupted_defect = composition_defect(
        direct=direct,
        first=first,
        second=corrupted,
        capacitances=capacitances,
    )
    codes = []
    if exact < 1e-14:
        codes.append("direct-composed-pass")
    if corrupted_defect > 0.05:
        codes.append("corrupted-composition-opposed")
    return tuple(sorted(codes or ("composition-discrimination-failed",)))


def _specification_leak(inputs: _TruthInput) -> tuple[str, ...]:
    members = tuple(
        sorted(
            (
                PhysicalScaleMorphismForecastMember(
                    member_id=f"member-{level.value}",
                    level=level,
                    property_ids=(f"property-{level.value}",),
                    coordinate_ids=(f"coordinate-{level.value}",),
                    prediction_artifact_id=f"artifact-{level.value}",
                    scoring_rule_id=f"rule-{level.value}",
                    tolerance=Decimal(0),
                )
                for level in PhysicalScaleMorphismForecastLevel
            ),
            key=lambda value: value.member_id,
        )
    )
    try:
        PhysicalScaleMorphismForecastTriplet(
            triplet_id="triplet-leaked",
            complete_unit_ids=("board-01",),
            assignment_id="assignment-leaked",
            causal_cutoff_id="cutoff-leaked",
            receiver_window_id="window-leaked",
            horizon_id="horizon-leaked",
            members=members,
            multiplicity_family_id="family-leaked",
            frozen_before_evaluation=True,
            protected_evaluation_outcome_count=inputs.integer("protected-evaluation-outcome-count"),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
    except ValueError:
        return ("outcome-leak-rejected",)
    return ("outcome-leak-accepted",)


def _specification_unit(inputs: _TruthInput) -> tuple[str, ...]:
    failures = 0
    try:
        PhysicalScaleMorphismNormalizationSpec(
            normalization_id="normalization-wrong-time",
            scale_cells=16,
            component_basis_sha256="0" * 64,
            resistance_summary_rule_id="dossier-resistance-mean",
            capacitance_summary_rule_id="dossier-capacitance-mean",
            resistance_bar_ohms=Decimal(1),
            capacitance_bar_farads=Decimal(1),
            voltage_reference_volts=Decimal(1),
            realized_component_weights=True,
            diffusive_time_exponent=inputs.integer("invalid-time-exponent"),
        )
    except ValueError:
        failures += 1
    receiver = equal_width_receiver_spec(receiver_id="r8-unit", scale_cells=16, bin_count=8)
    try:
        reduce_receiver(
            observation_id="observation-wrong-shape",
            spec=receiver,
            voltages=np.zeros(inputs.integer("wrong-voltage-vector-length")),
            voltage_uncertainties=np.zeros(inputs.integer("wrong-voltage-vector-length")),
            capacitances=np.ones(16),
            capacitance_uncertainties=np.zeros(16),
            terminal_currents=np.zeros(2),
            terminal_current_uncertainties=np.zeros(2),
            uncertainty_method_id="synthetic-exact",
        )
    except ValueError:
        failures += 1
    try:
        valid_spec = PhysicalScaleMorphismNormalizationSpec(
            normalization_id="normalization-valid",
            scale_cells=16,
            component_basis_sha256="1" * 64,
            resistance_summary_rule_id="dossier-resistance-mean",
            capacitance_summary_rule_id="dossier-capacitance-mean",
            resistance_bar_ohms=Decimal(1),
            capacitance_bar_farads=Decimal(1),
            voltage_reference_volts=Decimal(1),
            realized_component_weights=True,
            diffusive_time_exponent=2,
        )
        compute_dimensionless_record(
            record_id="record-nonfinite",
            spec=valid_spec,
            uncertainty=_zero_normalization_uncertainty(valid_spec),
            voltages=np.full(
                16,
                (np.nan if inputs.category("invalid-voltage-value") == "NONFINITE_NAN" else 0.0),
            ),
            capacitances_farads=np.ones(16),
            terminal_current_amperes=0,
            native_time_seconds=1,
            slowest_decay_rate_per_second=1,
        )
    except ValueError:
        failures += 1
    return ("unit-clock-grouping-rejected",) if failures == 3 else ("unit-rejection-failed",)


def _specification_power(inputs: _TruthInput) -> tuple[str, ...]:
    adverse_count = inputs.integer("adverse-count")
    adverse_board_count = inputs.integer("adverse-board-count")
    adverse = qualify_method_power(
        qualification_id="power-adverse",
        requirements=(
            PhysicalScaleMorphismPowerRequirement(
                requirement_id="false-safe-rate",
                family_id="decision",
                method=PhysicalScaleMorphismPowerMethod.ZERO_ADVERSE_RATE,
                expected_location=Decimal(adverse_count) / Decimal(adverse_board_count),
                decision_boundary=inputs.number("adverse-rate-boundary"),
                development_standard_deviation=None,
                observed_adverse_count=adverse_count,
                development_board_count=adverse_board_count,
            ),
        ),
    )
    precision = qualify_method_power(
        qualification_id="power-precision",
        requirements=(
            PhysicalScaleMorphismPowerRequirement(
                requirement_id="metric-equivalence",
                family_id="metric",
                method=PhysicalScaleMorphismPowerMethod.CONTINUOUS_EQUIVALENCE,
                expected_location=inputs.number("precision-location"),
                decision_boundary=inputs.number("precision-boundary"),
                development_standard_deviation=inputs.number("precision-standard-deviation"),
                observed_adverse_count=None,
                development_board_count=8,
            ),
        ),
    )
    correct = (
        adverse.terminal is PhysicalScaleMorphismPowerTerminal.NO_FINITE_COUNT_UNDER_OBSERVED_RATE
        and precision.terminal is PhysicalScaleMorphismPowerTerminal.PRECISION_LIMITED_AT_RESOURCE_CEILING
    )
    return ("precision-vs-structural-distinguished",) if correct else ("power-classes-collapsed",)


def _specification_het(inputs: _TruthInput) -> tuple[str, ...]:
    board_results = (
        PhysicalScaleMorphismBoardLocalResult(
            result_id="result-board-01",
            board_id="board-01",
            batch_id=inputs.category("board-01-batch"),
            categorical_terminal=inputs.category("board-01-terminal"),
            admitted_fraction=inputs.number("board-01-admitted-fraction"),
            boundary_signature_ids=(inputs.category("board-01-boundary"),),
            active_gate_ids=(inputs.category("board-01-gate"),),
        ),
        PhysicalScaleMorphismBoardLocalResult(
            result_id="result-board-02",
            board_id="board-02",
            batch_id=inputs.category("board-02-batch"),
            categorical_terminal=inputs.category("board-02-terminal"),
            admitted_fraction=inputs.number("board-02-admitted-fraction"),
            boundary_signature_ids=(inputs.category("board-02-boundary"),),
            active_gate_ids=(inputs.category("board-02-gate"),),
        ),
    )
    panel = evaluate_heterogeneity(
        panel_id="heterogeneity-truth",
        board_results=board_results,
        material_admitted_fraction_range=Decimal("0.1"),
    )
    return ("aggregate-local-mixed",) if panel.mixed else ("heterogeneity-missed",)


_RUNNERS: dict[str, Callable[[_TruthInput], tuple[str, ...]]] = {
    "boundary-topology": _specification_boundary,
    "composition-discrimination": _specification_comp,
    "energy-mean-collision": _specification_energy,
    "exact-receiver-composition": _specification_exact,
    "future-closure": _specification_future,
    "guard-false-safe": _specification_guard,
    "heterogeneous-local-support": _specification_het,
    "hidden-history": _specification_hidden,
    "hold-state-grammar": _specification_hold,
    "leakage-firewall": _specification_leak,
    "numerical-qualification": _specification_num,
    "power-and-precision": _specification_power,
    "saturation-coordinate": _specification_saturation,
    "scale-morphism": _specification_scale,
    "unit-clock-grouping": _specification_unit,
}


def run_truth_blind_case(case: PhysicalScaleMorphismTruthCase) -> PhysicalScaleMorphismTruthBlindObservation:
    try:
        inputs = _TruthInput(case)
        codes = _RUNNERS[case.fixture_id](inputs)
        inputs.require_complete_consumption()
        exception_codes: tuple[str, ...] = ()
    except Exception as error:  # evaluator must receive an explicit failed method case
        codes = ("method-exception-observed",)
        exception_codes = (f"method-exception-{type(error).__name__.lower()}",)
    return PhysicalScaleMorphismTruthBlindObservation(
        observation_id=f"observation.{case.case_id}",
        case_id=case.case_id,
        truth_input_sha256=case.input_data_sha256,
        observed_code_ids=tuple(sorted(codes)),
        method_exception_code_ids=exception_codes,
        privileged_label_access_count=0,
    )


def run_truth_blind_suite(
    cases: tuple[PhysicalScaleMorphismTruthCase, ...],
) -> tuple[PhysicalScaleMorphismTruthBlindObservation, ...]:
    return tuple(
        sorted(
            (run_truth_blind_case(value) for value in cases), key=lambda value: value.observation_id
        )
    )


__all__ = ["run_truth_blind_case", "run_truth_blind_suite"]

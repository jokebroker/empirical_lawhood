"""Substrate-general prepared-system contract for direct native TORAX episodes."""

from __future__ import annotations

from decimal import Decimal

from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.systems import (
    BalanceRole,
    IndependentUnitSpec,
    PortDirection,
    PortSpec,
    RelationalIdentity,
    SystemBoundaryKind,
    SystemSpec,
)
from empirical_lawhood.kernel.time import (
    AvailabilitySpec,
    CausalPhase,
    ClockLabelSemantics,
    ClockSpec,
    HoldSemantics,
    HorizonSpec,
    SamplingSemantics,
)
from empirical_lawhood.kernel.worlds import (
    ComputabilityEnvelope,
    EvidenceUnitScope,
    NumericalCoordinateKind,
    NumericalCoordinateSpec,
    NumericalViewSpec,
    RandomnessSemantics,
    WorldKind,
    WorldSpec,
)

from .contracts import NativeToraxView


def _quantity(
    quantity_id: str,
    *,
    kind: QuantityKind,
    unit: str,
    frame: str,
    phase: CausalPhase,
    access: OutcomeAccess,
    direction: ResponseDirection = ResponseDirection.NOT_APPLICABLE,
) -> QuantitySpec:
    return QuantitySpec(
        quantity_id=quantity_id,
        label=quantity_id.replace(".", " "),
        kind=kind,
        dimension=quantity_id.split(".", 1)[0],
        native_unit=unit,
        coordinate_frame=frame,
        clock_id="torax-episode-clock",
        availability=AvailabilitySpec(
            clock_id="torax-episode-clock",
            phase=phase,
            outcome_access=access,
        ),
        response_direction=direction,
    )


def build_native_torax_system(
    views: tuple[NativeToraxView, ...],
    *,
    authority_policy: AuthorityPolicy,
) -> SystemSpec:
    """Build direct TORAX while leaving experiment authority to the consumer."""

    clock = ClockSpec(
        clock_id="torax-episode-clock",
        label="direct TORAX episode time",
        time_unit="s",
        coordinate_frame="torax-episode-relative-time",
        sampling=SamplingSemantics.REGULAR,
        hold=HoldSemantics.ZERO_ORDER,
        label_semantics=ClockLabelSemantics.INSTANT,
        nominal_period=Decimal("0.001"),
        alignment_tolerance=Decimal(0),
    )
    denominator_rows = (
        ("denominator.chi-e", "m2/s"),
        ("denominator.chi-i", "m2/s"),
        ("denominator.core-density", "m-3"),
        ("denominator.core-te", "eV"),
        ("denominator.plasma-current", "A"),
        ("denominator.source-location", "1"),
        ("denominator.source-width", "1"),
        ("denominator.target-delta-core-te", "eV"),
    )
    history_rows = (
        ("history.core-edge-contrast", "eV"),
        ("history.core-te", "eV"),
        ("history.radial-mean-te", "eV"),
    )
    quantities = [
        *(
            _quantity(
                quantity_id,
                kind=QuantityKind.DENOMINATOR,
                unit=unit,
                frame="torax-preparation",
                phase=CausalPhase.PRE_ACTION,
                access=OutcomeAccess.OUTCOME_BLIND,
            )
            for quantity_id, unit in denominator_rows
        ),
        *(
            _quantity(
                quantity_id,
                kind=QuantityKind.HISTORY,
                unit=unit,
                frame="torax-state",
                phase=CausalPhase.PRE_ACTION,
                access=OutcomeAccess.OUTCOME_BLIND,
            )
            for quantity_id, unit in history_rows
        ),
        _quantity(
            "action.proxy-heating-power",
            kind=QuantityKind.ACTION,
            unit="W",
            frame="torax-generic-heat-source",
            phase=CausalPhase.ACTION_APPLIED,
            access=OutcomeAccess.OUTCOME_BLIND,
        ),
        _quantity(
            "effort.proxy-energy",
            kind=QuantityKind.EFFORT,
            unit="J",
            frame="torax-generic-heat-source",
            phase=CausalPhase.RECEIVER,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            direction=ResponseDirection.LOWER_IS_BETTER,
        ),
        _quantity(
            "receiver.delta-core-te",
            kind=QuantityKind.RECEIVER,
            unit="eV",
            frame="torax-temperature",
            phase=CausalPhase.RECEIVER,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            direction=ResponseDirection.TARGET_BAND,
        ),
        _quantity(
            "receiver.endpoint-core-edge-contrast",
            kind=QuantityKind.RECEIVER,
            unit="eV",
            frame="torax-temperature",
            phase=CausalPhase.RECEIVER,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            direction=ResponseDirection.TARGET_BAND,
        ),
        _quantity(
            "sink.core-temperature-safety-margin",
            kind=QuantityKind.SINK,
            unit="eV",
            frame="torax-temperature",
            phase=CausalPhase.RECEIVER,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            direction=ResponseDirection.HIGHER_IS_BETTER,
        ),
    ]
    quantities_tuple = tuple(sorted(quantities, key=lambda value: value.quantity_id))
    relation = RelationalIdentity(
        relation_id="relation.torax-native.denominator-local",
        denominator_quantity_ids=tuple(value[0] for value in denominator_rows),
        history_quantity_ids=tuple(value[0] for value in history_rows),
        memoryless=False,
        action_quantity_ids=("action.proxy-heating-power",),
        receiver_quantity_ids=(
            "receiver.delta-core-te",
            "receiver.endpoint-core-edge-contrast",
        ),
        horizon=HorizonSpec(
            horizon_id="horizon.torax-40ms",
            clock_id=clock.clock_id,
            duration=Decimal("0.04"),
            time_unit="s",
        ),
    )
    world = WorldSpec(
        world_id="world.torax-native",
        label="direct TORAX 1.4.2 prepared numerical response medium",
        kind=WorldKind.NUMERICAL_SIMULATOR,
        represented_physics=(
            "coupled-ion-electron-heat-evolution",
            "prescribed-constant-transport",
            "radial-heat-diffusion",
        ),
        unrepresented_physics=(
            "facility-actuator-dynamics",
            "kinetic-turbulence",
            "real-mast-geometry-and-boundary",
        ),
        privileged_truth_quantity_ids=(),
        maximum_evidence=EvidenceCeiling.CONTROLLER_USE,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.EVALUATION_REVEALED,
            }
        ),
    )
    budget = authority_policy.budget_ceiling
    envelope = ComputabilityEnvelope(
        envelope_id="compute.torax-native",
        represented_effect_ids=world.represented_physics,
        unresolved_effect_ids=world.unrepresented_physics,
        required_structure_ids=(
            "finite-action-response",
            "nested-numerical-view-convergence",
        ),
        computable_structure_ids=(
            "finite-action-response",
            "nested-numerical-view-convergence",
        ),
        max_cpu_cores=budget.cpu_cores,
        max_memory_bytes=budget.memory_bytes,
        max_gpu_devices=budget.gpu_devices,
        max_wall_time_seconds=budget.wall_time_seconds,
        max_output_bytes=budget.output_bytes,
        worst_case_latency_seconds=Decimal(budget.wall_time_seconds),
        deadline_seconds=None,
    )
    numerical_views = tuple(
        NumericalViewSpec(
            view_id=view.view_id,
            world_id=world.world_id,
            physical_preparation_id="unit.torax-native-preparation",
            equations_id="torax-1.4.2-ion-electron-heat",
            closure_ids=(view.closure_id,),
            boundary_condition_ids=("boundary.fixed-profile-right-edge",),
            coordinates=(
                NumericalCoordinateSpec(
                    coordinate_id="coordinate.radial-cells",
                    kind=NumericalCoordinateKind.SPATIAL_GRID,
                    value=Decimal(view.radial_cells),
                    unit="cells",
                    refinement_level=2 if view.radial_cells == 24 else 1,
                ),
                NumericalCoordinateSpec(
                    coordinate_id="coordinate.timestep",
                    kind=NumericalCoordinateKind.TIMESTEP,
                    value=view.timestep_s,
                    unit="s",
                    refinement_level=2 if view.timestep_s == Decimal("0.001") else 1,
                ),
            ),
            solver_id=view.solver_id,
            solver_version="1.4.2",
            precision=view.precision,
            device_class="cpu",
            runtime_id="torax-1.4.2-jax-0.10.2-cpu-x64",
            randomness=RandomnessSemantics.DETERMINISTIC,
            observation_operator_id="observer.torax-native-temperature",
            computability_envelope_id=envelope.envelope_id,
        )
        for view in views
    )
    ports = tuple(
        sorted(
            (
                PortSpec(
                    port_id="port.torax-proxy-heating-power.input",
                    quantity_id="action.proxy-heating-power",
                    clock_id=clock.clock_id,
                    direction=PortDirection.INPUT,
                    balance_role=BalanceRole.COMMAND,
                    authority_action=AuthorityAction.SIMULATION_EXECUTION,
                ),
                *(
                    PortSpec(
                        port_id=f"port.{value.quantity_id}.output",
                        quantity_id=value.quantity_id,
                        clock_id=clock.clock_id,
                        direction=PortDirection.OUTPUT,
                        balance_role=(
                            BalanceRole.ENERGY
                            if value.kind in {QuantityKind.SINK, QuantityKind.EFFORT}
                            else BalanceRole.OBSERVATION
                        ),
                    )
                    for value in quantities_tuple
                    if value.kind in {QuantityKind.RECEIVER, QuantityKind.SINK, QuantityKind.EFFORT}
                ),
            ),
            key=lambda value: value.port_id,
        )
    )
    return SystemSpec(
        system_id="system.torax-native",
        label="direct TORAX prepared response medium",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities_tuple,
        independent_unit=IndependentUnitSpec(
            unit_id="unit.torax-native-preparation",
            label="one frozen TORAX preparation task",
            grouping_key="group.torax-native-task-id",
            scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        ),
        authority_policy=authority_policy,
        ports=ports,
        computability_envelopes=(envelope,),
        numerical_views=tuple(sorted(numerical_views, key=lambda value: value.view_id)),
    )


__all__ = ["build_native_torax_system"]

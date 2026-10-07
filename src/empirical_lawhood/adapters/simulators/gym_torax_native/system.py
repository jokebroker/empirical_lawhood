"""Current prepared-system contract for Gym--TORAX Gym-TORAX episodes."""

from __future__ import annotations

from decimal import Decimal

from empirical_lawhood.kernel.authority import AuthorityAction, AuthorityPolicy, ResourceBudget
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

from .diagnostic_contracts import GymToraxNumericalMember, gym_torax_numerical_members


GYM_TORAX_SYSTEM_ID = 'system.tokamak-control.prepared-native-tokamak'
GYM_TORAX_WORLD_ID = 'world.tokamak-control.gym-torax-1-1-1'
GYM_TORAX_INDEPENDENT_UNIT_ID = 'unit.tokamak-control.gym-torax-preparation'
GYM_TORAX_REQUEST_CLOCK_ID = 'clock.tokamak-control.request-index'
GYM_TORAX_STATE_CLOCK_ID = 'clock.tokamak-control.state-index'
GYM_TORAX_ACTION_QUANTITY_ID = 'quantity.tokamak-control.ip-target'
GYM_TORAX_RECEIVER_QUANTITY_ID = 'receiver.tokamak-control.q-fusion-phase-mean-0105-0110'


def _quantity(
    quantity_id: str,
    *,
    kind: QuantityKind,
    unit: str,
    frame: str,
    clock_id: str,
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
        clock_id=clock_id,
        availability=AvailabilitySpec(
            clock_id=clock_id,
            phase=phase,
            outcome_access=access,
        ),
        response_direction=direction,
    )


def _numerical_view(member: GymToraxNumericalMember) -> NumericalViewSpec:
    slug = member.member_id.removeprefix('member.tokamak-control.')
    refinement = 1 if slug == "primary" else 2
    return NumericalViewSpec(
        view_id=f'view.tokamak-control.{slug}',
        world_id=GYM_TORAX_WORLD_ID,
        physical_preparation_id=GYM_TORAX_INDEPENDENT_UNIT_ID,
        equations_id="equations.torax-1-4-2-iter-hybrid",
        closure_ids=("closure.gymtorax-1-1-1-default-physics",),
        boundary_condition_ids=("boundary.gymtorax-1-1-1-default",),
        coordinates=tuple(
            sorted(
                (
                    NumericalCoordinateSpec(
                        coordinate_id="coordinate.corrector-steps",
                        kind=NumericalCoordinateKind.SOLVER_REFINEMENT,
                        value=Decimal(member.corrector_steps),
                        unit="steps",
                        refinement_level=refinement,
                    ),
                    NumericalCoordinateSpec(
                        coordinate_id="coordinate.internal-timestep",
                        kind=NumericalCoordinateKind.TIMESTEP,
                        value=member.internal_timestep_s,
                        unit="s",
                        refinement_level=refinement,
                    ),
                    NumericalCoordinateSpec(
                        coordinate_id="coordinate.radial-cells",
                        kind=NumericalCoordinateKind.SPATIAL_GRID,
                        value=Decimal(member.radial_cells),
                        unit="cells",
                        refinement_level=refinement,
                    ),
                ),
                key=lambda value: value.coordinate_id,
            )
        ),
        solver_id=member.solver_id,
        solver_version="1.4.2",
        precision=member.precision,
        device_class=member.backend,
        runtime_id='runtime.tokamak-control.gymtorax-1-1-1-torax-1-4-2-jax-0-10-2-cpu-x64',
        randomness=RandomnessSemantics.DETERMINISTIC,
        observation_operator_id='observer.tokamak-control.gym-torax-native-source',
        computability_envelope_id='compute.tokamak-control.gym-torax-native',
    )


def build_gym_torax_gym_torax_system(
    *,
    authority_policy: AuthorityPolicy,
    native_resource_budget: ResourceBudget | None = None,
) -> SystemSpec:
    """Build the exact numerical evidence world without contacting Gym/TORAX.

    ``AuthorityPolicy.budget_ceiling`` may cover the larger post-processing
    stage.  ``native_resource_budget`` keeps the simulator's computability
    envelope at the narrower Gym acquisition ceiling and must remain within
    that policy.  Existing callers that have one common ceiling retain the
    original behavior.
    """

    budget = (
        authority_policy.budget_ceiling
        if native_resource_budget is None
        else native_resource_budget
    )
    if not authority_policy.budget_ceiling.contains(budget):
        raise ValueError("Gym native resource budget exceeds the authority policy")

    clocks = tuple(
        sorted(
            (
                ClockSpec(
                    clock_id=GYM_TORAX_REQUEST_CLOCK_ID,
                    label="Gym TORAX action request index",
                    time_unit="request",
                    coordinate_frame='frame.tokamak-control.request-index',
                    sampling=SamplingSemantics.REGULAR,
                    hold=HoldSemantics.ZERO_ORDER,
                    label_semantics=ClockLabelSemantics.INTERVAL_START,
                    nominal_period=Decimal(1),
                    alignment_tolerance=Decimal(0),
                ),
                ClockSpec(
                    clock_id=GYM_TORAX_STATE_CLOCK_ID,
                    label="Gym TORAX receiver state index",
                    time_unit="state",
                    coordinate_frame='frame.tokamak-control.state-index',
                    sampling=SamplingSemantics.REGULAR,
                    hold=HoldSemantics.NONE,
                    label_semantics=ClockLabelSemantics.INSTANT,
                    nominal_period=Decimal(1),
                    alignment_tolerance=Decimal(0),
                ),
            ),
            key=lambda value: value.clock_id,
        )
    )
    denominator_rows = (
        ('tokamak-control.preparation.bootstrap-multiplier', "1"),
        ('tokamak-control.preparation.initial-density-nbar', "1"),
        ('tokamak-control.preparation.initial-temperature-scale', "1"),
        ('tokamak-control.preparation.inner-transport-scale', "1"),
    )
    history_rows = (
        ('history.tokamak-control.preaction-config', "1"),
        ('history.tokamak-control.preaction-state-surface', "1"),
    )
    receiver_rows = ((GYM_TORAX_RECEIVER_QUANTITY_ID, "1", ResponseDirection.HIGHER_IS_BETTER),)
    quantities = [
        *(
            _quantity(
                quantity_id,
                kind=QuantityKind.DENOMINATOR,
                unit=unit,
                frame='frame.tokamak-control.preparation-chart',
                clock_id=GYM_TORAX_STATE_CLOCK_ID,
                phase=CausalPhase.PREPARATION,
                access=OutcomeAccess.OUTCOME_BLIND,
            )
            for quantity_id, unit in denominator_rows
        ),
        *(
            _quantity(
                quantity_id,
                kind=QuantityKind.HISTORY,
                unit=unit,
                frame='frame.tokamak-control.preaction-through-state-0104',
                clock_id=GYM_TORAX_STATE_CLOCK_ID,
                phase=CausalPhase.PRE_ACTION,
                access=OutcomeAccess.OUTCOME_BLIND,
            )
            for quantity_id, unit in history_rows
        ),
        _quantity(
            GYM_TORAX_ACTION_QUANTITY_ID,
            kind=QuantityKind.ACTION,
            unit="A",
            frame='frame.tokamak-control.absolute-ip-target',
            clock_id=GYM_TORAX_REQUEST_CLOCK_ID,
            phase=CausalPhase.ACTION_REQUESTED,
            access=OutcomeAccess.OUTCOME_BLIND,
        ),
        *(
            _quantity(
                quantity_id,
                kind=QuantityKind.RECEIVER,
                unit=unit,
                frame=(
                    'frame.tokamak-control.q-fusion-phase-mean-states-0105-0110'
                    if quantity_id == GYM_TORAX_RECEIVER_QUANTITY_ID
                    else 'frame.tokamak-control.receiver-state-surface'
                ),
                clock_id=GYM_TORAX_STATE_CLOCK_ID,
                phase=CausalPhase.RECEIVER,
                access=OutcomeAccess.EVALUATOR_REVEAL,
                direction=direction,
            )
            for quantity_id, unit, direction in receiver_rows
        ),
        _quantity(
            'sink.tokamak-control.native-safety-conjunction',
            kind=QuantityKind.SINK,
            unit="1",
            frame='frame.tokamak-control.receiver-state-surface',
            clock_id=GYM_TORAX_STATE_CLOCK_ID,
            phase=CausalPhase.RECEIVER,
            access=OutcomeAccess.EVALUATOR_REVEAL,
            direction=ResponseDirection.HIGHER_IS_BETTER,
        ),
        _quantity(
            'effort.tokamak-control.absolute-ip',
            kind=QuantityKind.EFFORT,
            unit="A.s",
            frame='frame.tokamak-control.absolute-ip-target',
            clock_id=GYM_TORAX_STATE_CLOCK_ID,
            phase=CausalPhase.RECEIVER,
            access=OutcomeAccess.EVALUATOR_REVEAL,
            direction=ResponseDirection.LOWER_IS_BETTER,
        ),
        _quantity(
            'observation.tokamak-control.native-completeness',
            kind=QuantityKind.OBSERVATION,
            unit="1",
            frame='frame.tokamak-control.receiver-state-surface',
            clock_id=GYM_TORAX_STATE_CLOCK_ID,
            phase=CausalPhase.RECEIVER,
            access=OutcomeAccess.EVALUATOR_REVEAL,
        ),
    ]
    quantities_tuple = tuple(sorted(quantities, key=lambda value: value.quantity_id))
    relation = RelationalIdentity(
        relation_id='relation.tokamak-control.joint-face-finite-current-response',
        denominator_quantity_ids=tuple(value[0] for value in denominator_rows),
        history_quantity_ids=tuple(value[0] for value in history_rows),
        memoryless=False,
        action_quantity_ids=(GYM_TORAX_ACTION_QUANTITY_ID,),
        receiver_quantity_ids=tuple(sorted(value[0] for value in receiver_rows)),
        horizon=HorizonSpec(
            horizon_id='horizon.tokamak-control.states-0001-0120',
            clock_id=GYM_TORAX_STATE_CLOCK_ID,
            duration=Decimal(120),
            time_unit="state",
        ),
    )
    world = WorldSpec(
        world_id=GYM_TORAX_WORLD_ID,
        label="Gym TORAX 1.1.1 over TORAX 1.4.2 prepared numerical medium",
        kind=WorldKind.NUMERICAL_SIMULATOR,
        represented_physics=tuple(
            sorted(
                (
                    "gymtorax-iter-hybrid-action-delivery",
                    "torax-coupled-core-transport",
                    "torax-neoclassical-bootstrap-current",
                    "torax-native-source-and-sink-diagnostics",
                )
            )
        ),
        unrepresented_physics=tuple(
            sorted(
                (
                    "facility-actuator-dynamics",
                    "physical-device-uncertainty",
                    "unmodelled-plasma-closure-error",
                )
            )
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
    envelope = ComputabilityEnvelope(
        envelope_id='compute.tokamak-control.gym-torax-native',
        represented_effect_ids=world.represented_physics,
        unresolved_effect_ids=world.unrepresented_physics,
        required_structure_ids=("finite-action-response",),
        computable_structure_ids=("finite-action-response",),
        max_cpu_cores=budget.cpu_cores,
        max_memory_bytes=budget.memory_bytes,
        max_gpu_devices=0,
        max_wall_time_seconds=budget.wall_time_seconds,
        max_output_bytes=budget.output_bytes,
        worst_case_latency_seconds=Decimal(budget.wall_time_seconds),
        deadline_seconds=None,
    )
    views = tuple(
        sorted(
            (_numerical_view(value) for value in gym_torax_numerical_members()),
            key=lambda value: value.view_id,
        )
    )
    ports = tuple(
        sorted(
            (
                PortSpec(
                    port_id="port.gym-torax.ip-target",
                    quantity_id=GYM_TORAX_ACTION_QUANTITY_ID,
                    clock_id=GYM_TORAX_REQUEST_CLOCK_ID,
                    direction=PortDirection.INPUT,
                    balance_role=BalanceRole.COMMAND,
                    authority_action=AuthorityAction.SIMULATION_EXECUTION,
                ),
                *(
                    PortSpec(
                        port_id=f"port.{value.quantity_id}.output",
                        quantity_id=value.quantity_id,
                        clock_id=value.clock_id,
                        direction=PortDirection.OUTPUT,
                        balance_role=(
                            BalanceRole.ENERGY
                            if value.kind in {QuantityKind.SINK, QuantityKind.EFFORT}
                            else BalanceRole.OBSERVATION
                        ),
                    )
                    for value in quantities_tuple
                    if value.kind
                    in {
                        QuantityKind.EFFORT,
                        QuantityKind.OBSERVATION,
                        QuantityKind.RECEIVER,
                        QuantityKind.SINK,
                    }
                ),
            ),
            key=lambda value: value.port_id,
        )
    )
    return SystemSpec(
        system_id=GYM_TORAX_SYSTEM_ID,
        label='Tokamak Gym TORAX prepared open driven response system',
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=clocks,
        quantities=quantities_tuple,
        independent_unit=IndependentUnitSpec(
            unit_id=GYM_TORAX_INDEPENDENT_UNIT_ID,
            label="one fresh Gym TORAX preparation and acquisition task",
            grouping_key='group.tokamak-control.physical-preparation-id',
            scope=EvidenceUnitScope.PHYSICAL_INDEPENDENT_UNIT,
        ),
        authority_policy=authority_policy,
        ports=ports,
        computability_envelopes=(envelope,),
        numerical_views=views,
    )


__all__ = [
    'GYM_TORAX_ACTION_QUANTITY_ID',
    'GYM_TORAX_INDEPENDENT_UNIT_ID',
    'GYM_TORAX_RECEIVER_QUANTITY_ID',
    'GYM_TORAX_REQUEST_CLOCK_ID',
    'GYM_TORAX_STATE_CLOCK_ID',
    'GYM_TORAX_SYSTEM_ID',
    'GYM_TORAX_WORLD_ID',
    'build_gym_torax_gym_torax_system',
]

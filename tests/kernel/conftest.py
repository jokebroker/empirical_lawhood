# SPDX-License-Identifier: MPL-2.0
"""Private synthetic fixtures for direct kernel invariants."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
import pytest
from empirical_lawhood.kernel.authority import AuthorityAction, AuthorityPolicy, ResourceBudget, SourceAccessClass
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.systems import BalanceRole, IndependentUnitSpec, PortDirection, PortSpec, RelationalIdentity, SystemBoundaryKind, SystemSpec
from empirical_lawhood.kernel.time import AvailabilitySpec, CausalPhase, ClockLabelSemantics, ClockSpec, HoldSemantics, HorizonSpec, SamplingSemantics
from empirical_lawhood.kernel.worlds import ComputabilityEnvelope, NumericalCoordinateKind, NumericalCoordinateSpec, NumericalViewSpec, RandomnessSemantics, WorldKind, WorldSpec


@pytest.fixture
def authority_policy() -> AuthorityPolicy:
    return AuthorityPolicy(
        policy_id="test-nonactuating-policy",
        delegator_id="user-handoff",
        delegate_id="approval-gate",
        scope_ids=("reference-worlds", "simulation"),
        allowed_world_kinds=frozenset(
            {WorldKind.ANALYTIC_REFERENCE, WorldKind.NUMERICAL_SIMULATOR}
        ),
        allowed_actions=frozenset(
            {
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
                AuthorityAction.REFERENCE_WORLD_EXECUTION,
                AuthorityAction.SIMULATION_EXECUTION,
            }
        ),
        allowed_source_classes=frozenset(
            {SourceAccessClass.NONE, SourceAccessClass.OFFICIAL_OPEN_PUBLIC}
        ),
        required_gate_ids=("budget", "clean-commit", "outcome-separation"),
        nondelegable_actions=frozenset(
            {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.LIVE_ACTUATION,
            }
        ),
        budget_ceiling=ResourceBudget(
            cpu_cores=8,
            memory_bytes=16_000_000_000,
            gpu_devices=1,
            wall_time_seconds=3_600,
            source_scan_bytes=10_000_000,
            output_bytes=1_000_000,
        ),
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        expires_at_utc="2030-01-01T00:00:00Z",
    )


@pytest.fixture
def numerical_system(authority_policy: AuthorityPolicy) -> SystemSpec:
    clock = ClockSpec(
        clock_id="experiment-clock",
        label="Experiment time",
        time_unit="s",
        coordinate_frame="experiment-start",
        sampling=SamplingSemantics.REGULAR,
        hold=HoldSemantics.ZERO_ORDER,
        label_semantics=ClockLabelSemantics.INTERVAL_END,
        nominal_period=Decimal("0.1"),
        alignment_tolerance=Decimal("0.001"),
    )

    def availability(phase: CausalPhase, access: OutcomeAccess) -> AvailabilitySpec:
        return AvailabilitySpec(
            clock_id=clock.clock_id,
            phase=phase,
            outcome_access=access,
        )

    quantities = (
        QuantitySpec(
            quantity_id="action-power",
            label="Applied heater power",
            kind=QuantityKind.ACTION,
            dimension="power",
            native_unit="W",
            coordinate_frame="heater-terminal",
            clock_id=clock.clock_id,
            availability=availability(CausalPhase.ACTION_APPLIED, OutcomeAccess.OUTCOME_BLIND),
        ),
        QuantitySpec(
            quantity_id="ambient-boundary",
            label="Ambient temperature",
            kind=QuantityKind.BOUNDARY,
            dimension="temperature",
            native_unit="K",
            coordinate_frame="chamber",
            clock_id=clock.clock_id,
            availability=availability(CausalPhase.PREPARATION, OutcomeAccess.OUTCOME_BLIND),
        ),
        QuantitySpec(
            quantity_id="receiver-temperature",
            label="Receiver temperature",
            kind=QuantityKind.RECEIVER,
            dimension="temperature",
            native_unit="K",
            coordinate_frame="sample-centre",
            clock_id=clock.clock_id,
            availability=availability(CausalPhase.RECEIVER, OutcomeAccess.EVALUATION_SEALED),
            response_direction=ResponseDirection.TARGET_BAND,
        ),
        QuantitySpec(
            quantity_id="sink-heat-loss",
            label="Heat lost to bath",
            kind=QuantityKind.SINK,
            dimension="power",
            native_unit="W",
            coordinate_frame="chamber-wall",
            clock_id=clock.clock_id,
            availability=availability(CausalPhase.RECEIVER, OutcomeAccess.EVALUATION_SEALED),
            response_direction=ResponseDirection.LOWER_IS_BETTER,
        ),
        QuantitySpec(
            quantity_id="substrate",
            label="Prepared response medium",
            kind=QuantityKind.DENOMINATOR,
            dimension="material-identity",
            native_unit="1",
            coordinate_frame="prepared-cell",
            clock_id=clock.clock_id,
            availability=availability(CausalPhase.PREPARATION, OutcomeAccess.OUTCOME_BLIND),
        ),
    )
    relation = RelationalIdentity(
        relation_id="thermal-response",
        denominator_quantity_ids=("ambient-boundary", "substrate"),
        history_quantity_ids=(),
        memoryless=True,
        action_quantity_ids=("action-power",),
        receiver_quantity_ids=("receiver-temperature",),
        horizon=HorizonSpec(
            horizon_id="ten-second-horizon",
            clock_id=clock.clock_id,
            duration=Decimal("10"),
            time_unit="s",
        ),
    )
    world = WorldSpec(
        world_id="thermal-simulator",
        label="Truth-known thermal simulator",
        kind=WorldKind.NUMERICAL_SIMULATOR,
        represented_physics=("linear-conduction", "thermal-capacitance"),
        unrepresented_physics=("physical-plant-discrepancy",),
        privileged_truth_quantity_ids=(),
        maximum_evidence=EvidenceCeiling.CONTROLLER_USE,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.EVALUATION_REVEALED,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.OUTCOME_BLIND,
            }
        ),
    )
    envelope = ComputabilityEnvelope(
        envelope_id="local-classical-envelope",
        represented_effect_ids=("linear-conduction", "thermal-capacitance"),
        unresolved_effect_ids=(),
        required_structure_ids=("finite-horizon-response", "response-rank"),
        computable_structure_ids=("finite-horizon-response", "response-rank"),
        max_cpu_cores=4,
        max_memory_bytes=4_000_000_000,
        max_gpu_devices=0,
        max_wall_time_seconds=600,
        max_output_bytes=100_000,
        worst_case_latency_seconds=Decimal("0.01"),
        deadline_seconds=Decimal("0.1"),
    )
    view = NumericalViewSpec(
        view_id="thermal-view-fine",
        world_id=world.world_id,
        physical_preparation_id="prepared-cell",
        equations_id="linear-thermal-ode",
        closure_ids=("constant-capacitance",),
        boundary_condition_ids=("fixed-ambient",),
        coordinates=(
            NumericalCoordinateSpec(
                coordinate_id="time-step",
                kind=NumericalCoordinateKind.TIMESTEP,
                value=Decimal("0.01"),
                unit="s",
                refinement_level=2,
            ),
        ),
        solver_id="rk4",
        solver_version="1.0.0",
        precision="float64",
        device_class="cpu",
        runtime_id="python-reference",
        randomness=RandomnessSemantics.DETERMINISTIC,
        observation_operator_id="identity-temperature",
        computability_envelope_id=envelope.envelope_id,
    )
    coarse_view = replace(
        view,
        view_id="thermal-view-coarse",
        coordinates=(
            replace(
                view.coordinates[0],
                value=Decimal("0.02"),
                refinement_level=1,
            ),
        ),
    )
    return SystemSpec(
        system_id="thermal-open-system",
        label="Open driven thermal response system",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=quantities,
        independent_unit=IndependentUnitSpec(
            unit_id="prepared-cell",
            label="Independently prepared thermal cell",
            grouping_key="preparation-id",
        ),
        authority_policy=authority_policy,
        ports=(
            PortSpec(
                port_id="action-in",
                quantity_id="action-power",
                clock_id=clock.clock_id,
                direction=PortDirection.INPUT,
                balance_role=BalanceRole.COMMAND,
                authority_action=AuthorityAction.SIMULATION_EXECUTION,
            ),
            PortSpec(
                port_id="receiver-out",
                quantity_id="receiver-temperature",
                clock_id=clock.clock_id,
                direction=PortDirection.OUTPUT,
                balance_role=BalanceRole.OBSERVATION,
            ),
            PortSpec(
                port_id="sink-out",
                quantity_id="sink-heat-loss",
                clock_id=clock.clock_id,
                direction=PortDirection.OUTPUT,
                balance_role=BalanceRole.ENERGY,
            ),
        ),
        computability_envelopes=(envelope,),
        numerical_views=(coarse_view, view),
    )

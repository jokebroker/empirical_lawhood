"""Small SystemSpec builders for deterministic analytic/numerical worlds."""

from __future__ import annotations

from decimal import Decimal

from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.systems import (
    BalanceRole,
    BalanceSemantics,
    ComponentSpec,
    IndependentUnitSpec,
    InterfaceSpec,
    PortDirection,
    PortRef,
    PortSpec,
    RelationalIdentity,
    SystemBoundaryKind,
    SystemSpec,
)
from empirical_lawhood.kernel.time import (
    AvailabilitySpec,
    CausalPhase,
    ClockLabelSemantics,
    ClockRelationKind,
    ClockRelationSpec,
    ClockSpec,
    HoldSemantics,
    HorizonSpec,
    SamplingSemantics,
)
from empirical_lawhood.kernel.worlds import (
    ComputabilityEnvelope,
    NumericalCoordinateKind,
    NumericalCoordinateSpec,
    NumericalViewSpec,
    RandomnessSemantics,
    WorldKind,
    WorldSpec,
)


def _authority() -> AuthorityPolicy:
    return AuthorityPolicy(
        policy_id="reference-world-authority",
        delegator_id="platform-remediation",
        delegate_id="reference-world-gate",
        scope_ids=("reference-world",),
        allowed_world_kinds=frozenset(
            {WorldKind.ANALYTIC_REFERENCE, WorldKind.NUMERICAL_SIMULATOR}
        ),
        allowed_actions=frozenset(
            {
                AuthorityAction.REFERENCE_WORLD_EXECUTION,
                AuthorityAction.SIMULATION_EXECUTION,
            }
        ),
        allowed_source_classes=frozenset({SourceAccessClass.NONE}),
        required_gate_ids=("clean-implementation", "resource-envelope"),
        nondelegable_actions=frozenset(
            {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.LIVE_ACTUATION,
                AuthorityAction.SAFETY_SIGNIFICANT_OPERATION,
            }
        ),
        budget_ceiling=ResourceBudget(
            cpu_cores=2,
            memory_bytes=1_000_000_000,
            gpu_devices=0,
            wall_time_seconds=60,
            source_scan_bytes=10_000,
            output_bytes=1_000_000,
        ),
        maximum_outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
    )


def _clock(clock_id: str) -> ClockSpec:
    return ClockSpec(
        clock_id=clock_id,
        label=clock_id.replace("-", " ").title(),
        time_unit="s",
        coordinate_frame="reference-time",
        sampling=SamplingSemantics.REGULAR,
        hold=HoldSemantics.ZERO_ORDER,
        label_semantics=ClockLabelSemantics.INSTANT,
        nominal_period=Decimal("1"),
        alignment_tolerance=Decimal("0"),
    )


def _quantity(
    quantity_id: str,
    kind: QuantityKind,
    clock_id: str,
    phase: CausalPhase,
    access: OutcomeAccess,
    *,
    dimension: str = "reference",
    native_unit: str = "1",
    coordinate_frame: str = "reference-frame",
    direction: ResponseDirection = ResponseDirection.NOT_APPLICABLE,
) -> QuantitySpec:
    return QuantitySpec(
        quantity_id=quantity_id,
        label=quantity_id.replace("-", " ").title(),
        kind=kind,
        dimension=dimension,
        native_unit=native_unit,
        coordinate_frame=coordinate_frame,
        clock_id=clock_id,
        availability=AvailabilitySpec(
            clock_id=clock_id,
            phase=phase,
            outcome_access=access,
        ),
        response_direction=direction,
    )


def _numerical_contract(
    reference_id: str,
    *,
    latency_seconds: Decimal,
    deadline_seconds: Decimal,
) -> tuple[ComputabilityEnvelope, tuple[NumericalViewSpec, ...]]:
    envelope = ComputabilityEnvelope(
        envelope_id="reference-compute-envelope",
        represented_effect_ids=("declared-reference-dynamics",),
        unresolved_effect_ids=(),
        required_structure_ids=(
            "admission-boundary",
            "finite-horizon-response",
            "response-rank",
        ),
        computable_structure_ids=(
            "admission-boundary",
            "finite-horizon-response",
            "response-rank",
        ),
        max_cpu_cores=2,
        max_memory_bytes=1_000_000_000,
        max_gpu_devices=0,
        max_wall_time_seconds=60,
        max_output_bytes=1_000_000,
        worst_case_latency_seconds=latency_seconds,
        deadline_seconds=deadline_seconds,
    )
    views = tuple(
        NumericalViewSpec(
            view_id=view_id,
            world_id=reference_id,
            physical_preparation_id="reference-preparation",
            equations_id="declared-reference-equations",
            closure_ids=("declared-reference-closure",),
            boundary_condition_ids=("declared-reference-boundary",),
            coordinates=(
                NumericalCoordinateSpec(
                    coordinate_id="time-step",
                    kind=NumericalCoordinateKind.TIMESTEP,
                    value=value,
                    unit="s",
                    refinement_level=level,
                ),
            ),
            solver_id="analytic-reference-evaluator",
            solver_version="1.0.0",
            precision="decimal-exact",
            device_class="cpu",
            runtime_id="python-standard-library",
            randomness=RandomnessSemantics.DETERMINISTIC,
            observation_operator_id="declared-reference-observer",
            computability_envelope_id=envelope.envelope_id,
        )
        for view_id, value, level in (
            ("coarse-view", Decimal("0.1"), 0),
            ("fine-view", Decimal("0.01"), 1),
        )
    )
    return envelope, views


def build_reference_system(
    reference_id: str,
    *,
    history_dependent: bool = False,
    multirate: bool = False,
    coupled: bool = False,
    numerical: bool = False,
    latency_seconds: Decimal = Decimal("0.01"),
    deadline_seconds: Decimal = Decimal("0.1"),
) -> SystemSpec:
    """Build a compact, fully typed SystemSpec around one reference oracle."""

    if multirate:
        clocks = tuple(
            _clock(clock_id) for clock_id in ("applied-clock", "receiver-clock", "requested-clock")
        )
        action_clock = "requested-clock"
        receiver_clock = "receiver-clock"
        denominator_clock = "requested-clock"
    else:
        clocks = (_clock("reference-clock"),)
        action_clock = receiver_clock = denominator_clock = "reference-clock"
    quantities = [
        _quantity(
            "action",
            QuantityKind.ACTION,
            action_clock,
            CausalPhase.ACTION_REQUESTED,
            OutcomeAccess.OUTCOME_BLIND,
        ),
        _quantity(
            "denominator",
            QuantityKind.DENOMINATOR,
            denominator_clock,
            CausalPhase.PRE_ACTION,
            OutcomeAccess.OUTCOME_BLIND,
        ),
        _quantity(
            "receiver",
            QuantityKind.RECEIVER,
            receiver_clock,
            CausalPhase.RECEIVER,
            OutcomeAccess.EVALUATOR_REVEAL,
            direction=ResponseDirection.HIGHER_IS_BETTER,
        ),
        _quantity(
            "sink",
            QuantityKind.SINK,
            receiver_clock,
            CausalPhase.RECEIVER,
            OutcomeAccess.EVALUATOR_REVEAL,
            direction=ResponseDirection.LOWER_IS_BETTER,
        ),
    ]
    history_ids: tuple[str, ...] = ()
    if history_dependent:
        quantities.append(
            _quantity(
                "history",
                QuantityKind.HISTORY,
                denominator_clock,
                CausalPhase.PRE_ACTION,
                OutcomeAccess.OUTCOME_BLIND,
            )
        )
        history_ids = ("history",)
    if coupled:
        quantities.extend(
            (
                _quantity(
                    "coupling-input",
                    QuantityKind.ACTION,
                    action_clock,
                    CausalPhase.ACTION_APPLIED,
                    OutcomeAccess.OUTCOME_BLIND,
                    dimension="command",
                    coordinate_frame="component-link",
                ),
                _quantity(
                    "coupling-output",
                    QuantityKind.RECEIVER,
                    action_clock,
                    CausalPhase.RECEIVER,
                    OutcomeAccess.EVALUATOR_REVEAL,
                    dimension="command",
                    coordinate_frame="component-link",
                    direction=ResponseDirection.HIGHER_IS_BETTER,
                ),
            )
        )
    quantities_tuple = tuple(sorted(quantities, key=lambda item: item.quantity_id))
    horizon = HorizonSpec(
        horizon_id="reference-horizon",
        clock_id=receiver_clock,
        duration=Decimal("10"),
        time_unit="s",
    )
    relation = RelationalIdentity(
        relation_id=f"{reference_id}-relation",
        denominator_quantity_ids=("denominator",),
        history_quantity_ids=history_ids,
        memoryless=not history_dependent,
        action_quantity_ids=("action",),
        receiver_quantity_ids=("receiver",),
        horizon=horizon,
    )
    world_kind = WorldKind.NUMERICAL_SIMULATOR if numerical else WorldKind.ANALYTIC_REFERENCE
    world = WorldSpec(
        world_id=reference_id,
        label=f"Truth-known {reference_id}",
        kind=world_kind,
        represented_physics=("declared-reference-dynamics",),
        unrepresented_physics=("physical-transport",) if numerical else (),
        privileged_truth_quantity_ids=(),
        maximum_evidence=EvidenceCeiling.CONTROLLER_USE,
        available_outcome_access=frozenset(
            {
                OutcomeAccess.EVALUATION_REVEALED,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.PRIVILEGED_TRUTH,
            }
        ),
    )
    envelope, views = _numerical_contract(
        reference_id,
        latency_seconds=latency_seconds,
        deadline_seconds=deadline_seconds,
    )
    authority = _authority()
    action_authority = (
        AuthorityAction.SIMULATION_EXECUTION
        if numerical
        else AuthorityAction.REFERENCE_WORLD_EXECUTION
    )
    ports = (
        PortSpec(
            port_id="action-in",
            quantity_id="action",
            clock_id=action_clock,
            direction=PortDirection.INPUT,
            balance_role=BalanceRole.COMMAND,
            authority_action=action_authority,
        ),
        PortSpec(
            port_id="receiver-out",
            quantity_id="receiver",
            clock_id=receiver_clock,
            direction=PortDirection.OUTPUT,
            balance_role=BalanceRole.OBSERVATION,
        ),
        PortSpec(
            port_id="sink-out",
            quantity_id="sink",
            clock_id=receiver_clock,
            direction=PortDirection.OUTPUT,
            balance_role=BalanceRole.ENERGY,
        ),
    )
    components: tuple[ComponentSpec, ...] = ()
    interfaces: tuple[InterfaceSpec, ...] = ()
    if coupled:
        source_relation = RelationalIdentity(
            relation_id=f"{reference_id}-source-relation",
            denominator_quantity_ids=("denominator",),
            history_quantity_ids=(),
            memoryless=True,
            action_quantity_ids=("action",),
            receiver_quantity_ids=("coupling-output",),
            horizon=horizon,
        )
        receiver_relation = RelationalIdentity(
            relation_id=f"{reference_id}-receiver-relation",
            denominator_quantity_ids=("denominator",),
            history_quantity_ids=(),
            memoryless=True,
            action_quantity_ids=("coupling-input",),
            receiver_quantity_ids=("receiver",),
            horizon=horizon,
        )
        components = (
            ComponentSpec(
                component_id="receiver-component",
                label="Receiver component",
                relation=receiver_relation,
                ports=(
                    PortSpec(
                        port_id="coupling-in",
                        quantity_id="coupling-input",
                        clock_id=action_clock,
                        direction=PortDirection.INPUT,
                        balance_role=BalanceRole.COMMAND,
                        authority_action=action_authority,
                    ),
                ),
                numerical_view_ids=("coarse-view", "fine-view"),
            ),
            ComponentSpec(
                component_id="source-component",
                label="Source component",
                relation=source_relation,
                ports=(
                    PortSpec(
                        port_id="coupling-out",
                        quantity_id="coupling-output",
                        clock_id=action_clock,
                        direction=PortDirection.OUTPUT,
                        balance_role=BalanceRole.COMMAND,
                    ),
                ),
                numerical_view_ids=("coarse-view", "fine-view"),
            ),
        )
        interfaces = (
            InterfaceSpec(
                interface_id="valid-coupling",
                source=PortRef(owner_id="source-component", port_id="coupling-out"),
                target=PortRef(owner_id="receiver-component", port_id="coupling-in"),
                balance_role=BalanceRole.COMMAND,
                balance_semantics=BalanceSemantics.COMMAND,
                uncertainty_contract_id="exact-reference-uncertainty",
                validity_contract_id="exact-reference-validity",
                clock_relation=ClockRelationSpec(
                    relation_id="component-clock-identity",
                    source_clock_id=action_clock,
                    target_clock_id=action_clock,
                    kind=ClockRelationKind.IDENTITY,
                    delay=None,
                    tolerance=Decimal("0"),
                    evidence_contract_id="exact-clock-evidence",
                ),
            ),
        )
    return SystemSpec(
        system_id=f"{reference_id}-system",
        label=f"System for {reference_id}",
        boundary_kind=SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world=world,
        relation=relation,
        clocks=clocks,
        quantities=quantities_tuple,
        independent_unit=IndependentUnitSpec(
            unit_id="reference-preparation",
            label="One truth-known physical preparation",
            grouping_key="reference-preparation-id",
        ),
        authority_policy=authority,
        ports=ports,
        components=components,
        interfaces=interfaces,
        computability_envelopes=(envelope,),
        numerical_views=views,
    )

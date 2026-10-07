"""Translate one declared RC study into native quantities and causal roles."""

from decimal import Decimal

from empirical_lawhood.adapters.simulators.rc_ladder_response.campaign import VIEWS
from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderStudyConfig
from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.quantities import (
    QuantityKind,
    QuantitySpec,
    ResponseDirection,
)
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
    NumericalCoordinateKind,
    NumericalCoordinateSpec,
    NumericalViewSpec,
    RandomnessSemantics,
    WorldKind,
    WorldSpec,
)

CLOCK = "rc-ladder-response-simulation-clock"
FRAME = "rc-ladder-response-elapsed-seconds"
UNIT = "rc-ladder-response-model-preparation"
CUTOFF = "rc-ladder-response-pre-action-cutoff"
ACTION_LEFT = "rc-ladder-response-left-requested-voltage"
ACTION_RIGHT = "rc-ladder-response-right-requested-voltage"
RECEIVER_VOLTAGE = "rc-ladder-response-node-voltage"
RECEIVER_CURRENT = "rc-ladder-response-boundary-current"
DENOMINATOR = "rc-ladder-response-component-roster"
HISTORY = "rc-ladder-response-initial-node-voltage"
SUPPORT = "rc-ladder-response-frozen-piecewise-action"
CHART = "rc-ladder-response-two-boundary-voltage-chart"
BUDGET = ResourceBudget(1, 1024**3, 0, 360, 1024**2, 3 * 1024**2)


def rc_system(study: ResistorCapacitorLadderStudyConfig, *, experiment_id: str) -> SystemSpec:
    """Represent SI circuit science without turning a numerical view into a unit."""

    stem = experiment_id
    pre = AvailabilitySpec(
        CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0)
    )
    action = AvailabilitySpec(
        CLOCK, CausalPhase.ACTION_REQUESTED, OutcomeAccess.OUTCOME_BLIND, Decimal(0)
    )
    observed = AvailabilitySpec(
        CLOCK,
        CausalPhase.RECEIVER,
        OutcomeAccess.OUTCOME_BLIND,
        study.model.output_times_seconds[1],
    )
    quantities = tuple(
        sorted(
            (
                QuantitySpec(
                    ACTION_LEFT,
                    "Left requested source voltage",
                    QuantityKind.ACTION,
                    "voltage",
                    "V",
                    "rc-boundary-source",
                    CLOCK,
                    action,
                ),
                QuantitySpec(
                    ACTION_RIGHT,
                    "Right requested boundary voltage",
                    QuantityKind.ACTION,
                    "voltage",
                    "V",
                    "rc-boundary-source",
                    CLOCK,
                    action,
                ),
                QuantitySpec(
                    DENOMINATOR,
                    "Frozen capacitances and resistances",
                    QuantityKind.DENOMINATOR,
                    "component-roster",
                    "F-and-ohm",
                    "rc-circuit-components",
                    CLOCK,
                    pre,
                ),
                QuantitySpec(
                    "rc-ladder-response-source-boundary",
                    "Finite source and termination impedance",
                    QuantityKind.BOUNDARY,
                    "resistance",
                    "ohm",
                    "rc-circuit-components",
                    CLOCK,
                    pre,
                ),
                QuantitySpec(
                    HISTORY,
                    "Initial capacitor node voltage",
                    QuantityKind.HISTORY,
                    "voltage",
                    "V",
                    "rc-node",
                    CLOCK,
                    pre,
                ),
                QuantitySpec(
                    RECEIVER_VOLTAGE,
                    "Capacitor node voltage",
                    QuantityKind.RECEIVER,
                    "voltage",
                    "V",
                    "rc-node",
                    CLOCK,
                    observed,
                    ResponseDirection.SIGNED_VECTOR,
                ),
                QuantitySpec(
                    RECEIVER_CURRENT,
                    "Source and termination currents",
                    QuantityKind.RECEIVER,
                    "electric-current",
                    "A",
                    "rc-boundary-source",
                    CLOCK,
                    observed,
                    ResponseDirection.SIGNED_VECTOR,
                ),
            ),
            key=lambda quantity: quantity.quantity_id,
        )
    )
    world = WorldSpec(
        f"{stem}.world",
        "Four-cell deterministic numerical RC model",
        WorldKind.NUMERICAL_SIMULATOR,
        (
            "finite-source-impedance",
            "four-cell-kirchhoff-ode",
            "piecewise-boundary-voltage",
        ),
        ("board-metrology", "component-temperature-drift", "physical-circuit-noise"),
        (),
        EvidenceCeiling.MEASUREMENT,
        frozenset(
            (
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
            )
        ),
    )
    compute = ComputabilityEnvelope(
        f"{stem}.computability",
        ("four-cell-linear-state-equation",),
        (),
        ("charge-balance", "finite-source-impedance", "two-solver-views"),
        ("charge-balance", "finite-source-impedance", "two-solver-views"),
        1,
        1024**3,
        0,
        360,
        3 * 1024**2,
        Decimal(360),
        None,
    )
    policy = AuthorityPolicy(
        f"{stem}.authority-policy",
        "owner.empirical-lawhood",
        "operator.empirical-lawhood",
        (stem,),
        frozenset((WorldKind.NUMERICAL_SIMULATOR,)),
        frozenset((AuthorityAction.SIMULATION_EXECUTION,)),
        frozenset((SourceAccessClass.NONE,)),
        (f"{stem}.exact-production-proof", f"{stem}.typed-execution-authority"),
        frozenset(),
        BUDGET,
        OutcomeAccess.OUTCOME_BLIND,
    )
    views = tuple(
        NumericalViewSpec(
            view,
            world.world_id,
            UNIT,
            "four-cell-rc-kirchhoff-equations",
            ("frozen-capacitance-and-resistance",),
            ("piecewise-left-and-right-voltage",),
            (
                NumericalCoordinateSpec(
                    f"{stem}.coordinate.{view}",
                    NumericalCoordinateKind.TIMESTEP,
                    study.backward_euler_maximum_step_seconds
                    if view == "backward-euler"
                    else study.model.output_times_seconds[1],
                    "s",
                    index,
                ),
            ),
            view,
            "1.0.0",
            "float64",
            "cpu",
            "cpython-3.11-numpy-2.4.6-scipy-1.17.1",
            RandomnessSemantics.NUMERICAL_REPEAT,
            "node-voltage-and-boundary-current",
            compute.envelope_id,
        )
        for index, view in enumerate(VIEWS)
    )
    return SystemSpec(
        f"{stem}.system",
        "One prepared four-cell RC model, two nested solver views",
        SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world,
        RelationalIdentity(
            f"{stem}.relation",
            (DENOMINATOR,),
            (HISTORY,),
            False,
            (ACTION_LEFT, ACTION_RIGHT),
            (RECEIVER_CURRENT, RECEIVER_VOLTAGE),
            HorizonSpec(
                f"{stem}.horizon", CLOCK, study.model.output_times_seconds[-1], "s"
            ),
        ),
        (
            ClockSpec(
                CLOCK,
                "Model elapsed time",
                "s",
                FRAME,
                SamplingSemantics.REGULAR,
                HoldSemantics.ZERO_ORDER,
                ClockLabelSemantics.INSTANT,
                study.model.output_times_seconds[1],
                Decimal(0),
            ),
        ),
        quantities,
        IndependentUnitSpec(
            UNIT,
            "One independently prepared model/component roster",
            "model-preparation-id",
        ),
        policy,
        tuple(
            PortSpec(
                f"{stem}.port.{action_id}",
                action_id,
                CLOCK,
                PortDirection.INPUT,
                BalanceRole.COMMAND,
                AuthorityAction.SIMULATION_EXECUTION,
            )
            for action_id in (ACTION_LEFT, ACTION_RIGHT)
        ),
        computability_envelopes=(compute,),
        numerical_views=views,
    )


__all__ = [
    "BUDGET",
    "CHART",
    "CLOCK",
    "CUTOFF",
    "DENOMINATOR",
    "HISTORY",
    "RECEIVER_CURRENT",
    "RECEIVER_VOLTAGE",
    "SUPPORT",
    "UNIT",
    "rc_system",
]

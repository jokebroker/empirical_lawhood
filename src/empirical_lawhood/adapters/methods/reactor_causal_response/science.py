"""Pre-acquisition absolute forecast question and its explicit policy support."""

from decimal import Decimal as D

from .config import EmpiricalRecipe
from .numerical import COLUMNS
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

PREFIX = "reactor-causal-response-study-programme"
CLOCK = "reactor-clock"
UNIT = "reactor-assigned-scenario-seed"
SUPPORT = "reactor-empirical-training-support"
CHART = "reactor-empirical-nine-words"
CUTOFF = "reactor-empirical-causal-callback"
METHOD = f"{PREFIX}.empirical-response"
BUDGET = ResourceBudget(1, 8 * 1024**3, 0, 12 * 3600, 256 * 1024**3, 256 * 1024**3)
# The task-allowance compiler sums task-local wall allowances, even for a concurrent
# graph. This is that serial sum, not the study's elapsed or CPU allocation.
# The recipe and injected guard separately retain the 12-hour whole-study cap.
PROTOCOL_BUDGET = ResourceBudget(4, 32 * 1024**3, 0, 238500, 256 * 1024**3, 256 * 1024**3)
CALIBRATION_UNITS = tuple(f"reactor-empirical-calibration-{i:03d}" for i in range(32))
QUALIFICATION_UNITS = tuple(f"reactor-empirical-qualification-{i:03d}" for i in range(32))
RECEIVERS = ("reactor-peak-temperature", "reactor-end-conversion")
UNITS = ("K", "1")
ASSUMPTIONS = tuple(
    sorted(
        (
            "fresh-assigned-simulation-roots",
            "empirical-affine-feature-basis",
            "chosen-path-only",
            "private-benchmark-not-training",
            "whole-root-maxima",
        )
    )
)


def empirical_system(design: EmpiricalRecipe) -> SystemSpec:
    pre = AvailabilitySpec(CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, D(0))
    action = AvailabilitySpec(
        CLOCK, CausalPhase.ACTION_REQUESTED, OutcomeAccess.OUTCOME_BLIND, D(0)
    )
    observed = AvailabilitySpec(CLOCK, CausalPhase.RECEIVER, OutcomeAccess.EVALUATION_SEALED, D(10))

    def quantity(key: str, kind: QuantityKind, dimension: str, unit: str) -> QuantitySpec:
        return QuantitySpec(
            key,
            key.replace("-", " "),
            kind,
            dimension,
            unit,
            "reactor-native",
            CLOCK,
            observed
            if kind is QuantityKind.RECEIVER
            else action
            if kind is QuantityKind.ACTION
            else pre,
            ResponseDirection.SIGNED_VECTOR
            if kind is QuantityKind.RECEIVER
            else ResponseDirection.NOT_APPLICABLE,
        )

    features = tuple((f"feature-{i:02d}", "dimensionless", "1") for i, _ in enumerate(COLUMNS)) + (
        ("callback-time", "time", "s"),
        ("action-id", "dimensionless", "1"),
        ("previous-applied-feed", "mass-rate", "kg/s"),
        ("previous-applied-jacket", "temperature", "K"),
        ("observed-dose", "mass", "kg"),
    )
    actions = ("reactor-feed", "reactor-jacket-command")
    receivers = tuple(sorted(RECEIVERS))
    quantities = tuple(
        sorted(
            (
                *(quantity(key, QuantityKind.HISTORY, dim, unit) for key, dim, unit in features),
                quantity(
                    "empirical-policy-version", QuantityKind.DENOMINATOR, "dimensionless", "1"
                ),
                quantity("reactor-jacket-boundary", QuantityKind.BOUNDARY, "temperature", "K"),
                quantity(actions[0], QuantityKind.ACTION, "mass-rate", "kg/s"),
                quantity(actions[1], QuantityKind.ACTION, "temperature", "K"),
                *(
                    quantity(key, QuantityKind.RECEIVER, dim, unit)
                    for key, dim, unit in zip(
                        RECEIVERS, ("temperature", "dimensionless"), UNITS, strict=True
                    )
                ),
            ),
            key=lambda q: q.quantity_id,
        )
    )
    world = WorldSpec(
        f"{PREFIX}.world",
        "Pinned reactor with preassigned public-envelope simulation units",
        WorldKind.NUMERICAL_SIMULATOR,
        ("native-actuator-stages", "native-delayed-noisy-observer", "native-rk4"),
        (
            "physical-reactor-transport",
            "private-panel-coverage-guarantee",
            "uniform-full-menu-coverage",
        ),
        (),
        EvidenceCeiling.CONTROLLER_USE,
        frozenset(
            (
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
            )
        ),
    )
    compute = ComputabilityEnvelope(
        f"{PREFIX}.computability",
        ("empirical-ten-second-forecast",),
        (),
        ("two-coupled-timesteps", "whole-episode-calibration"),
        ("two-coupled-timesteps", "whole-episode-calibration"),
        BUDGET.cpu_cores,
        BUDGET.memory_bytes,
        0,
        BUDGET.wall_time_seconds,
        BUDGET.output_bytes,
        D(BUDGET.wall_time_seconds),
        None,
    )
    policy = AuthorityPolicy(
        f"{PREFIX}.authority",
        "owner.empirical-lawhood",
        "operator.empirical-lawhood",
        (PREFIX,),
        frozenset((WorldKind.NUMERICAL_SIMULATOR,)),
        frozenset((AuthorityAction.SIMULATION_EXECUTION,)),
        frozenset((SourceAccessClass.NONE,)),
        (f"{PREFIX}.exact-production-proof", f"{PREFIX}.typed-execution-authority"),
        frozenset(),
        PROTOCOL_BUDGET,
        OutcomeAccess.OUTCOME_BLIND,
    )
    views = tuple(
        NumericalViewSpec(
            f"reactor-{name}",
            world.world_id,
            UNIT,
            "upstream-reactor-equations",
            ("upstream-plant-params",),
            (SUPPORT,),
            (
                NumericalCoordinateSpec(
                    f"reactor-dt-{name}", NumericalCoordinateKind.TIMESTEP, dt, "s", index
                ),
            ),
            "upstream-rk4",
            "1.0.0",
            "float64",
            "cpu",
            "pinned-runtime-required-before-issue",
            RandomnessSemantics.NUMERICAL_REPEAT,
            "upstream-delayed-noisy-callback",
            compute.envelope_id,
        )
        for index, (name, dt) in enumerate(zip(("native", "refined"), (D(1), D(".5")), strict=True))
    )
    return SystemSpec(
        f"{PREFIX}.system",
        "Frozen empirical reactor chosen-path response law",
        SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world,
        RelationalIdentity(
            f"{PREFIX}.relation",
            ("empirical-policy-version",),
            tuple(sorted(key for key, _, _ in features)),
            False,
            actions,
            receivers,
            HorizonSpec(f"{PREFIX}.horizon", CLOCK, D(10), "s"),
        ),
        (
            ClockSpec(
                CLOCK,
                "Upstream simulator time",
                "s",
                "reactor-episode-time",
                SamplingSemantics.REGULAR,
                HoldSemantics.ZERO_ORDER,
                ClockLabelSemantics.INSTANT,
                D(10),
                D(0),
            ),
        ),
        quantities,
        IndependentUnitSpec(
            UNIT, "One preassigned plant/fault/noise seed episode", "assigned-scenario-seed"
        ),
        policy,
        tuple(
            PortSpec(
                f"port.{a}",
                a,
                CLOCK,
                PortDirection.INPUT,
                BalanceRole.COMMAND,
                AuthorityAction.SIMULATION_EXECUTION,
            )
            for a in actions
        ),
        computability_envelopes=(compute,),
        numerical_views=views,
    )

"""Pre-acquisition absolute forecast question and its explicit policy support."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import ReactorBatchConfig, assigned_scenarios
from empirical_lawhood.adapters.methods.reactor_prefix_response.batch_calibration import NUMERICAL_TOLERANCE, RECEIVERS, UNITS, Triple
from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
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

PREFIX = "reactor-batch-native"
CLOCK = "reactor-clock"
UNIT = "reactor-assigned-scenario-seed"
SUPPORT = "reactor-reference-policy-public-envelope"
CHART = "reactor-reference-policy-singleton-chart"
CUTOFF = "reactor-forecast-pre-acquisition-cutoff"
METHOD = f"{PREFIX}.absolute-forecast"
BUDGET = ResourceBudget(1, 4 * 1024**3, 0, 30000, 9 * 1024**3, 9 * 1024**3)
CALIBRATION_UNITS = tuple(s.unit_id for s in assigned_scenarios() if s.split == "calibration")
ALL_UNITS = tuple(s.unit_id for s in assigned_scenarios())
ASSUMPTIONS = tuple(
    sorted(
        (
            "exchangeable-assigned-simulation-episodes",
            "fixed-reference-observer-and-nominal-rk4",
            "on-policy-singleton-action-only",
            "private-benchmark-distribution-not-calibrated",
            "whole-episode-maxima-not-callback-replication",
        )
    )
)


@dataclass(frozen=True, slots=True)
class ReactorForecastDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-reference-policy-forecast/reactor-forecast-design'
    config_id: str
    native: ReactorBatchConfig
    confidence_level: D = D(".90")
    numerical_allowances: Triple = NUMERICAL_TOLERANCE
    calibration_rank: int = 32
    minimum_heldout_covered: int = 9
    forecast_horizon_s: D = D(10)
    receiver_ids: tuple[str, ...] = RECEIVERS
    policy: str = "unmodified-upstream-reference-command-or-nonattempt"
    invalid_unit_rule: str = "infinite-score-no-exclusion-no-top-up"

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.confidence_level != D(".90")
            or self.numerical_allowances != NUMERICAL_TOLERANCE
            or self.calibration_rank != 32
            or self.minimum_heldout_covered != 9
            or self.forecast_horizon_s != 10
            or self.receiver_ids != RECEIVERS
            or self.policy != "unmodified-upstream-reference-command-or-nonattempt"
            or self.invalid_unit_rule != "infinite-score-no-exclusion-no-top-up"
        ):
            raise ValueError("forecast design changes its frozen estimand, support or refusal rule")


def forecast_design() -> ReactorForecastDesign:
    return ReactorForecastDesign(
        f"{PREFIX}.design", ReactorBatchConfig(f"{PREFIX}.native", assigned_scenarios())
    )


def forecast_system(design: ReactorForecastDesign) -> SystemSpec:
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

    features = (
        ("observation-time", "time", "s"),
        ("observed-dose", "mass", "kg"),
        ("observer-n-a", "amount", "mol"),
        ("observer-n-b", "amount", "mol"),
        ("observer-volume", "volume", "m^3"),
        ("observer-temperature", "temperature", "K"),
        ("observer-jacket", "temperature", "K"),
        ("observer-ua", "heat-conductance", "W/K"),
        ("observer-kinetic-multiplier", "dimensionless", "1"),
        ("previous-feed", "mass-rate", "kg/s"),
        ("previous-jacket", "temperature", "K"),
    )
    actions = ("reactor-feed", "reactor-jacket-command")
    receivers = tuple(sorted(RECEIVERS))
    quantities = tuple(
        sorted(
            (
                *(quantity(key, QuantityKind.HISTORY, dim, unit) for key, dim, unit in features),
                quantity(
                    "reference-policy-version", QuantityKind.DENOMINATOR, "dimensionless", "1"
                ),
                quantity("reactor-jacket-boundary", QuantityKind.BOUNDARY, "temperature", "K"),
                quantity(actions[0], QuantityKind.ACTION, "mass-rate", "kg/s"),
                quantity(actions[1], QuantityKind.ACTION, "temperature", "K"),
                *(
                    quantity(key, QuantityKind.RECEIVER, dim, unit)
                    for key, dim, unit in zip(
                        RECEIVERS, ("temperature", "mass", "dimensionless"), UNITS, strict=True
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
            "off-policy-action-exchange",
            "physical-reactor-transport",
            "private-panel-coverage-guarantee",
        ),
        (),
        EvidenceCeiling.LOCAL_LAW,
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
        ("rolling-absolute-forecasts",),
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
        BUDGET,
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
            f"python-{design.native.python_version}-numpy-{design.native.numpy_version}",
            RandomnessSemantics.NUMERICAL_REPEAT,
            "upstream-delayed-noisy-callback",
            compute.envelope_id,
        )
        for index, (name, dt) in enumerate(
            zip(("native", "refined"), design.native.plant_timesteps_s, strict=True)
        )
    )
    return SystemSpec(
        f"{PREFIX}.system",
        "Rolling on-policy absolute reactor forecasts over full episodes",
        SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world,
        RelationalIdentity(
            f"{PREFIX}.relation",
            ("reference-policy-version",),
            tuple(sorted(key for key, _, _ in features)),
            False,
            actions,
            receivers,
            HorizonSpec(f"{PREFIX}.horizon", CLOCK, design.forecast_horizon_s, "s"),
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

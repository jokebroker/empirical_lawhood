"""Outcome-blind scientific design for the bounded reactor integration slice.

This record contains no future artifact identities. Materialization may bind
those identities later; it may not change these scientific operands.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import CALIBRATION, HELDOUT, SCENARIOS, VIEWS, ReactorPrefixConfig
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

PREFIX = "reactor-prefix-native"
CLOCK = "reactor-clock"
FRAME = "reactor-episode-time"
RECEIVER = "reactor-delayed-temperature"
UNIT = "reactor-public-scenario-seed"
SUPPORT = "reactor-initial-zero-feed-prefix"
CHART = "reactor-two-decision-jacket-chart"
HISTORY = "reactor-prescribed-initial-state"
CUTOFF = "tbs-reactor-prefix-design-cutoff"
METHOD = "finite-action.compatibility-set"


@dataclass(frozen=True, slots=True)
class ReactorScienceDesign(CanonicalRecord):
    "A finite contrast question, not an absolute forecast or controller-use controller.\n\n    The two arms differ only at the first jacket command. Receiver values come\n    from the callback at 20 s (native temperature at 10 s plus native noise).\n    Whole scenario/seed units are allocated before contact; arms and numerical\n    views stay nested. Thresholds describe this fixed panel in kelvin.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prefix-response/reactor-science-design'
    config_id: str
    native: ReactorPrefixConfig
    maximum_heldout_absolute_error_k: Decimal
    maximum_refinement_absolute_delta_k: Decimal
    minimum_complete_units: int
    bootstrap_replicates: int
    bootstrap_seed: int
    simultaneous_confidence_level: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        if (
            self.maximum_heldout_absolute_error_k != Decimal("0.01")
            or self.maximum_refinement_absolute_delta_k != Decimal("0.001")
            or self.minimum_complete_units != 2
            or self.bootstrap_replicates != 200
            or self.bootstrap_seed != 19
            or self.simultaneous_confidence_level != Decimal("0.95")
        ):
            raise ValueError("reactor science changes its frozen finite-panel design")


def reactor_science_design() -> ReactorScienceDesign:
    return ReactorScienceDesign(
        f"{PREFIX}.science-design",
        ReactorPrefixConfig(
            f"{PREFIX}.native-config",
            CALIBRATION,
            HELDOUT,
            Decimal(20),
            "2.4.6",
            "3.11.14",
        ),
        Decimal("0.01"),
        Decimal("0.001"),
        2,
        200,
        19,
        Decimal("0.95"),
    )


def reactor_system(design: ReactorScienceDesign) -> SystemSpec:
    """Code-owned translation of the frozen design into ordinary system records."""
    pre = AvailabilitySpec(
        CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0)
    )
    action = AvailabilitySpec(
        CLOCK, CausalPhase.ACTION_REQUESTED, OutcomeAccess.OUTCOME_BLIND, Decimal(0)
    )
    observed = AvailabilitySpec(
        CLOCK, CausalPhase.RECEIVER, OutcomeAccess.OUTCOME_BLIND, Decimal(10)
    )
    quantities = tuple(
        sorted(
            (
                QuantitySpec(
                    "reactor-feed",
                    "Requested feed",
                    QuantityKind.ACTION,
                    "mass-rate",
                    "kg/s",
                    "reactor-native-actuator",
                    CLOCK,
                    action,
                ),
                QuantitySpec(
                    "reactor-jacket-command",
                    "Requested jacket temperature",
                    QuantityKind.ACTION,
                    "temperature",
                    "K",
                    "reactor-native-actuator",
                    CLOCK,
                    action,
                ),
                QuantitySpec(
                    "reactor-initial-temperature",
                    "Declared initial temperature",
                    QuantityKind.DENOMINATOR,
                    "temperature",
                    "K",
                    "reactor-native-temperature",
                    CLOCK,
                    pre,
                ),
                QuantitySpec(
                    "reactor-initial-jacket",
                    "Declared initial jacket state",
                    QuantityKind.HISTORY,
                    "temperature",
                    "K",
                    "reactor-native-temperature",
                    CLOCK,
                    pre,
                ),
                QuantitySpec(
                    "reactor-jacket-boundary",
                    "Jacket heat-exchange boundary",
                    QuantityKind.BOUNDARY,
                    "temperature",
                    "K",
                    "reactor-native-temperature",
                    CLOCK,
                    pre,
                ),
                QuantitySpec(
                    RECEIVER,
                    "Delayed noisy reactor temperature",
                    QuantityKind.RECEIVER,
                    "temperature",
                    "K",
                    "reactor-native-temperature",
                    CLOCK,
                    observed,
                    ResponseDirection.SIGNED_VECTOR,
                ),
            ),
            key=lambda q: q.quantity_id,
        )
    )
    world = WorldSpec(
        f"{PREFIX}.world",
        "Pinned upstream reactor with explicitly declared scenario seeds",
        WorldKind.NUMERICAL_SIMULATOR,
        ("delayed-noisy-temperature", "native-actuator-limits", "native-rk4-reactor"),
        ("full-batch-task-efficacy", "physical-reactor-validation"),
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
        ("jacket-prefix-temperature-contrast",),
        (),
        ("paired-whole-units", "timestep-refinement"),
        ("paired-whole-units", "timestep-refinement"),
        1,
        1024**3,
        0,
        6300,
        48 * 1024**2,
        Decimal(6300),
        None,
    )
    policy = AuthorityPolicy(
        f"{PREFIX}.authority-policy",
        "owner.empirical-lawhood",
        "operator.empirical-lawhood",
        (PREFIX,),
        frozenset((WorldKind.NUMERICAL_SIMULATOR,)),
        frozenset((AuthorityAction.SIMULATION_EXECUTION,)),
        frozenset((SourceAccessClass.NONE,)),
        (f"{PREFIX}.exact-production-proof", f"{PREFIX}.typed-execution-authority"),
        frozenset(),
        ResourceBudget(1, 1024**3, 0, 6300, 28 * 1024**2, 48 * 1024**2),
        OutcomeAccess.OUTCOME_BLIND,
    )
    views = tuple(
        NumericalViewSpec(
            f"reactor-{view}",
            world.world_id,
            UNIT,
            "upstream-reactor-equations",
            ("upstream-plant-params",),
            ("initial-zero-feed-jacket-prefix",),
            (
                NumericalCoordinateSpec(
                    f"reactor-dt-{view}", NumericalCoordinateKind.TIMESTEP, dt, "s", i
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
        for i, (view, dt) in enumerate(VIEWS)
    )
    return SystemSpec(
        f"{PREFIX}.system",
        "Two-decision local jacket contrast on five declared units",
        SystemBoundaryKind.OPEN_DRIVEN_DISSIPATIVE,
        world,
        RelationalIdentity(
            f"{PREFIX}.relation",
            ("reactor-initial-temperature",),
            ("reactor-initial-jacket",),
            False,
            ("reactor-feed", "reactor-jacket-command"),
            (RECEIVER,),
            HorizonSpec(f"{PREFIX}.horizon", CLOCK, design.native.horizon_s, "s"),
        ),
        (
            ClockSpec(
                CLOCK,
                "Upstream simulator time",
                "s",
                FRAME,
                SamplingSemantics.REGULAR,
                HoldSemantics.ZERO_ORDER,
                ClockLabelSemantics.INSTANT,
                Decimal(10),
                Decimal(0),
            ),
        ),
        quantities,
        IndependentUnitSpec(
            UNIT,
            "One declared scenario and its fixed noise seed",
            "scenario-seed-unit-id",
        ),
        policy,
        tuple(
            PortSpec(
                f"port.{q}",
                q,
                CLOCK,
                PortDirection.INPUT,
                BalanceRole.COMMAND,
                AuthorityAction.SIMULATION_EXECUTION,
            )
            for q in ("reactor-feed", "reactor-jacket-command")
        ),
        computability_envelopes=(compute,),
        numerical_views=views,
    )


def unit_ids(design: ReactorScienceDesign | None = None) -> tuple[str, ...]:
    scenarios = (
        SCENARIOS
        if design is None
        else tuple(
            sorted(
                (*design.native.calibration_scenarios, *design.native.heldout_scenarios)
            )
        )
    )
    return tuple(f"unit.{s.replace('_', '-')}" for s in scenarios)

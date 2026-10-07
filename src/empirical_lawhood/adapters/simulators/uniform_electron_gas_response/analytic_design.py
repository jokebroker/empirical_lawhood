"""Declare the one-unit synthetic uniform electron gas reference world and its SI roles."""

from decimal import Decimal

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

from .native_quickstart import UniformElectronGasAnalyticReferenceConfig

CLOCK = "uniform-electron-gas-static-acquisition-stage"
FRAME = "uniform-electron-gas-request-accept-apply-receive"
UNIT = "uniform-electron-gas-complete-analytic-acquisition"
CUTOFF = "uniform-electron-gas-pre-action-cutoff"
ACTION = "uniform-electron-gas-transverse-vector-potential"
RECEIVER = "uniform-electron-gas-signed-transverse-current-density"
DENOMINATOR = "uniform-electron-gas-neutralizing-jellium-denominator"
SUPPORT = "uniform-electron-gas-signed-finite-q-action-chart"
CHART = "uniform-electron-gas-u-and-q-chart"
BUDGET = ResourceBudget(1, 1024**3, 0, 360, 2 * 1024**2, 3 * 1024**2)


def analytic_system(config: UniformElectronGasAnalyticReferenceConfig, *, experiment_id: str) -> SystemSpec:
    """One analytic acquisition; q/u conditions never inflate unit count."""

    stem = experiment_id
    pre = AvailabilitySpec(
        CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0)
    )
    action = AvailabilitySpec(
        CLOCK,
        CausalPhase.ACTION_REQUESTED,
        OutcomeAccess.OUTCOME_BLIND,
        Decimal(config.requested_clock),
    )
    observed = AvailabilitySpec(
        CLOCK,
        CausalPhase.RECEIVER,
        OutcomeAccess.OUTCOME_BLIND,
        Decimal(config.receiver_clock),
    )
    quantities = tuple(
        sorted(
            (
                QuantitySpec(
                    DENOMINATOR,
                    "3D neutralizing jellium at frozen r_s and temperature",
                    QuantityKind.DENOMINATOR,
                    "free-electron-state",
                    "dimensionless-r_s-and-K",
                    "uniform-electron-gas-response-denominator",
                    CLOCK,
                    pre,
                ),
                QuantitySpec(
                    ACTION,
                    "Signed transverse vector potential at q along x",
                    QuantityKind.ACTION,
                    "vector-potential",
                    "T*m",
                    "uniform-electron-gas-transverse-y",
                    CLOCK,
                    action,
                ),
                QuantitySpec(
                    RECEIVER,
                    "Signed y-current density after static convergence",
                    QuantityKind.RECEIVER,
                    "current-density",
                    "A/m^2",
                    "uniform-electron-gas-transverse-y",
                    CLOCK,
                    observed,
                    ResponseDirection.SIGNED_VECTOR,
                ),
            ),
            key=lambda value: value.quantity_id,
        )
    )
    world = WorldSpec(
        f"{stem}.world",
        "Disclosed synthetic uniform electron gas analytic UEG reference",
        WorldKind.ANALYTIC_REFERENCE,
        (
            "3d-neutralizing-jellium",
            "finite-q-signed-current",
            "static-transverse-london-reference",
        ),
        (
            "gauge-closed-material-data",
            "independent-numerical-solver-views",
            "order-parameter-measurement",
        ),
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
        ("closed-ueg-scaling-and-slab-profile",),
        (),
        ("finite-q-intercept", "signed-realized-si-action", "slab-centre-inverse"),
        ("finite-q-intercept", "signed-realized-si-action", "slab-centre-inverse"),
        1,
        BUDGET.memory_bytes,
        0,
        BUDGET.wall_time_seconds,
        BUDGET.output_bytes,
        Decimal(BUDGET.wall_time_seconds),
        None,
    )
    policy = AuthorityPolicy(
        f"{stem}.authority-policy",
        "owner.empirical-lawhood",
        "operator.empirical-lawhood",
        (stem,),
        frozenset((WorldKind.ANALYTIC_REFERENCE,)),
        frozenset((AuthorityAction.REFERENCE_WORLD_EXECUTION,)),
        frozenset((SourceAccessClass.NONE,)),
        (f"{stem}.exact-production-proof", f"{stem}.typed-execution-authority"),
        frozenset(),
        BUDGET,
        OutcomeAccess.OUTCOME_BLIND,
    )
    view = NumericalViewSpec(
        "analytic-signed-si",
        world.world_id,
        UNIT,
        "closed-k-of-q-positive-reference",
        ("r_s-and-reference-depth",),
        ("five-signed-u-values", "four-finite-q-values"),
        (
            NumericalCoordinateSpec(
                f"{stem}.coordinate.profile-points",
                NumericalCoordinateKind.SPATIAL_GRID,
                Decimal(config.analytic_profile_points),
                "points",
                0,
            ),
        ),
        "analytic-closed-form",
        "1.0.0",
        "decimal-and-float64",
        "cpu",
        "cpython-3.11-scipy-1.17.1-uniform-electron-gas-analytic-reference",
        RandomnessSemantics.DETERMINISTIC,
        "signed-current-and-slab-centre",
        compute.envelope_id,
    )
    return SystemSpec(
        f"{stem}.system",
        "One synthetic UEG acquisition with 20 nested q/u conditions",
        SystemBoundaryKind.CLOSED_REFERENCE,
        world,
        RelationalIdentity(
            f"{stem}.relation",
            (DENOMINATOR,),
            (),
            True,
            (ACTION,),
            (RECEIVER,),
            HorizonSpec(
                f"{stem}.horizon", CLOCK, Decimal(config.receiver_clock), "stage"
            ),
        ),
        (
            ClockSpec(
                CLOCK,
                "Static acquisition event stage",
                "stage",
                FRAME,
                SamplingSemantics.REGULAR,
                HoldSemantics.ZERO_ORDER,
                ClockLabelSemantics.INSTANT,
                Decimal(1),
                Decimal(0),
            ),
        ),
        quantities,
        IndependentUnitSpec(
            UNIT,
            "One complete prepared analytic acquisition, not each q/u cell",
            "complete-acquisition-id",
        ),
        policy,
        (
            PortSpec(
                f"{stem}.port.action",
                ACTION,
                CLOCK,
                PortDirection.INPUT,
                BalanceRole.COMMAND,
                AuthorityAction.REFERENCE_WORLD_EXECUTION,
            ),
        ),
        computability_envelopes=(compute,),
        numerical_views=(view,),
    )


__all__ = [
    "ACTION",
    "BUDGET",
    "CHART",
    "CLOCK",
    "CUTOFF",
    "DENOMINATOR",
    "RECEIVER",
    "SUPPORT",
    "UNIT",
    "analytic_system",
]

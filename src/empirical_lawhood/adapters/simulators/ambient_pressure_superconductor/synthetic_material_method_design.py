'Declare the one-unit synthetic ambient pressure superconductor reference world and its SI roles.'

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

from .synthetic_material_method_contracts import SyntheticMaterialResponseMethodConfig

CLOCK = 'ambient-pressure-superconductor-gauge-covariant-response-method-stage'
FRAME = 'ambient-pressure-superconductor-request-accept-apply-receive'
UNIT = 'ambient-pressure-superconductor-complete-synthetic-method-suite'
CUTOFF = 'ambient-pressure-superconductor-pre-action-cutoff'
ACTION = 'ambient-pressure-superconductor-signed-transverse-peierls-probe'
RECEIVER = 'ambient-pressure-superconductor-signed-lattice-current'
DENOMINATOR = 'ambient-pressure-superconductor-clean-lattice-bcs-denominator'
SUPPORT = 'ambient-pressure-superconductor-nine-nested-gauge-covariant-response-fixtures'
CHART = 'ambient-pressure-superconductor-finite-q-signed-probe-chart'
BUDGET = ResourceBudget(1, 1024**3, 0, 900, 2 * 1024**2, 3 * 1024**2)


def synthetic_material_response_method_method_system(config: SyntheticMaterialResponseMethodConfig, *, experiment_id: str) -> SystemSpec:
    """One disclosed method suite; fixtures and meshes are nested conditions."""

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
                    "Clean lattice BCS at 300 K and ambient pressure; no material mapping",
                    QuantityKind.DENOMINATOR,
                    "lattice-bcs-state",
                    "eV-and-K-and-Pa",
                    'ambient-pressure-superconductor-effective-lattice-frame',
                    CLOCK,
                    pre,
                ),
                QuantitySpec(
                    ACTION,
                    "Signed transverse Peierls bond phase at finite q",
                    QuantityKind.ACTION,
                    "bond-phase",
                    "dimensionless-phase",
                    'ambient-pressure-superconductor-lattice-transverse-y',
                    CLOCK,
                    action,
                ),
                QuantitySpec(
                    RECEIVER,
                    "Signed transverse current from the BdG free-energy derivative",
                    QuantityKind.RECEIVER,
                    "lattice-current",
                    "eV/link",
                    'ambient-pressure-superconductor-lattice-transverse-y',
                    CLOCK,
                    observed,
                    ResponseDirection.SIGNED_VECTOR,
                ),
            ),
            key=lambda value: value.quantity_id,
        )
    )
    world = WorldSpec(
        f'{stem}.world',
        'Disclosed synthetic ambient pressure superconductor gauge-covariant gauge covariant response method suite',
        WorldKind.NUMERICAL_SIMULATOR,
        tuple(
            sorted(
                (
                    "clean-lattice-bcs",
                    "finite-q-signed-current-and-normal-subtraction",
                    "nine-nested-method-fixtures",
                )
            )
        ),
        tuple(
            sorted(
                (
                    "material-specific-300k-pairing-state",
                    "wannier-to-lattice-and-si-compatibility",
                    'material-order-or-admission-admission',
                )
            )
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
        f'{stem}.computability',
        ("finite-temperature-lattice-bdg-eigensolver",),
        (),
        tuple(
            sorted(
                (
                    "signed-current",
                    "normal-subtraction",
                    "ward-residual",
                    "base-refined-interval",
                )
            )
        ),
        tuple(
            sorted(
                (
                    "signed-current",
                    "normal-subtraction",
                    "ward-residual",
                    "base-refined-interval",
                )
            )
        ),
        1,
        BUDGET.memory_bytes,
        0,
        BUDGET.wall_time_seconds,
        BUDGET.output_bytes,
        Decimal(BUDGET.wall_time_seconds),
        None,
    )
    policy = AuthorityPolicy(
        f'{stem}.authority-policy',
        "owner.empirical-lawhood",
        "operator.empirical-lawhood",
        (stem,),
        frozenset((WorldKind.NUMERICAL_SIMULATOR,)),
        frozenset((AuthorityAction.SIMULATION_EXECUTION,)),
        frozenset((SourceAccessClass.NONE,)),
        (f'{stem}.exact-production-proof', f'{stem}.typed-execution-authority'),
        frozenset(),
        BUDGET,
        OutcomeAccess.OUTCOME_BLIND,
    )

    def view(view_id: str, nx: int, nky: int) -> NumericalViewSpec:
        return NumericalViewSpec(
            view_id,
            world.world_id,
            UNIT,
            "finite-temperature-lattice-bdg",
            ("hopping-mu-gap-temperature-pressure",),
            ("nine-nested-fixtures", "signed-finite-q-probe"),
            tuple(
                sorted(
                    (
                        NumericalCoordinateSpec(
                            f'{stem}.coordinate.{view_id}.nx',
                            NumericalCoordinateKind.SPATIAL_GRID,
                            Decimal(nx),
                            "sites",
                            0,
                        ),
                        NumericalCoordinateSpec(
                            f'{stem}.coordinate.{view_id}.nky',
                            NumericalCoordinateKind.SPATIAL_GRID,
                            Decimal(nky),
                            "points",
                            0,
                        ),
                    ),
                    key=lambda value: value.coordinate_id,
                )
            ),
            "numpy-eigvalsh",
            "1.0.0",
            "complex128-and-float64",
            "cpu",
            "cpython-3.11-numpy-synthetic-material-response-method",
            RandomnessSemantics.DETERMINISTIC,
            "signed-lattice-current-and-stiffness",
            compute.envelope_id,
        )

    return SystemSpec(
        f'{stem}.system',
        'One synthetic gauge covariant response method suite with nine nested fixtures and two meshes',
        SystemBoundaryKind.CLOSED_REFERENCE,
        world,
        RelationalIdentity(
            f'{stem}.relation',
            (DENOMINATOR,),
            (),
            True,
            (ACTION,),
            (RECEIVER,),
            HorizonSpec(
                f'{stem}.horizon', CLOCK, Decimal(config.receiver_clock), "stage"
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
            "One complete synthetic method suite, not each fixture or mesh",
            "complete-method-suite-id",
        ),
        policy,
        (
            PortSpec(
                f'{stem}.port.action',
                ACTION,
                CLOCK,
                PortDirection.INPUT,
                BalanceRole.COMMAND,
                AuthorityAction.SIMULATION_EXECUTION,
            ),
        ),
        computability_envelopes=(compute,),
        numerical_views=(
            view(
                'ambient-pressure-superconductor-gauge-covariant-response-base',
                config.fixture_suite.base_nx,
                config.fixture_suite.base_nky,
            ),
            view(
                'ambient-pressure-superconductor-gauge-covariant-response-refined',
                config.fixture_suite.refined_nx,
                config.fixture_suite.refined_nky,
            ),
        ),
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
    'synthetic_material_response_method_method_system',
]

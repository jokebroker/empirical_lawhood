"""Fresh upper response relation and explicit compatibility with unchanged F."""

from decimal import Decimal as D
from dataclasses import replace
from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.systems import (
    SystemSpec,
    SystemBoundaryKind,
    IndependentUnitSpec,
    RelationalIdentity,
    PortSpec,
    PortDirection,
    BalanceRole,
)
from empirical_lawhood.kernel.time import (
    AvailabilitySpec,
    CausalPhase,
    ClockSpec,
    ClockLabelSemantics,
    HoldSemantics,
    SamplingSemantics,
    HorizonSpec,
)
from empirical_lawhood.kernel.worlds import (
    WorldSpec,
    WorldKind,
    ComputabilityEnvelope,
    NumericalViewSpec,
    NumericalCoordinateSpec,
    NumericalCoordinateKind,
    RandomnessSemantics,
)
from empirical_lawhood.adapters.methods.preparation_applicability.config import PreparationApplicabilityStage
from empirical_lawhood.adapters.methods.preparation_applicability.records import PREFIX

CLOCK = f"{PREFIX}.reference-clock"
FRAME = f"{PREFIX}.fixed-prefix-frame"
UNIT = f"{PREFIX}.independent-stochastic-root"
CUTOFF = f"{PREFIX}.prefix-cutoff"
CHART = f"{PREFIX}.three-preparation-chart"


def budget(stage: PreparationApplicabilityStage) -> ResourceBudget:
    return ResourceBudget(
        4,
        32 * 1024**3,
        0,
        stage.cpu_seconds,
        32 * 1024**3,
        (5 if stage.phase == "Q" else 10) * 1024**3,
    )


def system(stage: PreparationApplicabilityStage) -> SystemSpec:
    stem = stage.config_id
    pre = AvailabilitySpec(CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, D(4096))
    observed = AvailabilitySpec(
        CLOCK, CausalPhase.RECEIVER, OutcomeAccess.EVALUATION_SEALED, D(4688)
    )
    quantities = tuple(
        QuantitySpec(
            f"{stem}.{key}", label, kind, dimension, unit, FRAME, CLOCK, available, direction
        )
        for key, label, kind, dimension, unit, available, direction in (
            (
                "denominator",
                "q2 six-matrix ordinary independent innovations; couplings2/3 and22/3",
                QuantityKind.DENOMINATOR,
                "medium-identity",
                "identity",
                pre,
                ResponseDirection.NOT_APPLICABLE,
            ),
            (
                "history",
                "Causal SKR24 at 4096; no preparation outcomes or requests",
                QuantityKind.HISTORY,
                "native-history",
                "native-hs-phase-history",
                pre,
                ResponseDirection.NOT_APPLICABLE,
            ),
            (
                "preparation",
                "HOLD or two fixed returned Y preparation pulses",
                QuantityKind.ACTION,
                "native-schedule-identity",
                "native-word",
                replace(pre, phase=CausalPhase.ACTION_REQUESTED),
                ResponseDirection.NOT_APPLICABLE,
            ),
            (
                "validity-margin",
                "Minimum normalized full lower-law validity margin",
                QuantityKind.RECEIVER,
                "dimensionless",
                "1",
                observed,
                ResponseDirection.SIGNED_VECTOR,
            ),
        )
    )
    relation = RelationalIdentity(
        f"{stem}.relation",
        (quantities[0].quantity_id,),
        (quantities[1].quantity_id,),
        False,
        (quantities[2].quantity_id,),
        (quantities[3].quantity_id,),
        HorizonSpec(f"{stem}.horizon", CLOCK, D(592), "reference-tick"),
    )
    world = WorldSpec(
        f"{PREFIX}.world",
        "Constructed boundary-population simulation evidence only",
        WorldKind.NUMERICAL_SIMULATOR,
        ("BAOAB-unit-kinetic-mass", "fixed-prefix-two-port-frame", "q2-six-Hermitian-matrices"),
        ("physical-material-realization",),
        (),
        EvidenceCeiling.RESPONSE,
        frozenset(
            (
                OutcomeAccess.OUTCOME_BLIND,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                OutcomeAccess.EVALUATION_SEALED,
                OutcomeAccess.EVALUATOR_REVEAL,
            )
        ),
    )
    compute = ComputabilityEnvelope(
        f"{stem}.computability",
        ("finite-native-assay-and-paired-root-comparison",),
        (),
        ("bounded-native-checkpoint",),
        ("bounded-native-checkpoint",),
        4,
        32 * 1024**3,
        0,
        stage.wall_seconds,
        budget(stage).output_bytes,
        D(stage.cpu_seconds),
        None,
    )
    views = tuple(
        NumericalViewSpec(
            f"{PREFIX}.view.r{r}",
            world.world_id,
            UNIT,
            "prepared-response.six-matrix-baoab-equations",
            ("prepared-response.native-closure",),
            (CHART,),
            (
                NumericalCoordinateSpec(
                    f"{PREFIX}.dt.r{r}",
                    NumericalCoordinateKind.TIMESTEP,
                    D(".001") / r,
                    "native-langevin-time",
                    r - 1,
                ),
            ),
            "prepared-response.solver.baoab",
            "1.0.0",
            "complex128",
            "cpu",
            f"{PREFIX}.pcg64-runtime",
            RandomnessSemantics.GENERATIVE_PREPARATION,
            "prepared-response.native-observer",
            compute.envelope_id,
        )
        for r in (1, 2)
    )
    policy = AuthorityPolicy(
        f"{stem}.authority",
        "owner.empirical-lawhood",
        "operator.empirical-lawhood",
        (PREFIX,),
        frozenset((WorldKind.NUMERICAL_SIMULATOR,)),
        frozenset(
            (
                AuthorityAction.SIMULATION_EXECUTION,
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
            )
        ),
        frozenset((SourceAccessClass.NONE,)),
        (f"{PREFIX}.exact-route-gate", f"{PREFIX}.typed-authority-gate"),
        frozenset(),
        budget(stage),
        OutcomeAccess.EVALUATOR_REVEAL,
    )
    return SystemSpec(
        f"{stem}.system",
        "Ordinary-source preparation applicability screening",
        SystemBoundaryKind.CLOSED_REFERENCE,
        world,
        relation,
        (
            ClockSpec(
                CLOCK,
                "Reference tick is .001 native Langevin time",
                "reference-tick",
                FRAME,
                SamplingSemantics.REGULAR,
                HoldSemantics.NONE,
                ClockLabelSemantics.INSTANT,
                D(1),
                D(0),
            ),
        ),
        tuple(sorted(quantities, key=lambda q: q.quantity_id)),
        IndependentUnitSpec(
            UNIT,
            "One independent root; schedules, words, futures and views are nested",
            f"{PREFIX}.root-id",
        ),
        policy,
        (
            PortSpec(
                f"{stem}.preparation-port",
                quantities[2].quantity_id,
                CLOCK,
                PortDirection.INPUT,
                BalanceRole.NONE,
                AuthorityAction.SIMULATION_EXECUTION,
            ),
            PortSpec(
                f"{stem}.validity-port",
                quantities[3].quantity_id,
                CLOCK,
                PortDirection.OUTPUT,
                BalanceRole.OBSERVATION,
            ),
        ),
        computability_envelopes=(compute,),
        numerical_views=views,
    )

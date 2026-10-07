"""Outcome-blind development scientific bindings; execution stays on the public programme route."""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import (
    ClaimSpec,
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.experiments import (
    AssignmentKind,
    AssignmentSpec,
    ControlKind,
    ControlSpec,
    ExperimentSpec,
    PrecisionGoal,
    RevealBarrierSpec,
)
from empirical_lawhood.kernel.obligations import (
    ClosureSpec,
    ComputabilityEvidence,
    FalsifierKind,
    FalsifierSpec,
    ObligationStatus,
    ScientificObligations,
    StructuralConvergenceSpec,
    SupportSpec,
    UncertaintySpec,
    ValiditySpec,
)
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.systems import IndependentUnitSpec, RelationalIdentity, SystemSpec
from empirical_lawhood.kernel.time import AvailabilitySpec, CausalPhase, HorizonSpec, InformationCutoff
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import ResponseGeometryDevelopmentNativeConfig
from empirical_lawhood.adapters.methods.response_geometry_development.extension_bundle import DEVELOPMENT_CAPABILITIES, DEVELOPMENT_ROLES
from empirical_lawhood.adapters.methods.response_geometry_prospective.development_law import DEVELOPMENT_INVOCATION_QUANTITY, DEVELOPMENT_LATENT_QUANTITIES
from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_actions import DEVELOPMENT_CLOCK, DEVELOPMENT_EPISODE_FRAME, DEVELOPMENT_FORCE_FRAME, DEVELOPMENT_FORCE_UNIT

from .design import _system


DEVELOPMENT_PREFIX = "response-geometry-development"
# Sum of declared worker forecasts is 377,784 seconds. The resource-ledger envelope
# independently excludes elapsed time from execution-stop authority.
DEVELOPMENT_BUDGET = ResourceBudget(8, 32 * 1024**3, 0, 432000, 192 * 1024**3, 64 * 1024**3)
DEVELOPMENT_RECEIVER = f"{DEVELOPMENT_PREFIX}.quantity.receiver"
DEVELOPMENT_ACTION = "response-geometry.x-force-command"
DEVELOPMENT_UNIT = f"{DEVELOPMENT_PREFIX}.independent-stochastic-root"


def response_geometry_development_system() -> SystemSpec:
    """Reuse the qualified native medium; give development its own relation and law qualification scope."""
    source = _system()
    world = replace(
        source.world,
        world_id=f"{DEVELOPMENT_PREFIX}.world",
        maximum_evidence=EvidenceCeiling.LOCAL_LAW,
        available_outcome_access=source.world.available_outcome_access
        | {OutcomeAccess.DEVELOPMENT_VISIBLE},
    )
    clock = replace(source.clocks[0], clock_id=DEVELOPMENT_CLOCK, coordinate_frame=DEVELOPMENT_EPISODE_FRAME)
    compute = replace(
        source.computability_envelopes[0],
        envelope_id=f"{DEVELOPMENT_PREFIX}.computability",
        max_memory_bytes=DEVELOPMENT_BUDGET.memory_bytes,
        max_wall_time_seconds=DEVELOPMENT_BUDGET.wall_time_seconds,
        max_output_bytes=DEVELOPMENT_BUDGET.output_bytes,
    )
    quantities = []
    mapping = {}
    for quantity in source.quantities:
        role = quantity.quantity_id.rsplit(".", 1)[-1]
        quantity_id = DEVELOPMENT_ACTION if role == "action" else f"{DEVELOPMENT_PREFIX}.quantity.{role}"
        mapping[quantity.quantity_id] = quantity_id
        quantities.append(
            replace(
                quantity,
                quantity_id=quantity_id,
                clock_id=DEVELOPMENT_CLOCK,
                availability=replace(quantity.availability, clock_id=DEVELOPMENT_CLOCK),
                native_unit=DEVELOPMENT_FORCE_UNIT if role == "action" else quantity.native_unit,
                coordinate_frame=DEVELOPMENT_FORCE_FRAME
                if role in ("action", "receiver")
                else quantity.coordinate_frame,
                label="Declared short-pulse-response signed X force in the pre-parent mode"
                if role == "action"
                else quantity.label,
            )
        )
    available = AvailabilitySpec(
        DEVELOPMENT_CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0)
    )
    for index, quantity_id in enumerate(DEVELOPMENT_LATENT_QUANTITIES):
        quantities.append(
            QuantitySpec(
                quantity_id,
                f"Causal fitted invocation state coordinate {index}",
                QuantityKind.HISTORY,
                "native-mode-position"
                if index == 0
                else "native-mode-momentum"
                if index == 1
                else "latent-coordinate",
                "hilbert-schmidt-native"
                if index == 0
                else "native-hs-momentum"
                if index == 1
                else "1",
                DEVELOPMENT_FORCE_FRAME if index < 2 else f"{DEVELOPMENT_PREFIX}.frozen-feature-map",
                DEVELOPMENT_CLOCK,
                available,
                ResponseDirection.NOT_APPLICABLE,
            )
        )
    quantities.append(
        QuantitySpec(
            DEVELOPMENT_INVOCATION_QUANTITY,
            "Invocation offset from the pre-parent landmark",
            QuantityKind.HISTORY,
            "time",
            "reference-tick",
            DEVELOPMENT_EPISODE_FRAME,
            DEVELOPMENT_CLOCK,
            available,
            ResponseDirection.NOT_APPLICABLE,
        )
    )
    relation = RelationalIdentity(
        f"{DEVELOPMENT_PREFIX}.relation",
        (f"{DEVELOPMENT_PREFIX}.quantity.denominator",),
        tuple(
            sorted((f"{DEVELOPMENT_PREFIX}.quantity.history", *DEVELOPMENT_LATENT_QUANTITIES, DEVELOPMENT_INVOCATION_QUANTITY))
        ),
        False,
        (DEVELOPMENT_ACTION,),
        (DEVELOPMENT_RECEIVER,),
        HorizonSpec(f"{DEVELOPMENT_PREFIX}.horizon", DEVELOPMENT_CLOCK, Decimal(320), "reference-tick"),
    )
    policy = replace(
        source.authority_policy,
        policy_id=f"{DEVELOPMENT_PREFIX}.authority-policy",
        scope_ids=(DEVELOPMENT_PREFIX,),
        budget_ceiling=DEVELOPMENT_BUDGET,
        required_gate_ids=(f"{DEVELOPMENT_PREFIX}.exact-route-gate", f"{DEVELOPMENT_PREFIX}.typed-authority-gate"),
    )
    return replace(
        source,
        system_id=f"{DEVELOPMENT_PREFIX}.system",
        label="Finite stochastic response-law development",
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=tuple(sorted(quantities, key=lambda value: value.quantity_id)),
        independent_unit=IndependentUnitSpec(
            DEVELOPMENT_UNIT,
            "One independently seeded development root; parents, signs and numerical views remain nested",
            f"{DEVELOPMENT_PREFIX}.physical-unit-id",
        ),
        authority_policy=policy,
        ports=tuple(
            replace(
                port,
                port_id=f"{DEVELOPMENT_PREFIX}.{port.port_id.rsplit('.', 1)[-1]}",
                quantity_id=mapping[port.quantity_id],
                clock_id=DEVELOPMENT_CLOCK,
            )
            for port in source.ports
        ),
        computability_envelopes=(compute,),
        numerical_views=tuple(
            replace(
                view,
                world_id=world.world_id,
                physical_preparation_id=DEVELOPMENT_UNIT,
                computability_envelope_id=compute.envelope_id,
                observation_operator_id=f"{DEVELOPMENT_PREFIX}.native-observer",
            )
            for view in source.numerical_views[:2]
        ),
    )


def response_geometry_development_experiment(
    system: SystemSpec, source: ResponseGeometryDevelopmentNativeConfig, prior_unit_ids: tuple[str, ...]
) -> ExperimentSpec:
    """Declare local law/REP/ACQ development, retaining the unentered ACT boundary."""
    prefix = DEVELOPMENT_PREFIX
    cutoff = InformationCutoff(
        f"{prefix}.design-cutoff", DEVELOPMENT_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    development = tuple(
        sorted(
            set(prior_unit_ids) | {source.physical_unit_id(r) for r in source.roots if r.index < 32}
        )
    )
    validation = tuple(sorted(source.physical_unit_id(r) for r in source.roots if r.index >= 32))
    evaluator = DEVELOPMENT_CAPABILITIES[DEVELOPMENT_ROLES.index("evaluation")].capability_key
    required = ObligationStatus.REQUIRED
    obligations = ScientificObligations(
        f"{prefix}.obligations",
        SupportSpec(
            f"{prefix}.support",
            system.relation.relation_id,
            DEVELOPMENT_UNIT,
            128,
            2,
            cutoff.cutoff_id,
            (f"{prefix}.expected-short-pulse-response-chart",),
            (f"{prefix}.assembling", f"{prefix}.prepared"),
            (),
            required,
        ),
        ValiditySpec(
            f"{prefix}.validity",
            (f"{prefix}.fixed-roster",),
            (
                "finite-native-short-pulse-response-delivery",
                "frozen-fit-calibration-validation-cutoffs",
                "simulator-local-denominator",
            ),
            ("NUMERICAL_OR_OBSERVATION_FAILURE", "UNAVAILABLE_FIT_OR_VALIDATION"),
            required,
        ),
        UncertaintySpec(
            f"{prefix}.uncertainty",
            f"{prefix}.rootwise-model-conditional-calibration",
            DEVELOPMENT_UNIT,
            Decimal("0.90"),
            (DEVELOPMENT_RECEIVER,),
            ("FINITE_STRATIFIED_PANEL_NOT_DISTRIBUTION_FREE", "NO_ADAPTIVE_POLICY_GUARANTEE"),
            required,
        ),
        (
            FalsifierSpec(
                f"{prefix}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                evaluator,
                "Held validation bias, RMSE, whole-root coverage, uncertainty width or paired numerical disagreement refuses the local candidate; missing contrast strata stop only the consuming claim.",
                "No validation-driven refit, threshold change, seed top-up or automatic ACT promotion.",
                required,
            ),
        ),
        ClosureSpec(
            f"{prefix}.closure",
            (f"{prefix}.five-parent-three-sign-cells",),
            (f"{prefix}.complete-initial-affine-noise-forecast",),
            system.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{prefix}.convergence",
            ("paired-member-worst-case-root-reduction",),
            tuple(v.view_id for v in system.numerical_views),
            (),
            required,
        ),
        ComputabilityEvidence(
            f"{prefix}.compute-evidence",
            system.computability_envelopes[0].envelope_id,
            tuple(v.view_id for v in system.numerical_views),
            ReadinessStatus.READY,
            (),
            (f"{prefix}.bounded-reservation",),
        ),
    )
    claim = ClaimSpec(
        f"{prefix}.local-law-development",
        system.world.world_id,
        system.relation.relation_id,
        "A frozen causal representation can predict the finite short-pulse-response native response within calibrated local support; matched representation and offline acquisition contrasts remain separately eligible.",
        "Per-context held-root prediction loss, coverage, width, passive response and paired representation/acquisition contrasts over the declared nine invocation strata.",
        DEVELOPMENT_UNIT,
        EvidenceRung.LOCAL_LAW,
        EvidenceCeiling.LOCAL_LAW,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "The sole law owner adjudicates local validity. development can nominate protected work only; operational ACT requires separately qualified state/window support and fresh authority.",
        (
            "finite-nine-origin-support",
            "independent-root-only-inference",
            "model-conditional-uncertainty",
        ),
        numerical_view_ids=tuple(v.view_id for v in system.numerical_views),
    )
    return ExperimentSpec(
        prefix,
        system.system_id,
        system.world.world_id,
        system.relation,
        DEVELOPMENT_UNIT,
        (claim,),
        AssignmentSpec(
            f"{prefix}.assignment",
            AssignmentKind.SIMULATOR_INTERVENTION,
            DEVELOPMENT_UNIT,
            system.relation.action_quantity_ids,
            "128 fresh independently seeded roots; five parents and NEG/HOLD/POS fork within each root with common noise and paired numerical views. Root index fixes invocation offset, fit, calibration and validation role before source contact.",
            (f"{prefix}.fixed-roster",),
        ),
        (DEVELOPMENT_RECEIVER,),
        (
            ControlSpec(
                f"{prefix}.control",
                ControlKind.BASELINE_COMPARATOR,
                evaluator,
                (DEVELOPMENT_RECEIVER,),
                "Matched-information untyped, alternative-subspace and state/clock models, typed-plus-microstate diagnostic, passive no-change/frozen-Y/momentum baselines, native HOLD and both numerical views.",
            ),
        ),
        (
            PrecisionGoal(
                f"{prefix}.precision",
                f"{prefix}.normalized-endpoint-halfwidth",
                Decimal(2),
                "task-delta",
                128,
                "Complete the fixed panel without top-up; retain unavailable, negative, mixed and stopped contrasts.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{prefix}.reveal",
            development,
            f"{prefix}.held-validation-cohort",
            sha256(canonical_json_bytes(validation)).hexdigest(),
            tuple(f"{prefix}.assess.{c}.report" for c in ("assembling", "prepared")),
            OutcomeAccess.EVALUATION_SEALED,
            True,
        ),
        obligations,
        VisibilityCeiling.OUTCOME_VISIBLE,
        VisibilityCeiling.PROSPECTIVE,
        system.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )

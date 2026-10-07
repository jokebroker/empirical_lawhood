"""Scientific source-completion scope for the eligible FLH native tranche."""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256

from empirical_lawhood.kernel.authority import (
    AuthorityAction,
    AuthorityPolicy,
    ResourceBudget,
    SourceAccessClass,
)
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
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.kernel.status import ReadinessStatus
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
    InformationCutoff,
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
from empirical_lawhood.adapters.methods.finite_response_law.science import PROGRAMME
from empirical_lawhood.adapters.simulators.finite_response_law.protocol import native_capabilities
from empirical_lawhood.adapters.simulators.finite_response_law.contracts import FiniteResponseLawNativeConfig
from empirical_lawhood.adapters.simulators.finite_response_law.roster import Q_CLOCK, Q_FRAME, Q_RECEIVERS
from empirical_lawhood.adapters.simulators.prepared_response.contracts import CAMPAIGN, prepared_numerical_view

UNIT = f"{PROGRAMME}.independent-stochastic-root"
BUDGET = ResourceBudget(8, 16 * 1024**3, 0, 86400, 100 * 1024**3, 2 * 1024**3)
PREPARATION_POLICY_ORIGINAL_AGGREGATE_WALL_SECONDS = 354_675_600


def native_budget(source: FiniteResponseLawNativeConfig) -> ResourceBudget:
    """Bind storage to the declared census, preserving the scientific time cap."""
    if source.stage == "preparation-screening":
        # Preparation-policy development uses deadline-free execution. The required authoring duration
        # field is descriptive and has no admission or scientific control effect.
        return ResourceBudget(
            8,
            16 * 1024**3,
            0,
            PREPARATION_POLICY_ORIGINAL_AGGREGATE_WALL_SECONDS,
            256 * 1024**3,
            100 * 1024**3,
        )
    if source.stage == "prospective-evaluation":
        # Includes the six immutable compiled controller records per root and
        # their exact causal custody. This is a storage bound, not a reset of
        # the shared cumulative 24-hour scientific clock.
        return ResourceBudget(8, 16 * 1024**3, 0, 86400, 256 * 1024**3, 24 * 1024**3)
    if source.stage == "calibration":
        # 32 prefixes at 20 MiB; 32*(1 parent+18 futures) at 2.5 MiB;
        # 64 view reports at 1 MiB and one 8 MiB native completion report.
        return ResourceBudget(8, 16 * 1024**3, 0, 86400, 100 * 1024**3, 2232 * 1024**2)
    return BUDGET


def native_system(source: FiniteResponseLawNativeConfig) -> SystemSpec:
    pre = AvailabilitySpec(Q_CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0))
    observed = replace(
        pre, phase=CausalPhase.RECEIVER, outcome_access=OutcomeAccess.EVALUATION_SEALED
    )
    quantities = tuple(
        QuantitySpec(
            f"{PROGRAMME}.quantity.{key}",
            label,
            kind,
            dimension,
            unit,
            Q_FRAME,
            Q_CLOCK,
            available,
            direction,
        )
        for key, label, kind, dimension, unit, available, direction in (
            (
                "two-port-action",
                "A declared native parent schedule or signed axial pulse word",
                QuantityKind.ACTION,
                "native-word-identity",
                "native-word",
                replace(pre, phase=CausalPhase.ACTION_REQUESTED),
                ResponseDirection.NOT_APPLICABLE,
            ),
            (
                "denominator",
                "q=2 six-matrix BAOAB medium; PCG64 new futures and paired numerical views",
                QuantityKind.DENOMINATOR,
                "medium-identity",
                "identity",
                pre,
                ResponseDirection.NOT_APPLICABLE,
            ),
            (
                "history",
                "Causal native history; retained prepared-response lineage or excluded fresh canary; fixed pre-parent frame",
                QuantityKind.HISTORY,
                "native-history",
                "native-hs-phase-history",
                pre,
                ResponseDirection.NOT_APPLICABLE,
            ),
            (
                "receiver.m1",
                "Raw first handoff-relative displacement; signed pairing remains a separate reduction",
                QuantityKind.RECEIVER,
                "displacement",
                "hilbert-schmidt-native",
                observed,
                ResponseDirection.SIGNED_VECTOR,
            ),
            (
                "receiver.m2",
                "Raw second handoff-relative displacement; signed pairing remains a separate reduction",
                QuantityKind.RECEIVER,
                "displacement",
                "hilbert-schmidt-native",
                observed,
                ResponseDirection.SIGNED_VECTOR,
            ),
        )
    )
    relation = RelationalIdentity(
        f"{PROGRAMME}.relation",
        (quantities[1].quantity_id,),
        (quantities[2].quantity_id,),
        False,
        (quantities[0].quantity_id,),
        tuple(q.quantity_id for q in quantities[3:]),
        HorizonSpec(f"{PROGRAMME}.horizon", Q_CLOCK, Decimal(192), "reference-tick"),
    )
    world = WorldSpec(
        f"{PROGRAMME}.world",
        "Prepared six-matrix native measurement medium",
        WorldKind.NUMERICAL_SIMULATOR,
        ("BAOAB-unit-kinetic-mass", "q2-six-Hermitian-matrices", "two-frozen-preparent-HS-ports"),
        ("physical-material-realization",),
        (),
        EvidenceCeiling.MEASUREMENT,
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
        f"{PROGRAMME}.computability",
        (f"finite-{source.stage}-native-census",),
        # The finite prospective evaluation physical and control operations are represented. Their
        # required production-readiness proof is an issue gate, not an
        # unresolved physical effect or a declaration that proof already passed.
        () if source.stage == "prospective-evaluation" else ("complete-no-effect-production-proof-required",),
        ("bounded-native-phase-checkpoint",),
        ("bounded-native-phase-checkpoint",),
        8,
        16 * 1024**3,
        0,
        86400,
        native_budget(source).output_bytes if source.stage == "prospective-evaluation" else 2 * 1024**3,
        Decimal(69120),
        None,
    )
    views = tuple(
        NumericalViewSpec(
            prepared_numerical_view(r).view_id,
            world.world_id,
            UNIT,
            f"{CAMPAIGN}.six-matrix-baoab-equations",
            (f"{CAMPAIGN}.native-closure",),
            (f"{CAMPAIGN}.finite-assay-boundary",),
            (
                NumericalCoordinateSpec(
                    f"{PROGRAMME}.dt.r{r}",
                    NumericalCoordinateKind.TIMESTEP,
                    Decimal("0.001") / r,
                    "native-langevin-time",
                    r - 1,
                ),
            ),
            f"{CAMPAIGN}.solver.baoab",
            "1.0.0",
            "complex128",
            "cpu",
            f"{PROGRAMME}.native-runtime-pcg64",
            RandomnessSemantics.GENERATIVE_PREPARATION,
            f"{CAMPAIGN}.native-observer",
            compute.envelope_id,
        )
        for r in (1, 2)
    )
    policy = AuthorityPolicy(
        f"{PROGRAMME}.authority-policy",
        "owner.empirical-lawhood",
        "operator.empirical-lawhood",
        (PROGRAMME,),
        frozenset((WorldKind.NUMERICAL_SIMULATOR,)),
        frozenset(
            (
                AuthorityAction.SIMULATION_EXECUTION,
                AuthorityAction.EVALUATOR_REVEAL,
                AuthorityAction.NONACTUATING_PROSPECTIVE_FREEZE,
            )
        ),
        frozenset((SourceAccessClass.NONE,)),
        (f"{PROGRAMME}.exact-route-gate", f"{PROGRAMME}.typed-authority-gate"),
        frozenset(),
        native_budget(source),
        OutcomeAccess.EVALUATOR_REVEAL,
    )
    return SystemSpec(
        f"{PROGRAMME}.system",
        "Native finite-action acquisition and paired projection completion",
        SystemBoundaryKind.CLOSED_REFERENCE,
        world,
        relation,
        (
            ClockSpec(
                Q_CLOCK,
                "One reference tick is 0.001 native Langevin time",
                "reference-tick",
                Q_FRAME,
                SamplingSemantics.REGULAR,
                HoldSemantics.NONE,
                ClockLabelSemantics.INSTANT,
                Decimal(1),
                Decimal(0),
            ),
        ),
        tuple(sorted(quantities, key=lambda q: q.quantity_id)),
        IndependentUnitSpec(
            UNIT,
            "One independent root; retained parents, words, futures and views never increase n",
            f"{PROGRAMME}.root-id",
        ),
        policy,
        (
            PortSpec(
                f"{PROGRAMME}.native-action-port",
                quantities[0].quantity_id,
                Q_CLOCK,
                PortDirection.INPUT,
                BalanceRole.NONE,
                AuthorityAction.SIMULATION_EXECUTION,
            ),
            *(
                PortSpec(
                    receiver, q.quantity_id, Q_CLOCK, PortDirection.OUTPUT, BalanceRole.OBSERVATION
                )
                for receiver, q in zip(Q_RECEIVERS, quantities[3:], strict=True)
            ),
        ),
        computability_envelopes=(compute,),
        numerical_views=views,
    )


def qualification_system(source: FiniteResponseLawNativeConfig) -> SystemSpec:
    "Declare the paired local law question using the existing typed native system.\n\n    This specifies the requested evidence ceiling, not an achieved rung. The\n    source-completion system and its measurement adjudication remain separate records.\n    "
    from empirical_lawhood.adapters.methods.finite_response_law.law_binding import feature_quantities, output_quantities, parent_quantity

    if source.stage not in ("calibration", "prospective-evaluation"):
        raise ValueError("FLH paired law declarations require fresh calibration or evaluation")
    native = native_system(source)
    outputs = output_quantities()
    return replace(
        native,
        system_id=f"{PROGRAMME}.paired-law-system",
        label="Frozen native-action paired response law; qualification requested",
        world=replace(native.world, maximum_evidence=EvidenceCeiling.LOCAL_LAW),
        relation=replace(
            native.relation,
            relation_id=f"{PROGRAMME}.paired-law-relation",
            receiver_quantity_ids=tuple(sorted(q.quantity_id for q in outputs)),
        ),
        quantities=tuple(
            sorted(
                (
                    *native.quantities,
                    *feature_quantities("lower"),
                    *feature_quantities("direct"),
                    parent_quantity(),
                    *outputs,
                ),
                key=lambda q: q.quantity_id,
            )
        ),
        ports=tuple(
            sorted(
                (
                    *native.ports,
                    *(
                        PortSpec(
                            f"{q.quantity_id}.computed-port",
                            q.quantity_id,
                            q.clock_id,
                            PortDirection.OUTPUT,
                            BalanceRole.OBSERVATION,
                        )
                        for q in outputs
                    ),
                ),
                key=lambda p: p.port_id,
            )
        ),
    )


def evaluation_experiment(source: FiniteResponseLawNativeConfig, system: SystemSpec) -> ExperimentSpec:
    "Request the frozen independent evaluation prospective test; native acquisition remains measurement."
    from empirical_lawhood.adapters.simulators.finite_response_law.evaluation.discovery import REVEAL_CAPABILITY
    from empirical_lawhood.adapters.methods.finite_response_law.law_binding import output_quantities

    if source.stage != "prospective-evaluation":
        raise ValueError("Independent evaluation prospective claims cannot reinterpret a predecessor tranche")
    base = native_experiment(source, system)
    key = REVEAL_CAPABILITY.capability_key
    obligations = replace(
        base.obligations,
        uncertainty=replace(
            base.obligations.uncertainty,
            status=ObligationStatus.REQUIRED,
            confidence_level=Decimal(".95"),
            method_key='finite-response-law.prospective-evaluation.complete-root-cp-and-bootstrap',
            limitation_codes=(
                "paired-simulator-receiver-only",
                "synthetic-fixed-request-distribution",
            ),
        ),
        falsifiers=tuple(
            replace(
                f,
                capability_key=key,
                description="Fewer than 52 joint successes or more than two false-admission episodes fails use. Less than 10% response-loss reduction or nonnegative upper one-sided 95% paired-root bootstrap bound fails information. Missing assigned response operands make information unevaluable.",
                decisive_rule="Retain all 64 roots, refusals and cancellations. No refit, reselection, complete-case filtering, retuning or additional acquisitions.",
            )
            for f in base.obligations.falsifiers
        ),
    )
    claim = replace(
        base.claims[0],
        claim_id="finite-response-law.prospective-evaluation.prospective-use-and-information",
        proposition="Frozen pre-parent interface predictions support both finite-native consumers and improve all-menu response prediction beyond the frozen cached comparator.",
        estimand="64 fresh independent assigned roots: both consumers and both futures/views jointly; all-root signed-response MSE under the same primary-input predictions.",
        requested_rung=EvidenceRung.CONTROLLER_USE,
        evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
        promotion_rule="Use and information are separate noncompensating tests. Added use is tested only after use, information and cached qualification; preparation-policy eligibility requires lower-law qualification and use, not information.",
    )
    return replace(
        base,
        claims=(claim,),
        obligations=obligations,
        controls=tuple(replace(c, capability_key=key) for c in base.controls),
        precision_goals=tuple(
            PrecisionGoal(
                f"{base.experiment_id}.precision.{i}",
                q.quantity_id,
                source.science.delta[i],
                q.native_unit,
                64,
                "Frozen native-unit admission half-width plus numerical floor; no new qualification gate.",
            )
            for i, q in enumerate(output_quantities())
        ),
        reveal_barrier=replace(
            base.reveal_barrier,
            evaluation_cohort_id="finite-response-law.prospective-evaluation.frozen-prospective-census",
        ),
    )


def native_experiment(source: FiniteResponseLawNativeConfig, system: SystemSpec) -> ExperimentSpec:
    _, _, evaluation_capability = native_capabilities(source)
    prefix = f"{PROGRAMME}.{source.stage}"
    units = tuple(r.physical_unit_id for r in source.roots)
    receivers = system.relation.receiver_quantity_ids
    cutoff = InformationCutoff(
        f"{prefix}.design-cutoff", Q_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    required = ObligationStatus.REQUIRED
    obligations = ScientificObligations(
        f"{prefix}.obligations",
        SupportSpec(
            f"{prefix}.support",
            system.relation.relation_id,
            UNIT,
            len(units),
            2,
            cutoff.cutoff_id,
            (),
            (f"{prefix}.denominator",),
            (),
            required,
        ),
        ValiditySpec(
            f"{prefix}.validity",
            (f"{prefix}.fixed-roster",),
            ("finite-complete-native-delivery", "simulator-local-denominator"),
            ("NUMERICAL_OR_OBSERVATION_FAILURE",),
            required,
        ),
        # Inactive schema field 0.95 is not an asserted confidence interval.
        # This stage adjudicates a finite acquisition census. measurement oracle precision
        # is assessed separately on the complete joined two-future panel.
        UncertaintySpec(
            f"{prefix}.uncertainty",
            f"{prefix}.native-availability-only",
            UNIT,
            Decimal("0.95"),
            receivers,
            ("NO_INTERVAL_OR_POPULATION_INFERENCE_AT_NATIVE_COMPLETION",),
            ObligationStatus.NOT_APPLICABLE,
        ),
        (
            FalsifierSpec(
                f"{prefix}.falsifier",
                FalsifierKind.WRONG_ACTION,
                evaluation_capability.capability_key,
                "A requested native word, handoff, same-purpose HOLD, numerical view or +192 measurement is unavailable or differs from its declared contract.",
                "Retain every assigned path and its evaluability; no retry, retiming, replacement roots or retuning.",
                required,
            ),
        ),
        ClosureSpec(
            f"{prefix}.closure",
            (f"{prefix}.fixed-nine-word-chart",),
            (f"{prefix}.native-handoff-relative-displacement-and-paired-hold-preservation",),
            system.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{prefix}.convergence",
            ("retain-both-coupled-numerical-views-without-pooling",),
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
        f"{prefix}.native-completion-claim",
        system.world.world_id,
        system.relation.relation_id,
        "The complete declared finite native census is delivered and reduced to measured +192 displacement, same-purpose HOLD preservation and work, with exact predecessor and numerical-view accounting.",
        "Every assigned word and numerical view; retained roots keep their physical identities; no population or predictive claim.",
        UNIT,
        EvidenceRung.MEASUREMENT,
        EvidenceCeiling.MEASUREMENT,
        OutcomeAccess.EVALUATION_SEALED,
        # The declaration precedes this new acquisition. This does not turn
        # retained roots into fresh independent evidence: all roots remain in
        # the draft's development set and the claim is only measurement census.
        VisibilityCeiling.PROSPECTIVE,
        "Native acquisition/projection completion only. The two-future task gate, interface information and F(U) usefulness remain separate questions.",
        ("fixed-native-menu", "no-independent-unit-inflation"),
        numerical_view_ids=tuple(v.view_id for v in system.numerical_views),
    )
    return ExperimentSpec(
        prefix,
        system.system_id,
        system.world.world_id,
        system.relation,
        UNIT,
        (claim,),
        AssignmentSpec(
            f"{prefix}.assignment",
            AssignmentKind.SIMULATOR_INTERVENTION,
            UNIT,
            system.relation.action_quantity_ids,
            "64 fresh independent roots; three frozen qualified predictors and two independent-request consumers commit before each preassigned native parent; nine acquired native words, two future purposes and two coupled views. This acquisition ends sealed; separate prospective controller evaluation reveal and inference follow recovery."
            if source.stage == "prospective-evaluation"
            else "32 fresh independent roots, each assigned one uniformly random old parent before outcomes; nine native words, two independent future purposes and two coupled numerical views."
            if source.stage == "calibration"
            else "One excluded +272 canary with two futures, or exactly the missing second future from all 80 retained prepared-response handoffs; nine native words and two coupled views.",
            (f"{prefix}.fixed-roster",),
        ),
        receivers,
        (
            ControlSpec(
                f"{prefix}.control",
                ControlKind.BASELINE_COMPARATOR,
                evaluation_capability.capability_key,
                receivers,
                "Same-handoff, same-purpose native HOLD for each signed word; paired numerical views; signed odd receiver remains explicitly paired in downstream analysis.",
            ),
        ),
        (
            PrecisionGoal(
                f"{prefix}.precision",
                f"{prefix}.native-delivery-readout-completeness",
                Decimal("0.01"),
                "hilbert-schmidt-native",
                len(units),
                'Fixed full census; retain both views and all measured validity. This native completion act makes no interval precision claim. The joined measurement oracle retains the frozen benchmark tolerances.',
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{prefix}.reveal",
            units,
            f"{prefix}.frozen-fresh-calibration-census"
            if source.stage == "calibration"
            else f"{prefix}.development-only-native-census",
            sha256(canonical_json_bytes(units)).hexdigest(),
            (f"{prefix}.sealed-native-projection",),
            OutcomeAccess.EVALUATION_SEALED,
            True,
        ),
        obligations,
        VisibilityCeiling.OUTCOME_VISIBLE,
        VisibilityCeiling.PROSPECTIVE,
        system.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )

"""Exact native medium, nine observable coordinates and development estimand."""

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
from empirical_lawhood.kernel.quantities import QuantityKind, QuantitySpec, ResponseDirection
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.systems import IndependentUnitSpec, RelationalIdentity, SystemSpec
from empirical_lawhood.kernel.time import AvailabilitySpec, CausalPhase, HorizonSpec, InformationCutoff
from empirical_lawhood.adapters.composition.response_geometry_prospective.design import _system
from empirical_lawhood.adapters.methods.matrix_preparation.contracts import FEATURES
from empirical_lawhood.adapters.methods.matrix_preparation.law import FEATURE_QUANTITIES, FEATURE_UNITS
from empirical_lawhood.adapters.simulators.response_geometry_prospective.development_actions import DEVELOPMENT_CLOCK as NATIVE_REFERENCE_CLOCK, DEVELOPMENT_EPISODE_FRAME as NATIVE_EPISODE_FRAME, DEVELOPMENT_FORCE_FRAME as FROZEN_PREPARATION_MODE, DEVELOPMENT_FORCE_UNIT as NATIVE_FORCE_UNIT
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT, PreparationSourceConfig


DEVELOPMENT_BUDGET = ResourceBudget(8, 16 * 1024**3, 0, 2_000_000, 128 * 1024**3, 32 * 1024**3)
RECEIVER = f"{DEVELOPMENT}.quantity.receiver"
ACTION = "matrix-preparation-adequacy.development.x-force-command"
UNIT = f"{DEVELOPMENT}.independent-stochastic-root"


def preparation_system() -> SystemSpec:
    source = _system()
    world = replace(
        source.world,
        world_id=f"{DEVELOPMENT}.world",
        maximum_evidence=EvidenceCeiling.LOCAL_LAW,
        available_outcome_access=source.world.available_outcome_access
        | {OutcomeAccess.DEVELOPMENT_VISIBLE},
    )
    clock = replace(source.clocks[0], clock_id=NATIVE_REFERENCE_CLOCK, coordinate_frame=NATIVE_EPISODE_FRAME)
    compute = replace(
        source.computability_envelopes[0],
        envelope_id=f"{DEVELOPMENT}.computability",
        max_memory_bytes=DEVELOPMENT_BUDGET.memory_bytes,
        max_wall_time_seconds=DEVELOPMENT_BUDGET.wall_time_seconds,
        max_output_bytes=DEVELOPMENT_BUDGET.output_bytes,
    )
    quantities, mapping = [], {}
    for q in source.quantities:
        role = q.quantity_id.rsplit(".", 1)[-1]
        key = ACTION if role == "action" else f"{DEVELOPMENT}.quantity.{role}"
        mapping[q.quantity_id] = key
        quantities.append(
            replace(
                q,
                quantity_id=key,
                clock_id=NATIVE_REFERENCE_CLOCK,
                availability=replace(q.availability, clock_id=NATIVE_REFERENCE_CLOCK),
                native_unit=NATIVE_FORCE_UNIT if role == "action" else q.native_unit,
                coordinate_frame=FROZEN_PREPARATION_MODE
                if role in ("action", "receiver")
                else NATIVE_EPISODE_FRAME,
                label="Fixed 64-tick signed X force in the common preparent mode"
                if role == "action"
                else "Absolute X displacement from the observed handoff"
                if role == "receiver"
                else "Completed preparation and causal observation history"
                if role == "history"
                else q.label,
            )
        )
    available = AvailabilitySpec(
        NATIVE_REFERENCE_CLOCK, CausalPhase.PRE_ACTION, OutcomeAccess.OUTCOME_BLIND, Decimal(0)
    )
    for key, label, unit in zip(FEATURE_QUANTITIES, FEATURES, FEATURE_UNITS, strict=True):
        quantities.append(
            QuantitySpec(
                key,
                label,
                QuantityKind.HISTORY,
                unit,
                unit,
                FROZEN_PREPARATION_MODE
                if key.endswith(("receiver_position", "receiver_velocity"))
                else f"{DEVELOPMENT}.observable-chart",
                NATIVE_REFERENCE_CLOCK,
                available,
                ResponseDirection.NOT_APPLICABLE,
            )
        )
    relation = RelationalIdentity(
        f"{DEVELOPMENT}.relation",
        (f"{DEVELOPMENT}.quantity.denominator",),
        tuple(sorted((f"{DEVELOPMENT}.quantity.history", *FEATURE_QUANTITIES))),
        False,
        (ACTION,),
        (RECEIVER,),
        HorizonSpec(f"{DEVELOPMENT}.horizon", NATIVE_REFERENCE_CLOCK, Decimal(320), "reference-tick"),
    )
    policy = replace(
        source.authority_policy,
        policy_id=f"{DEVELOPMENT}.authority-policy",
        delegator_id="owner.empirical-lawhood",
        delegate_id="operator.empirical-lawhood",
        scope_ids=(DEVELOPMENT,),
        budget_ceiling=DEVELOPMENT_BUDGET,
        required_gate_ids=(
            f"{DEVELOPMENT}.exact-route-gate",
            f"{DEVELOPMENT}.typed-authority-gate",
        ),
    )
    return replace(
        source,
        system_id=f"{DEVELOPMENT}.system",
        label="Exposed-root preparation and observable response development",
        world=world,
        relation=relation,
        clocks=(clock,),
        quantities=tuple(sorted(quantities, key=lambda q: q.quantity_id)),
        independent_unit=IndependentUnitSpec(
            UNIT,
            "One original stochastic root; every parent, sign, conditional response and numerical view remains nested",
            f"{DEVELOPMENT}.original-physical-unit-id",
        ),
        authority_policy=policy,
        ports=tuple(
            replace(
                p,
                port_id=f"{DEVELOPMENT}.{p.port_id.rsplit('.', 1)[-1]}",
                quantity_id=mapping[p.quantity_id],
                clock_id=NATIVE_REFERENCE_CLOCK,
            )
            for p in source.ports
        ),
        computability_envelopes=(compute,),
        numerical_views=tuple(
            replace(
                v,
                world_id=world.world_id,
                physical_preparation_id=UNIT,
                computability_envelope_id=compute.envelope_id,
                observation_operator_id=f"{DEVELOPMENT}.nine-observable-projection",
            )
            for v in source.numerical_views[:2]
        ),
    )


def preparation_experiment(
    system: SystemSpec, source: PreparationSourceConfig, evaluator: str
) -> ExperimentSpec:
    required, prefix = ObligationStatus.REQUIRED, DEVELOPMENT
    cutoff = InformationCutoff(
        f"{prefix}.specification-cutoff", NATIVE_REFERENCE_CLOCK, CausalPhase.PRE_ACTION, Decimal(0)
    )
    units = tuple(sorted(r.physical_unit_id for r in source.roots))
    obligations = ScientificObligations(
        f"{prefix}.obligations",
        SupportSpec(
            f"{prefix}.support",
            system.relation.relation_id,
            UNIT,
            128,
            2,
            cutoff.cutoff_id,
            (f"{prefix}.expected-force-chart",),
            (f"{prefix}.assembling", f"{prefix}.prepared"),
            (),
            required,
        ),
        ValiditySpec(
            f"{prefix}.validity",
            (f"{prefix}.fixed-exposed-root-roster",),
            (
                "causal-nine-observations-only",
                "independent-response-and-prospective-task-innovations",
                "native-three-sign-five-readout-chart",
            ),
            ("NATIVE_PRESERVATION_OR_CONTACT_FAILED", "NUMERICAL_OR_OBSERVATION_UNRESOLVED"),
            required,
        ),
        UncertaintySpec(
            f"{prefix}.uncertainty",
            "whole-root-full-menu-crossfit-development",
            UNIT,
            Decimal(".95"),
            (RECEIVER,),
            ("EXPOSED_DEVELOPMENT_ONLY", "NO_PROTECTED_OR_ADMITTED_CONDITIONAL_COVERAGE_GUARANTEE"),
            required,
        ),
        (
            FalsifierSpec(
                f"{prefix}.falsifier",
                FalsifierKind.WRONG_ACTION,
                evaluator,
                "Wrong-sign, absolute/odd error, view disagreement and width/contact checks retain their independent failures.",
                "No outcome-driven target, threshold, menu, horizon or model expansion.",
                required,
            ),
        ),
        ClosureSpec(
            f"{prefix}.closure",
            (f"{prefix}.five-parent-three-sign-five-readout-grid",),
            ("finite-action-predictor-not-an-input-recurrence",),
            system.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{prefix}.convergence",
            ("full-discrete-native-tangent-and-secant", "separate-primary-half-view-errors"),
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
            (f"{prefix}.bounded-root-bundles",),
        ),
    )
    claim = ClaimSpec(
        f"{prefix}.description-and-preparation-feasibility",
        system.world.world_id,
        system.relation.relation_id,
        "New conditional continuations of all exposed original stochastic roots can nominate an adequate observable response description and separately test preparation/window/task feasibility.",
        "Per-context fixed held-screen errors, A and U, event forecasts, and qualified task contrasts; fresh physical-root confirmation is a conditional follow-up.",
        UNIT,
        EvidenceRung.LOCAL_LAW,
        EvidenceCeiling.LOCAL_LAW,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "This specification precedes its new continuation outcomes. All original roots remain exposed development; no protected scientific support or preparation or prospective admission follows this panel.",
        (
            "all-old-roots-exposed",
            "conditional-new-innovations-not-new-independent-units",
            "frozen-target-family-and-model-library",
        ),
        numerical_view_ids=tuple(v.view_id for v in system.numerical_views),
    )
    # Fresh here binds only as-yet-unseen continuation outcomes. The enclosing
    # draft has NO fresh evaluation physical units and retains all old exposure.
    continuation_cohort = tuple(
        f"{r.root_id}.conditional-response-and-prospective-task-futures" for r in source.roots
    )
    barrier = RevealBarrierSpec(
        f"{prefix}.reveal",
        units,
        f"{prefix}.new-continuation-outcomes-on-exposed-roots",
        sha256(canonical_json_bytes(continuation_cohort)).hexdigest(),
        tuple(f"{prefix}.task-assessment.{c}.report" for c in ("assembling", "prepared")),
        OutcomeAccess.EVALUATION_SEALED,
        True,
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
            (ACTION,),
            "Restore all 128 exposed stochastic prefixes. Cross five preparations, one common response bundle, three independent response bundles and one independent prospective task bundle with both numerical views; none creates a new physical root.",
            (f"{prefix}.fixed-exposed-root-roster",),
        ),
        (RECEIVER,),
        (
            ControlSpec(
                f"{prefix}.controls",
                ControlKind.BASELINE_COMPARATOR,
                evaluator,
                (RECEIVER,),
                "Native HOLD, development adequacy-best and task-best fixed parents, strongest qualified observable controller, and separately privileged calibrated gain controls.",
            ),
        ),
        (
            PrecisionGoal(
                f"{prefix}.development-precision",
                "receiver-absolute-and-odd",
                Decimal(".125"),
                "hilbert-schmidt-native",
                128,
                "Complete all assigned exposed roots; a failed development gate leaves fresh successors unentered.",
            ),
        ),
        (cutoff,),
        barrier,
        obligations,
        VisibilityCeiling.OUTCOME_VISIBLE,
        VisibilityCeiling.PROSPECTIVE,
        system.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )

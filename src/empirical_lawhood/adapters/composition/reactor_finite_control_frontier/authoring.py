"""Outcome-blind phase authoring through the existing strict public entry records."""

from __future__ import annotations

from empirical_lawhood.adapters.composition.phase_authoring import PhaseAuthoringBundle as FiniteControlAuthoringBundle, PhaseContextProvider as FiniteControlContextProvider, finish_simulator_authoring

__all__ = [
    'FiniteControlAuthoringBundle',
    'FiniteControlContextProvider',
    "build_authoring",
    "budget",
    "experiment_spec",
]
from decimal import Decimal as D
from hashlib import sha256

from empirical_lawhood.adapters.composition.experiment_authoring import (
    single_experiment_campaign,
)
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.science import CLOCK, CUTOFF
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.config import ROOTS
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.phase import FrontierPhase

from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.executable_binding import BINDING as METHOD_BINDING
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.extension_bundle import CAPABILITY as METHOD_CAPABILITY
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.records import FrontierAssay
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.science import CHART, RECEIVERS, UNIT, frontier_system
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.config import FrontierNativeConfig
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.executable_binding import BINDING as SOURCE_BINDING
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.extension_bundle import CAPABILITY as SOURCE_CAPABILITY
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
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.time import CausalPhase, InformationCutoff
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, DesignInputRole, DesignOrigin, DesignOriginKind, MaterializationQualificationReceipt, StudyDraft, StudyDraftLifecycle, SourceAccessDisposition, SourceMaterializationRef, SourceMaterializationRole
from empirical_lawhood.runtime.capabilities import CapabilityRegistry


from .protocol import study_template


def budget(phase: FrontierPhase) -> ResourceBudget:
    # Protocol budgets reserve a sum of task seconds; wall/CPU caps are enforced separately.
    return ResourceBudget(4, 32 * 1024**3, 0, phase.cpu_seconds, 512 * 1024**3, 128 * 1024**3)


def experiment_spec(phase: FrontierPhase) -> ExperimentSpec:
    system, stem = frontier_system(), phase.config_id
    views = tuple(v.view_id for v in system.numerical_views)
    cutoff = InformationCutoff(CUTOFF, CLOCK, CausalPhase.PRE_ACTION, D(0))
    required, support = ObligationStatus.REQUIRED, f"{stem}.causal-chart-support"
    ceiling = {
        "B": EvidenceCeiling.RESPONSE,
        "C": EvidenceCeiling.LOCAL_LAW,
        "D": EvidenceCeiling.CONTROLLER_USE,
    }[phase.phase]
    rung = EvidenceRung(ceiling.value)
    obligations = ScientificObligations(
        f"{stem}.obligations",
        SupportSpec(
            f"{stem}.support",
            system.relation.relation_id,
            UNIT,
            len(phase.roots),
            2,
            CUTOFF,
            (CHART,),
            (support,),
            (),
            required,
        ),
        ValiditySpec(
            f"{stem}.validity",
            (support,),
            (
                "causal-prefix-only",
                "paired-numerical-views",
                "requested-accepted-applied-realized-macro",
            ),
            ("INVALID_NATIVE_DELIVERY", "MISSING_REQUIRED_EVIDENCE"),
            required,
        ),
        UncertaintySpec(
            f"{stem}.uncertainty",
            "finite-empirical-bounds-root-exact-intervals",
            UNIT,
            D(".95"),
            tuple(sorted(RECEIVERS)),
            ("NO_OUTSIDE_SUPPORT_PROMOTION",),
            required,
        ),
        (
            FalsifierSpec(
                f"{stem}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                METHOD_CAPABILITY.capability_key,
                "Assigned opportunities retain contact, delivery, response, numerical and safety failures independently.",
                "B nominations only; C 95/96 joint roots at alpha=.05/72 with no unsafe roots; D 56/64 joined uses, zero false admissions, alpha=.05/24; four separate paired comparator contrasts.",
                required,
            ),
        ),
        ClosureSpec(
            f"{stem}.closure",
            (support,),
            ("native-integrator-timestep",),
            system.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{stem}.convergence",
            ("paired-peak-response-window-and-distinct-120s-guard",),
            views,
            (),
            required,
        ),
        ComputabilityEvidence(
            f"{stem}.computability",
            system.computability_envelopes[0].envelope_id,
            views,
            ReadinessStatus.READY,
            (),
            (f"{stem}.finite-native-chart",),
        ),
    )
    claim = ClaimSpec(
        f"{stem}.claim",
        system.world.world_id,
        system.relation.relation_id,
        {
            "B": "Measure a bounded development response frontier without empirical qualification.",
            "C": "Independently qualify nominated local pulse/window relations on fresh assigned roots.",
            "D": "Use qualified local relations prospectively and compare safe service over the fixed finite request grid.",
        }[phase.phase],
        "Four exploration-unshifted causal contexts, nine pulses, 10/30/60-second response windows and separate 120-second guard; independent local branches.",
        UNIT,
        rung,
        ceiling,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "No whole-batch completion, new-regime, latent-coefficient or equal-total-cost superiority claim.",
        ("development-frozen-bounds", "fixed-root-rosters", "independent-coordinate-admission"),
        numerical_view_ids=views,
    )
    development = tuple(r for r, role, _, _ in ROOTS if role == "development")
    return ExperimentSpec(
        stem,
        system.system_id,
        system.world.world_id,
        system.relation,
        UNIT,
        (claim,),
        AssignmentSpec(
            f"{stem}.assignment",
            AssignmentKind.SIMULATOR_INTERVENTION,
            UNIT,
            system.relation.action_quantity_ids,
            f"Phase {phase.phase}: {len(phase.roots)} assigned physical roots, no replacements; retained development roots never become fresh confirmation.",
            (support,),
        ),
        tuple(sorted(RECEIVERS)),
        (
            ControlSpec(
                f"{stem}.control",
                ControlKind.BASELINE_COMPARATOR,
                METHOD_CAPABILITY.capability_key,
                tuple(sorted(RECEIVERS)),
                "Matched zero-feed branch, fixed jacket and same prepared history; four frozen control comparators.",
            ),
        ),
        (
            PrecisionGoal(
                f"{stem}.precision",
                "all-assigned-finite-coordinate-service",
                D(".90") if phase.phase != "D" else D(".70"),
                "1",
                len(phase.roots),
                "Exact family tails, fixed root denominator, numerical views never increase n.",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{stem}.reveal",
            development,
            f"{stem}.fixed-assignment",
            sha256(canonical_json_bytes(phase.roots)).hexdigest(),
            tuple(
                f"artifact.frontier.{('action' if phase.phase == 'D' else 'assay')}.{r}.assay"
                for r in phase.roots
            ),
            OutcomeAccess.EVALUATION_SEALED,
            True,
        ),
        obligations,
        VisibilityCeiling.DEVELOPMENT_ONLY,
        VisibilityCeiling.PROSPECTIVE,
        system.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )


def build_authoring(
    *,
    phase: FrontierPhase,
    source: EmpiricalStudySource,
    native: FrontierNativeConfig,
    implementation_sha256: str,
    specifications: tuple[tuple[str, str], ...],
) -> FiniteControlAuthoringBundle:
    from empirical_lawhood.adapters.composition.specification_inputs import require_specifications

    require_specifications(specifications, count=1)
    if source.batch.fingerprint() != native.source_sha256 or phase.design != native.design:
        raise ValueError("frontier authoring changed native source or design")
    stem, system = phase.config_id, frontier_system()
    experiment = experiment_spec(phase)
    campaign = single_experiment_campaign(
        system,
        experiment,
        prefix=stem,
        budget=budget(phase),
        objective=experiment.claims[0].proposition,
        actions=(
            ("reveal", AuthorityAction.EVALUATOR_REVEAL),
            ("simulation", AuthorityAction.SIMULATION_EXECUTION),
        ),
    )
    caps = tuple(sorted((METHOD_CAPABILITY, SOURCE_CAPABILITY), key=lambda c: c.registry_id))
    registry = CapabilityRegistry(f"{stem}.registry", caps)
    template = study_template(phase, native, source, experiment, registry)
    qualifications, sources = [], []
    design_identity = ObjectIdentity.from_record(phase.config_id, phase)
    inputs = [
        DesignInputRecord(
            f"{stem}.design",
            design_identity,
            phase.fingerprint(),
            experiment.information_cutoffs[0],
            DesignInputRole.MOTIVATION,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            "owner.empirical-lawhood",
        )
    ]
    for i, (_, digest) in enumerate(specifications):
        identity = ObjectIdentity(
            f"{stem}.specification-{i}",
            'empirical-lawhood/document/scientific-specification',
            "1.0.0",
            digest,
        )
        inputs.append(
            DesignInputRecord(
                identity.object_id,
                identity,
                digest,
                experiment.information_cutoffs[0],
                DesignInputRole.MOTIVATION,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                "owner.empirical-lawhood",
            )
        )

    def qualify_source(
        source_id: str,
        identity: ObjectIdentity,
        config_sha: str,
        observer: ObjectIdentity,
        access: OutcomeAccess,
    ) -> None:
        q = MaterializationQualificationReceipt(
            f"{source_id}.configuration-qualification",
            source_id,
            identity,
            identity.object_fingerprint,
            system.world.world_id,
            observer,
            tuple(v.view_id for v in system.numerical_views),
            tuple(sorted({v.native_unit for v in system.quantities})),
            tuple(sorted({v.coordinate_frame for v in system.quantities})),
            (CLOCK,),
            system.relation.relation_id,
            experiment.obligations.validity.validity_id,
            experiment.obligations.uncertainty.uncertainty_id,
            SourceAccessDisposition.VERIFIED_ACCESS,
            access,
            VisibilityCeiling.PROSPECTIVE,
        )
        qualifications.append(q)
        sources.append(
            SourceMaterializationRef(
                source_id,
                SourceMaterializationRole.NUMERICAL_CONFIGURATION,
                system.world.world_id,
                identity,
                identity.object_fingerprint,
                config_sha,
                observer,
                q.numerical_view_ids,
                ObjectIdentity.from_record(q.receipt_id, q),
                SourceAccessDisposition.VERIFIED_ACCESS,
            )
        )

    qualify_source(
        f"{stem}.source-bundle",
        ObjectIdentity.from_record(f"{stem}.source-bundle", source),
        native.fingerprint(),
        ObjectIdentity.from_record(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY),
        OutcomeAccess.OUTCOME_BLIND,
    )
    for parent in phase.upstream:
        a = parent.artifact
        identity = ObjectIdentity(a.artifact_id, a.payload_schema, "1.0.0", a.sha256)
        # Exposure is declared separately from immutable model/evidence custody, as in the regime response design.
        units = (
            (parent.key,)
            if phase.phase == "B"
            else tuple(
                r
                for r, role, _, _ in ROOTS
                if role == ("qualification" if parent.key == "laws" else "development")
            )
        )
        seeds = tuple(sorted(f"seed.{s}" for r, _, _, s in ROOTS if r in units))
        inputs.append(
            DesignInputRecord(
                f"{stem}.prior-{parent.key}",
                identity,
                a.sha256,
                experiment.information_cutoffs[0],
                DesignInputRole.DEVELOPMENT_TUNING,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                "owner.empirical-lawhood",
                units,
                seeds,
            )
        )
        qualify_source(
            a.artifact_id,
            identity,
            phase.fingerprint(),
            ObjectIdentity.from_record(METHOD_CAPABILITY.capability_key, METHOD_CAPABILITY),
            OutcomeAccess.EVALUATION_SEALED,
        )
    prior_roles = (
        ("development",) if phase.phase in ("B", "C") else ("development", "qualification")
    )
    prior_units = tuple(sorted(r for r, role, _, _ in ROOTS if role in prior_roles))
    prior_seeds = tuple(sorted(f"seed.{s}" for _, role, _, s in ROOTS if role in prior_roles))
    eval_units = () if phase.phase == "B" else phase.roots
    eval_seeds = (
        ()
        if phase.phase == "B"
        else tuple(sorted(f"seed.{s}" for r, _, _, s in ROOTS if r in eval_units))
    )
    inputs = sorted(inputs, key=lambda i: i.input_id)
    draft = StudyDraft(
        f"{stem}.draft",
        StudyDraftLifecycle.DRAFT,
        experiment.claims[0].proposition,
        (f"{stem}.adequate", f"{stem}.inadequate"),
        DesignOrigin(
            f"{stem}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            tuple(i.input_id for i in inputs),
            VisibilityCeiling.PROSPECTIVE,
        ),
        tuple(inputs),
        prior_units,
        eval_units,
        prior_seeds,
        eval_seeds,
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(c.capability_key, c.capability_version, c.implementation_sha256)
            for c in caps
        ),
        tuple(sorted(sources, key=lambda s: s.source_id)),
        budget(phase),
    )
    payloads: tuple[FrontierPhase | FrontierNativeConfig, ...] = tuple(
        sorted((phase, native), key=lambda v: v.SCHEMA)
    )
    return finish_simulator_authoring(
        draft=draft,
        registry=registry,
        template=template,
        qualifications=tuple(qualifications),
        inputs=tuple(inputs),
        implementation_sha256=implementation_sha256,
        design_identity=design_identity,
        evaluator=METHOD_CAPABILITY,
        input_schema=FrontierAssay.SCHEMA,
        evidence_units=phase.roots,
        payloads=payloads,
        config_ids=tuple(v.config_id for v in payloads),
        executable_bindings=(METHOD_BINDING, SOURCE_BINDING),
        operand_description="Finite frontier {domain} operands with independent per-coordinate admission.",
        estimator="fixed-extrema-envelope-finite-frontier",
        uncertainty="root-exact-family-bounds-and-paired-bootstrap",
    )

"""Strict prospective authoring; outcome hashes bind only completed predecessors."""

from decimal import Decimal as D
from hashlib import sha256

from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign
from empirical_lawhood.adapters.composition.phase_authoring import (
    PhaseAuthoringBundle,
    PhaseContextProvider,
    finish_simulator_authoring,
)
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import CALIBRATION_ROOTS, DEVELOPMENT_ROOTS, ROOTS, STAGES, ClassicalStage, assignment, roots
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.executable_binding import BINDING as METHOD_BINDING
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.extension_bundle import CAPABILITY as METHOD
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.records import ClassicalAssay
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.science import CLOCK, CUTOFF, UNIT, stage_system
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.words import CHART
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.config import ClassicalNativeConfig
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.executable_binding import BINDING as SOURCE_BINDING
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.extension_bundle import CAPABILITY as SOURCE
from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
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

__all__ = [
    "PhaseContextProvider",
    "build_authoring",
    "budget",
    "experiment_spec",
]
# Stage outputs reserve 86 GiB, retaining 10 GiB for shared control/scratch
# within the complete 96 GiB new-artifact allocation.
OUTPUT_GIB = (1, 6, 17, 3, 6, 17, 3, 6, 17)


def budget(stage: ClassicalStage) -> ResourceBudget:
    return ResourceBudget(
        4,
        32 * 1024**3,
        0,
        stage.cpu_seconds,
        512 * 1024**3,
        OUTPUT_GIB[STAGES.index(stage.stage)] * 1024**3,
    )


def experiment_spec(stage: ClassicalStage) -> ExperimentSpec:
    system, stem = stage_system(stage), stage.config_id
    required = ObligationStatus.REQUIRED
    support = f"{stem}.declared-native-chart"
    views = tuple(v.view_id for v in system.numerical_views)
    receivers = tuple(sorted(system.relation.receiver_quantity_ids))
    cutoff = InformationCutoff(CUTOFF, CLOCK, CausalPhase.PRE_ACTION, D(0))
    ceiling = {
        "NOMINATION": EvidenceCeiling.RESPONSE,
        "QUALIFICATION": EvidenceCeiling.LOCAL_LAW,
        "PROSPECTIVE": EvidenceCeiling.CONTROLLER_USE,
    }[stage.role]
    scope = (
        "Two retained exploration-unshifted causal contexts; 10/30-second cooling and 120-second guard; separate assigned owner uses."
        if stage.block != "staged-sequence-comparison"
        else "Early exploration-unshifted first pulse; actual induced history at t0+120; conditional second cooling and 240-second joint peak, dose and safety; known second refusals stop the actual branch."
    )
    falsifier = "law qualification: 95/96 whole-root joint passes, exact lower bound .90, family 72/36/5, zero unsafe; prospective controller evaluation: 56/64 whole-root service, exact lower .70, zero own-law false admission and upper .10, family 8/4/6. Missing mandatory evidence is UNEVALUABLE. Comparison endpoints retain fixed roots, refusals, multiplicity and undefined cost resamples."
    obligations = ScientificObligations(
        f"{stem}.obligations",
        SupportSpec(
            f"{stem}.support",
            system.relation.relation_id,
            UNIT,
            len(stage.root_ids),
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
            tuple(
                sorted(
                    (
                        "causal-prefix-only",
                        "paired-same-tape-views",
                        "requested-accepted-applied-realized-words",
                        "raw-numerics-before-receiver-map",
                        "actual-predecessor-and-refusal-prefix",
                    )
                )
            ),
            ("INVALID_NATIVE_DELIVERY", "MISSING_REQUIRED_EVIDENCE"),
            required,
        ),
        UncertaintySpec(
            f"{stem}.uncertainty",
            "fixed-extrema-thermal-and-root-exact-calibrated-families",
            UNIT,
            D(".95"),
            receivers,
            ("NO_OUTSIDE_SUPPORT_PROMOTION",),
            required,
        ),
        (
            FalsifierSpec(
                f"{stem}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                METHOD.capability_key,
                scope,
                falsifier,
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
            ("paired-raw-response-thermal-and-applied-mass",),
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
            (f"{stem}.bounded-native-chart",),
        ),
    )
    proposition = {
        "NOMINATION": "Nominate bounded empirical response relations and fixed treatment recipes from declared development evidence.",
        "QUALIFICATION": "Qualify each frozen bounded empirical response relation on its complete fresh assigned root census.",
        "PROSPECTIVE": "Prospectively use qualified response relations for fixed local service or an actual two-decision sequence; separately estimate method contribution.",
    }[stage.role]
    claim = ClaimSpec(
        f"{stem}.claim",
        system.world.world_id,
        system.relation.relation_id,
        proposition,
        scope,
        UNIT,
        EvidenceRung(ceiling.value),
        ceiling,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "Simulator-local control claims; no whole-batch benchmark completion or universal mechanistic identification claim.",
        (
            "complete-independent-root-census",
            "fixed-preregistered-recipes",
            "individual-coordinate-admission",
        ),
        numerical_view_ids=views,
    )
    development = tuple(sorted((*DEVELOPMENT_ROOTS, *CALIBRATION_ROOTS)))
    outcomes = (
        ("artifact.classical.calibration.calibration",)
        if stage.stage == "base-menu-comparison-NOMINATION"
        else tuple(
            f"artifact.classical.{'action' if stage.role == 'PROSPECTIVE' else 'assay'}.{r}.assay"
            for r in stage.root_ids
        )
    )
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
            f"{stage.stage}: fixed assigned roots without replacements; retained B/C units remain development exposure.",
            (support,),
        ),
        receivers,
        (
            ControlSpec(
                f"{stem}.control",
                ControlKind.BASELINE_COMPARATOR,
                METHOD.capability_key,
                receivers,
                "Matched zero-feed or A0 reference with same exploration-unshifted and native scenario; separately committed treatment/comparator branches.",
            ),
        ),
        (
            PrecisionGoal(
                f"{stem}.precision",
                "whole-root-service-and-own-law-failure",
                D(".70") if stage.role == "PROSPECTIVE" else D(".90"),
                "1",
                len(stage.root_ids),
                falsifier,
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{stem}.reveal",
            development,
            f"{stem}.fixed-assignment",
            sha256(canonical_json_bytes(stage.root_ids)).hexdigest(),
            outcomes,
            OutcomeAccess.EVALUATION_SEALED,
            True,
        ),
        obligations,
        VisibilityCeiling.DEVELOPMENT_ONLY,
        VisibilityCeiling.PROSPECTIVE,
        system.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )


def exposed_units(stage: ClassicalStage) -> tuple[str, ...]:
    prior = list((*DEVELOPMENT_ROOTS, *CALIBRATION_ROOTS))
    for root, block, role, _ in ROOTS:
        source_stage = f"{block}-{'QUALIFICATION' if role == 'qualification' else 'PROSPECTIVE'}"
        if STAGES.index(source_stage) < STAGES.index(stage.stage):
            prior.append(root)
    return tuple(sorted(prior))


def build_authoring(
    *,
    stage: ClassicalStage,
    source: EmpiricalStudySource,
    native: ClassicalNativeConfig,
    implementation_sha256: str,
    specifications: tuple[tuple[str, str], ...],
) -> PhaseAuthoringBundle:
    from empirical_lawhood.adapters.composition.specification_inputs import require_specifications

    require_specifications(specifications, count=1)
    if source.batch.fingerprint() != native.source_sha256 or stage.design != native.design:
        raise ValueError("authoring changes the exact native source or design")
    system, stem = stage_system(stage), stage.config_id
    experiment = experiment_spec(stage)
    campaign = single_experiment_campaign(
        system,
        experiment,
        prefix=stem,
        budget=budget(stage),
        objective=experiment.claims[0].proposition,
        actions=(
            ("reveal", AuthorityAction.EVALUATOR_REVEAL),
            ("simulation", AuthorityAction.SIMULATION_EXECUTION),
        ),
    )
    caps = tuple(sorted((METHOD, SOURCE), key=lambda c: c.registry_id))
    registry = CapabilityRegistry(f"{stem}.registry", caps)
    template = study_template(stage, native, source, experiment, registry)
    qualifications, materials = [], []
    identity = ObjectIdentity.from_record(stem, stage)
    inputs = [
        DesignInputRecord(
            f"{stem}.design",
            identity,
            stage.fingerprint(),
            experiment.information_cutoffs[0],
            DesignInputRole.MOTIVATION,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            "owner.empirical-lawhood",
        )
    ]
    for i, (_, digest) in enumerate(specifications):
        spec = ObjectIdentity(
            f"{stem}.specification-{i}",
            'empirical-lawhood/document/scientific-specification',
            "1.0.0",
            digest,
        )
        inputs.append(
            DesignInputRecord(
                spec.object_id,
                spec,
                digest,
                experiment.information_cutoffs[0],
                DesignInputRole.MOTIVATION,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
                "owner.empirical-lawhood",
            )
        )

    def qualify(
        source_id: str,
        parent: ObjectIdentity,
        config_sha: str,
        observer: ObjectIdentity,
        access: OutcomeAccess,
    ) -> None:
        q = MaterializationQualificationReceipt(
            f"{source_id}.configuration-qualification",
            source_id,
            parent,
            parent.object_fingerprint,
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
        materials.append(
            SourceMaterializationRef(
                source_id,
                SourceMaterializationRole.NUMERICAL_CONFIGURATION,
                system.world.world_id,
                parent,
                parent.object_fingerprint,
                config_sha,
                observer,
                q.numerical_view_ids,
                ObjectIdentity.from_record(q.receipt_id, q),
                SourceAccessDisposition.VERIFIED_ACCESS,
            )
        )

    qualify(
        f"{stem}.source-bundle",
        ObjectIdentity.from_record(f"{stem}.source-bundle", source),
        native.fingerprint(),
        ObjectIdentity.from_record(SOURCE.capability_key, SOURCE),
        OutcomeAccess.OUTCOME_BLIND,
    )
    for parent in stage.upstream:
        a = parent.artifact
        subject = ObjectIdentity(a.artifact_id, a.payload_schema, "1.0.0", a.sha256)
        units = (
            (CALIBRATION_ROOTS[int(parent.key.split("-")[1])],)
            if parent.key.startswith("calibration-")
            else (DEVELOPMENT_ROOTS[int(parent.key.split("-")[1])],)
            if parent.key.startswith("development-")
            else roots(stage.block, "qualification")
            if parent.key == "laws"
            else tuple(sorted((*DEVELOPMENT_ROOTS, *CALIBRATION_ROOTS)))
            if stage.block == "base-menu-comparison" and parent.key == "nomination"
            else DEVELOPMENT_ROOTS
        )
        seeds = tuple(sorted(f"seed.{assignment(r)[2]}" for r in units))
        inputs.append(
            DesignInputRecord(
                f"{stem}.prior-{parent.key}",
                subject,
                a.sha256,
                experiment.information_cutoffs[0],
                DesignInputRole.DEVELOPMENT_TUNING,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                "owner.empirical-lawhood",
                tuple(sorted(units)),
                seeds,
            )
        )
        qualify(
            a.artifact_id,
            subject,
            stage.fingerprint(),
            ObjectIdentity.from_record(METHOD.capability_key, METHOD),
            OutcomeAccess.EVALUATION_SEALED,
        )
    prior = exposed_units(stage)
    evaluation = () if stage.role == "NOMINATION" else stage.root_ids
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
        prior,
        evaluation,
        tuple(sorted(f"seed.{assignment(r)[2]}" for r in prior)),
        tuple(sorted(f"seed.{assignment(r)[2]}" for r in evaluation)),
        (),
        system,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(c.capability_key, c.capability_version, c.implementation_sha256)
            for c in caps
            if c.capability_key in {step.capability_key for step in template.protocol.steps}
        ),
        tuple(sorted(materials, key=lambda m: m.source_id)),
        budget(stage),
    )
    payloads: tuple[ClassicalStage | ClassicalNativeConfig, ...] = tuple(
        sorted((stage, native), key=lambda p: p.SCHEMA)
    )
    return finish_simulator_authoring(
        draft=draft,
        registry=registry,
        template=template,
        qualifications=tuple(qualifications),
        inputs=tuple(inputs),
        implementation_sha256=implementation_sha256,
        design_identity=identity,
        evaluator=METHOD,
        input_schema=ClassicalAssay.SCHEMA,
        evidence_units=stage.root_ids,
        payloads=payloads,
        config_ids=tuple(p.config_id for p in payloads),
        executable_bindings=(METHOD_BINDING, SOURCE_BINDING),
        operand_description="Bounded classical response {domain} with raw tests, fixed receiver maps and actual owner lineage.",
        estimator="fixed-empirical-extrema-and-declared-calibration",
        uncertainty="whole-independent-root-exact-families-and-fixed-bootstrap",
    )

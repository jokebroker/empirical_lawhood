"""Strict noncontact scientific authoring for the bounded D/E successor."""

from typing import Any


from decimal import Decimal as D
from hashlib import sha256
from empirical_lawhood.adapters.composition.phase_authoring import finish_simulator_authoring
from empirical_lawhood.adapters.composition.experiment_authoring import single_experiment_campaign
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.config import (
    ConstructedPreparationStage,
    ConstructedPreparationNativeConfig,
    ConstructedPreparationSource,
)
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.extension_bundle import CAPABILITY as METHOD
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.extension_bundle import CAPABILITY as SOURCE
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.executable_binding import (
    BINDING as METHOD_BINDING,
)
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.executable_binding import (
    BINDING as SOURCE_BINDING,
)
from empirical_lawhood.adapters.simulators.constructed_preparation_applicability.records import ConstructedPreparationPanel
from empirical_lawhood.kernel.authority import AuthorityAction
from empirical_lawhood.kernel.evidence import (
    ClaimSpec,
    EvidenceCeiling,
    EvidenceRung,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.experiments import (
    ExperimentSpec,
    AssignmentSpec,
    AssignmentKind,
    ControlSpec,
    ControlKind,
    PrecisionGoal,
    RevealBarrierSpec,
)
from empirical_lawhood.kernel.obligations import (
    ScientificObligations,
    SupportSpec,
    ValiditySpec,
    UncertaintySpec,
    FalsifierSpec,
    FalsifierKind,
    ClosureSpec,
    StructuralConvergenceSpec,
    ComputabilityEvidence,
    ObligationStatus,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import ReadinessStatus
from empirical_lawhood.kernel.time import InformationCutoff, CausalPhase
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from empirical_lawhood.planning.study_authoring import (
    StudyDraft,
    StudyDraftLifecycle,
    DesignInputRecord,
    DesignInputRole,
    DesignOrigin,
    DesignOriginKind,
    SourceMaterializationRef,
    SourceMaterializationRole,
    MaterializationQualificationReceipt,
    CapabilitySelection,
)
from empirical_lawhood.planning.study_authoring import SourceAccessDisposition
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.adapters.methods.constructed_preparation_applicability.qualification import require_qualified_constructor
from empirical_lawhood.adapters.methods.preparation_applicability.exposure import effective_seed_ids
from .science import system, budget, UNIT, CLOCK, CUTOFF, CHART, FRAME
from .protocol import study_template

SPECIFICATION_PATH = "docs/plans/empirical-lawhood.constructed-preparation-applicability.md"


def experiment_spec(stage: ConstructedPreparationStage) -> ExperimentSpec:
    s, stem = system(stage), stage.config_id
    required = ObligationStatus.REQUIRED
    support = f"{stem}.declared-native-chart"
    views = tuple(v.view_id for v in s.numerical_views)
    receivers = s.relation.receiver_quantity_ids
    cutoff = InformationCutoff(CUTOFF, CLOCK, CausalPhase.PRE_ACTION, D(4096))
    obligations = ScientificObligations(
        f"{stem}.obligations",
        SupportSpec(
            f"{stem}.support",
            s.relation.relation_id,
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
            ("causal-preparation-seal", "observed-handoff-lower-seal", "unchanged-full-F-payload"),
            ("INVALID_NATIVE_DELIVERY", "MISSING_REQUIRED_EVIDENCE"),
            required,
        ),
        UncertaintySpec(
            f"{stem}.uncertainty",
            "exact-paired-root-discordance-and-fixed-whole-root-bootstrap",
            UNIT,
            D(".95"),
            receivers,
            ("NESTED_REQUESTS_AND_VIEWS_NEVER_INCREASE_N",),
            required,
        ),
        (
            FalsifierSpec(
                f"{stem}.falsifier",
                FalsifierKind.STRUCTURAL_CONVERGENCE,
                METHOD.capability_key,
                "Constructed boundary geometry and fresh N versus waiting applicability",
                "Q requires complete observations, numerical agreement and intended two-view support crossing in at least 6/8 assigned units; no coverage accuracy gate. E supports positive full-validity N-minus-HOLD effect, exact one-sided p<=.05, positive lower .025 whole-root covered-service interval and zero N false admissions. All assigned units remain visible.",
                required,
            ),
        ),
        ClosureSpec(
            f"{stem}.closure",
            (support,),
            ("native-integrator-timestep",),
            s.relation.history_quantity_ids,
            required,
        ),
        StructuralConvergenceSpec(
            f"{stem}.convergence", ("raw-native-paired-receivers",), views, (), required
        ),
        ComputabilityEvidence(
            f"{stem}.computability-evidence",
            s.computability_envelopes[0].envelope_id,
            views,
            ReadinessStatus.READY,
            (),
            (f"{stem}.bounded-assigned-census",),
        ),
    )
    ceiling = EvidenceCeiling.RESPONSE
    claim = ClaimSpec(
        f"{stem}.claim",
        s.world.world_id,
        s.relation.relation_id,
        "Independent native preparations reproduce the declared two-view support boundary geometry."
        if stage.phase == "Q"
        else "Constructed-population N preparation increases fresh full validity and covered service of unchanged F over equal-duration native HOLD.",
        "One prepared six-matrix simulator, three preparations, same full lower chart; no constitutive or physical transport claim.",
        UNIT,
        EvidenceRung(ceiling.value),
        ceiling,
        OutcomeAccess.EVALUATION_SEALED,
        VisibilityCeiling.PROSPECTIVE,
        "Development nominates; fresh metric intervention comparison does not certify an upper interval controller or P5.",
        ("complete-independent-root-census", "no-outcome-built-features", "unchanged-lower-law"),
        numerical_view_ids=views,
    )
    development = stage.exposure.excluded_unit_ids


    return ExperimentSpec(
        stem,
        s.system_id,
        s.world.world_id,
        s.relation,
        UNIT,
        (claim,),
        AssignmentSpec(
            f"{stem}.assignment",
            AssignmentKind.SIMULATOR_INTERVENTION,
            UNIT,
            s.relation.action_quantity_ids,
            "Fixed root census, every preparation with paired innovations; no replacements",
            (support,),
        ),
        receivers,
        (
            ControlSpec(
                f"{stem}.control",
                ControlKind.BASELINE_COMPARATOR,
                METHOD.capability_key,
                receivers,
                "Equal-duration native HOLD, evidence-selected fixed N, and the alternative fixed P; no adaptive superiority claim",
            ),
        ),
        (
            PrecisionGoal(
                f"{stem}.precision",
                "root-full-validity-gain",
                D(".05"),
                "1",
                len(stage.root_ids),
                "32 confirmation roots resolve effects in increments of 1/32; five all-positive discordances are needed for p<.05; Q has eight independent constructor units",
            ),
        ),
        (cutoff,),
        RevealBarrierSpec(
            f"{stem}.reveal",
            development,
            f"{stem}.fixed-assignment",
            sha256(canonical_json_bytes(stage.root_ids)).hexdigest(),
            tuple(f"artifact.ap.assay.{r}.science" for r in stage.root_ids),
            OutcomeAccess.EVALUATION_SEALED,
            True,
        ),
        obligations,
        VisibilityCeiling.DEVELOPMENT_ONLY,
        VisibilityCeiling.PROSPECTIVE,
        s.authority_policy.policy_id,
        ReadinessStatus.AUTHORITY_REQUIRED,
    )


def build_authoring(
    stage: ConstructedPreparationStage, source: ConstructedPreparationSource, *, implementation_sha256: str, specification_sha256: str, qualified_report=None
) -> Any:
    if source.design != stage.design or source.fingerprint() != stage.source.object_fingerprint or source.implementation_plan_sha256 != specification_sha256:
        raise ValueError("authoring source/specification differs from its frozen current binding")
    if stage.phase == "E":
        if qualified_report is None or qualified_report.fingerprint() != stage.upstream[1].artifact.sha256:
            raise ValueError("E authoring requires the exact receipted current Q report")
        require_qualified_constructor(stage, qualified_report)
    s, stem = system(stage), stage.config_id
    native = ConstructedPreparationNativeConfig(stage)
    experiment = experiment_spec(stage)
    campaign = single_experiment_campaign(
        s,
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
    cutoff = experiment.information_cutoffs[0]
    stage_identity = ObjectIdentity.from_record(stem, stage)
    spec_identity = ObjectIdentity(
        f"{stem}.specification",
        "empirical-lawhood/document/scientific-specification",
        "1.0.0",
        specification_sha256,
    )
    inputs = [
        DesignInputRecord(
            f"{stem}.design",
            stage_identity,
            stage.fingerprint(),
            cutoff,
            DesignInputRole.MOTIVATION,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            "owner.empirical-lawhood",
        ),
        DesignInputRecord(
            spec_identity.object_id,
            spec_identity,
            specification_sha256,
            cutoff,
            DesignInputRole.MOTIVATION,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
            "owner.empirical-lawhood",
        ),
    ]
    qualifications, materials = [], []
    subjects = [
        (
            f"{stem}.source-bundle",
            ObjectIdentity.from_record(f"{stem}.source-bundle", source),
            native.fingerprint(),
            SOURCE,
        )
    ]
    for prior in stage.upstream:
        a = prior.artifact
        identity = ObjectIdentity(a.artifact_id, a.payload_schema, "1.0.0", a.sha256)
        subjects.append((a.artifact_id, identity, stage.fingerprint(), METHOD))
        inputs.append(
            DesignInputRecord(
                f"{stem}.prior-{prior.key}",
                identity,
                a.sha256,
                cutoff,
                DesignInputRole.DEVELOPMENT_TUNING,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                VisibilityCeiling.DEVELOPMENT_ONLY,
                "owner.empirical-lawhood",
            )
        )
    for source_id, identity, config_sha256, owner in subjects:
        q = MaterializationQualificationReceipt(
            f"{source_id}.configuration-qualification",
            source_id,
            identity,
            identity.object_fingerprint,
            s.world.world_id,
            ObjectIdentity.from_record(owner.capability_key, owner),
            tuple(v.view_id for v in s.numerical_views),
            tuple(sorted({v.native_unit for v in s.quantities})),
            (FRAME,),
            (CLOCK,),
            s.relation.relation_id,
            experiment.obligations.validity.validity_id,
            experiment.obligations.uncertainty.uncertainty_id,
            SourceAccessDisposition.VERIFIED_ACCESS,
            OutcomeAccess.OUTCOME_BLIND if owner is SOURCE or source_id == stage.upstream[0].artifact.artifact_id else OutcomeAccess.EVALUATION_SEALED,
            VisibilityCeiling.PROSPECTIVE,
        )
        qualifications.append(q)
        materials.append(
            SourceMaterializationRef(
                source_id,
                SourceMaterializationRole.NUMERICAL_CONFIGURATION,
                s.world.world_id,
                identity,
                identity.object_fingerprint,
                config_sha256,
                q.observation_operator,
                q.numerical_view_ids,
                ObjectIdentity.from_record(q.receipt_id, q),
                SourceAccessDisposition.VERIFIED_ACCESS,
            )
        )
    frozen_inputs = tuple(sorted(inputs, key=lambda i: i.input_id))
    prior_roots = stage.exposure.excluded_unit_ids

    draft = StudyDraft(
        f"{stem}.draft",
        StudyDraftLifecycle.DRAFT,
        experiment.claims[0].proposition,
        (f"{stem}.active-production", f"{stem}.waiting-explanation"),
        DesignOrigin(
            f"{stem}.origin",
            DesignOriginKind.ORDINARY_THEORY_PREDECLARED,
            tuple(i.input_id for i in frozen_inputs),
            VisibilityCeiling.PROSPECTIVE,
        ),
        frozen_inputs,
        prior_roots,
        stage.root_ids,
        (),
        effective_seed_ids(stage.allocation),
        (),
        s,
        experiment,
        campaign,
        template.template_key,
        tuple(
            CapabilitySelection(c.capability_key, c.capability_version, c.implementation_sha256)
            for c in caps
        ),
        tuple(sorted(materials, key=lambda m: m.source_id)),
        budget(stage),
    )
    payloads: Any = tuple(sorted((stage, native), key=lambda p: p.SCHEMA))
    return finish_simulator_authoring(
        draft=draft,
        registry=registry,
        template=template,
        qualifications=tuple(qualifications),
        inputs=frozen_inputs,
        implementation_sha256=implementation_sha256,
        design_identity=stage_identity,
        evaluator=METHOD,
        input_schema=ConstructedPreparationPanel.SCHEMA,
        evidence_units=stage.root_ids,
        payloads=payloads,
        config_ids=tuple(p.config_id for p in payloads),
        executable_bindings=(METHOD_BINDING, SOURCE_BINDING),
        operand_description="Full unchanged lower-law validity in the bounded prepared domain {domain}",
        estimator="paired-independent-root-discordance",
        uncertainty="exact-one-sided-binomial-and-whole-root-bootstrap",
    )

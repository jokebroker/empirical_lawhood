"""Bind acquired projection identities into the predeclared finite-action recipe."""

from decimal import Decimal

from empirical_lawhood.adapters.methods.backbone_finite_action.extension_bundle import (
    FINITE_ACTION_MANIFEST,
)
from empirical_lawhood.adapters.methods.contracts import (
    CandidateClaimTemplate,
    CandidateFamilyLedger,
    CandidateFamilyMember,
    CandidateRosterDisposition,
    FalsifierObligationTemplate,
    LawCandidateAxisBinding,
    LawCandidateAxisMap,
    LawObligationTemplate,
)
from empirical_lawhood.adapters.methods.finite_action_identification import FiniteActionCandidateScaffold, FiniteActionComparatorBinding, FiniteActionIdentificationConfig, FiniteActionResponseQuantity, IndependentUnitResamplingMethod
from empirical_lawhood.adapters.methods.finite_action_registration import (
    compose_finite_action_identification,
)
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadPublisher
from empirical_lawhood.adapters.methods.qualification_profiles import (
    MethodEquivalentQualificationProfileEvaluator,
)
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind
from empirical_lawhood.kernel.obligations import FalsifierKind
from empirical_lawhood.kernel.provenance import (
    EvidenceLink,
    EvidenceRelation,
    ObjectIdentity,
)
from empirical_lawhood.kernel.references import NamedDecimal, QuantityBound

from .chain import FiniteChainConfig
from .design import CHART, CLOCK, CUTOFF, HISTORY, METHOD, PREFIX, RECEIVER, SUPPORT, UNIT, reactor_system, unit_ids
from .projection import ReactorProjection
from .words import action_words


def bind_finite_chain(
    bundle: ReactorProjection, publisher: CandidatePayloadPublisher
) -> FiniteChainConfig:
    """Binding reads identities only; all scientific settings come from design."""
    if bundle.extension is None:
        raise ValueError(
            "reactor panel lacks complete observed delivery operands; response/local law unevaluable"
        )
    design = bundle.design
    system = reactor_system(design)
    words = action_words(design)
    views = ("reactor-native", "reactor-refined")
    system_identity = ObjectIdentity.from_record(system.system_id, system)
    projection_identity = ObjectIdentity.from_record(
        bundle.projection.projection_id, bundle.projection
    )
    axis = LawCandidateAxisMap(
        f"{PREFIX}.axes",
        (
            LawCandidateAxisBinding(
                f"{PREFIX}.axis",
                "reactor-finite-candidate",
                "reactor-fixed-denominator",
                views,
                ("finite-jacket-temperature-contrast",),
                ("absolute-state-forecast", "full-batch-safety"),
            ),
        ),
    )
    method = FiniteActionIdentificationConfig(
        f"{PREFIX}.method",
        METHOD,
        "1.0.0",
        FINITE_ACTION_MANIFEST.implementation_sha256,
        projection_identity,
        ObjectIdentity.from_record(bundle.extension.extension_id, bundle.extension),
        ObjectIdentity.from_record(
            system.relation.horizon.horizon_id, system.relation.horizon
        ),
        SUPPORT,
        (HISTORY,),
        words,
        tuple(
            FiniteActionComparatorBinding(
                f"comparator.{w.word_id}", w.word_id, words[1 - i].word_id
            )
            for i, w in enumerate(words)
        ),
        axis,
        (SUPPORT,),
        unit_ids(design),
        ("calibration",),
        ("held-out",),
        "reactor-native",
        (
            FiniteActionResponseQuantity(
                f"{PREFIX}.response",
                RECEIVER,
                RECEIVER,
                "K",
                "reactor-native-temperature",
                CLOCK,
                "callback-at-20s-state-at-10s",
                design.maximum_heldout_absolute_error_k,
                design.maximum_refinement_absolute_delta_k,
            ),
        ),
        design.minimum_complete_units,
        IndependentUnitResamplingMethod.COMPLETE_UNIT_BOOTSTRAP,
        design.bootstrap_replicates,
        design.bootstrap_seed,
        design.simultaneous_confidence_level,
        CUTOFF,
    )
    claim = CandidateClaimTemplate(
        f"{PREFIX}.claim-template",
        f"{PREFIX}.qualification",
        None,
        f"{PREFIX}.claim",
        f"{PREFIX}.law",
        "The fixed public panel supports the declared local finite jacket-temperature contrast.",
        "Paired callback-temperature difference at 20 s between a 315 K first jacket command and 316 K comparator, with the common 316 K second command.",
        "Every existing finite-action qualification obligation must pass; no controller admission forecast or benchmark safety claim follows.",
        CausalStrength.SIMULATOR_INTERVENTION,
        EvidenceCeiling.LOCAL_LAW,
        (
            "reactor-feed",
            "reactor-initial-jacket",
            "reactor-initial-temperature",
            "reactor-jacket-command",
        ),
        (RECEIVER,),
        ("fixed-public-panel-only", "native-observation-delay-retained"),
        False,
    )
    falsifiers = (
        (
            "heldout",
            FalsifierKind.WITHIN_CELL_RECURRENCE,
            "Whole heldout-unit contrast error exceeds 0.01 K.",
        ),
        (
            "one-factor",
            FalsifierKind.ONE_FACTOR_EXCHANGE,
            "Paired words differ in more than the first jacket occurrence.",
        ),
        (
            "paired",
            FalsifierKind.BASELINE_COMPARATOR,
            "A complete whole-unit paired comparator is absent.",
        ),
        (
            "refinement",
            FalsifierKind.STRUCTURAL_CONVERGENCE,
            "Paired numerical-view contrast differs by more than 0.001 K.",
        ),
        (
            "temporal",
            FalsifierKind.TEMPORAL_SUPPORT,
            "The declared native prefix or delayed receiver is unavailable.",
        ),
    )
    obligation = LawObligationTemplate(
        f"{PREFIX}.obligation-template",
        f"{PREFIX}.obligations",
        f"{PREFIX}.support",
        f"{PREFIX}.validity",
        f"{PREFIX}.uncertainty",
        f"{PREFIX}.closure",
        f"{PREFIX}.convergence",
        f"{PREFIX}.compute-obligation",
        UNIT,
        5,
        2,
        CUTOFF,
        (CHART,),
        (SUPPORT,),
        (
            QuantityBound("bound.feed", "reactor-feed", "kg/s", Decimal(0), Decimal(0)),
            QuantityBound(
                "bound.jacket",
                "reactor-jacket-command",
                "K",
                Decimal(315),
                Decimal(316),
            ),
        ),
        (SUPPORT,),
        claim.mapping_assumption_ids,
        "complete-unit-bootstrap",
        design.simultaneous_confidence_level,
        (RECEIVER,),
        ("five-fixed-public-units-not-population-safety",),
        tuple(
            FalsifierObligationTemplate(
                f"falsifier.reactor.{key}", kind, METHOD, description, description
            )
            for key, kind, description in falsifiers
        ),
        (SUPPORT,),
        ("first-jacket-command",),
        (HISTORY,),
        ("paired-whole-units", "timestep-refinement"),
        views,
        (
            NamedDecimal(
                "maximum-refinement-temperature",
                design.maximum_refinement_absolute_delta_k,
                "K",
            ),
        ),
        system.computability_envelopes[0].envelope_id,
    )
    unit_binding = ObjectIdentity.from_record(
        bundle.projection.claim_unit_binding.binding_id,
        bundle.projection.claim_unit_binding,
    )
    evidence = EvidenceLink(
        f"{PREFIX}.projection-evidence",
        EvidenceRelation.DERIVED_FROM,
        projection_identity,
        ObjectIdentity.from_record(system.relation.relation_id, system.relation),
        bundle.projection.source_payload_ids,
        system.world.world_id,
        CUTOFF,
        OutcomeAccess.EVALUATOR_REVEAL,
        VisibilityCeiling.PROSPECTIVE,
        (VisibilityCeiling.PROSPECTIVE,),
        "Exact published native prefixes under the pre-acquisition reactor design.",
    )
    scaffold = FiniteActionCandidateScaffold(
        f"{PREFIX}.scaffold",
        f"{PREFIX}.evidence",
        f"{PREFIX}.candidate",
        system_identity,
        axis,
        unit_binding,
        claim,
        obligation,
        (evidence,),
        OutcomeAccess.EVALUATOR_REVEAL,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    components = compose_finite_action_identification(
        config=method, payload_publisher=publisher
    )
    profile = components.profile_registry.evaluators[0]
    if not isinstance(profile, MethodEquivalentQualificationProfileEvaluator):
        raise TypeError(
            "reactor finite binding requires the existing method-equivalent owner"
        )
    owner = ObjectIdentity.from_record(profile.owner.owner_id, profile.owner)
    family = CandidateFamilyLedger(
        f"{PREFIX}.family",
        system_identity,
        projection_identity,
        METHOD,
        "1.0.0",
        LawMethodKind.NONLINEAR_LOCAL,
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        CHART,
        axis,
        unit_binding,
        claim,
        obligation,
        (
            CandidateFamilyMember(
                scaffold.candidate_id,
                ObjectIdentity.from_record(method.config_id, method),
                0,
                CandidateRosterDisposition.ASSESS_REQUIRED,
                (),
            ),
        ),
        ("complete-unit-bootstrap",),
        ("predeclared-simultaneous-interval",),
        (bundle.projection.projection_id,),
        owner,
        ObjectIdentity.from_record(profile.profile.profile_id, profile.profile),
        owner,
        f"{PREFIX}.multiplicity",
        "singleton-predeclared-family",
        ("seed.19",),
        "singleton-no-tie",
        OutcomeAccess.EVALUATOR_REVEAL,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    return FiniteChainConfig(
        f"{PREFIX}.bound-finite-chain", system, method, scaffold, family
    )

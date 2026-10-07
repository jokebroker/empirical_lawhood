"""Independent finite-row claims; no failed action becomes a universal veto."""

from decimal import Decimal as D
from empirical_lawhood.adapters.methods.family_evidence import singleton_candidate_evidence

from empirical_lawhood.adapters.methods.contracts import (
    CandidateClaimTemplate,
    CandidateFamilyLedger,
    CandidateFamilyMember,
    CandidateMethodEvidenceReceipt,
    CandidateRosterDisposition,
    FalsifierObligationTemplate,
    LawCandidateAxisBinding,
    LawCandidateAxisMap,
    LawCandidateEvidence,
    LawObligationTemplate,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind
from empirical_lawhood.kernel.obligations import FalsifierKind
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import ExtensionBinding
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding, QualificationScopeSpec
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt
from .config import FrontierDesign
from .law_payload import FrontierLawDecoder, FrontierLawPayload
from .qualification import QUALIFICATION_ROOTS
from .science import CHART, CUTOFF, METHOD, RECEIVERS, UNIT, frontier_system

ASSUMPTIONS = tuple(
    sorted(
        (
            "fresh-assigned-simulation-roots",
            "fixed-unshifted-exploration-context-and-jacket",
            "causal-delayed-observation-prefix",
            "complete-precommitted-120s-word",
            "matched-zero-counterfactual",
            "root-maxima-two-numerical-views",
            "no-physical-or-full-benchmark-transport",
        )
    )
)


def law_family(
    payload: FrontierLawPayload, capability: ObjectIdentity, owner: ObjectIdentity
) -> tuple[ClaimUnitBinding, CandidateFamilyLedger]:
    system, stem, c = frontier_system(), payload.stem, payload.bound.coordinate
    views = tuple(v.view_id for v in system.numerical_views)
    receivers = tuple(sorted(RECEIVERS))
    axes = LawCandidateAxisMap(
        f"{stem}.axes",
        (
            LawCandidateAxisBinding(
                f"{stem}.axis",
                f"{stem}.fixed",
                CHART,
                views,
                ("one-finite-word-joint-response-and-safety",),
                tuple(
                    sorted(
                        (
                            "other-contexts",
                            "other-response-windows",
                            "other-actions",
                            "physical-transport",
                            "full-batch-control",
                        )
                    )
                ),
            ),
        ),
    )
    scope = QualificationScopeSpec(
        f"{stem}.scope",
        f"{stem}.claim",
        f"{stem}.all-assigned-population",
        UNIT,
        QUALIFICATION_ROOTS,
        tuple(sorted((CHART, *views))),
        "whole-root-two-view-joint-event",
        UNIT,
        CHART,
        EvidenceCeiling.LOCAL_LAW,
    )
    binding = ClaimUnitBinding(
        f"{stem}.claim-units",
        scope.claim_id,
        ObjectIdentity.from_record(scope.scope_id, scope),
        QUALIFICATION_ROOTS,
        scope.aggregation_level_id,
    )
    claim = CandidateClaimTemplate(
        f"{stem}.claim-template",
        f"{stem}.qualification",
        None,
        scope.claim_id,
        f"{stem}.law",
        f"A fixed empirical interval bounds {c.pulse.word_id} at {c.context}, response window {c.horizon_s} seconds, with the complete 120-second guard.",
        "One native feed word under a fixed jacket; all assigned roots, paired numerical views and a measured zero reference.",
        "No additional action, context, regime, hidden coefficient, physical or full-batch claim.",
        CausalStrength.SIMULATOR_INTERVENTION,
        EvidenceCeiling.LOCAL_LAW,
        tuple(
            sorted(
                (
                    *system.relation.denominator_quantity_ids,
                    *system.relation.history_quantity_ids,
                    *system.relation.action_quantity_ids,
                )
            )
        ),
        receivers,
        ASSUMPTIONS,
        False,
    )
    template = LawObligationTemplate(
        template_id=f"{stem}.template",
        obligations_id=f"{stem}.obligations",
        support_id=CHART,
        validity_id=f"{stem}.validity",
        uncertainty_id=f"{stem}.uncertainty",
        closure_id=f"{stem}.closure",
        structural_convergence_id=f"{stem}.convergence",
        computability_id=f"{stem}.computability",
        independent_unit_id=UNIT,
        physical_unit_count=96,
        nested_numerical_view_count=2,
        information_cutoff_id=CUTOFF,
        chart_ids=(CHART,),
        denominator_cell_ids=(CHART,),
        action_bounds=(),
        validity_domain_ids=(CHART,),
        assumption_ids=ASSUMPTIONS,
        uncertainty_method_key="frozen-finite-row-cp-family72",
        uncertainty_confidence_level=D(".90"),
        interval_quantity_ids=receivers,
        uncertainty_limitation_codes=(
            "ALL_ASSIGNED_MARGINAL_ROOT_ADEQUACY_ONLY",
            "NO_OTHER_ACTION_TRANSPORT",
            "NO_PHYSICAL_TRANSPORT",
        ),
        falsifiers=(
            FalsifierObligationTemplate(
                f"{stem}.falsifier",
                FalsifierKind.RECEIVER_GATE,
                METHOD,
                "No-contact, invalid native delivery/numerics, uncovered response/envelope or unsafe action defeats the joint root event.",
                "All 96 assigned roots; CP lower using alpha=.05/72 at least .90; zero unsafe prefix/action roots; development nomination required.",
            ),
        ),
        recurrence_cell_ids=(CHART,),
        exchange_factor_ids=("native-feed-word", "native-integrator-timestep"),
        retained_history_ids=system.relation.history_quantity_ids,
        required_structure_ids=system.computability_envelopes[0].required_structure_ids,
        numerical_view_ids=views,
        structural_tolerances=tuple(
            sorted(
                (
                    NamedDecimal(RECEIVERS[0], D(".01"), "K"),
                    NamedDecimal(RECEIVERS[1], c.epsilon_K, "K"),
                ),
                key=lambda v: v.value_id,
            )
        ),
        computability_envelope_id=system.computability_envelopes[0].envelope_id,
    )
    design = FrontierDesign()
    ledger = CandidateFamilyLedger(
        f"{stem}.family",
        ObjectIdentity.from_record(system.system_id, system),
        payload.qualification,
        METHOD,
        "1.0.0",
        LawMethodKind.NONLINEAR_LOCAL,
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        CHART,
        axes,
        ObjectIdentity.from_record(binding.binding_id, binding),
        claim,
        template,
        (
            CandidateFamilyMember(
                f"{stem}.fixed",
                payload.development,
                0,
                CandidateRosterDisposition.ASSESS_REQUIRED,
                (),
            ),
        ),
        ("32-retained-root-expanded-development",),
        ("fixed-outward-extrema-rule",),
        (c.context,),
        capability,
        ObjectIdentity.from_record(design.config_id, design),
        owner,
        f"{stem}.singleton",
        "one-row-from-predeclared-family-of-72",
        ("all-assigned-qualification-roots",),
        "Assess this unchanged development bound independently; never refit or require another action to succeed.",
        OutcomeAccess.EVALUATOR_REVEAL,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    return binding, ledger


def candidate_evidence(
    family: CandidateFamilyLedger,
    binding: ClaimUnitBinding,
    publication: CandidatePayloadPublicationReceipt,
    receipts: tuple[CandidateMethodEvidenceReceipt, ...],
    links: tuple[EvidenceLink, ...],
) -> LawCandidateEvidence:
    return singleton_candidate_evidence(
        family,
        binding,
        publication,
        receipts,
        links,
        ExtensionBinding(
            FrontierLawDecoder().extension_namespace,
            FrontierLawPayload.SCHEMA,
            publication.content_sha256,
        ),
    )

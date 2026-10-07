"One frozen selected reactor law candidate for the generic local law owner."

from __future__ import annotations

from decimal import Decimal as D

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

from .config import ReactorRegimeResponseDesign
from .law_payload import RegimeJointLawDecoder, RegimeJointLawPayload
from .law_qualification import QUALIFICATION_ROOTS, STEM
from .science import CHART, METHOD, RECEIVERS, UNIT, regime_system

CUTOFF = "reactor-regime-selected-causal-callback"
ASSUMPTIONS = tuple(sorted((
    "fresh-assigned-simulation-roots",
    "frozen-selected-preparation-and-fixed-jacket",
    "causal-delayed-observation-prefix",
    "three-scalar-feed-words-with-measured-zero-reference",
    "root-maxima-two-numerical-views",
    "no-physical-or-full-benchmark-transport",
)))


def joint_law_family(
    design: ReactorRegimeResponseDesign,
    payload: RegimeJointLawPayload,
    operands: ObjectIdentity,
    capability: ObjectIdentity,
    owner: ObjectIdentity,
) -> tuple[ClaimUnitBinding, CandidateFamilyLedger]:
    system = regime_system()
    # The scalar ActionWord uses this exact denominator identity.  controller admission and prospective controller evaluation
    # must preserve it through candidate axes, support cells and custody.
    support = CHART
    views = tuple(value.view_id for value in system.numerical_views)
    receivers = tuple(sorted(RECEIVERS))
    axis_map = LawCandidateAxisMap(
        f"{STEM}.axes",
        (LawCandidateAxisBinding(
            f"{STEM}.axis",
            f"{STEM}.fixed",
            support,
            views,
            ("joint-absolute-and-scalar-action-response",),
            tuple(sorted((
                "other-preparations", "full-batch-empirical-control",
                "physical-transport", "regime-added-value", "method-superiority",
            ))),
        ),),
    )
    scope = QualificationScopeSpec(
        f"{STEM}.scope",
        f"{STEM}.claim",
        f"{STEM}.all-assigned-qualification-population",
        UNIT,
        QUALIFICATION_ROOTS,
        tuple(sorted((support, *views))),
        "whole-root-all-chart-two-view-maximum",
        UNIT,
        support,
        EvidenceCeiling.LOCAL_LAW,
    )
    binding = ClaimUnitBinding(
        f"{STEM}.claim-units",
        scope.claim_id,
        ObjectIdentity.from_record(scope.scope_id, scope),
        QUALIFICATION_ROOTS,
        scope.aggregation_level_id,
    )
    claim = CandidateClaimTemplate(
        f"{STEM}.claim-template",
        f"{STEM}.qualification",
        None,
        scope.claim_id,
        f"{STEM}.law",
        "Joint local peak-temperature and paired feed-cooling law on one frozen preparation.",
        "Ten-second absolute and measured action-response predictions in the selected fixed-jacket chart, with all-assigned root adequacy.",
        "No physical, cross-preparation, global regime, method-superiority or full-batch controller claim.",
        CausalStrength.SIMULATOR_INTERVENTION,
        EvidenceCeiling.LOCAL_LAW,
        tuple(sorted((
            *system.relation.denominator_quantity_ids,
            *system.relation.history_quantity_ids,
            *system.relation.action_quantity_ids,
        ))),
        receivers,
        ASSUMPTIONS,
        False,
    )
    template = LawObligationTemplate(
        template_id=f"{STEM}.template",
        obligations_id=f"{STEM}.obligations",
        support_id=support,
        validity_id=f"{STEM}.validity",
        uncertainty_id=f"{STEM}.uncertainty",
        closure_id=f"{STEM}.closure",
        structural_convergence_id=f"{STEM}.convergence",
        computability_id=f"{STEM}.computability",
        independent_unit_id=UNIT,
        physical_unit_count=64,
        nested_numerical_view_count=2,
        information_cutoff_id=CUTOFF,
        chart_ids=(CHART,),
        denominator_cell_ids=(support,),
        action_bounds=(),
        validity_domain_ids=(support,),
        assumption_ids=ASSUMPTIONS,
        uncertainty_method_key="selected-joint-root-maxima-cp95",
        uncertainty_confidence_level=D(".90"),
        interval_quantity_ids=receivers,
        uncertainty_limitation_codes=tuple(sorted((
            "ALL_ASSIGNED_MARGINAL_ROOT_ADEQUACY_ONLY",
            "NO_PER_LEAF_COVERAGE_GUARANTEE",
            "NO_CROSS_PREPARATION_TRANSPORT",
            "NO_PHYSICAL_REACTOR_TRANSPORT",
        ))),
        falsifiers=(FalsifierObligationTemplate(
            f"{STEM}.falsifier",
            FalsifierKind.RECEIVER_GATE,
            METHOD,
            "Missing/invalid native contact, unsupported input, excessive calibrated precision or uncovered joint receiver falsifies the bounded law claim.",
            "At least 29 valid contacted calibration roots, both fixed q factors at most one, and CP95 lower bound at least .90 for 64 all-assigned qualification roots.",
        ),),
        recurrence_cell_ids=(support,),
        exchange_factor_ids=tuple(sorted(("native-feed-word", "native-integrator-timestep"))),
        retained_history_ids=system.relation.history_quantity_ids,
        required_structure_ids=system.computability_envelopes[0].required_structure_ids,
        numerical_view_ids=views,
        structural_tolerances=tuple(sorted((
            NamedDecimal(RECEIVERS[0], D(".01"), "K"),
            NamedDecimal(RECEIVERS[1], D(".000001"), "K"),
        ), key=lambda value: value.value_id)),
        computability_envelope_id=system.computability_envelopes[0].envelope_id,
    )
    family = CandidateFamilyLedger(
        f"{STEM}.family",
        ObjectIdentity.from_record(system.system_id, system),
        operands,
        METHOD,
        "1.0.0",
        LawMethodKind.NONLINEAR_LOCAL,
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        CHART,
        axis_map,
        ObjectIdentity.from_record(binding.binding_id, binding),
        claim,
        template,
        (CandidateFamilyMember(
            f"{STEM}.fixed",
            ObjectIdentity.from_record(design.config_id, design),
            0,
            CandidateRosterDisposition.ASSESS_REQUIRED,
            (),
        ),),
        ("fit-only-root-blocked-K-S-L-and-independent-absolute-baseline",),
        ("nomination-only-finite-route-choice",),
        (payload.route or "no-selected-route",),
        capability,
        ObjectIdentity.from_record(design.config_id, design),
        owner,
        f"{STEM}.singleton",
        "one-frozen-development-nominee-no-heldout-selection",
        ("all-assigned-qualification-roots",),
        "Assess the selected law even when fresh precision or adequacy fails; no C outcome can change the nominee.",
        OutcomeAccess.EVALUATOR_REVEAL,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    return binding, family


def joint_candidate_evidence(
    family: CandidateFamilyLedger,
    binding: ClaimUnitBinding,
    publication: CandidatePayloadPublicationReceipt,
    receipts: tuple[CandidateMethodEvidenceReceipt, ...],
    links: tuple[EvidenceLink, ...],
) -> LawCandidateEvidence:
    member = family.members[0]
    return LawCandidateEvidence(
        f"{member.candidate_id}.evidence",
        member.candidate_id,
        family.system,
        family.dataset_or_projection,
        member.config,
        METHOD,
        "1.0.0",
        family.method_kind,
        family.representation_kind,
        publication.candidate_evaluator,
        family.axis_map,
        family.claim_unit_binding,
        binding.independent_unit_instance_ids,
        family.claim_template,
        family.obligation_template,
        receipts,
        publication,
        links,
        family.outcome_access,
        family.parent_visibility_ceilings,
        family.visibility_ceiling,
        ExtensionBinding(
            RegimeJointLawDecoder().extension_namespace,
            RegimeJointLawPayload.SCHEMA,
            publication.content_sha256,
        ),
    )

"""Separate frozen candidate family for each discovered domain and receiver."""

from decimal import Decimal as D

from empirical_lawhood.adapters.methods.contracts import (
    CandidateClaimTemplate,
    CandidateFamilyLedger,
    CandidateFamilyMember,
    CandidateRosterDisposition,
    FalsifierObligationTemplate,
    LawCandidateAxisBinding,
    LawCandidateAxisMap,
    LawObligationTemplate,
    LawCandidateEvidence,
    CandidateMethodEvidenceReceipt,
)
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind
from empirical_lawhood.kernel.obligations import FalsifierKind
from empirical_lawhood.kernel.provenance import ObjectIdentity, EvidenceLink
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import ExtensionBinding
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding, QualificationScopeSpec
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt
from .config import FeedQualificationDesign, PREFIX
from .science import feed_system, RECEIVERS, UNITS, UNIT, CHART, CUTOFF, METHOD, QUALIFICATION_UNITS, ASSUMPTIONS
from .payload import FeedDecoder


def feed_family(
    design: FeedQualificationDesign,
    domain: str,
    operands: ObjectIdentity,
    capability: ObjectIdentity,
    owner: ObjectIdentity,
) -> tuple[ClaimUnitBinding, CandidateFamilyLedger]:
    system = feed_system()
    stem = f"{PREFIX}.{domain}.joint"
    support = f"{stem}.support"
    model = next(d for d in design.atlas.candidates if d.domain_id == domain)
    if model.nominated_receivers != (0, 1):
        raise ValueError("qualification receiver was not nominated before fresh acquisition")
    config = ObjectIdentity.from_record(design.config_id, design)
    views = tuple(v.view_id for v in system.numerical_views)
    outputs = tuple(sorted(RECEIVERS))
    axes = LawCandidateAxisMap(
        f"{stem}.axes",
        (
            LawCandidateAxisBinding(
                f"{stem}.axis",
                f"{stem}.fixed",
                support,
                views,
                ("local-absolute-and-measured-action-response",),
                (
                    "arbitrary-policy-coverage",
                    "cross-domain-transitions",
                    "full-batch-completion",
                    "global-effect-or-method-superiority",
                ),
            ),
        ),
    )
    scope = QualificationScopeSpec(
        f"{stem}.scope",
        f"{stem}.claim",
        f"{stem}.causal-contact-population",
        UNIT,
        QUALIFICATION_UNITS,
        tuple(sorted((support, *views))),
        "whole-root-local-maximum",
        UNIT,
        support,
        EvidenceCeiling.LOCAL_LAW,
    )
    binding = ClaimUnitBinding(
        f"{stem}.claim-units",
        scope.claim_id,
        ObjectIdentity.from_record(scope.scope_id, scope),
        QUALIFICATION_UNITS,
        scope.aggregation_level_id,
    )
    claim = CandidateClaimTemplate(
        f"{stem}.claim-template",
        f"{stem}.qualification",
        None,
        scope.claim_id,
        f"{stem}.law",
        f"Joint empirical peak-temperature and feed-cooling law in prepared domain {domain}.",
        "Ten-second absolute and paired-action predictions in the measured local chart; conditional on frozen causal encounter, with whole-root calibration and fresh qualification.",
        "No global coverage, cross-domain transition, physical transport, resolved causal effect, superiority or task-completion claim.",
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
        outputs,
        ASSUMPTIONS,
        False,
    )
    obligations = LawObligationTemplate(
        template_id=f"{stem}.template",
        obligations_id=f"{stem}.obligations",
        support_id=support,
        validity_id=f"{stem}.validity",
        uncertainty_id=f"{stem}.uncertainty",
        closure_id=f"{stem}.closure",
        structural_convergence_id=f"{stem}.convergence",
        computability_id=f"{stem}.computability",
        independent_unit_id=UNIT,
        physical_unit_count=32,
        nested_numerical_view_count=2,
        information_cutoff_id=CUTOFF,
        chart_ids=(CHART,),
        denominator_cell_ids=(support,),
        action_bounds=(),
        validity_domain_ids=(support,),
        assumption_ids=ASSUMPTIONS,
        uncertainty_method_key="conditional-local-root-maxima-cp95",
        uncertainty_confidence_level=D(".90"),
        interval_quantity_ids=outputs,
        uncertainty_limitation_codes=(
            "CONDITIONAL_ON_CAUSAL_DOMAIN_CONTACT",
            "NO_CROSS_DOMAIN_SIMULTANEOUS_COVERAGE",
            "NO_PRIVATE_PANEL_COVERAGE",
            "NUMERICAL_SELF_CONSISTENCY_ONLY",
        ),
        falsifiers=(
            FalsifierObligationTemplate(
                f"{stem}.falsifier",
                FalsifierKind.RECEIVER_GATE,
                METHOD,
                "Invalid/missing local measurements, inadequate contact, or receiver-relative predictive/action-response precision prevents promotion.",
                "At least 29 contacted roots in each split, calibrated halfwidth within its frozen cap and one-sided CP lower bound at least .90 on fresh local root adequacy.",
            ),
        ),
        recurrence_cell_ids=(support,),
        exchange_factor_ids=tuple(
            sorted(
                (
                    *(("native-feed-word",) if len({a // 3 for a in model.actions}) > 1 else ()),
                    *(("native-jacket-word",) if len({a % 3 for a in model.actions}) > 1 else ()),
                    "native-integrator-timestep",
                )
            )
        ),
        retained_history_ids=system.relation.history_quantity_ids,
        required_structure_ids=("causal-domains", "matched-input-views"),
        numerical_view_ids=views,
        structural_tolerances=tuple(
            sorted(
                (NamedDecimal(RECEIVERS[r], design.numerical_padding[r], UNITS[r]) for r in (0, 1)),
                key=lambda v: v.value_id,
            )
        ),
        computability_envelope_id=system.computability_envelopes[0].envelope_id,
    )
    ledger = CandidateFamilyLedger(
        f"{stem}.family",
        ObjectIdentity.from_record(system.system_id, system),
        operands,
        METHOD,
        "1.0.0",
        LawMethodKind.NONLINEAR_LOCAL,
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        CHART,
        axes,
        ObjectIdentity.from_record(binding.binding_id, binding),
        claim,
        obligations,
        (
            CandidateFamilyMember(
                f"{stem}.fixed", config, 0, CandidateRosterDisposition.ASSESS_REQUIRED, ()
            ),
        ),
        ("root-blocked-model-tree-and-paired-response-fit",),
        ("receiver-specific-development-nomination",),
        (design.atlas.atlas_id,),
        capability,
        config,
        owner,
        f"{stem}.singleton",
        "fixed-candidate-no-heldout-selection",
        ("assigned-public-envelope-units",),
        "Assess every nominated domain/receiver separately; no selection among fresh outcomes.",
        OutcomeAccess.EVALUATOR_REVEAL,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    return binding, ledger


def candidate_evidence(
    ledger: CandidateFamilyLedger,
    binding: ClaimUnitBinding,
    publication: CandidatePayloadPublicationReceipt,
    receipts: tuple[CandidateMethodEvidenceReceipt, ...],
    links: tuple[EvidenceLink, ...],
) -> LawCandidateEvidence:
    decoder = FeedDecoder()
    member = ledger.members[0]
    return LawCandidateEvidence(
        f"{member.candidate_id}.evidence",
        member.candidate_id,
        ledger.system,
        ledger.dataset_or_projection,
        member.config,
        METHOD,
        "1.0.0",
        ledger.method_kind,
        ledger.representation_kind,
        publication.candidate_evaluator,
        ledger.axis_map,
        ledger.claim_unit_binding,
        binding.independent_unit_instance_ids,
        ledger.claim_template,
        ledger.obligation_template,
        receipts,
        publication,
        links,
        ledger.outcome_access,
        ledger.parent_visibility_ceilings,
        ledger.visibility_ceiling,
        ExtensionBinding(
            decoder.extension_namespace, decoder.payload_schema, publication.content_sha256
        ),
    )

"One development-frozen selected action bound for the generic local law owner."

from __future__ import annotations

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

from .config import ClassicalDesign
from .law_payload import ClassicalLawDecoder, ClassicalLawPayload
from .law_qualification import QUALIFICATION_ROOTS, STEM
from .science import CHART, METHOD, RECEIVERS, UNIT, classical_system

CUTOFF = "reactor-classical-selected-causal-callback"
ASSUMPTIONS = tuple(
    sorted(
        (
            "fresh-assigned-simulation-roots",
            "frozen-selected-preparation-and-fixed-jacket",
            "causal-delayed-observation-prefix",
            "one-positive-feed-word-with-measured-zero-reference",
            "root-maxima-two-numerical-views",
            "no-physical-or-full-benchmark-transport",
        )
    )
)


def classical_law_family(
    design: ClassicalDesign,
    payload: ClassicalLawPayload,
    operands: ObjectIdentity,
    capability: ObjectIdentity,
    owner: ObjectIdentity,
) -> tuple[ClaimUnitBinding, CandidateFamilyLedger]:
    system = classical_system()
    # The scalar ActionWord uses this exact denominator identity.  controller admission and prospective controller evaluation
    # must preserve it through candidate axes, support cells and custody.
    support = CHART
    views = tuple(value.view_id for value in system.numerical_views)
    receivers = tuple(sorted(RECEIVERS))
    axis_map = LawCandidateAxisMap(
        f"{STEM}.axes",
        (
            LawCandidateAxisBinding(
                f"{STEM}.axis",
                f"{STEM}.fixed",
                support,
                views,
                ("selected-finite-response-and-safety-bound",),
                tuple(
                    sorted(
                        (
                            "other-preparations",
                            "full-batch-empirical-control",
                            "physical-transport",
                            "regime-added-value",
                            "method-superiority",
                        )
                    )
                ),
            ),
        ),
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
        "One selected finite cooling interval and coarse causal safety envelope under exploration-unshifted preparation.",
        "One 0.16 kg feed word at fixed jacket; empirical paired peak-cooling interval and one-kelvin causal temperature envelope, tested on all assigned roots.",
        "No physical, cross-preparation, global regime, method-superiority or full-batch controller claim.",
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
        uncertainty_method_key="selected-action-frozen-bound-cp95",
        uncertainty_confidence_level=D(".90"),
        interval_quantity_ids=receivers,
        uncertainty_limitation_codes=tuple(
            sorted(
                (
                    "ALL_ASSIGNED_MARGINAL_ROOT_ADEQUACY_ONLY",
                    "NO_PER_LEAF_COVERAGE_GUARANTEE",
                    "NO_CROSS_PREPARATION_TRANSPORT",
                    "NO_PHYSICAL_REACTOR_TRANSPORT",
                )
            )
        ),
        falsifiers=(
            FalsifierObligationTemplate(
                f"{STEM}.falsifier",
                FalsifierKind.RECEIVER_GATE,
                METHOD,
                "Missing/invalid native contact, unsupported input, excessive calibrated precision or uncovered joint receiver falsifies the bounded law claim.",
                "All 64 fresh assigned roots retained, no unsafe positive delivery, and CP95 lower bound at least .90 for the development-frozen selected response and safety bound.",
            ),
        ),
        recurrence_cell_ids=(support,),
        exchange_factor_ids=tuple(sorted(("native-feed-word", "native-integrator-timestep"))),
        retained_history_ids=system.relation.history_quantity_ids,
        required_structure_ids=system.computability_envelopes[0].required_structure_ids,
        numerical_view_ids=views,
        structural_tolerances=tuple(
            sorted(
                (
                    NamedDecimal(RECEIVERS[0], D(".01"), "K"),
                    NamedDecimal(RECEIVERS[1], D(".000001"), "K"),
                ),
                key=lambda value: value.value_id,
            )
        ),
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
        (
            CandidateFamilyMember(
                f"{STEM}.fixed",
                ObjectIdentity.from_record(design.config_id, design),
                0,
                CandidateRosterDisposition.ASSESS_REQUIRED,
                (),
            ),
        ),
        ("exposed-144-root-finite-word-discovery",),
        ("least-dose-largest-preexisting-target-with-ten-percent-margin",),
        (payload.route,),
        capability,
        ObjectIdentity.from_record(design.config_id, design),
        owner,
        f"{STEM}.singleton",
        "one-frozen-development-nominee-no-heldout-selection",
        ("all-assigned-qualification-roots",),
        "Assess the one selected bound; no fresh outcome can change its word, interval or support.",
        OutcomeAccess.EVALUATOR_REVEAL,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    return binding, family


def classical_candidate_evidence(
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
            ClassicalLawDecoder().extension_namespace,
            ClassicalLawPayload.SCHEMA,
            publication.content_sha256,
        ),
    )

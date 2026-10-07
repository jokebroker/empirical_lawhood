"""Frozen on-policy payload and ordinary candidate-family records."""

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
from .science import RECEIVERS, UNITS
from .payload import DevelopmentBoundReactorResponseDecoder

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind
from empirical_lawhood.kernel.obligations import FalsifierKind
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import ExtensionBinding
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding, QualificationScopeSpec
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt
from .science import ASSUMPTIONS, QUALIFICATION_UNITS, CHART, CUTOFF, METHOD, PREFIX, SUPPORT, UNIT

NUMERICAL_TOLERANCE = (D(".01"), D(".0002"))

MAXIMUM_LAW_BYTES = 1024**2
EVIDENCE_KIND = f"{PREFIX}.whole-episode-forecast-operands"


def empirical_family(
    system: SystemSpec,
    config: ObjectIdentity,
    calibration: ObjectIdentity,
    capability: ObjectIdentity,
    implementation: ObjectIdentity,
) -> tuple[ClaimUnitBinding, CandidateFamilyLedger]:
    views = tuple(v.view_id for v in system.numerical_views)
    outputs = tuple(sorted(RECEIVERS))
    axes = LawCandidateAxisMap(
        f"{PREFIX}.axes",
        (
            LawCandidateAxisBinding(
                f"{PREFIX}.axis",
                f"{PREFIX}.fixed-version",
                SUPPORT,
                views,
                ("on-policy-absolute-response",),
                ("off-policy-action-exchange", "physical-transport", "private-panel-coverage"),
            ),
        ),
    )
    scope = QualificationScopeSpec(
        f"{PREFIX}.scope",
        f"{PREFIX}.claim",
        f"{PREFIX}.calibration-population",
        UNIT,
        QUALIFICATION_UNITS,
        tuple(sorted((SUPPORT, *views))),
        "complete-assigned-episode",
        UNIT,
        SUPPORT,
        EvidenceCeiling.LOCAL_LAW,
    )
    binding = ClaimUnitBinding(
        f"{PREFIX}.claim-units",
        scope.claim_id,
        ObjectIdentity.from_record(scope.scope_id, scope),
        QUALIFICATION_UNITS,
        scope.aggregation_level_id,
    )
    claim = CandidateClaimTemplate(
        f"{PREFIX}.claim-template",
        f"{PREFIX}.qualification",
        None,
        scope.claim_id,
        f"{PREFIX}.law",
        "Empirically learned peak-temperature and conversion law for the frozen consumer's chosen path.",
        "Temperature peak and endpoint conversion over ten seconds; errors maximized over complete assigned episodes and both numerical views.",
        "Fresh-root frozen-consumer chosen-path scope; private-panel and post-admission conditional coverage are not established.",
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
        template_id=f"{PREFIX}.template",
        obligations_id=f"{PREFIX}.obligations",
        support_id=SUPPORT,
        validity_id=f"{PREFIX}.validity",
        uncertainty_id=f"{PREFIX}.uncertainty",
        closure_id=f"{PREFIX}.closure",
        structural_convergence_id=f"{PREFIX}.convergence",
        computability_id=f"{PREFIX}.computability",
        independent_unit_id=UNIT,
        physical_unit_count=32,
        nested_numerical_view_count=2,
        information_cutoff_id=CUTOFF,
        chart_ids=(CHART,),
        denominator_cell_ids=(SUPPORT,),
        action_bounds=(),
        validity_domain_ids=(SUPPORT,),
        assumption_ids=ASSUMPTIONS,
        uncertainty_method_key="whole-root-joint-rank32-of32",
        uncertainty_confidence_level=D(".90"),
        interval_quantity_ids=outputs,
        uncertainty_limitation_codes=(
            "MARGINAL_WHOLE_EPISODE_ONLY",
            "NO_COMPONENT_IDENTIFICATION",
            "NO_PRIVATE_PANEL_COVERAGE_GUARANTEE",
            "NUMERICAL_SELF_CONSISTENCY_ONLY",
        ),
        falsifiers=(
            FalsifierObligationTemplate(
                f"{PREFIX}.coverage-falsifier",
                FalsifierKind.RECEIVER_GATE,
                METHOD,
                "All assigned episodes remain accounted; numerical discrepancy or missing data gives an infinite root score.",
                "Any infinite calibration score or any of 32 qualification roots failing adequacy refuses qualification.",
            ),
        ),
        recurrence_cell_ids=(SUPPORT,),
        exchange_factor_ids=("native-integrator-timestep",),
        retained_history_ids=system.relation.history_quantity_ids,
        required_structure_ids=("complete-whole-episode-maxima", "coupled-numerical-views"),
        numerical_view_ids=views,
        structural_tolerances=tuple(
            sorted(
                (
                    NamedDecimal(q, n, u)
                    for q, n, u in zip(RECEIVERS, NUMERICAL_TOLERANCE, UNITS, strict=True)
                ),
                key=lambda v: v.value_id,
            )
        ),
        computability_envelope_id=system.computability_envelopes[0].envelope_id,
    )
    return binding, CandidateFamilyLedger(
        f"{PREFIX}.family",
        ObjectIdentity.from_record(system.system_id, system),
        calibration,
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
                f"{PREFIX}.fixed", config, 0, CandidateRosterDisposition.ASSESS_REQUIRED, ()
            ),
        ),
        ("nine-development-ridge-fits-frozen-nominee",),
        ("rank32-calibration-and-32-of-32-qualification",),
        (config.object_id,),
        capability,
        config,
        implementation,
        f"{PREFIX}.singleton",
        "fixed-candidate-no-selection",
        ("assigned-public-envelope-units",),
        "One empirically nominated frozen predictor, with calibration followed by fresh chosen-path qualification.",
        OutcomeAccess.EVALUATOR_REVEAL,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )


def candidate_evidence(
    ledger: CandidateFamilyLedger,
    binding: ClaimUnitBinding,
    publication: CandidatePayloadPublicationReceipt,
    receipts: tuple[CandidateMethodEvidenceReceipt, ...],
    links: tuple[EvidenceLink, ...],
) -> LawCandidateEvidence:
    decoder = DevelopmentBoundReactorResponseDecoder()
    member = ledger.members[0]
    return LawCandidateEvidence(
        f"{PREFIX}.candidate-evidence",
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

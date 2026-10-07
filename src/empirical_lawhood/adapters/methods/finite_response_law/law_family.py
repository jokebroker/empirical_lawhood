"""Four separate, fixed singleton families in the existing qualification IR."""

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
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind
from empirical_lawhood.kernel.obligations import FalsifierKind
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import ExtensionBinding
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.identification_evidence import (
    ClaimUnitBinding,
    QualificationScopeSpec,
)
from empirical_lawhood.runtime.candidate_payloads import (
    CandidatePayloadPublicationReceipt,
)

from .calibration_panel import FIXED_CALIBRATION_ROOT_IDS, valid_calibration_root_ids
from .law_binding import PARENT_QUANTITY, feature_quantities, output_quantities
from .law_payloads import FiniteResponseLawConditionalDecoder, FiniteResponseLawLowerDecoder
from .science import PROGRAMME, FiniteResponseLawScienceSpec

EVIDENCE_KIND = f"{PROGRAMME}.fresh-calibration-operands"
ASSUMPTIONS = tuple(
    sorted(
        (
            "exchangeable-independent-assigned-roots",
            "fixed-finite-native-action-menu",
            "frozen-precalibration-coefficients-scales-and-support",
            "marginal-complete-root-coverage-only",
            "paired-receiver-is-not-a-single-run-outcome",
            "two-futures-and-two-views-do-not-increase-n",
        )
    )
)


def law_family(
    *,
    system: SystemSpec,
    config: ObjectIdentity,
    calibration: ObjectIdentity,
    boundary: str,
    root_ids: tuple[str, ...] = FIXED_CALIBRATION_ROOT_IDS,
    selector_capability: ObjectIdentity,
    selector_implementation: ObjectIdentity,
) -> tuple[QualificationScopeSpec, ClaimUnitBinding, CandidateFamilyLedger]:
    if not valid_calibration_root_ids(root_ids):
        raise ValueError("Finite response-law family changes its assigned physical-root cohort")
    inputs = tuple(q.quantity_id for q in feature_quantities(boundary))
    if boundary != "lower":
        inputs = (*inputs, PARENT_QUANTITY)
    prefix = f"{PROGRAMME}.{boundary}-law"
    views = tuple(v.view_id for v in system.numerical_views)
    outputs = tuple(sorted(q.quantity_id for q in output_quantities()))
    if system.relation.receiver_quantity_ids != outputs or len(views) != 2:
        raise ValueError(
            "Finite response-law family changes its paired receiver or two-view denominator"
        )
    # One finite law over the declared parent-assignment distribution. Parent is
    # an explicit conditional input, not five independently calibrated strata.
    axes = LawCandidateAxisMap(
        f"{prefix}.axes",
        (
            LawCandidateAxisBinding(
                f"{prefix}.axis",
                f"{prefix}.frozen-version",
                f"{PROGRAMME}.assigned-native-medium",
                views,
                ("joint-finite-native-paired-response",),
                (
                    "conditional-interface-coverage",
                    "continuous-actuation",
                    "physical-transport",
                    "single-run-differential-outcome",
                ),
            ),
        ),
    )
    scope = QualificationScopeSpec(
        f"{prefix}.scope",
        f"{prefix}.claim",
        f"{PROGRAMME}.calibration.population",
        system.independent_unit.unit_id,
        root_ids,
        tuple(sorted((*axes.denominator_member_ids, *views))),
        "complete-assigned-root",
        system.independent_unit.unit_id,
        f"{prefix}.frozen-support",
        EvidenceCeiling.LOCAL_LAW,
    )
    binding = ClaimUnitBinding(
        f"{prefix}.claim-units",
        scope.claim_id,
        ObjectIdentity.from_record(scope.scope_id, scope),
        root_ids,
        scope.aggregation_level_id,
    )
    claim = CandidateClaimTemplate(
        f"{prefix}.claim-template",
        f"{prefix}.qualification",
        None,
        scope.claim_id,
        f"{prefix}.law",
        "Frozen finite native-action paired response with fresh rootwise predictive calibration.",
        "Eight oriented paired response/preservation coordinates for four signed pairs, two futures and two coupled numerical views at +192 ticks.",
        "90% marginal complete-root scope under exchangeability; Prospective evaluation usefulness and information remain separate claims.",
        CausalStrength.SIMULATOR_INTERVENTION,
        EvidenceCeiling.LOCAL_LAW,
        tuple(sorted((*inputs, *system.relation.action_quantity_ids))),
        outputs,
        ASSUMPTIONS,
        False,
    )
    obligations = LawObligationTemplate(
        template_id=f"{prefix}.template",
        obligations_id=f"{prefix}.obligations",
        support_id=f"{prefix}.support",
        validity_id=f"{prefix}.validity",
        uncertainty_id=f"{prefix}.uncertainty",
        closure_id=f"{prefix}.closure",
        structural_convergence_id=f"{prefix}.convergence",
        computability_id=f"{prefix}.computability",
        independent_unit_id=system.independent_unit.unit_id,
        physical_unit_count=32,
        nested_numerical_view_count=2,
        information_cutoff_id=f"{prefix}.{'handoff-4368' if boundary == 'lower' else 'prefix-4096'}",
        chart_ids=(f"{prefix}.paired-finite-chart",),
        denominator_cell_ids=axes.denominator_member_ids,
        action_bounds=(),
        validity_domain_ids=(scope.locality_scope_id,),
        assumption_ids=ASSUMPTIONS,
        uncertainty_method_key="frozen-scale-rootwise-rank30-of32",
        uncertainty_confidence_level=D(".90"),
        interval_quantity_ids=outputs,
        uncertainty_limitation_codes=(
            "MARGINAL_COMPLETE_ROOT_ONLY",
            "NO_LATENT_COMPONENT_IDENTIFICATION",
            "NO_POST_USABILITY_CONDITIONAL_GUARANTEE",
            "NUMERICAL_ALLOWANCE_IS_SELF_CONSISTENCY_NOT_TRUE_ERROR_BOUND",
        ),
        falsifiers=(
            FalsifierObligationTemplate(
                f"{prefix}.unbounded-calibration",
                FalsifierKind.RECEIVER_GATE,
                prefix,
                "Retain invalid, unsupported and numerically discrepant roots as infinite maximum scores.",
                "A nonfinite rank-30 multiplier refuses the boundary; unaccounted custody is unevaluable.",
            ),
        ),
        recurrence_cell_ids=("two-independent-future-purposes-per-assigned-root",),
        exchange_factor_ids=(
            "native-pulse-magnitude",
            "native-pulse-port",
            "native-pulse-sign",
        ),
        retained_history_ids=system.relation.history_quantity_ids,
        required_structure_ids=(
            "complete-finite-paired-root-score",
            "coupled-two-view-discrepancy-in-score",
        ),
        numerical_view_ids=views,
        structural_tolerances=tuple(
            sorted(
                (
                    NamedDecimal(q.quantity_id, d / 8, q.native_unit)
                    for q, d in zip(
                        output_quantities(), FiniteResponseLawScienceSpec().delta, strict=True
                    )
                ),
                key=lambda v: v.value_id,
            )
        ),
        computability_envelope_id=system.computability_envelopes[0].envelope_id,
    )
    ledger = CandidateFamilyLedger(
        f"{prefix}.family",
        ObjectIdentity.from_record(system.system_id, system),
        calibration,
        prefix,
        "1.0.0",
        LawMethodKind.NONLINEAR_LOCAL,
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        obligations.chart_ids[0],
        axes,
        ObjectIdentity.from_record(binding.binding_id, binding),
        claim,
        obligations,
        (
            CandidateFamilyMember(
                f"{prefix}.frozen",
                config,
                0,
                CandidateRosterDisposition.ASSESS_REQUIRED,
                (),
            ),
        ),
        ('precalibration-fitted-law-nomination-no-refit',),
        ("finite-rank30-complete-root-calibration",),
        (config.object_id,),
        selector_capability,
        config,
        selector_implementation,
        f"{prefix}.singleton",
        "separate-boundary-no-winner-selection",
        ("calibration-fixed-parent-allocation",),
        "One frozen candidate; no comparison or calibration-dependent selection.",
        OutcomeAccess.EVALUATOR_REVEAL,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    return scope, binding, ledger


def candidate_evidence(
    ledger: CandidateFamilyLedger,
    binding: ClaimUnitBinding,
    publication: CandidatePayloadPublicationReceipt,
    receipt: CandidateMethodEvidenceReceipt,
    evidence: tuple[EvidenceLink, ...],
    *,
    boundary: str,
) -> LawCandidateEvidence:
    decoder = FiniteResponseLawLowerDecoder() if boundary == "lower" else FiniteResponseLawConditionalDecoder()
    member = ledger.members[0]
    return LawCandidateEvidence(
        f"evidence.{member.candidate_id}",
        member.candidate_id,
        ledger.system,
        ledger.dataset_or_projection,
        member.config,
        ledger.method_key,
        ledger.method_version,
        ledger.method_kind,
        ledger.representation_kind,
        publication.candidate_evaluator,
        ledger.axis_map,
        ledger.claim_unit_binding,
        binding.independent_unit_instance_ids,
        ledger.claim_template,
        ledger.obligation_template,
        (receipt,),
        publication,
        evidence,
        ledger.outcome_access,
        ledger.parent_visibility_ceilings,
        ledger.visibility_ceiling,
        ExtensionBinding(
            decoder.extension_namespace,
            decoder.payload_schema,
            publication.content_sha256,
        ),
    )

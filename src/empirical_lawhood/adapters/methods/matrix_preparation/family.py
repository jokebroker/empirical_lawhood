"""Finite-chart claim metadata for the shared candidate-family owners."""

from decimal import Decimal

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
from empirical_lawhood.kernel.references import NamedDecimal, QuantityBound
from empirical_lawhood.kernel.serialization import ExtensionBinding
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding, QualificationScopeSpec
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import DEVELOPMENT, PARENTS, preparation_roots
from .contracts import CANDIDATES, PreparationMethodConfig
from .law import FEATURE_QUANTITIES, LAW_KEY, PreparationLawDecoder, PreparationLawPayload
from .qualification import SCREEN_EVIDENCE


def candidate_id(context: str, candidate: str) -> str:
    if context not in ("assembling", "prepared") or candidate not in CANDIDATES:
        raise ValueError("finite candidate changes its frozen context/model roster")
    return f"{LAW_KEY}.{context}.{candidate}"


def preparation_family(
    *,
    system: SystemSpec,
    config: PreparationMethodConfig,
    context: str,
    projection: ObjectIdentity,
    selector_capability: ObjectIdentity,
    selector_implementation: ObjectIdentity,
) -> tuple[QualificationScopeSpec, ClaimUnitBinding, CandidateFamilyLedger]:
    prefix = f"{LAW_KEY}.{context}"
    unit_ids = tuple(
        sorted(
            r.physical_unit_id
            for r in preparation_roots()
            if r.context == context and r.development_role == "screen"
        )
    )
    views = tuple(sorted(v.view_id for v in system.numerical_views))
    if len(unit_ids) != 16 or len(views) != 2:
        raise ValueError("finite family changes the sixteen exposed held roots or numerical twins")
    axes = LawCandidateAxisMap(
        f"{prefix}.axes",
        tuple(
            LawCandidateAxisBinding(
                f"{prefix}.axis.{p}",
                f"{prefix}.version.{p}",
                f"{DEVELOPMENT}.{context}.{p}",
                views,
                ("fixed-three-sign-five-readout-observable-response",),
                (
                    "arbitrary-amplitude-response",
                    "arbitrary-input-recurrence",
                    "physical-transport",
                    "protected-preparation-benefit",
                ),
            )
            for p in sorted(PARENTS)
        ),
    )
    scope = QualificationScopeSpec(
        f"{prefix}.scope",
        f"{prefix}.claim",
        f"{prefix}.exposed-screen-population",
        system.independent_unit.unit_id,
        unit_ids,
        tuple(sorted((*axes.denominator_member_ids, *views))),
        "whole-original-stochastic-root",
        system.independent_unit.unit_id,
        f"{prefix}.single-handoff-fixed-chart",
        EvidenceCeiling.LOCAL_LAW,
    )
    binding = ClaimUnitBinding(
        f"{prefix}.claim-units",
        scope.claim_id,
        ObjectIdentity.from_record(scope.scope_id, scope),
        unit_ids,
        scope.aggregation_level_id,
    )
    assumptions = (
        "development-crossfit-interval-has-no-protected-coverage-guarantee",
        "finite-action-map-is-not-an-amplitude-polynomial",
        "nine-causal-observations-at-common-handoff",
        "original-exposed-stochastic-roots-are-the-independent-units",
    )
    claim = CandidateClaimTemplate(
        f"{prefix}.claim-template",
        f"{prefix}.qualification",
        None,
        scope.claim_id,
        f"{prefix}.law",
        "Observable finite-chart native response after a completed preparation; development feasibility only.",
        "Native absolute X displacement, baseline, odd and even response at 64,128,192,256,320 ticks from the observed common handoff.",
        "Exposed sixteen-root screen and all fixed model attempts; independent prospective calibration and validation are separate unentered claims.",
        CausalStrength.SIMULATOR_INTERVENTION,
        EvidenceCeiling.LOCAL_LAW,
        tuple(sorted((*FEATURE_QUANTITIES, *system.relation.action_quantity_ids))),
        system.relation.receiver_quantity_ids,
        assumptions,
        False,
    )
    action = next(
        q for q in system.quantities if q.quantity_id in system.relation.action_quantity_ids
    )
    obligations = LawObligationTemplate(
        template_id=f"{prefix}.obligation-template",
        obligations_id=f"{prefix}.obligations",
        support_id=f"{prefix}.support",
        validity_id=f"{prefix}.validity",
        uncertainty_id=f"{prefix}.uncertainty",
        closure_id=f"{prefix}.closure",
        structural_convergence_id=f"{prefix}.convergence",
        computability_id=f"{prefix}.computability",
        independent_unit_id=system.independent_unit.unit_id,
        physical_unit_count=16,
        nested_numerical_view_count=2,
        information_cutoff_id=f"{prefix}.observed-handoff-cutoff",
        chart_ids=(f"{prefix}.finite-action-chart",),
        denominator_cell_ids=axes.denominator_member_ids,
        action_bounds=(
            QuantityBound(
                f"{prefix}.force-bound",
                action.quantity_id,
                action.native_unit,
                Decimal(-8),
                Decimal(8),
            ),
        ),
        validity_domain_ids=(scope.locality_scope_id,),
        assumption_ids=assumptions,
        uncertainty_method_key="state-dependent-rootwise-full-menu-development-envelope",
        uncertainty_confidence_level=Decimal(".95"),
        interval_quantity_ids=system.relation.receiver_quantity_ids,
        uncertainty_limitation_codes=(
            "DEVELOPMENT_CROSSFIT_NOT_A_PROTECTED_COVERAGE_GUARANTEE",
            "EMPIRICAL_RESIDUAL_BOUND_NONADDITIVE",
        ),
        falsifiers=(
            FalsifierObligationTemplate(
                f"{prefix}.wrong-sign",
                FalsifierKind.WRONG_ACTION,
                LAW_KEY,
                "Swap NEG and POS predictions with observed futures and whole-root pairing retained.",
                "At least eight informative held roots and normalized wrong-sign MSE increase at least 1e-4.",
            ),
        ),
        recurrence_cell_ids=(f"{prefix}.fixed-handoff-four-audits",),
        exchange_factor_ids=("native-force-sign",),
        retained_history_ids=system.relation.history_quantity_ids,
        required_structure_ids=(
            "complete-absolute-finite-action-response",
            "paired-native-view-agreement",
        ),
        numerical_view_ids=views,
        structural_tolerances=(
            NamedDecimal(
                "native-absolute-paired-error", Decimal(".0078125"), "hilbert-schmidt-native"
            ),
            NamedDecimal("native-odd-paired-error", Decimal(".00390625"), "hilbert-schmidt-native"),
        ),
        computability_envelope_id=system.computability_envelopes[0].envelope_id,
    )
    config_id = ObjectIdentity.from_record(config.config_id, config)
    ledger = CandidateFamilyLedger(
        f"{prefix}.family",
        ObjectIdentity.from_record(system.system_id, system),
        projection,
        LAW_KEY,
        "1.0.0",
        LawMethodKind.NONLINEAR_LOCAL,
        LawRepresentationKind.FINITE_ACTION_OPERATOR,
        obligations.chart_ids[0],
        axes,
        ObjectIdentity.from_record(binding.binding_id, binding),
        claim,
        obligations,
        tuple(
            CandidateFamilyMember(
                candidate_id(context, c),
                config_id,
                rank,
                CandidateRosterDisposition.ASSESS_REQUIRED,
                (),
            )
            for rank, c in enumerate(CANDIDATES)
        ),
        ("whole-root-fit-folds-mean-loss-ridge",),
        ("frozen-absolute-and-odd-prediction-tolerances",),
        (config.config_id,),
        selector_capability,
        config_id,
        selector_implementation,
        f"{prefix}.all-three-observable-candidates",
        "each-model-own-development-full-menu-envelope",
        ("development-split-and-fold-sha256",),
        "Equal nine-observation acquisition cost; constant-gain, direct-linear, then direct-quadratic arithmetic/storage cost. No held prediction loss tie break.",
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        (VisibilityCeiling.OUTCOME_VISIBLE,),
        VisibilityCeiling.OUTCOME_VISIBLE,
    )
    return scope, binding, ledger


def preparation_candidate_evidence(
    ledger: CandidateFamilyLedger,
    binding: ClaimUnitBinding,
    payload: PreparationLawPayload,
    publication: CandidatePayloadPublicationReceipt,
    metrics: tuple[NamedDecimal, ...],
    evidence_links: tuple[EvidenceLink, ...],
) -> LawCandidateEvidence:
    candidate = candidate_id(payload.context, payload.candidate)
    member = next(m for m in ledger.members if m.candidate_id == candidate)
    if (
        publication.content_sha256 != payload.fingerprint()
        or ledger.claim_unit_binding != ObjectIdentity.from_record(binding.binding_id, binding)
    ):
        raise ValueError(
            "candidate evidence changes the published payload or original root denominator"
        )
    evidence_ids = tuple(sorted(e.link_id for e in evidence_links))
    artifacts = tuple(sorted({a for e in evidence_links for a in e.artifact_ids}))
    decoder = PreparationLawDecoder()
    return LawCandidateEvidence(
        evidence_id=f"evidence.{candidate}",
        candidate_id=candidate,
        system=ledger.system,
        dataset_or_projection=ledger.dataset_or_projection,
        config=member.config,
        method_key=ledger.method_key,
        method_version=ledger.method_version,
        method_kind=ledger.method_kind,
        representation_kind=ledger.representation_kind,
        candidate_evaluator=publication.candidate_evaluator,
        axis_map=ledger.axis_map,
        claim_unit_binding=ledger.claim_unit_binding,
        physical_independent_unit_ids=binding.independent_unit_instance_ids,
        claim_template=ledger.claim_template,
        obligation_template=ledger.obligation_template,
        method_receipts=(
            CandidateMethodEvidenceReceipt(
                f"operands.{candidate}", SCREEN_EVIDENCE, metrics, artifacts, evidence_ids, ()
            ),
        ),
        payload_publication=publication,
        evidence_links=evidence_links,
        outcome_access=ledger.outcome_access,
        parent_visibility_ceilings=ledger.parent_visibility_ceilings,
        visibility_ceiling=ledger.visibility_ceiling,
        candidate_extension=ExtensionBinding(
            decoder.extension_namespace, payload.SCHEMA, payload.fingerprint()
        ),
    )

"""development metadata and evidence adapters for the existing response-law family owners."""

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
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import ResponseGeometryDevelopmentNativeConfig
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import PARENTS
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

from .development_law import ResponseGeometryDevelopmentAffineLawDecoder, ResponseGeometryDevelopmentAffineLawPayload, DEVELOPMENT_INVOCATION_QUANTITY, DEVELOPMENT_LATENT_QUANTITIES, DEVELOPMENT_LAW_KEY
from .development_models import ResponseGeometryDevelopmentModelGroup, REPRESENTATIONS
from .development_qualification import DEVELOPMENT_SCREEN_EVIDENCE
from .development_records import ResponseGeometryDevelopmentMethodConfig


def response_geometry_development_candidate_id(context: str, representation: str) -> str:
    if context not in ("assembling", "prepared") or representation not in REPRESENTATIONS:
        raise ValueError("development candidate is outside its frozen context/representation roster")
    return f"development.affine.{context}.{representation}"


def response_geometry_development_family(
    *,
    system: SystemSpec,
    source: ResponseGeometryDevelopmentNativeConfig,
    config: ResponseGeometryDevelopmentMethodConfig,
    projection: ObjectIdentity,
    groups: tuple[ResponseGeometryDevelopmentModelGroup, ...],
    selector_capability: ObjectIdentity,
    selector_implementation: ObjectIdentity,
) -> tuple[QualificationScopeSpec, ClaimUnitBinding, CandidateFamilyLedger]:
    """Bind actual fit complexity and all 32 validation units; no validation loss selects.

    The design fixes this rule before contact. Runtime supplies authenticated fit
    groups and the validation projection identity, never a selected subset.
    """
    if tuple(g.representation for g in groups) != REPRESENTATIONS:
        raise ValueError("development qualification requires all five representation attempts")
    context = groups[0].context
    if any(g.context != context for g in groups):
        raise ValueError("development qualification cannot pool contexts")
    prefix = f"development.affine.{context}"
    unit_ids = tuple(
        sorted(
            source.physical_unit_id(r)
            for r in source.roots
            if r.context == context and 32 <= r.index < 64
        )
    )
    views = tuple(sorted(v.view_id for v in system.numerical_views))
    if len(unit_ids) != 32 or len(views) != 2:
        raise ValueError("development qualification changes its independent-unit/numerical-view scope")
    axes = LawCandidateAxisMap(
        f"{prefix}.axes",
        tuple(
            LawCandidateAxisBinding(
                f"{prefix}.axis.{parent}",
                f"{prefix}.version.{parent}",
                f"development.{context}.{parent}",
                views,
                ("finite-signed-affine-response",),
                ("joint-sink-effort-distribution", "online-stopped-policy", "physical-transport"),
            )
            for parent in sorted(PARENTS)
        ),
    )
    scope = QualificationScopeSpec(
        f"{prefix}.scope",
        f"{prefix}.claim",
        f"{prefix}.validation-population",
        system.independent_unit.unit_id,
        unit_ids,
        tuple(sorted((*axes.denominator_member_ids, *views))),
        "whole-stochastic-root",
        system.independent_unit.unit_id,
        f"{prefix}.nine-sampled-invocation-offsets",
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
        "development-calibration-finite-design-model-conditional",
        "development-native-recorded-coordinate-estimand",
        "development-no-online-stopping-or-continuous-window-claim",
        "development-payload-selected-dimension-and-frozen-feature-map",
    )
    claim = CandidateClaimTemplate(
        f"{prefix}.claim-template",
        f"{prefix}.qualification",
        None,
        scope.claim_id,
        f"{prefix}.law",
        "Finite simulator-local affine response on the declared nine invocation offsets.",
        "Signed native X displacement at 320 ticks conditional on the observed invocation state and short-pulse-response force word.",
        'All frozen development empirical measurement-through-law-qualification obligations. Development only, with fresh evidence required for protected descendants.',
        CausalStrength.SIMULATOR_INTERVENTION,
        EvidenceCeiling.LOCAL_LAW,
        tuple(
            sorted(
                (*DEVELOPMENT_LATENT_QUANTITIES, DEVELOPMENT_INVOCATION_QUANTITY, *system.relation.action_quantity_ids)
            )
        ),
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
        physical_unit_count=32,
        nested_numerical_view_count=2,
        information_cutoff_id=f"{prefix}.invocation-cutoff",
        chart_ids=(f"{prefix}.short-pulse-response-chart",),
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
        uncertainty_method_key="development-whole-root-split-calibration",
        uncertainty_confidence_level=Decimal(".9"),
        interval_quantity_ids=system.relation.receiver_quantity_ids,
        uncertainty_limitation_codes=("DEVELOPMENT_MODEL_CONDITIONAL_NONADDITIVE_EMPIRICAL_BOUND",),
        falsifiers=(
            FalsifierObligationTemplate(
                f"{prefix}.wrong-sign",
                FalsifierKind.WRONG_ACTION,
                DEVELOPMENT_LAW_KEY,
                "Swap NEG/POS predictions while retaining observed responses and whole-root pairing.",
                "Eight informative roots and normalized wrong-sign MSE increase at least 1e-4.",
            ),
        ),
        recurrence_cell_ids=tuple(f"{prefix}.offset.{i}" for i in range(384, 513, 16)),
        exchange_factor_ids=("short-pulse-response-force-sign",),
        retained_history_ids=system.relation.history_quantity_ids,
        required_structure_ids=("complete-affine-initial-state", "native-response-refinement"),
        numerical_view_ids=views,
        structural_tolerances=(
            NamedDecimal("development-native-paired-difference", Decimal(".03125"), "hilbert-schmidt-native"),
        ),
        computability_envelope_id=system.computability_envelopes[0].envelope_id,
    )

    def complexity(group: ResponseGeometryDevelopmentModelGroup) -> tuple[int, int, int, float, str]:
        if not group.models or group.selected_index is None:
            return (100, 100, 1000000, 0.0, group.representation)
        selected = group.scores[group.selected_index]
        return (
            selected.dimension,
            int(selected.split_at_force_off),
            group.models[0].feature_map.center.size,
            -selected.ridge,
            group.representation,
        )

    config_identity = ObjectIdentity.from_record(config.config_id, config)
    ledger = CandidateFamilyLedger(
        family_id=f"{prefix}.family",
        system=ObjectIdentity.from_record(system.system_id, system),
        dataset_or_projection=projection,
        method_key=DEVELOPMENT_LAW_KEY,
        method_version="1.0.0",
        method_kind=LawMethodKind.LOCAL_STATE_SPACE,
        representation_kind=LawRepresentationKind.LOCAL_STATE_SPACE,
        chart_id=obligations.chart_ids[0],
        axis_map=axes,
        claim_unit_binding=ObjectIdentity.from_record(binding.binding_id, binding),
        claim_template=claim,
        obligation_template=obligations,
        members=tuple(
            CandidateFamilyMember(
                response_geometry_development_candidate_id(context, g.representation),
                config_identity,
                rank,
                CandidateRosterDisposition.ASSESS_REQUIRED,
                (),
            )
            for rank, g in enumerate(sorted(groups, key=complexity))
        ),
        candidate_generation_rule_ids=("development-fit-only-four-fold-one-standard-error",),
        selection_threshold_ids=("development-frozen-finite-law-qualification-profile",),
        development_input_ids=(config.config_id,),
        selector_capability=selector_capability,
        selector_config=config_identity,
        selector_implementation=selector_implementation,
        multiplicity_family_id=f"{prefix}.all-five-representations",
        multiplicity_rule_id="development-calibration-root-maximum-over-five-representations",
        randomness_seed_ids=("development-alternative-pcg64dxsm-20260907",),
        tie_break_rule="Least dimension, unsplit first, fewer raw feature coordinates, greater ridge, stable representation ID; no held validation loss.",
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        parent_visibility_ceilings=(VisibilityCeiling.DEVELOPMENT_ONLY,),
        visibility_ceiling=VisibilityCeiling.DEVELOPMENT_ONLY,
    )
    return scope, binding, ledger


def response_geometry_development_candidate_evidence(
    ledger: CandidateFamilyLedger,
    claim_units: ClaimUnitBinding,
    payload: ResponseGeometryDevelopmentAffineLawPayload,
    publication: CandidatePayloadPublicationReceipt,
    metrics: tuple[NamedDecimal, ...],
    evidence_links: tuple[EvidenceLink, ...],
) -> LawCandidateEvidence:
    """Attach raw operands and authoritative publication; emit no generic verdict."""
    candidate_id = response_geometry_development_candidate_id(payload.context, payload.representation)
    member = next((m for m in ledger.members if m.candidate_id == candidate_id), None)
    if (
        member is None
        or publication.content_sha256 != payload.fingerprint()
        or ledger.claim_unit_binding
        != ObjectIdentity.from_record(claim_units.binding_id, claim_units)
    ):
        raise ValueError("development evidence changes its frozen roster or published model")
    evidence_ids = tuple(sorted(e.link_id for e in evidence_links))
    artifact_ids = tuple(sorted({a for e in evidence_links for a in e.artifact_ids}))
    decoder = ResponseGeometryDevelopmentAffineLawDecoder()
    return LawCandidateEvidence(
        evidence_id=f"evidence.{candidate_id}",
        candidate_id=candidate_id,
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
        physical_independent_unit_ids=claim_units.independent_unit_instance_ids,
        claim_template=ledger.claim_template,
        obligation_template=ledger.obligation_template,
        method_receipts=(
            CandidateMethodEvidenceReceipt(
                f"operands.{candidate_id}",
                DEVELOPMENT_SCREEN_EVIDENCE,
                metrics,
                artifact_ids,
                evidence_ids,
                (),
            ),
        ),
        payload_publication=publication,
        evidence_links=evidence_links,
        outcome_access=ledger.outcome_access,
        parent_visibility_ceilings=ledger.parent_visibility_ceilings,
        visibility_ceiling=ledger.visibility_ceiling,
        candidate_extension=ExtensionBinding(
            decoder.extension_namespace,
            payload.SCHEMA,
            payload.fingerprint(),
        ),
    )

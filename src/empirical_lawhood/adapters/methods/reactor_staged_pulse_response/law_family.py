"Explicit staged-pulse claim/denominator bindings for the existing singleton family owner."

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
)
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.words import CHART
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.identification import LawMethodKind
from empirical_lawhood.kernel.laws import CausalStrength, LawRepresentationKind
from empirical_lawhood.kernel.obligations import FalsifierKind
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.planning.identification_evidence import ClaimUnitBinding, QualificationScopeSpec
from .config import ClassicalDesign, roots
from .law_payload import ClassicalLawPayload
from .measurement import epsilon, receiver_ids
from .science import CUTOFF, METHOD, UNIT, staged_pulse_reactor_system

ASSUMPTIONS = tuple(
    sorted(
        (
            "fresh-assigned-simulation-roots",
            "fixed-unshifted-exploration-or-exact-first-induced-history",
            "causal-delayed-observation-prefix",
            "complete-precommitted-native-feed-word",
            "matched-counterfactual-reference",
            "raw-two-view-correspondence-before-receiver-map",
            "no-physical-or-full-benchmark-transport",
        )
    )
)


def law_family(
    payload: ClassicalLawPayload, capability: ObjectIdentity, owner: ObjectIdentity
) -> tuple[ClaimUnitBinding, CandidateFamilyLedger]:
    c, stem = payload.recipe.bound.coordinate, payload.stem
    system = staged_pulse_reactor_system(c)
    views = tuple(v.view_id for v in system.numerical_views)
    receivers = tuple(sorted(receiver_ids(c)))
    assigned = roots(payload.block, "qualification")
    family_size = {"base-menu-comparison": 72, "expanded-menu-comparison": 36, "staged-sequence-comparison": 5}[payload.block]
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
                            "other-histories",
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
        assigned,
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
        assigned,
        scope.aggregation_level_id,
    )
    claim = CandidateClaimTemplate(
        f"{stem}.claim-template",
        f"{stem}.qualification",
        None,
        scope.claim_id,
        f"{stem}.law",
        f"{payload.recipe.arm} bounds the declared raw/saturated responses of {c.pulse.word_id} under {c.kind}/{c.context} history with its exact response and guard windows.",
        "A finite native feed word at fixed jacket, matched references and two numerical views; empirical recipes use only interventions and permitted causal observations.",
        "No additional action, history, hidden coefficient, physical or full-batch control claim.",
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
        uncertainty_method_key=f"frozen-finite-row-cp-family{family_size}",
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
                "An absent contact, changed native history/stage, raw numerical mismatch, failed response map, uncovered thermal guard or unsafe prefix/action defeats the joint root event.",
                f"All 96 assigned roots, >=95 joint successes, CP lower at .05/{family_size} >=.90, zero unsafe roots and the exact development/calibration prerequisite.",
            ),
        ),
        recurrence_cell_ids=(CHART,),
        exchange_factor_ids=("native-feed-word", "native-integrator-timestep"),
        retained_history_ids=system.relation.history_quantity_ids,
        required_structure_ids=system.computability_envelopes[0].required_structure_ids,
        numerical_view_ids=views,
        structural_tolerances=tuple(
            sorted(
                (NamedDecimal(key, epsilon(c, key), "K") for key in receivers),
                key=lambda v: v.value_id,
            )
        ),
        computability_envelope_id=system.computability_envelopes[0].envelope_id,
    )
    design = ClassicalDesign()
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
                payload.nomination,
                0,
                CandidateRosterDisposition.ASSESS_REQUIRED,
                (),
            ),
        ),
        ("32-retained-development-roots",),
        ("fixed-outward-extrema-and-prespecified-calibration",),
        (c.context,),
        capability,
        ObjectIdentity.from_record(design.config_id, design),
        owner,
        f"{stem}.singleton",
        f"one-row-from-predeclared-family-of-{family_size}",
        ("all-assigned-qualification-roots",),
        "Assess this frozen relation independently; do not refit, drop roots or require an alternative word to qualify.",
        OutcomeAccess.EVALUATOR_REVEAL,
        (VisibilityCeiling.PROSPECTIVE,),
        VisibilityCeiling.PROSPECTIVE,
    )
    return binding, ledger

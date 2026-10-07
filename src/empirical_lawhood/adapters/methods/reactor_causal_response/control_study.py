"""Assemble Empirical finite inputs for the existing native-magnitude controller owner."""

from decimal import Decimal as D

from empirical_lawhood.kernel.evidence import EvidenceCeiling, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.controller_study import ActHoldActionCandidate, CandidatePriorityOrientation, DeliveryControllerStudy, ActHoldControllerSynthesisPlan, DeliveryEquivalenceSpec, ActHoldActionCandidateChart, ControllerInstanceBinding, ImplementationBinding, ImplementationRole, NativeCandidateKind, OnlineSupportMonitorSpec, ControllerActionBinding, ReidentificationTriggerSpec
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionAdmissionCandidateCell
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionSpec, FiniteCertificateReachabilitySpec, FiniteCertificateAdmissionReceiptCorpus, derive_finite_certificate_admission_comparison, derive_finite_certificate_reachability_comparison
from empirical_lawhood.runtime.controller_runtime import RuntimeObservation
from empirical_lawhood.adapters.control.controller_authoring import DeliveryControllerAuthor, DeliveryControllerAuthoringRequest


def control_study(
    *,
    system: SystemSpec,
    corpus: FiniteCertificateAdmissionReceiptCorpus,
    action_bindings: tuple[ControllerActionBinding, ...],
    implementations: tuple[ImplementationBinding, ...],
    observation: RuntimeObservation,
    frozen_recipe: ObjectIdentity,
    compiler_release_id: str,
    decision_budget_seconds: D,
    prospective_evaluation: ObjectIdentity | None = None,
) -> DeliveryControllerStudy:
    """No fitted parameters or selector here; refusal remains a valid outcome.

    Causal action evidence and installed implementation identities must be
    supplied by their actual owners, not a reference-world fixture.
    """
    stem = corpus.corpus_id
    plan = corpus.plan
    roles = {v.role: v.binding_id for v in implementations}
    if (ImplementationRole.OUTCOME_EVALUATOR in roles) != (prospective_evaluation is not None):
        raise ValueError("Empirical prospective binding requires its separate exact controller-use owner")
    if prospective_evaluation is not None and prospective_evaluation.object_schema != (
        'empirical-lawhood/planning/trajectory-controller-evaluation-plan'
    ):
        raise ValueError("Empirical use requires its frozen prepared controller-use census")
    candidates = tuple(
        ReceiptAdmissionAdmissionCandidateCell(
            f"{stem}.{f.action_binding_id}.cell",
            ObjectIdentity.from_record(f.action_binding_id, f),
            ObjectIdentity.from_record(s.cell_id, s),
            tuple(
                c.coordinate_id
                for c in plan.coordinates
                if c.action_fibre == ObjectIdentity.from_record(f.action_binding_id, f)
                and c.support_cell == ObjectIdentity.from_record(s.cell_id, s)
            ),
        )
        for f in plan.action_fibres
        for s in plan.support_cells
    )
    admission = FiniteCertificateAdmissionSpec(
        f"{stem}.admission",
        corpus,
        plan.nominal_denominator_member_id,
        plan.atlas.laws[0].relation.receiver_quantity_ids,
        candidates,
        EvidenceCeiling.ADMISSION,
        VisibilityCeiling.PROSPECTIVE,
    )
    admitted = derive_finite_certificate_admission_comparison(admission)
    reachability = FiniteCertificateReachabilitySpec(
        f"{stem}.reachability",
        admission,
        admitted,
        plan.initial_set_id,
        EvidenceCeiling.ADMISSION,
        VisibilityCeiling.PROSPECTIVE,
    )
    reachable = derive_finite_certificate_reachability_comparison(reachability)
    utility = corpus.utility_receipts[0]
    chart = ActHoldActionCandidateChart(
        f"{stem}.chart",
        ObjectIdentity.from_record(plan.model_set.model_set_id, plan.model_set),
        plan.nominal_denominator_member_id,
        utility.utility_definition,
        utility.direction,
        utility.minimum_utility,
        CandidatePriorityOrientation.EARLIER_WINS,
        tuple(
            ActHoldActionCandidate(
                f"{cell.candidate_cell_id}.candidate",
                f"{stem}.decision",
                ObjectIdentity.from_record(cell.candidate_cell_id, cell),
                i,
                NativeCandidateKind.ACT,
            )
            for i, cell in enumerate(candidates)
        ),
    )
    synthesis = ActHoldControllerSynthesisPlan(
        f"{stem}.synthesis",
        ObjectIdentity.from_record(plan.atlas.atlas_id, plan.atlas),
        ObjectIdentity.from_record(corpus.corpus_id, corpus),
        ObjectIdentity.from_record(admitted.comparison_id, admitted),
        ObjectIdentity.from_record(reachable.comparison_id, reachable),
        chart.model_set,
        chart,
        roles[ImplementationRole.ADMISSION_DERIVER],
        roles[ImplementationRole.REACHABILITY_DERIVER],
        roles[ImplementationRole.SYNTHESIZER],
        roles[ImplementationRole.OBSERVER],
        roles[ImplementationRole.ONLINE_GATE_EVALUATOR],
        roles[ImplementationRole.DELIVERY],
        tuple(v.value_id for v in observation.values),
        (
            OnlineSupportMonitorSpec(
                f"{stem}.exact-inputs",
                tuple(s.cell_id for s in plan.support_cells),
                "The finite instance observer requires identical committed primary inputs, table and request; later handoff can cancel, never reselect.",
                "flh-committed-input-changed",
            ),
        ),
        (
            ReidentificationTriggerSpec(
                f"{stem}.invalidate",
                "Changed committed input or missing exact native delivery cancels use; no online refit.",
                True,
                "flh-committed-input-changed",
            ),
        ),
        system.authority_policy.policy_id,
        decision_budget_seconds,
        decision_budget_seconds,
        EvidenceCeiling.ADMISSION,
        VisibilityCeiling.PROSPECTIVE,
    )
    first = corpus.reachability_receipts[0].request
    joint = first.response_set.joint_calibration
    instance = ControllerInstanceBinding(
        f"{stem}.instance",
        frozen_recipe,
        tuple(
            sorted({law.evaluator.payload for law in plan.atlas.laws}, key=lambda a: a.artifact_id)
        ),
        (ObjectIdentity.from_record(joint.calibration_id, joint),),
        first.response_set.public_handoff,
        ObjectIdentity.from_record(first.task.functional_id, first.task),
        first.response_set.frame_translation,
        ObjectIdentity.from_record(observation.observation_id, observation),
        plan.information_cutoff,
    )
    request = DeliveryControllerAuthoringRequest(
        f"{stem}.authoring",
        f"{stem}.programme",
        compiler_release_id,
        system,
        action_bindings,
        admission,
        reachability,
        synthesis,
        implementations,
        None,
        DeliveryEquivalenceSpec(
            f"{stem}.delivery-equivalence",
            True,
            True,
            NamedDecimal("exact-per-event-native-value", D(0), "per-event-native-unit"),
        ),
        False,
        instance,
        prospective_evaluation,
    )
    return DeliveryControllerAuthor.author(request)

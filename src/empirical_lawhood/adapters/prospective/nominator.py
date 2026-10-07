"""Deterministic advisory nomination of fresh-evidence questions."""

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import AssignmentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.exploration import (
    ExploratoryFinding,
    HypothesisSet,
    OutcomeInterpretation,
    OutcomeInterpretationKind,
    ProspectiveNomination,
)
from empirical_lawhood.planning.prospective import (
    DesignNominationSelection,
    NominationDecisionKind,
    NominationDisposition,
    NominationObligations,
    ProspectiveDesignContext,
    ProspectiveDesignKind,
    ProspectiveNominationDecision,
    nomination_obligations_extension,
)


@dataclass(frozen=True, slots=True)
class NominationBatch:
    """Decision plus separately materializable, fingerprint-bound obligations."""

    decision: ProspectiveNominationDecision
    obligations: tuple[NominationObligations, ...]


class ProspectiveNominator:
    """Translate fixed outcome-visible hypotheses into advisory questions only."""

    capability_key = "prospective.fresh-evidence-nominator"
    capability_version = "1.0.0"

    def nominate(
        self,
        *,
        hypotheses: HypothesisSet,
        findings: tuple[ExploratoryFinding, ...],
        system: SystemSpec,
        context: ProspectiveDesignContext,
    ) -> NominationBatch:
        self._validate_inputs(hypotheses, findings, system, context)
        if context.evidence_contract is None:
            selections = tuple(
                DesignNominationSelection(
                    design_kind=kind,
                    disposition=NominationDisposition.BLOCKED,
                    nomination_id=None,
                    reason_codes=context.no_acquisition_reason_codes,
                )
                for kind in context.design_kinds
            )
            return NominationBatch(
                decision=ProspectiveNominationDecision(
                    decision_id=f"nomination-decision.{context.context_id}",
                    source_hypothesis_set=ObjectIdentity.from_record(
                        hypotheses.hypothesis_set_id, hypotheses
                    ),
                    kind=NominationDecisionKind.STOP_NO_ACQUISITION,
                    selections=selections,
                    nominations=(),
                    stop_reason_codes=context.no_acquisition_reason_codes,
                    nominator_key=self.capability_key,
                    nominator_version=self.capability_version,
                    outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                    visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                ),
                obligations=(),
            )

        nominations: list[ProspectiveNomination] = []
        obligations: list[NominationObligations] = []
        hypothesis_ids = tuple(h.hypothesis_id for h in hypotheses.hypotheses)
        source_finding_ids = tuple(f.finding_id for f in findings)
        for kind in context.design_kinds:
            slug = kind.value.lower().replace("_", "-")
            nomination_id = f"nomination.{context.context_id}.{slug}"
            objective = next(item for item in context.objectives if item.design_kind is kind)
            obligation = NominationObligations(
                obligations_id=f"nomination-obligations.{context.context_id}.{slug}",
                nomination_id=nomination_id,
                source_finding_ids=source_finding_ids,
                design_kind=kind,
                evidence_contract=context.evidence_contract,
                measurements=context.measurements,
                action_bounds=context.action_bounds,
                denominator_cell_ids=context.denominator_cell_ids,
                controls=context.controls,
                information_cutoff=context.information_cutoff,
                objective=objective,
                alternative_design_ids=context.alternative_design_ids,
                safety_constraint_ids=context.safety_constraint_ids,
                budget=context.budget,
                authority_action=context.authority_action,
                source_access=context.source_access,
                claim_proposition=context.claim_proposition,
                claim_estimand=context.claim_estimand,
                claim_promotion_rule=context.claim_promotion_rule,
                claim_assumption_ids=context.claim_assumption_ids,
                requested_rung=context.requested_rung,
                evidence_ceiling=context.evidence_ceiling,
                non_promotion_statement=(
                    "This outcome-visible nomination cannot promote its motivating finding; "
                    "only the separately sealed fresh experiment can earn its declared rung."
                ),
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                parent_visibility_ceiling=hypotheses.visibility_ceiling,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            )
            nomination = ProspectiveNomination(
                nomination_id=nomination_id,
                source_hypothesis_set=ObjectIdentity.from_record(
                    hypotheses.hypothesis_set_id, hypotheses
                ),
                relation=system.relation,
                question=self._question(kind, hypotheses),
                hypothesis_ids=hypothesis_ids,
                assignment=AssignmentSpec(
                    assignment_id=f"assignment.{context.context_id}.{slug}",
                    kind=context.assignment_kind,
                    independent_unit_id=system.independent_unit.unit_id,
                    action_quantity_ids=system.relation.action_quantity_ids,
                    mechanism=context.assignment_mechanism,
                    support_restriction_ids=context.assignment_support_restriction_ids,
                    randomization_unit_id=context.randomization_unit_id,
                ),
                measurement_quantity_ids=tuple(
                    measurement.quantity_id for measurement in context.measurements
                ),
                falsifier_ids=tuple(
                    sorted(
                        {
                            f"decisive-{slug}",
                            *(control.control.control_id for control in context.controls),
                        }
                    )
                ),
                admission_gate_ids=context.admission_gate_ids,
                precision_goals=context.precision_goals,
                outcome_interpretations=self._interpretations(context.context_id, slug),
                authority_action=context.authority_action,
                fresh_evidence_required=True,
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                parent_visibility_ceiling=hypotheses.visibility_ceiling,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                stop_conditions=tuple(
                    sorted(
                        {
                            "authority-not-granted",
                            "precision-unit-cap",
                            "safety-constraint-failed",
                            *(
                                goal.stopping_rule.lower().replace(" ", "-").replace(".", "")
                                for goal in context.precision_goals
                            ),
                        }
                    )
                ),
                extensions=(nomination_obligations_extension(obligation),),
            )
            nominations.append(nomination)
            obligations.append(obligation)
        nominations_tuple = tuple(sorted(nominations, key=lambda item: item.nomination_id))
        selections = tuple(
            DesignNominationSelection(
                design_kind=kind,
                disposition=NominationDisposition.SELECTED,
                nomination_id=next(
                    item.nomination_id
                    for item in nominations_tuple
                    if item.nomination_id.endswith(kind.value.lower().replace("_", "-"))
                ),
                reason_codes=("fresh-evidence-contract-complete",),
            )
            for kind in context.design_kinds
        )
        return NominationBatch(
            decision=ProspectiveNominationDecision(
                decision_id=f"nomination-decision.{context.context_id}",
                source_hypothesis_set=ObjectIdentity.from_record(
                    hypotheses.hypothesis_set_id, hypotheses
                ),
                kind=NominationDecisionKind.NOMINATE,
                selections=selections,
                nominations=nominations_tuple,
                stop_reason_codes=(),
                nominator_key=self.capability_key,
                nominator_version=self.capability_version,
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            ),
            obligations=tuple(sorted(obligations, key=lambda item: item.obligations_id)),
        )

    @staticmethod
    def _validate_inputs(
        hypotheses: HypothesisSet,
        findings: tuple[ExploratoryFinding, ...],
        system: SystemSpec,
        context: ProspectiveDesignContext,
    ) -> None:
        if context.system != ObjectIdentity.from_record(system.system_id, system):
            raise ValueError("prospective context binds another immutable system")
        if tuple(sorted(f.finding_id for f in findings)) != hypotheses.finding_ids:
            raise ValueError("nominator requires the exact findings bound by the hypothesis set")
        if not findings:
            raise ValueError("prospective nomination requires motivating findings")
        if any(f.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE for f in findings):
            raise ValueError("prospective nomination requires outcome-visible source findings")
        if hypotheses.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("prospective nomination cannot change the hypothesis ceiling")
        ProspectiveNominator._validate_measurements(context, system)
        ProspectiveNominator._validate_action_bounds(context, system)
        if context.information_cutoff.clock_id not in {clock.clock_id for clock in system.clocks}:
            raise ValueError("prospective cutoff uses an unknown system clock")

    @staticmethod
    def _validate_measurements(context: ProspectiveDesignContext, system: SystemSpec) -> None:
        quantities = {quantity.quantity_id: quantity for quantity in system.quantities}
        for requirement in context.measurements:
            try:
                requirement.validate_quantity(quantities[requirement.quantity_id])
            except KeyError as error:
                raise ValueError("prospective measurement is absent from the system") from error

    @staticmethod
    def _validate_action_bounds(context: ProspectiveDesignContext, system: SystemSpec) -> None:
        if context.evidence_contract is not None:
            action_quantities = {quantity.quantity_id: quantity for quantity in system.quantities}
            if {bound.quantity_id for bound in context.action_bounds} != set(
                system.relation.action_quantity_ids
            ):
                raise ValueError("prospective action bounds must cover the native action chart")
            for bound in context.action_bounds:
                if action_quantities[bound.quantity_id].native_unit != bound.native_unit:
                    raise ValueError("prospective action bound changes native units")

    @staticmethod
    def _question(kind: ProspectiveDesignKind, hypotheses: HypothesisSet) -> str:
        subject = " versus ".join(h.hypothesis_id for h in hypotheses.hypotheses)
        prompts = {
            ProspectiveDesignKind.SIMPLE_FACTORIAL: "Which predeclared factor exchange adjudicates",
            ProspectiveDesignKind.COVERAGE_EXPANSION: "Does fresh support expansion adjudicate",
            ProspectiveDesignKind.INFORMATION_RANK: "Does the native measurement design resolve",
            ProspectiveDesignKind.MODEL_DISCRIMINATION: "Which decisive intervention distinguishes",
        }
        return f"{prompts[kind]} {subject}?"

    @staticmethod
    def _interpretations(context_id: str, slug: str) -> tuple[OutcomeInterpretation, ...]:
        return tuple(
            sorted(
                (
                    OutcomeInterpretation(
                        interpretation_id=f"interpretation.{context_id}.{slug}.opposing",
                        kind=OutcomeInterpretationKind.OPPOSING,
                        criterion=(
                            "The decisive effect is absent or the competing explanation's "
                            "predeclared prediction is uniquely retained."
                        ),
                    ),
                    OutcomeInterpretation(
                        interpretation_id=f"interpretation.{context_id}.{slug}.supporting",
                        kind=OutcomeInterpretationKind.SUPPORTING,
                        criterion=(
                            "The fresh independent-unit effect and decisive falsifiers clear "
                            "the frozen precision and validity gates."
                        ),
                    ),
                    OutcomeInterpretation(
                        interpretation_id=f"interpretation.{context_id}.{slug}.unresolved",
                        kind=OutcomeInterpretationKind.UNRESOLVED,
                        criterion=(
                            "Support, precision, safety, validity or evaluator completeness "
                            "fails without uniquely favoring an alternative."
                        ),
                    ),
                ),
                key=lambda item: item.interpretation_id,
            )
        )

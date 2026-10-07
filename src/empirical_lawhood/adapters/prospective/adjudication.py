"""Immutable adjudication of exploratory hypotheses with fresh evidence."""

from __future__ import annotations

from dataclasses import replace

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.experiments import ExperimentSpec
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.planning.design import ExperimentProposal
from empirical_lawhood.planning.exploration import (
    HypothesisSet,
    OutcomeInterpretationKind,
    ProspectiveNomination,
)
from empirical_lawhood.planning.prospective import (
    HypothesisAdjudication,
    HypothesisAdjudicationDisposition,
    NominationObligations,
    ProspectiveAdjudication,
    ProspectiveEvidenceResult,
)


class ProspectiveAdjudicator:
    """Interpret a sealed evaluator result without rewriting any source object."""

    capability_key = "prospective.hypothesis-adjudicator"
    capability_version = "1.0.0"

    def adjudicate(
        self,
        *,
        hypotheses: HypothesisSet,
        nomination: ProspectiveNomination,
        obligations: NominationObligations,
        proposal: ExperimentProposal,
        experiment: ExperimentSpec,
        result: ProspectiveEvidenceResult,
    ) -> ProspectiveAdjudication:
        self._validate(hypotheses, nomination, obligations, proposal, experiment, result)
        hypothesis_ids = {hypothesis.hypothesis_id for hypothesis in hypotheses.hypotheses}
        favored = set(result.favored_hypothesis_ids)
        if not favored.issubset(hypothesis_ids):
            raise ValueError("fresh evaluator favored an unregistered hypothesis")
        unresolved = result.status in {
            ScientificStatus.MIXED,
            ScientificStatus.NOT_TESTED,
            ScientificStatus.PARTIAL,
            ScientificStatus.UNEVALUABLE,
        }
        rows: list[HypothesisAdjudication] = []
        for hypothesis in hypotheses.hypotheses:
            if unresolved:
                disposition = HypothesisAdjudicationDisposition.UNRESOLVED
                reasons = result.terminal_reason_codes
            elif hypothesis.hypothesis_id in favored:
                disposition = HypothesisAdjudicationDisposition.SUPPORTED
                reasons = ("favored-by-predeclared-fresh-interpretation",)
            else:
                disposition = HypothesisAdjudicationDisposition.OPPOSED
                reasons = ("opposed-by-predeclared-fresh-interpretation",)
            rows.append(
                HypothesisAdjudication(
                    adjudication_id=(
                        f"hypothesis-adjudication.{result.result_id}.{hypothesis.hypothesis_id}"
                    ),
                    hypothesis_id=hypothesis.hypothesis_id,
                    disposition=disposition,
                    reason_codes=reasons,
                )
            )
        return ProspectiveAdjudication(
            adjudication_id=f"prospective-adjudication.{result.result_id}",
            source_hypothesis_set=ObjectIdentity.from_record(
                hypotheses.hypothesis_set_id, hypotheses
            ),
            nomination=ObjectIdentity.from_record(nomination.nomination_id, nomination),
            proposal=ObjectIdentity.from_record(proposal.proposal_id, proposal),
            fresh_result=ObjectIdentity.from_record(result.result_id, result),
            hypothesis_adjudications=tuple(sorted(rows, key=lambda row: row.adjudication_id)),
            parent_fingerprints_preserved=True,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            parent_visibility_ceilings=(
                hypotheses.visibility_ceiling,
                nomination.visibility_ceiling,
                proposal.visibility_ceiling,
                result.visibility_ceiling,
            ),
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        )

    @staticmethod
    def _validate(
        hypotheses: HypothesisSet,
        nomination: ProspectiveNomination,
        obligations: NominationObligations,
        proposal: ExperimentProposal,
        experiment: ExperimentSpec,
        result: ProspectiveEvidenceResult,
    ) -> None:
        ProspectiveAdjudicator._validate_lineage(
            hypotheses, nomination, obligations, proposal, experiment
        )
        ProspectiveAdjudicator._validate_result(nomination, obligations, experiment, result)

    @staticmethod
    def _validate_lineage(
        hypotheses: HypothesisSet,
        nomination: ProspectiveNomination,
        obligations: NominationObligations,
        proposal: ExperimentProposal,
        experiment: ExperimentSpec,
    ) -> None:
        if nomination.source_hypothesis_set != ObjectIdentity.from_record(
            hypotheses.hypothesis_set_id, hypotheses
        ):
            raise ValueError("nomination does not bind the immutable hypothesis set")
        if proposal.nomination != ObjectIdentity.from_record(nomination.nomination_id, nomination):
            raise ValueError("proposal does not bind the immutable nomination")
        if obligations.nomination_id != nomination.nomination_id:
            raise ValueError("adjudication obligations bind another nomination")
        binding = next(
            (
                item
                for item in nomination.extensions
                if item.namespace == "prospective-nomination-obligations"
            ),
            None,
        )
        if binding is None or binding.payload_sha256 != obligations.fingerprint():
            raise ValueError("adjudication obligations differ from the frozen nomination")
        expected = replace(
            proposal.candidate_experiment,
            authorization_record_id=experiment.authorization_record_id,
            readiness=experiment.readiness,
        )
        if experiment != expected or experiment.authorization_record_id is None:
            raise ValueError("adjudication requires the separately authorized frozen experiment")

    @staticmethod
    def _validate_result(
        nomination: ProspectiveNomination,
        obligations: NominationObligations,
        experiment: ExperimentSpec,
        result: ProspectiveEvidenceResult,
    ) -> None:
        if result.experiment != ObjectIdentity.from_record(experiment.experiment_id, experiment):
            raise ValueError("fresh result belongs to another experiment")
        if result.claim_ids != tuple(claim.claim_id for claim in experiment.claims):
            raise ValueError("fresh result claim identities differ from the frozen experiment")
        if result.evidence_ceiling is not EvidenceCeiling.lowest(
            *(claim.evidence_ceiling for claim in experiment.claims)
        ):
            raise ValueError("fresh result evidence ceiling differs from the frozen claims")
        interpretation = next(
            (
                item
                for item in nomination.outcome_interpretations
                if item.interpretation_id == result.interpreted_outcome_id
            ),
            None,
        )
        if interpretation is None:
            raise ValueError("fresh result uses an undeclared terminal interpretation")
        expected_kind = {
            ScientificStatus.SUPPORTED: OutcomeInterpretationKind.SUPPORTING,
            ScientificStatus.NOT_SUPPORTED: OutcomeInterpretationKind.OPPOSING,
            ScientificStatus.MIXED: OutcomeInterpretationKind.UNRESOLVED,
            ScientificStatus.NOT_TESTED: OutcomeInterpretationKind.UNRESOLVED,
            ScientificStatus.PARTIAL: OutcomeInterpretationKind.UNRESOLVED,
            ScientificStatus.UNEVALUABLE: OutcomeInterpretationKind.UNRESOLVED,
        }[result.status]
        if interpretation.kind is not expected_kind:
            raise ValueError("fresh status contradicts its predeclared terminal interpretation")
        if (
            result.fresh_independent_unit_ids
            != obligations.evidence_contract.fresh_independent_unit_ids
        ):
            raise ValueError("fresh evaluator result uses units outside the frozen contract")

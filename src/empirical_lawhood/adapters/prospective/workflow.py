"""Concrete prospective nomination/design service for application composition."""

from __future__ import annotations

from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.design import ExperimentProposal
from empirical_lawhood.planning.exploration import ExploratoryFinding, HypothesisSet
from empirical_lawhood.planning.prospective import (
    NominationObligations,
    ProspectiveDesignContext,
    ProspectiveNominationDecision,
)

from .designers import default_designer_registry
from .nominator import ProspectiveNominator


class RegisteredProspectiveWorkflow:
    """Use only the statically registered nominator and design families."""

    def nominate(
        self,
        *,
        hypotheses: HypothesisSet,
        findings: tuple[ExploratoryFinding, ...],
        system: SystemSpec,
        context: ProspectiveDesignContext,
    ) -> tuple[ProspectiveNominationDecision, tuple[NominationObligations, ...]]:
        batch = ProspectiveNominator().nominate(
            hypotheses=hypotheses,
            findings=findings,
            system=system,
            context=context,
        )
        return batch.decision, batch.obligations

    def design(
        self,
        *,
        decision: ProspectiveNominationDecision,
        obligations: tuple[NominationObligations, ...],
        system: SystemSpec,
    ) -> tuple[ExperimentProposal, ...]:
        by_nomination = {item.nomination_id: item for item in obligations}
        registry = default_designer_registry()
        return tuple(
            registry.resolve(by_nomination[item.nomination_id].design_kind)
            .design(
                nomination=item,
                obligations=by_nomination[item.nomination_id],
                system=system,
            )
            .proposal
            for item in decision.nominations
        )

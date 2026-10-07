"""Rank already assessed finite candidates and compile only the selected owner."""

from dataclasses import dataclass
from decimal import Decimal as D

from empirical_lawhood.adapters.control.composition import ControllerStudyComposition
from empirical_lawhood.adapters.control.publication import ControlPublisher
from empirical_lawhood.adapters.methods.receiver_conditioned_io.finite_action_mpc import (
    FiniteChartMPC,
    FiniteMPCConfig,
    FiniteMPCMemberPrediction,
    FiniteMPCProposal,
)
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.controller_study import DeliveryControllerStudy
from empirical_lawhood.runtime.controller_compiler import CandidateEligibility, AdmissionCandidateAudit, CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_runtime import CommitmentDisposition, DeliveryControllerDecisionCommitment, RuntimeObservation


@dataclass(frozen=True)
class AssessedFiniteAction:
    candidate_id: str
    word: OccurrenceActionWord
    normalized_cost: D
    study: DeliveryControllerStudy
    audits: tuple[AdmissionCandidateAudit, ...]
    services: ControllerStudyComposition
    observation: RuntimeObservation


@dataclass(frozen=True)
class CommittedFiniteAction:
    assessed: AssessedFiniteAction
    compiled: CompiledDeliveryControllerStudy
    commitment: DeliveryControllerDecisionCommitment


def rank_finite_actions(
    *,
    actions: tuple[AssessedFiniteAction, ...],
    stem: str,
    chart: str,
    config: FiniteMPCConfig,
    publisher: ControlPublisher,
    commit_nonattempt: bool = False,
) -> tuple[FiniteMPCProposal, CommittedFiniteAction | None]:
    if (
        not actions
        or len({a.candidate_id for a in actions}) != len(actions)
        or len({(a.observation.independent_unit_id, a.observation.coordinate) for a in actions})
        != 1
        or any(len(a.audits) != 1 for a in actions)
    ):
        raise ValueError("ranking needs distinct single-action audits at one actual root/cutoff")
    predictions = tuple(
        (
            a.candidate_id,
            ObjectIdentity.from_record(a.word.word_id, a.word),
            (
                FiniteMPCMemberPrediction(
                    f"{a.candidate_id}.owned-closure",
                    chart,
                    D(0),
                    D(0),
                    a.normalized_cost,
                    D(1) if a.audits[0].eligibility is CandidateEligibility.ELIGIBLE else D(-1),
                    ObjectIdentity.from_record(a.audits[0].audit_id, a.audits[0]),
                ),
            ),
        )
        for a in actions
    )
    publisher.publish_record(config.config_id, config)
    proposal = FiniteChartMPC().propose(
        proposal_id=stem,
        action_predictions=predictions,
        expected_member_ids=(chart,),
        config=config,
    )
    publisher.publish_record(proposal.proposal_id, proposal)
    if not proposal.ranked_candidate_ids and not commit_nonattempt:
        return proposal, None
    chosen = (
        next(a for a in actions if a.candidate_id == proposal.ranked_candidate_ids[0])
        if proposal.ranked_candidate_ids
        else min(actions, key=lambda a: a.candidate_id)
    )
    compiled = chosen.services.compile(chosen.study)
    if compiled.candidate_audit != chosen.audits:
        raise ValueError("selected compilation differs from the owned candidate assessment")
    publisher.publish_record(compiled.compiled_study_id, compiled)
    commitment = chosen.services.prepare_commitment(
        compiled, chosen.observation, commitment_coordinate=chosen.observation.coordinate
    )
    if proposal.ranked_candidate_ids:
        if (
            commitment.action_binding is None
            or commitment.action_binding.action_word != chosen.word
        ):
            raise ValueError("owner commitment differs from the ranked admissible action")
    elif (
        commitment.disposition is not CommitmentDisposition.NONATTEMPT
        or commitment.action_binding is not None
    ):
        raise ValueError("refused finite ranking cannot produce an action or native fallback")
    publisher.publish_record(commitment.commitment_id, commitment)
    return proposal, CommittedFiniteAction(chosen, compiled, commitment)

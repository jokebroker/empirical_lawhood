"""Transparent Pareto portfolio planning with complete dispositions and stop semantics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from itertools import product

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.provenance import EvidenceSnapshot, ObjectIdentity
from empirical_lawhood.planning.discovery import (
    ExplorationEvidenceView,
    PortfolioCandidate,
    PortfolioDecision,
    PortfolioDecisionKind,
    PortfolioPolicy,
    PortfolioScore,
    TemplateInstantiation,
)
from empirical_lawhood.planning.exploration import (
    AnomalySignal,
    ExplorationPlan,
    ProposalDisposition,
    ProposalSelection,
)
from empirical_lawhood.runtime.compiler import compile_exploration_plan
from empirical_lawhood.runtime.plans import SnapshotVerification


@dataclass(frozen=True, slots=True)
class PlannedPortfolio:
    decision: PortfolioDecision
    plan: ExplorationPlan | None


def _clamp(value: Decimal) -> Decimal:
    return min(Decimal(1), max(Decimal(0), value))


def build_portfolio_candidates(
    instantiations: tuple[TemplateInstantiation, ...],
    signals: tuple[AnomalySignal, ...],
    view: ExplorationEvidenceView,
    *,
    executed_template_ids: tuple[str, ...] = (),
) -> tuple[PortfolioCandidate, ...]:
    """Construct declared objective vectors without a positivity objective."""

    signal_by_id = {signal.signal_id: signal for signal in signals}
    template_counts: dict[str, int] = {}
    for instantiation in instantiations:
        template_id = instantiation.template.object_id
        template_counts[template_id] = template_counts.get(template_id, 0) + 1
    physical_units = min(
        (measure.physical_independent_unit_count for measure in view.measures),
        default=1,
    )
    relation_quantity_count = len(
        {
            *view.relation.denominator_quantity_ids,
            *view.relation.history_quantity_ids,
            *view.relation.action_quantity_ids,
            *view.relation.receiver_quantity_ids,
        }
    )
    candidates = []
    for instantiation in instantiations:
        proposal = instantiation.proposal
        signal = signal_by_id[proposal.anomaly_signal_ids[0]]
        template_id = instantiation.template.object_id
        family_size = len(instantiation.obligations.registered_family_member_ids)
        candidates.append(
            PortfolioCandidate(
                instantiation=instantiation,
                score=PortfolioScore(
                    proposal_id=proposal.proposal_id,
                    falsification_value=_clamp(signal.severity),
                    unresolved_coverage=_clamp(
                        Decimal(len(signal.affected_quantity_ids))
                        / Decimal(max(1, relation_quantity_count))
                    ),
                    novelty=Decimal(0) if template_id in executed_template_ids else Decimal(1),
                    independent_unit_adequacy=_clamp(Decimal(physical_units) / Decimal(8)),
                    prospective_testability=(
                        Decimal("0.75") if instantiation.ready else Decimal(0)
                    ),
                    redundancy_risk=_clamp(
                        Decimal(template_counts[template_id] - 1)
                        / Decimal(max(1, template_counts[template_id]))
                    ),
                    search_slicing_risk=_clamp(Decimal(max(0, family_size - 1)) / Decimal(12)),
                ),
                redundancy_group_id=template_id,
            )
        )
    return tuple(sorted(candidates, key=lambda candidate: candidate.score.proposal_id))


def default_portfolio_policy() -> PortfolioPolicy:
    return PortfolioPolicy(
        policy_id="exploration-default-pareto-policy",
        budget=ResourceBudget(
            cpu_cores=12,
            memory_bytes=24_000_000_000,
            gpu_devices=0,
            wall_time_seconds=10_800,
            source_scan_bytes=300_000_000_000,
            output_bytes=3_000_000_000,
        ),
        maximum_selected_proposals=3,
        maximum_per_redundancy_group=1,
        maximum_family_members_per_analysis=64,
        maximum_total_family_members=128,
        minimum_falsification_value=Decimal("0.05"),
        minimum_independent_unit_adequacy=Decimal("0.25"),
        priority_axis_ids=(
            "falsification-value",
            "unresolved-coverage",
            "prospective-testability",
            "independent-unit-adequacy",
            "novelty",
            "redundancy-risk",
            "search-slicing-risk",
        ),
        stop_when_no_discriminating_analysis=True,
    )


_BENEFIT_FIELDS = (
    "falsification_value",
    "unresolved_coverage",
    "novelty",
    "independent_unit_adequacy",
    "prospective_testability",
)
_RISK_FIELDS = ("redundancy_risk", "search_slicing_risk")


def _dominates(left: PortfolioCandidate, right: PortfolioCandidate) -> bool:
    no_worse = all(
        getattr(left.score, field) >= getattr(right.score, field) for field in _BENEFIT_FIELDS
    ) and all(getattr(left.score, field) <= getattr(right.score, field) for field in _RISK_FIELDS)
    strictly_better = any(
        getattr(left.score, field) > getattr(right.score, field) for field in _BENEFIT_FIELDS
    ) or any(getattr(left.score, field) < getattr(right.score, field) for field in _RISK_FIELDS)
    return no_worse and strictly_better


def _aggregate_budget(candidates: tuple[PortfolioCandidate, ...]) -> ResourceBudget:
    budgets = tuple(candidate.instantiation.proposal.analysis.budget for candidate in candidates)
    return ResourceBudget(
        cpu_cores=sum(value.cpu_cores for value in budgets),
        memory_bytes=sum(value.memory_bytes for value in budgets),
        gpu_devices=sum(value.gpu_devices for value in budgets),
        wall_time_seconds=sum(value.wall_time_seconds for value in budgets),
        source_scan_bytes=sum(value.source_scan_bytes for value in budgets),
        output_bytes=sum(value.output_bytes for value in budgets),
    )


def _family_members(candidate: PortfolioCandidate) -> tuple[str, ...]:
    axes = candidate.instantiation.proposal.analysis.search_axes
    return tuple(
        sorted(
            ".".join(
                f"{axis.axis_id}--{member}" for axis, member in zip(axes, members, strict=True)
            )
            for members in product(*(axis.candidate_ids for axis in axes))
        )
    )


class ExplorationPortfolioPlanner:
    planner_key = "exploration.pareto-portfolio-planner"
    planner_version = "1.0.0"

    def select(
        self,
        *,
        plan_id: str,
        snapshot: EvidenceSnapshot,
        snapshot_verification: SnapshotVerification,
        candidates: tuple[PortfolioCandidate, ...],
        policy: PortfolioPolicy,
    ) -> PlannedPortfolio:
        self._validate_candidates(snapshot, candidates)
        family_feasible_ids = {
            candidate.score.proposal_id
            for candidate in candidates
            if len(candidate.instantiation.obligations.registered_family_member_ids)
            <= policy.maximum_family_members_per_analysis
        }
        qualifying = tuple(
            candidate
            for candidate in candidates
            if candidate.instantiation.ready
            and candidate.score.falsification_value >= policy.minimum_falsification_value
            and candidate.score.independent_unit_adequacy
            >= policy.minimum_independent_unit_adequacy
            and candidate.score.novelty > 0
            and candidate.score.proposal_id in family_feasible_ids
        )
        frontier = tuple(
            candidate
            for candidate in qualifying
            if not any(
                _dominates(other, candidate)
                for other in qualifying
                if other.score.proposal_id != candidate.score.proposal_id
            )
        )
        ordered = tuple(
            sorted(frontier, key=lambda candidate: self._priority_key(candidate, policy))
        )
        selected: list[PortfolioCandidate] = []
        group_counts: dict[str, int] = {}
        for candidate in ordered:
            if len(selected) >= policy.maximum_selected_proposals:
                break
            if group_counts.get(candidate.redundancy_group_id, 0) >= (
                policy.maximum_per_redundancy_group
            ):
                continue
            proposed = (*selected, candidate)
            if (
                sum(
                    len(value.instantiation.obligations.registered_family_member_ids)
                    for value in proposed
                )
                > policy.maximum_total_family_members
            ):
                continue
            if not policy.budget.contains(_aggregate_budget(tuple(proposed))):
                continue
            selected.append(candidate)
            group_counts[candidate.redundancy_group_id] = (
                group_counts.get(candidate.redundancy_group_id, 0) + 1
            )
        selected_ids = {candidate.score.proposal_id for candidate in selected}
        frontier_ids = {candidate.score.proposal_id for candidate in frontier}
        qualifying_ids = {candidate.score.proposal_id for candidate in qualifying}
        selections = tuple(
            sorted(
                (
                    self._selection(
                        candidate,
                        selected_ids=selected_ids,
                        frontier_ids=frontier_ids,
                        qualifying_ids=qualifying_ids,
                        family_feasible_ids=family_feasible_ids,
                    )
                    for candidate in candidates
                ),
                key=lambda selection: selection.proposal_id,
            )
        )
        proposals = tuple(
            sorted(
                (candidate.instantiation.proposal for candidate in candidates),
                key=lambda proposal: proposal.proposal_id,
            )
        )
        snapshot_identity = ObjectIdentity.from_record(snapshot.snapshot_id, snapshot)
        if selected:
            plan = compile_exploration_plan(
                plan_id=plan_id,
                snapshot=snapshot,
                snapshot_verification=snapshot_verification,
                proposals=proposals,
                selections=selections,
                budget=policy.budget,
            )
            decision = PortfolioDecision(
                decision_id=f"portfolio-decision.{plan_id}",
                kind=PortfolioDecisionKind.EXECUTE,
                snapshot=snapshot_identity,
                proposal_ids=tuple(proposal.proposal_id for proposal in proposals),
                selections=selections,
                pareto_frontier_proposal_ids=tuple(sorted(frontier_ids)),
                selected_plan=ObjectIdentity.from_record(plan.plan_id, plan),
                stop_reason_codes=(),
                planner_key=self.planner_key,
                planner_version=self.planner_version,
                outcome_access=snapshot.outcome_access,
                visibility_ceiling=snapshot.visibility_ceiling,
            )
            return PlannedPortfolio(decision=decision, plan=plan)
        stop_reasons = ["no-feasible-pareto-portfolio"]
        if not candidates:
            stop_reasons.append("no-anomaly-proposals")
        elif not qualifying:
            stop_reasons.append("no-discriminating-analysis")
        if not policy.stop_when_no_discriminating_analysis:
            raise ValueError("policy forbids a required stop/no-analysis decision")
        decision = PortfolioDecision(
            decision_id=f"portfolio-decision.{plan_id}",
            kind=PortfolioDecisionKind.STOP_NO_ANALYSIS,
            snapshot=snapshot_identity,
            proposal_ids=tuple(proposal.proposal_id for proposal in proposals),
            selections=selections,
            pareto_frontier_proposal_ids=tuple(sorted(frontier_ids)),
            selected_plan=None,
            stop_reason_codes=tuple(sorted(stop_reasons)),
            planner_key=self.planner_key,
            planner_version=self.planner_version,
            outcome_access=snapshot.outcome_access,
            visibility_ceiling=snapshot.visibility_ceiling,
        )
        return PlannedPortfolio(decision=decision, plan=None)

    @staticmethod
    def _validate_candidates(
        snapshot: EvidenceSnapshot,
        candidates: tuple[PortfolioCandidate, ...],
    ) -> None:
        proposal_ids = tuple(candidate.score.proposal_id for candidate in candidates)
        if tuple(sorted(set(proposal_ids))) != proposal_ids:
            raise ValueError("portfolio candidates must be sorted and unique by proposal")
        for candidate in candidates:
            analysis = candidate.instantiation.proposal.analysis
            if analysis.snapshot_id != snapshot.snapshot_id:
                raise ValueError("portfolio candidate binds another EvidenceSnapshot")
            if candidate.instantiation.obligations.registered_family_member_ids != (
                _family_members(candidate)
            ):
                raise ValueError("portfolio candidate omits registered family members")

    @staticmethod
    def _priority_key(
        candidate: PortfolioCandidate,
        policy: PortfolioPolicy,
    ) -> tuple[Decimal | str, ...]:
        values: dict[str, Decimal] = {
            "falsification-value": -candidate.score.falsification_value,
            "unresolved-coverage": -candidate.score.unresolved_coverage,
            "novelty": -candidate.score.novelty,
            "independent-unit-adequacy": -candidate.score.independent_unit_adequacy,
            "prospective-testability": -candidate.score.prospective_testability,
            "redundancy-risk": candidate.score.redundancy_risk,
            "search-slicing-risk": candidate.score.search_slicing_risk,
        }
        return (
            *tuple(values[axis] for axis in policy.priority_axis_ids),
            candidate.score.proposal_id,
        )

    def _selection(
        self,
        candidate: PortfolioCandidate,
        *,
        selected_ids: set[str],
        frontier_ids: set[str],
        qualifying_ids: set[str],
        family_feasible_ids: set[str],
    ) -> ProposalSelection:
        proposal_id = candidate.score.proposal_id
        if not candidate.instantiation.ready:
            disposition = ProposalDisposition.BLOCKED
            reasons = candidate.instantiation.reason_codes
        elif candidate.score.novelty == 0:
            disposition = ProposalDisposition.SUPERSEDED
            reasons = ("registered-template-already-executed",)
        elif proposal_id not in family_feasible_ids:
            disposition = ProposalDisposition.REJECTED
            reasons = ("registered-family-exceeds-declared-budget",)
        elif proposal_id in selected_ids:
            disposition = ProposalDisposition.SELECTED
            reasons = ("pareto-feasible-under-declared-budget",)
        elif proposal_id not in qualifying_ids:
            disposition = ProposalDisposition.REJECTED
            reasons = ("below-declared-discrimination-or-information-floor",)
        elif proposal_id not in frontier_ids:
            disposition = ProposalDisposition.REJECTED
            reasons = ("pareto-dominated",)
        else:
            disposition = ProposalDisposition.REJECTED
            reasons = ("frontier-budget-or-redundancy-limit",)
        return ProposalSelection(
            proposal_id=proposal_id,
            disposition=disposition,
            reason_codes=tuple(sorted(reasons)),
            planner_key=self.planner_key,
        )

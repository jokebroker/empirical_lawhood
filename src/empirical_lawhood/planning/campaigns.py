"""Immutable dual-lane campaign objectives and scientific lineage graph."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.authority import AuthorityAction, ResourceBudget
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import LifecycleStatus


class CampaignLane(StrEnum):
    EXPLORATORY = "EXPLORATORY"
    PROSPECTIVE = "PROSPECTIVE"


class CampaignNodeKind(StrEnum):
    EVIDENCE_SNAPSHOT = "EVIDENCE_SNAPSHOT"
    ANALYSIS_PROPOSAL = "ANALYSIS_PROPOSAL"
    EXPLORATION_PLAN = "EXPLORATION_PLAN"
    EXPLORATORY_FINDING = "EXPLORATORY_FINDING"
    HYPOTHESIS_SET = "HYPOTHESIS_SET"
    PROSPECTIVE_NOMINATION = "PROSPECTIVE_NOMINATION"
    EXPERIMENT_PROPOSAL = "EXPERIMENT_PROPOSAL"
    DECISION = "DECISION"
    AUTHORIZATION = "AUTHORIZATION"
    EXPERIMENT_SPEC = "EXPERIMENT_SPEC"
    RUN = "RUN"
    CLAIM = "CLAIM"


_EXPLORATORY_KINDS = {
    CampaignNodeKind.EVIDENCE_SNAPSHOT,
    CampaignNodeKind.ANALYSIS_PROPOSAL,
    CampaignNodeKind.EXPLORATION_PLAN,
    CampaignNodeKind.EXPLORATORY_FINDING,
    CampaignNodeKind.HYPOTHESIS_SET,
    CampaignNodeKind.PROSPECTIVE_NOMINATION,
}


@dataclass(frozen=True, slots=True)
class CampaignNode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/campaign-node'

    node_id: str
    kind: CampaignNodeKind
    lane: CampaignLane
    object_identity: ObjectIdentity
    parent_node_ids: tuple[str, ...]
    lifecycle_status: LifecycleStatus

    def __post_init__(self) -> None:
        validate_stable_id(self.node_id, field_name="node_id")
        require_sorted_unique_strings(self.parent_node_ids, field_name="parent_node_ids")
        if self.kind in _EXPLORATORY_KINDS:
            if self.lane is not CampaignLane.EXPLORATORY:
                raise ValueError("exploratory object is assigned to prospective lane")
        elif self.lane is not CampaignLane.PROSPECTIVE:
            raise ValueError("prospective object is assigned to exploratory lane")


@dataclass(frozen=True, slots=True)
class DecisionRight(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/decision-right'

    decision_right_id: str
    action: AuthorityAction
    decision_maker_id: str
    authority_policy_id: str
    delegated: bool

    def __post_init__(self) -> None:
        for name, value in (
            ("decision_right_id", self.decision_right_id),
            ("decision_maker_id", self.decision_maker_id),
            ("authority_policy_id", self.authority_policy_id),
        ):
            validate_stable_id(value, field_name=name)
        if (
            self.action
            in {
                AuthorityAction.FACILITY_OR_INSTRUMENT_COMMAND,
                AuthorityAction.HIL_ACTUATION,
                AuthorityAction.HUMAN_OR_ANIMAL_INTERVENTION,
                AuthorityAction.LIVE_ACTUATION,
                AuthorityAction.SAFETY_SIGNIFICANT_OPERATION,
            }
            and self.delegated
        ):
            raise ValueError("non-delegable physical authority cannot be delegated")


@dataclass(frozen=True, slots=True)
class CampaignSpec(CanonicalRecord):
    """Versioned objective and immutable lineage across adaptive frozen waves."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/campaign-spec'

    campaign_id: str
    objective: str
    system_ids: tuple[str, ...]
    world_ids: tuple[str, ...]
    target_claim_ids: tuple[str, ...]
    budget: ResourceBudget
    authority_policy: ObjectIdentity
    decision_rights: tuple[DecisionRight, ...]
    nodes: tuple[CampaignNode, ...]
    root_node_ids: tuple[str, ...]
    active_node_ids: tuple[str, ...]
    evidence_state: tuple[ObjectIdentity, ...]
    lifecycle_status: LifecycleStatus
    predecessor_campaign_ids: tuple[str, ...] = ()
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.campaign_id, field_name="campaign_id")
        validate_nonempty(self.objective, field_name="objective")
        for field_name, values in (
            ("system_ids", self.system_ids),
            ("world_ids", self.world_ids),
            ("target_claim_ids", self.target_claim_ids),
            ("root_node_ids", self.root_node_ids),
        ):
            require_sorted_unique_strings(values, field_name=field_name, allow_empty=False)
        require_sorted_unique_strings(self.active_node_ids, field_name="active_node_ids")
        require_sorted_unique_strings(
            self.predecessor_campaign_ids,
            field_name="predecessor_campaign_ids",
        )
        require_sorted_unique_ids(
            self.decision_rights,
            attribute="decision_right_id",
            field_name="decision_rights",
        )
        if not self.decision_rights:
            raise ValueError("campaign requires explicit decision rights")
        require_sorted_unique_ids(self.nodes, attribute="node_id", field_name="nodes")
        require_sorted_unique_ids(
            self.evidence_state, attribute="object_id", field_name="evidence_state"
        )
        self._validate_graph()
        require_extensions(self.extensions)

    def _validate_graph(self) -> None:
        node_ids = {node.node_id for node in self.nodes}
        if not set(self.root_node_ids).issubset(node_ids):
            raise ValueError("campaign roots reference unknown nodes")
        if not set(self.active_node_ids).issubset(node_ids):
            raise ValueError("campaign active set references unknown nodes")
        parents = {node.node_id: node.parent_node_ids for node in self.nodes}
        observed_roots = tuple(sorted(node_id for node_id, value in parents.items() if not value))
        if observed_roots != self.root_node_ids:
            raise ValueError("declared campaign roots differ from graph roots")
        for node_id, parent_ids in parents.items():
            unknown = set(parent_ids) - node_ids
            if unknown:
                raise ValueError(
                    f"campaign node {node_id!r} has unknown parents: {sorted(unknown)}"
                )
        for start in parents:
            self._assert_acyclic(start, parents)
        self._validate_lane_bridges()

    @staticmethod
    def _assert_acyclic(start: str, parents: dict[str, tuple[str, ...]]) -> None:
        pending = [start]
        path: set[str] = set()
        completed: set[str] = set()
        while pending:
            current = pending[-1]
            if current in completed:
                pending.pop()
                continue
            if current in path:
                path.remove(current)
                completed.add(current)
                pending.pop()
                continue
            path.add(current)
            for parent in parents[current]:
                if parent in path:
                    raise ValueError("campaign lineage graph contains a cycle")
                if parent not in completed:
                    pending.append(parent)

    def _validate_lane_bridges(self) -> None:
        nodes = {node.node_id: node for node in self.nodes}
        for node in self.nodes:
            if node.lane is not CampaignLane.PROSPECTIVE:
                continue
            exploratory_parents = [
                nodes[parent_id]
                for parent_id in node.parent_node_ids
                if nodes[parent_id].lane is CampaignLane.EXPLORATORY
            ]
            if exploratory_parents and node.kind is not CampaignNodeKind.EXPERIMENT_PROPOSAL:
                raise ValueError(
                    "only an ExperimentProposal may bridge exploration to prospective work"
                )
            if any(
                parent.kind is not CampaignNodeKind.PROSPECTIVE_NOMINATION
                for parent in exploratory_parents
            ):
                raise ValueError("prospective bridge must originate at a ProspectiveNomination")

    def nodes_in_lane(self, lane: CampaignLane) -> tuple[CampaignNode, ...]:
        return tuple(node for node in self.nodes if node.lane is lane)

    def node(self, node_id: str) -> CampaignNode:
        validate_stable_id(node_id, field_name="node_id")
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        raise KeyError(node_id)

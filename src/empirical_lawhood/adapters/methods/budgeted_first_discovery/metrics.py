"""Family/world-level first-discovery estimands and censoring semantics."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_decimal, validate_stable_id

from .contracts import (
    DiscoveryObservation,
    ObservationOrigin,
    PolicyDecision,
    PolicyDecisionKind,
)


class DiscoveryDisposition(StrEnum):
    DISCOVERED = "DISCOVERED"
    RIGHT_CENSORED = "RIGHT_CENSORED"
    CORRECT_HOLD = "CORRECT_HOLD"
    UNSUPPORTED_HOLD = "UNSUPPORTED_HOLD"


@dataclass(frozen=True, slots=True)
class WorldPolicyResult(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/budgeted-first-discovery/world-policy-result'

    result_id: str
    world_id: str
    policy_id: str
    disposition: DiscoveryDisposition
    discovered_candidate_id: str | None
    recommendations_to_discovery: int | None
    cost_to_discovery: Decimal | None
    total_recommendations: int
    total_cost: Decimal
    unlabelled_observations: int
    invalid_observations: int
    information_queries: int
    promotion_count: int
    false_promotion_count: int
    unsupported_action_count: int
    hold_count: int

    def __post_init__(self) -> None:
        for name in ("result_id", "world_id", "policy_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "total_recommendations",
            "unlabelled_observations",
            "invalid_observations",
            "information_queries",
            "promotion_count",
            "false_promotion_count",
            "unsupported_action_count",
            "hold_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.false_promotion_count > self.promotion_count:
            raise ValueError("false promotions exceed target-seeking promotions")
        validate_decimal(self.total_cost, field_name="total_cost", minimum=Decimal(0))
        if self.disposition is DiscoveryDisposition.DISCOVERED:
            if (
                self.discovered_candidate_id is None
                or self.recommendations_to_discovery is None
                or self.cost_to_discovery is None
            ):
                raise ValueError("discovery disposition requires first-discovery estimands")
            validate_stable_id(
                self.discovered_candidate_id,
                field_name="discovered_candidate_id",
            )
            validate_decimal(
                self.cost_to_discovery,
                field_name="cost_to_discovery",
                minimum=Decimal(0),
            )
        elif any(
            value is not None
            for value in (
                self.discovered_candidate_id,
                self.recommendations_to_discovery,
                self.cost_to_discovery,
            )
        ):
            raise ValueError("censored/HOLD result cannot invent discovery estimands")


def evaluate_policy_history(
    *,
    world_id: str,
    policy_id: str,
    decisions: tuple[PolicyDecision, ...],
    observations: tuple[DiscoveryObservation, ...],
    target_candidate_ids: frozenset[str],
    empty_or_disconnected_world: bool,
    unsupported_action_count: int = 0,
) -> WorldPolicyResult:
    """Evaluate one ordered history without treating missing labels as negatives."""

    if any(value.origin is not ObservationOrigin.QUERY_RESPONSE for value in observations):
        raise ValueError("policy-history metrics accept query responses, not initial labels")
    observed_ids = tuple(value.candidate_id for value in observations)
    if len(observed_ids) != len(set(observed_ids)):
        raise ValueError("candidate was measured more than once")
    requested_ids = tuple(
        candidate_id for decision in decisions for candidate_id in decision.requested_candidate_ids
    )
    if requested_ids != observed_ids:
        raise ValueError("requested and observed candidate order differs")
    discovery_batch_index = next(
        (
            index
            for index, decision in enumerate(decisions)
            if set(decision.requested_candidate_ids) & target_candidate_ids
        ),
        None,
    )
    total_cost = sum((value.query_cost for value in observations), Decimal(0))
    holds = sum(value.kind is PolicyDecisionKind.HOLD for value in decisions)
    information_ids = {value for decision in decisions for value in decision.information_query_ids}
    promotion_ids = set(observed_ids) - information_ids
    if discovery_batch_index is not None:
        successful_decision = decisions[discovery_batch_index]
        discovered_candidate_id = next(
            candidate_id
            for candidate_id in successful_decision.requested_candidate_ids
            if candidate_id in target_candidate_ids
        )
        charged_ids = {
            candidate_id
            for decision in decisions[: discovery_batch_index + 1]
            for candidate_id in decision.requested_candidate_ids
        }
        disposition = DiscoveryDisposition.DISCOVERED
        # The receiver reveals a committed batch atomically. Discovery cost is
        # therefore charged through the complete successful batch; ordering
        # candidate IDs within that batch must not create a free advantage.
        recommendations = sum(
            len(value.requested_candidate_ids) for value in decisions[: discovery_batch_index + 1]
        )
        cost_to_discovery = sum(
            (value.query_cost for value in observations if value.candidate_id in charged_ids),
            Decimal(0),
        )
    else:
        disposition = (
            DiscoveryDisposition.CORRECT_HOLD
            if holds and empty_or_disconnected_world
            else DiscoveryDisposition.UNSUPPORTED_HOLD
            if holds
            else DiscoveryDisposition.RIGHT_CENSORED
        )
        discovered_candidate_id = None
        recommendations = None
        cost_to_discovery = None
    return WorldPolicyResult(
        result_id=f"result.{world_id}.{policy_id}",
        world_id=world_id,
        policy_id=policy_id,
        disposition=disposition,
        discovered_candidate_id=discovered_candidate_id,
        recommendations_to_discovery=recommendations,
        cost_to_discovery=cost_to_discovery,
        total_recommendations=len(observations),
        total_cost=total_cost,
        unlabelled_observations=sum(value.tc_kelvin is None for value in observations),
        invalid_observations=sum(value.state.value == "INVALID" for value in observations),
        information_queries=sum(value in information_ids for value in observed_ids),
        promotion_count=sum(value in promotion_ids for value in observed_ids),
        false_promotion_count=sum(
            value.candidate_id in promotion_ids and value.state.value == "EXPLICIT_NEGATIVE"
            for value in observations
        ),
        unsupported_action_count=unsupported_action_count,
        hold_count=holds,
    )


__all__ = ["DiscoveryDisposition", "WorldPolicyResult", "evaluate_policy_history"]

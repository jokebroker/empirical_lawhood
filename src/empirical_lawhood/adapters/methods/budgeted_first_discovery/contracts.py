"""Canonical policy decisions and private in-process scientific views.

Family truth is deliberately absent from :class:`CandidateView` and
:class:`DiscoveryObservation`.  It is evaluator-only state owned by the
reference-world adapter.  Selectors therefore cannot infer a target by reading
the benchmark's family field or a target flag.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)


class LabelState(StrEnum):
    """Outcome state returned by the simulated measurement receiver."""

    MEASURED = "MEASURED"
    EXPLICIT_NEGATIVE = "EXPLICIT_NEGATIVE"
    UNLABELLED = "UNLABELLED"
    INVALID = "INVALID"


class ObservationOrigin(StrEnum):
    INITIAL_LABEL = "INITIAL_LABEL"
    QUERY_RESPONSE = "QUERY_RESPONSE"


class PolicyKind(StrEnum):
    STRATIFIED_RANDOM = "STRATIFIED_RANDOM"
    MAXIMIN = "MAXIMIN"
    GREEDY_SCALAR_ENSEMBLE = "GREEDY_SCALAR_ENSEMBLE"
    BOTORCH_DISCRETE_UCB = "BOTORCH_DISCRETE_UCB"
    LOCAL_LAW_BOUNDARY = "LOCAL_LAW_BOUNDARY"


class PolicyDecisionKind(StrEnum):
    QUERY = "QUERY"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class CandidateView:
    """Policy-visible candidate projection with no receiver or family truth."""

    candidate_id: str
    stratum_id: str
    features: tuple[float, ...]
    valid: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_stable_id(self.stratum_id, field_name="stratum_id")
        if not self.features or len(self.features) > 256:
            raise ValueError("candidate features must have between 1 and 256 values")
        if any(not (-1e12 < value < 1e12) for value in self.features):
            raise ValueError("candidate features must be finite and bounded")


@dataclass(frozen=True, slots=True)
class DiscoveryObservation(CanonicalRecord):
    """One committed-query observation visible to the next selection round."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/budgeted-first-discovery/discovery-observation'

    candidate_id: str
    origin: ObservationOrigin
    state: LabelState
    tc_kelvin: Decimal | None
    query_cost: Decimal
    round_index: int | None

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_decimal(self.query_cost, field_name="query_cost", minimum=Decimal(0))
        if self.origin is ObservationOrigin.INITIAL_LABEL:
            if self.query_cost != 0 or self.round_index is not None:
                raise ValueError("initial labels cannot spend budget or claim a query round")
        elif (
            self.query_cost <= 0
            or self.round_index is None
            or isinstance(self.round_index, bool)
            or self.round_index < 0
        ):
            raise ValueError("query responses require positive cost and a nonnegative round")
        if self.state in {LabelState.MEASURED, LabelState.EXPLICIT_NEGATIVE}:
            if self.tc_kelvin is None:
                raise ValueError("measured observations require bounded Tc")
            validate_decimal(
                self.tc_kelvin,
                field_name="tc_kelvin",
                minimum=Decimal(0),
            )
            if self.tc_kelvin > Decimal(400):
                raise ValueError("tc_kelvin exceeds the bounded receiver range")
            if self.state is LabelState.EXPLICIT_NEGATIVE and self.tc_kelvin >= Decimal(5):
                raise ValueError("explicit-negative observations must remain below 5 K")
        elif self.tc_kelvin is not None:
            raise ValueError("unlabelled or invalid observations cannot carry Tc")


def visible_prefix_sha256(observations: tuple[DiscoveryObservation, ...]) -> str:
    """Hash exactly the selector-visible ordered history prefix."""

    payload = tuple(
        {
            "candidate_id": value.candidate_id,
            "origin": value.origin.value,
            "query_cost": str(value.query_cost),
            "round_index": value.round_index,
            "state": value.state.value,
            "tc_kelvin": value.tc_kelvin,
        }
        for value in observations
    )
    return sha256(canonical_json_bytes(payload)).hexdigest()


@dataclass(frozen=True, slots=True)
class DiscoveryPolicyConfig(CanonicalRecord):
    """Frozen selector semantics shared across all material-family worlds."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/budgeted-first-discovery/discovery-policy-config'

    config_id: str
    policy_kind: PolicyKind
    batch_size: int
    total_query_budget: int
    seed: int
    query_cost: Decimal
    ensemble_members: int
    ensemble_min_leaf: int
    ucb_beta: Decimal
    local_neighbor_count: int
    local_min_effective_rank: int
    local_anchor_count: int
    local_anchor_quantile: Decimal
    local_ridge: Decimal
    local_support_multiplier: Decimal
    local_boundary_weight: Decimal
    information_queries_per_batch: int
    bo_training_limit: int
    bo_projection_dimensions: int
    bo_fit_max_iterations: int
    feature_schema_id: str
    implementation_id: str

    def __post_init__(self) -> None:
        for name in ("config_id", "feature_schema_id", "implementation_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "batch_size",
            "total_query_budget",
            "ensemble_members",
            "ensemble_min_leaf",
            "local_neighbor_count",
            "local_min_effective_rank",
            "local_anchor_count",
            "bo_training_limit",
            "bo_projection_dimensions",
            "bo_fit_max_iterations",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.batch_size > self.total_query_budget:
            raise ValueError("batch_size exceeds total query budget")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be nonnegative")
        for name in (
            "query_cost",
            "ucb_beta",
            "local_anchor_quantile",
            "local_ridge",
            "local_support_multiplier",
            "local_boundary_weight",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        if self.query_cost <= 0 or self.local_ridge <= 0 or self.local_support_multiplier <= 0:
            raise ValueError("cost, ridge and support multiplier must be positive")
        if not Decimal(0) < self.local_anchor_quantile < Decimal(1):
            raise ValueError("local_anchor_quantile must lie strictly inside (0, 1)")
        if self.information_queries_per_batch < 0:
            raise ValueError("information_queries_per_batch must be nonnegative")
        if self.information_queries_per_batch > self.batch_size:
            raise ValueError("information-query quota exceeds batch size")
        if self.local_min_effective_rank > self.local_neighbor_count:
            raise ValueError("local effective-rank floor exceeds neighborhood size")
        if self.bo_projection_dimensions > 64:
            raise ValueError("BO projection exceeds its bounded dimension ceiling")
        if self.bo_training_limit < self.bo_projection_dimensions * 4:
            raise ValueError("BO training limit is inadequate for its projection")


@dataclass(frozen=True, slots=True)
class PolicyDecision(CanonicalRecord):
    """Immutable selection commitment emitted before evaluator access."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/budgeted-first-discovery/policy-decision'

    decision_id: str
    world_id: str
    policy_config_sha256: str
    round_index: int
    visible_prefix_sha256: str
    eligible_pool_sha256: str
    selector_state_sha256: str
    kind: PolicyDecisionKind
    requested_candidate_ids: tuple[str, ...]
    information_query_ids: tuple[str, ...]
    incremental_cost: Decimal
    hold_reason_id: str | None

    def __post_init__(self) -> None:
        for name in ("decision_id", "world_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "policy_config_sha256",
            "visible_prefix_sha256",
            "eligible_pool_sha256",
            "selector_state_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if isinstance(self.round_index, bool) or self.round_index < 0:
            raise ValueError("round_index must be nonnegative")
        if self.requested_candidate_ids != tuple(sorted(set(self.requested_candidate_ids))):
            raise ValueError("requested candidate IDs must be sorted and unique")
        if self.information_query_ids != tuple(sorted(set(self.information_query_ids))):
            raise ValueError("information query IDs must be sorted and unique")
        if not set(self.information_query_ids).issubset(self.requested_candidate_ids):
            raise ValueError("information queries must be requested candidates")
        validate_decimal(self.incremental_cost, field_name="incremental_cost", minimum=Decimal(0))
        if self.kind is PolicyDecisionKind.HOLD:
            if self.requested_candidate_ids or self.incremental_cost != 0:
                raise ValueError("HOLD cannot request candidates or spend query cost")
            if self.hold_reason_id is None:
                raise ValueError("HOLD requires a typed reason")
            validate_stable_id(self.hold_reason_id, field_name="hold_reason_id")
        else:
            if not self.requested_candidate_ids or self.incremental_cost <= 0:
                raise ValueError("QUERY requires candidates and positive cost")
            if self.hold_reason_id is not None:
                raise ValueError("QUERY cannot carry a HOLD reason")


__all__ = [
    "CandidateView",
    "DiscoveryObservation",
    "DiscoveryPolicyConfig",
    "LabelState",
    "ObservationOrigin",
    "PolicyDecision",
    "PolicyDecisionKind",
    "PolicyKind",
    "visible_prefix_sha256",
]

"""Pure constrained finite-chart MPC candidate and utility evidence."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import decimal_from_float


class FiniteMPCDisposition(StrEnum):
    CANDIDATE = "CANDIDATE"
    CONSTRAINT_REJECTED = "CONSTRAINT_REJECTED"
    UNEVALUABLE = "UNEVALUABLE"


class FiniteMPCAggregationRule(StrEnum):
    WORST_MEMBER_COST = "WORST_MEMBER_COST"


class FiniteMPCTieBreakRule(StrEnum):
    UTILITY_DESCENDING_CANDIDATE_ID_ASCENDING = "UTILITY_DESCENDING_CANDIDATE_ID_ASCENDING"


@dataclass(frozen=True, slots=True)
class FiniteMPCConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/finite-mpc-config'

    config_id: str
    response_cost_weight: Decimal
    sink_cost_weight: Decimal
    effort_cost_weight: Decimal
    minimum_constraint_margin: Decimal
    normalized_cost_unit: str
    constraint_margin_unit: str
    aggregation_rule: FiniteMPCAggregationRule
    tie_break_rule: FiniteMPCTieBreakRule

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        for name, value in (
            ("response_cost_weight", self.response_cost_weight),
            ("sink_cost_weight", self.sink_cost_weight),
            ("effort_cost_weight", self.effort_cost_weight),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
        if not any(
            value > 0
            for value in (
                self.response_cost_weight,
                self.sink_cost_weight,
                self.effort_cost_weight,
            )
        ):
            raise ValueError("finite MPC requires at least one positive cost weight")
        validate_decimal(self.minimum_constraint_margin, field_name="minimum_constraint_margin")
        if self.normalized_cost_unit != "1":
            raise ValueError("finite MPC costs must be pre-normalized dimensionless values")
        if not self.constraint_margin_unit.strip():
            raise ValueError("finite MPC constraint margin unit must not be empty")


@dataclass(frozen=True, slots=True)
class FiniteMPCMemberPrediction(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/finite-mpc-member-prediction'

    prediction_id: str
    denominator_member_id: str
    response_cost: Decimal
    sink_cost: Decimal
    effort_cost: Decimal
    constraint_margin: Decimal
    evidence_result: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.prediction_id, field_name="prediction_id")
        validate_stable_id(self.denominator_member_id, field_name="denominator_member_id")
        for name, value in (
            ("response_cost", self.response_cost),
            ("sink_cost", self.sink_cost),
            ("effort_cost", self.effort_cost),
            ("constraint_margin", self.constraint_margin),
        ):
            validate_decimal(value, field_name=name)


@dataclass(frozen=True, slots=True)
class FiniteMPCActionCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/finite-mpc-action-candidate'

    candidate_id: str
    action_word: ObjectIdentity
    member_predictions: tuple[FiniteMPCMemberPrediction, ...]
    robust_utility: NamedDecimal | None
    disposition: FiniteMPCDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        if self.action_word.object_schema != OccurrenceActionWord.SCHEMA:
            raise ValueError("finite MPC candidate requires an exact ActionWord")
        require_sorted_unique_ids(
            self.member_predictions,
            attribute="denominator_member_id",
            field_name="member_predictions",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is FiniteMPCDisposition.CANDIDATE:
            if self.robust_utility is None or not self.member_predictions or self.reason_codes:
                raise ValueError("eligible MPC candidate requires complete member utility")
        elif self.robust_utility is not None or not self.reason_codes:
            raise ValueError("rejected MPC candidate cannot carry robust utility")


@dataclass(frozen=True, slots=True)
class FiniteMPCProposal(CanonicalRecord):
    """Ranked candidates only; contains no commitment/final-action field."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/receiver-conditioned-io/finite-mpc-proposal'

    proposal_id: str
    config: ObjectIdentity
    candidates: tuple[FiniteMPCActionCandidate, ...]
    ranked_candidate_ids: tuple[str, ...]
    expected_member_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.proposal_id, field_name="proposal_id")
        if self.config.object_schema != FiniteMPCConfig.SCHEMA:
            raise ValueError("finite MPC proposal requires an exact config")
        require_sorted_unique_ids(
            self.candidates, attribute="candidate_id", field_name="candidates"
        )
        require_sorted_unique_strings(
            tuple(sorted(self.ranked_candidate_ids)),
            field_name="ranked_candidate_ids",
        )
        require_sorted_unique_strings(
            self.expected_member_ids,
            field_name="expected_member_ids",
            allow_empty=False,
        )
        eligible = {
            value.candidate_id
            for value in self.candidates
            if value.disposition is FiniteMPCDisposition.CANDIDATE
        }
        if set(self.ranked_candidate_ids) != eligible:
            raise ValueError("MPC ranking differs from eligible candidate roster")


@dataclass(frozen=True, slots=True)
class FiniteChartMPC:
    def propose(
        self,
        *,
        proposal_id: str,
        action_predictions: tuple[
            tuple[str, ObjectIdentity, tuple[FiniteMPCMemberPrediction, ...]], ...
        ],
        expected_member_ids: tuple[str, ...],
        config: FiniteMPCConfig,
    ) -> FiniteMPCProposal:
        candidates: list[FiniteMPCActionCandidate] = []
        utilities: dict[str, float] = {}
        for candidate_id, action_word, predictions in sorted(
            action_predictions,
            key=lambda value: value[0],
        ):
            ordered = tuple(sorted(predictions, key=lambda value: value.denominator_member_id))
            reasons: tuple[str, ...]
            if tuple(value.denominator_member_id for value in ordered) != expected_member_ids:
                disposition = FiniteMPCDisposition.UNEVALUABLE
                robust = None
                reasons = ("member-prediction-roster-incomplete",)
            elif any(
                value.constraint_margin < config.minimum_constraint_margin for value in ordered
            ):
                disposition = FiniteMPCDisposition.CONSTRAINT_REJECTED
                robust = None
                reasons = ("member-constraint-margin-negative",)
            else:
                costs = tuple(
                    float(
                        config.response_cost_weight * value.response_cost
                        + config.sink_cost_weight * value.sink_cost
                        + config.effort_cost_weight * value.effort_cost
                    )
                    for value in ordered
                )
                utility = -max(costs)
                utilities[candidate_id] = utility
                disposition = FiniteMPCDisposition.CANDIDATE
                robust = NamedDecimal(
                    value_id=f"finite-mpc-robust-utility.{candidate_id}",
                    value=decimal_from_float(utility),
                    unit="1",
                )
                reasons = ()
            candidates.append(
                FiniteMPCActionCandidate(
                    candidate_id=candidate_id,
                    action_word=action_word,
                    member_predictions=ordered,
                    robust_utility=robust,
                    disposition=disposition,
                    reason_codes=reasons,
                )
            )
        ranked = tuple(
            candidate_id
            for candidate_id, _ in sorted(
                utilities.items(),
                key=lambda item: (-item[1], item[0]),
            )
        )
        return FiniteMPCProposal(
            proposal_id=proposal_id,
            config=ObjectIdentity.from_record(config.config_id, config),
            candidates=tuple(candidates),
            ranked_candidate_ids=ranked,
            expected_member_ids=expected_member_ids,
        )

"""Excluded, nonpromotable source/action/runtime canary reduction."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar, Iterable

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)

from .target_execution import TargetConstructValidationCompleteUnitResult, TargetConstructValidationExecutionPhase
from .target_freeze import TargetConstructValidationTargetDesignFreeze


@dataclass(frozen=True, slots=True)
class TargetConstructValidationCanaryQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-canary-qualification'

    qualification_id: str
    target_id: str
    target_design: ObjectIdentity
    canary_results: tuple[ObjectIdentity, ...]
    canary_unit_ids: tuple[str, ...]
    simulator_version: str
    complete_unit_count: int
    condition_count: int
    maximum_elapsed_seconds: Decimal
    unit_timeout_seconds: int
    action_chain_qualified: bool
    receiver_decoding_qualified: bool
    complete_reset_qualified: bool
    resource_envelope_qualified: bool
    scientific_response_used_for_design_count: int
    excluded_from_development_and_evaluation: bool
    qualified: bool
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("qualification_id", "target_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_ids(
            self.canary_results,
            attribute="object_id",
            field_name="canary_results",
        )
        require_sorted_unique_strings(
            self.canary_unit_ids,
            field_name="canary_unit_ids",
            allow_empty=False,
        )
        validate_nonempty(self.simulator_version, field_name="simulator_version")
        validate_decimal(self.maximum_elapsed_seconds, field_name="maximum_elapsed_seconds")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.complete_unit_count != len(self.canary_unit_ids):
            raise ValueError("canary complete-unit count differs")
        expected = all(
            (
                self.action_chain_qualified,
                self.receiver_decoding_qualified,
                self.complete_reset_qualified,
                self.resource_envelope_qualified,
                self.scientific_response_used_for_design_count == 0,
                self.excluded_from_development_and_evaluation,
                not self.reason_codes,
            )
        )
        if self.qualified != expected:
            raise ValueError("canary qualification is not check-derived")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("canary qualification must remain development-only")


def qualify_canary(
    design: TargetConstructValidationTargetDesignFreeze,
    results: Iterable[TargetConstructValidationCompleteUnitResult],
) -> TargetConstructValidationCanaryQualification:
    values = tuple(sorted(results, key=lambda value: value.complete_unit_id))
    reasons: set[str] = set()
    if tuple(value.complete_unit_id for value in values) != design.task.canary_unit_ids:
        reasons.add("CANARY_ROSTER_INCOMPLETE")
    if any(
        value.phase is not TargetConstructValidationExecutionPhase.CANARY
        or value.target_design != ObjectIdentity.from_record(design.freeze_id, design)
        for value in values
    ):
        reasons.add("CANARY_DESIGN_OR_PHASE_MISMATCH")
    expected_conditions = (
        len(design.task.denominator_value_ids)
        * len(design.task.history_value_ids)
        * len(design.task.native_action_value_ids)
    )
    condition_count = sum(len(value.conditions) for value in values)
    if any(len(value.conditions) != expected_conditions for value in values):
        reasons.add("CANARY_CONDITION_ROSTER_INCOMPLETE")
    versions = {value.simulator_version for value in values}
    if len(versions) != 1 or versions != {design.task.package_version}:
        reasons.add("CANARY_SIMULATOR_VERSION_DRIFT")
    action_chain = all(
        condition.action.requested_clock
        <= condition.action.accepted_clock
        <= condition.action.applied_clock
        <= condition.action.realized_clock
        and condition.action.native_action_id == condition.native_action_id
        for value in values
        for condition in value.conditions
    )
    if not action_chain:
        reasons.add("CANARY_ACTION_CHAIN_INVALID")
    receiver_decoding = all(
        len(condition.receivers) == len(design.task.receiver_value_ids)
        and tuple(receiver.receiver_id for receiver in condition.receivers)
        == design.task.receiver_value_ids
        and all(receiver.finite for receiver in condition.receivers)
        for value in values
        for condition in value.conditions
    )
    if not receiver_decoding:
        reasons.add("CANARY_RECEIVER_DECODING_INVALID")
    maximum_elapsed = max((value.elapsed_seconds for value in values), default=Decimal("0"))
    resource_qualified = maximum_elapsed <= Decimal(design.task.unit_timeout_seconds)
    if not resource_qualified:
        reasons.add("CANARY_UNIT_TIMEOUT_EXCEEDED")
    reset_qualified = len({value.seed for value in values}) == len(values)
    if not reset_qualified:
        reasons.add("CANARY_UNIT_SEEDS_NOT_DISTINCT")
    return TargetConstructValidationCanaryQualification(
        qualification_id=f"canary-qualification.{design.task.target_id}",
        target_id=design.task.target_id,
        target_design=ObjectIdentity.from_record(design.freeze_id, design),
        canary_results=tuple(
            ObjectIdentity.from_record(value.result_id, value) for value in values
        ),
        canary_unit_ids=design.task.canary_unit_ids,
        simulator_version=next(iter(versions), "unavailable"),
        complete_unit_count=len(values),
        condition_count=condition_count,
        maximum_elapsed_seconds=maximum_elapsed,
        unit_timeout_seconds=design.task.unit_timeout_seconds,
        action_chain_qualified=action_chain,
        receiver_decoding_qualified=receiver_decoding,
        complete_reset_qualified=reset_qualified,
        resource_envelope_qualified=resource_qualified,
        scientific_response_used_for_design_count=0,
        excluded_from_development_and_evaluation=True,
        qualified=not reasons,
        reason_codes=tuple(sorted(reasons)),
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
    )


__all__ = ['TargetConstructValidationCanaryQualification', "qualify_canary"]

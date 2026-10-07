"""Pure delegated-authority and resource-budget contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from .evidence import OutcomeAccess
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_strings,
    validate_stable_id,
)
from .time import parse_utc_timestamp
from .worlds import WorldKind


class AuthorityAction(StrEnum):
    REPOSITORY_IMPLEMENTATION = "REPOSITORY_IMPLEMENTATION"
    REFERENCE_WORLD_EXECUTION = "REFERENCE_WORLD_EXECUTION"
    SIMULATION_EXECUTION = "SIMULATION_EXECUTION"
    READ_ONLY_EXPLORATION = "READ_ONLY_EXPLORATION"
    PUBLIC_SOURCE_ACQUISITION = "PUBLIC_SOURCE_ACQUISITION"
    DATASET_REGISTRATION = "DATASET_REGISTRATION"
    DATASET_TRANSFORMATION = "DATASET_TRANSFORMATION"
    DATASET_BINDING = "DATASET_BINDING"
    DATASET_PROJECTION_REBUILD = "DATASET_PROJECTION_REBUILD"
    NONACTUATING_PROSPECTIVE_FREEZE = "NONACTUATING_PROSPECTIVE_FREEZE"
    EVALUATOR_REVEAL = "EVALUATOR_REVEAL"
    HIL_ACTUATION = "HIL_ACTUATION"
    LIVE_ACTUATION = "LIVE_ACTUATION"
    FACILITY_OR_INSTRUMENT_COMMAND = "FACILITY_OR_INSTRUMENT_COMMAND"
    SAFETY_SIGNIFICANT_OPERATION = "SAFETY_SIGNIFICANT_OPERATION"
    HUMAN_OR_ANIMAL_INTERVENTION = "HUMAN_OR_ANIMAL_INTERVENTION"
    PAID_OR_EXTERNALLY_BILLED_RESOURCE = "PAID_OR_EXTERNALLY_BILLED_RESOURCE"


class SourceAccessClass(StrEnum):
    NONE = "NONE"
    OFFICIAL_OPEN_PUBLIC = "OFFICIAL_OPEN_PUBLIC"
    CLICK_THROUGH_TERMS = "CLICK_THROUGH_TERMS"
    PRIVATE_CREDENTIAL = "PRIVATE_CREDENTIAL"
    PAID = "PAID"


_OUTCOME_ACCESS_RANK = {
    OutcomeAccess.OUTCOME_BLIND: 0,
    OutcomeAccess.EVALUATION_SEALED: 0,
    OutcomeAccess.DEVELOPMENT_VISIBLE: 1,
    OutcomeAccess.EVALUATOR_REVEAL: 2,
    OutcomeAccess.EVALUATION_REVEALED: 2,
    OutcomeAccess.PRIVILEGED_TRUTH: 3,
}


@dataclass(frozen=True, slots=True)
class ResourceBudget(CanonicalRecord):
    """Worst-case resource request or policy ceiling."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/resource-budget'

    cpu_cores: int
    memory_bytes: int
    gpu_devices: int
    wall_time_seconds: int
    source_scan_bytes: int
    output_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("cpu_cores", self.cpu_cores),
            ("memory_bytes", self.memory_bytes),
            ("wall_time_seconds", self.wall_time_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in (
            ("gpu_devices", self.gpu_devices),
            ("source_scan_bytes", self.source_scan_bytes),
            ("output_bytes", self.output_bytes),
        ):
            if value < 0:
                raise ValueError(f"{name} must be nonnegative")

    def contains(self, requested: ResourceBudget) -> bool:
        return all(
            requested_value <= ceiling_value
            for requested_value, ceiling_value in (
                (requested.cpu_cores, self.cpu_cores),
                (requested.memory_bytes, self.memory_bytes),
                (requested.gpu_devices, self.gpu_devices),
                (requested.wall_time_seconds, self.wall_time_seconds),
                (requested.source_scan_bytes, self.source_scan_bytes),
                (requested.output_bytes, self.output_bytes),
            )
        )


@dataclass(frozen=True, slots=True)
class AuthorityPolicy(CanonicalRecord):
    """Immutable delegated and non-delegable action boundary."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/authority-policy'

    policy_id: str
    delegator_id: str
    delegate_id: str
    scope_ids: tuple[str, ...]
    allowed_world_kinds: frozenset[WorldKind]
    allowed_actions: frozenset[AuthorityAction]
    allowed_source_classes: frozenset[SourceAccessClass]
    required_gate_ids: tuple[str, ...]
    nondelegable_actions: frozenset[AuthorityAction]
    budget_ceiling: ResourceBudget
    maximum_outcome_access: OutcomeAccess
    expires_at_utc: str | None = None
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("policy_id", self.policy_id),
            ("delegator_id", self.delegator_id),
            ("delegate_id", self.delegate_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_strings(self.scope_ids, field_name="scope_ids", allow_empty=False)
        require_sorted_unique_strings(
            self.required_gate_ids,
            field_name="required_gate_ids",
            allow_empty=False,
        )
        if not self.allowed_world_kinds:
            raise ValueError("authority policy must allow at least one world kind")
        if not self.allowed_actions:
            raise ValueError("authority policy must allow at least one action")
        if not self.allowed_source_classes:
            raise ValueError("authority policy must allow at least one source class")
        overlap = self.allowed_actions & self.nondelegable_actions
        if overlap:
            raise ValueError(
                "non-delegable actions cannot also be delegated: "
                f"{sorted(action.value for action in overlap)}"
            )
        if self.expires_at_utc is not None:
            parse_utc_timestamp(self.expires_at_utc, field_name="expires_at_utc")
        require_extensions(self.extensions)

    def refusal_reasons(
        self,
        *,
        action: AuthorityAction,
        world_kind: WorldKind,
        source_access: SourceAccessClass,
        passed_gate_ids: frozenset[str],
        requested_budget: ResourceBudget,
        requested_outcome_access: OutcomeAccess,
        requested_scope_id: str,
        proposer_id: str,
        approver_id: str,
        at_utc: str,
    ) -> tuple[str, ...]:
        """Return deterministic reason codes; an empty tuple means permitted."""

        validate_stable_id(proposer_id, field_name="proposer_id")
        validate_stable_id(approver_id, field_name="approver_id")
        validate_stable_id(requested_scope_id, field_name="requested_scope_id")
        now = parse_utc_timestamp(at_utc, field_name="at_utc")
        reasons: list[str] = []
        if action in self.nondelegable_actions:
            reasons.append("NON_DELEGABLE_ACTION")
        if action not in self.allowed_actions:
            reasons.append("ACTION_NOT_ALLOWED")
        if world_kind not in self.allowed_world_kinds:
            reasons.append("WORLD_NOT_ALLOWED")
        if requested_scope_id not in self.scope_ids:
            reasons.append("SCOPE_NOT_ALLOWED")
        if source_access not in self.allowed_source_classes:
            reasons.append("SOURCE_ACCESS_NOT_ALLOWED")
        if not set(self.required_gate_ids).issubset(passed_gate_ids):
            reasons.append("REQUIRED_GATES_MISSING")
        if not self.budget_ceiling.contains(requested_budget):
            reasons.append("RESOURCE_BUDGET_EXCEEDED")
        if (
            _OUTCOME_ACCESS_RANK[requested_outcome_access]
            > _OUTCOME_ACCESS_RANK[self.maximum_outcome_access]
        ):
            reasons.append("OUTCOME_ACCESS_EXCEEDED")
        if proposer_id in {self.delegate_id, approver_id}:
            reasons.append("SELF_APPROVAL_FORBIDDEN")
        if approver_id != self.delegate_id:
            reasons.append("APPROVER_NOT_DELEGATE")
        if self.expires_at_utc is not None and now > parse_utc_timestamp(
            self.expires_at_utc,
            field_name="expires_at_utc",
        ):
            reasons.append("POLICY_EXPIRED")
        return tuple(reasons)

    def permits(
        self,
        *,
        action: AuthorityAction,
        world_kind: WorldKind,
        source_access: SourceAccessClass,
        passed_gate_ids: frozenset[str],
        requested_budget: ResourceBudget,
        requested_outcome_access: OutcomeAccess,
        requested_scope_id: str,
        proposer_id: str,
        approver_id: str,
        at_utc: str,
    ) -> bool:
        """Convenience predicate over the same explicit request contract."""

        return not self.refusal_reasons(
            action=action,
            world_kind=world_kind,
            source_access=source_access,
            passed_gate_ids=passed_gate_ids,
            requested_budget=requested_budget,
            requested_outcome_access=requested_outcome_access,
            requested_scope_id=requested_scope_id,
            proposer_id=proposer_id,
            approver_id=approver_id,
            at_utc=at_utc,
        )

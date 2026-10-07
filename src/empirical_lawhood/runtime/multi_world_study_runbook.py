"""Canonical operation runbook generated only from implemented public routes."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_relative_locator,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)


FLAGSHIP_OPERATION_PATHS = (
    ("campaign.bundle-candidate-compile", "campaign compile-bundle"),
    ("campaign.close-bundle", "campaign close-bundle"),
    ("campaign.issue-bundle", "campaign issue-bundle"),
    ("campaign.bundle-status", "campaign bundle-status"),
    ("campaign.advance-bundle", "campaign transition-bundle"),
    ("campaign.compile-candidate", "campaign compile-candidate"),
    ("campaign.issue-extensions", "campaign issue-extensions"),
    ("campaign.resume", "campaign resume"),
    ("campaign.run", "campaign run"),
    ("campaign.status", "campaign status"),
)


@dataclass(frozen=True, slots=True)
class MultiWorldStudyOperationRequestContract(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-operation-request-contract'

    contract_id: str
    operation_id: str
    request_type: str
    field_names: tuple[str, ...]
    field_contract_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.contract_id, field_name="contract_id")
        validate_stable_id(self.operation_id, field_name="operation_id")
        validate_nonempty(self.request_type, field_name="request_type")
        require_sorted_unique_strings(self.field_names, field_name="field_names", allow_empty=False)
        validate_sha256(self.field_contract_sha256, field_name="field_contract_sha256")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyOperationHelpBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-operation-help-binding'

    binding_id: str
    operation_id: str
    command_path: str
    help_sha256: str
    help_size_bytes: int

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.operation_id, field_name="operation_id")
        validate_nonempty(self.command_path, field_name="command_path")
        validate_sha256(self.help_sha256, field_name="help_sha256")
        if self.help_size_bytes <= 0:
            raise ValueError("runbook help binding requires nonempty implemented help")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyOperationRunbookStep(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-operation-runbook-step'

    step_id: str
    ordinal: int
    operation_id: str
    command_path: str
    request_contract: ObjectIdentity
    help_binding: ObjectIdentity
    success_payload_schema_ids: tuple[str, ...]
    blocked_payload_schema_ids: tuple[str, ...]
    prerequisite_step_ids: tuple[str, ...]
    requires_write_confirmation: bool
    just_in_time_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.step_id, field_name="step_id")
        validate_stable_id(self.operation_id, field_name="operation_id")
        validate_nonempty(self.command_path, field_name="command_path")
        if self.ordinal < 0:
            raise ValueError("runbook step ordinal must be nonnegative")
        if self.request_contract.object_schema != MultiWorldStudyOperationRequestContract.SCHEMA:
            raise ValueError("runbook step binds another request-contract schema")
        if self.help_binding.object_schema != MultiWorldStudyOperationHelpBinding.SCHEMA:
            raise ValueError("runbook step binds another help-binding schema")
        for name, values in (
            ("success_payload_schema_ids", self.success_payload_schema_ids),
            ("blocked_payload_schema_ids", self.blocked_payload_schema_ids),
        ):
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            for value in values:
                validate_schema(value)
        require_sorted_unique_strings(
            self.prerequisite_step_ids,
            field_name="prerequisite_step_ids",
        )
        if self.just_in_time_authority != (self.operation_id == "campaign.advance-bundle"):
            raise ValueError("runbook just-in-time authority is assigned to another operation")


@dataclass(frozen=True, slots=True)
class MultiWorldStudyOperationRunbook(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/multi-world-study-operation-runbook'

    runbook_id: str
    config_root: str
    request_contracts: tuple[MultiWorldStudyOperationRequestContract, ...]
    help_bindings: tuple[MultiWorldStudyOperationHelpBinding, ...]
    steps: tuple[MultiWorldStudyOperationRunbookStep, ...]
    cli_tree_sha256: str
    contains_credentials: bool
    contains_authority_material: bool
    contains_future_reveal_record: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.runbook_id, field_name="runbook_id")
        validate_relative_locator(self.config_root)
        require_sorted_unique_ids(
            self.request_contracts,
            attribute="contract_id",
            field_name="request_contracts",
        )
        require_sorted_unique_ids(
            self.help_bindings,
            attribute="binding_id",
            field_name="help_bindings",
        )
        require_sorted_unique_ids(self.steps, attribute="step_id", field_name="steps")
        validate_sha256(self.cli_tree_sha256, field_name="cli_tree_sha256")
        if (
            self.contains_credentials
            or self.contains_authority_material
            or self.contains_future_reveal_record
        ):
            raise ValueError("flagship runbook contains prohibited authority material")
        expected = dict(FLAGSHIP_OPERATION_PATHS)
        if {value.operation_id: value.command_path for value in self.steps} != expected:
            raise ValueError("flagship runbook does not cover the exact implemented route")
        if tuple(
            value.ordinal for value in sorted(self.steps, key=lambda value: value.ordinal)
        ) != (tuple(range(len(self.steps)))):
            raise ValueError("flagship runbook ordinals are not contiguous")
        contracts = {value.operation_id: value for value in self.request_contracts}
        helps = {value.operation_id: value for value in self.help_bindings}
        if set(contracts) != set(expected) or set(helps) != set(expected):
            raise ValueError("flagship runbook request/help roster is incomplete")
        steps_by_id = {value.step_id: value for value in self.steps}
        for step in self.steps:
            if (
                step.request_contract
                != ObjectIdentity.from_record(
                    contracts[step.operation_id].contract_id,
                    contracts[step.operation_id],
                )
                or step.help_binding
                != ObjectIdentity.from_record(
                    helps[step.operation_id].binding_id,
                    helps[step.operation_id],
                )
                or not set(step.prerequisite_step_ids).issubset(steps_by_id)
            ):
                raise ValueError("flagship runbook step changes its implemented contract")


def build_multi_world_study_operation_runbook(
    *,
    config_root: str,
    request_contracts: tuple[MultiWorldStudyOperationRequestContract, ...],
    help_bindings: tuple[MultiWorldStudyOperationHelpBinding, ...],
    step_payload_schemas: dict[str, tuple[tuple[str, ...], tuple[str, ...]]],
    prerequisite_operations: dict[str, tuple[str, ...]],
    write_confirmation_operations: frozenset[str],
) -> MultiWorldStudyOperationRunbook:
    validate_relative_locator(config_root)
    expected = dict(FLAGSHIP_OPERATION_PATHS)
    contracts = {value.operation_id: value for value in request_contracts}
    helps = {value.operation_id: value for value in help_bindings}
    if (
        set(contracts) != set(expected)
        or set(helps) != set(expected)
        or set(step_payload_schemas) != set(expected)
        or set(prerequisite_operations) != set(expected)
    ):
        raise ValueError("runbook builder inputs do not cover the implemented operation roster")
    ordered_operations = (
        "campaign.compile-candidate",
        "campaign.issue-extensions",
        "campaign.bundle-candidate-compile",
        "campaign.issue-bundle",
        "campaign.run",
        "campaign.status",
        "campaign.resume",
        "campaign.advance-bundle",
        "campaign.bundle-status",
        "campaign.close-bundle",
    )
    step_id_by_operation = {
        operation: f"runbook-step.{index:02d}.{operation}"
        for index, operation in enumerate(ordered_operations)
    }
    steps = tuple(
        sorted(
            (
                MultiWorldStudyOperationRunbookStep(
                    step_id=step_id_by_operation[operation],
                    ordinal=index,
                    operation_id=operation,
                    command_path=expected[operation],
                    request_contract=ObjectIdentity.from_record(
                        contracts[operation].contract_id,
                        contracts[operation],
                    ),
                    help_binding=ObjectIdentity.from_record(
                        helps[operation].binding_id,
                        helps[operation],
                    ),
                    success_payload_schema_ids=tuple(sorted(step_payload_schemas[operation][0])),
                    blocked_payload_schema_ids=tuple(sorted(step_payload_schemas[operation][1])),
                    prerequisite_step_ids=tuple(
                        sorted(
                            step_id_by_operation[value]
                            for value in prerequisite_operations[operation]
                        )
                    ),
                    requires_write_confirmation=operation in write_confirmation_operations,
                    just_in_time_authority=operation == "campaign.advance-bundle",
                )
                for index, operation in enumerate(ordered_operations)
            ),
            key=lambda value: value.step_id,
        )
    )
    sorted_contracts = tuple(sorted(request_contracts, key=lambda value: value.contract_id))
    sorted_helps = tuple(sorted(help_bindings, key=lambda value: value.binding_id))
    tree_sha = hashlib.sha256(
        canonical_json_bytes(
            tuple(
                (value.operation_id, value.command_path, value.help_sha256)
                for value in sorted_helps
            )
        )
    ).hexdigest()
    seed = {
        "request_contracts": tuple(
            ObjectIdentity.from_record(value.contract_id, value) for value in sorted_contracts
        ),
        "help_bindings": tuple(
            ObjectIdentity.from_record(value.binding_id, value) for value in sorted_helps
        ),
        "steps": tuple(ObjectIdentity.from_record(value.step_id, value) for value in steps),
        "cli_tree_sha256": tree_sha,
    }
    return MultiWorldStudyOperationRunbook(
        runbook_id=f"flagship-operation-runbook.{hashlib.sha256(canonical_json_bytes(seed)).hexdigest()[:32]}",
        config_root=config_root,
        request_contracts=sorted_contracts,
        help_bindings=sorted_helps,
        steps=steps,
        cli_tree_sha256=tree_sha,
        contains_credentials=False,
        contains_authority_material=False,
        contains_future_reveal_record=False,
    )


__all__ = [
    "FLAGSHIP_OPERATION_PATHS",
    'MultiWorldStudyOperationHelpBinding',
    'MultiWorldStudyOperationRequestContract',
    'MultiWorldStudyOperationRunbookStep',
    'MultiWorldStudyOperationRunbook',
    'build_multi_world_study_operation_runbook',
]

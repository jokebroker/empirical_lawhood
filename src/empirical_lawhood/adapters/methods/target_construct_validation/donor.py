"Bind target construct validation to the current margin structural recurrence forecast runtime port."

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.structural_recurrence_runtime import MarginStructuralRecurrenceForecastPortKind, method_source_manifest, semantic_port_inventory
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class TargetConstructValidationDonorBinding(CanonicalRecord):
    """current donor-binding record proving that construct-validation reuses the current registered donor."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-donor-binding'

    binding_id: str
    method_source_manifest: ObjectIdentity
    semantic_port_inventory: ObjectIdentity
    exact_frozen_operation_ids: tuple[str, ...]
    additive_boundary_operation_ids: tuple[str, ...]
    scientific_operators_changed: bool
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        require_sorted_unique_strings(
            self.exact_frozen_operation_ids,
            field_name="exact_frozen_operation_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.additive_boundary_operation_ids,
            field_name="additive_boundary_operation_ids",
            allow_empty=False,
        )
        if len(self.exact_frozen_operation_ids) != 4:
            raise ValueError("target construct validation requires the exact four frozen structural recurrence operations")
        if self.additive_boundary_operation_ids != ("TARGET_COMPATIBILITY_AUDIT",):
            raise ValueError("target construct validation must keep target compatibility visibly additive")
        if self.scientific_operators_changed:
            raise ValueError("target construct validation cannot change frozen structural recurrence semantics")
        if self.protected_outcome_access_count != 0:
            raise ValueError("donor binding must be completed without protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("donor binding must remain outcome-blind")


def current_donor_binding() -> TargetConstructValidationDonorBinding:
    """Construct the construct-validation binding from the current, already registered port."""

    manifest = method_source_manifest()
    inventory = semantic_port_inventory()
    exact = tuple(
        sorted(
            binding.operation.value
            for binding in inventory.bindings
            if binding.port_kind is MarginStructuralRecurrenceForecastPortKind.EXACT_FROZEN_FUNCTION
        )
    )
    additive = tuple(
        sorted(
            binding.operation.value
            for binding in inventory.bindings
            if binding.port_kind is MarginStructuralRecurrenceForecastPortKind.ADDITIVE_BOUNDARY_AUDIT
        )
    )
    return TargetConstructValidationDonorBinding(
        binding_id='target-construct-validation.frozen-margin-structural-recurrence-forecast-donor',
        method_source_manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        semantic_port_inventory=ObjectIdentity.from_record(
            inventory.inventory_id,
            inventory,
        ),
        exact_frozen_operation_ids=exact,
        additive_boundary_operation_ids=additive,
        scientific_operators_changed=inventory.scientific_operators_changed,
        protected_outcome_access_count=0,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )

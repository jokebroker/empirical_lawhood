"Standard outcome-blind formal-analysis capability and source contracts.\n\nThese formal-analysis records make the formal-gap register executable without\nchanging the frozen register/coverage records.  Source inventories describe\nwhat a qualified denominator can provide; method bindings map the register's\nscientific family IDs to exact static capabilities.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.worlds import EvidenceUnitScope

from .formal_gaps import (
    FormalGapApplicability,
    FormalGapEvidenceWorld,
    FormalGapRegister,
)


class FormalMethodRole(StrEnum):
    ESTIMATOR = "ESTIMATOR"
    MULTIPLICITY = "MULTIPLICITY"


@dataclass(frozen=True, slots=True)
class FormalMethodBinding(CanonicalRecord):
    """Map one formal family to an exact registered implementation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-method-binding'

    family_id: str
    role: FormalMethodRole
    capability_key: str
    capability_version: str
    implementation_sha256: str
    supported_gap_ids: tuple[str, ...]
    input_schema_id: str
    output_schema_id: str
    maximum_claim_ceiling: EvidenceCeiling
    maximum_outcome_access: OutcomeAccess

    @property
    def binding_id(self) -> str:
        return f"{self.role.value.lower()}.{self.family_id}"

    def __post_init__(self) -> None:
        for name, value in (
            ("family_id", self.family_id),
            ("capability_key", self.capability_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.capability_version)
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )
        require_sorted_unique_strings(
            self.supported_gap_ids,
            field_name="supported_gap_ids",
            allow_empty=False,
        )
        validate_schema(self.input_schema_id)
        validate_schema(self.output_schema_id)
        if self.maximum_outcome_access not in {
            OutcomeAccess.DEVELOPMENT_VISIBLE,
            OutcomeAccess.EVALUATION_SEALED,
            OutcomeAccess.EVALUATOR_REVEAL,
        }:
            raise ValueError("formal method binding has incompatible outcome access")
        if self.maximum_claim_ceiling in {
            EvidenceCeiling.ADMISSION,
            EvidenceCeiling.CONTROLLER_USE,
        }:
            raise ValueError("formal method binding cannot exceed local law")


@dataclass(frozen=True, slots=True)
class FormalMethodCatalog(CanonicalRecord):
    """Closed family-to-capability map used by standard candidate compilation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-method-catalog'

    catalog_id: str
    bindings: tuple[FormalMethodBinding, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.catalog_id, field_name="catalog_id")
        require_sorted_unique_ids(
            self.bindings,
            attribute="binding_id",
            field_name="bindings",
        )
        if not self.bindings:
            raise ValueError("formal method catalog cannot be empty")
        keys = {(value.role, value.family_id) for value in self.bindings}
        if len(keys) != len(self.bindings):
            raise ValueError("formal method catalog contains duplicate family roles")

    def resolve(
        self,
        role: FormalMethodRole,
        family_id: str,
    ) -> FormalMethodBinding | None:
        return next(
            (
                value
                for value in self.bindings
                if value.role is role and value.family_id == family_id
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class FormalGapSourceCapabilityInventory(CanonicalRecord):
    """Qualified outcome-blind operands available in one denominator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/formal-gap-source-capability-inventory'

    inventory_id: str
    denominator_id: str
    evidence_world: FormalGapEvidenceWorld
    source_materializations: tuple[ObjectIdentity, ...]
    present_operand_ids: tuple[str, ...]
    satisfied_prerequisite_ids: tuple[str, ...]
    independent_unit_ids: tuple[str, ...]
    independent_unit_scope: EvidenceUnitScope
    numerical_view_ids: tuple[str, ...]
    available_estimator_family_ids: tuple[str, ...]
    available_control_ids: tuple[str, ...]
    multiplicity_family_ids: tuple[str, ...]
    denominator_inapplicable_gap_ids: tuple[str, ...]
    resource_blocked_gap_ids: tuple[str, ...]
    requested_claim_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("inventory_id", self.inventory_id),
            ("denominator_id", self.denominator_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.source_materializations,
            attribute="object_id",
            field_name="source_materializations",
        )
        if not self.source_materializations:
            raise ValueError("formal source inventory requires a qualified source")
        for name, values in (
            ("present_operand_ids", self.present_operand_ids),
            ("satisfied_prerequisite_ids", self.satisfied_prerequisite_ids),
            ("independent_unit_ids", self.independent_unit_ids),
            ("numerical_view_ids", self.numerical_view_ids),
            (
                "available_estimator_family_ids",
                self.available_estimator_family_ids,
            ),
            ("available_control_ids", self.available_control_ids),
            ("multiplicity_family_ids", self.multiplicity_family_ids),
            (
                "denominator_inapplicable_gap_ids",
                self.denominator_inapplicable_gap_ids,
            ),
            ("resource_blocked_gap_ids", self.resource_blocked_gap_ids),
        ):
            require_sorted_unique_strings(values, field_name=name)
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("formal source inventory must remain outcome-blind")


def derive_formal_gap_applicability(
    register: FormalGapRegister,
    inventory: FormalGapSourceCapabilityInventory,
) -> tuple[FormalGapApplicability, ...]:
    """Derive every row from one independently qualified source inventory."""

    known_gap_ids = {value.gap_id for value in register.gaps}
    configured_gap_ids = set(inventory.denominator_inapplicable_gap_ids) | set(
        inventory.resource_blocked_gap_ids
    )
    unknown = configured_gap_ids - known_gap_ids
    if unknown:
        raise ValueError(f"formal source inventory references unknown gaps: {sorted(unknown)}")
    overlap = set(inventory.denominator_inapplicable_gap_ids) & set(
        inventory.resource_blocked_gap_ids
    )
    if overlap:
        raise ValueError(f"formal source inventory gives conflicting gap facts: {sorted(overlap)}")
    return tuple(
        FormalGapApplicability(
            gap_id=gap.gap_id,
            evidence_world=inventory.evidence_world,
            present_operand_ids=inventory.present_operand_ids,
            satisfied_prerequisite_ids=inventory.satisfied_prerequisite_ids,
            independent_unit_ids=inventory.independent_unit_ids,
            independent_unit_scope=inventory.independent_unit_scope,
            numerical_view_ids=inventory.numerical_view_ids,
            available_estimator_family_ids=(inventory.available_estimator_family_ids),
            available_control_ids=inventory.available_control_ids,
            multiplicity_family_ids=inventory.multiplicity_family_ids,
            requested_claim_ceiling=inventory.requested_claim_ceiling,
            denominator_applicable=(gap.gap_id not in inventory.denominator_inapplicable_gap_ids),
            resource_envelope_satisfied=(gap.gap_id not in inventory.resource_blocked_gap_ids),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
        )
        for gap in register.gaps
    )


__all__ = [
    "FormalGapSourceCapabilityInventory",
    "FormalMethodBinding",
    "FormalMethodCatalog",
    "FormalMethodRole",
    "derive_formal_gap_applicability",
]

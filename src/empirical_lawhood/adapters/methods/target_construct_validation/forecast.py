"""Direct, finite target construct validation projection records with no target execution."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_sha256,
    validate_stable_id,
)

from .contracts import TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID


class TargetConstructValidationForecastDisposition(StrEnum):
    EXACT = "EXACT"
    AMBIGUOUS = "AMBIGUOUS"
    NO_PREDICTION = "NO_PREDICTION"


@dataclass(frozen=True, slots=True)
class TargetConstructValidationStateProjection(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-state-projection'

    projection_id: str
    donor_state_id: str
    target_state_id: str

    def __post_init__(self) -> None:
        for name in ("projection_id", "donor_state_id", "target_state_id"):
            validate_stable_id(getattr(self, name), field_name=name)


@dataclass(frozen=True, slots=True)
class TargetConstructValidationDirectForecastCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-direct-forecast-cell'

    cell_id: str
    target_id: str
    primary_relation_id: str
    denominator_stratum_id: str
    history_condition_id: str
    native_action_id: str
    receiver_id: str
    horizon_id: str
    legal_state_ids: tuple[str, ...]
    unsafe_admission_state_ids: tuple[str, ...]
    complete_unit_ids_sha256: str
    authored_before_development: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "target_id",
            "primary_relation_id",
            "denominator_stratum_id",
            "history_condition_id",
            "native_action_id",
            "receiver_id",
            "horizon_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.legal_state_ids,
            field_name="legal_state_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.unsafe_admission_state_ids,
            field_name="unsafe_admission_state_ids",
        )
        if not set(self.unsafe_admission_state_ids).issubset(self.legal_state_ids):
            raise ValueError("unsafe states lie outside the target-native alphabet")
        validate_sha256(
            self.complete_unit_ids_sha256,
            field_name="complete_unit_ids_sha256",
        )
        if self.primary_relation_id != TARGET_CONSTRUCT_VALIDATION_PRIMARY_RELATION_ID:
            raise ValueError("forecast cell does not bind the sole primary relation")
        if not self.authored_before_development:
            raise ValueError("forecast cells cannot be added after development")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("forecast cells must freeze outcome-blindly")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationForecastEmission(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-forecast-emission'

    emission_id: str
    cell: ObjectIdentity
    donor_prediction: ObjectIdentity
    donor_field: str
    selected_map: ObjectIdentity
    role_binding_ids: tuple[str, ...]
    projections: tuple[TargetConstructValidationStateProjection, ...]
    donor_state_ids: tuple[str, ...]
    emitted_state_ids: tuple[str, ...]
    disposition: TargetConstructValidationForecastDisposition
    derivation_rule: str
    rationale: str
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.emission_id, field_name="emission_id")
        validate_nonempty(self.donor_field, field_name="donor_field")
        validate_nonempty(self.derivation_rule, field_name="derivation_rule")
        validate_nonempty(self.rationale, field_name="rationale")
        require_sorted_unique_strings(
            self.role_binding_ids,
            field_name="role_binding_ids",
            allow_empty=False,
        )
        require_sorted_unique_ids(
            self.projections,
            attribute="projection_id",
            field_name="projections",
        )
        require_sorted_unique_strings(
            self.donor_state_ids,
            field_name="donor_state_ids",
        )
        require_sorted_unique_strings(
            self.emitted_state_ids,
            field_name="emitted_state_ids",
        )
        table = {value.donor_state_id: value.target_state_id for value in self.projections}
        if len(table) != len(self.projections):
            raise ValueError("state projection reuses a donor state")
        projected = tuple(
            sorted({table[value] for value in self.donor_state_ids if value in table})
        )
        if len(projected) != len(self.donor_state_ids):
            raise ValueError("every donor state must have exactly one frozen projection")
        expected_count = {
            TargetConstructValidationForecastDisposition.EXACT: 1,
            TargetConstructValidationForecastDisposition.AMBIGUOUS: 2,
            TargetConstructValidationForecastDisposition.NO_PREDICTION: 0,
        }[self.disposition]
        if len(self.emitted_state_ids) != expected_count:
            raise ValueError("forecast disposition and emitted state cardinality differ")
        if self.disposition is TargetConstructValidationForecastDisposition.NO_PREDICTION:
            if self.donor_state_ids or self.projections:
                raise ValueError("NO_PREDICTION cannot hide an available donor projection")
        elif projected != self.emitted_state_ids:
            raise ValueError("emission adds content outside the frozen donor projection")
        if self.protected_outcome_access_count != 0:
            raise ValueError("direct forecast cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("direct forecast issue must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class TargetConstructValidationPredictionIssue(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/target-construct-validation/target-construct-validation-prediction-issue'

    issue_id: str
    target_id: str
    relation_binding: ObjectIdentity
    cells: tuple[TargetConstructValidationDirectForecastCell, ...]
    emissions: tuple[TargetConstructValidationForecastEmission, ...]
    protected_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.issue_id, field_name="issue_id")
        validate_stable_id(self.target_id, field_name="target_id")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_ids(
            self.emissions,
            attribute="emission_id",
            field_name="emissions",
        )
        if not 6 <= len(self.cells) <= 24:
            raise ValueError("prediction issue must contain 6--24 primary cells")
        if any(cell.target_id != self.target_id for cell in self.cells):
            raise ValueError("prediction issue cells cross targets")
        cell_identities = {
            ObjectIdentity.from_record(cell.cell_id, cell): cell.cell_id for cell in self.cells
        }
        if len(self.emissions) != len(self.cells) or {
            emission.cell for emission in self.emissions
        } != set(cell_identities):
            raise ValueError("prediction issue must emit exactly once for every frozen cell")
        if self.protected_outcome_access_count != 0:
            raise ValueError("prediction issue cannot access protected outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("prediction issue must remain outcome-blind")

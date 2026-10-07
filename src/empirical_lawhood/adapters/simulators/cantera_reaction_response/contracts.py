"""Strict native records for the selective dependence response Cantera CSTR target."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_semantic_version,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class CanteraReactionResponseCanteraDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/cantera-reaction-response/cantera-reaction-response-cantera-design'

    design_id: str
    target_id: str
    cantera_version: str
    mechanism_id: str
    phase_name: str
    reactor_model_id: str
    denominator_ids: tuple[str, ...]
    history_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    horizon_ids: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    action_multipliers: tuple[Decimal, ...]
    horizon_seconds: tuple[Decimal, ...]
    base_residence_seconds: Decimal
    heat_transfer_coefficients: tuple[Decimal, ...]
    target_temperature_lower: Decimal
    target_temperature_upper: Decimal
    minimum_conversion: Decimal
    maximum_co_mole_fraction: Decimal
    maximum_peak_temperature: Decimal
    maximum_element_error: Decimal
    outside_action_multiplier: Decimal
    maximum_solver_steps: int
    maximum_wall_seconds_per_unit: Decimal
    construct_validation_numeric_identity_reused: bool
    development_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("design_id", "target_id", "mechanism_id", "reactor_model_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.cantera_version)
        validate_nonempty(self.phase_name, field_name="phase_name")
        for name in (
            "denominator_ids",
            "history_ids",
            "action_ids",
            "horizon_ids",
            "receiver_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if self.target_id != "target.cantera-selective-dependence-response":
            raise ValueError("Cantera target identity differs")
        if len(self.denominator_ids) != 2 or len(self.history_ids) != 2:
            raise ValueError("Cantera design requires two denominators and histories")
        if len(self.action_ids) != 4 or len(self.action_multipliers) != 3:
            raise ValueError("Cantera design requires low/hold/high plus one refusal")
        if len(self.horizon_ids) != 2 or len(self.horizon_seconds) != 2:
            raise ValueError("Cantera design requires two horizons")
        for name in (
            "base_residence_seconds",
            "target_temperature_lower",
            "target_temperature_upper",
            "minimum_conversion",
            "maximum_co_mole_fraction",
            "maximum_peak_temperature",
            "maximum_element_error",
            "outside_action_multiplier",
            "maximum_wall_seconds_per_unit",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        for index, value in enumerate(
            (*self.action_multipliers, *self.horizon_seconds, *self.heat_transfer_coefficients)
        ):
            validate_decimal(value, field_name=f"positive_design_value_{index}", minimum=Decimal(0))
            if value <= 0:
                raise ValueError("Cantera design values must be positive")
        if not self.target_temperature_lower < self.target_temperature_upper:
            raise ValueError("temperature band is empty")
        if self.maximum_solver_steps <= 0:
            raise ValueError("solver step ceiling must be positive")
        if self.construct_validation_numeric_identity_reused:
            raise ValueError("selective-response cannot reuse the construct-validation numeric task identity")
        if self.development_outcome_count:
            raise ValueError("design freeze cannot use development outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("Cantera design must be outcome blind")


@dataclass(frozen=True, slots=True)
class CanteraReactionResponseCanteraSourceQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/cantera-reaction-response/cantera-reaction-response-cantera-source-qualification'

    qualification_id: str
    design_id: str
    observed_cantera_version: str
    version_matches_design: bool
    mechanism_resolved: bool
    methane_species_present: bool
    carbon_monoxide_species_present: bool
    energy_enabled: bool
    action_stage_observable: bool
    reset_reproducible: bool
    elemental_balance_qualified: bool
    source_ready: bool
    canary_panel: ObjectIdentity
    canary_result_identities: tuple[ObjectIdentity, ...]
    excluded_complete_unit_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("qualification_id", "design_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.observed_cantera_version)
        observed = all(
            (
                self.mechanism_resolved,
                self.version_matches_design,
                self.methane_species_present,
                self.carbon_monoxide_species_present,
                self.energy_enabled,
                self.action_stage_observable,
                self.reset_reproducible,
                self.elemental_balance_qualified,
            )
        )
        if self.source_ready != observed:
            raise ValueError("Cantera source readiness is not qualification-derived")
        if self.canary_panel.object_schema != 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-panel':
            raise ValueError("Cantera qualification does not bind a complete canary panel")
        require_sorted_unique_strings(
            tuple(value.object_id for value in self.canary_result_identities),
            field_name="canary_result_ids",
            allow_empty=False,
        )
        if self.excluded_complete_unit_count != len(self.canary_result_identities):
            raise ValueError("Cantera excluded count differs from canary identities")
        if self.excluded_complete_unit_count < 1:
            raise ValueError("real qualification requires an excluded complete unit")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("excluded Cantera canary is development visible")


__all__ = ['CanteraReactionResponseCanteraDesign', 'CanteraReactionResponseCanteraSourceQualification']

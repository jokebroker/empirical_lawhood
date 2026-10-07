"""Strict native records for the selective dependence response FiPy target."""

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
    validate_semantic_version,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class FipyReactionDiffusionResponseFiPyDesign(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/fipy-reaction-diffusion-response/fipy-reaction-diffusion-response-fi-py-design'

    design_id: str
    target_id: str
    fipy_version: str
    equation_id: str
    denominator_ids: tuple[str, ...]
    history_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    horizon_ids: tuple[str, ...]
    receiver_ids: tuple[str, ...]
    diffusivities: tuple[Decimal, ...]
    action_rates: tuple[Decimal, ...]
    horizon_seconds: tuple[Decimal, ...]
    decay_rate: Decimal
    action_duration_seconds: Decimal
    mesh_cells: int
    domain_length: Decimal
    timestep_seconds: Decimal
    target_downstream_lower: Decimal
    target_downstream_upper: Decimal
    maximum_peak_field: Decimal
    maximum_boundary_flux: Decimal
    maximum_balance_error: Decimal
    nonnegativity_tolerance: Decimal
    outside_action_rate: Decimal
    maximum_solver_steps: int
    construct_validation_numeric_identity_reused: bool
    development_outcome_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("design_id", "target_id", "equation_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.fipy_version)
        for name in (
            "denominator_ids",
            "history_ids",
            "action_ids",
            "horizon_ids",
            "receiver_ids",
        ):
            require_sorted_unique_strings(getattr(self, name), field_name=name, allow_empty=False)
        if self.target_id != "target.fipy-selective-dependence-response":
            raise ValueError("FiPy target identity differs")
        if len(self.denominator_ids) != 2 or len(self.diffusivities) != 2:
            raise ValueError("FiPy design requires two diffusivity denominators")
        if len(self.history_ids) != 2:
            raise ValueError("FiPy design requires cleared and residual histories")
        if len(self.action_ids) != 4 or len(self.action_rates) != 3:
            raise ValueError("FiPy design requires inject/hold/remove plus refusal")
        if len(self.horizon_ids) != 2 or len(self.horizon_seconds) != 2:
            raise ValueError("FiPy design requires two horizons")
        for index, value in enumerate(
            (
                *self.diffusivities,
                *self.horizon_seconds,
                self.decay_rate,
                self.action_duration_seconds,
                self.domain_length,
                self.timestep_seconds,
                self.target_downstream_lower,
                self.target_downstream_upper,
                self.maximum_peak_field,
                self.maximum_boundary_flux,
                self.maximum_balance_error,
                self.nonnegativity_tolerance,
            )
        ):
            validate_decimal(value, field_name=f"design_value_{index}", minimum=Decimal(0))
        for index, value in enumerate((*self.action_rates, self.outside_action_rate)):
            validate_decimal(value, field_name=f"signed_action_value_{index}")
        if not self.target_downstream_lower < self.target_downstream_upper:
            raise ValueError("downstream target band is empty")
        if self.mesh_cells < 20 or self.maximum_solver_steps < 1:
            raise ValueError("FiPy numerical resolution or step ceiling is invalid")
        if self.construct_validation_numeric_identity_reused:
            raise ValueError("selective-response cannot reuse the construct-validation numeric task identity")
        if self.development_outcome_count:
            raise ValueError("FiPy design cannot use development outcomes")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("FiPy design must be outcome blind")


@dataclass(frozen=True, slots=True)
class FipyReactionDiffusionResponseFiPySourceQualification(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/fipy-reaction-diffusion-response/fipy-reaction-diffusion-response-fi-py-source-qualification'

    qualification_id: str
    design_id: str
    observed_fipy_version: str
    version_matches_design: bool
    equation_constructed: bool
    signed_source_observable: bool
    realized_integral_observable: bool
    reset_reproducible: bool
    balance_qualified: bool
    source_ready: bool
    canary_panel: ObjectIdentity
    canary_result_identities: tuple[ObjectIdentity, ...]
    excluded_complete_unit_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("qualification_id", "design_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_semantic_version(self.observed_fipy_version)
        observed = all(
            (
                self.equation_constructed,
                self.version_matches_design,
                self.signed_source_observable,
                self.realized_integral_observable,
                self.reset_reproducible,
                self.balance_qualified,
            )
        )
        if self.source_ready != observed:
            raise ValueError("FiPy source readiness is not qualification-derived")
        if self.canary_panel.object_schema != 'empirical-lawhood/methods/selective-dependence-response/selective-dependence-response-target-panel':
            raise ValueError("FiPy qualification does not bind a complete canary panel")
        require_sorted_unique_strings(
            tuple(value.object_id for value in self.canary_result_identities),
            field_name="canary_result_ids",
            allow_empty=False,
        )
        if self.excluded_complete_unit_count != len(self.canary_result_identities):
            raise ValueError("FiPy excluded count differs from canary identities")
        if self.excluded_complete_unit_count < 1:
            raise ValueError("real qualification requires an excluded complete unit")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("excluded FiPy canary is development visible")


__all__ = ['FipyReactionDiffusionResponseFiPyDesign', 'FipyReactionDiffusionResponseFiPySourceQualification']

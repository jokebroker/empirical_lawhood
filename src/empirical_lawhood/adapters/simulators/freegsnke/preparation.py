"""Outcome-separated preparation qualification for the independent substrate grounding FreeGSNKE target."""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)

from .contracts import FreeGsnkeActionChart, FreeGsnkeNumericalView, FreeGsnkePhase, FreeGsnkePreparation, FreeGsnkeSavedPreparation, FreeGsnkeSourceBinding, freegsnke_saved_preparation_artifact


FREEGSNKE_SOLVER_CLOCK_TOLERANCE_S = Decimal("1e-12")


def canonicalize_freegsnke_solver_clock(
    value: Decimal | str | float,
    *,
    timestep_s: Decimal,
) -> Decimal:
    """Recover one upstream float-formatted clock as an exact solver tick."""

    clock = value if isinstance(value, Decimal) else Decimal(str(value))
    validate_decimal(clock, field_name="clock", minimum=Decimal(0))
    validate_decimal(timestep_s, field_name="timestep_s", minimum=Decimal(0))
    if timestep_s == 0:
        raise ValueError("FreeGSNKE solver timestep must be positive")
    steps = (clock / timestep_s).to_integral_value()
    snapped = steps * timestep_s
    if abs(snapped - clock) > FREEGSNKE_SOLVER_CLOCK_TOLERANCE_S:
        raise ValueError("upstream FreeGSNKE clock is not solver-tick aligned")
    return snapped


@dataclass(frozen=True, slots=True)
class FreeGsnkePreparationWorkerInput(CanonicalRecord):
    """One fixed, target-outcome-blind equilibrium authoring request."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-preparation-worker-input'

    input_id: str
    source_binding: FreeGsnkeSourceBinding
    recipe: FreeGsnkePreparation
    numerical_view: FreeGsnkeNumericalView
    inverse_recipe_id: str
    solenoid_current_a: Decimal
    inverse_relative_tolerance: Decimal
    inverse_relative_psit_update: Decimal
    maximum_elongation_error: Decimal
    target_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.input_id, field_name="input_id")
        validate_stable_id(self.inverse_recipe_id, field_name="inverse_recipe_id")
        if self.recipe.saved_state is not None:
            raise ValueError("FreeGSNKE preparation worker requires an unmaterialized recipe")
        if self.recipe.phase not in {
            FreeGsnkePhase.SCOUT,
            FreeGsnkePhase.DEVELOPMENT,
            FreeGsnkePhase.EVALUATION,
            FreeGsnkePhase.PROSPECTIVE_VALIDATION,
        }:
            raise ValueError("FreeGSNKE preparation worker phase differs")
        validate_decimal(self.solenoid_current_a, field_name="solenoid_current_a")
        for name in (
            "inverse_relative_tolerance",
            "inverse_relative_psit_update",
            "maximum_elongation_error",
        ):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value == 0:
                raise ValueError(f"{name} must be positive")
        if self.inverse_recipe_id != "mastu-like-lsn-p4-p5-fixed-geometry":
            raise ValueError("FreeGSNKE inverse recipe is not statically registered")
        if self.target_outcome_access_count:
            raise ValueError("FreeGSNKE preparation crossed target outcome access")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("FreeGSNKE preparation input must remain target-outcome blind")


@dataclass(frozen=True, slots=True)
class FreeGsnkePreparedUnit(CanonicalRecord):
    """One qualified saved preparation plus its preparation-local action chart."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/freegsnke/free-gsnke-prepared-unit'

    unit_id: str
    worker_input: ObjectIdentity
    recipe: FreeGsnkePreparation
    preparation: FreeGsnkePreparation | None
    action_chart: FreeGsnkeActionChart | None
    saved_preparation: FreeGsnkeSavedPreparation | None
    eligible: bool
    reason_codes: tuple[str, ...]
    excluded_from_target_evidence: bool
    excluded_qualification_unit_count: int
    target_outcome_access_count: int
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        if self.worker_input.object_schema != FreeGsnkePreparationWorkerInput.SCHEMA:
            raise ValueError("FreeGSNKE prepared unit worker input differs")
        if self.recipe.saved_state is not None:
            raise ValueError("FreeGSNKE prepared-unit recipe is already materialized")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        payload_values = (
            self.preparation,
            self.action_chart,
            self.saved_preparation,
        )
        payload_present = all(value is not None for value in payload_values)
        if self.eligible:
            if not payload_present or self.reason_codes:
                raise ValueError("eligible FreeGSNKE prepared unit lacks its payload")
            assert self.action_chart is not None
            assert self.saved_preparation is not None
            expected_preparation = replace(
                self.recipe,
                saved_state=freegsnke_saved_preparation_artifact(self.saved_preparation),
            )
            if (
                self.preparation != expected_preparation
                or self.saved_preparation.preparation_id != self.recipe.preparation_id
                or self.saved_preparation.preparation_recipe_sha256 != self.recipe.fingerprint()
                or self.saved_preparation.phase is not self.recipe.phase
                or self.saved_preparation.action_chart
                != ObjectIdentity.from_record(
                    self.action_chart.chart_id,
                    self.action_chart,
                )
                or not self.saved_preparation.eligible
            ):
                raise ValueError("FreeGSNKE prepared-unit operands differ")
        elif any(value is not None for value in payload_values) or not self.reason_codes:
            raise ValueError("ineligible FreeGSNKE preparation must retain a typed stop")
        if (
            not self.excluded_from_target_evidence
            or self.excluded_qualification_unit_count != 1
            or self.target_outcome_access_count
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("FreeGSNKE prepared unit crossed target evidence")


__all__ = [
    "FREEGSNKE_SOLVER_CLOCK_TOLERANCE_S",
    'FreeGsnkePreparationWorkerInput',
    'FreeGsnkePreparedUnit',
    "canonicalize_freegsnke_solver_clock",
]

"""Bounded target-owned input for selective dependence response native development panels."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseCompleteUnitResult, SelectiveDependenceResponsePhase
from empirical_lawhood.adapters.simulators.cantera_reaction_response.contracts import CanteraReactionResponseCanteraDesign
from empirical_lawhood.adapters.simulators.fipy_reaction_diffusion_response.contracts import FipyReactionDiffusionResponseFiPyDesign
from empirical_lawhood.adapters.methods.selective_dependence_response.preparation_inputs import SelectiveDependenceResponsePreparationInput
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id


@dataclass(frozen=True, slots=True)
class ReactionDiffusionDevelopmentInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reaction-diffusion-development/reaction-diffusion-development-input'

    config_id: str
    independent_unit_id: str
    design: CanteraReactionResponseCanteraDesign | FipyReactionDiffusionResponseFiPyDesign
    preparation_input: SelectiveDependenceResponsePreparationInput | None = None

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        validate_stable_id(self.independent_unit_id)
        if not self.config_id.startswith("empirical-lawhood-"):
            raise ValueError("quick-start config needs a new target-owned identity")
        if not self.independent_unit_id.startswith("unit.empirical-lawhood-"):
            raise ValueError("quick-start unit needs a new target-owned identity")
        if not self.design.design_id.startswith("empirical-lawhood-"):
            raise ValueError("quick-start design needs a new target-owned identity")
        if isinstance(self.design, CanteraReactionResponseCanteraDesign):
            design = self.design
            if (
                design.denominator_ids != ("heat-loss-high", "heat-loss-low")
                or design.history_ids != ("cold-checkpoint", "hot-checkpoint")
                or design.action_ids
                != ("flow-high", "flow-hold", "flow-low", "flow-outside")
                or design.horizon_ids != ("long", "short")
                or design.receiver_ids
                != (
                    "carbon-monoxide",
                    "element-error",
                    "methane-conversion",
                    "peak-temperature",
                    "temperature",
                )
            ):
                raise ValueError(
                    "Cantera quick-start axes differ from the native panel"
                )
            if (
                not design.action_multipliers[0]
                < design.action_multipliers[1]
                < design.action_multipliers[2]
                < design.outside_action_multiplier
                <= Decimal(2)
                or design.action_multipliers[1] != 1
                or max(design.horizon_seconds) > 1
                or design.maximum_solver_steps > 20_000
                or design.maximum_wall_seconds_per_unit > 30
            ):
                raise ValueError("Cantera quick-start action or resource bound differs")
        else:
            design = self.design
            if (
                design.denominator_ids != ("diffusivity-high", "diffusivity-low")
                or design.history_ids != ("cleared-checkpoint", "residual-checkpoint")
                or design.action_ids
                != ("source-hold", "source-inject", "source-outside", "source-remove")
                or design.horizon_ids != ("long", "short")
                or design.receiver_ids
                != (
                    "balance-error",
                    "boundary-flux",
                    "downstream-mean",
                    "field-mass",
                    "field-maximum",
                    "field-minimum",
                )
            ):
                raise ValueError("FiPy quick-start axes differ from the native panel")
            if (
                not design.outside_action_rate
                < design.action_rates[0]
                < design.action_rates[1]
                < design.action_rates[2]
                <= Decimal("0.5")
                or design.action_rates[1] != 0
                or design.mesh_cells > 120
                or design.maximum_solver_steps > 1000
                or design.timestep_seconds < Decimal("0.001")
                or max(design.horizon_seconds) > 1
            ):
                raise ValueError("FiPy quick-start action or resource bound differs")

        self.validate_preparation_input()

    def validate_preparation_input(self) -> None:
        if type(self.preparation_input) is not SelectiveDependenceResponsePreparationInput:
            raise ValueError("native development requires explicit full numerical preparation inputs before solver contact")
        if isinstance(self.design, CanteraReactionResponseCanteraDesign):
            from .cantera_reaction_response.preparation_inputs import DRAW_VARIABLE_IDS
        elif isinstance(self.design, FipyReactionDiffusionResponseFiPyDesign):
            from .fipy_reaction_diffusion_response.preparation_inputs import DRAW_VARIABLE_IDS
        else:
            raise TypeError("native development requires a supported current design")
        if self.preparation_input.roster_index != 0:
            raise ValueError("singleton native development preparation requires its exact first unit position")
        self.preparation_input.validate_context(
            complete_unit_id=self.independent_unit_id,
            target_id=self.design.target_id,
            target_design=ObjectIdentity.from_record(self.design.design_id, self.design),
            phase=SelectiveDependenceResponsePhase.CANARY.value,
            draw_variable_ids=DRAW_VARIABLE_IDS,
        )


def execute_native_complete_unit(
    request: ReactionDiffusionDevelopmentInput,
    *,
    phase: SelectiveDependenceResponsePhase,
) -> SelectiveDependenceResponseCompleteUnitResult:
    """Execute one complete native preparation in the matching pinned solver."""

    request.validate_preparation_input()
    if isinstance(request.design, CanteraReactionResponseCanteraDesign):
        import cantera

        from empirical_lawhood.adapters.simulators.cantera_reaction_response.runtime import execute_cantera_complete_unit

        if cantera.__version__ != request.design.cantera_version:
            raise ValueError("installed Cantera version differs from the design")
        result = execute_cantera_complete_unit(
            request.design,
            complete_unit_id=request.independent_unit_id,
            phase=phase,
            preparation_input=request.preparation_input,
        )
    elif isinstance(request.design, FipyReactionDiffusionResponseFiPyDesign):
        import fipy

        from empirical_lawhood.adapters.simulators.fipy_reaction_diffusion_response.runtime import execute_fipy_complete_unit

        if fipy.__version__ != request.design.fipy_version:
            raise ValueError("installed FiPy version differs from the design")
        result = execute_fipy_complete_unit(
            request.design,
            complete_unit_id=request.independent_unit_id,
            phase=phase,
            preparation_input=request.preparation_input,
        )
    else:
        raise TypeError("unsupported selective dependence response native design")
    stopped = tuple(
        f"{condition.condition_id}:{condition.stop_code}"
        for condition in result.conditions
        if condition.stopped
    )
    if stopped:
        raise RuntimeError(f"native condition stopped: {stopped}")
    return result


def run_native_development_check(
    request: ReactionDiffusionDevelopmentInput,
) -> dict[str, object]:
    """Report one explicitly exposed canary without a campaign claim."""

    result = execute_native_complete_unit(request, phase=SelectiveDependenceResponsePhase.CANARY)
    family = (
        "cantera-selective-dependence-response"
        if isinstance(request.design, CanteraReactionResponseCanteraDesign)
        else "fipy-selective-dependence-response"
    )
    return {
        "config_id": request.config_id,
        "design_id": request.design.design_id,
        "family": family,
        "independent_unit_id": result.complete_unit_id,
        "independent_units": 1,
        "nested_conditions": len(result.conditions),
        "nested_conditions_count_as_units": result.nested_conditions_count_as_units,
        "native_result_sha256": result.fingerprint(),
        "source_version": result.source_version,
        "solver_id": result.solver_id,
        "conditions": [
            {
                "condition_id": condition.condition_id,
                "action": {
                    "requested": str(condition.action.requested_value),
                    "requested_unit": condition.action.requested_unit,
                    "accepted": str(condition.action.accepted_value),
                    "accepted_unit": condition.action.accepted_unit,
                    "applied": str(condition.action.applied_value),
                    "applied_unit": condition.action.applied_unit,
                    "realized": str(condition.action.realized_value),
                    "realized_unit": condition.action.realized_unit,
                    "realized_clock_seconds": str(condition.action.realized_clock),
                    "acceptance_state": condition.action.acceptance_state,
                },
                "receivers": {
                    receiver.receiver_id: {
                        "value": str(receiver.value),
                        "unit": receiver.native_unit,
                    }
                    for receiver in condition.receivers
                },
                "disposition": condition.disposition.value,
            }
            for condition in result.conditions
        ],
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }


__all__ = [
    'ReactionDiffusionDevelopmentInput',
    "execute_native_complete_unit",
    "run_native_development_check",
]

"""Accepted planning projection for the selective dependence response Cantera target."""

from __future__ import annotations

from empirical_lawhood.adapters.independent_source_exports import IndependentSourceExport
from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseInspectedPilotInventory

from decimal import Decimal

from empirical_lawhood.adapters.methods.selective_dependence_response.authoring import SelectiveDependenceResponseTargetAuthoringAct, SelectiveDependenceResponseTargetPlanningProjection, build_target_development_authoring_act, build_target_source_authoring_act
from empirical_lawhood.adapters.methods.selective_dependence_response.composition import compose_target_source_canary
from empirical_lawhood.adapters.methods.selective_dependence_response.contracts import SelectiveDependenceResponseConstructReviewAttestation
from empirical_lawhood.adapters.methods.selective_dependence_response.method_completion import SelectiveDependenceResponseMethodCompletionEnvelope
from empirical_lawhood.adapters.methods.selective_dependence_response.source_completion import SelectiveDependenceResponseSourceCanaryCompletionEnvelope
from empirical_lawhood.kernel.quantities import ResponseDirection
from empirical_lawhood.kernel.worlds import NumericalCoordinateKind, NumericalCoordinateSpec
from empirical_lawhood.planning.formal_gaps import FormalGapRegister

from .design import cantera_design
from .contracts import CanteraReactionResponseCanteraSourceQualification
from .registration import cantera_target_binding


def cantera_planning_projection(
    *,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    method_completion: SelectiveDependenceResponseMethodCompletionEnvelope,
    implementation_sha256: str,
    inspected_pilot_inventory: SelectiveDependenceResponseInspectedPilotInventory | None = None,
    inspected_pilot_export: IndependentSourceExport | None = None,
) -> SelectiveDependenceResponseTargetPlanningProjection:
    design = cantera_design()
    binding = cantera_target_binding(
        construct_review,
        method_completion=method_completion,
        inspected_pilot_inventory=inspected_pilot_inventory,
        inspected_pilot_export=inspected_pilot_export,
    )
    composition = compose_target_source_canary(
        binding,
        implementation_sha256=implementation_sha256,
    )
    return SelectiveDependenceResponseTargetPlanningProjection(
        target_slug="cantera",
        target_label="Cantera non-isothermal open CSTR",
        binding=binding,
        source_composition=composition,
        represented_physics=(
            "finite-rate-gas-kinetics",
            "nonisothermal-open-cstr",
            "prescribed-wall-heat-loss",
        ),
        unrepresented_physics=(
            "multiphase-combustion",
            "physical-reactor-fabrication-and-instrumentation",
            "spatially-resolved-turbulence",
        ),
        equations_id="cantera-ideal-gas-reactor-network-energy-on",
        closure_ids=(
            "closure.cantera-gri30-finite-rate-kinetics",
            "closure.cantera-prescribed-heat-transfer",
        ),
        boundary_condition_ids=(
            "boundary.cantera-fixed-inlet-composition",
            "boundary.cantera-pressure-controller",
        ),
        solver_id="cantera-reactor-network",
        solver_version=design.cantera_version,
        precision="float64",
        device_class="cpu",
        runtime_id="cpython-3.11-cantera-3.2.0",
        numerical_coordinates=(
            NumericalCoordinateSpec(
                coordinate_id="coordinate.cantera.maximum-solver-steps",
                kind=NumericalCoordinateKind.SOLVER_REFINEMENT,
                value=Decimal(design.maximum_solver_steps),
                unit="steps",
                refinement_level=0,
            ),
            NumericalCoordinateSpec(
                coordinate_id="coordinate.cantera.precision",
                kind=NumericalCoordinateKind.PRECISION,
                value=Decimal(64),
                unit="bits",
                refinement_level=0,
            ),
        ),
        action_stage_units=(
            "dimensionless-flow-multiplier",
            "dimensionless-flow-multiplier",
            "kilogram-per-second",
            "kilogram",
        ),
        receiver_units=(
            (
                "carbon-monoxide",
                "dimensionless-fraction",
                ResponseDirection.LOWER_IS_BETTER,
            ),
            (
                "element-error",
                "dimensionless-fraction",
                ResponseDirection.LOWER_IS_BETTER,
            ),
            (
                "methane-conversion",
                "dimensionless-fraction",
                ResponseDirection.HIGHER_IS_BETTER,
            ),
            ("peak-temperature", "kelvin", ResponseDirection.LOWER_IS_BETTER),
            ("temperature", "kelvin", ResponseDirection.TARGET_BAND),
        ),
        action_stage_bounds=(
            (min(design.action_multipliers), design.outside_action_multiplier),
            (min(design.action_multipliers), design.outside_action_multiplier),
            (Decimal(0), None),
            (Decimal(0), None),
        ),
        maximum_horizon_seconds=max(design.horizon_seconds),
    )


def build_cantera_source_authoring_act(
    *,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    method_completion: SelectiveDependenceResponseMethodCompletionEnvelope,
    implementation_sha256: str,
    register: FormalGapRegister,
    inspected_pilot_inventory: SelectiveDependenceResponseInspectedPilotInventory | None = None,
    inspected_pilot_export: IndependentSourceExport | None = None,
) -> SelectiveDependenceResponseTargetAuthoringAct:
    return build_target_source_authoring_act(
        projection=cantera_planning_projection(
            construct_review=construct_review,
            method_completion=method_completion,
            implementation_sha256=implementation_sha256,
            inspected_pilot_inventory=inspected_pilot_inventory,
            inspected_pilot_export=inspected_pilot_export,
        ),
        register=register,
    )


def build_cantera_development_authoring_act(
    *,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    method_completion: SelectiveDependenceResponseMethodCompletionEnvelope,
    source_qualification: CanteraReactionResponseCanteraSourceQualification,
    source_completion: SelectiveDependenceResponseSourceCanaryCompletionEnvelope,
    implementation_sha256: str,
    register: FormalGapRegister,
    inspected_pilot_inventory: SelectiveDependenceResponseInspectedPilotInventory | None = None,
    inspected_pilot_export: IndependentSourceExport | None = None,
) -> SelectiveDependenceResponseTargetAuthoringAct:
    return build_target_development_authoring_act(
        projection=cantera_planning_projection(
            construct_review=construct_review,
            method_completion=method_completion,
            implementation_sha256=implementation_sha256,
            inspected_pilot_inventory=inspected_pilot_inventory,
            inspected_pilot_export=inspected_pilot_export,
        ),
        source_qualification=source_qualification,
        source_completion=source_completion,
        register=register,
    )


__all__ = [
    "build_cantera_development_authoring_act",
    "build_cantera_source_authoring_act",
    "cantera_planning_projection",
]

"""Accepted planning projection for the selective dependence response FiPy target."""

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

from .design import fipy_design
from .contracts import FipyReactionDiffusionResponseFiPySourceQualification
from .registration import fipy_target_binding


def fipy_planning_projection(
    *,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    method_completion: SelectiveDependenceResponseMethodCompletionEnvelope,
    implementation_sha256: str,
    inspected_pilot_inventory: SelectiveDependenceResponseInspectedPilotInventory | None = None,
    inspected_pilot_export: IndependentSourceExport | None = None,
) -> SelectiveDependenceResponseTargetPlanningProjection:
    design = fipy_design()
    binding = fipy_target_binding(
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
        target_slug="fipy",
        target_label="FiPy signed reaction-diffusion field",
        binding=binding,
        source_composition=composition,
        represented_physics=(
            "linear-decay",
            "one-dimensional-diffusion",
            "signed-localized-source",
        ),
        unrepresented_physics=(
            "multidimensional-transport",
            "nonlinear-reaction-kinetics",
            "physical-field-actuation-and-sensing",
        ),
        equations_id=design.equation_id,
        closure_ids=("closure.fipy-linear-decay",),
        boundary_condition_ids=("boundary.fipy-frozen-zero-flux",),
        solver_id="fipy-linear-lu-solver",
        solver_version=design.fipy_version,
        precision="float64",
        device_class="cpu",
        runtime_id="cpython-3.11-fipy-4.0.3",
        numerical_coordinates=(
            NumericalCoordinateSpec(
                coordinate_id="coordinate.fipy.mesh",
                kind=NumericalCoordinateKind.SPATIAL_GRID,
                value=Decimal(design.mesh_cells),
                unit="cells",
                refinement_level=0,
            ),
            NumericalCoordinateSpec(
                coordinate_id="coordinate.fipy.precision",
                kind=NumericalCoordinateKind.PRECISION,
                value=Decimal(64),
                unit="bits",
                refinement_level=0,
            ),
            NumericalCoordinateSpec(
                coordinate_id="coordinate.fipy.timestep",
                kind=NumericalCoordinateKind.TIMESTEP,
                value=design.timestep_seconds,
                unit="second",
                refinement_level=0,
            ),
        ),
        action_stage_units=(
            "field-amplitude-per-second",
            "field-amplitude-per-second",
            "field-amplitude-per-second",
            "field-mass",
        ),
        receiver_units=(
            ("balance-error", "field-mass", ResponseDirection.LOWER_IS_BETTER),
            ("boundary-flux", "field-flux", ResponseDirection.LOWER_IS_BETTER),
            ("downstream-mean", "field-amplitude", ResponseDirection.TARGET_BAND),
            ("field-mass", "field-mass", ResponseDirection.SIGNED_VECTOR),
            ("field-maximum", "field-amplitude", ResponseDirection.LOWER_IS_BETTER),
            ("field-minimum", "field-amplitude", ResponseDirection.HIGHER_IS_BETTER),
        ),
        action_stage_bounds=(
            (design.outside_action_rate, max(design.action_rates)),
            (design.outside_action_rate, max(design.action_rates)),
            (design.outside_action_rate, max(design.action_rates)),
            (Decimal("-1"), Decimal("1")),
        ),
        maximum_horizon_seconds=max(design.horizon_seconds),
    )


def build_fipy_source_authoring_act(
    *,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    method_completion: SelectiveDependenceResponseMethodCompletionEnvelope,
    implementation_sha256: str,
    register: FormalGapRegister,
    inspected_pilot_inventory: SelectiveDependenceResponseInspectedPilotInventory | None = None,
    inspected_pilot_export: IndependentSourceExport | None = None,
) -> SelectiveDependenceResponseTargetAuthoringAct:
    return build_target_source_authoring_act(
        projection=fipy_planning_projection(
            construct_review=construct_review,
            method_completion=method_completion,
            implementation_sha256=implementation_sha256,
            inspected_pilot_inventory=inspected_pilot_inventory,
            inspected_pilot_export=inspected_pilot_export,
        ),
        register=register,
    )


def build_fipy_development_authoring_act(
    *,
    construct_review: SelectiveDependenceResponseConstructReviewAttestation,
    method_completion: SelectiveDependenceResponseMethodCompletionEnvelope,
    source_qualification: FipyReactionDiffusionResponseFiPySourceQualification,
    source_completion: SelectiveDependenceResponseSourceCanaryCompletionEnvelope,
    implementation_sha256: str,
    register: FormalGapRegister,
    inspected_pilot_inventory: SelectiveDependenceResponseInspectedPilotInventory | None = None,
    inspected_pilot_export: IndependentSourceExport | None = None,
) -> SelectiveDependenceResponseTargetAuthoringAct:
    return build_target_development_authoring_act(
        projection=fipy_planning_projection(
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
    "build_fipy_development_authoring_act",
    "build_fipy_source_authoring_act",
    "fipy_planning_projection",
]

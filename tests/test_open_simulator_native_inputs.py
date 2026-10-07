"""Installed PyBaMM and direct TORAX input constructors retain their native contracts."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from importlib.metadata import version

import pytest

from empirical_lawhood.adapters.simulators.pybamm_electrothermal_hierarchy import BATTERY_ELECTROTHERMAL_SOURCE_VERSION, _runtime_objects, denominators, qualification_preparations
from empirical_lawhood.adapters.simulators.torax_native.contracts import NativeToraxActionStage, NativeToraxAction, NativeToraxPreparation, NativeToraxView
from empirical_lawhood.adapters.simulators.torax_native.runtime import build_native_torax_config
from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage


@pytest.mark.native('pybamm')
def test_pybamm_native_model_and_inputs_match_frozen_version() -> None:
    pybamm = pytest.importorskip("pybamm")
    assert pybamm.__version__ == BATTERY_ELECTROTHERMAL_SOURCE_VERSION == "26.6.2.0"
    preparation = qualification_preparations()[0]
    denominator = next(
        item for item in denominators(include_x_full=False)
        if item.denominator_id == "denominator.pybamm.spme.isothermal.casadi-coarse"
    )
    native, simulation = _runtime_objects(preparation, denominator)
    assert native is pybamm
    assert simulation.model.rhs
    assert isinstance(
        simulation.parameter_values["Current function [A]"], pybamm.InputParameter
    )
    assert isinstance(
        simulation.parameter_values["Ambient temperature [K]"], pybamm.InputParameter
    )


@pytest.mark.native('torax')
def test_direct_torax_builds_native_config_and_rejects_horizon_drift() -> None:
    pytest.importorskip("torax")
    assert version("torax") == "1.4.2"
    d = Decimal
    preparation = NativeToraxPreparation(
        preparation_id="preparation.public-native-canary",
        radial_coordinates=(d("0"), d("0.5"), d("1")),
        electron_temperature_ev=(d("1000"), d("800"), d("600")),
        ion_temperature_ev=(d("900"), d("750"), d("550")),
        electron_density_m3=(d("1e19"), d("9e18"), d("8e18")),
        plasma_current_a=d("8e5"),
        main_ion="deuterium",
        impurity="carbon",
        zeff=d("1.5"),
        major_radius_m=d("0.85"),
        minor_radius_m=d("0.65"),
        toroidal_field_t=d("0.55"),
        elongation_lcfs=d("1.6"),
        source_radial_location=d("0.3"),
        source_width=d("0.2"),
        electron_heat_fraction=d("0.6"),
        absorbed_power_fraction=d("0.8"),
        chi_i_m2_s=d("1"),
        chi_e_m2_s=d("2"),
        particle_diffusivity_m2_s=d("0.5"),
        particle_convection_m_s=d("0"),
        maximum_core_temperature_ev=d("3000"),
        assumption_ids=("public-synthetic-input",),
    )
    view = NativeToraxView(
        view_id="view.public-native-canary",
        radial_cells=8,
        timestep_s=d("0.001"),
        horizon_s=d("0.001"),
        solver_id="linear-theta-fully-implicit",
        linear_solver_id="thomas",
        precision="float64",
        closure_id='constant-transport',
    )
    action = NativeToraxAction(
        action_id="action.public-native-canary",
        action_label="hold",
        stages=tuple(
            NativeToraxActionStage(stage, d("1e6"), d("0"))
            for stage in ActionDeliveryStage
        ),
        duration_s=view.horizon_s,
        source_model_id="generic-ion-electron-gaussian-proxy",
    )
    config = build_native_torax_config(preparation, action, view)
    assert type(config).__module__.startswith("torax.")
    assert config.numerics.t_final == float(view.horizon_s)
    with pytest.raises(ValueError, match="horizon"):
        build_native_torax_config(
            preparation, replace(action, duration_s=d("0.002")), view
        )

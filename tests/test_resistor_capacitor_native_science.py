"""RC voltage, current and time semantics against an analytic circuit."""

from __future__ import annotations

import math
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from scipy.integrate import simpson

from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderActionSegment, ResistorCapacitorLadderModelConfig
from empirical_lawhood.adapters.simulators.rc_ladder_response.matrix_exponential import solve_matrix_exponential
from empirical_lawhood.adapters.simulators.rc_ladder_response.native_quickstart import check_native_model
from empirical_lawhood.adapters.simulators.rc_ladder_response.refinement import solve_backward_euler
from empirical_lawhood.kernel.decoding import decode_canonical_bytes


def _scalar_circuit() -> ResistorCapacitorLadderModelConfig:
    return ResistorCapacitorLadderModelConfig(
        config_id='rc-analytic-one-cell',
        scale_cells=1,
        capacitances_farads=(Decimal(1),),
        interior_resistances_ohms=(),
        left_source_resistance_ohms=Decimal(1),
        right_termination_resistance_ohms=Decimal(1),
        initial_voltages_volts=(Decimal(0),),
        action_segments=(
            ResistorCapacitorLadderActionSegment(
                "one-volt-step", Decimal(0), Decimal(1), Decimal(1), Decimal(0)
            ),
        ),
        output_times_seconds=(Decimal(0), Decimal("0.5"), Decimal(1)),
        component_metrology_frozen_before_response=True,
    )


def test_one_cell_solution_matches_closed_form_voltage_and_charge_balance() -> None:
    # KCL: C dV/dt = (V_left-V)/R_left - V/R_right = 1 - 2V.
    # Thus V(t) = (1-exp(-2t))/2 for the frozen 1 F, 1 ohm circuit.
    model = _scalar_circuit()
    exact = solve_matrix_exponential(model)
    euler = solve_backward_euler(model, maximum_step_seconds=0.001)
    for index, t in enumerate((0.0, 0.5, 1.0)):
        expected = (1.0 - math.exp(-2.0 * t)) / 2.0
        assert exact.voltages_volts[index, 0] == pytest.approx(expected, abs=1e-12)
        assert euler.voltages_volts[index, 0] == pytest.approx(expected, abs=2e-4)
    midpoint = exact.voltages_volts[1, 0]
    left_current = (1.0 - midpoint) / 1.0
    right_current = midpoint / 1.0
    charge_rate = math.exp(-1.0)  # C dV/dt at t=0.5 s, in amperes.
    assert left_current - right_current == pytest.approx(charge_rate, abs=1e-12)
    assert exact.times_seconds.tolist() == [0.0, 0.5, 1.0]


def test_quickstart_model_has_four_cells_and_frozen_native_units() -> None:
    path = Path("experiments/rc-ladder-response/model.json")
    model = decode_canonical_bytes(
        path.read_bytes(), ResistorCapacitorLadderModelConfig, maximum_bytes=64 * 1024
    )
    assert model.scale_cells == 4
    assert model.capacitances_farads == (Decimal("0.001"),) * 4
    assert model.left_source_resistance_ohms == Decimal(50)
    assert model.output_times_seconds[-1] == Decimal(1)
    exact = solve_matrix_exponential(model)
    assert 0 < exact.voltages_volts[-1, -1] < exact.voltages_volts[-1, 0] < 1
    assert exact.voltages_volts[-1].tolist() == sorted(exact.voltages_volts[-1], reverse=True)

    # A series circuit with the declared resistances has a 1/450 A
    # steady current. At 1 s the capacitors are close to, but not at,
    # this independent Ohm-law reference. Omitting the 50 ohm source
    # impedance gives a decisive, different endpoint.
    steady_current = 1.0 / (50.0 + 3.0 * 100.0 + 100.0)
    steady_nodes = tuple(
        1.0 - steady_current * (50.0 + index * 100.0) for index in range(4)
    )
    assert max(abs(a - b) for a, b in zip(exact.voltages_volts[-1], steady_nodes)) < 0.007
    omitted = solve_matrix_exponential(model, omit_source_impedance=True)
    assert abs(omitted.voltages_volts[-1, 0] - exact.voltages_volts[-1, 0]) > 0.1


def test_shipped_model_boundary_current_integrates_to_stored_charge() -> None:
    path = Path("experiments/rc-ladder-response/model.json")
    model = decode_canonical_bytes(
        path.read_bytes(), ResistorCapacitorLadderModelConfig, maximum_bytes=64 * 1024
    )
    report = check_native_model(path)
    assert report["receiver_unit"] == "V"
    assert report["boundary_current_unit"] == "A"
    assert report["campaign_candidate_compiled"] is False

    refined = replace(
        model,
        output_times_seconds=tuple(Decimal(index) / 1000 for index in range(1001)),
    )
    trajectory = solve_matrix_exponential(refined)
    # The step switches off at exactly 1 s; use the last pre-switch sample.
    # Interior branch currents cancel in global Kirchhoff current balance.
    stored_charge_c = sum(
        float(capacitance) * (end - start)
        for capacitance, start, end in zip(
            refined.capacitances_farads,
            trajectory.voltages_volts[0],
            trajectory.voltages_volts[-2],
        )
    )
    delivered_charge_c = simpson(
        trajectory.left_current_amperes[:-1] + trajectory.right_current_amperes[:-1],
        x=trajectory.times_seconds[:-1],
    )
    assert stored_charge_c == pytest.approx(delivered_charge_c, abs=1e-8)


def test_unfrozen_metrology_or_invalid_capacitance_refuses() -> None:
    model = _scalar_circuit()
    with pytest.raises(ValueError, match="pre-response metrology"):
        replace(model, component_metrology_frozen_before_response=False)
    with pytest.raises(ValueError, match="at least 0"):
        replace(model, capacitances_farads=(Decimal(-1),))

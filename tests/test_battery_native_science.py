# SPDX-License-Identifier: MPL-2.0
"""Bounded native battery response, restart, and donor-support counterexamples."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest

from empirical_lawhood.adapters.simulators import pybamm_electrothermal_hierarchy as battery_electrothermal
from empirical_lawhood.adapters.simulators import pybamm_exact_restart as battery_restart
from empirical_lawhood.adapters.simulators import pybamm_reduced_coordinate_sufficiency as battery_reduced_observation
from empirical_lawhood.kernel.evidence import OutcomeAccess

pytestmark = pytest.mark.native("pybamm")


def _spme_isothermal_casadi() -> battery_electrothermal.BatteryElectrothermalDenominator:
    return next(
        item
        for item in battery_electrothermal.denominators(include_x_full=False)
        if item.denominator_id == "denominator.pybamm.spme.isothermal.casadi-coarse"
    )


def test_electrothermal_native_current_changes_capacity_and_voltage_after_action() -> None:
    unit = battery_electrothermal.qualification_preparations()[0]
    denominator = _spme_isothermal_casadi()
    words = {item.word_id: item for item in battery_electrothermal.action_words()}
    _, hold, _, _ = battery_electrothermal._step_schedule(
        unit=unit,
        denominator=denominator,
        word=words["word.battery-electrothermal.hold"],
        stop_s=150,
    )
    _, driven, _, _ = battery_electrothermal._step_schedule(
        unit=unit,
        denominator=denominator,
        word=words["word.battery-electrothermal.i-early-150"],
        stop_s=150,
    )
    _, future, _, _ = battery_electrothermal._step_schedule(
        unit=unit,
        denominator=denominator,
        word=words["word.battery-electrothermal.future-i-plus"],
        stop_s=150,
    )
    assert all(simulation.solution.termination == "final time" for simulation in (hold, driven, future))
    at_end = 150.0
    capacity = lambda simulation: float(simulation.solution["Discharge capacity [A.h]"](at_end))
    voltage = lambda simulation: float(simulation.solution["Terminal voltage [V]"](at_end))
    current = lambda simulation: float(simulation.solution["Current [A]"](75.0))
    assert capacity(hold) == pytest.approx(0.0, abs=1e-10)
    assert capacity(driven) == pytest.approx(150.0 / 3600.0, abs=1e-8)
    assert current(driven) == pytest.approx(1.0, abs=1e-10)
    assert voltage(driven) < voltage(hold) - 0.01
    # A word whose only current begins at 600 s cannot alter a 150 s receiver.
    assert capacity(future) == pytest.approx(capacity(hold), abs=1e-10)
    assert voltage(future) == pytest.approx(voltage(hold), abs=1e-7)


def test_state_restart_lossless_state_and_source_input_reconstruct_native_hold() -> None:
    unit = battery_restart.build_design().localization_units[0]
    denominator = _spme_isothermal_casadi()
    history = next(
        item for item in battery_restart.histories() if item.word.word_id == "word.battery-electrothermal.i-plus-600"
    )
    pybamm, source, prefix, _ = battery_restart._step_prefix(
        unit=unit, denominator=denominator, history=history
    )
    assert prefix.termination == "final time"
    assert prefix.t[-1] == history.checkpoint_s
    state_hex = battery_restart._hex_values(np.asarray(prefix.all_ys[0], dtype=np.float64).reshape(-1))
    assert battery_restart._hex_values(battery_restart.decode_hex(state_hex)) == state_hex
    source_inputs = {
        name: float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
        for name, value in prefix.all_inputs[-1].items()
    }
    assert source_inputs["Current function [A]"] == pytest.approx(1.0)
    hold_inputs = battery_restart._inputs(unit, Decimal(0), Decimal(0))
    clock = np.asarray([300.0, 450.0, 600.0])
    native = source.step(
        300, t_eval=clock - 300, starting_solution=prefix, inputs=hold_inputs
    )
    _, rebuilt = battery_restart._runtime_objects(unit, denominator)
    rebuilt.build(initial_soc=float(unit.initial_soc), inputs=hold_inputs)
    start = battery_restart._starting_solution(
        pybamm,
        checkpoint_s=history.checkpoint_s,
        state=battery_restart.decode_hex(state_hex),
        model=rebuilt.built_model,
        inputs=source_inputs,
    )
    reconstructed = rebuilt.step(
        300, t_eval=clock - 300, starting_solution=start, inputs=hold_inputs
    )
    for name, tolerance in (
        ("Discharge capacity [A.h]", 1e-10),
        ("Volume-averaged cell temperature [K]", 1e-8),
        ("Terminal voltage [V]", 1e-8),
    ):
        native_values = np.asarray(native[name](clock), dtype=np.float64)
        rebuilt_values = np.asarray(reconstructed[name](clock), dtype=np.float64)
        assert np.max(np.abs(native_values - rebuilt_values)) <= tolerance
    assert reconstructed.termination == "final time"


def test_reduced_observation_native_restart_and_distinct_target_refuse_unsupported_donor() -> None:
    design = battery_reduced_observation.build_design()
    denominator = _spme_isothermal_casadi()
    history = next(
        item for item in design.histories if item.word_id == "word.battery-electrothermal.i-plus-600"
    )
    captures = []
    for unit in design.development_units[:3]:
        capture, restart = battery_reduced_observation.acquire_cell(
            design=design,
            unit_id=unit.unit_id,
            denominator_id=denominator.denominator_id,
            history_id=history.history_id,
            source_identity_sha256="0" * 64,
            implementation_sha256="1" * 64,
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
        assert capture["value"]["disposition"] == "COMPLETE"
        assert restart["value"]["disposition"] == "COMPLETE"
        assert restart["value"]["receiver_closed"] is True
        assert restart["value"]["state_bit_exact"] is True
        assert capture["value"]["checkpoint_s"] == 300
        assert capture["value"]["end_s"] == 600
        captures.append(capture)

    chart = design.charts[0]
    scale, predictions = battery_reduced_observation.build_prediction_rows(
        development_captures=captures[:2],
        target_captures=captures[2:],
        chart=chart,
        leave_one_out=False,
    )
    assert len(scale) == len(chart.coordinate_ids)
    assert len(predictions) == 1
    row = predictions[0]
    assert row.target_unit_id != row.donor_unit_id
    assert row.distance is not None and row.distance > battery_reduced_observation.BATTERY_REDUCED_OBSERVATION_SUPPORT_RADIUS
    assert row.inside_support is False
    assert row.adequate is None
    assert row.reason_codes == ("OUTSIDE_FROZEN_LOCAL_SUPPORT",)

# SPDX-License-Identifier: MPL-2.0
"""Shipped battery input reaches all three distinct native scientific contracts."""

from __future__ import annotations

import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from empirical_lawhood.adapters.simulators import pybamm_reduced_coordinate_sufficiency as battery_reduced_observation
from empirical_lawhood.adapters.simulators.pybamm_development_input import PyBaMMDevelopmentInput, run_native_development_check
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

CONFIG = (
    Path(__file__).resolve().parents[1] / "experiments/battery-response-and-restart/config.json"
)


def _config() -> PyBaMMDevelopmentInput:
    return decode_canonical_bytes(
        CONFIG.read_bytes(), PyBaMMDevelopmentInput, maximum_bytes=16 * 1024
    )


@pytest.mark.native('pybamm')
def test_shipped_input_reaches_electrothermal_response_state_restart_restart_and_reduced_observation_refusal() -> None:
    pytest.importorskip("pybamm")
    report = run_native_development_check(_config())
    assert report["source_version"] == "26.6.2.0"
    assert report["independent_preparations"] == 5
    assert report["nested_electrothermal_words"] == 3
    assert report["nested_restart_routes"] == 2
    battery_electrothermal = report["electrothermal_response"]
    assert battery_electrothermal["driven_capacity_ah"] == pytest.approx(1 * 150 / 3600, abs=1e-8)
    assert battery_electrothermal["future_capacity_ah"] == pytest.approx(
        battery_electrothermal["hold_capacity_ah"], abs=1e-10
    )
    assert battery_electrothermal["driven_voltage_v"] < battery_electrothermal["hold_voltage_v"]
    battery_restart = report["exact_restart"]
    assert battery_restart["source_current_a"] == pytest.approx(1)
    assert battery_restart["checkpoint_s"] == 300 and battery_restart["continuation_end_s"] == 600
    assert max(battery_restart["maximum_restart_defects"].values()) <= 1e-8
    battery_reduced_observation = report["reduced_observation"]
    assert battery_reduced_observation["checkpoint_s"] == 300 and battery_reduced_observation["continuation_end_s"] == 600
    assert battery_reduced_observation["target_unit_id"] != battery_reduced_observation["donor_unit_id"]
    assert battery_reduced_observation["inside_support"] is False
    assert battery_reduced_observation["reason_codes"] == ["OUTSIDE_FROZEN_LOCAL_SUPPORT"]
    assert report["campaign_candidate_compiled"] is False


def test_starter_refuses_wrong_stratum_and_wrong_native_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="strata"):
        replace(
            _config(),
            reduced_observation_initial_socs=(Decimal("0.3"), Decimal("0.35"), Decimal("0.5")),
        )
    monkeypatch.setitem(sys.modules, "pybamm", SimpleNamespace(__version__="wrong"))
    with pytest.raises(ValueError, match="version"):
        run_native_development_check(_config())


def test_reduced_observation_input_refuses_wrong_identity_before_native_contact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.setattr(battery_reduced_observation, "_step_prefix", lambda *_args: calls.append(True))
    unit = battery_reduced_observation.BatteryReducedObservationPreparation(
        "unit.empirical-lawhood-pybamm-rejection-001",
        battery_reduced_observation.BatteryReducedObservationStage.DEVELOPMENT,
        'low-state-of-charge-cool',
        Decimal("0.30"),
        Decimal("291.5"),
        False,
    )
    denominator = next(
        item
        for item in battery_reduced_observation.denominators()
        if item.denominator_id == _config().denominator_id
    )
    history = next(
        item for item in battery_reduced_observation.histories() if item.history_id == _config().reduced_observation_history_id
    )
    with pytest.raises(ValueError, match="SHA-256|sha256"):
        battery_reduced_observation.acquire_preparation(
            unit=unit,
            denominator=denominator,
            history=history,
            source_identity_sha256="invalid",
            implementation_sha256="a" * 64,
            outcome_access=battery_reduced_observation.OutcomeAccess.DEVELOPMENT_VISIBLE,
        )
    assert calls == []

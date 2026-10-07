"""Direct TORAX native energy, temperature and refusal contract."""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

import pytest
from typer.testing import CliRunner

from empirical_lawhood.adapters.simulators.torax_native.native_quickstart import NativeToraxQuickstart
from empirical_lawhood.adapters.simulators.torax_native.runtime import build_native_torax_config
from empirical_lawhood.adapters.simulators.torax_native import runtime
from empirical_lawhood.adapters.simulators.torax_native.contracts import NativeToraxEpisodeDisposition
from empirical_lawhood.cli.app import app
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/tokamak-heat-response/config.json"


@pytest.mark.native('torax')
def test_shipped_torax_pair_runs_native_energy_and_response() -> None:
    pytest.importorskip("torax")
    config = decode_canonical_bytes(
        CONFIG.read_bytes(), NativeToraxQuickstart, maximum_bytes=32 * 1024
    )
    assert config.view.horizon_s == Decimal("0.04")
    for action in (config.low_action, config.high_action):
        native = build_native_torax_config(config.preparation, action, config.view)
        assert native.sources.generic_heat.P_total.value[0] == float(
            action.realized_power_w
        )
    result = CliRunner().invoke(
        app,
        ["campaign", "torax-native-check", "--config", str(CONFIG)],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["native_torax_version"] == "1.4.2"
    assert (report["independent_units"], report["nested_action_views"]) == (1, 2)
    assert report["native_time_points"] == [41, 41]
    assert report["receiver_unit"] == "eV"
    for label in ("low", "high"):
        action = report[label]
        assert Decimal(action["effort_j"]) == (
            Decimal(action["power_w"]) * Decimal(report["horizon_s"])
        )
        assert set(action["stages"]) == {"REQUESTED", "ACCEPTED", "APPLIED", "REALIZED"}
        assert {
            Decimal(stage["coordinate_s"]) for stage in action["stages"].values()
        } == {0}
    assert Decimal(report["high"]["delta_core_temperature_ev"]) > Decimal(
        report["low"]["delta_core_temperature_ev"]
    )
    assert report["campaign_candidate_compiled"] is False


def test_torax_action_clock_and_resource_refuse_before_native() -> None:
    config = decode_canonical_bytes(
        CONFIG.read_bytes(), NativeToraxQuickstart, maximum_bytes=32 * 1024
    )
    with pytest.raises(ValueError, match="clock differs"):
        replace(
            config,
            low_action=replace(
                config.low_action,
                duration_s=Decimal("0.02"),
            ),
        )
    with pytest.raises(ValueError, match="resource bound"):
        replace(config, view=replace(config.view, radial_cells=25))


@pytest.mark.parametrize(
    "temperatures,electron_edge,solver_error,times,disposition,reasons,margin",
    [
        ((1000, 6000, 1500), 600, "NO_ERROR", (0, .02, .04), "NUMERICAL_INVALID", ("CORE_TEMPERATURE_SAFETY_CEILING_EXCEEDED",), None),
        ((4000, 2000, 1500), 600, "NO_ERROR", (0, .02, .04), "NUMERICAL_INVALID", ("CORE_TEMPERATURE_SAFETY_CEILING_EXCEEDED",), None),
        ((1000, 2000, 4000), 600, "NO_ERROR", (0, .02, .04), "NUMERICAL_INVALID", ("CORE_TEMPERATURE_SAFETY_CEILING_EXCEEDED",), None),
        ((1000, 3000, 1500), 600, "NO_ERROR", (0, .02, .04), "COMPLETE", (), Decimal(0)),
        ((1000, 2000, 1500), 600, "NO_ERROR", (0, .02, .04), "COMPLETE", (), Decimal(1000)),
        ((1000, 2000, 1500), 0, "NO_ERROR", (0, .02, .04), "NUMERICAL_INVALID", ("NONPOSITIVE_ELECTRON_TEMPERATURE",), None),
        ((1000, 6000, 1500), 600, "REVIEW_SOLVER_FAILURE", (0, .02, .04), "SOLVER_FAILURE", ("CORE_TEMPERATURE_SAFETY_CEILING_EXCEEDED", "TORAX_SIM_ERROR"), None),
        ((1000, 2000, 1500), 600, "NO_ERROR", (0, .02, .03), "NUMERICAL_INVALID", ("TORAX_CLOCK_OR_HORIZON_INVALID",), None),
    ],
)
def test_retained_temperature_peak_and_distinct_failures(
    monkeypatch, temperatures, electron_edge, solver_error, times, disposition, reasons, margin,
):
    """Injected native histories independently falsify an endpoint-only ceiling."""
    config = decode_canonical_bytes(CONFIG.read_bytes(), NativeToraxQuickstart, maximum_bytes=32 * 1024)
    profiles = tuple(SimpleNamespace(
        T_e=SimpleNamespace(value=np.asarray((core, electron_edge), dtype=np.float64) / 1000),
        T_i=SimpleNamespace(value=np.asarray((.9, .55), dtype=np.float64)),
    ) for core in temperatures)
    history = SimpleNamespace(times=np.asarray(times), core_profiles=profiles, sim_error=SimpleNamespace(name=solver_error))
    monkeypatch.setitem(sys.modules, "torax", SimpleNamespace(run_simulation=lambda *_a, **_k: (None, history)))
    monkeypatch.setattr(runtime, "build_native_torax_config", lambda *_: None)
    trajectory, episode = runtime.execute_native_torax(config.preparation, config.low_action, config.view)
    assert trajectory.core_electron_temperature_ev == tuple(Decimal(value) for value in temperatures)
    assert episode.disposition is NativeToraxEpisodeDisposition(disposition)
    assert episode.reason_codes == reasons and episode.safety_margin_ev == margin
    if margin is None:
        assert episode.endpoint_delta_core_temperature_ev is None and episode.trajectory_sha256 is None
    else:
        assert episode.endpoint_delta_core_temperature_ev == Decimal(temperatures[-1] - temperatures[0])

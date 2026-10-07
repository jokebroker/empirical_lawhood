"""Pinned Brian2 LIF action and receiver against the analytic threshold clock."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from empirical_lawhood.adapters.simulators.brian2_neuron_current_response.contracts import Brian2LIFNativeConfig
from empirical_lawhood.adapters.simulators.brian2_neuron_current_response.native_quickstart import run_native_development_check
from empirical_lawhood.cli.app import app
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/neuron-current-response/config.json"
NATIVE_PYTHON = Path(os.environ.get(
    "EMPIRICAL_LAWHOOD_BRIAN2_PYTHON",
    str(ROOT / "experiments/neuron-current-response/native-env/.venv/bin/python"),
))


@pytest.mark.native()
def test_pinned_native_lif_matches_independent_spike_clock_and_delivery() -> None:
    if not NATIVE_PYTHON.is_file():
        pytest.skip("install the pinned Brian2 native-env profile")
    config = decode_canonical_bytes(
        CONFIG.read_bytes(), Brian2LIFNativeConfig, maximum_bytes=16 * 1024
    )
    result = CliRunner().invoke(
        app,
        [
            "campaign", "brian2-native-check", "--config", str(CONFIG),
            "--native-python", str(NATIVE_PYTHON),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    hold, step = report["native"]["arms"]
    assert report["independent_units"] == 1
    assert report["independent_unit_id"] == config.independent_unit_id
    assert not report["campaign_candidate_compiled"] and not report["campaign_issued"]
    assert hold["spike_count"] == 0
    assert hold["post_step_mean_mv"] == pytest.approx(float(config.rest_mv), abs=1e-12)
    assert hold["realized_pa"] == 0
    assert step["requested_pa"] == float(config.requested_step_pa)
    assert step["accepted_pa"] == step["applied_pa"] == step["realized_pa"] == float(
        config.maximum_accepted_pa
    )
    assert step["receiver_horizon_ms"] == float(config.horizon_ms)
    # A reset LIF neuron under a constant accepted current crosses threshold
    # after tau*ln(drive/(drive-threshold_delta)). This reference does not use
    # Brian2's Euler state update or spike monitor.
    drive_mv = float(config.membrane_resistance_mohm * config.maximum_accepted_pa) / 1000
    delta_mv = float(config.threshold_mv - config.rest_mv)
    interval_ms = float(config.membrane_tau_ms) * math.log(
        drive_mv / (drive_mv - delta_mv)
    )
    assert step["spike_count"] == math.floor(float(config.horizon_ms) / interval_ms)
    assert float(config.rest_mv) < step["post_step_mean_mv"] < float(config.threshold_mv)


def test_invalid_clock_refuses_before_native_worker(tmp_path: Path) -> None:
    config = decode_canonical_bytes(
        CONFIG.read_bytes(), Brian2LIFNativeConfig, maximum_bytes=16 * 1024
    )
    with pytest.raises(ValueError, match="grid"):
        replace(config, timestep_ms=Decimal("0.3"))
    malformed = tmp_path / "not-canonical.json"
    malformed.write_bytes(CONFIG.read_bytes() + b"\n")
    result = CliRunner().invoke(
        app,
        [
            "campaign", "brian2-native-check", "--config", str(malformed),
            "--native-python", sys.executable,
        ],
    )
    assert result.exit_code == 3
    assert "not canonical" in result.output


def test_spoofed_realized_current_refuses_before_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = decode_canonical_bytes(
        CONFIG.read_bytes(), Brian2LIFNativeConfig, maximum_bytes=16 * 1024
    )
    payload = {
        "native_runtime": "cpython-3.11.14-brian2-2.9.0-numpy-1.26.4",
        "arms": [
            {
                "arm": name,
                "requested_pa": requested,
                "accepted_pa": accepted,
                "applied_pa": accepted,
                "realized_pa": realized,
                "receiver_horizon_ms": float(config.horizon_ms),
                "post_step_mean_mv": float(config.rest_mv),
                "spike_count": 0,
            }
            for name, requested, accepted, realized in (
                ("hold", 0.0, 0.0, 0.0),
                (
                    "step",
                    float(config.requested_step_pa),
                    float(config.maximum_accepted_pa),
                    float(config.maximum_accepted_pa) + 1.0,
                ),
            )
        ],
    }
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=(), returncode=0, stdout=json.dumps(payload), stderr=""
        ),
    )
    with pytest.raises(ValueError, match="step realized_pa"):
        run_native_development_check(config, native_python=Path(sys.executable))

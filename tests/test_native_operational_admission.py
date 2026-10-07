"""Editable native clocks refuse excessive work before native imports/launches."""

from dataclasses import replace
from decimal import Decimal
import builtins
import json
from pathlib import Path
import subprocess
import sys

import pytest
from typer.testing import CliRunner

from empirical_lawhood.adapters.simulators._native_admission import (
    MAX_NATIVE_UPDATES_PER_ARM,
    TORAX_ESTIMATED_BYTES_PER_RADIAL_CELL,
    TORAX_ESTIMATED_BYTES_PER_RETAINED_SAMPLE,
    TORAX_MAX_ESTIMATED_HISTORY_BYTES,
    admit_native_time_grid,
)
from empirical_lawhood.adapters.simulators.brian2_neuron_current_response.contracts import Brian2LIFNativeConfig
from empirical_lawhood.adapters.simulators.brian2_neuron_current_response import native_quickstart as brian
from empirical_lawhood.adapters.simulators.torax_native.native_quickstart import NativeToraxQuickstart
from empirical_lawhood.adapters.simulators.torax_native import runtime as torax
from empirical_lawhood.cli.app import app
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

ROOT = Path(__file__).resolve().parents[1]
TORAX_INPUT = ROOT / "experiments/tokamak-heat-response/config.json"
BRIAN_INPUT = ROOT / "experiments/neuron-current-response/config.json"


@pytest.fixture
def torax_config():
    return decode_canonical_bytes(TORAX_INPUT.read_bytes(), NativeToraxQuickstart, maximum_bytes=32 * 1024)


@pytest.fixture
def brian_config():
    return decode_canonical_bytes(BRIAN_INPUT.read_bytes(), Brian2LIFNativeConfig, maximum_bytes=16 * 1024)


def test_shipped_clocks_and_canonical_bytes_are_unchanged(torax_config, brian_config):
    assert torax_config.canonical_bytes() == TORAX_INPUT.read_bytes()
    assert brian_config.canonical_bytes() == BRIAN_INPUT.read_bytes()
    assert (torax_config.view.operational_admission().update_count,
            torax_config.view.operational_admission().sample_count) == (40, 41)
    assert brian_config.operational_admission().interval_updates == (200, 800)
    assert brian_config.operational_admission().sample_count == 1000


@pytest.mark.parametrize("timestep", ["1e-12", "1e-10000", "0"])
def test_torax_excessive_or_unrepresentable_clocks_refuse(torax_config, timestep):
    with pytest.raises(ValueError, match="updates|float64|positive"):
        replace(torax_config.view, timestep_s=Decimal(timestep))


@pytest.mark.parametrize("timestep", ["1e-9", "1e-10000", "0", "0.3"])
def test_brian_excessive_or_invalid_clocks_refuse(brian_config, timestep):
    with pytest.raises(ValueError, match="updates|float64|positive|integral"):
        replace(brian_config, timestep_ms=Decimal(timestep))


def test_exact_update_limit_and_one_over(brian_config):
    admitted = replace(brian_config, timestep_ms=Decimal("0.001"))
    assert admitted.operational_admission().update_count == MAX_NATIVE_UPDATES_PER_ARM
    with pytest.raises(ValueError, match="updates"):
        replace(admitted, horizon_ms=Decimal("80.001"))


def test_torax_history_budget_largest_admitted_sample_count_and_one_over(torax_config):
    per_sample = 8 * TORAX_ESTIMATED_BYTES_PER_RADIAL_CELL + TORAX_ESTIMATED_BYTES_PER_RETAINED_SAMPLE
    samples = TORAX_MAX_ESTIMATED_HISTORY_BYTES // per_sample
    view = replace(torax_config.view, radial_cells=8, timestep_s=Decimal(1), horizon_s=Decimal(samples - 1))
    assert view.operational_admission().estimated_history_bytes == samples * per_sample
    with pytest.raises(ValueError, match="history estimate"):
        replace(view, horizon_s=Decimal(samples))


def test_partial_final_step_and_exact_byte_budget():
    kwargs = dict(timestep=Decimal(1), label="review", require_integral_intervals=False,
                  sample_offset=1, estimated_bytes_per_sample=128, maximum_estimated_bytes=384)
    admission = admit_native_time_grid(intervals=(Decimal("1.5"),), **kwargs)
    assert (admission.update_count, admission.sample_count, admission.estimated_history_bytes) == (2, 3, 384)
    with pytest.raises(ValueError, match="history estimate"):
        admit_native_time_grid(intervals=(Decimal("2.01"),), **kwargs)


@pytest.mark.parametrize("times", [("1e-10000", "1"), ("1", "1e10000"), ("1e-20", "1")])
def test_native_float_underflow_overflow_and_nonprogress_refuse(times):
    with pytest.raises(ValueError, match="float64"):
        admit_native_time_grid(timestep=Decimal(times[0]), intervals=(Decimal(times[1]),),
                               label="review", require_integral_intervals=False,
                               sample_offset=1, estimated_bytes_per_sample=128, maximum_estimated_bytes=1024)


def test_decimal_rounding_cannot_make_nonintegral_brian_intervals_valid():
    with pytest.raises(ValueError, match="integral"):
        admit_native_time_grid(timestep=Decimal(1),
                               intervals=(Decimal("1.00000000000000000000000000001"), Decimal(1)),
                               label="Brian2", require_integral_intervals=True,
                               sample_offset=0, estimated_bytes_per_sample=128, maximum_estimated_bytes=1024)


def test_direct_torax_entries_recheck_before_native_import(torax_config, monkeypatch):
    view = replace(torax_config.view)
    object.__setattr__(view, "timestep_s", Decimal("1e-12"))
    original_import = builtins.__import__
    def no_native(name, *args, **kwargs):
        if name == "torax" or name.startswith("torax."):
            pytest.fail("over-budget clock contacted TORAX")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", no_native)
    for entry in (torax.build_native_torax_config, torax.execute_native_torax):
        with pytest.raises(ValueError, match="updates"):
            entry(torax_config.preparation, torax_config.low_action, view)


def test_direct_brian_entry_rechecks_before_worker(brian_config, monkeypatch):
    config = replace(brian_config)
    object.__setattr__(config, "timestep_ms", Decimal("1e-9"))
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("over-budget clock launched Brian2 worker"))
    with pytest.raises(ValueError, match="updates"):
        brian.run_native_development_check(config, native_python=Path(sys.executable))


@pytest.mark.parametrize("substrate", ["torax", "brian2"])
def test_edited_public_commands_refuse_before_native(substrate, tmp_path, monkeypatch):
    path = TORAX_INPUT if substrate == "torax" else BRIAN_INPUT
    raw = json.loads(path.read_bytes())
    if substrate == "torax":
        raw["value"]["view"]["value"]["timestep_s"] = {"decimal": "1e-12"}
        from empirical_lawhood.adapters.simulators.torax_native import native_quickstart
        monkeypatch.setattr(native_quickstart, "execute_native_torax", lambda *_: pytest.fail("edited clock contacted native TORAX"))
        command = ["campaign", "torax-native-check"]
    else:
        raw["value"]["timestep_ms"] = {"decimal": "0.000000001"}
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("edited clock launched native worker"))
        command = ["campaign", "brian2-native-check", "--native-python", sys.executable]
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(raw, sort_keys=True, separators=(",", ":")))
    result = CliRunner().invoke(app, [*command, "--config", str(edited)])
    assert result.exit_code == 3, result.output
    assert "updates per arm" in result.output


@pytest.mark.parametrize("timestep,sample_count,reason", [("1e-9", 100000000000, "updates per arm"), ("0.1", 999, "sample count")])
def test_standalone_worker_admission_precedes_native_import(timestep, sample_count, reason):
    worker = ROOT / "src/empirical_lawhood/adapters/simulators/brian2_neuron_current_response/worker.py"
    result = subprocess.run([sys.executable, "-I", str(worker)],
                            input=json.dumps(dict(timestep_ms=timestep, baseline_ms="20", horizon_ms="80", sample_count=sample_count)),
                            text=True, capture_output=True, check=False, timeout=10)
    assert result.returncode != 0 and reason in result.stderr
    assert "No module named 'brian2'" not in result.stderr

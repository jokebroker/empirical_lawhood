"""Bounded Brian2 development check in its separately pinned native runtime."""

from __future__ import annotations

import json
import math
import subprocess
from importlib import resources
from pathlib import Path

from .contracts import Brian2LIFNativeConfig

_RUNTIME = "cpython-3.11.14-brian2-2.9.0-numpy-1.26.4"
_FIELDS = (
    "timestep_ms",
    "baseline_ms",
    "horizon_ms",
    "membrane_tau_ms",
    "membrane_resistance_mohm",
    "rest_mv",
    "reset_mv",
    "threshold_mv",
    "requested_step_pa",
    "maximum_accepted_pa",
)


def run_native_development_check(
    config: Brian2LIFNativeConfig, *, native_python: Path
) -> dict[str, object]:
    """Execute one reset block and verify the native action/receiver contract."""

    admission = config.operational_admission()
    worker = resources.files(__package__).joinpath("worker.py")
    inputs: dict[str, object] = {name: str(getattr(config, name)) for name in _FIELDS}
    inputs["sample_count"] = admission.sample_count
    process = subprocess.run(
        (str(native_python), str(worker)),
        input=json.dumps(inputs),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if process.returncode:
        raise RuntimeError(process.stderr.strip()[-1200:] or "native worker stopped")
    native = json.loads(process.stdout)
    if not isinstance(native, dict) or native.get("native_runtime") != _RUNTIME:
        raise ValueError("Brian2 worker returned another runtime contract")
    arms = native.get("arms")
    if not isinstance(arms, list) or len(arms) != 2:
        raise ValueError("Brian2 worker omitted the paired reset arms")
    for arm, name, requested, accepted in (
        (arms[0], "hold", 0.0, 0.0),
        (
            arms[1],
            "step",
            float(config.requested_step_pa),
            min(float(config.requested_step_pa), float(config.maximum_accepted_pa)),
        ),
    ):
        if not isinstance(arm, dict) or arm.get("arm") != name:
            raise ValueError("Brian2 worker changed the reset-arm identity")
        for stage, expected in (
            ("requested_pa", requested),
            ("accepted_pa", accepted),
            ("applied_pa", accepted),
            ("realized_pa", accepted),
            ("receiver_horizon_ms", float(config.horizon_ms)),
        ):
            observed = arm.get(stage)
            if type(observed) not in (float, int) or not math.isfinite(observed) or not math.isclose(observed, expected, rel_tol=0, abs_tol=1e-9):
                raise ValueError(f"Brian2 worker changed {name} {stage}")
        mean = arm.get("post_step_mean_mv")
        spikes = arm.get("spike_count")
        if (
            type(mean) not in (float, int)
            or not math.isfinite(mean)
            or type(spikes) is not int
            or spikes < 0
        ):
            raise ValueError("Brian2 worker returned invalid receiver values")
    if arms[0]["spike_count"] != 0 or not math.isclose(
        arms[0]["post_step_mean_mv"], float(config.rest_mv), rel_tol=0, abs_tol=1e-9
    ):
        raise ValueError("Brian2 hold arm changed its reset receiver")
    return {
        "config_id": config.config_id,
        "independent_unit_id": config.independent_unit_id,
        "independent_units": 1,
        "native": native,
        "campaign_candidate_compiled": False,
        "campaign_issued": False,
    }

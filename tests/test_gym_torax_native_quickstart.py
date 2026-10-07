"""Pinned native Gym--TORAX action, receiver and causal-cutoff check."""

from __future__ import annotations

import json
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from empirical_lawhood.adapters.simulators.gym_torax_native.native_quickstart import GymToraxNativeQuickstart
from empirical_lawhood.cli.app import app
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/tokamak-control-response/config.json"


@pytest.mark.native('gymtorax', 'torax')
def test_shipped_gym_torax_input_runs_native_future_cutoff() -> None:
    pytest.importorskip("gymtorax")
    request = decode_canonical_bytes(
        CONFIG.read_bytes(), GymToraxNativeQuickstart, maximum_bytes=16 * 1024
    )
    assert request.preparation.physical_independent_unit_id.startswith(
        "unit.empirical-lawhood-"
    )
    result = CliRunner().invoke(
        app,
        [
            "campaign",
            "gym-torax-native-check",
            "--config",
            str(CONFIG),
            "--source-checkout",
            str(ROOT),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(result.output)
    assert report["native_state_clocks"] == 121
    assert (report["independent_units"], report["nested_action_episodes"]) == (1, 2)
    assert report["first_future_action_request_clock"] == 111
    assert report["pre_action_receiver_clocks_equal"] is True
    assert report["post_action_q_fusion_changed"] is True
    assert report["q_fusion_native_unit"] == "1"
    action = report["future_action_111"]
    assert action["delivery_disposition"] == "COMPLETE"
    assert Decimal(action["requested_ip_a"]) == Decimal(12600000)
    assert abs(
        Decimal(action["realized_ip_a"]) - Decimal(action["applied_ip_a"])
    ) < Decimal("1e-6")
    assert report["controlled_io_operator_available"] is False
    assert report["campaign_candidate_compiled"] is False


def test_gym_torax_missing_target_checkout_and_member_refuse_before_native(
    tmp_path,
) -> None:
    request = decode_canonical_bytes(
        CONFIG.read_bytes(), GymToraxNativeQuickstart, maximum_bytes=16 * 1024
    )
    result = CliRunner().invoke(
        app,
        [
            "campaign",
            "gym-torax-native-check",
            "--config",
            str(CONFIG),
            "--source-checkout",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 3
    assert "target source checkout is absent" in result.output
    with pytest.raises(ValueError, match="checked native member"):
        replace(request, numerical_member_id='member.tokamak-control.other')

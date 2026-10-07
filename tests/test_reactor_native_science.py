"""Independent native-actuator and delayed-observer checks for the public reactor."""

from __future__ import annotations

import json
import math
from decimal import Decimal

import pytest

from empirical_lawhood.adapters.simulators.reactor_prefix_response.bridge import ReactorBridge
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorCommand
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import native_task_id
from empirical_lawhood.adapters.simulators.reactor_prefix_response.packaged_source import load_packaged_reactor_source


def test_native_clipping_rate_limit_realized_exposure_and_delayed_receiver() -> None:
    source = load_packaged_reactor_source()
    params = json.loads(source.plant_params)
    sample_s = Decimal(str(params["timing"]["sample_dt_s"]))
    initial_jacket = Decimal(str(params["initial_charge"]["tj0_k"]))
    initial_reactor = float(params["initial_charge"]["t0_k"])
    upper_jacket = Decimal(str(params["actuators"]["tj_cmd_max_k"]))
    rate = Decimal(str(params["actuators"]["tj_cmd_rate_k_s"]))
    noise = float(params["measurement"]["noise_t_k"])

    with ReactorBridge(
        plant_bytes=source.plant_source.encode(),
        params_bytes=source.plant_params.encode(),
        public_scenarios_bytes=source.public_scenarios.encode(),
        scenario_id="nominal",
        plant_dt_s=Decimal(1),
    ) as bridge:
        start = bridge.start()
        first = bridge.deliver(
            ReactorCommand("reference.first", Decimal(0), Decimal(0), Decimal(1000))
        )
        second = bridge.deliver(
            ReactorCommand("reference.second", sample_s, Decimal(0), Decimal(1000))
        )

    assert first.command.jacket_k > first.accepted_jacket_k == upper_jacket
    assert first.applied_jacket_k == initial_jacket + rate * sample_s
    assert first.applied_jacket_k < first.accepted_jacket_k
    assert second.applied_jacket_k == first.applied_jacket_k + rate * sample_s
    assert all(exposure.jacket_k == first.applied_jacket_k for exposure in first.exposures)
    assert all(exposure.feed_kg_s == 0 for exposure in (*first.exposures, *second.exposures))
    assert sum((exposure.duration_s for exposure in first.exposures), Decimal(0)) == sample_s
    assert (start.time_s, first.next_measurement.time_s, second.next_measurement.time_s) == (
        Decimal(0), sample_s, 2 * sample_s
    )
    assert second.next_measurement.dosed_kg == 0

    # With zero feed and no initial reactant A, reaction heat vanishes. For the
    # nominal scenario the first interval is a coupled linear jacket/reactor
    # ODE. This closed-form reference is independent of the RK4 implementation.
    ua = float(params["thermal"]["ua0_w_per_k"])
    rho_cp = float(params["thermal"]["rho_cp_j_per_m3_k"])
    volume = float(params["initial_charge"]["v0_m3"])
    tau = float(params["thermal"]["jacket_tau_s"])
    a = ua / (rho_cp * volume)
    t = float(sample_s)
    command = float(first.applied_jacket_k)
    jacket_initial = float(initial_jacket)
    reactor_at_10_s = (
        command
        + (initial_reactor - command) * math.exp(-a * t)
        + a * (jacket_initial - command)
        * (math.exp(-t / tau) - math.exp(-a * t)) / (a - 1 / tau)
    )
    # The 10 s callback observes the initial state; the 20 s callback sees
    # the 10 s state. Noise is bounded here at three stated standard deviations.
    assert abs(float(first.next_measurement.t_reactor_k) - initial_reactor) < 3 * noise
    assert abs(float(second.next_measurement.t_reactor_k) - reactor_at_10_s) < 3 * noise


def test_undeclared_scenario_refuses_before_native_contact() -> None:
    with pytest.raises(ValueError, match="declared public scenario"):
        native_task_id("fabricated", "native", "pulse")

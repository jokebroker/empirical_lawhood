"""Independent bounded checks for distinct retained reactor transforms."""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import FIRST
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.discovery import positive_scale
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import Observation, features
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.config import Pulse
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.qualification import cp
from empirical_lawhood.adapters.methods.reactor_regime_response.causal_preparation import causal_delivery_prefix
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.words import project_pulse as project_staged_pulse
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator, project_tape_stages
from empirical_lawhood.adapters.simulators.reactor_prepared_feed_qualification.acquisition import candidate_features as feed_candidate_features
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.words import project_pulse as project_frontier_pulse
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.trace import pulse_command
from empirical_lawhood.adapters.simulators.reactor_local_domain_qualification.acquisition import candidate_features as local_candidate_features
from empirical_lawhood.adapters.simulators.reactor_regime_response.panel import preparation_tape
from empirical_lawhood.adapters.simulators.reactor_prefix_response.contracts import ReactorCommand


def test_empirical_actuator_keeps_four_action_stages_and_kg_conservation() -> None:
    actuator = Actuator()
    observation = Observation(600, 320, 316, 287.29)
    request = (0.032, 340.0)
    one_second = actuator.project_request(observation, (0, 316), request, 0, 1)
    half_second = actuator.project_request(observation, (0, 316), request, 0, 0.5)
    assert one_second.requested == half_second.requested == request
    assert one_second.accepted == half_second.accepted == request
    # The source actuator can move feed by at most 10 × 0.002 kg/s and
    # jacket by at most 10 × 0.23 K during one 10 s callback.
    assert one_second.applied == half_second.applied == (0.02, 318.3)
    for projection, dt, count in ((one_second, 1, 10), (half_second, 0.5, 20)):
        assert len(projection.exposure) == count
        realized_kg = sum(step * feed for _, step, feed, _ in projection.exposure)
        assert realized_kg == pytest.approx(0.01, abs=1e-10)
        assert projection.next_dose == pytest.approx(287.3)
        assert (
            max(feed for _, _, feed, _ in projection.exposure) < projection.applied[0]
        )
        assert projection.mean_feed == pytest.approx(realized_kg / 10)
    with pytest.raises(ValueError, match="invalid native projection input"):
        actuator.project_request(observation, (0, 316), request, 0, 0.25)


def test_history_pulse_is_only_inside_predeclared_six_hundred_seconds() -> None:
    def jacket(time: int, value: str = "300") -> Decimal:
        command = ReactorCommand(
            "target.history-check", Decimal(time), Decimal(0), Decimal(value)
        )
        return pulse_command(command).jacket_k

    assert jacket(7190) == Decimal(300)
    assert jacket(7200) == jacket(7790) == Decimal(299)
    assert jacket(7800) == Decimal(300)
    assert jacket(7200, "279.5") == Decimal("279.3")  # physical lower clip


def test_frontier_exact_root_interval_and_classical_adverse_scale() -> None:
    lower = float(cp(95, 96, Decimal("0.05"), lower=True))
    # Binomial inversion: at the one-sided lower endpoint P(X >= 95) = .05.
    upper_tail = 96 * lower**95 * (1 - lower) + lower**96
    assert upper_tail == pytest.approx(0.05, abs=1e-11)
    assert float(cp(94, 96, Decimal("0.05"), lower=True)) < lower
    assert positive_scale(Decimal("0.00123"), Decimal("0.4")) == Decimal("0.00049")
    assert positive_scale(Decimal("-0.00123"), Decimal("0.4")) == Decimal("-0.00123")
    with pytest.raises(ValueError, match="exact binomial"):
        cp(97, 96, Decimal("0.05"), lower=True)


def test_empirical_features_use_only_callback_history_and_applied_action() -> None:
    actuator = Actuator()
    history = (
        Observation(0, 320, 316, 0),
        Observation(10, 322, 317, 0),
        Observation(20, 324, 318, 0),
    )
    candidate = actuator.project_request(history[-1], (0, 316), (0, 340), 0, 1)
    vector = features(history, 316, candidate)
    assert len(vector) == 23
    assert vector[7] == pytest.approx((324 - 318.4) / 40)  # current K receiver
    assert vector[6] == pytest.approx((candidate.applied[1] - 316) / 40)
    assert vector[13] == pytest.approx((324 - 320) / 40)  # available lag
    assert np.isfinite(vector).all()
    with pytest.raises(ValueError, match="available callback"):
        features((history[0], history[2]), 316, candidate)


def test_regime_preparation_preserves_prefix_and_equal_requested_feed_mass() -> None:
    requests = np.tile(np.array((0.0, 316.0)), (2880, 1))
    requests[59] = (0.008, 317.0)
    control = preparation_tape(requests, 60, "C", 317.0)
    probe = preparation_tape(requests, 60, "P", 317.0)
    np.testing.assert_array_equal(control[:60], requests[:60])
    np.testing.assert_array_equal(probe[:60], requests[:60])
    assert np.sum(control[60:90, 0]) * 10 == pytest.approx(4.8)
    assert np.sum(probe[60:90, 0]) * 10 == pytest.approx(4.8)
    assert not np.array_equal(control[60:90, 0], probe[60:90, 0])
    np.testing.assert_array_equal(control[155:], requests[155:])
    np.testing.assert_array_equal(probe[155:], requests[155:])
    with pytest.raises(ValueError, match="preparation tape input differs"):
        preparation_tape(requests, 60, "unregistered", 317.0)


def test_regime_causal_prefix_rejects_false_realized_dose_in_both_views() -> None:
    requests = np.tile(np.array((0.016, 316.0)), (2880, 1))
    callback = 65
    for dt in (1.0, 0.5):
        stages, exposure = project_tape_stages(requests, dt)
        observations = np.zeros((callback + 1, 4))
        observations[:, 0] = np.arange(callback + 1) * 10
        observations[:, 1:3] = (320.0, 316.0)
        mass = 0.0
        for index in range(callback):
            for _, duration, feed, _ in exposure[index]:
                mass += duration * feed
            observations[index + 1, 3] = mass
        assert mass == pytest.approx(0.8)
        assert causal_delivery_prefix(
            observations, requests, stages, exposure, callback, dt
        )
        forged = observations.copy()
        forged[-1, 3] += 0.001
        assert not causal_delivery_prefix(
            forged, requests, stages, exposure, callback, dt
        )
        later = requests.copy()
        later[callback:] = (0.032, 340.0)
        assert causal_delivery_prefix(
            observations, later, stages, exposure, callback, dt
        )


def test_feed_contrast_zeros_only_declared_action_features_and_refuses_outside_support() -> (
    None
):
    operator = [(Decimal(0), Decimal(0))] * 24
    operator[5] = (Decimal(3), Decimal(0))
    operator[11] = (Decimal(2), Decimal(0))
    operator[21] = (Decimal(-1), Decimal(0))
    operator[-1] = (Decimal(320), Decimal(0))
    domain = FeedDomain(
        "d11010",
        ((3, Decimal(".500001"), True),),
        (Decimal(0),) * 23,
        (Decimal(1),) * 23,
        tuple(operator),
        (Decimal(-100),) * 14,
        (Decimal(100),) * 14,
        tuple(16 if action in (1, 4, 7) else 0 for action in range(9)),
        (0, 1),
    )
    x = np.zeros((3, 23))
    x[:, 3] = 0.5
    x[:, (5, 11, 21)] = 0.001
    prediction = domain.predict(x)
    np.testing.assert_allclose(prediction, [[320.004, -0.004]] * 3, atol=1e-12)
    assert domain.proposed_support(x, np.array((1, 4, 7))).all()
    outside = x.copy()
    outside[:, 5] = 1
    assert not domain.proposed_support(outside, np.array((1, 4, 7))).any()
    at_edge = x.copy()
    at_edge[:, 3] = 0.500001
    assert domain.contains(at_edge).all()
    beyond = at_edge.copy()
    beyond[:, 3] = 0.500002
    assert not domain.contains(beyond).any()


def test_feed_candidate_context_ignores_future_observations_and_native_truth() -> None:
    observations = np.column_stack(
        (
            np.arange(2880) * 10.0,
            np.full(2880, 320.0),
            np.full(2880, 316.0),
            np.zeros(2880),
        )
    )
    episode = NativeEpisode(
        "synthetic-root",
        "exploration-unshifted",
        1.0,
        observations,
        np.tile((0.0, 316.0), (2880, 1)),
        np.tile((0.0, 316.0, 0.0, 316.0), (2880, 1)),
        np.zeros((2880, 10, 4)),
        np.zeros((28801, 8)),
        np.zeros(2880),
        None,
    )
    base = feed_candidate_features(episode, 60)
    future = observations.copy()
    future[61:, 1:] = np.nan
    poisoned = replace(
        episode,
        observations=future,
        grid=np.full_like(episode.grid, np.nan),
    )
    for first, second in zip(base, feed_candidate_features(poisoned, 60), strict=True):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(base[1][[1, 4, 7], 0], [0, 0.016, 0.032])
    local = local_candidate_features(episode, 60)
    changed = local_candidate_features(poisoned, 60)
    for first, second in zip(local, changed, strict=True):
        np.testing.assert_array_equal(first, second)
    assert local[0].shape == (9, 23)
    np.testing.assert_array_equal(local[1][:, 0], [0] * 3 + [0.016] * 3 + [0.032] * 3)


def test_frontier_and_classical_pulses_keep_distinct_native_delivery() -> None:
    observation = Observation(600, 320, 316, 0)
    frontier = project_frontier_pulse(
        observation,
        (0, 316),
        Pulse(Decimal(".016"), 30),
        decision_id="target.frontier",
        history_id="target.history",
        horizon_id="target.horizon",
    )
    # Three 10 s requests at 0.016 kg/s and nine shutdown commands.
    assert len(frontier.mappings) == 12
    assert frontier.applied_mass_kg == frontier.realized_mass_kg == Decimal(".48")
    assert [m.command.feed_kg_s for m in frontier.mappings[3:]] == [Decimal(0)] * 9

    classical = project_staged_pulse(
        observation,
        (0, 316),
        FIRST,
        decision_id="target.classical",
        history_id="target.history",
        horizon_id="target.horizon",
        guard_s=240,
        receiver_id="c1",
    )
    # A requested 0.032 kg/s for 10 s is rate-limited to 0.020 kg/s;
    # 23 further zero commands preserve the induced-history guard.
    assert len(classical.mappings) == 24
    assert classical.mappings[0].command.feed_kg_s == Decimal(".032")
    assert classical.mappings[0].applied[0] == Decimal(".020")
    assert classical.applied_mass_kg == classical.realized_mass_kg == Decimal(".20")
    assert [m.command.feed_kg_s for m in classical.mappings[1:]] == [Decimal(0)] * 23
    with pytest.raises(ValueError, match="callback clock/dose"):
        project_frontier_pulse(
            Observation(605, 320, 316, 0),
            (0, 316),
            Pulse(Decimal(".016"), 30),
            decision_id="target.frontier",
            history_id="target.history",
            horizon_id="target.horizon",
        )

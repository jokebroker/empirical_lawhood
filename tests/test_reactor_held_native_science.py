"""One exposed held-source root per selected batch, history, feed and local contract."""

import os
from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.held

from empirical_lawhood.adapters.methods.reactor_selected_action_response.config import ClassicalDesign
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import FIRST, SECONDS
from empirical_lawhood.adapters.methods.reactor_staged_pulse_response.config import ZERO as classical_zero
from empirical_lawhood.adapters.methods.reactor_causal_response.numerical import CausalFeatures, Observation, features
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.config import ROOTS as feed_roots
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.config import FeedQualificationDesign
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.config import freeze_discovery as freeze_feed_discovery
from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.scoring import root_scores as feed_root_scores
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.config import ROOTS as frontier_roots
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.config import ZERO, FrontierDesign, Pulse
from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.qualification import cp as frontier_cp
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import ROOTS as local_roots
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import LocalQualificationDesign
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import freeze_discovery as freeze_local_discovery
from empirical_lawhood.adapters.methods.reactor_regime_response.config import ROOTS as regime_roots
from empirical_lawhood.adapters.methods.reactor_regime_response.config import ReactorRegimeResponseDesign
from empirical_lawhood.adapters.methods.reactor_regime_response.preparation_evidence import preparation_evidence
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.acquisition import causal_context as staged_pulse_causal_context
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.acquisition import native_window as staged_pulse_native_window
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.words import project_pulse as project_staged_pulse
from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import ExplorationController, TapeController, acquire_episode
from empirical_lawhood.adapters.simulators.reactor_causal_response.design import draw_scenario
from empirical_lawhood.adapters.simulators.reactor_causal_response.interface import Actuator, measured_labels
from empirical_lawhood.adapters.simulators.reactor_prepared_feed_qualification.acquisition import acquire_feed_root
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.acquisition import native_window as frontier_native_window
from empirical_lawhood.adapters.simulators.reactor_finite_control_frontier.words import project_pulse, pulse_tape
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.design import assigned_scenarios as history_scenarios
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.trace import acquire_history_unit
from empirical_lawhood.adapters.simulators.reactor_local_domain_qualification.acquisition import assay_anchors as local_assay_anchors
from empirical_lawhood.adapters.simulators.reactor_local_domain_qualification.acquisition import candidate_features as local_candidate_features
from empirical_lawhood.adapters.simulators.reactor_regime_response.acquisition import _episode as regime_episode
from empirical_lawhood.adapters.simulators.reactor_regime_response.acquisition import acquire_preparation as acquire_regime_preparation
from empirical_lawhood.adapters.simulators.reactor_regime_response.panel import assay_tape as regime_assay_tape
from empirical_lawhood.adapters.simulators.reactor_regime_response.panel import select_anchors
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import assigned_scenarios as batch_scenarios
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_input import ReactorBatchInput, check_batch_input, load_batch_source
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_trace import acquire_reference_batch
from empirical_lawhood.adapters.simulators.reactor_prefix_response.finite_words import overlay_feed_tape
from empirical_lawhood.adapters.simulators.reactor_prefix_response.prepared_input import ReactorPreparedInput, check_prepared_input
from empirical_lawhood.api.codecs import load_registered_authoring

ROOT = Path(__file__).parents[1]


def _held_root() -> Path:
    source_name = os.environ.get("REACTOR_HELD_SOURCE_ROOT")
    if not source_name:
        pytest.skip("authentic held reactor source is not mounted")
    return Path(source_name)


def _local_discovery() -> Path:
    source_name = os.environ.get("REACTOR_LOCAL_DISCOVERY")
    if not source_name:
        pytest.skip("authentic held reactor discovery is not mounted")
    return Path(source_name)


def test_shipped_batch_selection_has_two_native_views_and_realized_mass() -> None:
    root = _held_root()
    config = load_registered_authoring(
        ROOT / "experiments/reactor-response/batch-input.json",
        root_schemas={ReactorBatchInput.SCHEMA: ReactorBatchInput},
        maximum_bytes=16 * 1024,
    )
    selection = check_batch_input(config, root)
    assert selection["provider_built"] and selection["native_contact"] is False
    source = load_batch_source(root)
    unit = batch_scenarios()[0]
    traces = tuple(
        acquire_reference_batch(source, unit, dt) for dt in (Decimal(1), Decimal(".5"))
    )
    for dt, trace in zip((Decimal(1), Decimal(".5")), traces, strict=True):
        assert trace.failure_code is None
        assert (
            trace.scenario == unit and trace.source_sha256 == selection["source_sha256"]
        )
        assert (
            len(trace.states)
            == len(trace.forecasts)
            == len(trace.observed_stages)
            == 2880
        )
        assert len(trace.native_grid) == int(Decimal(28800) / dt) + 1
        steps_per_callback = int(Decimal(10) / dt)
        assert all(
            stages[2:] == trace.native_grid[index * steps_per_callback + 1][6:8]
            for index, stages in enumerate(trace.observed_stages)
        )
        dose = sum((row[6] * dt for row in trace.native_grid[1:]), Decimal(0))
        terminal = trace.native_grid[-1]
        assert abs(dose - terminal[3]) < Decimal("1e-7")
        assert terminal[0] == 28800
        assert Decimal(250) < terminal[1] < Decimal(400)  # K
        assert Decimal(0) <= terminal[3] <= Decimal("287.300001")  # kg
        assert Decimal(0) <= terminal[4] <= 1  # conversion, dimensionless
        assert trace.observed_stages[60][0] == Decimal(".032")
        assert trace.observed_stages[60][2] == Decimal(".02")
    for column, tolerance in ((1, Decimal(".001")), (4, Decimal(".00001"))):
        assert (
            abs(traces[0].native_grid[-1][column] - traces[1].native_grid[-1][column])
            < tolerance
        )


def test_shipped_history_selection_preserves_matched_pulse_and_recovery() -> None:
    root = _held_root()
    config = load_registered_authoring(
        ROOT / 'experiments/reactor-response/matched-replay-history-input.json',
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    selection = check_prepared_input(config, source_root=root, discovery_file=None)
    assert selection["provider_built"] and selection["native_contact"] is False
    source = load_batch_source(root)
    unit = history_scenarios()[0]
    trace = acquire_history_unit(source, unit)
    assert trace.failure_code is None and trace.scenario == unit
    assert len(trace.reference) == len(trace.history_forecasts) == 2
    assert [len(view.native_grid) for view in trace.reference] == [28801, 57601]
    assert trace.matched_refined is not None
    assert trace.pulse_native is not None and trace.pulse_refined is not None
    assert trace.reference[0].observed_stages == trace.matched_refined.observed_stages
    assert trace.pulse_native.observed_stages == trace.pulse_refined.observed_stages
    donor = trace.reference[0].observed_stages
    pulse = trace.pulse_native.observed_stages
    for callback in (719, 780, 900):
        assert pulse[callback][1] == donor[callback][1]
    for callback in (720, 779):
        assert pulse[callback][1] == donor[callback][1] - 1  # requested K
    assert pulse[720][3] > pulse[720][1]  # actuator rate limit at onset
    assert pulse[900] == donor[900]  # recovered common input
    assert trace.history_forecasts[0].baseline[0] == trace.history_forecasts[0].pulse[0]
    assert len(trace.pulse_native.native_grid) == 28801
    assert len(trace.pulse_refined.native_grid) == 57601


def test_shipped_feed_selection_has_three_matched_native_actions_and_two_receivers() -> (
    None
):
    root, discovery = _held_root(), _local_discovery()
    config = load_registered_authoring(
        ROOT / "experiments/reactor-response/feed-input.json",
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    selection = check_prepared_input(config, source_root=root, discovery_file=discovery)
    assert selection["factory"] == 'FeedNativeFactory'
    assert selection["status"] == "UPSTREAM_PORTS_REQUIRED"
    assert selection["provider_built"] is False
    assert selection["native_contact"] is False

    design = FeedQualificationDesign(freeze_feed_discovery(discovery.read_bytes()))
    evidence = acquire_feed_root(design, load_batch_source(root), feed_roots[0][0])
    assert evidence.assays == (("d11010", 0, 326, (1, 4, 7)),)
    assert evidence.failures == () and evidence.native_calls == 8
    arrays = evidence.arrays.unpack()
    callback = evidence.assays[0][2]
    assert callback is not None
    native_peaks = np.empty((3, 2))
    for action, requested, applied in (
        (1, 0.0, 0.0),
        (4, 0.016, 0.016),
        (7, 0.032, 0.020),
    ):
        for view, dt in enumerate((1.0, 0.5)):
            key = f"d11010_a{action}_v{view}"
            request, stage = arrays[f"{key}_request"], arrays[f"{key}_stages"]
            exposure, grid = arrays[f"{key}_exposure"], arrays[f"{key}_grid"]
            assert request[0] == pytest.approx(requested)
            assert stage[0] == pytest.approx(requested)
            assert stage[2] == pytest.approx(applied)
            assert stage[3] == pytest.approx(
                arrays[f"exploration_unshifted_v{view}_stages"][callback - 1, 3]
            )
            assert exposure.shape == (int(10 / dt), 4)
            assert np.all(exposure[:, 2] == pytest.approx(applied))
            assert np.all(exposure[:, 3] == stage[3])
            assert sum(exposure[:, 1] * exposure[:, 2]) == pytest.approx(10 * applied)
            assert grid.shape == (int(10 / dt) + 1, 8)
            assert grid[-1, 3] - grid[0, 3] == pytest.approx(10 * applied)
            assert np.array_equal(
                arrays[f"{key}_prefix_sha256"],
                np.frombuffer(
                    sha256(
                        arrays[f"exploration_unshifted_v{view}_observations"][: callback + 1].tobytes()
                    ).digest(),
                    dtype=np.uint8,
                ),
            )
            native_peaks[(1, 4, 7).index(action), view] = grid[:, 1].max()

    assert np.max(np.abs(native_peaks[:, 0] - native_peaks[:, 1])) < 0.01  # K
    cooling = native_peaks[0:1] - native_peaks
    assert cooling[0].tolist() == [0, 0]
    assert 0 < cooling[1, 0] < cooling[2, 0] < 0.01  # K, zero action contrast
    assert np.max(np.abs(cooling[:, 0] - cooling[:, 1])) < 1e-6  # K

    # Independent affine calculation of the frozen model's zero-feed contrast.
    # Only the three declared action columns may differ from the zero action.
    domain = design.atlas.domains[0]
    x = arrays["d11010_features"][list(domain.actions)]
    operator = np.asarray(domain.operator, dtype=float)
    scale = np.asarray(domain.scale, dtype=float)
    fixture = -sum(x[:, j] * operator[j, 0] / scale[j] for j in (5, 11, 21))
    np.testing.assert_allclose(domain.predict(x)[:, 1], fixture, atol=1e-12)
    assert fixture[0] == pytest.approx(0.0)
    scores = feed_root_scores(design, evidence)
    assert tuple(score.receiver for score in scores) == (0, 1)
    assert all(score.covered_rows == 1 and not score.invalidity for score in scores)


def test_shipped_local_selection_checks_all_nine_native_actions_after_causal_cutoff() -> (
    None
):
    root, discovery = _held_root(), _local_discovery()
    config = load_registered_authoring(
        ROOT / "experiments/reactor-response/local-input.json",
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    selection = check_prepared_input(config, source_root=root, discovery_file=discovery)
    assert selection["factory"] == 'LocalNativeFactory'
    assert selection["status"] == "UPSTREAM_PORTS_REQUIRED"
    assert selection["provider_built"] is False
    assert selection["native_contact"] is False

    source = load_batch_source(root)
    design = LocalQualificationDesign(freeze_local_discovery(discovery.read_bytes()))
    root_id, role, ordinal, seed = local_roots[0]
    assert role == "calibration"
    scenario = draw_scenario(root_id, role, seed)
    donors = tuple(
        acquire_episode(
            source, scenario, f"E{policy}", ExplorationController(ordinal % 3, policy)
        )
        for policy in range(3)
    )
    assert all(donor.complete for donor in donors)
    domain_id, policy, callback, actions = next(
        row for row in local_assay_anchors(design, donors) if row[0] == "d11010"
    )
    assert (domain_id, policy, callback, actions) == ("d11010", 0, 290, tuple(range(9)))
    assert policy is not None and callback is not None
    donor = donors[policy]
    features, requests = local_candidate_features(donor, callback)
    domain = next(d for d in design.atlas.candidates if d.domain_id == domain_id)
    assert domain.nominated_receivers == (0,)
    assert domain.proposed_support(features[list(actions)], np.asarray(actions)).all()
    outside = features[list(actions)].copy()
    outside[:, 0] = 1e9
    assert not domain.proposed_support(outside, np.asarray(actions)).any()
    changed_observations, changed_grid = donor.observations.copy(), donor.grid.copy()
    changed_observations[callback + 1 :, 1:] = 1e8
    changed_grid[(callback + 1) * 10 :, 1:] = 1e8
    poisoned = replace(donor, observations=changed_observations, grid=changed_grid)
    poisoned_features, poisoned_requests = local_candidate_features(poisoned, callback)
    np.testing.assert_array_equal(poisoned_features, features)
    np.testing.assert_array_equal(poisoned_requests, requests)

    refined_donor = acquire_episode(
        source, scenario, "exploration-unshifted", TapeController(donor.requests), dt=0.5
    )
    assert refined_donor.complete
    assert np.array_equal(refined_donor.requests, donor.requests)
    peaks = np.empty((9, 2))
    for action in actions:
        tape = donor.requests.copy()
        tape[callback] = requests[action]
        for view, (dt, view_donor) in enumerate(((1.0, donor), (0.5, refined_donor))):
            branch = acquire_episode(
                source, scenario, f"{domain_id}-a{action}", TapeController(tape), dt=dt
            )
            assert branch.complete and branch.root == root_id
            np.testing.assert_array_equal(branch.requests, tape)
            np.testing.assert_array_equal(
                branch.observations[: callback + 1],
                view_donor.observations[: callback + 1],
            )
            np.testing.assert_array_equal(
                branch.stages[:callback], view_donor.stages[:callback]
            )
            assert branch.stages[callback, 0] == pytest.approx(tape[callback, 0])
            assert branch.stages[callback, 1] == pytest.approx(tape[callback, 1])
            step = int(10 / dt)
            window = branch.grid[callback * step : (callback + 1) * step + 1]
            assert window.shape == (step + 1, 8)
            assert np.array_equal(window[1:, 6], branch.exposure[callback, :, 2])
            assert np.array_equal(window[1:, 7], branch.exposure[callback, :, 3])
            assert window[-1, 3] - window[0, 3] == pytest.approx(
                sum(branch.exposure[callback, :, 1] * branch.exposure[callback, :, 2])
            )
            peaks[action, view] = window[:, 1].max()  # nominated K receiver
            labels = measured_labels(
                branch.grid[:, 0],
                branch.grid[:, 1],
                branch.grid[:, 4],
                branch.grid[:, 3],
                dt,
            )
            assert labels[callback, 0] == pytest.approx(peaks[action, view])
            assert 0 <= labels[callback, 1] <= 1  # conversion, dimensionless
    assert np.isfinite(peaks).all()
    assert np.ptp(peaks[:, 0]) > 0.001  # distinct native action responses, K
    assert np.max(np.abs(peaks[:, 0] - peaks[:, 1])) < 0.01  # nested views, K


def test_shipped_regime_selection_has_common_c_p_preparation_and_phase_receivers() -> (
    None
):
    root, discovery = _held_root(), _local_discovery()
    config = load_registered_authoring(
        ROOT / "experiments/reactor-response/regime-input.json",
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    selection = check_prepared_input(config, source_root=root, discovery_file=discovery)
    assert selection["factory"] == 'RegimeNativeFactory'
    assert selection["status"] == "UPSTREAM_PORTS_REQUIRED"
    assert selection["provider_built"] is False
    assert selection["native_contact"] is False

    source = load_batch_source(root)
    prepared_domain = freeze_feed_discovery(discovery.read_bytes()).domains[0]
    design = ReactorRegimeResponseDesign()
    root_id = regime_roots[0][0]
    causal, private = acquire_regime_preparation(
        design, source, prepared_domain, root_id
    )
    assert causal.root == private.root == root_id
    assert causal.native_calls == private.native_calls == 6
    assert not causal.failures and not private.failures
    t0 = {name: callback for name, callback, _ in causal.anchors}["prepared_t0"]
    assert t0 == 304
    arrays, grids = causal.arrays.unpack(), private.arrays.unpack()
    for view in (0, 1):
        reference = arrays[f"exploration_unshifted_v{view}_requests"]
        c, p = (arrays[f"{path}_v{view}_requests"] for path in ("c", "p"))
        np.testing.assert_array_equal(c[:t0], reference[:t0])
        np.testing.assert_array_equal(p[:t0], reference[:t0])
        np.testing.assert_array_equal(
            arrays[f"c_v{view}_observations"][: t0 + 1],
            arrays[f"p_v{view}_observations"][: t0 + 1],
        )
        assert not np.array_equal(c[t0 : t0 + 30, 0], p[t0 : t0 + 30, 0])
        for path, tape in (("c", c), ("p", p)):
            assert sum(tape[t0 : t0 + 30, 0] * 10) == pytest.approx(4.8)
            exposure = arrays[f"{path}_v{view}_exposure"]
            realized = sum(
                exposure[t0 : t0 + 30, :, 1].ravel()
                * exposure[t0 : t0 + 30, :, 2].ravel()
            )
            assert realized == pytest.approx(4.8, abs=1e-8)
            grid = grids[f"{path}_v{view}_grid"]
            step = 10 if view == 0 else 20
            assert grid[(t0 + 30) * step, 3] - grid[t0 * step, 3] == pytest.approx(
                realized, abs=1e-8
            )
    for route in ("c_q", "p_q"):
        selected = preparation_evidence(causal, private, route)
        assert selected.safe and selected.prefix_valid and not selected.reasons

    scenario = draw_scenario(root_id, "calibration", regime_roots[0][3])
    for path in ("c", "p"):
        for label, offset in (("q", 33), ("q600", 93)):
            callback = t0 + offset
            peaks = np.empty((2, 2))
            for view, dt in enumerate((1.0, 0.5)):
                donor = regime_episode(root_id, f"{path}_v{view}", arrays, grids)
                assert donor is not None and donor.complete
                for word, feed in enumerate((0.0, 0.032)):
                    tape = regime_assay_tape(donor, callback, feed)
                    branch = acquire_episode(
                        source,
                        scenario,
                        f"{path}_{label}_v{view}_a{word}",
                        TapeController(tape),
                        dt=dt,
                    )
                    assert branch.complete and branch.root == root_id
                    np.testing.assert_array_equal(
                        branch.observations[: callback + 1],
                        donor.observations[: callback + 1],
                    )
                    np.testing.assert_array_equal(branch.requests, tape)
                    assert branch.requests[callback, 0] == pytest.approx(feed)
                    assert branch.stages[callback, 3] == pytest.approx(
                        donor.stages[callback - 1, 3]
                    )
                    exposure = branch.exposure[callback]
                    assert np.all(exposure[:, 3] == branch.stages[callback, 3])
                    mass = sum(exposure[:, 1] * exposure[:, 2])
                    assert mass == pytest.approx(
                        0 if word == 0 else 10 * branch.stages[callback, 2]
                    )
                    if word == 0:
                        assert mass == 0
                    else:
                        assert mass > 0
                    step = int(10 / dt)
                    window = branch.grid[callback * step : (callback + 1) * step + 1]
                    peaks[word, view] = window[:, 1].max()
                    assert window[-1, 3] - window[0, 3] == pytest.approx(mass)
            cooling = peaks[0] - peaks[1]
            assert np.all(cooling > 0)  # phase-local K receiver
            assert abs(cooling[0] - cooling[1]) < 1e-6


def test_shipped_frontier_selection_checks_zero_pulse_and_120_second_guard() -> None:
    root, discovery = _held_root(), _local_discovery()
    config = load_registered_authoring(
        ROOT / 'experiments/reactor-response/finite-control-frontier-input.json',
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    selection = check_prepared_input(config, source_root=root, discovery_file=discovery)
    assert selection["factory"] == 'FrontierNativeFactory'
    assert selection["status"] == "UPSTREAM_PORTS_REQUIRED"
    assert selection["provider_built"] is False
    assert selection["native_contact"] is False

    source = load_batch_source(root)
    prepared_domain = freeze_feed_discovery(discovery.read_bytes()).domains[0]
    root_id, role, _, seed = frontier_roots[0]
    assert role == "development"
    scenario = draw_scenario(root_id, "calibration", seed)
    donor = acquire_episode(source, scenario, "exploration-unshifted", ExplorationController(0, 0))
    refined_donor = acquire_episode(
        source, scenario, "exploration-unshifted", TapeController(donor.requests), dt=0.5
    )
    assert donor.complete and refined_donor.complete
    callback = next(
        a.callback for a in select_anchors(donor, prepared_domain) if a.name == "early"
    )
    assert callback == 60
    jacket = float(donor.stages[callback - 1, 3])
    observation = Observation(*map(float, donor.observations[callback]))
    previous = tuple(map(float, donor.stages[callback - 1, 2:]))
    pulse = Pulse(Decimal(".016"), 30)
    projected = project_pulse(
        observation,
        previous,
        pulse,
        decision_id="frontier-held-development-check",
        history_id="frontier-held-development-history",
        horizon_id="frontier-held-development-guard-120",
    )
    assert projected.realized_mass_kg == Decimal(".48")
    assert len(projected.mappings) == 12  # complete 120 s guard
    peaks = np.empty((2, 2))
    for word, selected in enumerate((ZERO, pulse)):
        tape = pulse_tape(donor.requests, callback, selected, jacket)
        assert np.all(tape[callback : callback + 3, 0] == (0 if word == 0 else 0.016))
        assert np.all(tape[callback + 3 : callback + 12, 0] == 0)
        for view, view_donor in enumerate((donor, refined_donor)):
            branch = acquire_episode(
                source,
                scenario,
                f"frontier-{selected.word_id}-v{view}",
                TapeController(tape),
                dt=view_donor.dt,
            )
            assert branch.complete and branch.root == root_id
            window = frontier_native_window(view_donor, branch, callback, tape)
            assert window["valid"].tolist() == [1]
            grid = window["grid"]
            assert grid.shape == (int(120 / branch.dt) + 1, 8)
            assert grid[:, 0].tolist() == pytest.approx(
                (callback * 10 + np.arange(len(grid)) * branch.dt).tolist()
            )
            exposure = window["exposure"]
            realized = sum(exposure[:, :, 1].ravel() * exposure[:, :, 2].ravel())
            assert realized == pytest.approx(0 if word == 0 else 0.48, abs=1e-8)
            assert grid[-1, 3] - grid[0, 3] == pytest.approx(realized, abs=1e-8)
            peaks[word, view] = grid[:, 1].max()
            assert peaks[word, view] <= 356.2  # K safety guard
    # On this early exposed root the finite pulse warms the short guard peak.
    # Preserve the observed sign; the frozen contract asks for peak and safety.
    assert np.all(peaks[1] - peaks[0] > 0.003)  # K
    assert np.max(np.abs(peaks[:, 0] - peaks[:, 1])) < 0.01

    # Exact one-sided binomial denominator is physical roots, never two views.
    design = FrontierDesign()
    q = float(frontier_cp(95, 96, design.qualification_alpha, lower=True))
    alpha = float(design.qualification_alpha)
    assert 96 * q**95 * (1 - q) + q**96 == pytest.approx(alpha, abs=1e-11)
    assert q > float(frontier_cp(94, 96, design.qualification_alpha, lower=True))


def test_shipped_classical_selection_measures_selected_native_action() -> None:
    root, discovery = _held_root(), _local_discovery()
    config = load_registered_authoring(
        ROOT / 'experiments/reactor-response/selected-action-response-input.json',
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    selection = check_prepared_input(config, source_root=root, discovery_file=discovery)
    assert selection["factory"] == 'ClassicalNativeFactory'
    assert selection["status"] == "UPSTREAM_PORTS_REQUIRED"
    assert selection["provider_built"] is False
    assert selection["native_contact"] is False

    source = load_batch_source(root)
    domain = freeze_feed_discovery(discovery.read_bytes()).domains[0]
    root_id, _, _, seed = regime_roots[0]  # historical exposed development root
    scenario = draw_scenario(root_id, "calibration", seed)
    nominal = acquire_episode(source, scenario, "exploration-unshifted", ExplorationController(0, 0))
    refined = acquire_episode(
        source, scenario, "exploration-unshifted", TapeController(nominal.requests), dt=0.5
    )
    assert nominal.complete and refined.complete
    callback = next(
        a.callback for a in select_anchors(nominal, domain) if a.name == "prepared_t0"
    )
    assert callback == 304
    peaks = np.empty((2, 2))
    for view, donor in enumerate((nominal, refined)):
        for word, feed in enumerate((0.0, 0.016)):
            tape = regime_assay_tape(donor, callback, feed)
            branch = acquire_episode(
                source,
                scenario,
                f"selected-action-response-action-{word}-v{view}",
                TapeController(tape),
                dt=donor.dt,
            )
            assert branch.complete and branch.root == root_id
            np.testing.assert_array_equal(
                branch.observations[: callback + 1], donor.observations[: callback + 1]
            )
            np.testing.assert_array_equal(branch.requests, tape)
            stage, exposure = branch.stages[callback], branch.exposure[callback]
            assert stage[0] == stage[2] == pytest.approx(feed)
            assert stage[1] == stage[3] == pytest.approx(donor.stages[callback - 1, 3])
            assert np.all(exposure[:, 2] == feed)
            realized = sum(exposure[:, 1] * exposure[:, 2])
            assert realized == pytest.approx(0 if word == 0 else 0.16)
            step = int(10 / donor.dt)
            grid = branch.grid[callback * step : (callback + 1) * step + 1]
            assert grid[-1, 3] - grid[0, 3] == pytest.approx(realized)
            peaks[word, view] = grid[:, 1].max()  # K
    cooling = peaks[0] - peaks[1]
    design = ClassicalDesign()
    assert all(
        float(design.response_lower_K) <= value <= float(design.response_upper_K)
        for value in cooling
    )
    assert np.all((cooling > 0.002) & (cooling < 0.003))  # held K receiver
    assert abs(cooling[0] - cooling[1]) < float(design.numerical_cooling_tolerance_K)
    assert np.max(peaks) < float(design.temperature_limit_K)


def test_shipped_classical_selection_preserves_first_and_induced_pair() -> None:
    root, discovery = _held_root(), _local_discovery()
    config = load_registered_authoring(
        ROOT / 'experiments/reactor-response/staged-pulse-response-input.json',
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    selection = check_prepared_input(config, source_root=root, discovery_file=discovery)
    assert selection["factory"] == 'ClassicalNativeFactory'
    assert selection["status"] == "UPSTREAM_PORTS_REQUIRED"
    assert selection["provider_built"] is False
    assert selection["native_contact"] is False

    source = load_batch_source(root)
    domain = freeze_feed_discovery(discovery.read_bytes()).domains[0]
    root_id, _, _, seed = regime_roots[0]  # original exposed block/role root
    scenario = draw_scenario(root_id, "calibration", seed)
    nominal = acquire_episode(source, scenario, "exploration-unshifted", ExplorationController(0, 0))
    refined = acquire_episode(
        source, scenario, "exploration-unshifted", TapeController(nominal.requests), dt=0.5
    )
    assert nominal.complete and refined.complete
    callback = next(
        a.callback for a in select_anchors(nominal, domain) if a.name == "early"
    )
    assert callback == 60
    jacket = float(nominal.stages[callback - 1, 3])
    projected = project_staged_pulse(
        Observation(*map(float, nominal.observations[callback])),
        tuple(map(float, nominal.stages[callback - 1, 2:])),
        FIRST,
        decision_id="staged-pulse-response-held-first",
        history_id="staged-pulse-response-held-history",
        horizon_id="staged-pulse-response-held-240-second-guard",
        guard_s=240,
    )
    assert len(projected.mappings) == 24
    assert projected.mappings[0].command.feed_kg_s == Decimal(".032")
    assert projected.mappings[0].applied[0] == Decimal(".020")
    assert projected.realized_mass_kg == Decimal(".20")

    second = SECONDS[1]  # 0.016 kg/s for the induced ten-second word
    rates = {
        "00": classical_zero.rates(240),
        "a0": FIRST.rates() + classical_zero.rates(),
        "aw": FIRST.rates() + second.rates(),
    }
    local = np.empty(2)
    induced = np.empty(2)
    for view, donor in enumerate((nominal, refined)):
        branches = {}
        for name, words in rates.items():
            tape = overlay_feed_tape(nominal.requests, callback, words, jacket)
            branch = acquire_episode(
                source,
                scenario,
                f"staged-pulse-response-{name}-v{view}",
                TapeController(tape),
                dt=donor.dt,
            )
            assert branch.complete and branch.root == root_id
            window = staged_pulse_native_window(
                donor, branch, callback, words, tape, name
            )
            assert window.failure is None
            arrays = window.arrays.unpack()
            assert arrays["valid"].tolist() == [1]
            assert arrays["grid"].shape == (int(240 / donor.dt) + 1, 8)
            np.testing.assert_array_equal(
                branch.observations[: callback + 1], donor.observations[: callback + 1]
            )
            np.testing.assert_array_equal(branch.requests, tape)
            first_stage = arrays["stages"][0]
            if name == "00":
                assert first_stage[0] == first_stage[2] == 0
            else:
                assert first_stage[0] == pytest.approx(0.032)
                assert first_stage[2] == pytest.approx(0.020)
                first_mass = sum(
                    arrays["exposure"][0, :, 1] * arrays["exposure"][0, :, 2]
                )
                assert first_mass == pytest.approx(0.20)
            assert np.all(arrays["stages"][:, 3] == jacket)
            guard_mass = sum(
                arrays["exposure"][:, :, 1].ravel()
                * arrays["exposure"][:, :, 2].ravel()
            )
            assert arrays["grid"][-1, 3] - arrays["grid"][0, 3] == pytest.approx(
                guard_mass
            )
            assert guard_mass == pytest.approx(
                0 if name == "00" else 0.20 if name == "a0" else 0.36
            )
            branches[name] = branch
        first_tape = overlay_feed_tape(
            nominal.requests, callback, FIRST.rates(), jacket
        )
        first_window = staged_pulse_native_window(
            donor, branches["a0"], callback, FIRST.rates(), first_tape, "first-local"
        )
        assert first_window.failure is None and first_window.duration_s == 120
        assert first_window.arrays.unpack()["grid"].shape == (
            int(120 / donor.dt) + 1,
            8,
        )
        k = int(10 / donor.dt)
        first = callback * k
        second_start = (callback + 12) * k
        local[view] = (
            branches["00"].grid[first : first + k + 1, 1].max()
            - branches["a0"].grid[first : first + k + 1, 1].max()
        )
        induced[view] = (
            branches["a0"].grid[second_start : second_start + k + 1, 1].max()
            - branches["aw"].grid[second_start : second_start + k + 1, 1].max()
        )
        if view == 0:
            first_context = staged_pulse_causal_context(
                root_id,
                "induced",
                callback + 12,
                branches["a0"],
                first_callback=callback,
            )
            second_context = staged_pulse_causal_context(
                root_id,
                "induced",
                callback + 12,
                branches["aw"],
                first_callback=callback,
            )
            assert first_context == second_context  # nominal pre-second seal
        else:
            # The typed pre-second publisher is nominal; refined exposure has
            # twenty substeps. Its raw causal prefix must still match exactly.
            for field, count in (
                ("observations", callback + 13),
                ("requests", callback + 12),
                ("stages", callback + 12),
                ("exposure", callback + 12),
            ):
                np.testing.assert_array_equal(
                    getattr(branches["a0"], field)[:count],
                    getattr(branches["aw"], field)[:count],
                )
    assert np.isfinite(local).all() and np.isfinite(induced).all()
    assert np.all((local > 0.0017) & (local < 0.0018))  # K, first local receiver
    assert np.all((induced > 0.0014) & (induced < 0.0015))  # K, second receiver
    assert abs(local[0] - local[1]) < 0.01
    assert abs(induced[0] - induced[1]) < 0.01


def test_shipped_empirical_selection_has_causal_action_and_native_receiver() -> None:
    root = _held_root()
    config = load_registered_authoring(
        ROOT / 'experiments/reactor-response/causal-response-study-input.json',
        root_schemas={ReactorPreparedInput.SCHEMA: ReactorPreparedInput},
        maximum_bytes=16 * 1024,
    )
    selection = check_prepared_input(config, source_root=root, discovery_file=None)
    assert selection["factory"] == 'EmpiricalNativeFactory'
    assert selection["comparator_sources_authenticated"] is True
    assert selection["independent_roots_in_retained_roster"] == 96
    assert selection["status"] == "UPSTREAM_PORTS_REQUIRED"
    assert selection["provider_built"] is False
    assert selection["native_contact"] is False

    source = load_batch_source(root)
    root_id, _, _, seed = regime_roots[0]  # old exposed development root
    scenario = draw_scenario(root_id, "calibration", seed)
    nominal = acquire_episode(source, scenario, "exploration-unshifted", ExplorationController(0, 0))
    refined = acquire_episode(
        source, scenario, "exploration-unshifted", TapeController(nominal.requests), dt=0.5
    )
    assert nominal.complete and refined.complete
    callback = 60  # early, at the 600 s dosing-window onset
    actuator = Actuator(source.plant_source.encode(), source.plant_params.encode())
    receivers = np.empty((2, 2, 3))
    for view, donor in enumerate((nominal, refined)):
        history = tuple(
            Observation(*map(float, row)) for row in donor.observations[: callback + 1]
        )
        state = CausalFeatures()
        for observation in history:
            state.append(observation)
        previous = tuple(map(float, donor.stages[callback - 1, 2:]))
        projections = actuator.project(history[-1], previous, donor.dt)
        assert tuple(p.action for p in projections) == tuple(range(9))
        # The menu has nine action IDs; the upper jacket clip aliases two
        # requests per feed rate on this particular causal history.
        assert len({p.requested for p in projections}) == 6
        for projection in projections:
            np.testing.assert_array_equal(
                state.features(previous[1], projection),
                features(history, previous[1], projection),
            )
        # Indices 1 and 7 keep the same jacket request; only feed changes.
        for word, action in enumerate((1, 7)):
            projection = projections[action]
            tape = nominal.requests.copy()
            tape[callback] = projection.requested
            branch = acquire_episode(
                source,
                scenario,
                f"causal-response-action-{action}-v{view}",
                TapeController(tape),
                dt=donor.dt,
            )
            assert branch.complete and branch.root == root_id
            np.testing.assert_array_equal(
                branch.observations[: callback + 1], donor.observations[: callback + 1]
            )
            np.testing.assert_array_equal(branch.requests, tape)
            np.testing.assert_allclose(
                branch.stages[callback],
                (*projection.accepted, *projection.applied),
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                branch.exposure[callback], projection.exposure, rtol=0, atol=1e-12
            )
            realized = sum(dt * feed for _, dt, feed, _ in projection.exposure)
            assert projection.next_dose - history[-1].dose == pytest.approx(realized)
            step = int(10 / donor.dt)
            grid = branch.grid[callback * step : (callback + 1) * step + 1]
            assert grid[-1, 3] - grid[0, 3] == pytest.approx(realized)
            labels = measured_labels(
                branch.grid[:, 0],
                branch.grid[:, 1],
                branch.grid[:, 4],
                branch.grid[:, 3],
                donor.dt,
            )
            assert labels.shape == (2880, 3)
            np.testing.assert_allclose(
                labels[callback],
                (grid[:, 1].max(), grid[-1, 4], grid[-1, 3]),
                rtol=0,
                atol=1e-12,
            )
            receivers[word, view] = labels[callback]
    assert np.all((receivers[1, :, 2] - receivers[0, :, 2]) > 0)
    assert np.all((receivers[0, :, 0] - receivers[1, :, 0]) > 0)
    assert np.max(np.abs(receivers[:, 0] - receivers[:, 1])) < 0.01

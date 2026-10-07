"""Full-size public joins with explicitly fictional, exposed source observations.

Native failure and differential tests separately exercise real mechanics. These
stored fixtures prove software custody and complete denominators, never science.
"""

from dataclasses import replace
from decimal import Decimal
from hashlib import sha256
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest

from tests.numerical_provenance_fixtures import PROJECT_ROOT, numerical_plane

from empirical_lawhood.api.matrix_preparation_analysis import tangent_analysis_read, tangent_preparation_configuration, tangent_preparation_source_export, tangent_response_analysis
from empirical_lawhood.adapters.methods.matrix_preparation.projection import PROJECTION_SHAPES
from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent import acquire_tangent_root, tangent_analysis
from empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent_records import TangentNativeResult, TangentPrefixResult, TangentPrefixSegment
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import NATIVE_SCHEMA, PARENTS, PreparationDelivery, preparation_actions
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import PreparationRoot, SEED_SHA256
from empirical_lawhood.adapters.simulators.matrix_preparation.scientific_inputs import preparation_scientific_root_inputs
from empirical_lawhood.adapters.simulators.matrix_preparation.source import innovation_bundle
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeDelivery, ResponseGeometryAssayNativeViewSegment
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import _checkpoint, _rng, _start, response_hdf5_text, response_hdf5_writer
from empirical_lawhood.kernel.references import ArtifactIdentity


def _synthetic_prefix(config, root):
    segment = TangentPrefixSegment(root)
    views = []
    for refinement in (1, 2):
        continuation = _start(config, segment, refinement, None, np.eye(4, dtype=np.complex128))
        continuation.state = replace(continuation.state, step_index=root.landmark * refinement,
                                     alpha_tilde_x=2 / 3, alpha_tilde_y=22 / 3)
        continuation.rng, continuation.stream = _rng(config, root, "assay.continuation")
        observer = continuation.observer
        original = observer.pending[0]
        latest = root.landmark * refinement
        observer.native_step = latest
        observer.pending = [replace(original, origin_native_step=latest - 16 * refinement), replace(original, origin_native_step=latest)]
        transfer = observer.transfers[0][1]
        observer.transfers = [(latest - 16 * refinement, transfer), (latest, transfer)]
        sample = continuation.window[0]
        continuation.window = [replace(sample, reference_tick=tick) for tick in range(root.landmark - 240, root.landmark + 1, 16)]
        continuation.x_window = [continuation.state.positions[0]] * 16
        continuation.mode = np.zeros((3, 4, 4), dtype=np.complex128)
        continuation.mode[0] = np.diag([1, -1, 0, 0]) / np.sqrt(2)
        checkpoint = _checkpoint(continuation, segment, refinement)
        delivered = ResponseGeometryAssayNativeDelivery(True, latest, 2 * latest, 0, Decimal(0), "a" * 64, None)
        views.append(ResponseGeometryAssayNativeViewSegment(refinement, "COMPLETE", latest, checkpoint, None, Decimal(0), Decimal(0), delivered))
    return TangentPrefixResult(root.root_id + ".prefix-result", config.identity, root, tuple(views))


def _software_source(config, root, progress=None):
    prefix = _synthetic_prefix(config, root)
    raw = prefix.canonical_bytes()
    artifact = ArtifactIdentity(root.root_id + ".prefix-result", "current-prefix", prefix.SCHEMA, sha256(raw).hexdigest(), "application/json", len(raw))
    stream = BytesIO()
    with response_hdf5_writer(stream) as native:
        for key, value in (("schema", NATIVE_SCHEMA), ("source_config_sha256", config.fingerprint()), ("root_sha256", root.fingerprint()), ("original_prefix_sha256", prefix.fingerprint())):
            response_hdf5_text(native, key, value)
    payload = stream.getvalue()
    deliveries = []
    for refinement in (1, 2):
        for parent in PARENTS:
            deliveries.append(PreparationDelivery(f"{root.root_id}.{parent}.parent.r{refinement}", refinement, root.landmark, root.handoff, True, 144 * refinement, 288 * refinement, 0, Decimal(0), Decimal(0), Decimal(0), "b" * 64, "COMPLETE", None))
        for action in preparation_actions(root):
            nonzero = 64 * refinement if action.sign else 0
            deliveries.append(PreparationDelivery(action.action_id + f".r{refinement}", refinement, root.handoff, root.handoff + 320, True, 320 * refinement, 640 * refinement, nonzero,
                                                  Decimal(action.sign) * action.magnitude * Decimal("0.064"), Decimal(0), Decimal(0), "c" * 64, "COMPLETE", None))
    result = TangentNativeResult(root.root_id + ".native-result", config.identity, root, artifact, None,
                                 tuple(sorted(deliveries, key=lambda item: item.occurrence_id)), sha256(payload).hexdigest(), sum(item.completed_intervals for item in deliveries))
    arrays = {name: np.zeros((2, *shape), dtype=np.float64) for name, shape in PROJECTION_SHAPES.items()}
    arrays.update(prefix_complete=np.ones(2, dtype=np.bool_), mechanism_complete=np.ones(2, dtype=np.bool_),
                  native_complete=np.asarray(True), prefix_completed_updates=np.asarray([root.landmark, 2 * root.landmark], dtype=np.int64))
    arrays["coupled_tangent"][:] = 0.25
    arrays["amplitude_chi"][:] = 0.25
    return prefix, b"explicit fictional software prefix bytes", result, payload, arrays


def test_full128_public_source_analysis_read_negative_and_recovery(tmp_path, monkeypatch):
    config = tangent_preparation_configuration(config_id="test.tangent", namespace="test.current", master_seed=90217, project_root=PROJECT_ROOT)
    monkeypatch.setattr("empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent.acquire_tangent_root", _software_source)
    source = tangent_preparation_source_export(tmp_path / "source", config=config, artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)
    assert source.disposition == "COMPLETE" and len(source.root_ids) == 128 and source.qualification == "NONE"
    result = tangent_response_analysis(tmp_path / "analysis", config=config, source_directory=tmp_path / "source", artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)
    assert result.disposition == "COMPLETE" and result.qualification == "NONE"
    metrics = {metric.name: metric.value for metric in result.metrics}
    assert metrics["independent_source_roots"] == 128 and metrics["independent_mechanism_roots"] == 32
    assert metrics["independent_view_roots"] == 128 and metrics["native_view_passing_roots"] == 128
    assert metrics["nested_mechanism_comparisons"] == 6400
    assert metrics["coupled_tangent_mse"] == 0 and metrics["scalar_tangent_mse"] > 0
    loaded, retained_config, arrays = tangent_analysis_read(tmp_path / "analysis", artifact_writer=numerical_plane(tmp_path))
    assert loaded == result and retained_config == config
    assert arrays["numerical_subset"].sum() == 32
    assert arrays["native_view_pass"].sum() == 128
    np.testing.assert_array_equal(arrays["finite_action_secant_tangent_difference"], -0.25)
    np.testing.assert_array_equal(arrays["scalar_coupled_receiver_difference"], -0.25)
    from tests.test_matrix_preparation_analysis import _assert_report_only_mutations_refuse
    _assert_report_only_mutations_refuse(tmp_path / "source", source, numerical_plane(tmp_path))
    _assert_report_only_mutations_refuse(tmp_path / "analysis", result, numerical_plane(tmp_path))
    member = tmp_path / "source" / (config.roots[20].root_id + ".native.h5")
    member.write_bytes(member.read_bytes() + b"changed")
    with pytest.raises((ValueError, RuntimeError), match="(member|byte limit)"):
        tangent_response_analysis(tmp_path / "wrong", config=config, source_directory=tmp_path / "source", artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)
    assert not (tmp_path / "wrong").exists()
    with pytest.raises(FileExistsError):
        tangent_preparation_source_export(tmp_path / "source", config=config, artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)


def test_current_real_orchestration_keeps_failed_prefix_and_every_unentered_action(monkeypatch):
    config = tangent_preparation_configuration(config_id="test.tangent-failure", namespace="test.failed", master_seed=90218, project_root=PROJECT_ROOT)
    def failure(config, segment, refinement, previous, group, progress=None):
        delivery = ResponseGeometryAssayNativeDelivery(True, 0, 0, 0, Decimal(0), "a" * 64, None)
        return ResponseGeometryAssayNativeViewSegment(refinement, "NUMERICAL_FAILURE", 0, None, "NATIVE_NUMERICAL_FAILURE", Decimal(0), Decimal(0), delivery)
    monkeypatch.setattr("empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent._run_view", failure)
    root = next(root for root in config.roots if root.numerical_semantics)
    prefix, _, result, _, arrays = acquire_tangent_root(config, root)
    assert all(view.disposition == "NUMERICAL_FAILURE" for view in prefix.views)
    assert len(result.deliveries) == 220
    assert all(delivery.disposition == "UNENTERED" for delivery in result.deliveries)
    assert not arrays["native_complete"] and not arrays["prefix_complete"].any()
    assert np.isnan(arrays["response"]).all()


def test_tangent_common_noise_and_labels_do_not_change_numeric_draws():
    first = tangent_preparation_configuration(config_id="test.one", namespace="test.one", master_seed=90219, project_root=PROJECT_ROOT)
    renamed = tangent_preparation_configuration(config_id="test.two", namespace="test.two", master_seed=90219, project_root=PROJECT_ROOT)
    a = innovation_bundle(first, first.roots[0], "parent", 144)
    b = innovation_bundle(renamed, renamed.roots[0], "parent", 144)
    np.testing.assert_array_equal(a.primary, b.primary)
    np.testing.assert_array_equal(a.half, b.half)
    independent = innovation_bundle(first, first.roots[0], "common-response", 320)
    assert not np.array_equal(a.primary[0], independent.primary[0])
    with pytest.raises(ValueError, match="clock"):
        innovation_bundle(first, first.roots[0], "parent", 400)


def test_current_and_frozen_noise_are_equivalent_for_equal_numeric_commitments():
    old_root = PreparationRoot("assembling", 0)
    old_config = SimpleNamespace(roots=(old_root,), seed_sha256=SEED_SHA256)
    current = tangent_preparation_configuration(config_id="test.equal-inputs", namespace="test.current-label", master_seed=90222, project_root=PROJECT_ROOT)
    root = replace(current.roots[0], preparation_seed_sha256=preparation_scientific_root_inputs("assembling", 0).native_seed_sha256)
    current = replace(current, roots=(root, *current.roots[1:]))
    for purpose, ticks in (("parent", 144), ("common-response", 320), ("independent-response-3", 320)):
        frozen = innovation_bundle(old_config, old_root, purpose, ticks)
        new = innovation_bundle(current, root, purpose, ticks)
        np.testing.assert_array_equal(frozen.primary, new.primary)
        np.testing.assert_array_equal(frozen.half, new.half)
        assert frozen.stream.derived_seed_sha256 == new.stream.derived_seed_sha256


def test_complete_negative_and_missing_units_keep_declared_denominators():
    config = tangent_preparation_configuration(config_id="test.evaluate", namespace="test.fixture", master_seed=90220, project_root=PROJECT_ROOT)
    arrays = {name: np.zeros((128, 2, *shape), dtype=np.float64) for name, shape in PROJECTION_SHAPES.items()}
    arrays.update(prefix_complete=np.ones((128, 2), dtype=np.bool_), mechanism_complete=np.ones((128, 2), dtype=np.bool_), native_complete=np.ones(128, dtype=np.bool_))
    chosen = next(index for index, root in enumerate(config.roots) if root.numerical_semantics)
    arrays["response"][chosen, 1, 2, 0, 2, -1] = 1 / 64
    _, metrics, incomplete = tangent_analysis(config, arrays)
    assert not incomplete and metrics["absolute_view_failing_roots"] == 1
    assert metrics["odd_view_failing_roots"] == 1 and metrics["independent_mechanism_roots"] == 32
    unselected = next(index for index, root in enumerate(config.roots) if not root.numerical_semantics)
    arrays["response"][unselected, 1, 2, 0, 2, -1] = 1 / 64
    _, metrics, incomplete = tangent_analysis(config, arrays)
    assert not incomplete and metrics["absolute_view_failing_roots"] == 2
    assert metrics["odd_view_failing_roots"] == 2 and metrics["native_view_passing_roots"] == 126
    arrays["secant_position_error"][chosen, 0, 0, 0] = 2e-8
    _, metrics, _ = tangent_analysis(config, arrays)
    assert metrics["secant_failing_roots"] == 1
    arrays["native_complete"][127] = False
    _, metrics, incomplete = tangent_analysis(config, arrays)
    assert incomplete == (config.roots[127].root_id,) and metrics["independent_source_roots"] == 128
    arrays["response"][unselected, 0, 0, 0, 0, 0] = np.nan
    _, _, incomplete = tangent_analysis(config, arrays)
    assert set(incomplete) == {config.roots[unselected].root_id, config.roots[127].root_id}


def test_interrupted_tangent_source_never_has_completed_manifest(tmp_path, monkeypatch):
    config = tangent_preparation_configuration(config_id="test.partial", namespace="test.partial", master_seed=90221, project_root=PROJECT_ROOT)
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt
        return _software_source(*args, **kwargs)
    monkeypatch.setattr("empirical_lawhood.adapters.methods.matrix_preparation_analysis.tangent.acquire_tangent_root", interrupt)
    with pytest.raises(KeyboardInterrupt):
        tangent_preparation_source_export(tmp_path / "partial", config=config, artifact_writer=numerical_plane(tmp_path), project_root=PROJECT_ROOT)
    assert (tmp_path / "partial" / (config.roots[0].root_id + ".native.h5")).is_file()
    assert not (tmp_path / "partial" / "analysis-report.canonical.json").exists()
    assert not (tmp_path / "partial" / "arrays.canonical.json").exists()
    with pytest.raises(ValueError, match="(cannot be read|completed publication)"):
        tangent_analysis_read(tmp_path / "partial", artifact_writer=numerical_plane(tmp_path))

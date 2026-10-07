"""Independent kernel limits, cell exclusion and complete current root folds."""

from dataclasses import replace

import numpy as np
import pytest

from empirical_lawhood.api.finite_operands import original_f_operand_export
from empirical_lawhood.adapters.methods.finite_response_law.transient_bridge import RADIAL_STIFFNESS, TransientBridgeInput, current_root_fold_arrays, fit_bridge, mechanical_kernel
from empirical_lawhood.kernel.matrix_inputs import MatrixAllocation
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity


def test_nonrestoring_kernel_matches_independent_convolution():
    kernel, waveform = mechanical_kernel(0)
    age0 = (399 - np.arange(400)) * 0.001
    age1 = age0 + 0.001
    velocity = np.exp(-age0) - np.exp(-age1)
    np.testing.assert_allclose(kernel[:, -1, 1], waveform @ velocity, rtol=0, atol=1e-14)
    np.testing.assert_allclose(kernel[:, -1, 0], waveform @ (0.001 - velocity), rtol=0, atol=1e-14)
    np.testing.assert_array_equal(kernel[0], 0)
    np.testing.assert_array_equal(kernel[1:5], -kernel[5:9])
    np.testing.assert_array_equal(kernel[:, 2:, 2:], (kernel[:, 2:, :2] - kernel[:, :-2, :2]) / 0.032)
    assert not kernel.flags.writeable


def test_radial_free_recovery_matches_homogeneous_oscillator():
    kernel, _ = mechanical_kernel()
    q, v = kernel[2, 16, :2]
    t = np.arange(10) * 0.016
    omega = np.sqrt(RADIAL_STIFFNESS - 0.25)
    expected = np.exp(-t / 2) * (q * np.cos(omega * t) + (v + q / 2) * np.sin(omega * t) / omega)
    np.testing.assert_allclose(kernel[2, 16:, 0], expected, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="Undeclared"):
        mechanical_kernel(1)


def test_fit_cannot_read_unentered_menu_or_later_cells():
    rng = np.random.default_rng(13)
    kernel, _ = mechanical_kernel()
    prefix = rng.normal(size=(12, 24))
    coefficients = rng.normal(size=(4, 24))
    delta = np.repeat(np.einsum("stk,kj->stj", kernel, coefficients)[None], 12, axis=0)
    schedules, times = (1, 2, 5, 6), tuple(range(2, 18))
    fit = fit_bridge(prefix, delta, kernel, schedules=schedules, times=times)
    poisoned = delta.copy()
    poisoned[:, (3, 4, 7, 8)] = 1e20
    poisoned[:, :, 18:] = -1e20
    other = fit_bridge(prefix, poisoned, kernel, schedules=schedules, times=times)
    np.testing.assert_array_equal(fit.operator, other.operator)
    np.testing.assert_allclose(fit.predict(rng.normal(size=(6, 24)), kernel), np.repeat(delta[:1], 6, axis=0), atol=1e-11)
    with pytest.raises(ValueError, match="rank four"):
        fit_bridge(prefix, delta, np.zeros_like(kernel))


def test_outer_fit_excludes_whole_held_out_root_trajectories(tmp_path):
    lower = original_f_operand_export(tmp_path, payload_id="test.original-f")
    rng = np.random.default_rng(18)
    prefix = rng.normal(size=(24, 24))
    hold = rng.normal(size=(24, 24))
    kernel, _ = mechanical_kernel()
    coefficients = rng.normal(size=(24, 4, 24))
    shift = np.einsum("stk,rkj->rstj", kernel, coefficients) * np.asarray(lower.scale, dtype=np.float64)
    baseline = prefix[:, None, None] + np.linspace(0, 1, 26)[None, None, :, None] * (hold - prefix)[:, None, None]
    trajectories = np.repeat((baseline + shift)[:, :, None], 2, axis=2)
    handoff = trajectories[:, :, :, -1].transpose(0, 1, 3, 2).copy()
    artifact = ArtifactIdentity("test.trajectory", "trajectory", "empirical-lawhood/kernel/matrix-array-bundle", "a" * 64, "application/x-npz", 100)
    source = TransientBridgeInput("test.bridge", ObjectIdentity("test.allocation", MatrixAllocation.SCHEMA, "1.0.0", "b" * 64), lower.identity, artifact, replace(artifact, artifact_id="test.panel"), tuple(f"test.root.{i}" for i in range(24)), ("q2",) * 8 + ("cir1",) * 16)
    result = current_root_fold_arrays(current_input=source, original_f=lower, primary_prefix=prefix, native_handoff=handoff, trajectories=trajectories)
    held = np.arange(24) % 4 == 0
    poisoned = trajectories.copy()
    poisoned[held, 1:, :, 1:] += 1e6
    poisoned_handoff = poisoned[:, :, :, -1].transpose(0, 1, 3, 2).copy()
    other = current_root_fold_arrays(current_input=source, original_f=lower, primary_prefix=prefix, native_handoff=poisoned_handoff, trajectories=poisoned)
    np.testing.assert_array_equal(result.fits[0]["operator"], other.fits[0]["operator"])
    np.testing.assert_array_equal(result.arrays["radial.all.forecast_delta"][held], other.arrays["radial.all.forecast_delta"][held])
    assert len(result.fits) == 4
    for row in result.fits:
        assert len(row["training_roots"]) == 18
        assert len(row["test_roots"]) == 6
        assert not set(row["training_roots"]) & set(row["test_roots"])
    assert result.arrays["radial.all.native"].shape == (24, 9, 24)
    assert all(not value.flags.writeable for value in result.arrays.values())
    with pytest.raises(ValueError, match="endpoints"):
        current_root_fold_arrays(current_input=source, original_f=lower, primary_prefix=prefix, native_handoff=handoff + 1, trajectories=trajectories)

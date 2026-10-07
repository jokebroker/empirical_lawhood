# SPDX-License-Identifier: MPL-2.0

"""Bounded interchange retains original roots and absolute native channels."""

import base64
from dataclasses import replace

import numpy as np
import pytest

from empirical_lawhood.adapters.composition.response_composition.scalar_parent import SHAPES, ResponseCompositionScalarParent, analyze_scalar_parent
from empirical_lawhood.adapters.methods.response_composition.development import ResponseCompositionDevelopmentSpec
from empirical_lawhood.kernel.decoding import decode_canonical_bytes


def test_scalar_parent_keeps_hold_subtraction_and_root_denominator(monkeypatch):
    arrays = {key: np.zeros(shape, dtype="<f8") for key, shape in SHAPES.items()}
    arrays["original_channels"][..., :2] = 10
    arrays["original_channels"][:, :, :, 1:, :, :2] = 11
    arrays["parent_work"][0, 1, 2] = 33  # One adverse nested view fails that parent.
    parent = ResponseCompositionScalarParent(
        tuple(
            f"synthetic.qualification.{context}.r{i:03d}"
            for context in ("assembling", "prepared")
            for i in range(16)
        ),
        ResponseCompositionDevelopmentSpec(*("a" * 64 for _ in range(4))),
        tuple(
            (key, base64.b64encode(value.tobytes()).decode())
            for key, value in arrays.items()
        ),
    )
    assert (
        decode_canonical_bytes(
            parent.canonical_bytes(), type(parent), maximum_bytes=8 * 1024**2
        )
        == parent
    )
    observed = []

    def consume(features, response, preservation):
        observed.append((features, response, preservation))
        return {"synthetic": True}

    monkeypatch.setattr(
        'empirical_lawhood.adapters.composition.response_composition.scalar_parent.develop_context',
        consume,
    )
    analyze_scalar_parent(parent)
    assert len(observed) == 2
    for features, response, preservation in observed:
        assert features.shape[0] == response.shape[0] == preservation.shape[0] == 16
        assert np.all(response[..., 0, :, :] == 0)
        assert np.all(response[..., 1:, :, :] == 1)
    assert not observed[0][2][0, 2]
    assert observed[1][2].all()
    with pytest.raises(ValueError, match="SIZE_INVALID"):
        replace(parent, arrays_base64=(("features", "AA=="), *parent.arrays_base64[1:]))
    arrays["features"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NONFINITE"):
        replace(
            parent,
            arrays_base64=(
                ("features", base64.b64encode(arrays["features"].tobytes()).decode()),
                *parent.arrays_base64[1:],
            ),
        )

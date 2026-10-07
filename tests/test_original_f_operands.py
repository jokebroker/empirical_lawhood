"""Original byte custody and unchanged arithmetic, without qualification effects."""

from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from empirical_lawhood.api.finite_operands import original_f_operand_export, original_f_operand_import
from empirical_lawhood.adapters.methods.finite_response_law.original_f import ORIGINAL_CALIBRATION, ORIGINAL_Q
from empirical_lawhood.adapters.methods.finite_response_law.law_payloads import _supported


@pytest.fixture
def original_f(tmp_path):
    return original_f_operand_export(tmp_path, payload_id="test.original-f")


def test_public_original_f_import_retains_distinct_identity(original_f, tmp_path):
    imported = original_f_operand_import(tmp_path / "original-f.canonical.json", source_directory=tmp_path)
    assert imported == original_f
    assert imported.q == ORIGINAL_Q
    assert imported.original_calibration == ORIGINAL_CALIBRATION
    assert imported.identity.object_fingerprint != imported.original_payload.sha256
    assert imported.original_payload.sha256 == "ab880c96a5e4d4e5070e4e2268331ea542f5cc0bedf18da43cded949d2462b1b"
    assert replace(imported, payload_id="test.other-current-id").identity != imported.identity
    assert not hasattr(imported, "calibration")


@pytest.mark.parametrize("field", ("center", "scale", "operator", "base", "multiplier"))
def test_original_f_refuses_altered_coefficients(original_f, field):
    values = getattr(original_f, field)
    with pytest.raises(ValueError, match="numerical fields"):
        replace(original_f, **{field: (values[0] + Decimal("0.000001"), *values[1:])})


def test_original_f_cannot_replace_q_with_new_calibration(original_f):
    with pytest.raises(ValueError, match="crosswalk"):
        replace(original_f, q=Decimal("0.736"))


def test_original_f_arithmetic_is_independent_affine_formula(original_f):
    center = np.asarray(original_f.center, dtype=np.float64)
    scale = np.asarray(original_f.scale, dtype=np.float64)
    handoff = center + np.linspace(-2, 2, 24)[None] * scale
    normalized = (handoff - center) / scale
    design = np.column_stack((normalized, np.ones(len(normalized))))
    expected = (design @ np.asarray(original_f.operator, dtype=np.float64).reshape(25, 32)).reshape(-1, 4, 8)
    expected[..., 2:] = np.maximum(expected[..., 2:], 0)
    factor = np.exp(np.clip(design @ np.asarray(original_f.multiplier, dtype=np.float64).reshape(25, 1), np.log(0.25), np.log(4)))
    sigma = np.asarray(original_f.base, dtype=np.float64).reshape(4, 8)[None] * factor[..., None]
    actual = original_f.predict(handoff)
    np.testing.assert_allclose(actual.mean, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(actual.sigma, sigma, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(actual.supported, [True])
    np.testing.assert_array_equal(original_f.normalize(handoff), normalized)
    with pytest.raises(ValueError, match="24 native"):
        original_f.predict(handoff.astype(np.float32))


def test_original_support_inclusive_all24_and_nonfinite():
    values = np.zeros((4, 24), dtype=np.float64)
    values[0, -1] = 6
    values[1, -1] = np.nextafter(6.0, np.inf)
    values[2, -1] = np.nan
    values[3, -1] = -6
    np.testing.assert_array_equal(_supported(values), [True, False, False, True])


def test_original_import_authenticates_every_evidence_file(original_f, tmp_path):
    target = tmp_path / "qualification-report.json"
    raw = target.read_bytes()
    target.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    with pytest.raises(ValueError, match="source bytes"):
        original_f_operand_import(tmp_path / "original-f.canonical.json", source_directory=tmp_path)


def test_original_export_refuses_overwrite_before_members(tmp_path):
    (tmp_path / "original-f.canonical.json").write_text("reserved")
    with pytest.raises(ValueError, match="existing"):
        original_f_operand_export(tmp_path, payload_id="test.original-f")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["original-f.canonical.json"]


def test_original_import_refuses_symlink_parent(original_f, tmp_path):
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="safely|symlink|contained"):
        original_f_operand_import(alias / "original-f.canonical.json", source_directory=alias)

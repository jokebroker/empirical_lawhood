"""Frozen coefficient transport cannot silently refit or reuse original q."""

from dataclasses import replace
import numpy as np
import pytest

from empirical_lawhood.api.finite_operands import current_nomination_operand_export, current_nomination_operand_import, original_f_operand_export
from empirical_lawhood.adapters.methods.finite_response_law.array_transport import decode_npz_transport
from empirical_lawhood.adapters.methods.finite_response_law.frozen_package import frozen_development
from empirical_lawhood.adapters.methods.finite_response_law.nominated_package import ORIGINAL_COEFFICIENT_SCHEMA, validate_nomination_transport_inputs
from empirical_lawhood.adapters.methods.finite_response_law.method_records import COEFFICIENT_SCHEMA


def test_public_nomination_exact_npz_transport_and_frozen_decoder(tmp_path):
    operand = current_nomination_operand_export(tmp_path, nomination_id="test.nomination")
    imported = current_nomination_operand_import(tmp_path / "nomination.canonical.json", source_directory=tmp_path)
    assert imported == operand
    assert not hasattr(operand, "q")
    current = (tmp_path / "coefficients.transport.json").read_bytes()
    original = (tmp_path / "original-coefficients.transport.json").read_bytes()
    assert decode_npz_transport(current, COEFFICIENT_SCHEMA) == decode_npz_transport(original, ORIGINAL_COEFFICIENT_SCHEMA)
    points, widths = frozen_development(report_bytes=(tmp_path / "development-report.json").read_bytes(), coefficient_bytes=current, report_identity=operand.development_report, coefficient_identity=operand.coefficients)
    assert points.roots == tuple(range(48))
    assert points.lower.shape == (25, 32)
    assert set(widths) == {"lower", "composed", "cached", "direct"}
    assert not points.lower.flags.writeable
    assert np.isfinite(points.lower).all()
    original_directory = tmp_path / "original-f"
    original_directory.mkdir()
    lower = original_f_operand_export(original_directory, payload_id="test.original-f")
    np.testing.assert_array_equal(points.nh.center, np.asarray(lower.center, dtype=np.float64))
    np.testing.assert_array_equal(points.nh.scale, np.asarray(lower.scale, dtype=np.float64))
    np.testing.assert_array_equal(points.lower, np.asarray(lower.operator, dtype=np.float64).reshape(25, 32))
    np.testing.assert_array_equal(widths["lower"].base, np.asarray(lower.base, dtype=np.float64).reshape(4, 8))
    np.testing.assert_array_equal(widths["lower"].multiplier, np.asarray(lower.multiplier, dtype=np.float64).reshape(25, 1))
    raw_inputs = tuple((tmp_path / name).read_bytes() for name in ("development-report.json", "coefficients.transport.json", "development-manifest.json"))
    validate_nomination_transport_inputs(operand.current_artifacts, raw_inputs)
    with pytest.raises(ValueError, match="crosswalk"):
        validate_nomination_transport_inputs(operand.current_artifacts, (raw_inputs[0] + b" ", *raw_inputs[1:]))


def test_nomination_original_and_current_custody_are_separate(tmp_path):
    operand = current_nomination_operand_export(tmp_path, nomination_id="test.nomination")
    assert operand.original_coefficients.sha256 == "ef2d19c9dbad93904b2bc5908af3d10c26f9506b7a52d8f0bf4579013c3de072"
    assert operand.coefficients.sha256 != operand.original_coefficients.sha256
    assert operand.development_report.sha256 != operand.original_report.sha256
    assert operand.development_report.artifact_id != operand.original_report.artifact_id
    with pytest.raises(ValueError, match="transport"):
        replace(operand, coefficients=replace(operand.coefficients, sha256="0" * 64))


def test_nomination_import_refuses_altered_current_bytes(tmp_path):
    current_nomination_operand_export(tmp_path, nomination_id="test.nomination")
    target = tmp_path / "coefficients.transport.json"
    raw = target.read_bytes()
    target.write_bytes(raw[:-1] + b" ")
    with pytest.raises(ValueError, match="coefficient transport"):
        current_nomination_operand_import(tmp_path / "nomination.canonical.json", source_directory=tmp_path)

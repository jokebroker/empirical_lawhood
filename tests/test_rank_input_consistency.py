# SPDX-License-Identifier: MPL-2.0
"""Bind certified rank and family records to the same entered matrix values."""

from hashlib import sha256
from types import SimpleNamespace

import numpy as np
import pytest

from empirical_lawhood.adapters.methods import certified_rank as rank
from empirical_lawhood.kernel.serialization import canonical_json_bytes
from tests.probes.rank_enclosure_probe import FAMILIES


def _record(module, matrix):
    return module.discrete_rank_bracket(
        SimpleNamespace(discrete_rank_algorithm_id=rank.ALGORITHM_ID),
        SimpleNamespace(scale_cells=2),
        SimpleNamespace(depth=0, coordinate_id="snapshot-coordinate"),
        SimpleNamespace(coordinate_id="snapshot-coordinate", structural_rank=2),
        matrix=matrix,
    )


def _matrix_digest(matrix):
    return sha256(canonical_json_bytes({
        "dtype": "float64", "shape": matrix.shape,
        "sha256": sha256(matrix.tobytes(order="C")).hexdigest(),
    })).hexdigest()


@pytest.mark.parametrize("family", tuple(FAMILIES))
@pytest.mark.parametrize("supplied", (False, True))
@pytest.mark.parametrize("mutation_time", ("before", "after"))
def test_family_hash_identifies_certified_values_after_caller_mutation(
    monkeypatch, family, supplied, mutation_time,
):
    module = FAMILIES[family]
    original = np.vstack((np.eye(2), np.zeros((6, 2))))
    caller_matrix = original.copy()
    certify = module.certify_rank

    def certify_then_mutate(matrix, **kwargs):
        if mutation_time == "before":
            caller_matrix.fill(0)
        certificate = certify(matrix, **kwargs)
        if mutation_time == "after":
            caller_matrix.fill(0)
        return certificate

    monkeypatch.setattr(module, "certify_rank", certify_then_mutate)
    monkeypatch.setattr(module, "sampled_history_matrix", lambda *args: caller_matrix)
    record = _record(module, caller_matrix if supplied else None)

    assert not caller_matrix.any()  # Mutation really occurred before record assembly.
    assert (record.lower_rank, record.upper_rank) == (2, 2)
    assert record.matrix_sha256 == _matrix_digest(original)
    assert record.matrix_sha256 != _matrix_digest(caller_matrix)


def test_direct_certificate_detaches_before_validation_can_yield_to_caller(monkeypatch):
    caller_matrix = np.eye(2)
    isfinite = np.isfinite

    def validate_then_mutate(matrix):
        finite = isfinite(matrix)
        caller_matrix.fill(0)
        return finite

    monkeypatch.setattr(rank.np, "isfinite", validate_then_mutate)
    certificate = rank.certify_rank(caller_matrix)

    assert not caller_matrix.any()
    assert (certificate.lower, certificate.upper) == (2, 2)


class _ChangingArray:
    """A conversion source that represents a different matrix on each call."""

    def __init__(self, first, later):
        self.first, self.later = first, later
        self.calls = 0

    def __array__(self, dtype=None, copy=None):
        self.calls += 1
        values = self.first if self.calls == 1 else self.later
        return np.array(values, dtype=dtype, copy=True)


def test_direct_certificate_converts_custom_source_once():
    source = _ChangingArray(np.eye(2), np.zeros((2, 2)))
    certificate = rank.certify_rank(source)
    assert (certificate.lower, certificate.upper) == (2, 2)
    assert source.calls == 1


def test_direct_certificate_refuses_complex_values_returned_by_custom_source():
    source = _ChangingArray(np.eye(2) + 1j, np.eye(2))
    with pytest.raises(ValueError, match="real"):
        rank.certify_rank(source)
    assert source.calls == 1


@pytest.mark.parametrize("family", tuple(FAMILIES))
def test_family_converts_custom_source_once_without_discarding_complex_input(family):
    matrix = np.vstack((np.eye(2), np.zeros((6, 2))))
    source = _ChangingArray(matrix, np.zeros_like(matrix))
    record = _record(FAMILIES[family], source)
    assert (record.lower_rank, record.upper_rank) == (2, 2)
    assert record.matrix_sha256 == _matrix_digest(matrix)
    assert source.calls == 1

    complex_source = _ChangingArray(matrix + 1j, matrix)
    with pytest.raises(ValueError, match="real"):
        _record(FAMILIES[family], complex_source)
    assert complex_source.calls == 1


@pytest.mark.parametrize("family", tuple(FAMILIES))
@pytest.mark.parametrize("invalid", (
    np.ones((7, 2)), np.ones(16), np.empty((0, 2)),
    np.full((8, 2), np.inf), np.full((8, 2), np.nan), np.ones((8, 2), dtype=complex),
))
def test_family_snapshot_preserves_shape_real_and_finite_refusals(family, invalid):
    before = invalid.copy()
    with pytest.raises(ValueError):
        _record(FAMILIES[family], invalid)
    np.testing.assert_array_equal(invalid, before)


@pytest.mark.parametrize("family", tuple(FAMILIES))
def test_noncontiguous_caller_input_remains_unchanged(family):
    backing = np.arange(32, dtype=np.float64).reshape(8, 4)
    matrix = backing[:, ::2]
    before = backing.copy()
    record = _record(FAMILIES[family], matrix)
    assert (record.lower_rank, record.upper_rank) == (2, 2)
    assert record.matrix_sha256 == _matrix_digest(matrix)
    np.testing.assert_array_equal(backing, before)

# SPDX-License-Identifier: MPL-2.0

"""Retained development input joins and missing-data counterexamples, all synthetic."""

import base64
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from empirical_lawhood.adapters.composition.finite_response_law.informative_composition_input import ARRAYS, ROOTS, FiniteResponseLawInformativeCompositionParent, analyze_informative_composition_parent, check_informative_composition_parent
from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec
from empirical_lawhood.kernel.decoding import decode_canonical_bytes


def fixture():
    rng = np.random.default_rng(73281)
    arrays = {
        key: np.zeros(shape, dtype=dtype) for key, (shape, dtype) in ARRAYS.items()
    }
    arrays["x"][:] = rng.normal(size=arrays["x"].shape)
    arrays["z"][:] = rng.normal(size=arrays["z"].shape)
    arrays["observed"][:16] = arrays["valid"][:16] = 1
    arrays["hold_observed"][:16] = 1
    arrays["observed"][16:, :, 2:, :2] = arrays["valid"][16:, :, 2:, :2] = 1
    arrays["y"][16:] = np.nan
    arrays["y"][16:, :, 2:, :2] = 0
    roots = tuple((r, f"synthetic.{r}") for r in ROOTS)
    encoded = tuple(
        (key, base64.b64encode(a.tobytes()).decode()) for key, a in arrays.items()
    )
    return FiniteResponseLawInformativeCompositionParent(FiniteResponseLawScienceSpec(), roots, encoded)


def test_retained_census_masks_and_authenticated_join():
    record = fixture()
    assert (
        decode_canonical_bytes(
            record.canonical_bytes(), FiniteResponseLawInformativeCompositionParent, maximum_bytes=8 * 1024**2
        )
        == record
    )
    panel, _ = record.operands()
    assert panel.observed[:16].all() and not panel.observed[16:, :, :2].any()
    parent = SimpleNamespace(
        raw=record.canonical_bytes(),
        grant=SimpleNamespace(
            parent=SimpleNamespace(
                source_schema=record.SCHEMA, physical_sha256=record.fingerprint()
            )
        ),
        custody=SimpleNamespace(
            original_root_ids=tuple(sorted(r for _, r in record.root_identity_map))
        ),
    )
    result = check_informative_composition_parent(parent, expected_plan_sha256=record.science.plan_sha256)
    assert (result["independent_units"], result["complete_decision_roots"]) == (48, 16)
    parent.custody.original_root_ids = parent.custody.original_root_ids[:-1]
    with pytest.raises(ValueError, match="PARENT_PLAN_OR_ROOTS_MISMATCH"):
        check_informative_composition_parent(parent, expected_plan_sha256=record.science.plan_sha256)
    altered = dict(record.arrays_base64)
    mask = np.frombuffer(base64.b64decode(altered["valid"]), dtype=np.uint8).copy()
    mask[0] = 0
    altered["valid"] = base64.b64encode(mask.tobytes()).decode()
    with pytest.raises(ValueError, match="COMPLETE_TWO_FUTURE_PANEL_REQUIRED"):
        replace(record, arrays_base64=tuple(altered.items()))


def test_retained_owner_keeps_outer_folds_and_refuses_zero_response_use():
    result, arrays = analyze_informative_composition_parent(fixture())
    assert result["fresh_and_preparation_policy_condition"] is False
    assert arrays

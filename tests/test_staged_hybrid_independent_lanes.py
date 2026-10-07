# SPDX-License-Identifier: MPL-2.0
"""The retained staged freeze can represent the owner-scoped two-lane route."""

from __future__ import annotations

from dataclasses import replace

import pytest

from empirical_lawhood.adapters.simulators.staged_hybrid_response import StagedHybridResponseTrancheFreeze
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity


def _identity(label: str) -> ObjectIdentity:
    return ObjectIdentity(
        label, 'empirical-lawhood/test/staged-input', "1.0.0", "a" * 64
    )


def _freeze() -> StagedHybridResponseTrancheFreeze:
    return StagedHybridResponseTrancheFreeze(
        freeze_id='freeze.staged-hybrid-two-lane-development',
        gym_design=_identity("gym.design"),
        gym_qualification_config=_identity("gym.qualification"),
        gym_evaluation_config=_identity("gym.evaluation"),
        pybamm_design=_identity("pybamm.design"),
        pybamm_qualification_config=_identity("pybamm.qualification"),
        pybamm_development_config=_identity("pybamm.development"),
        pybamm_evaluation_config=_identity("pybamm.evaluation"),
        boptest_conditional_design=None,
        formal_register=_identity("formal.register"),
        config_file_sha256=(
            ("gym/config.json", "b" * 64),
            ("pybamm/config.json", "c" * 64),
        ),
        no_cross_lane_learning=True,
        frozen_at_utc="2026-09-29T00:00:00Z",
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )


def test_two_lane_freeze_omits_boptest_and_requires_independent_contexts() -> None:
    freeze = _freeze()
    assert freeze.active_lanes == ("gym-torax", "pybamm")
    with pytest.raises(ValueError, match="cross-lane"):
        replace(freeze, no_cross_lane_learning=False)
    with pytest.raises(ValueError, match="lane identities"):
        replace(freeze, pybamm_design=freeze.gym_design)
    assert replace(
        freeze, boptest_conditional_design=_identity("old.boptest")
    ).active_lanes == ("gym-torax", "pybamm", "boptest")

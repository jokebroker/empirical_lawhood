"""Finite native canary: independent impulse equation and causal boundary."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from empirical_lawhood.adapters.simulators.finite_response_law.native_quickstart import FiniteResponseLawCanary, run_finite_canary
from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_force_components
from empirical_lawhood.api.codecs import load_registered_authoring

CONFIG = Path(__file__).parents[1] / "experiments/finite-response-law/finite-canary.json"


def _config() -> FiniteResponseLawCanary:
    return load_registered_authoring(
        CONFIG,
        root_schemas={FiniteResponseLawCanary.SCHEMA: FiniteResponseLawCanary},
        maximum_bytes=16 * 1024,
    )


def test_shipped_finite_canary_runs_two_futures_under_one_root() -> None:
    result = run_finite_canary(_config())
    assert result["independent_roots"] == 1
    assert result["nested_numerical_views"] == 2
    assert result["development_only"] and not result["campaign_candidate_compiled"]
    assert not result["campaign_issued"]
    for view in result["views"]:
        refinement = view["refinement"]
        assert (
            view["prefix_end_tick"],
            view["parent_end_tick"],
            view["receiver_end_tick"],
        ) == (4096, 4368, 4560)
        assert [future["purpose"] for future in view["futures"]] == [
            "future-1",
            "future-2",
        ]
        for future in view["futures"]:
            assert future["accepted"] is True
            assert future["applied_force_kicks"] == 384 * refinement
            assert future["nonzero_force_intervals"] == 64 * refinement
            # Constant 8 native force for 64 reference ticks of 0.001 time.
            np.testing.assert_allclose(
                [float(v) for v in future["realized_impulse"]],
                [0.512, 0],
                atol=1e-12,
                rtol=0,
            )
            assert [float(v) for v in future["hold_realized_impulse"]] == [0, 0]
            assert np.isfinite(future["paired_receiver"]).all()
        assert (
            view["futures"][0]["paired_receiver"]
            != view["futures"][1]["paired_receiver"]
        )
    for future_index in (0, 1):
        first = result["views"][0]["futures"][future_index]["paired_receiver"]
        second = result["views"][1]["futures"][future_index]["paired_receiver"]
        np.testing.assert_allclose(first, second, atol=1e-4, rtol=0)


def test_finite_canary_causal_cutoff_and_refusals() -> None:
    config = _config()
    word = PreparedForceWord(config.magnitude, config.direction_index, config.sign)
    for refinement in (1, 2):
        cutoff = 4368 * refinement
        assert not prepared_force_components(
            word, native_step=cutoff - 1, invocation_tick=4368, refinement=refinement
        ).any()
        np.testing.assert_array_equal(
            prepared_force_components(
                word, native_step=cutoff, invocation_tick=4368, refinement=refinement
            ),
            [8, 0],
        )
        assert not prepared_force_components(
            word,
            native_step=cutoff + 64 * refinement,
            invocation_tick=4368,
            refinement=refinement,
        ).any()
    with pytest.raises(ValueError):
        replace(config, stage="calibration")
    with pytest.raises(ValueError):
        replace(config, evidence_role="PROSPECTIVE")
    with pytest.raises(ValueError):
        replace(config, magnitude=Decimal(12))
    with pytest.raises(ValueError):
        replace(config, direction_index=2)

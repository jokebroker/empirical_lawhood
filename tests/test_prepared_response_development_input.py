"Shipped prepared response canary, independent impulse equation and causal refusals."

from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from empirical_lawhood.adapters.simulators.prepared_response.contracts import PreparedForceWord
from empirical_lawhood.adapters.simulators.prepared_response.instruments import prepared_force_components
from empirical_lawhood.adapters.simulators.prepared_response.native_quickstart import PreparedResponsePreparedCanary, run_prepared_canary
from empirical_lawhood.api.codecs import load_registered_authoring

CONFIG = Path(__file__).parents[1] / "experiments/prepared-response/prepared-canary.json"


def _config() -> PreparedResponsePreparedCanary:
    return load_registered_authoring(
        CONFIG,
        root_schemas={PreparedResponsePreparedCanary.SCHEMA: PreparedResponsePreparedCanary},
        maximum_bytes=16 * 1024,
    )


def test_shipped_canary_runs_paired_native_hold_action_and_keeps_one_root() -> None:
    config = _config()
    result = run_prepared_canary(config)
    assert result["independent_roots"] == 1
    assert result["nested_numerical_views"] == 2
    assert result["development_only"] and not result["campaign_candidate_compiled"]
    assert not result["campaign_issued"]
    views = result["views"]
    assert len(views) == 2
    # Integral of a constant 8-unit native force over the fixed 64 x .001
    # clock interval; the half view doubles steps but preserves this impulse.
    reference_impulse = float(config.magnitude) * 64 * 0.001
    for view in views:
        assert (
            view["prefix_end_tick"],
            view["parent_end_tick"],
            view["receiver_end_tick"],
        ) == (1024, 1296, 1616)
        assert view["accepted"] is True
        assert view["applied_force_kicks"] == 640 * view["refinement"]
        assert view["nonzero_force_intervals"] == 64 * view["refinement"]
        np.testing.assert_allclose(
            [float(v) for v in view["realized_impulse"]],
            [reference_impulse, 0],
            atol=1e-12,
            rtol=0,
        )
        assert [float(v) for v in view["hold_realized_impulse"]] == [0, 0]
    assert abs(views[0]["paired_receiver"][0] - views[1]["paired_receiver"][0]) < 1e-4


def test_force_is_zero_before_cutoff_and_after_pulse_and_bad_inputs_refuse() -> None:
    config = _config()
    word = PreparedForceWord(config.magnitude, config.direction_index, config.sign)
    for refinement in (1, 2):
        cutoff = 1296 * refinement
        assert not prepared_force_components(
            word, native_step=cutoff - 1, invocation_tick=1296, refinement=refinement
        ).any()
        np.testing.assert_array_equal(
            prepared_force_components(
                word, native_step=cutoff, invocation_tick=1296, refinement=refinement
            ),
            [8, 0],
        )
        assert not prepared_force_components(
            word,
            native_step=cutoff + 64 * refinement,
            invocation_tick=1296,
            refinement=refinement,
        ).any()
    with pytest.raises(ValueError):
        replace(config, stage='qualification')
    with pytest.raises(ValueError):
        replace(config, evidence_role="PROSPECTIVE")
    with pytest.raises(ValueError):
        replace(config, seed_label="old-q-seed")
    with pytest.raises(ValueError):
        replace(config, direction_index=4)
    with pytest.raises(ValueError):
        PreparedForceWord(Decimal(12), 0, 1)

from __future__ import annotations

import numpy as np

from empirical_lawhood.adapters.physical.mastu_freegsnke_response.canary import ACTIVE_LABELS, FRACTIONS, HORIZONS_S, VIEWS, action_roster, select_canary, select_preparation_slice


def _episodes(*, contrast: float, invalid: bool = False) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for view in VIEWS:
        rows.append(
            {
                "view": view,
                "word": "hold",
                "valid": not invalid,
                "pickup_1_t": {
                    "0.000": 1.0,
                    **{format(value, ".3f"): 1.0 for value in HORIZONS_S},
                },
            }
        )
        for fraction in FRACTIONS:
            scale = contrast if fraction == FRACTIONS[0] else contrast * 2
            for sign, name in ((-1, "minus"), (1, "plus")):
                rows.append(
                    {
                        "view": view,
                        "word": f"{name}-{fraction:g}",
                        "valid": not invalid,
                        "pickup_1_t": {
                            "0.000": 1.0,
                            **{format(value, ".3f"): 1.0 + sign * scale for value in HORIZONS_S},
                        },
                    }
                )
    return rows


def test_action_roster_is_smallest_to_largest_with_one_hold() -> None:
    assert ACTIVE_LABELS == (
        "Solenoid",
        "px",
        "d1",
        "d2",
        "d3",
        "dp",
        "d5",
        "d6",
        "d7",
        "p4",
        "p5",
        "p6",
    )


def test_preparation_selection_is_earliest_cross_view_valid_slice() -> None:
    assert (
        select_preparation_slice(
            np.asarray([0.15, 0.25, 0.35]),
            np.asarray([False, True, True]),
            np.asarray([True, True, True]),
        )
        == 1
    )
    assert action_roster(4.0) == (
        ("hold", 0.0),
        ("minus-0.0625", -0.25),
        ("plus-0.0625", 0.25),
        ("minus-0.125", -0.5),
        ("plus-0.125", 0.5),
        ("minus-0.25", -1.0),
        ("plus-0.25", 1.0),
    )


def test_selection_uses_smallest_amplitude_and_earliest_horizon() -> None:
    result = select_canary(_episodes(contrast=0.002))
    assert result["disposition"] == "SUPPORTED"
    assert result["selected_fraction"] == 0.0625
    assert result["selected_horizon_s"] == 0.005


def test_selection_preserves_materiality_and_scientific_stops() -> None:
    assert (
        select_canary(_episodes(contrast=0.001))["disposition"] == "METHOD_MATERIALITY_INCOMPATIBLE"
    )
    assert (
        select_canary(_episodes(contrast=0.002, invalid=True))["disposition"]
        == "SHORT_HORIZON_SOURCE_UNEVALUABLE"
    )

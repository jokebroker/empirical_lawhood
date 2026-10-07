"""response composition's registered post-hoc operator on nonpromotable synthetic scalars."""

from pathlib import Path

import numpy as np

from empirical_lawhood.adapters.composition.response_composition.input import ResponseCompositionInput
from empirical_lawhood.adapters.methods.response_composition.development import COMPOSITION_EPSILON, develop_context
from empirical_lawhood.api.codecs import load_registered_authoring

ROOT = Path(__file__).parents[1]


def test_response_composition_whole_root_nested_folds_and_adverse_numerical_view() -> None:
    selection = load_registered_authoring(
        ROOT / 'experiments/causal-transfer-audit/analysis-input.json',
        root_schemas={ResponseCompositionInput.SCHEMA: ResponseCompositionInput},
        maximum_bytes=16 * 1024,
    )
    assert selection.evidence_role == "EXPOSED_SOURCE_QUALIFICATION_DEVELOPMENT_NONPROMOTABLE"
    features = np.zeros((16, 2, 6, 200), dtype=np.float64)
    response = np.zeros((16, 2, 5, 9, 5, 2), dtype=np.float64)
    preservation = np.ones((16, 5), dtype=np.bool_)
    baseline = develop_context(features, response, preservation)
    logs = baseline["summary"]["selection_log"]
    assert len(logs) == 4
    assert tuple(row["outer_held"] for row in logs) == tuple(
        tuple(range(k, 16, 4)) for k in range(4)
    )
    for row in logs:
        assert set(row["outer_held"]).isdisjoint(row["training_roots"])
        assert len(row["training_roots"]) == 12
        assert len(row["inner_folds"]) == 3
        assert set().union(*map(set, row["inner_folds"])) == set(row["training_roots"])
        assert set(row["scenario_donors"]) == set(row["training_roots"])
    assert baseline["summary"]["qualified_roots"] == 16
    assert baseline["summary"]["design1_passing_roots"] == 16

    # One post-HOLD scalar in the second numerical view is outside the
    # composition precision, while the corresponding nominal view stays zero.
    adverse = response.copy()
    adverse[0, 1, 2, 2, 4, 0] = 0.02
    changed = develop_context(features, adverse, preservation)
    assert changed["summary"]["selection_log"][0] == logs[0]
    np.testing.assert_array_equal(
        changed["lower"][[0, 4, 8, 12]], baseline["lower"][[0, 4, 8, 12]]
    )
    assert changed["summary"]["numerical_response_discrepancies"][0] > (
        COMPOSITION_EPSILON / 8
    )
    assert changed["summary"]["numerical_contrast_discrepancies"][0] > (
        COMPOSITION_EPSILON / 8
    )
    assert changed["summary"]["qualified_roots"] == 15
    assert changed["summary"]["design1_passing_roots"] == 15
    assert changed["summary"]["design1_root_pass"][0] is False

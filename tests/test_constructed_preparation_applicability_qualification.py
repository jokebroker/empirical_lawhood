"""Synthetic constructor counterexamples, never a native Q qualification."""

from decimal import Decimal as D
from types import SimpleNamespace

import numpy as np
import pytest

from empirical_lawhood.adapters.methods.constructed_preparation_applicability.qualification import (
    constructor_crossing,
    qualify_constructor,
)


class _SyntheticLower:
    center = (D(0),) * 24
    scale = (D(1),) * 24

    def normalize(self, handoff: np.ndarray) -> np.ndarray:
        return handoff.copy()


def _measurement(root: str, *, crossing: bool = True):
    handoff = np.zeros((3, 24, 2))
    handoff[0, 17] = 7 if crossing else 5
    handoff[1, 17] = 5
    maxima = np.zeros((3, 7))
    # Q qualification deliberately does not require these six non-numeric gates.
    maxima[:, [0, 2, 3, 4, 5, 6]] = 99
    return SimpleNamespace(
        root_id=root, complete=True,
        handoff=tuple(D(str(value)) for value in handoff.ravel()),
        maxima=tuple(D(str(value)) for value in maxima.ravel()),
    )


def _set_handoff(row, schedule: int, coordinate: int, values) -> None:
    handoff = np.asarray(row.handoff, dtype=float).reshape(3, 24, 2)
    handoff[schedule, coordinate] = values
    row.handoff = tuple(D(str(value)) for value in handoff.ravel())


def _census(crossings: int = 6):
    roots = tuple(f"exposed.synthetic.q.unit-{i}" for i in range(8))
    rows = tuple(_measurement(root, crossing=i < crossings) for i, root in enumerate(roots))
    return roots, rows


def test_exact_six_crossings_qualify_without_additional_q_gates() -> None:
    roots, rows = _census(6)
    result = qualify_constructor(rows, roots, _SyntheticLower())
    assert result.complete and result.qualified and result.disposition == "QUALIFIED"
    assert result.crossings == (True,) * 6 + (False,) * 2
    roots, rows = _census(5)
    result = qualify_constructor(rows, roots, _SyntheticLower())
    assert result.complete and not result.qualified
    assert result.disposition == "CONSTRUCTOR_NOT_QUALIFIED"


@pytest.mark.parametrize(
    "schedule,coordinate,values",
    [
        (0, 17, (7, 5)),  # H must cross in both views.
        (0, 17, (6.25, 6.5)),  # Strict outward margin exceeds disagreement.
        (1, 17, (5.5, 5.75)),  # Strict inward margin exceeds disagreement.
        (1, 23, (0, 7)),  # Every N coordinate is supported in both views.
    ],
)
def test_crossing_rejects_primary_only_and_unresolved_faces(schedule, coordinate, values) -> None:
    row = _measurement("exposed.synthetic.q.root")
    _set_handoff(row, schedule, coordinate, values)
    assert not constructor_crossing(row, _SyntheticLower())


def test_support_face_is_inclusive_for_other_n_coordinates() -> None:
    row = _measurement("exposed.synthetic.q.root")
    _set_handoff(row, 1, 23, (-6, 6))
    assert constructor_crossing(row, _SyntheticLower())


def test_last_root_last_arm_numerical_failure_blocks_qualification() -> None:
    roots, rows = _census(8)
    maxima = np.asarray(rows[-1].maxima, dtype=float).reshape(3, 7)
    maxima[2, 1] = 1.125
    rows[-1].maxima = tuple(D(str(value)) for value in maxima.ravel())
    result = qualify_constructor(rows, roots, _SyntheticLower())
    assert result.complete and not result.qualified and all(result.crossings)


def test_q_numerical_bound_is_inclusive_for_every_arm() -> None:
    roots, rows = _census(6)
    for row in rows:
        maxima = np.asarray(row.maxima, dtype=float).reshape(3, 7)
        maxima[:, 1] = 1
        row.maxima = tuple(D(str(value)) for value in maxima.ravel())
    assert qualify_constructor(rows, roots, _SyntheticLower()).qualified


def test_missing_native_unit_is_unevaluable_before_partial_reduction() -> None:
    roots, rows = _census(8)
    rows[-1].complete = False
    rows[-1].handoff = ()
    rows[-1].maxima = ()
    result = qualify_constructor(rows, roots, _SyntheticLower())
    assert not result.complete and not result.qualified and result.crossings == ()
    assert result.disposition == "UNEVALUABLE"


@pytest.mark.parametrize("change", ["seven", "duplicate", "reordered", "substituted"])
def test_q_readout_requires_exact_declared_complete_census(change) -> None:
    roots, rows = _census(8)
    if change == "seven":
        roots, rows = roots[:-1], rows[:-1]
    elif change == "duplicate":
        roots = roots[:-1] + (roots[0],)
    elif change == "reordered":
        rows = tuple(reversed(rows))
    else:
        rows[-1].root_id = "other-physical-unit"
    with pytest.raises(ValueError, match="assigned Q census"):
        qualify_constructor(rows, roots, _SyntheticLower())

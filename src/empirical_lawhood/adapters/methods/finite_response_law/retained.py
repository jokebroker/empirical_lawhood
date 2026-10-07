"""Typed masked view of authenticated prepared-response/information-response-prediction operands; no source access or fit."""

from dataclasses import dataclass
from typing import Mapping, cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class FiniteResponseLawObservedPanel:
    # Axes: root, parent, positive canonical pair, output, future, numerical view.
    y: FloatArray
    observed: BoolArray
    valid: BoolArray
    parent_work: FloatArray  # root,parent,view
    hold_observed: BoolArray  # root,parent,future,view
    root_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        shape = (48, 5, 4, 8, 2, 2)
        if self.y.shape != shape or self.observed.shape != shape or self.valid.shape != shape:
            raise ValueError("Finite response-law panel changes named axes or 16/32/48 root census")
        if self.observed.dtype != np.bool_ or self.valid.dtype != np.bool_:
            raise ValueError("observed and valid masks must be Boolean")
        if self.parent_work.shape != (48, 5, 2) or self.hold_observed.shape != (48, 5, 2, 2):
            raise ValueError("work/HOLD axes differ")
        expected = tuple(
            f"{c}.prepared.r{r:03d}" for c, n in (("prepared-response", 16), ("information-response-prediction", 32)) for r in range(n)
        )
        if self.root_ids != expected or len(set(self.root_ids)) != 48:
            raise ValueError("root identities or multiplicities differ")
        if np.any(self.valid & (~self.observed | ~np.isfinite(self.y))):
            raise ValueError("valid output must be observed and finite")
        for a in (self.y, self.observed, self.valid, self.parent_work, self.hold_observed):
            a.flags.writeable = False


def _array(
    values: Mapping[str, NDArray[np.generic]], key: str, shape: tuple[int, ...]
) -> FloatArray:
    result = values[key]
    if result.shape != shape or result.dtype != np.float64:
        raise ValueError(f"Wrong {key} axes/dtype; implicit transpose or coercion forbidden")
    return cast(FloatArray, result)


def source_qualification_failure_counts(row: Mapping[str, object]) -> tuple[int, ...]:
    """Read current failure fields after a separately verified historical export."""
    expected = ("off_force_failures", "relative_y_failures", "transfer_failures", "parent_work_failures")
    if not set(expected).issubset(row):
        raise ValueError("PREPARED_FAILURE_SUMMARY_VERIFIED_TARGET_EXPORT_REQUIRED")
    values = tuple(
        row[key]
        for key in expected
    )
    if any(type(x) is not int or x < 0 for x in values):
        raise ValueError("Expected the prepared-response summary's nonnegative integer failure counts")
    return cast(tuple[int, ...], values)


def import_retained(
    source_qualification: Mapping[str, NDArray[np.generic]],
    source_qualification_response: Mapping[str, NDArray[np.generic]],
    information_prediction: Mapping[str, NDArray[np.generic]],
    *,
    lineage_authenticated: bool,
    horizon_ticks: int = 192,
) -> FiniteResponseLawObservedPanel:
    if not lineage_authenticated or horizon_ticks != 192:
        raise ValueError("Unauthenticated lineage or wrong receiver horizon")
    values = _array(source_qualification, "values", (32, 2, 5, 17, 5, 7))[16:]
    normalized_gain = _array(source_qualification_response, "gain", (32, 2, 5, 2, 4, 5, 2))[16:]
    endpoints = _array(information_prediction, "endpoints", (2, 32, 2, 5, 2, 4, 5, 2))[1]
    information_gain = _array(information_prediction, "gains", (2, 32, 2, 5, 2, 5, 2, 2))[1]
    y = np.full((48, 5, 4, 8, 2, 2), np.nan)
    observed = np.zeros(y.shape, dtype=bool)
    for k, (amplitude, direction) in enumerate(((8, 0), (8, 1), (16, 0), (16, 1))):
        negative = 1 + (8 if amplitude == 16 else 0) + 2 * direction
        positive = negative + 1
        response = (values[:, :, :, positive, 2, :2] - values[:, :, :, negative, 2, :2]) / 2
        # prepared-response's gain is per unit magnitude; FLH's response remains in native units.
        np.testing.assert_allclose(
            response,
            amplitude * normalized_gain[:, :, :, amplitude // 8 - 1, direction, 2],
            rtol=0,
            atol=2e-12,
        )
        y[:16, :, k, :2, 0, :] = response.transpose(0, 2, 3, 1)
        y[:16, :, k, 2:5, 0, :] = values[:, :, :, positive, 2, 2:5].transpose(0, 2, 3, 1)
        y[:16, :, k, 5:8, 0, :] = values[:, :, :, negative, 2, 2:5].transpose(0, 2, 3, 1)
        observed[:16, :, k, :, 0, :] = True
    for direction in (0, 1):
        response = (
            endpoints[:, :, :, :, 2 * direction + 1, 2, :]
            - endpoints[:, :, :, :, 2 * direction, 2, :]
        ) / 2
        np.testing.assert_allclose(
            response, information_gain[:, :, :, :, 2, :, direction], rtol=0, atol=2e-12
        )
        y[16:, :, 2 + direction, :2, :, :] = response.transpose(0, 2, 4, 3, 1)
        observed[16:, :, 2 + direction, :2, :, :] = True
    work = np.concatenate(
        (
            _array(source_qualification, "parent_work", (32, 2, 5))[16:].transpose(0, 2, 1),
            _array(information_prediction, "parent_work", (2, 32, 2, 5))[1].transpose(0, 2, 1),
        )
    )
    hold = np.zeros((48, 5, 2, 2), dtype=bool)
    hold[:16, :, 0, :] = True
    return FiniteResponseLawObservedPanel(
        y,
        observed,
        observed & np.isfinite(y),
        work,
        hold,
        tuple(f"{c}.prepared.r{r:03d}" for c, n in (("prepared-response", 16), ("information-response-prediction", 32)) for r in range(n)),
    )

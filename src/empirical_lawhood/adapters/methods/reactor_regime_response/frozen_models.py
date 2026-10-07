"""Lossless canonical custody for frozen empirical coefficient fits.

Each binary64 operand is written as its round-trip decimal representation.
Neither a later normalization nor a qualification label can reconstruct a
different fit under the same model identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from math import isfinite
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id

from .models import Fit, LocalLeaf


def _decimal(value: float | np.floating) -> D:
    native = float(value)
    if not isfinite(native):
        raise ValueError("nonfinite fitted operand cannot be frozen")
    return D(repr(native))


def _array(values: tuple[D, ...]) -> np.ndarray:
    return np.asarray(tuple(float(value) for value in values), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class FrozenLeaf(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/frozen-leaf'

    path: tuple[tuple[int, D, bool], ...]
    operator: tuple[D, ...]

    def __post_init__(self) -> None:
        if not self.path or len(self.path) > 2 or not self.operator:
            raise ValueError("local leaf has no bounded path or operator")
        if any(type(index) is not int or index < 0 or type(left) is not bool or not threshold.is_finite()
               for index, threshold, left in self.path):
            raise ValueError("local split is nonfinite or malformed")
        if any(not value.is_finite() for value in self.operator):
            raise ValueError("local leaf coefficient is nonfinite")


@dataclass(frozen=True, slots=True)
class FrozenFit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-regime-response/frozen-fit'

    fit_id: str
    kind: str
    mean: tuple[D, ...]
    scale: tuple[D, ...]
    operator: tuple[D, ...]
    penalty: D
    length: D | None
    training: tuple[tuple[D, ...], ...]
    leaves: tuple[FrozenLeaf, ...]
    support_lower: tuple[D, ...]
    support_upper: tuple[D, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.fit_id, field_name="fit_id")
        dimension = len(self.mean)
        if (
            self.kind not in ("K", "affine", "rbf", "local")
            or dimension == 0
            or len(self.scale) != dimension
            or len(self.support_lower) != dimension
            or len(self.support_upper) != dimension
            or not self.penalty.is_finite()
            or self.penalty < 0
        ):
            raise ValueError("frozen model structure or support dimension differs")
        values = (
            *self.mean,
            *self.scale,
            *self.operator,
            *self.support_lower,
            *self.support_upper,
            *(value for row in self.training for value in row),
        )
        if any(not value.is_finite() for value in values) or any(value <= 0 for value in self.scale):
            raise ValueError("frozen model has nonfinite values or nonpositive scale")
        if any(lo > hi for lo, hi in zip(self.support_lower, self.support_upper, strict=True)):
            raise ValueError("frozen support box is inverted")
        if self.kind == "K":
            if len(self.operator) != 1 or self.training or self.leaves or self.length is not None:
                raise ValueError("frozen constant model shape differs")
        elif self.kind == "affine":
            if len(self.operator) != dimension + 1 or self.training or self.leaves or self.length is not None:
                raise ValueError("frozen affine model shape differs")
        elif self.kind == "rbf":
            if (
                self.length is None
                or not self.length.is_finite()
                or self.length <= 0
                or not self.training
                or len(self.operator) != len(self.training)
                or any(len(row) != dimension for row in self.training)
                or self.leaves
            ):
                raise ValueError("frozen RBF model shape differs")
        elif (
            self.operator
            or self.training
            or self.length is not None
            or not 2 <= len(self.leaves) <= 3
            or any(len(leaf.operator) != dimension + 1 for leaf in self.leaves)
            or any(index >= dimension for leaf in self.leaves for index, _, _ in leaf.path)
        ):
            raise ValueError("frozen local model shape differs")

    @classmethod
    def from_fit(
        cls,
        fit_id: str,
        fit: Fit,
        support_lower: np.ndarray,
        support_upper: np.ndarray,
    ) -> FrozenFit:
        return cls(
            fit_id,
            fit.kind,
            tuple(map(_decimal, fit.mean)),
            tuple(map(_decimal, fit.scale)),
            tuple(map(_decimal, fit.operator)),
            _decimal(fit.penalty),
            None if fit.length is None else _decimal(fit.length),
            () if fit.training is None else tuple(tuple(map(_decimal, row)) for row in fit.training),
            tuple(
                FrozenLeaf(
                    tuple((index, _decimal(threshold), left) for index, threshold, left in leaf.path),
                    tuple(map(_decimal, leaf.operator)),
                )
                for leaf in fit.leaves
            ),
            tuple(map(_decimal, support_lower)),
            tuple(map(_decimal, support_upper)),
        )

    def to_fit(self) -> Fit:
        return Fit(
            self.kind,  # type: ignore[arg-type]
            _array(self.mean),
            _array(self.scale),
            _array(self.operator),
            float(self.penalty),
            None if self.length is None else float(self.length),
            None if not self.training else np.asarray(
                tuple(tuple(float(value) for value in row) for row in self.training),
                dtype=np.float64,
            ),
            tuple(
                LocalLeaf(
                    tuple((index, float(threshold), left) for index, threshold, left in leaf.path),
                    _array(leaf.operator),
                )
                for leaf in self.leaves
            ),
        )

    def contains(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2 or features.shape[1] != len(self.mean):
            raise ValueError("frozen support input dimension differs")
        return np.asarray(
            ((features >= _array(self.support_lower)) & (features <= _array(self.support_upper))).all(axis=1),
            dtype=bool,
        )

"""Saved batched ridge operators and excluded-root uncertainty operands."""

from typing import Any

from dataclasses import dataclass, field
import numpy as np
from ..fitting import DELTA, Normalizer
from empirical_lawhood.adapters.methods.response_formalization import affine_prediction, fit_affine_operator


@dataclass
class FitStore:
    arrays: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    designs: dict[Any, Any] = field(default_factory=dict)
    solves: int = 0

    def fit(self, key: Any, x: Any, targets: Any, train: Any, *, excluded: Any, kind: Any) -> Any:
        train = np.asarray(train, dtype=np.int64)
        excluded = np.asarray(excluded, dtype=np.int64)
        if np.intersect1d(train, excluded).size or not len(train):
            raise ValueError("Empty or leaking fit dependency")
        cache_key = (x.shape[1], tuple(train))
        if cache_key not in self.designs:
            norm = Normalizer.fit(x[train])
            design = np.column_stack((norm.apply(x[train]), np.ones(len(train))))
            # The installed ridge owner solves all root basis RHS together once.
            # Applying this linear map shares the factorization across all targets.
            solver = fit_affine_operator(norm.apply(x[train]), np.eye(len(train)), ridge=10)
            self.designs[cache_key] = (norm, design, solver)
        norm, design, solver = self.designs[cache_key]
        target = np.asarray(targets, dtype=np.float64).reshape(len(train), -1)
        if not np.isfinite(target).all():
            raise ValueError("Empty/nonfinite required regression population")
        op = solver @ target
        self.solves += 1
        self.save(key, norm, op, train, excluded, kind, target)
        return affine_prediction(op, norm.apply(x))

    def save(
        self, key: Any, norm: Any, op: Any, train: Any, excluded: Any, kind: Any, target: Any
    ) -> None:
        for suffix, value in (
            ("center", norm.center),
            ("scale", norm.scale),
            ("operator", op),
            ("train", train),
            ("excluded", excluded),
            ("target", target),
        ):
            self.arrays[f"fit.{key}.{suffix}"] = np.asarray(value)
        self.records.append({"key": key, "kind": kind, "inputs": len(norm.center)})


def modal_operands(delta: Any, kernel: Any) -> Any:
    selected = kernel[1:, 2:]
    scale = np.maximum(np.sqrt(np.mean(selected**2, axis=(0, 1))), 1e-12)
    design = (selected / scale).reshape(-1, 4)
    target = delta[:, 1:, 2:].transpose(1, 2, 0, 3).reshape(-1, 24 * 24)
    weights, _, rank, _ = np.linalg.lstsq(design, target, rcond=1e-10)
    if rank != 4:
        raise ValueError("Radial modal rank differs")
    return weights.reshape(4, 24, 24).transpose(1, 0, 2).reshape(24, 96), scale, design, target


def radial_prediction(weights: Any, kernel: Any, scale: Any, lower_scale: Any) -> Any:
    return np.einsum("sk,rkj->rsj", kernel[:, -1] / scale, weights.reshape(-1, 4, 24)) * lower_scale


def scales(
    store: Any,
    key: Any,
    x: Any,
    train: Any,
    excluded: Any,
    mean: Any,
    mhat: Any,
    y: Any,
    margin: Any,
) -> Any:
    residual = y - mean[..., None, None]
    # Required retained panels are complete; missing score cases are handled separately.
    if not np.isfinite(residual).all() or not np.isfinite(margin).all():
        raise ValueError("UNEVALUABLE: incomplete required scale population")
    base = np.maximum(np.sqrt(np.mean(residual[..., 0] ** 2, axis=(0, 1, 4))), DELTA / 20)
    response_target = np.log(
        np.maximum(
            0.25, (abs(residual) / base[None, None, :, :, None, None]).max(axis=(2, 3, 4, 5))
        )
    )
    error = mhat - margin
    margin_base = max(float(np.sqrt(np.mean(error**2))), 1e-8)
    margin_target = np.log(np.maximum(0.25, np.maximum(error, 0) / margin_base))
    targets = np.column_stack((response_target, margin_target))
    logg = store.fit(key, x, targets, train, excluded=excluded, kind="scale")
    mult = np.exp(np.clip(logg, np.log(0.25), np.log(4)))
    store.arrays[f"{key}.response_base"] = base
    store.arrays[f"{key}.margin_base"] = np.asarray(margin_base)
    return base[None, None] * mult[:, :9, None, None], margin_base * mult[:, 9:]

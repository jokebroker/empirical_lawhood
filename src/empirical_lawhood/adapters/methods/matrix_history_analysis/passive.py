"""Origin-only causal seals and complete-operator evaluator diagnostics."""

from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from scipy.linalg import expm

from .algebra import operator

CAUSAL_PREDICTORS = ("frozen", "radius", "nochange", "twoband")
EVALUATOR_ORACLES = ("endpoint_oracle", "integrated_future_oracle")


@dataclass(frozen=True, slots=True)
class PassiveForecastSeal:
    """Immutable copied origin-only forecast. No future operand is accepted."""
    origin_operator_sha256: str
    kappa: float
    horizon: float
    predictions: tuple[np.ndarray, ...]
    slow: np.ndarray
    fast: np.ndarray
    sha256: str


def _immutable(value):
    return np.frombuffer(np.asarray(value, dtype="<f8").tobytes(), dtype="<f8").reshape(value.shape)


def seal_causal_forecast(origin_y, *, kappa, horizon):
    if not np.isfinite((kappa, horizon)).all() or kappa <= 0 or horizon <= 0:
        raise ValueError("Passive causal seal needs positive finite kappa/horizon")
    initial = operator(origin_y)
    rates, vectors = np.linalg.eigh(initial)
    slow, fast = vectors[:, :3], vectors[:, 3:]
    projector = slow @ slow.T
    twoband = rates[:3].mean() * projector + rates[3:].mean() * (np.eye(15) - projector)
    predictions = tuple(_immutable(p) for p in (
        expm(-kappa * horizon * initial),
        np.eye(15) * np.exp(-kappa * horizon * rates.mean()),
        np.eye(15), expm(-kappa * horizon * twoband),
    ))
    origin_sha = sha256(initial.astype("<f8").tobytes()).hexdigest()
    digest = sha256(bytes.fromhex(origin_sha) + np.asarray((kappa, horizon), dtype="<f8").tobytes() + b"".join(p.tobytes() for p in predictions)).hexdigest()
    return PassiveForecastSeal(origin_sha, float(kappa), float(horizon), predictions, _immutable(slow), _immutable(fast), digest)


def propagators(y_path, dt, kappas):
    """Full15-dimensional RK4 with midpoint in Y and Simpson integral."""
    path = np.asarray(y_path)
    kappas = np.asarray(kappas, dtype=np.float64)
    if path.ndim != 4 or path.shape[1:] != (3, 4, 4) or len(path) < 2 or not np.isfinite(dt) or dt <= 0 or kappas.ndim != 1 or not len(kappas) or not np.isfinite(kappas).all() or (kappas <= 0).any():
        raise ValueError("Passive trajectory/operator census differs")
    current = np.repeat(np.eye(15)[None], len(kappas), axis=0)
    integral = np.zeros((15, 15))
    rates = [operator(path[0])]
    outputs, integrals = [current.copy()], [integral.copy()]
    scale = kappas[:, None, None]
    for left, right in zip(path[:-1], path[1:], strict=True):
        a, b, c = rates[-1], operator((left + right) / 2), operator(right)
        k1 = -scale * (a @ current)
        k2 = -scale * (b @ (current + dt * k1 / 2))
        k3 = -scale * (b @ (current + dt * k2 / 2))
        k4 = -scale * (c @ (current + dt * k3))
        current = current + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        integral = integral + dt * (a + 4 * b + c) / 6
        rates.append(c)
        outputs.append(current.copy())
        integrals.append(integral.copy())
    if not np.isfinite(outputs).all():
        raise FloatingPointError("Passive full operator propagation became nonfinite")
    return np.asarray(outputs), np.asarray(rates), np.asarray(integrals)


def evaluate_forecast(actual, *, seal, initial_operator, endpoint_operator, integrated_operator):
    """Evaluator-only actual directions, future oracles and rank-resolved losses."""
    values = tuple(np.asarray(v, dtype=np.float64) for v in (actual, initial_operator, endpoint_operator, integrated_operator))
    if any(v.shape != (15, 15) or not np.isfinite(v).all() for v in values):
        raise ValueError("Passive evaluator requires complete finite15×15 operators")
    actual, initial, endpoint, integrated = values
    if sha256(initial.astype("<f8").tobytes()).hexdigest() != seal.origin_operator_sha256:
        raise ValueError("Passive evaluator substitutes the sealed origin operator")
    delta = actual - np.eye(15)
    inverse = np.linalg.pinv(delta, rcond=1e-12)
    predictions = (*seal.predictions, expm(-seal.kappa * seal.horizon * endpoint), expm(-seal.kappa * integrated))
    metrics = {}
    for name, prediction in zip((*CAUSAL_PREDICTORS, *EVALUATOR_ORACLES), predictions, strict=True):
        error = prediction - actual
        metrics[f"{name}.full"] = float(np.linalg.norm(error)**2 / max(np.linalg.norm(delta)**2, 1e-30))
        metrics[f"{name}.worst"] = float(np.linalg.norm(error @ inverse, ord=2)**2)
        for label, basis in (("slow", seal.slow), ("fast", seal.fast)):
            metrics[f"{name}.{label}"] = float(np.linalg.norm(error @ basis)**2 / max(np.linalg.norm(delta @ basis)**2, 1e-30))
    increment_singular = np.linalg.svd(delta, compute_uv=False)
    survival_singular = np.linalg.svd(actual, compute_uv=False)
    rank = int(np.linalg.matrix_rank(delta, tol=1e-12))
    metrics.update({"slow_fast_leakage": float(np.linalg.norm(seal.fast.T @ actual @ seal.slow)**2 / 3),
                    "generator_drift": float(np.linalg.norm(endpoint - initial) / max(np.linalg.norm(initial), 1e-30)),
                    "endpoint_commutator": float(np.linalg.norm(initial @ endpoint - endpoint @ initial) / max(np.linalg.norm(initial) * np.linalg.norm(endpoint), 1e-30))})
    return {"metrics": metrics, "increment_singular_values": increment_singular,
            "survival_singular_values": survival_singular, "increment_rank_absolute": rank,
            "increment_unresolved": rank < 15, "predictions": np.asarray(predictions)}

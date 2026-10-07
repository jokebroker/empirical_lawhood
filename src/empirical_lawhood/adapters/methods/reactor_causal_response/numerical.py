"Frozen causal feature/interval/selector arithmetic, also shipped verbatim.\n\nThis is a numerical policy, not admission, commitment, qualification or controller use.\nNo outcome, scenario, native state or mechanistic response model is an input.\n"

from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any
import numpy as np

EXPLORATION_EPISODES = ('exploration-unshifted', 'exploration-shifted-one-position', 'exploration-shifted-two-positions')
EXPLORATION_ARRAY_STEMS = ('exploration_unshifted', 'exploration_shifted_one_position', 'exploration_shifted_two_positions')

COLUMNS = (
    "clock",
    "dose",
    "window",
    "previous-feed",
    "previous-jacket",
    "candidate-feed",
    "candidate-jacket",
    "temperature",
    "jacket",
    "temperature-square",
    "temperature-jacket",
    "temperature-feed",
    "temperature-action-jacket",
    "temperature-difference-60",
    "temperature-difference-300",
    "jacket-difference-60",
    "jacket-difference-300",
    "feed-mean-60",
    "feed-mean-300",
    "temperature-excess-integral",
    "thermal-gap-integral",
    "rate-feed",
    "rate-jacket",
)
DIMENSIONS = (7, 13, 23)
LAMBDAS = (0.000001, 0.001, 0.1)


def digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Observation:
    time: float
    temperature: float
    jacket: float
    dose: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(x) for x in (self.time, self.temperature, self.jacket, self.dose)):
            raise ValueError("nonfinite native observation")
        if self.time % 10 or not 0 <= self.time < 28800 or self.dose < 0:
            raise ValueError("invalid callback clock/dose")


@dataclass(frozen=True)
class Projection:
    action: int
    requested: tuple[float, float]
    accepted: tuple[float, float]
    applied: tuple[float, float]
    # Each sample is (absolute start, duration, feed, jacket); never average for delivery.
    exposure: tuple[tuple[float, float, float, float], ...]
    next_dose: float

    @property
    def mean_feed(self) -> float:
        return sum(dt * feed for _, dt, feed, _ in self.exposure) / 10

    @property
    def delivery_key(self) -> tuple[tuple[float, float, float, float], ...]:
        return self.exposure


def menu(previous_jacket: float) -> tuple[tuple[float, float], ...]:
    return tuple(
        (f, min(347.6, max(279.3, previous_jacket + delta)))
        for f in (0.0, 0.016, 0.032)
        for delta in (-2.3, 0.0, 2.3)
    )


def features(
    history: tuple[Observation, ...], previous_jacket: float, candidate: Projection
) -> np.ndarray:
    if not history or tuple(x.time for x in history) != tuple(
        float(10 * k) for k in range(len(history))
    ):
        raise ValueError("history must contain each available callback from zero")
    k = len(history) - 1
    current = history[-1]
    dose = np.array([x.dose for x in history])
    feeds = np.diff(dose) / 10
    if np.any(feeds < 0):
        raise ValueError("dose decreased")
    y, j = current.temperature, current.jacket
    u, v = (y - 318.4) / 40, (j - 316) / 40
    result = [
        current.time / 28800,
        current.dose / 287.3,
        float(600 <= current.time < 25200),
        float(feeds[-1]) / 0.032 if k else 0.0,
        (previous_jacket - 316) / 40,
        candidate.mean_feed / 0.032,
        (candidate.applied[1] - 316) / 40,
        u,
        v,
        u * u,
        u * v,
        u * candidate.mean_feed / 0.032,
        u * (candidate.applied[1] - 316) / 40,
    ]
    result.extend((y - history[max(0, k - n)].temperature) / 40 for n in (6, 30))
    result.extend((j - history[max(0, k - n)].jacket) / 40 for n in (6, 30))
    result.extend(float(np.sum(feeds[max(0, k - n) :])) / n / 0.032 for n in (6, 30))
    result.extend(
        (
            sum(max(o.temperature - 330, 0) * 10 for o in history[:-1]) / (40 * 28800),
            sum((o.temperature - o.jacket) * 10 for o in history[:-1]) / (40 * 28800),
        )
    )
    result.extend((result[13] * result[5], result[13] * result[6]))
    return np.asarray(result, dtype=np.float64)


def stratum(time: float) -> int:
    if not math.isfinite(time) or not 0 <= time < 28800:
        raise ValueError("clock outside law horizon")
    return 0 if time < 600 else 1 if time < 25200 else 2


class CausalFeatures:
    """Incremental implementation of the declared left-constant history.

    Prefix hashing has exactly the canonical JSON-list digest used by features
    callers. Integrals retain the original chronological addition order; lag
    means retain NumPy's original bounded-window reduction arithmetic.
    """

    def __init__(self) -> None:
        self.observations: list[Observation] = []
        self.feeds: list[float] = []
        self.excess = 0.0
        self.gap = 0.0
        self._prefix = hashlib.sha256(b"[")
        self._base: np.ndarray | None = None

    def append(self, observation: Observation) -> None:
        k = len(self.observations)
        if observation.time != k * 10:
            raise ValueError("missing/reordered causal callback")
        if k:
            prior = self.observations[-1]
            if observation.dose < prior.dose:
                raise ValueError("dose decreased")
            self.feeds.append((observation.dose - prior.dose) / 10)
            self.excess += max(prior.temperature - 330, 0) * 10
            self.gap += (prior.temperature - prior.jacket) * 10
            self._prefix.update(b",")
        self._prefix.update(
            json.dumps(
                (observation.time, observation.temperature, observation.jacket, observation.dose),
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        )
        self.observations.append(observation)
        self._base = None

    @property
    def history_digest(self) -> str:
        result = self._prefix.copy()
        result.update(b"]")
        return result.hexdigest()

    def features(self, previous_jacket: float, candidate: Projection) -> np.ndarray:
        if not self.observations:
            raise ValueError("no causal observation")
        if self._base is None:
            h, k = self.observations, len(self.observations) - 1
            o = h[-1]
            u, v = (o.temperature - 318.4) / 40, (o.jacket - 316) / 40
            x = np.zeros(23, dtype=np.float64)
            x[:5] = (
                o.time / 28800,
                o.dose / 287.3,
                float(600 <= o.time < 25200),
                self.feeds[-1] / 0.032 if k else 0.0,
                0.0,
            )
            x[7:11] = (u, v, u * u, u * v)
            x[13:17] = tuple(
                (o.temperature - h[max(0, k - n)].temperature) / 40 for n in (6, 30)
            ) + tuple((o.jacket - h[max(0, k - n)].jacket) / 40 for n in (6, 30))
            x[17:19] = tuple(
                float(np.sum(self.feeds[max(0, k - n) :])) / n / 0.032 for n in (6, 30)
            )
            x[19:21] = (self.excess / (40 * 28800), self.gap / (40 * 28800))
            self._base = x
        x = self._base.copy()
        x[4:7] = (
            (previous_jacket - 316) / 40,
            candidate.mean_feed / 0.032,
            (candidate.applied[1] - 316) / 40,
        )
        # Preserve multiplication/division order of the reference feature code.
        x[11:13] = (x[7] * candidate.mean_feed / 0.032, x[7] * (candidate.applied[1] - 316) / 40)
        x[21:23] = (x[13] * x[5], x[13] * x[6])
        return x


@dataclass(frozen=True)
class SupportCell:
    clock_stratum: int
    action: int
    roots: tuple[str, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.clock_stratum) is not int
            or self.clock_stratum not in range(3)
            or type(self.action) is not int
            or self.action not in range(9)
        ):
            raise ValueError("undeclared support coordinate")
        if len(self.lower) != len(self.upper) or len(set(self.roots)) != len(self.roots):
            raise ValueError("invalid support axes")
        if any(
            not math.isfinite(a) or not math.isfinite(b) or a > b
            for a, b in zip(self.lower, self.upper, strict=True)
        ):
            raise ValueError("invalid support box")

    def contains(self, x: np.ndarray) -> bool:
        return (
            len(self.roots) >= 8
            and len(set(self.roots)) == len(self.roots)
            and x.shape == (len(self.lower),)
            and bool(np.isfinite(x).all())
            and bool(np.all(x >= self.lower) and np.all(x <= self.upper))
        )


@dataclass(frozen=True)
class FrozenFit:
    family: int
    penalty: float
    mean: tuple[float, ...]
    scale: tuple[float, ...]
    operator: tuple[tuple[float, float], ...]
    support: tuple[SupportCell, ...]
    training_digest: str
    rows: int
    columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            type(self.family) is not int
            or self.family not in (0, 1, 2)
            or self.penalty not in LAMBDAS
        ):
            raise ValueError("undeclared family/penalty")
        if (
            type(self.rows) is not int
            or self.rows <= 0
            or len(self.training_digest) != 64
            or any(c not in "0123456789abcdef" for c in self.training_digest)
        ):
            raise ValueError("invalid training identity/count")
        d = DIMENSIONS[self.family]
        if not self.columns:
            object.__setattr__(self, "columns", COLUMNS[:d])
        if self.columns != COLUMNS[:d]:
            raise ValueError("saved feature order differs")
        if len(self.mean) != d or len(self.scale) != d or len(self.operator) != d + 1:
            raise ValueError("intercept-last model dimensions differ")
        if any(len(row) != 2 for row in self.operator) or any(s <= 0 for s in self.scale):
            raise ValueError("invalid model shape/normalization")
        if not all(
            math.isfinite(v)
            for v in (*self.mean, *self.scale, *(v for row in self.operator for v in row))
        ):
            raise ValueError("nonfinite fit")
        keys = [(c.clock_stratum, c.action) for c in self.support]
        if len(set(keys)) != len(keys) or any(
            len(c.lower) != d or len(c.upper) != d for c in self.support
        ):
            raise ValueError("support cells differ")

    def predict(self, x: np.ndarray) -> np.ndarray:
        d = DIMENSIONS[self.family]
        z = (np.asarray(x)[..., :d] - self.mean) / self.scale
        return np.asarray(z @ np.asarray(self.operator[:-1]) + self.operator[-1], dtype=np.float64)

    def supported(self, x: np.ndarray, time: float, action: int) -> bool:
        return any(
            c.clock_stratum == stratum(time)
            and c.action == action
            and c.contains(x[: DIMENSIONS[self.family]])
            for c in self.support
        )

    def support_mask(self, x: np.ndarray, clocks: np.ndarray, actions: np.ndarray) -> np.ndarray:
        values = x[:, : DIMENSIONS[self.family]]
        strata = np.where(clocks < 600, 0, np.where(clocks < 25200, 1, 2))
        result = np.zeros(len(values), dtype=bool)
        for cell in self.support:
            if len(set(cell.roots)) < 8:
                continue
            mask = (strata == cell.clock_stratum) & (actions == cell.action)
            result[mask] = np.all(
                (values[mask] >= cell.lower) & (values[mask] <= cell.upper), axis=1
            )
        return np.asarray(
            result & np.isfinite(values).all(axis=1) & (clocks >= 0) & (clocks < 28800), dtype=bool
        )


@dataclass(frozen=True)
class Candidate:
    projection: Projection
    features: tuple[float, ...]
    point: tuple[float, float]
    halfwidth: tuple[float, float]
    reasons: tuple[str, ...]
    aliases: tuple[int, ...]

    @property
    def robust(self) -> float:
        return self.point[1] - self.halfwidth[1] + self.projection.next_dose / 287.3

    @property
    def nominal(self) -> float:
        return self.point[1] + self.projection.next_dose / 287.3


@dataclass(frozen=True)
class Decision:
    selected: int | None
    candidates: tuple[Candidate, ...]
    history_digest: str
    reasons: tuple[str, ...]


def select(
    history: tuple[Observation, ...],
    previous_jacket: float,
    projections: tuple[Projection, ...],
    model: FrozenFit,
    q: float,
    gate_reasons: tuple[tuple[str, ...], ...] = ((),) * 9,
    *,
    causal_state: CausalFeatures | None = None,
    temperature_halfwidth: float | None = None,
) -> Decision:
    if tuple(p.action for p in projections) != tuple(range(9)) or len(gate_reasons) != 9:
        raise ValueError("complete nine-word census required")
    if not math.isfinite(q) or q < 0:
        raise ValueError("invalid calibrated q")
    if temperature_halfwidth not in (None, 0.25):
        raise ValueError("undeclared diagnostic margin")
    width = (
        q * 0.25 + 0.01 if temperature_halfwidth is None else temperature_halfwidth,
        q * 0.01 + 0.0002,
    )
    candidates = []
    for p, gates in zip(projections, gate_reasons, strict=True):
        x = (
            features(history, previous_jacket, p)
            if causal_state is None
            else causal_state.features(previous_jacket, p)
        )
        point = tuple(float(v) for v in model.predict(x))
        reasons = list(gates)
        if not model.supported(x, history[-1].time, p.action):
            reasons.append("OUTSIDE_SUPPORT")
        if not all(math.isfinite(v) for v in point):
            reasons.append("NONFINITE_PREDICTION")
        if point[0] + width[0] > 356.2:
            reasons.append("PEAK_TEMPERATURE_BOUND")
        if width[0] > 0.26 or width[1] > 0.0102:
            reasons.append("PRECISION")
        aliases = tuple(v.action for v in projections if v.delivery_key == p.delivery_key)
        candidates.append(
            Candidate(p, tuple(x), (point[0], point[1]), width, tuple(reasons), aliases)
        )
    eligible = [c for c in candidates if not c.reasons and c.projection.action == min(c.aliases)]
    chosen = (
        min(eligible, key=lambda c: (-c.robust, -c.nominal, c.projection.action))
        if eligible
        else None
    )
    return Decision(
        None if chosen is None else chosen.projection.action,
        tuple(candidates),
        (
            digest([(o.time, o.temperature, o.jacket, o.dose) for o in history])
            if causal_state is None
            else causal_state.history_digest
        ),
        () if chosen else ("NONATTEMPT",),
    )


def exploration(
    history: tuple[Observation, ...], previous_jacket: float, root: int, policy: int
) -> int:
    if root not in range(24) or policy not in range(3):
        raise ValueError("undeclared exploration root/policy")
    o = history[-1]
    feed_index = (root + policy + int(o.time // 120)) % 3
    target = (340, 350, 355)[(root + policy + int(o.time // 2400)) % 3]
    rate = (o.temperature - history[max(0, len(history) - 2)].temperature) / 10
    jacket = min(347.6, max(279.3, 316 + 8 * (target - o.temperature) - 120 * rate))
    words = menu(previous_jacket)
    j = min(range(3), key=lambda i: (abs(words[i][1] - jacket), i))
    return 3 * feed_index + j


def branch_tape(
    tape: tuple[tuple[float, float], ...], root: int, branch: str, anchor_previous_jacket: float
) -> tuple[tuple[float, float], ...]:
    if len(tape) != 2880 or branch not in ("feed-intervention", "jacket-intervention") or root not in range(24):
        raise ValueError("complete prefix-response donor and declared branch required")
    k = (3600, 9000, 18000, 24000)[root % 4] // 10
    out = list(tape)
    f, j = out[k]
    extremes = menu(anchor_previous_jacket)
    out[k] = (
        (0.032 if f == 0 else 0.0, j)
        if branch == "feed-intervention"
        else (f, extremes[2][1] if j == extremes[0][1] else extremes[0][1])
    )
    return tuple(out)

"""Independent native operands for each finite coordinate, without law verdicts."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.simulators.reactor_prefix_response.finite_words import measure_feed_delivery
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import COORDINATES, CONTEXTS, PULSES, GUARD, Coordinate, FrontierDesign, Pulse, ZERO, assignment
from .records import FrontierAssay, FrontierPreparation


@dataclass(frozen=True, slots=True)
class FrontierObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-observation'
    coordinate: Coordinate
    contact: bool
    evaluable: bool
    delivery_valid: bool
    guard_numerical_valid: bool
    numerical_valid: bool
    prefix_safe: bool
    action_safe: bool
    effects_K: tuple[D, ...]
    action_peaks_K: tuple[D, ...]
    reference_peaks_K: tuple[D, ...]
    applied_masses_kg: tuple[D, ...]
    realized_masses_kg: tuple[D, ...]
    observed_temperature_K: D | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        values = (
            self.effects_K,
            self.action_peaks_K,
            self.reference_peaks_K,
            self.applied_masses_kg,
            self.realized_masses_kg,
        )
        if (
            len({len(v) for v in values}) != 1
            or len(self.effects_K) not in (0, 2)
            or any(type(x) is not D or not x.is_finite() for row in values for x in row)
            or (not self.contact and any(values))
            or (self.contact and self.evaluable and not self.effects_K)
            or (not self.contact and not self.reasons)
        ):
            raise ValueError("frontier observation changes contact or paired-view operands")


@dataclass(frozen=True, slots=True)
class FrontierGuard(CanonicalRecord):
    """One branch's guard, independent of other words and response windows."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-guard'
    context: str
    pulse: Pulse
    valid: bool
    peaks_K: tuple[D, ...]
    observed_temperature_K: D | None
    unsafe: bool


@dataclass(frozen=True, slots=True)
class FrontierMeasuredRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-measured-root'
    root: str
    preparation: ObjectIdentity
    assay: ObjectIdentity
    observations: tuple[FrontierObservation, ...]
    guards: tuple[FrontierGuard, ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        if (
            tuple(o.coordinate for o in self.observations) != COORDINATES
            or self.preparation.object_schema != FrontierPreparation.SCHEMA
            or self.assay.object_schema != FrontierAssay.SCHEMA
            or tuple((g.context, g.pulse) for g in self.guards)
            != tuple((c, w) for c in CONTEXTS for w in (ZERO, *PULSES))
        ):
            raise ValueError("measurement lost its complete coordinate/receipt census")


def _delivery(
    data: dict[str, np.ndarray],
    key: str,
    anchor: np.ndarray,
    coordinate: Coordinate,
    view: int,
    *,
    reference: bool,
) -> tuple[bool, float, float]:
    """Replay pure actuator stages against every actual command and exposure."""
    pulse = ZERO if reference else coordinate.pulse
    dt = 0.5 if view else 1.0
    requested, stages, exposure = (
        data[f"{key}_{field}"] for field in ("requests", "stages", "exposure")
    )
    return measure_feed_delivery(
        anchor,
        requested,
        stages,
        exposure,
        tuple(pulse.request(i) for i in range(0, GUARD, 10)),
        dt,
    )


def measure_coordinate(
    preparation: FrontierPreparation,
    coordinate: Coordinate,
    data: dict[str, np.ndarray],
) -> FrontierObservation:
    context = next(c for c in preparation.contexts if c.context == coordinate.context)
    callback = context.callback
    empty = ((), (), (), (), ())
    if callback is None or context.reasons:
        return FrontierObservation(
            coordinate,
            False,
            not any("INCOMPLETE" in r or r.startswith("VIEW_") for r in preparation.reasons),
            False,
            False,
            False,
            False,
            False,
            *empty,
            None,
            context.reasons or ("NO_CAUSAL_CONTACT",),
        )
    observed = D(repr(float(context.arrays.unpack()["observations"][callback, 1])))
    required = {
        f"{coordinate.context}_{pulse.word_id}_v{view}_{field}"
        for pulse in (ZERO, coordinate.pulse)
        for view in (0, 1)
        for field in (
            "grid",
            "requests",
            "stages",
            "exposure",
            "valid",
            "prefix_sha256",
            "full_grid_sha256",
        )
    }
    required |= {
        f"{coordinate.context}_v{view}_{field}"
        for view in (0, 1)
        for field in ("anchor", "prefix_peak")
    }
    if not required <= data.keys() or any(not np.isfinite(data[k]).all() for k in required):
        return FrontierObservation(
            coordinate,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            *empty,
            observed,
            ("INCOMPLETE_NATIVE_MEASUREMENT",),
        )
    effects, peaks, zeros, applied, realized = [], [], [], [], []
    reasons: set[str] = set()
    delivery = True
    prefix_safe = True
    limit = float(FrontierDesign().temperature_limit_K)
    tau_peaks = []
    for view in (0, 1):
        dt = 0.5 if view else 1.0
        prefix = f"{coordinate.context}_v{view}"
        anchor, prefix_peak = data[f"{prefix}_anchor"], data[f"{prefix}_prefix_peak"]
        if anchor.shape != (6,) or anchor[0] != callback * 10 or prefix_peak.shape != (1,):
            return FrontierObservation(
                coordinate,
                True,
                False,
                False,
                False,
                False,
                False,
                False,
                *empty,
                observed,
                ("INVALID_ANCHOR_MEASUREMENT",),
            )
        prefix_safe &= prefix_peak[0] <= limit
        grids = []
        keys = []
        for pulse in (ZERO, coordinate.pulse):
            key = f"{coordinate.context}_{pulse.word_id}_v{view}"
            keys.append(key)
            grid = data[f"{key}_grid"]
            if grid.shape != (int(GUARD / dt) + 1, 8) or not np.array_equal(
                grid[:, 0], callback * 10 + np.arange(len(grid)) * dt
            ):
                return FrontierObservation(
                    coordinate,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    *empty,
                    observed,
                    ("INVALID_NATIVE_GRID",),
                )
            if (
                data[f"{key}_requests"].shape != (12, 2)
                or data[f"{key}_stages"].shape != (12, 4)
                or data[f"{key}_exposure"].shape != (12, int(10 / dt), 4)
                or data[f"{key}_prefix_sha256"].shape != (32,)
                or data[f"{key}_full_grid_sha256"].shape != (32,)
            ):
                return FrontierObservation(
                    coordinate,
                    True,
                    False,
                    False,
                    False,
                    False,
                    False,
                    False,
                    *empty,
                    observed,
                    ("INVALID_NATIVE_DELIVERY_SHAPE",),
                )
            valid, mass, actual = _delivery(
                data, key, anchor, coordinate, view, reference=pulse == ZERO
            )
            delivery &= (
                valid and data[f"{key}_valid"].shape == (1,) and bool(data[f"{key}_valid"].all())
            )
            exposure = data[f"{key}_exposure"].reshape(-1, 4)
            delivery &= (
                exposure.shape == (len(grid) - 1, 4)
                and np.array_equal(exposure[:, 0] + exposure[:, 1], grid[1:, 0])
                and np.array_equal(exposure[:, 2:], grid[1:, 6:])
            )
            if pulse != ZERO:
                applied.append(D(repr(mass)))
                realized.append(D(repr(actual)))
            grids.append(grid)
        delivery &= np.array_equal(
            data[f"{keys[0]}_prefix_sha256"], data[f"{keys[1]}_prefix_sha256"]
        )
        local = tuple(float(grid[: int(coordinate.horizon_s / dt) + 1, 1].max()) for grid in grids)
        tau_peaks.append(local)
        effects.append(D(repr(local[0] - local[1])))
        zeros.append(D(repr(float(grids[0][:, 1].max()))))
        peaks.append(D(repr(float(grids[1][:, 1].max()))))
    guard_numerical = (
        max(abs(peaks[0] - peaks[1]), abs(zeros[0] - zeros[1])) <= D(".01")
        and abs(applied[0] - applied[1]) <= D("1e-10")
        and abs(realized[0] - realized[1]) <= D("1e-10")
    )
    numerical = (
        guard_numerical
        and max(D(repr(abs(tau_peaks[0][i] - tau_peaks[1][i]))) for i in (0, 1)) <= D(".01")
        and abs(effects[0] - effects[1]) <= coordinate.epsilon_K
    )
    safe = max(peaks) <= D("356.2")
    for condition, reason in (
        (delivery, "WRONG_NATIVE_DELIVERY_OR_PREFIX"),
        (numerical, "NUMERICAL_VIEW_MISMATCH"),
        (prefix_safe, "UNSAFE_PREPARATION_PREFIX"),
        (safe, "UNSAFE_ACTION_WINDOW"),
    ):
        if not condition:
            reasons.add(reason)
    return FrontierObservation(
        coordinate,
        True,
        True,
        bool(delivery),
        bool(guard_numerical),
        bool(numerical),
        bool(prefix_safe),
        bool(safe),
        tuple(effects),
        tuple(peaks),
        tuple(zeros),
        tuple(applied),
        tuple(realized),
        observed,
        tuple(sorted(reasons)),
    )


def measure_guard(
    preparation: FrontierPreparation, context: str, pulse: Pulse, data: dict[str, np.ndarray]
) -> FrontierGuard:
    ca = next(c for c in preparation.contexts if c.context == context)
    if ca.callback is None or ca.reasons:
        return FrontierGuard(context, pulse, False, (), None, False)
    observed = D(repr(float(ca.arrays.unpack()["observations"][-1, 1])))
    peaks, masses, actuals = [], [], []
    delivery, unsafe = True, False
    # The coordinate only supplies the pulse to the pure actuator checker.
    coordinate = Coordinate(
        context, pulse if pulse != ZERO else PULSES[0], max(10, pulse.duration_s)
    )
    try:
        for view in (0, 1):
            dt = 0.5 if view else 1.0
            key = f"{context}_{pulse.word_id}_v{view}"
            grid, anchor = data[f"{key}_grid"], data[f"{context}_v{view}_anchor"]
            prefix = data[f"{context}_v{view}_prefix_peak"]
            if (
                grid.shape != (int(GUARD / dt) + 1, 8)
                or anchor.shape != (6,)
                or prefix.shape != (1,)
            ):
                raise ValueError("incomplete guard")
            if not all(np.isfinite(a).all() for a in (grid, anchor, prefix)) or not np.array_equal(
                grid[:, 0], ca.callback * 10 + np.arange(len(grid)) * dt
            ):
                raise ValueError("invalid guard")
            peak = D(repr(float(grid[:, 1].max())))
            peaks.append(peak)
            unsafe |= peak > D("356.2") or prefix[0] > 356.2
            valid, mass, actual = _delivery(
                data, key, anchor, coordinate, view, reference=pulse == ZERO
            )
            exposure = data[f"{key}_exposure"].reshape(-1, 4)
            delivery &= (
                valid
                and data[f"{key}_valid"].shape == (1,)
                and bool(data[f"{key}_valid"].all())
                and np.array_equal(exposure[:, 0] + exposure[:, 1], grid[1:, 0])
                and np.array_equal(exposure[:, 2:], grid[1:, 6:])
            )
            masses.append(D(repr(mass)))
            actuals.append(D(repr(actual)))
    except (KeyError, ValueError, IndexError):
        return FrontierGuard(context, pulse, False, tuple(peaks), observed, bool(unsafe))
    valid = (
        delivery
        and abs(peaks[0] - peaks[1]) <= D(".01")
        and abs(masses[0] - masses[1]) <= D("1e-10")
        and abs(actuals[0] - actuals[1]) <= D("1e-10")
    )
    return FrontierGuard(context, pulse, bool(valid), tuple(peaks), observed, bool(unsafe))


def measure_root(
    preparation: FrontierPreparation, assay: FrontierAssay
) -> FrontierMeasuredRoot:
    parent = ObjectIdentity.from_record(preparation.record_id, preparation)
    if assay.root != preparation.root or assay.preparation != parent:
        raise ValueError("measurement changed its receipt-bound causal parent")
    data = assay.arrays.unpack()
    return FrontierMeasuredRoot(
        preparation.root,
        parent,
        ObjectIdentity.from_record(f"{assay.root}.frontier-assay", assay),
        tuple(measure_coordinate(preparation, c, data) for c in COORDINATES),
        tuple(measure_guard(preparation, c, w, data) for c in CONTEXTS for w in (ZERO, *PULSES)),
    )

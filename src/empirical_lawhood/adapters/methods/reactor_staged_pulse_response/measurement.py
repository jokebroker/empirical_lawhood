"""Native response operands, with the fixed receiver and independent-root census."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.reactor_finite_control_frontier.measurement import FrontierMeasuredRoot
from empirical_lawhood.adapters.simulators.reactor_prefix_response.finite_words import measure_feed_delivery
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import Coordinate, JOINT, LOCAL_BASE, LOCAL_EXPANDED, ZERO, assignment
from .records import ClassicalAssay, ClassicalContext, ClassicalNativeWindow, ClassicalPreparation


def receiver_ids(c: Coordinate) -> tuple[str, ...]:
    return (
        ("c1", "c2", "g", "s", "s2")
        if c.kind == "joint"
        else ("c1", "s")
        if c.kind in ("first", "baseline")
        else ("c", "s")
    )


def epsilon(c: Coordinate, receiver: str) -> D:
    return (
        D(".01") if receiver.startswith("s") else D(".000024") if receiver == "g" else c.epsilon_K
    )


def decimal(value: float) -> D:
    return D(repr(float(value)))


@dataclass(frozen=True, slots=True)
class ClassicalObservation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-observation'
    coordinate: Coordinate
    contact: bool
    evaluable: bool
    delivery_valid: bool
    numerical_valid: bool
    unsafe: bool
    values: tuple[tuple[NamedDecimal, ...], ...]
    applied_masses_kg: tuple[D, ...]
    realized_masses_kg: tuple[D, ...]
    observed_temperature_K: D | None
    second_temperature_K: D | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.values) not in (0, 2)
            or len(self.applied_masses_kg) != len(self.values)
            or len(self.realized_masses_kg) != len(self.values)
            or any(
                tuple(v.value_id for v in row) != receiver_ids(self.coordinate)
                or any(v.unit != "K" for v in row)
                for row in self.values
            )
            or (self.contact and self.evaluable and not self.values)
            or (not self.contact and self.values)
            or any(
                type(v) is not D or not v.is_finite()
                for v in (*self.applied_masses_kg, *self.realized_masses_kg)
            )
            or (not self.evaluable and not self.reasons)
        ):
            raise ValueError("response operands changed the exact paired-view receiver census")

    @property
    def valid(self) -> bool:
        return self.contact and self.evaluable and self.delivery_valid and self.numerical_valid

    def pair(self, receiver: str) -> tuple[D, ...]:
        if receiver not in receiver_ids(self.coordinate):
            raise ValueError("undeclared receiver")
        return tuple(next(v.value for v in row if v.value_id == receiver) for row in self.values)


@dataclass(frozen=True, slots=True)
class ClassicalMeasuredRoot(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-measured-root'
    root: str
    block: str
    preparation: ObjectIdentity
    assay: ObjectIdentity
    observations: tuple[ClassicalObservation, ...]

    def __post_init__(self) -> None:
        assignment(self.root)
        if (
            self.block not in ("base-menu-comparison", "expanded-menu-comparison", "staged-sequence-comparison")
            or tuple(o.coordinate for o in self.observations)
            != {"base-menu-comparison": LOCAL_BASE, "expanded-menu-comparison": LOCAL_EXPANDED, "staged-sequence-comparison": JOINT}[self.block]
        ):
            raise ValueError("measurement changes its assigned coordinate denominator")

    @property
    def record_id(self) -> str:
        return f"{self.root}.measurement"


def retained_observation(old: FrontierMeasuredRoot, c: Coordinate) -> ClassicalObservation:
    if c not in LOCAL_BASE and c != JOINT[0]:
        raise ValueError("retained evidence does not measure this staged-pulse relation")
    row = next(
        o
        for o in old.observations
        if (
            o.coordinate.context,
            o.coordinate.horizon_s,
            o.coordinate.pulse.rate_kg_s,
            o.coordinate.pulse.duration_s,
        )
        == (c.context, c.horizon_s, c.pulse.rate_kg_s, c.pulse.duration_s)
    )
    key = "c1" if c.kind == "first" else "c"
    values = tuple(
        (NamedDecimal(key, effect, "K"), NamedDecimal("s", peak, "K"))
        for effect, peak in zip(row.effects_K, row.action_peaks_K, strict=True)
    )
    unsafe = any(
        g.unsafe for g in old.guards if g.context == c.context and g.pulse == row.coordinate.pulse
    ) or (row.contact and row.evaluable and (not row.prefix_safe or not row.action_safe))
    # Compatibility preserves all original raw/numerical tests; it does not add
    # the old zero-reference thermal admission veto to the staged-pulse action contract.
    return ClassicalObservation(
        c,
        row.contact,
        row.evaluable,
        row.delivery_valid,
        row.numerical_valid,
        unsafe,
        values,
        row.applied_masses_kg,
        row.realized_masses_kg,
        row.observed_temperature_K,
        None,
        row.reasons,
    )


def _window(window: ClassicalNativeWindow) -> tuple[dict[str, np.ndarray], bool, D, D] | None:
    data = window.arrays.unpack()
    dt = 0.5 if window.refined else 1.0
    count = window.duration_s // 10
    shapes = {
        "grid": (int(window.duration_s / dt) + 1, 8),
        "anchor": (6,),
        "prefix_peak": (1,),
        "valid": (1,),
        "requests": (count, 2),
        "stages": (count, 4),
        "exposure": (count, int(10 / dt), 4),
        "prefix_sha256": (32,),
        "full_grid_sha256": (32,),
    }
    if any(
        k not in data or data[k].shape != shape or not np.isfinite(data[k]).all()
        for k, shape in shapes.items()
    ):
        return None
    grid = data["grid"]
    if not np.array_equal(grid[:, 0], float(window.start_s) + np.arange(len(grid)) * dt) or data[
        "anchor"
    ][0] != float(window.start_s):
        return None
    valid, mass, realized = measure_feed_delivery(
        data["anchor"],
        data["requests"],
        data["stages"],
        data["exposure"],
        window.requested_rates,
        dt,
    )
    exposure = data["exposure"].reshape(-1, 4)
    valid &= (
        bool(data["valid"].all())
        and np.array_equal(exposure[:, 0] + exposure[:, 1], grid[1:, 0])
        and np.array_equal(exposure[:, 2:], grid[1:, 6:])
    )
    return data, valid, decimal(mass), decimal(realized)


def measure_coordinate(
    preparation: ClassicalPreparation, assay: ClassicalAssay, c: Coordinate
) -> ClassicalObservation:
    return measure_windows(
        preparation, assay.windows, c, induced=assay.induced[0] if assay.induced else None
    )


def measure_windows(
    preparation: ClassicalPreparation,
    windows: tuple[ClassicalNativeWindow, ...],
    c: Coordinate,
    *,
    induced: ClassicalContext | None = None,
) -> ClassicalObservation:
    context = next(
        x
        for x in preparation.contexts
        if x.context == ("early" if c.kind != "local" else c.context)
    )
    observed = (
        None
        if context.callback is None
        else decimal(context.arrays.unpack()["observations"][-1, 1])
    )

    def absent(
        reason: str, *, contact: bool = True, evaluable: bool = False
    ) -> ClassicalObservation:
        return ClassicalObservation(
            c, contact, evaluable, False, False, False, (), (), (), observed, None, (reason,)
        )

    if context.callback is None or context.reasons:
        unknown = any("INCOMPLETE" in r or r.startswith("VIEW_") for r in preparation.reasons)
        return absent("NO_CAUSAL_CONTACT", contact=False, evaluable=not unknown)
    branch_ids = (
        (f"{c.context}.{ZERO.word_id}", f"{c.context}.{c.pulse.word_id}")
        if c.kind == "local"
        else ("00", "a0", f"aw.{c.pulse.word_id}")
        if c.kind == "joint"
        else ("00", "a0")
    )
    indexed = {(w.branch_id, w.refined): w for w in windows}
    values, masses, actual, peaks = [], [], [], []
    valid = True
    unsafe = False
    second_temperature = None
    for view in (False, True):
        raw = [indexed.get((name, view)) for name in branch_ids]
        decoded = [None if w is None else _window(w) for w in raw]
        if any(v is None for v in decoded):
            return absent("INCOMPLETE_NATIVE_MEASUREMENT")
        data = [v[0] for v in decoded if v is not None]
        valid &= all(v[1] for v in decoded if v is not None)
        valid &= all(np.array_equal(d["prefix_sha256"], data[0]["prefix_sha256"]) for d in data)
        dt = 0.5 if view else 1.0

        def peak(index: int, start: int, stop: int) -> float:
            return float(data[index]["grid"][int(start / dt) : int(stop / dt) + 1, 1].max())

        tau = c.horizon_s
        first = peak(0, 0, tau) - peak(len(data) - 1, 0, tau)
        guard = c.episode_guard_s
        action = len(data) - 1
        response = {
            "c" if c.kind == "local" else "c1": decimal(first),
            "s": decimal(peak(action, 0, guard)),
        }
        if c.kind == "joint":
            # All Aw branches share the same actual A prefix. A0 is a paired
            # counterfactual only for the conditional second response.
            boundary = int(120 / dt) + 1
            valid &= np.array_equal(data[1]["grid"][:boundary], data[2]["grid"][:boundary])
            valid &= np.array_equal(data[1]["requests"][:12], data[2]["requests"][:12])
            valid &= np.array_equal(data[1]["stages"][:12], data[2]["stages"][:12])
            response.update(
                c2=decimal(peak(1, 120, 130) - peak(2, 120, 130)),
                g=decimal(peak(0, 0, 240) - peak(2, 0, 240)),
                s2=decimal(peak(2, 120, 240)),
            )
            if not view:
                if induced is None:
                    return absent("MISSING_ACTUAL_INDUCED_CONTEXT")
                second_temperature = decimal(induced.arrays.unpack()["observations"][-1, 1])
        selected = decoded[action]
        assert selected is not None
        mass, realized = selected[2:]
        if c.kind == "first":
            # A0 is acquired over 240 s, but the unconditional first contract
            # ends at 120 s. Its zero suffix adds no dose after that boundary.
            mass = sum((decimal(x) * 10 for x in data[1]["stages"][:12, 2]), D(0))
            exp = data[1]["exposure"][:12].reshape(-1, 4)
            realized = decimal(float(np.sum(exp[:, 1] * exp[:, 2])))
        masses.append(mass)
        actual.append(realized)
        unsafe |= float(data[action]["prefix_peak"][0]) > 356.2 or response["s"] > D("356.2")
        values.append(tuple(NamedDecimal(k, response[k], "K") for k in receiver_ids(c)))
        peaks.append(tuple(peak(i, 0, guard) for i in range(len(data))))
    numerical = (
        all(abs(a.value - b.value) <= epsilon(c, a.value_id) for a, b in zip(*values, strict=True))
        and all(abs(decimal(a) - decimal(b)) <= D(".01") for a, b in zip(*peaks, strict=True))
        and abs(masses[0] - masses[1]) <= D("1e-12")
        and abs(actual[0] - actual[1]) <= D("1e-12")
    )
    valid &= all(abs(a - b) <= D("1e-12") for a, b in zip(masses, actual, strict=True))
    reasons = tuple(
        r
        for condition, r in (
            (valid, "WRONG_NATIVE_DELIVERY_OR_PREFIX"),
            (numerical, "NUMERICAL_VIEW_MISMATCH"),
            (not unsafe, "NATIVE_UNSAFE"),
        )
        if not condition
    )
    return ClassicalObservation(
        c,
        True,
        True,
        valid,
        numerical,
        unsafe,
        tuple(values),
        tuple(masses),
        tuple(actual),
        observed,
        second_temperature,
        reasons,
    )


def measure_root(
    preparation: ClassicalPreparation,
    assay: ClassicalAssay,
    *,
    retained: FrontierMeasuredRoot | None = None,
) -> ClassicalMeasuredRoot:
    parent = ObjectIdentity.from_record(preparation.record_id, preparation)
    if (
        assay.root != preparation.root
        or assay.preparation != parent
        or (retained is not None and retained.root != preparation.root)
    ):
        raise ValueError("measurement replaces its actual root or native parent")
    coordinates = {"base-menu-comparison": LOCAL_BASE, "expanded-menu-comparison": LOCAL_EXPANDED, "staged-sequence-comparison": JOINT}[preparation.block]
    return ClassicalMeasuredRoot(
        preparation.root,
        preparation.block,
        parent,
        ObjectIdentity.from_record(assay.record_id, assay),
        tuple(
            retained_observation(retained, c)
            if retained is not None and c in LOCAL_BASE
            else measure_coordinate(preparation, assay, c)
            for c in coordinates
        ),
    )

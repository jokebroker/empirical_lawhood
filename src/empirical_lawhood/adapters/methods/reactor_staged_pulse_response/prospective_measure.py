"""Measure each actual owner branch separately from its paired references."""

from dataclasses import dataclass, replace
from decimal import Decimal as D
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalArrayPayload
from empirical_lawhood.adapters.simulators.reactor_staged_pulse_response.prospective import ClassicalNativeRoot, ClassicalNativeStage, ClassicalNativeUse
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.controller_runtime import TickDisposition
from .config import ZERO
from .measurement import ClassicalObservation, _window, measure_windows
from .records import ClassicalNativeWindow, ClassicalPreparation


def slice_window(
    window: ClassicalNativeWindow, start: int, duration: int, branch: str
) -> ClassicalNativeWindow:
    if start not in (0, 120) or duration not in (120, 240) or start + duration > window.duration_s:
        raise ValueError("measurement slice changes its declared native guard")
    values = window.arrays.unpack()
    dt = 0.5 if window.refined else 1.0
    if values:
        first, stop = int(start / dt), int((start + duration) / dt) + 1
        original = values["grid"]
        k, n = start // 10, duration // 10
        values["grid"] = original[first:stop].copy()
        values["prefix_peak"] = np.asarray(
            [max(float(values["prefix_peak"][0]), float(original[: first + 1, 1].max()))]
        )
        if start:
            if "observations" not in values or len(values["observations"]) <= k:
                values = {}
            else:
                values["anchor"] = np.asarray(
                    (*values["observations"][k], *values["stages"][k - 1, 2:])
                )
        if values:
            for name in ("requests", "stages", "exposure"):
                values[name] = values[name][k : k + n].copy()
            if "observations" in values:
                values["observations"] = values["observations"][k : k + n + 1].copy()
    return replace(
        window,
        branch_id=branch,
        start_s=window.start_s + start,
        duration_s=duration,
        requested_rates=window.requested_rates[start // 10 : (start + duration) // 10],
        arrays=LocalArrayPayload.pack(values),
        failure=window.failure if values else "MISSING_NATIVE_SLICE",
    )


def reference_window(
    native: ClassicalNativeRoot, stage: ClassicalNativeStage, refined: bool
) -> ClassicalNativeWindow | None:
    frozen = stage.unpack()
    key = (
        f"{frozen.request.context}.{ZERO.word_id}"
        if frozen.stage_kind == "local"
        else "a0"
        if frozen.stage_kind == "joint"
        else "00"
    )
    window = next(
        (w for w in native.references if w.branch_id == key and w.refined is refined), None
    )
    if window is None:
        return None
    return slice_window(
        window,
        120 if frozen.stage_kind == "joint" else 0,
        240 if frozen.stage_kind == "baseline" else 120,
        key,
    )


def actual_window(
    use: ClassicalNativeUse, stage: ClassicalNativeStage, refined: bool
) -> ClassicalNativeWindow | None:
    kind = stage.unpack().stage_kind
    key = "actual.first" if kind == "first" else "actual.episode"
    return next((w for w in use.windows if w.branch_id == key and w.refined is refined), None)


@dataclass(frozen=True, slots=True)
class ClassicalMeasuredStage(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-measured-stage'
    frozen: ObjectIdentity
    observation: ClassicalObservation | None
    owner_delivered: bool
    parent_peaks_K: tuple[D, ...]
    reference_known: bool
    reference_valid: bool


@dataclass(frozen=True, slots=True)
class ClassicalMeasuredProspective(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-measured-prospective'
    root: str
    native: ObjectIdentity
    stages: tuple[ClassicalMeasuredStage, ...]

    @property
    def record_id(self) -> str:
        return f"{self.root}.prospective-measurement"


def measure_prospective(
    preparation: ClassicalPreparation, native: ClassicalNativeRoot
) -> ClassicalMeasuredProspective:
    if preparation.root != native.root:
        raise ValueError("measurement substitutes its assigned preparation")
    measurements = []
    for use in native.uses:
        for stage in use.stages:
            frozen = stage.unpack()
            if not frozen.admitted:
                references = tuple(reference_window(native, stage, view) for view in (False, True))
                decoded_references = tuple(None if r is None else _window(r) for r in references)
                known = all(r is not None for r in decoded_references)
                measurements.append(
                    ClassicalMeasuredStage(
                        stage.frozen.subject,
                        None,
                        False,
                        (),
                        known,
                        known and all(r is not None and r[1] for r in decoded_references),
                    )
                )
                continue
            assert frozen.predicted is not None and frozen.projection is not None
            coordinate = frozen.predicted.coordinate
            windows = list(native.references)
            kind = frozen.stage_kind
            action_id = (
                f"{coordinate.context}.{coordinate.pulse.word_id}"
                if kind == "local"
                else f"aw.{coordinate.pulse.word_id}"
                if kind == "joint"
                else "a0"
            )
            # Aliasing a measured owner window to its named response operand
            # never fills a missing view with a separately acquired action.
            windows = [w for w in windows if w.branch_id != action_id]
            for w in use.windows:
                if w.branch_id == ("actual.first" if kind == "first" else "actual.episode"):
                    windows.append(replace(w, branch_id=action_id))
            row = measure_windows(
                preparation,
                tuple(windows),
                coordinate,
                induced=frozen.causal if kind == "joint" else None,
            )
            parents = []
            for view in (False, True):
                raw = actual_window(use, stage, view)
                decoded_actual = None if raw is None else _window(raw)
                if decoded_actual is not None:
                    grid = decoded_actual[0]["grid"]
                    prefix = D(repr(float(decoded_actual[0]["prefix_peak"][0])))
                    if kind == "joint":
                        dt = 0.5 if view else 1.0
                        prefix = max(prefix, D(repr(float(grid[: int(120 / dt) + 1, 1].max()))))
                    parents.append(prefix)
            refs = tuple(reference_window(native, stage, view) for view in (False, True))
            decoded = tuple(None if r is None else _window(r) for r in refs)
            reference_known = all(v is not None for v in decoded)
            reference_valid = reference_known and all(v is not None and v[1] for v in decoded)
            delivered = (
                stage.tick is not None
                and stage.tick.disposition is TickDisposition.ACTION_DELIVERED
                and stage.tick.delivery_trace.exact
                and len(stage.deliveries) == frozen.projection.guard_s // 10
            )
            raw_nominal = actual_window(use, stage, False)
            if delivered and raw_nominal is not None:
                if kind == "joint":
                    raw_nominal = slice_window(raw_nominal, 120, 120, "actual.suffix")
                data = raw_nominal.arrays.unpack()
                delivered &= all(n in data for n in ("requests", "stages", "exposure"))
                for i, delivery in enumerate(stage.deliveries):
                    if not delivered:
                        break
                    delivered &= (
                        np.array_equal(
                            data["requests"][i],
                            [float(delivery.command.feed_kg_s), float(delivery.command.jacket_k)],
                        )
                        and np.array_equal(
                            data["stages"][i],
                            [
                                float(v)
                                for v in (
                                    delivery.accepted_feed_kg_s,
                                    delivery.accepted_jacket_k,
                                    delivery.applied_feed_kg_s,
                                    delivery.applied_jacket_k,
                                )
                            ],
                        )
                        and np.array_equal(
                            data["exposure"][i],
                            [
                                [
                                    float(v)
                                    for v in (e.time_s, e.duration_s, e.feed_kg_s, e.jacket_k)
                                ]
                                for e in delivery.exposures
                            ],
                        )
                    )
            else:
                delivered = False
            measurements.append(
                ClassicalMeasuredStage(
                    stage.frozen.subject,
                    row,
                    bool(delivered),
                    tuple(parents),
                    reference_known,
                    bool(reference_valid),
                )
            )
    return ClassicalMeasuredProspective(
        native.root, ObjectIdentity.from_record(native.record_id, native), tuple(measurements)
    )

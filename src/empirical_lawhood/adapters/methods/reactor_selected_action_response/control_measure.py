"""Reveal-only local cooling operands from the complete D native continuations."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal as D
from math import isfinite
from typing import ClassVar

import numpy as np

from empirical_lawhood.adapters.simulators.reactor_causal_response.acquisition import NativeEpisode
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.acquisition import donor_episode
from empirical_lawhood.adapters.simulators.reactor_regime_response.panel import FEED_WORDS, assay_tape
from empirical_lawhood.adapters.simulators.reactor_selected_action_response.prospective_records import ClassicalNativeBranch, ClassicalNativeRoot
from empirical_lawhood.runtime.controller_runtime import TickDisposition
from empirical_lawhood.kernel.serialization import CanonicalRecord

from empirical_lawhood.adapters.methods.reactor_regime_response.control_math import MeasuredWord
from empirical_lawhood.adapters.methods.reactor_regime_response.causal_preparation import causal_delivery_prefix
from .records import ClassicalDecision
from .records import ClassicalCausal, ClassicalPrivate


@dataclass(frozen=True, slots=True)
class ClassicalMeasuredWord(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-measured-word'

    word: int
    view: int
    measured_peak_K: D
    measured_cooling_K: D
    grid_temperature_K: tuple[D, ...]
    delivered_mass_kg: D
    dose_valid: bool
    window_valid: bool
    receipt_valid: bool
    observation_valid: bool

    def __post_init__(self) -> None:
        if len(self.grid_temperature_K) != (11 if self.view == 0 else 21):
            raise ValueError("D measured word loses its full ten-second numerical window")
        self.to_measured()

    @classmethod
    def from_measured(cls, value: MeasuredWord) -> ClassicalMeasuredWord:
        return cls(
            value.word,
            value.view,
            D(repr(value.measured_peak_K)),
            D(repr(value.measured_cooling_K)),
            tuple(D(repr(item)) for item in value.grid_temperature_K),
            D(repr(value.delivered_mass_kg)),
            value.dose_valid,
            value.window_valid,
            value.receipt_valid,
            value.observation_valid,
        )

    def to_measured(self) -> MeasuredWord:
        return MeasuredWord(
            self.word,
            self.view,
            float(self.measured_peak_K),
            float(self.measured_cooling_K),
            tuple(float(item) for item in self.grid_temperature_K),
            float(self.delivered_mass_kg),
            self.dose_valid,
            self.window_valid,
            self.receipt_valid,
            self.observation_valid,
        )


@dataclass(frozen=True, slots=True)
class ClassicalMeasuredOperands(CanonicalRecord):
    """Four evaluator slots and one paired owner slot; None means missing."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-measured-operands'

    chart: tuple[ClassicalMeasuredWord | None, ...]
    owner: tuple[tuple[ClassicalMeasuredWord, ClassicalMeasuredWord] | None, ...]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            len(self.chart) != 4
            or len(self.owner) != 1
            or any(
                row is not None and (row.word, row.view) != (index // 2, index % 2)
                for index, row in enumerate(self.chart)
            )
            or any(
                pair is not None and tuple(row.view for row in pair) != (0, 1)
                for pair in self.owner
            )
        ):
            raise ValueError("D measured operands changed their six/4 slot census")


def _window(
    *,
    branch: ClassicalNativeBranch,
    base: NativeEpisode,
    callback: int,
    feed: float,
    zero_peak: float | None,
    owner_receipt_valid: bool,
) -> MeasuredWord | None:
    episode = branch.episode()
    if (
        episode is None
        or zero_peak is None
        or len(episode.exposure) <= callback
        or base.requests.shape != (2880, 2)
        or len(base.observations) <= callback
        or len(base.stages) < callback
    ):
        return None
    view = branch.view
    step = int(10 / episode.dt)
    grid = episode.grid[callback * step : (callback + 1) * step + 1]
    exposure = episode.exposure[callback]
    if (
        episode.root != base.root
        or episode.dt != base.dt
        or grid.shape != (step + 1, 8)
        or exposure.shape != (step, 4)
        or not np.isfinite(grid).all()
        or not np.isfinite(exposure).all()
        or not np.array_equal(grid[:, 0], callback * 10 + np.arange(step + 1) * episode.dt)
        or not np.array_equal(
            episode.observations[: callback + 1], base.observations[: callback + 1]
        )
        or not np.array_equal(
            episode.requests[: callback + 1], assay_tape(base, callback, feed)[: callback + 1]
        )
        or not causal_delivery_prefix(
            episode.observations,
            episode.requests,
            episode.stages,
            episode.exposure,
            callback + 1,
            episode.dt,
        )
        or np.any(exposure[:, 2] < 0)
        or not np.all(exposure[:, 3] == base.stages[callback - 1, 3])
        or (feed == 0 and np.any(exposure[:, 2] != 0))
    ):
        return None
    temperatures = tuple(float(value) for value in grid[:, 1])
    peak = max(temperatures)
    mass = float(np.sum(exposure[:, 1] * exposure[:, 2]))
    cooling = 0.0 if feed == 0 else zero_peak - peak
    if not all(isfinite(value) for value in (peak, cooling, mass)):
        return None
    return MeasuredWord(
        FEED_WORDS.index(feed),
        view,
        peak,
        cooling,
        temperatures,
        mass,
        True,
        True,
        owner_receipt_valid,
        True,
    )


def measure_native_root(
    *,
    native: ClassicalNativeRoot,
    assignment: ClassicalDecision,
    causal: ClassicalCausal,
    private: ClassicalPrivate,
) -> ClassicalMeasuredOperands:
    """Validate entire native arrays before deriving each paired cooling operand.

    A missing or invalid branch stays missing. No surviving view may stand in
    for its paired view, and the evaluator zero branch is the only reference.
    """
    if (
        native.root != assignment.root
        or native.assignment.object_fingerprint != assignment.fingerprint()
        or causal.root != native.root
        or private.root != native.root
    ):
        raise ValueError("D reveal changed an assigned native root or preparation")
    decision = assignment
    if decision is None or not decision.causal_preparation_valid:
        if native.native_calls or any(native.ticks):
            raise ValueError("nonentered D root acquired a primary native action")
        return ClassicalMeasuredOperands((None,) * 4, (None,), ("NO_SAFE_PREPARATION",))
    assert decision.callback is not None
    bases = tuple(donor_episode(causal, private, view) for view in (0, 1))
    rows = {(row.role, row.index, row.view): row for row in native.branches}
    reasons: set[str] = set()
    zero_peak: list[float | None] = []
    for view in (0, 1):
        zero = _window(
            branch=rows["EVALUATOR", 0, view],
            base=bases[view],
            callback=decision.callback,
            feed=0,
            zero_peak=0.0,
            owner_receipt_valid=True,
        )
        zero_peak.append(None if zero is None else zero.measured_peak_K)
        if zero is None:
            reasons.add(f"INVALID_ZERO_REFERENCE_V{view}")
    chart: list[MeasuredWord | None] = []
    for word in range(2):
        for view in (0, 1):
            measured = _window(
                branch=rows["EVALUATOR", word, view],
                base=bases[view],
                callback=decision.callback,
                feed=FEED_WORDS[word],
                zero_peak=zero_peak[view],
                owner_receipt_valid=True,
            )
            chart.append(measured)
            if measured is None:
                reasons.add(f"INVALID_CHART_V{view}_A{word}")
    owner: list[tuple[MeasuredWord, MeasuredWord] | None] = []
    for request, selected_word in enumerate((1,)):
        tick = native.ticks[request]
        if selected_word is None:
            if tick is not None and tick.disposition is not TickDisposition.NONATTEMPT:
                raise ValueError("D refusal gained an owner native action")
            owner.append(None)
            continue
        valid_tick = bool(
            tick is not None
            and tick.disposition is TickDisposition.ACTION_DELIVERED
            and tick.delivery_trace is not None
            and tick.delivery_trace.exact
        )
        pair = tuple(
            _window(
                branch=rows["PRIMARY", request, view],
                base=bases[view],
                callback=decision.callback,
                feed=FEED_WORDS[selected_word],
                zero_peak=zero_peak[view],
                owner_receipt_valid=valid_tick,
            )
            for view in (0, 1)
        )
        if pair[0] is None or pair[1] is None:
            owner.append(None)
            reasons.add(f"INVALID_OWNER_REQUEST_{request}")
        else:
            owner.append((pair[0], pair[1]))
    return ClassicalMeasuredOperands(
        tuple(None if row is None else ClassicalMeasuredWord.from_measured(row) for row in chart),
        tuple(
            None
            if pair is None
            else (
                ClassicalMeasuredWord.from_measured(pair[0]),
                ClassicalMeasuredWord.from_measured(pair[1]),
            )
            for pair in owner
        ),
        tuple(sorted(reasons)),
    )

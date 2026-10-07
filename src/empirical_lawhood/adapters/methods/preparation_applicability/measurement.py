"""Shared frozen lower-law panel reduction and post-handoff consumer arithmetic.

Providers authenticate current OriginalFiniteResponseLaw and native records.
Their pre-future seals and terminal canonical records remain separate owners.
No policy fitting, native acquisition or historical seed-label derivation is
performed here. Numerical semantics follow the donor applicability reducer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from empirical_lawhood.adapters.methods.prepared_response.projection import (
    NativeResponseArrays,
    reduce_native_response_arrays,
)
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_instruments import (
    preparation_policy_compact_interface,
)
from empirical_lawhood.adapters.simulators.preparation_applicability.contracts import WORDS
from empirical_lawhood.adapters.simulators.prepared_response.instruments import PreparedPortFrame
from empirical_lawhood.adapters.simulators.six_matrix_response.response_observer import _decode

from .operands import NativePhaseOperand, PanelOperand, PrefixOperand
from .science import CAPS, DELTA, HANDOFF_TICK, RESPONSE_TICKS, SELECTION_TICK

if TYPE_CHECKING:
    from empirical_lawhood.adapters.methods.finite_response_law.original_f import (
        OriginalFiniteResponseLaw,
    )

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.int64]


def response_arrays(phase: NativePhaseOperand) -> NativeResponseArrays:
    count = len(phase.ticks)
    return NativeResponseArrays(
        np.asarray(phase.ticks, dtype=np.int64),
        _decode(phase.positions_base64, (count, 2, 3, 4, 4)),
        _decode(phase.transfer_base64, (count, 15, 15), real=True),
        np.asarray(phase.transfer_known, dtype=bool),
        float(phase.signed_work),
        float(phase.absolute_work),
    )


def reduce_panel(
    prefix: PrefixOperand, panel: PanelOperand
) -> tuple[Array, Array, Array] | None:
    if (
        panel.root_id != prefix.root_id
        or panel.prefix_sha256 != prefix.fingerprint()
        or prefix.frame_base64 is None
        or tuple(phase.refinement for phase in prefix.phases) != (1, 2)
        or any(phase.root_id != prefix.root_id for phase in panel.phases)
    ):
        raise ValueError("measurement changes the authenticated prefix/frame/root views")
    frame = PreparedPortFrame(
        SELECTION_TICK, _decode(prefix.frame_base64, (2, 3, 4, 4))
    )
    phases = {
        (phase.schedule_index, phase.refinement, phase.phase, phase.future_index, phase.word_index): phase
        for phase in panel.phases
    }
    if len(phases) != len(panel.phases):
        raise ValueError("measurement duplicates an acquisition cell")
    expected = {(s, v, "parent", None, None) for s in range(3) for v in (1, 2)} | {
        (s, v, "future", f, w)
        for s in range(3)
        for v in (1, 2)
        for f in (0, 1)
        for w in range(9)
    }
    if set(phases) != expected or any(
        phase.disposition != "COMPLETE" for phase in phases.values()
    ):
        return None
    z = np.empty((3, 24, 2))
    y = np.empty((3, 4, 8, 2, 2))
    work = np.empty((3, 2))
    for s in range(3):
        for v in (1, 2):
            parent = phases[s, v, "parent", None, None]
            if parent.incoming_sha256 != prefix.phases[v - 1].fingerprint():
                raise ValueError("measurement substitutes preparation predecessor")
            work[s, v - 1] = float(parent.parent_work)
            z[s, :, v - 1] = preparation_policy_compact_interface(
                frame=frame,
                ticks=parent.history_ticks,
                positions=_decode(parent.history_positions_base64, (31, 2, 3, 4, 4)),
                momenta=_decode(parent.history_momenta_base64, (31, 2, 3, 4, 4)),
            ).values
            for f in (0, 1):
                hold = phases[s, v, "future", f, 0]
                rows = []
                for w in range(1, 9):
                    phase = phases[s, v, "future", f, w]
                    if (
                        phase.incoming_sha256 != parent.fingerprint()
                        or hold.incoming_sha256 != parent.fingerprint()
                        or phase.streams != hold.streams
                        or phase.innovation_sha256 != hold.innovation_sha256
                    ):
                        raise ValueError(
                            "signed branch substitutes handoff or same-purpose HOLD innovations"
                        )
                    observed, known = reduce_native_response_arrays(
                        frame=frame,
                        word=WORDS[w],
                        handoff_tick=HANDOFF_TICK,
                        readouts=(RESPONSE_TICKS,),
                        future=response_arrays(phase),
                        paired_hold=response_arrays(hold),
                    )
                    if not known[0, :5].all():
                        return None
                    rows.append(observed[0, :5])
                for pair in range(4):
                    minus, plus = rows[2 * pair : 2 * pair + 2]
                    y[s, pair, :, f, v - 1] = np.r_[
                        (plus[:2] - minus[:2]) / 2, plus[2:], minus[2:]
                    ]
    return z, y, work


def validity(
    lower: OriginalFiniteResponseLaw, z: Array, y: Array, work: Array
) -> tuple[Array, BoolArray, Array, Array, Array, BoolArray]:
    """Seven preserved maxima, without dropping adverse cells or changing q.

    The current OriginalFiniteResponseLaw constructor pins the original numerical
    operand digest; its current artifact identity belongs to the caller's seal.
    """
    delta = np.asarray(DELTA)
    nu = delta / 8
    pred = lower.predict(z[:, :, 0])
    mean, width = pred.mean, float(lower.q) * pred.sigma + nu
    error = abs(y - mean[..., None, None])
    maxima = np.column_stack(
        (
            np.max(abs(lower.normalize(z[:, :, 0])), axis=1) / 6,
            np.max(abs(y[..., 0] - y[..., 1]) / nu[None, None, :, None], axis=(1, 2, 3)),
            np.max(error / delta[None, None, :, None, None], axis=(1, 2, 3, 4)),
            np.max(width / delta, axis=(1, 2)),
            np.max(y[:, :, 2:] / np.asarray(CAPS)[None, None, :, None, None], axis=(1, 2, 3, 4)),
            np.max(work, axis=1) / 32,
            np.max(error / width[..., None, None], axis=(1, 2, 3, 4)),
        )
    )
    return maxima, maxima <= 1, 1 - np.max(maxima, axis=1), mean, width, pred.supported


def requests(scientific_seed: int) -> tuple[IntArray, Array]:
    """Consume an explicit PCG64 allocation, with the original 256 paired draws.

    Scientific eligibility and allocation/exposure checks belong to current
    authoring. A different public label does not alter this consumed integer.
    """
    if type(scientific_seed) is not int or not 0 <= scientific_seed < 2**128:
        raise ValueError("paired requests require an explicit 128-bit PCG64 allocation")
    rng = np.random.Generator(np.random.PCG64(scientific_seed))
    direction = rng.integers(0, 4, (256, 2))
    requirement = rng.uniform((0.02, 0.02), (0.12, 0.06), (256, 2))
    return direction, requirement


def lower_choices(
    mean: Array, width: Array, support: BoolArray, direction: IntArray, requirement: Array
) -> IntArray:
    delta = np.asarray(DELTA)
    caps = np.asarray(CAPS)
    low, high = mean - width, mean + width
    low[:, :, 2:] = np.maximum(0, low[:, :, 2:])
    high[:, :, 2:] = np.maximum(0, high[:, :, 2:])
    selected = np.full((3, 256, 2), -1, dtype=np.int64)
    for s in range(3):
        for consumer in (0, 1):
            axis = direction[:, consumer] // 2
            polarity = np.where(direction[:, consumer] % 2 == 0, 1, -1)
            upper, transverse = ((0.16, 0.02), (0.10, 0.01))[consumer]
            for word in range(8):
                pair, sign = word // 2, (-1 if word % 2 == 0 else 1)
                lo, hi = low[s, pair].copy(), high[s, pair].copy()
                if sign < 0:
                    lo[:2], hi[:2] = -hi[:2].copy(), -lo[:2].copy()
                    lo[2:], hi[2:] = lo[np.r_[5:8, 2:5]], hi[np.r_[5:8, 2:5]]
                longlo = np.where(polarity > 0, lo[axis], -hi[axis])
                longhi = np.where(polarity > 0, hi[axis], -lo[axis])
                feasible = (
                    support[s]
                    & (width[s, pair] <= delta).all()
                    & (hi[2:] <= caps).all()
                    & (longlo >= requirement[:, consumer])
                    & (longhi <= upper)
                    & (np.maximum(abs(lo[1 - axis]), abs(hi[1 - axis])) <= transverse)
                )
                chosen = feasible & (selected[s, :, consumer] == -1)
                selected[s, chosen, consumer] = word
    return selected


def service(
    mean: Array,
    width: Array,
    support: BoolArray,
    y: Array,
    work: Array,
    direction: IntArray,
    requirement: Array,
) -> tuple[IntArray, BoolArray]:
    selected = lower_choices(mean, width, support, direction, requirement)
    success = np.zeros_like(selected, dtype=bool)
    delta, caps = np.asarray(DELTA), np.asarray(CAPS)
    for s in range(3):
        for consumer in (0, 1):
            axis = direction[:, consumer] // 2
            polarity = np.where(direction[:, consumer] % 2 == 0, 1, -1)
            upper, transverse = ((0.16, 0.02), (0.10, 0.01))[consumer]
            for word in range(8):
                pair, sign = word // 2, (-1 if word % 2 == 0 else 1)
                chosen = selected[s, :, consumer] == word
                values = y[s, pair]
                source_valid = (
                    np.isfinite(values).all()
                    and (abs(values[..., 0] - values[..., 1]) <= (delta / 8)[:, None]).all()
                    and (values[2:] <= caps[:, None, None]).all()
                    and np.isfinite(work[s]).all()
                    and (work[s] <= 32).all()
                )
                response = sign * values[:2]
                longitudinal = polarity[:, None, None] * response[axis]
                transverse_actual = response[1 - axis]
                actual = (
                    source_valid
                    & (longitudinal >= requirement[:, consumer, None, None]).all(axis=(1, 2))
                    & (longitudinal <= upper).all(axis=(1, 2))
                    & (abs(transverse_actual) <= transverse).all(axis=(1, 2))
                )
                success[s, chosen, consumer] = actual[chosen]
    return selected, success

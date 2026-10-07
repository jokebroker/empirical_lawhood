"""Explicitly synthetic reducer counterexamples; no qualification evidence.

The native panel fixture supplies authenticated-shape operands at the reducer
boundary. It never acquires a matrix trajectory or supplies a public provider.
"""

from decimal import Decimal as D
from types import SimpleNamespace

import numpy as np
import pytest

from empirical_lawhood.adapters.methods.finite_response_law.assigned_prediction import AssignedPrediction
from empirical_lawhood.adapters.methods.preparation_applicability import measurement
from empirical_lawhood.adapters.methods.preparation_applicability.science import CAPS, DELTA
from empirical_lawhood.adapters.methods.preparation_applicability.statistics import paired_binary_test
from empirical_lawhood.adapters.simulators.preparation_applicability.contracts import WORDS


class _SyntheticLower:
    """Exposed arithmetic fixture, never a current-law/custody substitute."""

    q = D("0.73599618025169766")

    def normalize(self, handoff: np.ndarray) -> np.ndarray:
        return handoff.copy()

    def predict(self, handoff: np.ndarray) -> AssignedPrediction:
        return AssignedPrediction(
            np.zeros((len(handoff), 4, 8)),
            np.full((len(handoff), 4, 8), 1e-8),
            (np.abs(handoff) <= 6).all(axis=1),
        )


def test_finite_word_order_matches_signed_pair_denominator() -> None:
    assert tuple((word.magnitude, word.direction_index, word.sign) for word in WORDS) == (
        (D(0), 0, 0),
        (D(8), 0, -1), (D(8), 0, 1),
        (D(8), 1, -1), (D(8), 1, 1),
        (D(16), 0, -1), (D(16), 0, 1),
        (D(16), 1, -1), (D(16), 1, 1),
    )


def test_exact_one_sided_paired_binary_tail_and_concordant_denominator() -> None:
    selected = np.array([True] * 5 + [False] * 3, dtype=bool)
    fixed = np.zeros(8, dtype=bool)
    assert paired_binary_test(selected, fixed) == (5 / 8, 5, 0, 1 / 32)
    assert paired_binary_test(fixed, selected) == (-5 / 8, 0, 5, 1.0)
    concordant = np.array([True, False, True], dtype=bool)
    assert paired_binary_test(concordant, concordant) == (0.0, 0, 0, 1.0)


@pytest.mark.parametrize(
    "selected,fixed",
    [
        (np.ones((8, 256), dtype=bool), np.zeros((8, 256), dtype=bool)),
        (np.ones(8, dtype=np.int64), np.zeros(8, dtype=np.int64)),
        (np.ones(8, dtype=bool), np.zeros(7, dtype=bool)),
        (np.array([], dtype=bool), np.array([], dtype=bool)),
    ],
)
def test_paired_test_rejects_nested_events_and_unpaired_units(selected, fixed) -> None:
    with pytest.raises(ValueError, match="independent-root"):
        paired_binary_test(selected, fixed)


def test_explicit_request_seed_preserves_pcg64_draw_order() -> None:
    seed = int("0123456789abcdef0123456789abcdef", 16)
    direction, requirement = measurement.requests(seed)
    reference = np.random.Generator(np.random.PCG64(seed))
    np.testing.assert_array_equal(direction, reference.integers(0, 4, (256, 2)))
    np.testing.assert_array_equal(
        requirement, reference.uniform((0.02, 0.02), (0.12, 0.06), (256, 2))
    )


@pytest.mark.parametrize("seed", [True, -1, 2**128, 1.0])
def test_request_seed_rejects_non_consumed_allocations(seed) -> None:
    with pytest.raises(ValueError, match="128-bit"):
        measurement.requests(seed)


def _consumer_operands(value: float = 0.04):
    mean = np.zeros((3, 4, 8))
    mean[:, :, 0] = value
    width = np.full_like(mean, 1e-6)
    support = np.ones(3, dtype=bool)
    direction = np.zeros((256, 2), dtype=np.int64)
    requirement = np.full((256, 2), 0.03)
    actual = np.broadcast_to(mean[..., None, None], (3, 4, 8, 2, 2)).copy()
    work = np.zeros((3, 2))
    return mean, width, support, actual, work, direction, requirement


def test_consumer_limits_first_feasible_word_and_nonattempt() -> None:
    mean, width, support, actual, work, direction, requirement = _consumer_operands(0.12)
    selected, success = measurement.service(
        mean, width, support, actual, work, direction, requirement
    )
    assert (selected[:, :, 0] == 1).all()
    assert (selected[:, :, 1] == -1).all()
    assert success[:, :, 0].all()
    assert not success[:, :, 1].any()
    refused = measurement.lower_choices(mean, width, np.zeros(3, dtype=bool), direction, requirement)
    assert (refused == -1).all()
    width[:] = np.asarray(DELTA) * 2
    assert (measurement.lower_choices(mean, width, support, direction, requirement) == -1).all()


def test_negative_word_maps_sign_without_using_future_to_choose() -> None:
    mean, width, support, actual, work, direction, requirement = _consumer_operands(-0.04)
    selected, success = measurement.service(
        mean, width, support, actual, work, direction, requirement
    )
    assert (selected == 0).all()
    assert success.all()
    changed = actual.copy()
    changed[0, 0, 0, 1, 1] = 0.0
    after, failed = measurement.service(
        mean, width, support, changed, work, direction, requirement
    )
    np.testing.assert_array_equal(after, selected)
    assert not failed[0].any()
    assert failed[1:].all()


def test_preservation_and_late_work_refuse_actual_service() -> None:
    mean, width, support, actual, work, direction, requirement = _consumer_operands()
    actual[1, 0, 2] = CAPS[0] * 2
    work[2, 1] = 33
    selected, success = measurement.service(
        mean, width, support, actual, work, direction, requirement
    )
    assert (selected == 1).all()
    assert success[0].all()
    assert not success[1:].any()


def test_seven_maxima_keep_last_schedule_future_view_and_work() -> None:
    z = np.zeros((3, 24, 2))
    y = np.zeros((3, 4, 8, 2, 2))
    work = np.zeros((3, 2))
    z[0, 23, 0] = 7
    y[2, 3, 7, 1, 1] = DELTA[7] / 4
    work[1, 1] = 64
    maxima, conjuncts, margins, _, width, support = measurement.validity(
        _SyntheticLower(), z, y, work
    )
    assert maxima.shape == (3, 7)
    assert maxima[0, 0] == 7 / 6
    assert maxima[1, 5] == 2
    assert maxima[2, 1] == 2
    assert not conjuncts[0, 0] and not conjuncts[1, 5] and not conjuncts[2, 1]
    assert not support[0] and support[1:].all()
    assert (width > np.asarray(DELTA) / 8).all()
    np.testing.assert_array_equal(margins, 1 - maxima.max(axis=1))


def _synthetic_panel(monkeypatch):
    root = "exposed.synthetic.root"
    prefix_phases = tuple(
        SimpleNamespace(root_id=root, refinement=v, fingerprint=lambda v=v: f"prefix-{v}")
        for v in (1, 2)
    )
    prefix = SimpleNamespace(
        root_id=root, phases=prefix_phases, frame_base64="frame", fingerprint=lambda: "prefix"
    )
    phases = []
    for s in range(3):
        for v in (1, 2):
            parent_id = f"parent-{s}-{v}"
            phases.append(SimpleNamespace(
                root_id=root, phase="parent", schedule_index=s, refinement=v,
                future_index=None, word_index=None, disposition="COMPLETE",
                incoming_sha256=f"prefix-{v}", parent_work=D(s + v),
                history_ticks=tuple(range(4016, 4497, 16)),
                history_positions_base64=f"history:{s}:{v}",
                history_momenta_base64=f"history:{s}:{v}",
                fingerprint=lambda parent_id=parent_id: parent_id,
            ))
            for f in (0, 1):
                for w in range(9):
                    phases.append(SimpleNamespace(
                        root_id=root, phase="future", schedule_index=s, refinement=v,
                        future_index=f, word_index=w, disposition="COMPLETE",
                        incoming_sha256=parent_id, streams=(f"stream-{f}",),
                        innovation_sha256=f"innovation-{s}-{v}-{f}",
                    ))
    panel = SimpleNamespace(root_id=root, prefix_sha256="prefix", phases=tuple(phases))

    def decode(raw, shape, *, real=False):
        result = np.zeros(shape, dtype=np.float64 if real else np.complex128)
        if raw == "frame":
            result[0, 0, 0, 0] = 1
            result[1, 0, 1, 1] = 1
        elif raw.startswith("history:"):
            _, s, v = raw.split(":")
            result[:] = 10 * int(s) + int(v)
        return result

    def compact(*, frame, ticks, positions, momenta):
        assert frame.cutoff_tick == 4096
        assert ticks == tuple(range(4016, 4497, 16))
        return SimpleNamespace(values=np.full(24, positions[0, 0, 0, 0, 0].real))

    def project(*, frame, word, handoff_tick, readouts, future, paired_hold):
        assert frame.cutoff_tick == 4096 and handoff_tick == 4496 and readouts == (192,)
        assert paired_hold.word_index == 0
        assert future.future_index == paired_hold.future_index
        pair = (future.word_index - 1) // 2
        observed = np.zeros((1, 7))
        observed[0, :5] = (
            word.sign * (pair + 1) / 100,
            word.sign * (pair + 1) / 200,
            0.01 + pair / 1000, 0.02, 0.03,
        )
        return observed, np.ones((1, 7), dtype=bool)

    monkeypatch.setattr(measurement, "_decode", decode)
    monkeypatch.setattr(measurement, "preparation_policy_compact_interface", compact)
    monkeypatch.setattr(measurement, "response_arrays", lambda phase: phase)
    monkeypatch.setattr(measurement, "reduce_native_response_arrays", project)
    return prefix, panel


def test_complete_panel_keeps_paired_signs_all_views_and_causal_clocks(monkeypatch) -> None:
    prefix, panel = _synthetic_panel(monkeypatch)
    result = measurement.reduce_panel(prefix, panel)
    assert result is not None
    z, y, work = result
    assert z.shape == (3, 24, 2) and y.shape == (3, 4, 8, 2, 2)
    np.testing.assert_array_equal(z[2], np.tile([21, 22], (24, 1)))
    np.testing.assert_array_equal(work, [[1, 2], [2, 3], [3, 4]])
    np.testing.assert_allclose(
        y[2, 3, :, 1, 1],
        [0.04, 0.02, 0.013, 0.02, 0.03, 0.013, 0.02, 0.03],
        rtol=0, atol=1e-15,
    )


@pytest.mark.parametrize("change", ["missing", "failure"])
def test_incomplete_panel_retains_unevaluable_cells(monkeypatch, change) -> None:
    prefix, panel = _synthetic_panel(monkeypatch)
    if change == "missing":
        panel.phases = panel.phases[:-1]
    else:
        panel.phases[-1].disposition = "OBSERVATION_FAILURE"
    assert measurement.reduce_panel(prefix, panel) is None


@pytest.mark.parametrize("change", ["fingerprint", "unresolved-frame", "view-order"])
def test_panel_rejects_prefix_and_frame_substitution(monkeypatch, change) -> None:
    prefix, panel = _synthetic_panel(monkeypatch)
    if change == "fingerprint":
        panel.prefix_sha256 = "other-prefix"
    elif change == "unresolved-frame":
        prefix.frame_base64 = None
    else:
        prefix.phases = tuple(reversed(prefix.phases))
    with pytest.raises(ValueError, match="prefix/frame"):
        measurement.reduce_panel(prefix, panel)


@pytest.mark.parametrize("change", ["duplicate", "root", "parent", "hold", "stream", "innovations"])
def test_panel_rejects_acquisition_and_same_purpose_substitutions(monkeypatch, change) -> None:
    prefix, panel = _synthetic_panel(monkeypatch)
    if change == "duplicate":
        panel.phases += (panel.phases[-1],)
    elif change == "root":
        panel.phases[-1].root_id = "other-root"
    elif change == "parent":
        panel.phases[0].incoming_sha256 = "other-prefix"
    elif change == "hold":
        panel.phases[1].incoming_sha256 = "other-parent"
    elif change == "stream":
        panel.phases[2].streams = ("other-stream",)
    else:
        panel.phases[2].innovation_sha256 = "other-innovations"
    with pytest.raises(ValueError):
        measurement.reduce_panel(prefix, panel)


def test_unresolved_late_projection_does_not_become_zero_response(monkeypatch) -> None:
    prefix, panel = _synthetic_panel(monkeypatch)
    original = measurement.reduce_native_response_arrays

    def unresolved(**kwargs):
        observed, known = original(**kwargs)
        if kwargs["future"].schedule_index == 2 and kwargs["future"].word_index == 8:
            known[0, 4] = False
        return observed, known

    monkeypatch.setattr(measurement, "reduce_native_response_arrays", unresolved)
    assert measurement.reduce_panel(prefix, panel) is None

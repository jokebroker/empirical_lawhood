# SPDX-License-Identifier: MPL-2.0
"""Independent one- and two-future native opportunity readouts.

These exposed-development checks preserve the original 16-root, five-parent,
256-request-pair census. Callers authenticate arrays, science freeze, producer
reports and panel lineage under their separate analysis authority before entry.
The checks do not import producer predicates, contact sources or grant authority.
"""

import json
from hashlib import sha256
from typing import Any

import numpy as np

from .science import FiniteResponseLawScienceSpec


def _stats(event: Any) -> tuple[float, list[float], float]:
    per_request = np.sum(event, axis=1, dtype=float) / 5
    means = [sum(row) / 256 for row in per_request]
    mean = sum(means) / 16
    variance = sum(
        (sum((row - means[r]) ** 2) / 255 for r, row in enumerate(per_request))
    ) / (16 * 16 * 256)
    return (float(mean), [float(x) for x in means], float(variance))


def _check_stats(event: Any, reported: dict[str, Any]) -> None:
    mean, roots, var = _stats(event)
    np.testing.assert_allclose(
        [mean, var],
        [reported["frequency"], reported["conditional_request_mc_variance"]],
        rtol=0,
        atol=1e-14,
    )
    np.testing.assert_allclose(roots, reported["per_root"], rtol=0, atol=1e-14)


def verify_specification_opportunity(
    raw: dict[str, Any],
    requests: dict[str, Any],
    spec: FiniteResponseLawScienceSpec,
    produced: dict[str, Any],
    summary_bytes: bytes,
) -> dict[str, Any]:
    """Check the original single-future prepared-response opportunity and magnitude gates.

    Input arrays and report bytes must already be authenticated. This readout
    checks consistency on revealed data and supplies no fresh-entry permission.
    """
    summary = json.loads(summary_bytes)
    directions, thresholds = requests["direction"][:16], requests["lower"][:16]
    values = raw["values"][16:, :, :, :, 2]
    work = raw["parent_work"][16:]
    admissible = np.zeros((16, 5, 256, 2, 8), dtype=bool)
    delta = np.asarray(spec.delta, dtype=float)
    preservation_caps = np.asarray(spec.preservation, dtype=float)
    for r in range(16):
        for p in range(5):
            for w in range(8):
                amplitude = 8 if w < 4 else 16
                direction = w % 4 // 2
                sign = -1 if w % 2 == 0 else 1
                neg = 1 + (8 if amplitude == 16 else 0) + 2 * direction
                pos = neg + 1
                a, b = (
                    values[r, :, p, pos if sign == 1 else neg],
                    values[r, :, p, neg if sign == 1 else pos],
                )
                response = (a[:, :2] - b[:, :2]) / 2
                joined = np.concatenate((response, a[:, 2:5], b[:, 2:5]), axis=1)
                base = (
                    np.isfinite(joined).all()
                    and np.isfinite(work[r, :, p]).all()
                    and np.all(work[r, :, p] <= float(spec.parent_work_maximum))
                    and np.all(np.abs(joined[0] - joined[1]) <= delta / 8)
                    and np.all(a[:, 2:5] <= preservation_caps)
                    and np.all(b[:, 2:5] <= preservation_caps)
                )
                if not base:
                    continue
                for c in range(2):
                    for request in range(256):
                        axis = int(directions[r, request, c]) // 2
                        polarity = 1 if directions[r, request, c] % 2 == 0 else -1
                        along = polarity * response[:, axis]
                        across = response[:, 1 - axis]
                        admissible[r, p, request, c, w] = bool(
                            min(along) >= thresholds[r, request, c]
                            and max(along) <= float(spec.upper[c])
                            and (max(abs(across)) <= float(spec.transverse[c]))
                        )
    np.testing.assert_array_equal(admissible, produced["feasible"])
    choices = np.full((16, 5, 256, 2), -1, dtype=np.int64)
    for w in reversed(range(8)):
        choices[admissible[..., w]] = w
    np.testing.assert_array_equal(choices, produced["selected"])
    joint = np.asarray((choices != -1).all(axis=3))
    np.testing.assert_array_equal(joint, produced["joint"])
    low = np.zeros((16, 5, 256, 2), dtype=bool)
    high = low.copy()
    for r in range(16):
        for q in range(256):
            for c in range(2):
                d = int(directions[r, q, c])
                w = 2 * (d // 2) + (1 if d % 2 == 0 else 0)
                low[r, :, q, c] = admissible[r, :, q, c, w]
                high[r, :, q, c] = admissible[r, :, q, c, w + 4]
    np.testing.assert_array_equal(low, produced["aligned_low"])
    np.testing.assert_array_equal(high, produced["aligned_high"])
    _check_stats(joint, summary["joint"])
    for c, name in enumerate(("A", "B")):
        z = summary["consumers"][name]
        lo, hi = (low[..., c], high[..., c])
        for event, key in (
            (choices[..., c] != -1, "oracle_success"),
            (lo != hi, "aligned_magnitude_discrimination"),
            (lo & ~hi, "low_only"),
            (hi & ~lo, "high_only"),
            (lo & hi, "both"),
            (~lo & ~hi, "neither"),
        ):
            _check_stats(event, z[key])
    gates = {
        "joint_oracle_success": _stats(joint)[0] >= 0.8,
        "A_magnitude_discrimination": _stats(low[..., 0] != high[..., 0])[0] >= 0.2,
        "B_magnitude_discrimination": _stats(low[..., 1] != high[..., 1])[0] >= 0.2,
    }
    if gates != summary["gates"] or all(gates.values()) != summary["opportunity_screen_eligible"]:
        raise ValueError("Gate/continuation mismatch")
    result = {
        "schema": 'finite-response-law-preparation-screen-independent-readout',
        "verified": True,
        "producer_result_sha256": sha256(summary_bytes).hexdigest(),
        "independent_roots": 16,
        "word_request_cells_checked": int(admissible.size),
        "choices_checked": int(choices.size),
        "root_request_joint_events_checked": int(joint.size),
        "gates": gates,
        "source": "authenticated native prepared-response projection values; no importer/oracle acceptance predicate",
        "new_native_updates": 0,
    }
    return result


def verify_opportunity_screen_opportunity(
    raw: dict[str, Any],
    requests: dict[str, Any],
    spec: FiniteResponseLawScienceSpec,
    second: Any,
    seen: Any,
    valid: Any,
    panel: dict[str, Any],
    baseline_panel: dict[str, Any],
    produced: dict[str, Any],
    summary_bytes: bytes,
) -> dict[str, Any]:
    "Check complete measured futures and preserve the original prepared-response paired panel.\n\n    Input arrays and report bytes must already be authenticated. This readout\n    checks consistency on revealed data and supplies no fresh-entry permission.\n    "
    summary = json.loads(summary_bytes)
    directions, thresholds = requests["direction"][:16], requests["lower"][:16]
    if not seen.all() or not valid.all():
        raise ValueError("Independent native opportunity verification requires the complete measured supplement")
    native_words = (0, 1, 2, 3, 4, 9, 10, 11, 12)
    first = raw["values"][16:, :, :, :, 2][:, :, :, native_words, :]
    values = np.stack((first, second), axis=1)
    z = baseline_panel
    for key in ("y", "observed", "valid", "parent_work", "hold_observed"):
        np.testing.assert_equal(panel[key][16:], z[key][16:])
    np.testing.assert_equal(panel["y"][:16, :, :, :, 0], z["y"][:16, :, :, :, 0])
    work = raw["parent_work"][16:]
    admissible = np.zeros((16, 5, 256, 2, 8), dtype=bool)
    delta = np.asarray(spec.delta, dtype=float)
    preservation_caps = np.asarray(spec.preservation, dtype=float)
    for r in range(16):
        for p in range(5):
            for w in range(8):
                amplitude = 8 if w < 4 else 16
                direction = w % 4 // 2
                sign = -1 if w % 2 == 0 else 1
                neg = 1 + (4 if amplitude == 16 else 0) + 2 * direction
                pos = neg + 1
                a, b = (
                    values[r, :, :, p, pos if sign == 1 else neg],
                    values[r, :, :, p, neg if sign == 1 else pos],
                )
                response = (a[..., :2] - b[..., :2]) / 2
                joined = np.concatenate((response, a[..., 2:5], b[..., 2:5]), axis=2)
                if sign == 1:
                    np.testing.assert_array_equal(
                        panel["y"][r, p, w // 2].transpose(1, 2, 0), joined
                    )
                base = (
                    np.isfinite(joined).all()
                    and np.isfinite(work[r, :, p]).all()
                    and np.all(work[r, :, p] <= float(spec.parent_work_maximum))
                    and np.all(np.abs(joined[:, 0] - joined[:, 1]) <= delta / 8)
                    and np.all(a[..., 2:5] <= preservation_caps)
                    and np.all(b[..., 2:5] <= preservation_caps)
                )
                if not base:
                    continue
                for c in range(2):
                    for request in range(256):
                        axis = int(directions[r, request, c]) // 2
                        polarity = 1 if directions[r, request, c] % 2 == 0 else -1
                        along = polarity * response[..., axis]
                        across = response[..., 1 - axis]
                        admissible[r, p, request, c, w] = bool(
                            float(np.min(along)) >= thresholds[r, request, c]
                            and float(np.max(along)) <= float(spec.upper[c])
                            and (
                                float(np.max(np.abs(across)))
                                <= float(spec.transverse[c])
                            )
                        )
    np.testing.assert_array_equal(admissible, produced["feasible"])
    choices = np.full((16, 5, 256, 2), -1, dtype=np.int64)
    for w in reversed(range(8)):
        choices[admissible[..., w]] = w
    np.testing.assert_array_equal(choices, produced["selected"])
    joint = np.asarray((choices != -1).all(axis=3))
    np.testing.assert_array_equal(joint, produced["joint"])
    low = np.zeros((16, 5, 256, 2), dtype=bool)
    high = low.copy()
    for r in range(16):
        for q in range(256):
            for c in range(2):
                d = int(directions[r, q, c])
                w = 2 * (d // 2) + (1 if d % 2 == 0 else 0)
                low[r, :, q, c] = admissible[r, :, q, c, w]
                high[r, :, q, c] = admissible[r, :, q, c, w + 4]
    np.testing.assert_array_equal(low, produced["aligned_low"])
    np.testing.assert_array_equal(high, produced["aligned_high"])
    _check_stats(joint, summary["joint"])
    for c, name in enumerate(("A", "B")):
        z = summary["consumers"][name]
        lo, hi = (low[..., c], high[..., c])
        for event, key in (
            (choices[..., c] != -1, "oracle_success"),
            (lo != hi, "aligned_magnitude_discrimination"),
            (lo & ~hi, "low_only"),
            (hi & ~lo, "high_only"),
            (lo & hi, "both"),
            (~lo & ~hi, "neither"),
        ):
            _check_stats(event, z[key])
    gates = {
        "joint_oracle_success": _stats(joint)[0] >= 0.8,
        "A_magnitude_discrimination": _stats(low[..., 0] != high[..., 0])[0] >= 0.2,
        "B_magnitude_discrimination": _stats(low[..., 1] != high[..., 1])[0] >= 0.2,
    }
    if gates != summary["gates"] or all(gates.values()) != summary["fit_nomination_eligible"]:
        raise ValueError("Gate/continuation mismatch")
    result = {
        "schema": 'finite-response-law-opportunity-screen-independent-readout',
        "verified": True,
        "producer_result_sha256": sha256(summary_bytes).hexdigest(),
        "independent_roots": 16,
        "word_request_cells_checked": int(admissible.size),
        "choices_checked": int(choices.size),
        "root_request_joint_events_checked": int(joint.size),
        "gates": gates,
        "source": "authenticated retained prepared-response and actual D1 projection values; independent native pairing, four-way constraints and reductions",
        "futures": 2,
        "numerical_views": 2,
        "complete_prepared_response_scalar_cells_checked": 10240,
        'fit_nomination_eligible': all(gates.values()),
        "new_native_updates": 0,
    }
    return result

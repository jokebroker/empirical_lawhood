"Independent saved-operand checker. Imports no producer, scoring or verdict code.\n\nReconstructs normal equations/support/ranks directly, never refits or acquires.\nThis checks raw numerical operands and inference. Prospective owner evidence is\nvalidated separately by the existing trajectory controller-use owner, never by this checker.\n"

from __future__ import annotations
import argparse
from hashlib import sha256
import json
from pathlib import Path
from typing import Any
import numpy as np

FEATURE_COLUMNS = (
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


def content_digest(value: object) -> str:
    return sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def check_discovery(raw_path: Path, output: Path) -> dict[str, int]:
    saved = json.loads(output.read_bytes())
    with np.load(raw_path, allow_pickle=False) as archive:
        raw = {key: archive[key] for key in archive.files}
    return check_discovery_arrays(raw, saved)


def check_discovery_arrays(raw: dict[str, np.ndarray], saved: dict[str, Any]) -> dict[str, int]:
    train = raw["roles"] == "fit"
    nomination = raw["roles"] == "nomination"
    x, y = raw["features"][train], raw["labels"][train, 0, :2]
    require(
        saved["fit_mask"] == train.tolist() and saved["nomination_mask"] == nomination.tolist(),
        "saved scientific masks",
    )
    require(len(saved["fits"]) == 9 and len(saved["calls"]) == 9, "nine fits missing")
    recomputed = []
    training_digest = content_digest(
        {"x": x.tolist(), "y": y.tolist(), "roots": raw["roots"][train].tolist()}
    )
    for index, (family, d, penalty) in enumerate(
        (f, d, p) for f, d in enumerate((7, 13, 23)) for p in (1e-6, 0.001, 0.1)
    ):
        model = saved["fits"][index]
        require(model["family"] == family and model["penalty"] == penalty, "fit roster")
        require(model["columns"] == list(FEATURE_COLUMNS[:d]), "saved feature order")
        require(
            model["rows"] == len(x) and model["training_digest"] == training_digest,
            "training identity and census",
        )
        require(
            saved["calls"][index]
            == dict(
                family=family,
                penalty=penalty,
                rows=len(x),
                ridge=len(x) * penalty,
                input_digest=training_digest,
                output_digest=content_digest(model["operator"]),
            ),
            "observed fit call operands",
        )
        mean, scale = x[:, :d].mean(0), x[:, :d].std(0)
        scale[scale < 1e-12] = 1
        require(
            np.array_equal(mean, model["mean"]) and np.array_equal(scale, model["scale"]),
            "training-only normalizer",
        )
        design = np.column_stack(((x[:, :d] - mean) / scale, np.ones(len(x))))
        b = np.array(model["operator"])
        require(b.shape == (d + 1, 2), "intercept-last shape")
        regularizer = np.diag([len(x) * penalty] * d + [0])
        require(
            bool(
                np.allclose(
                    (design.T @ design + regularizer) @ b, design.T @ y, rtol=1e-9, atol=1e-7
                )
            ),
            "saved normal equations",
        )
        cells = []
        strata = np.where(raw["clocks"] < 600, 0, np.where(raw["clocks"] < 25200, 1, 2))
        for s in range(3):
            for a in range(9):
                mask = train & (strata == s) & (raw["actions"] == a)
                if not mask.any():
                    continue
                lo, hi = raw["features"][mask, :d].min(0), raw["features"][mask, :d].max(0)
                pad = np.where(hi == lo, 1e-12, 0.05 * (hi - lo))
                cells.append(
                    dict(
                        clock_stratum=s,
                        action=a,
                        roots=sorted(set(raw["roots"][mask].tolist())),
                        lower=(lo - pad).tolist(),
                        upper=(hi + pad).tolist(),
                    )
                )
        require(cells == model["support"], "training-only support")
        query = raw["features"][nomination, :d]
        point = ((query - mean) / scale) @ b[:-1] + b[-1]
        labels = raw["labels"][nomination]
        all_labels = raw["labels"]
        valid = bool(
            np.isfinite(all_labels).all()
            and (np.abs(all_labels[:, 0] - all_labels[:, 1]) <= (0.01, 0.0002, 0.000001)).all()
            and raw["delivery_valid"].all()
        )
        supported = all(
            any(
                c["clock_stratum"] == int(s)
                and c["action"] == int(a)
                and len(c["roots"]) >= 8
                and np.all(q >= c["lower"])
                and np.all(q <= c["upper"])
                for c in cells
            )
            for q, s, a in zip(query, strata[nomination], raw["actions"][nomination], strict=True)
        )
        residual = labels[:, :, :2] - point[:, None, :]
        maximum = np.max(np.abs(residual), axis=(0, 1))
        rmse = np.sqrt(np.mean(residual[:, 0] ** 2, axis=0))
        eligible = (
            valid
            and supported
            and bool(np.all(maximum <= (0.25, 0.01)) and np.all(rmse <= (0.1, 0.005)))
        )
        score = saved["nominations"][index]
        require(score["family"] == family and score["penalty"] == penalty, "nomination roster")
        require(score["eligible"] == eligible, "numerical validity must determine nomination")
        require(
            bool(np.allclose(score["maximum"], maximum, rtol=1e-10, atol=1e-10)),
            "nomination maximum",
        )
        require(bool(np.allclose(score["rmse"], rmse, rtol=1e-10, atol=1e-10)), "nomination rmse")
        q_dev = float(max(maximum / (0.25, 0.01))) if eligible else None
        require(
            (score["q_dev"] is None and q_dev is None)
            or (
                score["q_dev"] is not None
                and q_dev is not None
                and bool(np.isclose(score["q_dev"], q_dev, rtol=1e-10, atol=1e-10))
            ),
            "frozen development q",
        )
        recomputed.append((eligible, family, float(rmse[0] / 0.1 + rmse[1] / 0.005), -penalty))
    eligible_indices = [i for i, s in enumerate(recomputed) if s[0]]
    choice = min(eligible_indices, key=lambda i: recomputed[i][1:]) if eligible_indices else None
    require(saved["selected"] == choice, "nomination selection")
    rivals = [min(range(f * 3, f * 3 + 3), key=lambda i: recomputed[i][2:]) for f in (0, 1)]
    require(saved["rivals"] == rivals, "frozen simpler rivals")
    require(
        saved["disposition"]
        == ("NOMINATED" if choice is not None else "IDENTIFICATION_NOT_SUPPORTED"),
        "nomination disposition",
    )
    return {"normal_equations_checked": 9, "nomination_results_checked": 9}


def check_calibration(
    raw_path: Path,
    fit_path: Path,
    output: Path,
    *,
    qualification: bool = False,
    frozen_q: float | None = None,
) -> dict[str, int]:
    model, saved = json.loads(fit_path.read_bytes()), json.loads(output.read_bytes())
    d = (7, 13, 23)[model["family"]]
    role = "qualification" if qualification else "calibration"
    adequate = []
    with np.load(raw_path, allow_pickle=False) as raw:
        require(
            raw["roots"].tolist() == [f"reactor-empirical-{role}-{i:03d}" for i in range(32)],
            "assigned roots",
        )
        if not qualification:
            require(len(saved["scores"]) == 32, "missing final score")
        scores: list[float | None] = []
        for r in range(32):
            clocks, labels = raw["clocks"][r], raw["labels"][r]
            complete = (
                labels.shape == (2880, 2, 3)
                and clocks.shape == (2880,)
                and raw["features"][r].shape == (2880, 23)
                and raw["actions"][r].shape == (2880,)
                and raw["requests"][r].shape == (2880, 2, 2)
            )
            if not complete:
                if qualification:
                    adequate.append(False)
                else:
                    require(
                        saved["scores"][r]["value"] is None and bool(saved["scores"][r]["reasons"]),
                        "missing census concealed",
                    )
                    scores.append(None)
                continue
            valid = bool(
                np.array_equal(clocks, np.arange(2880) * 10)
                and np.isfinite(labels).all()
                and np.all(np.abs(labels[:, 0] - labels[:, 1]) <= (0.01, 0.0002, 0.000001))
                and np.array_equal(raw["requests"][r, :, 0], raw["requests"][r, :, 1])
            )
            for view, name, steps, dt in ((0, "nominal", 10, 1.0), (1, "refined", 20, 0.5)):
                p, a = raw[f"projected_{name}"][r], raw[f"delivered_{name}"][r]
                if p.shape != (2880, steps, 4) or a.shape != p.shape:
                    valid = False
                    continue
                valid = (
                    valid
                    and a.shape == (2880, steps, 4)
                    and np.array_equal(p, a)
                    and bool(np.isfinite(a).all())
                )
                valid = (
                    valid
                    and bool(np.all(a[:, :, 1] == dt))
                    and np.array_equal(a[:, :, 0], clocks[:, None] + np.arange(steps) * dt)
                )
                requests = raw["requests"][r, :, view]
                feed, jacket, dosed = 0.0, 316.0, 0.0
                rebuilt = np.empty((2880, steps, 4))
                rebuilt_stages = np.empty((2880, 4))
                for callback, (f, j) in enumerate(requests):
                    target = (
                        min(0.032, max(0.0, f))
                        if 600 <= callback * 10 < 25200 and dosed < 287.3
                        else 0.0
                    )
                    feed = (
                        min(0.032, max(feed - 0.02, min(feed + 0.02, target)))
                        if dosed < 287.3
                        else 0.0
                    )
                    target_j = min(347.6, max(279.3, j))
                    jacket = max(jacket - 2.3, min(jacket + 2.3, target_j))
                    rebuilt_stages[callback] = target, target_j, feed, jacket
                    for step in range(steps):
                        exposure = min(feed, max(287.3 - dosed, 0.0) / dt)
                        rebuilt[callback, step] = (callback * 10 + step * dt, dt, exposure, jacket)
                        dosed += exposure * dt
                valid = valid and "stages" in raw and raw["stages"][r].shape == (2880, 2, 4)
                if valid:
                    valid = bool(np.array_equal(raw["stages"][r, :, view], rebuilt_stages))
                valid = valid and np.array_equal(a, rebuilt)
                mass = np.cumsum(np.sum(a[:, :, 1] * a[:, :, 2], axis=1))
                valid = valid and bool(np.all(np.abs(mass - labels[:, view, 2]) <= 1e-6))
            try:
                reconstructed = reconstruct_causal_path(
                    raw["observations"][r], model, float(raw["consumer_q"][r, 0])
                )
                expected_actions, expected_features, expected_requests = reconstructed
                causal_valid = (
                    raw["actions"].dtype.kind in "iu"
                    and np.array_equal(
                        raw["model_sha256"][r], list(bytes.fromhex(content_digest(model)))
                    )
                    and (not qualification or raw["consumer_q"][r, 0] == frozen_q)
                    and np.array_equal(raw["actions"][r], expected_actions)
                    and np.array_equal(raw["features"][r], expected_features)
                    and np.array_equal(raw["requests"][r, :, 0], expected_requests)
                    and raw["observations"][r, 0, 3] == 0
                    and np.all(np.abs(raw["observations"][r, 1:, 3] - labels[:-1, 0, 2]) <= 1e-6)
                )
            except (KeyError, ValueError, IndexError):
                causal_valid = False
            valid = valid and bool(causal_valid)
            x = raw["features"][r, :, :d]
            z = (x - model["mean"]) / model["scale"]
            operator = np.asarray(model["operator"])
            point = z @ operator[:-1] + operator[-1]
            actions = raw["actions"][r]
            for k in range(2880):
                s = 0 if clocks[k] < 600 else 1 if clocks[k] < 25200 else 2
                valid = valid and any(
                    c["clock_stratum"] == s
                    and c["action"] == int(actions[k])
                    and len(set(c["roots"])) >= 8
                    and np.all(x[k] >= c["lower"])
                    and np.all(x[k] <= c["upper"])
                    for c in model["support"]
                )
            expected = (
                float(np.max(np.abs(labels[:, :, :2] - point[:, None, :]) / (0.25, 0.01)))
                if valid
                else None
            )
            if qualification:
                covered = False
                if frozen_q is not None:
                    width = np.array([frozen_q * 0.25 + 0.01, frozen_q * 0.01 + 0.0002])
                    covered = valid and bool(
                        np.all(width <= (0.26, 0.0102))
                        and np.all(np.abs(labels[:, :, :2] - point[:, None, :]) <= width)
                    )
                adequate.append(covered)
                continue
            record = saved["scores"][r]
            require(record["root"] == str(raw["roots"][r]), "root score identity")
            require(
                record["value"] == expected,
                "raw validity/residual disagrees with published root score",
            )
            require(bool(record["reasons"]) == (not valid), "validity reasons concealed")
            require(
                record["callbacks"] == 2880 and record["receiver_views"] == 11520,
                "last callback/receiver/view accounting",
            )
            scores.append(expected)
        if qualification:
            from scipy.stats import beta

            operands = saved["value"]["operands"]["value"]
            require(operands["adequacy"] == adequate, "raw qualification A conjunction differs")
            k = sum(adequate)
            lower = 0.0 if k == 0 else float(beta.ppf(0.05, k, 33 - k))
            require(
                float(operands["lower_bound"]["decimal"]) == lower,
                "independent one-sided binomial bound",
            )
            expected_status = "SUPPORTED" if all(adequate) and lower >= 0.90 else "NOT_SUPPORTED"
            require(
                saved["value"]["qualification"]["value"]["scientific_status"] == expected_status,
                "sole-owner qualification disagrees with raw operands",
            )
            return {
                "qualification_roots_checked": 32,
                "callback_receiver_views_checked": 32 * 2880 * 2 * 2,
            }
        q = (
            None
            if any(s is None for s in scores)
            else sorted(s for s in scores if s is not None)[31]
        )
        require(saved["q"] == q and saved["rank"] == 32, "rank32 calibration")
        require(
            saved["halfwidth"] == (None if q is None else [q * 0.25 + 0.01, q * 0.01 + 0.0002]),
            "interval padding",
        )
        require(
            saved["disposition"]
            == ("CALIBRATED" if q is not None and q <= 1 else "CALIBRATION_UNUSABLE"),
            "calibration disposition",
        )
    return {"roots_checked": 32, "callback_receiver_views_checked": 32 * 2880 * 2 * 2}


def reconstruct_causal_path(
    observations: np.ndarray, model: dict[str, Any], q: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent native-actuator/feature/consumer equations; no producer import."""
    require(
        observations.shape == (2880, 4) and bool(np.isfinite(observations).all()),
        "native observation census",
    )
    require(
        np.array_equal(observations[:, 0], np.arange(2880) * 10) and np.isfinite(q) and q >= 0,
        "causal clock/consumer q",
    )
    t, temperature, jacket, dose = observations.T
    past_feed = np.concatenate(([0.0], np.diff(dose) / 10))
    require(bool(np.all(past_feed >= 0)), "monotone native dose")
    it = np.concatenate(([0.0], np.cumsum(np.maximum(temperature[:-1] - 330, 0) * 10)))
    ij = np.concatenate(([0.0], np.cumsum((temperature[:-1] - jacket[:-1]) * 10)))
    d = (7, 13, 23)[model["family"]]
    op = np.asarray(model["operator"])
    width = np.array((q * 0.25 + 0.01, q * 0.01 + 0.0002))
    previous_feed, previous_jacket = 0.0, 316.0
    actions, rows, requests = [], [], []
    for k in range(2880):
        candidates, seen = [], []
        for a in range(9):
            f = (0.0, 0.016, 0.032)[a // 3]
            j = min(347.6, max(279.3, previous_jacket + (-2.3, 0.0, 2.3)[a % 3]))
            target = f if 600 <= t[k] < 25200 and dose[k] < 287.3 else 0.0
            af = (
                min(0.032, max(previous_feed - 0.02, min(previous_feed + 0.02, target)))
                if dose[k] < 287.3
                else 0.0
            )
            aj = max(previous_jacket - 2.3, min(previous_jacket + 2.3, j))
            mass, profile = dose[k], []
            for _ in range(10):
                flow = min(af, max(287.3 - mass, 0.0))
                profile.append((flow, aj))
                mass += flow
            mean_feed = sum(p[0] for p in profile) / 10
            u, v = (temperature[k] - 318.4) / 40, (jacket[k] - 316) / 40
            x = [
                t[k] / 28800,
                dose[k] / 287.3,
                float(600 <= t[k] < 25200),
                past_feed[k] / 0.032,
                (previous_jacket - 316) / 40,
                mean_feed / 0.032,
                (aj - 316) / 40,
                u,
                v,
                u * u,
                u * v,
                u * mean_feed / 0.032,
                u * (aj - 316) / 40,
            ]
            x.extend((temperature[k] - temperature[max(0, k - lag)]) / 40 for lag in (6, 30))
            x.extend((jacket[k] - jacket[max(0, k - lag)]) / 40 for lag in (6, 30))
            x.extend(
                float(np.sum(past_feed[max(1, k - lag + 1) : k + 1])) / lag / 0.032
                for lag in (6, 30)
            )
            x.extend((it[k] / (40 * 28800), ij[k] / (40 * 28800), x[13] * x[5], x[13] * x[6]))
            point = ((np.array(x[:d]) - model["mean"]) / model["scale"]) @ op[:-1] + op[-1]
            s = 0 if t[k] < 600 else 1 if t[k] < 25200 else 2
            support = any(
                c["clock_stratum"] == s
                and c["action"] == a
                and len(set(c["roots"])) >= 8
                and np.all(np.array(x[:d]) >= c["lower"])
                and np.all(np.array(x[:d]) <= c["upper"])
                for c in model["support"]
            )
            if (
                profile not in seen
                and support
                and np.isfinite(point).all()
                and point[0] + width[0] <= 356.2
                and np.all(width <= (0.26, 0.0102))
            ):
                candidates.append(
                    (
                        -(point[1] - width[1] + mass / 287.3),
                        -(point[1] + mass / 287.3),
                        a,
                        x,
                        (f, j),
                        (af, aj),
                    )
                )
            seen.append(profile)
        require(bool(candidates), "frozen consumer NONATTEMPT")
        chosen = min(candidates, key=lambda c: c[:3])
        actions.append(chosen[2])
        rows.append(chosen[3])
        requests.append(chosen[4])
        previous_feed, previous_jacket = chosen[5]
    return np.asarray(actions), np.asarray(rows), np.asarray(requests)


def check_policy(raw_path: Path) -> dict[str, int]:
    saved = json.loads(raw_path.read_bytes())
    history, decisions, model, q = saved["history"], saved["decisions"], saved["model"], saved["q"]
    require(len(history) == len(decisions) == 2880, "complete callback census")
    require([o["time"] for o in history] == list(range(0, 28800, 10)), "callback clocks")
    temperature = np.array([o["temperature"] for o in history])
    jacket = np.array([o["jacket"] for o in history])
    dose = np.array([o["dose"] for o in history])
    past_feed = np.concatenate(([0.0], np.diff(dose) / 10))
    integral_t = np.concatenate(([0.0], np.cumsum(np.maximum(temperature[:-1] - 330, 0) * 10)))
    integral_gap = np.concatenate(([0.0], np.cumsum((temperature[:-1] - jacket[:-1]) * 10)))
    previous_feed, previous_jacket = 0.0, 316.0
    d = (7, 13, 23)[model["family"]]
    operator = np.array(model["operator"])
    width = np.array([q * 0.25 + 0.01, q * 0.01 + 0.0002])
    candidate_count = 0
    for k, decision in enumerate(decisions):
        t = k * 10
        require(len(decision["candidates"]) == 9, "nine candidates per callback")
        profiles, eligible = [], []
        for action, c in enumerate(decision["candidates"]):
            request = (
                (0.0, 0.016, 0.032)[action // 3],
                min(347.6, max(279.3, previous_jacket + (-2.3, 0, 2.3)[action % 3])),
            )
            target = request[0] if 600 <= t < 25200 and dose[k] < 287.3 else 0.0
            applied_feed = (
                min(0.032, max(previous_feed - 0.02, min(previous_feed + 0.02, target)))
                if dose[k] < 287.3
                else 0.0
            )
            applied_jacket = max(previous_jacket - 2.3, min(previous_jacket + 2.3, request[1]))
            mass = dose[k]
            profile = []
            for step in range(10):
                feed = min(applied_feed, max(287.3 - mass, 0.0))
                profile.append([float(t + step), 1.0, feed, applied_jacket])
                mass += feed
            p = c["projection"]
            require(
                p["action"] == action and p["requested"] == list(request),
                "nominal menu/action identity",
            )
            require(
                p["accepted"] == [target, request[1]]
                and p["applied"] == [applied_feed, applied_jacket],
                "actuator stages",
            )
            require(
                p["exposure"] == profile and p["next_dose"] == mass, "lossless projected exposure"
            )
            projected_mean = sum(v[2] for v in profile) / 10
            u, v = (temperature[k] - 318.4) / 40, (jacket[k] - 316) / 40
            x = [
                t / 28800,
                dose[k] / 287.3,
                float(600 <= t < 25200),
                past_feed[k] / 0.032,
                (previous_jacket - 316) / 40,
                projected_mean / 0.032,
                (applied_jacket - 316) / 40,
                u,
                v,
                u * u,
                u * v,
                u * projected_mean / 0.032,
                u * (applied_jacket - 316) / 40,
            ]
            x += [(temperature[k] - temperature[max(0, k - lag)]) / 40 for lag in (6, 30)]
            x += [(jacket[k] - jacket[max(0, k - lag)]) / 40 for lag in (6, 30)]
            x += [
                float(sum(past_feed[max(1, k - lag + 1) : k + 1])) / lag / 0.032 for lag in (6, 30)
            ]
            x += [integral_t[k] / (40 * 28800), integral_gap[k] / (40 * 28800)]
            x += [x[13] * x[5], x[13] * x[6]]
            require(
                bool(np.allclose(c["features"], x, rtol=1e-13, atol=1e-14)),
                "causal features reconstructed from raw observations",
            )
            z = (np.array(x[:d]) - model["mean"]) / model["scale"]
            point = z @ operator[:-1] + operator[-1]
            require(
                bool(np.array_equal(c["point"], point)) and np.array_equal(c["halfwidth"], width),
                "prediction/uncertainty",
            )
            s = 0 if t < 600 else 1 if t < 25200 else 2
            supported = any(
                cell["clock_stratum"] == s
                and cell["action"] == action
                and len(set(cell["roots"])) >= 8
                and np.all(np.array(x[:d]) >= cell["lower"])
                and np.all(np.array(x[:d]) <= cell["upper"])
                for cell in model["support"]
            )
            admissible = (
                supported
                and bool(np.isfinite(point).all())
                and point[0] + width[0] <= 356.2
                and bool(np.all(width <= (0.26, 0.0102)))
            )
            require((not c["reasons"]) == admissible, "noncompensating numerical gates")
            profiles.append(profile)
            if admissible:
                eligible.append(
                    (-(point[1] - width[1] + mass / 287.3), -(point[1] + mass / 287.3), action)
                )
            candidate_count += 1
        for action, c in enumerate(decision["candidates"]):
            aliases = [i for i, profile in enumerate(profiles) if profile == profiles[action]]
            require(c["aliases"] == aliases, "delivery aliases")
        canonical = [
            item for item in eligible if item[2] == min(decision["candidates"][item[2]]["aliases"])
        ]
        chosen = min(canonical)[2] if canonical else None
        require(decision["selected"] == chosen, "robust/nominal/action tie ordering")
        if chosen is None:
            require(k == len(decisions) - 1, "refusal must terminate callback stream")
        else:
            previous_feed, previous_jacket = decision["candidates"][chosen]["projection"]["applied"]
    return {"callbacks_checked": 2880, "candidate_queries_checked": candidate_count}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "operation",
        choices=(
            "discovery",
            "development",
            "calibration",
            "qualification",
            "policy",
            "comparisons",
            "campaign",
        ),
    )
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--fit", type=Path)
    p.add_argument("--calibration-raw", type=Path)
    p.add_argument("--calibration-artifact", type=Path)
    a = p.parse_args()
    if a.operation == "policy":
        print(json.dumps(check_policy(a.raw), sort_keys=True))
        return
    if a.output is None:
        p.error("numerical artifact output required")
    if a.operation == "campaign":
        print(json.dumps(check_campaign_publication(a.raw, a.output), sort_keys=True))
        return
    if a.operation == "comparisons":
        require(
            a.raw.stat().st_size <= 2 * 1024**2 and a.output.stat().st_size <= 131072,
            "comparison input bounds",
        )
        print(
            json.dumps(
                check_paired_contrasts(
                    json.loads(a.raw.read_bytes()), json.loads(a.output.read_bytes())
                ),
                sort_keys=True,
            )
        )
        return
    if a.operation == "development":
        print(json.dumps(check_development_publication(a.raw, a.output), sort_keys=True))
        return
    if a.operation == "qualification":
        if a.fit is None or a.calibration_raw is None or a.calibration_artifact is None:
            p.error("qualification needs frozen fit and raw/serialized calibration")
        check_calibration(a.calibration_raw, a.fit, a.calibration_artifact)
        q = json.loads(a.calibration_artifact.read_bytes())["q"]
        print(
            json.dumps(
                check_calibration(a.raw, a.fit, a.output, qualification=True, frozen_q=q),
                sort_keys=True,
            )
        )
        return
    if a.operation == "discovery":
        result = check_discovery(a.raw, a.output)
    else:
        if a.fit is None:
            p.error("calibration requires --fit")
        result = check_calibration(a.raw, a.fit, a.output)
    print(json.dumps(result, sort_keys=True))


def independent_development_projection(
    observations: np.ndarray, requests: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the declared development rows without importing projection/feature code."""
    require(
        observations.shape == (2880, 4) and requests.shape == (2880, 2), "development causal axes"
    )
    require(
        bool(np.isfinite(observations).all() and np.isfinite(requests).all()),
        "finite development causal inputs",
    )
    require(np.array_equal(observations[:, 0], np.arange(2880) * 10), "development causal clocks")
    _, temperatures, jackets, doses = observations.T
    increments = np.concatenate(([0.0], np.diff(doses) / 10))
    require(bool((increments >= 0).all()) and doses[0] == 0.0, "development observed dose prefix")
    integral_t = np.concatenate(([0.0], np.cumsum(np.maximum(temperatures[:-1] - 330, 0) * 10)))
    integral_j = np.concatenate(([0.0], np.cumsum((temperatures[:-1] - jackets[:-1]) * 10)))
    last_feed, last_jacket = 0.0, 316.0
    rows = []
    ids = []
    for k, (f, j) in enumerate(requests):
        t = k * 10
        mass = doses[k]
        words = tuple(
            (feed, min(347.6, max(279.3, last_jacket + shift)))
            for feed in (0.0, 0.016, 0.032)
            for shift in (-2.3, 0.0, 2.3)
        )

        def apply(
            request: tuple[float, float],
        ) -> tuple[float, float, list[tuple[float, float]], float]:
            feed, jacket = request
            target = min(0.032, max(0.0, feed)) if 600 <= t < 25200 and mass < 287.3 else 0.0
            applied_f = (
                min(0.032, max(last_feed - 0.02, min(last_feed + 0.02, target)))
                if mass < 287.3
                else 0.0
            )
            applied_j = max(
                last_jacket - 2.3, min(last_jacket + 2.3, min(347.6, max(279.3, jacket)))
            )
            dose = mass
            profile = []
            for _ in range(10):
                actual = min(applied_f, max(287.3 - dose, 0.0))
                profile.append((actual, applied_j))
                dose += actual
            return applied_f, applied_j, profile, dose

        feed, jacket, profile, next_dose = apply((f, j))
        if k < 2879:
            require(doses[k + 1] == next_dose, "development dose is not prior actual exposure")
        aliases = [i for i, word in enumerate(words) if apply(word)[2] == profile]
        require(bool(aliases), "unmapped exact branch exposure")
        direct = [i for i, word in enumerate(words) if word == (f, j)]
        ids.append(min(direct) if direct else min(aliases))
        mean = sum(p[0] for p in profile) / 10
        u = (temperatures[k] - 318.4) / 40
        v = (jackets[k] - 316) / 40
        x = [
            t / 28800,
            mass / 287.3,
            float(600 <= t < 25200),
            increments[k] / 0.032,
            (last_jacket - 316) / 40,
            mean / 0.032,
            (jacket - 316) / 40,
            u,
            v,
            u * u,
            u * v,
            u * mean / 0.032,
            u * (jacket - 316) / 40,
        ]
        x.extend((temperatures[k] - temperatures[max(0, k - lag)]) / 40 for lag in (6, 30))
        x.extend((jackets[k] - jackets[max(0, k - lag)]) / 40 for lag in (6, 30))
        x.extend(
            float(np.sum(increments[max(1, k - lag + 1) : k + 1])) / lag / 0.032 for lag in (6, 30)
        )
        x.extend(
            (integral_t[k] / (40 * 28800), integral_j[k] / (40 * 28800), x[13] * x[5], x[13] * x[6])
        )
        rows.append(x)
        last_feed, last_jacket = feed, jacket
    return np.asarray(rows), np.asarray(ids)


def independent_native_arrays(
    raw: dict[str, np.ndarray], dt: float
) -> tuple[np.ndarray, np.ndarray]:
    """Independently reduce raw grids and reconstruct every known input stage."""
    n = int(10 / dt)
    grid = raw["grid"]
    requests = raw["requests"]
    require(
        dt in (1.0, 0.5) and requests.shape == (2880, 2) and grid.shape == (int(28800 / dt) + 1, 8),
        "native development full-grid census",
    )
    require(np.array_equal(grid[:, 0], np.arange(len(grid)) * dt), "native development grid clock")
    require(bool(np.isfinite(grid).all()), "native development finite labels")
    labels = np.asarray(
        [
            (float(np.max(grid[k : k + n + 1, 1])), grid[k + n, 4], grid[k + n, 3])
            for k in range(0, len(grid) - 1, n)
        ]
    )
    valid = np.zeros(2880, dtype=bool)
    if (
        raw["stages"].shape != (2880, 4)
        or raw["exposure"].shape != (2880, n, 4)
        or raw["observations"].shape != (2880, 4)
        or raw["callback_cpu"].shape != (2880,)
        or any(a.dtype != np.float64 or not np.isfinite(a).all() for a in raw.values())
        or (raw["callback_cpu"] < 0).any()
    ):
        return labels, valid
    feed, jacket, dose = 0.0, 316.0, 0.0
    for k, (f, j) in enumerate(requests):
        target = min(0.032, max(0.0, f)) if 600 <= k * 10 < 25200 and dose < 287.3 else 0.0
        feed = min(0.032, max(feed - 0.02, min(feed + 0.02, target))) if dose < 287.3 else 0.0
        target_j = min(347.6, max(279.3, j))
        jacket = max(jacket - 2.3, min(jacket + 2.3, target_j))
        profile = []
        before = dose
        for step in range(n):
            actual = min(feed, max(287.3 - dose, 0.0) / dt)
            profile.append((k * 10 + step * dt, dt, actual, jacket))
            dose += actual * dt
        valid[k] = (
            np.array_equal(raw["stages"][k], (target, target_j, feed, jacket))
            and np.array_equal(raw["exposure"][k], profile)
            and raw["observations"][k, 0] == k * 10
            and raw["observations"][k, 3] == before
            and labels[k, 2] == dose
        )
    return labels, valid


def check_development_projection(
    episodes: dict[tuple[int, int, bool], dict[str, np.ndarray]], saved: dict[str, np.ndarray]
) -> dict[str, int]:
    """The complete 24 × five × two raw census, never saved features as input truth."""
    expected = {(r, e, v) for r in range(24) for e in range(5) for v in (False, True)}
    require(set(episodes) == expected, "development assigned raw episode census")
    require(
        np.array_equal(saved["root_index"], np.repeat(np.arange(24), 14400))
        and np.array_equal(saved["episode_index"], np.tile(np.repeat(np.arange(5), 2880), 24)),
        "saved development root/episode axes",
    )
    require(
        np.array_equal(saved["clocks"], np.tile(np.arange(2880) * 10, 120)), "saved causal clocks"
    )
    for r in range(24):
        for e in range(5):
            nominal = episodes[r, e, False]
            refined = episodes[r, e, True]
            require(
                np.array_equal(nominal["requests"], refined["requests"]),
                "refinement changed donor requested tape",
            )
            x, a = independent_development_projection(nominal["observations"], nominal["requests"])
            lo = (r * 5 + e) * 2880
            hi = lo + 2880
            require(
                np.array_equal(saved["features"][lo:hi], x)
                and np.array_equal(saved["actions"][lo:hi], a),
                "raw-derived development features/support IDs",
            )
            left, lv = independent_native_arrays(nominal, 1.0)
            right, rv = independent_native_arrays(refined, 0.5)
            require(
                np.array_equal(saved["labels"][lo:hi], np.stack((left, right), axis=1)),
                "raw-derived closed-window/end-point labels",
            )
            require(
                np.array_equal(saved["delivery_valid"][lo:hi].astype(bool), lv & rv),
                "raw-derived assigned delivery validity",
            )
            # Check actual exploration/branch recipe, not just arbitrary legal requests.
            if e < 3:
                previous = 316.0
                for k, values in enumerate(nominal["observations"]):
                    target = (340.0, 350.0, 355.0)[(r + e + (k * 10) // 2400) % 3]
                    rate = (values[1] - nominal["observations"][max(0, k - 1), 1]) / 10
                    desired = min(347.6, max(279.3, 316.0 + 8 * (target - values[1]) - 120 * rate))
                    menu = [min(347.6, max(279.3, previous + delta)) for delta in (-2.3, 0.0, 2.3)]
                    wanted = (
                        (0.0, 0.016, 0.032)[(r + e + (k * 10) // 120) % 3],
                        menu[min(range(3), key=lambda j: (abs(menu[j] - desired), j))],
                    )
                    require(
                        np.array_equal(nominal["requests"][k], wanted),
                        "raw acquisition changed exploration policy",
                    )
                    previous = nominal["stages"][k, 3]
            else:
                donor = episodes[r, 1, False]
                branch_requests = donor["requests"].copy()
                anchor = (3600, 9000, 18000, 24000)[r % 4] // 10
                if e == 3:
                    branch_requests[anchor, 0] = 0.032 if branch_requests[anchor, 0] == 0 else 0.0
                else:
                    previous = donor["stages"][anchor - 1, 3]
                    low = max(279.3, previous - 2.3)
                    high = min(347.6, previous + 2.3)
                    branch_requests[anchor, 1] = high if branch_requests[anchor, 1] == low else low
                require(
                    np.array_equal(nominal["requests"], branch_requests),
                    "raw branch changed an unassigned donor request",
                )
    return dict(independent_roots=24, paired_episodes=120, causal_rows=345600)


def _canonical_file(path: Path, schema: str | tuple[str, ...]) -> tuple[dict[str, Any], str]:
    require(path.stat().st_size <= 256 * 1024**2, "checker input exceeds byte bound")
    data = path.read_bytes()
    record = json.loads(data)
    require(
        (
            json.dumps(
                record, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
            )
            + "\n"
        ).encode()
        == data,
        "input is not canonical JSON",
    )
    require(
        set(record) == {"schema", "value", "version"}
        and record["schema"] in ((schema,) if isinstance(schema, str) else schema)
        and record["version"] == "1.0.0",
        "checker input schema",
    )
    return record["value"], sha256(data).hexdigest()


def _arrays(document: dict[str, Any]) -> dict[str, np.ndarray]:
    import base64
    from io import BytesIO
    import zipfile

    require(document["schema"] == 'empirical-lawhood/methods/reactor-causal-response/empirical-array-payload', "array schema")
    value = document["value"]
    require(len(value["npz_base64"]) <= 128 * 1024**2, "array encoding bound")
    raw = base64.b64decode(value["npz_base64"], validate=True)
    require(sha256(raw).hexdigest() == value["sha256"], "array content hash")
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        require(
            len(archive.infolist()) <= 128
            and sum(i.file_size for i in archive.infolist()) <= 256 * 1024**2,
            "array decoded byte bound",
        )
    with np.load(BytesIO(raw), allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    require(all(a.dtype.kind in "fiu" for a in result.values()), "numeric-only arrays")
    return result


def check_development_publication(index: Path, output: Path) -> dict[str, int]:
    """Index maps the 24 assigned root IDs to exact acquisition JSON paths.

    Authentication uses the discovery's retained acquisition fingerprints;
    no fitter, actuator implementation, verdict or simulator is imported.
    """
    require(index.stat().st_size <= 32768, "development index bound")
    paths = json.loads(index.read_bytes())
    names = tuple(
        f"reactor-empirical-{role}-{i:03d}"
        for role, n in (("fit", 16), ("nomination", 8))
        for i in range(n)
    )
    require(set(paths) == set(names), "development raw index census")
    publication, _ = _canonical_file(output, 'empirical-lawhood/methods/reactor-causal-response/empirical-discovery-envelope')
    identities = {
        v["value"]["object_id"]: v["value"]["object_fingerprint"]
        for v in publication["acquisitions"]
    }
    require(
        set(identities) == set(names) and len(publication["acquisitions"]) == 24,
        "discovery acquisition identities",
    )
    episodes = {}
    invalid_inputs = 0
    for i, root in enumerate(names):
        raw, digest = _canonical_file(
            index.parent / paths[root], 'empirical-lawhood/methods/reactor-causal-response/empirical-acquisition-envelope'
        )
        require(digest == identities[root] and raw["root"] == root, "raw development custody")
        require(raw["role"] == ("fit" if i < 16 else "nomination"), "development split")
        roster = [(v["value"]["episode"], v["value"]["refined"]) for v in raw["episodes"]]
        require(
            roster
            == [(name, view) for name in ("exploration-unshifted", "exploration-shifted-one-position", "exploration-shifted-two-positions", "feed-intervention", "jacket-intervention") for view in (False, True)],
            "raw development episode roster",
        )
        for item in raw["episodes"]:
            value = item["value"]
            require(value["root"] == root, "substituted raw episode root")
            e = ("exploration-unshifted", "exploration-shifted-one-position", "exploration-shifted-two-positions", "feed-intervention", "jacket-intervention").index(value["episode"])
            arrays = _arrays(value["arrays"])
            require(
                set(arrays)
                == {"observations", "requests", "stages", "exposure", "grid", "callback_cpu"},
                "raw episode array roster",
            )
            episodes[i, e, value["refined"]] = arrays
            n = 20 if value["refined"] else 10
            invalid_inputs += bool(
                value["failure"] is not None
                or any(a.dtype != np.float64 or not np.isfinite(a).all() for a in arrays.values())
                or arrays["observations"].shape != (2880, 4)
                or arrays["grid"].shape != (2880 * n + 1, 8)
            )
    saved = json.loads(publication["result_json"])
    if publication["rows"] is None:
        if not invalid_inputs:
            for (r, e, refined), raw in episodes.items():
                if refined:
                    continue
                try:
                    independent_development_projection(raw["observations"], raw["requests"])
                except ValueError:
                    invalid_inputs += 1
        require(
            invalid_inputs > 0
            and saved["selected"] is None
            and saved["fit_calls"] == 0
            and saved["disposition"] == "IDENTIFICATION_NOT_SUPPORTED",
            "unexplained no-fit terminal",
        )
        return dict(
            independent_roots=24,
            raw_episodes=240,
            invalid_inputs=invalid_inputs,
            normal_equations_checked=0,
        )
    require(not invalid_inputs, "missing required raw inputs were fitted")
    rows = _arrays(publication["rows"])
    projection = check_development_projection(episodes, rows)
    rows["roots"] = np.repeat(np.asarray(names), 14400)
    rows["roles"] = np.repeat(np.asarray(["fit"] * 16 + ["nomination"] * 8), 14400)
    return {**projection, **check_discovery_arrays(rows, saved)}


def independent_endpoints(
    arrays: dict[str, np.ndarray], *, complete: bool, reference_time: float | None
) -> dict[str, Any]:
    """Raw nominal grid reduction, independent of the producer endpoint function."""
    grid = arrays["grid"]
    require(grid.ndim == 2 and grid.shape[1] == 8, "nominal grid columns")
    finite = bool(np.isfinite(grid).all())
    full = (
        complete
        and finite
        and grid.shape == (28801, 8)
        and np.array_equal(grid[:, 0], np.arange(28801))
    )
    unsafe = bool(np.any(grid[:, 1] > 356.2))
    finished = (
        np.flatnonzero((grid[:, 3] >= 0.999 * 287.3) & (grid[:, 4] >= 0.98))
        if finite
        else np.empty(0, dtype=int)
    )
    time = float(grid[finished[0], 0]) if full and len(finished) else None
    ratio = (
        time / reference_time
        if time is not None
        and reference_time is not None
        and np.isfinite(reference_time)
        and reference_time > 0
        else None
    )
    peak = float(np.max(grid[:, 1])) if full else None
    return dict(
        complete=bool(full),
        task_success=bool(
            full
            and not unsafe
            and grid[-1, 3] >= 0.999 * 287.3
            and grid[-1, 4] >= 0.98
            and ratio is not None
            and ratio <= 1.06
        ),
        unsafe_observed=unsafe,
        unsafe_or_unobserved=unsafe or not full,
        restricted_time=min(time, 28800.0) if time is not None else 28800.0,
        peak=peak,
        minimum_margin=None if peak is None else 356.2 - peak,
        violation_duration=float(np.sum(grid[1:, 1] > 356.2)) if full else None,
        time_ratio=ratio,
    )


def check_paired_contrasts(
    rows: list[dict[str, Any]], saved: list[dict[str, Any]]
) -> dict[str, int]:
    """Independent declared 32-way inference; never imports the analysis owner.

    The input endpoints need their separate raw-grid audit; this operation checks
    pairing, fixed denominator, missingness, interval arithmetic and decisions.
    """
    import math
    from scipy.stats import beta

    arms = ("EL", "F0", "F1", "MARGIN", "FIXED", "REF", "EKF", "SCHEDULED_BACKOFF_ZERO", "SCHEDULED_BACKOFF_HALF")
    census = {(f"reactor-empirical-confirmation-{i:03d}", arm) for i in range(8) for arm in arms}
    index = {(row["root"], row["arm"]): row for row in rows}
    require(
        len(index) == len(rows) and set(index).issubset(census), "paired independent-root census"
    )
    comparisons = [
        (arm, e)
        for arm in arms[1:]
        for e in ("J", "unsafe-or-unobserved", "restricted-completion-time")
    ]
    comparisons += [(arm, e) for arm in ("F0", "F1") for e in ("temperature-mae", "conversion-mae")]
    comparisons += [(arm, "log-online-cpu-ratio") for arm in arms[5:]]
    # The serialized order is the fixed comparison-ID order. Check the full
    # set then address each member, so this checker does not borrow that order.
    outputs = {(c["comparator"], c["endpoint"]): c for c in saved}
    require(
        len(saved) == len(outputs) == 32 and set(outputs) == set(comparisons),
        "32-way contrast census",
    )
    draws = np.random.Generator(np.random.PCG64(86104)).integers(0, 8, (100000, 8))
    mapping = {
        "J": "task_success",
        "unsafe-or-unobserved": "unsafe_or_unobserved",
        "restricted-completion-time": "restricted_time",
        "temperature-mae": "temperature_mae",
        "conversion-mae": "conversion_mae",
    }
    for arm, endpoint in comparisons:
        actual = outputs[arm, endpoint]
        require(
            actual["analysis_scope"] == "DESCRIPTIVE_EFFECT_ESTIMATION_AND_MECHANISM_DIAGNOSTICS"
            and actual["confirmatory_superiority_claim"] is False,
            "eight-root contrast cannot promote confirmatory superiority",
        )
        pairs = []
        for i in range(8):
            values = []
            for who in ("EL", arm):
                row = index.get((f"reactor-empirical-confirmation-{i:03d}", who))
                value = None
                if row is not None:
                    if endpoint == "log-online-cpu-ratio":
                        cpu = row["online_cpu"]
                        value = (
                            math.log(cpu)
                            if row["complete"] and cpu is not None and cpu > 0
                            else None
                        )
                    else:
                        value = row[mapping[endpoint]]
                values.append(value)
            pairs.append(values)
        missing = sum(any(v is None or not np.isfinite(v) for v in p) for p in pairs)
        resolution = {
            "temperature-mae": 0.01,
            "conversion-mae": 0.001,
            "restricted-completion-time": 120.0,
            "log-online-cpu-ratio": -math.log(0.9),
        }.get(endpoint, 0.0)
        require(
            actual["assigned"] == 8
            and actual["missing_pairs"] == missing
            and actual["resolution"] == resolution
            and actual["favorable_direction"] == ("positive" if endpoint == "J" else "negative"),
            "contrast denominator/direction/resolution",
        )
        if missing:
            require(
                actual["disposition"] == "UNEVALUABLE"
                and all(actual[k] is None for k in ("estimate", "adjusted", "unadjusted")),
                "missing pairs must remain unevaluable",
            )
            continue
        differences = np.diff(np.asarray(pairs, dtype=float), axis=1).ravel() * -1
        estimate = float(differences.mean())
        bounds = []
        for alpha in (0.05 / (2 * 32), 0.025):
            if endpoint in ("J", "unsafe-or-unobserved"):
                b, c = int(np.sum(differences == 1)), int(np.sum(differences == -1))

                def cp(k: int) -> tuple[float, float]:
                    return (
                        0.0 if k == 0 else float(beta.ppf(alpha / 2, k, 9 - k)),
                        1.0 if k == 8 else float(beta.isf(alpha / 2, k + 1, 8 - k)),
                    )

                bl, bh = cp(b)
                cl, ch = cp(c)
                interval = (bl - ch, bh - cl)
            else:
                distribution = np.mean(differences[draws], axis=1)
                low, high = np.quantile(distribution, (alpha, 1 - alpha))
                interval = (float(low), float(high))
            bounds.append(interval)
        require(
            estimate == actual["estimate"]
            and all(
                np.array_equal(value, actual[key])
                for value, key in zip(bounds, ("adjusted", "unadjusted"), strict=True)
            ),
            "paired inference arithmetic",
        )
        if np.all(differences == 0):
            disposition = "IDENTICAL"
        elif estimate == 0:
            disposition = "ZERO_MEAN_ESTIMATE"
        elif abs(estimate) < resolution:
            disposition = "BELOW_RESOLUTION_ESTIMATE"
        elif (estimate > 0) == (endpoint == "J"):
            disposition = "FAVORABLE_ESTIMATE"
        else:
            disposition = "ADVERSE_ESTIMATE"
        require(actual["disposition"] == disposition, "contrast decision changed")
    return dict(independent_roots=8, contrasts_checked=32)


def check_campaign_publication(index: Path, output: Path) -> dict[str, int]:
    "Authenticate all eight raw publications and recompute endpoints and contrasts.\n\n    The index has discovery/qualification paths and a roots mapping. The output\n    is the canonical contribution record, which embeds all root analyses. controller use A\n    still belongs to the prospective owner; this checks its J/C join, not a\n    substitute adequacy verdict.\n    "
    require(index.stat().st_size <= 65536, "campaign index bound")
    paths = json.loads(index.read_bytes())
    arms = ("EL", "F0", "F1", "MARGIN", "FIXED", "REF", "EKF", "SCHEDULED_BACKOFF_ZERO", "SCHEDULED_BACKOFF_HALF")
    names = tuple(f"reactor-empirical-confirmation-{i:03d}" for i in range(8))
    require(set(paths["roots"]) == set(names), "campaign assigned root index")
    discovery, discovery_sha = _canonical_file(
        index.parent / paths["discovery"], 'empirical-lawhood/methods/reactor-causal-response/empirical-discovery-envelope'
    )
    qualification, qualification_sha = _canonical_file(
        index.parent / paths["qualification"],
        'empirical-lawhood/methods/reactor-causal-response/empirical-qualification-terminal',
    )
    cohort, _ = _canonical_file(output, 'empirical-lawhood/methods/reactor-causal-response/empirical-contribution-result')
    analyses = [item["value"] for item in cohort["roots"]]
    matrix = json.loads(cohort["matrix_json"])
    require(
        matrix["completion"]["native_benchmark_arms"] == ["EL", "REF", "EKF", "SCHEDULED_BACKOFF_ZERO", "SCHEDULED_BACKOFF_HALF"]
        and matrix["completion"]["deferred_native_benchmark_arms"]
        == ["F0", "F1", "MARGIN", "FIXED"],
        "campaign native score scope differs from the five declared treatments",
    )
    for key in (
        "completion",
        "safety-productivity",
        "prediction",
        "measured-cost",
        "qualification",
        "history-law-information",
        "action-selection",
        "reference-policy-admission",
    ):
        require(
            matrix[key]["analysis_scope"]
            == "DESCRIPTIVE_EFFECT_ESTIMATION_AND_MECHANISM_DIAGNOSTICS"
            and matrix[key]["confirmatory_superiority_claim"] is False
            and matrix[key]["overall_superiority_claim"] is False,
            "campaign contribution exceeds descriptive claim ceiling",
        )
    require(tuple(r["root"] for r in analyses) == names, "campaign analysis census")
    stop = cohort["prerequisite"]
    frozen = None if qualification["result"] is None else qualification["result"]["value"]
    model = None if frozen is None else json.loads(frozen["payload"]["value"]["model_json"])
    q = None if frozen is None else float(frozen["payload"]["value"]["q"]["decimal"])
    saved_discovery = json.loads(discovery["result_json"])
    checked_rows = []
    for root, analysis in zip(names, analyses, strict=True):
        raw, digest = _canonical_file(
            index.parent / paths["roots"][root],
            (
                'empirical-lawhood/reactor-response/reactor-confirmation-envelope',
                'empirical-lawhood/reactor-response/timed-reactor-confirmation-envelope',
            ),
        )
        require(
            raw["root"] == root
            and analysis["confirmation"]["value"]["object_fingerprint"] == digest
            and raw["discovery"]["value"]["object_fingerprint"] == discovery_sha
            and raw["qualification"]["value"]["object_fingerprint"] == qualification_sha,
            "campaign raw/analysis/frozen-parent identity",
        )
        require(raw["prerequisite"] == analysis["prerequisite"] == stop, "campaign entry gate")
        rows = json.loads(analysis["endpoints_json"])
        if stop is not None:
            require(
                raw["acquisition"] is None
                and not raw["children"]
                and not raw.get("telemetry")
                and not rows
                and analysis['prospective_evaluation'] is None,
                "unentered campaign invented observations",
            )
            continue
        require(model is not None and q is not None, "entered campaign lacks qualified law")
        assert model is not None and q is not None
        acquisition = raw["acquisition"]["value"]
        episodes = [e["value"] for e in acquisition["episodes"]]
        roster = [(e["episode"], e["refined"]) for e in episodes]
        require(
            acquisition["root"] == root
            and acquisition["role"] == "confirmation"
            and len(roster) == 10
            and set(roster) == {(a, False) for a in arms} | {("EL", True)}
            and all(e["root"] == root for e in episodes),
            "campaign raw episode pairing/census",
        )
        nominal = {
            e["episode"]: (_arrays(e["arrays"]), e["failure"]) for e in episodes if not e["refined"]
        }
        if "telemetry" in raw:
            timing = [t["value"] for t in raw["telemetry"]]
            require(tuple(t["arm"] for t in timing) == arms, "timing arm census")
            cost = json.loads(analysis["contact_json"])["cost"]
            for t in timing:
                wall = np.asarray([v["decimal"] for v in t["callback_wall_seconds"]], dtype=float)
                require(
                    bool(
                        len(wall) == len(nominal[t["arm"]][0]["callback_cpu"])
                        and len(wall) <= 2880
                        and np.isfinite(wall).all()
                        and (wall >= 0).all()
                        and np.isfinite(float(t["episode_wall_seconds"]["decimal"]))
                        and wall.sum() <= float(t["episode_wall_seconds"]["decimal"]) + 1e-12
                        and type(t["worker_peak_rss_bytes"]) is int
                        and t["worker_peak_rss_bytes"] > 0
                    ),
                    "raw timing census or measurement",
                )
                expected_timing = dict(
                    measured_callbacks=len(wall),
                    callback_wall_seconds_sum=float(wall.sum()),
                    callback_wall_seconds_max=float(wall.max()) if len(wall) else None,
                    callback_wall_seconds_median=float(np.median(wall)) if len(wall) else None,
                    callback_wall_seconds_p95=float(np.quantile(wall, 0.95)) if len(wall) else None,
                    episode_wall_seconds=float(t["episode_wall_seconds"]["decimal"]),
                    worker_peak_rss_bytes=t["worker_peak_rss_bytes"],
                )
                require(
                    cost["timing"][t["arm"]] == expected_timing, "published timing differs from raw"
                )

        def complete(a: dict[str, np.ndarray], failure: str | None) -> bool:
            shapes = dict(
                observations=(2880, 4),
                requests=(2880, 2),
                stages=(2880, 4),
                exposure=(2880, 10, 4),
                grid=(28801, 8),
                callback_cpu=(2880,),
            )
            return (
                failure is None
                and set(a) == set(shapes)
                and all(
                    a[k].shape == shape and a[k].dtype == np.float64 for k, shape in shapes.items()
                )
            )

        ref, failure = nominal["REF"]
        reference = independent_endpoints(ref, complete=complete(ref, failure), reference_time=None)
        # A missing completion must not become the 28800-second restriction.
        ref_grid = ref["grid"]
        attained = np.flatnonzero((ref_grid[:, 3] >= 0.999 * 287.3) & (ref_grid[:, 4] >= 0.98))
        reference_time = (
            reference["restricted_time"] if reference["complete"] and len(attained) else None
        )
        primary, failure = nominal["EL"]
        x, labels = None, None
        if complete(primary, failure):
            try:
                _, x, _ = reconstruct_causal_path(primary["observations"], model, q)
            except ValueError as error:
                if str(error) != "frozen consumer NONATTEMPT":
                    raise
            grid = primary["grid"]
            labels = np.asarray(
                [(np.max(grid[k : k + 11, 1]), grid[k + 10, 4]) for k in range(0, 28800, 10)]
            )
        require(
            tuple(r["arm"] for r in rows) == arms and all(r["root"] == root for r in rows),
            "campaign endpoint census",
        )
        for row in rows:
            a, failure = nominal[row["arm"]]
            expected = independent_endpoints(
                a, complete=complete(a, failure), reference_time=reference_time
            )
            expected["online_cpu"] = (
                float(a["callback_cpu"].sum()) if len(a["callback_cpu"]) else None
            )
            expected.update(temperature_mae=None, conversion_mae=None)
            if row["arm"] in ("EL", "F0", "F1") and x is not None and labels is not None:
                fit = (
                    model
                    if row["arm"] == "EL"
                    else saved_discovery["fits"][saved_discovery["rivals"][int(row["arm"][-1])]]
                )
                d = (7, 13, 23)[fit["family"]]
                op = np.asarray(fit["operator"])
                point = ((x[:, :d] - fit["mean"]) / fit["scale"]) @ op[:-1] + op[-1]
                if np.isfinite(point).all():
                    errors = np.mean(np.abs(labels - point), axis=0)
                    expected.update(
                        temperature_mae=float(errors[0]), conversion_mae=float(errors[1])
                    )
            require(set(row) == set(expected) | {"root", "arm"}, "campaign endpoint fields")
            for key, value in expected.items():
                actual = row[key]
                require(
                    actual == value
                    if value is None or isinstance(value, bool)
                    else actual is not None and np.isclose(actual, value, rtol=1e-11, atol=1e-12),
                    f"raw campaign endpoint differs: {root}/{row['arm']}/{key}",
                )
            checked_rows.append(row)
        prospective_evaluation = analysis['prospective_evaluation']["value"]
        require(
            prospective_evaluation["root"] == root
            and prospective_evaluation["J"] == rows[0]["task_success"]
            and prospective_evaluation["C"] == (prospective_evaluation["A"] and prospective_evaluation["J"]),
            "campaign native J and owner C join",
        )
    contrasts = json.loads(cohort["contrasts_json"])
    if stop is not None:
        require(cohort['prospective_evaluation'] is None and not contrasts, "unentered cohort invented contrasts")
        return dict(independent_roots=8, raw_episodes_checked=0, contrasts_checked=0)
    return {**check_paired_contrasts(checked_rows, contrasts), "nominal_episodes_checked": 72}


if __name__ == "__main__":
    main()

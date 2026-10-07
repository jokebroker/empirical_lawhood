# SPDX-License-Identifier: MPL-2.0

"""Native K and conversion-fraction prediction diagnostics. 24 original roots (16 fit,
eight nomination), five episodes, two views, 2,880 ten-second decisions; no fits or new
integrations.

Callers authenticate retained inputs and supply analysis authority separately.
Historical operators and their custody remain external. No storage effects.
"""

from __future__ import annotations
from typing import Any
import numpy as np
from .experiment_records import EmpiricalAcquisitionEnvelope

PHASES = ((0, 600), (600, 9000), (9000, 18000), (18000, 25200), (25200, 28800))
EPISODES = ("exploration-unshifted", "exploration-shifted-one-position", "exploration-shifted-two-positions", "feed-intervention", "jacket-intervention")
ROOTS = tuple(
    f"reactor-empirical-{role}-{i:03d}"
    for role, n in (("fit", 16), ("nomination", 8))
    for i in range(n)
)


def predict(fit: dict[str, Any], x: np.ndarray) -> np.ndarray:
    """Recompute saved operator predictions without the producer predictor."""
    d = len(fit["mean"])
    z = (x[:, :d] - fit["mean"]) / fit["scale"]
    return np.asarray(z @ np.asarray(fit["operator"][:-1]) + fit["operator"][-1])


def metrics(error: np.ndarray) -> dict[str, Any]:
    """Rows x views x receivers; signs are prediction minus observation."""
    if error.ndim != 3 or not len(error) or not np.isfinite(error).all():
        raise ValueError("Finite nonempty row/view/receiver errors required")
    nominal = error[:, 0]
    return dict(
        rows=len(error),
        rmse=np.sqrt(np.mean(nominal**2, axis=0)).tolist(),
        bias=nominal.mean(axis=0).tolist(),
        centered_rmse=nominal.std(axis=0).tolist(),
        maximum=np.abs(error).max(axis=(0, 1)).tolist(),
        absolute_quantiles=np.quantile(
            np.abs(nominal), [0.5, 0.9, 0.95, 0.99], axis=0
        ).tolist(),
        rows_exceeding_maximum=np.any(np.abs(error) > (0.25, 0.01), axis=1)
        .sum(axis=0)
        .tolist(),
    )


def support(
    fit: dict[str, Any], a: dict[str, np.ndarray]
) -> tuple[np.ndarray, dict[str, Any]]:
    x = a["features"][:, : len(fit["mean"])]
    strata = np.where(a["clocks"] < 600, 0, np.where(a["clocks"] < 25200, 1, 2))
    supported = np.zeros(len(x), dtype=bool)
    failures = np.zeros(x.shape[1], dtype=int)
    census = []
    for c in fit["support"]:
        m = (strata == c["clock_stratum"]) & (a["actions"] == c["action"])
        inside = (x[m] >= c["lower"]) & (x[m] <= c["upper"])
        enough = len(set(c["roots"])) >= 8
        supported[m] = inside.all(axis=1) & enough
        nom = a["root_index"][m] >= 16
        failures += (~inside[nom]).sum(axis=0)
        census.append(
            dict(
                stratum=c["clock_stratum"],
                action=c["action"],
                fit_roots=len(c["roots"]),
                nomination_rows=int(nom.sum()),
                unsupported_nomination=int((~supported[m][nom]).sum()),
            )
        )
    bad = (a["root_index"] >= 16) & ~supported
    return supported, dict(
        cells=census,
        feature_failures=dict(zip(fit["columns"], failures.tolist())),
        unsupported_rows=int(bad.sum()),
        roots=sorted(set(a["root_index"][bad].tolist())),
        clocks=sorted(set(a["clocks"][bad].tolist())),
    )


def branch_error(predicted: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    """One contrast per root; retain the zero-response competitor explicitly."""
    err = predicted[:, None, :] - actual
    answer = metrics(err)
    zero = np.mean(actual[:, 0] ** 2, axis=0)
    mse = np.mean(err[:, 0] ** 2, axis=0)
    answer.update(
        zero_rmse=np.sqrt(zero).tolist(),
        mse_ratio_to_zero=[
            float(e / z) if z else None for e, z in zip(mse, zero, strict=True)
        ],
        wins_zero=(err[:, 0] ** 2 < actual[:, 0] ** 2).sum(axis=0).tolist(),
        maximum_actual=np.abs(actual).max(axis=(0, 1)).tolist(),
        maximum_predicted=np.abs(predicted).max(axis=0).tolist(),
    )
    return answer


def analyze(a: dict[str, np.ndarray], v: dict[str, Any]) -> dict[str, Any]:
    n = 24 * 5 * 2880
    if not (
        np.array_equal(a["root_index"], np.repeat(np.arange(24), 5 * 2880))
        and np.array_equal(
            a["episode_index"], np.tile(np.repeat(np.arange(5), 2880), 24)
        )
        and np.array_equal(a["clocks"], np.tile(np.arange(2880) * 10, 120))
    ):
        raise ValueError("Unexpected root/episode/clock alignment")
    if len(a["features"]) != n or len(v["fits"]) != 9 or v["selected"] is not None:
        raise ValueError("Wrong discovery identity")
    outputs = []
    anchors = np.asarray([3600, 9000, 18000, 24000] * 6)
    donor = (np.arange(24) * 5 + 1) * 2880 + anchors // 10
    for f, frozen in zip(v["fits"], v["nominations"], strict=True):
        p = predict(f, a["features"])
        err = p[:, None, :] - a["labels"][:, :, :2]
        supported, supp = support(f, a)
        fit_mask, nom = a["root_index"] < 16, a["root_index"] >= 16
        train, nomination = metrics(err[fit_mask]), metrics(err[nom])
        np.testing.assert_allclose(
            nomination["rmse"], frozen["rmse"], rtol=1e-10, atol=1e-12
        )
        np.testing.assert_allclose(
            nomination["maximum"], frozen["maximum"], rtol=1e-10, atol=1e-12
        )
        row = dict(
            family=f["family"],
            penalty=f["penalty"],
            train=train,
            nomination=nomination,
            support=supp,
            in_support_nomination=metrics(err[nom & supported]),
            per_root=[],
            per_phase=[],
            per_episode=[],
            extremes=[],
            contrasts=[],
        )
        for i, name in enumerate(ROOTS):
            row["per_root"].append(
                dict(root=name, **metrics(err[a["root_index"] == i]))
            )
        for lo, hi in PHASES:
            m = nom & (a["clocks"] >= lo) & (a["clocks"] < hi)
            row["per_phase"].append(dict(start=lo, end=hi, **metrics(err[m])))
        for i, name in enumerate(EPISODES):
            row["per_episode"].append(
                dict(episode=name, **metrics(err[nom & (a["episode_index"] == i)]))
            )
        ids = np.flatnonzero(nom)
        for h, name in enumerate(("Tpeak10_K", "Cend10_fraction")):
            loc, view = np.unravel_index(np.abs(err[nom, :, h]).argmax(), (len(ids), 2))
            k = ids[loc]
            row["extremes"].append(
                dict(
                    receiver=name,
                    root=ROOTS[a["root_index"][k]],
                    episode=EPISODES[a["episode_index"][k]],
                    clock=float(a["clocks"][k]),
                    view=int(view),
                    prediction=float(p[k, h]),
                    truth=float(a["labels"][k, view, h]),
                    signed_error=float(err[k, view, h]),
                    supported=bool(supported[k]),
                )
            )
        for e, name in ((3, "feed-intervention"), (4, "jacket-intervention")):
            branch = (np.arange(24) * 5 + e) * 2880 + anchors // 10
            # Only action and its interactions may change at the shared anchor.
            fixed = [i for i in range(23) if i not in (5, 6, 11, 12, 21, 22)]
            np.testing.assert_array_equal(
                a["features"][branch][:, fixed], a["features"][donor][:, fixed]
            )
            dp = p[branch] - p[donor]
            dy = a["labels"][branch, :, :2] - a["labels"][donor, :, :2]
            contact = np.any(
                a["features"][branch, 5:7] != a["features"][donor, 5:7], axis=1
            )
            details = []
            for i in range(24):
                details.append(
                    dict(
                        root=ROOTS[i],
                        anchor=int(anchors[i]),
                        contact=bool(contact[i]),
                        predicted=dp[i].tolist(),
                        measured=dy[i].tolist(),
                        both_supported=bool(
                            supported[branch[i]] and supported[donor[i]]
                        ),
                    )
                )
            groups = {}
            for role, m in (
                ("fit", np.arange(24) < 16),
                ("nomination", np.arange(24) >= 16),
            ):
                for label, mask in (("assigned", m), ("contact", m & contact)):
                    groups[f"{role}_{label}"] = branch_error(dp[mask], dy[mask])
            row["contrasts"].append(dict(branch=name, groups=groups, roots=details))
        outputs.append(row)
    return dict(
        outcome_visible=True,
        promotion=False,
        native_runs=0,
        fits=0,
        independent_roots=24,
        fit_roots=16,
        nomination_roots=8,
        paired_rows=n,
        numerical_maximum=np.abs(a["labels"][:, 0] - a["labels"][:, 1])
        .max(axis=0)
        .tolist(),
        all_delivery_valid=bool(a["delivery_valid"].all()),
        candidates=outputs,
    )


def audit_acquisitions(
    acquisitions: dict[str, EmpiricalAcquisitionEnvelope], v: dict[str, Any]
) -> dict[str, Any]:
    """Audit already authenticated typed acquisitions under separate authority."""
    if set(acquisitions) != set(ROOTS):
        raise ValueError("Empirical post-hoc audit requires all 24 original roots")
    episodes, branches = [], []
    for i, name in enumerate(ROOTS):
        env = acquisitions[name]
        arrays = {(e.episode, e.refined): e.arrays.unpack() for e in env.episodes}
        anchor = (3600, 9000, 18000, 24000)[i % 4]
        for (label, refined), raw in arrays.items():
            g = raw["grid"]
            peak = float(g[:, 1].max())
            joint = (g[:, 3] >= 0.999 * 287.3) & (g[:, 4] >= 0.98)
            episodes.append(
                dict(
                    root=name,
                    episode=label,
                    refined=refined,
                    peak_K=peak,
                    final_dose_kg=float(g[-1, 3]),
                    final_conversion=float(g[-1, 4]),
                    dose_at_window_end_kg=float(g[np.searchsorted(g[:, 0], 25200), 3]),
                    first_joint_dose_conversion_s=float(g[np.flatnonzero(joint)[0], 0])
                    if joint.any()
                    else None,
                    measured_receiver_gates=bool(peak <= 356.2 and joint.any()),
                    benchmark_grade=False,
                )
            )
        for branch in ("feed-intervention", "jacket-intervention"):
            views: list[dict[str, Any]] = []
            for refined in (False, True):
                donor, arm = arrays[("exploration-shifted-one-position", refined)], arrays[(branch, refined)]
                before = donor["grid"][:, 0] <= anchor
                np.testing.assert_array_equal(
                    donor["grid"][before, :6], arm["grid"][before, :6]
                )
                np.testing.assert_array_equal(
                    donor["observations"][: anchor // 10 + 1],
                    arm["observations"][: anchor // 10 + 1],
                )
                changed = np.any(donor["exposure"] != arm["exposure"], axis=(1, 2))
                differences = []
                for h in (10, 60, 300):
                    k = np.searchsorted(donor["grid"][:, 0], anchor + h)
                    differences.append(
                        (arm["grid"][k, [1, 4]] - donor["grid"][k, [1, 4]]).tolist()
                    )
                views.append(
                    dict(
                        refined=refined,
                        contact=bool(changed[anchor // 10]),
                        changed_exposure_clocks=(np.flatnonzero(changed) * 10).tolist(),
                        dose_at_anchor=float(donor["observations"][anchor // 10, 3]),
                        endpoint_differences=differences,
                    )
                )
            retained = next(
                p
                for p in v["branch_readout"]["pairs"]
                if p["root"] == name and p["branch"] == branch
            )
            np.testing.assert_allclose(
                views[0]["endpoint_differences"],
                retained["differences"],
                rtol=0,
                atol=1e-14,
            )
            if views[0]["contact"] != retained["contact"]:
                raise ValueError("Retained intervention contact differs")
            branches.append(dict(root=name, branch=branch, anchor=anchor, views=views))
    return dict(
        episodes=episodes,
        branches=branches,
        independent_roots=24,
        total_episodes=len(episodes),
        protected_benchmark_episodes=0,
    )

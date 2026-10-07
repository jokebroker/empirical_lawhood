# SPDX-License-Identifier: MPL-2.0
"""Outcome-visible information-response diagnostics on retained scalar arrays.

These reductions preserve the parent operator's two-future, two-view, seven-arm
contract. They do not establish a new law, source identity or execution authority.
Callers must authenticate and authorize retained inputs before calling them.
"""

import numpy as np
from scipy.spatial.distance import cdist
from .models import MODEL_IDS, FITTED_IDS, features
from empirical_lawhood.adapters.methods.response_formalization import (
    fit_affine_operator,
    affine_prediction,
)

PRIMARY = (2, 4)
RIDGES = (10.0, 1.0, 0.1)
SEED = 20260913
CEILING = "POST_HOC_OUTCOME_VISIBLE_NONPROMOTABLE"


def root_loss(predicted, observed, target="G", primary=True):
    """Pred root,view,model,parent,horizon,receiver,direction; obs adds future."""
    if target == "C":
        predicted = predicted[:, :, :, 1:] - predicted[:, :, :, :1]
        observed = observed[:, :, :, 1:] - observed[:, :, :, :1]
    if primary:
        predicted = np.take(predicted, PRIMARY, axis=4)
        observed = np.take(observed, PRIMARY, axis=4)
    error = predicted[:, :, :, None] - observed[:, :, None]
    return np.mean(error**2, axis=(3, 4, 5, 6, 7))


def interval(values, indices):
    values = np.asarray(values)
    means = values[indices].mean(axis=1)
    return {
        "mean": values.mean(axis=0),
        "lower": np.quantile(means, 0.025, axis=0),
        "upper": np.quantile(means, 0.975, axis=0),
    }


def pair_table(losses, indices):
    rows = []
    for a in range(7):
        for b in range(a + 1, 7):
            difference = losses[:, :, a] - losses[:, :, b]
            primary = difference[:, 0]
            drop = (primary.sum() - primary) / (len(primary) - 1)
            rows.append(
                {
                    "first": MODEL_IDS[a],
                    "second": MODEL_IDS[b],
                    "first_minus_second_mse": interval(difference, indices),
                    "first_concordant_wins": int((difference < 0).all(1).sum()),
                    "second_concordant_wins": int((difference > 0).all(1).sum()),
                    "first_mean_loss_reduction": 1
                    - losses[:, :, a].mean(0) / losses[:, :, b].mean(0),
                    "leave_one_root_out_difference_range": [drop.min(), drop.max()],
                    "largest_absolute_effect_roots": np.argsort(abs(primary))[-3:][
                        ::-1
                    ],
                    "ceiling": CEILING,
                }
            )
    return rows


def array_fit(x, y, ridge):
    "information response prediction parent centering with the existing numeric fitter, on measured G only."
    mean = x.mean(0)
    centered = (x - mean).reshape(-1, x.shape[-1])
    scale = np.maximum(centered.std(0), 1e-12)
    baseline = y.mean(0)
    operator = fit_affine_operator(
        centered / scale, (y - baseline).reshape(-1, int(np.prod(y.shape[2:]))), ridge
    )
    return mean, scale, baseline, operator


def array_predict(model, x):
    mean, scale, baseline, operator = model
    return baseline + affine_prediction(
        operator, ((x - mean) / scale).reshape(-1, x.shape[-1])
    ).reshape(len(x), *baseline.shape)


def target_mse(predicted, observed, target):
    # shape root,parent,horizon,receiver,direction
    if target == "C":
        predicted = predicted[:, 1:] - predicted[:, :1]
        observed = observed[:, 1:] - observed[:, :1]
    return np.mean((predicted[..., PRIMARY, :, :] - observed[..., PRIMARY, :, :]) ** 2)


def fit_predict(x, y, prior, train, test_x, test_prior, arm, ridge):
    if arm == "mechanism":
        return test_prior.copy()
    if arm == "training-mean":
        return np.broadcast_to(y[train].mean(0), (len(test_x), *y.shape[1:])).copy()
    residual = y - prior if arm == "mechanism-with-residual" else y
    model = array_fit(x[train], residual[train], ridge)
    return array_predict(model, test_x) + (test_prior if arm == "mechanism-with-residual" else 0)


def select_ridge(x, y, prior, train, arm, target="G"):
    if arm not in FITTED_IDS:
        return None, []
    losses = []
    for ridge in RIDGES:
        held = np.empty((len(train), *y.shape[1:]))
        for fold in range(3):
            local_test = np.flatnonzero(np.arange(len(train)) % 3 == fold)
            local_train = np.flatnonzero(np.arange(len(train)) % 3 != fold)
            ti, tr = train[local_test], train[local_train]
            held[local_test] = fit_predict(
                x, y, prior, tr, x[ti], prior[ti], arm, ridge
            )
        losses.append(target_mse(held, y[train], target))
    return RIDGES[int(np.argmin(losses))], losses


def feature_arrays(history, sketch, arm):
    return features(history.reshape(-1, 16, 12), sketch.reshape(-1, 8), arm).reshape(
        *history.shape[:-2], -1
    )


def energy_test(old, new, permutations=999):
    combined = np.concatenate((old, new))
    distance = cdist(combined, combined) / np.sqrt(combined.shape[1])
    n, total = len(old), len(combined)

    def statistic(order):
        a, b = order[:n], order[n:]
        return (
            2 * distance[np.ix_(a, b)].mean()
            - distance[np.ix_(a, a)].mean()
            - distance[np.ix_(b, b)].mean()
        )

    actual = statistic(np.arange(total))
    rng = np.random.default_rng(SEED)
    exceed = sum(
        statistic(rng.permutation(total)) >= actual for _ in range(permutations)
    )
    return {
        "energy_distance": actual,
        "permutation_tail_descriptive": (exceed + 1) / (permutations + 1),
        "permutations": permutations,
    }


def matrix_metrics(g):
    singular = np.linalg.svd(g, compute_uv=False)
    norm2 = np.sum(g * g, axis=(-2, -1))
    return {
        "singular_values": singular,
        "condition_ratio": singular[..., 0] / np.maximum(singular[..., 1], 1e-15),
        "off_diagonal_fraction": (g[..., 0, 1] ** 2 + g[..., 1, 0] ** 2)
        / np.maximum(norm2, 1e-30),
        "antisymmetric_fraction": np.sum(
            ((g - g.swapaxes(-1, -2)) / 2) ** 2, axis=(-2, -1)
        )
        / np.maximum(norm2, 1e-30),
        "determinant": np.linalg.det(g),
    }


def decision_arrays(predicted, observed):
    angles = np.arange(16) * np.pi / 8
    directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    words_f = np.stack(
        (-predicted[..., 0], predicted[..., 0], -predicted[..., 1], predicted[..., 1]),
        axis=-1,
    )
    words_g = np.stack(
        (-observed[..., 0], observed[..., 0], -observed[..., 1], observed[..., 1]),
        axis=-1,
    )
    fscore = np.einsum("...rw,dr->...dw", words_f, directions)
    gscore = np.einsum("...rw,dr->...dw", words_g, directions)
    choice = fscore.argmax(-1)
    actual = np.take_along_axis(
        gscore[:, :, None], choice[:, :, :, None, ..., None], axis=-1
    )[..., 0]
    oracle = gscore.max(-1)
    regret = oracle[:, :, None] - actual
    margin = np.sort(gscore, axis=-1)[..., -1] - np.sort(gscore, axis=-1)[..., -2]
    return choice, regret, margin, gscore.argmax(-1)


def model_inputs(history, sketch):
    result = {
        arm: np.stack(
            [feature_arrays(history[:, :, v], sketch[:, :, v], arm) for v in range(2)],
            axis=1,
        )
        for arm in ("snapshot", "history", "snapshot-with-mechanism", "history-with-mechanism")
    }
    result["mechanism-with-residual"] = result["history-with-mechanism"]
    result["training-mean"] = result["snapshot"]
    result["mechanism"] = result["snapshot"]
    return result


def crossfit(x, y, prior, *, size=24, order=None, target="G"):
    n = len(y)
    predictions = np.empty((n, 2, 7, 5, 5, 2, 2))
    selections = []
    for fold in range(4):
        test = np.flatnonzero(np.arange(n) % 4 == fold)
        train = np.flatnonzero(np.arange(n) % 4 != fold)
        if order is not None:
            train = np.random.default_rng(SEED + 100 * order + fold).permutation(train)
        train = train[:size]
        assert not set(train) & set(test)
        for mi, arm in enumerate(MODEL_IDS):
            ridge, scores = select_ridge(
                x[arm][:, 0], y, prior[:, 0], train, arm, target
            )
            for vi in range(2):
                predictions[test, vi, mi] = fit_predict(
                    x[arm][:, 0],
                    y,
                    prior[:, 0],
                    train,
                    x[arm][test, vi],
                    prior[test, vi],
                    arm,
                    ridge,
                )
            selections.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "train_roots": train,
                    "test_roots": test,
                    "ridge": ridge,
                    "inner_scores": scores,
                }
            )
    return predictions, selections


def prefix_crossfit(history, sketch, y):
    output = np.empty((32, 2, 5, 5, 5, 2, 2))
    selections = []
    for fold in range(4):
        train = np.flatnonzero(np.arange(32) % 4 != fold)
        test = np.flatnonzero(np.arange(32) % 4 == fold)
        output[test, :, 0] = y[train].mean(0)
        for mi, arm in enumerate(("snapshot", "history", "snapshot-with-mechanism", "history-with-mechanism"), start=1):
            x = np.stack(
                [features(history[:, v], sketch[:, v], arm) for v in range(2)], axis=1
            )
            scores = []
            for ridge in RIDGES:
                held = np.empty((len(train), 5, 5, 2, 2))
                for inner in range(3):
                    ti = np.flatnonzero(np.arange(len(train)) % 3 == inner)
                    tr = train[np.arange(len(train)) % 3 != inner]
                    model = array_fit(x[tr, 0, None], y[tr, None], ridge)
                    held[ti] = array_predict(model, x[train[ti], 0, None])[:, 0]
                scores.append(target_mse(held, y[train], "G"))
            ridge = RIDGES[int(np.argmin(scores))]
            model = array_fit(x[train, 0, None], y[train, None], ridge)
            for vi in range(2):
                output[test, vi, mi] = array_predict(model, x[test, vi, None])[:, 0]
            selections.append(
                {
                    "fold": fold,
                    "arm": arm,
                    "ridge": ridge,
                    "inner_scores": scores,
                    "train_roots": train,
                    "test_roots": test,
                }
            )
    return output, selections

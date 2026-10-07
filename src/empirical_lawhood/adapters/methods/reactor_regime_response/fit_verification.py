# SPDX-License-Identifier: MPL-2.0

"""Independent root-weighted local/affine/RBF coefficient reconstruction. Fixed four folds,
12 distinct roots per leaf, native cooling K and delivered mass kg. No producer fitting
implementation is shared.

Callers authenticate retained inputs and supply analysis authority separately.
Historical operators and their custody remain external. No storage effects.
"""

from __future__ import annotations
from collections import Counter
import numpy as np
from empirical_lawhood.adapters.methods.reactor_regime_response.discovery_records import RegimeDiscoveryTraining
from empirical_lawhood.adapters.methods.reactor_regime_response.model_records import RegimeFitPackage
from empirical_lawhood.kernel.provenance import ObjectIdentity
from .discovery_records import RegimeDiscoveryDevelopment
from .nomination_records import RegimeNominationPackage


PathSpec = tuple[tuple[int, float, bool], ...]
Proposal = tuple[tuple[bool, ...], int, float]


def _threshold(values: np.ndarray, q: float) -> float:
    values = np.unique(values)
    if len(values) < 2:
        raise ValueError("no distinct threshold")
    i = min(len(values) - 2, max(0, int(q * (len(values) - 1))))
    return float((values[i] + values[i + 1]) / 2)


def _weights(roots: tuple[str, ...]) -> np.ndarray:
    counts = Counter(roots)
    return np.asarray([1 / counts[root] for root in roots])


def _moments(x: np.ndarray, roots: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    weights = _weights(roots)
    weights /= weights.sum()
    mean = np.sum(x * weights[:, None], axis=0)
    return mean, np.maximum(
        np.sqrt(np.sum((x - mean) ** 2 * weights[:, None], axis=0)), 1e-12
    )


def _linear(z: np.ndarray, y: np.ndarray, w: np.ndarray, penalty: float) -> np.ndarray:
    matrix = np.c_[np.ones(len(z)), z]
    regularizer = np.diag([0.0, *([penalty] * z.shape[1])])
    return np.linalg.solve(
        matrix.T @ (w[:, None] * matrix) + regularizer, matrix.T @ (w * y)
    )


def independent_local(
    x: np.ndarray,
    masses: np.ndarray,
    contrasts: np.ndarray,
    roots: tuple[str, ...],
    penalty: float,
) -> tuple[tuple[PathSpec, np.ndarray], ...]:
    """Reconstruct split proposals and CV losses without the discovery fitter."""
    target = np.sum(masses * contrasts, axis=1) / np.sum(masses**2, axis=1)
    folds = np.asarray([int(root.rsplit("-", 1)[-1]) % 4 for root in roots])
    if set(folds) != {0, 1, 2, 3}:
        raise ValueError("four assigned folds unavailable")

    def tree(
        indices: np.ndarray, proposals: tuple[Proposal, ...]
    ) -> tuple[np.ndarray, np.ndarray, tuple[tuple[PathSpec, np.ndarray], ...]]:
        features = x[indices]
        names = tuple(roots[int(i)] for i in indices)
        mean, scale = _moments(features, names)
        z = (features - mean) / scale
        weights = _weights(names) * np.mean(masses[indices] ** 2, axis=1)
        leaves: dict[tuple[bool, ...], tuple[np.ndarray, PathSpec]] = {
            (): (np.ones(len(indices), dtype=bool), ())
        }
        for parent, coordinate, q in proposals:
            mask, path = leaves.pop(parent)
            boundary = _threshold(features[mask, coordinate], q)
            values = features[mask, coordinate]
            if any(
                np.mean((values <= boundary) != (values + shift <= boundary)) > 0.01
                for shift in (-1e-8, 1e-8)
            ):
                raise ValueError("arithmetic instability")
            for side in (True, False):
                child = mask & ((features[:, coordinate] <= boundary) == side)
                if (
                    len({name for name, keep in zip(names, child, strict=True) if keep})
                    < 12
                ):
                    raise ValueError("child lacks distinct roots")
                leaves[parent + (side,)] = (
                    child,
                    path + ((coordinate, boundary, side),),
                )
        return (
            mean,
            scale,
            tuple(
                (path, _linear(z[mask], target[indices][mask], weights[mask], penalty))
                for mask, path in leaves.values()
            ),
        )

    def loss(proposals: tuple[Proposal, ...], parent: tuple[bool, ...] = ()) -> float:
        losses = []
        for fold in range(4):
            training, held = (
                np.flatnonzero(folds != fold),
                np.flatnonzero(folds == fold),
            )
            mean, scale, leaves = tree(training, proposals)
            predicted = np.empty(len(held))
            selected = np.zeros(len(held), dtype=bool)
            for path, operator in leaves:
                mask = np.ones(len(held), dtype=bool)
                for coordinate, boundary, side in path:
                    mask &= (x[held, coordinate] <= boundary) == side
                predicted[mask] = (
                    np.c_[np.ones(mask.sum()), (x[held[mask]] - mean) / scale]
                    @ operator
                )
                if (
                    not parent
                    or tuple(value[2] for value in path[: len(parent)]) == parent
                ):
                    selected |= mask
            errors = np.mean(
                (predicted[:, None] * masses[held] - contrasts[held]) ** 2, axis=1
            )
            for root_name in sorted({roots[int(i)] for i in held[selected]}):
                mask = selected & np.asarray([roots[int(i)] == root_name for i in held])
                losses.append(float(np.mean(errors[mask])))
        if not losses:
            raise ValueError("no held-out parent contact")
        return float(np.mean(losses))

    chosen: tuple[Proposal, ...] = ()
    for _ in range(2):
        baseline = loss(chosen)
        options = []
        for parent in ((),) if not chosen else ((True,), (False,)):
            before = loss(chosen, parent)
            for coordinate in range(x.shape[1]):
                for q in (0.25, 0.5, 0.75):
                    proposal = (parent, coordinate, q)
                    try:
                        proposed = chosen + (proposal,)
                        after = loss(proposed, parent)
                        gain = baseline - loss(proposed)
                        _, _, full = tree(np.arange(len(roots)), proposed)
                    except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                        continue
                    if (
                        gain > 0
                        and before - after > 1e-12
                        and before - after >= 0.1 * before
                    ):
                        boundary = next(
                            path[len(parent)][1]
                            for path, _ in full
                            if len(path) > len(parent)
                            and tuple(value[2] for value in path[: len(parent)])
                            == parent
                        )
                        options.append(
                            (-gain, coordinate, boundary, parent, q, proposal)
                        )
        if not options:
            break
        chosen += (min(options)[-1],)
    if not chosen:
        raise ValueError("no qualifying split")
    return tuple(
        sorted(tree(np.arange(len(roots)), chosen)[2], key=lambda item: item[0])
    )


def check_fit_arithmetic(
    fit: RegimeFitPackage, training: RegimeDiscoveryTraining
) -> dict[str, object]:
    if training.fit_package != ObjectIdentity.from_record(fit.package_id, fit):
        raise ValueError("saved training is not the receipted fit source")
    contexts = tuple(item.to_context() for item in training.contexts)
    checked = []
    local_checked = []
    for candidate in (*fit.coefficients, *fit.absolute):
        frozen = candidate.fitted
        if frozen is None:
            continue
        absolute = candidate.model_id.startswith('absolute-observable.')
        rows = tuple(
            row
            for row in contexts
            if row.valid
            and (
                row.causal_current
                if candidate.mask == "current"
                else row.causal_history
            )
            is not None
            and (
                row.nominal_peak_K is not None
                if absolute
                else row.nominal_cooling_K is not None
                and row.nominal_mass_kg is not None
            )
        )
        x = np.asarray(
            [
                row.causal_current
                if candidate.mask == "current"
                else row.causal_history
                for row in rows
            ]
        )
        roots = tuple(row.root for row in rows)
        if not roots:
            raise ValueError(
                "saved fitted model has no valid assigned training contexts"
            )
        context_weights = _weights(roots)
        if absolute:
            target = np.asarray(
                [
                    row.nominal_peak_K[0]
                    for row in rows
                    if row.nominal_peak_K is not None
                ]
            )
            weights = context_weights
        else:
            masses = np.asarray(
                [
                    row.nominal_mass_kg[1:]
                    for row in rows
                    if row.nominal_mass_kg is not None
                ]
            )
            contrasts = np.asarray(
                [
                    row.nominal_cooling_K[1:]
                    for row in rows
                    if row.nominal_cooling_K is not None
                ]
            )
            target = np.sum(masses * contrasts, axis=1) / np.sum(masses**2, axis=1)
            weights = context_weights * np.mean(masses**2, axis=1)
        low, high = x.min(axis=0), x.max(axis=0)
        np.testing.assert_allclose(
            np.asarray(frozen.support_lower, dtype=float),
            low - 0.05 * (high - low),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            np.asarray(frozen.support_upper, dtype=float),
            high + 0.05 * (high - low),
            rtol=0,
            atol=1e-12,
        )
        mean, scale = _moments(x, roots)
        if candidate.family == "K":
            expected = np.asarray([np.sum(weights * target) / weights.sum()])
        else:
            np.testing.assert_allclose(
                np.asarray(frozen.mean, dtype=float), mean, rtol=0, atol=1e-12
            )
            np.testing.assert_allclose(
                np.asarray(frozen.scale, dtype=float), scale, rtol=1e-12, atol=1e-12
            )
            z = (x - mean) / scale
            if candidate.family == "affine":
                expected = _linear(z, target, weights, float(candidate.penalty))
            elif candidate.family == "rbf":
                distance2 = np.maximum(
                    np.sum(z * z, axis=1)[:, None]
                    + np.sum(z * z, axis=1)[None, :]
                    - 2 * z @ z.T,
                    0,
                )
                distances = np.sqrt(distance2[np.triu_indices(len(z), 1)])
                expected_length = float(candidate.multiplier or 0) * np.median(
                    distances[distances > 1e-12]
                )
                np.testing.assert_allclose(
                    float(frozen.length or 0), expected_length, rtol=1e-12
                )
                np.testing.assert_allclose(
                    np.asarray(frozen.training, dtype=float), z, rtol=1e-12, atol=1e-12
                )
                # The coefficient fitter materializes Euclidean distances
                # before squaring them; the absolute fitter uses distance2.
                # Preserve those floating-point operations at the existing
                # coefficient tolerance, without sharing producer code.
                kernel_distance2 = distance2 if absolute else np.sqrt(distance2) ** 2
                kernel = np.exp(-kernel_distance2 / (2 * expected_length**2))
                expected = np.linalg.solve(
                    kernel + np.diag(float(candidate.penalty) / weights), target
                )
            else:
                assert not absolute
                leaves = independent_local(
                    x, masses, contrasts, roots, float(candidate.penalty)
                )
                saved = tuple(sorted(frozen.leaves, key=lambda leaf: leaf.path))
                if len(leaves) != len(saved):
                    raise ValueError(
                        "independent fourfold discovery disagrees on leaf count"
                    )
                for (path, operator), leaf in zip(leaves, saved, strict=True):
                    if path != tuple((i, float(t), left) for i, t, left in leaf.path):
                        raise ValueError(
                            "independent split search disagrees with the frozen path"
                        )
                    np.testing.assert_allclose(
                        np.asarray(leaf.operator, dtype=float),
                        operator,
                        rtol=1e-7,
                        atol=1e-10,
                    )
                local_checked.append(candidate.model_id)
                checked.append(candidate.model_id)
                continue
        np.testing.assert_allclose(
            np.asarray(frozen.operator, dtype=float), expected, rtol=1e-7, atol=1e-10
        )
        checked.append(candidate.model_id)
    return {
        "fit_sha256": fit.fingerprint(),
        "training_sha256": training.fingerprint(),
        "checked_fitted_models": checked,
        "independent_local_searches": local_checked,
        "assigned_roots": 32,
        "assigned_contexts": len(contexts),
        "native_calls": 0,
    }


def verify_saved_development(
    fit: RegimeFitPackage,
    training: RegimeDiscoveryTraining,
    nomination: RegimeNominationPackage,
    development: RegimeDiscoveryDevelopment,
) -> dict[str, object]:
    """Check authenticated frozen fits and all original ordinary-root draws.

    The caller authenticates original successful fit/nomination receipts and
    their saved training/development operands, and supplies analysis authority.
    This exposed-data readout verifies 200 draws of 32 roots with the original
    seed and independent coefficient arithmetic. It creates no qualification.
    """
    if development.nomination != ObjectIdentity.from_record(
        nomination.package_id, nomination
    ):
        raise ValueError("bootstrap report changed the original nomination")
    draws = np.random.default_rng(20260923).integers(0, 32, size=(200, 32))
    for value in development.local_stability:
        if [item.root_ordinals for item in value.refits] != [
            tuple(item) for item in draws
        ]:
            raise ValueError(
                "saved bootstrap substituted its original ordinary-root draws"
            )
    result = check_fit_arithmetic(fit, training)
    result.update(
        {
            "development_sha256": development.fingerprint(),
            "bootstrap_models": len(development.local_stability),
            "nomination_sha256": nomination.fingerprint(),
        }
    )
    return result

"""Original whole-root bootstrap discovery; no Bayesian reweighting or refit nominee."""

from __future__ import annotations

from decimal import Decimal as D

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity

from .config import ROOTS
from .cv_local import fit_local_cv
from .discovery_records import RegimeContextMeasurement, RegimeDiscoveryDevelopment, RegimeDiscoveryTraining, RegimeLocalStability, RegimePartitionRefit
from .measured_panel import MeasuredContext
from .model_records import RegimeFitPackage
from .model_selection import _coefficient_rows, _leaf_contacts
from .models import Fit, split_stable
from .nomination_records import RegimeNominationPackage
from .qualification_comparisons import _nominated_model_id, _scored_candidates


def membership(fit: Fit, x: np.ndarray) -> tuple[int, ...]:
    result = []
    for values in x:
        found = [i for i, leaf in enumerate(fit.leaves)
                 if all((values[j] <= boundary) == left for j, boundary, left in leaf.path)]
        if len(found) != 1:
            raise ValueError("partition does not uniquely cover a nomination context")
        result.append(found[0])
    return tuple(result)


def adjusted_rand(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("adjusted Rand requires the same nonempty context census")
    if len(a) == 1:
        return 1.0
    joint = sum(count * (count - 1) / 2 for count in
                (sum(x == u and y == v for x, y in zip(a, b, strict=True))
                 for u in set(a) for v in set(b)))
    left = sum(a.count(u) * (a.count(u) - 1) / 2 for u in set(a))
    right = sum(b.count(v) * (b.count(v) - 1) / 2 for v in set(b))
    expected = left * right / (len(a) * (len(a) - 1) / 2)
    denominator = (left + right) / 2 - expected
    return 1.0 if denominator == 0 else float((joint - expected) / denominator)


def arithmetic_stable(fit: Fit, x: np.ndarray) -> bool:
    for leaf in fit.leaves:
        parent = np.ones(len(x), dtype=bool)
        for coordinate, boundary, left in leaf.path:
            if parent.any() and not split_stable(x[parent], coordinate, boundary):
                return False
            parent &= (x[:, coordinate] <= boundary) == left
    return True


def build_discovery_development(
    training: RegimeDiscoveryTraining,
    fit: RegimeFitPackage,
    nomination: RegimeNominationPackage,
    contexts: tuple[MeasuredContext, ...],
) -> RegimeDiscoveryDevelopment:
    fit_id = ObjectIdentity.from_record(fit.package_id, fit)
    if training.fit_package != fit_id or nomination.fit_package != fit_id:
        raise ValueError("discovery bootstrap changed its original fit")
    fitted = {value.model_id: value for value in fit.coefficients}
    scored = _scored_candidates(fit, nomination)
    ids = {value.coefficient_model_id for value in nomination.routes
           if value.coefficient_model_id is not None
           and fitted[value.coefficient_model_id].family == "local"}
    for mask in ("current", "history"):
        selected = _nominated_model_id(tuple(value for value in scored
                                           if value.mask == mask and value.family == "local"))
        if selected is not None:
            ids.add(selected)
    fit_roots = tuple(root for root, role, _, _ in ROOTS if role == "fit")
    fit_contexts = tuple(row.to_context() for row in training.contexts)
    draws = np.random.default_rng(20260923).integers(0, 32, size=(200, 32))
    results = []
    for model_id in sorted(ids):
        candidate = fitted[model_id]
        assert candidate.fitted is not None
        original = candidate.fitted.to_fit()
        rows = _coefficient_rows(fit_roots, fit_contexts, candidate.mask)
        points = tuple(row for row in contexts if (row.causal_current if candidate.mask == "current"
                                                  else row.causal_history) is not None)
        features = np.asarray([row.causal_current if candidate.mask == "current"
                               else row.causal_history for row in points], dtype=float)
        original_membership = membership(original, features)
        original_stable = arithmetic_stable(original, rows.features) and arithmetic_stable(original, features)
        refits = []
        for index, draw in enumerate(draws):
            drawn = tuple(fit_roots[int(i)] for i in draw)
            available = tuple(root for root in drawn if root in rows.roots)
            paths: tuple[tuple[tuple[int, D, bool], ...], ...] = ()
            labels: tuple[int, ...] = ()
            contacts: tuple[int, ...] = ()
            ari = None
            stable = False
            reasons: tuple[str, ...] = ()
            try:
                sample = rows.resample_roots(available)
                refitted = fit_local_cv(sample, float(candidate.penalty))
                labels = membership(refitted, features)
                ari = D(repr(adjusted_rand(original_membership, labels)))
                contacts = _leaf_contacts(refitted, sample)
                paths = tuple(tuple((j, D(repr(t)), side) for j, t, side in leaf.path)
                              for leaf in refitted.leaves)
                stable = (len(refitted.leaves) == len(original.leaves) and ari >= D('.8')
                          and arithmetic_stable(refitted, sample.features)
                          and arithmetic_stable(refitted, features))
                if not stable:
                    reasons = ("PARTITION_NOT_REPRODUCED",)
            except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
                reasons = (f"DISCOVERY_REFIT_UNAVAILABLE:{type(error).__name__}:{error}",)
            refits.append(RegimePartitionRefit(
                index, tuple(int(i) for i in draw), len(set(available)), paths, labels,
                contacts, ari, stable, reasons,
            ))
        count = sum(row.stable for row in refits)
        passed = count >= 160 and original_stable
        results.append(RegimeLocalStability(
            model_id, tuple((row.root, row.context) for row in points), original_membership,
            len(original.leaves), original_stable, tuple(refits), count, passed,
            () if passed else ("LOCAL_PARTITION_STABILITY_NOT_ESTABLISHED",),
        ))
    return RegimeDiscoveryDevelopment(
        ObjectIdentity.from_record("regime.discovery-training", training), fit_id,
        ObjectIdentity.from_record(nomination.package_id, nomination), 20260923,
        tuple(results), tuple(RegimeContextMeasurement.from_context(row) for row in contexts),
        () if results else ("NO_NOMINATED_LOCAL_PARTITION_FOR_STABILITY",),
    )

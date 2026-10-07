"""Post-seal measurement projection for all eight prescribed root contexts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from empirical_lawhood.kernel.provenance import ObjectIdentity

from .features import causal_features
from .records import RegimeAssayPanel, RegimeCausalPreparation

CONTEXTS = (
    "early",
    "middle",
    "late",
    "prepared_t0",
    "c_q",
    "c_q600",
    "p_q",
    "p_q600",
)


@dataclass(frozen=True)
class MeasuredContext:
    root: str
    context: str
    callback: int | None
    nominal_mass_kg: tuple[float, float, float] | None
    refined_mass_kg: tuple[float, float, float] | None
    nominal_peak_K: tuple[float, float, float] | None
    refined_peak_K: tuple[float, float, float] | None
    nominal_cooling_K: tuple[float, float, float] | None
    refined_cooling_K: tuple[float, float, float] | None
    coefficient_K_per_kg: float | None
    scalar_residual_K: float | None
    causal_current: tuple[float, ...] | None
    causal_history: tuple[float, ...] | None
    reasons: tuple[str, ...]

    @property
    def valid(self) -> bool:
        # A failed scalar approximation remains a measured two-word response.
        return not (set(self.reasons) - {"SCALAR_COEFFICIENT_INADEQUATE"})


def _empty(root: str, name: str, callback: int | None, reasons: set[str]) -> MeasuredContext:
    return MeasuredContext(
        root,
        name,
        callback,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        tuple(sorted(reasons)),
    )


def measured_contexts(
    causal_record: RegimeCausalPreparation,
    assay_record: RegimeAssayPanel,
) -> tuple[MeasuredContext, ...]:
    if (
        (assay_record.root, assay_record.role, assay_record.seed)
        != (causal_record.root, causal_record.role, causal_record.seed)
        or assay_record.causal_preparation
        != ObjectIdentity.from_record(f"{causal_record.root}.causal-preparation", causal_record)
    ):
        raise ValueError("assay panel changed its sealed causal parent")
    causal, assays = causal_record.arrays.unpack(), assay_record.arrays.unpack()
    slots = {(name, view, word): (callback, status) for name, callback, view, word, status in assay_record.slots}
    failures = {key: reason for key, reason in assay_record.failures}
    results = []
    for name in CONTEXTS:
        base = "exploration_unshifted" if name in CONTEXTS[:4] else name[0]
        callback = slots[(name, 0, 0)][0]
        reasons: set[str] = set()
        if callback is None:
            results.append(_empty(causal_record.root, name, None, {"NO_CONTACT"}))
            continue
        views: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []
        dose_ends: list[tuple[float, float, float]] = []
        for view in (0, 1):
            source_key = f"{base}_v{view}"
            source_observations = causal.get(f"{source_key}_observations")
            source_stages = causal.get(f"{source_key}_stages")
            if (
                source_observations is None
                or source_stages is None
                or len(source_observations) <= callback
                or len(source_stages) < callback
            ):
                reasons.add(f"MISSING_PREPARED_PREFIX_V{view}")
                continue
            previous_jacket = float(source_stages[callback - 1, 3])
            masses = []
            peaks = []
            dose_end = []
            for word, requested in enumerate((0.0, 0.016, 0.032)):
                key = f"{name}_v{view}_a{word}"
                if slots[(name, view, word)] != (callback, "MEASURED"):
                    reasons.add(f"MISSING_ASSAY_V{view}_A{word}")
                    continue
                required = ("grid", "exposure", "stages", "request", "valid", "prefix_sha256")
                if any(f"{key}_{field}" not in assays for field in required):
                    reasons.add(f"MISSING_OUTPUT_V{view}_A{word}")
                    continue
                grid = assays[f"{key}_grid"]
                exposure = assays[f"{key}_exposure"]
                stage = assays[f"{key}_stages"]
                request = assays[f"{key}_request"]
                prefix = assays[f"{key}_prefix_sha256"]
                dt = 0.5 if view else 1.0
                expected_times = callback * 10 + np.arange(int(10 / dt) + 1) * dt
                if (
                    grid.shape != (len(expected_times), 8)
                    or not np.array_equal(grid[:, 0], expected_times)
                    or exposure.shape != (int(10 / dt), 4)
                    or stage.shape != (4,)
                    or request.shape != (2,)
                    or not np.isfinite(grid).all()
                    or not np.isfinite(exposure).all()
                    or not np.isfinite(stage).all()
                    or not np.isfinite(request).all()
                    or assays[f"{key}_valid"].tolist() != [1]
                ):
                    reasons.add(f"INVALID_NATIVE_OUTPUT_V{view}_A{word}")
                    continue
                expected_prefix = np.frombuffer(
                    sha256(source_observations[: callback + 1].tobytes()).digest(), dtype=np.uint8
                )
                if not np.array_equal(prefix, expected_prefix):
                    reasons.add(f"PREPARED_PREFIX_CHANGED_V{view}_A{word}")
                if key in failures:
                    reasons.add(failures[key])
                if (
                    request[0] != requested
                    or request[1] != previous_jacket
                    or stage[3] != previous_jacket
                    or not np.all(exposure[:, 3] == previous_jacket)
                    or np.any(exposure[:, 2] < 0)
                    or (word == 0 and np.any(exposure[:, 2] != 0))
                ):
                    reasons.add(f"FIXED_JACKET_OR_REFERENCE_INVALID_V{view}_A{word}")
                masses.append(float(np.sum(exposure[:, 1] * exposure[:, 2])))
                peaks.append(float(np.max(grid[:, 1])))
                dose_end.append(float(grid[-1, 3]))
            if len(masses) != 3 or len(peaks) != 3:
                continue
            if not masses[0] == 0 < masses[1] < masses[2]:
                reasons.add(f"ALIASED_OR_NONZERO_REFERENCE_V{view}")
            cooling = (0.0, peaks[0] - peaks[1], peaks[0] - peaks[2])
            views.append((tuple(masses), tuple(peaks), cooling))  # type: ignore[arg-type]
            dose_ends.append(tuple(dose_end))  # type: ignore[arg-type]
        if len(views) != 2:
            reasons.add("MISSING_NUMERICAL_VIEW")
            results.append(_empty(causal_record.root, name, callback, reasons))
            continue
        nominal, refined = views
        if any(abs(a - b) > 0.01 for a, b in zip(nominal[1], refined[1], strict=True)):
            reasons.add("ABSOLUTE_NUMERICAL_INVALID")
        if any(abs(a - b) > 1e-6 for a, b in zip(nominal[2], refined[2], strict=True)):
            reasons.add("CONTRAST_NUMERICAL_INVALID")
        if any(abs(a - b) > 1e-6 for a, b in zip(nominal[0], refined[0], strict=True)):
            reasons.add("MASS_NUMERICAL_INVALID")
        if any(abs(a - b) > 1e-6 for a, b in zip(dose_ends[0], dose_ends[1], strict=True)):
            reasons.add("DOSE_NUMERICAL_INVALID")
        masses_np = np.asarray(nominal[0][1:])
        cooling_np = np.asarray(nominal[2][1:])
        kappa = (
            float(np.dot(masses_np, cooling_np) / np.dot(masses_np, masses_np))
            if np.all(masses_np > 0) and masses_np[1] > masses_np[0]
            else None
        )
        residual = (
            float(np.max(np.abs(cooling_np - kappa * masses_np)))
            if kappa is not None
            else None
        )
        if residual is not None and residual > 1e-5:
            reasons.add("SCALAR_COEFFICIENT_INADEQUATE")
        source_observations = causal[f"{base}_v0_observations"]
        source_stages = causal[f"{base}_v0_stages"]
        features = causal_features(source_observations, source_stages, callback)
        results.append(
            MeasuredContext(
                causal_record.root,
                name,
                callback,
                nominal[0],
                refined[0],
                nominal[1],
                refined[1],
                nominal[2],
                refined[2],
                kappa,
                residual,
                tuple(float(x) for x in features[:6]),
                tuple(float(x) for x in features),
                tuple(sorted(reasons)),
            )
        )
    return tuple(results)

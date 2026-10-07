"Bounded reconstruction of frozen development coefficients before calibration; never fitting."

from hashlib import sha256
from io import BytesIO
import json

import numpy as np

from empirical_lawhood.kernel.references import ArtifactIdentity
from .fitting import Array, Normalizer, PointFit, Recipe
from .intervals import WidthFit
from collections.abc import Mapping
import ast
import os
import platform
from typing import Any


def frozen_development(
    *,
    report_bytes: bytes,
    coefficient_bytes: bytes,
    report_identity: ArtifactIdentity,
    coefficient_identity: ArtifactIdentity,
) -> tuple[PointFit, dict[str, WidthFit]]:
    "Consume identities supplied by the frozen stage, not selected from outcomes.\n\n    Input custody and pre-calibration authorization are the provider's separate\n    responsibilities. This decoder authenticates bytes and reproduces the\n    nominated coefficients and scales without consulting calibration.\n    "
    for raw, identity in (
        (report_bytes, report_identity),
        (coefficient_bytes, coefficient_identity),
    ):
        if (
            len(raw) > 2 * 1024**2
            or len(raw) != identity.size_bytes
            or sha256(raw).hexdigest() != identity.sha256
        ):
            raise ValueError(
                "Frozen development bytes differ from the declared artifact"
            )
    report = json.loads(report_bytes)
    recipe = {"dimension": 24, "family": "SNAPSHOT_REFERENCE_SKETCH_AND_RATE", "gamma": 1.0, "ridge": 10.0}
    if (
        report["recipe"] != recipe
        or report["direct_recipe"] != recipe
        or report["training_roots"] != list(range(48))
        or report["held_roots"] != []
        or report["normalizer_prefix_root_ids"] != list(range(48))
        or report["normalizer_handoff_rows"]
        != [[r, p] for r in range(48) for p in range(5)]
    ):
        raise ValueError(
            "Frozen development changes the nominated recipe or training denominator"
        )
    if coefficient_identity.media_type == "application/json":
        from .array_transport import decode_npz_transport

        coefficient_bytes = decode_npz_transport(
            coefficient_bytes, coefficient_identity.payload_schema
        )
    with np.load(BytesIO(coefficient_bytes), allow_pickle=False) as archive:
        infos = archive.zip.infolist()
        if len(infos) > 256 or sum(v.file_size for v in infos) > 64 * 1024**2:
            raise ValueError("Frozen development exceeds its bounded array expansion")

        def array(key: str, shape: tuple[int, ...], *, positive: bool = False) -> Array:
            value = archive[key]
            if (
                value.dtype != np.float64
                or value.shape != shape
                or not np.isfinite(value).all()
                or positive
                and (value <= 0).any()
            ):
                raise ValueError(
                    "Frozen development changes finite coefficient/scale axes"
                )
            return np.frombuffer(value.tobytes(), dtype=np.float64).reshape(shape)

        points = PointFit(
            tuple(range(48)),
            Recipe(**report["recipe"]),
            Recipe(**report["direct_recipe"]),
            Normalizer(
                array("point_n0_center", (24,)),
                array("point_n0_scale", (24,), positive=True),
            ),
            Normalizer(
                array("point_nh_center", (24,)),
                array("point_nh_scale", (24,), positive=True),
            ),
            array("point_lower", (25, 32)),
            array("point_lower_mean", (32,)),
            array("point_upper", (5, 25, 24)),
            array("point_cached", (5, 32)),
            array("point_direct", (5, 25, 32)),
        )
        widths = {}
        for name, multiplier_shape in (
            ("lower", (25, 1)),
            ("composed", (5, 25, 1)),
            ("cached", (5,)),
            ("direct", (5, 25, 1)),
        ):
            target = archive[f"{name}_log_multiplier_target"]
            eligible = archive[f"{name}_log_multiplier_eligible"]
            counts = archive[f"{name}_base_scale_root_counts"]
            if (
                target.shape != (48, 5)
                or target.dtype != np.float64
                or eligible.shape != (48,)
                or eligible.dtype != np.bool_
                or counts.shape != (4, 8)
                or counts.dtype != np.int64
                or (counts <= 0).any()
            ):
                raise ValueError("Frozen development changes its scale-training census")
            # Diagnostic targets may be unavailable outside their frozen mask.
            # They are retained, not refitted or used to choose fresh support.
            widths[name] = WidthFit(
                name,
                array(f"{name}_base_scale", (4, 8), positive=True),
                array(f"{name}_log_multiplier", multiplier_shape),
                np.frombuffer(target.tobytes(), dtype=np.float64).reshape(target.shape),
                np.frombuffer(eligible.tobytes(), dtype=np.bool_),
                np.frombuffer(counts.tobytes(), dtype=np.int64).reshape(counts.shape),
            )
    return points, widths


def thread_settings() -> dict[str, Any]:
    from threadpoolctl import threadpool_info

    settings = {
        key: os.environ.get(key)
        for key in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        )
    }
    backends = threadpool_info()
    if any((value != "1" for value in settings.values())) or any(
        (backend["num_threads"] != 1 for backend in backends)
    ):
        raise ValueError(
            "Finite response-law qualification requires one BLAS/OpenMP thread per worker, set before Python starts"
        )
    return {"environment": settings, "loaded_backends": backends, "maximum_workers": 8}


def verify_frozen_dependencies(
    *,
    result: Mapping[str, Any],
    independent: Mapping[str, Any],
    result_sha256: str,
    manifest: Mapping[str, Any],
    lock_bytes: bytes,
    source_files: Mapping[str, bytes],
    required_paths: tuple[str, ...],
    scientific_prefixes: tuple[str, ...],
    interval_path: str,
    original_interval_source: bytes,
) -> dict[str, Any]:
    """Verify the original frozen numerical and complete scientific dependency cut.

    Authenticate result, independent readout, manifest, lock and source bytes
    before entry. Supply the six exact paths and four source-family prefixes
    from the already frozen scope: feature extraction, development, fitting,
    science, retained panel and instrument; prepared response source, prepared source/method
    and response formalization. The caller must authenticate that scope rather
    than choosing it from later outcomes. Previous interval source bytes must
    come from the manifest's exact source commit. This helper rechecks every
    declared selected hash and the original score/scale function bodies.
    It opens no file, executes no old source, refits nothing and grants no new
    qualification. Namespace migration alone does not restore a held freeze.
    """
    if (
        len(required_paths) != 6
        or len(set(required_paths)) != 6
        or len(scientific_prefixes) != 4
        or len(set(scientific_prefixes)) != 4
        or any(
            not value
            for value in (*required_paths, *scientific_prefixes, interval_path)
        )
    ):
        raise ValueError(
            "Frozen dependency scope changes its required scientific census"
        )
    prefixes = scientific_prefixes
    exact = set(required_paths)
    if (
        result["evaluability"] != "EVALUABLE"
        or not result["fresh_and_preparation_policy_condition"]
        or (not independent["verified"])
        or (not independent["fresh_and_preparation_policy_condition"])
        or (independent["result_sha256"] != result_sha256)
    ):
        raise ValueError("Finite response-law qualification lacks its independently verified development nomination")
    if (
        np.__version__ != manifest["numpy"]
        or platform.python_version() != manifest["python"].split()[0]
        or sha256(lock_bytes).hexdigest() != manifest["uv_lock_sha256"]
    ):
        raise ValueError("Finite response-law qualification changes the frozen numerical environment")
    unchanged = {}
    for row in manifest["loaded_source_files"]:
        path = row["path"]
        if path in exact or path.startswith(prefixes):
            if sha256(source_files[path]).hexdigest() != row["sha256"]:
                raise ValueError(f"Frozen scientific operator changed: {path}")
            unchanged[path] = row["sha256"]
    if not exact.issubset(unchanged):
        raise ValueError("development manifest lacks a required frozen scientific operator")
    if sha256(original_interval_source).hexdigest() != next(
        (
            r["sha256"]
            for r in manifest["loaded_source_files"]
            if r["path"] == interval_path
        )
    ):
        raise ValueError("Retained scale source differs from the frozen manifest")

    def bodies(raw: bytes) -> dict[str, str]:
        return {
            n.name: ast.dump(
                ast.Module(body=n.body, type_ignores=[]), include_attributes=False
            )
            for n in ast.parse(raw).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        }

    before = bodies(original_interval_source)
    after = bodies(source_files[interval_path])
    if any((after.get(k) != v for k, v in before.items())):
        raise ValueError("Frozen scale/score function behavior changed")
    return {
        "source_commit": manifest["source_commit"],
        "unchanged_scientific_files": unchanged,
        "interval_function_bodies_unchanged": True,
        "scope": 'FROZEN_POINT_SCALE_SUPPORT_AND_INSTRUMENT;FIT_NOMINATION_ONLY;NO_REFIT_OR_NEW_Q',
    }

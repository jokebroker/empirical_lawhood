# SPDX-License-Identifier: MPL-2.0
"""Shared precontact numerical enforcement and complete loaded-source observation."""

from hashlib import sha256
from pathlib import Path
import sys

from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.infrastructure.bounded_process import run_bounded_command
from empirical_lawhood.kernel.numerical_provenance import NumericalProducingProvenance
from empirical_lawhood.kernel.serialization import canonical_json_bytes


def numerical_package_sources(package_root=None):
    """Bound every installed package source/resource; omit only interpreter caches."""
    root = Path(__file__).parents[1] if package_root is None else Path(package_root)
    entries, total = [], 0
    for path in sorted(root.rglob("*")):
        if "__pycache__" in path.relative_to(root).parts or path.suffix in (".pyc", ".pyo"):
            continue
        if path.is_symlink():
            raise ValueError("Numerical package source inventory refuses symbolic links")
        if not path.is_file():
            continue
        raw = read_bounded_bytes(path, maximum_bytes=16 * 1024**2)
        total += len(raw)
        if len(entries) >= 8192 or total > 256 * 1024**2:
            raise ValueError("Numerical producing package exceeds its bounded source inventory")
        entries.append((path.relative_to(root).as_posix(), sha256(raw).hexdigest()))
    if not entries:
        raise ValueError("Numerical producing package is empty")
    return tuple(entries)


def numerical_package_code_sha256():
    return sha256(canonical_json_bytes(numerical_package_sources())).hexdigest()


def matrix_native_runtime_observation():
    """Observe and enforce the existing locked Linux numerical profile."""
    import platform
    import numpy as np
    import scipy
    from threadpoolctl import threadpool_info
    from empirical_lawhood.api.matrix_native_custody import MatrixNativeRuntimeObservation, _immutable_json
    observed = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "system": sys.platform, "machine": platform.machine(), "threadpools": threadpool_info()}
    MatrixNativeRuntimeObservation(canonical_json_bytes(_immutable_json(observed)).decode())
    return _immutable_json(observed)


def selected_dependency_lock_sha256(project_root):
    return sha256(read_bounded_bytes(Path(project_root) / "uv.lock", maximum_bytes=16 * 1024**2)).hexdigest()


def preflight_matrix_output_directory(*, directory, artifact_writer, project_root,
                                      allow_existing_empty=False):
    """Reuse the existing guarded output preflight before numerical effects."""
    from empirical_lawhood.api.authoring_handoff import preflight_authoring_output_directory
    return preflight_authoring_output_directory(directory=Path(directory), repo_root=Path(project_root),
        artifact_writer=artifact_writer, allow_existing_empty=allow_existing_empty)


def capture_matrix_provenance(*, project_root, expected_code_sha256=None, expected_lock_sha256=None):
    """Authenticate selected current bytes; HEAD is observation, not a clean-source claim."""
    root = Path(project_root).resolve(strict=True)
    package_root = root / "src" / "empirical_lawhood"
    import empirical_lawhood
    if Path(empirical_lawhood.__file__).resolve() != package_root / "__init__.py":
        raise ValueError("Numerical provenance executing package differs from selected checkout")
    for name, module in tuple(sys.modules.items()):
        if name == "empirical_lawhood" or name.startswith("empirical_lawhood."):
            origin = getattr(module, "__file__", None)
            if origin and origin.endswith(".py") and not Path(origin).resolve().is_relative_to(package_root):
                raise ValueError("Numerical provenance loaded module differs from selected checkout")
    def git(*args):
        result = run_bounded_command(("git", *args), cwd=root, timeout_seconds=5,
                                     maximum_stdout_bytes=1024**2, maximum_stderr_bytes=4096)
        if result.returncode:
            raise ValueError("Numerical producing checkout Git observation failed")
        return result.stdout.decode().strip()
    if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("Numerical provenance requires the selected checkout root")
    rows = numerical_package_sources(package_root)
    code = sha256(canonical_json_bytes(rows)).hexdigest()
    lock = selected_dependency_lock_sha256(root)
    if expected_code_sha256 is not None and code != expected_code_sha256:
        raise ValueError("Numerical producing code/source bytes changed after input preparation")
    if expected_lock_sha256 is not None and lock != expected_lock_sha256:
        raise ValueError("Numerical selected dependency lock changed after input preparation")
    observed = matrix_native_runtime_observation()
    return NumericalProducingProvenance(canonical_json_bytes(observed).decode(), lock, rows, code,
        git("rev-parse", "HEAD"), not bool(git("status", "--porcelain")))


def authenticate_recorded_matrix_provenance(provenance):
    """Validate retained producing evidence without observing the reader environment."""
    from empirical_lawhood.api.matrix_native_custody import MatrixNativeRuntimeObservation
    if not isinstance(provenance, NumericalProducingProvenance):
        raise ValueError("Exact numerical producing provenance is required")
    MatrixNativeRuntimeObservation(provenance.runtime_observation_json)
    return provenance

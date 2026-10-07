"""Pre-import numerical-runtime bootstrap for the project CLI.

Native BLAS/OpenMP runtimes commonly read their thread limits when NumPy or a
dependent extension is first imported.  This module therefore has no project,
numerical, or Typer imports: the console-script loader reaches :func:`main`,
sets the frozen single-thread environment, and only then imports the ordinary
CLI composition root.
"""

from __future__ import annotations

from collections.abc import MutableMapping
import os


NUMERICAL_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
NUMERICAL_BOOTSTRAP_ENV_VAR = "EMPIRICAL_LAWHOOD_NUMERICAL_BOOTSTRAP"
NUMERICAL_BOOTSTRAP_POLICY = "single-thread-before-numerical-import"


def enforce_single_thread_environment(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Overwrite every frozen native-runtime thread limit with ``1``.

    Existing values are deliberately not respected: accepting a caller's
    wider thread pool would make numerical behavior depend on the host shell.
    The bootstrap marker lets battery commands reject an invocation path that
    set the variables only after numerical modules had already been imported.
    """

    target = os.environ if environ is None else environ
    for name in NUMERICAL_THREAD_ENV_VARS:
        target[name] = "1"
    target[NUMERICAL_BOOTSTRAP_ENV_VAR] = NUMERICAL_BOOTSTRAP_POLICY
    return numerical_environment_payload(target)


def numerical_environment_payload(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Return the observed, serializable numerical bootstrap contract."""

    target = os.environ if environ is None else environ
    return {
        **{name: target.get(name, "") for name in NUMERICAL_THREAD_ENV_VARS},
        NUMERICAL_BOOTSTRAP_ENV_VAR: target.get(NUMERICAL_BOOTSTRAP_ENV_VAR, ""),
    }


def require_single_thread_bootstrap(
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Fail closed unless the pre-import single-thread contract is present."""

    observed = numerical_environment_payload(environ)
    violations = [
        f"{name}={observed[name]!r}" for name in NUMERICAL_THREAD_ENV_VARS if observed[name] != "1"
    ]
    if observed[NUMERICAL_BOOTSTRAP_ENV_VAR] != NUMERICAL_BOOTSTRAP_POLICY:
        violations.append(
            f"{NUMERICAL_BOOTSTRAP_ENV_VAR}={observed[NUMERICAL_BOOTSTRAP_ENV_VAR]!r}"
        )
    if violations:
        details = ", ".join(violations)
        raise ValueError(
            "battery execution requires the empirical-lawhood pre-import "
            f"single-thread bootstrap ({details})"
        )
    return observed


def main() -> None:
    """Set numerical limits before importing the full CLI and dispatch it."""

    enforce_single_thread_environment()
    from empirical_lawhood.cli.app import main as cli_main

    cli_main()


__all__ = [
    "NUMERICAL_BOOTSTRAP_ENV_VAR",
    "NUMERICAL_BOOTSTRAP_POLICY",
    "NUMERICAL_THREAD_ENV_VARS",
    "enforce_single_thread_environment",
    "main",
    "numerical_environment_payload",
    "require_single_thread_bootstrap",
]

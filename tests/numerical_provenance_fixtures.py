# SPDX-License-Identifier: MPL-2.0
"""Disposable guarded development storage, without scientific authority."""

from pathlib import Path

from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane, GuardedExternalRoot
from empirical_lawhood.runtime.artifacts import ExternalRootContract

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def numerical_plane(directory):
    directory.mkdir(parents=True, exist_ok=True)
    contract = ExternalRootContract("test.numerical-provenance.root", "synthetic software storage",
        str(directory.absolute()), "/", "empirical-lawhood/test/local-root", 1)
    return ExternalArtifactPlane(GuardedExternalRoot(contract))

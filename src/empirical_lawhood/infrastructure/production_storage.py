"""Exact production external-storage identity used by every follow-up composition."""

from __future__ import annotations

from typing import Final

from empirical_lawhood.runtime.artifacts import ArtifactProfile

from .artifacts import ArtifactProfileValidatorRegistry


# Dataset-shaped binary profiles are deliberately absent until a dataset-class
# adapter supplies an exact structural contract for its payload schema.  This
# static empty structural registry prevents the production composition from
# advertising generic Arrow, Parquet, HDF5, or NumPy admission.
# Historical validation identities belong to their frozen source closure. The
# current composition never silently promotes an old implementation identity.
PRODUCTION_ARTIFACT_PROFILE_VALIDATORS: Final = ArtifactProfileValidatorRegistry()
PRODUCTION_DISABLED_DATASET_STRUCTURAL_PROFILES: Final = (
    ArtifactProfile.ARROW_IPC,
    ArtifactProfile.AUDITED_HDF5,
    ArtifactProfile.NUMPY_NO_PICKLE,
    ArtifactProfile.PARQUET,
)


__all__ = [
    "PRODUCTION_ARTIFACT_PROFILE_VALIDATORS",
    "PRODUCTION_DISABLED_DATASET_STRUCTURAL_PROFILES",
]

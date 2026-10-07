# SPDX-License-Identifier: MPL-2.0
"""Closed artifact profiles shared by current public integration composition."""

from empirical_lawhood.adapters.simulator_morphism_challenges.authoring import ARRAY_PAYLOAD_SCHEMA
from empirical_lawhood.infrastructure.artifact_validation import ArtifactProfileValidatorRegistry, NumpyArrayContract


# Preserve every default production profile and the one exact RC array shape.
# No generic NumPy, Arrow, Parquet or HDF5 admission is introduced.
INTEGRATION_ARTIFACT_PROFILE_VALIDATORS = ArtifactProfileValidatorRegistry(
    numpy_contracts=(NumpyArrayContract(ARRAY_PAYLOAD_SCHEMA, "float64", (None,)),),
)


def integration_artifact_profile_validators() -> ArtifactProfileValidatorRegistry:
    """Return the code-owned registry; custody and current authority stay separate."""
    return INTEGRATION_ARTIFACT_PROFILE_VALIDATORS

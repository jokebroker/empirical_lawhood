'One-CLI ambient pressure superconductor gauge covariant response method packet and pure preissue plan projection.'

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure

from .synthetic_material_method_authoring import SyntheticMaterialResponseMethodAuthoringBundle, build_synthetic_material_response_method_authoring
from .synthetic_material_method_contracts import SyntheticMaterialResponseMethodConfig

_MAX_CONFIG_BYTES = 128 * 1024
_MAX_PACKET_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class SyntheticMaterialMethodProfile(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/ambient-pressure-superconductor/synthetic-material-method-profile'
    )

    profile_id: str
    experiment_id: str
    config_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_stable_id(self.experiment_id, field_name="experiment_id")
        validate_sha256(self.config_sha256, field_name="config_sha256")


def load_synthetic_material_response_config(path: Path) -> SyntheticMaterialResponseMethodConfig:
    return decode_canonical_bytes(
        read_bounded_bytes(path, maximum_bytes=_MAX_CONFIG_BYTES),
        SyntheticMaterialResponseMethodConfig,
        maximum_bytes=_MAX_CONFIG_BYTES,
    )


def load_synthetic_material_response_bundle(directory: Path) -> SyntheticMaterialResponseMethodAuthoringBundle:
    'Recompute the exact installed ambient pressure superconductor graph from bounded packet records.'

    profile = decode_canonical_bytes(
        read_bounded_bytes(directory / "profile.json", maximum_bytes=_MAX_CONFIG_BYTES),
        SyntheticMaterialMethodProfile,
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    config = load_synthetic_material_response_config(directory / "config.json")
    if profile.config_sha256 != config.fingerprint():
        raise ValueError('ambient pressure superconductor packet config differs from frozen profile')
    closure = decode_canonical_bytes(
        read_bounded_bytes(
            directory / "source-closure.json", maximum_bytes=_MAX_CONFIG_BYTES
        ),
        ImplementationSourceClosure,
        maximum_bytes=_MAX_CONFIG_BYTES,
    )
    expected = decode_canonical_bytes(
        read_bounded_bytes(
            directory / "authoring.json", maximum_bytes=_MAX_PACKET_BYTES
        ),
        ExecutableStudyDefinition,
        maximum_bytes=_MAX_PACKET_BYTES,
    )
    bundle = build_synthetic_material_response_method_authoring(
        config=config,
        experiment_id=profile.experiment_id,
        implementation_sha256=closure.implementation_sha256,
    )
    if bundle.authoring != expected:
        raise ValueError('ambient pressure superconductor packet authoring differs from installed science')
    return bundle


__all__ = [
    'SyntheticMaterialMethodProfile',
    'load_synthetic_material_response_config',
    'load_synthetic_material_response_bundle',
]

"""One-CLI RC candidate packet; authoring and projection have no native effects."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters.simulators.rc_ladder_response.contracts import ResistorCapacitorLadderStudyConfig
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure

from .authoring import RCLadderResponseAuthoringBundle, build_rc_authoring

_MAX_STUDY_BYTES = 128 * 1024
_MAX_PACKET_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class FreshRCProfile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/rc-ladder-response/fresh-rc-profile'

    profile_id: str
    experiment_id: str
    study_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id, field_name="profile_id")
        validate_stable_id(self.experiment_id, field_name="experiment_id")
        validate_sha256(self.study_sha256, field_name="study_sha256")


def load_rc_study(path: Path) -> ResistorCapacitorLadderStudyConfig:
    return decode_canonical_bytes(
        read_bounded_bytes(path, maximum_bytes=_MAX_STUDY_BYTES),
        ResistorCapacitorLadderStudyConfig,
        maximum_bytes=_MAX_STUDY_BYTES,
    )


def load_fresh_rc_bundle(directory: Path) -> RCLadderResponseAuthoringBundle:
    """Recompose the exact installed owner graph from bounded packet records."""

    profile = decode_canonical_bytes(
        read_bounded_bytes(directory / "profile.json", maximum_bytes=_MAX_STUDY_BYTES),
        FreshRCProfile,
        maximum_bytes=_MAX_STUDY_BYTES,
    )
    study = load_rc_study(directory / "study.json")
    if profile.study_sha256 != study.fingerprint():
        raise ValueError("RC packet study differs from its frozen profile")
    closure = decode_canonical_bytes(
        read_bounded_bytes(
            directory / "source-closure.json", maximum_bytes=_MAX_STUDY_BYTES
        ),
        ImplementationSourceClosure,
        maximum_bytes=_MAX_STUDY_BYTES,
    )
    expected = decode_canonical_bytes(
        read_bounded_bytes(
            directory / "authoring.json", maximum_bytes=_MAX_PACKET_BYTES
        ),
        ExecutableStudyDefinition,
        maximum_bytes=_MAX_PACKET_BYTES,
    )
    bundle = build_rc_authoring(
        study=study,
        experiment_id=profile.experiment_id,
        implementation_sha256=closure.implementation_sha256,
    )
    if bundle.authoring != expected:
        raise ValueError("RC packet authoring differs from its installed science")
    return bundle


__all__ = [
    'FreshRCProfile',
    "load_fresh_rc_bundle",
    "load_rc_study",
]

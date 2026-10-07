"""Strict new-identity reactor authoring through the installed candidate compiler.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

from empirical_lawhood.adapters.composition.reactor_prefix_response.authoring import ReactorAuthoringBundle, build_reactor_authoring
from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned import ReactorPrefixAssignment, ReactorPrefixPriorCensus
from empirical_lawhood.adapters.simulators.reactor_prefix_response.packaged_source import load_packaged_reactor_source
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.experiment_entry import ExecutableStudyDefinition
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure

_MAX_PROFILE_BYTES = 64 * 1024**2
_MAX_AUTHORING_BYTES = 128 * 1024**2
_MAX_ARCHIVE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class FreshReactorAuthoringProfile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/reactor-prefix-response/fresh-reactor-authoring-profile'

    profile_id: str
    experiment_id: str
    public_source_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.profile_id)
        validate_stable_id(self.experiment_id)
        validate_sha256(self.public_source_sha256)
        if self.experiment_id in ("reactor-prefix-native", "tbs-reactor-prefix-v1-r4"):
            raise ValueError(
                "fresh profile cannot issue the historical reactor identity"
            )


@dataclass(frozen=True, slots=True)
class AssignedReactorAuthoringProfile(FreshReactorAuthoringProfile):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/reactor-prefix-response/assigned-reactor-authoring-profile'
    VERSION: ClassVar[str] = '1.0.0'
    assignment: ReactorPrefixAssignment
    prior_census: ReactorPrefixPriorCensus

    def __post_init__(self) -> None:
        FreshReactorAuthoringProfile.__post_init__(self)
        self.assignment.check_prior_census(self.prior_census)
        if self.assignment.assignment_id != f"{self.experiment_id}.cohort":
            raise ValueError("REACTOR_PROFILE_ASSIGNMENT_ID_MISMATCH")


def load_fresh_reactor_profile(path: Path) -> FreshReactorAuthoringProfile:
    raw = read_bounded_bytes(path, maximum_bytes=_MAX_PROFILE_BYTES)
    document = json.loads(raw)
    if not isinstance(document, dict):
        raise ValueError("REACTOR_AUTHORING_PROFILE_DOCUMENT_INVALID")
    schema = document.get("schema")
    if not isinstance(schema, str):
        raise ValueError("REACTOR_AUTHORING_PROFILE_SCHEMA_UNSUPPORTED")
    record_type = {
        FreshReactorAuthoringProfile.SCHEMA: FreshReactorAuthoringProfile,
        AssignedReactorAuthoringProfile.SCHEMA: AssignedReactorAuthoringProfile,
    }.get(schema)
    if record_type is None:
        raise ValueError("REACTOR_AUTHORING_PROFILE_SCHEMA_UNSUPPORTED")
    return decode_canonical_bytes(raw, record_type, maximum_bytes=_MAX_PROFILE_BYTES)


def fresh_assignment_status(
    profile: FreshReactorAuthoringProfile,
) -> dict[str, object]:
    assignment = getattr(profile, "assignment", None)
    eligible = (
        assignment is not None
        and assignment.evidence_role == "PROSPECTIVE_RELEASE_QUALIFICATION"
    )
    return {
        "prospective_issue_eligible": eligible,
        "issue_stop_codes": ()
        if eligible
        else ("PUBLIC_EVALUATION_UNITS_AND_SEEDS_EXPOSED",),
        "assignment_sha256": None if assignment is None else assignment.fingerprint(),
        "independent_units": 5,
        "native_seed_assignment": "PUBLIC_EXPOSED"
        if assignment is None
        else assignment.evidence_role,
    }


def load_fresh_reactor_bundle(directory: Path) -> ReactorAuthoringBundle:
    """Recompose only the exact installed authoring, never a config-selected callable."""

    profile = load_fresh_reactor_profile(directory / "profile.json")
    closure = decode_canonical_bytes(
        read_bounded_bytes(
            directory / "source-closure.json", maximum_bytes=_MAX_PROFILE_BYTES
        ),
        ImplementationSourceClosure,
        maximum_bytes=_MAX_PROFILE_BYTES,
    )
    expected = decode_canonical_bytes(
        read_bounded_bytes(
            directory / "authoring.json", maximum_bytes=_MAX_AUTHORING_BYTES
        ),
        ExecutableStudyDefinition,
        maximum_bytes=_MAX_AUTHORING_BYTES,
    )
    source = load_packaged_reactor_source()
    if source.fingerprint() != profile.public_source_sha256:
        raise ValueError("fresh reactor profile source drifted")
    bundle = build_reactor_authoring(
        source=source,
        implementation_sha256=closure.implementation_sha256,
        fresh_experiment_id=profile.experiment_id,
        assignment=getattr(profile, "assignment", None),
        prior_census=getattr(profile, "prior_census", None),
    )
    if bundle.authoring != expected:
        raise ValueError(
            "fresh reactor authoring differs from its exact installed composition"
        )
    return bundle

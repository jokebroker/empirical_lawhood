# SPDX-License-Identifier: MPL-2.0
"""Immutable observed producing bytes and numerical environment, without authority."""

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from .provenance import ObjectIdentity
from .serialization import CanonicalRecord, canonical_json_bytes, validate_relative_locator, validate_sha256


@dataclass(frozen=True, slots=True)
class NumericalProducingProvenance(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/kernel/numerical-producing-provenance"
    runtime_observation_json: str
    dependency_lock_sha256: str
    package_sources: tuple[tuple[str, str], ...]
    code_sources_sha256: str
    observed_git_head: str
    observed_checkout_clean: bool

    def __post_init__(self):
        validate_sha256(self.dependency_lock_sha256)
        validate_sha256(self.code_sources_sha256)
        if (not 0 < len(self.package_sources) <= 8192
            or self.package_sources != tuple(sorted(set(self.package_sources)))
            or len({name for name, _ in self.package_sources}) != len(self.package_sources)
            or sha256(canonical_json_bytes(self.package_sources)).hexdigest() != self.code_sources_sha256
            or len(self.observed_git_head) != 40
            or any(c not in "0123456789abcdef" for c in self.observed_git_head)
            or type(self.observed_checkout_clean) is not bool):
            raise ValueError("Numerical provenance omits exact producing bytes or observed Git state")
        for name, digest in self.package_sources:
            validate_relative_locator(name)
            validate_sha256(digest)
        # The API's shared native contract validates the exact observed profile.
        # This pure record retains its bytes, without importing a native runtime.
        import json
        observation = json.loads(self.runtime_observation_json)
        if not isinstance(observation, dict) or not all(key in observation for key in (
            "python", "numpy", "scipy", "system", "machine", "threadpools")):
            raise ValueError("Numerical provenance omits its actual runtime observation")

    @property
    def identity(self):
        return ObjectIdentity.from_record("numerical.producing-provenance", self)

# SPDX-License-Identifier: MPL-2.0
"""Exact member custody for an exposed, completed analysis directory."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_relative_locator, validate_stable_id

MAX_RETAINED_ANALYSIS_MEMBER_BYTES = 96 * 1024**2
# The generic current source can retain 32 roots with 344 native cells each.
# This purpose-specific census leaves unrelated compact controls unchanged.
MAX_RETAINED_ANALYSIS_COMPLETION_BYTES = 16 * 1024**2
COMPLETION_FILENAME = "analysis-completion.canonical.json"


@dataclass(frozen=True, slots=True)
class RetainedAnalysisMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/runtime/retained-analysis-member"
    relative_path: str
    artifact: ArtifactIdentity

    def __post_init__(self):
        validate_relative_locator(self.relative_path)
        if (self.relative_path == COMPLETION_FILENAME or self.relative_path.endswith(".manifest.json")
                or ".publication-batches" in self.relative_path.split("/")
                or self.artifact.size_bytes > MAX_RETAINED_ANALYSIS_MEMBER_BYTES):
            raise ValueError("Retained analysis member changes its bounded original-file contract")


@dataclass(frozen=True, slots=True)
class RetainedAnalysisCompletion(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/runtime/retained-analysis-completion"
    completion_id: str
    members: tuple[RetainedAnalysisMember, ...]
    evidence_role: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
    scientific_authority: str = "NONE"

    def __post_init__(self):
        validate_stable_id(self.completion_id, field_name="completion_id")
        names = tuple(member.relative_path for member in self.members)
        if (not 0 < len(names) <= 16384 or names != tuple(sorted(set(names)))
                or self.evidence_role != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
                or self.scientific_authority != "NONE"):
            raise ValueError("Retained analysis completion changes its complete exposed member census")

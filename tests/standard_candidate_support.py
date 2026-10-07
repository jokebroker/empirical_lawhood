# SPDX-License-Identifier: MPL-2.0
# Adapted from the source project; synthetic software conformance only.
"""Fixed standard-candidate composition used by public API/CLI contract tests."""

from __future__ import annotations

from dataclasses import dataclass

from empirical_lawhood.planning.experiment_entry import StudyDefinition
from empirical_lawhood.planning.study_authoring import StudyDraft
from empirical_lawhood.runtime.candidate_compiler import StandardCandidateCompilationContext
from empirical_lawhood.runtime.candidate_composition import (
    CandidateCapabilityCatalog,
    CandidateCapabilityRegistration,
    CandidateContextResolution,
    StandardCandidateContextResolution,
)
from empirical_lawhood.runtime.source_resolution import CandidateSourceResolution


@dataclass(frozen=True, slots=True)
class FixedStandardCandidateContextProvider:
    """Return one injected truth-known context without touching source bytes."""

    context: StandardCandidateCompilationContext

    @property
    def catalog(self) -> CandidateCapabilityCatalog:
        return CandidateCapabilityCatalog(
            catalog_id="catalog.standard-candidate-test",
            registrations=tuple(
                CandidateCapabilityRegistration(
                    manifest=manifest,
                    provider_key=manifest.capability_key,
                    provider_version=manifest.capability_version,
                    config_media_type="application/json",
                    maximum_config_bytes=1024 * 1024,
                )
                for manifest in self.context.base.registry.capabilities
            ),
            templates=self.context.base.templates,
        )

    def _source_resolution(self) -> CandidateSourceResolution:
        return CandidateSourceResolution(
            qualifications=self.context.base.qualifications,
            receipts=(),
            diagnostics=(),
        )

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        if draft.draft_id == "":
            raise ValueError("draft identity is empty")
        return CandidateContextResolution(
            context=self.context.base,
            diagnostics=(),
            source_resolution=self._source_resolution(),
        )

    def resolve_standard(
        self,
        package: StudyDefinition,
    ) -> StandardCandidateContextResolution:
        if package.draft.draft_id == "":
            raise ValueError("authoring package draft identity is empty")
        return StandardCandidateContextResolution(
            context=self.context,
            diagnostics=(),
            source_resolution=self._source_resolution(),
        )


__all__ = ["FixedStandardCandidateContextProvider"]

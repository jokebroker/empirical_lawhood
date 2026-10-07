"""Per-capability composition for nonactuating programme candidates."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.study_authoring import CapabilitySelection, DesignInputRecord, StudyDraft
from empirical_lawhood.planning.experiment_entry import StudyDefinition
from empirical_lawhood.planning.formal_analysis import (
    FormalGapSourceCapabilityInventory,
    FormalMethodCatalog,
)

from .candidate_compiler import CandidateCompilationContext, CandidateDiagnostic, StudyTemplate, StandardCandidateCompilationContext
from .capabilities import CapabilityManifest, CapabilityRegistry
from .source_resolution import (
    CandidateSourceResolution,
    CandidateSourceResolutionService,
)


@dataclass(frozen=True, slots=True)
class CandidateCapabilityRegistration(CanonicalRecord):
    """One static manifest and its bounded configuration materialization contract."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-capability-registration'

    manifest: CapabilityManifest
    provider_key: str
    provider_version: str
    config_media_type: str
    maximum_config_bytes: int

    @property
    def registration_id(self) -> str:
        return self.manifest.registry_id

    def __post_init__(self) -> None:
        validate_stable_id(self.provider_key, field_name="provider_key")
        validate_semantic_version(self.provider_version)
        validate_nonempty(self.config_media_type, field_name="config_media_type")
        if (
            not isinstance(self.maximum_config_bytes, int)
            or isinstance(self.maximum_config_bytes, bool)
            or self.maximum_config_bytes <= 0
            or self.maximum_config_bytes > 16 * 1024**2
        ):
            raise ValueError("maximum_config_bytes must be in (0, 16 MiB]")


@dataclass(frozen=True, slots=True)
class CandidateCapabilityCatalog(CanonicalRecord):
    """Closed discovery/composition catalog; it contains no executable objects."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/candidate-capability-catalog'

    catalog_id: str
    registrations: tuple[CandidateCapabilityRegistration, ...]
    templates: tuple[StudyTemplate, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.catalog_id, field_name="catalog_id")
        if len(self.registrations) > 4096 or len(self.templates) > 1024:
            raise ValueError("candidate capability catalog exceeds its static bounds")
        require_sorted_unique_ids(
            self.registrations,
            attribute="registration_id",
            field_name="registrations",
        )
        require_sorted_unique_ids(
            self.templates,
            attribute="template_key",
            field_name="templates",
        )
        if not self.registrations:
            raise ValueError("candidate capability catalog cannot be empty")

    def registration(
        self,
        capability_key: str,
        capability_version: str,
    ) -> CandidateCapabilityRegistration | None:
        validate_stable_id(capability_key, field_name="capability_key")
        validate_semantic_version(capability_version)
        return next(
            (
                value
                for value in self.registrations
                if value.manifest.capability_key == capability_key
                and value.manifest.capability_version == capability_version
            ),
            None,
        )

    def resolve_exact(
        self,
        selection: CapabilitySelection,
    ) -> CandidateCapabilityRegistration | None:
        registration = self.registration(
            selection.capability_key,
            selection.capability_version,
        )
        if (
            registration is None
            or registration.manifest.implementation_sha256 != selection.implementation_sha256
        ):
            return None
        return registration

    def template(self, template_key: str) -> StudyTemplate | None:
        return next(
            (value for value in self.templates if value.template_key == template_key),
            None,
        )

    def compose_registry(
        self,
        selections: tuple[CapabilitySelection, ...],
    ) -> CapabilityRegistry:
        """Aggregate only selected key/version registrations into frozen run identity."""

        by_id = {value.registration_id: value.manifest for value in self.registrations}
        selected = tuple(
            sorted(
                (by_id[value.selection_id] for value in selections if value.selection_id in by_id),
                key=lambda value: value.registry_id,
            )
        )
        # A blocked all-missing draft still needs a valid context so the pure
        # compiler can return its exact CAPABILITY_REQUIRED diagnostics. Such a
        # context is never a candidate or frozen run identity.
        manifests = selected if selected else tuple(value.manifest for value in self.registrations)
        subject = canonical_json_bytes(manifests)
        digest = hashlib.sha256(subject).hexdigest()
        return CapabilityRegistry(
            registry_id=f"candidate-registry.{digest[:24]}",
            capabilities=manifests,
        )


@dataclass(frozen=True, slots=True)
class CandidateContextResolution:
    """Resolved pure compiler context plus preflight diagnostics and read receipts."""

    context: CandidateCompilationContext
    diagnostics: tuple[CandidateDiagnostic, ...]
    source_resolution: CandidateSourceResolution

    @property
    def registry_sha256(self) -> str:
        """Aggregate selected-registry fingerprint retained for issue/freeze."""

        return self.context.registry.fingerprint()


@dataclass(frozen=True, slots=True)
class StandardCandidateContextResolution:
    """Resolved context and source evidence, with separate immutable-issue stops.

    Pure development compilation may be useful even when a known exposure makes
    prospective issue invalid. The application replays these typed diagnostics
    at both base and extension issue; an attestation cannot clear them.
    """

    context: StandardCandidateCompilationContext
    diagnostics: tuple[CandidateDiagnostic, ...]
    source_resolution: CandidateSourceResolution
    issue_diagnostics: tuple[CandidateDiagnostic, ...] = ()

    @property
    def registry_sha256(self) -> str:
        return self.context.base.registry.fingerprint()


class CandidateContextProvider(Protocol):
    @property
    def catalog(self) -> CandidateCapabilityCatalog: ...

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution: ...

    def resolve_standard(
        self,
        package: StudyDefinition,
    ) -> StandardCandidateContextResolution: ...


@dataclass(frozen=True, slots=True)
class StaticCandidateContextProvider:
    """Compose selected manifests and externally verified source metadata."""

    catalog: CandidateCapabilityCatalog
    source_resolution_service: CandidateSourceResolutionService
    implementation_sha256: str
    known_design_inputs: tuple[DesignInputRecord, ...] = ()
    formal_methods: FormalMethodCatalog | None = None
    formal_source_inventories: tuple[FormalGapSourceCapabilityInventory, ...] = ()

    def __post_init__(self) -> None:
        validate_sha256(
            self.implementation_sha256,
            field_name="implementation_sha256",
        )
        require_sorted_unique_ids(
            self.known_design_inputs,
            attribute="input_id",
            field_name="known_design_inputs",
        )
        require_sorted_unique_ids(
            self.formal_source_inventories,
            attribute="denominator_id",
            field_name="formal_source_inventories",
        )
        if (self.formal_methods is None) != (not self.formal_source_inventories):
            raise ValueError(
                "standard formal methods and source inventories must be composed together"
            )

    def resolve(self, draft: StudyDraft) -> CandidateContextResolution:
        primary = self.catalog.template(draft.dag_template_key)
        templates = (
            ()
            if primary is None
            else (
                primary,
                *(
                    ()
                    if draft.conditional_successor is None
                    else tuple(
                        value
                        for value in (
                            self.catalog.template(draft.conditional_successor.template_key),
                        )
                        if value is not None and value.template_key != primary.template_key
                    )
                ),
            )
        )
        templates = tuple(sorted(templates, key=lambda value: value.template_key))
        source_resolution = self.source_resolution_service.resolve(
            draft=draft,
            templates=templates,
            catalog=self.catalog,
        )
        design_inputs = {value.input_id: value for value in self.known_design_inputs}
        for value in draft.design_inputs:
            design_inputs.setdefault(value.input_id, value)
        context = CandidateCompilationContext(
            context_id=f"context.{draft.draft_id}",
            registry=self.catalog.compose_registry(draft.capability_selections),
            templates=templates,
            qualifications=source_resolution.qualifications,
            known_design_inputs=tuple(
                sorted(design_inputs.values(), key=lambda value: value.input_id)
            ),
            implementation_sha256=self.implementation_sha256,
        )
        return CandidateContextResolution(
            context=context,
            diagnostics=source_resolution.diagnostics,
            source_resolution=source_resolution,
        )

    def resolve_standard(
        self,
        package: StudyDefinition,
    ) -> StandardCandidateContextResolution:
        if self.formal_methods is None or not self.formal_source_inventories:
            raise ValueError("standard formal candidate context is not composed")
        resolution = self.resolve(package.draft)
        denominator_id = "" if package.draft.system is None else package.draft.system.system_id
        inventories = tuple(
            value
            for value in self.formal_source_inventories
            if value.denominator_id == denominator_id
        )
        if not inventories:
            inventories = self.formal_source_inventories
        context = StandardCandidateCompilationContext(
            context_id=f"standard-context.{package.package_id}",
            base=resolution.context,
            formal_methods=self.formal_methods,
            source_inventories=inventories,
        )
        return StandardCandidateContextResolution(
            context=context,
            diagnostics=resolution.diagnostics,
            source_resolution=resolution.source_resolution,
        )


__all__ = [
    "CandidateCapabilityCatalog",
    "CandidateCapabilityRegistration",
    "CandidateContextProvider",
    "CandidateContextResolution",
    "StandardCandidateContextResolution",
    "StaticCandidateContextProvider",
]

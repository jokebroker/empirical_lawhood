"""Content-addressed, read-only source readiness contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar, Protocol

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_nonempty,
    validate_semantic_version,
    validate_schema,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.study_authoring import MaterializationQualificationReceipt, StudyDraft, SourceMaterializationRole

from .candidate_compiler import CandidateDiagnostic, CandidateDiagnosticClass, CandidateDiagnosticCode, StudyTemplate, make_candidate_diagnostic

if TYPE_CHECKING:
    from .candidate_composition import CandidateCapabilityCatalog


class ContentAddressedInputKind(StrEnum):
    CAPABILITY_CONFIG = "CAPABILITY_CONFIG"
    SOURCE_CONFIG = "SOURCE_CONFIG"
    SOURCE_MATERIALIZATION = "SOURCE_MATERIALIZATION"
    MATERIALIZATION_QUALIFICATION = "MATERIALIZATION_QUALIFICATION"


class SourceReadMode(StrEnum):
    ORDINARY_BOUNDED = "ORDINARY_BOUNDED"
    CHUNKED_HIGH_VOLUME = "CHUNKED_HIGH_VOLUME"


@dataclass(frozen=True, slots=True)
class ContentAddressedInputRequirement(CanonicalRecord):
    """One derived content address; callers supply no filesystem locator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/content-addressed-input-requirement'

    requirement_id: str
    kind: ContentAddressedInputKind
    content_sha256: str
    payload_schema: str
    media_type: str
    maximum_bytes: int
    expected_size_bytes: int | None
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.requirement_id, field_name="requirement_id")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")
        if (
            not isinstance(self.maximum_bytes, int)
            or isinstance(self.maximum_bytes, bool)
            or self.maximum_bytes <= 0
        ):
            raise ValueError("maximum_bytes must be a positive integer")
        if self.expected_size_bytes is not None and (
            not isinstance(self.expected_size_bytes, int)
            or isinstance(self.expected_size_bytes, bool)
            or self.expected_size_bytes < 0
            or self.expected_size_bytes > self.maximum_bytes
        ):
            raise ValueError("expected size must fit the input work envelope")

    @property
    def relative_locator(self) -> str:
        return f"candidate-inputs/sha256/{self.content_sha256[:2]}/{self.content_sha256}"


@dataclass(frozen=True, slots=True)
class ContentResolutionReceipt(CanonicalRecord):
    """Operational measurement for one bounded, read-only exact-byte verification."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/content-resolution-receipt'

    receipt_id: str
    requirement: ObjectIdentity
    observed_sha256: str
    observed_size_bytes: int
    chunk_count: int
    maximum_chunk_bytes: int
    verification_pass_count: int
    consumer_pass_count: int
    observed_total_read_bytes: int
    payload_materialized: bool
    materialized_bytes: int
    algorithmic_peak_buffer_bound_bytes: int
    observed_throughput_bytes_per_second: int
    elapsed_nanoseconds: int
    repository_bytes_written: int
    external_bytes_written: int
    network_bytes: int
    read_only: bool
    nofollow_enforced: bool
    file_identity_stable: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_sha256(self.observed_sha256, field_name="observed_sha256")
        for name, value in (
            ("observed_size_bytes", self.observed_size_bytes),
            ("chunk_count", self.chunk_count),
            ("maximum_chunk_bytes", self.maximum_chunk_bytes),
            ("verification_pass_count", self.verification_pass_count),
            ("consumer_pass_count", self.consumer_pass_count),
            ("observed_total_read_bytes", self.observed_total_read_bytes),
            ("materialized_bytes", self.materialized_bytes),
            (
                "algorithmic_peak_buffer_bound_bytes",
                self.algorithmic_peak_buffer_bound_bytes,
            ),
            (
                "observed_throughput_bytes_per_second",
                self.observed_throughput_bytes_per_second,
            ),
            ("elapsed_nanoseconds", self.elapsed_nanoseconds),
            ("repository_bytes_written", self.repository_bytes_written),
            ("external_bytes_written", self.external_bytes_written),
            ("network_bytes", self.network_bytes),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.elapsed_nanoseconds == 0:
            raise ValueError("content resolution requires a positive elapsed interval")
        if (self.payload_materialized and self.materialized_bytes != self.observed_size_bytes) or (
            not self.payload_materialized and self.materialized_bytes != 0
        ):
            raise ValueError("content materialization accounting is inconsistent")
        if self.materialized_bytes > self.observed_size_bytes:
            raise ValueError("materialized bytes exceed the resolved source")
        if self.algorithmic_peak_buffer_bound_bytes < self.maximum_chunk_bytes:
            raise ValueError("content buffer bound is smaller than one observed chunk")
        if self.repository_bytes_written or self.external_bytes_written or self.network_bytes:
            raise ValueError("content resolution must remain read-only and local")
        if not self.read_only or not self.nofollow_enforced or not self.file_identity_stable:
            raise ValueError("content resolution lacks its read-only custody guarantees")


@dataclass(frozen=True, slots=True)
class ResolvedContent:
    receipt: ContentResolutionReceipt
    payload: bytes | None


class ContentAddressedResolutionError(RuntimeError):
    pass


class ContentAddressedInputResolver(Protocol):
    def resolve(
        self,
        requirement: ContentAddressedInputRequirement,
        *,
        materialize: bool,
    ) -> ResolvedContent: ...


class CandidateCapabilityConfigDecoder(Protocol):
    """One statically composed adapter-local config decoder."""

    @property
    def provider_key(self) -> str: ...

    @property
    def provider_version(self) -> str: ...

    def validate_config(
        self,
        payload: bytes,
        *,
        expected_schema: str,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class SourceMaterializationConfig(CanonicalRecord):
    """Adapter-neutral exact-byte envelope selected by a source reference."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/source-materialization-config'

    config_id: str
    source_id: str
    role: SourceMaterializationRole
    content_sha256: str
    expected_size_bytes: int
    maximum_bytes: int
    payload_schema: str
    media_type: str
    read_mode: SourceReadMode
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.source_id, field_name="source_id")
        validate_sha256(self.content_sha256, field_name="content_sha256")
        if (
            not isinstance(self.expected_size_bytes, int)
            or isinstance(self.expected_size_bytes, bool)
            or self.expected_size_bytes < 0
        ):
            raise ValueError("expected_size_bytes must be nonnegative")
        if (
            not isinstance(self.maximum_bytes, int)
            or isinstance(self.maximum_bytes, bool)
            or self.maximum_bytes <= 0
            or self.expected_size_bytes > self.maximum_bytes
        ):
            raise ValueError("source size must fit its positive read envelope")
        validate_schema(self.payload_schema)
        validate_nonempty(self.media_type, field_name="media_type")


@dataclass(frozen=True, slots=True)
class CandidateSourceResolution:
    qualifications: tuple[MaterializationQualificationReceipt, ...]
    receipts: tuple[ContentResolutionReceipt, ...]
    diagnostics: tuple[CandidateDiagnostic, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_ids(
            self.qualifications,
            attribute="receipt_id",
            field_name="qualifications",
        )
        require_sorted_unique_ids(
            self.receipts,
            attribute="receipt_id",
            field_name="receipts",
        )


@dataclass(frozen=True, slots=True)
class CandidateSourceResolutionService:
    """Verify declared configs/source bytes and load qualification metadata only."""

    resolver: ContentAddressedInputResolver
    capability_config_decoders: tuple[CandidateCapabilityConfigDecoder, ...] = ()
    maximum_source_config_bytes: int = 1024 * 1024
    maximum_qualification_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("maximum_source_config_bytes", self.maximum_source_config_bytes),
            ("maximum_qualification_bytes", self.maximum_qualification_bytes),
        ):
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
                or value > 16 * 1024**2
            ):
                raise ValueError(f"{name} must be in (0, 16 MiB]")
        provider_ids: list[str] = []
        for decoder in self.capability_config_decoders:
            validate_stable_id(decoder.provider_key, field_name="provider_key")
            validate_semantic_version(decoder.provider_version)
            provider_ids.append(f"{decoder.provider_key}@{decoder.provider_version}")
        if provider_ids != sorted(set(provider_ids)):
            raise ValueError("capability config decoders must be sorted and unique")

    def resolve(
        self,
        *,
        draft: StudyDraft,
        templates: tuple[StudyTemplate, ...],
        catalog: CandidateCapabilityCatalog,
    ) -> CandidateSourceResolution:
        diagnostics: list[CandidateDiagnostic] = []
        receipts: list[ContentResolutionReceipt] = []
        qualifications: list[MaterializationQualificationReceipt] = []

        seen_configs: set[str] = set()
        for template in templates:
            for step in template.protocol.steps:
                registration = catalog.registration(
                    step.capability_key,
                    step.capability_version,
                )
                if registration is None or step.config.content_sha256 in seen_configs:
                    continue
                seen_configs.add(step.config.content_sha256)
                requirement = ContentAddressedInputRequirement(
                    requirement_id=f"capability-config.{step.config.config_id}",
                    kind=ContentAddressedInputKind.CAPABILITY_CONFIG,
                    content_sha256=step.config.content_sha256,
                    payload_schema=step.config.config_schema,
                    media_type=registration.config_media_type,
                    maximum_bytes=registration.maximum_config_bytes,
                    expected_size_bytes=None,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                )
                resolved_config = self._resolve_requirement(
                    requirement,
                    materialize=True,
                    diagnostics=diagnostics,
                    receipts=receipts,
                    diagnostic_class=CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    diagnostic_code=CandidateDiagnosticCode.CAPABILITY_REQUIRED,
                    field_path=f"protocol.{step.step_id}.config",
                    label="capability config",
                )
                if resolved_config is None or resolved_config.payload is None:
                    continue
                decoder = next(
                    (
                        value
                        for value in self.capability_config_decoders
                        if value.provider_key == registration.provider_key
                        and value.provider_version == registration.provider_version
                    ),
                    None,
                )
                if decoder is None:
                    diagnostics.append(
                        make_candidate_diagnostic(
                            CandidateDiagnosticClass.UNRESOLVED_READINESS,
                            CandidateDiagnosticCode.CAPABILITY_REQUIRED,
                            f"protocol.{step.step_id}.config",
                            (
                                "static config decoder is not composed for provider "
                                f"{registration.provider_key}@"
                                f"{registration.provider_version}"
                            ),
                        )
                    )
                    continue
                try:
                    decoder.validate_config(
                        resolved_config.payload,
                        expected_schema=step.config.config_schema,
                    )
                except (TypeError, ValueError) as error:
                    diagnostics.append(
                        make_candidate_diagnostic(
                            CandidateDiagnosticClass.FATAL,
                            CandidateDiagnosticCode.SCIENTIFIC_SPEC_INCOMPLETE,
                            f"protocol.{step.step_id}.config",
                            f"capability config decoding failed: {error}",
                        )
                    )

        for source in draft.source_materializations:
            source_field = f"source_materializations.{source.source_id}"
            config_requirement = ContentAddressedInputRequirement(
                requirement_id=f"source-config.{source.source_id}",
                kind=ContentAddressedInputKind.SOURCE_CONFIG,
                content_sha256=source.source_config_sha256,
                payload_schema=SourceMaterializationConfig.SCHEMA,
                media_type="application/json",
                maximum_bytes=self.maximum_source_config_bytes,
                expected_size_bytes=None,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            )
            resolved_config = self._resolve_requirement(
                config_requirement,
                materialize=True,
                diagnostics=diagnostics,
                receipts=receipts,
                diagnostic_class=CandidateDiagnosticClass.UNRESOLVED_READINESS,
                diagnostic_code=CandidateDiagnosticCode.SOURCE_UNRESOLVED,
                field_path=f"{source_field}.source_config_sha256",
                label="source config",
            )
            source_config: SourceMaterializationConfig | None = None
            if resolved_config is not None and resolved_config.payload is not None:
                try:
                    source_config = decode_canonical_bytes(
                        resolved_config.payload,
                        SourceMaterializationConfig,
                        maximum_bytes=self.maximum_source_config_bytes,
                    )
                    if (
                        source_config.source_id != source.source_id
                        or source_config.role is not source.role
                        or source_config.content_sha256 != source.content_sha256
                    ):
                        raise ValueError("source config bindings differ from the draft")
                except ValueError as error:
                    diagnostics.append(
                        make_candidate_diagnostic(
                            CandidateDiagnosticClass.FATAL,
                            CandidateDiagnosticCode.SCIENTIFIC_SPEC_INCOMPLETE,
                            f"{source_field}.source_config_sha256",
                            str(error),
                        )
                    )
                    source_config = None
            if source_config is not None:
                source_requirement = ContentAddressedInputRequirement(
                    requirement_id=f"source-materialization.{source.source_id}",
                    kind=ContentAddressedInputKind.SOURCE_MATERIALIZATION,
                    content_sha256=source.content_sha256,
                    payload_schema=source_config.payload_schema,
                    media_type=source_config.media_type,
                    maximum_bytes=source_config.maximum_bytes,
                    expected_size_bytes=source_config.expected_size_bytes,
                    outcome_access=source_config.outcome_access,
                    visibility_ceiling=source_config.visibility_ceiling,
                )
                self._resolve_requirement(
                    source_requirement,
                    materialize=(source_config.read_mode is SourceReadMode.ORDINARY_BOUNDED),
                    diagnostics=diagnostics,
                    receipts=receipts,
                    diagnostic_class=CandidateDiagnosticClass.UNRESOLVED_READINESS,
                    diagnostic_code=CandidateDiagnosticCode.SOURCE_UNRESOLVED,
                    field_path=f"{source_field}.content_sha256",
                    label="source materialization",
                )

            qualification_identity = source.qualification_receipt
            if qualification_identity is None:
                continue
            qualification_requirement = ContentAddressedInputRequirement(
                requirement_id=f"qualification.{source.source_id}",
                kind=ContentAddressedInputKind.MATERIALIZATION_QUALIFICATION,
                content_sha256=qualification_identity.object_fingerprint,
                payload_schema=MaterializationQualificationReceipt.SCHEMA,
                media_type="application/json",
                maximum_bytes=self.maximum_qualification_bytes,
                expected_size_bytes=None,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            )
            resolved_qualification = self._resolve_requirement(
                qualification_requirement,
                materialize=True,
                diagnostics=diagnostics,
                receipts=receipts,
                diagnostic_class=CandidateDiagnosticClass.UNRESOLVED_READINESS,
                diagnostic_code=CandidateDiagnosticCode.SOURCE_QUALIFICATION_REQUIRED,
                field_path=f"{source_field}.qualification_receipt",
                label="materialization qualification",
            )
            if resolved_qualification is None or resolved_qualification.payload is None:
                continue
            try:
                qualification = decode_canonical_bytes(
                    resolved_qualification.payload,
                    MaterializationQualificationReceipt,
                    maximum_bytes=self.maximum_qualification_bytes,
                )
                observed_identity = ObjectIdentity.from_record(
                    qualification.receipt_id,
                    qualification,
                )
                if observed_identity != qualification_identity:
                    raise ValueError("qualification identity differs from the draft")
            except ValueError as error:
                diagnostics.append(
                    make_candidate_diagnostic(
                        CandidateDiagnosticClass.UNRESOLVED_READINESS,
                        CandidateDiagnosticCode.SOURCE_QUALIFICATION_REQUIRED,
                        f"{source_field}.qualification_receipt",
                        str(error),
                    )
                )
            else:
                qualifications.append(qualification)

        return CandidateSourceResolution(
            qualifications=tuple(
                sorted(
                    {value.receipt_id: value for value in qualifications}.values(),
                    key=lambda value: value.receipt_id,
                )
            ),
            receipts=tuple(
                sorted(
                    {value.receipt_id: value for value in receipts}.values(),
                    key=lambda value: value.receipt_id,
                )
            ),
            diagnostics=tuple(
                sorted(
                    {value.diagnostic_id: value for value in diagnostics}.values(),
                    key=lambda value: value.diagnostic_id,
                )
            ),
        )

    def _resolve_requirement(
        self,
        requirement: ContentAddressedInputRequirement,
        *,
        materialize: bool,
        diagnostics: list[CandidateDiagnostic],
        receipts: list[ContentResolutionReceipt],
        diagnostic_class: CandidateDiagnosticClass,
        diagnostic_code: CandidateDiagnosticCode,
        field_path: str,
        label: str,
    ) -> ResolvedContent | None:
        try:
            resolved = self.resolver.resolve(
                requirement,
                materialize=materialize,
            )
        except (ContentAddressedResolutionError, OSError, ValueError) as error:
            diagnostics.append(
                make_candidate_diagnostic(
                    diagnostic_class,
                    diagnostic_code,
                    field_path,
                    (
                        f"{label} is unavailable or invalid; "
                        f"kind={requirement.kind.value}; "
                        f"content_sha256={requirement.content_sha256}; "
                        f"derived_locator={requirement.relative_locator}; "
                        f"schema={requirement.payload_schema}; "
                        f"media_type={requirement.media_type}; "
                        f"maximum_bytes={requirement.maximum_bytes}; "
                        f"outcome_access={requirement.outcome_access.value}; "
                        f"visibility={requirement.visibility_ceiling.value}; "
                        f"reason={error}"
                    ),
                )
            )
            return None
        receipts.append(resolved.receipt)
        return resolved


__all__ = [
    "CandidateSourceResolution",
    "CandidateSourceResolutionService",
    "CandidateCapabilityConfigDecoder",
    "ContentAddressedInputKind",
    "ContentAddressedInputRequirement",
    "ContentAddressedInputResolver",
    "ContentAddressedResolutionError",
    "ContentResolutionReceipt",
    "ResolvedContent",
    "SourceMaterializationConfig",
    "SourceReadMode",
]

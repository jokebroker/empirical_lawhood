"""Outcome-blind MAST-U archive query declarations and receipt-first custody joins."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar, Protocol

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)


MAX_MASTU_ARCHIVE_QUERIES = 4096


@dataclass(frozen=True, slots=True)
class MASTUArchiveQuery(CanonicalRecord):
    """One pre-contact UDA selector; it predicts no response property."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-query'

    query_id: str
    acquisition_group_id: str
    preparation_instance_id: str
    physical_independent_unit_id: str
    shot_id: str
    signal_ids: tuple[str, ...]
    time_window_id: str
    view_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "query_id",
            "acquisition_group_id",
            "preparation_instance_id",
            "physical_independent_unit_id",
            "shot_id",
            "time_window_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.signal_ids,
            field_name="signal_ids",
            allow_empty=False,
        )
        for signal_id in self.signal_ids:
            validate_stable_id(signal_id, field_name="signal_ids")
        require_sorted_unique_strings(self.view_ids, field_name="view_ids", allow_empty=False)
        for view_id in self.view_ids:
            validate_stable_id(view_id, field_name="view_ids")


@dataclass(frozen=True, slots=True)
class MASTUArchiveQueryManifest(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-query-manifest'

    manifest_id: str
    queries: tuple[MASTUArchiveQuery, ...]
    provider_key: str
    provider_version: str
    request_schema: str
    response_object_schema: str
    maximum_query_count: int
    maximum_object_count: int
    maximum_total_response_bytes: int
    frozen_before_source_contact: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        validate_stable_id(self.provider_key, field_name="provider_key")
        validate_semantic_version(self.provider_version)
        validate_schema(self.request_schema)
        validate_schema(self.response_object_schema)
        require_sorted_unique_ids(self.queries, attribute="query_id", field_name="queries")
        if not self.queries or len(self.queries) > MAX_MASTU_ARCHIVE_QUERIES:
            raise ValueError("MAST-U query-manifest query count is outside its bound")
        if self.maximum_query_count != len(self.queries):
            raise ValueError("MAST-U query-manifest count differs from its frozen roster")
        # This bounded port returns exactly one custodied response object per query.
        # A smaller object bound would admit a manifest that can never complete
        # without silently dropping a predeclared query.
        if self.maximum_object_count != self.maximum_query_count:
            raise ValueError("MAST-U query-manifest requires one object per query")
        if self.maximum_total_response_bytes < self.maximum_object_count:
            raise ValueError("MAST-U query-manifest response-byte bound is invalid")
        if (
            not self.frozen_before_source_contact
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("MAST-U query manifest must freeze outcome-blind before contact")


@dataclass(frozen=True, slots=True)
class MASTUArchiveQuerySourceConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-query-source-config'

    config_id: str
    provider_key: str
    provider_version: str
    provider_implementation_sha256: str
    query_manifest: ObjectIdentity
    query_manifest_relative_locator: str
    request_schema: str
    response_object_schema: str
    maximum_query_count: int
    maximum_object_count: int
    maximum_total_response_bytes: int
    grants_authority: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.provider_key, field_name="provider_key")
        validate_semantic_version(self.provider_version)
        validate_sha256(
            self.provider_implementation_sha256,
            field_name="provider_implementation_sha256",
        )
        if self.query_manifest.object_schema != MASTUArchiveQueryManifest.SCHEMA:
            raise ValueError("MAST-U query config binds another manifest schema")
        validate_relative_locator(self.query_manifest_relative_locator)
        validate_schema(self.request_schema)
        validate_schema(self.response_object_schema)
        if (
            self.maximum_query_count < 1
            or self.maximum_object_count < 1
            or self.maximum_object_count != self.maximum_query_count
            or self.maximum_total_response_bytes < self.maximum_object_count
        ):
            raise ValueError("MAST-U query config bounds are invalid")
        if self.grants_authority or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("MAST-U query config cannot contact or authorize the source")


@dataclass(frozen=True, slots=True)
class MASTUArchiveObjectReceipt(CanonicalRecord):
    """Post-response facts learned from one immediately custodied UDA object."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-object-receipt'

    receipt_id: str
    query: ObjectIdentity
    native_object: ArtifactIdentity
    observed_data_sha256: str
    observed_metadata_sha256: str
    observed_size_bytes: int
    custody_relative_locator: str
    custodied_before_next_retrieval: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.query.object_schema != MASTUArchiveQuery.SCHEMA:
            raise ValueError("MAST-U object receipt binds another query schema")
        validate_sha256(self.observed_data_sha256, field_name="observed_data_sha256")
        validate_sha256(
            self.observed_metadata_sha256,
            field_name="observed_metadata_sha256",
        )
        if self.observed_size_bytes < 1:
            raise ValueError("MAST-U object receipt requires observed response bytes")
        validate_relative_locator(self.custody_relative_locator)
        if not self.custodied_before_next_retrieval:
            raise ValueError("MAST-U response was not custodied before the next retrieval")


@dataclass(frozen=True, slots=True)
class MASTUArchiveCompactUnitManifest(CanonicalRecord):
    """Compact post-custody query/object/unit/view lineage; never response bytes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-compact-unit-manifest'

    manifest_id: str
    query_manifest: ObjectIdentity
    object_receipts: tuple[ObjectIdentity, ...]
    preparation_instance_ids: tuple[str, ...]
    acquisition_group_ids: tuple[str, ...]
    physical_independent_unit_ids: tuple[str, ...]
    view_ids: tuple[str, ...]
    complete_after_all_object_custody: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.manifest_id, field_name="manifest_id")
        if self.query_manifest.object_schema != MASTUArchiveQueryManifest.SCHEMA:
            raise ValueError("compact unit manifest binds another query manifest")
        require_sorted_unique_ids(
            self.object_receipts,
            attribute="object_id",
            field_name="object_receipts",
        )
        if not self.object_receipts or any(
            value.object_schema != MASTUArchiveObjectReceipt.SCHEMA
            for value in self.object_receipts
        ):
            raise ValueError("compact unit manifest lacks exact object receipts")
        for name in (
            "preparation_instance_ids",
            "acquisition_group_ids",
            "physical_independent_unit_ids",
            "view_ids",
        ):
            values = getattr(self, name)
            require_sorted_unique_strings(values, field_name=name, allow_empty=False)
            for value in values:
                validate_stable_id(value, field_name=name)
        if not self.complete_after_all_object_custody:
            raise ValueError("compact unit manifest cannot precede complete object custody")


@dataclass(frozen=True, slots=True)
class MASTUArchiveAcquisitionUnitManifest(CanonicalRecord):
    "One acquisition-group custody terminal used by fan-out."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-acquisition-unit-manifest'

    manifest_id: str
    query_manifest: ObjectIdentity
    query: ObjectIdentity
    object_receipt: ObjectIdentity
    acquisition_group_id: str
    preparation_instance_id: str
    physical_independent_unit_id: str
    view_ids: tuple[str, ...]
    observed_total_response_bytes: int
    complete_after_object_custody: bool

    def __post_init__(self) -> None:
        for name in (
            "manifest_id",
            "acquisition_group_id",
            "preparation_instance_id",
            "physical_independent_unit_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.query_manifest.object_schema != MASTUArchiveQueryManifest.SCHEMA:
            raise ValueError("acquisition unit manifest binds another query manifest")
        if self.query.object_schema != MASTUArchiveQuery.SCHEMA:
            raise ValueError("acquisition unit manifest binds another query")
        if self.object_receipt.object_schema != MASTUArchiveObjectReceipt.SCHEMA:
            raise ValueError("acquisition unit manifest binds another object receipt")
        require_sorted_unique_strings(self.view_ids, field_name="view_ids", allow_empty=False)
        if self.observed_total_response_bytes < 1:
            raise ValueError("acquisition unit manifest requires total observed response bytes")
        if not self.complete_after_object_custody:
            raise ValueError("acquisition unit manifest precedes its native object custody")


class MASTUArchiveAcquisitionDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    INVALID_RESPONSE = "INVALID_RESPONSE"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"


@dataclass(frozen=True, slots=True)
class MASTUArchiveAcquisitionResult(CanonicalRecord):
    "One acquisition terminal; incomplete objects never masquerade as custody."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-acquisition-result'

    result_id: str
    request: ObjectIdentity
    query: ObjectIdentity
    disposition: MASTUArchiveAcquisitionDisposition
    object_receipt: MASTUArchiveObjectReceipt | None
    unit_manifest: MASTUArchiveAcquisitionUnitManifest | None
    reason_codes: tuple[str, ...]
    source_contacted: bool
    object_custodied: bool
    scientific_verdict_constructed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.request.object_schema != 'empirical-lawhood/runtime/native-acquisition-request':
            raise ValueError("archive acquisition result binds another neutral request")
        if self.query.object_schema != MASTUArchiveQuery.SCHEMA:
            raise ValueError("archive acquisition result binds another query")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        complete = self.disposition is MASTUArchiveAcquisitionDisposition.COMPLETE
        if complete:
            if (
                self.object_receipt is None
                or self.unit_manifest is None
                or self.reason_codes
                or not self.source_contacted
                or not self.object_custodied
            ):
                raise ValueError("complete archive acquisition lacks exact custody")
            if self.unit_manifest.object_receipt != ObjectIdentity.from_record(
                self.object_receipt.receipt_id,
                self.object_receipt,
            ):
                raise ValueError("archive acquisition unit binds another receipt")
        elif (
            self.object_receipt is not None
            or self.unit_manifest is not None
            or not self.reason_codes
        ):
            raise ValueError("noncomplete archive acquisition cannot claim object custody")
        if self.object_custodied and not self.source_contacted:
            raise ValueError("archive object custody cannot precede source contact")
        if self.scientific_verdict_constructed:
            raise ValueError("archive source adapter cannot construct a scientific verdict")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("archive acquisition result must remain sealed")


@dataclass(frozen=True, slots=True)
class MASTUArchiveCompactManifestResult(CanonicalRecord):
    """All-group custody reduction or an exact upstream obstruction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/physical/mast-archive-response-qualification/mastu-archive-compact-manifest-result'

    result_id: str
    query_manifest: ObjectIdentity
    acquisition_results: tuple[ObjectIdentity, ...]
    compact_manifest: MASTUArchiveCompactUnitManifest | None
    disposition: MASTUArchiveAcquisitionDisposition
    reason_codes: tuple[str, ...]
    scientific_verdict_constructed: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.result_id, field_name="result_id")
        if self.query_manifest.object_schema != MASTUArchiveQueryManifest.SCHEMA:
            raise ValueError("compact result binds another query manifest")
        require_sorted_unique_ids(
            self.acquisition_results,
            attribute="object_id",
            field_name="acquisition_results",
        )
        if not self.acquisition_results or any(
            value.object_schema != MASTUArchiveAcquisitionResult.SCHEMA
            for value in self.acquisition_results
        ):
            raise ValueError("compact result lacks its exact acquisition terminals")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        complete = self.disposition is MASTUArchiveAcquisitionDisposition.COMPLETE
        if complete != (self.compact_manifest is not None) or complete == bool(self.reason_codes):
            raise ValueError("compact-result disposition/product/reasons disagree")
        if self.scientific_verdict_constructed:
            raise ValueError("compact custody reduction cannot construct a verdict")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("compact custody result must remain sealed")


@dataclass(frozen=True, slots=True)
class MASTUArchiveResponseObject:
    """Transient provider response; bytes must not escape the custody operation."""

    query_id: str
    data_bytes: bytes
    metadata_bytes: bytes
    media_type: str


class MASTUArchiveQueryClient(Protocol):
    """Authorized provider client; constructing a manifest never invokes this port."""

    def retrieve(self, query: MASTUArchiveQuery) -> MASTUArchiveResponseObject: ...


class MASTUArchiveCustodyPort(Protocol):
    """Durably custody one response before another source retrieval is permitted."""

    def custody(
        self,
        *,
        query: MASTUArchiveQuery,
        native_object: ArtifactIdentity,
        data_bytes: bytes,
        metadata_bytes: bytes,
    ) -> str: ...


def retrieve_mastu_archive_query(
    *,
    manifest: MASTUArchiveQueryManifest,
    config: MASTUArchiveQuerySourceConfig,
    query: MASTUArchiveQuery,
    client: MASTUArchiveQueryClient,
    custody: MASTUArchiveCustodyPort,
    unit_manifest_id: str,
    maximum_remaining_response_bytes: int | None = None,
) -> tuple[MASTUArchiveObjectReceipt, MASTUArchiveAcquisitionUnitManifest]:
    """Retrieve and custody exactly one predeclared acquisition-group object."""

    manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
    if config.query_manifest != manifest_identity or query not in manifest.queries:
        raise ValueError("MAST-U group retrieval substitutes its frozen query manifest")
    if (
        config.provider_key != manifest.provider_key
        or config.provider_version != manifest.provider_version
        or config.request_schema != manifest.request_schema
        or config.response_object_schema != manifest.response_object_schema
        or config.maximum_query_count != manifest.maximum_query_count
        or config.maximum_object_count != manifest.maximum_object_count
        or config.maximum_total_response_bytes != manifest.maximum_total_response_bytes
    ):
        raise ValueError("MAST-U group retrieval config differs from frozen manifest bounds")
    response = client.retrieve(query)
    if response.query_id != query.query_id or not response.data_bytes:
        raise ValueError("MAST-U provider returned another or empty group object")
    observed_bytes = len(response.data_bytes) + len(response.metadata_bytes)
    remaining_bound = (
        manifest.maximum_total_response_bytes
        if maximum_remaining_response_bytes is None
        else maximum_remaining_response_bytes
    )
    if remaining_bound < 1 or observed_bytes > min(
        manifest.maximum_total_response_bytes,
        remaining_bound,
    ):
        raise ValueError("MAST-U group response exceeds the frozen byte bound")
    data_sha256 = sha256(response.data_bytes).hexdigest()
    metadata_sha256 = sha256(response.metadata_bytes).hexdigest()
    native_object = ArtifactIdentity(
        artifact_id=f"artifact.{query.query_id}",
        role="mastu-native-response-object",
        payload_schema=manifest.response_object_schema,
        sha256=data_sha256,
        media_type=response.media_type,
        size_bytes=len(response.data_bytes),
    )
    locator = custody.custody(
        query=query,
        native_object=native_object,
        data_bytes=response.data_bytes,
        metadata_bytes=response.metadata_bytes,
    )
    receipt = MASTUArchiveObjectReceipt(
        receipt_id=f"receipt.{query.query_id}",
        query=ObjectIdentity.from_record(query.query_id, query),
        native_object=native_object,
        observed_data_sha256=data_sha256,
        observed_metadata_sha256=metadata_sha256,
        observed_size_bytes=len(response.data_bytes),
        custody_relative_locator=locator,
        custodied_before_next_retrieval=True,
    )
    unit = MASTUArchiveAcquisitionUnitManifest(
        manifest_id=unit_manifest_id,
        query_manifest=manifest_identity,
        query=ObjectIdentity.from_record(query.query_id, query),
        object_receipt=ObjectIdentity.from_record(receipt.receipt_id, receipt),
        acquisition_group_id=query.acquisition_group_id,
        preparation_instance_id=query.preparation_instance_id,
        physical_independent_unit_id=query.physical_independent_unit_id,
        view_ids=query.view_ids,
        observed_total_response_bytes=observed_bytes,
        complete_after_object_custody=True,
    )
    return receipt, unit


def retrieve_mastu_archive_query_manifest(
    *,
    manifest: MASTUArchiveQueryManifest,
    config: MASTUArchiveQuerySourceConfig,
    client: MASTUArchiveQueryClient,
    custody: MASTUArchiveCustodyPort,
    compact_manifest_id: str,
) -> tuple[tuple[MASTUArchiveObjectReceipt, ...], MASTUArchiveCompactUnitManifest]:
    """Retrieve sequentially, custody immediately, and emit only compact lineage.

    The function deliberately has no batching hook: a failed or delayed custody call
    raises before the next ``client.retrieve`` invocation. Response hashes and sizes
    are calculated from returned bytes and cannot be supplied by issued config.
    """

    manifest_identity = ObjectIdentity.from_record(manifest.manifest_id, manifest)
    if config.query_manifest != manifest_identity:
        raise ValueError("MAST-U query config substitutes the frozen query manifest")
    if (
        config.provider_key != manifest.provider_key
        or config.provider_version != manifest.provider_version
        or config.request_schema != manifest.request_schema
        or config.response_object_schema != manifest.response_object_schema
        or config.maximum_query_count != manifest.maximum_query_count
        or config.maximum_object_count != manifest.maximum_object_count
        or config.maximum_total_response_bytes != manifest.maximum_total_response_bytes
    ):
        raise ValueError("MAST-U query config differs from frozen manifest bounds")

    receipts: list[MASTUArchiveObjectReceipt] = []
    total_bytes = 0
    for query in manifest.queries:
        response = client.retrieve(query)
        if response.query_id != query.query_id:
            raise ValueError("MAST-U provider response is attributed to another query")
        if not response.data_bytes:
            raise ValueError("MAST-U provider returned an empty native object")
        total_bytes += len(response.data_bytes) + len(response.metadata_bytes)
        if len(receipts) + 1 > manifest.maximum_object_count:
            raise ValueError("MAST-U provider exceeded the frozen object count")
        if total_bytes > manifest.maximum_total_response_bytes:
            raise ValueError("MAST-U provider exceeded the frozen response-byte bound")

        data_sha256 = sha256(response.data_bytes).hexdigest()
        metadata_sha256 = sha256(response.metadata_bytes).hexdigest()
        native_object = ArtifactIdentity(
            artifact_id=f"artifact.{query.query_id}",
            role="mastu-native-response-object",
            payload_schema=manifest.response_object_schema,
            sha256=data_sha256,
            media_type=response.media_type,
            size_bytes=len(response.data_bytes),
        )
        custody_locator = custody.custody(
            query=query,
            native_object=native_object,
            data_bytes=response.data_bytes,
            metadata_bytes=response.metadata_bytes,
        )
        receipts.append(
            MASTUArchiveObjectReceipt(
                receipt_id=f"receipt.{query.query_id}",
                query=ObjectIdentity.from_record(query.query_id, query),
                native_object=native_object,
                observed_data_sha256=data_sha256,
                observed_metadata_sha256=metadata_sha256,
                observed_size_bytes=len(response.data_bytes),
                custody_relative_locator=custody_locator,
                custodied_before_next_retrieval=True,
            )
        )

    compact = MASTUArchiveCompactUnitManifest(
        manifest_id=compact_manifest_id,
        query_manifest=manifest_identity,
        object_receipts=tuple(
            sorted(
                (ObjectIdentity.from_record(value.receipt_id, value) for value in receipts),
                key=lambda value: value.object_id,
            )
        ),
        preparation_instance_ids=tuple(
            sorted({value.preparation_instance_id for value in manifest.queries})
        ),
        acquisition_group_ids=tuple(
            sorted({value.acquisition_group_id for value in manifest.queries})
        ),
        physical_independent_unit_ids=tuple(
            sorted({value.physical_independent_unit_id for value in manifest.queries})
        ),
        view_ids=tuple(sorted({view for value in manifest.queries for view in value.view_ids})),
        complete_after_all_object_custody=True,
    )
    return tuple(receipts), compact


def reduce_mastu_archive_acquisition_results(
    *,
    manifest: MASTUArchiveQueryManifest,
    results: tuple[MASTUArchiveAcquisitionResult, ...],
    result_id: str,
    compact_manifest_id: str,
) -> MASTUArchiveCompactManifestResult:
    """Reduce exact group terminals after all source tasks, without source contact."""

    query_by_id = {value.query_id: value for value in manifest.queries}
    result_by_query = {value.query.object_id: value for value in results}
    if len(result_by_query) != len(results) or set(result_by_query) != set(query_by_id):
        raise ValueError("archive compact reduction differs from the frozen query roster")
    identities = tuple(
        sorted(
            (ObjectIdentity.from_record(value.result_id, value) for value in results),
            key=lambda value: value.object_id,
        )
    )
    failures = tuple(
        value
        for value in results
        if value.disposition is not MASTUArchiveAcquisitionDisposition.COMPLETE
    )
    if failures:
        disposition = (
            MASTUArchiveAcquisitionDisposition.AUTHORITY_REQUIRED
            if any(
                value.disposition is MASTUArchiveAcquisitionDisposition.AUTHORITY_REQUIRED
                for value in failures
            )
            else MASTUArchiveAcquisitionDisposition.PROVIDER_ERROR
            if any(
                value.disposition is MASTUArchiveAcquisitionDisposition.PROVIDER_ERROR
                for value in failures
            )
            else MASTUArchiveAcquisitionDisposition.INVALID_RESPONSE
        )
        return MASTUArchiveCompactManifestResult(
            result_id=result_id,
            query_manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
            acquisition_results=identities,
            compact_manifest=None,
            disposition=disposition,
            reason_codes=tuple(
                sorted(
                    {
                        f"{value.query.object_id}:{reason}"
                        for value in failures
                        for reason in value.reason_codes
                    }
                )
            ),
            scientific_verdict_constructed=False,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )
    receipts = tuple(value.object_receipt for value in results)
    if any(value is None for value in receipts):
        raise AssertionError("complete archive acquisition lost its receipt")
    compact = MASTUArchiveCompactUnitManifest(
        manifest_id=compact_manifest_id,
        query_manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        object_receipts=tuple(
            sorted(
                (
                    ObjectIdentity.from_record(value.receipt_id, value)
                    for value in receipts
                    if value is not None
                ),
                key=lambda value: value.object_id,
            )
        ),
        preparation_instance_ids=tuple(
            sorted({value.preparation_instance_id for value in manifest.queries})
        ),
        acquisition_group_ids=tuple(
            sorted({value.acquisition_group_id for value in manifest.queries})
        ),
        physical_independent_unit_ids=tuple(
            sorted({value.physical_independent_unit_id for value in manifest.queries})
        ),
        view_ids=tuple(sorted({view for value in manifest.queries for view in value.view_ids})),
        complete_after_all_object_custody=True,
    )
    return MASTUArchiveCompactManifestResult(
        result_id=result_id,
        query_manifest=ObjectIdentity.from_record(manifest.manifest_id, manifest),
        acquisition_results=identities,
        compact_manifest=compact,
        disposition=MASTUArchiveAcquisitionDisposition.COMPLETE,
        reason_codes=(),
        scientific_verdict_constructed=False,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )


__all__ = [
    'MASTUArchiveAcquisitionDisposition',
    'MASTUArchiveAcquisitionResult',
    'MASTUArchiveCompactManifestResult',
    'MASTUArchiveAcquisitionUnitManifest',
    'MASTUArchiveCompactUnitManifest',
    'MASTUArchiveObjectReceipt',
    'MASTUArchiveQueryManifest',
    'MASTUArchiveQueryClient',
    'MASTUArchiveQuerySourceConfig',
    'MASTUArchiveQuery',
    'MASTUArchiveResponseObject',
    'MASTUArchiveCustodyPort',
    "MAX_MASTU_ARCHIVE_QUERIES",
    'retrieve_mastu_archive_query_manifest',
    'retrieve_mastu_archive_query',
    'reduce_mastu_archive_acquisition_results',
]

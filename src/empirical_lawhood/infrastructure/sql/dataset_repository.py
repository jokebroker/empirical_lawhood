"""Append-only SQLite implementation of the runtime dataset catalog port."""

from __future__ import annotations

import threading
from collections.abc import Callable, Mapping
from math import gcd
from typing import Any, TypeVar

from sqlalchemy import Connection, Engine, Table, and_, exists, or_, select
from sqlalchemy.exc import IntegrityError, OperationalError

from empirical_lawhood.infrastructure.dataset_projection import decode_dataset_record_json
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    CanonicalizationError,
    canonical_json_bytes,
)
from empirical_lawhood.planning.datasets import (
    AcquisitionAttempt,
    DatasetFamily,
    DatasetMaterialization,
    DatasetObservation,
    DatasetRelease,
    EvidenceReference,
    ExperimentDatasetBinding,
    ExternalIdentifier,
    ExternalIdentifierKind,
)
from empirical_lawhood.runtime.datasets import (
    DatasetCatalogCursor,
    DatasetCatalogPage,
    DatasetCatalogProjectionState,
    DatasetCatalogQuery,
    DatasetCatalogRecord,
    DatasetCatalogRecordKind,
    DatasetCatalogSnapshot,
    DatasetCatalogWorkLimit,
    dataset_catalog_record_key,
)

from .database import dataset_catalog_query_preflight_connection, schema_audit
from .migrations.dataset_tables import DATASET_PROJECTION_DIRTY_TRIGGER_DEFINITIONS, DATASET_PROJECTION_DIRTY_TRIGGER_NAMES, normalize_dataset_projection_trigger_sql
from .schema import (
    acquisition_attempt,
    dataset_external_identifier,
    dataset_family,
    dataset_materialization,
    dataset_observation,
    dataset_projection_state,
    dataset_release,
    experiment_dataset_binding,
    storage_root,
)


_SQLITE_PROGRESS_INTERVAL = 100
_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


class DatasetCatalogWriteConflict(RuntimeError):
    """An append attempted to reuse an immutable identity with different content."""


class DatasetCatalogCorruption(RuntimeError):
    """Persisted JSON, fingerprints, indexes, or normalized identifiers disagree."""


class DatasetCatalogWriterViolation(RuntimeError):
    """A non-owner thread attempted a dataset write."""


class DatasetCatalogSnapshotMismatch(RuntimeError):
    """A query names a snapshot other than the currently installed projection."""


class DatasetCatalogQueryWorkLimitExceeded(RuntimeError):
    def __init__(
        self,
        *,
        records_examined: int,
        record_limit: int,
        query_steps: int,
        step_limit: int,
    ) -> None:
        self.records_examined = records_examined
        self.record_limit = record_limit
        self.query_steps = query_steps
        self.step_limit = step_limit
        super().__init__("dataset catalog query exceeded its enforceable work limit")


def _digest(value: str) -> bytes:
    return bytes.fromhex(value)


def _canonical_array(values: tuple[str, ...]) -> str:
    return canonical_json_bytes(values).decode("utf-8")


def _row_mapping(row: Any) -> Mapping[str, Any]:
    return row._mapping  # type: ignore[no-any-return]


def _record_base(record: CanonicalRecord) -> dict[str, object]:
    return {
        "record_schema": record.SCHEMA,
        "record_fingerprint": _digest(record.fingerprint()),
        "record_json": record.canonical_bytes().decode("utf-8"),
    }


def _reference_ids(values: tuple[EvidenceReference, ...]) -> tuple[str, ...]:
    return tuple(value.evidence_id for value in values)


class SQLiteDatasetRepository:
    """Single-writer, insert-or-verify projection over migration-0003 tables."""

    def __init__(self, engine: Engine) -> None:
        self.engine = engine
        self._writer_thread = threading.get_ident()

    def _assert_writer(self) -> None:
        if threading.get_ident() != self._writer_thread:
            raise DatasetCatalogWriterViolation(
                "dataset catalog writes are restricted to the owner thread"
            )

    @staticmethod
    def _pk(
        connection: Connection,
        table: Table,
        id_column: Any,
        identifier: str,
    ) -> int:
        value = connection.execute(select(table.c.pk).where(id_column == identifier)).scalar_one()
        return int(value)

    @classmethod
    def _root_pk(cls, connection: Connection, root_id: str) -> int:
        return cls._pk(connection, storage_root, storage_root.c.storage_root_id, root_id)

    @classmethod
    def _family_values(
        cls,
        _connection: Connection,
        record: DatasetFamily,
    ) -> dict[str, object]:
        return {
            **_record_base(record),
            "family_id": record.family_id,
            "canonical_name": record.canonical_name,
            "provider_id": record.provider_id,
            "description": record.description,
            "keywords_json": _canonical_array(record.keywords),
            "first_observation_id": record.first_observation_id,
            "latest_observation_id": record.latest_observation_id,
            "identity_state": record.identity_state.value,
            "reason_codes_json": _canonical_array(record.reason_codes),
        }

    @classmethod
    def _release_values(
        cls,
        connection: Connection,
        record: DatasetRelease,
    ) -> dict[str, object]:
        selector = record.local_selector
        return {
            **_record_base(record),
            "release_id": record.release_id,
            "family_pk": cls._pk(
                connection,
                dataset_family,
                dataset_family.c.family_id,
                record.family_id,
            ),
            "identity_state": record.identity_state.value,
            "resolution_class": record.resolution_class.value,
            "access_state": record.access_state.value,
            "provider_id": record.provider_id,
            "local_selector_id": None if selector is None else selector.selector_id,
            "local_selector_kind": None if selector is None else selector.kind.value,
            "local_selector_schema": None if selector is None else selector.selector_schema,
            "local_selector_sha256": None
            if selector is None
            else _digest(selector.selector_sha256),
            "local_selector_complete": None if selector is None else int(selector.complete_release),
            "publication_at_utc": record.publication_at_utc,
            "observed_at_utc": record.observed_at_utc,
            "mutable_snapshot": int(record.mutable_snapshot),
            "expected_format_profile_ids_json": _canonical_array(
                record.expected_format_profile_ids
            ),
            "expected_file_count_minimum": record.expected_file_count_minimum,
            "expected_file_count_maximum": record.expected_file_count_maximum,
            "expected_byte_count_minimum": record.expected_byte_count_minimum,
            "expected_byte_count_maximum": record.expected_byte_count_maximum,
            "expected_physical_sha256": (
                None
                if record.expected_physical_sha256 is None
                else _digest(record.expected_physical_sha256)
            ),
            "manifest_sha256": (
                None if record.manifest_sha256 is None else _digest(record.manifest_sha256)
            ),
            "licence_evidence_id": (
                None if record.licence_evidence is None else record.licence_evidence.evidence_id
            ),
            "access_evidence_id": (
                None if record.access_evidence is None else record.access_evidence.evidence_id
            ),
            "evidence_ref_ids_json": _canonical_array(_reference_ids(record.evidence_refs)),
            "reason_codes_json": _canonical_array(record.reason_codes),
        }

    @classmethod
    def _observation_values(
        cls,
        connection: Connection,
        record: DatasetObservation,
    ) -> dict[str, object]:
        return {
            **_record_base(record),
            "observation_id": record.observation_id,
            "provider_id": record.provider_id,
            "adapter_key": record.adapter.implementation_key,
            "adapter_version": record.adapter.implementation_version,
            "adapter_sha256": _digest(record.adapter.implementation_sha256),
            "query_id": record.query_id,
            "canonical_request_sha256": _digest(record.canonical_request_sha256),
            "retrieved_at_utc": record.retrieved_at_utc,
            "response_sha256": _digest(record.response_sha256),
            "storage_root_pk": cls._root_pk(connection, record.storage_root_id),
            "response_relative_locator": record.response_relative_locator,
            "page_index": record.page_index,
            "predecessor_observation_pk": (
                None
                if record.predecessor_observation_id is None
                else cls._pk(
                    connection,
                    dataset_observation,
                    dataset_observation.c.observation_id,
                    record.predecessor_observation_id,
                )
            ),
            "pagination_token_sha256": (
                None
                if record.pagination_token_sha256 is None
                else _digest(record.pagination_token_sha256)
            ),
            "truncated": int(record.truncated),
            "complete": int(record.complete),
            "rate_limited": int(record.rate_limited),
            "schema_drift": int(record.schema_drift),
            "discovery_state": record.discovery_state.value,
            "candidate_release_ids_json": _canonical_array(record.candidate_release_ids),
            "unresolved_lead_ids_json": _canonical_array(record.unresolved_lead_ids),
            "evidence_ref_ids_json": _canonical_array(_reference_ids(record.evidence_refs)),
            "reason_codes_json": _canonical_array(record.reason_codes),
        }

    @classmethod
    def _materialization_values(
        cls,
        connection: Connection,
        record: DatasetMaterialization,
    ) -> dict[str, object]:
        selector = record.selector
        logical = record.logical_identity
        verifier = record.verifier
        return {
            **_record_base(record),
            "materialization_id": record.materialization_id,
            "release_pk": cls._pk(
                connection,
                dataset_release,
                dataset_release.c.release_id,
                record.release_id,
            ),
            "materialization_class": record.materialization_class.value,
            "evidence_class": record.evidence_class.value,
            "outcome_access": record.outcome_access.value,
            "storage_root_pk": cls._root_pk(connection, record.storage_root_id),
            "relative_locator": record.relative_locator,
            "selector_id": selector.selector_id,
            "selector_kind": selector.kind.value,
            "selector_schema": selector.selector_schema,
            "selector_sha256": _digest(selector.selector_sha256),
            "selector_complete": int(selector.complete_release),
            "physical_sha256": (
                None if record.physical_sha256 is None else _digest(record.physical_sha256)
            ),
            "byte_size": record.byte_size,
            "file_count": record.file_count,
            "media_type": record.media_type,
            "format_profile_id": record.format_profile_id,
            "logical_decoder_key": (
                None if logical is None else logical.decoder.implementation_key
            ),
            "logical_decoder_version": (
                None if logical is None else logical.decoder.implementation_version
            ),
            "logical_decoder_sha256": (
                None if logical is None else _digest(logical.decoder.implementation_sha256)
            ),
            "logical_schema": None if logical is None else logical.logical_schema,
            "logical_sha256": None if logical is None else _digest(logical.logical_sha256),
            "custody_state": record.custody_state.value,
            "verifier_key": None if verifier is None else verifier.implementation_key,
            "verifier_version": None if verifier is None else verifier.implementation_version,
            "verifier_sha256": (
                None if verifier is None else _digest(verifier.implementation_sha256)
            ),
            "verification_policy_id": record.verification_policy_id,
            "verification_policy_sha256": (
                None
                if record.verification_policy_sha256 is None
                else _digest(record.verification_policy_sha256)
            ),
            "verified_at_utc": record.verified_at_utc,
            "verification_evidence_ref_ids_json": _canonical_array(
                _reference_ids(record.verification_evidence_refs)
            ),
            "manifest_relative_locator": record.manifest_relative_locator,
            "receipt_relative_locator": record.receipt_relative_locator,
            "reason_codes_json": _canonical_array(record.reason_codes),
        }

    @classmethod
    def _attempt_values(
        cls,
        connection: Connection,
        record: AcquisitionAttempt,
    ) -> dict[str, object]:
        preview = record.preview
        authorization = record.authorization
        return {
            **_record_base(record),
            "attempt_id": record.attempt_id,
            "release_pk": cls._pk(
                connection,
                dataset_release,
                dataset_release.c.release_id,
                record.release_id,
            ),
            "provider_capability_key": record.provider_capability_key,
            "source_endpoint_id": record.source_endpoint_id,
            "preview_evidence_id": preview.evidence_id,
            "preview_evidence_schema": preview.evidence_schema,
            "preview_sha256": _digest(preview.evidence_sha256),
            "preview_storage_root_pk": cls._root_pk(connection, preview.storage_root_id),
            "preview_relative_locator": preview.relative_locator,
            "expected_physical_sha256": (
                None
                if record.expected_physical_sha256 is None
                else _digest(record.expected_physical_sha256)
            ),
            "expected_size_bytes": record.expected_size_bytes,
            "expected_format_profile_ids_json": _canonical_array(
                record.expected_format_profile_ids
            ),
            "authorization_evidence_id": (
                None if authorization is None else authorization.evidence_id
            ),
            "authorization_evidence_schema": (
                None if authorization is None else authorization.evidence_schema
            ),
            "authorization_sha256": (
                None if authorization is None else _digest(authorization.evidence_sha256)
            ),
            "authorization_storage_root_pk": (
                None
                if authorization is None
                else cls._root_pk(connection, authorization.storage_root_id)
            ),
            "authorization_relative_locator": (
                None if authorization is None else authorization.relative_locator
            ),
            "authorization_valid_from_utc": record.authorization_valid_from_utc,
            "authorization_valid_until_utc": record.authorization_valid_until_utc,
            "requested_bytes": record.requested_bytes,
            "observed_bytes": record.observed_bytes,
            "storage_root_pk": cls._root_pk(connection, record.storage_root_id),
            "temporary_relative_locator": record.temporary_relative_locator,
            "final_relative_locator": record.final_relative_locator,
            "retry_count": record.retry_count,
            "resumed_from_attempt_pk": (
                None
                if record.resumed_from_attempt_id is None
                else cls._pk(
                    connection,
                    acquisition_attempt,
                    acquisition_attempt.c.attempt_id,
                    record.resumed_from_attempt_id,
                )
            ),
            "acquisition_state": record.acquisition_state.value,
            "result_materialization_pk": (
                None
                if record.result_materialization_id is None
                else cls._pk(
                    connection,
                    dataset_materialization,
                    dataset_materialization.c.materialization_id,
                    record.result_materialization_id,
                )
            ),
            "reason_codes_json": _canonical_array(record.reason_codes),
            "evidence_ref_ids_json": _canonical_array(_reference_ids(record.evidence_refs)),
        }

    @classmethod
    def _binding_values(
        cls,
        connection: Connection,
        record: ExperimentDatasetBinding,
    ) -> dict[str, object]:
        selector = record.selector
        transform = record.transform_reference
        implementation = record.transform_implementation
        receipt = record.binding_receipt
        return {
            **_record_base(record),
            "binding_id": record.binding_id,
            "experiment_spec_id": record.experiment_spec_id,
            "run_id": record.run_id,
            "release_pk": cls._pk(
                connection,
                dataset_release,
                dataset_release.c.release_id,
                record.release_id,
            ),
            "materialization_pk": cls._pk(
                connection,
                dataset_materialization,
                dataset_materialization.c.materialization_id,
                record.materialization_id,
            ),
            "selector_id": selector.selector_id,
            "selector_kind": selector.kind.value,
            "selector_schema": selector.selector_schema,
            "selector_sha256": _digest(selector.selector_sha256),
            "selector_complete": int(selector.complete_release),
            "binding_role": record.role.value,
            "outcome_access": record.outcome_access.value,
            "transform_reference_id": None if transform is None else transform.evidence_id,
            "transform_reference_schema": (
                None if transform is None else transform.evidence_schema
            ),
            "transform_reference_sha256": (
                None if transform is None else _digest(transform.evidence_sha256)
            ),
            "transform_storage_root_pk": (
                None if transform is None else cls._root_pk(connection, transform.storage_root_id)
            ),
            "transform_relative_locator": (
                None if transform is None else transform.relative_locator
            ),
            "transform_implementation_key": (
                None if implementation is None else implementation.implementation_key
            ),
            "transform_implementation_version": (
                None if implementation is None else implementation.implementation_version
            ),
            "transform_implementation_sha256": (
                None if implementation is None else _digest(implementation.implementation_sha256)
            ),
            "binding_state": record.binding_state.value,
            "reason_codes_json": _canonical_array(record.reason_codes),
            "binding_receipt_id": None if receipt is None else receipt.evidence_id,
            "binding_receipt_schema": None if receipt is None else receipt.evidence_schema,
            "binding_receipt_sha256": (
                None if receipt is None else _digest(receipt.evidence_sha256)
            ),
            "binding_receipt_storage_root_pk": (
                None if receipt is None else cls._root_pk(connection, receipt.storage_root_id)
            ),
            "binding_receipt_relative_locator": (
                None if receipt is None else receipt.relative_locator
            ),
            "evidence_ref_ids_json": _canonical_array(_reference_ids(record.evidence_refs)),
        }

    @staticmethod
    def _assert_indexed_values(
        row: Mapping[str, Any],
        expected: Mapping[str, object],
    ) -> None:
        for column, value in expected.items():
            if row[column] != value:
                raise DatasetCatalogCorruption(
                    f"dataset indexed column {column} differs from canonical record_json"
                )

    @classmethod
    def _decode_and_verify_row(
        cls,
        connection: Connection,
        row: Mapping[str, Any],
        record_type: type[_RecordT],
        values: Callable[[Connection, _RecordT], dict[str, object]],
    ) -> _RecordT:
        try:
            record = decode_dataset_record_json(row["record_json"], record_type)
        except (CanonicalizationError, TypeError, ValueError) as error:
            raise DatasetCatalogCorruption("dataset record_json failed strict decode") from error
        expected = values(connection, record)
        cls._assert_indexed_values(row, expected)
        return record

    @classmethod
    def _insert_or_verify(
        cls,
        connection: Connection,
        table: Table,
        id_column: Any,
        identifier: str,
        record: _RecordT,
        values: Callable[[Connection, _RecordT], dict[str, object]],
    ) -> tuple[int, bool]:
        existing = connection.execute(select(table).where(id_column == identifier)).first()
        if existing is not None:
            row = _row_mapping(existing)
            observed = cls._decode_and_verify_row(connection, row, type(record), values)
            if observed != record:
                raise DatasetCatalogWriteConflict(
                    "append-only dataset identity already exists with different content"
                )
            return int(row["pk"]), False
        result = connection.execute(table.insert().values(**values(connection, record)))
        primary_key = result.inserted_primary_key
        if not primary_key or primary_key[0] is None:
            raise RuntimeError("dataset catalog insert did not return a primary key")
        return int(primary_key[0]), True

    @classmethod
    def _verify_external_identifiers(
        cls,
        connection: Connection,
        *,
        owner_kind: str,
        owner_pk: int,
        identifiers: tuple[ExternalIdentifier, ...],
        insert_missing: bool,
    ) -> None:
        owner_column = {
            "FAMILY": dataset_external_identifier.c.family_pk,
            "RELEASE": dataset_external_identifier.c.release_pk,
            "OBSERVATION": dataset_external_identifier.c.observation_pk,
        }[owner_kind]
        rows = tuple(
            map(
                _row_mapping,
                connection.execute(
                    select(dataset_external_identifier)
                    .where(owner_column == owner_pk)
                    .order_by(
                        dataset_external_identifier.c.identifier_kind,
                        dataset_external_identifier.c.namespace,
                        dataset_external_identifier.c.value,
                    )
                ),
            )
        )
        if not rows and insert_missing:
            owner_values = {
                "family_pk": owner_pk if owner_kind == "FAMILY" else None,
                "release_pk": owner_pk if owner_kind == "RELEASE" else None,
                "observation_pk": owner_pk if owner_kind == "OBSERVATION" else None,
            }
            for identifier in identifiers:
                connection.execute(
                    dataset_external_identifier.insert().values(
                        identifier_fingerprint=_digest(identifier.fingerprint()),
                        owner_kind=owner_kind,
                        identifier_kind=identifier.kind.value,
                        namespace=identifier.namespace,
                        value=identifier.value,
                        **owner_values,
                    )
                )
            rows = tuple(
                map(
                    _row_mapping,
                    connection.execute(
                        select(dataset_external_identifier)
                        .where(owner_column == owner_pk)
                        .order_by(
                            dataset_external_identifier.c.identifier_kind,
                            dataset_external_identifier.c.namespace,
                            dataset_external_identifier.c.value,
                        )
                    ),
                )
            )
        observed: list[ExternalIdentifier] = []
        for row in rows:
            try:
                identifier = ExternalIdentifier(
                    kind=ExternalIdentifierKind(row["identifier_kind"]),
                    namespace=row["namespace"],
                    value=row["value"],
                )
            except (TypeError, ValueError) as error:
                raise DatasetCatalogCorruption(
                    "normalized external identifier is invalid"
                ) from error
            if row["owner_kind"] != owner_kind or row["identifier_fingerprint"] != _digest(
                identifier.fingerprint()
            ):
                raise DatasetCatalogCorruption(
                    "normalized external identifier differs from its fingerprint or owner"
                )
            observed.append(identifier)
        if tuple(observed) != identifiers:
            raise DatasetCatalogCorruption(
                "normalized external identifiers differ from canonical record_json"
            )

    def append_snapshot(self, snapshot: DatasetCatalogSnapshot) -> None:
        if not isinstance(snapshot, DatasetCatalogSnapshot):
            raise TypeError("snapshot must be DatasetCatalogSnapshot")
        self._assert_writer()
        try:
            with self.engine.begin() as connection:
                self._snapshot(connection)
                for family_record in snapshot.families:
                    owner_pk, inserted = self._insert_or_verify(
                        connection,
                        dataset_family,
                        dataset_family.c.family_id,
                        family_record.family_id,
                        family_record,
                        self._family_values,
                    )
                    self._verify_external_identifiers(
                        connection,
                        owner_kind="FAMILY",
                        owner_pk=owner_pk,
                        identifiers=family_record.external_identifiers,
                        insert_missing=inserted,
                    )
                for release_record in snapshot.releases:
                    owner_pk, inserted = self._insert_or_verify(
                        connection,
                        dataset_release,
                        dataset_release.c.release_id,
                        release_record.release_id,
                        release_record,
                        self._release_values,
                    )
                    self._verify_external_identifiers(
                        connection,
                        owner_kind="RELEASE",
                        owner_pk=owner_pk,
                        identifiers=release_record.external_identifiers,
                        insert_missing=inserted,
                    )
                for observation_record in sorted(
                    snapshot.observations,
                    key=lambda value: (value.page_index, value.observation_id),
                ):
                    owner_pk, inserted = self._insert_or_verify(
                        connection,
                        dataset_observation,
                        dataset_observation.c.observation_id,
                        observation_record.observation_id,
                        observation_record,
                        self._observation_values,
                    )
                    self._verify_external_identifiers(
                        connection,
                        owner_kind="OBSERVATION",
                        owner_pk=owner_pk,
                        identifiers=observation_record.provider_object_ids,
                        insert_missing=inserted,
                    )
                for materialization_record in snapshot.materializations:
                    self._insert_or_verify(
                        connection,
                        dataset_materialization,
                        dataset_materialization.c.materialization_id,
                        materialization_record.materialization_id,
                        materialization_record,
                        self._materialization_values,
                    )
                for attempt_record in sorted(
                    snapshot.acquisition_attempts,
                    key=lambda value: (value.retry_count, value.attempt_id),
                ):
                    self._insert_or_verify(
                        connection,
                        acquisition_attempt,
                        acquisition_attempt.c.attempt_id,
                        attempt_record.attempt_id,
                        attempt_record,
                        self._attempt_values,
                    )
                for binding_record in snapshot.bindings:
                    self._insert_or_verify(
                        connection,
                        experiment_dataset_binding,
                        experiment_dataset_binding.c.binding_id,
                        binding_record.binding_id,
                        binding_record,
                        self._binding_values,
                    )
                observed = self._snapshot(connection, verify_state=False)
                self._advance_projection_state(connection, observed)
        except IntegrityError as error:
            raise DatasetCatalogWriteConflict(
                "dataset append conflicts with an immutable catalog identity"
            ) from error

    @classmethod
    def _read_records(
        cls,
        connection: Connection,
        table: Table,
        id_column: Any,
        record_type: type[_RecordT],
        values: Callable[[Connection, _RecordT], dict[str, object]],
    ) -> tuple[_RecordT, ...]:
        records: list[_RecordT] = []
        rows = connection.execute(select(table).order_by(id_column))
        for row in map(_row_mapping, rows):
            records.append(
                cls._decode_and_verify_row(
                    connection,
                    row,
                    record_type,
                    values,
                )
            )
        return tuple(records)

    @classmethod
    def _snapshot(
        cls,
        connection: Connection,
        *,
        verify_state: bool = True,
    ) -> DatasetCatalogSnapshot:
        if verify_state:
            cls._require_projection_triggers(connection)
            cls._require_clean_projection_state(connection)
        families = cls._read_records(
            connection,
            dataset_family,
            dataset_family.c.family_id,
            DatasetFamily,
            cls._family_values,
        )
        releases = cls._read_records(
            connection,
            dataset_release,
            dataset_release.c.release_id,
            DatasetRelease,
            cls._release_values,
        )
        observations = cls._read_records(
            connection,
            dataset_observation,
            dataset_observation.c.observation_id,
            DatasetObservation,
            cls._observation_values,
        )
        materializations = cls._read_records(
            connection,
            dataset_materialization,
            dataset_materialization.c.materialization_id,
            DatasetMaterialization,
            cls._materialization_values,
        )
        attempts = cls._read_records(
            connection,
            acquisition_attempt,
            acquisition_attempt.c.attempt_id,
            AcquisitionAttempt,
            cls._attempt_values,
        )
        bindings = cls._read_records(
            connection,
            experiment_dataset_binding,
            experiment_dataset_binding.c.binding_id,
            ExperimentDatasetBinding,
            cls._binding_values,
        )
        for family_record in families:
            cls._verify_external_identifiers(
                connection,
                owner_kind="FAMILY",
                owner_pk=cls._pk(
                    connection,
                    dataset_family,
                    dataset_family.c.family_id,
                    family_record.family_id,
                ),
                identifiers=family_record.external_identifiers,
                insert_missing=False,
            )
        for release_record in releases:
            cls._verify_external_identifiers(
                connection,
                owner_kind="RELEASE",
                owner_pk=cls._pk(
                    connection,
                    dataset_release,
                    dataset_release.c.release_id,
                    release_record.release_id,
                ),
                identifiers=release_record.external_identifiers,
                insert_missing=False,
            )
        for observation_record in observations:
            cls._verify_external_identifiers(
                connection,
                owner_kind="OBSERVATION",
                owner_pk=cls._pk(
                    connection,
                    dataset_observation,
                    dataset_observation.c.observation_id,
                    observation_record.observation_id,
                ),
                identifiers=observation_record.provider_object_ids,
                insert_missing=False,
            )
        snapshot = DatasetCatalogSnapshot(
            families=families,
            releases=releases,
            observations=observations,
            materializations=materializations,
            acquisition_attempts=attempts,
            bindings=bindings,
        )
        if verify_state:
            cls._verify_projection_state(connection, snapshot)
        return snapshot

    @staticmethod
    def _projection_state_values(
        snapshot: DatasetCatalogSnapshot,
        *,
        revision: int,
    ) -> dict[str, object]:
        return {
            "state_id": 1,
            "snapshot_schema": snapshot.SCHEMA,
            "snapshot_fingerprint": _digest(snapshot.fingerprint()),
            "family_count": len(snapshot.families),
            "release_count": len(snapshot.releases),
            "observation_count": len(snapshot.observations),
            "materialization_count": len(snapshot.materializations),
            "acquisition_attempt_count": len(snapshot.acquisition_attempts),
            "binding_count": len(snapshot.bindings),
            "canonical_byte_count": len(snapshot.canonical_bytes()),
            "projection_valid": 1,
            "projection_revision": revision,
        }

    @classmethod
    def _projection_state(cls, connection: Connection) -> Mapping[str, Any]:
        row = connection.execute(
            select(dataset_projection_state).where(dataset_projection_state.c.state_id == 1)
        ).first()
        if row is None:
            raise DatasetCatalogCorruption("dataset projection state singleton is missing")
        return _row_mapping(row)

    @staticmethod
    def _require_projection_triggers(connection: Connection) -> None:
        observed_list: list[tuple[str, str]] = []
        for raw_name, raw_sql in connection.exec_driver_sql(
            "SELECT name, sql FROM sqlite_schema WHERE type = 'trigger' ORDER BY name LIMIT ?",
            (len(DATASET_PROJECTION_DIRTY_TRIGGER_NAMES) + 1,),
        ):
            try:
                normalized_sql = normalize_dataset_projection_trigger_sql(raw_sql)
            except (TypeError, UnicodeError, ValueError) as error:
                raise DatasetCatalogCorruption(
                    "dataset projection dirty-trigger definition is invalid"
                ) from error
            observed_list.append((str(raw_name), normalized_sql))
        observed = tuple(sorted(observed_list))
        if observed != DATASET_PROJECTION_DIRTY_TRIGGER_DEFINITIONS:
            raise DatasetCatalogCorruption("dataset projection dirty-trigger definitions differ")

    @classmethod
    def _require_clean_projection_state(
        cls,
        connection: Connection,
    ) -> DatasetCatalogProjectionState:
        row = cls._projection_state(connection)
        projection_valid = row["projection_valid"]
        if (
            not isinstance(projection_valid, int)
            or isinstance(projection_valid, bool)
            or projection_valid not in {0, 1}
        ):
            raise DatasetCatalogCorruption("dataset projection validity marker is invalid")
        if projection_valid != 1:
            raise DatasetCatalogCorruption(
                "dataset projection is dirty after an unauthenticated mutation"
            )
        fingerprint = row["snapshot_fingerprint"]
        if not isinstance(fingerprint, bytes) or len(fingerprint) != 32:
            raise DatasetCatalogCorruption("dataset projection state fingerprint is invalid")
        try:
            return DatasetCatalogProjectionState(
                snapshot_schema=row["snapshot_schema"],
                snapshot_fingerprint=fingerprint.hex(),
                family_count=row["family_count"],
                release_count=row["release_count"],
                observation_count=row["observation_count"],
                materialization_count=row["materialization_count"],
                acquisition_attempt_count=row["acquisition_attempt_count"],
                binding_count=row["binding_count"],
                canonical_byte_count=row["canonical_byte_count"],
                projection_revision=row["projection_revision"],
            )
        except (TypeError, ValueError) as error:
            raise DatasetCatalogCorruption(
                "dataset projection state metadata is invalid"
            ) from error

    @classmethod
    def _validate_query_projection_state(
        cls,
        connection: Connection,
        *,
        snapshot_fingerprint: str,
        projection_state_fingerprint: str,
    ) -> DatasetCatalogProjectionState:
        state = cls._require_clean_projection_state(connection)
        if state.snapshot_fingerprint != snapshot_fingerprint:
            raise DatasetCatalogSnapshotMismatch(
                "dataset query snapshot fingerprint differs from installed projection"
            )
        if state.fingerprint() != projection_state_fingerprint:
            raise DatasetCatalogSnapshotMismatch(
                "dataset query projection state differs from installed projection"
            )
        return state

    @classmethod
    def _verify_projection_state(
        cls,
        connection: Connection,
        snapshot: DatasetCatalogSnapshot,
    ) -> None:
        state = cls._require_clean_projection_state(connection)
        row = cls._projection_state(connection)
        cls._assert_indexed_values(
            row,
            cls._projection_state_values(
                snapshot,
                revision=state.projection_revision,
            ),
        )

    @classmethod
    def _advance_projection_state(
        cls,
        connection: Connection,
        snapshot: DatasetCatalogSnapshot,
    ) -> None:
        row = cls._projection_state(connection)
        revision = row["projection_revision"]
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise DatasetCatalogCorruption("dataset projection revision is invalid")
        projection_valid = row["projection_valid"]
        if projection_valid not in {0, 1}:
            raise DatasetCatalogCorruption("dataset projection validity marker is invalid")
        fingerprint = _digest(snapshot.fingerprint())
        if projection_valid == 1:
            cls._assert_indexed_values(
                row,
                cls._projection_state_values(snapshot, revision=revision),
            )
            return
        if row["snapshot_fingerprint"] == fingerprint:
            raise DatasetCatalogCorruption(
                "dirty dataset projection did not change its snapshot identity"
            )
        result = connection.execute(
            dataset_projection_state.update()
            .where(dataset_projection_state.c.state_id == 1)
            .values(**cls._projection_state_values(snapshot, revision=revision))
        )
        if result.rowcount != 1:
            raise DatasetCatalogCorruption(
                "dataset projection state singleton could not be authenticated"
            )

    def snapshot(self) -> DatasetCatalogSnapshot:
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN")
            try:
                return self._snapshot(connection)
            finally:
                connection.rollback()

    def projection_state(
        self,
        work_limit: DatasetCatalogWorkLimit,
    ) -> DatasetCatalogProjectionState:
        """Return the authenticated clean cursor boundary without loading records."""

        if not isinstance(work_limit, DatasetCatalogWorkLimit):
            raise TypeError("work_limit must be DatasetCatalogWorkLimit")
        work_steps = 0
        work_exhausted = False
        interval = gcd(_SQLITE_PROGRESS_INTERVAL, work_limit.max_query_steps)

        def enforce_work_limit() -> int:
            nonlocal work_steps, work_exhausted
            work_steps += interval
            if work_steps >= work_limit.max_query_steps:
                work_exhausted = True
                return 1
            return 0

        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN")
            try:
                if not dataset_catalog_query_preflight_connection(connection).passed:
                    raise DatasetCatalogCorruption("dataset catalog query preflight failed")
                driver_connection = connection.connection.driver_connection
                set_progress_handler = getattr(
                    driver_connection,
                    "set_progress_handler",
                    None,
                )
                if set_progress_handler is None:
                    raise RuntimeError("dataset catalog backend cannot enforce a SQLite work limit")
                set_progress_handler(enforce_work_limit, interval)
                try:
                    return self._require_clean_projection_state(connection)
                except OperationalError as error:
                    if work_exhausted:
                        raise DatasetCatalogQueryWorkLimitExceeded(
                            records_examined=0,
                            record_limit=work_limit.max_records_examined,
                            query_steps=work_steps,
                            step_limit=work_limit.max_query_steps,
                        ) from error
                    raise
                finally:
                    set_progress_handler(None, 0)
            finally:
                connection.rollback()

    def has_records(self) -> bool:
        with self.engine.connect() as connection:
            for table in (
                dataset_family,
                dataset_release,
                dataset_observation,
                dataset_materialization,
                acquisition_attempt,
                experiment_dataset_binding,
                dataset_external_identifier,
            ):
                if connection.execute(select(table.c.pk).limit(1)).first() is not None:
                    return True
            return False

    @staticmethod
    def _external_owner_predicate(
        owner_column: Any,
        owner_pk_column: Any,
        query: DatasetCatalogQuery,
    ) -> Any:
        conditions = tuple(
            and_(
                dataset_external_identifier.c.identifier_kind == identifier.kind.value,
                dataset_external_identifier.c.namespace == identifier.namespace,
                dataset_external_identifier.c.value == identifier.value,
            )
            for identifier in query.boundary.external_identifiers
        )
        return exists(
            select(1).where(
                owner_column == owner_pk_column,
                or_(*conditions),
            )
        )

    @staticmethod
    def _cursor_predicates(
        kind: DatasetCatalogRecordKind,
        id_column: Any,
        query: DatasetCatalogQuery,
    ) -> tuple[Any, ...] | None:
        if query.cursor is None:
            return ()
        if kind.value < query.cursor.after_record_kind.value:
            return None
        if kind is query.cursor.after_record_kind:
            return (id_column > query.cursor.after_record_id,)
        return ()

    @classmethod
    def _query_statements(
        cls,
        query: DatasetCatalogQuery,
        *,
        row_cap: int,
    ) -> tuple[tuple[DatasetCatalogRecordKind, Any, type[CanonicalRecord]], ...]:
        boundary = query.boundary
        statements: list[tuple[DatasetCatalogRecordKind, Any, type[CanonicalRecord]]] = []

        def add(
            kind: DatasetCatalogRecordKind,
            table: Table,
            id_column: Any,
            record_type: type[CanonicalRecord],
            statement: Any,
            predicates: list[Any],
            *,
            eligible: bool,
        ) -> None:
            cursor = cls._cursor_predicates(kind, id_column, query)
            if eligible and cursor is not None:
                statements.append(
                    (
                        kind,
                        statement.where(*predicates, *cursor).order_by(id_column).limit(row_cap),
                        record_type,
                    )
                )

        family_predicates: list[Any] = []
        if boundary.family_ids:
            family_predicates.append(dataset_family.c.family_id.in_(boundary.family_ids))
        if boundary.provider_ids:
            family_predicates.append(dataset_family.c.provider_id.in_(boundary.provider_ids))
        if boundary.external_identifiers:
            family_predicates.append(
                cls._external_owner_predicate(
                    dataset_external_identifier.c.family_pk,
                    dataset_family.c.pk,
                    query,
                )
            )
        add(
            DatasetCatalogRecordKind.FAMILY,
            dataset_family,
            dataset_family.c.family_id,
            DatasetFamily,
            select(dataset_family),
            family_predicates,
            eligible=not any(
                (
                    boundary.release_ids,
                    boundary.custody_states,
                    boundary.acquisition_states,
                    boundary.experiment_spec_ids,
                    boundary.roles,
                )
            ),
        )

        release_statement = select(
            dataset_release,
            dataset_family.c.family_id.label("_family_id"),
        ).select_from(
            dataset_release.join(
                dataset_family,
                dataset_release.c.family_pk == dataset_family.c.pk,
            )
        )
        release_predicates: list[Any] = []
        if boundary.family_ids:
            release_predicates.append(dataset_family.c.family_id.in_(boundary.family_ids))
        if boundary.release_ids:
            release_predicates.append(dataset_release.c.release_id.in_(boundary.release_ids))
        if boundary.provider_ids:
            release_predicates.append(dataset_release.c.provider_id.in_(boundary.provider_ids))
        if boundary.external_identifiers:
            release_predicates.append(
                cls._external_owner_predicate(
                    dataset_external_identifier.c.release_pk,
                    dataset_release.c.pk,
                    query,
                )
            )
        add(
            DatasetCatalogRecordKind.RELEASE,
            dataset_release,
            dataset_release.c.release_id,
            DatasetRelease,
            release_statement,
            release_predicates,
            eligible=not any(
                (
                    boundary.custody_states,
                    boundary.acquisition_states,
                    boundary.experiment_spec_ids,
                    boundary.roles,
                )
            ),
        )

        observation_predicates: list[Any] = []
        if boundary.provider_ids:
            observation_predicates.append(
                dataset_observation.c.provider_id.in_(boundary.provider_ids)
            )
        if boundary.external_identifiers:
            observation_predicates.append(
                cls._external_owner_predicate(
                    dataset_external_identifier.c.observation_pk,
                    dataset_observation.c.pk,
                    query,
                )
            )
        add(
            DatasetCatalogRecordKind.OBSERVATION,
            dataset_observation,
            dataset_observation.c.observation_id,
            DatasetObservation,
            select(dataset_observation),
            observation_predicates,
            eligible=not any(
                (
                    boundary.family_ids,
                    boundary.release_ids,
                    boundary.custody_states,
                    boundary.acquisition_states,
                    boundary.experiment_spec_ids,
                    boundary.roles,
                )
            ),
        )

        materialization_statement = select(
            dataset_materialization,
            dataset_release.c.release_id.label("_release_id"),
            dataset_family.c.family_id.label("_family_id"),
            dataset_release.c.provider_id.label("_release_provider_id"),
        ).select_from(
            dataset_materialization.join(
                dataset_release,
                dataset_materialization.c.release_pk == dataset_release.c.pk,
            ).join(
                dataset_family,
                dataset_release.c.family_pk == dataset_family.c.pk,
            )
        )
        materialization_predicates: list[Any] = []
        if boundary.family_ids:
            materialization_predicates.append(dataset_family.c.family_id.in_(boundary.family_ids))
        if boundary.release_ids:
            materialization_predicates.append(
                dataset_release.c.release_id.in_(boundary.release_ids)
            )
        if boundary.provider_ids:
            materialization_predicates.append(
                dataset_release.c.provider_id.in_(boundary.provider_ids)
            )
        if boundary.custody_states:
            materialization_predicates.append(
                dataset_materialization.c.custody_state.in_(
                    tuple(value.value for value in boundary.custody_states)
                )
            )
        add(
            DatasetCatalogRecordKind.MATERIALIZATION,
            dataset_materialization,
            dataset_materialization.c.materialization_id,
            DatasetMaterialization,
            materialization_statement,
            materialization_predicates,
            eligible=not any(
                (
                    boundary.external_identifiers,
                    boundary.acquisition_states,
                    boundary.experiment_spec_ids,
                    boundary.roles,
                )
            ),
        )

        attempt_statement = select(
            acquisition_attempt,
            dataset_release.c.release_id.label("_release_id"),
            dataset_family.c.family_id.label("_family_id"),
            dataset_release.c.provider_id.label("_release_provider_id"),
        ).select_from(
            acquisition_attempt.join(
                dataset_release,
                acquisition_attempt.c.release_pk == dataset_release.c.pk,
            ).join(
                dataset_family,
                dataset_release.c.family_pk == dataset_family.c.pk,
            )
        )
        attempt_predicates: list[Any] = []
        if boundary.family_ids:
            attempt_predicates.append(dataset_family.c.family_id.in_(boundary.family_ids))
        if boundary.release_ids:
            attempt_predicates.append(dataset_release.c.release_id.in_(boundary.release_ids))
        if boundary.provider_ids:
            attempt_predicates.append(dataset_release.c.provider_id.in_(boundary.provider_ids))
        if boundary.acquisition_states:
            attempt_predicates.append(
                acquisition_attempt.c.acquisition_state.in_(
                    tuple(value.value for value in boundary.acquisition_states)
                )
            )
        add(
            DatasetCatalogRecordKind.ACQUISITION_ATTEMPT,
            acquisition_attempt,
            acquisition_attempt.c.attempt_id,
            AcquisitionAttempt,
            attempt_statement,
            attempt_predicates,
            eligible=not any(
                (
                    boundary.external_identifiers,
                    boundary.custody_states,
                    boundary.experiment_spec_ids,
                    boundary.roles,
                )
            ),
        )

        binding_statement = select(
            experiment_dataset_binding,
            dataset_release.c.release_id.label("_release_id"),
            dataset_family.c.family_id.label("_family_id"),
            dataset_release.c.provider_id.label("_release_provider_id"),
            dataset_materialization.c.materialization_id.label("_materialization_id"),
        ).select_from(
            experiment_dataset_binding.join(
                dataset_release,
                experiment_dataset_binding.c.release_pk == dataset_release.c.pk,
            )
            .join(
                dataset_family,
                dataset_release.c.family_pk == dataset_family.c.pk,
            )
            .join(
                dataset_materialization,
                experiment_dataset_binding.c.materialization_pk == dataset_materialization.c.pk,
            )
        )
        binding_predicates: list[Any] = []
        if boundary.family_ids:
            binding_predicates.append(dataset_family.c.family_id.in_(boundary.family_ids))
        if boundary.release_ids:
            binding_predicates.append(dataset_release.c.release_id.in_(boundary.release_ids))
        if boundary.provider_ids:
            binding_predicates.append(dataset_release.c.provider_id.in_(boundary.provider_ids))
        if boundary.experiment_spec_ids:
            binding_predicates.append(
                experiment_dataset_binding.c.experiment_spec_id.in_(boundary.experiment_spec_ids)
            )
        if boundary.roles:
            binding_predicates.append(
                experiment_dataset_binding.c.binding_role.in_(
                    tuple(value.value for value in boundary.roles)
                )
            )
        add(
            DatasetCatalogRecordKind.BINDING,
            experiment_dataset_binding,
            experiment_dataset_binding.c.binding_id,
            ExperimentDatasetBinding,
            binding_statement,
            binding_predicates,
            eligible=not any(
                (
                    boundary.external_identifiers,
                    boundary.custody_states,
                    boundary.acquisition_states,
                )
            ),
        )
        return tuple(sorted(statements, key=lambda value: value[0].value))

    @staticmethod
    def _query_record_id(kind: DatasetCatalogRecordKind, row: Mapping[str, Any]) -> str:
        return str(
            row[
                {
                    DatasetCatalogRecordKind.FAMILY: "family_id",
                    DatasetCatalogRecordKind.RELEASE: "release_id",
                    DatasetCatalogRecordKind.OBSERVATION: "observation_id",
                    DatasetCatalogRecordKind.MATERIALIZATION: "materialization_id",
                    DatasetCatalogRecordKind.ACQUISITION_ATTEMPT: "attempt_id",
                    DatasetCatalogRecordKind.BINDING: "binding_id",
                }[kind]
            ]
        )

    @classmethod
    def _decode_query_record(
        cls,
        connection: Connection,
        kind: DatasetCatalogRecordKind,
        row: Mapping[str, Any],
        record_type: type[CanonicalRecord],
    ) -> DatasetCatalogRecord:
        try:
            record = decode_dataset_record_json(row["record_json"], record_type)
        except (CanonicalizationError, TypeError, ValueError) as error:
            raise DatasetCatalogCorruption("dataset record_json failed strict decode") from error
        if isinstance(record, DatasetFamily):
            expected = cls._family_values(connection, record)
        elif isinstance(record, DatasetRelease):
            expected = cls._release_values(connection, record)
        elif isinstance(record, DatasetObservation):
            expected = cls._observation_values(connection, record)
        elif isinstance(record, DatasetMaterialization):
            expected = cls._materialization_values(connection, record)
        elif isinstance(record, AcquisitionAttempt):
            expected = cls._attempt_values(connection, record)
        else:
            assert isinstance(record, ExperimentDatasetBinding)
            expected = cls._binding_values(connection, record)
        cls._assert_indexed_values(row, expected)
        observed_kind, observed_id = dataset_catalog_record_key(record)
        if observed_kind is not kind or observed_id != cls._query_record_id(kind, row):
            raise DatasetCatalogCorruption("dataset query identity differs from record_json")
        if isinstance(record, DatasetFamily):
            cls._verify_external_identifiers(
                connection,
                owner_kind="FAMILY",
                owner_pk=int(row["pk"]),
                identifiers=record.external_identifiers,
                insert_missing=False,
            )
        elif isinstance(record, DatasetRelease):
            cls._verify_external_identifiers(
                connection,
                owner_kind="RELEASE",
                owner_pk=int(row["pk"]),
                identifiers=record.external_identifiers,
                insert_missing=False,
            )
        elif isinstance(record, DatasetObservation):
            cls._verify_external_identifiers(
                connection,
                owner_kind="OBSERVATION",
                owner_pk=int(row["pk"]),
                identifiers=record.provider_object_ids,
                insert_missing=False,
            )
        return record

    def query(self, query: DatasetCatalogQuery) -> DatasetCatalogPage:
        if not isinstance(query, DatasetCatalogQuery):
            raise TypeError("query must be DatasetCatalogQuery")
        work_steps = 0
        work_exhausted = False
        interval = gcd(
            _SQLITE_PROGRESS_INTERVAL,
            query.boundary.work_limit.max_query_steps,
        )

        def enforce_work_limit() -> int:
            nonlocal work_steps, work_exhausted
            work_steps += interval
            if work_steps >= query.boundary.work_limit.max_query_steps:
                work_exhausted = True
                return 1
            return 0

        with self.engine.connect() as connection:
            # Pysqlite's default transaction mode does not BEGIN for SELECT.
            # Pin preflight, state, candidates, and strict decode to one read view.
            connection.exec_driver_sql("BEGIN")
            try:
                if not dataset_catalog_query_preflight_connection(connection).passed:
                    raise DatasetCatalogCorruption("dataset catalog query preflight failed")
                driver_connection = connection.connection.driver_connection
                set_progress_handler = getattr(
                    driver_connection,
                    "set_progress_handler",
                    None,
                )
                if set_progress_handler is None:
                    raise RuntimeError("dataset catalog backend cannot enforce a SQLite work limit")
                raw_candidates: list[
                    tuple[
                        DatasetCatalogRecordKind,
                        str,
                        Mapping[str, Any],
                        type[CanonicalRecord],
                    ]
                ] = []
                set_progress_handler(enforce_work_limit, interval)
                try:
                    self._validate_query_projection_state(
                        connection,
                        snapshot_fingerprint=query.boundary.snapshot_fingerprint,
                        projection_state_fingerprint=(query.boundary.projection_state_fingerprint),
                    )
                    row_cap = min(
                        query.limit + 1,
                        query.boundary.work_limit.max_records_examined + 1,
                    )
                    for kind, statement, record_type in self._query_statements(
                        query,
                        row_cap=row_cap,
                    ):
                        remaining = row_cap - len(raw_candidates)
                        if remaining == 0:
                            break
                        for row in map(
                            _row_mapping,
                            connection.execute(statement.limit(remaining)),
                        ):
                            raw_candidates.append(
                                (
                                    kind,
                                    self._query_record_id(kind, row),
                                    row,
                                    record_type,
                                )
                            )
                    selected = tuple(raw_candidates)
                    records_examined = len(selected)
                    if records_examined > query.boundary.work_limit.max_records_examined:
                        raise DatasetCatalogQueryWorkLimitExceeded(
                            records_examined=records_examined,
                            record_limit=query.boundary.work_limit.max_records_examined,
                            query_steps=max(work_steps, 1),
                            step_limit=query.boundary.work_limit.max_query_steps,
                        )
                    decoded = tuple(
                        self._decode_query_record(connection, kind, row, record_type)
                        for kind, _identifier, row, record_type in selected
                    )
                except OperationalError as error:
                    if work_exhausted:
                        raise DatasetCatalogQueryWorkLimitExceeded(
                            records_examined=len(raw_candidates),
                            record_limit=query.boundary.work_limit.max_records_examined,
                            query_steps=work_steps,
                            step_limit=query.boundary.work_limit.max_query_steps,
                        ) from error
                    raise
                finally:
                    set_progress_handler(None, 0)
            finally:
                connection.rollback()
        has_more = len(decoded) > query.limit
        page_records = decoded[: query.limit]
        next_cursor = None
        if has_more and page_records:
            final_kind, final_id = dataset_catalog_record_key(page_records[-1])
            next_cursor = DatasetCatalogCursor(
                snapshot_fingerprint=query.boundary.snapshot_fingerprint,
                projection_state_fingerprint=(query.boundary.projection_state_fingerprint),
                projection_anchor_fingerprint=(query.boundary.projection_anchor_fingerprint),
                query_boundary_fingerprint=query.boundary.fingerprint(),
                after_record_kind=final_kind,
                after_record_id=final_id,
            )
        return DatasetCatalogPage(
            boundary=query.boundary,
            limit=query.limit,
            records=page_records,
            records_examined=records_examined,
            query_steps=max(work_steps, 1) if records_examined else work_steps,
            has_more=has_more,
            next_cursor=next_cursor,
        )

    def integrity_check(self) -> tuple[str, ...]:
        reasons: list[str] = []
        audit = schema_audit(self.engine)
        if not audit.passed:
            reasons.append("schema-audit-failed")
        try:
            self.snapshot()
        except (DatasetCatalogCorruption, CanonicalizationError, TypeError, ValueError):
            reasons.append("dataset-record-corrupt")
        return tuple(sorted(set(reasons)))

    def family(self, family_id: str) -> DatasetFamily | None:
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN")
            try:
                self._require_projection_triggers(connection)
                self._require_clean_projection_state(connection)
                row = connection.execute(
                    select(dataset_family).where(dataset_family.c.family_id == family_id)
                ).first()
                if row is None:
                    return None
                row_mapping = _row_mapping(row)
                record = self._decode_and_verify_row(
                    connection,
                    row_mapping,
                    DatasetFamily,
                    self._family_values,
                )
                self._verify_external_identifiers(
                    connection,
                    owner_kind="FAMILY",
                    owner_pk=int(row_mapping["pk"]),
                    identifiers=record.external_identifiers,
                    insert_missing=False,
                )
                return record
            finally:
                connection.rollback()

    def release(self, release_id: str) -> DatasetRelease | None:
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN")
            try:
                self._require_projection_triggers(connection)
                self._require_clean_projection_state(connection)
                row = connection.execute(
                    select(dataset_release).where(dataset_release.c.release_id == release_id)
                ).first()
                if row is None:
                    return None
                row_mapping = _row_mapping(row)
                record = self._decode_and_verify_row(
                    connection,
                    row_mapping,
                    DatasetRelease,
                    self._release_values,
                )
                self._verify_external_identifiers(
                    connection,
                    owner_kind="RELEASE",
                    owner_pk=int(row_mapping["pk"]),
                    identifiers=record.external_identifiers,
                    insert_missing=False,
                )
                return record
            finally:
                connection.rollback()

    def materialization(self, materialization_id: str) -> DatasetMaterialization | None:
        with self.engine.connect() as connection:
            connection.exec_driver_sql("BEGIN")
            try:
                self._require_projection_triggers(connection)
                self._require_clean_projection_state(connection)
                row = connection.execute(
                    select(dataset_materialization).where(
                        dataset_materialization.c.materialization_id == materialization_id
                    )
                ).first()
                if row is None:
                    return None
                return self._decode_and_verify_row(
                    connection,
                    _row_mapping(row),
                    DatasetMaterialization,
                    self._materialization_values,
                )
            finally:
                connection.rollback()


__all__ = [
    "DatasetCatalogCorruption",
    "DatasetCatalogQueryWorkLimitExceeded",
    "DatasetCatalogSnapshotMismatch",
    "DatasetCatalogWriteConflict",
    "DatasetCatalogWriterViolation",
    "SQLiteDatasetRepository",
]

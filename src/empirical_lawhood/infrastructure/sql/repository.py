"""Single-writer append-only implementation of the compact catalog port."""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from sqlalchemy import Connection, Engine, alias, exists, func, or_, select
from sqlalchemy.engine import CursorResult
from sqlalchemy.exc import IntegrityError, OperationalError

from empirical_lawhood.kernel.evidence import VisibilityCeiling
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.catalog import (
    ArtifactLocatorRecord,
    CatalogSnapshot,
    CatalogVerificationStatus,
    ExplorationAttemptRecord,
    KnowledgeEdgeRecord,
    LogicalArtifactRecord,
    MetricDefinitionRecord,
    MetricObservationRecord,
    ReceiptLocatorRecord,
    ScientificObjectKind,
    ScientificObjectRecord,
    StorageRootRecord,
)

from .database import schema_audit
from .schema import (
    artifact,
    exploration_attempt,
    exploration_attempt_artifact,
    exploration_attempt_reason,
    knowledge_edge,
    logical_artifact,
    metric_definition,
    metric_observation,
    projection_tables,
    receipt,
    scientific_object,
    storage_root,
)


class CatalogWriteConflict(RuntimeError):
    pass


class CatalogWriterViolation(RuntimeError):
    pass


DEFAULT_CATALOG_MATCH_COUNT_LIMIT = 1_000
MAX_CATALOG_MATCH_COUNT_LIMIT = 10_000
DEFAULT_CATALOG_QUERY_WORK_LIMIT = 100_000
MAX_CATALOG_QUERY_WORK_LIMIT = 1_000_000
_SQLITE_PROGRESS_INTERVAL = 100


class CatalogQueryWorkLimitExceeded(RuntimeError):
    """Raised when SQLite exhausts the repository-owned VM instruction budget."""

    def __init__(self, *, work_limit: int, work_steps: int) -> None:
        self.work_limit = work_limit
        self.work_steps = work_steps
        super().__init__(
            f"catalog query exceeded its SQLite work limit of {work_limit} instructions"
        )


@dataclass(frozen=True, slots=True)
class CatalogObjectQuery:
    object_id: str | None = None
    kind: str | None = None
    categorical_status: str | None = None
    relation_id: str | None = None
    denominator_gauge_id: str | None = None
    response_gauge_id: str | None = None
    horizon_id: str | None = None
    native_unit: str | None = None
    knowledge_edge_relation: str | None = None
    knowledge_edge_scope_id: str | None = None
    lineage_object_id: str | None = None
    after_object_id: str | None = None
    limit: int = 100
    match_count_limit: int = DEFAULT_CATALOG_MATCH_COUNT_LIMIT

    def __post_init__(self) -> None:
        if self.limit <= 0 or self.limit > 1_000:
            raise ValueError("catalog query limit must be in [1, 1000]")
        if self.match_count_limit <= 0 or self.match_count_limit > MAX_CATALOG_MATCH_COUNT_LIMIT:
            raise ValueError(
                f"catalog match count limit must be in [1, {MAX_CATALOG_MATCH_COUNT_LIMIT}]"
            )


@dataclass(frozen=True, slots=True)
class CatalogObjectQueryRow:
    object_id: str
    kind: str
    object_schema: str
    object_fingerprint: str
    visibility_ceiling: str
    categorical_status: str
    artifact_materialization_id: str | None
    receipt_id: str | None
    storage_root_id: str | None
    external_relative_path: str | None
    external_identity_kind: str | None


@dataclass(frozen=True, slots=True)
class CatalogObjectQueryPage:
    matched_count: int
    match_count_limit: int
    matched_count_truncated: bool
    rows: tuple[CatalogObjectQueryRow, ...]
    has_more: bool
    next_after_object_id: str | None
    query_work_limit: int
    query_work_steps: int


def _digest(value: str) -> bytes:
    return bytes.fromhex(value)


def _hex(value: bytes) -> str:
    return value.hex()


def _row_mapping(row: Any) -> Mapping[str, Any]:
    return row._mapping  # type: ignore[no-any-return]


def _inserted_pk(result: CursorResult[Any]) -> int:
    primary_key = result.inserted_primary_key
    if not primary_key or primary_key[0] is None:
        raise RuntimeError("catalog insert did not return a primary key")
    return int(primary_key[0])


class SQLiteCatalogRepository:
    """One owner-thread writer; all persisted scientific rows are append-only."""

    def __init__(
        self,
        engine: Engine,
        *,
        query_work_limit: int = DEFAULT_CATALOG_QUERY_WORK_LIMIT,
    ) -> None:
        if query_work_limit <= 0 or query_work_limit > MAX_CATALOG_QUERY_WORK_LIMIT:
            raise ValueError(
                f"catalog query work limit must be in [1, {MAX_CATALOG_QUERY_WORK_LIMIT}]"
            )
        self.engine = engine
        self.query_work_limit = query_work_limit
        self._writer_thread = threading.get_ident()

    def _assert_writer(self) -> None:
        if threading.get_ident() != self._writer_thread:
            raise CatalogWriterViolation("catalog writes are restricted to the owner thread")

    @staticmethod
    def _pk(
        connection: Connection,
        table: Any,
        id_column: Any,
        identifier: str,
    ) -> int:
        value = connection.execute(select(table.c.pk).where(id_column == identifier)).scalar_one()
        return int(value)

    def append_snapshot(self, snapshot: CatalogSnapshot) -> None:
        self._assert_writer()
        try:
            with self.engine.begin() as connection:
                self._insert_roots(connection, snapshot)
                self._insert_artifacts(connection, snapshot)
                self._insert_receipts(connection, snapshot)
                self._insert_scientific_objects(connection, snapshot)
                self._insert_edges_and_metrics(connection, snapshot)
                self._insert_exploration_attempts(connection, snapshot)
        except IntegrityError as error:
            raise CatalogWriteConflict("append-only catalog identity already exists") from error

    def _insert_roots(self, connection: Connection, snapshot: CatalogSnapshot) -> None:
        for record in snapshot.storage_roots:
            connection.execute(
                storage_root.insert().values(
                    storage_root_id=record.storage_root_id,
                    logical_name=record.logical_name,
                    canonical_path=record.canonical_path,
                    mount_contract_schema=record.mount_contract_schema,
                    verification_status=record.verification_status.value,
                )
            )

    def _insert_artifacts(self, connection: Connection, snapshot: CatalogSnapshot) -> None:
        for logical_record in snapshot.logical_artifacts:
            connection.execute(
                logical_artifact.insert().values(
                    logical_artifact_id=logical_record.logical_artifact_id,
                    content_sha256=_digest(logical_record.content_sha256),
                    payload_schema=logical_record.payload_schema,
                    profile=logical_record.profile.value,
                    media_type=logical_record.media_type,
                    visibility_ceiling=logical_record.visibility_ceiling.value,
                )
            )
        for locator in snapshot.artifact_locators:
            connection.execute(
                artifact.insert().values(
                    materialization_id=locator.materialization_id,
                    logical_artifact_pk=self._pk(
                        connection,
                        logical_artifact,
                        logical_artifact.c.logical_artifact_id,
                        locator.logical_artifact_id,
                    ),
                    storage_root_pk=self._pk(
                        connection,
                        storage_root,
                        storage_root.c.storage_root_id,
                        locator.storage_root_id,
                    ),
                    relative_path=locator.relative_path,
                    physical_sha256=_digest(locator.physical_sha256),
                    size_bytes=locator.size_bytes,
                    compression=locator.compression,
                    partition_selector=locator.partition_selector,
                    verification_status=locator.verification_status.value,
                )
            )

    def _insert_receipts(self, connection: Connection, snapshot: CatalogSnapshot) -> None:
        for record in snapshot.receipts:
            connection.execute(
                receipt.insert().values(
                    receipt_id=record.receipt_id,
                    run_id=record.run_id,
                    storage_root_pk=self._pk(
                        connection,
                        storage_root,
                        storage_root.c.storage_root_id,
                        record.storage_root_id,
                    ),
                    relative_path=record.relative_path,
                    sha256=_digest(record.sha256),
                    receipt_schema=record.receipt_schema,
                    verification_status=record.verification_status.value,
                )
            )

    def _insert_scientific_objects(self, connection: Connection, snapshot: CatalogSnapshot) -> None:
        for record in snapshot.scientific_objects:
            artifact_pk = (
                None
                if record.artifact_materialization_id is None
                else self._pk(
                    connection,
                    artifact,
                    artifact.c.materialization_id,
                    record.artifact_materialization_id,
                )
            )
            receipt_pk = (
                None
                if record.receipt_id is None
                else self._pk(
                    connection,
                    receipt,
                    receipt.c.receipt_id,
                    record.receipt_id,
                )
            )
            result = connection.execute(
                scientific_object.insert().values(
                    scientific_object_id=record.scientific_object_id,
                    kind=record.kind.value,
                    object_schema=record.object_schema,
                    fingerprint=_digest(record.object_sha256),
                    visibility_ceiling=record.visibility_ceiling.value,
                    categorical_status=record.categorical_status,
                    artifact_pk=artifact_pk,
                    receipt_pk=receipt_pk,
                )
            )
            object_pk = _inserted_pk(result)
            projection = projection_tables[record.kind.value.lower()]
            connection.execute(projection.insert().values(scientific_object_pk=object_pk))

    def _object_pk(self, connection: Connection, object_id: str) -> int:
        return self._pk(
            connection,
            scientific_object,
            scientific_object.c.scientific_object_id,
            object_id,
        )

    def _insert_edges_and_metrics(self, connection: Connection, snapshot: CatalogSnapshot) -> None:
        for edge_record in snapshot.knowledge_edges:
            connection.execute(
                knowledge_edge.insert().values(
                    edge_id=edge_record.edge_id,
                    relation=edge_record.relation,
                    source_object_pk=self._object_pk(connection, edge_record.source_object_id),
                    target_object_pk=self._object_pk(connection, edge_record.target_object_id),
                    evidence_object_pk=(
                        None
                        if edge_record.evidence_object_id is None
                        else self._object_pk(connection, edge_record.evidence_object_id)
                    ),
                    scope_id=edge_record.scope_id,
                )
            )
        for definition_record in snapshot.metric_definitions:
            connection.execute(
                metric_definition.insert().values(
                    metric_definition_id=definition_record.metric_definition_id,
                    metric_name=definition_record.metric_name,
                    dataset_version_id=definition_record.dataset_version_id,
                    relation_id=definition_record.relation_id,
                    denominator_gauge_id=definition_record.denominator_gauge_id,
                    response_gauge_id=definition_record.response_gauge_id,
                    horizon_id=definition_record.horizon_id,
                    native_unit=definition_record.native_unit,
                    aggregation=definition_record.aggregation,
                    direction=definition_record.direction,
                )
            )
        for observation_record in snapshot.metric_observations:
            connection.execute(
                metric_observation.insert().values(
                    metric_observation_id=observation_record.metric_observation_id,
                    metric_definition_pk=self._pk(
                        connection,
                        metric_definition,
                        metric_definition.c.metric_definition_id,
                        observation_record.metric_definition_id,
                    ),
                    scientific_object_pk=self._object_pk(
                        connection, observation_record.scientific_object_id
                    ),
                    point_decimal=str(observation_record.point),
                    lower_decimal=(
                        None if observation_record.lower is None else str(observation_record.lower)
                    ),
                    upper_decimal=(
                        None if observation_record.upper is None else str(observation_record.upper)
                    ),
                    physical_independent_unit_count=(
                        observation_record.physical_independent_unit_count
                    ),
                    numerical_view_count=observation_record.numerical_view_count,
                )
            )

    def _insert_exploration_attempts(
        self, connection: Connection, snapshot: CatalogSnapshot
    ) -> None:
        for record in snapshot.exploration_attempts:
            result = connection.execute(
                exploration_attempt.insert().values(
                    attempt_id=record.attempt_id,
                    analysis_spec_object_pk=self._object_pk(
                        connection, record.analysis_spec_object_id
                    ),
                    disposition=record.disposition,
                )
            )
            attempt_pk = _inserted_pk(result)
            for ordinal, reason_code in enumerate(record.reason_codes):
                connection.execute(
                    exploration_attempt_reason.insert().values(
                        attempt_pk=attempt_pk,
                        ordinal=ordinal,
                        reason_code=reason_code,
                    )
                )
            for materialization_id in record.artifact_materialization_ids:
                connection.execute(
                    exploration_attempt_artifact.insert().values(
                        attempt_pk=attempt_pk,
                        artifact_pk=self._pk(
                            connection,
                            artifact,
                            artifact.c.materialization_id,
                            materialization_id,
                        ),
                    )
                )

    @staticmethod
    def _object_query_predicates(query: CatalogObjectQuery) -> tuple[Any, ...]:
        predicates: list[Any] = []
        if query.object_id is not None:
            predicates.append(scientific_object.c.scientific_object_id == query.object_id)
        if query.kind is not None:
            predicates.append(scientific_object.c.kind == query.kind)
        if query.categorical_status is not None:
            predicates.append(scientific_object.c.categorical_status == query.categorical_status)

        metric_filters = tuple(
            predicate
            for value, predicate in (
                (query.relation_id, metric_definition.c.relation_id == query.relation_id),
                (
                    query.denominator_gauge_id,
                    metric_definition.c.denominator_gauge_id == query.denominator_gauge_id,
                ),
                (
                    query.response_gauge_id,
                    metric_definition.c.response_gauge_id == query.response_gauge_id,
                ),
                (query.horizon_id, metric_definition.c.horizon_id == query.horizon_id),
                (query.native_unit, metric_definition.c.native_unit == query.native_unit),
            )
            if value is not None
        )
        if metric_filters:
            predicates.append(
                exists(
                    select(1)
                    .select_from(
                        metric_observation.join(
                            metric_definition,
                            metric_observation.c.metric_definition_pk == metric_definition.c.pk,
                        )
                    )
                    .where(
                        metric_observation.c.scientific_object_pk == scientific_object.c.pk,
                        *metric_filters,
                    )
                ).correlate(scientific_object)
            )

        has_edge_filters = any(
            value is not None
            for value in (
                query.knowledge_edge_relation,
                query.knowledge_edge_scope_id,
                query.lineage_object_id,
            )
        )
        if has_edge_filters:
            edge_filters: list[Any] = []
            if query.knowledge_edge_relation is not None:
                edge_filters.append(knowledge_edge.c.relation == query.knowledge_edge_relation)
            if query.knowledge_edge_scope_id is not None:
                edge_filters.append(knowledge_edge.c.scope_id == query.knowledge_edge_scope_id)
            participation = or_(
                knowledge_edge.c.source_object_pk == scientific_object.c.pk,
                knowledge_edge.c.target_object_pk == scientific_object.c.pk,
                knowledge_edge.c.evidence_object_pk == scientific_object.c.pk,
            )
            if query.lineage_object_id is not None:
                lineage_object = alias(scientific_object, name="lineage_object")
                lineage_pk = (
                    select(lineage_object.c.pk)
                    .where(lineage_object.c.scientific_object_id == query.lineage_object_id)
                    .scalar_subquery()
                )
                edge_filters.extend(
                    (
                        or_(
                            knowledge_edge.c.source_object_pk == lineage_pk,
                            knowledge_edge.c.target_object_pk == lineage_pk,
                            knowledge_edge.c.evidence_object_pk == lineage_pk,
                        ),
                        scientific_object.c.scientific_object_id != query.lineage_object_id,
                    )
                )
            predicates.append(
                exists(
                    select(1)
                    .select_from(knowledge_edge)
                    .where(
                        participation,
                        *edge_filters,
                    )
                ).correlate(scientific_object)
            )
        return tuple(predicates)

    def query_objects(self, query: CatalogObjectQuery) -> CatalogObjectQueryPage:
        """Execute two row-capped SQL statements; never materialize a snapshot."""

        artifact_alias = alias(artifact, name="query_artifact")
        artifact_root = alias(storage_root, name="query_artifact_root")
        receipt_alias = alias(receipt, name="query_receipt")
        receipt_root = alias(storage_root, name="query_receipt_root")
        predicates = self._object_query_predicates(query)
        page_predicates = list(predicates)
        if query.after_object_id is not None:
            page_predicates.append(scientific_object.c.scientific_object_id > query.after_object_id)
        statement = (
            select(
                scientific_object.c.scientific_object_id,
                scientific_object.c.kind,
                scientific_object.c.object_schema,
                scientific_object.c.fingerprint,
                scientific_object.c.visibility_ceiling,
                scientific_object.c.categorical_status,
                artifact_alias.c.materialization_id.label("artifact_materialization_id"),
                artifact_alias.c.relative_path.label("artifact_relative_path"),
                artifact_root.c.storage_root_id.label("artifact_storage_root_id"),
                receipt_alias.c.receipt_id,
                receipt_alias.c.relative_path.label("receipt_relative_path"),
                receipt_root.c.storage_root_id.label("receipt_storage_root_id"),
            )
            .outerjoin(
                artifact_alias,
                scientific_object.c.artifact_pk == artifact_alias.c.pk,
            )
            .outerjoin(
                artifact_root,
                artifact_alias.c.storage_root_pk == artifact_root.c.pk,
            )
            .outerjoin(
                receipt_alias,
                scientific_object.c.receipt_pk == receipt_alias.c.pk,
            )
            .outerjoin(
                receipt_root,
                receipt_alias.c.storage_root_pk == receipt_root.c.pk,
            )
            .where(*page_predicates)
            .order_by(scientific_object.c.scientific_object_id)
            .limit(query.limit + 1)
        )
        count_probe = (
            select(scientific_object.c.pk)
            .where(*predicates)
            .limit(query.match_count_limit + 1)
            .subquery("bounded_catalog_match_probe")
        )
        count_statement = select(func.count()).select_from(count_probe)
        work_steps = 0
        work_exhausted = False

        def enforce_work_limit() -> int:
            nonlocal work_steps, work_exhausted
            work_steps += _SQLITE_PROGRESS_INTERVAL
            if work_steps > self.query_work_limit:
                work_exhausted = True
                return 1
            return 0

        with self.engine.connect() as connection:
            driver_connection = connection.connection.driver_connection
            set_progress_handler = getattr(driver_connection, "set_progress_handler", None)
            if set_progress_handler is None:
                raise RuntimeError("catalog query backend cannot enforce a SQLite work limit")
            set_progress_handler(enforce_work_limit, _SQLITE_PROGRESS_INTERVAL)
            try:
                probed_count = int(connection.execute(count_statement).scalar_one())
                raw_rows = tuple(map(_row_mapping, connection.execute(statement)))
            except OperationalError as error:
                if work_exhausted:
                    raise CatalogQueryWorkLimitExceeded(
                        work_limit=self.query_work_limit,
                        work_steps=work_steps,
                    ) from error
                raise
            finally:
                set_progress_handler(None, 0)
        matched_count_truncated = probed_count > query.match_count_limit
        matched_count = min(probed_count, query.match_count_limit)
        has_more = len(raw_rows) > query.limit
        page_rows = raw_rows[: query.limit]
        rows = []
        for row in page_rows:
            has_artifact = row["artifact_materialization_id"] is not None
            rows.append(
                CatalogObjectQueryRow(
                    object_id=row["scientific_object_id"],
                    kind=row["kind"],
                    object_schema=row["object_schema"],
                    object_fingerprint=_hex(row["fingerprint"]),
                    visibility_ceiling=row["visibility_ceiling"],
                    categorical_status=row["categorical_status"],
                    artifact_materialization_id=row["artifact_materialization_id"],
                    receipt_id=row["receipt_id"],
                    storage_root_id=(
                        row["artifact_storage_root_id"]
                        if has_artifact
                        else row["receipt_storage_root_id"]
                    ),
                    external_relative_path=(
                        row["artifact_relative_path"]
                        if has_artifact
                        else row["receipt_relative_path"]
                    ),
                    external_identity_kind=("ARTIFACT" if has_artifact else "RECEIPT"),
                )
            )
        return CatalogObjectQueryPage(
            matched_count=matched_count,
            match_count_limit=query.match_count_limit,
            matched_count_truncated=matched_count_truncated,
            rows=tuple(rows),
            has_more=has_more,
            next_after_object_id=(rows[-1].object_id if has_more and rows else None),
            query_work_limit=self.query_work_limit,
            query_work_steps=work_steps,
        )

    def snapshot(self) -> CatalogSnapshot:
        with self.engine.connect() as connection:
            return CatalogSnapshot(
                storage_roots=self._read_roots(connection),
                logical_artifacts=self._read_logical_artifacts(connection),
                artifact_locators=self._read_artifacts(connection),
                receipts=self._read_receipts(connection),
                scientific_objects=self._read_scientific_objects(connection),
                knowledge_edges=self._read_edges(connection),
                metric_definitions=self._read_metric_definitions(connection),
                metric_observations=self._read_metric_observations(connection),
                exploration_attempts=self._read_attempts(connection),
            )

    @staticmethod
    def _read_roots(connection: Connection) -> tuple[StorageRootRecord, ...]:
        rows = connection.execute(select(storage_root).order_by(storage_root.c.storage_root_id))
        return tuple(
            StorageRootRecord(
                storage_root_id=row["storage_root_id"],
                logical_name=row["logical_name"],
                canonical_path=row["canonical_path"],
                mount_contract_schema=row["mount_contract_schema"],
                verification_status=CatalogVerificationStatus(row["verification_status"]),
            )
            for row in map(_row_mapping, rows)
        )

    @staticmethod
    def _read_logical_artifacts(connection: Connection) -> tuple[LogicalArtifactRecord, ...]:
        rows = connection.execute(
            select(logical_artifact).order_by(logical_artifact.c.logical_artifact_id)
        )
        return tuple(
            LogicalArtifactRecord(
                logical_artifact_id=row["logical_artifact_id"],
                content_sha256=_hex(row["content_sha256"]),
                payload_schema=row["payload_schema"],
                profile=ArtifactProfile(row["profile"]),
                media_type=row["media_type"],
                visibility_ceiling=VisibilityCeiling(row["visibility_ceiling"]),
            )
            for row in map(_row_mapping, rows)
        )

    @staticmethod
    def _read_artifacts(connection: Connection) -> tuple[ArtifactLocatorRecord, ...]:
        statement = (
            select(
                artifact,
                logical_artifact.c.logical_artifact_id,
                storage_root.c.storage_root_id,
            )
            .join(logical_artifact, artifact.c.logical_artifact_pk == logical_artifact.c.pk)
            .join(storage_root, artifact.c.storage_root_pk == storage_root.c.pk)
            .order_by(artifact.c.materialization_id)
        )
        return tuple(
            ArtifactLocatorRecord(
                materialization_id=row["materialization_id"],
                logical_artifact_id=row["logical_artifact_id"],
                storage_root_id=row["storage_root_id"],
                relative_path=row["relative_path"],
                physical_sha256=_hex(row["physical_sha256"]),
                size_bytes=int(row["size_bytes"]),
                compression=row["compression"],
                partition_selector=row["partition_selector"],
                verification_status=CatalogVerificationStatus(row["verification_status"]),
            )
            for row in map(_row_mapping, connection.execute(statement))
        )

    @staticmethod
    def _read_receipts(connection: Connection) -> tuple[ReceiptLocatorRecord, ...]:
        statement = (
            select(receipt, storage_root.c.storage_root_id)
            .join(storage_root, receipt.c.storage_root_pk == storage_root.c.pk)
            .order_by(receipt.c.receipt_id)
        )
        return tuple(
            ReceiptLocatorRecord(
                receipt_id=row["receipt_id"],
                run_id=row["run_id"],
                storage_root_id=row["storage_root_id"],
                relative_path=row["relative_path"],
                sha256=_hex(row["sha256"]),
                receipt_schema=row["receipt_schema"],
                verification_status=CatalogVerificationStatus(row["verification_status"]),
            )
            for row in map(_row_mapping, connection.execute(statement))
        )

    @staticmethod
    def _read_scientific_objects(
        connection: Connection,
    ) -> tuple[ScientificObjectRecord, ...]:
        artifact_alias = alias(artifact, name="object_artifact")
        receipt_alias = alias(receipt, name="object_receipt")
        statement = (
            select(
                scientific_object,
                artifact_alias.c.materialization_id,
                receipt_alias.c.receipt_id,
            )
            .outerjoin(artifact_alias, scientific_object.c.artifact_pk == artifact_alias.c.pk)
            .outerjoin(receipt_alias, scientific_object.c.receipt_pk == receipt_alias.c.pk)
            .order_by(scientific_object.c.scientific_object_id)
        )
        return tuple(
            ScientificObjectRecord(
                scientific_object_id=row["scientific_object_id"],
                kind=ScientificObjectKind(row["kind"]),
                object_schema=row["object_schema"],
                object_sha256=_hex(row["fingerprint"]),
                visibility_ceiling=VisibilityCeiling(row["visibility_ceiling"]),
                categorical_status=row["categorical_status"],
                artifact_materialization_id=row["materialization_id"],
                receipt_id=row["receipt_id"],
            )
            for row in map(_row_mapping, connection.execute(statement))
        )

    @staticmethod
    def _read_edges(connection: Connection) -> tuple[KnowledgeEdgeRecord, ...]:
        source = alias(scientific_object, name="source_object")
        target = alias(scientific_object, name="target_object")
        evidence = alias(scientific_object, name="evidence_object")
        statement = (
            select(
                knowledge_edge,
                source.c.scientific_object_id.label("source_id"),
                target.c.scientific_object_id.label("target_id"),
                evidence.c.scientific_object_id.label("evidence_id"),
            )
            .join(source, knowledge_edge.c.source_object_pk == source.c.pk)
            .join(target, knowledge_edge.c.target_object_pk == target.c.pk)
            .outerjoin(evidence, knowledge_edge.c.evidence_object_pk == evidence.c.pk)
            .order_by(knowledge_edge.c.edge_id)
        )
        return tuple(
            KnowledgeEdgeRecord(
                edge_id=row["edge_id"],
                relation=row["relation"],
                source_object_id=row["source_id"],
                target_object_id=row["target_id"],
                evidence_object_id=row["evidence_id"],
                scope_id=row["scope_id"],
            )
            for row in map(_row_mapping, connection.execute(statement))
        )

    @staticmethod
    def _read_metric_definitions(
        connection: Connection,
    ) -> tuple[MetricDefinitionRecord, ...]:
        rows = connection.execute(
            select(metric_definition).order_by(metric_definition.c.metric_definition_id)
        )
        return tuple(
            MetricDefinitionRecord(
                metric_definition_id=row["metric_definition_id"],
                metric_name=row["metric_name"],
                dataset_version_id=row["dataset_version_id"],
                relation_id=row["relation_id"],
                denominator_gauge_id=row["denominator_gauge_id"],
                response_gauge_id=row["response_gauge_id"],
                horizon_id=row["horizon_id"],
                native_unit=row["native_unit"],
                aggregation=row["aggregation"],
                direction=row["direction"],
            )
            for row in map(_row_mapping, rows)
        )

    @staticmethod
    def _read_metric_observations(
        connection: Connection,
    ) -> tuple[MetricObservationRecord, ...]:
        definition = alias(metric_definition, name="observation_definition")
        object_alias = alias(scientific_object, name="observation_object")
        statement = (
            select(
                metric_observation,
                definition.c.metric_definition_id,
                object_alias.c.scientific_object_id,
            )
            .join(definition, metric_observation.c.metric_definition_pk == definition.c.pk)
            .join(object_alias, metric_observation.c.scientific_object_pk == object_alias.c.pk)
            .order_by(metric_observation.c.metric_observation_id)
        )
        return tuple(
            MetricObservationRecord(
                metric_observation_id=row["metric_observation_id"],
                metric_definition_id=row["metric_definition_id"],
                scientific_object_id=row["scientific_object_id"],
                point=Decimal(row["point_decimal"]),
                lower=None if row["lower_decimal"] is None else Decimal(row["lower_decimal"]),
                upper=None if row["upper_decimal"] is None else Decimal(row["upper_decimal"]),
                physical_independent_unit_count=int(row["physical_independent_unit_count"]),
                numerical_view_count=int(row["numerical_view_count"]),
            )
            for row in map(_row_mapping, connection.execute(statement))
        )

    @staticmethod
    def _read_attempts(connection: Connection) -> tuple[ExplorationAttemptRecord, ...]:
        analysis = alias(scientific_object, name="attempt_analysis")
        statement = (
            select(
                exploration_attempt,
                analysis.c.scientific_object_id.label("analysis_id"),
            )
            .join(analysis, exploration_attempt.c.analysis_spec_object_pk == analysis.c.pk)
            .order_by(exploration_attempt.c.attempt_id)
        )
        records: list[ExplorationAttemptRecord] = []
        for row in map(_row_mapping, connection.execute(statement)):
            attempt_pk = int(row["pk"])
            reasons = tuple(
                connection.execute(
                    select(exploration_attempt_reason.c.reason_code)
                    .where(exploration_attempt_reason.c.attempt_pk == attempt_pk)
                    .order_by(exploration_attempt_reason.c.ordinal)
                ).scalars()
            )
            artifacts = tuple(
                connection.execute(
                    select(artifact.c.materialization_id)
                    .join(
                        exploration_attempt_artifact,
                        exploration_attempt_artifact.c.artifact_pk == artifact.c.pk,
                    )
                    .where(exploration_attempt_artifact.c.attempt_pk == attempt_pk)
                    .order_by(artifact.c.materialization_id)
                ).scalars()
            )
            records.append(
                ExplorationAttemptRecord(
                    attempt_id=row["attempt_id"],
                    analysis_spec_object_id=row["analysis_id"],
                    disposition=row["disposition"],
                    reason_codes=tuple(sorted(reasons)),
                    artifact_materialization_ids=artifacts,
                )
            )
        return tuple(records)

    def integrity_check(self) -> tuple[str, ...]:
        reasons: list[str] = []
        audit = schema_audit(self.engine)
        if not audit.passed:
            reasons.append("schema-audit-failed")
        with self.engine.connect() as connection:
            for kind, projection in projection_tables.items():
                expected = connection.execute(
                    select(scientific_object.c.pk).where(scientific_object.c.kind == kind.upper())
                ).scalars()
                projected = set(
                    connection.execute(select(projection.c.scientific_object_pk)).scalars()
                )
                if not set(expected).issubset(projected):
                    reasons.append("projection-row-missing")
                    break
        return tuple(sorted(set(reasons)))

    def scientific_object(self, object_id: str) -> ScientificObjectRecord | None:
        page = self.query_objects(CatalogObjectQuery(object_id=object_id, limit=1))
        if not page.rows:
            return None
        row = page.rows[0]
        return ScientificObjectRecord(
            scientific_object_id=row.object_id,
            kind=ScientificObjectKind(row.kind),
            object_schema=row.object_schema,
            object_sha256=row.object_fingerprint,
            visibility_ceiling=VisibilityCeiling(row.visibility_ceiling),
            categorical_status=row.categorical_status,
            artifact_materialization_id=row.artifact_materialization_id,
            receipt_id=row.receipt_id,
        )

"""Atomic reconstruction of the ignored local SQLite catalog projection."""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final, Iterator

from sqlalchemy.exc import SQLAlchemyError

from empirical_lawhood.infrastructure.artifacts import (
    ArtifactPlaneError,
    ExternalArtifactPlane,
)
from empirical_lawhood.infrastructure.bounded_io import (
    BoundedFileIOError,
    MAX_ARTIFACT_MANIFEST_BYTES,
    MAX_CATALOG_DATABASE_BYTES,
    bounded_file_contains_any,
    bounded_file_sha256,
    read_bounded_bytes,
)
from empirical_lawhood.infrastructure.catalog_projection import decode_catalog_snapshot
from empirical_lawhood.infrastructure.catalog_mutation import (
    DEFAULT_CATALOG_MUTATION_LOCK_TIMEOUT_SECONDS,
    CatalogMutationLockError,
    production_catalog_mutation_lock,
)
from empirical_lawhood.infrastructure.dataset_projection import decode_unified_catalog_snapshot
from empirical_lawhood.infrastructure.task_receipts import decode_artifact_manifest
from empirical_lawhood.kernel.serialization import CanonicalizationError, validate_schema
from empirical_lawhood.planning.datasets import (
    DatasetMaterialization,
    DatasetMaterializationVerificationReceipt,
    EvidenceReference,
    EvidenceReferenceKind,
)
from empirical_lawhood.planning.dataset_rebuild import LocalCatalogTargetState
from empirical_lawhood.runtime.catalog import CatalogSnapshot, CatalogVerificationStatus
from empirical_lawhood.runtime.datasets import (
    DatasetCatalogSnapshot,
    DatasetCatalogProjectionState,
    DatasetCatalogWorkLimit,
    DatasetEvidenceOwnerKind,
    DatasetEvidenceVerification,
    DatasetEvidenceVerificationRequest,
    DatasetEvidenceVerifier,
    DatasetEvidenceVerifierRegistration,
    DatasetImplementationRegistry,
    UnifiedCatalogSnapshot,
)
from empirical_lawhood.infrastructure.sql import (
    SQLiteCatalogRepository,
    SQLiteDatasetRepository,
    create_catalog_engine,
    create_read_only_catalog_engine,
    production_catalog_path,
    schema_audit,
    upgrade_catalog,
)
from empirical_lawhood.infrastructure.sql.database import (
    ALEMBIC_HEAD,
    APPLICATION_ID,
    SCHEMA_VERSION,
)
from empirical_lawhood.infrastructure.sql.migrations.catalog_tables import CATALOG_METADATA, CATALOG_TABLE_NAMES
from empirical_lawhood.infrastructure.sql.migrations.dataset_tables import DATASET_PROJECTION_DIRTY_TRIGGER_NAMES, DATASET_EXPLICIT_INDEX_NAMES, DATASET_TABLE_NAMES, DATASET_VIEW_NAMES


MAX_CATALOG_REBUILD_VERIFIED_ARTIFACTS: Final[int] = 100_000
MAX_CATALOG_REBUILD_VERIFIED_ARTIFACT_BYTES: Final[int] = 1024**4
MAX_CATALOG_REBUILD_VERIFIED_RECEIPTS: Final[int] = 100_000
MAX_CATALOG_REBUILD_RECEIPT_BYTES: Final[int] = 4 * 1024**3
MAX_CATALOG_REBUILD_RECEIPT_MEMBER_BYTES: Final[int] = 64 * 1024**2
MAX_DATASET_REBUILD_EVIDENCE_MEMBERS: Final[int] = 100_000
MAX_DATASET_REBUILD_EVIDENCE_MEMBER_BYTES: Final[int] = 64 * 1024**2
MAX_DATASET_REBUILD_EVIDENCE_AGGREGATE_BYTES: Final[int] = 4 * 1024**3
_CURRENT_SCHEMA_IDENTITY: Final[tuple[int, int, str]] = (
    APPLICATION_ID,
    SCHEMA_VERSION,
    ALEMBIC_HEAD,
)
_CATALOG_SCHEMA_OBJECT_INVENTORY: Final[tuple[tuple[str, str], ...]] = tuple(
    sorted(
        (
            *(("table", table_name) for table_name in CATALOG_TABLE_NAMES),
            ("table", "alembic_version"),
            *(
                ("index", str(index.name))
                for table in CATALOG_METADATA.tables.values()
                for index in table.indexes
                if index.name is not None
            ),
        )
    )
)
_CURRENT_SCHEMA_OBJECT_INVENTORY: Final[tuple[tuple[str, str], ...]] = tuple(
    sorted(
        (
            *_CATALOG_SCHEMA_OBJECT_INVENTORY,
            *(("table", name) for name in DATASET_TABLE_NAMES),
            *(("view", name) for name in DATASET_VIEW_NAMES),
            *(("index", name) for name in DATASET_EXPLICIT_INDEX_NAMES),
            *(("trigger", name) for name in DATASET_PROJECTION_DIRTY_TRIGGER_NAMES),
        )
    )
)
_SQLITE_STAGE_SIDECAR_SUFFIXES: Final[tuple[str, ...]] = (
    "-journal",
    "-wal",
    "-shm",
)


class CatalogRebuildError(RuntimeError):
    """A projection could not safely reconstruct the local catalog."""


@dataclass(frozen=True, slots=True)
class CatalogRebuildResult:
    database_relative_path: str
    database_sha256: str
    database_size_bytes: int
    projection_sha256: str
    snapshot_sha256: str
    integrity_reasons: tuple[str, ...]
    schema_audit: Mapping[str, object]
    dataset_evidence_verifications: tuple[DatasetEvidenceVerification, ...] = ()
    dataset_projection_state: DatasetCatalogProjectionState | None = None


@dataclass(frozen=True, slots=True)
class CatalogRebuildPrepared:
    """Verified stage identity available before the local visibility commit."""

    database_sha256: str
    database_size_bytes: int
    projection_sha256: str
    snapshot_sha256: str
    dataset_projection_state: DatasetCatalogProjectionState | None


@dataclass(frozen=True, slots=True)
class _CatalogFileIdentity:
    device: int
    inode: int
    size_bytes: int
    modified_ns: int
    changed_ns: int
    sha256: str


def _catalog_rebuild_stage_path(
    database_path: Path,
    projection_fingerprint: str,
) -> Path:
    return database_path.with_name(f".{database_path.name}.rebuild.{projection_fingerprint[:16]}")


def _remove_catalog_rebuild_stage(stage_path: Path) -> None:
    cleanup_error: OSError | None = None
    for candidate in (
        stage_path,
        *(Path(f"{stage_path}{suffix}") for suffix in _SQLITE_STAGE_SIDECAR_SUFFIXES),
    ):
        try:
            candidate.unlink(missing_ok=True)
        except OSError as error:
            if cleanup_error is None:
                cleanup_error = error
    if cleanup_error is not None:
        raise CatalogRebuildError(
            "catalog rebuild stage could not be cleaned safely"
        ) from cleanup_error


@contextmanager
def _catalog_rebuild_stage(
    database_path: Path,
    projection_fingerprint: str,
) -> Iterator[Path]:
    """Own one exact stage and remove its database and sidecars on every exit."""

    stage_path = _catalog_rebuild_stage_path(database_path, projection_fingerprint)
    descriptor: int | None = None
    created = False
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if nofollow is None:
        raise CatalogRebuildError("platform cannot enforce a non-symlink rebuild stage")
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | nofollow
    try:
        try:
            descriptor = os.open(stage_path, flags, 0o600)
            created = True
        except FileExistsError as error:
            raise CatalogRebuildError("catalog rebuild stage already exists") from error
        except OSError as error:
            raise CatalogRebuildError(
                "catalog rebuild stage could not be created safely"
            ) from error
        identity = os.fstat(descriptor)
        if (
            not stat.S_ISREG(identity.st_mode)
            or identity.st_nlink != 1
            or identity.st_uid != os.getuid()
            or stat.S_IMODE(identity.st_mode) & 0o077
        ):
            raise CatalogRebuildError("catalog rebuild stage has an unsafe identity or mode")
        os.close(descriptor)
        descriptor = None
        yield stage_path
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            _remove_catalog_rebuild_stage(stage_path)


def rebuild_production_catalog(
    repo_root: Path,
    projection: bytes,
    *,
    forbidden_payloads: tuple[bytes, ...] = (),
    artifact_plane: ExternalArtifactPlane | None = None,
    implementation_registry: DatasetImplementationRegistry | None = None,
    evidence_verifier_registry_id: str | tuple[str, ...] | None = None,
    expected_prior_database: LocalCatalogTargetState | None = None,
    before_install: Callable[[CatalogRebuildPrepared], None] | None = None,
    mutation_lock_timeout_seconds: float = DEFAULT_CATALOG_MUTATION_LOCK_TIMEOUT_SECONDS,
) -> CatalogRebuildResult:
    """Build from an external projection, verify, then atomically install locally."""

    unified_snapshot: UnifiedCatalogSnapshot | None = None
    try:
        unified_snapshot = decode_unified_catalog_snapshot(projection)
        snapshot = unified_snapshot.scientific
    except (CanonicalizationError, ValueError):
        try:
            snapshot = decode_catalog_snapshot(projection)
        except (CanonicalizationError, ValueError) as error:
            raise CatalogRebuildError("external catalog projection is invalid") from error
    logical_by_id = {record.logical_artifact_id: record for record in snapshot.logical_artifacts}
    if any(
        locator.compression == "none"
        and locator.partition_selector is None
        and logical_by_id[locator.logical_artifact_id].content_sha256 != locator.physical_sha256
        for locator in snapshot.artifact_locators
    ):
        raise CatalogRebuildError(
            "direct catalog artifact lacks verified logical/physical identity"
        )
    _verify_projected_artifacts(snapshot, artifact_plane)
    _verify_projected_receipts(snapshot, artifact_plane)
    if unified_snapshot is None:
        _resolve_dataset_evidence_verifiers(
            implementation_registry,
            evidence_verifier_registry_id,
            required=False,
        )
        dataset_evidence_verifications: tuple[DatasetEvidenceVerification, ...] = ()
    else:
        dataset_evidence_verifications = _verify_dataset_evidence(
            unified_snapshot.datasets,
            implementation_registry,
            evidence_verifier_registry_id,
        )
    try:
        with production_catalog_mutation_lock(
            repo_root,
            timeout_seconds=mutation_lock_timeout_seconds,
        ):
            return _rebuild_production_catalog_locked(
                repo_root=repo_root,
                database_path=production_catalog_path(repo_root),
                projection=projection,
                snapshot=snapshot,
                unified_snapshot=unified_snapshot,
                forbidden_payloads=forbidden_payloads,
                dataset_evidence_verifications=dataset_evidence_verifications,
                expected_prior_database=expected_prior_database,
                before_install=before_install,
            )
    except CatalogMutationLockError as error:
        raise CatalogRebuildError(
            "production catalog mutation lock could not be acquired safely"
        ) from error


def _rebuild_production_catalog_locked(
    *,
    repo_root: Path,
    database_path: Path,
    projection: bytes,
    snapshot: CatalogSnapshot,
    unified_snapshot: UnifiedCatalogSnapshot | None,
    forbidden_payloads: tuple[bytes, ...],
    dataset_evidence_verifications: tuple[DatasetEvidenceVerification, ...],
    expected_prior_database: LocalCatalogTargetState | None,
    before_install: Callable[[CatalogRebuildPrepared], None] | None,
) -> CatalogRebuildResult:
    """Stage, promote, and verify while the production mutation lock is held."""

    if database_path.is_symlink():
        raise CatalogRebuildError("repository-local catalog cannot be symlinked")
    installed_identity = _catalog_file_identity(database_path)
    if expected_prior_database is not None:
        observed_prior = (
            LocalCatalogTargetState(
                target_id=expected_prior_database.target_id,
                relative_path=expected_prior_database.relative_path,
                exists=False,
                size_bytes=None,
                sha256=None,
            )
            if installed_identity is None
            else LocalCatalogTargetState(
                target_id=expected_prior_database.target_id,
                relative_path=expected_prior_database.relative_path,
                exists=True,
                size_bytes=installed_identity.size_bytes,
                sha256=installed_identity.sha256,
            )
        )
        if observed_prior != expected_prior_database:
            raise CatalogRebuildError(
                "installed catalog differs from the authorized prior target state"
            )
    if (
        unified_snapshot is None
        and database_path.exists()
        and _installed_dataset_projection_nonempty(database_path)
    ):
        raise CatalogRebuildError(
            "scientific-only rebuild refuses to erase the installed dataset projection"
        )
    projection_snapshot: CatalogSnapshot | UnifiedCatalogSnapshot = (
        snapshot if unified_snapshot is None else unified_snapshot
    )
    with _catalog_rebuild_stage(
        database_path,
        projection_snapshot.fingerprint(),
    ) as stage_path:
        stage_engine = create_catalog_engine(f"sqlite+pysqlite:///{stage_path}")
        try:
            upgrade_catalog(stage_engine)
            repository = SQLiteCatalogRepository(stage_engine)
            repository.append_snapshot(snapshot)
            if unified_snapshot is None:
                observed: CatalogSnapshot | UnifiedCatalogSnapshot = repository.snapshot()
                integrity_reasons = repository.integrity_check()
            else:
                dataset_repository = SQLiteDatasetRepository(stage_engine)
                dataset_repository.append_snapshot(unified_snapshot.datasets)
                observed = UnifiedCatalogSnapshot(
                    repository.snapshot(),
                    dataset_repository.snapshot(),
                )
                integrity_reasons = tuple(
                    sorted(
                        set(repository.integrity_check()).union(
                            dataset_repository.integrity_check()
                        )
                    )
                )
            audit = schema_audit(stage_engine)
        finally:
            stage_engine.dispose()
        if observed.canonical_bytes() != projection:
            raise CatalogRebuildError("SQLite round trip differs from the external projection")
        if integrity_reasons or not audit.passed:
            raise CatalogRebuildError("staged SQLite catalog failed integrity or schema policy")
        try:
            if bounded_file_contains_any(
                stage_path,
                forbidden_payloads,
                maximum_bytes=MAX_CATALOG_DATABASE_BYTES,
            ):
                raise CatalogRebuildError("scientific or receipt payload escaped into SQLite")
        except BoundedFileIOError as error:
            raise CatalogRebuildError("staged SQLite catalog exceeds its work bound") from error

        prepared_projection_state: DatasetCatalogProjectionState | None = None
        if unified_snapshot is not None:
            stage_read_only_engine = create_read_only_catalog_engine(stage_path)
            try:
                prepared_projection_state = SQLiteDatasetRepository(
                    stage_read_only_engine
                ).projection_state(
                    DatasetCatalogWorkLimit(
                        max_records_examined=1,
                        max_query_steps=100_000,
                    )
                )
            finally:
                stage_read_only_engine.dispose()
        try:
            prepared_size_bytes, prepared_database_sha256 = bounded_file_sha256(
                stage_path,
                maximum_bytes=MAX_CATALOG_DATABASE_BYTES,
            )
        except BoundedFileIOError as error:
            raise CatalogRebuildError("staged SQLite catalog exceeds its work bound") from error
        prepared = CatalogRebuildPrepared(
            database_sha256=prepared_database_sha256,
            database_size_bytes=prepared_size_bytes,
            projection_sha256=hashlib.sha256(projection).hexdigest(),
            snapshot_sha256=projection_snapshot.fingerprint(),
            dataset_projection_state=prepared_projection_state,
        )
        if _catalog_file_identity(database_path) != installed_identity:
            raise CatalogRebuildError("installed catalog changed during rebuild")
        if before_install is not None:
            before_install(prepared)
        if _catalog_file_identity(database_path) != installed_identity:
            raise CatalogRebuildError("installed catalog changed before the visibility commit")
        os.replace(stage_path, database_path)
        directory_descriptor = os.open(database_path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

        dataset_projection_state: DatasetCatalogProjectionState | None = None
        installed_engine = create_read_only_catalog_engine(database_path)
        try:
            installed_repository = SQLiteCatalogRepository(installed_engine)
            if unified_snapshot is None:
                installed: CatalogSnapshot | UnifiedCatalogSnapshot = (
                    installed_repository.snapshot()
                )
                installed_reasons = installed_repository.integrity_check()
            else:
                installed_dataset_repository = SQLiteDatasetRepository(installed_engine)
                installed = UnifiedCatalogSnapshot(
                    installed_repository.snapshot(),
                    installed_dataset_repository.snapshot(),
                )
                installed_reasons = tuple(
                    sorted(
                        set(installed_repository.integrity_check()).union(
                            installed_dataset_repository.integrity_check()
                        )
                    )
                )
            installed_audit = schema_audit(installed_engine)
        finally:
            installed_engine.dispose()
        if unified_snapshot is not None:
            read_only_engine = create_read_only_catalog_engine(database_path)
            try:
                dataset_projection_state = SQLiteDatasetRepository(
                    read_only_engine
                ).projection_state(
                    DatasetCatalogWorkLimit(
                        max_records_examined=1,
                        max_query_steps=100_000,
                    )
                )
            finally:
                read_only_engine.dispose()
        if installed.canonical_bytes() != projection:
            raise CatalogRebuildError("installed SQLite catalog differs from projection")
        if installed_reasons or not installed_audit.passed:
            raise CatalogRebuildError("installed SQLite catalog failed integrity or schema policy")

        try:
            database_size_bytes, database_sha256 = bounded_file_sha256(
                database_path,
                maximum_bytes=MAX_CATALOG_DATABASE_BYTES,
            )
        except BoundedFileIOError as error:
            raise CatalogRebuildError("installed SQLite catalog exceeds its work bound") from error
        if (
            database_size_bytes != prepared.database_size_bytes
            or database_sha256 != prepared.database_sha256
            or installed.fingerprint() != prepared.snapshot_sha256
            or dataset_projection_state != prepared.dataset_projection_state
        ):
            raise CatalogRebuildError("installed catalog differs from its verified stage")
        return CatalogRebuildResult(
            database_relative_path=str(database_path.relative_to(repo_root.resolve(strict=True))),
            database_sha256=database_sha256,
            database_size_bytes=database_size_bytes,
            projection_sha256=prepared.projection_sha256,
            snapshot_sha256=installed.fingerprint(),
            integrity_reasons=installed_reasons,
            schema_audit=asdict(installed_audit),
            dataset_evidence_verifications=dataset_evidence_verifications,
            dataset_projection_state=dataset_projection_state,
        )


def _installed_dataset_projection_nonempty(database_path: Path) -> bool:
    engine = create_read_only_catalog_engine(database_path)
    try:
        try:
            with engine.connect() as connection:
                application_id = int(
                    str(connection.exec_driver_sql("PRAGMA application_id").scalar_one())
                )
                user_version = int(
                    str(connection.exec_driver_sql("PRAGMA user_version").scalar_one())
                )
                revisions = tuple(
                    str(row[0])
                    for row in connection.exec_driver_sql(
                        "SELECT version_num FROM alembic_version ORDER BY version_num LIMIT 2"
                    )
                )
                schema_inventory = tuple(
                    (str(row[0]), str(row[1]))
                    for row in connection.exec_driver_sql(
                        "SELECT type, name FROM sqlite_schema "
                        "WHERE type IN ('table', 'view', 'index', 'trigger') "
                        "AND name NOT LIKE 'sqlite\\_%' ESCAPE '\\' "
                        "ORDER BY type, name"
                    )
                )
            if len(revisions) != 1:
                raise CatalogRebuildError("installed catalog has an ambiguous Alembic revision")
            schema_identity = (application_id, user_version, revisions[0])
            if schema_identity != _CURRENT_SCHEMA_IDENTITY:
                raise CatalogRebuildError(
                    "installed catalog schema identity is unsupported or inconsistent"
                )
            if schema_inventory != _CURRENT_SCHEMA_OBJECT_INVENTORY:
                raise CatalogRebuildError(
                    "installed catalog schema inventory is partial or corrupt"
                )
            return SQLiteDatasetRepository(engine).has_records()
        except CatalogRebuildError:
            raise
        except (SQLAlchemyError, TypeError, ValueError) as error:
            raise CatalogRebuildError(
                "installed dataset projection could not be inspected safely"
            ) from error
    finally:
        engine.dispose()


def _catalog_file_identity(database_path: Path) -> _CatalogFileIdentity | None:
    if not database_path.exists():
        return None
    if database_path.is_symlink():
        raise CatalogRebuildError("repository-local catalog cannot be symlinked")
    try:
        before = database_path.stat()
        size_bytes, sha256 = bounded_file_sha256(
            database_path,
            maximum_bytes=MAX_CATALOG_DATABASE_BYTES,
        )
        after = database_path.stat()
    except (BoundedFileIOError, FileNotFoundError, OSError) as error:
        raise CatalogRebuildError(
            "installed catalog identity could not be captured safely"
        ) from error
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != after_identity or size_bytes != after.st_size:
        raise CatalogRebuildError("installed catalog changed during identity capture")
    return _CatalogFileIdentity(
        device=after.st_dev,
        inode=after.st_ino,
        size_bytes=size_bytes,
        modified_ns=after.st_mtime_ns,
        changed_ns=after.st_ctime_ns,
        sha256=sha256,
    )


def _dataset_evidence_requests(
    snapshot: DatasetCatalogSnapshot,
) -> tuple[DatasetEvidenceVerificationRequest, ...]:
    by_id: dict[str, EvidenceReference] = {}
    requests: list[DatasetEvidenceVerificationRequest] = []

    def include(
        *,
        owner_kind: DatasetEvidenceOwnerKind,
        owner_id: str,
        references: tuple[EvidenceReference, ...],
        materialization: object | None = None,
    ) -> None:
        for reference in references:
            existing = by_id.get(reference.evidence_id)
            if existing is not None and existing != reference:
                raise CatalogRebuildError(
                    "dataset evidence ID resolves to conflicting external references"
                )
            by_id[reference.evidence_id] = reference
            if owner_kind is DatasetEvidenceOwnerKind.MATERIALIZATION:
                if not isinstance(materialization, DatasetMaterialization):
                    raise CatalogRebuildError("materialization evidence lacks exact owner context")
                manifest_references = tuple(
                    value
                    for value in materialization.verification_evidence_refs
                    if value.kind is EvidenceReferenceKind.MANIFEST
                )
                if len(manifest_references) > 1:
                    raise CatalogRebuildError(
                        "materialization evidence has ambiguous manifest context"
                    )
                request = DatasetEvidenceVerificationRequest(
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                    reference=reference,
                    materialization_subject=materialization.verification_subject(),
                    expected_verifier=materialization.verifier,
                    expected_verification_policy_id=(materialization.verification_policy_id),
                    expected_verification_policy_sha256=(
                        materialization.verification_policy_sha256
                    ),
                    expected_verified_at_utc=materialization.verified_at_utc,
                    expected_manifest_evidence=(
                        manifest_references[0] if manifest_references else None
                    ),
                )
            else:
                request = DatasetEvidenceVerificationRequest(
                    owner_kind=owner_kind,
                    owner_id=owner_id,
                    reference=reference,
                    materialization_subject=None,
                    expected_verifier=None,
                    expected_verification_policy_id=None,
                    expected_verification_policy_sha256=None,
                    expected_verified_at_utc=None,
                    expected_manifest_evidence=None,
                )
            requests.append(request)

    for release in snapshot.releases:
        include(
            owner_kind=DatasetEvidenceOwnerKind.RELEASE,
            owner_id=release.release_id,
            references=release.evidence_refs,
        )
    for observation in snapshot.observations:
        include(
            owner_kind=DatasetEvidenceOwnerKind.OBSERVATION,
            owner_id=observation.observation_id,
            references=observation.evidence_refs,
        )
    for materialization in snapshot.materializations:
        include(
            owner_kind=DatasetEvidenceOwnerKind.MATERIALIZATION,
            owner_id=materialization.materialization_id,
            references=materialization.verification_evidence_refs,
            materialization=materialization,
        )
    for attempt in snapshot.acquisition_attempts:
        include(
            owner_kind=DatasetEvidenceOwnerKind.ACQUISITION_ATTEMPT,
            owner_id=attempt.attempt_id,
            references=attempt.evidence_refs,
        )
    for binding in snapshot.bindings:
        include(
            owner_kind=DatasetEvidenceOwnerKind.BINDING,
            owner_id=binding.binding_id,
            references=binding.evidence_refs,
        )

    receipt_subjects: dict[str, str] = {}
    by_request_fingerprint: dict[str, DatasetEvidenceVerificationRequest] = {}
    for request in requests:
        by_request_fingerprint[request.fingerprint()] = request
        if request.reference.evidence_schema != DatasetMaterializationVerificationReceipt.SCHEMA:
            continue
        subject = request.materialization_subject
        if subject is None:
            raise CatalogRebuildError("typed materialization receipt lacks an exact subject")
        subject_fingerprint = subject.fingerprint()
        existing_subject = receipt_subjects.get(request.reference.evidence_sha256)
        if existing_subject is not None and existing_subject != subject_fingerprint:
            raise CatalogRebuildError(
                "materialization verification receipt is reused across subjects"
            )
        receipt_subjects[request.reference.evidence_sha256] = subject_fingerprint

    bounded_requests = tuple(
        sorted(
            by_request_fingerprint.values(),
            key=lambda value: (
                value.owner_kind.value,
                value.owner_id,
                value.reference.evidence_id,
            ),
        )
    )
    if len(bounded_requests) > MAX_DATASET_REBUILD_EVIDENCE_MEMBERS:
        raise CatalogRebuildError("dataset evidence exceeds its aggregate member limit")
    return bounded_requests


def _evidence_verifier_registry_ids(
    value: str | tuple[str, ...] | None,
) -> tuple[str, ...]:
    if value is None:
        return ()
    values: tuple[str, ...]
    if isinstance(value, str):
        values = (value,)
    elif isinstance(value, tuple):
        values = value
    else:
        raise CatalogRebuildError("dataset evidence-verifier registry IDs have the wrong type")
    if (
        not values
        or any(not isinstance(registry_id, str) or not registry_id for registry_id in values)
        or tuple(sorted(set(values))) != values
    ):
        raise CatalogRebuildError(
            "dataset evidence-verifier registry IDs must be sorted and unique"
        )
    return values


def _resolve_dataset_evidence_verifiers(
    implementation_registry: DatasetImplementationRegistry | None,
    evidence_verifier_registry_id: str | tuple[str, ...] | None,
    *,
    required: bool,
) -> tuple[tuple[DatasetEvidenceVerifierRegistration, DatasetEvidenceVerifier], ...]:
    registry_ids = _evidence_verifier_registry_ids(evidence_verifier_registry_id)
    if (implementation_registry is None) != (not registry_ids):
        raise CatalogRebuildError(
            "dataset evidence verification requires both an implementation registry "
            "and evidence-verifier registry IDs"
        )
    if implementation_registry is None:
        if required:
            raise CatalogRebuildError(
                "unified dataset evidence requires a registered read-only verifier"
            )
        return ()
    if not isinstance(implementation_registry, DatasetImplementationRegistry):
        raise CatalogRebuildError("dataset evidence implementation registry has the wrong type")
    try:
        resolved = tuple(
            (
                implementation_registry.static_registry.evidence_verifier(registry_id),
                implementation_registry.evidence_verifier(registry_id),
            )
            for registry_id in registry_ids
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CatalogRebuildError(
            "dataset evidence verifier registration or implementation is invalid"
        ) from error
    return resolved


def _verify_dataset_evidence(
    snapshot: DatasetCatalogSnapshot,
    implementation_registry: DatasetImplementationRegistry | None,
    evidence_verifier_registry_id: str | tuple[str, ...] | None,
) -> tuple[DatasetEvidenceVerification, ...]:
    has_evidence = any(
        (
            any(record.evidence_refs for record in snapshot.releases),
            any(record.evidence_refs for record in snapshot.observations),
            any(record.verification_evidence_refs for record in snapshot.materializations),
            any(record.evidence_refs for record in snapshot.acquisition_attempts),
            any(record.evidence_refs for record in snapshot.bindings),
        )
    )
    resolved = _resolve_dataset_evidence_verifiers(
        implementation_registry,
        evidence_verifier_registry_id,
        required=has_evidence,
    )
    try:
        requests = _dataset_evidence_requests(snapshot)
    except ValueError as error:
        raise CatalogRebuildError("dataset evidence owner context is invalid") from error
    if not requests:
        return ()
    assert resolved
    unsupported_schemas = tuple(
        sorted(
            {
                request.reference.evidence_schema
                for request in requests
                if not any(
                    request.reference.evidence_schema in registration.supported_evidence_schema_ids
                    for registration, _verifier in resolved
                )
            }
        )
    )
    if unsupported_schemas:
        raise CatalogRebuildError(
            "dataset evidence verifier registration does not support every evidence schema"
        )
    # Each registered capability limits its own invocation. Aggregate evidence
    # retained by the rebuild remains bounded independently across this loop.
    remaining_input_bytes = MAX_DATASET_REBUILD_EVIDENCE_AGGREGATE_BYTES
    remaining_output_bytes = MAX_DATASET_REBUILD_EVIDENCE_AGGREGATE_BYTES
    results: list[DatasetEvidenceVerification] = []
    for request in requests:
        candidates = tuple(
            (registration, verifier)
            for registration, verifier in resolved
            if request.reference.evidence_schema in registration.supported_evidence_schema_ids
        )
        verified: list[tuple[DatasetEvidenceVerifierRegistration, DatasetEvidenceVerification]] = []
        last_error: Exception | None = None
        validation_errors: list[CatalogRebuildError] = []
        input_limit_exhausted = True
        for registration, verifier in candidates:
            maximum_bytes = min(
                MAX_DATASET_REBUILD_EVIDENCE_MEMBER_BYTES,
                remaining_input_bytes,
                registration.limits.max_input_bytes,
            )
            if maximum_bytes <= 0:
                continue
            input_limit_exhausted = False
            try:
                result = verifier.verify(request, maximum_bytes=maximum_bytes)
            except Exception as error:
                last_error = error
                continue
            if not isinstance(result, DatasetEvidenceVerification) or not result.validates(request):
                validation_errors.append(
                    CatalogRebuildError(
                        "dataset evidence verification differs from its exact reference"
                    )
                )
                continue
            if result.observed_size_bytes > maximum_bytes:
                validation_errors.append(
                    CatalogRebuildError("dataset evidence exceeds the registered input byte limit")
                )
                continue
            result_size_bytes = len(result.canonical_bytes())
            if result_size_bytes > min(
                remaining_output_bytes,
                registration.limits.max_output_bytes,
            ):
                validation_errors.append(
                    CatalogRebuildError("dataset evidence exceeds the registered output byte limit")
                )
                continue
            verified.append((registration, result))
        if input_limit_exhausted:
            raise CatalogRebuildError("dataset evidence exceeds the registered input byte limit")
        if not verified:
            if validation_errors:
                raise validation_errors[-1]
            raise CatalogRebuildError(
                "dataset evidence failed bounded read-only verification"
            ) from last_error
        distinct_results = {result.fingerprint(): result for _registration, result in verified}
        if len(distinct_results) != 1:
            raise CatalogRebuildError("registered dataset evidence verifiers disagree")
        result = next(iter(distinct_results.values()))
        result_size_bytes = len(result.canonical_bytes())
        remaining_input_bytes -= result.observed_size_bytes
        remaining_output_bytes -= result_size_bytes
        results.append(result)
    return tuple(results)


def _verify_projected_artifacts(
    snapshot: CatalogSnapshot,
    artifact_plane: ExternalArtifactPlane | None,
) -> None:
    verified_locators = tuple(
        locator
        for locator in snapshot.artifact_locators
        if locator.verification_status is CatalogVerificationStatus.VERIFIED
    )
    if not verified_locators:
        return
    if len(verified_locators) > MAX_CATALOG_REBUILD_VERIFIED_ARTIFACTS:
        raise CatalogRebuildError(
            "verified catalog artifact scan exceeds its aggregate member limit"
        )
    remaining_declared_bytes = MAX_CATALOG_REBUILD_VERIFIED_ARTIFACT_BYTES
    for locator in verified_locators:
        if locator.size_bytes > remaining_declared_bytes:
            raise CatalogRebuildError(
                "verified catalog artifact scan exceeds its aggregate byte limit"
            )
        remaining_declared_bytes -= locator.size_bytes
    if artifact_plane is None:
        raise CatalogRebuildError("verified catalog locators require an external artifact verifier")
    contract = artifact_plane.root.contract
    roots = {root.storage_root_id: root for root in snapshot.storage_roots}
    logical = {artifact.logical_artifact_id: artifact for artifact in snapshot.logical_artifacts}
    for locator in verified_locators:
        root = roots[locator.storage_root_id]
        if (
            root.storage_root_id != contract.storage_root_id
            or root.canonical_path != contract.canonical_path
            or root.mount_contract_schema != contract.mount_contract_schema
        ):
            raise CatalogRebuildError("catalog storage root differs from the guarded root")
        try:
            sidecar = artifact_plane.root.resolve(
                f"{locator.relative_path}.manifest.json",
                for_write=False,
            )
            manifest = decode_artifact_manifest(
                read_bounded_bytes(
                    sidecar,
                    maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
                )
            )
            artifact_plane.verify_manifest(manifest)
        except (ArtifactPlaneError, BoundedFileIOError, OSError, ValueError) as error:
            raise CatalogRebuildError(
                "catalog artifact failed custody or semantic verification"
            ) from error
        catalog_logical = logical[locator.logical_artifact_id]
        if (
            manifest.logical.logical_artifact_id != catalog_logical.logical_artifact_id
            or manifest.logical.content_sha256 != catalog_logical.content_sha256
            or manifest.logical.payload_schema != catalog_logical.payload_schema
            or manifest.logical.profile is not catalog_logical.profile
            or manifest.logical.media_type != catalog_logical.media_type
            or manifest.logical.visibility_ceiling is not catalog_logical.visibility_ceiling
        ):
            raise CatalogRebuildError("catalog logical artifact differs from its sidecar")
        materialization = manifest.materialization
        if (
            materialization.materialization_id != locator.materialization_id
            or materialization.logical_artifact_id != locator.logical_artifact_id
            or materialization.storage_root_id != locator.storage_root_id
            or materialization.relative_path != locator.relative_path
            or materialization.physical_sha256 != locator.physical_sha256
            or materialization.size_bytes != locator.size_bytes
            or materialization.compression != locator.compression
            or materialization.partition_selector != locator.partition_selector
        ):
            raise CatalogRebuildError("catalog locator differs from its sidecar")


def _json_mapping(payload: bytes, *, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as error:
        raise CatalogRebuildError(f"{label} is not UTF-8 JSON") from error
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise CatalogRebuildError(f"{label} is not a string-keyed JSON object")
    return value


def _declared_document_schema(payload: bytes, *, label: str) -> str:
    value = _json_mapping(payload, label=label)
    schema = value.get("schema")
    if not isinstance(schema, str):
        raise CatalogRebuildError(f"{label} does not declare a schema")
    try:
        return validate_schema(schema)
    except ValueError as error:
        raise CatalogRebuildError(f"{label} declares an invalid schema") from error




def _verify_projected_receipts(
    snapshot: CatalogSnapshot,
    artifact_plane: ExternalArtifactPlane | None,
) -> None:
    """Verify run/result receipts without conflating them with task receipts."""

    verified_receipts = tuple(
        receipt
        for receipt in snapshot.receipts
        if receipt.verification_status is CatalogVerificationStatus.VERIFIED
    )
    if not verified_receipts:
        return
    if len(verified_receipts) > MAX_CATALOG_REBUILD_VERIFIED_RECEIPTS:
        raise CatalogRebuildError("verified catalog receipt scan exceeds its member limit")
    if artifact_plane is None:
        raise CatalogRebuildError("verified catalog receipts require an external receipt verifier")

    contract = artifact_plane.root.contract
    roots = {root.storage_root_id: root for root in snapshot.storage_roots}
    remaining_bytes = MAX_CATALOG_REBUILD_RECEIPT_BYTES
    for receipt in verified_receipts:
        root = roots[receipt.storage_root_id]
        if (
            root.storage_root_id != contract.storage_root_id
            or root.canonical_path != contract.canonical_path
            or root.mount_contract_schema != contract.mount_contract_schema
        ):
            raise CatalogRebuildError("catalog receipt root differs from the guarded root")
        try:
            receipt_path = artifact_plane.root.resolve(receipt.relative_path, for_write=False)
            sidecar_path = artifact_plane.root.resolve(
                f"{receipt.relative_path}.manifest.json",
                for_write=False,
            )
            sidecar_payload = read_bounded_bytes(
                sidecar_path,
                maximum_bytes=MAX_ARTIFACT_MANIFEST_BYTES,
            )
            manifest = decode_artifact_manifest(sidecar_payload)
            materialization = manifest.materialization
            if (
                materialization.relative_path != receipt.relative_path
                or materialization.physical_sha256 != receipt.sha256
                or materialization.size_bytes
                > min(MAX_CATALOG_REBUILD_RECEIPT_MEMBER_BYTES, remaining_bytes)
                or manifest.logical.payload_schema != receipt.receipt_schema
                or manifest.logical.content_sha256 != receipt.sha256
            ):
                raise CatalogRebuildError(
                    "catalog receipt differs from its current custody sidecar"
                )
            artifact_plane.verify_manifest(manifest)
            receipt_payload = read_bounded_bytes(
                receipt_path,
                maximum_bytes=min(
                    MAX_CATALOG_REBUILD_RECEIPT_MEMBER_BYTES,
                    remaining_bytes,
                ),
            )
        except (
            ArtifactPlaneError,
            BoundedFileIOError,
            OSError,
            ValueError,
        ) as error:
            if isinstance(error, CatalogRebuildError):
                raise
            raise CatalogRebuildError("catalog receipt failed custody verification") from error
        if (
            len(receipt_payload) > remaining_bytes
            or hashlib.sha256(receipt_payload).hexdigest() != receipt.sha256
            or _declared_document_schema(
                receipt_payload,
                label="catalog receipt",
            )
            != receipt.receipt_schema
        ):
            raise CatalogRebuildError("catalog receipt bytes differ from their locator")
        remaining_bytes -= len(receipt_payload)

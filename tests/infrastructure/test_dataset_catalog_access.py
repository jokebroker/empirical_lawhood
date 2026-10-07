# SPDX-License-Identifier: MPL-2.0
# Adapted from icf-yolo: synthetic shared-core contract regressions.
from __future__ import annotations

from collections.abc import Callable
import hashlib
from pathlib import Path
from typing import cast

import pytest
from sqlalchemy import Engine, event
from sqlalchemy.exc import DatabaseError

from empirical_lawhood.infrastructure.dataset_catalog_access import (
    DatasetCatalogReadSessionProvider,
    load_local_dataset_projection_receipt_reference,
)
from empirical_lawhood.infrastructure.sql import (
    create_catalog_engine,
    create_read_only_catalog_engine,
    upgrade_catalog,
)
from empirical_lawhood.planning.datasets import (
    DatasetFamily,
    EvidenceReference,
    EvidenceReferenceKind,
    IdentityState,
)
from empirical_lawhood.runtime.dataset_catalog_service import (
    DatasetCatalogLookup,
    DatasetCatalogReadService,
)
from empirical_lawhood.runtime.datasets import (
    DatasetCatalogProjectionReceipt,
    DatasetCatalogProjectionAnchor,
    DatasetCatalogProjectionState,
    DatasetCatalogSnapshot,
    DatasetCatalogWorkLimit,
)


def _projection_anchor(
    snapshot: DatasetCatalogSnapshot,
    *,
    revision: int = 0,
) -> DatasetCatalogProjectionAnchor:
    state = DatasetCatalogProjectionState(
        snapshot_schema=snapshot.SCHEMA,
        snapshot_fingerprint=snapshot.fingerprint(),
        family_count=len(snapshot.families),
        release_count=len(snapshot.releases),
        observation_count=len(snapshot.observations),
        materialization_count=len(snapshot.materializations),
        acquisition_attempt_count=len(snapshot.acquisition_attempts),
        binding_count=len(snapshot.bindings),
        canonical_byte_count=len(snapshot.canonical_bytes()),
        projection_revision=revision,
    )
    return DatasetCatalogProjectionAnchor(
        receipt_id=f"receipt.dataset-projection.{snapshot.fingerprint()[:16]}",
        receipt_schema='empirical-lawhood/testing/fixtures/dataset-projection-receipt',
        receipt_sha256=hashlib.sha256(
            b"authenticated-projection-receipt\0" + snapshot.canonical_bytes()
        ).hexdigest(),
        projection_state=state,
    )


def _work_limit() -> DatasetCatalogWorkLimit:
    return DatasetCatalogWorkLimit(
        max_records_examined=10,
        max_query_steps=100_000,
    )


def _trust_reference() -> EvidenceReference:
    return EvidenceReference(
        evidence_id="receipt.dataset-projection.production",
        kind=EvidenceReferenceKind.RECEIPT,
        evidence_schema=DatasetCatalogProjectionReceipt.SCHEMA,
        evidence_sha256="a" * 64,
        storage_root_id="semios-empirical-lawhood",
        relative_locator="catalog/receipts/receipt.dataset-projection.production.json",
    )


def test_local_projection_trust_loads_only_owner_only_canonical_reference(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "authority"
    directory.mkdir(mode=0o700)
    path = directory / "trusted-dataset-projection.json"
    expected = _trust_reference()
    path.write_bytes(expected.canonical_bytes())
    path.chmod(0o600)

    assert load_local_dataset_projection_receipt_reference(path) == expected

    path.chmod(0o644)
    with pytest.raises(PermissionError, match="owner-only 0600"):
        load_local_dataset_projection_receipt_reference(path)


def test_local_projection_trust_rejects_wrong_record_kind(tmp_path: Path) -> None:
    directory = tmp_path / "authority"
    directory.mkdir(mode=0o700)
    path = directory / "trusted-dataset-projection.json"
    reference = _trust_reference()
    path.write_bytes(
        EvidenceReference(
            evidence_id=reference.evidence_id,
            kind=EvidenceReferenceKind.MANIFEST,
            evidence_schema=reference.evidence_schema,
            evidence_sha256=reference.evidence_sha256,
            storage_root_id=reference.storage_root_id,
            relative_locator=reference.relative_locator,
        ).canonical_bytes()
    )
    path.chmod(0o600)

    with pytest.raises(ValueError, match="projection receipt"):
        load_local_dataset_projection_receipt_reference(path)


@pytest.fixture
def empty_catalog_path(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.sqlite3"
    engine = create_catalog_engine(f"sqlite+pysqlite:///{path}")
    try:
        upgrade_catalog(engine)
    finally:
        engine.dispose()
    return path


class _TrackingReadOnlyEngineFactory:
    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.disposed: list[Engine] = []

    def __call__(self, path: Path) -> Engine:
        self.paths.append(path)
        engine = create_read_only_catalog_engine(path)
        event.listen(
            engine,
            "engine_disposed",
            lambda disposed_engine: self.disposed.append(disposed_engine),
        )
        return engine


def test_provider_composes_exact_authenticated_view_and_disposes_engine(
    empty_catalog_path: Path,
) -> None:
    snapshot = DatasetCatalogSnapshot.empty()
    anchor = _projection_anchor(snapshot)
    factory = _TrackingReadOnlyEngineFactory()
    provider = DatasetCatalogReadSessionProvider(
        empty_catalog_path,
        snapshot,
        anchor,
        engine_factory=factory,
    )

    with provider.open() as service:
        assert isinstance(service, DatasetCatalogReadService)
        assert service.trusted_snapshot is snapshot
        assert service.trusted_anchor is anchor
        page = service.page(DatasetCatalogLookup(work_limit=_work_limit()))
        assert page.records == ()

    assert factory.paths == [empty_catalog_path]
    assert len(factory.disposed) == 1


def test_provider_disposes_engine_when_session_consumer_raises(
    empty_catalog_path: Path,
) -> None:
    snapshot = DatasetCatalogSnapshot.empty()
    factory = _TrackingReadOnlyEngineFactory()
    provider = DatasetCatalogReadSessionProvider(
        empty_catalog_path,
        snapshot,
        _projection_anchor(snapshot),
        engine_factory=factory,
    )

    with pytest.raises(RuntimeError, match="consumer failed"):
        with provider.open():
            raise RuntimeError("consumer failed")

    assert len(factory.disposed) == 1


def test_provider_construction_does_not_touch_path_or_invoke_factory(tmp_path: Path) -> None:
    path = tmp_path / "absent.sqlite3"
    snapshot = DatasetCatalogSnapshot.empty()
    calls: list[Path] = []

    def fail_if_opened(received_path: Path) -> Engine:
        calls.append(received_path)
        raise FileNotFoundError(received_path)

    provider = DatasetCatalogReadSessionProvider(
        path,
        snapshot,
        _projection_anchor(snapshot),
        engine_factory=fail_if_opened,
    )

    assert calls == []
    assert not path.exists()
    with pytest.raises(FileNotFoundError):
        with provider.open():
            pytest.fail("the failing factory cannot yield a service")
    assert calls == [path]
    assert not path.exists()


def test_default_provider_does_not_create_or_migrate_absent_database(tmp_path: Path) -> None:
    path = tmp_path / "absent.sqlite3"
    snapshot = DatasetCatalogSnapshot.empty()
    provider = DatasetCatalogReadSessionProvider(
        path,
        snapshot,
        _projection_anchor(snapshot),
    )

    with pytest.raises(FileNotFoundError):
        with provider.open():
            pytest.fail("an absent catalog cannot yield a service")
    assert not path.exists()


def test_invalid_existing_database_fails_closed_through_read_service(tmp_path: Path) -> None:
    path = tmp_path / "invalid.sqlite3"
    invalid_bytes = b"not a SQLite database"
    path.write_bytes(invalid_bytes)
    snapshot = DatasetCatalogSnapshot.empty()
    provider = DatasetCatalogReadSessionProvider(
        path,
        snapshot,
        _projection_anchor(snapshot),
    )

    with provider.open() as service:
        with pytest.raises(DatabaseError):
            service.page(DatasetCatalogLookup(work_limit=_work_limit()))

    assert path.read_bytes() == invalid_bytes


def test_provider_rejects_non_path_and_noncanonical_trust_inputs(tmp_path: Path) -> None:
    snapshot = DatasetCatalogSnapshot.empty()
    anchor = _projection_anchor(snapshot)

    with pytest.raises(TypeError, match="database_path"):
        DatasetCatalogReadSessionProvider(
            cast(Path, str(tmp_path / "catalog.sqlite3")),
            snapshot,
            anchor,
        )
    with pytest.raises(TypeError, match="trusted_snapshot"):
        DatasetCatalogReadSessionProvider(
            tmp_path / "catalog.sqlite3",
            cast(DatasetCatalogSnapshot, object()),
            anchor,
        )
    with pytest.raises(TypeError, match="trusted_anchor"):
        DatasetCatalogReadSessionProvider(
            tmp_path / "catalog.sqlite3",
            snapshot,
            cast(DatasetCatalogProjectionAnchor, object()),
        )
    with pytest.raises(TypeError, match="engine_factory"):
        DatasetCatalogReadSessionProvider(
            tmp_path / "catalog.sqlite3",
            snapshot,
            anchor,
            engine_factory=cast(Callable[[Path], Engine], object()),
        )


def test_provider_rejects_anchor_for_another_exact_snapshot(tmp_path: Path) -> None:
    other_snapshot = DatasetCatalogSnapshot(
        families=(
            DatasetFamily(
                family_id="family.other",
                canonical_name="Other dataset",
                provider_id=None,
                external_identifiers=(),
                description="",
                keywords=(),
                first_observation_id=None,
                latest_observation_id=None,
                identity_state=IdentityState.FAMILY_RESOLVED,
                reason_codes=(),
            ),
        ),
        releases=(),
        observations=(),
        materializations=(),
        acquisition_attempts=(),
        bindings=(),
    )

    with pytest.raises(ValueError, match="exact snapshot"):
        DatasetCatalogReadSessionProvider(
            tmp_path / "catalog.sqlite3",
            DatasetCatalogSnapshot.empty(),
            _projection_anchor(other_snapshot, revision=1),
        )

# SPDX-License-Identifier: MPL-2.0
# Adapted from icf-yolo: synthetic shared-core contract regressions.
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import event

import empirical_lawhood.api.facade as api_facade
from empirical_lawhood.api import (
    CatalogQueryRequest,
    CatalogRebuildRequest,
    EmpiricalLawhoodApi,
)
from empirical_lawhood.infrastructure.catalog_projection import encode_catalog_snapshot
from empirical_lawhood.infrastructure.dataset_projection import encode_unified_catalog_snapshot
from empirical_lawhood.infrastructure.sql import (
    CatalogSchemaAuditLimitExceeded,
    CatalogSchemaAuditLimitReason,
)
from empirical_lawhood.kernel.evidence import VisibilityCeiling
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.catalog import (
    ArtifactLocatorRecord,
    CatalogSnapshot,
    CatalogVerificationStatus,
    LogicalArtifactRecord,
    ScientificObjectKind,
    ScientificObjectRecord,
    StorageRootRecord,
)
from empirical_lawhood.runtime.datasets import DatasetCatalogSnapshot, UnifiedCatalogSnapshot


def _snapshot(external: Path) -> CatalogSnapshot:
    return CatalogSnapshot(
        storage_roots=(
            StorageRootRecord(
                storage_root_id="test-external",
                logical_name="Test external artifacts",
                canonical_path=str(external.resolve()),
                mount_contract_schema='empirical-lawhood/testing/fixtures/catalog-root',
                verification_status=CatalogVerificationStatus.UNAVAILABLE,
            ),
        ),
        logical_artifacts=(
            LogicalArtifactRecord(
                logical_artifact_id="system-artifact.test-system",
                content_sha256="1" * 64,
                payload_schema='empirical-lawhood/kernel/system-spec',
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/json",
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            ),
        ),
        artifact_locators=(
            ArtifactLocatorRecord(
                materialization_id="materialization.test-system",
                logical_artifact_id="system-artifact.test-system",
                storage_root_id="test-external",
                relative_path="runs/test-run/system.json",
                physical_sha256="1" * 64,
                size_bytes=123,
                compression="none",
                partition_selector=None,
                verification_status=CatalogVerificationStatus.UNAVAILABLE,
            ),
        ),
        receipts=(),
        scientific_objects=(
            ScientificObjectRecord(
                scientific_object_id="test-system",
                kind=ScientificObjectKind.SYSTEM_SPEC,
                object_schema='empirical-lawhood/kernel/system-spec',
                object_sha256="1" * 64,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                categorical_status="ready",
                artifact_materialization_id="materialization.test-system",
                receipt_id=None,
            ),
        ),
        knowledge_edges=(),
        metric_definitions=(),
        metric_observations=(),
        exploration_attempts=(),
    )


def test_catalog_rebuild_is_confirmed_bounded_and_fingerprint_stable(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    repository.joinpath(".git").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    projection = external / "catalog-projection.json"
    projection.write_bytes(encode_catalog_snapshot(_snapshot(external)))
    api = EmpiricalLawhoodApi(repo_root=repository, external_root=external)

    preview = api.rebuild_catalog(CatalogRebuildRequest(projection))
    assert not preview.succeeded
    assert preview.reason_codes == ("WRITE_CONFIRMATION_REQUIRED",)
    assert not repository.joinpath(".empirical-lawhood").exists()

    first = api.rebuild_catalog(CatalogRebuildRequest(projection, confirmed=True))
    queried = api.query_catalog(CatalogQueryRequest(object_id="test-system"))
    inspected = api.inspect_system_id("test-system")
    second = api.rebuild_catalog(CatalogRebuildRequest(projection, confirmed=True))

    assert first.succeeded
    assert second.succeeded
    assert first.payload is not None and second.payload is not None
    assert first.payload.projection_sha256 == second.payload.projection_sha256
    assert first.payload.snapshot_sha256 == second.payload.snapshot_sha256
    assert first.payload.database_sha256 == second.payload.database_sha256
    assert queried.succeeded
    assert queried.payload is not None
    assert queried.payload.matched_count == 1
    assert queried.payload.objects[0].external_relative_path == "runs/test-run/system.json"
    assert inspected.succeeded
    serialized = queried.to_mapping()
    assert "payload" not in str(serialized["payload"]).casefold()
    assert "row_data" not in str(serialized).casefold()


def test_catalog_check_surfaces_a_typed_schema_audit_work_stop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    repository.joinpath(".git").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    projection = external / "catalog-projection.json"
    projection.write_bytes(encode_catalog_snapshot(CatalogSnapshot.empty()))
    api = EmpiricalLawhoodApi(repo_root=repository, external_root=external)
    assert api.rebuild_catalog(CatalogRebuildRequest(projection, confirmed=True)).succeeded

    def stopped_audit(_engine):  # type: ignore[no-untyped-def]
        raise CatalogSchemaAuditLimitExceeded(
            reason=CatalogSchemaAuditLimitReason.SQLITE_VM_STEPS,
            phase="integrity diagnostics",
            limit=100,
            observed=100,
        )

    monkeypatch.setattr(api_facade, "schema_audit", stopped_audit)
    result = api.check_catalog()

    assert result.status.value == "BLOCKED"
    assert result.payload is None
    assert result.reason_codes == ("SCHEMA_AUDIT_SQLITE_VM_STEP_LIMIT_EXCEEDED",)
    assert result.errors[0].category.value == "CAPABILITY"


def test_catalog_rebuild_refuses_projection_outside_external_root(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    repository.joinpath(".git").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    projection = tmp_path / "outside.json"
    projection.write_bytes(encode_catalog_snapshot(CatalogSnapshot.empty()))
    api = EmpiricalLawhoodApi(repo_root=repository, external_root=external)

    result = api.rebuild_catalog(CatalogRebuildRequest(projection, confirmed=True))

    assert not result.succeeded
    assert result.status.value == "INVALID"
    assert not repository.joinpath(".empirical-lawhood").exists()


def test_unified_catalog_rebuild_requires_dedicated_dataset_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    repository.joinpath(".git").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    projection = external / "unified-catalog-projection.json"
    projection.write_bytes(
        encode_unified_catalog_snapshot(
            UnifiedCatalogSnapshot(
                scientific=CatalogSnapshot.empty(),
                datasets=DatasetCatalogSnapshot.empty(),
            )
        )
    )

    def forbidden_rebuild(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unified rebuild must stop before catalog mutation")

    monkeypatch.setattr(api_facade, "rebuild_production_catalog", forbidden_rebuild)
    api = EmpiricalLawhoodApi(repo_root=repository, external_root=external)

    result = api.rebuild_catalog(CatalogRebuildRequest(projection, confirmed=True))

    assert result.status.value == "BLOCKED"
    assert result.reason_codes == ("DATASET_PROJECTION_REBUILD_AUTHORITY_REQUIRED",)
    assert result.errors[0].category.value == "AUTHORITY"
    assert result.payload is not None
    assert result.payload.confirmed
    assert not repository.joinpath(".empirical-lawhood").exists()


def test_catalog_rebuild_returns_typed_failure_before_decoding_oversized_projection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    repository.joinpath(".git").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    projection = external / "oversized-projection.json"
    with projection.open("wb") as handle:
        handle.truncate(api_facade.MAX_CATALOG_PROJECTION_BYTES + 1)
    monkeypatch.setattr(api_facade, "rebuild_production_catalog", lambda *_args, **_kwargs: 1 / 0)
    api = EmpiricalLawhoodApi(repo_root=repository, external_root=external)

    result = api.rebuild_catalog(CatalogRebuildRequest(projection, confirmed=True))

    assert result.status.value == "FAILED"
    assert result.reason_codes == ("CATALOG_REBUILD_FAILED",)
    assert result.errors[0].category.value == "STORAGE"
    assert result.errors[0].message == "catalog.rebuild storage boundary failed"
    assert not repository.joinpath(".empirical-lawhood").exists()


def test_catalog_cursor_is_stable_filter_bound_and_strict(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    repository.joinpath(".git").mkdir()
    external = tmp_path / "external"
    external.mkdir()
    base = _snapshot(external)
    template = base.scientific_objects[0]
    snapshot = replace(
        base,
        scientific_objects=tuple(
            replace(
                template,
                scientific_object_id=f"test-system-{index:03d}",
                object_sha256=hashlib.sha256(f"system-{index}".encode()).hexdigest(),
            )
            for index in range(128)
        ),
    )
    projection = external / "catalog-projection.json"
    projection.write_bytes(encode_catalog_snapshot(snapshot))
    api = EmpiricalLawhoodApi(repo_root=repository, external_root=external)
    assert api.rebuild_catalog(CatalogRebuildRequest(projection, confirmed=True)).succeeded

    statements: list[str] = []
    create_read_only = api_facade.create_read_only_catalog_engine

    def instrumented_read_only_engine(path: Path):  # type: ignore[no-untyped-def]
        engine = create_read_only(path)

        def record_statement(
            _connection: object,
            _cursor: object,
            statement: str,
            _parameters: object,
            _context: object,
            _many: bool,
        ) -> None:
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", record_statement)
        return engine

    monkeypatch.setattr(
        api_facade,
        "create_read_only_catalog_engine",
        instrumented_read_only_engine,
    )

    first = api.query_catalog(CatalogQueryRequest(limit=1, match_count_limit=7))
    assert first.payload is not None
    assert first.payload.matched_count == 7
    assert first.payload.match_count_limit == 7
    assert first.payload.matched_count_truncated
    assert first.payload.returned_count == 1
    assert first.payload.has_more
    assert first.payload.next_cursor is not None
    assert first.payload.query_work_limit == 100_000
    assert 0 < first.payload.query_work_steps <= first.payload.query_work_limit
    assert len(statements) == 8
    normalized_statements = tuple(statement.upper() for statement in statements)
    assert not any("INTEGRITY_CHECK" in statement for statement in normalized_statements)
    assert not any("FOREIGN_KEY_CHECK" in statement for statement in normalized_statements)
    bounded_queries = tuple(
        statement
        for statement in normalized_statements
        if "BOUNDED_CATALOG_MATCH_PROBE" in statement or "QUERY_ARTIFACT" in statement
    )
    assert len(bounded_queries) == 2
    assert all(" LIMIT " in statement for statement in bounded_queries)
    repeat = api.query_catalog(CatalogQueryRequest(limit=1, match_count_limit=7))
    assert repeat.to_mapping() == first.to_mapping()

    second = api.query_catalog(
        CatalogQueryRequest(
            limit=1,
            match_count_limit=7,
            cursor=first.payload.next_cursor,
        )
    )
    assert second.payload is not None
    assert second.payload.matched_count == 7
    assert second.payload.matched_count_truncated
    assert second.payload.has_more
    assert second.payload.objects[0].object_id == "test-system-001"

    mismatched = api.query_catalog(
        CatalogQueryRequest(
            kind="SYSTEM_SPEC",
            limit=1,
            cursor=first.payload.next_cursor,
        )
    )
    assert mismatched.status.value == "INVALID"
    assert mismatched.reason_codes == ("AUTHORING_DOCUMENT_INVALID",)
    tampered = api.query_catalog(
        CatalogQueryRequest(limit=1, cursor=f"{first.payload.next_cursor}!")
    )
    assert tampered.status.value == "INVALID"

    excessive_count_budget = api.query_catalog(CatalogQueryRequest(match_count_limit=10_001))
    assert excessive_count_budget.status.value == "INVALID"
    assert excessive_count_budget.reason_codes == ("AUTHORING_DOCUMENT_INVALID",)

    repository_type = api_facade.SQLiteCatalogRepository
    monkeypatch.setattr(
        api_facade,
        "SQLiteCatalogRepository",
        lambda engine: repository_type(engine, query_work_limit=100),
    )
    exhausted = api.query_catalog(CatalogQueryRequest(categorical_status="absent"))
    assert exhausted.status.value == "BLOCKED"
    assert exhausted.payload is None
    assert exhausted.reason_codes == ("CATALOG_QUERY_WORK_LIMIT_EXCEEDED",)
    assert exhausted.errors[0].category.value == "CAPABILITY"

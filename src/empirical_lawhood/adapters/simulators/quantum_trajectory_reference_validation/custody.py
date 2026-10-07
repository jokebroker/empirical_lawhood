"""External-only immutable artifact publication and receipt verification."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from typing import Mapping, Protocol, Sequence, cast

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .schemas import QuantumTrajectoryReferenceValidationConfig, stable_json_bytes


class _StorageDiagnostic(Protocol):
    mount_active: bool
    canonical_contained: bool
    symlink_safe: bool
    mount_source: str | None
    mount_source_matches: bool
    volume_identity: str | None
    volume_identity_matches: bool
    filesystem_type: str | None
    filesystem_allowed: bool
    writable: bool
    observed_free_bytes: int
    effective_write_floor_bytes: int
    read_ready: bool
    write_ready: bool
    reason_codes: tuple[str, ...]


class QuantumTrajectoryReferenceValidationExternalRoot(Protocol):
    def diagnostic(self, *, operation_minimum_free_bytes: int = 0) -> _StorageDiagnostic: ...

    def verify(
        self,
        *,
        for_write: bool,
        operation_minimum_free_bytes: int = 0,
    ) -> Path: ...

    def resolve(
        self,
        relative_path: str,
        *,
        for_write: bool,
        operation_minimum_free_bytes: int = 0,
    ) -> Path: ...


def _diagnostic_document(diagnostic: _StorageDiagnostic) -> dict[str, object]:
    return {
        "mount_active": diagnostic.mount_active,
        "canonical_contained": diagnostic.canonical_contained,
        "symlink_safe": diagnostic.symlink_safe,
        "mount_source": diagnostic.mount_source,
        "mount_source_matches": diagnostic.mount_source_matches,
        "volume_identity": diagnostic.volume_identity,
        "volume_identity_matches": diagnostic.volume_identity_matches,
        "filesystem_type": diagnostic.filesystem_type,
        "filesystem_allowed": diagnostic.filesystem_allowed,
        "writable": diagnostic.writable,
        "observed_free_bytes": diagnostic.observed_free_bytes,
        "effective_write_floor_bytes": diagnostic.effective_write_floor_bytes,
        "read_ready": diagnostic.read_ready,
        "write_ready": diagnostic.write_ready,
        "reason_codes": diagnostic.reason_codes,
    }


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _parquet_table(rows: Sequence[Mapping[str, object]]) -> pa.Table:
    """Infer unsigned top-level integer columns without signed narrowing."""

    materialized = [dict(row) for row in rows]
    if not materialized:
        return pa.Table.from_pylist(materialized)
    unsigned_fields: set[str] = set()
    for field in set().union(*(row.keys() for row in materialized)):
        values = [row.get(field) for row in materialized if row.get(field) is not None]
        integers = [
            value for value in values if isinstance(value, int) and not isinstance(value, bool)
        ]
        if integers and len(integers) == len(values) and max(integers) > (1 << 63) - 1:
            if min(integers) < 0 or max(integers) > (1 << 64) - 1:
                raise OverflowError(f"{field} is outside the exact uint64 domain")
            unsigned_fields.add(field)
    inference_rows = [
        {
            key: (0 if key in unsigned_fields and value is not None else value)
            for key, value in row.items()
        }
        for row in materialized
    ]
    schema = pa.Table.from_pylist(inference_rows).schema
    for field in sorted(unsigned_fields):
        index = schema.get_field_index(field)
        schema = schema.set(index, pa.field(field, pa.uint64()))
    return pa.Table.from_pylist(materialized, schema=schema)


class QuantumTrajectoryReferenceValidationArtifactStore:
    def __init__(
        self,
        config: QuantumTrajectoryReferenceValidationConfig,
        *,
        guard: QuantumTrajectoryReferenceValidationExternalRoot,
    ) -> None:
        self.config = config
        self._guard = guard

    def storage_diagnostic(self, *, initial: bool) -> dict[str, object]:
        minimum = (
            self.config.initial_minimum_free_bytes
            if initial
            else self.config.subsequent_minimum_free_bytes
        )
        diagnostic = self._guard.diagnostic(operation_minimum_free_bytes=minimum)
        return {
            "schema": 'empirical-lawhood/infrastructure/quantum-trajectory-reference-validation/storage-diagnostic',
            "version": '1.0.0',
            "value": _diagnostic_document(diagnostic),
        }

    def preflight(self, *, initial: bool) -> Path:
        minimum = (
            self.config.initial_minimum_free_bytes
            if initial
            else self.config.subsequent_minimum_free_bytes
        )
        return self._guard.verify(
            for_write=True,
            operation_minimum_free_bytes=minimum,
        )

    def path(
        self,
        relative_path: str,
        *,
        for_write: bool,
        minimum_free_bytes: int = 0,
    ) -> Path:
        if relative_path.startswith("/") or ".." in Path(relative_path).parts:
            raise ValueError("quantum trajectory reference validation artifact path escapes its run root")
        return self._guard.resolve(
            f"{self.config.external_root}/{relative_path}",
            for_write=for_write,
            operation_minimum_free_bytes=minimum_free_bytes,
        )

    def run_root(self, *, for_write: bool) -> Path:
        return self._guard.resolve(
            self.config.external_root,
            for_write=for_write,
            operation_minimum_free_bytes=(
                self.config.subsequent_minimum_free_bytes if for_write else 0
            ),
        )

    def _receipt(
        self,
        *,
        relative_path: str,
        payload: bytes,
        predecessor_sha256: str | None,
        principal: str,
    ) -> dict[str, object]:
        mount = self._guard.verify(
            for_write=True,
            operation_minimum_free_bytes=max(len(payload), 1),
        )
        return {
            "schema": 'empirical-lawhood/runtime/quantum-trajectory-reference-validation/publication-receipt',
            "version": '1.0.0',
            "value": {
                "relative_path": relative_path,
                "size_bytes": len(payload),
                "sha256": sha256(payload).hexdigest(),
                "predecessor_sha256": predecessor_sha256,
                "principal": principal,
                "published_at_utc": utc_now(),
                "storage_root": str(mount),
                "write_flush": "COMPLETE",
                "read_back": "VERIFIED",
            },
        }

    @staticmethod
    def _write_once(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staged = path.with_name(f"{path.name}.{os.getpid()}.{os.urandom(8).hex()}.partial")
        try:
            with staged.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if path.exists():
                if path.read_bytes() != payload:
                    raise RuntimeError(f"immutable artifact conflict: {path}")
                return
            os.replace(staged, path)
        finally:
            if staged.exists():
                staged.unlink()

    def publish_bytes(
        self,
        relative_path: str,
        payload: bytes,
        *,
        predecessor_sha256: str | None = None,
        principal: str,
    ) -> dict[str, object]:
        destination = self.path(
            relative_path,
            for_write=True,
            minimum_free_bytes=max(
                self.config.subsequent_minimum_free_bytes,
                len(payload),
            ),
        )
        receipt_path = destination.with_suffix(destination.suffix + ".receipt.json")
        if destination.exists() and receipt_path.exists():
            receipt = cast(dict[str, object], json.loads(receipt_path.read_text()))
            self._verify_receipt(relative_path, destination, receipt)
            receipt_value = receipt.get("value")
            if (
                not isinstance(receipt_value, Mapping)
                or receipt_value.get("predecessor_sha256") != predecessor_sha256
                or receipt_value.get("principal") != principal
            ):
                raise RuntimeError(f"immutable receipt ancestry differs: {relative_path}")
            if destination.read_bytes() != payload:
                raise RuntimeError(f"immutable artifact conflict: {relative_path}")
            return receipt
        if receipt_path.exists() and not destination.exists():
            receipt = cast(dict[str, object], json.loads(receipt_path.read_text()))
            receipt_value = receipt.get("value")
            if (
                not isinstance(receipt_value, Mapping)
                or receipt_value.get("relative_path") != relative_path
                or receipt_value.get("size_bytes") != len(payload)
                or receipt_value.get("sha256") != sha256(payload).hexdigest()
                or receipt_value.get("predecessor_sha256") != predecessor_sha256
                or receipt_value.get("principal") != principal
            ):
                raise RuntimeError(
                    f"orphan receipt differs from requested artifact: {relative_path}"
                )
            self._write_once(destination, payload)
            self._verify_receipt(relative_path, destination, receipt)
            return receipt
        receipt = self._receipt(
            relative_path=relative_path,
            payload=payload,
            predecessor_sha256=predecessor_sha256,
            principal=principal,
        )
        receipt_payload = stable_json_bytes(receipt)
        self._write_once(destination, payload)
        if destination.read_bytes() != payload:
            raise RuntimeError(f"artifact read-back failed: {relative_path}")
        self._write_once(receipt_path, receipt_payload)
        self._verify_receipt(relative_path, destination, receipt)
        return receipt

    def publish_json(
        self,
        relative_path: str,
        document: Mapping[str, object],
        *,
        predecessor_sha256: str | None = None,
        principal: str,
    ) -> dict[str, object]:
        payload = stable_json_bytes(document)
        if len(payload) > self.config.maximum_json_bytes:
            raise RuntimeError("quantum trajectory reference validation JSON artifact exceeds its bound")
        return self.publish_bytes(
            relative_path,
            payload,
            predecessor_sha256=predecessor_sha256,
            principal=principal,
        )

    def publish_parquet(
        self,
        relative_path: str,
        rows: Sequence[Mapping[str, object]],
        *,
        predecessor_sha256: str | None = None,
        principal: str,
    ) -> dict[str, object]:
        table = _parquet_table(rows)
        sink = BytesIO()
        pq.write_table(
            table,
            sink,
            compression="zstd",
            version="2.6",
            write_statistics=True,
        )
        return self.publish_bytes(
            relative_path,
            sink.getvalue(),
            predecessor_sha256=predecessor_sha256,
            principal=principal,
        )

    def _verify_receipt(
        self,
        relative_path: str,
        destination: Path,
        receipt: Mapping[str, object],
    ) -> None:
        value = receipt.get("value")
        if not isinstance(value, Mapping):
            raise RuntimeError(f"invalid receipt: {relative_path}")
        if (
            value.get("relative_path") != relative_path
            or value.get("size_bytes") != destination.stat().st_size
            or value.get("sha256") != _sha256_file(destination)
            or value.get("read_back") != "VERIFIED"
        ):
            raise RuntimeError(f"receipt mismatch: {relative_path}")

    def load_json(self, relative_path: str) -> dict[str, object]:
        path = self.path(relative_path, for_write=False)
        receipt_path = path.with_suffix(path.suffix + ".receipt.json")
        receipt = cast(dict[str, object], json.loads(receipt_path.read_text()))
        self._verify_receipt(relative_path, path, receipt)
        payload = path.read_bytes()
        document = cast(dict[str, object], json.loads(payload))
        if stable_json_bytes(document) != payload:
            raise RuntimeError(f"JSON artifact is not canonical: {relative_path}")
        return document

    def load_bytes(self, relative_path: str) -> bytes:
        path = self.path(relative_path, for_write=False)
        receipt_path = path.with_suffix(path.suffix + ".receipt.json")
        receipt = cast(dict[str, object], json.loads(receipt_path.read_text()))
        self._verify_receipt(relative_path, path, receipt)
        return path.read_bytes()

    def load_parquet(self, relative_path: str) -> list[dict[str, object]]:
        payload = self.load_bytes(relative_path)
        return cast(
            list[dict[str, object]],
            pq.read_table(BytesIO(payload)).to_pylist(),
        )

    def exists_verified(self, relative_path: str) -> bool:
        try:
            path = self.path(relative_path, for_write=False)
            receipt_path = path.with_suffix(path.suffix + ".receipt.json")
            receipt = cast(dict[str, object], json.loads(receipt_path.read_text()))
            self._verify_receipt(relative_path, path, receipt)
        except (FileNotFoundError, RuntimeError, json.JSONDecodeError):
            return False
        return True

    def verify_all(self) -> dict[str, object]:
        run_root = self.run_root(for_write=False)
        artifact_paths = sorted(
            path
            for path in run_root.rglob("*")
            if path.is_file()
            and not path.name.endswith(".receipt.json")
            and ".partial" not in path.name
        )
        total_bytes = 0
        rows: list[dict[str, object]] = []
        for path in artifact_paths:
            relative = path.relative_to(run_root).as_posix()
            receipt_path = path.with_suffix(path.suffix + ".receipt.json")
            receipt = cast(dict[str, object], json.loads(receipt_path.read_text()))
            self._verify_receipt(relative, path, receipt)
            total_bytes += path.stat().st_size
            rows.append(
                {
                    "relative_path": relative,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
        return {
            "artifact_count": len(rows),
            "scientific_payload_bytes": total_bytes,
            "artifacts": rows,
        }


__all__ = ['QuantumTrajectoryReferenceValidationArtifactStore', "utc_now"]

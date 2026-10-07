"""Pure receipt and intended-key contracts for external quantum support conditioned identification custody."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import PurePosixPath
from typing import Mapping, Sequence

from .contracts import Disposition


def validate_relative_path(relative_path: str) -> None:
    path = PurePosixPath(relative_path)
    if not relative_path or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError("quantum support conditioned identification artifact path escapes its run root")


def receipt_document(
    *,
    relative_path: str,
    payload: bytes,
    principal: str,
    published_at_utc: str,
    storage_root: str,
    predecessor_sha256: str | None = None,
) -> dict[str, object]:
    validate_relative_path(relative_path)
    if not principal or not published_at_utc or not storage_root:
        raise ValueError("quantum support conditioned identification receipt identity is incomplete")
    return {
        "schema": 'empirical-lawhood/runtime/quantum-support-conditioned-identification/publication-receipt',
        "version": '1.0.0',
        "value": {
            "relative_path": relative_path,
            "size_bytes": len(payload),
            "sha256": sha256(payload).hexdigest(),
            "predecessor_sha256": predecessor_sha256,
            "principal": principal,
            "published_at_utc": published_at_utc,
            "storage_root": storage_root,
            "write_flush": "COMPLETE",
            "read_back": "VERIFIED",
        },
    }


def validate_receipt(
    receipt: Mapping[str, object],
    *,
    relative_path: str,
    payload: bytes,
) -> None:
    value = receipt.get("value")
    if (
        receipt.get("schema") != 'empirical-lawhood/runtime/quantum-support-conditioned-identification/publication-receipt'
        or receipt.get("version") != "3.2.0"
        or not isinstance(value, Mapping)
        or value.get("relative_path") != relative_path
        or value.get("size_bytes") != len(payload)
        or value.get("sha256") != sha256(payload).hexdigest()
        or value.get("write_flush") != "COMPLETE"
        or value.get("read_back") != "VERIFIED"
    ):
        raise ValueError("quantum support conditioned identification receipt does not bind the payload")


@dataclass(frozen=True, slots=True)
class IntendedKey:
    key_id: str
    parent_id: str
    branch_id: str
    disposition: Disposition
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if self.disposition is not Disposition.COMPLETE and not self.reason_code:
            raise ValueError("noncomplete intended key lacks a reason code")


def close_intended_keys(
    expected_key_ids: Sequence[str],
    rows: Sequence[IntendedKey],
) -> None:
    observed = [row.key_id for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != set(expected_key_ids):
        raise ValueError("intended-key closure is duplicated or incomplete")


__all__ = [
    "IntendedKey",
    "close_intended_keys",
    "receipt_document",
    "validate_receipt",
    "validate_relative_path",
]

"""Bind retained terminal exports to an independently checked current closeout.

The caller obtains export verification and target custody outside this pure
calculation. These records check exact input bindings. They neither authenticate
a supplied verification identity nor transfer qualification or execution grants.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
import re
from typing import Any, ClassVar, Mapping

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_sha256,
    validate_stable_id,
)


_REQUEST_SCHEMA = "empirical-lawhood/methods/integrated-response-closeout-request"
_TERMINAL_SCHEMA = "empirical-lawhood/methods/integrated-response-terminal-export"
_DETAIL_SCHEMA = "empirical-lawhood/methods/integrated-response-detail-export"


def _document_bytes(document: Mapping[str, Any]) -> bytes:
    """Encode finite JSON observations without changing their numeric values."""

    try:
        return (
            json.dumps(
                dict(document),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError("closeout input requires finite JSON observations") from exc


def integrated_closeout_request_identity(
    *,
    request_id: str,
    terminals: Mapping[str, Mapping[str, Any]],
    details: Mapping[str, Mapping[str, Any]],
    retained_attempts: Mapping[str, Mapping[str, Any]],
) -> ObjectIdentity:
    """Identify the exact current input request; confer no source authority."""

    validate_stable_id(request_id, field_name="request_id")
    payload = _document_bytes(
        {
            "request_id": request_id,
            "terminal_sha256": sha256(_document_bytes(terminals)).hexdigest(),
            "detail_sha256": sha256(_document_bytes(details)).hexdigest(),
            "retained_attempt_sha256": sha256(_document_bytes(retained_attempts)).hexdigest(),
        }
    )
    return ObjectIdentity(
        object_id=request_id,
        object_schema=_REQUEST_SCHEMA,
        object_version="1.0.0",
        object_fingerprint=sha256(payload).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class IntegratedCloseoutSourceExport(CanonicalRecord):
    """Separate immutable original lineage from current terminal input custody."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/integrated-closeout-source-export"

    source_id: str
    original_terminal: ArtifactIdentity
    original_detail: ArtifactIdentity | None
    original_task_receipt: ObjectIdentity
    original_manifest_sha256: str
    original_source_commit: str
    interpretation_sha256: str
    verified_export: ObjectIdentity
    target_terminal: ArtifactIdentity
    target_detail: ArtifactIdentity | None
    target_task_receipt: ObjectIdentity
    target_manifest_sha256: str
    target_request: ObjectIdentity
    grants_authority: bool = False

    def __post_init__(self) -> None:
        validate_stable_id(self.source_id, field_name="source_id")
        for name in (
            "original_manifest_sha256",
            "interpretation_sha256",
            "target_manifest_sha256",
        ):
            validate_sha256(getattr(self, name), field_name=name)
        if (
            re.fullmatch(r"[0-9a-f]{40}", self.original_source_commit) is None
            or self.verified_export.object_schema
            != "empirical-lawhood/migration/verified-source-export"
            or self.verified_export.object_version != "1.0.0"
            or self.original_terminal == self.target_terminal
            or self.original_task_receipt == self.target_task_receipt
            or self.target_terminal.payload_schema != _TERMINAL_SCHEMA
            or (self.original_detail is None) != (self.target_detail is None)
            or (
                self.target_detail is not None
                and (
                    self.target_detail.payload_schema != _DETAIL_SCHEMA
                    or self.original_detail == self.target_detail
                )
            )
            or self.target_task_receipt.object_schema
            != "empirical-lawhood/runtime/canonical-task-receipt"
            or self.target_task_receipt.object_version != "1.0.0"
            or self.target_request.object_schema != _REQUEST_SCHEMA
            or self.target_request.object_version != "1.0.0"
            or self.grants_authority is not False
        ):
            raise ValueError(
                "closeout requires independently verified original export and separate target custody; qualification and grants do not transfer"
            )

    def require_current_inputs(
        self,
        *,
        terminal: Mapping[str, Any],
        detail: Mapping[str, Any] | None,
        target_request: ObjectIdentity,
    ) -> None:
        terminal_bytes = _document_bytes(terminal)
        detail_bytes = None if detail is None else _document_bytes(detail)
        if (
            self.target_request != target_request
            or self.target_terminal.sha256 != sha256(terminal_bytes).hexdigest()
            or self.target_terminal.size_bytes != len(terminal_bytes)
            or (self.target_detail is None) != (detail_bytes is None)
            or (
                self.target_detail is not None
                and detail_bytes is not None
                and (
                    self.target_detail.sha256 != sha256(detail_bytes).hexdigest()
                    or self.target_detail.size_bytes != len(detail_bytes)
                )
            )
        ):
            raise ValueError("retained export does not bind the exact current closeout input")


def require_closeout_exports(
    *,
    terminals: Mapping[str, Mapping[str, Any]],
    details: Mapping[str, Mapping[str, Any]],
    retained_attempts: Mapping[str, Mapping[str, Any]],
    source_exports: tuple[IntegratedCloseoutSourceExport, ...],
    target_request: ObjectIdentity,
) -> None:
    """Refuse absent or mismatched external mappings before scientific reduction."""

    if (
        not isinstance(target_request, ObjectIdentity)
        or not isinstance(source_exports, tuple)
        or any(not isinstance(value, IntegratedCloseoutSourceExport) for value in source_exports)
    ):
        raise ValueError("retained closeout requires typed verified exports and current request custody")
    expected_request = integrated_closeout_request_identity(
        request_id=target_request.object_id,
        terminals=terminals,
        details=details,
        retained_attempts=retained_attempts,
    )
    if target_request != expected_request:
        raise ValueError("closeout target request differs from the exact current inputs")
    if set(terminals) & set(retained_attempts):
        raise ValueError("closeout terminal and retained source identities overlap")
    records = {**terminals, **retained_attempts}
    source_ids = tuple(value.source_id for value in source_exports)
    if source_ids != tuple(sorted(records)) or not set(details).issubset(terminals):
        raise ValueError("retained closeout requires a complete ordered verified export census")
    for value in source_exports:
        value.require_current_inputs(
            terminal=records[value.source_id],
            detail=details.get(value.source_id),
            target_request=target_request,
        )

"""Read-only target predecessor compatibility.

Historical bytes require a separately verified external export/import mapping
to current descriptors, exact source interpretation and target custody. This
reader transfers no original qualification or receipt grant.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import ClassVar, Mapping

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_relative_locator,
    validate_sha256,
)
from empirical_lawhood.infrastructure.bounded_io import read_bounded_bytes

from .contracts import MAXIMUM_JSON_BYTES, Verdict
from ..quantum_response_control.contracts import Verdict as ResponseControlVerdict


_RESPONSE_CONTROL_PATHS = (
    "controls/freeze.json",
    "stages/source-compatibility/result.json",
    "stages/instrument-conformance/result.json",
    "stages/observation-order-evaluation/result.json",
    "stages/observation-order-evaluation/result.json.receipt.json",
    "closeout/result.json",
    "closeout/result.json.receipt.json",
    "controls/implementation-manifest.json",
)
_PREPARATION_QUALIFICATION_PATHS = ("closeout/result.json", "closeout/result.json.receipt.json")


@dataclass(frozen=True, slots=True)
class PredecessorInputs(CanonicalRecord):
    """Caller-frozen predecessor bytes, never a shipped historical authority."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/quantum-marked-record-identification/predecessor-inputs'
    response_control_root: str
    preparation_qualification_root: str
    payload_sha256s: tuple[tuple[str, str], ...]
    implementation_sha256: str

    def __post_init__(self) -> None:
        validate_relative_locator(self.response_control_root)
        validate_relative_locator(self.preparation_qualification_root)
        validate_sha256(self.implementation_sha256)
        expected = tuple(
            sorted(
                [f"{self.response_control_root}/{p}" for p in _RESPONSE_CONTROL_PATHS]
                + [f"{self.preparation_qualification_root}/{p}" for p in _PREPARATION_QUALIFICATION_PATHS]
            )
        )
        if tuple(path for path, _ in self.payload_sha256s) != expected:
            raise ValueError("predecessor input must bind every payload and receipt")
        for _, digest in self.payload_sha256s:
            validate_sha256(digest)


@dataclass(frozen=True, slots=True)
class CompatibilityResult:
    verdict: Verdict
    checked_sha256: Mapping[str, str]
    reason_codes: tuple[str, ...]
    field_classification: Mapping[str, str]


def _bounded_json(path: Path) -> tuple[Mapping[str, object], bytes]:
    payload = read_bounded_bytes(path, maximum_bytes=MAXIMUM_JSON_BYTES)
    if len(payload) > MAXIMUM_JSON_BYTES:
        raise ValueError("predecessor JSON exceeds the bounded decoder")
    document = json.loads(payload)
    if not isinstance(document, Mapping):
        raise TypeError("predecessor JSON root is not an object")
    return document, payload


def _validate_receipt(path: Path, payload_path: Path) -> None:
    receipt, _ = _bounded_json(path)
    value = receipt.get("value")
    if not isinstance(value, Mapping):
        raise ValueError("predecessor receipt has no value object")
    payload = read_bounded_bytes(payload_path, maximum_bytes=MAXIMUM_JSON_BYTES)
    if (
        value.get("sha256") != sha256(payload).hexdigest()
        or value.get("size_bytes") != len(payload)
        or value.get("write_flush") != "COMPLETE"
        or value.get("read_back") != "VERIFIED"
    ):
        raise ValueError("predecessor receipt does not bind its payload")


def compare_predecessors(
    storage_root: Path, *, inputs: PredecessorInputs
) -> CompatibilityResult:
    checked: dict[str, str] = {}
    reasons: list[str] = []
    try:
        if type(inputs) is not PredecessorInputs:
            raise ValueError("typed predecessor inputs required")
        resolved_root = storage_root.resolve(strict=True)
        response_control_root, preparation_qualification_root = inputs.response_control_root, inputs.preparation_qualification_root
        for relative, expected in inputs.payload_sha256s:
            path = (resolved_root / relative).resolve(strict=True)
            if not path.is_relative_to(resolved_root):
                raise ValueError("predecessor path escapes storage")
            payload = read_bounded_bytes(path, maximum_bytes=MAXIMUM_JSON_BYTES)
            observed = sha256(payload).hexdigest()
            checked[relative] = observed
            if observed != expected:
                reasons.append(f"identity-differs:{relative}")
        implementation_path = (
            resolved_root / response_control_root / "controls/implementation-manifest.json"
        )
        implementation, _ = _bounded_json(implementation_path)
        implementation_value = implementation.get("value")
        if (
            not isinstance(implementation_value, Mapping)
            or implementation_value.get("implementation_sha256")
            != inputs.implementation_sha256
        ):
            reasons.append("quantum-response-control-implementation-identity-differs")
        for receipt_relative in (
            f"{response_control_root}/stages/observation-order-evaluation/result.json.receipt.json",
            f"{response_control_root}/closeout/result.json.receipt.json",
            f"{preparation_qualification_root}/closeout/result.json.receipt.json",
        ):
            receipt_path = resolved_root / receipt_relative
            _validate_receipt(
                receipt_path,
                Path(str(receipt_path).removesuffix(".receipt.json")),
            )
        observation_order_result, _ = _bounded_json(resolved_root / response_control_root / "stages/observation-order-evaluation/result.json")
        observation_order_value = observation_order_result.get("value")
        if (
            not isinstance(observation_order_value, Mapping)
            or observation_order_value.get("verdict") != ResponseControlVerdict.OBSERVATION_ORDER_RECEIVER_PROXY_STOP.value
            or observation_order_value.get("parent_count") != 2048
        ):
            reasons.append("quantum-response-control-terminal-semantics-differ")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        return CompatibilityResult(
            verdict=Verdict.COMPATIBILITY_UNEVALUABLE,
            checked_sha256=checked,
            reason_codes=(f"bounded-predecessor-read-failed:{type(error).__name__}",),
            field_classification={},
        )
    classification = {
        "basis_order": "IDENTICAL",
        "hamiltonian_H0": "IDENTICAL",
        "preparation_distribution": "IDENTICAL",
        "monitoring_jump_law": "IDENTICAL",
        "cutoff_and_event_ordering": "IDENTICAL",
        "future_hold_window": "COMPATIBLE_ADDITIVE",
        "record_feature_grammar": "COMPATIBLE_ADDITIVE",
        "recognition_compiler": "COMPATIBLE_ADDITIVE",
        "nonhold_actions": "OMITTED_BY_PASSIVE_CONTRACT",
    }
    verdict = (
        Verdict.PREDECESSOR_IDENTITY_STOP
        if reasons
        else Verdict.COMPATIBLE_RECEIVER_IDENTIFICATION_SOURCE_READY
    )
    return CompatibilityResult(
        verdict=verdict,
        checked_sha256=checked,
        reason_codes=tuple(reasons),
        field_classification=classification,
    )


__all__ = [
    "CompatibilityResult",
    'PredecessorInputs',
    "compare_predecessors",
]

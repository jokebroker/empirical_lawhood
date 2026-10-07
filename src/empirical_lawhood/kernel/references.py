"""Safe content identities and registered executable-reference protocols."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar, Protocol

from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class ArtifactIdentity(CanonicalRecord):
    """Content identity for bytes stored outside the scientific IR."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/artifact-identity'

    artifact_id: str
    role: str
    payload_schema: str
    sha256: str
    media_type: str
    size_bytes: int
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.artifact_id, field_name="artifact_id")
        validate_stable_id(self.role, field_name="role")
        validate_schema(self.payload_schema)
        validate_sha256(self.sha256)
        validate_nonempty(self.media_type, field_name="media_type")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be nonnegative")
        require_extensions(self.extensions)


class SafePayloadFormat(StrEnum):
    CANONICAL_JSON = "CANONICAL_JSON"
    ARROW_IPC = "ARROW_IPC"
    PARQUET = "PARQUET"
    ONNX = "ONNX"
    SAFETENSORS = "SAFETENSORS"
    NUMPY_NO_PICKLE = "NUMPY_NO_PICKLE"
    TEXT_PARAMETERS = "TEXT_PARAMETERS"


@dataclass(frozen=True, slots=True)
class ExecutableReference(CanonicalRecord):
    """Static capability and safe payload identity, never an import path."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/executable-reference'

    reference_id: str
    capability_key: str
    capability_version: str
    evaluator_key: str
    payload: ArtifactIdentity
    payload_format: SafePayloadFormat
    input_schema: str
    output_schema: str
    deterministic: bool
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("reference_id", self.reference_id),
            ("capability_key", self.capability_key),
            ("evaluator_key", self.evaluator_key),
        ):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.capability_version)
        validate_schema(self.input_schema)
        validate_schema(self.output_schema)
        if not isinstance(self.payload_format, SafePayloadFormat):
            raise ValueError("executable payload format is not in the safe allowlist")
        unsafe_media_fragments = ("joblib", "pickle", "python-object")
        if any(fragment in self.payload.media_type.lower() for fragment in unsafe_media_fragments):
            raise ValueError("executable serialization is forbidden")
        require_extensions(self.extensions)


@dataclass(frozen=True, slots=True)
class NamedDecimal(CanonicalRecord):
    """One named scalar with an explicit native unit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/named-decimal'

    value_id: str
    value: Decimal
    unit: str

    def __post_init__(self) -> None:
        validate_stable_id(self.value_id, field_name="value_id")
        validate_decimal(self.value, field_name="value")
        validate_nonempty(self.unit, field_name="unit")


@dataclass(frozen=True, slots=True)
class QuantityBound(CanonicalRecord):
    """Closed native-unit bound for one quantity or response coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/quantity-bound'

    bound_id: str
    quantity_id: str
    native_unit: str
    lower: Decimal | None
    upper: Decimal | None

    def __post_init__(self) -> None:
        validate_stable_id(self.bound_id, field_name="bound_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")
        validate_nonempty(self.native_unit, field_name="native_unit")
        if self.lower is None and self.upper is None:
            raise ValueError("a quantity bound requires a lower or upper value")
        if self.lower is not None:
            validate_decimal(self.lower, field_name="lower")
        if self.upper is not None:
            validate_decimal(self.upper, field_name="upper")
        if self.lower is not None and self.upper is not None:
            if self.lower > self.upper:
                raise ValueError("quantity-bound lower value exceeds upper value")


@dataclass(frozen=True, slots=True)
class EvaluationRequest(CanonicalRecord):
    """Pure law-evaluation request passed to a registered implementation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/evaluation-request'

    reference_id: str
    relation_id: str
    horizon_id: str
    values: tuple[NamedDecimal, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("reference_id", self.reference_id),
            ("relation_id", self.relation_id),
            ("horizon_id", self.horizon_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(self.values, attribute="value_id", field_name="values")


@dataclass(frozen=True, slots=True)
class EvaluationResult(CanonicalRecord):
    """Pure registered-evaluator output without executable serialization."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/evaluation-result'

    reference_id: str
    values: tuple[NamedDecimal, ...]
    support_passed: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.reference_id, field_name="reference_id")
        require_sorted_unique_ids(self.values, attribute="value_id", field_name="values")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.support_passed and self.reason_codes:
            raise ValueError("a supported evaluation cannot carry failure reasons")


class ControllerDecisionKind(StrEnum):
    ACTION = "ACTION"
    HOLD = "HOLD"


class LawEvaluator(Protocol):
    """Runtime port implemented only by a statically registered capability."""

    reference: ExecutableReference

    def evaluate(self, request: EvaluationRequest) -> EvaluationResult: ...


def validate_named_values_against_units(
    values: tuple[NamedDecimal, ...], expected_units: Mapping[str, str]
) -> None:
    """Require exact native-unit identity at an evaluator boundary."""

    observed = {value.value_id: value.unit for value in values}
    if observed != dict(expected_units):
        raise ValueError("named evaluator values differ from expected native units")

"""Canonical empirical payload and strict decoder for the existing owner seam."""

from dataclasses import dataclass
from decimal import Decimal
import json
from typing import ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from .serialization import read_fit


@dataclass(frozen=True, slots=True)
class CalibratedReactorResponsePayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-response/calibrated-reactor-response-payload'
    model_json: str
    policy_sha256: str
    calibration_sha256: str
    q: Decimal | None
    horizon_s: Decimal = Decimal(10)
    receivers: tuple[str, ...] = ("reactor-peak-temperature", "reactor-end-conversion")

    def __post_init__(self) -> None:
        validate_sha256(self.policy_sha256, field_name="policy_sha256")
        validate_sha256(self.calibration_sha256, field_name="calibration_sha256")
        if len(self.model_json.encode()) > 1024**2:
            raise ValueError("model payload exceeds bound")
        read_fit(json.loads(self.model_json))
        if self.horizon_s != 10 or self.receivers != (
            "reactor-peak-temperature",
            "reactor-end-conversion",
        ):
            raise ValueError("empirical receiver/horizon differs")
        if self.q is not None and (not self.q.is_finite() or self.q < 0 or self.q > 1):
            raise ValueError("unusable calibration cannot be an empirical payload")


@dataclass(frozen=True, slots=True)
class CalibratedReactorResponseDecoder:
    extension_namespace: str = "tbs-reactor-empirical-response"
    decoder_schema: str = 'empirical-lawhood/methods/reactor-causal-response/calibrated-response-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = CalibratedReactorResponsePayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, CalibratedReactorResponsePayload, maximum_bytes=min(maximum_bytes, 2 * 1024**2)
        )


@dataclass(frozen=True, slots=True)
class DevelopmentBoundReactorResponsePayload(CalibratedReactorResponsePayload):
    """Bind the pre-calibration consumer separately from deployed calibrated q."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/reactor-response/development-bound-reactor-response-payload'
    VERSION: ClassVar[str] = '1.0.0'
    development_q: Decimal | None = None

    def __post_init__(self) -> None:
        CalibratedReactorResponsePayload.__post_init__(self)
        if (
            self.development_q is None
            or not self.development_q.is_finite()
            or not 0 <= self.development_q <= 1
        ):
            raise ValueError("frozen nominee development q is required")


@dataclass(frozen=True, slots=True)
class DevelopmentBoundReactorResponseDecoder:
    extension_namespace: str = "tbs-reactor-empirical-response"
    decoder_schema: str = 'empirical-lawhood/methods/reactor-causal-response/development-bound-response-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = DevelopmentBoundReactorResponsePayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, DevelopmentBoundReactorResponsePayload, maximum_bytes=min(maximum_bytes, 2 * 1024**2)
        )

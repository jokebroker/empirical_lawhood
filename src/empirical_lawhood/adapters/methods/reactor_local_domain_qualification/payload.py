"""One calibrated local receiver payload; collection membership grants no support."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from .records import LocalDomain


@dataclass(frozen=True, slots=True)
class LocalPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-local-domain-qualification/local-payload'
    domain: LocalDomain
    receiver: int
    design_sha256: str
    calibration_sha256: str
    halfwidth: D | None

    def __post_init__(self) -> None:
        validate_sha256(self.design_sha256, field_name="design_sha256")
        validate_sha256(self.calibration_sha256, field_name="calibration_sha256")
        if self.receiver not in self.domain.nominated_receivers or (
            self.halfwidth is not None
            and (
                not self.halfwidth.is_finite()
                or not 0 <= self.halfwidth <= (D(".26"), D(".0102"))[self.receiver]
            )
        ):
            raise ValueError("uncalibrated receiver or excessive local interval")


@dataclass(frozen=True, slots=True)
class LocalDecoder:
    extension_namespace: str = "tbs-reactor-local-response"
    decoder_schema: str = 'empirical-lawhood/methods/reactor-local-domain-qualification/response-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = LocalPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, LocalPayload, maximum_bytes=min(maximum_bytes, 65536)
        )

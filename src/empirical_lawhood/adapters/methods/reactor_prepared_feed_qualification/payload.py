"""Joint temperature/cooling payload with separate calibrated error envelopes."""

from dataclasses import dataclass
from decimal import Decimal as D
from typing import ClassVar
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from .records import FeedDomain


@dataclass(frozen=True, slots=True)
class FeedPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/feed-payload'
    domain: FeedDomain
    design_sha256: str
    calibration_sha256: str
    halfwidths: tuple[D | None, D | None]
    coefficients: tuple[D | None, D | None]

    def __post_init__(self) -> None:
        validate_sha256(self.design_sha256, field_name="design_sha256")
        validate_sha256(self.calibration_sha256, field_name="calibration_sha256")
        if len(self.halfwidths) != 2 or len(self.coefficients) != 2:
            raise ValueError("joint feed payload requires both receiver envelopes")
        for r, (width, q) in enumerate(zip(self.halfwidths, self.coefficients, strict=True)):
            if (
                (width is None) != (q is None)
                or (q is not None and (not q.is_finite() or not 0 <= q <= 1))
                or (
                    width is not None
                    and (not width.is_finite() or not 0 <= width <= (D(".26"), D(".005051"))[r])
                )
            ):
                raise ValueError("uncalibrated or excessive feed envelope")


@dataclass(frozen=True, slots=True)
class FeedDecoder:
    extension_namespace: str = "tbs-reactor-feed-response"
    decoder_schema: str = 'empirical-lawhood/methods/reactor-prepared-feed-qualification/response-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = FeedPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, FeedPayload, maximum_bytes=min(maximum_bytes, 65536)
        )

"""One unchanged development bound and its fresh coverage-qualification parent."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.simulators.reactor_regime_response.config import OLD_PREPARED_DOMAIN_SHA256
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .discovery import FrontierBound, FrontierDevelopment
from .qualification import FrontierQualificationRow


@dataclass(frozen=True, slots=True)
class FrontierLawPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-finite-control-frontier/frontier-law-payload'
    bound: FrontierBound
    prepared_domain: FeedDomain
    development: ObjectIdentity
    qualification: ObjectIdentity

    def __post_init__(self) -> None:
        if (
            self.prepared_domain.fingerprint() != OLD_PREPARED_DOMAIN_SHA256
            or self.development.object_schema != FrontierDevelopment.SCHEMA
            or self.qualification.object_schema != FrontierQualificationRow.SCHEMA
        ):
            raise ValueError("finite law changed its exact development or qualification identity")

    @property
    def stem(self) -> str:
        return f"reactor-finite-control-frontier.{self.bound.coordinate.coordinate_id}"


@dataclass(frozen=True, slots=True)
class FrontierLawDecoder:
    extension_namespace: str = "reactor-finite-control-frontier-response"
    decoder_schema: str = 'empirical-lawhood/methods/reactor-finite-control-frontier/law-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = FrontierLawPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, FrontierLawPayload, maximum_bytes=min(maximum_bytes, 4 * 1024**2)
        )

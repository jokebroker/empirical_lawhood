"""One finite empirical interval, frozen before fresh evidence."""

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.simulators.reactor_regime_response.config import OLD_PREPARED_DOMAIN_SHA256
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .config import ClassicalDesign
from .records import ClassicalQualification


@dataclass(frozen=True, slots=True)
class ClassicalLawPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-selected-action-response/classical-law-payload'
    design: ClassicalDesign
    prepared_domain: FeedDomain
    qualification: ObjectIdentity

    def __post_init__(self) -> None:
        if (
            self.design != ClassicalDesign()
            or self.prepared_domain.fingerprint() != OLD_PREPARED_DOMAIN_SHA256
            or self.qualification.object_schema != ClassicalQualification.SCHEMA
        ):
            raise ValueError("classical law changed its finite bound or qualification")

    @property
    def route(self) -> str:
        return "prepared_t0"

    @property
    def nomination(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.design.config_id, self.design)

    @property
    def calibration(self) -> ObjectIdentity:
        # Generic finite-set calibration consumes fresh *coverage qualification*;
        # these roots do not select or refit the empirical interval.
        return self.qualification


@dataclass(frozen=True, slots=True)
class ClassicalLawDecoder:
    extension_namespace: str = "reactor-selected-action-response"
    decoder_schema: str = 'empirical-lawhood/methods/reactor-selected-action-response/law-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = ClassicalLawPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, ClassicalLawPayload, maximum_bytes=min(maximum_bytes, 4 * 1024**2)
        )

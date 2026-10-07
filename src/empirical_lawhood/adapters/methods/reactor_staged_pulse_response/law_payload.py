"Exact staged-pulse receiver compatibility maps; no diagnostic upper service bound."

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.methods.reactor_prepared_feed_qualification.records import FeedDomain
from empirical_lawhood.adapters.simulators.reactor_regime_response.config import OLD_PREPARED_DOMAIN_SHA256
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from .discovery import ClassicalNomination, ClassicalRecipe
from .qualification import ClassicalQualificationRow


@dataclass(frozen=True, slots=True)
class ClassicalLawPayload(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/reactor-staged-pulse-response/classical-law-payload'
    block: str
    recipe: ClassicalRecipe
    prepared_domain: FeedDomain
    nomination: ObjectIdentity
    qualification: ObjectIdentity
    receiver_map: str = "raw-interval-or-min-raw-q-exact-kelvin-point"

    def __post_init__(self) -> None:
        if (
            self.block not in ("base-menu-comparison", "expanded-menu-comparison", "staged-sequence-comparison")
            or self.prepared_domain.fingerprint() != OLD_PREPARED_DOMAIN_SHA256
            or self.nomination.object_schema != ClassicalNomination.SCHEMA
            or self.qualification.object_schema != ClassicalQualificationRow.SCHEMA
            or self.receiver_map != "raw-interval-or-min-raw-q-exact-kelvin-point"
        ):
            raise ValueError(
                "law payload changes the qualified history or raw/saturated receiver map"
            )

    @property
    def stem(self) -> str:
        return f"reactor-staged-pulse-response.{self.block}.{self.recipe.recipe_id}"


@dataclass(frozen=True, slots=True)
class ClassicalLawDecoder:
    extension_namespace: str = "reactor-staged-pulse-response-programme"
    decoder_schema: str = 'empirical-lawhood/methods/reactor-staged-pulse-response/law-decoder'
    decoder_version: str = "1.0.0"
    payload_schema: str = ClassicalLawPayload.SCHEMA

    def decode(self, payload: bytes, *, maximum_bytes: int) -> CanonicalRecord:
        return decode_canonical_bytes(
            payload, ClassicalLawPayload, maximum_bytes=min(maximum_bytes, 4 * 1024**2)
        )

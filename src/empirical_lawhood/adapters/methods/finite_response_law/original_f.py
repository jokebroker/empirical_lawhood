"""An exact original F operand under a distinct current record identity.

The source records retain their historical namespaces. The current record does
not manufacture a current calibration or qualification. Custody of the source
artifacts belongs to the caller's authenticated input service, not this module.
No filesystem access, fitting, quantile estimation or native work occurs here.
"""

from dataclasses import dataclass, field
from decimal import Decimal
import hashlib
import json
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    validate_nonempty,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.adapters.simulators.finite_response_law.instruments import REFERENCE_INSTRUMENT
from .assigned_prediction import AssignedPrediction
from .fitting import Array, Normalizer
from .law_payloads import (
    FiniteResponseLawResponseChart,
    MAXIMUM_LAW_BYTES,
    _array,
    _features,
    _lower_prediction,
    _lower_response,
)

SOURCE_BYTES_SCHEMA = "empirical-lawhood/methods/finite-response-law/original-source-bytes"
ORIGINAL_Q = Decimal("0.73599618025169766")
ORIGINAL_PAYLOAD_SCHEMA = "icf-yolo/cc1-finite-lawhood/lower-law-payload/v1"
ORIGINAL_NUMERICAL_SHA256 = "28e0fe0ec4a69bd6cee14fbb4b243602556a39eefe198cf80cd1fc6803a8d440"


@dataclass(frozen=True, slots=True)
class OriginalFiniteResponseObjectIdentity(CanonicalRecord):
    """Verbatim historical identity; it is not a current ObjectIdentity."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/original-object-identity"
    object_id: str
    source_schema: str
    source_version: str
    source_fingerprint: str

    def __post_init__(self) -> None:
        validate_stable_id(self.object_id, field_name="object_id")
        validate_nonempty(self.source_schema, field_name="source_schema")
        validate_semantic_version(self.source_version)
        validate_sha256(self.source_fingerprint, field_name="source_fingerprint")


ORIGINAL_CALIBRATION = OriginalFiniteResponseObjectIdentity(
    "cc1-finite-lawhood-v1.t1-cal.lower.calibration",
    "icf-yolo/cc1-finite-lawhood/boundary-calibration/v1",
    "1.0.0",
    "f94ee6a720915abf2f46f9f78001d64735a5d97f9f18087724e2b7f8ce033b8b",
)
ORIGINAL_INSTRUMENT = OriginalFiniteResponseObjectIdentity(
    "cc1-finite-lawhood-v1.baseline-x-force-hessian-backward-rate",
    "icf-yolo/cc1-finite-lawhood/reference-instrument/v1",
    "1.0.0",
    "033595d1d80436bc54845d5a216c0165f22b60bf1d04badb304273c2f072e999",
)

# Owned ArtifactIdentity records describe opaque original bytes; their payload
# schema is not a claim that the historical document has been rewritten.
_SOURCE_PINS = (
    ("original-payload", 37921, "ab880c96a5e4d4e5070e4e2268331ea542f5cc0bedf18da43cded949d2462b1b"),
    ("original-manifest", 3919, "9b3973e222515cc33fc97f216f03a81b3409f932ea4d1ce982e22eb491256cbb"),
    ("original-publication-commit", 1428, "c47baa951b195dc27512cdcfe917416ce5350b4e72e3c570fa5d5e06c2de00e1"),
    ("original-calibration-report", 17166, "7b9a6ec11681f003bbc6f713e7532f475bb032516722c6bc0cadd02ddd0137b1"),
    ("original-qualification-report", 522736, "717d5810a4aedb3d2fb8fa9595a618839dfad1b2ee2b679208926116f6126951"),
)


def original_f_source_identities(
    artifact_ids: tuple[str, ...],
) -> tuple[ArtifactIdentity, ...]:
    """Describe five exact original files with caller-selected custody IDs."""
    if len(artifact_ids) != 5 or len(set(artifact_ids)) != 5:
        raise ValueError("Original F requires five distinct source artifact IDs")
    return tuple(
        ArtifactIdentity(artifact_id, role, SOURCE_BYTES_SCHEMA, digest, "application/json", size)
        for artifact_id, (role, size, digest) in zip(artifact_ids, _SOURCE_PINS, strict=True)
    )


def _source_identities(identities: tuple[ArtifactIdentity, ...]) -> None:
    if len(identities) != 5 or len({item.artifact_id for item in identities}) != 5:
        raise ValueError("Original F source artifact census differs")
    for identity, (role, size, digest) in zip(identities, _SOURCE_PINS, strict=True):
        if (
            identity.role != role
            or identity.payload_schema != SOURCE_BYTES_SCHEMA
            or identity.media_type != "application/json"
            or identity.size_bytes != size
            or identity.sha256 != digest
            or identity.extensions
        ):
            raise ValueError("Original F source identity differs from the frozen crosswalk")


def authenticate_original_f_sources(
    identities: tuple[ArtifactIdentity, ...], source_bytes: tuple[bytes, ...]
) -> None:
    """Check bounded supplied bytes; store lookup and exact receipts stay external."""
    _source_identities(identities)
    if len(source_bytes) != 5:
        raise ValueError("Original F requires all five original source files")
    for identity, raw in zip(identities, source_bytes, strict=True):
        if (
            type(raw) is not bytes
            or len(raw) != identity.size_bytes
            or hashlib.sha256(raw).hexdigest() != identity.sha256
        ):
            raise ValueError("Original F source bytes differ from their exact identity")


@dataclass(frozen=True, slots=True)
class OriginalFiniteResponseLaw(CanonicalRecord):
    """Fixed original arithmetic and q; a new canonical operand, never a new law."""

    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/original-finite-response-law"
    payload_id: str
    original_payload: ArtifactIdentity
    original_manifest: ArtifactIdentity
    publication_commit: ArtifactIdentity
    calibration_report: ArtifactIdentity
    qualification_report: ArtifactIdentity
    original_calibration: OriginalFiniteResponseObjectIdentity
    original_instrument: OriginalFiniteResponseObjectIdentity
    q: Decimal
    center: tuple[Decimal, ...]
    scale: tuple[Decimal, ...]
    operator: tuple[Decimal, ...]
    base: tuple[Decimal, ...]
    multiplier: tuple[Decimal, ...]
    instrument: ObjectIdentity = field(default=REFERENCE_INSTRUMENT.identity, kw_only=True)
    chart: FiniteResponseLawResponseChart = field(default=FiniteResponseLawResponseChart(), kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.payload_id, field_name="payload_id")
        _source_identities(self.source_identities)
        if (
            self.original_calibration != ORIGINAL_CALIBRATION
            or self.original_instrument != ORIGINAL_INSTRUMENT
            or self.instrument != REFERENCE_INSTRUMENT.identity
            or self.chart != FiniteResponseLawResponseChart()
            or type(self.q) is not Decimal
            or self.q != ORIGINAL_Q
        ):
            raise ValueError("Original F changes its historical or current instrument crosswalk")
        _array(self.center, (24,))
        _array(self.scale, (24,), positive=True)
        _array(self.operator, (25, 32))
        _array(self.base, (4, 8), positive=True)
        _array(self.multiplier, (25, 1))
        numerical = {name: getattr(self, name) for name in ("q", "center", "scale", "operator", "base", "multiplier")}
        if hashlib.sha256(canonical_json_bytes(numerical)).hexdigest() != ORIGINAL_NUMERICAL_SHA256:
            raise ValueError("Original F numerical fields differ from the exact frozen payload")
        if len(self.canonical_bytes()) > MAXIMUM_LAW_BYTES:
            raise ValueError("Original F current operand exceeds its bounded decoder")

    @property
    def source_identities(self) -> tuple[ArtifactIdentity, ...]:
        return (
            self.original_payload,
            self.original_manifest,
            self.publication_commit,
            self.calibration_report,
            self.qualification_report,
        )

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.payload_id, self)

    def normalize(self, handoff: Array) -> Array:
        _features(handoff)
        return Normalizer(_array(self.center, (24,)), _array(self.scale, (24,))).apply(handoff)

    def response(self, normalized: Array) -> Array:
        return _lower_response(_array(self.operator, (25, 32)), normalized)

    def predict(self, handoff: Array) -> AssignedPrediction:
        return _lower_prediction(
            handoff,
            Normalizer(_array(self.center, (24,)), _array(self.scale, (24,))),
            _array(self.operator, (25, 32)),
            _array(self.base, (4, 8)),
            _array(self.multiplier, (25, 1)),
        )


def original_f_from_bytes(
    raw: bytes,
    *,
    payload_id: str,
    source_identities: tuple[ArtifactIdentity, ...],
) -> OriginalFiniteResponseLaw:
    """Extract only exact pinned original JSON, retaining Decimal spellings.

    Other four evidence artifacts must be authenticated by the caller (or by
    authenticate_original_f_sources) before this operand is admitted to custody.
    The source hash pins the entire original chart, coefficients and provenance.
    The current chart rebinds compatible native units/actions/clocks; its current
    plan identity does not replace the original qualification or calibration.
    """
    _source_identities(source_identities)
    payload = source_identities[0]
    if type(raw) is not bytes or len(raw) != payload.size_bytes or hashlib.sha256(raw).hexdigest() != payload.sha256:
        raise ValueError("Original F payload bytes differ from the exact frozen artifact")
    document = json.loads(raw)
    if (
        set(document) != {"schema", "value", "version"}
        or document["schema"] != ORIGINAL_PAYLOAD_SCHEMA
        or document["version"] != "1.0.0"
    ):
        raise ValueError("Original F source envelope differs")
    value = document["value"]
    names = ("q", "center", "scale", "operator", "base", "multiplier")
    numerical = {
        name: Decimal(value[name]["decimal"])
        if name == "q"
        else tuple(Decimal(item["decimal"]) for item in value[name])
        for name in names
    }
    # No legacy object is decoded as a current ObjectIdentity. Its pinned raw
    # bytes authenticate the historical metadata above without namespace aliases.
    return OriginalFiniteResponseLaw(
        payload_id,
        *source_identities,
        ORIGINAL_CALIBRATION,
        ORIGINAL_INSTRUMENT,
        **numerical,
    )

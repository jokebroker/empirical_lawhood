"""Exact original nomination transported to current calibration input artifacts.

The manifest retains its original bytes; the report maps the two fixed SKR24
family labels to the current owned name, and the coefficient JSON envelope
changes only its namespace, preserving its exact NPZ bytes. No fitting or old q
is promoted into the new calibration. Old source files remain independently
authenticated and distinct from the current operand's identity and custody.
"""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from .array_transport import decode_npz_transport, encode_npz_transport
from .method_records import COEFFICIENT_SCHEMA, MANIFEST_SCHEMA, REPORT_SCHEMA
from .original_f import SOURCE_BYTES_SCHEMA

ORIGINAL_COEFFICIENT_SCHEMA = "icf-yolo/cc1-finite-lawhood/frozen-development-coefficients/v1"
CURRENT_COEFFICIENT_BYTES = 763818
CURRENT_COEFFICIENT_SHA256 = "90e170a4cce3f1742131be23b7c9b259ad0c8ca500391957eec376c442128a74"
CURRENT_REPORT_BYTES = 32218
CURRENT_REPORT_SHA256 = "fe0e5582323f35a429e091560cff65ee04e89fb3eacb29434177ec8f7419c313"
NOMINATION_SOURCE_PINS = (
    ("original-nomination-report", 32160, "26da86c4a17fe79ea4637be8514881810f79e94344b6455d23c4d7f6ab9f03ac"),
    ("original-nomination-coefficients", 763803, "ef2d19c9dbad93904b2bc5908af3d10c26f9506b7a52d8f0bf4579013c3de072"),
    ("original-nomination-manifest", 227387, "f48176bafcdab1c23a9717f10256e8dd262fe31222f777cc7af12da7c03f3964"),
)


@dataclass(frozen=True, slots=True)
class CurrentNominationOperand(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/methods/finite-response-law/current-nomination-operand"
    nomination_id: str
    original_report: ArtifactIdentity
    original_coefficients: ArtifactIdentity
    original_manifest: ArtifactIdentity
    development_report: ArtifactIdentity
    coefficients: ArtifactIdentity
    development_manifest: ArtifactIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.nomination_id, field_name="nomination_id")
        for source, (role, size, digest) in zip(self.original_artifacts, NOMINATION_SOURCE_PINS, strict=True):
            if source.role != role or source.size_bytes != size or source.sha256 != digest or source.payload_schema != SOURCE_BYTES_SCHEMA or source.media_type != "application/json" or source.extensions:
                raise ValueError("Nomination original source provenance differs")
        for current, schema, role in zip(self.current_artifacts, (REPORT_SCHEMA, COEFFICIENT_SCHEMA, MANIFEST_SCHEMA), ("development-report", "coefficients", "development-manifest"), strict=True):
            if current.payload_schema != schema or current.role != role or current.extensions or current.media_type != "application/json" or not 0 < current.size_bytes <= 1024**2:
                raise ValueError("Nomination current input artifact contract differs")
        if (
            self.development_report.sha256 != CURRENT_REPORT_SHA256
            or self.development_report.size_bytes != CURRENT_REPORT_BYTES
            or self.development_manifest.sha256 != self.original_manifest.sha256
            or self.development_manifest.size_bytes != self.original_manifest.size_bytes
            or self.coefficients.size_bytes != CURRENT_COEFFICIENT_BYTES
            or self.coefficients.sha256 != CURRENT_COEFFICIENT_SHA256
            or len({item.artifact_id for item in (*self.original_artifacts, *self.current_artifacts)}) != 6
        ):
            raise ValueError("Nomination transport differs from the exact metadata crosswalk or aliases custody IDs")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.nomination_id, self)

    @property
    def original_artifacts(self) -> tuple[ArtifactIdentity, ...]:
        return self.original_report, self.original_coefficients, self.original_manifest

    @property
    def current_artifacts(self) -> tuple[ArtifactIdentity, ...]:
        return self.development_report, self.coefficients, self.development_manifest


def current_nomination_from_sources(
    nomination_id: str, sources: tuple[bytes, bytes, bytes]
) -> tuple[CurrentNominationOperand, tuple[bytes, bytes, bytes]]:
    """Authenticate all original files and transport coefficients without refitting."""
    if len(sources) != 3:
        raise ValueError("Nomination requires its report, coefficients and manifest")
    originals = []
    for raw, (role, size, digest) in zip(sources, NOMINATION_SOURCE_PINS, strict=True):
        if type(raw) is not bytes or len(raw) != size or sha256(raw).hexdigest() != digest:
            raise ValueError("Nomination original source bytes differ from their exact identity")
        originals.append(ArtifactIdentity(f"{nomination_id}.{role}", role, SOURCE_BYTES_SCHEMA, digest, "application/json", size))
    original_report, coefficient_source, manifest = sources
    report_document = json.loads(original_report)
    for recipe in ("recipe", "direct_recipe"):
        if report_document[recipe] != {"dimension": 24, "family": "SKR24", "gamma": 1.0, "ridge": 10.0}:
            raise ValueError("Original nomination changes its fixed SKR24 numerical recipe")
        report_document[recipe]["family"] = "SNAPSHOT_REFERENCE_SKETCH_AND_RATE"
    report = (json.dumps(report_document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n").encode()
    coefficients = encode_npz_transport(decode_npz_transport(coefficient_source, ORIGINAL_COEFFICIENT_SCHEMA), COEFFICIENT_SCHEMA)
    current_bytes = (report, coefficients, manifest)
    current = tuple(
        ArtifactIdentity(f"{nomination_id}.{role}", role, schema, sha256(raw).hexdigest(), "application/json", len(raw))
        for role, schema, raw in zip(("development-report", "coefficients", "development-manifest"), (REPORT_SCHEMA, COEFFICIENT_SCHEMA, MANIFEST_SCHEMA), current_bytes, strict=True)
    )
    return CurrentNominationOperand(nomination_id, *originals, *current), current_bytes


def authenticate_current_nomination(
    operand: CurrentNominationOperand,
    sources: tuple[bytes, bytes, bytes],
    current_bytes: tuple[bytes, bytes, bytes],
) -> None:
    """Authenticate both physical namespaces and their exact transport relation."""
    expected, expected_bytes = current_nomination_from_sources(operand.nomination_id, sources)
    if operand != expected or current_bytes != expected_bytes:
        raise ValueError("Current nomination changed its authenticated original coefficient transport")


def validate_nomination_transport_inputs(
    identities: tuple[ArtifactIdentity, ArtifactIdentity, ArtifactIdentity],
    current_bytes: tuple[bytes, bytes, bytes],
) -> None:
    """Validate issued current input bytes when the original manifest is selected.

    Current source/runtime/environment are independent authoring/runtime inputs.
    This check does not compare the original manifest to currently loaded code,
    and the ordinary frozen_development scientific decoder must still run.
    """
    pins = (
        (REPORT_SCHEMA, CURRENT_REPORT_BYTES, CURRENT_REPORT_SHA256),
        (COEFFICIENT_SCHEMA, CURRENT_COEFFICIENT_BYTES, CURRENT_COEFFICIENT_SHA256),
        (MANIFEST_SCHEMA, NOMINATION_SOURCE_PINS[2][1], NOMINATION_SOURCE_PINS[2][2]),
    )
    if len(identities) != 3 or len(current_bytes) != 3:
        raise ValueError("Current nomination transport input census differs")
    for identity, raw, (schema, size, digest) in zip(identities, current_bytes, pins, strict=True):
        if identity.payload_schema != schema or identity.size_bytes != size or identity.sha256 != digest or type(raw) is not bytes or len(raw) != size or sha256(raw).hexdigest() != digest:
            raise ValueError("Current nomination transport changes its exact source crosswalk")

"""Lossless bounded transport for repeated immutable controller evidence."""

import base64
from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar
import zlib
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256

MAX_ARCHIVED_RECORD_BYTES = 64 * 1024**2


@dataclass(frozen=True, slots=True)
class CanonicalRecordArchive(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/canonical-record-archive'
    subject: ObjectIdentity
    decoded_bytes: int
    compressed_sha256: str
    zlib_base64: str

    def __post_init__(self) -> None:
        validate_sha256(self.compressed_sha256, field_name="compressed_sha256")
        if (
            type(self.decoded_bytes) is not int
            or not 0 < self.decoded_bytes <= MAX_ARCHIVED_RECORD_BYTES
            or len(self.zlib_base64) > MAX_ARCHIVED_RECORD_BYTES
        ):
            raise ValueError("archived canonical record exceeds its explicit bound")

    @classmethod
    def pack(cls, object_id: str, record: CanonicalRecord) -> 'CanonicalRecordArchive':
        raw = record.canonical_bytes()
        compressed = zlib.compress(raw, level=6)
        return cls(
            ObjectIdentity.from_record(object_id, record),
            len(raw),
            sha256(compressed).hexdigest(),
            base64.b64encode(compressed).decode(),
        )

    def unpack(self) -> bytes:
        compressed = base64.b64decode(self.zlib_base64, validate=True)
        if sha256(compressed).hexdigest() != self.compressed_sha256:
            raise ValueError("archived transport digest differs")
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, self.decoded_bytes + 1)
        if (
            not decoder.eof
            or decoder.unconsumed_tail
            or decoder.unused_data
            or len(raw) != self.decoded_bytes
            or sha256(raw).hexdigest() != self.subject.object_fingerprint
        ):
            raise ValueError("archived canonical payload/size differs or has trailing data")
        return raw

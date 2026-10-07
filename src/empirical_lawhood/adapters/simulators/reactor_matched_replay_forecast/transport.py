"""Bounded lossless wire envelope; the full canonical scientific trace is retained."""

import base64
from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar
import zlib

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
from .trace import MAXIMUM_BATCH_BYTES, ReactorBatchTrace

CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class ReactorTraceEnvelope(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-matched-replay-forecast/reactor-trace-envelope'
    trace_sha256: str
    uncompressed_bytes: int
    chunks: tuple[str, ...]
    encoding: str = "canonical-json-zlib-base64"

    def __post_init__(self) -> None:
        validate_sha256(self.trace_sha256, field_name="trace_sha256")
        if (
            type(self.uncompressed_bytes) is not int
            or not 0 < self.uncompressed_bytes <= MAXIMUM_BATCH_BYTES
            or self.encoding != "canonical-json-zlib-base64"
        ):
            raise ValueError("trace envelope changes encoding or uncompressed bound")
        if not 1 <= len(self.chunks) <= 3072 or any(
            not isinstance(c, str) or not 0 < len(c) <= CHUNK_BYTES for c in self.chunks
        ):
            raise ValueError("trace envelope changes bounded chunk roster")
        if any(len(c) != CHUNK_BYTES for c in self.chunks[:-1]):
            raise ValueError("trace envelope has noncanonical chunk boundaries")

    @classmethod
    def pack(cls, trace: ReactorBatchTrace) -> 'ReactorTraceEnvelope':
        raw = trace.canonical_bytes()
        text = base64.b64encode(zlib.compress(raw, level=1)).decode("ascii")
        return cls(
            sha256(raw).hexdigest(),
            len(raw),
            tuple(text[i : i + CHUNK_BYTES] for i in range(0, len(text), CHUNK_BYTES)),
        )

    def unpack(self) -> ReactorBatchTrace:
        compressed = base64.b64decode("".join(self.chunks), validate=True)
        decoder = zlib.decompressobj()
        raw = decoder.decompress(compressed, self.uncompressed_bytes + 1)
        if (
            len(raw) != self.uncompressed_bytes
            or not decoder.eof
            or decoder.unused_data
            or decoder.unconsumed_tail
            or sha256(raw).hexdigest() != self.trace_sha256
        ):
            raise ValueError("trace envelope length, digest or compressed framing differs")
        return decode_canonical_bytes(raw, ReactorBatchTrace, maximum_bytes=MAXIMUM_BATCH_BYTES)

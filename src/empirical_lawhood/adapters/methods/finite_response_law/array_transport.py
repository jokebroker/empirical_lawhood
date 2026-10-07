"""Bounded canonical framing for unchanged NPZ bytes on the existing JSON route."""

import base64
from hashlib import sha256
import json

from empirical_lawhood.kernel.serialization import canonical_json_bytes

MAX_NPZ_BYTES = 700 * 1024


def encode_npz_transport(raw: bytes, schema: str) -> bytes:
    if not 0 < len(raw) <= MAX_NPZ_BYTES:
        raise ValueError("Finite response-law NPZ transport exceeds its bounded byte census")
    return canonical_json_bytes(
        {
            "schema": schema,
            "version": "1.0.0",
            "value": {
                "npz_base64": base64.b64encode(raw).decode("ascii"),
                "npz_bytes": len(raw),
                "npz_sha256": sha256(raw).hexdigest(),
            },
        }
    )


def decode_npz_transport(raw: bytes, schema: str) -> bytes:
    if not 0 < len(raw) <= 1024**2:
        raise ValueError("Finite response-law NPZ transport exceeds its bounded JSON census")
    document = json.loads(raw)
    if (
        set(document) != {"schema", "version", "value"}
        or document["schema"] != schema
        or document["version"] != "1.0.0"
    ):
        raise ValueError("Finite response-law NPZ transport schema or version differs")
    value = document["value"]
    if set(value) != {"npz_base64", "npz_bytes", "npz_sha256"}:
        raise ValueError("Finite response-law NPZ transport fields differ")
    decoded = base64.b64decode(value["npz_base64"], validate=True)
    if (
        len(decoded) != value["npz_bytes"]
        or sha256(decoded).hexdigest() != value["npz_sha256"]
        or encode_npz_transport(decoded, schema) != raw
    ):
        raise ValueError("Finite response-law NPZ transport changes its exact original bytes")
    return decoded

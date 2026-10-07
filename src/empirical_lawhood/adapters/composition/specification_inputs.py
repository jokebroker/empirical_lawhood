"""Validate caller-provided scientific specifications before authoring."""

# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from hashlib import file_digest, sha256
from pathlib import Path
import re

from empirical_lawhood._required_inputs import ExternalInputRequired


def require_specifications(
    specifications: tuple[tuple[str, str], ...], *, count: int
) -> None:
    """Check actual external bytes against the declared authoring identities.

    Paths are absolute caller inputs. No package or checkout document is
    silently substituted for a missing scientific specification.
    """
    if len(specifications) != count:
        raise ExternalInputRequired(
            f"authoring requires {count} explicit scientific specification file(s)"
        )
    seen: set[Path] = set()
    for index, (raw_path, expected) in enumerate(specifications):
        path = Path(raw_path)
        if not path.is_absolute():
            raise ValueError(f"specification {index} must use an absolute path")
        if re.fullmatch(r"[0-9a-f]{64}", expected) is None:
            raise ValueError(f"specification {index} requires a lowercase SHA-256")
        if not path.is_file():
            raise ExternalInputRequired(f"scientific specification file is missing: {path}")
        resolved = path.resolve()
        if resolved in seen:
            raise ValueError(f"scientific specification file is repeated: {path}")
        seen.add(resolved)
        with path.open("rb") as source:
            actual = file_digest(source, sha256).hexdigest()
        if actual != expected:
            raise ValueError(f"scientific specification SHA-256 differs: {path}")

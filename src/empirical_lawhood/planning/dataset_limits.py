"Dataset contract limits shared by codecs and projections.\n\nThese limits belong to the durable dataset record contract. They must not be\nraised in place after schema migration 0003 is released; a larger contract\nrequires a new record version and an additive catalog migration.\n"

from __future__ import annotations

from typing import Final

# One valid observation can contain 4,096 bounded provider-object identities.
# Sixty-four MiB conservatively covers their worst-case canonical JSON escaping,
# all other bounded record fields, and the canonical envelope without admitting
# scientific rows, arrays, or provider response bodies into the local catalog.
MAX_DATASET_RECORD_CANONICAL_BYTES: Final[int] = 64 * 1_024**2
# SQLite's durable INTEGER representation is signed 64-bit.  Enforcing the
# same ceiling in pure records prevents valid Python values becoming
# unprojectable or driver-dependent.
MAX_DATASET_INTEGER: Final[int] = 2**63 - 1


__all__ = ["MAX_DATASET_INTEGER", "MAX_DATASET_RECORD_CANONICAL_BYTES"]

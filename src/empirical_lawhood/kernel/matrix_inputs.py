"""Portable matrix input allocations and external array identities.

Numerical purposes, generators and complete root ordering are scientific input.
Operational paths and environment selection belong to the consuming API. These
records describe exposed development and never grant freshness or qualification.
"""

from dataclasses import dataclass, field
from math import prod
from typing import ClassVar

from .provenance import ObjectIdentity
from .references import ArtifactIdentity
from .serialization import CanonicalRecord, validate_sha256, validate_stable_id

MATRIX_NATIVE_PURPOSES = (
    "future-1", "future-1-bridge", "future-2", "future-2-bridge",
    "parent", "parent-bridge", "prefix", "prefix-bridge", "requests",
)
MATRIX_ARRAY_SCHEMA = "empirical-lawhood/kernel/matrix-array-bundle"
MAXIMUM_MATRIX_ARRAY_BYTES = 64 * 1024**2
_DTYPES = {"<f8": 8, "<c16": 16, "<i8": 8, "|b1": 1}


@dataclass(frozen=True, slots=True)
class MatrixPurposeSeed(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/kernel/matrix-purpose-seed"
    purpose_id: str
    seed: int
    generator: str = "PCG64"

    def __post_init__(self) -> None:
        passive = self.purpose_id == "passive-probes"
        if (
            self.purpose_id not in (*MATRIX_NATIVE_PURPOSES, "passive-probes")
            or type(self.seed) is not int
            or not 0 <= self.seed < 2 ** (256 if passive else 128)
            or self.generator != ("PCG64DXSM" if passive else "PCG64")
        ):
            raise ValueError("Matrix seed must bind a declared purpose, generator and full-width integer")

    @property
    def effective_seed(self) -> int:
        """The probe owner consumes the leading 128 bits of its full digest."""
        return self.seed >> 128 if self.purpose_id == "passive-probes" else self.seed


@dataclass(frozen=True, slots=True)
class MatrixRootAllocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/kernel/matrix-root-allocation"
    root_id: str
    cohort: str
    scientific_seeds: tuple[MatrixPurposeSeed, ...]
    split_role: str = field(default="DEVELOPMENT", kw_only=True)
    conditioning_seeds: tuple[MatrixPurposeSeed, ...] = field(default=(), kw_only=True)
    source_prefix: ObjectIdentity | None = field(default=None, kw_only=True)
    source_artifact: ArtifactIdentity | None = field(default=None, kw_only=True)
    source_receipt: ObjectIdentity | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.root_id, field_name="root_id")
        if len(self.root_id) > 128 or self.cohort not in ("q2", "cir1", "constructed") or self.split_role not in ("DEVELOPMENT", "CALIBRATION", "EVALUATION"):
            raise ValueError("Undeclared matrix prefix source kind")
        expected = MATRIX_NATIVE_PURPOSES + (() if self.cohort == "constructed" else ("passive-probes",))
        names = tuple(item.purpose_id for item in self.scientific_seeds)
        if names != tuple(sorted(expected)):
            raise ValueError("Matrix root needs its complete sorted independent purpose census")
        conditioning = tuple(item.purpose_id for item in self.conditioning_seeds)
        if conditioning != (("passive-probes",) if self.cohort == "constructed" else ()):
            raise ValueError("Only constructed roots bind a separate exposed fixed probe allocation")
        lineage = (self.source_prefix, self.source_artifact, self.source_receipt)
        if any(item is None for item in lineage) and any(item is not None for item in lineage):
            raise ValueError("An imported prefix requires its exact object, bytes and receipt identities")

    def seed_for(self, purpose: str) -> int:
        for seed in (*self.scientific_seeds, *self.conditioning_seeds):
            if seed.purpose_id == purpose:
                return seed.seed
        raise ValueError("Matrix root has no allocation for the requested scientific purpose")


@dataclass(frozen=True, slots=True)
class MatrixAllocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/kernel/matrix-allocation"
    allocation_id: str
    roots: tuple[MatrixRootAllocation, ...]
    bootstrap_seed: int | None = field(default=None, kw_only=True)
    exposure: str = field(default="EXPOSED_DEVELOPMENT_NONPROMOTABLE", kw_only=True)

    def __post_init__(self) -> None:
        validate_stable_id(self.allocation_id, field_name="allocation_id")
        if (
            not 0 < len(self.roots) <= 128
            or len({root.root_id for root in self.roots}) != len(self.roots)
            or self.exposure not in ("EXPOSED_DEVELOPMENT_NONPROMOTABLE", "PROPOSED_UNRUN")
            or self.bootstrap_seed is not None and (type(self.bootstrap_seed) is not int or not 0 <= self.bootstrap_seed < 2**128)
        ):
            raise ValueError("Matrix allocation root census, bootstrap or evidence ceiling differs")
        streams = [(seed.generator, seed.effective_seed) for root in self.roots for seed in root.scientific_seeds]
        if self.bootstrap_seed is not None:
            streams.append(("PCG64", self.bootstrap_seed))
        if len(set(streams)) != len(streams):
            raise ValueError("Independent root/purpose allocations collide; labels cannot supply independence")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.allocation_id, self)


@dataclass(frozen=True, slots=True)
class MatrixArrayMember(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/kernel/matrix-array-member"
    name: str
    dtype: str
    shape: tuple[int, ...]
    sha256: str
    finite_required: bool = True

    def __post_init__(self) -> None:
        validate_stable_id(self.name, field_name="array name")
        validate_sha256(self.sha256)
        if (
            self.dtype not in _DTYPES
            or len(self.shape) > 12
            or any(type(size) is not int or not 0 <= size <= 1000000 for size in self.shape)
            or self.size_bytes > MAXIMUM_MATRIX_ARRAY_BYTES
            or type(self.finite_required) is not bool
        ):
            raise ValueError("Matrix array member exceeds its closed dtype, axis or byte contract")

    @property
    def size_bytes(self) -> int:
        return prod(self.shape) * _DTYPES.get(self.dtype, MAXIMUM_MATRIX_ARRAY_BYTES + 1)


@dataclass(frozen=True, slots=True)
class SavedMatrixArrays(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/kernel/saved-matrix-arrays"
    operand_id: str
    allocation: ObjectIdentity
    array_artifact: ArtifactIdentity
    members: tuple[MatrixArrayMember, ...]
    producers: tuple[ObjectIdentity, ...]
    source_artifacts: tuple[ArtifactIdentity, ...] = ()
    source_receipts: tuple[ObjectIdentity, ...] = ()
    original_f: ObjectIdentity | None = None
    exposure: str = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"

    def __post_init__(self) -> None:
        validate_stable_id(self.operand_id, field_name="operand_id")
        names = tuple(member.name for member in self.members)
        if (
            self.allocation.object_schema != MatrixAllocation.SCHEMA
            or not 0 < len(self.members) <= 256
            or names != tuple(sorted(set(names)))
            or sum(member.size_bytes for member in self.members) > MAXIMUM_MATRIX_ARRAY_BYTES
            or self.array_artifact.payload_schema != MATRIX_ARRAY_SCHEMA
            or self.array_artifact.media_type != "application/x-npz"
            or not 0 < self.array_artifact.size_bytes <= MAXIMUM_MATRIX_ARRAY_BYTES
            or not self.producers
            or len(self.producers) > 256
            or len(self.source_artifacts) > 256
            or len(self.source_receipts) > 256
            or self.exposure != "EXPOSED_DEVELOPMENT_NONPROMOTABLE"
        ):
            raise ValueError("Saved matrix operand changes its array census or current lineage")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.operand_id, self)

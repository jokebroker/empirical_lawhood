"""Current complete, numeric-coordinate allocations for the geometry scan.

Scientific coordinates, rather than display labels, determine streams.  The
historical table remains preserved provenance and is never a new-run default.
"""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.simulators.six_matrix_response.contracts import SixMatrixResponseSixMatrixSourceConfig
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, canonical_json_bytes, validate_sha256, validate_stable_id

PUBLIC_GEOMETRY_MASTER_SEED = 706606
GEOMETRY_ALLOCATION_MAXIMUM_BYTES = 8 * 1024**2


def geometry_coordinates() -> tuple[tuple[int, ...], ...]:
    """All potential cells; only the selected member enters confirmation."""
    return tuple(sorted(
        (member, stage, x, y, history, seed, view)
        for member in range(6)
        for stage in range(3)
        for x in range(13 if stage == 0 else 9)
        for y in range(13 if stage == 0 else 9)
        if stage != 2 or (x in (1, 4, 7) and y in (1, 4, 7))
        for history in ((0, 2, 3) if stage == 0 else range(4))
        for seed in range(3 if stage == 0 else 4 if stage == 1 else 1)
        for view in (int(stage == 2),)
    ))


def _seed(master_seed: int, coordinate: tuple[int, ...]) -> str:
    return sha256(canonical_json_bytes(("matrix-geometry.current-pcg64dxsm-v1", master_seed, coordinate))).hexdigest()


@lru_cache(maxsize=1)
def _original_seeds() -> frozenset[str]:
    from .scientific_seed_inputs import anisotropic_scientific_seed_sha256
    return frozenset(anisotropic_scientific_seed_sha256(
        member_ordinal=c[0], stage_ordinal=c[1], alpha_x_index=c[2], alpha_y_index=c[3],
        history_ordinal=c[4], seed_index=c[5], view_ordinal=c[6],
    ) for c in geometry_coordinates())


@dataclass(frozen=True, slots=True)
class MatrixGeometrySeedCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-geometry/seed-cell"
    coordinate: tuple[int, ...]
    full_seed_sha256: str

    def __post_init__(self) -> None:
        if len(self.coordinate) != 7 or any(type(v) is not int for v in self.coordinate):
            raise ValueError("Geometry seed cell requires seven integer scientific coordinates")
        validate_sha256(self.full_seed_sha256, field_name="full_seed_sha256")


@dataclass(frozen=True, slots=True)
class MatrixGeometryAllocation(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/matrix-geometry/allocation"
    allocation_id: str
    cells: tuple[MatrixGeometrySeedCell, ...]
    prior_exposed_seed_sha256s: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.allocation_id, field_name="allocation_id")
        if tuple(cell.coordinate for cell in self.cells) != geometry_coordinates():
            raise ValueError("Geometry allocation omits, duplicates or changes the complete scientific census")
        seeds = tuple(cell.full_seed_sha256 for cell in self.cells)
        if len({seed[:32] for seed in seeds}) != len(seeds):
            raise ValueError("Geometry allocation reuses an independent numerical stream")
        if tuple(sorted(set(self.prior_exposed_seed_sha256s))) != self.prior_exposed_seed_sha256s:
            raise ValueError("Geometry prior exposure census must be sorted and unique")
        for value in self.prior_exposed_seed_sha256s:
            validate_sha256(value, field_name="prior_exposed_seed_sha256s")
        if {seed[:32] for seed in seeds} & {seed[:32] for seed in (_original_seeds() | set(self.prior_exposed_seed_sha256s))}:
            raise ValueError("Geometry allocation reuses historical or previously exposed numerical streams")

    @property
    def identity(self) -> ObjectIdentity:
        return ObjectIdentity.from_record(self.allocation_id, self)

    @property
    def exposed_example(self) -> bool:
        return any(cell.full_seed_sha256 == _seed(PUBLIC_GEOMETRY_MASTER_SEED, cell.coordinate) for cell in self.cells)


def allocate_matrix_geometry(*, allocation_id: str, master_seed: int,
                             prior_exposed_seed_sha256s: tuple[str, ...] = ()) -> MatrixGeometryAllocation:
    if type(master_seed) is not int or not 0 <= master_seed < 2**256:
        raise ValueError("Geometry master seed requires an unsigned 256-bit integer")
    return MatrixGeometryAllocation(allocation_id, tuple(MatrixGeometrySeedCell(c, _seed(master_seed, c))
        for c in geometry_coordinates()), prior_exposed_seed_sha256s)


def validate_geometry_source(source: SixMatrixResponseSixMatrixSourceConfig) -> None:
    """Lock the preset mathematics, permitting only explicit identifier edits."""
    from empirical_lawhood.adapters.composition.matrix_response_study.design import _source_config

    def numeric(value):
        if isinstance(value, CanonicalRecord):
            return numeric(value.to_document()["value"])
        if isinstance(value, dict):
            return {key: numeric(item) for key, item in value.items()
                if not (key.endswith("_id") or key.endswith("_ids") or key in ("config_version",))}
        if isinstance(value, (list, tuple)):
            return tuple(numeric(item) for item in value)
        return value

    # Order members by their mathematical values, independently of labels.
    from dataclasses import replace
    ordered = replace(source, anisotropic_model=replace(source.anisotropic_model,
        family_members=tuple(sorted(source.anisotropic_model.family_members, key=lambda m: m.member_id))))
    baseline = _source_config()
    if (source.feasibility_envelope.feasibility_history_ids != baseline.feasibility_envelope.feasibility_history_ids
        or source.feasibility_envelope.confirmation_history_ids != baseline.feasibility_envelope.confirmation_history_ids):
        raise ValueError("Geometry source changes the declared preparation history equations")
    if numeric(ordered) != numeric(baseline):
        # Numeric member order can differ after renaming. Compare its set separately.
        def rest(config):
            doc = numeric(config)
            members = doc["anisotropic_model"]["value"]["family_members"]
            doc["anisotropic_model"]["value"]["family_members"] = tuple(sorted(members, key=canonical_json_bytes))
            return doc
        if rest(ordered) != rest(baseline):
            raise ValueError("Geometry source changes the locked six-member numerical preset")

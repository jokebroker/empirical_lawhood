"""Actual consumed allocations and a receipted prior-exposure projection."""

from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

from empirical_lawhood.kernel.matrix_inputs import MatrixAllocation
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, require_sorted_unique_strings


def effective_seed_ids(allocation: MatrixAllocation) -> tuple[str, ...]:
    ids = tuple(
        f"seed.{seed.generator.lower()}.{(seed.seed >> 128 if seed.generator == 'PCG64DXSM' else seed.seed):032x}"
        for root in allocation.roots for seed in root.scientific_seeds
    )
    if allocation.bootstrap_seed is not None:
        ids += (f"seed.pcg64.{allocation.bootstrap_seed:032x}",)
    if len(set(ids)) != len(ids):
        raise ValueError("applicability repeats an effective native allocation")
    return tuple(sorted(ids))


@dataclass(frozen=True, slots=True)
class PreparationApplicabilityExposure(CanonicalRecord):
    SCHEMA: ClassVar[str] = "empirical-lawhood/composition/preparation-applicability/exposure"
    inspection: ObjectIdentity
    custody_receipt: ObjectIdentity
    excluded_unit_ids: tuple[str, ...]
    excluded_seed_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_strings(self.excluded_unit_ids, field_name="excluded_unit_ids", allow_empty=False)
        require_sorted_unique_strings(self.excluded_seed_ids, field_name="excluded_seed_ids", allow_empty=False)

    def check(self, allocation: MatrixAllocation) -> None:
        public_units,public_seeds=public_historical_exposure()
        if set(root.root_id for root in allocation.roots) & set((*self.excluded_unit_ids,*public_units)):
            raise ValueError("applicability allocation reuses an exposed physical unit")
        if set(effective_seed_ids(allocation)) & set((*self.excluded_seed_ids,*public_seeds)):
            raise ValueError("applicability allocation reuses an exposed effective RNG seed")


@lru_cache(maxsize=1)
def public_historical_exposure():
    """Known exposed seed recipes; historical counts never gate a new result."""
    from hashlib import sha256
    from empirical_lawhood.kernel.matrix_inputs import MATRIX_NATIVE_PURPOSES
    units=[]; seeds=[]
    rule="cc1-applicability-v1.passive-probes"
    for prefix,phases,padded in (("cc1-applicability-r1-v1",(("D",32),("E",64)),True),
                                ("cc1-boundary-production-v1",(("Q",8),("E",32)),False)):
        for phase,count in phases:
            for index in range(count):
                units.append(f"{prefix}.{phase.lower()}.r{index:03d}")
                def digest(purpose):
                    number=f"{index:03d}" if padded else str(index)
                    return sha256(f"{prefix}:{phase}:{number}:{purpose}".encode()).hexdigest()
                for purpose in MATRIX_NATIVE_PURPOSES:
                    seeds.append(f"seed.pcg64.{digest(purpose)[:32]}")
                if padded:
                    probe=sha256(f"{rule}\0{digest('prefix-probes')}\0q2-n4-fields12".encode()).hexdigest()
                    seeds.append(f"seed.pcg64dxsm.{probe[:32]}")
            number="000" if padded else "0"
            bootstrap=sha256(f"{prefix}:{phase}:{number}:bootstrap".encode()).hexdigest()
            seeds.append(f"seed.pcg64.{bootstrap[:32]}")
    # The constructed chart deliberately reuses this exposed conditioning field.
    # It is separate from fresh scientific innovations and never counts as fresh.
    seeds.append("seed.pcg64dxsm.138298f06df15e5a8b7b6e15eab582a30")
    # Current public editable samples are also exposed numerical operands.
    # Their fixed integers stay reserved if users change IDs, roles or labels.
    for label,count in (("preparation",24),("conformance",3)):
        for index in range(count):
            units.append(f"example.exposed.{label}.root-{index:03d}")
            for purpose in MATRIX_NATIVE_PURPOSES:
                digest=sha256(f"public-exposed-matrix-input|{label}|{index}|{purpose}".encode()).hexdigest()
                seeds.append(f"seed.pcg64.{digest[:32]}")
            if label!="conformance" or index!=2:
                probe=sha256(f"public-exposed-matrix-probe|{label}|{index}".encode()).hexdigest()
                seeds.append(f"seed.pcg64dxsm.{probe[:32]}")
    # These owners expose immutable seed recipes, not learned fit machinery.
    from empirical_lawhood.adapters.composition.finite_response_law.rerun_input import original_development_exposure,public_rerun_sample_exposure
    for known in (original_development_exposure(),public_rerun_sample_exposure()):
        units.extend(known[0]);seeds.extend(known[1])
    return tuple(sorted(set(units))),tuple(sorted(set(seeds)))

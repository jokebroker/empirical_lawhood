"Typed, outcome-blind calibration cohort reservation before any native binding.\n\nThe reservation is an external input. It does not make the retained fixed-roster\nnative source executable under these names or grant issue authority.\n"

from dataclasses import dataclass
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.adapters.methods.finite_response_law.science import FiniteResponseLawScienceSpec, validate_assigned_seeds
from empirical_lawhood.adapters.simulators.finite_response_law.randomness import SUBSTREAMS, native_rng
from empirical_lawhood.adapters.simulators.six_matrix_response.passive_probe import probe_roster_seed_sha256
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_sha256,
    validate_stable_id,
)

_COUNTS = {"calibration": 32, "prospective-evaluation": 64}
_ROLE = "EXPOSED_DEVELOPMENT_NONPROMOTABLE"


def proposed_scientific_seeds(stage: str, master_seed: int) -> tuple[tuple[str, int, int], ...]:
    """Materialize a caller-proposed seed census from an explicit numeric input.

    The caller must review and reserve the returned census before outcomes. It
    defines a new allocation, with no inherited source qualification or grant.
    The cohort namespace does not change the allocation because it is not an input.
    The stage and purpose strings are inputs to this new allocation.
    For preservation of an existing allocation, supply its reviewed integers
    directly instead of generating a new census.
    """
    if stage not in _COUNTS or type(master_seed) is not int or not 0 <= master_seed < 2**128:
        raise ValueError("proposed scientific census requires a declared stage and 128-bit master seed")
    slots = [(p, i) for i in range(_COUNTS[stage]) for p in ("prefix", "parent", "future-1", "future-2", "parent-allocation", f"{stage}-request", "passive-probes")]
    if stage == "prospective-evaluation":
        slots.append(("bootstrap", -1))
    rows = []
    for purpose, index in sorted(slots):
        digest = sha256(f"finite-response-law-scientific-seed-census|{master_seed}|{stage}|{purpose}|{index}".encode()).digest()
        seed = int.from_bytes(digest if purpose == "passive-probes" else digest[:16], "big")
        rows.append((purpose, index, seed))
    result = tuple(rows)
    validate_assigned_seeds(stage, result)
    return result


@dataclass(frozen=True, slots=True)
class FiniteResponseLawCohortAssignment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/finite-response-law/finite-response-law-cohort-assignment'
    stage: str
    cohort_namespace: str
    science_plan_sha256: str
    root_count: int
    evidence_role: str
    scientific_seeds: tuple[tuple[str, int, int], ...]

    def __post_init__(self) -> None:
        validate_cohort_assignment(self, allowed_roles=(_ROLE,))

    @property
    def stage_units(self) -> tuple[str, ...]:
        return tuple(
            f"{self.cohort_namespace}.r{i:03d}" for i in range(self.root_count)
        )

    @property
    def physical_unit_ids(self) -> tuple[str, ...]:
        return tuple(
            f"{unit}.prepared.seed.{sha256(str(self.seed_for('prefix', i)).encode()).hexdigest()}"
            for i, unit in enumerate(self.stage_units)
        )

    def seed_for(self, purpose: str, index: int = -1) -> int:
        for p, i, seed in self.scientific_seeds:
            if (p, i) == (purpose, index):
                return seed
        raise ValueError("assigned scientific seed purpose/index is absent from its census")


def validate_cohort_assignment(
    assignment: FiniteResponseLawCohortAssignment, *, allowed_roles: tuple[str, ...]
) -> None:
    """Shared numeric census policy; a role label conveys no eligibility."""
    validate_assigned_seeds(assignment.stage, assignment.scientific_seeds)
    validate_stable_id(assignment.cohort_namespace, field_name="cohort_namespace")
    validate_sha256(assignment.science_plan_sha256, field_name="science_plan_sha256")
    if (
        assignment.stage not in _COUNTS
        or type(assignment.root_count) is not int
        or assignment.root_count != _COUNTS.get(assignment.stage)
        or assignment.science_plan_sha256 != FiniteResponseLawScienceSpec().plan_sha256
        or not assignment.cohort_namespace.startswith(
            f"empirical-lawhood.finite-response-law.{assignment.stage}."
        )
        or ".r" in assignment.cohort_namespace
        or len(assignment.cohort_namespace) > 128
        or assignment.cohort_namespace.endswith(".")
        or assignment.evidence_role not in allowed_roles
    ):
        raise ValueError(
            "FINITE_RESPONSE_LAW_ASSIGNMENT_INVALID: stage, namespace, plan, count or role"
        )


def assigned_native_seed_ids(
    assignment: FiniteResponseLawCohortAssignment,
) -> tuple[str, ...]:
    "Reserve every existing calibration cohort purpose, jumped state and passive probe seed."

    seeds: set[str] = set()
    if assignment.stage == "prospective-evaluation":
        seeds.add(
            f"seed.pcg64.{assignment.seed_for('bootstrap'):032x}"
        )
    for index, unit in enumerate(assignment.stage_units):
        for purpose in ("parent-allocation", f"{assignment.stage}-request"):
            seeds.add(f"seed.pcg64.{assignment.seed_for(purpose, index):032x}")
        for purpose in ("prefix", "parent", "future-1", "future-2"):
            for substream in SUBSTREAMS if purpose == "prefix" else SUBSTREAMS[:2]:
                record = native_rng(unit, purpose, substream, committed_seed=assignment.seed_for(purpose, index))
                seeds.add(f"seed.pcg64.{int(record.committed_seed_decimal):032x}")
                seeds.add(f"seed-state.pcg64.{record.initial_state_sha256}")
                if substream == "passive-probes":
                    probe = probe_roster_seed_sha256(
                        config_fingerprint=record.initial_state_sha256,
                        rule_id="finite-response-law.passive-probes",
                        scientific_seed=assignment.seed_for("passive-probes", index),
                    )
                    seeds.add(f"seed.pcg64dxsm.{probe[:32]}")
    return tuple(sorted(seeds))


__all__ = ['FiniteResponseLawCohortAssignment', "assigned_native_seed_ids", "proposed_scientific_seeds"]

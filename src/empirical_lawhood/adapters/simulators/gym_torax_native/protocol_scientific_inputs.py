"""Caller-bound numerical seed inputs for matched and recurrence protocols.

These protocols have distinct seed arithmetic. Neither imports entropy from a
public physical-unit name or supplies an inferred historical seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    validate_sha256,
    validate_stable_id,
)


@dataclass(frozen=True, slots=True)
class GymToraxProtocolEnvironmentSeedInput(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        "empirical-lawhood/simulators/gym-torax-native/protocol-scientific-environment-seed"
    )
    physical_unit_id: str
    scientific_role: str
    full_seed_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.physical_unit_id, field_name="physical_unit_id")
        validate_sha256(self.full_seed_sha256, field_name="full_seed_sha256")
        if self.scientific_role not in (
            "matched-evaluation",
            "recurrence-main",
            "recurrence-route",
            "hold-calibration",
            "recurrence-recovery",
        ):
            raise ValueError("protocol seed has another scientific role")

    @property
    def environment_seed(self) -> int:
        word = int(self.full_seed_sha256[:16], 16)
        if self.scientific_role == "recurrence-recovery":
            return word & (2**63 - 1)
        if self.scientific_role == "matched-evaluation":
            return (word & (2**63 - 1)) or 1
        return word % (2**63 - 1) + 1

    def require_root(self, physical_unit_id: str, scientific_role: str) -> None:
        if (
            self.physical_unit_id != physical_unit_id
            or self.scientific_role != scientific_role
        ):
            raise ValueError(
                "protocol scientific seed input binds another physical root or role"
            )


def require_protocol_seed_census(
    inputs: tuple[GymToraxProtocolEnvironmentSeedInput, ...] | None,
    *,
    matched_evaluation: bool,
    recurrence_recovery: bool = False,
) -> dict[str, GymToraxProtocolEnvironmentSeedInput]:
    if (
        not isinstance(inputs, tuple)
        or not inputs
        or any(
            not isinstance(seed, GymToraxProtocolEnvironmentSeedInput)
            for seed in inputs
        )
    ):
        raise ValueError(
            "protocol requires a complete typed numeric seed census; original name-derived inputs need an explicit external export"
        )
    identities = tuple(seed.physical_unit_id for seed in inputs)
    if identities != tuple(sorted(set(identities))):
        raise ValueError(
            "protocol scientific seed inputs must be ordered and unique by physical root"
        )
    if matched_evaluation and recurrence_recovery:
        raise ValueError("matched and recovery seed rules are distinct")
    allowed_roles = (
        {"recurrence-recovery"} if recurrence_recovery
        else {"matched-evaluation"} if matched_evaluation
        else {"recurrence-main", "recurrence-route", "hold-calibration"}
    )
    if any(seed.scientific_role not in allowed_roles for seed in inputs):
        raise ValueError("protocol scientific seed census mixes distinct seed rules")
    if matched_evaluation and len(inputs) != 72:
        raise ValueError(
            "matched evaluation requires exactly seventy-two scientific seed roots"
        )
    if recurrence_recovery and len(inputs) != 22:
        raise ValueError("recurrence recovery requires exactly twenty-two scientific seed roots")
    return {seed.physical_unit_id: seed for seed in inputs}


def require_protocol_root_seed(
    inputs: dict[str, GymToraxProtocolEnvironmentSeedInput],
    physical_unit_id: str,
    scientific_role: str,
) -> GymToraxProtocolEnvironmentSeedInput:
    seed = inputs.get(physical_unit_id)
    if seed is None:
        raise ValueError(
            f"missing numeric scientific seed for {physical_unit_id}; supply the complete root census"
        )
    seed.require_root(physical_unit_id, scientific_role)
    return seed

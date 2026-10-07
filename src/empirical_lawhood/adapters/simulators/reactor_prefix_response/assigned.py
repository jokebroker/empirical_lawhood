# SPDX-License-Identifier: MPL-2.0

"Prefix assignment with five fresh noise seeds, paired arms and nested views.\n\nThe public panel remains exposed. This assignment preserves its five declared\nscenario conditions and command/receiver recipe while passing fresh seeds to\nthe native plant. No function here draws a roster or grants execution authority.\n"

from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)

from .panel import ARMS, CALIBRATION, HELDOUT, SCENARIOS, VIEWS, ReactorPrefixConfig, ReactorPrefixEpisode


@dataclass(frozen=True, slots=True)
class ReactorPrefixAssignedUnit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-prefix-assigned-unit'
    unit_id: str
    public_scenario: str
    noise_seed: int

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id)
        if (
            self.public_scenario not in SCENARIOS
            or self.unit_id in SCENARIOS
            or type(self.noise_seed) is not int
            or not 0 < self.noise_seed < 2**63
        ):
            raise ValueError("REACTOR_ASSIGNED_UNIT_INVALID")

    @property
    def seed_id(self) -> str:
        return f"seed.native-reactor.{self.noise_seed}"


@dataclass(frozen=True, slots=True)
class ReactorPrefixPriorCensus(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-prefix-prior-census'
    census_id: str
    prior_unit_ids: tuple[str, ...]
    prior_seed_ids: tuple[str, ...]
    source_inventory: ObjectIdentity

    def __post_init__(self) -> None:
        validate_stable_id(self.census_id)
        for name in ("prior_unit_ids", "prior_seed_ids"):
            require_sorted_unique_strings(
                getattr(self, name), field_name=name, allow_empty=False
            )

    @property
    def authoring_seed_ids(self) -> tuple[str, ...]:
        """Project exclusions onto the native seed domain without changing the ledger.

        Complete source-purpose/PCG/state IDs remain present. A numeric alias in
        the native-reactor namespace cannot denote a valid assigned prefix seed
        when it lies outside (0, 2**63). Those impossible aliases add no collision
        protection and need not be duplicated in the bounded issue document.
        The exact full census remains separately fingerprinted and authenticated.
        """
        prefix = "seed.native-reactor."
        return tuple(
            value
            for value in self.prior_seed_ids
            if not (
                value.startswith(prefix)
                and value[len(prefix) :].isdecimal()
                and not 0 < int(value[len(prefix) :]) < 2**63
            )
        )


@dataclass(frozen=True, slots=True)
class ReactorPrefixAssignment(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-prefix-assignment'
    assignment_id: str
    units: tuple[ReactorPrefixAssignedUnit, ...]
    prior_census: ObjectIdentity
    evidence_role: str

    def __post_init__(self) -> None:
        validate_stable_id(self.assignment_id)
        if (
            tuple(unit.public_scenario for unit in self.units) != SCENARIOS
            or len({unit.unit_id for unit in self.units}) != 5
            or len({unit.noise_seed for unit in self.units}) != 5
            or self.evidence_role
            not in (
                "PROSPECTIVE_RELEASE_QUALIFICATION",
                "EXPOSED_DEVELOPMENT_NONPROMOTABLE",
            )
            or any(
                unit.unit_id
                != f"{self.assignment_id}.{unit.public_scenario.replace('_', '-')}"
                for unit in self.units
            )
        ):
            raise ValueError("REACTOR_ASSIGNMENT_ROSTER_OR_EXPOSURE_COLLISION")

    def check_prior_census(self, census: ReactorPrefixPriorCensus) -> None:
        """Authenticate complete pre-contact exclusions without copying them into panels."""
        if self.prior_census != ObjectIdentity.from_record(census.census_id, census):
            raise ValueError("REACTOR_ASSIGNMENT_CENSUS_IDENTITY_MISMATCH")
        if {
            identity
            for unit in self.units
            for identity in (unit.unit_id, f"unit.{unit.unit_id}")
        } & set(census.prior_unit_ids) or {
            identity
            for unit in self.units
            for identity in (str(unit.noise_seed), unit.seed_id)
        } & set(census.prior_seed_ids):
            raise ValueError("REACTOR_ASSIGNMENT_ROSTER_OR_EXPOSURE_COLLISION")

    @property
    def independent_unit_ids(self) -> tuple[str, ...]:
        return tuple(sorted(f"unit.{unit.unit_id}" for unit in self.units))

    @property
    def seed_ids(self) -> tuple[str, ...]:
        return tuple(sorted(unit.seed_id for unit in self.units))


@dataclass(frozen=True, slots=True)
class ReactorAssignedPrefixConfig(ReactorPrefixConfig):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-assigned-prefix-config'
    VERSION: ClassVar[str] = '1.0.0'
    assignment: ReactorPrefixAssignment

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id)
        if (
            self.calibration_scenarios
            != tuple(
                unit.unit_id
                for unit in self.assignment.units
                if unit.public_scenario in CALIBRATION
            )
            or self.heldout_scenarios
            != tuple(
                unit.unit_id
                for unit in self.assignment.units
                if unit.public_scenario in HELDOUT
            )
            or self.horizon_s != Decimal(20)
            or self.numpy_version != "2.4.6"
            or self.python_version != "3.11.14"
        ):
            raise ValueError("REACTOR_ASSIGNED_PREFIX_RECIPE_MISMATCH")

    @property
    def branches(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(
            (unit.unit_id, view, arm)
            for unit in self.assignment.units
            for view, _ in VIEWS
            for arm, _ in ARMS
        )


def assigned_prefix_config(
    assignment: ReactorPrefixAssignment,
) -> ReactorAssignedPrefixConfig:
    return ReactorAssignedPrefixConfig(
        f"{assignment.assignment_id}.native-config",
        tuple(
            unit.unit_id
            for unit in assignment.units
            if unit.public_scenario in CALIBRATION
        ),
        tuple(
            unit.unit_id for unit in assignment.units if unit.public_scenario in HELDOUT
        ),
        Decimal(20),
        "2.4.6",
        "3.11.14",
        assignment,
    )


@dataclass(frozen=True, slots=True)
class ReactorAssignedPrefixEpisode(ReactorPrefixEpisode):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-assigned-prefix-episode'
    VERSION: ClassVar[str] = '1.0.0'
    unit: ReactorPrefixAssignedUnit

    def __post_init__(self) -> None:
        from .panel import validate_prefix_episode

        if self.scenario_id != self.unit.unit_id:
            raise ValueError("REACTOR_ASSIGNED_EPISODE_UNIT_MISMATCH")
        validate_prefix_episode(self, (self.unit.unit_id,))


@dataclass(frozen=True, slots=True)
class ReactorAssignedPrefixPanel(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/reactor-prefix-response/reactor-assigned-prefix-panel'
    VERSION: ClassVar[str] = '1.0.0'
    config: ReactorAssignedPrefixConfig
    episodes: tuple[ReactorAssignedPrefixEpisode, ...]
    branch: tuple[str, str, str] | None = None

    def __post_init__(self) -> None:
        expected = self.config.branches if self.branch is None else (self.branch,)
        if (
            self.branch is not None
            and self.branch not in self.config.branches
            or tuple((e.scenario_id, e.view_id, e.arm_id) for e in self.episodes)
            != expected
            or any(e.unit not in self.config.assignment.units for e in self.episodes)
        ):
            raise ValueError("REACTOR_ASSIGNED_PANEL_CENSUS_MISMATCH")


class AssignedPrefixBridge:
    """Construct only the installed native bridge, with a typed actual seed."""

    @staticmethod
    def create(*, unit: ReactorPrefixAssignedUnit, **kwargs):
        from .bridge import ReactorBridge

        bridge = ReactorBridge(scenario_id=unit.public_scenario, **kwargs)
        bridge._scenario = {**bridge._scenario, "seed": unit.noise_seed}
        return bridge

"""Evidence worlds, numerical views and computability boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from .evidence import ClaimSpec, EvidenceCeiling, OutcomeAccess
from .serialization import (
    CanonicalRecord,
    ExtensionBinding,
    require_extensions,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_nonempty,
    validate_stable_id,
)
from .status import ReadinessStatus


class WorldKind(StrEnum):
    ANALYTIC_REFERENCE = "ANALYTIC_REFERENCE"
    NUMERICAL_SIMULATOR = "NUMERICAL_SIMULATOR"
    HARDWARE_IN_LOOP = "HARDWARE_IN_LOOP"
    PHYSICAL_EXPERIMENT = "PHYSICAL_EXPERIMENT"


class EvidenceUnitScope(StrEnum):
    PHYSICAL_INDEPENDENT_UNIT = "PHYSICAL_INDEPENDENT_UNIT"
    NESTED_NUMERICAL_VIEW = "NESTED_NUMERICAL_VIEW"


class RandomnessSemantics(StrEnum):
    DETERMINISTIC = "DETERMINISTIC"
    NUMERICAL_REPEAT = "NUMERICAL_REPEAT"
    STOCHASTIC_MODEL_DRAW = "STOCHASTIC_MODEL_DRAW"
    GENERATIVE_PREPARATION = "GENERATIVE_PREPARATION"


class NumericalCoordinateKind(StrEnum):
    SPATIAL_GRID = "SPATIAL_GRID"
    TIMESTEP = "TIMESTEP"
    SOLVER_REFINEMENT = "SOLVER_REFINEMENT"
    PRECISION = "PRECISION"
    MONTE_CARLO_SAMPLES = "MONTE_CARLO_SAMPLES"
    SURROGATE_FIDELITY = "SURROGATE_FIDELITY"


@dataclass(frozen=True, slots=True)
class WorldTransitionRule(CanonicalRecord):
    """Declared non-promoting route from one evidence world to another."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/world-transition-rule'

    target_kind: WorldKind
    evidence_ceiling: EvidenceCeiling
    requires_independent_validation: bool = True

    def __post_init__(self) -> None:
        if self.evidence_ceiling is EvidenceCeiling.NON_PROMOTABLE:
            return
        if not self.requires_independent_validation:
            raise ValueError("a promotable cross-world transition requires independent validation")


@dataclass(frozen=True, slots=True)
class WorldSpec(CanonicalRecord):
    """Evidence-bearing world in which a system is prepared and observed."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/world-spec'

    world_id: str
    label: str
    kind: WorldKind
    represented_physics: tuple[str, ...]
    unrepresented_physics: tuple[str, ...]
    privileged_truth_quantity_ids: tuple[str, ...]
    maximum_evidence: EvidenceCeiling
    available_outcome_access: frozenset[OutcomeAccess]
    transition_rules: tuple[WorldTransitionRule, ...] = ()
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.world_id, field_name="world_id")
        validate_nonempty(self.label, field_name="label")
        require_sorted_unique_strings(
            self.represented_physics,
            field_name="represented_physics",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.unrepresented_physics, field_name="unrepresented_physics"
        )
        require_sorted_unique_strings(
            self.privileged_truth_quantity_ids,
            field_name="privileged_truth_quantity_ids",
        )
        overlap = set(self.represented_physics) & set(self.unrepresented_physics)
        if overlap:
            raise ValueError(f"represented and unrepresented physics overlap: {sorted(overlap)}")
        if self.kind is WorldKind.NUMERICAL_SIMULATOR and not (self.unrepresented_physics):
            raise ValueError("a numerical world must declare its unrepresented physical scope")
        if not self.available_outcome_access:
            raise ValueError("world must declare at least one outcome-access capability")
        if self.privileged_truth_quantity_ids and (
            OutcomeAccess.PRIVILEGED_TRUTH not in self.available_outcome_access
        ):
            raise ValueError("privileged truth quantities require privileged-truth access")
        targets = tuple(rule.target_kind.value for rule in self.transition_rules)
        if tuple(sorted(set(targets))) != targets:
            raise ValueError("transition rules must have sorted, unique target kinds")
        require_extensions(self.extensions)

    def transition_rule_to(self, target: WorldSpec) -> WorldTransitionRule | None:
        return next(
            (rule for rule in self.transition_rules if rule.target_kind is target.kind),
            None,
        )


def validate_world_transition(
    source: WorldSpec,
    target: WorldSpec,
    requested_ceiling: EvidenceCeiling,
    *,
    independent_validation_bound: bool,
) -> EvidenceCeiling:
    """Validate a world transition and return its effective evidence ceiling."""

    if source.world_id == target.world_id:
        effective = EvidenceCeiling.lowest(source.maximum_evidence, requested_ceiling)
        if effective is not requested_ceiling:
            raise ValueError("requested evidence exceeds the source-world ceiling")
        return effective
    rule = source.transition_rule_to(target)
    if rule is None:
        raise ValueError(f"world {source.world_id!r} has no transition to {target.kind.value}")
    effective = EvidenceCeiling.lowest(
        source.maximum_evidence,
        target.maximum_evidence,
        rule.evidence_ceiling,
        requested_ceiling,
    )
    if effective is not requested_ceiling:
        raise ValueError("a world transition cannot promote its evidence ceiling")
    if rule.requires_independent_validation and not independent_validation_bound:
        raise ValueError("world transition lacks independent validation")
    return effective


def validate_claim_in_world(claim: ClaimSpec, world: WorldSpec) -> None:
    if claim.world_id != world.world_id:
        raise ValueError("claim world does not match the evidence world")
    if claim.outcome_access not in world.available_outcome_access:
        raise ValueError("claim requests outcome access unavailable in its world")
    if (
        EvidenceCeiling.lowest(claim.evidence_ceiling, world.maximum_evidence)
        is not claim.evidence_ceiling
    ):
        raise ValueError("claim evidence ceiling exceeds the world ceiling")
    if claim.requested_rung is not None:
        if not world.maximum_evidence.allows(claim.requested_rung):
            raise ValueError("claim rung exceeds the evidence-world ceiling")


@dataclass(frozen=True, slots=True)
class NumericalCoordinateSpec(CanonicalRecord):
    """One numerical rather than physical refinement coordinate."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/numerical-coordinate-spec'

    coordinate_id: str
    kind: NumericalCoordinateKind
    value: Decimal
    unit: str
    refinement_level: int

    def __post_init__(self) -> None:
        validate_stable_id(self.coordinate_id, field_name="coordinate_id")
        validate_nonempty(self.unit, field_name="unit")
        validate_decimal(self.value, field_name="value", minimum=Decimal("0"))
        if self.value == 0:
            raise ValueError("numerical-coordinate value must be positive")
        if self.refinement_level < 0:
            raise ValueError("refinement_level must be nonnegative")


@dataclass(frozen=True, slots=True)
class ComputabilityEnvelope(CanonicalRecord):
    """Feasible classical-compute regime and explicit structural boundary."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/computability-envelope'

    envelope_id: str
    represented_effect_ids: tuple[str, ...]
    unresolved_effect_ids: tuple[str, ...]
    required_structure_ids: tuple[str, ...]
    computable_structure_ids: tuple[str, ...]
    max_cpu_cores: int
    max_memory_bytes: int
    max_gpu_devices: int
    max_wall_time_seconds: int
    max_output_bytes: int
    worst_case_latency_seconds: Decimal
    deadline_seconds: Decimal | None = None
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.envelope_id, field_name="envelope_id")
        for field_name, values in (
            ("represented_effect_ids", self.represented_effect_ids),
            ("unresolved_effect_ids", self.unresolved_effect_ids),
            ("required_structure_ids", self.required_structure_ids),
            ("computable_structure_ids", self.computable_structure_ids),
        ):
            require_sorted_unique_strings(
                values,
                field_name=field_name,
                allow_empty=field_name not in {"represented_effect_ids", "required_structure_ids"},
            )
        overlap = set(self.represented_effect_ids) & set(self.unresolved_effect_ids)
        if overlap:
            raise ValueError(f"represented and unresolved effects overlap: {sorted(overlap)}")
        for field_name, value in (
            ("max_cpu_cores", self.max_cpu_cores),
            ("max_memory_bytes", self.max_memory_bytes),
            ("max_wall_time_seconds", self.max_wall_time_seconds),
            ("max_output_bytes", self.max_output_bytes),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        if self.max_gpu_devices < 0:
            raise ValueError("max_gpu_devices must be nonnegative")
        validate_decimal(
            self.worst_case_latency_seconds,
            field_name="worst_case_latency_seconds",
            minimum=Decimal("0"),
        )
        if self.deadline_seconds is not None:
            validate_decimal(
                self.deadline_seconds,
                field_name="deadline_seconds",
                minimum=Decimal("0"),
            )
        require_extensions(self.extensions)

    @property
    def missing_structure_ids(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.required_structure_ids) - set(self.computable_structure_ids)))

    @property
    def readiness(self) -> ReadinessStatus:
        missed_deadline = (
            self.deadline_seconds is not None
            and self.worst_case_latency_seconds > self.deadline_seconds
        )
        if self.unresolved_effect_ids or self.missing_structure_ids or missed_deadline:
            return ReadinessStatus.COMPUTABILITY_BOUNDARY
        return ReadinessStatus.READY


@dataclass(frozen=True, slots=True)
class NumericalViewSpec(CanonicalRecord):
    """One immutable numerical view nested inside a physical preparation."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/kernel/numerical-view-spec'

    view_id: str
    world_id: str
    physical_preparation_id: str
    equations_id: str
    closure_ids: tuple[str, ...]
    boundary_condition_ids: tuple[str, ...]
    coordinates: tuple[NumericalCoordinateSpec, ...]
    solver_id: str
    solver_version: str
    precision: str
    device_class: str
    runtime_id: str
    randomness: RandomnessSemantics
    observation_operator_id: str
    computability_envelope_id: str
    evidence_scope: EvidenceUnitScope = EvidenceUnitScope.NESTED_NUMERICAL_VIEW
    extensions: tuple[ExtensionBinding, ...] = ()

    def __post_init__(self) -> None:
        for name, value in (
            ("view_id", self.view_id),
            ("world_id", self.world_id),
            ("physical_preparation_id", self.physical_preparation_id),
            ("equations_id", self.equations_id),
            ("solver_id", self.solver_id),
            ("runtime_id", self.runtime_id),
            ("observation_operator_id", self.observation_operator_id),
            ("computability_envelope_id", self.computability_envelope_id),
        ):
            validate_stable_id(value, field_name=name)
        for name, value in (
            ("solver_version", self.solver_version),
            ("precision", self.precision),
            ("device_class", self.device_class),
        ):
            validate_nonempty(value, field_name=name)
        require_sorted_unique_strings(self.closure_ids, field_name="closure_ids")
        require_sorted_unique_strings(
            self.boundary_condition_ids,
            field_name="boundary_condition_ids",
            allow_empty=False,
        )
        if not self.closure_ids:
            raise ValueError("closure_ids must not be empty")
        if not self.coordinates:
            raise ValueError("a numerical view requires numerical coordinates")
        require_sorted_unique_ids(
            self.coordinates,
            attribute="coordinate_id",
            field_name="coordinates",
        )
        if self.evidence_scope is not EvidenceUnitScope.NESTED_NUMERICAL_VIEW:
            raise ValueError("a NumericalViewSpec cannot count as a physical independent unit")
        require_extensions(self.extensions)

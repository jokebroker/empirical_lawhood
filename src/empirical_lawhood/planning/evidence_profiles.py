"""Closed evidence-world and experiment-objective classification.

The registry answers whether a scientific rung can be requested before a
candidate graph is constructed.  It classifies existing contracts; it does
not redefine evidence, authority, evaluation, or capability ownership.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, EvidenceRung
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_schema,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)


class EvidenceWorldKind(StrEnum):
    TRUTH_KNOWN_REFERENCE = "TRUTH_KNOWN_REFERENCE"
    DETERMINISTIC_SIMULATOR = "DETERMINISTIC_SIMULATOR"
    FIXED_ARCHIVE = "FIXED_ARCHIVE"
    BENCHMARK = "BENCHMARK"
    STREAM = "STREAM"
    SHADOW_HIL = "SHADOW_HIL"
    PHYSICAL = "PHYSICAL"
    HYBRID_TRANSPORT = "HYBRID_TRANSPORT"
    STRUCTURAL_METATHEORY = "STRUCTURAL_METATHEORY"


class ExperimentObjectiveKind(StrEnum):
    LAW_CONTROL = "LAW_CONTROL"
    OBSERVATION_ORDER = "OBSERVATION_ORDER"
    PARTIAL_MORPHISM = "PARTIAL_MORPHISM"
    STRUCTURAL_METATHEORY = "STRUCTURAL_METATHEORY"
    HISTORICAL_PREDICTION = "HISTORICAL_PREDICTION"


class EvidenceRole(StrEnum):
    SOURCE = "SOURCE"
    EVIDENCE_PROJECTION = "EVIDENCE_PROJECTION"
    ACTION_OCCURRENCE = "ACTION_OCCURRENCE"
    EFFECT_COMPUTABILITY = "EFFECT_COMPUTABILITY"
    LAW_METHOD = "LAW_METHOD"
    SOLE_LAW_FINALIZER = "SOLE_LAW_FINALIZER"
    PROSPECTIVE_ACTUATION = "PROSPECTIVE_ACTUATION"
    CONTROLLER = "CONTROLLER"
    SEALED_EVALUATOR = "SEALED_EVALUATOR"
    PROPERTY_MAP = "PROPERTY_MAP"
    STRUCTURAL_HANDOFF = "STRUCTURAL_HANDOFF"


class RoleApplicabilityStatus(StrEnum):
    REQUIRED = "REQUIRED"
    AVAILABLE = "AVAILABLE"
    INAPPLICABLE = "INAPPLICABLE"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"


class RungFeasibilityStatus(StrEnum):
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    INAPPLICABLE = "INAPPLICABLE"
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"


class EvidenceProfileReasonCode(StrEnum):
    PROFILE_ROLE_AVAILABLE = "PROFILE_ROLE_AVAILABLE"
    PROFILE_ROLE_REQUIRED = "PROFILE_ROLE_REQUIRED"
    PROFILE_ROLE_INAPPLICABLE = "PROFILE_ROLE_INAPPLICABLE"
    PROFILE_ROLE_AUTHORITY_REQUIRED = "PROFILE_ROLE_AUTHORITY_REQUIRED"
    TRUTH_KNOWN_CONFORMANCE_ONLY = "TRUTH_KNOWN_CONFORMANCE_ONLY"
    WORLD_SUPPORTS_REQUESTED_RUNG = "WORLD_SUPPORTS_REQUESTED_RUNG"
    OBJECTIVE_TERMINAL_OUTSIDE_EVIDENCE_RUNGS = "OBJECTIVE_TERMINAL_OUTSIDE_EVIDENCE_RUNGS"
    STRUCTURAL_WORLD_HAS_NO_EVIDENCE_RUNG_TARGET = "STRUCTURAL_WORLD_HAS_NO_EVIDENCE_RUNG_TARGET"
    NO_PROSPECTIVE_INTERVENTION = "NO_PROSPECTIVE_INTERVENTION"
    NO_LAW_QUALIFICATION_ROLE = "NO_LAW_QUALIFICATION_ROLE"
    NO_CONTROLLER_EVALUATION_ROLE = "NO_CONTROLLER_EVALUATION_ROLE"
    OPERATION_AUTHORITY_REQUIRED = "OPERATION_AUTHORITY_REQUIRED"
    PROFILE_IDENTITY_MISMATCH = "PROFILE_IDENTITY_MISMATCH"
    OBSERVATION_ORDER_ONLY_ORDER_RELATION = "OBSERVATION_ORDER_ONLY_ORDER_RELATION"
    HISTORICAL_EXPERIMENT_ROOT_REQUIRED = "HISTORICAL_EXPERIMENT_ROOT_REQUIRED"


class ProfileCompatibilitySurface(StrEnum):
    SYSTEM = "SYSTEM"
    EVIDENCE = "EVIDENCE"
    CAPABILITY = "CAPABILITY"
    AUTHORITY = "AUTHORITY"
    EVALUATION = "EVALUATION"


@dataclass(frozen=True, slots=True)
class RoleApplicability(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/role-applicability'

    role: EvidenceRole
    status: RoleApplicabilityStatus
    reason_code: EvidenceProfileReasonCode

    def __post_init__(self) -> None:
        expected = {
            RoleApplicabilityStatus.REQUIRED: EvidenceProfileReasonCode.PROFILE_ROLE_REQUIRED,
            RoleApplicabilityStatus.AVAILABLE: EvidenceProfileReasonCode.PROFILE_ROLE_AVAILABLE,
            RoleApplicabilityStatus.INAPPLICABLE: (
                EvidenceProfileReasonCode.PROFILE_ROLE_INAPPLICABLE
            ),
            RoleApplicabilityStatus.AUTHORITY_REQUIRED: (
                EvidenceProfileReasonCode.PROFILE_ROLE_AUTHORITY_REQUIRED
            ),
        }
        if self.reason_code is not expected[self.status]:
            raise ValueError("role applicability reason differs from its closed status")


@dataclass(frozen=True, slots=True)
class EvidenceWorldProfile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-world-profile'

    profile_id: str
    profile_key: str
    profile_version: str
    world_kind: EvidenceWorldKind
    maximum_evidence_ceiling: EvidenceCeiling
    role_applicabilities: tuple[RoleApplicability, ...]

    def __post_init__(self) -> None:
        for name, value in (("profile_id", self.profile_id), ("profile_key", self.profile_key)):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.profile_version)
        roles = tuple(value.role.value for value in self.role_applicabilities)
        if roles != tuple(sorted(role.value for role in EvidenceRole)):
            raise ValueError("world profile must classify every evidence role exactly once")


@dataclass(frozen=True, slots=True)
class ExperimentObjectiveProfile(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/experiment-objective-profile'

    profile_id: str
    profile_key: str
    profile_version: str
    objective_kind: ExperimentObjectiveKind
    required_roles: tuple[EvidenceRole, ...]
    terminal_product_schema: str
    uses_evidence_rungs: bool

    def __post_init__(self) -> None:
        for name, value in (("profile_id", self.profile_id), ("profile_key", self.profile_key)):
            validate_stable_id(value, field_name=name)
        validate_semantic_version(self.profile_version)
        if tuple(role.value for role in self.required_roles) != tuple(
            sorted({role.value for role in self.required_roles})
        ):
            raise ValueError("objective required roles must be sorted and unique")
        validate_schema(self.terminal_product_schema)
        if self.objective_kind in {
            ExperimentObjectiveKind.LAW_CONTROL,
            ExperimentObjectiveKind.OBSERVATION_ORDER,
        }:
            if not self.uses_evidence_rungs:
                raise ValueError("rung-addressed objective must use the evidence-rung lattice")
        elif self.uses_evidence_rungs:
            raise ValueError("objective-specific terminal products do not use evidence rungs")


@dataclass(frozen=True, slots=True)
class RungFeasibilityCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/rung-feasibility-cell'

    cell_id: str
    evidence_world_profile_id: str
    objective_profile_id: str
    rung: EvidenceRung
    status: RungFeasibilityStatus
    required_roles: tuple[EvidenceRole, ...]
    maximum_evidence_ceiling: EvidenceCeiling
    adjudication_owner_id: str
    reason_codes: tuple[EvidenceProfileReasonCode, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("evidence_world_profile_id", self.evidence_world_profile_id),
            ("objective_profile_id", self.objective_profile_id),
            ("adjudication_owner_id", self.adjudication_owner_id),
        ):
            validate_stable_id(value, field_name=name)
        if tuple(role.value for role in self.required_roles) != tuple(
            sorted({role.value for role in self.required_roles})
        ):
            raise ValueError("rung required roles must be sorted and unique")
        if tuple(reason.value for reason in self.reason_codes) != tuple(
            sorted({reason.value for reason in self.reason_codes})
        ):
            raise ValueError("rung reason codes must be sorted and unique")
        if not self.reason_codes:
            raise ValueError("rung feasibility requires a typed reason")
        if self.status is RungFeasibilityStatus.FEASIBLE:
            truth_only = EvidenceProfileReasonCode.TRUTH_KNOWN_CONFORMANCE_ONLY in self.reason_codes
            if not truth_only and not self.maximum_evidence_ceiling.allows(self.rung):
                raise ValueError("feasible rung exceeds its evidence ceiling")


@dataclass(frozen=True, slots=True)
class ProfileCompatibilityBinding(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/profile-compatibility-binding'

    surface: ProfileCompatibilitySurface
    schema_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        require_sorted_unique_strings(self.schema_ids, field_name="schema_ids", allow_empty=False)
        for schema_id in self.schema_ids:
            validate_schema(schema_id)


@dataclass(frozen=True, slots=True)
class EvidenceProfileSelection(CanonicalRecord):
    """Exact profile identities bound to one candidate request."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-profile-selection'

    selection_id: str
    draft_id: str
    registry_id: str
    registry_fingerprint: str
    evidence_world_profile_id: str
    evidence_world_profile_version: str
    evidence_world_profile_fingerprint: str
    objective_profile_id: str
    objective_profile_version: str
    objective_profile_fingerprint: str
    requested_rungs: tuple[EvidenceRung, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("selection_id", self.selection_id),
            ("draft_id", self.draft_id),
            ("registry_id", self.registry_id),
            ("evidence_world_profile_id", self.evidence_world_profile_id),
            ("objective_profile_id", self.objective_profile_id),
        ):
            validate_stable_id(value, field_name=name)
        for value in (
            self.evidence_world_profile_version,
            self.objective_profile_version,
        ):
            validate_semantic_version(value)
        for name, value in (
            ("registry_fingerprint", self.registry_fingerprint),
            ("evidence_world_profile_fingerprint", self.evidence_world_profile_fingerprint),
            ("objective_profile_fingerprint", self.objective_profile_fingerprint),
        ):
            validate_sha256(value, field_name=name)
        if tuple(rung.value for rung in self.requested_rungs) != tuple(
            sorted({rung.value for rung in self.requested_rungs})
        ):
            raise ValueError("requested rungs must be sorted and unique")


@dataclass(frozen=True, slots=True)
class EvidenceWorldProfileRegistry(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/planning/evidence-world-profile-registry'

    registry_id: str
    registry_version: str
    world_profiles: tuple[EvidenceWorldProfile, ...]
    objective_profiles: tuple[ExperimentObjectiveProfile, ...]
    rung_cells: tuple[RungFeasibilityCell, ...]
    compatibility_bindings: tuple[ProfileCompatibilityBinding, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.registry_id, field_name="registry_id")
        validate_semantic_version(self.registry_version)
        require_sorted_unique_ids(
            self.world_profiles, attribute="profile_id", field_name="world_profiles"
        )
        require_sorted_unique_ids(
            self.objective_profiles, attribute="profile_id", field_name="objective_profiles"
        )
        require_sorted_unique_ids(self.rung_cells, attribute="cell_id", field_name="rung_cells")
        surfaces = tuple(value.surface.value for value in self.compatibility_bindings)
        if surfaces != tuple(sorted(surface.value for surface in ProfileCompatibilitySurface)):
            raise ValueError("registry must bind every compatibility surface exactly once")
        world_ids = {value.profile_id for value in self.world_profiles}
        objective_ids = {value.profile_id for value in self.objective_profiles}
        expected = {
            (world_id, objective_id, rung)
            for world_id in world_ids
            for objective_id in objective_ids
            for rung in EvidenceRung
        }
        actual = {
            (value.evidence_world_profile_id, value.objective_profile_id, value.rung)
            for value in self.rung_cells
        }
        if actual != expected or len(actual) != len(self.rung_cells):
            raise ValueError(
                "registry rung cells must cover the exact world/objective/evidence-rung product"
            )


_ROLE_REASON = {
    RoleApplicabilityStatus.REQUIRED: EvidenceProfileReasonCode.PROFILE_ROLE_REQUIRED,
    RoleApplicabilityStatus.AVAILABLE: EvidenceProfileReasonCode.PROFILE_ROLE_AVAILABLE,
    RoleApplicabilityStatus.INAPPLICABLE: EvidenceProfileReasonCode.PROFILE_ROLE_INAPPLICABLE,
    RoleApplicabilityStatus.AUTHORITY_REQUIRED: (
        EvidenceProfileReasonCode.PROFILE_ROLE_AUTHORITY_REQUIRED
    ),
}


def _role_roster(
    *,
    required: frozenset[EvidenceRole] = frozenset(),
    authority: frozenset[EvidenceRole] = frozenset(),
    inapplicable: frozenset[EvidenceRole] = frozenset(),
) -> tuple[RoleApplicability, ...]:
    values = []
    for role in EvidenceRole:
        status = (
            RoleApplicabilityStatus.REQUIRED
            if role in required
            else RoleApplicabilityStatus.AUTHORITY_REQUIRED
            if role in authority
            else RoleApplicabilityStatus.INAPPLICABLE
            if role in inapplicable
            else RoleApplicabilityStatus.AVAILABLE
        )
        values.append(
            RoleApplicability(role=role, status=status, reason_code=_ROLE_REASON[status])
        )
    return tuple(sorted(values, key=lambda value: value.role.value))


_LAW_ROLES = frozenset(
    {
        EvidenceRole.SOURCE,
        EvidenceRole.EVIDENCE_PROJECTION,
        EvidenceRole.ACTION_OCCURRENCE,
        EvidenceRole.EFFECT_COMPUTABILITY,
        EvidenceRole.LAW_METHOD,
        EvidenceRole.SOLE_LAW_FINALIZER,
    }
)
_CONTROL_ROLES = frozenset(
    {EvidenceRole.PROSPECTIVE_ACTUATION, EvidenceRole.CONTROLLER, EvidenceRole.SEALED_EVALUATOR}
)


def _world_profiles() -> tuple[EvidenceWorldProfile, ...]:
    definitions: tuple[
        tuple[
            EvidenceWorldKind,
            EvidenceCeiling,
            frozenset[EvidenceRole],
            frozenset[EvidenceRole],
            frozenset[EvidenceRole],
        ],
        ...,
    ] = (
        (
            EvidenceWorldKind.TRUTH_KNOWN_REFERENCE,
            EvidenceCeiling.NON_PROMOTABLE,
            _LAW_ROLES,
            frozenset(),
            _CONTROL_ROLES,
        ),
        (
            EvidenceWorldKind.DETERMINISTIC_SIMULATOR,
            EvidenceCeiling.CONTROLLER_USE,
            _LAW_ROLES,
            frozenset(),
            frozenset(),
        ),
        (
            EvidenceWorldKind.FIXED_ARCHIVE,
            EvidenceCeiling.LOCAL_LAW,
            _LAW_ROLES,
            frozenset(),
            _CONTROL_ROLES,
        ),
        (
            EvidenceWorldKind.BENCHMARK,
            EvidenceCeiling.RESPONSE,
            frozenset({EvidenceRole.SOURCE, EvidenceRole.EVIDENCE_PROJECTION}),
            frozenset(),
            frozenset({EvidenceRole.CONTROLLER, EvidenceRole.PROSPECTIVE_ACTUATION}),
        ),
        (
            EvidenceWorldKind.STREAM,
            EvidenceCeiling.LOCAL_LAW,
            _LAW_ROLES,
            frozenset(),
            _CONTROL_ROLES,
        ),
        (
            EvidenceWorldKind.SHADOW_HIL,
            EvidenceCeiling.CONTROLLER_USE,
            _LAW_ROLES,
            _CONTROL_ROLES,
            frozenset(),
        ),
        (
            EvidenceWorldKind.PHYSICAL,
            EvidenceCeiling.CONTROLLER_USE,
            _LAW_ROLES,
            _CONTROL_ROLES,
            frozenset(),
        ),
        (
            EvidenceWorldKind.HYBRID_TRANSPORT,
            EvidenceCeiling.CONTROLLER_USE,
            _LAW_ROLES | frozenset({EvidenceRole.PROPERTY_MAP}),
            _CONTROL_ROLES,
            frozenset(),
        ),
        (
            EvidenceWorldKind.STRUCTURAL_METATHEORY,
            EvidenceCeiling.NON_PROMOTABLE,
            frozenset({EvidenceRole.STRUCTURAL_HANDOFF}),
            frozenset(),
            frozenset(
                set(EvidenceRole) - {EvidenceRole.STRUCTURAL_HANDOFF, EvidenceRole.PROPERTY_MAP}
            ),
        ),
    )
    values = []
    for kind, ceiling, required, authority, inapplicable in definitions:
        key = kind.value.lower().replace("_", "-")
        values.append(
            EvidenceWorldProfile(
                profile_id=f"evidence-world.{key}",
                profile_key=f"evidence-world.{key}",
                profile_version="1.0.0",
                world_kind=kind,
                maximum_evidence_ceiling=ceiling,
                role_applicabilities=_role_roster(
                    required=required, authority=authority, inapplicable=inapplicable
                ),
            )
        )
    return tuple(sorted(values, key=lambda value: value.profile_id))


def _objective_profiles() -> tuple[ExperimentObjectiveProfile, ...]:
    values = (
        ExperimentObjectiveProfile(
            profile_id="objective.law-control",
            profile_key="objective.law-control",
            profile_version="1.0.0",
            objective_kind=ExperimentObjectiveKind.LAW_CONTROL,
            required_roles=tuple(sorted(_LAW_ROLES | _CONTROL_ROLES, key=lambda role: role.value)),
            terminal_product_schema='empirical-lawhood/runtime/repeated-delivery-controller-cohort-adjudication',
            uses_evidence_rungs=True,
        ),
        ExperimentObjectiveProfile(
            profile_id="objective.partial-morphism",
            profile_key="objective.partial-morphism",
            profile_version="1.0.0",
            objective_kind=ExperimentObjectiveKind.PARTIAL_MORPHISM,
            required_roles=(EvidenceRole.PROPERTY_MAP,),
            terminal_product_schema='empirical-lawhood/methods/partial-morphism-assessment',
            uses_evidence_rungs=False,
        ),
        ExperimentObjectiveProfile(
            profile_id="objective.structural-metatheory",
            profile_key="objective.structural-metatheory",
            profile_version="1.0.0",
            objective_kind=ExperimentObjectiveKind.STRUCTURAL_METATHEORY,
            required_roles=(EvidenceRole.STRUCTURAL_HANDOFF,),
            terminal_product_schema='empirical-lawhood/planning/metatheory-fan-in-assessment',
            uses_evidence_rungs=False,
        ),
    )
    return tuple(sorted(values, key=lambda value: value.profile_id))


def _observation_order_objective() -> ExperimentObjectiveProfile:
    return ExperimentObjectiveProfile(
        profile_id="objective.observation-order",
        profile_key="objective.observation-order",
        profile_version="1.0.0",
        objective_kind=ExperimentObjectiveKind.OBSERVATION_ORDER,
        required_roles=(
            EvidenceRole.EVIDENCE_PROJECTION,
            EvidenceRole.SEALED_EVALUATOR,
            EvidenceRole.SOURCE,
        ),
        terminal_product_schema='empirical-lawhood/runtime/scientific-adjudication-record',
        uses_evidence_rungs=True,
    )


def _cell_status(
    world: EvidenceWorldProfile,
    objective: ExperimentObjectiveProfile,
    rung: EvidenceRung,
) -> tuple[RungFeasibilityStatus, tuple[EvidenceProfileReasonCode, ...]]:
    if objective.objective_kind is ExperimentObjectiveKind.OBSERVATION_ORDER:
        if rung is not EvidenceRung.ORDER_RELATION:
            return (
                RungFeasibilityStatus.INAPPLICABLE,
                (EvidenceProfileReasonCode.OBSERVATION_ORDER_ONLY_ORDER_RELATION,),
            )
        if world.world_kind is EvidenceWorldKind.STRUCTURAL_METATHEORY:
            return (
                RungFeasibilityStatus.INAPPLICABLE,
                (EvidenceProfileReasonCode.STRUCTURAL_WORLD_HAS_NO_EVIDENCE_RUNG_TARGET,),
            )
        if world.world_kind is EvidenceWorldKind.TRUTH_KNOWN_REFERENCE:
            return (
                RungFeasibilityStatus.FEASIBLE,
                (EvidenceProfileReasonCode.TRUTH_KNOWN_CONFORMANCE_ONLY,),
            )
        return (
            RungFeasibilityStatus.FEASIBLE,
            (EvidenceProfileReasonCode.WORLD_SUPPORTS_REQUESTED_RUNG,),
        )
    if not objective.uses_evidence_rungs:
        return (
            RungFeasibilityStatus.INAPPLICABLE,
            (EvidenceProfileReasonCode.OBJECTIVE_TERMINAL_OUTSIDE_EVIDENCE_RUNGS,),
        )
    kind = world.world_kind
    if kind is EvidenceWorldKind.STRUCTURAL_METATHEORY:
        return (
            RungFeasibilityStatus.INAPPLICABLE,
            (EvidenceProfileReasonCode.STRUCTURAL_WORLD_HAS_NO_EVIDENCE_RUNG_TARGET,),
        )
    if kind is EvidenceWorldKind.TRUTH_KNOWN_REFERENCE:
        return (
            RungFeasibilityStatus.FEASIBLE,
            (EvidenceProfileReasonCode.TRUTH_KNOWN_CONFORMANCE_ONLY,),
        )
    if kind in {EvidenceWorldKind.FIXED_ARCHIVE, EvidenceWorldKind.STREAM} and rung in {
        EvidenceRung.ADMISSION,
        EvidenceRung.CONTROLLER_USE,
    }:
        return (
            RungFeasibilityStatus.INFEASIBLE,
            (EvidenceProfileReasonCode.NO_PROSPECTIVE_INTERVENTION,),
        )
    if kind is EvidenceWorldKind.BENCHMARK and rung in {
        EvidenceRung.LOCAL_LAW,
        EvidenceRung.ADMISSION,
        EvidenceRung.CONTROLLER_USE,
    }:
        reason = (
            EvidenceProfileReasonCode.NO_LAW_QUALIFICATION_ROLE
            if rung is EvidenceRung.LOCAL_LAW
            else EvidenceProfileReasonCode.NO_CONTROLLER_EVALUATION_ROLE
        )
        return RungFeasibilityStatus.INFEASIBLE, (reason,)
    if kind in {
        EvidenceWorldKind.SHADOW_HIL,
        EvidenceWorldKind.PHYSICAL,
        EvidenceWorldKind.HYBRID_TRANSPORT,
    } and rung in {EvidenceRung.ADMISSION, EvidenceRung.CONTROLLER_USE}:
        return (
            RungFeasibilityStatus.AUTHORITY_REQUIRED,
            (EvidenceProfileReasonCode.OPERATION_AUTHORITY_REQUIRED,),
        )
    return (
        RungFeasibilityStatus.FEASIBLE,
        (EvidenceProfileReasonCode.WORLD_SUPPORTS_REQUESTED_RUNG,),
    )


def build_law_and_structural_evidence_world_registry() -> EvidenceWorldProfileRegistry:
    """Return the finite built-in registry used by the F0--F4 pilots."""

    worlds = _world_profiles()
    objectives = _objective_profiles()
    cells = []
    for world in worlds:
        for objective in objectives:
            for rung in EvidenceRung:
                status, reasons = _cell_status(world, objective, rung)
                suffix = rung.value.lower().replace("_", "-")
                cells.append(
                    RungFeasibilityCell(
                        cell_id=(f"cell.{world.world_kind.value.lower().replace('_', '-')}.")
                        + f"{objective.objective_kind.value.lower().replace('_', '-')}.{suffix}",
                        evidence_world_profile_id=world.profile_id,
                        objective_profile_id=objective.profile_id,
                        rung=rung,
                        status=status,
                        required_roles=objective.required_roles,
                        maximum_evidence_ceiling=world.maximum_evidence_ceiling,
                        adjudication_owner_id=(
                            "owner.response-law-qualification"
                            if objective.uses_evidence_rungs
                            else "owner.objective-specific-terminal"
                        ),
                        reason_codes=reasons,
                    )
                )
    compatibility = (
        ProfileCompatibilityBinding(
            surface=ProfileCompatibilitySurface.AUTHORITY,
            schema_ids=('empirical-lawhood/planning/study-operation-authority',),
        ),
        ProfileCompatibilityBinding(
            surface=ProfileCompatibilitySurface.CAPABILITY,
            schema_ids=('empirical-lawhood/runtime/capability-registry',),
        ),
        ProfileCompatibilityBinding(
            surface=ProfileCompatibilitySurface.EVALUATION,
            schema_ids=(
                'empirical-lawhood/kernel/evaluation-request',
                'empirical-lawhood/kernel/evaluation-result',
            ),
        ),
        ProfileCompatibilityBinding(
            surface=ProfileCompatibilitySurface.EVIDENCE,
            schema_ids=(
                'empirical-lawhood/planning/identification-evidence-manifest',
                'empirical-lawhood/planning/identification-evidence-projection',
            ),
        ),
        ProfileCompatibilityBinding(
            surface=ProfileCompatibilitySurface.SYSTEM,
            schema_ids=('empirical-lawhood/kernel/system-spec',),
        ),
    )
    return EvidenceWorldProfileRegistry(
        registry_id="registry.evidence-world-law-and-structural-objectives",
        registry_version="1.0.0",
        world_profiles=worlds,
        objective_profiles=objectives,
        rung_cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        compatibility_bindings=compatibility,
    )


def build_observation_evidence_world_registry() -> EvidenceWorldProfileRegistry:
    """Classify observation-order objectives alongside law and structural objectives."""

    base = build_law_and_structural_evidence_world_registry()
    objectives = tuple(
        sorted(
            (*base.objective_profiles, _observation_order_objective()), key=lambda x: x.profile_id
        )
    )
    cells = []
    for world in base.world_profiles:
        for objective in objectives:
            for rung in EvidenceRung:
                status, reasons = _cell_status(world, objective, rung)
                suffix = rung.value.lower().replace("_", "-")
                cells.append(
                    RungFeasibilityCell(
                        cell_id=(f"cell.{world.world_kind.value.lower().replace('_', '-')}.")
                        + f"{objective.objective_kind.value.lower().replace('_', '-')}.{suffix}",
                        evidence_world_profile_id=world.profile_id,
                        objective_profile_id=objective.profile_id,
                        rung=rung,
                        status=status,
                        required_roles=objective.required_roles,
                        maximum_evidence_ceiling=world.maximum_evidence_ceiling,
                        adjudication_owner_id=(
                            "owner.observation-order-adjudication"
                            if objective.objective_kind
                            is ExperimentObjectiveKind.OBSERVATION_ORDER
                            else "owner.response-law-qualification"
                            if objective.uses_evidence_rungs
                            else "owner.objective-specific-terminal"
                        ),
                        reason_codes=reasons,
                    )
                )
    return EvidenceWorldProfileRegistry(
        registry_id="registry.evidence-world-observation-order-objectives",
        registry_version="1.0.0",
        world_profiles=base.world_profiles,
        objective_profiles=objectives,
        rung_cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        compatibility_bindings=base.compatibility_bindings,
    )


def build_held_prediction_evidence_world_registry() -> EvidenceWorldProfileRegistry:
    """Classify held-source prediction without promoting exposed historical evidence."""
    base = build_observation_evidence_world_registry()
    objective = ExperimentObjectiveProfile(
        "objective.historical-prediction",
        "objective.historical-prediction",
        "1.0.0",
        ExperimentObjectiveKind.HISTORICAL_PREDICTION,
        (EvidenceRole.EVIDENCE_PROJECTION, EvidenceRole.SEALED_EVALUATOR, EvidenceRole.SOURCE),
        'empirical-lawhood/runtime/scientific-adjudication-record',
        False,
    )
    cells = tuple(
        RungFeasibilityCell(
            f"cell.{world.world_kind.value.lower().replace('_', '-')}.historical-prediction."
            + rung.value.lower().replace("_", "-"),
            world.profile_id,
            objective.profile_id,
            rung,
            RungFeasibilityStatus.INAPPLICABLE,
            objective.required_roles,
            EvidenceCeiling.NON_PROMOTABLE,
            "owner.historical-prediction-evaluation",
            (EvidenceProfileReasonCode.OBJECTIVE_TERMINAL_OUTSIDE_EVIDENCE_RUNGS,),
        )
        for world in base.world_profiles
        for rung in EvidenceRung
    )
    return EvidenceWorldProfileRegistry(
        "registry.evidence-world-held-source-prediction-objectives",
        "1.0.0",
        base.world_profiles,
        tuple(sorted((*base.objective_profiles, objective), key=lambda x: x.profile_id)),
        tuple(sorted((*base.rung_cells, *cells), key=lambda x: x.cell_id)),
        base.compatibility_bindings,
    )


def bind_profile_selection(
    *,
    selection_id: str,
    draft_id: str,
    registry: EvidenceWorldProfileRegistry,
    world_profile_id: str,
    objective_profile_id: str,
    requested_rungs: tuple[EvidenceRung, ...],
) -> EvidenceProfileSelection:
    worlds = {value.profile_id: value for value in registry.world_profiles}
    objectives = {value.profile_id: value for value in registry.objective_profiles}
    try:
        world = worlds[world_profile_id]
        objective = objectives[objective_profile_id]
    except KeyError as error:
        raise ValueError("profile selection names an unregistered profile") from error
    return EvidenceProfileSelection(
        selection_id=selection_id,
        draft_id=draft_id,
        registry_id=registry.registry_id,
        registry_fingerprint=registry.fingerprint(),
        evidence_world_profile_id=world.profile_id,
        evidence_world_profile_version=world.profile_version,
        evidence_world_profile_fingerprint=world.fingerprint(),
        objective_profile_id=objective.profile_id,
        objective_profile_version=objective.profile_version,
        objective_profile_fingerprint=objective.fingerprint(),
        requested_rungs=requested_rungs,
    )

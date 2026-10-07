"""Frozen scientific specification for the TORAX method value consumer."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.adapters.simulators.torax_native.contracts import NativeToraxAction, NativeToraxEpisode, NativeToraxPreparation, NativeToraxView
from empirical_lawhood.kernel.evidence import EvidenceRung, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import parse_utc_timestamp
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.planning.identification_evidence import ObservationActionDeliveryBinding
from empirical_lawhood.runtime.route_conformance import PUBLIC_ROUTE_STAGE_ORDER, PublicRouteConformanceStage


class ToraxMethodValueTaskRole(StrEnum):
    CANARY_EXCLUDED = "CANARY_EXCLUDED"
    DEVELOPMENT_COMMON = "DEVELOPMENT_COMMON"
    DEVELOPMENT_CANDIDATE = "DEVELOPMENT_CANDIDATE"
    PROTECTED_REFERENCE = "PROTECTED_REFERENCE"
    PROSPECTIVE_CONFIRMATION = "PROSPECTIVE_CONFIRMATION"


class ToraxMethodValueArm(StrEnum):
    WITNESS_GATED_IO = "WITNESS_GATED_IO"
    IO_ONLY = "IO_ONLY"
    FINITE_MPC = "FINITE_MPC"
    FARTHEST_POINT_MAXIMIN = "FARTHEST_POINT_MAXIMIN"


class ToraxMethodValueAcquisitionMethod(StrEnum):
    RCJ_IO_FALLBACK = "RCJ_IO_FALLBACK"
    IO_ONLY = "IO_ONLY"
    D_OPTIMAL = "D_OPTIMAL"
    FARTHEST_POINT = "FARTHEST_POINT"


class ToraxMethodValuePhase(StrEnum):
    CANARY = "CANARY"
    DEVELOPMENT = "DEVELOPMENT"
    PROTECTED_REFERENCE = "PROTECTED_REFERENCE"
    PROSPECTIVE_CONFIRMATION = "PROSPECTIVE_CONFIRMATION"


class ToraxMethodValueBackboneStageStatus(StrEnum):
    COMPLETE = "COMPLETE"
    OBSTRUCTED = "OBSTRUCTED"
    CONDITION_FALSE = "CONDITION_FALSE"


@dataclass(frozen=True, slots=True)
class ToraxMethodValueTaskSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-task-spec'

    task_id: str
    role: ToraxMethodValueTaskRole
    core_electron_temperature_ev: Decimal
    edge_electron_temperature_ev: Decimal
    temperature_profile_exponent: Decimal
    ion_to_electron_temperature_ratio: Decimal
    core_electron_density_m3: Decimal
    edge_electron_density_m3: Decimal
    plasma_current_a: Decimal
    chi_i_m2_s: Decimal
    chi_e_m2_s: Decimal
    source_radial_location: Decimal
    source_width: Decimal
    target_delta_core_temperature_ev: Decimal
    maximum_core_temperature_ev: Decimal

    def __post_init__(self) -> None:
        validate_stable_id(self.task_id, field_name="task_id")
        for name in (
            "core_electron_temperature_ev",
            "edge_electron_temperature_ev",
            "temperature_profile_exponent",
            "ion_to_electron_temperature_ratio",
            "core_electron_density_m3",
            "edge_electron_density_m3",
            "plasma_current_a",
            "chi_i_m2_s",
            "chi_e_m2_s",
            "source_width",
            "maximum_core_temperature_ev",
        ):
            value = getattr(self, name)
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value == 0:
                raise ValueError(f"{name} must be positive")
        validate_decimal(
            self.source_radial_location,
            field_name="source_radial_location",
            minimum=Decimal(0),
        )
        if self.source_radial_location > 1:
            raise ValueError("source radial location exceeds normalized radius")
        validate_decimal(
            self.target_delta_core_temperature_ev,
            field_name="target_delta_core_temperature_ev",
        )
        if self.edge_electron_temperature_ev >= self.core_electron_temperature_ev:
            raise ValueError("TORAX method value temperature profile must decrease from core to edge")
        if self.edge_electron_density_m3 > self.core_electron_density_m3:
            raise ValueError("TORAX method value density profile must not rise from core to edge")
        if self.maximum_core_temperature_ev <= self.core_electron_temperature_ev:
            raise ValueError("TORAX method value safety ceiling must exceed initial core temperature")


@dataclass(frozen=True, slots=True)
class ToraxMethodValueExperimentSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-experiment-spec'

    experiment_id: str
    lifecycle: str
    runtime_qualification: ObjectIdentity
    corrected_release_attestation: ObjectIdentity
    readiness_release_attestation: ObjectIdentity
    readiness_consumer_pin: ObjectIdentity
    v4_consumer_platform_binding: ObjectIdentity
    v5_archive_amendment: ObjectIdentity
    tasks: tuple[ToraxMethodValueTaskSpec, ...]
    actions: tuple[NativeToraxAction, ...]
    views: tuple[NativeToraxView, ...]
    radial_coordinates: tuple[Decimal, ...]
    main_ion: str
    impurity: str
    zeff: Decimal
    major_radius_m: Decimal
    minor_radius_m: Decimal
    toroidal_field_t: Decimal
    elongation_lcfs: Decimal
    electron_heat_fraction: Decimal
    absorbed_power_fraction: Decimal
    particle_diffusivity_m2_s: Decimal
    particle_convection_m_s: Decimal
    reference_tolerance_ev: Decimal
    rcj_contact_margin: Decimal
    maximum_adaptive_rounds: int
    maximum_episode_launches: int
    external_relative_root: str
    fair_mast_sidecar_disposition: str
    frozen: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.experiment_id, field_name="experiment_id")
        if self.lifecycle != "ISSUED_FROZEN":
            raise ValueError("TORAX method value experiment config must be issued and frozen")
        require_sorted_unique_ids(self.tasks, attribute="task_id", field_name="tasks")
        require_sorted_unique_ids(self.actions, attribute="action_id", field_name="actions")
        require_sorted_unique_ids(self.views, attribute="view_id", field_name="views")
        expected_roles = {
            ToraxMethodValueTaskRole.CANARY_EXCLUDED: 1,
            ToraxMethodValueTaskRole.DEVELOPMENT_COMMON: 2,
            ToraxMethodValueTaskRole.DEVELOPMENT_CANDIDATE: 4,
            ToraxMethodValueTaskRole.PROTECTED_REFERENCE: 8,
            ToraxMethodValueTaskRole.PROSPECTIVE_CONFIRMATION: 4,
        }
        counts = {role: sum(value.role is role for value in self.tasks) for role in ToraxMethodValueTaskRole}
        if counts != expected_roles:
            raise ValueError("TORAX method value task roster differs from the exact 1+2+4+8+4 design")
        if tuple(value.action_label for value in self.actions) != ("down", "hold", "up"):
            raise ValueError("TORAX method value action roster must be ordered DOWN/HOLD/UP")
        if tuple(value.realized_power_w for value in self.actions) != (
            Decimal("500000"),
            Decimal("800000"),
            Decimal("1100000"),
        ):
            raise ValueError("TORAX method value action powers differ from the freeze")
        if tuple(value.view_id for value in self.views) != (
            "torax-primary-view",
            "torax-sensitivity-view",
        ):
            raise ValueError("TORAX method value numerical-view roster differs from the freeze")
        if self.radial_coordinates != (
            Decimal(0),
            Decimal("0.25"),
            Decimal("0.5"),
            Decimal("0.75"),
            Decimal(1),
        ):
            raise ValueError("TORAX method value profile authoring grid differs from the freeze")
        for name in (
            "zeff",
            "major_radius_m",
            "minor_radius_m",
            "toroidal_field_t",
            "elongation_lcfs",
            "electron_heat_fraction",
            "absorbed_power_fraction",
            "particle_diffusivity_m2_s",
            "reference_tolerance_ev",
            "rcj_contact_margin",
        ):
            validate_decimal(getattr(self, name), field_name=name, minimum=Decimal(0))
        validate_decimal(
            self.particle_convection_m_s,
            field_name="particle_convection_m_s",
        )
        if self.maximum_adaptive_rounds != 2 or self.maximum_episode_launches != 240:
            raise ValueError("TORAX method value adaptive or physical episode ceiling differs")
        if self.external_relative_root != "torax-receiver-conditioned-io-method-value":
            raise ValueError("TORAX method value external relative root differs")
        if self.fair_mast_sidecar_disposition != "FAIR_MAST_SOURCE_SEMANTICS_UNRESOLVED":
            raise ValueError("TORAX method value FAIR-MAST sidecar cannot be promoted")
        if not self.frozen:
            raise ValueError("TORAX method value experiment config must be frozen before execution")

    def tasks_for(self, role: ToraxMethodValueTaskRole) -> tuple[ToraxMethodValueTaskSpec, ...]:
        return tuple(value for value in self.tasks if value.role is role)


@dataclass(frozen=True, slots=True)
class ToraxMethodValuePhaseSpec(CanonicalRecord):
    """One frozen TORAX method value execution subject with an exact outcome-access ceiling."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-phase-spec'

    phase_id: str
    phase: ToraxMethodValuePhase
    experiment: ObjectIdentity
    allowed_task_roles: tuple[ToraxMethodValueTaskRole, ...]
    maximum_episode_launches: int
    outcome_access: OutcomeAccess
    frozen: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.phase_id, field_name="phase_id")
        if tuple(sorted(set(self.allowed_task_roles))) != self.allowed_task_roles:
            raise ValueError("TORAX method value phase task roles must be sorted and unique")
        expected = {
            ToraxMethodValuePhase.CANARY: (
                (ToraxMethodValueTaskRole.CANARY_EXCLUDED,),
                6,
                OutcomeAccess.OUTCOME_BLIND,
            ),
            ToraxMethodValuePhase.DEVELOPMENT: (
                (
                    ToraxMethodValueTaskRole.DEVELOPMENT_CANDIDATE,
                    ToraxMethodValueTaskRole.DEVELOPMENT_COMMON,
                ),
                96,
                OutcomeAccess.OUTCOME_BLIND,
            ),
            ToraxMethodValuePhase.PROTECTED_REFERENCE: (
                (ToraxMethodValueTaskRole.PROTECTED_REFERENCE,),
                48,
                OutcomeAccess.EVALUATION_SEALED,
            ),
            ToraxMethodValuePhase.PROSPECTIVE_CONFIRMATION: (
                (ToraxMethodValueTaskRole.PROSPECTIVE_CONFIRMATION,),
                64,
                OutcomeAccess.EVALUATION_SEALED,
            ),
        }[self.phase]
        if (
            self.allowed_task_roles,
            self.maximum_episode_launches,
            self.outcome_access,
        ) != expected:
            raise ValueError("TORAX method value phase execution boundary differs from the freeze")
        if not self.frozen:
            raise ValueError("TORAX method value phase must be frozen before authority issue")


@dataclass(frozen=True, slots=True)
class ToraxMethodValueScientificApproval(CanonicalRecord):
    """Owner approval for one exact, non-actuating frozen TORAX method value phase."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-scientific-approval'

    approval_id: str
    phase: ToraxMethodValuePhase
    subject: ObjectIdentity
    owner: ObjectIdentity
    authorization_basis_sha256: str
    passed_gate_ids: tuple[str, ...]
    issued_at_utc: str
    codex_or_chat_is_approver_or_issuer: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        from empirical_lawhood.kernel.serialization import validate_sha256

        validate_stable_id(self.approval_id, field_name="approval_id")
        validate_sha256(
            self.authorization_basis_sha256,
            field_name="authorization_basis_sha256",
        )
        require_sorted_unique_strings(
            self.passed_gate_ids,
            field_name="passed_gate_ids",
            allow_empty=False,
        )
        parse_utc_timestamp(self.issued_at_utc, field_name="issued_at_utc")
        if self.codex_or_chat_is_approver_or_issuer:
            raise ValueError("Codex/chat cannot approve or issue TORAX method value authority")
        expected_access = (
            OutcomeAccess.OUTCOME_BLIND
            if self.phase in {ToraxMethodValuePhase.CANARY, ToraxMethodValuePhase.DEVELOPMENT}
            else OutcomeAccess.EVALUATION_SEALED
        )
        if self.outcome_access != expected_access:
            raise ValueError("TORAX method value approval outcome access differs from its phase")


@dataclass(frozen=True, slots=True)
class ToraxMethodValueNativeExecutionBatch(CanonicalRecord):
    """Externally persisted native episode index for one exact TORAX method value phase."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-native-execution-batch'

    batch_id: str
    experiment: ObjectIdentity
    phase: ObjectIdentity
    task_ids: tuple[str, ...]
    preparation_identities: tuple[ObjectIdentity, ...]
    trajectory_identities: tuple[ObjectIdentity, ...]
    episodes: tuple[NativeToraxEpisode, ...]
    launch_count: int
    completed_count: int
    terminal: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.batch_id, field_name="batch_id")
        require_sorted_unique_strings(self.task_ids, field_name="task_ids", allow_empty=False)
        for name, values in (
            ("preparation_identities", self.preparation_identities),
            ("trajectory_identities", self.trajectory_identities),
        ):
            require_sorted_unique_ids(values, attribute="object_id", field_name=name)
        require_sorted_unique_ids(self.episodes, attribute="episode_id", field_name="episodes")
        if self.launch_count != len(self.episodes):
            raise ValueError("TORAX method value native batch launch count differs from its episode roster")
        observed_completed = sum(value.disposition.value == "COMPLETE" for value in self.episodes)
        if self.completed_count != observed_completed:
            raise ValueError("TORAX method value native batch completion count differs from episodes")
        if not self.terminal:
            raise ValueError("TORAX method value native execution batches are published only at terminal phase")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("TORAX method value native execution batch has the wrong access ceiling")


@dataclass(frozen=True, slots=True)
class ToraxMethodValueCanaryAdapterFailure(CanonicalRecord):
    """Terminal record for an excluded canary stopped in native extraction."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-canary-adapter-failure'

    failure_id: str
    phase: ObjectIdentity
    task_id: str
    action_id: str
    view_id: str
    solver_completed: bool
    launch_count: int
    exception_type: str
    reason_codes: tuple[str, ...]
    terminal: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("failure_id", "task_id", "action_id", "view_id", "exception_type"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )
        if not self.solver_completed or self.launch_count != 1 or not self.terminal:
            raise ValueError("TORAX method value excluded canary failure accounting differs from observation")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("TORAX method value canary failure must remain development-visible")


@dataclass(frozen=True, slots=True)
class ToraxMethodValueAcquisitionRoundDecision(CanonicalRecord):
    """One outcome-cutoff-bound arm query selected by a released scorer."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-acquisition-round-decision'

    decision_id: str
    arm: ToraxMethodValueArm
    method: ToraxMethodValueAcquisitionMethod
    round_number: int
    already_acquired_task_ids: tuple[str, ...]
    eligible_task_ids: tuple[str, ...]
    selected_task_id: str
    scorer_output: ObjectIdentity
    rcj_witness_qualified: bool
    rcj_contacted: bool
    information_cutoff_id: str
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("decision_id", "selected_task_id", "information_cutoff_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.already_acquired_task_ids,
            field_name="already_acquired_task_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.eligible_task_ids,
            field_name="eligible_task_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.round_number not in {1, 2}:
            raise ValueError("TORAX method value acquisition decision lies outside the two-round graph")
        if (
            self.selected_task_id not in self.eligible_task_ids
            or self.selected_task_id in self.already_acquired_task_ids
        ):
            raise ValueError("TORAX method value acquisition decision selected an illegal task")
        if self.arm is ToraxMethodValueArm.WITNESS_GATED_IO:
            if self.method is not ToraxMethodValueAcquisitionMethod.RCJ_IO_FALLBACK:
                raise ValueError("TORAX method value witness-gated I/O arm decision must retain its intention-to-treat method")
            if self.rcj_witness_qualified or self.rcj_contacted:
                if self.reason_codes:
                    raise ValueError("contacted TORAX method value RCJ decision cannot carry fallback reasons")
            elif self.reason_codes != ("RCJ_WITNESS_UNQUALIFIED_IO_FALLBACK",):
                raise ValueError("unqualified TORAX method value RCJ decision lacks the exact fallback reason")
        elif self.rcj_witness_qualified or self.rcj_contacted or self.reason_codes:
            raise ValueError("non-RCJ TORAX method value arm carries RCJ or refusal state")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("TORAX method value acquisition decisions require development-visible evidence")


@dataclass(frozen=True, slots=True)
class ToraxMethodValueBackboneProjectionConfig(CanonicalRecord):
    """Frozen TORAX method value translation choices at the released manifest/R6 seam."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-backbone-projection-config'
    )

    config_id: str
    arm: ToraxMethodValueArm
    chart_id: str
    calibration_task_ids: tuple[str, ...]
    held_out_task_ids: tuple[str, ...]
    denominator_value_ids: tuple[str, ...]
    history_value_ids: tuple[str, ...]
    action_value_ids: tuple[str, ...]
    receiver_value_ids: tuple[str, ...]
    include_only_complete_native_episodes: bool
    action_delivery_lowering_supported: bool
    reference_tolerance_ev: Decimal
    frozen: bool

    def __post_init__(self) -> None:
        for name in ("config_id", "chart_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "calibration_task_ids",
            "held_out_task_ids",
            "denominator_value_ids",
            "history_value_ids",
            "action_value_ids",
            "receiver_value_ids",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        validate_decimal(
            self.reference_tolerance_ev,
            field_name="reference_tolerance_ev",
            minimum=Decimal(0),
        )
        if not self.include_only_complete_native_episodes:
            raise ValueError("TORAX method value backbone projection cannot score invalid native episodes")
        if self.action_delivery_lowering_supported:
            raise ValueError("response-method compatibility dataset cannot lower current exact ActionWord identity")
        if not self.frozen:
            raise ValueError("TORAX method value backbone projection config must be frozen")


@dataclass(frozen=True, slots=True)
class ToraxMethodValueObservationActionBinding(CanonicalRecord):
    "Additive exact action map retained beside the response-method compatibility dataset."

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-observation-action-binding'
    )

    binding_id: str
    observation_id: str
    delivery: ObservationActionDeliveryBinding

    def __post_init__(self) -> None:
        validate_stable_id(self.binding_id, field_name="binding_id")
        validate_stable_id(self.observation_id, field_name="observation_id")


@dataclass(frozen=True, slots=True)
class ToraxMethodValueResponseMethodActionCompatibilityReceipt(CanonicalRecord):
    """Honest compatibility receipt for the released unlinked R6 lowering seam."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-response-method-action-compatibility-receipt'

    receipt_id: str
    manifest: ObjectIdentity
    bindings: tuple[ToraxMethodValueObservationActionBinding, ...]
    exact_action_words_retained: bool
    response_method_observation_links_retained: bool
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_ids(
            self.bindings,
            attribute="binding_id",
            field_name="bindings",
        )
        if not self.bindings or not self.exact_action_words_retained:
            raise ValueError("TORAX method value compatibility receipt must retain every exact action map")
        if self.response_method_observation_links_retained:
            raise ValueError("response-method compatibility dataset cannot claim linked current ActionWord identity")
        require_sorted_unique_strings(
            self.reason_codes,
            field_name="reason_codes",
            allow_empty=False,
        )


@dataclass(frozen=True, slots=True)
class ToraxMethodValueBackboneStageDisposition(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-backbone-stage-disposition'
    )

    disposition_id: str
    stage: PublicRouteConformanceStage
    status: ToraxMethodValueBackboneStageStatus
    record: ObjectIdentity | None
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.disposition_id, field_name="disposition_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is ToraxMethodValueBackboneStageStatus.COMPLETE:
            if self.record is None or self.reason_codes:
                raise ValueError("complete TORAX method value backbone stage requires a record and no reasons")
        elif self.status is ToraxMethodValueBackboneStageStatus.OBSTRUCTED:
            if self.record is None or not self.reason_codes:
                raise ValueError("obstructed TORAX method value backbone stage requires record and reasons")
        elif self.record is not None or not self.reason_codes:
            raise ValueError("condition-false TORAX method value stage requires reasons and no record")


@dataclass(frozen=True, slots=True)
class ToraxMethodValueBackboneArmReceipt(CanonicalRecord):
    """Terminal per-arm receipt over the real consolidated route and its stop."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/adapters/simulators/torax-method-value/torax-method-value-backbone-arm-receipt'

    receipt_id: str
    experiment: ObjectIdentity
    arm: ToraxMethodValueArm
    system: ObjectIdentity
    projection_config: ObjectIdentity
    manifest: ObjectIdentity
    projection: ObjectIdentity
    action_compatibility: ObjectIdentity
    dataset: ObjectIdentity
    candidate_family_assessment: ObjectIdentity
    qualification_profile_assessment: ObjectIdentity
    qualification_result: ObjectIdentity
    qualification_batch: ObjectIdentity
    atlas_obstruction: ObjectIdentity
    selected_task_ids: tuple[str, ...]
    native_episode_count: int
    projected_episode_count: int
    excluded_invalid_episode_count: int
    scientific_status: ScientificStatus
    highest_supported_rung: EvidenceRung | None
    stages: tuple[ToraxMethodValueBackboneStageDisposition, ...]
    terminal: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_strings(
            self.selected_task_ids,
            field_name="selected_task_ids",
            allow_empty=False,
        )
        if self.native_episode_count <= 0:
            raise ValueError("TORAX method value backbone receipt requires native episodes")
        if (
            self.projected_episode_count <= 0
            or self.projected_episode_count + self.excluded_invalid_episode_count
            != self.native_episode_count
        ):
            raise ValueError("TORAX method value backbone episode accounting differs from native evidence")
        if tuple(value.stage for value in self.stages) != PUBLIC_ROUTE_STAGE_ORDER:
            raise ValueError("TORAX method value backbone receipt must cover the public stage order")
        if not self.terminal or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("TORAX method value development backbone receipt must be terminal and visible")


def _power_profile(
    core: Decimal,
    edge: Decimal,
    exponent: Decimal,
    radial_coordinates: tuple[Decimal, ...],
) -> tuple[Decimal, ...]:
    with localcontext() as context:
        context.prec = 28
        return tuple(
            edge + (core - edge) * (Decimal(1) - radius**exponent) for radius in radial_coordinates
        )


def materialize_preparation(
    spec: ToraxMethodValueExperimentSpec,
    task: ToraxMethodValueTaskSpec,
) -> NativeToraxPreparation:
    """Pure outcome-blind TORAX method value task-to-native-preparation translation."""

    if task not in spec.tasks:
        raise ValueError("TORAX method value task is outside the frozen experiment roster")
    te = _power_profile(
        task.core_electron_temperature_ev,
        task.edge_electron_temperature_ev,
        task.temperature_profile_exponent,
        spec.radial_coordinates,
    )
    ti = tuple(value * task.ion_to_electron_temperature_ratio for value in te)
    ne = _power_profile(
        task.core_electron_density_m3,
        task.edge_electron_density_m3,
        Decimal(1),
        spec.radial_coordinates,
    )
    return NativeToraxPreparation(
        preparation_id=f"preparation.{task.task_id}",
        radial_coordinates=spec.radial_coordinates,
        electron_temperature_ev=te,
        ion_temperature_ev=ti,
        electron_density_m3=ne,
        plasma_current_a=task.plasma_current_a,
        main_ion=spec.main_ion,
        impurity=spec.impurity,
        zeff=spec.zeff,
        major_radius_m=spec.major_radius_m,
        minor_radius_m=spec.minor_radius_m,
        toroidal_field_t=spec.toroidal_field_t,
        elongation_lcfs=spec.elongation_lcfs,
        source_radial_location=task.source_radial_location,
        source_width=task.source_width,
        electron_heat_fraction=spec.electron_heat_fraction,
        absorbed_power_fraction=spec.absorbed_power_fraction,
        chi_i_m2_s=task.chi_i_m2_s,
        chi_e_m2_s=task.chi_e_m2_s,
        particle_diffusivity_m2_s=spec.particle_diffusivity_m2_s,
        particle_convection_m_s=spec.particle_convection_m_s,
        maximum_core_temperature_ev=task.maximum_core_temperature_ev,
        assumption_ids=(
            "circular-geometry-testing-assumption",
            "constant-transport-testing-assumption",
            "fixed-current-density-horizon",
            "generic-gaussian-heat-proxy",
        ),
    )


__all__ = [
    'ToraxMethodValueArm',
    'ToraxMethodValueAcquisitionMethod',
    'ToraxMethodValueAcquisitionRoundDecision',
    'ToraxMethodValueBackboneArmReceipt',
    'ToraxMethodValueBackboneProjectionConfig',
    'ToraxMethodValueBackboneStageDisposition',
    'ToraxMethodValueBackboneStageStatus',
    'ToraxMethodValueCanaryAdapterFailure',
    'ToraxMethodValueExperimentSpec',
    'ToraxMethodValueObservationActionBinding',
    'ToraxMethodValuePhase',
    'ToraxMethodValuePhaseSpec',
    'ToraxMethodValueResponseMethodActionCompatibilityReceipt',
    'ToraxMethodValueScientificApproval',
    'ToraxMethodValueNativeExecutionBatch',
    'ToraxMethodValueTaskRole',
    'ToraxMethodValueTaskSpec',
    'materialize_preparation',
]

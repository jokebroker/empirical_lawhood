"""Outcome-blind TORAX source/runtime qualification for the flagship.

The records in this module bind the finite package/runtime choice and construct
the predevelopment JIT census.  They contain no simulator response and grant no
execution or reveal authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from hashlib import sha256
from typing import ClassVar

from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    canonical_json_bytes,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_nonempty,
    validate_decimal,
    validate_semantic_version,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.runtime.execution_envelope import JitCellSignatureProjection, JitExpressionFieldValue, PredevelopmentJitSignatureCensus, jit_graph_signature_sha256


class ToraxPinDisposition(StrEnum):
    SELECTED = "SELECTED"
    INELIGIBLE = "INELIGIBLE"


@dataclass(frozen=True, slots=True)
class TokamakControlSourceBundleSpec(CanonicalRecord):
    """Finite public/archive source subject to which acquisition authority binds."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/tokamak-control-source-bundle-spec'

    source_bundle_id: str
    fair_mast_source_id: str
    fair_mast_allowed_https_origins: tuple[str, ...]
    fair_mast_campaign_ids: tuple[str, ...]
    gymtorax_candidates: tuple[str, ...]
    torax_candidates: tuple[str, ...]
    unpinned_fallback_forbidden: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name in ("source_bundle_id", "fair_mast_source_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        for name in (
            "fair_mast_allowed_https_origins",
            "fair_mast_campaign_ids",
            "gymtorax_candidates",
            "torax_candidates",
        ):
            require_sorted_unique_strings(
                getattr(self, name),
                field_name=name,
                allow_empty=False,
            )
        if (
            self.fair_mast_allowed_https_origins != ("https://mastapp.site",)
            or self.fair_mast_campaign_ids != ("M7", "M8", "M9")
            or self.gymtorax_candidates != ("1.1.1",)
            or self.torax_candidates != ("1.4.2", "1.4.3")
            or not self.unpinned_fallback_forbidden
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("flagship source bundle differs from the finite authored source set")


@dataclass(frozen=True, slots=True)
class RuntimeDependencyIdentity(CanonicalRecord):
    """One resolved installed distribution in the simulator denominator."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/runtime-dependency-identity'

    distribution_id: str
    version: str
    installed_files_sha256: str
    metadata_sha256: str

    def __post_init__(self) -> None:
        validate_stable_id(self.distribution_id, field_name="distribution_id")
        validate_semantic_version(self.version)
        validate_sha256(self.installed_files_sha256, field_name="installed_files_sha256")
        validate_sha256(self.metadata_sha256, field_name="metadata_sha256")


@dataclass(frozen=True, slots=True)
class ToraxPinCandidate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/torax-pin-candidate'

    candidate_id: str
    gymtorax_version: str
    torax_version: str
    disposition: ToraxPinDisposition
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.candidate_id, field_name="candidate_id")
        validate_semantic_version(self.gymtorax_version)
        validate_semantic_version(self.torax_version)
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.gymtorax_version != "1.1.1" or self.torax_version not in {"1.4.2", "1.4.3"}:
            raise ValueError("TORAX candidate lies outside the plan's finite pin set")
        if self.disposition is ToraxPinDisposition.SELECTED and self.reason_codes:
            raise ValueError("selected TORAX candidate retains an ineligibility reason")
        if self.disposition is ToraxPinDisposition.INELIGIBLE and not self.reason_codes:
            raise ValueError("ineligible TORAX candidate lacks a reason")


@dataclass(frozen=True, slots=True)
class ToraxRuntimeQualification(CanonicalRecord):
    """Actual package/backend/solver closure established before development."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/torax-runtime-qualification'

    qualification_id: str
    dependencies: tuple[RuntimeDependencyIdentity, ...]
    python_version: str
    backend: str
    device_count: int
    x64_enabled: bool
    gym_environment_id: str
    solver_type: str
    implicit_solver_type: str
    differentiates_through_torax_run: bool
    requested_accepted_applied_realized_observable: bool
    reset_and_one_step_smoke_passed: bool
    finite_observation_passed: bool
    smoke_receipt_sha256: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.qualification_id, field_name="qualification_id")
        require_sorted_unique_ids(
            self.dependencies,
            attribute="distribution_id",
            field_name="dependencies",
        )
        validate_nonempty(self.python_version, field_name="python_version")
        validate_stable_id(self.backend, field_name="backend")
        validate_stable_id(self.gym_environment_id, field_name="gym_environment_id")
        validate_stable_id(self.solver_type, field_name="solver_type")
        validate_stable_id(self.implicit_solver_type, field_name="implicit_solver_type")
        validate_sha256(self.smoke_receipt_sha256, field_name="smoke_receipt_sha256")
        versions = {value.distribution_id: value.version for value in self.dependencies}
        if versions.get("gymtorax") != "1.1.1" or versions.get("torax") != "1.4.2":
            raise ValueError("runtime qualification differs from the rule-selected pin")
        required = {"gymtorax", "jax", "jaxlib", "numpy", "scipy", "torax"}
        if set(versions) != required:
            raise ValueError("runtime qualification dependency closure is incomplete")
        if (
            self.backend != "cpu"
            or self.device_count < 1
            or not self.x64_enabled
            or self.gym_environment_id != "gymtorax.test-v0"
            or self.solver_type != "linear"
            or self.implicit_solver_type != "thomas"
            or self.differentiates_through_torax_run
            or not self.requested_accepted_applied_realized_observable
            or not self.reset_and_one_step_smoke_passed
            or not self.finite_observation_passed
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("TORAX runtime qualification is not source/runtime complete")


@dataclass(frozen=True, slots=True)
class ToraxRuntimeSmokeObservation(CanonicalRecord):
    """One excluded reset/step observation proving the selected wrapper path."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/torax-runtime-smoke-observation'

    smoke_id: str
    environment_id: str
    requested_ip_a: Decimal
    accepted_ip_a: Decimal
    applied_ip_a: Decimal
    realized_ip_a: Decimal
    initial_time_s: Decimal
    final_time_s: Decimal
    reset_elapsed_s: Decimal
    step_elapsed_s: Decimal
    action_clipped: bool
    finite_observation_count: int
    structurally_missing_observation_ids: tuple[str, ...]
    smoke_attempt_count: int
    cache_reused_after_validation_refusal: bool
    simulation_success: bool
    excluded_from_scientific_endpoints: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.smoke_id, field_name="smoke_id")
        validate_stable_id(self.environment_id, field_name="environment_id")
        for name in (
            "requested_ip_a",
            "accepted_ip_a",
            "applied_ip_a",
            "realized_ip_a",
            "initial_time_s",
            "final_time_s",
            "reset_elapsed_s",
            "step_elapsed_s",
        ):
            validate_decimal(getattr(self, name), field_name=name)
        require_sorted_unique_strings(
            self.structurally_missing_observation_ids,
            field_name="structurally_missing_observation_ids",
        )
        permitted_missing = tuple(
            sorted(
                (
                    "scalars.rho_q_2_1_first",
                    "scalars.rho_q_2_1_second",
                    "scalars.rho_q_3_1_first",
                    "scalars.rho_q_3_1_second",
                    "scalars.rho_q_3_2_first",
                    "scalars.rho_q_3_2_second",
                )
            )
        )
        if (
            self.environment_id != "gymtorax.test-v0"
            or self.requested_ip_a != self.accepted_ip_a
            or self.accepted_ip_a != self.applied_ip_a
            or self.applied_ip_a != self.realized_ip_a
            or self.final_time_s <= self.initial_time_s
            or self.reset_elapsed_s < 0
            or self.step_elapsed_s < 0
            or self.action_clipped
            or self.finite_observation_count <= 0
            or self.structurally_missing_observation_ids != permitted_missing
            or self.smoke_attempt_count not in {1, 4}
            or self.cache_reused_after_validation_refusal != (self.smoke_attempt_count == 4)
            or not self.simulation_success
            or not self.excluded_from_scientific_endpoints
            or self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
        ):
            raise ValueError("TORAX runtime smoke did not prove the exact excluded wrapper path")


@dataclass(frozen=True, slots=True)
class ToraxPinApplicabilityDecision(CanonicalRecord):
    """Rule-derived 1.4.2 selection; no favorable response may enter."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/torax-pin-applicability-decision'

    decision_id: str
    candidates: tuple[ToraxPinCandidate, ...]
    selected_runtime: ObjectIdentity
    selected_candidate_id: str
    all_selected_solvers_linear: bool
    any_newton_solver_used: bool
    any_autodiff_crosses_torax_run: bool
    torax_1_4_3_newton_gradient_fix_applicability: str
    post_development_fallback_forbidden: bool
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.decision_id, field_name="decision_id")
        validate_stable_id(self.selected_candidate_id, field_name="selected_candidate_id")
        require_sorted_unique_ids(
            self.candidates, attribute="candidate_id", field_name="candidates"
        )
        selected = tuple(
            value
            for value in self.candidates
            if value.disposition is ToraxPinDisposition.SELECTED
        )
        if (
            len(self.candidates) != 2
            or len(selected) != 1
            or selected[0].candidate_id != self.selected_candidate_id
            or selected[0].torax_version != "1.4.2"
            or self.selected_runtime.object_schema != ToraxRuntimeQualification.SCHEMA
            or not self.all_selected_solvers_linear
            or self.any_newton_solver_used
            or self.any_autodiff_crosses_torax_run
            or self.torax_1_4_3_newton_gradient_fix_applicability != "NOT_APPLICABLE"
            or not self.post_development_fallback_forbidden
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError(
                "TORAX pin decision does not follow the predeclared applicability rule"
            )


@dataclass(frozen=True, slots=True)
class JitExpressionFieldProbe(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/jit-expression-field-probe'

    probe_id: str
    selected_runtime: ObjectIdentity
    expression_changing_field_ids: tuple[str, ...]
    runtime_dynamic_field_ids: tuple[str, ...]
    complete_development_cell_count: int
    maximum_distinct_signatures: int

    def __post_init__(self) -> None:
        validate_stable_id(self.probe_id, field_name="probe_id")
        require_sorted_unique_strings(
            self.expression_changing_field_ids,
            field_name="expression_changing_field_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.runtime_dynamic_field_ids,
            field_name="runtime_dynamic_field_ids",
            allow_empty=False,
        )
        if set(self.expression_changing_field_ids) & set(self.runtime_dynamic_field_ids):
            raise ValueError("JIT field probe gives a field two classifications")
        if self.complete_development_cell_count != 1029 or self.maximum_distinct_signatures != 149:
            raise ValueError("JIT field probe differs from the complete predevelopment graph")


@dataclass(frozen=True, slots=True)
class JaxPersistentCacheSpec(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/tokamak-prospective-control/jax-persistent-cache-spec'

    cache_id: str
    operator_storage_profile: ObjectIdentity
    relative_locator: str
    initialized_empty: bool
    development_and_protected_identity_stable: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.cache_id, field_name="cache_id")
        if self.relative_locator != "studies/tokamak-response-control/scratch/jax-cache":
            raise ValueError("JAX cache is outside the flagship guarded namespace")
        if not self.initialized_empty or not self.development_and_protected_identity_stable:
            raise ValueError("JAX cache was not frozen before development")


_STATIC_FIELDS = tuple(
    sorted(
        (
            "static.archetype",
            "static.closure",
            "static.corrector",
            "static.geometry",
            "static.grid",
            "static.member",
            "static.precision",
            "static.solver",
            "static.source-model",
            "static.timestep",
            "static.wrapper",
        )
    )
)
_DYNAMIC_FIELDS = tuple(
    sorted(("dynamic.action", "dynamic.preparation", "dynamic.seed", "dynamic.state"))
)


def _value_digest(value: str) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _projection(
    *,
    cell_id: str,
    archetype: str,
    member: str,
    grid: str,
    timestep: str,
    corrector: str,
    action: str,
    preparation: str,
    state: str,
) -> JitCellSignatureProjection:
    values = {
        "dynamic.action": action,
        "dynamic.preparation": preparation,
        "dynamic.seed": preparation,
        "dynamic.state": state,
        "static.archetype": archetype,
        "static.closure": "constant-plus-bohm-gyrobohm",
        "static.corrector": corrector,
        "static.geometry": "circular-analytic",
        "static.grid": grid,
        "static.member": member,
        "static.precision": "float64-cpu",
        "static.solver": "linear-thomas",
        "static.source-model": "gaussian-heat-current-active-radiation",
        "static.timestep": timestep,
        "static.wrapper": "gymtorax-1.1.1-torax-1.4.2",
    }
    fields = tuple(
        JitExpressionFieldValue(
            field_id=field_id,
            canonical_value_sha256=_value_digest(values[field_id]),
            expression_changing=field_id in _STATIC_FIELDS,
        )
        for field_id in sorted(values)
    )
    return JitCellSignatureProjection(
        cell_id=cell_id,
        fields=fields,
        signature_sha256=jit_graph_signature_sha256(fields),
    )


def build_predevelopment_jit_signature_census(
    *,
    probe: JitExpressionFieldProbe,
    cache: JaxPersistentCacheSpec,
) -> PredevelopmentJitSignatureCensus:
    """Build the plan's exact 192+432+405 outcome-free cell census."""

    projections: list[JitCellSignatureProjection] = []
    archetypes = ("receiver-sink", "hidden-history", "action-order", "member-hold")
    actions = ("a0", "a1", "a2", "hold")
    members = ("primary", "challenger")
    for archetype in archetypes:
        for occurrence in range(3):
            task = f"{archetype}.{occurrence}"
            for preparation in range(2):
                for action in actions:
                    for member in members:
                        cell_id = f"jit.generated-development.{task}.p{preparation}.{action}.{member}"
                        projections.append(
                            _projection(
                                cell_id=cell_id,
                                archetype=archetype,
                                member=member,
                                grid="central",
                                timestep="central",
                                corrector="central",
                                action=action,
                                preparation=f"p{preparation}",
                                state=task,
                            )
                        )
    for state_index in range(24):
        state = f"development-state-{state_index:02d}"
        for action in ("down", "hold", "up"):
            for theta in ("theta-low", "theta-central", "theta-high"):
                for view in ("nominal", "sensitivity"):
                    member = f"{theta}.{view}"
                    projections.append(
                        _projection(
                            cell_id=f"jit.mapped-development.{state}.{action}.{theta}.{view}",
                            archetype="mapped",
                            member=member,
                            grid="central",
                            timestep="central",
                            corrector="central",
                            action=action,
                            preparation=state,
                            state=state,
                        )
                    )
    sentinel_preparations = ("mapped", "receiver-sink", "hidden-history", "action-order", "member-hold")
    for sentinel_preparation in sentinel_preparations:
        for grid in ("low", "central", "high"):
            for timestep in ("low", "central", "high"):
                for corrector in ("low", "central", "high"):
                    for action in ("down", "hold", "up"):
                        cell_id = (
                            f"jit.sentinel.{sentinel_preparation}.{grid}.{timestep}."
                            f"{corrector}.{action}"
                        )
                        projections.append(
                            _projection(
                                cell_id=cell_id,
                                archetype=f"sentinel-{sentinel_preparation}",
                                member="central",
                                grid=grid,
                                timestep=timestep,
                                corrector=corrector,
                                action=action,
                                preparation=sentinel_preparation,
                                state="source-qualified-central",
                            )
                        )
    ordered = tuple(sorted(projections, key=lambda value: value.cell_id))
    if len(ordered) != 1029 or len({value.signature_sha256 for value in ordered}) != 149:
        raise ValueError("predevelopment JIT graph does not realize the exact 1029/149 census")
    return PredevelopmentJitSignatureCensus(
        census_id='tokamak-control-predevelopment-jit-signature-census',
        expression_field_probe=ObjectIdentity.from_record(probe.probe_id, probe),
        persistent_cache=ObjectIdentity.from_record(cache.cache_id, cache),
        expression_changing_field_ids=_STATIC_FIELDS,
        runtime_dynamic_field_ids=_DYNAMIC_FIELDS,
        cell_projections=ordered,
        maximum_distinct_signatures=149,
    )


__all__ = [
    'TokamakControlSourceBundleSpec',
    'JaxPersistentCacheSpec',
    'JitExpressionFieldProbe',
    'RuntimeDependencyIdentity',
    'ToraxPinApplicabilityDecision',
    'ToraxPinCandidate',
    'ToraxPinDisposition',
    'ToraxRuntimeQualification',
    'ToraxRuntimeSmokeObservation',
    'build_predevelopment_jit_signature_census',
]

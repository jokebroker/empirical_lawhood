"""Truth-known, zero-native-read conformance for the empirical consumer."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.adapters.composition.multi_world_response_study.contracts.platform_binding import MultiWorldStudyConsumerPlatformBinding
from empirical_lawhood.adapters.composition.multi_world_response_study.contracts.science import MultiWorldStudyPayloadNamespaceBinding, MultiWorldStudyPayloadOwner, MultiWorldStudyScientificBinding, SyntheticMultiWorldStudyParentConformance
from empirical_lawhood.adapters.physical.mast_archive_response_qualification import ArchiveLawSpec, build_archive_law_spec
from empirical_lawhood.adapters.simulators.tokamak_prospective_control.contracts import ActionAlgebraCoveragePlan, ActionSegmentPort, DenominatorAuditSpec, GeneratedActionSegment, GeneratedActionWordSpec, GeneratedArmBinding, GeneratedArmKind, GeneratedChildScientificConfig, GeneratedReceiverDimension, LawToActionTaskSpec, OutsideSupportCalibrationPlan, PreparationOccurrence, PreparationPoolRole, TaskArchetype, TaskPreparationPlan, validate_generated_task_roster
from empirical_lawhood.adapters.simulators.mast_torax_state_transport.contracts.science import MappedActionChart, MappedActionWord, MappedChildScientificConfig, MappedFieldOriginKind, MappedFieldOrigin, MappedMethodBinding, MappedNumericalViewKind, MappedReceiverSpec, MappedToraxAssumptionMember, MappedToraxEnsembleSpec, MappedToraxNumericalView, validate_mapped_member_product
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)
from empirical_lawhood.planning.multi_world_study import MorphismNegativeControlKind
from empirical_lawhood.runtime.multi_world_study import PropertyMorphismDisposition


@dataclass(frozen=True, slots=True)
class SyntheticTaskRoster(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/synthetic-task-roster'

    roster_id: str
    tasks: tuple[LawToActionTaskSpec, ...]
    architecture_conformance_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        validate_generated_task_roster(self.tasks, selected_count=8)
        if not self.architecture_conformance_only:
            raise ValueError("synthetic task roster cannot become empirical evidence")


@dataclass(frozen=True, slots=True)
class SyntheticPreparationRoster(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/composition/multi-world-response-study/synthetic-preparation-roster'

    roster_id: str
    plans: tuple[TaskPreparationPlan, ...]
    architecture_conformance_only: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.roster_id, field_name="roster_id")
        require_sorted_unique_ids(self.plans, attribute="task_id", field_name="plans")
        if len(self.plans) != 8 or not self.architecture_conformance_only:
            raise ValueError("synthetic preparation roster is not a nonempirical C8")


def _identity(object_id: str, schema: str) -> ObjectIdentity:
    return ObjectIdentity(
        object_id=object_id,
        object_schema=schema,
        object_version="1.0.0",
        object_fingerprint=hashlib.sha256(f"{object_id}|{schema}".encode()).hexdigest(),
    )


def _mapped_config() -> MappedChildScientificConfig:
    field_origins = (
        MappedFieldOrigin(
            field_binding_id="mapping-field.electron-density",
            archive_field_id="electron-density-core-profile",
            torax_field_id="torax-electron-density-profile",
            origin=MappedFieldOriginKind.OBSERVED,
            archive_unit="m^-3",
            torax_unit="m^-3",
            unit_transform_id="identity-si-density",
            lower_bound=None,
            upper_bound=None,
            acceptance_predicate_id="density-profile-valid",
            archive_outcome_used=False,
        ),
        MappedFieldOrigin(
            field_binding_id="mapping-field.transport-coefficient",
            archive_field_id="transport-unidentified",
            torax_field_id="torax-transport-coefficient",
            origin=MappedFieldOriginKind.ASSUMED,
            archive_unit="m^2/s",
            torax_unit="m^2/s",
            unit_transform_id="identity-si-transport",
            lower_bound=Decimal("0.5"),
            upper_bound=Decimal("2.0"),
            acceptance_predicate_id="transport-interval-qualified",
            archive_outcome_used=False,
        ),
    )
    assumed = ("mapping-field.transport-coefficient",)
    assumptions = (
        MappedToraxAssumptionMember(
            member_id="theta.central",
            quantile=Decimal("0.50"),
            assumed_field_binding_ids=assumed,
            joint_vector_required=True,
        ),
        MappedToraxAssumptionMember(
            member_id="theta.high",
            quantile=Decimal("0.80"),
            assumed_field_binding_ids=assumed,
            joint_vector_required=True,
        ),
        MappedToraxAssumptionMember(
            member_id="theta.low",
            quantile=Decimal("0.20"),
            assumed_field_binding_ids=assumed,
            joint_vector_required=True,
        ),
    )
    views = (
        MappedToraxNumericalView(
            view_id="view.challenger",
            kind=MappedNumericalViewKind.CHALLENGER,
            radial_cell_count=32,
            timestep_seconds=Decimal("0.0005"),
            corrector_depth=2,
            selection_rule_id="truth-known-interaction-maximin",
        ),
        MappedToraxNumericalView(
            view_id="view.primary",
            kind=MappedNumericalViewKind.PRIMARY,
            radial_cell_count=24,
            timestep_seconds=Decimal("0.001"),
            corrector_depth=1,
            selection_rule_id="source-qualified-central",
        ),
    )
    ensemble = MappedToraxEnsembleSpec(
        ensemble_id="mapped-ensemble.truth-known-c8",
        field_origins=field_origins,
        assumption_members=assumptions,
        numerical_views=views,
        complete_cartesian_product_required=True,
        archive_law_is_member=False,
    )
    validate_mapped_member_product(
        ensemble=ensemble,
        observed_member_ids=ensemble.member_ids,
    )
    shared = (
        "controller.action-aware-nested-controller-use",
        "qualification.action-preparation-recurrence",
        "response-law.admission_receipts",
        "response-law.controller_compile",
        "response-law.law_qualification",
        "runtime.deadline-free-resource-recovery",
    )
    method_rows = (
        ("arm-a", "rcj-io", 'empirical-lawhood/composition/archive-to-tokamak-response-transport/witness-gated-input-output/method-config'),
        ("arm-b", "io-only", 'empirical-lawhood/composition/archive-to-tokamak-response-transport/input-output-only/method-config'),
        ("arm-c", "finite-mpc", 'empirical-lawhood/composition/archive-to-tokamak-response-transport/finite-model-predictive-control/method-config'),
        (
            "arm-d",
            "robust-maximin",
            'empirical-lawhood/composition/archive-to-tokamak-response-transport/robust-maximin-control/method-config',
        ),
    )
    methods = tuple(
        sorted(
            (
                MappedMethodBinding(
                    binding_id=f"mapped-method.{arm_id}",
                    arm_id=arm_id,
                    method_family_id=family,
                    method_config_schema=schema,
                    method_config=_identity(f"mapped-config.{arm_id}", schema),
                    shared_capability_ids=shared,
                    fit_role="mapped-development-fit",
                    contact_role="mapped-development-contact",
                    protected_refit_allowed=False,
                )
                for arm_id, family, schema in method_rows
            ),
            key=lambda value: value.binding_id,
        )
    )
    return MappedChildScientificConfig(
        config_id="mapped-scientific-config.truth-known-c8",
        ensemble=ensemble,
        action_chart=MappedActionChart(
            chart_id="mapped-action-chart.truth-known-c8",
            actions=tuple(MappedActionWord),
            active_fraction_of_available_range=Decimal("0.20"),
            ramp_duration_ms=Decimal("2"),
            receiver_horizon_ms=Decimal("40"),
            delivery_relative_tolerance=Decimal("1e-9"),
            ordinal_transport_only=True,
            archive_threshold_transport_forbidden=True,
        ),
        receiver=MappedReceiverSpec(
            receiver_spec_id="mapped-receiver.truth-known-c8",
            quantity="electron-temperature",
            radial_coordinate="normalized-toroidal-flux-rho",
            radial_min=Decimal("0"),
            radial_max=Decimal("0.20"),
            weighting="GEOMETRY_CELL_VOLUME",
            minimum_valid_cells=2,
            response_definition="T40MS_MINUS_PREACTION",
            native_unit="eV",
        ),
        method_bindings=methods,
        mapped_development_count=24,
        mapped_development_fit_count=16,
        mapped_development_contact_count=8,
        mapped_prospective_attempt_count=45,
        mapped_prospective_capacity_options=(24, 30),
        nominal_reference_episode_count=18,
        maximum_confirmatory_episode_count=54,
        protected_adaptive_acquisition_allowed=False,
    )


def _generated_arm_bindings() -> tuple[GeneratedArmBinding, ...]:
    rows = (
        (GeneratedArmKind.WITNESS_GATED_IO, 'witness-gated-input-output'),
        (GeneratedArmKind.IO_ONLY, 'input-output-only'),
        (GeneratedArmKind.FINITE_MPC, 'finite-model-predictive-control'),
        (GeneratedArmKind.ROBUST_MAXIMIN, 'robust-maximin-control'),
    )
    shared = (
        "response-law.admission_receipts",
        "response-law.controller_compile",
        "response-law.law_qualification",
    )
    return tuple(
        sorted(
            (
                GeneratedArmBinding(
                    binding_id=f"generated-arm.{arm.value.lower()}",
                    arm=arm,
                    method_config_schema=f'empirical-lawhood/composition/generated-response-control/{family}/method-config',
                    method_config=_identity(
                        f"generated-method-config.{arm.value.lower()}",
                        f'empirical-lawhood/composition/generated-response-control/{family}/method-config',
                    ),
                    shared_capability_ids=shared,
                    qualification_profile=_identity(
                        f"qualification-profile.{arm.value.lower()}",
                        'empirical-lawhood/testing/qualification-profile',
                    ),
                    admission_service=_identity(
                        "shared-service.controller-admission",
                        'empirical-lawhood/testing/shared-service',
                    ),
                    controller_bridge=_identity(
                        "shared-service.controller-bridge",
                        'empirical-lawhood/testing/shared-service',
                    ),
                    local_decision_allowed=False,
                )
                for arm, family in rows
            ),
            key=lambda value: value.binding_id,
        )
    )


def _generated_config() -> GeneratedChildScientificConfig:
    return GeneratedChildScientificConfig(
        config_id="generated-scientific-config.truth-known-c8",
        roster_options=(8, 12),
        archetypes=tuple(TaskArchetype),
        model_member_count=2,
        action_count_per_task=4,
        arm_bindings=_generated_arm_bindings(),
        acquisition_preparation_count=2,
        common_bundle_count=4,
        adaptive_round_count=2,
        reference_preparation_count=2,
        confirmatory_preparation_count=3,
        confirmatory_seed_reducer="MEDIAN",
        member_reducer="MINIMUM",
        denominator_audit=DenominatorAuditSpec(
            audit_spec_id="denominator-audit.truth-known-c8",
            denominator_axis_ids=(
                "backend",
                "closure",
                "grid",
                "observation-operator",
                "precision",
                "solver",
                "timestep",
            ),
            numerical_interaction_axes=(
                "corrector-depth",
                "radial-cell-count",
                "timestep",
            ),
            interaction_levels=("central", "high", "low"),
            source_and_receiver_exact_product_required=True,
            member_drop_allowed=False,
            convergence_claim_authorized=False,
        ),
        action_algebra=ActionAlgebraCoveragePlan(
            plan_id="action-algebra.truth-known-c8",
            sentinel_task_ids=("task.action-order.01", "task.action-order.02"),
            word_ids=tuple(f"sentinel-word.{index:02d}" for index in range(1, 10)),
            preparation_count_per_task=2,
            member_count=2,
            relation_ids=(
                "composition",
                "duration",
                "future-null",
                "inverse",
                "order",
                "repetition",
                "sign",
            ),
            total_episode_count=72,
            can_supply_candidate_to_admission_commitment_or_controller_evaluation=False,
        ),
        outside_support_calibration=OutsideSupportCalibrationPlan(
            plan_id="outside-support-calibration.truth-known-c8",
            preparation_count_per_task=2,
            member_count=2,
            physical_episode_count_per_task=4,
            logical_arm_decision_count_per_task=8,
            permitted_dispositions=("HOLD", "NONATTEMPT"),
            active_action_is_failure=True,
            false_safe_hold_is_failure=True,
            can_rescue_efficacy=False,
        ),
    )


def _task_actions(task_id: str) -> tuple[GeneratedActionWordSpec, ...]:
    rows = (
        ("active-a", ActionSegmentPort.CURRENT, Decimal("-0.1"), False),
        ("active-b", ActionSegmentPort.CURRENT, Decimal("0.1"), False),
        ("active-c", ActionSegmentPort.HEATING, Decimal("0.2"), False),
        ("hold", ActionSegmentPort.HOLD, Decimal("0"), True),
    )
    return tuple(
        GeneratedActionWordSpec(
            word_id=f"action.{label}",
            label=label,
            segments=(
                GeneratedActionSegment(
                    segment_id=f"segment.{label}",
                    port=port,
                    signed_amplitude=amplitude,
                    native_unit=("MA" if port is ActionSegmentPort.CURRENT else "MW"),
                    onset_seconds=Decimal("0.01"),
                    duration_seconds=(Decimal("0") if is_hold else Decimal("0.02")),
                    occurrence_id=f"{task_id}.{label}",
                ),
            ),
            receiver_clock_id="receiver-clock.nominal",
            is_native_hold=is_hold,
            requested_accepted_applied_realized_distinct=True,
            realized_effort_is_controlling=True,
            clipping_invalidates_delivery=True,
        )
        for label, port, amplitude, is_hold in rows
    )


def _tasks() -> tuple[LawToActionTaskSpec, ...]:
    tasks: list[LawToActionTaskSpec] = []
    for archetype_index, archetype in enumerate(TaskArchetype, start=1):
        for replicate in range(1, 3):
            task_id = f"task.{archetype.value.lower().replace('_', '-')}.{replicate:02d}"
            tasks.append(
                LawToActionTaskSpec(
                    task_id=task_id,
                    archetype=archetype,
                    source_origin_id=f"source-origin.{archetype.value.lower().replace('_', '-')}.{replicate:02d}",
                    formula_origin_id=f"formula-origin.{archetype.value.lower().replace('_', '-')}.{replicate:02d}",
                    grid_origin_id=f"grid-origin.{archetype.value.lower().replace('_', '-')}.{replicate:02d}",
                    denominator_id=f"denominator.{archetype.value.lower().replace('_', '-')}.{replicate:02d}",
                    native_action_words=_task_actions(task_id),
                    receiver_dimensions=(
                        GeneratedReceiverDimension(
                            receiver_id="receiver.sink",
                            native_quantity="pathwise-sink-load",
                            native_unit="native-sink-unit",
                            direction=-1,
                            role_ids=("gate.sink", "sink"),
                            horizon_seconds=Decimal("0.04"),
                            materiality=Decimal("0.1"),
                            numeric_floor=Decimal("1e-9"),
                            validity_rule_id="sink-profile-valid",
                        ),
                        GeneratedReceiverDimension(
                            receiver_id="receiver.target",
                            native_quantity="core-electron-temperature",
                            native_unit="keV",
                            direction=1,
                            role_ids=("gate.target", "target"),
                            horizon_seconds=Decimal("0.04"),
                            materiality=Decimal("0.1"),
                            numeric_floor=Decimal("1e-9"),
                            validity_rule_id="target-profile-valid",
                        ),
                    ),
                    model_member_ids=("member.challenger", "member.primary"),
                    preparation_generator=_identity(
                        "truth-known.preparation-generator",
                        'empirical-lawhood/testing/preparation-generator',
                    ),
                    causal_cutoff=_identity(
                        f"causal-cutoff.{task_id}",
                        'empirical-lawhood/testing/causal-cutoff',
                    ),
                    generated_before_outcomes=True,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                )
            )
    result = tuple(sorted(tasks, key=lambda value: value.task_id))
    validate_generated_task_roster(result, selected_count=8)
    return result


def _preparation_plans(tasks: tuple[LawToActionTaskSpec, ...]) -> tuple[TaskPreparationPlan, ...]:
    plans = []
    for task in tasks:
        rows = [
            (PreparationPoolRole.ACQUISITION, 1),
            (PreparationPoolRole.ACQUISITION, 2),
            (PreparationPoolRole.CONFIRMATORY, 1),
            (PreparationPoolRole.CONFIRMATORY, 2),
            (PreparationPoolRole.CONFIRMATORY, 3),
            (PreparationPoolRole.OUTSIDE_SUPPORT_CALIBRATION, 1),
            (PreparationPoolRole.OUTSIDE_SUPPORT_CALIBRATION, 2),
            (PreparationPoolRole.REFERENCE, 1),
            (PreparationPoolRole.REFERENCE, 2),
        ]
        sentinel = task.archetype is TaskArchetype.ACTION_ORDER
        if sentinel:
            rows.extend(
                [
                    (PreparationPoolRole.ACTION_ALGEBRA, 1),
                    (PreparationPoolRole.ACTION_ALGEBRA, 2),
                ]
            )
        occurrences = tuple(
            sorted(
                (
                    PreparationOccurrence(
                        occurrence_id=f"occurrence.{role.value.lower()}.{index:02d}",
                        pool_role=role,
                        seed_domain_id=f"seed-domain.{role.value.lower()}",
                        requested_seed=f"seed.{task.task_id}.{role.value.lower()}.{index:02d}",
                        clone_group_id=(
                            f"clone-group.{task.task_id}.{role.value.lower()}.{index:02d}"
                        ),
                        realized_source_identity_required=True,
                    )
                    for role, index in rows
                ),
                key=lambda value: value.occurrence_id,
            )
        )
        plans.append(
            TaskPreparationPlan(
                plan_id=f"preparation-plan.{task.task_id}",
                task_id=task.task_id,
                occurrences=occurrences,
                action_algebra_sentinel_task=sentinel,
                pool_seed_domains_disjoint=True,
                cross_pool_clone_groups_disjoint=True,
                independent_unit_count=1,
                collapse_disposition="UNEVALUABLE_PREPARATION_DISTINCTNESS",
            )
        )
    return tuple(sorted(plans, key=lambda value: value.task_id))


def _namespace_bindings(
    custody: ObjectIdentity,
) -> tuple[MultiWorldStudyPayloadNamespaceBinding, ...]:
    rows = {
        "archive-qualification": (
            MultiWorldStudyPayloadOwner.ARCHIVE,
            ('empirical-lawhood/physical/mast-archive-response-qualification/archive-law-spec',),
        ),
        "archive-to-simulator-map": (
            MultiWorldStudyPayloadOwner.MAPPED,
            ('empirical-lawhood/simulators/mast-torax-state-transport/contracts/mapped-child-scientific-config',),
        ),
        "law-to-action-acquisition": (
            MultiWorldStudyPayloadOwner.GENERATED,
            ('empirical-lawhood/planning/bounded-query-acquisition-plan',),
        ),
        "law-to-action-calibration": (
            MultiWorldStudyPayloadOwner.GENERATED,
            (
                'empirical-lawhood/simulators/tokamak-prospective-control/action-algebra-coverage-plan',
                'empirical-lawhood/simulators/tokamak-prospective-control/outside-support-calibration-plan',
            ),
        ),
        "law-to-action-denominator-audit": (
            MultiWorldStudyPayloadOwner.GENERATED,
            ('empirical-lawhood/simulators/tokamak-prospective-control/denominator-audit-spec',),
        ),
        "law-to-action-domain": (
            MultiWorldStudyPayloadOwner.GENERATED,
            ('empirical-lawhood/simulators/tokamak-prospective-control/task-preparation-plan',),
        ),
        "law-to-action-nested-prospective-controller-evaluation": (
            MultiWorldStudyPayloadOwner.GENERATED,
            ('empirical-lawhood/planning/action-aware-controller-evaluation-plan',),
        ),
        "law-to-action-reference": (
            MultiWorldStudyPayloadOwner.GENERATED,
            ('empirical-lawhood/planning/finite-chart-reference-design',),
        ),
        "law-to-action-resource-envelope": (
            MultiWorldStudyPayloadOwner.JOINT,
            ('empirical-lawhood/runtime/compiled-resource-envelope-reference',),
        ),
        "multi-world-parent": (
            MultiWorldStudyPayloadOwner.JOINT,
            ('empirical-lawhood/runtime/multi-world-study-issue-manifest',),
        ),
        "property-morphism": (
            MultiWorldStudyPayloadOwner.JOINT,
            ('empirical-lawhood/planning/archive-to-simulator-partial-morphism-spec',),
        ),
    }
    return tuple(
        MultiWorldStudyPayloadNamespaceBinding(
            namespace_id=namespace,
            owner=owner,
            payload_schema_ids=schemas,
            shared_custody_capability=custody,
            local_decision_service_allowed=False,
        )
        for namespace, (owner, schemas) in sorted(rows.items())
    )


def build_synthetic_multi_world_parent_conformance(
    *,
    platform_binding: MultiWorldStudyConsumerPlatformBinding,
    public_bundle_route: ObjectIdentity,
) -> tuple[
    ArchiveLawSpec,
    MappedChildScientificConfig,
    GeneratedChildScientificConfig,
    SyntheticTaskRoster,
    SyntheticPreparationRoster,
    MultiWorldStudyScientificBinding,
    SyntheticMultiWorldStudyParentConformance,
]:
    """Build the complete fake C8 consumer with no native read or simulator launch."""

    archive_law = build_archive_law_spec()
    mapped = _mapped_config()
    generated = _generated_config()
    task_roster = SyntheticTaskRoster(
        roster_id="synthetic-task-roster.c8",
        tasks=_tasks(),
        architecture_conformance_only=True,
    )
    preparation_roster = SyntheticPreparationRoster(
        roster_id="synthetic-preparation-roster.c8",
        plans=_preparation_plans(task_roster.tasks),
        architecture_conformance_only=True,
    )
    capabilities = {value.object_id: value for value in platform_binding.shared_capabilities}
    custody = capabilities["bundle.multi-world-issue-recovery"]
    binding = MultiWorldStudyScientificBinding(
        binding_id="empirical-flagship.scientific-binding.truth-known-c8",
        g0_platform_binding=ObjectIdentity.from_record(
            platform_binding.binding_id,
            platform_binding,
        ),
        archive_law=ObjectIdentity.from_record(archive_law.law_spec_id, archive_law),
        mapped_scientific_config=ObjectIdentity.from_record(mapped.config_id, mapped),
        generated_scientific_config=ObjectIdentity.from_record(generated.config_id, generated),
        archive_protection_plan=_identity(
            "archive-protection.truth-known-c8",
            'empirical-lawhood/planning/archive-outcome-protection-plan',
        ),
        outcome_barrier_plan=_identity(
            "outcome-barriers.truth-known-c8",
            'empirical-lawhood/planning/multi-world-outcome-barrier-plan',
        ),
        partial_morphism=_identity(
            "partial-morphism.truth-known-c8",
            'empirical-lawhood/planning/archive-to-simulator-partial-morphism-spec',
        ),
        joint_adjudication_plan=_identity(
            "joint-adjudication.truth-known-c8",
            'empirical-lawhood/planning/multi-world-joint-adjudication-plan',
        ),
        namespace_bindings=_namespace_bindings(custody),
        archive_child_id="flagship-child.archive",
        mapped_child_id="flagship-child.mapped-direct-torax",
        generated_child_id="flagship-child.generated-gym-torax",
        world_local_evidence_pooling_forbidden=True,
        simulator_result_can_promote_archive=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    receipt = SyntheticMultiWorldStudyParentConformance(
        receipt_id="empirical-flagship.synthetic-parent-conformance-c8",
        scientific_binding=ObjectIdentity.from_record(binding.binding_id, binding),
        archive_law=binding.archive_law,
        mapped_scientific_config=binding.mapped_scientific_config,
        generated_scientific_config=binding.generated_scientific_config,
        generated_task_roster=ObjectIdentity.from_record(task_roster.roster_id, task_roster),
        preparation_plan_roster=ObjectIdentity.from_record(
            preparation_roster.roster_id,
            preparation_roster,
        ),
        contacted_node_family_ids=(
            "archive-exposure-staging",
            "archive-late-reveal-evaluation",
            "archive-to-torax-mapping",
            "atlas-controller-admission-reachability",
            "common-and-adaptive-acquisition",
            "confirmatory-cells",
            "denominator-algebra-calibration",
            "final-arm-qualification",
            "joint-adjudication-report",
            "mapped-excluded-development",
            "programme-compile-binding-tick",
            "property-fan-in",
            "reference-cells-class",
            "resource-recovery",
            "roster-domain-freeze",
            "simulator-reveal-evaluation",
            "source-runtime-revalidation",
        ),
        child_ids=(
            "flagship-child.archive",
            "flagship-child.generated-gym-torax",
            "flagship-child.mapped-direct-torax",
        ),
        observed_morphism_dispositions=tuple(PropertyMorphismDisposition),
        morphism_negative_controls=tuple(MorphismNegativeControlKind),
        reveal_order=(
            "generated-torax-outcome",
            "mapped-torax-outcome",
            "archive-receiver-outcome",
        ),
        public_bundle_route=public_bundle_route,
        generated_task_count=8,
        generated_task_arm_coordinate_count=32,
        mapped_member_count=6,
        shared_service_fork_count=0,
        archived_semantic_import_count=0,
        cross_child_evidence_pool_count=0,
        native_archive_read_count=0,
        native_simulator_launch_count=0,
        simulator_only_nonpromotion_passed=True,
        crash_recovery_passed=True,
        catalog_rebuild_passed=True,
        terminal=True,
        architecture_conformance_only=True,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    return (
        archive_law,
        mapped,
        generated,
        task_roster,
        preparation_roster,
        binding,
        receipt,
    )


__all__ = [
    'SyntheticPreparationRoster',
    'SyntheticTaskRoster',
    'build_synthetic_multi_world_parent_conformance',
]

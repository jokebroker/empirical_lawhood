"Outcome-visible-parent, new-outcome-sealed Gym-TORAX finite-action recurrence recovery.\n\nThis adapter is deliberately local to this technical recovery.  It reuses\nthe immutable parent freeze qualification as a prerequisite, mechanically reissues the\nsame 18+4/80 recurrence design under fresh identities, preserves partial\nepisodes, and delegates complete or typed-partial adjudication to the existing\nfinite-action recurrence owner.  It cannot promote the parent qualification or its recovery.\n"

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from enum import StrEnum
import hashlib
from pathlib import Path
import re
from statistics import median
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.finite_action_occurrence import finite_action_branch_selection_receipt_id
from empirical_lawhood.planning.finite_action_recurrence import FiniteActionPreactionDisposition, FiniteActionRecurrenceAssignmentBindingReceipt, FiniteActionRecurrenceAssignmentTemplate, FiniteActionRecurrenceAssignment, FiniteActionRecurrencePredicateResult, FiniteActionRecurrenceProspectiveBundle, FiniteActionRecurrenceResponseSample, FiniteActionRecurrenceRevealIntegrity, FiniteActionRecurrenceRevealedBundle, FiniteActionRecurrenceRevealedEpisode, FiniteActionRecurrenceRouteRole, FiniteActionRecurrenceRouteSpec, FiniteActionRecurrenceSealedEpisode, FiniteActionRecurrenceUnitSpec, bind_finite_action_recurrence_assignment_template
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, StudyAuthorityKind, StudyOperationAuthority, SourceClosureKind, require_study_authority
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactManifest
from empirical_lawhood.runtime.finite_action_recurrence import FiniteActionRecurrenceAdjudication, FiniteActionRecurrenceResult, MatchedFiniteActionHoldRecurrenceEvaluator

from .action_word import build_gym_torax_action_word, build_gym_torax_native_schedule
from .diagnostic_contracts import GymToraxDeliveryDisposition, GymToraxNumericalDisposition, GymToraxObservationDisposition, GymToraxNumericalMember, GymToraxSourceDisposition
from .field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxFieldMetadataNativeEpisode, GymToraxRequestRole
from .extraction_manifest import GymToraxBoundedExtractionManifest
from .field_metadata import GymToraxFieldMetadataManifest
from .finite_action_recurrence_execution import GymToraxFiniteActionRecurrenceTerminalAcquisitionReceipt
from .retained_inputs import GymToraxRecoveryParents
from .protocol_scientific_inputs import GymToraxProtocolEnvironmentSeedInput, require_protocol_seed_census, require_protocol_root_seed

from .finite_action_recurrence_protocol import GymToraxFiniteActionRecurrenceCellStage, GymToraxFiniteActionRecurrenceCell, GymToraxFiniteActionRecurrencePreparedUnit, GymToraxFiniteActionRecurrenceProspectiveFreeze
from .finite_action_recurrence_science import GymToraxFiniteActionRecurrenceQualificationBundle, _artifact, _content_artifact, _episode_identity, _observed, _prefix_sha, _validate_hfr_episode
from .selected_parent_protocol import GymToraxSelectedParentFreeze
from .selected_parent_science import GymToraxSelectedParentScienceEvaluationBundle
from .source_assessment_execution import GymToraxArtifactPublicationItem, GymToraxOperationalDisposition, GymToraxBoundedArtifactStore
from .runtime import GymToraxEnvironmentFactory, GymToraxRuntimeInspector, acquire_gym_torax_episode, inspect_gym_torax_runtime
from .source_qualification import GymToraxNativeSourceQualificationDisposition, GymToraxNativeSourceQualificationReceipt


GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_FREEZE_ID = 'freeze.tokamak-control.finite-action-recurrence-recovery'
GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_RUN_ID = 'run.tokamak-control.finite-action-recurrence-recovery'
GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_RELATIVE_ROOT = (
    'tokamak-control/publication/runs/run.tokamak-control.finite-action-recurrence-recovery'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_EVALUATOR_ID = 'evaluator.tokamak-control.finite-action-recurrence-recovery'
GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_AUTHORITY_IDS = tuple(
    sorted(
        f'authority.tokamak-control.finite-action-recurrence-recovery.{suffix}'
        for suffix in (
            "custody-publication",
            "experiment-execution",
            "outcome-reveal",
            "source-acquisition",
        )
    )
)


_TASK_BUDGET = ResourceBudget(
    cpu_cores=8,
    memory_bytes=32 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=12 * 60 * 60,
    source_scan_bytes=0,
    output_bytes=64 * 1024**2,
)
_CAMPAIGN_BUDGET = ResourceBudget(
    cpu_cores=8,
    memory_bytes=32 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=80 * 12 * 60 * 60,
    source_scan_bytes=0,
    output_bytes=8 * 1024**3,
)


def _identity(identifier: str, record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(identifier, record)


def _parent(identity: ObjectIdentity, access: OutcomeAccess) -> ArtifactLineageParent:
    return ArtifactLineageParent(
        identity=identity,
        visibility_ceiling=(
            VisibilityCeiling.PROSPECTIVE
            if access in {OutcomeAccess.OUTCOME_BLIND, OutcomeAccess.EVALUATION_SEALED}
            else VisibilityCeiling.OUTCOME_VISIBLE
        ),
        outcome_access=access,
    )


def _fresh(value: str) -> str:
    if 'finite-action-recurrence' in value:
        return value.replace('finite-action-recurrence', 'finite-action-recurrence-recovery')
    return f'{value}.finite-action-recurrence-recovery'




@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceRecoveryRouteMap(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-recovery-route-map'
    )

    parent_episode_id: str
    recovery_episode_id: str
    parent_execution_id: str
    recovery_execution_id: str
    parent_environment_seed_id: str
    recovery_environment_seed_id: str
    parent_assignment_unit_id: str
    recovery_assignment_unit_id: str
    parent_physical_unit_id: str
    recovery_physical_unit_id: str
    model_member_id: str
    role: FiniteActionRecurrenceRouteRole
    action_word_id: str

    def __post_init__(self) -> None:
        for name in (
            "parent_episode_id",
            "recovery_episode_id",
            "parent_execution_id",
            "recovery_execution_id",
            "parent_environment_seed_id",
            "recovery_environment_seed_id",
            "parent_assignment_unit_id",
            "recovery_assignment_unit_id",
            "parent_physical_unit_id",
            "recovery_physical_unit_id",
            "model_member_id",
            "action_word_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        for left, right in (
            (self.parent_episode_id, self.recovery_episode_id),
            (self.parent_execution_id, self.recovery_execution_id),
            (self.parent_environment_seed_id, self.recovery_environment_seed_id),
            (self.parent_assignment_unit_id, self.recovery_assignment_unit_id),
            (self.parent_physical_unit_id, self.recovery_physical_unit_id),
        ):
            if left == right:
                raise ValueError("Finite-action recurrence recovery route reuses a parent identity")


def _scientific_design_fingerprint(
    assignment: FiniteActionRecurrenceAssignment,
) -> str:
    rows = []
    units = {value.unit_id: value for value in assignment.units}
    for route in assignment.routes:
        unit = units[route.unit_id]
        rows.append(
            "|".join(
                (
                    unit.preparation_coordinate_id,
                    unit.role.value,
                    unit.local_support_id or "-",
                    unit.stratum_id or "-",
                    unit.hold_anchor_slot_id or "-",
                    route.model_member_id,
                    route.role.value,
                    route.action_word_id,
                )
            )
        )
    fixed = (
        assignment.active_word.fingerprint(),
        assignment.matched_hold_word.fingerprint(),
        assignment.source.object_fingerprint,
        assignment.schedule.object_fingerprint,
        assignment.retained_history.object_fingerprint,
        assignment.receiver.object_fingerprint,
        assignment.horizon.object_fingerprint,
        assignment.causal_cutoff.fingerprint(),
        *(value.fingerprint() for value in assignment.response_coordinates),
        assignment.effect_quantity_id,
        assignment.effect_native_unit,
        str(assignment.minimum_evaluable_units),
        str(assignment.minimum_active_coverage),
        str(assignment.alpha),
        str(assignment.degrees_of_freedom),
        str(assignment.one_sided_critical_value),
        assignment.materiality.fingerprint(),
        str(assignment.strict_materiality),
        *assignment.reduction_order,
        *assignment.terminal_precedence,
    )
    return hashlib.sha256("\n".join((*sorted(rows), *fixed)).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceRecoveryFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-recovery-freeze'

    freeze_id: str
    run_id: str
    relative_root: str
    parent_freeze: ObjectIdentity
    parent_qualification: ObjectIdentity
    parent_terminal: ObjectIdentity
    parent_assignment: ObjectIdentity
    parent_law_qualification_result: ObjectIdentity
    evidence_preservation_defect_id: str
    scientific_design_fingerprint: str
    assignment: FiniteActionRecurrenceAssignment
    assignment_binding: FiniteActionRecurrenceAssignmentBindingReceipt
    route_mapping: tuple[GymToraxFiniteActionRecurrenceRecoveryRouteMap, ...]
    source_qualification: ObjectIdentity
    extraction_manifest: ObjectIdentity
    field_metadata_manifest: ObjectIdentity
    implementation_source_closure: ImplementationSourceClosure
    scientific_approval: ObjectIdentity
    units: tuple[GymToraxFiniteActionRecurrencePreparedUnit, ...]
    numerical_members: tuple[GymToraxNumericalMember, ...]
    cells: tuple[GymToraxFiniteActionRecurrenceCell, ...]
    task_resource_budget: ResourceBudget
    campaign_resource_budget: ResourceBudget
    maximum_concurrency: int
    required_operation_authority_ids: tuple[str, ...]
    parent_outcomes_visible: bool
    new_recurrence_outcomes_sealed: bool
    scientific_design_unchanged: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        for name in ("freeze_id", "run_id", "evidence_preservation_defect_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_relative_locator(self.relative_root)
        validate_sha256(
            self.scientific_design_fingerprint,
            field_name="scientific_design_fingerprint",
        )
        if self.parent_freeze.object_schema != GymToraxFiniteActionRecurrenceProspectiveFreeze.SCHEMA:
            raise ValueError("Finite-action recurrence recovery binds another parent freeze schema")
        if (
            self.parent_qualification.object_schema
            != GymToraxFiniteActionRecurrenceQualificationBundle.SCHEMA
        ):
            raise ValueError("Finite-action recurrence recovery binds another qualification schema")
        if (
            self.parent_terminal.object_schema
            != GymToraxFiniteActionRecurrenceTerminalAcquisitionReceipt.SCHEMA
        ):
            raise ValueError("Finite-action recurrence recovery binds another terminal schema")
        if (
            self.parent_assignment.object_schema
            != FiniteActionRecurrenceAssignment.SCHEMA
        ):
            raise ValueError("Finite-action recurrence recovery binds another assignment schema")
        if (
            self.implementation_source_closure.kind
            is not SourceClosureKind.CLEAN_GIT_COMMIT
        ):
            raise ValueError("Finite-action recurrence recovery requires a clean committed implementation")
        require_sorted_unique_ids(self.units, attribute="unit_id", field_name="units")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_ids(
            self.route_mapping,
            attribute="recovery_episode_id",
            field_name="route_mapping",
        )
        require_sorted_unique_strings(
            self.required_operation_authority_ids,
            field_name="required_operation_authority_ids",
            allow_empty=False,
        )
        if (
            len(self.units) != 22
            or any(unit.scientific_seed_input.scientific_role != "recurrence-recovery" or not unit.scientific_primary for unit in self.units)
            or len(self.cells) != 80
            or len(self.route_mapping) != 80
            or len(self.assignment.routes) != 80
            or any(
                value.stage is not GymToraxFiniteActionRecurrenceCellStage.RECURRENCE for value in self.cells
            )
            or any(
                value.episode_id
                != f"episode.{value.request_id.removeprefix('request.')}"
                for value in self.cells
            )
        ):
            raise ValueError("Finite-action recurrence recovery changes the exact 22-unit/80-route product")
        if {value.episode_id for value in self.cells} != {
            value.episode_id for value in self.assignment.routes
        }:
            raise ValueError("Finite-action recurrence recovery cells differ from its assignment")
        if self.scientific_design_fingerprint != _scientific_design_fingerprint(
            self.assignment
        ):
            raise ValueError("Finite-action recurrence recovery scientific design fingerprint differs")
        if (
            not self.parent_outcomes_visible
            or not self.new_recurrence_outcomes_sealed
            or not self.scientific_design_unchanged
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError(
                "Finite-action recurrence recovery changes its outcome-visible nonpromotable lane"
            )

    def unit_for(self, unit_id: str) -> GymToraxFiniteActionRecurrencePreparedUnit:
        return next(value for value in self.units if value.unit_id == unit_id)


def build_gym_torax_finite_action_recurrence_recovery_freeze(
    *,
    expected_parents: GymToraxRecoveryParents,
    parent_freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    parent_qualification: GymToraxFiniteActionRecurrenceQualificationBundle,
    parent_terminal: GymToraxFiniteActionRecurrenceTerminalAcquisitionReceipt,
    source_qualification: GymToraxNativeSourceQualificationReceipt,
    extraction_manifest: GymToraxBoundedExtractionManifest,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    implementation_source_closure: ImplementationSourceClosure,
    scientific_approval: ObjectIdentity,
    scientific_seed_inputs: tuple[GymToraxProtocolEnvironmentSeedInput, ...] | None = None,
) -> GymToraxFiniteActionRecurrenceRecoveryFreeze:
    """Mechanically reissue the exact r5 scientific topology under fresh identities."""

    seed_inputs = require_protocol_seed_census(scientific_seed_inputs, matched_evaluation=False, recurrence_recovery=True)
    parent_assignment = parent_qualification.assignment
    expected_seed_roots = {_fresh(value.unit_id) for value in parent_freeze.units if value.scientific_primary}
    if set(seed_inputs) != expected_seed_roots:
        raise ValueError("recovery requires exactly the complete numeric census of its twenty-two new physical roots")
    if (
        parent_freeze.fingerprint() != expected_parents.freeze.object_fingerprint
        or parent_qualification.fingerprint()
        != expected_parents.qualification.object_fingerprint
        or parent_terminal.fingerprint() != expected_parents.terminal.object_fingerprint
        or parent_assignment.fingerprint()
        != expected_parents.assignment.object_fingerprint
        or parent_qualification.law_qualification_result.object_fingerprint
        != expected_parents.law_qualification.object_fingerprint
        or parent_terminal.completed_episode_count != 18
        or parent_terminal.failed_cell_count != 80
        or parent_terminal.scientific_values_decoded
        or source_qualification.disposition
        is not GymToraxNativeSourceQualificationDisposition.QUALIFIED
        or source_qualification.extraction_manifest != _identity(extraction_manifest.manifest_id, extraction_manifest)
        or source_qualification.field_metadata_manifest != _identity(field_metadata_manifest.manifest_id, field_metadata_manifest)
        or parent_freeze.source_qualification
        != _identity(source_qualification.receipt_id, source_qualification)
    ):
        raise ValueError("Finite-action recurrence recovery parent chain differs from immutable parent freeze")

    parent_prepared_by_coordinate = {
        value.source_cell_id: value
        for value in parent_freeze.units
        if value.scientific_primary
    }
    old_seed_values = {
        value.preparation.environment_seed
        for value in parent_prepared_by_coordinate.values()
    }
    prepared_by_coordinate: dict[str, GymToraxFiniteActionRecurrencePreparedUnit] = {}
    new_unit_specs: list[FiniteActionRecurrenceUnitSpec] = []
    assignment_unit_map: dict[str, str] = {}
    physical_unit_map: dict[str, str] = {}
    for old in parent_assignment.units:
        parent_prepared = parent_prepared_by_coordinate[old.preparation_coordinate_id]
        new_physical_id = _fresh(parent_prepared.unit_id)
        scientific_seed = require_protocol_root_seed(seed_inputs, new_physical_id, "recurrence-recovery")
        seed = scientific_seed.environment_seed
        if seed in old_seed_values:
            raise ValueError("Finite-action recurrence recovery seed aliases the parent freeze")
        preparation = replace(
            parent_prepared.preparation,
            preparation_id=_fresh(parent_prepared.preparation.preparation_id),
            physical_independent_unit_id=new_physical_id,
            environment_seed=seed,
        )
        prepared = replace(
            parent_prepared, unit_id=new_physical_id, preparation=preparation, scientific_seed_input=scientific_seed
        )
        prepared_by_coordinate[old.preparation_coordinate_id] = prepared
        new_assignment_unit_id = _fresh(old.unit_id)
        assignment_unit_map[old.unit_id] = new_assignment_unit_id
        physical_unit_map[old.physical_independent_unit_id] = new_physical_id
        new_unit_specs.append(
            replace(
                old,
                unit_id=new_assignment_unit_id,
                physical_independent_unit_id=new_physical_id,
                # The assignment binds the scientific preparation-value projection,
                # not the fresh request/seed identity.  Those values are unchanged.
                preparation_sha256=old.preparation_sha256,
            )
        )

    new_routes: list[FiniteActionRecurrenceRouteSpec] = []
    route_mapping: list[GymToraxFiniteActionRecurrenceRecoveryRouteMap] = []
    for old_route in parent_assignment.routes:
        old_unit = next(
            value
            for value in parent_assignment.units
            if value.unit_id == old_route.unit_id
        )
        new_physical = physical_unit_map[old_unit.physical_independent_unit_id]
        new_route = replace(
            old_route,
            episode_id=f"episode.{_fresh(old_route.episode_id)}",
            execution_id=_fresh(old_route.execution_id),
            environment_seed_id=f"environment-seed.{new_physical}",
            unit_id=assignment_unit_map[old_route.unit_id],
        )
        new_routes.append(new_route)
        route_mapping.append(
            GymToraxFiniteActionRecurrenceRecoveryRouteMap(
                parent_episode_id=old_route.episode_id,
                recovery_episode_id=new_route.episode_id,
                parent_execution_id=old_route.execution_id,
                recovery_execution_id=new_route.execution_id,
                parent_environment_seed_id=old_route.environment_seed_id,
                recovery_environment_seed_id=new_route.environment_seed_id,
                parent_assignment_unit_id=old_route.unit_id,
                recovery_assignment_unit_id=new_route.unit_id,
                parent_physical_unit_id=old_unit.physical_independent_unit_id,
                recovery_physical_unit_id=new_physical,
                model_member_id=old_route.model_member_id,
                role=old_route.role,
                action_word_id=old_route.action_word_id,
            )
        )

    new_strata = tuple(
        replace(
            value, unit_ids=tuple(assignment_unit_map[item] for item in value.unit_ids)
        )
        for value in parent_assignment.strata
    )
    template = FiniteActionRecurrenceAssignmentTemplate(
        template_id='template.tokamak-control.finite-action-recurrence-recovery-assignment',
        expected_assignment_id='assignment.tokamak-control.finite-action-recurrence-recovery',
        frozen_roster=parent_assignment.frozen_roster,
        roster_collision_audit=parent_assignment.roster_collision_audit,
        expected_branch_selection_receipt_id=(
            finite_action_branch_selection_receipt_id(
                parent_assignment.branch_selection
            )
        ),
        expected_branch_selection_receipt_schema=parent_assignment.branch_selection.SCHEMA,
        expected_occurrence_receipt_ids=tuple(
            value.receipt_id for value in parent_assignment.occurrence_receipts
        ),
        expected_occurrence_receipt_schema=parent_assignment.occurrence_receipts[
            0
        ].SCHEMA,
        expected_native_hold_calibration_receipt_id=(
            parent_assignment.native_hold_calibration.receipt_id
        ),
        expected_native_hold_calibration_receipt_schema=(
            parent_assignment.native_hold_calibration.SCHEMA
        ),
        expected_route_qualification_receipt_id=parent_assignment.route_qualification.receipt_id,
        expected_route_qualification_receipt_schema=parent_assignment.route_qualification.SCHEMA,
        units=tuple(sorted(new_unit_specs, key=lambda value: value.unit_id)),
        routes=tuple(sorted(new_routes, key=lambda value: value.episode_id)),
        model_member_ids=parent_assignment.model_member_ids,
        strata=tuple(sorted(new_strata, key=lambda value: value.stratum_id)),
        active_word=parent_assignment.active_word,
        matched_hold_word=parent_assignment.matched_hold_word,
        source=parent_assignment.source,
        schedule=parent_assignment.schedule,
        retained_history=parent_assignment.retained_history,
        receiver=parent_assignment.receiver,
        horizon=parent_assignment.horizon,
        causal_cutoff=parent_assignment.causal_cutoff,
        response_coordinates=parent_assignment.response_coordinates,
        effect_quantity_id=parent_assignment.effect_quantity_id,
        effect_native_unit=parent_assignment.effect_native_unit,
        control_anchor_slot_ids=parent_assignment.control_anchor_slot_ids,
        hold_reserve_anchor_slot_id=parent_assignment.hold_reserve_anchor_slot_id,
        reserves=parent_assignment.reserves,
        reserve_order=parent_assignment.reserve_order,
        minimum_evaluable_units=parent_assignment.minimum_evaluable_units,
        minimum_active_coverage=parent_assignment.minimum_active_coverage,
        alpha=parent_assignment.alpha,
        degrees_of_freedom=parent_assignment.degrees_of_freedom,
        one_sided_critical_value=parent_assignment.one_sided_critical_value,
        materiality=parent_assignment.materiality,
        strict_materiality=parent_assignment.strict_materiality,
        reduction_order=parent_assignment.reduction_order,
        terminal_precedence=parent_assignment.terminal_precedence,
        preaction_prefix_must_be_byte_equal=parent_assignment.preaction_prefix_must_be_byte_equal,
        route_preassigned_without_online_choice=(
            parent_assignment.route_preassigned_without_online_choice
        ),
        claim_ceiling=parent_assignment.claim_ceiling,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )
    assignment, binding = bind_finite_action_recurrence_assignment_template(
        receipt_id='binding-receipt.assignment.tokamak-control.finite-action-recurrence-recovery',
        template=template,
        branch_selection=parent_assignment.branch_selection,
        occurrence_receipts=parent_assignment.occurrence_receipts,
        native_hold_calibration=parent_assignment.native_hold_calibration,
        route_qualification=parent_assignment.route_qualification,
        binding_implementation_id='binder.tokamak-control.finite-action-recurrence-recovery',
        binding_implementation_sha256=extraction_manifest.fingerprint(),
    )
    if _scientific_design_fingerprint(assignment) != _scientific_design_fingerprint(
        parent_assignment
    ):
        raise ValueError("Finite-action recurrence recovery changes a scientific field")

    cells: list[GymToraxFiniteActionRecurrenceCell] = []
    route_by_episode = {value.episode_id: value for value in assignment.routes}
    parent_route_by_episode = {
        value.episode_id: value for value in parent_assignment.routes
    }
    mapping_by_recovery = {value.recovery_episode_id: value for value in route_mapping}
    member_by_id = {value.member_id: value for value in parent_freeze.numerical_members}
    for episode_id, route in route_by_episode.items():
        mapping = mapping_by_recovery[episode_id]
        old_route = parent_route_by_episode[mapping.parent_episode_id]
        old_unit = next(
            value
            for value in parent_assignment.units
            if value.unit_id == old_route.unit_id
        )
        prepared = prepared_by_coordinate[old_unit.preparation_coordinate_id]
        stem = episode_id.removeprefix("episode.")
        request = GymToraxFieldMetadataEpisodeRequest(
            request_id=f"request.{stem}",
            request_role=GymToraxRequestRole.SCIENTIFIC_EPISODE,
            source_qualification=_identity(
                source_qualification.receipt_id, source_qualification
            ),
            extraction_manifest=_identity(
                extraction_manifest.manifest_id, extraction_manifest
            ),
            field_metadata_manifest=_identity(
                field_metadata_manifest.manifest_id, field_metadata_manifest
            ),
            preparation=prepared.preparation,
            numerical_member=member_by_id[route.model_member_id],
            schedule=build_gym_torax_native_schedule(
                build_gym_torax_action_word(route.action_word_id)
            ),
            maximum_output_bytes=_TASK_BUDGET.output_bytes,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        )
        cells.append(
            GymToraxFiniteActionRecurrenceCell(
                cell_id=f"cell.{stem}",
                episode_id=episode_id,
                execution_id=route.execution_id,
                stage=GymToraxFiniteActionRecurrenceCellStage.RECURRENCE,
                unit_id=prepared.unit_id,
                model_member_id=route.model_member_id,
                route_role=route.role.value.lower().replace("_", "-"),
                action_word_id=route.action_word_id,
                request_id=request.request_id,
                request_sha256=request.fingerprint(),
            )
        )
    return GymToraxFiniteActionRecurrenceRecoveryFreeze(
        freeze_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_FREEZE_ID,
        run_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_RUN_ID,
        relative_root=GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_RELATIVE_ROOT,
        parent_freeze=_identity(parent_freeze.freeze_id, parent_freeze),
        parent_qualification=_identity(
            parent_qualification.bundle_id, parent_qualification
        ),
        parent_terminal=_identity(parent_terminal.receipt_id, parent_terminal),
        parent_assignment=_identity(parent_assignment.assignment_id, parent_assignment),
        parent_law_qualification_result=parent_qualification.law_qualification_result,
        evidence_preservation_defect_id='defect.tokamak-control.source-availability-by-clock-defect',
        scientific_design_fingerprint=_scientific_design_fingerprint(assignment),
        assignment=assignment,
        assignment_binding=binding,
        route_mapping=tuple(
            sorted(route_mapping, key=lambda value: value.recovery_episode_id)
        ),
        source_qualification=_identity(
            source_qualification.receipt_id, source_qualification
        ),
        extraction_manifest=_identity(
            extraction_manifest.manifest_id, extraction_manifest
        ),
        field_metadata_manifest=_identity(
            field_metadata_manifest.manifest_id, field_metadata_manifest
        ),
        implementation_source_closure=implementation_source_closure,
        scientific_approval=scientific_approval,
        units=tuple(
            sorted(prepared_by_coordinate.values(), key=lambda value: value.unit_id)
        ),
        numerical_members=parent_freeze.numerical_members,
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        task_resource_budget=_TASK_BUDGET,
        campaign_resource_budget=_CAMPAIGN_BUDGET,
        maximum_concurrency=1,
        required_operation_authority_ids=GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_AUTHORITY_IDS,
        parent_outcomes_visible=True,
        new_recurrence_outcomes_sealed=True,
        scientific_design_unchanged=True,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def materialize_gym_torax_finite_action_recurrence_recovery_request(
    freeze: GymToraxFiniteActionRecurrenceRecoveryFreeze, cell: GymToraxFiniteActionRecurrenceCell
) -> GymToraxFieldMetadataEpisodeRequest:
    unit = freeze.unit_for(cell.unit_id)
    member = next(
        value
        for value in freeze.numerical_members
        if value.member_id == cell.model_member_id
    )
    request = GymToraxFieldMetadataEpisodeRequest(
        request_id=cell.request_id,
        request_role=GymToraxRequestRole.SCIENTIFIC_EPISODE,
        source_qualification=freeze.source_qualification,
        extraction_manifest=freeze.extraction_manifest,
        field_metadata_manifest=freeze.field_metadata_manifest,
        preparation=unit.preparation,
        numerical_member=member,
        schedule=build_gym_torax_native_schedule(
            build_gym_torax_action_word(cell.action_word_id)
        ),
        maximum_output_bytes=freeze.task_resource_budget.output_bytes,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    )
    if request.fingerprint() != cell.request_sha256:
        raise ValueError("Finite-action recurrence recovery request differs from frozen bytes")
    return request


class GymToraxFiniteActionRecurrenceRecoveryTerminalDisposition(StrEnum):
    COMPLETE_VALID = "COMPLETE_VALID"
    COMPLETE_WITH_PARTIAL = "COMPLETE_WITH_PARTIAL"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceRecoveryCellReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-recovery-cell-receipt'
    )

    receipt_id: str
    freeze: ObjectIdentity
    cell: ObjectIdentity
    request: ObjectIdentity
    episode: ObjectIdentity | None
    disposition: GymToraxOperationalDisposition
    complete_valid_episode: bool
    operational_reason_codes: tuple[str, ...]
    episode_reason_codes: tuple[str, ...]
    technical_detail_sha256: str | None
    attempt_count: int
    automatic_retry_performed: bool
    source_reset_attempted: bool
    scientific_values_decoded: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.freeze.object_schema != GymToraxFiniteActionRecurrenceRecoveryFreeze.SCHEMA:
            raise ValueError("Finite-action recurrence recovery receipt binds another freeze")
        require_sorted_unique_strings(
            self.operational_reason_codes, field_name="operational_reason_codes"
        )
        require_sorted_unique_strings(
            self.episode_reason_codes, field_name="episode_reason_codes"
        )
        if self.technical_detail_sha256 is not None:
            validate_sha256(
                self.technical_detail_sha256, field_name="technical_detail_sha256"
            )
        if self.attempt_count != 1 or self.automatic_retry_performed:
            raise ValueError("Finite-action recurrence recovery permits one attempt and no automatic retry")
        if self.disposition is GymToraxOperationalDisposition.SUCCEEDED:
            if self.episode is None or self.operational_reason_codes:
                raise ValueError("successful finite-action recurrence recovery requires one episode")
        elif (
            self.episode is not None
            or self.complete_valid_episode
            or not self.operational_reason_codes
        ):
            raise ValueError("failed finite-action recurrence recovery cannot claim an episode")
        if (
            self.scientific_values_decoded
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("Finite-action recurrence recovery receipt changes its nonpromotable lane")


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceRecoveryTerminalReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-recovery-terminal-receipt'
    )

    receipt_id: str
    freeze: ObjectIdentity
    cell_receipts: tuple[ObjectIdentity, ...]
    episodes: tuple[ObjectIdentity, ...]
    accounted_cell_count: int
    episode_count: int
    complete_valid_episode_count: int
    partial_episode_count: int
    failed_cell_count: int
    disposition: GymToraxFiniteActionRecurrenceRecoveryTerminalDisposition
    reason_codes: tuple[str, ...]
    scientific_values_decoded: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        require_sorted_unique_ids(
            self.cell_receipts, attribute="object_id", field_name="cell_receipts"
        )
        require_sorted_unique_ids(
            self.episodes, attribute="object_id", field_name="episodes"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.freeze.object_schema != GymToraxFiniteActionRecurrenceRecoveryFreeze.SCHEMA
            or self.accounted_cell_count != 80
            or len(self.cell_receipts) != 80
            or self.episode_count != len(self.episodes)
            or self.complete_valid_episode_count + self.partial_episode_count
            != self.episode_count
            or self.episode_count + self.failed_cell_count != 80
        ):
            raise ValueError("Finite-action recurrence recovery terminal accounting differs")
        expected = (
            GymToraxFiniteActionRecurrenceRecoveryTerminalDisposition.INCOMPLETE
            if self.failed_cell_count
            else (
                GymToraxFiniteActionRecurrenceRecoveryTerminalDisposition.COMPLETE_WITH_PARTIAL
                if self.partial_episode_count
                else GymToraxFiniteActionRecurrenceRecoveryTerminalDisposition.COMPLETE_VALID
            )
        )
        if self.disposition is not expected:
            raise ValueError("Finite-action recurrence recovery terminal disposition differs")
        if (
            self.scientific_values_decoded
            or self.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("Finite-action recurrence recovery terminal changes its evidence lane")


def _is_complete_valid(episode: GymToraxFieldMetadataNativeEpisode) -> bool:
    return (
        episode.state_clocks == tuple(range(121))
        and not episode.missing_required_state_clocks
        and episode.last_valid_state_clock == 120
        and episode.source_disposition is GymToraxSourceDisposition.AVAILABLE
        and episode.delivery_disposition is GymToraxDeliveryDisposition.COMPLETE
        and episode.numerical_disposition is GymToraxNumericalDisposition.VALID
        and episode.observation_disposition is GymToraxObservationDisposition.COMPLETE
        and not episode.termination
        and not episode.truncation
    )


def _require_authority(
    *,
    freeze: GymToraxFiniteActionRecurrenceRecoveryFreeze,
    authority: StudyOperationAuthority,
    kind: StudyAuthorityKind,
    at_utc: str,
    store: GymToraxBoundedArtifactStore | None = None,
    prerequisite: ObjectIdentity | None = None,
) -> None:
    require_study_authority(
        authority,
        kind=kind,
        subject=_identity(freeze.freeze_id, freeze),
        prerequisite_authority=prerequisite,
        grantee_id=(
            GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_EVALUATOR_ID
            if kind is StudyAuthorityKind.OUTCOME_REVEAL
            else "operator.execution-service"
        ),
        storage_root_id=(
            store.storage_root_id
            if store and kind is StudyAuthorityKind.CUSTODY_PUBLICATION
            else None
        ),
        relative_root=(
            freeze.relative_root
            if kind is StudyAuthorityKind.CUSTODY_PUBLICATION
            else None
        ),
        at_utc=at_utc,
    )
    if authority.authority_id not in freeze.required_operation_authority_ids:
        raise PermissionError("Finite-action recurrence recovery authority is outside the freeze")


def publish_gym_torax_finite_action_recurrence_recovery_freeze(
    *,
    freeze: GymToraxFiniteActionRecurrenceRecoveryFreeze,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> ObjectIdentity:
    _require_authority(
        freeze=freeze,
        authority=custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        at_utc=at_utc,
        store=store,
    )
    parents = tuple(
        sorted(
            (
                _parent(freeze.parent_freeze, OutcomeAccess.EVALUATION_REVEALED),
                _parent(freeze.parent_qualification, OutcomeAccess.EVALUATOR_REVEAL),
                _parent(freeze.parent_terminal, OutcomeAccess.EVALUATION_REVEALED),
                _parent(freeze.extraction_manifest, OutcomeAccess.OUTCOME_BLIND),
            ),
            key=lambda value: value.identity.object_id,
        )
    )
    return store.publish_atomic(
        publication_scope_id=f"publication.{freeze.run_id}",
        publication_scope_relative_root=freeze.relative_root,
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id=f"artifact.{freeze.run_id}.freeze",
                relative_path=f"{freeze.relative_root}/freeze.json",
                record=freeze,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                lineage_parents=parents,
            ),
        ),
    )[0]


def _load_freeze(store: GymToraxBoundedArtifactStore) -> GymToraxFiniteActionRecurrenceRecoveryFreeze:
    value = store.load_optional(
        logical_artifact_id=f'artifact.{GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_RUN_ID}.freeze',
        relative_path=f'{GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_RELATIVE_ROOT}/freeze.json',
        record_type=GymToraxFiniteActionRecurrenceRecoveryFreeze,
        maximum_bytes=32 * 1024**2,
    )
    if value is None:
        raise RuntimeError("GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_FREEZE_NOT_PUBLISHED")
    return value


def _technical_reason(error: Exception) -> tuple[tuple[str, ...], str]:
    token = (
        re.sub(r"[^A-Z0-9]+", "_", type(error).__name__.upper()).strip("_") or "ERROR"
    )
    detail = f"{type(error).__name__}:{str(error)[:1024]}"
    return (f"FINITE_ACTION_RECURRENCE_RECOVERY_TECHNICAL_{token}",), hashlib.sha256(
        detail.encode()
    ).hexdigest()


def run_gym_torax_finite_action_recurrence_recovery(
    *,
    freeze: GymToraxFiniteActionRecurrenceRecoveryFreeze,
    source_qualification: GymToraxNativeSourceQualificationReceipt,
    extraction_manifest: GymToraxBoundedExtractionManifest,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    source_authority: StudyOperationAuthority,
    execution_authority: StudyOperationAuthority,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    repository_root: Path,
    at_utc: str,
    environment_factory: GymToraxEnvironmentFactory | None = None,
    runtime_inspector: GymToraxRuntimeInspector = inspect_gym_torax_runtime,
    episode_acquirer: Callable[[GymToraxFieldMetadataEpisodeRequest], GymToraxFieldMetadataNativeEpisode]
    | None = None,
) -> tuple[GymToraxFiniteActionRecurrenceRecoveryCellReceipt, ...]:
    if _load_freeze(store) != freeze:
        raise RuntimeError("GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_FREEZE_DIVERGED")
    _require_authority(
        freeze=freeze,
        authority=source_authority,
        kind=StudyAuthorityKind.SOURCE_ACQUISITION,
        at_utc=at_utc,
    )
    _require_authority(
        freeze=freeze,
        authority=execution_authority,
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        at_utc=at_utc,
        prerequisite=freeze.scientific_approval,
    )
    _require_authority(
        freeze=freeze,
        authority=custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        at_utc=at_utc,
        store=store,
    )
    if freeze.source_qualification != _identity(
        source_qualification.receipt_id, source_qualification
    ):
        raise ValueError("Finite-action recurrence recovery source qualification differs")
    receipts = []
    for cell in freeze.cells:
        request = materialize_gym_torax_finite_action_recurrence_recovery_request(freeze, cell)
        request_path = f"{freeze.relative_root}/requests/{request.request_id}.json"
        episode_path = f"{freeze.relative_root}/episodes/{cell.episode_id}.json"
        receipt_path = f"{freeze.relative_root}/receipts/{cell.cell_id}.json"
        recovered_receipt = store.load_optional(
            logical_artifact_id=f"artifact.receipt.{cell.cell_id}",
            relative_path=receipt_path,
            record_type=GymToraxFiniteActionRecurrenceRecoveryCellReceipt,
            maximum_bytes=2 * 1024**2,
        )
        if recovered_receipt is not None:
            receipts.append(recovered_receipt)
            continue
        recovered_request = store.load_optional(
            logical_artifact_id=f"artifact.{request.request_id}",
            relative_path=request_path,
            record_type=GymToraxFieldMetadataEpisodeRequest,
            maximum_bytes=2 * 1024**2,
        )
        if recovered_request is not None:
            raise RuntimeError("GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_INTERRUPTED_REQUEST_NO_RETRY")
        store.publish_atomic(
            publication_scope_id=f"publication.{freeze.run_id}",
            publication_scope_relative_root=freeze.relative_root,
            implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
            items=(
                GymToraxArtifactPublicationItem(
                    logical_artifact_id=f"artifact.{request.request_id}",
                    relative_path=request_path,
                    record=request,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.EVALUATION_SEALED,
                    # The visible declaration already binds this exact request
                    # hash. The fresh acquisition remains sealed from the old,
                    # revealed outcome lineage and is joined only at reveal.
                    lineage_parents=tuple(
                        sorted(
                            (
                                _parent(
                                    freeze.extraction_manifest,
                                    OutcomeAccess.OUTCOME_BLIND,
                                ),
                                _parent(
                                    freeze.field_metadata_manifest,
                                    OutcomeAccess.OUTCOME_BLIND,
                                ),
                                _parent(
                                    freeze.source_qualification,
                                    OutcomeAccess.OUTCOME_BLIND,
                                ),
                            ),
                            key=lambda value: value.identity.object_id,
                        )
                    ),
                ),
            ),
        )
        source_reset_attempted = False

        def observed_reset() -> None:
            nonlocal source_reset_attempted
            source_reset_attempted = True

        try:
            episode = (
                episode_acquirer(request)
                if episode_acquirer
                else acquire_gym_torax_episode(
                    request,
                    extraction_manifest=extraction_manifest,
                    field_metadata_manifest=field_metadata_manifest,
                    repository_root=repository_root,
                    environment_factory=environment_factory,
                    runtime_inspector=runtime_inspector,
                    source_reset_observer=observed_reset,
                )
            )
            if (
                episode.episode_id != cell.episode_id
                or episode.request != _identity(request.request_id, request)
                or episode.preparation
                != _identity(request.preparation.preparation_id, request.preparation)
            ):
                raise RuntimeError("GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_EPISODE_LINEAGE_DIVERGED")
            store.publish_atomic(
                publication_scope_id=f"publication.{freeze.run_id}",
                publication_scope_relative_root=freeze.relative_root,
                implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
                items=(
                    GymToraxArtifactPublicationItem(
                        logical_artifact_id=f"artifact.{cell.episode_id}",
                        relative_path=episode_path,
                        record=episode,
                        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                        outcome_access=OutcomeAccess.EVALUATION_SEALED,
                        lineage_parents=(
                            _parent(
                                _identity(request.request_id, request),
                                OutcomeAccess.EVALUATION_SEALED,
                            ),
                        ),
                    ),
                ),
            )
            disposition = GymToraxOperationalDisposition.SUCCEEDED
            complete_valid = _is_complete_valid(episode)
            operational_reasons: tuple[str, ...] = ()
            episode_reasons = episode.reason_codes
            detail_sha = None
        except Exception as error:
            episode = None
            disposition = GymToraxOperationalDisposition.FAILED
            complete_valid = False
            operational_reasons, detail_sha = _technical_reason(error)
            episode_reasons = ()
        receipt = GymToraxFiniteActionRecurrenceRecoveryCellReceipt(
            receipt_id=f"receipt.{cell.cell_id.removeprefix('cell.')}",
            freeze=_identity(freeze.freeze_id, freeze),
            cell=_identity(cell.cell_id, cell),
            request=_identity(request.request_id, request),
            episode=None if episode is None else _identity(episode.episode_id, episode),
            disposition=disposition,
            complete_valid_episode=complete_valid,
            operational_reason_codes=operational_reasons,
            episode_reason_codes=episode_reasons,
            technical_detail_sha256=detail_sha,
            attempt_count=1,
            automatic_retry_performed=False,
            source_reset_attempted=(
                source_reset_attempted
                if episode is None
                else episode.source_reset_attempted
            ),
            scientific_values_decoded=False,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        )
        store.publish_atomic(
            publication_scope_id=f"publication.{freeze.run_id}",
            publication_scope_relative_root=freeze.relative_root,
            implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
            items=(
                GymToraxArtifactPublicationItem(
                    logical_artifact_id=f"artifact.receipt.{cell.cell_id}",
                    relative_path=receipt_path,
                    record=receipt,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.EVALUATION_SEALED,
                    lineage_parents=tuple(
                        sorted(
                            (
                                _parent(
                                    _identity(request.request_id, request),
                                    OutcomeAccess.EVALUATION_SEALED,
                                ),
                                *(
                                    ()
                                    if episode is None
                                    else (
                                        _parent(
                                            _identity(episode.episode_id, episode),
                                            OutcomeAccess.EVALUATION_SEALED,
                                        ),
                                    )
                                ),
                            ),
                            key=lambda value: value.identity.object_id,
                        )
                    ),
                ),
            ),
        )
        receipts.append(receipt)
        if receipt.disposition is GymToraxOperationalDisposition.FAILED:
            break
    return tuple(sorted(receipts, key=lambda value: value.cell.object_id))


def finalize_gym_torax_finite_action_recurrence_recovery(
    *,
    freeze: GymToraxFiniteActionRecurrenceRecoveryFreeze,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> GymToraxFiniteActionRecurrenceRecoveryTerminalReceipt:
    _require_authority(
        freeze=freeze,
        authority=custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        at_utc=at_utc,
        store=store,
    )
    receipts = []
    episodes = []
    for cell in freeze.cells:
        receipt = store.load_optional(
            logical_artifact_id=f"artifact.receipt.{cell.cell_id}",
            relative_path=f"{freeze.relative_root}/receipts/{cell.cell_id}.json",
            record_type=GymToraxFiniteActionRecurrenceRecoveryCellReceipt,
            maximum_bytes=2 * 1024**2,
        )
        if receipt is None:
            raise RuntimeError(f"GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_RECEIPT_MISSING:{cell.cell_id}")
        receipts.append(receipt)
        if receipt.episode is not None:
            episodes.append(receipt.episode)
    partial = sum(
        value.disposition is GymToraxOperationalDisposition.SUCCEEDED
        and not value.complete_valid_episode
        for value in receipts
    )
    failed = sum(
        value.disposition is GymToraxOperationalDisposition.FAILED for value in receipts
    )
    valid = len(episodes) - partial
    terminal = GymToraxFiniteActionRecurrenceRecoveryTerminalReceipt(
        receipt_id='receipt.tokamak-control.finite-action-recurrence-recovery.terminal',
        freeze=_identity(freeze.freeze_id, freeze),
        cell_receipts=tuple(
            sorted(
                (_identity(value.receipt_id, value) for value in receipts),
                key=lambda value: value.object_id,
            )
        ),
        episodes=tuple(sorted(episodes, key=lambda value: value.object_id)),
        accounted_cell_count=80,
        episode_count=len(episodes),
        complete_valid_episode_count=valid,
        partial_episode_count=partial,
        failed_cell_count=failed,
        disposition=(
            GymToraxFiniteActionRecurrenceRecoveryTerminalDisposition.INCOMPLETE
            if failed
            else (
                GymToraxFiniteActionRecurrenceRecoveryTerminalDisposition.COMPLETE_WITH_PARTIAL
                if partial
                else GymToraxFiniteActionRecurrenceRecoveryTerminalDisposition.COMPLETE_VALID
            )
        ),
        reason_codes=tuple(
            sorted(
                ({"GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_TECHNICAL_CELL_FAILURE"} if failed else set())
                | ({"GYM_TORAX_FINITE_ACTION_RECURRENCE_RECOVERY_PARTIAL_EPISODE_EVIDENCE"} if partial else set())
            )
        ),
        scientific_values_decoded=False,
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )
    store.publish_atomic(
        publication_scope_id=f"publication.{freeze.run_id}",
        publication_scope_relative_root=freeze.relative_root,
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id=f"artifact.{freeze.run_id}.run-receipt",
                relative_path=f"{freeze.relative_root}/run-receipt.json",
                record=terminal,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
                lineage_parents=tuple(
                    _parent(value, OutcomeAccess.EVALUATION_SEALED)
                    for value in terminal.cell_receipts
                ),
            ),
        ),
    )
    return terminal


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceRecoveryEvaluationBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-recovery-evaluation-bundle'
    )

    bundle_id: str
    freeze: ObjectIdentity
    parent_qualification: ObjectIdentity
    terminal: ObjectIdentity
    reveal_authority: ObjectIdentity
    sealed_bundle: FiniteActionRecurrenceProspectiveBundle | None
    revealed_bundle: FiniteActionRecurrenceRevealedBundle | None
    adjudication: FiniteActionRecurrenceAdjudication | None
    result: FiniteActionRecurrenceResult
    episode_count_read: int
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        complete = self.sealed_bundle is not None and self.revealed_bundle is not None
        if complete != (self.adjudication is not None):
            raise ValueError("Finite-action recurrence recovery evaluation products are incomplete")
        if (
            self.adjudication is not None
            and self.adjudication.result is not self.result
        ):
            raise ValueError("Finite-action recurrence recovery outer result differs from sole evaluator")
        if (
            not 0 <= self.episode_count_read <= 80
            or self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("Finite-action recurrence recovery evaluation changes its nonpromotable lane")


def _q_values(episode: GymToraxFieldMetadataNativeEpisode) -> np.ndarray:
    blocks = tuple(
        value
        for value in episode.blocks
        if value.category == "source-scalar" and value.native_field_id == "Q_fusion"
    )
    if len(blocks) != 1:
        raise ValueError("Finite-action recurrence recovery complete episode lacks Q_fusion")
    return np.asarray(blocks[0].array()[:, 0], dtype=np.float64)


def evaluate_gym_torax_finite_action_recurrence_recovery(
    *,
    freeze: GymToraxFiniteActionRecurrenceRecoveryFreeze,
    parent_qualification: GymToraxFiniteActionRecurrenceQualificationBundle,
    terminal: GymToraxFiniteActionRecurrenceRecoveryTerminalReceipt,
    episodes: Mapping[str, GymToraxFieldMetadataNativeEpisode],
    manifests: Mapping[str, ArtifactManifest],
    metadata: GymToraxFieldMetadataManifest,
    reveal_authority: StudyOperationAuthority,
    execution_authority: StudyOperationAuthority,
    at_utc: str,
) -> GymToraxFiniteActionRecurrenceRecoveryEvaluationBundle:
    _require_authority(
        freeze=freeze,
        authority=reveal_authority,
        kind=StudyAuthorityKind.OUTCOME_REVEAL,
        at_utc=at_utc,
        prerequisite=_identity(execution_authority.authority_id, execution_authority),
    )
    if (
        ObjectIdentity.from_record(parent_qualification.bundle_id, parent_qualification)
        != freeze.parent_qualification
    ):
        raise ValueError("Finite-action recurrence recovery evaluation received another qualification")
    if terminal.episode_count != 80:
        return GymToraxFiniteActionRecurrenceRecoveryEvaluationBundle(
            bundle_id='evaluation-bundle.tokamak-control.finite-action-recurrence-recovery',
            freeze=_identity(freeze.freeze_id, freeze),
            parent_qualification=_identity(
                parent_qualification.bundle_id, parent_qualification
            ),
            terminal=_identity(terminal.receipt_id, terminal),
            reveal_authority=_identity(reveal_authority.authority_id, reveal_authority),
            sealed_bundle=None,
            revealed_bundle=None,
            adjudication=None,
            result=FiniteActionRecurrenceResult.AUTHORITY_OR_RESOURCE_STOP,
            episode_count_read=len(episodes),
            reason_codes=("FINITE_ACTION_RECURRENCE_RECOVERY_EXACT_80_EPISODES_UNAVAILABLE",),
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        )

    routes = {value.episode_id: value for value in freeze.assignment.routes}
    cells = {value.episode_id: value for value in freeze.cells}
    unit_specs = {value.unit_id: value for value in freeze.assignment.units}
    prepared_by_coordinate = {value.source_cell_id: value for value in freeze.units}
    sealed_rows = []
    q_by_episode: dict[str, np.ndarray] = {}
    complete_by_episode = {
        key: _is_complete_valid(value) for key, value in episodes.items()
    }
    for episode_id, route in routes.items():
        episode = episodes[episode_id]
        cell = cells[episode_id]
        complete = complete_by_episode[episode_id]
        if complete:
            _validate_hfr_episode(
                freeze=freeze,  # type: ignore[arg-type]
                cell=cell,
                episode=episode,
                metadata=metadata,
            )
            q_by_episode[episode_id] = _q_values(episode)
        unit = unit_specs[route.unit_id]
        prepared = prepared_by_coordinate[unit.preparation_coordinate_id]
        member = next(
            value
            for value in freeze.numerical_members
            if value.member_id == route.model_member_id
        )
        manifest = manifests[f"artifact.{episode_id}"]
        prefix_complete = all(value in episode.state_clocks for value in range(105))
        pair_incomplete = False
        if route.role in {
            FiniteActionRecurrenceRouteRole.ACTIVE,
            FiniteActionRecurrenceRouteRole.MATCHED_HOLD,
        }:
            mate_role = (
                FiniteActionRecurrenceRouteRole.MATCHED_HOLD
                if route.role is FiniteActionRecurrenceRouteRole.ACTIVE
                else FiniteActionRecurrenceRouteRole.ACTIVE
            )
            mate = next(
                value
                for value in freeze.assignment.routes
                if value.unit_id == route.unit_id
                and value.model_member_id == route.model_member_id
                and value.role is mate_role
            )
            pair_incomplete = not prefix_complete or not all(
                value in episodes[mate.episode_id].state_clocks for value in range(105)
            )
        nonattempt = pair_incomplete or (
            route.role is FiniteActionRecurrenceRouteRole.HOLD_CONTROL
            and not prefix_complete
        )
        config_sha = hashlib.sha256(
            (
                prepared.preparation.fingerprint()
                + ":"
                + member.fingerprint()
                + ':tokamak-control.action-neutral-config'
            ).encode()
        ).hexdigest()
        sealed_rows.append(
            FiniteActionRecurrenceSealedEpisode(
                episode_id=episode_id,
                route=route,
                preparation_artifact=_content_artifact(
                    artifact_id=f"artifact.preparation.{episode_id}",
                    role="preparation",
                    payload_schema='empirical-lawhood/simulators/gym-torax-native/finite-action-recurrence/preparation-artifact',
                    sha256=unit.preparation_sha256,
                ),
                config_artifact=_content_artifact(
                    artifact_id=f"artifact.config.{episode_id}",
                    role="action-neutral-config",
                    payload_schema='empirical-lawhood/simulators/gym-torax-native/finite-action-recurrence/action-neutral-config',
                    sha256=config_sha,
                ),
                preaction_prefix_artifact=_content_artifact(
                    artifact_id=f"artifact.prefix.{episode_id}",
                    role="preaction-prefix",
                    payload_schema='empirical-lawhood/simulators/gym-torax-native/finite-action-recurrence/preaction-prefix',
                    sha256=_prefix_sha(episode),
                ),
                prefix_complete_through_state_104=prefix_complete,
                preaction_disposition=(
                    FiniteActionPreactionDisposition.NONATTEMPT
                    if nonattempt
                    else FiniteActionPreactionDisposition.READY
                ),
                observed_occurrences=(
                    ()
                    if nonattempt
                    else _observed(
                        freeze.assignment.active_word
                        if route.role is FiniteActionRecurrenceRouteRole.ACTIVE
                        else freeze.assignment.matched_hold_word,
                        episode_id,
                    )
                ),
                clipped=False,
                rejected=False,
                substituted=False,
                early_terminated=(
                    False if nonattempt else episode.termination or episode.truncation
                ),
                delivery_trace=_episode_identity(episode),
                sealed_outcome_artifact=_artifact(manifest),
                custody_evidence=(
                    _identity(
                        manifest.materialization.materialization_id,
                        manifest.materialization,
                    ),
                ),
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
            )
        )
    sealed = FiniteActionRecurrenceProspectiveBundle(
        bundle_id='sealed-bundle.tokamak-control.finite-action-recurrence-recovery',
        assignment=freeze.assignment,
        episodes=tuple(sorted(sealed_rows, key=lambda value: value.episode_id)),
        issue_receipt=_identity(freeze.freeze_id, freeze),
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )
    predicate_kinds = tuple(
        sorted(
            {
                predicate.kind
                for receipt in freeze.assignment.occurrence_receipts
                for predicate in receipt.qualification_spec.predicates
            },
            key=lambda value: value.value,
        )
    )
    revealed_rows = []
    for source in sealed.episodes:
        episode = episodes[source.episode_id]
        complete = complete_by_episode[source.episode_id]
        q = q_by_episode.get(source.episode_id)
        samples = (
            ()
            if q is None
            or source.route.role is FiniteActionRecurrenceRouteRole.HOLD_CONTROL
            else tuple(
                FiniteActionRecurrenceResponseSample(
                    sample_id=f"sample.{source.episode_id}.{index:02d}",
                    quantity_id=freeze.assignment.effect_quantity_id,
                    coordinate=coordinate,
                    value=NamedDecimal(
                        value_id=f"value.{source.episode_id}.{index:02d}",
                        value=Decimal(str(float(q[int(coordinate.coordinate)]))),
                        unit=freeze.assignment.effect_native_unit,
                    ),
                )
                for index, coordinate in enumerate(
                    freeze.assignment.response_coordinates
                )
            )
        )
        predicates = tuple(
            FiniteActionRecurrencePredicateResult(
                result_id=f"predicate-result.{source.episode_id}.{kind.value.lower().replace('_', '-')}",
                kind=kind,
                status=(GateStatus.PASS if complete else GateStatus.UNEVALUABLE),
                reason_codes=() if complete else ("FINITE_ACTION_RECURRENCE_RECOVERY_EPISODE_INCOMPLETE",),
            )
            for kind in predicate_kinds
        )
        revealed_rows.append(
            FiniteActionRecurrenceRevealedEpisode(
                reveal_id=f"reveal.{source.episode_id}",
                episode_id=source.episode_id,
                sealed_episode=_identity(source.episode_id, source),
                delivery_trace=source.delivery_trace,
                sealed_outcome_artifact=source.sealed_outcome_artifact,
                outcome_trace_id=f"outcome-trace.{source.episode_id}",
                response_samples=samples,
                predicate_results=predicates,
                technical_reason_codes=()
                if complete
                else tuple(
                    sorted({"FINITE_ACTION_RECURRENCE_RECOVERY_TECHNICAL_PARTIAL", *episode.reason_codes})
                ),
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            )
        )
    revealed = FiniteActionRecurrenceRevealedBundle(
        reveal_id='revealed-bundle.tokamak-control.finite-action-recurrence-recovery',
        sealed_bundle=sealed,
        integrity=FiniteActionRecurrenceRevealIntegrity.VALID,
        reveal_authorization=_identity(reveal_authority.authority_id, reveal_authority),
        integrity_evidence=tuple(
            sorted(
                (
                    _identity(reveal_authority.authority_id, reveal_authority),
                    _identity(terminal.receipt_id, terminal),
                ),
                key=lambda value: value.object_id,
            )
        ),
        integrity_reason_codes=(),
        episodes=tuple(sorted(revealed_rows, key=lambda value: value.episode_id)),
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    adjudication = MatchedFiniteActionHoldRecurrenceEvaluator().evaluate(
        assignment=freeze.assignment, revealed=revealed
    )
    return GymToraxFiniteActionRecurrenceRecoveryEvaluationBundle(
        bundle_id='evaluation-bundle.tokamak-control.finite-action-recurrence-recovery',
        freeze=_identity(freeze.freeze_id, freeze),
        parent_qualification=_identity(
            parent_qualification.bundle_id, parent_qualification
        ),
        terminal=_identity(terminal.receipt_id, terminal),
        reveal_authority=_identity(reveal_authority.authority_id, reveal_authority),
        sealed_bundle=sealed,
        revealed_bundle=revealed,
        adjudication=adjudication,
        result=adjudication.result,
        episode_count_read=80,
        reason_codes=adjudication.reason_codes,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


@dataclass(frozen=True, slots=True)
class GymToraxPostHocEffectSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-post-hoc-effect-summary'
    )

    summary_id: str
    cell_kind: str
    member_id: str
    action_word_id: str
    physical_unit_count: int
    mean_effect: NamedDecimal
    median_effect: NamedDecimal
    minimum_effect: NamedDecimal
    maximum_effect: NamedDecimal
    positive_count: int
    onset_state_minimum: int | None
    onset_state_median: Decimal | None
    onset_state_maximum: int | None

    def __post_init__(self) -> None:
        for name in ("summary_id", "cell_kind", "member_id", "action_word_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.physical_unit_count != 18 or not 0 <= self.positive_count <= 18:
            raise ValueError("Gym-TORAX posthoc summary inflates or drops physical units")


@dataclass(frozen=True, slots=True)
class GymToraxPostHocAnalysis(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-post-hoc-analysis'

    analysis_id: str
    selected_parent_freeze: ObjectIdentity
    selected_parent_science: ObjectIdentity
    hfr_qualification: ObjectIdentity
    hfr_terminal: ObjectIdentity
    recovery_freeze: ObjectIdentity
    selected_parent_episode_count_read: int
    selected_parent_physical_unit_count: int
    hfr_qualification_episode_count: int
    hfr_failed_recurrence_count: int
    occurrence_timing_rows: tuple[NamedDecimal, ...]
    effect_summaries: tuple[GymToraxPostHocEffectSummary, ...]
    qualification_reason_codes: tuple[str, ...]
    failure_reason_counts: tuple[NamedDecimal, ...]
    no_feedback_into_recovery_freeze: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        require_sorted_unique_ids(
            self.effect_summaries, attribute="summary_id", field_name="effect_summaries"
        )
        require_sorted_unique_strings(
            self.qualification_reason_codes, field_name="qualification_reason_codes"
        )
        if (
            self.selected_parent_episode_count_read != 576
            or self.selected_parent_physical_unit_count != 72
            or self.hfr_qualification_episode_count != 18
            or self.hfr_failed_recurrence_count != 80
            or not self.no_feedback_into_recovery_freeze
            or self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError(
                "Gym-TORAX posthoc analysis changes its nonpromotable exact roster"
            )


def analyze_gym_torax_posthoc(
    *,
    selected_parent_freeze: GymToraxSelectedParentFreeze,
    selected_parent_science: GymToraxSelectedParentScienceEvaluationBundle,
    qualification: GymToraxFiniteActionRecurrenceQualificationBundle,
    hfr_terminal: GymToraxFiniteActionRecurrenceTerminalAcquisitionReceipt,
    recovery_freeze: GymToraxFiniteActionRecurrenceRecoveryFreeze,
    selected_parent_episode_loader: Callable[[str], GymToraxFieldMetadataNativeEpisode],
    failure_reason_counts: Mapping[str, int],
) -> GymToraxPostHocAnalysis:
    q_by_product: dict[tuple[str, str, str], np.ndarray] = {}
    for cell in selected_parent_freeze.cells:
        episode = selected_parent_episode_loader(cell.episode_id)
        if not _is_complete_valid(episode):
            raise ValueError(f'posthoc measurement episode is not complete:{cell.episode_id}')
        q_by_product[
            (
                cell.physical_unit_instance_id,
                cell.numerical_member_id,
                cell.action_word_id,
            )
        ] = _q_values(episode)
    grouped: dict[tuple[str, str, str], list[tuple[Decimal, int | None]]] = defaultdict(
        list
    )
    hold_id = 'action-word.tokamak-control.native-hold'
    for unit in selected_parent_freeze.units:
        for member in ('member.tokamak-control.primary', 'member.tokamak-control.refined'):
            hold = q_by_product[(unit.unit_id, member, hold_id)]
            for action in (
                'action-word.tokamak-control.future-ip',
                'action-word.tokamak-control.lower-ip',
                'action-word.tokamak-control.wrong-sign-ip',
            ):
                value = q_by_product[(unit.unit_id, member, action)]
                effect = Decimal(
                    str(float(np.mean(value[105:111]) - np.mean(hold[105:111])))
                )
                changed = np.flatnonzero(np.abs(value - hold) > 1e-12)
                onset = None if not len(changed) else int(changed[0])
                grouped[(unit.cell_kind, member, action)].append((effect, onset))
    summaries = []
    for (kind, member, action), rows in sorted(grouped.items()):
        effects = [value for value, _ in rows]
        onsets = [value for _, value in rows if value is not None]
        stem = f"{kind}.{member.rsplit('.', 1)[-1]}.{action.rsplit('.', 1)[-1]}"
        summaries.append(
            GymToraxPostHocEffectSummary(
                summary_id=f'summary.tokamak-control.posthoc-analysis.{stem}',
                cell_kind=kind,
                member_id=member,
                action_word_id=action,
                physical_unit_count=len(rows),
                mean_effect=NamedDecimal(
                    f"mean.{stem}", sum(effects, Decimal(0)) / len(effects), "1"
                ),
                median_effect=NamedDecimal(
                    f"median.{stem}", Decimal(str(median(effects))), "1"
                ),
                minimum_effect=NamedDecimal(f"minimum.{stem}", min(effects), "1"),
                maximum_effect=NamedDecimal(f"maximum.{stem}", max(effects), "1"),
                positive_count=sum(value > 0 for value in effects),
                onset_state_minimum=min(onsets) if onsets else None,
                onset_state_median=(Decimal(str(median(onsets))) if onsets else None),
                onset_state_maximum=max(onsets) if onsets else None,
            )
        )
    timings: list[NamedDecimal] = []
    for receipt in qualification.occurrence_receipts:
        stem = receipt.receipt_id.removeprefix('receipt.finite-action-occurrence.finite-action-recurrence.')
        active_first = receipt.evidence.active_first_difference_state
        future_first = receipt.evidence.future_first_difference_state
        future_hold_max = (
            receipt.evidence.future_hold_max_abs_difference_through_state_110
        )
        if active_first is None or future_first is None or future_hold_max is None:
            raise ValueError(
                "supported finite-action recurrence qualification lacks occurrence timing evidence"
            )
        timings.extend(
            (
                NamedDecimal(
                    f"timing.{stem}.active-first", Decimal(active_first), "state-clock"
                ),
                NamedDecimal(
                    f"timing.{stem}.future-first", Decimal(future_first), "state-clock"
                ),
                NamedDecimal(f"timing.{stem}.future-hold-max", future_hold_max, "1"),
            )
        )
    return GymToraxPostHocAnalysis(
        analysis_id='analysis.tokamak-control.finite-action-posthoc-analysis',
        selected_parent_freeze=_identity(selected_parent_freeze.freeze_id, selected_parent_freeze),
        selected_parent_science=_identity(selected_parent_science.bundle_id, selected_parent_science),
        hfr_qualification=_identity(qualification.bundle_id, qualification),
        hfr_terminal=_identity(hfr_terminal.receipt_id, hfr_terminal),
        recovery_freeze=_identity(recovery_freeze.freeze_id, recovery_freeze),
        selected_parent_episode_count_read=576,
        selected_parent_physical_unit_count=72,
        hfr_qualification_episode_count=18,
        hfr_failed_recurrence_count=80,
        occurrence_timing_rows=tuple(sorted(timings, key=lambda value: value.value_id)),
        effect_summaries=tuple(sorted(summaries, key=lambda value: value.summary_id)),
        qualification_reason_codes=(),
        failure_reason_counts=tuple(
            NamedDecimal(
                value_id=f"failure-count.{reason.lower().replace('_', '-')}",
                value=Decimal(count),
                unit="cell",
            )
            for reason, count in sorted(failure_reason_counts.items())
        ),
        no_feedback_into_recovery_freeze=True,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


__all__ = [
    name
    for name in globals()
    if name.startswith('TOKAMAK-CONTROL')
    or name.startswith("analyze_")
    or name.startswith("build_")
    or name.startswith("evaluate_")
    or name.startswith("finalize_")
    or name.startswith("materialize_")
    or name.startswith("publish_")
    or name.startswith("run_")
]

"Prospective correction and exact native roster for the selected Gym-TORAX finite-action recurrence route.\n\nThe immutable pre-qualification finite-action recurrence assigned a different environment seed to\nevery nested member/action view while calling the enclosing coordinate one\nphysical unit.  This module preserves that record as provenance and, before\nthe corrected cohort's measurement-through-law-qualification reveal, issues an additive correction with one\nfresh seed shared by all views of each physical unit.  It also makes explicit the native-HOLD\ncalibration act that the pre-qualification template required but did not assign.\n\nNo threshold, action, support tier, stratum, primary unit, control anchor,\nreducer, or terminal rule is selected here.  The parser accepts only the\nauthenticated Gym-TORAX source roster and prospective protocol bytes.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
import hashlib
import json
from typing import Any, ClassVar, cast

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.planning.finite_action_alternate_branch_selection import FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_BRANCH_ID, FiniteActionAlternateBranchSelectionBridge
from empirical_lawhood.planning.finite_action_occurrence import FiniteActionOccurrenceQualificationTemplateSet
from empirical_lawhood.planning.finite_action_recurrence import FiniteActionRecurrenceAssignmentTemplate, validate_finite_action_occurrence_recurrence_expectations
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind

from .retained_inputs import GymToraxImportedProspectiveTopology, GymToraxPredecessorInputs
from .protocol_scientific_inputs import (
    GymToraxProtocolEnvironmentSeedInput,
    require_protocol_root_seed,
    require_protocol_seed_census,
)

from .action_word import build_gym_torax_action_word, build_gym_torax_native_schedule
from .diagnostic_contracts import GymToraxNumericalMember, GymToraxPreparation, gym_torax_numerical_members
from .field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxRequestRole
from .extraction_manifest import GymToraxBoundedExtractionManifest
from .field_metadata import GymToraxFieldMetadataManifest
from .selected_parent_protocol import GymToraxSelectedParentFreeze, GymToraxSourceQualificationCarryforwardEntry
from .selected_parent_science import GymToraxSelectedParentScienceFreeze
from .source_assessment_adjudication import GymToraxSourceAssessmentBranchSelectionDisposition, GymToraxSourceAssessmentBranchSelectionReceipt
from .source_qualification import GymToraxNativeSourceQualificationDisposition, GymToraxNativeSourceQualificationReceipt


GYM_TORAX_FINITE_ACTION_RECURRENCE_FREEZE_ID = 'freeze.tokamak-control.finite-action-recurrence'
GYM_TORAX_FINITE_ACTION_RECURRENCE_RUN_ID = 'run.tokamak-control.finite-action-recurrence'
GYM_TORAX_FINITE_ACTION_RECURRENCE_RELATIVE_ROOT = (
    'tokamak-control/publication/runs/run.tokamak-control.finite-action-recurrence'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_SELECTION_BRIDGE_ID = 'selection-bridge.tokamak-control.finite-action-recurrence'
GYM_TORAX_FINITE_ACTION_RECURRENCE_OCCURRENCE_TEMPLATE_SET_ID = (
    'template-set.tokamak-control.finite-action-occurrence.finite-action-recurrence'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_RECURRENCE_TEMPLATE_ID = (
    'template.tokamak-control.finite-action-recurrence-assignment'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_ASSIGNMENT_ID = 'assignment.tokamak-control.finite-action-recurrence'
GYM_TORAX_FINITE_ACTION_RECURRENCE_CALIBRATION_RECEIPT_ID = (
    'receipt.tokamak-control.native-hold-decision-cell-calibration.finite-action-recurrence'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_ROUTE_QUALIFICATION_RECEIPT_ID = (
    'route-qualification.tokamak-control.finite-action-recurrence'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_LAW_QUALIFICATION_RESULT_ID = 'qualification-result.tokamak-control.matched-evaluation'
GYM_TORAX_FINITE_ACTION_RECURRENCE_REPAIR_RULE_ID = (
    'repair.tokamak-control.shared-seed-plus-explicit-hold-calibration'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_AUTHORITY_IDS = tuple(
    sorted(
        (
            'authority.tokamak-control.finite-action-recurrence.custody-publication',
            'authority.tokamak-control.finite-action-recurrence.experiment-execution',
            'authority.tokamak-control.finite-action-recurrence.measurement-occurrence-reveal',
            'authority.tokamak-control.finite-action-recurrence.route-outcome-reveal',
            'authority.tokamak-control.finite-action-recurrence.recurrence-outcome-reveal',
            'authority.tokamak-control.finite-action-recurrence.source-acquisition',
        )
    )
)

_PREPARATION_VALUE_IDS = {
    'tokamak-control.preparation.bootstrap-multiplier',
    'tokamak-control.preparation.initial-density-nbar',
    'tokamak-control.preparation.initial-temperature-scale',
    'tokamak-control.preparation.inner-transport-scale',
}
_MEMBERS = ('member.tokamak-control.primary', 'member.tokamak-control.refined')
_HOLD_WORD_ID = 'action-word.tokamak-control.native-hold'
_ACTIVE_WORD_ID = 'action-word.tokamak-control.lower-ip'
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
    wall_time_seconds=99 * 12 * 60 * 60,
    source_scan_bytes=0,
    output_bytes=8 * 1024**3,
)


class GymToraxFiniteActionRecurrenceCellStage(StrEnum):
    HOLD_CALIBRATION = "HOLD_CALIBRATION"
    RECURRENCE = "RECURRENCE"
    ROUTE_QUALIFICATION = "ROUTE_QUALIFICATION"


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrencePreparedUnit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-prepared-unit'

    unit_id: str
    source_cell_id: str
    source_preparation_fingerprint: str
    prospective_role: str
    source_anchor_slot_id: str | None
    stratum_id: str | None
    local_support_id: str | None
    preparation: GymToraxPreparation
    scientific_seed_input: GymToraxProtocolEnvironmentSeedInput
    shared_environment_seed: bool
    scientific_primary: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.unit_id, field_name="unit_id")
        validate_stable_id(self.source_cell_id, field_name="source_cell_id")
        validate_sha256(
            self.source_preparation_fingerprint,
            field_name="source_preparation_fingerprint",
        )
        if not isinstance(self.scientific_seed_input, GymToraxProtocolEnvironmentSeedInput):
            raise ValueError("recurrence preparation requires its typed scientific seed input")
        if self.scientific_seed_input.scientific_role == "matched-evaluation":
            raise ValueError("recurrence preparation cannot use the matched-evaluation seed rule")
        if self.scientific_seed_input.scientific_role == "recurrence-recovery":
            if not self.scientific_primary:
                raise ValueError("recovery seed requires a scientific primary physical root")
            scientific_role = "recurrence-recovery"
        else:
            scientific_role = (
                "recurrence-main" if self.scientific_primary
                else "hold-calibration" if self.prospective_role == "HOLD_CALIBRATION"
                else "recurrence-route"
            )
        self.scientific_seed_input.require_root(self.unit_id, scientific_role)
        if self.preparation.environment_seed != self.scientific_seed_input.environment_seed:
            raise ValueError("recurrence preparation changes its declared scientific seed")
        if self.preparation.physical_independent_unit_id != self.unit_id:
            raise ValueError("Finite-action recurrence preparation changes its physical unit")
        if not self.shared_environment_seed:
            raise ValueError("Finite-action recurrence nested routes require one shared environment seed")


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-cell'

    cell_id: str
    episode_id: str
    execution_id: str
    stage: GymToraxFiniteActionRecurrenceCellStage
    unit_id: str
    model_member_id: str
    route_role: str
    action_word_id: str
    request_id: str
    request_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "episode_id",
            "execution_id",
            "unit_id",
            "model_member_id",
            "route_role",
            "action_word_id",
            "request_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.request_sha256, field_name="request_sha256")
        if self.model_member_id not in _MEMBERS:
            raise ValueError("Finite-action recurrence cell names another numerical member")
        if self.action_word_id not in {_ACTIVE_WORD_ID, _HOLD_WORD_ID}:
            raise ValueError("Finite-action recurrence cell names another action word")


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceProspectiveFreeze(CanonicalRecord):
    predecessors: GymToraxPredecessorInputs
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-prospective-freeze'
    )
    VERSION: ClassVar[str] = '1.0.0'

    freeze_id: str
    run_id: str
    relative_root: str
    source_campaign_roster: ObjectIdentity
    source_prospective_protocol: ObjectIdentity
    original_occurrence_templates: ObjectIdentity
    original_recurrence_template: ObjectIdentity
    repair_rule_id: str
    selected_branch: ObjectIdentity
    selection_bridge: FiniteActionAlternateBranchSelectionBridge
    matched_evaluation_acquisition_freeze: ObjectIdentity
    matched_evaluation_science_freeze: ObjectIdentity
    corrected_occurrence_templates: FiniteActionOccurrenceQualificationTemplateSet
    corrected_recurrence_template: FiniteActionRecurrenceAssignmentTemplate
    source_qualification: ObjectIdentity
    source_qualification_carryforward: tuple[
        GymToraxSourceQualificationCarryforwardEntry, ...
    ]
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
    original_route_seed_count: int
    corrected_recurrence_seed_count: int
    calibration_unit_count: int
    route_qualification_episode_count: int
    recurrence_episode_count: int
    frozen_before_matched_evaluation_reveal: bool
    matched_evaluation_outcomes_read: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        for name in ("freeze_id", "run_id", "repair_rule_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_relative_locator(self.relative_root)
        if self.source_campaign_roster != self.predecessors.source_roster:
            raise ValueError("Finite-action recurrence freeze changes the authenticated source roster")
        if self.matched_evaluation_acquisition_freeze.object_schema != GymToraxSelectedParentFreeze.SCHEMA:
            raise ValueError("Finite-action recurrence freeze binds another selected-parent freeze")
        if self.matched_evaluation_science_freeze.object_schema != GymToraxSelectedParentScienceFreeze.SCHEMA:
            raise ValueError("Finite-action recurrence freeze binds another measurement through local law science design")
        if (
            self.source_qualification.object_schema
            != GymToraxNativeSourceQualificationReceipt.SCHEMA
        ):
            raise ValueError("Finite-action recurrence freeze lacks the exact native source qualification")
        if (
            self.extraction_manifest.object_schema
            != GymToraxBoundedExtractionManifest.SCHEMA
        ):
            raise ValueError("Finite-action recurrence freeze binds another extraction manifest")
        if (
            self.field_metadata_manifest.object_schema
            != GymToraxFieldMetadataManifest.SCHEMA
        ):
            raise ValueError("Finite-action recurrence freeze binds another field metadata manifest")
        if (
            self.implementation_source_closure.kind
            is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.implementation_source_closure.clean_worktree
            or not self.implementation_source_closure.implementation_sha256
        ):
            raise ValueError("Finite-action recurrence freeze requires a clean committed implementation")
        require_sorted_unique_ids(self.units, attribute="unit_id", field_name="units")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_sorted_unique_ids(
            self.numerical_members,
            attribute="member_id",
            field_name="numerical_members",
        )
        require_sorted_unique_strings(
            self.required_operation_authority_ids,
            field_name="required_operation_authority_ids",
            allow_empty=False,
        )
        if len(self.units) != 30 or len(self.cells) != 98:
            raise ValueError(
                "Finite-action recurrence corrected freeze requires 30 preparations and 98 episodes"
            )
        stage_counts = {
            stage: sum(cell.stage is stage for cell in self.cells)
            for stage in GymToraxFiniteActionRecurrenceCellStage
        }
        if stage_counts != {
            GymToraxFiniteActionRecurrenceCellStage.HOLD_CALIBRATION: 10,
            GymToraxFiniteActionRecurrenceCellStage.RECURRENCE: 80,
            GymToraxFiniteActionRecurrenceCellStage.ROUTE_QUALIFICATION: 8,
        }:
            raise ValueError(
                "Finite-action recurrence corrected freeze changes the exact 10+8+80 stage roster"
            )
        units = {value.unit_id: value for value in self.units}
        if set(cell.unit_id for cell in self.cells) - set(units):
            raise ValueError("Finite-action recurrence cell names an absent preparation unit")
        if any(
            len(
                {
                    units[cell.unit_id].preparation.environment_seed
                    for cell in self.cells
                    if cell.unit_id == unit_id
                }
            )
            != 1
            for unit_id in units
        ):
            raise ValueError("Finite-action recurrence nested views change their shared unit seed")
        if (
            self.original_route_seed_count != 80
            or self.corrected_recurrence_seed_count != 22
            or self.calibration_unit_count != 5
            or self.route_qualification_episode_count != 8
            or self.recurrence_episode_count != 80
        ):
            raise ValueError(
                "Finite-action recurrence correction accounting differs from the audited defect"
            )
        validate_finite_action_occurrence_recurrence_expectations(
            occurrence_templates=self.corrected_occurrence_templates,
            recurrence_template=self.corrected_recurrence_template,
        )
        if not self.frozen_before_matched_evaluation_reveal or self.matched_evaluation_outcomes_read:
            raise ValueError("Finite-action recurrence correction must freeze before access to outcomes from measurement through local law")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("Finite-action recurrence freeze changes its prospective nonpromotable ceiling")

    def unit_for(self, unit_id: str) -> GymToraxFiniteActionRecurrencePreparedUnit:
        return next(value for value in self.units if value.unit_id == unit_id)


def _unwrap(value: Any, *, schema: str | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Gym-TORAX finite-action recurrence input is not an object")
    if schema is not None and value.get("schema") != schema:
        raise ValueError(f"Gym-TORAX finite-action recurrence input has another schema:{schema}")
    body = value.get("value", value)
    if not isinstance(body, dict):
        raise ValueError("Gym-TORAX finite-action recurrence input value is not an object")
    return cast(dict[str, Any], body)


def _canonical_document(schema: str, value: dict[str, Any]) -> bytes:
    return (
        json.dumps(
            {"schema": schema, "value": value, "version": "1.0.0"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )


def _preparation_values(row: dict[str, Any]) -> dict[str, Decimal]:
    raw = row.get("preparation_values")
    if not isinstance(raw, list):
        raise ValueError("prospective source cell lacks preparation values")
    values: dict[str, Decimal] = {}
    for item in raw:
        body = _unwrap(item, schema='empirical-lawhood/kernel/named-decimal')
        value_id = body.get("value_id")
        decimal = body.get("value")
        if not isinstance(value_id, str) or not isinstance(decimal, dict):
            raise ValueError("prospective preparation value is malformed")
        values[value_id] = Decimal(str(decimal.get("decimal")))
    if set(values) != _PREPARATION_VALUE_IDS:
        raise ValueError("prospective preparation value roster differs")
    return values


def _preparation(
    *,
    source: dict[str, Any],
    unit_id: str,
    scientific_seed_input: GymToraxProtocolEnvironmentSeedInput,
) -> GymToraxPreparation:
    if not isinstance(scientific_seed_input, GymToraxProtocolEnvironmentSeedInput):
        raise ValueError("recurrence preparation requires its typed scientific seed input")
    if scientific_seed_input.scientific_role == "matched-evaluation":
        raise ValueError("recurrence preparation cannot use the matched-evaluation seed rule")
    scientific_seed_input.require_root(unit_id, scientific_seed_input.scientific_role)
    values = _preparation_values(source)
    return GymToraxPreparation(
        preparation_id=f"preparation.{unit_id.removeprefix('unit.')}",
        physical_independent_unit_id=unit_id,
        environment_seed=scientific_seed_input.environment_seed,
        initial_temperature_scale=values['tokamak-control.preparation.initial-temperature-scale'],
        initial_density_nbar=values['tokamak-control.preparation.initial-density-nbar'],
        bootstrap_multiplier=values['tokamak-control.preparation.bootstrap-multiplier'],
        inner_transport_scale=values['tokamak-control.preparation.inner-transport-scale'],
    )


def _corrected_templates(
    *,
    protocol: dict[str, Any],
    bridge: FiniteActionAlternateBranchSelectionBridge,
) -> tuple[
    ObjectIdentity,
    ObjectIdentity,
    FiniteActionOccurrenceQualificationTemplateSet,
    FiniteActionRecurrenceAssignmentTemplate,
]:
    raw_occurrence = cast(dict[str, Any], protocol["hfr_occurrence_design"])
    occurrence_value = json.loads(json.dumps(_unwrap(raw_occurrence)))
    original_occurrence = ObjectIdentity(
        object_id=cast(str, occurrence_value["template_set_id"]),
        object_schema=cast(str, raw_occurrence["schema"]),
        object_version=cast(str, raw_occurrence["version"]),
        object_fingerprint=hashlib.sha256(
            _canonical_document(raw_occurrence["schema"], occurrence_value)
        ).hexdigest(),
    )
    occurrence_value["template_set_id"] = GYM_TORAX_FINITE_ACTION_RECURRENCE_OCCURRENCE_TEMPLATE_SET_ID
    for wrapped in occurrence_value["templates"]:
        row = _unwrap(wrapped)
        template_id = row["template_id"]
        if (
            not isinstance(template_id, str)
            or not template_id.startswith("template.finite-action-occurrence.")
            or template_id == "template.finite-action-occurrence."
        ):
            raise ValueError("FINITE_ACTION_OCCURRENCE_TEMPLATE_ID_REQUIRES_CURRENT_SOURCE_EXPORT")
        suffix = template_id.removeprefix("template.finite-action-occurrence.")
        row["template_id"] = f'template.finite-action-occurrence.finite-action-recurrence.{suffix}'
        row["expected_qualification_spec_id"] = (
            f'spec.finite-action-occurrence.finite-action-recurrence.{suffix}'
        )
        row["expected_qualification_receipt_id"] = (
            f'receipt.finite-action-occurrence.finite-action-recurrence.{suffix}'
        )
        row["expected_branch_selection_receipt_id"] = bridge.receipt_id
        row["expected_branch_selection_receipt_schema"] = bridge.SCHEMA
        row['expected_law_qualification_result_id'] = GYM_TORAX_FINITE_ACTION_RECURRENCE_LAW_QUALIFICATION_RESULT_ID
    occurrence = decode_canonical_bytes(
        _canonical_document(raw_occurrence["schema"], occurrence_value),
        FiniteActionOccurrenceQualificationTemplateSet,
        maximum_bytes=4 * 1024**2,
    )

    raw_recurrence = cast(dict[str, Any], protocol["hfr_assignment_recurrence"])
    recurrence_value = json.loads(json.dumps(_unwrap(raw_recurrence)))
    original_recurrence = ObjectIdentity(
        object_id=cast(str, recurrence_value["template_id"]),
        object_schema=cast(str, raw_recurrence["schema"]),
        object_version=cast(str, raw_recurrence["version"]),
        object_fingerprint=hashlib.sha256(
            _canonical_document(raw_recurrence["schema"], recurrence_value)
        ).hexdigest(),
    )
    recurrence_value["template_id"] = GYM_TORAX_FINITE_ACTION_RECURRENCE_RECURRENCE_TEMPLATE_ID
    recurrence_value["expected_assignment_id"] = GYM_TORAX_FINITE_ACTION_RECURRENCE_ASSIGNMENT_ID
    recurrence_value["expected_branch_selection_receipt_id"] = bridge.receipt_id
    recurrence_value["expected_branch_selection_receipt_schema"] = bridge.SCHEMA
    recurrence_value["expected_occurrence_receipt_ids"] = [
        value.expected_qualification_receipt_id for value in occurrence.templates
    ]
    recurrence_value["expected_native_hold_calibration_receipt_id"] = (
        GYM_TORAX_FINITE_ACTION_RECURRENCE_CALIBRATION_RECEIPT_ID
    )
    recurrence_value["expected_route_qualification_receipt_id"] = (
        GYM_TORAX_FINITE_ACTION_RECURRENCE_ROUTE_QUALIFICATION_RECEIPT_ID
    )
    unit_seed_ids: dict[str, str] = {}
    for wrapped in recurrence_value["units"]:
        row = _unwrap(wrapped)
        coordinate_id = cast(str, row["preparation_coordinate_id"])
        fresh_physical_id = f"unit.tokamak-control.finite-action-prospective-unit.{coordinate_id.removeprefix('cell.tokamak-control.')}"
        row["physical_independent_unit_id"] = fresh_physical_id
        unit_seed_ids[cast(str, row["unit_id"])] = (
            f'environment-seed.{fresh_physical_id}.finite-action-recurrence'
        )
    for wrapped in recurrence_value["routes"]:
        row = _unwrap(wrapped)
        old_episode = cast(str, row["episode_id"])
        row["episode_id"] = f'{old_episode}.finite-action-recurrence'
        row["execution_id"] = f"{cast(str, row['execution_id'])}.finite-action-recurrence"
        row["environment_seed_id"] = unit_seed_ids[row["unit_id"]]
    recurrence = decode_canonical_bytes(
        _canonical_document(raw_recurrence["schema"], recurrence_value),
        FiniteActionRecurrenceAssignmentTemplate,
        maximum_bytes=8 * 1024**2,
    )
    return original_occurrence, original_recurrence, occurrence, recurrence


def materialize_gym_torax_finite_action_recurrence_request(
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    cell: GymToraxFiniteActionRecurrenceCell,
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
        raise ValueError("Finite-action recurrence request differs from its frozen bytes")
    return request


def build_gym_torax_finite_action_recurrence_prospective_freeze(
    *,
    predecessors: GymToraxPredecessorInputs,
    source_campaign_roster_bytes: bytes,
    prospective_protocol_bytes: bytes,
    branch_selection: GymToraxSourceAssessmentBranchSelectionReceipt,
    matched_evaluation_acquisition_freeze: GymToraxSelectedParentFreeze,
    matched_evaluation_science_freeze: GymToraxSelectedParentScienceFreeze,
    source_qualification: GymToraxNativeSourceQualificationReceipt,
    source_qualification_carryforward: tuple[
        GymToraxSourceQualificationCarryforwardEntry, ...
    ],
    extraction_manifest: GymToraxBoundedExtractionManifest,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    implementation_source_closure: ImplementationSourceClosure,
    scientific_approval: ObjectIdentity,
    scientific_seed_inputs: tuple[GymToraxProtocolEnvironmentSeedInput, ...] | None = None,
) -> GymToraxFiniteActionRecurrenceProspectiveFreeze:
    "Build the correction without reading any measurement through local law or finite-action recurrence outcome payload."

    seed_inputs = require_protocol_seed_census(scientific_seed_inputs, matched_evaluation=False)
    if matched_evaluation_acquisition_freeze.predecessors != predecessors:
        raise ValueError("Measurement through local law predecessor closure differs")
    if branch_selection.hfr_contract_evidence != predecessors.hfr_contract_evidence:
        raise ValueError("branch selection predecessor evidence differs")
    if (
        hashlib.sha256(source_campaign_roster_bytes).hexdigest()
        != predecessors.source_roster.object_fingerprint
    ):
        raise ValueError("authenticated Gym-TORAX source roster bytes differ")
    if (
        hashlib.sha256(prospective_protocol_bytes).hexdigest()
        != predecessors.protocol.object_fingerprint
    ):
        raise ValueError("authenticated Gym-TORAX prospective protocol bytes differ")
    if (
        branch_selection.disposition
        is not GymToraxSourceAssessmentBranchSelectionDisposition.FINITE_ACTION_RECURRENCE_SELECTED
        or branch_selection.selected_branch_id != FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_BRANCH_ID
        or branch_selection.parent_issue_started
        or branch_selection.controlling_reason_code is None
    ):
        raise ValueError("Finite-action recurrence prospective correction requires the sealed finite-action recurrence selection")
    if (
        source_qualification.disposition
        is not GymToraxNativeSourceQualificationDisposition.QUALIFIED
    ):
        raise ValueError(
            "Finite-action recurrence prospective correction requires the qualified native source"
        )
    if (
        source_qualification.extraction_manifest != ObjectIdentity.from_record(extraction_manifest.manifest_id, extraction_manifest)
        or source_qualification.field_metadata_manifest != ObjectIdentity.from_record(field_metadata_manifest.manifest_id, field_metadata_manifest)
        or matched_evaluation_acquisition_freeze.source_qualification != ObjectIdentity.from_record(source_qualification.receipt_id, source_qualification)
    ):
        raise ValueError("recurrence qualification must bind the exact current target source manifests and selected parent")
    decode_canonical_bytes(source_campaign_roster_bytes, GymToraxImportedProspectiveTopology, maximum_bytes=8 * 1024**2)
    bridge = FiniteActionAlternateBranchSelectionBridge(
        receipt_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_SELECTION_BRIDGE_ID,
        source_selection=ObjectIdentity.from_record(
            branch_selection.receipt_id, branch_selection
        ),
        g2_operator_feasibility=branch_selection.adjudication,
        selected_branch_id=FINITE_ACTION_HISTORICAL_FIDELITY_RECURRENCE_BRANCH_ID,
        controlling_reason_code=branch_selection.controlling_reason_code,
        hfr_contract_evidence=branch_selection.hfr_contract_evidence,
        parent_issue_started=False,
        measurement_through_law_qualification_outcome_used=False,
        issues_parent=False,
        grants_authority=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )
    protocol_document = json.loads(prospective_protocol_bytes)
    protocol = _unwrap(protocol_document)
    original_occurrence, original_recurrence, occurrence, recurrence = (
        _corrected_templates(
            protocol=protocol,
            bridge=bridge,
        )
    )

    roster = _unwrap(
        json.loads(source_campaign_roster_bytes),
        schema=predecessors.source_roster.object_schema,
    )
    raw_cells = roster.get("prospective_cells")
    if not isinstance(raw_cells, list) or len(raw_cells) != 28:
        raise ValueError("Gym-TORAX source roster changes its 28 prospective cells")
    source_rows = {_unwrap(value)["cell_id"]: _unwrap(value) for value in raw_cells}
    if len(source_rows) != 28:
        raise ValueError("Gym-TORAX prospective source cell IDs are not unique")

    units: list[GymToraxFiniteActionRecurrencePreparedUnit] = []

    def add_unit(
        source: dict[str, Any],
        *,
        unit_id: str,
        role: str,
        stratum_id: str | None,
        support_id: str | None,
        scientific_primary: bool,
    ) -> GymToraxFiniteActionRecurrencePreparedUnit:
        scientific_role = (
            "recurrence-main" if scientific_primary
            else "hold-calibration" if role == "HOLD_CALIBRATION"
            else "recurrence-route"
        )
        scientific_seed = require_protocol_root_seed(seed_inputs, unit_id, scientific_role)
        raw_fingerprint = source.get("preparation_fingerprint")
        if not isinstance(raw_fingerprint, str):
            raise ValueError("Finite-action recurrence source cell lacks preparation fingerprint")
        unit = GymToraxFiniteActionRecurrencePreparedUnit(
            unit_id=unit_id,
            source_cell_id=cast(str, source["cell_id"]),
            source_preparation_fingerprint=raw_fingerprint,
            prospective_role=role,
            source_anchor_slot_id=cast(str | None, source.get("source_anchor_slot_id")),
            stratum_id=stratum_id,
            local_support_id=support_id,
            preparation=_preparation(source=source, unit_id=unit_id, scientific_seed_input=scientific_seed),
            scientific_seed_input=scientific_seed,
            shared_environment_seed=True,
            scientific_primary=scientific_primary,
        )
        units.append(unit)
        return unit

    original_units = {
        value.preparation_coordinate_id: value for value in recurrence.units
    }
    main_unit_by_original: dict[str, GymToraxFiniteActionRecurrencePreparedUnit] = {}
    for coordinate_id, spec in sorted(original_units.items()):
        source = source_rows[coordinate_id]
        unit = add_unit(
            source,
            unit_id=f"unit.tokamak-control.finite-action-prospective-unit.{coordinate_id.removeprefix('cell.tokamak-control.')}",
            role=cast(str, source["prospective_role"]),
            stratum_id=spec.stratum_id,
            support_id=spec.local_support_id,
            scientific_primary=True,
        )
        main_unit_by_original[spec.unit_id] = unit

    route_sources = {
        cast(str, row["prospective_role"]): row
        for row in source_rows.values()
        if str(row.get("prospective_role", "")).startswith("ROUTE_")
    }
    route_units = {
        role: add_unit(
            source,
            unit_id=f"unit.tokamak-control.finite-action-route-unit.{role.lower().replace('_', '-')}",
            role=role,
            stratum_id=None,
            support_id=(
                'local-support.tokamak-control.joint-depth-middle'
                if role == "ROUTE_ACTIVE_CANARY"
                else None
            ),
            scientific_primary=False,
        )
        for role, source in sorted(route_sources.items())
    }

    hold_sources = tuple(
        row
        for row in source_rows.values()
        if row.get("prospective_role") in {"HOLD_CONTROL", "HOLD_RESERVE"}
    )
    if len(hold_sources) != 5:
        raise ValueError("Finite-action recurrence source roster changes its five HOLD anchors")
    calibration_units = {
        cast(str, source["source_anchor_slot_id"]): add_unit(
            source,
            unit_id=(
                'unit.tokamak-control.native-hold-calibration.'
                f"{cast(str, source['source_anchor_slot_id'])}"
            ),
            role="HOLD_CALIBRATION",
            stratum_id=None,
            support_id=None,
            scientific_primary=False,
        )
        for source in sorted(
            hold_sources, key=lambda value: cast(str, value["source_anchor_slot_id"])
        )
    }

    members = gym_torax_numerical_members()
    qualification_identity = ObjectIdentity.from_record(
        source_qualification.receipt_id, source_qualification
    )
    extraction_identity = ObjectIdentity.from_record(
        extraction_manifest.manifest_id, extraction_manifest
    )
    metadata_identity = ObjectIdentity.from_record(
        field_metadata_manifest.manifest_id, field_metadata_manifest
    )
    cells: list[GymToraxFiniteActionRecurrenceCell] = []

    def add_cell(
        *,
        episode_id: str,
        execution_id: str,
        stage: GymToraxFiniteActionRecurrenceCellStage,
        unit: GymToraxFiniteActionRecurrencePreparedUnit,
        member: GymToraxNumericalMember,
        route_role: str,
        word_id: str,
    ) -> None:
        stem = episode_id.removeprefix("episode.")
        request = GymToraxFieldMetadataEpisodeRequest(
            request_id=f"request.{stem}",
            request_role=GymToraxRequestRole.SCIENTIFIC_EPISODE,
            source_qualification=qualification_identity,
            extraction_manifest=extraction_identity,
            field_metadata_manifest=metadata_identity,
            preparation=unit.preparation,
            numerical_member=member,
            schedule=build_gym_torax_native_schedule(build_gym_torax_action_word(word_id)),
            maximum_output_bytes=_TASK_BUDGET.output_bytes,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        )
        cells.append(
            GymToraxFiniteActionRecurrenceCell(
                cell_id=f"cell.{stem}",
                episode_id=episode_id,
                execution_id=execution_id,
                stage=stage,
                unit_id=unit.unit_id,
                model_member_id=member.member_id,
                route_role=route_role.lower().replace("_", "-"),
                action_word_id=word_id,
                request_id=request.request_id,
                request_sha256=request.fingerprint(),
            )
        )

    for route in recurrence.routes:
        unit = main_unit_by_original[route.unit_id]
        member = next(
            value for value in members if value.member_id == route.model_member_id
        )
        add_cell(
            episode_id=route.episode_id,
            execution_id=route.execution_id,
            stage=GymToraxFiniteActionRecurrenceCellStage.RECURRENCE,
            unit=unit,
            member=member,
            route_role=route.role.value,
            word_id=route.action_word_id,
        )

    route_products = (
        ("ROUTE_ACTIVE_CANARY", "ACTIVE", _ACTIVE_WORD_ID),
        ("ROUTE_ACTIVE_CANARY", "MATCHED_HOLD", _HOLD_WORD_ID),
        ("ROUTE_HOLD_CANARY", "HOLD_CONTROL", _HOLD_WORD_ID),
        ("ROUTE_HOLD_REPEAT", "HOLD_QUALIFICATION_REPEAT", _HOLD_WORD_ID),
    )
    for member in members:
        member_slug = member.member_id.rsplit(".", 1)[-1]
        for unit_role, route_role, word_id in route_products:
            stem = f'tokamak-control.finite-action-route-unit.{unit_role.lower()}.{member_slug}.{route_role.lower()}'
            add_cell(
                episode_id=f"episode.{stem}",
                execution_id=f"execution.{stem}",
                stage=GymToraxFiniteActionRecurrenceCellStage.ROUTE_QUALIFICATION,
                unit=route_units[unit_role],
                member=member,
                route_role=route_role,
                word_id=word_id,
            )

    for slot, unit in sorted(calibration_units.items()):
        for member in members:
            member_slug = member.member_id.rsplit(".", 1)[-1]
            stem = f'tokamak-control.native-hold-calibration.{slot}.{member_slug}'
            add_cell(
                episode_id=f"episode.{stem}",
                execution_id=f"execution.{stem}",
                stage=GymToraxFiniteActionRecurrenceCellStage.HOLD_CALIBRATION,
                unit=unit,
                member=member,
                route_role="HOLD_CALIBRATION",
                word_id=_HOLD_WORD_ID,
            )

    if set(seed_inputs) != {unit.unit_id for unit in units}:
        raise ValueError("recurrence seed census differs from its complete physical root roster")
    return GymToraxFiniteActionRecurrenceProspectiveFreeze(
        predecessors=predecessors,
        freeze_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_FREEZE_ID,
        run_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_RUN_ID,
        relative_root=GYM_TORAX_FINITE_ACTION_RECURRENCE_RELATIVE_ROOT,
        source_campaign_roster=predecessors.source_roster,
        source_prospective_protocol=ObjectIdentity(
            object_id='protocol.tokamak-control.direct-science',
            object_schema=cast(str, protocol_document["schema"]),
            object_version=cast(str, protocol_document["version"]),
            object_fingerprint=predecessors.protocol.object_fingerprint,
        ),
        original_occurrence_templates=original_occurrence,
        original_recurrence_template=original_recurrence,
        repair_rule_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_REPAIR_RULE_ID,
        selected_branch=ObjectIdentity.from_record(
            branch_selection.receipt_id, branch_selection
        ),
        selection_bridge=bridge,
        matched_evaluation_acquisition_freeze=ObjectIdentity.from_record(
            matched_evaluation_acquisition_freeze.freeze_id,
            matched_evaluation_acquisition_freeze,
        ),
        matched_evaluation_science_freeze=ObjectIdentity.from_record(
            matched_evaluation_science_freeze.freeze_id,
            matched_evaluation_science_freeze,
        ),
        corrected_occurrence_templates=occurrence,
        corrected_recurrence_template=recurrence,
        source_qualification=qualification_identity,
        source_qualification_carryforward=source_qualification_carryforward,
        extraction_manifest=extraction_identity,
        field_metadata_manifest=metadata_identity,
        implementation_source_closure=implementation_source_closure,
        scientific_approval=scientific_approval,
        units=tuple(sorted(units, key=lambda value: value.unit_id)),
        numerical_members=members,
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        task_resource_budget=_TASK_BUDGET,
        campaign_resource_budget=_CAMPAIGN_BUDGET,
        maximum_concurrency=1,
        required_operation_authority_ids=GYM_TORAX_FINITE_ACTION_RECURRENCE_AUTHORITY_IDS,
        original_route_seed_count=80,
        corrected_recurrence_seed_count=22,
        calibration_unit_count=5,
        route_qualification_episode_count=8,
        recurrence_episode_count=80,
        frozen_before_matched_evaluation_reveal=True,
        matched_evaluation_outcomes_read=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


__all__ = [
    'GymToraxFiniteActionRecurrenceCellStage',
    'GymToraxFiniteActionRecurrenceCell',
    'GymToraxFiniteActionRecurrencePreparedUnit',
    'GymToraxFiniteActionRecurrenceProspectiveFreeze',
    'GYM_TORAX_FINITE_ACTION_RECURRENCE_AUTHORITY_IDS',
    'GYM_TORAX_FINITE_ACTION_RECURRENCE_CALIBRATION_RECEIPT_ID',
    'GYM_TORAX_FINITE_ACTION_RECURRENCE_FREEZE_ID',
    'GYM_TORAX_FINITE_ACTION_RECURRENCE_ROUTE_QUALIFICATION_RECEIPT_ID',
    'build_gym_torax_finite_action_recurrence_prospective_freeze',
    'materialize_gym_torax_finite_action_recurrence_request',
]

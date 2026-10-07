"Exact selected Gym-TORAX finite-action recurrence measurement-through-law-qualification freeze and roster repair.\n\nThe immutable donor roster is an authenticated design/topology parent, but its\neight nested views assign eight different environment seeds to one claimed\nphysical preparation unit.  That repeats the preparation-identity defect found\nin the source-assessment diagnostic corpus.  This new selected parent leaves the terminal predecessor\nacquisition immutable and freezes the same 18 primary quartets and preparation\ncoordinates under new identities and seeds, with one preparation record and\none seed shared by all eight member/action views of each physical unit.\n\nThe parser below is intentionally not a general roster decoder.  It accepts\nonly the one byte-authenticated prospective Gym-TORAX roster and extracts only the\nclosed 18-primary-quartet topology needed by the selected parent.\n"

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
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
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind

from .action_word import GYM_TORAX_ACTION_WORD_IDS, build_gym_torax_action_word, build_gym_torax_native_schedule
from .diagnostic_contracts import GymToraxNumericalMember, GymToraxPreparation, gym_torax_numerical_members
from .field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxRequestRole
from .extraction_manifest import GymToraxBoundedExtractionManifest
from .field_metadata import GymToraxFieldMetadataManifest
from .source_assessment_adjudication import GymToraxSourceAssessmentBranchSelectionDisposition, GymToraxSourceAssessmentBranchSelectionReceipt
from .source_assessment_protocol import GYM_TORAX_FINITE_ACTION_RECURRENCE_BRANCH_ID
from .retained_inputs import GymToraxImportedProspectiveTopology, GymToraxPredecessorInputs
from .protocol_scientific_inputs import (
    GymToraxProtocolEnvironmentSeedInput,
    require_protocol_root_seed,
    require_protocol_seed_census,
)
from .source_assessment_protocol import GymToraxNativeMetadataCanaryFreeze
from .source_qualification import GymToraxNativeSourceQualificationDisposition, GymToraxNativeSourceQualificationReceipt


GYM_TORAX_MATCHED_EVALUATION_RUN_ID = 'run.tokamak-control.matched-evaluation'
GYM_TORAX_MATCHED_EVALUATION_FREEZE_ID = 'freeze.tokamak-control.matched-evaluation'
GYM_TORAX_MATCHED_EVALUATION_RELATIVE_ROOT = (
    'tokamak-control/publication/runs/run.tokamak-control.matched-evaluation'
)
GYM_TORAX_MATCHED_EVALUATION_ROSTER_REPAIR_RULE = (
    'repair.tokamak-control.shared-preparation-identity-and-seed-per-physical-unit'
)
GYM_TORAX_MATCHED_EVALUATION_AUTHORITY_IDS = (
    'authority.tokamak-control.matched-evaluation.custody-publication',
    'authority.tokamak-control.matched-evaluation.experiment-execution',
    'authority.tokamak-control.matched-evaluation.outcome-reveal',
    'authority.tokamak-control.matched-evaluation.source-acquisition',
)

_PREPARATION_VALUE_IDS = {
    'tokamak-control.preparation.bootstrap-multiplier',
    'tokamak-control.preparation.initial-density-nbar',
    'tokamak-control.preparation.initial-temperature-scale',
    'tokamak-control.preparation.inner-transport-scale',
}
_CELL_KINDS = {"b", "bt", "c", "t"}
_TIERS = {"TIER_01", "TIER_02", "TIER_03"}
_SOURCE_PACKAGE_PATH = 'src/empirical_lawhood/adapters/simulators/gym_torax_native'
GYM_TORAX_MATCHED_EVALUATION_SOURCE_QUALIFICATION_CARRYFORWARD_PATHS = tuple(
    sorted(
        (
            f"{_SOURCE_PACKAGE_PATH}/action_word.py",
            f'{_SOURCE_PACKAGE_PATH}/diagnostic_contracts.py',
            f'{_SOURCE_PACKAGE_PATH}/field_metadata_contracts.py',
            f"{_SOURCE_PACKAGE_PATH}/field_metadata.py",
            f"{_SOURCE_PACKAGE_PATH}/metadata_barrier.py",
            f"{_SOURCE_PACKAGE_PATH}/runtime.py",
            f"{_SOURCE_PACKAGE_PATH}/source_qualification.py",
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
    wall_time_seconds=577 * 12 * 60 * 60,
    source_scan_bytes=0,
    output_bytes=44 * 1024**3,
)


@dataclass(frozen=True, slots=True)
class GymToraxSourceQualificationCarryforwardEntry(CanonicalRecord):
    "Exact unchanged source-facing file across qualification and measurement issue."

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-source-qualification-carryforward-entry'
    )

    relative_path: str
    qualified_file_sha256: str
    selected_parent_file_sha256: str

    def __post_init__(self) -> None:
        validate_relative_locator(self.relative_path)
        validate_sha256(self.qualified_file_sha256, field_name="qualified_file_sha256")
        validate_sha256(
            self.selected_parent_file_sha256,
            field_name="selected_parent_file_sha256",
        )
        if self.qualified_file_sha256 != self.selected_parent_file_sha256:
            raise ValueError(
                "source-qualified runtime file changed before response qualification issue"
            )


@dataclass(frozen=True, slots=True)
class GymToraxSelectedParentSelectedUnit(CanonicalRecord):
    """One corrected physical unit derived from an exact source-roster cell."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-selected-parent-selected-unit'

    unit_id: str
    source_quartet_id: str
    source_cell_id: str
    tier_id: str
    cell_kind: str
    preparation: GymToraxPreparation
    source_environment_seeds: tuple[int, ...]
    scientific_seed_input: GymToraxProtocolEnvironmentSeedInput
    shared_preparation_identity: bool
    shared_environment_seed: bool

    def __post_init__(self) -> None:
        for name in ("unit_id", "source_quartet_id", "source_cell_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        if self.cell_kind not in _CELL_KINDS or self.tier_id not in _TIERS:
            raise ValueError("selected response qualification unit has an unknown cell kind or tier")
        if not isinstance(self.scientific_seed_input, GymToraxProtocolEnvironmentSeedInput):
            raise ValueError("matched preparation requires its typed scientific seed input")
        self.scientific_seed_input.require_root(self.unit_id, "matched-evaluation")
        if self.preparation.environment_seed != self.scientific_seed_input.environment_seed:
            raise ValueError("matched preparation changes its declared scientific seed")
        if self.preparation.physical_independent_unit_id != self.unit_id:
            raise ValueError("selected response qualification unit changes its preparation identity")
        if (
            len(self.source_environment_seeds) != 8
            or len(set(self.source_environment_seeds)) != 8
            or any(
                value <= 0 or value >= 2**63 for value in self.source_environment_seeds
            )
        ):
            raise ValueError(
                "source response qualification unit does not expose eight exact source seeds"
            )
        if not self.shared_preparation_identity or not self.shared_environment_seed:
            raise ValueError(
                "selected response qualification nested views must share preparation and seed"
            )


@dataclass(frozen=True, slots=True)
class GymToraxSelectedParentCell(CanonicalRecord):
    """One frozen member/action view whose full request is deterministically rebuilt."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-selected-parent-cell'

    cell_id: str
    coordinate_id: str
    episode_id: str
    physical_unit_instance_id: str
    numerical_member_id: str
    action_word_id: str
    request_id: str
    request_sha256: str

    def __post_init__(self) -> None:
        for name in (
            "cell_id",
            "coordinate_id",
            "episode_id",
            "physical_unit_instance_id",
            "numerical_member_id",
            "action_word_id",
            "request_id",
        ):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_sha256(self.request_sha256, field_name="request_sha256")
        stem = self.cell_id.removeprefix("cell.")
        if (
            self.cell_id == stem
            or self.coordinate_id != f"coordinate.{stem}"
            or self.episode_id != f"episode.{stem}"
            or self.request_id != f"request.{stem}"
        ):
            raise ValueError("selected response qualification row identities diverged")
        if self.action_word_id not in GYM_TORAX_ACTION_WORD_IDS:
            raise ValueError("selected response qualification row names another action word")


@dataclass(frozen=True, slots=True)
class GymToraxSelectedParentFreeze(CanonicalRecord):
    predecessors: GymToraxPredecessorInputs
    """Exact 576-view selected HFR parent freeze after revealed Q selection."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-selected-parent-freeze'
    VERSION: ClassVar[str] = '1.0.0'

    freeze_id: str
    run_id: str
    relative_root: str
    selected_branch: ObjectIdentity
    prospective_protocol_parent: ObjectIdentity
    source_campaign_roster: ObjectIdentity
    roster_repair_rule_id: str
    source_canary_freeze: ObjectIdentity
    source_qualification: ObjectIdentity
    qualified_implementation_source_closure: ImplementationSourceClosure
    source_qualification_carryforward: tuple[
        GymToraxSourceQualificationCarryforwardEntry, ...
    ]
    extraction_manifest: ObjectIdentity
    field_metadata_manifest: ObjectIdentity
    implementation_source_closure: ImplementationSourceClosure
    scientific_approval: ObjectIdentity
    selected_quartet_ids: tuple[str, ...]
    units: tuple[GymToraxSelectedParentSelectedUnit, ...]
    numerical_members: tuple[GymToraxNumericalMember, ...]
    cells: tuple[GymToraxSelectedParentCell, ...]
    task_resource_budget: ResourceBudget
    campaign_resource_budget: ResourceBudget
    maximum_concurrency: int
    required_operation_authority_ids: tuple[str, ...]
    frozen_before_source_reset: bool
    source_reset_count_at_freeze: int
    matched_evaluation_outcomes_read: bool
    source_roster_seeds_reused: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        if (
            self.freeze_id != GYM_TORAX_MATCHED_EVALUATION_FREEZE_ID
            or self.run_id != GYM_TORAX_MATCHED_EVALUATION_RUN_ID
        ):
            raise ValueError("selected response qualification freeze identity changed")
        validate_relative_locator(self.relative_root)
        if self.relative_root != GYM_TORAX_MATCHED_EVALUATION_RELATIVE_ROOT:
            raise ValueError("selected response qualification storage namespace changed")
        if self.selected_branch.object_schema != GymToraxSourceAssessmentBranchSelectionReceipt.SCHEMA:
            raise ValueError("selected response qualification freeze lacks the exact source assessment selector")
        if self.prospective_protocol_parent != self.predecessors.protocol:
            raise ValueError(
                "selected response qualification freeze names another prospective protocol"
            )
        if self.source_campaign_roster != self.predecessors.source_roster:
            raise ValueError("selected response qualification freeze names another source roster")
        if self.roster_repair_rule_id != GYM_TORAX_MATCHED_EVALUATION_ROSTER_REPAIR_RULE:
            raise ValueError("selected response qualification preparation repair rule changed")
        if (
            self.source_qualification.object_schema
            != GymToraxNativeSourceQualificationReceipt.SCHEMA
        ):
            raise ValueError("selected response qualification freeze lacks source qualification")
        if (
            self.source_canary_freeze.object_schema
            != GymToraxNativeMetadataCanaryFreeze.SCHEMA
        ):
            raise ValueError("selected response qualification freeze lacks the qualified canary freeze")
        if (
            self.qualified_implementation_source_closure.kind
            is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.qualified_implementation_source_closure.clean_worktree
        ):
            raise ValueError(
                "selected response qualification freeze lacks the canary-qualified source closure"
            )
        require_sorted_unique_ids(
            self.source_qualification_carryforward,
            attribute="relative_path",
            field_name="source_qualification_carryforward",
        )
        if tuple(
            value.relative_path for value in self.source_qualification_carryforward
        ) != (GYM_TORAX_MATCHED_EVALUATION_SOURCE_QUALIFICATION_CARRYFORWARD_PATHS):
            raise ValueError(
                "selected response qualification source-qualification carry-forward is incomplete"
            )
        if (
            self.extraction_manifest.object_schema
            != GymToraxBoundedExtractionManifest.SCHEMA
        ):
            raise ValueError("selected response qualification freeze lacks the bounded extraction")
        if (
            self.field_metadata_manifest.object_schema
            != GymToraxFieldMetadataManifest.SCHEMA
        ):
            raise ValueError("selected response qualification freeze lacks the metadata contract")
        if (
            self.implementation_source_closure.kind
            is not SourceClosureKind.CLEAN_GIT_COMMIT
            or not self.implementation_source_closure.clean_worktree
        ):
            raise ValueError("selected response qualification freeze requires a clean committed source")
        require_sorted_unique_strings(
            self.selected_quartet_ids, field_name="selected_quartet_ids"
        )
        require_sorted_unique_ids(self.units, attribute="unit_id", field_name="units")
        require_sorted_unique_ids(
            self.numerical_members,
            attribute="member_id",
            field_name="numerical_members",
        )
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        if (
            len(self.selected_quartet_ids) != 18
            or len(self.units) != 72
            or len(self.cells) != 576
        ):
            raise ValueError("selected response qualification freeze changes the 18/72/576 design")
        unit_counts = Counter(value.physical_unit_instance_id for value in self.cells)
        if set(unit_counts) != {value.unit_id for value in self.units} or set(
            unit_counts.values()
        ) != {8}:
            raise ValueError(
                "selected response qualification freeze breaks eight nested views per unit"
            )
        member_ids = {value.member_id for value in self.numerical_members}
        if member_ids != {'member.tokamak-control.primary', 'member.tokamak-control.refined'}:
            raise ValueError("selected response qualification numerical members changed")
        for unit_id in unit_counts:
            nested = tuple(
                value
                for value in self.cells
                if value.physical_unit_instance_id == unit_id
            )
            if {value.numerical_member_id for value in nested} != member_ids or {
                value.action_word_id for value in nested
            } != set(GYM_TORAX_ACTION_WORD_IDS):
                raise ValueError("selected response qualification unit lacks the member/action product")
        if (
            self.task_resource_budget != _TASK_BUDGET
            or self.campaign_resource_budget != _CAMPAIGN_BUDGET
        ):
            raise ValueError("selected response qualification resource envelope changed")
        if self.maximum_concurrency != 1:
            raise ValueError("selected response qualification execution must remain sequential")
        if self.required_operation_authority_ids != GYM_TORAX_MATCHED_EVALUATION_AUTHORITY_IDS:
            raise ValueError("selected response qualification authority roster changed")
        if (
            not self.frozen_before_source_reset
            or self.source_reset_count_at_freeze != 0
            or self.matched_evaluation_outcomes_read
            or self.source_roster_seeds_reused
        ):
            raise ValueError(
                "selected response qualification freeze is not prospective or keeps bad seeds"
            )
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("selected response qualification freeze changes its evidence boundary")


def _unwrap(value: object, *, schema: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"schema", "value", "version"}:
        raise ValueError(
            "source campaign roster contains a malformed canonical envelope"
        )
    if (
        value["schema"] != schema
        or value["version"] != "1.0.0"
        or not isinstance(value["value"], dict)
    ):
        raise ValueError("source campaign roster canonical type differs")
    return cast(dict[str, Any], value["value"])


def _preparation_values(cell: dict[str, Any]) -> dict[str, Decimal]:
    raw = cell.get("preparation_values")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError("source response qualification cell lacks four preparation values")
    values: dict[str, Decimal] = {}
    for item in raw:
        row = _unwrap(item, schema='empirical-lawhood/kernel/named-decimal')
        decimal_envelope = row.get("value")
        if row.get("unit") != "1" or not isinstance(decimal_envelope, dict):
            raise ValueError("source response qualification preparation value changed unit/type")
        text = decimal_envelope.get("decimal")
        value_id = row.get("value_id")
        if not isinstance(value_id, str) or not isinstance(text, str):
            raise ValueError("source response qualification preparation value is malformed")
        values[value_id] = Decimal(text)
    if set(values) != _PREPARATION_VALUE_IDS:
        raise ValueError("source response qualification preparation chart changed")
    return values


def materialize_gym_torax_response_experiment_request(
    freeze: GymToraxSelectedParentFreeze,
    cell: GymToraxSelectedParentCell,
) -> GymToraxFieldMetadataEpisodeRequest:
    unit = next(
        (
            value
            for value in freeze.units
            if value.unit_id == cell.physical_unit_instance_id
        ),
        None,
    )
    member = next(
        (
            value
            for value in freeze.numerical_members
            if value.member_id == cell.numerical_member_id
        ),
        None,
    )
    if unit is None or member is None:
        raise ValueError("selected response qualification cell cannot resolve its frozen unit/member")
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
        raise ValueError("selected response qualification request differs from its frozen bytes")
    return request


def build_gym_torax_response_experiment_freeze(
    *,
    predecessors: GymToraxPredecessorInputs,
    source_campaign_roster_bytes: bytes,
    branch_selection: GymToraxSourceAssessmentBranchSelectionReceipt,
    source_canary_freeze: GymToraxNativeMetadataCanaryFreeze,
    source_qualification: GymToraxNativeSourceQualificationReceipt,
    source_qualification_carryforward: tuple[
        GymToraxSourceQualificationCarryforwardEntry, ...
    ],
    extraction_manifest: GymToraxBoundedExtractionManifest,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    implementation_source_closure: ImplementationSourceClosure,
    scientific_approval: ObjectIdentity,
    scientific_seed_inputs: tuple[GymToraxProtocolEnvironmentSeedInput, ...] | None = None,
) -> GymToraxSelectedParentFreeze:
    """Build the corrected 576-view freeze without reading scientific values."""

    seed_inputs = require_protocol_seed_census(scientific_seed_inputs, matched_evaluation=True)
    if source_canary_freeze.predecessors != predecessors:
        raise ValueError("source canary predecessor closure differs")
    if branch_selection.hfr_contract_evidence != predecessors.hfr_contract_evidence:
        raise ValueError("branch selection predecessor evidence differs")
    if hashlib.sha256(source_campaign_roster_bytes).hexdigest() != (
        predecessors.source_roster.object_fingerprint
    ):
        raise ValueError("authenticated source campaign roster bytes differ")
    decode_canonical_bytes(source_campaign_roster_bytes, GymToraxImportedProspectiveTopology, maximum_bytes=8 * 1024**2)
    document = json.loads(source_campaign_roster_bytes)
    roster = _unwrap(document, schema=predecessors.source_roster.object_schema)
    if (
        roster.get("roster_id") != predecessors.source_roster.object_id
        or roster.get("frozen") is not True
        or roster.get("outcome_access") != OutcomeAccess.OUTCOME_BLIND.value
    ):
        raise ValueError("source campaign roster identity/freeze boundary changed")
    if (
        branch_selection.disposition
        is not GymToraxSourceAssessmentBranchSelectionDisposition.FINITE_ACTION_RECURRENCE_SELECTED
        or branch_selection.selected_branch_id != GYM_TORAX_FINITE_ACTION_RECURRENCE_BRANCH_ID
        or branch_selection.selected_parent_episode_count != 576
        or branch_selection.parent_issue_started
    ):
        raise ValueError(
            "selected response qualification freeze requires the sealed preissue finite-action recurrence selection"
        )
    if (
        source_qualification.disposition
        is not GymToraxNativeSourceQualificationDisposition.QUALIFIED
    ):
        raise ValueError("selected response qualification freeze requires a qualified source")
    if (
        source_qualification.extraction_manifest
        != source_canary_freeze.extraction_manifest
        or source_qualification.field_metadata_manifest
        != source_canary_freeze.field_metadata_manifest
        or source_canary_freeze.extraction_manifest
        != ObjectIdentity.from_record(extraction_manifest.manifest_id, extraction_manifest)
        or source_canary_freeze.field_metadata_manifest
        != ObjectIdentity.from_record(
            field_metadata_manifest.manifest_id, field_metadata_manifest
        )
    ):
        raise ValueError("selected response qualification source qualification/canary lineage differs")

    raw_quartets = roster.get("evaluation_primary_quartets")
    raw_coordinates = roster.get("episode_coordinates")
    if not isinstance(raw_quartets, list) or not isinstance(raw_coordinates, list):
        raise ValueError("source campaign roster lacks primary quartets/coordinates")
    coordinate_rows = tuple(
        _unwrap(
            value, schema='empirical-lawhood/composition/tokamak-control-replication/gym-torax-episode-coordinate'
        )
        for value in raw_coordinates
    )
    primary_coordinates = tuple(
        value for value in coordinate_rows if value.get("stage") == 'MATCHED_EVALUATION_PRIMARY'
    )
    if len(primary_coordinates) != 576:
        raise ValueError("source campaign roster changes the 576 primary coordinates")
    coordinates_by_source_cell: dict[str, tuple[dict[str, Any], ...]] = {}
    for source_cell_id in sorted(
        {cast(str, value.get("cell_id")) for value in primary_coordinates}
    ):
        coordinates_by_source_cell[source_cell_id] = tuple(
            value
            for value in primary_coordinates
            if value.get("cell_id") == source_cell_id
        )

    units: list[GymToraxSelectedParentSelectedUnit] = []
    selected_quartet_ids: list[str] = []
    unit_source_stem: dict[str, str] = {}
    for raw_quartet in raw_quartets:
        quartet = _unwrap(
            raw_quartet,
            schema='empirical-lawhood/composition/tokamak-control-replication/gym-torax-matched-quartet',
        )
        quartet_id = quartet.get("quartet_id")
        if (
            not isinstance(quartet_id, str)
            or quartet.get("role") != "EVALUATION_PRIMARY"
            or quartet.get("tier") not in _TIERS
        ):
            raise ValueError("source campaign primary quartet changed role/tier")
        selected_quartet_ids.append(quartet_id)
        raw_cells = quartet.get("cells")
        if not isinstance(raw_cells, list) or len(raw_cells) != 4:
            raise ValueError("source campaign primary quartet is not whole")
        for raw_cell in raw_cells:
            source_cell = _unwrap(
                raw_cell, schema='empirical-lawhood/composition/tokamak-control-replication/gym-torax-roster-cell'
            )
            raw_source_cell_id = source_cell.get("cell_id")
            source_unit_id = source_cell.get("physical_unit_instance_id")
            cell_kind = str(source_cell.get("cell_kind", "")).lower()
            if (
                not isinstance(raw_source_cell_id, str)
                or not isinstance(source_unit_id, str)
                or source_cell.get("block_id") != quartet_id
                or source_cell.get("tier") != quartet.get("tier")
                or source_cell.get("quartet_role") != "EVALUATION_PRIMARY"
                or source_cell.get("reserve") is not False
                or cell_kind not in _CELL_KINDS
            ):
                raise ValueError("source campaign primary cell topology changed")
            source_cell_id = raw_source_cell_id
            nested = coordinates_by_source_cell.get(source_cell_id, ())
            raw_source_seeds = tuple(value.get("environment_seed") for value in nested)
            if (
                len(nested) != 8
                or {value.get("physical_unit_instance_id") for value in nested}
                != {source_unit_id}
                or {value.get("model_member_id") for value in nested}
                != {'member.tokamak-control.primary', 'member.tokamak-control.refined'}
                or {value.get("action_word_id") for value in nested}
                != set(GYM_TORAX_ACTION_WORD_IDS)
                or any(value.get("reserve") is not False for value in nested)
                or any(
                    not isinstance(value, int) or isinstance(value, bool)
                    for value in raw_source_seeds
                )
            ):
                raise ValueError(
                    "source campaign primary cell lacks its exact eight views"
                )
            source_seeds = tuple(sorted(cast(tuple[int, ...], raw_source_seeds)))
            values = _preparation_values(source_cell)
            source_stem = source_cell_id.removeprefix('cell.tokamak-control.')
            unit_id = f'unit.tokamak-control.matched-evaluation.{source_stem}'
            scientific_seed = require_protocol_root_seed(seed_inputs, unit_id, "matched-evaluation")
            preparation = GymToraxPreparation(
                preparation_id=f'preparation.tokamak-control.matched-evaluation.{source_stem}',
                physical_independent_unit_id=unit_id,
                environment_seed=scientific_seed.environment_seed,
                initial_temperature_scale=values[
                    'tokamak-control.preparation.initial-temperature-scale'
                ],
                initial_density_nbar=values['tokamak-control.preparation.initial-density-nbar'],
                bootstrap_multiplier=values['tokamak-control.preparation.bootstrap-multiplier'],
                inner_transport_scale=values['tokamak-control.preparation.inner-transport-scale'],
            )
            units.append(
                GymToraxSelectedParentSelectedUnit(
                    unit_id=unit_id,
                    source_quartet_id=quartet_id,
                    source_cell_id=source_cell_id,
                    tier_id=cast(str, quartet["tier"]),
                    cell_kind=cell_kind,
                    preparation=preparation,
                    source_environment_seeds=source_seeds,
                    scientific_seed_input=scientific_seed,
                    shared_preparation_identity=True,
                    shared_environment_seed=True,
                )
            )
            unit_source_stem[unit_id] = source_stem

    units_tuple = tuple(sorted(units, key=lambda value: value.unit_id))
    members = gym_torax_numerical_members()
    extraction_identity = ObjectIdentity.from_record(
        extraction_manifest.manifest_id, extraction_manifest
    )
    metadata_identity = ObjectIdentity.from_record(
        field_metadata_manifest.manifest_id, field_metadata_manifest
    )
    qualification_identity = ObjectIdentity.from_record(
        source_qualification.receipt_id, source_qualification
    )
    cells: list[GymToraxSelectedParentCell] = []
    for unit in units_tuple:
        source_stem = unit_source_stem[unit.unit_id]
        for member in members:
            member_slug = member.member_id.rsplit(".", maxsplit=1)[-1]
            for word_id in GYM_TORAX_ACTION_WORD_IDS:
                word_slug = word_id.rsplit(".", maxsplit=1)[-1]
                stem = f'tokamak-control.matched-evaluation.{source_stem}.{member_slug}.{word_slug}'
                request = GymToraxFieldMetadataEpisodeRequest(
                    request_id=f"request.{stem}",
                    request_role=GymToraxRequestRole.SCIENTIFIC_EPISODE,
                    source_qualification=qualification_identity,
                    extraction_manifest=extraction_identity,
                    field_metadata_manifest=metadata_identity,
                    preparation=unit.preparation,
                    numerical_member=member,
                    schedule=build_gym_torax_native_schedule(
                        build_gym_torax_action_word(word_id)
                    ),
                    maximum_output_bytes=_TASK_BUDGET.output_bytes,
                    outcome_access=OutcomeAccess.EVALUATION_SEALED,
                    evidence_ceiling=EvidenceCeiling.MEASUREMENT,
                )
                cells.append(
                    GymToraxSelectedParentCell(
                        cell_id=f"cell.{stem}",
                        coordinate_id=f"coordinate.{stem}",
                        episode_id=f"episode.{stem}",
                        physical_unit_instance_id=unit.unit_id,
                        numerical_member_id=member.member_id,
                        action_word_id=word_id,
                        request_id=request.request_id,
                        request_sha256=request.fingerprint(),
                    )
                )
    if set(seed_inputs) != {unit.unit_id for unit in units}:
        raise ValueError("matched seed census differs from the complete selected physical root roster")
    freeze = GymToraxSelectedParentFreeze(
        predecessors=predecessors,
        freeze_id=GYM_TORAX_MATCHED_EVALUATION_FREEZE_ID,
        run_id=GYM_TORAX_MATCHED_EVALUATION_RUN_ID,
        relative_root=GYM_TORAX_MATCHED_EVALUATION_RELATIVE_ROOT,
        selected_branch=ObjectIdentity.from_record(
            branch_selection.receipt_id, branch_selection
        ),
        prospective_protocol_parent=predecessors.protocol,
        source_campaign_roster=predecessors.source_roster,
        roster_repair_rule_id=GYM_TORAX_MATCHED_EVALUATION_ROSTER_REPAIR_RULE,
        source_canary_freeze=ObjectIdentity.from_record(
            source_canary_freeze.freeze_id,
            source_canary_freeze,
        ),
        source_qualification=qualification_identity,
        qualified_implementation_source_closure=(
            source_canary_freeze.implementation_source_closure
        ),
        source_qualification_carryforward=source_qualification_carryforward,
        extraction_manifest=extraction_identity,
        field_metadata_manifest=metadata_identity,
        implementation_source_closure=implementation_source_closure,
        scientific_approval=scientific_approval,
        selected_quartet_ids=tuple(sorted(selected_quartet_ids)),
        units=units_tuple,
        numerical_members=members,
        cells=tuple(sorted(cells, key=lambda value: value.cell_id)),
        task_resource_budget=_TASK_BUDGET,
        campaign_resource_budget=_CAMPAIGN_BUDGET,
        maximum_concurrency=1,
        required_operation_authority_ids=GYM_TORAX_MATCHED_EVALUATION_AUTHORITY_IDS,
        frozen_before_source_reset=True,
        source_reset_count_at_freeze=0,
        matched_evaluation_outcomes_read=False,
        source_roster_seeds_reused=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )
    for cell in freeze.cells:
        materialize_gym_torax_response_experiment_request(freeze, cell)
    return freeze


__all__ = [
    'GYM_TORAX_MATCHED_EVALUATION_AUTHORITY_IDS',
    'GYM_TORAX_MATCHED_EVALUATION_FREEZE_ID',
    'GYM_TORAX_MATCHED_EVALUATION_RELATIVE_ROOT',
    'GYM_TORAX_MATCHED_EVALUATION_RUN_ID',
    'GYM_TORAX_MATCHED_EVALUATION_SOURCE_QUALIFICATION_CARRYFORWARD_PATHS',
    'GymToraxSelectedParentCell',
    'GymToraxSelectedParentFreeze',
    'GymToraxSelectedParentSelectedUnit',
    'GymToraxSourceQualificationCarryforwardEntry',
    'build_gym_torax_response_experiment_freeze',
    'materialize_gym_torax_response_experiment_request',
]

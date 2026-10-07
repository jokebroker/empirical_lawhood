"Fresh, outcome-blind Gym-TORAX native canary and 96-cell source assessment freezes.\n\nThis is the smallest prospective execution surface recovered from the\nquarantine implementation.  It reuses the already-issued outcome-blind Gym-TORAX\nprotocol only as an immutable semantic parent, creates fresh preparation units\nand seeds, and binds every field-metadata request byte before source reset.\n"

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import ClassVar

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_nonempty,
    validate_relative_locator,
    validate_stable_id,
)
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, SourceClosureKind

from .retained_inputs import GymToraxPredecessorInputs
from .source_assessment_seeds import (
    GymToraxSourceAssessmentSeed,
    gym_torax_metadata_canary_seed,
    gym_torax_source_qualification_seeds,
    require_source_qualification_seed_census,
)

from .action_word import GYM_TORAX_ACTION_WORD_IDS, build_gym_torax_action_word, build_gym_torax_native_schedule
from .diagnostic_contracts import GYM_TORAX_PRIMARY_MEMBER_ID, GymToraxPreparation, gym_torax_numerical_members
from .field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxRequestRole
from .extraction_manifest import GymToraxBoundedExtractionManifest
from .field_metadata import GymToraxFieldMetadataManifest
from .metadata_barrier import GymToraxNativeMetadataCanaryReceipt
from .source_qualification import GymToraxNativeSourceQualificationDisposition, GymToraxNativeSourceQualificationReceipt


GYM_TORAX_NATIVE_CANARY_RUN_ID = 'run.tokamak-control.native-metadata-canary'
GYM_TORAX_FRESH_SOURCE_ASSESSMENT_RUN_ID = 'run.tokamak-control.source-assessment'
GYM_TORAX_PRIMARY_BRANCH_ID = 'TOKAMAK-ORDINARY-CONTROLLED-RESPONSE'
GYM_TORAX_FINITE_ACTION_RECURRENCE_BRANCH_ID = 'TOKAMAK-FINITE-ACTION-RECURRENCE'
GYM_TORAX_NATIVE_CANARY_RELATIVE_ROOT = (
    'tokamak-control/publication/runs/run.tokamak-control.native-metadata-canary'
)
GYM_TORAX_FRESH_SOURCE_ASSESSMENT_RELATIVE_ROOT = (
    'tokamak-control/publication/runs/run.tokamak-control.source-assessment'
)
GYM_TORAX_CANARY_FREEZE_ID = 'freeze.tokamak-control.native-metadata-canary'
GYM_TORAX_FRESH_SOURCE_ASSESSMENT_FREEZE_ID = 'freeze.tokamak-control.source-assessment'
_ALTERNATE_REASON_CODE_PRECEDENCE = (
    "CONTROLLED_IO_OPERATOR_API_UNAVAILABLE",
    "CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE",
    "CONTROLLED_IO_STATE_OR_OBSERVATION_REPRESENTATION_UNAVAILABLE",
    "CONTROLLED_IO_REPRESENTATION_BOUND_EXCEEDED",
    "CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_NOT_SUPPORTED",
    "CONTROLLED_IO_EXCLUDED_NUMERICAL_FEASIBILITY_UNEVALUABLE",
)
_SOURCE_ASSESSMENT_AUTHORITY_IDS = (
    'authority.tokamak-control.source-assessment.custody-publication',
    'authority.tokamak-control.source-assessment.experiment-execution',
    'authority.tokamak-control.source-assessment.outcome-reveal',
    'authority.tokamak-control.source-assessment.source-acquisition',
)
_CANARY_AUTHORITY_IDS = (
    'authority.tokamak-control.native-metadata-canary.custody-publication',
    'authority.tokamak-control.native-metadata-canary.experiment-execution',
    'authority.tokamak-control.native-metadata-canary.source-acquisition',
)
_TASK_BUDGET = ResourceBudget(
    cpu_cores=8,
    memory_bytes=32 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=12 * 60 * 60,
    source_scan_bytes=0,
    output_bytes=64 * 1024**2,
)
_SOURCE_ASSESSMENT_CAMPAIGN_BUDGET = ResourceBudget(
    cpu_cores=8,
    memory_bytes=32 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=48 * 24 * 60 * 60,
    source_scan_bytes=0,
    output_bytes=6 * 1024**3,
)


@dataclass(frozen=True, slots=True)
class GymToraxExcludedDiagnosticBoundary(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-excluded-diagnostic-boundary'
    )
    VERSION: ClassVar[str] = '1.0.0'
    predecessors: GymToraxPredecessorInputs

    boundary_id: str
    freeze_artifact: ObjectIdentity
    terminal_acquisition_artifact: ObjectIdentity
    disposition: str
    outcomes_read: bool
    may_select_branch: bool
    may_promote_science: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.boundary_id, field_name="boundary_id")
        if (
            self.freeze_artifact != self.predecessors.diagnostic_freeze
            or self.terminal_acquisition_artifact
            != self.predecessors.diagnostic_terminal
        ):
            raise ValueError("diagnostic boundary binds another corpus")
        if self.disposition != 'EXPOSED_SOURCE_ASSESSMENT_DIAGNOSTIC_NONPROMOTING':
            raise ValueError("diagnostic boundary disposition changed")
        if not self.outcomes_read or self.may_select_branch or self.may_promote_science:
            raise ValueError("source-assessment diagnostic cannot be treated as fresh evidence")
        if self.outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE:
            raise ValueError("source-assessment diagnostic outcome access changed")
        if self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE:
            raise ValueError("source-assessment diagnostic visibility changed")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("source-assessment diagnostic evidence ceiling changed")


@dataclass(frozen=True, slots=True)
class GymToraxFrozenBranchSelectionRule(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-frozen-branch-selection-rule'
    VERSION: ClassVar[str] = '1.0.0'
    predecessors: GymToraxPredecessorInputs

    rule_id: str
    prospective_protocol_parent: ObjectIdentity
    excluded_feasibility_spec: ObjectIdentity
    primary_branch_id: str
    alternate_branch_id: str
    alternate_reason_code_precedence: tuple[str, ...]
    hfr_contract_evidence: tuple[ObjectIdentity, ...]
    ordinary_parent_episode_count: int
    frozen_before_q: bool
    primary_requires_supported_g2: bool
    alternate_is_nonpromotable: bool
    parent_issue_started: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.rule_id, field_name="rule_id")
        if self.prospective_protocol_parent != self.predecessors.protocol:
            raise ValueError("branch rule binds another prospective protocol")
        if self.excluded_feasibility_spec != self.predecessors.excluded_feasibility:
            raise ValueError("branch rule changes the excluded-feasibility question")
        for name, value, expected in (
            ("primary_branch_id", self.primary_branch_id, GYM_TORAX_PRIMARY_BRANCH_ID),
            ("alternate_branch_id", self.alternate_branch_id, GYM_TORAX_FINITE_ACTION_RECURRENCE_BRANCH_ID),
        ):
            validate_nonempty(value, field_name=name)
            if value != expected:
                raise ValueError(f"{name} changed")
        if self.alternate_reason_code_precedence != _ALTERNATE_REASON_CODE_PRECEDENCE:
            raise ValueError("branch-rule reason precedence changed")
        require_sorted_unique_ids(
            self.hfr_contract_evidence,
            attribute="object_id",
            field_name="hfr_contract_evidence",
        )
        if self.hfr_contract_evidence != self.predecessors.hfr_contract_evidence:
            raise ValueError("branch rule finite-action recurrence contract closure changed")
        if self.ordinary_parent_episode_count != 576:
            raise ValueError("branch rule parent roster size changed")
        if not all(
            (
                self.frozen_before_q,
                self.primary_requires_supported_g2,
                self.alternate_is_nonpromotable,
            )
        ):
            raise ValueError("branch rule weakens a frozen condition")
        if self.parent_issue_started:
            raise ValueError("branch selection must precede parent issue")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("branch rule must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("branch rule must remain prospective")


@dataclass(frozen=True, slots=True)
class GymToraxSourceAssessmentCell(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-source-assessment-cell'

    cell_id: str
    coordinate_id: str
    episode_id: str
    physical_unit_instance_id: str
    request: GymToraxFieldMetadataEpisodeRequest

    def __post_init__(self) -> None:
        for name, value in (
            ("cell_id", self.cell_id),
            ("coordinate_id", self.coordinate_id),
            ("episode_id", self.episode_id),
            ("physical_unit_instance_id", self.physical_unit_instance_id),
        ):
            validate_stable_id(value, field_name=name)
        expected_stem = self.request.request_id.removeprefix("request.")
        if (
            self.request.request_id == expected_stem
            or self.episode_id != f"episode.{expected_stem}"
        ):
            raise ValueError("fresh source assessment request/episode identity diverged")
        if self.request.request_role is not GymToraxRequestRole.SCIENTIFIC_EPISODE:
            raise ValueError(
                "fresh source assessment cell is not a qualified scientific source episode"
            )
        if (
            self.request.preparation.physical_independent_unit_id
            != self.physical_unit_instance_id
        ):
            raise ValueError("fresh source assessment cell changes its physical preparation unit")
        if self.request.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("excluded source assessment must remain nonpromotable")
        if self.request.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("fresh source assessment outcomes must remain sealed through acquisition")


@dataclass(frozen=True, slots=True)
class GymToraxNativeMetadataCanaryFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-metadata-canary-freeze'
    )
    VERSION: ClassVar[str] = '1.0.0'
    predecessors: GymToraxPredecessorInputs

    freeze_id: str
    run_id: str
    relative_root: str
    prospective_protocol_parent: ObjectIdentity
    extraction_manifest: ObjectIdentity
    field_metadata_manifest: ObjectIdentity
    implementation_source_closure: ImplementationSourceClosure
    scientific_approval: ObjectIdentity
    request: GymToraxFieldMetadataEpisodeRequest
    scientific_seed_input: GymToraxSourceAssessmentSeed
    task_resource_budget: ResourceBudget
    maximum_concurrency: int
    required_operation_authority_ids: tuple[str, ...]
    frozen_before_source_reset: bool
    source_reset_count_at_freeze: int
    outcomes_read: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        if (
            self.freeze_id != GYM_TORAX_CANARY_FREEZE_ID
            or self.run_id != GYM_TORAX_NATIVE_CANARY_RUN_ID
        ):
            raise ValueError("native canary freeze identity changed")
        validate_relative_locator(self.relative_root)
        if self.relative_root != GYM_TORAX_NATIVE_CANARY_RELATIVE_ROOT:
            raise ValueError("native canary storage namespace changed")
        if self.prospective_protocol_parent != self.predecessors.protocol:
            raise ValueError("native canary binds another protocol parent")
        if (
            self.extraction_manifest.object_schema
            != GymToraxBoundedExtractionManifest.SCHEMA
        ):
            raise ValueError("native canary lacks the bounded source closure")
        if (
            self.field_metadata_manifest.object_schema
            != GymToraxFieldMetadataManifest.SCHEMA
        ):
            raise ValueError("native canary lacks the field metadata closure")
        _validate_clean_source_closure(self.implementation_source_closure)
        if (
            not isinstance(self.scientific_seed_input, GymToraxSourceAssessmentSeed)
            or self.scientific_seed_input.role != "excluded-metadata-canary"
        ):
            raise ValueError("metadata canary requires its typed scientific seed input")
        if (
            self.request.preparation.environment_seed
            != self.scientific_seed_input.environment_seed
        ):
            raise ValueError("metadata canary preparation changes its scientific seed")
        if self.request.request_role is not GymToraxRequestRole.METADATA_CANARY:
            raise ValueError("native canary freeze contains another request role")
        if self.request.extraction_manifest != self.extraction_manifest:
            raise ValueError("native canary request changes the extraction closure")
        if self.request.field_metadata_manifest != self.field_metadata_manifest:
            raise ValueError("native canary request changes the metadata closure")
        if self.task_resource_budget != _TASK_BUDGET:
            raise ValueError("native canary task budget changed")
        if self.maximum_concurrency != 1:
            raise ValueError("native canary concurrency must remain one")
        if self.required_operation_authority_ids != _CANARY_AUTHORITY_IDS:
            raise ValueError("native canary authority roster changed")
        _validate_prereset_freeze(
            frozen=self.frozen_before_source_reset,
            source_reset_count=self.source_reset_count_at_freeze,
            outcomes_read=self.outcomes_read,
            outcome_access=self.outcome_access,
            visibility_ceiling=self.visibility_ceiling,
            evidence_ceiling=self.evidence_ceiling,
        )


@dataclass(frozen=True, slots=True)
class GymToraxSourceAssessmentFreeze(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/simulators/gym-torax-native/gym-torax-source-assessment-freeze'
    VERSION: ClassVar[str] = '1.0.0'
    predecessors: GymToraxPredecessorInputs

    freeze_id: str
    run_id: str
    relative_root: str
    prospective_protocol_parent: ObjectIdentity
    diagnostic_boundary: GymToraxExcludedDiagnosticBoundary
    canary_freeze: ObjectIdentity
    canary_receipt: ObjectIdentity
    source_qualification: ObjectIdentity
    extraction_manifest: ObjectIdentity
    field_metadata_manifest: ObjectIdentity
    implementation_source_closure: ImplementationSourceClosure
    scientific_approval: ObjectIdentity
    branch_rule: GymToraxFrozenBranchSelectionRule
    cells: tuple[GymToraxSourceAssessmentCell, ...]
    scientific_seed_inputs: tuple[GymToraxSourceAssessmentSeed, ...]
    task_resource_budget: ResourceBudget
    campaign_resource_budget: ResourceBudget
    maximum_concurrency: int
    required_operation_authority_ids: tuple[str, ...]
    frozen_before_source_reset: bool
    source_reset_count_at_freeze: int
    source_assessment_outcomes_read: bool
    diagnostic_outcomes_used: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        if (
            self.freeze_id != GYM_TORAX_FRESH_SOURCE_ASSESSMENT_FREEZE_ID
            or self.run_id != GYM_TORAX_FRESH_SOURCE_ASSESSMENT_RUN_ID
        ):
            raise ValueError("fresh source assessment freeze identity changed")
        validate_relative_locator(self.relative_root)
        if self.relative_root != GYM_TORAX_FRESH_SOURCE_ASSESSMENT_RELATIVE_ROOT:
            raise ValueError("fresh source assessment storage namespace changed")
        if self.prospective_protocol_parent != self.predecessors.protocol:
            raise ValueError("fresh source assessment binds another protocol parent")
        if self.canary_freeze.object_schema != GymToraxNativeMetadataCanaryFreeze.SCHEMA:
            raise ValueError("fresh source assessment lacks the exact canary freeze")
        if (
            self.canary_receipt.object_schema
            != GymToraxNativeMetadataCanaryReceipt.SCHEMA
        ):
            raise ValueError("fresh source assessment lacks the typed canary receipt")
        if (
            self.source_qualification.object_schema
            != GymToraxNativeSourceQualificationReceipt.SCHEMA
        ):
            raise ValueError("fresh source assessment lacks the typed source qualification")
        if (
            self.extraction_manifest.object_schema
            != GymToraxBoundedExtractionManifest.SCHEMA
        ):
            raise ValueError("fresh source assessment lacks the bounded source closure")
        if (
            self.field_metadata_manifest.object_schema
            != GymToraxFieldMetadataManifest.SCHEMA
        ):
            raise ValueError("fresh source assessment lacks the metadata closure")
        _validate_clean_source_closure(self.implementation_source_closure)
        if (
            self.branch_rule.prospective_protocol_parent
            != self.prospective_protocol_parent
        ):
            raise ValueError("fresh source assessment branch rule changes its protocol parent")
        require_sorted_unique_ids(self.cells, attribute="cell_id", field_name="cells")
        require_source_qualification_seed_census(self.scientific_seed_inputs)
        _validate_source_assessment_roster(self.cells, self)
        if self.task_resource_budget != _TASK_BUDGET:
            raise ValueError("fresh source assessment task budget changed")
        if self.campaign_resource_budget != _SOURCE_ASSESSMENT_CAMPAIGN_BUDGET:
            raise ValueError("fresh source assessment campaign budget changed")
        if not self.campaign_resource_budget.contains(self.task_resource_budget):
            raise ValueError("fresh source assessment campaign budget cannot contain one task")
        if self.maximum_concurrency != 1:
            raise ValueError("fresh source assessment concurrency must remain one")
        if self.required_operation_authority_ids != _SOURCE_ASSESSMENT_AUTHORITY_IDS:
            raise ValueError("fresh source assessment authority roster changed")
        if self.diagnostic_outcomes_used:
            raise ValueError("outcome-visible source-assessment diagnostic cannot inform fresh source assessment")
        _validate_prereset_freeze(
            frozen=self.frozen_before_source_reset,
            source_reset_count=self.source_reset_count_at_freeze,
            outcomes_read=self.source_assessment_outcomes_read,
            outcome_access=self.outcome_access,
            visibility_ceiling=self.visibility_ceiling,
            evidence_ceiling=self.evidence_ceiling,
        )


_PREPARATION_ROWS = (
    (0, "b", "0.995473", "0.8513313", "0.9946155", "1.000861"),
    (0, "bt", "1.010769", "0.8513313", "0.9946155", "1.000861"),
    (0, "c", "0.995473", "0.8513313", "1.0045729", "1.000861"),
    (0, "t", "1.010769", "0.8513313", "1.0045729", "1.000861"),
    (1, "b", "1.0061309", "0.8498214", "0.9933037", "1.0088225"),
    (1, "bt", "1.0133926", "0.8498214", "0.9933037", "1.0088225"),
    (1, "c", "1.0061309", "0.8498214", "0.99543", "1.0088225"),
    (1, "t", "1.0133926", "0.8498214", "0.99543", "1.0088225"),
    (2, "b", "1.0097284", "0.8489359", "0.9904436", "1.0019396"),
    (2, "bt", "1.0191128", "0.8489359", "0.9904436", "1.0019396"),
    (2, "c", "1.0097284", "0.8489359", "1.0003706", "1.0019396"),
    (2, "t", "1.0191128", "0.8489359", "1.0003706", "1.0019396"),
)


def _validate_clean_source_closure(value: ImplementationSourceClosure) -> None:
    if value.kind is not SourceClosureKind.CLEAN_GIT_COMMIT or not value.clean_worktree:
        raise ValueError("Gym-TORAX execution freezes require a clean Git commit")


def _validate_prereset_freeze(
    *,
    frozen: bool,
    source_reset_count: int,
    outcomes_read: bool,
    outcome_access: OutcomeAccess,
    visibility_ceiling: VisibilityCeiling,
    evidence_ceiling: EvidenceCeiling,
) -> None:
    if not frozen or source_reset_count != 0 or outcomes_read:
        raise ValueError(
            "Gym-TORAX freeze must precede every bound source reset and outcome"
        )
    if outcome_access is not OutcomeAccess.OUTCOME_BLIND:
        raise ValueError("Gym-TORAX freeze must remain outcome-blind")
    if visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
        raise ValueError("Gym-TORAX freeze must remain prospective")
    if evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
        raise ValueError("Gym-TORAX source assessment freeze cannot promote scientific truth")


def _diagnostic_boundary(
    *, predecessors: GymToraxPredecessorInputs
) -> GymToraxExcludedDiagnosticBoundary:
    return GymToraxExcludedDiagnosticBoundary(
        predecessors=predecessors,
        boundary_id='boundary.tokamak-control.direct-q-diagnostic',
        freeze_artifact=predecessors.diagnostic_freeze,
        terminal_acquisition_artifact=predecessors.diagnostic_terminal,
        disposition='EXPOSED_SOURCE_ASSESSMENT_DIAGNOSTIC_NONPROMOTING',
        outcomes_read=True,
        may_select_branch=False,
        may_promote_science=False,
        outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def build_gym_torax_frozen_branch_rule(
    *, predecessors: GymToraxPredecessorInputs
) -> GymToraxFrozenBranchSelectionRule:
    return GymToraxFrozenBranchSelectionRule(
        predecessors=predecessors,
        rule_id='branch-rule.tokamak-control.fresh-source-branch-selection',
        prospective_protocol_parent=predecessors.protocol,
        excluded_feasibility_spec=predecessors.excluded_feasibility,
        primary_branch_id=GYM_TORAX_PRIMARY_BRANCH_ID,
        alternate_branch_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_BRANCH_ID,
        alternate_reason_code_precedence=_ALTERNATE_REASON_CODE_PRECEDENCE,
        hfr_contract_evidence=predecessors.hfr_contract_evidence,
        ordinary_parent_episode_count=576,
        frozen_before_q=True,
        primary_requires_supported_g2=True,
        alternate_is_nonpromotable=True,
        parent_issue_started=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
    )


def build_gym_torax_native_metadata_canary_freeze(
    *,
    predecessors: GymToraxPredecessorInputs,
    extraction_manifest: GymToraxBoundedExtractionManifest,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    implementation_source_closure: ImplementationSourceClosure,
    scientific_approval: ObjectIdentity,
    scientific_seed_input: GymToraxSourceAssessmentSeed | None = None,
) -> GymToraxNativeMetadataCanaryFreeze:
    seed = (
        gym_torax_metadata_canary_seed()
        if scientific_seed_input is None else scientific_seed_input
    )
    if (
        not isinstance(seed, GymToraxSourceAssessmentSeed)
        or seed.role != "excluded-metadata-canary"
    ):
        raise ValueError("metadata canary requires its typed scientific seed input")
    extraction_identity = ObjectIdentity.from_record(
        extraction_manifest.manifest_id,
        extraction_manifest,
    )
    metadata_identity = ObjectIdentity.from_record(
        field_metadata_manifest.manifest_id,
        field_metadata_manifest,
    )
    unit_id = 'unit.tokamak-control.native-metadata-canary'
    preparation = GymToraxPreparation(
        preparation_id='preparation.tokamak-control.native-metadata-canary',
        physical_independent_unit_id=unit_id,
        environment_seed=seed.environment_seed,
        initial_temperature_scale=Decimal("1"),
        initial_density_nbar=Decimal("0.85"),
        bootstrap_multiplier=Decimal("1"),
        inner_transport_scale=Decimal("1"),
    )
    request = GymToraxFieldMetadataEpisodeRequest(
        request_id='request.tokamak-control.native-metadata-canary',
        request_role=GymToraxRequestRole.METADATA_CANARY,
        source_qualification=None,
        extraction_manifest=extraction_identity,
        field_metadata_manifest=metadata_identity,
        preparation=preparation,
        numerical_member=next(
            value
            for value in gym_torax_numerical_members()
            if value.member_id == GYM_TORAX_PRIMARY_MEMBER_ID
        ),
        schedule=build_gym_torax_native_schedule(
            build_gym_torax_action_word('action-word.tokamak-control.lower-ip')
        ),
        maximum_output_bytes=_TASK_BUDGET.output_bytes,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )
    return GymToraxNativeMetadataCanaryFreeze(
        predecessors=predecessors,
        freeze_id=GYM_TORAX_CANARY_FREEZE_ID,
        run_id=GYM_TORAX_NATIVE_CANARY_RUN_ID,
        relative_root=GYM_TORAX_NATIVE_CANARY_RELATIVE_ROOT,
        prospective_protocol_parent=predecessors.protocol,
        extraction_manifest=extraction_identity,
        field_metadata_manifest=metadata_identity,
        implementation_source_closure=implementation_source_closure,
        scientific_approval=scientific_approval,
        request=request,
        scientific_seed_input=seed,
        task_resource_budget=_TASK_BUDGET,
        maximum_concurrency=1,
        required_operation_authority_ids=_CANARY_AUTHORITY_IDS,
        frozen_before_source_reset=True,
        source_reset_count_at_freeze=0,
        outcomes_read=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


def _source_assessment_cells(
    *,
    source_qualification: GymToraxNativeSourceQualificationReceipt,
    extraction_identity: ObjectIdentity,
    metadata_identity: ObjectIdentity,
    scientific_seed_inputs: tuple[GymToraxSourceAssessmentSeed, ...],
) -> tuple[GymToraxSourceAssessmentCell, ...]:
    require_source_qualification_seed_census(scientific_seed_inputs)
    seeds = {(seed.counter, seed.cell_kind): seed for seed in scientific_seed_inputs}
    qualification_identity = ObjectIdentity.from_record(
        source_qualification.receipt_id,
        source_qualification,
    )
    members = gym_torax_numerical_members()
    cells: list[GymToraxSourceAssessmentCell] = []
    for counter, kind, temperature, density, bootstrap, transport in _PREPARATION_ROWS:
        unit_stem = f"{counter:08d}.{kind}"
        unit_id = f'unit.tokamak-control.source-assessment.{unit_stem}'
        preparation = GymToraxPreparation(
            preparation_id=f'preparation.tokamak-control.source-assessment.{unit_stem}',
            physical_independent_unit_id=unit_id,
            environment_seed=seeds[(counter, kind)].environment_seed,
            initial_temperature_scale=Decimal(temperature),
            initial_density_nbar=Decimal(density),
            bootstrap_multiplier=Decimal(bootstrap),
            inner_transport_scale=Decimal(transport),
        )
        for member in members:
            member_slug = member.member_id.rsplit(".", maxsplit=1)[-1]
            for word_id in GYM_TORAX_ACTION_WORD_IDS:
                word_slug = word_id.rsplit(".", maxsplit=1)[-1]
                stem = f'tokamak-control.source-assessment.{unit_stem}.{member_slug}.{word_slug}'
                request = GymToraxFieldMetadataEpisodeRequest(
                    request_id=f"request.{stem}",
                    request_role=GymToraxRequestRole.SCIENTIFIC_EPISODE,
                    source_qualification=qualification_identity,
                    extraction_manifest=extraction_identity,
                    field_metadata_manifest=metadata_identity,
                    preparation=preparation,
                    numerical_member=member,
                    schedule=build_gym_torax_native_schedule(
                        build_gym_torax_action_word(word_id)
                    ),
                    maximum_output_bytes=_TASK_BUDGET.output_bytes,
                    outcome_access=OutcomeAccess.EVALUATION_SEALED,
                    evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                )
                cells.append(
                    GymToraxSourceAssessmentCell(
                        cell_id=f"cell.{stem}",
                        coordinate_id=f"coordinate.{stem}",
                        episode_id=f"episode.{stem}",
                        physical_unit_instance_id=unit_id,
                        request=request,
                    )
                )
    return tuple(sorted(cells, key=lambda value: value.cell_id))


def _validate_source_assessment_roster(
    cells: tuple[GymToraxSourceAssessmentCell, ...],
    freeze: GymToraxSourceAssessmentFreeze,
) -> None:
    if len(cells) != 96:
        raise ValueError("fresh source assessment requires exactly 96 request cells")
    episode_ids = tuple(value.episode_id for value in cells)
    coordinate_ids = tuple(value.coordinate_id for value in cells)
    if len(set(episode_ids)) != 96 or len(set(coordinate_ids)) != 96:
        raise ValueError("fresh source assessment episode or coordinate identity repeats")
    unit_counts = Counter(value.physical_unit_instance_id for value in cells)
    if len(unit_counts) != 12 or set(unit_counts.values()) != {8}:
        raise ValueError("fresh source assessment must nest eight views in each of 12 physical units")
    expected_seeds = {
        (seed.counter, seed.cell_kind): seed.environment_seed
        for seed in freeze.scientific_seed_inputs
    }
    for unit_id in unit_counts:
        nested = tuple(
            value for value in cells if value.physical_unit_instance_id == unit_id
        )
        counter_string, kind = unit_id.rsplit(".", maxsplit=2)[-2:]
        if nested[0].request.preparation.environment_seed != expected_seeds.get(
            (int(counter_string), kind)
        ):
            raise ValueError("fresh source assessment root changes its declared scientific seed")
        if len({value.request.preparation for value in nested}) != 1:
            raise ValueError("fresh source assessment nested views change their preparation")
        if {value.request.numerical_member.member_id for value in nested} != {
            value.member_id for value in gym_torax_numerical_members()
        }:
            raise ValueError("fresh source assessment physical unit lacks a numerical member")
        if {value.request.schedule.action_word.word_id for value in nested} != set(
            GYM_TORAX_ACTION_WORD_IDS
        ):
            raise ValueError("fresh source assessment physical unit lacks an action word")
    for cell in cells:
        request = cell.request
        if (
            request.source_qualification != freeze.source_qualification
            or request.extraction_manifest != freeze.extraction_manifest
            or request.field_metadata_manifest != freeze.field_metadata_manifest
        ):
            raise ValueError("fresh source assessment request changes its frozen source closure")


def build_gym_torax_source_assessment_freeze(
    *,
    predecessors: GymToraxPredecessorInputs,
    canary_freeze: GymToraxNativeMetadataCanaryFreeze,
    canary_receipt: GymToraxNativeMetadataCanaryReceipt,
    source_qualification: GymToraxNativeSourceQualificationReceipt,
    extraction_manifest: GymToraxBoundedExtractionManifest,
    field_metadata_manifest: GymToraxFieldMetadataManifest,
    implementation_source_closure: ImplementationSourceClosure,
    scientific_approval: ObjectIdentity,
    scientific_seed_inputs: tuple[GymToraxSourceAssessmentSeed, ...] | None = None,
) -> GymToraxSourceAssessmentFreeze:
    seeds = (
        gym_torax_source_qualification_seeds()
        if scientific_seed_inputs is None else scientific_seed_inputs
    )
    require_source_qualification_seed_census(seeds)
    if canary_freeze.predecessors != predecessors:
        raise ValueError("fresh source assessment canary predecessor closure differs")
    if (
        source_qualification.disposition
        is not GymToraxNativeSourceQualificationDisposition.QUALIFIED
    ):
        raise ValueError("fresh source assessment cannot freeze before source qualification passes")
    canary_receipt_identity = ObjectIdentity.from_record(
        canary_receipt.receipt_id, canary_receipt
    )
    if source_qualification.canary_receipt != canary_receipt_identity:
        raise ValueError("fresh source assessment source qualification changes the canary receipt")
    extraction_identity = ObjectIdentity.from_record(
        extraction_manifest.manifest_id,
        extraction_manifest,
    )
    metadata_identity = ObjectIdentity.from_record(
        field_metadata_manifest.manifest_id,
        field_metadata_manifest,
    )
    if (
        source_qualification.extraction_manifest != extraction_identity
        or source_qualification.field_metadata_manifest != metadata_identity
    ):
        raise ValueError("fresh source assessment source qualification changes a source manifest")
    qualification_identity = ObjectIdentity.from_record(
        source_qualification.receipt_id,
        source_qualification,
    )
    cells = _source_assessment_cells(
        source_qualification=source_qualification,
        extraction_identity=extraction_identity,
        metadata_identity=metadata_identity,
        scientific_seed_inputs=seeds,
    )
    return GymToraxSourceAssessmentFreeze(
        predecessors=predecessors,
        freeze_id=GYM_TORAX_FRESH_SOURCE_ASSESSMENT_FREEZE_ID,
        run_id=GYM_TORAX_FRESH_SOURCE_ASSESSMENT_RUN_ID,
        relative_root=GYM_TORAX_FRESH_SOURCE_ASSESSMENT_RELATIVE_ROOT,
        prospective_protocol_parent=predecessors.protocol,
        diagnostic_boundary=_diagnostic_boundary(predecessors=predecessors),
        canary_freeze=ObjectIdentity.from_record(
            canary_freeze.freeze_id, canary_freeze
        ),
        canary_receipt=canary_receipt_identity,
        source_qualification=qualification_identity,
        extraction_manifest=extraction_identity,
        field_metadata_manifest=metadata_identity,
        implementation_source_closure=implementation_source_closure,
        scientific_approval=scientific_approval,
        branch_rule=build_gym_torax_frozen_branch_rule(predecessors=predecessors),
        cells=cells,
        scientific_seed_inputs=seeds,
        task_resource_budget=_TASK_BUDGET,
        campaign_resource_budget=_SOURCE_ASSESSMENT_CAMPAIGN_BUDGET,
        maximum_concurrency=1,
        required_operation_authority_ids=_SOURCE_ASSESSMENT_AUTHORITY_IDS,
        frozen_before_source_reset=True,
        source_reset_count_at_freeze=0,
        source_assessment_outcomes_read=False,
        diagnostic_outcomes_used=False,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


__all__ = [
    'GYM_TORAX_CANARY_FREEZE_ID',
    'GYM_TORAX_FRESH_SOURCE_ASSESSMENT_FREEZE_ID',
    'GYM_TORAX_FRESH_SOURCE_ASSESSMENT_RELATIVE_ROOT',
    'GYM_TORAX_FRESH_SOURCE_ASSESSMENT_RUN_ID',
    'GYM_TORAX_FINITE_ACTION_RECURRENCE_BRANCH_ID',
    'GYM_TORAX_NATIVE_CANARY_RELATIVE_ROOT',
    'GYM_TORAX_NATIVE_CANARY_RUN_ID',
    'GYM_TORAX_PRIMARY_BRANCH_ID',
    'GymToraxExcludedDiagnosticBoundary',
    'GymToraxSourceAssessmentCell',
    'GymToraxSourceAssessmentFreeze',
    'GymToraxFrozenBranchSelectionRule',
    'GymToraxNativeMetadataCanaryFreeze',
    'build_gym_torax_source_assessment_freeze',
    'build_gym_torax_frozen_branch_rule',
    'build_gym_torax_native_metadata_canary_freeze',
]

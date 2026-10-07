"Reveal-gated adjudication and frozen branch selection for fresh Gym-TORAX source assessment.\n\nThe evaluator reads only the exact 96-cell roster after terminal acquisition\nand typed reveal authority.  It checks delivery, clock, numerical, metadata,\npreparation and operator-source dispositions without materializing any array.\nThe current bounded source closure deliberately has no numerical controlled-I/O feasibility operator\nevaluator: an available operator surface is therefore UNEVALUABLE, never\nsilently promoted to supported.\n"

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

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
    validate_stable_id,
)
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent

from .diagnostic_contracts import GYM_TORAX_HORIZON_REQUESTS, GymToraxDeliveryDisposition, GymToraxNumericalDisposition, GymToraxObservationDisposition, GymToraxSourceDisposition
from .field_metadata_contracts import GymToraxFieldMetadataNativeEpisode
from .source_assessment_execution import GymToraxArtifactPublicationItem, GymToraxSourceAssessmentTerminalAcquisitionDisposition, GymToraxSourceAssessmentTerminalAcquisitionReceipt, GymToraxBoundedArtifactStore
from .source_assessment_protocol import GymToraxSourceAssessmentCell, GymToraxSourceAssessmentFreeze, GYM_TORAX_FINITE_ACTION_RECURRENCE_BRANCH_ID, GYM_TORAX_PRIMARY_BRANCH_ID


GYM_TORAX_SOURCE_ASSESSMENT_ADJUDICATION_RECEIPT_ID = 'result.tokamak-control.source-assessment.g2'
GYM_TORAX_SOURCE_ASSESSMENT_BRANCH_SELECTION_RECEIPT_ID = 'selection.tokamak-control.source-assessment'
CONTROLLED_IO_OPERATOR_API_UNAVAILABLE = "CONTROLLED_IO_OPERATOR_API_UNAVAILABLE"
CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE = (
    "CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE"
)
CONTROLLED_IO_EXCLUDED_SOURCE_REPEATABILITY_EVIDENCE_MISSING = (
    "CONTROLLED_IO_EXCLUDED_SOURCE_REPEATABILITY_EVIDENCE_MISSING"
)
CONTROLLED_IO_EXCLUDED_OPERATOR_SOURCE_DISPOSITION_INCONSISTENT = (
    "CONTROLLED_IO_EXCLUDED_OPERATOR_SOURCE_DISPOSITION_INCONSISTENT"
)
GYM_TORAX_SOURCE_ASSESSMENT_ORDINARY_READINESS_FAILED = "GYM_TORAX_SOURCE_ASSESSMENT_ORDINARY_READINESS_FAILED"
_EXECUTION_GRANTEE_ID = "operator.execution-service"
_EVALUATOR_GRANTEE_ID = "operator.evaluator-service"


class GymToraxSourceAssessmentAdjudicationDisposition(StrEnum):
    SUPPORTED = "SUPPORTED"
    NOT_SUPPORTED = "NOT_SUPPORTED"
    UNEVALUABLE = "UNEVALUABLE"
    ORDINARY_FAILURE = "ORDINARY_FAILURE"


class GymToraxSourceAssessmentBranchSelectionDisposition(StrEnum):
    PRIMARY_SELECTED = "PRIMARY_SELECTED"
    FINITE_ACTION_RECURRENCE_SELECTED = "FINITE_ACTION_RECURRENCE_SELECTED"
    NO_SELECTION = "NO_SELECTION"


@dataclass(frozen=True, slots=True)
class GymToraxSourceAssessmentAdjudicationReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-source-assessment-adjudication-receipt'
    )

    receipt_id: str
    freeze: ObjectIdentity
    terminal_acquisition: ObjectIdentity
    reveal_authority: ObjectIdentity
    episode_ids: tuple[ObjectIdentity, ...]
    physical_independent_unit_count: int
    numerical_member_count: int
    action_word_count: int
    ordinary_readiness_passed: bool
    disposition: GymToraxSourceAssessmentAdjudicationDisposition
    g2_reason_code: str | None
    reason_codes: tuple[str, ...]
    scientific_arrays_materialized: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.freeze.object_schema != GymToraxSourceAssessmentFreeze.SCHEMA:
            raise ValueError("source assessment adjudication binds another freeze schema")
        if (
            self.terminal_acquisition.object_schema
            != GymToraxSourceAssessmentTerminalAcquisitionReceipt.SCHEMA
        ):
            raise ValueError("source assessment adjudication binds another terminal schema")
        if self.reveal_authority.object_schema != StudyOperationAuthority.SCHEMA:
            raise ValueError("source assessment adjudication lacks typed reveal authority")
        require_sorted_unique_ids(
            self.episode_ids,
            attribute="object_id",
            field_name="episode_ids",
        )
        if len(self.episode_ids) != 96:
            raise ValueError("source assessment adjudication requires exactly 96 episodes")
        if (
            self.physical_independent_unit_count != 12
            or self.numerical_member_count != 2
            or self.action_word_count != 4
        ):
            raise ValueError("source assessment adjudication changes the frozen nested design")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is GymToraxSourceAssessmentAdjudicationDisposition.SUPPORTED:
            if (
                not self.ordinary_readiness_passed
                or self.g2_reason_code
                or self.reason_codes
            ):
                raise ValueError(
                    "supported source assessment requires complete ordinary and controlled-I/O feasibility evidence"
                )
        elif self.disposition is GymToraxSourceAssessmentAdjudicationDisposition.NOT_SUPPORTED:
            if (
                not self.ordinary_readiness_passed
                or self.g2_reason_code is None
                or self.reason_codes != (self.g2_reason_code,)
            ):
                raise ValueError("not-supported source assessment requires one closed controlled-I/O feasibility obstruction")
        elif self.disposition is GymToraxSourceAssessmentAdjudicationDisposition.ORDINARY_FAILURE:
            if self.ordinary_readiness_passed or self.g2_reason_code is not None:
                raise ValueError("ordinary source assessment failure cannot claim a controlled-I/O feasibility result")
        elif not self.ordinary_readiness_passed or self.g2_reason_code is not None:
            raise ValueError(
                "unevaluable controlled-I/O feasibility requires ordinary readiness and no obstruction"
            )
        if (
            self.disposition is not GymToraxSourceAssessmentAdjudicationDisposition.SUPPORTED
            and not self.reason_codes
        ):
            raise ValueError("non-supported source assessment adjudication requires exact reasons")
        if self.scientific_arrays_materialized:
            raise ValueError(
                "bounded source assessment adjudication cannot materialize scientific arrays"
            )
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("source assessment adjudication changes its nonpromotable reveal boundary")


@dataclass(frozen=True, slots=True)
class GymToraxSourceAssessmentBranchSelectionReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-source-assessment-branch-selection-receipt'
    )

    receipt_id: str
    freeze: ObjectIdentity
    adjudication: ObjectIdentity
    branch_rule: ObjectIdentity
    disposition: GymToraxSourceAssessmentBranchSelectionDisposition
    selected_branch_id: str | None
    selected_parent_episode_count: int
    controlling_reason_code: str | None
    hfr_contract_evidence: tuple[ObjectIdentity, ...]
    parent_issue_started: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.freeze.object_schema != GymToraxSourceAssessmentFreeze.SCHEMA:
            raise ValueError("branch selection binds another source assessment freeze")
        if self.adjudication.object_schema != GymToraxSourceAssessmentAdjudicationReceipt.SCHEMA:
            raise ValueError("branch selection binds another source assessment adjudication")
        require_sorted_unique_ids(
            self.hfr_contract_evidence,
            attribute="object_id",
            field_name="hfr_contract_evidence",
        )
        if self.disposition is GymToraxSourceAssessmentBranchSelectionDisposition.PRIMARY_SELECTED:
            if (
                self.selected_branch_id != GYM_TORAX_PRIMARY_BRANCH_ID
                or self.controlling_reason_code is not None
                or self.hfr_contract_evidence
            ):
                raise ValueError("PRIMARY selection changes its frozen rule")
        elif self.disposition is GymToraxSourceAssessmentBranchSelectionDisposition.FINITE_ACTION_RECURRENCE_SELECTED:
            if (
                self.selected_branch_id != GYM_TORAX_FINITE_ACTION_RECURRENCE_BRANCH_ID
                or self.controlling_reason_code is None
                or not self.hfr_contract_evidence
            ):
                raise ValueError(
                    "finite-action recurrence selection lacks its obstruction and static closure"
                )
        elif (
            self.selected_branch_id is not None
            or self.controlling_reason_code is not None
            or self.hfr_contract_evidence
        ):
            raise ValueError("no-selection receipt cannot imply a branch")
        if self.selected_parent_episode_count != (
            576 if self.selected_branch_id else 0
        ):
            raise ValueError("branch selection changes the conditional parent roster")
        if self.parent_issue_started:
            raise ValueError("branch receipt cannot claim parent issue")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("branch receipt lowers its revealed outcome ceiling")


def _identity(identifier: str, record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(identifier, record)


def _parent(
    identity: ObjectIdentity,
    *,
    visibility: VisibilityCeiling,
    access: OutcomeAccess,
) -> ArtifactLineageParent:
    return ArtifactLineageParent(
        identity=identity,
        visibility_ceiling=visibility,
        outcome_access=access,
    )


def _terminal_logical_id(freeze: GymToraxSourceAssessmentFreeze) -> str:
    return f"artifact.{freeze.run_id}.run-receipt"


def _episode_logical_id(episode_id: str) -> str:
    return f"artifact.{episode_id}"


def _episode_path(freeze: GymToraxSourceAssessmentFreeze, episode_id: str) -> str:
    return f"{freeze.relative_root}/episodes/{episode_id}.json"


def _adjudication_logical_id() -> str:
    return f'artifact.{GYM_TORAX_SOURCE_ASSESSMENT_ADJUDICATION_RECEIPT_ID}'


def _selection_logical_id() -> str:
    return f'artifact.{GYM_TORAX_SOURCE_ASSESSMENT_BRANCH_SELECTION_RECEIPT_ID}'


def _require_evaluation_authorities(
    *,
    freeze: GymToraxSourceAssessmentFreeze,
    terminal: GymToraxSourceAssessmentTerminalAcquisitionReceipt,
    execution_authority: StudyOperationAuthority,
    reveal_authority: StudyOperationAuthority,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> None:
    freeze_identity = _identity(freeze.freeze_id, freeze)
    require_study_authority(
        execution_authority,
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=freeze_identity,
        prerequisite_authority=freeze.scientific_approval,
        grantee_id=_EXECUTION_GRANTEE_ID,
        at_utc=at_utc,
    )
    execution_identity = _identity(
        execution_authority.authority_id, execution_authority
    )
    require_study_authority(
        reveal_authority,
        kind=StudyAuthorityKind.OUTCOME_REVEAL,
        subject=_identity(terminal.receipt_id, terminal),
        prerequisite_authority=execution_identity,
        grantee_id=_EVALUATOR_GRANTEE_ID,
        at_utc=at_utc,
    )
    require_study_authority(
        custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=freeze_identity,
        prerequisite_authority=None,
        grantee_id=_EXECUTION_GRANTEE_ID,
        storage_root_id=store.storage_root_id,
        relative_root=freeze.relative_root,
        at_utc=at_utc,
    )
    required = set(freeze.required_operation_authority_ids)
    if {
        execution_authority.authority_id,
        reveal_authority.authority_id,
        custody_authority.authority_id,
    } - required:
        raise PermissionError("source assessment evaluation authority ID is outside the freeze")
    if execution_authority.allows_actuation or reveal_authority.allows_actuation:
        raise PermissionError("Gym-TORAX simulator evaluation cannot permit actuation")


def _require_episode_lineage(
    *,
    freeze: GymToraxSourceAssessmentFreeze,
    cell: GymToraxSourceAssessmentCell,
    episode: GymToraxFieldMetadataNativeEpisode,
) -> None:
    request = cell.request
    if (
        episode.episode_id != cell.episode_id
        or episode.request != _identity(request.request_id, request)
        or episode.preparation
        != _identity(request.preparation.preparation_id, request.preparation)
        or episode.numerical_member
        != _identity(request.numerical_member.member_id, request.numerical_member)
        or episode.action_word
        != _identity(request.schedule.action_word.word_id, request.schedule.action_word)
        or episode.extraction_manifest != freeze.extraction_manifest
        or episode.field_metadata_manifest != freeze.field_metadata_manifest
        or episode.outcome_access is not OutcomeAccess.EVALUATION_SEALED
        or episode.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
    ):
        raise ValueError("source assessment episode lineage differs from the frozen cell")


def _ordinary_episode_ready(
    cell: GymToraxSourceAssessmentCell,
    episode: GymToraxFieldMetadataNativeEpisode,
) -> bool:
    expected_clocks = tuple(range(GYM_TORAX_HORIZON_REQUESTS + 1))
    if (
        not episode.source_reset_attempted
        or episode.state_clocks != expected_clocks
        or episode.missing_required_state_clocks
        or episode.last_valid_state_clock != GYM_TORAX_HORIZON_REQUESTS
        or episode.source_disposition is not GymToraxSourceDisposition.AVAILABLE
        or episode.delivery_disposition is not GymToraxDeliveryDisposition.COMPLETE
        or episode.numerical_disposition is not GymToraxNumericalDisposition.VALID
        or episode.observation_disposition
        is not GymToraxObservationDisposition.COMPLETE
        or episode.termination
        or episode.truncation
        or len(episode.deliveries) != GYM_TORAX_HORIZON_REQUESTS
        or not episode.blocks
        or any(value.nonfinite_value_count for value in episode.blocks)
    ):
        return False
    by_clock = {value.request_clock: value for value in episode.deliveries}
    if len(by_clock) != GYM_TORAX_HORIZON_REQUESTS:
        return False
    for row in cell.request.schedule.rows:
        delivery = by_clock.get(row.request_clock)
        if (
            delivery is None
            or delivery.receiver_clock != row.request_clock + 1
            or delivery.occurrence_id != row.controlled_occurrence_id
            or delivery.disposition is not GymToraxDeliveryDisposition.COMPLETE
            or delivery.requested != row.action
            or delivery.accepted != row.action
            or delivery.applied != row.action
            or delivery.realized is None
        ):
            return False
    return True


def _classify(
    episodes: tuple[GymToraxFieldMetadataNativeEpisode, ...],
    *,
    ordinary_ready: bool,
) -> tuple[GymToraxSourceAssessmentAdjudicationDisposition, str | None, tuple[str, ...]]:
    if not ordinary_ready:
        return (
            GymToraxSourceAssessmentAdjudicationDisposition.ORDINARY_FAILURE,
            None,
            (GYM_TORAX_SOURCE_ASSESSMENT_ORDINARY_READINESS_FAILED,),
        )
    dispositions = {value.operator_source.disposition for value in episodes}
    if dispositions == {GymToraxSourceDisposition.OPERATOR_API_UNAVAILABLE}:
        return (
            GymToraxSourceAssessmentAdjudicationDisposition.NOT_SUPPORTED,
            CONTROLLED_IO_OPERATOR_API_UNAVAILABLE,
            (CONTROLLED_IO_OPERATOR_API_UNAVAILABLE,),
        )
    if dispositions == {GymToraxSourceDisposition.SOURCE_UNAVAILABLE}:
        return (
            GymToraxSourceAssessmentAdjudicationDisposition.NOT_SUPPORTED,
            CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE,
            (CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE,),
        )
    if dispositions == {GymToraxSourceDisposition.AVAILABLE}:
        return (
            GymToraxSourceAssessmentAdjudicationDisposition.UNEVALUABLE,
            None,
            (CONTROLLED_IO_EXCLUDED_SOURCE_REPEATABILITY_EVIDENCE_MISSING,),
        )
    return (
        GymToraxSourceAssessmentAdjudicationDisposition.UNEVALUABLE,
        None,
        (CONTROLLED_IO_EXCLUDED_OPERATOR_SOURCE_DISPOSITION_INCONSISTENT,),
    )


def evaluate_gym_torax_q(
    *,
    freeze: GymToraxSourceAssessmentFreeze,
    terminal: GymToraxSourceAssessmentTerminalAcquisitionReceipt,
    execution_authority: StudyOperationAuthority,
    reveal_authority: StudyOperationAuthority,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> GymToraxSourceAssessmentAdjudicationReceipt:
    "Reveal and classify exact source assessment status, without numerical array access."

    if (
        terminal.freeze != _identity(freeze.freeze_id, freeze)
        or terminal.disposition is not GymToraxSourceAssessmentTerminalAcquisitionDisposition.COMPLETE
    ):
        raise ValueError(
            "source assessment evaluation requires the exact complete terminal acquisition"
        )
    persisted_terminal = store.load_optional(
        logical_artifact_id=_terminal_logical_id(freeze),
        relative_path=f"{freeze.relative_root}/run-receipt.json",
        record_type=GymToraxSourceAssessmentTerminalAcquisitionReceipt,
        maximum_bytes=4 * 1024**2,
    )
    if persisted_terminal != terminal:
        raise ValueError("source assessment terminal acquisition is not persisted exactly")
    _require_evaluation_authorities(
        freeze=freeze,
        terminal=terminal,
        execution_authority=execution_authority,
        reveal_authority=reveal_authority,
        custody_authority=custody_authority,
        store=store,
        at_utc=at_utc,
    )
    result_path = f"{freeze.relative_root}/adjudication/source-assessment-result.json"
    recovered = store.load_optional(
        logical_artifact_id=_adjudication_logical_id(),
        relative_path=result_path,
        record_type=GymToraxSourceAssessmentAdjudicationReceipt,
        maximum_bytes=4 * 1024**2,
    )
    if recovered is not None:
        if recovered.terminal_acquisition != _identity(terminal.receipt_id, terminal):
            raise ValueError(
                "recovered source assessment adjudication changes its terminal acquisition"
            )
        return recovered

    episodes: list[GymToraxFieldMetadataNativeEpisode] = []
    for cell in freeze.cells:
        episode = store.load_optional(
            logical_artifact_id=_episode_logical_id(cell.episode_id),
            relative_path=_episode_path(freeze, cell.episode_id),
            record_type=GymToraxFieldMetadataNativeEpisode,
            maximum_bytes=cell.request.maximum_output_bytes,
        )
        if episode is None:
            raise ValueError("complete source assessment terminal lacks an exact episode")
        _require_episode_lineage(freeze=freeze, cell=cell, episode=episode)
        episodes.append(episode)
    episode_tuple = tuple(episodes)
    episode_identities = tuple(
        sorted(
            (_identity(value.episode_id, value) for value in episode_tuple),
            key=lambda value: value.object_id,
        )
    )
    if episode_identities != terminal.episodes:
        raise ValueError("source assessment episode roster differs from terminal acquisition")
    ordinary_ready = all(
        _ordinary_episode_ready(cell, episode)
        for cell, episode in zip(freeze.cells, episode_tuple, strict=True)
    )
    disposition, g2_reason, reasons = _classify(
        episode_tuple,
        ordinary_ready=ordinary_ready,
    )
    result = GymToraxSourceAssessmentAdjudicationReceipt(
        receipt_id=GYM_TORAX_SOURCE_ASSESSMENT_ADJUDICATION_RECEIPT_ID,
        freeze=_identity(freeze.freeze_id, freeze),
        terminal_acquisition=_identity(terminal.receipt_id, terminal),
        reveal_authority=_identity(reveal_authority.authority_id, reveal_authority),
        episode_ids=episode_identities,
        physical_independent_unit_count=len(
            {value.physical_unit_instance_id for value in freeze.cells}
        ),
        numerical_member_count=len(
            {value.request.numerical_member.member_id for value in freeze.cells}
        ),
        action_word_count=len(
            {value.request.schedule.action_word.word_id for value in freeze.cells}
        ),
        ordinary_readiness_passed=ordinary_ready,
        disposition=disposition,
        g2_reason_code=g2_reason,
        reason_codes=reasons,
        scientific_arrays_materialized=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )
    store.publish_atomic(
        publication_scope_id=f"publication.{freeze.run_id}.adjudication",
        publication_scope_relative_root=freeze.relative_root,
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id=_adjudication_logical_id(),
                relative_path=result_path,
                record=result,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                lineage_parents=tuple(
                    sorted(
                        (
                            _parent(
                                _identity(terminal.receipt_id, terminal),
                                visibility=VisibilityCeiling.PROSPECTIVE,
                                access=OutcomeAccess.EVALUATION_SEALED,
                            ),
                            _parent(
                                _identity(
                                    reveal_authority.authority_id, reveal_authority
                                ),
                                visibility=VisibilityCeiling.PROSPECTIVE,
                                access=OutcomeAccess.EVALUATOR_REVEAL,
                            ),
                        ),
                        key=lambda value: value.identity.object_id,
                    )
                ),
            ),
        ),
    )
    return result


def select_gym_torax_q_branch(
    *,
    freeze: GymToraxSourceAssessmentFreeze,
    adjudication: GymToraxSourceAssessmentAdjudicationReceipt,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> GymToraxSourceAssessmentBranchSelectionReceipt:
    "Apply only the branch rule frozen before source assessment; never issue a parent."

    if adjudication.freeze != _identity(freeze.freeze_id, freeze):
        raise ValueError("branch selection adjudication binds another freeze")
    persisted = store.load_optional(
        logical_artifact_id=_adjudication_logical_id(),
        relative_path=f"{freeze.relative_root}/adjudication/source-assessment-result.json",
        record_type=GymToraxSourceAssessmentAdjudicationReceipt,
        maximum_bytes=4 * 1024**2,
    )
    if persisted != adjudication:
        raise ValueError("branch selection requires the exact persisted source assessment adjudication")
    require_study_authority(
        custody_authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=_identity(freeze.freeze_id, freeze),
        prerequisite_authority=None,
        grantee_id=_EXECUTION_GRANTEE_ID,
        storage_root_id=store.storage_root_id,
        relative_root=freeze.relative_root,
        at_utc=at_utc,
    )
    if custody_authority.authority_id not in freeze.required_operation_authority_ids:
        raise PermissionError(
            "branch-selection custody authority is outside the freeze"
        )
    selection_path = f"{freeze.relative_root}/adjudication/branch-selection.json"
    recovered = store.load_optional(
        logical_artifact_id=_selection_logical_id(),
        relative_path=selection_path,
        record_type=GymToraxSourceAssessmentBranchSelectionReceipt,
        maximum_bytes=2 * 1024**2,
    )
    if recovered is not None:
        if recovered.adjudication != _identity(adjudication.receipt_id, adjudication):
            raise ValueError("recovered branch selection changes its source assessment adjudication")
        return recovered

    if adjudication.disposition is GymToraxSourceAssessmentAdjudicationDisposition.SUPPORTED:
        branch_disposition = GymToraxSourceAssessmentBranchSelectionDisposition.PRIMARY_SELECTED
        selected_branch_id = freeze.branch_rule.primary_branch_id
        controlling_reason = None
        hfr_evidence: tuple[ObjectIdentity, ...] = ()
    elif (
        adjudication.disposition is GymToraxSourceAssessmentAdjudicationDisposition.NOT_SUPPORTED
        and adjudication.g2_reason_code
        in freeze.branch_rule.alternate_reason_code_precedence
    ):
        branch_disposition = GymToraxSourceAssessmentBranchSelectionDisposition.FINITE_ACTION_RECURRENCE_SELECTED
        selected_branch_id = freeze.branch_rule.alternate_branch_id
        controlling_reason = adjudication.g2_reason_code
        hfr_evidence = freeze.branch_rule.hfr_contract_evidence
    else:
        branch_disposition = GymToraxSourceAssessmentBranchSelectionDisposition.NO_SELECTION
        selected_branch_id = None
        controlling_reason = None
        hfr_evidence = ()
    selection = GymToraxSourceAssessmentBranchSelectionReceipt(
        receipt_id=GYM_TORAX_SOURCE_ASSESSMENT_BRANCH_SELECTION_RECEIPT_ID,
        freeze=_identity(freeze.freeze_id, freeze),
        adjudication=_identity(adjudication.receipt_id, adjudication),
        branch_rule=_identity(freeze.branch_rule.rule_id, freeze.branch_rule),
        disposition=branch_disposition,
        selected_branch_id=selected_branch_id,
        selected_parent_episode_count=576 if selected_branch_id else 0,
        controlling_reason_code=controlling_reason,
        hfr_contract_evidence=hfr_evidence,
        parent_issue_started=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )
    store.publish_atomic(
        publication_scope_id=f"publication.{freeze.run_id}.selection",
        publication_scope_relative_root=freeze.relative_root,
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id=_selection_logical_id(),
                relative_path=selection_path,
                record=selection,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                lineage_parents=tuple(
                    sorted(
                        (
                            _parent(
                                _identity(adjudication.receipt_id, adjudication),
                                visibility=VisibilityCeiling.OUTCOME_VISIBLE,
                                access=OutcomeAccess.EVALUATION_REVEALED,
                            ),
                            _parent(
                                _identity(freeze.freeze_id, freeze),
                                visibility=VisibilityCeiling.PROSPECTIVE,
                                access=OutcomeAccess.OUTCOME_BLIND,
                            ),
                        ),
                        key=lambda value: value.identity.object_id,
                    )
                ),
            ),
        ),
    )
    return selection


__all__ = [
    "CONTROLLED_IO_DERIVATIVE_SURFACE_UNAVAILABLE",
    "CONTROLLED_IO_EXCLUDED_OPERATOR_SOURCE_DISPOSITION_INCONSISTENT",
    "CONTROLLED_IO_EXCLUDED_SOURCE_REPEATABILITY_EVIDENCE_MISSING",
    "CONTROLLED_IO_OPERATOR_API_UNAVAILABLE",
    'GYM_TORAX_SOURCE_ASSESSMENT_ADJUDICATION_RECEIPT_ID',
    'GYM_TORAX_SOURCE_ASSESSMENT_BRANCH_SELECTION_RECEIPT_ID',
    "GYM_TORAX_SOURCE_ASSESSMENT_ORDINARY_READINESS_FAILED",
    'GymToraxSourceAssessmentAdjudicationDisposition',
    'GymToraxSourceAssessmentAdjudicationReceipt',
    'GymToraxSourceAssessmentBranchSelectionDisposition',
    'GymToraxSourceAssessmentBranchSelectionReceipt',
    'evaluate_gym_torax_q',
    'select_gym_torax_q_branch',
]

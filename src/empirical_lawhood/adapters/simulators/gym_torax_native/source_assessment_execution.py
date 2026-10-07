"Authority-guarded, bounded acquisition orchestration for bounded Gym-TORAX source assessment.\n\nThe store port owns guarded persistence and exact canonical decoding.  This\nmodule owns only freeze-before-reset ordering, one-cell-at-a-time acquisition,\nbyte-identical recovery and terminal acquisition accounting.  It never reads\nscientific array values and never performs source assessment adjudication.\n"

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
from typing import ClassVar, Protocol, TypeVar

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
    validate_stable_id,
)
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent

from .field_metadata_contracts import GymToraxFieldMetadataNativeEpisode
from .extraction_manifest import GymToraxBoundedExtractionManifest
from .field_metadata import GymToraxFieldMetadataManifest
from .metadata_barrier import GymToraxNativeMetadataCanaryDisposition, adjudicate_gym_torax_native_metadata_canary
from .source_assessment_protocol import GymToraxSourceAssessmentCell, GymToraxSourceAssessmentFreeze, GymToraxNativeMetadataCanaryFreeze
from .runtime import GymToraxEnvironmentFactory, GymToraxOutputBoundError, GymToraxRuntimePreflightError, GymToraxRuntimeInspector, acquire_gym_torax_episode, inspect_gym_torax_runtime
from .source_qualification import GymToraxNativeSourceQualificationDisposition, GymToraxNativeSourceQualificationReceipt, qualify_gym_torax_native_source


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)
_GYM_TORAX_EXECUTION_GRANTEE_ID = "operator.execution-service"


@dataclass(frozen=True, slots=True)
class GymToraxArtifactPublicationItem:
    logical_artifact_id: str
    relative_path: str
    record: CanonicalRecord
    visibility_ceiling: VisibilityCeiling
    outcome_access: OutcomeAccess
    lineage_parents: tuple[ArtifactLineageParent, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.logical_artifact_id, field_name="logical_artifact_id")
        validate_relative_locator(self.relative_path)
        require_sorted_unique_strings(
            tuple(value.identity.object_id for value in self.lineage_parents),
            field_name="lineage_parents",
        )


class GymToraxBoundedArtifactStore(Protocol):
    """Exact no-replace canonical artifact port; implementations may not scan."""

    @property
    def storage_root_id(self) -> str: ...

    def publish_atomic(
        self,
        *,
        publication_scope_id: str,
        publication_scope_relative_root: str,
        implementation_sha256: str,
        items: tuple[GymToraxArtifactPublicationItem, ...],
    ) -> tuple[ObjectIdentity, ...]: ...

    def load_optional(
        self,
        *,
        logical_artifact_id: str,
        relative_path: str,
        record_type: type[_RecordT],
        maximum_bytes: int,
    ) -> _RecordT | None: ...


class GymToraxOperationalDisposition(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"


class GymToraxSourceAssessmentTerminalAcquisitionDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class GymToraxNativeCanaryRunReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-native-canary-run-receipt'
    )

    receipt_id: str
    freeze: ObjectIdentity
    request: ObjectIdentity
    episode: ObjectIdentity | None
    metadata_canary: ObjectIdentity | None
    source_qualification: ObjectIdentity | None
    disposition: GymToraxOperationalDisposition
    reason_codes: tuple[str, ...]
    source_reset_attempted: bool
    scientific_values_decoded: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling
    grants_source_assessment_freeze_authority: bool
    grants_source_assessment_execution_authority: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.freeze.object_schema != GymToraxNativeMetadataCanaryFreeze.SCHEMA:
            raise ValueError("canary run receipt binds another freeze schema")
        if (
            self.request.object_schema
            != 'empirical-lawhood/simulators/gym-torax-native/gym-torax-field-metadata-episode-request'
        ):
            raise ValueError("canary run receipt binds another request schema")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        complete = all(
            value is not None
            for value in (self.episode, self.metadata_canary, self.source_qualification)
        )
        if self.disposition is GymToraxOperationalDisposition.SUCCEEDED:
            if not complete or self.reason_codes or not self.source_reset_attempted:
                raise ValueError(
                    "successful canary run requires its complete passing closure"
                )
        elif not self.reason_codes:
            raise ValueError("failed canary run requires typed reasons")
        presence_count = sum(
            value is not None
            for value in (self.episode, self.metadata_canary, self.source_qualification)
        )
        if presence_count not in {0, 3}:
            raise ValueError(
                "canary run receipt cannot retain a partial artifact closure"
            )
        if self.scientific_values_decoded:
            raise ValueError("metadata canary cannot decode scientific values")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("metadata canary run must remain outcome-blind")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("metadata canary run must remain prospective")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("metadata canary run cannot promote science")
        if self.grants_source_assessment_freeze_authority or self.grants_source_assessment_execution_authority:
            raise ValueError("metadata canary run receipt cannot grant authority")


@dataclass(frozen=True, slots=True)
class GymToraxSourceAssessmentCellAcquisitionReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-source-assessment-cell-acquisition-receipt'
    )

    receipt_id: str
    freeze: ObjectIdentity
    cell: ObjectIdentity
    request: ObjectIdentity
    expected_episode_id: str
    episode: ObjectIdentity | None
    disposition: GymToraxOperationalDisposition
    reason_codes: tuple[str, ...]
    attempt_count: int
    automatic_retry_performed: bool
    source_reset_attempted: bool
    scientific_values_decoded: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        validate_stable_id(self.expected_episode_id, field_name="expected_episode_id")
        if self.freeze.object_schema != GymToraxSourceAssessmentFreeze.SCHEMA:
            raise ValueError("source assessment cell receipt binds another freeze schema")
        if self.cell.object_schema != GymToraxSourceAssessmentCell.SCHEMA:
            raise ValueError("source assessment cell receipt binds another cell schema")
        if (
            self.request.object_schema
            != 'empirical-lawhood/simulators/gym-torax-native/gym-torax-field-metadata-episode-request'
        ):
            raise ValueError("source assessment cell receipt binds another request schema")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.attempt_count != 1 or self.automatic_retry_performed:
            raise ValueError(
                "bounded source assessment permits exactly one attempt and no automatic retry"
            )
        if self.disposition is GymToraxOperationalDisposition.SUCCEEDED:
            if self.episode is None or self.reason_codes:
                raise ValueError("successful source assessment acquisition requires one exact episode")
            if (
                self.episode.object_schema != GymToraxFieldMetadataNativeEpisode.SCHEMA
                or self.episode.object_id != self.expected_episode_id
            ):
                raise ValueError("source assessment cell receipt episode identity differs")
        elif self.episode is not None or not self.reason_codes:
            raise ValueError("failed source assessment acquisition cannot claim an episode")
        if self.scientific_values_decoded:
            raise ValueError("source assessment acquisition receipt cannot decode scientific values")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("source assessment acquisition receipt must remain evaluation-sealed")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("source assessment acquisition receipt must remain prospectively sealed")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("source assessment acquisition receipt cannot promote science")


@dataclass(frozen=True, slots=True)
class GymToraxSourceAssessmentTerminalAcquisitionReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-source-assessment-terminal-acquisition-receipt'
    )

    receipt_id: str
    freeze: ObjectIdentity
    cell_receipts: tuple[ObjectIdentity, ...]
    episodes: tuple[ObjectIdentity, ...]
    accounted_cell_count: int
    completed_episode_count: int
    failed_cell_count: int
    disposition: GymToraxSourceAssessmentTerminalAcquisitionDisposition
    reason_codes: tuple[str, ...]
    scientific_values_decoded: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.freeze.object_schema != GymToraxSourceAssessmentFreeze.SCHEMA:
            raise ValueError("source assessment terminal receipt binds another freeze schema")
        require_sorted_unique_ids(
            self.cell_receipts,
            attribute="object_id",
            field_name="cell_receipts",
        )
        require_sorted_unique_ids(
            self.episodes, attribute="object_id", field_name="episodes"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.accounted_cell_count != 96 or len(self.cell_receipts) != 96:
            raise ValueError("source assessment terminal acquisition requires all 96 cell receipts")
        if self.completed_episode_count != len(self.episodes):
            raise ValueError(
                "source assessment terminal completed count differs from episode identities"
            )
        if self.completed_episode_count + self.failed_cell_count != 96:
            raise ValueError("source assessment terminal counts do not cover the frozen roster")
        if self.disposition is GymToraxSourceAssessmentTerminalAcquisitionDisposition.COMPLETE:
            if (
                self.completed_episode_count != 96
                or self.failed_cell_count
                or self.reason_codes
            ):
                raise ValueError("complete source assessment terminal requires all 96 episodes")
        elif self.failed_cell_count <= 0 or not self.reason_codes:
            raise ValueError("incomplete source assessment terminal requires exact failed cells")
        if self.scientific_values_decoded:
            raise ValueError("source assessment terminal acquisition cannot decode scientific values")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("source assessment terminal acquisition must remain sealed")
        if self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE:
            raise ValueError("source assessment terminal acquisition must remain prospective")
        if self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE:
            raise ValueError("source assessment terminal acquisition cannot promote science")


class GymToraxSourceAssessmentIncompleteError(RuntimeError):
    "The frozen source assessment roster lacks at least one terminal cell receipt."

    def __init__(self, missing_cell_ids: Sequence[str]) -> None:
        self.missing_cell_ids = tuple(sorted(set(missing_cell_ids)))
        super().__init__(f"GYM_TORAX_SOURCE_ASSESSMENT_CELL_RECEIPTS_MISSING:{len(self.missing_cell_ids)}")


GymToraxFieldMetadataEpisodeAcquirer = Callable[[object], GymToraxFieldMetadataNativeEpisode]


def _record_identity(identifier: str, record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(identifier, record)


def _lineage_parent(
    identity: ObjectIdentity,
    outcome_access: OutcomeAccess,
) -> ArtifactLineageParent:
    return ArtifactLineageParent(
        identity=identity,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=outcome_access,
    )


def _freeze_logical_id(run_id: str) -> str:
    return f"artifact.{run_id}.freeze"


def _terminal_logical_id(run_id: str) -> str:
    return f"artifact.{run_id}.run-receipt"


def _episode_logical_id(episode_id: str) -> str:
    return f"artifact.{episode_id}"


def _cell_receipt_logical_id(cell_id: str) -> str:
    return f"artifact.receipt.{cell_id}"


def _episode_path(root: str, episode_id: str) -> str:
    return f"{root}/episodes/{episode_id}.json"


def _cell_receipt_path(root: str, cell_id: str) -> str:
    return f"{root}/receipts/{cell_id}.json"


def _require_custody_authority(
    *,
    freeze: CanonicalRecord,
    freeze_id: str,
    relative_root: str,
    authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> None:
    require_study_authority(
        authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=_record_identity(freeze_id, freeze),
        prerequisite_authority=None,
        grantee_id=_GYM_TORAX_EXECUTION_GRANTEE_ID,
        storage_root_id=store.storage_root_id,
        relative_root=relative_root,
        at_utc=at_utc,
    )


def _require_execution_authorities(
    *,
    freeze: GymToraxNativeMetadataCanaryFreeze | GymToraxSourceAssessmentFreeze,
    source_authority: StudyOperationAuthority,
    execution_authority: StudyOperationAuthority,
    at_utc: str,
) -> None:
    subject = _record_identity(freeze.freeze_id, freeze)
    require_study_authority(
        source_authority,
        kind=StudyAuthorityKind.SOURCE_ACQUISITION,
        subject=subject,
        prerequisite_authority=None,
        grantee_id=_GYM_TORAX_EXECUTION_GRANTEE_ID,
        at_utc=at_utc,
    )
    require_study_authority(
        execution_authority,
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=subject,
        prerequisite_authority=freeze.scientific_approval,
        grantee_id=_GYM_TORAX_EXECUTION_GRANTEE_ID,
        at_utc=at_utc,
    )
    if execution_authority.allows_actuation:
        raise PermissionError(
            "Gym-TORAX simulator execution authority cannot permit actuation"
        )
    expected_access = (
        freeze.request.outcome_access
        if isinstance(freeze, GymToraxNativeMetadataCanaryFreeze)
        else OutcomeAccess.EVALUATION_SEALED
    )
    if execution_authority.outcome_access is not expected_access:
        raise PermissionError("Gym-TORAX execution authority changes outcome access")
    if source_authority.authority_id not in freeze.required_operation_authority_ids:
        raise PermissionError("Gym-TORAX source authority ID is outside the freeze")
    if execution_authority.authority_id not in freeze.required_operation_authority_ids:
        raise PermissionError("Gym-TORAX execution authority ID is outside the freeze")


def _publish_freeze(
    *,
    freeze: GymToraxNativeMetadataCanaryFreeze | GymToraxSourceAssessmentFreeze,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> ObjectIdentity:
    _require_custody_authority(
        freeze=freeze,
        freeze_id=freeze.freeze_id,
        relative_root=freeze.relative_root,
        authority=custody_authority,
        store=store,
        at_utc=at_utc,
    )
    if custody_authority.authority_id not in freeze.required_operation_authority_ids:
        raise PermissionError("Gym-TORAX custody authority ID is outside the freeze")
    logical_id = _freeze_logical_id(freeze.run_id)
    return store.publish_atomic(
        publication_scope_id=f"publication.{freeze.run_id}",
        publication_scope_relative_root=freeze.relative_root,
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id=logical_id,
                relative_path=f"{freeze.relative_root}/freeze.json",
                record=freeze,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                lineage_parents=(
                    _lineage_parent(
                        freeze.prospective_protocol_parent,
                        OutcomeAccess.OUTCOME_BLIND,
                    ),
                ),
            ),
        ),
    )[0]


def publish_gym_torax_native_canary_freeze(
    *,
    freeze: GymToraxNativeMetadataCanaryFreeze,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> ObjectIdentity:
    return _publish_freeze(
        freeze=freeze,
        custody_authority=custody_authority,
        store=store,
        at_utc=at_utc,
    )


def publish_gym_torax_q_freeze(
    *,
    freeze: GymToraxSourceAssessmentFreeze,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> ObjectIdentity:
    return _publish_freeze(
        freeze=freeze,
        custody_authority=custody_authority,
        store=store,
        at_utc=at_utc,
    )


def _require_persisted_freeze(
    store: GymToraxBoundedArtifactStore,
    freeze: GymToraxNativeMetadataCanaryFreeze | GymToraxSourceAssessmentFreeze,
) -> None:
    observed = store.load_optional(
        logical_artifact_id=_freeze_logical_id(freeze.run_id),
        relative_path=f"{freeze.relative_root}/freeze.json",
        record_type=type(freeze),
        maximum_bytes=32 * 1024**2,
    )
    if observed != freeze:
        raise RuntimeError("GYM_TORAX_FREEZE_NOT_PERSISTED_EXACTLY")


def _technical_reason(error: Exception) -> tuple[str, ...]:
    if isinstance(error, GymToraxRuntimePreflightError):
        return tuple(f"NATIVE_SOURCE_ASSESSMENT_{value}" for value in error.reason_codes)
    if isinstance(error, GymToraxOutputBoundError):
        return ("NATIVE_SOURCE_ASSESSMENT_OUTPUT_BOUND_EXCEEDED",)
    token = re.sub(r"[^A-Z0-9]+", "_", type(error).__name__.upper()).strip("_")
    return (f"NATIVE_SOURCE_ASSESSMENT_TECHNICAL_{token or 'ERROR'}",)


def execute_gym_torax_native_metadata_canary(
    *,
    freeze: GymToraxNativeMetadataCanaryFreeze,
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
) -> GymToraxNativeCanaryRunReceipt:
    "Execute or recover the one real metadata canary, never scientific source assessment."

    _require_execution_authorities(
        freeze=freeze,
        source_authority=source_authority,
        execution_authority=execution_authority,
        at_utc=at_utc,
    )
    _require_custody_authority(
        freeze=freeze,
        freeze_id=freeze.freeze_id,
        relative_root=freeze.relative_root,
        authority=custody_authority,
        store=store,
        at_utc=at_utc,
    )
    if custody_authority.authority_id not in freeze.required_operation_authority_ids:
        raise PermissionError("Gym-TORAX canary custody authority ID is outside the freeze")
    _require_persisted_freeze(store, freeze)
    terminal_id = _terminal_logical_id(freeze.run_id)
    terminal_path = f"{freeze.relative_root}/run-receipt.json"
    recovered = store.load_optional(
        logical_artifact_id=terminal_id,
        relative_path=terminal_path,
        record_type=GymToraxNativeCanaryRunReceipt,
        maximum_bytes=2 * 1024**2,
    )
    if recovered is not None:
        if recovered.freeze != _record_identity(freeze.freeze_id, freeze):
            raise RuntimeError("GYM_TORAX_CANARY_RECOVERY_FREEZE_DIVERGENCE")
        return recovered

    freeze_identity = _record_identity(freeze.freeze_id, freeze)
    request_identity = _record_identity(freeze.request.request_id, freeze.request)
    items: list[GymToraxArtifactPublicationItem] = []
    source_reset_attempted = False

    def observe_source_reset() -> None:
        nonlocal source_reset_attempted
        source_reset_attempted = True

    try:
        episode = acquire_gym_torax_episode(
            freeze.request,
            extraction_manifest=extraction_manifest,
            field_metadata_manifest=field_metadata_manifest,
            repository_root=repository_root,
            environment_factory=environment_factory,
            runtime_inspector=runtime_inspector,
            source_reset_observer=observe_source_reset,
        )
        episode_identity = _record_identity(episode.episode_id, episode)
        canary = adjudicate_gym_torax_native_metadata_canary(
            episode=episode,
            field_metadata_manifest=field_metadata_manifest,
        )
        qualification = qualify_gym_torax_native_source(
            episode=episode,
            canary=canary,
            extraction_manifest=extraction_manifest,
            field_metadata_manifest=field_metadata_manifest,
        )
        passed = (
            canary.disposition is GymToraxNativeMetadataCanaryDisposition.PASS
            and qualification.disposition
            is GymToraxNativeSourceQualificationDisposition.QUALIFIED
        )
        reasons = (
            ()
            if passed
            else tuple(sorted({*canary.reason_codes, *qualification.reason_codes}))
        )
        terminal = GymToraxNativeCanaryRunReceipt(
            receipt_id=f"receipt.{freeze.run_id.removeprefix('run.')}.terminal",
            freeze=freeze_identity,
            request=request_identity,
            episode=episode_identity,
            metadata_canary=_record_identity(canary.receipt_id, canary),
            source_qualification=_record_identity(
                qualification.receipt_id, qualification
            ),
            disposition=(
                GymToraxOperationalDisposition.SUCCEEDED
                if passed
                else GymToraxOperationalDisposition.FAILED
            ),
            reason_codes=reasons,
            source_reset_attempted=episode.source_reset_attempted,
            scientific_values_decoded=False,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            grants_source_assessment_freeze_authority=False,
            grants_source_assessment_execution_authority=False,
        )
        for logical_id, path, record, parents in (
            (
                _episode_logical_id(episode.episode_id),
                _episode_path(freeze.relative_root, episode.episode_id),
                episode,
                (freeze_identity,),
            ),
            (
                f"artifact.{canary.receipt_id}",
                f"{freeze.relative_root}/metadata-canary-receipt.json",
                canary,
                tuple(
                    sorted(
                        (freeze_identity, episode_identity),
                        key=lambda value: value.object_id,
                    )
                ),
            ),
            (
                f"artifact.{qualification.receipt_id}",
                f"{freeze.relative_root}/source-qualification.json",
                qualification,
                (
                    _record_identity(canary.receipt_id, canary),
                    episode_identity,
                    freeze_identity,
                ),
            ),
        ):
            items.append(
                GymToraxArtifactPublicationItem(
                    logical_artifact_id=logical_id,
                    relative_path=path,
                    record=record,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    lineage_parents=tuple(
                        sorted(
                            (
                                _lineage_parent(value, OutcomeAccess.OUTCOME_BLIND)
                                for value in parents
                            ),
                            key=lambda value: value.identity.object_id,
                        )
                    ),
                )
            )
    except Exception as error:
        terminal = GymToraxNativeCanaryRunReceipt(
            receipt_id=f"receipt.{freeze.run_id.removeprefix('run.')}.terminal",
            freeze=freeze_identity,
            request=request_identity,
            episode=None,
            metadata_canary=None,
            source_qualification=None,
            disposition=GymToraxOperationalDisposition.FAILED,
            reason_codes=_technical_reason(error),
            source_reset_attempted=source_reset_attempted,
            scientific_values_decoded=False,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            grants_source_assessment_freeze_authority=False,
            grants_source_assessment_execution_authority=False,
        )
    items.append(
        GymToraxArtifactPublicationItem(
            logical_artifact_id=terminal_id,
            relative_path=terminal_path,
            record=terminal,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            lineage_parents=(
                _lineage_parent(freeze_identity, OutcomeAccess.OUTCOME_BLIND),
            ),
        )
    )
    store.publish_atomic(
        publication_scope_id=f"publication.{freeze.run_id}",
        publication_scope_relative_root=freeze.relative_root,
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
        items=tuple(sorted(items, key=lambda value: value.logical_artifact_id)),
    )
    return terminal


def _validate_recovered_cell(
    *,
    freeze: GymToraxSourceAssessmentFreeze,
    cell: GymToraxSourceAssessmentCell,
    receipt: GymToraxSourceAssessmentCellAcquisitionReceipt,
    episode: GymToraxFieldMetadataNativeEpisode | None,
) -> None:
    if (
        receipt.freeze != _record_identity(freeze.freeze_id, freeze)
        or receipt.cell != _record_identity(cell.cell_id, cell)
        or receipt.request != _record_identity(cell.request.request_id, cell.request)
        or receipt.expected_episode_id != cell.episode_id
    ):
        raise RuntimeError("GYM_TORAX_SOURCE_ASSESSMENT_RECOVERY_LINEAGE_DIVERGENCE")
    if receipt.disposition is GymToraxOperationalDisposition.SUCCEEDED:
        if episode is None or receipt.episode != _record_identity(
            episode.episode_id, episode
        ):
            raise RuntimeError("GYM_TORAX_SOURCE_ASSESSMENT_RECOVERY_EPISODE_DIVERGENCE")
        if episode.request != receipt.request or episode.episode_id != cell.episode_id:
            raise RuntimeError("GYM_TORAX_SOURCE_ASSESSMENT_RECOVERY_REQUEST_DIVERGENCE")
    elif episode is not None:
        raise RuntimeError("GYM_TORAX_SOURCE_ASSESSMENT_FAILED_RECEIPT_HAS_EPISODE")


def run_gym_torax_q(
    *,
    freeze: GymToraxSourceAssessmentFreeze,
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
    episode_acquirer: Callable[[GymToraxSourceAssessmentCell], GymToraxFieldMetadataNativeEpisode]
    | None = None,
) -> tuple[GymToraxSourceAssessmentCellAcquisitionReceipt, ...]:
    """Execute or exactly recover every frozen cell, sequentially and sealed."""

    if freeze.maximum_concurrency != 1:
        raise ValueError("bounded source assessment runner only supports frozen concurrency one")
    if freeze.source_qualification != _record_identity(
        source_qualification.receipt_id,
        source_qualification,
    ):
        raise ValueError("source assessment runner source qualification differs from its freeze")
    if (
        source_qualification.disposition
        is not GymToraxNativeSourceQualificationDisposition.QUALIFIED
    ):
        raise ValueError("source assessment runner requires a qualified source")
    _require_execution_authorities(
        freeze=freeze,
        source_authority=source_authority,
        execution_authority=execution_authority,
        at_utc=at_utc,
    )
    _require_custody_authority(
        freeze=freeze,
        freeze_id=freeze.freeze_id,
        relative_root=freeze.relative_root,
        authority=custody_authority,
        store=store,
        at_utc=at_utc,
    )
    if custody_authority.authority_id not in freeze.required_operation_authority_ids:
        raise PermissionError("Gym-TORAX source assessment custody authority ID is outside the freeze")
    _require_persisted_freeze(store, freeze)
    freeze_identity = _record_identity(freeze.freeze_id, freeze)
    receipts: list[GymToraxSourceAssessmentCellAcquisitionReceipt] = []
    for cell in freeze.cells:
        receipt_id = _cell_receipt_logical_id(cell.cell_id)
        receipt_path = _cell_receipt_path(freeze.relative_root, cell.cell_id)
        episode_id = _episode_logical_id(cell.episode_id)
        episode_path = _episode_path(freeze.relative_root, cell.episode_id)
        recovered_receipt = store.load_optional(
            logical_artifact_id=receipt_id,
            relative_path=receipt_path,
            record_type=GymToraxSourceAssessmentCellAcquisitionReceipt,
            maximum_bytes=2 * 1024**2,
        )
        recovered_episode = store.load_optional(
            logical_artifact_id=episode_id,
            relative_path=episode_path,
            record_type=GymToraxFieldMetadataNativeEpisode,
            maximum_bytes=cell.request.maximum_output_bytes,
        )
        if recovered_receipt is not None:
            _validate_recovered_cell(
                freeze=freeze,
                cell=cell,
                receipt=recovered_receipt,
                episode=recovered_episode,
            )
            receipts.append(recovered_receipt)
            continue
        if recovered_episode is not None:
            raise RuntimeError("GYM_TORAX_SOURCE_ASSESSMENT_ORPHAN_EPISODE_WITHOUT_RECEIPT")
        items: tuple[GymToraxArtifactPublicationItem, ...]
        source_reset_attempted = False

        def observe_source_reset() -> None:
            nonlocal source_reset_attempted
            source_reset_attempted = True

        try:
            episode = (
                episode_acquirer(cell)
                if episode_acquirer is not None
                else acquire_gym_torax_episode(
                    cell.request,
                    extraction_manifest=extraction_manifest,
                    field_metadata_manifest=field_metadata_manifest,
                    repository_root=repository_root,
                    environment_factory=environment_factory,
                    runtime_inspector=runtime_inspector,
                    source_reset_observer=observe_source_reset,
                )
            )
            episode_identity = _record_identity(episode.episode_id, episode)
            receipt = GymToraxSourceAssessmentCellAcquisitionReceipt(
                receipt_id=f"receipt.{cell.cell_id.removeprefix('cell.')}",
                freeze=freeze_identity,
                cell=_record_identity(cell.cell_id, cell),
                request=_record_identity(cell.request.request_id, cell.request),
                expected_episode_id=cell.episode_id,
                episode=episode_identity,
                disposition=GymToraxOperationalDisposition.SUCCEEDED,
                reason_codes=(),
                attempt_count=1,
                automatic_retry_performed=False,
                source_reset_attempted=episode.source_reset_attempted,
                scientific_values_decoded=False,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            )
            _validate_recovered_cell(
                freeze=freeze,
                cell=cell,
                receipt=receipt,
                episode=episode,
            )
            items = (
                GymToraxArtifactPublicationItem(
                    logical_artifact_id=episode_id,
                    relative_path=episode_path,
                    record=episode,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.EVALUATION_SEALED,
                    lineage_parents=(
                        _lineage_parent(freeze_identity, OutcomeAccess.OUTCOME_BLIND),
                    ),
                ),
                GymToraxArtifactPublicationItem(
                    logical_artifact_id=receipt_id,
                    relative_path=receipt_path,
                    record=receipt,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.EVALUATION_SEALED,
                    lineage_parents=tuple(
                        sorted(
                            (
                                _lineage_parent(
                                    episode_identity,
                                    OutcomeAccess.EVALUATION_SEALED,
                                ),
                                _lineage_parent(
                                    freeze_identity,
                                    OutcomeAccess.OUTCOME_BLIND,
                                ),
                            ),
                            key=lambda value: value.identity.object_id,
                        )
                    ),
                ),
            )
        except Exception as error:
            receipt = GymToraxSourceAssessmentCellAcquisitionReceipt(
                receipt_id=f"receipt.{cell.cell_id.removeprefix('cell.')}",
                freeze=freeze_identity,
                cell=_record_identity(cell.cell_id, cell),
                request=_record_identity(cell.request.request_id, cell.request),
                expected_episode_id=cell.episode_id,
                episode=None,
                disposition=GymToraxOperationalDisposition.FAILED,
                reason_codes=_technical_reason(error),
                attempt_count=1,
                automatic_retry_performed=False,
                source_reset_attempted=source_reset_attempted,
                scientific_values_decoded=False,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
            )
            items = (
                GymToraxArtifactPublicationItem(
                    logical_artifact_id=receipt_id,
                    relative_path=receipt_path,
                    record=receipt,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.EVALUATION_SEALED,
                    lineage_parents=(
                        _lineage_parent(freeze_identity, OutcomeAccess.OUTCOME_BLIND),
                    ),
                ),
            )
        store.publish_atomic(
            publication_scope_id=f"publication.{freeze.run_id}",
            publication_scope_relative_root=freeze.relative_root,
            implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
            items=tuple(sorted(items, key=lambda value: value.logical_artifact_id)),
        )
        receipts.append(receipt)
    return tuple(sorted(receipts, key=lambda value: value.cell.object_id))


def finalize_gym_torax_q(
    *,
    freeze: GymToraxSourceAssessmentFreeze,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> GymToraxSourceAssessmentTerminalAcquisitionReceipt:
    """Publish the terminal roster receipt only after all 96 cells are accounted."""

    _require_custody_authority(
        freeze=freeze,
        freeze_id=freeze.freeze_id,
        relative_root=freeze.relative_root,
        authority=custody_authority,
        store=store,
        at_utc=at_utc,
    )
    if custody_authority.authority_id not in freeze.required_operation_authority_ids:
        raise PermissionError("Gym-TORAX source assessment custody authority ID is outside the freeze")
    _require_persisted_freeze(store, freeze)
    terminal_id = _terminal_logical_id(freeze.run_id)
    terminal_path = f"{freeze.relative_root}/run-receipt.json"
    recovered = store.load_optional(
        logical_artifact_id=terminal_id,
        relative_path=terminal_path,
        record_type=GymToraxSourceAssessmentTerminalAcquisitionReceipt,
        maximum_bytes=4 * 1024**2,
    )
    if recovered is not None:
        if recovered.freeze != _record_identity(freeze.freeze_id, freeze):
            raise RuntimeError("GYM_TORAX_SOURCE_ASSESSMENT_TERMINAL_RECOVERY_FREEZE_DIVERGENCE")
        return recovered

    receipts: list[GymToraxSourceAssessmentCellAcquisitionReceipt] = []
    episodes: list[ObjectIdentity] = []
    missing: list[str] = []
    for cell in freeze.cells:
        receipt = store.load_optional(
            logical_artifact_id=_cell_receipt_logical_id(cell.cell_id),
            relative_path=_cell_receipt_path(freeze.relative_root, cell.cell_id),
            record_type=GymToraxSourceAssessmentCellAcquisitionReceipt,
            maximum_bytes=2 * 1024**2,
        )
        episode = store.load_optional(
            logical_artifact_id=_episode_logical_id(cell.episode_id),
            relative_path=_episode_path(freeze.relative_root, cell.episode_id),
            record_type=GymToraxFieldMetadataNativeEpisode,
            maximum_bytes=cell.request.maximum_output_bytes,
        )
        if receipt is None:
            missing.append(cell.cell_id)
            if episode is not None:
                raise RuntimeError("GYM_TORAX_SOURCE_ASSESSMENT_ORPHAN_EPISODE_WITHOUT_RECEIPT")
            continue
        _validate_recovered_cell(
            freeze=freeze, cell=cell, receipt=receipt, episode=episode
        )
        receipts.append(receipt)
        if episode is not None:
            episodes.append(_record_identity(episode.episode_id, episode))
    if missing:
        raise GymToraxSourceAssessmentIncompleteError(missing)
    failed_count = sum(
        value.disposition is GymToraxOperationalDisposition.FAILED for value in receipts
    )
    terminal = GymToraxSourceAssessmentTerminalAcquisitionReceipt(
        receipt_id=f"receipt.{freeze.run_id.removeprefix('run.')}.terminal-acquisition",
        freeze=_record_identity(freeze.freeze_id, freeze),
        cell_receipts=tuple(
            sorted(
                (_record_identity(value.receipt_id, value) for value in receipts),
                key=lambda value: value.object_id,
            )
        ),
        episodes=tuple(sorted(episodes, key=lambda value: value.object_id)),
        accounted_cell_count=len(receipts),
        completed_episode_count=len(episodes),
        failed_cell_count=failed_count,
        disposition=(
            GymToraxSourceAssessmentTerminalAcquisitionDisposition.COMPLETE
            if failed_count == 0
            else GymToraxSourceAssessmentTerminalAcquisitionDisposition.INCOMPLETE
        ),
        reason_codes=() if failed_count == 0 else ("GYM_TORAX_SOURCE_ASSESSMENT_TECHNICAL_CELL_FAILURE",),
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
                logical_artifact_id=terminal_id,
                relative_path=terminal_path,
                record=terminal,
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
                lineage_parents=tuple(
                    sorted(
                        (
                            _lineage_parent(
                                _record_identity(value.receipt_id, value),
                                OutcomeAccess.EVALUATION_SEALED,
                            )
                            for value in receipts
                        ),
                        key=lambda value: value.identity.object_id,
                    )
                ),
            ),
        ),
    )
    return terminal


__all__ = [
    'GymToraxArtifactPublicationItem',
    'GymToraxNativeCanaryRunReceipt',
    'GymToraxOperationalDisposition',
    'GymToraxSourceAssessmentCellAcquisitionReceipt',
    'GymToraxSourceAssessmentIncompleteError',
    'GymToraxSourceAssessmentTerminalAcquisitionDisposition',
    'GymToraxSourceAssessmentTerminalAcquisitionReceipt',
    'GymToraxBoundedArtifactStore',
    'execute_gym_torax_native_metadata_canary',
    'finalize_gym_torax_q',
    'publish_gym_torax_q_freeze',
    'publish_gym_torax_native_canary_freeze',
    'run_gym_torax_q',
]

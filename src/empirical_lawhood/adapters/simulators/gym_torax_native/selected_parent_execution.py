"Authority-guarded acquisition for the selected Gym-TORAX finite-action recurrence measurement-through-law-qualification parent.\n\nThe selected branch is outcome-visible because source assessment has been adjudicated.  The\nnew measurement-through-law-qualification requests and episodes nevertheless remain evaluation-sealed until\nan explicit reveal.  Operational receipts inherit the revealed branch\nselection; scientific episode artifacts inherit their exact sealed requests.\n"

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
import re
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

from .field_metadata_contracts import GymToraxFieldMetadataEpisodeRequest, GymToraxFieldMetadataNativeEpisode
from .extraction_manifest import GymToraxBoundedExtractionManifest
from .field_metadata import GymToraxFieldMetadataManifest
from .selected_parent_protocol import GymToraxSelectedParentCell, GymToraxSelectedParentFreeze, materialize_gym_torax_response_experiment_request
from .source_assessment_execution import GymToraxArtifactPublicationItem, GymToraxOperationalDisposition, GymToraxBoundedArtifactStore
from .runtime import GymToraxEnvironmentFactory, GymToraxOutputBoundError, GymToraxRuntimePreflightError, GymToraxRuntimeInspector, acquire_gym_torax_episode, inspect_gym_torax_runtime
from .source_qualification import GymToraxNativeSourceQualificationDisposition, GymToraxNativeSourceQualificationReceipt


_EXECUTION_GRANTEE_ID = "operator.execution-service"


class GymToraxSelectedParentTerminalAcquisitionDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    INCOMPLETE = "INCOMPLETE"


@dataclass(frozen=True, slots=True)
class GymToraxSelectedParentCellAcquisitionReceipt(CanonicalRecord):
    "One immutable acquisition attempt for one frozen response qualification cell."

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-selected-parent-cell-acquisition-receipt'
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
        if self.freeze.object_schema != GymToraxSelectedParentFreeze.SCHEMA:
            raise ValueError("Response qualification cell receipt binds another freeze schema")
        if self.cell.object_schema != GymToraxSelectedParentCell.SCHEMA:
            raise ValueError("Response qualification cell receipt binds another cell schema")
        if self.request.object_schema != GymToraxFieldMetadataEpisodeRequest.SCHEMA:
            raise ValueError("Response qualification cell receipt binds another request schema")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.attempt_count != 1 or self.automatic_retry_performed:
            raise ValueError(
                "Response qualification permits exactly one attempt and no automatic retry"
            )
        if self.disposition is GymToraxOperationalDisposition.SUCCEEDED:
            if self.episode is None or self.reason_codes:
                raise ValueError(
                    "successful response qualification acquisition requires one exact episode"
                )
            if (
                self.episode.object_schema != GymToraxFieldMetadataNativeEpisode.SCHEMA
                or self.episode.object_id != self.expected_episode_id
            ):
                raise ValueError("Response qualification receipt episode identity differs")
        elif self.episode is not None or not self.reason_codes:
            raise ValueError("failed response qualification acquisition cannot claim an episode")
        if self.scientific_values_decoded:
            raise ValueError("Response qualification acquisition cannot decode scientific values")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.MEASUREMENT
        ):
            raise ValueError("Response qualification receipt changes its selected-parent evidence lane")


@dataclass(frozen=True, slots=True)
class GymToraxSelectedParentTerminalAcquisitionReceipt(CanonicalRecord):
    """Exact terminal accounting for all 576 selected-parent cells."""

    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-selected-parent-terminal-acquisition-receipt'
    )

    receipt_id: str
    freeze: ObjectIdentity
    cell_receipts: tuple[ObjectIdentity, ...]
    episodes: tuple[ObjectIdentity, ...]
    accounted_cell_count: int
    completed_episode_count: int
    failed_cell_count: int
    disposition: GymToraxSelectedParentTerminalAcquisitionDisposition
    reason_codes: tuple[str, ...]
    scientific_values_decoded: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if self.freeze.object_schema != GymToraxSelectedParentFreeze.SCHEMA:
            raise ValueError("Response qualification terminal binds another freeze schema")
        require_sorted_unique_ids(
            self.cell_receipts,
            attribute="object_id",
            field_name="cell_receipts",
        )
        require_sorted_unique_ids(
            self.episodes, attribute="object_id", field_name="episodes"
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.accounted_cell_count != 576 or len(self.cell_receipts) != 576:
            raise ValueError("Response qualification terminal requires all 576 cell receipts")
        if self.completed_episode_count != len(self.episodes):
            raise ValueError("Response qualification completed count differs from episode identities")
        if self.completed_episode_count + self.failed_cell_count != 576:
            raise ValueError("Response qualification terminal counts do not cover the frozen roster")
        if self.disposition is GymToraxSelectedParentTerminalAcquisitionDisposition.COMPLETE:
            if (
                self.completed_episode_count != 576
                or self.failed_cell_count
                or self.reason_codes
            ):
                raise ValueError("complete response qualification terminal requires all episodes")
        elif self.failed_cell_count <= 0 or not self.reason_codes:
            raise ValueError("incomplete response qualification terminal requires exact failed cells")
        if self.scientific_values_decoded:
            raise ValueError("Response qualification terminal cannot decode scientific values")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.MEASUREMENT
        ):
            raise ValueError(
                "Response qualification terminal changes its selected-parent evidence lane"
            )


class GymToraxSelectedParentIncompleteError(RuntimeError):
    def __init__(self, missing_cell_ids: Sequence[str]) -> None:
        self.missing_cell_ids = tuple(sorted(set(missing_cell_ids)))
        super().__init__(
            f"GYM_TORAX_MEASUREMENT_LOCAL_LAW_CELL_RECEIPTS_MISSING:{len(self.missing_cell_ids)}"
        )


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


def _freeze_logical_id(freeze: GymToraxSelectedParentFreeze) -> str:
    return f"artifact.{freeze.run_id}.freeze"


def _request_logical_id(request_id: str) -> str:
    return f"artifact.{request_id}"


def _episode_logical_id(episode_id: str) -> str:
    return f"artifact.{episode_id}"


def _receipt_logical_id(cell_id: str) -> str:
    return f"artifact.receipt.{cell_id}"


def _request_path(freeze: GymToraxSelectedParentFreeze, request_id: str) -> str:
    return f"{freeze.relative_root}/requests/{request_id}.json"


def _episode_path(freeze: GymToraxSelectedParentFreeze, episode_id: str) -> str:
    return f"{freeze.relative_root}/episodes/{episode_id}.json"


def _receipt_path(freeze: GymToraxSelectedParentFreeze, cell_id: str) -> str:
    return f"{freeze.relative_root}/receipts/{cell_id}.json"


def _require_custody(
    freeze: GymToraxSelectedParentFreeze,
    authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    *,
    at_utc: str,
) -> None:
    require_study_authority(
        authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=_identity(freeze.freeze_id, freeze),
        prerequisite_authority=None,
        grantee_id=_EXECUTION_GRANTEE_ID,
        storage_root_id=store.storage_root_id,
        relative_root=freeze.relative_root,
        at_utc=at_utc,
    )
    if authority.authority_id not in freeze.required_operation_authority_ids:
        raise PermissionError("Response qualification custody authority ID is outside the freeze")


def _require_execution(
    freeze: GymToraxSelectedParentFreeze,
    *,
    source_authority: StudyOperationAuthority,
    execution_authority: StudyOperationAuthority,
    at_utc: str,
) -> None:
    subject = _identity(freeze.freeze_id, freeze)
    require_study_authority(
        source_authority,
        kind=StudyAuthorityKind.SOURCE_ACQUISITION,
        subject=subject,
        prerequisite_authority=None,
        grantee_id=_EXECUTION_GRANTEE_ID,
        at_utc=at_utc,
    )
    require_study_authority(
        execution_authority,
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=subject,
        prerequisite_authority=freeze.scientific_approval,
        grantee_id=_EXECUTION_GRANTEE_ID,
        at_utc=at_utc,
    )
    if execution_authority.allows_actuation:
        raise PermissionError("Response qualification simulator authority cannot permit actuation")
    if execution_authority.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
        raise PermissionError(
            "Response qualification execution authority must keep new outcomes sealed"
        )
    if {
        source_authority.authority_id,
        execution_authority.authority_id,
    } - set(freeze.required_operation_authority_ids):
        raise PermissionError("Response qualification execution authority ID is outside the freeze")


def _require_persisted_freeze(
    store: GymToraxBoundedArtifactStore,
    freeze: GymToraxSelectedParentFreeze,
) -> None:
    observed = store.load_optional(
        logical_artifact_id=_freeze_logical_id(freeze),
        relative_path=f"{freeze.relative_root}/freeze.json",
        record_type=GymToraxSelectedParentFreeze,
        maximum_bytes=32 * 1024**2,
    )
    if observed != freeze:
        raise RuntimeError("GYM_TORAX_MEASUREMENT_LOCAL_LAW_FREEZE_NOT_PERSISTED_EXACTLY")


def _technical_reason(error: Exception) -> tuple[str, ...]:
    if isinstance(error, GymToraxRuntimePreflightError):
        return tuple(f"GYM_TORAX_MEASUREMENT_LOCAL_LAW_{value}" for value in error.reason_codes)
    if isinstance(error, GymToraxOutputBoundError):
        return ("GYM_TORAX_MEASUREMENT_LOCAL_LAW_OUTPUT_BOUND_EXCEEDED",)
    token = re.sub(r"[^A-Z0-9]+", "_", type(error).__name__.upper()).strip("_")
    return (f"GYM_TORAX_MEASUREMENT_LOCAL_LAW_TECHNICAL_{token or 'ERROR'}",)


def publish_gym_torax_response_experiment_freeze(
    *,
    freeze: GymToraxSelectedParentFreeze,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> ObjectIdentity:
    "Publish the exact selected-parent freeze with revealed source assessment lineage."

    _require_custody(freeze, custody_authority, store, at_utc=at_utc)
    parents = (
        _parent(
            freeze.prospective_protocol_parent,
            visibility=VisibilityCeiling.PROSPECTIVE,
            access=OutcomeAccess.OUTCOME_BLIND,
        ),
        _parent(
            freeze.selected_branch,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            access=OutcomeAccess.EVALUATION_REVEALED,
        ),
        _parent(
            freeze.source_campaign_roster,
            visibility=VisibilityCeiling.PROSPECTIVE,
            access=OutcomeAccess.OUTCOME_BLIND,
        ),
    )
    return store.publish_atomic(
        publication_scope_id=f"publication.{freeze.run_id}",
        publication_scope_relative_root=freeze.relative_root,
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id=_freeze_logical_id(freeze),
                relative_path=f"{freeze.relative_root}/freeze.json",
                record=freeze,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                lineage_parents=tuple(
                    sorted(parents, key=lambda value: value.identity.object_id)
                ),
            ),
        ),
    )[0]


def _validate_cell_products(
    *,
    freeze: GymToraxSelectedParentFreeze,
    cell: GymToraxSelectedParentCell,
    request: GymToraxFieldMetadataEpisodeRequest | None,
    receipt: GymToraxSelectedParentCellAcquisitionReceipt,
    episode: GymToraxFieldMetadataNativeEpisode | None,
) -> None:
    expected_request = materialize_gym_torax_response_experiment_request(freeze, cell)
    if request != expected_request:
        raise RuntimeError("GYM_TORAX_MEASUREMENT_LOCAL_LAW_RECOVERY_REQUEST_DIVERGENCE")
    if (
        receipt.freeze != _identity(freeze.freeze_id, freeze)
        or receipt.cell != _identity(cell.cell_id, cell)
        or receipt.request != _identity(expected_request.request_id, expected_request)
        or receipt.expected_episode_id != cell.episode_id
    ):
        raise RuntimeError("GYM_TORAX_MEASUREMENT_LOCAL_LAW_RECOVERY_LINEAGE_DIVERGENCE")
    if receipt.disposition is GymToraxOperationalDisposition.SUCCEEDED:
        if episode is None or receipt.episode != _identity(episode.episode_id, episode):
            raise RuntimeError("GYM_TORAX_MEASUREMENT_LOCAL_LAW_RECOVERY_EPISODE_DIVERGENCE")
        if (
            episode.episode_id != cell.episode_id
            or episode.request != receipt.request
            or episode.preparation
            != _identity(
                expected_request.preparation.preparation_id,
                expected_request.preparation,
            )
            or episode.numerical_member
            != _identity(
                expected_request.numerical_member.member_id,
                expected_request.numerical_member,
            )
            or episode.action_word
            != _identity(
                expected_request.schedule.action_word.word_id,
                expected_request.schedule.action_word,
            )
            or episode.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or episode.evidence_ceiling is not EvidenceCeiling.MEASUREMENT
        ):
            raise RuntimeError("GYM_TORAX_MEASUREMENT_LOCAL_LAW_RECOVERY_EPISODE_LINEAGE_DIVERGENCE")
    elif episode is not None:
        raise RuntimeError("GYM_TORAX_MEASUREMENT_LOCAL_LAW_FAILED_RECEIPT_HAS_EPISODE")


def run_gym_torax_response_experiment(
    *,
    freeze: GymToraxSelectedParentFreeze,
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
) -> tuple[GymToraxSelectedParentCellAcquisitionReceipt, ...]:
    """Execute or exactly recover all 576 cells, sequentially and sealed."""

    if freeze.maximum_concurrency != 1:
        raise ValueError("Response qualification runner only supports frozen concurrency one")
    if freeze.source_qualification != _identity(
        source_qualification.receipt_id,
        source_qualification,
    ):
        raise ValueError("Response qualification source qualification differs from its freeze")
    if (
        source_qualification.disposition
        is not GymToraxNativeSourceQualificationDisposition.QUALIFIED
    ):
        raise ValueError("Response qualification requires a qualified source")
    _require_execution(
        freeze,
        source_authority=source_authority,
        execution_authority=execution_authority,
        at_utc=at_utc,
    )
    _require_custody(freeze, custody_authority, store, at_utc=at_utc)
    _require_persisted_freeze(store, freeze)
    freeze_identity = _identity(freeze.freeze_id, freeze)
    source_qualification_identity = _identity(
        source_qualification.receipt_id,
        source_qualification,
    )
    receipts: list[GymToraxSelectedParentCellAcquisitionReceipt] = []
    for cell in freeze.cells:
        request = materialize_gym_torax_response_experiment_request(freeze, cell)
        request_id = _request_logical_id(request.request_id)
        request_path = _request_path(freeze, request.request_id)
        receipt_id = _receipt_logical_id(cell.cell_id)
        receipt_path = _receipt_path(freeze, cell.cell_id)
        episode_id = _episode_logical_id(cell.episode_id)
        episode_path = _episode_path(freeze, cell.episode_id)
        recovered_request = store.load_optional(
            logical_artifact_id=request_id,
            relative_path=request_path,
            record_type=GymToraxFieldMetadataEpisodeRequest,
            maximum_bytes=2 * 1024**2,
        )
        recovered_receipt = store.load_optional(
            logical_artifact_id=receipt_id,
            relative_path=receipt_path,
            record_type=GymToraxSelectedParentCellAcquisitionReceipt,
            maximum_bytes=2 * 1024**2,
        )
        recovered_episode = store.load_optional(
            logical_artifact_id=episode_id,
            relative_path=episode_path,
            record_type=GymToraxFieldMetadataNativeEpisode,
            maximum_bytes=request.maximum_output_bytes,
        )
        if recovered_receipt is not None:
            _validate_cell_products(
                freeze=freeze,
                cell=cell,
                request=recovered_request,
                receipt=recovered_receipt,
                episode=recovered_episode,
            )
            receipts.append(recovered_receipt)
            continue
        if recovered_episode is not None and recovered_request is None:
            raise RuntimeError("GYM_TORAX_MEASUREMENT_LOCAL_LAW_ORPHAN_EPISODE_WITHOUT_REQUEST")

        request_was_persisted = recovered_request is not None
        if recovered_request is None:
            store.publish_atomic(
                publication_scope_id=f"publication.{freeze.run_id}",
                publication_scope_relative_root=freeze.relative_root,
                implementation_sha256=(
                    freeze.implementation_source_closure.implementation_sha256
                ),
                items=(
                    GymToraxArtifactPublicationItem(
                        logical_artifact_id=request_id,
                        relative_path=request_path,
                        record=request,
                        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                        outcome_access=OutcomeAccess.EVALUATION_SEALED,
                        lineage_parents=tuple(
                            sorted(
                                (
                                    _parent(
                                        freeze.prospective_protocol_parent,
                                        visibility=VisibilityCeiling.PROSPECTIVE,
                                        access=OutcomeAccess.OUTCOME_BLIND,
                                    ),
                                    _parent(
                                        freeze.source_campaign_roster,
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
            recovered_request = request

        source_reset_attempted = False

        def observe_source_reset() -> None:
            nonlocal source_reset_attempted
            source_reset_attempted = True

        if recovered_episode is not None:
            episode = recovered_episode
            disposition = GymToraxOperationalDisposition.SUCCEEDED
            reason_codes: tuple[str, ...] = ()
            source_reset_attempted = episode.source_reset_attempted
        elif request_was_persisted:
            episode = None
            disposition = GymToraxOperationalDisposition.FAILED
            reason_codes = ("GYM_TORAX_MEASUREMENT_LOCAL_LAW_INTERRUPTED_ATTEMPT_WITHOUT_EPISODE",)
        else:
            try:
                episode = (
                    episode_acquirer(request)
                    if episode_acquirer is not None
                    else acquire_gym_torax_episode(
                        request,
                        extraction_manifest=extraction_manifest,
                        field_metadata_manifest=field_metadata_manifest,
                        repository_root=repository_root,
                        environment_factory=environment_factory,
                        runtime_inspector=runtime_inspector,
                        source_reset_observer=observe_source_reset,
                    )
                )
                episode_identity = _identity(episode.episode_id, episode)
                provisional_receipt = GymToraxSelectedParentCellAcquisitionReceipt(
                    receipt_id=f"receipt.{cell.cell_id.removeprefix('cell.')}",
                    freeze=freeze_identity,
                    cell=_identity(cell.cell_id, cell),
                    request=_identity(request.request_id, request),
                    expected_episode_id=cell.episode_id,
                    episode=episode_identity,
                    disposition=GymToraxOperationalDisposition.SUCCEEDED,
                    reason_codes=(),
                    attempt_count=1,
                    automatic_retry_performed=False,
                    source_reset_attempted=episode.source_reset_attempted,
                    scientific_values_decoded=False,
                    outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                    visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                    evidence_ceiling=EvidenceCeiling.MEASUREMENT,
                )
                _validate_cell_products(
                    freeze=freeze,
                    cell=cell,
                    request=request,
                    receipt=provisional_receipt,
                    episode=episode,
                )
                store.publish_atomic(
                    publication_scope_id=f"publication.{freeze.run_id}",
                    publication_scope_relative_root=freeze.relative_root,
                    implementation_sha256=(
                        freeze.implementation_source_closure.implementation_sha256
                    ),
                    items=(
                        GymToraxArtifactPublicationItem(
                            logical_artifact_id=episode_id,
                            relative_path=episode_path,
                            record=episode,
                            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                            outcome_access=OutcomeAccess.EVALUATION_SEALED,
                            lineage_parents=tuple(
                                sorted(
                                    (
                                        _parent(
                                            _identity(request.request_id, request),
                                            visibility=VisibilityCeiling.PROSPECTIVE,
                                            access=OutcomeAccess.EVALUATION_SEALED,
                                        ),
                                        _parent(
                                            source_qualification_identity,
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
                disposition = GymToraxOperationalDisposition.SUCCEEDED
                reason_codes = ()
            except Exception as error:
                episode = None
                disposition = GymToraxOperationalDisposition.FAILED
                reason_codes = _technical_reason(error)

        if episode is not None:
            receipt_episode_identity: ObjectIdentity | None = _identity(
                episode.episode_id,
                episode,
            )
        else:
            receipt_episode_identity = None
        receipt = GymToraxSelectedParentCellAcquisitionReceipt(
            receipt_id=f"receipt.{cell.cell_id.removeprefix('cell.')}",
            freeze=freeze_identity,
            cell=_identity(cell.cell_id, cell),
            request=_identity(request.request_id, request),
            expected_episode_id=cell.episode_id,
            episode=receipt_episode_identity,
            disposition=disposition,
            reason_codes=reason_codes,
            attempt_count=1,
            automatic_retry_performed=False,
            source_reset_attempted=source_reset_attempted,
            scientific_values_decoded=False,
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            evidence_ceiling=EvidenceCeiling.MEASUREMENT,
        )
        _validate_cell_products(
            freeze=freeze,
            cell=cell,
            request=request,
            receipt=receipt,
            episode=episode,
        )
        receipt_parents = [
            _parent(
                freeze_identity,
                visibility=VisibilityCeiling.OUTCOME_VISIBLE,
                access=OutcomeAccess.EVALUATION_REVEALED,
            ),
            _parent(
                _identity(request.request_id, request),
                visibility=VisibilityCeiling.PROSPECTIVE,
                access=OutcomeAccess.EVALUATION_SEALED,
            ),
        ]
        if episode is not None:
            receipt_parents.append(
                _parent(
                    _identity(episode.episode_id, episode),
                    visibility=VisibilityCeiling.PROSPECTIVE,
                    access=OutcomeAccess.EVALUATION_SEALED,
                )
            )
        store.publish_atomic(
            publication_scope_id=f"publication.{freeze.run_id}",
            publication_scope_relative_root=freeze.relative_root,
            implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
            items=(
                GymToraxArtifactPublicationItem(
                    logical_artifact_id=receipt_id,
                    relative_path=receipt_path,
                    record=receipt,
                    visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                    outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                    lineage_parents=tuple(
                        sorted(
                            receipt_parents, key=lambda value: value.identity.object_id
                        )
                    ),
                ),
            ),
        )
        receipts.append(receipt)
    return tuple(sorted(receipts, key=lambda value: value.cell.object_id))


def finalize_gym_torax_response_experiment(
    *,
    freeze: GymToraxSelectedParentFreeze,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> GymToraxSelectedParentTerminalAcquisitionReceipt:
    """Publish terminal accounting only after every frozen cell has a receipt."""

    _require_custody(freeze, custody_authority, store, at_utc=at_utc)
    _require_persisted_freeze(store, freeze)
    logical_id = f"artifact.{freeze.run_id}.run-receipt"
    path = f"{freeze.relative_root}/run-receipt.json"
    recovered = store.load_optional(
        logical_artifact_id=logical_id,
        relative_path=path,
        record_type=GymToraxSelectedParentTerminalAcquisitionReceipt,
        maximum_bytes=16 * 1024**2,
    )
    if recovered is not None:
        if recovered.freeze != _identity(freeze.freeze_id, freeze):
            raise RuntimeError("GYM_TORAX_MEASUREMENT_LOCAL_LAW_TERMINAL_RECOVERY_FREEZE_DIVERGENCE")
        return recovered

    receipts: list[GymToraxSelectedParentCellAcquisitionReceipt] = []
    episodes: list[ObjectIdentity] = []
    missing: list[str] = []
    for cell in freeze.cells:
        request = materialize_gym_torax_response_experiment_request(freeze, cell)
        recovered_request = store.load_optional(
            logical_artifact_id=_request_logical_id(request.request_id),
            relative_path=_request_path(freeze, request.request_id),
            record_type=GymToraxFieldMetadataEpisodeRequest,
            maximum_bytes=2 * 1024**2,
        )
        receipt = store.load_optional(
            logical_artifact_id=_receipt_logical_id(cell.cell_id),
            relative_path=_receipt_path(freeze, cell.cell_id),
            record_type=GymToraxSelectedParentCellAcquisitionReceipt,
            maximum_bytes=2 * 1024**2,
        )
        episode = store.load_optional(
            logical_artifact_id=_episode_logical_id(cell.episode_id),
            relative_path=_episode_path(freeze, cell.episode_id),
            record_type=GymToraxFieldMetadataNativeEpisode,
            maximum_bytes=request.maximum_output_bytes,
        )
        if receipt is None:
            missing.append(cell.cell_id)
            if recovered_request is not None or episode is not None:
                raise RuntimeError("GYM_TORAX_MEASUREMENT_LOCAL_LAW_ORPHAN_ARTIFACT_WITHOUT_RECEIPT")
            continue
        _validate_cell_products(
            freeze=freeze,
            cell=cell,
            request=recovered_request,
            receipt=receipt,
            episode=episode,
        )
        receipts.append(receipt)
        if episode is not None:
            episodes.append(_identity(episode.episode_id, episode))
    if missing:
        raise GymToraxSelectedParentIncompleteError(missing)
    failed = sum(
        value.disposition is GymToraxOperationalDisposition.FAILED for value in receipts
    )
    terminal = GymToraxSelectedParentTerminalAcquisitionReceipt(
        receipt_id=f"receipt.{freeze.run_id.removeprefix('run.')}.terminal-acquisition",
        freeze=_identity(freeze.freeze_id, freeze),
        cell_receipts=tuple(
            sorted(
                (_identity(value.receipt_id, value) for value in receipts),
                key=lambda value: value.object_id,
            )
        ),
        episodes=tuple(sorted(episodes, key=lambda value: value.object_id)),
        accounted_cell_count=len(receipts),
        completed_episode_count=len(episodes),
        failed_cell_count=failed,
        disposition=(
            GymToraxSelectedParentTerminalAcquisitionDisposition.COMPLETE
            if failed == 0
            else GymToraxSelectedParentTerminalAcquisitionDisposition.INCOMPLETE
        ),
        reason_codes=() if failed == 0 else ("GYM_TORAX_MEASUREMENT_LOCAL_LAW_TECHNICAL_CELL_FAILURE",),
        scientific_values_decoded=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.MEASUREMENT,
    )
    store.publish_atomic(
        publication_scope_id=f"publication.{freeze.run_id}",
        publication_scope_relative_root=freeze.relative_root,
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id=logical_id,
                relative_path=path,
                record=terminal,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                lineage_parents=tuple(
                    sorted(
                        (
                            _parent(
                                _identity(value.receipt_id, value),
                                visibility=VisibilityCeiling.OUTCOME_VISIBLE,
                                access=OutcomeAccess.EVALUATION_REVEALED,
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
    'GymToraxSelectedParentCellAcquisitionReceipt',
    'GymToraxSelectedParentIncompleteError',
    'GymToraxSelectedParentTerminalAcquisitionDisposition',
    'GymToraxSelectedParentTerminalAcquisitionReceipt',
    'finalize_gym_torax_response_experiment',
    'publish_gym_torax_response_experiment_freeze',
    'run_gym_torax_response_experiment',
]

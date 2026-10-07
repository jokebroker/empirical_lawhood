"Streaming, evaluation-only recovery for the complete sealed finite-action recurrence recovery cohort."

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
import gc
import hashlib
from typing import ClassVar

import numpy as np

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
    require_sorted_unique_strings,
    validate_relative_locator,
    validate_stable_id,
)
from empirical_lawhood.planning.finite_action_recurrence import FiniteActionPreactionDisposition, FiniteActionRecurrencePredicateResult, FiniteActionRecurrenceProspectiveBundle, FiniteActionRecurrenceResponseSample, FiniteActionRecurrenceRevealIntegrity, FiniteActionRecurrenceRevealedBundle, FiniteActionRecurrenceRevealedEpisode, FiniteActionRecurrenceRouteRole, FiniteActionRecurrenceSealedEpisode
from empirical_lawhood.planning.study_issue import ImplementationSourceClosure, StudyAuthorityKind, StudyOperationAuthority, SourceClosureKind, require_study_authority
from empirical_lawhood.runtime.artifacts import ArtifactManifest
from empirical_lawhood.runtime.finite_action_recurrence import FiniteActionRecurrenceResult, MatchedFiniteActionHoldRecurrenceEvaluator

from .field_metadata_contracts import GymToraxFieldMetadataNativeEpisode
from .extraction_manifest import GymToraxBoundedExtractionManifest
from .field_metadata import GymToraxFieldMetadataManifest
from .retained_inputs import GymToraxEvaluationParents

from .finite_action_recurrence_recovery import GymToraxFiniteActionRecurrenceRecoveryEvaluationBundle, GymToraxFiniteActionRecurrenceRecoveryFreeze, GymToraxFiniteActionRecurrenceRecoveryTerminalReceipt, materialize_gym_torax_finite_action_recurrence_recovery_request
from .finite_action_recurrence_science import GymToraxFiniteActionRecurrenceQualificationBundle, _artifact, _content_artifact, _episode_identity, _observed, _prefix_sha, _validate_hfr_episode


GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_FREEZE_ID = (
    'freeze.tokamak-control.finite-action-evaluation-recovery'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_RUN_ID = (
    'run.tokamak-control.finite-action-evaluation-recovery'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_RELATIVE_ROOT = (
    'tokamak-control/publication/runs/run.tokamak-control.finite-action-evaluation-recovery'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_EVALUATOR_ID = (
    'evaluator.tokamak-control.finite-action-evaluation-recovery'
)
GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_AUTHORITY_IDS = tuple(
    sorted(
        f'authority.tokamak-control.finite-action-evaluation-recovery.{suffix}'
        for suffix in (
            "custody-publication",
            "experiment-execution",
            "outcome-reveal",
        )
    )
)


def _identity(identifier: str, record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(identifier, record)


def _is_complete_valid(episode: GymToraxFieldMetadataNativeEpisode) -> bool:
    return (
        episode.state_clocks == tuple(range(121))
        and not episode.missing_required_state_clocks
        and episode.last_valid_state_clock == 120
        and episode.source_disposition.value == "AVAILABLE"
        and episode.delivery_disposition.value == "COMPLETE"
        and episode.numerical_disposition.value == "VALID"
        and episode.observation_disposition.value == "COMPLETE"
        and not episode.termination
        and not episode.truncation
    )


def _q_values(episode: GymToraxFieldMetadataNativeEpisode) -> np.ndarray:
    blocks = tuple(
        value
        for value in episode.blocks
        if value.category == "source-scalar" and value.native_field_id == "Q_fusion"
    )
    if len(blocks) != 1:
        raise ValueError("Finite-action recurrence streaming recovery complete episode lacks Q_fusion")
    return np.asarray(blocks[0].array()[:, 0], dtype=np.float64)


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceEvaluationRecoveryFreeze(CanonicalRecord):
    expected_parents: GymToraxEvaluationParents
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-evaluation-recovery-freeze'
    )
    VERSION: ClassVar[str] = '1.0.0'

    freeze_id: str
    run_id: str
    relative_root: str
    acquisition_freeze: ObjectIdentity
    acquisition_terminal: ObjectIdentity
    parent_qualification: ObjectIdentity
    extraction_manifest: ObjectIdentity
    implementation_source_closure: ImplementationSourceClosure
    scientific_approval: ObjectIdentity
    episode_ids: tuple[str, ...]
    defect_id: str
    maximum_episode_resident_count: int
    required_operation_authority_ids: tuple[str, ...]
    simulator_execution_permitted: bool
    acquisition_bytes_mutable: bool
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        for name in ("freeze_id", "run_id", "defect_id"):
            validate_stable_id(getattr(self, name), field_name=name)
        validate_relative_locator(self.relative_root)
        require_sorted_unique_strings(
            self.episode_ids, field_name="episode_ids", allow_empty=False
        )
        require_sorted_unique_strings(
            self.required_operation_authority_ids,
            field_name="required_operation_authority_ids",
            allow_empty=False,
        )
        if (
            self.freeze_id != GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_FREEZE_ID
            or self.run_id != GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_RUN_ID
            or self.relative_root != GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_RELATIVE_ROOT
            or self.acquisition_freeze.object_fingerprint
            != self.expected_parents.acquisition_freeze.object_fingerprint
            or self.acquisition_terminal.object_fingerprint
            != self.expected_parents.acquisition_terminal.object_fingerprint
            or self.parent_qualification.object_fingerprint
            != self.expected_parents.qualification.object_fingerprint
            or len(self.episode_ids) != 80
            or self.maximum_episode_resident_count != 1
            or self.required_operation_authority_ids
            != GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_AUTHORITY_IDS
            or self.simulator_execution_permitted
            or self.acquisition_bytes_mutable
            or self.implementation_source_closure.kind
            is not SourceClosureKind.CLEAN_GIT_COMMIT
            or self.outcome_access is not OutcomeAccess.EVALUATION_REVEALED
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError(
                "Finite-action recurrence streaming evaluation recovery changes its exact sealed input"
            )


def build_gym_torax_finite_action_recurrence_evaluation_recovery_freeze(
    *,
    expected_parents: GymToraxEvaluationParents,
    acquisition_freeze: GymToraxFiniteActionRecurrenceRecoveryFreeze,
    acquisition_terminal: GymToraxFiniteActionRecurrenceRecoveryTerminalReceipt,
    parent_qualification: GymToraxFiniteActionRecurrenceQualificationBundle,
    extraction_manifest: GymToraxBoundedExtractionManifest,
    implementation_source_closure: ImplementationSourceClosure,
    scientific_approval: ObjectIdentity,
) -> GymToraxFiniteActionRecurrenceEvaluationRecoveryFreeze:
    if (
        acquisition_freeze.fingerprint()
        != expected_parents.acquisition_freeze.object_fingerprint
        or acquisition_terminal.fingerprint()
        != expected_parents.acquisition_terminal.object_fingerprint
        or parent_qualification.fingerprint()
        != expected_parents.qualification.object_fingerprint
        or acquisition_terminal.disposition.value != "COMPLETE_VALID"
        or acquisition_terminal.complete_valid_episode_count != 80
        or acquisition_terminal.partial_episode_count
        or acquisition_terminal.failed_cell_count
    ):
        raise ValueError("Finite-action recurrence evaluation recovery received another acquisition chain")
    return GymToraxFiniteActionRecurrenceEvaluationRecoveryFreeze(
        expected_parents=expected_parents,
        freeze_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_FREEZE_ID,
        run_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_RUN_ID,
        relative_root=GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_RELATIVE_ROOT,
        acquisition_freeze=_identity(acquisition_freeze.freeze_id, acquisition_freeze),
        acquisition_terminal=_identity(
            acquisition_terminal.receipt_id, acquisition_terminal
        ),
        parent_qualification=_identity(
            parent_qualification.bundle_id, parent_qualification
        ),
        extraction_manifest=_identity(
            extraction_manifest.manifest_id, extraction_manifest
        ),
        implementation_source_closure=implementation_source_closure,
        scientific_approval=scientific_approval,
        episode_ids=tuple(
            sorted(value.object_id for value in acquisition_terminal.episodes)
        ),
        defect_id='defect.tokamak-control.evaluation-batch-residency-defect',
        maximum_episode_resident_count=1,
        required_operation_authority_ids=GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_AUTHORITY_IDS,
        simulator_execution_permitted=False,
        acquisition_bytes_mutable=False,
        outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceStreamingEvaluationBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-streaming-evaluation-bundle'
    )

    bundle_id: str
    evaluation_freeze: ObjectIdentity
    scientific_evaluation: GymToraxFiniteActionRecurrenceRecoveryEvaluationBundle
    episode_count_read: int
    peak_episode_resident_count: int
    simulator_executions: int
    acquisition_writes: int
    result: FiniteActionRecurrenceResult
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    evidence_ceiling: EvidenceCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if (
            self.evaluation_freeze.object_schema
            != GymToraxFiniteActionRecurrenceEvaluationRecoveryFreeze.SCHEMA
            or self.scientific_evaluation.result is not self.result
            or self.scientific_evaluation.reason_codes != self.reason_codes
            or self.episode_count_read != 80
            or self.peak_episode_resident_count != 1
            or self.simulator_executions
            or self.acquisition_writes
            or self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
        ):
            raise ValueError("Finite-action recurrence streaming evaluation changes its recovery boundary")


@dataclass(frozen=True, slots=True)
class _CompactEpisode:
    identity: ObjectIdentity
    manifest: ArtifactManifest
    prefix_sha256: str
    prefix_complete: bool
    early_terminated: bool
    reason_codes: tuple[str, ...]


GymToraxEpisodeArtifactLoader = Callable[
    [str], tuple[GymToraxFieldMetadataNativeEpisode, ArtifactManifest]
]


def _require_evaluation_authorities(
    *,
    freeze: GymToraxFiniteActionRecurrenceEvaluationRecoveryFreeze,
    execution_authority: StudyOperationAuthority,
    reveal_authority: StudyOperationAuthority,
    at_utc: str,
) -> None:
    subject = _identity(freeze.freeze_id, freeze)
    require_study_authority(
        execution_authority,
        kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
        subject=subject,
        prerequisite_authority=freeze.scientific_approval,
        grantee_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_EVALUATOR_ID,
        at_utc=at_utc,
    )
    require_study_authority(
        reveal_authority,
        kind=StudyAuthorityKind.OUTCOME_REVEAL,
        subject=subject,
        prerequisite_authority=_identity(
            execution_authority.authority_id, execution_authority
        ),
        grantee_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_EVALUATOR_ID,
        at_utc=at_utc,
    )
    if {
        execution_authority.authority_id,
        reveal_authority.authority_id,
    }.difference(freeze.required_operation_authority_ids):
        raise PermissionError("Finite-action recurrence evaluation authority is outside the recovery freeze")


def evaluate_gym_torax_finite_action_recurrence_streaming_recovery(
    *,
    evaluation_freeze: GymToraxFiniteActionRecurrenceEvaluationRecoveryFreeze,
    acquisition_freeze: GymToraxFiniteActionRecurrenceRecoveryFreeze,
    acquisition_terminal: GymToraxFiniteActionRecurrenceRecoveryTerminalReceipt,
    parent_qualification: GymToraxFiniteActionRecurrenceQualificationBundle,
    field_metadata: GymToraxFieldMetadataManifest,
    load_episode_product: GymToraxEpisodeArtifactLoader,
    execution_authority: StudyOperationAuthority,
    reveal_authority: StudyOperationAuthority,
    at_utc: str,
) -> GymToraxFiniteActionRecurrenceStreamingEvaluationBundle:
    _require_evaluation_authorities(
        freeze=evaluation_freeze,
        execution_authority=execution_authority,
        reveal_authority=reveal_authority,
        at_utc=at_utc,
    )
    if (
        _identity(acquisition_freeze.freeze_id, acquisition_freeze)
        != evaluation_freeze.acquisition_freeze
        or _identity(acquisition_terminal.receipt_id, acquisition_terminal)
        != evaluation_freeze.acquisition_terminal
        or _identity(parent_qualification.bundle_id, parent_qualification)
        != evaluation_freeze.parent_qualification
    ):
        raise ValueError("Finite-action recurrence streaming evaluator received another frozen chain")

    routes = {value.episode_id: value for value in acquisition_freeze.assignment.routes}
    cells = {value.episode_id: value for value in acquisition_freeze.cells}
    unit_specs = {value.unit_id: value for value in acquisition_freeze.assignment.units}
    prepared_by_coordinate = {
        value.source_cell_id: value for value in acquisition_freeze.units
    }
    compact: dict[str, _CompactEpisode] = {}
    complete_by_episode: dict[str, bool] = {}
    q_by_episode: dict[str, np.ndarray] = {}
    for episode_id in sorted(routes):
        episode, manifest = load_episode_product(episode_id)
        cell = cells[episode_id]
        request = materialize_gym_torax_finite_action_recurrence_recovery_request(acquisition_freeze, cell)
        if (
            episode.episode_id != episode_id
            or episode.request != _identity(request.request_id, request)
            or episode.preparation
            != _identity(request.preparation.preparation_id, request.preparation)
            or episode.numerical_member
            != _identity(request.numerical_member.member_id, request.numerical_member)
            or episode.action_word
            != _identity(
                request.schedule.action_word.word_id, request.schedule.action_word
            )
        ):
            raise ValueError(f'Finite-action recurrence streaming episode lineage differs:{episode_id}')
        complete = _is_complete_valid(episode)
        if complete:
            _validate_hfr_episode(
                freeze=acquisition_freeze,  # type: ignore[arg-type]
                cell=cell,
                episode=episode,
                metadata=field_metadata,
            )
            q_by_episode[episode_id] = _q_values(episode)
        compact[episode_id] = _CompactEpisode(
            identity=_episode_identity(episode),
            manifest=manifest,
            prefix_sha256=_prefix_sha(episode),
            prefix_complete=all(value in episode.state_clocks for value in range(105)),
            early_terminated=episode.termination or episode.truncation,
            reason_codes=episode.reason_codes,
        )
        complete_by_episode[episode_id] = complete
        del episode
        gc.collect()

    sealed_rows = []
    for episode_id, route in routes.items():
        summary = compact[episode_id]
        unit = unit_specs[route.unit_id]
        prepared = prepared_by_coordinate[unit.preparation_coordinate_id]
        member = next(
            value
            for value in acquisition_freeze.numerical_members
            if value.member_id == route.model_member_id
        )
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
                for value in acquisition_freeze.assignment.routes
                if value.unit_id == route.unit_id
                and value.model_member_id == route.model_member_id
                and value.role is mate_role
            )
            pair_incomplete = (
                not summary.prefix_complete
                or not compact[mate.episode_id].prefix_complete
            )
        nonattempt = pair_incomplete or (
            route.role is FiniteActionRecurrenceRouteRole.HOLD_CONTROL
            and not summary.prefix_complete
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
                    sha256=summary.prefix_sha256,
                ),
                prefix_complete_through_state_104=summary.prefix_complete,
                preaction_disposition=(
                    FiniteActionPreactionDisposition.NONATTEMPT
                    if nonattempt
                    else FiniteActionPreactionDisposition.READY
                ),
                observed_occurrences=(
                    ()
                    if nonattempt
                    else _observed(
                        acquisition_freeze.assignment.active_word
                        if route.role is FiniteActionRecurrenceRouteRole.ACTIVE
                        else acquisition_freeze.assignment.matched_hold_word,
                        episode_id,
                    )
                ),
                clipped=False,
                rejected=False,
                substituted=False,
                early_terminated=False if nonattempt else summary.early_terminated,
                delivery_trace=summary.identity,
                sealed_outcome_artifact=_artifact(summary.manifest),
                custody_evidence=(
                    _identity(
                        summary.manifest.materialization.materialization_id,
                        summary.manifest.materialization,
                    ),
                ),
                outcome_access=OutcomeAccess.EVALUATION_SEALED,
            )
        )
    sealed = FiniteActionRecurrenceProspectiveBundle(
        bundle_id='sealed-bundle.tokamak-control.finite-action-recurrence-recovery.streamed-evaluation',
        assignment=acquisition_freeze.assignment,
        episodes=tuple(sorted(sealed_rows, key=lambda value: value.episode_id)),
        issue_receipt=_identity(evaluation_freeze.freeze_id, evaluation_freeze),
        outcome_access=OutcomeAccess.EVALUATION_SEALED,
    )
    predicate_kinds = tuple(
        sorted(
            {
                predicate.kind
                for receipt in acquisition_freeze.assignment.occurrence_receipts
                for predicate in receipt.qualification_spec.predicates
            },
            key=lambda value: value.value,
        )
    )
    revealed_rows = []
    for source in sealed.episodes:
        complete = complete_by_episode[source.episode_id]
        q = q_by_episode.get(source.episode_id)
        samples = (
            ()
            if q is None
            or source.route.role is FiniteActionRecurrenceRouteRole.HOLD_CONTROL
            else tuple(
                FiniteActionRecurrenceResponseSample(
                    sample_id=f"sample.{source.episode_id}.{index:02d}",
                    quantity_id=acquisition_freeze.assignment.effect_quantity_id,
                    coordinate=coordinate,
                    value=NamedDecimal(
                        value_id=f"value.{source.episode_id}.{index:02d}",
                        value=Decimal(str(float(q[int(coordinate.coordinate)]))),
                        unit=acquisition_freeze.assignment.effect_native_unit,
                    ),
                )
                for index, coordinate in enumerate(
                    acquisition_freeze.assignment.response_coordinates
                )
            )
        )
        predicates = tuple(
            FiniteActionRecurrencePredicateResult(
                result_id=(
                    f"predicate-result.{source.episode_id}.{kind.value.lower().replace('_', '-')}"
                ),
                kind=kind,
                status=GateStatus.PASS if complete else GateStatus.UNEVALUABLE,
                reason_codes=() if complete else ("FINITE_ACTION_RECURRENCE_RECOVERY_EPISODE_INCOMPLETE",),
            )
            for kind in predicate_kinds
        )
        revealed_rows.append(
            FiniteActionRecurrenceRevealedEpisode(
                reveal_id=f'reveal.{source.episode_id}.streamed-evaluation',
                episode_id=source.episode_id,
                sealed_episode=_identity(source.episode_id, source),
                delivery_trace=source.delivery_trace,
                sealed_outcome_artifact=source.sealed_outcome_artifact,
                outcome_trace_id=f'outcome-trace.{source.episode_id}.streamed-evaluation',
                response_samples=samples,
                predicate_results=predicates,
                technical_reason_codes=(
                    ()
                    if complete
                    else tuple(
                        sorted(
                            {
                                "FINITE_ACTION_RECURRENCE_RECOVERY_TECHNICAL_PARTIAL",
                                *compact[source.episode_id].reason_codes,
                            }
                        )
                    )
                ),
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            )
        )
    revealed = FiniteActionRecurrenceRevealedBundle(
        reveal_id='revealed-bundle.tokamak-control.finite-action-recurrence-recovery.streamed-evaluation',
        sealed_bundle=sealed,
        integrity=FiniteActionRecurrenceRevealIntegrity.VALID,
        reveal_authorization=_identity(reveal_authority.authority_id, reveal_authority),
        integrity_evidence=tuple(
            sorted(
                (
                    _identity(reveal_authority.authority_id, reveal_authority),
                    _identity(acquisition_terminal.receipt_id, acquisition_terminal),
                    _identity(evaluation_freeze.freeze_id, evaluation_freeze),
                ),
                key=lambda value: value.object_id,
            )
        ),
        integrity_reason_codes=(),
        episodes=tuple(sorted(revealed_rows, key=lambda value: value.episode_id)),
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    adjudication = MatchedFiniteActionHoldRecurrenceEvaluator().evaluate(
        assignment=acquisition_freeze.assignment,
        revealed=revealed,
    )
    scientific = GymToraxFiniteActionRecurrenceRecoveryEvaluationBundle(
        bundle_id='evaluation-bundle.tokamak-control.finite-action-recurrence-recovery.streamed-evaluation',
        freeze=_identity(acquisition_freeze.freeze_id, acquisition_freeze),
        parent_qualification=_identity(
            parent_qualification.bundle_id, parent_qualification
        ),
        terminal=_identity(acquisition_terminal.receipt_id, acquisition_terminal),
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
    return GymToraxFiniteActionRecurrenceStreamingEvaluationBundle(
        bundle_id='evaluation-recovery.tokamak-control.finite-action-recurrence-recovery.streamed-evaluation',
        evaluation_freeze=_identity(evaluation_freeze.freeze_id, evaluation_freeze),
        scientific_evaluation=scientific,
        episode_count_read=80,
        peak_episode_resident_count=1,
        simulator_executions=0,
        acquisition_writes=0,
        result=scientific.result,
        reason_codes=scientific.reason_codes,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
    )


__all__ = [
    'GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_AUTHORITY_IDS',
    'GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_EVALUATOR_ID',
    'GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_FREEZE_ID',
    'GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_RELATIVE_ROOT',
    'GYM_TORAX_FINITE_ACTION_RECURRENCE_EVALUATION_RECOVERY_RUN_ID',
    'GymToraxFiniteActionRecurrenceEvaluationRecoveryFreeze',
    'GymToraxFiniteActionRecurrenceStreamingEvaluationBundle',
    'build_gym_torax_finite_action_recurrence_evaluation_recovery_freeze',
    'evaluate_gym_torax_finite_action_recurrence_streaming_recovery',
]

"""Authorized occurrence, calibration, route and recurrence adjudication for Gym-TORAX finite-action recurrence."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import Decimal
import hashlib
from typing import ClassVar

import numpy as np

from empirical_lawhood.kernel.action_contracts import ActionDeliveryStage, OccurrenceActionWord, ObservedActionOccurrence
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.identification import LawQualificationResult
from empirical_lawhood.kernel.provenance import (
    EvidenceLink,
    EvidenceRelation,
    ObjectIdentity,
)
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    NamedDecimal,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.planning.controller_study import DeliveryStageValueTolerance, StageAwareDeliveryEquivalenceSpec
from empirical_lawhood.planning.finite_action_occurrence import FiniteActionOccurrenceEvaluator, FiniteActionOccurrencePredicateEvidence, FiniteActionOccurrenceQualificationReceipt, FiniteActionOccurrenceTemplateBinder, FiniteActionWordDeliveryEvidence, FiniteActionWordRole, FiniteActionOccurrenceEvidence, finite_action_branch_operator_identity
from empirical_lawhood.planning.finite_action_recurrence import FINITE_ACTION_RECURRENCE_CALIBRATION_OCCURRENCE_ROLE_ID, FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS, FINITE_ACTION_RECURRENCE_HOLD_RESERVE_SLOT_ID, FiniteActionPreactionDisposition, FiniteActionRecurrenceAssignmentBindingReceipt, FiniteActionRecurrenceAssignmentTemplateBinder, FiniteActionRecurrenceAssignment, FiniteActionRecurrencePredicateResult, FiniteActionRecurrenceProspectiveBundle, FiniteActionRecurrenceRevealIntegrity, FiniteActionRecurrenceRevealedBundle, FiniteActionRecurrenceRevealedEpisode, FiniteActionRecurrenceResponseSample, FiniteActionRecurrenceRouteQualificationReceipt, FiniteActionRecurrenceRouteRole, FiniteActionRecurrenceSealedEpisode, FiniteActionRouteCanaryCoordinateRole, FiniteActionRouteCanaryDisposition, FiniteActionRouteCanaryEpisodeReceipt, FiniteActionRouteCanaryEpisodeRole
from empirical_lawhood.planning.native_hold_decision_cell_calibration import NativeHoldAnchorSelectionKind, NativeHoldCalibrationAnchorOption, NativeHoldCalibrationAnchorSlot, NativeHoldCalibrationPredicateKind, NativeHoldCalibrationPredicateSpec, NativeHoldCalibrationSelectedAnchor, NativeHoldCalibrationSelectedRoster, NativeHoldDecisionCellCalibrationReceipt, NativeHoldDecisionCellCalibrationSpec, NativeHoldSiblingSelectorRule, NativeHoldSiblingSelector, preparation_values_fingerprint
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactManifest
from empirical_lawhood.runtime.finite_action_recurrence import FiniteActionRecurrenceAdjudication, MatchedFiniteActionHoldRecurrenceEvaluator
from empirical_lawhood.runtime.native_hold_decision_cell_calibration import NativeHoldActionLocalProjection, NativeHoldCalibrationOccurrenceEvidence, NativeHoldCalibrationObservedPredicate, NativeHoldCalibrationOperandKind, NativeHoldDecisionCellCalibrationEvaluator

from .diagnostic_contracts import GymToraxDeliveryDisposition, GymToraxNumericalDisposition, GymToraxObservationDisposition, GymToraxSourceDisposition
from .field_metadata_contracts import GymToraxFieldMetadataNativeEpisode
from .field_metadata import GymToraxFieldMetadataManifest
from .finite_action_recurrence_execution import GymToraxFiniteActionRecurrenceStageAcquisitionReceipt, GymToraxFiniteActionRecurrenceTerminalAcquisitionDisposition, GymToraxFiniteActionRecurrenceTerminalAcquisitionReceipt
from .finite_action_recurrence_protocol import GymToraxFiniteActionRecurrenceCellStage, GymToraxFiniteActionRecurrenceCell, GymToraxFiniteActionRecurrenceProspectiveFreeze, GYM_TORAX_FINITE_ACTION_RECURRENCE_CALIBRATION_RECEIPT_ID, GYM_TORAX_FINITE_ACTION_RECURRENCE_ROUTE_QUALIFICATION_RECEIPT_ID, materialize_gym_torax_finite_action_recurrence_request
from .selected_parent_execution import GymToraxSelectedParentTerminalAcquisitionReceipt
from .selected_parent_protocol import GymToraxSelectedParentFreeze
from .selected_parent_science import GymToraxSelectedParentScienceEvaluationBundle, _precutoff_equal, _validate_episode_and_q
from .source_assessment_execution import GymToraxArtifactPublicationItem, GymToraxBoundedArtifactStore


_FINITE_ACTION_RECURRENCE_EVALUATOR_ID = 'evaluator.tokamak-control.finite-action-recurrence'
_FINITE_ACTION_RECURRENCE_OCCURRENCE_REVEAL_AUTHORITY_ID = (
    'authority.tokamak-control.finite-action-recurrence.measurement-occurrence-reveal'
)
_FINITE_ACTION_RECURRENCE_ROUTE_REVEAL_AUTHORITY_ID = (
    'authority.tokamak-control.finite-action-recurrence.route-outcome-reveal'
)
_FINITE_ACTION_RECURRENCE_RECURRENCE_REVEAL_AUTHORITY_ID = (
    'authority.tokamak-control.finite-action-recurrence.recurrence-outcome-reveal'
)

GymToraxFieldMetadataEpisodeLoader = Callable[[str], GymToraxFieldMetadataNativeEpisode]


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceQualificationBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-qualification-bundle'
    )

    bundle_id: str
    freeze: ObjectIdentity
    law_qualification_result: ObjectIdentity
    selected_parent_occurrence_reveal_authority: ObjectIdentity
    route_reveal_authority: ObjectIdentity
    occurrence_receipts: tuple[FiniteActionOccurrenceQualificationReceipt, ...]
    native_hold_calibration: NativeHoldDecisionCellCalibrationReceipt
    route_qualification: FiniteActionRecurrenceRouteQualificationReceipt
    assignment: FiniteActionRecurrenceAssignment
    assignment_binding: FiniteActionRecurrenceAssignmentBindingReceipt
    selected_parent_episode_count_read: int
    hfr_stage_episode_count_read: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        if self.freeze.object_schema != GymToraxFiniteActionRecurrenceProspectiveFreeze.SCHEMA:
            raise ValueError("Finite-action recurrence qualification binds another freeze")
        if self.law_qualification_result.object_schema != LawQualificationResult.SCHEMA:
            raise ValueError("Finite-action recurrence qualification binds another local law result")
        require_sorted_unique_ids(
            self.occurrence_receipts,
            attribute="receipt_id",
            field_name="occurrence_receipts",
        )
        if (
            len(self.occurrence_receipts) != 6
            or self.selected_parent_episode_count_read != 144
            or self.hfr_stage_episode_count_read != 18
        ):
            raise ValueError("Finite-action recurrence qualification changes the exact 144+18 reveal roster")
        if self.assignment.route_qualification != self.route_qualification:
            raise ValueError("Finite-action recurrence assignment changes its exact route qualification")
        if self.assignment.native_hold_calibration != self.native_hold_calibration:
            raise ValueError("Finite-action recurrence assignment changes its native HOLD calibration")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("Finite-action recurrence qualification changes its reveal custody")


@dataclass(frozen=True, slots=True)
class GymToraxFiniteActionRecurrenceRecurrenceEvaluationBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = (
        'empirical-lawhood/simulators/gym-torax-native/gym-torax-finite-action-recurrence-recurrence-evaluation-bundle'
    )

    bundle_id: str
    freeze: ObjectIdentity
    qualification: ObjectIdentity
    terminal_acquisition: ObjectIdentity
    reveal_authority: ObjectIdentity
    sealed_bundle: FiniteActionRecurrenceProspectiveBundle
    revealed_bundle: FiniteActionRecurrenceRevealedBundle
    adjudication: FiniteActionRecurrenceAdjudication
    episode_count_read: int
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        if self.freeze.object_schema != GymToraxFiniteActionRecurrenceProspectiveFreeze.SCHEMA:
            raise ValueError("Finite-action recurrence result binds another freeze")
        if self.qualification.object_schema != GymToraxFiniteActionRecurrenceQualificationBundle.SCHEMA:
            raise ValueError("Finite-action recurrence result binds another qualification")
        if (
            self.terminal_acquisition.object_schema
            != GymToraxFiniteActionRecurrenceTerminalAcquisitionReceipt.SCHEMA
        ):
            raise ValueError("Finite-action recurrence result binds another terminal")
        if self.episode_count_read != 80:
            raise ValueError("Finite-action recurrence evaluator must read exactly 80 episodes")
        if self.adjudication.assignment != ObjectIdentity.from_record(
            self.sealed_bundle.assignment.assignment_id,
            self.sealed_bundle.assignment,
        ):
            raise ValueError("Finite-action recurrence adjudication changes its assignment")
        if (
            self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or self.visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
        ):
            raise ValueError("Finite-action recurrence result changes its reveal custody")


def _identity(identifier: str, record: CanonicalRecord) -> ObjectIdentity:
    return ObjectIdentity.from_record(identifier, record)


def _artifact(manifest: ArtifactManifest) -> ArtifactIdentity:
    logical = manifest.logical
    return ArtifactIdentity(
        artifact_id=logical.logical_artifact_id,
        role='tokamak-control.finite-action-native-episode',
        payload_schema=logical.payload_schema,
        sha256=logical.content_sha256,
        media_type=logical.media_type,
        size_bytes=manifest.materialization.size_bytes,
    )


def _episode_identity(episode: GymToraxFieldMetadataNativeEpisode) -> ObjectIdentity:
    return _identity(episode.episode_id, episode)


def _observed(word: OccurrenceActionWord, stem: str) -> tuple[ObservedActionOccurrence, ...]:
    return tuple(
        sorted(
            (
                ObservedActionOccurrence(
                    observation_id=f"observed.{stem}.{value.occurrence_id}",
                    expected_occurrence_id=value.occurrence_id,
                    requested=value.requested,
                    accepted=value.accepted,
                    applied=value.applied,
                    realized=value.realized,
                    reason_codes=(),
                )
                for value in word.occurrences
            ),
            key=lambda value: value.expected_occurrence_id,
        )
    )


def _require_reveal(
    *,
    authority: StudyOperationAuthority,
    authority_id: str,
    subject: ObjectIdentity,
    prerequisite: ObjectIdentity,
    at_utc: str,
) -> None:
    require_study_authority(
        authority,
        kind=StudyAuthorityKind.OUTCOME_REVEAL,
        subject=subject,
        prerequisite_authority=prerequisite,
        grantee_id=_FINITE_ACTION_RECURRENCE_EVALUATOR_ID,
        at_utc=at_utc,
    )
    if (
        authority.authority_id != authority_id
        or not authority.allows_reveal
        or authority.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
    ):
        raise PermissionError("Finite-action recurrence evaluator lacks its exact reveal authority")


def _load_exact(
    *,
    expected_ids: tuple[str, ...],
    manifests: tuple[ArtifactManifest, ...],
    loader: GymToraxFieldMetadataEpisodeLoader,
) -> tuple[dict[str, GymToraxFieldMetadataNativeEpisode], dict[str, ArtifactManifest]]:
    by_manifest = {value.logical.logical_artifact_id: value for value in manifests}
    expected_logical = {f"artifact.{value}" for value in expected_ids}
    if set(by_manifest) != expected_logical:
        raise ValueError("Finite-action recurrence episode sidecars differ from the exact reveal roster")
    episodes: dict[str, GymToraxFieldMetadataNativeEpisode] = {}
    for episode_id in expected_ids:
        episode = loader(episode_id)
        manifest = by_manifest[f"artifact.{episode_id}"]
        if episode.fingerprint() != manifest.logical.content_sha256:
            raise ValueError(f'Finite-action recurrence episode bytes differ from sidecar:{episode_id}')
        episodes[episode_id] = episode
    return episodes, by_manifest


def _validate_hfr_episode(
    *,
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    cell: GymToraxFiniteActionRecurrenceCell,
    episode: GymToraxFieldMetadataNativeEpisode,
    metadata: GymToraxFieldMetadataManifest,
) -> np.ndarray:
    request = materialize_gym_torax_finite_action_recurrence_request(freeze, cell)
    if (
        episode.episode_id != cell.episode_id
        or episode.request != _identity(request.request_id, request)
        or episode.preparation
        != _identity(request.preparation.preparation_id, request.preparation)
        or episode.numerical_member
        != _identity(request.numerical_member.member_id, request.numerical_member)
        or episode.action_word
        != _identity(request.schedule.action_word.word_id, request.schedule.action_word)
    ):
        raise ValueError(f'Finite-action recurrence episode lineage differs:{cell.cell_id}')
    if (
        episode.state_clocks != tuple(range(121))
        or episode.missing_required_state_clocks
        or episode.last_valid_state_clock != 120
        or episode.source_disposition is not GymToraxSourceDisposition.AVAILABLE
        or episode.delivery_disposition is not GymToraxDeliveryDisposition.COMPLETE
        or episode.numerical_disposition is not GymToraxNumericalDisposition.VALID
        or episode.observation_disposition
        is not GymToraxObservationDisposition.COMPLETE
        or episode.termination
        or episode.truncation
        or episode.outcome_access is not OutcomeAccess.EVALUATION_SEALED
    ):
        raise ValueError(f'Finite-action recurrence episode is not complete valid evidence:{cell.cell_id}')
    deliveries = {value.request_clock: value for value in episode.deliveries}
    if set(deliveries) != set(range(120)):
        raise ValueError(f'Finite-action recurrence delivery clock roster differs:{cell.cell_id}')
    for expected in request.schedule.rows:
        observed = deliveries[expected.request_clock]
        if (
            observed.receiver_clock != expected.request_clock + 1
            or observed.occurrence_id != expected.controlled_occurrence_id
            or observed.disposition is not GymToraxDeliveryDisposition.COMPLETE
            or observed.requested != expected.action
            or observed.accepted is None
            or observed.applied is None
            or observed.realized is None
        ):
            raise ValueError(f'Finite-action recurrence four-stage delivery differs:{cell.cell_id}')
    blocks = tuple(
        value
        for value in episode.blocks
        if value.category == "source-scalar" and value.native_field_id == "Q_fusion"
    )
    if len(blocks) != 1:
        raise ValueError(f'Finite-action recurrence episode lacks one Q_fusion block:{cell.cell_id}')
    block = blocks[0]
    expected_metadata = metadata.by_key()[("scalars", "Q_fusion")]
    q = block.array()
    if (
        block.native_unit != expected_metadata.native_unit
        or block.native_frame_id != expected_metadata.native_frame_id
        or block.field_metadata_id != expected_metadata.field_metadata_id
        or block.dimension_ids
        != ("state-clock", *expected_metadata.native_dimension_ids)
        or block.clock_values != tuple(range(121))
        or q.shape != (121, 1)
        or not np.isfinite(q).all()
    ):
        raise ValueError(f'Finite-action recurrence Q_fusion metadata/shape differs:{cell.cell_id}')
    return np.asarray(q[:, 0], dtype=np.float64)


def _prefix_sha(episode: GymToraxFieldMetadataNativeEpisode) -> str:
    digest = hashlib.sha256()
    for block in episode.blocks:
        if block.category != "coordinate" and not block.category.startswith("source-"):
            continue
        digest.update(block.block_id.encode())
        values = block.array()
        if block.clock_values:
            mask = np.asarray(block.clock_values) <= 104
            values = values[mask]
        digest.update(np.ascontiguousarray(values, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def _preparation_values(unit) -> tuple[NamedDecimal, ...]:  # type: ignore[no-untyped-def]
    value = unit.preparation
    return tuple(
        sorted(
            (
                NamedDecimal(
                    'tokamak-control.preparation.bootstrap-multiplier',
                    value.bootstrap_multiplier,
                    "1",
                ),
                NamedDecimal(
                    'tokamak-control.preparation.initial-density-nbar',
                    value.initial_density_nbar,
                    "1",
                ),
                NamedDecimal(
                    'tokamak-control.preparation.initial-temperature-scale',
                    value.initial_temperature_scale,
                    "1",
                ),
                NamedDecimal(
                    'tokamak-control.preparation.inner-transport-scale',
                    value.inner_transport_scale,
                    "1",
                ),
            ),
            key=lambda item: item.value_id,
        )
    )


def _content_artifact(
    *, artifact_id: str, role: str, payload_schema: str, sha256: str
) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=artifact_id,
        role=role,
        payload_schema=payload_schema,
        sha256=sha256,
        media_type="application/json",
        size_bytes=0,
    )


def _first_difference(left: np.ndarray, right: np.ndarray, *, start: int) -> int | None:
    indices = np.flatnonzero(np.abs(left[start:] - right[start:]) > 1e-12)
    return None if not len(indices) else int(indices[0]) + start


def _occurrence_qualification(
    *,
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    selected_parent_freeze: GymToraxSelectedParentFreeze,
    law_qualification_result: LawQualificationResult,
    selected_parent_episodes: Mapping[str, GymToraxFieldMetadataNativeEpisode],
    metadata: GymToraxFieldMetadataManifest,
) -> tuple[FiniteActionOccurrenceQualificationReceipt, ...]:
    if (
        law_qualification_result.result_id != 'qualification-result.tokamak-control.matched-evaluation'
        or law_qualification_result.scientific_status is not ScientificStatus.SUPPORTED
        or law_qualification_result.response_law is None
    ):
        raise ValueError(
            "Finite-action recurrence occurrence qualification requires the supported exact local law result"
        )
    units = {value.unit_id: value for value in selected_parent_freeze.units}
    cells = {
        (
            value.physical_unit_instance_id,
            value.numerical_member_id,
            value.action_word_id,
        ): value
        for value in selected_parent_freeze.cells
        if units[value.physical_unit_instance_id].cell_kind == "bt"
    }
    receipts: list[FiniteActionOccurrenceQualificationReceipt] = []
    binder = FiniteActionOccurrenceTemplateBinder(
        implementation_id='binder.tokamak-control.finite-action-occurrence',
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
    )
    role_by_word = {
        'action-word.tokamak-control.lower-ip': FiniteActionWordRole.ACTIVE,
        'action-word.tokamak-control.future-ip': FiniteActionWordRole.FUTURE_CAUSAL_FALSIFIER,
        'action-word.tokamak-control.native-hold': FiniteActionWordRole.MATCHED_HOLD,
        'action-word.tokamak-control.wrong-sign-ip': FiniteActionWordRole.WRONG_SIGN_FALSIFIER,
    }
    for template in freeze.corrected_occurrence_templates.templates:
        spec, _ = binder.bind(
            receipt_id=f"binding.{template.expected_qualification_spec_id}",
            template=template,
            branch_selection=freeze.selection_bridge,
            law_qualification_result=law_qualification_result,
        )
        tier = f"TIER_{template.local_support_id.rsplit('-', 1)[-1]}"
        unit_ids = tuple(
            sorted(
                value.unit_id
                for value in selected_parent_freeze.units
                if value.cell_kind == "bt" and value.tier_id == tier
            )
        )
        if len(unit_ids) != 6:
            raise ValueError("Finite-action recurrence occurrence source changes its exact six-unit tier")
        q_by_unit_word: dict[tuple[str, str], np.ndarray] = {}
        episode_by_unit_word: dict[tuple[str, str], GymToraxFieldMetadataNativeEpisode] = {}
        decisive: list[ObjectIdentity] = []
        for unit_id in unit_ids:
            for word_id in role_by_word:
                cell = cells[(unit_id, template.model_member_id, word_id)]
                episode = selected_parent_episodes[cell.episode_id]
                q_by_unit_word[(unit_id, word_id)] = _validate_episode_and_q(
                    acquisition_freeze=selected_parent_freeze,
                    cell=cell,
                    episode=episode,
                    field_metadata_manifest=metadata,
                )
                episode_by_unit_word[(unit_id, word_id)] = episode
                decisive.append(_episode_identity(episode))

        active_onsets: list[int | None] = []
        future_onsets: list[int | None] = []
        future_null = Decimal(0)
        for unit_id in unit_ids:
            hold_q = q_by_unit_word[(unit_id, 'action-word.tokamak-control.native-hold')]
            active_q = q_by_unit_word[(unit_id, 'action-word.tokamak-control.lower-ip')]
            future_q = q_by_unit_word[(unit_id, 'action-word.tokamak-control.future-ip')]
            active_episode = episode_by_unit_word[
                (unit_id, 'action-word.tokamak-control.lower-ip')
            ]
            future_episode = episode_by_unit_word[
                (unit_id, 'action-word.tokamak-control.future-ip')
            ]
            hold_episode = episode_by_unit_word[
                (unit_id, 'action-word.tokamak-control.native-hold')
            ]
            if not _precutoff_equal(
                active_episode, hold_episode
            ) or not _precutoff_equal(future_episode, hold_episode):
                raise ValueError(
                    "Finite-action recurrence occurrence source violates its retained causal cutoff"
                )
            active_onsets.append(_first_difference(active_q, hold_q, start=0))
            future_onsets.append(_first_difference(future_q, hold_q, start=0))
            future_null = max(
                future_null,
                Decimal(str(float(np.max(np.abs(future_q[:111] - hold_q[:111]))))),
            )
        active_first = (
            None
            if any(value is None for value in active_onsets)
            else min(active_onsets)  # type: ignore[type-var]
        )
        future_first = (
            None
            if any(value is None for value in future_onsets)
            else min(future_onsets)  # type: ignore[type-var]
        )
        deliveries = tuple(
            sorted(
                (
                    FiniteActionWordDeliveryEvidence(
                        delivery_evidence_id=(
                            f"delivery.{spec.qualification_spec_id}.{role.value.lower()}"
                        ),
                        role=role,
                        action_word=_identity(word.word_id, word),
                        occurrences=_observed(
                            word, f"{spec.qualification_spec_id}.{role.value.lower()}"
                        ),
                        clipped=False,
                        rejected=False,
                        substituted=False,
                        early_terminated=False,
                        trace=_episode_identity(
                            episode_by_unit_word[(unit_ids[0], word.word_id)]
                        ),
                    )
                    for role, word in (
                        (FiniteActionWordRole.ACTIVE, spec.active_word),
                        (
                            FiniteActionWordRole.FUTURE_CAUSAL_FALSIFIER,
                            spec.future_word,
                        ),
                        (FiniteActionWordRole.MATCHED_HOLD, spec.matched_hold_word),
                        (
                            FiniteActionWordRole.WRONG_SIGN_FALSIFIER,
                            spec.wrong_sign_word,
                        ),
                    )
                ),
                key=lambda value: value.delivery_evidence_id,
            )
        )
        law_qualification_identity = _identity(law_qualification_result.result_id, law_qualification_result)
        predicates = tuple(
            sorted(
                (
                    FiniteActionOccurrencePredicateEvidence(
                        evidence_id=f"evidence.{spec.qualification_spec_id}.{value.predicate_id}",
                        predicate_id=value.predicate_id,
                        kind=value.kind,
                        status=GateStatus.PASS,
                        observed=NamedDecimal(
                            value_id=f"value.{spec.qualification_spec_id}.{value.predicate_id}",
                            value=Decimal(1),
                            unit="1",
                        ),
                        evidence_identity=law_qualification_identity,
                        reason_codes=(),
                    )
                    for value in spec.predicates
                ),
                key=lambda value: value.evidence_id,
            )
        )
        evidence = FiniteActionOccurrenceEvidence(
            evidence_id=f"evidence.{spec.qualification_spec_id}",
            qualification_spec=_identity(spec.qualification_spec_id, spec),
            word_deliveries=deliveries,
            predicate_evidence=predicates,
            active_first_difference_state=active_first,
            future_first_difference_state=future_first,
            future_hold_max_abs_difference_through_state_110=future_null,
            decisive_evidence=tuple(
                sorted(set(decisive), key=lambda value: value.object_id)
            ),
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
        )
        receipts.append(
            FiniteActionOccurrenceEvaluator().evaluate(
                receipt_id=template.expected_qualification_receipt_id,
                spec=spec,
                evidence=evidence,
            )
        )
    return tuple(sorted(receipts, key=lambda value: value.receipt_id))


def _calibration_spec_and_roster(
    *, freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze
) -> tuple[
    NativeHoldDecisionCellCalibrationSpec, NativeHoldCalibrationSelectedRoster
]:
    template = freeze.corrected_recurrence_template
    calibration_units = tuple(
        sorted(
            (
                value
                for value in freeze.units
                if value.prospective_role == "HOLD_CALIBRATION"
            ),
            key=lambda value: value.source_anchor_slot_id or "",
        )
    )
    if tuple(value.source_anchor_slot_id for value in calibration_units) != (
        *FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS,
        FINITE_ACTION_RECURRENCE_HOLD_RESERVE_SLOT_ID,
    ):
        raise ValueError("Finite-action recurrence calibration changes the exact five-anchor roster")
    options = tuple(
        NativeHoldCalibrationAnchorOption(
            option_id=f'option.tokamak-control.finite-action-recurrence.{value.source_anchor_slot_id}',
            coordinate_id=value.source_cell_id,
            preparation_values=_preparation_values(value),
            preparation_fingerprint=preparation_values_fingerprint(
                _preparation_values(value)
            ),
        )
        for value in calibration_units
    )
    slots = tuple(
        NativeHoldCalibrationAnchorSlot(
            slot_id=unit.source_anchor_slot_id or "",
            primary=option,
            conditional_alternative=None,
        )
        for unit, option in zip(calibration_units, options, strict=True)
    )
    evaluator_sha = freeze.implementation_source_closure.implementation_sha256
    evaluator_payload = _content_artifact(
        artifact_id='artifact.tokamak-control.native-hold-calibration-evaluator',
        role="evaluator-implementation",
        payload_schema='empirical-lawhood/simulators/gym-torax-native/finite-action-recurrence/native-hold-calibration-evaluator',
        sha256=evaluator_sha,
    )
    spec = NativeHoldDecisionCellCalibrationSpec(
        calibration_spec_id='spec.tokamak-control.native-hold-decision-cell',
        hold_decision_cell_id='decision-cell.tokamak-control.native-hold-decision-cell',
        active_decision_cell_ids=('decision-cell.tokamak-control.active-decision-cell',),
        anchor_slots=slots,
        model_member_ids=template.model_member_ids,
        occurrence_role_ids=(FINITE_ACTION_RECURRENCE_CALIBRATION_OCCURRENCE_ROLE_ID,),
        source=template.source,
        schedule=template.schedule,
        native_hold_action_word=template.matched_hold_word,
        retained_history=template.retained_history,
        horizon=template.horizon,
        delivery_equivalence=StageAwareDeliveryEquivalenceSpec(
            equivalence_id='equivalence.tokamak-control.native-hold-decision-cell',
            require_exact_word_identity=True,
            require_all_delivery_stages=True,
            stage_value_tolerances=tuple(
                DeliveryStageValueTolerance(
                    stage=stage,
                    tolerance=NamedDecimal(
                        value_id=f'tolerance.tokamak-control.finite-action-recurrence.{stage.value.lower()}',
                        value=(
                            Decimal(0)
                            if stage is ActionDeliveryStage.REQUESTED
                            else Decimal("0.5")
                        ),
                        unit="A",
                    ),
                )
                for stage in ActionDeliveryStage
            ),
        ),
        calibration_predicates=tuple(
            sorted(
                (
                    NativeHoldCalibrationPredicateSpec(
                        predicate_id='predicate.tokamak-control.surgical-hfr.delivery-complete',
                        quantity_id='tokamak-control.hfr.delivery-complete',
                        predicate_kind=NativeHoldCalibrationPredicateKind.BOOLEAN_EQUALS,
                        native_unit=None,
                        lower=None,
                        upper=None,
                        expected_boolean=True,
                        expected_identity=None,
                    ),
                    NativeHoldCalibrationPredicateSpec(
                        predicate_id='predicate.tokamak-control.surgical-hfr.observation-complete',
                        quantity_id='tokamak-control.hfr.observation-complete',
                        predicate_kind=NativeHoldCalibrationPredicateKind.BOOLEAN_EQUALS,
                        native_unit=None,
                        lower=None,
                        upper=None,
                        expected_boolean=True,
                        expected_identity=None,
                    ),
                ),
                key=lambda value: value.predicate_id,
            )
        ),
        sibling_hold_selector=NativeHoldSiblingSelector(
            selector_id='selector.tokamak-control.native-hold-decision-cell',
            candidate_key_ids=('candidate-key.tokamak-control.native-hold-decision-cell',),
            selected_candidate_key_id='candidate-key.tokamak-control.native-hold-decision-cell',
            rule=NativeHoldSiblingSelectorRule.BYTEWISE_FIRST,
        ),
        evaluator=ExecutableReference(
            reference_id='reference.tokamak-control.native-hold-calibration-evaluator',
            capability_key="control.native-hold-calibration",
            capability_version="1.0.0",
            evaluator_key='evaluator.tokamak-control.native-hold-calibration',
            payload=evaluator_payload,
            payload_format=SafePayloadFormat.CANONICAL_JSON,
            input_schema=NativeHoldActionLocalProjection.SCHEMA,
            output_schema=NativeHoldDecisionCellCalibrationReceipt.SCHEMA,
            deterministic=True,
        ),
        evaluator_implementation_sha256=evaluator_sha,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    spec_identity = _identity(spec.calibration_spec_id, spec)
    selections = tuple(
        NativeHoldCalibrationSelectedAnchor(
            selection_id=f'selection.tokamak-control.finite-action-recurrence.{slot.slot_id}',
            slot_id=slot.slot_id,
            selected=slot.primary,
            selection_kind=NativeHoldAnchorSelectionKind.PRIMARY,
        )
        for slot in slots
    )
    selection_receipt = _identity('selection-receipt.tokamak-control.finite-action-recurrence', freeze)
    link = EvidenceLink(
        link_id='link.tokamak-control.native-hold-calibration-selection',
        relation=EvidenceRelation.DERIVED_FROM,
        source=selection_receipt,
        target=spec_identity,
        artifact_ids=(evaluator_payload.artifact_id,),
        world_id='world.tokamak-control.gym-torax',
        information_cutoff_id='cutoff.tokamak-control.pre-hfr-calibration',
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
        reason="Outcome-blind selection of the five predeclared native-HOLD anchors.",
    )
    roster = NativeHoldCalibrationSelectedRoster(
        roster_id='roster.tokamak-control.native-hold-calibration',
        calibration_spec=spec_identity,
        selections=selections,
        selection_receipt=selection_receipt,
        input_artifacts=(evaluator_payload,),
        evidence_links=(link,),
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
    )
    return spec, roster


def _calibrate_native_hold(
    *,
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    episodes: Mapping[str, GymToraxFieldMetadataNativeEpisode],
    manifests: Mapping[str, ArtifactManifest],
    metadata: GymToraxFieldMetadataManifest,
) -> NativeHoldDecisionCellCalibrationReceipt:
    spec, roster = _calibration_spec_and_roster(freeze=freeze)
    spec_identity = _identity(spec.calibration_spec_id, spec)
    roster_identity = _identity(roster.roster_id, roster)
    selection_by_slot = {value.slot_id: value for value in roster.selections}
    cells = tuple(
        value
        for value in freeze.cells
        if value.stage is GymToraxFiniteActionRecurrenceCellStage.HOLD_CALIBRATION
    )
    units = {value.unit_id: value for value in freeze.units}
    evaluator_payload = spec.evaluator.payload
    occurrence_evidence: list[NativeHoldCalibrationOccurrenceEvidence] = []
    all_artifacts: dict[str, ArtifactIdentity] = {
        evaluator_payload.artifact_id: evaluator_payload
    }
    all_links: dict[str, EvidenceLink] = {}
    for cell in cells:
        unit = units[cell.unit_id]
        slot_id = unit.source_anchor_slot_id
        if slot_id is None:
            raise ValueError("Finite-action recurrence calibration cell lacks its anchor slot")
        episode = episodes[cell.episode_id]
        _validate_hfr_episode(
            freeze=freeze, cell=cell, episode=episode, metadata=metadata
        )
        manifest = manifests[f"artifact.{cell.episode_id}"]
        episode_artifact = _artifact(manifest)
        all_artifacts[episode_artifact.artifact_id] = episode_artifact
        evidence_artifacts = tuple(
            sorted(
                (evaluator_payload, episode_artifact),
                key=lambda value: value.artifact_id,
            )
        )
        link = EvidenceLink(
            link_id=f"link.{cell.episode_id}",
            relation=EvidenceRelation.DERIVED_FROM,
            source=_episode_identity(episode),
            target=spec_identity,
            artifact_ids=tuple(value.artifact_id for value in evidence_artifacts),
            world_id='world.tokamak-control.gym-torax',
            information_cutoff_id='cutoff.tokamak-control.state-0104',
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
            parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
            reason="Exact fresh native-HOLD calibration episode and code-owned evaluator.",
        )
        all_links[link.link_id] = link
        observations = tuple(
            sorted(
                (
                    NativeHoldCalibrationObservedPredicate(
                        observation_id=f"observation.{cell.episode_id}.{predicate.predicate_id}",
                        predicate_id=predicate.predicate_id,
                        operand_kind=NativeHoldCalibrationOperandKind.BOOLEAN,
                        observed_scalar=None,
                        observed_boolean=True,
                        observed_identity=None,
                    )
                    for predicate in spec.calibration_predicates
                ),
                key=lambda value: value.predicate_id,
            )
        )
        values = _preparation_values(unit)
        occurrence_evidence.append(
            NativeHoldCalibrationOccurrenceEvidence(
                evidence_id=f"evidence.{cell.episode_id}",
                calibration_spec=spec_identity,
                selected_roster=roster_identity,
                selected_anchor=selection_by_slot[slot_id],
                observed_preparation_values=values,
                observed_preparation_fingerprint=preparation_values_fingerprint(values),
                model_member_id=cell.model_member_id,
                occurrence_role_id=FINITE_ACTION_RECURRENCE_CALIBRATION_OCCURRENCE_ROLE_ID,
                source=spec.source,
                schedule=spec.schedule,
                native_hold_action_word=_identity(
                    spec.native_hold_action_word.word_id,
                    spec.native_hold_action_word,
                ),
                retained_history=spec.retained_history,
                horizon=spec.horizon,
                observed_occurrences=_observed(
                    spec.native_hold_action_word,
                    f"calibration.{cell.episode_id}",
                ),
                predicate_observations=observations,
                input_artifacts=evidence_artifacts,
                evidence_links=(link,),
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            )
        )
    selection_links = {value.link_id: value for value in roster.evidence_links}
    projection = NativeHoldActionLocalProjection(
        projection_id='projection.tokamak-control.native-hold-calibration',
        calibration_spec=spec_identity,
        selected_roster=roster_identity,
        occurrence_evidence=tuple(
            sorted(occurrence_evidence, key=lambda value: value.evidence_id)
        ),
        input_artifacts=tuple(all_artifacts[key] for key in sorted(all_artifacts)),
        evidence_links=tuple(
            {**selection_links, **all_links}[key]
            for key in sorted({**selection_links, **all_links})
        ),
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )
    return NativeHoldDecisionCellCalibrationEvaluator(spec).evaluate(
        receipt_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_CALIBRATION_RECEIPT_ID,
        spec=spec,
        selected_roster=roster,
        projection=projection,
    )


def _route_qualification(
    *,
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    occurrence_receipts: tuple[FiniteActionOccurrenceQualificationReceipt, ...],
    calibration: NativeHoldDecisionCellCalibrationReceipt,
    episodes: Mapping[str, GymToraxFieldMetadataNativeEpisode],
    metadata: GymToraxFieldMetadataManifest,
) -> FiniteActionRecurrenceRouteQualificationReceipt:
    template = freeze.corrected_recurrence_template
    units = {value.unit_id: value for value in freeze.units}
    role_map = {
        "active": (
            FiniteActionRouteCanaryCoordinateRole.ACTIVE_COORDINATE,
            FiniteActionRouteCanaryEpisodeRole.ACTIVE,
        ),
        "matched-hold": (
            FiniteActionRouteCanaryCoordinateRole.ACTIVE_COORDINATE,
            FiniteActionRouteCanaryEpisodeRole.MATCHED_HOLD,
        ),
        "hold-control": (
            FiniteActionRouteCanaryCoordinateRole.HOLD_ANCHOR_01,
            FiniteActionRouteCanaryEpisodeRole.HOLD_CONTROL,
        ),
        "hold-qualification-repeat": (
            FiniteActionRouteCanaryCoordinateRole.HOLD_ANCHOR_01,
            FiniteActionRouteCanaryEpisodeRole.HOLD_QUALIFICATION_REPEAT,
        ),
    }
    canaries: list[FiniteActionRouteCanaryEpisodeReceipt] = []
    for cell in freeze.cells:
        if cell.stage is not GymToraxFiniteActionRecurrenceCellStage.ROUTE_QUALIFICATION:
            continue
        episode = episodes[cell.episode_id]
        _validate_hfr_episode(
            freeze=freeze, cell=cell, episode=episode, metadata=metadata
        )
        unit = units[cell.unit_id]
        coordinate_role, episode_role = role_map[cell.route_role]
        word = (
            template.active_word
            if episode_role is FiniteActionRouteCanaryEpisodeRole.ACTIVE
            else template.matched_hold_word
        )
        canaries.append(
            FiniteActionRouteCanaryEpisodeReceipt(
                canary_episode_id=cell.episode_id,
                execution_id=cell.execution_id,
                physical_unit_id=cell.unit_id,
                model_member_id=cell.model_member_id,
                coordinate_role=coordinate_role,
                episode_role=episode_role,
                action_word=_identity(word.word_id, word),
                preparation_fingerprint=preparation_values_fingerprint(
                    _preparation_values(unit)
                ),
                preaction_prefix_sha256=_prefix_sha(episode),
                preaction_disposition=FiniteActionPreactionDisposition.READY,
                delivery_status=GateStatus.PASS,
                publication_recovery_status=GateStatus.PASS,
                seal_reveal_decoding_status=GateStatus.PASS,
                delivery_trace=_episode_identity(episode),
                evidence_identities=(_episode_identity(episode),),
                reason_codes=(),
            )
        )
    return FiniteActionRecurrenceRouteQualificationReceipt(
        receipt_id=GYM_TORAX_FINITE_ACTION_RECURRENCE_ROUTE_QUALIFICATION_RECEIPT_ID,
        branch_selection=freeze.selection_bridge,
        occurrence_receipts=tuple(
            sorted(
                (_identity(value.receipt_id, value) for value in occurrence_receipts),
                key=lambda value: value.object_id,
            )
        ),
        native_hold_calibration=_identity(calibration.receipt_id, calibration),
        active_local_support_id='local-support.tokamak-control.joint-depth-middle',
        hold_anchor_slot_id=FINITE_ACTION_RECURRENCE_CONTROL_ANCHOR_SLOT_IDS[0],
        model_member_ids=template.model_member_ids,
        episodes=tuple(sorted(canaries, key=lambda value: value.canary_episode_id)),
        nonattempt_refusal_conformance=GateStatus.PASS,
        evaluator_decoding_conformance=GateStatus.PASS,
        disposition=FiniteActionRouteCanaryDisposition.QUALIFIED,
        reason_codes=(),
        g2_operator_feasibility=finite_action_branch_operator_identity(
            freeze.selection_bridge
        ),
        g2_controlling_reason_code=freeze.selection_bridge.controlling_reason_code,
        claim_bearing=False,
        evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


def evaluate_gym_torax_finite_action_recurrence_qualification(
    *,
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    selected_parent_freeze: GymToraxSelectedParentFreeze,
    selected_parent_terminal: GymToraxSelectedParentTerminalAcquisitionReceipt,
    selected_parent_science: GymToraxSelectedParentScienceEvaluationBundle,
    hfr_stage_terminal: GymToraxFiniteActionRecurrenceStageAcquisitionReceipt,
    selected_parent_episode_manifests: tuple[ArtifactManifest, ...],
    hfr_episode_manifests: tuple[ArtifactManifest, ...],
    selected_parent_episode_loader: GymToraxFieldMetadataEpisodeLoader,
    hfr_episode_loader: GymToraxFieldMetadataEpisodeLoader,
    metadata: GymToraxFieldMetadataManifest,
    selected_parent_reveal_authority: StudyOperationAuthority,
    selected_parent_execution_authority: StudyOperationAuthority,
    route_reveal_authority: StudyOperationAuthority,
    hfr_execution_authority: StudyOperationAuthority,
    at_utc: str,
) -> GymToraxFiniteActionRecurrenceQualificationBundle:
    "Reveal only the exact 144 measurement and 18 qualification-stage episodes."

    if (
        selected_parent_terminal.freeze != _identity(selected_parent_freeze.freeze_id, selected_parent_freeze)
        or selected_parent_terminal.completed_episode_count != 576
        or selected_parent_terminal.failed_cell_count
    ):
        raise ValueError("Finite-action recurrence occurrence source lacks complete measurement terminal accounting")
    if (
        hfr_stage_terminal.freeze != _identity(freeze.freeze_id, freeze)
        or hfr_stage_terminal.stages
        != (
            GymToraxFiniteActionRecurrenceCellStage.HOLD_CALIBRATION,
            GymToraxFiniteActionRecurrenceCellStage.ROUTE_QUALIFICATION,
        )
        or hfr_stage_terminal.completed_episode_count != 18
        or hfr_stage_terminal.failed_cell_count
        or hfr_stage_terminal.disposition
        is not GymToraxFiniteActionRecurrenceTerminalAcquisitionDisposition.COMPLETE
    ):
        raise ValueError("Finite-action recurrence qualification lacks its exact complete 10+8 terminal")
    _require_reveal(
        authority=selected_parent_reveal_authority,
        authority_id=_FINITE_ACTION_RECURRENCE_OCCURRENCE_REVEAL_AUTHORITY_ID,
        subject=_identity(selected_parent_terminal.receipt_id, selected_parent_terminal),
        prerequisite=_identity(
            selected_parent_execution_authority.authority_id, selected_parent_execution_authority
        ),
        at_utc=at_utc,
    )
    _require_reveal(
        authority=route_reveal_authority,
        authority_id=_FINITE_ACTION_RECURRENCE_ROUTE_REVEAL_AUTHORITY_ID,
        subject=_identity(hfr_stage_terminal.receipt_id, hfr_stage_terminal),
        prerequisite=_identity(
            hfr_execution_authority.authority_id, hfr_execution_authority
        ),
        at_utc=at_utc,
    )
    selected_parent_cells = tuple(
        sorted(
            (
                value
                for value in selected_parent_freeze.cells
                if next(
                    unit
                    for unit in selected_parent_freeze.units
                    if unit.unit_id == value.physical_unit_instance_id
                ).cell_kind
                == "bt"
            ),
            key=lambda value: value.episode_id,
        )
    )
    if len(selected_parent_cells) != 144:
        raise ValueError("Finite-action recurrence occurrence source changes its exact 144 measurement roster")
    selected_parent_episodes, _ = _load_exact(
        expected_ids=tuple(value.episode_id for value in selected_parent_cells),
        manifests=selected_parent_episode_manifests,
        loader=selected_parent_episode_loader,
    )
    hfr_cells = tuple(
        sorted(
            (
                value
                for value in freeze.cells
                if value.stage
                in {
                    GymToraxFiniteActionRecurrenceCellStage.HOLD_CALIBRATION,
                    GymToraxFiniteActionRecurrenceCellStage.ROUTE_QUALIFICATION,
                }
            ),
            key=lambda value: value.episode_id,
        )
    )
    hfr_episodes, hfr_manifests = _load_exact(
        expected_ids=tuple(value.episode_id for value in hfr_cells),
        manifests=hfr_episode_manifests,
        loader=hfr_episode_loader,
    )
    law_qualification = selected_parent_science.evaluation.qualification_result
    occurrence = _occurrence_qualification(
        freeze=freeze,
        selected_parent_freeze=selected_parent_freeze,
        law_qualification_result=law_qualification,
        selected_parent_episodes=selected_parent_episodes,
        metadata=metadata,
    )
    calibration = _calibrate_native_hold(
        freeze=freeze,
        episodes=hfr_episodes,
        manifests=hfr_manifests,
        metadata=metadata,
    )
    route = _route_qualification(
        freeze=freeze,
        occurrence_receipts=occurrence,
        calibration=calibration,
        episodes=hfr_episodes,
        metadata=metadata,
    )
    assignment, binding = FiniteActionRecurrenceAssignmentTemplateBinder(
        implementation_id='binder.tokamak-control.finite-action-assignment',
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
    ).bind(
        receipt_id='binding-receipt.assignment.tokamak-control.finite-action-recurrence',
        template=freeze.corrected_recurrence_template,
        branch_selection=freeze.selection_bridge,
        occurrence_receipts=occurrence,
        native_hold_calibration=calibration,
        route_qualification=route,
    )
    return GymToraxFiniteActionRecurrenceQualificationBundle(
        bundle_id='qualification-bundle.tokamak-control.finite-action-recurrence',
        freeze=_identity(freeze.freeze_id, freeze),
        law_qualification_result=_identity(law_qualification.result_id, law_qualification),
        selected_parent_occurrence_reveal_authority=_identity(
            selected_parent_reveal_authority.authority_id, selected_parent_reveal_authority
        ),
        route_reveal_authority=_identity(
            route_reveal_authority.authority_id, route_reveal_authority
        ),
        occurrence_receipts=occurrence,
        native_hold_calibration=calibration,
        route_qualification=route,
        assignment=assignment,
        assignment_binding=binding,
        selected_parent_episode_count_read=144,
        hfr_stage_episode_count_read=18,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


def _recurrence_sealed_bundle(
    *,
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    qualification: GymToraxFiniteActionRecurrenceQualificationBundle,
    episodes: Mapping[str, GymToraxFieldMetadataNativeEpisode],
    manifests: Mapping[str, ArtifactManifest],
    metadata: GymToraxFieldMetadataManifest,
) -> tuple[FiniteActionRecurrenceProspectiveBundle, dict[str, np.ndarray]]:
    assignment = qualification.assignment
    route_by_episode = {value.episode_id: value for value in assignment.routes}
    cell_by_episode = {
        value.episode_id: value
        for value in freeze.cells
        if value.stage is GymToraxFiniteActionRecurrenceCellStage.RECURRENCE
    }
    unit_by_physical = {value.unit_id: value for value in freeze.units}
    member_by_id = {value.member_id: value for value in freeze.numerical_members}
    q_values: dict[str, np.ndarray] = {}
    sealed: list[FiniteActionRecurrenceSealedEpisode] = []
    for episode_id, route in route_by_episode.items():
        cell = cell_by_episode[episode_id]
        episode = episodes[episode_id]
        q_values[episode_id] = _validate_hfr_episode(
            freeze=freeze,
            cell=cell,
            episode=episode,
            metadata=metadata,
        )
        physical = unit_by_physical[cell.unit_id]
        member = member_by_id[cell.model_member_id]
        config_sha = hashlib.sha256(
            (
                physical.preparation.fingerprint()
                + ":"
                + member.fingerprint()
                + ':tokamak-control.action-neutral-config'
            ).encode()
        ).hexdigest()
        manifest = manifests[f"artifact.{episode_id}"]
        sealed.append(
            FiniteActionRecurrenceSealedEpisode(
                episode_id=episode_id,
                route=route,
                preparation_artifact=_content_artifact(
                    artifact_id=f"artifact.preparation.{episode_id}",
                    role="preparation",
                    payload_schema='empirical-lawhood/simulators/gym-torax-native/finite-action-recurrence/preparation-artifact',
                    sha256=next(
                        value.preparation_sha256
                        for value in assignment.units
                        if value.unit_id == route.unit_id
                    ),
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
                prefix_complete_through_state_104=True,
                preaction_disposition=FiniteActionPreactionDisposition.READY,
                observed_occurrences=_observed(
                    (
                        assignment.active_word
                        if route.role is FiniteActionRecurrenceRouteRole.ACTIVE
                        else assignment.matched_hold_word
                    ),
                    episode_id,
                ),
                clipped=False,
                rejected=False,
                substituted=False,
                early_terminated=False,
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
    return (
        FiniteActionRecurrenceProspectiveBundle(
            bundle_id='sealed-bundle.tokamak-control.finite-action-recurrence',
            assignment=assignment,
            episodes=tuple(sorted(sealed, key=lambda value: value.episode_id)),
            issue_receipt=_identity(qualification.bundle_id, qualification),
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        ),
        q_values,
    )


def _recurrence_reveal(
    *,
    sealed: FiniteActionRecurrenceProspectiveBundle,
    q_values: Mapping[str, np.ndarray],
    reveal_authority: StudyOperationAuthority,
    terminal: GymToraxFiniteActionRecurrenceTerminalAcquisitionReceipt,
) -> FiniteActionRecurrenceRevealedBundle:
    assignment = sealed.assignment
    predicate_kinds = tuple(
        sorted(
            {
                predicate.kind
                for receipt in assignment.occurrence_receipts
                for predicate in receipt.qualification_spec.predicates
            },
            key=lambda value: value.value,
        )
    )
    if len(predicate_kinds) != 7:
        raise ValueError("Finite-action recurrence lost its seven noncompensating predicates")
    revealed: list[FiniteActionRecurrenceRevealedEpisode] = []
    for source in sealed.episodes:
        route = source.route
        q = q_values[source.episode_id]
        samples = (
            ()
            if route.role is FiniteActionRecurrenceRouteRole.HOLD_CONTROL
            else tuple(
                FiniteActionRecurrenceResponseSample(
                    sample_id=f"sample.{source.episode_id}.{index:02d}",
                    quantity_id=assignment.effect_quantity_id,
                    coordinate=coordinate,
                    value=NamedDecimal(
                        value_id=f"value.{source.episode_id}.{index:02d}",
                        value=Decimal(str(float(q[int(coordinate.coordinate)]))),
                        unit=assignment.effect_native_unit,
                    ),
                )
                for index, coordinate in enumerate(assignment.response_coordinates)
            )
        )
        predicates = tuple(
            sorted(
                (
                    FiniteActionRecurrencePredicateResult(
                        result_id=(
                            f"predicate-result.{source.episode_id}."
                            f"{kind.value.lower().replace('_', '-')}"
                        ),
                        kind=kind,
                        status=GateStatus.PASS,
                        reason_codes=(),
                    )
                    for kind in predicate_kinds
                ),
                key=lambda value: value.result_id,
            )
        )
        revealed.append(
            FiniteActionRecurrenceRevealedEpisode(
                reveal_id=f"reveal.{source.episode_id}",
                episode_id=source.episode_id,
                sealed_episode=_identity(source.episode_id, source),
                delivery_trace=source.delivery_trace,
                sealed_outcome_artifact=source.sealed_outcome_artifact,
                outcome_trace_id=f"outcome-trace.{source.episode_id}",
                response_samples=samples,
                predicate_results=predicates,
                technical_reason_codes=(),
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            )
        )
    return FiniteActionRecurrenceRevealedBundle(
        reveal_id='revealed-bundle.tokamak-control.finite-action-recurrence',
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
        episodes=tuple(sorted(revealed, key=lambda value: value.episode_id)),
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
    )


def evaluate_gym_torax_finite_action_recurrence_recurrence(
    *,
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    qualification: GymToraxFiniteActionRecurrenceQualificationBundle,
    terminal: GymToraxFiniteActionRecurrenceTerminalAcquisitionReceipt,
    episode_manifests: tuple[ArtifactManifest, ...],
    episode_loader: GymToraxFieldMetadataEpisodeLoader,
    metadata: GymToraxFieldMetadataManifest,
    reveal_authority: StudyOperationAuthority,
    execution_authority: StudyOperationAuthority,
    at_utc: str,
) -> GymToraxFiniteActionRecurrenceRecurrenceEvaluationBundle:
    """Reveal and adjudicate the exact assigned 80-episode recurrence product."""

    if (
        terminal.freeze != _identity(freeze.freeze_id, freeze)
        or terminal.disposition is not GymToraxFiniteActionRecurrenceTerminalAcquisitionDisposition.COMPLETE
        or terminal.completed_episode_count != 98
        or terminal.failed_cell_count
    ):
        raise ValueError("Finite-action recurrence lacks complete 98-cell terminal accounting")
    if qualification.freeze != _identity(freeze.freeze_id, freeze):
        raise ValueError("Finite-action recurrence qualification binds another freeze")
    _require_reveal(
        authority=reveal_authority,
        authority_id=_FINITE_ACTION_RECURRENCE_RECURRENCE_REVEAL_AUTHORITY_ID,
        subject=_identity(terminal.receipt_id, terminal),
        prerequisite=_identity(execution_authority.authority_id, execution_authority),
        at_utc=at_utc,
    )
    cells = tuple(
        sorted(
            (
                value
                for value in freeze.cells
                if value.stage is GymToraxFiniteActionRecurrenceCellStage.RECURRENCE
            ),
            key=lambda value: value.episode_id,
        )
    )
    episodes, manifests = _load_exact(
        expected_ids=tuple(value.episode_id for value in cells),
        manifests=episode_manifests,
        loader=episode_loader,
    )
    sealed, q_values = _recurrence_sealed_bundle(
        freeze=freeze,
        qualification=qualification,
        episodes=episodes,
        manifests=manifests,
        metadata=metadata,
    )
    revealed = _recurrence_reveal(
        sealed=sealed,
        q_values=q_values,
        reveal_authority=reveal_authority,
        terminal=terminal,
    )
    adjudication = MatchedFiniteActionHoldRecurrenceEvaluator().evaluate(
        assignment=qualification.assignment,
        revealed=revealed,
    )
    return GymToraxFiniteActionRecurrenceRecurrenceEvaluationBundle(
        bundle_id='evaluation-bundle.tokamak-control.finite-action-recurrence',
        freeze=_identity(freeze.freeze_id, freeze),
        qualification=_identity(qualification.bundle_id, qualification),
        terminal_acquisition=_identity(terminal.receipt_id, terminal),
        reveal_authority=_identity(reveal_authority.authority_id, reveal_authority),
        sealed_bundle=sealed,
        revealed_bundle=revealed,
        adjudication=adjudication,
        episode_count_read=80,
        outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
    )


def _require_hfr_custody(
    *,
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> None:
    require_study_authority(
        authority,
        kind=StudyAuthorityKind.CUSTODY_PUBLICATION,
        subject=_identity(freeze.freeze_id, freeze),
        prerequisite_authority=None,
        grantee_id="operator.execution-service",
        storage_root_id=store.storage_root_id,
        relative_root=freeze.relative_root,
        at_utc=at_utc,
    )


def publish_gym_torax_finite_action_recurrence_qualification(
    *,
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    bundle: GymToraxFiniteActionRecurrenceQualificationBundle,
    stage_terminal: GymToraxFiniteActionRecurrenceStageAcquisitionReceipt,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> ObjectIdentity:
    _require_hfr_custody(
        freeze=freeze,
        authority=custody_authority,
        store=store,
        at_utc=at_utc,
    )
    if bundle.freeze != _identity(freeze.freeze_id, freeze):
        raise ValueError("Finite-action recurrence qualification publication changes its freeze")
    parents = tuple(
        sorted(
            (
                ArtifactLineageParent(
                    identity=_identity(freeze.freeze_id, freeze),
                    visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                    outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                ),
                ArtifactLineageParent(
                    identity=_identity(stage_terminal.receipt_id, stage_terminal),
                    visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                    outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                ),
                ArtifactLineageParent(
                    identity=bundle.law_qualification_result,
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                ),
            ),
            key=lambda value: value.identity.object_id,
        )
    )
    return store.publish_atomic(
        publication_scope_id='publication.tokamak-control.finite-action-qualification',
        publication_scope_relative_root=freeze.relative_root,
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id='artifact.qualification-bundle.tokamak-control.finite-action-recurrence',
                relative_path=f"{freeze.relative_root}/science/qualification-bundle.json",
                record=bundle,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                lineage_parents=parents,
            ),
        ),
    )[0]


def publish_gym_torax_finite_action_recurrence_recurrence(
    *,
    freeze: GymToraxFiniteActionRecurrenceProspectiveFreeze,
    bundle: GymToraxFiniteActionRecurrenceRecurrenceEvaluationBundle,
    custody_authority: StudyOperationAuthority,
    store: GymToraxBoundedArtifactStore,
    at_utc: str,
) -> ObjectIdentity:
    _require_hfr_custody(
        freeze=freeze,
        authority=custody_authority,
        store=store,
        at_utc=at_utc,
    )
    if bundle.freeze != _identity(freeze.freeze_id, freeze):
        raise ValueError("Finite-action recurrence publication changes its freeze")
    parents = tuple(
        sorted(
            (
                ArtifactLineageParent(
                    identity=bundle.qualification,
                    visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                    outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                ),
                ArtifactLineageParent(
                    identity=bundle.terminal_acquisition,
                    visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                    outcome_access=OutcomeAccess.EVALUATION_REVEALED,
                ),
            ),
            key=lambda value: value.identity.object_id,
        )
    )
    return store.publish_atomic(
        publication_scope_id='publication.tokamak-control.finite-action-recurrence',
        publication_scope_relative_root=freeze.relative_root,
        implementation_sha256=freeze.implementation_source_closure.implementation_sha256,
        items=(
            GymToraxArtifactPublicationItem(
                logical_artifact_id='artifact.evaluation-bundle.tokamak-control.finite-action-recurrence',
                relative_path=(
                    f"{freeze.relative_root}/science/recurrence-evaluation-bundle.json"
                ),
                record=bundle,
                visibility_ceiling=VisibilityCeiling.OUTCOME_VISIBLE,
                outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
                lineage_parents=parents,
            ),
        ),
    )[0]


__all__ = [
    'GymToraxFiniteActionRecurrenceQualificationBundle',
    'GymToraxFiniteActionRecurrenceRecurrenceEvaluationBundle',
    'evaluate_gym_torax_finite_action_recurrence_qualification',
    'evaluate_gym_torax_finite_action_recurrence_recurrence',
    'publish_gym_torax_finite_action_recurrence_qualification',
    'publish_gym_torax_finite_action_recurrence_recurrence',
]

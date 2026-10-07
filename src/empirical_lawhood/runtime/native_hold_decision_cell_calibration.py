"""Sole reducer for finite native-HOLD decision-cell calibration evidence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import (
    ActionDeliveryStage,
    ActionOccurrence,
    ActionStageEvent,
    ObservedActionOccurrence,
)
from empirical_lawhood.planning.controller_study import DeliveryEquivalenceSpec, StageAwareDeliveryEquivalenceSpec
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import EvidenceLink, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    validate_stable_id,
)
from empirical_lawhood.planning.native_hold_decision_cell_calibration import NativeHoldAnchorSelectionKind, NativeHoldCalibrationDisposition, NativeHoldCalibrationOccurrenceReceipt, NativeHoldCalibrationPredicateKind, NativeHoldCalibrationPredicateReceipt, NativeHoldCalibrationSelectedAnchor, NativeHoldCalibrationSelectedRoster, NativeHoldDecisionCellCalibrationReceipt, NativeHoldDecisionCellCalibrationSpec, preparation_values_fingerprint
from empirical_lawhood.planning.source_pipelines import SourceRuntimeQualificationBindingReceipt


class NativeHoldCalibrationOperandKind(StrEnum):
    SCALAR = "SCALAR"
    BOOLEAN = "BOOLEAN"
    IDENTITY = "IDENTITY"


@dataclass(frozen=True, slots=True)
class NativeHoldCalibrationObservedPredicate(CanonicalRecord):
    """One revealed native operand, without an evaluator-authored status."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/native-hold-calibration-observed-predicate'

    observation_id: str
    predicate_id: str
    operand_kind: NativeHoldCalibrationOperandKind
    observed_scalar: NamedDecimal | None
    observed_boolean: bool | None
    observed_identity: ObjectIdentity | None

    def __post_init__(self) -> None:
        validate_stable_id(self.observation_id, field_name="observation_id")
        validate_stable_id(self.predicate_id, field_name="predicate_id")
        supplied = sum(
            value is not None
            for value in (
                self.observed_scalar,
                self.observed_boolean,
                self.observed_identity,
            )
        )
        if supplied != 1:
            raise ValueError("native-HOLD predicate observation requires one operand")
        expected_kind = (
            NativeHoldCalibrationOperandKind.SCALAR
            if self.observed_scalar is not None
            else NativeHoldCalibrationOperandKind.BOOLEAN
            if self.observed_boolean is not None
            else NativeHoldCalibrationOperandKind.IDENTITY
        )
        if self.operand_kind is not expected_kind:
            raise ValueError("native-HOLD predicate operand kind differs from its value")


@dataclass(frozen=True, slots=True)
class NativeHoldCalibrationOccurrenceEvidence(CanonicalRecord):
    """Revealed action-local evidence for one slot/member/occurrence role."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/native-hold-calibration-occurrence-evidence'

    evidence_id: str
    calibration_spec: ObjectIdentity
    selected_roster: ObjectIdentity
    selected_anchor: NativeHoldCalibrationSelectedAnchor
    observed_preparation_values: tuple[NamedDecimal, ...]
    observed_preparation_fingerprint: str
    model_member_id: str
    occurrence_role_id: str
    source: ObjectIdentity
    schedule: ObjectIdentity
    native_hold_action_word: ObjectIdentity
    retained_history: ObjectIdentity
    horizon: ObjectIdentity
    observed_occurrences: tuple[ObservedActionOccurrence, ...]
    predicate_observations: tuple[NativeHoldCalibrationObservedPredicate, ...]
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.evidence_id, field_name="evidence_id")
        validate_stable_id(self.model_member_id, field_name="model_member_id")
        validate_stable_id(self.occurrence_role_id, field_name="occurrence_role_id")
        if self.calibration_spec.object_schema != NativeHoldDecisionCellCalibrationSpec.SCHEMA:
            raise ValueError("native-HOLD evidence binds another calibration spec")
        if self.selected_roster.object_schema != NativeHoldCalibrationSelectedRoster.SCHEMA:
            raise ValueError("native-HOLD evidence binds another selected roster")
        require_sorted_unique_ids(
            self.observed_preparation_values,
            attribute="value_id",
            field_name="observed_preparation_values",
        )
        if not self.observed_preparation_values:
            raise ValueError("native-HOLD evidence requires observed preparation values")
        if self.observed_preparation_fingerprint != preparation_values_fingerprint(
            self.observed_preparation_values
        ):
            raise ValueError("native-HOLD observed preparation fingerprint differs")
        identifiers = tuple(value.expected_occurrence_id for value in self.observed_occurrences)
        if identifiers != tuple(sorted(set(identifiers))):
            raise ValueError("observed native-HOLD occurrences must be sorted and unique")
        require_sorted_unique_ids(
            self.predicate_observations,
            attribute="predicate_id",
            field_name="predicate_observations",
        )
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if not self.input_artifacts or not self.evidence_links:
            raise ValueError("native-HOLD occurrence evidence requires authenticated inputs")
        artifact_ids = {value.artifact_id for value in self.input_artifacts}
        linked_artifacts = {
            artifact_id for link in self.evidence_links for artifact_id in link.artifact_ids
        }
        if artifact_ids != linked_artifacts:
            raise ValueError("native-HOLD occurrence artifacts differ from evidence links")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("native-HOLD occurrence evidence requires evaluator reveal")

    @property
    def entry_key(self) -> tuple[str, str, str]:
        return (
            self.selected_anchor.slot_id,
            self.model_member_id,
            self.occurrence_role_id,
        )


@dataclass(frozen=True, slots=True)
class NativeHoldActionLocalProjection(CanonicalRecord):
    """Bounded revealed calibration projection; missing entries stay explicit."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/native-hold-action-local-projection'

    projection_id: str
    calibration_spec: ObjectIdentity
    selected_roster: ObjectIdentity
    occurrence_evidence: tuple[NativeHoldCalibrationOccurrenceEvidence, ...]
    input_artifacts: tuple[ArtifactIdentity, ...]
    evidence_links: tuple[EvidenceLink, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.projection_id, field_name="projection_id")
        if self.calibration_spec.object_schema != NativeHoldDecisionCellCalibrationSpec.SCHEMA:
            raise ValueError("native-HOLD projection binds another calibration spec")
        if self.selected_roster.object_schema != NativeHoldCalibrationSelectedRoster.SCHEMA:
            raise ValueError("native-HOLD projection binds another selected roster")
        require_sorted_unique_ids(
            self.occurrence_evidence,
            attribute="evidence_id",
            field_name="occurrence_evidence",
        )
        if len({value.entry_key for value in self.occurrence_evidence}) != len(
            self.occurrence_evidence
        ):
            raise ValueError("native-HOLD projection duplicates a slot/member/role entry")
        require_sorted_unique_ids(
            self.input_artifacts,
            attribute="artifact_id",
            field_name="input_artifacts",
        )
        require_sorted_unique_ids(
            self.evidence_links,
            attribute="link_id",
            field_name="evidence_links",
        )
        if not self.input_artifacts or not self.evidence_links:
            raise ValueError("native-HOLD projection requires exact artifacts and evidence")
        artifacts = {value.artifact_id for value in self.input_artifacts}
        linked = {value for link in self.evidence_links for value in link.artifact_ids}
        if artifacts != linked:
            raise ValueError("native-HOLD projection artifacts differ from its evidence")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("native-HOLD projection requires evaluator reveal")


def _merge_artifacts(
    *groups: tuple[ArtifactIdentity, ...],
) -> tuple[ArtifactIdentity, ...]:
    by_id: dict[str, ArtifactIdentity] = {}
    for value in (item for group in groups for item in group):
        previous = by_id.setdefault(value.artifact_id, value)
        if previous != value:
            raise ValueError("artifact ID is reused with different native-HOLD evidence")
    return tuple(by_id[key] for key in sorted(by_id))


def _merge_links(*groups: tuple[EvidenceLink, ...]) -> tuple[EvidenceLink, ...]:
    by_id: dict[str, EvidenceLink] = {}
    for value in (item for group in groups for item in group):
        previous = by_id.setdefault(value.link_id, value)
        if previous != value:
            raise ValueError("evidence-link ID is reused with different native-HOLD evidence")
    return tuple(by_id[key] for key in sorted(by_id))


def _events_match(
    expected: ActionStageEvent,
    observed: ActionStageEvent,
    *,
    tolerance: NamedDecimal,
) -> bool:
    return (
        observed.stage is expected.stage
        and observed.quantity_id == expected.quantity_id
        and observed.native_unit == expected.native_unit == tolerance.unit
        and observed.native_action_frame == expected.native_action_frame
        and observed.native_direction == expected.native_direction
        and observed.coordinate == expected.coordinate
        and abs(observed.value - expected.value) <= tolerance.value
    )


def _observed_occurrence_status(
    expected: tuple[ActionOccurrence, ...],
    observed: tuple[ObservedActionOccurrence, ...],
    *,
    equivalence: DeliveryEquivalenceSpec | StageAwareDeliveryEquivalenceSpec,
) -> tuple[NativeHoldCalibrationDisposition, tuple[str, ...]]:
    expected_by_id = {value.occurrence_id: value for value in expected}
    observed_by_id = {value.expected_occurrence_id: value for value in observed}
    if set(observed_by_id) != set(expected_by_id):
        return (
            NativeHoldCalibrationDisposition.UNEVALUABLE,
            ("NATIVE_HOLD_OCCURRENCE_ROSTER_INCOMPLETE",),
        )
    if any(not value.complete for value in observed):
        return (
            NativeHoldCalibrationDisposition.UNEVALUABLE,
            ("NATIVE_HOLD_DELIVERY_STAGE_MISSING",),
        )
    reasons: set[str] = set()
    for occurrence_id, expected_occurrence in expected_by_id.items():
        available = observed_by_id[occurrence_id]
        events = (
            available.requested,
            available.accepted,
            available.applied,
            available.realized,
        )
        if any(value is None for value in events):  # pragma: no cover - complete narrowed above
            raise AssertionError("complete native-HOLD occurrence lost a delivery stage")
        expected_events = (
            expected_occurrence.requested,
            expected_occurrence.accepted,
            expected_occurrence.applied,
            expected_occurrence.realized,
        )
        if not all(
            _events_match(
                expected_event,
                observed_event,
                tolerance=(
                    equivalence.stage_value_tolerance
                    if isinstance(equivalence, DeliveryEquivalenceSpec)
                    else equivalence.tolerance_for(ActionDeliveryStage(expected_event.stage))
                ),
            )
            for expected_event, observed_event in zip(
                expected_events,
                events,
                strict=True,
            )
            if observed_event is not None
        ):
            reasons.add("NATIVE_HOLD_DELIVERY_NOT_EQUIVALENT")
        if available.reason_codes:
            reasons.add("NATIVE_HOLD_DELIVERY_REPORTED_INVALID")
    return (
        (
            NativeHoldCalibrationDisposition.NOT_SUPPORTED
            if reasons
            else NativeHoldCalibrationDisposition.SUPPORTED
        ),
        tuple(sorted(reasons)),
    )


def _predicate_receipts(
    *,
    evidence_id: str,
    spec: NativeHoldDecisionCellCalibrationSpec,
    observations: tuple[NativeHoldCalibrationObservedPredicate, ...],
) -> tuple[NativeHoldCalibrationPredicateReceipt, ...]:
    observed = {value.predicate_id: value for value in observations}
    receipts: list[NativeHoldCalibrationPredicateReceipt] = []
    for predicate in spec.calibration_predicates:
        value = observed.get(predicate.predicate_id)
        scalar = value.observed_scalar if value is not None else None
        boolean = value.observed_boolean if value is not None else None
        identity = value.observed_identity if value is not None else None
        expected_operand_kind = (
            NativeHoldCalibrationOperandKind.SCALAR
            if predicate.predicate_kind
            in {
                NativeHoldCalibrationPredicateKind.SCALAR_GREATER_THAN,
                NativeHoldCalibrationPredicateKind.SCALAR_AT_LEAST,
                NativeHoldCalibrationPredicateKind.SCALAR_AT_MOST,
                NativeHoldCalibrationPredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL,
            }
            else NativeHoldCalibrationOperandKind.BOOLEAN
            if predicate.predicate_kind is NativeHoldCalibrationPredicateKind.BOOLEAN_EQUALS
            else NativeHoldCalibrationOperandKind.IDENTITY
        )
        if value is not None and value.operand_kind is not expected_operand_kind:
            scalar = None
            boolean = None
            identity = None
        status = GateStatus.UNEVALUABLE
        reasons = ("CALIBRATION_OPERAND_UNAVAILABLE",)
        if sum(item is not None for item in (scalar, boolean, identity)) == 1:
            try:
                probe = NativeHoldCalibrationPredicateReceipt(
                    receipt_id=f"predicate-receipt.{evidence_id}.{predicate.predicate_id}",
                    predicate=predicate,
                    observed_scalar=scalar,
                    observed_boolean=boolean,
                    observed_identity=identity,
                    status=GateStatus.PASS,
                    reason_codes=(),
                )
            except ValueError:
                try:
                    probe = NativeHoldCalibrationPredicateReceipt(
                        receipt_id=f"predicate-receipt.{evidence_id}.{predicate.predicate_id}",
                        predicate=predicate,
                        observed_scalar=scalar,
                        observed_boolean=boolean,
                        observed_identity=identity,
                        status=GateStatus.FAIL,
                        reason_codes=("CALIBRATION_PREDICATE_FAILED",),
                    )
                except ValueError:
                    scalar = None
                    boolean = None
                    identity = None
                else:
                    receipts.append(probe)
                    continue
            else:
                receipts.append(probe)
                continue
            status = GateStatus.UNEVALUABLE
            reasons = ("CALIBRATION_OPERAND_UNAVAILABLE",)
        if scalar is None and boolean is None and identity is None:
            receipts.append(
                NativeHoldCalibrationPredicateReceipt(
                    receipt_id=f"predicate-receipt.{evidence_id}.{predicate.predicate_id}",
                    predicate=predicate,
                    observed_scalar=None,
                    observed_boolean=None,
                    observed_identity=None,
                    status=status,
                    reason_codes=reasons,
                )
            )
            continue
        raise AssertionError("native-HOLD predicate evaluation did not terminate")
    return tuple(sorted(receipts, key=lambda value: value.receipt_id))


class NativeHoldDecisionCellCalibrationEvaluator:
    """Reduce the complete finite anchor/member/role product without pooling."""

    def __init__(self, spec: NativeHoldDecisionCellCalibrationSpec) -> None:
        self.evaluator = spec.evaluator
        self.implementation_sha256 = spec.evaluator_implementation_sha256

    @staticmethod
    def _validate_selected_roster(
        spec: NativeHoldDecisionCellCalibrationSpec,
        roster: NativeHoldCalibrationSelectedRoster,
    ) -> dict[str, NativeHoldCalibrationSelectedAnchor]:
        if roster.calibration_spec != ObjectIdentity.from_record(
            spec.calibration_spec_id,
            spec,
        ):
            raise ValueError("selected native-HOLD roster binds another exact spec")
        slots = {value.slot_id: value for value in spec.anchor_slots}
        selected = {value.slot_id: value for value in roster.selections}
        if set(selected) != set(slots):
            raise ValueError("selected native-HOLD roster omits or adds anchor slots")
        for slot_id, selection in selected.items():
            slot = slots[slot_id]
            expected = (
                slot.primary
                if selection.selection_kind is NativeHoldAnchorSelectionKind.PRIMARY
                else slot.conditional_alternative
            )
            if expected is None or selection.selected != expected:
                raise ValueError("selected native-HOLD anchor is not its predeclared option")
        return selected

    def evaluate(
        self,
        *,
        receipt_id: str,
        spec: NativeHoldDecisionCellCalibrationSpec,
        selected_roster: NativeHoldCalibrationSelectedRoster,
        projection: NativeHoldActionLocalProjection,
        source_runtime_qualification_binding: (
            SourceRuntimeQualificationBindingReceipt | None
        ) = None,
    ) -> NativeHoldDecisionCellCalibrationReceipt:
        validate_stable_id(receipt_id, field_name="receipt_id")
        if (
            spec.evaluator != self.evaluator
            or spec.evaluator_implementation_sha256 != self.implementation_sha256
        ):
            raise ValueError("native-HOLD evaluator differs from the issued implementation")
        spec_identity = ObjectIdentity.from_record(spec.calibration_spec_id, spec)
        roster_identity = ObjectIdentity.from_record(selected_roster.roster_id, selected_roster)
        if (
            projection.calibration_spec != spec_identity
            or projection.selected_roster != roster_identity
        ):
            raise ValueError("native-HOLD projection changes spec or selected roster")
        expected_source = spec.source
        source_runtime_binding_identity: ObjectIdentity | None = None
        if source_runtime_qualification_binding is not None:
            if (
                source_runtime_qualification_binding.source_expectation != spec.source
                or source_runtime_qualification_binding.runtime_expectation != spec.source
            ):
                raise ValueError("native-HOLD source/runtime mapping names another expectation")
            expected_source = source_runtime_qualification_binding.runtime_qualification
            source_runtime_binding_identity = ObjectIdentity.from_record(
                source_runtime_qualification_binding.receipt_id,
                source_runtime_qualification_binding,
            )
        if spec.evaluator.payload not in projection.input_artifacts:
            raise ValueError("native-HOLD projection omits the exact evaluator payload")
        selected = self._validate_selected_roster(spec, selected_roster)
        evidence_by_key = {value.entry_key: value for value in projection.occurrence_evidence}
        if not set(evidence_by_key) <= set(spec.expected_entry_keys):
            raise ValueError("native-HOLD projection contains an undeclared entry")

        expected_word_identity = ObjectIdentity.from_record(
            spec.native_hold_action_word.word_id,
            spec.native_hold_action_word,
        )
        receipts: list[NativeHoldCalibrationOccurrenceReceipt] = []
        for slot_id, member_id, role_id in spec.expected_entry_keys:
            anchor = selected[slot_id]
            key = (slot_id, member_id, role_id)
            evidence = evidence_by_key.get(key)
            entry_id = (
                evidence.evidence_id
                if evidence is not None
                else f"missing-evidence.{slot_id}.{member_id}.{role_id}"
            )
            predicate_receipts = _predicate_receipts(
                evidence_id=entry_id,
                spec=spec,
                observations=(evidence.predicate_observations if evidence is not None else ()),
            )
            reasons: set[str] = set()
            disposition = NativeHoldCalibrationDisposition.SUPPORTED
            if evidence is None:
                disposition = NativeHoldCalibrationDisposition.UNEVALUABLE
                reasons.add("NATIVE_HOLD_CALIBRATION_EVIDENCE_MISSING")
            else:
                observed_predicate_ids = tuple(
                    value.predicate_id for value in evidence.predicate_observations
                )
                expected_predicate_ids = tuple(
                    value.predicate_id for value in spec.calibration_predicates
                )
                if observed_predicate_ids != expected_predicate_ids:
                    disposition = NativeHoldCalibrationDisposition.UNEVALUABLE
                    reasons.add("NATIVE_HOLD_CALIBRATION_PREDICATE_ROSTER_MISMATCH")
                identity_match = (
                    evidence.calibration_spec == spec_identity
                    and evidence.selected_roster == roster_identity
                    and evidence.selected_anchor == anchor
                    and evidence.observed_preparation_values == anchor.selected.preparation_values
                    and evidence.observed_preparation_fingerprint
                    == anchor.selected.preparation_fingerprint
                    and evidence.model_member_id == member_id
                    and evidence.occurrence_role_id == role_id
                    and evidence.source == expected_source
                    and evidence.schedule == spec.schedule
                    and evidence.native_hold_action_word == expected_word_identity
                    and evidence.retained_history == spec.retained_history
                    and evidence.horizon == spec.horizon
                    and evidence.selected_anchor.selected.preparation_fingerprint
                    == preparation_values_fingerprint(
                        evidence.selected_anchor.selected.preparation_values
                    )
                )
                if not identity_match:
                    disposition = NativeHoldCalibrationDisposition.UNEVALUABLE
                    reasons.add("NATIVE_HOLD_CALIBRATION_IDENTITY_MISMATCH")
                delivery_disposition, delivery_reasons = _observed_occurrence_status(
                    spec.native_hold_action_word.occurrences,
                    evidence.observed_occurrences,
                    equivalence=spec.delivery_equivalence,
                )
                reasons.update(delivery_reasons)
                if delivery_disposition is NativeHoldCalibrationDisposition.UNEVALUABLE:
                    disposition = NativeHoldCalibrationDisposition.UNEVALUABLE
                elif (
                    delivery_disposition is NativeHoldCalibrationDisposition.NOT_SUPPORTED
                    and disposition is NativeHoldCalibrationDisposition.SUPPORTED
                ):
                    disposition = NativeHoldCalibrationDisposition.NOT_SUPPORTED
            if any(value.status is GateStatus.UNEVALUABLE for value in predicate_receipts):
                disposition = NativeHoldCalibrationDisposition.UNEVALUABLE
                reasons.add("NATIVE_HOLD_CALIBRATION_PREDICATE_UNEVALUABLE")
            elif any(value.status is GateStatus.FAIL for value in predicate_receipts):
                if disposition is NativeHoldCalibrationDisposition.SUPPORTED:
                    disposition = NativeHoldCalibrationDisposition.NOT_SUPPORTED
                reasons.add("NATIVE_HOLD_CALIBRATION_PREDICATE_FAILED")

            artifacts = _merge_artifacts(
                projection.input_artifacts,
                evidence.input_artifacts if evidence is not None else (),
            )
            links = _merge_links(
                projection.evidence_links,
                evidence.evidence_links if evidence is not None else (),
            )
            receipts.append(
                NativeHoldCalibrationOccurrenceReceipt(
                    receipt_id=f"occurrence-receipt.{slot_id}.{member_id}.{role_id}",
                    slot_id=slot_id,
                    coordinate_id=anchor.selected.coordinate_id,
                    preparation_fingerprint=anchor.selected.preparation_fingerprint,
                    model_member_id=member_id,
                    occurrence_role_id=role_id,
                    source=expected_source,
                    schedule=spec.schedule,
                    native_hold_action_word=expected_word_identity,
                    retained_history=spec.retained_history,
                    horizon=spec.horizon,
                    delivery_equivalence=spec.delivery_equivalence,
                    observed_occurrences=(
                        evidence.observed_occurrences if evidence is not None else ()
                    ),
                    expected_occurrence_ids=tuple(
                        sorted(
                            value.occurrence_id
                            for value in spec.native_hold_action_word.occurrences
                        )
                    ),
                    observed_occurrence_ids=(
                        tuple(
                            sorted(
                                value.expected_occurrence_id
                                for value in evidence.observed_occurrences
                            )
                        )
                        if evidence is not None
                        else ()
                    ),
                    expected_predicate_ids=tuple(
                        sorted(value.predicate_id for value in spec.calibration_predicates)
                    ),
                    predicate_receipts=predicate_receipts,
                    input_artifacts=artifacts,
                    evidence_links=links,
                    disposition=disposition,
                    reason_codes=tuple(sorted(reasons)),
                )
            )

        occurrence_receipts = tuple(sorted(receipts, key=lambda value: value.receipt_id))
        disposition = (
            NativeHoldCalibrationDisposition.UNEVALUABLE
            if any(
                value.disposition is NativeHoldCalibrationDisposition.UNEVALUABLE
                for value in occurrence_receipts
            )
            else NativeHoldCalibrationDisposition.NOT_SUPPORTED
            if any(
                value.disposition is NativeHoldCalibrationDisposition.NOT_SUPPORTED
                for value in occurrence_receipts
            )
            else NativeHoldCalibrationDisposition.SUPPORTED
        )
        aggregate_reasons: tuple[str, ...] = (
            ()
            if disposition is NativeHoldCalibrationDisposition.SUPPORTED
            else ("NATIVE_HOLD_CALIBRATION_ENTRY_UNEVALUABLE",)
            if disposition is NativeHoldCalibrationDisposition.UNEVALUABLE
            else ("NATIVE_HOLD_CALIBRATION_ENTRY_NOT_SUPPORTED",)
        )
        all_artifacts = _merge_artifacts(
            projection.input_artifacts,
            selected_roster.input_artifacts,
            *(value.input_artifacts for value in occurrence_receipts),
        )
        all_links = _merge_links(
            projection.evidence_links,
            selected_roster.evidence_links,
            *(value.evidence_links for value in occurrence_receipts),
        )
        return NativeHoldDecisionCellCalibrationReceipt(
            receipt_id=receipt_id,
            calibration_spec=spec_identity,
            selected_roster=roster_identity,
            hold_decision_cell_id=spec.hold_decision_cell_id,
            active_decision_cell_ids=spec.active_decision_cell_ids,
            selected_sibling_hold_candidate_key_id=(
                spec.sibling_hold_selector.selected_candidate_key_id
            ),
            expected_entries=spec.expected_entries,
            occurrence_receipts=occurrence_receipts,
            evaluator=self.evaluator,
            input_artifacts=all_artifacts,
            evidence_links=all_links,
            disposition=disposition,
            reason_codes=aggregate_reasons,
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            source_runtime_qualification_binding=source_runtime_binding_identity,
        )


__all__ = [
    'NativeHoldActionLocalProjection',
    'NativeHoldCalibrationOccurrenceEvidence',
    'NativeHoldCalibrationObservedPredicate',
    'NativeHoldCalibrationOperandKind',
    'NativeHoldDecisionCellCalibrationEvaluator',
]

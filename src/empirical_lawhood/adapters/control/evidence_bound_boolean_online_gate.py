"Evidence-bound Boolean continuation for the current admission/controller-use live-gate port."

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.admission import AdmissionGateKind, GateStatus
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import EvidenceLink, EvidenceRelation, ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    validate_stable_id,
)
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy, ImplementationBinding, ImplementationRole
from empirical_lawhood.planning.evidence_geometry import AdmissionCoordinateGateReceipt, GatePredicateKind, ReceiptAdmissionAdmissionCandidateCell, derive_admission_gate_observation
from empirical_lawhood.planning.native_hold_decision_cell_calibration import NativeHoldCalibrationDisposition, NativeHoldCalibrationSelectedRoster, NativeHoldDecisionCellCalibrationReceipt
from empirical_lawhood.planning.prospective_controller_study import ControllerImplementationBindingTemplate
from empirical_lawhood.runtime.controller_runtime import AdmissionLiveGateReceipt, AdmissionLiveGateEvaluation, RuntimeObservation

MAX_EVIDENCE_BOUND_BOOLEAN_ONLINE_GATE_CONFIG_TEMPLATE_BYTES = 1024 * 1024


class BooleanOnlineGateRequestKind(StrEnum):
    ACTIVE = "ACTIVE"
    HOLD = "HOLD"


@dataclass(frozen=True, slots=True)
class EvidenceBoundBooleanOnlineGateConfigTemplate(CanonicalRecord):
    "Outcome-blind exact config for scalar delegation and TARGET replay.\n\n    The record freezes the implementation port and refusal semantics without\n    fabricating the later admission receipts, selected anchor roster or aggregate\n    native-HOLD calibration.  Those eligible admission objects are supplied to the\n    evaluator only after their independently owned bindings exist.\n    "

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/evidence-bound-boolean-online-gate-config-template'

    config_id: str
    implementation_template: ControllerImplementationBindingTemplate
    scalar_delegate_id: str
    delegated_scalar_predicate_kinds: tuple[GatePredicateKind, ...]
    replay_gate_kind: AdmissionGateKind
    replay_predicate_kind: GatePredicateKind
    replay_expected_boolean: bool
    supported_request_kinds: tuple[BooleanOnlineGateRequestKind, ...]
    required_active_continuity_fields: tuple[str, ...]
    required_hold_continuity_fields: tuple[str, ...]
    offline_pass_status: GateStatus
    binding_failure_status: GateStatus
    immutable_offline_receipt_required: bool
    exact_complete_gate_grid_required: bool
    retain_offline_native_operands_margins_and_evidence: bool
    require_artifact_identity_compatibility: bool
    refuse_missing_boolean: bool
    refuse_non_target_boolean: bool
    refuse_identity_predicate: bool
    refuse_foreign_or_tampered_binding: bool
    post_cutoff_outcome_query_allowed: bool
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling

    def __post_init__(self) -> None:
        validate_stable_id(self.config_id, field_name="config_id")
        validate_stable_id(self.scalar_delegate_id, field_name="scalar_delegate_id")
        if (
            self.implementation_template.expected_config_id != self.config_id
            or self.implementation_template.expected_config_schema != self.SCHEMA
            or self.implementation_template.role is not ImplementationRole.ONLINE_GATE_EVALUATOR
        ):
            raise ValueError("online-gate config changes its implementation binding")
        scalar_kinds = (
            GatePredicateKind.SCALAR_AT_LEAST,
            GatePredicateKind.SCALAR_AT_MOST,
            GatePredicateKind.SCALAR_WITHIN_CLOSED_INTERVAL,
        )
        if self.delegated_scalar_predicate_kinds != tuple(
            sorted(scalar_kinds, key=lambda value: value.value)
        ):
            raise ValueError("online-gate config changes unchanged scalar delegation")
        if (
            self.replay_gate_kind is not AdmissionGateKind.TARGET
            or self.replay_predicate_kind is not GatePredicateKind.BOOLEAN_EQUALS
            or self.replay_expected_boolean is not True
            or self.supported_request_kinds
            != tuple(sorted(BooleanOnlineGateRequestKind, key=lambda value: value.value))
            or self.offline_pass_status is not GateStatus.PASS
            or self.binding_failure_status is not GateStatus.UNEVALUABLE
        ):
            raise ValueError("online-gate config changes exact Boolean replay semantics")
        require_sorted_unique_strings(
            self.required_active_continuity_fields,
            field_name="required_active_continuity_fields",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.required_hold_continuity_fields,
            field_name="required_hold_continuity_fields",
            allow_empty=False,
        )
        exact_active = tuple(
            sorted(
                (
                    "action-binding-and-word",
                    "candidate-and-decision-cell",
                    "complete-offline-coordinate-gate-grid",
                    "implementation-binding",
                    "controller-admission-candidate-cell",
                    "runtime-observation-artifacts",
                )
            )
        )
        exact_hold = tuple(
            sorted(
                (
                    *exact_active,
                    "aggregate-calibration-evidence",
                    "member-local-supported-occurrence",
                    "measured-hold-fibre",
                    "selected-anchor-preparation",
                    "selected-anchor-roster",
                    "selected-sibling-hold-controller-admission-cell",
                )
            )
        )
        if (
            self.required_active_continuity_fields != exact_active
            or self.required_hold_continuity_fields != exact_hold
        ):
            raise ValueError("online-gate config omits exact action/HOLD continuity")
        required_true = (
            self.immutable_offline_receipt_required,
            self.exact_complete_gate_grid_required,
            self.retain_offline_native_operands_margins_and_evidence,
            self.require_artifact_identity_compatibility,
            self.refuse_missing_boolean,
            self.refuse_non_target_boolean,
            self.refuse_identity_predicate,
            self.refuse_foreign_or_tampered_binding,
        )
        if (
            not all(required_true)
            or self.post_cutoff_outcome_query_allowed
            or self.evidence_ceiling is not EvidenceCeiling.NON_PROMOTABLE
            or self.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or self.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("online-gate config weakens refusal or outcome blindness")


def decode_evidence_bound_boolean_online_gate_config_template(
    payload: bytes,
) -> EvidenceBoundBooleanOnlineGateConfigTemplate:
    return decode_canonical_bytes(
        payload,
        EvidenceBoundBooleanOnlineGateConfigTemplate,
        maximum_bytes=MAX_EVIDENCE_BOUND_BOOLEAN_ONLINE_GATE_CONFIG_TEMPLATE_BYTES,
    )


def _candidate_cell(
    study: AdmissionControllerStudy,
    candidate_cell_id: str,
) -> ReceiptAdmissionAdmissionCandidateCell | None:
    return next(
        (
            value
            for value in study.admission.candidate_cells
            if value.candidate_cell_id == candidate_cell_id
        ),
        None,
    )


def _merge_links(*groups: tuple[EvidenceLink, ...]) -> tuple[EvidenceLink, ...]:
    by_id: dict[str, EvidenceLink] = {}
    for value in (item for group in groups for item in group):
        previous = by_id.setdefault(value.link_id, value)
        if previous != value:
            raise ValueError("live-gate evidence ID is reused with different content")
    return tuple(by_id[key] for key in sorted(by_id))


def _artifacts_compatible(*groups: tuple[ArtifactIdentity, ...]) -> bool:
    by_id: dict[str, ArtifactIdentity] = {}
    for value in (item for group in groups for item in group):
        previous = by_id.setdefault(value.artifact_id, value)
        if previous != value:
            return False
    return True


class EvidenceBoundBooleanOnlineGateEvaluator:
    "Delegate scalar gates and replay only exact offline TARGET Booleans.\n\n    Boolean replay is certificate continuity, not a pre-action remeasurement of\n    a post-action response.  The immutable raw admission receipt remains attached to\n    every live receipt.  Missing or foreign evidence is therefore represented\n    by the existing ``UNEVALUABLE`` observation shape.\n    "

    def __init__(
        self,
        implementation_binding: ImplementationBinding,
        *,
        hold_calibration_receipt: NativeHoldDecisionCellCalibrationReceipt | None = None,
        hold_selected_roster: NativeHoldCalibrationSelectedRoster | None = None,
    ) -> None:
        if implementation_binding.role is not ImplementationRole.ONLINE_GATE_EVALUATOR:
            raise ValueError("evidence-bound live gate requires ONLINE_GATE_EVALUATOR role")
        if (hold_calibration_receipt is None) != (hold_selected_roster is None):
            raise ValueError("HOLD calibration receipt and selected roster travel together")
        if (
            hold_calibration_receipt is not None
            and hold_selected_roster is not None
            and hold_calibration_receipt.selected_roster
            != ObjectIdentity.from_record(hold_selected_roster.roster_id, hold_selected_roster)
        ):
            raise ValueError("HOLD calibration receipt binds another selected roster")
        from empirical_lawhood.adapters.control.reference import AdmissionReferenceLiveGateEvaluator

        self.implementation_binding = implementation_binding
        self._scalar_delegate = AdmissionReferenceLiveGateEvaluator(implementation_binding)
        self._hold_calibration_receipt = hold_calibration_receipt
        self._hold_selected_roster = hold_selected_roster

    def _request_kind(
        self,
        *,
        study: AdmissionControllerStudy,
        decision_cell_id: str,
        admission_candidate_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        planned_coordinate_ids: tuple[str, ...],
        offline_gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...],
    ) -> str | None:
        bound = next(
            (
                value
                for value in study.implementations
                if value.role is ImplementationRole.ONLINE_GATE_EVALUATOR
            ),
            None,
        )
        if bound is None:  # pragma: no cover - programme invariant normally prevents this
            return None
        if bound != self.implementation_binding:
            return None
        cell = _candidate_cell(study, admission_candidate_cell_id)
        if cell is None or cell.planned_coordinate_ids != planned_coordinate_ids:
            return None
        try:
            fibre = study.admission.corpus.plan.action_fibre(cell.action_fibre)
        except StopIteration:
            return None
        if fibre.action_binding_id != action_binding_id:
            return None
        programme_action = next(
            (
                value
                for value in study.action_bindings
                if value.action_binding_id == action_binding_id
            ),
            None,
        )
        if programme_action is None or programme_action.action_word != fibre.action_word:
            return None
        expected = {
            value.receipt_id: value
            for value in study.admission.corpus.gate_receipts
            if value.planned_coordinate.coordinate_id in set(planned_coordinate_ids)
        }
        supplied = {value.receipt_id: value for value in offline_gate_receipts}
        if (
            len(supplied) != len(offline_gate_receipts)
            or supplied != expected
            or len(expected) != len(planned_coordinate_ids) * len(AdmissionGateKind)
        ):
            return None
        if any(
            value.planned_coordinate.action_fibre != cell.action_fibre
            or value.planned_coordinate.support_cell != cell.support_cell
            for value in offline_gate_receipts
        ):
            return None

        active = next(
            (
                value
                for value in study.synthesis.candidate_chart.candidates
                if value.candidate_id == candidate_id
            ),
            None,
        )
        if active is not None:
            if (
                active.decision_cell_id != decision_cell_id
                or active.admission_candidate_cell
                != ObjectIdentity.from_record(cell.candidate_cell_id, cell)
            ):
                return None
            return "ACTIVE"
        hold = study.measured_hold_fibre
        if hold is None or (
            hold.hold_fibre_id != candidate_id
            or hold.decision_cell_id != decision_cell_id
            or hold.admission_candidate_cell != ObjectIdentity.from_record(cell.candidate_cell_id, cell)
        ):
            return None
        return "HOLD"

    def _hold_calibration_valid(
        self,
        *,
        study: AdmissionControllerStudy,
        observation: RuntimeObservation,
        admission_candidate_cell_id: str,
        offline: AdmissionCoordinateGateReceipt,
    ) -> bool:
        hold = study.measured_hold_fibre
        receipt = self._hold_calibration_receipt
        roster = self._hold_selected_roster
        if hold is None or receipt is None or roster is None:
            return False
        if (
            receipt.disposition is not NativeHoldCalibrationDisposition.SUPPORTED
            or hold.calibration_id != receipt.receipt_id
            or hold.calibration_evidence_links != receipt.evidence_links
            or hold.decision_cell_id != receipt.hold_decision_cell_id
            or receipt.selected_sibling_hold_candidate_key_id != admission_candidate_cell_id
            or receipt.selected_roster != ObjectIdentity.from_record(roster.roster_id, roster)
        ):
            return False
        active_ids = tuple(
            sorted(
                {value.decision_cell_id for value in study.synthesis.candidate_chart.candidates}
            )
        )
        if receipt.active_decision_cell_ids != active_ids:
            return False
        observed = {value.value_id: value for value in observation.values}
        matching_slots: list[str] = []
        for selection in roster.selections:
            expected = selection.selected.preparation_values
            if all(observed.get(value.value_id) == value for value in expected):
                matching_slots.append(selection.slot_id)
        if len(matching_slots) != 1:
            return False
        slot_id = matching_slots[0]
        member_id = offline.law_evaluation_binding.denominator_member_id
        entries = tuple(
            value
            for value in receipt.occurrence_receipts
            if value.slot_id == slot_id and value.model_member_id == member_id
        )
        return bool(entries) and all(
            value.disposition is NativeHoldCalibrationDisposition.SUPPORTED for value in entries
        )

    def _boolean_binding_valid(
        self,
        *,
        request_kind: str | None,
        study: AdmissionControllerStudy,
        observation: RuntimeObservation,
        admission_candidate_cell_id: str,
        offline: AdmissionCoordinateGateReceipt,
    ) -> bool:
        if request_kind is None:
            return False
        predicate = offline.predicate
        if (
            predicate.gate_kind is not AdmissionGateKind.TARGET
            or predicate.predicate_kind is not GatePredicateKind.BOOLEAN_EQUALS
            or predicate.expected_boolean is not True
            or offline.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or offline.status is not GateStatus.PASS
            or offline.observed_boolean is None
            or offline.observed_scalar is not None
            or offline.observed_identity is not None
            or offline.evaluator != predicate.evaluator
            or not offline.input_artifacts
            or not offline.evidence_links
        ):
            return False
        calibration_artifacts = (
            self._hold_calibration_receipt.input_artifacts
            if request_kind == "HOLD" and self._hold_calibration_receipt is not None
            else ()
        )
        if not _artifacts_compatible(
            observation.input_artifacts,
            offline.input_artifacts,
            calibration_artifacts,
        ):
            return False
        return request_kind != "HOLD" or self._hold_calibration_valid(
            study=study,
            observation=observation,
            admission_candidate_cell_id=admission_candidate_cell_id,
            offline=offline,
        )

    def evaluate_admission_live_gate(
        self,
        *,
        study: AdmissionControllerStudy,
        observation: RuntimeObservation,
        decision_cell_id: str,
        admission_candidate_cell_id: str,
        candidate_id: str,
        action_binding_id: str,
        planned_coordinate_ids: tuple[str, ...],
        offline_gate_receipts: tuple[AdmissionCoordinateGateReceipt, ...],
    ) -> AdmissionLiveGateEvaluation:
        "Evaluate the exact live grid without reading post-cutoff outcomes."

        delegated = self._scalar_delegate.evaluate_admission(
            study=study,
            observation=observation,
            decision_cell_id=decision_cell_id,
            admission_candidate_cell_id=admission_candidate_cell_id,
            candidate_id=candidate_id,
            action_binding_id=action_binding_id,
            planned_coordinate_ids=planned_coordinate_ids,
            offline_gate_receipts=offline_gate_receipts,
        )
        request_kind = self._request_kind(
            study=study,
            decision_cell_id=decision_cell_id,
            admission_candidate_cell_id=admission_candidate_cell_id,
            candidate_id=candidate_id,
            action_binding_id=action_binding_id,
            planned_coordinate_ids=planned_coordinate_ids,
            offline_gate_receipts=offline_gate_receipts,
        )
        offline_by_id = {value.receipt_id: value for value in offline_gate_receipts}
        live: list[AdmissionLiveGateReceipt] = []
        for delegated_receipt in delegated.receipts:
            offline = offline_by_id[delegated_receipt.offline_gate_receipt.object_id]
            if not self._boolean_binding_valid(
                request_kind=request_kind,
                study=study,
                observation=observation,
                admission_candidate_cell_id=admission_candidate_cell_id,
                offline=offline,
            ):
                live.append(delegated_receipt)
                continue
            status, margin, reasons = derive_admission_gate_observation(
                receipt_id=delegated_receipt.receipt_id,
                predicate=delegated_receipt.predicate,
                observed_scalar=None,
                observed_boolean=offline.observed_boolean,
                observed_identity=None,
            )
            offline_link = EvidenceLink(
                link_id=f"live-offline-evidence.{candidate_id}.{offline.receipt_id}",
                relation=EvidenceRelation.DERIVED_FROM,
                source=ObjectIdentity.from_record(offline.receipt_id, offline),
                target=ObjectIdentity.from_record(
                    delegated_receipt.predicate.predicate_id,
                    delegated_receipt.predicate,
                ),
                artifact_ids=tuple(sorted(value.artifact_id for value in offline.input_artifacts)),
                world_id=study.system.world.world_id,
                information_cutoff_id=delegated_receipt.information_cutoff.cutoff_id,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                visibility_ceiling=VisibilityCeiling.most_restrictive(
                    study.visibility_ceiling,
                    *(value.visibility_ceiling for value in offline.evidence_links),
                ),
                parent_visibility_ceilings=tuple(
                    value.visibility_ceiling for value in offline.evidence_links
                ),
                reason=(
                    "The immutable raw-CONTROLLER-ADMISSION target certificate supplies this static live "
                    "Boolean operand."
                ),
            )
            artifacts = {
                value.artifact_id: value
                for value in (*delegated_receipt.input_artifacts, *offline.input_artifacts)
            }
            calibration_links: tuple[EvidenceLink, ...] = ()
            if request_kind == "HOLD" and self._hold_calibration_receipt is not None:
                calibration = self._hold_calibration_receipt
                artifacts.update(
                    {value.artifact_id: value for value in calibration.input_artifacts}
                )
                calibration_links = calibration.evidence_links
            live.append(
                replace(
                    delegated_receipt,
                    observed_boolean=offline.observed_boolean,
                    evidence_links=_merge_links(
                        delegated_receipt.evidence_links,
                        offline.evidence_links,
                        calibration_links,
                        (offline_link,),
                    ),
                    input_artifacts=tuple(artifacts[key] for key in sorted(artifacts)),
                    status=status,
                    margin=margin,
                    reason_codes=reasons,
                )
            )
        return replace(
            delegated,
            receipts=tuple(sorted(live, key=lambda value: value.receipt_id)),
        )


__all__ = [
    'BooleanOnlineGateRequestKind',
    'EvidenceBoundBooleanOnlineGateConfigTemplate',
    'EvidenceBoundBooleanOnlineGateEvaluator',
    "MAX_EVIDENCE_BOUND_BOOLEAN_ONLINE_GATE_CONFIG_TEMPLATE_BYTES",
    'decode_evidence_bound_boolean_online_gate_config_template',
]

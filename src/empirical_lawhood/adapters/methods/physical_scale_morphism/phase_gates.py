"""Typed completion audit for the physical scale morphism IP-1--IP-13 lifecycle.

This is a read-only gate projection.  Authority-record identities are evidence
that an operator can present to the real issue/execution services; this module
does not validate signatures, issue candidates, reveal outcomes, or actuate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, ClassVar

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_strings,
    require_unique_ids,
    validate_stable_id,
)

from .adjudication import PhysicalScaleMorphismAdjudicationSeal, PhysicalScaleMorphismCompletionEnvelope
from .governance import PhysicalScaleMorphismCompleteFanIn, PhysicalScaleMorphismConstructQualification, PhysicalScaleMorphismConstructTerminal, PhysicalScaleMorphismDevelopmentHandoff, PhysicalScaleMorphismExecutionSeal, PhysicalScaleMorphismIssueRecord, PhysicalScaleMorphismIssueTerminal, PhysicalScaleMorphismPredictionFreeze, PhysicalScaleMorphismSourceApparatusQualification, PhysicalScaleMorphismSourceApparatusTerminal
from .inference import PhysicalScaleMorphismPowerTerminal

if TYPE_CHECKING:
    from .provider import PhysicalScaleMorphismSyntheticReadiness


class PhysicalScaleMorphismPhase(StrEnum):
    SYNTHETIC_IMPLEMENTATION_REHEARSAL = "synthetic-implementation-rehearsal"
    INDEPENDENT_CONSTRUCT_REVIEW = "independent-construct-review"
    SOURCE_APPARATUS_QUALIFICATION = "source-apparatus-qualification"
    DEVELOPMENT_EXECUTION = "development-execution"
    PREDICTION_FREEZE = "prediction-freeze"
    COMPLETE_ROSTER_ISSUE = "complete-roster-issue"
    PHYSICAL_EXECUTION = "physical-execution"
    REVEAL_ADJUDICATION = "reveal-adjudication"
    COMPLETION_CLOSEOUT = "completion-closeout"


class PhysicalScaleMorphismPhaseDisposition(StrEnum):
    AUTHORITY_REQUIRED = "AUTHORITY_REQUIRED"
    COMPLETE = "COMPLETE"
    INPUT_REQUIRED = "INPUT_REQUIRED"
    NOT_ENTERED = "NOT_ENTERED"
    READY = "READY"
    TERMINAL_STOP = "TERMINAL_STOP"


PHYSICAL_SCALE_MORPHISM_PHASE_ROSTER = tuple(PhysicalScaleMorphismPhase)

PHYSICAL_SCALE_MORPHISM_PHASE_AUTHORITY_REQUIREMENTS: dict[PhysicalScaleMorphismPhase, tuple[str, ...]] = {
    PhysicalScaleMorphismPhase.SYNTHETIC_IMPLEMENTATION_REHEARSAL: (),
    PhysicalScaleMorphismPhase.INDEPENDENT_CONSTRUCT_REVIEW: (),
    PhysicalScaleMorphismPhase.SOURCE_APPARATUS_QUALIFICATION: (
        "authority.physical-canary",
        "authority.source-custody",
        "authority.storage-write",
    ),
    PhysicalScaleMorphismPhase.DEVELOPMENT_EXECUTION: (
        "authority.development-execution",
        "authority.storage-write",
    ),
    PhysicalScaleMorphismPhase.PREDICTION_FREEZE: (),
    PhysicalScaleMorphismPhase.COMPLETE_ROSTER_ISSUE: (
        "authority.custody",
        "authority.fabrication",
        "authority.issue",
        "authority.source",
    ),
    PhysicalScaleMorphismPhase.PHYSICAL_EXECUTION: (
        "authority.instrument-acquisition",
        "authority.physical-execution",
        "authority.storage-write",
    ),
    PhysicalScaleMorphismPhase.REVEAL_ADJUDICATION: ("authority.evaluator-reveal",),
    PhysicalScaleMorphismPhase.COMPLETION_CLOSEOUT: (),
}


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismPhaseGate(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-phase-gate'

    gate_id: str
    phase: PhysicalScaleMorphismPhase
    disposition: PhysicalScaleMorphismPhaseDisposition
    evidence_objects: tuple[ObjectIdentity, ...]
    missing_requirement_ids: tuple[str, ...]
    missing_authority_ids: tuple[str, ...]
    terminal_ids: tuple[str, ...]
    material_external_action_required: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.gate_id, field_name="gate_id")
        object_keys = tuple(
            (value.object_schema, value.object_id, value.object_fingerprint)
            for value in self.evidence_objects
        )
        if tuple(sorted(set(object_keys))) != object_keys:
            raise ValueError("phase-gate evidence objects must be sorted and unique")
        for name in ("missing_requirement_ids", "missing_authority_ids", "terminal_ids"):
            require_sorted_unique_strings(getattr(self, name), field_name=name)
        if self.disposition is PhysicalScaleMorphismPhaseDisposition.AUTHORITY_REQUIRED:
            if not self.missing_authority_ids:
                raise ValueError("authority-required gate must name missing exact authority")
        elif self.missing_authority_ids:
            raise ValueError("only an authority-required gate may retain missing authorities")
        if self.disposition is PhysicalScaleMorphismPhaseDisposition.INPUT_REQUIRED:
            if not self.missing_requirement_ids:
                raise ValueError("input-required gate must name missing independent inputs")
        elif self.missing_requirement_ids:
            raise ValueError("only an input-required gate may retain missing inputs")
        if self.disposition is PhysicalScaleMorphismPhaseDisposition.TERMINAL_STOP:
            if not self.terminal_ids:
                raise ValueError("terminal-stop gate must retain its exact terminal")
        elif self.terminal_ids:
            raise ValueError("only a terminal-stop gate may retain terminal IDs")


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismLifecycleAudit(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/physical-scale-morphism/physical-scale-morphism-lifecycle-audit'

    audit_id: str
    gates: tuple[PhysicalScaleMorphismPhaseGate, ...]
    next_phase_id: str
    evidence_ceiling: EvidenceCeiling
    outcome_access: OutcomeAccess
    visibility_ceiling: VisibilityCeiling
    physical_action_authorized: bool
    claim_promotion_allowed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.audit_id, field_name="audit_id")
        require_unique_ids(self.gates, attribute="gate_id", field_name="gates")
        if tuple(value.phase for value in self.gates) != PHYSICAL_SCALE_MORPHISM_PHASE_ROSTER:
            raise ValueError("lifecycle audit does not cover IP-1--IP-13 in order")
        expected_next = next(
            (
                value.phase.value
                for value in self.gates
                if value.disposition is not PhysicalScaleMorphismPhaseDisposition.COMPLETE
            ),
            "none",
        )
        if self.next_phase_id != expected_next:
            raise ValueError("lifecycle next phase is not gate-derived")
        if self.physical_action_authorized or self.claim_promotion_allowed:
            raise ValueError("read-only lifecycle audit grants no action or claim promotion")


def _identity(value: CanonicalRecord, object_id: str) -> ObjectIdentity:
    return ObjectIdentity.from_record(object_id, value)


def _gate(
    phase: PhysicalScaleMorphismPhase,
    disposition: PhysicalScaleMorphismPhaseDisposition,
    *,
    evidence: tuple[ObjectIdentity, ...] = (),
    missing_requirements: tuple[str, ...] = (),
    missing_authorities: tuple[str, ...] = (),
    terminals: tuple[str, ...] = (),
) -> PhysicalScaleMorphismPhaseGate:
    return PhysicalScaleMorphismPhaseGate(
        gate_id=f"gate.{phase.value}",
        phase=phase,
        disposition=disposition,
        evidence_objects=tuple(
            sorted(
                evidence,
                key=lambda value: (
                    value.object_schema,
                    value.object_id,
                    value.object_fingerprint,
                ),
            )
        ),
        missing_requirement_ids=tuple(sorted(missing_requirements)),
        missing_authority_ids=tuple(sorted(missing_authorities)),
        terminal_ids=tuple(sorted(terminals)),
        material_external_action_required=phase
        in {
            PhysicalScaleMorphismPhase.SOURCE_APPARATUS_QUALIFICATION,
            PhysicalScaleMorphismPhase.DEVELOPMENT_EXECUTION,
            PhysicalScaleMorphismPhase.COMPLETE_ROSTER_ISSUE,
            PhysicalScaleMorphismPhase.PHYSICAL_EXECUTION,
            PhysicalScaleMorphismPhase.REVEAL_ADJUDICATION,
        },
    )


def _pending_gate(
    *,
    phase: PhysicalScaleMorphismPhase,
    prior_complete: bool,
    available_authority_ids: frozenset[str],
) -> PhysicalScaleMorphismPhaseGate:
    if not prior_complete:
        return _gate(phase, PhysicalScaleMorphismPhaseDisposition.NOT_ENTERED)
    missing = tuple(
        sorted(set(PHYSICAL_SCALE_MORPHISM_PHASE_AUTHORITY_REQUIREMENTS[phase]) - available_authority_ids)
    )
    return _gate(
        phase,
        PhysicalScaleMorphismPhaseDisposition.AUTHORITY_REQUIRED if missing else PhysicalScaleMorphismPhaseDisposition.READY,
        missing_authorities=missing,
    )


def audit_physical_scale_morphism_lifecycle(
    *,
    audit_id: str,
    synthetic_readiness: PhysicalScaleMorphismSyntheticReadiness,
    construct: PhysicalScaleMorphismConstructQualification | None = None,
    source_apparatus: PhysicalScaleMorphismSourceApparatusQualification | None = None,
    development: PhysicalScaleMorphismDevelopmentHandoff | None = None,
    prediction_freeze: PhysicalScaleMorphismPredictionFreeze | None = None,
    issue: PhysicalScaleMorphismIssueRecord | None = None,
    execution: PhysicalScaleMorphismExecutionSeal | None = None,
    fan_in: PhysicalScaleMorphismCompleteFanIn | None = None,
    adjudication_seal: PhysicalScaleMorphismAdjudicationSeal | None = None,
    completion: PhysicalScaleMorphismCompletionEnvelope | None = None,
    available_authority_records: tuple[ObjectIdentity, ...] = (),
) -> PhysicalScaleMorphismLifecycleAudit:
    """Project exact supplied evidence onto every frozen lifecycle gate."""

    authority_ids = tuple(value.object_id for value in available_authority_records)
    if tuple(sorted(set(authority_ids))) != authority_ids:
        raise ValueError("available authority records must be sorted and unique by object ID")
    available = frozenset(authority_ids)
    available_authority_by_id = {value.object_id: value for value in available_authority_records}
    gates: list[PhysicalScaleMorphismPhaseGate] = []

    readiness_identity = _identity(synthetic_readiness, synthetic_readiness.readiness_id)
    synthetic_rehearsal_complete = synthetic_readiness.runtime_rehearsal_qualified
    gates.append(
        _gate(
            PhysicalScaleMorphismPhase.SYNTHETIC_IMPLEMENTATION_REHEARSAL,
            PhysicalScaleMorphismPhaseDisposition.COMPLETE
            if synthetic_rehearsal_complete
            else PhysicalScaleMorphismPhaseDisposition.TERMINAL_STOP,
            evidence=(readiness_identity,),
            terminals=() if synthetic_rehearsal_complete else ("IMPLEMENTATION_REHEARSAL_NOT_QUALIFIED",),
        )
    )

    if construct is None:
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.INDEPENDENT_CONSTRUCT_REVIEW,
                PhysicalScaleMorphismPhaseDisposition.INPUT_REQUIRED
                if synthetic_rehearsal_complete
                else PhysicalScaleMorphismPhaseDisposition.NOT_ENTERED,
                missing_requirements=(
                    "independent-construct-review",
                    "independent-native-dossier",
                    "native-to-repository-mapping",
                    "role-and-access-contamination-ledger",
                )
                if synthetic_rehearsal_complete
                else (),
            )
        )
        construct_review_complete = False
    else:
        construct_identity = _identity(construct, construct.qualification_id)
        construct_review_complete = construct.terminal is PhysicalScaleMorphismConstructTerminal.CONSTRUCT_INDEPENDENCE_QUALIFIED
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.INDEPENDENT_CONSTRUCT_REVIEW,
                PhysicalScaleMorphismPhaseDisposition.COMPLETE
                if construct_review_complete
                else PhysicalScaleMorphismPhaseDisposition.TERMINAL_STOP,
                evidence=(construct_identity,),
                terminals=() if construct_review_complete else (construct.terminal.value,),
            )
        )

    if source_apparatus is None:
        gates.append(
            _pending_gate(
                phase=PhysicalScaleMorphismPhase.SOURCE_APPARATUS_QUALIFICATION,
                prior_complete=construct_review_complete,
                available_authority_ids=available,
            )
        )
        source_apparatus_complete = False
    else:
        if construct is None:
            raise ValueError("source/apparatus qualification lacks construct parent")
        if not construct_review_complete:
            raise ValueError("source/apparatus qualification cannot follow a failed IP-6 gate")
        if source_apparatus.construct_qualification != _identity(
            construct, construct.qualification_id
        ):
            raise ValueError("source/apparatus qualification binds another construct")
        source_identity = _identity(source_apparatus, source_apparatus.qualification_id)
        source_apparatus_complete = (
            source_apparatus.terminal is PhysicalScaleMorphismSourceApparatusTerminal.SOURCE_AND_APPARATUS_QUALIFIED
        )
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.SOURCE_APPARATUS_QUALIFICATION,
                PhysicalScaleMorphismPhaseDisposition.COMPLETE
                if source_apparatus_complete
                else PhysicalScaleMorphismPhaseDisposition.TERMINAL_STOP,
                evidence=(source_identity,),
                terminals=() if source_apparatus_complete else (source_apparatus.terminal.value,),
            )
        )

    if development is None:
        gates.append(
            _pending_gate(
                phase=PhysicalScaleMorphismPhase.DEVELOPMENT_EXECUTION,
                prior_complete=source_apparatus_complete,
                available_authority_ids=available,
            )
        )
        development_complete = False
    else:
        if not source_apparatus_complete:
            raise ValueError("development handoff cannot follow a failed IP-7 gate")
        if source_apparatus is None or development.source_apparatus_qualification != _identity(
            source_apparatus, source_apparatus.qualification_id
        ):
            raise ValueError("development handoff binds another source/apparatus qualification")
        development_complete = (
            development.power_terminal is PhysicalScaleMorphismPowerTerminal.METHOD_POWER_QUALIFIED
            and development.morphism_domain_complete
        )
        development_terminals = (
            ()
            if development_complete
            else (
                "MORPHISM_DOMAIN_INCOMPLETE"
                if not development.morphism_domain_complete
                else development.power_terminal.value,
            )
        )
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.DEVELOPMENT_EXECUTION,
                PhysicalScaleMorphismPhaseDisposition.COMPLETE
                if development_complete
                else PhysicalScaleMorphismPhaseDisposition.TERMINAL_STOP,
                evidence=(_identity(development, development.handoff_id),),
                terminals=development_terminals,
            )
        )

    if prediction_freeze is None:
        gates.append(
            _pending_gate(
                phase=PhysicalScaleMorphismPhase.PREDICTION_FREEZE,
                prior_complete=development_complete,
                available_authority_ids=available,
            )
        )
        prediction_freeze_complete = False
    else:
        if not development_complete:
            raise ValueError("prediction freeze cannot follow an incomplete IP-8 gate")
        if development is None or prediction_freeze.development_handoff != _identity(
            development, development.handoff_id
        ):
            raise ValueError("prediction freeze binds another development handoff")
        if (
            development.power_qualification.selected_board_count_per_scale
            != prediction_freeze.board_count_per_scale
        ):
            raise ValueError("prediction freeze board count differs from the power handoff")
        prediction_freeze_complete = True
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.PREDICTION_FREEZE,
                PhysicalScaleMorphismPhaseDisposition.COMPLETE,
                evidence=(_identity(prediction_freeze, prediction_freeze.freeze_id),),
            )
        )

    if issue is None:
        gates.append(
            _pending_gate(
                phase=PhysicalScaleMorphismPhase.COMPLETE_ROSTER_ISSUE,
                prior_complete=prediction_freeze_complete,
                available_authority_ids=available,
            )
        )
        issue_complete = False
    else:
        if not prediction_freeze_complete:
            raise ValueError("issue record cannot follow an incomplete IP-9 gate")
        if prediction_freeze is None or issue.prediction_freeze != _identity(
            prediction_freeze, prediction_freeze.freeze_id
        ):
            raise ValueError("issue record binds another prediction freeze")
        if issue.issued_candidate != prediction_freeze.candidate:
            raise ValueError("issue record binds another frozen evaluation candidate")
        if set(issue.qualified_board_ids) != set(prediction_freeze.evaluation_board_ids):
            raise ValueError("issue board roster differs from the prediction freeze")
        if issue.expected_board_count != len(prediction_freeze.evaluation_board_ids):
            raise ValueError("issue board count differs from the prediction freeze")
        issue_complete = issue.terminal is PhysicalScaleMorphismIssueTerminal.ISSUED_COMPLETE_ROSTER
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.COMPLETE_ROSTER_ISSUE,
                PhysicalScaleMorphismPhaseDisposition.COMPLETE
                if issue_complete
                else PhysicalScaleMorphismPhaseDisposition.TERMINAL_STOP,
                evidence=(_identity(issue, issue.issue_id),),
                terminals=() if issue_complete else (issue.terminal.value,),
            )
        )

    if execution is None:
        gates.append(
            _pending_gate(
                phase=PhysicalScaleMorphismPhase.PHYSICAL_EXECUTION,
                prior_complete=issue_complete,
                available_authority_ids=available,
            )
        )
        execution_complete = False
    else:
        if not issue_complete:
            raise ValueError("execution seal cannot follow an incomplete IP-10 gate")
        if issue is None or execution.issue_record != _identity(issue, issue.issue_id):
            raise ValueError("execution seal binds another issue record")
        if issue is None or execution.run != issue.issued_run:
            raise ValueError("execution seal binds another issued run plan")
        if prediction_freeze is None or execution.prediction_freeze != _identity(
            prediction_freeze, prediction_freeze.freeze_id
        ):
            raise ValueError("execution seal binds another prediction freeze")
        execution_complete = execution.complete
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.PHYSICAL_EXECUTION,
                PhysicalScaleMorphismPhaseDisposition.COMPLETE
                if execution_complete
                else PhysicalScaleMorphismPhaseDisposition.TERMINAL_STOP,
                evidence=(_identity(execution, execution.seal_id),),
                terminals=() if execution_complete else ("INCOMPLETE_EXECUTION_ROSTER",),
            )
        )

    if fan_in is not None:
        if execution is None or fan_in.execution_seal != _identity(execution, execution.seal_id):
            raise ValueError("fan-in record binds another execution seal")
        if prediction_freeze is None or fan_in.prediction_freeze != _identity(
            prediction_freeze, prediction_freeze.freeze_id
        ):
            raise ValueError("fan-in record binds another prediction freeze")
        if execution is None:
            raise ValueError("fan-in record lacks its execution seal")
        if fan_in.expected_unit_ids != execution.expected_unit_ids:
            raise ValueError("fan-in expected roster differs from execution")
        if fan_in.received_unit_ids != execution.sealed_bundle_unit_ids:
            raise ValueError("fan-in received roster differs from sealed execution bundles")
        if fan_in.terminal_unit_ids != execution.typed_terminal_unit_ids:
            raise ValueError("fan-in terminal roster differs from execution")
        if not set(execution.publication_receipt_ids).issubset(fan_in.publication_receipt_ids):
            raise ValueError("fan-in omits execution publication receipts")
        expected_sealed_artifacts = {
            *execution.physical_bundle_artifact_ids,
            *execution.numerical_artifact_ids,
        }
        if set(fan_in.sealed_payload_artifact_ids) != expected_sealed_artifacts:
            raise ValueError("fan-in payload roster differs from sealed execution artifacts")
    if fan_in is None:
        gates.append(
            _pending_gate(
                phase=PhysicalScaleMorphismPhase.REVEAL_ADJUDICATION,
                prior_complete=execution_complete,
                available_authority_ids=available,
            )
        )
        adjudication_complete = False
    elif not fan_in.complete:
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.REVEAL_ADJUDICATION,
                PhysicalScaleMorphismPhaseDisposition.TERMINAL_STOP,
                evidence=(_identity(fan_in, fan_in.fan_in_id),),
                terminals=("INCOMPLETE_OR_UNEXPECTED_FAN_IN",),
            )
        )
        adjudication_complete = False
    elif adjudication_seal is None:
        missing = tuple(
            sorted(set(PHYSICAL_SCALE_MORPHISM_PHASE_AUTHORITY_REQUIREMENTS[PhysicalScaleMorphismPhase.REVEAL_ADJUDICATION]) - available)
        )
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.REVEAL_ADJUDICATION,
                PhysicalScaleMorphismPhaseDisposition.AUTHORITY_REQUIRED
                if missing
                else PhysicalScaleMorphismPhaseDisposition.READY,
                evidence=(_identity(fan_in, fan_in.fan_in_id),),
                missing_authorities=missing,
            )
        )
        adjudication_complete = False
    else:
        if prediction_freeze is None:
            raise ValueError("adjudication seal lacks its prediction freeze")
        if adjudication_seal.prediction_freeze != _identity(
            prediction_freeze, prediction_freeze.freeze_id
        ):
            raise ValueError("adjudication seal binds another prediction freeze")
        if adjudication_seal.complete_fan_in != _identity(fan_in, fan_in.fan_in_id):
            raise ValueError("adjudication seal binds another fan-in")
        reveal_authority_identity = _identity(
            adjudication_seal.reveal_authority,
            adjudication_seal.reveal_authority.authority_id,
        )
        if (
            available_authority_by_id.get(reveal_authority_identity.object_id)
            != reveal_authority_identity
        ):
            raise ValueError(
                "adjudication seal reveal authority is absent or fingerprint-different"
            )
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.REVEAL_ADJUDICATION,
                PhysicalScaleMorphismPhaseDisposition.COMPLETE,
                evidence=(
                    _identity(fan_in, fan_in.fan_in_id),
                    _identity(adjudication_seal, adjudication_seal.seal_id),
                    reveal_authority_identity,
                ),
            )
        )
        adjudication_complete = True

    if completion is None:
        gates.append(
            _pending_gate(
                phase=PhysicalScaleMorphismPhase.COMPLETION_CLOSEOUT,
                prior_complete=adjudication_complete,
                available_authority_ids=available,
            )
        )
    else:
        if not adjudication_complete:
            raise ValueError("completion envelope cannot follow an incomplete IP-12 gate")
        if adjudication_seal is None or completion.adjudication_seal != adjudication_seal:
            raise ValueError("completion envelope binds another adjudication seal")
        if prediction_freeze is None or completion.prediction_freeze != _identity(
            prediction_freeze, prediction_freeze.freeze_id
        ):
            raise ValueError("completion envelope binds another prediction freeze")
        if fan_in is None or completion.complete_fan_in != _identity(fan_in, fan_in.fan_in_id):
            raise ValueError("completion envelope binds another fan-in")
        gates.append(
            _gate(
                PhysicalScaleMorphismPhase.COMPLETION_CLOSEOUT,
                PhysicalScaleMorphismPhaseDisposition.COMPLETE,
                evidence=(_identity(completion, completion.envelope_id),),
            )
        )

    if completion is not None:
        outcome_access = OutcomeAccess.EVALUATION_REVEALED
        visibility = VisibilityCeiling.OUTCOME_VISIBLE
        evidence_ceiling = completion.adjudication.evidence_ceiling
    elif execution is not None or fan_in is not None:
        outcome_access = OutcomeAccess.EVALUATION_SEALED
        visibility = VisibilityCeiling.PROSPECTIVE
        evidence_ceiling = EvidenceCeiling.ADMISSION
    elif development is not None:
        outcome_access = OutcomeAccess.DEVELOPMENT_VISIBLE
        visibility = VisibilityCeiling.DEVELOPMENT_ONLY
        evidence_ceiling = EvidenceCeiling.NON_PROMOTABLE
    else:
        outcome_access = OutcomeAccess.OUTCOME_BLIND
        visibility = VisibilityCeiling.PROSPECTIVE
        evidence_ceiling = EvidenceCeiling.NON_PROMOTABLE
    gate_tuple = tuple(gates)
    next_phase_id = next(
        (
            value.phase.value
            for value in gate_tuple
            if value.disposition is not PhysicalScaleMorphismPhaseDisposition.COMPLETE
        ),
        "none",
    )
    return PhysicalScaleMorphismLifecycleAudit(
        audit_id=audit_id,
        gates=gate_tuple,
        next_phase_id=next_phase_id,
        evidence_ceiling=evidence_ceiling,
        outcome_access=outcome_access,
        visibility_ceiling=visibility,
        physical_action_authorized=False,
        claim_promotion_allowed=False,
    )


__all__ = [
    "PHYSICAL_SCALE_MORPHISM_PHASE_AUTHORITY_REQUIREMENTS",
    "PHYSICAL_SCALE_MORPHISM_PHASE_ROSTER",
    'PhysicalScaleMorphismLifecycleAudit',
    'PhysicalScaleMorphismPhase',
    'PhysicalScaleMorphismPhaseDisposition',
    'PhysicalScaleMorphismPhaseGate',
    'audit_physical_scale_morphism_lifecycle',
]

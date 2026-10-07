"Parameterised, member-local action-versus-qualified-HOLD controller use evaluation.\n\nThis module is deliberately additive.  It binds a separately issued matched\nprecommitment to a compiled controller capped at admission, retains one controller tick\nfor each committed-controller episode, and represents the independently\nqualified HOLD counterfactual without inventing a controller decision.\n"

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from typing import ClassVar

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.admission import GateStatus
from empirical_lawhood.kernel.control import (
    OperationalDeliveryState,
    ScientificCommitmentKind,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, NamedDecimal
from empirical_lawhood.kernel.serialization import (
    CanonicalRecord,
    require_sorted_unique_ids,
    require_sorted_unique_strings,
    validate_decimal,
    validate_sha256,
    validate_stable_id,
)
from empirical_lawhood.kernel.time import ClockCoordinate, parse_utc_timestamp
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy, ImplementationBinding, ImplementationRole, StageAwareDeliveryEquivalenceSpec
from empirical_lawhood.planning.matched_action_hold_controller_evaluation import MATCHED_ACTION_HOLD_OUTCOME_PROJECTION_RECEIPT_SCHEMA, MatchedActionHoldControllerEvaluationPlan, MatchedActionHoldAcquisitionPlanSeal, MatchedActionHoldEvaluationPrecommitment, MatchedActionHoldExecutionBranch, MatchedActionHoldExecutionCellSpec, MatchedActionHoldOutcomeProjectionEvidenceContract
from empirical_lawhood.runtime.controller_compiler import CompiledDisposition, CompiledAdmissionControllerStudy, ProspectiveBindingDisposition
from empirical_lawhood.runtime.controller_evaluation import OutcomePredicateEvaluation, controller_prospective_student_summary, evaluate_controller_outcome_predicate
from empirical_lawhood.runtime.controller_evaluation_nested import DurableProspectiveExecutionEventStore, ProspectiveBindingConflict, ProspectiveExecutionEventKind, ProspectiveExecutionEventPrefix, ProspectiveExecutionEvent
from empirical_lawhood.runtime.controller_runtime import CommitmentDisposition, AdmissionControllerTickReceipt, StageAwareAdmissionControllerTickReceipt, ExactActionDeliveryTrace, StageAwareActionDeliveryTrace
from empirical_lawhood.runtime.finite_chart_reference import ReferenceClassMembershipReceipt, ReferenceDispositionClass, ReferenceMembershipStatus, evaluate_reference_class_membership


MAX_MATCHED_ACTION_HOLD_BUNDLE_BYTES = 64 * 1024 * 1024


class MatchedActionHoldRevealIntegrity(StrEnum):
    VALID = "VALID"
    RECEIPT_CUSTODY_OR_REVEAL_INVALID = "RECEIPT_CUSTODY_OR_REVEAL_INVALID"
    AUTHORITY_OR_RESOURCE_STOP = "AUTHORITY_OR_RESOURCE_STOP"


class MatchedActionHoldOutcomeProjectionDisposition(StrEnum):
    COMPLETE = "COMPLETE"
    UNEVALUABLE = "UNEVALUABLE"


class MatchedActionHoldUnitDisposition(StrEnum):
    EFFICACY_EVALUABLE = "EFFICACY_EVALUABLE"
    HOLD_CONTROL_CORRECT = "HOLD_CONTROL_CORRECT"
    NONATTEMPT = "NONATTEMPT"
    REFERENCE_INCORRECT = "REFERENCE_INCORRECT"
    UNSAFE = "UNSAFE"
    DELIVERY_INVALID = "DELIVERY_INVALID"
    TECHNICAL_PARTIAL = "TECHNICAL_PARTIAL"
    UNEVALUABLE = "UNEVALUABLE"


class MatchedActionHoldProspectiveResult(StrEnum):
    AUTHORITY_OR_RESOURCE_STOP = "AUTHORITY_OR_RESOURCE_STOP"
    RECEIPT_CUSTODY_OR_REVEAL_INVALID = "RECEIPT_CUSTODY_OR_REVEAL_INVALID"
    UNSAFE_OR_FALSE_SAFE_DISPOSITION = "UNSAFE_OR_FALSE_SAFE_DISPOSITION"
    DELIVERY_INVALID = "DELIVERY_INVALID"
    TECHNICAL_PARTIAL = "TECHNICAL_PARTIAL"
    EFFICACY_UNEVALUABLE = "EFFICACY_UNEVALUABLE"
    MIXED_MEMBER_OR_STRATUM_EFFICACY = "MIXED_MEMBER_OR_STRATUM_EFFICACY"
    VALID_NEGATIVE_OR_SUBMATERIAL_EFFICACY = "VALID_NEGATIVE_OR_SUBMATERIAL_EFFICACY"
    CONTROLLER_USE_HOLD_CONTROL_FAILED = "CONTROLLER_USE_HOLD_CONTROL_FAILED"
    TERMINAL_PRIMARY_CONTROLLER_USE_REPLICATION_POSITIVE = "TERMINAL_PRIMARY_CONTROLLER_USE_REPLICATION_POSITIVE"


@dataclass(frozen=True, slots=True)
class MatchedActionHoldEvaluationBindingReceipt(CanonicalRecord):
    """Atomic post-compile binding of the separately issued matched plan."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-evaluation-binding-receipt'

    binding_id: str
    coordinate_id: str
    issued_manifest: ObjectIdentity
    extension_set: ObjectIdentity
    acquisition_plan_seal: ObjectIdentity
    precommitment: MatchedActionHoldEvaluationPrecommitment
    study: ObjectIdentity
    compiled_study: ObjectIdentity
    reference_design: ObjectIdentity
    matched_plan: ObjectIdentity
    evaluator_binding: ImplementationBinding
    causal_cutoff_id: str
    acquisition_domain_id: str
    reference_domain_id: str
    evaluation_domain_id: str
    prefix_before: ProspectiveExecutionEventPrefix
    accepted_event: ProspectiveExecutionEvent
    accepted_prefix_root_sha256: str
    bound_at_utc: str
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("binding_id", self.binding_id),
            ("coordinate_id", self.coordinate_id),
            ("causal_cutoff_id", self.causal_cutoff_id),
            ("acquisition_domain_id", self.acquisition_domain_id),
            ("reference_domain_id", self.reference_domain_id),
            ("evaluation_domain_id", self.evaluation_domain_id),
        ):
            validate_stable_id(value, field_name=name)
        parse_utc_timestamp(self.bound_at_utc, field_name="bound_at_utc")
        if self.study.object_schema != AdmissionControllerStudy.SCHEMA:
            raise ValueError("matched binding requires an admission controller study")
        if self.compiled_study.object_schema != CompiledAdmissionControllerStudy.SCHEMA:
            raise ValueError("matched binding requires a compiled admission controller study")
        if self.matched_plan.object_schema != (MatchedActionHoldControllerEvaluationPlan.SCHEMA):
            raise ValueError("matched binding requires the matched plan schema")
        if (
            self.acquisition_plan_seal.object_schema
            != MatchedActionHoldAcquisitionPlanSeal.SCHEMA
        ):
            raise ValueError("matched binding requires the acquisition-plan seal schema")
        if self.accepted_event.kind is not ProspectiveExecutionEventKind.EVALUATOR_BOUND:
            raise ValueError("matched binding receipt has another event kind")
        precommitment_identity = ObjectIdentity.from_record(
            self.precommitment.precommitment_id,
            self.precommitment,
        )
        if (
            self.accepted_event.coordinate_id != self.coordinate_id
            or self.accepted_event.subject != self.compiled_study
            or self.accepted_event.parent != precommitment_identity
        ):
            raise ValueError("matched binding event changes its frozen identities")
        plan = self.precommitment.matched_plan
        if (
            self.issued_manifest != self.precommitment.issued_manifest
            or self.extension_set != self.precommitment.extension_set
            or self.acquisition_plan_seal != self.precommitment.acquisition_plan_seal
            or self.reference_design != self.precommitment.reference_design
            or self.matched_plan != ObjectIdentity.from_record(plan.evaluation_plan_id, plan)
            or self.evaluator_binding != self.precommitment.evaluator_binding
            or self.causal_cutoff_id != self.precommitment.causal_cutoff_id
            or self.acquisition_domain_id != self.precommitment.acquisition_domain_id
            or self.reference_domain_id != self.precommitment.reference_domain_id
            or self.evaluation_domain_id != self.precommitment.evaluation_domain_id
            or self.study != self.precommitment.eligible_admission_study
            or self.bound_at_utc != self.accepted_event.occurred_at_utc
        ):
            raise ValueError("matched binding receipt changes its precommitment")
        accepted = self.prefix_before.append(self.accepted_event)
        if (
            self.prefix_before.coordinate_id != self.coordinate_id
            or any(value.kind.protected for value in self.prefix_before.events)
            or self.accepted_event.sequence_number != len(self.prefix_before.events) + 1
            or accepted.event_root_sha256 != self.accepted_prefix_root_sha256
        ):
            raise ValueError("matched binding is not derived from a clean event prefix")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("matched binding cannot reveal outcomes")


class MatchedActionHoldEvaluationBindingCoordinator:
    "Reuse the durable compare-and-append event algorithm for matched controller use."

    def __init__(self, store: DurableProspectiveExecutionEventStore) -> None:
        self._store = store

    def initialize(
        self,
        *,
        prefix_id: str,
        coordinate_id: str,
        study: AdmissionControllerStudy,
        compiled: CompiledAdmissionControllerStudy,
        issued_at_utc: str,
        compiled_at_utc: str,
    ) -> ProspectiveExecutionEventPrefix:
        if compiled.study != study:
            raise ValueError("compiled programme substitutes another programme")
        programme_identity = ObjectIdentity.from_record(study.study_id, study)
        compiled_identity = ObjectIdentity.from_record(
            compiled.compiled_study_id,
            compiled,
        )
        prefix = ProspectiveExecutionEventPrefix.from_events(
            prefix_id=prefix_id,
            coordinate_id=coordinate_id,
            events=(
                ProspectiveExecutionEvent(
                    event_id=f"event.{coordinate_id}.programme-issued",
                    sequence_number=1,
                    kind=ProspectiveExecutionEventKind.PROGRAMME_ISSUED,
                    coordinate_id=coordinate_id,
                    subject=programme_identity,
                    parent=None,
                    occurred_at_utc=issued_at_utc,
                ),
                ProspectiveExecutionEvent(
                    event_id=f"event.{coordinate_id}.compiled",
                    sequence_number=2,
                    kind=ProspectiveExecutionEventKind.COMPILED,
                    coordinate_id=coordinate_id,
                    subject=compiled_identity,
                    parent=programme_identity,
                    occurred_at_utc=compiled_at_utc,
                ),
            ),
        )
        self._store.create(prefix)
        return prefix

    @staticmethod
    def _validate_admission_only(
        study: AdmissionControllerStudy,
        compiled: CompiledAdmissionControllerStudy,
    ) -> None:
        if (
            study.prospective_evaluation is not None
            or compiled.disposition is not CompiledDisposition.CONTROLLER
            or compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.NOT_DECLARED
            or compiled.prospective_evaluation_binding.evaluation_plan is not None
            or compiled.prospective_evaluation_binding.evaluator is not None
        ):
            raise ValueError("matched binding requires compiled controller use NOT_DECLARED")
        if compiled.study != study:
            raise ValueError("matched binding substitutes another programme")
        if any(
            value.role is ImplementationRole.OUTCOME_EVALUATOR
            for value in study.implementations
        ):
            raise ValueError("programme capped at admission cannot embed the matched evaluator")

    @staticmethod
    def _receipt(
        *,
        precommitment: MatchedActionHoldEvaluationPrecommitment,
        study: AdmissionControllerStudy,
        compiled: CompiledAdmissionControllerStudy,
        prefix_before: ProspectiveExecutionEventPrefix,
        accepted_event: ProspectiveExecutionEvent,
        accepted_prefix: ProspectiveExecutionEventPrefix,
    ) -> MatchedActionHoldEvaluationBindingReceipt:
        plan = precommitment.matched_plan
        return MatchedActionHoldEvaluationBindingReceipt(
            binding_id=f"matched-binding.{prefix_before.coordinate_id}",
            coordinate_id=prefix_before.coordinate_id,
            issued_manifest=precommitment.issued_manifest,
            extension_set=precommitment.extension_set,
            acquisition_plan_seal=precommitment.acquisition_plan_seal,
            precommitment=precommitment,
            study=ObjectIdentity.from_record(study.study_id, study),
            compiled_study=ObjectIdentity.from_record(
                compiled.compiled_study_id,
                compiled,
            ),
            reference_design=precommitment.reference_design,
            matched_plan=ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
            evaluator_binding=precommitment.evaluator_binding,
            causal_cutoff_id=precommitment.causal_cutoff_id,
            acquisition_domain_id=precommitment.acquisition_domain_id,
            reference_domain_id=precommitment.reference_domain_id,
            evaluation_domain_id=precommitment.evaluation_domain_id,
            prefix_before=prefix_before,
            accepted_event=accepted_event,
            accepted_prefix_root_sha256=accepted_prefix.event_root_sha256,
            bound_at_utc=accepted_event.occurred_at_utc,
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
        )

    def bind(
        self,
        *,
        prefix_id: str,
        precommitment: MatchedActionHoldEvaluationPrecommitment,
        study: AdmissionControllerStudy,
        compiled: CompiledAdmissionControllerStudy,
        bound_at_utc: str,
    ) -> MatchedActionHoldEvaluationBindingReceipt:
        self._validate_admission_only(study, compiled)
        if ObjectIdentity.from_record(study.study_id, study) != (
            precommitment.eligible_admission_study
        ):
            raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
        current = self._store.load(prefix_id)
        programme_identity = ObjectIdentity.from_record(study.study_id, study)
        compiled_identity = ObjectIdentity.from_record(
            compiled.compiled_study_id,
            compiled,
        )
        expected_initial = (
            (ProspectiveExecutionEventKind.PROGRAMME_ISSUED, programme_identity, None),
            (ProspectiveExecutionEventKind.COMPILED, compiled_identity, programme_identity),
        )
        observed_initial = tuple(
            (value.kind, value.subject, value.parent) for value in current.events[:2]
        )
        if observed_initial != expected_initial:
            raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
        precommitment_identity = ObjectIdentity.from_record(
            precommitment.precommitment_id,
            precommitment,
        )
        existing = next(
            (
                value
                for value in current.events
                if value.kind is ProspectiveExecutionEventKind.EVALUATOR_BOUND
            ),
            None,
        )
        if existing is not None:
            if (
                existing.subject != compiled_identity
                or existing.parent != precommitment_identity
                or any(
                    value.kind.protected for value in current.events[: existing.sequence_number - 1]
                )
            ):
                raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
            before = ProspectiveExecutionEventPrefix.from_events(
                prefix_id=current.prefix_id,
                coordinate_id=current.coordinate_id,
                events=current.events[: existing.sequence_number - 1],
            )
            accepted = ProspectiveExecutionEventPrefix.from_events(
                prefix_id=current.prefix_id,
                coordinate_id=current.coordinate_id,
                events=current.events[: existing.sequence_number],
            )
            return self._receipt(
                precommitment=precommitment,
                study=study,
                compiled=compiled,
                prefix_before=before,
                accepted_event=existing,
                accepted_prefix=accepted,
            )
        if len(current.events) != 2 or any(value.kind.protected for value in current.events):
            raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
        event = ProspectiveExecutionEvent(
            event_id=f"event.{current.coordinate_id}.matched-evaluator-bound",
            sequence_number=3,
            kind=ProspectiveExecutionEventKind.EVALUATOR_BOUND,
            coordinate_id=current.coordinate_id,
            subject=compiled_identity,
            parent=precommitment_identity,
            occurred_at_utc=bound_at_utc,
        )
        updated = current.append(event)
        self._store.compare_and_append(
            expected_prefix_sha256=current.fingerprint(),
            updated=updated,
        )
        return self._receipt(
            precommitment=precommitment,
            study=study,
            compiled=compiled,
            prefix_before=current,
            accepted_event=event,
            accepted_prefix=updated,
        )

    def reserve_protected_event(
        self,
        *,
        prefix_id: str,
        receipt: MatchedActionHoldEvaluationBindingReceipt,
        event_id: str,
        kind: ProspectiveExecutionEventKind,
        subject: ObjectIdentity,
        occurred_at_utc: str,
    ) -> ProspectiveExecutionEventPrefix:
        if not kind.protected:
            raise ValueError("requested event is not protected")
        current = self._store.load(prefix_id)
        bound = next(
            (
                value
                for value in current.events
                if value.kind is ProspectiveExecutionEventKind.EVALUATOR_BOUND
            ),
            None,
        )
        if bound is None or bound != receipt.accepted_event:
            raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
        accepted = ProspectiveExecutionEventPrefix.from_events(
            prefix_id=current.prefix_id,
            coordinate_id=current.coordinate_id,
            events=current.events[: bound.sequence_number],
        )
        if accepted.event_root_sha256 != receipt.accepted_prefix_root_sha256:
            raise ProspectiveBindingConflict("PROSPECTIVE_BINDING_CONFLICT")
        event = ProspectiveExecutionEvent(
            event_id=event_id,
            sequence_number=len(current.events) + 1,
            kind=kind,
            coordinate_id=current.coordinate_id,
            subject=subject,
            parent=ObjectIdentity.from_record(receipt.binding_id, receipt),
            occurred_at_utc=occurred_at_utc,
        )
        updated = current.append(event)
        self._store.compare_and_append(
            expected_prefix_sha256=current.fingerprint(),
            updated=updated,
        )
        return updated


@dataclass(frozen=True, slots=True)
class MatchedActionHoldQualifiedHoldCommitment(CanonicalRecord):
    """Outcome-blind qualified-HOLD commitment, explicitly not a controller tick."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-qualified-hold-commitment'

    commitment_id: str
    compiled_study: ObjectIdentity
    execution_cell: ObjectIdentity
    measured_hold_fibre: ObjectIdentity
    native_hold_calibration: ObjectIdentity
    commitment_kind: ScientificCommitmentKind
    action_word: OccurrenceActionWord | None
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.commitment_id, field_name="commitment_id")
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.compiled_study.object_schema != CompiledAdmissionControllerStudy.SCHEMA:
            raise ValueError("qualified HOLD requires a compiled admission controller study")
        if self.execution_cell.object_schema != MatchedActionHoldExecutionCellSpec.SCHEMA:
            raise ValueError("qualified HOLD binds another execution-cell schema")
        if self.commitment_kind is ScientificCommitmentKind.MEASURED_HOLD:
            if self.action_word is None or self.reason_codes:
                raise ValueError("qualified HOLD commitment lacks its exact action word")
        elif self.commitment_kind is ScientificCommitmentKind.NONATTEMPT:
            if self.action_word is not None or not self.reason_codes:
                raise ValueError("qualified HOLD NONATTEMPT fabricates an action")
        else:
            raise ValueError("qualified HOLD counterfactual cannot commit ACTION")
        if self.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("qualified HOLD commitment must remain outcome-blind")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldSealedEpisode(CanonicalRecord):
    """One member-local execution with fresh delivery and sealed outcome bytes."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-sealed-episode'

    episode_id: str
    execution_cell: MatchedActionHoldExecutionCellSpec
    controller_tick: AdmissionControllerTickReceipt | StageAwareAdmissionControllerTickReceipt | None
    qualified_hold_commitment: MatchedActionHoldQualifiedHoldCommitment | None
    delivery_trace: ExactActionDeliveryTrace | StageAwareActionDeliveryTrace
    preparation_artifact: ArtifactIdentity
    config_artifact: ArtifactIdentity
    preaction_prefix_artifact: ArtifactIdentity
    preaction_prefix_cutoff: ClockCoordinate
    prefix_complete: bool
    sealed_outcome_artifact: ArtifactIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.episode_id, field_name="episode_id")
        if not isinstance(
            self.delivery_trace,
            (ExactActionDeliveryTrace, StageAwareActionDeliveryTrace),
        ):
            raise ValueError("matched episode uses an unknown delivery-trace schema")
        if self.controller_tick is not None and not isinstance(
            self.controller_tick,
            (AdmissionControllerTickReceipt, StageAwareAdmissionControllerTickReceipt),
        ):
            raise ValueError("matched episode uses an unknown controller-tick schema")
        if self.episode_id != self.execution_cell.episode_id:
            raise ValueError("sealed episode changes its planned identity")
        branch = self.execution_cell.branch
        if branch is MatchedActionHoldExecutionBranch.QUALIFIED_HOLD_COUNTERFACTUAL:
            if self.controller_tick is not None or self.qualified_hold_commitment is None:
                raise ValueError("qualified-HOLD counterfactual cannot fabricate a tick")
            commitment = self.qualified_hold_commitment
            if (
                commitment.execution_cell
                != ObjectIdentity.from_record(self.execution_cell.episode_id, self.execution_cell)
                or self.delivery_trace.commitment
                != ObjectIdentity.from_record(commitment.commitment_id, commitment)
                or self.delivery_trace.commitment_kind is not commitment.commitment_kind
                or self.delivery_trace.expected_action_word != commitment.action_word
            ):
                raise ValueError("qualified-HOLD delivery changes its commitment")
        else:
            if self.controller_tick is None or self.qualified_hold_commitment is not None:
                raise ValueError("controller episode requires exactly one closed-version tick")
            if (
                self.delivery_trace != self.controller_tick.delivery_trace
                or self.controller_tick.commitment.observation.independent_unit_id
                != self.execution_cell.execution_id
            ):
                raise ValueError("controller episode changes its member-local tick trace")
        if self.delivery_trace.commitment_kind is ScientificCommitmentKind.NONATTEMPT:
            if isinstance(self.delivery_trace, StageAwareActionDeliveryTrace) or isinstance(
                self.controller_tick,
                StageAwareAdmissionControllerTickReceipt,
            ):
                raise ValueError("matched NONATTEMPT cannot claim stage-aware delivery")
        elif self.controller_tick is not None and (
            isinstance(self.delivery_trace, StageAwareActionDeliveryTrace)
            != isinstance(self.controller_tick, StageAwareAdmissionControllerTickReceipt)
        ):
            raise ValueError("matched tick/trace delivery versions differ")
        if not self.prefix_complete and (
            self.delivery_trace.commitment_kind is not ScientificCommitmentKind.NONATTEMPT
        ):
            raise ValueError("incomplete pre-action prefix must be NONATTEMPT")
        if self.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError("sealed matched episode must remain sealed")


def _artifact_byte_identity(artifact: ArtifactIdentity) -> tuple[str, str, str, int]:
    return (
        artifact.sha256,
        artifact.payload_schema,
        artifact.media_type,
        artifact.size_bytes,
    )


def _episode_is_nonattempt(episode: MatchedActionHoldSealedEpisode) -> bool:
    return episode.delivery_trace.commitment_kind is ScientificCommitmentKind.NONATTEMPT


@dataclass(frozen=True, slots=True)
class MatchedActionHoldProspectiveBundle(CanonicalRecord):
    """Exact parameterised sealed product with genuine controller ticks."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-prospective-bundle'

    bundle_id: str
    evaluation_plan: MatchedActionHoldControllerEvaluationPlan
    binding_receipt: MatchedActionHoldEvaluationBindingReceipt
    compiled_study: ObjectIdentity
    protected_event_prefix: ProspectiveExecutionEventPrefix
    episodes: tuple[MatchedActionHoldSealedEpisode, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.bundle_id, field_name="bundle_id")
        require_sorted_unique_ids(self.episodes, attribute="episode_id", field_name="episodes")
        plan_identity = ObjectIdentity.from_record(
            self.evaluation_plan.evaluation_plan_id,
            self.evaluation_plan,
        )
        if (
            self.compiled_study != self.binding_receipt.compiled_study
            or self.binding_receipt.matched_plan != plan_identity
        ):
            raise ValueError("matched bundle substitutes its binding or plan")
        if self.protected_event_prefix.coordinate_id != self.binding_receipt.coordinate_id:
            raise ValueError("matched bundle uses another protected event coordinate")
        bound = next(
            (
                value
                for value in self.protected_event_prefix.events
                if value.kind is ProspectiveExecutionEventKind.EVALUATOR_BOUND
            ),
            None,
        )
        protected = tuple(
            value for value in self.protected_event_prefix.events if value.kind.protected
        )
        expected_subjects = {
            ProspectiveExecutionEventKind.TICK_RESERVED: self.compiled_study,
            ProspectiveExecutionEventKind.REFERENCE_RESERVED: (
                self.evaluation_plan.finite_chart_reference_design
            ),
            ProspectiveExecutionEventKind.CONFIRMATORY_RESERVED: (
                self.binding_receipt.acquisition_plan_seal
            ),
        }
        binding_identity = ObjectIdentity.from_record(
            self.binding_receipt.binding_id,
            self.binding_receipt,
        )
        if (
            bound != self.binding_receipt.accepted_event
            or any(value.sequence_number <= bound.sequence_number for value in protected)
            or {value.kind for value in protected}
            != {
                ProspectiveExecutionEventKind.TICK_RESERVED,
                ProspectiveExecutionEventKind.REFERENCE_RESERVED,
                ProspectiveExecutionEventKind.CONFIRMATORY_RESERVED,
            }
            or any(
                value.subject != expected_subjects[value.kind] or value.parent != binding_identity
                for value in protected
            )
        ):
            raise ValueError("matched controller use protected work was not reserved after binding")
        expected_episode_count = len(self.evaluation_plan.execution_cells)
        if len(self.episodes) != expected_episode_count or tuple(
            value.episode_id for value in self.episodes
        ) != tuple(value.episode_id for value in self.evaluation_plan.execution_cells):
            raise ValueError("matched sealed bundle differs from its exact episode plan")

        cells = {value.episode_id: value for value in self.evaluation_plan.execution_cells}
        units = {value.unit_id: value for value in self.evaluation_plan.units}
        ticks = tuple(
            value.controller_tick for value in self.episodes if value.controller_tick is not None
        )
        expected_tick_count = sum(
            value.branch is not MatchedActionHoldExecutionBranch.QUALIFIED_HOLD_COUNTERFACTUAL
            for value in self.evaluation_plan.execution_cells
        )
        if (
            len(ticks) != expected_tick_count
            or len({value.tick_id for value in ticks}) != len(ticks)
            or len({value.commitment.commitment_id for value in ticks}) != len(ticks)
            or len({value.commitment.observation.observation_id for value in ticks}) != len(ticks)
        ):
            raise ValueError("matched bundle lacks its exact unique member-local ticks")
        trace_ids = tuple(value.delivery_trace.trace_id for value in self.episodes)
        if len(set(trace_ids)) != expected_episode_count:
            raise ValueError("matched scientific episodes reuse a delivery trace")
        commitment_ids = tuple(
            (
                value.controller_tick.commitment.commitment_id
                if value.controller_tick is not None
                else value.qualified_hold_commitment.commitment_id
                if value.qualified_hold_commitment is not None
                else ""
            )
            for value in self.episodes
        )
        if "" in commitment_ids or len(set(commitment_ids)) != len(self.episodes):
            raise ValueError("matched scientific episodes reuse a commitment identity")
        for attribute in (
            "preparation_artifact",
            "config_artifact",
            "preaction_prefix_artifact",
            "sealed_outcome_artifact",
        ):
            artifact_ids = tuple(getattr(value, attribute).artifact_id for value in self.episodes)
            if len(set(artifact_ids)) != expected_episode_count:
                raise ValueError("matched scientific episodes reuse an artifact identity")

        by_product = {value.execution_cell.product_key: value for value in self.episodes}
        for episode in self.episodes:
            cell = cells[episode.episode_id]
            unit = units[cell.unit_id]
            if episode.execution_cell != cell:
                raise ValueError("matched sealed episode changes its planned cell")
            if episode.preparation_artifact.sha256 != unit.preparation_sha256:
                raise ValueError("matched episode changes its issued preparation bytes")
            if episode.preaction_prefix_cutoff != self.evaluation_plan.causal_cutoff:
                raise ValueError("matched episode changes the frozen pre-action cutoff")
            if episode.delivery_trace.delivery.role is not ImplementationRole.DELIVERY:
                raise ValueError("matched episode uses another delivery role")
            branch = cell.branch
            if branch is MatchedActionHoldExecutionBranch.QUALIFIED_HOLD_COUNTERFACTUAL:
                commitment = episode.qualified_hold_commitment
                if commitment is None:  # pragma: no cover - narrowed by episode
                    raise AssertionError("qualified HOLD episode lost its commitment")
                if (
                    commitment.compiled_study != self.compiled_study
                    or commitment.measured_hold_fibre != self.evaluation_plan.measured_hold_fibre
                    or commitment.native_hold_calibration
                    != self.evaluation_plan.native_hold_calibration
                ):
                    raise ValueError("qualified HOLD changes compiled/calibration lineage")
                if not _episode_is_nonattempt(episode) and (
                    commitment.commitment_kind is not ScientificCommitmentKind.MEASURED_HOLD
                    or commitment.action_word != self.evaluation_plan.qualified_hold_word
                ):
                    raise ValueError("qualified-HOLD branch executes another action")
            else:
                tick = episode.controller_tick
                if tick is None:  # pragma: no cover - narrowed by episode
                    raise AssertionError("controller episode lost its tick")
                if tick.compiled_study != self.compiled_study:
                    raise ValueError("matched tick binds another compiled programme")
                if _episode_is_nonattempt(episode):
                    if tick.commitment.disposition is not CommitmentDisposition.NONATTEMPT:
                        raise ValueError("matched NONATTEMPT tick changes disposition")
                else:
                    action_binding = tick.commitment.action_binding
                    expected_word = (
                        self.evaluation_plan.qualified_action_word
                        if branch is MatchedActionHoldExecutionBranch.COMMITTED_CONTROLLER_ACTION
                        else self.evaluation_plan.qualified_hold_word
                    )
                    expected_disposition = (
                        CommitmentDisposition.ACTION_COMMITTED
                        if branch is MatchedActionHoldExecutionBranch.COMMITTED_CONTROLLER_ACTION
                        else CommitmentDisposition.MEASURED_HOLD_COMMITTED
                    )
                    if (
                        action_binding is None
                        or action_binding.action_word != expected_word
                        or tick.commitment.disposition is not expected_disposition
                    ):
                        raise ValueError("matched tick commits another branch action")

        delivered_traces = tuple(
            value.delivery_trace for value in self.episodes if not _episode_is_nonattempt(value)
        )
        stage_aware_traces = tuple(
            value for value in delivered_traces if isinstance(value, StageAwareActionDeliveryTrace)
        )
        if stage_aware_traces and len(stage_aware_traces) != len(delivered_traces):
            raise ValueError("matched delivery mixes exact and stage-aware routes")
        if stage_aware_traces and any(
            value.delivery_equivalence != stage_aware_traces[0].delivery_equivalence
            for value in stage_aware_traces[1:]
        ):
            raise ValueError("matched delivery changes its stage-aware specification")

        for unit in self.evaluation_plan.efficacy_units:
            for member_id in self.evaluation_plan.model_member_ids:
                action = by_product[
                    (
                        unit.unit_id,
                        member_id,
                        MatchedActionHoldExecutionBranch.COMMITTED_CONTROLLER_ACTION,
                    )
                ]
                hold = by_product[
                    (
                        unit.unit_id,
                        member_id,
                        MatchedActionHoldExecutionBranch.QUALIFIED_HOLD_COUNTERFACTUAL,
                    )
                ]
                pair_matches = (
                    action.prefix_complete
                    and hold.prefix_complete
                    and _artifact_byte_identity(action.preparation_artifact)
                    == _artifact_byte_identity(hold.preparation_artifact)
                    and _artifact_byte_identity(action.config_artifact)
                    == _artifact_byte_identity(hold.config_artifact)
                    and _artifact_byte_identity(action.preaction_prefix_artifact)
                    == _artifact_byte_identity(hold.preaction_prefix_artifact)
                )
                if not pair_matches and not (
                    _episode_is_nonattempt(action) and _episode_is_nonattempt(hold)
                ):
                    raise ValueError("unmatched pre-action pair must be paired NONATTEMPT")

        for control in self.evaluation_plan.hold_controls:
            for member_id in self.evaluation_plan.model_member_ids:
                episode = by_product[
                    (
                        control.unit_id,
                        member_id,
                        MatchedActionHoldExecutionBranch.COMMITTED_HOLD_CONTROL,
                    )
                ]
                if not episode.prefix_complete and not _episode_is_nonattempt(episode):
                    raise ValueError("incomplete HOLD control must be NONATTEMPT")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldProjectionOperandEvidence(CanonicalRecord):
    """One substrate-neutral native operand retained without adjudication."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-projection-operand-evidence'

    operand_id: str
    source_field_ids: tuple[str, ...]
    source_evidence_ids: tuple[str, ...]
    reduction_id: str
    raw_value: NamedDecimal
    threshold: NamedDecimal
    source_numeric_padding: NamedDecimal
    signed_margin: NamedDecimal
    certified_margin: NamedDecimal
    passed: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.operand_id, field_name="operand_id")
        require_sorted_unique_strings(
            self.source_field_ids,
            field_name="source_field_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(
            self.source_evidence_ids,
            field_name="source_evidence_ids",
            allow_empty=False,
        )
        validate_stable_id(self.reduction_id, field_name="reduction_id")
        values = (
            self.raw_value,
            self.threshold,
            self.source_numeric_padding,
            self.signed_margin,
            self.certified_margin,
        )
        if len({value.unit for value in values}) != 1:
            raise ValueError("matched projection operand changes native units")
        if self.source_numeric_padding.value < 0:
            raise ValueError("matched projection operand padding cannot be negative")
        if self.certified_margin.value != (
            self.signed_margin.value + self.source_numeric_padding.value
        ):
            raise ValueError("matched projection operand changes certified-margin arithmetic")
        if self.passed != (self.certified_margin.value >= 0):
            raise ValueError("matched projection operand pass differs from its margin")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldProjectionConstraintEvidence(CanonicalRecord):
    """One exact frozen predicate constraint and its retained evidence links."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-projection-constraint-evidence'

    predicate_id: str
    constraint_id: str
    status: GateStatus
    operand_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.predicate_id, field_name="predicate_id")
        validate_stable_id(self.constraint_id, field_name="constraint_id")
        require_sorted_unique_strings(self.operand_ids, field_name="operand_ids")
        require_sorted_unique_strings(
            self.evidence_ids,
            field_name="evidence_ids",
            allow_empty=False,
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.status is GateStatus.PASS:
            if self.reason_codes:
                raise ValueError("passing projection constraint carries reason codes")
        elif not self.reason_codes:
            raise ValueError("nonpassing projection constraint lacks a reason code")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldReceiverSample(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-receiver-sample'

    sample_id: str
    quantity_id: str
    coordinate: ClockCoordinate
    value: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.sample_id, field_name="sample_id")
        validate_stable_id(self.quantity_id, field_name="quantity_id")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldOutcomeProjectionEvidenceReceipt(CanonicalRecord):
    """Exact per-episode bridge from a substrate projection into G3 evidence."""

    SCHEMA: ClassVar[str] = MATCHED_ACTION_HOLD_OUTCOME_PROJECTION_RECEIPT_SCHEMA

    receipt_id: str
    projection_contract: ObjectIdentity
    projection_specification: ObjectIdentity
    projection: ObjectIdentity
    sealed_outcome_artifact: ArtifactIdentity
    episode_id: str
    projector_implementation_id: str
    projector_implementation_sha256: str
    response_samples: tuple[MatchedActionHoldReceiverSample, ...]
    raw_values: tuple[NamedDecimal, ...]
    operands: tuple[MatchedActionHoldProjectionOperandEvidence, ...]
    constraint_evidence: tuple[MatchedActionHoldProjectionConstraintEvidence, ...]
    disposition: MatchedActionHoldOutcomeProjectionDisposition
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for field_name, value in (
            ("receipt_id", self.receipt_id),
            ("episode_id", self.episode_id),
            ("projector_implementation_id", self.projector_implementation_id),
        ):
            validate_stable_id(value, field_name=field_name)
        validate_sha256(
            self.projector_implementation_sha256,
            field_name="projector_implementation_sha256",
        )
        if self.projection_contract.object_schema != (
            MatchedActionHoldOutcomeProjectionEvidenceContract.SCHEMA
        ):
            raise ValueError("matched projection receipt binds another contract schema")
        require_sorted_unique_ids(
            self.response_samples,
            attribute="sample_id",
            field_name="response_samples",
        )
        require_sorted_unique_ids(
            self.raw_values,
            attribute="value_id",
            field_name="raw_values",
        )
        require_sorted_unique_ids(
            self.operands,
            attribute="operand_id",
            field_name="operands",
        )
        require_sorted_unique_ids(
            self.constraint_evidence,
            attribute="constraint_id",
            field_name="constraint_evidence",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        operand_ids = {value.operand_id for value in self.operands}
        if any(not set(value.operand_ids) <= operand_ids for value in self.constraint_evidence):
            raise ValueError("matched projection constraint cites an absent operand")
        operand_by_id = {value.operand_id: value for value in self.operands}
        if any(
            value.status is GateStatus.PASS
            and any(not operand_by_id[operand_id].passed for operand_id in value.operand_ids)
            for value in self.constraint_evidence
        ):
            raise ValueError("passing matched projection constraint cites a failed operand")
        if self.disposition is MatchedActionHoldOutcomeProjectionDisposition.COMPLETE:
            if (
                self.reason_codes
                or not self.response_samples
                or not self.raw_values
                or not self.constraint_evidence
            ):
                raise ValueError("complete matched projection receipt is incomplete")
        elif (
            not self.reason_codes
            or self.response_samples
            or self.raw_values
            or self.operands
            or self.constraint_evidence
        ):
            raise ValueError("unevaluable matched projection receipt fabricates evidence")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("matched projection receipt is evaluator-reveal only")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldRevealedEpisode(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-revealed-episode'

    reveal_id: str
    sealed_episode: ObjectIdentity
    episode_id: str
    outcome_trace_id: str
    sealed_outcome_artifact: ArtifactIdentity
    delivery_trace: ExactActionDeliveryTrace | StageAwareActionDeliveryTrace
    response_samples: tuple[MatchedActionHoldReceiverSample, ...]
    predicate_values: tuple[NamedDecimal, ...]
    projection_receipt: MatchedActionHoldOutcomeProjectionEvidenceReceipt
    technical_reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        for name, value in (
            ("reveal_id", self.reveal_id),
            ("episode_id", self.episode_id),
            ("outcome_trace_id", self.outcome_trace_id),
        ):
            validate_stable_id(value, field_name=name)
        if not isinstance(
            self.delivery_trace,
            (ExactActionDeliveryTrace, StageAwareActionDeliveryTrace),
        ):
            raise ValueError("matched reveal uses an unknown delivery-trace schema")
        if self.sealed_episode.object_schema != MatchedActionHoldSealedEpisode.SCHEMA:
            raise ValueError("matched reveal binds another sealed episode schema")
        require_sorted_unique_ids(
            self.response_samples,
            attribute="sample_id",
            field_name="response_samples",
        )
        require_sorted_unique_ids(
            self.predicate_values,
            attribute="value_id",
            field_name="predicate_values",
        )
        require_sorted_unique_strings(
            self.technical_reason_codes,
            field_name="technical_reason_codes",
        )
        if (
            self.projection_receipt.episode_id != self.episode_id
            or self.projection_receipt.sealed_outcome_artifact != self.sealed_outcome_artifact
            or self.projection_receipt.response_samples != self.response_samples
            or self.projection_receipt.raw_values != self.predicate_values
        ):
            raise ValueError("matched reveal differs from its exact projection receipt")
        if self.outcome_trace_id != self.delivery_trace.trace_id:
            raise ValueError("matched episode must use one delivery/outcome trace identity")
        if self.delivery_trace.commitment_kind is ScientificCommitmentKind.NONATTEMPT and (
            self.response_samples or self.predicate_values
        ):
            raise ValueError("matched NONATTEMPT fabricates post-cutoff outcomes")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("matched revealed episode is evaluator-only")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldRevealedReference(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-revealed-reference'

    reference_id: str
    episode_id: str
    reference_class: ReferenceDispositionClass
    membership: ReferenceClassMembershipReceipt

    def __post_init__(self) -> None:
        validate_stable_id(self.reference_id, field_name="reference_id")
        validate_stable_id(self.episode_id, field_name="episode_id")
        if self.membership.reference_class != ObjectIdentity.from_record(
            self.reference_class.class_id,
            self.reference_class,
        ):
            raise ValueError("matched reference membership binds another sealed class")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldRevealIntegrityReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-reveal-integrity-receipt'

    integrity_id: str
    disposition: MatchedActionHoldRevealIntegrity
    binding_receipt: ObjectIdentity
    sealed_bundle: ObjectIdentity
    evidence_identities: tuple[ObjectIdentity, ...]
    reason_codes: tuple[str, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.integrity_id, field_name="integrity_id")
        require_sorted_unique_ids(
            self.evidence_identities,
            attribute="object_id",
            field_name="evidence_identities",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.binding_receipt.object_schema != (
            MatchedActionHoldEvaluationBindingReceipt.SCHEMA
        ):
            raise ValueError("matched reveal integrity binds another binding schema")
        if self.sealed_bundle.object_schema != MatchedActionHoldProspectiveBundle.SCHEMA:
            raise ValueError("matched reveal integrity binds another sealed bundle schema")
        if self.disposition is MatchedActionHoldRevealIntegrity.VALID:
            if not self.evidence_identities or self.reason_codes:
                raise ValueError("valid matched reveal lacks custody evidence")
        elif not self.reason_codes:
            raise ValueError("nonvalid matched reveal requires a decisive reason")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("matched reveal integrity is evaluator-only")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldRevealedBundle(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-revealed-bundle'

    reveal_id: str
    sealed_bundle: MatchedActionHoldProspectiveBundle
    integrity: MatchedActionHoldRevealIntegrityReceipt
    reveal_authorization: ObjectIdentity | None
    episodes: tuple[MatchedActionHoldRevealedEpisode, ...]
    references: tuple[MatchedActionHoldRevealedReference, ...]
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.reveal_id, field_name="reveal_id")
        require_sorted_unique_ids(
            self.episodes,
            attribute="episode_id",
            field_name="episodes",
        )
        require_sorted_unique_ids(
            self.references,
            attribute="episode_id",
            field_name="references",
        )
        bundle_identity = ObjectIdentity.from_record(
            self.sealed_bundle.bundle_id,
            self.sealed_bundle,
        )
        if (
            self.integrity.sealed_bundle != bundle_identity
            or self.integrity.binding_receipt
            != ObjectIdentity.from_record(
                self.sealed_bundle.binding_receipt.binding_id,
                self.sealed_bundle.binding_receipt,
            )
        ):
            raise ValueError("matched reveal integrity substitutes its sealed lineage")
        valid = self.integrity.disposition is MatchedActionHoldRevealIntegrity.VALID
        if valid and self.reveal_authorization is None:
            raise ValueError("valid matched reveal lacks reveal authority")
        sealed = {value.episode_id: value for value in self.sealed_bundle.episodes}
        if not set(value.episode_id for value in self.episodes) <= set(sealed):
            raise ValueError("matched reveal adds an unsealed episode")
        for value in self.episodes:
            source = sealed[value.episode_id]
            if (
                value.sealed_episode != ObjectIdentity.from_record(source.episode_id, source)
                or value.sealed_outcome_artifact != source.sealed_outcome_artifact
                or value.delivery_trace != source.delivery_trace
            ):
                raise ValueError("matched revealed episode changes its seal or delivery")
        if len({value.outcome_trace_id for value in self.episodes}) != len(self.episodes):
            raise ValueError("matched revealed episodes reuse an outcome trace")
        projection_receipts = tuple(value.projection_receipt for value in self.episodes)
        if len({value.receipt_id for value in projection_receipts}) != len(
            projection_receipts
        ) or len({value.projection.object_id for value in projection_receipts}) != len(
            projection_receipts
        ):
            raise ValueError("matched revealed episodes reuse projection evidence")

        reference_episodes = {
            value.episode_id
            for value in self.sealed_bundle.episodes
            if value.controller_tick is not None
        }
        if not set(value.episode_id for value in self.references) <= reference_episodes:
            raise ValueError("matched reference is not a committed-controller episode")
        plan = self.sealed_bundle.evaluation_plan
        for reference in self.references:
            source = sealed[reference.episode_id]
            tick = source.controller_tick
            if tick is None:  # pragma: no cover - narrowed by roster
                raise AssertionError("reference-bearing episode lost its tick")
            if reference.reference_class.reference_design != plan.finite_chart_reference_design:
                raise ValueError("matched reference uses another frozen chart")
            binding = tick.commitment.action_binding
            action_word_id = (
                binding.action_word.word_id
                if tick.commitment.disposition is CommitmentDisposition.ACTION_COMMITTED
                and binding is not None
                else None
            )
            expected = evaluate_reference_class_membership(
                reference_class=reference.reference_class,
                commitment_disposition=tick.commitment.disposition,
                committed_action_word_id=action_word_id,
                receipt_id=reference.membership.receipt_id,
            )
            if reference.membership != expected:
                raise ValueError("matched reference membership is not mechanically derived")

        if valid:
            expected_tick_count = sum(
                value.branch is not MatchedActionHoldExecutionBranch.QUALIFIED_HOLD_COUNTERFACTUAL
                for value in plan.execution_cells
            )
            if (
                len(self.episodes) != len(plan.execution_cells)
                or set(value.episode_id for value in self.episodes) != set(sealed)
                or len(self.references) != expected_tick_count
                or set(value.episode_id for value in self.references) != reference_episodes
            ):
                raise ValueError("valid matched reveal is not complete")
            projection_contract = plan.outcome_projection_evidence_contract
            projection_contract_identity = ObjectIdentity.from_record(
                projection_contract.contract_id,
                projection_contract,
            )
            predicates = {
                value.predicate_id: value for value in plan.evaluator_boundary.outcome_predicates
            }
            expected_constraint_pairs = {
                (predicate.predicate_id, constraint_id)
                for predicate in predicates.values()
                for constraint_id in predicate.constraint_ids
            }
            expected_raw_ids = {value.quantity_id for value in predicates.values()}
            for episode in self.episodes:
                projection = episode.projection_receipt
                if (
                    projection.projection_contract != projection_contract_identity
                    or projection.projection_specification
                    != projection_contract.projection_specification
                    or projection.projection.object_schema != projection_contract.projection_schema
                    or projection.projector_implementation_id
                    != projection_contract.projector_implementation_id
                    or projection.projector_implementation_sha256
                    != projection_contract.projector_implementation_sha256
                ):
                    raise ValueError("matched reveal uses foreign projection evidence")
                if _episode_is_nonattempt(sealed[episode.episode_id]):
                    if (
                        projection.disposition
                        is not MatchedActionHoldOutcomeProjectionDisposition.UNEVALUABLE
                        or episode.response_samples
                        or episode.predicate_values
                    ):
                        raise ValueError("matched NONATTEMPT carries outcomes")
                    continue
                if (
                    projection.disposition
                    is MatchedActionHoldOutcomeProjectionDisposition.UNEVALUABLE
                ):
                    continue
                if {value.value_id for value in episode.predicate_values} != expected_raw_ids:
                    raise ValueError("matched reveal changes the exact raw outcome roster")
                if set(plan.evaluator_boundary.raw_outcome_quantity_ids) != (
                    expected_raw_ids | {plan.effect_quantity_id}
                ):
                    raise ValueError("matched evaluator boundary changes scalar/response roles")
                observed_constraint_pairs = {
                    (value.predicate_id, value.constraint_id)
                    for value in projection.constraint_evidence
                }
                if observed_constraint_pairs != expected_constraint_pairs or any(
                    value.constraint_id not in predicates[value.predicate_id].constraint_ids
                    for value in projection.constraint_evidence
                ):
                    raise ValueError("matched projection constraint evidence is incomplete")
                if len(episode.response_samples) != len(plan.response_coordinates):
                    raise ValueError("matched episode lacks its projection-bound samples")
                if tuple(value.coordinate for value in episode.response_samples) != (
                    self.sealed_bundle.evaluation_plan.response_coordinates
                ):
                    raise ValueError("matched response samples change frozen coordinates")
                if any(
                    value.quantity_id != plan.effect_quantity_id
                    or value.value.unit != plan.effect_native_unit
                    for value in episode.response_samples
                ):
                    raise ValueError("matched response samples change quantity or unit")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("matched revealed bundle is evaluator-only")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldMemberEffect(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-member-effect'

    member_effect_id: str
    unit_id: str
    model_member_id: str
    action_phase_mean: NamedDecimal
    qualified_hold_phase_mean: NamedDecimal
    effect: NamedDecimal

    def __post_init__(self) -> None:
        for name, value in (
            ("member_effect_id", self.member_effect_id),
            ("unit_id", self.unit_id),
            ("model_member_id", self.model_member_id),
        ):
            validate_stable_id(value, field_name=name)
        if not (
            self.action_phase_mean.unit == self.qualified_hold_phase_mean.unit == self.effect.unit
        ):
            raise ValueError("matched member effect changes native units")
        if self.effect.value != (
            self.action_phase_mean.value - self.qualified_hold_phase_mean.value
        ):
            raise ValueError("matched member effect is not action minus qualified HOLD")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldEfficacyUnitEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-efficacy-unit-evaluation'

    unit_evaluation_id: str
    unit_id: str
    physical_independent_unit_id: str
    stratum_id: str
    disposition: MatchedActionHoldUnitDisposition
    member_effects: tuple[MatchedActionHoldMemberEffect, ...]
    robust_effect: NamedDecimal | None
    reference_memberships: tuple[ReferenceClassMembershipReceipt, ...]
    predicate_evaluations: tuple[OutcomePredicateEvaluation, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("unit_evaluation_id", self.unit_evaluation_id),
            ("unit_id", self.unit_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
            ("stratum_id", self.stratum_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.member_effects,
            attribute="model_member_id",
            field_name="member_effects",
        )
        require_sorted_unique_ids(
            self.reference_memberships,
            attribute="receipt_id",
            field_name="reference_memberships",
        )
        require_sorted_unique_ids(
            self.predicate_evaluations,
            attribute="evaluation_id",
            field_name="predicate_evaluations",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is MatchedActionHoldUnitDisposition.EFFICACY_EVALUABLE:
            if (
                not self.member_effects
                or self.robust_effect is None
                or self.robust_effect.value
                != min(value.effect.value for value in self.member_effects)
                or self.reason_codes
            ):
                raise ValueError("evaluable matched efficacy unit is incomplete")
        elif self.member_effects or self.robust_effect is not None or not self.reason_codes:
            raise ValueError("nonevaluable matched efficacy unit retains effects")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldControlMemberEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-control-member-evaluation'

    member_evaluation_id: str
    model_member_id: str
    disposition: MatchedActionHoldUnitDisposition
    reference_membership: ReferenceClassMembershipReceipt
    predicate_evaluations: tuple[OutcomePredicateEvaluation, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.member_evaluation_id, field_name="member_evaluation_id")
        validate_stable_id(self.model_member_id, field_name="model_member_id")
        require_sorted_unique_ids(
            self.predicate_evaluations,
            attribute="evaluation_id",
            field_name="predicate_evaluations",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if self.disposition is MatchedActionHoldUnitDisposition.HOLD_CONTROL_CORRECT:
            if self.reason_codes:
                raise ValueError("correct matched HOLD control cannot carry reasons")
        elif not self.reason_codes:
            raise ValueError("failed matched HOLD control requires a reason")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldControlEvaluation(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-control-evaluation'

    control_evaluation_id: str
    unit_id: str
    physical_independent_unit_id: str
    hold_anchor_slot_id: str
    disposition: MatchedActionHoldUnitDisposition
    member_evaluations: tuple[MatchedActionHoldControlMemberEvaluation, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        for name, value in (
            ("control_evaluation_id", self.control_evaluation_id),
            ("unit_id", self.unit_id),
            ("physical_independent_unit_id", self.physical_independent_unit_id),
            ("hold_anchor_slot_id", self.hold_anchor_slot_id),
        ):
            validate_stable_id(value, field_name=name)
        require_sorted_unique_ids(
            self.member_evaluations,
            attribute="model_member_id",
            field_name="member_evaluations",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        if not self.member_evaluations:
            if (
                self.disposition is not MatchedActionHoldUnitDisposition.UNEVALUABLE
                or not self.reason_codes
            ):
                raise ValueError("empty HOLD control is reserved for an integrity stop")
            return
        all_correct = all(
            value.disposition is MatchedActionHoldUnitDisposition.HOLD_CONTROL_CORRECT
            for value in self.member_evaluations
        )
        if self.disposition is MatchedActionHoldUnitDisposition.HOLD_CONTROL_CORRECT:
            if not all_correct or self.reason_codes:
                raise ValueError("matched HOLD control aggregate is not correct")
        elif all_correct or not self.reason_codes:
            raise ValueError("failed matched HOLD control aggregate is malformed")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldDistributionSummary(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-distribution-summary'

    summary_id: str
    group_id: str
    independent_unit_count: int
    mean_effect: NamedDecimal
    sample_sd: NamedDecimal
    median_effect: NamedDecimal
    minimum_effect: NamedDecimal
    maximum_effect: NamedDecimal

    def __post_init__(self) -> None:
        validate_stable_id(self.summary_id, field_name="summary_id")
        validate_stable_id(self.group_id, field_name="group_id")
        if self.independent_unit_count < 1:
            raise ValueError("matched distribution requires physical units")
        units = {
            self.mean_effect.unit,
            self.sample_sd.unit,
            self.median_effect.unit,
            self.minimum_effect.unit,
            self.maximum_effect.unit,
        }
        if len(units) != 1 or self.minimum_effect.value > self.maximum_effect.value:
            raise ValueError("matched distribution changes units or range")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldStratumStrengthening(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-stratum-strengthening'

    stratum_id: str
    all_unit_effects_positive: bool

    def __post_init__(self) -> None:
        validate_stable_id(self.stratum_id, field_name="stratum_id")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldAllPositiveStrengtheningAnalysis(CanonicalRecord):
    """Diagnostic strengthening analysis; never changes the primary result."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-all-positive-strengthening-analysis'

    analysis_id: str
    evaluable: bool
    all_efficacy_effects_positive: bool | None
    strata: tuple[MatchedActionHoldStratumStrengthening, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.analysis_id, field_name="analysis_id")
        require_sorted_unique_ids(self.strata, attribute="stratum_id", field_name="strata")
        if self.evaluable:
            if self.all_efficacy_effects_positive is None or not self.strata:
                raise ValueError("evaluable strengthening analysis is incomplete")
        elif self.all_efficacy_effects_positive is not None or self.strata:
            raise ValueError("unevaluable strengthening analysis fabricates diagnostics")


@dataclass(frozen=True, slots=True)
class MatchedActionHoldCohortAdjudication(CanonicalRecord):
    "Sole matched controller use result; controls remain outside every efficacy statistic."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/runtime/matched-action-hold-cohort-adjudication'

    adjudication_id: str
    evaluation_plan: ObjectIdentity
    binding_receipt: ObjectIdentity
    compiled_study: ObjectIdentity
    revealed_bundle: ObjectIdentity
    efficacy_units: tuple[MatchedActionHoldEfficacyUnitEvaluation, ...]
    hold_controls: tuple[MatchedActionHoldControlEvaluation, ...]
    rostered_efficacy_unit_count: int
    effective_independent_unit_count: int
    mean_effect: NamedDecimal | None
    sample_sd: NamedDecimal | None
    median_effect: NamedDecimal | None
    minimum_effect: NamedDecimal | None
    maximum_effect: NamedDecimal | None
    one_sided_lower_bound: NamedDecimal | None
    active_coverage: Decimal
    reference_correctness_coverage: Decimal
    member_distributions: tuple[MatchedActionHoldDistributionSummary, ...]
    stratum_distributions: tuple[MatchedActionHoldDistributionSummary, ...]
    strengthening_analysis: MatchedActionHoldAllPositiveStrengtheningAnalysis
    result: MatchedActionHoldProspectiveResult
    claim_ceiling: str
    evidence_ceiling: EvidenceCeiling
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        validate_stable_id(self.adjudication_id, field_name="adjudication_id")
        require_sorted_unique_ids(
            self.efficacy_units,
            attribute="unit_id",
            field_name="efficacy_units",
        )
        require_sorted_unique_ids(
            self.hold_controls,
            attribute="unit_id",
            field_name="hold_controls",
        )
        require_sorted_unique_ids(
            self.member_distributions,
            attribute="group_id",
            field_name="member_distributions",
        )
        require_sorted_unique_ids(
            self.stratum_distributions,
            attribute="group_id",
            field_name="stratum_distributions",
        )
        require_sorted_unique_strings(self.reason_codes, field_name="reason_codes")
        for name, value in (
            ("active_coverage", self.active_coverage),
            ("reference_correctness_coverage", self.reference_correctness_coverage),
        ):
            validate_decimal(value, field_name=name, minimum=Decimal(0))
            if value > 1:
                raise ValueError(f"{name} cannot exceed one")
        if (
            len(self.efficacy_units) < 2
            or not self.hold_controls
            or self.rostered_efficacy_unit_count != len(self.efficacy_units)
        ):
            raise ValueError("matched adjudication changes its scientific roster")
        evaluable = sum(
            value.disposition is MatchedActionHoldUnitDisposition.EFFICACY_EVALUABLE
            for value in self.efficacy_units
        )
        if self.effective_independent_unit_count != evaluable:
            raise ValueError("matched effective n is not the physical-unit count")
        summaries = (
            self.mean_effect,
            self.sample_sd,
            self.median_effect,
            self.minimum_effect,
            self.maximum_effect,
            self.one_sided_lower_bound,
        )
        if evaluable == self.rostered_efficacy_unit_count:
            member_ids = {
                value.model_member_id
                for unit in self.efficacy_units
                for value in unit.member_effects
            }
            if any(
                {value.model_member_id for value in unit.member_effects} != member_ids
                for unit in self.efficacy_units
            ) or any(
                {value.model_member_id for value in control.member_evaluations} != member_ids
                for control in self.hold_controls
            ):
                raise ValueError("matched adjudication has inconsistent nested member rosters")
            stratum_ids = {value.stratum_id for value in self.efficacy_units}
            if (
                any(value is None for value in summaries)
                or {value.group_id for value in self.member_distributions} != member_ids
                or {value.group_id for value in self.stratum_distributions} != stratum_ids
                or not self.strengthening_analysis.evaluable
                or {value.stratum_id for value in self.strengthening_analysis.strata} != stratum_ids
            ):
                raise ValueError("complete matched efficacy lacks frozen summaries")
        elif (
            any(value is not None for value in summaries)
            or self.member_distributions
            or self.stratum_distributions
            or self.strengthening_analysis.evaluable
        ):
            raise ValueError("partial matched efficacy fabricates cohort inference")
        if self.result is MatchedActionHoldProspectiveResult.TERMINAL_PRIMARY_CONTROLLER_USE_REPLICATION_POSITIVE:
            if self.reason_codes:
                raise ValueError("positive matched adjudication cannot carry reasons")
        elif not self.reason_codes:
            raise ValueError("nonpositive matched adjudication requires a reason")
        if self.evidence_ceiling is not EvidenceCeiling.CONTROLLER_USE:
            raise ValueError("matched adjudication must remain at controller use")


def _median(values: tuple[Decimal, ...]) -> Decimal:
    if not values:
        raise ValueError("matched median requires values")
    ordered = tuple(sorted(values))
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / Decimal(2)


def _phase_mean(
    samples: tuple[MatchedActionHoldReceiverSample, ...],
) -> Decimal:
    return sum((value.value.value for value in samples), Decimal(0)) / Decimal(len(samples))


def _distribution_summary(
    *,
    summary_id: str,
    group_id: str,
    values: tuple[Decimal, ...],
    unit: str,
) -> MatchedActionHoldDistributionSummary:
    mean, sample_sd, _ = controller_prospective_student_summary(
        values,
        one_sided_critical_value=Decimal(0),
    )
    return MatchedActionHoldDistributionSummary(
        summary_id=summary_id,
        group_id=group_id,
        independent_unit_count=len(values),
        mean_effect=NamedDecimal(
            value_id=f"mean.{summary_id}",
            value=mean,
            unit=unit,
        ),
        sample_sd=NamedDecimal(
            value_id=f"sample-sd.{summary_id}",
            value=sample_sd,
            unit=unit,
        ),
        median_effect=NamedDecimal(
            value_id=f"median.{summary_id}",
            value=_median(values),
            unit=unit,
        ),
        minimum_effect=NamedDecimal(
            value_id=f"minimum.{summary_id}",
            value=min(values),
            unit=unit,
        ),
        maximum_effect=NamedDecimal(
            value_id=f"maximum.{summary_id}",
            value=max(values),
            unit=unit,
        ),
    )


def _failure_disposition(
    dispositions: set[MatchedActionHoldUnitDisposition],
) -> MatchedActionHoldUnitDisposition:
    return next(
        value
        for value in (
            MatchedActionHoldUnitDisposition.REFERENCE_INCORRECT,
            MatchedActionHoldUnitDisposition.UNSAFE,
            MatchedActionHoldUnitDisposition.DELIVERY_INVALID,
            MatchedActionHoldUnitDisposition.TECHNICAL_PARTIAL,
            MatchedActionHoldUnitDisposition.NONATTEMPT,
            MatchedActionHoldUnitDisposition.UNEVALUABLE,
        )
        if value in dispositions
    )


class MatchedActionHoldProspectiveEvaluator:
    """Pure evaluator for one exact frozen parameterised matched design."""

    def __init__(self, implementation_binding: ImplementationBinding) -> None:
        if implementation_binding.role is not ImplementationRole.OUTCOME_EVALUATOR:
            raise ValueError("matched controller use evaluator uses another implementation role")
        self.implementation_binding = implementation_binding

    def _validate_context(
        self,
        *,
        compiled: CompiledAdmissionControllerStudy,
        revealed: MatchedActionHoldRevealedBundle,
    ) -> MatchedActionHoldControllerEvaluationPlan:
        bundle = revealed.sealed_bundle
        receipt = bundle.binding_receipt
        compiled_identity = ObjectIdentity.from_record(
            compiled.compiled_study_id,
            compiled,
        )
        programme_identity = ObjectIdentity.from_record(
            compiled.study.study_id,
            compiled.study,
        )
        if (
            compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.NOT_DECLARED
            or compiled.prospective_evaluation_binding.evaluation_plan is not None
            or compiled.prospective_evaluation_binding.evaluator is not None
            or compiled.study.prospective_evaluation is not None
        ):
            raise ValueError("matched controller use requires a compiled admission controller with controller use NOT_DECLARED")
        if any(
            value.role is ImplementationRole.OUTCOME_EVALUATOR
            for value in compiled.study.implementations
        ):
            raise ValueError("matched programme capped at admission embeds an outcome evaluator")
        plan = bundle.evaluation_plan
        if (
            bundle.compiled_study != compiled_identity
            or receipt.compiled_study != compiled_identity
            or receipt.study != programme_identity
            or receipt.evaluator_binding != self.implementation_binding
            or plan.evaluator_binding != self.implementation_binding
            or plan.evaluator_boundary.outcome_evaluator_binding_id
            != self.implementation_binding.binding_id
        ):
            raise ValueError("matched evaluator context substitutes frozen identities")
        if (
            self.implementation_binding.reference.input_schema
            != MatchedActionHoldRevealedBundle.SCHEMA
            or self.implementation_binding.reference.output_schema
            != MatchedActionHoldCohortAdjudication.SCHEMA
        ):
            raise ValueError("matched evaluator implementation uses another exact port")
        measured_hold = compiled.study.measured_hold_fibre
        if (
            measured_hold is None
            or plan.measured_hold_fibre
            != ObjectIdentity.from_record(measured_hold.hold_fibre_id, measured_hold)
            or measured_hold.calibration_id != plan.native_hold_calibration.object_id
            or compiled.hold_fibre is None
            or compiled.hold_fibre.action_binding.action_word != plan.qualified_hold_word
            or compiled.hold_fibre.source.decision_cell_id != plan.hold_decision_cell_id
            or tuple(value.decision_cell_id for value in compiled.action_table)
            != plan.active_decision_cell_ids
            or not any(
                value.action_binding.action_word == plan.qualified_action_word
                for value in compiled.action_table
            )
        ):
            raise ValueError("matched evaluator changes action/HOLD compiled semantics")
        stage_aware_delivery = isinstance(
            measured_hold.delivery_equivalence,
            StageAwareDeliveryEquivalenceSpec,
        )
        for episode in bundle.episodes:
            trace = episode.delivery_trace
            tick = episode.controller_tick
            if trace.commitment_kind is ScientificCommitmentKind.NONATTEMPT:
                if isinstance(trace, StageAwareActionDeliveryTrace) or isinstance(
                    tick,
                    StageAwareAdmissionControllerTickReceipt,
                ):
                    raise ValueError("matched NONATTEMPT cannot claim stage-aware delivery")
                continue
            branch = episode.execution_cell.branch
            active_branch = branch is MatchedActionHoldExecutionBranch.COMMITTED_CONTROLLER_ACTION
            expected_word = (
                plan.qualified_action_word if active_branch else plan.qualified_hold_word
            )
            if trace.expected_action_word != expected_word:
                raise ValueError("matched delivery changes its exact compiled action word")
            if tick is not None:
                action_binding = tick.commitment.action_binding
                if active_branch:
                    compiled_bindings = tuple(
                        value.action_binding for value in compiled.action_table
                    )
                    if action_binding not in compiled_bindings:
                        raise ValueError(
                            "matched active commitment is absent from the compiled action table"
                        )
                elif (
                    compiled.hold_fibre is None
                    or action_binding != compiled.hold_fibre.action_binding
                ):
                    raise ValueError(
                        "matched HOLD commitment differs from the compiled measured fibre"
                    )
            if stage_aware_delivery:
                if (
                    not isinstance(trace, StageAwareActionDeliveryTrace)
                    or trace.delivery_equivalence != measured_hold.delivery_equivalence
                    or (tick is not None and not isinstance(tick, StageAwareAdmissionControllerTickReceipt))
                ):
                    raise ValueError(
                        "matched delivery differs from its compiled stage-aware specification"
                    )
            elif isinstance(trace, StageAwareActionDeliveryTrace) or isinstance(
                tick,
                StageAwareAdmissionControllerTickReceipt,
            ):
                raise ValueError("matched delivery invents an unbound stage equivalence")
        delivery = compiled.implementation(ImplementationRole.DELIVERY)
        if any(value.delivery_trace.delivery != delivery for value in bundle.episodes):
            raise ValueError("matched episode uses another compiled delivery")
        return plan

    @staticmethod
    def _predicate_evaluations(
        *,
        plan: MatchedActionHoldControllerEvaluationPlan,
        episode: MatchedActionHoldRevealedEpisode,
        label: str,
    ) -> tuple[OutcomePredicateEvaluation, ...]:
        receipt = episode.projection_receipt
        if receipt.disposition is MatchedActionHoldOutcomeProjectionDisposition.UNEVALUABLE:
            return tuple(
                OutcomePredicateEvaluation(
                    evaluation_id=f"outcome-gate.{label}.{predicate.predicate_id}",
                    predicate_id=predicate.predicate_id,
                    status=GateStatus.UNEVALUABLE,
                    margin=None,
                    reason_codes=tuple(
                        sorted(
                            {
                                "OUTCOME_PROJECTION_UNEVALUABLE",
                                *receipt.reason_codes,
                            }
                        )
                    ),
                )
                for predicate in plan.evaluator_boundary.outcome_predicates
            )
        raw = {value.value_id: value for value in episode.predicate_values}
        constraint_by_predicate = {
            predicate.predicate_id: tuple(
                value
                for value in receipt.constraint_evidence
                if value.predicate_id == predicate.predicate_id
            )
            for predicate in plan.evaluator_boundary.outcome_predicates
        }
        evaluations: list[OutcomePredicateEvaluation] = []
        for predicate in plan.evaluator_boundary.outcome_predicates:
            scalar = evaluate_controller_outcome_predicate(
                evaluation_id=f"outcome-gate.{label}.{predicate.predicate_id}",
                predicate=predicate,
                raw=raw,
            )
            constraints = constraint_by_predicate[predicate.predicate_id]
            failed = tuple(value for value in constraints if value.status is GateStatus.FAIL)
            unevaluable = tuple(
                value for value in constraints if value.status is GateStatus.UNEVALUABLE
            )
            if scalar.status is GateStatus.FAIL or failed:
                reasons = {
                    *scalar.reason_codes,
                    *(reason for value in failed for reason in value.reason_codes),
                }
                if failed:
                    reasons.add("OUTCOME_CONSTRAINT_FAILED")
                evaluations.append(
                    OutcomePredicateEvaluation(
                        evaluation_id=scalar.evaluation_id,
                        predicate_id=scalar.predicate_id,
                        status=GateStatus.FAIL,
                        margin=scalar.margin,
                        reason_codes=tuple(sorted(reasons)),
                    )
                )
            elif scalar.status is GateStatus.UNEVALUABLE or unevaluable:
                reasons = {
                    *scalar.reason_codes,
                    *(reason for value in unevaluable for reason in value.reason_codes),
                }
                if unevaluable:
                    reasons.add("OUTCOME_CONSTRAINT_UNEVALUABLE")
                evaluations.append(
                    OutcomePredicateEvaluation(
                        evaluation_id=scalar.evaluation_id,
                        predicate_id=scalar.predicate_id,
                        status=GateStatus.UNEVALUABLE,
                        margin=scalar.margin,
                        reason_codes=tuple(sorted(reasons)),
                    )
                )
            else:
                evaluations.append(scalar)
        return tuple(evaluations)

    @staticmethod
    def _delivery_valid(
        episode: MatchedActionHoldRevealedEpisode,
        *,
        expected_kind: ScientificCommitmentKind,
        expected_word: OccurrenceActionWord,
    ) -> bool:
        trace = episode.delivery_trace
        return (
            trace.commitment_kind is expected_kind
            and trace.expected_action_word == expected_word
            and trace.operational_state is OperationalDeliveryState.DELIVERED
            and (
                trace.equivalent
                if isinstance(trace, StageAwareActionDeliveryTrace)
                else trace.exact
            )
        )

    @staticmethod
    def _reference_map(
        revealed: MatchedActionHoldRevealedBundle,
    ) -> dict[str, MatchedActionHoldRevealedReference]:
        return {value.episode_id: value for value in revealed.references}

    def _evaluate_efficacy_unit(
        self,
        *,
        plan: MatchedActionHoldControllerEvaluationPlan,
        revealed: MatchedActionHoldRevealedBundle,
        unit_id: str,
    ) -> MatchedActionHoldEfficacyUnitEvaluation:
        unit = next(value for value in plan.efficacy_units if value.unit_id == unit_id)
        if unit.stratum_id is None:  # pragma: no cover - frozen by plan
            raise AssertionError("matched efficacy unit lost its stratum")
        sealed_by_product = {
            value.execution_cell.product_key: value for value in revealed.sealed_bundle.episodes
        }
        outcomes = {value.episode_id: value for value in revealed.episodes}
        references = self._reference_map(revealed)
        effects: list[MatchedActionHoldMemberEffect] = []
        memberships: list[ReferenceClassMembershipReceipt] = []
        predicates: list[OutcomePredicateEvaluation] = []
        failures: set[MatchedActionHoldUnitDisposition] = set()
        reasons: set[str] = set()
        for member_id in plan.model_member_ids:
            action_sealed = sealed_by_product[
                (
                    unit_id,
                    member_id,
                    MatchedActionHoldExecutionBranch.COMMITTED_CONTROLLER_ACTION,
                )
            ]
            hold_sealed = sealed_by_product[
                (
                    unit_id,
                    member_id,
                    MatchedActionHoldExecutionBranch.QUALIFIED_HOLD_COUNTERFACTUAL,
                )
            ]
            reference = references[action_sealed.episode_id]
            memberships.append(reference.membership)
            action = outcomes[action_sealed.episode_id]
            hold = outcomes[hold_sealed.episode_id]
            if _episode_is_nonattempt(action_sealed) or _episode_is_nonattempt(hold_sealed):
                failures.add(MatchedActionHoldUnitDisposition.NONATTEMPT)
                reasons.add("MATCHED_PAIR_NONATTEMPT")
                continue
            action_predicates = self._predicate_evaluations(
                plan=plan,
                episode=action,
                label=f"{unit_id}.{member_id}.action",
            )
            hold_predicates = self._predicate_evaluations(
                plan=plan,
                episode=hold,
                label=f"{unit_id}.{member_id}.hold",
            )
            predicates.extend((*action_predicates, *hold_predicates))
            if reference.membership.status is not ReferenceMembershipStatus.MEMBER:
                failures.add(MatchedActionHoldUnitDisposition.REFERENCE_INCORRECT)
                reasons.add("COMMITTED_ACTION_OUTSIDE_SEALED_REFERENCE")
                continue
            if any(
                value.status is GateStatus.FAIL for value in (*action_predicates, *hold_predicates)
            ):
                failures.add(MatchedActionHoldUnitDisposition.UNSAFE)
                reasons.add("MATCHED_NATIVE_SAFETY_PREDICATE_FAILED")
                continue
            if not self._delivery_valid(
                action,
                expected_kind=ScientificCommitmentKind.ACTION,
                expected_word=plan.qualified_action_word,
            ) or not self._delivery_valid(
                hold,
                expected_kind=ScientificCommitmentKind.MEASURED_HOLD,
                expected_word=plan.qualified_hold_word,
            ):
                failures.add(MatchedActionHoldUnitDisposition.DELIVERY_INVALID)
                reasons.add("REQUESTED_ACCEPTED_APPLIED_REALIZED_MISMATCH")
                continue
            if action.technical_reason_codes or hold.technical_reason_codes:
                failures.add(MatchedActionHoldUnitDisposition.TECHNICAL_PARTIAL)
                reasons.update(action.technical_reason_codes)
                reasons.update(hold.technical_reason_codes)
                continue
            if any(
                value.status is GateStatus.UNEVALUABLE
                for value in (*action_predicates, *hold_predicates)
            ):
                failures.add(MatchedActionHoldUnitDisposition.UNEVALUABLE)
                reasons.add("MATCHED_NATIVE_SAFETY_PREDICATE_UNEVALUABLE")
                continue
            if len(action.response_samples) != len(plan.response_coordinates) or len(
                hold.response_samples
            ) != len(plan.response_coordinates):
                failures.add(MatchedActionHoldUnitDisposition.UNEVALUABLE)
                reasons.add("MATCHED_RECEIVER_PHASE_INCOMPLETE")
                continue
            action_mean = _phase_mean(action.response_samples)
            hold_mean = _phase_mean(hold.response_samples)
            effects.append(
                MatchedActionHoldMemberEffect(
                    member_effect_id=f"member-effect.{unit_id}.{member_id}",
                    unit_id=unit_id,
                    model_member_id=member_id,
                    action_phase_mean=NamedDecimal(
                        value_id=f"action-phase-mean.{unit_id}.{member_id}",
                        value=action_mean,
                        unit=plan.effect_native_unit,
                    ),
                    qualified_hold_phase_mean=NamedDecimal(
                        value_id=f"hold-phase-mean.{unit_id}.{member_id}",
                        value=hold_mean,
                        unit=plan.effect_native_unit,
                    ),
                    effect=NamedDecimal(
                        value_id=f"member-effect-value.{unit_id}.{member_id}",
                        value=action_mean - hold_mean,
                        unit=plan.effect_native_unit,
                    ),
                )
            )
        if len(effects) == len(plan.model_member_ids) and not failures:
            disposition = MatchedActionHoldUnitDisposition.EFFICACY_EVALUABLE
            robust = NamedDecimal(
                value_id=f"robust-effect.{unit_id}",
                value=min(value.effect.value for value in effects),
                unit=plan.effect_native_unit,
            )
            output_reasons: tuple[str, ...] = ()
        else:
            effects = []
            disposition = _failure_disposition(failures)
            robust = None
            output_reasons = tuple(sorted(reasons))
        return MatchedActionHoldEfficacyUnitEvaluation(
            unit_evaluation_id=f"efficacy-evaluation.{unit_id}",
            unit_id=unit_id,
            physical_independent_unit_id=unit.physical_independent_unit_id,
            stratum_id=unit.stratum_id,
            disposition=disposition,
            member_effects=tuple(sorted(effects, key=lambda value: value.model_member_id)),
            robust_effect=robust,
            reference_memberships=tuple(sorted(memberships, key=lambda value: value.receipt_id)),
            predicate_evaluations=tuple(sorted(predicates, key=lambda value: value.evaluation_id)),
            reason_codes=output_reasons,
        )

    def _evaluate_control(
        self,
        *,
        plan: MatchedActionHoldControllerEvaluationPlan,
        revealed: MatchedActionHoldRevealedBundle,
        unit_id: str,
    ) -> MatchedActionHoldControlEvaluation:
        unit = next(value for value in plan.hold_controls if value.unit_id == unit_id)
        if unit.hold_anchor_slot_id is None:  # pragma: no cover - frozen by plan
            raise AssertionError("matched HOLD control lost its anchor")
        sealed_by_product = {
            value.execution_cell.product_key: value for value in revealed.sealed_bundle.episodes
        }
        outcomes = {value.episode_id: value for value in revealed.episodes}
        references = self._reference_map(revealed)
        member_evaluations: list[MatchedActionHoldControlMemberEvaluation] = []
        for member_id in plan.model_member_ids:
            sealed = sealed_by_product[
                (
                    unit_id,
                    member_id,
                    MatchedActionHoldExecutionBranch.COMMITTED_HOLD_CONTROL,
                )
            ]
            outcome = outcomes[sealed.episode_id]
            membership = references[sealed.episode_id].membership
            predicate_evaluations: tuple[OutcomePredicateEvaluation, ...] = ()
            reasons: set[str] = set()
            if _episode_is_nonattempt(sealed):
                disposition = MatchedActionHoldUnitDisposition.NONATTEMPT
                reasons.add("HOLD_CONTROL_NONATTEMPT")
            else:
                predicate_evaluations = self._predicate_evaluations(
                    plan=plan,
                    episode=outcome,
                    label=f"{unit_id}.{member_id}.control",
                )
                if membership.status is not ReferenceMembershipStatus.MEMBER:
                    disposition = MatchedActionHoldUnitDisposition.REFERENCE_INCORRECT
                    reasons.add("HOLD_CONTROL_OUTSIDE_SEALED_REFERENCE")
                elif any(value.status is GateStatus.FAIL for value in predicate_evaluations):
                    disposition = MatchedActionHoldUnitDisposition.UNSAFE
                    reasons.add("HOLD_CONTROL_SAFETY_PREDICATE_FAILED")
                elif not self._delivery_valid(
                    outcome,
                    expected_kind=ScientificCommitmentKind.MEASURED_HOLD,
                    expected_word=plan.qualified_hold_word,
                ):
                    disposition = MatchedActionHoldUnitDisposition.DELIVERY_INVALID
                    reasons.add("HOLD_CONTROL_DELIVERY_INVALID")
                elif outcome.technical_reason_codes:
                    disposition = MatchedActionHoldUnitDisposition.TECHNICAL_PARTIAL
                    reasons.update(outcome.technical_reason_codes)
                elif any(value.status is GateStatus.UNEVALUABLE for value in predicate_evaluations):
                    disposition = MatchedActionHoldUnitDisposition.UNEVALUABLE
                    reasons.add("HOLD_CONTROL_SAFETY_PREDICATE_UNEVALUABLE")
                else:
                    disposition = MatchedActionHoldUnitDisposition.HOLD_CONTROL_CORRECT
            member_evaluations.append(
                MatchedActionHoldControlMemberEvaluation(
                    member_evaluation_id=f"control-member.{unit_id}.{member_id}",
                    model_member_id=member_id,
                    disposition=disposition,
                    reference_membership=membership,
                    predicate_evaluations=predicate_evaluations,
                    reason_codes=tuple(sorted(reasons)),
                )
            )
        failures: set[MatchedActionHoldUnitDisposition] = {
            value.disposition
            for value in member_evaluations
            if value.disposition is not MatchedActionHoldUnitDisposition.HOLD_CONTROL_CORRECT
        }
        if not failures:
            disposition = MatchedActionHoldUnitDisposition.HOLD_CONTROL_CORRECT
            aggregate_reasons: tuple[str, ...] = ()
        else:
            disposition = _failure_disposition(failures)
            aggregate_reasons = tuple(
                sorted({reason for value in member_evaluations for reason in value.reason_codes})
            )
        return MatchedActionHoldControlEvaluation(
            control_evaluation_id=f"hold-control-evaluation.{unit_id}",
            unit_id=unit_id,
            physical_independent_unit_id=unit.physical_independent_unit_id,
            hold_anchor_slot_id=unit.hold_anchor_slot_id,
            disposition=disposition,
            member_evaluations=tuple(
                sorted(member_evaluations, key=lambda value: value.model_member_id)
            ),
            reason_codes=aggregate_reasons,
        )

    @staticmethod
    def _integrity_stop(
        *,
        plan: MatchedActionHoldControllerEvaluationPlan,
        revealed: MatchedActionHoldRevealedBundle,
        result: MatchedActionHoldProspectiveResult,
    ) -> MatchedActionHoldCohortAdjudication:
        reasons = tuple(
            sorted(
                {
                    result.value,
                    *revealed.integrity.reason_codes,
                }
            )
        )
        efficacy = tuple(
            MatchedActionHoldEfficacyUnitEvaluation(
                unit_evaluation_id=f"efficacy-evaluation.{unit.unit_id}",
                unit_id=unit.unit_id,
                physical_independent_unit_id=unit.physical_independent_unit_id,
                stratum_id=unit.stratum_id or "missing-stratum",
                disposition=MatchedActionHoldUnitDisposition.UNEVALUABLE,
                member_effects=(),
                robust_effect=None,
                reference_memberships=(),
                predicate_evaluations=(),
                reason_codes=reasons,
            )
            for unit in plan.efficacy_units
        )
        controls = tuple(
            MatchedActionHoldControlEvaluation(
                control_evaluation_id=f"hold-control-evaluation.{unit.unit_id}",
                unit_id=unit.unit_id,
                physical_independent_unit_id=unit.physical_independent_unit_id,
                hold_anchor_slot_id=unit.hold_anchor_slot_id or "missing-anchor",
                disposition=MatchedActionHoldUnitDisposition.UNEVALUABLE,
                member_evaluations=(),
                reason_codes=reasons,
            )
            for unit in plan.hold_controls
        )
        return MatchedActionHoldCohortAdjudication(
            adjudication_id=f"matched-adjudication.{plan.evaluation_plan_id}",
            evaluation_plan=ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
            binding_receipt=ObjectIdentity.from_record(
                revealed.sealed_bundle.binding_receipt.binding_id,
                revealed.sealed_bundle.binding_receipt,
            ),
            compiled_study=revealed.sealed_bundle.compiled_study,
            revealed_bundle=ObjectIdentity.from_record(revealed.reveal_id, revealed),
            efficacy_units=efficacy,
            hold_controls=controls,
            rostered_efficacy_unit_count=len(plan.efficacy_units),
            effective_independent_unit_count=0,
            mean_effect=None,
            sample_sd=None,
            median_effect=None,
            minimum_effect=None,
            maximum_effect=None,
            one_sided_lower_bound=None,
            active_coverage=Decimal(0),
            reference_correctness_coverage=Decimal(0),
            member_distributions=(),
            stratum_distributions=(),
            strengthening_analysis=(
                MatchedActionHoldAllPositiveStrengtheningAnalysis(
                    analysis_id=f"all-positive.{plan.evaluation_plan_id}",
                    evaluable=False,
                    all_efficacy_effects_positive=None,
                    strata=(),
                )
            ),
            result=result,
            claim_ceiling=plan.maximum_claim_ceiling,
            evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
            reason_codes=reasons,
        )

    def evaluate(
        self,
        *,
        compiled: CompiledAdmissionControllerStudy,
        revealed: MatchedActionHoldRevealedBundle,
    ) -> MatchedActionHoldCohortAdjudication:
        plan = self._validate_context(compiled=compiled, revealed=revealed)
        integrity = revealed.integrity.disposition
        if integrity is MatchedActionHoldRevealIntegrity.AUTHORITY_OR_RESOURCE_STOP:
            return self._integrity_stop(
                plan=plan,
                revealed=revealed,
                result=MatchedActionHoldProspectiveResult.AUTHORITY_OR_RESOURCE_STOP,
            )
        if integrity is not MatchedActionHoldRevealIntegrity.VALID:
            return self._integrity_stop(
                plan=plan,
                revealed=revealed,
                result=(MatchedActionHoldProspectiveResult.RECEIPT_CUSTODY_OR_REVEAL_INVALID),
            )

        efficacy = tuple(
            self._evaluate_efficacy_unit(
                plan=plan,
                revealed=revealed,
                unit_id=unit.unit_id,
            )
            for unit in plan.efficacy_units
        )
        controls = tuple(
            self._evaluate_control(
                plan=plan,
                revealed=revealed,
                unit_id=unit.unit_id,
            )
            for unit in plan.hold_controls
        )
        evaluable = tuple(
            value
            for value in efficacy
            if value.disposition is MatchedActionHoldUnitDisposition.EFFICACY_EVALUABLE
        )
        effective_n = len(evaluable)
        all_references = tuple(value.membership for value in revealed.references)
        correct_references = sum(
            value.status is ReferenceMembershipStatus.MEMBER for value in all_references
        )
        expected_reference_count = sum(
            value.branch is not MatchedActionHoldExecutionBranch.QUALIFIED_HOLD_COUNTERFACTUAL
            for value in plan.execution_cells
        )
        reference_coverage = Decimal(correct_references) / Decimal(expected_reference_count)
        active_committed = sum(
            episode.execution_cell.branch
            is MatchedActionHoldExecutionBranch.COMMITTED_CONTROLLER_ACTION
            and episode.controller_tick is not None
            and episode.controller_tick.commitment.disposition
            is CommitmentDisposition.ACTION_COMMITTED
            for episode in revealed.sealed_bundle.episodes
        )
        active_coverage = Decimal(active_committed) / Decimal(
            len(plan.efficacy_units) * len(plan.model_member_ids)
        )

        mean_effect: NamedDecimal | None = None
        sample_sd: NamedDecimal | None = None
        median_effect: NamedDecimal | None = None
        minimum_effect: NamedDecimal | None = None
        maximum_effect: NamedDecimal | None = None
        lower_bound: NamedDecimal | None = None
        member_distributions: tuple[MatchedActionHoldDistributionSummary, ...] = ()
        stratum_distributions: tuple[MatchedActionHoldDistributionSummary, ...] = ()
        strengthening = MatchedActionHoldAllPositiveStrengtheningAnalysis(
            analysis_id=f"all-positive.{plan.evaluation_plan_id}",
            evaluable=False,
            all_efficacy_effects_positive=None,
            strata=(),
        )
        member_mixed = False
        stratum_mixed = False
        if effective_n == len(plan.efficacy_units):
            robust_values = tuple(
                value.robust_effect.value for value in efficacy if value.robust_effect is not None
            )
            mean, sd, lower = controller_prospective_student_summary(
                robust_values,
                one_sided_critical_value=plan.one_sided_critical_value,
            )
            mean_effect = NamedDecimal(
                value_id=f"mean-effect.{plan.evaluation_plan_id}",
                value=mean,
                unit=plan.effect_native_unit,
            )
            sample_sd = NamedDecimal(
                value_id=f"sample-sd.{plan.evaluation_plan_id}",
                value=sd,
                unit=plan.effect_native_unit,
            )
            median_effect = NamedDecimal(
                value_id=f"median-effect.{plan.evaluation_plan_id}",
                value=_median(robust_values),
                unit=plan.effect_native_unit,
            )
            minimum_effect = NamedDecimal(
                value_id=f"minimum-effect.{plan.evaluation_plan_id}",
                value=min(robust_values),
                unit=plan.effect_native_unit,
            )
            maximum_effect = NamedDecimal(
                value_id=f"maximum-effect.{plan.evaluation_plan_id}",
                value=max(robust_values),
                unit=plan.effect_native_unit,
            )
            lower_bound = NamedDecimal(
                value_id=f"lower-bound.{plan.evaluation_plan_id}",
                value=lower,
                unit=plan.effect_native_unit,
            )
            member_distributions = tuple(
                _distribution_summary(
                    summary_id=f"member-distribution.{member_id}",
                    group_id=member_id,
                    values=tuple(
                        next(
                            member.effect.value
                            for member in unit.member_effects
                            if member.model_member_id == member_id
                        )
                        for unit in efficacy
                    ),
                    unit=plan.effect_native_unit,
                )
                for member_id in plan.model_member_ids
            )
            robust_by_unit = {
                value.unit_id: value.robust_effect.value
                for value in efficacy
                if value.robust_effect is not None
            }
            stratum_distributions = tuple(
                _distribution_summary(
                    summary_id=f"stratum-distribution.{stratum.stratum_id}",
                    group_id=stratum.stratum_id,
                    values=tuple(
                        robust_by_unit[unit_id] for unit_id in stratum.independent_unit_ids
                    ),
                    unit=plan.effect_native_unit,
                )
                for stratum in plan.strata
            )
            strengthening_strata = tuple(
                MatchedActionHoldStratumStrengthening(
                    stratum_id=stratum.stratum_id,
                    all_unit_effects_positive=all(
                        robust_by_unit[unit_id] > 0 for unit_id in stratum.independent_unit_ids
                    ),
                )
                for stratum in plan.strata
            )
            strengthening = MatchedActionHoldAllPositiveStrengtheningAnalysis(
                analysis_id=f"all-positive.{plan.evaluation_plan_id}",
                evaluable=True,
                all_efficacy_effects_positive=all(value > 0 for value in robust_values),
                strata=strengthening_strata,
            )
            member_mixed = len({value.mean_effect.value > 0 for value in member_distributions}) > 1
            stratum_mixed = (
                len({value.mean_effect.value > 0 for value in stratum_distributions}) > 1
            )

        efficacy_dispositions = {value.disposition for value in efficacy}
        reasons: set[str] = set()
        if efficacy_dispositions & {
            MatchedActionHoldUnitDisposition.REFERENCE_INCORRECT,
            MatchedActionHoldUnitDisposition.UNSAFE,
        }:
            result = MatchedActionHoldProspectiveResult.UNSAFE_OR_FALSE_SAFE_DISPOSITION
            reasons.add("EFFICACY_UNSAFE_OR_FALSE_SAFE_DISPOSITION")
        elif MatchedActionHoldUnitDisposition.DELIVERY_INVALID in efficacy_dispositions:
            result = MatchedActionHoldProspectiveResult.DELIVERY_INVALID
            reasons.add("EFFICACY_DELIVERY_INVALID")
        elif MatchedActionHoldUnitDisposition.TECHNICAL_PARTIAL in efficacy_dispositions:
            result = MatchedActionHoldProspectiveResult.TECHNICAL_PARTIAL
            reasons.add("EFFICACY_TECHNICAL_PARTIAL")
        elif (
            effective_n < plan.minimum_evaluable_efficacy_units
            or active_coverage < plan.minimum_active_coverage
        ):
            result = MatchedActionHoldProspectiveResult.EFFICACY_UNEVALUABLE
            reasons.add("COMPLETE_EFFICACY_ITT_NOT_EVALUABLE")
        elif member_mixed or stratum_mixed:
            result = MatchedActionHoldProspectiveResult.MIXED_MEMBER_OR_STRATUM_EFFICACY
            if member_mixed:
                reasons.add("MEMBER_EFFICACY_SIGN_MIXED")
            if stratum_mixed:
                reasons.add("STRATUM_EFFICACY_SIGN_MIXED")
        elif lower_bound is None or lower_bound.value <= plan.materiality.value:
            result = MatchedActionHoldProspectiveResult.VALID_NEGATIVE_OR_SUBMATERIAL_EFFICACY
            reasons.add("LOWER_BOUND_DOES_NOT_STRICTLY_EXCEED_MATERIALITY")
        elif (
            any(
                value.disposition is not MatchedActionHoldUnitDisposition.HOLD_CONTROL_CORRECT
                for value in controls
            )
            or reference_coverage < plan.minimum_reference_correctness_coverage
        ):
            result = MatchedActionHoldProspectiveResult.CONTROLLER_USE_HOLD_CONTROL_FAILED
            reasons.add("OUTSIDE_SUPPORT_HOLD_CONTROL_FAILED")
        else:
            result = MatchedActionHoldProspectiveResult.TERMINAL_PRIMARY_CONTROLLER_USE_REPLICATION_POSITIVE

        return MatchedActionHoldCohortAdjudication(
            adjudication_id=f"matched-adjudication.{plan.evaluation_plan_id}",
            evaluation_plan=ObjectIdentity.from_record(plan.evaluation_plan_id, plan),
            binding_receipt=ObjectIdentity.from_record(
                revealed.sealed_bundle.binding_receipt.binding_id,
                revealed.sealed_bundle.binding_receipt,
            ),
            compiled_study=revealed.sealed_bundle.compiled_study,
            revealed_bundle=ObjectIdentity.from_record(revealed.reveal_id, revealed),
            efficacy_units=efficacy,
            hold_controls=controls,
            rostered_efficacy_unit_count=len(plan.efficacy_units),
            effective_independent_unit_count=effective_n,
            mean_effect=mean_effect,
            sample_sd=sample_sd,
            median_effect=median_effect,
            minimum_effect=minimum_effect,
            maximum_effect=maximum_effect,
            one_sided_lower_bound=lower_bound,
            active_coverage=active_coverage,
            reference_correctness_coverage=reference_coverage,
            member_distributions=member_distributions,
            stratum_distributions=stratum_distributions,
            strengthening_analysis=strengthening,
            result=result,
            claim_ceiling=plan.maximum_claim_ceiling,
            evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
            reason_codes=tuple(sorted(reasons)),
        )


def decode_matched_action_hold_prospective_bundle(
    payload: bytes,
) -> MatchedActionHoldProspectiveBundle:
    return decode_canonical_bytes(
        payload,
        MatchedActionHoldProspectiveBundle,
        maximum_bytes=MAX_MATCHED_ACTION_HOLD_BUNDLE_BYTES,
    )


def decode_matched_action_hold_revealed_bundle(
    payload: bytes,
) -> MatchedActionHoldRevealedBundle:
    return decode_canonical_bytes(
        payload,
        MatchedActionHoldRevealedBundle,
        maximum_bytes=MAX_MATCHED_ACTION_HOLD_BUNDLE_BYTES,
    )


def decode_matched_action_hold_cohort_adjudication(
    payload: bytes,
) -> MatchedActionHoldCohortAdjudication:
    return decode_canonical_bytes(
        payload,
        MatchedActionHoldCohortAdjudication,
        maximum_bytes=MAX_MATCHED_ACTION_HOLD_BUNDLE_BYTES,
    )


__all__ = [
    "MAX_MATCHED_ACTION_HOLD_BUNDLE_BYTES",
    'MatchedActionHoldAllPositiveStrengtheningAnalysis',
    'MatchedActionHoldCohortAdjudication',
    'MatchedActionHoldControlEvaluation',
    'MatchedActionHoldControlMemberEvaluation',
    'MatchedActionHoldProspectiveEvaluator',
    'MatchedActionHoldDistributionSummary',
    'MatchedActionHoldEfficacyUnitEvaluation',
    'MatchedActionHoldEvaluationBindingCoordinator',
    'MatchedActionHoldEvaluationBindingReceipt',
    'MatchedActionHoldMemberEffect',
    'MatchedActionHoldOutcomeProjectionDisposition',
    'MatchedActionHoldOutcomeProjectionEvidenceReceipt',
    'MatchedActionHoldProspectiveResult',
    'MatchedActionHoldProjectionConstraintEvidence',
    'MatchedActionHoldProjectionOperandEvidence',
    'MatchedActionHoldProspectiveBundle',
    'MatchedActionHoldQualifiedHoldCommitment',
    'MatchedActionHoldReceiverSample',
    'MatchedActionHoldRevealIntegrityReceipt',
    'MatchedActionHoldRevealIntegrity',
    'MatchedActionHoldRevealedBundle',
    'MatchedActionHoldRevealedEpisode',
    'MatchedActionHoldRevealedReference',
    'MatchedActionHoldSealedEpisode',
    'MatchedActionHoldStratumStrengthening',
    'MatchedActionHoldUnitDisposition',
    'decode_matched_action_hold_cohort_adjudication',
    'decode_matched_action_hold_prospective_bundle',
    'decode_matched_action_hold_revealed_bundle',
]

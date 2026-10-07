"""Injected campaign bridge over the sole closed controller composition."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from empirical_lawhood.adapters.control.composition import ControllerStudyComposition
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_stable_id
from empirical_lawhood.kernel.time import ClockCoordinate
from empirical_lawhood.planning.controller_study import AdmissionControllerStudy, DeliveryControllerStudy
from empirical_lawhood.planning.nested_controller_evaluation import RepeatedDeliveryControllerEvaluationPlan
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, StudyOperationAuthority, require_study_authority
from empirical_lawhood.runtime.controller_compiler import CompiledAdmissionControllerStudy, CompiledDeliveryControllerStudy, ProspectiveBindingDisposition
from empirical_lawhood.runtime.controller_evaluation_nested import MAX_NESTED_CONTROLLER_EVALUATION_BUNDLE_BYTES, NestedControllerUseEvaluator, PreparedInstanceBindingReceipt, PreparedPolicyUnitEvaluation, SealedPreparedPolicyBundle, RevealedPreparedPolicyBundle, ActionAwareNestedControllerUseEvaluator, RepeatedDeliveryControllerCohortAdjudication, ActionAwareControllerCohortAdjudication, RepeatedDeliveryControllerUnitEvaluation, ActionAwareControllerUnitEvaluation, ProspectiveEvaluationBindingCoordinator, ProspectiveEvaluationBindingReceipt, ProspectiveExecutionEventKind, SealedRepeatedDeliveryControllerBundle, SealedActionAwareControllerBundle, RevealedRepeatedDeliveryControllerBundle, RevealedActionAwareControllerBundle, decode_sealed_repeated_delivery_controller_bundle, decode_sealed_action_aware_controller_bundle, decode_revealed_repeated_delivery_controller_bundle, decode_revealed_action_aware_controller_bundle
from empirical_lawhood.runtime.controller_runtime import AdmissionControllerTickReceipt, DeliveryControllerTickReceipt, CommitmentDisposition, RuntimeObservation
from empirical_lawhood.planning.nested_controller_evaluation import ProspectiveEvaluationPrecommitment


@dataclass(frozen=True, slots=True)
class NestedControllerRevealReceipt(CanonicalRecord):
    """Authority and byte-exact seal/reveal binding; never an outcome verdict."""

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/nested-controller-reveal-receipt'

    receipt_id: str
    sealed_bundle: ObjectIdentity
    revealed_bundle: ObjectIdentity
    reveal_authority: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (
            self.sealed_bundle.object_schema != SealedRepeatedDeliveryControllerBundle.SCHEMA
            or self.revealed_bundle.object_schema != RevealedRepeatedDeliveryControllerBundle.SCHEMA
            or self.reveal_authority.object_schema != StudyOperationAuthority.SCHEMA
        ):
            raise ValueError("controller campaign reveal receipt binds incompatible records")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("controller campaign reveal receipt is not evaluator-only")


@dataclass(frozen=True, slots=True)
class NestedAdmissionControllerBridge:
    """Thin reusable route; construction belongs only to the API composition root."""

    composition: ControllerStudyComposition

    def compile(self, study: AdmissionControllerStudy) -> CompiledAdmissionControllerStudy:
        return self.composition.compile(study)

    def tick(
        self,
        compiled: CompiledAdmissionControllerStudy,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> AdmissionControllerTickReceipt:
        return self.composition.tick(
            compiled,
            observation,
            commitment_coordinate=commitment_coordinate,
        )

    def reveal_nested(
        self,
        *,
        sealed_payload: bytes,
        revealed_payload: bytes,
        reveal_authority: StudyOperationAuthority,
        issued_study: ObjectIdentity,
        execution_authority: ObjectIdentity,
        grantee_id: str,
        at_utc: str,
    ) -> tuple[
        RevealedRepeatedDeliveryControllerBundle,
        NestedControllerRevealReceipt,
    ]:
        """Validate external seal/reveal bytes after exact loaded reveal authority."""

        require_study_authority(
            reveal_authority,
            kind=StudyAuthorityKind.OUTCOME_REVEAL,
            subject=issued_study,
            prerequisite_authority=execution_authority,
            grantee_id=grantee_id,
            at_utc=at_utc,
        )
        sealed = decode_sealed_repeated_delivery_controller_bundle(sealed_payload)
        revealed = decode_revealed_repeated_delivery_controller_bundle(revealed_payload)
        if revealed.sealed_bundle != sealed:
            raise ValueError("revealed controller bundle substitutes another seal")
        authority_identity = ObjectIdentity.from_record(
            reveal_authority.authority_id,
            reveal_authority,
        )
        if revealed.reveal_authorization != authority_identity:
            raise ValueError("revealed controller bundle binds another reveal authority")
        receipt = NestedControllerRevealReceipt(
            receipt_id=f"controller-campaign-reveal.{sealed.bundle_id}",
            sealed_bundle=ObjectIdentity.from_record(sealed.bundle_id, sealed),
            revealed_bundle=ObjectIdentity.from_record(revealed.reveal_id, revealed),
            reveal_authority=authority_identity,
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        )
        return revealed, receipt

    def evaluate_nested_unit(
        self,
        compiled: CompiledAdmissionControllerStudy,
        revealed: RevealedRepeatedDeliveryControllerBundle,
        *,
        independent_unit_id: str,
    ) -> RepeatedDeliveryControllerUnitEvaluation:
        return self.composition.evaluate_nested_unit(
            compiled,
            revealed,
            independent_unit_id=independent_unit_id,
        )

    def adjudicate_nested_cohort(
        self,
        compiled: CompiledAdmissionControllerStudy,
        plan: RepeatedDeliveryControllerEvaluationPlan,
        revealed: RevealedRepeatedDeliveryControllerBundle,
        units: tuple[RepeatedDeliveryControllerUnitEvaluation, ...],
    ) -> RepeatedDeliveryControllerCohortAdjudication:
        return self.composition.adjudicate_nested_cohort(
            compiled,
            plan,
            revealed,
            units,
        )


@dataclass(frozen=True, slots=True)
class BoundActionAwareControllerRevealReceipt(CanonicalRecord):
    "Exact seal/reveal binding for the action-aware delivery."

    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/bound-action-aware-controller-reveal-receipt'

    receipt_id: str
    sealed_bundle: ObjectIdentity
    revealed_bundle: ObjectIdentity
    binding_receipt: ObjectIdentity
    reveal_authority: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (
            self.sealed_bundle.object_schema != SealedActionAwareControllerBundle.SCHEMA
            or self.revealed_bundle.object_schema != RevealedActionAwareControllerBundle.SCHEMA
            or self.binding_receipt.object_schema != ProspectiveEvaluationBindingReceipt.SCHEMA
            or self.reveal_authority.object_schema != StudyOperationAuthority.SCHEMA
        ):
            raise ValueError("action-aware controller reveal binds incompatible records")
        if self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL:
            raise ValueError("action-aware controller reveal is not evaluator-only")


@dataclass(frozen=True, slots=True)
class BoundActionAwareControllerBridge:
    "Admission-only compile/tick plus a separately bound action-aware evaluator."

    composition: ControllerStudyComposition
    evaluator: ActionAwareNestedControllerUseEvaluator
    binding_coordinator: ProspectiveEvaluationBindingCoordinator

    def __post_init__(self) -> None:
        if self.composition.outcome_evaluator is not None:
            raise ValueError("action-aware controller bridge requires a controller composition capped at admission")

    def compile(self, study: AdmissionControllerStudy) -> CompiledAdmissionControllerStudy:
        if study.prospective_evaluation is not None:
            raise ValueError("action-aware bridge requires prospective_evaluation=None")
        compiled = self.composition.compile(study)
        if compiled.prospective_evaluation_binding.disposition is not ProspectiveBindingDisposition.NOT_DECLARED:
            raise ValueError("action-aware controller bridge compiled product must retain controller use NOT_DECLARED")
        return compiled

    def bind_evaluator(
        self,
        *,
        prefix_id: str,
        precommitment: ProspectiveEvaluationPrecommitment,
        study: AdmissionControllerStudy,
        compiled: CompiledAdmissionControllerStudy,
        bound_at_utc: str,
    ) -> ProspectiveEvaluationBindingReceipt:
        if precommitment.evaluator_binding != self.evaluator.implementation_binding:
            raise ValueError("separately composed evaluator differs from precommitment")
        return self.binding_coordinator.bind(
            prefix_id=prefix_id,
            precommitment=precommitment,
            study=study,
            compiled=compiled,
            bound_at_utc=bound_at_utc,
        )

    def tick(
        self,
        compiled: CompiledAdmissionControllerStudy,
        observation: RuntimeObservation,
        *,
        binding_receipt: ProspectiveEvaluationBindingReceipt,
        prefix_id: str,
        event_id: str,
        occurred_at_utc: str,
        commitment_coordinate: ClockCoordinate,
    ) -> AdmissionControllerTickReceipt:
        if binding_receipt.compiled_study != ObjectIdentity.from_record(
            compiled.compiled_study_id,
            compiled,
        ):
            raise ValueError("tick uses another prospective binding receipt")
        self.binding_coordinator.reserve_protected_event(
            prefix_id=prefix_id,
            receipt=binding_receipt,
            event_id=event_id,
            kind=ProspectiveExecutionEventKind.TICK_RESERVED,
            subject=ObjectIdentity.from_record(observation.observation_id, observation),
            occurred_at_utc=occurred_at_utc,
        )
        return self.composition.tick(
            compiled,
            observation,
            commitment_coordinate=commitment_coordinate,
        )

    def reserve_reference(
        self,
        *,
        prefix_id: str,
        binding_receipt: ProspectiveEvaluationBindingReceipt,
        event_id: str,
        reference_subject: ObjectIdentity,
        occurred_at_utc: str,
    ) -> None:
        self.binding_coordinator.reserve_protected_event(
            prefix_id=prefix_id,
            receipt=binding_receipt,
            event_id=event_id,
            kind=ProspectiveExecutionEventKind.REFERENCE_RESERVED,
            subject=reference_subject,
            occurred_at_utc=occurred_at_utc,
        )

    def reveal_action_aware_nested(
        self,
        *,
        sealed_payload: bytes,
        revealed_payload: bytes,
        reveal_authority: StudyOperationAuthority,
        issued_study: ObjectIdentity,
        execution_authority: ObjectIdentity,
        grantee_id: str,
        at_utc: str,
    ) -> tuple[
        RevealedActionAwareControllerBundle,
        BoundActionAwareControllerRevealReceipt,
    ]:
        require_study_authority(
            reveal_authority,
            kind=StudyAuthorityKind.OUTCOME_REVEAL,
            subject=issued_study,
            prerequisite_authority=execution_authority,
            grantee_id=grantee_id,
            at_utc=at_utc,
        )
        sealed = decode_sealed_action_aware_controller_bundle(sealed_payload)
        revealed = decode_revealed_action_aware_controller_bundle(revealed_payload)
        if revealed.sealed_bundle != sealed:
            raise ValueError("revealed action-aware controller bundle substitutes another seal")
        authority_identity = ObjectIdentity.from_record(
            reveal_authority.authority_id,
            reveal_authority,
        )
        if revealed.reveal_authorization != authority_identity:
            raise ValueError("revealed action-aware controller bundle binds another reveal authority")
        binding_identity = ObjectIdentity.from_record(
            sealed.binding_receipt.binding_id,
            sealed.binding_receipt,
        )
        receipt = BoundActionAwareControllerRevealReceipt(
            receipt_id=f"action-aware-controller-reveal.{sealed.bundle_id}",
            sealed_bundle=ObjectIdentity.from_record(sealed.bundle_id, sealed),
            revealed_bundle=ObjectIdentity.from_record(revealed.reveal_id, revealed),
            binding_receipt=binding_identity,
            reveal_authority=authority_identity,
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        )
        return revealed, receipt

    def evaluate_nested_unit(
        self,
        compiled: CompiledAdmissionControllerStudy,
        revealed: RevealedActionAwareControllerBundle,
        *,
        independent_unit_id: str,
    ) -> ActionAwareControllerUnitEvaluation:
        return self.evaluator.evaluate_unit(
            compiled=compiled,
            revealed=revealed,
            independent_unit_id=independent_unit_id,
        )

    def adjudicate_nested_cohort(
        self,
        compiled: CompiledAdmissionControllerStudy,
        revealed: RevealedActionAwareControllerBundle,
        units: tuple[ActionAwareControllerUnitEvaluation, ...],
    ) -> ActionAwareControllerCohortAdjudication:
        return self.evaluator.adjudicate_cohort(
            compiled=compiled,
            revealed=revealed,
            units=units,
        )


__all__ = [
    'NestedAdmissionControllerBridge',
    'BoundActionAwareControllerBridge',
    'NestedControllerRevealReceipt',
    'BoundActionAwareControllerRevealReceipt',
]


@dataclass(frozen=True, slots=True)
class PreparedPolicyControllerRevealReceipt(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/control/prepared-policy-controller-reveal-receipt'

    receipt_id: str
    sealed_bundle: ObjectIdentity
    revealed_bundle: ObjectIdentity
    reveal_authority: ObjectIdentity
    outcome_access: OutcomeAccess

    def __post_init__(self) -> None:
        validate_stable_id(self.receipt_id, field_name="receipt_id")
        if (
            self.sealed_bundle.object_schema != SealedPreparedPolicyBundle.SCHEMA
            or self.revealed_bundle.object_schema != RevealedPreparedPolicyBundle.SCHEMA
            or self.reveal_authority.object_schema != StudyOperationAuthority.SCHEMA
            or self.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
        ):
            raise ValueError(
                "prepared campaign reveal requires its exact sealed/revealed/authority versions"
            )


@dataclass(frozen=True, slots=True)
class PreparedPolicyControllerBridge:
    """Finite controller route with durable commitment before native task launch."""

    composition: ControllerStudyComposition
    binding_coordinator: ProspectiveEvaluationBindingCoordinator

    def __post_init__(self) -> None:
        if not isinstance(self.composition.outcome_evaluator, NestedControllerUseEvaluator):
            raise ValueError("prepared campaign requires the sole nested controller use evaluator")

    def compile(self, study: DeliveryControllerStudy) -> CompiledDeliveryControllerStudy:
        if study.prospective_evaluation is None:
            raise ValueError("prepared campaign requires its frozen prospective design")
        return self.composition.compile(study)

    def tick(
        self,
        compiled: CompiledDeliveryControllerStudy,
        observation: RuntimeObservation,
        *,
        binding_receipt: PreparedInstanceBindingReceipt,
        prefix_id: str,
        occurred_at_utc: str,
        commitment_coordinate: ClockCoordinate,
    ) -> DeliveryControllerTickReceipt:
        if binding_receipt.compiled_study != ObjectIdentity.from_record(
            compiled.compiled_study_id, compiled
        ):
            raise ValueError("prepared tick substitutes its concrete instance binding")
        commitment = self.composition.prepare_commitment(
            compiled, observation, commitment_coordinate=commitment_coordinate
        )
        self.binding_coordinator.store_prepared_commitment(
            prefix_id=prefix_id,
            instance=binding_receipt,
            commitment=commitment,
            occurred_at_utc=occurred_at_utc,
        )
        if commitment.disposition is not CommitmentDisposition.NONATTEMPT:
            self.binding_coordinator.reserve_prepared_task(
                prefix_id=prefix_id, commitment=commitment, occurred_at_utc=occurred_at_utc
            )
        tick = self.composition.deliver_prepared_commitment(compiled, commitment)
        self.binding_coordinator.store_prepared_delivery(prefix_id=prefix_id, tick=tick)
        return tick

    def reveal_prepared(
        self,
        *,
        sealed_payload: bytes,
        revealed_payload: bytes,
        reveal_authority: StudyOperationAuthority,
        issued_study: ObjectIdentity,
        execution_authority: ObjectIdentity,
        grantee_id: str,
        at_utc: str,
    ) -> tuple[RevealedPreparedPolicyBundle, PreparedPolicyControllerRevealReceipt]:
        require_study_authority(
            reveal_authority,
            kind=StudyAuthorityKind.OUTCOME_REVEAL,
            subject=issued_study,
            prerequisite_authority=execution_authority,
            grantee_id=grantee_id,
            at_utc=at_utc,
        )
        sealed = decode_canonical_bytes(
            sealed_payload, SealedPreparedPolicyBundle, maximum_bytes=MAX_NESTED_CONTROLLER_EVALUATION_BUNDLE_BYTES
        )
        revealed = decode_canonical_bytes(
            revealed_payload,
            RevealedPreparedPolicyBundle,
            maximum_bytes=MAX_NESTED_CONTROLLER_EVALUATION_BUNDLE_BYTES,
        )
        authority = ObjectIdentity.from_record(reveal_authority.authority_id, reveal_authority)
        if (
            revealed.sealed != sealed
            or sealed.design.issued_manifest != issued_study
            or revealed.reveal_authorization != authority
        ):
            raise ValueError(
                "prepared reveal substitutes sealed bytes, issue or exact reveal authority"
            )
        return revealed, PreparedPolicyControllerRevealReceipt(
            f"prepared-reveal-receipt.{revealed.reveal_id}",
            ObjectIdentity.from_record(sealed.bundle_id, sealed),
            ObjectIdentity.from_record(revealed.reveal_id, revealed),
            authority,
            OutcomeAccess.EVALUATOR_REVEAL,
        )

    def evaluate_unit(
        self,
        *,
        compiled: CompiledDeliveryControllerStudy | None,
        revealed: RevealedPreparedPolicyBundle,
    ) -> PreparedPolicyUnitEvaluation:
        return self.composition.evaluate_prepared_unit(compiled=compiled, revealed=revealed)

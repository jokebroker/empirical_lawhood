"""Closed capability composition for the sole ControllerProgramme route."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import overload

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.time import ClockCoordinate
from empirical_lawhood.planning.controller_study import AtlasControllerStudy, AdmissionControllerStudy, DeliveryControllerStudy, LeastMagnitudeControllerStudy, ImplementationBinding, ImplementationRole, ProspectiveControllerEvaluationPlan
from empirical_lawhood.planning.evidence_geometry import AtlasAdmissionSpec, ReceiptAdmissionSpec, AtlasReachabilitySpec, ControlledMapReachabilitySpec, ControlledMapReachabilityComparison
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionSpec, FiniteCertificateReachabilitySpec, FiniteCertificateReachabilityComparison
from empirical_lawhood.planning.geometry import AdmissionComparison, ReachabilityComparison
from empirical_lawhood.planning.nested_controller_evaluation import RepeatedDeliveryControllerEvaluationPlan
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
    CapabilityRequirement,
)
from empirical_lawhood.runtime.controller_compiler import AtlasAdmissionDeriverPort, ReceiptAdmissionDeriverPort, FiniteCertificateAdmissionDeriverPort, CompiledAtlasControllerStudy, CompiledAdmissionControllerStudy, CompiledDeliveryControllerStudy, CompiledLeastMagnitudeControllerStudy, AdmissionCandidateAudit, ControllerCompiler, AtlasReachabilityDeriverPort, ReceiptReachabilityDeriverPort, FiniteCertificateReachabilityDeriverPort
from empirical_lawhood.runtime.controller_evaluation import (
    ControllerCohortAdjudication,
    ControllerUseEvaluator,
    ControllerUnitEvaluation,
    RevealedProspectiveControllerBundle,
)
from empirical_lawhood.runtime.controller_evaluation_nested import RepeatedDeliveryControllerCohortAdjudication, RepeatedDeliveryControllerUnitEvaluation, NestedControllerUseEvaluator, RevealedRepeatedDeliveryControllerBundle, RevealedPreparedPolicyBundle, RevealedPreparedForecastPolicyBundle, PreparedPolicyUnitEvaluation, PreparedControllerCohortEvaluation
from empirical_lawhood.runtime.controller_runtime import ControllerRuntime, AtlasControllerTickReceipt, AdmissionControllerTickReceipt, StageAwareAdmissionControllerTickReceipt, DeliveryControllerTickReceipt, DeliveryControllerDecisionCommitment, NativeDeliveryPort, ObserverPort, AtlasLiveGateEvaluatorPort, AdmissionLiveGateEvaluatorPort, NumericalViewLiveGateEvaluatorPort, AdmissionLiveGateEvaluation, NumericalViewLiveGateEvaluation, ExactActionDeliveryTrace, RuntimeObservation


CONTROLLER_CAPABILITY_CONFIG_SCHEMA = 'empirical-lawhood/control/controller-capability'
CONTROLLER_CAPABILITY_CONFIG_SCHEMA_SHA256 = sha256(
    CONTROLLER_CAPABILITY_CONFIG_SCHEMA.encode()
).hexdigest()


_ROLE_KIND = {
    ImplementationRole.ADMISSION_DERIVER: CapabilityKind.ADMISSION_EVALUATOR,
    ImplementationRole.REACHABILITY_DERIVER: CapabilityKind.REACHABILITY_EVALUATOR,
    ImplementationRole.SYNTHESIZER: CapabilityKind.CONTROLLER_SYNTHESIZER,
    ImplementationRole.OBSERVER: CapabilityKind.CONTROLLER_OBSERVER,
    ImplementationRole.ONLINE_GATE_EVALUATOR: CapabilityKind.ONLINE_GATE_EVALUATOR,
    ImplementationRole.DELIVERY: CapabilityKind.NATIVE_DELIVERY,
    ImplementationRole.OUTCOME_EVALUATOR: CapabilityKind.EVALUATOR,
}

_ROLE_SCHEMAS = {
    ImplementationRole.ADMISSION_DERIVER: (
        (AtlasAdmissionSpec.SCHEMA,),
        (AdmissionComparison.SCHEMA,),
    ),
    ImplementationRole.REACHABILITY_DERIVER: (
        (AtlasReachabilitySpec.SCHEMA,),
        (ReachabilityComparison.SCHEMA,),
    ),
    ImplementationRole.SYNTHESIZER: (
        (AtlasControllerStudy.SCHEMA,),
        (CompiledAtlasControllerStudy.SCHEMA,),
    ),
    ImplementationRole.OBSERVER: (
        (RuntimeObservation.SCHEMA,),
        ('empirical-lawhood/runtime/observer-evaluation',),
    ),
    ImplementationRole.ONLINE_GATE_EVALUATOR: (
        (AtlasControllerStudy.SCHEMA, RuntimeObservation.SCHEMA),
        ('empirical-lawhood/runtime/atlas-live-gate-evaluation',),
    ),
    ImplementationRole.DELIVERY: (
        (OccurrenceActionWord.SCHEMA,),
        (ExactActionDeliveryTrace.SCHEMA,),
    ),
    ImplementationRole.OUTCOME_EVALUATOR: (
        (RevealedProspectiveControllerBundle.SCHEMA,),
        (ControllerCohortAdjudication.SCHEMA, ControllerUnitEvaluation.SCHEMA),
    ),
}

_ADMISSION_ROLE_SCHEMAS = {
    ImplementationRole.ADMISSION_DERIVER: (
        (ReceiptAdmissionSpec.SCHEMA,),
        (AdmissionComparison.SCHEMA,),
    ),
    ImplementationRole.REACHABILITY_DERIVER: (
        (ControlledMapReachabilitySpec.SCHEMA,),
        (ControlledMapReachabilityComparison.SCHEMA,),
    ),
    ImplementationRole.SYNTHESIZER: (
        (AdmissionControllerStudy.SCHEMA,),
        (CompiledAdmissionControllerStudy.SCHEMA,),
    ),
    ImplementationRole.OBSERVER: (
        (RuntimeObservation.SCHEMA,),
        ('empirical-lawhood/runtime/observer-evaluation',),
    ),
    ImplementationRole.ONLINE_GATE_EVALUATOR: (
        (AdmissionControllerStudy.SCHEMA, RuntimeObservation.SCHEMA),
        (AdmissionLiveGateEvaluation.SCHEMA,),
    ),
    ImplementationRole.DELIVERY: (
        (OccurrenceActionWord.SCHEMA,),
        (ExactActionDeliveryTrace.SCHEMA,),
    ),
    ImplementationRole.OUTCOME_EVALUATOR: (
        (RevealedRepeatedDeliveryControllerBundle.SCHEMA,),
        (RepeatedDeliveryControllerCohortAdjudication.SCHEMA, RepeatedDeliveryControllerUnitEvaluation.SCHEMA),
    ),
}


_FINITE_DELIVERY_ROLE_SCHEMAS = {
    ImplementationRole.ADMISSION_DERIVER: (
        (FiniteCertificateAdmissionSpec.SCHEMA,),
        (AdmissionComparison.SCHEMA,),
    ),
    ImplementationRole.REACHABILITY_DERIVER: (
        (FiniteCertificateReachabilitySpec.SCHEMA,),
        (FiniteCertificateReachabilityComparison.SCHEMA,),
    ),
    ImplementationRole.SYNTHESIZER: (
        (DeliveryControllerStudy.SCHEMA,),
        (CompiledDeliveryControllerStudy.SCHEMA,),
    ),
    ImplementationRole.OBSERVER: (
        (RuntimeObservation.SCHEMA,),
        ('empirical-lawhood/runtime/observer-evaluation',),
    ),
    ImplementationRole.ONLINE_GATE_EVALUATOR: (
        (DeliveryControllerStudy.SCHEMA, RuntimeObservation.SCHEMA),
        (NumericalViewLiveGateEvaluation.SCHEMA,),
    ),
    ImplementationRole.DELIVERY: ((OccurrenceActionWord.SCHEMA,), (ExactActionDeliveryTrace.SCHEMA,)),
    ImplementationRole.OUTCOME_EVALUATOR: (
        (RevealedPreparedPolicyBundle.SCHEMA,),
        (PreparedPolicyUnitEvaluation.SCHEMA, PreparedControllerCohortEvaluation.SCHEMA),
    ),
}


def _role_schemas(
    role: ImplementationRole,
    controller_study_schema: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if controller_study_schema == LeastMagnitudeControllerStudy.SCHEMA:
        if role is ImplementationRole.SYNTHESIZER:
            return ((LeastMagnitudeControllerStudy.SCHEMA,), (CompiledLeastMagnitudeControllerStudy.SCHEMA,))
        if role is ImplementationRole.ONLINE_GATE_EVALUATOR:
            return (
                (LeastMagnitudeControllerStudy.SCHEMA, RuntimeObservation.SCHEMA),
                (NumericalViewLiveGateEvaluation.SCHEMA,),
            )
        if role is ImplementationRole.OUTCOME_EVALUATOR:
            return (
                (
                    RevealedPreparedPolicyBundle.SCHEMA,
                    RevealedPreparedForecastPolicyBundle.SCHEMA,
                ),
                _FINITE_DELIVERY_ROLE_SCHEMAS[role][1],
            )
        return _FINITE_DELIVERY_ROLE_SCHEMAS[role]
    if controller_study_schema == AtlasControllerStudy.SCHEMA:
        return _ROLE_SCHEMAS[role]
    if controller_study_schema == AdmissionControllerStudy.SCHEMA:
        return _ADMISSION_ROLE_SCHEMAS[role]
    if controller_study_schema == DeliveryControllerStudy.SCHEMA:
        return _FINITE_DELIVERY_ROLE_SCHEMAS[role]
    raise ValueError("controller capability contract schema is not registered")


def _budget() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=128 * 1024 * 1024,
        gpu_devices=0,
        wall_time_seconds=10,
        source_scan_bytes=0,
        output_bytes=16 * 1024 * 1024,
    )


def controller_capability_manifest(
    binding: ImplementationBinding,
    *,
    controller_study_schema: str = AtlasControllerStudy.SCHEMA,
) -> CapabilityManifest:
    inputs, outputs = _role_schemas(binding.role, controller_study_schema)
    evaluator = binding.role is ImplementationRole.OUTCOME_EVALUATOR
    permissions = (
        (
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        )
        if evaluator
        else ()
    )
    return CapabilityManifest(
        capability_key=binding.reference.capability_key,
        capability_version=binding.reference.capability_version,
        kind=_ROLE_KIND[binding.role],
        config_schema=CONTROLLER_CAPABILITY_CONFIG_SCHEMA,
        config_schema_sha256=CONTROLLER_CAPABILITY_CONFIG_SCHEMA_SHA256,
        input_schema_ids=tuple(sorted(inputs)),
        output_schema_ids=tuple(sorted(outputs)),
        permissions=tuple(sorted(permissions)),
        maximum_evidence_ceiling=(
            EvidenceCeiling.CONTROLLER_USE if evaluator else EvidenceCeiling.ADMISSION
        ),
        maximum_outcome_access=(
            OutcomeAccess.EVALUATOR_REVEAL if evaluator else OutcomeAccess.OUTCOME_BLIND
        ),
        resource_ceiling=_budget(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="empirical-lawhood-controller-runtime",
        requires_clean_commit=False,
        requires_active_mount=False,
        requires_network=False,
        conformance_check_ids=(f"conformance.{binding.role.value.lower()}",),
        implementation_sha256=binding.implementation_sha256,
    )


def controller_capability_registry(
    implementations: tuple[ImplementationBinding, ...],
    *,
    controller_study_schema: str = AtlasControllerStudy.SCHEMA,
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="registry.controller-programme-sole-route",
        capabilities=tuple(
            sorted(
                (
                    controller_capability_manifest(
                        binding,
                        controller_study_schema=controller_study_schema,
                    )
                    for binding in implementations
                ),
                key=lambda value: value.registry_id,
            )
        ),
    )


def _requirement(
    binding: ImplementationBinding,
    *,
    controller_study_schema: str,
) -> CapabilityRequirement:
    inputs, outputs = _role_schemas(binding.role, controller_study_schema)
    evaluator = binding.role is ImplementationRole.OUTCOME_EVALUATOR
    return CapabilityRequirement(
        capability_key=binding.reference.capability_key,
        capability_version=binding.reference.capability_version,
        kind=_ROLE_KIND[binding.role],
        config=CapabilityConfigRef(
            config_id=f"config.{binding.binding_id}",
            config_schema=CONTROLLER_CAPABILITY_CONFIG_SCHEMA,
            config_schema_sha256=CONTROLLER_CAPABILITY_CONFIG_SCHEMA_SHA256,
            content_sha256=binding.config_sha256,
            artifact_id=f"artifact.config.{binding.binding_id}",
        ),
        required_input_schema_ids=tuple(sorted(inputs)),
        required_output_schema_ids=tuple(sorted(outputs)),
        required_permissions=(
            (
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            )
            if evaluator
            else ()
        ),
        requested_evidence_ceiling=(
            EvidenceCeiling.CONTROLLER_USE if evaluator else EvidenceCeiling.ADMISSION
        ),
        requested_outcome_access=(
            OutcomeAccess.EVALUATOR_REVEAL if evaluator else OutcomeAccess.OUTCOME_BLIND
        ),
        requested_resources=_budget(),
    )


@dataclass(frozen=True, slots=True)
class ControllerStudyComposition:
    """One closed composition; config cannot name imports, callables or policies."""

    registry: CapabilityRegistry
    admission_deriver: AtlasAdmissionDeriverPort | ReceiptAdmissionDeriverPort | FiniteCertificateAdmissionDeriverPort
    reachability_deriver: (
        AtlasReachabilityDeriverPort | ReceiptReachabilityDeriverPort | FiniteCertificateReachabilityDeriverPort
    )
    observer: ObserverPort
    online_gate_evaluator: (
        AtlasLiveGateEvaluatorPort | AdmissionLiveGateEvaluatorPort | NumericalViewLiveGateEvaluatorPort
    )
    native_delivery: NativeDeliveryPort
    outcome_evaluator: ControllerUseEvaluator | NestedControllerUseEvaluator | None

    def _validate_study(
        self,
        study: AtlasControllerStudy | AdmissionControllerStudy | DeliveryControllerStudy,
    ) -> None:
        services: dict[
            ImplementationRole,
            AtlasAdmissionDeriverPort
            | ReceiptAdmissionDeriverPort
            | FiniteCertificateAdmissionDeriverPort
            | AtlasReachabilityDeriverPort
            | ReceiptReachabilityDeriverPort
            | FiniteCertificateReachabilityDeriverPort
            | ObserverPort
            | AtlasLiveGateEvaluatorPort
            | AdmissionLiveGateEvaluatorPort
            | NumericalViewLiveGateEvaluatorPort
            | NativeDeliveryPort,
        ] = {
            ImplementationRole.ADMISSION_DERIVER: self.admission_deriver,
            ImplementationRole.REACHABILITY_DERIVER: self.reachability_deriver,
            ImplementationRole.OBSERVER: self.observer,
            ImplementationRole.ONLINE_GATE_EVALUATOR: self.online_gate_evaluator,
            ImplementationRole.DELIVERY: self.native_delivery,
        }
        programme_schema = study.SCHEMA
        for binding in study.implementations:
            manifest = self.registry.require(
                _requirement(
                    binding,
                    controller_study_schema=programme_schema,
                )
            )
            if manifest.implementation_sha256 != binding.implementation_sha256:
                raise ValueError("controller capability implementation fingerprint differs")
            service = services.get(binding.role)
            if service is not None and service.implementation_binding != binding:
                raise ValueError("controller service differs from its capability binding")
            if binding.role is ImplementationRole.OUTCOME_EVALUATOR:
                if (
                    self.outcome_evaluator is None
                    or self.outcome_evaluator.implementation_binding != binding
                ):
                    raise ValueError("controller use evaluator differs from its capability binding")
        if study.prospective_evaluation is None and self.outcome_evaluator is not None:
            raise ValueError("composition capped at admission cannot expose an outcome evaluator")

    def assess_finite_candidates(
        self, study: DeliveryControllerStudy
    ) -> tuple[AdmissionCandidateAudit, ...]:
        """Validate installed owners, then assess alternatives without compilation."""
        self._validate_study(study)
        return ControllerCompiler.assess_finite_candidates(study)

    @overload
    def compile(self, study: AtlasControllerStudy) -> CompiledAtlasControllerStudy: ...

    @overload
    def compile(self, study: AdmissionControllerStudy) -> CompiledAdmissionControllerStudy: ...

    @overload
    def compile(self, study: DeliveryControllerStudy) -> CompiledDeliveryControllerStudy: ...

    def compile(
        self,
        study: AtlasControllerStudy | AdmissionControllerStudy | DeliveryControllerStudy,
    ) -> (
        CompiledAtlasControllerStudy | CompiledAdmissionControllerStudy | CompiledDeliveryControllerStudy
    ):
        self._validate_study(study)
        return ControllerCompiler(
            admission_deriver=self.admission_deriver,
            reachability_deriver=self.reachability_deriver,
        ).compile(study)

    @overload
    def tick(
        self,
        compiled: CompiledAtlasControllerStudy,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> AtlasControllerTickReceipt: ...

    @overload
    def tick(
        self,
        compiled: CompiledAdmissionControllerStudy,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> AdmissionControllerTickReceipt: ...

    @overload
    def tick(
        self,
        compiled: CompiledDeliveryControllerStudy,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> DeliveryControllerTickReceipt: ...

    def tick(
        self,
        compiled: CompiledAtlasControllerStudy
        | CompiledAdmissionControllerStudy
        | CompiledDeliveryControllerStudy,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> AtlasControllerTickReceipt | AdmissionControllerTickReceipt | DeliveryControllerTickReceipt:
        self._validate_study(compiled.study)
        return ControllerRuntime(
            compiled=compiled,
            observer=self.observer,
            online_gates=self.online_gate_evaluator,
            delivery=self.native_delivery,
        ).tick(observation, commitment_coordinate=commitment_coordinate)

    def tick_stage_aware(
        self,
        compiled: CompiledAdmissionControllerStudy | CompiledDeliveryControllerStudy,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> AdmissionControllerTickReceipt | StageAwareAdmissionControllerTickReceipt | DeliveryControllerTickReceipt:
        """Delegate stage-aware delivery through the sole closed runtime route."""

        self._validate_study(compiled.study)
        return ControllerRuntime(
            compiled=compiled,
            observer=self.observer,
            online_gates=self.online_gate_evaluator,
            delivery=self.native_delivery,
        ).tick_stage_aware(
            observation,
            commitment_coordinate=commitment_coordinate,
        )

    def prepare_commitment(
        self,
        compiled: CompiledDeliveryControllerStudy,
        observation: RuntimeObservation,
        *,
        commitment_coordinate: ClockCoordinate,
    ) -> DeliveryControllerDecisionCommitment:
        """Prepare a concrete reviewable decision without native delivery."""

        self._validate_study(compiled.study)
        return ControllerRuntime(
            compiled=compiled,
            observer=self.observer,
            online_gates=self.online_gate_evaluator,
            delivery=self.native_delivery,
        ).prepare_commitment(observation, commitment_coordinate=commitment_coordinate)

    def deliver_prepared_commitment(
        self,
        compiled: CompiledDeliveryControllerStudy,
        commitment: DeliveryControllerDecisionCommitment,
    ) -> DeliveryControllerTickReceipt:
        self._validate_study(compiled.study)
        return ControllerRuntime(
            compiled=compiled,
            observer=self.observer,
            online_gates=self.online_gate_evaluator,
            delivery=self.native_delivery,
        ).deliver_prepared_commitment(commitment)

    def evaluate_unit(
        self,
        compiled: CompiledAtlasControllerStudy,
        tick: AtlasControllerTickReceipt,
        revealed: RevealedProspectiveControllerBundle,
    ) -> ControllerUnitEvaluation:
        self._validate_study(compiled.study)
        if self.outcome_evaluator is None:
            raise ValueError("controller use evaluation capability is not composed")
        if not isinstance(self.outcome_evaluator, ControllerUseEvaluator):
            raise ValueError("frozen controller use evaluation requires the prospective controller evaluator")
        return self.outcome_evaluator.evaluate_unit(compiled=compiled, tick=tick, revealed=revealed)

    def adjudicate_cohort(
        self,
        compiled: CompiledAtlasControllerStudy,
        plan: ProspectiveControllerEvaluationPlan,
        units: tuple[ControllerUnitEvaluation, ...],
    ) -> ControllerCohortAdjudication:
        self._validate_study(compiled.study)
        if self.outcome_evaluator is None:
            raise ValueError("controller use evaluation capability is not composed")
        if not isinstance(self.outcome_evaluator, ControllerUseEvaluator):
            raise ValueError("frozen controller use adjudication requires the prospective controller evaluator")
        return self.outcome_evaluator.adjudicate_cohort(compiled=compiled, plan=plan, units=units)

    def evaluate_prepared_unit(
        self,
        *,
        compiled: CompiledDeliveryControllerStudy | None,
        revealed: RevealedPreparedPolicyBundle,
    ) -> PreparedPolicyUnitEvaluation:
        if not isinstance(self.outcome_evaluator, NestedControllerUseEvaluator):
            raise ValueError("prepared controller use requires the composed nested evaluator")
        return self.outcome_evaluator.evaluate_prepared_unit(compiled=compiled, revealed=revealed)

    def evaluate_nested_unit(
        self,
        compiled: CompiledAdmissionControllerStudy,
        revealed: RevealedRepeatedDeliveryControllerBundle,
        *,
        independent_unit_id: str,
    ) -> RepeatedDeliveryControllerUnitEvaluation:
        self._validate_study(compiled.study)
        if not isinstance(self.outcome_evaluator, NestedControllerUseEvaluator):
            raise ValueError("nested controller use evaluation capability is not composed")
        return self.outcome_evaluator.evaluate_unit(
            compiled=compiled,
            revealed=revealed,
            independent_unit_id=independent_unit_id,
        )

    def adjudicate_nested_cohort(
        self,
        compiled: CompiledAdmissionControllerStudy,
        plan: RepeatedDeliveryControllerEvaluationPlan,
        revealed: RevealedRepeatedDeliveryControllerBundle,
        units: tuple[RepeatedDeliveryControllerUnitEvaluation, ...],
    ) -> RepeatedDeliveryControllerCohortAdjudication:
        self._validate_study(compiled.study)
        if revealed.sealed_bundle.evaluation_plan != plan:
            raise ValueError("nested controller use revealed bundle differs from frozen plan")
        if not isinstance(self.outcome_evaluator, NestedControllerUseEvaluator):
            raise ValueError("nested controller use adjudication capability is not composed")
        return self.outcome_evaluator.adjudicate_cohort(
            compiled=compiled,
            revealed=revealed,
            units=units,
        )


__all__ = [
    "CONTROLLER_CAPABILITY_CONFIG_SCHEMA",
    "CONTROLLER_CAPABILITY_CONFIG_SCHEMA_SHA256",
    'ControllerStudyComposition',
    "controller_capability_manifest",
    "controller_capability_registry",
]

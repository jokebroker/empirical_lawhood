"Closed finite response-law bindings to existing controller, runtime and prepared controller-use owners.\n\nThe outer issued executable closure pins this adapter and every implementation.\nThe small static role payloads are canonical configuration, not executable code.\n"

from hashlib import sha256
from dataclasses import dataclass
from typing import ClassVar, cast

from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, SafePayloadFormat
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import LeastMagnitudeControllerStudy, ImplementationBinding, ImplementationRole
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionSpec, FiniteCertificateReachabilitySpec, FiniteCertificateReachabilityComparison
from empirical_lawhood.planning.geometry import AdmissionComparison
from empirical_lawhood.planning.nested_controller_evaluation import CommonStartControllerEvaluationPlan
from empirical_lawhood.runtime.controller_compiler import CompiledLeastMagnitudeControllerStudy
from empirical_lawhood.runtime.controller_runtime import RuntimeObservation, ObserverEvaluation, NumericalViewLiveGateEvaluation, DeliveryPortResult, ObserverPort, NumericalViewLiveGateEvaluatorPort, NativeDeliveryPort
from empirical_lawhood.runtime.controller_evaluation_nested import NestedControllerUseEvaluator, RevealedPreparedForecastPolicyBundle, PreparedPolicyUnitEvaluation
from empirical_lawhood.adapters.control.composition import ControllerStudyComposition, controller_capability_registry
from empirical_lawhood.adapters.control.evidence_services import FiniteCertificateAdmissionDeriver, FiniteCertificateReachabilityDeriver
from empirical_lawhood.adapters.control.finite_instance_observer import (
    FiniteInstanceObserver,
    FiniteInstanceOnlineGateEvaluator,
)
from .control_delivery import FiniteResponseLawNativeReceiptDelivery
from .science import FiniteResponseLawScienceSpec

_OWNERS = (
    (
        ImplementationRole.ADMISSION_DERIVER,
        'FiniteCertificateAdmissionDeriver',
        FiniteCertificateAdmissionSpec.SCHEMA,
        AdmissionComparison.SCHEMA,
    ),
    (
        ImplementationRole.REACHABILITY_DERIVER,
        'FiniteCertificateReachabilityDeriver',
        FiniteCertificateReachabilitySpec.SCHEMA,
        FiniteCertificateReachabilityComparison.SCHEMA,
    ),
    (
        ImplementationRole.SYNTHESIZER,
        "ControllerCompiler.least-magnitude",
        LeastMagnitudeControllerStudy.SCHEMA,
        CompiledLeastMagnitudeControllerStudy.SCHEMA,
    ),
    (
        ImplementationRole.OBSERVER,
        "FiniteInstanceObserver",
        RuntimeObservation.SCHEMA,
        ObserverEvaluation.SCHEMA,
    ),
    (
        ImplementationRole.ONLINE_GATE_EVALUATOR,
        "FiniteInstanceOnlineGateEvaluator",
        LeastMagnitudeControllerStudy.SCHEMA,
        NumericalViewLiveGateEvaluation.SCHEMA,
    ),
    (
        ImplementationRole.DELIVERY,
        'FiniteResponseLawNativeReceiptDelivery',
        OccurrenceActionWord.SCHEMA,
        DeliveryPortResult.SCHEMA,
    ),
    (
        ImplementationRole.OUTCOME_EVALUATOR,
        "NestedControllerUseEvaluator.prepared-forecast",
        RevealedPreparedForecastPolicyBundle.SCHEMA,
        PreparedPolicyUnitEvaluation.SCHEMA,
    ),
)


@dataclass(frozen=True, slots=True)
class FiniteResponseLawControllerBindingConfig(CanonicalRecord):
    SCHEMA: ClassVar[str] = 'empirical-lawhood/methods/finite-response-law/finite-response-law-controller-binding-config'
    role: ImplementationRole
    owner: str
    science_sha256: str

    def __post_init__(self) -> None:
        if (self.role, self.owner) not in {
            (r, o) for r, o, _, _ in _OWNERS
        } or self.science_sha256 != FiniteResponseLawScienceSpec().fingerprint():
            raise ValueError("Finite response-law evaluation controller role configuration changes its closed owner or science")

    @property
    def config_id(self) -> str:
        return f"finite-response-law.prospective-control.{self.role.value.lower()}.config"


def implementation_configurations() -> tuple[FiniteResponseLawControllerBindingConfig, ...]:
    return tuple(
        sorted(
            (
                FiniteResponseLawControllerBindingConfig(r, owner, FiniteResponseLawScienceSpec().fingerprint())
                for r, owner, _, _ in _OWNERS
            ),
            key=lambda c: c.config_id,
        )
    )


def implementation_payloads() -> tuple[tuple[ImplementationBinding, bytes], ...]:
    """Exact metadata for these composed owners; caller must publish all payloads."""
    result = []
    for role, owner, input_schema, output_schema in _OWNERS:
        stem = f"finite-response-law.prospective-control.{role.value.lower()}"
        raw = FiniteResponseLawControllerBindingConfig(
            role, owner, FiniteResponseLawScienceSpec().fingerprint()
        ).canonical_bytes()
        digest = sha256(raw).hexdigest()
        reference = ExecutableReference(
            stem,
            stem,
            "1.0.0",
            f"{stem}.owner",
            ArtifactIdentity(
                f"{stem}.config",
                "prepared-control-record",
                FiniteResponseLawControllerBindingConfig.SCHEMA,
                digest,
                "application/json",
                len(raw),
            ),
            SafePayloadFormat.CANONICAL_JSON,
            input_schema,
            output_schema,
            True,
        )
        result.append(
            (
                ImplementationBinding(
                    stem,
                    role,
                    reference,
                    digest,
                    sha256(f"{owner}:finite-response-law-native-binding".encode()).hexdigest(),
                ),
                raw,
            )
        )
    return tuple(sorted(result, key=lambda item: item[0].binding_id))


def control_services(
    study: LeastMagnitudeControllerStudy,
    evaluation_plan: CommonStartControllerEvaluationPlan | None,
    delivery: FiniteResponseLawNativeReceiptDelivery,
) -> ControllerStudyComposition:
    """Construct only the declared existing owners, with no reference-world port."""
    bindings = tuple(
        b
        for b, _ in implementation_payloads()
        if evaluation_plan is not None or b.role is not ImplementationRole.OUTCOME_EVALUATOR
    )
    if study.implementations != bindings:
        raise ValueError("Finite response-law controller changed its closed implementation owners")
    roles = {b.role: b for b in bindings}
    if delivery.implementation_binding != roles[ImplementationRole.DELIVERY]:
        raise ValueError("Finite response-law controller changed its actual receipt delivery owner")
    from empirical_lawhood.kernel.provenance import ObjectIdentity

    identity = (
        None
        if evaluation_plan is None
        else ObjectIdentity.from_record(evaluation_plan.evaluation_plan_id, evaluation_plan)
    )
    if study.prospective_evaluation != identity:
        raise ValueError("Finite response-law controller changed its exact prospective evaluation census")
    cells = {c.decision_cell_id for c in study.synthesis.candidate_chart.candidates}
    if len(cells) != 1:
        raise ValueError("Finite response-law instances have exactly one committed root/request decision cell")
    return ControllerStudyComposition(
        controller_capability_registry(
            bindings, controller_study_schema=LeastMagnitudeControllerStudy.SCHEMA
        ),
        FiniteCertificateAdmissionDeriver(roles[ImplementationRole.ADMISSION_DERIVER]),
        FiniteCertificateReachabilityDeriver(roles[ImplementationRole.REACHABILITY_DERIVER]),
        cast(
            ObserverPort,
            FiniteInstanceObserver(
                roles[ImplementationRole.OBSERVER], study.instance_binding, next(iter(cells))
            ),
        ),
        cast(
            NumericalViewLiveGateEvaluatorPort,
            FiniteInstanceOnlineGateEvaluator(
                roles[ImplementationRole.ONLINE_GATE_EVALUATOR], study.instance_binding
            ),
        ),
        cast(NativeDeliveryPort, delivery),
        None
        if evaluation_plan is None
        else NestedControllerUseEvaluator(
            roles[ImplementationRole.OUTCOME_EVALUATOR], evaluation_plan.reducer
        ),
    )

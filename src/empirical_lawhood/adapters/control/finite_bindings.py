"Substrate-neutral contracts for finite admission and prepared controller use implementations."

from hashlib import sha256
from empirical_lawhood.kernel.action_contracts import OccurrenceActionWord
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, SafePayloadFormat
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.controller_study import DeliveryControllerStudy, ImplementationRole, ImplementationBinding
from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionSpec, FiniteCertificateReachabilitySpec, FiniteCertificateReachabilityComparison
from empirical_lawhood.planning.geometry import AdmissionComparison
from empirical_lawhood.runtime.controller_compiler import CompiledDeliveryControllerStudy
from empirical_lawhood.runtime.controller_runtime import RuntimeObservation, ObserverEvaluation, NumericalViewLiveGateEvaluation, DeliveryPortResult
from empirical_lawhood.runtime.controller_evaluation_nested import RevealedPreparedFuture, PreparedPolicyUnitEvaluation


def finite_owner_contracts(
    delivery_owner: str,
) -> tuple[tuple[ImplementationRole, str, str, str], ...]:
    """The installed finite compiler/runtime owners plus a native delivery adapter."""
    return (
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
            "ControllerCompiler.finite-delivery",
            DeliveryControllerStudy.SCHEMA,
            CompiledDeliveryControllerStudy.SCHEMA,
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
            DeliveryControllerStudy.SCHEMA,
            NumericalViewLiveGateEvaluation.SCHEMA,
        ),
        (
            ImplementationRole.DELIVERY,
            delivery_owner,
            OccurrenceActionWord.SCHEMA,
            DeliveryPortResult.SCHEMA,
        ),
        (
            ImplementationRole.OUTCOME_EVALUATOR,
            "NestedControllerUseEvaluator.prepared",
            RevealedPreparedFuture.SCHEMA,
            PreparedPolicyUnitEvaluation.SCHEMA,
        ),
    )


def finite_implementation_payloads(
    configurations: tuple[tuple[str, ImplementationRole, CanonicalRecord], ...],
    *,
    delivery_owner: str,
    implementation_namespace: str,
) -> tuple[tuple[ImplementationBinding, bytes], ...]:
    """Bind canonical configurations to installed owners; never resolve code from config."""
    owners = {
        role: (owner, source, target)
        for role, owner, source, target in finite_owner_contracts(delivery_owner)
    }
    if len({role for _, role, _ in configurations}) != len(configurations):
        raise ValueError("finite implementation configurations repeat a role")
    result = []
    for config_id, role, config in configurations:
        if not config_id.endswith(".config"):
            raise ValueError("finite implementation configuration lacks its exact config ID")
        owner, input_schema, output_schema = owners[role]
        stem, raw = config_id.removesuffix(".config"), config.canonical_bytes()
        digest = sha256(raw).hexdigest()
        reference = ExecutableReference(
            stem,
            stem,
            "1.0.0",
            f"{stem}.owner",
            ArtifactIdentity(
                config_id,
                "prepared-control-record",
                config.SCHEMA,
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
                    sha256(f"{owner}:{implementation_namespace}".encode()).hexdigest(),
                ),
                raw,
            )
        )
    return tuple(sorted(result, key=lambda item: item[0].binding_id))

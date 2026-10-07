"Fingerprint-bound capability manifests for the baseline parametric response-method family."

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.identification import LawIdentificationResult, LawQualificationResult, StructuralConvergenceResult
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.response_algebra import ResponseAlgebraIdentificationResult
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.kernel.thermodynamic_response import (
    StructuralTransportWitness,
    ThermodynamicResponseSignature,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPublicationReceipt

from .contracts import (
    CandidateFit,
    IdentificationDataset,
    LawCandidateAxisMap,
    LawIdentificationConfig,
)
from .law_evaluation import LawEvaluationRequest, LawEvaluationResult, LawEvaluatorRegistration
from .odd_jacobian import (
    ODD_RELATIONAL_JACOBIAN_METHOD_KEY,
    ODD_RELATIONAL_JACOBIAN_METHOD_VERSION,
    OddJacobianDataset,
    OddJacobianFit,
    OddRelationalJacobianConfig,
)
from .response_algebra import (
    ResponseAlgebraMethodConfig,
    ResponseAlgebraMethodInput,
    ResponseAlgebraMethodResult,
)
from .thermodynamic_response_analysis import (
    ThermodynamicResponseMethodConfig,
    ThermodynamicResponseMethodInput,
    ThermodynamicResponseMethodResult,
)
from .thermodynamic_response_posthoc import (
    ThermodynamicPosthocConfig,
    ThermodynamicPosthocInputBinding,
    ThermodynamicPosthocResultIndex,
)
from .transport import (
    DirectionalTransportAssessment,
    DirectionalTransportDataset,
    DiscrepancyEstimatorConfig,
    ObservationOperatorSpec,
    PairedTransportObservation,
)

_METHOD_KEYS = (
    "baseline.local-linear",
    "baseline.local-state-space",
    "baseline.nonlinear-local",
)

_RESPONSE_METHOD_AUXILIARY_KEYS = (
    "baseline.directional-max-residual",
    "baseline.identity-observation-operator",
    "baseline.numerical-qualifier",
)

_LAW_EVALUATOR_KEYS = (
    "law-evaluator.canonical-controlled-io",
    "law-evaluator.canonical-finite-action",
    "law-evaluator.canonical-parametric",
)

THERMODYNAMIC_RESPONSE_ANALYSIS_KEY = "thermodynamic-response.finite-analysis"
THERMODYNAMIC_RESPONSE_EVALUATOR_KEY = "thermodynamic-response.sealed-evaluator"
THERMODYNAMIC_RESPONSE_TRANSPORT_KEY = "thermodynamic-response.structural-transport"
THERMODYNAMIC_RESPONSE_POSTHOC_ANALYSIS_KEY = "thermodynamic-response.outcome-visible-posthoc"

_THERMODYNAMIC_RESPONSE_KEYS = (
    THERMODYNAMIC_RESPONSE_ANALYSIS_KEY,
    THERMODYNAMIC_RESPONSE_EVALUATOR_KEY,
    THERMODYNAMIC_RESPONSE_TRANSPORT_KEY,
)


def _schema_identity_sha256(schema: str) -> str:
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def law_method_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    """Build immutable manifests from package/build-provided source identities."""

    if set(implementation_sha256_by_key) != set(_METHOD_KEYS):
        raise ValueError("law method implementation identities must cover the exact registry")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    return tuple(
        CapabilityManifest(
            capability_key=key,
            capability_version="1.0.0",
            kind=CapabilityKind.LAW_IDENTIFIER,
            config_schema=LawIdentificationConfig.SCHEMA,
            config_schema_sha256=_schema_identity_sha256(LawIdentificationConfig.SCHEMA),
            input_schema_ids=(IdentificationDataset.SCHEMA,),
            output_schema_ids=(LawIdentificationResult.SCHEMA,),
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            resource_ceiling=ResourceBudget(
                cpu_cores=4,
                memory_bytes=8_000_000_000,
                gpu_devices=0,
                wall_time_seconds=3_600,
                source_scan_bytes=100_000_000_000,
                output_bytes=1_000_000_000,
            ),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=(
                "response-method-complete-adequacy-family",
                "response-method-independent-unit-scope",
                "response-method-method-interchangeability",
                "response-method-no-predictive-bypass",
                "response-method-safe-canonical-payload",
                "response-method-structural-refinement",
                "response-method-uncertainty-separation",
            ),
            implementation_sha256=implementation_sha256_by_key[key],
        )
        for key in _METHOD_KEYS
    )


def law_method_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="response-method-baseline-law-methods",
        capabilities=law_method_manifests(implementation_sha256_by_key),
    )


def response_method_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    """Register law, numerical, observation and discrepancy seams separately."""

    expected = {*_METHOD_KEYS, *_RESPONSE_METHOD_AUXILIARY_KEYS}
    if set(implementation_sha256_by_key) != expected:
        raise ValueError("Parametric response-method implementation identities must cover the exact capability family")
    law_manifests = law_method_manifests(
        {key: implementation_sha256_by_key[key] for key in _METHOD_KEYS}
    )
    specifications = (
        (
            "baseline.directional-max-residual",
            CapabilityKind.DISCREPANCY_ESTIMATOR,
            DiscrepancyEstimatorConfig.SCHEMA,
            (DirectionalTransportDataset.SCHEMA,),
            (DirectionalTransportAssessment.SCHEMA,),
            (
                "response-method-calibration-heldout-disjoint",
                "response-method-directional-transport",
                "response-method-heldout-does-not-fit-discrepancy",
                "response-method-native-unit-alignment",
            ),
        ),
        (
            "baseline.identity-observation-operator",
            CapabilityKind.OBSERVATION_OPERATOR,
            ObservationOperatorSpec.SCHEMA,
            (PairedTransportObservation.SCHEMA,),
            ('empirical-lawhood/methods/aligned-transport-observation',),
            (
                "response-method-clock-alignment",
                "response-method-native-unit-alignment",
                "response-method-safe-canonical-payload",
            ),
        ),
        (
            "baseline.numerical-qualifier",
            CapabilityKind.NUMERICAL_QUALIFIER,
            LawIdentificationConfig.SCHEMA,
            tuple(sorted((CandidateFit.SCHEMA, IdentificationDataset.SCHEMA))),
            (StructuralConvergenceResult.SCHEMA,),
            (
                "response-method-independent-unit-scope",
                "response-method-structural-refinement",
                "response-method-uncertainty-separation",
            ),
        ),
    )
    auxiliary = tuple(
        CapabilityManifest(
            capability_key=key,
            capability_version="1.0.0",
            kind=kind,
            config_schema=config_schema,
            config_schema_sha256=_schema_identity_sha256(config_schema),
            input_schema_ids=input_schemas,
            output_schema_ids=output_schemas,
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            resource_ceiling=ResourceBudget(
                cpu_cores=4,
                memory_bytes=8_000_000_000,
                gpu_devices=0,
                wall_time_seconds=3_600,
                source_scan_bytes=100_000_000_000,
                output_bytes=1_000_000_000,
            ),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=checks,
            implementation_sha256=implementation_sha256_by_key[key],
        )
        for key, kind, config_schema, input_schemas, output_schemas, checks in specifications
    )
    return tuple(sorted((*law_manifests, *auxiliary), key=lambda value: value.registry_id))


def response_method_capability_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="response-method-scientific-capabilities",
        capabilities=response_method_capability_manifests(implementation_sha256_by_key),
    )


def law_evaluation_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    """Register the three closed support-limited law evaluators."""

    if set(implementation_sha256_by_key) != set(_LAW_EVALUATOR_KEYS):
        raise ValueError("law evaluator identities must cover the exact representation family")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    inputs = tuple(
        sorted(
            (
                CandidatePayloadPublicationReceipt.SCHEMA,
                LawCandidateAxisMap.SCHEMA,
                LawEvaluationRequest.SCHEMA,
                LawQualificationResult.SCHEMA,
                ResponseLaw.SCHEMA,
                SystemSpec.SCHEMA,
            )
        )
    )
    return tuple(
        CapabilityManifest(
            capability_key=key,
            capability_version="1.0.0",
            kind=CapabilityKind.EVALUATOR,
            config_schema=LawEvaluatorRegistration.SCHEMA,
            config_schema_sha256=_schema_identity_sha256(LawEvaluatorRegistration.SCHEMA),
            input_schema_ids=inputs,
            output_schema_ids=(LawEvaluationResult.SCHEMA,),
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
            resource_ceiling=ResourceBudget(
                cpu_cores=4,
                memory_bytes=8 * 1024**3,
                gpu_devices=0,
                wall_time_seconds=3_600,
                source_scan_bytes=1024**3,
                output_bytes=512 * 1024**2,
            ),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11-law-evaluation",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=(
                "bounded-canonical-payload-decode",
                "exact-action-word-and-realized-occurrence",
                "exact-denominator-member-version-refinement",
                "native-unit-frame-clock-preservation",
                "no-outside-support-interpolation",
                "payload-and-implementation-drift-refusal",
                "registered-offline-online-resource-envelope",
                "typed-member-operand-product-refusal",
            ),
            implementation_sha256=implementation_sha256_by_key[key],
        )
        for key in _LAW_EVALUATOR_KEYS
    )


def law_evaluation_capability_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="response-law-evaluators",
        capabilities=law_evaluation_capability_manifests(implementation_sha256_by_key),
    )


def odd_relational_jacobian_manifest(*, implementation_sha256: str) -> CapabilityManifest:
    """Register the corrected odd state-dependent response-law method."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    return CapabilityManifest(
        capability_key=ODD_RELATIONAL_JACOBIAN_METHOD_KEY,
        capability_version=ODD_RELATIONAL_JACOBIAN_METHOD_VERSION,
        kind=CapabilityKind.LAW_IDENTIFIER,
        config_schema=OddRelationalJacobianConfig.SCHEMA,
        config_schema_sha256=_schema_identity_sha256(OddRelationalJacobianConfig.SCHEMA),
        input_schema_ids=(OddJacobianDataset.SCHEMA,),
        output_schema_ids=(OddJacobianFit.SCHEMA,),
        permissions=(
            CapabilityPermission.READ_DEVELOPMENT,
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        resource_ceiling=ResourceBudget(
            cpu_cores=4,
            memory_bytes=8 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=3_600,
            source_scan_bytes=4 * 1024**3,
            output_bytes=512 * 1024**2,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11-odd-relational-jacobian",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "complete-comparator-plus-minus-cell",
            "development-only-scaling-regularization-support",
            "exact-zero-action",
            "family-balanced-receiver-loss",
            "native-action-and-receiver-units",
            "odd-action-sign-reversal",
            "physical-preparation-independent-unit",
            "state-observed-by-causal-cutoff",
        ),
        implementation_sha256=implementation_sha256,
    )


def odd_relational_jacobian_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="odd-relational-jacobian-method",
        capabilities=(
            odd_relational_jacobian_manifest(implementation_sha256=implementation_sha256),
        ),
    )


def response_algebra_identifier_manifest(*, implementation_sha256: str) -> CapabilityManifest:
    """Register the transparent development-side identifier used by an evaluator wrapper."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    return CapabilityManifest(
        capability_key="response-algebra.direct-affine-finite",
        capability_version="1.0.0",
        kind=CapabilityKind.RESPONSE_ALGEBRA_IDENTIFIER,
        config_schema=ResponseAlgebraMethodConfig.SCHEMA,
        config_schema_sha256=_schema_identity_sha256(ResponseAlgebraMethodConfig.SCHEMA),
        input_schema_ids=(ResponseAlgebraMethodInput.SCHEMA,),
        output_schema_ids=tuple(
            sorted(
                (
                    ResponseAlgebraIdentificationResult.SCHEMA,
                    ResponseAlgebraMethodResult.SCHEMA,
                )
            )
        ),
        permissions=(
            CapabilityPermission.READ_DEVELOPMENT,
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resource_ceiling=ResourceBudget(
            cpu_cores=4,
            memory_bytes=8 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=3_600,
            source_scan_bytes=4 * 1024**3,
            output_bytes=512 * 1024**2,
        ),
        deterministic=False,
        seed_required=True,
        language_id="python",
        runtime_id="cpython-3.11-response-algebra",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "affine-intercept-composition",
            "complete-independent-unit-bootstrap",
            "constituent-port-materiality",
            "direct-word-timing-control",
            "exact-four-stage-delivery",
            "finite-stochastic-commutator",
            "no-llm-or-rl",
            "receiver-faithfulness",
            "smooth-lie-eligibility",
            "state-view-coordinate-unit-binding",
            "truth-blind-input",
            "wrong-horizon-specificity",
        ),
        implementation_sha256=implementation_sha256,
    )


def response_algebra_identifier_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="response-algebra-identifier",
        capabilities=(
            response_algebra_identifier_manifest(
                implementation_sha256=implementation_sha256,
            ),
        ),
    )


def thermodynamic_response_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    """Register analysis, sealed evaluation and transport as separate authorities."""

    if set(implementation_sha256_by_key) != set(_THERMODYNAMIC_RESPONSE_KEYS):
        raise ValueError(
            "thermodynamic-response implementation identities must cover the exact roles"
        )
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")

    def build(
        *,
        key: str,
        kind: CapabilityKind,
        input_schema_ids: tuple[str, ...],
        output_schema_ids: tuple[str, ...],
        permissions: tuple[CapabilityPermission, ...],
        outcome_access: OutcomeAccess,
        checks: tuple[str, ...],
    ) -> CapabilityManifest:
        return CapabilityManifest(
            capability_key=key,
            capability_version="1.0.0",
            kind=kind,
            config_schema=ThermodynamicResponseMethodConfig.SCHEMA,
            config_schema_sha256=_schema_identity_sha256(ThermodynamicResponseMethodConfig.SCHEMA),
            input_schema_ids=input_schema_ids,
            output_schema_ids=output_schema_ids,
            permissions=permissions,
            maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
            maximum_outcome_access=outcome_access,
            resource_ceiling=ResourceBudget(
                cpu_cores=4,
                memory_bytes=8 * 1024**3,
                gpu_devices=0,
                wall_time_seconds=3_600,
                source_scan_bytes=4 * 1024**3,
                output_bytes=512 * 1024**2,
            ),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="cpython-3.11-thermodynamic-response",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=checks,
            implementation_sha256=implementation_sha256_by_key[key],
        )

    return (
        build(
            key=THERMODYNAMIC_RESPONSE_ANALYSIS_KEY,
            kind=CapabilityKind.ANALYSIS,
            input_schema_ids=(ThermodynamicResponseMethodInput.SCHEMA,),
            output_schema_ids=(ThermodynamicResponseMethodResult.SCHEMA,),
            permissions=(
                CapabilityPermission.READ_DEVELOPMENT,
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            checks=(
                "absolute-versus-differenced-exchange",
                "finite-word-physical-unit-grouping",
                "no-llm-or-rl",
                "receiver-naturality-faithfulness-separation",
                "truth-blind-method-input",
                "typed-thermodynamic-claim-ceiling",
            ),
        ),
        build(
            key=THERMODYNAMIC_RESPONSE_EVALUATOR_KEY,
            kind=CapabilityKind.EVALUATOR,
            input_schema_ids=(ThermodynamicResponseMethodInput.SCHEMA,),
            output_schema_ids=(ThermodynamicResponseMethodResult.SCHEMA,),
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            checks=(
                "evaluator-only-sealed-outcome-reveal",
                "finite-word-physical-unit-grouping",
                "immutable-evaluation-input",
                "no-llm-or-rl",
                "typed-negative-annular-unevaluable-results",
            ),
        ),
        build(
            key=THERMODYNAMIC_RESPONSE_TRANSPORT_KEY,
            kind=CapabilityKind.TRANSPORT_TESTER,
            input_schema_ids=(ThermodynamicResponseSignature.SCHEMA,),
            output_schema_ids=(StructuralTransportWitness.SCHEMA,),
            permissions=(
                CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                CapabilityPermission.READ_OUTCOME_VISIBLE,
                CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
            ),
            outcome_access=OutcomeAccess.EVALUATION_REVEALED,
            checks=(
                "common-support-required",
                "decisive-counterexample-preservation",
                "native-axis-tolerances",
                "no-llm-or-rl",
                "no-native-numeric-pooling",
                "receiver-faithfulness-limit",
                "typed-action-time-receiver-thermodynamic-maps",
            ),
        ),
    )


def thermodynamic_response_capability_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="thermodynamic-response-capabilities",
        capabilities=thermodynamic_response_capability_manifests(implementation_sha256_by_key),
    )


def thermodynamic_response_posthoc_analysis_manifest(
    *, implementation_sha256: str
) -> CapabilityManifest:
    """Register the separate outcome-visible, nonpromotable analysis wave."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    return CapabilityManifest(
        capability_key=THERMODYNAMIC_RESPONSE_POSTHOC_ANALYSIS_KEY,
        capability_version="1.0.0",
        kind=CapabilityKind.ANALYSIS,
        config_schema=ThermodynamicPosthocConfig.SCHEMA,
        config_schema_sha256=_schema_identity_sha256(ThermodynamicPosthocConfig.SCHEMA),
        input_schema_ids=(ThermodynamicPosthocInputBinding.SCHEMA,),
        output_schema_ids=(ThermodynamicPosthocResultIndex.SCHEMA,),
        permissions=(
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.READ_OUTCOME_VISIBLE,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.EVALUATION_REVEALED,
        resource_ceiling=ResourceBudget(
            cpu_cores=4,
            memory_bytes=8 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=7_200,
            source_scan_bytes=512 * 1024**2,
            output_bytes=512 * 1024**2,
        ),
        # The implementation is reproducible only when the frozen seed is
        # supplied. Runtime semantics prohibit describing a seed-requiring
        # capability as intrinsically deterministic.
        deterministic=False,
        seed_required=True,
        language_id="python",
        runtime_id="cpython-3.11-thermodynamic-response-posthoc",
        requires_clean_commit=False,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "causal-cutoff-and-clock-preservation",
            "claim-monotonicity-under-information-removal",
            "complete-analysis-family-reporting",
            "complete-unit-resampling",
            "deterministic-fixed-seed-resampling",
            "immutable-parent-content-hashes",
            "low-snr-and-absent-operand-dispositions",
            "native-unit-and-evidence-world-separation",
            "no-llm-or-rl",
            "nonpromotion-after-outcome-visibility",
            "receiver-role-closure",
            "truth-oracle-input-separation",
        ),
        implementation_sha256=implementation_sha256,
    )


def thermodynamic_response_posthoc_analysis_registry(
    *, implementation_sha256: str
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="thermodynamic-response-posthoc-analysis",
        capabilities=(
            thermodynamic_response_posthoc_analysis_manifest(
                implementation_sha256=implementation_sha256
            ),
        ),
    )

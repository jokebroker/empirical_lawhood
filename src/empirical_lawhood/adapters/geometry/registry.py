"""Fingerprint-bound atlas, admission, and reachability capabilities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from empirical_lawhood.kernel.admission import AdmissionSet, ReachabilityResult
from empirical_lawhood.kernel.atlases import ResponseAtlas
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.laws import ResponseLaw
from empirical_lawhood.kernel.models import ViewModelSetSpec
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.kernel.systems import SystemSpec
from empirical_lawhood.planning.geometry import (
    AdmissionComparison,
    AdmissionEvaluationSpec,
    AtlasAssemblyResult,
    AtlasAssemblySpec,
    ReachabilityComparison,
    ReachabilityEvaluationSpec,
)
from empirical_lawhood.planning.evidence_geometry import ReceiptAdmissionSpec, ControlledMapReachabilitySpec, AdmissionCoordinateGateReceipt, ControlledMapAdmissionReceiptCorpus, ReceiptAdmissionReceiptProductionPlan, VerifiedAdmissionReachabilityCategoricalProjection, ControlledMapReachabilityComparison, ControlledMapReachabilityReceipt, UtilityEvaluationReceipt
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from empirical_lawhood.planning.finite_admission import FiniteCertificateAdmissionSpec, FiniteCertificateReachabilitySpec, FiniteCertificateReachabilityComparison
from empirical_lawhood.planning.finite_response_geometry import FiniteReachabilityMethodReceipt, FiniteResponseSetReachabilityRequest, FiniteResponseSetReachabilityResult
from .finite_response_reachability import (
    FiniteReachabilityAdmissionReceiptProducer,
    FiniteResponseSetReachabilityMethod,
)
from .controlled_io_reachability import (
    ControlledInputOutputReachabilityAdmissionMethod,
    ControlledIOReachabilityRequest,
)
from empirical_lawhood.adapters.methods.receiver_conditioned_io.contracts import ControlledIOMember
from .reachability import FiniteGridReachability
from .admission_receipts import AdmissionGateRawInput, AdmissionReachabilityRawInput, AdmissionUtilityRawInput, RawAdmissionGateReceiptProducer, RawReachabilityAdmissionReceiptProducer, RawUtilityAdmissionReceiptProducer
from .services import (
    AtlasAssembler,
    FiniteAdmissionProjector,
    FiniteReachabilityAdmissionProjector,
    RawAdmissionProjector,
    RawAdmissionReachabilityCategoricalProjectionEvaluator,
    RawReachabilityAdmissionProjector,
    ReceiverAdmissionEvaluator,
)


def atlas_capability_keys() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                AtlasAssembler.capability_key,
                FiniteGridReachability.capability_key,
                ReceiverAdmissionEvaluator.capability_key,
            )
        )
    )


def _schema_sha256(schema: str) -> str:
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def _resources() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=4,
        memory_bytes=8_000_000_000,
        gpu_devices=0,
        wall_time_seconds=3_600,
        source_scan_bytes=100_000_000_000,
        output_bytes=1_000_000_000,
    )


def _manifest(
    *,
    key: str,
    kind: CapabilityKind,
    config_schema: str,
    input_schemas: tuple[str, ...],
    output_schemas: tuple[str, ...],
    checks: tuple[str, ...],
    maximum_evidence_ceiling: EvidenceCeiling,
    implementation_sha256: str,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version="1.0.0",
        kind=kind,
        config_schema=config_schema,
        config_schema_sha256=_schema_sha256(config_schema),
        input_schema_ids=tuple(sorted(input_schemas)),
        output_schema_ids=tuple(sorted(output_schemas)),
        permissions=(
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=maximum_evidence_ceiling,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        resource_ceiling=_resources(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=tuple(sorted(checks)),
        implementation_sha256=implementation_sha256,
    )


def atlas_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    expected = set(atlas_capability_keys())
    if set(implementation_sha256_by_key) != expected:
        raise ValueError("R8 implementation identities must cover the exact capability family")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    specifications = (
        (
            AtlasAssembler.capability_key,
            CapabilityKind.ATLAS_ASSEMBLER,
            AtlasAssemblySpec.SCHEMA,
            (ResponseLaw.SCHEMA, SystemSpec.SCHEMA),
            (AtlasAssemblyResult.SCHEMA, ResponseAtlas.SCHEMA),
            (
                "atlas-explicit-gaps-and-overlaps",
                "atlas-no-global-smoothness",
                "atlas-source-law-immutability",
                "atlas-transition-evidence",
            ),
            EvidenceCeiling.LOCAL_LAW,
        ),
        (
            ReceiverAdmissionEvaluator.capability_key,
            CapabilityKind.ADMISSION_EVALUATOR,
            AdmissionEvaluationSpec.SCHEMA,
            (ViewModelSetSpec.SCHEMA, ResponseAtlas.SCHEMA),
            (AdmissionComparison.SCHEMA, AdmissionSet.SCHEMA),
            (
                "admission-complete-nine-gate-intersection",
                "admission-mandatory-hold-outside",
                "admission-nominal-robust-separation",
                "admission-reason-coded-exclusions",
            ),
            EvidenceCeiling.ADMISSION,
        ),
        (
            FiniteGridReachability.capability_key,
            CapabilityKind.REACHABILITY_EVALUATOR,
            ReachabilityEvaluationSpec.SCHEMA,
            (AdmissionComparison.SCHEMA, ViewModelSetSpec.SCHEMA),
            (ReachabilityComparison.SCHEMA, ReachabilityResult.SCHEMA),
            (
                "reachability-computability-boundary",
                "reachability-finite-grid-method-registration",
                "reachability-numerical-qualification",
                "reachability-structural-stability",
            ),
            EvidenceCeiling.ADMISSION,
        ),
    )
    return tuple(
        sorted(
            (
                _manifest(
                    key=key,
                    kind=kind,
                    config_schema=config_schema,
                    input_schemas=input_schemas,
                    output_schemas=output_schemas,
                    checks=checks,
                    maximum_evidence_ceiling=maximum_evidence_ceiling,
                    implementation_sha256=implementation_sha256_by_key[key],
                )
                for (
                    key,
                    kind,
                    config_schema,
                    input_schemas,
                    output_schemas,
                    checks,
                    maximum_evidence_ceiling,
                ) in specifications
            ),
            key=lambda item: item.registry_id,
        )
    )


def atlas_capability_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="atlas-admission-reachability-capabilities",
        capabilities=atlas_capability_manifests(implementation_sha256_by_key),
    )


def admission_capability_keys() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                RawAdmissionProjector.capability_key,
                ControlledInputOutputReachabilityAdmissionMethod.capability_key,
                RawAdmissionGateReceiptProducer.capability_key,
                RawAdmissionReachabilityCategoricalProjectionEvaluator.capability_key,
                RawReachabilityAdmissionProjector.capability_key,
                RawReachabilityAdmissionReceiptProducer.capability_key,
                RawUtilityAdmissionReceiptProducer.capability_key,
            )
        )
    )


def _admission_manifest(
    *,
    key: str,
    kind: CapabilityKind,
    input_schemas: tuple[str, ...],
    output_schemas: tuple[str, ...],
    checks: tuple[str, ...],
    implementation_sha256: str,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version="1.0.0",
        kind=kind,
        config_schema=ReceiptAdmissionReceiptProductionPlan.SCHEMA,
        config_schema_sha256=_schema_sha256(ReceiptAdmissionReceiptProductionPlan.SCHEMA),
        input_schema_ids=tuple(sorted(input_schemas)),
        output_schema_ids=tuple(sorted(output_schemas)),
        permissions=(CapabilityPermission.READ_EXTERNAL_ARTIFACTS,),
        maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        resource_ceiling=ResourceBudget(
            cpu_cores=4,
            memory_bytes=4 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=600,
            source_scan_bytes=8 * 1024**3,
            output_bytes=512 * 1024**2,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11-controlled-map-admission",
        requires_clean_commit=True,
        requires_active_mount=False,
        requires_network=False,
        conformance_check_ids=tuple(sorted(checks)),
        implementation_sha256=implementation_sha256,
    )


def admission_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    expected = set(admission_capability_keys())
    if set(implementation_sha256_by_key) != expected:
        raise ValueError("Controlled-map admission implementation identities must cover the exact capability family")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    specifications = (
        (
            ControlledInputOutputReachabilityAdmissionMethod.capability_key,
            CapabilityKind.REACHABILITY_EVALUATOR,
            (
                ControlledIOMember.SCHEMA,
                ControlledIOReachabilityRequest.SCHEMA,
                ReceiptAdmissionReceiptProductionPlan.SCHEMA,
            ),
            (AdmissionReachabilityRawInput.SCHEMA,),
            (
                "admission-active-minus-qualified-hold-exact-action-word",
                "admission-cphi-b-full-map-only",
                "admission-occurrence-clock-and-input-step-projection",
                "admission-rank-conditioning-and-native-output-constraints",
            ),
        ),
        (
            RawAdmissionGateReceiptProducer.capability_key,
            CapabilityKind.ADMISSION_EVALUATOR,
            (AdmissionGateRawInput.SCHEMA, ReceiptAdmissionReceiptProductionPlan.SCHEMA),
            (AdmissionCoordinateGateReceipt.SCHEMA,),
            (
                "admission-exact-action-law-member-version-binding",
                "admission-nine-gate-noncompensating-derivation",
                "admission-typed-non-evaluation-dispositions",
            ),
        ),
        (
            RawReachabilityAdmissionReceiptProducer.capability_key,
            CapabilityKind.REACHABILITY_EVALUATOR,
            (AdmissionReachabilityRawInput.SCHEMA, ReceiptAdmissionReceiptProductionPlan.SCHEMA),
            (ControlledMapReachabilityReceipt.SCHEMA,),
            (
                "admission-cphi-b-full-map-only",
                "admission-direction-gate-consistency",
                "admission-rank-conditioning-veto",
            ),
        ),
        (
            RawUtilityAdmissionReceiptProducer.capability_key,
            CapabilityKind.CONTROLLER_SYNTHESIZER,
            (ReceiptAdmissionReceiptProductionPlan.SCHEMA, AdmissionUtilityRawInput.SCHEMA),
            (UtilityEvaluationReceipt.SCHEMA,),
            (
                "admission-native-unit-robust-utility",
                "admission-no-outcome-visible-selection",
                "admission-uncertainty-noncompensation",
            ),
        ),
        (
            RawAdmissionProjector.capability_key,
            CapabilityKind.ADMISSION_EVALUATOR,
            (ReceiptAdmissionSpec.SCHEMA, ControlledMapAdmissionReceiptCorpus.SCHEMA),
            (AdmissionComparison.SCHEMA,),
            (
                "admission-complete-member-version-intersection",
                "admission-nine-gate-noncompensating-derivation",
                "admission-outside-support-retention",
            ),
        ),
        (
            RawReachabilityAdmissionProjector.capability_key,
            CapabilityKind.REACHABILITY_EVALUATOR,
            (ControlledMapReachabilitySpec.SCHEMA, ControlledMapAdmissionReceiptCorpus.SCHEMA),
            (ControlledMapReachabilityComparison.SCHEMA,),
            (
                "admission-no-favorable-version-selection",
                "admission-reachability-separate-from-delivery",
                "admission-shared-raw-corpus",
            ),
        ),
        (
            RawAdmissionReachabilityCategoricalProjectionEvaluator.capability_key,
            CapabilityKind.ANALYSIS,
            (
                ReceiptAdmissionSpec.SCHEMA,
                ControlledMapReachabilitySpec.SCHEMA,
                ControlledMapAdmissionReceiptCorpus.SCHEMA,
            ),
            (VerifiedAdmissionReachabilityCategoricalProjection.SCHEMA,),
            (
                "admission-categorical-equality",
                "admission-categorical-never-reconstructs-raw-receipts",
                "admission-shared-raw-corpus",
            ),
        ),
    )
    return tuple(
        sorted(
            (
                _admission_manifest(
                    key=key,
                    kind=kind,
                    input_schemas=input_schemas,
                    output_schemas=output_schemas,
                    checks=checks,
                    implementation_sha256=implementation_sha256_by_key[key],
                )
                for key, kind, input_schemas, output_schemas, checks in specifications
            ),
            key=lambda value: value.registry_id,
        )
    )


def admission_capability_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="raw-admission-receipt-capabilities",
        capabilities=admission_capability_manifests(implementation_sha256_by_key),
    )


def finite_admission_capability_keys() -> tuple[str, ...]:
    return tuple(
        sorted(
            (
                FiniteResponseSetReachabilityMethod.capability_key,
                FiniteReachabilityAdmissionReceiptProducer.capability_key,
                FiniteAdmissionProjector.capability_key,
                FiniteReachabilityAdmissionProjector.capability_key,
            )
        )
    )


def finite_admission_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    """Exact finite-method additions; unchanged gate/utility owners retain their keys."""

    if set(implementation_sha256_by_key) != set(finite_admission_capability_keys()):
        raise ValueError(
            "finite admission implementation identities must cover the exact capability family"
        )
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    specifications = (
        (
            FiniteResponseSetReachabilityMethod.capability_key,
            CapabilityKind.REACHABILITY_EVALUATOR,
            (FiniteResponseSetReachabilityRequest.SCHEMA,),
            (FiniteResponseSetReachabilityResult.SCHEMA,),
            (
                "finite-absolute-joint-set-containment",
                "finite-all-qualified-views",
                "finite-native-word-occurrence-clock-continuity",
                "finite-useful-hold-without-map-rank",
            ),
        ),
        (
            FiniteReachabilityAdmissionReceiptProducer.capability_key,
            CapabilityKind.REACHABILITY_EVALUATOR,
            (ReceiptAdmissionReceiptProductionPlan.SCHEMA, FiniteResponseSetReachabilityResult.SCHEMA),
            (FiniteReachabilityMethodReceipt.SCHEMA,),
            ("finite-exact-plan-authority-evidence-binding", "finite-raw-result-replay"),
        ),
        (
            FiniteAdmissionProjector.capability_key,
            CapabilityKind.ADMISSION_EVALUATOR,
            (FiniteCertificateAdmissionSpec.SCHEMA,),
            (AdmissionComparison.SCHEMA,),
            ("finite-complete-gate-member-version-view-roster", "finite-nine-gate-noncompensation"),
        ),
        (
            FiniteReachabilityAdmissionProjector.capability_key,
            CapabilityKind.REACHABILITY_EVALUATOR,
            (FiniteCertificateReachabilitySpec.SCHEMA,),
            (FiniteCertificateReachabilityComparison.SCHEMA,),
            (
                "finite-member-version-intersection",
                "finite-certification-refusal-not-physical-impossibility",
            ),
        ),
    )
    return tuple(
        sorted(
            (
                _admission_manifest(
                    key=key,
                    kind=kind,
                    input_schemas=inputs,
                    output_schemas=outputs,
                    checks=checks,
                    implementation_sha256=implementation_sha256_by_key[key],
                )
                for key, kind, inputs, outputs, checks in specifications
            ),
            key=lambda manifest: manifest.registry_id,
        )
    )


def finite_admission_capability_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id="finite-admission-certification-capabilities",
        capabilities=finite_admission_capability_manifests(implementation_sha256_by_key),
    )

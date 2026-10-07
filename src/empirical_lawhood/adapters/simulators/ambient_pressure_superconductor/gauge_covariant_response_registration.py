'Static least-privilege capability registrations for ambient pressure superconductor gauge covariant response.'

from __future__ import annotations

from hashlib import sha256
from typing import Any

from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .gauge_covariant_response_contracts import GAUGE_COVARIANT_RESPONSE_ADJUDICATION_SCHEMA, GAUGE_COVARIANT_RESPONSE_CONTROL_SCIENCE_FREEZE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_MATCHED_WAVE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_TRANSPORT_NOMINATION_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_ADMISSION_COMPILER_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CONDITIONAL_PROSPECTIVE_CONTROL_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_INDEPENDENT_RECURRENCE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION, GAUGE_COVARIANT_RESPONSE_CLOSEOUT_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA, GAUGE_COVARIANT_RESPONSE_DEVELOPMENT_BASIS_FREEZE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_MATERIAL_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CONFORMANCE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY, GaugeCovariantResponseCloseout, GaugeCovariantResponseConfig, GaugeCovariantResponseDevelopmentBasisFreeze, GaugeCovariantResponseSourceQualification, GaugeCovariantResponseStageResult, GaugeCovariantResponseConformanceResult, GaugeCovariantResponseMaterialInput, GaugeCovariantResponseMaterialResult, GaugeCovariantResponseResponseObservation, decode_gauge_covariant_response_config_bytes, SyntheticMaterialMethodPrerequisites
from .material_source_design_system import task_budget


GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA_SHA256 = sha256(GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA.encode()).hexdigest()
GAUGE_COVARIANT_RESPONSE_RUNTIME_ID = 'cpython-ambient-pressure-superconductor-gauge-covariant-response-staged'


class GaugeCovariantResponseConfigDecoder:
    def __init__(
        self, manifest: CapabilityManifest, config: GaugeCovariantResponseConfig, payload: bytes
    ) -> None:
        if manifest.config_schema != GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA:
            raise ValueError('gauge covariant response capability uses another config schema')
        if (
            decode_gauge_covariant_response_config_bytes(
                payload, expected_parents=SyntheticMaterialMethodPrerequisites.from_config(config)
            )
            != config
        ):
            raise ValueError('gauge covariant response registered config bytes fail strict decoding')
        self.manifest = manifest
        self.config = config
        self.payload = payload

    @property
    def provider_key(self) -> str:
        return self.manifest.capability_key

    @property
    def provider_version(self) -> str:
        return self.manifest.capability_version

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA or payload != self.payload:
            raise ValueError('gauge covariant response config materialization differs from registration')
        if (
            decode_gauge_covariant_response_config_bytes(
                payload, expected_parents=SyntheticMaterialMethodPrerequisites.from_config(self.config)
            )
            != self.config
        ):
            raise ValueError('gauge covariant response config does not satisfy its closed decoder')


def _common(config: GaugeCovariantResponseConfig, implementation_sha256: str) -> dict[str, Any]:
    return {
        "capability_version": GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION,
        "config_schema": GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA,
        "config_schema_sha256": GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA_SHA256,
        "resource_ceiling": task_budget(config),
        "deterministic": True,
        "seed_required": False,
        "language_id": "python",
        "runtime_id": GAUGE_COVARIANT_RESPONSE_RUNTIME_ID,
        "requires_clean_commit": True,
        "requires_active_mount": True,
        "requires_network": False,
        "implementation_sha256": implementation_sha256,
    }


def _permissions(
    *, development: bool = False, reveal: bool = False
) -> tuple[CapabilityPermission, ...]:
    values = {
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    }
    if development:
        values.add(CapabilityPermission.READ_DEVELOPMENT)
    if reveal:
        values.update(
            {
                CapabilityPermission.READ_SEALED_OUTCOMES,
                CapabilityPermission.REVEAL_OUTCOMES,
            }
        )
    return tuple(sorted(values))


def gauge_covariant_response_development_registry(
    *, config: GaugeCovariantResponseConfig, implementation_sha256: str
) -> CapabilityRegistry:
    common = _common(config, implementation_sha256)
    source = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY,
        kind=CapabilityKind.NUMERICAL_QUALIFIER,
        input_schema_ids=tuple(
            sorted((GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA, GaugeCovariantResponseSourceQualification.SCHEMA))
        ),
        output_schema_ids=(GaugeCovariantResponseSourceQualification.SCHEMA,),
        permissions=_permissions(),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        conformance_check_ids=(
            'exact-gauge-covariant-response-source-digests-and-archive-bounds',
            "no-material-target-values-projected",
            "unlicensed-synthesis-corpus-nonpromoting",
        ),
        **common,
    )
    gauge_covariant_response = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_CONFORMANCE_CAPABILITY_KEY,
        kind=CapabilityKind.FALSIFIER,
        input_schema_ids=(GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA,),
        output_schema_ids=tuple(
            sorted((GaugeCovariantResponseConformanceResult.SCHEMA, GaugeCovariantResponseResponseObservation.SCHEMA))
        ),
        permissions=_permissions(),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        conformance_check_ids=(
            "finite-temperature-finite-q-signed-current",
            "normal-diamagnetic-paramagnetic-cancellation",
            "pure-gauge-covariance",
            "truth-fixture-noncompensating-intersection",
        ),
        **common,
    )
    design = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_DEVELOPMENT_BASIS_FREEZE_CAPABILITY_KEY,
        kind=CapabilityKind.TRANSFORM,
        input_schema_ids=tuple(
            sorted(
                (
                    GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA,
                    GaugeCovariantResponseSourceQualification.SCHEMA,
                    GaugeCovariantResponseConformanceResult.SCHEMA,
                )
            )
        ),
        output_schema_ids=(GaugeCovariantResponseDevelopmentBasisFreeze.SCHEMA,),
        permissions=_permissions(),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        conformance_check_ids=(
            'material-source-design-source-design-audit-imported-by-fingerprint',
            "full-development-and-conditional-protocols-frozen",
            'strict-gauge-covariant-response-provider-and-computational-admission-slots-frozen',
            "target-contact-count-zero",
        ),
        **common,
    )
    material_control = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_CONTROL_SCIENCE_FREEZE_CAPABILITY_KEY,
        kind=CapabilityKind.NUMERICAL_QUALIFIER,
        input_schema_ids=tuple(
            sorted(
                (
                    GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA,
                    GaugeCovariantResponseDevelopmentBasisFreeze.SCHEMA,
                    GaugeCovariantResponseSourceQualification.SCHEMA,
                    GaugeCovariantResponseConformanceResult.SCHEMA,
                )
            )
        ),
        output_schema_ids=(GaugeCovariantResponseStageResult.SCHEMA,),
        permissions=_permissions(development=True),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        conformance_check_ids=(
            'excluded-solver-control-corrective-solver-control-pb-scf-control-receipt-bound',
            "material-control-operands-noncompensating",
            'resource-first-source-stop-before-development-atlas',
        ),
        **common,
    )
    response_guided_waves = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_MATCHED_WAVE_CAPABILITY_KEY,
        kind=CapabilityKind.ANALYSIS,
        input_schema_ids=tuple(sorted((GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA, GaugeCovariantResponseStageResult.SCHEMA))),
        output_schema_ids=(GaugeCovariantResponseStageResult.SCHEMA,),
        permissions=_permissions(development=True),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        conformance_check_ids=(
            "frozen-wave-slot",
            "policy-history-separation",
            "typed-upstream-stop-nonattempt",
        ),
        **common,
    )
    transport_nomination = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_TRANSPORT_NOMINATION_CAPABILITY_KEY,
        kind=CapabilityKind.TRANSPORT_TESTER,
        input_schema_ids=tuple(sorted((GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA, GaugeCovariantResponseStageResult.SCHEMA))),
        output_schema_ids=(GaugeCovariantResponseStageResult.SCHEMA,),
        permissions=_permissions(development=True),
        maximum_evidence_ceiling=EvidenceCeiling.LOCAL_LAW,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        conformance_check_ids=(
            'maximum-two-gauge-covariant-response-nominations',
            'no-admission-before-strict-gauge-covariant-response',
            "typed-upstream-stop-nonattempt",
        ),
        **common,
    )
    computational_admission = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_ADMISSION_COMPILER_CAPABILITY_KEY,
        kind=CapabilityKind.ADMISSION_EVALUATOR,
        input_schema_ids=tuple(
            sorted(
                (
                    GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA,
                    GaugeCovariantResponseDevelopmentBasisFreeze.SCHEMA,
                    GaugeCovariantResponseStageResult.SCHEMA,
                )
            )
        ),
        output_schema_ids=(GaugeCovariantResponseStageResult.SCHEMA,),
        permissions=_permissions(development=True),
        maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
        maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
        conformance_check_ids=(
            "complete-sixteen-to-nine-gate-intersection",
            "current-controller-compiler-only",
            'strict-gauge-covariant-response-required-before-admission',
            "typed-upstream-stop-nonattempt",
        ),
        **common,
    )
    closeout = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_CLOSEOUT_CAPABILITY_KEY,
        kind=CapabilityKind.EVALUATOR,
        input_schema_ids=tuple(
            sorted(
                (
                    GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA,
                    GaugeCovariantResponseDevelopmentBasisFreeze.SCHEMA,
                    GaugeCovariantResponseStageResult.SCHEMA,
                )
            )
        ),
        output_schema_ids=tuple(sorted((GAUGE_COVARIANT_RESPONSE_ADJUDICATION_SCHEMA, GaugeCovariantResponseCloseout.SCHEMA))),
        permissions=_permissions(reveal=True),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        conformance_check_ids=(
            "orthogonal-terminal-axes-retained",
            "receipted-downstream-nonattempts",
            'target-not-found-not-inferred-without-development-atlas-contact',
        ),
        **common,
    )
    return CapabilityRegistry(
        registry_id='capabilities.ambient-pressure-superconductor-gauge-covariant-response-development',
        capabilities=tuple(
            sorted(
                (source, gauge_covariant_response, design, material_control, response_guided_waves, transport_nomination, computational_admission, closeout),
                key=lambda value: value.registry_id,
            )
        ),
    )


def gauge_covariant_response_conditional_registry(
    *, config: GaugeCovariantResponseConfig, implementation_sha256: str
) -> CapabilityRegistry:
    common = _common(config, implementation_sha256)
    material_gauge_covariant_response = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_MATERIAL_CAPABILITY_KEY,
        kind=CapabilityKind.SIMULATOR,
        input_schema_ids=tuple(sorted((GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA, GaugeCovariantResponseMaterialInput.SCHEMA))),
        output_schema_ids=(GaugeCovariantResponseMaterialResult.SCHEMA,),
        permissions=_permissions(),
        maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
        maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
        conformance_check_ids=(
            "explicit-300k-material-pairing-state",
            "material-wannier-and-native-unit-compatibility",
            "single-band-closure-and-strong-coupling-validity",
            'strict-gauge-covariant-response-does-not-emit-admission',
        ),
        **common,
    )
    prospective_controller_validation = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_CONDITIONAL_PROSPECTIVE_CONTROL_CAPABILITY_KEY,
        kind=CapabilityKind.SIMULATOR,
        input_schema_ids=tuple(sorted((GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA, GaugeCovariantResponseStageResult.SCHEMA))),
        output_schema_ids=(GaugeCovariantResponseStageResult.SCHEMA,),
        permissions=_permissions(),
        maximum_evidence_ceiling=EvidenceCeiling.CONTROLLER_USE,
        maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
        conformance_check_ids=(
            "fresh-held-family-preparation",
            "parent-receipt-bound-action",
            'strict-gauge-covariant-response-and-admission-recomputed',
        ),
        **common,
    )
    independent_recurrence = CapabilityManifest(
        capability_key=GAUGE_COVARIANT_RESPONSE_INDEPENDENT_RECURRENCE_CAPABILITY_KEY,
        kind=CapabilityKind.ADMISSION_EVALUATOR,
        input_schema_ids=tuple(sorted((GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA, GaugeCovariantResponseStageResult.SCHEMA))),
        output_schema_ids=(GaugeCovariantResponseStageResult.SCHEMA,),
        permissions=_permissions(),
        maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
        maximum_outcome_access=OutcomeAccess.EVALUATION_SEALED,
        conformance_check_ids=(
            'independent-pairing-gauge-covariant-response-and-formation-recurrence',
            "no-primary-view-result-reuse",
            "uncertainty-intersection-not-averaging",
        ),
        **common,
    )
    return CapabilityRegistry(
        registry_id='capabilities.ambient-pressure-superconductor-gauge-covariant-response-conditional-controller-use-recurrence',
        capabilities=tuple(
            sorted((material_gauge_covariant_response, prospective_controller_validation, independent_recurrence), key=lambda value: value.registry_id)
        ),
    )


def gauge_covariant_response_config_decoders(
    *, registries: tuple[CapabilityRegistry, ...], config: GaugeCovariantResponseConfig, payload: bytes
) -> tuple[GaugeCovariantResponseConfigDecoder, ...]:
    values = tuple(
        GaugeCovariantResponseConfigDecoder(manifest, config, payload)
        for registry in registries
        for manifest in registry.capabilities
    )
    return tuple(
        sorted(values, key=lambda value: (value.provider_key, value.provider_version))
    )


__all__ = [
    'GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA_SHA256',
    'GAUGE_COVARIANT_RESPONSE_RUNTIME_ID',
    'GaugeCovariantResponseConfigDecoder',
    'gauge_covariant_response_conditional_registry',
    'gauge_covariant_response_config_decoders',
    'gauge_covariant_response_development_registry',
]

"""Static capability closure for FreeGSNKE magnetic-response science."""

from __future__ import annotations

from hashlib import sha256
from typing import Final

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .contracts import FreeGsnkeProcessRequest, FreeGsnkeSavedPreparation, FreeGsnkeSourceBinding, FreeGsnkeTargetProcessResponse


FREEGSNKE_CAPABILITY_VERSION: Final = "1.0.0"
FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY: Final = "freegsnke.generate-development"
FREEGSNKE_EVALUATION_CAPABILITY_KEY: Final = "freegsnke.generate-evaluation"
FREEGSNKE_FREEZE_CAPABILITY_KEY: Final = "freegsnke.freeze-development"
FREEGSNKE_EVALUATOR_CAPABILITY_KEY: Final = "freegsnke.evaluate-response-law"
FREEGSNKE_DEVELOPMENT_FREEZE_SCHEMA: Final = (
    'empirical-lawhood/simulators/freegsnke/development-freeze'
)
FREEGSNKE_SCIENTIFIC_RESULT_SCHEMA: Final = (
    'empirical-lawhood/simulators/freegsnke/scientific-result'
)


def _generation_resources() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=4 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=12 * 60 * 60,
        source_scan_bytes=2 * 1024**3,
        output_bytes=256 * 1024**2,
    )


def build_freegsnke_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    """Return four static roles with sealed evaluation and evaluator-only reveal."""

    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    common_permissions = (
        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
    )
    generation = tuple(
        CapabilityManifest(
            capability_key=key,
            capability_version=FREEGSNKE_CAPABILITY_VERSION,
            kind=CapabilityKind.SIMULATOR,
            config_schema=FreeGsnkeProcessRequest.SCHEMA,
            config_schema_sha256=sha256(FreeGsnkeProcessRequest.SCHEMA.encode()).hexdigest(),
            input_schema_ids=tuple(
                sorted(
                    (
                        FreeGsnkeProcessRequest.SCHEMA,
                        FreeGsnkeSavedPreparation.SCHEMA,
                        FreeGsnkeSourceBinding.SCHEMA,
                    )
                )
            ),
            output_schema_ids=(FreeGsnkeTargetProcessResponse.SCHEMA,),
            permissions=permissions,
            maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
            maximum_outcome_access=access,
            resource_ceiling=_generation_resources(),
            deterministic=True,
            seed_required=False,
            language_id="python",
            runtime_id="freegsnke-3.0.1-freegs4e-0.13.1-cpu-float64",
            requires_clean_commit=True,
            requires_active_mount=True,
            requires_network=False,
            conformance_check_ids=(
                "action-stages-and-clocks-distinct",
                "branch-nested-under-preparation",
                "causal-prefix-precedes-request",
                "dynamic-gs-nonconvergence-retained",
                "exact-source-runtime-machine-binding",
                "fixed-two-port-native-voltage-chart",
                "linearization-domain-departure-retained",
                "native-receiver-units-preserved",
                "passive-and-plasma-currents-realized",
            ),
            implementation_sha256=implementation_sha256,
        )
        for key, access, permissions in (
            (
                FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY,
                OutcomeAccess.DEVELOPMENT_VISIBLE,
                common_permissions,
            ),
            (
                FREEGSNKE_EVALUATION_CAPABILITY_KEY,
                OutcomeAccess.EVALUATION_SEALED,
                common_permissions,
            ),
        )
    )
    freeze = CapabilityManifest(
        capability_key=FREEGSNKE_FREEZE_CAPABILITY_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        kind=CapabilityKind.TRANSFORM,
        config_schema=FreeGsnkeProcessRequest.SCHEMA,
        config_schema_sha256=sha256(FreeGsnkeProcessRequest.SCHEMA.encode()).hexdigest(),
        input_schema_ids=tuple(
            sorted(
                (
                    FreeGsnkeProcessRequest.SCHEMA,
                    FreeGsnkeTargetProcessResponse.SCHEMA,
                )
            )
        ),
        output_schema_ids=(FREEGSNKE_DEVELOPMENT_FREEZE_SCHEMA,),
        permissions=tuple(sorted((CapabilityPermission.READ_DEVELOPMENT, *common_permissions))),
        maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resource_ceiling=_generation_resources(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-freegsnke-source-continuation-science",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "development-only-scaling-support-and-ridge",
            "evaluation-cells-and-outcomes-sealed",
            "independent-preparation-partition-frozen",
        ),
        implementation_sha256=implementation_sha256,
    )
    evaluator = CapabilityManifest(
        capability_key=FREEGSNKE_EVALUATOR_CAPABILITY_KEY,
        capability_version=FREEGSNKE_CAPABILITY_VERSION,
        kind=CapabilityKind.EVALUATOR,
        config_schema=FreeGsnkeProcessRequest.SCHEMA,
        config_schema_sha256=sha256(FreeGsnkeProcessRequest.SCHEMA.encode()).hexdigest(),
        input_schema_ids=tuple(
            sorted(
                (
                    FREEGSNKE_DEVELOPMENT_FREEZE_SCHEMA,
                    FreeGsnkeProcessRequest.SCHEMA,
                    FreeGsnkeTargetProcessResponse.SCHEMA,
                )
            )
        ),
        output_schema_ids=(FREEGSNKE_SCIENTIFIC_RESULT_SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_DEVELOPMENT,
                    *common_permissions,
                    CapabilityPermission.READ_SEALED_OUTCOMES,
                    CapabilityPermission.REVEAL_OUTCOMES,
                )
            )
        ),
        maximum_evidence_ceiling=EvidenceCeiling.ADMISSION,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        resource_ceiling=_generation_resources(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-freegsnke-source-continuation-science",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "complete-signed-cells-only",
            "generic-odd-relational-jacobian-composed",
            "receipt-dependent-evaluator-reveal",
            "receiver-families-reported-separately",
            "zero-half-dose-composition-and-numerical-falsifiers",
        ),
        implementation_sha256=implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id="freegsnke-magnetic-response-registry",
        capabilities=tuple(
            sorted((*generation, freeze, evaluator), key=lambda value: value.registry_id)
        ),
    )


__all__ = [
    "FREEGSNKE_CAPABILITY_VERSION",
    "FREEGSNKE_DEVELOPMENT_CAPABILITY_KEY",
    "FREEGSNKE_DEVELOPMENT_FREEZE_SCHEMA",
    "FREEGSNKE_EVALUATION_CAPABILITY_KEY",
    "FREEGSNKE_EVALUATOR_CAPABILITY_KEY",
    "FREEGSNKE_FREEZE_CAPABILITY_KEY",
    "FREEGSNKE_SCIENTIFIC_RESULT_SCHEMA",
    "build_freegsnke_registry",
]

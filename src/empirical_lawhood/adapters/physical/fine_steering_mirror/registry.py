"""Fingerprint-bound FSM analysis and held-out evaluation capabilities."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .contracts import FineSteeringMirrorBlockResponse, FineSteeringMirrorDevelopmentModel, FineSteeringMirrorProtocolSpec, FineSteeringMirrorVerticalSliceResult

FSM_DEVELOPMENT_CAPABILITY = "physical.fsm.development-analysis"
FSM_EVALUATOR_CAPABILITY = "physical.fsm.sealed-evaluator"


def fine_steering_mirror_capability_keys() -> tuple[str, ...]:
    return (FSM_DEVELOPMENT_CAPABILITY, FSM_EVALUATOR_CAPABILITY)


def _schema_sha256(schema: str) -> str:
    return hashlib.sha256(schema.encode("utf-8")).hexdigest()


def _resources() -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=4,
        memory_bytes=4_000_000_000,
        gpu_devices=0,
        wall_time_seconds=3_600,
        source_scan_bytes=30_000_000,
        output_bytes=10_000_000,
    )


def fine_steering_mirror_capability_manifests(
    implementation_sha256_by_key: Mapping[str, str],
) -> tuple[CapabilityManifest, ...]:
    """Register development and reveal capabilities with disjoint permissions."""

    if set(implementation_sha256_by_key) != set(fine_steering_mirror_capability_keys()):
        raise ValueError("R10 FSM implementation identities must cover the exact registry")
    for key, digest in implementation_sha256_by_key.items():
        validate_sha256(digest, field_name=f"implementation_sha256_by_key[{key}]")
    development = CapabilityManifest(
        capability_key=FSM_DEVELOPMENT_CAPABILITY,
        capability_version="1.0.0",
        kind=CapabilityKind.ANALYSIS,
        config_schema=FineSteeringMirrorProtocolSpec.SCHEMA,
        config_schema_sha256=_schema_sha256(FineSteeringMirrorProtocolSpec.SCHEMA),
        input_schema_ids=('empirical-lawhood/source/fine-steering-mirror-npy',),
        output_schema_ids=tuple(sorted((FineSteeringMirrorBlockResponse.SCHEMA, FineSteeringMirrorDevelopmentModel.SCHEMA))),
        permissions=(
            CapabilityPermission.READ_DEVELOPMENT,
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.RESPONSE,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resource_ceiling=_resources(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            'fine-steering-mirror-development-only-source-scope',
            'fine-steering-mirror-independent-block-semantics',
            'fine-steering-mirror-safe-numpy-observation',
        ),
        implementation_sha256=implementation_sha256_by_key[FSM_DEVELOPMENT_CAPABILITY],
    )
    evaluator = CapabilityManifest(
        capability_key=FSM_EVALUATOR_CAPABILITY,
        capability_version="1.0.0",
        kind=CapabilityKind.EVALUATOR,
        config_schema=FineSteeringMirrorProtocolSpec.SCHEMA,
        config_schema_sha256=_schema_sha256(FineSteeringMirrorProtocolSpec.SCHEMA),
        input_schema_ids=tuple(
            sorted(
                (
                    FineSteeringMirrorBlockResponse.SCHEMA,
                    FineSteeringMirrorDevelopmentModel.SCHEMA,
                    'empirical-lawhood/source/fine-steering-mirror-npy',
                )
            )
        ),
        output_schema_ids=(FineSteeringMirrorVerticalSliceResult.SCHEMA,),
        permissions=(
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.RESPONSE,
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        resource_ceiling=_resources(),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-3.11",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            'fine-steering-mirror-clean-freeze-before-reveal',
            'fine-steering-mirror-explicit-law-qualification-atlas-gap',
            'fine-steering-mirror-no-controller-without-admission',
            'fine-steering-mirror-sealed-receiver-boundary',
        ),
        implementation_sha256=implementation_sha256_by_key[FSM_EVALUATOR_CAPABILITY],
    )
    return tuple(sorted((development, evaluator), key=lambda item: item.registry_id))


def fine_steering_mirror_capability_registry(
    implementation_sha256_by_key: Mapping[str, str],
) -> CapabilityRegistry:
    return CapabilityRegistry(
        registry_id='fine-steering-mirror-empirical-capabilities',
        capabilities=fine_steering_mirror_capability_manifests(implementation_sha256_by_key),
    )

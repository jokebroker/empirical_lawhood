"""Static least-privilege capability registrations for uniform electron gas transverse receiver screen."""

from __future__ import annotations

from hashlib import sha256
from typing import Any

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .contracts import ADMISSION_CAPABILITY_KEY, ADJUDICATION_SCHEMA, CAPABILITY_VERSION, CONFIG_SCHEMA, EVALUATOR_CAPABILITY_KEY, METHOD_CAPABILITY_KEY, MethodConformanceResult, SOURCE_CAPABILITY_KEY, SourceStopResult, UniformElectronGasTransverseScreenConfig, TruthInputBatch, TruthLawBatch, TruthOracleBatch, TruthScreenBatch, decode_config_bytes


CONFIG_SCHEMA_SHA256 = sha256(CONFIG_SCHEMA.encode("utf-8")).hexdigest()
RUNTIME_ID = "cpython-uniform-electron-gas-transverse-screen"


class UniformElectronGasTransverseScreenConfigDecoder:
    """Bind a capability to one exact frozen config materialization."""

    def __init__(self, manifest: CapabilityManifest, config: UniformElectronGasTransverseScreenConfig, payload: bytes) -> None:
        if manifest.config_schema != CONFIG_SCHEMA:
            raise ValueError("uniform electron gas capability uses another config schema")
        if decode_config_bytes(payload) != config:
            raise ValueError("uniform electron gas registered config bytes fail strict decoding")
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
        if expected_schema != CONFIG_SCHEMA or payload != self.payload:
            raise ValueError("uniform electron gas config materialization differs from registration")
        if decode_config_bytes(payload) != self.config:
            raise ValueError("uniform electron gas config does not satisfy its closed decoder")


def _resources(config: UniformElectronGasTransverseScreenConfig) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=int(config.resource("cpu_cores")),
        memory_bytes=int(config.resource("memory_bytes")),
        gpu_devices=int(config.resource("gpu_devices")),
        wall_time_seconds=int(config.resource("wall_time_seconds")),
        source_scan_bytes=int(config.resource("maximum_all_inputs_bytes")),
        output_bytes=int(config.resource("output_bytes")),
    )


def uniform_electron_gas_transverse_screen_registry(*, config: UniformElectronGasTransverseScreenConfig, implementation_sha256: str) -> CapabilityRegistry:
    resources = _resources(config)
    common: dict[str, Any] = {
        "capability_version": CAPABILITY_VERSION,
        "config_schema": CONFIG_SCHEMA,
        "config_schema_sha256": CONFIG_SCHEMA_SHA256,
        "resource_ceiling": resources,
        "deterministic": True,
        "seed_required": False,
        "language_id": "python",
        "runtime_id": RUNTIME_ID,
        "requires_clean_commit": True,
        "requires_active_mount": True,
        "requires_network": False,
        "implementation_sha256": implementation_sha256,
        "maximum_evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE,
    }
    source = CapabilityManifest(
        capability_key=SOURCE_CAPABILITY_KEY,
        kind=CapabilityKind.SIMULATOR,
        input_schema_ids=(CONFIG_SCHEMA,),
        output_schema_ids=tuple(sorted((TruthInputBatch.SCHEMA, TruthOracleBatch.SCHEMA))),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
        conformance_check_ids=(
            "bounded-truth-known-source",
            "privileged-oracle-separate",
            "target-contact-count-zero",
            "truth-known-fixed-roster",
        ),
        **common,
    )
    method = CapabilityManifest(
        capability_key=METHOD_CAPABILITY_KEY,
        kind=CapabilityKind.LAW_IDENTIFIER,
        input_schema_ids=(TruthInputBatch.SCHEMA,),
        output_schema_ids=(TruthLawBatch.SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_DEVELOPMENT,
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        conformance_check_ids=(
            "amplitude-and-q-predeclared",
            "normal-and-ward-noncompensating",
            "realized-si-vector-potential-derivative",
            "view-local-no-pooling",
        ),
        **common,
    )
    admission = CapabilityManifest(
        capability_key=ADMISSION_CAPABILITY_KEY,
        kind=CapabilityKind.ADMISSION_EVALUATOR,
        input_schema_ids=tuple(sorted((TruthInputBatch.SCHEMA, TruthLawBatch.SCHEMA))),
        output_schema_ids=(TruthScreenBatch.SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_DEVELOPMENT,
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        conformance_check_ids=tuple(
            sorted(
                (
                    "closure-lane-never-enters-admission",
                    "nine-gate-intersection",
                    "nonpositive-bound-no-lambda",
                    "nonpass-mandatory-hold",
                )
            )
        ),
        **common,
    )
    evaluator = CapabilityManifest(
        capability_key=EVALUATOR_CAPABILITY_KEY,
        kind=CapabilityKind.EVALUATOR,
        input_schema_ids=tuple(sorted((TruthOracleBatch.SCHEMA, TruthScreenBatch.SCHEMA))),
        output_schema_ids=tuple(
            sorted(
                (
                    ADJUDICATION_SCHEMA,
                    MethodConformanceResult.SCHEMA,
                    SourceStopResult.SCHEMA,
                )
            )
        ),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.READ_SEALED_OUTCOMES,
                    CapabilityPermission.REVEAL_OUTCOMES,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        conformance_check_ids=(
            "oracle-evaluator-only",
            "source-stop-nonattempts-exact",
            "truth-family-classification-exact",
            "truth-known-nonpromotable-ceiling",
        ),
        **common,
    )
    return CapabilityRegistry(
        registry_id="capabilities.uniform-electron-gas-transverse-screen",
        capabilities=tuple(
            sorted((source, method, admission, evaluator), key=lambda value: value.registry_id)
        ),
    )


def config_decoders(
    *, registry: CapabilityRegistry, config: UniformElectronGasTransverseScreenConfig, payload: bytes
) -> tuple[UniformElectronGasTransverseScreenConfigDecoder, ...]:
    return tuple(UniformElectronGasTransverseScreenConfigDecoder(manifest, config, payload) for manifest in registry.capabilities)


__all__ = [
    "CONFIG_SCHEMA_SHA256",
    "RUNTIME_ID",
    "UniformElectronGasTransverseScreenConfigDecoder",
    "config_decoders",
    "uniform_electron_gas_transverse_screen_registry",
]

'Static least-privilege registrations for the fresh ambient pressure superconductor material control act.'

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

from .gauge_covariant_response_contracts import GaugeCovariantResponseConformanceResult, GaugeCovariantResponseResponseObservation
from .material_control_contracts import MATERIAL_CONTROL_ADJUDICATION_SCHEMA, MATERIAL_CONTROL_CAPABILITY_VERSION, MATERIAL_CONTROL_CLOSEOUT_CAPABILITY_KEY, MATERIAL_CONTROL_CONFIG_SCHEMA, MATERIAL_CONTROL_METHOD_CAPABILITY_KEY, MATERIAL_CONTROL_PANEL_CAPABILITY_KEY, MATERIAL_CONTROL_SOURCE_CAPABILITY_KEY, MATERIAL_CONTROL_WORKFLOW_CAPABILITY_KEY, MATERIAL_CONTROL_SEARCH_WORLD_CAPABILITY_KEY, MaterialControlCloseout, MaterialControlConfig, MaterialControlControlBundleQualification, MaterialControlControlObservation, MaterialControlControlPanel, MaterialControlStageResult, MaterialControlTutorialReproduction, MaterialControlTruthWorldConformance, MaterialControlTruthWorldResult, MultibandStrongCouplingMaterialCompatibility, MultibandStrongCouplingMaterialViewResult, MultibandStrongCouplingBridgeQualification, decode_material_control_config_bytes
from .material_control_raw import RAW_ARCHIVE_SCHEMA


MATERIAL_CONTROL_CONFIG_SCHEMA_SHA256 = sha256(MATERIAL_CONTROL_CONFIG_SCHEMA.encode()).hexdigest()
MATERIAL_CONTROL_RUNTIME_ID = 'cpython-ambient-pressure-superconductor-material-control-qe76-epw61'


def material_control_task_budget(config: MaterialControlConfig) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=int(config.resource("cpu_cores")),
        memory_bytes=int(config.resource("memory_bytes")),
        gpu_devices=int(config.resource("gpu_devices")),
        wall_time_seconds=int(config.resource("wall_time_seconds")),
        source_scan_bytes=int(config.resource("scratch_bytes")),
        output_bytes=int(config.resource("output_bytes")),
    )


def material_control_metadata_task_budget(config: MaterialControlConfig) -> ResourceBudget:
    """Bound compact typed tasks without reserving a QE-sized scratch ceiling."""

    maximum_bytes = min(int(config.resource("output_bytes")), 64 * 1024**2)
    return ResourceBudget(
        cpu_cores=int(config.resource("cpu_cores")),
        memory_bytes=int(config.resource("memory_bytes")),
        gpu_devices=int(config.resource("gpu_devices")),
        wall_time_seconds=int(config.resource("wall_time_seconds")),
        source_scan_bytes=int(config.resource("scratch_bytes")),
        output_bytes=maximum_bytes,
    )


class MaterialControlConfigDecoder:
    def __init__(
        self, manifest: CapabilityManifest, config: MaterialControlConfig, payload: bytes
    ) -> None:
        if manifest.config_schema != MATERIAL_CONTROL_CONFIG_SCHEMA:
            raise ValueError('material control capability uses another config schema')
        if (
            decode_material_control_config_bytes(payload, expected_parents=config.predecessors)
            != config
        ):
            raise ValueError('material control registered config bytes fail strict decoding')
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
        if expected_schema != MATERIAL_CONTROL_CONFIG_SCHEMA or payload != self.payload:
            raise ValueError('material control config materialization differs from registration')
        if (
            decode_material_control_config_bytes(payload, expected_parents=self.config.predecessors)
            != self.config
        ):
            raise ValueError('material control config does not satisfy its closed decoder')


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


def _common(config: MaterialControlConfig, implementation_sha256: str) -> dict[str, Any]:
    return {
        "capability_version": MATERIAL_CONTROL_CAPABILITY_VERSION,
        "config_schema": MATERIAL_CONTROL_CONFIG_SCHEMA,
        "config_schema_sha256": MATERIAL_CONTROL_CONFIG_SCHEMA_SHA256,
        "resource_ceiling": material_control_task_budget(config),
        "deterministic": True,
        "seed_required": False,
        "language_id": "python",
        "runtime_id": MATERIAL_CONTROL_RUNTIME_ID,
        "requires_clean_commit": True,
        "requires_active_mount": True,
        "requires_network": False,
        "implementation_sha256": implementation_sha256,
        "maximum_evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE,
    }


def material_control_registry(
    *, config: MaterialControlConfig, implementation_sha256: str
) -> CapabilityRegistry:
    common = _common(config, implementation_sha256)
    source = CapabilityManifest(
        capability_key=MATERIAL_CONTROL_SOURCE_CAPABILITY_KEY,
        kind=CapabilityKind.NUMERICAL_QUALIFIER,
        input_schema_ids=tuple(
            sorted((MATERIAL_CONTROL_CONFIG_SCHEMA, MaterialControlControlBundleQualification.SCHEMA))
        ),
        output_schema_ids=(MaterialControlControlBundleQualification.SCHEMA,),
        permissions=_permissions(),
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        conformance_check_ids=(
            "exact-public-source-environment-and-binary-digests",
            "safe-bounded-archive-inventories",
            "target-outcomes-not-projected",
        ),
        **common,
    )
    method = CapabilityManifest(
        capability_key=MATERIAL_CONTROL_METHOD_CAPABILITY_KEY,
        kind=CapabilityKind.FALSIFIER,
        input_schema_ids=(MATERIAL_CONTROL_CONFIG_SCHEMA,),
        output_schema_ids=tuple(
            sorted(
                (
                    MultibandStrongCouplingBridgeQualification.SCHEMA,
                    GaugeCovariantResponseConformanceResult.SCHEMA,
                    GaugeCovariantResponseResponseObservation.SCHEMA,
                )
            )
        ),
        permissions=_permissions(),
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        conformance_check_ids=tuple(
            sorted(
                (
                    'gauge-covariant-response-nine-fixture-gauge-covariant-response-exact-recurrence',
                    "multiband-strong-coupling-conditional-method-fixture",
                    "material-promotion-remains-disabled",
                )
            )
        ),
        **common,
    )
    workflow = CapabilityManifest(
        capability_key=MATERIAL_CONTROL_WORKFLOW_CAPABILITY_KEY,
        kind=CapabilityKind.SIMULATOR,
        input_schema_ids=tuple(
            sorted(
                (
                    MATERIAL_CONTROL_CONFIG_SCHEMA,
                    MaterialControlControlBundleQualification.SCHEMA,
                    MultibandStrongCouplingBridgeQualification.SCHEMA,
                )
            )
        ),
        output_schema_ids=tuple(
            sorted(
                (
                    MaterialControlControlObservation.SCHEMA,
                    MaterialControlTutorialReproduction.SCHEMA,
                    MultibandStrongCouplingMaterialCompatibility.SCHEMA,
                    MultibandStrongCouplingMaterialViewResult.SCHEMA,
                    RAW_ARCHIVE_SCHEMA,
                )
            )
        ),
        permissions=_permissions(development=True),
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        conformance_check_ids=tuple(
            sorted(
                (
                    "closed-profile-no-network-qe-epw-execution",
                    "requested-accepted-applied-realized-preparation-bound",
                    "raw-output-published-through-audited-hdf5-envelope",
                    "material-validity-failure-is-typed",
                )
            )
        ),
        **common,
    )
    panel = CapabilityManifest(
        capability_key=MATERIAL_CONTROL_PANEL_CAPABILITY_KEY,
        kind=CapabilityKind.TRANSFORM,
        input_schema_ids=tuple(
            sorted((MATERIAL_CONTROL_CONFIG_SCHEMA, MaterialControlControlObservation.SCHEMA))
        ),
        output_schema_ids=(MaterialControlControlPanel.SCHEMA,),
        permissions=_permissions(development=True),
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        conformance_check_ids=tuple(
            sorted(
                (
                    "one-physical-material-unit-two-nested-views",
                    "noncompensating-class-recovery",
                    'positive-gauge-covariant-response-common-interval-required',
                )
            )
        ),
        **common,
    )
    constructive_search = CapabilityManifest(
        capability_key=MATERIAL_CONTROL_SEARCH_WORLD_CAPABILITY_KEY,
        kind=CapabilityKind.FALSIFIER,
        input_schema_ids=(MATERIAL_CONTROL_CONFIG_SCHEMA,),
        output_schema_ids=tuple(
            sorted((MaterialControlTruthWorldConformance.SCHEMA, MaterialControlTruthWorldResult.SCHEMA))
        ),
        permissions=_permissions(),
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        conformance_check_ids=tuple(
            sorted(
                (
                    "six-frozen-truth-worlds",
                    "three-policy-history-separation",
                    "matched-budget-accounting-and-hold",
                    "target-contact-count-zero",
                )
            )
        ),
        **common,
    )
    closeout = CapabilityManifest(
        capability_key=MATERIAL_CONTROL_CLOSEOUT_CAPABILITY_KEY,
        kind=CapabilityKind.EVALUATOR,
        input_schema_ids=tuple(
            sorted(
                (
                    MATERIAL_CONTROL_CONFIG_SCHEMA,
                    MaterialControlControlBundleQualification.SCHEMA,
                    MaterialControlControlPanel.SCHEMA,
                    MaterialControlTutorialReproduction.SCHEMA,
                    MaterialControlTruthWorldConformance.SCHEMA,
                    MultibandStrongCouplingMaterialViewResult.SCHEMA,
                    MultibandStrongCouplingBridgeQualification.SCHEMA,
                    GaugeCovariantResponseConformanceResult.SCHEMA,
                    GaugeCovariantResponseResponseObservation.SCHEMA,
                )
            )
        ),
        output_schema_ids=tuple(
            sorted((MATERIAL_CONTROL_ADJUDICATION_SCHEMA, MaterialControlCloseout.SCHEMA, MaterialControlStageResult.SCHEMA))
        ),
        permissions=_permissions(development=True, reveal=True),
        maximum_outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
        conformance_check_ids=tuple(
            sorted(
                (
                    "five-control-panel-intersection",
                    "low-temperature-controls-not-300k-evidence",
                    'calibration-fields-derived-before-development-atlas',
                    "science-freeze-only-on-complete-intersection",
                )
            )
        ),
        **common,
    )
    return CapabilityRegistry(
        registry_id='capabilities.ambient-pressure-superconductor-material-control-control-science-freeze',
        capabilities=tuple(
            sorted(
                (source, method, workflow, panel, constructive_search, closeout),
                key=lambda value: value.registry_id,
            )
        ),
    )


def material_control_config_decoders(
    *, registry: CapabilityRegistry, config: MaterialControlConfig, payload: bytes
) -> tuple[MaterialControlConfigDecoder, ...]:
    return tuple(
        MaterialControlConfigDecoder(manifest, config, payload)
        for manifest in registry.capabilities
    )


__all__ = [
    "MATERIAL_CONTROL_CONFIG_SCHEMA_SHA256",
    "MATERIAL_CONTROL_RUNTIME_ID",
    'MaterialControlConfigDecoder',
    'material_control_config_decoders',
    'material_control_metadata_task_budget',
    'material_control_registry',
    'material_control_task_budget',
]

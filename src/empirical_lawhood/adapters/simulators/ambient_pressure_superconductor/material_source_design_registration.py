'Static least-privilege ambient pressure superconductor material source design capability registrations.'

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

from .material_source_design_contracts import MATERIAL_SOURCE_DESIGN_ADJUDICATION_SCHEMA, MATERIAL_SOURCE_DESIGN_CAPABILITY_VERSION, MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA, MATERIAL_SOURCE_DESIGN_DESIGN_CAPABILITY_KEY, MATERIAL_SOURCE_DESIGN_EVALUATOR_CAPABILITY_KEY, MATERIAL_SOURCE_DESIGN_SOURCE_CAPABILITY_KEY, MaterialSourceDesignConfig, MaterialSourceDesignDesignBasisAudit, MaterialSourceDesignResult, MaterialSourceDesignSourceQualification, ExplorationDesignFreeze, MaterialRosterFreeze, ScienceDesignFreeze, decode_material_source_design_config_bytes
from .material_source_design_system import task_budget


MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA_SHA256 = sha256(MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA.encode()).hexdigest()
MATERIAL_SOURCE_DESIGN_RUNTIME_ID = 'cpython-ambient-pressure-superconductor-material-source-design-source-design'


class MaterialSourceDesignConfigDecoder:
    def __init__(self, manifest: CapabilityManifest, config: MaterialSourceDesignConfig, payload: bytes) -> None:
        if manifest.config_schema != MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA:
            raise ValueError('material source design capability uses another config schema')
        if decode_material_source_design_config_bytes(payload) != config:
            raise ValueError('material source design registered config bytes fail strict decoding')
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
        if expected_schema != MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA or payload != self.payload:
            raise ValueError('material source design config materialization differs from registration')
        if decode_material_source_design_config_bytes(payload) != self.config:
            raise ValueError('material source design config does not satisfy its closed decoder')


def material_source_design_registry(*, config: MaterialSourceDesignConfig, implementation_sha256: str) -> CapabilityRegistry:
    common: dict[str, Any] = {
        "capability_version": MATERIAL_SOURCE_DESIGN_CAPABILITY_VERSION,
        "config_schema": MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA,
        "config_schema_sha256": MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA_SHA256,
        "resource_ceiling": task_budget(config),
        "deterministic": True,
        "seed_required": False,
        "language_id": "python",
        "runtime_id": MATERIAL_SOURCE_DESIGN_RUNTIME_ID,
        "requires_clean_commit": True,
        "requires_active_mount": True,
        "requires_network": False,
        "implementation_sha256": implementation_sha256,
        "maximum_evidence_ceiling": EvidenceCeiling.NON_PROMOTABLE,
    }
    source = CapabilityManifest(
        capability_key=MATERIAL_SOURCE_DESIGN_SOURCE_CAPABILITY_KEY,
        kind=CapabilityKind.NUMERICAL_QUALIFIER,
        input_schema_ids=tuple(sorted((MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA, MaterialSourceDesignSourceQualification.SCHEMA))),
        output_schema_ids=(MaterialSourceDesignSourceQualification.SCHEMA,),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        conformance_check_ids=(
            "archive-and-member-bounds-verified",
            "exact-source-and-pseudopotential-digests",
            "historical-target-values-not-projected",
            "reference-xi-fixture-independently-reproduced",
        ),
        **common,
    )
    design = CapabilityManifest(
        capability_key=MATERIAL_SOURCE_DESIGN_DESIGN_CAPABILITY_KEY,
        kind=CapabilityKind.TRANSFORM,
        input_schema_ids=tuple(sorted((MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA, MaterialSourceDesignSourceQualification.SCHEMA))),
        output_schema_ids=tuple(
            sorted(
                (
                    MaterialSourceDesignDesignBasisAudit.SCHEMA,
                    ExplorationDesignFreeze.SCHEMA,
                    MaterialRosterFreeze.SCHEMA,
                    ScienceDesignFreeze.SCHEMA,
                )
            )
        ),
        permissions=tuple(
            sorted(
                (
                    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                )
            )
        ),
        maximum_outcome_access=OutcomeAccess.OUTCOME_BLIND,
        conformance_check_ids=tuple(
            sorted(
                (
                    "finite-family-partition-closed",
                    "target-contact-count-zero",
                    "three-policy-six-world-design-closed",
                    "sixteen-gate-and-three-receiver-level-design-closed",
                )
            )
        ),
        **common,
    )
    evaluator = CapabilityManifest(
        capability_key=MATERIAL_SOURCE_DESIGN_EVALUATOR_CAPABILITY_KEY,
        kind=CapabilityKind.EVALUATOR,
        input_schema_ids=tuple(
            sorted(
                (
                    MaterialSourceDesignDesignBasisAudit.SCHEMA,
                    MaterialSourceDesignSourceQualification.SCHEMA,
                    ExplorationDesignFreeze.SCHEMA,
                    MaterialRosterFreeze.SCHEMA,
                    ScienceDesignFreeze.SCHEMA,
                )
            )
        ),
        output_schema_ids=tuple(sorted((MATERIAL_SOURCE_DESIGN_ADJUDICATION_SCHEMA, MaterialSourceDesignResult.SCHEMA))),
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
            "component-fingerprint-intersection",
            "design-freeze-target-contact-count-zero",
            'nonpromotable-material-source-design-ceiling',
        ),
        **common,
    )
    return CapabilityRegistry(
        registry_id='capabilities.ambient-pressure-superconductor-material-source-design-design-basis',
        capabilities=tuple(
            sorted((source, design, evaluator), key=lambda value: value.registry_id)
        ),
    )


def material_source_design_config_decoders(
    *, registry: CapabilityRegistry, config: MaterialSourceDesignConfig, payload: bytes
) -> tuple[MaterialSourceDesignConfigDecoder, ...]:
    return tuple(MaterialSourceDesignConfigDecoder(manifest, config, payload) for manifest in registry.capabilities)


__all__ = [
    "MATERIAL_SOURCE_DESIGN_CONFIG_SCHEMA_SHA256",
    "MATERIAL_SOURCE_DESIGN_RUNTIME_ID",
    'MaterialSourceDesignConfigDecoder',
    'material_source_design_config_decoders',
    'material_source_design_registry',
]

"""Static non-promotable capability for MAST-U grounding reproduction."""

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

from .contracts import MastuGroundingReuseFreeze, MastuGroundingReuseResult


MASTU_GROUNDING_REUSE_CAPABILITY_KEY: Final = "mastu.recompute-grounding-reuse"
MASTU_GROUNDING_REUSE_CAPABILITY_VERSION: Final = "1.0.0"


def build_mastu_grounding_reuse_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    capability = CapabilityManifest(
        capability_key=MASTU_GROUNDING_REUSE_CAPABILITY_KEY,
        capability_version=MASTU_GROUNDING_REUSE_CAPABILITY_VERSION,
        kind=CapabilityKind.TRANSFORM,
        config_schema=MastuGroundingReuseFreeze.SCHEMA,
        config_schema_sha256=sha256(MastuGroundingReuseFreeze.SCHEMA.encode()).hexdigest(),
        input_schema_ids=(MastuGroundingReuseFreeze.SCHEMA,),
        output_schema_ids=(MastuGroundingReuseResult.SCHEMA,),
        permissions=(
            CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
            CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
        ),
        maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
        maximum_outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
        resource_ceiling=ResourceBudget(
            cpu_cores=4,
            memory_bytes=16 * 1024**3,
            gpu_devices=0,
            wall_time_seconds=4 * 60 * 60,
            source_scan_bytes=8 * 1024**3,
            output_bytes=512 * 1024**2,
        ),
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-mastu-compatibility-reproduction",
        requires_clean_commit=True,
        requires_active_mount=True,
        requires_network=False,
        conformance_check_ids=(
            "discharge-is-independent-unit",
            "exact-held-public-source-hashes",
            "outcome-visible-reuse-not-sealed",
            "prior-result-byte-and-numeric-comparison",
            "requested-flow-not-delivered-flow",
            "zero-new-independent-units",
        ),
        implementation_sha256=implementation_sha256,
    )
    return CapabilityRegistry(
        registry_id="mastu-public-grounding-reuse-registry",
        capabilities=(capability,),
    )


__all__ = [
    "MASTU_GROUNDING_REUSE_CAPABILITY_KEY",
    "MASTU_GROUNDING_REUSE_CAPABILITY_VERSION",
    "build_mastu_grounding_reuse_registry",
]

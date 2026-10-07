"""Static fingerprinted capability catalog for the FAIR-MAST archive adapter."""

from __future__ import annotations

from hashlib import sha256

from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess
from empirical_lawhood.kernel.serialization import validate_sha256
from empirical_lawhood.runtime.capabilities import (
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)

from .contracts import (
    FairMastMaterializationReceipt,
    FairMastSourceDeclaration,
    MastCausalState,
    MastEndpointResult,
    MastEvent,
)
from .endpoints import MastEndpointRule
from .events import MastEventRule
from .features import MastFeatureRule
from .source import FairMastMaterializationRequest, FairMastOperatorConfig, FairMastPreview


def build_mast_archive_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    pure_budget = ResourceBudget(
        cpu_cores=2,
        memory_bytes=2 * 1024**3,
        gpu_devices=0,
        wall_time_seconds=600,
        source_scan_bytes=0,
        output_bytes=16 * 1024**2,
    )
    manifests = (
        _manifest(
            key="mast-archive.endpoint-reducer",
            kind=CapabilityKind.TRANSFORM,
            config_schema=MastEndpointRule.SCHEMA,
            inputs=(MastEvent.SCHEMA,),
            outputs=(MastEndpointResult.SCHEMA,),
            permissions=(CapabilityPermission.READ_DEVELOPMENT,),
            ceiling=EvidenceCeiling.RESPONSE,
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            budget=pure_budget,
            network=False,
            mount=False,
            checks=("clock-and-mapping-states-distinct", "frozen-40ms-endpoint"),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key="mast-archive.event-builder",
            kind=CapabilityKind.TRANSFORM,
            config_schema=MastEventRule.SCHEMA,
            inputs=('empirical-lawhood/physical/mast-archive/mast-action-trace',),
            outputs=(MastEvent.SCHEMA,),
            permissions=(CapabilityPermission.READ_DEVELOPMENT,),
            ceiling=EvidenceCeiling.ORDER_RELATION,
            access=OutcomeAccess.OUTCOME_BLIND,
            budget=pure_budget,
            network=False,
            mount=False,
            checks=("one-event-per-shot", "post-action-outcome-port-absent"),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key="mast-archive.fair-mast-source",
            kind=CapabilityKind.SOURCE,
            config_schema=FairMastOperatorConfig.SCHEMA,
            inputs=tuple(
                sorted(
                    (
                        FairMastMaterializationRequest.SCHEMA,
                        FairMastSourceDeclaration.SCHEMA,
                    )
                )
            ),
            outputs=tuple(sorted((FairMastMaterializationReceipt.SCHEMA, FairMastPreview.SCHEMA))),
            permissions=tuple(
                sorted(
                    (
                        CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
                        CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
                    )
                )
            ),
            ceiling=EvidenceCeiling.MEASUREMENT,
            access=OutcomeAccess.OUTCOME_BLIND,
            budget=ResourceBudget(
                cpu_cores=4,
                memory_bytes=8 * 1024**3,
                gpu_devices=0,
                wall_time_seconds=3600,
                source_scan_bytes=8 * 1024**3,
                output_bytes=8 * 1024**3,
            ),
            network=True,
            mount=True,
            checks=(
                "closed-operator-endpoint",
                "external-only-receipt-first-publication",
                "source-drift-detected",
            ),
            implementation_sha256=implementation_sha256,
        ),
        _manifest(
            key="mast-archive.feature-builder",
            kind=CapabilityKind.TRANSFORM,
            config_schema=MastFeatureRule.SCHEMA,
            inputs=tuple(
                sorted(
                    (
                        MastEvent.SCHEMA,
                        'empirical-lawhood/physical/mast-archive/mast-pre-action-field',
                    )
                )
            ),
            outputs=(MastCausalState.SCHEMA,),
            permissions=(CapabilityPermission.READ_DEVELOPMENT,),
            ceiling=EvidenceCeiling.RESPONSE,
            access=OutcomeAccess.OUTCOME_BLIND,
            budget=pure_budget,
            network=False,
            mount=False,
            checks=("d-h-r-roles-distinct", "strict-pre-t0-cutoff"),
            implementation_sha256=implementation_sha256,
        ),
    )
    return CapabilityRegistry(
        registry_id="mast-archive-flagship-registry",
        capabilities=manifests,
    )


def _manifest(
    *,
    key: str,
    kind: CapabilityKind,
    config_schema: str,
    inputs: tuple[str, ...],
    outputs: tuple[str, ...],
    permissions: tuple[CapabilityPermission, ...],
    ceiling: EvidenceCeiling,
    access: OutcomeAccess,
    budget: ResourceBudget,
    network: bool,
    mount: bool,
    checks: tuple[str, ...],
    implementation_sha256: str,
) -> CapabilityManifest:
    return CapabilityManifest(
        capability_key=key,
        capability_version="1.0.0",
        kind=kind,
        config_schema=config_schema,
        config_schema_sha256=sha256(config_schema.encode()).hexdigest(),
        input_schema_ids=inputs,
        output_schema_ids=outputs,
        permissions=permissions,
        maximum_evidence_ceiling=ceiling,
        maximum_outcome_access=access,
        resource_ceiling=budget,
        deterministic=True,
        seed_required=False,
        language_id="python",
        runtime_id="cpython-mast-archive-flagship",
        requires_clean_commit=False,
        requires_active_mount=mount,
        requires_network=network,
        conformance_check_ids=checks,
        implementation_sha256=implementation_sha256,
    )


__all__ = ["build_mast_archive_registry"]

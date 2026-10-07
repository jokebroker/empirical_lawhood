"""Exact target-owned ports for the bounded fresh reactor campaign.

SPDX-License-Identifier: MPL-2.0
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from multiprocessing.reduction import ForkingPickler

from empirical_lawhood.adapters.composition.reactor_prefix_response.authoring import ReactorAuthoringBundle
from empirical_lawhood.adapters.composition.reactor_prefix_response.custody_ports import ReactorCustodyPorts
from empirical_lawhood.adapters.methods.reactor_prefix_response.design import ReactorScienceDesign
from empirical_lawhood.adapters.methods.reactor_prefix_response.provider import FiniteChainProvider
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import ReactorPrefixConfig
from empirical_lawhood.adapters.simulators.reactor_prefix_response.provider import ReactorPrefixProvider
from empirical_lawhood.adapters.simulators.reactor_prefix_response.packaged_source import load_packaged_reactor_source
from empirical_lawhood.infrastructure.artifacts import ExternalArtifactPlane
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile
from empirical_lawhood.runtime.executable_bindings import ExecutablePlatformPort
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    ExternalInputPayload,
    capability_semantic_validation_registry,
)


@dataclass(frozen=True, slots=True)
class _PackagedReactorSourceBytes:
    """Reopen only the immutable public bytes already pinned by this composition.

    Platform ports survive successive API calls. A single-use from_bytes source
    would fail when resume reconstructs the provider after sealed acquisition.
    The enclosing ExternalInputPayload rechecks the full byte digest and limits
    on every read. This is not a replay policy for live or outcome-bearing input.
    """

    payload: bytes

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]:
        if maximum_chunk_bytes <= 0:
            raise ValueError("reactor source chunk bound must be positive")
        for offset in range(0, len(self.payload), maximum_chunk_bytes):
            yield self.payload[offset : offset + maximum_chunk_bytes]

    def close(self) -> None:
        return


def fresh_reactor_platform_ports(
    *,
    plane: ExternalArtifactPlane,
    bundle: ReactorAuthoringBundle,
    projected_execution: ProtocolExecutionPlan,
    run_id: str,
) -> tuple[ExecutablePlatformPort, ...]:
    """Resolve all four factory ports and prove runner serialization without executing."""

    registry = bundle.standard_context.base.registry
    parent = ArtifactLineageParent(
        ObjectIdentity.from_record(bundle.authoring.package_id, bundle.authoring),
        VisibilityCeiling.PROSPECTIVE,
        OutcomeAccess.OUTCOME_BLIND,
    )
    custody = ReactorCustodyPorts(
        root=plane.root,
        run_id=run_id,
        parent=parent,
        outputs=tuple(
            output for task in projected_execution.tasks for output in task.outputs
        ),
        minimum_free_bytes=plane.root.contract.minimum_free_bytes,
    )
    source = load_packaged_reactor_source()
    source_bytes = source.canonical_bytes()
    native = ExternalInputPayload.from_bytes(
        logical_artifact_id=f"{run_id}.source-bundle",
        payload_schema=source.SCHEMA,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/vnd.empirical-lawhood.canonical+json",
        payload=source_bytes,
        visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
        outcome_access=OutcomeAccess.OUTCOME_BLIND,
        parent_visibility_ceilings=(parent.visibility_ceiling,),
        lineage_parents=(parent,),
        logical_content_sha256=source.fingerprint(),
    )
    native = replace(native, source=_PackagedReactorSourceBytes(source_bytes))
    native_config = next(
        p for p in bundle.payloads if isinstance(p, ReactorPrefixConfig)
    )
    design = next(p for p in bundle.payloads if isinstance(p, ReactorScienceDesign))
    from empirical_lawhood.adapters.methods.reactor_prefix_response.assigned.executable_binding import AssignedFiniteChainProvider
    from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned import ReactorAssignedPrefixConfig
    from empirical_lawhood.adapters.simulators.reactor_prefix_response.assigned_binding.executable_binding import ReactorAssignedPrefixProvider

    assigned = type(native_config) is ReactorAssignedPrefixConfig
    source_type = (
        ReactorAssignedPrefixProvider if assigned else ReactorPrefixProvider
    )
    method_type = AssignedFiniteChainProvider if assigned else FiniteChainProvider
    native_provider = source_type(registry, native_config, native)
    method_provider = method_type(registry, design, custody, custody, custody)
    semantics = capability_semantic_validation_registry(
        registry_id=f"execution-semantics.{projected_execution.execution_plan_id}",
        plan=projected_execution,
        contracts=(
            *native_provider.output_semantic_contracts(registry),
            *method_provider.output_semantic_contracts(registry),
        ),
    )
    custody = replace(custody, semantic_validations=semantics)
    method_provider = method_type(registry, design, custody, custody, custody)
    for provider in (native_provider, method_provider):
        for runner in provider.runners(registry):
            ForkingPickler.dumps(runner)
    return tuple(
        sorted(
            (
                ExecutablePlatformPort("candidate-payload-publisher", custody),
                ExecutablePlatformPort("candidate-payload-reader", custody),
                ExecutablePlatformPort("dependency-custody-reader", custody),
                ExecutablePlatformPort("tbs-reactor-source-input", native),
            ),
            key=lambda port: port.port_key,
        )
    )

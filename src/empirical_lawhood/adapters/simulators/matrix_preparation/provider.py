"""Installed native provider over exact, lazy, authenticated prefix ports."""

from collections.abc import Callable
from decimal import Decimal
from typing import Protocol, cast

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import TaskContext, RunnerResult, TaskRunner, TaskProgressEmitter
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan, ExecutionTask
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
    ExternalInputSource,
    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
)
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import decode_port, envelope, output_result, semantic_contracts
from .contracts import CANONICAL_MEDIA_TYPE, NATIVE_SCHEMA, PreparationNativeResult, PreparationSourceConfig, RetainedPreparationPrefix, PreparationPrefixBundleRef
from .prefix_bundles import PreparationPrefixBundle
from .continuation import PreparationDevelopmentContinuation


PREFIX_SOURCE_PORT = "matrix-preparation-retained-prefix-sources"


def preparation_source_semantic_contracts(
    manifest: CapabilityManifest,
) -> tuple[CapabilityOutputSemanticContract, ...]:
    """Resolve the same closed output contracts for execution and read-only custody."""
    return semantic_contracts(
        manifest, (PreparationNativeResult, LinkedCampaignStageEnvelope), NATIVE_SCHEMA
    )


class PreparationPrefixSourceFactory(Protocol):
    @property
    def prefixes(self) -> tuple[RetainedPreparationPrefix, ...]: ...

    @property
    def prefix_bundles(self) -> tuple[PreparationPrefixBundleRef, ...]: ...

    def create_source(self, declaration: PreparationPrefixBundleRef) -> ExternalInputSource: ...


def preparation_source_stage(result: PreparationNativeResult) -> LinkedCampaignStageEnvelope:
    reasons = tuple(sorted({d.reason for d in result.deliveries if d.reason is not None}))
    return envelope(
        f"{result.root.root_id}.native",
        result,
        result.result_id,
        LinkedCampaignStageRole.SOURCE_MATERIALIZATION,
        not reasons,
        reasons,
    )


def preparation_config_payloads(
    plan: ProtocolExecutionPlan,
    manifest: CapabilityManifest,
    config: CanonicalRecord,
    config_id: str,
    expected_tasks: set[str],
    *,
    source: bool = False,
) -> tuple[ExternalInputPayload, ...]:
    tasks = tuple(t for t in plan.tasks if t.capability.capability_key == manifest.capability_key)
    if {t.task_id for t in tasks} != expected_tasks:
        raise ValueError("preparation provider changes its exact task roster")
    expected = {f"config-artifact.{config_id}", *((config_id,) if source else ())}
    config_payload = config.canonical_bytes()
    config_sha256 = config.fingerprint()
    specs = {}
    for task in tasks:
        values = tuple(
            v for v in task.external_inputs if v.expected_payload_schema == config.SCHEMA
        )
        if (
            len(values) != len(expected)
            or {v.logical_artifact_id for v in values} != expected
            or any(v.expected_content_sha256 != config_sha256 for v in values)
        ):
            raise ValueError("preparation task changes its exact configuration input roles")
        specs.update({v.logical_artifact_id: v for v in values})
    parent = ArtifactLineageParent(
        ObjectIdentity.from_record(config_id, config),
        VisibilityCeiling.PROSPECTIVE,
        OutcomeAccess.OUTCOME_BLIND,
    )
    return tuple(
        ExternalInputPayload.from_bytes(
            logical_artifact_id=key,
            payload_schema=config.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type=CANONICAL_MEDIA_TYPE,
            payload=config_payload,
            visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            parent_visibility_ceilings=(parent.visibility_ceiling,),
            lineage_parents=(parent,),
            logical_content_sha256=config_sha256,
        )
        for key in sorted(specs)
    )


class PreparationSourceTask:
    def __init__(self, manifest: CapabilityManifest, config: PreparationSourceConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        return self._execute(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        return self._execute(context, lambda value: emitter.advance(Decimal(value)))

    def _execute(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        prefix = next(
            (p for p in self.config.prefixes if context.task_id == f"{p.root_id}.native"), None
        )
        if prefix is None:
            raise ValueError("preparation native task is outside its issued root census")
        bundle = next(b for b in self.config.prefix_bundles if prefix.root in b.roots)
        expected = {
            f"config-artifact.{self.config.config_id}": self.config.SCHEMA,
            self.config.config_id: self.config.SCHEMA,
            bundle.artifact.artifact_id: bundle.artifact.payload_schema,
        }
        try:
            if (
                len(context.input_ports) != len(expected)
                or {p.artifact_id: p.payload_schema for p in context.input_ports} != expected
            ):
                raise ValueError(
                    "preparation source changes its root-specific external input roster"
                )
            imported = None
            for port in context.input_ports:
                if port.payload_schema == self.config.SCHEMA:
                    if decode_port(port, PreparationSourceConfig) != self.config:
                        raise ValueError("preparation source received another native config")
                else:
                    retained = decode_port(port, PreparationPrefixBundle)
                    if retained.reference != bundle:
                        raise ValueError("preparation task received another retained-prefix bundle")
                    imported = retained.for_root(prefix)
            if imported is None or imported.fingerprint() != prefix.artifact.sha256:
                raise ValueError("preparation source lost its authenticated prefix identity")
            from .source import execute_preparation_root

            result, data = execute_preparation_root(
                self.config, prefix.root, imported, progress=progress
            )
            stage = preparation_source_stage(result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    NATIVE_SCHEMA: data,
                    stage.SCHEMA: stage.canonical_bytes(),
                },
                (
                    "exact-native-root-branch-census",
                    "authenticated-exposed-prefix",
                    "separate-noise-purposes",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class PreparationSourceProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: PreparationSourceConfig,
        sources: PreparationPrefixSourceFactory,
        *,
        continuation: PreparationDevelopmentContinuation | None = None,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
            or sources.prefixes != config.prefixes
            or sources.prefix_bundles != config.prefix_bundles
        ):
            raise ValueError(
                "preparation native provider changes its exact registry/config/prefix port"
            )
        self.registry, self.manifest, self.config, self.sources = (
            registry,
            manifest,
            config,
            sources,
        )
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self.continuation = continuation
        if continuation is not None:
            continuation.validate_source(config)

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("preparation native runner registry/records differ")
        return (cast(TaskRunner, PreparationSourceTask(self.manifest, self.config)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("preparation native execution registry/records differ")
        continuation = self.continuation
        if continuation is not None and plan.source_plan.object_id != continuation.run_id:
            raise ValueError("development continuation requires its new execution identity")
        roots = self.config.roots if continuation is None else continuation.missing_roots
        values = list(
            preparation_config_payloads(
                plan,
                self.manifest,
                self.config,
                self.config.config_id,
                {f"{root.root_id}.native" for root in roots},
                source=True,
            )
        )
        by_id = {t.task_id: t for t in plan.tasks}
        config_size = len(self.config.canonical_bytes())
        emitted = set()
        for prefix in self.config.prefixes:
            if prefix.root not in roots:
                continue
            bundle = next(b for b in self.config.prefix_bundles if prefix.root in b.roots)
            task = by_id[f"{prefix.root_id}.native"]
            if not isinstance(task, ExecutionTask) or task.dependency_task_ids:
                raise ValueError(
                    "preparation source must restore one exact root without another native task"
                )
            imported = tuple(
                p
                for p in task.external_inputs
                if p.expected_payload_schema == bundle.artifact.payload_schema
            )
            scientific = tuple(
                p
                for p in task.scientific_inputs
                if p.external_input_id == bundle.imported_artifact_id
            )
            if (
                len(task.external_inputs) != 3
                or len(imported) != 1
                or len(scientific) != 1
                or imported[0].logical_artifact_id != bundle.artifact.artifact_id
                or imported[0].expected_content_sha256 != bundle.artifact.sha256
                or imported[0].expected_media_type != CANONICAL_MEDIA_TYPE
                or scientific[0].maximum_size_bytes != bundle.artifact.size_bytes
                or imported[0].expected_visibility_ceiling is not VisibilityCeiling.OUTCOME_VISIBLE
                or imported[0].expected_outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
                or 2 * config_size + bundle.artifact.size_bytes
                > task.capability.requested_resources.source_scan_bytes
            ):
                raise ValueError(
                    "preparation source changes exact retained-content visibility or scan bounds"
                )
            if bundle.artifact.artifact_id in emitted:
                continue
            emitted.add(bundle.artifact.artifact_id)
            parents = tuple(
                sorted(
                    {
                        ArtifactLineageParent(
                            identity,
                            VisibilityCeiling.OUTCOME_VISIBLE,
                            OutcomeAccess.DEVELOPMENT_VISIBLE,
                        )
                        for retained in self.config.prefixes
                        if retained.root in bundle.roots
                        for identity in (
                            retained.task_receipt,
                            ObjectIdentity.from_record(retained.artifact.artifact_id, retained.artifact),
                            *retained.source_export.lineage_identities,
                        )
                    },
                    key=lineage_parent_sort_key,
                )
            )
            values.append(
                ExternalInputPayload(
                    bundle.artifact.artifact_id,
                    bundle.artifact.payload_schema,
                    ArtifactProfile.CANONICAL_JSON,
                    CANONICAL_MEDIA_TYPE,
                    self.sources.create_source(bundle),
                    bundle.artifact.size_bytes,
                    bundle.artifact.sha256,
                    bundle.artifact.size_bytes,
                    min(bundle.artifact.size_bytes, MAX_EXTERNAL_INPUT_CHUNK_BYTES),
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                    tuple(p.visibility_ceiling for p in parents),
                    parents,
                    bundle.artifact.sha256,
                )
            )
        return tuple(sorted(values, key=lambda row: row.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("preparation native semantic registry differs")
        return preparation_source_semantic_contracts(self.manifest)

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("preparation native adjudication registry differs")
        return None

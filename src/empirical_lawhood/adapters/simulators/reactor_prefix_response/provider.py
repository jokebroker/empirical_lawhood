"""Native effects only behind an installed campaign runner and scoped input ports."""

from collections.abc import Callable
from dataclasses import fields
from decimal import Decimal

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskProgressEmitter,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .extension_bundle import CAPABILITY
from .panel import ReactorPrefixConfig, ReactorPrefixPanel, ReactorSourceBundle, acquire_prefix_branch, native_task_id


class ReactorPrefixRunner:
    manifest = CAPABILITY
    panel_type = ReactorPrefixPanel

    def __init__(self, config: ReactorPrefixConfig) -> None:
        self.config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        return self._execute(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        return self._execute(context, lambda n: emitter.advance(Decimal(n)))

    def _execute(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        by_task = {
            native_task_id(*b, config=self.config): b for b in self.config.branches
        }
        if (
            context.task_id not in by_task
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != self.manifest.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or len(context.input_ports) != 2
            or {p.payload_schema for p in context.input_ports}
            != set(self.manifest.input_schema_ids)
            or len(context.output_ports) != 1
            or context.output_ports[0].payload_schema != self.panel_type.SCHEMA
            or any(
                p.outcome_access is not OutcomeAccess.OUTCOME_BLIND
                or p.size_bytes > 1024**2
                for p in context.input_ports
            )
        ):
            raise ValueError(
                "reactor task identity, permissions, visibility or ports differ"
            )
        ports = {p.payload_schema: p for p in context.input_ports}
        config = decode_canonical_bytes(
            ports[self.config.SCHEMA].read(), type(self.config), maximum_bytes=1024**2
        )
        if config != self.config:
            raise ValueError("reactor task configuration bytes differ")
        source = decode_canonical_bytes(
            ports[ReactorSourceBundle.SCHEMA].read(),
            ReactorSourceBundle,
            maximum_bytes=1024**2,
        )
        panel = acquire_prefix_branch(
            config, source, by_task[context.task_id], progress
        )
        return RunnerResult(
            (
                TaskOutputPayload(
                    context.output_ports[0].output_id, panel.canonical_bytes()
                ),
            ),
            tuple(
                ReceiptCheck(check, True, ())
                for check in (
                    "reactor-assigned-prefix-roster-retained",
                    "reactor-native-callback-and-delivery-extraction",
                    "reactor-native-source-pin-authenticated",
                    "reactor-numerical-runtime-exact",
                )
            ),
        )


class ReactorPrefixProvider(CampaignRuntimeProvider):
    manifest = CAPABILITY
    panel_type = ReactorPrefixPanel
    runner_type = ReactorPrefixRunner

    def __init__(
        self,
        registry: CapabilityRegistry,
        config: ReactorPrefixConfig,
        source: ExternalInputPayload,
    ) -> None:
        if (
            registry.resolve(
                self.manifest.capability_key, self.manifest.capability_version
            )
            != self.manifest
        ):
            raise ValueError("reactor provider installed manifest differs")
        if (
            source.payload_schema != ReactorSourceBundle.SCHEMA
            or source.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("reactor source port has another schema or visibility")
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.config, self.source = config, source

    def _check(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("reactor provider registry differs")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._check(registry)
        if source_records:
            raise ValueError(
                "reactor source records must use their authenticated input port"
            )
        return (self.runner_type(self.config),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("reactor provider plan or source differs")
        native_tasks = tuple(
            t
            for t in plan.tasks
            if t.capability.capability_key == self.manifest.capability_key
        )
        if tuple(sorted(t.task_id for t in native_tasks)) != tuple(
            native_task_id(*b, config=self.config) for b in self.config.branches
        ):
            raise ValueError(
                "reactor provider requires the twenty exact native branch tasks"
            )
        specs_by_id = {
            s.logical_artifact_id: s
            for t in plan.tasks
            if t.capability.capability_key == self.manifest.capability_key
            for s in t.external_inputs
        }
        specs = tuple(specs_by_id.values())
        if len(specs) != 2 or {s.expected_payload_schema for s in specs} != set(
            self.manifest.input_schema_ids
        ):
            raise ValueError(
                "reactor unit tasks require the exact shared config and source inputs"
            )
        config_spec = next(
            s for s in specs if s.expected_payload_schema == self.config.SCHEMA
        )
        source_spec = next(
            s for s in specs if s.expected_payload_schema == self.source.payload_schema
        )
        if (
            config_spec.expected_content_sha256 != self.config.fingerprint()
            or source_spec.logical_artifact_id != self.source.logical_artifact_id
            or source_spec.expected_content_sha256 != self.source.logical_content_sha256
        ):
            raise ValueError("reactor input content identity differs")
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.config.config_id, self.config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        config = ExternalInputPayload.from_bytes(
            logical_artifact_id=config_spec.logical_artifact_id,
            payload_schema=self.config.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/vnd.empirical-lawhood.canonical+json",
            payload=self.config.canonical_bytes(),
            visibility_ceiling=parent.visibility_ceiling,
            outcome_access=parent.outcome_access,
            parent_visibility_ceilings=(parent.visibility_ceiling,),
            lineage_parents=(parent,),
            logical_content_sha256=self.config.fingerprint(),
        )
        return tuple(sorted((config, self.source), key=lambda p: p.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._check(registry)
        return (
            CapabilityOutputSemanticContract.from_manifest(
                self.manifest,
                payload_schema=self.panel_type.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=self.panel_type.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(f.name for f in fields(self.panel_type))),
            ),
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        self._check(registry)

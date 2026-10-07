"""Native effects only behind an installed campaign runner and scoped input ports."""

from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.transport import ReactorTraceEnvelope

from dataclasses import fields
from decimal import Decimal
from typing import Callable

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
    TaskProgressEmitter,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from .extension_bundle import CAPABILITY
from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.design import ReactorBatchConfig, ReactorBatchSource, BRANCHES, native_task_id

from empirical_lawhood.adapters.simulators.reactor_matched_replay_forecast.trace import acquire_history_unit


class ReactorBatchRunner:
    manifest = CAPABILITY

    def __init__(self, config: ReactorBatchConfig) -> None:
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
        by_task = {native_task_id(*b): b for b in BRANCHES}
        if (
            context.task_id not in by_task
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != self.manifest.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or len(context.input_ports) != 2
            or {p.payload_schema for p in context.input_ports}
            != set(self.manifest.input_schema_ids)
            or len(context.output_ports) != 1
            or context.output_ports[0].payload_schema != ReactorTraceEnvelope.SCHEMA
            or any(
                p.outcome_access is not OutcomeAccess.OUTCOME_BLIND or p.size_bytes > 1024**2
                for p in context.input_ports
            )
        ):
            raise ValueError("reactor task identity, permissions, visibility or ports differ")
        ports = {p.payload_schema: p for p in context.input_ports}
        config = decode_canonical_bytes(
            ports[self.config.SCHEMA].read(), ReactorBatchConfig, maximum_bytes=1024**2
        )
        if config != self.config:
            raise ValueError("reactor task configuration bytes differ")
        source = decode_canonical_bytes(
            ports[ReactorBatchSource.SCHEMA].read(), ReactorBatchSource, maximum_bytes=1024**2
        )
        if source.fingerprint() != config.source_bundle_sha256:
            raise ValueError("native batch source bundle differs from the frozen configuration")
        import platform
        import numpy as np

        if (
            platform.python_version() != config.python_version
            or np.__version__ != config.numpy_version
        ):
            raise ValueError("reactor batch numerical runtime differs")
        unit, _ = by_task[context.task_id]
        scenario = next(s for s in config.scenarios if s.unit_id == unit)
        panel = acquire_history_unit(source, scenario, progress)
        return RunnerResult(
            (
                TaskOutputPayload(
                    context.output_ports[0].output_id,
                    ReactorTraceEnvelope.pack(panel).canonical_bytes(),
                ),
            ),
            tuple(
                ReceiptCheck(check, True, ())
                for check in (
                    "reactor-assigned-batch-roster-retained",
                    "reactor-native-callback-and-delivery-extraction",
                    "reactor-native-source-pin-authenticated",
                    "reactor-numerical-runtime-exact",
                )
            ),
        )


class ReactorBatchProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        config: ReactorBatchConfig,
        source: ExternalInputPayload,
    ) -> None:
        if registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version) != CAPABILITY:
            raise ValueError("reactor provider installed manifest differs")
        if (
            source.payload_schema != ReactorBatchSource.SCHEMA
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
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        self._check(registry)
        if source_records:
            raise ValueError("reactor source records must use their authenticated input port")
        return (ReactorBatchRunner(self.config),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("reactor provider plan or source differs")
        native_tasks = tuple(
            t for t in plan.tasks if t.capability.capability_key == CAPABILITY.capability_key
        )
        if tuple(sorted(t.task_id for t in native_tasks)) != tuple(
            native_task_id(*b) for b in BRANCHES
        ):
            raise ValueError("reactor provider requires the 42 exact paired history tasks")
        specs_by_id = {
            s.logical_artifact_id: s
            for t in plan.tasks
            if t.capability.capability_key == CAPABILITY.capability_key
            for s in t.external_inputs
        }
        specs = tuple(specs_by_id.values())
        if len(specs) != 2 or {s.expected_payload_schema for s in specs} != set(
            CAPABILITY.input_schema_ids
        ):
            raise ValueError("reactor unit tasks require the exact shared config and source inputs")
        config_spec = next(s for s in specs if s.expected_payload_schema == self.config.SCHEMA)
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
                CAPABILITY,
                payload_schema=ReactorTraceEnvelope.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=ReactorTraceEnvelope.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(f.name for f in fields(ReactorTraceEnvelope))),
            ),
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        self._check(registry)
        return None

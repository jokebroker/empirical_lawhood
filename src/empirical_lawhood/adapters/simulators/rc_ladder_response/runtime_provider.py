"""Selected native RC provider; all scientific work stays in the adapter."""

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

from .campaign import VIEWS, run_native_view
from .contracts import ResistorCapacitorLadderNativePanel, ResistorCapacitorLadderStudyConfig
from .extension_bundle import CAPABILITY


def native_task_id(study: ResistorCapacitorLadderStudyConfig, view_id: str) -> str:
    if view_id not in VIEWS:
        raise ValueError("unsupported RC native task view")
    return f"{study.study_id}.native.{view_id}"


class ResistorCapacitorLadderNativeRunner:
    manifest = CAPABILITY

    def __init__(self, study: ResistorCapacitorLadderStudyConfig) -> None:
        self.study = study

    def execute(self, context: TaskContext) -> RunnerResult:
        tasks = {native_task_id(self.study, view): view for view in VIEWS}
        if (
            context.task_id not in tasks
            or context.config.content_sha256 != self.study.fingerprint()
            or context.permissions != CAPABILITY.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or len(context.input_ports) != 2
            or len({port.logical_artifact_id for port in context.input_ports}) != 2
            or any(
                port.payload_schema != ResistorCapacitorLadderStudyConfig.SCHEMA
                or port.outcome_access is not OutcomeAccess.OUTCOME_BLIND
                or port.size_bytes > 128 * 1024
                for port in context.input_ports
            )
            or len(context.output_ports) != 1
            or context.output_ports[0].payload_schema != ResistorCapacitorLadderNativePanel.SCHEMA
        ):
            raise ValueError("RC native task, visibility, resource or ports differ")
        studies = tuple(
            decode_canonical_bytes(
                port.read(), ResistorCapacitorLadderStudyConfig, maximum_bytes=128 * 1024
            )
            for port in context.input_ports
        )
        if any(study != self.study for study in studies):
            raise ValueError(
                "RC native study input bytes differ from selected provider"
            )
        panel = run_native_view(self.study, tasks[context.task_id])
        return RunnerResult(
            (
                TaskOutputPayload(
                    context.output_ports[0].output_id, panel.canonical_bytes()
                ),
            ),
            (
                ReceiptCheck("rc-native-model-action-and-grid", True, ()),
                ReceiptCheck("rc-one-unit-nested-solver-view", True, ()),
            ),
        )

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        result = self.execute(context)
        emitter.advance(Decimal(1))
        return result


class ResistorCapacitorLadderNativeProvider(CampaignRuntimeProvider):
    def __init__(
        self, registry: CapabilityRegistry, study: ResistorCapacitorLadderStudyConfig
    ) -> None:
        if (
            registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version)
            != CAPABILITY
        ):
            raise ValueError("RC native provider installed manifest differs")
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.study = study

    def _check(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("RC native provider registry differs")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._check(registry)
        if source_records:
            raise ValueError("RC native provider requires its issued study config")
        return (ResistorCapacitorLadderNativeRunner(self.study),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("RC native plan or source differs")
        tasks = tuple(
            task
            for task in plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
        )
        if tuple(sorted(task.task_id for task in tasks)) != tuple(
            native_task_id(self.study, view) for view in VIEWS
        ):
            raise ValueError("RC provider requires the exact two solver tasks")
        if (
            len(tasks[0].external_inputs) != 2
            or any(task.external_inputs != tasks[0].external_inputs for task in tasks[1:])
        ):
            raise ValueError("RC provider requires exact shared config and source locators")
        specs = tasks[0].external_inputs
        if (
            len({spec.logical_artifact_id for spec in specs}) != 2
            or {spec.expected_payload_schema for spec in specs}
            != {ResistorCapacitorLadderStudyConfig.SCHEMA}
            or any(spec.expected_content_sha256 != self.study.fingerprint() for spec in specs)
        ):
            raise ValueError("RC native config/source content identities differ")
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.study.study_id, self.study),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return tuple(
            ExternalInputPayload.from_bytes(
                logical_artifact_id=spec.logical_artifact_id,
                payload_schema=ResistorCapacitorLadderStudyConfig.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                payload=self.study.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=self.study.fingerprint(),
            )
            for spec in specs
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._check(registry)
        return (
            CapabilityOutputSemanticContract.from_manifest(
                CAPABILITY,
                payload_schema=ResistorCapacitorLadderNativePanel.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=ResistorCapacitorLadderNativePanel.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(
                    sorted(field.name for field in fields(ResistorCapacitorLadderNativePanel))
                ),
            ),
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        self._check(registry)


__all__ = ['ResistorCapacitorLadderNativeProvider', "native_task_id"]

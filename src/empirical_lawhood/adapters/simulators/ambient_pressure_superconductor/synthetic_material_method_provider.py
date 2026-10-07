'Selected synthetic ambient pressure superconductor gauge covariant response method provider; science stays in the adapter.'

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

from .synthetic_gauge_covariant_response import run_gauge_covariant_response_conformance
from .extension_bundle import CAPABILITY
from .synthetic_material_method_contracts import SyntheticMaterialResponseMethodConfig, SyntheticMaterialResponseMethodPanel


def native_task_id(config: SyntheticMaterialResponseMethodConfig) -> str:
    return f'{config.config_id}.gauge-covariant-response-conformance'


class SyntheticMaterialResponseMethodRunner:
    manifest = CAPABILITY

    def __init__(self, config: SyntheticMaterialResponseMethodConfig) -> None:
        self.config = config

    def execute(self, context: TaskContext) -> RunnerResult:
        if (
            context.task_id != native_task_id(self.config)
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != CAPABILITY.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or len(context.input_ports) != 2
            or len({port.logical_artifact_id for port in context.input_ports}) != 2
            or any(
                port.payload_schema != SyntheticMaterialResponseMethodConfig.SCHEMA
                or port.outcome_access is not OutcomeAccess.OUTCOME_BLIND
                or port.size_bytes > 128 * 1024
                for port in context.input_ports
            )
            or len(context.output_ports) != 1
            or context.output_ports[0].payload_schema != SyntheticMaterialResponseMethodPanel.SCHEMA
        ):
            raise ValueError(
                'ambient pressure superconductor gauge covariant response method native task or ports differ from frozen plan'
            )
        inputs = tuple(
            decode_canonical_bytes(
                port.read(), SyntheticMaterialResponseMethodConfig, maximum_bytes=128 * 1024
            )
            for port in context.input_ports
        )
        if any(value != self.config for value in inputs):
            raise ValueError(
                'ambient pressure superconductor gauge covariant response method source bytes differ from selected provider'
            )
        conformance, observations = run_gauge_covariant_response_conformance()
        if (
            conformance.formalism_sha256 != self.config.formalism.fingerprint()
            or conformance.fixture_suite_sha256
            != self.config.fixture_suite.fingerprint()
            or conformance.material_result_count != 0
        ):
            raise ValueError(
                'ambient pressure superconductor method result differs from the selected fixture chart'
            )
        panel = SyntheticMaterialResponseMethodPanel(
            f'{self.config.config_id}.gauge-covariant-response-method-panel',
            self.config.independent_unit_id,
            self.config.fingerprint(),
            conformance,
            observations,
            True,
        )
        return RunnerResult(
            (
                TaskOutputPayload(
                    context.output_ports[0].output_id, panel.canonical_bytes()
                ),
            ),
            (
                ReceiptCheck('ambient-pressure-superconductor-gauge-covariant-response-method-suite-disclosed', True, ()),
                ReceiptCheck('ambient-pressure-superconductor-one-suite-nine-nested-fixtures', True, ()),
            ),
        )

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        result = self.execute(context)
        emitter.advance(Decimal(1))
        return result


class SyntheticMaterialResponseMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self, registry: CapabilityRegistry, config: SyntheticMaterialResponseMethodConfig
    ) -> None:
        if (
            registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version)
            != CAPABILITY
        ):
            raise ValueError('ambient pressure superconductor gauge covariant response method native manifest differs')
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.config = config

    def _check(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError('ambient pressure superconductor gauge covariant response method native registry differs')

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        self._check(registry)
        if source_records:
            raise ValueError(
                'ambient pressure superconductor gauge covariant response method provider needs its selected issued config'
            )
        return (SyntheticMaterialResponseMethodRunner(self.config),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError('ambient pressure superconductor gauge covariant response method native plan or source differs')
        tasks = tuple(
            task
            for task in plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
        )
        if (
            len(tasks) != 1
            or tasks[0].task_id != native_task_id(self.config)
            or len(tasks[0].external_inputs) != 2
        ):
            raise ValueError(
                'ambient pressure superconductor gauge covariant response method provider requires one source task and two config locators'
            )
        specs = tasks[0].external_inputs
        if len({spec.logical_artifact_id for spec in specs}) != 2 or any(
            spec.expected_payload_schema != SyntheticMaterialResponseMethodConfig.SCHEMA
            or spec.expected_content_sha256 != self.config.fingerprint()
            for spec in specs
        ):
            raise ValueError('ambient pressure superconductor gauge covariant response method source identity differs')
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.config.config_id, self.config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        return tuple(
            ExternalInputPayload.from_bytes(
                logical_artifact_id=spec.logical_artifact_id,
                payload_schema=SyntheticMaterialResponseMethodConfig.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                payload=self.config.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=self.config.fingerprint(),
            )
            for spec in sorted(specs, key=lambda value: value.logical_artifact_id)
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._check(registry)
        return (
            CapabilityOutputSemanticContract.from_manifest(
                CAPABILITY,
                payload_schema=SyntheticMaterialResponseMethodPanel.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=SyntheticMaterialResponseMethodPanel.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(
                    sorted(field.name for field in fields(SyntheticMaterialResponseMethodPanel))
                ),
            ),
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        self._check(registry)


__all__ = ['SyntheticMaterialResponseMethodProvider', "native_task_id"]

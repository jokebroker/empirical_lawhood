"No-science protocol and runtime provider for the margin structural recurrence forecast port.\n\nThis protocol exists only to prove ordinary compilation, execution, receipt,\nsemantic-validation and recovery plumbing before any target can issue. It\ncontains two independent outcome-blind compatibility fixtures. Neither is\nempirical target evidence or structural recurrence truth-known conformance.\n"

from __future__ import annotations

from dataclasses import fields

from empirical_lawhood.adapters.methods.structural_recurrence_runtime import MARGIN_FORECAST_CAPABILITY_VERSION, MarginStructuralRecurrenceForecastCompatibilityAudit, MarginStructuralRecurrenceForecastRuntimeOperation, structural_recurrence_runtime_registry
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)


def _no_science_budget(*, output_bytes: int) -> ResourceBudget:
    return ResourceBudget(
        cpu_cores=1,
        memory_bytes=1_000_000,
        gpu_devices=0,
        wall_time_seconds=2,
        source_scan_bytes=4_000,
        output_bytes=output_bytes,
    )


def build_structural_recurrence_no_science_protocol(
    *,
    registry: CapabilityRegistry,
    config_by_step_id: dict[str, CapabilityConfigRef],
) -> ProtocolTemplate:
    "Build the exact parallel two-node compatibility-audit protocol."

    definitions = (
        (
            "predecessor-audit",
            MarginStructuralRecurrenceForecastRuntimeOperation.TARGET_COMPATIBILITY_AUDIT,
            MarginStructuralRecurrenceForecastCompatibilityAudit.SCHEMA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        ),
        (
            "target-compatibility-audit",
            MarginStructuralRecurrenceForecastRuntimeOperation.TARGET_COMPATIBILITY_AUDIT,
            MarginStructuralRecurrenceForecastCompatibilityAudit.SCHEMA,
            OutcomeAccess.OUTCOME_BLIND,
            VisibilityCeiling.PROSPECTIVE,
        ),
    )
    if set(config_by_step_id) != {value[0] for value in definitions}:
        raise ValueError('margin structural recurrence forecast no-science configs differ from the exact step roster')
    steps = []
    for step_id, operation, output_schema, access, visibility in definitions:
        capability_key = 'method.margin-structural-recurrence-forecast.target-compatibility-audit'
        manifest = registry.resolve(capability_key, MARGIN_FORECAST_CAPABILITY_VERSION)
        config = config_by_step_id[step_id]
        if (
            config.config_schema != manifest.config_schema
            or config.config_schema_sha256 != manifest.config_schema_sha256
        ):
            raise ValueError('margin structural recurrence forecast no-science config differs from capability schema')
        steps.append(
            ProtocolStepTemplate(
                step_id=step_id,
                stage=ScientificStage.QUALIFY,
                capability_key=capability_key,
                capability_version=MARGIN_FORECAST_CAPABILITY_VERSION,
                config=config,
                dependency_step_ids=(),
                outputs=(
                    OutputTemplate(
                        output_id=f"{step_id}.record",
                        payload_schema=output_schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/vnd.empirical-lawhood.canonical+json",
                        filename_suffix=".json",
                    ),
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=access,
                visibility_ceiling=visibility,
                resource_budget=_no_science_budget(output_bytes=700),
                resource_lock_ids=(f"margin-structural-recurrence-forecast-{step_id}",),
                barrier=BarrierKind.NONE,
                maximum_attempts=2,
                obligation_ids=(f"margin-structural-recurrence-forecast-{step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id='margin-structural-recurrence-forecast-conformance-protocol',
        template_version=MARGIN_FORECAST_CAPABILITY_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


class _MarginStructuralRecurrenceForecastStaticRunner:
    def __init__(self, manifest: CapabilityManifest, record: CanonicalRecord) -> None:
        if record.SCHEMA not in manifest.output_schema_ids:
            raise ValueError('margin structural recurrence forecast static runner output is not registered')
        self.manifest = manifest
        self.record = record
        self.execution_count = 0

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        if len(context.input_ports) != 1:
            raise ValueError('margin structural recurrence forecast no-science runner requires its frozen config input')
        if len(context.output_ports) != 1:
            raise ValueError('margin structural recurrence forecast no-science runner requires exactly one output')
        port = context.output_ports[0]
        if port.payload_schema != self.record.SCHEMA:
            raise ValueError('margin structural recurrence forecast no-science task requests the wrong output schema')
        return RunnerResult(
            outputs=(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=self.record.canonical_bytes(),
                ),
            ),
            checks=(ReceiptCheck('margin-structural-recurrence-forecast-runtime-contract', True, ()),),
        )


class StructuralRecurrencePlumbingProvider(CampaignRuntimeProvider):
    "Static outcome-blind provider used only for compatibility-audit runtime plumbing proofs."

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        compatibility_audit: MarginStructuralRecurrenceForecastCompatibilityAudit,
    ) -> None:
        expected = structural_recurrence_runtime_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError('margin structural recurrence forecast no-science provider registry differs')
        self.registry = registry
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self._runners = (
            _MarginStructuralRecurrenceForecastStaticRunner(
                registry.resolve(
                    'method.margin-structural-recurrence-forecast.target-compatibility-audit',
                    MARGIN_FORECAST_CAPABILITY_VERSION,
                ),
                compatibility_audit,
            ),
        )

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError('margin structural recurrence forecast no-science provider registry/source differs')
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError('margin structural recurrence forecast no-science plan registry/source differs')
        if any(task.external_inputs for task in plan.tasks):
            raise ValueError('margin structural recurrence forecast no-science protocol cannot have external inputs')
        return ()

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError('margin structural recurrence forecast no-science semantic registry differs')
        if execution_plan is not None and execution_plan.registry_sha256 != self.registry_sha256:
            raise ValueError('margin structural recurrence forecast no-science semantic plan differs')
        record_types = {
            MarginStructuralRecurrenceForecastCompatibilityAudit.SCHEMA: MarginStructuralRecurrenceForecastCompatibilityAudit,
        }
        values = []
        for runner in self._runners:
            for schema in runner.manifest.output_schema_ids:
                record_type = record_types[schema]
                values.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        runner.manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(sorted(field.name for field in fields(record_type))),
                    )
                )
        return tuple(
            sorted(
                values,
                key=lambda value: (value.capability_key, value.payload_schema),
            )
        )

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> None:
        if registry != self.registry:
            raise ValueError('margin structural recurrence forecast no-science adjudication registry differs')
        return None


__all__ = [
    'StructuralRecurrencePlumbingProvider',
    'build_structural_recurrence_no_science_protocol',
]

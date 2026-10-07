"""development native task binding over the existing custody/progress/publication runner."""

from collections.abc import Callable
from typing import cast

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import ResponseGeometryDevelopmentNativeConfig, ResponseGeometryDevelopmentNativeSegmentResult, ResponseGeometryDevelopmentNativeSegment, development_segments, execute_development_segment
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeSegmentResult, ResponseGeometryAssayNativeSegment
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import DEVELOPMENT_HDF5_SCHEMA
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import TaskRunner
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .provider import ResponseGeometryAssaySourceTask, config_payloads, semantic_contracts


class ResponseGeometryDevelopmentSourceTask(ResponseGeometryAssaySourceTask):
    def __init__(self, manifest: CapabilityManifest, config: ResponseGeometryDevelopmentNativeConfig) -> None:
        if type(config) is not ResponseGeometryDevelopmentNativeConfig or manifest.config_schema != config.SCHEMA:
            raise ValueError("development source task requires its exact native config binding")
        super().__init__(manifest, config)

    def segments(self) -> tuple[ResponseGeometryDevelopmentNativeSegment, ...]:
        return development_segments()

    def execute_native(
        self,
        segment: ResponseGeometryAssayNativeSegment,
        previous: ResponseGeometryAssayNativeSegmentResult | None,
        progress: Callable[[int], None] | None,
    ) -> tuple[ResponseGeometryDevelopmentNativeSegmentResult, bytes]:
        return execute_development_segment(
            cast(ResponseGeometryDevelopmentNativeConfig, self.config),
            cast(ResponseGeometryDevelopmentNativeSegment, segment),
            cast(ResponseGeometryDevelopmentNativeSegmentResult | None, previous),
            progress=progress,
        )


class ResponseGeometryDevelopmentSourceProvider(CampaignRuntimeProvider):
    def __init__(
        self, registry: CapabilityRegistry, manifest: CapabilityManifest, config: ResponseGeometryDevelopmentNativeConfig
    ) -> None:
        if registry.resolve(manifest.capability_key, manifest.capability_version) != manifest or (
            type(config) is not ResponseGeometryDevelopmentNativeConfig or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("development native provider changes its installed capability/config")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("development native runner registry/records differ")
        return (cast(TaskRunner, ResponseGeometryDevelopmentSourceTask(self.manifest, self.config)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("development native execution registry/records differ")
        return config_payloads(
            plan,
            self.manifest,
            self.config,
            self.config.config_id,
            {v.task_id for v in development_segments()},
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("development native semantic registry differs")
        return semantic_contracts(
            self.manifest, (ResponseGeometryDevelopmentNativeSegmentResult, LinkedCampaignStageEnvelope), DEVELOPMENT_HDF5_SCHEMA
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("development native adjudication registry differs")
        return None

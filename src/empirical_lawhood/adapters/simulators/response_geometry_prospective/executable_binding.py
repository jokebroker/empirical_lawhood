"""Installed native factory and carrier-to-template binding for assay."""

from empirical_lawhood.adapters.composition.protocol_helpers import (
    capability_config_ref as _config_ref,
    protocol_outputs as _outputs,
)

from dataclasses import dataclass, replace
from math import ceil
from typing import cast

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.source_qualification import FreshSourceQualificationExperiment
from empirical_lawhood.runtime.capabilities import (
    CapabilityRegistry,
)
from empirical_lawhood.runtime.executable_bindings import ExecutableBindingContribution, ExecutableCapabilityBinding, ExecutablePlatformPort
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ProtocolTemplate,
    ScientificStage,
)
from empirical_lawhood.runtime.source_qualification import FreshSourceQualificationSubstrateBinding, derive_source_qualification_topology
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.adapters.simulators.six_matrix_response.response_qualification import ResponseGeometryAssayNativeConfig, ResponseGeometryAssayNativeSegmentResult, assay_segments
from empirical_lawhood.adapters.simulators.six_matrix_response.response_source import ASSAY_HDF5_SCHEMA

from .discovery import executable
from .extension_bundle import ASSAY_SOURCE_CAPABILITY, ASSAY_SOURCE_COMPONENTS, SOURCE_RECORDS
from .provider import ResponseGeometryAssaySourceProvider, validate_response_geometry_assay_native_roster


ASSAY_SOURCE_BINDING = executable(ASSAY_SOURCE_CAPABILITY, ASSAY_SOURCE_COMPONENTS, SOURCE_RECORDS)


@dataclass(frozen=True, slots=True)
class ResponseGeometryAssaySourceProviderFactory:
    binding: ExecutableCapabilityBinding = ASSAY_SOURCE_BINDING

    def expand_parameterised_protocol(
        self, *, records: tuple[CanonicalRecord, ...], template: ProtocolTemplate
    ) -> ProtocolTemplate:
        from empirical_lawhood.adapters.methods.response_geometry_prospective.extension_bundle import ASSAY_PROJECTION_CAPABILITY, ASSAY_EVALUATION_CAPABILITY
        from empirical_lawhood.adapters.methods.response_geometry_prospective.projection import ASSAY_DIAGNOSTICS_SCHEMA
        from empirical_lawhood.adapters.methods.response_geometry_prospective.qualification import ResponseGeometryAssayProjectionConfig, ResponseGeometryAssayEvaluationConfig, ResponseGeometryAssayViewReport, ResponseGeometryAssayEvaluation

        by_type = {type(value): value for value in records}
        if (
            len(by_type) != 5
            or len(records) != 5
            or set(by_type) != {*SOURCE_RECORDS, ResponseGeometryAssayProjectionConfig, ResponseGeometryAssayEvaluationConfig}
        ):
            raise ValueError("assay expansion requires its five exact issued config/carrier records")
        carrier = cast(FreshSourceQualificationExperiment, by_type[FreshSourceQualificationExperiment])
        association = cast(
            FreshSourceQualificationSubstrateBinding, by_type[FreshSourceQualificationSubstrateBinding]
        )
        source = cast(ResponseGeometryAssayNativeConfig, by_type[ResponseGeometryAssayNativeConfig])
        projection = cast(ResponseGeometryAssayProjectionConfig, by_type[ResponseGeometryAssayProjectionConfig])
        evaluation = cast(ResponseGeometryAssayEvaluationConfig, by_type[ResponseGeometryAssayEvaluationConfig])
        source_identity = ObjectIdentity.from_record(source.config_id, source)
        projection_identity = ObjectIdentity.from_record(projection.config_id, projection)
        evaluation_identity = ObjectIdentity.from_record(evaluation.config_id, evaluation)
        if (
            carrier.source_config != source_identity
            or carrier.projection_config != projection_identity
            or carrier.evaluator_config != evaluation_identity
            or projection.source_config != source_identity
            or evaluation.projection_config != projection_identity
            or association.qualification_carrier
            != ObjectIdentity.from_record(carrier.extension_set_id, carrier)
            or association.source_config != source_identity
            or association.substrate.installed_executable_binding
            != ObjectIdentity.from_record(self.binding.binding_id, self.binding)
        ):
            raise ValueError("assay expansion changes native/config/owner lineage")
        native = {segment.task_id: segment for segment in assay_segments()}
        validate_response_geometry_assay_native_roster(source, carrier)
        owners = {
            ObjectIdentity.from_record(manifest.capability_key, manifest): (
                manifest,
                config,
                identity,
            )
            for manifest, config, identity in (
                (ASSAY_SOURCE_CAPABILITY, source, source_identity),
                (ASSAY_PROJECTION_CAPABILITY, projection, projection_identity),
                (ASSAY_EVALUATION_CAPABILITY, evaluation, evaluation_identity),
            )
        }
        outputs = {
            ScientificStage.PREPARE: _outputs(
                (
                    ("native-result", ResponseGeometryAssayNativeSegmentResult.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                ("native-observations", ASSAY_HDF5_SCHEMA),
            ),
            ScientificStage.TRANSFORM: _outputs(
                (
                    ("view-report", ResponseGeometryAssayViewReport.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ),
                ("diagnostics", ASSAY_DIAGNOSTICS_SCHEMA),
            ),
            ScientificStage.EVALUATE: _outputs(
                (
                    ("evaluation", ResponseGeometryAssayEvaluation.SCHEMA),
                    ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                    ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
                )
            ),
        }
        stages = derive_source_qualification_topology(carrier).stages
        stage_by_id = {stage.task_id: stage for stage in stages}
        config_sizes = {
            identity: len(config.canonical_bytes()) for _, config, identity in owners.values()
        }
        steps = []
        for stage in stages:
            if stage.owner not in owners:
                raise ValueError("assay expansion selects another scientific owner")
            manifest, config, identity = owners[stage.owner]
            if stage.config != identity:
                raise ValueError("assay expansion stage config differs")
            # An unmeasured duration forecast for authoring authoring metadata:
            # 30 ms/native update plus 2 s per source segment, 120 s/projection.
            # The compiler/runtime-owner execution envelope has no elapsed-time stop.
            seconds = (
                ceil(2 + native[stage.task_id].native_updates * 0.03)
                if stage.stage is ScientificStage.PREPARE
                else 120
            )
            # The compiler supplies the capability config. Preparation prefixes
            # also retain the graph's explicit MODEL config role. Charge both reads,
            # plus each predecessor's whole output allocation exactly once.
            input_bytes = (
                2 if stage.stage is ScientificStage.PREPARE and not stage.dependency_task_ids else 1
            ) * config_sizes[identity]
            input_bytes += sum(
                owners[stage_by_id[dependency].owner][0].resource_ceiling.output_bytes
                for dependency in stage.dependency_task_ids
            )
            if input_bytes > manifest.resource_ceiling.source_scan_bytes:
                raise ValueError("assay input roster exceeds its installed scan ceiling")
            steps.append(
                ProtocolStepTemplate(
                    stage.task_id,
                    stage.stage,
                    manifest.capability_key,
                    manifest.capability_version,
                    _config_ref(config, identity, manifest),
                    stage.dependency_task_ids,
                    outputs[stage.stage],
                    manifest.permissions,
                    OutcomeAccess.EVALUATION_REVEALED
                    if stage.stage is ScientificStage.EVALUATE
                    else OutcomeAccess.EVALUATION_SEALED,
                    VisibilityCeiling.PROSPECTIVE,
                    replace(
                        manifest.resource_ceiling,
                        wall_time_seconds=seconds,
                        source_scan_bytes=input_bytes,
                    ),
                    (),
                    BarrierKind.REVEAL
                    if stage.stage is ScientificStage.EVALUATE
                    else BarrierKind.FREEZE
                    if stage.stage is ScientificStage.TRANSFORM
                    else BarrierKind.NONE,
                    1,
                    ("response-geometry-assay.single-terminal",)
                    if stage.stage is ScientificStage.EVALUATE
                    else (f"custody.{stage.task_id}",),
                )
            )
        if len(steps) != 865:
            raise ValueError("assay expansion changes the complete 865-task production topology")
        return replace(
            template,
            template_id=f"{template.template_id}.{carrier.fingerprint()[:16]}",
            steps=tuple(steps),
            requires_model_set=False,
            requests_controller=False,
            nonactuating=True,
        )

    def build_provider(
        self,
        *,
        registry: CapabilityRegistry,
        records: tuple[CanonicalRecord, ...],
        platform_ports: tuple[ExecutablePlatformPort, ...],
    ) -> ResponseGeometryAssaySourceProvider:
        by_type = {type(value): value for value in records}
        if platform_ports or len(records) != 3 or set(by_type) != set(SOURCE_RECORDS):
            raise ValueError("assay native provider requires its three exact issued records")
        return ResponseGeometryAssaySourceProvider(
            registry,
            ASSAY_SOURCE_CAPABILITY,
            cast(ResponseGeometryAssayNativeConfig, by_type[ResponseGeometryAssayNativeConfig]),
            cast(FreshSourceQualificationExperiment, by_type[FreshSourceQualificationExperiment]),
            cast(
                FreshSourceQualificationSubstrateBinding,
                by_type[FreshSourceQualificationSubstrateBinding],
            ),
            ObjectIdentity.from_record(self.binding.binding_id, self.binding),
        )


EXECUTABLE_BINDING_CONTRIBUTION = ExecutableBindingContribution(
    "executable-contribution.response-geometry-assay.source", "1.0.0", (ASSAY_SOURCE_BINDING,)
)
EXECUTABLE_BINDING_FACTORIES = (ResponseGeometryAssaySourceProviderFactory(),)
EXECUTABLE_RECORD_TYPES = tuple(sorted(SOURCE_RECORDS, key=lambda value: value.SCHEMA))

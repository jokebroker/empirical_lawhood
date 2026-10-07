"""Closed, write-free composition for each conditional selective dependence response stage.

These builders deliberately require the exact records that become external
inputs.  They do not invent construct reviews, development outcomes,
authorities, reveals or completion evidence, and configuration cannot select
callables or imports.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from hashlib import sha256

from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256, validate_stable_id
from empirical_lawhood.planning.formal_analysis import FormalGapSourceCapabilityInventory
from empirical_lawhood.planning.study_authoring import DesignInputRecord
from empirical_lawhood.runtime.capabilities import CapabilityConfigRef, CapabilityRegistry
from empirical_lawhood.runtime.candidate_composition import CandidateCapabilityCatalog
from empirical_lawhood.runtime.providers import CampaignRuntimeProvider
from empirical_lawhood.runtime.source_resolution import CandidateCapabilityConfigDecoder

from .completion import SelectiveDependenceResponseTargetCompletionEnvelope
from .contracts import SelectiveDependenceResponseMethodQuestionFreeze
from .cross_target_protocol import SelectiveDependenceResponseCrossTargetOperation, SelectiveDependenceResponseCrossTargetRuntimeProvider, build_cross_target_protocol, cross_target_candidate_catalog, cross_target_config_decoders, cross_target_registry, cross_target_runtime_config
from .evaluation_protocol import SelectiveDependenceResponseEvaluationBinding, SelectiveDependenceResponseEvaluationOperation, SelectiveDependenceResponseEvaluationRuntimeProvider, build_evaluation_protocol, build_evaluation_registry, evaluation_candidate_catalog, evaluation_config_decoders, evaluation_runtime_config
from .development_completion import SelectiveDependenceResponseDevelopmentCompletionEnvelope
from .study_forecast_protocol import SelectiveDependenceResponseStudyForecastRuntimeProvider, build_study_forecast_protocol, study_forecast_candidate_catalog, study_forecast_config_decoders, study_forecast_registry, study_forecast_runtime_config
from .target_protocol import SELECTIVE_DEPENDENCE_RESPONSE_TARGET_DEVELOPMENT_OPERATIONS, SELECTIVE_DEPENDENCE_RESPONSE_TARGET_SOURCE_OPERATIONS, SelectiveDependenceResponseTargetBinding, SelectiveDependenceResponseTargetOperation, SelectiveDependenceResponseTargetRuntimeProvider, SelectiveDependenceResponseTargetStage, build_target_development_protocol, build_target_registry, build_target_source_canary_protocol, target_config_decoders, target_development_candidate_catalog, target_runtime_config, target_source_canary_candidate_catalog
from .source_completion import SelectiveDependenceResponseSourceCanaryCompletionEnvelope


@dataclass(frozen=True, slots=True)
class SelectiveDependenceResponseStageComposition:
    """One exact candidate catalog plus its closed decoder/provider bindings."""

    stage_id: str
    implementation_sha256: str
    registry: CapabilityRegistry
    catalog: CandidateCapabilityCatalog
    config_decoders: tuple[CandidateCapabilityConfigDecoder, ...]
    runtime_provider: CampaignRuntimeProvider
    configs: tuple[CanonicalRecord, ...]
    known_design_inputs: tuple[DesignInputRecord, ...] = ()
    formal_source_inventories: tuple[FormalGapSourceCapabilityInventory, ...] = ()

    def __post_init__(self) -> None:
        validate_stable_id(self.stage_id, field_name="stage_id")
        validate_sha256(self.implementation_sha256, field_name="implementation_sha256")
        if not self.configs:
            raise ValueError("selective dependence response stage composition requires canonical configs")
        if tuple(value.input_id for value in self.known_design_inputs) != tuple(
            sorted({value.input_id for value in self.known_design_inputs})
        ):
            raise ValueError("selective dependence response stage design inputs must be sorted and unique")
        if tuple(value.denominator_id for value in self.formal_source_inventories) != tuple(
            sorted({value.denominator_id for value in self.formal_source_inventories})
        ):
            raise ValueError("selective dependence response stage formal inventories must be sorted and unique")

    @property
    def config_payloads(self) -> tuple[bytes, ...]:
        return tuple(value.canonical_bytes() for value in self.configs)


def _config_ref(
    config: CanonicalRecord,
    *,
    config_id: str,
    capability_key: str,
    registry: CapabilityRegistry,
) -> CapabilityConfigRef:
    payload = config.canonical_bytes()
    manifest = registry.resolve(capability_key, "1.0.0")
    return CapabilityConfigRef(
        config_id=config_id,
        config_schema=config.SCHEMA,
        config_schema_sha256=manifest.config_schema_sha256,
        content_sha256=sha256(payload).hexdigest(),
        artifact_id=f"config-artifact.{config_id}",
    )


def compose_target_source_canary(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    implementation_sha256: str,
) -> SelectiveDependenceResponseStageComposition:
    if binding.method_completion is None:
        raise ValueError("source-canary composition requires terminal method completion")
    if binding.method_completion.method_implementation_sha256 != implementation_sha256:
        raise ValueError("method/target implementation identity differs")
    registry = build_target_registry(
        binding,
        implementation_sha256=implementation_sha256,
        operations=SELECTIVE_DEPENDENCE_RESPONSE_TARGET_SOURCE_OPERATIONS,
        stage=SelectiveDependenceResponseTargetStage.SOURCE_CANARY,
    )
    configs = tuple(
        target_runtime_config(binding, value) for value in SELECTIVE_DEPENDENCE_RESPONSE_TARGET_SOURCE_OPERATIONS
    )
    step_by_operation = {
        SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN: "freeze-target-design",
        SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS: "freeze-target-analysis",
        SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW: "bind-construct-review",
        SelectiveDependenceResponseTargetOperation.QUALIFY_SOURCE: "qualify-target-source",
    }
    refs = {
        step_by_operation[config.operation]: _config_ref(
            config,
            config_id=config.config_id,
            capability_key=config.capability_key,
            registry=registry,
        )
        for config in configs
    }
    protocol = build_target_source_canary_protocol(
        binding, registry=registry, config_by_step_id=refs
    )
    catalog = target_source_canary_candidate_catalog(binding, protocol=protocol, registry=registry)
    return SelectiveDependenceResponseStageComposition(
        stage_id=f"selective-dependence-response.{binding.target_slug}.source-canary",
        implementation_sha256=implementation_sha256,
        registry=registry,
        catalog=catalog,
        config_decoders=target_config_decoders(binding, catalog),
        runtime_provider=SelectiveDependenceResponseTargetRuntimeProvider(registry=registry, binding=binding),
        configs=configs,
    )


def compose_target_development(
    binding: SelectiveDependenceResponseTargetBinding,
    *,
    source_qualification: CanonicalRecord,
    source_completion: SelectiveDependenceResponseSourceCanaryCompletionEnvelope,
    implementation_sha256: str,
) -> SelectiveDependenceResponseStageComposition:
    if binding.method_completion is None:
        raise ValueError("development composition requires terminal method completion")
    if binding.method_completion.method_implementation_sha256 != implementation_sha256:
        raise ValueError("method/target implementation identity differs")
    development_binding = replace(
        binding,
        source_qualification=source_qualification,
        source_completion=source_completion,
    )
    if source_completion.source_implementation_sha256 != implementation_sha256:
        raise ValueError("source/development implementation identity differs")
    registry = build_target_registry(
        development_binding,
        implementation_sha256=implementation_sha256,
        operations=SELECTIVE_DEPENDENCE_RESPONSE_TARGET_DEVELOPMENT_OPERATIONS,
        stage=SelectiveDependenceResponseTargetStage.DEVELOPMENT,
    )
    configs = tuple(
        target_runtime_config(development_binding, value)
        for value in SELECTIVE_DEPENDENCE_RESPONSE_TARGET_DEVELOPMENT_OPERATIONS
    )
    step_by_operation = {
        SelectiveDependenceResponseTargetOperation.FREEZE_DESIGN: "freeze-target-design",
        SelectiveDependenceResponseTargetOperation.FREEZE_ANALYSIS: "freeze-target-analysis",
        SelectiveDependenceResponseTargetOperation.BIND_CONSTRUCT_REVIEW: "bind-construct-review",
        SelectiveDependenceResponseTargetOperation.GENERATE_DEVELOPMENT: "generate-development-panel",
        SelectiveDependenceResponseTargetOperation.ANALYZE_DEVELOPMENT: "analyze-development",
    }
    refs = {
        step_by_operation[config.operation]: _config_ref(
            config,
            config_id=config.config_id,
            capability_key=config.capability_key,
            registry=registry,
        )
        for config in configs
    }
    protocol = build_target_development_protocol(
        development_binding,
        registry=registry,
        config_by_step_id=refs,
    )
    catalog = target_development_candidate_catalog(
        development_binding,
        protocol=protocol,
        registry=registry,
    )
    return SelectiveDependenceResponseStageComposition(
        stage_id=f"selective-dependence-response.{binding.target_slug}.development",
        implementation_sha256=implementation_sha256,
        registry=registry,
        catalog=catalog,
        config_decoders=target_config_decoders(development_binding, catalog),
        runtime_provider=SelectiveDependenceResponseTargetRuntimeProvider(
            registry=registry, binding=development_binding
        ),
        configs=configs,
    )


def compose_study_forecast(
    completions: tuple[
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
        SelectiveDependenceResponseDevelopmentCompletionEnvelope,
    ],
    *,
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
    implementation_sha256: str,
) -> SelectiveDependenceResponseStageComposition:
    registry = study_forecast_registry(implementation_sha256=implementation_sha256)
    config = study_forecast_runtime_config(completions)
    ref = _config_ref(
        config,
        config_id=config.config_id,
        capability_key=config.capability_key,
        registry=registry,
    )
    protocol = build_study_forecast_protocol(registry=registry, config=ref)
    catalog = study_forecast_candidate_catalog(
        protocol=protocol,
        registry=registry,
        completions=completions,
        method_question=method_question,
    )
    return SelectiveDependenceResponseStageComposition(
        stage_id="selective-dependence-response.programme-forecast",
        implementation_sha256=implementation_sha256,
        registry=registry,
        catalog=catalog,
        config_decoders=study_forecast_config_decoders(),
        runtime_provider=SelectiveDependenceResponseStudyForecastRuntimeProvider(
            registry=registry,
            config=config,
            completions=completions,
            method_question=method_question,
        ),
        configs=(config,),
    )


def compose_target_evaluation(
    binding: SelectiveDependenceResponseEvaluationBinding,
    *,
    implementation_sha256: str,
) -> SelectiveDependenceResponseStageComposition:
    registry = build_evaluation_registry(binding, implementation_sha256=implementation_sha256)
    configs = tuple(
        evaluation_runtime_config(binding, value) for value in SelectiveDependenceResponseEvaluationOperation
    )
    step_by_operation = {
        SelectiveDependenceResponseEvaluationOperation.FREEZE_EVALUATION_PACKAGE: "freeze-evaluation-package",
        SelectiveDependenceResponseEvaluationOperation.GENERATE_EVALUATION: "generate-sealed-evaluation-panel",
        SelectiveDependenceResponseEvaluationOperation.REVEAL_AND_ADJUDICATE: "reveal-and-adjudicate-target",
        SelectiveDependenceResponseEvaluationOperation.PUBLISH_TARGET_HANDOFF: "publish-target-handoff",
    }
    refs = {
        step_by_operation[config.operation]: _config_ref(
            config,
            config_id=config.config_id,
            capability_key=config.capability_key,
            registry=registry,
        )
        for config in configs
    }
    protocol = build_evaluation_protocol(binding, registry=registry, config_by_step_id=refs)
    catalog = evaluation_candidate_catalog(binding, protocol=protocol, registry=registry)
    return SelectiveDependenceResponseStageComposition(
        stage_id=f"selective-dependence-response.{binding.target_slug}.evaluation",
        implementation_sha256=implementation_sha256,
        registry=registry,
        catalog=catalog,
        config_decoders=evaluation_config_decoders(binding, catalog),
        runtime_provider=SelectiveDependenceResponseEvaluationRuntimeProvider(registry=registry, binding=binding),
        configs=configs,
    )


def compose_cross_target(
    completion_envelopes: tuple[SelectiveDependenceResponseTargetCompletionEnvelope, SelectiveDependenceResponseTargetCompletionEnvelope],
    *,
    method_question: SelectiveDependenceResponseMethodQuestionFreeze,
    implementation_sha256: str,
) -> SelectiveDependenceResponseStageComposition:
    registry = cross_target_registry(implementation_sha256=implementation_sha256)
    configs = tuple(
        cross_target_runtime_config(completion_envelopes, method_question, value)
        for value in SelectiveDependenceResponseCrossTargetOperation
    )
    refs = {
        config.operation: _config_ref(
            config,
            config_id=config.config_id,
            capability_key=config.capability_key,
            registry=registry,
        )
        for config in configs
    }
    protocol = build_cross_target_protocol(registry=registry, config_by_operation=refs)
    catalog = cross_target_candidate_catalog(
        protocol=protocol,
        registry=registry,
        completion_envelopes=completion_envelopes,
        method_question=method_question,
    )
    return SelectiveDependenceResponseStageComposition(
        stage_id="selective-dependence-response.cross-target",
        implementation_sha256=implementation_sha256,
        registry=registry,
        catalog=catalog,
        config_decoders=cross_target_config_decoders(),
        runtime_provider=SelectiveDependenceResponseCrossTargetRuntimeProvider(
            registry=registry,
            configs=configs,
            completion_envelopes=completion_envelopes,
            method_question=method_question,
        ),
        configs=configs,
    )


__all__ = [
    'SelectiveDependenceResponseStageComposition',
    "compose_cross_target",
    'compose_study_forecast',
    "compose_target_development",
    "compose_target_source_canary",
    "compose_target_evaluation",
]

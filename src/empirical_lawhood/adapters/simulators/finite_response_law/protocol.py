"""Adapter-owned source-only protocol from the exact retained qualification carrier."""

from dataclasses import replace
from math import ceil

from empirical_lawhood.kernel.evidence import (
    EvidenceCeiling,
    OutcomeAccess,
    VisibilityCeiling,
)
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import (
    BarrierKind,
    ProtocolStepTemplate,
    ScientificStage,
)
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding, derive_source_qualification_topology
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.adapters.methods.finite_response_law.extension_bundle import PROJECTION_CAPABILITY, EVALUATION_CAPABILITY
from empirical_lawhood.adapters.methods.finite_response_law.native_records import FiniteResponseLawProjectionConfig, FiniteResponseLawNativeEvaluationConfig
from empirical_lawhood.adapters.methods.finite_response_law.native_provider import EVALUATOR_TASK_ID
from .contracts import FiniteResponseLawNativeConfig, native_invocations
from .discovery import SOURCE_CAPABILITY
from .native_artifact import NATIVE_PAIR_SCHEMA
from .provider import native_result_type
from .fresh_contracts import FiniteResponseLawCalibrationConfig
from .assigned_contracts import FiniteResponseLawAssignedCalibrationConfig, FiniteResponseLawAssignedEvaluationConfig
from empirical_lawhood.adapters.methods.finite_response_law.native_records import native_method_types
from empirical_lawhood.runtime.capabilities import CapabilityManifest
from .roster import Q_CLOCK, Q_RECEIVERS, native_declarations, substrate_binding


def native_capabilities(
    config: FiniteResponseLawNativeConfig,
) -> tuple[CapabilityManifest, CapabilityManifest, CapabilityManifest]:
    from .evaluation_contracts import FiniteResponseLawEvaluationConfig

    if type(config) is FiniteResponseLawAssignedEvaluationConfig:
        from .evaluation import discovery as evaluation_discovery

        return (
            evaluation_discovery.SOURCE_CAPABILITY,
            evaluation_discovery.PROJECTION_CAPABILITY,
            evaluation_discovery.EVALUATION_CAPABILITY,
        )
    if type(config) is FiniteResponseLawAssignedCalibrationConfig:
        from .calibration import discovery

        return (
            discovery.SOURCE_CAPABILITY,
            discovery.PROJECTION_CAPABILITY,
            discovery.EVALUATION_CAPABILITY,
        )
    if type(config) is FiniteResponseLawCalibrationConfig:
        raise ValueError(
            "Exposed fixed calibration roots have no installed source route"
        )
    if type(config) is FiniteResponseLawEvaluationConfig:
        raise ValueError("Exposed fixed evaluation roots have no installed controlled source route")
    return SOURCE_CAPABILITY, PROJECTION_CAPABILITY, EVALUATION_CAPABILITY


def validate_source_records(
    config: FiniteResponseLawNativeConfig,
    carrier: PredecessorBoundSourceQualificationExperiment,
    association: PredecessorBoundSourceQualificationSubstrateBinding,
    installed_binding: ObjectIdentity,
) -> None:
    declarations = native_declarations(config)
    source_capability, _, _ = native_capabilities(config)
    if (
        carrier.source_capability != config.native_owner
        or config.native_owner
        != ObjectIdentity.from_record(
            source_capability.capability_key, source_capability
        )
        or carrier.source_config != ObjectIdentity.from_record(config.spec_id, config)
        or (carrier.physical_units, carrier.segments, carrier.views)
        != (declarations.units, declarations.segments, declarations.views)
        or carrier.retained_predecessors != config.retained_predecessors
        or carrier.receiver_ids != Q_RECEIVERS
        or carrier.clock_ids != (Q_CLOCK,)
        or carrier.evaluator_task_id != EVALUATOR_TASK_ID
        or carrier.evidence_ceiling is not EvidenceCeiling.MEASUREMENT
        or association != substrate_binding(config, carrier, installed_binding)
    ):
        raise ValueError(
            "Finite response-law source records change native units, clocks, actions, views or source-completion ceiling"
        )


def native_protocol_steps(
    config: FiniteResponseLawNativeConfig,
    carrier: PredecessorBoundSourceQualificationExperiment,
    projection: FiniteResponseLawProjectionConfig,
    evaluation: FiniteResponseLawNativeEvaluationConfig,
) -> tuple[ProtocolStepTemplate, ...]:
    source_capability, projection_capability, evaluation_capability = (
        native_capabilities(config)
    )
    _, _, _, view_type, evaluation_type = native_method_types(config)
    if (
        projection.native_spec != config
        or evaluation.projection != projection
        or carrier.projection_config
        != ObjectIdentity.from_record(projection.config_id, projection)
        or carrier.evaluator_config
        != ObjectIdentity.from_record(evaluation.config_id, evaluation)
        or carrier.projection_capability
        != ObjectIdentity.from_record(
            projection_capability.capability_key, projection_capability
        )
        or carrier.evaluator_capability
        != ObjectIdentity.from_record(
            evaluation_capability.capability_key, evaluation_capability
        )
    ):
        raise ValueError("Finite response-law protocol changes its exact projection/completion owners")
    owners = {
        ObjectIdentity.from_record(manifest.capability_key, manifest): (
            manifest,
            value,
            identity,
        )
        for manifest, value, identity in (
            (source_capability, config, carrier.source_config),
            (projection_capability, projection, carrier.projection_config),
            (evaluation_capability, evaluation, carrier.evaluator_config),
        )
    }
    outputs = {
        ScientificStage.PREPARE: _outputs(
            (
                ("native-result", native_result_type(config).SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            ),
            ("native-observations", NATIVE_PAIR_SCHEMA),
        ),
        ScientificStage.TRANSFORM: _outputs(
            (
                ("view-report", view_type.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            )
        ),
        ScientificStage.EVALUATE: _outputs(
            (
                ("evaluation", evaluation_type.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
            )
        ),
    }
    stages = derive_source_qualification_topology(carrier).stages
    native = {t.task_id: t for t in native_invocations(config)}
    retained = {r.segment_id: r for r in config.retained_predecessors}
    output_bounds = {
        stage.task_id: (
            5 * 512 * 1024
            if stage.task_id in native and native[stage.task_id].phase != "prefix"
            else owners[stage.owner][0].resource_ceiling.output_bytes
        )
        for stage in stages
    }
    steps = []
    for stage in stages:
        manifest, value, identity = owners[stage.owner]
        if stage.config != identity:
            raise ValueError("Finite response-law protocol stage changes its exact config identity")
        scan = len(value.canonical_bytes()) * (
            2
            if stage.stage is ScientificStage.PREPARE and not stage.dependency_task_ids
            else 1
        )
        scan += sum(output_bounds[d] for d in stage.dependency_task_ids)
        if stage.stage is ScientificStage.PREPARE:
            prior = retained.get(native[stage.task_id].predecessor_segment_id or "")
            if prior is not None:
                scan += sum(
                    a.size_bytes for a in (*prior.artifacts, prior.task_receipt)
                )
        if scan > manifest.resource_ceiling.source_scan_bytes:
            raise ValueError(
                "Finite response-law native protocol exceeds the complete declared input scan bound"
            )
        wall = (
            ceil(2 + native[stage.task_id].maximum_native_updates * 0.004)
            if stage.stage is ScientificStage.PREPARE
            else manifest.resource_ceiling.wall_time_seconds
        )
        steps.append(
            ProtocolStepTemplate(
                stage.task_id,
                stage.stage,
                manifest.capability_key,
                manifest.capability_version,
                _config_ref(value, identity, manifest),
                stage.dependency_task_ids,
                outputs[stage.stage],
                manifest.permissions,
                OutcomeAccess.EVALUATOR_REVEAL
                if stage.stage is ScientificStage.EVALUATE
                and type(config) is FiniteResponseLawAssignedCalibrationConfig
                else OutcomeAccess.EVALUATION_REVEALED
                if stage.stage is ScientificStage.EVALUATE
                else OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE
                if type(config) is FiniteResponseLawAssignedCalibrationConfig
                else VisibilityCeiling.OUTCOME_VISIBLE
                if stage.stage is ScientificStage.EVALUATE
                or config.retained_predecessors
                else VisibilityCeiling.DEVELOPMENT_ONLY,
                replace(
                    manifest.resource_ceiling,
                    source_scan_bytes=scan,
                    wall_time_seconds=wall,
                    output_bytes=output_bounds[stage.task_id],
                ),
                (),
                BarrierKind.REVEAL
                if stage.stage is ScientificStage.EVALUATE
                else BarrierKind.FREEZE
                if stage.stage is ScientificStage.TRANSFORM
                else BarrierKind.NONE,
                1,
                (f"finite-response-law.{config.stage}.single-terminal",)
                if stage.stage is ScientificStage.EVALUATE
                else (f"custody.{stage.task_id}",),
            )
        )
    if len(steps) != (
        705
        if type(config) is FiniteResponseLawAssignedCalibrationConfig
        else 23
        if config.stage == "native-canary"
        else 753
    ):
        raise ValueError(
            "Finite response-law protocol differs from its complete source/projection/evaluation census"
        )
    return tuple(sorted(steps, key=lambda s: s.step_id))

"""Exact 4,104-source/24-projection/one-screen preparation-policy development protocol."""

from dataclasses import replace

from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_projection import FiniteResponseLawPreparationPolicyProjectionConfig, FiniteResponseLawPreparationPolicyRootPanel
from empirical_lawhood.adapters.methods.finite_response_law.preparation_policy_provider import PREPARATION_POLICY_EVALUATOR_TASK_ID
from empirical_lawhood.adapters.methods.finite_response_law.preparation_screen_results import FiniteResponseLawPreparationScreenResult, FiniteResponseLawPreparationPolicyScreenConfig, FiniteResponseLawPreparationPolicyUpperPayloadFreeze
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_contracts import FiniteResponseLawPreparationPolicyNativeConfig, preparation_policy_native_invocations
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_native_artifact import PREPARATION_POLICY_NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_source_outputs import FiniteResponseLawPreparationPolicyNativeTaskResult
from empirical_lawhood.adapters.composition.protocol_helpers import capability_config_ref as _config_ref, protocol_outputs as _outputs
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.planning.source_qualification import PredecessorBoundSourceQualificationExperiment
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationRecord
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolStepTemplate, ScientificStage
from empirical_lawhood.runtime.source_qualification import PredecessorBoundSourceQualificationSubstrateBinding, derive_source_qualification_topology

from .discovery import EVALUATION_CAPABILITY, PROJECTION_CAPABILITY, SOURCE_CAPABILITY
from .roster import Q_CLOCK, Q_RECEIVERS, preparation_policy_declarations, preparation_policy_substrate_binding


def validate_preparation_policy_source_records(
    config: FiniteResponseLawPreparationPolicyNativeConfig,
    carrier: PredecessorBoundSourceQualificationExperiment,
    association: PredecessorBoundSourceQualificationSubstrateBinding,
    installed_binding: ObjectIdentity,
) -> None:
    declarations = preparation_policy_declarations(config)
    retained = tuple(row.declaration for row in config.retained_prefixes)
    if (
        carrier.source_capability
        != ObjectIdentity.from_record(SOURCE_CAPABILITY.capability_key, SOURCE_CAPABILITY)
        or carrier.source_config != ObjectIdentity.from_record(config.spec_id, config)
        or (carrier.physical_units, carrier.segments, carrier.views)
        != (declarations.units, declarations.segments, declarations.views)
        or carrier.retained_predecessors != retained
        or carrier.receiver_ids != Q_RECEIVERS
        or carrier.clock_ids != (Q_CLOCK,)
        or carrier.evaluator_task_id != PREPARATION_POLICY_EVALUATOR_TASK_ID
        or carrier.evidence_ceiling is not EvidenceCeiling.RESPONSE
        or association != preparation_policy_substrate_binding(config, carrier, installed_binding)
    ):
        raise ValueError("preparation-policy development source records change units, clocks, panels, or evaluator")


def preparation_policy_protocol_steps(
    config: FiniteResponseLawPreparationPolicyNativeConfig,
    carrier: PredecessorBoundSourceQualificationExperiment,
    projection: FiniteResponseLawPreparationPolicyProjectionConfig,
    evaluation: FiniteResponseLawPreparationPolicyScreenConfig,
) -> tuple[ProtocolStepTemplate, ...]:
    if (
        projection.native_spec != config
        or evaluation.projection != projection
        or carrier.projection_config != ObjectIdentity.from_record(projection.config_id, projection)
        or carrier.evaluator_config != ObjectIdentity.from_record(evaluation.config_id, evaluation)
    ):
        raise ValueError("preparation-policy development protocol changes projection/evaluator configuration")
    owners = {
        ObjectIdentity.from_record(manifest.capability_key, manifest): (manifest, value, identity)
        for manifest, value, identity in (
            (SOURCE_CAPABILITY, config, carrier.source_config),
            (PROJECTION_CAPABILITY, projection, carrier.projection_config),
            (EVALUATION_CAPABILITY, evaluation, carrier.evaluator_config),
        )
    }
    outputs = {
        ScientificStage.PREPARE: _outputs(
            (
                ("native-result", FiniteResponseLawPreparationPolicyNativeTaskResult.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            ),
            ("native-observations", PREPARATION_POLICY_NATIVE_PAIR_SCHEMA),
        ),
        ScientificStage.TRANSFORM: _outputs(
            (
                ("root-panel", FiniteResponseLawPreparationPolicyRootPanel.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
            )
        ),
        ScientificStage.EVALUATE: _outputs(
            (
                ("screen-result", FiniteResponseLawPreparationScreenResult.SCHEMA),
                ("upper-freeze", FiniteResponseLawPreparationPolicyUpperPayloadFreeze.SCHEMA),
                ("stage-envelope", LinkedCampaignStageEnvelope.SCHEMA),
                ("scientific-adjudication", ScientificAdjudicationRecord.SCHEMA),
            )
        ),
    }
    native = {task.task_id: task for task in preparation_policy_native_invocations(config)}
    retained = {row.declaration.segment_id: row.declaration for row in config.retained_prefixes}
    stages = derive_source_qualification_topology(carrier).stages
    output_bounds = {
        stage.task_id: (
            16 * 1024**2
            if stage.stage is ScientificStage.PREPARE
            else 8 * 1024**2
            if stage.stage is ScientificStage.TRANSFORM
            else 16 * 1024**2
        )
        for stage in stages
    }
    steps = []
    for stage in stages:
        manifest, value, identity = owners[stage.owner]
        if stage.config != identity:
            raise ValueError("preparation-policy development protocol stage changes exact config identity")
        scan = len(value.canonical_bytes())
        scan += sum(output_bounds[dependency] for dependency in stage.dependency_task_ids)
        if stage.stage is ScientificStage.PREPARE:
            prior = retained.get(native[stage.task_id].predecessor_segment_id or "")
            if prior is not None:
                scan += sum(
                    artifact.size_bytes for artifact in (*prior.artifacts, prior.task_receipt)
                )
            if not stage.dependency_task_ids:
                scan += len(value.canonical_bytes())
        elif stage.stage is ScientificStage.EVALUATE:
            scan += sum(
                artifact.size_bytes
                for artifact in (
                    evaluation.lower_artifact,
                    evaluation.lower_qualification,
                    evaluation.prospective_adjudication,
                    evaluation.prospective_closeout,
                )
            )
        if scan > manifest.resource_ceiling.source_scan_bytes:
            raise ValueError("preparation-policy development protocol exceeds complete declared input scan bound")
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
                else OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.OUTCOME_VISIBLE,
                replace(
                    manifest.resource_ceiling,
                    source_scan_bytes=scan,
                    output_bytes=output_bounds[stage.task_id],
                ),
                (),
                BarrierKind.REVEAL
                if stage.stage is ScientificStage.EVALUATE
                else BarrierKind.FREEZE
                if stage.stage is ScientificStage.TRANSFORM
                else BarrierKind.NONE,
                1,
                ("finite-response-law.preparation-policy.single-terminal",)
                if stage.stage is ScientificStage.EVALUATE
                else (f"custody.{stage.task_id}",),
            )
        )
    if len(steps) != 4_129:
        raise ValueError("preparation-policy development protocol differs from 4,104 source + 24 projection + one screen")
    return tuple(sorted(steps, key=lambda value: value.step_id))

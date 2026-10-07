"fresh calibration per-view projection and protected simultaneous-calibration provider."

from dataclasses import replace
from typing import cast

from empirical_lawhood.adapters.simulators.prepared_response.native_pair import MAXIMUM_PAIR_BYTES, NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.policy_native import CALIBRATION_POLICIES, PreparedResponseCalibrationNativeTaskResult
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, envelope, output_result, read_port, semantic_contracts
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner, WorkerInputKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .calibration_records import PreparedResponseCalibrationCalibratedLibrary, PreparedResponseCalibrationCalibrationConfig, PreparedResponseCalibrationProjectionConfig, PreparedResponseCalibrationViewProjection, calibrate_prepared_response_calibration, project_prepared_response_calibration_view
from .policy_decision import PreparedParentDecision
from .stage_envelopes import prepared_response_calibration_native_stage_envelope, prepared_parent_decision_envelope


def prepared_response_calibration_projection_envelope(
    report: PreparedResponseCalibrationViewProjection,
) -> LinkedCampaignStageEnvelope:
    stage = envelope(
        report.report_id,
        report,
        report.report_id,
        LinkedCampaignStageRole.EVIDENCE_PROJECTION,
        report.disposition == "COMPLETE",
        () if report.disposition == "COMPLETE" else (report.disposition,),
    )
    return (
        stage
        if report.disposition == "COMPLETE"
        else replace(stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE)
    )


def _config(context: TaskContext, config: CanonicalRecord, manifest: CapabilityManifest) -> None:
    external = tuple(value for value in context.input_ports if value.kind is WorkerInputKind.EXTERNAL)
    if (
        len(external) != 1
        or external[0].artifact_id != f"config-artifact.{getattr(config, 'config_id')}"
        or external[0].payload_schema != config.SCHEMA
        or decode_port(external[0], type(config), maximum=16 * 1024**2) != config
        or context.config.config_id != getattr(config, "config_id")
        or context.config.config_schema != config.SCHEMA
        or context.config.config_schema_sha256 != manifest.config_schema_sha256
        or context.config.content_sha256 != config.fingerprint()
    ):
        raise ValueError("prepared fresh calibration method changes its exact configuration")


class PreparedResponseCalibrationProjectionTask:
    def __init__(self, manifest: CapabilityManifest, config: PreparedResponseCalibrationProjectionConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            slots = {
                f"{root.root_id}.policy.{policy}.project.r{refinement}": (
                    root,
                    policy,
                    refinement,
                )
                for root in self.config.native_spec.roots
                for policy in CALIBRATION_POLICIES
                for refinement in (1, 2)
            }
            selected = slots.get(context.task_id)
            if selected is None:
                raise ValueError("prepared fresh calibration projection is outside its root/policy/view roster")
            _config(context, self.config, self.manifest)
            source_inputs = []
            decision = None
            for receipt in context.dependency_receipts:
                ports = tuple(
                    value
                    for value in context.input_ports
                    if value.materialization_id in receipt.output_materialization_ids
                )
                by_schema = {value.payload_schema: value for value in ports}
                if set(by_schema) == {
                    PreparedResponseCalibrationNativeTaskResult.SCHEMA,
                    NATIVE_PAIR_SCHEMA,
                    LinkedCampaignStageEnvelope.SCHEMA,
                } and len(ports) == 3:
                    result = decode_port(
                        by_schema[PreparedResponseCalibrationNativeTaskResult.SCHEMA],
                        PreparedResponseCalibrationNativeTaskResult,
                    )
                    stage = decode_port(
                        by_schema[LinkedCampaignStageEnvelope.SCHEMA],
                        LinkedCampaignStageEnvelope,
                    )
                    if (
                        result.result_id != f"{receipt.task_id}.result"
                        or stage != prepared_response_calibration_native_stage_envelope(result)
                    ):
                        raise ValueError("prepared fresh calibration projection received detached native input")
                    source_inputs.append(
                        (
                            result,
                            read_port(by_schema[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES),
                        )
                    )
                elif set(by_schema) == {
                    PreparedParentDecision.SCHEMA,
                    LinkedCampaignStageEnvelope.SCHEMA,
                } and len(ports) == 2:
                    if decision is not None:
                        raise ValueError("prepared fresh calibration projection repeats its parent decision")
                    decision = decode_port(
                        by_schema[PreparedParentDecision.SCHEMA], PreparedParentDecision
                    )
                    stage = decode_port(
                        by_schema[LinkedCampaignStageEnvelope.SCHEMA],
                        LinkedCampaignStageEnvelope,
                    )
                    if (
                        decision.decision_id != f"{receipt.task_id}.result"
                        or stage != prepared_parent_decision_envelope(decision)
                    ):
                        raise ValueError("prepared fresh calibration projection received detached parent decision")
                else:
                    raise ValueError("prepared fresh calibration projection dependency changes output schemas")
            root, policy, refinement = selected
            if decision is None or decision.root != root or decision.policy_id != policy:
                raise ValueError("prepared fresh calibration projection omits its exact parent decision")
            if decision.config != ObjectIdentity.from_record(
                self.config.policy_config.config_id, self.config.policy_config
            ):
                raise ValueError("prepared fresh calibration projection received another policy freeze")
            report = project_prepared_response_calibration_view(
                self.config,
                root,
                policy,
                refinement,
                decision,
                tuple(source_inputs),
            )
            stage = prepared_response_calibration_projection_envelope(report)
            return output_result(
                context,
                {report.SCHEMA: report.canonical_bytes(), stage.SCHEMA: stage.canonical_bytes()},
                (
                    "complete-policy-chart-retained",
                    "paired-views-one-physical-root",
                    "same-innovation-native-hold-preservation",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


def prepared_response_calibration_calibration_adjudication(
    context: TaskContext, result: PreparedResponseCalibrationCalibratedLibrary
) -> ScientificAdjudicationRecord:
    authority = context.scientific_adjudication_context
    if authority is None:
        raise ValueError("prepared fresh calibration calibration lacks authenticated adjudication context")
    calibrated = result.disposition == "CALIBRATED_FOR_C_R"
    return ScientificAdjudicationRecord(
        f"adjudication.{context.run_id}.{context.task_id}",
        context.run_id,
        context.task_id,
        authority.execution_plan,
        context.input_materialization_ids,
        tuple(
            sorted(
                value.logical_artifact_id
                for value in context.output_ports
                if value.logical_artifact_id is not None
            )
        ),
        context.dependency_receipt_ids,
        authority.evidence_world_id,
        authority.evidence_world_kind,
        authority.relation,
        authority.independent_unit_id,
        authority.information_cutoffs,
        authority.visibility_ceiling,
        authority.outcome_access,
        AdjudicationEvaluability.EVALUABLE,
        ScientificStatus.NOT_TESTED if calibrated else ScientificStatus.NOT_SUPPORTED,
        AdmissionStatus.NOT_EVALUATED,
        ("FRESH_RESPONSE_CALIBRATION_ONLY_NOT_RESPONSE_LAW_QUALIFICATION",)
        if calibrated
        else ("FRESH_RESPONSE_CALIBRATION_SIMULTANEOUS_MULTIPLIER_NOT_FINITE",),
        authority.fixture_scope_id,
        authority.plumbing_only,
    )


class PreparedResponseCalibrationCalibrationTask:
    TASK_ID = "prepared-response.fresh-response-calibration.calibrate-simultaneous-library"

    def __init__(
        self, manifest: CapabilityManifest, config: PreparedResponseCalibrationCalibrationConfig
    ) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            if context.task_id != self.TASK_ID:
                raise ValueError("prepared fresh calibration calibration uses another terminal task")
            _config(context, self.config, self.manifest)
            reports = []
            for receipt in context.dependency_receipts:
                ports = tuple(
                    value
                    for value in context.input_ports
                    if value.materialization_id in receipt.output_materialization_ids
                )
                by_schema = {value.payload_schema: value for value in ports}
                if set(by_schema) != {
                    PreparedResponseCalibrationViewProjection.SCHEMA,
                    LinkedCampaignStageEnvelope.SCHEMA,
                } or len(ports) != 2:
                    raise ValueError("prepared fresh calibration calibration dependency changes output schemas")
                report = decode_port(
                    by_schema[PreparedResponseCalibrationViewProjection.SCHEMA], PreparedResponseCalibrationViewProjection
                )
                stage = decode_port(
                    by_schema[LinkedCampaignStageEnvelope.SCHEMA],
                    LinkedCampaignStageEnvelope,
                )
                if report.report_id != receipt.task_id or stage != prepared_response_calibration_projection_envelope(
                    report
                ):
                    raise ValueError("prepared fresh calibration calibration received detached projection")
                reports.append(report)
            expected = tuple(
                (root, policy, refinement)
                for root in self.config.projection.native_spec.roots
                for policy in CALIBRATION_POLICIES
                for refinement in (1, 2)
            )
            ordered = tuple(sorted(reports, key=lambda value: value.report_id))
            by_key = {
                (value.root, value.policy_id, value.refinement): value for value in ordered
            }
            if set(by_key) != set(expected) or len(ordered) != len(expected):
                raise ValueError("prepared fresh calibration calibration omits an assigned root/policy/view")
            result = calibrate_prepared_response_calibration(
                self.config, tuple(by_key[value] for value in expected)
            )
            complete = result.disposition == "CALIBRATED_FOR_C_R"
            stage = envelope(
                context.task_id,
                result,
                result.library_id,
                LinkedCampaignStageRole.METHOD_IDENTIFICATION,
                complete,
                () if complete else ("FRESH_RESPONSE_CALIBRATION_SIMULTANEOUS_MULTIPLIER_NOT_FINITE",),
            )
            if not complete:
                stage = replace(stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE)
            adjudication = prepared_response_calibration_calibration_adjudication(context, result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                    adjudication.SCHEMA: adjudication.canonical_bytes(),
                },
                (
                    "ninety-six-root-context-calibration",
                    "maximum-policy-view-word-readout-output-score",
                    "invalid-required-coordinate-is-infinite",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class PreparedResponseCalibrationMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: PreparedResponseCalibrationProjectionConfig | PreparedResponseCalibrationCalibrationConfig,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("prepared fresh calibration method provider changes registry/configuration")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("prepared fresh calibration method runner registry/records differ")
        runner = (
            PreparedResponseCalibrationProjectionTask(self.manifest, self.config)
            if isinstance(self.config, PreparedResponseCalibrationProjectionConfig)
            else PreparedResponseCalibrationCalibrationTask(self.manifest, self.config)
        )
        return (cast(TaskRunner, runner),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("prepared fresh calibration method execution registry/records differ")
        expected = (
            {
                f"{root.root_id}.policy.{policy}.project.r{refinement}"
                for root in self.config.native_spec.roots
                for policy in CALIBRATION_POLICIES
                for refinement in (1, 2)
            }
            if isinstance(self.config, PreparedResponseCalibrationProjectionConfig)
            else {PreparedResponseCalibrationCalibrationTask.TASK_ID}
        )
        return config_payloads(
            plan, self.manifest, self.config, self.config.config_id, expected
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("prepared fresh calibration method semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (PreparedResponseCalibrationViewProjection, LinkedCampaignStageEnvelope)
            if isinstance(self.config, PreparedResponseCalibrationProjectionConfig)
            else (
                PreparedResponseCalibrationCalibratedLibrary,
                LinkedCampaignStageEnvelope,
                ScientificAdjudicationRecord,
            ),
            None,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("prepared fresh calibration method adjudication registry differs")
        return (
            ScientificAdjudicationOutputContract(
                capability_key=self.manifest.capability_key,
                capability_version=self.manifest.capability_version,
                output_id=f"{PreparedResponseCalibrationCalibrationTask.TASK_ID}.scientific-adjudication",
                payload_schema=ScientificAdjudicationRecord.SCHEMA,
            )
            if isinstance(self.config, PreparedResponseCalibrationCalibrationConfig)
            else None
        )

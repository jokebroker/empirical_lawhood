"""Registered runtime provider for target-blind prepared parent decisions."""

from typing import cast

from empirical_lawhood.adapters.simulators.prepared_response.policy_native import CALIBRATION_POLICIES, PreparedResponseCalibrationNativeTaskResult
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, output_result, semantic_contracts
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner, WorkerInputKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .policy_decision import PreparedParentDecision, PreparedPolicyDecisionConfig, prepare_parent_decision
from .stage_envelopes import prepared_response_calibration_native_stage_envelope, prepared_parent_decision_envelope


class PreparedPolicyDecisionTask:
    def __init__(
        self, manifest: CapabilityManifest, config: PreparedPolicyDecisionConfig
    ) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            slots = {
                f"{root.root_id}.policy.{policy}.decision": (root, policy)
                for root in self.config.native_spec.roots
                for policy in CALIBRATION_POLICIES
            }
            selected = slots.get(context.task_id)
            if selected is None:
                raise ValueError("prepared policy decision task is outside its root/policy roster")
            external = tuple(
                value for value in context.input_ports if value.kind is WorkerInputKind.EXTERNAL
            )
            if (
                len(external) != 1
                or external[0].artifact_id != f"config-artifact.{self.config.config_id}"
                or external[0].payload_schema != self.config.SCHEMA
                or decode_port(
                    external[0], PreparedPolicyDecisionConfig, maximum=16 * 1024**2
                ) != self.config
                or context.config.config_id != self.config.config_id
                or context.config.config_schema != self.config.SCHEMA
                or context.config.config_schema_sha256 != self.manifest.config_schema_sha256
                or context.config.content_sha256 != self.config.fingerprint()
            ):
                raise ValueError("prepared policy decision changes its exact configuration")
            if len(context.dependency_receipts) != 1:
                raise ValueError("prepared policy decision requires one common-start predecessor")
            receipt = context.dependency_receipts[0]
            ports = tuple(
                value
                for value in context.input_ports
                if value.materialization_id in receipt.output_materialization_ids
            )
            by_schema = {value.payload_schema: value for value in ports}
            if len(ports) != 2 or set(by_schema) != {
                PreparedResponseCalibrationNativeTaskResult.SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
            }:
                raise ValueError("prepared policy decision changes prefix output custody")
            prefix = decode_port(
                by_schema[PreparedResponseCalibrationNativeTaskResult.SCHEMA],
                PreparedResponseCalibrationNativeTaskResult,
            )
            stage = decode_port(
                by_schema[LinkedCampaignStageEnvelope.SCHEMA],
                LinkedCampaignStageEnvelope,
            )
            root, policy = selected
            if (
                prefix.result_id != f"{receipt.task_id}.result"
                or prefix.invocation.phase != "prefix"
                or prefix.invocation.root != root
                or stage != prepared_response_calibration_native_stage_envelope(prefix)
            ):
                raise ValueError("prepared policy decision received another root's prefix")
            result = prepare_parent_decision(
                self.config, root, prefix.common_start, policy
            )
            stage = prepared_parent_decision_envelope(result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                },
                (
                    "target-blind-preparent-policy",
                    "development-frozen-adequacy-score",
                    "no-native-effect-in-policy-decision",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class PreparedPolicyDecisionProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: PreparedPolicyDecisionConfig,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("prepared policy provider changes its registry/configuration")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("prepared policy runner registry/records differ")
        return (cast(TaskRunner, PreparedPolicyDecisionTask(self.manifest, self.config)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("prepared policy execution registry/records differ")
        expected = {
            f"{root.root_id}.policy.{policy}.decision"
            for root in self.config.native_spec.roots
            for policy in CALIBRATION_POLICIES
        }
        return config_payloads(
            plan, self.manifest, self.config, self.config.config_id, expected
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("prepared policy semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (PreparedParentDecision, LinkedCampaignStageEnvelope),
            None,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("prepared policy adjudication registry differs")
        return None

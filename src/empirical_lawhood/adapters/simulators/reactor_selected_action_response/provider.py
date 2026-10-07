"""Native acquisition binding on the existing receipted campaign runner seam."""

from dataclasses import dataclass
from decimal import Decimal as D
import platform

import numpy as np

from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.reactor_selected_action_response.config import assignment
from empirical_lawhood.adapters.methods.reactor_selected_action_response.control_prospective_plan import ReactorSelectedActionResponseProspectivePlan
from empirical_lawhood.adapters.methods.reactor_selected_action_response.law_terminal import ClassicalLaw, RESULT_ID
from empirical_lawhood.adapters.methods.reactor_selected_action_response.provider import ClassicalControlCustody
from empirical_lawhood.adapters.methods.reactor_selected_action_response.provider_common import artifact, config_input, contracts, dependency, external_config, result, verify_registry
from empirical_lawhood.adapters.methods.reactor_selected_action_response.records import ClassicalCausal, ClassicalPrivate, ClassicalDecision, ClassicalQualification
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationOutputContract
from empirical_lawhood.runtime.execution import TaskContext, RunnerResult, TaskRunner, TaskProgressEmitter
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from .acquisition import acquire_preparation, acquire_assays
from .config import ClassicalNativeConfig
from .extension_bundle import CAPABILITY, OUTPUT_RECORDS
from .prospective_acquisition import acquire_prospective_root
from .prospective_records import ClassicalNativeRoot


@dataclass(frozen=True)
class ClassicalNativeRunner:
    config: ClassicalNativeConfig
    custody: DependencyCustodyReader
    limits: EmpiricalResourceGuard
    control: ClassicalControlCustody
    reader: CandidatePayloadReader
    manifest = CAPABILITY

    def execute(self, context: TaskContext) -> RunnerResult:
        return self.execute_with_progress(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter | None
    ) -> RunnerResult:
        with self.limits.task(context.task_id):
            config_input(context, self.config)
            ports = tuple(
                p for p in context.input_ports if p.payload_schema == EmpiricalStudySource.SCHEMA
            )
            if len(ports) != 1:
                raise ValueError("classical native task requires its exact source input")
            source = decode_canonical_bytes(
                ports[0].read(), EmpiricalStudySource, maximum_bytes=4 * 1024**2
            )
            if (
                source.batch.fingerprint() != self.config.source_sha256
                or platform.python_version() != "3.11.14"
                or np.__version__ != "2.4.6"
            ):
                raise ValueError("classical simulator source or numerical environment changed")
            kind, root = context.task_id.split(".", 2)[1:]
            role, _ = assignment(root)
            progress = None if emitter is None else lambda value: emitter.advance(D(value))
            if kind == "prepare":
                released = True
                if role == "prospective":
                    law, _, _ = dependency(context, self.custody, ClassicalLaw, "classical.law")
                    released = law.qualification.scientific_status is ScientificStatus.SUPPORTED
                return result(
                    context,
                    acquire_preparation(
                        self.config, source.batch, root, released=released, progress=progress
                    ),
                )
            causal, parent_manifest, parent_receipt = dependency(
                context, self.custody, ClassicalCausal, f"classical.prepare.{root}"
            )
            private, _, _ = dependency(
                context, self.custody, ClassicalPrivate, f"classical.prepare.{root}"
            )
            decision, _, _ = dependency(
                context, self.custody, ClassicalDecision, f"classical.decision.{root}"
            )
            if kind == "assay" and role == "qualification":
                return result(
                    context,
                    (acquire_assays(source.batch, causal, private, decision, progress=progress),),
                )
            if kind != "action" or role != "prospective":
                raise ValueError("classical source task changes its declared acquisition role")
            law, _, _ = dependency(context, self.custody, ClassicalLaw, "classical.law")
            qualification, q_manifest, _ = dependency(
                context, self.custody, ClassicalQualification, "classical.qualification"
            )
            plan, _, _ = dependency(context, self.custody, ReactorSelectedActionResponseProspectivePlan, "classical.prospective-evaluation-plan")
            if (
                qualification != law.operands
                or decision.causal_preparation_valid
                and (
                    law.qualification.scientific_status is not ScientificStatus.SUPPORTED
                    or decision.qualified_law != ObjectIdentity.from_record(RESULT_ID, law)
                )
            ):
                raise ValueError("classical native action lost its exact fresh qualified law")
            acquired = acquire_prospective_root(
                source=source.batch,
                causal=causal,
                private=private,
                assignment=decision,
                report=law,
                prospective_plan=plan,
                calibration_artifact=artifact(q_manifest),
                publisher=self.control.open_control_store(root),
                reader=self.reader,
                authority=self.control.authority,
                resources=self.control.resources,
                causal_receipt=parent_receipt,
                causal_artifact=artifact(parent_manifest),
                issued_study=self.control.issued_study,
                prepared_store=self.control.open_prepared_store(),
                occurred_at_utc=self.control.event_clock(root, "freeze"),
                progress=progress,
            )
            plan_id = (
                None
                if plan.plan is None or not decision.causal_preparation_valid
                else ObjectIdentity.from_record(plan.plan.evaluation_plan_id, plan.plan)
            )
            return result(
                context, (ClassicalNativeRoot.from_acquisition(decision, plan_id, acquired),)
            )


class ClassicalNativeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        runner: ClassicalNativeRunner,
        source_input: ExternalInputPayload,
    ) -> None:
        self.registry_sha256 = verify_registry(registry, CAPABILITY)
        self.capability_count, self.runner, self.source_input = 1, runner, source_input

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry.fingerprint() != self.registry_sha256 or source_records:
            raise ValueError("classical native registry/source changed")
        return (self.runner,)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("classical native plan/source changed")
        specs = {
            s.logical_artifact_id: s
            for task in plan.tasks
            for s in task.external_inputs
            if s.expected_payload_schema == EmpiricalStudySource.SCHEMA
        }
        if (
            set(specs) != {self.source_input.logical_artifact_id}
            or specs[self.source_input.logical_artifact_id].expected_content_sha256
            != self.source_input.logical_content_sha256
        ):
            raise ValueError("classical source payload differs from compiled source custody")
        return tuple(
            sorted(
                (*external_config(plan, CAPABILITY, self.runner.config), self.source_input),
                key=lambda v: v.logical_artifact_id,
            )
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self.runners(registry)
        return contracts(CAPABILITY, OUTPUT_RECORDS)

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        self.runners(registry)
        return None

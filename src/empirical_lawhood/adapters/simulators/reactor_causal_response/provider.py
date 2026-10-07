"""Installed empirical native task provider, using the shared executor and custody."""

from dataclasses import fields
import json
import platform
from decimal import Decimal
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from typing import Callable
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource, TimedReactorConfirmationEnvelope
from empirical_lawhood.adapters.methods.reactor_causal_response.experiment_records import EmpiricalQualificationTerminal
from .campaign import ControlCustodyPort, acquire_confirmation
import numpy as np
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    TaskContext,
    TaskRunner,
    RunnerResult,
    TaskOutputPayload,
    WorkerInputKind,
    TaskProgressEmitter,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    ExternalInputPayload,
    CapabilityOutputSemanticContract,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.experiment_records import EmpiricalAcquisitionEnvelope, EmpiricalDiscoveryEnvelope, EmpiricalCalibrationEnvelope
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader, validate_dependency
from empirical_lawhood.adapters.methods.reactor_causal_response.serialization import read_fit
from empirical_lawhood.adapters.methods.reactor_causal_response.transport import CausalReactorRootEnvelope
from empirical_lawhood.adapters.methods.reactor_causal_response.controller import NumericalController
from .config import EmpiricalNativeConfig, NATIVE_TASKS
from .design import scenario
from .interface import Actuator
from .acquisition import acquire_episode, development_root, TapeController, paired_operands
from .extension_bundle import CAPABILITY


class EmpiricalNativeRunner:
    manifest = CAPABILITY

    def __init__(
        self,
        config: EmpiricalNativeConfig,
        custody: DependencyCustodyReader,
        control: ControlCustodyPort,
        reader: CandidatePayloadReader,
        limits: EmpiricalResourceGuard,
    ) -> None:
        self.config, self.custody, self.control, self.reader = config, custody, control, reader
        self.limits = limits

    def execute(self, context: TaskContext) -> RunnerResult:
        return self._guarded(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        return self._guarded(context, lambda n: emitter.advance(Decimal(n)))

    def _guarded(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        with self.limits.task(context.task_id):
            return self._execute(context, progress)

    def _execute(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        assignments = {task: (role, i) for task, role, i in NATIVE_TASKS}
        if (
            context.task_id not in assignments
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != CAPABILITY.permissions
            or len(context.output_ports) != 1
        ):
            raise ValueError("native empirical task/config/authority/output differs")
        role, ordinal = assignments[context.task_id]
        expected_access = (
            OutcomeAccess.EVALUATION_SEALED
            if role in ("fit", "nomination")
            else OutcomeAccess.EVALUATOR_REVEAL
        )
        if context.outcome_access is not expected_access:
            raise ValueError("native empirical phase access differs")
        expected = {EmpiricalNativeConfig.SCHEMA, EmpiricalStudySource.SCHEMA}
        if role == "confirmation":
            expected.update(
                (EmpiricalDiscoveryEnvelope.SCHEMA, EmpiricalQualificationTerminal.SCHEMA)
            )
        if role == "calibration":
            expected.add(EmpiricalDiscoveryEnvelope.SCHEMA)
        if role == "qualification":
            expected.add(EmpiricalCalibrationEnvelope.SCHEMA)
        ports = {p.payload_schema: p for p in context.input_ports}
        if len(ports) != len(context.input_ports) or set(ports) != expected:
            raise ValueError("native empirical input roster differs")
        if (
            decode_canonical_bytes(
                ports[self.config.SCHEMA].read(), EmpiricalNativeConfig, maximum_bytes=32768
            )
            != self.config
        ):
            raise ValueError("native empirical config bytes differ")
        source_port = ports[EmpiricalStudySource.SCHEMA]
        if source_port.outcome_access is not OutcomeAccess.OUTCOME_BLIND:
            raise ValueError("native sources must be predeclared outcome-blind inputs")
        study = decode_canonical_bytes(
            source_port.read(), EmpiricalStudySource, maximum_bytes=1024**2
        )
        source = study.batch
        if (
            source.fingerprint() != self.config.source_sha256
            or platform.python_version() != self.config.python_version
            or np.__version__ != self.config.numpy_version
        ):
            raise ValueError("native source/runtime denominator changed")
        assignment = scenario(role, ordinal)
        if role == "confirmation":
            parents = []
            for record_type in (EmpiricalDiscoveryEnvelope, EmpiricalQualificationTerminal):
                port = ports[record_type.SCHEMA]
                if port.kind is not WorkerInputKind.DEPENDENCY:
                    raise ValueError("control parent must be an authenticated dependency")
                manifest, receipt = self.custody.read_dependency(context, port.binding)
                validate_dependency(port.binding, manifest, receipt)
                record = decode_canonical_bytes(
                    port.read(), record_type, maximum_bytes=256 * 1024**2
                )
                if record.fingerprint() != manifest.logical.content_sha256:
                    raise ValueError("control parent bytes differ from custody")
                parents.append(record)
            discovery, qualification = parents
            assert isinstance(discovery, EmpiricalDiscoveryEnvelope) and isinstance(
                qualification, EmpiricalQualificationTerminal
            )
            return self._output(
                context,
                acquire_confirmation(
                    source,
                    study.comparators,
                    discovery,
                    qualification,
                    ordinal,
                    custody=self.control,
                    reader=self.reader,
                    progress=progress,
                ),
            )
        if role in ("fit", "nomination"):
            index = ordinal if role == "fit" else 16 + ordinal
            episodes = development_root(source, assignment, index, progress=progress)
            output: CanonicalRecord = EmpiricalAcquisitionEnvelope(
                assignment.unit_id, role, tuple(e.envelope() for e in episodes)
            )
        else:
            schema = (
                EmpiricalDiscoveryEnvelope.SCHEMA
                if role == "calibration"
                else EmpiricalCalibrationEnvelope.SCHEMA
            )
            parent = ports[schema]
            if parent.kind is not WorkerInputKind.DEPENDENCY:
                raise ValueError("native policy must come from its authenticated frozen parent")
            manifest, receipt = self.custody.read_dependency(context, parent.binding)
            validate_dependency(parent.binding, manifest, receipt)
            if role == "calibration":
                discovery = decode_canonical_bytes(
                    parent.read(), EmpiricalDiscoveryEnvelope, maximum_bytes=256 * 1024**2
                )
                if discovery.fingerprint() != manifest.logical.content_sha256:
                    raise ValueError("frozen discovery custody differs")
                value = json.loads(discovery.result_json)
                chosen = value["selected"]
                if chosen is None:
                    return self._output(
                        context,
                        CausalReactorRootEnvelope.nonattempt(
                            assignment.unit_id, "IDENTIFICATION_NOT_SUPPORTED"
                        ),
                    )
                model = read_fit(value["fits"][chosen])
                q = float(value["nominations"][chosen]["q_dev"])
            else:
                calibration = decode_canonical_bytes(
                    parent.read(), EmpiricalCalibrationEnvelope, maximum_bytes=4 * 1024**2
                )
                if calibration.fingerprint() != manifest.logical.content_sha256:
                    raise ValueError("frozen calibration custody differs")
                if calibration.payload is None or calibration.payload.q is None:
                    reason = (
                        "IDENTIFICATION_NOT_SUPPORTED"
                        if calibration.payload is None
                        else "CALIBRATION_UNUSABLE"
                    )
                    return self._output(
                        context, CausalReactorRootEnvelope.nonattempt(assignment.unit_id, reason)
                    )
                model = read_fit(json.loads(calibration.payload.model_json))
                q = float(calibration.payload.q)
            nominal = acquire_episode(
                source,
                assignment,
                role,
                NumericalController(model, q, Actuator()),
                progress=progress,
            )
            refined = acquire_episode(
                source,
                assignment,
                role,
                TapeController(nominal.requests),
                dt=0.5,
                progress=None
                if progress is None
                else lambda n: progress(len(nominal.requests) + n),
            )
            output = CausalReactorRootEnvelope.pack(paired_operands(nominal, refined, model, q))
        return self._output(context, output)

    @staticmethod
    def _output(context: TaskContext, output: CanonicalRecord) -> RunnerResult:
        if context.output_ports[0].payload_schema != output.SCHEMA:
            raise ValueError("native empirical output schema differs")
        return RunnerResult(
            (TaskOutputPayload(context.output_ports[0].output_id, output.canonical_bytes()),),
            (ReceiptCheck("native-empirical-assignment-and-paired-exposure", True, ()),),
        )


class EmpiricalNativeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        config: EmpiricalNativeConfig,
        source: ExternalInputPayload,
        custody: DependencyCustodyReader,
        control: ControlCustodyPort,
        reader: CandidatePayloadReader,
        limits: EmpiricalResourceGuard,
    ) -> None:
        if registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version) != CAPABILITY:
            raise ValueError("native empirical installed manifest differs")
        if (
            source.payload_schema != EmpiricalStudySource.SCHEMA
            or source.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("native empirical source identity/visibility differs")
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.limits = limits
        self.config, self.source, self.custody, self.control, self.reader = (
            config,
            source,
            custody,
            control,
            reader,
        )

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry.fingerprint() != self.registry_sha256 or source_records:
            raise ValueError("native empirical registry/source differs")
        return (
            EmpiricalNativeRunner(
                self.config, self.custody, self.control, self.reader, self.limits
            ),
        )

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("native empirical plan/source differs")
        tasks = tuple(
            t for t in plan.tasks if t.capability.capability_key == CAPABILITY.capability_key
        )
        if not tasks or not {t.task_id for t in tasks}.issubset({t for t, _, _ in NATIVE_TASKS}):
            raise ValueError("undeclared native empirical acquisition task")
        specs = {s.logical_artifact_id: s for t in tasks for s in t.external_inputs}
        config_specs = [
            s for s in specs.values() if s.expected_payload_schema == self.config.SCHEMA
        ]
        source_specs = [
            s for s in specs.values() if s.expected_payload_schema == self.source.payload_schema
        ]
        if len(config_specs) != 1 or len(source_specs) != 1 or len(specs) != 2:
            raise ValueError("native empirical exact external input roster differs")
        config, source = config_specs[0], source_specs[0]
        if (
            config.expected_content_sha256 != self.config.fingerprint()
            or source.expected_content_sha256 != self.source.logical_content_sha256
            or source.logical_artifact_id != self.source.logical_artifact_id
        ):
            raise ValueError("native empirical external input content differs")
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.config.config_id, self.config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        payload = ExternalInputPayload.from_bytes(
            logical_artifact_id=config.logical_artifact_id,
            payload_schema=self.config.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            media_type="application/json",
            payload=self.config.canonical_bytes(),
            visibility_ceiling=parent.visibility_ceiling,
            outcome_access=parent.outcome_access,
            parent_visibility_ceilings=(parent.visibility_ceiling,),
            lineage_parents=(parent,),
            logical_content_sha256=self.config.fingerprint(),
        )
        return tuple(sorted((payload, self.source), key=lambda p: p.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("native empirical registry differs")
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                CAPABILITY,
                payload_schema=record.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=record.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(f.name for f in fields(record))),
            )
            for record in (
                EmpiricalAcquisitionEnvelope,
                CausalReactorRootEnvelope,
                TimedReactorConfirmationEnvelope,
            )
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("native empirical registry differs")
        return None

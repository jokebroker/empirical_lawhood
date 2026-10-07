"""Installed native runner; no direct operator or private execution route."""

import platform
import numpy as np
from dataclasses import fields
from decimal import Decimal as D
from typing import Callable

from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader, validate_dependency
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.config import ROOTS, SOURCE_TASKS
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.records import LocalRootEvidence
from empirical_lawhood.adapters.methods.reactor_local_domain_qualification.scoring import LocalCalibration
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
    TaskProgressEmitter,
    WorkerInputKind,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from .config import LocalNativeConfig
from .extension_bundle import CAPABILITY
from .acquisition import acquire_local_root


class LocalNativeRunner:
    manifest = CAPABILITY

    def __init__(
        self,
        config: LocalNativeConfig,
        custody: DependencyCustodyReader,
        limits: EmpiricalResourceGuard,
    ) -> None:
        self.config, self.custody, self.limits = config, custody, limits

    def execute(self, context: TaskContext) -> RunnerResult:
        return self._guarded(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        return self._guarded(context, lambda count: emitter.advance(D(count)))

    def _guarded(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        with self.limits.task(context.task_id):
            return self._execute(context, progress)

    def _execute(
        self, context: TaskContext, progress: Callable[[int], None] | None
    ) -> RunnerResult:
        if (
            context.task_id not in SOURCE_TASKS
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != CAPABILITY.permissions
            or len(context.output_ports) != 1
        ):
            raise ValueError("local native task/config/permission/output differs")
        root = context.task_id.removeprefix("local.")
        role = next(role for r, role, _, _ in ROOTS if r == root)
        ports = {p.payload_schema: p for p in context.input_ports}
        expected = {LocalNativeConfig.SCHEMA, EmpiricalStudySource.SCHEMA}
        if role == "qualification":
            expected.add(LocalCalibration.SCHEMA)
        if (
            set(ports) != expected
            or len(ports) != len(context.input_ports)
            or context.outcome_access
            not in (OutcomeAccess.EVALUATION_SEALED, OutcomeAccess.EVALUATOR_REVEAL)
        ):
            raise ValueError("local native input census/visibility differs")
        config = decode_canonical_bytes(
            ports[LocalNativeConfig.SCHEMA].read(), LocalNativeConfig, maximum_bytes=256 * 1024
        )
        if (
            config != self.config
            or platform.python_version() != config.python_version
            or np.__version__ != config.numpy_version
        ):
            raise ValueError("local native config payload differs")
        if role == "qualification":
            parent = ports[LocalCalibration.SCHEMA]
            if parent.kind is not WorkerInputKind.DEPENDENCY:
                raise ValueError("qualification lacks a persisted calibration barrier")
            manifest, receipt = self.custody.read_dependency(context, parent.binding)
            validate_dependency(parent.binding, manifest, receipt)
            calibrated = decode_canonical_bytes(
                parent.read(), LocalCalibration, maximum_bytes=4 * 1024**2
            )
            if (
                receipt.task_id != "local.calibration"
                or calibrated.fingerprint() != manifest.logical.content_sha256
                or calibrated.recipe
                != ObjectIdentity.from_record(config.design.config_id, config.design)
            ):
                raise ValueError("qualification changed calibration or recipe")
        source = decode_canonical_bytes(
            ports[EmpiricalStudySource.SCHEMA].read(),
            EmpiricalStudySource,
            maximum_bytes=4 * 1024**2,
        )
        if source.batch.fingerprint() != self.config.source_sha256:
            raise ValueError("local native medium pin differs")
        output = acquire_local_root(config.design, source.batch, root, progress=progress)
        if context.output_ports[0].payload_schema != output.SCHEMA:
            raise ValueError("local source output schema differs")
        return RunnerResult(
            (TaskOutputPayload(context.output_ports[0].output_id, output.canonical_bytes()),),
            (ReceiptCheck("local-native-assigned-causal-assays", True, ()),),
        )


class LocalNativeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        config: LocalNativeConfig,
        source: ExternalInputPayload,
        custody: DependencyCustodyReader,
        limits: EmpiricalResourceGuard,
    ) -> None:
        if (
            registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version) != CAPABILITY
            or source.payload_schema != EmpiricalStudySource.SCHEMA
            or source.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("local source registry or native-medium visibility differs")
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self.config, self.source = config, source
        self.runner = LocalNativeRunner(config, custody, limits)

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry.fingerprint() != self.registry_sha256 or source_records:
            raise ValueError("local native registry/source differs")
        return (self.runner,)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("local native plan/source differs")
        specs = {
            s.logical_artifact_id: s
            for t in plan.tasks
            if t.capability.capability_key == CAPABILITY.capability_key
            for s in t.external_inputs
        }
        configs = [s for s in specs.values() if s.expected_payload_schema == self.config.SCHEMA]
        sources = [
            s for s in specs.values() if s.expected_payload_schema == self.source.payload_schema
        ]
        if len(specs) != 2 or len(configs) != 1 or len(sources) != 1:
            raise ValueError("local native external input census differs")
        config, source = configs[0], sources[0]
        if (
            config.expected_content_sha256 != self.config.fingerprint()
            or source.expected_content_sha256 != self.source.logical_content_sha256
            or source.logical_artifact_id != self.source.logical_artifact_id
        ):
            raise ValueError("local native external input identity differs")
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
        self.runners(registry)
        return (
            CapabilityOutputSemanticContract.from_manifest(
                CAPABILITY,
                payload_schema=LocalRootEvidence.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=LocalRootEvidence.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(f.name for f in fields(LocalRootEvidence))),
            ),
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        self.runners(registry)
        return None

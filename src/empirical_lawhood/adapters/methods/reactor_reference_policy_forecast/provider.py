"""Closed campaign runner for the existing finite-action scientific owners."""

from dataclasses import fields
from typing import Protocol

from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
    ArtifactManifest,
    CanonicalTaskReceipt,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
    WorkerInputBinding,
    WorkerInputKind,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .science import ReactorForecastDesign, forecast_system
from .records import ReactorForecastResult, ReactorBatchCustody, QUALIFY, REVEAL
from .terminal import qualify_forecasts
from empirical_lawhood.adapters.methods.reactor_prefix_response.batch_calibration import ReactorForecastCalibration, score_unit
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_design import BRANCHES, native_task_id
from empirical_lawhood.adapters.simulators.reactor_prefix_response.batch_trace import ReactorBatchTrace, MAXIMUM_BATCH_BYTES
from empirical_lawhood.kernel.references import ArtifactIdentity
from .extension_bundle import CAPABILITY, OUTPUT_TYPES


class DependencyCustodyReader(Protocol):
    def read_dependency(
        self, context: TaskContext, binding: WorkerInputBinding
    ) -> tuple[ArtifactManifest, CanonicalTaskReceipt]: ...


class ForecastChainRunner:
    manifest = CAPABILITY

    def __init__(
        self,
        config: ReactorForecastDesign,
        publisher: CandidatePayloadPublisher,
        reader: CandidatePayloadReader,
        custody: DependencyCustodyReader,
    ) -> None:
        self.config, self.publisher, self.reader = config, publisher, reader
        self.custody = custody

    def execute(self, context: TaskContext) -> RunnerResult:
        if (
            context.task_id not in (QUALIFY, REVEAL)
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != self.manifest.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or len(context.output_ports) != 1
        ):
            raise ValueError("forecast task identity, permissions or output count differs")
        configs = tuple(p for p in context.input_ports if p.payload_schema == self.config.SCHEMA)
        if (
            len(configs) != 1
            or configs[0].outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or configs[0].size_bytes > 1024**2
        ):
            raise ValueError("forecast task requires its bounded outcome-blind design")
        if (
            decode_canonical_bytes(
                configs[0].read(), ReactorForecastDesign, maximum_bytes=1024**2
            )
            != self.config
        ):
            raise ValueError("forecast task substitutes design bytes")
        if context.task_id == QUALIFY:
            record: CanonicalRecord = self._qualify(context, configs[0].artifact_id)
        else:
            record = self._reveal(context)
        if context.output_ports[0].payload_schema != record.SCHEMA:
            raise ValueError("forecast task output schema differs")
        return RunnerResult(
            (TaskOutputPayload(context.output_ports[0].output_id, record.canonical_bytes()),),
            (ReceiptCheck("reactor-forecast-existing-owner-and-exact-custody", True, ()),),
        )

    def _qualify(self, context: TaskContext, design_artifact_id: str) -> ReactorForecastResult:
        if len(context.input_ports) != 85:
            raise ValueError("forecast qualification requires all 84 native traces")
        # Read and score one complete paired unit at a time; never retain 84 grids.
        by_task = {}
        for port in context.input_ports:
            if port.payload_schema == self.config.SCHEMA:
                continue
            if (
                port.kind is not WorkerInputKind.DEPENDENCY
                or port.payload_schema != ReactorBatchTrace.SCHEMA
                or port.outcome_access is not OutcomeAccess.EVALUATION_SEALED
                or port.size_bytes > MAXIMUM_BATCH_BYTES
            ):
                raise ValueError("forecast qualification changes a sealed dependency port")
            manifest, receipt = self.custody.read_dependency(context, port.binding)
            if receipt.task_id in by_task:
                raise ValueError("duplicate assigned native task")
            by_task[receipt.task_id] = (port, manifest, receipt)
        if set(by_task) != {native_task_id(*b) for b in BRANCHES}:
            raise ValueError("forecast qualification changes assigned native task roster")
        ordered = tuple(by_task[native_task_id(*b)] for b in BRANCHES)
        custody = ReactorBatchCustody(
            tuple(m for _, m, _ in ordered), tuple(r for _, _, r in ordered)
        )
        scores = []
        source_sha = None
        for scenario in self.config.native.scenarios:
            pair = []
            for view in ("native", "refined"):
                port, manifest, _ = by_task[native_task_id(scenario.unit_id, view)]
                raw = port.read()
                trace = decode_canonical_bytes(
                    raw, ReactorBatchTrace, maximum_bytes=MAXIMUM_BATCH_BYTES
                )
                if (
                    trace.scenario != scenario
                    or trace.fingerprint() != manifest.logical.content_sha256
                    or len(raw) != manifest.materialization.size_bytes
                ):
                    raise ValueError("forecast trace bytes or assignment differ from custody")
                if trace.source_sha256 != self.config.native.source_bundle_sha256 or (
                    source_sha is not None and source_sha != trace.source_sha256
                ):
                    raise ValueError("forecast traces mix source bundles")
                source_sha = trace.source_sha256
                pair.append(trace)
            scores.append(score_unit(pair[0], pair[1]))
        assert source_sha is not None
        calibration = ReactorForecastCalibration(tuple(scores))
        design_artifact = ArtifactIdentity(
            design_artifact_id,
            "forecast-design",
            self.config.SCHEMA,
            self.config.fingerprint(),
            "application/vnd.empirical-lawhood.canonical+json",
            len(self.config.canonical_bytes()),
        )
        return qualify_forecasts(
            self.config,
            calibration,
            custody,
            source_sha,
            self.manifest,
            self.publisher,
            self.reader,
            design_artifact,
        )

    def _reveal(self, context: TaskContext) -> ScientificAdjudicationRecord:
        ports = tuple(
            p for p in context.input_ports if p.payload_schema == ReactorForecastResult.SCHEMA
        )
        if (
            len(context.input_ports) != 2
            or len(ports) != 1
            or ports[0].kind is not WorkerInputKind.DEPENDENCY
            or ports[0].size_bytes > 16 * 1024**2
        ):
            raise ValueError("forecast reveal requires its single exact qualification parent")
        manifest, receipt = self.custody.read_dependency(context, ports[0].binding)
        result = decode_canonical_bytes(
            ports[0].read(), ReactorForecastResult, maximum_bytes=16 * 1024**2
        )
        if (
            receipt.task_id != QUALIFY
            or result.design != ObjectIdentity.from_record(self.config.config_id, self.config)
            or result.fingerprint() != manifest.logical.content_sha256
            or manifest.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("forecast reveal substitutes its prospective qualified parent")
        a = context.scientific_adjudication_context
        system = forecast_system(self.config)
        if (
            a is None
            or a.relation
            != ObjectIdentity.from_record(system.relation.relation_id, system.relation)
            or a.independent_unit_id != system.independent_unit.unit_id
        ):
            raise ValueError("forecast reveal requires its compiled scientific context")
        return ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=a.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=tuple(
                sorted(
                    p.logical_artifact_id
                    for p in context.output_ports
                    if p.logical_artifact_id is not None
                )
            ),
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=a.evidence_world_id,
            evidence_world_kind=a.evidence_world_kind,
            relation=a.relation,
            independent_unit_id=a.independent_unit_id,
            information_cutoffs=a.information_cutoffs,
            visibility_ceiling=a.visibility_ceiling,
            outcome_access=a.outcome_access,
            evaluability=AdjudicationEvaluability.UNEVALUABLE
            if result.scientific_status is ScientificStatus.UNEVALUABLE
            else AdjudicationEvaluability.EVALUABLE,
            scientific_status=result.scientific_status,
            admission_status=AdmissionStatus.UNEVALUABLE
            if result.scientific_status is ScientificStatus.UNEVALUABLE
            else AdmissionStatus.NOT_EVALUATED,
            reason_codes=("REACTOR_FORECAST_QUALIFICATION_EXISTING_OWNER_RESULT",),
        )


class ForecastChainProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        config: ReactorForecastDesign,
        publisher: CandidatePayloadPublisher,
        reader: CandidatePayloadReader,
        custody: DependencyCustodyReader,
    ) -> None:
        if registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version) != CAPABILITY:
            raise ValueError("finite chain installed capability differs")
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self._records: tuple[CanonicalRecord, ...] = (config,)
        self._runner = ForecastChainRunner(config, publisher, reader, custody)

    def _check(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("finite chain registry changed")

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        self._check(registry)
        if source_records:
            raise ValueError("finite chain does not acquire or reopen a source")
        return (self._runner,)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if source_records or plan.registry_sha256 != self.registry_sha256:
            raise ValueError("finite chain external input registry/source differs")
        by_schema = {r.SCHEMA: r for r in self._records}
        inputs: dict[str, ExternalInputPayload] = {}
        for task in plan.tasks:
            if task.capability.capability_key != CAPABILITY.capability_key:
                continue
            for spec in task.external_inputs:
                record = by_schema.get(spec.expected_payload_schema or "")
                if record is None or spec.expected_content_sha256 != record.fingerprint():
                    raise ValueError("finite chain external parent schema or content differs")
                identity = ObjectIdentity.from_record(spec.logical_artifact_id, record)
                parent = ArtifactLineageParent(
                    identity, VisibilityCeiling.PROSPECTIVE, OutcomeAccess.OUTCOME_BLIND
                )
                payload = ExternalInputPayload.from_bytes(
                    logical_artifact_id=spec.logical_artifact_id,
                    payload_schema=record.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    payload=record.canonical_bytes(),
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    parent_visibility_ceilings=(parent.visibility_ceiling,),
                    lineage_parents=(parent,),
                    logical_content_sha256=record.fingerprint(),
                )
                if (
                    spec.logical_artifact_id in inputs
                    and inputs[spec.logical_artifact_id] != payload
                ):
                    raise ValueError("finite chain external identity collision")
                inputs[spec.logical_artifact_id] = payload
        return tuple(inputs[k] for k in sorted(inputs))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self._check(registry)
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                CAPABILITY,
                payload_schema=t.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=t.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(f.name for f in fields(t))),
            )
            for t in OUTPUT_TYPES
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        self._check(registry)
        if execution_plan is None:
            raise ValueError("finite chain adjudication requires the exact compiled output locator")
        outputs = [
            o
            for t in execution_plan.tasks
            if t.capability.capability_key == CAPABILITY.capability_key
            for o in t.outputs
            if o.payload_schema == ScientificAdjudicationRecord.SCHEMA
        ]
        if len(outputs) != 1:
            raise ValueError("finite chain requires exactly one compiled adjudication output")
        return ScientificAdjudicationOutputContract(
            CAPABILITY.capability_key,
            CAPABILITY.capability_version,
            outputs[0].output_id,
            ScientificAdjudicationRecord.SCHEMA,
        )

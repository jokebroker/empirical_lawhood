"""Closed campaign runner for the existing finite-action scientific owners."""

from dataclasses import fields
from typing import Protocol

from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.simulators.reactor_prefix_response.panel import ReactorPrefixPanel
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
    ArtifactManifest,
    ArtifactProfile,
    CanonicalTaskReceipt,
    ReceiptCheck,
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

from .binding import bind_finite_chain
from .chain import identify_and_qualify
from .design import ReactorScienceDesign, reactor_system
from .extension_bundle import CAPABILITY, INPUT_TYPES, OUTPUT_TYPES
from .projection import ReactorNativeCustody, project_panel
from .result import ReactorScienceResult


class DependencyCustodyReader(Protocol):
    def read_dependency(
        self, context: TaskContext, binding: WorkerInputBinding
    ) -> tuple[ArtifactManifest, CanonicalTaskReceipt]: ...


class FiniteChainRunner:
    manifest = CAPABILITY
    input_types = INPUT_TYPES
    output_types = OUTPUT_TYPES
    panel_type = ReactorPrefixPanel
    result_type = ReactorScienceResult

    def native_custody(self, receipts):
        return ReactorNativeCustody(receipts)

    def __init__(
        self,
        config: ReactorScienceDesign,
        publisher: CandidatePayloadPublisher,
        reader: CandidatePayloadReader,
        custody: DependencyCustodyReader,
    ) -> None:
        self.config, self.publisher, self.reader = config, publisher, reader
        self.custody = custody

    def execute(self, context: TaskContext) -> RunnerResult:
        schemas = {t.SCHEMA: t for t in self.input_types}
        if (
            context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != self.manifest.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or len(context.input_ports) != 21
            or {p.payload_schema for p in context.input_ports} != set(schemas)
            or len(context.output_ports) != len(self.output_types)
            or {p.payload_schema for p in context.output_ports}
            != {t.SCHEMA for t in self.output_types}
        ):
            raise ValueError(
                "finite chain task identity, permissions or exact port roster differs"
            )
        adjudication = context.scientific_adjudication_context
        system = reactor_system(self.config)
        if (
            adjudication is None
            or not context.dependency_receipt_ids
            or any(p.logical_artifact_id is None for p in context.output_ports)
            or adjudication.evidence_world_id != system.world.world_id
            or adjudication.evidence_world_kind != system.world.kind
            or adjudication.relation
            != ObjectIdentity.from_record(system.relation.relation_id, system.relation)
            or adjudication.independent_unit_id != system.independent_unit.unit_id
        ):
            raise ValueError(
                "finite chain requires its exact scientific adjudication context"
            )
        decoded = {}
        for port in context.input_ports:
            if port.size_bytes > 8 * 1024**2 or port.outcome_access is not (
                OutcomeAccess.EVALUATION_SEALED
                if port.payload_schema == self.panel_type.SCHEMA
                else OutcomeAccess.OUTCOME_BLIND
            ):
                raise ValueError(
                    "finite chain input exceeds its bound or exposes protected outcomes"
                )
            decoded[port.artifact_id] = decode_canonical_bytes(
                port.read(), schemas[port.payload_schema], maximum_bytes=8 * 1024**2
            )
        design_ports = tuple(
            p for p in context.input_ports if p.payload_schema == self.config.SCHEMA
        )
        if (
            len(design_ports) != 1
            or decoded[design_ports[0].artifact_id] != self.config
        ):
            raise ValueError("finite chain task config bytes differ")
        units, receipts = [], []
        for panel_port in context.input_ports:
            if panel_port.payload_schema != self.panel_type.SCHEMA:
                continue
            unit_panel = decoded[panel_port.artifact_id]
            assert isinstance(unit_panel, self.panel_type)
            if (
                panel_port.kind is not WorkerInputKind.DEPENDENCY
                or unit_panel.branch is None
            ):
                raise ValueError(
                    "reactor science requires the twenty acquired dependency branches"
                )
            artifact, receipt = self.custody.read_dependency(
                context, panel_port.binding
            )
            units.append((unit_panel, artifact))
            receipts.append(receipt)
        units.sort(key=lambda pair: pair[0].branch or ("", "", ""))
        panel = self.panel_type(
            self.config.native, tuple(e for p, _ in units for e in p.episodes)
        )
        # This executed task receipt qualifies exact native input authentication,
        # numerical runtime and callback extraction. It is not a safety verdict.
        custody = self.native_custody(
            tuple(sorted(receipts, key=lambda r: r.receipt_id))
        )
        qualification_identity = ObjectIdentity.from_record(
            "tbs-reactor-native-custody", custody
        )
        projection = project_panel(
            design=self.config,
            panel=panel,
            unit_inputs=tuple(units),
            native_custody=custody,
            source_qualification=qualification_identity,
            runtime_qualification=qualification_identity,
            observer_qualification=qualification_identity,
        )
        if projection.extension is None:
            result = self.result_type(projection, None, None, None)
        else:
            config = bind_finite_chain(projection, self.publisher)
            identification, qualification = identify_and_qualify(
                config=config,
                projection=projection.projection,
                extension=projection.extension,
                publisher=self.publisher,
                reader=self.reader,
            )
            result = self.result_type(projection, config, identification, qualification)
        verdict = ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=adjudication.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=tuple(
                sorted(
                    p.logical_artifact_id
                    for p in context.output_ports
                    if p.logical_artifact_id is not None
                )
            ),
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=adjudication.evidence_world_id,
            evidence_world_kind=adjudication.evidence_world_kind,
            relation=adjudication.relation,
            independent_unit_id=adjudication.independent_unit_id,
            information_cutoffs=adjudication.information_cutoffs,
            visibility_ceiling=adjudication.visibility_ceiling,
            outcome_access=adjudication.outcome_access,
            evaluability=(
                AdjudicationEvaluability.UNEVALUABLE
                if result.scientific_status is ScientificStatus.UNEVALUABLE
                else AdjudicationEvaluability.EVALUABLE
            ),
            scientific_status=result.scientific_status,
            admission_status=(
                AdmissionStatus.UNEVALUABLE
                if result.scientific_status is ScientificStatus.UNEVALUABLE
                else AdmissionStatus.NOT_EVALUATED
            ),
            reason_codes=(
                ("REACTOR_PREFIX_NATIVE_DELIVERY_INCOMPLETE_RESPONSE_LOCAL_LAW_UNEVALUABLE",)
                if result.qualification is None
                else ("TBS_EXISTING_FINITE_ACTION_OWNER_RESULT",)
            ),
        )
        records: dict[str, CanonicalRecord] = {r.SCHEMA: r for r in (result, verdict)}
        return RunnerResult(
            tuple(
                TaskOutputPayload(
                    p.output_id, records[p.payload_schema].canonical_bytes()
                )
                for p in context.output_ports
            ),
            (ReceiptCheck("tbs-existing-scientific-owners", True, ()),),
        )


class FiniteChainProvider(CampaignRuntimeProvider):
    manifest = CAPABILITY
    output_types = OUTPUT_TYPES
    runner_type = FiniteChainRunner

    def __init__(
        self,
        registry: CapabilityRegistry,
        config: ReactorScienceDesign,
        publisher: CandidatePayloadPublisher,
        reader: CandidatePayloadReader,
        custody: DependencyCustodyReader,
    ) -> None:
        if (
            registry.resolve(
                self.manifest.capability_key, self.manifest.capability_version
            )
            != self.manifest
        ):
            raise ValueError("finite chain installed capability differs")
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self._records: tuple[CanonicalRecord, ...] = (config,)
        self._runner = self.runner_type(config, publisher, reader, custody)

    def _check(self, registry: CapabilityRegistry) -> None:
        if registry.fingerprint() != self.registry_sha256:
            raise ValueError("finite chain registry changed")

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
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
            if task.capability.capability_key != self.manifest.capability_key:
                continue
            for spec in task.external_inputs:
                record = by_schema.get(spec.expected_payload_schema or "")
                if (
                    record is None
                    or spec.expected_content_sha256 != record.fingerprint()
                ):
                    raise ValueError(
                        "finite chain external parent schema or content differs"
                    )
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
                self.manifest,
                payload_schema=t.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=t.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(f.name for f in fields(t))),
            )
            for t in self.output_types
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        self._check(registry)
        if execution_plan is None:
            raise ValueError(
                "finite chain adjudication requires the exact compiled output locator"
            )
        outputs = [
            o
            for t in execution_plan.tasks
            if t.capability.capability_key == self.manifest.capability_key
            for o in t.outputs
            if o.payload_schema == ScientificAdjudicationRecord.SCHEMA
        ]
        if len(outputs) != 1:
            raise ValueError(
                "finite chain requires exactly one compiled adjudication output"
            )
        return ScientificAdjudicationOutputContract(
            self.manifest.capability_key,
            self.manifest.capability_version,
            outputs[0].output_id,
            ScientificAdjudicationRecord.SCHEMA,
        )

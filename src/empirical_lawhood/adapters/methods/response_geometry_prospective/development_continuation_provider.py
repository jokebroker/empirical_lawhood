"""Nine existing development method runners over explicitly imported retained projections."""

from dataclasses import replace
from typing import Protocol, cast

from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, lineage_parent_sort_key
from empirical_lawhood.runtime.adjudication import ScientificAdjudicationOutputContract
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import TaskRunner, TaskContext, RunnerResult, WorkerInputKind
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan, ExecutionTask
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider, CapabilityOutputSemanticContract, ExternalInputPayload,
    ExternalInputSource, MAX_EXTERNAL_INPUT_CHUNK_BYTES,
)
from .development_continuation import ResponseGeometryDevelopmentAnalysisContinuationConfig, ResponseGeometryDevelopmentProjectionCustodyInput
from .development_provider import ResponseGeometryDevelopmentDevelopmentProvider, ResponseGeometryDevelopmentDevelopmentTask
from .development_assessment_provider import ResponseGeometryDevelopmentAssessmentProvider, ResponseGeometryDevelopmentAssessmentTask, ResponseGeometryDevelopmentEvaluationTask
from .development_closeout import ResponseGeometryDevelopmentCloseoutConfig
from .development_records import ResponseGeometryDevelopmentMethodConfig
from .development_terminal import ResponseGeometryDevelopmentQualificationConfig
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import decode_port
from .development_continuation import RETAINED_ANALYSIS_RUN


DEVELOPMENT_RETAINED_PROJECTION_SOURCE_PORT = "retained-response-geometry-analysis-retained-projection-sources"


class ResponseGeometryDevelopmentRetainedProjectionSourceFactory(Protocol):
    """Create a lazy source; only chunks() may access the authenticated artifact."""

    @property
    def continuation(self) -> ResponseGeometryDevelopmentAnalysisContinuationConfig: ...

    def create_source(self, declaration: ResponseGeometryDevelopmentProjectionCustodyInput) -> ExternalInputSource: ...


class ResponseGeometryDevelopmentAnalysisContinuationProvider(CampaignRuntimeProvider):
    def __init__(self, registry: CapabilityRegistry, manifest: CapabilityManifest,
                 config: ResponseGeometryDevelopmentMethodConfig | ResponseGeometryDevelopmentQualificationConfig | ResponseGeometryDevelopmentCloseoutConfig,
                 continuation: ResponseGeometryDevelopmentAnalysisContinuationConfig,
                 sources: ResponseGeometryDevelopmentRetainedProjectionSourceFactory | None,
                 payload_plane: CandidatePayloadPlane | None = None) -> None:
        expected = (continuation.qualification.method if isinstance(config, ResponseGeometryDevelopmentMethodConfig)
            else continuation.qualification if isinstance(config, ResponseGeometryDevelopmentQualificationConfig)
            else ResponseGeometryDevelopmentCloseoutConfig(config.config_id, continuation.qualification))
        if (config != expected or manifest.config_schema != config.SCHEMA
                or registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
                or ((sources is None) != isinstance(config, ResponseGeometryDevelopmentCloseoutConfig))
                or ((payload_plane is not None) != isinstance(config, ResponseGeometryDevelopmentQualificationConfig))):
            raise ValueError("development analysis provider substitutes the frozen science or required input ports")
        self.registry, self.manifest, self.config, self.continuation = registry, manifest, config, continuation
        self.sources, self.payload_plane = sources, payload_plane
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self._delegate: ResponseGeometryDevelopmentDevelopmentProvider | ResponseGeometryDevelopmentAssessmentProvider = (
            ResponseGeometryDevelopmentDevelopmentProvider(registry, manifest, config, "development")
            if isinstance(config, ResponseGeometryDevelopmentMethodConfig)
            else ResponseGeometryDevelopmentAssessmentProvider(registry, manifest, config, payload_plane))

    def runners(self, registry: CapabilityRegistry,
                source_records: tuple[CanonicalRecord, ...] = ()) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("development analysis runner registry/records differ")
        config = self.config
        if isinstance(config, ResponseGeometryDevelopmentMethodConfig):
            runner = ResponseGeometryDevelopmentDevelopmentTask(self.manifest, config, self.continuation)
        elif isinstance(config, ResponseGeometryDevelopmentQualificationConfig):
            assert self.payload_plane is not None
            return (cast(TaskRunner, ResponseGeometryDevelopmentAssessmentTask(self.manifest, config, self.payload_plane, self.continuation)),)
        else:
            return (cast(TaskRunner, ResponseGeometryDevelopmentEvaluationTask(self.manifest, config)),)
        return (cast(TaskRunner, runner),)

    def external_inputs(self, plan: ProtocolExecutionPlan,
                        source_records: tuple[CanonicalRecord, ...] = ()) -> tuple[ExternalInputPayload, ...]:
        if (source_records or plan.registry_sha256 != self.registry_sha256
                or tuple(sorted(task.task_id for task in plan.tasks)) != self.continuation.method_task_ids):
            raise ValueError("development analysis provider requires the complete nine-task non-native route")
        values = list(self._delegate.external_inputs(plan))
        tasks = tuple(task for task in plan.tasks if task.capability.capability_key == self.manifest.capability_key)
        declaration_id = self.continuation.artifact_id(self.continuation.run_id, self.continuation.config_id, task_id=tasks[0].task_id)
        declarations = {self.continuation.artifact_id(self.continuation.run_id, row.slot_id, task_id=task.task_id): row for task in tasks
            for row in self.continuation.inputs_for_task(task.task_id)}
        for task in tasks:
            if not isinstance(task, ExecutionTask):
                raise ValueError("development analysis task lacks its compiled scientific input edges")
            expected = {self.continuation.artifact_id(self.continuation.run_id, row.slot_id, task_id=task.task_id): row
                for row in self.continuation.inputs_for_task(task.task_id)}
            assessment = ".assess." in task.task_id
            task_config = self.continuation.qualification if assessment else self.continuation.qualification.method
            declaration_input = assessment or ".fit." in task.task_id
            expected_ids = {f"config-artifact.{task_config.config_id}", *expected,
                *((declaration_id,) if declaration_input else ())}
            if {spec.logical_artifact_id for spec in task.external_inputs} != expected_ids:
                raise ValueError("development analysis task changes its complete config/declaration/input roster")
            actual = {spec.logical_artifact_id: spec for spec in task.external_inputs
                if not spec.logical_artifact_id.startswith("config-artifact.")
                and spec.logical_artifact_id != declaration_id}
            if set(actual) != set(expected):
                raise ValueError("development analysis task changes its original projection split before input access")
            # Graph compilation retains the declared maximum on scientific
            # edges; the optional operational exact-size field remains unset.
            # The frozen digest and emitted payload still bind the exact bytes.
            scientific = {value.operational_logical_artifact_id: value
                for value in task.scientific_inputs if value.external_input_id is not None}
            for key, row in expected.items():
                spec = actual[key]
                if (spec.expected_content_sha256 != row.artifact.sha256
                        or spec.expected_payload_schema != row.artifact.payload_schema
                        or spec.expected_media_type != row.artifact.media_type
                        or spec.expected_size_bytes not in (None, row.artifact.size_bytes)
                        or key not in scientific
                        or scientific[key].maximum_size_bytes != row.artifact.size_bytes
                        or spec.expected_outcome_access is not OutcomeAccess.DEVELOPMENT_VISIBLE
                        or spec.expected_visibility_ceiling is not VisibilityCeiling.DEVELOPMENT_ONLY):
                    raise ValueError("development analysis task changes retained content or visibility")
            required_scan = len(task_config.canonical_bytes()) + sum(row.artifact.size_bytes for row in expected.values())
            if declaration_input:
                declaration = next(spec for spec in task.external_inputs if spec.logical_artifact_id == declaration_id)
                if (declaration.expected_content_sha256 != self.continuation.fingerprint()
                        or declaration.expected_payload_schema != self.continuation.SCHEMA
                        or declaration.expected_size_bytes not in (None, len(self.continuation.canonical_bytes()))
                        or declaration_id not in scientific
                        or scientific[declaration_id].maximum_size_bytes != len(self.continuation.canonical_bytes())
                        or declaration.expected_outcome_access is not OutcomeAccess.OUTCOME_BLIND
                        or declaration.expected_visibility_ceiling is not VisibilityCeiling.PROSPECTIVE):
                    raise ValueError("development analysis task changes its exact retained-input declaration")
                required_scan += len(self.continuation.canonical_bytes())
            by_id = {value.task_id: value for value in plan.tasks}
            required_scan += sum(by_id[dep].capability.requested_resources.output_bytes for dep in task.dependency_task_ids)
            if required_scan > task.capability.requested_resources.source_scan_bytes:
                raise ValueError("development analysis task omits imported bytes from its scan budget")
        if isinstance(self.config, (ResponseGeometryDevelopmentMethodConfig, ResponseGeometryDevelopmentQualificationConfig)):
            parent = ArtifactLineageParent(ObjectIdentity.from_record(self.continuation.config_id, self.continuation),
                VisibilityCeiling.PROSPECTIVE, OutcomeAccess.OUTCOME_BLIND)
            values.append(ExternalInputPayload.from_bytes(logical_artifact_id=declaration_id,
                payload_schema=self.continuation.SCHEMA, profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json", payload=self.continuation.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE, outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(parent.visibility_ceiling,), lineage_parents=(parent,)))
        if declarations:
            assert self.sources is not None
            for key, row in sorted(declarations.items()):
                parents = tuple(sorted((ArtifactLineageParent(row.task_receipt, VisibilityCeiling.DEVELOPMENT_ONLY,
                    OutcomeAccess.DEVELOPMENT_VISIBLE), ArtifactLineageParent(
                    ObjectIdentity.from_record(row.artifact.artifact_id, row.artifact),
                    VisibilityCeiling.DEVELOPMENT_ONLY, OutcomeAccess.DEVELOPMENT_VISIBLE)),
                    key=lineage_parent_sort_key))
                values.append(ExternalInputPayload(key, row.artifact.payload_schema,
                    ArtifactProfile.AUDITED_HDF5 if row.slot_id.endswith(".data") else ArtifactProfile.CANONICAL_JSON,
                    row.artifact.media_type, self.sources.create_source(row), row.artifact.size_bytes,
                    row.artifact.sha256, row.artifact.size_bytes, min(row.artifact.size_bytes, MAX_EXTERNAL_INPUT_CHUNK_BYTES),
                    VisibilityCeiling.DEVELOPMENT_ONLY, OutcomeAccess.DEVELOPMENT_VISIBLE,
                    tuple(parent.visibility_ceiling for parent in parents), parents, row.artifact.sha256))
        return tuple(sorted(values, key=lambda value: value.logical_artifact_id))

    def output_semantic_contracts(self, registry: CapabilityRegistry,
                                  execution_plan: ProtocolExecutionPlan | None = None) -> tuple[CapabilityOutputSemanticContract, ...]:
        return self._delegate.output_semantic_contracts(registry, execution_plan)

    def scientific_adjudication_contract(self, registry: CapabilityRegistry,
                                         execution_plan: ProtocolExecutionPlan | None = None) -> ScientificAdjudicationOutputContract | None:
        return self._delegate.scientific_adjudication_contract(registry, execution_plan)


class ResponseGeometryDevelopmentRetainedAssessmentTask:
    """Authenticate the shared task input, then invoke the original development owner."""

    def __init__(self, manifest: CapabilityManifest, config: ResponseGeometryDevelopmentQualificationConfig,
                 payload_plane: CandidatePayloadPlane) -> None:
        self.manifest, self.config, self.payload_plane = manifest, config, payload_plane

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            declarations = tuple(port for port in context.input_ports if port.payload_schema == ResponseGeometryDevelopmentAnalysisContinuationConfig.SCHEMA)
            if (context.run_id != RETAINED_ANALYSIS_RUN or len(declarations) != 1
                    or declarations[0].kind is not WorkerInputKind.EXTERNAL):
                raise ValueError("development retained assessment lacks its exact shared declaration input")
            port = declarations[0]
            continuation = decode_port(port, ResponseGeometryDevelopmentAnalysisContinuationConfig)
            if continuation.artifact_id(context.run_id, continuation.config_id, task_id=context.task_id) != port.artifact_id or continuation.qualification != self.config:
                raise ValueError("development retained assessment substitutes its frozen science")
            ports = tuple(value for value in context.input_ports if value is not port)
            delegated = replace(context, input_ports=ports, input_bindings=tuple(value.binding for value in ports))
            return ResponseGeometryDevelopmentAssessmentTask(self.manifest, self.config, self.payload_plane, continuation).execute(delegated)
        finally:
            for value in context.input_ports:
                value.close()


class ResponseGeometryDevelopmentAnalysisAssessmentProvider(ResponseGeometryDevelopmentAssessmentProvider):
    def __init__(self, registry: CapabilityRegistry, manifest: CapabilityManifest, config: ResponseGeometryDevelopmentQualificationConfig,
                 payload_plane: CandidatePayloadPlane, sources: ResponseGeometryDevelopmentRetainedProjectionSourceFactory) -> None:
        super().__init__(registry, manifest, config, payload_plane)
        self.sources = sources

    def external_inputs(self, plan: ProtocolExecutionPlan,
                        source_records: tuple[CanonicalRecord, ...] = ()) -> tuple[ExternalInputPayload, ...]:
        # Source-port metadata is accepted only when the issued task graph
        # commits its exact bytes. Execution reads that same declared artifact.
        continuation = self.sources.continuation
        tasks = tuple(task for task in plan.tasks if task.capability.capability_key == self.manifest.capability_key)
        if not tasks or continuation.qualification != self.config:
            raise ValueError("development assessment source metadata changes its frozen qualification")
        for task in tasks:
            declarations = tuple(spec for spec in task.external_inputs if spec.expected_payload_schema == continuation.SCHEMA)
            if len(declarations) != 1 or declarations[0].expected_content_sha256 != continuation.fingerprint():
                raise ValueError("development assessment source metadata is not authenticated by its issued task input")
        return ResponseGeometryDevelopmentAnalysisContinuationProvider(self.registry, self.manifest, self.config,
            continuation, self.sources, self.payload_plane).external_inputs(plan, source_records)

    def runners(self, registry: CapabilityRegistry,
                source_records: tuple[CanonicalRecord, ...] = ()) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records or not isinstance(self.config, ResponseGeometryDevelopmentQualificationConfig):
            raise ValueError("development retained assessment changes its provider records")
        assert self.payload_plane is not None
        return (cast(TaskRunner, ResponseGeometryDevelopmentRetainedAssessmentTask(self.manifest, self.config, self.payload_plane)),)

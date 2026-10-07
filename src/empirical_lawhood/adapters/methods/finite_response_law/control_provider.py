"Registered finite response-law causal tasks over existing worker, law and control owners."

from dataclasses import replace
from hashlib import sha256
from typing import cast

from empirical_lawhood.adapters.methods.prepared_response.qualification_provider import _group
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.simulators.finite_response_law.evaluation_contracts import FiniteResponseLawEvaluationInvocation
from empirical_lawhood.adapters.simulators.finite_response_law.assigned_contracts import FiniteResponseLawAssignedEvaluationConfig, FiniteResponseLawAssignedEvaluationInvocation
from empirical_lawhood.adapters.simulators.finite_response_law.native_artifact import MAXIMUM_PAIR_BYTES, NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.finite_response_law.provider import native_stage
from empirical_lawhood.adapters.simulators.finite_response_law.source_outputs import FiniteResponseLawAssignedEvaluationTaskResult, FiniteResponseLawEvaluationTaskResult
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, output_result, read_port, semantic_contracts
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import (
    ArtifactIdentity,
    ExecutableReference,
    SafePayloadFormat,
)
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus
from empirical_lawhood.planning.controller_study import ImplementationRole
from empirical_lawhood.planning.finite_action_support import support_map_for_plan
from empirical_lawhood.planning.study_issue import StudyAuthorityKind, require_study_authority
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile
from empirical_lawhood.runtime.capabilities import (
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskRunner,
    WorkerInputKind,
    WorkerInputPort,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.runtime.source_resolution import (
    ContentAddressedInputKind,
    ContentAddressedInputRequirement,
    ContentAddressedInputResolver,
)

from .consumer import SPEC, FiniteResponseLawNativeUnitReadoutMap, evaluation_requests, response_coordinates
from .control_evaluation import evaluation_design, measured_hold_word
from .control_forecast import forecast_root
from .control_inputs import authenticated_control_inputs
from .control_locking import freeze_root_choices
from .control_parent import join_root_parent
from .control_plan import FiniteResponseLawControlLawContext, control_law_context
from .control_ports import FiniteResponseLawControlRuntimePort
from .control_records import FiniteResponseLawControlConfig, FiniteResponseLawRootControlLock, FiniteResponseLawRootForecast, FiniteResponseLawRootParentJoin, FiniteResponseLawRootRequestReveal
from .control_services import implementation_configurations, implementation_payloads
from .method_provider import _ResolvedInputSource
from .method_records import FiniteResponseLawAssignedQualificationReport, FiniteResponseLawQualificationReport
from .native_records import FiniteResponseLawCalibrationNativeEvaluation

CONTROL_RECORDS: tuple[type[CanonicalRecord], ...] = (
    FiniteResponseLawRootForecast,
    FiniteResponseLawRootRequestReveal,
    FiniteResponseLawRootControlLock,
    FiniteResponseLawRootParentJoin,
    FiniteResponseLawQualificationReport,
)
OPERATIONS = ("forecast", "reveal-request", "lock-choices", "join-parent", "freeze-prior")
FREEZE_PRIOR = "finite-response-law.prospective-evaluation.freeze-prior"


def control_tasks(config: FiniteResponseLawControlConfig) -> dict[str, tuple[int, str]]:
    return {
        FREEZE_PRIOR: (-1, "freeze-prior"),
        **{
            f"{r.stage_unit}.{op}": (r.index, op)
            for r in config.source.roots
            for op in OPERATIONS[:4]
        },
    }


def control_dependencies(config: FiniteResponseLawControlConfig, index: int, operation: str) -> tuple[str, ...]:
    if operation == "freeze-prior" and index == -1:
        return ()
    root = config.source.roots[index]
    source = ObjectIdentity.from_record(config.source.spec_id, config.source)
    invocation_type = (
        FiniteResponseLawAssignedEvaluationInvocation
        if type(config.source) is FiniteResponseLawAssignedEvaluationConfig
        else FiniteResponseLawEvaluationInvocation
    )
    prefix = invocation_type(source, root, "prefix", None, None, "prefix")
    parent = invocation_type(source, root, "parent", root.assigned_parent, None, "parent")
    return tuple(
        sorted(
            {
                "forecast": (prefix.task_id,),
                "reveal-request": (f"{root.stage_unit}.forecast",),
                "lock-choices": (
                    f"{root.stage_unit}.forecast",
                    f"{root.stage_unit}.reveal-request",
                ),
                "join-parent": (parent.task_id, f"{root.stage_unit}.lock-choices"),
            }[operation]
        )
    )


def input_artifact(port: WorkerInputPort, raw: bytes) -> ArtifactIdentity:
    if len(raw) != port.size_bytes:
        raise ValueError("finite response-law input identity loses its exact bounded worker bytes")
    return ArtifactIdentity(
        port.artifact_id,
        "prospective-control-causal-input",
        port.payload_schema,
        sha256(raw).hexdigest(),
        port.media_type,
        len(raw),
    )


class FiniteResponseLawControlTask:
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: FiniteResponseLawControlConfig,
        runtime: FiniteResponseLawControlRuntimePort,
        reader: CandidatePayloadReader,
    ):
        self.manifest, self.config, self.runtime, self.reader = manifest, config, runtime, reader
        self.tasks = control_tasks(config)

    def _external(self, context: TaskContext, operation: str) -> dict[str, bytes]:
        external = tuple(p for p in context.input_ports if p.kind is WorkerInputKind.EXTERNAL)
        expected = {
            f"config-artifact.{self.config.config_id}": (
                self.config.SCHEMA,
                self.config.fingerprint(),
            )
        }
        if operation in ("forecast", "lock-choices", "freeze-prior"):
            expected.update(
                {a.artifact_id: (a.payload_schema, a.sha256) for a in self.config.inputs}
            )
        if (
            len(external) != len(expected)
            or {p.artifact_id for p in external} != set(expected)
            or context.config.config_id != self.config.config_id
            or context.config.config_schema != self.config.SCHEMA
            or context.config.config_schema_sha256 != self.manifest.config_schema_sha256
            or context.config.content_sha256 != self.config.fingerprint()
            or context.config.artifact_id != f"config-artifact.{self.config.config_id}"
        ):
            raise ValueError("finite response-law control task substitutes its exact configuration/input census")
        payloads = {}
        for port in external:
            raw = read_port(port, 8 * 1024**2)
            if (port.payload_schema, sha256(raw).hexdigest()) != expected[port.artifact_id]:
                raise ValueError("finite response-law control input bytes differ from their issued identity")
            if port.artifact_id != context.config.artifact_id:
                payloads[port.artifact_id] = raw
        return payloads

    def _group(
        self, context: TaskContext, task_id: str, schemas: tuple[str, ...]
    ) -> dict[str, WorkerInputPort]:
        receipts = tuple(r for r in context.dependency_receipts if r.task_id == task_id)
        if len(receipts) != 1:
            raise ValueError("finite response-law causal task lacks its actual dependency receipt")
        return _group(context, task_id, receipts[0].output_materialization_ids, schemas)

    def _contexts(
        self,
        context: TaskContext,
        report: FiniteResponseLawQualificationReport,
        native: FiniteResponseLawCalibrationNativeEvaluation,
    ) -> tuple[FiniteResponseLawControlLawContext, ...]:
        from empirical_lawhood.adapters.composition.finite_response_law.design import qualification_system

        authority = self.runtime.execution_authority()
        require_study_authority(
            authority,
            kind=StudyAuthorityKind.EXPERIMENT_EXECUTION,
            subject=self.runtime.issued_study,
            prerequisite_authority=self.runtime.prerequisite_authority,
            grantee_id=self.runtime.grantee_id,
            at_utc=self.runtime.now(),
        )
        config = self.config
        reference = ExecutableReference(
            f"{config.config_id}.method",
            self.manifest.capability_key,
            self.manifest.capability_version,
            self.manifest.capability_key,
            ArtifactIdentity(
                context.config.artifact_id,
                "method-config",
                config.SCHEMA,
                config.fingerprint(),
                "application/vnd.empirical-lawhood.canonical+json",
                len(config.canonical_bytes()),
            ),
            SafePayloadFormat.CANONICAL_JSON,
            FiniteResponseLawRootForecast.SCHEMA,
            FiniteResponseLawRootControlLock.SCHEMA,
            True,
        )
        contexts = tuple(
            control_law_context(
                system=qualification_system(native.config.projection.native_spec),
                report=report,
                boundary=b.boundary,
                method=reference,
                producer=ObjectIdentity.from_record(self.manifest.capability_key, self.manifest),
                resource=ObjectIdentity.from_record(
                    f"{self.manifest.capability_key}.resources", self.manifest.resource_ceiling
                ),
                authority=ObjectIdentity.from_record(authority.authority_id, authority),
            )
            for b, q in zip(report.calibration.boundaries, report.qualifications, strict=True)
            if b.boundary != "lower" and q.scientific_status is ScientificStatus.SUPPORTED
        )
        primary = tuple(
            c for c in contexts if c.model_set.model_set_id == config.primary_model_set.model_set_id
        )
        if len(primary) != 1:
            raise ValueError("finite response-law controller differs from its predeclared qualified model set")
        # The declared outer set retains the exact law-qualified members. The inner admission
        # set adds only the already validated finite-word compatibility map.
        expected = replace(
            config.primary_model_set,
            extensions=tuple(
                sorted(
                    (
                        *config.primary_model_set.extensions,
                        support_map_for_plan(primary[0].plan).extension,
                    ),
                    key=lambda extension: extension.namespace,
                )
            ),
        )
        if primary[0].model_set != expected:
            raise ValueError("finite response-law controller differs from its predeclared qualified model set")
        return contexts

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            if context.task_id not in self.tasks:
                raise ValueError("finite response-law causal task is outside its exact census")
            index, operation = self.tasks[context.task_id]
            if tuple(
                sorted(r.task_id for r in context.dependency_receipts)
            ) != control_dependencies(self.config, index, operation):
                raise ValueError("finite response-law changes prediction/request/decision/parent ordering")
            data = self._external(context, operation)
            if operation == "freeze-prior":
                report, _ = authenticated_control_inputs(self.config, data)
                return output_result(
                    context,
                    {report.SCHEMA: report.canonical_bytes()},
                    ("authenticated-prior-qualification-no-new-fit",),
                )
            root = self.config.source.roots[index]
            result_type = (
                FiniteResponseLawAssignedEvaluationTaskResult
                if type(self.config.source) is FiniteResponseLawAssignedEvaluationConfig
                else FiniteResponseLawEvaluationTaskResult
            )
            result: CanonicalRecord
            if operation in ("forecast", "lock-choices", "freeze-prior"):
                report, native = authenticated_control_inputs(self.config, data)
                contexts = self._contexts(context, report, native)
            if operation == "forecast":
                group = self._group(
                    context,
                    control_dependencies(self.config, index, operation)[0],
                    (
                        result_type.SCHEMA,
                        LinkedCampaignStageEnvelope.SCHEMA,
                        NATIVE_PAIR_SCHEMA,
                    ),
                )
                port = group[result_type.SCHEMA]
                prefix = decode_port(port, result_type)
                if decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
                ) != native_stage(prefix):
                    raise ValueError("finite response-law prefix stage substitutes its native result")
                result = forecast_root(
                    config=self.config,
                    report=report,
                    contexts=contexts,
                    prefix=prefix,
                    prefix_bytes=read_port(group[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES),
                    prefix_artifacts=(input_artifact(port, prefix.canonical_bytes()),),
                    payload_reader=self.reader,
                )
            elif operation in ("reveal-request", "lock-choices"):
                group = self._group(
                    context, f"{root.stage_unit}.forecast", (FiniteResponseLawRootForecast.SCHEMA,)
                )
                forecast_port = group[FiniteResponseLawRootForecast.SCHEMA]
                forecast = decode_port(forecast_port, FiniteResponseLawRootForecast, maximum=8 * 1024**2)
                forecast_artifact = input_artifact(forecast_port, forecast.canonical_bytes())
                if forecast.root != root or forecast.config != ObjectIdentity.from_record(
                    self.config.config_id, self.config
                ):
                    raise ValueError("finite response-law request/choice substitutes its committed root forecast")
                if operation == "reveal-request":
                    receipt = next(
                        r for r in context.dependency_receipts if r.task_id.endswith(".forecast")
                    )
                    result = FiniteResponseLawRootRequestReveal(
                        root,
                        ObjectIdentity.from_record(forecast.forecast_id, forecast),
                        forecast_artifact,
                        receipt.receipt_id,
                        evaluation_requests(root),
                    )
                else:
                    group = self._group(
                        context,
                        f"{root.stage_unit}.reveal-request",
                        (FiniteResponseLawRootRequestReveal.SCHEMA,),
                    )
                    request_port = group[FiniteResponseLawRootRequestReveal.SCHEMA]
                    requests = decode_port(request_port, FiniteResponseLawRootRequestReveal)
                    receipt = next(
                        r for r in context.dependency_receipts if r.task_id.endswith(".forecast")
                    )
                    if (
                        requests.forecast_artifact != forecast_artifact
                        or requests.forecast_receipt_id != receipt.receipt_id
                    ):
                        raise ValueError(
                            "finite response-law request precedes or replaces the persisted full forecast"
                        )
                    store = self.runtime.prepared_store
                    scope = f"{root.stage_unit}.control-inputs"
                    mapping = FiniteResponseLawNativeUnitReadoutMap(
                        response_coordinates(ObjectIdentity.from_record("flh-science", SPEC))
                    )
                    authority = self.runtime.execution_authority()
                    artifacts = [
                        # _external authenticated these exact configuration bytes.
                        # controller admission requires the evaluator's complete artifact identity,
                        # including its declared role and media type.
                        contexts[0].plan.gate_predicates[0].evaluator.payload,
                        self.config.qualification,
                        self.config.calibration_native,
                        forecast_artifact,
                        input_artifact(request_port, requests.canonical_bytes()),
                        *{a for t in forecast.tables for a in t.prefix_artifacts},
                    ]
                    for object_id, record in (
                        (mapping.identity.object_id, mapping),
                        (authority.authority_id, authority),
                        *((c.config_id, c) for c in implementation_configurations()),
                    ):
                        artifacts.append(
                            store.publish_prepared_record(
                                prefix_id=scope, object_id=object_id, record=record
                            )
                        )
                    roles = {b.role: b for b, _ in implementation_payloads()}
                    word = contexts[0].word_maps[0].controller_word
                    plan, _ = evaluation_design(
                        source=self.config.source,
                        evaluator=roles[ImplementationRole.OUTCOME_EVALUATOR],
                        readout_map=mapping,
                        view_ids=contexts[0].plan.coordinates[0].qualification_view_ids,
                        hold_word=measured_hold_word(word.denominator_id, word.retained_history_id),
                    )
                    result = freeze_root_choices(
                        forecast=forecast,
                        requests=requests,
                        report=report,
                        native=native,
                        contexts=contexts,
                        readout_map=mapping,
                        evaluation_plan=plan,
                        artifacts=tuple(sorted(set(artifacts), key=lambda a: a.artifact_id)),
                        qualification_artifact=self.config.qualification,
                        authority=authority,
                        issued_study=self.runtime.issued_study,
                        prerequisite_authority=self.runtime.prerequisite_authority,
                        grantee_id=self.runtime.grantee_id,
                        compiler_release_id=self.runtime.compiler_release_id,
                        store=store,
                        now=self.runtime.now,
                    )
            else:
                group = self._group(
                    context, f"{root.stage_unit}.lock-choices", (FiniteResponseLawRootControlLock.SCHEMA,)
                )
                lock = decode_port(
                    group[FiniteResponseLawRootControlLock.SCHEMA], FiniteResponseLawRootControlLock, maximum=16 * 1024**2
                )
                receipt_binding = next(
                    r
                    for r in context.dependency_receipts
                    if r.task_id != f"{root.stage_unit}.lock-choices"
                )
                group = self._group(
                    context,
                    receipt_binding.task_id,
                    (
                        result_type.SCHEMA,
                        LinkedCampaignStageEnvelope.SCHEMA,
                        NATIVE_PAIR_SCHEMA,
                    ),
                )
                port = group[result_type.SCHEMA]
                parent = decode_port(port, result_type)
                if decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
                ) != native_stage(parent):
                    raise ValueError("finite response-law actual parent stage changes its native result")
                result = join_root_parent(
                    lock=lock,
                    parent=parent,
                    parent_bytes=read_port(group[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES),
                    receipt=self.runtime.dependency_receipt(context.run_id, receipt_binding),
                    artifact=input_artifact(port, parent.canonical_bytes()),
                    store=self.runtime.prepared_store,
                    now=self.runtime.now,
                )
            return output_result(
                context, {result.SCHEMA: result.canonical_bytes()}, ("exact-prospective-control-causal-boundary",)
            )
        finally:
            for port in context.input_ports:
                port.close()


class FiniteResponseLawControlProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: FiniteResponseLawControlConfig,
        runtime: FiniteResponseLawControlRuntimePort,
        reader: CandidatePayloadReader,
        resolver: ContentAddressedInputResolver,
    ):
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("finite response-law control provider changes its installed registry/configuration")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.runtime, self.reader, self.resolver = runtime, reader, resolver
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("finite response-law control runner changes registry or inputs")
        return (
            cast(
                TaskRunner, FiniteResponseLawControlTask(self.manifest, self.config, self.runtime, self.reader)
            ),
        )

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("finite response-law control execution changes its registry/records")
        tasks = control_tasks(self.config)
        values = list(
            config_payloads(plan, self.manifest, self.config, self.config.config_id, set(tasks))
        )
        for task in plan.tasks:
            if task.capability.capability_key != self.manifest.capability_key:
                continue
            index, operation = tasks[task.task_id]
            expected = {
                f"config-artifact.{self.config.config_id}": (
                    self.config.SCHEMA,
                    self.config.fingerprint(),
                )
            }
            if operation in ("forecast", "lock-choices", "freeze-prior"):
                expected.update(
                    {a.artifact_id: (a.payload_schema, a.sha256) for a in self.config.inputs}
                )
            if (
                task.dependency_task_ids != control_dependencies(self.config, index, operation)
                or task.maximum_attempts != 1
                or len(task.external_inputs) != len(expected)
                or {
                    v.logical_artifact_id: (v.expected_payload_schema, v.expected_content_sha256)
                    for v in task.external_inputs
                }
                != expected
            ):
                raise ValueError("finite response-law execution changes its causal graph or exact input census")
        for artifact in self.config.inputs:
            # These are the already revealed law qualification and prior witnesses,
            # frozen before any prospective evaluation root. Original receipts and exposure remain in
            # the authenticated bytes; this declaration concerns the prospective evaluation cutoff.
            requirement = ContentAddressedInputRequirement(
                f"input.{artifact.artifact_id}",
                ContentAddressedInputKind.SOURCE_MATERIALIZATION,
                artifact.sha256,
                artifact.payload_schema,
                artifact.media_type,
                8 * 1024**2,
                artifact.size_bytes,
                OutcomeAccess.OUTCOME_BLIND,
                VisibilityCeiling.PROSPECTIVE,
            )
            lineage = ArtifactLineageParent(
                ObjectIdentity.from_record(self.config.config_id, self.config),
                VisibilityCeiling.PROSPECTIVE,
                OutcomeAccess.OUTCOME_BLIND,
            )
            values.append(
                ExternalInputPayload(
                    artifact.artifact_id,
                    artifact.payload_schema,
                    ArtifactProfile.CANONICAL_JSON,
                    artifact.media_type,
                    _ResolvedInputSource(self.resolver, requirement),
                    artifact.size_bytes,
                    artifact.sha256,
                    artifact.size_bytes,
                    min(artifact.size_bytes, MAX_EXTERNAL_INPUT_CHUNK_BYTES),
                    VisibilityCeiling.PROSPECTIVE,
                    OutcomeAccess.OUTCOME_BLIND,
                    (lineage.visibility_ceiling,),
                    (lineage,),
                    artifact.sha256,
                )
            )
        return tuple(sorted(values, key=lambda v: v.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("finite response-law control semantics change registry")
        records = (
            (*CONTROL_RECORDS[:4], FiniteResponseLawAssignedQualificationReport)
            if type(self.config.source) is FiniteResponseLawAssignedEvaluationConfig
            else CONTROL_RECORDS
        )
        return semantic_contracts(self.manifest, records, None)

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("finite response-law control adjudication changes registry")

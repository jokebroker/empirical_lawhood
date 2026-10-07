"""Registered bounded workers with exact per-role artifact and receipt inputs."""

from .method_continuation import PreparationRetainedMethodPort, PreparationMethodContinuation, retained_method_payloads, retained_method_artifact_id

from dataclasses import dataclass, replace
from hashlib import sha256
from functools import cached_property
from typing import TypeVar, cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import ArtifactProfile
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.adapters.methods.contracts import CandidateEvaluatorImplementation
from empirical_lawhood.adapters.methods.qualification_profiles import QualificationProofOwner
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import envelope, output_result, read_port, semantic_contracts
from empirical_lawhood.adapters.simulators.matrix_preparation.contracts import CONTEXTS, DEVELOPMENT, NATIVE_SCHEMA, PreparationNativeResult
from empirical_lawhood.adapters.simulators.matrix_preparation.provider import preparation_config_payloads, preparation_source_stage
from empirical_lawhood.adapters.simulators.matrix_preparation.source import NATIVE_MAXIMUM_BYTES
from .continuation import PreparationRetainedNativePort, preparation_retained_native_payloads
from .closeout import PreparationCloseoutConfig, PreparationDevelopmentResult, close_preparation_development, context_disposition
from .contracts import FIT_SCHEMA, PRIVILEGED_SCHEMA, PROJECTION_SCHEMA, PROSPECTIVE_TASK_SCHEMA, PreparationMethodConfig, PreparationPrivilegedReport, PreparationProjectionConfig, PreparationProjectionReport, PreparationProspectiveTaskReport, PreparationProjectionRecord
from .data import BENCHMARK_SCHEMA, DECISION_SCHEMA, DESCRIPTION_SCHEMA, FORECAST_SCHEMA, METHOD_MAXIMUM_BYTES, NUMERICAL_SEMANTICS_SCHEMA, TASK_ASSESSMENT_SCHEMA, ProjectedContext, collect_projected_context
from .description import assess_description
from .fitting import fit_preparation_context
from .forecasting import forecast_preparation_context
from .law import LAW_KEY
from .projection import PROJECTION_MAXIMUM_BYTES, project_preparation_root
from .records import PreparationForecastConfig, PreparationDecisionConfig, PreparationTaskAssessmentConfig, PreparationAssessmentConfig, PreparationBenchmarkReport, PreparationDecisionReport, PreparationDescriptionReport, PreparationFitReport, PreparationForecastReport, PreparationNumericalSemanticsConfig, PreparationNumericalSemanticsReport, PreparationTaskAssessmentReport
from .numerical_semantics import qualify_numerical_semantics
from .task_assessment import assess_development_tasks, lock_development_decisions
from .terminal import PreparationLawReport, qualify_preparation_description
from .topology import DATA_OUTPUTS, METHOD_CONFIG_TYPES, METHOD_ROLES, RECORD_OUTPUTS, STAGE_SCHEMA, PreparationTaskDeclaration, expected_method_inputs, preparation_task_declarations, task_output_schemas


CANDIDATE_PAYLOAD_PORT = "candidate-payload-publisher"
Record = TypeVar("Record", bound=CanonicalRecord)
_DECODERS = {kind.SCHEMA: kind for values in RECORD_OUTPUTS.values() for _, kind in values}
_DECODERS[STAGE_SCHEMA] = LinkedCampaignStageEnvelope


def preparation_method_semantic_contracts(
    manifest: CapabilityManifest, role: str
) -> tuple[CapabilityOutputSemanticContract, ...]:
    """Resolve the same closed output contracts for execution and read-only custody."""
    values = list(
        semantic_contracts(
            manifest,
            tuple(kind for _, kind in RECORD_OUTPUTS[role]) + (LinkedCampaignStageEnvelope,),
            None,
        )
    )
    values.extend(
        CapabilityOutputSemanticContract.from_manifest(
            manifest, payload_schema=schema, profile=ArtifactProfile.AUDITED_HDF5
        )
        for _, schema in DATA_OUTPUTS[role]
    )
    return tuple(sorted(values, key=lambda v: v.key))


def preparation_worker_artifact_id(context: TaskContext, logical: str) -> str:
    return (
        logical
        if logical.startswith("config-artifact.")
        else f"artifact.{context.run_id}.{logical}"
    )


def preparation_method_stage(
    task_id: str, role: str, record: CanonicalRecord
) -> LinkedCampaignStageEnvelope:
    record_id = str(getattr(record, "report_id", getattr(record, "result_id", "")))
    reasons: tuple[str, ...] = ()
    stage_role = LinkedCampaignStageRole.METHOD_IDENTIFICATION
    if isinstance(record, PreparationProjectionReport):
        stage_role = LinkedCampaignStageRole.EVIDENCE_PROJECTION
        reasons = (
            ()
            if not record.incomplete_native_occurrences
            else ("OBSERVABLE_NATIVE_OCCURRENCES_INCOMPLETE",)
        )
    elif isinstance(record, PreparationNumericalSemanticsReport):
        stage_role = LinkedCampaignStageRole.EVIDENCE_PROJECTION
        reasons = record.reasons
    elif isinstance(record, PreparationFitReport):
        reasons = () if record.available_candidates else ("ALL_OBSERVABLE_MODELS_UNAVAILABLE",)
    elif isinstance(record, PreparationDescriptionReport):
        reasons = (
            ()
            if record.qualified_candidates
            else ("NO_OBSERVABLE_DESCRIPTION_PASSES_DEVELOPMENT_SCREEN",)
        )
    elif isinstance(record, (PreparationForecastReport, PreparationTaskAssessmentReport)):
        reasons = record.gate.reasons
    elif isinstance(record, PreparationDecisionReport):
        pass
    elif isinstance(record, PreparationDevelopmentResult):
        stage_role = LinkedCampaignStageRole.LAW_QUALIFICATION
        reasons = record.reasons
    else:
        raise TypeError("preparation stage has no declared primary output owner")
    stage = envelope(task_id, record, record_id, stage_role, not reasons, reasons)
    if reasons and isinstance(
        record, (PreparationDescriptionReport, PreparationTaskAssessmentReport)
    ):
        stage = replace(stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE)
    if (
        isinstance(record, PreparationDevelopmentResult)
        and record.scientific_status is ScientificStatus.NOT_SUPPORTED
    ):
        stage = replace(stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE)
    return stage


@dataclass(frozen=True)
class PreparationWorkerInputs:
    records: dict[str, CanonicalRecord]
    data: dict[str, bytes]

    def one(self, kind: type[Record], *, context: str | None = None) -> Record:
        selected = tuple(
            r
            for r in self.records.values()
            if type(r) is kind and (context is None or getattr(r, "context", None) == context)
        )
        if len(selected) != 1:
            raise ValueError(f"preparation task requires exactly one {kind.SCHEMA} for its context")
        return selected[0]

    @cached_property
    def payload_index(self) -> dict[tuple[str, str], bytes]:
        result = {}
        for key, data in self.data.items():
            index = key.rsplit("|", 1)[-1], sha256(data).hexdigest()
            if index in result:
                raise ValueError("preparation inputs repeat a binary payload role/content identity")
            result[index] = data
        return result

    def payload(self, schema: str, report: CanonicalRecord) -> bytes:
        digest = str(getattr(report, "data_sha256", getattr(report, "observations_sha256", "")))
        data = self.payload_index.get((schema, digest))
        if data is None:
            raise ValueError("preparation method lacks its exact report-bound binary payload")
        return data

    def projected(self, context: str, role: str) -> ProjectedContext:
        kind, schema = {
            "observable": (PreparationProjectionReport, PROJECTION_SCHEMA),
            "privileged": (PreparationPrivilegedReport, PRIVILEGED_SCHEMA),
            "prospective-task": (PreparationProspectiveTaskReport, PROSPECTIVE_TASK_SCHEMA),
        }[role]
        values = tuple(
            (cast(PreparationProjectionRecord, r), self.payload(schema, r))
            for r in self.records.values()
            if type(r) is kind and cast(PreparationProjectionRecord, r).root.context == context
        )
        return collect_projected_context(context, values, role=role)


def read_preparation_method_inputs(
    context: TaskContext,
    task: PreparationTaskDeclaration,
    config: CanonicalRecord,
    config_id: str,
    retained_slots: tuple[str, ...] = (),
    continuation: PreparationMethodContinuation | None = None,
) -> PreparationWorkerInputs:
    expected = expected_method_inputs(task, config_id, config.SCHEMA)
    if continuation is not None:
        expected[f"config-artifact.{continuation.config_id}"] = ObjectIdentity.SCHEMA

    def artifact_id(logical: str) -> str:
        return (
            retained_method_artifact_id(context.run_id, task.role, logical)
            if logical in retained_slots
            else preparation_worker_artifact_id(context, logical)
        )

    try:
        # Check the full logical port census before any scientific input read.
        if len(context.input_ports) != len(expected) or {
            p.artifact_id: p.payload_schema for p in context.input_ports
        } != {artifact_id(k): s for k, s in expected.items()}:
            raise ValueError(
                "preparation task input root/role/locator differs before outcome access"
            )
        actual = {p.artifact_id: p for p in context.input_ports}
        records: dict[str, CanonicalRecord] = {}
        data: dict[str, bytes] = {}
        for logical, schema in expected.items():
            port = actual[artifact_id(logical)]
            if continuation is not None and schema == ObjectIdentity.SCHEMA:
                if (
                    decode_canonical_bytes(
                        read_port(port, 4096),
                        ObjectIdentity,
                        maximum_bytes=4096,
                    )
                    != ObjectIdentity.from_record(continuation.config_id, continuation)
                ):
                    raise ValueError(
                        "retained input declaration changed before scientific assessment"
                    )
            elif schema == config.SCHEMA:
                if (
                    decode_canonical_bytes(
                        read_port(port, 4 * 1024**2), type(config), maximum_bytes=4 * 1024**2
                    )
                    != config
                ):
                    raise ValueError("preparation task input configuration changed")
            elif schema in _DECODERS:
                records[logical] = decode_canonical_bytes(
                    read_port(port, METHOD_MAXIMUM_BYTES),
                    _DECODERS[schema],
                    maximum_bytes=METHOD_MAXIMUM_BYTES,
                )
            else:
                maximum = (
                    NATIVE_MAXIMUM_BYTES
                    if schema == NATIVE_SCHEMA
                    else PROJECTION_MAXIMUM_BYTES
                    if schema in (PROJECTION_SCHEMA, PRIVILEGED_SCHEMA, PROSPECTIVE_TASK_SCHEMA)
                    else METHOD_MAXIMUM_BYTES
                )
                data[logical + "|" + schema] = read_port(port, maximum)
        by_id = {t.task_id: t for t in preparation_task_declarations()}
        for dependency in task.dependencies:
            declaration = by_id[dependency]
            primary_label = "native-result" if declaration.role == "source" else "report"
            primary = records.get(f"{dependency}.{primary_label}")
            if primary is None:
                raise ValueError(
                    "method dependency has no primary report within its scientific role"
                )
            expected_stage = (
                preparation_source_stage(cast(PreparationNativeResult, primary))
                if declaration.role == "source"
                else preparation_method_stage(dependency, declaration.role, primary)
            )
            key = f"{dependency}.{'stage-envelope' if declaration.role == 'source' else 'stage'}"
            if records[key] != expected_stage:
                raise ValueError(
                    "preparation method input stage differs from its exact prior report"
                )
        return PreparationWorkerInputs(records, data)
    finally:
        for port in context.input_ports:
            port.close()


def preparation_input_artifact(
    context: TaskContext, logical: str, digest: str, retained_role: str | None = None
) -> ArtifactIdentity:
    key = (
        preparation_worker_artifact_id(context, logical)
        if retained_role is None
        else retained_method_artifact_id(context.run_id, retained_role, logical)
    )
    binding = next((b for b in context.input_bindings if b.artifact_id == key), None)
    if binding is None:
        raise ValueError("preparation scientific owner lacks an authorized artifact binding")
    return ArtifactIdentity(
        key,
        "preparation-authenticated-input",
        binding.payload_schema,
        digest,
        binding.media_type,
        binding.size_bytes,
    )


def preparation_adjudication(
    context: TaskContext, result: PreparationDevelopmentResult
) -> ScientificAdjudicationRecord:
    a = context.scientific_adjudication_context
    if a is None:
        raise ValueError(
            "preparation terminal lacks its separate authenticated adjudication authority"
        )
    unresolved = result.scientific_status is ScientificStatus.UNEVALUABLE
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
        if unresolved
        else AdjudicationEvaluability.EVALUABLE,
        scientific_status=result.scientific_status,
        admission_status=AdmissionStatus.UNEVALUABLE
        if unresolved
        else AdmissionStatus.NOT_EVALUATED,
        reason_codes=result.reasons,
        fixture_scope_id=a.fixture_scope_id,
        plumbing_only=a.plumbing_only,
    )


class PreparationMethodTask:
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: CanonicalRecord,
        role: str,
        payload_plane: CandidatePayloadPlane | None = None,
        retained_slots: tuple[str, ...] = (),
        continuation: PreparationMethodContinuation | None = None,
    ) -> None:
        self.continuation = continuation
        self.retained_slots = retained_slots
        self.manifest, self.config, self.role, self.payload_plane = (
            manifest,
            config,
            role,
            payload_plane,
        )

    @property
    def worker_chunk_limit(self) -> int:
        # Only the two pure views of one root share a bounded worker. Native
        # acquisition, method publication and evaluation retain one-shot workers.
        return 2 if self.role == "projection" else 1

    def execute(self, context: TaskContext) -> RunnerResult:
        task = next(
            (
                t
                for t in preparation_task_declarations()
                if t.task_id == context.task_id and t.role == self.role
            ),
            None,
        )
        if task is None:
            raise ValueError("preparation method task is outside its immutable role/roster")
        expected_outputs = {
            f"{task.task_id}.{name}": schema
            for name, schema in task_output_schemas(self.role).items()
        }
        if (
            len(context.output_ports) != len(expected_outputs)
            or {p.output_id: p.payload_schema for p in context.output_ports} != expected_outputs
            or any(
                p.logical_artifact_id != preparation_worker_artifact_id(context, p.output_id)
                for p in context.output_ports
            )
        ):
            raise ValueError("preparation worker changes its exact production output locators")
        config_id = str(getattr(self.config, "config_id"))
        inputs = read_preparation_method_inputs(
            context,
            task,
            self.config,
            config_id,
            tuple(
                s
                for s in self.retained_slots
                if s in expected_method_inputs(task, config_id, self.config.SCHEMA)
            ),
            self.continuation,
        )
        values: dict[str, bytes] = {}
        primary: CanonicalRecord
        native_context = context.task_id.rsplit(".", 1)[-1]
        if self.role == "projection":
            config = cast(PreparationProjectionConfig, self.config)
            native = inputs.one(PreparationNativeResult)
            refinement = int(context.task_id.rsplit("r", 1)[-1])
            if context.task_id != f"{native.root.root_id}.project.r{refinement}":
                raise ValueError("projection task changes its native root/view")
            reports, data = project_preparation_root(
                config, native, inputs.payload(NATIVE_SCHEMA, native), refinement
            )
            primary = reports.observable
            values.update(
                {
                    r.SCHEMA: r.canonical_bytes()
                    for r in (reports.observable, reports.privileged, reports.prospective_task)
                }
            )
            values.update(
                {
                    PROJECTION_SCHEMA: data.observable,
                    PRIVILEGED_SCHEMA: data.privileged,
                    PROSPECTIVE_TASK_SCHEMA: data.prospective_task,
                }
            )
        elif self.role == "numerical-semantics":
            primary, data_bytes = qualify_numerical_semantics(
                cast(PreparationNumericalSemanticsConfig, self.config),
                inputs.projected(native_context, "observable"),
                inputs.projected(native_context, "privileged"),
            )
            values[NUMERICAL_SEMANTICS_SCHEMA] = data_bytes
        elif self.role == "fit":
            primary, data_bytes = fit_preparation_context(
                cast(PreparationMethodConfig, self.config),
                inputs.projected(native_context, "observable"),
                inputs.one(PreparationNumericalSemanticsReport),
            )
            values[FIT_SCHEMA] = data_bytes
        elif self.role == "description":
            assessment = cast(PreparationAssessmentConfig, self.config)
            fit, numerical_semantics = inputs.one(PreparationFitReport), inputs.one(PreparationNumericalSemanticsReport)
            fit_bytes, observable = (
                inputs.payload(FIT_SCHEMA, fit),
                inputs.projected(native_context, "observable"),
            )
            description, description_data, benchmark, benchmark_data = assess_description(
                assessment,
                observable,
                inputs.projected(native_context, "privileged"),
                fit,
                fit_bytes,
                numerical_semantics,
            )
            if self.payload_plane is None:
                raise ValueError("description has no declared candidate-payload publication port")
            implementation = CandidateEvaluatorImplementation(
                f"{DEVELOPMENT}.finite-law-evaluator",
                self.manifest.capability_key,
                self.manifest.capability_version,
                LAW_KEY,
                self.manifest.implementation_sha256,
            )
            law = qualify_preparation_description(
                config=assessment,
                fit=fit,
                fit_payload=fit_bytes,
                fit_artifact=preparation_input_artifact(
                    context,
                    f"{DEVELOPMENT}.fit.{native_context}.data",
                    fit.data_sha256,
                    self.role
                    if f"{DEVELOPMENT}.fit.{native_context}.data" in self.retained_slots
                    else None,
                ),
                description=description,
                description_artifact_id=preparation_worker_artifact_id(
                    context, f"{context.task_id}.data"
                ),
                numerical_semantics=numerical_semantics,
                observed=observable,
                payload_plane=self.payload_plane,
                profile_owner=QualificationProofOwner(
                    f"{DEVELOPMENT}.finite-law-proof-owner",
                    self.manifest.capability_key,
                    self.manifest.capability_version,
                    self.manifest.implementation_sha256,
                ),
                evaluator_implementation=implementation,
                selector_capability=ObjectIdentity.from_record(
                    self.manifest.capability_key, self.manifest
                ),
                selector_implementation=ObjectIdentity.from_record(
                    implementation.implementation_id, implementation
                ),
            )
            primary = description
            values.update(
                {
                    benchmark.SCHEMA: benchmark.canonical_bytes(),
                    law.SCHEMA: law.canonical_bytes(),
                    DESCRIPTION_SCHEMA: description_data,
                    BENCHMARK_SCHEMA: benchmark_data,
                }
            )
        elif self.role == "forecast":
            fit = inputs.one(PreparationFitReport)
            primary, data_bytes = forecast_preparation_context(
                cast(PreparationForecastConfig, self.config).method,
                inputs.projected(native_context, "observable"),
                fit,
                inputs.payload(FIT_SCHEMA, fit),
                inputs.one(PreparationDescriptionReport),
            )
            values[FORECAST_SCHEMA] = data_bytes
        elif self.role == "decision":
            fit, forecast = (
                inputs.one(PreparationFitReport),
                inputs.one(PreparationForecastReport),
            )
            primary, data_bytes = lock_development_decisions(
                cast(PreparationDecisionConfig, self.config).method,
                fit,
                inputs.payload(FIT_SCHEMA, fit),
                inputs.one(PreparationDescriptionReport),
                forecast,
                inputs.payload(FORECAST_SCHEMA, forecast),
            )
            values[DECISION_SCHEMA] = data_bytes
        elif self.role == "task-assessment":
            fit, forecast, decision = (
                inputs.one(PreparationFitReport),
                inputs.one(PreparationForecastReport),
                inputs.one(PreparationDecisionReport),
            )
            primary, data_bytes = assess_development_tasks(
                cast(PreparationTaskAssessmentConfig, self.config).method,
                inputs.projected(native_context, "observable"),
                inputs.projected(native_context, "prospective-task"),
                fit,
                inputs.payload(FIT_SCHEMA, fit),
                inputs.one(PreparationDescriptionReport),
                forecast,
                inputs.payload(FORECAST_SCHEMA, forecast),
                decision,
                inputs.payload(DECISION_SCHEMA, decision),
            )
            values[TASK_ASSESSMENT_SCHEMA] = data_bytes
        else:
            closeout = cast(PreparationCloseoutConfig, self.config)
            contexts = []
            for c in CONTEXTS:
                records = (
                    inputs.one(PreparationNumericalSemanticsReport, context=c),
                    inputs.one(PreparationDescriptionReport, context=c),
                    inputs.one(PreparationBenchmarkReport, context=c),
                    inputs.one(PreparationLawReport, context=c),
                    inputs.one(PreparationForecastReport, context=c),
                    inputs.one(PreparationDecisionReport, context=c),
                    inputs.one(PreparationTaskAssessmentReport, context=c),
                )
                # Authenticate every binary report digest before terminal reduction.
                for record, schema in zip(
                    (records[0], records[1], records[2], records[4], records[5], records[6]),
                    (
                        NUMERICAL_SEMANTICS_SCHEMA,
                        DESCRIPTION_SCHEMA,
                        BENCHMARK_SCHEMA,
                        FORECAST_SCHEMA,
                        DECISION_SCHEMA,
                        TASK_ASSESSMENT_SCHEMA,
                    ),
                    strict=True,
                ):
                    inputs.payload(schema, record)
                contexts.append(context_disposition(closeout.assessment, *records))
            result = close_preparation_development(closeout, tuple(contexts))
            primary = result
            values[ScientificAdjudicationRecord.SCHEMA] = preparation_adjudication(
                context, result
            ).canonical_bytes()
        values[primary.SCHEMA] = primary.canonical_bytes()
        values[STAGE_SCHEMA] = preparation_method_stage(
            context.task_id, self.role, primary
        ).canonical_bytes()
        return output_result(
            context,
            values,
            (
                "complete-input-role-before-read",
                "exact-report-payload-custody",
                "all-root-denominators-retained",
            ),
        )


class PreparationMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: CanonicalRecord,
        role: str,
        payload_plane: CandidatePayloadPlane | None = None,
        *,
        retained_sources: PreparationRetainedNativePort | None = None,
        retained_method_sources: PreparationRetainedMethodPort | None = None,
    ) -> None:
        if (
            role not in METHOD_ROLES
            or type(config) is not METHOD_CONFIG_TYPES[role]
            or registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
            or (role == "description") != (payload_plane is not None)
        ):
            raise ValueError("preparation method factory changed its installed role/config/port")
        self.registry, self.manifest, self.config, self.role, self.payload_plane = (
            registry,
            manifest,
            config,
            role,
            payload_plane,
        )
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        if retained_sources is not None and role != "projection":
            raise ValueError("only the projection owner may import retained native observations")
        self.retained_sources = retained_sources
        self.retained_method_sources = retained_method_sources

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("preparation method runner registry/records differ")
        return (
            cast(
                TaskRunner,
                PreparationMethodTask(
                    self.manifest,
                    self.config,
                    self.role,
                    self.payload_plane,
                    tuple(r.slot_id for r in self.retained_method_sources.continuation.inputs)
                    if self.retained_method_sources is not None
                    and self.retained_method_sources.continuation is not None
                    else (),
                    self.retained_method_sources.continuation
                    if self.role == "description" and self.retained_method_sources is not None
                    else None,
                ),
            ),
        )

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("preparation method execution registry/records differ")
        values = preparation_config_payloads(
            plan,
            self.manifest,
            self.config,
            str(getattr(self.config, "config_id")),
            {t.task_id for t in preparation_task_declarations() if t.role == self.role},
        )
        if self.retained_sources is not None:
            values += preparation_retained_native_payloads(plan, self.retained_sources)
        if self.retained_method_sources is not None:
            continuation = self.retained_method_sources.continuation
            if self.role == "description" and continuation is not None:
                values += preparation_config_payloads(
                    plan,
                    self.manifest,
                    ObjectIdentity.from_record(continuation.config_id, continuation),
                    continuation.config_id,
                    {t.task_id for t in preparation_task_declarations() if t.role == self.role},
                )
            values += retained_method_payloads(plan, self.retained_method_sources, self.role)
        return tuple(sorted(values, key=lambda value: value.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("preparation method semantic registry differs")
        return preparation_method_semantic_contracts(self.manifest, self.role)

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("preparation method adjudication registry differs")
        return (
            None
            if self.role != "evaluation"
            else ScientificAdjudicationOutputContract(
                self.manifest.capability_key,
                self.manifest.capability_version,
                f"{DEVELOPMENT}.evaluate.scientific-adjudication",
                ScientificAdjudicationRecord.SCHEMA,
                fixture_scope_id=None,
                plumbing_only=False,
            )
        )

"""Installed development qualification/analysis and final adjudication over worker ports."""

from dataclasses import replace
from hashlib import sha256
from typing import cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity, ExecutableReference, SafePayloadFormat
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.kernel.time import ClockCoordinate, ClockProjection
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
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
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, envelope, output_result, read_port, semantic_contracts
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import DEVELOPMENT_PANEL_ID
from .development_acquisition import assess_response_geometry_development_offline_acquisition
from .development_analysis import analyze_response_geometry_development_validation
from .development_closeout import ResponseGeometryDevelopmentCloseoutConfig, ResponseGeometryDevelopmentContextResult, ResponseGeometryDevelopmentDevelopmentResult, close_response_geometry_development_development
from .development_geometry import DEVELOPMENT_GEOMETRY_MAX_STAGE_BYTES
from .development_geometry_assessment import ResponseGeometryDevelopmentGeometryReport, assess_response_geometry_development_geometry
from .development_law import DEVELOPMENT_LAW_KEY
from .development_provider import response_geometry_development_worker_artifact_id, read_response_geometry_development_method_inputs
from .development_continuation import ResponseGeometryDevelopmentAnalysisContinuationConfig
from .development_records import ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentSupportResult
from .development_terminal import ResponseGeometryDevelopmentQualificationConfig, DEVELOPMENT_VALIDATION_SCHEMA, DEVELOPMENT_VALIDATION_MAXIMUM_BYTES, qualify_response_geometry_development_context


DEVELOPMENT_CONTEXT_MAXIMUM_BYTES = 32 * 1024**2
DEVELOPMENT_CANDIDATE_PAYLOAD_PORT = "candidate-payload-publisher"
DEVELOPMENT_EVALUATION_TASK = f"{DEVELOPMENT_PANEL_ID}.evaluate"


def response_geometry_development_context_stage(result: ResponseGeometryDevelopmentContextResult) -> LinkedCampaignStageEnvelope:
    status = result.qualification.scientific_status
    stage = envelope(
        result.result_id,
        result,
        result.result_id,
        LinkedCampaignStageRole.LAW_QUALIFICATION,
        status is ScientificStatus.SUPPORTED,
        result.qualification.reason_codes or ("DEVELOPMENT_LOCAL_LAW_NOT_QUALIFIED",),
    )
    return (
        replace(stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE)
        if status is ScientificStatus.NOT_SUPPORTED
        else stage
    )


def _input_artifact(context: TaskContext, artifact_id: str, digest: str,
                    continuation: ResponseGeometryDevelopmentAnalysisContinuationConfig | None = None) -> ArtifactIdentity:
    artifact_id = response_geometry_development_worker_artifact_id(context, artifact_id, continuation)
    binding = next((b for b in context.input_bindings if b.artifact_id == artifact_id), None)
    if binding is None:
        raise ValueError("development assessment lacks an actual authorized artifact binding")
    return ArtifactIdentity(
        artifact_id,
        "development-authenticated-input",
        binding.payload_schema,
        digest,
        binding.media_type,
        binding.size_bytes,
    )


def response_geometry_development_assessment_references(
    config: ResponseGeometryDevelopmentQualificationConfig, manifest: CapabilityManifest, config_artifact: ArtifactIdentity
) -> tuple[ExecutableReference, ExecutableReference]:
    if (
        config_artifact.payload_schema != config.SCHEMA
        or config_artifact.sha256 != config.fingerprint()
        or config_artifact.size_bytes != len(config.canonical_bytes())
    ):
        raise ValueError("development scientific references lack their actual configuration artifact")
    clock = ExecutableReference(
        f"{config.config_id}.native-clock-projector",
        manifest.capability_key,
        manifest.capability_version,
        "kernel.clock-transport.project",
        config_artifact,
        SafePayloadFormat.CANONICAL_JSON,
        ClockCoordinate.SCHEMA,
        ClockProjection.SCHEMA,
        True,
    )
    geometry = ExecutableReference(
        f"{config.config_id}.finite-model-geometry",
        manifest.capability_key,
        manifest.capability_version,
        f"{DEVELOPMENT_PANEL_ID}.finite-model-geometry",
        config_artifact,
        SafePayloadFormat.CANONICAL_JSON,
        ResponseGeometryDevelopmentFitResult.SCHEMA,
        ResponseGeometryDevelopmentGeometryReport.SCHEMA,
        True,
    )
    return clock, geometry


class ResponseGeometryDevelopmentAssessmentTask:
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: ResponseGeometryDevelopmentQualificationConfig,
        payload_plane: CandidatePayloadPlane,
        continuation: ResponseGeometryDevelopmentAnalysisContinuationConfig | None = None,
    ) -> None:
        self.manifest, self.config, self.payload_plane = manifest, config, payload_plane
        if continuation is not None and continuation.qualification != config:
            raise ValueError("development correction changes its frozen qualification configuration")
        self.continuation = continuation

    def execute(self, context: TaskContext) -> RunnerResult:
        native_context = context.task_id.rsplit(".", 1)[-1]
        if context.task_id != f"{DEVELOPMENT_PANEL_ID}.assess.{native_context}" or native_context not in (
            "assembling",
            "prepared",
        ):
            raise ValueError("development assessment task changes its exact context role")
        inputs = read_response_geometry_development_method_inputs(
            context,
            config=self.config.method,
            native_context=native_context,
            role="validation",
            preceding=(ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentSupportResult),
            include_fit_views=True,
            task_config=self.config,
            task_config_id=self.config.config_id,
            continuation=self.continuation,
        )
        fit, calibration, support = (
            inputs.record(ResponseGeometryDevelopmentFitResult),
            inputs.record(ResponseGeometryDevelopmentCalibrationResult),
            inputs.record(ResponseGeometryDevelopmentSupportResult),
        )
        assert inputs.fit_payload is not None
        config_artifact = _input_artifact(
            context, f"config-artifact.{self.config.config_id}", self.config.fingerprint()
        )
        fit_artifact = _input_artifact(context, f"{fit.result_id}.data", fit.data_sha256)
        artifacts = [
            config_artifact,
            fit_artifact,
            _input_artifact(context, f"{fit.result_id}.report", fit.fingerprint()),
        ]
        for view in inputs.views:
            task = f"{view.report.root.root_id}.project.r{view.report.refinement}"
            artifacts.extend(
                (
                    _input_artifact(context, f"{task}.report", view.report.fingerprint(), self.continuation),
                    _input_artifact(context, f"{task}.data", view.report.data_sha256, self.continuation),
                )
            )
        clock, geometry_evaluator = response_geometry_development_assessment_references(
            self.config, self.manifest, config_artifact
        )
        owner = QualificationProofOwner(
            f"{DEVELOPMENT_PANEL_ID}.finite-law-qualification-owner",
            self.manifest.capability_key,
            self.manifest.capability_version,
            self.manifest.implementation_sha256,
        )
        implementation = CandidateEvaluatorImplementation(
            f"{DEVELOPMENT_PANEL_ID}.complete-affine-evaluator",
            self.manifest.capability_key,
            self.manifest.capability_version,
            DEVELOPMENT_LAW_KEY,
            self.manifest.implementation_sha256,
        )
        data_ports = [p for p in context.output_ports if p.payload_schema == DEVELOPMENT_VALIDATION_SCHEMA]
        if len(data_ports) != 1 or data_ports[0].logical_artifact_id != response_geometry_development_worker_artifact_id(context, f"{context.task_id}.data"):
            raise ValueError("development assessment loses its actual validation-array output locator")
        validation, data, scope, claim_units, family, qualification = qualify_response_geometry_development_context(
            config=self.config,
            fit=fit,
            fit_payload=inputs.fit_payload,
            fit_artifact=fit_artifact,
            calibration=calibration,
            views=inputs.views,
            validation_artifact_id=data_ports[0].logical_artifact_id,
            payload_plane=self.payload_plane,
            profile_owner=owner,
            evaluator_implementation=implementation,
            selector_capability=ObjectIdentity.from_record(
                self.manifest.capability_key, self.manifest
            ),
            selector_implementation=ObjectIdentity.from_record(
                implementation.implementation_id, implementation
            ),
        )
        geometry = assess_response_geometry_development_geometry(
            config=self.config,
            fit=fit,
            fit_payload=inputs.fit_payload,
            views=inputs.views,
            clock_evaluator=clock,
            source_evaluator=geometry_evaluator,
            implementation_sha256=self.manifest.implementation_sha256,
            input_artifacts=tuple(sorted(artifacts, key=lambda a: a.artifact_id)),
        )
        acquisition = assess_response_geometry_development_offline_acquisition(
            config=self.config,
            fit=fit,
            fit_payload=inputs.fit_payload,
            calibration=calibration,
            support=support,
            geometry=geometry,
            fit_views=inputs.fit_views,
            validation_views=inputs.views,
        )
        analysis = analyze_response_geometry_development_validation(
            fit=fit,
            fit_payload=inputs.fit_payload,
            calibration=calibration,
            support=support,
            validation=validation,
            views=inputs.views,
        )
        result = ResponseGeometryDevelopmentContextResult(
            ObjectIdentity.from_record(self.config.config_id, self.config),
            validation,
            analysis,
            scope,
            claim_units,
            family,
            qualification,
            ObjectIdentity.from_record(geometry.report_id, geometry),
            acquisition,
        )
        payload = result.canonical_bytes()
        if len(payload) > DEVELOPMENT_CONTEXT_MAXIMUM_BYTES:
            raise ValueError("development assessment context report exceeds its declared size")
        return output_result(
            context,
            {
                result.SCHEMA: payload,
                DEVELOPMENT_VALIDATION_SCHEMA: data,
                geometry.SCHEMA: geometry.canonical_bytes(),
                LinkedCampaignStageEnvelope.SCHEMA: response_geometry_development_context_stage(result).canonical_bytes(),
            },
            (
                "development-exact-fit-calibration-before-validation",
                "development-existing-sole-law-finalizer",
                "development-complete-rep-acq-support-diagnostic-rosters",
            ),
        )


def _adjudication(
    context: TaskContext, result: ResponseGeometryDevelopmentDevelopmentResult
) -> ScientificAdjudicationRecord:
    authority = context.scientific_adjudication_context
    if authority is None:
        raise ValueError("development evaluator lacks authenticated scientific adjudication authority")
    unavailable = result.scientific_status is ScientificStatus.UNEVALUABLE
    return ScientificAdjudicationRecord(
        adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
        run_id=context.run_id,
        adjudication_task_id=context.task_id,
        execution_plan=authority.execution_plan,
        input_materialization_ids=context.input_materialization_ids,
        output_logical_artifact_ids=tuple(
            sorted(
                p.logical_artifact_id
                for p in context.output_ports
                if p.logical_artifact_id is not None
            )
        ),
        required_receipt_ids=context.dependency_receipt_ids,
        evidence_world_id=authority.evidence_world_id,
        evidence_world_kind=authority.evidence_world_kind,
        relation=authority.relation,
        independent_unit_id=authority.independent_unit_id,
        information_cutoffs=authority.information_cutoffs,
        visibility_ceiling=authority.visibility_ceiling,
        outcome_access=authority.outcome_access,
        evaluability=AdjudicationEvaluability.UNEVALUABLE
        if unavailable
        else AdjudicationEvaluability.EVALUABLE,
        scientific_status=result.scientific_status,
        admission_status=AdmissionStatus.UNEVALUABLE
        if unavailable
        else AdmissionStatus.NOT_EVALUATED,
        reason_codes=result.reasons,
        fixture_scope_id=authority.fixture_scope_id,
        plumbing_only=authority.plumbing_only,
    )


class ResponseGeometryDevelopmentEvaluationTask:
    def __init__(self, manifest: CapabilityManifest, config: ResponseGeometryDevelopmentCloseoutConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        expected = {f"config-artifact.{self.config.config_id}": self.config.SCHEMA}
        for native_context in ("assembling", "prepared"):
            task = f"{DEVELOPMENT_PANEL_ID}.assess.{native_context}"
            expected.update(
                {
                    f"{task}.report": ResponseGeometryDevelopmentContextResult.SCHEMA,
                    f"{task}.data": DEVELOPMENT_VALIDATION_SCHEMA,
                    f"{task}.geometry": ResponseGeometryDevelopmentGeometryReport.SCHEMA,
                    f"{task}.stage": LinkedCampaignStageEnvelope.SCHEMA,
                }
            )
        try:
            if (
                context.task_id != DEVELOPMENT_EVALUATION_TASK
                or len(context.input_ports) != len(expected)
                or {p.artifact_id: p.payload_schema for p in context.input_ports}
                != {response_geometry_development_worker_artifact_id(context, key): schema for key, schema in expected.items()}
            ):
                raise ValueError("development closeout changes its complete context-output locators")
            actual_ports = {p.artifact_id: p for p in context.input_ports}
            ports = {key: actual_ports[response_geometry_development_worker_artifact_id(context, key)] for key in expected}
            if (
                decode_port(ports[f"config-artifact.{self.config.config_id}"], ResponseGeometryDevelopmentCloseoutConfig)
                != self.config
            ):
                raise ValueError("development closeout config differs from issue")
            contexts = []
            for native_context in ("assembling", "prepared"):
                task = f"{DEVELOPMENT_PANEL_ID}.assess.{native_context}"
                result = decode_canonical_bytes(
                    read_port(ports[f"{task}.report"], DEVELOPMENT_CONTEXT_MAXIMUM_BYTES),
                    ResponseGeometryDevelopmentContextResult,
                    maximum_bytes=DEVELOPMENT_CONTEXT_MAXIMUM_BYTES,
                )
                geometry = read_port(ports[f"{task}.geometry"], DEVELOPMENT_GEOMETRY_MAX_STAGE_BYTES)
                data = read_port(ports[f"{task}.data"], DEVELOPMENT_VALIDATION_MAXIMUM_BYTES)
                if (
                    result.context != native_context
                    or result.config
                    != ObjectIdentity.from_record(
                        self.config.qualification.config_id, self.config.qualification
                    )
                    or sha256(geometry).hexdigest() != result.geometry_report.object_fingerprint
                    or sha256(data).hexdigest() != result.validation.data_sha256
                    or decode_port(ports[f"{task}.stage"], LinkedCampaignStageEnvelope)
                    != response_geometry_development_context_stage(result)
                ):
                    raise ValueError("development closeout changes authentic context custody")
                contexts.append(result)
            final = close_response_geometry_development_development(self.config.qualification, tuple(contexts))
            stage = envelope(
                context.task_id,
                final,
                final.result_id,
                LinkedCampaignStageRole.LAW_QUALIFICATION,
                final.scientific_status in (ScientificStatus.SUPPORTED, ScientificStatus.MIXED),
                final.reasons,
            )
            adjudication = _adjudication(context, final)
            return output_result(
                context,
                {
                    final.SCHEMA: final.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                    adjudication.SCHEMA: adjudication.canonical_bytes(),
                },
                ("development-complete-two-context-custody", "development-separate-rep-acq-act-eligibility"),
            )
        finally:
            for port in context.input_ports:
                port.close()


class ResponseGeometryDevelopmentAssessmentProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: ResponseGeometryDevelopmentQualificationConfig | ResponseGeometryDevelopmentCloseoutConfig,
        payload_plane: CandidatePayloadPlane | None = None,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
            or (isinstance(config, ResponseGeometryDevelopmentQualificationConfig) != (payload_plane is not None))
        ):
            raise ValueError("development assessment provider changes its installed config/custody port")
        self.registry, self.manifest, self.config, self.payload_plane = (
            registry,
            manifest,
            config,
            payload_plane,
        )
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("development assessment runner registry/records differ")
        if isinstance(self.config, ResponseGeometryDevelopmentQualificationConfig):
            assert self.payload_plane is not None
            return (
                cast(TaskRunner, ResponseGeometryDevelopmentAssessmentTask(self.manifest, self.config, self.payload_plane)),
            )
        return (cast(TaskRunner, ResponseGeometryDevelopmentEvaluationTask(self.manifest, self.config)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("development assessment execution registry/records differ")
        expected = (
            {f"{DEVELOPMENT_PANEL_ID}.assess.{c}" for c in ("assembling", "prepared")}
            if isinstance(self.config, ResponseGeometryDevelopmentQualificationConfig)
            else {DEVELOPMENT_EVALUATION_TASK}
        )
        return config_payloads(plan, self.manifest, self.config, self.config.config_id, expected)

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("development assessment semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (ResponseGeometryDevelopmentContextResult, ResponseGeometryDevelopmentGeometryReport, LinkedCampaignStageEnvelope)
            if isinstance(self.config, ResponseGeometryDevelopmentQualificationConfig)
            else (
                ResponseGeometryDevelopmentDevelopmentResult,
                ScientificAdjudicationRecord,
                LinkedCampaignStageEnvelope,
            ),
            DEVELOPMENT_VALIDATION_SCHEMA if isinstance(self.config, ResponseGeometryDevelopmentQualificationConfig) else None,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("development assessment adjudication registry differs")
        if isinstance(self.config, ResponseGeometryDevelopmentQualificationConfig):
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=f"{DEVELOPMENT_EVALUATION_TASK}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            fixture_scope_id=None,
            plumbing_only=False,
        )

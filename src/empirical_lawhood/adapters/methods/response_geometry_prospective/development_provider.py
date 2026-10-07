"""development projection and development methods over bounded ordinary worker ports."""

from dataclasses import dataclass
from hashlib import sha256
from typing import Literal, TypeVar, cast

from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner, WorkerInputKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, envelope, output_result, read_port, semantic_contracts, source_stage_envelope
from empirical_lawhood.adapters.simulators.six_matrix_response.response_development import DEVELOPMENT_PANEL_ID, DEVELOPMENT_HDF5_SCHEMA, ResponseGeometryDevelopmentNativeSegmentResult, development_roots
from .development_assessment import calibrate_response_geometry_development_context, fit_response_geometry_development_support
from .development_continuation import ResponseGeometryDevelopmentAnalysisContinuationConfig
from .development_models import ResponseGeometryDevelopmentMeasuredView, fit_response_geometry_development_context, organize_response_geometry_development_views, read_response_geometry_development_measurements
from .development_projection import DEVELOPMENT_DATA_SCHEMA, ResponseGeometryDevelopmentProjectionConfig, ResponseGeometryDevelopmentViewReport, project_response_geometry_development_view
from .development_records import DEVELOPMENT_FIT_MAXIMUM_BYTES, DEVELOPMENT_FIT_SCHEMA, ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentMethodConfig, ResponseGeometryDevelopmentSupportResult, ResponseGeometryDevelopmentSupportRow, read_response_geometry_development_fit, report_identities, write_response_geometry_development_fit


DInputRole = Literal["fit", "calibration", "validation"]
Record = TypeVar("Record", bound=CanonicalRecord)


def response_geometry_development_projection_stage(report: ResponseGeometryDevelopmentViewReport) -> LinkedCampaignStageEnvelope:
    return envelope(
        f"{report.root.root_id}.project.r{report.refinement}",
        report,
        report.report_id,
        LinkedCampaignStageRole.EVIDENCE_PROJECTION,
        all(p.delivered for p in report.packets),
        tuple(sorted({r for p in report.packets for r in p.qualification_reasons})),
    )


def response_geometry_development_method_stage(
    result: ResponseGeometryDevelopmentFitResult | ResponseGeometryDevelopmentCalibrationResult | ResponseGeometryDevelopmentSupportResult,
) -> LinkedCampaignStageEnvelope:
    if isinstance(result, ResponseGeometryDevelopmentFitResult):
        available = bool(result.available_representations)
    elif isinstance(result, ResponseGeometryDevelopmentCalibrationResult):
        available = result.calibration.quantile is not None
    else:
        available = any(row.predictor_json is not None for row in result.models)
    return envelope(
        result.result_id,
        result,
        result.result_id,
        LinkedCampaignStageRole.METHOD_IDENTIFICATION,
        available,
        () if available else ("DEVELOPMENT_METHOD_OPERANDS_UNAVAILABLE",),
    )


class ResponseGeometryDevelopmentProjectionTask:
    def __init__(self, manifest: CapabilityManifest, config: ResponseGeometryDevelopmentProjectionConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        slot = next(
            (
                (root, r)
                for root in development_roots()
                for r in (1, 2)
                if context.task_id == f"{root.root_id}.project.r{r}"
            ),
            None,
        )
        if slot is None:
            raise ValueError("development projection task is outside its exact root/view roster")
        configs, results, stages, payloads = [], [], [], {}
        try:
            for port in context.input_ports:
                if port.payload_schema == self.config.SCHEMA:
                    configs.append(decode_port(port, ResponseGeometryDevelopmentProjectionConfig))
                elif port.payload_schema == ResponseGeometryDevelopmentNativeSegmentResult.SCHEMA:
                    results.append(decode_port(port, ResponseGeometryDevelopmentNativeSegmentResult))
                elif port.payload_schema == LinkedCampaignStageEnvelope.SCHEMA:
                    stages.append(decode_port(port, LinkedCampaignStageEnvelope))
                elif port.payload_schema == DEVELOPMENT_HDF5_SCHEMA:
                    data = read_port(port, 16 * 1024**2)
                    digest = sha256(data).hexdigest()
                    if digest in payloads:
                        raise ValueError("development projection repeats a native shard")
                    payloads[digest] = data
                else:
                    raise ValueError("development projection input schema differs")
            if (
                configs != [self.config]
                or len(results) != 21
                or len(stages) != 21
                or len(payloads) != 21
                or set(stages) != {source_stage_envelope(r) for r in results}
                or set(payloads) != {r.observations_sha256 for r in results}
            ):
                raise ValueError("development projection changes its exact source custody roster")
            report, data = project_response_geometry_development_view(
                self.config,
                *slot,
                tuple((r, payloads[r.observations_sha256]) for r in results),
            )
            stage = response_geometry_development_projection_stage(report)
            return output_result(
                context,
                {
                    report.SCHEMA: report.canonical_bytes(),
                    DEVELOPMENT_DATA_SCHEMA: data,
                    stage.SCHEMA: stage.canonical_bytes(),
                },
                ("exact-development-root-view-custody", "causal-observed-features-only"),
            )
        finally:
            for port in context.input_ports:
                port.close()


@dataclass(frozen=True)
class ResponseGeometryDevelopmentMethodInputs:
    views: tuple[ResponseGeometryDevelopmentMeasuredView, ...]
    calibration_views: tuple[ResponseGeometryDevelopmentMeasuredView, ...]
    records: tuple[CanonicalRecord, ...]
    fit_payload: bytes | None
    fit_views: tuple[ResponseGeometryDevelopmentMeasuredView, ...] = ()

    def record(self, kind: type[Record]) -> Record:
        values = [r for r in self.records if type(r) is kind]
        if len(values) != 1:
            raise ValueError("development task lacks its one exact preceding method record")
        return values[0]


def response_geometry_development_worker_artifact_id(
    context: TaskContext, input_id: str, continuation: ResponseGeometryDevelopmentAnalysisContinuationConfig | None = None,
) -> str:
    """Use the exact run-qualified artifact identity supplied by production.

    Protocol output IDs and worker artifact identities are distinct. External
    config identities are already fully bound; dependency identities include
    the actual run. A separate correction may declare the exact retained
    projection roster; it cannot substitute a bare ID or choose another run.
    """
    if continuation is not None:
        return continuation.artifact_id(context.run_id, input_id, task_id=context.task_id)
    return input_id if input_id.startswith("config-artifact.") else f"artifact.{context.run_id}.{input_id}"


def read_response_geometry_development_method_inputs(
    context: TaskContext,
    *,
    config: ResponseGeometryDevelopmentMethodConfig,
    native_context: str,
    role: DInputRole,
    preceding: tuple[type[CanonicalRecord], ...],
    include_calibration_views: bool = False,
    include_fit_views: bool = False,
    task_config: CanonicalRecord | None = None,
    task_config_id: str | None = None,
    continuation: ResponseGeometryDevelopmentAnalysisContinuationConfig | None = None,
) -> ResponseGeometryDevelopmentMethodInputs:
    """Reject foreign root roles by logical locator before opening outcome bytes.

    Production expansion must use these exact report/data/stage output locators.
    The later production proof checks them against the lowered graph.
    """
    roots = {"fit": range(16), "calibration": range(16, 32), "validation": range(32, 64)}[role]
    if include_calibration_views and (role != "fit" or ResponseGeometryDevelopmentCalibrationResult not in preceding):
        raise ValueError("development outer-fold calibration is permitted only for support fitting")
    if include_fit_views and (role != "validation" or ResponseGeometryDevelopmentFitResult not in preceding):
        raise ValueError("development shared fit views are permitted only at the post-fit validation stage")
    if (task_config is None) != (task_config_id is None):
        raise ValueError("development task config override requires its exact identity")
    actual_config, actual_config_id = task_config or config, task_config_id or config.config_id
    expected = {f"config-artifact.{actual_config_id}": actual_config.SCHEMA}
    declaration_input = continuation is not None and context.task_id == f"{DEVELOPMENT_PANEL_ID}.fit.{native_context}"
    if declaration_input:
        assert continuation is not None
        expected[continuation.config_id] = continuation.SCHEMA
    report_ports = {}
    for i in (
        *roots,
        *(range(16, 32) if include_calibration_views else ()),
        *(range(16) if include_fit_views else ()),
    ):
        for r in (1, 2):
            task = f"{DEVELOPMENT_PANEL_ID}.{native_context}.r{i:02d}.project.r{r}"
            report_ports[task] = (i, r)
            expected.update(
                {
                    f"{task}.report": ResponseGeometryDevelopmentViewReport.SCHEMA,
                    f"{task}.data": DEVELOPMENT_DATA_SCHEMA,
                    f"{task}.stage": LinkedCampaignStageEnvelope.SCHEMA,
                }
            )
    for kind in preceding:
        name = {
            ResponseGeometryDevelopmentFitResult: "fit",
            ResponseGeometryDevelopmentCalibrationResult: "calibrate",
            ResponseGeometryDevelopmentSupportResult: "support",
        }[kind]
        task = f"{DEVELOPMENT_PANEL_ID}.{name}.{native_context}"
        expected.update(
            {f"{task}.report": kind.SCHEMA, f"{task}.stage": LinkedCampaignStageEnvelope.SCHEMA}
        )
        if kind is ResponseGeometryDevelopmentFitResult:
            expected[f"{task}.data"] = DEVELOPMENT_FIT_SCHEMA
    try:
        retained = () if continuation is None else continuation.inputs_for_task(context.task_id)
        if continuation is not None and continuation.qualification.method != config:
            raise ValueError("development correction changes the original method configuration")
        if native_context not in ("assembling", "prepared") or (
            len(context.input_ports) != len(expected)
            or {p.artifact_id: p.payload_schema for p in context.input_ports}
            != {response_geometry_development_worker_artifact_id(context, key, continuation): schema for key, schema in expected.items()}
        ):
            raise ValueError("development method input locator/root role differs before outcome access")
        actual_ports = {p.artifact_id: p for p in context.input_ports}
        ports = {key: actual_ports[response_geometry_development_worker_artifact_id(context, key, continuation)] for key in expected}
        for row in retained:
            binding = ports[row.slot_id].binding
            if (binding.payload_schema != row.artifact.payload_schema
                    or binding.media_type != row.artifact.media_type or binding.size_bytes != row.artifact.size_bytes
                    or binding.kind is not WorkerInputKind.EXTERNAL):
                raise ValueError("development correction changes the retained projection port metadata")
        if (
            decode_port(ports[f"config-artifact.{actual_config_id}"], type(actual_config))
            != actual_config
        ):
            raise ValueError("development method input config differs from issued config")
        if declaration_input:
            assert continuation is not None
            if decode_port(ports[continuation.config_id], ResponseGeometryDevelopmentAnalysisContinuationConfig) != continuation:
                raise ValueError("development correction changes its exact retained-input declaration")
        records: list[CanonicalRecord] = []
        fit_payload = None
        for kind in preceding:
            name = {
                ResponseGeometryDevelopmentFitResult: "fit",
                ResponseGeometryDevelopmentCalibrationResult: "calibrate",
                ResponseGeometryDevelopmentSupportResult: "support",
            }[kind]
            task = f"{DEVELOPMENT_PANEL_ID}.{name}.{native_context}"
            record = decode_port(ports[f"{task}.report"], kind)
            if not isinstance(record, (ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentCalibrationResult, ResponseGeometryDevelopmentSupportResult)):
                raise ValueError("development preceding method record type differs")
            if record.config != ObjectIdentity.from_record(config.config_id, config) or (
                decode_port(ports[f"{task}.stage"], LinkedCampaignStageEnvelope)
                != response_geometry_development_method_stage(record)
            ):
                raise ValueError("development preceding method config/stage custody differs")
            records.append(record)
            if isinstance(record, ResponseGeometryDevelopmentFitResult):
                fit_payload = read_port(ports[f"{task}.data"], DEVELOPMENT_FIT_MAXIMUM_BYTES)
                if sha256(fit_payload).hexdigest() != record.data_sha256:
                    raise ValueError("development fitted model artifact differs from frozen result")
        by_type = {type(r): r for r in records}
        if ResponseGeometryDevelopmentCalibrationResult in by_type and ResponseGeometryDevelopmentFitResult in by_type:
            fit = cast(ResponseGeometryDevelopmentFitResult, by_type[ResponseGeometryDevelopmentFitResult])
            calibration = cast(ResponseGeometryDevelopmentCalibrationResult, by_type[ResponseGeometryDevelopmentCalibrationResult])
            if calibration.fit_result != ObjectIdentity.from_record(fit.result_id, fit):
                raise ValueError("development calibration belongs to a different frozen fit")
        if ResponseGeometryDevelopmentSupportResult in by_type:
            support = cast(ResponseGeometryDevelopmentSupportResult, by_type[ResponseGeometryDevelopmentSupportResult])
            fit = cast(ResponseGeometryDevelopmentFitResult, by_type[ResponseGeometryDevelopmentFitResult])
            calibration = cast(ResponseGeometryDevelopmentCalibrationResult, by_type[ResponseGeometryDevelopmentCalibrationResult])
            if (
                support.fit_result != ObjectIdentity.from_record(fit.result_id, fit)
                or support.calibration_result
                != ObjectIdentity.from_record(calibration.result_id, calibration)
                or support.input_reports != fit.input_reports
                or support.calibration_input_reports != calibration.input_reports
            ):
                raise ValueError("development support changes preceding fit/calibration custody")
        views = []
        for task, (i, r) in report_ports.items():
            report = decode_port(ports[f"{task}.report"], ResponseGeometryDevelopmentViewReport)
            if continuation is not None:
                report_ref = next(row for row in retained if row.slot_id == f"{task}.report")
                data_ref = next(row for row in retained if row.slot_id == f"{task}.data")
                stage_ref = next(row for row in retained if row.slot_id == f"{task}.stage")
                if (report.fingerprint() != report_ref.artifact.sha256
                        or report.data_sha256 != data_ref.artifact.sha256
                        or response_geometry_development_projection_stage(report).fingerprint() != stage_ref.artifact.sha256):
                    raise ValueError("development correction changes retained projection content")
            if (report.root.context, report.root.index, report.refinement) != (
                native_context,
                i,
                r,
            ) or (
                report.projection_config != config.projection_config
                or decode_port(ports[f"{task}.stage"], LinkedCampaignStageEnvelope)
                != response_geometry_development_projection_stage(report)
            ):
                raise ValueError("development measurement report changes locator/config/stage custody")
            data = read_port(ports[f"{task}.data"], 16 * 1024**2)
            views.append(read_response_geometry_development_measurements(report, data))
        measured = tuple(v for v in views if v.report.root.index in roots)
        calibration_views = tuple(
            v for v in views if include_calibration_views and v.report.root.index in range(16, 32)
        )
        fit_views = tuple(
            v for v in views if include_fit_views and v.report.root.index in range(16)
        )
        organize_response_geometry_development_views(measured, context=native_context, role=role)
        if role == "fit" and ResponseGeometryDevelopmentFitResult in by_type:
            fit = cast(ResponseGeometryDevelopmentFitResult, by_type[ResponseGeometryDevelopmentFitResult])
            if report_identities(measured) != fit.input_reports:
                raise ValueError("development support changes the preceding fit inputs")
        if include_calibration_views:
            organize_response_geometry_development_views(calibration_views, context=native_context, role="calibration")
            calibration = cast(ResponseGeometryDevelopmentCalibrationResult, by_type[ResponseGeometryDevelopmentCalibrationResult])
            if report_identities(calibration_views) != calibration.input_reports:
                raise ValueError(
                    "development outer-fold calibration changes the preceding calibration inputs"
                )
        if include_fit_views:
            organize_response_geometry_development_views(fit_views, context=native_context, role="fit")
            fit = cast(ResponseGeometryDevelopmentFitResult, by_type[ResponseGeometryDevelopmentFitResult])
            if report_identities(fit_views) != fit.input_reports:
                raise ValueError("development shared fitting views change the preceding frozen fit inputs")
        return ResponseGeometryDevelopmentMethodInputs(measured, calibration_views, tuple(records), fit_payload, fit_views)
    finally:
        for port in context.input_ports:
            port.close()


class ResponseGeometryDevelopmentDevelopmentTask:
    def __init__(self, manifest: CapabilityManifest, config: ResponseGeometryDevelopmentMethodConfig,
                 continuation: ResponseGeometryDevelopmentAnalysisContinuationConfig | None = None) -> None:
        self.manifest, self.config = manifest, config
        self.continuation = continuation

    def execute(self, context: TaskContext) -> RunnerResult:
        native_context = context.task_id.rsplit(".", 1)[-1]
        role = context.task_id.rsplit(".", 2)[-2]
        if (
            role not in ("fit", "calibrate", "support")
            or context.task_id != f"{DEVELOPMENT_PANEL_ID}.{role}.{native_context}"
        ):
            raise ValueError("development method task changes its declared role")
        preceding: tuple[type[CanonicalRecord], ...] = {
            "fit": (),
            "calibrate": (ResponseGeometryDevelopmentFitResult,),
            "support": (ResponseGeometryDevelopmentFitResult, ResponseGeometryDevelopmentCalibrationResult),
        }[role]
        inputs = read_response_geometry_development_method_inputs(
            context,
            config=self.config,
            native_context=native_context,
            role="calibration" if role == "calibrate" else "fit",
            preceding=preceding,
            include_calibration_views=role == "support",
            continuation=self.continuation,
        )
        result: ResponseGeometryDevelopmentFitResult | ResponseGeometryDevelopmentCalibrationResult | ResponseGeometryDevelopmentSupportResult
        data = None
        if role == "fit":
            groups = fit_response_geometry_development_context(inputs.views, context=native_context)
            result, data = write_response_geometry_development_fit(self.config, inputs.views, groups)
        else:
            fit = inputs.record(ResponseGeometryDevelopmentFitResult)
            assert inputs.fit_payload is not None
            groups = read_response_geometry_development_fit(fit, inputs.fit_payload)
            config_id = ObjectIdentity.from_record(self.config.config_id, self.config)
            fit_id = ObjectIdentity.from_record(fit.result_id, fit)
            if role == "calibrate":
                calibration = calibrate_response_geometry_development_context(groups, inputs.views, context=native_context)
                result = ResponseGeometryDevelopmentCalibrationResult(
                    config_id,
                    fit_id,
                    report_identities(inputs.views),
                    calibration,
                )
            else:
                calibrated = inputs.record(ResponseGeometryDevelopmentCalibrationResult)
                fitted, outer_calibrations = fit_response_geometry_development_support(
                    groups,
                    calibrated.calibration,
                    inputs.views,
                    calibration_views=inputs.calibration_views,
                    context=native_context,
                )
                result = ResponseGeometryDevelopmentSupportResult(
                    native_context,
                    config_id,
                    fit_id,
                    ObjectIdentity.from_record(calibrated.result_id, calibrated),
                    report_identities(inputs.views),
                    report_identities(inputs.calibration_views),
                    outer_calibrations,
                    tuple(ResponseGeometryDevelopmentSupportRow.from_fit(f) for f in fitted),
                )
        stage = response_geometry_development_method_stage(result)
        values = {result.SCHEMA: result.canonical_bytes(), stage.SCHEMA: stage.canonical_bytes()}
        if data is not None:
            values[DEVELOPMENT_FIT_SCHEMA] = data
        return output_result(
            context,
            values,
            (
                "exact-development-role-before-outcome-read",
                "authenticated-preceding-method-custody",
            ),
        )


class ResponseGeometryDevelopmentDevelopmentProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: ResponseGeometryDevelopmentProjectionConfig | ResponseGeometryDevelopmentMethodConfig,
        role: str,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
            or role not in ("projection", "development")
            or (role == "projection") != (type(config) is ResponseGeometryDevelopmentProjectionConfig)
        ):
            raise ValueError("development provider config/role/registry differs")
        self.registry, self.manifest, self.config, self.role = registry, manifest, config, role
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("development method runner registry/records differ")
        runner = (
            ResponseGeometryDevelopmentProjectionTask(self.manifest, self.config)
            if isinstance(self.config, ResponseGeometryDevelopmentProjectionConfig)
            else ResponseGeometryDevelopmentDevelopmentTask(self.manifest, self.config)
        )
        return (cast(TaskRunner, runner),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("development method execution registry/records differ")
        expected = (
            {f"{root.root_id}.project.r{r}" for root in development_roots() for r in (1, 2)}
            if self.role == "projection"
            else {
                f"{DEVELOPMENT_PANEL_ID}.{role}.{c}"
                for role in ("fit", "calibrate", "support")
                for c in ("assembling", "prepared")
            }
        )
        return config_payloads(plan, self.manifest, self.config, self.config.config_id, expected)

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("development method semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (ResponseGeometryDevelopmentViewReport, LinkedCampaignStageEnvelope)
            if self.role == "projection"
            else (
                ResponseGeometryDevelopmentFitResult,
                ResponseGeometryDevelopmentCalibrationResult,
                ResponseGeometryDevelopmentSupportResult,
                LinkedCampaignStageEnvelope,
            ),
            DEVELOPMENT_DATA_SCHEMA if self.role == "projection" else DEVELOPMENT_FIT_SCHEMA,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        if registry != self.registry:
            raise ValueError("development method adjudication registry differs")
        return None

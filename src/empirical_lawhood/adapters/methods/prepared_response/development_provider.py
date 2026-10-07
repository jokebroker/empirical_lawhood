"dependent refinement per-view reduction and nominal nested-fit runtime providers."

from dataclasses import replace
from typing import cast

from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskRunner,
    WorkerInputKind,
    WorkerInputPort,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, envelope, output_result, read_port, semantic_contracts
from empirical_lawhood.adapters.simulators.prepared_response.native_pair import MAXIMUM_PAIR_BYTES, NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.prepared_response.provider import prepared_native_stage_envelope
from empirical_lawhood.adapters.simulators.prepared_response.source_outputs import PreparedNativeTaskResult

from .development_fit import MAXIMUM_DEVELOPMENT_FIT_BYTES, PreparedResponseDevelopmentFitConfig, PreparedResponseDevelopmentModelFit, fit_prepared_response_development_structure
from .development_projection import PreparedResponseDevelopmentProjectionConfig, PreparedResponseDevelopmentViewProjection, project_prepared_response_development_view
from .development_selection import PreparedResponseDevelopmentNominalLibrary, PreparedResponseDevelopmentSelectionConfig, select_prepared_response_development_nominal_library
from .development_benchmarks import PreparedResponseDevelopmentConstantGainReport, build_prepared_response_development_constant_gain_report
from .models import STRUCTURES


def prepared_response_development_projection_envelope(
    report: PreparedResponseDevelopmentViewProjection,
) -> LinkedCampaignStageEnvelope:
    complete = report.mode_disposition == "RESOLVED" and bool(report.arrays()["fit_valid"].all())
    return envelope(
        report.report_id,
        report,
        report.report_id,
        LinkedCampaignStageRole.EVIDENCE_PROJECTION,
        complete,
        () if complete else ("INCOMPLETE_OR_UNAVAILABLE_NATIVE_DEVELOPMENT_VIEW",),
    )


def prepared_response_development_fit_envelope(result: PreparedResponseDevelopmentModelFit) -> LinkedCampaignStageEnvelope:
    stage = envelope(
        result.report_id,
        result,
        result.report_id,
        LinkedCampaignStageRole.METHOD_IDENTIFICATION,
        result.failure_reason is None,
        () if result.failure_reason is None else (result.failure_reason,),
    )
    if result.failure_reason is not None:
        stage = replace(stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE)
    return stage


def _read_config(
    context: TaskContext,
    config: PreparedResponseDevelopmentProjectionConfig | PreparedResponseDevelopmentFitConfig | PreparedResponseDevelopmentSelectionConfig,
    manifest: CapabilityManifest,
) -> None:
    external = tuple(port for port in context.input_ports if port.kind is WorkerInputKind.EXTERNAL)
    if (
        len(external) != 1
        or external[0].artifact_id != f"config-artifact.{config.config_id}"
        or external[0].payload_schema != config.SCHEMA
        or context.config.config_id != config.config_id
        or context.config.config_schema != config.SCHEMA
        or context.config.config_schema_sha256 != manifest.config_schema_sha256
        or context.config.content_sha256 != config.fingerprint()
        or context.config.artifact_id != external[0].artifact_id
        or decode_port(external[0], type(config)) != config
    ):
        raise ValueError("prepared dependent refinement method changes its exact external configuration")


def _group(
    context: TaskContext,
    materializations: tuple[str, ...],
    expected_schemas: tuple[str, ...],
) -> dict[str, WorkerInputPort]:
    ports = tuple(
        port for port in context.input_ports if port.materialization_id in materializations
    )
    if len(ports) != len(expected_schemas) or {port.payload_schema for port in ports} != set(
        expected_schemas
    ):
        raise ValueError("prepared dependent refinement dependency changes its complete output census")
    return {port.payload_schema: port for port in ports}


class PreparedResponseDevelopmentProjectionTask:
    def __init__(self, manifest: CapabilityManifest, config: PreparedResponseDevelopmentProjectionConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            slots = {
                f"{root.root_id}.project.r{refinement}": (root, refinement)
                for root in self.config.native_spec.roots
                for refinement in (1, 2)
            }
            if context.task_id not in slots:
                raise ValueError("prepared dependent refinement projection task is outside its root/view census")
            _read_config(context, self.config, self.manifest)
            inputs = []
            for receipt in context.dependency_receipts:
                group = _group(
                    context,
                    receipt.output_materialization_ids,
                    (
                        PreparedNativeTaskResult.SCHEMA,
                        NATIVE_PAIR_SCHEMA,
                        LinkedCampaignStageEnvelope.SCHEMA,
                    ),
                )
                result = decode_port(
                    group[PreparedNativeTaskResult.SCHEMA], PreparedNativeTaskResult
                )
                stage = decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA],
                    LinkedCampaignStageEnvelope,
                )
                if (
                    result.invocation.task_id != receipt.task_id
                    or stage != prepared_native_stage_envelope(result)
                ):
                    raise ValueError("prepared dependent refinement projection received detached native evidence")
                inputs.append((result, read_port(group[NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES)))
            report = project_prepared_response_development_view(
                self.config, *slots[context.task_id], tuple(inputs)
            )
            stage = prepared_response_development_projection_envelope(report)
            return output_result(
                context,
                {report.SCHEMA: report.canonical_bytes(), stage.SCHEMA: stage.canonical_bytes()},
                (
                    "complete-native-development-root-census",
                    "shared-acquisition-separate-numerical-view",
                    "no-native-effect-in-d-projection",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class PreparedResponseDevelopmentFitTask:
    def __init__(self, manifest: CapabilityManifest, config: PreparedResponseDevelopmentFitConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            slots = {
                f"prepared-response.dependent-refinement.fit.{context_name}.{structure}": (context_name, structure)
                for context_name in ("assembling", "prepared")
                for structure in STRUCTURES
            }
            if context.task_id not in slots:
                raise ValueError("prepared dependent refinement fit task is outside its context/structure census")
            _read_config(context, self.config, self.manifest)
            reports = {}
            for receipt in context.dependency_receipts:
                group = _group(
                    context,
                    receipt.output_materialization_ids,
                    (PreparedResponseDevelopmentViewProjection.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
                )
                report = decode_port(
                    group[PreparedResponseDevelopmentViewProjection.SCHEMA], PreparedResponseDevelopmentViewProjection
                )
                stage = decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA],
                    LinkedCampaignStageEnvelope,
                )
                if report.report_id != receipt.task_id or stage != prepared_response_development_projection_envelope(
                    report
                ):
                    raise ValueError("prepared dependent refinement fit received a detached view product")
                reports[report.report_id] = report
            context_name, structure = slots[context.task_id]
            expected = tuple(
                f"{root.root_id}.project.r{refinement}"
                for root in self.config.projection.native_spec.roots
                if root.context == context_name
                for refinement in (1, 2)
            )
            if set(reports) != set(expected):
                raise ValueError("prepared dependent refinement fit omits an assigned root/view product")
            result = fit_prepared_response_development_structure(
                self.config,
                context_name,
                structure,
                tuple(reports[report_id] for report_id in expected),
            )
            stage = prepared_response_development_fit_envelope(result)
            return output_result(
                context,
                {result.SCHEMA: result.canonical_bytes(), stage.SCHEMA: stage.canonical_bytes()},
                (
                    "frozen-whole-root-crossfit-plan",
                    "all-assigned-development-roots-retained",
                    "nominal-fit-is-not-law-qualification",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


def prepared_response_development_selection_adjudication(
    context: TaskContext, result: PreparedResponseDevelopmentNominalLibrary
) -> ScientificAdjudicationRecord:
    authority = context.scientific_adjudication_context
    if authority is None:
        raise ValueError("prepared dependent refinement selection lacks its authenticated adjudication context")
    selected = result.selected_structure is not None
    return ScientificAdjudicationRecord(
        f"adjudication.{context.run_id}.{context.task_id}",
        context.run_id,
        context.task_id,
        authority.execution_plan,
        context.input_materialization_ids,
        tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
            )
        ),
        context.dependency_receipt_ids,
        authority.evidence_world_id,
        authority.evidence_world_kind,
        authority.relation,
        authority.independent_unit_id,
        authority.information_cutoffs,
        authority.visibility_ceiling,
        authority.outcome_access,
        AdjudicationEvaluability.EVALUABLE,
        ScientificStatus.NOT_TESTED if selected else ScientificStatus.NOT_SUPPORTED,
        AdmissionStatus.NOT_EVALUATED,
        ("DEVELOPMENT_NOMINATION_ONLY_NOT_PROTECTED_CALIBRATION",)
        if selected
        else result.reason_codes,
        authority.fixture_scope_id,
        authority.plumbing_only,
    )


class PreparedResponseDevelopmentSelectionTask:
    TASK_ID = "prepared-response.dependent-refinement.select-nominal-library"

    def __init__(self, manifest: CapabilityManifest, config: PreparedResponseDevelopmentSelectionConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            if context.task_id != self.TASK_ID or context.scientific_adjudication_context is None:
                raise ValueError("prepared dependent refinement selection requires its exact adjudication task")
            _read_config(context, self.config, self.manifest)
            projections = {}
            fits = {}
            for receipt in context.dependency_receipts:
                ports = tuple(
                    port
                    for port in context.input_ports
                    if port.materialization_id in receipt.output_materialization_ids
                )
                schemas = {port.payload_schema for port in ports}
                if schemas == {
                    PreparedResponseDevelopmentViewProjection.SCHEMA,
                    LinkedCampaignStageEnvelope.SCHEMA,
                }:
                    group = {port.payload_schema: port for port in ports}
                    projection = decode_port(
                        group[PreparedResponseDevelopmentViewProjection.SCHEMA], PreparedResponseDevelopmentViewProjection
                    )
                    stage = decode_port(
                        group[LinkedCampaignStageEnvelope.SCHEMA],
                        LinkedCampaignStageEnvelope,
                    )
                    if (
                        projection.report_id != receipt.task_id
                        or stage != prepared_response_development_projection_envelope(projection)
                    ):
                        raise ValueError("prepared dependent refinement selection received a detached view product")
                    projections[projection.report_id] = projection
                elif schemas == {
                    PreparedResponseDevelopmentModelFit.SCHEMA,
                    LinkedCampaignStageEnvelope.SCHEMA,
                }:
                    group = {port.payload_schema: port for port in ports}
                    fit = decode_port(
                        group[PreparedResponseDevelopmentModelFit.SCHEMA], PreparedResponseDevelopmentModelFit,
                        maximum=MAXIMUM_DEVELOPMENT_FIT_BYTES,
                    )
                    stage = decode_port(
                        group[LinkedCampaignStageEnvelope.SCHEMA],
                        LinkedCampaignStageEnvelope,
                    )
                    if fit.report_id != receipt.task_id or stage != prepared_response_development_fit_envelope(fit):
                        raise ValueError("prepared dependent refinement selection received a detached nominal fit")
                    fits[fit.report_id] = fit
                else:
                    raise ValueError("prepared dependent refinement selection dependency changes its output schemas")
            expected_projections = tuple(
                f"{root.root_id}.project.r{refinement}"
                for root in self.config.fit.projection.native_spec.roots
                for refinement in (1, 2)
            )
            expected_fits = tuple(
                f"prepared-response.dependent-refinement.fit.{context_name}.{structure}"
                for context_name in ("assembling", "prepared")
                for structure in STRUCTURES
            )
            if set(projections) != set(expected_projections) or set(fits) != set(expected_fits):
                raise ValueError("prepared dependent refinement selection omits a required view or candidate fit")
            result = select_prepared_response_development_nominal_library(
                self.config,
                tuple(projections[value] for value in expected_projections),
                tuple(fits[value] for value in expected_fits),
            )
            benchmark = build_prepared_response_development_constant_gain_report(
                self.config.fit,
                tuple(projections[value] for value in expected_projections),
                tuple(fits[value] for value in expected_fits),
            )
            stage = envelope(
                context.task_id,
                result,
                result.library_id,
                LinkedCampaignStageRole.METHOD_IDENTIFICATION,
                result.selected_structure is not None,
                result.reason_codes,
            )
            if result.selected_structure is None:
                stage = replace(stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE)
            adjudication = prepared_response_development_selection_adjudication(context, result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    benchmark.SCHEMA: benchmark.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                    adjudication.SCHEMA: adjudication.canonical_bytes(),
                },
                (
                    "both-context-d-eligibility-intersection",
                    "measured-excluded-canary-cost-selection",
                    "nominal-library-does-not-publish-law",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class PreparedResponseDevelopmentMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: PreparedResponseDevelopmentProjectionConfig | PreparedResponseDevelopmentFitConfig | PreparedResponseDevelopmentSelectionConfig,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("prepared dependent refinement method changes its registry/configuration")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("prepared dependent refinement method runner registry/records differ")
        runner = (
            PreparedResponseDevelopmentProjectionTask(self.manifest, self.config)
            if isinstance(self.config, PreparedResponseDevelopmentProjectionConfig)
            else PreparedResponseDevelopmentFitTask(self.manifest, self.config)
            if isinstance(self.config, PreparedResponseDevelopmentFitConfig)
            else PreparedResponseDevelopmentSelectionTask(self.manifest, self.config)
        )
        return (cast(TaskRunner, runner),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("prepared dependent refinement method execution registry/records differ")
        expected = (
            {
                f"{root.root_id}.project.r{refinement}"
                for root in self.config.native_spec.roots
                for refinement in (1, 2)
            }
            if isinstance(self.config, PreparedResponseDevelopmentProjectionConfig)
            else {
                f"prepared-response.dependent-refinement.fit.{context_name}.{structure}"
                for context_name in ("assembling", "prepared")
                for structure in STRUCTURES
            }
            if isinstance(self.config, PreparedResponseDevelopmentFitConfig)
            else {PreparedResponseDevelopmentSelectionTask.TASK_ID}
        )
        return config_payloads(plan, self.manifest, self.config, self.config.config_id, expected)

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("prepared dependent refinement method semantic registry differs")
        return semantic_contracts(
            self.manifest,
            (
                (PreparedResponseDevelopmentViewProjection, LinkedCampaignStageEnvelope)
                if isinstance(self.config, PreparedResponseDevelopmentProjectionConfig)
                else (PreparedResponseDevelopmentModelFit, LinkedCampaignStageEnvelope)
                if isinstance(self.config, PreparedResponseDevelopmentFitConfig)
                else (
                    PreparedResponseDevelopmentNominalLibrary,
                    PreparedResponseDevelopmentConstantGainReport,
                    LinkedCampaignStageEnvelope,
                    ScientificAdjudicationRecord,
                )
            ),
            None,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("prepared dependent refinement method adjudication registry differs")
        if not isinstance(self.config, PreparedResponseDevelopmentSelectionConfig):
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=(
                f"{PreparedResponseDevelopmentSelectionTask.TASK_ID}.scientific-adjudication"
            ),
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            fixture_scope_id=None,
            plumbing_only=False,
        )

"""Production projection and protected preparation-policy development screen providers."""

from hashlib import sha256
from typing import Protocol, cast

from empirical_lawhood.adapters.methods.prepared_response.qualification_provider import _group
from empirical_lawhood.adapters.simulators.finite_response_law.native_artifact import MAXIMUM_PAIR_BYTES
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_native_artifact import PREPARATION_POLICY_NATIVE_PAIR_SCHEMA
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_provider import preparation_policy_native_stage
from empirical_lawhood.adapters.simulators.finite_response_law.preparation_policy_source_outputs import FiniteResponseLawPreparationPolicyNativeTaskResult
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, envelope, output_result, read_port, semantic_contracts
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner, WorkerInputKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
    ExternalInputSource,
    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
)
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.decoding import decode_canonical_bytes

from .law_payloads import FiniteResponseLawLowerPayload, MAXIMUM_LAW_BYTES
from .method_records import FiniteResponseLawQualificationReport
from .preparation_policy_projection import FiniteResponseLawPreparationPolicyProjectionConfig, FiniteResponseLawPreparationPolicyRootPanel, project_preparation_policy_root
from .preparation_screen_results import FiniteResponseLawRetainedProspectiveCloseoutReference, FiniteResponseLawPreparationScreenResult, FiniteResponseLawPreparationPolicyScreenConfig, FiniteResponseLawPreparationPolicyUpperPayloadFreeze, freeze_preparation_screen_outputs, unevaluable_preparation_screen_outputs
from .preparation_policy_screen import compute_preparation_screen_screen

PREPARATION_POLICY_EVALUATOR_TASK_ID = "finite-response-law.preparation-policy.screen"
PREPARATION_POLICY_EVIDENCE_PORT = "finite-response-law-preparation-policy-evidence-sources"


class FiniteResponseLawPreparationPolicyEvidenceSources(Protocol):
    @property
    def artifacts(self) -> tuple[ArtifactIdentity, ...]: ...

    def create_source(self, artifact: ArtifactIdentity) -> ExternalInputSource: ...


def preparation_policy_projection_stage(report: FiniteResponseLawPreparationPolicyRootPanel) -> LinkedCampaignStageEnvelope:
    return envelope(
        report.panel_id,
        report,
        report.panel_id,
        LinkedCampaignStageRole.EVIDENCE_PROJECTION,
        True,
        (),
    )


def _read_config(
    context: TaskContext,
    config: FiniteResponseLawPreparationPolicyProjectionConfig | FiniteResponseLawPreparationPolicyScreenConfig,
    manifest: CapabilityManifest,
    *,
    extra_external: tuple[ArtifactIdentity, ...] = (),
    outputs_per_dependency: int,
) -> dict[str, object]:
    external = tuple(port for port in context.input_ports if port.kind is WorkerInputKind.EXTERNAL)
    expected = {
        f"config-artifact.{config.config_id}": config.SCHEMA,
        **{artifact.artifact_id: artifact.payload_schema for artifact in extra_external},
    }
    if (
        {port.artifact_id: port.payload_schema for port in external} != expected
        or context.config.config_id != config.config_id
        or context.config.config_schema != config.SCHEMA
        or context.config.config_schema_sha256 != manifest.config_schema_sha256
        or context.config.content_sha256 != config.fingerprint()
        or context.config.artifact_id != f"config-artifact.{config.config_id}"
        or len(context.input_ports)
        != len(external) + outputs_per_dependency * len(context.dependency_receipts)
    ):
        raise ValueError("preparation-policy method changes config, evidence, or dependency roster")
    config_port = next(
        port for port in external if port.artifact_id == f"config-artifact.{config.config_id}"
    )
    if decode_port(config_port, type(config)) != config:
        raise ValueError("preparation-policy method received another frozen configuration")
    return {port.artifact_id: port for port in external}


class FiniteResponseLawPreparationPolicyProjectionTask:
    def __init__(self, manifest: CapabilityManifest, config: FiniteResponseLawPreparationPolicyProjectionConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            slots = {f"{root.stage_unit}.project": root for root in self.config.native_spec.roots}
            root = slots.get(context.task_id)
            if root is None:
                raise ValueError("preparation-policy projection task changes assigned root")
            _read_config(
                context, self.config, self.manifest, outputs_per_dependency=3
            )
            inputs = []
            for receipt in context.dependency_receipts:
                group = _group(
                    context,
                    receipt.task_id,
                    receipt.output_materialization_ids,
                    (
                        FiniteResponseLawPreparationPolicyNativeTaskResult.SCHEMA,
                        PREPARATION_POLICY_NATIVE_PAIR_SCHEMA,
                        LinkedCampaignStageEnvelope.SCHEMA,
                    ),
                )
                record = decode_port(
                    group[FiniteResponseLawPreparationPolicyNativeTaskResult.SCHEMA],
                    FiniteResponseLawPreparationPolicyNativeTaskResult,
                )
                stage = decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA],
                    LinkedCampaignStageEnvelope,
                )
                if record.invocation.task_id != receipt.task_id or stage != preparation_policy_native_stage(
                    record
                ):
                    raise ValueError("preparation-policy projection received detached native evidence")
                inputs.append(
                    (
                        record,
                        read_port(group[PREPARATION_POLICY_NATIVE_PAIR_SCHEMA], MAXIMUM_PAIR_BYTES),
                    )
                )
            report = project_preparation_policy_root(
                self.config,
                root,
                tuple(sorted(inputs, key=lambda value: value[0].invocation.task_id)),
            )
            stage = preparation_policy_projection_stage(report)
            return output_result(
                context,
                {report.SCHEMA: report.canonical_bytes(), stage.SCHEMA: stage.canonical_bytes()},
                (
                    "complete-nine-schedule-root-panel",
                    "explicit-observed-valid-masks",
                    "retained-prefix-and-handoff-features",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


def _authenticate_lower_qualification(
    lower: FiniteResponseLawLowerPayload, artifact: ArtifactIdentity, qualification: FiniteResponseLawQualificationReport
) -> None:
    # The report's constructor authenticates the ordered four-boundary join.
    # Bind the lower boundary's actual publication, not text elsewhere in it.
    result = qualification.qualifications[0]
    calibration = qualification.calibration.boundaries[0]
    publication = qualification.publications[0]
    published = publication.artifact
    if (
        lower.q is None
        or lower.calibration != calibration.identity
        or lower.q != calibration.q
        or lower.development_manifest != calibration.development_manifest
        or lower.frozen_coefficients != calibration.frozen_coefficients
        or published.artifact_id != lower.payload_id
        or (published.sha256, published.payload_schema, published.media_type, published.size_bytes)
        != (
            artifact.sha256,
            artifact.payload_schema,
            artifact.media_type,
            artifact.size_bytes,
        )
        or result.scientific_status is not ScientificStatus.SUPPORTED
        or result.payload_publication
        != ObjectIdentity.from_record(publication.receipt_id, publication)
        or result.candidate_evaluator != publication.candidate_evaluator
        or result.response_law is None
        or result.response_law.evaluator != publication.candidate_evaluator
    ):
        raise ValueError("preparation-policy development entry lacks the exact supported lower publication")


def _authenticate_preparation_screen_entry(
    config: FiniteResponseLawPreparationPolicyScreenConfig, ports: dict[str, object]
) -> FiniteResponseLawLowerPayload:
    raise ValueError("FINITE_RESPONSE_LAW_CURRENT_PREPARATION_ELIGIBILITY_EXPORT_REQUIRED")
    raw = {}
    for artifact in (
        config.lower_artifact,
        config.lower_qualification,
        config.prospective_adjudication,
        config.prospective_closeout,
    ):
        port = ports[artifact.artifact_id]
        value = read_port(port, artifact.size_bytes)  # type: ignore[arg-type]
        if len(value) != artifact.size_bytes or sha256(value).hexdigest() != artifact.sha256:
            raise ValueError("preparation-policy development entry evidence differs from its exact artifact identity")
        raw[artifact.artifact_id] = value
    lower = decode_canonical_bytes(
        raw[config.lower_artifact.artifact_id],
        FiniteResponseLawLowerPayload,
        maximum_bytes=MAXIMUM_LAW_BYTES,
    )
    qualification = decode_canonical_bytes(
        raw[config.lower_qualification.artifact_id],
        FiniteResponseLawQualificationReport,
        maximum_bytes=config.lower_qualification.size_bytes,
    )
    adjudication = decode_canonical_bytes(
        raw[config.prospective_adjudication.artifact_id],
        ScientificAdjudicationRecord,
        maximum_bytes=config.prospective_adjudication.size_bytes,
    )
    closeout = decode_canonical_bytes(
        raw[config.prospective_closeout.artifact_id],
        FiniteResponseLawRetainedProspectiveCloseoutReference,
        maximum_bytes=config.prospective_closeout.size_bytes,
    )
    _authenticate_lower_qualification(lower, config.lower_artifact, qualification)
    if (
        lower.identity != config.lower_identity
        or adjudication.scientific_status is not ScientificStatus.SUPPORTED
        or adjudication.evaluability is not AdjudicationEvaluability.EVALUABLE
        or closeout.preparation_policy_eligible is not True
        or closeout.formal_completion is not True
        or closeout.canonical_adjudication_status != "ADJUDICATED"
    ):
        raise ValueError("preparation-policy development entry lacks exact lower qualification and finite response-law evaluation eligibility")
    return lower


def preparation_screen_adjudication(
    context: TaskContext, result: FiniteResponseLawPreparationScreenResult
) -> ScientificAdjudicationRecord:
    authority = context.scientific_adjudication_context
    if authority is None:
        raise ValueError("preparation-policy development screen lacks authenticated adjudication context")
    return ScientificAdjudicationRecord(
        adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
        run_id=context.run_id,
        adjudication_task_id=context.task_id,
        execution_plan=authority.execution_plan,
        input_materialization_ids=context.input_materialization_ids,
        output_logical_artifact_ids=tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
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
        if result.scientific_status.value == "UNEVALUABLE"
        else AdjudicationEvaluability.EVALUABLE,
        scientific_status=result.scientific_status,
        admission_status=AdmissionStatus.UNEVALUABLE
        if result.scientific_status.value == "UNEVALUABLE"
        else AdmissionStatus.NOT_EVALUATED,
        reason_codes=result.unevaluable_reason_codes
        if result.unevaluable_reason_codes
        else (
            "FINITE_RESPONSE_LAW_PREPARATION_SCREENING_ALL_THREE_DEVELOPMENT_GATES_PASSED"
            if result.gates_passed
            else "FINITE_RESPONSE_LAW_PREPARATION_SCREENING_ONE_OR_MORE_DEVELOPMENT_GATES_NOT_PASSED",
        ),
        fixture_scope_id=authority.fixture_scope_id,
        plumbing_only=authority.plumbing_only,
    )


class FiniteResponseLawPreparationPolicyScreenTask:
    def __init__(self, manifest: CapabilityManifest, config: FiniteResponseLawPreparationPolicyScreenConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        raise ValueError("FINITE_RESPONSE_LAW_CURRENT_PREPARATION_ELIGIBILITY_EXPORT_REQUIRED")
        try:
            if (
                context.task_id != PREPARATION_POLICY_EVALUATOR_TASK_ID
                or context.scientific_adjudication_context is None
            ):
                raise ValueError("preparation-policy development evaluator changes task or lacks adjudication authority")
            evidence = (
                self.config.lower_artifact,
                self.config.lower_qualification,
                self.config.prospective_adjudication,
                self.config.prospective_closeout,
            )
            ports = _read_config(
                context,
                self.config,
                self.manifest,
                extra_external=evidence,
                outputs_per_dependency=2,
            )
            lower = _authenticate_preparation_screen_entry(self.config, ports)
            panels = []
            for receipt in context.dependency_receipts:
                group = _group(
                    context,
                    receipt.task_id,
                    receipt.output_materialization_ids,
                    (FiniteResponseLawPreparationPolicyRootPanel.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
                )
                report = decode_port(group[FiniteResponseLawPreparationPolicyRootPanel.SCHEMA], FiniteResponseLawPreparationPolicyRootPanel)
                stage = decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA],
                    LinkedCampaignStageEnvelope,
                )
                if report.panel_id != receipt.task_id.removesuffix(".project") + ".preparation-policy-root-panel" or stage != preparation_policy_projection_stage(report):
                    raise ValueError("preparation-policy development evaluator received detached root projection")
                panels.append(report)
            ordered = tuple(
                next(report for report in panels if report.root == root)
                for root in self.config.projection.native_spec.roots
            )
            missing_reasons = tuple(
                sorted(
                    {
                        *(
                            "FINITE_RESPONSE_LAW_PREPARATION_SCREENING_REQUIRED_HANDOFF_FEATURE_MISSING"
                            for panel in ordered
                            if any(
                                value is None
                                for schedule in panel.handoff_features
                                for view in schedule
                                for value in view
                            )
                        ),
                        *(
                            "FINITE_RESPONSE_LAW_PREPARATION_SCREENING_REQUIRED_RESPONSE_UNOBSERVED"
                            for panel in ordered
                            if any(not cell.observed for cell in panel.cells)
                        ),
                        *(
                            "FINITE_RESPONSE_LAW_PREPARATION_SCREENING_REQUIRED_RESPONSE_INVALID"
                            for panel in ordered
                            if any(cell.observed and not cell.valid for cell in panel.cells)
                        ),
                    }
                )
            )
            if missing_reasons:
                result = unevaluable_preparation_screen_outputs(self.config, ordered, lower, missing_reasons)
                upper = None
            else:
                computation = compute_preparation_screen_screen(ordered, lower)
                result, upper = freeze_preparation_screen_outputs(self.config, ordered, lower, computation)
            freeze = FiniteResponseLawPreparationPolicyUpperPayloadFreeze(
                ObjectIdentity.from_record(result.result_id, result),
                upper,
                "FROZEN"
                if upper is not None
                else "UNEVALUABLE"
                if result.unevaluable_reason_codes
                else "GATES_NOT_PASSED",
            )
            stage = envelope(
                context.task_id,
                result,
                result.result_id,
                LinkedCampaignStageRole.METHOD_IDENTIFICATION,
                result.gates_passed,
                ()
                if result.gates_passed
                else result.unevaluable_reason_codes
                or ("FINITE_RESPONSE_LAW_PREPARATION_SCREENING_DEVELOPMENT_GATE_NEGATIVE",),
            )
            adjudication = preparation_screen_adjudication(context, result)
            return output_result(
                context,
                {
                    result.SCHEMA: result.canonical_bytes(),
                    freeze.SCHEMA: freeze.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                    adjudication.SCHEMA: adjudication.canonical_bytes(),
                },
                (
                    "unchanged-qualified-lower-package",
                    "four-outer-three-coefficient-fold-screen",
                    "separate-a-j-c-and-three-development-gates",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class FiniteResponseLawPreparationPolicyMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: FiniteResponseLawPreparationPolicyProjectionConfig | FiniteResponseLawPreparationPolicyScreenConfig,
        evidence_sources: FiniteResponseLawPreparationPolicyEvidenceSources | None = None,
    ) -> None:
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("preparation-policy method changes registry/configuration")
        expected = (
            ()
            if isinstance(config, FiniteResponseLawPreparationPolicyProjectionConfig)
            else (
                config.lower_artifact,
                config.lower_qualification,
                config.prospective_adjudication,
                config.prospective_closeout,
            )
        )
        if (evidence_sources is None) != (not expected) or (
            evidence_sources is not None
            and evidence_sources.artifacts != tuple(sorted(expected, key=lambda value: value.artifact_id))
        ):
            raise ValueError("preparation-policy method lacks exact lower/finite response-law evaluation evidence port")
        self.registry, self.manifest, self.config = registry, manifest, config
        self.evidence_sources = evidence_sources
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("preparation-policy method runner registry/records differ")
        runner = (
            FiniteResponseLawPreparationPolicyProjectionTask(self.manifest, self.config)
            if isinstance(self.config, FiniteResponseLawPreparationPolicyProjectionConfig)
            else FiniteResponseLawPreparationPolicyScreenTask(self.manifest, self.config)
        )
        return (cast(TaskRunner, runner),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if isinstance(self.config, FiniteResponseLawPreparationPolicyScreenConfig):
            raise ValueError("FINITE_RESPONSE_LAW_CURRENT_PREPARATION_ELIGIBILITY_EXPORT_REQUIRED")
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("preparation-policy method execution changes registry/records")
        expected_tasks = (
            {f"{root.stage_unit}.project" for root in self.config.native_spec.roots}
            if isinstance(self.config, FiniteResponseLawPreparationPolicyProjectionConfig)
            else {PREPARATION_POLICY_EVALUATOR_TASK_ID}
        )
        values = list(
            config_payloads(
                plan, self.manifest, self.config, self.config.config_id, expected_tasks
            )
        )
        if self.evidence_sources is not None:
            for artifact in self.evidence_sources.artifacts:
                lineage = ArtifactLineageParent(
                    ObjectIdentity.from_record(artifact.artifact_id, artifact),
                    VisibilityCeiling.OUTCOME_VISIBLE,
                    OutcomeAccess.DEVELOPMENT_VISIBLE,
                )
                values.append(
                    ExternalInputPayload(
                        artifact.artifact_id,
                        artifact.payload_schema,
                        ArtifactProfile.CANONICAL_JSON,
                        artifact.media_type,
                        self.evidence_sources.create_source(artifact),
                        artifact.size_bytes,
                        artifact.sha256,
                        artifact.size_bytes,
                        min(artifact.size_bytes, MAX_EXTERNAL_INPUT_CHUNK_BYTES),
                        VisibilityCeiling.OUTCOME_VISIBLE,
                        OutcomeAccess.DEVELOPMENT_VISIBLE,
                        (lineage.visibility_ceiling,),
                        (lineage,),
                        artifact.sha256,
                    )
                )
        return tuple(sorted(values, key=lambda value: value.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("preparation-policy method semantic registry differs")
        records: tuple[type[CanonicalRecord], ...] = (
            (FiniteResponseLawPreparationPolicyRootPanel, LinkedCampaignStageEnvelope)
            if isinstance(self.config, FiniteResponseLawPreparationPolicyProjectionConfig)
            else (
                FiniteResponseLawPreparationScreenResult,
                FiniteResponseLawPreparationPolicyUpperPayloadFreeze,
                LinkedCampaignStageEnvelope,
                ScientificAdjudicationRecord,
            )
        )
        return semantic_contracts(self.manifest, records, None)

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        if registry != self.registry:
            raise ValueError("preparation-policy method adjudication registry differs")
        if isinstance(self.config, FiniteResponseLawPreparationPolicyProjectionConfig):
            return None
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=f"{PREPARATION_POLICY_EVALUATOR_TASK_ID}.scientific-adjudication",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            fixture_scope_id=None,
            plumbing_only=False,
        )

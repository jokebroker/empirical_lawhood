"""Aggregate authenticated projections, then adjudicate through a separate task."""

from hashlib import sha256
from typing import cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import OperationalStatus
from empirical_lawhood.planning.linked_campaign import LinkedCampaignStageRole
from empirical_lawhood.runtime.adjudication import (
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, CanonicalTaskReceipt
from empirical_lawhood.runtime.capabilities import CapabilityManifest, CapabilityRegistry
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskRunner, WorkerInputKind
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignStageEnvelope
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
    MAX_EXTERNAL_INPUT_CHUNK_BYTES,
)
from empirical_lawhood.runtime.source_resolution import (
    ContentAddressedInputRequirement,
    ContentAddressedInputKind,
    ContentAddressedInputResolver,
)
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, read_port, output_result, semantic_contracts, envelope
from empirical_lawhood.adapters.methods.prepared_response.qualification_provider import _group
from ..method_provider import _ResolvedInputSource
from ..native_provider import native_adjudication, projection_stage
from ..native_records import FiniteResponseLawNativeEvaluation, native_method_types
from .contracts import FiniteResponseLawRetainedCompletionConfig, MAX_INPUT_BYTES


def aggregate_retained(
    config: FiniteResponseLawRetainedCompletionConfig, raw: dict[str, bytes]
) -> FiniteResponseLawNativeEvaluation:
    """Authenticate every original task; reuse the unchanged native census record."""
    if set(raw) != {a.artifact_id for a in config.inputs}:
        raise ValueError("Retained completion omits or adds a projection input")
    for artifact in config.inputs:
        value = raw[artifact.artifact_id]
        if len(value) != artifact.size_bytes or sha256(value).hexdigest() != artifact.sha256:
            raise ValueError("Retained completion input differs from its frozen identity")
    reports = []
    projection_type, evaluation_type, _, view_type, completion_type = native_method_types(
        config.native_source
    )
    for row in config.projections:
        receipt = decode_canonical_bytes(
            raw[row.receipt.artifact_id], CanonicalTaskReceipt, maximum_bytes=MAX_INPUT_BYTES
        )
        if (
            ObjectIdentity.from_record(receipt.receipt_id, receipt) != row.expected_receipt
            or receipt.run_id != config.retained_run_id
            or receipt.task_id != row.task_id
            or receipt.implementation_commit != config.retained_implementation_commit
            or receipt.operational_status is not OperationalStatus.SUCCEEDED
        ):
            raise ValueError("Retained projection task custody is contradictory")
        (logical,) = (
            a
            for a in receipt.output_logical_artifacts
            if a.payload_schema == row.report.payload_schema
        )
        (physical,) = (
            a
            for a in receipt.output_materializations
            if a.logical_artifact_id == row.report.artifact_id
        )
        if (
            logical.logical_artifact_id != row.report.artifact_id
            or logical.content_sha256 != row.report.sha256
            or logical.outcome_access is not OutcomeAccess.EVALUATION_SEALED
            or logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
            or physical.physical_sha256 != row.report.sha256
            or physical.size_bytes != row.report.size_bytes
            or physical.compression != "none"
            or physical.partition_selector is not None
        ):
            raise ValueError("Retained projection is detached from protected output custody")
        report = decode_canonical_bytes(
            raw[row.report.artifact_id],
            view_type,
            maximum_bytes=MAX_INPUT_BYTES,
        )
        (stage,) = (
            a
            for a in receipt.output_logical_artifacts
            if a.payload_schema == LinkedCampaignStageEnvelope.SCHEMA
        )
        if (
            report.report_id != row.task_id
            or projection_stage(report).fingerprint() != stage.content_sha256
        ):
            raise ValueError("Retained projection changes its original completed stage")
        reports.append(report)
    return completion_type(
        evaluation_type(projection_type(config.native_source)),
        tuple(reports),
    )


def completion_stage(
    config: FiniteResponseLawRetainedCompletionConfig, result: FiniteResponseLawNativeEvaluation
) -> LinkedCampaignStageEnvelope:
    return envelope(
        config.aggregate_task_id,
        result,
        result.evaluation_id,
        LinkedCampaignStageRole.METHOD_IDENTIFICATION,
        not result.reasons,
        result.reasons,
    )


class FiniteResponseLawRetainedCompletionTask:
    def __init__(self, manifest: CapabilityManifest, config: FiniteResponseLawRetainedCompletionConfig) -> None:
        self.manifest, self.config = manifest, config

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            config = self.config
            freeze, aggregate, adjudicate = (
                config.freeze_task_id,
                config.aggregate_task_id,
                config.adjudicate_task_id,
            )
            (config_port,) = (
                p
                for p in context.input_ports
                if p.artifact_id == f"config-artifact.{config.config_id}"
            )
            if (
                decode_port(config_port, type(config)) != config
                or context.config.content_sha256 != config.fingerprint()
            ):
                raise ValueError("Retained completion substitutes its issued config")
            if context.task_id == freeze:
                if context.dependency_receipts or len(context.input_ports) != 2:
                    raise ValueError("Retained input freeze may read only its issued configuration")
                (source_port,) = (
                    p for p in context.input_ports if p.artifact_id == config.native_source.spec_id
                )
                if decode_port(source_port, type(config.native_source)) != config.native_source:
                    raise ValueError(
                        "Retained completion changes its native numerical configuration"
                    )
                identity = ObjectIdentity.from_record(config.config_id, config)
                return output_result(
                    context,
                    {identity.SCHEMA: identity.canonical_bytes()},
                    ("retained-input-identities-frozen",),
                )
            if context.task_id == aggregate:
                if (
                    len(context.dependency_receipts) != 1
                    or context.dependency_receipts[0].task_id != freeze
                    or len(context.input_ports) != len(config.inputs) + 2
                ):
                    raise ValueError("Retained aggregation changes its exact external-input census")
                dependency = context.dependency_receipts[0]
                group = _group(
                    context, freeze, dependency.output_materialization_ids, (ObjectIdentity.SCHEMA,)
                )
                if decode_port(
                    group[ObjectIdentity.SCHEMA], ObjectIdentity
                ) != ObjectIdentity.from_record(config.config_id, config):
                    raise ValueError("Retained aggregation changes its frozen input identities")
                raw = {
                    p.artifact_id: read_port(p, MAX_INPUT_BYTES)
                    for p in context.input_ports
                    if p is not config_port and p.kind is WorkerInputKind.EXTERNAL
                }
                result = aggregate_retained(config, raw)
                stage = completion_stage(config, result)
                values = {
                    result.SCHEMA: result.canonical_bytes(),
                    stage.SCHEMA: stage.canonical_bytes(),
                }
            elif context.task_id == adjudicate:
                if (
                    len(context.dependency_receipts) != 1
                    or context.dependency_receipts[0].task_id != aggregate
                    or len(context.input_ports) != 3
                ):
                    raise ValueError("Native adjudication lacks its completed aggregate dependency")
                dependency = context.dependency_receipts[0]
                *_, completion_type = native_method_types(config.native_source)
                group = _group(
                    context,
                    aggregate,
                    dependency.output_materialization_ids,
                    (completion_type.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
                )
                result = decode_port(
                    group[completion_type.SCHEMA], completion_type
                )
                if result.config.projection.native_spec != config.native_source or decode_port(
                    group[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
                ) != completion_stage(config, result):
                    raise ValueError("Native adjudication changes its retained aggregate")
                adjudication = native_adjudication(context, result)
                values = {adjudication.SCHEMA: adjudication.canonical_bytes()}
            else:
                raise ValueError("Retained completion task is outside the issued roster")
            return output_result(
                context,
                values,
                (
                    "retained-projection-custody",
                    "no-native-recomputation",
                    "separate-adjudication-publication",
                ),
            )
        finally:
            for port in context.input_ports:
                port.close()


class FiniteResponseLawRetainedCompletionProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: FiniteResponseLawRetainedCompletionConfig,
        resolver: ContentAddressedInputResolver,
    ):
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version) != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError("Retained completion registry/configuration differs")
        self.registry, self.manifest, self.config, self.resolver = (
            registry,
            manifest,
            config,
            resolver,
        )
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("Retained completion runner registry differs")
        return (cast(TaskRunner, FiniteResponseLawRetainedCompletionTask(self.manifest, self.config)),)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if (
            plan.registry_sha256 != self.registry_sha256
            or source_records
            or {t.task_id for t in plan.tasks}
            != {
                self.config.freeze_task_id,
                self.config.aggregate_task_id,
                self.config.adjudicate_task_id,
            }
        ):
            raise ValueError("Retained completion changes its nonacquiring task roster")
        config = self.config
        values = list(
            config_payloads(
                plan,
                self.manifest,
                config,
                config.config_id,
                {config.freeze_task_id, config.aggregate_task_id, config.adjudicate_task_id},
            )
        )
        values.append(
            ExternalInputPayload.from_bytes(
                logical_artifact_id=config.native_source.spec_id,
                payload_schema=config.native_source.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                media_type="application/vnd.empirical-lawhood.canonical+json",
                payload=config.native_source.canonical_bytes(),
                visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                outcome_access=OutcomeAccess.OUTCOME_BLIND,
                parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                lineage_parents=(
                    ArtifactLineageParent(
                        ObjectIdentity.from_record(config.config_id, config),
                        VisibilityCeiling.PROSPECTIVE,
                        OutcomeAccess.OUTCOME_BLIND,
                    ),
                ),
                logical_content_sha256=config.native_source.fingerprint(),
            )
        )
        expected = {a.artifact_id: a for a in config.inputs}
        for task in plan.tasks:
            actual = {
                s.logical_artifact_id: s
                for s in task.external_inputs
                if s.logical_artifact_id
                not in (f"config-artifact.{config.config_id}", config.native_source.spec_id)
            }
            if set(actual) != (set(expected) if task.task_id == config.aggregate_task_id else set()):
                raise ValueError(
                    "Retained completion omits or substitutes an issued external input"
                )
            for key, spec in actual.items():
                artifact = expected[key]
                if (
                    spec.expected_content_sha256 != artifact.sha256
                    or spec.expected_payload_schema != artifact.payload_schema
                    or spec.expected_media_type != artifact.media_type
                    or spec.expected_outcome_access is not OutcomeAccess.EVALUATION_SEALED
                    or spec.expected_visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
                ):
                    raise ValueError("Retained completion changes input content or evidence access")
        for artifact in config.inputs:
            requirement = ContentAddressedInputRequirement(
                f"input.{artifact.artifact_id}",
                ContentAddressedInputKind.SOURCE_MATERIALIZATION,
                artifact.sha256,
                artifact.payload_schema,
                artifact.media_type,
                MAX_INPUT_BYTES,
                artifact.size_bytes,
                OutcomeAccess.EVALUATION_SEALED,
                VisibilityCeiling.PROSPECTIVE,
            )
            parent = ArtifactLineageParent(
                ObjectIdentity.from_record(config.config_id, config),
                VisibilityCeiling.PROSPECTIVE,
                OutcomeAccess.OUTCOME_BLIND,
            )
            values.append(
                ExternalInputPayload(
                    logical_artifact_id=artifact.artifact_id,
                    payload_schema=artifact.payload_schema,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type=artifact.media_type,
                    source=_ResolvedInputSource(self.resolver, requirement),
                    size_bytes=artifact.size_bytes,
                    source_sha256=artifact.sha256,
                    maximum_bytes=artifact.size_bytes,
                    maximum_chunk_bytes=min(artifact.size_bytes, MAX_EXTERNAL_INPUT_CHUNK_BYTES),
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.EVALUATION_SEALED,
                    parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                    lineage_parents=(parent,),
                    logical_content_sha256=artifact.sha256,
                )
            )
        return tuple(sorted(values, key=lambda p: p.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("Retained completion semantic registry differs")
        *_, completion_type = native_method_types(self.config.native_source)
        return semantic_contracts(
            self.manifest,
            (
                ObjectIdentity,
                completion_type,
                LinkedCampaignStageEnvelope,
                ScientificAdjudicationRecord,
            ),
            None,
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        if registry != self.registry:
            raise ValueError("Retained completion adjudication registry differs")
        return ScientificAdjudicationOutputContract(
            capability_key=self.manifest.capability_key,
            capability_version=self.manifest.capability_version,
            output_id=f"{self.config.adjudicate_task_id}.report",
            payload_schema=ScientificAdjudicationRecord.SCHEMA,
            fixture_scope_id=None,
            plumbing_only=False,
        )

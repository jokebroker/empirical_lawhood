"Finite response-law tasks over existing worker, custody, payload and qualification ports."

import json
from collections.abc import Iterator
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import cast

from empirical_lawhood.adapters.methods.prepared_response.qualification_provider import _group
from empirical_lawhood.adapters.simulators.response_geometry_prospective.provider import config_payloads, decode_port, envelope, output_result, read_port, semantic_contracts
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
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
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    CanonicalTaskReceipt,
)
from empirical_lawhood.runtime.candidate_payloads import CandidatePayloadPlane
from empirical_lawhood.runtime.capabilities import (
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskRunner,
    WorkerInputKind,
)
from empirical_lawhood.runtime.linked_campaigns import LinkedCampaignDisposition, LinkedCampaignStageEnvelope
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

from .calibration_operands import BOUNDARIES, PREDICTION_SCHEMA, authenticated_evaluation, calibration_operands, restore_calibration_operands
from .method_records import ADJUDICATE, CALIBRATE, COEFFICIENT_SCHEMA, FREEZE, QUALIFY, READOUT_SCHEMA, FiniteResponseLawAssignedCalibrationMethodConfig, FiniteResponseLawAssignedCalibrationReport, FiniteResponseLawAssignedQualificationReport, FiniteResponseLawCalibrationMethodConfig, FiniteResponseLawCalibrationReport, FiniteResponseLawQualificationReport
from .terminal import qualify_calibration

CANDIDATE_PAYLOAD_PORT = "candidate-payload-publisher"
INPUT_RESOLVER_PORT = "candidate-input-resolver"
MAXIMUM_INPUT_BYTES = 8 * 1024**2


def _report_types(
    config: FiniteResponseLawCalibrationMethodConfig,
) -> tuple[type[FiniteResponseLawCalibrationReport], type[FiniteResponseLawQualificationReport]]:
    if type(config) is FiniteResponseLawAssignedCalibrationMethodConfig:
        return FiniteResponseLawAssignedCalibrationReport, FiniteResponseLawAssignedQualificationReport
    if type(config) is FiniteResponseLawCalibrationMethodConfig:
        return FiniteResponseLawCalibrationReport, FiniteResponseLawQualificationReport
    raise ValueError("Finite response-law method has an undeclared issued configuration")


@dataclass
class _ResolvedInputSource:
    """Use the existing stream port; resolve bytes only when custody consumes it."""

    resolver: ContentAddressedInputResolver
    requirement: ContentAddressedInputRequirement
    closed: bool = False

    def chunks(self, maximum_chunk_bytes: int) -> Iterator[bytes]:
        if self.closed or not 0 < maximum_chunk_bytes <= MAX_EXTERNAL_INPUT_CHUNK_BYTES:
            raise ValueError("Finite response-law input source is closed or has an invalid chunk bound")
        self.closed = True
        resolved = self.resolver.resolve(self.requirement, materialize=True)
        if resolved.payload is None:
            raise ValueError(
                "Finite response-law resolver did not materialize the declared bounded input"
            )
        for start in range(0, len(resolved.payload), maximum_chunk_bytes):
            yield resolved.payload[start : start + maximum_chunk_bytes]

    def close(self) -> None:
        self.closed = True


def calibration_stage(report: FiniteResponseLawCalibrationReport) -> LinkedCampaignStageEnvelope:
    # Computing an infinite quantile is a completed calibration. Only the sole
    # qualification service decides its scientific consequence in the next task.
    return envelope(
        CALIBRATE,
        report,
        report.report_id,
        LinkedCampaignStageRole.METHOD_IDENTIFICATION,
        True,
        (),
    )


def _artifact(
    context: TaskContext, schema: str, raw: bytes, role: str, media: str
) -> ArtifactIdentity:
    ports = tuple(p for p in context.output_ports if p.payload_schema == schema)
    if len(ports) != 1 or ports[0].logical_artifact_id is None:
        raise ValueError(
            "Finite response-law output lacks its unique production logical artifact identity"
        )
    return ArtifactIdentity(
        ports[0].logical_artifact_id,
        role,
        schema,
        sha256(raw).hexdigest(),
        media,
        len(raw),
    )


def qualification_disposition(
    result: FiniteResponseLawQualificationReport,
) -> tuple[ScientificStatus, tuple[str, ...], LinkedCampaignStageEnvelope]:
    statuses = tuple(q.scientific_status for q in result.qualifications[:2])
    status = (
        ScientificStatus.UNEVALUABLE
        if ScientificStatus.UNEVALUABLE in statuses
        else ScientificStatus.NOT_SUPPORTED
        if ScientificStatus.NOT_SUPPORTED in statuses
        else ScientificStatus.SUPPORTED
        if statuses == (ScientificStatus.SUPPORTED, ScientificStatus.SUPPORTED)
        else ScientificStatus.UNEVALUABLE
    )
    reasons = tuple(
        sorted({r for q in result.qualifications[:2] for r in q.reason_codes})
    ) or ("PRIMARY_LAW_QUALIFICATIONS_COMPLETE",)
    stage = envelope(
        QUALIFY,
        result,
        result.report_id,
        LinkedCampaignStageRole.LAW_QUALIFICATION,
        status is ScientificStatus.SUPPORTED,
        () if status is ScientificStatus.SUPPORTED else reasons,
    )
    if status is ScientificStatus.NOT_SUPPORTED:
        stage = replace(
            stage, disposition=LinkedCampaignDisposition.SCIENTIFIC_NEGATIVE
        )
    return status, reasons, stage


class FiniteResponseLawCalibrationMethodTask:
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: FiniteResponseLawCalibrationMethodConfig,
        plane: CandidatePayloadPlane,
    ):
        self.manifest, self.config, self.plane = manifest, config, plane

    def _inputs(self, context: TaskContext) -> dict[str, bytes]:
        config = self.config
        external = tuple(
            p for p in context.input_ports if p.kind is WorkerInputKind.EXTERNAL
        )
        expected = {a.artifact_id: a for a in config.inputs}
        config_ports = tuple(
            p
            for p in external
            if p.artifact_id == f"config-artifact.{config.config_id}"
        )
        if (
            len(external) != (1 if context.task_id in (FREEZE, ADJUDICATE) else 6)
            or len(config_ports) != 1
            or context.config.config_id != config.config_id
            or context.config.config_schema != config.SCHEMA
            or context.config.config_schema_sha256 != self.manifest.config_schema_sha256
            or context.config.content_sha256 != config.fingerprint()
            or context.config.artifact_id != config_ports[0].artifact_id
            or decode_port(config_ports[0], type(config)) != config
        ):
            raise ValueError(
                "Finite response-law method changes its exact issued configuration/input census"
            )
        if context.task_id == FREEZE:
            if context.dependency_receipts or len(context.input_ports) != 1:
                raise ValueError(
                    "Method input freeze may read only its issued configuration"
                )
            return {}
        if context.task_id == ADJUDICATE:
            return {}
        data = {}
        for port in external:
            if port is config_ports[0]:
                continue
            identity = expected.pop(port.artifact_id, None)
            if (
                identity is None
                or port.payload_schema != identity.payload_schema
                or port.size_bytes != identity.size_bytes
            ):
                raise ValueError("Finite response-law method input is absent, repeated or substituted")
            raw = read_port(port, MAXIMUM_INPUT_BYTES)
            if sha256(raw).hexdigest() != identity.sha256:
                raise ValueError("Finite response-law method input digest differs")
            data[identity.artifact_id] = raw
        if expected:
            raise ValueError("Finite response-law method lacks frozen/native inputs")
        # Manifest bytes are bound before any fresh observation. The transported
        # original nomination manifest records historical provenance, not current
        # loaded code/environment; current authoring/runtime pins stay separate.
        manifest = json.loads(data[config.development_manifest.artifact_id])
        if not isinstance(manifest, dict):
            raise TypeError("Frozen development manifest must be an object")
        from .nominated_package import NOMINATION_SOURCE_PINS, validate_nomination_transport_inputs

        if config.development_manifest.sha256 == NOMINATION_SOURCE_PINS[2][2]:
            identities = (config.development_report, config.coefficients, config.development_manifest)
            validate_nomination_transport_inputs(identities, tuple(data[item.artifact_id] for item in identities))
        return data

    def execute(self, context: TaskContext) -> RunnerResult:
        try:
            if context.task_id not in (FREEZE, CALIBRATE, QUALIFY, ADJUDICATE):
                raise ValueError("Finite response-law method task is outside its frozen roster")
            data = self._inputs(context)
            config = self.config
            if context.task_id == FREEZE:
                identity = ObjectIdentity.from_record(config.config_id, config)
                return output_result(
                    context,
                    {identity.SCHEMA: identity.canonical_bytes()},
                    ("issued-input-identities-committed-before-protected-read",),
                )
            if context.task_id == ADJUDICATE:
                return self._adjudicate(context)
            native_receipt = decode_canonical_bytes(
                data[config.native_receipt.artifact_id],
                CanonicalTaskReceipt,
                maximum_bytes=MAXIMUM_INPUT_BYTES,
            )
            if context.task_id == CALIBRATE:
                if (
                    len(context.dependency_receipts) != 1
                    or len(context.input_ports) != 7
                    or context.dependency_receipts[0].task_id != FREEZE
                ):
                    raise ValueError("Calibration requires its frozen input commitment")
                dependency = context.dependency_receipts[0]
                group = _group(
                    context,
                    FREEZE,
                    dependency.output_materialization_ids,
                    (ObjectIdentity.SCHEMA,),
                )
                if decode_port(
                    group[ObjectIdentity.SCHEMA], ObjectIdentity
                ) != ObjectIdentity.from_record(config.config_id, config):
                    raise ValueError(
                        "Calibration changes its frozen method input identities"
                    )
                prediction_ports = tuple(
                    p
                    for p in context.output_ports
                    if p.payload_schema == PREDICTION_SCHEMA
                )
                if (
                    len(prediction_ports) != 1
                    or prediction_ports[0].logical_artifact_id is None
                ):
                    raise ValueError(
                        "Calibration lacks its committed prediction output"
                    )
                operands = calibration_operands(
                    native_bytes=data[config.native_evaluation.artifact_id],
                    native_artifact=config.native_evaluation,
                    native_receipt=native_receipt,
                    expected_native_receipt=config.expected_native_receipt,
                    source=config.native_source,
                    development_report_bytes=data[
                        config.development_report.artifact_id
                    ],
                    coefficient_bytes=data[config.coefficients.artifact_id],
                    development_report=config.development_report,
                    coefficients=config.coefficients,
                    development_manifest=config.development_manifest,
                    prediction_artifact_id=prediction_ports[0].logical_artifact_id,
                )
                raw = (
                    json.dumps(
                        operands.readout,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode()
                readout_artifact = _artifact(
                    context,
                    READOUT_SCHEMA,
                    raw,
                    "calibration-diagnostics",
                    "application/json",
                )
                calibration_type, _ = _report_types(config)
                report = calibration_type(
                    f"{CALIBRATE}.result",
                    ObjectIdentity.from_record(config.config_id, config),
                    operands.prediction_artifact,
                    readout_artifact,
                    operands.boundaries,
                    tuple(
                        (
                            b,
                            operands.readout["usability"][b][
                                "joint_decision_opportunity"
                            ],
                        )
                        for b in BOUNDARIES
                    ),
                )
                stage = calibration_stage(report)
                return output_result(
                    context,
                    {
                        report.SCHEMA: report.canonical_bytes(),
                        PREDICTION_SCHEMA: operands.prediction_bytes,
                        READOUT_SCHEMA: raw,
                        stage.SCHEMA: stage.canonical_bytes(),
                    },
                    (
                        "frozen-model-no-refit",
                        "all-32-roots-retained",
                        "rank-30-including-infinity",
                    ),
                )
            return self._qualify(context, data, native_receipt)
        finally:
            for port in context.input_ports:
                port.close()

    def _qualify(
        self,
        context: TaskContext,
        data: dict[str, bytes],
        native_receipt: CanonicalTaskReceipt,
    ) -> RunnerResult:
        from empirical_lawhood.adapters.composition.finite_response_law.design import qualification_system

        config = self.config
        if len(context.dependency_receipts) != 1 or len(context.input_ports) != 10:
            raise ValueError(
                "Qualification requires exactly the committed calibration dependency"
            )
        dependency = context.dependency_receipts[0]
        if dependency.task_id != CALIBRATE:
            raise ValueError("Qualification changes its calibration predecessor")
        calibration_type, _ = _report_types(config)
        group = _group(
            context,
            dependency.task_id,
            dependency.output_materialization_ids,
            (
                calibration_type.SCHEMA,
                PREDICTION_SCHEMA,
                READOUT_SCHEMA,
                LinkedCampaignStageEnvelope.SCHEMA,
            ),
        )
        report = decode_port(group[calibration_type.SCHEMA], calibration_type)
        stage = decode_port(
            group[LinkedCampaignStageEnvelope.SCHEMA], LinkedCampaignStageEnvelope
        )
        if stage != calibration_stage(
            report
        ) or report.config != ObjectIdentity.from_record(config.config_id, config):
            raise ValueError(
                "Qualification received a detached calibration report/envelope"
            )
        for identity in (report.prediction_artifact, report.readout_artifact):
            if group[identity.payload_schema].artifact_id != identity.artifact_id:
                raise ValueError("Qualification changes its committed output identity")
        evaluation = authenticated_evaluation(
            data[config.native_evaluation.artifact_id],
            config.native_evaluation,
            native_receipt,
            expected_receipt=config.expected_native_receipt,
            source=config.native_source,
        )
        operands = restore_calibration_operands(
            evaluation=evaluation,
            prediction_bytes=read_port(group[PREDICTION_SCHEMA], MAXIMUM_INPUT_BYTES),
            prediction_artifact=report.prediction_artifact,
            readout_bytes=read_port(group[READOUT_SCHEMA], MAXIMUM_INPUT_BYTES),
            readout_artifact=report.readout_artifact,
            boundaries=report.boundaries,
        )
        if report.joint_opportunities != tuple(
            (b, operands.readout["usability"][b]["joint_decision_opportunity"])
            for b in BOUNDARIES
        ):
            raise ValueError("Qualification changes its committed usability readout")
        result = qualify_calibration(
            config=config,
            report=report,
            operands=operands,
            development_report_bytes=data[config.development_report.artifact_id],
            coefficient_bytes=data[config.coefficients.artifact_id],
            system=qualification_system(config.native_source),
            manifest=self.manifest,
            payload_plane=self.plane,
            dependency=dependency,
            calibration_materialization_id=group[
                calibration_type.SCHEMA
            ].materialization_id,
            prediction_materialization_id=group[PREDICTION_SCHEMA].materialization_id,
        )
        _, _, stage = qualification_disposition(result)
        return output_result(
            context,
            {
                result.SCHEMA: result.canonical_bytes(),
                stage.SCHEMA: stage.canonical_bytes(),
            },
            (
                "committed-calibration-receipt",
                "four-singleton-families",
                "sole-response-law-qualifier",
            ),
        )

    def _adjudicate(self, context: TaskContext) -> RunnerResult:
        if (
            len(context.dependency_receipts) != 1
            or len(context.input_ports) != 3
            or context.dependency_receipts[0].task_id != QUALIFY
        ):
            raise ValueError(
                "Adjudication requires exactly its committed qualification dependency"
            )
        dependency = context.dependency_receipts[0]
        _, qualification_type = _report_types(self.config)
        group = _group(
            context,
            QUALIFY,
            dependency.output_materialization_ids,
            (qualification_type.SCHEMA, LinkedCampaignStageEnvelope.SCHEMA),
        )
        result = decode_port(group[qualification_type.SCHEMA], qualification_type)
        status, reasons, stage = qualification_disposition(result)
        if (
            result.calibration.config
            != ObjectIdentity.from_record(self.config.config_id, self.config)
            or decode_port(
                group[LinkedCampaignStageEnvelope.SCHEMA],
                LinkedCampaignStageEnvelope,
            )
            != stage
        ):
            raise ValueError("Adjudication changes its committed qualification")
        authority = context.scientific_adjudication_context
        if authority is None:
            raise ValueError(
                "Finite response-law qualification lacks authenticated adjudication authority"
            )
        adjudication = ScientificAdjudicationRecord(
            f"adjudication.{context.run_id}.{context.task_id}",
            context.run_id,
            context.task_id,
            authority.execution_plan,
            context.input_materialization_ids,
            tuple(
                sorted(
                    p.logical_artifact_id
                    for p in context.output_ports
                    if p.logical_artifact_id is not None
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
            AdjudicationEvaluability.UNEVALUABLE
            if status is ScientificStatus.UNEVALUABLE
            else AdjudicationEvaluability.EVALUABLE,
            status,
            AdmissionStatus.NOT_EVALUATED,
            reasons,
            authority.fixture_scope_id,
            authority.plumbing_only,
        )
        return output_result(
            context,
            {
                adjudication.SCHEMA: adjudication.canonical_bytes(),
            },
            (
                "committed-calibration-receipt",
                "four-singleton-families",
                "sole-response-law-qualifier",
            ),
        )


class FiniteResponseLawCalibrationMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        manifest: CapabilityManifest,
        config: FiniteResponseLawCalibrationMethodConfig,
        resolver: ContentAddressedInputResolver,
        plane: CandidatePayloadPlane,
    ):
        if (
            registry.resolve(manifest.capability_key, manifest.capability_version)
            != manifest
            or manifest.config_schema != config.SCHEMA
        ):
            raise ValueError(
                "Finite response-law calibration factory changes its registry/configuration"
            )
        self.registry, self.manifest, self.config, self.resolver, self.plane = (
            registry,
            manifest,
            config,
            resolver,
            plane,
        )
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("Finite response-law method runner registry/records differ")
        return (
            cast(
                TaskRunner,
                FiniteResponseLawCalibrationMethodTask(self.manifest, self.config, self.plane),
            ),
        )

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("Finite response-law method execution registry/records differ")
        values = list(
            config_payloads(
                plan,
                self.manifest,
                self.config,
                self.config.config_id,
                {FREEZE, CALIBRATE, QUALIFY, ADJUDICATE},
            )
        )
        expected_ids = {
            f"config-artifact.{self.config.config_id}",
            *(a.artifact_id for a in self.config.inputs),
        }
        for task in plan.tasks:
            if task.capability.capability_key != self.manifest.capability_key:
                continue
            if {i.logical_artifact_id for i in task.external_inputs} != (
                {f"config-artifact.{self.config.config_id}"}
                if task.task_id in (FREEZE, ADJUDICATE)
                else expected_ids
            ):
                raise ValueError(
                    "Finite response-law method execution changes its full external input roster"
                )
            for spec in task.external_inputs:
                artifact = next(
                    (
                        a
                        for a in self.config.inputs
                        if a.artifact_id == spec.logical_artifact_id
                    ),
                    None,
                )
                if artifact is not None and (
                    spec.expected_content_sha256 != artifact.sha256
                    or spec.expected_payload_schema != artifact.payload_schema
                ):
                    raise ValueError(
                        "Finite response-law execution input detaches its issued artifact identity"
                    )
        for artifact in self.config.inputs:
            is_native = artifact in (
                self.config.native_evaluation,
                self.config.native_receipt,
            )
            access = (
                OutcomeAccess.EVALUATOR_REVEAL
                if is_native
                else OutcomeAccess.OUTCOME_BLIND
            )
            # Fixed model parameters are input specifications for the fresh
            # experiment. Their original fitted-law origins remain explicit in the frozen config;
            # Original fitted-law observations/results themselves are never relabelled or promoted.
            requirement = ContentAddressedInputRequirement(
                f"input.{artifact.artifact_id}",
                ContentAddressedInputKind.SOURCE_MATERIALIZATION,
                artifact.sha256,
                artifact.payload_schema,
                artifact.media_type,
                MAXIMUM_INPUT_BYTES,
                artifact.size_bytes,
                access,
                VisibilityCeiling.PROSPECTIVE,
            )
            profile = (
                ArtifactProfile.CANONICAL_JSON
                if artifact.payload_schema == COEFFICIENT_SCHEMA
                else ArtifactProfile.CANONICAL_JSON
                if is_native
                else ArtifactProfile.TEXT_PARAMETERS
            )
            parent = ArtifactLineageParent(
                ObjectIdentity.from_record(self.config.config_id, self.config),
                VisibilityCeiling.PROSPECTIVE,
                OutcomeAccess.OUTCOME_BLIND,
            )
            values.append(
                ExternalInputPayload(
                    logical_artifact_id=artifact.artifact_id,
                    payload_schema=artifact.payload_schema,
                    profile=profile,
                    media_type=artifact.media_type,
                    source=_ResolvedInputSource(self.resolver, requirement),
                    size_bytes=artifact.size_bytes,
                    source_sha256=artifact.sha256,
                    maximum_bytes=artifact.size_bytes,
                    maximum_chunk_bytes=min(
                        artifact.size_bytes, MAX_EXTERNAL_INPUT_CHUNK_BYTES
                    ),
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=access,
                    parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
                    lineage_parents=(parent,),
                    logical_content_sha256=artifact.sha256,
                )
            )
        return tuple(sorted(values, key=lambda v: v.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("Finite response-law semantic registry differs")
        calibration_type, qualification_type = _report_types(self.config)
        values = semantic_contracts(
            self.manifest,
            (
                ObjectIdentity,
                calibration_type,
                qualification_type,
                LinkedCampaignStageEnvelope,
                ScientificAdjudicationRecord,
            ),
            None,
        )
        return tuple(
            sorted(
                (
                    *values,
                    CapabilityOutputSemanticContract.from_manifest(
                        self.manifest,
                        payload_schema=PREDICTION_SCHEMA,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=("npz_base64", "npz_bytes", "npz_sha256"),
                        record_version="1.0.0",
                    ),
                    CapabilityOutputSemanticContract.from_manifest(
                        self.manifest,
                        payload_schema=READOUT_SCHEMA,
                        profile=ArtifactProfile.TEXT_PARAMETERS,
                    ),
                ),
                key=lambda v: v.key,
            )
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        if registry != self.registry:
            raise ValueError("Finite response-law adjudication registry differs")
        return ScientificAdjudicationOutputContract(
            self.manifest.capability_key,
            self.manifest.capability_version,
            f"{ADJUDICATE}.report",
            ScientificAdjudicationRecord.SCHEMA,
            fixture_scope_id=None,
            plumbing_only=False,
        )

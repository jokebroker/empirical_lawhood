'Receipt-first material control control execution and conditional science freeze.'

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
import tarfile
from typing import Any, Callable, TypeVar, cast

import numpy as np

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
    encode_scientific_adjudication,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    StreamingOutputEmitter,
    TaskContext,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)

from .gauge_covariant_response_contracts import GaugeCovariantResponseConformanceResult, GaugeCovariantResponseResponseObservation
from .synthetic_gauge_covariant_response import run_gauge_covariant_response_conformance
from .material_control_contracts import MATERIAL_CONTROL_ADJUDICATION_SCHEMA, MATERIAL_CONTROL_CAPABILITY_VERSION, MATERIAL_CONTROL_CLOSEOUT_CAPABILITY_KEY, MATERIAL_CONTROL_CONFIG_SCHEMA, MATERIAL_CONTROL_CONFIG_VERSION, MATERIAL_CONTROL_METHOD_CAPABILITY_KEY, MATERIAL_CONTROL_PANEL_CAPABILITY_KEY, MATERIAL_CONTROL_SOURCE_CAPABILITY_KEY, MATERIAL_CONTROL_WORKFLOW_CAPABILITY_KEY, MATERIAL_CONTROL_SEARCH_WORLD_CAPABILITY_KEY, MaterialControlCloseout, MaterialControlConfig, MaterialControlControlBundleQualification, MaterialControlControlObservation, MaterialControlControlPanel, MaterialControlStageResult, MaterialControlTutorialReproduction, MaterialControlTruthWorldConformance, MaterialControlTruthWorldResult, BridgeDisposition, MultibandStrongCouplingMaterialCompatibility, MultibandStrongCouplingMaterialViewResult, MultibandStrongCouplingBridgeQualification, decode_material_control_config_bytes
from .material_control_design import build_material_control_closeout, build_control_observation, build_control_panel, build_tutorial_reproduction, profile_source_identity
from .material_control_material_operands import evaluate_material_view, reduce_material_operands, unresolved_material_operands
from .multiband_strong_coupling_response import qualify_multiband_bridge
from .material_control_raw import RAW_ARCHIVE_MEDIA_TYPE, RAW_ARCHIVE_SCHEMA
from .material_control_registration import material_control_metadata_task_budget, material_control_task_budget
from .material_control_solver import MATERIAL_CONTROL_WORKFLOW_PROFILES, MaterialControlWorkflowProfile, MaterialControlWorkflowRequest, FileTaskOutputSource, FixedMaterialControlWorkflowExecutor, WorkflowKind, WorkflowSource, cleanup_material_control_workflow_scratch
from .material_control_source import expected_material_control_control_bundle, inspect_material_control_control_bundle
from .constructive_search_conformance import run_constructive_search_conformance


CONFIG_ARTIFACT_ID = 'config-artifact.ambient-pressure-superconductor-material-control'
SOURCE_CONTRACT_ARTIFACT_ID = 'source-contract.ambient-pressure-superconductor-material-control-control-bundle'
SOURCE_STEP = 'material-control-source-qualification'
METHOD_STEP = 'material-control-gauge-covariant-response-method-qualification'
CONSTRUCTIVE_SEARCH_CONFORMANCE_STEP = 'material-control-constructive-search-search-world-conformance'
CLOSEOUT_STEP = 'material-control-science-freeze-closeout'

_WORKFLOW_SEQUENCE = (
    'workflow.material-control-tutorial04-pb',
    'workflow.material-control-tutorial04-mgb2',
    'workflow.material-control-pbe-efficiency-base-calibration-pb-fcc',
    'workflow.material-control-pbe-precision-refined-calibration-pb-fcc',
    'workflow.material-control-pbe-efficiency-base-calibration-mgb2-alb2',
    'workflow.material-control-pbe-precision-refined-calibration-mgb2-alb2',
    'workflow.material-control-pbe-efficiency-base-calibration-cu-fcc',
    'workflow.material-control-pbe-precision-refined-calibration-cu-fcc',
    'workflow.material-control-pbe-efficiency-base-calibration-c-diamond',
    'workflow.material-control-pbe-precision-refined-calibration-c-diamond',
    'workflow.material-control-pbe-efficiency-base-calibration-pb-sc-compressed',
    'workflow.material-control-pbe-precision-refined-calibration-pb-sc-compressed',
)
_PROFILE_BY_ID = {value.profile_id: value for value in MATERIAL_CONTROL_WORKFLOW_PROFILES}
if set(_WORKFLOW_SEQUENCE) != set(_PROFILE_BY_ID):
    raise ValueError(
        'material control workflow execution sequence differs from its closed profile set'
    )


def _workflow_step_id(index: int, profile_id: str) -> str:
    return f"material-control-workflow-{index:02d}-{profile_id.removeprefix('workflow.material-control-')}"


WORKFLOW_TASKS = {
    _workflow_step_id(index, profile_id): _PROFILE_BY_ID[profile_id]
    for index, profile_id in enumerate(_WORKFLOW_SEQUENCE, start=1)
}
PROFILE_TASKS = {value.profile_id: key for key, value in WORKFLOW_TASKS.items()}
PANEL_TASKS = {
    'material-control-panel-c-diamond': 'structure.calibration-c-diamond',
    'material-control-panel-cu-fcc': 'structure.calibration-cu-fcc',
    'material-control-panel-mgb2-alb2': 'structure.calibration-mgb2-alb2',
    'material-control-panel-pb-fcc': 'structure.calibration-pb-fcc',
    'material-control-panel-pb-sc-compressed': 'structure.calibration-pb-sc-compressed',
}


def material_control_config_ref(config: MaterialControlConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id='config.ambient-pressure-superconductor-material-control-control-science-freeze',
        config_schema=MATERIAL_CONTROL_CONFIG_SCHEMA,
        config_schema_sha256=sha256(MATERIAL_CONTROL_CONFIG_SCHEMA.encode()).hexdigest(),
        content_sha256=config.payload_sha256,
        artifact_id=CONFIG_ARTIFACT_ID,
    )


def _json(output_id: str, schema: str) -> OutputTemplate:
    return OutputTemplate(
        output_id=output_id,
        payload_schema=schema,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/json",
        filename_suffix=".json",
    )


def _raw() -> OutputTemplate:
    return OutputTemplate(
        output_id="raw-workflow",
        payload_schema=RAW_ARCHIVE_SCHEMA,
        profile=ArtifactProfile.AUDITED_HDF5,
        media_type=RAW_ARCHIVE_MEDIA_TYPE,
        filename_suffix=".h5",
    )


def _step(
    *,
    registry: CapabilityRegistry,
    config: MaterialControlConfig,
    step_id: str,
    stage: ScientificStage,
    capability_key: str,
    dependencies: tuple[str, ...],
    outputs: tuple[OutputTemplate, ...],
    access: OutcomeAccess,
    visibility: VisibilityCeiling,
    barrier: BarrierKind,
    obligations: tuple[str, ...],
    resource_locks: tuple[str, ...] = (),
) -> ProtocolStepTemplate:
    manifest = registry.resolve(capability_key, MATERIAL_CONTROL_CAPABILITY_VERSION)
    return ProtocolStepTemplate(
        step_id=step_id,
        stage=stage,
        capability_key=capability_key,
        capability_version=MATERIAL_CONTROL_CAPABILITY_VERSION,
        config=material_control_config_ref(config),
        dependency_step_ids=tuple(sorted(dependencies)),
        outputs=tuple(sorted(outputs, key=lambda value: value.output_id)),
        required_permissions=manifest.permissions,
        requested_outcome_access=access,
        visibility_ceiling=visibility,
        resource_budget=(
            material_control_task_budget(config)
            if capability_key == MATERIAL_CONTROL_WORKFLOW_CAPABILITY_KEY
            else material_control_metadata_task_budget(config)
        ),
        resource_lock_ids=tuple(sorted(resource_locks)),
        barrier=barrier,
        maximum_attempts=1,
        obligation_ids=tuple(sorted(obligations)),
    )


def material_control_protocol(
    *, registry: CapabilityRegistry, config: MaterialControlConfig
) -> ProtocolTemplate:
    steps: list[ProtocolStepTemplate] = [
        _step(
            registry=registry,
            config=config,
            step_id=SOURCE_STEP,
            stage=ScientificStage.QUALIFY,
            capability_key=MATERIAL_CONTROL_SOURCE_CAPABILITY_KEY,
            dependencies=(),
            outputs=(_json("control-bundle", MaterialControlControlBundleQualification.SCHEMA),),
            access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.FREEZE,
            obligations=('material-control-exact-public-control-source-and-environment',),
            resource_locks=('ambient-pressure-superconductor-material-control-source-scan',),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=METHOD_STEP,
            stage=ScientificStage.FALSIFY,
            capability_key=MATERIAL_CONTROL_METHOD_CAPABILITY_KEY,
            dependencies=(),
            outputs=(
                _json('gauge-covariant-response-gauge-covariant-response-conformance', GaugeCovariantResponseConformanceResult.SCHEMA),
                _json('multiband-gauge-covariant-response-bridge', MultibandStrongCouplingBridgeQualification.SCHEMA),
                *tuple(
                    _json(f'gauge-covariant-response-observation-{index:02d}', GaugeCovariantResponseResponseObservation.SCHEMA)
                    for index in range(1, 10)
                ),
            ),
            access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.FREEZE,
            obligations=('material-control-gauge-covariant-response-gauge-covariant-response-recurrence-and-multiband-bridge-conformance',),
            resource_locks=('ambient-pressure-superconductor-material-control-gauge-covariant-response-cpu',),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=CONSTRUCTIVE_SEARCH_CONFORMANCE_STEP,
            stage=ScientificStage.FALSIFY,
            capability_key=MATERIAL_CONTROL_SEARCH_WORLD_CAPABILITY_KEY,
            dependencies=(),
            outputs=(
                _json('constructive-search-conformance', MaterialControlTruthWorldConformance.SCHEMA),
                *tuple(
                    _json(f'constructive-search-world-{index:02d}', MaterialControlTruthWorldResult.SCHEMA)
                    for index in range(1, 7)
                ),
            ),
            access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
            obligations=('material-control-six-world-three-policy-search-conformance',),
        ),
    ]
    for task_id, profile in WORKFLOW_TASKS.items():
        outputs: tuple[OutputTemplate, ...]
        if profile.source is WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE:
            outputs = (
                _raw(),
                _json("tutorial-reproduction", MaterialControlTutorialReproduction.SCHEMA),
            )
        elif profile.structure_id in {
            'structure.calibration-pb-fcc',
            'structure.calibration-mgb2-alb2',
        }:
            outputs = (
                _json("control-observation", MaterialControlControlObservation.SCHEMA),
                _json('material-gauge-covariant-response-compatibility', MultibandStrongCouplingMaterialCompatibility.SCHEMA),
                _json('material-gauge-covariant-response-result', MultibandStrongCouplingMaterialViewResult.SCHEMA),
                _raw(),
            )
        else:
            outputs = (
                _json("control-observation", MaterialControlControlObservation.SCHEMA),
                _raw(),
            )
        steps.append(
            _step(
                registry=registry,
                config=config,
                step_id=task_id,
                stage=ScientificStage.ACQUIRE,
                capability_key=MATERIAL_CONTROL_WORKFLOW_CAPABILITY_KEY,
                dependencies=(SOURCE_STEP, METHOD_STEP),
                outputs=outputs,
                access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
                barrier=BarrierKind.NONE,
                obligations=(
                    f"material-control-control-workflow-{profile.profile_id.removeprefix('workflow.')}",
                ),
                resource_locks=('ambient-pressure-superconductor-material-control-qe-epw',),
            )
        )

    for task_id, structure_id in PANEL_TASKS.items():
        profile_tasks = tuple(
            sorted(
                PROFILE_TASKS[value.profile_id]
                for value in MATERIAL_CONTROL_WORKFLOW_PROFILES
                if value.source is WorkflowSource.SSSP_CONTROL
                and value.structure_id == structure_id
            )
        )
        if len(profile_tasks) != 2:
            raise ValueError('material control panel lacks its two frozen science views')
        steps.append(
            _step(
                registry=registry,
                config=config,
                step_id=task_id,
                stage=ScientificStage.TRANSFORM,
                capability_key=MATERIAL_CONTROL_PANEL_CAPABILITY_KEY,
                dependencies=profile_tasks,
                outputs=(_json("control-panel", MaterialControlControlPanel.SCHEMA),),
                access=OutcomeAccess.DEVELOPMENT_VISIBLE,
                visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
                barrier=BarrierKind.NONE,
                obligations=(
                    f"material-control-one-unit-two-view-panel-{structure_id.removeprefix('structure.')}",
                ),
            )
        )

    positive_tasks = tuple(
        PROFILE_TASKS[value.profile_id]
        for value in MATERIAL_CONTROL_WORKFLOW_PROFILES
        if value.source is WorkflowSource.SSSP_CONTROL
        and value.structure_id in {'structure.calibration-pb-fcc', 'structure.calibration-mgb2-alb2'}
    )
    tutorial_tasks = tuple(
        PROFILE_TASKS[value.profile_id]
        for value in MATERIAL_CONTROL_WORKFLOW_PROFILES
        if value.source is WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE
    )
    steps.append(
        _step(
            registry=registry,
            config=config,
            step_id=CLOSEOUT_STEP,
            stage=ScientificStage.EVALUATE,
            capability_key=MATERIAL_CONTROL_CLOSEOUT_CAPABILITY_KEY,
            dependencies=tuple(
                sorted(
                    (
                        SOURCE_STEP,
                        METHOD_STEP,
                        CONSTRUCTIVE_SEARCH_CONFORMANCE_STEP,
                        *tutorial_tasks,
                        *positive_tasks,
                        *PANEL_TASKS,
                    )
                )
            ),
            outputs=(
                _json('material-control-closeout', MaterialControlCloseout.SCHEMA),
                _json('material-control-stage-result', MaterialControlStageResult.SCHEMA),
                _json("scientific-adjudication", MATERIAL_CONTROL_ADJUDICATION_SCHEMA),
            ),
            access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.REVEAL,
            obligations=(
                'material-control-calibration-and-science-freeze-before-development-atlas-or-typed-stop',
                'material-control-low-temperature-controls-not-300k-material-evidence',
                'material-control-target-contact-count-zero',
            ),
        )
    )
    return ProtocolTemplate(
        template_id='protocol.ambient-pressure-superconductor-material-control-control-science-freeze',
        template_version=MATERIAL_CONTROL_CAPABILITY_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=True,
        requests_controller=False,
        nonactuating=True,
    )


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _records(
    records: tuple[CanonicalRecord, ...], record_type: type[_RecordT]
) -> tuple[_RecordT, ...]:
    return tuple(value for value in records if isinstance(value, record_type))


def _one(records: tuple[CanonicalRecord, ...], record_type: type[_RecordT]) -> _RecordT:
    values = _records(records, record_type)
    if len(values) != 1:
        raise ValueError(f'material control task requires exactly one {record_type.__name__}')
    return values[0]


_INPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    MaterialControlControlBundleQualification.SCHEMA: MaterialControlControlBundleQualification,
    MaterialControlControlObservation.SCHEMA: MaterialControlControlObservation,
    MaterialControlControlPanel.SCHEMA: MaterialControlControlPanel,
    MaterialControlTutorialReproduction.SCHEMA: MaterialControlTutorialReproduction,
    MaterialControlTruthWorldConformance.SCHEMA: MaterialControlTruthWorldConformance,
    MaterialControlTruthWorldResult.SCHEMA: MaterialControlTruthWorldResult,
    MultibandStrongCouplingMaterialCompatibility.SCHEMA: MultibandStrongCouplingMaterialCompatibility,
    MultibandStrongCouplingMaterialViewResult.SCHEMA: MultibandStrongCouplingMaterialViewResult,
    MultibandStrongCouplingBridgeQualification.SCHEMA: MultibandStrongCouplingBridgeQualification,
    GaugeCovariantResponseConformanceResult.SCHEMA: GaugeCovariantResponseConformanceResult,
    GaugeCovariantResponseResponseObservation.SCHEMA: GaugeCovariantResponseResponseObservation,
}


def _emit_bytes(
    emitter: StreamingOutputEmitter, output_id: str, payload: bytes
) -> None:
    chunk_size = 1024 * 1024
    for offset in range(0, len(payload), chunk_size):
        emitter.write(output_id, payload[offset : offset + chunk_size])


class MaterialControlRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: MaterialControlConfig,
        config_payload: bytes,
        source_contract: MaterialControlControlBundleQualification,
        workflow_executor: FixedMaterialControlWorkflowExecutor,
        source_identity_provider: Callable[[MaterialControlWorkflowProfile], tuple[str, str, str]],
    ) -> None:
        if (
            decode_material_control_config_bytes(
                config_payload, expected_parents=config.predecessors
            )
            != config
        ):
            raise ValueError('material control runner config differs from registered bytes')
        self.manifest = manifest
        self.config = config
        self.config_payload = config_payload
        self.source_contract = source_contract
        self.workflow_executor = workflow_executor
        self.source_identity_provider = source_identity_provider

    def _read(self, context: TaskContext) -> tuple[CanonicalRecord, ...]:
        records: list[CanonicalRecord] = []
        observed_config = False
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError('material control task input size differs')
            if port.payload_schema == MATERIAL_CONTROL_CONFIG_SCHEMA:
                if (
                    payload != self.config_payload
                    or decode_material_control_config_bytes(
                        payload, expected_parents=self.config.predecessors
                    )
                    != self.config
                ):
                    raise ValueError('material control task config materialization differs')
                observed_config = True
                continue
            record_type = _INPUT_TYPES.get(port.payload_schema)
            if record_type is None:
                raise ValueError('material control task received an unknown input schema')
            records.append(
                decode_canonical_bytes(
                    payload, record_type, maximum_bytes=port.size_bytes
                )
            )
        if (
            not observed_config
            or context.config.content_sha256 != self.config.payload_sha256
        ):
            raise ValueError('material control task lacks its exact config')
        return tuple(records)

    @staticmethod
    def _adjudication(
        context: TaskContext, closeout: MaterialControlCloseout
    ) -> ScientificAdjudicationRecord:
        adjudication_context = context.scientific_adjudication_context
        if adjudication_context is None:
            raise ValueError('material control closeout lacks scientific adjudication context')
        output_ids = tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
            )
        )
        if len(output_ids) != len(context.output_ports):
            raise ValueError('material control closeout output lacks logical identity')
        stage = closeout.stage_result
        evaluability = (
            AdjudicationEvaluability.EVALUABLE
            if stage.evaluable
            else AdjudicationEvaluability.UNEVALUABLE
        )
        scientific_status = (
            ScientificStatus.SUPPORTED
            if stage.disposition.value.startswith("MATERIAL_CONTROL_RECOVERY_PASS")
            else ScientificStatus.NOT_SUPPORTED
        )
        admission_status = AdmissionStatus.NOT_EVALUATED
        if evaluability is AdjudicationEvaluability.UNEVALUABLE:
            scientific_status = ScientificStatus.UNEVALUABLE
            admission_status = AdmissionStatus.UNEVALUABLE
        return ScientificAdjudicationRecord(
            adjudication_id=f'adjudication.{context.run_id}.{context.task_id}',
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=adjudication_context.execution_plan,
            input_materialization_ids=tuple(sorted(context.input_materialization_ids)),
            output_logical_artifact_ids=output_ids,
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=adjudication_context.evidence_world_id,
            evidence_world_kind=adjudication_context.evidence_world_kind,
            relation=adjudication_context.relation,
            independent_unit_id=adjudication_context.independent_unit_id,
            information_cutoffs=adjudication_context.information_cutoffs,
            visibility_ceiling=adjudication_context.visibility_ceiling,
            outcome_access=adjudication_context.outcome_access,
            evaluability=evaluability,
            scientific_status=scientific_status,
            admission_status=admission_status,
            reason_codes=closeout.reason_codes,
            fixture_scope_id=None,
            plumbing_only=False,
        )

    @staticmethod
    def _output_port_id(context: TaskContext, output_id: str, schema: str) -> str:
        values = tuple(
            port.output_id
            for port in context.output_ports
            if port.payload_schema == schema
            and port.output_id.removeprefix(f'{context.task_id}.') == output_id
        )
        if len(values) != 1:
            raise ValueError(f'material control task lacks one compiled output for {output_id}')
        return values[0]

    @staticmethod
    def _emit_record(
        context: TaskContext,
        emitter: StreamingOutputEmitter,
        output_id: str,
        record: CanonicalRecord,
    ) -> None:
        compiled_output_id = MaterialControlRunner._output_port_id(
            context, output_id, record.SCHEMA
        )
        _emit_bytes(emitter, compiled_output_id, record.canonical_bytes())

    def _execute_workflow(
        self,
        context: TaskContext,
        emitter: StreamingOutputEmitter,
        records: tuple[CanonicalRecord, ...],
    ) -> tuple[ReceiptCheck, ...]:
        source_contract = _one(records, MaterialControlControlBundleQualification)
        bridge = _one(records, MultibandStrongCouplingBridgeQualification)
        if (
            source_contract != self.source_contract
            or source_contract.fingerprint() != self.config.control_bundle_sha256
        ):
            raise ValueError('material control workflow source qualification differs')
        if (
            bridge.disposition is not BridgeDisposition.CONDITIONAL_METHOD_PASS
            or bridge.fingerprint() != self.config.bridge_qualification_sha256
        ):
            raise ValueError('material control workflow gauge covariant response method qualification differs')
        profile = WORKFLOW_TASKS[context.task_id]
        postprocessing_reserve = (
            12 * 3600
            if profile.kind is WorkflowKind.POSITIVE
            and profile.source is WorkflowSource.SSSP_CONTROL
            else 3600
        )
        request_wall = max(
            1, context.resource_budget.wall_time_seconds - postprocessing_reserve
        )
        operands = self.workflow_executor.execute(
            MaterialControlWorkflowRequest(
                request_id=f'request.{context.run_id}.{context.task_id}.{context.attempt_id}',
                run_id=context.run_id,
                task_id=context.task_id,
                profile_id=profile.profile_id,
                wall_time_seconds=request_wall,
                output_bytes=context.resource_budget.output_bytes,
            )
        )
        source_identity = self.source_identity_provider(profile)
        if profile.source is WorkflowSource.EPW_SUPERCONDUCTING_REFERENCE:
            reproduction = build_tutorial_reproduction(
                profile=profile,
                operands=operands,
                source_identity=source_identity,
            )
            self._emit_record(context, emitter, "tutorial-reproduction", reproduction)
        else:
            material_result: MultibandStrongCouplingMaterialViewResult | None = None
            if profile.structure_id in {
                'structure.calibration-pb-fcc',
                'structure.calibration-mgb2-alb2',
            }:
                try:
                    reduction = reduce_material_operands(
                        raw_archive_path=operands.raw_archive_path,
                        profile=profile,
                        fermi_energy_eV=operands.fermi_energy_eV,
                        electron_phonon_lambda=operands.electron_phonon_lambda,
                    )
                    material_result = evaluate_material_view(
                        reduction=reduction, profile=profile
                    )
                except (
                    ArithmeticError,
                    OSError,
                    ValueError,
                    tarfile.TarError,
                    np.linalg.LinAlgError,
                ):
                    reduction = unresolved_material_operands(
                        profile=profile,
                        raw_archive_sha256=operands.raw_archive_sha256,
                        reason_code='reason.material-gauge-covariant-response-operands-could-not-be-reduced',
                    )
                    material_result = evaluate_material_view(
                        reduction=reduction, profile=profile
                    )
                self._emit_record(
                    context,
                    emitter,
                    'material-gauge-covariant-response-compatibility',
                    reduction.compatibility,
                )
                self._emit_record(
                    context, emitter, 'material-gauge-covariant-response-result', material_result
                )
            observation = build_control_observation(
                profile=profile,
                operands=operands,
                material_gauge_covariant_response=material_result,
                source_identity=source_identity,
            )
            self._emit_record(context, emitter, "control-observation", observation)
        source = FileTaskOutputSource(operands.raw_hdf5_path)
        try:
            raw_output_id = self._output_port_id(
                context,
                "raw-workflow",
                RAW_ARCHIVE_SCHEMA,
            )
            for chunk in source.chunks(1024 * 1024):
                emitter.write(raw_output_id, chunk)
        finally:
            source.close()
        cleanup_material_control_workflow_scratch(operands)
        return (
            ReceiptCheck(
                check_id='material-control-workflow-no-target-contact',
                passed=True,
                reason_codes=(),
            ),
            ReceiptCheck(
                check_id='material-control-workflow-typed-output-complete',
                passed=True,
                reason_codes=(),
            ),
        )

    def execute_streaming(
        self, context: TaskContext, emitter: StreamingOutputEmitter
    ) -> tuple[ReceiptCheck, ...]:
        records = self._read(context)
        if context.task_id == SOURCE_STEP:
            contract = _one(records, MaterialControlControlBundleQualification)
            if contract != self.source_contract:
                raise ValueError('material control source contract differs')
            observed = inspect_material_control_control_bundle(
                base_source_sha256=self.config.predecessors.development_source_sha256
            )
            if (
                observed != contract
                or observed.fingerprint() != self.config.control_bundle_sha256
            ):
                raise ValueError('material control physical control bundle differs')
            self._emit_record(context, emitter, "control-bundle", observed)
            checks = ('material-control-control-bundle-qualified',)
        elif context.task_id == METHOD_STEP:
            if records:
                raise ValueError('material control method task received scientific inputs')
            bridge = qualify_multiband_bridge(
                implementation_sha256=self.config.implementation_sha256
            )
            if bridge.fingerprint() != self.config.bridge_qualification_sha256:
                raise ValueError('material control multiband bridge differs from config')
            gauge_covariant_conformance, observations = run_gauge_covariant_response_conformance()
            if gauge_covariant_conformance.fingerprint() != self.config.gauge_covariant_conformance_sha256:
                raise ValueError('material control gauge covariant response gauge covariant response recurrence differs from frozen evidence')
            self._emit_record(context, emitter, 'multiband-gauge-covariant-response-bridge', bridge)
            self._emit_record(context, emitter, 'gauge-covariant-response-gauge-covariant-response-conformance', gauge_covariant_conformance)
            for index, observation in enumerate(observations, start=1):
                self._emit_record(
                    context,
                    emitter,
                    f'gauge-covariant-response-observation-{index:02d}',
                    observation,
                )
            checks = ('material-control-gauge-covariant-response-methods-qualified',)
        elif context.task_id in WORKFLOW_TASKS:
            return self._execute_workflow(context, emitter, records)
        elif context.task_id in PANEL_TASKS:
            control_observations = _records(records, MaterialControlControlObservation)
            panel = build_control_panel(
                structure_id=PANEL_TASKS[context.task_id],
                observations=control_observations,
            )
            self._emit_record(context, emitter, "control-panel", panel)
            checks = ('material-control-control-panel-adjudicated',)
        elif context.task_id == CONSTRUCTIVE_SEARCH_CONFORMANCE_STEP:
            if records:
                raise ValueError('material control constructive search task received scientific inputs')
            conformance, worlds = run_constructive_search_conformance()
            self._emit_record(context, emitter, 'constructive-search-conformance', conformance)
            for index, world in enumerate(worlds, start=1):
                self._emit_record(context, emitter, f'constructive-search-world-{index:02d}', world)
            checks = ('material-control-constructive-search-conformance-adjudicated',)
        elif context.task_id == CLOSEOUT_STEP:
            source = _one(records, MaterialControlControlBundleQualification)
            closeout = build_material_control_closeout(
                config=self.config,
                control_bundle_sha256=source.fingerprint(),
                bridge=_one(records, MultibandStrongCouplingBridgeQualification),
                gauge_covariant_conformance=_one(records, GaugeCovariantResponseConformanceResult),
                gauge_covariant_observations=_records(records, GaugeCovariantResponseResponseObservation),
                tutorials=_records(records, MaterialControlTutorialReproduction),
                material_results=_records(records, MultibandStrongCouplingMaterialViewResult),
                panels=_records(records, MaterialControlControlPanel),
                constructive_search=_one(records, MaterialControlTruthWorldConformance),
            )
            self._emit_record(context, emitter, 'material-control-closeout', closeout)
            self._emit_record(
                context, emitter, 'material-control-stage-result', closeout.stage_result
            )
            adjudication = encode_scientific_adjudication(
                self._adjudication(context, closeout),
                payload_schema=MATERIAL_CONTROL_ADJUDICATION_SCHEMA,
            )
            _emit_bytes(
                emitter,
                self._output_port_id(
                    context,
                    "scientific-adjudication",
                    MATERIAL_CONTROL_ADJUDICATION_SCHEMA,
                ),
                adjudication,
            )
            checks = ('material-control-terminal-intersection-adjudicated',)
        else:
            raise ValueError('unknown material control task identity')
        return tuple(
            ReceiptCheck(check_id=value, passed=True, reason_codes=())
            for value in checks
        )


_OUTPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    MaterialControlCloseout.SCHEMA: MaterialControlCloseout,
    MaterialControlControlBundleQualification.SCHEMA: MaterialControlControlBundleQualification,
    MaterialControlControlObservation.SCHEMA: MaterialControlControlObservation,
    MaterialControlControlPanel.SCHEMA: MaterialControlControlPanel,
    MaterialControlStageResult.SCHEMA: MaterialControlStageResult,
    MaterialControlTutorialReproduction.SCHEMA: MaterialControlTutorialReproduction,
    MaterialControlTruthWorldConformance.SCHEMA: MaterialControlTruthWorldConformance,
    MaterialControlTruthWorldResult.SCHEMA: MaterialControlTruthWorldResult,
    MultibandStrongCouplingMaterialCompatibility.SCHEMA: MultibandStrongCouplingMaterialCompatibility,
    MultibandStrongCouplingMaterialViewResult.SCHEMA: MultibandStrongCouplingMaterialViewResult,
    MultibandStrongCouplingBridgeQualification.SCHEMA: MultibandStrongCouplingBridgeQualification,
    GaugeCovariantResponseConformanceResult.SCHEMA: GaugeCovariantResponseConformanceResult,
    GaugeCovariantResponseResponseObservation.SCHEMA: GaugeCovariantResponseResponseObservation,
}


class MaterialControlRuntimeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: MaterialControlConfig,
        config_payload: bytes,
        source_contract: MaterialControlControlBundleQualification,
        workflow_executor: FixedMaterialControlWorkflowExecutor | None = None,
        source_identity_provider: (
            Callable[[MaterialControlWorkflowProfile], tuple[str, str, str]] | None
        ) = None,
    ) -> None:
        if any(
            value.implementation_sha256
            != registry.capabilities[0].implementation_sha256
            for value in registry.capabilities
        ):
            raise ValueError('material control registry mixes implementation identities')
        if (
            decode_material_control_config_bytes(
                config_payload, expected_parents=config.predecessors
            )
            != config
        ):
            raise ValueError('material control provider config differs from registered bytes')
        if source_contract != expected_material_control_control_bundle(
            base_source_sha256=config.predecessors.development_source_sha256
        ):
            raise ValueError('material control provider source-contract binding differs')
        self.registry = registry
        self.config = config
        self.config_payload = config_payload
        self.source_contract = source_contract
        self.workflow_executor = workflow_executor or FixedMaterialControlWorkflowExecutor()
        self.source_identity_provider = (
            source_identity_provider or profile_source_identity
        )
        self.registry_sha256 = registry.fingerprint()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        del source_records
        if registry != self.registry:
            raise ValueError('material control runner registry differs')
        return cast(
            tuple[TaskRunner, ...],
            tuple(
                MaterialControlRunner(
                    manifest,
                    self.config,
                    self.config_payload,
                    self.source_contract,
                    self.workflow_executor,
                    self.source_identity_provider,
                )
                for manifest in registry.capabilities
            ),
        )

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        del source_records
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError('material control execution plan registry differs')
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        if set(specs) != {CONFIG_ARTIFACT_ID, SOURCE_CONTRACT_ARTIFACT_ID}:
            raise ValueError('material control plan external input set differs')

        def payload(
            logical_id: str,
            *,
            schema: str,
            content: bytes,
            version: str,
        ) -> ExternalInputPayload:
            spec = specs[logical_id]
            parent = ArtifactLineageParent(
                identity=ObjectIdentity(
                    object_id=logical_id,
                    object_schema=schema,
                    object_version=version,
                    object_fingerprint=sha256(content).hexdigest(),
                ),
                visibility_ceiling=(
                    spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                ),
                outcome_access=spec.expected_outcome_access
                or OutcomeAccess.OUTCOME_BLIND,
            )
            return ExternalInputPayload.from_bytes(
                logical_artifact_id=logical_id,
                payload_schema=schema,
                profile=ArtifactProfile.TEXT_PARAMETERS,
                media_type="application/json",
                payload=content,
                visibility_ceiling=parent.visibility_ceiling,
                outcome_access=parent.outcome_access,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=sha256(content).hexdigest(),
            )

        return (
            payload(
                CONFIG_ARTIFACT_ID,
                schema=MATERIAL_CONTROL_CONFIG_SCHEMA,
                content=self.config_payload,
                version=MATERIAL_CONTROL_CONFIG_VERSION,
            ),
            payload(
                SOURCE_CONTRACT_ARTIFACT_ID,
                schema=MaterialControlControlBundleQualification.SCHEMA,
                content=self.source_contract.canonical_bytes(),
                version=MaterialControlControlBundleQualification.VERSION,
            ),
        )

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        del execution_plan
        if registry != self.registry:
            raise ValueError('material control semantic registry differs')
        contracts: list[CapabilityOutputSemanticContract] = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                if schema == RAW_ARCHIVE_SCHEMA:
                    contracts.append(
                        CapabilityOutputSemanticContract.from_manifest(
                            manifest,
                            payload_schema=schema,
                            profile=ArtifactProfile.AUDITED_HDF5,
                        )
                    )
                    continue
                record_type = (
                    ScientificAdjudicationRecord
                    if schema == MATERIAL_CONTROL_ADJUDICATION_SCHEMA
                    else _OUTPUT_TYPES[schema]
                )
                contracts.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(
                            sorted(
                                value.name for value in fields(cast(Any, record_type))
                            )
                        ),
                    )
                )
        return tuple(sorted(contracts, key=lambda value: value.key))

    def scientific_adjudication_contract(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> ScientificAdjudicationOutputContract:
        del execution_plan
        if registry != self.registry:
            raise ValueError('material control adjudication registry differs')
        return ScientificAdjudicationOutputContract(
            capability_key=MATERIAL_CONTROL_CLOSEOUT_CAPABILITY_KEY,
            capability_version=MATERIAL_CONTROL_CAPABILITY_VERSION,
            output_id=f'{CLOSEOUT_STEP}.scientific-adjudication',
            payload_schema=MATERIAL_CONTROL_ADJUDICATION_SCHEMA,
            maximum_bytes=128 * 1024,
            fixture_scope_id=None,
            plumbing_only=False,
        )


__all__ = [
    "CLOSEOUT_STEP",
    "CONFIG_ARTIFACT_ID",
    "METHOD_STEP",
    "PANEL_TASKS",
    "PROFILE_TASKS",
    "SOURCE_CONTRACT_ARTIFACT_ID",
    "SOURCE_STEP",
    "WORKFLOW_TASKS",
    'CONSTRUCTIVE_SEARCH_CONFORMANCE_STEP',
    'MaterialControlRunner',
    'MaterialControlRuntimeProvider',
    'material_control_config_ref',
    'material_control_protocol',
]

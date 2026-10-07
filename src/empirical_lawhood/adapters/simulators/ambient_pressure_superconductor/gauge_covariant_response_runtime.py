'Receipt-first runtime and frozen protocols for ambient pressure superconductor amendment gauge covariant response.'

from __future__ import annotations

from dataclasses import fields
from hashlib import sha256
from typing import Any, TypeVar, cast

from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.planning.study_issue import StudyOperationAuthority
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
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)

from .gauge_covariant_response_contracts import GAUGE_COVARIANT_RESPONSE_ADJUDICATION_SCHEMA, GAUGE_COVARIANT_RESPONSE_CONTROL_SCIENCE_FREEZE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_MATCHED_WAVE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_TRANSPORT_NOMINATION_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_ADMISSION_COMPILER_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CONDITIONAL_PROSPECTIVE_CONTROL_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_INDEPENDENT_RECURRENCE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION, GAUGE_COVARIANT_RESPONSE_CLOSEOUT_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA, GAUGE_COVARIANT_RESPONSE_CONFIG_VERSION, GAUGE_COVARIANT_RESPONSE_DEVELOPMENT_BASIS_FREEZE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_MATERIAL_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_CONFORMANCE_CAPABILITY_KEY, GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY, GaugeCovariantResponseCloseout, GaugeCovariantResponseConfig, GaugeCovariantResponseDevelopmentBasisFreeze, GaugeCovariantResponseSourceQualification, GaugeCovariantResponseStage, GaugeCovariantResponseStageResult, GaugeCovariantResponseConformanceResult, GaugeCovariantResponseMaterialInput, GaugeCovariantResponseMaterialResult, GaugeCovariantResponseResponseObservation, decode_gauge_covariant_response_config_bytes, SyntheticMaterialMethodPrerequisites
from .gauge_covariant_response_design import build_material_control_control_result, build_closeout, build_development_basis_freeze, build_nonattempt
from .synthetic_gauge_covariant_response import produce_material_gauge_covariant_response, run_gauge_covariant_response_conformance
from .gauge_covariant_response_source import expected_gauge_covariant_response_source_qualification, inspect_gauge_covariant_response_external_sources
from .material_source_design_system import task_budget


CONFIG_ARTIFACT_ID = 'config-artifact.ambient-pressure-superconductor-gauge-covariant-response'
SOURCE_CONTRACT_ARTIFACT_ID = 'source-contract.ambient-pressure-superconductor-gauge-covariant-response-gauge-covariant-response-extension'
SOURCE_STEP = 'gauge-covariant-response-source-qualification'
GAUGE_COVARIANT_RESPONSE_STEP = 'gauge-covariant-response-gauge-covariant-response-conformance'
DESIGN_STEP = 'gauge-covariant-response-design-freeze'
MATERIAL_CONTROL_GATE_STEP = 'gauge-covariant-response-material-control-control-gate'
DIVERSE_SEED_ACTIONS_STEP = 'gauge-covariant-response-diverse-seed-actions-seed-slot'
RESPONSE_GUIDED_WAVE_1_STEP = 'gauge-covariant-response-response-guided-exploration-wave-1-slot'
RESPONSE_GUIDED_WAVE_2_STEP = 'gauge-covariant-response-response-guided-exploration-wave-2-slot'
TRANSPORT_NOMINATION_STEP = 'gauge-covariant-response-transport-nomination-transport-slot'
COMPUTATIONAL_ADMISSION_STEP = 'gauge-covariant-response-computational-admission-admission-slot'
CLOSEOUT_STEP = 'gauge-covariant-response-development-closeout-closeout'

_GAUGE_COVARIANT_RESPONSE_FIXTURE_NAMES = (
    "finite-q-drift",
    "insulator",
    "missing-diamagnetic",
    "nonlinear",
    "normal",
    "positive",
    "view-disagreement",
    "ward-failure",
    "wrong-sign",
)


def gauge_covariant_response_config_ref(config: GaugeCovariantResponseConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id='config.ambient-pressure-superconductor-gauge-covariant-response-staged',
        config_schema=GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA,
        config_schema_sha256=sha256(GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA.encode()).hexdigest(),
        content_sha256=config.payload_sha256,
        artifact_id=CONFIG_ARTIFACT_ID,
    )


def _output(output_id: str, schema: str) -> OutputTemplate:
    return OutputTemplate(
        output_id=output_id,
        payload_schema=schema,
        profile=ArtifactProfile.CANONICAL_JSON,
        media_type="application/json",
        filename_suffix=".json",
    )


def _step(
    *,
    registry: CapabilityRegistry,
    config: GaugeCovariantResponseConfig,
    step_id: str,
    stage: ScientificStage,
    capability_key: str,
    dependencies: tuple[str, ...],
    outputs: tuple[OutputTemplate, ...],
    access: OutcomeAccess,
    visibility: VisibilityCeiling,
    barrier: BarrierKind,
    obligations: tuple[str, ...],
    resource_lock_ids: tuple[str, ...] = (),
) -> ProtocolStepTemplate:
    manifest = registry.resolve(capability_key, GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION)
    return ProtocolStepTemplate(
        step_id=step_id,
        stage=stage,
        capability_key=capability_key,
        capability_version=GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION,
        config=gauge_covariant_response_config_ref(config),
        dependency_step_ids=tuple(sorted(dependencies)),
        outputs=tuple(sorted(outputs, key=lambda value: value.output_id)),
        required_permissions=manifest.permissions,
        requested_outcome_access=access,
        visibility_ceiling=visibility,
        resource_budget=task_budget(config),
        resource_lock_ids=tuple(sorted(resource_lock_ids)),
        barrier=barrier,
        maximum_attempts=1,
        obligation_ids=tuple(sorted(obligations)),
    )


def gauge_covariant_response_development_protocol(
    *, registry: CapabilityRegistry, config: GaugeCovariantResponseConfig
) -> ProtocolTemplate:
    gauge_covariant_outputs = (_output('gauge-covariant-response-conformance', GaugeCovariantResponseConformanceResult.SCHEMA),) + tuple(
        _output(f'gauge-covariant-response-observation-{name}', GaugeCovariantResponseResponseObservation.SCHEMA)
        for name in _GAUGE_COVARIANT_RESPONSE_FIXTURE_NAMES
    )
    steps = (
        _step(
            registry=registry,
            config=config,
            step_id=SOURCE_STEP,
            stage=ScientificStage.QUALIFY,
            capability_key=GAUGE_COVARIANT_RESPONSE_EXTENSION_QUALIFICATION_CAPABILITY_KEY,
            dependencies=(),
            outputs=(_output("source-qualification", GaugeCovariantResponseSourceQualification.SCHEMA),),
            access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
            obligations=('gauge-covariant-response-exact-gauge-covariant-response-source-extension',),
            resource_lock_ids=('ambient-pressure-superconductor-gauge-covariant-response-source-scan',),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=GAUGE_COVARIANT_RESPONSE_STEP,
            stage=ScientificStage.FALSIFY,
            capability_key=GAUGE_COVARIANT_RESPONSE_CONFORMANCE_CAPABILITY_KEY,
            dependencies=(),
            outputs=gauge_covariant_outputs,
            access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
            obligations=('gauge-covariant-response-gauge-covariant-response-nine-fixture-conformance-intersection',),
            resource_lock_ids=('ambient-pressure-superconductor-gauge-covariant-response-gauge-covariant-response-cpu',),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=DESIGN_STEP,
            stage=ScientificStage.FREEZE,
            capability_key=GAUGE_COVARIANT_RESPONSE_DEVELOPMENT_BASIS_FREEZE_CAPABILITY_KEY,
            dependencies=(SOURCE_STEP, GAUGE_COVARIANT_RESPONSE_STEP),
            outputs=(
                _output("development-basis-freeze", GaugeCovariantResponseDevelopmentBasisFreeze.SCHEMA),
            ),
            access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.FREEZE,
            obligations=(
                'gauge-covariant-response-full-development-template-and-provider-freeze',
                'gauge-covariant-response-target-contact-count-zero',
            ),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=MATERIAL_CONTROL_GATE_STEP,
            stage=ScientificStage.QUALIFY,
            capability_key=GAUGE_COVARIANT_RESPONSE_CONTROL_SCIENCE_FREEZE_CAPABILITY_KEY,
            dependencies=(SOURCE_STEP, GAUGE_COVARIANT_RESPONSE_STEP, DESIGN_STEP),
            outputs=(_output('material-control-stage-result', GaugeCovariantResponseStageResult.SCHEMA),),
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.NONE,
            obligations=('material-control-noncompensating-control-and-science-freeze-gate',),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=DIVERSE_SEED_ACTIONS_STEP,
            stage=ScientificStage.EXPLORE,
            capability_key=GAUGE_COVARIANT_RESPONSE_MATCHED_WAVE_CAPABILITY_KEY,
            dependencies=(MATERIAL_CONTROL_GATE_STEP,),
            outputs=(_output('diverse-seed-actions-stage-result', GaugeCovariantResponseStageResult.SCHEMA),),
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.NONE,
            obligations=('diverse-seed-actions-frozen-diverse-seed-slot-or-typed-nonattempt',),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=RESPONSE_GUIDED_WAVE_1_STEP,
            stage=ScientificStage.EXPLORE,
            capability_key=GAUGE_COVARIANT_RESPONSE_MATCHED_WAVE_CAPABILITY_KEY,
            dependencies=(DIVERSE_SEED_ACTIONS_STEP,),
            outputs=(_output('response-guided-exploration-wave-1-stage-result', GaugeCovariantResponseStageResult.SCHEMA),),
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.FREEZE,
            obligations=('response-guided-exploration-frozen-wave-one-slot-or-typed-nonattempt',),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=RESPONSE_GUIDED_WAVE_2_STEP,
            stage=ScientificStage.EXPLORE,
            capability_key=GAUGE_COVARIANT_RESPONSE_MATCHED_WAVE_CAPABILITY_KEY,
            dependencies=(RESPONSE_GUIDED_WAVE_1_STEP,),
            outputs=(_output('response-guided-exploration-wave-2-stage-result', GaugeCovariantResponseStageResult.SCHEMA),),
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.FREEZE,
            obligations=('response-guided-exploration-frozen-wave-two-slot-or-typed-nonattempt',),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=TRANSPORT_NOMINATION_STEP,
            stage=ScientificStage.FALSIFY,
            capability_key=GAUGE_COVARIANT_RESPONSE_TRANSPORT_NOMINATION_CAPABILITY_KEY,
            dependencies=(RESPONSE_GUIDED_WAVE_2_STEP,),
            outputs=(_output('transport-nomination-stage-result', GaugeCovariantResponseStageResult.SCHEMA),),
            access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.NONE,
            obligations=(
                'transport-nomination-transport-and-maximum-two-gauge-covariant-response-nominations-or-typed-nonattempt',
            ),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=COMPUTATIONAL_ADMISSION_STEP,
            stage=ScientificStage.ADMISSION,
            capability_key=GAUGE_COVARIANT_RESPONSE_ADMISSION_COMPILER_CAPABILITY_KEY,
            dependencies=(TRANSPORT_NOMINATION_STEP, DESIGN_STEP),
            outputs=(_output('computational-admission-stage-result', GaugeCovariantResponseStageResult.SCHEMA),),
            access=OutcomeAccess.EVALUATION_SEALED,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.FREEZE,
            obligations=(
                'computational-admission-strict-gauge-covariant-response-complete-admission-and-current-compiler-or-typed-nonattempt',
            ),
        ),
        _step(
            registry=registry,
            config=config,
            step_id=CLOSEOUT_STEP,
            stage=ScientificStage.EVALUATE,
            capability_key=GAUGE_COVARIANT_RESPONSE_CLOSEOUT_CAPABILITY_KEY,
            dependencies=(COMPUTATIONAL_ADMISSION_STEP, DESIGN_STEP),
            outputs=(
                _output("closeout", GaugeCovariantResponseCloseout.SCHEMA),
                _output("scientific-adjudication", GAUGE_COVARIANT_RESPONSE_ADJUDICATION_SCHEMA),
            ),
            access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.REVEAL,
            obligations=('development-closeout-orthogonal-terminal-closeout',),
        ),
    )
    return ProtocolTemplate(
        template_id='protocol.ambient-pressure-superconductor-gauge-covariant-response-material-source-design-computational-admission-development',
        template_version=GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=True,
        requests_controller=False,
        nonactuating=True,
    )


def gauge_covariant_response_conditional_protocols(
    *, registry: CapabilityRegistry, config: GaugeCovariantResponseConfig
) -> tuple[ProtocolTemplate, ...]:
    prospective_controller_validation = _step(
        registry=registry,
        config=config,
        step_id='gauge-covariant-response-prospective-controller-validation-conditional-controller-use',
        stage=ScientificStage.PREPARE,
        capability_key=GAUGE_COVARIANT_RESPONSE_CONDITIONAL_PROSPECTIVE_CONTROL_CAPABILITY_KEY,
        dependencies=(),
        outputs=(_output('prospective-controller-validation-stage-result', GaugeCovariantResponseStageResult.SCHEMA),),
        access=OutcomeAccess.EVALUATION_SEALED,
        visibility=VisibilityCeiling.OUTCOME_VISIBLE,
        barrier=BarrierKind.FREEZE,
        obligations=('prospective-controller-validation-parent-receipt-bound-fresh-controller-use',),
    )
    independent_recurrence = _step(
        registry=registry,
        config=config,
        step_id='gauge-covariant-response-independent-recurrence-independent-recurrence',
        stage=ScientificStage.ADMISSION,
        capability_key=GAUGE_COVARIANT_RESPONSE_INDEPENDENT_RECURRENCE_CAPABILITY_KEY,
        dependencies=(),
        outputs=(_output('independent-recurrence-stage-result', GaugeCovariantResponseStageResult.SCHEMA),),
        access=OutcomeAccess.EVALUATION_SEALED,
        visibility=VisibilityCeiling.OUTCOME_VISIBLE,
        barrier=BarrierKind.FREEZE,
        obligations=('independent-recurrence-independent-pairing-gauge-covariant-response-receiver-and-formation-recurrence',),
    )
    material_gauge_covariant_response = _step(
        registry=registry,
        config=config,
        step_id='gauge-covariant-response-computational-admission-material-gauge-covariant-response-provider',
        stage=ScientificStage.PREPARE,
        capability_key=GAUGE_COVARIANT_RESPONSE_MATERIAL_CAPABILITY_KEY,
        dependencies=(),
        outputs=(_output('material-gauge-covariant-response-result', GaugeCovariantResponseMaterialResult.SCHEMA),),
        access=OutcomeAccess.EVALUATION_SEALED,
        visibility=VisibilityCeiling.OUTCOME_VISIBLE,
        barrier=BarrierKind.FREEZE,
        obligations=('computational-admission-material-specific-strict-gauge-covariant-response-provider-slot',),
    )
    return (
        ProtocolTemplate(
            template_id='protocol.ambient-pressure-superconductor-gauge-covariant-response-computational-admission-material-gauge-covariant-response-slot',
            template_version=GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION,
            steps=(material_gauge_covariant_response,),
            requires_model_set=True,
            requests_controller=False,
            nonactuating=True,
        ),
        ProtocolTemplate(
            template_id='protocol.ambient-pressure-superconductor-gauge-covariant-response-conditional-prospective-controller-validation-controller-use',
            template_version=GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION,
            steps=(prospective_controller_validation,),
            requires_model_set=True,
            requests_controller=False,
            nonactuating=True,
        ),
        ProtocolTemplate(
            template_id='protocol.ambient-pressure-superconductor-gauge-covariant-response-conditional-independent-recurrence-recurrence',
            template_version=GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION,
            steps=(independent_recurrence,),
            requires_model_set=True,
            requests_controller=False,
            nonactuating=True,
        ),
    )


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _one(records: tuple[CanonicalRecord, ...], record_type: type[_RecordT]) -> _RecordT:
    values = tuple(value for value in records if isinstance(value, record_type))
    if len(values) != 1:
        raise ValueError(f'gauge covariant response task requires exactly one {record_type.__name__}')
    return values[0]


def _stage(records: tuple[CanonicalRecord, ...], stage: GaugeCovariantResponseStage) -> GaugeCovariantResponseStageResult:
    values = tuple(
        value
        for value in records
        if isinstance(value, GaugeCovariantResponseStageResult) and value.stage is stage
    )
    if len(values) != 1:
        raise ValueError(f'gauge covariant response task requires exactly one {stage.value} result')
    return values[0]


_INPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    GaugeCovariantResponseSourceQualification.SCHEMA: GaugeCovariantResponseSourceQualification,
    GaugeCovariantResponseConformanceResult.SCHEMA: GaugeCovariantResponseConformanceResult,
    GaugeCovariantResponseResponseObservation.SCHEMA: GaugeCovariantResponseResponseObservation,
    GaugeCovariantResponseMaterialInput.SCHEMA: GaugeCovariantResponseMaterialInput,
    GaugeCovariantResponseDevelopmentBasisFreeze.SCHEMA: GaugeCovariantResponseDevelopmentBasisFreeze,
    GaugeCovariantResponseStageResult.SCHEMA: GaugeCovariantResponseStageResult,
}


class GaugeCovariantResponseRunner:
    def __init__(
        self,
        manifest: CapabilityManifest,
        config: GaugeCovariantResponseConfig,
        config_payload: bytes,
        source_contract: GaugeCovariantResponseSourceQualification,
        development_protocol: ProtocolTemplate,
        development_registry: CapabilityRegistry,
        conditional_protocols: tuple[ProtocolTemplate, ...],
        conditional_registry: CapabilityRegistry,
    ) -> None:
        if (
            decode_gauge_covariant_response_config_bytes(
                config_payload, expected_parents=SyntheticMaterialMethodPrerequisites.from_config(config)
            )
            != config
        ):
            raise ValueError('gauge covariant response runner config differs from registered bytes')
        self.manifest = manifest
        self.config = config
        self.config_payload = config_payload
        self.source_contract = source_contract
        self.development_protocol = development_protocol
        self.development_registry = development_registry
        self.conditional_protocols = conditional_protocols
        self.conditional_registry = conditional_registry
        self.execution_count = 0

    def _read(self, context: TaskContext) -> tuple[CanonicalRecord, ...]:
        records: list[CanonicalRecord] = []
        observed_config = False
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError('gauge covariant response task input size differs')
            if port.payload_schema == GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA:
                if (
                    payload != self.config_payload
                    or decode_gauge_covariant_response_config_bytes(
                        payload,
                        expected_parents=SyntheticMaterialMethodPrerequisites.from_config(self.config),
                    )
                    != self.config
                ):
                    raise ValueError('gauge covariant response task config materialization differs')
                observed_config = True
                continue
            try:
                record_type = _INPUT_TYPES[port.payload_schema]
            except KeyError as error:
                raise ValueError('gauge covariant response task received an unknown input schema') from error
            records.append(
                decode_canonical_bytes(
                    payload, record_type, maximum_bytes=port.size_bytes
                )
            )
        if (
            not observed_config
            or context.config.content_sha256 != self.config.payload_sha256
        ):
            raise ValueError('gauge covariant response task lacks its exact config')
        return tuple(records)

    @staticmethod
    def _adjudication(
        context: TaskContext, closeout: GaugeCovariantResponseCloseout
    ) -> ScientificAdjudicationRecord:
        adjudication_context = context.scientific_adjudication_context
        if adjudication_context is None:
            raise ValueError('gauge covariant response closeout lacks scientific adjudication context')
        output_ids = tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
            )
        )
        if len(output_ids) != len(context.output_ports):
            raise ValueError('gauge covariant response closeout output lacks logical identity')
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
            evaluability=AdjudicationEvaluability.UNEVALUABLE,
            scientific_status=ScientificStatus.UNEVALUABLE,
            admission_status=AdmissionStatus.UNEVALUABLE,
            reason_codes=closeout.reason_codes,
            fixture_scope_id=None,
            plumbing_only=False,
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        records = self._read(context)
        outputs: dict[str, CanonicalRecord | bytes]
        if context.task_id == SOURCE_STEP:
            contract = _one(records, GaugeCovariantResponseSourceQualification)
            if contract != self.source_contract:
                raise ValueError('gauge covariant response source contract differs')
            observed = inspect_gauge_covariant_response_external_sources(
                base_source_sha256=self.config.base_source_sha256
            )
            if observed != contract:
                raise ValueError('gauge covariant response physical source extension differs')
            outputs = {"source-qualification": observed}
        elif context.task_id == GAUGE_COVARIANT_RESPONSE_STEP:
            conformance, observations = run_gauge_covariant_response_conformance()
            if conformance.formalism_sha256 != self.config.gauge_covariant_formalism_sha256:
                raise ValueError('gauge covariant response executed gauge covariant response formalism differs from config')
            if conformance.fixture_suite_sha256 != self.config.gauge_covariant_fixture_suite_sha256:
                raise ValueError('gauge covariant response executed gauge covariant response fixture suite differs from config')
            outputs = {'gauge-covariant-response-conformance': conformance}
            outputs.update(
                {
                    f"gauge-covariant-response-observation-{value.fixture_id.removeprefix('fixture.ambient-pressure-superconductor-gauge-covariant-response-')}": value
                    for value in observations
                }
            )
        elif context.task_id == 'gauge-covariant-response-computational-admission-material-gauge-covariant-response-provider':
            outputs = {
                'material-gauge-covariant-response-result': produce_material_gauge_covariant_response(
                    _one(records, GaugeCovariantResponseMaterialInput)
                )
            }
        elif context.task_id == DESIGN_STEP:
            outputs = {
                "development-basis-freeze": build_development_basis_freeze(
                    config=self.config,
                    source=_one(records, GaugeCovariantResponseSourceQualification),
                    conformance=_one(records, GaugeCovariantResponseConformanceResult),
                    development_protocol=self.development_protocol,
                    development_registry=self.development_registry,
                    conditional_protocols=self.conditional_protocols,
                    conditional_registry=self.conditional_registry,
                )
            }
        elif context.task_id == MATERIAL_CONTROL_GATE_STEP:
            source = _one(records, GaugeCovariantResponseSourceQualification)
            if source.fingerprint() != self.config.source_extension_sha256:
                raise ValueError('material control source binding differs')
            outputs = {
                'material-control-stage-result': build_material_control_control_result(
                    design=_one(records, GaugeCovariantResponseDevelopmentBasisFreeze),
                    conformance=_one(records, GaugeCovariantResponseConformanceResult),
                )
            }
        elif context.task_id == DIVERSE_SEED_ACTIONS_STEP:
            outputs = {
                'diverse-seed-actions-stage-result': build_nonattempt(
                    stage=GaugeCovariantResponseStage.DIVERSE_SEED_ACTIONS, upstream=_stage(records, GaugeCovariantResponseStage.MATERIAL_CONTROL)
                )
            }
        elif context.task_id == RESPONSE_GUIDED_WAVE_1_STEP:
            outputs = {
                'response-guided-exploration-wave-1-stage-result': build_nonattempt(
                    stage=GaugeCovariantResponseStage.RESPONSE_GUIDED_WAVE_1, upstream=_stage(records, GaugeCovariantResponseStage.DIVERSE_SEED_ACTIONS)
                )
            }
        elif context.task_id == RESPONSE_GUIDED_WAVE_2_STEP:
            outputs = {
                'response-guided-exploration-wave-2-stage-result': build_nonattempt(
                    stage=GaugeCovariantResponseStage.RESPONSE_GUIDED_WAVE_2,
                    upstream=_stage(records, GaugeCovariantResponseStage.RESPONSE_GUIDED_WAVE_1),
                )
            }
        elif context.task_id == TRANSPORT_NOMINATION_STEP:
            outputs = {
                'transport-nomination-stage-result': build_nonattempt(
                    stage=GaugeCovariantResponseStage.TRANSPORT_NOMINATION, upstream=_stage(records, GaugeCovariantResponseStage.RESPONSE_GUIDED_WAVE_2)
                )
            }
        elif context.task_id == COMPUTATIONAL_ADMISSION_STEP:
            design = _one(records, GaugeCovariantResponseDevelopmentBasisFreeze)
            if not design.computational_admission_slots_frozen:
                raise ValueError('computational admission received an unfrozen conditional slot')
            outputs = {
                'computational-admission-stage-result': build_nonattempt(
                    stage=GaugeCovariantResponseStage.COMPUTATIONAL_ADMISSION, upstream=_stage(records, GaugeCovariantResponseStage.TRANSPORT_NOMINATION)
                )
            }
        elif context.task_id == CLOSEOUT_STEP:
            closeout = build_closeout(
                config=self.config,
                design=_one(records, GaugeCovariantResponseDevelopmentBasisFreeze),
                computational_admission=_stage(records, GaugeCovariantResponseStage.COMPUTATIONAL_ADMISSION),
            )
            outputs = {
                "closeout": closeout,
                "scientific-adjudication": encode_scientific_adjudication(
                    self._adjudication(context, closeout),
                    payload_schema=GAUGE_COVARIANT_RESPONSE_ADJUDICATION_SCHEMA,
                ),
            }
        else:
            raise ValueError('unknown gauge covariant response task identity')
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=(
                        value
                        if isinstance(
                            value := outputs[
                                port.output_id.removeprefix(f'{context.task_id}.')
                            ],
                            bytes,
                        )
                        else value.canonical_bytes()
                    ),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("exact-config-decoded", True, ()),
                ReceiptCheck("external-scientific-root-only", True, ()),
                ReceiptCheck("target-cutoff-preserved", True, ()),
                ReceiptCheck("typed-output-produced", True, ()),
            ),
        )


_OUTPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    GaugeCovariantResponseSourceQualification.SCHEMA: GaugeCovariantResponseSourceQualification,
    GaugeCovariantResponseConformanceResult.SCHEMA: GaugeCovariantResponseConformanceResult,
    GaugeCovariantResponseResponseObservation.SCHEMA: GaugeCovariantResponseResponseObservation,
    GaugeCovariantResponseMaterialResult.SCHEMA: GaugeCovariantResponseMaterialResult,
    GaugeCovariantResponseDevelopmentBasisFreeze.SCHEMA: GaugeCovariantResponseDevelopmentBasisFreeze,
    GaugeCovariantResponseStageResult.SCHEMA: GaugeCovariantResponseStageResult,
    GaugeCovariantResponseCloseout.SCHEMA: GaugeCovariantResponseCloseout,
}


class GaugeCovariantResponseRuntimeProvider(CampaignRuntimeProvider):
    issued_source_schema_ids: tuple[str, ...] = (StudyOperationAuthority.SCHEMA,)

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        conditional_registry: CapabilityRegistry,
        config: GaugeCovariantResponseConfig,
        config_payload: bytes,
        source_contract: GaugeCovariantResponseSourceQualification,
    ) -> None:
        if any(
            manifest.implementation_sha256
            != registry.capabilities[0].implementation_sha256
            for manifest in (*registry.capabilities, *conditional_registry.capabilities)
        ):
            raise ValueError('gauge covariant response registries mix implementation identities')
        if (
            decode_gauge_covariant_response_config_bytes(
                config_payload, expected_parents=SyntheticMaterialMethodPrerequisites.from_config(config)
            )
            != config
        ):
            raise ValueError('gauge covariant response provider config differs from registered bytes')
        if source_contract != expected_gauge_covariant_response_source_qualification(
            base_source_sha256=config.base_source_sha256
        ):
            raise ValueError('gauge covariant response provider source-contract binding differs')
        self.registry = registry
        self.conditional_registry = conditional_registry
        self.config = config
        self.config_payload = config_payload
        self.source_contract = source_contract
        self.development_protocol = gauge_covariant_response_development_protocol(
            registry=registry, config=config
        )
        self.conditional_protocols = gauge_covariant_response_conditional_protocols(
            registry=conditional_registry, config=config
        )
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = len(registry.capabilities)
        self._runners: tuple[GaugeCovariantResponseRunner, ...] = ()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        del source_records
        if registry != self.registry:
            raise ValueError('gauge covariant response runner registry differs')
        self._runners = tuple(
            GaugeCovariantResponseRunner(
                manifest,
                self.config,
                self.config_payload,
                self.source_contract,
                self.development_protocol,
                self.registry,
                self.conditional_protocols,
                self.conditional_registry,
            )
            for manifest in registry.capabilities
        )
        return cast(tuple[TaskRunner, ...], self._runners)

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        del source_records
        if plan.registry_sha256 != self.registry_sha256:
            raise ValueError('gauge covariant response execution plan registry differs')
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        expected = {CONFIG_ARTIFACT_ID, SOURCE_CONTRACT_ARTIFACT_ID}
        if set(specs) != expected:
            raise ValueError('gauge covariant response plan external input set differs')

        def payload(
            logical_artifact_id: str,
            *,
            payload_schema: str,
            content: bytes,
            object_version: str,
        ) -> ExternalInputPayload:
            spec = specs[logical_artifact_id]
            parent = ArtifactLineageParent(
                identity=ObjectIdentity(
                    object_id=logical_artifact_id,
                    object_schema=payload_schema,
                    object_version=object_version,
                    object_fingerprint=sha256(content).hexdigest(),
                ),
                visibility_ceiling=spec.expected_visibility_ceiling
                or VisibilityCeiling.PROSPECTIVE,
                outcome_access=spec.expected_outcome_access
                or OutcomeAccess.OUTCOME_BLIND,
            )
            return ExternalInputPayload.from_bytes(
                logical_artifact_id=logical_artifact_id,
                payload_schema=payload_schema,
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
                payload_schema=GAUGE_COVARIANT_RESPONSE_CONFIG_SCHEMA,
                content=self.config_payload,
                object_version=GAUGE_COVARIANT_RESPONSE_CONFIG_VERSION,
            ),
            payload(
                SOURCE_CONTRACT_ARTIFACT_ID,
                payload_schema=GaugeCovariantResponseSourceQualification.SCHEMA,
                content=self.source_contract.canonical_bytes(),
                object_version=GaugeCovariantResponseSourceQualification.VERSION,
            ),
        )

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        del execution_plan
        if registry != self.registry:
            raise ValueError('gauge covariant response semantic registry differs')
        contracts = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                record_type = (
                    ScientificAdjudicationRecord
                    if schema == GAUGE_COVARIANT_RESPONSE_ADJUDICATION_SCHEMA
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
            raise ValueError('gauge covariant response adjudication registry differs')
        return ScientificAdjudicationOutputContract(
            capability_key=GAUGE_COVARIANT_RESPONSE_CLOSEOUT_CAPABILITY_KEY,
            capability_version=GAUGE_COVARIANT_RESPONSE_CAPABILITY_VERSION,
            output_id=f'{CLOSEOUT_STEP}.scientific-adjudication',
            payload_schema=GAUGE_COVARIANT_RESPONSE_ADJUDICATION_SCHEMA,
            maximum_bytes=128 * 1024,
            fixture_scope_id=None,
            plumbing_only=False,
        )


__all__ = [
    'GaugeCovariantResponseRuntimeProvider',
    "MATERIAL_CONTROL_GATE_STEP",
    'DIVERSE_SEED_ACTIONS_STEP',
    'RESPONSE_GUIDED_WAVE_1_STEP',
    'RESPONSE_GUIDED_WAVE_2_STEP',
    'TRANSPORT_NOMINATION_STEP',
    'COMPUTATIONAL_ADMISSION_STEP',
    "CLOSEOUT_STEP",
    "CONFIG_ARTIFACT_ID",
    "DESIGN_STEP",
    'GAUGE_COVARIANT_RESPONSE_STEP',
    "SOURCE_CONTRACT_ARTIFACT_ID",
    "SOURCE_STEP",
    'gauge_covariant_response_conditional_protocols',
    'gauge_covariant_response_config_ref',
    'gauge_covariant_response_development_protocol',
]

'Receipt-first ambient pressure superconductor excluded solver control protocol, runners and runtime provider.'

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
from empirical_lawhood.runtime.artifacts import ArtifactLineageParent, ArtifactProfile, ReceiptCheck
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityManifest,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import RunnerResult, TaskContext, TaskOutputPayload, TaskRunner
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)

from .contracts import ADJUDICATION_SCHEMA, ExcludedSolverControlConfig, ExcludedSolverControlSolverSmokeResult, CAPABILITY_VERSION, CONFIG_SCHEMA, CONFIG_VERSION, EVALUATOR_CAPABILITY_KEY, MaterialPreparationRecord, NumericalViewRecord, PREPARATION_CAPABILITY_KEY, SOLVER_CAPABILITY_KEY, SolverControlInputRecord, SolverRunObservation, SolverSmokeDisposition, ExcludedSolverControlSourceManifest, SOURCE_MANIFEST_SOURCE_ID, decode_config_bytes
from .domain import adjudicate_solver_smoke, build_control_input
from .solver import MaterialSolverExecutor, SolverExecutionRequest, extract_qe_observation
from .source import expected_excluded_solver_control_source_manifest
from .system import task_budget


CONFIG_ARTIFACT_ID = 'config-artifact.ambient-pressure-superconductor-excluded-solver-control'
SOURCE_MANIFEST_ARTIFACT_ID = SOURCE_MANIFEST_SOURCE_ID

PREPARATION_STEP = "material-preparation"
SOLVER_STEP = "qe-solver-smoke"
EVALUATOR_STEP = "solver-smoke-evaluator"


def config_ref(config: ExcludedSolverControlConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id='config.ambient-pressure-superconductor-excluded-solver-control-pb-smoke',
        config_schema=CONFIG_SCHEMA,
        config_schema_sha256=sha256(CONFIG_SCHEMA.encode()).hexdigest(),
        content_sha256=config.payload_sha256,
        artifact_id=CONFIG_ARTIFACT_ID,
    )


def excluded_solver_control_protocol(*, registry: CapabilityRegistry, config: ExcludedSolverControlConfig) -> ProtocolTemplate:
    reference = config_ref(config)

    def step(
        *,
        step_id: str,
        stage: ScientificStage,
        capability_key: str,
        dependencies: tuple[str, ...],
        outputs: tuple[tuple[str, str], ...],
        outcome_access: OutcomeAccess,
        visibility: VisibilityCeiling,
        barrier: BarrierKind,
        obligations: tuple[str, ...],
    ) -> ProtocolStepTemplate:
        manifest = registry.resolve(capability_key, CAPABILITY_VERSION)
        return ProtocolStepTemplate(
            step_id=step_id,
            stage=stage,
            capability_key=capability_key,
            capability_version=CAPABILITY_VERSION,
            config=reference,
            dependency_step_ids=dependencies,
            outputs=tuple(
                sorted(
                    (
                        OutputTemplate(
                            output_id=output_id,
                            payload_schema=schema,
                            profile=ArtifactProfile.CANONICAL_JSON,
                            media_type="application/json",
                            filename_suffix=".json",
                        )
                        for output_id, schema in outputs
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=manifest.permissions,
            requested_outcome_access=outcome_access,
            visibility_ceiling=visibility,
            resource_budget=task_budget(config),
            resource_lock_ids=('ambient-pressure-superconductor-qe76-epw61-fixed-environment',),
            barrier=barrier,
            maximum_attempts=1,
            obligation_ids=tuple(sorted(obligations)),
        )

    steps = (
        step(
            step_id=PREPARATION_STEP,
            stage=ScientificStage.ACQUIRE,
            capability_key=PREPARATION_CAPABILITY_KEY,
            dependencies=(),
            outputs=(
                ("material-preparation", MaterialPreparationRecord.SCHEMA),
                ("numerical-view", NumericalViewRecord.SCHEMA),
                ("solver-control-input", SolverControlInputRecord.SCHEMA),
            ),
            outcome_access=OutcomeAccess.OUTCOME_BLIND,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.NONE,
            obligations=('excluded-solver-control-exact-held-control-preparation',),
        ),
        step(
            step_id=SOLVER_STEP,
            stage=ScientificStage.ACQUIRE,
            capability_key=SOLVER_CAPABILITY_KEY,
            dependencies=(PREPARATION_STEP,),
            outputs=(("solver-observation", SolverRunObservation.SCHEMA),),
            outcome_access=OutcomeAccess.EVALUATION_SEALED,
            visibility=VisibilityCeiling.PROSPECTIVE,
            barrier=BarrierKind.FREEZE,
            obligations=(
                'excluded-solver-control-fixed-executor-binding',
                'excluded-solver-control-qe-output-extraction',
            ),
        ),
        step(
            step_id=EVALUATOR_STEP,
            stage=ScientificStage.EVALUATE,
            capability_key=EVALUATOR_CAPABILITY_KEY,
            dependencies=(PREPARATION_STEP, SOLVER_STEP),
            outputs=(
                ("scientific-adjudication", ADJUDICATION_SCHEMA),
                ("solver-smoke-result", ExcludedSolverControlSolverSmokeResult.SCHEMA),
            ),
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility=VisibilityCeiling.OUTCOME_VISIBLE,
            barrier=BarrierKind.REVEAL,
            obligations=('excluded-solver-control-excluded-control-adjudication',),
        ),
    )
    return ProtocolTemplate(
        template_id='protocol.ambient-pressure-superconductor-excluded-solver-control-pb-solver-smoke',
        template_version=CAPABILITY_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=True,
        requests_controller=False,
        nonactuating=True,
    )


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _one(records: tuple[CanonicalRecord, ...], record_type: type[_RecordT]) -> _RecordT:
    values = tuple(value for value in records if isinstance(value, record_type))
    if len(values) != 1:
        raise ValueError(f'excluded solver control task requires exactly one {record_type.__name__}')
    return values[0]


_INPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    ExcludedSolverControlSourceManifest.SCHEMA: ExcludedSolverControlSourceManifest,
    MaterialPreparationRecord.SCHEMA: MaterialPreparationRecord,
    NumericalViewRecord.SCHEMA: NumericalViewRecord,
    SolverControlInputRecord.SCHEMA: SolverControlInputRecord,
    SolverRunObservation.SCHEMA: SolverRunObservation,
}


class ExcludedSolverControlRunner:
    """Closed runner whose sole external process is the injected fixed profile."""

    def __init__(
        self,
        manifest: CapabilityManifest,
        config: ExcludedSolverControlConfig,
        payload: bytes,
        source_manifest: ExcludedSolverControlSourceManifest,
        executor: MaterialSolverExecutor,
    ) -> None:
        if decode_config_bytes(payload) != config:
            raise ValueError('excluded solver control runner config differs from registered bytes')
        if executor.profile_id != config.solver_profile_id:
            raise ValueError('excluded solver control executor differs from the exact configured profile')
        self.manifest = manifest
        self.config = config
        self.payload = payload
        self.source_manifest = source_manifest
        self.executor = executor
        self.execution_count = 0

    def _read(self, context: TaskContext) -> tuple[CanonicalRecord, ...]:
        records: list[CanonicalRecord] = []
        observed_config = False
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError('excluded solver control task input size differs')
            if port.payload_schema == CONFIG_SCHEMA:
                if payload != self.payload or decode_config_bytes(payload) != self.config:
                    raise ValueError('excluded solver control task config materialization differs')
                observed_config = True
                continue
            try:
                record_type = _INPUT_TYPES[port.payload_schema]
            except KeyError as error:
                raise ValueError('excluded solver control task received an unknown input schema') from error
            records.append(
                decode_canonical_bytes(payload, record_type, maximum_bytes=port.size_bytes)
            )
        if not observed_config or context.config.content_sha256 != self.config.payload_sha256:
            raise ValueError('excluded solver control task lacks its exact config')
        return tuple(records)

    @staticmethod
    def _scientific_adjudication(
        context: TaskContext, result: ExcludedSolverControlSolverSmokeResult
    ) -> ScientificAdjudicationRecord:
        adjudication_context = context.scientific_adjudication_context
        if adjudication_context is None:
            raise ValueError('excluded solver control evaluator lacks scientific adjudication context')
        output_ids = tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
            )
        )
        if len(output_ids) != len(context.output_ports):
            raise ValueError('excluded solver control evaluator output lacks logical identity')
        supported = result.disposition in {
            SolverSmokeDisposition.PASS,
            SolverSmokeDisposition.CORRECTIVE_PASS,
        }
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
            evaluability=AdjudicationEvaluability.EVALUABLE,
            scientific_status=(
                ScientificStatus.SUPPORTED if supported else ScientificStatus.NOT_SUPPORTED
            ),
            admission_status=AdmissionStatus.NOT_EVALUATED,
            reason_codes=result.reason_codes,
            fixture_scope_id=None,
            plumbing_only=False,
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        records = self._read(context)
        outputs: dict[str, CanonicalRecord | bytes]
        if context.task_id == PREPARATION_STEP:
            observed_source = _one(records, ExcludedSolverControlSourceManifest)
            if observed_source != self.source_manifest:
                raise ValueError('excluded solver control held solver-environment manifest differs')
            preparation, view, control = build_control_input(self.config)
            outputs = {
                "material-preparation": preparation,
                "numerical-view": view,
                "solver-control-input": control,
            }
        elif context.task_id == SOLVER_STEP:
            preparation = _one(records, MaterialPreparationRecord)
            view = _one(records, NumericalViewRecord)
            control = _one(records, SolverControlInputRecord)
            if control.preparation_fingerprint != preparation.fingerprint():
                raise ValueError('excluded solver control solver preparation edge differs')
            if control.numerical_view_fingerprint != view.fingerprint():
                raise ValueError('excluded solver control solver numerical-view edge differs')
            execution = self.executor.execute(
                SolverExecutionRequest(
                    request_id=f'{context.run_id}.{context.attempt_id}',
                    profile_id=self.config.solver_profile_id,
                    config_sha256=self.config.payload_sha256,
                    environment_image_sha256=self.config.environment_image_sha256,
                    qe_binary_sha256=self.config.qe_binary_sha256,
                    input_sha256=self.config.input_sha256,
                    pseudopotential_sha256=self.config.pseudopotential_sha256,
                    wall_time_seconds=context.resource_budget.wall_time_seconds,
                    cpu_cores=context.resource_budget.cpu_cores,
                )
            )
            outputs = {
                "solver-observation": extract_qe_observation(
                    config=self.config,
                    result=execution,
                    observation_id=f'observation.{context.run_id}.{context.attempt_id}',
                )
            }
        elif context.task_id == EVALUATOR_STEP:
            result = adjudicate_solver_smoke(
                self.config,
                _one(records, SolverControlInputRecord),
                _one(records, SolverRunObservation),
            )
            outputs = {
                "solver-smoke-result": result,
                "scientific-adjudication": encode_scientific_adjudication(
                    self._scientific_adjudication(context, result),
                    payload_schema=ADJUDICATION_SCHEMA,
                ),
            }
        else:
            raise ValueError('unknown excluded solver control task identity')
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=(
                        value
                        if isinstance(
                            value := outputs[port.output_id.removeprefix(f'{context.task_id}.')],
                            bytes,
                        )
                        else value.canonical_bytes()
                    ),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("exact-config-decoded", True, ()),
                ReceiptCheck("fixed-executor-profile-bound", True, ()),
                ReceiptCheck("target-contact-count-zero", True, ()),
                ReceiptCheck("typed-output-produced", True, ()),
            ),
        )


_OUTPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    ExcludedSolverControlSolverSmokeResult.SCHEMA: ExcludedSolverControlSolverSmokeResult,
    MaterialPreparationRecord.SCHEMA: MaterialPreparationRecord,
    NumericalViewRecord.SCHEMA: NumericalViewRecord,
    SolverControlInputRecord.SCHEMA: SolverControlInputRecord,
    SolverRunObservation.SCHEMA: SolverRunObservation,
}


class ExcludedSolverControlRuntimeProvider(CampaignRuntimeProvider):
    issued_source_schema_ids: tuple[str, ...] = (StudyOperationAuthority.SCHEMA,)

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: ExcludedSolverControlConfig,
        config_payload: bytes,
        source_manifest: ExcludedSolverControlSourceManifest,
        executor: MaterialSolverExecutor,
    ) -> None:
        if any(
            manifest.implementation_sha256 != registry.capabilities[0].implementation_sha256
            for manifest in registry.capabilities
        ):
            raise ValueError('excluded solver control registry mixes implementation identities')
        if decode_config_bytes(config_payload) != config:
            raise ValueError('excluded solver control provider config differs from registered bytes')
        if source_manifest != expected_excluded_solver_control_source_manifest(config):
            raise ValueError('excluded solver control provider source-manifest binding differs')
        if executor.profile_id != config.solver_profile_id:
            raise ValueError('excluded solver control provider executor binding differs')
        self.registry = registry
        self.config = config
        self.config_payload = config_payload
        self.source_manifest = source_manifest
        self.executor = executor
        self.registry_sha256 = registry.fingerprint()
        self._runners: tuple[ExcludedSolverControlRunner, ...] = ()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        del source_records
        if registry != self.registry:
            raise ValueError('excluded solver control runner registry differs')
        self._runners = tuple(
            ExcludedSolverControlRunner(
                manifest,
                self.config,
                self.config_payload,
                self.source_manifest,
                self.executor,
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
            raise ValueError('excluded solver control execution plan registry differs')
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        expected = {CONFIG_ARTIFACT_ID, SOURCE_MANIFEST_ARTIFACT_ID}
        if set(specs) != expected:
            raise ValueError('excluded solver control plan external input set differs')

        def payload(
            logical_artifact_id: str,
            *,
            payload_schema: str,
            content: bytes,
            object_id: str,
            object_version: str,
        ) -> ExternalInputPayload:
            spec = specs[logical_artifact_id]
            parent = ArtifactLineageParent(
                identity=ObjectIdentity(
                    object_id=object_id,
                    object_schema=payload_schema,
                    object_version=object_version,
                    object_fingerprint=sha256(content).hexdigest(),
                ),
                visibility_ceiling=(
                    spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                ),
                outcome_access=spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND,
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
                payload_schema=CONFIG_SCHEMA,
                content=self.config_payload,
                object_id=CONFIG_ARTIFACT_ID,
                object_version=CONFIG_VERSION,
            ),
            payload(
                SOURCE_MANIFEST_ARTIFACT_ID,
                payload_schema=ExcludedSolverControlSourceManifest.SCHEMA,
                content=self.source_manifest.canonical_bytes(),
                object_id=SOURCE_MANIFEST_SOURCE_ID,
                object_version=ExcludedSolverControlSourceManifest.VERSION,
            ),
        )

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        del execution_plan
        if registry != self.registry:
            raise ValueError('excluded solver control semantic registry differs')
        contracts = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                record_type = (
                    ScientificAdjudicationRecord
                    if schema == ADJUDICATION_SCHEMA
                    else _OUTPUT_TYPES[schema]
                )
                contracts.append(
                    CapabilityOutputSemanticContract.from_manifest(
                        manifest,
                        payload_schema=schema,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        top_level_keys=("schema", "value", "version"),
                        value_keys=tuple(
                            sorted(value.name for value in fields(cast(Any, record_type)))
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
            raise ValueError('excluded solver control adjudication registry differs')
        return ScientificAdjudicationOutputContract(
            capability_key=EVALUATOR_CAPABILITY_KEY,
            capability_version=CAPABILITY_VERSION,
            output_id=f'{EVALUATOR_STEP}.scientific-adjudication',
            payload_schema=ADJUDICATION_SCHEMA,
            maximum_bytes=128 * 1024,
            fixture_scope_id=None,
            plumbing_only=False,
        )


__all__ = [
    'ExcludedSolverControlRunner',
    'ExcludedSolverControlRuntimeProvider',
    "CONFIG_ARTIFACT_ID",
    "EVALUATOR_STEP",
    "SOURCE_MANIFEST_ARTIFACT_ID",
    "PREPARATION_STEP",
    "SOLVER_STEP",
    'excluded_solver_control_protocol',
    "config_ref",
]

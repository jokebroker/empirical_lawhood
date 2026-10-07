"""Closed capability/runtime composition for the synthetic physical scale morphism rehearsal."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import Any, TypeVar, cast

from empirical_lawhood.adapters.physical.rc_ladder_response.provider import RcLadderResponseSyntheticSourceQualification, qualify_synthetic_source_adapter
from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.contracts import PhysicalScaleMorphismTruthMethodSuiteResult
from empirical_lawhood.adapters.reference_worlds.physical_scale_morphism.provider import PhysicalScaleMorphismTruthCaseBatch, PhysicalScaleMorphismTruthObservationBatch, evaluate_truth_method, execute_truth_blind_method, generate_truth_case_batch
from empirical_lawhood.adapters.simulators.rc_ladder_response.provider import RcLadderResponseNumericalQualification, qualify_numerical_provider
from empirical_lawhood.kernel.authority import ResourceBudget
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import EvidenceCeiling, OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord, validate_sha256
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
    lineage_parent_sort_key,
)
from empirical_lawhood.runtime.capabilities import (
    CapabilityConfigRef,
    CapabilityKind,
    CapabilityManifest,
    CapabilityPermission,
    CapabilityRegistry,
)
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
    WorkerInputKind,
)
from empirical_lawhood.runtime.plans import BarrierKind, ProtocolExecutionPlan, OutputTemplate, ProtocolStepTemplate, ProtocolTemplate, ScientificStage
from empirical_lawhood.runtime.providers import (
    CapabilityOutputSemanticContract,
    CampaignRuntimeProvider,
    ExternalInputPayload,
)

from .provider import PhysicalScaleMorphismMethodFreeze, PhysicalScaleMorphismStudyConfig, PhysicalScaleMorphismSyntheticReadiness, adjudicate_synthetic_readiness


PHYSICAL_SCALE_MORPHISM_VERSION = "1.0.0"
PHYSICAL_SCALE_MORPHISM_SOURCE_ID = "source.physical-scale-morphism-synthetic-config"
PHYSICAL_SCALE_MORPHISM_SOURCE_ARTIFACT_ID = "artifact.physical-scale-morphism-synthetic-config"
PHYSICAL_SCALE_MORPHISM_CONFIG_ARTIFACT_ID = "config-artifact.physical-scale-morphism-synthetic"
PHYSICAL_SCALE_MORPHISM_ADJUDICATION_SCHEMA = 'empirical-lawhood/methods/physical-scale-morphism/synthetic-scientific-adjudication'
PHYSICAL_SCALE_MORPHISM_ADJUDICATION_SCOPE_ID = "physical-scale-morphism-synthetic-implementation-rehearsal"

TRUTH_GENERATE_KEY = "physical-scale-morphism.truth-generate"
MORPHISM_METHOD_KEY = "physical-scale-morphism.receiver-morphism-method"
TRUTH_EVALUATE_KEY = "physical-scale-morphism.truth-evaluate"
NUMERICAL_PROVIDER_KEY = "physical-scale-morphism.numerical-rc-ladder"
PHYSICAL_ARCHIVE_VERIFY_KEY = "physical-scale-morphism.physical-archive-verify"
METHOD_FREEZE_KEY = "physical-scale-morphism.method-freeze"
ADJUDICATE_KEY = "physical-scale-morphism.boundary-defect-adjudicate"

_RESOURCE = ResourceBudget(
    cpu_cores=2,
    memory_bytes=2 * 1024**3,
    gpu_devices=0,
    wall_time_seconds=300,
    source_scan_bytes=16 * 1024**2,
    output_bytes=16 * 1024**2,
)
_READ_WRITE = (
    CapabilityPermission.READ_EXTERNAL_ARTIFACTS,
    CapabilityPermission.WRITE_EXTERNAL_ARTIFACTS,
)
_DEVELOPMENT = tuple(sorted((*_READ_WRITE, CapabilityPermission.READ_DEVELOPMENT)))
_EVALUATOR = tuple(
    sorted(
        (
            *_DEVELOPMENT,
            CapabilityPermission.READ_SEALED_OUTCOMES,
            CapabilityPermission.REVEAL_OUTCOMES,
        )
    )
)


@dataclass(frozen=True, slots=True)
class _Definition:
    key: str
    kind: CapabilityKind
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    permissions: tuple[CapabilityPermission, ...]
    outcome_access: OutcomeAccess


_DEFINITIONS = (
    _Definition(
        ADJUDICATE_KEY,
        CapabilityKind.EVALUATOR,
        tuple(
            sorted(
                (
                    PhysicalScaleMorphismStudyConfig.SCHEMA,
                    RcLadderResponseNumericalQualification.SCHEMA,
                    RcLadderResponseSyntheticSourceQualification.SCHEMA,
                    PhysicalScaleMorphismTruthMethodSuiteResult.SCHEMA,
                )
            )
        ),
        tuple(sorted((PHYSICAL_SCALE_MORPHISM_ADJUDICATION_SCHEMA, PhysicalScaleMorphismSyntheticReadiness.SCHEMA))),
        _EVALUATOR,
        OutcomeAccess.EVALUATOR_REVEAL,
    ),
    _Definition(
        MORPHISM_METHOD_KEY,
        CapabilityKind.TRANSPORT_TESTER,
        tuple(sorted((PhysicalScaleMorphismStudyConfig.SCHEMA, PhysicalScaleMorphismTruthCaseBatch.SCHEMA))),
        (PhysicalScaleMorphismTruthObservationBatch.SCHEMA,),
        _DEVELOPMENT,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
    ),
    _Definition(
        METHOD_FREEZE_KEY,
        CapabilityKind.TRANSFORM,
        tuple(
            sorted(
                (
                    PhysicalScaleMorphismStudyConfig.SCHEMA,
                    PhysicalScaleMorphismTruthObservationBatch.SCHEMA,
                )
            )
        ),
        (PhysicalScaleMorphismMethodFreeze.SCHEMA,),
        _DEVELOPMENT,
        OutcomeAccess.DEVELOPMENT_VISIBLE,
    ),
    _Definition(
        NUMERICAL_PROVIDER_KEY,
        CapabilityKind.NUMERICAL_QUALIFIER,
        (PhysicalScaleMorphismStudyConfig.SCHEMA,),
        (RcLadderResponseNumericalQualification.SCHEMA,),
        _READ_WRITE,
        OutcomeAccess.OUTCOME_BLIND,
    ),
    _Definition(
        PHYSICAL_ARCHIVE_VERIFY_KEY,
        CapabilityKind.OBSERVATION_OPERATOR,
        (PhysicalScaleMorphismStudyConfig.SCHEMA,),
        (RcLadderResponseSyntheticSourceQualification.SCHEMA,),
        _READ_WRITE,
        OutcomeAccess.OUTCOME_BLIND,
    ),
    _Definition(
        TRUTH_EVALUATE_KEY,
        CapabilityKind.EVALUATOR,
        tuple(
            sorted(
                (
                    PhysicalScaleMorphismStudyConfig.SCHEMA,
                    PhysicalScaleMorphismTruthCaseBatch.SCHEMA,
                    PhysicalScaleMorphismMethodFreeze.SCHEMA,
                    PhysicalScaleMorphismTruthObservationBatch.SCHEMA,
                )
            )
        ),
        (PhysicalScaleMorphismTruthMethodSuiteResult.SCHEMA,),
        _EVALUATOR,
        OutcomeAccess.EVALUATOR_REVEAL,
    ),
    _Definition(
        TRUTH_GENERATE_KEY,
        CapabilityKind.SOURCE,
        (PhysicalScaleMorphismStudyConfig.SCHEMA,),
        (PhysicalScaleMorphismTruthCaseBatch.SCHEMA,),
        _READ_WRITE,
        OutcomeAccess.OUTCOME_BLIND,
    ),
)


def _schema_sha256(schema: str) -> str:
    return sha256(schema.encode()).hexdigest()


def physical_scale_morphism_config_ref(config: PhysicalScaleMorphismStudyConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id=config.config_id,
        config_schema=PhysicalScaleMorphismStudyConfig.SCHEMA,
        config_schema_sha256=_schema_sha256(PhysicalScaleMorphismStudyConfig.SCHEMA),
        content_sha256=config.fingerprint(),
        artifact_id=PHYSICAL_SCALE_MORPHISM_CONFIG_ARTIFACT_ID,
    )


def physical_scale_morphism_registry(*, implementation_sha256: str) -> CapabilityRegistry:
    validate_sha256(implementation_sha256, field_name="implementation_sha256")
    manifests = tuple(
        sorted(
            (
                CapabilityManifest(
                    capability_key=value.key,
                    capability_version=PHYSICAL_SCALE_MORPHISM_VERSION,
                    kind=value.kind,
                    config_schema=PhysicalScaleMorphismStudyConfig.SCHEMA,
                    config_schema_sha256=_schema_sha256(PhysicalScaleMorphismStudyConfig.SCHEMA),
                    input_schema_ids=value.inputs,
                    output_schema_ids=value.outputs,
                    permissions=value.permissions,
                    maximum_evidence_ceiling=EvidenceCeiling.NON_PROMOTABLE,
                    maximum_outcome_access=value.outcome_access,
                    resource_ceiling=_RESOURCE,
                    deterministic=True,
                    seed_required=False,
                    language_id="python",
                    runtime_id="cpython-3.11-physical-scale-morphism",
                    requires_clean_commit=False,
                    requires_active_mount=False,
                    requires_network=False,
                    conformance_check_ids=(
                        "closed-static-dispatch",
                        "no-physical-actuation",
                        "nonpromotable-synthetic-ceiling",
                        "truth-label-method-firewall",
                    ),
                    implementation_sha256=implementation_sha256,
                )
                for value in _DEFINITIONS
            ),
            key=lambda value: value.registry_id,
        )
    )
    return CapabilityRegistry(
        registry_id="physical-scale-morphism-synthetic-registry",
        capabilities=manifests,
    )


@dataclass(frozen=True, slots=True)
class _Step:
    step_id: str
    stage: ScientificStage
    capability_key: str
    dependencies: tuple[str, ...]
    outputs: tuple[tuple[str, str], ...]
    outcome_access: OutcomeAccess
    visibility: VisibilityCeiling
    barrier: BarrierKind


_STEPS = (
    _Step(
        "freeze-method",
        ScientificStage.FREEZE,
        METHOD_FREEZE_KEY,
        ("run-morphism-method",),
        (("method-freeze", PhysicalScaleMorphismMethodFreeze.SCHEMA),),
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        VisibilityCeiling.DEVELOPMENT_ONLY,
        BarrierKind.FREEZE,
    ),
    _Step(
        "adjudicate-synthetic",
        ScientificStage.EVALUATE,
        ADJUDICATE_KEY,
        ("qualify-numerical", "qualify-source-adapter", "truth-evaluate"),
        (
            ("scientific-adjudication", PHYSICAL_SCALE_MORPHISM_ADJUDICATION_SCHEMA),
            ("synthetic-readiness", PhysicalScaleMorphismSyntheticReadiness.SCHEMA),
        ),
        OutcomeAccess.EVALUATOR_REVEAL,
        VisibilityCeiling.PRIVILEGED_TRUTH,
        BarrierKind.REVEAL,
    ),
    _Step(
        "qualify-numerical",
        ScientificStage.QUALIFY,
        NUMERICAL_PROVIDER_KEY,
        (),
        (("numerical-qualification", RcLadderResponseNumericalQualification.SCHEMA),),
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        BarrierKind.NONE,
    ),
    _Step(
        "qualify-source-adapter",
        ScientificStage.QUALIFY,
        PHYSICAL_ARCHIVE_VERIFY_KEY,
        (),
        (("source-qualification", RcLadderResponseSyntheticSourceQualification.SCHEMA),),
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        BarrierKind.NONE,
    ),
    _Step(
        "run-morphism-method",
        ScientificStage.FALSIFY,
        MORPHISM_METHOD_KEY,
        ("truth-generate",),
        (("truth-observations", PhysicalScaleMorphismTruthObservationBatch.SCHEMA),),
        OutcomeAccess.DEVELOPMENT_VISIBLE,
        VisibilityCeiling.DEVELOPMENT_ONLY,
        BarrierKind.NONE,
    ),
    _Step(
        "truth-evaluate",
        ScientificStage.EVALUATE,
        TRUTH_EVALUATE_KEY,
        ("freeze-method", "run-morphism-method", "truth-generate"),
        (("truth-method-result", PhysicalScaleMorphismTruthMethodSuiteResult.SCHEMA),),
        OutcomeAccess.EVALUATOR_REVEAL,
        VisibilityCeiling.PRIVILEGED_TRUTH,
        BarrierKind.REVEAL,
    ),
    _Step(
        "truth-generate",
        ScientificStage.PREPARE,
        TRUTH_GENERATE_KEY,
        (),
        (("truth-cases", PhysicalScaleMorphismTruthCaseBatch.SCHEMA),),
        OutcomeAccess.OUTCOME_BLIND,
        VisibilityCeiling.PROSPECTIVE,
        BarrierKind.NONE,
    ),
)


def physical_scale_morphism_protocol_template(
    *, registry: CapabilityRegistry, config: PhysicalScaleMorphismStudyConfig
) -> ProtocolTemplate:
    config_ref = physical_scale_morphism_config_ref(config)
    steps = []
    for definition in _STEPS:
        manifest = registry.resolve(definition.capability_key, PHYSICAL_SCALE_MORPHISM_VERSION)
        steps.append(
            ProtocolStepTemplate(
                step_id=definition.step_id,
                stage=definition.stage,
                capability_key=definition.capability_key,
                capability_version=PHYSICAL_SCALE_MORPHISM_VERSION,
                config=config_ref,
                dependency_step_ids=definition.dependencies,
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
                            for output_id, schema in definition.outputs
                        ),
                        key=lambda value: value.output_id,
                    )
                ),
                required_permissions=manifest.permissions,
                requested_outcome_access=definition.outcome_access,
                visibility_ceiling=definition.visibility,
                resource_budget=_RESOURCE,
                resource_lock_ids=(f"physical-scale-morphism-{definition.step_id}",),
                barrier=definition.barrier,
                maximum_attempts=2,
                obligation_ids=(f"physical-scale-morphism-{definition.step_id}-contract",),
            )
        )
    return ProtocolTemplate(
        template_id="physical-scale-morphism-synthetic-protocol",
        template_version=PHYSICAL_SCALE_MORPHISM_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=False,
        requests_controller=False,
        nonactuating=True,
    )


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _one(records: tuple[CanonicalRecord, ...], kind: type[_RecordT]) -> _RecordT:
    values = {value.fingerprint(): value for value in records if isinstance(value, kind)}
    if len(values) != 1:
        raise ValueError(f"physical scale morphism task requires exactly one {kind.__name__}")
    return next(iter(values.values()))


_INPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    PhysicalScaleMorphismStudyConfig.SCHEMA: PhysicalScaleMorphismStudyConfig,
    PhysicalScaleMorphismMethodFreeze.SCHEMA: PhysicalScaleMorphismMethodFreeze,
    RcLadderResponseNumericalQualification.SCHEMA: RcLadderResponseNumericalQualification,
    RcLadderResponseSyntheticSourceQualification.SCHEMA: RcLadderResponseSyntheticSourceQualification,
    PhysicalScaleMorphismTruthCaseBatch.SCHEMA: PhysicalScaleMorphismTruthCaseBatch,
    PhysicalScaleMorphismTruthMethodSuiteResult.SCHEMA: PhysicalScaleMorphismTruthMethodSuiteResult,
    PhysicalScaleMorphismTruthObservationBatch.SCHEMA: PhysicalScaleMorphismTruthObservationBatch,
}
_OUTPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    PhysicalScaleMorphismMethodFreeze.SCHEMA: PhysicalScaleMorphismMethodFreeze,
    RcLadderResponseNumericalQualification.SCHEMA: RcLadderResponseNumericalQualification,
    PhysicalScaleMorphismSyntheticReadiness.SCHEMA: PhysicalScaleMorphismSyntheticReadiness,
    RcLadderResponseSyntheticSourceQualification.SCHEMA: RcLadderResponseSyntheticSourceQualification,
    PhysicalScaleMorphismTruthCaseBatch.SCHEMA: PhysicalScaleMorphismTruthCaseBatch,
    PhysicalScaleMorphismTruthMethodSuiteResult.SCHEMA: PhysicalScaleMorphismTruthMethodSuiteResult,
    PhysicalScaleMorphismTruthObservationBatch.SCHEMA: PhysicalScaleMorphismTruthObservationBatch,
}


class PhysicalScaleMorphismRunner:
    def __init__(self, manifest: CapabilityManifest, config: PhysicalScaleMorphismStudyConfig) -> None:
        self.manifest = manifest
        self.config = config
        self.execution_count = 0

    def _read(self, context: TaskContext) -> tuple[CanonicalRecord, ...]:
        records = []
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError("physical scale morphism input size differs")
            if port.kind is WorkerInputKind.EXTERNAL and payload != self.config.canonical_bytes():
                raise ValueError("physical scale morphism external input differs from the registered config")
            try:
                kind = _INPUT_TYPES[port.payload_schema]
            except KeyError as error:
                raise ValueError("physical scale morphism task received an unknown input schema") from error
            records.append(decode_canonical_bytes(payload, kind, maximum_bytes=port.size_bytes))
        return tuple(records)

    @staticmethod
    def _scientific_adjudication(
        context: TaskContext, readiness: PhysicalScaleMorphismSyntheticReadiness
    ) -> ScientificAdjudicationRecord:
        adjudication_context = context.scientific_adjudication_context
        if adjudication_context is None:
            raise ValueError("physical scale morphism terminal evaluator lacks adjudication context")
        output_ids = tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
            )
        )
        if len(output_ids) != len(context.output_ports):
            raise ValueError("physical scale morphism adjudication output lacks logical identity")
        return ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
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
                ScientificStatus.SUPPORTED
                if readiness.runtime_rehearsal_qualified
                else ScientificStatus.NOT_SUPPORTED
            ),
            admission_status=AdmissionStatus.NOT_EVALUATED,
            reason_codes=readiness.reason_codes,
            fixture_scope_id=PHYSICAL_SCALE_MORPHISM_ADJUDICATION_SCOPE_ID,
            plumbing_only=True,
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        records = self._read(context)
        config = _one(records, PhysicalScaleMorphismStudyConfig)
        if config != self.config or context.config.content_sha256 != config.fingerprint():
            raise ValueError("physical scale morphism task config differs from its static provider")
        key = self.manifest.capability_key
        outputs: dict[str, CanonicalRecord | bytes]
        if key == TRUTH_GENERATE_KEY:
            outputs = {"truth-cases": generate_truth_case_batch(config.truth_suite)}
        elif key == MORPHISM_METHOD_KEY:
            outputs = {
                "truth-observations": execute_truth_blind_method(_one(records, PhysicalScaleMorphismTruthCaseBatch))
            }
        elif key == METHOD_FREEZE_KEY:
            observations = _one(records, PhysicalScaleMorphismTruthObservationBatch)
            outputs = {
                "method-freeze": PhysicalScaleMorphismMethodFreeze(
                    freeze_id="freeze.physical-scale-morphism-truth-method",
                    case_batch_id=observations.case_batch_id,
                    observation_batch_sha256=observations.fingerprint(),
                    evaluation_outcome_count_at_freeze=0,
                    physical_outcome_count_at_freeze=0,
                    frozen=True,
                )
            }
        elif key == TRUTH_EVALUATE_KEY:
            observations = _one(records, PhysicalScaleMorphismTruthObservationBatch)
            method_freeze = _one(records, PhysicalScaleMorphismMethodFreeze)
            if (
                method_freeze.case_batch_id != observations.case_batch_id
                or method_freeze.observation_batch_sha256 != observations.fingerprint()
            ):
                raise ValueError("physical scale morphism evaluation input differs from the method freeze")
            outputs = {
                "truth-method-result": evaluate_truth_method(
                    config=config.truth_suite,
                    cases=_one(records, PhysicalScaleMorphismTruthCaseBatch),
                    observations=observations,
                )
            }
        elif key == NUMERICAL_PROVIDER_KEY:
            outputs = {"numerical-qualification": qualify_numerical_provider()}
        elif key == PHYSICAL_ARCHIVE_VERIFY_KEY:
            outputs = {"source-qualification": qualify_synthetic_source_adapter()}
        elif key == ADJUDICATE_KEY:
            readiness = adjudicate_synthetic_readiness(
                config=config,
                truth=_one(records, PhysicalScaleMorphismTruthMethodSuiteResult),
                numerical=_one(records, RcLadderResponseNumericalQualification),
                source=_one(records, RcLadderResponseSyntheticSourceQualification),
            )
            outputs = {
                "synthetic-readiness": readiness,
                "scientific-adjudication": encode_scientific_adjudication(
                    self._scientific_adjudication(context, readiness),
                    payload_schema=PHYSICAL_SCALE_MORPHISM_ADJUDICATION_SCHEMA,
                ),
            }
        else:  # pragma: no cover - the registry is closed.
            raise ValueError("unknown physical scale morphism capability")
        return RunnerResult(
            outputs=tuple(
                TaskOutputPayload(
                    output_id=port.output_id,
                    payload=(
                        value
                        if isinstance(
                            value := outputs[port.output_id.removeprefix(f"{context.task_id}.")],
                            bytes,
                        )
                        else value.canonical_bytes()
                    ),
                )
                for port in context.output_ports
            ),
            checks=(
                ReceiptCheck("actual-computation-completed", True, ()),
                ReceiptCheck("no-physical-action-or-source-write", True, ()),
                ReceiptCheck("truth-label-method-firewall-preserved", True, ()),
            ),
        )


class PhysicalScaleMorphismCampaignRuntimeProvider(CampaignRuntimeProvider):
    issued_source_schema_ids: tuple[str, ...] = ()

    def __init__(self, *, registry: CapabilityRegistry, config: PhysicalScaleMorphismStudyConfig) -> None:
        expected = physical_scale_morphism_registry(
            implementation_sha256=registry.capabilities[0].implementation_sha256
        )
        if registry != expected:
            raise ValueError("physical scale morphism runtime registry differs")
        self.registry = registry
        self.config = config
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = len(registry.capabilities)
        self._runners: tuple[PhysicalScaleMorphismRunner, ...] = ()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry != self.registry or source_records:
            raise ValueError("physical scale morphism runner binding differs")
        self._runners = tuple(
            PhysicalScaleMorphismRunner(manifest, self.config) for manifest in registry.capabilities
        )
        return self._runners

    def external_inputs(
        self,
        plan: ProtocolExecutionPlan,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("physical scale morphism external-input binding differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        allowed = {
            PHYSICAL_SCALE_MORPHISM_CONFIG_ARTIFACT_ID,
            PHYSICAL_SCALE_MORPHISM_SOURCE_ARTIFACT_ID,
            self.config.config_id,
        }
        if not set(specs).issubset(allowed):
            raise ValueError("physical scale morphism plan requests an unknown external input")
        identity = ObjectIdentity.from_record(self.config.config_id, self.config)
        payload = self.config.canonical_bytes()
        values = []
        for artifact_id, spec in sorted(specs.items()):
            parent = ArtifactLineageParent(
                identity=identity,
                visibility_ceiling=(
                    spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                ),
                outcome_access=spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND,
            )
            parents = [parent]
            if spec.identity_scope_sha256 is not None:
                parents.append(
                    ArtifactLineageParent(
                        identity=ObjectIdentity(
                            object_id=spec.input_id,
                            object_schema='empirical-lawhood/runtime/external-input-scope',
                            object_version="1.0.0",
                            object_fingerprint=spec.identity_scope_sha256,
                        ),
                        visibility_ceiling=parent.visibility_ceiling,
                        outcome_access=parent.outcome_access,
                    )
                )
            lineage = tuple(sorted(parents, key=lineage_parent_sort_key))
            values.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=artifact_id,
                    payload_schema=PhysicalScaleMorphismStudyConfig.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/json",
                    payload=payload,
                    visibility_ceiling=parent.visibility_ceiling,
                    outcome_access=parent.outcome_access,
                    parent_visibility_ceilings=tuple(
                        sorted(
                            (value.visibility_ceiling for value in lineage),
                            key=lambda value: value.value,
                        )
                    ),
                    lineage_parents=lineage,
                    logical_content_sha256=spec.expected_content_sha256,
                )
            )
        return tuple(values)

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        if registry != self.registry:
            raise ValueError("physical scale morphism semantic registry differs")
        contracts = []
        for manifest in registry.capabilities:
            for schema in manifest.output_schema_ids:
                record_type = (
                    ScientificAdjudicationRecord
                    if schema == PHYSICAL_SCALE_MORPHISM_ADJUDICATION_SCHEMA
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
        if registry != self.registry:
            raise ValueError("physical scale morphism adjudication registry differs")
        return ScientificAdjudicationOutputContract(
            capability_key=ADJUDICATE_KEY,
            capability_version=PHYSICAL_SCALE_MORPHISM_VERSION,
            output_id="adjudicate-synthetic.scientific-adjudication",
            payload_schema=PHYSICAL_SCALE_MORPHISM_ADJUDICATION_SCHEMA,
            maximum_bytes=128 * 1024,
            fixture_scope_id=PHYSICAL_SCALE_MORPHISM_ADJUDICATION_SCOPE_ID,
            plumbing_only=True,
        )


@dataclass(frozen=True, slots=True)
class PhysicalScaleMorphismCapabilityAdapter:
    manifest: CapabilityManifest
    config: PhysicalScaleMorphismStudyConfig

    @property
    def provider_key(self) -> str:
        return self.manifest.capability_key

    @property
    def provider_version(self) -> str:
        return self.manifest.capability_version

    def validate_config(self, payload: bytes, *, expected_schema: str) -> None:
        if expected_schema != self.manifest.config_schema:
            raise ValueError("physical scale morphism config schema differs from its manifest")
        decoded = decode_canonical_bytes(
            payload,
            PhysicalScaleMorphismStudyConfig,
            maximum_bytes=16 * 1024**2,
        )
        if decoded != self.config:
            raise ValueError("physical scale morphism config differs from its static provider")


__all__ = [
    "ADJUDICATE_KEY",
    "PHYSICAL_SCALE_MORPHISM_ADJUDICATION_SCHEMA",
    "PHYSICAL_SCALE_MORPHISM_CONFIG_ARTIFACT_ID",
    "PHYSICAL_SCALE_MORPHISM_SOURCE_ARTIFACT_ID",
    "PHYSICAL_SCALE_MORPHISM_SOURCE_ID",
    "PHYSICAL_SCALE_MORPHISM_VERSION",
    'PhysicalScaleMorphismCampaignRuntimeProvider',
    'PhysicalScaleMorphismCapabilityAdapter',
    "METHOD_FREEZE_KEY",
    "MORPHISM_METHOD_KEY",
    "NUMERICAL_PROVIDER_KEY",
    "PHYSICAL_ARCHIVE_VERIFY_KEY",
    "TRUTH_EVALUATE_KEY",
    "TRUTH_GENERATE_KEY",
    'physical_scale_morphism_config_ref',
    'physical_scale_morphism_protocol_template',
    'physical_scale_morphism_registry',
]

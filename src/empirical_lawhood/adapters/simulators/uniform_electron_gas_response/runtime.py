"""Receipt-first truth-known conformance protocol, runners and runtime provider for transverse receiver screen."""

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

from .contracts import ADMISSION_CAPABILITY_KEY, ADJUDICATION_SCHEMA, CAPABILITY_VERSION, CONFIG_SCHEMA, CONFIG_VERSION, EVALUATOR_CAPABILITY_KEY, METHOD_CAPABILITY_KEY, SOURCE_CAPABILITY_KEY, MethodConformanceResult, SourceStopResult, UniformElectronGasTransverseScreenConfig, TruthInputBatch, TruthLawBatch, TruthOracleBatch, TruthScreenBatch, decode_config_bytes
from .reference_world import adjudicate_truth_batch, build_truth_batches, evaluate_truth_batches, identify_truth_batch
from .system import task_budget


CONFIG_ARTIFACT_ID = "config-artifact.uniform-electron-gas-transverse-screen"
MEDIUM_ARTIFACT_ID = "prepared-medium.uniform-electron-gas-transverse-screen"
CONFIG_SOURCE_ID = "source.uniform-electron-gas-transverse-screen-config"
ADJUDICATION_SCOPE_ID = "uniform-electron-gas-transverse-screen-conformance"

TRUTH_INPUT_STEP = "truth-input-source"
TRUTH_ORACLE_STEP = "truth-oracle-custody"
METHOD_STEP = "method-identify"
ADMISSION_STEP = "admission-intersect"
EVALUATOR_STEP = "evaluator-reveal"


def config_ref(config: UniformElectronGasTransverseScreenConfig) -> CapabilityConfigRef:
    return CapabilityConfigRef(
        config_id="config.uniform-electron-gas-transverse-screen",
        config_schema=CONFIG_SCHEMA,
        config_schema_sha256=sha256(CONFIG_SCHEMA.encode()).hexdigest(),
        content_sha256=config.payload_sha256,
        artifact_id=CONFIG_ARTIFACT_ID,
    )


def uniform_electron_gas_transverse_screen_protocol(*, registry: CapabilityRegistry, config: UniformElectronGasTransverseScreenConfig) -> ProtocolTemplate:
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
                            payload_schema=payload_schema,
                            profile=ArtifactProfile.CANONICAL_JSON,
                            media_type="application/json",
                            filename_suffix=".json",
                        )
                        for output_id, payload_schema in outputs
                    ),
                    key=lambda value: value.output_id,
                )
            ),
            required_permissions=manifest.permissions,
            requested_outcome_access=outcome_access,
            visibility_ceiling=visibility,
            resource_budget=task_budget(config),
            resource_lock_ids=(f"uniform-electron-gas-transverse-screen-{step_id}",),
            barrier=barrier,
            maximum_attempts=2,
            obligation_ids=tuple(sorted(obligations)),
        )

    steps = (
        step(
            step_id=ADMISSION_STEP,
            stage=ScientificStage.ADMISSION,
            capability_key=ADMISSION_CAPABILITY_KEY,
            dependencies=(METHOD_STEP, TRUTH_INPUT_STEP),
            outputs=(("truth-screens", TruthScreenBatch.SCHEMA),),
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.FREEZE,
            obligations=("transverse-screen-admission-nine-gate-contract",),
        ),
        step(
            step_id=EVALUATOR_STEP,
            stage=ScientificStage.EVALUATE,
            capability_key=EVALUATOR_CAPABILITY_KEY,
            dependencies=(ADMISSION_STEP, TRUTH_ORACLE_STEP),
            outputs=(
                ("method-conformance", MethodConformanceResult.SCHEMA),
                ("scientific-adjudication", ADJUDICATION_SCHEMA),
                ("source-stop", SourceStopResult.SCHEMA),
            ),
            outcome_access=OutcomeAccess.EVALUATOR_REVEAL,
            visibility=VisibilityCeiling.PRIVILEGED_TRUTH,
            barrier=BarrierKind.REVEAL,
            obligations=(
                *(f"formal-obligation.{gap_id}" for gap_id in config.reference_conformance_gaps),
                "transverse-screen-evaluator-exact-classification-contract",
                "transverse-screen-source-stop-nonattempt-contract",
            ),
        ),
        step(
            step_id=METHOD_STEP,
            stage=ScientificStage.DEVELOP,
            capability_key=METHOD_CAPABILITY_KEY,
            dependencies=(TRUTH_INPUT_STEP,),
            outputs=(("truth-laws", TruthLawBatch.SCHEMA),),
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.FREEZE,
            obligations=("transverse-screen-method-measurement-through-law-freeze-contract",),
        ),
        step(
            step_id=TRUTH_INPUT_STEP,
            stage=ScientificStage.ACQUIRE,
            capability_key=SOURCE_CAPABILITY_KEY,
            dependencies=(),
            outputs=(("truth-inputs", TruthInputBatch.SCHEMA),),
            outcome_access=OutcomeAccess.DEVELOPMENT_VISIBLE,
            visibility=VisibilityCeiling.DEVELOPMENT_ONLY,
            barrier=BarrierKind.NONE,
            obligations=("transverse-screen-truth-input-roster-contract",),
        ),
        step(
            step_id=TRUTH_ORACLE_STEP,
            stage=ScientificStage.ACQUIRE,
            capability_key=SOURCE_CAPABILITY_KEY,
            dependencies=(),
            outputs=(("truth-oracles", TruthOracleBatch.SCHEMA),),
            outcome_access=OutcomeAccess.PRIVILEGED_TRUTH,
            visibility=VisibilityCeiling.PRIVILEGED_TRUTH,
            barrier=BarrierKind.NONE,
            obligations=("transverse-screen-privileged-oracle-custody-contract",),
        ),
    )
    return ProtocolTemplate(
        template_id="protocol.uniform-electron-gas-transverse-screen-conformance",
        template_version=CAPABILITY_VERSION,
        steps=tuple(sorted(steps, key=lambda value: value.step_id)),
        requires_model_set=True,
        requests_controller=False,
        nonactuating=True,
    )


_RecordT = TypeVar("_RecordT", bound=CanonicalRecord)


def _one(records: tuple[CanonicalRecord, ...], record_type: type[_RecordT]) -> _RecordT:
    values = {value.fingerprint(): value for value in records if isinstance(value, record_type)}
    if len(values) != 1:
        raise ValueError(f"uniform electron gas task requires exactly one {record_type.__name__}")
    return next(iter(values.values()))


_INPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    TruthInputBatch.SCHEMA: TruthInputBatch,
    TruthLawBatch.SCHEMA: TruthLawBatch,
    TruthOracleBatch.SCHEMA: TruthOracleBatch,
    TruthScreenBatch.SCHEMA: TruthScreenBatch,
}


class UniformElectronGasTransverseScreenRunner:
    """Closed runner whose truth-blind tasks never decode privileged oracles."""

    def __init__(self, manifest: CapabilityManifest, config: UniformElectronGasTransverseScreenConfig, payload: bytes) -> None:
        if decode_config_bytes(payload) != config:
            raise ValueError("uniform electron gas runner config differs from frozen bytes")
        self.manifest = manifest
        self.config = config
        self.payload = payload
        self.execution_count = 0

    def _read(self, context: TaskContext) -> tuple[CanonicalRecord, ...]:
        records: list[CanonicalRecord] = []
        observed_config = False
        for port in context.input_ports:
            payload = port.read(port.size_bytes + 1)
            if len(payload) != port.size_bytes:
                raise ValueError("uniform electron gas task input size differs")
            if port.payload_schema == CONFIG_SCHEMA:
                if payload != self.payload or decode_config_bytes(payload) != self.config:
                    raise ValueError("uniform electron gas task config materialization differs")
                observed_config = True
                continue
            try:
                record_type = _INPUT_TYPES[port.payload_schema]
            except KeyError as error:
                raise ValueError("uniform electron gas task received an unknown input schema") from error
            records.append(
                decode_canonical_bytes(payload, record_type, maximum_bytes=port.size_bytes)
            )
        if not observed_config or context.config.content_sha256 != self.config.payload_sha256:
            raise ValueError("uniform electron gas task lacks its exact frozen config")
        return tuple(records)

    @staticmethod
    def _source_stop(config: UniformElectronGasTransverseScreenConfig, conformance: MethodConformanceResult) -> SourceStopResult:
        conformance_passed = conformance.passed
        return SourceStopResult(
            result_id="uniform-electron-gas-transverse-screen-source-stop",
            config_sha256=config.payload_sha256,
            source_dispositions=(
                ("paper_rpa", "CLOSURE_ONLY_SOURCE"),
                ("target_gauge_closed_roster", "EMPTY"),
                ("vertex_reference", "CLOSURE_ONLY_SOURCE"),
            ),
            task_dispositions=(
                (
                    "conformance-truth-known",
                    ("TRANSVERSE_SCREEN_METHOD_CONFORMANCE_PASS" if conformance_passed else "TRANSVERSE_SCREEN_METHOD_CONFORMANCE_FAIL"),
                ),
                (
                    "development-law-commitment",
                    (
                        "NOT_ATTEMPTED_PREREQUISITE_TARGET_SOURCE"
                        if conformance_passed
                        else "NOT_ATTEMPTED_PREREQUISITE_TRUTH_KNOWN_CONFORMANCE"
                    ),
                ),
                (
                    "evaluation-target-reveal",
                    (
                        "NOT_ATTEMPTED_PREREQUISITE_TARGET_SOURCE"
                        if conformance_passed
                        else "NOT_ATTEMPTED_PREREQUISITE_TRUTH_KNOWN_CONFORMANCE"
                    ),
                ),
                (
                    "qualification-target-source",
                    (
                        "NOT_ATTEMPTED_PREREQUISITE_TARGET_SOURCE"
                        if conformance_passed
                        else "NOT_ATTEMPTED_PREREQUISITE_TRUTH_KNOWN_CONFORMANCE"
                    ),
                ),
                (
                    "supplementary-closure-diagnostic",
                    (
                        "NOT_ATTEMPTED_PREREQUISITE_EXACT_CLOSURE_BINDING"
                        if conformance_passed
                        else "NOT_ATTEMPTED_PREREQUISITE_TRUTH_KNOWN_CONFORMANCE"
                    ),
                ),
                ("terminal-closeout", "READY_FOR_IMMUTABLE_CLOSEOUT"),
            ),
            target_contact_count=0,
            status=(
                "TRANSVERSE_SCREEN_SOURCE_NOT_READY_TRANSVERSE_CURRENT_OPERAND_REQUIRED"
                if conformance_passed
                else "TRANSVERSE_SCREEN_METHOD_CONFORMANCE_FAILED"
            ),
        )

    @staticmethod
    def _scientific_adjudication(
        context: TaskContext, conformance: MethodConformanceResult
    ) -> ScientificAdjudicationRecord:
        adjudication_context = context.scientific_adjudication_context
        if adjudication_context is None:
            raise ValueError("uniform electron gas evaluator lacks scientific adjudication context")
        output_ids = tuple(
            sorted(
                port.logical_artifact_id
                for port in context.output_ports
                if port.logical_artifact_id is not None
            )
        )
        if len(output_ids) != len(context.output_ports):
            raise ValueError("uniform electron gas evaluator output lacks logical identity")
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
                ScientificStatus.SUPPORTED if conformance.passed else ScientificStatus.NOT_SUPPORTED
            ),
            admission_status=AdmissionStatus.NOT_EVALUATED,
            reason_codes=(
                (
                    "CONFORMANCE_TRUTH_KNOWN_METHOD_PASSED",
                    "TARGET_SOURCE_TRANSVERSE_CURRENT_OPERAND_REQUIRED",
                    "TRUTH_KNOWN_NONPROMOTABLE",
                )
                if conformance.passed
                else (
                    "CONFORMANCE_TRUTH_KNOWN_METHOD_FAILED",
                    "TARGET_EXECUTION_BLOCKED_BY_TRUTH_KNOWN_CONFORMANCE",
                    "TRUTH_KNOWN_NONPROMOTABLE",
                )
            ),
            fixture_scope_id=adjudication_context.fixture_scope_id,
            plumbing_only=adjudication_context.plumbing_only,
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        self.execution_count += 1
        records = self._read(context)
        outputs: dict[str, CanonicalRecord | bytes]
        if context.task_id == TRUTH_INPUT_STEP:
            inputs, _oracles = build_truth_batches(self.config)
            outputs = {"truth-inputs": inputs}
        elif context.task_id == TRUTH_ORACLE_STEP:
            _inputs, oracles = build_truth_batches(self.config)
            outputs = {"truth-oracles": oracles}
        elif context.task_id == METHOD_STEP:
            outputs = {
                "truth-laws": identify_truth_batch(_one(records, TruthInputBatch), self.config)
            }
        elif context.task_id == ADMISSION_STEP:
            outputs = {
                "truth-screens": adjudicate_truth_batch(
                    _one(records, TruthInputBatch),
                    _one(records, TruthLawBatch),
                    self.config,
                )
            }
        elif context.task_id == EVALUATOR_STEP:
            conformance = evaluate_truth_batches(
                _one(records, TruthScreenBatch),
                _one(records, TruthOracleBatch),
                self.config,
            )
            outputs = {
                "method-conformance": conformance,
                "scientific-adjudication": encode_scientific_adjudication(
                    self._scientific_adjudication(context, conformance),
                    payload_schema=ADJUDICATION_SCHEMA,
                ),
                "source-stop": self._source_stop(self.config, conformance),
            }
        else:
            raise ValueError("unknown uniform electron gas task identity")

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
                ReceiptCheck("exact-config-and-inputs-decoded", True, ()),
                ReceiptCheck("no-target-source-contact", True, ()),
                ReceiptCheck("privileged-oracle-edge-separated", True, ()),
            ),
        )


_OUTPUT_TYPES: dict[str, type[CanonicalRecord]] = {
    MethodConformanceResult.SCHEMA: MethodConformanceResult,
    SourceStopResult.SCHEMA: SourceStopResult,
    TruthInputBatch.SCHEMA: TruthInputBatch,
    TruthLawBatch.SCHEMA: TruthLawBatch,
    TruthOracleBatch.SCHEMA: TruthOracleBatch,
    TruthScreenBatch.SCHEMA: TruthScreenBatch,
}


class UniformElectronGasTransverseScreenRuntimeProvider(CampaignRuntimeProvider):
    """Bind the exact registry, frozen config and raw config materialization."""

    issued_source_schema_ids: tuple[str, ...] = (StudyOperationAuthority.SCHEMA,)

    def __init__(
        self,
        *,
        registry: CapabilityRegistry,
        config: UniformElectronGasTransverseScreenConfig,
        config_payload: bytes,
    ) -> None:
        if any(
            manifest.implementation_sha256 != registry.capabilities[0].implementation_sha256
            for manifest in registry.capabilities
        ):
            raise ValueError("uniform electron gas registry mixes implementation identities")
        if decode_config_bytes(config_payload) != config:
            raise ValueError("uniform electron gas provider config differs from frozen bytes")
        self.registry = registry
        self.config = config
        self.config_payload = config_payload
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = len(registry.capabilities)
        self._runners: tuple[UniformElectronGasTransverseScreenRunner, ...] = ()

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        del source_records
        if registry != self.registry:
            raise ValueError("uniform electron gas runner registry differs")
        self._runners = tuple(
            UniformElectronGasTransverseScreenRunner(manifest, self.config, self.config_payload)
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
            raise ValueError("uniform electron gas execution plan registry differs")
        specs = {
            value.logical_artifact_id: value
            for task in plan.tasks
            for value in task.external_inputs
        }
        expected_ids = {CONFIG_ARTIFACT_ID, MEDIUM_ARTIFACT_ID}
        unknown = set(specs) - expected_ids
        if unknown:
            raise ValueError(f"uniform electron gas plan requests unknown external inputs: {sorted(unknown)}")
        if set(specs) != expected_ids:
            raise ValueError("uniform electron gas plan omits an exact config or medium materialization")

        def payload(logical_artifact_id: str) -> ExternalInputPayload:
            spec = specs[logical_artifact_id]
            parent = ArtifactLineageParent(
                identity=ObjectIdentity(
                    object_id=logical_artifact_id,
                    object_schema=CONFIG_SCHEMA,
                    object_version=CONFIG_VERSION,
                    object_fingerprint=self.config.payload_sha256,
                ),
                visibility_ceiling=(
                    spec.expected_visibility_ceiling or VisibilityCeiling.PROSPECTIVE
                ),
                outcome_access=(spec.expected_outcome_access or OutcomeAccess.OUTCOME_BLIND),
            )
            return ExternalInputPayload.from_bytes(
                logical_artifact_id=logical_artifact_id,
                payload_schema=CONFIG_SCHEMA,
                profile=ArtifactProfile.TEXT_PARAMETERS,
                media_type="application/json",
                payload=self.config_payload,
                visibility_ceiling=parent.visibility_ceiling,
                outcome_access=parent.outcome_access,
                parent_visibility_ceilings=(parent.visibility_ceiling,),
                lineage_parents=(parent,),
                logical_content_sha256=self.config.payload_sha256,
            )

        return (
            payload(CONFIG_ARTIFACT_ID),
            payload(MEDIUM_ARTIFACT_ID),
        )

    def output_semantic_contracts(
        self,
        registry: CapabilityRegistry,
        execution_plan: ProtocolExecutionPlan | None = None,
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        del execution_plan
        if registry != self.registry:
            raise ValueError("uniform electron gas semantic registry differs")
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
            raise ValueError("uniform electron gas adjudication registry differs")
        return ScientificAdjudicationOutputContract(
            capability_key=EVALUATOR_CAPABILITY_KEY,
            capability_version=CAPABILITY_VERSION,
            output_id=f"{EVALUATOR_STEP}.scientific-adjudication",
            payload_schema=ADJUDICATION_SCHEMA,
            maximum_bytes=128 * 1024,
            fixture_scope_id=ADJUDICATION_SCOPE_ID,
            plumbing_only=True,
        )


__all__ = [
    "ADJUDICATION_SCOPE_ID",
    "ADMISSION_STEP",
    "CONFIG_ARTIFACT_ID",
    "CONFIG_SOURCE_ID",
    "UniformElectronGasTransverseScreenRunner",
    "UniformElectronGasTransverseScreenRuntimeProvider",
    "EVALUATOR_STEP",
    "METHOD_STEP",
    "TRUTH_INPUT_STEP",
    "TRUTH_ORACLE_STEP",
    "config_ref",
    "uniform_electron_gas_transverse_screen_protocol",
]

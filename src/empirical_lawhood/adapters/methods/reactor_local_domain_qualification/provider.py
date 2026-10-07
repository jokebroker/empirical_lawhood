"""Public-runner local calibration, qualification and receipt-bound readout."""

from dataclasses import fields

from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader, validate_dependency
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.kernel.status import ScientificStatus, AdmissionStatus
from empirical_lawhood.runtime.adjudication import (
    ScientificAdjudicationRecord,
    ScientificAdjudicationOutputContract,
    AdjudicationEvaluability,
)
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
    ArtifactManifest,
    CanonicalTaskReceipt,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
    WorkerInputKind,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)
from .config import LocalQualificationDesign, PREFIX, ROOTS
from .records import LocalRootEvidence
from .scoring import LocalCalibration, LocalQualificationOperands, calibration, root_scores, qualification_cells
from .terminal import LocalQualificationResult, qualify_local_cell
from .extension_bundle import CAPABILITY
from .science import local_system

TASKS = ("local.calibration", "local.qualification", "local.adjudication")


class LocalMethodRunner:
    manifest = CAPABILITY

    def __init__(
        self,
        config: LocalQualificationDesign,
        custody: DependencyCustodyReader,
        publisher: CandidatePayloadPublisher,
        reader: CandidatePayloadReader,
        limits: EmpiricalResourceGuard,
    ) -> None:
        self.config, self.custody, self.publisher, self.reader, self.limits = (
            config,
            custody,
            publisher,
            reader,
            limits,
        )

    def execute(self, context: TaskContext) -> RunnerResult:
        with self.limits.task(context.task_id):
            return self._execute(context)

    def _execute(self, context: TaskContext) -> RunnerResult:
        if (
            context.task_id not in TASKS
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != CAPABILITY.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or len(context.output_ports) != 1
        ):
            raise ValueError("local method task/config/permissions/output differs")
        designs = tuple(p for p in context.input_ports if p.payload_schema == self.config.SCHEMA)
        if (
            len(designs) != 1
            or decode_canonical_bytes(
                designs[0].read(), LocalQualificationDesign, maximum_bytes=256 * 1024
            )
            != self.config
        ):
            raise ValueError("local method frozen recipe differs")
        roots: dict[str, LocalRootEvidence] = {}
        custody: dict[str, tuple[ArtifactManifest, CanonicalTaskReceipt]] = {}
        calibrated = None
        terminal = None
        for port in context.input_ports:
            if port in designs:
                continue
            if port.kind is not WorkerInputKind.DEPENDENCY:
                raise ValueError("local scientific input lacks dependency custody")
            manifest, receipt = self.custody.read_dependency(context, port.binding)
            validate_dependency(port.binding, manifest, receipt)
            if port.payload_schema == LocalRootEvidence.SCHEMA:
                record = decode_canonical_bytes(
                    port.read(), LocalRootEvidence, maximum_bytes=128 * 1024**2
                )
                if (
                    record.root in roots
                    or receipt.task_id != f"local.{record.root}"
                    or record.recipe
                    != ObjectIdentity.from_record(self.config.config_id, self.config)
                ):
                    raise ValueError("local native root/recipe was substituted")
                roots[record.root] = record
                custody[record.root] = (manifest, receipt)
                parent: CanonicalRecord = record
            elif port.payload_schema == LocalCalibration.SCHEMA and calibrated is None:
                calibrated = decode_canonical_bytes(
                    port.read(), LocalCalibration, maximum_bytes=4 * 1024**2
                )
                if receipt.task_id != "local.calibration":
                    raise ValueError("local calibration task substituted")
                parent = calibrated
            elif port.payload_schema == LocalQualificationResult.SCHEMA and terminal is None:
                terminal = decode_canonical_bytes(
                    port.read(), LocalQualificationResult, maximum_bytes=128 * 1024**2
                )
                if receipt.task_id != "local.qualification":
                    raise ValueError("local qualification task substituted")
                parent = terminal
            else:
                raise ValueError("local method input schema or multiplicity differs")
            if parent.fingerprint() != manifest.logical.content_sha256:
                raise ValueError("local scientific payload differs from its authenticated receipt")
        output: CanonicalRecord
        if context.task_id == "local.calibration":
            expected = tuple(r for r, role, _, _ in ROOTS if role == "calibration")
            if set(roots) != set(expected) or calibrated is not None or terminal is not None:
                raise ValueError("local calibration requires its exact 32 assigned roots")
            output = calibration(self.config, tuple(roots[r] for r in expected))
        elif context.task_id == "local.qualification":
            expected = tuple(r for r, _, _, _ in ROOTS)
            if set(roots) != set(expected) or calibrated is None or terminal is not None:
                raise ValueError(
                    "local qualification requires calibration and all 64 assigned roots"
                )
            if tuple(i.object_fingerprint for i in calibrated.evidence) != tuple(
                roots[r].fingerprint() for r in expected[:32]
            ):
                raise ValueError("local qualification changed calibrated native evidence")
            scores = tuple(s for r in expected[32:] for s in root_scores(self.config, roots[r]))
            cells = qualification_cells(self.config, calibrated, scores)
            operands = LocalQualificationOperands(
                ObjectIdentity.from_record(self.config.config_id, self.config),
                ObjectIdentity.from_record(f"{PREFIX}.calibration", calibrated),
                tuple(ObjectIdentity.from_record(r, roots[r]) for r in expected),
                scores,
                cells,
            )
            design_artifact = ArtifactIdentity(
                self.config.config_id,
                "local-qualification-design",
                self.config.SCHEMA,
                self.config.fingerprint(),
                "application/json",
                len(self.config.canonical_bytes()),
            )
            laws = tuple(
                qualify_local_cell(
                    self.config,
                    calibrated,
                    cell,
                    operands,
                    tuple(custody[r][0] for r in expected),
                    tuple(custody[r][1] for r in expected),
                    CAPABILITY,
                    self.publisher,
                    self.reader,
                    design_artifact,
                )
                for cell in cells
            )
            output = LocalQualificationResult(operands, laws)
        else:
            if (
                terminal is None
                or roots
                or calibrated is not None
                or terminal.operands.recipe
                != ObjectIdentity.from_record(self.config.config_id, self.config)
            ):
                raise ValueError("local adjudication requires its exact qualified collection")
            output = self._adjudicate(context, terminal)
        if context.output_ports[0].payload_schema != output.SCHEMA:
            raise ValueError("local method output schema differs")
        return RunnerResult(
            (TaskOutputPayload(context.output_ports[0].output_id, output.canonical_bytes()),),
            (ReceiptCheck("local-law-existing-owner-and-assigned-census", True, ()),),
        )

    @staticmethod
    def _adjudicate(
        context: TaskContext, terminal: LocalQualificationResult
    ) -> ScientificAdjudicationRecord:
        a = context.scientific_adjudication_context
        system = local_system()
        if (
            a is None
            or a.relation
            != ObjectIdentity.from_record(system.relation.relation_id, system.relation)
            or a.independent_unit_id != system.independent_unit.unit_id
        ):
            raise ValueError("local readout lacks its compiled scientific context")
        statuses = {r.qualification.scientific_status for r in terminal.laws}
        status = (
            next(iter(statuses))
            if len(statuses) == 1
            else ScientificStatus.PARTIAL
            if ScientificStatus.SUPPORTED in statuses
            else ScientificStatus.NOT_SUPPORTED
            if ScientificStatus.NOT_SUPPORTED in statuses
            else ScientificStatus.UNEVALUABLE
        )
        return ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=a.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=tuple(
                sorted(
                    p.logical_artifact_id
                    for p in context.output_ports
                    if p.logical_artifact_id is not None
                )
            ),
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=a.evidence_world_id,
            evidence_world_kind=a.evidence_world_kind,
            relation=a.relation,
            independent_unit_id=a.independent_unit_id,
            information_cutoffs=a.information_cutoffs,
            visibility_ceiling=a.visibility_ceiling,
            outcome_access=a.outcome_access,
            evaluability=AdjudicationEvaluability.UNEVALUABLE
            if status is ScientificStatus.UNEVALUABLE
            else AdjudicationEvaluability.EVALUABLE,
            scientific_status=status,
            admission_status=AdmissionStatus.NOT_EVALUATED,
            reason_codes=(
                "LOCAL_RECEIVER_RESULTS_FROM_EXISTING_QUALIFICATION_OWNER",
                "ADMISSION_CONTROLLER_USE_PREREQUISITE_NONENTRY_UNCOVERED_INITIAL_DOMAIN",
            ),
        )


class LocalMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        config: LocalQualificationDesign,
        custody: DependencyCustodyReader,
        publisher: CandidatePayloadPublisher,
        reader: CandidatePayloadReader,
        limits: EmpiricalResourceGuard,
    ) -> None:
        if registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version) != CAPABILITY:
            raise ValueError("local method registry differs")
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self.config = config
        self.runner = LocalMethodRunner(config, custody, publisher, reader, limits)

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry.fingerprint() != self.registry_sha256 or source_records:
            raise ValueError("local method registry/source differs")
        return (self.runner,)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("local method plan/source differs")
        outputs = {}
        for task in plan.tasks:
            if task.capability.capability_key != CAPABILITY.capability_key:
                continue
            for spec in task.external_inputs:
                if (
                    spec.expected_payload_schema != self.config.SCHEMA
                    or spec.expected_content_sha256 != self.config.fingerprint()
                ):
                    raise ValueError("local method external recipe differs")
                parent = ArtifactLineageParent(
                    ObjectIdentity.from_record(spec.logical_artifact_id, self.config),
                    VisibilityCeiling.PROSPECTIVE,
                    OutcomeAccess.OUTCOME_BLIND,
                )
                outputs[spec.logical_artifact_id] = ExternalInputPayload.from_bytes(
                    logical_artifact_id=spec.logical_artifact_id,
                    payload_schema=self.config.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/json",
                    payload=self.config.canonical_bytes(),
                    visibility_ceiling=parent.visibility_ceiling,
                    outcome_access=parent.outcome_access,
                    parent_visibility_ceilings=(parent.visibility_ceiling,),
                    lineage_parents=(parent,),
                    logical_content_sha256=self.config.fingerprint(),
                )
        return tuple(outputs[k] for k in sorted(outputs))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self.runners(registry)
        return tuple(
            CapabilityOutputSemanticContract.from_manifest(
                CAPABILITY,
                payload_schema=record.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=record.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(f.name for f in fields(record))),
            )
            for record in (
                LocalCalibration,
                LocalQualificationResult,
                ScientificAdjudicationRecord,
            )
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        self.runners(registry)
        if execution_plan is None:
            raise ValueError("local adjudication requires its full compiled plan")
        outputs = [
            o
            for t in execution_plan.tasks
            if t.capability.capability_key == CAPABILITY.capability_key
            for o in t.outputs
            if o.payload_schema == ScientificAdjudicationRecord.SCHEMA
        ]
        if len(outputs) != 1:
            raise ValueError("local study requires one adjudication output")
        return ScientificAdjudicationOutputContract(
            CAPABILITY.capability_key,
            CAPABILITY.capability_version,
            outputs[0].output_id,
            ScientificAdjudicationRecord.SCHEMA,
        )

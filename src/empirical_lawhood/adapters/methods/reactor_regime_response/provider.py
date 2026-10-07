"""Public campaign fit, seal, nomination and C reduction tasks."""

from __future__ import annotations

from dataclasses import dataclass, fields
from hashlib import sha256
from typing import Protocol, TypeVar

from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader, validate_dependency
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.status import AdmissionStatus, ScientificStatus
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactManifest,
    ArtifactProfile,
    CanonicalTaskReceipt,
    ReceiptCheck,
)
from empirical_lawhood.runtime.adjudication import (
    AdjudicationEvaluability,
    ScientificAdjudicationOutputContract,
    ScientificAdjudicationRecord,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.capabilities import CapabilityPermission
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskRunner,
    WorkerInputPort,
    WorkerInputKind,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.planning.approval import DurableAuthorizationRecord
from empirical_lawhood.planning.study_issue import StudyOperationAuthority
from empirical_lawhood.runtime.execution_envelope import ExecutionResourceEnvelopeSpec
from empirical_lawhood.runtime.controller_evaluation_nested import (
    DurablePreparedExecutionEventStore,
)
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .config import ROOTS, ReactorRegimeResponseDesign
from .calibration_pipeline import RegimeCalibrationPackage, build_calibration_package
from .control_prospective_plan import ReactorRegimeResponsePreparedProspectivePlanBundle, build_prepared_prospective_plan
from .control_prospective_closeout import RegimeDRootSealResult, seal_native_root
from .control_prospective_reveal import RegimeDRootRevealResult, reveal_native_root
from .control_prospective_cohort import ReactorRegimeResponseCohortProspective, reduce_d_cohort
from .control_services import implementation_payloads
from .extension_bundle import CAPABILITY
from .fit_pipeline import build_fit_package
from .law_terminal import RegimeJointLawResult, qualify_joint_law
from .model_records import RegimeFitPackage
from .nomination_pipeline import build_nomination_package
from .nomination_records import RegimeNominationPackage
from .prediction_seal import seal_confirmation_predictions, seal_fit_acquisition, seal_nomination_predictions, seal_prospective_predictions
from .prior import RegimePriorArtifact
from .continuation import NoRegimeRetainedInputs, RegimeRetainedInputsPort, retained_input_map, retained_payloads
from .discovery_records import RegimeContextMeasurement, RegimeDiscoveryTraining, RegimeDiscoveryDevelopment
from .measured_panel import measured_contexts
from .regime_stability import build_discovery_development
from .preassay_readout import RegimeReferenceComparators, RegimePreassayReadout, RegimeOpportunityReadout, seal_preassay_readout, evaluate_opportunities
from .discovery_diagnostics import RegimePhaseDiagnostics, RegimeDiscoveryConfirmation, build_phase_diagnostics, confirm_regimes
from .prospective_decision import CausalValidityRegimeAssignment, seal_prospective_assignment
from .qualification_pipeline import RegimeQualificationPackage, build_qualification_package
from .records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal, RegimePrivatePreparation
from .release import require_scientific_release
from .science import regime_system
from empirical_lawhood.adapters.simulators.reactor_regime_response.prospective_records import RegimeDNativeRoot

RecordT = TypeVar("RecordT", bound=CanonicalRecord)


class RegimeDMethodCustodyPort(Protocol):
    def open_prepared_store(self) -> DurablePreparedExecutionEventStore: ...
    def event_clock(self, root: str, stage: str) -> str: ...


@dataclass(frozen=True, slots=True)
class RegimeDMethodBinding:
    """Replayed D scientific approval, execution authority and resource identity."""

    approval: ObjectIdentity
    execution: ObjectIdentity
    resources: ObjectIdentity
    reveal: ObjectIdentity

    def __post_init__(self) -> None:
        if (
            self.approval.object_schema != DurableAuthorizationRecord.SCHEMA
            or self.execution.object_schema != StudyOperationAuthority.SCHEMA
            or self.resources.object_schema != ExecutionResourceEnvelopeSpec.SCHEMA
            or self.reveal.object_schema != DurableAuthorizationRecord.SCHEMA
        ):
            raise ValueError("D method binding lacks its three exact typed authorities")


class RegimeMethodRunner:
    manifest = CAPABILITY

    def __init__(
        self,
        design: ReactorRegimeResponseDesign,
        custody: DependencyCustodyReader,
        limits: EmpiricalResourceGuard,
        publisher: CandidatePayloadPublisher,
        reader: CandidatePayloadReader,
        prior_artifacts: tuple[RegimePriorArtifact, ...] = (),
        d_binding: RegimeDMethodBinding | None = None,
        source: EmpiricalStudySource | None = None,
        d_custody: RegimeDMethodCustodyPort | None = None,
        references: RegimeReferenceComparators | None = None,
        retained: RegimeRetainedInputsPort = NoRegimeRetainedInputs(),
    ) -> None:
        self.design, self.custody, self.limits = design, custody, limits
        self.publisher, self.reader = publisher, reader
        self.d_binding, self.source, self.d_custody = d_binding, source, d_custody
        self.references = references
        if (d_binding is None) != (d_custody is None) or (
            d_binding is not None and source is None
        ):
            raise ValueError(
                "D method authority, source and custody must be reconstructed together"
            )
        self.prior_artifacts = {
            value.manifest.logical.payload_schema: value for value in prior_artifacts
        }
        if len(self.prior_artifacts) != len(prior_artifacts):
            raise ValueError("reactor prior package schemas repeat")
        self.retained_inputs = retained_input_map(retained)

    def execute(self, context: TaskContext) -> RunnerResult:
        with self.limits.task(context.task_id):
            return self._execute(context)

    def _read_dependency(
        self,
        context: TaskContext,
        port: WorkerInputPort,
        record_type: type[RecordT],
    ) -> tuple[RecordT, str]:
        if port.kind is WorkerInputKind.EXTERNAL:
            retained = self.retained_inputs.get(port.artifact_id)
            if retained is not None:
                raw = port.read()
                logical = retained.manifest.logical
                if (
                    port.payload_schema != record_type.SCHEMA
                    or logical.payload_schema != record_type.SCHEMA
                    or port.media_type != logical.media_type
                    or port.size_bytes != retained.manifest.materialization.size_bytes
                    or sha256(raw).hexdigest() != logical.content_sha256
                ):
                    raise ValueError(
                        "retained C method operand lost exact original custody"
                    )
                return decode_canonical_bytes(
                    raw, record_type, maximum_bytes=512 * 1024**2
                ), retained.task_id
            prior = self.prior_artifacts.get(port.payload_schema)
            if (
                prior is None
                or prior.manifest.logical.logical_artifact_id != port.artifact_id
                or prior.manifest.logical.payload_schema != record_type.SCHEMA
                or prior.manifest.logical.media_type != port.media_type
                or port.size_bytes != len(prior.payload)
                or sha256(port.read()).hexdigest()
                != prior.manifest.logical.content_sha256
            ):
                raise ValueError("method cross-issue B package lost exact custody")
            record = decode_canonical_bytes(
                prior.payload, record_type, maximum_bytes=512 * 1024**2
            )
            return record, prior.expected_task_id
        if port.kind is not WorkerInputKind.DEPENDENCY:
            raise ValueError("method scientific input lacks dependency custody")
        manifest, receipt = self.custody.read_dependency(context, port.binding)
        validate_dependency(port.binding, manifest, receipt)
        record = decode_canonical_bytes(
            port.read(), record_type, maximum_bytes=512 * 1024**2
        )
        if record.fingerprint() != manifest.logical.content_sha256:
            raise ValueError("method dependency differs from its authenticated receipt")
        return record, receipt.task_id

    def _d_assignment(
        self, context: TaskContext, scientific: list[WorkerInputPort]
    ) -> CausalValidityRegimeAssignment:
        if self.d_binding is None or self.source is None:
            raise ValueError(
                "D assignment lacks its exact source and authority binding"
            )
        root = context.task_id.removeprefix("regime.assignment.")
        expected = {
            RegimeCausalPreparation.SCHEMA: (
                RegimeCausalPreparation,
                f"regime.prepare.{root}",
            ),
            RegimePredictionSeal.SCHEMA: (
                RegimePredictionSeal,
                f"regime.seal.{root}",
            ),
            RegimeFitPackage.SCHEMA: (RegimeFitPackage, "regime.fit"),
            RegimeNominationPackage.SCHEMA: (
                RegimeNominationPackage,
                "regime.nomination",
            ),
            RegimeCalibrationPackage.SCHEMA: (
                RegimeCalibrationPackage,
                "regime.calibration",
            ),
            RegimeQualificationPackage.SCHEMA: (
                RegimeQualificationPackage,
                "regime.qualification",
            ),
            RegimeJointLawResult.SCHEMA: (
                RegimeJointLawResult,
                "regime.law-qualification",
            ),
        }
        ports = {port.payload_schema: port for port in scientific}
        if len(ports) != len(scientific) or set(ports) != set(expected):
            raise ValueError(
                "D assignment lacks its exact two local and five prior inputs"
            )
        values: dict[str, CanonicalRecord] = {}
        for schema, (record_type, task_id) in expected.items():
            record: CanonicalRecord
            record, observed_task = self._read_dependency(
                context, ports[schema], record_type
            )
            if observed_task != task_id:
                raise ValueError(
                    "D assignment changed a source or C package task receipt"
                )
            values[schema] = record
        causal = values[RegimeCausalPreparation.SCHEMA]
        seal = values[RegimePredictionSeal.SCHEMA]
        fit = values[RegimeFitPackage.SCHEMA]
        nomination = values[RegimeNominationPackage.SCHEMA]
        calibration = values[RegimeCalibrationPackage.SCHEMA]
        qualification = values[RegimeQualificationPackage.SCHEMA]
        law = values[RegimeJointLawResult.SCHEMA]
        assert isinstance(causal, RegimeCausalPreparation)
        assert isinstance(seal, RegimePredictionSeal)
        assert isinstance(fit, RegimeFitPackage)
        assert isinstance(nomination, RegimeNominationPackage)
        assert isinstance(calibration, RegimeCalibrationPackage)
        assert isinstance(qualification, RegimeQualificationPackage)
        assert isinstance(law, RegimeJointLawResult)
        if (
            causal.root != root
            or self.source.batch.fingerprint() != causal.source_sha256
        ):
            raise ValueError("D assignment source or independent root differs")
        return seal_prospective_assignment(
            self.source.batch,
            causal,
            seal,
            fit,
            nomination,
            calibration,
            qualification,
            law,
            self.d_binding.approval,
        )

    def _d_plan(
        self, context: TaskContext, scientific: list[WorkerInputPort]
    ) -> ReactorRegimeResponsePreparedProspectivePlanBundle:
        if self.d_binding is None:
            raise ValueError("D plan lacks its execution authority and resources")
        expected = tuple(root for root, role, _, _ in ROOTS if role == "prospective")
        if len(scientific) != 2 * len(expected) + 1:
            raise ValueError("D plan lacks its all-64 preparation and decision census")
        causal: dict[str, RegimeCausalPreparation] = {}
        assignments: dict[str, CausalValidityRegimeAssignment] = {}
        law: RegimeJointLawResult | None = None
        for port in scientific:
            if port.payload_schema == RegimeCausalPreparation.SCHEMA:
                causal_value, task_id = self._read_dependency(
                    context, port, RegimeCausalPreparation
                )
                if (
                    task_id != f"regime.prepare.{causal_value.root}"
                    or causal_value.root in causal
                ):
                    raise ValueError("D plan changed a causal preparation receipt")
                causal[causal_value.root] = causal_value
            elif port.payload_schema == CausalValidityRegimeAssignment.SCHEMA:
                assignment_value, task_id = self._read_dependency(
                    context, port, CausalValidityRegimeAssignment
                )
                if (
                    task_id != f"regime.assignment.{assignment_value.root}"
                    or assignment_value.root in assignments
                ):
                    raise ValueError("D plan changed a sealed root decision receipt")
                assignments[assignment_value.root] = assignment_value
            elif port.payload_schema == RegimeJointLawResult.SCHEMA:
                law_value, task_id = self._read_dependency(
                    context, port, RegimeJointLawResult
                )
                if law is not None or task_id != "regime.law-qualification":
                    raise ValueError("D plan substituted the released C law")
                law = law_value
            else:
                raise ValueError("D plan received an undeclared scientific input")
        if (
            set(causal) != set(expected)
            or set(assignments) != set(expected)
            or law is None
        ):
            raise ValueError("D plan changed the all-assigned root denominator")
        producer_binding = implementation_payloads()[0][0]
        return build_prepared_prospective_plan(
            design=self.design,
            report=law,
            prepared=tuple((causal[root], assignments[root]) for root in expected),
            producer=ObjectIdentity.from_record(
                producer_binding.binding_id, producer_binding
            ),
            resource=self.d_binding.resources,
            authority=self.d_binding.approval,
        )

    def _d_seal(
        self, context: TaskContext, scientific: list[WorkerInputPort]
    ) -> RegimeDRootSealResult:
        if self.d_custody is None or len(scientific) != 5:
            raise ValueError(
                "D controller use seal lacks its exact native root, preparation and event custody"
            )
        root = context.task_id.removeprefix("regime.d-seal.")
        ports = {port.payload_schema: port for port in scientific}
        if len(ports) != len(scientific) or set(ports) != {
            RegimeDNativeRoot.SCHEMA,
            ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA,
            CausalValidityRegimeAssignment.SCHEMA,
            RegimeCausalPreparation.SCHEMA,
            RegimePrivatePreparation.SCHEMA,
        }:
            raise ValueError(
                "D controller use seal changed its native root, precontact plan or preparation inputs"
            )
        native_port = ports[RegimeDNativeRoot.SCHEMA]
        if native_port.kind is not WorkerInputKind.DEPENDENCY:
            raise ValueError("D controller use seal native root lacks task receipt custody")
        manifest, receipt = self.custody.read_dependency(context, native_port.binding)
        validate_dependency(native_port.binding, manifest, receipt)
        native, task = self._read_dependency(context, native_port, RegimeDNativeRoot)
        plan, plan_task = self._read_dependency(
            context,
            ports[ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA],
            ReactorRegimeResponsePreparedProspectivePlanBundle,
        )
        assignment, assignment_task = self._read_dependency(
            context,
            ports[CausalValidityRegimeAssignment.SCHEMA],
            CausalValidityRegimeAssignment,
        )
        causal, causal_task = self._read_dependency(
            context,
            ports[RegimeCausalPreparation.SCHEMA],
            RegimeCausalPreparation,
        )
        private, private_task = self._read_dependency(
            context,
            ports[RegimePrivatePreparation.SCHEMA],
            RegimePrivatePreparation,
        )
        if (
            task != f"regime.action.{root}"
            or plan_task != "regime.d-plan"
            or native.root != root
            or manifest.logical.content_sha256 != native.fingerprint()
            or assignment_task != f"regime.assignment.{root}"
            or causal_task != f"regime.prepare.{root}"
            or private_task != f"regime.prepare.{root}"
        ):
            raise ValueError("D controller use seal substituted its exact action or plan receipt")
        logical = manifest.logical
        artifact = ArtifactIdentity(
            logical.logical_artifact_id,
            "reactor-regime-d-native-root",
            logical.payload_schema,
            logical.content_sha256,
            logical.media_type,
            manifest.materialization.size_bytes,
        )
        sealed = seal_native_root(
            native=native,
            plan_bundle=plan,
            assignment=assignment,
            causal=causal,
            private=private,
            receipt=receipt,
            artifact=artifact,
            store=self.d_custody.open_prepared_store(),
            occurred_at_utc=self.d_custody.event_clock(root, "seal"),
        )
        reasons = (
            ()
            if sealed is not None
            else (native.reasons or ("D_SELECTED_PREPARATION_NONENTRY",))
        )
        return RegimeDRootSealResult(
            root,
            ObjectIdentity.from_record(f"{root}.native-action", native),
            ObjectIdentity.from_record("regime.d-plan", plan),
            sealed,
            reasons,
        )

    def _d_reveal(
        self, context: TaskContext, scientific: list[WorkerInputPort]
    ) -> RegimeDRootRevealResult:
        if self.d_binding is None or len(scientific) != 6:
            raise ValueError(
                "D reveal lacks its separate authority and six sealed operands"
            )
        root = context.task_id.removeprefix("regime.d-reveal.")
        expected = {
            RegimeDRootSealResult.SCHEMA: (
                RegimeDRootSealResult,
                f"regime.d-seal.{root}",
            ),
            RegimeDNativeRoot.SCHEMA: (RegimeDNativeRoot, f"regime.action.{root}"),
            CausalValidityRegimeAssignment.SCHEMA: (
                CausalValidityRegimeAssignment,
                f"regime.assignment.{root}",
            ),
            RegimeCausalPreparation.SCHEMA: (
                RegimeCausalPreparation,
                f"regime.prepare.{root}",
            ),
            RegimePrivatePreparation.SCHEMA: (
                RegimePrivatePreparation,
                f"regime.prepare.{root}",
            ),
            ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA: (
                ReactorRegimeResponsePreparedProspectivePlanBundle,
                "regime.d-plan",
            ),
        }
        ports = {port.payload_schema: port for port in scientific}
        if len(ports) != len(scientific) or set(ports) != set(expected):
            raise ValueError("D reveal changed its exact root and plan input census")
        values: dict[str, CanonicalRecord] = {}
        for schema, (record_type, task_id) in expected.items():
            value: CanonicalRecord
            value, observed_task = self._read_dependency(
                context, ports[schema], record_type
            )
            if observed_task != task_id:
                raise ValueError("D reveal changed a sealed source task receipt")
            values[schema] = value
        sealed = values[RegimeDRootSealResult.SCHEMA]
        native = values[RegimeDNativeRoot.SCHEMA]
        assignment = values[CausalValidityRegimeAssignment.SCHEMA]
        causal = values[RegimeCausalPreparation.SCHEMA]
        private = values[RegimePrivatePreparation.SCHEMA]
        plan = values[ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA]
        assert isinstance(sealed, RegimeDRootSealResult)
        assert isinstance(native, RegimeDNativeRoot)
        assert isinstance(assignment, CausalValidityRegimeAssignment)
        assert isinstance(causal, RegimeCausalPreparation)
        assert isinstance(private, RegimePrivatePreparation)
        assert isinstance(plan, ReactorRegimeResponsePreparedProspectivePlanBundle)
        return reveal_native_root(
            sealed=sealed,
            native=native,
            plan_bundle=plan,
            assignment=assignment,
            causal=causal,
            private=private,
            reveal_authority=self.d_binding.reveal,
        )

    def _d_cohort(
        self, context: TaskContext, scientific: list[WorkerInputPort]
    ) -> ReactorRegimeResponseCohortProspective:
        expected = tuple(root for root, role, _, _ in ROOTS if role == "prospective")
        if len(scientific) != len(expected) + 1:
            raise ValueError("D cohort lacks its exact 64 root and plan inputs")
        roots: dict[str, RegimeDRootRevealResult] = {}
        plan: ReactorRegimeResponsePreparedProspectivePlanBundle | None = None
        for port in scientific:
            if port.payload_schema == RegimeDRootRevealResult.SCHEMA:
                value, task_id = self._read_dependency(
                    context, port, RegimeDRootRevealResult
                )
                if task_id != f"regime.d-reveal.{value.root}" or value.root in roots:
                    raise ValueError("D cohort substituted a root reveal receipt")
                roots[value.root] = value
            elif port.payload_schema == ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA:
                plan_value, task_id = self._read_dependency(
                    context, port, ReactorRegimeResponsePreparedProspectivePlanBundle
                )
                if plan is not None or task_id != "regime.d-plan":
                    raise ValueError("D cohort substituted its frozen controller use plan")
                plan = plan_value
            else:
                raise ValueError("D cohort received an undeclared scientific input")
        if set(roots) != set(expected) or plan is None:
            raise ValueError("D cohort changed its all-assigned root denominator")
        return reduce_d_cohort(
            plan_bundle=plan, roots=tuple(roots[root] for root in expected)
        )

    def _d_adjudicate(
        self, context: TaskContext, scientific: list[WorkerInputPort]
    ) -> ScientificAdjudicationRecord:
        if (
            context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or len(scientific) != 5
        ):
            raise ValueError(
                "D scientific adjudication lacks five separately revealed operands"
            )
        expected = {
            ReactorRegimeResponseCohortProspective.SCHEMA: (ReactorRegimeResponseCohortProspective, "regime.d-cohort"),
            ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA: (
                ReactorRegimeResponsePreparedProspectivePlanBundle,
                "regime.d-plan",
            ),
            RegimeNominationPackage.SCHEMA: (
                RegimeNominationPackage,
                "regime.nomination",
            ),
            RegimeQualificationPackage.SCHEMA: (
                RegimeQualificationPackage,
                "regime.qualification",
            ),
            RegimeJointLawResult.SCHEMA: (
                RegimeJointLawResult,
                "regime.law-qualification",
            ),
        }
        ports = {port.payload_schema: port for port in scientific}
        if set(ports) != set(expected) or len(ports) != len(scientific):
            raise ValueError("D readout changed its all-root controller use or released C source")
        values: dict[str, CanonicalRecord] = {}
        for schema, (record_type, task_id) in expected.items():
            value: CanonicalRecord
            value, observed_task = self._read_dependency(
                context, ports[schema], record_type
            )
            if observed_task != task_id:
                raise ValueError(
                    "D scientific adjudication changed a frozen task receipt"
                )
            values[schema] = value
        cohort = values[ReactorRegimeResponseCohortProspective.SCHEMA]
        plan = values[ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA]
        nomination = values[RegimeNominationPackage.SCHEMA]
        qualification = values[RegimeQualificationPackage.SCHEMA]
        law = values[RegimeJointLawResult.SCHEMA]
        assert isinstance(cohort, ReactorRegimeResponseCohortProspective)
        assert isinstance(plan, ReactorRegimeResponsePreparedProspectivePlanBundle)
        assert isinstance(nomination, RegimeNominationPackage)
        assert isinstance(qualification, RegimeQualificationPackage)
        assert isinstance(law, RegimeJointLawResult)
        require_scientific_release(nomination, qualification, law)
        if cohort.plan != ObjectIdentity.from_record("regime.d-plan", plan):
            raise ValueError("D controller use cohort substituted its frozen precontact plan")
        adjudication = context.scientific_adjudication_context
        system = regime_system()
        if (
            adjudication is None
            or adjudication.relation
            != ObjectIdentity.from_record(system.relation.relation_id, system.relation)
            or adjudication.independent_unit_id != system.independent_unit.unit_id
            or adjudication.evidence_world_id != system.world.world_id
        ):
            raise ValueError("D readout lacks the compiled reactor scientific relation")
        return ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=adjudication.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=tuple(
                sorted(
                    port.logical_artifact_id
                    for port in context.output_ports
                    if port.logical_artifact_id is not None
                )
            ),
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=adjudication.evidence_world_id,
            evidence_world_kind=adjudication.evidence_world_kind,
            relation=adjudication.relation,
            independent_unit_id=adjudication.independent_unit_id,
            information_cutoffs=adjudication.information_cutoffs,
            visibility_ceiling=adjudication.visibility_ceiling,
            outcome_access=adjudication.outcome_access,
            evaluability=AdjudicationEvaluability.EVALUABLE,
            scientific_status=ScientificStatus.SUPPORTED
            if cohort.adequate_use
            else ScientificStatus.NOT_SUPPORTED,
            admission_status=AdmissionStatus.ADMITTED
            if cohort.admitted_requests
            else AdmissionStatus.EMPTY,
            reason_codes=tuple(
                sorted(
                    (
                        "REACTOR_REGIME_ACTUAL_ADMISSION_CONTROLLER_USE_OWNER",
                        "REACTOR_D_ADEQUATE_USE"
                        if cohort.adequate_use
                        else "REACTOR_D_ADEQUATE_USE_NOT_ESTABLISHED",
                    )
                )
            ),
        )

    def _law_qualification(
        self, context: TaskContext, scientific: list[WorkerInputPort]
    ) -> RegimeJointLawResult:
        if context.outcome_access is not OutcomeAccess.EVALUATION_SEALED:
            raise ValueError(
                "sole-owner law qualification requires its frozen C qualification boundary"
            )
        expected = tuple(root for root, role, _, _ in ROOTS if role == "qualification")
        if len(scientific) != 4 + len(expected):
            raise ValueError("law qualification owner lacks B/C packages or its 64 native assays")
        packages: dict[str, CanonicalRecord] = {}
        assays: dict[str, tuple[ArtifactManifest, CanonicalTaskReceipt]] = {}
        for port in scientific:
            if port.payload_schema == RegimeAssayPanel.SCHEMA:
                if port.kind is not WorkerInputKind.DEPENDENCY:
                    raise ValueError(
                        "law qualification native assay lacks immutable dependency custody"
                    )
                manifest, receipt = self.custody.read_dependency(context, port.binding)
                validate_dependency(port.binding, manifest, receipt)
                assay = decode_canonical_bytes(
                    port.read(), RegimeAssayPanel, maximum_bytes=512 * 1024**2
                )
                if (
                    assay.role != "qualification"
                    or receipt.task_id != f"regime.assay.{assay.root}"
                    or assay.fingerprint() != manifest.logical.content_sha256
                    or assay.root in assays
                ):
                    raise ValueError("law qualification changed a fresh native assay identity")
                assays[assay.root] = manifest, receipt
                continue
            record_type = {
                RegimeFitPackage.SCHEMA: RegimeFitPackage,
                RegimeNominationPackage.SCHEMA: RegimeNominationPackage,
                RegimeCalibrationPackage.SCHEMA: RegimeCalibrationPackage,
                RegimeQualificationPackage.SCHEMA: RegimeQualificationPackage,
            }.get(port.payload_schema)
            if record_type is None or port.payload_schema in packages:
                raise ValueError("law qualification changed its four frozen package inputs")
            record: CanonicalRecord
            record, task_id = self._read_dependency(context, port, record_type)
            if (
                task_id
                != {
                    RegimeFitPackage.SCHEMA: "regime.fit",
                    RegimeNominationPackage.SCHEMA: "regime.nomination",
                    RegimeCalibrationPackage.SCHEMA: "regime.calibration",
                    RegimeQualificationPackage.SCHEMA: "regime.qualification",
                }[port.payload_schema]
            ):
                raise ValueError("law qualification package receipt was substituted")
            packages[port.payload_schema] = record
        if set(assays) != set(expected) or set(packages) != {
            RegimeFitPackage.SCHEMA,
            RegimeNominationPackage.SCHEMA,
            RegimeCalibrationPackage.SCHEMA,
            RegimeQualificationPackage.SCHEMA,
        }:
            raise ValueError("law qualification lost its all-assigned root or package census")
        fit = packages[RegimeFitPackage.SCHEMA]
        nomination = packages[RegimeNominationPackage.SCHEMA]
        calibration = packages[RegimeCalibrationPackage.SCHEMA]
        qualification = packages[RegimeQualificationPackage.SCHEMA]
        if not (
            isinstance(fit, RegimeFitPackage)
            and isinstance(nomination, RegimeNominationPackage)
            and isinstance(calibration, RegimeCalibrationPackage)
            and isinstance(qualification, RegimeQualificationPackage)
        ):
            raise TypeError("law qualification package schema decoding differs")
        design_artifact = ArtifactIdentity(
            self.design.config_id,
            "reactor-regime-response-design",
            self.design.SCHEMA,
            self.design.fingerprint(),
            "application/json",
            len(self.design.canonical_bytes()),
        )
        return qualify_joint_law(
            self.design,
            fit,
            nomination,
            calibration,
            qualification,
            tuple(assays[root][0] for root in expected),
            tuple(assays[root][1] for root in expected),
            CAPABILITY,
            self.publisher,
            self.reader,
            design_artifact,
        )

    def _b_adjudicate(
        self, context: TaskContext, scientific: list[WorkerInputPort]
    ) -> ScientificAdjudicationRecord:
        if (
            context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or len(scientific) != 3
        ):
            raise ValueError(
                "B development closeout lacks separately revealed fit/nomination"
            )
        by_schema = {port.payload_schema: port for port in scientific}
        if set(by_schema) != {
            RegimeFitPackage.SCHEMA,
            RegimeNominationPackage.SCHEMA,
            RegimeDiscoveryDevelopment.SCHEMA,
        }:
            raise ValueError("B closeout changed its two complete method packages")
        fit, fit_task = self._read_dependency(
            context, by_schema[RegimeFitPackage.SCHEMA], RegimeFitPackage
        )
        nomination, nomination_task = self._read_dependency(
            context,
            by_schema[RegimeNominationPackage.SCHEMA],
            RegimeNominationPackage,
        )
        discovery, discovery_task = self._read_dependency(
            context,
            by_schema[RegimeDiscoveryDevelopment.SCHEMA],
            RegimeDiscoveryDevelopment,
        )
        if (
            fit_task != "regime.fit"
            or nomination_task != "regime.nomination"
            or discovery_task != nomination_task
            or discovery.nomination
            != ObjectIdentity.from_record(nomination.package_id, nomination)
            or discovery.fit_package != nomination.fit_package
            or nomination.fit_package != ObjectIdentity.from_record(fit.package_id, fit)
        ):
            raise ValueError("B closeout substituted the frozen fit/nomination receipt")
        adjudication = context.scientific_adjudication_context
        system = regime_system()
        if (
            adjudication is None
            or adjudication.relation
            != ObjectIdentity.from_record(system.relation.relation_id, system.relation)
            or adjudication.independent_unit_id != system.independent_unit.unit_id
            or adjudication.evidence_world_id != system.world.world_id
        ):
            raise ValueError("B closeout lacks its compiled scientific relation")
        nominated = nomination.selected_route is not None
        nomination_roots = {root for root, role, _, _ in ROOTS if role == "nomination"}
        complete_losses = all(
            any(
                {root for root, _ in score.root_losses_K2} == nomination_roots
                for score in roster
            )
            for roster in (nomination.coefficient_scores, nomination.absolute_scores)
        )
        evaluable = nominated or complete_losses
        return ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=adjudication.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=tuple(
                sorted(
                    port.logical_artifact_id
                    for port in context.output_ports
                    if port.logical_artifact_id is not None
                )
            ),
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=adjudication.evidence_world_id,
            evidence_world_kind=adjudication.evidence_world_kind,
            relation=adjudication.relation,
            independent_unit_id=adjudication.independent_unit_id,
            information_cutoffs=adjudication.information_cutoffs,
            visibility_ceiling=adjudication.visibility_ceiling,
            outcome_access=adjudication.outcome_access,
            evaluability=AdjudicationEvaluability.EVALUABLE
            if evaluable
            else AdjudicationEvaluability.UNEVALUABLE,
            scientific_status=ScientificStatus.PARTIAL
            if nominated
            else ScientificStatus.NOT_SUPPORTED
            if evaluable
            else ScientificStatus.UNEVALUABLE,
            admission_status=AdmissionStatus.NOT_EVALUATED
            if evaluable
            else AdmissionStatus.UNEVALUABLE,
            reason_codes=tuple(
                sorted(
                    (
                        "REACTOR_B_DEVELOPMENT_NOMINATION_ONLY_NO_FRESH_QUALIFICATION",
                        "B_ROUTE_NOMINATED"
                        if nominated
                        else "B_NO_FINITE_NOMINATED_ROUTE",
                        "B_COMPLETE_DEVELOPMENT_LOSSES"
                        if complete_losses
                        else "B_INCOMPLETE_DEVELOPMENT_LOSSES",
                        "B_PARTITION_STABILITY_ESTABLISHED"
                        if any(
                            value.stability_pass for value in discovery.local_stability
                        )
                        else "B_PARTITION_STABILITY_NOT_ESTABLISHED",
                    )
                )
            ),
        )

    def _adjudicate(
        self, context: TaskContext, scientific: list[WorkerInputPort]
    ) -> ScientificAdjudicationRecord:
        if (
            context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or len(scientific) != 3
        ):
            raise ValueError("C readout lacks exact separately revealed law qualification operands")
        by_schema = {port.payload_schema: port for port in scientific}
        if set(by_schema) != {
            RegimeQualificationPackage.SCHEMA,
            RegimeJointLawResult.SCHEMA,
            RegimeDiscoveryConfirmation.SCHEMA,
        }:
            raise ValueError("C readout changed its method and sole-owner census")
        q, q_task = self._read_dependency(
            context,
            by_schema[RegimeQualificationPackage.SCHEMA],
            RegimeQualificationPackage,
        )
        law, law_task = self._read_dependency(
            context,
            by_schema[RegimeJointLawResult.SCHEMA],
            RegimeJointLawResult,
        )
        discovery, discovery_task = self._read_dependency(
            context,
            by_schema[RegimeDiscoveryConfirmation.SCHEMA],
            RegimeDiscoveryConfirmation,
        )
        if (
            q_task != "regime.qualification"
            or discovery_task != q_task
            or discovery.qualification != ObjectIdentity.from_record(q.package_id, q)
            or law_task != "regime.law-qualification"
            or law.qualification.dataset_or_projection
            != ObjectIdentity.from_record(q.package_id, q)
        ):
            raise ValueError("C readout substituted its law qualification qualification source")
        adjudication = context.scientific_adjudication_context
        system = regime_system()
        if (
            adjudication is None
            or adjudication.relation
            != ObjectIdentity.from_record(system.relation.relation_id, system.relation)
            or adjudication.independent_unit_id != system.independent_unit.unit_id
            or adjudication.evidence_world_id != system.world.world_id
        ):
            raise ValueError("C readout lacks its compiled scientific relation")
        status = law.qualification.scientific_status
        return ScientificAdjudicationRecord(
            adjudication_id=f"adjudication.{context.run_id}.{context.task_id}",
            run_id=context.run_id,
            adjudication_task_id=context.task_id,
            execution_plan=adjudication.execution_plan,
            input_materialization_ids=context.input_materialization_ids,
            output_logical_artifact_ids=tuple(
                sorted(
                    port.logical_artifact_id
                    for port in context.output_ports
                    if port.logical_artifact_id is not None
                )
            ),
            required_receipt_ids=context.dependency_receipt_ids,
            evidence_world_id=adjudication.evidence_world_id,
            evidence_world_kind=adjudication.evidence_world_kind,
            relation=adjudication.relation,
            independent_unit_id=adjudication.independent_unit_id,
            information_cutoffs=adjudication.information_cutoffs,
            visibility_ceiling=adjudication.visibility_ceiling,
            outcome_access=adjudication.outcome_access,
            evaluability=AdjudicationEvaluability.UNEVALUABLE
            if status is ScientificStatus.UNEVALUABLE
            else AdjudicationEvaluability.EVALUABLE,
            scientific_status=status,
            admission_status=AdmissionStatus.NOT_EVALUATED,
            reason_codes=tuple(
                sorted(
                    (
                        "REACTOR_C_SOLE_OWNER_LAW_RESULT",
                        discovery.verdict,
                        "D_JOINT_LAW_AND_PREPARATION_RELEASED"
                        if q.release_D
                        else "D_SCIENTIFIC_PREREQUISITE_NONENTRY",
                    )
                )
            ),
        )

    def _execute(self, context: TaskContext) -> RunnerResult:
        revealed = context.task_id in (
            "regime.adjudication",
            "regime.d-cohort",
            "regime.d-adjudication",
        ) or context.task_id.startswith("regime.d-reveal.")
        expected_permissions = (
            CAPABILITY.permissions
            if revealed
            else tuple(
                permission
                for permission in CAPABILITY.permissions
                if permission is not CapabilityPermission.REVEAL_OUTCOMES
            )
        )
        if (
            context.config.content_sha256 != self.design.fingerprint()
            or context.permissions != expected_permissions
            or context.outcome_access
            is not (
                OutcomeAccess.EVALUATOR_REVEAL
                if revealed
                else OutcomeAccess.EVALUATION_SEALED
            )
            or not context.output_ports
        ):
            raise ValueError("reactor method task/config/visibility/output differs")
        designs = [
            port
            for port in context.input_ports
            if port.payload_schema == ReactorRegimeResponseDesign.SCHEMA
        ]
        if (
            len(designs) != 1
            or decode_canonical_bytes(
                designs[0].read(),
                ReactorRegimeResponseDesign,
                maximum_bytes=256 * 1024,
            )
            != self.design
        ):
            raise ValueError("reactor method frozen design differs")
        scientific = [port for port in context.input_ports if port not in designs]
        preassay: dict[str, RegimePreassayReadout] = {}
        for port in tuple(scientific):
            if port.payload_schema != RegimePreassayReadout.SCHEMA:
                continue
            screen, producer = self._read_dependency(
                context, port, RegimePreassayReadout
            )
            if producer != f"regime.seal.{screen.root}" or screen.root in preassay:
                raise ValueError("preassay readout lost its unique causal seal receipt")
            preassay[screen.root] = screen
            scientific.remove(port)
        if context.task_id in (
            "regime.nomination",
            "regime.calibration",
            "regime.qualification",
        ):
            role = context.task_id.removeprefix("regime.")
            if set(preassay) != {
                root for root, assigned, _, _ in ROOTS if assigned == role
            }:
                raise ValueError(
                    "reduction lacks its all-assigned pre-label comparator/decision census"
                )
        elif preassay:
            raise ValueError("preassay readout reached an undeclared consumer")
        output: CanonicalRecord
        extra_outputs: list[CanonicalRecord] = []
        if context.task_id == "regime.law-qualification":
            output = self._law_qualification(context, scientific)
        elif context.task_id == "regime.adjudication":
            output = (
                self._b_adjudicate(context, scientific)
                if any(
                    port.payload_schema == RegimeFitPackage.SCHEMA
                    for port in scientific
                )
                else self._adjudicate(context, scientific)
            )
        elif context.task_id.startswith("regime.assignment."):
            output = self._d_assignment(context, scientific)
        elif context.task_id == "regime.d-plan":
            output = self._d_plan(context, scientific)
        elif context.task_id.startswith("regime.d-seal."):
            output = self._d_seal(context, scientific)
        elif context.task_id.startswith("regime.d-reveal."):
            output = self._d_reveal(context, scientific)
        elif context.task_id == "regime.d-cohort":
            output = self._d_cohort(context, scientific)
        elif context.task_id == "regime.d-adjudication":
            output = self._d_adjudicate(context, scientific)
        elif context.task_id == "regime.fit":
            expected = tuple(root for root, role, _, _ in ROOTS if role == "fit")
            if len(scientific) != 3 * len(expected):
                raise ValueError(
                    "fit task lacks its 32 preparation/seal/assay triplets"
                )
            causal: dict[str, RegimeCausalPreparation] = {}
            seals: dict[str, RegimePredictionSeal] = {}
            assays: dict[str, RegimeAssayPanel] = {}
            for port in scientific:
                if port.payload_schema == RegimeCausalPreparation.SCHEMA:
                    record, task_id = self._read_dependency(
                        context, port, RegimeCausalPreparation
                    )
                    if (
                        task_id != f"regime.prepare.{record.root}"
                        or record.root in causal
                    ):
                        raise ValueError("fit causal preparation receipt/root differs")
                    causal[record.root] = record
                elif port.payload_schema == RegimePredictionSeal.SCHEMA:
                    seal, task_id = self._read_dependency(
                        context, port, RegimePredictionSeal
                    )
                    if task_id != f"regime.seal.{seal.root}" or seal.root in seals:
                        raise ValueError("fit prediction seal receipt/root differs")
                    seals[seal.root] = seal
                elif port.payload_schema == RegimeAssayPanel.SCHEMA:
                    panel, task_id = self._read_dependency(
                        context, port, RegimeAssayPanel
                    )
                    if task_id != f"regime.assay.{panel.root}" or panel.root in assays:
                        raise ValueError("fit native assay receipt/root differs")
                    assays[panel.root] = panel
                else:
                    raise ValueError(
                        "fit task received an undeclared scientific schema"
                    )
            if (
                set(causal) != set(expected)
                or set(seals) != set(expected)
                or set(assays) != set(expected)
            ):
                raise ValueError("fit task changed its all-assigned root census")
            output = build_fit_package(
                tuple((causal[root], seals[root], assays[root]) for root in expected)
            )
            extra_outputs.append(
                RegimeDiscoveryTraining(
                    ObjectIdentity.from_record(output.package_id, output),
                    tuple(
                        RegimeContextMeasurement.from_context(row)
                        for root in expected
                        for row in measured_contexts(causal[root], assays[root])
                    ),
                )
            )
        elif context.task_id == "regime.qualification":
            expected = tuple(
                root for root, role, _, _ in ROOTS if role == "qualification"
            )
            discovery_ports = {
                port.payload_schema: port
                for port in scientific
                if port.payload_schema
                in (
                    RegimeDiscoveryDevelopment.SCHEMA,
                    RegimePhaseDiagnostics.SCHEMA,
                )
            }
            if len(discovery_ports) != 2:
                raise ValueError(
                    "qualification lacks B stability and calibration leaf operands"
                )
            development, development_task = self._read_dependency(
                context,
                discovery_ports[RegimeDiscoveryDevelopment.SCHEMA],
                RegimeDiscoveryDevelopment,
            )
            cal_diagnostics, cal_diagnostics_task = self._read_dependency(
                context,
                discovery_ports[RegimePhaseDiagnostics.SCHEMA],
                RegimePhaseDiagnostics,
            )
            if (
                development_task != "regime.nomination"
                or cal_diagnostics_task != "regime.calibration"
            ):
                raise ValueError("regime confirmation changed its prior task ancestry")
            scientific = [
                port for port in scientific if port not in discovery_ports.values()
            ]
            if len(scientific) != 3 + 4 * len(expected):
                raise ValueError("qualification lacks B/C packages or 64 root quartets")
            qual_fit: RegimeFitPackage | None = None
            qual_nominee: RegimeNominationPackage | None = None
            qual_calibration: RegimeCalibrationPackage | None = None
            qual_causal: dict[str, RegimeCausalPreparation] = {}
            qual_private: dict[str, RegimePrivatePreparation] = {}
            qual_seals: dict[str, RegimePredictionSeal] = {}
            qual_assays: dict[str, RegimeAssayPanel] = {}
            for port in scientific:
                if port.payload_schema == RegimeFitPackage.SCHEMA:
                    fit_value, task_id = self._read_dependency(
                        context, port, RegimeFitPackage
                    )
                    if qual_fit is not None or task_id != "regime.fit":
                        raise ValueError("C qualification changed its fit receipt")
                    qual_fit = fit_value
                elif port.payload_schema == RegimeNominationPackage.SCHEMA:
                    nomination_value, task_id = self._read_dependency(
                        context, port, RegimeNominationPackage
                    )
                    if qual_nominee is not None or task_id != "regime.nomination":
                        raise ValueError(
                            "C qualification changed its nomination receipt"
                        )
                    qual_nominee = nomination_value
                elif port.payload_schema == RegimeCalibrationPackage.SCHEMA:
                    calibration_value, task_id = self._read_dependency(
                        context, port, RegimeCalibrationPackage
                    )
                    if qual_calibration is not None or task_id != "regime.calibration":
                        raise ValueError(
                            "C qualification changed its calibration receipt"
                        )
                    qual_calibration = calibration_value
                elif port.payload_schema == RegimeCausalPreparation.SCHEMA:
                    causal_value, task_id = self._read_dependency(
                        context, port, RegimeCausalPreparation
                    )
                    if (
                        task_id != f"regime.prepare.{causal_value.root}"
                        or causal_value.root in qual_causal
                    ):
                        raise ValueError("C qualification causal root receipt differs")
                    qual_causal[causal_value.root] = causal_value
                elif port.payload_schema == RegimePrivatePreparation.SCHEMA:
                    private_value, task_id = self._read_dependency(
                        context, port, RegimePrivatePreparation
                    )
                    if (
                        task_id != f"regime.prepare.{private_value.root}"
                        or private_value.root in qual_private
                    ):
                        raise ValueError("C qualification private root receipt differs")
                    qual_private[private_value.root] = private_value
                elif port.payload_schema == RegimePredictionSeal.SCHEMA:
                    seal_value, task_id = self._read_dependency(
                        context, port, RegimePredictionSeal
                    )
                    if (
                        task_id != f"regime.seal.{seal_value.root}"
                        or seal_value.root in qual_seals
                    ):
                        raise ValueError("C qualification seal root receipt differs")
                    qual_seals[seal_value.root] = seal_value
                elif port.payload_schema == RegimeAssayPanel.SCHEMA:
                    assay_value, task_id = self._read_dependency(
                        context, port, RegimeAssayPanel
                    )
                    if (
                        task_id != f"regime.assay.{assay_value.root}"
                        or assay_value.root in qual_assays
                    ):
                        raise ValueError("C qualification assay root receipt differs")
                    qual_assays[assay_value.root] = assay_value
                else:
                    raise ValueError(
                        "C qualification received an undeclared scientific schema"
                    )
            if (
                qual_fit is None
                or qual_nominee is None
                or qual_calibration is None
                or qual_fit.design
                != ObjectIdentity.from_record(self.design.config_id, self.design)
                or any(
                    set(items) != set(expected)
                    for items in (qual_causal, qual_private, qual_seals, qual_assays)
                )
            ):
                raise ValueError(
                    "C qualification changed its all-assigned B/C root census"
                )
            output = build_qualification_package(
                qual_fit,
                qual_nominee,
                qual_calibration,
                tuple(
                    (
                        qual_causal[root],
                        qual_private[root],
                        qual_seals[root],
                        qual_assays[root],
                    )
                    for root in expected
                ),
            )
            diagnostics = build_phase_diagnostics(
                "qualification",
                qual_fit,
                qual_nominee,
                tuple(
                    (qual_causal[root], qual_seals[root], qual_assays[root])
                    for root in expected
                ),
            )
            extra_outputs.extend(
                (
                    diagnostics,
                    confirm_regimes(
                        development,
                        cal_diagnostics,
                        diagnostics,
                        output,
                        qual_fit,
                        qual_nominee,
                    ),
                )
            )
            extra_outputs.append(
                evaluate_opportunities(
                    "qualification",
                    tuple(
                        (
                            qual_causal[root],
                            qual_private[root],
                            qual_seals[root],
                            qual_assays[root],
                            preassay[root],
                        )
                        for root in expected
                    ),
                )
            )
        elif context.task_id == "regime.calibration":
            expected = tuple(
                root for root, role, _, _ in ROOTS if role == "calibration"
            )
            if len(scientific) != 2 + 4 * len(expected):
                raise ValueError("calibration lacks B packages or its 32 root quartets")
            fit_package: RegimeFitPackage | None = None
            nominee: RegimeNominationPackage | None = None
            cal_causal: dict[str, RegimeCausalPreparation] = {}
            cal_private: dict[str, RegimePrivatePreparation] = {}
            cal_seals: dict[str, RegimePredictionSeal] = {}
            cal_assays: dict[str, RegimeAssayPanel] = {}
            for port in scientific:
                if port.payload_schema == RegimeFitPackage.SCHEMA:
                    fit_value, task_id = self._read_dependency(
                        context, port, RegimeFitPackage
                    )
                    if fit_package is not None or task_id != "regime.fit":
                        raise ValueError("C calibration changed its fit receipt")
                    fit_package = fit_value
                elif port.payload_schema == RegimeNominationPackage.SCHEMA:
                    nominee_value, task_id = self._read_dependency(
                        context, port, RegimeNominationPackage
                    )
                    if nominee is not None or task_id != "regime.nomination":
                        raise ValueError("C calibration changed its nomination receipt")
                    nominee = nominee_value
                elif port.payload_schema == RegimeCausalPreparation.SCHEMA:
                    cal_value, task_id = self._read_dependency(
                        context, port, RegimeCausalPreparation
                    )
                    if (
                        task_id != f"regime.prepare.{cal_value.root}"
                        or cal_value.root in cal_causal
                    ):
                        raise ValueError("C calibration causal root receipt differs")
                    cal_causal[cal_value.root] = cal_value
                elif port.payload_schema == RegimePrivatePreparation.SCHEMA:
                    private_value, task_id = self._read_dependency(
                        context, port, RegimePrivatePreparation
                    )
                    if (
                        task_id != f"regime.prepare.{private_value.root}"
                        or private_value.root in cal_private
                    ):
                        raise ValueError("C calibration private root receipt differs")
                    cal_private[private_value.root] = private_value
                elif port.payload_schema == RegimePredictionSeal.SCHEMA:
                    seal_value, task_id = self._read_dependency(
                        context, port, RegimePredictionSeal
                    )
                    if (
                        task_id != f"regime.seal.{seal_value.root}"
                        or seal_value.root in cal_seals
                    ):
                        raise ValueError("C calibration seal root receipt differs")
                    cal_seals[seal_value.root] = seal_value
                elif port.payload_schema == RegimeAssayPanel.SCHEMA:
                    assay_value, task_id = self._read_dependency(
                        context, port, RegimeAssayPanel
                    )
                    if (
                        task_id != f"regime.assay.{assay_value.root}"
                        or assay_value.root in cal_assays
                    ):
                        raise ValueError("C calibration assay root receipt differs")
                    cal_assays[assay_value.root] = assay_value
                else:
                    raise ValueError(
                        "C calibration received an undeclared scientific schema"
                    )
            if (
                fit_package is None
                or nominee is None
                or fit_package.design
                != ObjectIdentity.from_record(self.design.config_id, self.design)
                or any(
                    set(items) != set(expected)
                    for items in (cal_causal, cal_private, cal_seals, cal_assays)
                )
            ):
                raise ValueError(
                    "C calibration changed its all-assigned B or root census"
                )
            output = build_calibration_package(
                fit_package,
                nominee,
                tuple(
                    (
                        cal_causal[root],
                        cal_private[root],
                        cal_seals[root],
                        cal_assays[root],
                    )
                    for root in expected
                ),
            )
            extra_outputs.append(
                build_phase_diagnostics(
                    "calibration",
                    fit_package,
                    nominee,
                    tuple(
                        (cal_causal[root], cal_seals[root], cal_assays[root])
                        for root in expected
                    ),
                )
            )
            extra_outputs.append(
                evaluate_opportunities(
                    "calibration",
                    tuple(
                        (
                            cal_causal[root],
                            cal_private[root],
                            cal_seals[root],
                            cal_assays[root],
                            preassay[root],
                        )
                        for root in expected
                    ),
                )
            )
        elif context.task_id == "regime.nomination":
            expected = tuple(root for root, role, _, _ in ROOTS if role == "nomination")
            training_ports = [
                port
                for port in scientific
                if port.payload_schema == RegimeDiscoveryTraining.SCHEMA
            ]
            if len(training_ports) != 1:
                raise ValueError(
                    "nomination lacks the authenticated fit-only discovery rows"
                )
            training, training_task = self._read_dependency(
                context, training_ports[0], RegimeDiscoveryTraining
            )
            if training_task != "regime.fit":
                raise ValueError("discovery rows came from another task")
            scientific = [port for port in scientific if port not in training_ports]
            if len(scientific) != 1 + 4 * len(expected):
                raise ValueError(
                    "nomination lacks its fit package and 16 complete root quartets"
                )
            package: RegimeFitPackage | None = None
            nom_causal: dict[str, RegimeCausalPreparation] = {}
            nom_private: dict[str, RegimePrivatePreparation] = {}
            nom_seals: dict[str, RegimePredictionSeal] = {}
            nom_assays: dict[str, RegimeAssayPanel] = {}
            for port in scientific:
                if port.payload_schema == RegimeFitPackage.SCHEMA:
                    fit_item, task_id = self._read_dependency(
                        context, port, RegimeFitPackage
                    )
                    if package is not None or task_id != "regime.fit":
                        raise ValueError("nomination has another fit package receipt")
                    package = fit_item
                elif port.payload_schema == RegimeCausalPreparation.SCHEMA:
                    causal_item, task_id = self._read_dependency(
                        context, port, RegimeCausalPreparation
                    )
                    if (
                        task_id != f"regime.prepare.{causal_item.root}"
                        or causal_item.root in nom_causal
                    ):
                        raise ValueError("nomination causal receipt/root differs")
                    nom_causal[causal_item.root] = causal_item
                elif port.payload_schema == RegimePrivatePreparation.SCHEMA:
                    private_item, task_id = self._read_dependency(
                        context, port, RegimePrivatePreparation
                    )
                    if (
                        task_id != f"regime.prepare.{private_item.root}"
                        or private_item.root in nom_private
                    ):
                        raise ValueError("nomination private receipt/root differs")
                    nom_private[private_item.root] = private_item
                elif port.payload_schema == RegimePredictionSeal.SCHEMA:
                    seal_item, task_id = self._read_dependency(
                        context, port, RegimePredictionSeal
                    )
                    if (
                        task_id != f"regime.seal.{seal_item.root}"
                        or seal_item.root in nom_seals
                    ):
                        raise ValueError("nomination seal receipt/root differs")
                    nom_seals[seal_item.root] = seal_item
                elif port.payload_schema == RegimeAssayPanel.SCHEMA:
                    assay_item, task_id = self._read_dependency(
                        context, port, RegimeAssayPanel
                    )
                    if (
                        task_id != f"regime.assay.{assay_item.root}"
                        or assay_item.root in nom_assays
                    ):
                        raise ValueError("nomination assay receipt/root differs")
                    nom_assays[assay_item.root] = assay_item
                else:
                    raise ValueError(
                        "nomination received an undeclared scientific schema"
                    )
            if (
                package is None
                or package.design
                != ObjectIdentity.from_record(self.design.config_id, self.design)
                or any(
                    set(items) != set(expected)
                    for items in (nom_causal, nom_private, nom_seals, nom_assays)
                )
            ):
                raise ValueError("nomination changed its all-assigned fit/root census")
            output = build_nomination_package(
                package,
                tuple(
                    (
                        nom_causal[root],
                        nom_private[root],
                        nom_seals[root],
                        nom_assays[root],
                    )
                    for root in expected
                ),
            )
            extra_outputs.append(
                build_discovery_development(
                    training,
                    package,
                    output,
                    tuple(
                        row
                        for root in expected
                        for row in measured_contexts(nom_causal[root], nom_assays[root])
                    ),
                )
            )
            extra_outputs.append(
                build_phase_diagnostics(
                    "nomination",
                    package,
                    output,
                    tuple(
                        (nom_causal[root], nom_seals[root], nom_assays[root])
                        for root in expected
                    ),
                )
            )
            extra_outputs.append(
                evaluate_opportunities(
                    "nomination",
                    tuple(
                        (
                            nom_causal[root],
                            nom_private[root],
                            nom_seals[root],
                            nom_assays[root],
                            preassay[root],
                        )
                        for root in expected
                    ),
                    output,
                )
            )
        elif context.task_id.startswith("regime.seal."):
            root = context.task_id.removeprefix("regime.seal.")
            assigned = [(role, seed) for name, role, _, seed in ROOTS if name == root]
            if len(assigned) != 1 or assigned[0][0] not in (
                "fit",
                "nomination",
                "calibration",
                "qualification",
                "prospective",
            ):
                raise ValueError("seal task root/role differs from B/C allocation")
            prep_ports = [
                port
                for port in scientific
                if port.payload_schema == RegimeCausalPreparation.SCHEMA
            ]
            expected_count = {
                "fit": 1,
                "nomination": 2,
                "calibration": 3,
                "qualification": 4,
                "prospective": 5,
            }[assigned[0][0]]
            if len(prep_ports) != 1 or len(scientific) != expected_count:
                raise ValueError("seal task input census differs")
            preparation, task_id = self._read_dependency(
                context, prep_ports[0], RegimeCausalPreparation
            )
            if task_id != f"regime.prepare.{root}" or (
                preparation.root,
                preparation.role,
                preparation.seed,
            ) != (root, *assigned[0]):
                raise ValueError("seal task changed its causal preparation")
            screen_nomination = None
            screen_calibration = None
            if assigned[0][0] == "fit":
                output = seal_fit_acquisition(preparation)
            else:
                package_ports = [
                    port
                    for port in scientific
                    if port.payload_schema == RegimeFitPackage.SCHEMA
                ]
                if len(package_ports) != 1:
                    raise ValueError("nomination seal lacks the persisted fit package")
                package, fit_task = self._read_dependency(
                    context, package_ports[0], RegimeFitPackage
                )
                if (
                    fit_task != "regime.fit"
                    or package.design
                    != ObjectIdentity.from_record(self.design.config_id, self.design)
                ):
                    raise ValueError(
                        "nomination seal substituted its fit-root model package"
                    )
                if assigned[0][0] == "nomination":
                    output = seal_nomination_predictions(preparation, package)
                else:
                    nomination_ports = [
                        port
                        for port in scientific
                        if port.payload_schema == RegimeNominationPackage.SCHEMA
                    ]
                    if len(nomination_ports) != 1:
                        raise ValueError("C seal lacks the persisted B nomination")
                    nomination, nomination_task = self._read_dependency(
                        context, nomination_ports[0], RegimeNominationPackage
                    )
                    if nomination_task != "regime.nomination":
                        raise ValueError("C seal changed its nomination task receipt")
                    screen_nomination = nomination
                    if assigned[0][0] == "qualification":
                        cal_ports = [
                            port
                            for port in scientific
                            if port.payload_schema == RegimeCalibrationPackage.SCHEMA
                        ]
                        if len(cal_ports) != 1:
                            raise ValueError(
                                "qualification decisions lack frozen calibration"
                            )
                        screen_calibration, cal_task = self._read_dependency(
                            context, cal_ports[0], RegimeCalibrationPackage
                        )
                        if cal_task != "regime.calibration":
                            raise ValueError(
                                "qualification decision calibration receipt changed"
                            )
                    if assigned[0][0] == "prospective":
                        qualification_ports = [
                            port
                            for port in scientific
                            if port.payload_schema
                            == RegimeQualificationPackage.SCHEMA
                        ]
                        if len(qualification_ports) != 1:
                            raise ValueError(
                                "D seal lacks the released C qualification"
                            )
                        qualification, qualification_task = self._read_dependency(
                            context,
                            qualification_ports[0],
                            RegimeQualificationPackage,
                        )
                        if qualification_task != "regime.qualification":
                            raise ValueError(
                                "D seal changed its C qualification receipt"
                            )
                        law_ports = [
                            port
                            for port in scientific
                            if port.payload_schema == RegimeJointLawResult.SCHEMA
                        ]
                        if len(law_ports) != 1:
                            raise ValueError("D seal lacks the supported law qualification owner law")
                        law, law_task = self._read_dependency(
                            context, law_ports[0], RegimeJointLawResult
                        )
                        if law_task != "regime.law-qualification":
                            raise ValueError(
                                "D seal changed its law qualification qualification receipt"
                            )
                        output = seal_prospective_predictions(
                            preparation, package, nomination, qualification, law
                        )
                    else:
                        output = seal_confirmation_predictions(
                            preparation, package, nomination
                        )
            if assigned[0][0] != "prospective":
                if self.source is None or self.references is None:
                    raise ValueError(
                        "preassay comparator source or retained operators unavailable"
                    )
                extra_outputs.append(
                    seal_preassay_readout(
                        self.source.batch,
                        self.references,
                        preparation,
                        output,
                        screen_nomination,
                        screen_calibration,
                    )
                )
        else:
            raise ValueError("undeclared reactor B method task")
        records = (output, *extra_outputs)
        by_schema = {record.SCHEMA: record for record in records}
        if (
            len(by_schema) != len(records)
            or len(context.output_ports) != len(records)
            or {port.payload_schema for port in context.output_ports} != set(by_schema)
        ):
            raise ValueError("reactor method output schema census differs")
        return RunnerResult(
            tuple(
                TaskOutputPayload(
                    port.output_id, by_schema[port.payload_schema].canonical_bytes()
                )
                for port in context.output_ports
            ),
            (ReceiptCheck("reactor-regime-fit-only-causal-seal", True, ()),),
        )


class RegimeMethodProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        design: ReactorRegimeResponseDesign,
        custody: DependencyCustodyReader,
        limits: EmpiricalResourceGuard,
        publisher: CandidatePayloadPublisher,
        reader: CandidatePayloadReader,
        prior_artifacts: tuple[RegimePriorArtifact, ...] = (),
        d_binding: RegimeDMethodBinding | None = None,
        source: EmpiricalStudySource | None = None,
        d_custody: RegimeDMethodCustodyPort | None = None,
        references: RegimeReferenceComparators | None = None,
        retained: RegimeRetainedInputsPort = NoRegimeRetainedInputs(),
    ) -> None:
        if (
            registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version)
            != CAPABILITY
        ):
            raise ValueError("reactor method registry differs")
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self.design = design
        self.prior_artifacts = prior_artifacts
        self.retained = retained
        self.runner = RegimeMethodRunner(
            design,
            custody,
            limits,
            publisher,
            reader,
            prior_artifacts,
            d_binding,
            source,
            d_custody,
            references,
            retained,
        )

    def runners(
        self,
        registry: CapabilityRegistry,
        source_records: tuple[CanonicalRecord, ...] = (),
    ) -> tuple[TaskRunner, ...]:
        if registry.fingerprint() != self.registry_sha256 or source_records:
            raise ValueError("reactor method registry/source differs")
        return (self.runner,)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("reactor method plan/source differs")
        specs = {
            spec.logical_artifact_id: spec
            for task in plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
            for spec in task.external_inputs
        }
        design_specs = [
            value
            for value in specs.values()
            if value.expected_payload_schema == self.design.SCHEMA
        ]
        if len(design_specs) != 1 or len(specs) != 1 + len(self.prior_artifacts) + len(
            retained_input_map(self.retained)
        ):
            raise ValueError("reactor method frozen design/prior input census differs")
        spec = design_specs[0]
        if (
            spec.expected_payload_schema != self.design.SCHEMA
            or spec.expected_content_sha256 != self.design.fingerprint()
        ):
            raise ValueError("reactor method design input differs")
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.design.config_id, self.design),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        prior_payloads = []
        for prior in self.prior_artifacts:
            logical = prior.manifest.logical
            matched = specs.get(logical.logical_artifact_id)
            if (
                matched is None
                or matched.expected_content_sha256 != logical.content_sha256
                or matched.expected_payload_schema != logical.payload_schema
                or matched.expected_size_bytes not in (None, len(prior.payload))
            ):
                raise ValueError("reactor C prior package input differs from B receipt")
            prior_parent = ArtifactLineageParent(
                ObjectIdentity.from_record(prior.receipt.receipt_id, prior.receipt),
                logical.visibility_ceiling,
                logical.outcome_access,
            )
            prior_payloads.append(
                ExternalInputPayload.from_bytes(
                    logical_artifact_id=logical.logical_artifact_id,
                    payload_schema=logical.payload_schema,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type=logical.media_type,
                    payload=prior.payload,
                    visibility_ceiling=logical.visibility_ceiling,
                    outcome_access=logical.outcome_access,
                    parent_visibility_ceilings=(prior_parent.visibility_ceiling,),
                    lineage_parents=(prior_parent,),
                    logical_content_sha256=logical.content_sha256,
                )
            )
        return tuple(
            sorted(
                (
                    ExternalInputPayload.from_bytes(
                        logical_artifact_id=spec.logical_artifact_id,
                        payload_schema=self.design.SCHEMA,
                        profile=ArtifactProfile.CANONICAL_JSON,
                        media_type="application/json",
                        payload=self.design.canonical_bytes(),
                        visibility_ceiling=parent.visibility_ceiling,
                        outcome_access=parent.outcome_access,
                        parent_visibility_ceilings=(parent.visibility_ceiling,),
                        lineage_parents=(parent,),
                        logical_content_sha256=self.design.fingerprint(),
                    ),
                    *prior_payloads,
                    *retained_payloads(plan, self.retained),
                ),
                key=lambda value: value.logical_artifact_id,
            )
        )

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self.runners(registry)
        return registered_output_contracts()

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract | None:
        self.runners(registry)
        if execution_plan is None:
            raise ValueError("reactor C adjudication requires its full compiled plan")
        outputs = [
            output
            for task in execution_plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
            for output in task.outputs
            if output.payload_schema == ScientificAdjudicationRecord.SCHEMA
        ]
        if not outputs:
            return None
        if len(outputs) != 1:
            raise ValueError("reactor C has multiple scientific adjudication outputs")
        return ScientificAdjudicationOutputContract(
            CAPABILITY.capability_key,
            CAPABILITY.capability_version,
            outputs[0].output_id,
            ScientificAdjudicationRecord.SCHEMA,
        )


def registered_output_contracts() -> tuple[CapabilityOutputSemanticContract, ...]:
    """Static record validators shared by execution and authenticated prior reads."""
    return tuple(
        CapabilityOutputSemanticContract.from_manifest(
            CAPABILITY,
            payload_schema=record.SCHEMA,
            profile=ArtifactProfile.CANONICAL_JSON,
            record_version=record.VERSION,
            top_level_keys=("schema", "value", "version"),
            value_keys=tuple(sorted(field.name for field in fields(record))),
        )
        for record in (
            RegimePredictionSeal,
            RegimeFitPackage,
            RegimeDiscoveryTraining,
            RegimeDiscoveryDevelopment,
            RegimePhaseDiagnostics,
            RegimeDiscoveryConfirmation,
            RegimePreassayReadout,
            RegimeOpportunityReadout,
            RegimeNominationPackage,
            RegimeCalibrationPackage,
            RegimeQualificationPackage,
            RegimeJointLawResult,
            CausalValidityRegimeAssignment,
            ReactorRegimeResponsePreparedProspectivePlanBundle,
            RegimeDRootSealResult,
            RegimeDRootRevealResult,
            ReactorRegimeResponseCohortProspective,
            ScientificAdjudicationRecord,
        )
    )

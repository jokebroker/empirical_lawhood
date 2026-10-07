"""Installed qualification runner; consumes authenticated raw operands, never acquires."""

from empirical_lawhood.runtime.task_records import (
    DependencyCustodyReader as DependencyCustodyReader,
    validate_dependency as validate_dependency,
)

from dataclasses import fields
import json
from contextlib import nullcontext
from .resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.law_assessment import (
    CandidatePayloadPublisher,
    CandidatePayloadReader,
)
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import (
    ArtifactProfile,
    ArtifactLineageParent,
    ReceiptCheck,
)
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    TaskContext,
    RunnerResult,
    TaskOutputPayload,
    TaskRunner,
    WorkerInputKind,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    ExternalInputPayload,
    CapabilityOutputSemanticContract,
)
from empirical_lawhood.runtime.adjudication import (
    ScientificAdjudicationRecord,
    ScientificAdjudicationOutputContract,
    AdjudicationEvaluability,
)
from empirical_lawhood.kernel.status import ScientificStatus, AdmissionStatus
from .science import empirical_system
from .config import EmpiricalRecipe, NATIVE_BENCHMARK_ARMS
from .payload import DevelopmentBoundReactorResponsePayload
from .transport import CausalReactorRootEnvelope
from .serialization import read_fit
from .records import build_operands
from .terminal import qualify_empirical, EmpiricalQualificationResult
from .experiment_records import EmpiricalAcquisitionEnvelope, EmpiricalDiscoveryEnvelope, EmpiricalCalibrationEnvelope, EmpiricalQualificationTerminal
from .extension_bundle import CAPABILITY

from .campaign_records import EmpiricalStudySource, EmpiricalRootAnalysis, EmpiricalContributionResult, EmpiricalNativeBenchmarkResult
from .campaign_runner import CAMPAIGN_TASKS, run_campaign_method
from .native_benchmark import NativeVerifierPort
from empirical_lawhood.adapters.simulators.reactor_causal_response.campaign import ControlCustodyPort


class EmpiricalQualificationRunner:
    manifest = CAPABILITY

    def __init__(
        self,
        config: EmpiricalRecipe,
        publisher: CandidatePayloadPublisher,
        reader: CandidatePayloadReader,
        custody: DependencyCustodyReader,
        control: ControlCustodyPort | None = None,
        verifier: NativeVerifierPort | None = None,
        source: ExternalInputPayload | None = None,
        limits: EmpiricalResourceGuard | None = None,
    ) -> None:
        self.config, self.publisher, self.reader, self.custody = config, publisher, reader, custody
        self.control, self.verifier = control, verifier
        self.limits = limits

    def execute(self, context: TaskContext) -> RunnerResult:
        with self.limits.task(context.task_id) if self.limits is not None else nullcontext():
            return self._execute(context)

    def _execute(self, context: TaskContext) -> RunnerResult:
        if context.task_id in CAMPAIGN_TASKS:
            if (
                self.control is None
                or self.verifier is None
                or context.permissions != self.manifest.permissions
                or len(context.output_ports) != 1
            ):
                raise ValueError("campaign requires its closed control/verifier ports")
            return self._output(
                context,
                run_campaign_method(
                    context, self.config, self.custody, self.control, self.verifier
                ),
                "empirical-existing-owner-prospective-controller-evaluation-and-paired-root-census",
            )
        if (
            context.task_id
            not in (
                "empirical.discovery",
                "empirical.calibration",
                "empirical.qualification",
                "empirical.adjudication",
            )
            or context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != self.manifest.permissions
            or context.outcome_access is not OutcomeAccess.EVALUATOR_REVEAL
            or len(context.output_ports) != 1
            or len(context.input_ports)
            != {
                "empirical.discovery": 25,
                "empirical.calibration": 34,
                "empirical.qualification": 66,
                "empirical.adjudication": len(context.input_ports)
                if len(context.input_ports) in (2, 3 + len(NATIVE_BENCHMARK_ARMS))
                else 0,
            }.get(context.task_id)
        ):
            raise ValueError(
                "empirical qualification identity/permissions/full input census differs"
            )
        if context.task_id in ("empirical.discovery", "empirical.calibration"):
            return self._development(context)
        if context.task_id == "empirical.adjudication":
            revealed_result = self._reveal(context)
            if context.output_ports[0].payload_schema != revealed_result.SCHEMA:
                raise ValueError("empirical adjudication output differs")
            return RunnerResult(
                (
                    TaskOutputPayload(
                        context.output_ports[0].output_id, revealed_result.canonical_bytes()
                    ),
                ),
                (ReceiptCheck("empirical-sole-owner-revealed_result-reveal", True, ()),),
            )
        production = any(
            p.payload_schema == EmpiricalCalibrationEnvelope.SCHEMA for p in context.input_ports
        )
        parent_schema = (
            EmpiricalCalibrationEnvelope.SCHEMA if production else DevelopmentBoundReactorResponsePayload.SCHEMA
        )
        by_schema = {
            s: tuple(p for p in context.input_ports if p.payload_schema == s)
            for s in (
                EmpiricalRecipe.SCHEMA,
                parent_schema,
                CausalReactorRootEnvelope.SCHEMA,
            )
        }
        if tuple(map(len, by_schema.values())) != (1, 1, 64):
            raise ValueError(
                "empirical qualification requires recipe, frozen model and 64 raw roots"
            )
        config_port = by_schema[EmpiricalRecipe.SCHEMA][0]
        if (
            config_port.outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or decode_canonical_bytes(config_port.read(), EmpiricalRecipe, maximum_bytes=32768)
            != self.config
        ):
            raise ValueError("empirical recipe changed")
        model_port = by_schema[parent_schema][0]
        if model_port.kind is not WorkerInputKind.DEPENDENCY:
            raise ValueError("frozen model must come from authenticated previous-phase dependency")
        model_manifest, model_receipt = self.custody.read_dependency(context, model_port.binding)
        validate_dependency(model_port.binding, model_manifest, model_receipt)
        calibration = (
            decode_canonical_bytes(
                model_port.read(), EmpiricalCalibrationEnvelope, maximum_bytes=4 * 1024**2
            )
            if production
            else None
        )
        payload = (
            calibration.payload
            if calibration is not None
            else decode_canonical_bytes(
                model_port.read(), DevelopmentBoundReactorResponsePayload, maximum_bytes=2 * 1024**2
            )
        )
        parent = calibration if calibration is not None else payload
        assert parent is not None
        if parent.fingerprint() != model_manifest.logical.content_sha256:
            raise ValueError("frozen model custody differs")
        roots, manifests, receipts, identities = [], [], [], []
        by_root = {}
        for port in by_schema[CausalReactorRootEnvelope.SCHEMA]:
            if port.kind is not WorkerInputKind.DEPENDENCY or port.outcome_access is not (
                OutcomeAccess.EVALUATOR_REVEAL if production else OutcomeAccess.EVALUATION_SEALED
            ):
                raise ValueError("raw empirical root must retain its private phase lineage")
            manifest, receipt = self.custody.read_dependency(context, port.binding)
            validate_dependency(port.binding, manifest, receipt)
            envelope = decode_canonical_bytes(
                port.read(), CausalReactorRootEnvelope, maximum_bytes=256 * 1024**2
            )
            if (
                envelope.root in by_root
                or envelope.fingerprint() != manifest.logical.content_sha256
            ):
                raise ValueError("duplicate/substituted raw root")
            by_root[envelope.root] = envelope, manifest, receipt
        expected = tuple(
            f"reactor-empirical-{role}-{r:03d}"
            for role in ("calibration", "qualification")
            for r in range(32)
        )
        if set(by_root) != set(expected):
            raise ValueError("assigned empirical roles differ")
        for root in expected:
            envelope, manifest, receipt = by_root[root]
            roots.append(envelope.unpack())
            manifests.append(manifest)
            receipts.append(receipt)
            identities.append(ObjectIdentity.from_record(root, envelope))
        if calibration is not None and calibration.roots != tuple(identities[:32]):
            raise ValueError("qualification changed calibration roots")
        if calibration is not None and (payload is None or payload.q is None):
            reason = "IDENTIFICATION_NOT_SUPPORTED" if payload is None else "CALIBRATION_UNUSABLE"
            if any(by_root[r][0].nonattempt_reason != reason for r in expected[32:]):
                raise ValueError("unqualified policy acquired downstream qualification roots")
            terminal = EmpiricalQualificationTerminal(
                ObjectIdentity.from_record(self.config.config_id, self.config),
                ObjectIdentity.from_record("reactor-empirical-calibration", calibration),
                tuple(identities),
                None,
                reason,
            )
            return self._output(context, terminal, "empirical-prerequisite-nonentry")
        assert payload is not None
        model = read_fit(json.loads(payload.model_json))
        operands = build_operands(
            tuple(roots[:32]),
            tuple(roots[32:]),
            model,
            tuple(identities),
            development_q=float(payload.development_q)
            if payload.development_q is not None
            else float("nan"),
        )
        # Qualification must use the already frozen q, never silently recalibrate deployment.
        if operands.q != payload.q or operands.calibration_digest != payload.calibration_sha256:
            # Calibration is bound only to its own raw roots, never qualification labels.
            raise ValueError("frozen calibration differs or depends on future qualification")
        design_artifact = ArtifactIdentity(
            config_port.artifact_id,
            "empirical-recipe",
            self.config.SCHEMA,
            self.config.fingerprint(),
            "application/json",
            len(self.config.canonical_bytes()),
        )
        result = qualify_empirical(
            self.config,
            operands,
            tuple(manifests),
            tuple(receipts),
            model,
            CAPABILITY,
            self.publisher,
            self.reader,
            design_artifact,
        )
        published: CanonicalRecord = result
        if calibration is not None:
            from .pipeline import policy_usefulness

            published = EmpiricalQualificationTerminal(
                ObjectIdentity.from_record(self.config.config_id, self.config),
                ObjectIdentity.from_record("reactor-empirical-calibration", calibration),
                tuple(identities),
                result,
                None,
                policy_usefulness(tuple(by_root[r][0] for r in expected[32:]), payload),
            )
        return self._output(context, published, "empirical-raw-operands-sole-qualification-owner")

    @staticmethod
    def _output(context: TaskContext, value: CanonicalRecord, check: str) -> RunnerResult:
        if context.output_ports[0].payload_schema != value.SCHEMA:
            raise ValueError("empirical method output schema differs")
        return RunnerResult(
            (TaskOutputPayload(context.output_ports[0].output_id, value.canonical_bytes()),),
            (ReceiptCheck(check, True, ()),),
        )

    def _development(self, context: TaskContext) -> RunnerResult:
        from .pipeline import discovery_from_acquisitions, calibration_from_roots

        recipes = [p for p in context.input_ports if p.payload_schema == self.config.SCHEMA]
        if (
            len(recipes) != 1
            or recipes[0].outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or decode_canonical_bytes(recipes[0].read(), EmpiricalRecipe, maximum_bytes=32768)
            != self.config
        ):
            raise ValueError("development recipe differs")
        acquisitions, roots, discoveries = [], [], []
        for port in context.input_ports:
            if port is recipes[0]:
                continue
            if port.kind is not WorkerInputKind.DEPENDENCY:
                raise ValueError("development evidence must be an authenticated dependency")
            manifest, receipt = self.custody.read_dependency(context, port.binding)
            validate_dependency(port.binding, manifest, receipt)
            record: CanonicalRecord
            if port.payload_schema == EmpiricalAcquisitionEnvelope.SCHEMA:
                record = decode_canonical_bytes(
                    port.read(), EmpiricalAcquisitionEnvelope, maximum_bytes=256 * 1024**2
                )
                acquisitions.append(record)
            elif port.payload_schema == CausalReactorRootEnvelope.SCHEMA:
                record = decode_canonical_bytes(
                    port.read(), CausalReactorRootEnvelope, maximum_bytes=256 * 1024**2
                )
                roots.append(record)
            elif port.payload_schema == EmpiricalDiscoveryEnvelope.SCHEMA:
                record = decode_canonical_bytes(
                    port.read(), EmpiricalDiscoveryEnvelope, maximum_bytes=256 * 1024**2
                )
                discoveries.append(record)
            else:
                raise ValueError("unexpected development input schema")
            if record.fingerprint() != manifest.logical.content_sha256:
                raise ValueError("development input custody differs")
        if context.task_id == "empirical.discovery":
            if roots or discoveries or len(acquisitions) != 24:
                raise ValueError("discovery dependency census differs")
            value: CanonicalRecord = discovery_from_acquisitions(self.config, tuple(acquisitions))
        else:
            if acquisitions or len(discoveries) != 1 or len(roots) != 32:
                raise ValueError("calibration dependency census differs")
            value = calibration_from_roots(self.config, discoveries[0], tuple(roots))
        return self._output(context, value, "empirical-frozen-development-census")

    def _reveal(self, context: TaskContext) -> ScientificAdjudicationRecord:
        ports = tuple(
            p
            for p in context.input_ports
            if p.payload_schema
            in (EmpiricalQualificationResult.SCHEMA, EmpiricalQualificationTerminal.SCHEMA)
        )
        if (
            len(context.input_ports) not in (2, 3 + len(NATIVE_BENCHMARK_ARMS))
            or len(ports) != 1
            or ports[0].kind is not WorkerInputKind.DEPENDENCY
            or ports[0].size_bytes > 16 * 1024**2
        ):
            raise ValueError("forecast reveal requires its single exact qualification parent")
        manifest, receipt = self.custody.read_dependency(context, ports[0].binding)
        validate_dependency(ports[0].binding, manifest, receipt)
        recipes = tuple(
            p for p in context.input_ports if p.payload_schema == EmpiricalRecipe.SCHEMA
        )
        if (
            len(recipes) != 1
            or recipes[0].outcome_access is not OutcomeAccess.OUTCOME_BLIND
            or decode_canonical_bytes(recipes[0].read(), EmpiricalRecipe, maximum_bytes=32768)
            != self.config
        ):
            raise ValueError("adjudication recipe changed")
        terminal = (
            decode_canonical_bytes(
                ports[0].read(), EmpiricalQualificationTerminal, maximum_bytes=16 * 1024**2
            )
            if ports[0].payload_schema == EmpiricalQualificationTerminal.SCHEMA
            else None
        )
        result = (
            terminal.result
            if terminal is not None
            else decode_canonical_bytes(
                ports[0].read(), EmpiricalQualificationResult, maximum_bytes=16 * 1024**2
            )
        )
        parent = terminal if terminal is not None else result
        assert parent is not None
        if (
            receipt.task_id != "empirical.qualification"
            or (
                terminal.recipe
                if terminal is not None
                else result.design
                if result is not None
                else None
            )
            != ObjectIdentity.from_record(self.config.config_id, self.config)
            or parent.fingerprint() != manifest.logical.content_sha256
            or manifest.logical.visibility_ceiling is not VisibilityCeiling.PROSPECTIVE
        ):
            raise ValueError("forecast reveal substitutes its prospective qualified parent")
        extras = []
        for port in context.input_ports:
            if port in (*ports, *recipes):
                continue
            t = (
                EmpiricalContributionResult
                if port.payload_schema == EmpiricalContributionResult.SCHEMA
                else EmpiricalNativeBenchmarkResult
                if port.payload_schema == EmpiricalNativeBenchmarkResult.SCHEMA
                else None
            )
            if t is None or port.kind is not WorkerInputKind.DEPENDENCY:
                raise ValueError("adjudication has undeclared comparative/native evidence")
            m, r = self.custody.read_dependency(context, port.binding)
            validate_dependency(port.binding, m, r)
            value = decode_canonical_bytes(port.read(), t, maximum_bytes=256 * 1024**2)
            if r.run_id != context.run_id or value.fingerprint() != m.logical.content_sha256:
                raise ValueError("comparative/native evidence custody differs")
            extras.append(value)
        if extras:
            from empirical_lawhood.adapters.simulators.reactor_causal_response.campaign import prerequisite

            assert terminal is not None
            contributions = [v for v in extras if isinstance(v, EmpiricalContributionResult)]
            native = [v for v in extras if isinstance(v, EmpiricalNativeBenchmarkResult)]
            if (
                len(contributions) != 1
                or sorted(v.arm for v in native) != sorted(NATIVE_BENCHMARK_ARMS)
                or any(v.prerequisite != prerequisite(terminal) for v in contributions)
                or any(v.prerequisite != prerequisite(terminal) for v in native)
            ):
                raise ValueError("final study census or prerequisite disposition differs")
        a = context.scientific_adjudication_context
        system = empirical_system(self.config)
        if (
            a is None
            or a.relation
            != ObjectIdentity.from_record(system.relation.relation_id, system.relation)
            or a.independent_unit_id != system.independent_unit.unit_id
        ):
            raise ValueError("forecast reveal requires its compiled scientific context")
        status = (
            ScientificStatus.UNEVALUABLE
            if result is None
            else result.qualification.scientific_status
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
            admission_status=AdmissionStatus.UNEVALUABLE
            if status is ScientificStatus.UNEVALUABLE
            else AdmissionStatus.NOT_EVALUATED,
            reason_codes=("REACTOR_FORECAST_QUALIFICATION_EXISTING_OWNER_RESULT",)
            if result is not None
            else ("REACTOR_FORECAST_PREREQUISITE_NONENTRY",),
        )


class EmpiricalProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        config: EmpiricalRecipe,
        publisher: CandidatePayloadPublisher,
        reader: CandidatePayloadReader,
        custody: DependencyCustodyReader,
        control: ControlCustodyPort | None = None,
        verifier: NativeVerifierPort | None = None,
        source: ExternalInputPayload | None = None,
        limits: EmpiricalResourceGuard | None = None,
    ) -> None:
        if registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version) != CAPABILITY:
            raise ValueError("empirical installed capability differs")
        self.registry_sha256 = registry.fingerprint()
        self.capability_count = 1
        self.config, self.source = config, source
        self.runner = EmpiricalQualificationRunner(
            config, publisher, reader, custody, control, verifier, limits=limits
        )

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry.fingerprint() != self.registry_sha256 or source_records:
            raise ValueError("empirical method never opens sources")
        return (self.runner,)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if source_records or plan.registry_sha256 != self.registry_sha256:
            raise ValueError("empirical source/registry differs")
        outputs = {}
        for task in plan.tasks:
            if task.capability.capability_key != CAPABILITY.capability_key:
                continue
            for spec in task.external_inputs:
                if spec.expected_payload_schema == EmpiricalStudySource.SCHEMA:
                    if (
                        self.source is None
                        or self.source.logical_artifact_id != spec.logical_artifact_id
                        or self.source.logical_content_sha256 != spec.expected_content_sha256
                    ):
                        raise ValueError("campaign source external input differs")
                    outputs[spec.logical_artifact_id] = self.source
                    continue
                if (
                    spec.expected_payload_schema != self.config.SCHEMA
                    or spec.expected_content_sha256 != self.config.fingerprint()
                ):
                    raise ValueError("empirical external input must be exact unissued recipe")
                parent = ArtifactLineageParent(
                    ObjectIdentity.from_record(spec.logical_artifact_id, self.config),
                    VisibilityCeiling.PROSPECTIVE,
                    OutcomeAccess.OUTCOME_BLIND,
                )
                outputs[spec.logical_artifact_id] = ExternalInputPayload.from_bytes(
                    logical_artifact_id=spec.logical_artifact_id,
                    payload_schema=self.config.SCHEMA,
                    profile=ArtifactProfile.CANONICAL_JSON,
                    media_type="application/vnd.empirical-lawhood.canonical+json",
                    payload=self.config.canonical_bytes(),
                    visibility_ceiling=VisibilityCeiling.PROSPECTIVE,
                    outcome_access=OutcomeAccess.OUTCOME_BLIND,
                    parent_visibility_ceilings=(VisibilityCeiling.PROSPECTIVE,),
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
                payload_schema=t.SCHEMA,
                profile=ArtifactProfile.CANONICAL_JSON,
                record_version=t.VERSION,
                top_level_keys=("schema", "value", "version"),
                value_keys=tuple(sorted(f.name for f in fields(t))),
            )
            for t in (
                EmpiricalDiscoveryEnvelope,
                EmpiricalCalibrationEnvelope,
                EmpiricalQualificationTerminal,
                EmpiricalQualificationResult,
                EmpiricalRootAnalysis,
                EmpiricalContributionResult,
                EmpiricalNativeBenchmarkResult,
                ScientificAdjudicationRecord,
            )
        )

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> ScientificAdjudicationOutputContract:
        self.runners(registry)
        if execution_plan is None:
            raise ValueError("exact compiled empirical adjudication locator required")
        outputs = [
            o
            for t in execution_plan.tasks
            if t.capability.capability_key == CAPABILITY.capability_key
            for o in t.outputs
            if o.payload_schema == ScientificAdjudicationRecord.SCHEMA
        ]
        if len(outputs) != 1:
            raise ValueError("one empirical adjudication output required")
        return ScientificAdjudicationOutputContract(
            CAPABILITY.capability_key,
            CAPABILITY.capability_version,
            outputs[0].output_id,
            ScientificAdjudicationRecord.SCHEMA,
        )

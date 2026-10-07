"""Public-runner source binding with a receipt-checked prediction barrier.

The assay runner cannot open the native source until the campaign has
persisted preparation and a separate prediction-seal task.  No predictor
input port receives the private native grid from this provider.
"""

from __future__ import annotations

from dataclasses import fields
from decimal import Decimal as D
from hashlib import sha256
import platform
from typing import Callable, Protocol, TypeVar

import numpy as np

from empirical_lawhood.adapters.methods.reactor_causal_response.campaign_records import EmpiricalStudySource
from empirical_lawhood.adapters.methods.reactor_causal_response.provider import DependencyCustodyReader, validate_dependency
from empirical_lawhood.adapters.methods.reactor_causal_response.resource_contract import EmpiricalResourceGuard
from empirical_lawhood.adapters.methods.reactor_regime_response.config import ROOTS
from empirical_lawhood.adapters.methods.reactor_regime_response.calibration_pipeline import RegimeCalibrationPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.causal_contexts import causal_contexts
from empirical_lawhood.adapters.methods.reactor_regime_response.records import RegimeAssayPanel, RegimeCausalPreparation, RegimePredictionSeal, RegimePrivatePreparation
from empirical_lawhood.adapters.methods.reactor_regime_response.nomination_records import RegimeNominationPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.control_prospective_plan import ReactorRegimeResponsePreparedProspectivePlanBundle
from empirical_lawhood.adapters.methods.reactor_regime_response.control_owner import ControlRecordPublisher
from empirical_lawhood.adapters.methods.reactor_regime_response.prospective_decision import CausalValidityRegimeAssignment
from empirical_lawhood.adapters.methods.law_assessment import CandidatePayloadReader
from empirical_lawhood.adapters.methods.reactor_regime_response.prior import RegimePriorArtifact
from empirical_lawhood.adapters.methods.reactor_regime_response.continuation import NoRegimeRetainedInputs, RegimeRetainedInputsPort, retained_input_map
from empirical_lawhood.adapters.methods.reactor_regime_response.qualification_pipeline import RegimeQualificationPackage
from empirical_lawhood.adapters.methods.reactor_regime_response.release import require_scientific_release
from empirical_lawhood.adapters.methods.reactor_regime_response.law_terminal import RESULT_ID, RegimeJointLawResult
from empirical_lawhood.kernel.decoding import decode_canonical_bytes
from empirical_lawhood.kernel.evidence import OutcomeAccess, VisibilityCeiling
from empirical_lawhood.kernel.provenance import ObjectIdentity
from empirical_lawhood.kernel.references import ArtifactIdentity
from empirical_lawhood.kernel.serialization import CanonicalRecord
from empirical_lawhood.runtime.artifacts import (
    ArtifactLineageParent,
    ArtifactProfile,
    ReceiptCheck,
)
from empirical_lawhood.runtime.controller_evaluation_nested import DurablePreparedExecutionEventStore
from empirical_lawhood.runtime.capabilities import CapabilityRegistry
from empirical_lawhood.runtime.execution import (
    RunnerResult,
    TaskContext,
    TaskOutputPayload,
    TaskProgressEmitter,
    TaskRunner,
    WorkerInputKind,
)
from empirical_lawhood.runtime.plans import ProtocolExecutionPlan
from empirical_lawhood.runtime.providers import (
    CampaignRuntimeProvider,
    CapabilityOutputSemanticContract,
    ExternalInputPayload,
)

from .acquisition import acquire_assays, acquire_preparation
from .config import ReactorRegimeNativeConfig
from .extension_bundle import CAPABILITY
from .prospective_acquisition import acquire_prospective_root
from .prospective_records import RegimeDNativeRoot

RecordT = TypeVar("RecordT", bound=CanonicalRecord)


class RegimeControlCustodyPort(Protocol):
    authority: ObjectIdentity
    resources: ObjectIdentity
    issued_study: ObjectIdentity

    def open_control_store(self, root: str) -> ControlRecordPublisher: ...
    def open_prepared_store(self) -> DurablePreparedExecutionEventStore: ...
    def freeze_clock(self, root: str) -> str: ...


def _task_assignment(task_id: str) -> tuple[str, str]:
    for task_kind in ("prepare", "assay", "action"):
        prefix = f"regime.{task_kind}."
        if task_id.startswith(prefix):
            root = task_id.removeprefix(prefix)
            for assigned_root, role, _, _ in ROOTS:
                if assigned_root == root:
                    if role == "prospective" and task_kind == "assay":
                        raise ValueError("D cannot use the 48-slot B/C assay route")
                    if role != "prospective" and task_kind == "action":
                        raise ValueError("B/C cannot use the D four-request action route")
                    return task_kind, role
    raise ValueError("native task is outside the frozen regime-response study root census")


class RegimeNativeRunner:
    manifest = CAPABILITY

    def __init__(
        self,
        config: ReactorRegimeNativeConfig,
        custody: DependencyCustodyReader,
        limits: EmpiricalResourceGuard,
        prior_artifacts: tuple[RegimePriorArtifact, ...] = (),
        control_custody: RegimeControlCustodyPort | None = None,
        law_reader: CandidatePayloadReader | None = None,
        retained: RegimeRetainedInputsPort = NoRegimeRetainedInputs(),
    ) -> None:
        self.config, self.custody, self.limits = config, custody, limits
        self.control_custody, self.law_reader = control_custody, law_reader
        self.retained_inputs = retained_input_map(retained)
        self.prior_artifacts = {
            value.manifest.logical.payload_schema: value for value in prior_artifacts
        }
        if len(self.prior_artifacts) != len(prior_artifacts):
            raise ValueError("reactor native prior-phase schemas repeat")

    def execute(self, context: TaskContext) -> RunnerResult:
        return self._guarded(context, None)

    def execute_with_progress(
        self, context: TaskContext, emitter: TaskProgressEmitter
    ) -> RunnerResult:
        return self._guarded(context, emitter)

    def _guarded(
        self, context: TaskContext, emitter: TaskProgressEmitter | None
    ) -> RunnerResult:
        with self.limits.task(context.task_id):
            return self._execute(context, emitter)

    def _read_dependency(
        self, context: TaskContext, schema: str, record_type: type[RecordT], task_id: str
    ) -> RecordT:
        ports = [port for port in context.input_ports if port.payload_schema == schema]
        if len(ports) != 1:
            raise ValueError("native assay lacks one persisted dependency")
        port = ports[0]
        if port.kind is WorkerInputKind.EXTERNAL:
            retained = self.retained_inputs.get(port.artifact_id)
            if retained is not None:
                raw = port.read()
                logical = retained.manifest.logical
                if (retained.task_id != task_id or logical.payload_schema != schema
                        or record_type.SCHEMA != schema or port.media_type != logical.media_type
                        or port.size_bytes != retained.manifest.materialization.size_bytes
                        or sha256(raw).hexdigest() != logical.content_sha256):
                    raise ValueError("retained C native operand lost exact original custody")
                return decode_canonical_bytes(raw, record_type, maximum_bytes=512 * 1024**2)
            prior = self.prior_artifacts.get(schema)
            if (
                prior is None
                or prior.expected_task_id != task_id
                or prior.manifest.logical.logical_artifact_id != port.artifact_id
                or prior.manifest.logical.payload_schema != schema
                or prior.manifest.logical.media_type != port.media_type
                or port.size_bytes != len(prior.payload)
                or sha256(port.read()).hexdigest()
                != prior.manifest.logical.content_sha256
            ):
                raise ValueError("native prior-phase package lost exact custody")
            return decode_canonical_bytes(
                prior.payload, record_type, maximum_bytes=512 * 1024**2
            )
        if port.kind is not WorkerInputKind.DEPENDENCY:
            raise ValueError("native scientific input lacks dependency custody")
        manifest, receipt = self.custody.read_dependency(context, port.binding)
        validate_dependency(port.binding, manifest, receipt)
        if receipt.task_id != task_id:
            raise ValueError("native assay dependency task identity differs")
        record = decode_canonical_bytes(
            port.read(), record_type, maximum_bytes=512 * 1024**2
        )
        if record.fingerprint() != manifest.logical.content_sha256:
            raise ValueError("native assay dependency bytes differ from custody")
        return record

    def _acquire_action(
        self,
        context: TaskContext,
        source: EmpiricalStudySource,
        root: str,
        progress: Callable[[int], None] | None,
    ) -> RegimeDNativeRoot:
        if self.control_custody is None or self.law_reader is None:
            raise ValueError("D action lacks installed finite controller custody and C law reader")
        causal = self._read_dependency(
            context, RegimeCausalPreparation.SCHEMA,
            RegimeCausalPreparation, f"regime.prepare.{root}",
        )
        private = self._read_dependency(
            context, RegimePrivatePreparation.SCHEMA,
            RegimePrivatePreparation, f"regime.prepare.{root}",
        )
        assignment = self._read_dependency(
            context, CausalValidityRegimeAssignment.SCHEMA,
            CausalValidityRegimeAssignment, f"regime.assignment.{root}",
        )
        prospective_plan = self._read_dependency(
            context, ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA,
            ReactorRegimeResponsePreparedProspectivePlanBundle, "regime.d-plan",
        )
        nomination = self._read_dependency(
            context, RegimeNominationPackage.SCHEMA,
            RegimeNominationPackage, "regime.nomination",
        )
        calibration = self._read_dependency(
            context, RegimeCalibrationPackage.SCHEMA,
            RegimeCalibrationPackage, "regime.calibration",
        )
        qualification = self._read_dependency(
            context, RegimeQualificationPackage.SCHEMA,
            RegimeQualificationPackage, "regime.qualification",
        )
        law = self._read_dependency(
            context, RegimeJointLawResult.SCHEMA,
            RegimeJointLawResult, "regime.law-qualification",
        )
        assert isinstance(causal, RegimeCausalPreparation)
        assert isinstance(private, RegimePrivatePreparation)
        assert isinstance(assignment, CausalValidityRegimeAssignment)
        assert isinstance(prospective_plan, ReactorRegimeResponsePreparedProspectivePlanBundle)
        assert isinstance(nomination, RegimeNominationPackage)
        assert isinstance(calibration, RegimeCalibrationPackage)
        assert isinstance(qualification, RegimeQualificationPackage)
        assert isinstance(law, RegimeJointLawResult)
        require_scientific_release(nomination, qualification, law)
        if (
            causal.root != root
            or assignment.qualified_law != ObjectIdentity.from_record(RESULT_ID, law)
            or tuple(row.root for row in prospective_plan.assigned_roots)
            != tuple(name for name, role, _, _ in ROOTS if role == "prospective")
            or calibration.fingerprint() != law.payload.calibration.object_fingerprint
        ):
            raise ValueError("D action changed its 64-root controller-use plan or exact C law")
        prior_calibration = self.prior_artifacts.get(RegimeCalibrationPackage.SCHEMA)
        if prior_calibration is None:
            raise ValueError("D action lacks exact C calibration publication")
        logical = prior_calibration.manifest.logical
        calibration_artifact = ArtifactIdentity(
            logical.logical_artifact_id, "reactor-regime-calibration",
            logical.payload_schema, logical.content_sha256,
            logical.media_type, len(prior_calibration.payload),
        )
        causal_ports = [
            port for port in context.input_ports
            if port.payload_schema == RegimeCausalPreparation.SCHEMA
        ]
        if len(causal_ports) != 1 or causal_ports[0].kind is not WorkerInputKind.DEPENDENCY:
            raise ValueError("D action lacks its authenticated prepared parent")
        causal_manifest, causal_receipt = self.custody.read_dependency(
            context, causal_ports[0].binding
        )
        validate_dependency(causal_ports[0].binding, causal_manifest, causal_receipt)
        if (
            causal_receipt.task_id != f"regime.prepare.{root}"
            or causal_manifest.logical.content_sha256 != causal.fingerprint()
        ):
            raise ValueError("D controller use parent preparation receipt changed before action")
        parent_artifact = ArtifactIdentity(
            causal_manifest.logical.logical_artifact_id,
            "reactor-regime-causal-preparation",
            causal_manifest.logical.payload_schema,
            causal_manifest.logical.content_sha256,
            causal_manifest.logical.media_type,
            causal_manifest.materialization.size_bytes,
        )
        result = acquire_prospective_root(
            source=source.batch, causal=causal, private=private,
            assignment=assignment, report=law, prospective_plan=prospective_plan,
            calibration_artifact=calibration_artifact,
            publisher=self.control_custody.open_control_store(root),
            reader=self.law_reader,
            authority=self.control_custody.authority,
            resources=self.control_custody.resources,
            causal_receipt=causal_receipt,
            causal_artifact=parent_artifact,
            issued_study=self.control_custody.issued_study,
            prepared_store=self.control_custody.open_prepared_store(),
            occurred_at_utc=self.control_custody.freeze_clock(root),
            progress=progress,
        )
        plan_identity = None if prospective_plan.plan is None else ObjectIdentity.from_record(
            prospective_plan.plan.evaluation_plan_id, prospective_plan.plan
        )
        return RegimeDNativeRoot.from_acquisition(
            assignment, plan_identity, result
        )

    def _execute(
        self, context: TaskContext, emitter: TaskProgressEmitter | None
    ) -> RunnerResult:
        kind, role = _task_assignment(context.task_id)
        root = context.task_id.removeprefix(f"regime.{kind}.")
        output_schemas = {port.payload_schema for port in context.output_ports}
        expected_outputs = (
            {RegimeCausalPreparation.SCHEMA, RegimePrivatePreparation.SCHEMA}
            if kind == "prepare"
            else {RegimeAssayPanel.SCHEMA} if kind == "assay"
            else {RegimeDNativeRoot.SCHEMA}
        )
        expected_inputs = {ReactorRegimeNativeConfig.SCHEMA, EmpiricalStudySource.SCHEMA}
        if kind == "prepare" and role == "prospective":
            expected_inputs |= {
                RegimeNominationPackage.SCHEMA,
                RegimeQualificationPackage.SCHEMA,
                RegimeJointLawResult.SCHEMA,
            }
        if kind == "assay":
            expected_inputs |= {
                RegimeCausalPreparation.SCHEMA,
                RegimePrivatePreparation.SCHEMA,
                RegimePredictionSeal.SCHEMA,
            }
            if role == "qualification":
                expected_inputs.add(RegimeCalibrationPackage.SCHEMA)
        if kind == "action":
            expected_inputs |= {
                RegimeCausalPreparation.SCHEMA,
                RegimePrivatePreparation.SCHEMA,
                CausalValidityRegimeAssignment.SCHEMA,
                ReactorRegimeResponsePreparedProspectivePlanBundle.SCHEMA,
                RegimeNominationPackage.SCHEMA,
                RegimeCalibrationPackage.SCHEMA,
                RegimeQualificationPackage.SCHEMA,
                RegimeJointLawResult.SCHEMA,
            }
        inputs = {port.payload_schema: port for port in context.input_ports}
        if (
            context.config.content_sha256 != self.config.fingerprint()
            or context.permissions != CAPABILITY.permissions
            or len(inputs) != len(context.input_ports)
            or set(inputs) != expected_inputs
            or len(context.output_ports) != len(expected_outputs)
            or output_schemas != expected_outputs
            or context.outcome_access
            not in (OutcomeAccess.EVALUATION_SEALED, OutcomeAccess.EVALUATOR_REVEAL)
        ):
            raise ValueError("native task configuration, ports or visibility differs")
        config = decode_canonical_bytes(
            inputs[ReactorRegimeNativeConfig.SCHEMA].read(),
            ReactorRegimeNativeConfig,
            maximum_bytes=256 * 1024,
        )
        source = decode_canonical_bytes(
            inputs[EmpiricalStudySource.SCHEMA].read(),
            EmpiricalStudySource,
            maximum_bytes=4 * 1024**2,
        )
        if (
            config != self.config
            or source.batch.fingerprint() != config.source_sha256
            or platform.python_version() != config.python_version
            or np.__version__ != config.numpy_version
        ):
            raise ValueError("native medium, source pin or runtime changed")
        progress = None if emitter is None else lambda value: emitter.advance(D(value))
        if kind == "action":
            outputs: tuple[CanonicalRecord, ...] = (
                self._acquire_action(context, source, root, progress),
            )
        elif kind == "prepare":
            selected_route = None
            if role == "prospective":
                nominee = self._read_dependency(
                    context,
                    RegimeNominationPackage.SCHEMA,
                    RegimeNominationPackage,
                    "regime.nomination",
                )
                qualification = self._read_dependency(
                    context,
                    RegimeQualificationPackage.SCHEMA,
                    RegimeQualificationPackage,
                    "regime.qualification",
                )
                law = self._read_dependency(
                    context,
                    RegimeJointLawResult.SCHEMA,
                    RegimeJointLawResult,
                    "regime.law-qualification",
                )
                require_scientific_release(nominee, qualification, law)
                selected_route = nominee.selected_route
            causal, private = acquire_preparation(
                config.design,
                source.batch,
                config.prepared_domain,
                root,
                selected_route=selected_route,
                progress=progress,
            )
            outputs = (causal, private)
        else:
            if role == "qualification":
                self._read_dependency(
                    context,
                    RegimeCalibrationPackage.SCHEMA,
                    RegimeCalibrationPackage,
                    "regime.calibration",
                )
            causal = self._read_dependency(
                context,
                RegimeCausalPreparation.SCHEMA,
                RegimeCausalPreparation,
                f"regime.prepare.{root}",
            )
            private = self._read_dependency(
                context,
                RegimePrivatePreparation.SCHEMA,
                RegimePrivatePreparation,
                f"regime.prepare.{root}",
            )
            seal = self._read_dependency(
                context,
                RegimePredictionSeal.SCHEMA,
                RegimePredictionSeal,
                f"regime.seal.{root}",
            )
            assert isinstance(causal, RegimeCausalPreparation)
            assert isinstance(private, RegimePrivatePreparation)
            assert isinstance(seal, RegimePredictionSeal)
            if (
                (causal.root, causal.role, private.root, private.role, seal.root, seal.role)
                != (root, role, root, role, root, role)
                or seal.causal_preparation
                != ObjectIdentity.from_record(f"{root}.causal-preparation", causal)
                or seal.context_sha256
                != tuple(row.input_sha256 for row in causal_contexts(causal))
                or (role == "fit") != (seal.seal_kind == "FIT_ACQUISITION_ONLY")
            ):
                raise ValueError("assay seal or preparation changed its root/role/cutoff")
            outputs = (
                acquire_assays(
                    source.batch,
                    causal,
                    private,
                    ObjectIdentity.from_record(f"{root}.prediction-seal", seal),
                    progress=progress,
                ),
            )
        records = {record.SCHEMA: record for record in outputs}
        return RunnerResult(
            tuple(
                TaskOutputPayload(port.output_id, records[port.payload_schema].canonical_bytes())
                for port in context.output_ports
            ),
            (ReceiptCheck("reactor-regime-sealed-source-closure", True, ()),),
        )


class RegimeNativeProvider(CampaignRuntimeProvider):
    def __init__(
        self,
        registry: CapabilityRegistry,
        config: ReactorRegimeNativeConfig,
        source: ExternalInputPayload,
        custody: DependencyCustodyReader,
        limits: EmpiricalResourceGuard,
        prior_artifacts: tuple[RegimePriorArtifact, ...] = (),
        control_custody: RegimeControlCustodyPort | None = None,
        law_reader: CandidatePayloadReader | None = None,
        retained: RegimeRetainedInputsPort = NoRegimeRetainedInputs(),
    ) -> None:
        if (
            registry.resolve(CAPABILITY.capability_key, CAPABILITY.capability_version) != CAPABILITY
            or source.payload_schema != EmpiricalStudySource.SCHEMA
            or source.outcome_access is not OutcomeAccess.OUTCOME_BLIND
        ):
            raise ValueError("native provider registry/source visibility differs")
        self.registry_sha256, self.capability_count = registry.fingerprint(), 1
        self.config, self.source = config, source
        self.prior_artifacts = prior_artifacts
        self.retained = retained
        self.runner = RegimeNativeRunner(
            config, custody, limits, prior_artifacts, control_custody, law_reader, retained
        )

    def runners(
        self, registry: CapabilityRegistry, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[TaskRunner, ...]:
        if registry.fingerprint() != self.registry_sha256 or source_records:
            raise ValueError("native provider registry/source differs")
        return (self.runner,)

    def external_inputs(
        self, plan: ProtocolExecutionPlan, source_records: tuple[CanonicalRecord, ...] = ()
    ) -> tuple[ExternalInputPayload, ...]:
        if plan.registry_sha256 != self.registry_sha256 or source_records:
            raise ValueError("native provider plan/source differs")
        specs = {
            spec.logical_artifact_id: spec
            for task in plan.tasks
            if task.capability.capability_key == CAPABILITY.capability_key
            for spec in task.external_inputs
        }
        configs = [spec for spec in specs.values() if spec.expected_payload_schema == self.config.SCHEMA]
        sources = [spec for spec in specs.values() if spec.expected_payload_schema == self.source.payload_schema]
        retained = retained_input_map(self.retained)
        retained_specs = {key: value for key, value in specs.items() if key in retained}
        if len(specs) != 2 + len(self.prior_artifacts) + len(retained_specs) or len(configs) != 1 or len(sources) != 1:
            raise ValueError("native external input census differs")
        for key, spec in retained_specs.items():
            row = retained[key]
            if (spec.expected_content_sha256 != row.artifact.sha256
                    or spec.expected_payload_schema != row.artifact.payload_schema
                    or spec.expected_media_type != row.artifact.media_type
                    or spec.expected_visibility_ceiling is not row.manifest.logical.visibility_ceiling
                    or spec.expected_outcome_access is not row.manifest.logical.outcome_access):
                raise ValueError("native retained C input differs from original custody")
        config, source = configs[0], sources[0]
        if (
            config.expected_content_sha256 != self.config.fingerprint()
            or source.expected_content_sha256 != self.source.logical_content_sha256
            or source.logical_artifact_id != self.source.logical_artifact_id
        ):
            raise ValueError("native external input identity differs")
        parent = ArtifactLineageParent(
            ObjectIdentity.from_record(self.config.config_id, self.config),
            VisibilityCeiling.PROSPECTIVE,
            OutcomeAccess.OUTCOME_BLIND,
        )
        config_payload = ExternalInputPayload.from_bytes(
            logical_artifact_id=config.logical_artifact_id,
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
        # The method provider owns the shared B/C prior input once for the
        # composite run. Native workers still receive and recheck those ports.
        for prior in self.prior_artifacts:
            logical = prior.manifest.logical
            matched = specs.get(logical.logical_artifact_id)
            if (
                matched is None
                or matched.expected_content_sha256 != logical.content_sha256
                or matched.expected_payload_schema != logical.payload_schema
                or matched.expected_size_bytes not in (None, len(prior.payload))
            ):
                raise ValueError("native prior-phase package input differs from receipt")
        return tuple(sorted((config_payload, self.source), key=lambda value: value.logical_artifact_id))

    def output_semantic_contracts(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> tuple[CapabilityOutputSemanticContract, ...]:
        self.runners(registry)
        return registered_output_contracts()

    def scientific_adjudication_contract(
        self, registry: CapabilityRegistry, execution_plan: ProtocolExecutionPlan | None = None
    ) -> None:
        self.runners(registry)
        return None


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
            RegimeCausalPreparation,
            RegimePrivatePreparation,
            RegimeAssayPanel,
            RegimeDNativeRoot,
        )
    )
